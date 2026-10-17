#include "stage_plan.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace {

using skippy::FrontierValue;
using skippy::LayerRange;
using skippy::PackageId;
using skippy::PlanError;
using skippy::PlanProfile;
using skippy::ProfileId;
using skippy::ProfileSpec;
using skippy::StagePlan;
using skippy::StagePlannerInputs;
using skippy::StateAccess;
using skippy::StateEffect;
using skippy::StateKind;
using skippy::StateResidency;
using skippy::ValueIdentity;

std::vector<FrontierValue> to_values(const std::vector<ValueIdentity>& identities) {
    std::vector<FrontierValue> values;
    values.reserve(identities.size());
    for (const ValueIdentity& identity : identities) {
        values.push_back(FrontierValue{.identity = identity, .binding = {}});
    }
    return values;
}

PlanProfile profile(LayerRange range, std::vector<ValueIdentity> imports,
                    std::vector<ValueIdentity> exports, std::string id = "decode") {
    return PlanProfile{
        .id = ProfileId{std::move(id)},
        .graph_identity = {"graph"},
        .profile_identity = {"profile"},
        .slice_identity = {"slice"},
        .source_snapshot = {"source"},
        .graph_configuration = {"graph-config"},
        .backend = {"metal"},
        .slice_range = range,
        .activation_imports = to_values(imports),
        .activation_exports = to_values(exports),
    };
}

StagePlan plan(LayerRange range, std::int32_t layer_count, std::vector<PlanProfile> profiles) {
    return StagePlan{
        .package = PackageId{"package"},
        .id = {"plan"},
        .layer_count = layer_count,
        .range = range,
        .execution_contract = "{}",
        .profiles = std::move(profiles),
        .resident_tensors = {},
    };
}

StagePlan single(std::int32_t layer_count = 12) {
    return plan({0, layer_count}, layer_count, {profile({0, layer_count}, {}, {})});
}

} // namespace

TEST(StagePlan, UnitRangeIsHalfOpen) {
    const LayerRange range{4, 8};
    EXPECT_EQ(range.count(), 4);
    EXPECT_TRUE(range.contains(4));
    EXPECT_TRUE(range.contains(7));
    EXPECT_FALSE(range.contains(8));
    EXPECT_FALSE(range.contains(3));
}

TEST(StagePlan, AWholeModelPlanIsValid) { EXPECT_TRUE(skippy::validate(single()).has_value()); }

TEST(StagePlan, RejectsAnEmptyOrInvertedRange) {
    EXPECT_EQ(skippy::validate(plan({4, 4}, 12, {profile({4, 4}, {}, {})})).error(),
              PlanError::InvalidRange);
    EXPECT_EQ(skippy::validate(plan({8, 4}, 12, {profile({8, 4}, {}, {})})).error(),
              PlanError::InvalidRange);
    EXPECT_EQ(skippy::validate(plan({-1, 4}, 12, {profile({-1, 4}, {}, {})})).error(),
              PlanError::InvalidRange);
    EXPECT_EQ(skippy::validate(plan({0, 13}, 12, {profile({0, 13}, {}, {})})).error(),
              PlanError::InvalidRange);
}

TEST(StagePlan, RejectsAPlanWithoutProfiles) {
    EXPECT_EQ(skippy::validate(plan({0, 12}, 12, {})).error(), PlanError::ProfileCountMismatch);
}

TEST(StagePlan, RejectsAnEmptyPackageIdentity) {
    StagePlan value = single();
    value.package = PackageId{};
    EXPECT_EQ(skippy::validate(value).error(), PlanError::PackageMismatch);
}

TEST(StagePlan, RejectsAProfileBuiltForAnotherRange) {
    StagePlan value = plan({0, 6}, 12, {profile({6, 12}, {}, {})});
    EXPECT_EQ(skippy::validate(value).error(), PlanError::ProfileIdentityMismatch);
}

TEST(StagePlanChain, RejectsAnEmptyChain) {
    EXPECT_EQ(skippy::validate_chain({}).error(), PlanError::EmptyChain);
}

TEST(StagePlanChain, AcceptsAContiguousPartition) {
    const StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    const StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    const std::vector<StagePlan> chain{first, second};
    EXPECT_TRUE(skippy::validate_chain(chain).has_value());
}

TEST(StagePlanChain, RejectsAGapBetweenStages) {
    const StagePlan first = plan({0, 5}, 12, {profile({0, 5}, {}, {})});
    const StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {}, {})});
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::IncompletePartition);
}

TEST(StagePlanChain, RejectsAChainThatDoesNotStartAtLayerZero) {
    const StagePlan stage = plan({4, 12}, 12, {profile({4, 12}, {}, {})});
    EXPECT_EQ(skippy::validate_chain(std::vector<StagePlan>{stage}).error(),
              PlanError::IncompletePartition);
}

TEST(StagePlanChain, RejectsAChainThatDoesNotReachTheLastLayer) {
    const StagePlan stage = plan({0, 11}, 12, {profile({0, 11}, {}, {})});
    EXPECT_EQ(skippy::validate_chain(std::vector<StagePlan>{stage}).error(),
              PlanError::IncompletePartition);
}

TEST(StagePlanChain, RejectsStagesFromDifferentPackages) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    second.package = PackageId{"other"};
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::PackageMismatch);
}

TEST(StagePlanChain, RejectsStagesThatDisagreeOnTheModel) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 13, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::LayerCountMismatch);
}

TEST(StagePlanChain, RejectsStagesWithDifferentProfileCounts) {
    StagePlan first = plan({0, 6}, 12,
                           {profile({0, 6}, {}, {ValueIdentity{"h.5"}}),
                            profile({0, 6}, {}, {ValueIdentity{"h.5"}}, "aux")});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::ProfileCountMismatch);
}

TEST(StagePlanChain, RejectsStagesWithDifferentGuards) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    second.profiles.front().graph_identity = {"other-graph"};
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::ProfileIdentityMismatch);
}

TEST(StagePlanChain, RequiresExportsToBeExactlyTheNextImports) {
    const StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    const StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.4"}}, {})});
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::ActivationFrontierMismatch);
}

TEST(StagePlanChain, FrontierOrderIsPartOfTheContract) {
    const StagePlan first =
        plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.4"}, ValueIdentity{"h.5"}})});
    const StagePlan second =
        plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}, ValueIdentity{"h.4"}}, {})});
    const std::vector<StagePlan> chain{first, second};
    EXPECT_EQ(skippy::validate_chain(chain).error(), PlanError::ActivationFrontierMismatch);
}

TEST(StagePlanChain, AcceptsAThreeStageChain) {
    const StagePlan first = plan({0, 4}, 12, {profile({0, 4}, {}, {ValueIdentity{"h.3"}})});
    const StagePlan middle =
        plan({4, 8}, 12, {profile({4, 8}, {ValueIdentity{"h.3"}}, {ValueIdentity{"h.7"}})});
    const StagePlan last = plan({8, 12}, 12, {profile({8, 12}, {ValueIdentity{"h.7"}}, {})});
    const std::vector<StagePlan> chain{first, middle, last};
    EXPECT_TRUE(skippy::validate_chain(chain).has_value());
    EXPECT_TRUE(last.terminal());
}

namespace {

StateEffect effect(StateKind kind, StateAccess access, std::int32_t layer,
                   StateResidency residency = StateResidency::LayerLocal) {
    return StateEffect{.identity = ValueIdentity{"state"},
                       .kind = kind,
                       .access = access,
                       .layer = layer,
                       .residency = residency,
                       .write_ordinal = 0};
}

StagePlan with_states(std::vector<StateEffect> effects, std::size_t imports = 0,
                      std::size_t exports = 0) {
    StagePlan value = plan({0, 6}, 6, {});
    PlanProfile entry = profile({0, 6}, {}, {});
    entry.state_effects = std::move(effects);
    for (std::size_t index = 0; index < imports; ++index) {
        entry.activation_imports.push_back(
            FrontierValue{.identity = ValueIdentity{"in"}, .binding = {}});
    }
    for (std::size_t index = 0; index < exports; ++index) {
        entry.activation_exports.push_back(
            FrontierValue{.identity = ValueIdentity{"out"}, .binding = {}});
    }
    value.profiles.push_back(std::move(entry));
    return value;
}

} // namespace

TEST(StagePlanSemantics, SummarizesKvEffectsPerLayer) {
    StagePlan value = with_states({effect(StateKind::KvKey, StateAccess::Write, 0),
                                   effect(StateKind::KvValue, StateAccess::Write, 0),
                                   effect(StateKind::KvKey, StateAccess::Write, 5)});
    const auto semantics = skippy::summarize(value, ProfileId{"decode"});
    ASSERT_TRUE(semantics.has_value());
    EXPECT_EQ(semantics->state_kinds.size(), 2U);
    EXPECT_FALSE(semantics->recurrent);
    EXPECT_EQ(semantics->first_layer, 0);
    EXPECT_EQ(semantics->last_layer, 5);
    EXPECT_EQ(semantics->written, 3U);
}

TEST(StagePlanSemantics, RecurrenTStateIsRecognisedByKindNotByFamily) {
    StagePlan value = with_states({effect(StateKind::RecurrentSsm, StateAccess::Write, 2),
                                   effect(StateKind::KvKey, StateAccess::Read, 2)});
    const auto semantics = skippy::summarize(value, ProfileId{"decode"});
    ASSERT_TRUE(semantics.has_value());
    EXPECT_TRUE(semantics->recurrent);
    EXPECT_EQ(semantics->written, 1U);
}

TEST(StagePlanSemantics, CountsFrontierValuesBothWays) {
    StagePlan value = with_states({}, 2, 1);
    const auto semantics = skippy::summarize(value, ProfileId{"decode"});
    ASSERT_TRUE(semantics.has_value());
    EXPECT_EQ(semantics->frontier_values, 3U);
    EXPECT_EQ(semantics->first_layer, -1);
}

TEST(StagePlanSemantics, AnUnknownProfileIsRejected) {
    StagePlan value = with_states({});
    EXPECT_EQ(skippy::summarize(value, ProfileId{"absent"}).error(),
              PlanError::ProfileIdentityMismatch);
}

TEST(StagePlanSemantics, StateEffectsAreFilteredByLayerInPlanOrder) {
    PlanProfile entry = profile({0, 6}, {}, {});
    entry.state_effects = {effect(StateKind::KvKey, StateAccess::Write, 1),
                           effect(StateKind::KvValue, StateAccess::Write, 1),
                           effect(StateKind::KvKey, StateAccess::Write, 2)};

    const std::vector<StateEffect> layer_one = skippy::state_effects_at(entry, 1);
    ASSERT_EQ(layer_one.size(), 2U);
    EXPECT_EQ(layer_one[0].kind, StateKind::KvKey);
    EXPECT_EQ(layer_one[1].kind, StateKind::KvValue);
    EXPECT_TRUE(skippy::state_effects_at(entry, 4).empty());
}

TEST(StagePlannerInputs, RejectsAPlannerWithoutShards) {
    StagePlannerInputs value{
        .package = PackageId{"package"},
        .shard_paths = {},
        .profiles = {ProfileSpec{.id = ProfileId{"decode"}, .tokens = 512}},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    const auto valid = skippy::validate_inputs(value);
    ASSERT_FALSE(valid.has_value());
    EXPECT_EQ(valid.error().code, PlanError::Rejected);
    EXPECT_NE(valid.error().message.find("shard"), std::string::npos);
}

TEST(StagePlannerInputs, RejectsAPlannerWithoutProfiles) {
    StagePlannerInputs value{
        .package = PackageId{"package"},
        .shard_paths = {"shard.gguf"},
        .profiles = {},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    const auto valid = skippy::validate_inputs(value);
    ASSERT_FALSE(valid.has_value());
    EXPECT_EQ(valid.error().code, PlanError::ProfileCountMismatch);
}

TEST(StagePlannerInputs, RejectsAnEmptyPackageIdentity) {
    StagePlannerInputs value{
        .package = PackageId{},
        .shard_paths = {"shard.gguf"},
        .profiles = {ProfileSpec{.id = ProfileId{"decode"}, .tokens = 512}},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    EXPECT_EQ(skippy::validate_inputs(value).error().code, PlanError::PackageMismatch);
}

TEST(StagePlannerInputs, AcceptsAPlausibleRequest) {
    StagePlannerInputs value{
        .package = PackageId{"package"},
        .shard_paths = {"shard.gguf"},
        .profiles = {ProfileSpec{.id = ProfileId{"decode"}, .tokens = 512}},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    EXPECT_TRUE(skippy::validate_inputs(value).has_value());
}

TEST(StagePlan, TerminalIsAboutTheLastLayerNotTheLastPlan) {
    EXPECT_TRUE(plan({6, 12}, 12, {profile({6, 12}, {}, {})}).terminal());
    EXPECT_FALSE(plan({0, 6}, 12, {profile({0, 6}, {}, {})}).terminal());
}
