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
using skippy::StagePlan;
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

TEST(StagePlan, TerminalIsAboutTheLastLayerNotTheLastPlan) {
    EXPECT_TRUE(plan({6, 12}, 12, {profile({6, 12}, {}, {})}).terminal());
    EXPECT_FALSE(plan({0, 6}, 12, {profile({0, 6}, {}, {})}).terminal());
}
