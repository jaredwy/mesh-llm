#include "stage_load.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using skippy::FrontierValue;
using skippy::LayerRange;
using skippy::PackageId;
using skippy::PlanError;
using skippy::PlanProfile;
using skippy::ProfileId;
using skippy::StagePlan;
using skippy::TensorId;
using skippy::ValueBinding;
using skippy::ValueIdentity;

FrontierValue value(std::string identity, std::string binding) {
    return FrontierValue{.identity = ValueIdentity{std::move(identity)},
                         .binding = ValueBinding{std::move(binding)}};
}

StagePlan plan(std::string contract = "{\"version\":1}") {
    PlanProfile profile{
        .id = ProfileId{"decode"},
        .graph_identity = {"graph"},
        .profile_identity = {"profile"},
        .slice_identity = {"slice"},
        .source_snapshot = {"source"},
        .graph_configuration = {"graph-config"},
        .backend = {"metal"},
        .slice_range = {0, 6},
        .activation_imports = {value("in.a", "bind.a")},
        .activation_exports = {value("out.b", "bind.b"), value("out.c", "bind.c")},
    };
    return StagePlan{
        .package = PackageId{"sha256:package"},
        .id = {"plan"},
        .layer_count = 6,
        .range = {0, 6},
        .execution_contract = std::move(contract),
        .profiles = {std::move(profile)},
        .resident_tensors = {TensorId{"t.0"}, TensorId{"t.1"}},
    };
}

} // namespace

TEST(StageLoadConfig, CarriesTheClosureAndContract) {
    const StagePlan value = plan();
    const auto config = skippy::stage_load_config(value, ProfileId{"decode"});
    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(config->staged());
    EXPECT_EQ(config->layers, (LayerRange{0, 6}));
    EXPECT_EQ(config->execution_contract, "{\"version\":1}");
    ASSERT_EQ(config->resident_tensors.size(), 2U);
    EXPECT_EQ(config->resident_tensors[0], "t.0");
    EXPECT_EQ(config->resident_tensors[1], "t.1");
}

TEST(StageLoadConfig, PairsFrontierIdentitiesWithTheirBindings) {
    const StagePlan value = plan();
    const auto config = skippy::stage_load_config(value, ProfileId{"decode"});
    ASSERT_TRUE(config.has_value());
    ASSERT_EQ(config->activation_import_identities.size(), 1U);
    EXPECT_EQ(config->activation_import_identities[0], "in.a");
    EXPECT_EQ(config->activation_import_bindings[0], "bind.a");
    ASSERT_EQ(config->activation_export_identities.size(), 2U);
    EXPECT_EQ(config->activation_export_identities[1], "out.c");
    EXPECT_EQ(config->activation_export_bindings[1], "bind.c");
}

TEST(StageLoadConfig, AnUnknownProfileIsRejected) {
    const StagePlan value = plan();
    EXPECT_EQ(skippy::stage_load_config(value, ProfileId{"absent"}).error(),
              PlanError::ProfileIdentityMismatch);
}

TEST(StageLoadConfig, AStageWithoutTensorsCannotBeLoaded) {
    StagePlan value = plan();
    value.resident_tensors.clear();
    EXPECT_EQ(skippy::stage_load_config(value, ProfileId{"decode"}).error(),
              PlanError::ProfileCountMismatch);
}

TEST(StageLoadConfig, AStageWithoutAContractCannotBeLoaded) {
    StagePlan value = plan("");
    EXPECT_EQ(skippy::stage_load_config(value, ProfileId{"decode"}).error(),
              PlanError::ActivationFrontierMismatch);
}
