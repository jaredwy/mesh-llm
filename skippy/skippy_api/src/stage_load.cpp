#include "stage_load.h"

#include <algorithm>

namespace skippy {

std::expected<StageLoadConfig, PlanError> stage_load_config(const StagePlan& plan,
                                                            const ProfileId& profile) {
    const auto found = std::ranges::find_if(
        plan.profiles, [&](const PlanProfile& entry) { return entry.id == profile; });
    if (found == plan.profiles.end()) return std::unexpected(PlanError::ProfileIdentityMismatch);
    if (plan.resident_identities.empty()) return std::unexpected(PlanError::ProfileCountMismatch);
    if (plan.execution_contract.empty()) {
        return std::unexpected(PlanError::ActivationFrontierMismatch);
    }

    StageLoadConfig config;
    config.layers = plan.range;
    config.execution_contract = plan.execution_contract;

    config.resident_identities.reserve(plan.resident_identities.size());
    std::ranges::transform(plan.resident_identities, std::back_inserter(config.resident_identities),
                           [](const TensorId& tensor) { return tensor.view(); });

    // Identities and bindings are paired: the loader indexes them together.
    config.activation_import_identities.reserve(found->activation_imports.size());
    config.activation_import_bindings.reserve(found->activation_imports.size());
    for (const FrontierValue& value : found->activation_imports) {
        config.activation_import_identities.push_back(value.identity.view());
        config.activation_import_bindings.push_back(value.binding.view());
    }
    config.activation_export_identities.reserve(found->activation_exports.size());
    config.activation_export_bindings.reserve(found->activation_exports.size());
    for (const FrontierValue& value : found->activation_exports) {
        config.activation_export_identities.push_back(value.identity.view());
        config.activation_export_bindings.push_back(value.binding.view());
    }
    return config;
}

} // namespace skippy
