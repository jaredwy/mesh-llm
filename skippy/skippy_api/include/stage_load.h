#ifndef SKIPPY_API_STAGE_LOAD_H
#define SKIPPY_API_STAGE_LOAD_H

/**
 * @file stage_load.h
 * @brief Everything the native loader needs to open one stage, taken from a plan.
 *
 * Patch 0003 lets llama load an explicit stage, but only when the caller
 * supplies the stage's exact resident tensor closure, its execution contract,
 * and the identities/bindings crossing both boundaries. A realized `StagePlan`
 * already holds all of it, so this header is the mapping between the two.
 *
 * Views borrow from the plan; the plan must outlive the config.
 */

#include "stage_plan.h"

#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

namespace skippy {

/// One stage as the native runtime configuration wants it.
struct StageLoadConfig {
    std::int32_t stage_index = 0;
    LayerRange layers;
    /// Opaque contract the planner produced for this stage.
    std::string_view execution_contract;
    /// Exact closure; never empty for a stage.
    std::vector<std::string_view> resident_tensors;
    std::vector<std::string_view> activation_import_identities;
    std::vector<std::string_view> activation_import_bindings;
    std::vector<std::string_view> activation_export_identities;
    std::vector<std::string_view> activation_export_bindings;

    [[nodiscard]] bool staged() const noexcept {
        return !resident_tensors.empty() && !execution_contract.empty();
    }
};

/// Derives the loader configuration for `profile` in `plan`.
[[nodiscard]] std::expected<StageLoadConfig, PlanError> stage_load_config(const StagePlan& plan,
                                                                          const ProfileId& profile);

} // namespace skippy

#endif // SKIPPY_API_STAGE_LOAD_H
