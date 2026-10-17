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
#include <string>
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
    std::vector<std::string_view> resident_identities;
    std::vector<std::string_view> activation_import_identities;
    std::vector<std::string_view> activation_import_bindings;
    std::vector<std::string_view> activation_export_identities;
    std::vector<std::string_view> activation_export_bindings;

    /// What the stage's state looks like, for callers that never materialize.
    std::size_t frontier_values = 0;
    std::size_t state_effects = 0;
    std::size_t written_state_effects = 0;
    bool recurrent_state = false;

    /// Owning form of the same data, for callers that outlive the plan.
    struct Spec {
        std::int32_t stage_index = 0;
        LayerRange layers;
        std::string execution_contract;
        std::vector<std::string> resident_identities;
        std::vector<std::string> import_identities;
        std::vector<std::string> import_bindings;
        std::vector<std::string> export_identities;
        std::vector<std::string> export_bindings;
    };

    [[nodiscard]] bool staged() const noexcept {
        return !resident_identities.empty() && !execution_contract.empty();
    }
};

/// Derives the loader configuration for `profile` in `plan`.
[[nodiscard]] std::expected<StageLoadConfig, PlanError> stage_load_config(const StagePlan& plan,
                                                                          const ProfileId& profile);

} // namespace skippy

#endif // SKIPPY_API_STAGE_LOAD_H
