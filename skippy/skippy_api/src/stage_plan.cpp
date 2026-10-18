#include "stage_plan.h"

#include <algorithm>
#include <iterator>
#include <ranges>

namespace skippy {
namespace {

std::vector<ValueIdentity> identities_of(const std::vector<FrontierValue>& values) {
    std::vector<ValueIdentity> identities;
    identities.reserve(values.size());
    std::ranges::transform(values, std::back_inserter(identities), &FrontierValue::identity);
    return identities;
}

// Guards that must be identical for adjacent stages to hand off a frontier.
bool guards_match(const PlanProfile& left, const PlanProfile& right) {
    return left.id == right.id && left.graph_identity == right.graph_identity &&
           left.profile_identity == right.profile_identity;
}

} // namespace

std::vector<ValueIdentity> PlanProfile::import_identities() const {
    return identities_of(activation_imports);
}

std::vector<ValueIdentity> PlanProfile::export_identities() const {
    return identities_of(activation_exports);
}

std::optional<PlanFailure> validate_inputs(const StagePlannerInputs& inputs) {
    if (inputs.package.empty()) {
        return PlanFailure{
            .code = PlanError::PackageMismatch,
            .message = "package id is empty",
        };
    }
    if (inputs.shard_paths.empty()) {
        return PlanFailure{
            .code = PlanError::Rejected,
            .message = "a planner needs at least one shard",
        };
    }
    if (inputs.profiles.empty()) {
        return PlanFailure{
            .code = PlanError::ProfileCountMismatch,
            .message = "a planner needs at least one profile",
        };
    }
    return std::nullopt;
}

std::expected<PlanSemantics, PlanError> summarize(const StagePlan& plan, const ProfileId& profile) {
    const auto found = std::ranges::find_if(
        plan.profiles, [&](const PlanProfile& entry) { return entry.id == profile; });
    if (found == plan.profiles.end()) return std::unexpected(PlanError::ProfileIdentityMismatch);

    PlanSemantics semantics;
    for (const StateEffect& effect : found->state_effects) {
        if (std::ranges::find(semantics.state_kinds, effect.kind) == semantics.state_kinds.end()) {
            semantics.state_kinds.push_back(effect.kind);
        }
        if (effect.kind == StateKind::RecurrentConvolution ||
            effect.kind == StateKind::RecurrentSsm) {
            semantics.recurrent = true;
        }
        if (effect.access == StateAccess::Write) ++semantics.written;
        semantics.first_layer = semantics.first_layer < 0
                                    ? effect.layer
                                    : std::min(semantics.first_layer, effect.layer);
        semantics.last_layer = std::max(semantics.last_layer, effect.layer);
    }
    std::ranges::sort(semantics.state_kinds);
    semantics.frontier_values = found->activation_imports.size() + found->activation_exports.size();
    return semantics;
}

std::vector<StateEffect> state_effects_at(const PlanProfile& profile, std::int32_t layer) {
    std::vector<StateEffect> effects;
    std::ranges::copy_if(profile.state_effects, std::back_inserter(effects),
                         [layer](const StateEffect& effect) { return effect.layer == layer; });
    return effects;
}

std::optional<PlanError> validate(const StagePlan& plan) {
    if (plan.layer_count <= 0 || plan.range.start < 0 || plan.range.start >= plan.range.end ||
        plan.range.end > plan.layer_count) {
        return PlanError::InvalidRange;
    }
    if (plan.package.empty()) return PlanError::PackageMismatch;
    if (plan.profiles.empty()) return PlanError::ProfileCountMismatch;
    const bool slice_matches = std::ranges::all_of(plan.profiles, [&](const PlanProfile& profile) {
        return profile.slice_range == plan.range;
    });
    if (!slice_matches) return PlanError::ProfileIdentityMismatch;
    return std::nullopt;
}

namespace {

bool ranges_join(std::span<const StagePlan> stages, std::size_t index) {
    const StagePlan& stage = stages[index];
    if (index == 0) return stage.range.start == 0;
    return stage.range.start == stages[index - 1].range.end;
}

bool ends_at_model_end(const StagePlan& stage) { return stage.range.end == stage.layer_count; }

std::optional<PlanError> validate_handoff(const StagePlan& previous, const StagePlan& stage) {
    if (stage.profiles.size() != previous.profiles.size()) {
        return PlanError::ProfileCountMismatch;
    }
    for (const std::size_t index : std::views::iota(std::size_t{0}, stage.profiles.size())) {
        const PlanProfile& profile = stage.profiles[index];
        const PlanProfile& handoff = previous.profiles[index];
        if (!guards_match(handoff, profile)) {
            return PlanError::ProfileIdentityMismatch;
        }
        // What the earlier stage exports must be what this one imports, in
        // order: the frontier is a list of values, not a set.
        if (handoff.export_identities() != profile.import_identities()) {
            return PlanError::ActivationFrontierMismatch;
        }
    }
    return std::nullopt;
}

std::optional<PlanError> validate_link(std::span<const StagePlan> stages, std::size_t index,
                                       const PackageId& package) {
    const StagePlan& stage = stages[index];
    if (stage.package != package) return PlanError::PackageMismatch;
    if (stage.layer_count != stages.front().layer_count) {
        return PlanError::LayerCountMismatch;
    }
    if (!ranges_join(stages, index)) return PlanError::IncompletePartition;
    const bool last = index + 1 == stages.size();
    if (last && !ends_at_model_end(stage)) {
        return PlanError::IncompletePartition;
    }
    if (index == 0) return std::nullopt;
    return validate_handoff(stages[index - 1], stage);
}

} // namespace

std::string_view describe(PlanError error) noexcept {
    switch (error) {
    case PlanError::EmptyChain: return "empty chain";
    case PlanError::InvalidRange: return "invalid range";
    case PlanError::IncompletePartition: return "incomplete partition";
    case PlanError::LayerCountMismatch: return "layer count mismatch";
    case PlanError::PackageMismatch: return "package mismatch";
    case PlanError::ProfileCountMismatch: return "profile count mismatch";
    case PlanError::ProfileIdentityMismatch: return "profile identity mismatch";
    case PlanError::ActivationFrontierMismatch: return "activation frontier mismatch";
    case PlanError::Rejected: return "rejected";
    case PlanError::BackendFailed: return "backend failed";
    }
    return "unknown";
}

std::optional<PlanError> validate_chain(std::span<const StagePlan> stages) {
    if (stages.empty()) return PlanError::EmptyChain;

    for (const StagePlan& stage : stages) {
        if (auto failure = validate(stage); failure.has_value()) return failure;
    }

    const PackageId package = stages.front().package;
    for (const std::size_t index : std::views::iota(std::size_t{0}, stages.size())) {
        if (auto failure = validate_link(stages, index, package); failure.has_value())
            return failure;
    }
    return std::nullopt;
}

} // namespace skippy
