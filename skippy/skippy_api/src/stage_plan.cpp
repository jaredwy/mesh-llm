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

std::expected<void, PlanError> validate(const StagePlan& plan) {
    if (plan.layer_count <= 0 || plan.range.start < 0 || plan.range.start >= plan.range.end ||
        plan.range.end > plan.layer_count) {
        return std::unexpected(PlanError::InvalidRange);
    }
    if (plan.package.empty()) return std::unexpected(PlanError::PackageMismatch);
    if (plan.profiles.empty()) return std::unexpected(PlanError::ProfileCountMismatch);
    const bool slice_matches = std::ranges::all_of(plan.profiles, [&](const PlanProfile& profile) {
        return profile.slice_range == plan.range;
    });
    if (!slice_matches) return std::unexpected(PlanError::ProfileIdentityMismatch);
    return {};
}

namespace {

bool ranges_join(std::span<const StagePlan> stages, std::size_t index) {
    const StagePlan& stage = stages[index];
    if (index == 0) return stage.range.start == 0;
    return stage.range.start == stages[index - 1].range.end;
}

bool ends_at_model_end(const StagePlan& stage) { return stage.range.end == stage.layer_count; }

std::expected<void, PlanError> validate_handoff(const StagePlan& previous, const StagePlan& stage) {
    if (stage.profiles.size() != previous.profiles.size()) {
        return std::unexpected(PlanError::ProfileCountMismatch);
    }
    for (const std::size_t index : std::views::iota(std::size_t{0}, stage.profiles.size())) {
        const PlanProfile& profile = stage.profiles[index];
        const PlanProfile& handoff = previous.profiles[index];
        if (!guards_match(handoff, profile)) {
            return std::unexpected(PlanError::ProfileIdentityMismatch);
        }
        // What the earlier stage exports must be what this one imports, in
        // order: the frontier is a list of values, not a set.
        if (handoff.export_identities() != profile.import_identities()) {
            return std::unexpected(PlanError::ActivationFrontierMismatch);
        }
    }
    return {};
}

std::expected<void, PlanError> validate_link(std::span<const StagePlan> stages, std::size_t index,
                                             const PackageId& package) {
    const StagePlan& stage = stages[index];
    if (stage.package != package) return std::unexpected(PlanError::PackageMismatch);
    if (stage.layer_count != stages.front().layer_count) {
        return std::unexpected(PlanError::LayerCountMismatch);
    }
    if (!ranges_join(stages, index)) return std::unexpected(PlanError::IncompletePartition);
    const bool last = index + 1 == stages.size();
    if (last && !ends_at_model_end(stage)) {
        return std::unexpected(PlanError::IncompletePartition);
    }
    if (index == 0) return {};
    return validate_handoff(stages[index - 1], stage);
}

} // namespace

std::expected<void, PlanError> validate_chain(std::span<const StagePlan> stages) {
    if (stages.empty()) return std::unexpected(PlanError::EmptyChain);

    for (const StagePlan& stage : stages) {
        if (auto valid = validate(stage); !valid) return valid;
    }

    const PackageId package = stages.front().package;
    for (const std::size_t index : std::views::iota(std::size_t{0}, stages.size())) {
        if (auto valid = validate_link(stages, index, package); !valid) return valid;
    }
    return {};
}

} // namespace skippy
