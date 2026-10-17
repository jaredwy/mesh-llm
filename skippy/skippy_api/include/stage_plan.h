#ifndef SKIPPY_API_STAGE_PLAN_H
#define SKIPPY_API_STAGE_PLAN_H

/**
 * @file stage_plan.h
 * @brief Typed, pointer-free description of a realized staged plan.
 *
 * A stage plan is the metadata answer to "which layers does this stage hold,
 * which activation values cross its boundaries, and which persistent state
 * does it touch, for each execution profile". It is pure data: describing and
 * validating a plan needs no llama and no backend, which is what makes it
 * testable without a model.
 *
 * The native planner produces these values; this header is the typed shape
 * they take once they enter the facade. Layer ranges are half-open:
 * `[start, end)`.
 */

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace skippy {

/// A string that means one specific thing. Tags stop ids from being swapped.
template <typename Tag> struct StrongString {
    std::string value;

    [[nodiscard]] std::string_view view() const noexcept { return value; }
    [[nodiscard]] bool empty() const noexcept { return value.empty(); }

    friend bool operator==(const StrongString&, const StrongString&) = default;
    friend auto operator<=>(const StrongString&, const StrongString&) = default;
};

struct PackageIdTag;
struct PlanIdTag;
struct ProfileIdTag;
struct GraphIdentityTag;
struct ProfileIdentityTag;
struct SliceIdentityTag;
struct SourceSnapshotIdentityTag;
struct GraphConfigurationIdTag;
struct BackendIdTag;
struct TensorIdTag;
struct ValueIdentityTag;
struct ValueBindingTag;

using PackageId = StrongString<PackageIdTag>;
using PlanId = StrongString<PlanIdTag>;
using ProfileId = StrongString<ProfileIdTag>;
using GraphIdentity = StrongString<GraphIdentityTag>;
using ProfileIdentity = StrongString<ProfileIdentityTag>;
using SliceIdentity = StrongString<SliceIdentityTag>;
using SourceSnapshotIdentity = StrongString<SourceSnapshotIdentityTag>;
using GraphConfigurationId = StrongString<GraphConfigurationIdTag>;
using BackendId = StrongString<BackendIdTag>;
using TensorId = StrongString<TensorIdTag>;
/// Semantic identity of an activation value crossing a stage boundary.
using ValueIdentity = StrongString<ValueIdentityTag>;
/// Where that value lives in the stage's tensor namespace.
using ValueBinding = StrongString<ValueBindingTag>;

/// Half-open layer range, `[start, end)`, within `[0, layer_count)`.
struct LayerRange {
    std::int32_t start = 0;
    std::int32_t end = 0;

    [[nodiscard]] bool empty() const noexcept { return start >= end; }
    [[nodiscard]] std::int32_t count() const noexcept { return end - start; }
    [[nodiscard]] bool contains(std::int32_t layer) const noexcept {
        return layer >= start && layer < end;
    }

    friend bool operator==(const LayerRange&, const LayerRange&) = default;
};

/// Which side of a profile boundary a value identity names.
enum class ValueKind : std::uint8_t {
    ActivationImport,
    ActivationExport,
    RequestInput,
};

/// Kind of persistent state a stage reads or writes.
enum class StateKind : std::uint8_t {
    KvKey,
    KvValue,
    RecurrentConvolution,
    RecurrentSsm,
};

enum class StateAccess : std::uint8_t {
    Read,
    Write,
};

enum class StateResidency : std::uint8_t {
    LayerLocal,
    PerStage,
};

/// One persistent-state effect of a profile.
struct StateEffect {
    ValueIdentity identity;
    StateKind kind = StateKind::KvKey;
    StateAccess access = StateAccess::Read;
    std::int32_t layer = 0;
    StateResidency residency = StateResidency::LayerLocal;
    std::int64_t write_ordinal = 0;
};

/// One activation value crossing a stage boundary.
struct FrontierValue {
    ValueIdentity identity;
    ValueBinding binding;
};

/// Guards and geometry for one execution profile of a stage.
struct PlanProfile {
    ProfileId id;
    GraphIdentity graph_identity;
    ProfileIdentity profile_identity;
    SliceIdentity slice_identity;
    SourceSnapshotIdentity source_snapshot;
    GraphConfigurationId graph_configuration;
    BackendId backend;

    /// The slice this profile was built for; must equal the owning stage range.
    LayerRange slice_range;

    std::uint32_t tokens = 0;
    std::uint32_t sequences = 0;
    std::uint32_t outputs = 0;
    std::uint32_t recurrent_rollback_sequences = 0;

    std::vector<FrontierValue> activation_imports;
    std::vector<FrontierValue> activation_exports;
    std::vector<FrontierValue> request_inputs;
    std::vector<StateEffect> state_effects;

    /// Identities only: the frontier contract is about semantics, not storage.
    [[nodiscard]] std::vector<ValueIdentity> import_identities() const;
    [[nodiscard]] std::vector<ValueIdentity> export_identities() const;
};

/// One realized, immutable stage.
struct StagePlan {
    PackageId package;
    PlanId id;
    std::int32_t layer_count = 0;
    LayerRange range;

    /// Opaque native execution contract, carried to the runtime unchanged.
    std::string execution_contract;

    std::vector<PlanProfile> profiles;
    std::vector<TensorId> resident_identities;

    [[nodiscard]] bool terminal() const noexcept { return range.end == layer_count; }
};

/// What a caller asks the planner to guard for one execution profile.
struct ProfileSpec {
    ProfileId id;
    std::uint32_t tokens = 0;
    std::uint32_t sequences = 1;
    std::uint32_t outputs = 0;
    std::uint32_t recurrent_rollback_sequences = 0;
};

/// Why a plan, or a chain of plans, is not usable.
enum class PlanError : std::uint8_t {
    EmptyChain,
    InvalidRange,
    IncompletePartition,
    LayerCountMismatch,
    PackageMismatch,
    ProfileCountMismatch,
    ProfileIdentityMismatch,
    ActivationFrontierMismatch,
    /// The backend refused the request; `PlanFailure::message` says why.
    Rejected,
    /// The backend failed after accepting the request.
    BackendFailed,
};

/// A plan error plus the backend's explanation, when there is one.
struct PlanFailure {
    PlanError code = PlanError::BackendFailed;
    std::string message;
};

/// What a plan's state effects say about the stage, without naming a family.
///
/// This is the facade's reading of the per-model graph semantics: a stage that
/// touches recurrent SSM state is recurrent whatever architecture produced it.
struct PlanSemantics {
    /// Distinct state kinds this stage touches, in ascending enum order.
    std::vector<StateKind> state_kinds;
    /// True when any layer holds recurrent (convolution or SSM) state.
    bool recurrent = false;
    /// Lowest and highest layer with a state effect; -1 when there are none.
    std::int32_t first_layer = -1;
    std::int32_t last_layer = -1;
    /// State effects that are written, not just read.
    std::size_t written = 0;
    /// Values crossing a stage boundary in either direction.
    std::size_t frontier_values = 0;
};

/// Summarizes one profile's semantics; the plan must have exactly one profile
/// for this to be unambiguous.
[[nodiscard]] std::expected<PlanSemantics, PlanError> summarize(const StagePlan& plan,
                                                                const ProfileId& profile);

/// State effects of one profile at one layer, in plan order.
[[nodiscard]] std::vector<StateEffect> state_effects_at(const PlanProfile& profile,
                                                        std::int32_t layer);

/// Exact package tensor binding the planner needs.
struct PackageTensor {
    TensorId id;
    std::string native_name;
    std::int32_t ggml_type = 0;
    std::uint32_t dimension_count = 0;
    std::array<std::int64_t, 4> dimensions{};
    std::uint32_t split_no = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t stored_length = 0;
};

/// Everything a planner needs to build a metadata-only model.
struct StagePlannerInputs {
    PackageId package;
    std::vector<std::string> shard_paths;
    std::vector<PackageTensor> tensors;
    std::vector<ProfileSpec> profiles;
    GraphConfigurationId graph_configuration;
    BackendId backend;
};

/// Rejects inputs no planner could use, before any backend work starts.
[[nodiscard]] std::expected<void, PlanFailure> validate_inputs(const StagePlannerInputs& inputs);

/// Checks one plan on its own: range is non-empty and inside the model.
[[nodiscard]] std::expected<void, PlanError> validate(const StagePlan& plan);

/// Checks that `stages` form one complete, compatible chain.
///
/// Rules, in the order they are reported: a non-empty chain; every plan
/// individually valid; ranges partition `[0, layer_count)` in order; adjacent
/// stages agree on model, profile count and per-profile guard; and each
/// stage's exports are exactly the next stage's imports.
[[nodiscard]] std::expected<void, PlanError> validate_chain(std::span<const StagePlan> stages);

} // namespace skippy

#endif // SKIPPY_API_STAGE_PLAN_H
