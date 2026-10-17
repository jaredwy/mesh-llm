#include "llama/stage_planner.h"

#include <algorithm>
#include <ranges>
#include <utility>

namespace skippy::llama {
namespace {

PlanError code_for(enum skippy_status status) noexcept {
    switch (status) {
    case SKIPPY_STATUS_OK:
    case SKIPPY_STATUS_IO_ERROR:
    case SKIPPY_STATUS_MODEL_ERROR:
    case SKIPPY_STATUS_RUNTIME_ERROR:
    case SKIPPY_STATUS_ERROR: return PlanError::BackendFailed;
    case SKIPPY_STATUS_INVALID_ARGUMENT:
    case SKIPPY_STATUS_UNSUPPORTED:
    case SKIPPY_STATUS_BUFFER_TOO_SMALL: return PlanError::Rejected;
    }
    return PlanError::BackendFailed;
}

/// Owns the native error until it is consumed, so no path leaks it.
struct ErrorSlot {
    skippy_error* raw = nullptr;

    ErrorSlot() = default;
    ErrorSlot(const ErrorSlot&) = delete;
    ErrorSlot& operator=(const ErrorSlot&) = delete;

    ~ErrorSlot() {
        if (raw != nullptr) skippy_error_free(raw);
    }
};

/// Turns a native status into our value type, consuming the message.
std::expected<void, PlanFailure> check(enum skippy_status status, ErrorSlot& error) {
    if (status == SKIPPY_STATUS_OK) return {};
    PlanFailure failure{.code = code_for(status), .message = {}};
    if (error.raw != nullptr) {
        if (error.raw->message != nullptr) failure.message = error.raw->message;
        skippy_error_free(error.raw);
        error.raw = nullptr;
    }
    return std::unexpected(std::move(failure));
}

std::expected<std::string, PlanFailure> read_string(const skippy_stage_plan& plan,
                                                    skippy_stage_plan_string_ref_v1 reference) {
    ErrorSlot error;
    const char* data = nullptr;
    std::size_t length = 0;
    const auto status = skippy_stage_plan_string_v1(&plan, reference, &data, &length, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    // Plan storage is not NUL-terminated, so the length is the only bound.
    return std::string{data, length};
}

std::expected<skippy_stage_plan_desc_v1, PlanFailure> describe(const skippy_stage_plan& plan) {
    skippy_stage_plan_desc_v1 desc{};
    desc.abi_version = SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    ErrorSlot error;
    const auto status = skippy_stage_plan_describe_v1(&plan, &desc, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    return desc;
}

std::vector<skippy_stage_planner_tensor_v1> native_tensors(const StagePlannerInputs& inputs) {
    std::vector<skippy_stage_planner_tensor_v1> tensors;
    tensors.reserve(inputs.tensors.size());
    for (const PackageTensor& tensor : inputs.tensors) {
        skippy_stage_planner_tensor_v1 entry{};
        entry.abi_version = SKIPPY_STAGE_PLANNER_TENSOR_V1_ABI_VERSION;
        entry.struct_size = sizeof(entry);
        entry.tensor_id = tensor.id.value.c_str();
        entry.native_name = tensor.native_name.c_str();
        entry.ggml_type = tensor.ggml_type;
        entry.dimension_count = tensor.dimension_count;
        std::ranges::copy(tensor.dimensions, std::begin(entry.dimensions));
        entry.split_no = tensor.split_no;
        entry.data_offset = tensor.data_offset;
        entry.stored_length = tensor.stored_length;
        tensors.push_back(entry);
    }
    return tensors;
}

std::vector<skippy_stage_planner_profile_v1> native_profiles(const StagePlannerInputs& inputs) {
    std::vector<skippy_stage_planner_profile_v1> profiles;
    profiles.reserve(inputs.profiles.size());
    for (const ProfileSpec& profile : inputs.profiles) {
        skippy_stage_planner_profile_v1 entry{};
        entry.abi_version = SKIPPY_STAGE_PLANNER_PROFILE_V1_ABI_VERSION;
        entry.struct_size = sizeof(entry);
        entry.profile_id = profile.id.value.c_str();
        entry.n_tokens = profile.tokens;
        entry.n_sequences = profile.sequences;
        entry.n_outputs = profile.outputs;
        entry.n_recurrent_rollback_sequences = profile.recurrent_rollback_sequences;
        profiles.push_back(entry);
    }
    return profiles;
}

} // namespace

void StagePlanDeleter::operator()(skippy_stage_plan* plan) const noexcept {
    if (plan != nullptr) skippy_stage_plan_free(plan);
}

void StagePlannerDeleter::operator()(skippy_stage_planner* planner) const noexcept {
    if (planner != nullptr) skippy_stage_planner_free(planner);
}

std::expected<StagePlanner, PlanFailure> StagePlanner::create(const StagePlannerInputs& inputs) {
    if (auto valid = validate_inputs(inputs); !valid.has_value()) {
        return std::unexpected(valid.error());
    }

    std::vector<std::string> shard_storage = inputs.shard_paths;
    std::vector<const char*> shards;
    shards.reserve(shard_storage.size());
    for (const std::string& path : shard_storage)
        shards.push_back(path.c_str());

    const std::vector<skippy_stage_planner_tensor_v1> tensors = native_tensors(inputs);
    const std::vector<skippy_stage_planner_profile_v1> profiles = native_profiles(inputs);

    skippy_stage_planner_config_v1 config{};
    config.abi_version = SKIPPY_STAGE_PLANNER_CONFIG_V1_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.package_id = inputs.package.value.c_str();
    config.shard_paths = shards.data();
    config.shard_count = shards.size();
    config.tensors = tensors.empty() ? nullptr : tensors.data();
    config.tensor_count = tensors.size();
    config.profiles = profiles.empty() ? nullptr : profiles.data();
    config.profile_count = profiles.size();
    config.graph_configuration_id = inputs.graph_configuration.value.c_str();
    config.backend_id = inputs.backend.value.c_str();

    ErrorSlot error;
    skippy_stage_planner* planner = nullptr;
    const auto status = skippy_stage_planner_create_v1(&config, &planner, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    if (planner == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::BackendFailed,
            .message = "planner was accepted but not returned",
        });
    }
    StagePlanner wrapper;
    wrapper.planner_.reset(planner);
    return wrapper;
}

std::expected<StagePlanHandle, PlanFailure> StagePlanner::realize(LayerRange range) const {
    if (planner_ == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = "planner is not loaded",
        });
    }
    ErrorSlot error;
    skippy_stage_plan* plan = nullptr;
    const auto status =
        skippy_stage_planner_realize_v1(planner_.get(), range.start, range.end, &plan, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    if (plan == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::BackendFailed,
            .message = "realize was accepted but returned no plan",
        });
    }
    return StagePlanHandle{std::unique_ptr<skippy_stage_plan, StagePlanDeleter>{plan}};
}

std::expected<LayerRange, PlanFailure> StagePlanHandle::range() const {
    if (plan_ == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = "plan is not loaded",
        });
    }
    const auto desc = describe(*plan_);
    if (!desc.has_value()) return std::unexpected(desc.error());
    return LayerRange{.start = desc->layer_start, .end = desc->layer_end};
}

std::expected<StagePlan, PlanFailure> StagePlanHandle::materialize() const {
    if (plan_ == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = "plan is not loaded",
        });
    }
    const auto desc = describe(*plan_);
    if (!desc.has_value()) return std::unexpected(desc.error());

    StagePlan plan;
    plan.layer_count = desc->layer_count;
    plan.range = LayerRange{.start = desc->layer_start, .end = desc->layer_end};

    auto package = read_string(*plan_, desc->package_id);
    if (!package.has_value()) return std::unexpected(package.error());
    plan.package = PackageId{std::move(*package)};

    auto id = read_string(*plan_, desc->plan_id);
    if (!id.has_value()) return std::unexpected(id.error());
    plan.id = PlanId{std::move(*id)};

    auto contract = read_string(*plan_, desc->execution_contract);
    if (!contract.has_value()) return std::unexpected(contract.error());
    plan.execution_contract = std::move(*contract);

    const auto resident_count = static_cast<std::size_t>(desc->resident_tensor_count);
    plan.resident_tensors.reserve(resident_count);
    for (const std::size_t index : std::views::iota(std::size_t{0}, resident_count)) {
        skippy_stage_plan_value_desc_v1 value{};
        value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
        value.struct_size = sizeof(value);
        ErrorSlot error;
        const auto status =
            skippy_stage_plan_resident_tensor_at_v1(plan_.get(), index, &value, &error.raw);
        if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
        auto identity = read_string(*plan_, value.identity);
        if (!identity.has_value()) return std::unexpected(identity.error());
        plan.resident_tensors.push_back(TensorId{std::move(*identity)});
    }

    const auto profile_count = static_cast<std::size_t>(desc->profile_count);
    plan.profiles.reserve(profile_count);
    for (const std::size_t index : std::views::iota(std::size_t{0}, profile_count)) {
        skippy_stage_plan_profile_desc_v1 profile{};
        profile.abi_version = SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION;
        profile.struct_size = sizeof(profile);
        ErrorSlot error;
        const auto status =
            skippy_stage_plan_profile_at_v1(plan_.get(), index, &profile, &error.raw);
        if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());

        PlanProfile entry;
        entry.tokens = profile.n_tokens;
        entry.sequences = profile.n_sequences;
        entry.outputs = profile.n_outputs;
        entry.recurrent_rollback_sequences = profile.n_recurrent_rollback_sequences;
        entry.slice_range = LayerRange{.start = desc->layer_start, .end = desc->layer_end};

        const auto assign = [&](skippy_stage_plan_string_ref_v1 reference, auto&& target) {
            auto text = read_string(*plan_, reference);
            if (!text.has_value()) return false;
            target.value = std::move(*text);
            return true;
        };
        if (!assign(profile.profile_id, entry.id) ||
            !assign(profile.graph_identity, entry.graph_identity) ||
            !assign(profile.profile_identity, entry.profile_identity) ||
            !assign(profile.slice_identity, entry.slice_identity) ||
            !assign(profile.source_snapshot_identity, entry.source_snapshot) ||
            !assign(profile.graph_configuration_id, entry.graph_configuration) ||
            !assign(profile.backend_id, entry.backend)) {
            return std::unexpected(PlanFailure{
                .code = PlanError::BackendFailed,
                .message = "plan profile identity is unreadable",
            });
        }
        plan.profiles.push_back(std::move(entry));
    }

    if (auto valid = validate(plan); !valid.has_value()) {
        return std::unexpected(PlanFailure{
            .code = valid.error(),
            .message = "materialized plan is invalid",
        });
    }
    return plan;
}

std::expected<void, PlanFailure> validate_chain(std::span<const StagePlanHandle> plans) {
    std::vector<const skippy_stage_plan*> native;
    native.reserve(plans.size());
    for (const StagePlanHandle& plan : plans) {
        if (!plan.valid()) {
            return std::unexpected(PlanFailure{
                .code = PlanError::Rejected,
                .message = "chain contains an empty plan",
            });
        }
        native.push_back(plan.native());
    }
    ErrorSlot error;
    const auto status =
        skippy_stage_plan_validate_chain_v1(native.data(), native.size(), &error.raw);
    return check(status, error);
}

} // namespace skippy::llama
