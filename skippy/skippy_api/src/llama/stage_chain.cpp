#include "llama/stage_chain.h"

#include "llama/stage_planner.h"

#include <cstddef>
#include <utility>

namespace skippy::llama {
namespace {} // namespace

std::expected<StageChain, StagedFailure> StageChain::open(const std::string& path,
                                                          std::span<const LayerRange> ranges) {
    auto configs = plan_model_stages(path, ranges);
    if (!configs) {
        return std::unexpected(StagedFailure{.status = 0, .message = configs.error().message});
    }

    StageChain chain;
    chain.stages_.reserve(configs->size());
    for (const StageLoadConfig& config : *configs) {
        auto model = StagedModel::open(path, own(config));
        if (!model) return std::unexpected(model.error());
        auto session = model->create_session();
        if (!session) return std::unexpected(session.error());
        chain.stages_.push_back(Stage{.model = std::move(*model), .session = std::move(*session)});
    }
    return chain;
}

std::expected<ActivationFrame, StagedFailure>
StageChain::run_stage(std::size_t index, std::span<const llama_token> tokens,
                      const ActivationFrame* input) {
    Stage& stage = stages_[index];
    skippy_activation_desc input_desc{};
    if (input != nullptr) input_desc = input->desc;

    ActivationFrame produced;
    produced.payload.resize(std::size_t{65536});
    std::size_t needed = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        skippy_error* error = nullptr;
        const auto status = skippy_prefill_chunk_frame(
            stage.session.native(), tokens.data(), tokens.size(),
            input == nullptr ? nullptr : &input_desc,
            input == nullptr ? nullptr : input->payload.data(), &produced.desc,
            produced.payload.data(), produced.payload.size(), &needed, &error);
        if (status == SKIPPY_STATUS_OK) {
            produced.payload.resize(needed);
            return produced;
        }
        StagedFailure failure{.status = static_cast<int>(status), .message = {}};
        if (error != nullptr) {
            if (error->message != nullptr) failure.message = error->message;
            skippy_error_free(error);
        }
        if (status != SKIPPY_STATUS_BUFFER_TOO_SMALL || needed == 0) {
            return std::unexpected(failure);
        }
        produced.payload.resize(needed);
    }
    return std::unexpected(
        StagedFailure{.status = 0, .message = "activation frame sizing did not converge"});
}

} // namespace skippy::llama
