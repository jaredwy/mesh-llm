#ifndef SKIPPY_API_LLAMA_STAGE_CHAIN_H
#define SKIPPY_API_LLAMA_STAGE_CHAIN_H

/// A model run as N chained stages.
///
/// Each stage owns a model and a session; the payload of one stage is the input
/// of the next. What happens to the payload crossing into the terminal stage is
/// the caller's decision, expressed as a compile-time sink: the library never
/// guesses whether that means pool it, sample it, or drop it.

#include "llama/staged.h"
#include "stage_plan.h"
#include "stage_sink.h"

#include "skippy.h"

#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace skippy::llama {

/// One activation boundary: payload plus the descriptor that shapes it.
struct ActivationFrame {
    std::vector<std::byte> payload;
    skippy_activation_desc desc{};
};

/// Owns N stages and drives them in order.
class StageChain {
public:
    StageChain() = default;
    StageChain(StageChain&&) noexcept = default;
    StageChain& operator=(StageChain&&) noexcept = default;
    StageChain(const StageChain&) = delete;
    StageChain& operator=(const StageChain&) = delete;

    /// Plans `ranges` over `path` and opens one model and session per stage.
    [[nodiscard]] static std::expected<StageChain, StagedFailure>
    open(const std::string& path, std::span<const LayerRange> ranges);

    /// Runs `tokens` through every stage, handing the terminal frame to `sink`.
    template <skippy::StageSink Sink>
    [[nodiscard]] std::expected<void, StagedFailure> prefill(std::span<const llama_token> tokens,
                                                             Sink& sink);

    [[nodiscard]] std::size_t stage_count() const noexcept { return stages_.size(); }

private:
    struct Stage {
        StagedModel model;
        StagedSession session;
    };

    [[nodiscard]] std::expected<ActivationFrame, StagedFailure>
    run_stage(std::size_t index, std::span<const llama_token> tokens, const ActivationFrame* input);

    std::vector<Stage> stages_;
};

template <skippy::StageSink Sink>
std::expected<void, StagedFailure> StageChain::prefill(std::span<const llama_token> tokens,
                                                       Sink& sink) {
    if (stages_.empty()) {
        return std::unexpected(StagedFailure{.status = 0, .message = "chain is empty"});
    }
    ActivationFrame carried;
    if (auto first = run_stage(0, tokens, nullptr); first.has_value()) {
        carried = std::move(*first);
    } else {
        return std::unexpected(first.error());
    }
    for (std::size_t index = 1; index < stages_.size(); ++index) {
        auto next = run_stage(index, tokens, &carried);
        if (!next.has_value()) return std::unexpected(next.error());
        if (next->payload.empty()) break; // terminal stage: nothing crosses further
        carried = std::move(*next);
    }
    if (auto accepted = sink.consume(skippy::FrameBytes{carried.payload}); !accepted) {
        return std::unexpected(StagedFailure{.status = 0, .message = std::move(accepted.error())});
    }
    return {};
}

} // namespace skippy::llama

#endif // SKIPPY_API_LLAMA_STAGE_CHAIN_H
