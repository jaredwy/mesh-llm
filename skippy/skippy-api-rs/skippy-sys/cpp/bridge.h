#pragma once

// The cxx-facing facade over NativeRuntime<LlamaBackend>.
//
// Ownership:
//   Bridge            owns the runtime (pimpl, so this header never sees llama.h)
//   Model / Session   skippy's own non-forgeable tokens; Rust holds them in
//                     UniquePtr and gives them back to unload/release
// Shared enums and structs below are defined by the cxx bridge (lib.rs); the
// generated header completes them. Nothing here throws: every status is a
// return value.

#include "runtime.h"

#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace skippy::ffi {

enum class RuntimeStatus : std::uint8_t;
enum class WorkloadStatus : std::uint8_t;
struct SessionConfig;
struct ModelResult;
struct SessionResult;
struct WorkloadResult;
struct EmbedResult;
struct RerankResult;
struct StagePlanInfo;
struct ChainValidation;
struct EncodeResult;

class Bridge {
public:
    Bridge();
    ~Bridge();

    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    [[nodiscard]] ModelResult load_model(rust::Str path) noexcept;

    /// Plans `stage_count` contiguous stages over the model at `path`.
    [[nodiscard]] rust::Vec<StagePlanInfo> plan_stages(rust::Str path,
                                                       std::uint32_t stage_count) noexcept;

    /// Checks that a plan chain is contiguous and hands over cleanly.
    [[nodiscard]] ChainValidation validate_chain(rust::Str path,
                                                 std::uint32_t stage_count) noexcept;
    [[nodiscard]] RuntimeStatus unload_model(Model& model) noexcept;

    [[nodiscard]] SessionResult create_session(const Model& model, SessionConfig config) noexcept;
    [[nodiscard]] RuntimeStatus release_session(Session& session) noexcept;

    [[nodiscard]] WorkloadResult workload(const Session& session) const noexcept;

    // Replaces the contents of `tokens`.
    [[nodiscard]] WorkloadStatus tokenize(const Session& session, rust::Str text,
                                          rust::Vec<std::int32_t>& tokens) noexcept;

    // Writes the normalised embedding into `output`; `dimensions` is also
    // filled on OutputTooSmall so the caller can resize and retry.
    [[nodiscard]] EmbedResult embed(const Session& session, rust::Slice<const std::int32_t> tokens,
                                    rust::Slice<float> output) noexcept;

    // `tokens` holds every group concatenated; `lengths[i]` is group i's size.
    [[nodiscard]] EmbedResult embed_batch(const Session& session,
                                          rust::Slice<const std::int32_t> tokens,
                                          rust::Slice<const std::uint32_t> lengths,
                                          rust::Slice<float> output) noexcept;

    [[nodiscard]] RerankResult rerank(const Session& session, rust::Str query, rust::Str document) noexcept;

    // One score per document; the whole group is decoded in a single pass.
    [[nodiscard]] WorkloadStatus rerank_batch(const Session& session, rust::Str query,
                                              rust::Slice<const rust::Str> documents,
                                              rust::Slice<float> scores) noexcept;

    [[nodiscard]] EncodeResult encode(const Session& session, rust::Slice<const std::int32_t> tokens) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Null when the runtime could not be allocated.
[[nodiscard]] std::unique_ptr<Bridge> new_bridge() noexcept;

} // namespace skippy::ffi
