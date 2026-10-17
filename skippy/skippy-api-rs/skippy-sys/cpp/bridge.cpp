#include "skippy-sys/cpp/bridge.h"

#include "skippy-sys/src/lib.rs.h"

#include "embed.h"
#include "encode.h"
#include "llama/backend.h"
#include "llama/model_inventory.h"
#include "llama/stage_planner.h"
#include "rerank.h"
#include "types.h"
#include "workload.h"
#include "workload_error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace skippy::ffi {
namespace {

constexpr RuntimeStatus to_status(RuntimeError error) noexcept {
  switch (error) {
  case RuntimeError::InvalidModel:
    return RuntimeStatus::InvalidModel;
  case RuntimeError::InvalidSession:
    return RuntimeStatus::InvalidSession;
  case RuntimeError::ModelInUse:
    return RuntimeStatus::ModelInUse;
  case RuntimeError::AllocationFailed:
    return RuntimeStatus::AllocationFailed;
  case RuntimeError::LoadFailed:
    return RuntimeStatus::LoadFailed;
  case RuntimeError::SessionCreationFailed:
    return RuntimeStatus::SessionCreationFailed;
  case RuntimeError::UnsupportedConfiguration:
    return RuntimeStatus::UnsupportedConfiguration;
  case RuntimeError::CapacityExceeded:
    return RuntimeStatus::CapacityExceeded;
  case RuntimeError::InvalidBackendResult:
    return RuntimeStatus::InvalidBackendResult;
  }
  return RuntimeStatus::InternalError;
}

constexpr WorkloadStatus to_status(WorkloadError error) noexcept {
  switch (error) {
  case WorkloadError::InvalidSession:
    return WorkloadStatus::InvalidSession;
  case WorkloadError::UnsupportedWorkload:
    return WorkloadStatus::UnsupportedWorkload;
  case WorkloadError::UnsupportedClassifierCount:
    return WorkloadStatus::UnsupportedClassifierCount;
  case WorkloadError::EmptyInput:
    return WorkloadStatus::EmptyInput;
  case WorkloadError::TooManyTokens:
    return WorkloadStatus::TooManyTokens;
  case WorkloadError::TooManySequences:
    return WorkloadStatus::TooManySequences;
  case WorkloadError::TokenizationFailed:
    return WorkloadStatus::TokenizationFailed;
  case WorkloadError::NoEmbeddingDimensions:
    return WorkloadStatus::NoEmbeddingDimensions;
  case WorkloadError::OutputTooSmall:
    return WorkloadStatus::OutputTooSmall;
  case WorkloadError::AllocationFailed:
    return WorkloadStatus::AllocationFailed;
  case WorkloadError::DecodeFailed:
    return WorkloadStatus::DecodeFailed;
  case WorkloadError::EncodeFailed:
    return WorkloadStatus::EncodeFailed;
  case WorkloadError::NoDecoderStartToken:
    return WorkloadStatus::NoDecoderStartToken;
  case WorkloadError::NoOutput:
    return WorkloadStatus::NoOutput;
  case WorkloadError::DimensionMismatch:
    return WorkloadStatus::DimensionMismatch;
  }
  return WorkloadStatus::InternalError;
}

template <typename Status> struct boundary_status;
template <> struct boundary_status<RuntimeStatus> {
  static constexpr auto allocation = RuntimeStatus::AllocationFailed;
  static constexpr auto internal = RuntimeStatus::InternalError;
};
template <> struct boundary_status<WorkloadStatus> {
  static constexpr auto allocation = WorkloadStatus::AllocationFailed;
  static constexpr auto internal = WorkloadStatus::InternalError;
};

template <typename Result> struct status_of {
  using type = decltype(Result::status);
};
template <typename Result>
  requires std::is_enum_v<Result>
struct status_of<Result> {
  using type = Result;
};

template <typename Result>
constexpr Result failed_with(typename status_of<Result>::type status) noexcept {
  if constexpr (std::is_enum_v<Result>) {
    return status;
  } else {
    Result result{};
    result.status = status;
    return result;
  }
}

// Last line of defence: nothing below may unwind into Rust.
template <typename Result, std::invocable Body>
  requires std::same_as<std::invoke_result_t<Body>, Result>
Result guarded(Body &&body) noexcept {
  using Boundary = boundary_status<typename status_of<Result>::type>;
  try {
    return std::forward<Body>(body)();
  } catch (const std::bad_alloc &) {
    return failed_with<Result>(Boundary::allocation);
  } catch (...) {
    return failed_with<Result>(Boundary::internal);
  }
}

constexpr NativePooling to_native(skippy::NativePooling pooling) noexcept {
  return static_cast<NativePooling>(std::to_underlying(pooling));
}

constexpr NativePooling to_native(std::optional<Pooling> pooling) noexcept {
  return pooling ? static_cast<NativePooling>(std::to_underlying(*pooling))
                 : NativePooling::Unspecified;
}

constexpr WorkloadInfo to_info(const skippy::WorkloadInfo &info) noexcept {
  return WorkloadInfo{
      .kind = static_cast<WorkloadKind>(std::to_underlying(info.kind)),
      .pooling = to_native(info.pooling),
      .embedding_pooling = to_native(info.embedding_pooling),
      .output_dimensions = info.output_dimensions,
      .classifier_outputs = info.classifier_outputs,
      .has_encoder = info.has_encoder,
      .has_decoder = info.has_decoder,
  };
}

std::string_view as_view(rust::Str text) noexcept {
  return {text.data(), text.size()};
}

// A Rust &[i32] is layout-compatible with Token; `scratch` is written only on
// standard libraries without start_lifetime_as.
std::span<const Token> as_tokens(rust::Slice<const std::int32_t> tokens,
                                 std::vector<Token> &scratch) {
  return detail::tokens_from_native(
      std::span<const std::int32_t>{tokens.data(), tokens.size()}, scratch);
}

// Moves a freshly minted handle onto the heap for Rust. If that allocation
// fails the handle has not been moved from yet, so the resource is given back.
ModelResult to_result(llama::LlamaRuntime &runtime,
                      std::expected<Model, RuntimeError> loaded) noexcept {
  if (!loaded)
    return {.status = to_status(loaded.error()), .model = nullptr};
  std::unique_ptr<Model> model{new (std::nothrow) Model(std::move(*loaded))};
  if (!model) {
    (void)runtime.unload_model(*loaded);
    return {.status = RuntimeStatus::AllocationFailed, .model = nullptr};
  }
  return {.status = RuntimeStatus::Ok, .model = std::move(model)};
}

SessionResult to_result(llama::LlamaRuntime &runtime,
                        std::expected<Session, RuntimeError> created) noexcept {
  if (!created)
    return {.status = to_status(created.error()), .session = nullptr};
  std::unique_ptr<Session> session{new (std::nothrow)
                                       Session(std::move(*created))};
  if (!session) {
    (void)runtime.release_session(*created);
    return {.status = RuntimeStatus::AllocationFailed, .session = nullptr};
  }
  return {.status = RuntimeStatus::Ok, .session = std::move(session)};
}

template <typename Status>
Status to_plain_status(std::expected<void, RuntimeError> result) noexcept {
  return result ? Status::Ok : to_status(result.error());
}

} // namespace

struct Bridge::Impl {
  llama::LlamaRuntime runtime{std::in_place};
  std::vector<Token> token_scratch;
};

Bridge::Bridge() : impl_(std::make_unique<Impl>()) {}
Bridge::~Bridge() = default;

std::unique_ptr<Bridge> new_bridge() noexcept {
  try {
    return std::make_unique<Bridge>();
  } catch (...) {
    return nullptr;
  }
}

namespace {

rust::String hex_of(std::string_view value) {
  static constexpr std::string_view kHex = "0123456789abcdef";
  std::string out;
  out.reserve(value.size() * 2);
  for (const char byte : value) {
    const auto raw = static_cast<unsigned char>(byte);
    out.push_back(kHex[(raw >> 4U) & 0xFU]);
    out.push_back(kHex[raw & 0xFU]);
  }
  return rust::String(std::move(out));
}

} // namespace

rust::Vec<StagePlanInfo>
Bridge::plan_stages(rust::Str path, std::uint32_t stage_count) noexcept {
  rust::Vec<StagePlanInfo> output;
  const auto fail = [&output](const std::string &message) {
    StagePlanInfo info{};
    info.ok = false;
    info.error = rust::String(message);
    output.push_back(std::move(info));
    return output;
  };
  if (stage_count == 0)
    return fail("a plan needs at least one stage");

  const std::string model_path(path);
  auto inventory = skippy::probe::read_inventory(model_path);
  if (!inventory)
    return fail(inventory.error());
  const auto layers =
      static_cast<std::int32_t>(skippy::probe::layer_count_from(*inventory));

  std::vector<skippy::LayerRange> ranges;
  ranges.reserve(stage_count);
  const std::int32_t base = layers / static_cast<std::int32_t>(stage_count);
  const std::int32_t remainder =
      layers % static_cast<std::int32_t>(stage_count);
  std::int32_t start = 0;
  for (std::uint32_t index = 0; index < stage_count; ++index) {
    const std::int32_t width =
        base + (static_cast<std::int32_t>(index) < remainder ? 1 : 0);
    ranges.push_back(skippy::LayerRange{start, start + width});
    start += width;
  }

  auto plans = skippy::llama::plan_model_stages(model_path, ranges);
  if (!plans)
    return fail(plans.error().message);

  for (const skippy::StageLoadConfig &config : *plans) {
    StagePlanInfo info{};
    info.ok = true;
    info.frontier_values = config.frontier_values;
    info.state_effects = config.state_effects;
    info.written_state_effects = config.written_state_effects;
    info.recurrent_state = config.recurrent_state;
    info.layer_start = config.layers.start;
    info.layer_end = config.layers.end;
    info.execution_contract_hex = hex_of(config.execution_contract);
    info.resident_identity_hex.reserve(config.resident_identities.size());
    for (const std::string_view identity : config.resident_identities) {
      info.resident_identity_hex.push_back(hex_of(identity));
    }
    info.import_binding_hex.reserve(config.activation_import_bindings.size());
    for (const std::string_view value : config.activation_import_identities) {
      info.import_identity_hex.push_back(hex_of(value));
    }
    for (const std::string_view value : config.activation_import_bindings) {
      info.import_binding_hex.push_back(hex_of(value));
    }
    info.export_binding_hex.reserve(config.activation_export_bindings.size());
    for (const std::string_view value : config.activation_export_identities) {
      info.export_identity_hex.push_back(hex_of(value));
    }
    for (const std::string_view value : config.activation_export_bindings) {
      info.export_binding_hex.push_back(hex_of(value));
    }
    output.push_back(std::move(info));
  }
  return output;
}

ChainValidation Bridge::validate_chain(rust::Str path,
                                       std::uint32_t stage_count) noexcept {
  ChainValidation result{};
  result.ok = false;

  const std::string model_path(path);
  auto inventory = skippy::probe::cached_inventory(model_path);
  if (!inventory) {
    result.error = rust::String(inventory.error());
    return result;
  }
  const auto layers =
      static_cast<std::int32_t>(skippy::probe::layer_count_from(*inventory));

  std::vector<skippy::LayerRange> ranges;
  ranges.reserve(stage_count);
  const std::int32_t base =
      layers / static_cast<std::int32_t>(stage_count == 0 ? 1 : stage_count);
  const std::int32_t remainder =
      layers % static_cast<std::int32_t>(stage_count == 0 ? 1 : stage_count);
  std::int32_t start = 0;
  for (std::uint32_t index = 0; index < stage_count; ++index) {
    const std::int32_t width =
        base + (static_cast<std::int32_t>(index) < remainder ? 1 : 0);
    ranges.push_back(skippy::LayerRange{start, start + width});
    start += width;
  }

  auto plans = skippy::llama::plan_model_plans(model_path, ranges);
  if (!plans) {
    result.error = rust::String(plans.error().message);
    return result;
  }
  if (const auto valid = skippy::validate_chain(*plans); !valid) {
    result.error = rust::String("plan chain rejected: " +
                                std::string(skippy::describe(valid.error())));
    return result;
  }
  result.ok = true;
  return result;
}

ModelResult Bridge::load_model(rust::Str path) noexcept {
  return guarded<ModelResult>([&] {
    return to_result(impl_->runtime, impl_->runtime.load_model(ModelConfig{
                                         .path = std::string(as_view(path))}));
  });
}

RuntimeStatus Bridge::unload_model(Model &model) noexcept {
  return guarded<RuntimeStatus>([&] {
    return to_plain_status<RuntimeStatus>(impl_->runtime.unload_model(model));
  });
}

SessionResult Bridge::create_session(const Model &model,
                                     SessionConfig config) noexcept {
  return guarded<SessionResult>([&] {
    return to_result(
        impl_->runtime,
        impl_->runtime.create_session(
            model, skippy::SessionConfig{.dedicated_context = true,
                                         .max_tokens = config.max_tokens,
                                         .sequences = config.sequences}));
  });
}

RuntimeStatus Bridge::release_session(Session &session) noexcept {
  return guarded<RuntimeStatus>([&] {
    return to_plain_status<RuntimeStatus>(
        impl_->runtime.release_session(session));
  });
}

WorkloadResult Bridge::workload(const Session &session) const noexcept {
  return guarded<WorkloadResult>([&] {
    const auto info = impl_->runtime.with_session(
        session, [](const llama::LlamaSession &native) {
          return to_info(native.workload());
        });
    if (!info)
      return WorkloadResult{.status = to_status(info.error()), .info = {}};
    return WorkloadResult{.status = RuntimeStatus::Ok, .info = *info};
  });
}

WorkloadStatus Bridge::tokenize(const Session &session, rust::Str text,
                                rust::Vec<std::int32_t> &tokens) noexcept {
  return guarded<WorkloadStatus>([&] {
    const auto tokenized =
        impl_->runtime.with_session(session, [&](llama::LlamaSession &native) {
          return native.tokenize(as_view(text));
        });
    if (!tokenized)
      return WorkloadStatus::InvalidSession;
    if (!*tokenized)
      return to_status(tokenized->error());

    tokens.clear();
    tokens.reserve((*tokenized)->size());
    std::ranges::for_each(**tokenized, [&](Token token) {
      tokens.push_back(std::to_underlying(token));
    });
    return WorkloadStatus::Ok;
  });
}

EmbedResult Bridge::embed(const Session &session,
                          rust::Slice<const std::int32_t> tokens,
                          rust::Slice<float> output) noexcept {
  return guarded<EmbedResult>([&] {
    const auto result = skippy::embed(
        impl_->runtime, session, as_tokens(tokens, impl_->token_scratch),
        EmbeddingOutput<F32Format>{.values = {output.data(), output.size()},
                                   .metadata = {}});
    if (result)
      return EmbedResult{.status = WorkloadStatus::Ok,
                         .dimensions = result->dimensions};

    EmbedResult failed{.status = to_status(result.error()), .dimensions = 0};
    if (result.error() == WorkloadError::OutputTooSmall) {
      failed.dimensions =
          impl_->runtime
              .with_session(session,
                            [](const llama::LlamaSession &native) {
                              return native.embedding_dimensions();
                            })
              .value_or(0);
    }
    return failed;
  });
}

EmbedResult Bridge::embed_batch(const Session &session,
                                rust::Slice<const std::int32_t> tokens,
                                rust::Slice<const std::uint32_t> lengths,
                                rust::Slice<float> output) noexcept {
  return guarded<EmbedResult>([&] {
    const auto flat = as_tokens(tokens, impl_->token_scratch);

    // Split the flattened buffer into the per-group views embed_batch takes.
    std::vector<std::span<const Token>> groups;
    groups.reserve(lengths.size());
    std::size_t at = 0;
    for (const std::uint32_t length : lengths) {
      if (at + length > flat.size())
        return EmbedResult{.status = WorkloadStatus::EmptyInput,
                           .dimensions = 0};
      groups.emplace_back(flat.subspan(at, length));
      at += length;
    }
    if (at != flat.size())
      return EmbedResult{.status = WorkloadStatus::EmptyInput, .dimensions = 0};

    const auto result = skippy::embed_batch(
        impl_->runtime, session, groups,
        EmbeddingOutput<F32Format>{.values = {output.data(), output.size()},
                                   .metadata = {}});
    if (result)
      return EmbedResult{.status = WorkloadStatus::Ok,
                         .dimensions = result->dimensions};

    EmbedResult failed{.status = to_status(result.error()), .dimensions = 0};
    if (result.error() == WorkloadError::OutputTooSmall) {
      failed.dimensions =
          impl_->runtime
              .with_session(session,
                            [](const llama::LlamaSession &native) {
                              return native.embedding_dimensions();
                            })
              .value_or(0);
    }
    return failed;
  });
}

RerankResult Bridge::rerank(const Session &session, rust::Str query,
                            rust::Str document) noexcept {
  return guarded<RerankResult>([&] {
    const auto result = skippy::rerank(impl_->runtime, session, as_view(query),
                                       as_view(document));
    if (!result)
      return RerankResult{
          .status = to_status(result.error()), .score = 0.0f, .token_count = 0};
    return RerankResult{.status = WorkloadStatus::Ok,
                        .score = result->score,
                        .token_count = result->token_count};
  });
}

WorkloadStatus Bridge::rerank_batch(const Session &session, rust::Str query,
                                    rust::Slice<const rust::Str> documents,
                                    rust::Slice<float> scores) noexcept {
  return guarded<WorkloadStatus>([&] {
    std::vector<std::string_view> views;
    views.reserve(documents.size());
    for (const rust::Str document : documents) {
      views.emplace_back(document.data(), document.size());
    }
    const auto result = skippy::rerank_batch(
        impl_->runtime, session, std::string_view{query.data(), query.size()},
        views, std::span<float>{scores.data(), scores.size()});
    return result ? WorkloadStatus::Ok : to_status(result.error());
  });
}

EncodeResult Bridge::encode(const Session &session,
                            rust::Slice<const std::int32_t> tokens) noexcept {
  return guarded<EncodeResult>([&] {
    const auto result = skippy::encode_prompt(
        impl_->runtime, session, as_tokens(tokens, impl_->token_scratch));
    if (!result)
      return EncodeResult{.status = to_status(result.error()),
                          .decoder_start = -1};
    return EncodeResult{.status = WorkloadStatus::Ok,
                        .decoder_start = std::to_underlying(*result)};
  });
}

// Compile-time tests: the boundary enums stay value-compatible with the core.
static_assert(std::to_underlying(WorkloadKind::CausalGeneration) ==
              std::to_underlying(skippy::WorkloadKind::CausalGeneration));
static_assert(std::to_underlying(WorkloadKind::Embedding) ==
              std::to_underlying(skippy::WorkloadKind::Embedding));
static_assert(std::to_underlying(WorkloadKind::Rerank) ==
              std::to_underlying(skippy::WorkloadKind::Rerank));
static_assert(std::to_underlying(WorkloadKind::EncoderDecoder) ==
              std::to_underlying(skippy::WorkloadKind::EncoderDecoder));

static_assert(std::to_underlying(NativePooling::Unspecified) ==
              std::to_underlying(skippy::NativePooling::Unspecified));
static_assert(std::to_underlying(NativePooling::None) ==
              std::to_underlying(skippy::NativePooling::None));
static_assert(std::to_underlying(NativePooling::Mean) ==
              std::to_underlying(skippy::NativePooling::Mean));
static_assert(std::to_underlying(NativePooling::Cls) ==
              std::to_underlying(skippy::NativePooling::Cls));
static_assert(std::to_underlying(NativePooling::Last) ==
              std::to_underlying(skippy::NativePooling::Last));
static_assert(std::to_underlying(NativePooling::Rank) ==
              std::to_underlying(skippy::NativePooling::Rank));
static_assert(std::to_underlying(NativePooling::None) ==
              std::to_underlying(Pooling::None));
static_assert(std::to_underlying(NativePooling::Last) ==
              std::to_underlying(Pooling::Last));

static_assert(to_status(RuntimeError::ModelInUse) == RuntimeStatus::ModelInUse);
static_assert(to_status(WorkloadError::OutputTooSmall) ==
              WorkloadStatus::OutputTooSmall);
static_assert(to_native(std::optional<Pooling>{}) ==
              NativePooling::Unspecified);
static_assert(to_native(std::optional<Pooling>{Pooling::Mean}) ==
              NativePooling::Mean);
static_assert(failed_with<EmbedResult>(WorkloadStatus::InternalError).status ==
              WorkloadStatus::InternalError);
static_assert(failed_with<RuntimeStatus>(RuntimeStatus::InternalError) ==
              RuntimeStatus::InternalError);

static_assert(sizeof(Token) == sizeof(std::int32_t));

} // namespace skippy::ffi
