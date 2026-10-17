# skippy_api

A C++23 façade over the patched llama.cpp stage runtime for the non-chat
workloads — embedding, reranking, and encoder prompt — with no chat or
generation paths in it.

```
include/            public headers; the only part consumers include
  runtime.h           RuntimeBackend concept, NativeRuntime registry, handles
  types.h             Token / SequenceId / TokenIndex, zero-copy token views
  workload.h          workload classification and pooling
  embed.h             embed() and embed_batch()
  rerank.h            rerank() and rerank_batch()
  encode.h            encode_prompt()
  llama/              the only code that includes llama.h
src/                implementations
tests/              mocked unit tests (no llama)
benchmarks/         FFI-vs-façade A/B harness and its report
```

## Rules the code follows

- **C++23, type-safe by default.** Concepts describe what a backend must
  provide; fallible calls return `std::expected`; absence is `std::optional`;
  ids are scoped enums over `int32_t` rather than implicit integers.
- **The core has no backend dependency.** Everything above `llama/` compiles
  without `llama.h`, which is what lets the unit tests run against mocks.
- **Ranges over loops**, and named algorithms over hand-written iteration.
- **Ownership is visible.** `std::unique_ptr` with a named deleter for native
  resources, move-only handles for models and sessions, no owning raw pointers.
  `shared_ptr` is not used; the registry owns, handles borrow.
- **One responsibility per file**, and new behaviour goes into the module that
  owns the capability rather than a shared bucket.

## Building and checking

```bash
just skippy-llama-api-test            # 133 mocked unit tests
just skippy-llama-api-format-check    # clang-format, using .clang-format
just skippy-llama-api-tidy            # clang-tidy, fails on any finding
just skippy-llama-api-ffi-bench-compare <model.gguf>
```

`clang-tidy` is not on `PATH` on every machine; set `SKIPPY_CLANG_TIDY` to its
path (Homebrew: `/opt/homebrew/opt/llvm/bin/clang-tidy`). Three checks are
disabled in `.clang-tidy` on purpose, with the reason recorded there.

Real-model behaviour is covered from the Rust side, not here:

```bash
just skippy-api-rs-e2e <embed.gguf> <reranker.gguf>
```

## Checkpoint workflow

```bash
just skippy-api-verify <embed.gguf> <reranker.gguf>   # format, tidy, tests
just skippy-api-checkpoint "feat(skippy-api): ..." <embed.gguf> <reranker.gguf>
```

`verify` is the gate CI-equivalent for this tree; `checkpoint` runs it, stages
`skippy/skippy_api`, `skippy/skippy-api-rs`, `just/skippy.just`, and
`.gitignore`, then commits with the subject you pass. Both accept the models
as optional arguments — without them the Rust tests run unit-only, and
`SKIPPY_TEST_REQUIRE_MODELS` is left unset so they skip rather than fail.
