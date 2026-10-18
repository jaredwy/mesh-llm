# Porting a llama.cpp patch into `skippy_api`

This file is for whoever (human or agent) ports the next patch from
`skippy/llama_cpp/patches` into this library, and for anyone writing code here.
It records rules that were agreed the hard way; most of them exist because
something broke.

## What this library is

A C++23 façade over the patched llama.cpp staged runtime, plus Rust bindings in
`skippy/skippy-api-rs`. It owns *use* of the runtime, never the runtime itself.

```
include/            public headers, one per capability
  runtime.h         NativeRuntime / Model / Session handles, SessionConfig
  workload.h        workload classification, pooling
  embed.h           embed, embed_batch, EmbeddingView
  rerank.h          rerank, rerank_batch
  stage_plan.h      stage plans, profiles, frontier values, state effects
  stage_load.h      plan -> loader configuration
  stage_sink.h      compile-time sink concept (StageSink, DiscardSink, CopyFrameSink)
  execution_contract.h
  family.h          typed model families
  llama/            the only headers allowed to include llama/skippy headers
src/                implementations; src/llama/ owns everything native
tests/              GoogleTest, **no llama dependency allowed**
benchmarks/         A/B harness, case manifest, comparison script
tools/              probes (planner, stage pipeline, stage chain)
```

## Code rules

### Types and interfaces

- **No out-parameters above the C ABI.** Functions return their result. The only
  out-params are inside `skippy_*` calls we do not own, and they stop at the
  wrapper.
- **`std::expected<T, E>` when a value comes back.** **`std::optional<E>` when
  only a failure can come back.** Never `std::expected<void, E>`.
  `nullopt` means "it worked"; an engaged optional is the reason it did not.
- **`bool` is only a yes/no predicate** (`contains`, `guards_match`, `ranges_join`).
  Never an error channel, never a placeholder, never "did it work".
- **Strong types for identities.** `ProfileId`, `TensorId`, `ValueIdentity`,
  `LayerRange` — not bare strings and ints that can be swapped by accident.
- **Move-only by default** for anything owning a resource (models, sessions,
  plans, chains, runtimes). Delete copy, default move. No `shared_ptr` without
  a stated reason; no raw owning pointers.
- **`const` means what it says.** Mark a method `const` only when the object is
  genuinely unchanged. Mutating through a pointer from a `const` method is legal
  C++ and a lie to the reader — do not do it. `plan_stages`/`validate_chain` are
  const because their cache is not their state; `tokenize`/`embed` are not const
  because their buffers are.

### Style

- C++23. Concepts over templates-by-convention. Ranges over index loops, except
  where the ABI is index-based.
- Prefer compile-time policy (concepts) over runtime plugin registries.
- 4-space indent, 100 columns, attached braces, single-statement guards may stay
  on one line. `clang-format` is authoritative: `.clang-format` in this directory.
- Files under 1,000 lines; split by responsibility, never by symmetry. No
  `utils`/`helpers`/`common` buckets.
- Do not add a method that is a dozen lines of straight-line logic with three
  early returns — name the phases instead (clang-tidy enforces complexity).

### Rust

- Map to idiomatic Rust, not to the C++ shape. Handles wrap `UniquePtr<_>`; a
  session borrows its model (`StagedSession<'m>`), a model borrows its runtime.
- `&self` for read-only calls, `Pin<&mut _>` for mutating ones. Do not make
  anything `&mut` "just to be safe".
- Errors are typed (`WorkloadError`, `StageError`, `PlanError`) and carry the
  backend's message; do not flatten to strings at the boundary.

## Never modify llama.cpp from here

`skippy/llama_cpp/patches` is the only place llama.cpp changes live. This
library **wraps** the runtime; it never edits it, and no PR here may touch a
patch file, regenerate it, or carry a private llama fork.

When a patch you are porting would work better if llama itself changed:

1. **Do not make the change here.** Flag it instead, with the exact evidence —
   the symbol that is missing, the function that re-derives state per call, the
   cost you measured.
2. **Record it as upstream-shaped work** in the manifest entry
   (`benchmarks/cases.txt`, acceptance `blocked-upstream`) and in the README's
   "not covered" section, so it is visible rather than lost.
3. **Port what can be done outside llama**, and say plainly which half is not:
   for example the prepared-export work (`PreparedExport`) removes our per-hop
   allocation and descriptor rebuild, while llama still maps parts internally —
   that half needs an ABI entry point, an ABI version bump and a per-family
   implementation in the patch queue.
4. **Prefer an upstream-worthy patch to a local workaround.** If the change is
   generally useful (a missing accessor, a per-call re-derivation, a buffer that
   should be reusable), it belongs in `llama_cpp/patches` as its own patch with
   its own ABI note — not in a façade shim that hides the cost.

The test for "is this upstream-shaped?" is simple: could the change be expressed
as a C ABI function that any consumer of the staged runtime would want? If yes,
it is a patch, not a façade feature.

## Testing rules

- **Mocked unit tests belong in C++** (`tests/`, no llama). They pin contracts:
  classification, buffer bounds, batch layout, rejection paths.
- **Anything that needs a real model belongs in Rust** (`skippy/tests/end_to_end.rs`).
  The mock cannot see pooling selection, context geometry, or the retained-context
  pool — every bug found in those areas was found by the Rust suite.
- `SKIPPY_TEST_REQUIRE_MODELS=1` turns a missing GGUF into a failure; use it in CI.
- A change is not done until tests are green **and** the A/B row for its case has
  a verdict (see below).

## Porting a patch: the procedure

1. **Read the patch, not the docs.** `skippy/llama_cpp/patches/NNNN-*.patch`.
   Identify whether it adds (a) an ABI entry point, (b) graph/family behaviour,
   (c) metadata/validation, or (d) build wiring.
2. **Classify it** — this decides how it can ever be "proven":
   - **A — replaces an existing call.** The old C ABI does the same work.
     Proof: an A/B row, gated on equal outputs, then timed. Must be ≥ parity.
   - **B — new capability.** No old path exists (generation, staged execution,
     prepared tokens). Proof: a *budget* row against the cheapest correct
     reference implementation we can write (e.g. our chain vs a hand-driven
     native chain), gated on byte-identical results.
   - **C — metadata/validation.** Pure computation over a plan. Proof: a
     *ceiling* row — cost must stay a small fraction of what it serves
     (e.g. validate ≤ 1% of realize).
   If a patch cannot be gated, it is not ported yet.
3. **Port the pure half into C++** with no llama dependency, and unit-test it.
   Types and rules that do not need the runtime belong here.
4. **Wrap the native half** in `src/llama/` and expose it through the façade.
   One `expected`/`optional` at the boundary; convert native status + message
   into a typed failure and free the native error exactly once.
5. **Bind it in Rust** (`skippy-sys/cpp/bridge.{h,cpp}` + `skippy/src/lib.rs`),
   then add an E2E test against a real GGUF.
6. **Add a manifest case** and the benchmark pair:
   - `benchmarks/cases.txt`: `patch <TAB> class <TAB> case <TAB> filter <TAB> acceptance`
   - benchmark pairs named `Family/FFI` and `Family/API` (the comparison tool
     pairs on that convention).
7. **Run it**: `just skippy-ab <case> <model.gguf>` and record the verdict.
8. **Commit** with `just skippy-api-checkpoint "<conventional message>"`, which
   runs format, tidy, C++ tests and the Rust E2E suite first.

### Workflow commands

```bash
just skippy-api-verify [embed.gguf] [reranker.gguf]   # format, tidy, ctest, Rust
just skippy-api-checkpoint "<type(scope): subject>" [embed.gguf] [reranker.gguf]
just skippy-ab                                        # list A/B cases
just skippy-ab <case> <model.gguf>                    # run one case
just skippy-ab <case> <model.gguf>                    # prints WIN/PARITY/LOSS verdict
just skippy-llama-api-planner-probe <model.gguf>      # plan/probe a model
just skippy-llama-api-stage-pipeline <model.gguf> N   # chain N stages, compare token
```

## Traps that have already bitten us

- **Do not trust the compiler to catch a polarity flip.** Converting
  `expected<void, E>` → `optional<E>` turns `if (!valid) return valid;` from
  "propagate the failure" into "swallow it". Six tests passed while validation
  rejected nothing. Convert one file at a time, update the assertions in the same
  commit, and grep for `!` before you build.
- **`return {}` becomes `nullopt` in some functions and an engaged `optional` in
  others.** Check every one.
- **`ctest` can pass on a stale binary.** If the build failed, the pass means
  nothing. Always read the compiler output before the test summary.
- **A benchmark row is only a comparison if both sides do the same work.** The
  materialise row read "2.8x slower" for two rounds because the reference loop
  skipped fields we were reading. Gate on outputs first.
- **Medians of five lie on fixed-cost rows.** Use
  `--benchmark_repetitions=15 --benchmark_enable_random_interleaving=true` and
  read `_min`.
- **`rust::String` requires UTF-8.** Planner identities are **binary**
  (`resident_tensor_at_v1` returns an identity, not a name). Hex-encode them.
- **cxx cannot hold a non-copyable C++ object by value.** Use `UniquePtr<T>`,
  methods taking `Pin<&mut T>`, and declare `type T;` in the bridge block.
- **The formatter reflows lambdas.** A `str.replace()` over formatted C++ can
  silently no-op; grep after editing, not after compiling.
- **`ffi_vs_api_benchmark.cpp` warnings are errors.** A `nodiscard` result must
  be `(void)`-cast, and the benchmark will fail to compile (not run) if missed.

## Where things stand

See `README.md` for the patch coverage table and the current benchmark verdicts.
