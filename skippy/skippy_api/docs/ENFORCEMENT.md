# Enforcement rules for `skippy_api`

These rules apply to every patch ported after #2481. They are written for
whoever ports the next one, human or model, and they are deliberately strict:
a rule that lives only in prose has already failed once in this tree.

A rule entry — here or in `AGENTS.md` — is not complete until it names:

1. **the rung** it is enforced at (§1), and why not a higher one;
2. **the artefact** that enforces it (a type in `include/`, a `static_assert`
   in `tests/type_contracts_test.cpp`, a check in `tidy/`, a test, or a
   benchmark gate);
3. **the finding** it would have caught, by review id or commit.

The type specifications referenced below are in
[`docs/TYPE_DESIGN.md`](TYPE_DESIGN.md).

---

## 1. The enforcement ladder

Climb from the top. Stop at the first rung that can express the rule, and
write down why the rungs above could not.

| rung | the rule becomes | when it applies |
|---|---|---|
| **1 — unrepresentable** | a type: the wrong value cannot be constructed | the rule is about a **value**: what it means, where it came from, which operations are valid, what invariant holds (presence, bound, capacity, ownership, lifetime relation, NUL-termination, byte order) |
| **2 — uncompilable** | a deleted or ref-qualified overload with a reason, pinned by `static_assert(!requires { … })` | the rule retires a **spelling** the type system can see (a call on a temporary, an old overload, a conversion) |
| **3 — unlintable** | a clang-tidy check in `tidy/`, `WarningsAsErrors: '*'` — **boundary code only** (§3) | the rule is about the **shape of code** at the llama or Rust boundary that no type can see: call order, what is inside a handler/scope/method, where a function may be called from, how native text and pointers cross |
| **4 — untestable to pass** | a mocked unit test, a Rust e2e, a hostile-input fixture, an A/B gate | the rule is about **behaviour with data**: results, byte equality, rejection of a corrupt file, a timing ratio |
| **5 — written down** | an `AGENTS.md` trap with the symptom, cause and rule | nothing above can express it — and the entry says so |

### The two questions that decide type vs check

1. **Can a signature change make the wrong program ill-formed?** If yes, it is
   a type (rung 1 or 2). A check is advisory, bypassable with `NOLINT`, sees
   one translation unit, and runs only where the recipe points it. A type
   runs everywhere the code compiles.
2. **Is the forbidden thing a pattern in the code rather than a property of a
   value?** If yes — and only at a boundary — it is a check (rung 3). No type
   can express "this call must be preceded by that call", "nothing allocates
   inside this handler", or "this getter is read in one place".

Both answers no → rung 4 or 5.

### Rules about the ladder itself

- **Types are the default. A check is the last resort.** A check for something
  a type could fix is a defect: the check will be `NOLINT`ed, the type cannot
  be. An agent proposing a check must write, in the rule entry, the signature
  change it tried first and why it could not express the rule.
- **Checks live at the boundaries and nowhere else.** The two places where C
  ABI shapes force patterns a type cannot carry are the llama side
  (`src/llama/`, `include/llama/`, `tools/`) and the Rust side
  (`skippy-api-rs/skippy-sys/cpp/`). Inside the core (`include/`, `src/`
  above `llama/`) a rule that cannot be a type is a design smell: change the
  design, do not lint around it. Scoping is mechanical: the boundary
  directories carry their own `.clang-tidy` that enables `skippy-*`; the
  top-level `.clang-tidy` enables only the migration checks (§3), each with a
  sunset condition.
- **Never write a type to encode a code-shape rule.** "No raw loops" is not a
  type. A type invented to carry a shape rule is a tax with no proof.
- **Every rung-1 type that retires a spelling also takes rung 2.** The old
  call must fail to compile with a message that names the rule
  (`SKIPPY_DELETE("…")`), and `tests/type_contracts_test.cpp` must assert the
  failure. Without the assertion the old spelling returns in the next
  refactor.
- **A check is not done without three things:** a fixture under `tidy/test/`
  with a positive and a negative case; a README row naming the finding it
  would have caught; its options in `getModuleOptions()`.
- **A check must not need whole-program knowledge.** If the rule depends on
  what another translation unit does, it is a type, a link-time symbol, or a
  test.
- **If a check's allowlist grows past its true positives, it is a review
  item, not a check.** Delete it and write the trap.
- **A `NOLINT` carries a reason** (`// NOLINT(skippy-x): <why the rule does
  not apply here>`). A bare `NOLINT` is a rule violation.
- **A rung-5 entry cites its evidence** — the llama/ggml source line, the
  measurement, or the commit — because two of this tree's traps were
  contradicted by the pinned source within a week.

---

## 2. When a pattern becomes a rule

`AGENTS.md` already says "fix the class, not the instance". These thresholds
make that mechanical, so an agent cannot finish a port by hand-applying the
same change N times.

| trigger | threshold | what must exist before the PR is complete |
|---|---|---|
| **the same defect** found in more than one place | **2** | a rule at the highest rung that can express it, and both sites migrated to it |
| **the same change** applied to several sites in one PR (the same shape, the same reason) | **3 sites, or 2 files** | the shape as a type or a named helper first; a check only if the shape is a boundary pattern no type can carry; all N sites use it |
| **a comment that legislates** ("must", "never", "always", "the caller guarantees") about code elsewhere | **2 comments on the same subject** | the subject as a type or a check; the comments deleted or reduced to one line that names the artefact |
| **a review finding** whose fix is a one-liner at one site | 1 | the one-liner **and** the question "where else can this shape occur?" answered in the commit message; if the answer is ≥ 2, the row above applies |

Rationale for the numbers: two of anything can be coincidence; three is a
convention, and a convention that is not a type or a check is one that the
next port will break. Two defects is already a class.

Procedure: the rule lands in its own commit *before* the port commit — type +
contract test (or check + fixture) + the AGENTS entry + the migration of the
existing sites — so the port is reviewed against the rule rather than the rule
against the port. The six hand-rolled `string_view → const char*` loops
(staged.cpp, three probes, the harness, the planner probe fix) are the shape
this table exists for: it crossed the threshold at the third copy and was
still prose at the sixth.

---

## 3. API surface rules: ownership, borrowing, and the Rust model

The public surface is `include/` plus what the bridge exposes. It is used from
Rust through cxx, so **the Rust memory model is the design target**: if an
ownership relation cannot be written as a Rust signature, it is not allowed on
the surface.

### 3.1 Every public type is one of three kinds, visibly

| kind | C++ shape | Rust shape | examples |
|---|---|---|---|
| **Owner** | move-only; `std::unique_ptr<T, NamedDeleter>` or a class with deleted copy; destructor is the complete, nonthrowing release | `UniquePtr<T>` / an owned struct with `Drop` | `LlamaModel`, `LlamaSession`, `StagedModel`, `StageChain`, `PlanStore` |
| **Borrow** | `std::span`, `std::string_view`, `T&`, a lease; the owner it borrows from is named in the same signature or documented as `*this` | `&'a T` / `&'a mut T` with the lifetime of the owner | `NativeEmbeddingView` (until the next pass), `StageLoadConfig` views (pinned by `PlanLease`), `ArenaView` |
| **Value** | trivially or cheaply copyable, holds no resource, no pointer | `Copy`/`Clone` struct or enum | `Token`, `SessionGeometry`, `WorkloadInfo`, `Failure<Code>` |

A type that is none of these (a copyable class holding a raw pointer, a struct
with a `shared_ptr` member, a handle that is both a value and an owner) is not
allowed on the surface. `shared_ptr` is an implementation detail of RCU inside
`PlanStore`; it never appears in a public signature. Rust cannot express
shared mutable ownership safely, so neither does this API.

### 3.2 Ownership rules

- **No raw owning pointers on the surface.** A `T*` in a public signature is a
  C-ABI parameter at the call that creates or frees it, wrapped in a
  `unique_ptr` with a named deleter *in the same statement*. A naked
  `skippy_model*`/`llama_context*` never outlives the statement that produced
  it. (`ErrorSlot`, `llama_context_ptr`, `EmbeddingBatchPtr` are the patterns.)
- **A borrow names its owner and its invalidation set.** A method that returns
  a view says, in its declaration comment, which calls invalidate it ("borrows
  the context until the next decode or encode"). If the invalidation set is
  more than one method, do not return a borrow: return an owned result, or a
  lease that pins (`PlanLease`). In Rust terms: a borrow that cannot be
  `fn f(&mut self) -> &[T]` or `fn f(&self) -> Lease<'_>` is not a borrow the
  API may hand out.
- **Mutability is in the signature, never behind `const`.** A method that
  mutates — including through a pointer member, including by calling a native
  function that synchronises — is non-const, so Rust can bind it
  `Pin<&mut T>`; a `const` method is one Rust may call through `&T` from any
  number of borrowers. (*Check at the boundary:*
  `skippy-const-native-mutation`.) `plan_stages`/`validate_chain` are const
  because their cache is not their state; `tokenize`/`embed` are not, because
  their buffers are.
- **Handles are tokens, not pointers.** `Model`/`Session` are non-forgeable
  slot+generation ids that own nothing; a stale handle is rejected, never
  dereferenced. A handle crosses the bridge as `UniquePtr<Handle>` and comes
  back to `unload`/`release`; it is never copied, never reused.
- **Length travels with the pointer.** `std::span` / `rust::Slice`, never
  `(ptr, n)` as two parameters above the C ABI, never a pointer with an
  implied length.
- **The caller owns output buffers.** A function that produces an array fills
  a caller-provided span and reports the required length on
  `OutputTooSmall { required }`. The callee never allocates a result for the
  caller, so Rust can pass `&mut [f32]` and reuse it. (`embed_into`,
  `rerank_batch` into `scores` are the pattern; `embed_batch` returning
  `Vec<Vec<f32>>` is the counter-example to retire.)
- **Nothing from cxx is stored.** `rust::Str`, `rust::Slice`, `rust::Vec&`
  are consumed within the call that received them; the bridge copies or
  converts, never keeps.
- **Threading is declared per type, once.** Every owner states its thread
  model in its header comment in these words: *not thread-affine; serialise
  calls* (→ `unsafe impl Send`, no `Sync`), *immutable after construction*
  (→ `Send + Sync`), or *single-threaded, owner serialises* (→ neither). The
  bridge's `unsafe impl Send` list must match that declaration type-for-type;
  a type missing from one list is a rule violation (`StagedModel`/
  `StagedSession`/`StageChainHandle` today).
- **Destruction order is encoded, not remembered.** Members that depend on
  another are declared after it, with the one-line comment that says so
  (`idle_contexts_` after `model_`; `backend_` first in `NativeRuntime`).
  Every owner's destructor is `noexcept` and complete; there is no fallible
  `close()`.
- **No exceptions cross the surface.** Every public function is `noexcept`
  and returns a result, or returns `expected<T, Failure<Code>>` /
  `Fallible<E>` and is not `noexcept`. The bridge is the exception barrier;
  nothing above it may assume one. (*Check at the boundary:*
  `skippy-bridge-barrier`, `skippy-no-alloc-in-oom-handler`.)
- **Native errors are freed exactly once, at the call.** `ErrorSlot` owns
  `skippy_error*`; its message is copied into the `Failure` before the slot
  dies.

### 3.3 The header ownership table

Every public header starts with the block `runtime.h` has — *who owns what,
top down; what each borrow borrows from; what cleanup is* — and every
function's comment answers, in one line each where it applies: **owns** (what
it takes ownership of), **borrows** (what it returns a view into and until
when), **invalidates** (which earlier borrows it kills). A function that needs
none of the three says nothing. This is the text an agent reads before
writing a Rust signature; if it cannot be turned into one, the C++ is wrong.

### 3.4 Verification

`tests/type_contracts_test.cpp` asserts, per public type:
`!std::is_copy_constructible_v` for owners, `std::is_nothrow_destructible_v`,
`std::is_nothrow_move_constructible_v`, and `std::is_trivially_copyable_v`
for values. The Rust side's `unsafe impl Send` list is compared to the C++
thread-model declarations by a test that greps both (until cxx can carry the
attribute). Borrow-returning methods have an ASan borrow-stress test
(`borrow_stress_test` is the pattern).

---

## 4. Type rules (rung 1 and 2)

Each rule names its type; the types are specified in
[`docs/TYPE_DESIGN.md`](TYPE_DESIGN.md) and land in `include/`.

- **A result that can only fail is `Fallible<E>`.** Never
  `std::expected<void, E>`, never a bare `optional<E>`, never `bool`.
  `Fallible` has no `operator bool`, so a polarity flip does not compile.
  *Migration checks:* `skippy-no-expected-void`,
  `skippy-optional-error-polarity`.
- **Parse, don't validate.** A value with an invariant has a private
  constructor and a `static expected<T, E> make(...)`. Code below the boundary
  takes `T` and cannot be reached with unchecked input. (`SessionGeometry`,
  `LayerRange::make`, `ArenaSlice` via `IdentityArena::put`,
  `SizedGgmlType::parse`.)
- **A capacity is computed from the resource, never supplied by the caller.**
  `ContextCapacity` is constructible from a `const llama_context&` and nothing
  else; the pool's `recycle` takes only the context. There is no parameter
  through which a wrong label can arrive. *Why not a check:* the bug was a
  correct-looking argument.
- **Admission is a type.** Anything llama will assert on (batch size vs
  `n_ubatch`, sequence count vs `n_seq_max`) is checked by a function that
  returns the only type the native call accepts (`BatchLimit::admit →
  AdmittedBatch`; `run_pass(const AdmittedBatch&)`). A field that "should be
  checked" (`batch_limit_`) is not a rule.
- **A native getter with a non-zero default, a padded value, or a sentinel is
  read in exactly one place, behind a type that records what it means.**
  `n_cls_out` (defaults to 1) → `std::optional<ClassifierHead>` detected from
  the tensor table; `n_ctx` (padded to 256) → `ContextCapacity`. The pure
  tables (`classify`) take `optional`/structural facts only and never a
  family. *Boundary check:* `skippy-single-call-site`.
- **Presence is structural, not nominal.** Whether a model has a head, a
  frontier, a pooling mode is decided by the tensor table or the plan, never
  by an architecture-name table. `ModelFamily` is a Rust-facing label and
  never appears in a classification decision.
- **A `const char*` reaches a C ABI only through `ZString` / `CStringTable`.**
  `string_view::data()` is never passed where a C string is wanted; arena
  views do not expose `data()`. One `StagedRuntimeConfig` owns the strings and
  the pointer arrays.
- **Integer width changes go through `narrow<To>()` (→ `optional`) or
  `checked_mul`/`checked_add`.** A `static_cast` between integer types is
  allowed only in `narrow.h`, on a constant that fits, or with `// bounded by
  <the proof>` on the line. *Boundary check:* `skippy-explicit-narrowing`;
  flags `-Wconversion -Wsign-conversion` everywhere.
- **Bytes from a file decode through `WireScalar`.** `Reader::read` is
  constrained to integral/floating types, swaps on big-endian hosts, and file
  integers become enums only through a bounded `parse` that returns
  `optional`. `static_assert(sizeof(std::size_t) >= 8)` where the parser
  assumes a 64-bit address space.
- **A member that returns something borrowing `*this` is `&`-qualified, and
  its `&&` overload is deleted with a reason.**
  (`LlamaBackend::plan_model_stages`.) A temporary cannot hand out a lease.
  This is §3.2's borrow rule, enforced.
- **Out of memory is a code, never text.** `Failure<Code>::out_of_memory()`
  allocates nothing; the barrier's `bad_alloc` arm returns it. A `noexcept`
  function is nothrow by construction, not by care.
- **Configuration is a type with a private constructor.** `BackendId::detect()`
  / `BackendId::named()`; no `BackendId{"metal"}` can be spelled. Everything
  that selects a plan (`StageRequest`) is one value and its `key()` is the
  only cache-key producer.
- **Row-major buffers are `std::mdspan`** (`[row][dim]`), feature-tested with
  a fallback and a "fallback can go" test.
- **Every forbidden spelling is pinned.** `tests/type_contracts_test.cpp`
  holds the `static_assert(!requires …)` / `!std::is_constructible_v` /
  `!std::is_invocable_v` lines for every rule above and §3.4's ownership
  assertions, plus the layout facts the ABI depends on.

---

## 5. Lint rules (rung 3, boundaries only)

The checks live in `tidy/`; the verify recipe loads them with `--load` and
runs with `--warnings-as-errors='*'`. Scope is set by per-directory
`.clang-tidy` files, not by discipline.

**Boundary checks** — enabled by `src/llama/.clang-tidy`, `tools/.clang-tidy`
and `skippy-api-rs/skippy-sys/cpp/.clang-tidy`. Each encodes a best practice
at the llama or Rust boundary that no signature can carry:

| check | boundary | rule |
|---|---|---|
| `skippy-preceded-by` | llama | `llama_memory_clear` immediately after `llama_synchronize` (one pair per protocol) |
| `skippy-single-call-site` | llama | a getter with a default/padded value is read in its one named place |
| `skippy-no-lock-across-native-call` | llama | no mutex held across a native call |
| `skippy-const-native-mutation` | llama | no mutation through a native pointer from a const method |
| `skippy-explicit-narrowing` | llama, Rust | ABI widths are converted with `narrow<>` or a stated bound |
| `skippy-bridge-barrier` | Rust | every `noexcept` bridge member is `return <barrier>(...)` |
| `skippy-no-alloc-in-oom-handler` | Rust, llama | nothing allocates in a `bad_alloc`/catch-all handler of a `noexcept` function |
| `skippy-rust-string-lossy` | Rust | native text crosses as `rust::String::lossy` |

**Migration checks** — enabled tree-wide by the top-level `.clang-tidy`, each
with a sunset:

| check | sunsets when |
|---|---|
| `skippy-no-expected-void`, `skippy-optional-error-polarity` | `Fallible<E>` has replaced every failure-only result (keep them anyway; they are free) |
| `skippy-no-void-silencing` | never — it is cheaper than the review comment |
| `skippy-no-process-wide-state` | the memo moves into `LlamaBackend`; keep for the allowlist |
| `skippy-no-raw-loops` | never — house style |

**Built-in checks that stay on:** `bugprone-use-after-move`,
`bugprone-implicit-widening-of-multiplication-result`,
`cppcoreguidelines-narrowing-conversions`,
`readability-implicit-bool-conversion`, `bugprone-exception-escape`,
`bugprone-suspicious-stringview-data-usage` — the last existed when the probe
bug shipped; `tools/` was outside the recipe. **Scope is part of every
rule.**

---

## 6. Test rules (rung 4)

`AGENTS.md` "Testing rules" stand; these are additions.

- **A hostile input is a fixture, not a thought.** Every parser branch that
  rejects file bytes has a fixture producing the typed error. A rejection
  without a fixture is unverified.
- **A benchmark gate must be able to fail.** Before timing, assert the
  compared artefact is non-empty on both sides and byte-equal; a gate guarded
  by `has_value()` on a frame that is empty by design proves nothing
  (`stage_chain`, P1-15). The manifest note names what was compared.
- **The causal fixture is required in the CI lane that has models**
  (`SKIPPY_TEST_CAUSAL` under `SKIPPY_TEST_REQUIRE_MODELS=1`).
- **Every rung-1 type lands with its negative compile test in the same
  commit; every borrow-returning method lands with its ASan stress case.**

---

## 7. Porting checklist for the next patch

Run before opening the PR. Each line names the rung and the artefact.

| when the patch adds… | ask | artefact |
|---|---|---|
| a public type | owner, borrow, or value (§3.1)? thread model declared? | the kind in its shape; the header ownership table; contract asserts |
| a public function | owns / borrows / invalidates — can it be written as a Rust signature? | the three one-liners; a `Pin<&mut>`/`&` decision that matches `const` |
| a call to a llama/ggml/skippy getter | default, padding, or sentinel in its value? | rung 1 type at one call site; `skippy-single-call-site.Functions` += getter |
| a native call that must follow another | is it a protocol? | rung 3: `skippy-preceded-by` pair |
| a value llama will assert on | can llama abort on it? | rung 1: an `Admitted*` type returned by the check |
| an integer that means something | tokens, sequences, layers, bytes, offsets? | the strong type; products through `make`/`checked_*` |
| a `static_cast` between integers | is the bound proven? | `narrow<>()`, or `// bounded by …` with the proof |
| a `const char*` into the ABI | where do the bytes live, NUL-free? | `ZString` / `CStringTable` / `StagedRuntimeConfig` |
| a literal naming a backend, profile, graph config | configuration? | private-constructor type; `StageRequest::key()` |
| a `noexcept` function | can every path allocate or convert? | barrier, or `bugprone-exception-escape` clean; OOM arm returns a code |
| a `static` or namespace-scope variable | who owns it, who resets it in tests? | the owning object, or an allowlist entry with the reason |
| a result type | can it carry a message? void-on-success? | `expected<T, Failure<Code>>` or `Fallible<E>` |
| a classification decision | structural (tensor table, plan) or nominal (family)? | structural |
| the same change at a third site | §2 | the rule commit first |
| a benchmark row | can the gate fail? | non-empty, byte-equal comparison before timing |
| an `AGENTS.md` trap | evidence? | the source line or measurement, cited |
| a `NOLINT` | why does the rule not apply here? | the reason on the line |
| a clang-tidy check | which signature change was tried, and why it could not express the rule? is this a boundary? | the answer in the rule entry; the per-directory `.clang-tidy` |

---

## 8. Rule-entry template

```
- **<rule, one sentence, imperative>.** <why: the failure it prevents, one sentence>.
  Rung <n>: <artefact path or check name>; proof: <static_assert / fixture / test name>.
  Would have caught: <review id or commit>.
  <If rung ≥ 3: the signature change tried first and why it could not express the rule; the boundary it applies to.>
```
