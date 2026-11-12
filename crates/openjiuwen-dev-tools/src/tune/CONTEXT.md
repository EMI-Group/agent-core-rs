# openjiuwen-dev-tools / tune

## Intent

Port the pure part of Python `openjiuwen.dev_tools.tune` (`base.py`, the case loader) and keep the deferred optimizer/trainer surface as API-shaped stubs.
`Case`/`EvaluatedCase` are the shared data models consumed by `prompt_builder` bad-case repair and by `skill_evaluator`, so they live here and are never duplicated.
Everything in this module is offline and deterministic: no model calls, no environment reads, no wall-clock dependence.

## API Surface

- `Case` — `inputs`, `label`, `tools`, `case_id` (Python field names); `new` (validating), `with_tools`, `with_case_id`, `validate`.
- `EvaluatedCase` — `case`, `answer`, `score`, `reason`; `new` (score bounds), `validate`, accessors `inputs`/`label`/`tools`/`case_id`, `From<Case>` producing score `0.0` and empty answer/reason.
- `CaseLoader` — `new`, `size`, `is_empty`, `cases`, `into_cases`, `get`, `iter`, `shuffle` (seeded xorshift64* Fisher-Yates), `split` (floor of `ratio * size`), iteration by value and by reference, JSON round-trip through `{"cases":[...]}`.
- `optimizer` — tuning constants (`DEFAULT_*`, `MIN_*`, `MAX_*`), `OptimizationResult { best_prompt, score, iterations }`, `InstructionOptimizer`, `ExampleOptimizer`, `JointOptimizer`, `PromptSearchOptimizer`, `Trainer`, `optimize_prompt`.
- The module root re-exports every type and constant required by consumers; `base`, `loader`, and `optimizer` stay private.

## Constraints

- Serialization: `Debug, Clone, PartialEq, Serialize, Deserialize` on every public data type, plus `schemars::JsonSchema` on `Case`/`EvaluatedCase`; unknown fields are ignored.
- Python JSON field names round-trip exactly: `inputs`, `label`, `tools`, `case_id`, `case`, `answer`, `score`, `reason`.
- Deserialization is permissive (missing `case_id` gets a fresh simple v4 UUID, missing `score` is `0.0`, missing `CaseLoader.cases` is empty); only `new`/`validate` enforce the Python `pydantic` constraints.
- Errors come from `openjiuwen_common::error` only: `Error::validation` for bad input, `Error::unsupported` for deferred pipelines.
- No `unsafe`, no panics on user input, no printing; tests may use `unwrap`/`expect`.
- Files ≤1000 lines, one responsibility per file.

## Design Decisions

- Shuffle uses a self-contained xorshift64* generator (`seed == 0` remapped to the golden-ratio constant) instead of Python's `random.shuffle`, so the module needs no `rand` dependency and a seed always yields the same permutation.
- `split` partitions a contiguous slice at `floor(ratio * n)` rather than sampling randomly; callers that want a random split shuffle first.
- Field-level serde defaults replace the container-level `#[serde(default)]` on `Case`/`EvaluatedCase` because those types intentionally have no `Default` impl (an all-empty case is invalid); `CaseLoader` does derive `Default` and carries the container attribute.
- Score bounds and map emptiness are enforced in constructors/`validate`, not in `Deserialize`, so malformed-but-parseable JSON round-trips and validation is an explicit, testable step.
- Deferred `optimize` methods stay `async` to mirror Python's `async def optimize`; `Trainer::train` stays synchronous like Python's `def train`.
- `OptimizationResult` is the reduced `{best_prompt, score, iterations}` shape rather than Python's per-iteration records and token usage, matching the documented ported surface.
- Constants are re-exported from the module root because private submodules would otherwise leave `pub const`s unreachable under the workspace `unreachable_pub` lint.

## Deferred

- `InstructionOptimizer` — Python instruction-optimization pipeline depends on Python-only optimizer internals (prompt-diffing agents); returns `Error::unsupported` naming the pipeline and `docs/PARITY.md`.
- `ExampleOptimizer` — Python few-shot example-selection pipeline; returns `Error::unsupported`.
- `JointOptimizer` — Python joint instruction-and-example pipeline; returns `Error::unsupported`.
- `PromptSearchOptimizer` — Python prompt-search pipeline; returns `Error::unsupported`.
- `Trainer` — Python trainer depends on the optimizer/evaluator internals; returns `Error::unsupported`.
- `optimize_prompt` — convenience wrapper that delegates to `InstructionOptimizer::optimize` and therefore inherits its deferred error.
- Python's dataset samplers, chat-agent adapters, and evaluator implementations under `tune/` are not ported.

## Test Strategy

- Unit tests live in each file's `#[cfg(test)]` module and are entirely offline: no network, no environment reads, no wall-clock dependence.
- `base.rs` covers constructor validation (empty inputs/label/case_id), score bounds (`0.0`/`1.0` accepted; `-0.1`, `1.1`, `NaN`, infinities rejected), Python-field JSON round-trips, UUID defaulting, unknown-field tolerance, schema exposure, and `From<Case>` accessors.
- `loader.rs` covers size/emptiness/get/iteration/split (including ratios `0.0`, `1.0`, `NaN`, and out-of-range), shuffle determinism and permutation preservation across seeds, tiny/empty loaders, and JSON round-trip plus defaulting.
- `optimizer.rs` covers constant values and ordering, every stub returning `Error::Unsupported` with messages naming the pipeline and `docs/PARITY.md`, default constructibility, and `OptimizationResult` JSON round-trip.
- `mod.rs` exercises the re-export surface end-to-end (loader → shuffle → split → evaluator) and re-exported constants.

## Routing Table

- `./base.rs` → `Case`, `EvaluatedCase`, validation, JSON round-trips (no children)
- `./loader.rs` → `CaseLoader`, seeded PRNG, shuffle, split, iteration
- `./optimizer.rs` → tuning constants, `OptimizationResult`, deferred optimizer/trainer stubs
- `./mod.rs` → module docs, private submodules, public re-exports
