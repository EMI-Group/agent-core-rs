# openjiuwen-dev-tools / tune

## Intent

Port the pure part of Python `openjiuwen.dev_tools.tune` (`base.py`, the case loader) and keep the deferred optimizer/trainer surface as API-shaped stubs.
`Case`/`EvaluatedCase` are the shared data models consumed by `prompt_builder` bad-case repair and by `skill_evaluator`.
Everything here is offline and deterministic: no model calls, no environment reads, no wall-clock dependence.

## API Surface

- `Case` — `inputs`, `label`, `tools`, `case_id` (Python field names); `new` (validating), `with_tools`, `with_case_id`, `validate`.
- `EvaluatedCase` — `case`, `answer`, `score`, `reason`; `new` (score bounds), `validate`, accessors `inputs`/`label`/`tools`/`case_id`, `From<Case>` with score `0.0`.
- `CaseLoader` — `new`, `size`, `is_empty`, `cases`, `into_cases`, `get`, `iter`, `shuffle` (seeded xorshift64* Fisher-Yates), `split` (floor of `ratio * size`), iteration by value and by reference.
- `optimizer` — tuning constants (`DEFAULT_*`, `MIN_*`, `MAX_*`), `OptimizationResult`, `InstructionOptimizer`, `ExampleOptimizer`, `JointOptimizer`, `PromptSearchOptimizer`, `Trainer`, `optimize_prompt`; all pipelines are deferred and return `Error::Unsupported` naming the Python pipeline and `docs/PARITY.md`.
- The module root re-exports every type and constant; submodules stay private.

## Constraints

- Serialization: `Debug, Clone, PartialEq, Serialize, Deserialize` on every public data type, plus `schemars::JsonSchema` on `Case`/`EvaluatedCase`; unknown fields are ignored.
- Deserialization is permissive (missing `case_id` gets a fresh simple v4 UUID, missing `score` is `0.0`); only `new`/`validate` enforce the Python `pydantic` constraints.
- Field defaults live on individual fields (not `#[serde(default)]` on the container) because `Case` deliberately has no `Default` impl.
- Errors come from `openjiuwen_common::error` only: `Error::validation` for bad input, `Error::unsupported` for deferred pipelines.
- Deferred `optimize` methods stay `async` to mirror Python's `async def optimize`; `Trainer::train` stays synchronous like Python's `def train`.
- No `unsafe`, no panics on user input, no printing; tests may use `unwrap`/`expect`.
- Files ≤1000 lines, one responsibility per file.

## Routing Table

- `./base.rs` → `Case`, `EvaluatedCase`, validation, JSON round-trips (`base.rs` has no children)
- `./loader.rs` → `CaseLoader`, seeded PRNG, shuffle, split
- `./optimizer.rs` → tuning constants, `OptimizationResult`, deferred optimizer/trainer stubs
- `./mod.rs` → module docs, private submodules, public re-exports
