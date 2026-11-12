# openjiuwen-auto-harness

## Intent

Standalone facade that preserves the Python import path `openjiuwen.rsi.auto_harness` for consumers that treat `auto_harness` as its own package.
It is a pure re-export shim over `openjiuwen_rsi::auto_harness` — the canonical implementation — and declares no types or behaviour of its own.
Python reference: `openjiuwen.rsi.harness_rsi.auto_harness` (v0.1.20), a bounded, resumable self-optimisation session loop.
Purpose: a consumer that only needs the auto-harness loop can depend on this crate without adopting the rest of the `openjiuwen-rsi` surface at the call site.

## API Surface

Exactly the canonical surface, re-exported at the crate root; no new items except `crate_name()`.
- `pub use openjiuwen_rsi::auto_harness::*;` guarantees future canonical items reach the facade.
- Explicit re-exports (greppable, documented in `src/lib.rs`):
  - config: `AutoHarnessConfig`, `PipelinePreference`
  - loop: `AutoHarnessOrchestrator`, `AutoHarnessOrchestratorBuilder`, `DEFAULT_SEED_BODY`, `SEED_CANDIDATE_ID`
  - harness seam: `HarnessRunner`, `HarnessEvaluator`, `HarnessRunRequest`, `HarnessRunOutcome`
  - budget: `Budget`, `BudgetLimits`, `BudgetStatus`
  - checkpoints: `CheckpointStore`, `InMemoryCheckpointStore`, `KvCheckpointStore`, `SessionCheckpointStore`, `SessionState`
  - events: `AutoHarnessEvent`
  - records: `OptimizationTask`, `IterationRecord`, `SessionSummary`, `Gap`, `StageResult`, `StageStatus`, `TaskStatus`, `StopConditions`, `StopDecision`, `StopReason`
  - submodules: `budget`, `builder`, `checkpoint`, `config`, `event`, `orchestrator`, `runner`, `task`
- `#[must_use] pub const fn crate_name() -> &'static str` returns `"openjiuwen-auto-harness"`.

## Constraints

- Facade only: no type, trait, or function may be defined here beyond `crate_name()`; never fork the canonical definitions.
- Because the facade re-exports the same items, `openjiuwen_auto_harness::T` and `openjiuwen_rsi::auto_harness::T` are the same type (pinned by `tests/reexports.rs` via `TypeId`).
- Dependency is `openjiuwen-rsi` only (plus the workspace lints); add a dependency only when the facade itself needs it.
- `#![forbid(unsafe_code)]`; library code defines no `unsafe`, no globals, and no I/O.
- All behaviour contracts (errors, async, serde, logging, determinism) are inherited from `openjiuwen-rsi`; do not document or assert different ones here.

## Routing Table

- `../openjiuwen-rsi/src/harness_rsi/auto_harness/` → canonical implementation (`config.rs`, `budget.rs`, `task.rs`, `event.rs`, `runner.rs`, `checkpoint.rs`, `builder.rs`, `orchestrator.rs`)
- `../openjiuwen-rsi/CONTEXT.md` → authoritative field-level surface documentation for the re-exported items
- `./src/lib.rs` → the entire facade (glob + explicit re-exports + `crate_name()`)
- `./tests/reexports.rs` → identity and value-interop tests against the canonical crate

## Test Strategy

- `tests/reexports.rs` asserts `TypeId` equality for `AutoHarnessConfig`, `AutoHarnessOrchestrator`, `AutoHarnessEvent`, `Budget`, `SessionState`, `StopReason`, and the `checkpoint`/`config` submodule spellings, against both `openjiuwen_rsi::auto_harness` and `openjiuwen_rsi::harness_rsi::auto_harness`.
- The same file checks value interop: facade-built `BudgetLimits`/`AutoHarnessConfig` feed the canonical `Budget` and `StopConditions`, and results type-check as facade items; plus `crate_name()` stability.
- Behavioural coverage lives in `openjiuwen-rsi` (`src/harness_rsi/auto_harness/orchestrator_tests.rs`); this crate must not duplicate it.
- `cargo test -p openjiuwen-auto-harness` runs offline and needs no fixtures, fake LLM server, or runtime.
