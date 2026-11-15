//! Cross-module integration suite for the `tune` layer.
//!
//! The crate's `tests/` directory is a sibling node, so the integration coverage
//! lives here as a `#[cfg(test)]` module: each test composes two or more `tune`
//! submodules through their public surface only.
//!
//! Covered seams, one test per scenario:
//!
//! - `dataset` → `loader` → `Trainer`: the tagged cut, the batch plan, and the
//!   samplers a training loop is driven with (`dataset`);
//! - `optimizer` → `dataset`: split provenance travelling from a tagged cut into
//!   an `OptimizationResult` and an `EvaluationSummary`, and the held-out
//!   isolation a fitted candidate is guaranteed (`trainer`);
//! - `evaluator` → `prompt_search`: measured telemetry feeding the latency,
//!   token-usage, and cost reward components (`telemetry`);
//! - `optimizer::adopt` → `prompt_search`: the adoption service over a JSONL
//!   prompt memory, plus the good/bad case split (`adopt`);
//! - `prompt_search` → filesystem: the JSONL run log round trip and the records a
//!   prompt search emits (`run_log`);
//! - `optimizer` → `binding`: an optimizer winner bound back into a parameter,
//!   a two-parameter agent, and a tool binding against in-process stubs
//!   (`binding`);
//! - `chat_agent` → `evaluator`: the offline chat adapter evaluating and
//!   optimizing a case list (`chat`).
//!
//! Everything is offline and deterministic: explicit case ids, caller-supplied
//! timestamps and seeds, `TempWorkspace` for the only filesystem use, and no
//! model, network, clock read, sleep, or environment read.

mod support;

mod adopt;
mod binding;
mod chat;
mod dataset;
mod run_log;
mod telemetry;
mod trainer;
