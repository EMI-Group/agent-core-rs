//! The run ledger: persisted model-call usage records with aggregation and
//! resume.
//!
//! Python's `openjiuwen.rsi.usage` keeps a `ModelUsageObserver` that writes one
//! JSON line per model call to `model_calls.jsonl` inside the run directory,
//! replays that file when a run resumes (so a call is never charged twice),
//! exposes cumulative totals, and reports interrupted calls through
//! `check_delivery`. This module is the offline port of that contract:
//!
//! - [`record`] — the persisted types ([`UsageRecord`],
//!   [`StageRecord`], [`RunEvent`]) and
//!   the Python token-alias normalisation.
//! - [`usage`] — cumulative [`UsageTotals`] and the
//!   per-model / per-stage [`LedgerMetrics`].
//! - [`store`] — the [`LedgerStore`] seam plus the in-memory
//!   and JSONL-file backends.
//! - [`ledger`] — [`RunLedger`], which binds a task's
//!   persisted events, deduplicates by stable `call_id`, and tracks pending
//!   calls.
//!
//! Everything is content-free: a record carries the model name, token counters,
//! a status, and references — never prompts, responses, or credentials.
// The ledger type lives in `ledger/ledger.rs` to keep one responsibility per file.
#[allow(clippy::module_inception)]
pub mod ledger;
pub mod record;
pub mod store;
pub mod usage;

pub use ledger::RunLedger;
pub use record::{
    format_rfc3339, now_rfc3339, tokens_from_usage, usage_snapshot, ModelCall, ModelCallStatus,
    RunEvent, StageRecord, UsageRecord, UsageSnapshot, UsageTokens, STAGE_EVENT_TYPE,
    USAGE_EVENT_TYPE, USAGE_FAMILY, USAGE_KIND,
};
pub use store::{FileLedgerStore, InMemoryLedgerStore, LedgerStore};
pub use usage::{aggregate, LedgerMetrics, UsageTotals, UNATTRIBUTED_STAGE};
