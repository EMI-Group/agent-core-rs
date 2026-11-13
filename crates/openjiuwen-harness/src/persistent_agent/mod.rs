//! Persistent (long-lived, resumable) agent support: execution log, session state, window
//! management, continuation and handoff seams.
//!
//! Ports the self-contained subset of `openjiuwen/harness/persistent_agent/execution/`.
//! The module owns the state that survives one execution — an append-only JSONL journal, the
//! persisted session record, the reconstructed model window and the transfer records a successor
//! or a share bundle consumes — and owns nothing else: no agent loop, no model call, no host
//! queueing or delivery.
//!
//! Every entry point takes its clock and root directory from the caller, so the module is
//! deterministic and offline in tests. See [`CONTEXT.md`](https://example.invalid) in this
//! directory for the API contract and routing table; the module-level [`Clock`] is the crate's
//! existing injectable clock ([`crate::subagents::Clock`]) re-exported here.
//!
//! ```
//! use openjiuwen_harness::persistent_agent::{
//!     EntryPayload, ExecutionIdentity, ExecutionLog, LogCapacity,
//! };
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let dir = tempfile::TempDir::new().unwrap();
//! let identity = ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source")?;
//! let log = ExecutionLog::open(dir.path(), identity, LogCapacity::default()).await?;
//! log.append(EntryPayload::execution_started("say hi")).await?;
//! let snapshot = log.read().await?;
//! assert_eq!(snapshot.entries.len(), 1);
//! # Ok(())
//! # }
//! ```
pub mod continuation;
pub mod control;
pub mod emitter;
pub mod entry;
pub mod handoff;
pub mod identity;
pub mod log;
pub mod rails;
pub mod session;
pub mod share_file;
pub mod util;
pub mod window;

pub use crate::subagents::{rfc3339_from_ms, Clock, ManualClock, SystemClock};
