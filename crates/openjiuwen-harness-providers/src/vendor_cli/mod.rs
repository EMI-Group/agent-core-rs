//! Shared subprocess transport for the vendor CLI harnesses (Claude Code, Codex, DSH).
//!
//! Every vendor harness in this crate drives an external CLI: it resolves the binary, spawns it,
//! speaks newline-delimited JSON over stdin/stdout, observes its stderr for diagnostics, derives
//! the provider session id and translates host permissions into CLI answers. Those mechanics are
//! provider-independent, so they live here and the three adapters only contribute their own wire
//! message mapping.
//!
//! # Structure
//!
//! - [`binary`] — locating and validating the vendor executable.
//! - [`jsonl`] — pure NDJSON/JSON-object accessors shared by every decoder.
//! - [`child`] — one spawned CLI process with piped stdin/stdout/stderr and a bounded stderr tail.
//! - [`events`] — the observation [`EventBus`] and [`EnvelopeBuilder`] that feed the harness SPI
//!   event cursor.
//! - [`permissions`] — host permission translation for tool approvals and user input.
//! - [`session`] — deterministic provider session ids and resume-id resolution.
//! - [`driver`] — the vendor driver seam: turn requests, the observation [`TurnContext`] and the
//!   shared terminal-result constructors.
//! - [`harness`] — [`VendorHarness`], the generic `HarnessProtocol` shell one driver is wrapped by.
//!
//! Nothing here reimplements a harness: this module is plumbing and protocol scaffolding, and it
//! never talks to a vendor SDK.

#![allow(clippy::module_name_repetitions)] // `child::ChildProcess`, `binary::BinarySpec` read naturally

pub mod binary;
pub mod child;
pub mod driver;
pub mod events;
pub mod harness;
pub mod jsonl;
pub mod permissions;
pub mod session;

use std::sync::{Mutex, MutexGuard, PoisonError};

pub use binary::{is_available, resolve_binary, BinarySpec};
pub use child::{ChildProcess, ChildProcessConfig};
pub use driver::{
    completed_turn, failed_turn, interrupted_turn, turn_error, TurnContext, TurnRequest,
    VendorDriver, PROVIDER_SCHEMA_VERSION, TOOL_CALL_ITEM_TYPE,
};
pub use events::{EnvelopeBuilder, EventBus};
pub use harness::VendorHarness;
pub use jsonl::{
    encode_line, get_array, get_bool, get_f64, get_object, get_str, get_u64, parse_line,
    parse_line_labelled, snippet, string_list,
};
pub use permissions::{ask_tool_approval, ask_user_input, ApprovalDecision, PermissionPolicy};
pub use session::{checkpoint_str, derive_session_id, resolve_resume_id};

/// Locks `mutex`, recovering the guard when a previous holder panicked.
///
/// The shared state behind these mutexes is a plain queue or buffer whose invariants do not depend
/// on the panicking critical section, so a poisoned lock is reused instead of panicking again (the
/// workspace forbids panics on user input).
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
