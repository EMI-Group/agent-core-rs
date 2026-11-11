//! Ordered, provider-neutral observations emitted by an external harness.
//!
//! Mirrors `openjiuwen.harness_protocol.events` (v0.1.20). This module currently exposes the
//! event value vocabulary and the bounded observation-buffer contract; the payload models, the
//! `HarnessEvent` envelope, and the wire codec arrive with the rest of the events round.
//!
//! - [`enums`] — every event value enum (`TurnEventKind`, `OutputKind`, …) with its Python
//!   `.value` wire strings, plus [`TERMINAL_TURN_EVENT_KINDS`].
//! - [`buffer`] — [`EventBufferConfig`], the capacity/overflow contract a harness advertises.
//!
//! Everything is re-exported from the crate root as well, mirroring Python's eager
//! `__init__.__all__`.

pub mod buffer;
pub mod enums;

pub use buffer::EventBufferConfig;
pub use enums::{
    DiagnosticLevel, EventOverflowPolicy, EventRetention, HookEventPhase, ItemEventKind,
    ModelRequestStatus, OutputChannel, OutputKind, OutputOperation, TurnEventKind, UsageUpdateMode,
    TERMINAL_TURN_EVENT_KINDS,
};
