//! Ordered, provider-neutral observations emitted by an external harness.
//!
//! Mirrors `openjiuwen.harness_protocol.events` (v0.1.20). The module exposes the event value
//! vocabulary, the payload models, the [`HarnessEvent`] envelope, and the bounded
//! observation-buffer contract; the wire codec lives in [`crate::serialization`].
//!
//! - [`buffer`] — [`EventBufferConfig`], the capacity/overflow contract a harness advertises.
//! - [`enums`] — every event value enum (`TurnEventKind`, `OutputKind`, …) with its Python
//!   `.value` wire strings, plus [`TERMINAL_TURN_EVENT_KINDS`].
//! - [`envelope`] — [`HarnessEvent`], the ordered observation envelope with its ids, timestamp,
//!   and validation rules.
//! - [`payloads`] — the ten payload models, the [`HarnessEventPayload`] union, retention classes
//!   ([`event_retention`]), and their validation rules.
//!
//! Everything is re-exported from the crate root as well, mirroring Python's eager
//! `__init__.__all__`.

pub mod buffer;
pub mod enums;
pub mod envelope;
pub mod payloads;

pub use buffer::EventBufferConfig;
pub use enums::{
    DiagnosticLevel, EventOverflowPolicy, EventRetention, HookEventPhase, ItemEventKind,
    ModelRequestStatus, OutputChannel, OutputKind, OutputOperation, TurnEventKind, UsageUpdateMode,
    TERMINAL_TURN_EVENT_KINDS,
};
pub use envelope::HarnessEvent;
pub use payloads::{
    event_retention, DiagnosticEvent, HarnessEventPayload, HookObservedEvent, ItemLifecycleEvent,
    ModelRequestEvent, OutputEvent, ProviderEvent, StateChangedEvent, TurnLifecycleEvent,
    UnknownEvent, UsageUpdatedEvent,
};
