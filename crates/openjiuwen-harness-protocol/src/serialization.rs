//! Shared event wire codec (schema `"1"`).
//!
//! Mirrors the Python `openjiuwen.harness_protocol.serialization` module (v0.1.20): the single
//! source of truth for the event envelope wire shape, the payload key sets, and the schema-version
//! gate that turns a newer-schema event into an `UnknownEvent`.
//!
//! This module currently defines only [`EVENT_WIRE_SCHEMA_VERSION`]; the envelope codec
//! (`harness_event_to_dict`, `harness_event_from_dict`, and their JSON-string forms) is added by
//! the envelope round so that payloads and the envelope stay independently testable.

/// Wire schema version of the event envelope understood by this crate.
///
/// Known payloads encode with this version; a decoded event carrying any other version becomes an
/// `UnknownEvent` and is re-encoded losslessly with the version it arrived with.
pub const EVENT_WIRE_SCHEMA_VERSION: &str = "1";
