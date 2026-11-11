//! The harness event envelope.
//!
//! Mirrors the Python `HarnessEvent` dataclass (v0.1.20): one ordered observation identified by
//! `sequence` and a Unix-epoch-second `timestamp`, correlated by session/agent/turn/item ids, and
//! carrying a [`HarnessEventPayload`].
//!
//! The envelope deliberately derives **neither** [`serde::Serialize`] nor [`serde::Deserialize`]:
//! a derived impl would emit the Rust field names and the payload as a tagged enum, bypassing the
//! wire key table and the schema-version gate in [`crate::serialization`]. Both impls below
//! delegate to that codec, so it stays the single source of truth. As a consequence,
//! `serde_json::to_value(&event)` equals
//! `json_object_to_value(&harness_event_to_dict(&event)?)`, and
//! `harness_event_from_dict(w)` equals `serde_json::from_value::<HarnessEvent>(json_object_to_value(&w))`
//! for every well-formed wire object.
//!
//! Constructors do not validate implicitly; decode paths (the wire codec,
//! [`from_value_validated`](crate::from_value_validated)) call [`Validate::validate`], while code
//! that builds an envelope with [`HarnessEvent::new`] and the `with_*` builders must call
//! `validate()` itself.

use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::events::payloads::HarnessEventPayload;
use crate::json::JsonValue;
use crate::validation::{require_finite_timestamp, require_unique_non_empty, Validate};

/// One ordered, provider-neutral observation emitted by a harness.
///
/// Every envelope field is always present on the wire (`null` for absent optional ids, `[]` for
/// an empty `causation_ids`); see [`crate::serialization`] for the exact key order.
#[derive(Debug, Clone, PartialEq)]
pub struct HarnessEvent {
    /// Monotonically increasing observation sequence within its host session.
    pub sequence: u64,
    /// Unix epoch seconds at which the harness observed the event.
    pub timestamp: f64,
    /// The payload this event observes.
    pub event: HarnessEventPayload,
    /// Host-side session identifier shared by every event of one run.
    pub host_session_id: String,
    /// Agent that produced the observation.
    pub agent_id: String,
    /// Provider-side session identifier, when the provider exposes one.
    pub provider_session_id: Option<String>,
    /// Turn this event belongs to; required for turn lifecycle and model request events.
    pub turn_id: Option<String>,
    /// Provider item this event belongs to, when it concerns a single item.
    pub item_id: Option<String>,
    /// Caller-supplied correlation id tying related observations together.
    pub correlation_id: Option<String>,
    /// Ids of the events that caused this one; never contains empty or duplicate entries.
    pub causation_ids: Vec<String>,
}

impl HarnessEvent {
    /// Builds an envelope without validating it.
    ///
    /// `event` accepts any payload type convertible into [`HarnessEventPayload`], and
    /// `host_session_id`/`agent_id` accept anything convertible into [`String`]. Optional ids and
    /// `causation_ids` start empty; use [`HarnessEvent::with_provider_session_id`],
    /// [`HarnessEvent::with_turn_id`], [`HarnessEvent::with_item_id`],
    /// [`HarnessEvent::with_correlation_id`], and [`HarnessEvent::with_causation_ids`] to set them.
    ///
    /// Callers must call [`Validate::validate`] before handing the event to a consumer; decode
    /// paths validate automatically.
    #[must_use]
    pub fn new(
        sequence: u64,
        timestamp: f64,
        event: impl Into<HarnessEventPayload>,
        host_session_id: impl Into<String>,
        agent_id: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            timestamp,
            event: event.into(),
            host_session_id: host_session_id.into(),
            agent_id: agent_id.into(),
            provider_session_id: None,
            turn_id: None,
            item_id: None,
            correlation_id: None,
            causation_ids: Vec::new(),
        }
    }

    /// Sets [`provider_session_id`](Self::provider_session_id).
    #[must_use]
    pub fn with_provider_session_id(mut self, provider_session_id: impl Into<String>) -> Self {
        self.provider_session_id = Some(provider_session_id.into());
        self
    }

    /// Sets [`turn_id`](Self::turn_id).
    #[must_use]
    pub fn with_turn_id(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    /// Sets [`item_id`](Self::item_id).
    #[must_use]
    pub fn with_item_id(mut self, item_id: impl Into<String>) -> Self {
        self.item_id = Some(item_id.into());
        self
    }

    /// Sets [`correlation_id`](Self::correlation_id).
    #[must_use]
    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    /// Replaces [`causation_ids`](Self::causation_ids) with `causation_ids`.
    #[must_use]
    pub fn with_causation_ids(mut self, causation_ids: Vec<String>) -> Self {
        self.causation_ids = causation_ids;
        self
    }
}

impl Validate for HarnessEvent {
    /// Checks the envelope invariants in Python's `__post_init__` order.
    ///
    /// `sequence` is unsigned, so Python's non-negative check is subsumed by the type.
    ///
    /// # Errors
    ///
    /// The first violated invariant as [`HarnessError::Validation`], followed by the payload's own
    /// validation error.
    fn validate(&self) -> HarnessResult<()> {
        require_finite_timestamp("event timestamp", self.timestamp)?;
        if self.host_session_id.is_empty() || self.agent_id.is_empty() {
            return Err(HarnessError::validation(
                "event host_session_id and agent_id must not be empty",
            ));
        }
        let optional_ids = [
            self.provider_session_id.as_deref(),
            self.turn_id.as_deref(),
            self.item_id.as_deref(),
            self.correlation_id.as_deref(),
        ];
        if optional_ids.into_iter().flatten().any(str::is_empty) {
            return Err(HarnessError::validation(
                "optional event IDs must not be empty strings",
            ));
        }
        require_unique_non_empty("event causation_ids", &self.causation_ids)?;
        if matches!(self.event, HarnessEventPayload::TurnLifecycle(_)) && self.turn_id.is_none() {
            return Err(HarnessError::validation(
                "turn lifecycle event requires turn_id",
            ));
        }
        if matches!(self.event, HarnessEventPayload::ModelRequest(_)) && self.turn_id.is_none() {
            return Err(HarnessError::validation(
                "model request event requires turn_id",
            ));
        }
        self.event.validate()
    }
}

impl Serialize for HarnessEvent {
    /// Serializes through the wire codec so the envelope key order and the schema gate apply.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value = crate::serialization::encode_event(self).map_err(serde::ser::Error::custom)?;
        Serialize::serialize(&value, serializer)
    }
}

impl<'de> Deserialize<'de> for HarnessEvent {
    /// Decodes through the wire codec, including its validation and schema-version gate.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        let object = crate::json::json_object_from_value(value)
            .ok_or_else(|| serde::de::Error::custom("payload must be a JSON object"))?;
        crate::serialization::decode_event(&object).map_err(serde::de::Error::custom)
    }
}
