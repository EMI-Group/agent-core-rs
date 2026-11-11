//! Payload models carried by the harness event envelope.
//!
//! Mirrors the payload section of the Python `openjiuwen.harness_protocol.events` module
//! (v0.1.20). [`HarnessEventPayload`] is the Rust analogue of Python's `HarnessEventPayload`
//! union: one variant per payload type, plus [`UnknownEvent`] for events from a newer schema.
//!
//! Wire rules (see `DESIGN.md` §4.4 and §6):
//!
//! - Field declaration order is the JSON key order; every key is always serialized, with `null`
//!   for [`None`] and `[]`/`{}` for empty containers. No `skip_serializing_if` anywhere.
//! - `#[serde(default)]` appears on exactly the fields that carry a Python default; a required
//!   field errors when absent. Unknown keys are tolerated (never `deny_unknown_fields`).
//! - [`HarnessEventPayload`] deliberately derives neither [`serde::Serialize`] nor
//!   [`serde::Deserialize`]: the Python wire shape keeps `event_type` and `schema_version` at the
//!   envelope level with the payload nested, which an internally tagged enum cannot express (it
//!   would collide with `ProviderEvent::event_type`). The envelope codec in
//!   [`serialization`](crate::serialization) is the only codec.
//! - Constructors do not validate implicitly; decode paths validate via [`Validate`]. Messages
//!   are the Python `__post_init__` strings, byte for byte.

use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::events::enums::{
    DiagnosticLevel, EventRetention, HookEventPhase, ItemEventKind, ModelRequestStatus,
    OutputChannel, OutputKind, OutputOperation, TurnEventKind, UsageUpdateMode,
};
use crate::json::{JsonObject, JsonValue};
use crate::results::{
    ContentBlock, MonetaryAmount, TurnError, TurnMessage, TurnResult, TurnStatus, TurnUsage,
};
use crate::serialization::EVENT_WIRE_SCHEMA_VERSION;
use crate::state::HarnessState;
use crate::validation::{require_finite_timestamp, require_non_empty, Validate};

/// Provider-neutral update for one stable output content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct OutputEvent {
    /// Provider-stable identifier of the output block this event updates.
    pub output_id: String,
    /// Whether `content` is free-form text or structured JSON.
    pub kind: OutputKind,
    /// The output content itself.
    pub content: JsonValue,
    /// How `content` relates to earlier events for the same `output_id`; defaults to `snapshot`.
    #[serde(default)]
    pub operation: OutputOperation,
    /// Semantic channel carrying this block; defaults to `answer`.
    #[serde(default)]
    pub channel: OutputChannel,
    /// Index of this block when one logical output is split across blocks; defaults to `0`.
    #[serde(default)]
    pub content_index: u64,
    /// Provider-native content type (for example `text/plain`), when stated.
    #[serde(default)]
    pub content_type: Option<String>,
    /// Provider-specific extension data; defaults to an empty object.
    #[serde(default)]
    pub data: JsonObject,
}

/// Lifecycle update for one provider item, preserving its native shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ItemLifecycleEvent {
    /// Which lifecycle transition happened.
    pub kind: ItemEventKind,
    /// Provider-native item type (for example `tool_call`).
    pub item_type: String,
    /// Provider-specific item data; defaults to an empty object.
    #[serde(default)]
    pub data: JsonObject,
}

/// Incremental or cumulative normalized usage information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UsageUpdatedEvent {
    /// The usage counters carried by this update.
    pub usage: TurnUsage,
    /// Whether `usage` replaces or adds to the previously reported counters; defaults to
    /// `cumulative`.
    #[serde(default)]
    pub mode: UsageUpdateMode,
}

/// One completed physical model request observed inside a turn.
///
/// A provider emits this only when the host declared
/// `HostCapability::ModelRequestObservation`; it is emitted once, after the request ended, and
/// always before any item the request caused and before the turn's terminal event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ModelRequestEvent {
    /// Provider-stable identity of the request within the session.
    pub request_id: String,
    /// Terminal status of the request.
    pub status: ModelRequestStatus,
    /// Unix timestamp in seconds when the request was sent.
    pub started_at: f64,
    /// Unix timestamp in seconds when the response completed.
    pub ended_at: f64,
    /// Model that served the request, when known.
    #[serde(default)]
    pub model: Option<String>,
    /// Model vendor that served the request, when known.
    #[serde(default)]
    pub provider_name: Option<String>,
    /// Instructions supplied apart from the conversation.
    #[serde(default)]
    pub system_instructions: Vec<ContentBlock>,
    /// Conversation sent with the request, in order.
    #[serde(default)]
    pub input_messages: Vec<TurnMessage>,
    /// Whether `system_instructions`/`input_messages` were observed from the request itself;
    /// defaults to `false`.
    #[serde(default)]
    pub input_observed: bool,
    /// The assistant message the request produced, when the provider could state it.
    #[serde(default)]
    pub output_message: Option<TurnMessage>,
    /// Tools offered to the model, each `{"name", "description", "parameters"}`; defaults to
    /// `null`.
    #[serde(default)]
    pub tool_definitions: JsonValue,
    /// Sampling parameters the request carried, under the `GenAI` names; defaults to an empty
    /// object.
    #[serde(default)]
    pub request_parameters: JsonObject,
    /// Provider id of the response, when it states one.
    #[serde(default)]
    pub response_id: Option<String>,
    /// Seconds from sending the request to its first streamed chunk, when measured.
    #[serde(default)]
    pub time_to_first_chunk: Option<f64>,
    /// Why generation stopped, in the provider's words; defaults to an empty list.
    #[serde(default)]
    pub finish_reasons: Vec<String>,
    /// Token usage of this request alone.
    #[serde(default)]
    pub usage: Option<TurnUsage>,
    /// What this request cost, when the provider states it.
    #[serde(default)]
    pub cost: Option<MonetaryAmount>,
    /// Normalized failure when `status` is not `completed`.
    #[serde(default)]
    pub error: Option<TurnError>,
    /// Namespaced provider diagnostics (timing, attempts, upstream ids); defaults to an empty
    /// object.
    #[serde(default)]
    pub data: JsonObject,
}

/// A high-level harness state transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct StateChangedEvent {
    /// The state the harness left.
    pub old: HarnessState,
    /// The state the harness entered.
    pub new: HarnessState,
}

/// A lifecycle transition of one turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct TurnLifecycleEvent {
    /// Which turn transition happened.
    pub kind: TurnEventKind,
    /// The terminal result; required for terminal kinds, forbidden otherwise.
    #[serde(default)]
    pub result: Option<TurnResult>,
}

/// An observational mirror of hook execution, never a control response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct HookObservedEvent {
    /// Whether the hook started or finished.
    pub phase: HookEventPhase,
    /// Stable name of the hook.
    pub hook_name: String,
    /// Per-invocation identifier of the hook execution.
    pub hook_id: String,
    /// Provider-specific hook details; defaults to an empty object.
    #[serde(default)]
    pub data: JsonObject,
}

/// A non-output diagnostic emitted by the harness implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DiagnosticEvent {
    /// Severity of the diagnostic.
    pub level: DiagnosticLevel,
    /// Human-readable diagnostic text.
    pub message: String,
    /// Provider-specific diagnostic details; defaults to an empty object.
    #[serde(default)]
    pub data: JsonObject,
}

/// Namespaced provider extension event preserved without core changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ProviderEvent {
    /// Namespace of the provider extension (for example `claude_code`).
    pub provider: String,
    /// Provider-defined event type within its namespace.
    pub event_type: String,
    /// Version of the provider-defined payload schema.
    pub schema_version: String,
    /// The raw provider payload; defaults to an empty object.
    #[serde(default)]
    pub payload: JsonObject,
}

/// Lossless shared event from a newer schema unknown to this consumer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UnknownEvent {
    /// The event type exactly as it arrived on the wire.
    pub event_type: String,
    /// The schema version exactly as it arrived on the wire.
    pub schema_version: String,
    /// The raw payload, preserved so re-encoding is lossless; defaults to an empty object.
    #[serde(default)]
    pub payload: JsonObject,
}

/// One payload of a harness event.
///
/// This enum is a Rust-side tagged union, not a wire shape: it deliberately derives neither
/// [`serde::Serialize`] nor [`serde::Deserialize`]. On the wire, `event_type` and
/// `schema_version` live on the envelope and the payload is nested under `payload`; an
/// internally tagged serde enum would emit a second tag and collide with
/// [`ProviderEvent::event_type`]. Use [`wire_event_type`](Self::wire_event_type) and
/// [`wire_schema_version`](Self::wire_schema_version) with the codec in
/// [`serialization`](crate::serialization).
#[derive(Debug, Clone, PartialEq, schemars::JsonSchema)]
pub enum HarnessEventPayload {
    /// An output content update.
    Output(OutputEvent),
    /// An item lifecycle update.
    ItemLifecycle(ItemLifecycleEvent),
    /// A normalized usage update.
    UsageUpdated(UsageUpdatedEvent),
    /// A completed physical model request.
    ModelRequest(ModelRequestEvent),
    /// A harness state transition.
    StateChanged(StateChangedEvent),
    /// A turn lifecycle transition.
    TurnLifecycle(TurnLifecycleEvent),
    /// An observed hook execution.
    HookObserved(HookObservedEvent),
    /// A non-output diagnostic.
    Diagnostic(DiagnosticEvent),
    /// A namespaced provider extension event.
    Provider(ProviderEvent),
    /// An event from a newer schema this crate does not understand.
    Unknown(UnknownEvent),
}

impl HarnessEventPayload {
    /// Returns the envelope `event_type` string for this payload.
    ///
    /// Every known variant has a fixed type; [`Unknown`](Self::Unknown) keeps the event type it
    /// arrived with, so a lossless re-encode preserves it.
    #[must_use]
    pub fn wire_event_type(&self) -> &str {
        match self {
            Self::Output(_) => "output",
            Self::ItemLifecycle(_) => "item_lifecycle",
            Self::UsageUpdated(_) => "usage_updated",
            Self::ModelRequest(_) => "model_request",
            Self::StateChanged(_) => "state_changed",
            Self::TurnLifecycle(_) => "turn_lifecycle",
            Self::HookObserved(_) => "hook_observed",
            Self::Diagnostic(_) => "diagnostic",
            Self::Provider(_) => "provider",
            Self::Unknown(payload) => &payload.event_type,
        }
    }

    /// Returns the envelope `schema_version` string for this payload.
    ///
    /// Known variants always use [`EVENT_WIRE_SCHEMA_VERSION`]; [`Unknown`](Self::Unknown) keeps
    /// the version it arrived with, which is how a newer-schema event re-encodes losslessly.
    #[must_use]
    pub fn wire_schema_version(&self) -> &str {
        match self {
            Self::Unknown(payload) => &payload.schema_version,
            Self::Output(_)
            | Self::ItemLifecycle(_)
            | Self::UsageUpdated(_)
            | Self::ModelRequest(_)
            | Self::StateChanged(_)
            | Self::TurnLifecycle(_)
            | Self::HookObserved(_)
            | Self::Diagnostic(_)
            | Self::Provider(_) => EVENT_WIRE_SCHEMA_VERSION,
        }
    }
}

impl Validate for HarnessEventPayload {
    /// Validates the contained payload with its own rules.
    ///
    /// # Errors
    ///
    /// The contained payload's [`HarnessError::Validation`] for the first violated invariant.
    fn validate(&self) -> HarnessResult<()> {
        match self {
            Self::Output(payload) => payload.validate(),
            Self::ItemLifecycle(payload) => payload.validate(),
            Self::UsageUpdated(payload) => payload.validate(),
            Self::ModelRequest(payload) => payload.validate(),
            Self::StateChanged(payload) => payload.validate(),
            Self::TurnLifecycle(payload) => payload.validate(),
            Self::HookObserved(payload) => payload.validate(),
            Self::Diagnostic(payload) => payload.validate(),
            Self::Provider(payload) => payload.validate(),
            Self::Unknown(payload) => payload.validate(),
        }
    }
}

impl From<OutputEvent> for HarnessEventPayload {
    fn from(payload: OutputEvent) -> Self {
        Self::Output(payload)
    }
}

impl From<ItemLifecycleEvent> for HarnessEventPayload {
    fn from(payload: ItemLifecycleEvent) -> Self {
        Self::ItemLifecycle(payload)
    }
}

impl From<UsageUpdatedEvent> for HarnessEventPayload {
    fn from(payload: UsageUpdatedEvent) -> Self {
        Self::UsageUpdated(payload)
    }
}

impl From<ModelRequestEvent> for HarnessEventPayload {
    fn from(payload: ModelRequestEvent) -> Self {
        Self::ModelRequest(payload)
    }
}

impl From<StateChangedEvent> for HarnessEventPayload {
    fn from(payload: StateChangedEvent) -> Self {
        Self::StateChanged(payload)
    }
}

impl From<TurnLifecycleEvent> for HarnessEventPayload {
    fn from(payload: TurnLifecycleEvent) -> Self {
        Self::TurnLifecycle(payload)
    }
}

impl From<HookObservedEvent> for HarnessEventPayload {
    fn from(payload: HookObservedEvent) -> Self {
        Self::HookObserved(payload)
    }
}

impl From<DiagnosticEvent> for HarnessEventPayload {
    fn from(payload: DiagnosticEvent) -> Self {
        Self::Diagnostic(payload)
    }
}

impl From<ProviderEvent> for HarnessEventPayload {
    fn from(payload: ProviderEvent) -> Self {
        Self::Provider(payload)
    }
}

impl From<UnknownEvent> for HarnessEventPayload {
    fn from(payload: UnknownEvent) -> Self {
        Self::Unknown(payload)
    }
}

/// Returns the only retention class a producer may use for `event`.
///
/// The rules are evaluated first match wins, exactly like the Python implementation:
///
/// 1. an [`OutputEvent`] with [`OutputOperation::Snapshot`] is `Coalescible` (a later snapshot
///    supersedes it);
/// 2. a [`UsageUpdatedEvent`] with [`UsageUpdateMode::Cumulative`] is `Coalescible` (a later
///    cumulative update supersedes it);
/// 3. a [`HookObservedEvent`] is `BestEffort` (it is observational only);
/// 4. a [`DiagnosticEvent`] at [`DiagnosticLevel::Debug`] or [`DiagnosticLevel::Info`] is
///    `BestEffort`;
/// 5. everything else — including `delta` outputs, `delta` usage, and higher-severity
///    diagnostics — is `Required`.
#[must_use]
pub fn event_retention(event: &HarnessEventPayload) -> EventRetention {
    match event {
        HarnessEventPayload::Output(payload) if payload.operation == OutputOperation::Snapshot => {
            EventRetention::Coalescible
        }
        HarnessEventPayload::UsageUpdated(payload)
            if payload.mode == UsageUpdateMode::Cumulative =>
        {
            EventRetention::Coalescible
        }
        HarnessEventPayload::HookObserved(_) => EventRetention::BestEffort,
        HarnessEventPayload::Diagnostic(payload)
            if matches!(
                payload.level,
                DiagnosticLevel::Debug | DiagnosticLevel::Info
            ) =>
        {
            EventRetention::BestEffort
        }
        _ => EventRetention::Required,
    }
}

impl Validate for OutputEvent {
    /// Requires a non-empty `output_id`.
    ///
    /// `content_index` is unsigned, which subsumes Python's
    /// `"output content_index must be non-negative"` check (a negative wire value is rejected by
    /// serde).
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"output_id must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("output_id", &self.output_id)
    }
}

impl Validate for ItemLifecycleEvent {
    /// Requires a non-empty `item_type`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"item_type must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("item_type", &self.item_type)
    }
}

impl Validate for UsageUpdatedEvent {
    /// Delegates to [`TurnUsage::validate`].
    ///
    /// # Errors
    ///
    /// The usage validation error, if any.
    fn validate(&self) -> HarnessResult<()> {
        self.usage.validate()
    }
}

impl Validate for ModelRequestEvent {
    /// Checks the request invariants, then the nested values.
    ///
    /// Order of checks (each failure reports the first violated rule):
    ///
    /// 1. `request_id` non-empty.
    /// 2. `started_at` then `ended_at` are finite, non-negative Unix timestamps.
    /// 3. `ended_at` does not precede `started_at`.
    /// 4. A `completed` request carries no `error`.
    /// 5. `time_to_first_chunk`, when present, is a finite non-negative duration.
    /// 6. `system_instructions`, `input_messages`, `output_message`, `usage`, `cost`, and `error`
    ///    are validated.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with the exact Python message for the first violated rule.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("model request_id", &self.request_id)?;
        require_finite_timestamp("model started_at", self.started_at)?;
        require_finite_timestamp("model ended_at", self.ended_at)?;
        if self.ended_at < self.started_at {
            return Err(HarnessError::validation(
                "model request ended_at must not precede started_at",
            ));
        }
        if self.status == ModelRequestStatus::Completed && self.error.is_some() {
            return Err(HarnessError::validation(
                "completed model request must not contain error",
            ));
        }
        if let Some(time_to_first_chunk) = self.time_to_first_chunk {
            if !time_to_first_chunk.is_finite() || time_to_first_chunk < 0.0 {
                return Err(HarnessError::validation(
                    "model request time_to_first_chunk must be a non-negative duration",
                ));
            }
        }

        for block in &self.system_instructions {
            block.validate()?;
        }
        for message in &self.input_messages {
            message.validate()?;
        }
        if let Some(message) = &self.output_message {
            message.validate()?;
        }
        if let Some(usage) = &self.usage {
            usage.validate()?;
        }
        if let Some(cost) = &self.cost {
            cost.validate()?;
        }
        if let Some(error) = &self.error {
            error.validate()?;
        }
        Ok(())
    }
}

impl Validate for StateChangedEvent {
    /// Always succeeds.
    ///
    /// Python declares no `__post_init__` rules for a state transition; the impl exists because
    /// [`HarnessEventPayload`] composes every payload type.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    fn validate(&self) -> HarnessResult<()> {
        Ok(())
    }
}

impl Validate for TurnLifecycleEvent {
    /// Checks the kind/result agreement, then validates the result.
    ///
    /// Order of checks (each failure reports the first violated rule):
    ///
    /// 1. A non-terminal kind (`started`, `paused`, `resumed`) forbids `result`.
    /// 2. A terminal kind (`finished`, `aborted`, `failed`) requires `result`.
    /// 3. The result status must match the kind: `finished` → `completed`, `aborted` →
    ///    `interrupted`, `failed` → `failed`.
    /// 4. The result is validated with [`TurnResult::validate`].
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with the exact Python message for the first violated rule.
    fn validate(&self) -> HarnessResult<()> {
        if !self.kind.is_terminal() {
            if self.result.is_some() {
                return Err(HarnessError::validation(format!(
                    "non-terminal {} turn event must not contain a result",
                    self.kind.as_str()
                )));
            }
            return Ok(());
        }

        let Some(result) = self.result.as_ref() else {
            return Err(HarnessError::validation(
                "terminal turn event requires a result",
            ));
        };
        let (expected_status, expected_value) = match self.kind {
            TurnEventKind::Finished => (TurnStatus::Completed, "completed"),
            TurnEventKind::Aborted => (TurnStatus::Interrupted, "interrupted"),
            TurnEventKind::Failed => (TurnStatus::Failed, "failed"),
            TurnEventKind::Started | TurnEventKind::Paused | TurnEventKind::Resumed => {
                return Ok(());
            }
        };
        if result.status != expected_status {
            return Err(HarnessError::validation(format!(
                "{} turn event requires {expected_value} result",
                self.kind.as_str()
            )));
        }
        result.validate()
    }
}

impl Validate for HookObservedEvent {
    /// Requires non-empty `hook_name` and `hook_id`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with
    /// `"hook observation name and id must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        if self.hook_name.is_empty() || self.hook_id.is_empty() {
            return Err(HarnessError::validation(
                "hook observation name and id must not be empty",
            ));
        }
        Ok(())
    }
}

impl Validate for DiagnosticEvent {
    /// Requires a non-empty `message`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"diagnostic message must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("diagnostic message", &self.message)
    }
}

impl Validate for ProviderEvent {
    /// Requires non-empty `provider`, `event_type`, and `schema_version`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with
    /// `"provider event namespace, type, and schema version must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        if self.provider.is_empty() || self.event_type.is_empty() || self.schema_version.is_empty()
        {
            return Err(HarnessError::validation(
                "provider event namespace, type, and schema version must not be empty",
            ));
        }
        Ok(())
    }
}

impl Validate for UnknownEvent {
    /// Requires non-empty `event_type` and `schema_version`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with
    /// `"unknown event type and schema version must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        if self.event_type.is_empty() || self.schema_version.is_empty() {
            return Err(HarnessError::validation(
                "unknown event type and schema version must not be empty",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::{HarnessEventPayload, ModelRequestEvent, OutputEvent, ProviderEvent, UnknownEvent};
    use crate::events::enums::{ModelRequestStatus, OutputKind};
    use crate::json::JsonObject;
    use crate::serialization::EVENT_WIRE_SCHEMA_VERSION;
    use crate::HarnessState;

    #[test]
    fn output_payload_serializes_every_key_in_declaration_order() {
        let event = OutputEvent {
            output_id: "out-1".to_owned(),
            kind: OutputKind::Text,
            content: serde_json::Value::String("hi".to_owned()),
            operation: crate::OutputOperation::Delta,
            channel: crate::OutputChannel::Reasoning,
            content_index: 2,
            content_type: None,
            data: JsonObject::new(),
        };
        let text = serde_json::to_string(&event).expect("serializes");
        assert_eq!(
            text,
            r#"{"output_id":"out-1","kind":"text","content":"hi","operation":"delta","channel":"reasoning","content_index":2,"content_type":null,"data":{}}"#
        );
    }

    #[test]
    fn model_request_defaults_keep_all_keys_present() {
        let event = ModelRequestEvent {
            request_id: "req-1".to_owned(),
            status: ModelRequestStatus::Completed,
            started_at: 1.0,
            ended_at: 1.5,
            model: None,
            provider_name: None,
            system_instructions: Vec::new(),
            input_messages: Vec::new(),
            input_observed: false,
            output_message: None,
            tool_definitions: serde_json::Value::Null,
            request_parameters: JsonObject::new(),
            response_id: None,
            time_to_first_chunk: None,
            finish_reasons: Vec::new(),
            usage: None,
            cost: None,
            error: None,
            data: JsonObject::new(),
        };
        let text = serde_json::to_string(&event).expect("serializes");
        assert_eq!(
            text,
            r#"{"request_id":"req-1","status":"completed","started_at":1.0,"ended_at":1.5,"model":null,"provider_name":null,"system_instructions":[],"input_messages":[],"input_observed":false,"output_message":null,"tool_definitions":null,"request_parameters":{},"response_id":null,"time_to_first_chunk":null,"finish_reasons":[],"usage":null,"cost":null,"error":null,"data":{}}"#
        );
    }

    #[test]
    fn wire_metadata_is_fixed_for_known_payloads_and_verbatim_for_unknown() {
        let provider: HarnessEventPayload = ProviderEvent {
            provider: "acme".to_owned(),
            event_type: "acme.progress".to_owned(),
            schema_version: "7".to_owned(),
            payload: JsonObject::new(),
        }
        .into();
        assert_eq!(provider.wire_event_type(), "provider");
        assert_eq!(
            provider.wire_schema_version(),
            EVENT_WIRE_SCHEMA_VERSION,
            "a provider extension still travels in the shared schema"
        );

        let state_changed: HarnessEventPayload = super::StateChangedEvent {
            old: HarnessState::Idle,
            new: HarnessState::Running,
        }
        .into();
        assert_eq!(state_changed.wire_event_type(), "state_changed");
        assert_eq!(
            state_changed.wire_schema_version(),
            EVENT_WIRE_SCHEMA_VERSION
        );

        let unknown: HarnessEventPayload = UnknownEvent {
            event_type: "future.event".to_owned(),
            schema_version: "2".to_owned(),
            payload: JsonObject::new(),
        }
        .into();
        assert_eq!(unknown.wire_event_type(), "future.event");
        assert_eq!(unknown.wire_schema_version(), "2");
    }

    #[test]
    fn payload_deserializes_unknown_keys_and_requires_required_ones() {
        let event: OutputEvent =
            serde_json::from_str(r#"{"output_id":"o","kind":"text","content":null,"future":1}"#)
                .expect("unknown keys are tolerated");
        assert_eq!(event.content, serde_json::Value::Null);
        assert_eq!(event.operation, crate::OutputOperation::Snapshot);

        let error = serde_json::from_str::<OutputEvent>(r#"{"kind":"text","content":null}"#)
            .expect_err("output_id is required");
        assert!(error.to_string().contains("output_id"));
    }
}
