//! Driver seam of the vendor CLI harnesses: the per-vendor wire protocol, plus the observation and
//! result constructors every adapter shares.
//!
//! A vendor CLI harness is split in two:
//!
//! - a [`VendorDriver`] implementation, which owns the vendor-specific protocol (spawn the CLI,
//!   encode a turn, decode its NDJSON stream, answer permission requests) and reports observations
//!   through a [`TurnContext`];
//! - the generic supervisor shell in [`harness`](super::harness), which owns *everything* the
//!   harness SPI requires and no vendor repeats: capability gating, turn ids, the serialized turn
//!   queue, lifecycle events, abort/pause/resume, checkpoints and teardown.
//!
//! The three shipped adapters (`claudecode`, `codex`, `dsh`) and the future `native_v2` provider
//! implement one trait each and get the whole [`HarnessProtocol`](openjiuwen_harness_protocol::HarnessProtocol)
//! surface from the shell.
//!
//! # Observations
//!
//! [`TurnContext`] is the only channel a driver uses to publish. It stamps every envelope from the
//! shared [`EnvelopeBuilder`] (session identity, provider session id, sequence, timestamp) and
//! applies the turn/item ids, so a driver never invents an envelope field. The `tool_*` helpers
//! emit the `tool_call` item lifecycle the IO adapter projects into `tool_call`/`tool_result`
//! chunks.
//!
//! # Results
//!
//! [`completed_turn`], [`interrupted_turn`], and [`failed_turn`] build [`TurnResult`] values that
//! pass [`TurnResult::validate`]: the terminal `error` is present only for `failed`, the
//! `termination` only for `interrupted`, and no result invents timestamps it did not measure.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    AbortMode, DiagnosticEvent, DiagnosticLevel, HarnessCheckpoint, HarnessContext, HarnessError,
    HarnessEvent, HarnessEventPayload, HarnessInput, HarnessResult, ItemEventKind,
    ItemLifecycleEvent, JsonObject, JsonValue, OutputEvent, ProviderEvent, TurnError, TurnMessage,
    TurnResult, TurnStatus, TurnTermination, TurnTerminationKind, TurnUsage, UsageUpdatedEvent,
    Validate,
};

use crate::json::to_json_object;

use super::events::{EnvelopeBuilder, EventBus};

/// Provider-native item type of a tool observation.
///
/// This is the one `item_type` the harness IO adapter projects into `tool_call`/`tool_result`
/// chunks, so every driver uses it for its tool lifecycle events.
pub const TOOL_CALL_ITEM_TYPE: &str = "tool_call";

/// Schema version stamped on [`ProviderEvent`] observations.
///
/// Provider extension payloads are vendor-defined; the shell forwards them verbatim under this
/// fixed version instead of asking each adapter for one.
pub const PROVIDER_SCHEMA_VERSION: &str = "1";

/// One accepted turn handed to a [`VendorDriver`].
///
/// The supervisor shell mints both identifiers; a driver echoes them on every observation so the
/// host can correlate an event with the receipt it received from
/// [`send`](openjiuwen_harness_protocol::HarnessProtocol::send).
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRequest {
    /// Turn identifier (`turn-<uuid>`), unique within the harness handle.
    pub turn_id: String,
    /// Message identifier (`message-<uuid>`) of the input that opened the turn.
    pub message_id: String,
    /// Provider-neutral input payload, forwarded verbatim from
    /// [`HarnessInput::content`](openjiuwen_harness_protocol::HarnessInput::content).
    pub content: JsonValue,
    /// Input metadata, forwarded verbatim from
    /// [`HarnessInput::metadata`](openjiuwen_harness_protocol::HarnessInput::metadata).
    pub metadata: JsonObject,
}

impl TurnRequest {
    /// Builds a turn request from its four parts.
    #[must_use]
    pub fn new(
        turn_id: impl Into<String>,
        message_id: impl Into<String>,
        content: JsonValue,
        metadata: JsonObject,
    ) -> Self {
        Self {
            turn_id: turn_id.into(),
            message_id: message_id.into(),
            content,
            metadata,
        }
    }

    /// Returns the textual projection of [`content`](Self::content).
    ///
    /// A JSON string is used verbatim, an object contributes its string `query`, and anything else
    /// uses its compact JSON rendering — the same rule the native harness applies.
    #[must_use]
    pub fn text(&self) -> String {
        match &self.content {
            JsonValue::String(text) => text.clone(),
            JsonValue::Object(object) => object
                .get("query")
                .and_then(JsonValue::as_str)
                .map_or_else(|| text_of_json(&self.content), str::to_owned),
            other => text_of_json(other),
        }
    }
}

/// Renders `value` as compact JSON, falling back to the empty string.
fn text_of_json(value: &JsonValue) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Observation channel handed to a [`VendorDriver`] for one start/stop cycle.
///
/// Every publisher stamps the shared envelope (host session, agent, provider session, sequence,
/// timestamp) and validates the event before it reaches the bus; a validation failure is logged
/// and the event dropped for that consumer, exactly like the native harness. Publishing while the
/// cycle is closed is a silent no-op: the event never fails a turn.
#[derive(Clone)]
pub struct TurnContext {
    /// Bus of the open cycle.
    bus: Arc<EventBus>,
    /// Envelope builder of the open cycle.
    envelope: Arc<EnvelopeBuilder>,
}

impl TurnContext {
    /// Wraps the bus and envelope builder of one cycle.
    #[must_use]
    pub fn new(bus: Arc<EventBus>, envelope: Arc<EnvelopeBuilder>) -> Self {
        Self { bus, envelope }
    }

    /// Publishes `payload`, tagged with `turn_id` when one is given.
    pub async fn emit(&self, payload: impl Into<HarnessEventPayload>, turn_id: Option<&str>) {
        self.emit_correlated(payload, turn_id, None).await;
    }

    /// Publishes `payload` with an optional envelope `turn_id` and `correlation_id`.
    ///
    /// The correlation id is the accepted message id, so a host can tie a lifecycle event back to
    /// the [`SendReceipt`](openjiuwen_harness_protocol::SendReceipt) that accepted the input.
    pub async fn emit_correlated(
        &self,
        payload: impl Into<HarnessEventPayload>,
        turn_id: Option<&str>,
        correlation_id: Option<&str>,
    ) {
        let mut event = match turn_id {
            Some(turn_id) => self.envelope.next_for_turn(payload, turn_id),
            None => self.envelope.next(payload),
        };
        if let Some(correlation_id) = correlation_id {
            event = event.with_correlation_id(correlation_id);
        }
        if let Err(error) = event.validate() {
            tracing::warn!(%error, "vendor harness event failed validation");
        }
        self.bus.emit(event).await;
    }

    /// Publishes the [`HarnessEvent`] `event` unchanged, validating it first.
    ///
    /// Adapters that already built an envelope (for example a re-encoded provider event) use this
    /// instead of the typed helpers.
    pub async fn emit_event(&self, event: HarnessEvent) {
        if let Err(error) = event.validate() {
            tracing::warn!(%error, "vendor harness event failed validation");
        }
        self.bus.emit(event).await;
    }

    /// Publishes one output content update of `turn_id`.
    pub async fn output(&self, turn_id: &str, output: OutputEvent) {
        self.emit(output, Some(turn_id)).await;
    }

    /// Publishes the `Started` item lifecycle of a tool call.
    ///
    /// The item data carries `tool_call_id` (when the vendor has one), `name` and `arguments`, which
    /// is what the harness IO adapter projects into a `tool_call` chunk.
    pub async fn tool_started(
        &self,
        turn_id: &str,
        item_id: &str,
        tool_call_id: Option<&str>,
        name: &str,
        arguments: JsonValue,
    ) {
        let mut data = JsonObject::new();
        insert_optional(&mut data, "tool_call_id", tool_call_id);
        data.insert("name".to_owned(), JsonValue::String(name.to_owned()));
        data.insert("arguments".to_owned(), arguments);
        let event = self.envelope.next_for_item(
            ItemLifecycleEvent {
                kind: ItemEventKind::Started,
                item_type: TOOL_CALL_ITEM_TYPE.to_owned(),
                data,
            },
            turn_id,
            item_id,
        );
        self.emit_event(event).await;
    }

    /// Publishes the `Completed` item lifecycle of a tool call.
    ///
    /// The item data carries `tool_call_id` (when the vendor has one), `tool_name`, `result` and,
    /// when the vendor rendered one, `rendered_result`.
    pub async fn tool_completed(
        &self,
        turn_id: &str,
        item_id: &str,
        tool_call_id: Option<&str>,
        name: &str,
        result: JsonValue,
        rendered_result: Option<String>,
    ) {
        let mut data = JsonObject::new();
        insert_optional(&mut data, "tool_call_id", tool_call_id);
        data.insert("tool_name".to_owned(), JsonValue::String(name.to_owned()));
        data.insert("result".to_owned(), result);
        if let Some(rendered) = rendered_result {
            data.insert("rendered_result".to_owned(), JsonValue::String(rendered));
        }
        let event = self.envelope.next_for_item(
            ItemLifecycleEvent {
                kind: ItemEventKind::Completed,
                item_type: TOOL_CALL_ITEM_TYPE.to_owned(),
                data,
            },
            turn_id,
            item_id,
        );
        self.emit_event(event).await;
    }

    /// Publishes a cumulative usage update of `turn_id`.
    pub async fn usage(&self, turn_id: &str, usage: TurnUsage) {
        self.emit(
            UsageUpdatedEvent {
                usage,
                mode: openjiuwen_harness_protocol::UsageUpdateMode::Cumulative,
            },
            Some(turn_id),
        )
        .await;
    }

    /// Publishes a namespaced provider extension event.
    ///
    /// `payload` is forwarded as-is when it is a JSON object and wrapped as `{"value": …}`
    /// otherwise; the envelope carries [`PROVIDER_SCHEMA_VERSION`].
    pub async fn provider_event(
        &self,
        turn_id: &str,
        provider: &str,
        event_type: &str,
        payload: JsonValue,
    ) {
        self.emit(
            ProviderEvent {
                provider: provider.to_owned(),
                event_type: event_type.to_owned(),
                schema_version: PROVIDER_SCHEMA_VERSION.to_owned(),
                payload: to_json_object(&payload),
            },
            Some(turn_id),
        )
        .await;
    }

    /// Publishes one diagnostic observation of `turn_id`.
    pub async fn diagnostic(
        &self,
        turn_id: &str,
        level: DiagnosticLevel,
        message: &str,
        data: JsonObject,
    ) {
        self.emit(
            DiagnosticEvent {
                level,
                message: message.to_owned(),
                data,
            },
            Some(turn_id),
        )
        .await;
    }
}

/// Inserts `value` under `key` when it is present.
fn insert_optional(data: &mut JsonObject, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        data.insert(key.to_owned(), JsonValue::String(value.to_owned()));
    }
}

/// Vendor-specific half of a CLI harness.
///
/// The generic supervisor shell ([`VendorHarness`](super::harness::VendorHarness)) implements
/// [`HarnessProtocol`](openjiuwen_harness_protocol::HarnessProtocol) on top of one driver: it
/// serializes turns, mints ids, gates capabilities, publishes the lifecycle events, and tears the
/// cycle down. A driver therefore only describes *one* provider session and *one* turn.
///
/// `label` is the human name used in error messages (`"Claude Code"`, `"Codex"`, `"DSH"`); the
/// default bodies of the optional commands write it into their
/// [`HarnessError::UnsupportedCapability`] message.
///
/// Every method takes `&self`: implementations synchronize their own session state internally so a
/// driver can be shared across the shell's tasks.
#[async_trait]
pub trait VendorDriver: Send + Sync {
    /// Human-readable provider name used in error messages, for example `"Claude Code"`.
    fn label(&self) -> &'static str;

    /// Starts the provider session and returns its provider-side session id, when the vendor has
    /// one.
    ///
    /// The shell calls this exactly once per successful
    /// [`start`](openjiuwen_harness_protocol::HarnessProtocol::start), before it publishes the
    /// `Terminated → Idle` transition. `context` is the host context that started the cycle
    /// (never to be logged: it may carry secrets) and `session` is the cycle's observation channel.
    ///
    /// # Errors
    ///
    /// Returns the vendor's transport/setup failure; the shell then closes the observation cycle,
    /// leaves the harness terminated, and reports the error unchanged.
    async fn open(
        &self,
        context: &HarnessContext,
        session: &TurnContext,
    ) -> HarnessResult<Option<String>>;

    /// Releases the provider session and unblocks a running turn.
    ///
    /// Called by [`stop`](openjiuwen_harness_protocol::HarnessProtocol::stop) under a ten-second
    /// timeout, and it must end every in-flight `run_turn` so the shell can join its supervisor.
    ///
    /// # Errors
    ///
    /// Returns the vendor's teardown failure; the shell still completes its own teardown and
    /// reports the error after settling in `Terminated`.
    async fn close(&self) -> HarnessResult<()>;

    /// Runs one accepted turn to its terminal result, publishing observations through `session`.
    ///
    /// # Errors
    ///
    /// Returns a transport failure the shell reports as a `Failed` turn. A *provider* failure
    /// (a failed model request, a refused tool) is an `Ok(`[`failed_turn`]`)` instead, so the turn
    /// carries its normalized [`TurnError`].
    async fn run_turn(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult>;

    /// Injects `content` into the running turn.
    ///
    /// The default body reports
    /// [`HarnessError::UnsupportedCapability`] naming [`label`](Self::label), because most vendor
    /// protocols have no in-turn input channel.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] by default, or the vendor's transport failure.
    async fn steer(
        &self,
        turn: &TurnRequest,
        content: &JsonValue,
        session: &TurnContext,
    ) -> HarnessResult<()> {
        let _ = (turn, content, session);
        Err(unsupported(self.label(), "steering"))
    }

    /// Interrupts the running turn in `mode`.
    ///
    /// The default body reports [`HarnessError::UnsupportedCapability`] naming
    /// [`label`](Self::label).
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] by default, or the vendor's transport failure.
    async fn interrupt(&self, mode: AbortMode, session: &TurnContext) -> HarnessResult<()> {
        let _ = (mode, session);
        Err(unsupported(self.label(), "aborting a turn"))
    }

    /// Pauses the running turn.
    ///
    /// The default body reports [`HarnessError::UnsupportedCapability`] naming
    /// [`label`](Self::label).
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] by default, or the vendor's transport failure.
    async fn pause(&self, session: &TurnContext) -> HarnessResult<()> {
        let _ = session;
        Err(unsupported(self.label(), "pausing"))
    }

    /// Resumes a paused turn, optionally with `query` as the continuing input.
    ///
    /// The default body reports [`HarnessError::UnsupportedCapability`] naming
    /// [`label`](Self::label).
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] by default, or the vendor's transport failure.
    async fn resume(
        &self,
        query: Option<&HarnessInput>,
        session: &TurnContext,
    ) -> HarnessResult<()> {
        let _ = (query, session);
        Err(unsupported(self.label(), "resuming"))
    }

    /// Returns the latest provider snapshot, or `None` when the vendor keeps no portable state.
    ///
    /// # Errors
    ///
    /// Returns the vendor's serialization failure; the default body reports `None`.
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        Ok(None)
    }
}

/// Builds the "does not support …" capability error of one driver.
fn unsupported(label: &str, capability: &str) -> HarnessError {
    HarnessError::unsupported_capability(format!(
        "the {label} harness does not support {capability}"
    ))
}

/// Builds the terminal result of a turn that reached a normal completion.
///
/// `final_output` is the convenience projection of the last assistant output, `messages` the
/// lossless normalized history, and `usage` the provider-reported counters. No timestamp is
/// invented: a driver that measured them fills `started_at`/`completed_at`/`duration_ms` itself.
#[must_use]
pub fn completed_turn(
    final_output: JsonValue,
    messages: Vec<TurnMessage>,
    usage: Option<TurnUsage>,
) -> TurnResult {
    TurnResult {
        final_output,
        messages,
        usage,
        ..bare_result(TurnStatus::Completed)
    }
}

/// Builds the terminal result of a turn that ended without reaching a normal completion.
///
/// The result carries a [`TurnTermination`] with `kind` and `message`, which is exactly the shape
/// `interrupted` requires and `completed`/`failed` forbid.
#[must_use]
pub fn interrupted_turn(kind: TurnTerminationKind, message: impl Into<String>) -> TurnResult {
    TurnResult {
        termination: Some(TurnTermination {
            kind,
            message: Some(message.into()),
            code: None,
            provider_data: JsonObject::new(),
        }),
        ..bare_result(TurnStatus::Interrupted)
    }
}

/// Builds the terminal result of a turn that failed with `error`.
#[must_use]
pub fn failed_turn(error: TurnError) -> TurnResult {
    TurnResult {
        error: Some(error),
        ..bare_result(TurnStatus::Failed)
    }
}

/// Builds a normalized [`TurnError`].
///
/// `category` is the coarse failure family; an empty `category` becomes `None` instead of an empty
/// string so consumers can distinguish "not stated" from "stated". `retryable` is always reported,
/// because a driver that knows the answer should not lose it.
#[must_use]
pub fn turn_error(
    message: impl Into<String>,
    code: impl Into<String>,
    category: &str,
    retryable: bool,
) -> TurnError {
    TurnError {
        message: message.into(),
        code: Some(code.into()),
        category: if category.is_empty() {
            None
        } else {
            Some(category.to_owned())
        },
        retryable: Some(retryable),
        provider_data: JsonObject::new(),
    }
}

/// Returns the empty result shell of `status` before a constructor fills its payload.
fn bare_result(status: TurnStatus) -> TurnResult {
    TurnResult {
        status,
        messages: Vec::new(),
        final_output: JsonValue::Null,
        structured_output: JsonValue::Null,
        stop_reason: None,
        termination: None,
        error: None,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: JsonObject::new(),
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        ContentBlock, HarnessEventPayload, ItemEventKind, JsonObject, JsonValue, MessageRole,
        OutputEvent, OutputKind, TurnError, TurnMessage, TurnStatus, TurnTerminationKind,
        TurnUsage, Validate,
    };
    use serde_json::json;

    use super::{
        completed_turn, failed_turn, interrupted_turn, turn_error, TurnContext, TurnRequest,
    };
    use crate::vendor_cli::{EnvelopeBuilder, EventBus};

    /// Builds a context over a fresh bus and envelope builder.
    fn context() -> (TurnContext, std::sync::Arc<EventBus>) {
        let bus = EventBus::new(16);
        let envelope = std::sync::Arc::new(EnvelopeBuilder::new("host", "agent"));
        (TurnContext::new(std::sync::Arc::clone(&bus), envelope), bus)
    }

    /// Drains the bus through its single-consumer cursor.
    async fn drain(
        bus: &std::sync::Arc<EventBus>,
    ) -> Vec<openjiuwen_harness_protocol::HarnessEvent> {
        bus.close();
        let mut cursor = bus.events().expect("no other consumer");
        let mut events = Vec::new();
        while let Some(event) = cursor.next_event().await.expect("bus is healthy") {
            events.push(event);
        }
        events
    }

    #[test]
    fn turn_request_text_projects_strings_queries_and_other_json() {
        let request = TurnRequest::new("t", "m", json!("hello"), JsonObject::new());
        assert_eq!(request.text(), "hello");
        let request = TurnRequest::new("t", "m", json!({"query": "ask"}), JsonObject::new());
        assert_eq!(request.text(), "ask");
        let request = TurnRequest::new("t", "m", json!({"other": 1}), JsonObject::new());
        assert_eq!(request.text(), r#"{"other":1}"#);
        let request = TurnRequest::new("t", "m", json!([1, 2]), JsonObject::new());
        assert_eq!(request.text(), "[1,2]");
    }

    #[tokio::test]
    async fn tool_events_carry_the_io_adapter_item_shape() {
        let (session, bus) = context();
        session
            .tool_started("t1", "item-1", Some("call-1"), "Bash", json!({"cmd": "ls"}))
            .await;
        session
            .tool_completed(
                "t1",
                "item-1",
                Some("call-1"),
                "Bash",
                json!("ok"),
                Some("ok".to_owned()),
            )
            .await;
        let events = drain(&bus).await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].turn_id.as_deref(), Some("t1"));
        assert_eq!(events[0].item_id.as_deref(), Some("item-1"));
        let HarnessEventPayload::ItemLifecycle(started) = &events[0].event else {
            panic!("expected an item lifecycle event");
        };
        assert_eq!(started.kind, ItemEventKind::Started);
        assert_eq!(started.item_type, "tool_call");
        assert_eq!(started.data["tool_call_id"], json!("call-1"));
        assert_eq!(started.data["name"], json!("Bash"));
        assert_eq!(started.data["arguments"], json!({"cmd": "ls"}));

        let HarnessEventPayload::ItemLifecycle(completed) = &events[1].event else {
            panic!("expected an item lifecycle event");
        };
        assert_eq!(completed.kind, ItemEventKind::Completed);
        assert_eq!(completed.data["tool_name"], json!("Bash"));
        assert_eq!(completed.data["result"], json!("ok"));
        assert_eq!(completed.data["rendered_result"], json!("ok"));
    }

    #[tokio::test]
    async fn tool_events_omit_absent_optional_fields() {
        let (session, bus) = context();
        session
            .tool_started("t1", "item-1", None, "Read", JsonValue::Null)
            .await;
        session
            .tool_completed("t1", "item-1", None, "Read", JsonValue::Null, None)
            .await;
        let events = drain(&bus).await;
        let HarnessEventPayload::ItemLifecycle(started) = &events[0].event else {
            panic!("expected an item lifecycle event");
        };
        assert!(!started.data.contains_key("tool_call_id"));
        let HarnessEventPayload::ItemLifecycle(completed) = &events[1].event else {
            panic!("expected an item lifecycle event");
        };
        assert!(!completed.data.contains_key("tool_call_id"));
        assert!(!completed.data.contains_key("rendered_result"));
    }

    #[tokio::test]
    async fn output_usage_and_diagnostic_events_are_typed() {
        let (session, bus) = context();
        session
            .output(
                "t1",
                OutputEvent {
                    output_id: "out-1".to_owned(),
                    kind: OutputKind::Text,
                    content: json!("hi"),
                    operation: openjiuwen_harness_protocol::OutputOperation::Snapshot,
                    channel: openjiuwen_harness_protocol::OutputChannel::Answer,
                    content_index: 0,
                    content_type: None,
                    data: JsonObject::new(),
                },
            )
            .await;
        session
            .usage(
                "t1",
                TurnUsage {
                    input_tokens: Some(1),
                    ..TurnUsage::default()
                },
            )
            .await;
        session
            .diagnostic(
                "t1",
                openjiuwen_harness_protocol::DiagnosticLevel::Info,
                "noted",
                JsonObject::new(),
            )
            .await;
        let events = drain(&bus).await;
        assert!(matches!(events[0].event, HarnessEventPayload::Output(_)));
        assert!(matches!(
            events[1].event,
            HarnessEventPayload::UsageUpdated(_)
        ));
        assert!(matches!(
            events[2].event,
            HarnessEventPayload::Diagnostic(_)
        ));
        assert!(events.iter().all(|event| event.validate().is_ok()));
    }

    #[tokio::test]
    async fn provider_events_keep_objects_and_wrap_other_payloads() {
        let (session, bus) = context();
        session
            .provider_event("t1", "claude_code", "progress", json!({"step": 1}))
            .await;
        session
            .provider_event("t1", "claude_code", "raw", json!("line"))
            .await;
        let events = drain(&bus).await;
        let HarnessEventPayload::Provider(first) = &events[0].event else {
            panic!("expected a provider event");
        };
        assert_eq!(first.provider, "claude_code");
        assert_eq!(first.event_type, "progress");
        assert_eq!(first.schema_version, "1");
        assert_eq!(first.payload["step"], json!(1));
        let HarnessEventPayload::Provider(second) = &events[1].event else {
            panic!("expected a provider event");
        };
        assert_eq!(second.payload["value"], json!("line"));
    }

    #[test]
    fn completed_turn_passes_validation() {
        let message = TurnMessage {
            message_id: "m1".to_owned(),
            role: MessageRole::Assistant,
            content: vec![ContentBlock {
                block_id: "b1".to_owned(),
                kind: "text".to_owned(),
                content: json!("hi"),
                data: JsonObject::new(),
            }],
            data: JsonObject::new(),
        };
        let result = completed_turn(json!("hi"), vec![message], None);
        assert_eq!(result.status, TurnStatus::Completed);
        assert!(result.error.is_none());
        assert!(result.termination.is_none());
        assert!(result.validate().is_ok());
    }

    #[test]
    fn interrupted_turn_passes_validation() {
        let result = interrupted_turn(TurnTerminationKind::UserAbort, "stopped");
        assert_eq!(result.status, TurnStatus::Interrupted);
        let termination = result.termination.as_ref().expect("termination");
        assert_eq!(termination.kind, TurnTerminationKind::UserAbort);
        assert_eq!(termination.message.as_deref(), Some("stopped"));
        assert!(result.error.is_none());
        assert!(result.validate().is_ok());
    }

    #[test]
    fn failed_turn_passes_validation() {
        let error = turn_error("boom", "e1", "protocol", true);
        assert_eq!(error.code.as_deref(), Some("e1"));
        assert_eq!(error.category.as_deref(), Some("protocol"));
        assert_eq!(error.retryable, Some(true));
        let result = failed_turn(error);
        assert_eq!(result.status, TurnStatus::Failed);
        assert!(result.termination.is_none());
        assert!(result.validate().is_ok());
    }

    #[test]
    fn an_empty_category_is_reported_as_absent() {
        let error = turn_error("boom", "e1", "", false);
        assert!(error.category.is_none());
        assert_eq!(error.retryable, Some(false));
    }

    #[test]
    fn result_constructors_do_not_invent_timestamps() {
        let failed = failed_turn(TurnError {
            message: "boom".to_owned(),
            code: None,
            category: None,
            retryable: None,
            provider_data: JsonObject::new(),
        });
        assert!(failed.started_at.is_none());
        assert!(failed.completed_at.is_none());
        assert!(failed.duration_ms.is_none());
        assert!(failed.provider_data.is_empty());
    }
}
