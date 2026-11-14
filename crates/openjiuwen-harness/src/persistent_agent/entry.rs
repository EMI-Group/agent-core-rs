//! Execution-journal entries for persistent agents.
//!
//! An execution journal is an append-only JSONL file whose lines are [`ExecutionEntry`] values:
//! a sequence number, an RFC 3339 timestamp and a flattened [`EntryPayload`]. The payload is
//! internally tagged by `kind`, so one line reads
//! `{"seq":3,"occurred_at":"…","kind":"tool_intent",…}`.
//!
//! The module also owns the redaction and summary helpers that keep tool arguments and results
//! bounded and credential-free before they reach the journal.
use chrono::DateTime;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::identity::ExecutionIdentity;
use super::util::{digest_hex, truncate_chars};

/// Schema version of a persisted execution entry.
pub const ENTRY_SCHEMA_VERSION: u32 = 1;

/// Maximum number of characters kept in a tool-argument or tool-result summary.
pub const SUMMARY_LIMIT: usize = 500;

/// Maximum number of characters kept in a pause or failure detail.
pub const DETAIL_LIMIT: usize = 4096;

/// Key fragments whose values are redacted before an entry is persisted.
const SENSITIVE_MARKERS: [&str; 6] = [
    "password",
    "secret",
    "token",
    "key",
    "credential",
    "authorization",
];

/// The kind of an [`ExecutionEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// The execution began with a goal.
    ExecutionStarted,
    /// A model call completed.
    ModelTurn,
    /// A non-read-only tool call is about to run.
    ToolIntent,
    /// A tool call completed, failed or has an unknown outcome.
    ToolCompleted,
    /// A non-empty fragment of the reply stream.
    ReplyDelta,
    /// The execution paused without a final outcome.
    ExecutionPaused,
    /// The execution reached a terminal outcome.
    ExecutionFinal,
    /// The model window was replaced by a new one.
    WindowSwitched,
    /// A free-form note recorded by a host.
    Note,
}

impl EntryKind {
    /// Returns the persisted snake-case spelling of the kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExecutionStarted => "execution_started",
            Self::ModelTurn => "model_turn",
            Self::ToolIntent => "tool_intent",
            Self::ToolCompleted => "tool_completed",
            Self::ReplyDelta => "reply_delta",
            Self::ExecutionPaused => "execution_paused",
            Self::ExecutionFinal => "execution_final",
            Self::WindowSwitched => "window_switched",
            Self::Note => "note",
        }
    }
}

/// The outcome of one tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    /// The tool completed successfully.
    Succeeded,
    /// The tool reported a failure.
    Failed,
    /// A side effect may have happened but its outcome could not be observed.
    Unknown,
}

/// Why an execution paused without a final outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PauseReason {
    /// The execution stopped to await the initiator's confirmation.
    AwaitingConfirmation,
    /// The initiator stopped the execution.
    Stopped,
    /// A budget or admission check denied the call.
    BudgetDenied,
    /// The context could not fit the execution.
    ContextUnfit,
    /// An internal error stopped the execution.
    Error,
}

/// The terminal outcome of an execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalStatus {
    /// The execution completed normally.
    Completed,
    /// The execution was intentionally not acted on.
    Ignored,
    /// The execution was cancelled.
    Cancelled,
    /// The execution failed.
    Failed,
}

/// Whether the token usage of a model call is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageStatus {
    /// Both counters were reported by the provider.
    Known,
    /// The provider did not report usage.
    Unknown,
}

/// Token usage reported for one model call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelUsage {
    /// Whether the counters are authoritative.
    pub status: UsageStatus,
    /// Input tokens, when known.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Output tokens, when known.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Why the usage is unknown, or any provider note.
    #[serde(default)]
    pub reason: Option<String>,
}

impl ModelUsage {
    /// Reports known input and output token counts.
    #[must_use]
    pub fn known(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            status: UsageStatus::Known,
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            reason: None,
        }
    }

    /// Reports that the provider did not expose usage, with a reason.
    #[must_use]
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            status: UsageStatus::Unknown,
            input_tokens: None,
            output_tokens: None,
            reason: Some(reason.into()),
        }
    }

    /// Returns the total token count, or `None` unless both counters are present.
    #[must_use]
    pub fn total_tokens(&self) -> Option<u64> {
        self.input_tokens
            .zip(self.output_tokens)
            .map(|(input, output)| input.saturating_add(output))
    }
}

/// The body of one execution entry, tagged by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EntryPayload {
    /// The execution began with a goal.
    ExecutionStarted {
        /// Goal the execution works towards.
        goal: String,
    },
    /// One completed model call.
    ModelTurn {
        /// Stable id of the model call.
        model_call_id: String,
        /// Response text produced by the call.
        response: String,
        /// Token usage of the call.
        usage: ModelUsage,
    },
    /// A non-read-only tool call is about to run.
    ToolIntent {
        /// Stable id of the tool call.
        tool_call_id: String,
        /// Name of the tool.
        tool_name: String,
        /// Redacted and bounded arguments.
        args_summary: String,
        /// Digest of the canonical arguments, used to match retries.
        params_digest: String,
        /// Opaque id of the intended side effect.
        operation_id: String,
        /// Optional explicit target reference supplied by the model.
        #[serde(default)]
        target_ref: Option<String>,
    },
    /// One tool call reached an outcome.
    ToolCompleted {
        /// Stable id of the tool call.
        tool_call_id: String,
        /// Name of the tool.
        tool_name: String,
        /// Redacted and bounded arguments.
        args_summary: String,
        /// Outcome of the call.
        status: ToolStatus,
        /// Redacted and bounded result.
        result_summary: String,
        /// Id of the matching intent, for calls that had one.
        #[serde(default)]
        operation_id: Option<String>,
        /// Why the call failed or its outcome is unknown.
        #[serde(default)]
        reason: Option<String>,
    },
    /// A non-empty fragment of the reply stream.
    ReplyDelta {
        /// Stream fragment; whitespace is preserved.
        text: String,
    },
    /// The execution paused without a final outcome.
    ExecutionPaused {
        /// Why the execution paused.
        reason: PauseReason,
        /// Human-readable detail, when the reason needs one.
        #[serde(default)]
        detail: Option<String>,
    },
    /// The execution reached a terminal outcome.
    ExecutionFinal {
        /// Terminal outcome.
        status: FinalStatus,
        /// Why the outcome is not a normal completion.
        #[serde(default)]
        reason: Option<String>,
    },
    /// The model window was replaced by a new one.
    WindowSwitched {
        /// Reference to the archive of the replaced window.
        archive_ref: String,
        /// Sequence number of the new window.
        window_seq: u64,
    },
    /// A free-form note recorded by a host.
    Note {
        /// Note text.
        text: String,
    },
}

impl EntryPayload {
    /// Returns the kind of this payload.
    #[must_use]
    pub const fn kind(&self) -> EntryKind {
        match self {
            Self::ExecutionStarted { .. } => EntryKind::ExecutionStarted,
            Self::ModelTurn { .. } => EntryKind::ModelTurn,
            Self::ToolIntent { .. } => EntryKind::ToolIntent,
            Self::ToolCompleted { .. } => EntryKind::ToolCompleted,
            Self::ReplyDelta { .. } => EntryKind::ReplyDelta,
            Self::ExecutionPaused { .. } => EntryKind::ExecutionPaused,
            Self::ExecutionFinal { .. } => EntryKind::ExecutionFinal,
            Self::WindowSwitched { .. } => EntryKind::WindowSwitched,
            Self::Note { .. } => EntryKind::Note,
        }
    }

    /// Builds an execution-start payload.
    #[must_use]
    pub fn execution_started(goal: impl Into<String>) -> Self {
        Self::ExecutionStarted { goal: goal.into() }
    }

    /// Builds a completed-model-turn payload.
    #[must_use]
    pub fn model_turn(
        model_call_id: impl Into<String>,
        response: impl Into<String>,
        usage: ModelUsage,
    ) -> Self {
        Self::ModelTurn {
            model_call_id: model_call_id.into(),
            response: response.into(),
            usage,
        }
    }

    /// Builds a tool-intent payload.
    ///
    /// Callers are expected to have already redacted and bounded `args_summary` through
    /// [`args_summary`] and to have derived `operation_id` through [`operation_id`].
    #[must_use]
    pub fn tool_intent(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        args_summary: impl Into<String>,
        params_digest: impl Into<String>,
        operation_id: impl Into<String>,
    ) -> Self {
        Self::ToolIntent {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            args_summary: args_summary.into(),
            params_digest: params_digest.into(),
            operation_id: operation_id.into(),
            target_ref: None,
        }
    }

    /// Builds a tool-outcome payload without an operation id or reason.
    #[must_use]
    pub fn tool_completed(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        status: ToolStatus,
        args_summary: impl Into<String>,
        result_summary: impl Into<String>,
    ) -> Self {
        Self::ToolCompleted {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            args_summary: args_summary.into(),
            status,
            result_summary: result_summary.into(),
            operation_id: None,
            reason: None,
        }
    }

    /// Returns this payload with an explicit target reference.
    #[must_use]
    pub fn with_target_ref(mut self, target_ref: impl Into<String>) -> Self {
        if let Self::ToolIntent {
            target_ref: slot, ..
        } = &mut self
        {
            *slot = Some(target_ref.into());
        }
        self
    }

    /// Returns this payload with the id of its matching intent.
    #[must_use]
    pub fn with_operation_id(mut self, operation_id: impl Into<String>) -> Self {
        if let Self::ToolCompleted {
            operation_id: slot, ..
        } = &mut self
        {
            *slot = Some(operation_id.into());
        }
        self
    }

    /// Builds a reply-delta payload.
    #[must_use]
    pub fn reply_delta(text: impl Into<String>) -> Self {
        Self::ReplyDelta { text: text.into() }
    }

    /// Builds a pause payload.
    #[must_use]
    pub fn execution_paused(reason: PauseReason, detail: Option<String>) -> Self {
        Self::ExecutionPaused { reason, detail }
    }

    /// Builds a terminal-outcome payload.
    #[must_use]
    pub fn execution_final(status: FinalStatus, reason: Option<String>) -> Self {
        Self::ExecutionFinal { status, reason }
    }

    /// Builds a window-switch payload.
    #[must_use]
    pub fn window_switched(archive_ref: impl Into<String>, window_seq: u64) -> Self {
        Self::WindowSwitched {
            archive_ref: archive_ref.into(),
            window_seq,
        }
    }

    /// Builds a free-form note payload.
    #[must_use]
    pub fn note(text: impl Into<String>) -> Self {
        Self::Note { text: text.into() }
    }
}

/// One line of an execution journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionEntry {
    /// Positive, monotonically increasing sequence number within the journal.
    pub seq: u64,
    /// RFC 3339 timestamp of the entry.
    pub occurred_at: String,
    /// Entry body, flattened into the same JSON object.
    #[serde(flatten)]
    pub payload: EntryPayload,
}

impl ExecutionEntry {
    /// Creates an entry.
    #[must_use]
    pub fn new(seq: u64, occurred_at: impl Into<String>, payload: EntryPayload) -> Self {
        Self {
            seq,
            occurred_at: occurred_at.into(),
            payload,
        }
    }

    /// Returns the entry's kind.
    #[must_use]
    pub const fn kind(&self) -> EntryKind {
        self.payload.kind()
    }

    /// Returns the SHA-256 of the entry's canonical JSON.
    ///
    /// The digest is stable across serialization key order, so it identifies the entry's content.
    /// A serialization failure never collapses entries onto the digest of `null`: the digest then
    /// falls back to the entry's stable identity fields, so two distinct entries still differ.
    #[must_use]
    pub fn digest(&self) -> String {
        self.digest_of(serde_json::to_value(self))
    }

    /// Digests `serialized`, or the entry's stable identity when it could not be encoded.
    fn digest_of(&self, serialized: serde_json::Result<Value>) -> String {
        match serialized {
            Ok(value) => digest_hex(&value),
            Err(error) => {
                tracing::warn!(
                    seq = self.seq,
                    kind = self.kind().as_str(),
                    error = %error,
                    "execution entry could not be serialized; digesting its stable identity instead"
                );
                digest_hex(&json!({
                    "fallback": "entry_serialization_failed",
                    "seq": self.seq,
                    "kind": self.kind().as_str(),
                    "occurred_at": self.occurred_at,
                }))
            }
        }
    }

    /// Validates the sequence number and the timestamp.
    ///
    /// # Errors
    /// `Error::Validation` when `seq` is zero or `occurred_at` is not an RFC 3339 timestamp.
    pub fn validate(&self) -> Result<()> {
        if self.seq == 0 {
            return Err(Error::validation(
                "execution entry sequence must be positive",
            ));
        }
        if DateTime::parse_from_rfc3339(&self.occurred_at).is_err() {
            return Err(Error::validation(format!(
                "occurred_at must be an RFC 3339 timestamp, got `{}`",
                self.occurred_at
            )));
        }
        Ok(())
    }
}

/// Returns whether `key` names a credential that must not be persisted.
fn is_sensitive_key(key: &str) -> bool {
    let lowered = key.to_lowercase();
    SENSITIVE_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

/// Replaces sensitive values recursively, leaving keys and shapes intact.
fn scrub(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let scrubbed = map
                .iter()
                .map(|(key, item)| {
                    let replacement = if is_sensitive_key(key) {
                        Value::String("***".to_string())
                    } else {
                        scrub(item)
                    };
                    (key.clone(), replacement)
                })
                .collect::<Map<String, Value>>();
            Value::Object(scrubbed)
        }
        Value::Array(items) => Value::Array(items.iter().map(scrub).collect()),
        other => other.clone(),
    }
}

/// Marker substituted when a redacted value cannot be rendered as JSON.
const UNSERIALIZABLE_MARKER: &str = "<unserializable>";

/// Renders a redacted value as compact JSON.
fn redacted_json(value: &Value) -> String {
    render_redacted(serde_json::to_string(&scrub(value)))
}

/// Returns the rendered value, or [`UNSERIALIZABLE_MARKER`] when it could not be encoded.
///
/// An empty string would be indistinguishable from a value that legitimately rendered as nothing,
/// so a failure is reported explicitly instead.
fn render_redacted(serialized: serde_json::Result<String>) -> String {
    match serialized {
        Ok(text) => text,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "a redacted persistent-agent value could not be serialized"
            );
            UNSERIALIZABLE_MARKER.to_string()
        }
    }
}

/// Returns `arguments` with values under sensitive key names replaced by `"***"`.
///
/// The input is parsed as JSON and re-serialized compactly; input that is not a JSON document
/// is returned unchanged.
#[must_use]
pub fn redact_arguments(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => redacted_json(&value),
        Err(_) => arguments.to_string(),
    }
}

/// Returns the bounded, redacted summary of tool arguments recorded in an intent.
///
/// An empty input becomes `"{}"`; invalid JSON becomes
/// `"Arguments omitted because they are not valid JSON"`; otherwise the redacted rendering is
/// truncated to [`SUMMARY_LIMIT`] characters.
#[must_use]
pub fn args_summary(arguments: &str) -> String {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return "{}".to_string();
    }
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        return "Arguments omitted because they are not valid JSON".to_string();
    };
    let summary = truncate_chars(&redacted_json(&value), SUMMARY_LIMIT);
    if summary.is_empty() {
        "{}".to_string()
    } else {
        summary
    }
}

/// Bounds a rendered summary, substituting `fallback` when nothing remains.
fn bounded_summary(rendered: &str, fallback: &str) -> String {
    let summary = truncate_chars(rendered, SUMMARY_LIMIT);
    if summary.is_empty() {
        fallback.to_string()
    } else {
        summary
    }
}

/// Returns the bounded, redacted summary of one tool result.
///
/// A JSON null becomes `"Tool returned no result"`. A JSON string that is not itself a JSON
/// document becomes `"Text result omitted (N characters)"`, so arbitrary transport text (which
/// may echo credentials) never reaches the journal. Everything else is redacted and truncated
/// to [`SUMMARY_LIMIT`] characters.
#[must_use]
pub fn result_summary(result: &Value) -> String {
    match result {
        Value::Null => "Tool returned no result".to_string(),
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(decoded) => {
                bounded_summary(&redacted_json(&decoded), "Tool returned an empty result")
            }
            Err(_) => format!("Text result omitted ({} characters)", text.chars().count()),
        },
        other => bounded_summary(&redacted_json(other), "Tool returned an empty result"),
    }
}

/// Returns the stable operation id of one tool call.
///
/// The id hashes the namespace, agent, session, request and tool-call ids, so it survives a
/// retry of the same call while differing for every other call.
#[must_use]
pub fn operation_id(identity: &ExecutionIdentity, tool_call_id: &str) -> String {
    let value = json!({
        "namespace_id": identity.namespace_id,
        "agent_id": identity.agent_id,
        "session_id": identity.session_id,
        "request_id": identity.request_id,
        "tool_call_id": tool_call_id,
    });
    format!("op_{}", digest_hex(&value))
}

/// Returns the digest of the canonical form of tool arguments.
///
/// # Errors
/// `Error::Validation("tool arguments could not be canonicalized")` when `arguments` is neither
/// empty nor a JSON document.
pub fn params_digest(arguments: &str) -> Result<String> {
    let trimmed = arguments.trim();
    let value = if trimmed.is_empty() {
        Value::Object(Map::new())
    } else {
        serde_json::from_str::<Value>(trimmed)
            .map_err(|_| Error::validation("tool arguments could not be canonicalized"))?
    };
    // Canonicalization itself does not fail for a parsed value; the helper is exercised here so
    // the digest definition stays in one place.
    Ok(digest_hex(&value))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::{json, Value};

    use super::{
        args_summary, operation_id, params_digest, redact_arguments, redacted_json,
        render_redacted, result_summary, EntryKind, EntryPayload, ExecutionEntry, FinalStatus,
        ModelUsage, PauseReason, ToolStatus, UsageStatus, SUMMARY_LIMIT, UNSERIALIZABLE_MARKER,
    };
    use crate::persistent_agent::identity::ExecutionIdentity;
    use crate::persistent_agent::util::digest_hex;

    /// A real `serde_json` error, standing in for a serialization failure.
    fn serialization_error() -> serde_json::Error {
        serde_json::from_str::<Value>("{").unwrap_err()
    }

    fn identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
    }

    fn all_payloads() -> Vec<EntryPayload> {
        vec![
            EntryPayload::execution_started("ship it"),
            EntryPayload::model_turn("mc_1", "hello", ModelUsage::known(3, 4)),
            EntryPayload::tool_intent("call_1", "bash", r#"{"cmd":"ls"}"#, "digest", "op_1"),
            EntryPayload::tool_completed(
                "call_1",
                "bash",
                ToolStatus::Succeeded,
                r#"{"cmd":"ls"}"#,
                "ok",
            )
            .with_operation_id("op_1"),
            EntryPayload::reply_delta("hi"),
            EntryPayload::execution_paused(
                PauseReason::AwaitingConfirmation,
                Some("confirm".to_string()),
            ),
            EntryPayload::execution_final(FinalStatus::Completed, None),
            EntryPayload::window_switched("archive:1", 2),
            EntryPayload::note("noted"),
        ]
    }

    #[test]
    fn flatten_renders_one_self_describing_object() {
        let entry = ExecutionEntry::new(
            3,
            "2023-11-14T22:13:20.000Z",
            EntryPayload::tool_intent("call", "bash", "{}", "digest", "op_1"),
        );
        let value = serde_json::to_value(&entry).unwrap();
        assert_eq!(value["seq"], 3);
        assert_eq!(value["occurred_at"], "2023-11-14T22:13:20.000Z");
        assert_eq!(value["kind"], "tool_intent");
        assert_eq!(value["tool_call_id"], "call");
        assert_eq!(value["tool_name"], "bash");
        assert_eq!(value["operation_id"], "op_1");
        assert!(value.get("payload").is_none());
    }

    #[test]
    fn deserialization_tolerates_unknown_keys() {
        let text = r#"{
            "seq": 1,
            "occurred_at": "2023-11-14T22:13:20.000Z",
            "kind": "note",
            "text": "hello",
            "future_field": {"nested": true}
        }"#;
        let entry: ExecutionEntry = serde_json::from_str(text).unwrap();
        assert_eq!(entry.kind(), EntryKind::Note);
        assert_eq!(entry.payload, EntryPayload::note("hello"));
        entry.validate().unwrap();
    }

    #[test]
    fn every_payload_variant_round_trips() {
        for (index, payload) in all_payloads().into_iter().enumerate() {
            let seq = u64::try_from(index).unwrap() + 1;
            let entry = ExecutionEntry::new(seq, "2023-11-14T22:13:20.000Z", payload.clone());
            let encoded = serde_json::to_string(&entry).unwrap();
            let decoded: ExecutionEntry = serde_json::from_str(&encoded).unwrap();
            assert_eq!(entry, decoded, "round trip failed for {encoded}");
            assert_eq!(decoded.payload, payload);
            decoded.validate().unwrap();
        }
    }

    #[test]
    fn digest_is_stable_and_ignores_key_order() {
        let first: ExecutionEntry = serde_json::from_str(
            r#"{"seq":1,"occurred_at":"2023-11-14T22:13:20.000Z","kind":"note","text":"hi"}"#,
        )
        .unwrap();
        let second: ExecutionEntry = serde_json::from_str(
            r#"{"kind":"note","text":"hi","seq":1,"occurred_at":"2023-11-14T22:13:20.000Z"}"#,
        )
        .unwrap();
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.digest().len(), 64);
        let changed = ExecutionEntry::new(1, "2023-11-14T22:13:20.000Z", EntryPayload::note("hI"));
        assert_ne!(first.digest(), changed.digest());
    }

    #[test]
    fn digest_falls_back_to_stable_identity_on_serialization_failure() {
        let first = ExecutionEntry::new(1, "2023-11-14T22:13:20.000Z", EntryPayload::note("same"));
        let second = ExecutionEntry::new(2, "2023-11-14T22:13:20.000Z", EntryPayload::note("same"));
        let other_kind = ExecutionEntry::new(
            1,
            "2023-11-14T22:13:20.000Z",
            EntryPayload::reply_delta("same"),
        );
        let first_digest = first.digest_of(Err(serialization_error()));
        let second_digest = second.digest_of(Err(serialization_error()));
        let kind_digest = other_kind.digest_of(Err(serialization_error()));
        assert_eq!(first_digest.len(), 64);
        assert_ne!(
            first_digest, second_digest,
            "different sequences must not collide"
        );
        assert_ne!(
            first_digest, kind_digest,
            "different kinds must not collide"
        );
        assert_ne!(
            first_digest,
            digest_hex(&Value::Null),
            "a failure must not digest as null"
        );
        assert_ne!(
            first_digest,
            first.digest(),
            "the fallback is a distinct digest"
        );
    }

    #[test]
    fn entry_validation_rejects_zero_seq_and_bad_timestamps() {
        let bad_seq = ExecutionEntry::new(0, "2023-11-14T22:13:20.000Z", EntryPayload::note("n"));
        let error = bad_seq.validate().unwrap_err();
        assert!(error.to_string().contains("must be positive"));

        let bad_time = ExecutionEntry::new(1, "yesterday", EntryPayload::note("n"));
        let error = bad_time.validate().unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error.to_string().contains("occurred_at"));

        let offset = ExecutionEntry::new(1, "2023-11-14T23:13:20+01:00", EntryPayload::note("n"));
        offset.validate().unwrap();
    }

    #[test]
    fn kinds_and_statuses_use_snake_case() {
        assert_eq!(
            serde_json::to_string(&EntryKind::ExecutionStarted).unwrap(),
            "\"execution_started\""
        );
        assert_eq!(
            serde_json::to_string(&EntryKind::ToolCompleted).unwrap(),
            "\"tool_completed\""
        );
        assert_eq!(EntryKind::ModelTurn.as_str(), "model_turn");
        assert_eq!(
            serde_json::to_string(&ToolStatus::Unknown).unwrap(),
            "\"unknown\""
        );
        assert_eq!(
            serde_json::to_string(&PauseReason::BudgetDenied).unwrap(),
            "\"budget_denied\""
        );
        assert_eq!(
            serde_json::to_string(&FinalStatus::Cancelled).unwrap(),
            "\"cancelled\""
        );
        assert_eq!(
            serde_json::to_string(&UsageStatus::Known).unwrap(),
            "\"known\""
        );
    }

    #[test]
    fn redaction_is_recursive_and_case_insensitive() {
        let arguments = r#"{
            "Password": "hunter2",
            "nested": {"API_TOKEN": "abc", "keep": 1},
            "list": [{"Secret": true}, {"ok": "fine"}],
            "authorization": "Bearer x",
            "credential_id": "c",
            "not_sensitive": "visible"
        }"#;
        let redacted: Value = serde_json::from_str(&redact_arguments(arguments)).unwrap();
        assert_eq!(redacted["Password"], "***");
        assert_eq!(redacted["nested"]["API_TOKEN"], "***");
        assert_eq!(redacted["nested"]["keep"], 1);
        assert_eq!(redacted["list"][0]["Secret"], "***");
        assert_eq!(redacted["list"][1]["ok"], "fine");
        assert_eq!(redacted["authorization"], "***");
        assert_eq!(redacted["credential_id"], "***");
        assert_eq!(redacted["not_sensitive"], "visible");
    }

    #[test]
    fn redaction_returns_non_json_unchanged() {
        assert_eq!(redact_arguments("not json at all"), "not json at all");
        assert_eq!(redact_arguments(""), "");
    }

    #[test]
    fn args_summary_handles_empty_invalid_and_truncated_input() {
        assert_eq!(args_summary(""), "{}");
        assert_eq!(args_summary("   "), "{}");
        assert_eq!(
            args_summary("{oops"),
            "Arguments omitted because they are not valid JSON"
        );
        assert_eq!(
            args_summary(r#"{"api_key":"secret"}"#),
            r#"{"api_key":"***"}"#
        );
        let long = format!(r#"{{"text":"{}"}}"#, "x".repeat(SUMMARY_LIMIT * 2));
        let summary = args_summary(&long);
        // `truncate_chars` keeps the limit and spends one character on the ellipsis.
        assert_eq!(summary.chars().count(), SUMMARY_LIMIT);
        assert!(summary.ends_with('…'));
        assert!(summary.starts_with(r#"{"text":"#));
    }

    #[test]
    fn unserializable_redactions_use_a_marker_instead_of_an_empty_string() {
        assert_eq!(
            render_redacted(Err(serialization_error())),
            UNSERIALIZABLE_MARKER
        );
        assert_ne!(render_redacted(Err(serialization_error())), "");
        assert_eq!(render_redacted(Ok(r#"{"a":1}"#.to_string())), r#"{"a":1}"#);
        assert_eq!(
            redacted_json(&json!({"password": "p", "ok": 1})),
            r#"{"password":"***","ok":1}"#
        );
    }

    #[test]
    fn result_summary_handles_null_text_and_objects() {
        assert_eq!(result_summary(&Value::Null), "Tool returned no result");
        assert_eq!(
            result_summary(&json!("hello")),
            "Text result omitted (5 characters)"
        );
        assert_eq!(
            result_summary(&json!("héllo")),
            "Text result omitted (5 characters)"
        );
        assert_eq!(
            result_summary(&json!(r#"{"token":"abc","ok":1}"#)),
            r#"{"token":"***","ok":1}"#
        );
        assert_eq!(
            result_summary(&json!({"password": "p", "nested": {"Authorization": "Bearer x"}})),
            r#"{"password":"***","nested":{"Authorization":"***"}}"#
        );
        assert_eq!(result_summary(&json!({})), "{}");
        assert_eq!(result_summary(&json!([])), "[]");
        let long = json!({"text": "x".repeat(SUMMARY_LIMIT * 2)});
        assert_eq!(result_summary(&long).chars().count(), SUMMARY_LIMIT);
        assert!(result_summary(&long).ends_with('…'));
    }

    #[test]
    fn operation_ids_are_stable_and_call_specific() {
        let identity = identity();
        let first = operation_id(&identity, "call_1");
        assert_eq!(first, operation_id(&identity, "call_1"));
        assert!(first.starts_with("op_"));
        assert_eq!(first.len(), 3 + 64);
        assert_ne!(first, operation_id(&identity, "call_2"));
        let other =
            ExecutionIdentity::new("ns2", "agent", "session", "request", "execution", "source")
                .unwrap();
        assert_ne!(first, operation_id(&other, "call_1"));
    }

    #[test]
    fn params_digest_uses_canonical_json_and_rejects_invalid_input() {
        assert_eq!(params_digest("").unwrap(), digest_hex(&json!({})));
        assert_eq!(
            params_digest(r#"{"b":2,"a":1}"#).unwrap(),
            params_digest(r#"{"a":1,"b":2}"#).unwrap()
        );
        let error = params_digest("{oops").unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error
            .to_string()
            .contains("tool arguments could not be canonicalized"));
    }

    #[test]
    fn usage_totals_require_both_counters() {
        let known = ModelUsage::known(3, 4);
        assert_eq!(known.total_tokens(), Some(7));
        let unknown = ModelUsage::unknown("provider omitted usage");
        assert_eq!(unknown.total_tokens(), None);
        assert_eq!(unknown.status, UsageStatus::Unknown);
        let partial = ModelUsage {
            status: UsageStatus::Known,
            input_tokens: Some(3),
            output_tokens: None,
            reason: None,
        };
        assert_eq!(partial.total_tokens(), None);
        let saturated = ModelUsage::known(u64::MAX, 1);
        assert_eq!(saturated.total_tokens(), Some(u64::MAX));
    }

    #[test]
    fn model_usage_tolerates_missing_optional_fields() {
        let usage: ModelUsage = serde_json::from_str(r#"{"status":"known"}"#).unwrap();
        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.output_tokens, None);
        assert_eq!(usage.reason, None);
    }
}
