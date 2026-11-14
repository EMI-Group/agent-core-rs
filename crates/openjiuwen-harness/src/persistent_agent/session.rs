//! Persisted persistent-agent session record and its file-backed store.
//!
//! The record is the state a Session keeps between executions: the model window, the
//! interruption that paused it, the Session's remembered confirmations, the `DeepAgent` state, the
//! current model-window identity and either one active checkpoint or one terminal ending.
//! Persisting Python's key names exactly keeps a record exchangeable with the Python harness.
//!
//! Rust adds three fields Python derives elsewhere — `capabilities`, `participation` and
//! `continuation` — and a flattened `extra` map that preserves unknown keys across a load/save
//! round-trip, so a record written by a newer writer survives this version untouched.
//!
//! All persistence is file-based under a caller-provided root; the store writes atomically and
//! never creates a record it could not load back.
// Size note: ~630 lines of implementation plus an inline test module; the tests stay here so the module contract and its verification live together.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::message::BaseMessage;
use openjiuwen_foundation::prompt::role::Role;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io;
use std::path::PathBuf;

use super::identity::{LinkRelation, SessionBinding};
use super::util::{is_valid_identifier, percent_encode_identifier, read_json, write_json_atomic};
use crate::workspace::Workspace;

/// Schema version of a persisted session record.
pub const SESSION_RECORD_VERSION: u32 = 1;

/// Result text recorded for a tool call whose result was never obtained.
///
/// The placeholder states neither success nor failure: a reader has to check the actual effect of
/// the call instead of retrying it or assuming it failed.
pub const PLACEHOLDER_TEXT: &str = "结果未取得，实际效果需核实";

/// Prefix of the synthetic user message carrying the answer a predecessor handed off.
pub const HANDOFF_TEXT_PREFIX: &str = "【实际交出的最终回答】\n";

/// Prefix of every message reporting that a stored session record cannot be used.
pub const RECORD_INVALID_PREFIX: &str = "session state record is invalid: ";

/// Outcome of a session that stopped to await the initiator's confirmation.
const AWAITING_CONFIRMATION: &str = "awaiting_confirmation";

/// Workspace node (or `<root_path>` fallback) holding persistent-agent state.
const PERSISTENT_AGENT_NODE: &str = "persistent_agent";

/// Directory under the persistent-agent node holding the per-session records.
const SESSIONS_DIR: &str = "sessions";

/// Name of the file holding one session record.
const STATE_FILE: &str = "state.json";

/// Builds the error reported for a stored record that cannot be used.
fn invalid_record(reason: impl fmt::Display) -> Error {
    Error::validation(format!("{RECORD_INVALID_PREFIX}{reason}"))
}

/// Returns whether `value` is a lowercase hex SHA-256 fingerprint.
fn is_spec_fingerprint(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// How a persisted session ended.
///
/// The ending is the session's terminal marker: it records who ended it, what the successor needs
/// in order to resume it, and which outputs the end record referenced.
///
/// `outcome` keeps Python's free-text string. This version knows `completed`, `ignored`,
/// `cancelled`, `failed`, `stopped`, `error` and `awaiting_confirmation`; an empty or unknown
/// outcome is preserved as-is for forward compatibility, but only the exact
/// `awaiting_confirmation` value marks a session as waiting (see
/// [`PersistentSessionRecord::waiting_scene`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordEnding {
    /// Lifecycle outcome, spelled as Python spells it.
    pub outcome: String,
    /// Request that produced this ending.
    pub request_id: String,
    /// Execution that produced this ending.
    pub execution_id: String,
    /// Spec the waiting confirmation was published with; mandatory while awaiting.
    pub spec_fingerprint: Option<String>,
    /// Where the confirmation reply must be delivered.
    pub reply_target: Option<String>,
    /// References to the outputs the end record referenced.
    pub output_refs: Vec<String>,
}

/// The safe boundary a record may resume from.
///
/// A `request_start` checkpoint has no completed iteration; every other kind has at least one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordCheckpoint {
    /// Request the checkpoint belongs to.
    pub request_id: String,
    /// Execution the checkpoint belongs to.
    pub execution_id: String,
    /// Number of completed `ReAct` iterations at the boundary.
    pub iteration_seq: u64,
    /// Which kind of boundary this is.
    pub kind: CheckpointKind,
    /// Last group message scanned before the boundary.
    pub last_group_message_id: Option<String>,
}

/// The kind of a [`RecordCheckpoint`].
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    /// A completed `ReAct` iteration.
    #[default]
    ReactIteration,
    /// The start of a request, before any iteration completed.
    RequestStart,
    /// A committed window rollover.
    Rollover,
}

impl CheckpointKind {
    /// Returns the persisted snake-case spelling of the kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReactIteration => "react_iteration",
            Self::RequestStart => "request_start",
            Self::Rollover => "rollover",
        }
    }
}

/// The latest committed transition from one archived window to the current one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordRollover {
    /// Identity of the window that was archived.
    pub old_window_id: String,
    /// Sequence number of the window that was archived.
    pub old_window_seq: u64,
    /// Identity of the window that replaced it.
    pub new_window_id: String,
    /// Sequence number of the window that replaced it.
    pub new_window_seq: u64,
    /// Reference to the stored archive of the old window.
    pub archive_ref: String,
    /// What triggered the rollover.
    pub trigger_kind: RolloverTrigger,
    /// Token usage accounting of the rollover.
    pub token_usage: BTreeMap<String, Value>,
}

/// What triggered a [`RecordRollover`].
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RolloverTrigger {
    /// The agent chose to start a new window.
    #[default]
    Agent,
    /// The window was full and a new window was started as a fallback.
    CapacityFallback,
    /// A new request started its own window.
    RequestEntry,
}

impl RolloverTrigger {
    /// Returns the persisted snake-case spelling of the trigger.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::CapacityFallback => "capacity_fallback",
            Self::RequestEntry => "request_entry",
        }
    }
}

/// The confirmation a waiting session is blocked on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaitingScene {
    /// Request that is waiting for the reply.
    pub request_id: String,
    /// Where the reply must be delivered; empty when the record names none.
    pub reply_target: String,
    /// Spec the confirmation was published with.
    pub spec_fingerprint: Option<String>,
}

/// The participation judgement recorded for the last group message.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticipationState {
    /// What the agent decided to do with the message.
    pub decision: ParticipationDecision,
    /// Why it decided that.
    pub reason: Option<String>,
    /// Id of the emitted decision event.
    pub event_id: Option<String>,
    /// RFC 3339 timestamp of the decision.
    pub decided_at: Option<String>,
}

/// What an agent decided about a group message.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ParticipationDecision {
    /// The message is not for this agent; stay silent.
    #[default]
    Ignore,
    /// Answer the message.
    Respond,
    /// Step into an exchange between others.
    Intervene,
}

impl ParticipationDecision {
    /// Returns the persisted snake-case spelling of the decision.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ignore => "ignore",
            Self::Respond => "respond",
            Self::Intervene => "intervene",
        }
    }
}

/// Where a continued execution picks up in the journal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContinuationCursor {
    /// Recovery reference naming the execution this one continues.
    pub recovery_ref: Option<String>,
    /// Execution id of the predecessor.
    pub predecessor_id: Option<String>,
    /// How this execution relates to its predecessor.
    pub relation: Option<LinkRelation>,
    /// Journal sequence number to resume after.
    pub resume_seq: u64,
    /// Last group message scanned by the predecessor.
    pub last_group_message_id: Option<String>,
}

/// The state of one persistent agent Session.
///
/// Python's `SessionStateRecord`: the model window plus either one active checkpoint or one
/// terminal ending. Field names are Python's, so a record round-trips through the same JSON.
/// The additive Rust fields ([`capabilities`](Self::capabilities),
/// [`participation`](Self::participation), [`continuation`](Self::continuation)) are documented
/// deviations; [`extra`](Self::extra) keeps every key this version does not know.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistentSessionRecord {
    /// Schema version of the record; always [`SESSION_RECORD_VERSION`] for a valid record.
    pub schema_version: u32,
    /// Agent that owns the Session; compared exactly, never normalized.
    pub agent_id: String,
    /// Session the record belongs to; compared exactly, never normalized.
    pub session_id: String,
    /// The persisted model window.
    pub messages: Vec<BaseMessage>,
    /// The interruption state that paused the Session, as opaque JSON.
    pub interruption: Option<Value>,
    /// Session-scoped remembered confirmations, keyed by the confirm key.
    pub auto_confirm: BTreeMap<String, bool>,
    /// The persisted `DeepAgent` state, as opaque JSON.
    pub deepagent_state: Option<Value>,
    /// Active boundary the next load resumes from.
    pub checkpoint: Option<RecordCheckpoint>,
    /// Last group message scanned by the Session.
    pub last_group_message_id: Option<String>,
    /// Identity of the current model window.
    pub current_window_id: Option<String>,
    /// Sequence number of the current model window.
    pub window_seq: Option<u64>,
    /// Latest committed window transition.
    pub last_rollover: Option<RecordRollover>,
    /// How the Session ended.
    pub ending: Option<RecordEnding>,
    /// Capability flags the Session was served with (additive Rust field).
    pub capabilities: BTreeSet<String>,
    /// Participation judgement recorded for the last group message (additive Rust field).
    pub participation: Option<ParticipationState>,
    /// Continuation cursor of the current execution (additive Rust field).
    pub continuation: Option<ContinuationCursor>,
    /// Keys this version does not know, preserved verbatim across a load/save round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl PersistentSessionRecord {
    /// Creates an empty draft record for `binding`.
    ///
    /// The draft has version [`SESSION_RECORD_VERSION`], no messages and neither a checkpoint nor
    /// an ending; it is valid and becomes a resumable record once [`Self::with_checkpoint`] or
    /// [`Self::with_ending`] is applied.
    ///
    /// # Errors
    /// `Error::Validation` when the binding's ids are not identifiers.
    pub fn new(binding: &SessionBinding) -> Result<Self> {
        let record = Self {
            schema_version: SESSION_RECORD_VERSION,
            agent_id: binding.agent_id.clone(),
            session_id: binding.session_id.clone(),
            ..Self::default()
        };
        record.validate()?;
        Ok(record)
    }

    /// Validates the whole record.
    ///
    /// # Errors
    /// `Error::Validation` prefixed with [`RECORD_INVALID_PREFIX`] naming the first rule that
    /// fails; the message never contains record content.
    pub fn validate(&self) -> Result<()> {
        match self.invalid_reason() {
            Some(reason) => Err(invalid_record(reason)),
            None => Ok(()),
        }
    }

    /// Returns the rule this record breaks, if any.
    fn invalid_reason(&self) -> Option<String> {
        if self.schema_version != SESSION_RECORD_VERSION {
            return Some(format!(
                "unsupported schema version {}",
                self.schema_version
            ));
        }
        if !is_valid_identifier(&self.agent_id) {
            return Some("agent_id must be an identifier".to_string());
        }
        if !is_valid_identifier(&self.session_id) {
            return Some("session_id must be an identifier".to_string());
        }
        if self.checkpoint.is_some() && self.ending.is_some() {
            return Some("checkpoint and ending must not both be present".to_string());
        }
        match (self.current_window_id.as_ref(), self.window_seq) {
            (Some(_), Some(0)) => {
                return Some("window sequence must be positive".to_string());
            }
            (Some(_), Some(_)) | (None, None) => {}
            _ => {
                return Some(
                    "current window identity and sequence must be present together".to_string(),
                );
            }
        }
        if let Some(rollover) = &self.last_rollover {
            if self.current_window_id.as_deref() != Some(rollover.new_window_id.as_str())
                || self.window_seq != Some(rollover.new_window_seq)
            {
                return Some("last rollover must point at the current window".to_string());
            }
            let consecutive = rollover.old_window_id != rollover.new_window_id
                && rollover.old_window_seq.checked_add(1) == Some(rollover.new_window_seq);
            if !consecutive {
                return Some("rollover must identify one consecutive successor window".to_string());
            }
        }
        if let Some(ending) = &self.ending {
            if ending.outcome == AWAITING_CONFIRMATION {
                match ending.spec_fingerprint.as_deref() {
                    None => {
                        return Some(
                            "a waiting confirmation must identify its published Spec".to_string(),
                        );
                    }
                    Some(fingerprint) if !is_spec_fingerprint(fingerprint) => {
                        return Some(
                            "a waiting confirmation must carry a 64-character lowercase hex Spec \
                             fingerprint"
                                .to_string(),
                        );
                    }
                    Some(_) => {}
                }
            }
        }
        if let Some(checkpoint) = &self.checkpoint {
            let request_start = checkpoint.kind == CheckpointKind::RequestStart;
            if (checkpoint.iteration_seq == 0) != request_start {
                return Some(if request_start {
                    "a request-start checkpoint has no completed iteration".to_string()
                } else {
                    "a checkpoint past the request start requires a completed iteration".to_string()
                });
            }
        }
        None
    }

    /// Returns the `(agent, session)` binding of this record.
    ///
    /// # Errors
    /// `Error::Validation` when either id is not an identifier.
    pub fn binding(&self) -> Result<SessionBinding> {
        SessionBinding::new(self.agent_id.clone(), self.session_id.clone())
    }

    /// Returns whether the Session is blocked on the initiator's confirmation.
    #[must_use]
    pub fn is_waiting(&self) -> bool {
        self.waiting_scene().is_some()
    }

    /// Returns the confirmation this Session is blocked on, when it is waiting.
    ///
    /// A scene exists only when the ending outcome is exactly `awaiting_confirmation`; any other
    /// (including an unknown) outcome is not a waiting scene. The reply target falls back to an
    /// empty string, and the fingerprint is copied from the ending.
    #[must_use]
    pub fn waiting_scene(&self) -> Option<WaitingScene> {
        let ending = self.ending.as_ref()?;
        if ending.outcome != AWAITING_CONFIRMATION {
            return None;
        }
        Some(WaitingScene {
            request_id: ending.request_id.clone(),
            reply_target: ending.reply_target.clone().unwrap_or_default(),
            spec_fingerprint: ending.spec_fingerprint.clone(),
        })
    }

    /// Returns this record with `ending` as its terminal marker, replacing any checkpoint.
    ///
    /// # Errors
    /// `Error::Validation` when the resulting record is invalid, for example an awaiting
    /// confirmation without a published Spec fingerprint.
    pub fn with_ending(mut self, ending: RecordEnding) -> Result<Self> {
        self.ending = Some(ending);
        self.checkpoint = None;
        self.validate()?;
        Ok(self)
    }

    /// Returns this record with `checkpoint` as its active boundary, replacing any ending.
    ///
    /// # Errors
    /// `Error::Validation` when the resulting record is invalid, for example a request-start
    /// checkpoint carrying completed iterations.
    pub fn with_checkpoint(mut self, checkpoint: RecordCheckpoint) -> Result<Self> {
        self.checkpoint = Some(checkpoint);
        self.ending = None;
        self.validate()?;
        Ok(self)
    }

    /// Appends a [`PLACEHOLDER_TEXT`] result for every tool call this window never answered.
    ///
    /// The method is pure and idempotent; see [`patch_stopped_calls`] for the exact rule.
    pub fn patch_stopped_calls(&mut self) {
        self.messages = patch_stopped_calls(&self.messages);
    }

    /// Settles a waiting scene `request_id` left behind, so the window can be appended to.
    ///
    /// A scene owned by another request is refused: the caller must not clear the other request's
    /// wait. Settling patches the stopped tool calls, clears the interruption state and drops the
    /// ending; the window/cursor fields and the remembered confirmations stay.
    ///
    /// # Errors
    /// `Error::Validation("another Request is waiting for a reply in this Session")` when the
    /// scene belongs to a different request.
    pub fn clear_stale_scene(&mut self, request_id: &str) -> Result<()> {
        let Some(scene) = self.waiting_scene() else {
            return Ok(());
        };
        if scene.request_id != request_id {
            return Err(Error::validation(
                "another Request is waiting for a reply in this Session",
            ));
        }
        self.patch_stopped_calls();
        self.interruption = None;
        self.ending = None;
        Ok(())
    }
}

/// Returns `messages` with a [`PLACEHOLDER_TEXT`] result for every unanswered tool call.
///
/// A call is answered by the contiguous run of tool messages right after its assistant message;
/// the placeholder for an unanswered call is appended after that run, so the result group stays
/// attached to its call. Calls without an id cannot be answered and are left alone. Tool messages
/// that follow neither an assistant message nor its group (stray results) pass through untouched.
/// Applying the function twice changes nothing.
#[must_use]
pub fn patch_stopped_calls(messages: &[BaseMessage]) -> Vec<BaseMessage> {
    let mut patched: Vec<BaseMessage> = Vec::with_capacity(messages.len());
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        patched.push(message.clone());
        index += 1;
        if message.role != Role::Assistant || message.tool_calls.is_empty() {
            continue;
        }
        let mut answered: BTreeSet<&str> = BTreeSet::new();
        while index < messages.len() && messages[index].role == Role::Tool {
            if let Some(call_id) = messages[index].tool_call_id.as_deref() {
                answered.insert(call_id);
            }
            patched.push(messages[index].clone());
            index += 1;
        }
        for call in &message.tool_calls {
            let Some(call_id) = call.id.as_deref() else {
                continue;
            };
            if answered.insert(call_id) {
                patched.push(BaseMessage::tool(call_id, PLACEHOLDER_TEXT));
            }
        }
    }
    patched
}

/// Storage of [`PersistentSessionRecord`]s, one per `(agent, session)` binding.
///
/// Implementations must be `Send + Sync`: a caller keeps one store for the whole process and
/// awaits its methods from unrelated tasks.
#[async_trait]
pub trait SessionStateStore: Send + Sync {
    /// Loads the record of `binding`; a binding without a record yields `Ok(None)`.
    ///
    /// # Errors
    /// An implementation-defined error when the record cannot be read, or `Error::Validation`
    /// when the stored record is not usable.
    async fn load(&self, binding: &SessionBinding) -> Result<Option<PersistentSessionRecord>>;

    /// Validates and atomically stores `record`.
    ///
    /// # Errors
    /// `Error::Validation` when the record is invalid, or an implementation-defined error when it
    /// cannot be written.
    async fn save(&self, record: &PersistentSessionRecord) -> Result<()>;

    /// Deletes the record of `binding`, reporting whether one existed.
    ///
    /// # Errors
    /// An implementation-defined error when the record cannot be removed.
    async fn delete(&self, binding: &SessionBinding) -> Result<bool>;
}

/// A [`SessionStateStore`] keeping one JSON file per binding.
///
/// The layout is `<root>/<percent-encoded agent>/<percent-encoded session>/state.json`, which is
/// exactly Python's encoded session key with `/` as the separator, so the encoding keeps any id
/// (colons, slashes, non-ASCII) inside a single path component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSessionStore {
    root: PathBuf,
}

impl FileSessionStore {
    /// Builds a store rooted at `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Builds a store under the workspace's `persistent_agent` node (`sessions/` inside it).
    ///
    /// A workspace whose schema has no `persistent_agent` node falls back to
    /// `<root_path>/persistent_agent/sessions`, the same rule the goal and cron stores use.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Self {
        let node = workspace
            .get_node_path(PERSISTENT_AGENT_NODE)
            .unwrap_or_else(|| workspace.root_path.join(PERSISTENT_AGENT_NODE));
        Self::new(node.join(SESSIONS_DIR))
    }

    /// Returns the file holding the record of `binding`.
    #[must_use]
    pub fn path_for(&self, binding: &SessionBinding) -> PathBuf {
        self.root
            .join(percent_encode_identifier(&binding.agent_id))
            .join(percent_encode_identifier(&binding.session_id))
            .join(STATE_FILE)
    }
}

#[async_trait]
impl SessionStateStore for FileSessionStore {
    async fn load(&self, binding: &SessionBinding) -> Result<Option<PersistentSessionRecord>> {
        let path = self.path_for(binding);
        let record = match read_json::<PersistentSessionRecord>(&path).await {
            Ok(Some(record)) => record,
            Ok(None) => return Ok(None),
            Err(Error::Validation(detail)) => return Err(invalid_record(detail)),
            Err(error) => return Err(error),
        };
        if let Some(reason) = record.invalid_reason() {
            return Err(invalid_record(format!(
                "{reason} (record at {})",
                path.display()
            )));
        }
        Ok(Some(record))
    }

    async fn save(&self, record: &PersistentSessionRecord) -> Result<()> {
        record.validate()?;
        let path = self.path_for(&record.binding()?);
        write_json_atomic(&path, record).await
    }

    async fn delete(&self, binding: &SessionBinding) -> Result<bool> {
        let path = self.path_for(binding);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(Error::session(format!(
                "failed to delete {}: {error}",
                path.display()
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    use openjiuwen_foundation::prompt::message::BaseMessage;
    use openjiuwen_foundation::prompt::role::Role;
    use openjiuwen_foundation::prompt::tool_call::ToolCall;
    use serde_json::json;

    use super::{
        patch_stopped_calls, CheckpointKind, ContinuationCursor, FileSessionStore,
        ParticipationDecision, ParticipationState, PersistentSessionRecord, RecordCheckpoint,
        RecordEnding, RecordRollover, RolloverTrigger, SessionStateStore, HANDOFF_TEXT_PREFIX,
        PLACEHOLDER_TEXT, RECORD_INVALID_PREFIX, SESSION_RECORD_VERSION,
    };
    use crate::persistent_agent::identity::{LinkRelation, SessionBinding};
    use crate::persistent_agent::util::write_json_atomic;
    use crate::workspace::Workspace;

    fn binding() -> SessionBinding {
        SessionBinding::new("agent", "s/1").unwrap()
    }

    fn fingerprint() -> String {
        "ab".repeat(32)
    }

    fn awaiting_ending(request_id: &str) -> RecordEnding {
        RecordEnding {
            outcome: "awaiting_confirmation".to_string(),
            request_id: request_id.to_string(),
            execution_id: "exec".to_string(),
            spec_fingerprint: Some(fingerprint()),
            reply_target: Some("group:1".to_string()),
            output_refs: vec!["file:1".to_string()],
        }
    }

    fn rollover() -> RecordRollover {
        RecordRollover {
            old_window_id: "w1".to_string(),
            old_window_seq: 1,
            new_window_id: "w2".to_string(),
            new_window_seq: 2,
            archive_ref: "archive:1".to_string(),
            trigger_kind: RolloverTrigger::CapacityFallback,
            token_usage: BTreeMap::from([("input_tokens".to_string(), json!(12))]),
        }
    }

    fn window_messages() -> Vec<BaseMessage> {
        vec![
            BaseMessage::user("hi"),
            BaseMessage::assistant("").with_tool_calls(vec![
                ToolCall::new("read_file").with_id("c1"),
                ToolCall::new("write_file").with_id("c2"),
            ]),
            BaseMessage::tool("c1", "ok"),
        ]
    }

    /// A record with every field populated and a checkpoint.
    fn full_record() -> PersistentSessionRecord {
        let mut record = PersistentSessionRecord::new(&binding()).unwrap();
        record.messages = window_messages();
        record.interruption = Some(json!({"iteration": 2, "original_query": "hi"}));
        record.auto_confirm.insert("bash".to_string(), true);
        record.deepagent_state = Some(json!({"iteration": 2, "todos": []}));
        record.checkpoint = Some(RecordCheckpoint {
            request_id: "req".to_string(),
            execution_id: "exec".to_string(),
            iteration_seq: 2,
            kind: CheckpointKind::ReactIteration,
            last_group_message_id: Some("g1".to_string()),
        });
        record.last_group_message_id = Some("g1".to_string());
        record.current_window_id = Some("w2".to_string());
        record.window_seq = Some(2);
        record.last_rollover = Some(rollover());
        record.capabilities = BTreeSet::from(["checkpoint".to_string(), "steer".to_string()]);
        record.participation = Some(ParticipationState {
            decision: ParticipationDecision::Respond,
            reason: Some("asked".to_string()),
            event_id: Some("evt_1".to_string()),
            decided_at: Some("2026-01-01T00:00:00.000Z".to_string()),
        });
        record.continuation = Some(ContinuationCursor {
            recovery_ref: Some("exec0".to_string()),
            predecessor_id: Some("exec0".to_string()),
            relation: Some(LinkRelation::Continuation),
            resume_seq: 7,
            last_group_message_id: Some("g0".to_string()),
        });
        record.validate().unwrap();
        record
    }

    #[test]
    fn new_builds_a_valid_draft_record() {
        let record = PersistentSessionRecord::new(&binding()).unwrap();
        assert_eq!(record.schema_version, SESSION_RECORD_VERSION);
        assert_eq!(record.agent_id, "agent");
        assert_eq!(record.session_id, "s/1");
        assert_eq!(record.messages, Vec::new());
        assert!(record.checkpoint.is_none());
        assert!(record.ending.is_none());
        assert!(record.capabilities.is_empty());
        assert!(record.extra.is_empty());
        assert!(!record.is_waiting());
        assert_eq!(record.binding().unwrap(), binding());
        record.validate().unwrap();
    }

    #[test]
    fn python_key_names_appear_in_the_serialized_record() {
        let mut value = serde_json::to_value(full_record()).unwrap();
        let object = value.as_object_mut().unwrap();
        for key in [
            "schema_version",
            "agent_id",
            "session_id",
            "messages",
            "interruption",
            "auto_confirm",
            "deepagent_state",
            "checkpoint",
            "last_group_message_id",
            "current_window_id",
            "window_seq",
            "last_rollover",
            "ending",
            "capabilities",
            "participation",
            "continuation",
        ] {
            assert!(object.contains_key(key), "missing key {key}");
        }
        assert_eq!(object["schema_version"], json!(1));
        assert_eq!(object["window_seq"], json!(2));
        assert_eq!(object["last_group_message_id"], json!("g1"));

        let ending_value = serde_json::to_value(awaiting_ending("req")).unwrap();
        assert_eq!(ending_value["outcome"], json!("awaiting_confirmation"));
        assert_eq!(ending_value["output_refs"], json!(["file:1"]));
        assert_eq!(
            serde_json::to_value(rollover()).unwrap()["trigger_kind"],
            json!("capacity_fallback")
        );
        assert_eq!(
            serde_json::to_value(RecordCheckpoint::default()).unwrap()["kind"],
            json!("react_iteration")
        );
        assert_eq!(
            serde_json::to_value(ParticipationState::default()).unwrap()["decision"],
            json!("ignore")
        );
    }

    #[test]
    fn nested_enums_use_snake_case_and_defaults() {
        let checkpoint: RecordCheckpoint =
            serde_json::from_str(r#"{"kind": "request_start"}"#).unwrap();
        assert_eq!(checkpoint.kind, CheckpointKind::RequestStart);
        assert_eq!(checkpoint.kind.as_str(), "request_start");
        assert_eq!(CheckpointKind::Rollover.as_str(), "rollover");
        assert_eq!(CheckpointKind::ReactIteration.as_str(), "react_iteration");
        assert_eq!(RolloverTrigger::Agent.as_str(), "agent");
        assert_eq!(RolloverTrigger::RequestEntry.as_str(), "request_entry");
        assert_eq!(ParticipationDecision::Intervene.as_str(), "intervene");
        assert_eq!(
            serde_json::to_string(&RolloverTrigger::RequestEntry).unwrap(),
            "\"request_entry\""
        );
        assert_eq!(
            serde_json::to_string(&LinkRelation::ConfirmationResume).unwrap(),
            "\"confirmation_resume\""
        );
        let ending: RecordEnding = serde_json::from_str(r#"{"outcome": "completed"}"#).unwrap();
        assert_eq!(ending.output_refs, Vec::<String>::new());
        assert!(ending.spec_fingerprint.is_none());
        let cursor: ContinuationCursor = serde_json::from_str("{}").unwrap();
        assert_eq!(cursor, ContinuationCursor::default());
    }

    #[test]
    fn waiting_scene_needs_the_awaiting_outcome() {
        let mut record = full_record();
        record.checkpoint = None;
        record.ending = Some(awaiting_ending("req"));
        record.validate().unwrap();
        assert!(record.is_waiting());
        let scene = record.waiting_scene().unwrap();
        assert_eq!(scene.request_id, "req");
        assert_eq!(scene.reply_target, "group:1");
        assert_eq!(scene.spec_fingerprint, Some(fingerprint()));

        record.ending.as_mut().unwrap().reply_target = None;
        assert_eq!(record.waiting_scene().unwrap().reply_target, "");

        record.ending.as_mut().unwrap().outcome = "completed".to_string();
        assert!(!record.is_waiting());
        assert!(record.waiting_scene().is_none());

        record.ending.as_mut().unwrap().outcome = "future_outcome".to_string();
        assert!(!record.is_waiting());
        record.validate().unwrap();
    }

    #[test]
    fn validate_rejects_lifecycle_conflicts() {
        let mut both = full_record();
        both.ending = Some(RecordEnding {
            outcome: "completed".to_string(),
            ..RecordEnding::default()
        });
        let error = both.validate().unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error
            .to_string()
            .contains("checkpoint and ending must not both be present"));
        assert!(error
            .to_string()
            .contains("session state record is invalid: "));

        let mut version = full_record();
        version.schema_version = 2;
        assert!(version
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unsupported schema version 2"));

        let mut agent = full_record();
        agent.agent_id = "bad agent".to_string();
        assert!(agent
            .validate()
            .unwrap_err()
            .to_string()
            .contains("agent_id must be an identifier"));

        let mut session = full_record();
        session.session_id = String::new();
        assert!(session
            .validate()
            .unwrap_err()
            .to_string()
            .contains("session_id must be an identifier"));
    }

    #[test]
    fn validate_rejects_window_identity_mismatches() {
        let mut id_only = full_record();
        id_only.window_seq = None;
        id_only.last_rollover = None;
        assert!(id_only
            .validate()
            .unwrap_err()
            .to_string()
            .contains("current window identity and sequence must be present together"));

        let mut seq_only = full_record();
        seq_only.current_window_id = None;
        seq_only.last_rollover = None;
        assert!(seq_only
            .validate()
            .unwrap_err()
            .to_string()
            .contains("current window identity and sequence must be present together"));

        let mut zero = full_record();
        zero.window_seq = Some(0);
        assert!(zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("window sequence must be positive"));

        let mut stale = full_record();
        stale.current_window_id = Some("w3".to_string());
        assert!(stale
            .validate()
            .unwrap_err()
            .to_string()
            .contains("last rollover must point at the current window"));

        let mut skipped = full_record();
        skipped.last_rollover = Some(RecordRollover {
            new_window_seq: 3,
            ..rollover()
        });
        skipped.window_seq = Some(3);
        assert!(skipped
            .validate()
            .unwrap_err()
            .to_string()
            .contains("rollover must identify one consecutive successor window"));

        let mut same_id = full_record();
        same_id.last_rollover = Some(RecordRollover {
            new_window_id: "w1".to_string(),
            ..rollover()
        });
        same_id.current_window_id = Some("w1".to_string());
        assert!(same_id
            .validate()
            .unwrap_err()
            .to_string()
            .contains("rollover must identify one consecutive successor window"));

        let mut no_window = full_record();
        no_window.current_window_id = None;
        no_window.window_seq = None;
        assert!(no_window
            .validate()
            .unwrap_err()
            .to_string()
            .contains("last rollover must point at the current window"));
    }

    #[test]
    fn validate_requires_a_fingerprint_only_while_awaiting() {
        let mut missing = full_record();
        missing.checkpoint = None;
        missing.ending = Some(RecordEnding {
            outcome: "awaiting_confirmation".to_string(),
            request_id: "req".to_string(),
            execution_id: "exec".to_string(),
            ..RecordEnding::default()
        });
        assert!(missing
            .validate()
            .unwrap_err()
            .to_string()
            .contains("a waiting confirmation must identify its published Spec"));

        let mut short = missing.clone();
        short.ending.as_mut().unwrap().spec_fingerprint = Some("abc".to_string());
        assert!(short
            .validate()
            .unwrap_err()
            .to_string()
            .contains("64-character lowercase hex Spec fingerprint"));

        let mut uppercase = missing.clone();
        uppercase.ending.as_mut().unwrap().spec_fingerprint = Some("AB".repeat(32));
        assert!(uppercase.validate().is_err());

        let mut completed = missing.clone();
        let ending = completed.ending.as_mut().unwrap();
        ending.outcome = "completed".to_string();
        ending.spec_fingerprint = None;
        completed.validate().unwrap();

        let mut empty = missing.clone();
        empty.ending.as_mut().unwrap().outcome = String::new();
        empty.ending.as_mut().unwrap().spec_fingerprint = None;
        empty.validate().unwrap();

        let mut unknown = missing;
        unknown.ending.as_mut().unwrap().outcome = "paused_for_review".to_string();
        unknown.ending.as_mut().unwrap().spec_fingerprint = None;
        unknown.validate().unwrap();
    }

    #[test]
    fn validate_matches_checkpoint_kind_and_iteration() {
        let mut start_with_iteration = full_record();
        start_with_iteration.checkpoint = Some(RecordCheckpoint {
            kind: CheckpointKind::RequestStart,
            iteration_seq: 3,
            ..RecordCheckpoint::default()
        });
        assert!(start_with_iteration
            .validate()
            .unwrap_err()
            .to_string()
            .contains("a request-start checkpoint has no completed iteration"));

        let mut iteration_zero = full_record();
        iteration_zero.checkpoint = Some(RecordCheckpoint {
            kind: CheckpointKind::ReactIteration,
            iteration_seq: 0,
            ..RecordCheckpoint::default()
        });
        assert!(iteration_zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("a checkpoint past the request start requires a completed iteration"));

        let mut rollover_zero = full_record();
        rollover_zero.checkpoint = Some(RecordCheckpoint {
            kind: CheckpointKind::Rollover,
            iteration_seq: 0,
            ..RecordCheckpoint::default()
        });
        assert!(rollover_zero.validate().is_err());

        let mut request_start = full_record();
        request_start.checkpoint = Some(RecordCheckpoint {
            kind: CheckpointKind::RequestStart,
            iteration_seq: 0,
            ..RecordCheckpoint::default()
        });
        request_start.validate().unwrap();
    }

    #[test]
    fn with_ending_and_with_checkpoint_replace_each_other() {
        let record = full_record();
        assert!(record.checkpoint.is_some());
        let ended = record.clone().with_ending(awaiting_ending("req")).unwrap();
        assert!(ended.checkpoint.is_none());
        assert!(ended.is_waiting());
        let resumed = ended
            .clone()
            .with_checkpoint(RecordCheckpoint {
                kind: CheckpointKind::RequestStart,
                request_id: "req2".to_string(),
                execution_id: "exec2".to_string(),
                iteration_seq: 0,
                last_group_message_id: None,
            })
            .unwrap();
        assert!(resumed.ending.is_none());
        assert!(resumed.checkpoint.is_some());

        let mut invalid = awaiting_ending("req");
        invalid.spec_fingerprint = None;
        assert!(record.with_ending(invalid).is_err());
    }

    #[test]
    fn patch_stopped_calls_adds_placeholders_only_for_unanswered_calls() {
        let patched = patch_stopped_calls(&window_messages());
        assert_eq!(patched.len(), 4);
        assert_eq!(patched[3].role, Role::Tool);
        assert_eq!(patched[3].tool_call_id.as_deref(), Some("c2"));
        assert_eq!(patched[3].text(), PLACEHOLDER_TEXT);
        assert_eq!(HANDOFF_TEXT_PREFIX, "【实际交出的最终回答】\n");

        // Idempotent: the placeholder answers the call on the next pass.
        assert_eq!(patch_stopped_calls(&patched), patched);
    }

    #[test]
    fn patch_stopped_calls_leaves_answers_and_stray_results_alone() {
        let answered = vec![
            BaseMessage::user("go"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read").with_id("c1")]),
            BaseMessage::tool("c1", "ok"),
            BaseMessage::assistant("done"),
        ];
        assert_eq!(patch_stopped_calls(&answered), answered);

        let stray = vec![
            BaseMessage::tool("ghost", "orphan"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read").with_id("c1")]),
            BaseMessage::tool("other", "unrelated"),
            BaseMessage::tool("c1", "ok"),
            BaseMessage::user("next"),
        ];
        let patched = patch_stopped_calls(&stray);
        assert_eq!(patched.len(), stray.len());
        assert_eq!(patched, stray);

        let no_ids = vec![
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read")]),
            BaseMessage::user("next"),
        ];
        assert_eq!(patch_stopped_calls(&no_ids), no_ids);

        assert_eq!(patch_stopped_calls(&[]), Vec::new());
    }

    #[test]
    fn patch_stopped_calls_reports_each_missing_call_once() {
        let messages = vec![BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("read").with_id("c1"),
            ToolCall::new("read").with_id("c1"),
        ])];
        let patched = patch_stopped_calls(&messages);
        assert_eq!(patched.len(), 2);
        assert_eq!(patched[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(patch_stopped_calls(&patched), patched);
    }

    #[test]
    fn clear_stale_scene_refuses_another_request() {
        let mut record = full_record();
        record.checkpoint = None;
        record.ending = Some(awaiting_ending("req"));
        record.validate().unwrap();
        let before = record.clone();
        let error = record.clear_stale_scene("other").unwrap_err();
        assert!(error
            .to_string()
            .contains("another Request is waiting for a reply in this Session"));
        assert_eq!(record, before);
    }

    #[test]
    fn clear_stale_scene_settles_its_own_scene() {
        let mut record = full_record();
        record.checkpoint = None;
        record.ending = Some(awaiting_ending("req"));
        record.validate().unwrap();
        let messages = record.messages.len();
        record.clear_stale_scene("req").unwrap();
        assert!(!record.is_waiting());
        assert!(record.ending.is_none());
        assert!(record.interruption.is_none());
        assert_eq!(record.messages.len(), messages + 1);
        assert_eq!(record.messages[3].text(), PLACEHOLDER_TEXT);
        // The window/cursor fields and the remembered confirmations stay.
        assert_eq!(record.current_window_id.as_deref(), Some("w2"));
        assert_eq!(record.window_seq, Some(2));
        assert!(record.last_rollover.is_some());
        assert_eq!(record.last_group_message_id.as_deref(), Some("g1"));
        assert!(record.continuation.is_some());
        assert_eq!(record.auto_confirm.get("bash"), Some(&true));

        // A record that is not waiting is returned unchanged.
        let mut idle = full_record();
        let before = idle.clone();
        idle.clear_stale_scene("req").unwrap();
        assert_eq!(idle, before);
    }

    #[tokio::test]
    async fn save_and_load_keep_every_field() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = FileSessionStore::new(dir.path());

        let checkpoint_record = full_record();
        store.save(&checkpoint_record).await.unwrap();
        let loaded = store.load(&binding()).await.unwrap();
        assert_eq!(loaded, Some(checkpoint_record));

        let mut ending_record = full_record();
        ending_record.checkpoint = None;
        ending_record.ending = Some(awaiting_ending("req"));
        ending_record.validate().unwrap();
        store.save(&ending_record).await.unwrap();
        let loaded = store.load(&binding()).await.unwrap().unwrap();
        assert_eq!(loaded, ending_record);
        assert!(loaded.is_waiting());
        assert!(dir
            .path()
            .join("agent")
            .join("s%2F1")
            .join("state.json")
            .exists());
    }

    #[tokio::test]
    async fn unknown_keys_survive_a_load_save_round_trip() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = FileSessionStore::new(dir.path());
        let path = store.path_for(&binding());
        let mut value = serde_json::to_value(full_record()).unwrap();
        value["future_window"] = json!({"candidate_version": 2});
        value["future_flag"] = json!(true);
        write_json_atomic(&path, &value).await.unwrap();

        let loaded = store.load(&binding()).await.unwrap().unwrap();
        assert_eq!(
            loaded.extra.get("future_window"),
            Some(&json!({"candidate_version": 2}))
        );
        assert_eq!(loaded.extra.get("future_flag"), Some(&json!(true)));
        loaded.validate().unwrap();

        store.save(&loaded).await.unwrap();
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(text.contains("future_window"));
        assert!(text.contains("future_flag"));
        assert_eq!(store.load(&binding()).await.unwrap(), Some(loaded));
    }

    #[tokio::test]
    async fn missing_and_malformed_records() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = FileSessionStore::new(dir.path());
        assert_eq!(store.load(&binding()).await.unwrap(), None);
        assert!(!store.delete(&binding()).await.unwrap());

        let path = store.path_for(&binding());
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&path, b"{not json").await.unwrap();
        let error = store.load(&binding()).await.unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error.to_string().contains(RECORD_INVALID_PREFIX));
        assert!(error.to_string().contains("state.json"));

        let mut invalid = serde_json::to_value(full_record()).unwrap();
        invalid["window_seq"] = json!(7);
        write_json_atomic(&path, &invalid).await.unwrap();
        let error = store.load(&binding()).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("last rollover must point at the current window"));
        assert!(error.to_string().contains("state.json"));
    }

    #[tokio::test]
    async fn save_rejects_an_invalid_record_and_delete_removes_the_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = FileSessionStore::new(dir.path());
        let mut invalid = full_record();
        invalid.window_seq = None;
        let error = store.save(&invalid).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("current window identity and sequence must be present together"));
        assert!(!store.path_for(&binding()).exists());

        store.save(&full_record()).await.unwrap();
        let path = store.path_for(&binding());
        assert!(path.exists());
        assert!(store.delete(&binding()).await.unwrap());
        assert!(!path.exists());
        assert_eq!(store.load(&binding()).await.unwrap(), None);
        assert!(!store.delete(&binding()).await.unwrap());
    }

    #[test]
    fn store_paths_encode_ids_and_follow_the_workspace_node() {
        let store = FileSessionStore::new("/tmp/pa");
        assert_eq!(
            store.path_for(&binding()),
            PathBuf::from("/tmp/pa/agent/s%2F1/state.json")
        );
        assert_eq!(
            store.path_for(&SessionBinding {
                agent_id: "a:b".to_string(),
                session_id: "c d".to_string(),
            }),
            PathBuf::from("/tmp/pa/a%3Ab/c%20d/state.json")
        );

        let dir = tempfile::TempDir::new().unwrap();
        let workspace = Workspace::for_language(dir.path(), "en");
        let store = FileSessionStore::for_workspace(&workspace);
        assert_eq!(
            store.path_for(&binding()),
            dir.path()
                .join("persistent_agent")
                .join("sessions")
                .join("agent")
                .join("s%2F1")
                .join("state.json")
        );
    }
}
