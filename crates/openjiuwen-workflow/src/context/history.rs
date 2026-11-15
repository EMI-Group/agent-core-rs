//! Session-backed conversation history shared by components that read past turns.
//!
//! Python reads and writes conversation history through
//! `ModelContext.get_messages()` / `ModelContext.add_messages()`
//! (`openjiuwen/core/context_engine/base.py`). [`openjiuwen_session::Session`] exposes no
//! message API, so this module reproduces those two operations as a session-backed
//! store over the global-lane key [`MESSAGES_STATE_KEY`].
//!
//! [`crate::components::intent_detection::IntentDetectionComponent`] already keeps its
//! conversation in that lane as a JSON array of `{"role": ..., "content": ...}` objects;
//! [`SessionMessageHistory`] uses the same key and shape, so a workflow running both
//! components shares one conversation.
//!
//! # Deviations from Python
//!
//! - **Session-backed.** The store *is* the workflow session's global lane
//!   `__messages__` — checkpointed and visible to every component — rather than an
//!   in-memory `ModelContext` owned by one component.
//! - **Text only.** Only plain text survives a round trip: [`BaseMessage::text`]
//!   flattens multimodal content, and tool calls, reasoning text, speaker names and
//!   per-message metadata are dropped. An entry whose `role` is missing or is not one of
//!   the four known roles, or that is not an object with a string `content`, is skipped
//!   when the lane is read back (and counted in a
//!   [`HistoryCorruptionReason::MalformedEntries`] record).
//! - **Failures are recorded, not raised.** [`MessageHistory`] methods cannot fail (every
//!   method returns `()` or a value), so a read failure, a non-array lane, skipped entries
//!   and a write failure are logged with `tracing::warn!` *and* reported through
//!   [`MessageHistory::corruption`] as a [`HistoryCorruption`]. A caller that wants to fail
//!   hard converts the record with [`history_corrupt_error`]. A non-array lane value is
//!   treated as empty history with a warning, never as a panic.
//!
//! # Status code
//!
//! [`history_corrupt_error`] renders [`WORKFLOW_HISTORY_CORRUPT`] (`100060`), a
//! Rust-only code. Python's `codes.py` allocated `100010`–`100015` (workflow definition),
//! `100054` (component execution) and `100100` (compile); `100060` sits in the unused gap
//! between them, so it cannot collide with a ported Python code and stays inside this
//! crate's workflow definition block.
//!
//! ```
//! use openjiuwen_workflow::context::history::{
//!     history_corrupt_error, HistoryCorruption, HistoryCorruptionReason,
//! };
//!
//! let corruption =
//!     HistoryCorruption::new(HistoryCorruptionReason::NotAnArray, "lane holds a JSON number");
//! let error = history_corrupt_error(&corruption);
//! assert!(error.message().starts_with("[100060] "));
//! ```
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Error;
use openjiuwen_foundation::{BaseMessage, MessageContent, Role};
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tracing::warn;

/// Session global-lane key holding the conversation as a JSON array of
/// `{"role": ..., "content": ...}` objects.
///
/// This is the exact key and shape `intent_detection` uses, so both components share one
/// conversation. [`openjiuwen_session::Session::update_state`] deep-merges objects but
/// replaces arrays wholesale, which makes a read-modify-write of this key safe.
pub const MESSAGES_STATE_KEY: &str = "__messages__";

/// The status code [`history_corrupt_error`] renders.
///
/// Python never allocated it (`100010`–`100015`, `100054` and `100100` are the ported
/// codes), so this Rust-only failure uses the free slot at the end of the workflow
/// definition block; see the [module docs](self).
pub const WORKFLOW_HISTORY_CORRUPT: u32 = 100_060;

/// Why a [`MessageHistory`] could not read or write the conversation.
///
/// The reasons are mutually exclusive observations of one lane; a store keeps the most
/// recent one, so a later success replaces or clears it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryCorruptionReason {
    /// The lane holds a JSON value that is not an array (so no entry can be read).
    NotAnArray,
    /// The session could not read the lane (the state store is unusable).
    Unreadable,
    /// One or more lane entries are not `{"role", "content"}` records and were skipped.
    MalformedEntries,
    /// The session could not persist the lane.
    WriteFailed,
}

impl HistoryCorruptionReason {
    /// The lowercase wire name (`"not_an_array"`, `"unreadable"`, …).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotAnArray => "not_an_array",
            Self::Unreadable => "unreadable",
            Self::MalformedEntries => "malformed_entries",
            Self::WriteFailed => "write_failed",
        }
    }
}

impl fmt::Display for HistoryCorruptionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One observed conversation-history problem: what went wrong and the human-readable
/// detail.
///
/// [`SessionMessageHistory`] records the most recent one instead of failing, because the
/// [`MessageHistory`] methods cannot return an error. Callers that want a run to fail
/// convert it with [`history_corrupt_error`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryCorruption {
    /// What kind of problem was observed.
    pub reason: HistoryCorruptionReason,
    /// A one-line explanation naming the lane and, where applicable, the entry counts.
    pub detail: String,
}

impl HistoryCorruption {
    /// Builds a record from `reason` and `detail`.
    #[must_use]
    pub fn new(reason: HistoryCorruptionReason, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for HistoryCorruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.reason, self.detail)
    }
}

/// Renders `corruption` as a coded [`Error::Workflow`] (`"[100060] …"`).
///
/// The builder stays here rather than in [`crate::error`] because it is a Rust-only code
/// for one component-local family, like the intent-detection and questioner builders.
pub fn history_corrupt_error(corruption: &HistoryCorruption) -> Error {
    Error::workflow(format!(
        "[{WORKFLOW_HISTORY_CORRUPT}] message history is corrupt, reason={}, detail='{}'",
        corruption.reason, corruption.detail
    ))
}

/// A conversation history that can be read, appended to, and cleared.
///
/// This stands in for Python's `ModelContext.get_messages()` / `add_messages()`; see the
/// [module docs](self) for the deviations. Implementations are `Send + Sync` so a handle
/// can be shared across a component's helper tasks.
#[async_trait]
pub trait MessageHistory: Send + Sync {
    /// Every recorded message, oldest first.
    async fn messages(&self) -> Vec<BaseMessage>;

    /// Persists `messages` after the existing ones.
    async fn append(&self, messages: Vec<BaseMessage>);

    /// Removes every recorded message.
    async fn clear(&self);

    /// The most recent conversation-history problem, or `None` when the last read or
    /// write was clean.
    ///
    /// Reporting is additive: the three methods above keep their signatures and still
    /// degrade to empty history / a logged warning, so an implementation that cannot
    /// observe problems simply keeps this default.
    fn corruption(&self) -> Option<HistoryCorruption> {
        None
    }
}

/// The default [`MessageHistory`]: the session's `__messages__` global lane.
///
/// Cloning is cheap and every clone shares the session handle *and* the corruption
/// record, so two handles over the same session always agree on both the conversation and
/// the last problem observed.
#[derive(Clone)]
pub struct SessionMessageHistory {
    session: Arc<dyn Session>,
    corruption: Arc<Mutex<Option<HistoryCorruption>>>,
}

impl SessionMessageHistory {
    /// Builds a history store over `session`'s global lane.
    #[must_use]
    pub fn new(session: Arc<dyn Session>) -> Self {
        Self {
            session,
            corruption: Arc::new(Mutex::new(None)),
        }
    }

    /// The session the store reads and writes.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }

    /// Replaces the recorded problem with `corruption`.
    fn record(&self, corruption: HistoryCorruption) {
        *self.corruption.lock() = Some(corruption);
    }

    /// Forgets the recorded problem after a successful lane write.
    fn forget_corruption(&self) {
        *self.corruption.lock() = None;
    }

    /// Reads the lane as a JSON array, tolerating every way it can be unusable.
    ///
    /// A missing lane is an empty history; a read failure or a non-array value is an empty
    /// history plus a `tracing::warn!` and a recorded [`HistoryCorruption`] — the
    /// [`MessageHistory`] methods cannot report errors, and a broken lane must never panic
    /// a run.
    async fn read_lane(&self) -> Vec<Value> {
        match self.session.get_state(MESSAGES_STATE_KEY).await {
            Ok(None) => Vec::new(),
            Ok(Some(Value::Array(items))) => items,
            Ok(Some(other)) => {
                warn!(
                    key = MESSAGES_STATE_KEY,
                    value = %other,
                    "message history lane is not a JSON array; treating it as empty"
                );
                self.record(HistoryCorruption::new(
                    HistoryCorruptionReason::NotAnArray,
                    format!(
                        "message history lane '{MESSAGES_STATE_KEY}' holds a JSON {}, not an array",
                        json_type_name(&other)
                    ),
                ));
                Vec::new()
            }
            Err(error) => {
                warn!(
                    key = MESSAGES_STATE_KEY,
                    %error,
                    "cannot read the message history lane; treating it as empty"
                );
                self.record(HistoryCorruption::new(
                    HistoryCorruptionReason::Unreadable,
                    format!("cannot read message history lane '{MESSAGES_STATE_KEY}': {error}"),
                ));
                Vec::new()
            }
        }
    }

    /// Replaces the lane with `items`.
    ///
    /// A write failure is logged and recorded, not returned: the trait cannot report it,
    /// and the caller (usually a component finishing a turn) has no recovery either. A
    /// successful write clears any earlier record, because the lane now holds exactly what
    /// this store wrote.
    async fn write_lane(&self, items: Vec<Value>) {
        let mut patch = Map::new();
        patch.insert(MESSAGES_STATE_KEY.to_owned(), Value::Array(items));
        match self.session.update_state(Value::Object(patch)).await {
            Ok(()) => self.forget_corruption(),
            Err(error) => {
                warn!(
                    key = MESSAGES_STATE_KEY,
                    %error,
                    "cannot persist the message history lane"
                );
                self.record(HistoryCorruption::new(
                    HistoryCorruptionReason::WriteFailed,
                    format!("cannot persist message history lane '{MESSAGES_STATE_KEY}': {error}"),
                ));
            }
        }
    }
}

impl fmt::Debug for SessionMessageHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionMessageHistory")
            .field("key", &MESSAGES_STATE_KEY)
            .field("session_id", &self.session.id())
            .field("corrupt", &self.corruption.lock().is_some())
            .finish()
    }
}

#[async_trait]
impl MessageHistory for SessionMessageHistory {
    async fn messages(&self) -> Vec<BaseMessage> {
        let items = self.read_lane().await;
        let (messages, skipped) = parse_messages(&items);
        if skipped > 0 {
            self.record(HistoryCorruption::new(
                HistoryCorruptionReason::MalformedEntries,
                format!(
                    "skipped {skipped} of {} entries in message history lane \
                     '{MESSAGES_STATE_KEY}' because they are not {{\"role\", \"content\"}} records",
                    items.len()
                ),
            ));
        }
        messages
    }

    async fn append(&self, messages: Vec<BaseMessage>) {
        if messages.is_empty() {
            return;
        }
        let mut items = self.read_lane().await;
        items.extend(messages.iter().map(|message| {
            json!({
                "role": message.role.as_str(),
                "content": message.text().into_owned(),
            })
        }));
        self.write_lane(items).await;
    }

    async fn clear(&self) {
        self.write_lane(Vec::new()).await;
    }

    fn corruption(&self) -> Option<HistoryCorruption> {
        self.corruption.lock().clone()
    }
}

/// Decodes every lane entry, returning the parsed messages and how many were skipped.
fn parse_messages(items: &[Value]) -> (Vec<BaseMessage>, usize) {
    let mut messages = Vec::with_capacity(items.len());
    let mut skipped = 0_usize;
    for item in items {
        match parse_message(item) {
            Some(message) => messages.push(message),
            None => skipped += 1,
        }
    }
    (messages, skipped)
}

/// Parses one lane entry into a [`BaseMessage`], or `None` when it is not a known shape.
///
/// The role must be a string naming one of the four known roles; a vendor role
/// ([`Role::Other`]) is skipped because the lane is a plain `{"role", "content"}` record
/// and Python's `ModelContext` counterpart is not reproduced here.
fn parse_message(item: &Value) -> Option<BaseMessage> {
    let object = item.as_object()?;
    let role = serde_json::from_value::<Role>(object.get("role")?.clone()).ok()?;
    if matches!(role, Role::Other(_)) {
        return None;
    }
    let content = object.get("content")?.as_str()?;
    Some(BaseMessage {
        role,
        content: MessageContent::Text(content.to_owned()),
        ..BaseMessage::default()
    })
}

/// The JSON type name of `value`, used in a [`HistoryCorruptionReason::NotAnArray`] detail.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::{ErrorKind, Result};
    use openjiuwen_session::stream::StreamItem;
    use openjiuwen_session::{
        InteractionRequest, InteractionResponse, NoopTraceSink, SessionType, TraceSink,
    };

    use super::*;

    fn store() -> (SessionMessageHistory, Arc<dyn Session>) {
        let session: Arc<dyn Session> =
            Arc::new(openjiuwen_session::WorkflowSession::new("history-tests"));
        (SessionMessageHistory::new(Arc::clone(&session)), session)
    }

    /// A session whose state store can be made unreadable or read-only, for the two
    /// corruption paths an in-process [`openjiuwen_session::WorkflowSession`] cannot reach.
    struct FaultySession {
        id: String,
        trace: NoopTraceSink,
        lane: Mutex<Value>,
        read_fails: bool,
        write_fails: bool,
    }

    impl FaultySession {
        fn new(lane: Value, read_fails: bool, write_fails: bool) -> Arc<Self> {
            Arc::new(Self {
                id: "faulty-history".to_owned(),
                trace: NoopTraceSink,
                lane: Mutex::new(lane),
                read_fails,
                write_fails,
            })
        }

        fn unreadable() -> Arc<Self> {
            Self::new(Value::Null, true, false)
        }

        fn unwritable() -> Arc<Self> {
            Self::new(Value::Array(Vec::new()), false, true)
        }
    }

    #[async_trait]
    impl Session for FaultySession {
        fn id(&self) -> &str {
            &self.id
        }

        fn parent_id(&self) -> Option<&str> {
            None
        }

        fn session_type(&self) -> SessionType {
            SessionType::Workflow
        }

        async fn get_state(&self, key: &str) -> Result<Option<Value>> {
            if self.read_fails {
                return Err(Error::session("state store is unavailable"));
            }
            if key != MESSAGES_STATE_KEY {
                return Ok(None);
            }
            Ok(Some(self.lane.lock().clone()))
        }

        async fn update_state(&self, patch: Value) -> Result<()> {
            if self.write_fails {
                return Err(Error::session("state store is read-only"));
            }
            *self.lane.lock() = patch
                .get(MESSAGES_STATE_KEY)
                .cloned()
                .unwrap_or(Value::Null);
            Ok(())
        }

        async fn get_component_state(&self, _component_id: &str) -> Result<Option<Value>> {
            Ok(None)
        }

        async fn update_component_state(&self, _component_id: &str, _patch: Value) -> Result<()> {
            Ok(())
        }

        async fn write_stream(&self, _item: StreamItem) -> Result<()> {
            Ok(())
        }

        async fn request_interaction(
            &self,
            _request: InteractionRequest,
        ) -> Result<InteractionResponse> {
            Err(Error::unsupported("the faulty session has no interactions"))
        }

        fn trace(&self) -> &dyn TraceSink {
            &self.trace
        }
    }

    #[tokio::test]
    async fn append_round_trips_roles_and_order() {
        let (history, session) = store();
        history
            .append(vec![BaseMessage::user("a"), BaseMessage::assistant("b")])
            .await;

        let messages = history.messages().await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].text(), "a");
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].text(), "b");
        assert_eq!(history.corruption(), None);

        assert_eq!(
            session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
            Some(json!([
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}
            ]))
        );
    }

    #[tokio::test]
    async fn append_merges_after_existing_entries() {
        let (history, session) = store();
        history.append(vec![BaseMessage::user("a")]).await;
        history.append(vec![BaseMessage::assistant("b")]).await;

        assert_eq!(
            session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
            Some(json!([
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}
            ]))
        );
        let messages = history.messages().await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text(), "a");
        assert_eq!(messages[1].text(), "b");
    }

    #[tokio::test]
    async fn clear_empties_the_lane() {
        let (history, session) = store();
        history
            .append(vec![BaseMessage::user("a"), BaseMessage::assistant("b")])
            .await;

        history.clear().await;

        assert_eq!(
            session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
            Some(json!([]))
        );
        assert!(history.messages().await.is_empty());
    }

    #[tokio::test]
    async fn a_non_array_lane_yields_empty_history_and_a_record() {
        let (history, session) = store();
        session
            .update_state(json!({MESSAGES_STATE_KEY: 7}))
            .await
            .unwrap();

        assert!(history.messages().await.is_empty());
        let corruption = history.corruption().expect("a recorded corruption");
        assert_eq!(corruption.reason, HistoryCorruptionReason::NotAnArray);
        assert!(corruption.detail.contains(MESSAGES_STATE_KEY));
        assert!(corruption.detail.contains("number"));

        // A successful append rewrites the lane, so the record is dropped.
        history.append(vec![BaseMessage::user("a")]).await;
        assert_eq!(history.corruption(), None);
        assert_eq!(history.messages().await.len(), 1);
    }

    #[tokio::test]
    async fn malformed_and_unknown_entries_are_skipped_and_counted() {
        let (history, session) = store();
        session
            .update_state(json!({
                MESSAGES_STATE_KEY: [
                    {"role": "user", "content": "kept"},
                    {"role": "developer", "content": "vendor role"},
                    {"role": 7, "content": "non-string role"},
                    {"content": "missing role"},
                    "not an object",
                    {"role": "assistant", "content": 42}
                ]
            }))
            .await
            .unwrap();

        let messages = history.messages().await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].text(), "kept");

        let corruption = history.corruption().expect("a recorded corruption");
        assert_eq!(corruption.reason, HistoryCorruptionReason::MalformedEntries);
        assert!(corruption.detail.contains("skipped 5 of 6"));
        assert!(corruption.detail.contains(MESSAGES_STATE_KEY));
    }

    #[tokio::test]
    async fn a_clean_lane_records_nothing() {
        let (history, session) = store();
        session
            .update_state(json!({MESSAGES_STATE_KEY: [{"role": "user", "content": "a"}]}))
            .await
            .unwrap();

        assert_eq!(history.messages().await.len(), 1);
        assert_eq!(history.corruption(), None);
    }

    #[tokio::test]
    async fn clear_forgets_a_recorded_corruption() {
        let (history, session) = store();
        session
            .update_state(json!({MESSAGES_STATE_KEY: {"role": "user"}}))
            .await
            .unwrap();
        assert!(history.messages().await.is_empty());
        assert!(history.corruption().is_some());

        history.clear().await;
        assert_eq!(history.corruption(), None);
    }

    #[tokio::test]
    async fn an_unreadable_lane_is_recorded_as_unreadable() {
        let session: Arc<dyn Session> = FaultySession::unreadable();
        let history = SessionMessageHistory::new(session);

        assert!(history.messages().await.is_empty());
        let corruption = history.corruption().expect("a recorded corruption");
        assert_eq!(corruption.reason, HistoryCorruptionReason::Unreadable);
        assert!(corruption.detail.contains(MESSAGES_STATE_KEY));
        assert!(corruption.detail.contains("state store is unavailable"));
    }

    #[tokio::test]
    async fn an_unwritable_lane_is_recorded_as_write_failed() {
        let session: Arc<dyn Session> = FaultySession::unwritable();
        let history = SessionMessageHistory::new(session);

        history.append(vec![BaseMessage::user("a")]).await;
        let corruption = history.corruption().expect("a recorded corruption");
        assert_eq!(corruption.reason, HistoryCorruptionReason::WriteFailed);
        assert!(corruption.detail.contains(MESSAGES_STATE_KEY));
        assert!(corruption.detail.contains("state store is read-only"));
    }

    #[tokio::test]
    async fn empty_append_writes_nothing() {
        let (history, session) = store();
        history.append(Vec::new()).await;

        assert_eq!(session.get_state(MESSAGES_STATE_KEY).await.unwrap(), None);
        assert_eq!(history.corruption(), None);
    }

    #[tokio::test]
    async fn debug_reports_the_lane_key_and_session() {
        let (history, _session) = store();
        let text = format!("{history:?}");
        assert!(text.contains(MESSAGES_STATE_KEY));
        assert!(text.contains("history-tests"));
        assert!(text.contains("corrupt: false"));
    }

    #[test]
    fn reason_wire_names_and_display() {
        assert_eq!(HistoryCorruptionReason::NotAnArray.as_str(), "not_an_array");
        assert_eq!(HistoryCorruptionReason::Unreadable.as_str(), "unreadable");
        assert_eq!(
            HistoryCorruptionReason::MalformedEntries.as_str(),
            "malformed_entries"
        );
        assert_eq!(
            HistoryCorruptionReason::WriteFailed.as_str(),
            "write_failed"
        );
        assert_eq!(
            HistoryCorruptionReason::MalformedEntries.to_string(),
            "malformed_entries"
        );
        assert_eq!(
            serde_json::to_value(HistoryCorruptionReason::WriteFailed).unwrap(),
            json!("write_failed")
        );
    }

    #[test]
    fn corruption_round_trips_through_json() {
        let corruption = HistoryCorruption::new(
            HistoryCorruptionReason::MalformedEntries,
            "skipped 2 of 4 entries",
        );
        let wire = serde_json::to_value(&corruption).unwrap();
        assert_eq!(
            wire,
            json!({"reason": "malformed_entries", "detail": "skipped 2 of 4 entries"})
        );
        assert_eq!(
            serde_json::from_value::<HistoryCorruption>(wire).unwrap(),
            corruption
        );
        assert_eq!(
            corruption.to_string(),
            "malformed_entries: skipped 2 of 4 entries"
        );
    }

    #[test]
    fn error_builder_renders_the_local_code() {
        let corruption =
            HistoryCorruption::new(HistoryCorruptionReason::NotAnArray, "lane holds 7");
        let error = history_corrupt_error(&corruption);
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert_eq!(
            error.message(),
            "[100060] message history is corrupt, reason=not_an_array, detail='lane holds 7'"
        );
        assert_eq!(WORKFLOW_HISTORY_CORRUPT, 100_060);
    }

    #[test]
    fn parse_messages_counts_every_skipped_shape() {
        let items = vec![
            json!({"role": "user", "content": "a"}),
            json!(null),
            json!({"role": "assistant", "content": null}),
            json!({"role": "tool", "content": "ok"}),
            json!(["not", "an", "object"]),
        ];

        let (messages, skipped) = parse_messages(&items);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Tool);
        assert_eq!(skipped, 3);
    }

    #[test]
    fn json_type_names_cover_every_json_kind() {
        assert_eq!(json_type_name(&Value::Null), "null");
        assert_eq!(json_type_name(&json!(true)), "boolean");
        assert_eq!(json_type_name(&json!(1.5)), "number");
        assert_eq!(json_type_name(&json!("s")), "string");
        assert_eq!(json_type_name(&json!([])), "array");
        assert_eq!(json_type_name(&json!({})), "object");
    }
}
