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
//!   when the lane is read back.
//! - **No error channel.** [`MessageHistory`] cannot report failures (every method
//!   returns `()` or a value), so a read failure yields an empty history and a write
//!   failure is logged with `tracing::warn!` instead of being propagated. A non-array
//!   lane value is treated as empty history with a warning, never as a panic.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_foundation::{BaseMessage, MessageContent, Role};
use openjiuwen_session::Session;
use serde_json::{json, Map, Value};
use tracing::warn;

/// Session global-lane key holding the conversation as a JSON array of
/// `{"role": ..., "content": ...}` objects.
///
/// This is the exact key and shape `intent_detection` uses, so both components share one
/// conversation. [`openjiuwen_session::Session::update_state`] deep-merges objects but
/// replaces arrays wholesale, which makes a read-modify-write of this key safe.
pub const MESSAGES_STATE_KEY: &str = "__messages__";

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
}

/// The default [`MessageHistory`]: the session's `__messages__` global lane.
///
/// Cloning is cheap — clones share the session handle — but each handle reads and writes
/// the lane on every call, so two handles over the same session always agree.
#[derive(Clone)]
pub struct SessionMessageHistory {
    session: Arc<dyn Session>,
}

impl SessionMessageHistory {
    /// Builds a history store over `session`'s global lane.
    #[must_use]
    pub fn new(session: Arc<dyn Session>) -> Self {
        Self { session }
    }

    /// The session the store reads and writes.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }

    /// Reads the lane as a JSON array, tolerating every way it can be unusable.
    ///
    /// A missing lane is an empty history, a read failure or a non-array value is an
    /// empty history plus a `tracing::warn!` — the [`MessageHistory`] methods cannot
    /// report errors, and a broken lane must never panic a run.
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
                Vec::new()
            }
            Err(error) => {
                warn!(
                    key = MESSAGES_STATE_KEY,
                    %error,
                    "cannot read the message history lane; treating it as empty"
                );
                Vec::new()
            }
        }
    }

    /// Replaces the lane with `items`.
    ///
    /// A write failure is logged, not returned: the trait cannot report it, and the
    /// caller (usually a component finishing a turn) has no recovery either.
    async fn write_lane(&self, items: Vec<Value>) {
        let mut patch = Map::new();
        patch.insert(MESSAGES_STATE_KEY.to_owned(), Value::Array(items));
        if let Err(error) = self.session.update_state(Value::Object(patch)).await {
            warn!(
                key = MESSAGES_STATE_KEY,
                %error,
                "cannot persist the message history lane"
            );
        }
    }
}

impl fmt::Debug for SessionMessageHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionMessageHistory")
            .field("key", &MESSAGES_STATE_KEY)
            .field("session_id", &self.session.id())
            .finish()
    }
}

#[async_trait]
impl MessageHistory for SessionMessageHistory {
    async fn messages(&self) -> Vec<BaseMessage> {
        self.read_lane()
            .await
            .into_iter()
            .filter_map(|item| parse_message(&item))
            .collect()
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn store() -> (SessionMessageHistory, Arc<dyn Session>) {
        let session: Arc<dyn Session> =
            Arc::new(openjiuwen_session::WorkflowSession::new("history-tests"));
        (SessionMessageHistory::new(Arc::clone(&session)), session)
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
    async fn a_non_array_lane_yields_empty_history() {
        let (history, session) = store();
        session
            .update_state(json!({MESSAGES_STATE_KEY: 7}))
            .await
            .unwrap();

        assert!(history.messages().await.is_empty());
    }

    #[tokio::test]
    async fn malformed_and_unknown_entries_are_skipped() {
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
    }

    #[tokio::test]
    async fn empty_append_writes_nothing() {
        let (history, session) = store();
        history.append(Vec::new()).await;

        assert_eq!(session.get_state(MESSAGES_STATE_KEY).await.unwrap(), None);
    }

    #[tokio::test]
    async fn debug_reports_the_lane_key_and_session() {
        let (history, _session) = store();
        let text = format!("{history:?}");
        assert!(text.contains(MESSAGES_STATE_KEY));
        assert!(text.contains("history-tests"));
    }
}
