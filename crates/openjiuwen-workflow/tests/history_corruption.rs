#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Conversation-history corruption is observable, not just logged.
//!
//! `MessageHistory` cannot return errors, so a broken `__messages__` lane degrades a run
//! silently. These tests pin the additive signal that fixes that: `corruption()` records
//! the last problem (`NotAnArray`, `Unreadable`, `MalformedEntries`, `WriteFailed`), a
//! successful `append`/`clear` forgets it, the record is reachable through
//! `ComponentContext`, and the default trait method keeps external implementations
//! source-compatible. Everything is offline and deterministic: an in-process
//! `WorkflowSession` holds the lane and no clock or network is involved.
mod common;

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::BaseMessage;
use openjiuwen_workflow::context::history::{
    history_corrupt_error, HistoryCorruptionReason, MessageHistory, SessionMessageHistory,
    MESSAGES_STATE_KEY,
};
use openjiuwen_workflow::ComponentContext;
use serde_json::json;

/// A history store that the crate knows nothing about: the trait's new method must stay
/// optional for external implementations.
#[derive(Debug, Default)]
struct ExternalHistory;

#[async_trait]
impl MessageHistory for ExternalHistory {
    async fn messages(&self) -> Vec<BaseMessage> {
        Vec::new()
    }

    async fn append(&self, _messages: Vec<BaseMessage>) {}

    async fn clear(&self) {}
}

#[tokio::test]
async fn a_non_array_lane_reports_not_an_array() {
    let session = common::session("history-not-an-array");
    session
        .update_state(json!({MESSAGES_STATE_KEY: {"role": "user"}}))
        .await
        .unwrap();
    let history = SessionMessageHistory::new(Arc::clone(&session));

    assert!(history.messages().await.is_empty());

    let corruption = history.corruption().expect("a corruption record");
    assert_eq!(corruption.reason, HistoryCorruptionReason::NotAnArray);
    assert!(
        corruption.detail.contains(MESSAGES_STATE_KEY),
        "detail names the lane: {}",
        corruption.detail
    );
    assert!(
        corruption.detail.contains("JSON object"),
        "detail names the JSON type: {}",
        corruption.detail
    );

    let error = history_corrupt_error(&corruption);
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(
        error.message().starts_with("[100060] "),
        "coded message: {}",
        error.message()
    );
    assert!(error.message().contains("not_an_array"));
    assert!(error.message().contains(MESSAGES_STATE_KEY));
}

#[tokio::test]
async fn malformed_entries_are_skipped_and_counted() {
    let session = common::session("history-malformed");
    session
        .update_state(json!({
            MESSAGES_STATE_KEY: [
                {"role": "user", "content": "kept"},
                {"role": "developer", "content": "unknown role"},
                {"role": 7, "content": "non-string role"},
                {"content": "missing role"},
                "not an object",
                {"role": "assistant", "content": 42},
                {"role": "assistant", "content": "also kept"}
            ]
        }))
        .await
        .unwrap();
    let history = SessionMessageHistory::new(session);

    let messages = history.messages().await;
    assert_eq!(messages.len(), 2, "the valid entries still decode");
    assert_eq!(messages[0].text(), "kept");
    assert_eq!(messages[1].text(), "also kept");

    let corruption = history.corruption().expect("a corruption record");
    assert_eq!(corruption.reason, HistoryCorruptionReason::MalformedEntries);
    assert!(
        corruption.detail.contains("skipped 5 of 7"),
        "the skipped count is aggregated: {}",
        corruption.detail
    );
    assert!(corruption.detail.contains(MESSAGES_STATE_KEY));
    assert!(history_corrupt_error(&corruption)
        .message()
        .contains("malformed_entries"));
}

#[tokio::test]
async fn a_clean_lane_reports_nothing_and_writes_clear_the_record() {
    let session = common::session("history-clean");
    let history = SessionMessageHistory::new(Arc::clone(&session));

    history.append(vec![BaseMessage::user("a")]).await;
    assert_eq!(history.messages().await.len(), 1);
    assert_eq!(history.corruption(), None);

    // Corrupt the lane, observe the record, then clear it with a successful append.
    session
        .update_state(json!({MESSAGES_STATE_KEY: 7}))
        .await
        .unwrap();
    assert!(history.messages().await.is_empty());
    assert_eq!(
        history.corruption().expect("a corruption record").reason,
        HistoryCorruptionReason::NotAnArray
    );

    history.append(vec![BaseMessage::assistant("b")]).await;
    assert_eq!(history.corruption(), None);
    assert_eq!(history.messages().await.len(), 1);

    // A skipped-entry record is forgotten by a successful clear() too.
    session
        .update_state(json!({MESSAGES_STATE_KEY: [7, 8]}))
        .await
        .unwrap();
    assert!(history.messages().await.is_empty());
    assert_eq!(
        history.corruption().expect("a corruption record").reason,
        HistoryCorruptionReason::MalformedEntries
    );

    history.clear().await;
    assert_eq!(history.corruption(), None);
}

#[tokio::test]
async fn handles_over_one_session_share_the_record() {
    let session = common::session("history-shared");
    session
        .update_state(json!({MESSAGES_STATE_KEY: "not an array"}))
        .await
        .unwrap();
    let reader = SessionMessageHistory::new(Arc::clone(&session));
    let sibling = reader.clone();

    assert!(reader.messages().await.is_empty());
    assert_eq!(
        sibling.corruption().expect("shared record").reason,
        HistoryCorruptionReason::NotAnArray
    );
}

#[tokio::test]
async fn an_external_implementation_inherits_the_default_none() {
    let history = ExternalHistory;

    assert_eq!(history.corruption(), None);
    assert!(history.messages().await.is_empty());
    history.append(vec![BaseMessage::user("a")]).await;
    history.clear().await;
    assert_eq!(history.corruption(), None);
}

#[tokio::test]
async fn a_context_observes_its_store_corruption() {
    let session = common::session("history-context");
    session
        .update_state(json!({MESSAGES_STATE_KEY: "nope"}))
        .await
        .unwrap();
    let ctx = ComponentContext::new(Arc::clone(&session), "llm", "wf");

    // Nothing has read the lane yet, so there is nothing to report.
    assert_eq!(ctx.history_corruption(), None);

    assert!(ctx.history().messages().await.is_empty());

    let corruption = ctx
        .history_corruption()
        .expect("the context sees the record");
    assert_eq!(corruption.reason, HistoryCorruptionReason::NotAnArray);
    assert!(corruption.detail.contains(MESSAGES_STATE_KEY));

    let error = ctx.history_corrupt_error().expect("a coded error");
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.message().starts_with("[100060] "));

    // A successful write through the same context clears it.
    ctx.history().clear().await;
    assert_eq!(ctx.history_corruption(), None);
    assert!(ctx.history_corrupt_error().is_none());
}

#[tokio::test]
async fn an_overridden_store_hides_the_session_lane_corruption() {
    let session = common::session("history-override");
    session
        .update_state(json!({MESSAGES_STATE_KEY: 7}))
        .await
        .unwrap();
    let ctx = ComponentContext::new(session, "llm", "wf")
        .with_history(Arc::new(ExternalHistory) as Arc<dyn MessageHistory>);

    assert!(ctx.history().messages().await.is_empty());
    assert_eq!(ctx.history_corruption(), None);
    assert!(ctx.history_corrupt_error().is_none());
}
