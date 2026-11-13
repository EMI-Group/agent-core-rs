//! Non-model frames the loop publishes: context statistics and model routing.
//!
//! Both helpers emit through a
//! [`DeltaSink`](crate::react::model_step::DeltaSink), so a frame reaches the
//! streaming publisher (when the caller streams) and is mirrored into the
//! session exactly like the model's own `llm_*` frames.
//!
//! The kind constants live in [`crate::output`]: `context.usage`
//! ([`CONTEXT_USAGE`]) after every successful window preparation and `llm_route`
//! ([`LLM_ROUTE`]) when a rail swapped the model for this call.
use crate::output::{AgentChunk, CONTEXT_USAGE, LLM_ROUTE};
use crate::react::model_step::DeltaSink;
use openjiuwen_context::PreparedContext;
use openjiuwen_foundation::ModelRequestConfig;
use openjiuwen_session::Session;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

/// Publishes one `context.usage` frame for the prepared window.
///
/// Payload shape:
///
/// ```json
/// {
///   "iteration": 1,
///   "message_count": 3,
///   "tool_count": 2,
///   "token_count": 128,
///   "overflow": false,
///   "window_size": null,
///   "result_type": "answer"
/// }
/// ```
///
/// `message_count`/`tool_count` count what will be sent, `token_count` and
/// `overflow` describe the prepared messages, and `window_size` is `null` when
/// no window limit applied.
pub(crate) async fn context_usage(
    publisher: Option<&UnboundedSender<AgentChunk>>,
    session: &Arc<dyn Session>,
    iteration: usize,
    prepared: &PreparedContext,
) {
    let payload = json!({
        "iteration": iteration,
        "message_count": prepared.messages.len(),
        "tool_count": prepared.tools.len(),
        "token_count": prepared.token_count,
        "overflow": prepared.overflow,
        "window_size": prepared.window_size,
        "result_type": "answer",
    });
    let mut sink = DeltaSink::new(publisher, Some(session));
    sink.emit(CONTEXT_USAGE, payload).await;
}

/// Publishes one `llm_route` frame announcing a non-primary model.
///
/// Payload shape:
///
/// ```json
/// {
///   "route": "backup",
///   "model": "gpt-4o",
///   "reason": "model_override",
///   "result_type": "answer"
/// }
/// ```
///
/// `reason` is `null` when the caller did not supply one; `result_type` keeps
/// the frames of a turn uniformly shaped.
pub(crate) async fn llm_route(
    publisher: Option<&UnboundedSender<AgentChunk>>,
    session: &Arc<dyn Session>,
    request: &ModelRequestConfig,
    route: &str,
    reason: Option<&str>,
) {
    let payload = json!({
        "route": route,
        "model": request.model_name,
        "reason": reason,
        "result_type": "answer",
    });
    let mut sink = DeltaSink::new(publisher, Some(session));
    sink.emit(LLM_ROUTE, payload).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_context::PreparedContext;
    use openjiuwen_foundation::BaseMessage;
    use openjiuwen_session::{AgentSession, StreamEvent};
    use std::time::Duration;

    fn prepared() -> PreparedContext {
        PreparedContext {
            messages: vec![BaseMessage::user("hi")],
            tools: vec![json!({"type": "function"})],
            token_count: 7,
            overflow: true,
            window_size: Some(4),
            ..PreparedContext::default()
        }
    }

    #[tokio::test]
    async fn context_usage_payload_and_session_mirror() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let mut events = session.subscribe();
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        context_usage(Some(&publisher), &session, 3, &prepared()).await;

        let frame = frames.recv().await.expect("a frame is published");
        assert_eq!(frame.kind, CONTEXT_USAGE);
        assert_eq!(frame.payload["iteration"], 3);
        assert_eq!(frame.payload["message_count"], 1);
        assert_eq!(frame.payload["tool_count"], 1);
        assert_eq!(frame.payload["token_count"], 7);
        assert_eq!(frame.payload["overflow"], true);
        assert_eq!(frame.payload["window_size"], 4);
        assert_eq!(frame.payload["result_type"], "answer");

        let event = tokio::time::timeout(Duration::from_millis(500), events.next())
            .await
            .expect("the session mirror receives the frame")
            .expect("the stream is open")
            .expect("the frame is valid");
        match event {
            StreamEvent::Item(item) => {
                assert_eq!(item.item_type, CONTEXT_USAGE);
                assert_eq!(item.payload["iteration"], 3);
            }
            StreamEvent::End => panic!("expected an item event, got End"),
        }
    }

    #[tokio::test]
    async fn context_usage_window_size_is_null_when_unset() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        let prepared = PreparedContext::default();
        context_usage(Some(&publisher), &session, 1, &prepared).await;
        let frame = frames.recv().await.expect("a frame is published");
        assert!(frame.payload["window_size"].is_null());
    }

    #[tokio::test]
    async fn llm_route_payload() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        let request = ModelRequestConfig::new("gpt-4o");
        llm_route(
            Some(&publisher),
            &session,
            &request,
            "backup",
            Some("model_override"),
        )
        .await;
        let frame = frames.recv().await.expect("a frame is published");
        assert_eq!(frame.kind, LLM_ROUTE);
        assert_eq!(frame.payload["route"], "backup");
        assert_eq!(frame.payload["model"], "gpt-4o");
        assert_eq!(frame.payload["reason"], "model_override");
        assert_eq!(frame.payload["result_type"], "answer");
    }

    #[tokio::test]
    async fn llm_route_reason_is_null_when_unset() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        llm_route(
            Some(&publisher),
            &session,
            &ModelRequestConfig::new("m"),
            "backup",
            None,
        )
        .await;
        let frame = frames.recv().await.expect("a frame is published");
        assert!(frame.payload["reason"].is_null());
    }
}
