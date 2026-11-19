//! Non-model frames the loop publishes: context statistics and model routing.
//!
//! Both helpers emit through a
//! [`DeltaSink`](crate::react::model_step::DeltaSink), so a frame reaches the
//! streaming publisher (when the caller streams) and is mirrored into the
//! session exactly like the model's own `llm_*` frames.
//!
//! The kind constants live in [`crate::output`]: `context.usage`
//! ([`CONTEXT_USAGE`]) after every successful window preparation
//! ([`context_usage`]) and after every successful model call
//! ([`context_usage_post_call`]), and `llm_route` ([`LLM_ROUTE`]) when a rail
//! swapped the model for this call.
use crate::output::{AgentChunk, CONTEXT_USAGE, LLM_ROUTE};
use crate::react::model_step::DeltaSink;
use openjiuwen_common::error::Result;
use openjiuwen_context::{ContextUsageSnapshot, PreparedContext};
use openjiuwen_foundation::{ModelRequestConfig, Usage};
use openjiuwen_session::Session;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

/// The prepared-window fields a post-call `context.usage` frame repeats.
///
/// The post-call frame is emitted after the model call, when the prepared value
/// may already have been consumed for the request, so the loop copies the two
/// window fields it needs first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct WindowSummary {
    /// Whether the prepared window exceeded its token budget.
    pub(crate) overflow: bool,
    /// The window size actually applied, when one was requested.
    pub(crate) window_size: Option<usize>,
}

impl WindowSummary {
    /// Copies the summary of `prepared`.
    pub(crate) fn of(prepared: &PreparedContext) -> Self {
        Self {
            overflow: prepared.overflow,
            window_size: prepared.window_size,
        }
    }
}

/// Publishes one pre-call `context.usage` frame for the prepared window.
///
/// Payload shape:
///
/// ```json
/// {
///   "iteration": 1,
///   "phase": "pre_call",
///   "message_count": 3,
///   "tool_count": 2,
///   "token_count": 128,
///   "overflow": false,
///   "window_size": null,
///   "usage": { "type": "context.usage", "phase": "pre_call", … },
///   "result_type": "answer"
/// }
/// ```
///
/// `message_count`/`tool_count` count what will be sent, `token_count` and
/// `overflow` describe the prepared messages, and `window_size` is `null` when
/// no window limit applied. `usage` is the engine's measured snapshot of that
/// same window, or `null` when no
/// [`ContextUsageAnalyzer`](openjiuwen_context::ContextUsageAnalyzer) is
/// installed — the frame itself is emitted either way.
pub(crate) async fn context_usage(
    publisher: Option<&UnboundedSender<AgentChunk>>,
    session: &Arc<dyn Session>,
    iteration: usize,
    prepared: &PreparedContext,
) -> Result<()> {
    let payload = json!({
        "iteration": iteration,
        "phase": "pre_call",
        "message_count": prepared.messages.len(),
        "tool_count": prepared.tools.len(),
        "token_count": prepared.token_count,
        "overflow": prepared.overflow,
        "window_size": prepared.window_size,
        "usage": prepared.usage,
        "result_type": "answer",
    });
    let mut sink = DeltaSink::new(publisher, Some(session));
    sink.emit(CONTEXT_USAGE, payload).await
}

/// Publishes one post-call `context.usage` frame for a completed model call.
///
/// Payload shape:
///
/// ```json
/// {
///   "iteration": 1,
///   "phase": "post_call",
///   "input_tokens": 128,
///   "output_tokens": 16,
///   "total_tokens": 144,
///   "tokens": 128,
///   "overflow": false,
///   "window_size": null,
///   "usage": { "type": "context.usage", "phase": "post_call", … },
///   "result_type": "answer"
/// }
/// ```
///
/// `input_tokens`/`output_tokens`/`total_tokens` are the provider's report for
/// the call, `tokens` is the reconciled window total of `snapshot` (the provider
/// input total when the provider reported one), and `usage` is the whole
/// snapshot. The loop only calls this with a snapshot, so `usage` is never
/// `null` here, and it stamps the snapshot `phase = "post_call"` with the call's
/// request id before publishing it.
pub(crate) async fn context_usage_post_call(
    publisher: Option<&UnboundedSender<AgentChunk>>,
    session: &Arc<dyn Session>,
    iteration: usize,
    usage: &Usage,
    snapshot: &ContextUsageSnapshot,
    window: WindowSummary,
) -> Result<()> {
    let payload = json!({
        "iteration": iteration,
        "phase": "post_call",
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "tokens": snapshot.context_window.tokens,
        "overflow": window.overflow,
        "window_size": window.window_size,
        "usage": snapshot,
        "result_type": "answer",
    });
    let mut sink = DeltaSink::new(publisher, Some(session));
    sink.emit(CONTEXT_USAGE, payload).await
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
) -> Result<()> {
    let payload = json!({
        "route": route,
        "model": request.model_name,
        "reason": reason,
        "result_type": "answer",
    });
    let mut sink = DeltaSink::new(publisher, Some(session));
    sink.emit(LLM_ROUTE, payload).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_context::{ContextUsageSnapshot, ContextWindowUsage, PreparedContext};
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

    fn snapshot(tokens: u64) -> ContextUsageSnapshot {
        ContextUsageSnapshot {
            phase: "post_call".to_owned(),
            request_id: "s#1".to_owned(),
            context_window: ContextWindowUsage {
                tokens,
                ..ContextWindowUsage::default()
            },
            ..ContextUsageSnapshot::default()
        }
    }

    #[tokio::test]
    async fn context_usage_payload_and_session_mirror() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let mut events = session.subscribe();
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        context_usage(Some(&publisher), &session, 3, &prepared())
            .await
            .unwrap();

        let frame = frames.recv().await.expect("a frame is published");
        assert_eq!(frame.kind, CONTEXT_USAGE);
        assert_eq!(frame.payload["iteration"], 3);
        assert_eq!(frame.payload["phase"], "pre_call");
        assert_eq!(frame.payload["message_count"], 1);
        assert_eq!(frame.payload["tool_count"], 1);
        assert_eq!(frame.payload["token_count"], 7);
        assert_eq!(frame.payload["overflow"], true);
        assert_eq!(frame.payload["window_size"], 4);
        // No analyzer is attached to a bare `PreparedContext`, so the block is null.
        assert!(frame.payload["usage"].is_null());
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
                assert_eq!(item.payload["phase"], "pre_call");
            }
            StreamEvent::End => panic!("expected an item event, got End"),
        }
    }

    #[tokio::test]
    async fn context_usage_carries_the_measured_snapshot_when_one_exists() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        let mut prepared = prepared();
        prepared.usage = Some(snapshot(7));
        context_usage(Some(&publisher), &session, 1, &prepared)
            .await
            .unwrap();

        let frame = frames.recv().await.expect("a frame is published");
        assert_eq!(frame.payload["usage"]["context_window"]["tokens"], 7);
        // The measured block is the engine's own payload, unreshaped.
        assert_eq!(frame.payload["usage"]["phase"], "post_call");
        assert_eq!(frame.payload["usage"]["request_id"], "s#1");
    }

    #[tokio::test]
    async fn context_usage_window_size_is_null_when_unset() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        let prepared = PreparedContext::default();
        context_usage(Some(&publisher), &session, 1, &prepared)
            .await
            .unwrap();
        let frame = frames.recv().await.expect("a frame is published");
        assert!(frame.payload["window_size"].is_null());
    }

    #[tokio::test]
    async fn context_usage_post_call_payload_and_session_mirror() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let mut events = session.subscribe();
        let (publisher, mut frames) = tokio::sync::mpsc::unbounded_channel();
        let usage = openjiuwen_foundation::Usage::new(128, 16);
        let measured = snapshot(128);
        context_usage_post_call(
            Some(&publisher),
            &session,
            2,
            &usage,
            &measured,
            WindowSummary {
                overflow: true,
                window_size: Some(8),
            },
        )
        .await
        .unwrap();

        let frame = frames.recv().await.expect("a frame is published");
        assert_eq!(frame.kind, CONTEXT_USAGE);
        assert_eq!(frame.payload["iteration"], 2);
        assert_eq!(frame.payload["phase"], "post_call");
        assert_eq!(frame.payload["input_tokens"], 128);
        assert_eq!(frame.payload["output_tokens"], 16);
        assert_eq!(frame.payload["total_tokens"], 144);
        assert_eq!(frame.payload["tokens"], 128);
        assert_eq!(frame.payload["overflow"], true);
        assert_eq!(frame.payload["window_size"], 8);
        assert_eq!(frame.payload["usage"]["context_window"]["tokens"], 128);
        assert_eq!(frame.payload["result_type"], "answer");

        let event = tokio::time::timeout(Duration::from_millis(500), events.next())
            .await
            .expect("the session mirror receives the frame")
            .expect("the stream is open")
            .expect("the frame is valid");
        match event {
            StreamEvent::Item(item) => {
                assert_eq!(item.item_type, CONTEXT_USAGE);
                assert_eq!(item.payload["phase"], "post_call");
                assert_eq!(item.payload["input_tokens"], 128);
            }
            StreamEvent::End => panic!("expected an item event, got End"),
        }
    }

    #[test]
    fn window_summary_copies_the_prepared_window_fields() {
        let summary = WindowSummary::of(&prepared());
        assert!(summary.overflow);
        assert_eq!(summary.window_size, Some(4));
        assert_eq!(
            WindowSummary::of(&PreparedContext::default()),
            WindowSummary::default()
        );
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
        .await
        .unwrap();
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
        .await
        .unwrap();
        let frame = frames.recv().await.expect("a frame is published");
        assert!(frame.payload["reason"].is_null());
    }
}
