//! Unit tests for the conversation context's history, preparation and state.
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::Error;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role};
use tokio_util::sync::CancellationToken;

use super::*;

use crate::engine::PrepareOptions;
use crate::events::ContextEvent;
use crate::offload::InMemoryOffloadStore;
use crate::processor::{ContextProcessor, ProcessorConfig, ProcessorContext, TruncationConfig};

fn counter() -> Arc<dyn TokenCounter + Send + Sync> {
    Arc::new(ApproxTokenCounter::with_chars_per_token(1))
}

fn config() -> Arc<ContextConfig> {
    Arc::new(ContextConfig::default())
}

/// A config whose budget is small enough to overflow with a few messages.
fn small_config() -> Arc<ContextConfig> {
    Arc::new(ContextConfig {
        max_context_tokens: Some(10),
        ..ContextConfig::default()
    })
}

fn build(config: Arc<ContextConfig>, processors: Vec<ProcessorConfig>) -> ConversationContext {
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let pipeline = ContextPipeline::from_configs(&processors, None, Arc::clone(&store))
        .unwrap_or_else(|error| panic!("pipeline should build: {error}"));
    ConversationContext::new(
        "main",
        "session",
        config,
        counter(),
        pipeline,
        store,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

fn truncating(max_tokens: usize) -> ConversationContext {
    build(
        small_config(),
        vec![ProcessorConfig::Truncation(TruncationConfig {
            max_tokens: Some(max_tokens),
            keep_recent_messages: 2,
            keep_system: true,
        })],
    )
}

/// A processor that sleeps so a deadline can expire mid-run.
struct Sleeper;

#[async_trait]
impl ContextProcessor for Sleeper {
    fn name(&self) -> &'static str {
        "sleeper"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok(None)
    }
}

#[tokio::test]
async fn prepare_does_not_mutate_the_stored_history() {
    let context = truncating(8);
    let messages = vec![
        BaseMessage::user("first question"),
        BaseMessage::assistant("first answer"),
        BaseMessage::user("second question"),
        BaseMessage::assistant("second answer"),
    ];
    context.set_messages(messages.clone()).await.unwrap();

    let prepared = context.prepare(&PrepareOptions::default()).await.unwrap();
    assert!(prepared.messages.len() < messages.len());
    assert!(prepared.overflow);
    assert_eq!(context.messages().await, messages);
}

#[tokio::test]
async fn prepare_puts_system_messages_first_and_honours_the_window() {
    let context = build(config(), Vec::new());
    context
        .set_messages(vec![
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
        ])
        .await
        .unwrap();

    let options = PrepareOptions {
        system_messages: vec![BaseMessage::system("persona")],
        dialogue_rounds: Some(2),
        window_size: Some(5),
        tools: vec![serde_json::json!({"name": "read_file"})],
        ..PrepareOptions::default()
    };
    let prepared = context.prepare(&options).await.unwrap();
    assert_eq!(prepared.messages.len(), 5);
    assert_eq!(prepared.messages[0].role, Role::System);
    assert_eq!(prepared.messages[0].text(), "persona");
    assert_eq!(prepared.messages[1].text(), "q1");
    assert_eq!(prepared.tools.len(), 1);
    assert_eq!(prepared.window_size, Some(5));
    assert!(!prepared.overflow);
    assert_eq!(prepared.stats.total_messages, 5);

    // A round window keeps only the newest round.
    let one_round = PrepareOptions {
        dialogue_rounds: Some(1),
        ..PrepareOptions::default()
    };
    let prepared = context.prepare(&one_round).await.unwrap();
    assert_eq!(prepared.messages.len(), 2);
    assert_eq!(prepared.messages[0].text(), "q2");
}

#[tokio::test]
async fn prepare_maps_an_expired_deadline_to_timeout() {
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let context = ConversationContext::new(
        "main",
        "session",
        config(),
        counter(),
        ContextPipeline::new(vec![Arc::new(Sleeper)]),
        store,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let options = PrepareOptions {
        timeout: Some(Duration::from_millis(1)),
        ..PrepareOptions::default()
    };
    let error = context.prepare(&options).await.unwrap_err();
    assert!(matches!(error, Error::Timeout(_)));
}

#[tokio::test]
async fn prepare_refuses_a_cancelled_request_or_context() {
    let context = build(config(), Vec::new());
    let cancelled = PrepareOptions {
        cancellation: CancellationToken::new(),
        ..PrepareOptions::default()
    };
    cancelled.cancellation.cancel();
    assert!(matches!(
        context.prepare(&cancelled).await.unwrap_err(),
        Error::Cancelled
    ));

    context.cancel();
    assert!(context.cancellation().is_cancelled());
    assert!(matches!(
        context
            .prepare(&PrepareOptions::default())
            .await
            .unwrap_err(),
        Error::Cancelled
    ));
}

#[tokio::test]
async fn history_mutations_bump_the_revision_and_keep_stats_fresh() {
    let context = build(config(), Vec::new());
    assert_eq!(context.revision(), 0);

    context
        .add_message(BaseMessage::user("abcd"))
        .await
        .unwrap();
    assert_eq!(context.revision(), 1);
    let stats = context.statistics().await;
    assert_eq!(stats.total_messages, 1);
    assert_eq!(stats.user_messages, 1);

    let cached = context.statistics().await;
    assert_eq!(cached, stats);

    context
        .add_message(BaseMessage::assistant("ef"))
        .await
        .unwrap();
    assert_eq!(context.revision(), 2);
    let stats = context.statistics().await;
    assert_eq!(stats.total_messages, 2);
    assert_eq!(stats.assistant_messages, 1);
    assert_eq!(stats.total_dialogues, 1);

    let popped = context.pop_messages(2).await.unwrap();
    assert_eq!(popped.len(), 2);
    assert_eq!(context.revision(), 3);
    assert_eq!(context.messages().await.len(), 0);

    context.add_message(BaseMessage::system("s")).await.unwrap();
    context.clear_messages().await.unwrap();
    assert_eq!(context.revision(), 5);
    assert_eq!(context.statistics().await.total_messages, 0);
    assert_eq!(context.messages().await.len(), 0);
}

#[tokio::test]
async fn pop_rejects_zero_and_clamps_to_the_history() {
    let context = build(config(), Vec::new());
    let error = context.pop_messages(0).await.unwrap_err();
    assert!(
        matches!(error, Error::Validation(message) if message == "pop size should be larger than 0")
    );

    context
        .set_messages(vec![BaseMessage::user("a"), BaseMessage::user("b")])
        .await
        .unwrap();
    let popped = context.pop_messages(9).await.unwrap();
    assert_eq!(popped.len(), 2);
    assert_eq!(popped[0].text(), "a");
    assert_eq!(context.messages().await.len(), 0);
    assert_eq!(context.pop_messages(1).await.unwrap().len(), 0);
}

#[tokio::test]
async fn empty_user_and_tool_messages_are_rejected_but_empty_assistants_are_not() {
    let context = build(config(), Vec::new());
    let error = context
        .add_message(BaseMessage::user(""))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
    assert!(context
        .add_message(BaseMessage::tool("call_1", ""))
        .await
        .is_err());
    context
        .add_message(BaseMessage::assistant(""))
        .await
        .unwrap();
    assert_eq!(context.messages().await.len(), 1);
}

#[tokio::test]
async fn state_round_trips_and_restores_the_revision() {
    let context = build(config(), Vec::new());
    context.add_message(BaseMessage::user("hi")).await.unwrap();
    let state = context.save_state().await;
    assert_eq!(state.revision, 1);
    let json = serde_json::to_string(&state).unwrap();
    let restored: ContextState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, state);

    let other = build(config(), Vec::new());
    other.load_state(restored).await.unwrap();
    assert_eq!(other.messages().await, state.messages);
    assert_eq!(other.revision(), 1);
    assert_eq!(other.statistics().await.total_messages, 1);
}

#[tokio::test]
async fn compression_totals_are_merged_into_statistics() {
    let context = build(config(), Vec::new());
    context
        .set_messages(vec![BaseMessage::user("abcd")])
        .await
        .unwrap();
    context.record_compression(40, 10).await;
    context.record_compression(10, 8).await;

    let stats = context.statistics().await;
    assert_eq!(stats.compressions, 2);
    assert_eq!(stats.tokens_before_compression, 50);
    assert_eq!(stats.tokens_after_compression, 18);
    assert_eq!(stats.total_messages, 1);
}

#[test]
fn context_identity_matches_the_constructor_arguments() {
    let context = build(config(), Vec::new());
    assert_eq!(context.context_id(), "main");
    assert_eq!(context.session_id(), "session");
    assert_eq!(context.scope().session_id, "session");
    assert_eq!(context.scope().context_id, "main");
    assert!(context.model().is_none());
    assert!(context.session().is_none());
    assert!(context.processors().is_empty());
    assert_eq!(
        context.offload_store().kind(),
        crate::offload::OffloadKind::InMemory
    );
    assert!(format!("{context:?}").contains("ConversationContext"));
}
