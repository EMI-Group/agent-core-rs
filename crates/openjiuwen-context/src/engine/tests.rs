//! Unit tests for the engine's pool, compression, usage and persistence seams.
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role};
use openjiuwen_session::{AgentSession, Session};

use super::*;

use crate::context::ModelContext;
use crate::events::CompressionStatus;
use crate::processor::{ProcessorConfig, TruncationConfig};

fn options() -> ContextOptions {
    ContextOptions {
        token_counter: Some(Arc::new(ApproxTokenCounter::with_chars_per_token(1))),
        ..ContextOptions::default()
    }
}

fn truncating_engine() -> ContextEngine {
    ContextEngine::new(ContextConfig {
        max_context_tokens: Some(8),
        processors: vec![ProcessorConfig::Truncation(TruncationConfig {
            max_tokens: Some(8),
            keep_recent_messages: 2,
            keep_system: true,
        })],
        ..ContextConfig::default()
    })
}

#[tokio::test]
async fn create_context_is_idempotent_and_normalizes_dots() {
    let engine = ContextEngine::new(ContextConfig::default());
    let first = engine.create_context("main.thread", None).await.unwrap();
    let second = engine.create_context("main.thread", None).await.unwrap();
    assert!(Arc::ptr_eq(&first, &second));

    let dotted = engine.create_context("main_thread", None).await.unwrap();
    assert!(Arc::ptr_eq(&first, &dotted));
    assert_eq!(first.context_id(), "main.thread");
    assert_eq!(first.session_id(), DEFAULT_SESSION_ID);

    let found = engine.get_context(DEFAULT_SESSION_ID, "main.thread").await;
    assert!(found.is_some());
    let removed = engine
        .remove_context(DEFAULT_SESSION_ID, "main.thread")
        .await;
    assert!(removed.is_some());
    assert!(engine
        .get_context(DEFAULT_SESSION_ID, "main.thread")
        .await
        .is_none());
    assert!(engine
        .remove_context(DEFAULT_SESSION_ID, "missing")
        .await
        .is_none());
}

#[tokio::test]
async fn sessions_isolate_contexts_with_the_same_id() {
    let engine = ContextEngine::new(ContextConfig::default());
    let alice: Arc<dyn Session> = Arc::new(AgentSession::new("alice"));
    let bob: Arc<dyn Session> = Arc::new(AgentSession::new("bob"));
    let alice_context = engine
        .create_context("main", Some(Arc::clone(&alice)))
        .await
        .unwrap();
    let bob_context = engine
        .create_context("main", Some(Arc::clone(&bob)))
        .await
        .unwrap();
    assert!(!Arc::ptr_eq(&alice_context, &bob_context));
    assert_eq!(alice_context.session_id(), "alice");
    assert_eq!(bob_context.session_id(), "bob");
    assert!(alice_context.session().is_some());
}

#[tokio::test]
async fn create_context_with_applies_history_and_options() {
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: vec![BaseMessage::user("q"), BaseMessage::assistant("a")],
                processors: Some(Vec::new()),
                ..options()
            },
        )
        .await
        .unwrap();
    assert_eq!(context.messages().await.len(), 2);
    assert!(context.processors().is_empty());
    assert_eq!(context.token_counter().count_text("abcd"), 4);

    let invalid = ContextEngine::new(ContextConfig {
        compression_threshold: 0.0,
        ..ContextConfig::default()
    });
    assert!(matches!(
        invalid.create_context("main", None).await.unwrap_err(),
        Error::Config(_)
    ));
}

#[tokio::test]
async fn prepare_delegates_and_honours_cancellation() {
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("hi"))
        .await
        .unwrap();

    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    assert_eq!(prepared.messages.len(), 1);

    let cancelled = PrepareOptions::default();
    cancelled.cancellation.cancel();
    assert!(matches!(
        engine.prepare(&context, &cancelled).await.unwrap_err(),
        Error::Cancelled
    ));
}

#[tokio::test]
async fn observe_counts_messages_and_tokens() {
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context_with("main", options()).await.unwrap();

    let outcome = engine
        .observe(&context, BaseMessage::user("abcd"))
        .await
        .unwrap();
    assert_eq!(outcome.added, 1);
    assert_eq!(outcome.evicted, 0);
    assert_eq!(outcome.total_messages, 1);
    assert_eq!(outcome.total_tokens, 8);

    let outcome = engine
        .observe_tool_result(&context, "call_1", "xy")
        .await
        .unwrap();
    assert_eq!(outcome.added, 1);
    assert_eq!(outcome.total_messages, 2);
    assert_eq!(context.messages().await[1].role, Role::Tool);
    assert_eq!(
        context.messages().await[1].tool_call_id.as_deref(),
        Some("call_1")
    );
}

#[tokio::test]
async fn observe_evicts_oldest_non_system_messages_but_keeps_the_newest() {
    let engine = ContextEngine::new(ContextConfig {
        max_context_messages: Some(3),
        ..ContextConfig::default()
    });
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: vec![
                    BaseMessage::system("persona"),
                    BaseMessage::user("q1"),
                    BaseMessage::assistant("a1"),
                ],
                ..options()
            },
        )
        .await
        .unwrap();

    let outcome = engine
        .observe(&context, BaseMessage::user("q2"))
        .await
        .unwrap();
    assert_eq!(outcome.evicted, 1);
    assert_eq!(outcome.total_messages, 3);
    let messages = context.messages().await;
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[1].text(), "a1");
    assert_eq!(messages[2].text(), "q2");

    // A history made only of system messages plus the new one cannot shrink
    // below the limit without dropping what must stay.
    let engine = ContextEngine::new(ContextConfig {
        max_context_messages: Some(1),
        ..ContextConfig::default()
    });
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: vec![BaseMessage::system("a"), BaseMessage::system("b")],
                ..options()
            },
        )
        .await
        .unwrap();
    let outcome = engine
        .observe(&context, BaseMessage::user("q"))
        .await
        .unwrap();
    assert_eq!(outcome.evicted, 0);
    assert_eq!(outcome.total_messages, 3);
}

#[tokio::test]
async fn compress_reports_compressed_then_noop_and_records_stats() {
    let engine = truncating_engine();
    let context = engine.create_context_with("main", options()).await.unwrap();
    context
        .set_messages(vec![
            BaseMessage::user("first question"),
            BaseMessage::assistant("first answer"),
            BaseMessage::user("second question"),
            BaseMessage::assistant("second answer"),
        ])
        .await
        .unwrap();

    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert_eq!(report.events.len(), 1);
    assert!(report.tokens_after < report.tokens_before);
    assert_eq!(report.messages, context.messages().await.len());

    let stats = context.statistics().await;
    assert_eq!(stats.compressions, 1);
    assert_eq!(stats.tokens_before_compression, report.tokens_before);
    assert_eq!(stats.tokens_after_compression, report.tokens_after);

    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Noop);
    assert_eq!(report.tokens_before, report.tokens_after);
    assert_eq!(context.statistics().await.compressions, 1);
}

#[tokio::test]
async fn compress_holds_the_lock_so_a_second_pass_reports_busy() {
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context("main", None).await.unwrap();
    // Hold the lock the engine's `compress` would take.
    let guard = context.compression_lock().try_lock();
    assert!(guard.is_ok());
    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Busy);
    assert_eq!(report.messages, 0);
    drop(guard);

    // With the lock released the pass runs again.
    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Noop);
}

#[tokio::test]
async fn recover_refuses_when_chunks_were_streamed_or_the_error_is_unrelated() {
    let engine = truncating_engine();
    let context = engine.create_context_with("main", options()).await.unwrap();
    context
        .set_messages(vec![
            BaseMessage::user("first question"),
            BaseMessage::assistant("first answer"),
            BaseMessage::user("second question"),
            BaseMessage::assistant("second answer"),
        ])
        .await
        .unwrap();

    let outcome = engine
        .recover_from_model_exception(&context, "rate limit exceeded", false, 0)
        .await
        .unwrap();
    assert!(!outcome.attempted);
    assert!(!outcome.recovered);
    assert!(outcome.report.is_none());
    assert_eq!(context.messages().await.len(), 4);

    let outcome = engine
        .recover_from_model_exception(&context, "context length exceeded", true, 2)
        .await
        .unwrap();
    assert!(!outcome.attempted);
    assert!(outcome.report.is_none());
    assert_eq!(context.messages().await.len(), 4);

    let outcome = engine
        .recover_from_model_exception(&context, "context length exceeded", false, 0)
        .await
        .unwrap();
    assert!(outcome.attempted);
    assert!(outcome.recovered);
    let report = outcome.report.unwrap();
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert!(context.messages().await.len() < 4);
}

#[tokio::test]
async fn recover_reports_an_ineffective_compression_as_not_recovered() {
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("hi"))
        .await
        .unwrap();

    let outcome = engine
        .recover_from_model_exception(&context, "maximum context length is 8192", false, 0)
        .await
        .unwrap();
    assert!(outcome.attempted);
    assert!(!outcome.recovered);
    assert_eq!(outcome.report.unwrap().status, CompressionStatus::Noop);
}

#[tokio::test]
async fn contexts_round_trip_through_session_state() {
    let engine = ContextEngine::new(ContextConfig::default());
    let session = AgentSession::new("s");
    let handle: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
    let context = engine
        .create_context("main.thread", Some(handle))
        .await
        .unwrap();
    engine
        .observe(&context, BaseMessage::user("hi"))
        .await
        .unwrap();
    engine
        .observe_tool_result(&context, "call_1", "42")
        .await
        .unwrap();

    assert_eq!(engine.save_contexts(&session).await.unwrap(), 1);
    let stored = session.get_state(STATE_KEY).await.unwrap().unwrap();
    assert!(stored.get("main.thread").is_some());

    let restored_engine = ContextEngine::new(ContextConfig::default());
    assert_eq!(restored_engine.load_contexts(&session).await.unwrap(), 1);
    let restored = restored_engine
        .get_context("s", "main.thread")
        .await
        .unwrap();
    assert_eq!(restored.messages().await, context.messages().await);
    assert_eq!(restored.revision(), context.revision());

    // A session with no contexts saves an empty map and restores nothing.
    let other = AgentSession::new("other");
    assert_eq!(engine.save_contexts(&other).await.unwrap(), 0);
    assert_eq!(restored_engine.load_contexts(&other).await.unwrap(), 0);
}

#[tokio::test]
async fn load_contexts_skips_corrupt_entries_and_missing_state() {
    let engine = ContextEngine::new(ContextConfig::default());
    let session = AgentSession::new("s");
    session
        .update_state(serde_json::json!({
            "context": {
                "good": {"messages": [], "revision": 3},
                "bad": {"messages": "not a list"}
            }
        }))
        .await
        .unwrap();

    assert_eq!(engine.load_contexts(&session).await.unwrap(), 1);
    let restored = engine.get_context("s", "good").await.unwrap();
    assert_eq!(restored.revision(), 3);

    let empty = AgentSession::new("empty");
    assert_eq!(engine.load_contexts(&empty).await.unwrap(), 0);

    let not_an_object = AgentSession::new("broken");
    not_an_object
        .update_state(serde_json::json!({"context": 7}))
        .await
        .unwrap();
    assert_eq!(engine.load_contexts(&not_an_object).await.unwrap(), 0);
}

#[test]
fn overflow_phrases_are_recognized_and_unrelated_errors_are_not() {
    let overflow = [
        "This model's maximum context length is 8192 tokens",
        "the request exceeded the CONTEXT WINDOW",
        "maximum context reached",
        "context limit reached",
        "prompt is too long: 210000 tokens",
        "prompt too long",
        "input is too long for the model",
        "input too long",
        "Range of input length should be [1, 1000000]",
        "token limit reached",
        "context_length_exceeded",
        "the prompt exceeded the limit",
    ];
    for phrase in overflow {
        assert!(ContextEngine::is_context_overflow_error(phrase), "{phrase}");
    }

    let unrelated = [
        "",
        "invalid api key",
        "rate limit exceeded",
        "connection reset by peer",
        "output token limit reached",
        "completion token limit reached",
        "response token limit reached",
        "too long",
    ];
    for phrase in unrelated {
        assert!(
            !ContextEngine::is_context_overflow_error(phrase),
            "{phrase}"
        );
    }
}

#[test]
fn engine_accessors_and_option_defaults() {
    let engine = ContextEngine::with_store(
        ContextConfig::default(),
        Arc::new(InMemoryOffloadStore::new()),
    );
    assert_eq!(engine.config().compression_threshold, 0.85);
    assert!(format!("{engine:?}").contains("ContextEngine"));
    assert!(format!("{:?}", options()).contains("ContextOptions"));

    let options = PrepareOptions::default();
    assert_eq!(options.system_messages.len(), 0);
    assert_eq!(options.tools.len(), 0);
    assert!(options.timeout.is_none());
    assert!(!options.cancellation.is_cancelled());
    assert_eq!(options.window_size, None);
}
