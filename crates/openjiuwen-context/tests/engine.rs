#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for [`ContextEngine`] and the context contract.
//!
//! These tests exercise the crate root the way an agent or the harness does:
//! creating contexts in a per-session pool, preparing request payloads, observing
//! messages, compressing, recovering from provider overflows, persisting contexts
//! through a `Session`, and driving the summarisation processor end to end.
//!
//! Every test is offline and deterministic: token counts come from the
//! `characters` counter with a ratio of one (one character plus framing per
//! message), the only model is `ScriptedModel`, and sessions are in-process
//! `AgentSession`s. Unit tests inside `src/` cover the same building blocks at
//! module scope; these tests pin the behavior callers see through the public
//! surface.

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::Error;
use openjiuwen_context::engine::DEFAULT_SESSION_ID;
use openjiuwen_context::{
    CompressionStatus, ContextConfig, ContextEngine, ContextEventKind, ContextOptions,
    InMemoryOffloadStore, ModelContext, OffloadKind, OffloadScope, PrepareOptions, ProcessorConfig,
    SummarizationConfig, TokenCounterConfig, TokenCounterKind, TruncationConfig,
    CONTEXT_SUMMARY_PREFIX,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;

/// A deterministic configuration: one character per token.
fn char_config() -> ContextConfig {
    ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    }
}

/// An engine with the deterministic configuration and no pipeline surprises.
fn default_engine() -> ContextEngine {
    ContextEngine::new(char_config())
}

/// `count` complete user/assistant rounds (`q1`,`a1`, ...), six tokens each.
fn rounds(count: usize) -> Vec<BaseMessage> {
    let mut messages = Vec::new();
    for index in 1..=count {
        messages.push(BaseMessage::user(format!("q{index}")));
        messages.push(BaseMessage::assistant(format!("a{index}")));
    }
    messages
}

/// Three rounds whose combined history is too large for the small test budget.
fn compressible_history() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("first question"),
        BaseMessage::assistant("first answer"),
        BaseMessage::user("second question"),
        BaseMessage::assistant("second answer"),
        BaseMessage::user("third question"),
        BaseMessage::assistant("third answer"),
    ]
}

/// An engine whose truncation processor keeps only the newest two messages.
fn truncating_engine() -> ContextEngine {
    ContextEngine::new(ContextConfig {
        max_context_tokens: Some(100),
        compression_threshold: 0.5,
        processors: vec![ProcessorConfig::Truncation(TruncationConfig {
            max_tokens: Some(20),
            keep_recent_messages: 2,
            keep_system: true,
        })],
        ..char_config()
    })
}

/// Asserts the leading roles/texts of `messages`.
fn assert_prefix(messages: &[BaseMessage], expected: &[(&str, Role)]) {
    assert_eq!(messages.len(), expected.len(), "{messages:?}");
    for (message, (text, role)) in messages.iter().zip(expected) {
        assert_eq!(message.role, *role);
        assert_eq!(message.text(), *text);
    }
}

#[tokio::test]
async fn with_store_exposes_the_configuration_and_offload_scope() {
    let store = Arc::new(InMemoryOffloadStore::new());
    let engine = ContextEngine::with_store(
        ContextConfig {
            compression_threshold: 0.5,
            ..char_config()
        },
        store.clone(),
    );

    assert_eq!(engine.config().compression_threshold, 0.5);
    assert_eq!(
        engine.config().token_counter.kind,
        TokenCounterKind::Characters
    );

    let context = engine.create_context("main", None).await.unwrap();
    assert_eq!(context.offload_store().kind(), OffloadKind::InMemory);
    assert_eq!(
        context.scope(),
        OffloadScope::new(DEFAULT_SESSION_ID, "main")
    );
    assert_eq!(store.len(), 0);
}

#[tokio::test]
async fn prepare_places_system_messages_first_and_keeps_history_untouched() {
    let engine = default_engine();
    let history = rounds(2);
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: history.clone(),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    let options = PrepareOptions {
        system_messages: vec![BaseMessage::system("persona"), BaseMessage::system("rules")],
        ..PrepareOptions::default()
    };
    let prepared = engine.prepare(&context, &options).await.unwrap();

    // System messages come first, then the stored context in order.
    assert_prefix(
        &prepared.messages,
        &[
            ("persona", Role::System),
            ("rules", Role::System),
            ("q1", Role::User),
            ("a1", Role::Assistant),
            ("q2", Role::User),
            ("a2", Role::Assistant),
        ],
    );
    assert_eq!(prepared.stats.system_messages, 2);
    assert_eq!(prepared.stats.total_dialogues, 2);
    assert!(!prepared.overflow);
    // Two system messages (11 + 9 tokens) plus four six-token messages.
    assert_eq!(prepared.token_count, 44);

    // `prepare` is read-only: the stored history is exactly what was given.
    assert_eq!(context.messages().await, history);
}

#[tokio::test]
async fn prepare_honours_the_dialogue_round_window() {
    let engine = default_engine();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: rounds(3),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    let prepared = engine
        .prepare(
            &context,
            &PrepareOptions {
                dialogue_rounds: Some(2),
                ..PrepareOptions::default()
            },
        )
        .await
        .unwrap();

    assert_prefix(
        &prepared.messages,
        &[
            ("q2", Role::User),
            ("a2", Role::Assistant),
            ("q3", Role::User),
            ("a3", Role::Assistant),
        ],
    );
    assert_eq!(prepared.window_size, None);
    assert_eq!(context.messages().await.len(), 6);
}

#[tokio::test]
async fn prepare_honours_the_message_window_size_with_systems_in_the_budget() {
    let engine = default_engine();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: rounds(3),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    let prepared = engine
        .prepare(
            &context,
            &PrepareOptions {
                system_messages: vec![BaseMessage::system("persona")],
                window_size: Some(3),
                ..PrepareOptions::default()
            },
        )
        .await
        .unwrap();

    // One system message consumes one slot, leaving room for the newest two.
    assert_prefix(
        &prepared.messages,
        &[
            ("persona", Role::System),
            ("q3", Role::User),
            ("a3", Role::Assistant),
        ],
    );
    assert_eq!(prepared.window_size, Some(3));
}

#[tokio::test]
async fn prepare_passes_tool_schemas_through_unchanged() {
    let engine = default_engine();
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("look it up"))
        .await
        .unwrap();

    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look something up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }
    })];
    let prepared = engine
        .prepare(
            &context,
            &PrepareOptions {
                tools: tools.clone(),
                ..PrepareOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(prepared.tools, tools);
    assert_eq!(prepared.messages.len(), 1);
    assert_eq!(prepared.messages[0].text(), "look it up");
}

#[tokio::test]
async fn prepare_reports_overflow_only_above_the_effective_budget() {
    let engine = ContextEngine::new(ContextConfig {
        max_context_tokens: Some(20),
        processors: Vec::new(),
        ..char_config()
    });

    let small = engine
        .create_context_with(
            "small",
            ContextOptions {
                history: vec![BaseMessage::user("hi")],
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    let prepared = engine
        .prepare(&small, &PrepareOptions::default())
        .await
        .unwrap();
    assert_eq!(prepared.token_count, 6);
    assert!(!prepared.overflow);

    let large = engine
        .create_context_with(
            "large",
            ContextOptions {
                history: vec![BaseMessage::user("x".repeat(30))],
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    let prepared = engine
        .prepare(&large, &PrepareOptions::default())
        .await
        .unwrap();
    assert_eq!(prepared.token_count, 34);
    assert!(prepared.overflow);
}

#[tokio::test]
async fn prepare_with_a_tiny_timeout_completes_before_the_deadline() {
    let engine = default_engine();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: rounds(3),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    // Preparation of an in-memory history never yields, so a one-millisecond
    // deadline is not reached: the future completes on its first poll. The
    // success path is therefore the deterministic outcome; a deadline that did
    // fire would surface as `Error::Timeout`.
    let prepared = engine
        .prepare(
            &context,
            &PrepareOptions {
                timeout: Some(Duration::from_millis(1)),
                ..PrepareOptions::default()
            },
        )
        .await
        .expect("in-memory preparation completes without yielding");
    assert_eq!(prepared.messages.len(), 6);
    assert!(prepared.token_count > 0);
}

#[tokio::test]
async fn prepare_rejects_cancelled_requests() {
    let engine = default_engine();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: rounds(1),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    // A caller-cancelled request is refused before any work runs.
    let cancelled = PrepareOptions::default();
    cancelled.cancellation.cancel();
    assert!(matches!(
        engine.prepare(&context, &cancelled).await.unwrap_err(),
        Error::Cancelled
    ));

    // Once the context itself is cancelled, every later request is refused too.
    context.cancel();
    assert!(matches!(
        engine
            .prepare(&context, &PrepareOptions::default())
            .await
            .unwrap_err(),
        Error::Cancelled
    ));
}

#[tokio::test]
async fn observe_reports_added_evicted_and_token_counts() {
    let engine = default_engine();
    let context = engine.create_context("main", None).await.unwrap();

    let outcome = engine
        .observe(&context, BaseMessage::user("abcd"))
        .await
        .unwrap();
    assert_eq!(outcome.added, 1);
    assert_eq!(outcome.evicted, 0);
    assert_eq!(outcome.total_messages, 1);
    // "abcd" is four characters plus four framing tokens.
    assert_eq!(outcome.total_tokens, 8);

    let outcome = engine
        .observe_tool_result(&context, "call_1", "xy")
        .await
        .unwrap();
    assert_eq!(outcome.added, 1);
    assert_eq!(outcome.evicted, 0);
    assert_eq!(outcome.total_messages, 2);
    // "xy" is two characters plus framing.
    assert_eq!(outcome.total_tokens, 14);

    let messages = context.messages().await;
    assert_eq!(messages[1].role, Role::Tool);
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(messages[1].text(), "xy");
}

#[tokio::test]
async fn observe_evicts_to_the_rolling_limit_and_keeps_systems_and_the_newest() {
    let engine = ContextEngine::new(ContextConfig {
        max_context_messages: Some(3),
        ..char_config()
    });
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: vec![
                    BaseMessage::system("persona A"),
                    BaseMessage::user("q1"),
                    BaseMessage::system("persona B"),
                    BaseMessage::assistant("a1"),
                ],
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    let outcome = engine
        .observe(&context, BaseMessage::user("q2"))
        .await
        .unwrap();
    // Both oldest non-system messages go; nothing else may be evicted.
    assert_eq!(outcome.evicted, 2);
    assert_eq!(outcome.total_messages, 3);

    let messages = context.messages().await;
    assert_prefix(
        &messages,
        &[
            ("persona A", Role::System),
            ("persona B", Role::System),
            ("q2", Role::User),
        ],
    );
    assert!(outcome.total_tokens > 0);
}

#[tokio::test]
async fn create_context_is_idempotent_and_normalizes_dotted_ids() {
    let engine = default_engine();
    let first = engine.create_context("main.thread", None).await.unwrap();
    let second = engine.create_context("main.thread", None).await.unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.context_id(), "main.thread");
    assert_eq!(first.session_id(), DEFAULT_SESSION_ID);

    // Dots and underscores are the same pool entry, but the original id sticks.
    let underscored = engine.create_context("main_thread", None).await.unwrap();
    assert!(Arc::ptr_eq(&first, &underscored));
    assert_eq!(underscored.context_id(), "main.thread");
}

#[tokio::test]
async fn contexts_with_the_same_id_are_isolated_per_session() {
    let engine = default_engine();
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
    assert!(bob_context.session().is_some());

    engine
        .observe(&alice_context, BaseMessage::user("alice only"))
        .await
        .unwrap();
    assert_eq!(bob_context.messages().await.len(), 0);

    // The pool lookup is keyed by session as well.
    let found = engine.get_context("alice", "main").await.unwrap();
    assert!(Arc::ptr_eq(&alice_context, &found));
    assert!(engine.get_context("carol", "main").await.is_none());
}

#[tokio::test]
async fn get_and_remove_context_manage_the_pool() {
    let engine = default_engine();
    let context = engine.create_context("main", None).await.unwrap();

    let found = engine
        .get_context(DEFAULT_SESSION_ID, "main")
        .await
        .expect("the pool holds the context");
    assert!(Arc::ptr_eq(&context, &found));
    assert!(engine
        .get_context(DEFAULT_SESSION_ID, "missing")
        .await
        .is_none());

    let removed = engine
        .remove_context(DEFAULT_SESSION_ID, "main")
        .await
        .expect("the pool held the context");
    assert!(Arc::ptr_eq(&context, &removed));
    assert!(engine
        .get_context(DEFAULT_SESSION_ID, "main")
        .await
        .is_none());
    assert!(engine
        .remove_context(DEFAULT_SESSION_ID, "main")
        .await
        .is_none());
}

#[tokio::test]
async fn compress_shrinks_the_history_then_reports_noop() {
    let engine = truncating_engine();
    let context = engine.create_context("main", None).await.unwrap();
    context.set_messages(compressible_history()).await.unwrap();

    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert_eq!(report.events.len(), 1);
    assert_eq!(report.events[0].processor, "truncation");
    assert_eq!(report.events[0].kind, ContextEventKind::Truncated);
    assert!(report.tokens_before > report.tokens_after);
    assert_eq!(report.messages, 2);

    // The persisted history really shrank to the newest round.
    let messages = context.messages().await;
    assert_prefix(
        &messages,
        &[
            ("third question", Role::User),
            ("third answer", Role::Assistant),
        ],
    );

    // The compression totals show up in the context statistics.
    let stats = context.statistics().await;
    assert_eq!(stats.compressions, 1);
    assert_eq!(stats.tokens_before_compression, report.tokens_before);
    assert_eq!(stats.tokens_after_compression, report.tokens_after);

    // A second pass has nothing left to remove.
    let second = engine.compress(&context).await.unwrap();
    assert_eq!(second.status, CompressionStatus::Noop);
    assert_eq!(second.events.len(), 0);
    assert_eq!(second.tokens_before, second.tokens_after);
    assert_eq!(second.messages, 2);
    assert_eq!(context.statistics().await.compressions, 1);
}

#[tokio::test]
async fn recover_skips_unrelated_errors() {
    let engine = truncating_engine();
    let context = engine.create_context("main", None).await.unwrap();
    context.set_messages(compressible_history()).await.unwrap();

    let outcome = engine
        .recover_from_model_exception(&context, "invalid api key", false, 0)
        .await
        .unwrap();
    assert!(!outcome.attempted);
    assert!(!outcome.recovered);
    assert!(outcome.report.is_none());
    // Nothing was compressed.
    assert_eq!(context.messages().await.len(), 6);
}

#[tokio::test]
async fn recover_skips_after_streamed_output_was_emitted() {
    let engine = truncating_engine();
    let context = engine.create_context("main", None).await.unwrap();
    context.set_messages(compressible_history()).await.unwrap();

    // Retrying after a visible prefix would duplicate it, so recovery declines
    // even though the error describes an overflow.
    let outcome = engine
        .recover_from_model_exception(
            &context,
            "This model's maximum context length is 8192 tokens",
            true,
            3,
        )
        .await
        .unwrap();
    assert!(!outcome.attempted);
    assert!(!outcome.recovered);
    assert!(outcome.report.is_none());
    assert_eq!(context.messages().await.len(), 6);
}

#[tokio::test]
async fn recover_compresses_after_an_overflow() {
    let engine = truncating_engine();
    let context = engine.create_context("main", None).await.unwrap();
    context.set_messages(compressible_history()).await.unwrap();

    let outcome = engine
        .recover_from_model_exception(
            &context,
            "context_length_exceeded: prompt is too long",
            false,
            0,
        )
        .await
        .unwrap();
    assert!(outcome.attempted);
    assert!(outcome.recovered);

    let report = outcome.report.expect("a recovery report is attached");
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert!(report.tokens_before > report.tokens_after);
    assert_eq!(context.messages().await.len(), 2);
}

#[tokio::test]
async fn recover_attempts_but_does_not_recover_when_compression_is_a_noop() {
    let engine = truncating_engine();
    let context = engine.create_context("main", None).await.unwrap();
    // Two short messages fit the truncation budget, so nothing can be removed.
    context
        .set_messages(vec![BaseMessage::user("hi"), BaseMessage::assistant("ok")])
        .await
        .unwrap();

    let outcome = engine
        .recover_from_model_exception(&context, "prompt is too long", false, 0)
        .await
        .unwrap();
    assert!(outcome.attempted);
    assert!(!outcome.recovered);

    let report = outcome.report.expect("a recovery report is attached");
    assert_eq!(report.status, CompressionStatus::Noop);
    assert_eq!(context.messages().await.len(), 2);
}

#[test]
fn overflow_phrases_are_classified_through_the_public_helper() {
    let overflow = [
        "This model's maximum context length is 8192 tokens",
        "the request exceeded the CONTEXT WINDOW",
        "maximum context reached",
        "context limit reached",
        "context_length_exceeded",
        "prompt is too long: 210000 tokens",
        "prompt too long",
        "input is too long for the model",
        "input too long",
        "Range of input length should be [1, 1000000]",
        "token limit reached",
    ];
    for phrase in overflow {
        assert!(
            ContextEngine::is_context_overflow_error(phrase),
            "expected an overflow: {phrase}"
        );
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
            "expected no overflow: {phrase}"
        );
    }
}

#[tokio::test]
async fn save_and_load_contexts_round_trip_through_a_session() {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
    let engine = default_engine();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                session: Some(Arc::clone(&session)),
                history: rounds(2),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    engine
        .observe(&context, BaseMessage::user("tail"))
        .await
        .unwrap();

    assert_eq!(engine.save_contexts(session.as_ref()).await.unwrap(), 1);

    let restored_engine = default_engine();
    assert_eq!(
        restored_engine
            .load_contexts(session.as_ref())
            .await
            .unwrap(),
        1
    );
    let restored = restored_engine
        .get_context("s", "main")
        .await
        .expect("the context was restored");
    assert_eq!(restored.messages().await, context.messages().await);
    assert_eq!(restored.revision(), context.revision());
    // A restored context carries no session handle; `load_contexts` only borrows.
    assert!(restored.session().is_none());

    // An unrelated session has nothing to save or load.
    let other = AgentSession::new("other");
    assert_eq!(engine.save_contexts(&other).await.unwrap(), 0);
    assert_eq!(restored_engine.load_contexts(&other).await.unwrap(), 0);
}

#[tokio::test]
async fn load_contexts_skips_corrupt_and_non_object_state() {
    let engine = default_engine();

    let session = AgentSession::new("s");
    session
        .update_state(json!({
            "context": {
                "good": {"messages": [], "revision": 3},
                "bad": {"messages": "not a list"}
            }
        }))
        .await
        .unwrap();
    // The corrupt entry is logged and skipped; the valid one still loads.
    assert_eq!(engine.load_contexts(&session).await.unwrap(), 1);
    let restored = engine
        .get_context("s", "good")
        .await
        .expect("the valid context was restored");
    assert_eq!(restored.revision(), 3);
    assert!(engine.get_context("s", "bad").await.is_none());

    // A session without context state restores nothing.
    let empty = AgentSession::new("empty");
    assert_eq!(engine.load_contexts(&empty).await.unwrap(), 0);

    // State that is not an object is skipped as a whole.
    let broken = AgentSession::new("broken");
    broken.update_state(json!({"context": 7})).await.unwrap();
    assert_eq!(engine.load_contexts(&broken).await.unwrap(), 0);
}

#[tokio::test]
async fn summarization_replaces_old_rounds_without_mutating_stored_history() {
    let model = Arc::new(ScriptedModel::builder().respond_text("summary").build());
    let engine = ContextEngine::new(ContextConfig {
        processors: vec![ProcessorConfig::Summarization(SummarizationConfig {
            tokens_threshold: Some(1),
            keep_recent_messages: 2,
            keep_last_rounds: 1,
            min_messages_to_summarize: 2,
            max_summary_tokens: 100,
        })],
        ..char_config()
    });
    let history = vec![
        BaseMessage::system("persona"),
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
    ];
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: history.clone(),
                model: Some(Arc::clone(&model) as Arc<dyn Model>),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();

    // The first two rounds collapse into one summary; the system message and
    // the newest round survive.
    assert_prefix(
        &prepared.messages,
        &[
            ("persona", Role::System),
            ("[CONTEXT_SUMMARY]\nsummary", Role::User),
            ("q3", Role::User),
            ("a3", Role::Assistant),
        ],
    );
    assert!(prepared.messages[1]
        .text()
        .starts_with(CONTEXT_SUMMARY_PREFIX));

    assert_eq!(prepared.events.len(), 1);
    assert_eq!(prepared.events[0].processor, "summarization");
    assert_eq!(prepared.events[0].kind, ContextEventKind::Summarized);
    assert_eq!(prepared.events[0].summary.as_deref(), Some("summary"));

    // Exactly one model call, and the engine's stored history is untouched.
    assert_eq!(model.call_count(), 1);
    assert_eq!(context.messages().await, history);
}

#[tokio::test]
async fn model_context_trait_manages_the_history() {
    let engine = default_engine();
    let context = engine.create_context("main", None).await.unwrap();

    context.add_message(BaseMessage::user("q1")).await.unwrap();
    context
        .add_message(BaseMessage::assistant("a1"))
        .await
        .unwrap();
    assert_eq!(context.revision(), 2);

    context
        .set_messages(vec![BaseMessage::user("only")])
        .await
        .unwrap();
    assert_eq!(context.messages().await.len(), 1);

    let popped = context.pop_messages(1).await.unwrap();
    assert_eq!(popped.len(), 1);
    assert_eq!(context.messages().await.len(), 0);

    context
        .add_message(BaseMessage::user("again"))
        .await
        .unwrap();
    context.clear_messages().await.unwrap();
    assert_eq!(context.messages().await.len(), 0);
    assert_eq!(context.statistics().await.total_messages, 0);
}
