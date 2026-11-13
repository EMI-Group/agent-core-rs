//! Context-engine compression scenario: a long transcript is squeezed into a
//! bounded window and the following turn still reaches the model.
//!
//! This binary is the cross-crate seam between `openjiuwen-context`, the
//! kernel's prompt/model types and the `ReActAgent` loop: each crate's own tests
//! pin its module, while this suite drives the public engine the way an agent
//! does — `observe` the conversation, `prepare` the outgoing payload, `compress`
//! the stored history — and then runs the same configuration through a real
//! agent turn.
//!
//! Everything is offline and deterministic: the budget is counted in characters
//! (one character per token, plus four framing tokens per message), the only
//! model is a `ScriptedModel`, sessions are in-process, and assertions read back
//! recorded evidence (message counts, token counts, the model's call log)
//! instead of timestamps, durations or generated ids.
//!
//! Two engine facts shape the assertions below and are deliberate, not bugs:
//! `prepare` runs the processor pipeline on a *copy* and never persists, while
//! `compress` is the only call that rewrites the stored history, and it reports
//! [`CompressionStatus::Noop`] — not `Compressed` — when no processor changed
//! anything.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;
use openjiuwen::core::context_engine::{
    count_messages, CompressionStatus, ContextConfig, ContextEngine, ConversationContext,
    ModelContext, PrepareOptions, ProcessorConfig, TokenCounterConfig, TokenCounterKind,
    TruncationConfig,
};
use openjiuwen::core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen::core::foundation::prompt::BaseMessage;
use openjiuwen::core::single_agent::{Agent, AgentCard, ReActAgent, ReActAgentConfig, ResultType};
use openjiuwen_test_support::{agent_session, ScriptedModel};
use serde_json::json;
use std::sync::Arc;
use support::within;
/// Deterministic token accounting: one character per token, four framing tokens
/// per message (see `openjiuwen_foundation::prompt::ApproxTokenCounter`).
fn character_counting() -> TokenCounterConfig {
    TokenCounterConfig {
        kind: TokenCounterKind::Characters,
        chars_per_token: 1,
    }
}
/// The tiny engine used by every standalone scenario: a 160-token window, a
/// compression trigger at half of it, and a truncation processor that protects
/// only the newest two messages.
fn tiny_config() -> ContextConfig {
    ContextConfig {
        max_context_tokens: Some(160),
        compression_threshold: 0.5,
        token_counter: character_counting(),
        processors: vec![ProcessorConfig::Truncation(TruncationConfig {
            keep_recent_messages: 2,
            ..TruncationConfig::default()
        })],
        ..ContextConfig::default()
    }
}
/// `rounds` complete user/assistant rounds; each round is far wider than the
/// 160-token budget, so the truncation processor really has work to do.
fn transcript(rounds: usize) -> Vec<BaseMessage> {
    let mut history = Vec::with_capacity(rounds * 2);
    for index in 1..=rounds {
        history.push(BaseMessage::user(format!(
            "question {index}: the retriever keeps returning stale chunks."
        )));
        history.push(BaseMessage::assistant(format!(
            "answer {index}: re-index the corpus, then lower the threshold."
        )));
    }
    history
}
/// The stored history of `context`.
///
/// `ConversationContext` exposes its history through the [`ModelContext`] trait
/// only, so every read goes through it.
async fn stored(context: &Arc<ConversationContext>) -> Vec<BaseMessage> {
    ModelContext::messages(context.as_ref()).await
}
/// Tokens the stored history of `context` currently costs.
async fn stored_tokens(context: &Arc<ConversationContext>) -> usize {
    let counter = context.token_counter();
    count_messages(counter.as_ref(), &stored(context).await)
}
/// Observed evidence of test 1: a long transcript really is compressed into a
/// smaller stored history, and the report matches what was persisted.
#[tokio::test]
async fn a_long_transcript_compresses_into_a_smaller_stored_history() {
    let engine = ContextEngine::new(tiny_config());
    let context = within("create_context", engine.create_context("main", None))
        .await
        .expect("the tiny configuration is valid");
    // 20 rounds = 40 messages, roughly 130 tokens per round: well past the
    // 160-token budget.
    let history = transcript(20);
    for message in history {
        within("observe", engine.observe(&context, message))
            .await
            .expect("appending a message never fails for this history");
    }
    assert_eq!(stored(&context).await.len(), 40);
    let tokens_before = stored_tokens(&context).await;
    assert!(
        tokens_before > engine.config().effective_max_tokens(),
        "the seeded transcript ({tokens_before} tokens) must overflow the budget"
    );
    let report = within("compress", engine.compress(&context))
        .await
        .expect("compression runs without a model for the truncation processor");
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert!(report.is_compressed());
    assert_eq!(report.tokens_before, tokens_before);
    assert!(
        report.tokens_after < report.tokens_before,
        "compression must shrink the history ({} -> {})",
        report.tokens_before,
        report.tokens_after
    );
    // One processor ran, it was the configured truncation pass.
    assert_eq!(report.events.len(), 1);
    assert_eq!(report.events[0].processor, "truncation");
    // Only the newest round is protected, so the whole transcript collapses to it.
    assert_eq!(report.messages, 2);
    let messages = stored(&context).await;
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0].text(),
        "question 20: the retriever keeps returning stale chunks."
    );
    assert_eq!(
        messages[1].text(),
        "answer 20: re-index the corpus, then lower the threshold."
    );
    // The persisted history is smaller than what was observed, and the report's
    // token count is the count of what was really stored.
    assert_eq!(stored_tokens(&context).await, report.tokens_after);
    // A second pass has nothing left to remove.
    let second = within("compress", engine.compress(&context))
        .await
        .expect("a second pass still runs");
    assert_eq!(second.status, CompressionStatus::Noop);
    assert_eq!(
        second.events.len(),
        0,
        "a second pass has nothing to remove"
    );
    assert_eq!(stored(&context).await.len(), 2);
}
/// Observed evidence of test 2: the turn after a compression still produces a
/// non-empty, in-budget window that a model accepts.
#[tokio::test]
async fn the_turn_after_a_compression_still_reaches_the_model() {
    let engine = ContextEngine::new(tiny_config());
    let context = within("create_context", engine.create_context("main", None))
        .await
        .expect("the tiny configuration is valid");
    for message in transcript(20) {
        within("observe", engine.observe(&context, message))
            .await
            .expect("appending a message never fails for this history");
    }
    let report = within("compress", engine.compress(&context))
        .await
        .expect("the truncation pass needs no model");
    assert_eq!(report.status, CompressionStatus::Compressed);
    // The next turn arrives: an assistant reply (continued from the truncated
    // tail) and a fresh user message.
    for message in [
        BaseMessage::assistant("answer 20 continued: the cache is already warm."),
        BaseMessage::user("question 21: does the bounded window still work?"),
    ] {
        within("observe", engine.observe(&context, message))
            .await
            .expect("appending a message never fails for this history");
    }
    let history = stored(&context).await;
    assert_eq!(history.len(), 4);
    let prepared = within(
        "prepare",
        engine.prepare(&context, &PrepareOptions::default()),
    )
    .await
    .expect("preparing a window never fails for this history");
    assert!(
        !prepared.messages.is_empty(),
        "the window must carry a turn"
    );
    assert!(!prepared.overflow, "the window must fit the budget");
    assert!(
        prepared.token_count <= engine.config().effective_max_tokens(),
        "prepared {} tokens against a {} token budget",
        prepared.token_count,
        engine.config().effective_max_tokens()
    );
    assert!(
        prepared.messages.len() < history.len(),
        "the pipeline had to shrink {} stored messages",
        history.len()
    );
    let newest = "question 21: does the bounded window still work?";
    assert!(
        prepared.messages.iter().any(|m| m.text() == newest),
        "the newest user line must survive the window: {:?}",
        prepared.messages
    );
    // The window is a valid model request: the scripted model answers it.
    let model = ScriptedModel::text("ok");
    let request = ModelRequestConfig::new("ctx-engine-e2e");
    let reply = within("model invoke", model.invoke(&prepared.messages, &request))
        .await
        .expect("the prepared window is a valid model request");
    assert_eq!(
        reply
            .message()
            .expect("the scripted reply carries a message")
            .text(),
        "ok"
    );
    assert_eq!(model.call_count(), 1);
    let calls = model.calls();
    assert_eq!(calls[0].messages.len(), prepared.messages.len());
    assert_eq!(calls[0].last_user_text().as_deref(), Some(newest));
    // The stored history is untouched by the model call: compression persisted
    // two messages, the two observed turns brought it to four.
    assert_eq!(stored(&context).await.len(), 4);
}
/// Observed evidence of test 3: `prepare` previews a bounded window without
/// shrinking the stored history, while `compress` persists the same shrink.
#[tokio::test]
async fn prepare_previews_the_bounded_window_without_shrinking_the_history() {
    let engine = ContextEngine::new(tiny_config());
    let context = within("create_context", engine.create_context("main", None))
        .await
        .expect("the tiny configuration is valid");
    for message in transcript(20) {
        within("observe", engine.observe(&context, message))
            .await
            .expect("appending a message never fails for this history");
    }
    assert_eq!(stored(&context).await.len(), 40);
    let prepared = within(
        "prepare",
        engine.prepare(&context, &PrepareOptions::default()),
    )
    .await
    .expect("preparing a window never fails for this history");
    // The preview did real work on its copy ...
    assert!(
        !prepared.events.is_empty(),
        "the preview must truncate a copy"
    );
    assert!(prepared.messages.len() < 40);
    assert!(prepared.token_count <= engine.config().effective_max_tokens());
    // ... but the stored history still holds every observed message.
    assert_eq!(
        stored(&context).await.len(),
        40,
        "prepare is read-only and must not persist its window"
    );
    let report = within("compress", engine.compress(&context))
        .await
        .expect("the truncation pass needs no model");
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert_eq!(report.messages, 2);
    assert_eq!(
        stored(&context).await.len(),
        2,
        "compress is the only call that persists the shrink"
    );
    // The persisted pass stuck: another preview finds nothing left to do.
    let after = within(
        "prepare",
        engine.prepare(&context, &PrepareOptions::default()),
    )
    .await
    .expect("preparing a compressed history never fails");
    assert!(after.events.is_empty(), "the history is already in budget");
    assert_eq!(after.messages.len(), 2);
}
/// Observed evidence of test 4: a `ReActAgent` wired to the same tiny engine
/// completes two turns, and the second turn's window is compressed inside the
/// real loop while the injected configuration stays the agent's.
#[tokio::test]
async fn an_agent_run_compresses_a_tiny_window_between_two_turns() {
    let config = tiny_config();
    let mut agent_config = ReActAgentConfig::new();
    agent_config.configure_context_engine(config.clone());
    let agent = ReActAgent::new(AgentCard::new("ctx-engine-agent"));
    // `configure` rebuilds the engine and drops existing contexts, so it runs
    // before the first turn.
    agent.configure(agent_config);
    let model = ScriptedModel::builder()
        .respond_text("first answer")
        .respond_text("second answer")
        .build();
    agent.set_model(Arc::new(model.clone()));
    // The injected configuration is what the agent's engine runs with.
    let engine = agent.context_engine();
    assert_eq!(engine.config(), &config);
    assert_eq!(engine.config().max_context_tokens, Some(160));
    assert_eq!(
        engine.config().token_counter.kind,
        TokenCounterKind::Characters
    );
    let session = agent_session("ctx-engine-agent");
    let first_prompt =
        "turn one: the transcript is long enough that a tiny budget must truncate a message.";
    let second_prompt =
        "turn two: keep the newest round and drop the oldest one from the window now.";
    let first = within(
        "first turn",
        agent.invoke(json!(first_prompt), Some(Arc::clone(&session))),
    )
    .await
    .expect("the first turn completes against the scripted model");
    assert_eq!(first.result_type, ResultType::Answer);
    assert_eq!(first.output, "first answer");
    let second = within(
        "second turn",
        agent.invoke(json!(second_prompt), Some(Arc::clone(&session))),
    )
    .await
    .expect("the second turn completes against the scripted model");
    assert_eq!(second.result_type, ResultType::Answer);
    assert_eq!(second.output, "second answer");
    // Two calls, one per turn: neither turn needed a retry.
    assert_eq!(model.call_count(), 2);
    let calls = model.calls();
    assert_eq!(calls[1].last_user_text().as_deref(), Some(second_prompt));
    // The loop observed both turns into the pool context ...
    let context = within(
        "pool context",
        engine.create_context("main", Some(Arc::clone(&session))),
    )
    .await
    .expect("the run's context is still in the pool");
    let history = stored(&context).await;
    assert_eq!(
        history.len(),
        4,
        "both turns must be observed: {:?}",
        history.iter().map(BaseMessage::text).collect::<Vec<_>>()
    );
    // ... and the second turn's window was compressed: the stored history now
    // overflows the budget while the prepared payload fits it.
    let tokens = stored_tokens(&context).await;
    assert!(
        tokens > engine.config().effective_max_tokens(),
        "the stored history ({tokens} tokens) must overflow the tiny budget"
    );
    let prepared = within(
        "prepare",
        engine.prepare(&context, &PrepareOptions::default()),
    )
    .await
    .expect("preparing the run's context never fails");
    assert!(!prepared.overflow);
    assert!(
        prepared.messages.len() < history.len(),
        "the pipeline must drop a round from the {} stored messages",
        history.len()
    );
    let prompt = prepared.messages.iter().any(|m| m.text() == second_prompt);
    assert!(
        prompt,
        "the newest prompt must survive: {:?}",
        prepared.messages
    );
}
