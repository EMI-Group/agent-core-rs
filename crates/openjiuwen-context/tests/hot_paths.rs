#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! The per-turn cost of `prepare` and `observe`.
//!
//! Both entry points run once per model turn, so neither may pay for the whole
//! history: `prepare` must not re-tokenize the window `compute_stats` just
//! measured, and `observe` must neither clone nor re-count the stored history on
//! every append.
//!
//! The seam is a [`TokenCounter`] that records how many *messages* it counted, so
//! a full recount over `n` messages shows up as `n` recorded messages. The
//! rendered output is unchanged by either optimization: the statistics after `n`
//! appends still equal one fresh measurement of the final history.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_context::{
    count_message, count_messages, ContextConfig, ContextEngine, ContextOptions, ModelContext,
    PrepareOptions,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, TokenCounter};

/// A token counter that records every message it was asked to count.
///
/// Its formula is the trait's default one (four framing tokens plus the plain
/// text, plus every tool call's name and arguments), so it is additive per
/// message — the property an incremental append relies on.
struct CountingCounter {
    /// Messages counted so far.
    counted: AtomicUsize,
}

impl CountingCounter {
    /// A counter that has counted nothing yet.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            counted: AtomicUsize::new(0),
        })
    }

    /// How many messages were counted since the last [`Self::reset`].
    fn counted(&self) -> usize {
        self.counted.load(Ordering::SeqCst)
    }

    /// Forgets the recorded count.
    fn reset(&self) {
        self.counted.store(0, Ordering::SeqCst);
    }
}

/// The same formula without the recording: one character per token.
fn plain_counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

impl TokenCounter for CountingCounter {
    fn count_text(&self, text: &str) -> usize {
        text.chars().count()
    }
    fn count_messages(&self, messages: &[BaseMessage]) -> usize {
        self.counted.fetch_add(messages.len(), Ordering::SeqCst);
        messages
            .iter()
            .map(|message| {
                4 + self.count_text(&message.text())
                    + message
                        .tool_calls
                        .iter()
                        .map(|call| self.count_text(&call.name) + self.count_text(&call.arguments))
                        .sum::<usize>()
            })
            .sum()
    }
}

/// Options for a context counted by `counter` with `history` and no processors.
fn options(counter: &Arc<CountingCounter>, history: Vec<BaseMessage>) -> ContextOptions {
    ContextOptions {
        history,
        // No processors: the tests measure the engine's own counting, not a
        // processor's.
        processors: Some(Vec::new()),
        token_counter: Some(counter.clone()),
        ..ContextOptions::default()
    }
}

/// `count` complete user/assistant rounds.
fn rounds(count: usize) -> Vec<BaseMessage> {
    let mut messages = Vec::new();
    for index in 0..count {
        messages.push(BaseMessage::user(format!("question {index}")));
        messages.push(BaseMessage::assistant(format!("answer {index}")));
    }
    messages
}

#[tokio::test]
async fn prepare_does_not_recount_the_window_it_measured() {
    let counter = CountingCounter::new();
    let engine = ContextEngine::new(ContextConfig::default());
    let history = rounds(6);
    let context = engine
        .create_context_with("main", options(&counter, history.clone()))
        .await
        .unwrap();
    counter.reset();

    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();

    // The reported total is the one `compute_stats` measured, so output is
    // unchanged whatever the counting strategy.
    assert_eq!(prepared.token_count, prepared.stats.total_tokens);
    assert_eq!(
        prepared.token_count,
        count_messages(&plain_counter(), &history)
    );
    // `compute_stats` measures every message twice — once for the total, once per
    // role — and the third pass `prepare` used to run on top of it is gone.
    let recorded = counter.counted();
    assert!(
        recorded <= 2 * history.len(),
        "prepare counted {recorded} messages for a {} message window; a third \
         pass over the window was reintroduced",
        history.len()
    );
}

#[tokio::test]
async fn observing_two_hundred_messages_does_not_copy_or_recount_the_history() {
    let counter = CountingCounter::new();
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine
        .create_context_with("main", options(&counter, Vec::new()))
        .await
        .unwrap();
    counter.reset();

    let mut expected: Vec<BaseMessage> = Vec::new();
    for index in 0..200_usize {
        let message = if index % 2 == 0 {
            BaseMessage::user(format!("question {index}"))
        } else {
            BaseMessage::assistant(format!("answer {index}"))
        };
        expected.push(message.clone());
        let outcome = engine.observe(&context, message).await.unwrap();
        assert_eq!(outcome.added, 1);
        assert_eq!(outcome.evicted, 0);
        assert_eq!(outcome.total_messages, index + 1);
    }

    // The history is the full ordered transcript, and appending mutated it in
    // place: one revision per append, never the two a `messages()` +
    // `set_messages()` round trip would leave behind.
    assert_eq!(context.messages().await, expected);
    assert_eq!(context.revision(), 200);

    let stats = context.statistics().await;
    assert_eq!(stats.total_messages, 200);
    assert_eq!(stats.total_dialogues, 100);
    assert_eq!(stats.user_messages, 100);
    assert_eq!(stats.assistant_messages, 100);
    assert_eq!(stats.tool_messages, 0);
    assert_eq!(
        stats.user_tokens,
        expected
            .iter()
            .step_by(2)
            .map(|message| count_message(&plain_counter(), message))
            .sum::<usize>()
    );
    // The cached total is what a full measurement of the same history says.
    let full_recount = count_messages(&plain_counter(), &expected);
    assert_eq!(stats.total_tokens, full_recount);
    assert_eq!(
        stats.total_tokens,
        stats.system_tokens + stats.user_tokens + stats.assistant_tokens + stats.tool_tokens
    );

    // The seam: one count for the appended message per append, plus at most one
    // cold-cache measurement of the first history. Counting the whole history on
    // every append — what this path used to do — records 20 100 messages here.
    let recorded = counter.counted();
    assert!(
        recorded <= 4 * expected.len(),
        "appending {} messages counted {recorded} messages; a per-append recount \
         of the whole history was reintroduced",
        expected.len()
    );
}

#[tokio::test]
async fn eviction_keeps_the_cached_statistics_equal_to_a_full_measurement() {
    let counter = CountingCounter::new();
    let engine = ContextEngine::new(ContextConfig {
        max_context_messages: Some(5),
        ..ContextConfig::default()
    });
    let context = engine
        .create_context_with(
            "main",
            options(&counter, vec![BaseMessage::system("persona")]),
        )
        .await
        .unwrap();
    counter.reset();

    let mut last_evicted = 0;
    for index in 0..20_usize {
        let message = if index % 2 == 0 {
            BaseMessage::user(format!("question {index}"))
        } else {
            BaseMessage::assistant(format!("answer {index}"))
        };
        let outcome = engine.observe(&context, message).await.unwrap();
        last_evicted = outcome.evicted;
        // The system message plus the appends grow to the limit and stay there.
        assert_eq!(outcome.total_messages, (index + 2).min(5));
    }
    assert_eq!(last_evicted, 1);

    let history = context.messages().await;
    assert_eq!(history.len(), 5);
    assert_eq!(history[0].role, Role::System);
    let stats = context.statistics().await;
    assert_eq!(stats.total_messages, 5);
    // A context built from the same final history measures it from scratch; the
    // incrementally maintained cache has to agree on every field.
    let shadow = engine
        .create_context_with("shadow", options(&counter, history.clone()))
        .await
        .unwrap();
    let fresh = shadow.statistics().await;
    assert_eq!(stats.total_messages, fresh.total_messages);
    assert_eq!(stats.total_tokens, fresh.total_tokens);
    assert_eq!(stats.total_dialogues, fresh.total_dialogues);
    assert_eq!(stats.system_tokens, fresh.system_tokens);
    assert_eq!(stats.user_messages, fresh.user_messages);
    assert_eq!(stats.user_tokens, fresh.user_tokens);
    assert_eq!(stats.assistant_messages, fresh.assistant_messages);
    assert_eq!(stats.assistant_tokens, fresh.assistant_tokens);
    assert_eq!(stats.tool_messages, fresh.tool_messages);
    assert_eq!(stats.tool_tokens, fresh.tool_tokens);
    assert_eq!(
        stats.total_tokens,
        count_messages(&plain_counter(), &history)
    );
}

#[tokio::test]
async fn an_empty_user_message_is_rejected_without_touching_the_history() {
    let counter = CountingCounter::new();
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine
        .create_context_with("main", options(&counter, Vec::new()))
        .await
        .unwrap();

    let error = engine
        .observe(&context, BaseMessage::user(""))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(context.messages().await, Vec::<BaseMessage>::new());
    assert_eq!(context.revision(), 0);
    assert_eq!(context.statistics().await.total_tokens, 0);
}
