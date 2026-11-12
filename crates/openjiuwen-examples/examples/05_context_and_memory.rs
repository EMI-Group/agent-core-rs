//! `05_context_and_memory` — the context window, compression, and long-term memory.
//!
//! Teaches the two kernel services behind an agent's memory: the context engine
//! (`openjiuwen::core::context_engine`), which previews the payload sent to a
//! model (`prepare`), appends turns (`observe`), and compresses the *stored*
//! history when it outgrows the budget (`compress`); and long-term memory
//! (`openjiuwen::core::memory`), which stores facts per `(user, workspace)` scope
//! and recalls them by a blend of vector similarity and keyword relevance.
//!
//! Python counterparts: `examples/context_evolver/quickstart.py` and
//! `examples/graph_memory/showcase_graph_memory.py`. Everything is offline and
//! deterministic: a character-based token counter, `HashEmbedding` instead of an
//! embedding provider, and in-memory stores throughout.

#![allow(
    // Crate convention for the example binaries: every helper returns
    // `anyhow::Result`, and an `# Errors` section would only restate the docs of
    // the SDK call it wraps.
    clippy::missing_errors_doc
)]

mod common;

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context};
use openjiuwen::core::context_engine::{
    ContextConfig, ContextEngine, ContextOptions, ConversationContext, ModelContext,
    PrepareOptions, PreparedContext, ProcessorConfig, TokenCounterConfig, TokenCounterKind,
    TruncationConfig,
};
use openjiuwen::core::foundation::prompt::{BaseMessage, Role};
use openjiuwen::core::memory::{
    AddOptions, LongTermMemory, MemoryEngineConfig, MemoryScope, MemoryStores, NewMemory,
    SearchOptions,
};
use openjiuwen::core::retrieval::HashEmbedding;

use common::banner;

/// System prompt of the seeded conversation; never compressed or evicted.
const SYSTEM: &str = "Answer in one sentence.";

/// The seeded conversation, as `(user, assistant)` rounds.
///
/// Long enough that the character-based counter overflows the budget below, so
/// both the preview (`prepare`) and the persisted pass (`compress`) do work.
const ROUNDS: [(&str, &str); 5] = [
    (
        "Building a retrieval demo on this laptop.",
        "Start with the loader, then the chunks.",
    ),
    (
        "The chunks came back in the wrong order.",
        "Sort by document id, then chunk index.",
    ),
    (
        "Vector search finds nothing for short queries.",
        "Lower the threshold to 0.3.",
    ),
    (
        "The reranker is slow on a cold start.",
        "Warm it once, then keep it in the pipeline.",
    ),
    (
        "I still need a working memory layer.",
        "Records live in a user and workspace scope.",
    ),
];

/// The turn `observe` appends after the seed; its user half becomes the fact the
/// memory half writes back.
const OBSERVED_TURN: (&str, &str) = (
    "From now on, always run the test suite offline against a scripted model.",
    "Understood.",
);

/// Facts the memory half seeds, written as `(constructor, text)` pairs.
const FACTS: [(&str, &str); 3] = [
    ("user_profile", "Alice prefers dark mode in the editor."),
    (
        "semantic",
        "Alice keeps design notes under notes/ in the workspace.",
    ),
    (
        "episodic",
        "Alice reviewed the retrieval pull request on Friday.",
    ),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("05 context and memory");
    banner::note(&format!(
        "run mode: {}; every store is in memory and every embedding is deterministic",
        mode.describe()
    ));
    if mode.is_live() {
        banner::note("live mode changes nothing here: this example calls no model");
    }

    // The two halves are independent seams; the conversation half hands one user
    // line to the memory half so the write-back is genuinely derived from it.
    let statement = window_the_context().await?;
    remember(&statement).await?;

    banner::title("done");
    banner::answer("the stored history was compressed and the written-back fact was recalled");
    Ok(())
}

/// Sizes, observes, and compresses one conversation; returns the last user line.
async fn window_the_context() -> anyhow::Result<String> {
    banner::title("context window");

    let engine = build_engine();
    let history = seeded_history();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: history.clone(),
                ..ContextOptions::default()
            },
        )
        .await
        .context("failed to create the conversation context")?;
    banner::step(&format!(
        "seeded {} messages ({} user/assistant rounds)",
        history.len(),
        ROUNDS.len()
    ));

    // `observe` is the append hook: one message at a time, with the rolling
    // message limit applied afterwards (unset here, so `evicted` stays zero).
    let (user, assistant) = OBSERVED_TURN;
    for message in [BaseMessage::user(user), BaseMessage::assistant(assistant)] {
        let appended = engine
            .observe(&context, message)
            .await
            .context("failed to observe a message")?;
        banner::step(&format!(
            "observed one turn: total_messages {}, total_tokens {}, evicted {}",
            appended.total_messages, appended.total_tokens, appended.evicted
        ));
    }

    // `prepare` is a read-only preview: the pipeline runs on a copy, then the
    // window is cut, and the stored history keeps every message.
    let wide = preview(&engine, &context).await?;
    let statement = wide
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.text().into_owned())
        .ok_or_else(|| anyhow!("the prepared window holds no user message to remember"))?;
    banner::note(&format!("the user's latest statement: {statement}"));

    compress(&engine, &context).await?;
    Ok(statement)
}

/// Builds a context engine whose budget is small enough to force compression.
fn build_engine() -> ContextEngine {
    // 160 tokens at 1 token per character: the history below overflows, so the
    // preview already trims its copy and `compress` has a real pass to persist.
    let config = ContextConfig {
        max_context_tokens: Some(160),
        compression_threshold: 0.5,
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        // One processor, protecting only the newest two messages: the system
        // prompt plus the newest round then fit the budget.
        processors: vec![ProcessorConfig::Truncation(TruncationConfig {
            keep_recent_messages: 2,
            ..TruncationConfig::default()
        })],
        ..ContextConfig::default()
    };
    tracing::info!(
        budget = config.effective_max_tokens(),
        trigger = config.compression_trigger_tokens(),
        "built a context engine with a tiny budget"
    );
    ContextEngine::new(config)
}

/// Prepares a wide and a narrow window and prints what the preview selected.
async fn preview(
    engine: &ContextEngine,
    context: &Arc<ConversationContext>,
) -> anyhow::Result<PreparedContext> {
    let wide = prepare(engine, context, 6).await?;
    let narrow = prepare(engine, context, 2).await?;
    banner::step(&format!(
        "prepare(window {:?}): messages {}, tokens {}, overflow {}, pipeline events {}",
        wide.window_size,
        wide.messages.len(),
        wide.token_count,
        wide.overflow,
        wide.events.len()
    ));
    banner::step(&format!(
        "stats: total_messages {}, total_dialogues {}, user_tokens {}, assistant_tokens {}",
        wide.stats.total_messages,
        wide.stats.total_dialogues,
        wide.stats.user_tokens,
        wide.stats.assistant_tokens
    ));
    banner::step(&format!(
        "prepare(window 2): messages {} -> the window knob decides the payload",
        narrow.messages.len()
    ));
    ensure!(
        narrow.messages.len() < wide.messages.len(),
        "a narrower window should select fewer messages"
    );
    ensure!(
        !wide.overflow,
        "the prepared payload ({0} tokens) should fit the budget",
        wide.token_count
    );
    banner::note(&format!(
        "prepare changed nothing on disk: the context still holds {} stored messages",
        stored_len(context).await
    ));
    Ok(wide)
}

/// Compresses the stored history and verifies that the pass really stuck.
async fn compress(
    engine: &ContextEngine,
    context: &Arc<ConversationContext>,
) -> anyhow::Result<()> {
    // `compress` is the only call that persists a rewritten history.
    let report = engine
        .compress(context)
        .await
        .context("failed to compress the context")?;
    banner::step(&format!(
        "compress: status {:?}, tokens {} -> {}, messages {}, events {}",
        report.status,
        report.tokens_before,
        report.tokens_after,
        report.messages,
        report.events.len()
    ));
    for event in &report.events {
        banner::step(&format!(
            "  event: {} {:?}, tokens {} -> {}",
            event.processor, event.kind, event.tokens_before, event.tokens_after
        ));
    }
    ensure!(
        report.is_compressed(),
        "the 160-token budget should force a real compression"
    );
    ensure!(
        report.tokens_after < report.tokens_before,
        "compression must shrink the history, not just report it"
    );
    let stored = stored_len(context).await;
    ensure!(
        stored == report.messages,
        "the persisted history ({stored} messages) must match the report ({})",
        report.messages
    );

    // A second preview now finds nothing left to do, which proves the pass stuck.
    let after = prepare(engine, context, 6).await?;
    banner::step(&format!(
        "prepare after compress: messages {}, tokens {}, events {}",
        after.messages.len(),
        after.token_count,
        after.events.len()
    ));
    ensure!(
        after.events.is_empty(),
        "a compressed history should need no further work"
    );
    Ok(())
}

/// Builds the seeded conversation: the system prompt, then every round.
fn seeded_history() -> Vec<BaseMessage> {
    let mut history = vec![BaseMessage::system(SYSTEM)];
    for (user, assistant) in ROUNDS {
        history.push(BaseMessage::user(user));
        history.push(BaseMessage::assistant(assistant));
    }
    history
}

/// Prepares `context` with `window_size` messages and the default options.
async fn prepare(
    engine: &ContextEngine,
    context: &Arc<ConversationContext>,
    window_size: usize,
) -> anyhow::Result<PreparedContext> {
    let options = PrepareOptions {
        window_size: Some(window_size),
        ..PrepareOptions::default()
    };
    engine
        .prepare(context, &options)
        .await
        .context("failed to prepare the context")
}

/// The number of messages the context currently stores.
async fn stored_len(context: &Arc<ConversationContext>) -> usize {
    ModelContext::messages(context.as_ref()).await.len()
}

/// Seeds and recalls long-term memory, writing `statement` back as a fact.
async fn remember(statement: &str) -> anyhow::Result<()> {
    banner::title("long-term memory");

    // In-memory stores plus a deterministic embedding model: no network, no
    // credential, and the same vectors on every run.
    let memory = LongTermMemory::builder(MemoryEngineConfig::default())
        .stores(MemoryStores::in_memory())
        .embedding(Arc::new(HashEmbedding::new(32)))
        .build()
        .context("failed to build long-term memory")?;
    let scope = MemoryScope::new("alice", "workspace");
    banner::step(&format!(
        "built LongTermMemory for user '{}', scope '{}'",
        scope.user_id, scope.scope_id
    ));

    for (kind, text) in FACTS {
        let outcome = memory
            .add(&scope, new_memory(kind, text), &AddOptions::default())
            .await
            .with_context(|| format!("failed to add the {kind} fact"))?;
        let landed = if outcome.added.is_empty() {
            "merged"
        } else {
            "added"
        };
        banner::step(&format!("{landed} {kind}: {text}"));
    }

    let query = "which editor theme does alice prefer";
    let hits = memory
        .search(&scope, query, &SearchOptions::default())
        .await
        .context("failed to search long-term memory")?;
    ensure!(
        !hits.is_empty(),
        "the seeded preference should be recalled for `{query}`"
    );
    banner::step(&format!("search `{query}` -> {} hit(s)", hits.len()));
    for hit in &hits {
        banner::step(&format!(
            "  {:.3} (sim {:.3}, keyword {:.3}) {}",
            hit.score, hit.similarity, hit.keyword, hit.record.content
        ));
    }

    // Write-back: the conversation's own words become a memory, and the next
    // recall has to find them — a demo that cannot recall its own write lies.
    let outcome = memory
        .add(
            &scope,
            NewMemory::semantic(statement),
            &AddOptions::default(),
        )
        .await
        .context("failed to write the conversation back into memory")?;
    ensure!(
        outcome.len() == 1,
        "the write-back should store exactly one record"
    );
    banner::note(&format!("wrote back: {statement}"));

    let query = "run the test suite offline with a scripted model";
    let hits = memory
        .search(&scope, query, &SearchOptions::default())
        .await
        .context("failed to recall the written-back fact")?;
    let Some(top) = hits
        .iter()
        .find(|hit| hit.record.content.contains("offline"))
    else {
        bail!(
            "the written-back fact was not recalled for `{query}` ({} hit(s))",
            hits.len()
        );
    };
    banner::step(&format!(
        "recalled the write-back at {:.3}: {}",
        top.score, top.record.content
    ));
    Ok(())
}

/// Builds one [`NewMemory`] from the shape named in [`FACTS`].
fn new_memory(kind: &str, text: &str) -> NewMemory {
    match kind {
        "user_profile" => NewMemory::user_profile(text),
        "episodic" => NewMemory::episodic(text),
        _ => NewMemory::semantic(text),
    }
}
