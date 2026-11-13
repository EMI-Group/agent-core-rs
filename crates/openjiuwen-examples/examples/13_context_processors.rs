//! `13_context_processors` — the context-engine processor pipeline, stage by stage.
//!
//! Python counterpart: `openjiuwen/context_engine/` (`processor/truncate.py`,
//! `processor/tool_result.py`, `processor/summarize.py`, `offload.py`, and the reload
//! hint). Example 05 drives the engine that owns this pipeline
//! (`prepare`/`observe`/`compress`); here we drive the seam below it, a
//! `ContextPipeline` over an explicit `ProcessorContext`, so every stage shows up as
//! one `(tokens before) -> (tokens after)` step:
//!
//! 1. accounting — `count_messages`, `check_budget`, `TokenBudget`;
//! 2. truncation — `TruncationProcessor` drops the oldest whole rounds;
//! 3. tool results — `ToolResultProcessor` offloads an oversized result and leaves a
//!    bounded excerpt plus a marker;
//! 4. compression — `SummarizationProcessor` replaces older rounds with a summary;
//! 5. reload — `has_offload_markers`, `reload_system_message`, `inject_reload_hint`,
//!    and `rehydrate` recover what stage 3 hid.
//!
//! Offline and deterministic: one character is one token, the store is in memory,
//! and only a live run's summary text varies.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::Arc;

use anyhow::{anyhow, ensure, Context, Result};
use openjiuwen::core::context_engine::{
    check_budget, count_messages, has_offload_markers, inject_reload_hint, rehydrate,
    reload_system_message, ContextConfig, ContextEvent, ContextEventKind, ContextPipeline,
    InMemoryOffloadStore, OffloadKind, OffloadMarker, OffloadScope, OffloadStore, ProcessorContext,
    ReloadLanguage, SummarizationConfig, SummarizationProcessor, TokenBudget, TokenCounterConfig,
    TokenCounterKind, ToolResultConfig, ToolResultProcessor, TruncationConfig, TruncationProcessor,
    CONTEXT_SUMMARY_PREFIX, OFFLOAD_MARKER_PREFIX, PERSISTED_OUTPUT_TAG,
};
use openjiuwen::core::foundation::llm::Model;
use openjiuwen::core::foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use tokio_util::sync::CancellationToken;

use common::banner;

/// System prompt of the seeded conversation; no stage may evict or rewrite it.
const SYSTEM: &str = "You are a coding assistant. Answer in one sentence.";

/// The four oldest rounds, `(user, assistant)`, oldest first.
const ROUNDS: [(&str, &str); 4] = [
    (
        "How do I list the largest files?",
        "Run `du -ah . | sort -h | tail -n 20`.",
    ),
    (
        "The incremental build is slow.",
        "Cache the target dir before profiling.",
    ),
    (
        "Why is the link step slow?",
        "Linking re-runs from scratch; build one target.",
    ),
    (
        "Can I keep the old snapshots?",
        "Move them under `snapshots/` and prune by date.",
    ),
];

/// The tool the assistant calls, the id that pairs its result, and the user line
/// that opens the tool round.
const TOOL_NAME: &str = "grep";
const TOOL_CALL_ID: &str = "call_todos";
const TOOL_REQUEST: &str = "Find every TODO marker under crates/ and say which file has the most.";

/// The oversized tool result: long enough to be worth offloading, and the only
/// content the reload stage has to recover.
const TOOL_OUTPUT: &str = "\
crates/openjiuwen-context/src/processor/truncate.rs:12: TODO: keep the protected tail whole when the oldest round is evicted
crates/openjiuwen-context/src/processor/truncate.rs:88: TODO: count the freed tokens instead of re-measuring the history
crates/openjiuwen-context/src/processor/tool_result.rs:31: TODO: document the wire shape of the persisted-output tag
crates/openjiuwen-context/src/processor/tool_result.rs:204: FIXME: the total-budget pass re-measures every tool message twice
crates/openjiuwen-context/src/processor/summarize.rs:58: TODO: keep the summary under the configured max_summary_tokens
crates/openjiuwen-context/src/processor/offloader.rs:549: TODO: validate the preview size against the eligibility threshold
crates/openjiuwen-context/src/processor/summary_offloader.rs:12: TODO: keep the offloaded_info tag on its own line
crates/openjiuwen-context/src/processor/compressor/dialogue.rs:41: TODO: archive the replaced rounds before the benefit gate runs
crates/openjiuwen-context/src/reload.rs:84: TODO: treat a marker without a handle as absent
crates/openjiuwen-context/src/offload.rs:449: FIXME: log a missing record once per handle instead of on every read";

/// The newest round: every stage keeps it raw, because it is the live question.
const CLOSING_ROUND: (&str, &str) = (
    "Summarise today's fixes in one line.",
    "Faster file listing, a build cache, and a note on the link step.",
);

/// Messages at the end of the history that truncation may never touch: the tool
/// round (three messages) plus the closing round (two).
const PROTECTED_TAIL: usize = 5;

/// What the scripted model answers for the summary request; unused in live mode.
const SCRIPTED_SUMMARY: &str =
    "Earlier rounds covered file listing, the build cache, the link step, and snapshots.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("13 context processors");
    banner::note(&format!(
        "run mode: {}; one character counts as one token and every store is in memory",
        mode.describe()
    ));

    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let mut messages = seeded_history();
    let seeded = count_messages(&counter, &messages);
    // The truncation target: everything but the two oldest rounds, rounded up to a
    // multiple of 50 so it reads like a budget. The rounding leaves too little slack
    // for a single round, so exactly two rounds have to leave.
    let oldest_two = count_messages(&counter, &messages[1..5]);
    let budget = (seeded - oldest_two).div_ceil(50) * 50;

    let config = ContextConfig {
        max_context_tokens: Some(budget),
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    };
    let store = Arc::new(InMemoryOffloadStore::new());
    let scope = OffloadScope::new("example-session", "main");
    let cancellation = CancellationToken::new();
    let stage = Stage {
        config: &config,
        counter: &counter,
        store: Arc::clone(&store) as Arc<dyn OffloadStore>,
        scope: &scope,
        cancellation: &cancellation,
    };
    banner::step(&format!(
        "seeded {} messages: system prompt, {} plain rounds, one tool round, one closing round",
        messages.len(),
        ROUNDS.len()
    ));
    banner::note(
        "the pipeline is hand-built here; the engine builds the same one from ContextConfig::processors",
    );

    // Stage 1, accounting: one counter, one budget, one comparison, reused by every
    // later stage.
    banner::step(&format!(
        "count_messages: {seeded} tokens across {} messages",
        messages.len()
    ));
    let accounting = measure(counter, &messages, budget, "check_budget");
    ensure!(
        accounting.overflow,
        "the seeded history should overflow the {budget}-token budget"
    );
    banner::step(&format!(
        "the tool result alone is {} characters; the history is {} over budget",
        TOOL_OUTPUT.chars().count(),
        accounting.tokens - accounting.budget
    ));

    truncate(&stage, &mut messages, budget).await?;
    shrink_tool_results(&stage, &mut messages, budget).await?;
    summarize(&stage, &mut messages, budget, &mode).await?;
    // The summary must not have touched the tool round: its marker is what the
    // reload stage below is about.
    ensure!(
        has_offload_markers(&messages),
        "a compression pass must leave the offloaded tool result in place"
    );
    reload(&stage, &mut messages).await?;
    ensure!(
        store.len() == 1,
        "one tool result was offloaded, so the store holds exactly one record"
    );

    banner::title("done");
    banner::answer(
        "truncation bounded the rounds, the tool result moved to the store, a summary replaced \
         the older turns, and the reload hint plus rehydrate recovered the original",
    );
    Ok(())
}

/// Shared inputs every stage reuses: one configuration, one counter, one store, one
/// scope, and one cancellation token.
struct Stage<'a> {
    config: &'a ContextConfig,
    counter: &'a ApproxTokenCounter,
    store: Arc<dyn OffloadStore>,
    scope: &'a OffloadScope,
    cancellation: &'a CancellationToken,
}

impl Stage<'_> {
    /// The processor context for one request measured against `budget` tokens.
    fn cx(&self, budget: usize) -> ProcessorContext<'_> {
        ProcessorContext {
            config: self.config,
            counter: self.counter,
            model: None,
            offload: &*self.store,
            scope: self.scope,
            budget,
            trigger_tokens: budget,
            manual: false,
            cancellation: self.cancellation,
            recall: None,
            session_memory: None,
        }
    }
}

/// Prints one accounting line and returns the comparison it made.
fn measure(
    counter: ApproxTokenCounter,
    messages: &[BaseMessage],
    budget: usize,
    label: &str,
) -> TokenBudget {
    let state = check_budget(&counter, messages, budget);
    let verdict = if state.overflow {
        "over budget, nothing left".to_owned()
    } else {
        format!("{} tokens left", state.remaining)
    };
    banner::step(&format!(
        "{label}: {} tokens of a {}-token budget, {verdict}",
        state.tokens, state.budget
    ));
    state
}

/// Runs one pipeline and reports every change it made.
async fn run_stage(
    pipeline: &ContextPipeline,
    messages: &mut Vec<BaseMessage>,
    cx: &ProcessorContext<'_>,
    label: &str,
) -> Result<Vec<ContextEvent>> {
    let events = pipeline
        .run(messages, cx)
        .await
        .with_context(|| format!("the {label} stage failed"))?;
    for event in &events {
        banner::step(&format!(
            "`{}` -> {:?}: {} message(s) changed, tokens {} -> {}",
            event.processor,
            event.kind,
            event.messages_changed.len(),
            event.tokens_before,
            event.tokens_after
        ));
    }
    Ok(events)
}

/// The single event a stage promises, checked for the kind it reports.
fn single_event(events: &[ContextEvent], kind: ContextEventKind) -> Result<&ContextEvent> {
    ensure!(
        events.len() == 1,
        "expected exactly one event, got {}",
        events.len()
    );
    ensure!(events[0].kind == kind, "expected {kind:?}");
    Ok(&events[0])
}

/// Stage 2: drop whole dialogue rounds from the oldest end until the budget fits.
async fn truncate(stage: &Stage<'_>, messages: &mut Vec<BaseMessage>, budget: usize) -> Result<()> {
    banner::title("truncation");
    let before = messages.len();
    let pipeline =
        ContextPipeline::new(vec![Arc::new(TruncationProcessor::new(TruncationConfig {
            // An explicit target acts even when the threshold has not been crossed;
            // `keep_system` keeps the persona prompt out of the race.
            max_tokens: Some(budget),
            keep_recent_messages: PROTECTED_TAIL,
            keep_system: true,
        }))]);
    let events = run_stage(&pipeline, messages, &stage.cx(budget), "truncation").await?;
    single_event(&events, ContextEventKind::Truncated)?;
    // Whole rounds only, so tool-call/tool-result pairs are never split.
    ensure!(
        before - messages.len() == 4,
        "the two oldest rounds (four messages) should have left the history"
    );
    let after = measure(*stage.counter, messages, budget, "after truncation");
    ensure!(
        !after.overflow,
        "truncation must leave the history inside the budget"
    );
    banner::note(
        "truncation cannot shrink one oversized message: the protected tail still carries the whole tool result",
    );
    Ok(())
}

/// Stage 3: the oversized tool result moves to the offload store, leaving a bounded
/// excerpt, a persisted-output tag, and an offload marker.
async fn shrink_tool_results(
    stage: &Stage<'_>,
    messages: &mut Vec<BaseMessage>,
    budget: usize,
) -> Result<()> {
    banner::title("tool results");
    let config = ToolResultConfig {
        max_result_tokens: 120, // backstop for results under the offload threshold
        head_chars: 160,
        tail_chars: 60,
        // Above this many characters the original goes to the store rather than
        // being trimmed, so nothing is lost.
        offload_over_chars: Some(400),
        // The history holds one tool result and it is the one to shrink, so no
        // result is protected and no tool is exempt.
        keep_recent_results: 0,
        protected_tools: Vec::new(),
        ..ToolResultConfig::default()
    };
    let pipeline = ContextPipeline::new(vec![Arc::new(ToolResultProcessor::new(
        config,
        Arc::clone(&stage.store),
    ))]);
    let events = run_stage(&pipeline, messages, &stage.cx(budget), "tool result").await?;
    let event = single_event(&events, ContextEventKind::Offloaded)?;
    ensure!(
        event.offload_handle.is_some(),
        "an offload names its handle"
    );

    let text = tool_text(messages)?;
    let marker = OffloadMarker::parse(&text)
        .ok_or_else(|| anyhow!("the shrunk tool result carries no offload marker"))?;
    ensure!(
        !marker.handle.is_empty() && marker.kind == OffloadKind::InMemory,
        "the marker should name an in-memory record"
    );
    banner::step(&format!(
        "`{OFFLOAD_MARKER_PREFIX} handle=<id>, type={}]]` inside `<{PERSISTED_OUTPUT_TAG}>`: \
         {} characters left of {}",
        marker.kind.as_marker_str(),
        text.chars().count(),
        TOOL_OUTPUT.chars().count()
    ));
    let after = measure(
        *stage.counter,
        messages,
        budget,
        "after the tool-result pass",
    );
    ensure!(
        after.tokens < event.tokens_before,
        "offloading must shrink the history"
    );
    banner::note(
        "ToolResultBudgetProcessor (largest result first) and ToolResultWindowProcessor (newest K per tool) write to the same store",
    );
    Ok(())
}

/// Stage 4: replace older complete rounds with one model-written summary.
async fn summarize(
    stage: &Stage<'_>,
    messages: &mut Vec<BaseMessage>,
    budget: usize,
    mode: &common::RunMode,
) -> Result<()> {
    banner::title("summarization");
    // Offline the answer is scripted and deterministic; live it is whatever the
    // configured provider writes.
    let model: Arc<dyn Model> = if mode.is_live() {
        common::live_model().await?
    } else {
        common::scripted(SCRIPTED_SUMMARY)
    };
    let pipeline = ContextPipeline::new(vec![Arc::new(SummarizationProcessor::new(
        SummarizationConfig {
            // Zero always counts as worth a summary: this stage shows the
            // replacement, not a trigger threshold.
            tokens_threshold: Some(0),
            keep_recent_messages: 4,
            keep_last_rounds: 1,
            min_messages_to_summarize: 2,
            max_summary_tokens: 200,
        },
        model,
    ))]);
    let events = run_stage(&pipeline, messages, &stage.cx(budget), "summarization").await?;
    let event = single_event(&events, ContextEventKind::Summarized)?;
    banner::step(&format!(
        "{} message(s) became one `{CONTEXT_SUMMARY_PREFIX}` message:",
        event.messages_changed.len()
    ));
    banner::note(&event.summary.clone().unwrap_or_default());
    ensure!(
        messages
            .iter()
            .any(|message| message.text().starts_with(CONTEXT_SUMMARY_PREFIX)),
        "the summary is stored as one prefixed user message"
    );
    ensure!(
        messages.iter().any(|message| message.role == Role::System),
        "compression never evicts system messages"
    );
    let after = measure(*stage.counter, messages, budget, "after summarization");
    ensure!(
        after.tokens < event.tokens_before,
        "the summary must be smaller than the rounds it replaced"
    );
    Ok(())
}

/// Stage 5: the marker on the shrunk result, the hint that explains it, and the
/// store read that recovers the original.
async fn reload(stage: &Stage<'_>, messages: &mut Vec<BaseMessage>) -> Result<()> {
    banner::title("offload and reload");
    ensure!(
        has_offload_markers(messages),
        "the history should still carry the offload marker"
    );
    // The hint is a system message the caller adds in front of the history: it says
    // what a marker means and how to ask for the hidden content.
    let hint = reload_system_message(messages, ReloadLanguage::En)
        .ok_or_else(|| anyhow!("a history with a marker should get a reload hint"))?;
    let headline = hint.text().lines().next().unwrap_or_default().to_owned();
    banner::step(&format!(
        "reload_system_message: a `{headline}` system message of {} lines",
        hint.text().lines().count()
    ));
    // Injection is idempotent, so a caller may run it before every request.
    ensure!(
        inject_reload_hint(messages, ReloadLanguage::En),
        "the first injection should insert the hint after the leading system messages"
    );
    ensure!(
        !inject_reload_hint(messages, ReloadLanguage::En),
        "a second injection must not duplicate the hint"
    );
    banner::step(&format!(
        "inject_reload_hint inserted one system message; the payload is now {} messages",
        messages.len()
    ));
    // `rehydrate` is the recovery half: the placeholder text alone locates the
    // record, and the original message comes back unchanged.
    let text = tool_text(messages)?;
    let restored = rehydrate(&*stage.store, stage.scope, &text)
        .await
        .context("failed to read the offload record")?
        .ok_or_else(|| anyhow!("the marker's record should still be in the store"))?;
    ensure!(
        restored.len() == 1,
        "one offloaded tool result should come back as one message"
    );
    let original = &restored[0];
    ensure!(
        original.text().as_ref() == TOOL_OUTPUT,
        "rehydrate must return the original tool output verbatim"
    );
    ensure!(
        original.tool_call_id.as_deref() == Some(TOOL_CALL_ID),
        "the recovered message keeps its tool_call_id, so the pairing survives"
    );
    banner::step(&format!(
        "rehydrate returned the original {} characters, still paired with `{TOOL_CALL_ID}`",
        original.text().chars().count()
    ));
    banner::note(
        "SessionOffloadStore keeps the same records in session state and KvOffloadStore in a KvStore",
    );
    Ok(())
}

/// The text of the single tool message in `messages`.
fn tool_text(messages: &[BaseMessage]) -> Result<String> {
    messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .map(|message| message.text().into_owned())
        .ok_or_else(|| anyhow!("the history no longer holds a tool result"))
}

/// Builds the seeded conversation: system prompt, four plain rounds, the tool
/// round, and the closing round.
fn seeded_history() -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system(SYSTEM)];
    for (user, assistant) in ROUNDS {
        messages.push(BaseMessage::user(user));
        messages.push(BaseMessage::assistant(assistant));
    }
    messages.push(BaseMessage::user(TOOL_REQUEST));
    messages.push(
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new(TOOL_NAME).with_id(TOOL_CALL_ID)]),
    );
    messages.push(BaseMessage::tool(TOOL_CALL_ID, TOOL_OUTPUT));
    let (user, assistant) = CLOSING_ROUND;
    messages.push(BaseMessage::user(user));
    messages.push(BaseMessage::assistant(assistant));
    messages
}
