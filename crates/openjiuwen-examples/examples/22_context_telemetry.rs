//! `22_context_telemetry` — where a request's prompt tokens went, and what the
//! processors did to the stored history.
//!
//! Two telemetry seams of the context engine
//! (`openjiuwen::core::context_engine`) run over one compressed transcript:
//!
//! 1. `context.usage` — [`ContextUsageAnalyzer`] measures the prepared window
//!    into four categories (`system_prompt`, `tools`, `skills`, `messages`),
//!    reconciles the local estimate with an authoritative provider total, and
//!    reports its measurement-cache outcome. `prepare` puts the snapshot on
//!    [`PreparedContext::usage`] and [`ContextEngine::usage_snapshot`] returns it
//!    directly; it is a return value, never a session-stream frame.
//! 2. `context.compression_state` — [`ContextProcessorStateRecorder`] brackets
//!    every processor pass with `started` and terminal records (before/after
//!    sizes, savings, duration, usage), keeps a bounded history, and fans the
//!    records out to a [`CompressionStateSink`] and optionally a session stream.
//!
//! The transcript is one character per token, the summary model is scripted
//! offline, and every store is in memory, so the numbers below are the same on
//! every run. Python counterparts: the `usage/` and `processor` telemetry paths
//! of `openjiuwen/context_engine/`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use openjiuwen::core::common::error::ErrorKind;
use openjiuwen::core::context_engine::usage::{MeasurementCacheHit, PHASE_POST_CALL};
use openjiuwen::core::context_engine::{
    AnalysisInput, CompressionStatePhase, CompressionStateSink, CompressionStateStatus,
    ContextCompressionState, ContextConfig, ContextEngine, ContextOptions,
    ContextProcessorStateRecorder, ContextUsageAnalyzer, ContextUsageSnapshot, ConversationContext,
    CounterIdentity, PrepareOptions, PreparedContext, ProcessorConfig, RecordingSink,
    SummarizationConfig, TokenCounterConfig, TokenCounterKind, TruncationConfig,
    CONTEXT_USAGE_EVENT_TYPE, CONTEXT_USAGE_SCHEMA_VERSION,
};
use openjiuwen::core::foundation::llm::Model;
use openjiuwen::core::foundation::prompt::{
    ApproxTokenCounter, BaseMessage, Role, TokenCounter, ToolCall,
};
use serde_json::{json, Value};

use common::banner;

/// System prompt of the seeded conversation; the telemetry never rewrites it.
const SYSTEM: &str =
    "You are a build assistant. Answer in one sentence and prefer commands you can verify.";

/// The four oldest rounds, `(user, assistant)`, oldest first.
const ROUNDS: [(&str, &str); 4] = [
    (
        "How do I list the largest files in a repository?",
        "Run `du -ah . | sort -h | tail -n 20`.",
    ),
    (
        "The incremental build keeps recompiling the workspace.",
        "Cache the target directory before profiling.",
    ),
    (
        "Why is the link step slow?",
        "Linking re-runs from scratch; build one target at a time.",
    ),
    (
        "Can I keep the old release snapshots?",
        "Move them under `snapshots/` and prune by date.",
    ),
];

/// The tool the assistant calls, the id pairing its result, and the user line
/// that opens the tool round.
const TOOL_NAME: &str = "grep";
const TOOL_CALL_ID: &str = "call_todos";
const TOOL_REQUEST: &str =
    "Find every TODO marker under crates/ and report the files with the most markers.";

/// The oversized tool result: it dominates the token budget before compression.
const TOOL_OUTPUT: &str = "\
crates/openjiuwen-context/src/processor/truncate.rs:12: TODO: keep the protected tail whole
crates/openjiuwen-context/src/processor/truncate.rs:88: TODO: count the freed tokens instead of re-measuring
crates/openjiuwen-context/src/processor/tool_result.rs:31: TODO: document the persisted-output tag
crates/openjiuwen-context/src/processor/tool_result.rs:204: FIXME: the total-budget pass re-measures
crates/openjiuwen-context/src/processor/summarize.rs:61: TODO: keep the summary prefix stable
crates/openjiuwen-context/src/processor/summarize.rs:180: TODO: record the model usage of the pass
crates/openjiuwen-context/src/processor/pipeline.rs:97: TODO: bracket every pass with started and terminal records
crates/openjiuwen-context/src/processor/pipeline.rs:410: TODO: clone the history once per recorded pass
crates/openjiuwen-context/src/telemetry/recorder/mod.rs:57: TODO: deliver records through the sink
crates/openjiuwen-context/src/telemetry/recorder/mod.rs:211: TODO: bound the in-memory history
crates/openjiuwen-context/src/telemetry/state.rs:141: TODO: keep the wire field names stable
crates/openjiuwen-context/src/usage/analyzer.rs:357: TODO: explain the measurement-cache outcome
crates/openjiuwen-context/src/usage/analyzer.rs:830: TODO: reconcile the provider total with the local estimate
crates/openjiuwen-context/src/usage/models.rs:488: TODO: keep the snapshot serde-tolerant
crates/openjiuwen-context/src/usage/parts.rs:73: TODO: route attachments into their category";

/// The answer the assistant gives once the tool result is in the history.
const CLOSING_TURN: &str = "Which file should I clean up first?";

/// What the scripted model answers for the summary request; unused when live.
const SCRIPTED_SUMMARY: &str =
    "Earlier rounds covered file listing, the build cache, the link step and snapshots.";

/// Literal identities so the printed narrative does not vary between runs.
const MODEL: &str = "telemetry-demo-model";
const PROVIDER: &str = "offline-provider";
const CONTEXT_ID: &str = "main";
/// Session id of the control engine's context, kept separate from the main one.
const CONTROL_CONTEXT_ID: &str = "control";

/// Slack added to the seeded transcript to get the context budget: enough that
/// `prepare` does not truncate, small enough that occupancy is interesting.
const BUDGET_SLACK: usize = 400;

/// Token delta the pretend provider reports above the local estimate.
const PROVIDER_OVERHEAD: u64 = 23;

/// How many state records the recorder keeps, and how many measurements the
/// analyzer memoizes.
const RECORD_HISTORY_LIMIT: usize = 64;
const MEASUREMENT_CACHE_CAPACITY: usize = 16;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("22 context telemetry");
    banner::note(&format!(
        "run mode: {}; one character is one token, every store is in memory, and only the live summary text varies",
        mode.describe()
    ));

    // One counter is shared by the recorder (for before/after metrics) and the
    // analyzer (for the category parts), so both seams report the same numbers.
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let history = seeded_history();
    let observed = observed_messages();
    let seeded_tokens = counter.count_messages(&history) + counter.count_messages(&observed);
    let budget = seeded_tokens + BUDGET_SLACK;
    banner::step(&format!(
        "seeded {} history messages + {} observed messages = {seeded_tokens} tokens; budget {budget}",
        history.len(),
        observed.len()
    ));

    // Offline the summary is fixed; live it is whatever the provider writes.
    let summary_model: Arc<dyn Model> = if mode.is_live() {
        banner::note("live mode: the compression summary comes from the configured provider");
        common::live_model().await?
    } else {
        common::scripted(SCRIPTED_SUMMARY)
    };

    let harness = build_harness(budget)?;
    let context = harness
        .engine
        .create_context_with(
            CONTEXT_ID,
            ContextOptions {
                history,
                model: Some(Arc::clone(&summary_model)),
                ..ContextOptions::default()
            },
        )
        .await
        .context("failed to create the conversation context")?;

    let before_tokens = observe_and_measure(&harness, &context, observed, budget).await?;
    compress(&harness, &context).await?;
    print_records(&harness)?;
    let after = measure_after(&harness, &context, before_tokens).await?;
    reconcile(&harness, &after)?;
    control().await?;

    banner::title("what this showed");
    banner::note(
        "context.usage is a return value: prepare puts it on PreparedContext.usage and usage_snapshot returns it, so it never arrives as a session-stream frame",
    );
    banner::note(
        "its measurement cache reuses unchanged parts: miss on the first measurement, partial after compression, full when the same window is measured again",
    );
    banner::note(
        "context.compression_state is a record: the recorder keeps a bounded history, fans it out to a CompressionStateSink, and can also write it to the session stream",
    );
    banner::answer(
        "the prepared window was measured category by category, reconciled with a provider total, compressed deterministically, and every processor pass was recorded before and after",
    );
    Ok(())
}

/// The engine plus the telemetry handles the example reads back at the end.
struct Harness {
    engine: ContextEngine,
    analyzer: Arc<ContextUsageAnalyzer>,
    recorder: Arc<ContextProcessorStateRecorder>,
    sink: Arc<RecordingSink>,
}

/// Builds the instrumented engine: recorder, sink, and usage analyzer shared by
/// every context the engine creates.
fn build_harness(budget: usize) -> Result<Harness> {
    let counter: Arc<dyn TokenCounter + Send + Sync> =
        Arc::new(ApproxTokenCounter::with_chars_per_token(1));
    let limit = u64::try_from(budget).context("the context budget does not fit in u64")?;
    let analyzer = Arc::new(
        ContextUsageAnalyzer::new(Arc::clone(&counter), CounterIdentity::approx())
            .with_model(MODEL)
            .with_context_window_limit(Some(limit))
            .with_measurement_cache_capacity(MEASUREMENT_CACHE_CAPACITY),
    );
    let sink = Arc::new(RecordingSink::new());
    let recorder = Arc::new(
        ContextProcessorStateRecorder::new(
            openjiuwen::core::context_engine::DEFAULT_SESSION_ID,
            CONTEXT_ID,
        )
        .with_sink(Arc::clone(&sink) as Arc<dyn CompressionStateSink>)
        .with_token_counter(counter)
        .with_history_limit(RECORD_HISTORY_LIMIT),
    );
    let engine = ContextEngine::new(context_config(budget))
        .with_recorder(Some(Arc::clone(&recorder)))
        .with_usage_analyzer(Some(Arc::clone(&analyzer)));
    Ok(Harness {
        engine,
        analyzer,
        recorder,
        sink,
    })
}

/// The processor pipeline and budget of the instrumented engine.
///
/// Summarisation has an explicit `tokens_threshold` of `usize::MAX`, so only the
/// manual `compress` pass — which bypasses the threshold — calls the model.
/// Truncation is the explicit backstop: it removes whole rounds when the budget
/// is still exceeded after summarisation.
fn context_config(budget: usize) -> ContextConfig {
    ContextConfig {
        max_context_tokens: Some(budget),
        // `1.0` puts the trigger at the budget: `prepare` never truncates the
        // seeded transcript, so its usage snapshot measures the whole history.
        compression_threshold: 1.0,
        model_name: Some(MODEL.to_owned()),
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        processors: vec![
            ProcessorConfig::Summarization(SummarizationConfig {
                tokens_threshold: Some(usize::MAX),
                keep_recent_messages: 2,
                keep_last_rounds: 1,
                min_messages_to_summarize: 2,
                max_summary_tokens: 200,
            }),
            ProcessorConfig::Truncation(TruncationConfig {
                max_tokens: None,
                keep_recent_messages: 4,
                keep_system: true,
            }),
        ],
        ..ContextConfig::default()
    }
}

/// Prepares the context with the tool schemas a real request would send.
async fn prepare(
    engine: &ContextEngine,
    context: &Arc<ConversationContext>,
) -> Result<PreparedContext> {
    engine
        .prepare(
            context,
            &PrepareOptions {
                tools: tool_schemas(),
                ..PrepareOptions::default()
            },
        )
        .await
        .context("failed to prepare the context")
}

/// Observes the two turns a real request would add, then measures the window.
///
/// `observe` appends the message and, with a recorder attached, runs the passive
/// pipeline over the incoming batch for telemetry. `prepare` then assembles the
/// request window and attaches the first `context.usage` snapshot.
async fn observe_and_measure(
    harness: &Harness,
    context: &Arc<ConversationContext>,
    observed: Vec<BaseMessage>,
    budget: usize,
) -> Result<u64> {
    for message in observed {
        let kind = match message.role {
            Role::Tool => "tool result",
            Role::User => "user turn",
            _ => "message",
        };
        let outcome = harness
            .engine
            .observe(context, message)
            .await
            .with_context(|| format!("failed to observe the {kind}"))?;
        banner::step(&format!(
            "observed {kind}: total_messages {}, total_tokens {}, evicted {}",
            outcome.total_messages, outcome.total_tokens, outcome.evicted
        ));
    }
    let seeded = prepare(&harness.engine, context).await?;
    banner::step(&format!(
        "prepare: {} messages, {} tokens, overflow {}, pipeline events {}",
        seeded.messages.len(),
        seeded.token_count,
        seeded.overflow,
        seeded.events.len()
    ));
    let before = seeded
        .usage
        .as_ref()
        .ok_or_else(|| anyhow!("the engine should attach a context.usage snapshot"))?;
    print_usage("context.usage from prepare", before)?;
    check_window("the prepared window", before, budget)?;
    ensure!(
        before.measurement_cache.hit == MeasurementCacheHit::Miss,
        "the first measurement of a window should be a cache miss"
    );
    Ok(before.context_window.tokens)
}

/// Measures the window a fresh `prepare` assembles after compression.
///
/// Only the messages changed, so the analyzer reuses the system-prompt and
/// tools measurements; a second accessor call then hits the cache fully.
async fn measure_after(
    harness: &Harness,
    context: &Arc<ConversationContext>,
    before_tokens: u64,
) -> Result<PreparedContext> {
    let after = prepare(&harness.engine, context).await?;
    let after_usage = after
        .usage
        .as_ref()
        .ok_or_else(|| anyhow!("the engine should attach a context.usage snapshot"))?;
    banner::step(&format!(
        "prepare after compress: {} messages, {} tokens, events {}",
        after.messages.len(),
        after.token_count,
        after.events.len()
    ));
    print_usage("context.usage after compress", after_usage)?;
    ensure!(
        after_usage.context_window.tokens < before_tokens,
        "compression must shrink the measured window, not just the stored report"
    );
    ensure!(
        after_usage.measurement_cache.hit == MeasurementCacheHit::Partial,
        "a changed messages part next to unchanged system/tools parts should be a partial hit"
    );

    // The engine accessor measures the same prepared window again, so every part
    // is now served from the analyzer's cache.
    let accessor = harness
        .engine
        .usage_snapshot(context, None)
        .await
        .context("usage_snapshot should work once an analyzer is configured")?;
    ensure!(
        accessor.measurement_cache.hit == MeasurementCacheHit::Full,
        "re-measuring an unchanged window should be a full cache hit"
    );
    ensure!(
        accessor.context_window.tokens == after_usage.context_window.tokens,
        "usage_snapshot must report the window prepare last assembled"
    );
    banner::step(&format!(
        "usage_snapshot: {} tokens of {}, cache {} ({} reused, {} recomputed)",
        accessor.context_window.tokens,
        display_u64(accessor.context_window.limit_tokens),
        accessor.measurement_cache.hit.as_str(),
        accessor.measurement_cache.reused_parts.len(),
        accessor.measurement_cache.recomputed_parts.len()
    ));
    Ok(after)
}

/// Calls the analyzer directly with an authoritative provider total.
fn reconcile(harness: &Harness, after: &PreparedContext) -> Result<()> {
    banner::title("reconcile a provider total");
    let local = after
        .usage
        .as_ref()
        .ok_or_else(|| anyhow!("the prepared window should carry a usage snapshot"))?
        .context_window
        .local_estimated_input_tokens;
    let provider_total = local + PROVIDER_OVERHEAD;
    let system_count = after
        .messages
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    let mut input = AnalysisInput::new("req-telemetry-2", harness.recorder.session_id());
    input.phase = PHASE_POST_CALL;
    input.provider = Some(PROVIDER);
    input.sequence = 2;
    input.provider_input_tokens = Some(provider_total);
    input.system_messages = Some(&after.messages[..system_count]);
    input.context_messages = &after.messages[system_count..];
    input.tools = &after.tools;
    input.context_stats = Some(&after.stats);
    let reconciled = harness.analyzer.analyze(&input);
    banner::step(&format!(
        "analyze with a provider total of {provider_total}: local {local}, delta {}, authoritative {}",
        display_i64(reconciled.context_window.reconciliation_delta_tokens),
        reconciled.measurement.authoritative_total
    ));
    print_usage("context.usage reconciled", &reconciled)?;
    ensure!(
        reconciled.measurement.authoritative_total,
        "a supplied provider total is authoritative"
    );
    ensure!(
        reconciled.measurement_cache.hit == MeasurementCacheHit::Full,
        "the reconciled measurement reuses the same window measurements"
    );
    let expected_delta = i64::try_from(provider_total).context("provider total exceeds i64")?
        - i64::try_from(local).context("local total exceeds i64")?;
    ensure!(
        reconciled.context_window.reconciliation_delta_tokens == Some(expected_delta),
        "the reported delta should be provider total minus local estimate"
    );
    let reconciled_parts: u64 = reconciled.parts.values().map(|part| part.tokens).sum();
    ensure!(
        reconciled_parts == provider_total,
        "proportional reconciliation should make the parts sum to the provider total"
    );
    Ok(())
}

/// Runs one compression pass and reports what the engine and the recorder saw.
async fn compress(harness: &Harness, context: &Arc<ConversationContext>) -> Result<()> {
    banner::title("compress the stored history");
    let report = harness
        .engine
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
            "  event {} {:?}: tokens {} -> {}",
            event.processor, event.kind, event.tokens_before, event.tokens_after
        ));
    }
    ensure!(report.is_compressed(), "the manual pass must compress");
    ensure!(
        report.tokens_after < report.tokens_before,
        "compression must shrink the stored history"
    );
    Ok(())
}

/// Prints the recorder's history and every terminal record it fanned out.
fn print_records(harness: &Harness) -> Result<()> {
    banner::title("context.compression_state records");
    let states = harness.sink.states();
    let history = harness.recorder.history();
    ensure!(
        history.len() == states.len(),
        "the recorder history ({}) and the sink ({}) should hold the same records",
        history.len(),
        states.len()
    );
    let statuses = harness.sink.statuses();
    let started = statuses
        .iter()
        .filter(|status| **status == CompressionStateStatus::Started)
        .count();
    banner::step(&format!(
        "{} records: {started} started, {} terminal; the recorder keeps at most {}",
        states.len(),
        statuses.len() - started,
        harness.recorder.history_limit()
    ));
    ensure!(
        statuses.contains(&CompressionStateStatus::Completed),
        "the summary pass should record a completed state"
    );
    ensure!(
        statuses.contains(&CompressionStateStatus::Noop),
        "a processor that changed nothing should record a noop state"
    );
    ensure!(
        states.iter().any(|state| {
            state.phase == CompressionStatePhase::ActiveCompress && state.processor == "summarization"
        }),
        "the compression pass should record the summarization processor"
    );

    for state in states.iter().filter(|state| state.after.is_some()) {
        let Some(after) = state.after.as_ref() else {
            continue;
        };
        banner::step(&format!(
            "{:<17} {:<9} {:<13} {} -> {} msgs, {} -> {} tokens{}",
            state.phase.as_str(),
            state.status.as_str(),
            state.processor,
            state.before.messages,
            after.messages,
            state.before.tokens,
            after.tokens,
            saved_text(state),
        ));
        if state.status == CompressionStateStatus::Completed {
            banner::note(&format!("{}: {}", state.processor, state.summary));
        }
    }
    Ok(())
}

/// A second engine without an analyzer: the usage seam is optional, and its
/// absence is a typed error rather than a fabricated snapshot.
async fn control() -> Result<()> {
    banner::title("control: an engine without an analyzer");
    let config = ContextConfig {
        max_context_tokens: Some(1_000),
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        processors: vec![ProcessorConfig::Truncation(TruncationConfig::default())],
        ..ContextConfig::default()
    };
    let engine = ContextEngine::new(config);
    let context = engine
        .create_context_with(
            CONTROL_CONTEXT_ID,
            ContextOptions {
                history: seeded_history(),
                ..ContextOptions::default()
            },
        )
        .await
        .context("failed to create the control context")?;

    match engine.usage_snapshot(&context, None).await {
        Ok(_) => bail!("an engine without an analyzer must not produce a snapshot"),
        Err(error) => {
            ensure!(
                error.kind() == ErrorKind::Config,
                "the missing-analyzer error should be Config, got {}",
                error.kind()
            );
            banner::step(&format!(
                "usage_snapshot: {} — {}",
                error.kind(),
                error.message()
            ));
        }
    }

    // Telemetry never fails a request: the same context still prepares.
    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .context("prepare must succeed without an analyzer")?;
    ensure!(
        prepared.usage.is_none(),
        "without an analyzer the prepared payload carries no usage block"
    );
    banner::step(&format!(
        "prepare still succeeded with {} messages; usage is None, not an error",
        prepared.messages.len()
    ));
    Ok(())
}

/// Prints every field of one `context.usage` snapshot that tells the story.
fn print_usage(label: &str, snapshot: &ContextUsageSnapshot) -> Result<()> {
    let request = if snapshot.request_id.is_empty() {
        "<caller-assigned>"
    } else {
        &snapshot.request_id
    };
    banner::step(&format!(
        "{label}: `{}` {} phase {} request {request}",
        snapshot.event_type, snapshot.schema_version, snapshot.phase
    ));
    let window = &snapshot.context_window;
    banner::step(&format!(
        "  window {} tokens, source {}, limit {}, occupancy {}, local {}, delta {}",
        window.tokens,
        window.tokens_source,
        display_u64(window.limit_tokens),
        display_rate(window.occupancy_rate),
        window.local_estimated_input_tokens,
        display_i64(window.reconciliation_delta_tokens),
    ));
    for (name, part) in &snapshot.parts {
        banner::step(&format!(
            "  part {name:<14} {:>5} tokens, {:>7} of window, source {}, measurement {}, estimated {}{}",
            part.tokens,
            display_rate(part.percentage_of_window),
            part.source,
            part.measurement_source,
            part.estimated,
            part.fallback_reason
                .as_deref()
                .map_or_else(String::new, |reason| format!(", fallback {reason}")),
        ));
    }
    banner::step(&format!(
        "  measurement: category {}, total {}, tokenizer {}, estimated {}, authoritative {}, cache {}",
        snapshot.measurement.category_source,
        snapshot.measurement.total_source,
        snapshot.measurement.tokenizer.as_deref().unwrap_or("none"),
        snapshot.measurement.estimated,
        snapshot.measurement.authoritative_total,
        snapshot.measurement_cache.hit.as_str(),
    ));
    let cache = match &snapshot.kv_cache.request {
        Some(request) => format!(
            "{} (read {}, miss {})",
            request.status,
            display_u64(request.cache_read_tokens),
            display_u64(request.cache_miss_tokens)
        ),
        None => "not_reported (no provider usage attached)".to_owned(),
    };
    banner::step(&format!("  kv cache: {cache}"));
    if let Some(stats) = &snapshot.context_stats {
        banner::step(&format!(
            "  stats: {} messages, {} dialogues, tokens user/assistant/tool = {}/{}/{}",
            stats.total_messages,
            stats.total_dialogues,
            stats.user_tokens,
            stats.assistant_tokens,
            stats.tool_tokens,
        ));
    }
    ensure!(
        snapshot.event_type == CONTEXT_USAGE_EVENT_TYPE,
        "the snapshot carries the context.usage event type"
    );
    ensure!(
        snapshot.schema_version == CONTEXT_USAGE_SCHEMA_VERSION,
        "the snapshot carries the context-usage.v1 schema version"
    );
    let part_total: u64 = snapshot.parts.values().map(|part| part.tokens).sum();
    // Without a provider total the parts sum to the local estimate; after
    // reconciliation they sum to the authoritative provider total instead.
    let expected_total = if snapshot.measurement.authoritative_total {
        snapshot.context_window.tokens
    } else {
        snapshot.context_window.local_estimated_input_tokens
    };
    ensure!(
        part_total == expected_total,
        "the four parts should sum to the reported window total"
    );
    Ok(())
}

/// Checks the window-level invariants a prepared snapshot must satisfy.
fn check_window(label: &str, snapshot: &ContextUsageSnapshot, budget: usize) -> Result<()> {
    let window = &snapshot.context_window;
    ensure!(
        window.limit_tokens == Some(u64::try_from(budget).context("budget exceeds u64")?),
        "{label} should carry the configured window limit"
    );
    ensure!(
        window.tokens > 0 && window.tokens < u64::try_from(budget).unwrap_or(u64::MAX),
        "{label} should fit the budget but not be empty"
    );
    ensure!(
        window.reconciliation_delta_tokens.is_none() && !snapshot.measurement.authoritative_total,
        "{label} has no provider total, so it stays a local estimate"
    );
    ensure!(
        !window.parts.is_empty(),
        "{label} should carry the category parts"
    );
    Ok(())
}

/// The `saved`/window suffix of one record line.
fn saved_text(state: &ContextCompressionState) -> String {
    let saved = state.saved.as_ref().map_or_else(String::new, |saved| {
        format!(", saved {} tokens ({:.1}%)", saved.tokens, saved.percent)
    });
    let percent = state
        .after
        .as_ref()
        .and_then(|after| after.context_percent)
        .map_or_else(String::new, |percent| format!(", window {percent}%"));
    format!("{saved}{percent}")
}

/// Renders an optional count.
fn display_u64(value: Option<u64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "n/a".to_owned(),
    }
}

/// Renders an optional signed count.
fn display_i64(value: Option<i64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "n/a".to_owned(),
    }
}

/// Renders an optional fraction as a percentage with one decimal.
fn display_rate(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{:.2}%", value * 100.0),
        None => "n/a".to_owned(),
    }
}

/// The two turns appended through `observe`: the tool result that closes the
/// tool round, then the user line the next model call answers.
fn observed_messages() -> Vec<BaseMessage> {
    vec![
        BaseMessage::tool(TOOL_CALL_ID, TOOL_OUTPUT),
        BaseMessage::user(CLOSING_TURN),
    ]
}

/// The seeded conversation: system prompt, four plain rounds, and the tool call
/// whose result `observe` appends.
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
    messages
}

/// One serialized tool schema, measured as the `tools` category of the window.
fn tool_schemas() -> Vec<Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Search repository files for a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"}
                },
                "required": ["pattern"]
            }
        }
    })]
}
