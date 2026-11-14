//! `25_harness_moa_persistent` — two harness seams that survive a single model call:
//! mixture-of-agents consultation and the persistent-agent journal.
//!
//! Part A drives `openjiuwen::harness::moa`, the Rust port of Python
//! `openjiuwen/harness/moa/` plus `rails/moa_rail.py`. Three scripted reference models are
//! asked the same problem concurrently; one of them fails, so the batch demonstrates the
//! partial-failure rule (a partial batch is fine, an all-fail batch is `Error::Llm`). The
//! retained proposals are ranked structurally (offline, no verifier model), folded into one
//! labelled answer, and then the same configuration is driven through `MoaRail`: the rail
//! publishes its bilingual prompt section, installs the per-invocation state, records
//! progress steps, scores them, publishes the outcome on the next model call, retires the
//! state at the end of the invocation, and round-trips its bounded history through a file.
//!
//! Part B drives `openjiuwen::harness::persistent_agent`, the Rust port of Python
//! `openjiuwen/harness/persistent_agent/execution/`. An execution journal is appended to,
//! rotated mid-run, and then abandoned — that is the "crash" — and reopened by directory,
//! so `replay()` has to rebuild the entries, the cursor and the digest from what really
//! reached disk. A session record and an execution link are persisted next to it, and
//! `resume()` traces that chain into a `ResumePlan`: the rebuilt model window, the
//! continuation cursor, the background document and the journal digest. The last step
//! assembles a window under a tiny budget with the model-summariser seam, which is a
//! documented `Error::Unsupported` path.
//!
//! Offline and deterministic: every model is a `ScriptedModel`, every timestamp comes from
//! a `ManualClock` frozen at one instant, the root is a `TempDir`, and only directory and
//! file names reach stdout — never the temporary path.
//!
//! Python counterparts: `harness/moa/` (no runnable example in the Python tree) and
//! `harness/persistent_agent/execution/` (likewise), plus the resume flow of
//! `examples/client_tool/client_tool_demo.py`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements,
    // Both halves are one linear tour (consult → rank → aggregate → rail, then append →
    // crash → replay → resume); splitting the narrative further would scatter the story.
    clippy::too_many_lines
)]

mod common;

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use openjiuwen::core::common::error::ErrorKind;
use openjiuwen::core::foundation::prompt::BaseMessage;
use openjiuwen::core::single_agent::RailEvent;
use openjiuwen::harness::moa::{
    consult, rank_with_config, AggregationStrategy, Aggregator, ConsultationOutcome, JevConfig,
    MoaConfig, MoaMode, MoaRail, MoaSnapshot, MOA_OUTCOME_EXTRA_KEY, MOA_PROMPT_SECTION,
    MOA_STATE_EXTRA_KEY,
};
use openjiuwen::harness::persistent_agent::{
    resume, Capacity, Clock, Continuation, EntryPayload, ExecutionIdentity, ExecutionLink,
    ExecutionLinkSource, ExecutionLog, FileLinkStore, FileSessionStore, FinalStatus, LinkRelation,
    LogCapacity, ManualClock, ModelUsage, PersistentSessionRecord, RecordEnding, RootLogSource,
    SessionStateStore, ToolStatus, UnsupportedModelSummarizer, WindowAssembler, WindowSummarizer,
};
use openjiuwen::harness::rails::prompt_sections;
use openjiuwen::harness::{HarnessRail, HarnessRailContext};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use tokio_util::sync::CancellationToken;

use common::banner;

/// The problem every reference model is asked about. Its keywords (four letters or more)
/// are what the structural ranking rewards, so a proposal that answers them scores higher.
const PROBLEM: &str = "the failing build in src/main.rs breaks the release pipeline";

/// A detailed, numbered, keyword-covering proposal: structured, long and fenced, so the
/// structural score is high.
const DETAILED_PROPOSAL: &str = "\
1. Reproduce the failing build: run `cargo build --release` from the workspace root.\n\
2. Read `src/main.rs` and the release pipeline definition to find the step that breaks.\n\
3. Fix the failing step, then re-run the build so the same failure cannot come back.\n\
```text\n\
build: ok\n\
```\n\
- Confirm the release pipeline is green before merging the change.";

/// A terse proposal: usable, but it answers one keyword and carries no structure, so it
/// ranks below the detailed one.
const TERSE_PROPOSAL: &str = "Rebuild the build and retry.";

/// The instruction a continuation appends to the original task input.
const RESUME_INSTRUCTION: &str = "finish the lint fix";

/// Goal recorded at the head of the execution journal.
const GOAL: &str = "fix the failing build";

/// Frozen instant every injected clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;

/// File the consultation history is saved to inside the temporary root.
const HISTORY_FILE: &str = "moa-history.json";

/// Token budget the recovered window is re-assembled under, small enough to force an
/// elision and therefore a summariser call.
const WINDOW_BUDGET: usize = 24;

/// Returns the first line of `text`, for one-line notes about multi-line payloads.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("")
}

/// Truncates `text` to `limit` characters for display, never splitting one in half.
fn trim_to(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_owned();
    }
    let head: String = trimmed.chars().take(limit).collect();
    format!("{head}…")
}

/// Names of the files in the journal directory, sorted.
///
/// The absolute path never reaches stdout: the reader sees `entries.jsonl`,
/// `entries.1.jsonl` and `meta.json`, which is what a rotation produces.
fn journal_files(dir: &Path) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| anyhow!("cannot read the journal directory: {error}"))?;
    let mut names: Vec<String> = entries
        .map(|entry| {
            let entry =
                entry.map_err(|error| anyhow!("cannot list the journal directory: {error}"))?;
            Ok(entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    names.sort();
    Ok(names)
}

/// Part A: consult the references, rank them, aggregate them, then drive the rail hooks.
async fn run_moa(root: &Path) -> Result<()> {
    banner::step("MoA: three reference models, one of them failing");

    let detailed = ScriptedModel::text(DETAILED_PROPOSAL);
    let terse = ScriptedModel::text(TERSE_PROPOSAL);
    let config = MoaConfig::new(vec![
        common::arc_model(detailed.clone()),
        common::arc_model(terse.clone()),
        // A reference that always fails: a partial batch records the failure and keeps the
        // other proposals, only an all-fail batch is an error.
        common::arc_model(ScriptedModel::new([ScriptedTurn::fail(
            "provider returned 503",
        )])),
    ])
    .with_mode(MoaMode::AggregatorOnDemand)
    .with_max_concurrency(3)
    .with_proposer_temperature(0.2)
    .with_history_limit(2);
    config.validate()?;
    banner::note(&format!(
        "config: mode={} references={} proposals-per-batch={} concurrency={}",
        config.mode,
        config.reference_models.len(),
        config.proposal_count(),
        config.max_concurrency
    ));
    banner::note(&format!("problem: {PROBLEM}"));

    let messages = vec![BaseMessage::user(PROBLEM)];
    let cancel = CancellationToken::new();

    // --- 1. consult ---------------------------------------------------------
    banner::step("consult references");
    let batch = consult(&config, &messages, &cancel).await?;
    ensure!(
        batch.total_models == 3 && batch.success_count == 2 && batch.failures.len() == 1,
        "expected two answers out of three references, got {batch:?}"
    );
    banner::note(&format!(
        "asked={} answered={} failed={}",
        batch.total_models,
        batch.success_count,
        batch.failures.len()
    ));
    for source in &batch.sources {
        banner::note(&format!(
            "proposal {} ({} chars): {}",
            source.index,
            source.text.chars().count(),
            trim_to(&source.text, 44)
        ));
    }
    for failure in &batch.failures {
        banner::note(&format!(
            "proposal {} failed: {}",
            failure.index, failure.error
        ));
    }
    ensure!(
        detailed.call_count() == 1 && terse.call_count() == 1,
        "each reference is asked exactly once"
    );
    banner::note(&format!(
        "model calls: detailed={} terse={} (the failing reference has no recorded call)",
        detailed.call_count(),
        terse.call_count()
    ));

    // --- 2. rank ------------------------------------------------------------
    banner::step("rank");
    let ranking = rank_with_config(&config, PROBLEM, &batch.sources).await?;
    ensure!(
        ranking.len() == batch.sources.len(),
        "every retained proposal is ranked"
    );
    for entry in &ranking {
        banner::note(&format!(
            "#{} score {:.2} — {}",
            entry.index,
            entry.score,
            entry.reasons.first().map_or("no reason", String::as_str)
        ));
    }
    ensure!(
        ranking.first().is_some_and(|top| top.index == 1),
        "the structured proposal must rank first: {ranking:?}"
    );

    // --- 3. aggregate -------------------------------------------------------
    banner::step("aggregate");
    let aggregate = Aggregator::new(AggregationStrategy::Labelled)
        .aggregate(&config, PROBLEM, &batch.sources, &cancel)
        .await?;
    let headings: Vec<&str> = aggregate
        .lines()
        .filter(|line| line.starts_with("### Proposal"))
        .collect();
    banner::note(&format!(
        "labelled answer: {} chars, {} proposal section(s)",
        aggregate.chars().count(),
        headings.len()
    ));
    banner::note(first_line(&aggregate));
    for heading in &headings {
        banner::note(heading);
    }
    let synthesis_error = Aggregator::new(AggregationStrategy::Synthesis)
        .aggregate(&config, PROBLEM, &batch.sources, &cancel)
        .await
        .err()
        .ok_or_else(|| anyhow!("synthesis without an aggregator model must be refused"))?;
    ensure!(
        synthesis_error.kind() == ErrorKind::Config,
        "an unconfigured synthesis is a configuration error"
    );
    banner::note(&format!(
        "synthesis without an aggregator model → {}: {}",
        synthesis_error.kind(),
        first_line(&synthesis_error.message())
    ));

    // --- 4. typed refusals --------------------------------------------------
    banner::step("typed failures");
    let all_fail = MoaConfig::new(vec![
        common::arc_model(ScriptedModel::new([ScriptedTurn::fail("first boom")])),
        common::arc_model(ScriptedModel::new([ScriptedTurn::fail("second boom")])),
    ]);
    let all_fail_error = consult(&all_fail, &messages, &cancel)
        .await
        .err()
        .ok_or_else(|| anyhow!("an all-fail batch must not succeed"))?;
    ensure!(
        all_fail_error.kind() == ErrorKind::Llm,
        "an all-fail batch is an LLM error"
    );
    banner::note(&format!(
        "all references failed → {}: {}",
        all_fail_error.kind(),
        first_line(&all_fail_error.message())
    ));

    // `jev_routed` needs an external System One router this port does not provide. The
    // configuration itself is valid, so the refusal comes from the consultation path.
    let jev = MoaConfig::new(vec![common::scripted(TERSE_PROPOSAL)])
        .with_mode(MoaMode::JevRouted)
        .with_jev(JevConfig::default());
    jev.validate()?;
    let jev_error = MoaRail::new(jev)?
        .consult(&messages, &cancel)
        .await
        .err()
        .ok_or_else(|| anyhow!("jev_routed must be a typed refusal"))?;
    ensure!(
        jev_error.kind() == ErrorKind::Unsupported,
        "jev_routed reports Unsupported, got {jev_error:?}"
    );
    banner::note(&format!(
        "jev_routed → {}: {}",
        jev_error.kind(),
        first_line(&jev_error.message())
    ));

    // --- 5. rail hooks and history -----------------------------------------
    banner::step("rail hooks and history");
    // A fresh pair of proposers: a `ScriptedModel` falls back once its script is spent, so
    // the rail gets its own handles instead of borrowing the ones above.
    let rail_detailed = ScriptedModel::text(DETAILED_PROPOSAL);
    let rail_terse = ScriptedModel::text(TERSE_PROPOSAL);
    let rail = MoaRail::new(
        MoaConfig::new(vec![
            common::arc_model(rail_detailed.clone()),
            common::arc_model(rail_terse.clone()),
        ])
        .with_max_concurrency(2)
        .with_history_limit(2),
    )?;
    let section = rail.prompt_section("en");
    banner::note(&format!(
        "rail `{}` publishes section `{}` (priority {}, {} chars en)",
        rail.name(),
        section.name,
        section.priority,
        section.char_count("en")
    ));

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
    ctx.base.messages = messages.clone();
    rail.before_invoke(&mut ctx).await?;
    let sections = prompt_sections(&ctx);
    ensure!(
        sections.len() == 1 && sections[0].name == MOA_PROMPT_SECTION,
        "before_invoke must publish exactly the MoA section: {sections:?}"
    );
    banner::note(&format!(
        "before_invoke: section `{}` published into `harness.prompt_sections`",
        sections[0].name
    ));
    let installed: MoaSnapshot = serde_json::from_value(
        ctx.extra()
            .get(MOA_STATE_EXTRA_KEY)
            .cloned()
            .ok_or_else(|| anyhow!("before_invoke must install the invocation state"))?,
    )?;
    banner::note(&format!(
        "installed state: reference_calls={} verifier_calls={} completed_turns={}",
        installed.reference_calls, installed.verifier_calls, installed.completed_turns
    ));

    let outcome: ConsultationOutcome = rail.consult(&ctx.base.messages, &cancel).await?;
    ensure!(
        outcome.success_count == 2 && outcome.ranking.len() == 2,
        "the rail consultation must rank both proposals: {outcome:?}"
    );
    ensure!(
        rail.last_outcome().as_ref() == Some(&outcome),
        "the rail must remember its last outcome"
    );
    let aggregate_chars = outcome
        .aggregate
        .as_deref()
        .map_or(0, |answer| answer.chars().count());
    banner::note(&format!(
        "rail.consult: answered={} failed={} ranked={} aggregate={} chars",
        outcome.success_count,
        outcome.failed_count,
        outcome.ranking.len(),
        aggregate_chars
    ));
    if let Some(top) = outcome.ranking.first() {
        banner::note(&format!(
            "top proposal: #{} score {:.2}",
            top.index, top.score
        ));
    }

    // The rail never calls a model from a hook: the host records the steps and asks for the
    // score itself, which is what the next two calls do.
    let accepted = rail.record_step(
        "Action: read src/main.rs; response: patch the failing step\n\
         Observations: cargo build --release fails on the release pipeline",
    );
    ensure!(
        accepted,
        "a progress step needs the installed invocation state"
    );
    let steps = rail.recorded_steps();
    let progress = rail.verify_progress(&steps).await?;
    ensure!(
        (0.0..=1.0).contains(&progress),
        "a progress score stays in 0..=1, got {progress}"
    );
    banner::note(&format!(
        "recorded_steps={} progress score={progress:.2} (structural verifier, no model)",
        steps.len()
    ));

    ctx.base.event = RailEvent::BeforeModelCall;
    rail.before_model_call(&mut ctx).await?;
    let published = ctx
        .extra()
        .get(MOA_OUTCOME_EXTRA_KEY)
        .cloned()
        .ok_or_else(|| anyhow!("before_model_call must publish the outcome"))?;
    ensure!(
        published.get("success_count") == Some(&serde_json::json!(2)),
        "the published outcome must be the recorded one: {published}"
    );
    banner::note(&format!(
        "before_model_call: outcome published under `{MOA_OUTCOME_EXTRA_KEY}` (success_count=2)"
    ));

    ctx.base.event = RailEvent::AfterInvoke;
    rail.after_invoke(&mut ctx).await?;
    ensure!(
        rail.snapshot().is_none() && !ctx.extra().contains_key(MOA_STATE_EXTRA_KEY),
        "after_invoke must retire the invocation state"
    );
    banner::note(&format!(
        "after_invoke: state retired, `{MOA_STATE_EXTRA_KEY}` gone, {} context key(s) left",
        ctx.extra().len()
    ));

    let history = rail.history();
    ensure!(
        history.len() == 1 && history[0].problem == PROBLEM,
        "one bounded history record must be appended: {history:?}"
    );
    banner::note(&format!(
        "history: {} record(s) (limit 2), newest problem: {}",
        history.len(),
        trim_to(&history[0].problem, 48)
    ));
    let history_path = root.join(HISTORY_FILE);
    rail.save_history(&history_path).await?;
    rail.load_history(&history_path).await?;
    ensure!(
        rail.history().len() == 1,
        "the history round-trip keeps the record"
    );
    banner::note(&format!(
        "history round-trip through `{HISTORY_FILE}`: {} record(s) loaded back",
        rail.history().len()
    ));

    Ok(())
}

/// Part B: turn the journal, crash on it, replay it, and resume from its record.
async fn run_persistent(root: &Path) -> Result<()> {
    banner::step("persistent agent: append, rotate, crash, replay");
    // The clock is a trait object so the same handle stamps the journal and the log source.
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(NOW_MS));
    let identity = ExecutionIdentity::new("ns", "agent", "session", "req1", "exec-1", "source")?;
    let binding = identity.session_binding();
    let logs_root = root.join("logs");
    // `RootLogSource` expects `<root>/logs/<directory name of the execution>`, so the journal
    // is opened at exactly that path.
    let log_dir = logs_root.join(identity.directory_name());
    banner::note(&format!(
        "journal: logs/{} (clock frozen at {NOW_MS} ms)",
        identity.directory_name()
    ));

    let digest = {
        let log = ExecutionLog::open_with_clock(
            &log_dir,
            identity.clone(),
            LogCapacity::default(),
            Arc::clone(&clock),
        )
        .await?;
        log.append_many(&[
            EntryPayload::execution_started(GOAL),
            EntryPayload::note("read the failing test"),
            EntryPayload::model_turn(
                "mc_1",
                "the release build fails on the lint step",
                ModelUsage::known(120, 40),
            ),
        ])
        .await?;
        // Rotating mid-run must not lose or renumber anything: sequence numbers are
        // journal-wide and keep growing across segments.
        log.rotate_now().await?;
        banner::note(&format!(
            "after the first segment: next_seq={} generation={}",
            log.next_seq().await,
            log.cursor().await.generation
        ));
        log.append_many(&[
            EntryPayload::tool_intent(
                "call_1",
                "read_file",
                "{\"path\":\"src/main.rs\"}",
                "d4e5f6",
                "op_1",
            ),
            EntryPayload::tool_completed(
                "call_1",
                "read_file",
                ToolStatus::Succeeded,
                "{\"path\":\"src/main.rs\"}",
                "fn main() { println!(\"build\"); }",
            ),
            EntryPayload::model_turn("mc_2", "fixed the lint step", ModelUsage::known(90, 30)),
            EntryPayload::reply_delta("fixed"),
            EntryPayload::execution_final(FinalStatus::Completed, None),
        ])
        .await?;
        log.digest().await?
    };
    // The handle above is gone; the process "crashed". Everything below reads what really
    // reached disk.
    banner::note(&format!("before the crash: {}", digest.summary()));
    banner::note(&format!(
        "segments on disk: {}",
        journal_files(&log_dir)?.join(", ")
    ));

    banner::step("replay");
    let reopened = ExecutionLog::open_with_clock(
        &log_dir,
        identity.clone(),
        LogCapacity::default(),
        Arc::clone(&clock),
    )
    .await?;
    let replay = reopened.replay().await?;
    ensure!(
        replay.entries.len() == 8,
        "the replay keeps every entry across the rotation, got {}",
        replay.entries.len()
    );
    ensure!(
        !replay.truncated,
        "the default capacity retains the whole run"
    );
    ensure!(
        replay.cursor.next_seq == 9,
        "the cursor continues after the last entry, got {}",
        replay.cursor.next_seq
    );
    ensure!(
        replay.digest.input_tokens == 210,
        "the digest totals both model turns, got {}",
        replay.digest.input_tokens
    );
    banner::note(&format!(
        "replay: entries={} next_seq={} first_seq={:?} last_seq={:?} truncated={}",
        replay.entries.len(),
        replay.cursor.next_seq,
        replay.digest.first_seq,
        replay.digest.last_seq,
        replay.truncated
    ));
    banner::note(&format!("digest: {}", replay.digest.summary()));
    let kinds: Vec<String> = replay
        .entries
        .iter()
        .map(|entry| format!("{}:{}", entry.seq, entry.kind().as_str()))
        .collect();
    banner::note(&format!("entries in order: {}", kinds.join(" ")));

    banner::step("resume plan");
    let store = FileSessionStore::new(root.join("sessions"));
    let mut record = PersistentSessionRecord::new(&binding)?;
    record.messages = vec![BaseMessage::user(GOAL)];
    let record = record.with_ending(RecordEnding {
        outcome: "completed".to_owned(),
        request_id: "req1".to_owned(),
        execution_id: "exec-1".to_owned(),
        ..RecordEnding::default()
    })?;
    store.save(&record).await?;
    let links = FileLinkStore::new(root.join("links"));
    links
        .write_link(&ExecutionLink::from_identity(
            &identity,
            None,
            LinkRelation::First,
        )?)
        .await?;
    banner::note(&format!(
        "session `{}` saved with a completed ending; `exec-1` is a first execution",
        binding.key_prefix()
    ));

    let continuation = Continuation::new("exec-1", Some(RESUME_INSTRUCTION), 4)?;
    let plan = resume(
        &store,
        &RootLogSource::new(logs_root.clone(), LogCapacity::default())
            .with_clock(Arc::clone(&clock)),
        &links,
        &binding,
        "req2",
        &continuation,
        &WindowAssembler::new(Capacity::default()),
    )
    .await?;
    ensure!(
        plan.cursor.resume_seq == 9,
        "the plan continues the journal at 9, got {}",
        plan.cursor.resume_seq
    );
    ensure!(
        plan.cursor.recovery_ref.as_deref() == Some("exec-1"),
        "the plan points at the journalled execution: {:?}",
        plan.cursor.recovery_ref
    );
    ensure!(
        plan.digest.entries == 8 && plan.digest.input_tokens == 210,
        "the plan carries the recovery journal's digest: {:?}",
        plan.digest
    );
    ensure!(
        plan.background.contains("exec-1"),
        "the background names the recovered execution"
    );
    ensure!(
        plan.record.continuation.as_ref() == Some(&plan.cursor),
        "the resumed record keeps the new cursor"
    );
    banner::note(&format!(
        "cursor: recovery_ref={:?} resume_seq={} predecessor={:?} relation={}",
        plan.cursor.recovery_ref,
        plan.cursor.resume_seq,
        plan.cursor.predecessor_id,
        plan.cursor.relation.map_or("none", LinkRelation::as_str)
    ));
    banner::note(&format!(
        "window: messages={} tokens={} budget={} over_budget={}",
        plan.window.messages.len(),
        plan.window.tokens,
        plan.window.budget,
        plan.window.over_budget
    ));
    for message in &plan.window.messages {
        banner::note(&format!(
            "  {}: {}",
            message.role.as_str(),
            trim_to(&message.text(), 64)
        ));
    }
    banner::note(&format!(
        "digest: entries={} input_tokens={} output_tokens={}",
        plan.digest.entries, plan.digest.input_tokens, plan.digest.output_tokens
    ));
    banner::note(&format!(
        "background: {} lines, chain={} link(s), names `exec-1`: {}",
        plan.background.lines().count(),
        plan.chain.len(),
        plan.background.contains("exec-1")
    ));
    if let Some(first) = plan.background.lines().next() {
        banner::note(&format!("  {}", trim_to(first, 110)));
    }
    ensure!(
        continuation
            .continued_query(GOAL)
            .contains(RESUME_INSTRUCTION),
        "the continuation appends its instruction to the original task input"
    );
    banner::note(&format!(
        "continued query: {}",
        trim_to(&continuation.continued_query(GOAL), 90)
    ));

    banner::step("window budget and the model-summariser seam");
    let summarizer_error = UnsupportedModelSummarizer
        .summarize(&replay.entries)
        .err()
        .ok_or_else(|| anyhow!("the model-backed summariser must be a typed refusal"))?;
    ensure!(
        summarizer_error.kind() == ErrorKind::Unsupported,
        "the model summariser reports Unsupported, got {summarizer_error:?}"
    );
    banner::note(&format!(
        "model summariser → {}: {}",
        summarizer_error.kind(),
        first_line(&summarizer_error.message())
    ));
    let assembly = WindowAssembler::new(Capacity::default())
        .with_summarizer(Arc::new(UnsupportedModelSummarizer))
        .assemble_with_budget(&replay.entries, WINDOW_BUDGET);
    banner::note(&format!(
        "assemble_with_budget({WINDOW_BUDGET} tokens): messages={} elided={} over_budget={}",
        assembly.messages.len(),
        assembly.elided_count(),
        assembly.over_budget
    ));
    banner::note(&format!(
        "summary: {}",
        assembly
            .summary
            .as_deref()
            .map_or_else(|| "<none>".to_owned(), |text| trim_to(text, 90))
    ));

    banner::step("what this showed");
    banner::note(
        "MoA is offline-composable: proposals are ranked structurally, a partial batch keeps \
         its successful references, and only an all-fail batch or an unported router is an error",
    );
    banner::note(
        "the rail is a pure context mutator: it publishes a prompt section and an outcome, while \
         the host records progress steps and decides when to consult",
    );
    banner::note(
        "the journal is the durable truth: it survives a rotation and a dropped handle, and a \
         resume plan is rebuilt from the entries rather than from process memory",
    );
    banner::answer("two proposals ranked, one labelled answer, one execution recovered from disk");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    banner::title("25 harness MoA and persistent agent");
    banner::note(&format!(
        "run mode: {}; every model is scripted, every timestamp is injected, the root is temporary",
        mode.describe()
    ));

    // One `TempDir` backs both halves and stays alive until the process exits.
    let root = common::tempdir()?;
    run_moa(root.path()).await?;
    run_persistent(root.path()).await?;

    Ok(())
}
