//! Sequential continuations keep every earlier turn in the seeded window.
//!
//! The scenario mirrors the CLI probe: the first turn journals one execution, and each
//! `--continue` resolves the persisted record's ending through [`resume`], seeds the window it
//! returns and journals one more execution. The assertion is on what the model actually receives,
//! because a dropped turn is invisible in the record (which is rewritten with the seeded window)
//! and only shows up as a model request missing the oldest questions.
//!
//! Everything runs offline against a `tempfile` root and a [`ScriptedModel`].
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_foundation::prompt::message::BaseMessage;
use openjiuwen_harness::cli::{
    CliIo, CliOptions, LineReader, RunArgs, Writer, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL, DEFAULT_PROVIDER,
};
use openjiuwen_harness::{
    resume, Capacity, Continuation, ExecutionCatalog, FileLinkStore, FileSessionStore, LogCapacity,
    OutputFormat, PersistentAgentLayout, ResumePlan, ResumedTurn, RootLogSource, SessionStateStore,
    WindowAssembler, Workspace,
};
use openjiuwen_test_support::ScriptedModel;

/// Session id every turn continues.
const SESSION_ID: &str = "session-continuation";
/// One question per turn, oldest first.
const QUESTIONS: [&str; 4] = [
    "first question",
    "second question",
    "third question",
    "fourth question",
];
/// One answer per turn, oldest first.
const ANSWERS: [&str; 4] = [
    "first answer",
    "second answer",
    "third answer",
    "fourth answer",
];
/// The chain limit the CLI resumes with.
const CHAIN_LIMIT: usize = 4;

/// [`LineReader`] that is never consulted: every turn carries its prompt in the arguments.
struct EmptyReader;

#[async_trait::async_trait]
impl LineReader for EmptyReader {
    async fn read_line(&mut self) -> openjiuwen_common::error::Result<Option<String>> {
        Ok(None)
    }
}

/// [`Writer`] that discards the rendered stream.
struct SilentWriter;

#[async_trait::async_trait]
impl Writer for SilentWriter {
    async fn write(&mut self, _text: &str) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    async fn flush(&mut self) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }
}

/// Builds the non-interactive CLI handle of a one-shot run.
fn silent_io() -> CliIo {
    CliIo::new(Box::new(EmptyReader), Box::new(SilentWriter))
}

/// Resolves the CLI options of a run rooted at `root`.
fn cli_options(root: &std::path::Path) -> Result<CliOptions> {
    let options = CliOptions {
        model: DEFAULT_MODEL.to_owned(),
        provider: DEFAULT_PROVIDER.to_owned(),
        api_key: None,
        api_base: None,
        workspace: root.join("workspace"),
        max_iterations: DEFAULT_MAX_ITERATIONS,
        max_tokens: DEFAULT_MAX_TOKENS,
        verbose: false,
        home: OpenJiuwenHome::from_root(root),
    };
    options.ensure_workspace()?;
    Ok(options)
}

/// Resolves the plan a `--continue` resumes from, exactly as the CLI resolves it.
///
/// The persisted record names the execution it ended with; [`resume`] traces that execution's
/// links back across the earlier continuations and rebuilds the seed window.
async fn continued_plan(layout: &PersistentAgentLayout, request_id: &str) -> Result<ResumePlan> {
    let store = FileSessionStore::new(layout.sessions_root());
    let summary = store
        .list()
        .await?
        .into_iter()
        .find(|summary| summary.binding.session_id == SESSION_ID)
        .ok_or_else(|| anyhow!("the continued session must have a persisted record"))?;
    let record = store
        .load(&summary.binding)
        .await?
        .ok_or_else(|| anyhow!("the persisted record must load"))?;
    let recovery = record
        .ending
        .as_ref()
        .map(|ending| ending.execution_id.clone())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| anyhow!("the completed turn must name its execution"))?;
    let plan = resume(
        &store,
        &RootLogSource::new(layout.logs_root(), LogCapacity::default()),
        &FileLinkStore::new(layout.links_root()),
        &summary.binding,
        request_id,
        &Continuation::new(recovery, None, CHAIN_LIMIT)?,
        &WindowAssembler::new(Capacity::default()),
    )
    .await?;
    Ok(plan)
}

/// Runs one turn and returns the model so the test can inspect what it was asked.
async fn run_turn(
    root: &std::path::Path,
    question: &str,
    answer: &str,
    history: Option<Vec<BaseMessage>>,
) -> Result<ScriptedModel> {
    let options = cli_options(root)?;
    let io = silent_io();
    let model = ScriptedModel::text(answer);
    let recorded = model.clone();
    let args = RunArgs {
        prompt: Some(question.to_owned()),
        output_format: OutputFormat::Text,
    };
    let resumed = match history {
        Some(history) => ResumedTurn::new(SESSION_ID).with_history(history),
        None => ResumedTurn::new(SESSION_ID),
    };
    let outcome =
        openjiuwen_harness::run_with_model_resumed(&options, &args, &io, Arc::new(model), resumed)
            .await?;
    ensure!(
        outcome.answer == answer && outcome.session_id == SESSION_ID,
        "unexpected turn outcome: {outcome:?}"
    );
    Ok(recorded)
}

/// Asserts every needle names one message and that they appear oldest first.
fn assert_once_in_order(messages: &[BaseMessage], needles: &[&str]) -> Result<()> {
    let mut previous: Option<usize> = None;
    for needle in needles {
        let positions: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message.text().contains(needle))
            .map(|(index, _)| index)
            .collect();
        ensure!(
            positions.len() == 1,
            "{needle:?} must appear exactly once, got messages {messages:?}"
        );
        if let Some(previous) = previous {
            ensure!(
                previous < positions[0],
                "{needle:?} must follow the earlier turn: {messages:?}"
            );
        }
        previous = Some(positions[0]);
    }
    Ok(())
}

/// Three continuations later the model still sees every earlier question exactly once.
#[tokio::test]
async fn sequential_continuations_keep_every_earlier_turn() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let layout = PersistentAgentLayout::for_workspace(&Workspace::for_language(
        root.path().join("workspace"),
        "en",
    ));

    // Turn one starts the session; it has nothing to continue.
    run_turn(root.path(), QUESTIONS[0], ANSWERS[0], None).await?;

    for index in 1..QUESTIONS.len() {
        let plan = continued_plan(&layout, &format!("req-continue-{index}")).await?;
        ensure!(
            plan.chain.len() == index,
            "continuation {index} must trace {index} earlier executions, got {}",
            plan.chain.len()
        );
        let history = plan.seed_history();
        ensure!(
            !history.is_empty(),
            "a continuation must seed a non-empty window"
        );
        let model = run_turn(root.path(), QUESTIONS[index], ANSWERS[index], Some(history)).await?;
        let call = model
            .last_call()
            .ok_or_else(|| anyhow!("turn {index} must call the model"))?;
        let expected = &QUESTIONS[..=index];
        assert_once_in_order(&call.messages, expected)?;
    }

    // Every turn journalled one more execution, so the chain the last record points at is whole.
    let executions = ExecutionCatalog::new(layout.logs_root(), LogCapacity::default())
        .list()
        .await?;
    ensure!(
        executions.len() == QUESTIONS.len(),
        "each continuation must journal its own execution, got {}",
        executions.len()
    );
    Ok(())
}
