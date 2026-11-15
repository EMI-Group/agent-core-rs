//! Offline integration coverage for `DeepAgent` persistent-agent recording.
//!
//! The suite drives two real turns through [`DeepAgent`] with `persistent_agent` enabled and then
//! re-derives everything from disk alone: the catalogued execution and its journal entries, the
//! persisted session record, the resume plan rebuilt from the journal, and a second agent that
//! continues the seeded transcript. Nothing is handed from the first turn to the second except the
//! `ResumePlan`'s history, which is the whole point of the feature.
//!
//! Every await is bounded and the model is a [`ScriptedModel`], so no network, wall clock or real
//! model is involved.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, ensure, Result};
use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_foundation::prompt::message::BaseMessage;
use openjiuwen_harness::{
    resume, Capacity, Continuation, DeepAgent, DeepAgentBuilder, DeepAgentConfig, EntryKind,
    EntryPayload, ExecutionCatalog, ExecutionLogSource, FileLinkStore, FileSessionStore,
    LogCapacity, PersistentAgentLayout, ResumePlan, RootLogSource, SessionBinding, WindowAssembler,
    Workspace,
};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;

/// Longest any single awaited step may take before the test fails.
const BOUND: Duration = Duration::from_secs(10);

/// Identity of the agent every turn runs on.
const AGENT_ID: &str = "recording-agent";

/// Session both turns share.
const SESSION_ID: &str = "recording-session";

/// Request id the second turn records its continuation under.
const NEXT_REQUEST: &str = "req-next";

/// Query of the first turn.
const QUESTION: &str = "hello question";

/// Answer of the first turn.
const ANSWER: &str = "hello answer";

/// Query of the second, continued turn.
const NEXT_QUESTION: &str = "follow-up question";

/// Answer of the second, continued turn.
const NEXT_ANSWER: &str = "follow-up answer";

/// Returns the card of the recorded agent, whose id is the journal's agent id.
///
/// `AgentCard::new` sets the name and mints a UUID id, so the id is set explicitly: the journal
/// records the card's id, and the resume fixture has to name the same one.
fn card() -> AgentCard {
    AgentCard::new(AGENT_ID).with_id(AGENT_ID)
}

/// Builds a recording agent over `workspace`.
fn build_agent(workspace: &Workspace, model: ScriptedModel) -> Result<Arc<DeepAgent>> {
    let config = DeepAgentConfig::default()
        .with_language("en".to_owned())
        .with_workspace(workspace.clone())
        .with_persistent_agent(true);
    Ok(DeepAgentBuilder::new(card())
        .config(config)
        .model(Arc::new(model))
        .build()?)
}

/// Builds the kernel session both turns share.
fn session() -> Arc<dyn Session> {
    Arc::new(AgentSession::new(SESSION_ID))
}

/// Returns the index of the first message whose text contains `needle`.
fn position_of(messages: &[BaseMessage], needle: &str) -> Result<usize> {
    messages
        .iter()
        .position(|message| message.text().contains(needle))
        .ok_or_else(|| anyhow!("no request message carries {needle:?}"))
}

/// Asserts the first turn is on disk: one execution, a complete journal and one session record.
///
/// Returns the catalogued execution id and the layout everything was read from.
async fn assert_first_turn_recorded(
    workspace: &Workspace,
) -> Result<(String, PersistentAgentLayout)> {
    let layout = PersistentAgentLayout::for_workspace(workspace);

    let executions = ExecutionCatalog::new(layout.logs_root(), LogCapacity::default())
        .list()
        .await?;
    ensure!(
        executions.len() == 1,
        "one turn journals exactly one execution, got {}",
        executions.len()
    );
    let execution = &executions[0];
    ensure!(
        execution.agent_id == AGENT_ID && execution.session_id == SESSION_ID,
        "the execution must belong to the recorded agent session: {execution:?}"
    );
    ensure!(
        execution.usable && execution.execution_id.starts_with("exec_"),
        "the journal must be readable and keep the minted execution id: {execution:?}"
    );
    ensure!(
        execution.request_id.starts_with("req_"),
        "an unnamed request gets a minted id: {execution:?}"
    );

    // The journal carries the question, the answer and the outcome, in that order.
    let log = RootLogSource::new(layout.logs_root(), LogCapacity::default())
        .open_log(&execution.execution_id)
        .await?;
    let entries = log.read().await?.entries;
    let kinds: Vec<EntryKind> = entries.iter().map(|entry| entry.kind()).collect();
    ensure!(
        kinds.len() == 3
            && kinds[0] == EntryKind::ExecutionStarted
            && kinds[1] == EntryKind::ModelTurn
            && kinds[2] == EntryKind::ExecutionFinal,
        "unexpected journal kinds: {kinds:?}"
    );
    match (&entries[0].payload, &entries[1].payload) {
        (EntryPayload::ExecutionStarted { goal }, EntryPayload::ModelTurn { response, .. }) => {
            ensure!(
                goal.contains(QUESTION),
                "the journalled goal carries the question, got {goal:?}"
            );
            ensure!(
                response.contains(ANSWER),
                "the journalled model turn carries the answer, got {response:?}"
            );
        }
        other => return Err(anyhow!("unexpected journal payloads: {other:?}")),
    }

    let sessions = FileSessionStore::new(layout.sessions_root()).list().await?;
    ensure!(
        sessions.len() == 1,
        "one turn persists exactly one session record, got {}",
        sessions.len()
    );
    ensure!(
        sessions[0].binding == SessionBinding::new(AGENT_ID, SESSION_ID)?,
        "the record must belong to the recorded session: {:?}",
        sessions[0]
    );
    ensure!(
        sessions[0].has_ending && !sessions[0].waiting,
        "a finished turn closes the record: {:?}",
        sessions[0]
    );
    Ok((execution.execution_id.clone(), layout))
}

/// A recording `DeepAgent` journals its turn, persists the record and resumes from the journal.
#[tokio::test]
async fn a_turn_is_journalled_and_resumes_into_a_continued_transcript() -> Result<()> {
    let dir = tempfile::TempDir::new()?;
    let workspace = Workspace::for_language(dir.path().join("workspace"), "en");

    // --- turn one: record it --------------------------------------------------
    tokio::time::timeout(BOUND, async {
        let agent = build_agent(&workspace, ScriptedModel::text(ANSWER))?;
        let output = agent
            .invoke(json!({ "query": QUESTION }), Some(session()))
            .await?;
        ensure!(output.output == ANSWER, "unexpected answer: {output:?}");
        Ok::<(), anyhow::Error>(())
    })
    .await
    .map_err(|_| anyhow!("the recorded turn must finish within {BOUND:?}"))??;

    let (execution_id, layout) = assert_first_turn_recorded(&workspace).await?;

    // --- rediscover and plan from disk alone ----------------------------------
    let binding = SessionBinding::new(AGENT_ID, SESSION_ID)?;
    let store = FileSessionStore::new(layout.sessions_root());
    let plan: ResumePlan = tokio::time::timeout(BOUND, async {
        resume(
            &store,
            &RootLogSource::new(layout.logs_root(), LogCapacity::default()),
            &FileLinkStore::new(layout.links_root()),
            &binding,
            NEXT_REQUEST,
            &Continuation::new(execution_id.clone(), None, 4)?,
            &WindowAssembler::new(Capacity::default()),
        )
        .await
    })
    .await
    .map_err(|_| anyhow!("resume must finish within {BOUND:?}"))??;

    let history = plan.seed_history();
    ensure!(
        history
            .iter()
            .any(|message| message.text().contains(QUESTION)),
        "the rebuilt window carries the journalled question: {history:?}"
    );
    ensure!(
        history
            .iter()
            .any(|message| message.text().contains(ANSWER)),
        "the rebuilt window carries the journalled answer: {history:?}"
    );
    ensure!(
        plan.cursor.resume_seq == 4,
        "the plan continues the three-entry journal, got {}",
        plan.cursor.resume_seq
    );

    // --- turn two: continue the seeded transcript ----------------------------
    let model = ScriptedModel::text(NEXT_ANSWER);
    let recorded = model.clone();
    tokio::time::timeout(BOUND, async {
        let agent = build_agent(&workspace, model)?;
        let session = session();
        agent.seed_history(&session, plan.seed_history()).await?;
        let output = agent
            .invoke(json!({ "query": NEXT_QUESTION }), Some(session))
            .await?;
        ensure!(
            output.output == NEXT_ANSWER,
            "unexpected answer: {output:?}"
        );
        Ok::<(), anyhow::Error>(())
    })
    .await
    .map_err(|_| anyhow!("the continued turn must finish within {BOUND:?}"))??;

    let call = recorded
        .last_call()
        .ok_or_else(|| anyhow!("the continued turn must call the model"))?;
    let question = position_of(&call.messages, QUESTION)?;
    let answer = position_of(&call.messages, ANSWER)?;
    let query = position_of(&call.messages, NEXT_QUESTION)?;
    ensure!(
        question < answer && answer < query,
        "the restored transcript must precede the new query: {question} < {answer} < {query}"
    );

    // The second turn journals its own execution beside the first, under the same session.
    let executions = ExecutionCatalog::new(layout.logs_root(), LogCapacity::default())
        .list()
        .await?;
    ensure!(
        executions.len() == 2,
        "the continued turn records a second execution, got {}",
        executions.len()
    );
    ensure!(
        executions
            .iter()
            .all(|execution| execution.session_id == SESSION_ID),
        "both executions belong to the same session: {executions:?}"
    );
    Ok(())
}

/// A turn whose recording is switched off writes nothing to the persistent-agent layout.
#[tokio::test]
async fn recording_is_off_by_default() -> Result<()> {
    let dir = tempfile::TempDir::new()?;
    let workspace = Workspace::for_language(dir.path().join("workspace"), "en");
    let config = DeepAgentConfig::default()
        .with_language("en".to_owned())
        .with_workspace(workspace.clone());
    assert!(!config.persistent_agent, "recording is opt-in");

    let agent = DeepAgentBuilder::new(card())
        .config(config)
        .model(Arc::new(ScriptedModel::text(ANSWER)))
        .build()?;
    tokio::time::timeout(
        BOUND,
        agent.invoke(json!({ "query": QUESTION }), Some(session())),
    )
    .await
    .map_err(|_| anyhow!("the turn must finish within {BOUND:?}"))??;

    let layout = PersistentAgentLayout::for_workspace(&workspace);
    ensure!(
        ExecutionCatalog::new(layout.logs_root(), LogCapacity::default())
            .list()
            .await?
            .is_empty(),
        "an unrecorded turn journals nothing"
    );
    ensure!(
        FileSessionStore::new(layout.sessions_root())
            .list()
            .await?
            .is_empty(),
        "an unrecorded turn persists no session record"
    );
    Ok(())
}
