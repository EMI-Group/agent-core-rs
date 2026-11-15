//! Offline integration coverage for the persistent-agent surface, driven through the crate root.
//!
//! The suites replay the three flows a crashed coding session depends on — resuming a journalled
//! execution from disk, moving its files through a share bundle, and the full host restart that
//! rediscovers the execution and its CLI session from disk and continues the conversation — plus
//! the typed refusals of the deferred transports.
//!
//! Everything runs against a `tempfile` root with an injected [`ManualClock`] and a
//! [`ScriptedModel`], so no network, wall clock or real model is involved.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_foundation::prompt::message::BaseMessage;
// `CliIo`/`CliOptions`/`RunArgs` and the defaults are only reachable through the `cli` module;
// the resume entry points are spelled through the flat crate root like every other item here.
use openjiuwen_harness::cli::{
    CliIo, CliOptions, LineReader, RunArgs, Writer, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL, DEFAULT_PROVIDER,
};
use openjiuwen_harness::{
    hand_off, resume, run_with_model_resumed, Capacity, Continuation, DistributedHandoff,
    EntryPayload, ExecutionCatalog, ExecutionIdentity, ExecutionLink, ExecutionLinkSource,
    ExecutionLog, ExecutionSummary, FileLinkStore, FileSessionStore, FinalStatus, LinkRelation,
    LogCapacity, ManualClock, ModelUsage, OutputFormat, PersistentAgentLayout,
    PersistentSessionRecord, RecordEnding, RemoteHandoff, ResumePlan, ResumedTurn, RootLogSource,
    SessionBinding, SessionStateStore, SessionSummary, ShareFile, ShareMode, SharedFileEntry,
    WindowAssembler, Workspace,
};
use openjiuwen_test_support::ScriptedModel;

/// Frozen instant every fixture clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;

/// A manual clock frozen at [`NOW_MS`], so every persisted timestamp is deterministic.
fn clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new(NOW_MS))
}

/// The `(agent, session)` pair every fixture targets.
fn binding() -> Result<SessionBinding> {
    Ok(SessionBinding::new("agent", "session")?)
}

/// The source execution the resume fixtures recover.
fn identity() -> Result<ExecutionIdentity> {
    Ok(ExecutionIdentity::new(
        "ns", "agent", "session", "req1", "exec-1", "source",
    )?)
}

/// Writes a completed journal of five entries, rotating in the middle so the journal spans two
/// segments, and returns the segment directory.
async fn journal_completed_execution(root: &std::path::Path) -> Result<std::path::PathBuf> {
    let identity = identity()?;
    let log_dir = root.join("logs").join(identity.directory_name());
    let log =
        ExecutionLog::open_with_clock(&log_dir, identity, LogCapacity::default(), clock()).await?;
    log.append(EntryPayload::execution_started("do work"))
        .await?;
    log.append(EntryPayload::note("first step")).await?;
    log.append(EntryPayload::model_turn(
        "mc_1",
        "done",
        ModelUsage::known(10, 5),
    ))
    .await?;
    // Rotating mid-run must not lose entries nor renumber them.
    log.rotate_now().await?;
    log.append(EntryPayload::note("second step")).await?;
    log.append(EntryPayload::execution_final(FinalStatus::Completed, None))
        .await?;
    ensure!(log.next_seq().await == 6, "five entries, so append at 6");
    Ok(log_dir)
}

/// Persists the session record and the link the resume resolves through.
async fn session_state(root: &std::path::Path) -> Result<(FileSessionStore, FileLinkStore)> {
    let binding = binding()?;
    let store = FileSessionStore::new(root.join("sessions"));
    let mut record = PersistentSessionRecord::new(&binding)?;
    record.messages = vec![BaseMessage::user("do work")];
    let record = record.with_ending(RecordEnding {
        outcome: "completed".to_string(),
        request_id: "req1".to_string(),
        execution_id: "exec-1".to_string(),
        ..RecordEnding::default()
    })?;
    store.save(&record).await?;
    let links = FileLinkStore::new(root.join("links"));
    links
        .write_link(&ExecutionLink::from_identity(
            &identity()?,
            None,
            LinkRelation::First,
        )?)
        .await?;
    Ok((store, links))
}

/// A crashed session reopens from disk, rebuilds its cursor, window and digest, and resumes.
#[tokio::test]
async fn a_crashed_session_resumes_from_its_journal() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let log_dir = journal_completed_execution(root.path()).await?;

    // The handle above was dropped, so everything below reads what actually reached disk.
    let segments = std::fs::read_dir(&log_dir)?
        .filter_map(std::result::Result::ok)
        .count();
    ensure!(
        segments >= 3,
        "two journal segments plus the metadata file, found {segments} entries in {log_dir:?}"
    );

    let reopened =
        ExecutionLog::open_with_clock(&log_dir, identity()?, LogCapacity::default(), clock())
            .await?;
    let replay = reopened.replay().await?;
    ensure!(replay.entries.len() == 5, "a rotation keeps every entry");
    ensure!(!replay.truncated, "the default capacity retains the run");
    ensure!(
        replay.cursor.next_seq == 6,
        "continue at 6, got {}",
        replay.cursor.next_seq
    );
    ensure!(
        replay.digest.first_seq == Some(1) && replay.digest.last_seq == Some(5),
        "sequence numbers survive rotation: {:?}",
        replay.digest
    );
    ensure!(
        replay.digest.input_tokens == 10,
        "the digest totals the model usage"
    );

    let (store, links) = session_state(root.path()).await?;
    let continuation = Continuation::new("exec-1", Some("keep going"), 4)?;
    let plan = resume(
        &store,
        &RootLogSource::new(root.path().join("logs"), LogCapacity::default()).with_clock(clock()),
        &links,
        &binding()?,
        "req2",
        &continuation,
        &WindowAssembler::new(Capacity::default()),
    )
    .await?;

    ensure!(
        plan.cursor.resume_seq == 6,
        "the plan continues the journal"
    );
    ensure!(
        plan.cursor.recovery_ref.as_deref() == Some("exec-1"),
        "the plan points at the journalled execution"
    );
    ensure!(plan.digest.entries == 5, "the plan carries the digest");
    ensure!(
        !plan.window.messages.is_empty(),
        "the recovery journal rebuilds a non-empty window"
    );
    ensure!(
        plan.window
            .messages
            .iter()
            .any(|message| message.text().contains("do work")),
        "the window is assembled from the journal, not from an empty default"
    );
    ensure!(
        plan.background.contains("exec-1"),
        "the continuation prompt names the recovered execution: {}",
        plan.background
    );
    ensure!(
        plan.record.continuation.as_ref() == Some(&plan.cursor),
        "the resumed record keeps the new cursor"
    );
    Ok(())
}

/// A share bundle round-trips its files and payload digests through the public service.
#[tokio::test]
async fn share_bundles_export_and_import_files() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let report = root.path().join("report.md");
    let chart = root.path().join("chart.bin");
    tokio::fs::write(&report, b"# report\n").await?;
    tokio::fs::write(&chart, [0_u8, 1, 2]).await?;

    let service = ShareFile::new(root.path().join("bundles")).with_clock(clock());
    let files = vec![
        SharedFileEntry {
            path: "report.md".to_string(),
            share_mode: ShareMode::Upload,
            caption: Some("the report".to_string()),
            sha256: None,
            size: None,
        },
        SharedFileEntry {
            path: "out/chart.bin".to_string(),
            share_mode: ShareMode::Attachment,
            caption: None,
            sha256: None,
            size: None,
        },
    ];
    let sources = BTreeMap::from([
        ("report.md".to_string(), report),
        ("out/chart.bin".to_string(), chart),
    ]);
    let bundle = service
        .export("bundle-1", &binding()?, &files, &sources)
        .await?;
    ensure!(
        bundle.exported_at == "2023-11-14T22:13:20.000Z",
        "the export is stamped by the injected clock, got {}",
        bundle.exported_at
    );
    ensure!(bundle.files.len() == 2, "both files are listed");

    let imported = service.import_bundle("bundle-1").await?;
    ensure!(
        imported.bundle == bundle,
        "the imported record round-trips through JSON"
    );
    ensure!(imported.files.len() == 2, "both payloads are restored");
    let report_bytes = tokio::fs::read(&imported.files[0].1).await?;
    ensure!(report_bytes == b"# report\n", "the payload bytes match");
    ensure!(
        imported.files[0].0.sha256.as_deref().map(str::len) == Some(64),
        "the digest is recorded as sha256 hex"
    );
    ensure!(
        imported.files[0].0.size == Some(9),
        "the size is recorded from the copied bytes"
    );

    // Importing by path is the same verification path.
    let by_path = service.import(&service.bundle_path("bundle-1")).await?;
    ensure!(by_path.files.len() == 2, "the bundle file imports directly");

    // A tampered payload fails the recorded-digest check instead of importing silently.
    tokio::fs::write(&imported.files[0].1, b"tampered").await?;
    let error = service
        .import_bundle("bundle-1")
        .await
        .expect_err("a tampered payload must be refused");
    ensure!(error.kind() == ErrorKind::Validation, "got {error:?}");
    ensure!(
        error
            .to_string()
            .contains("does not match its recorded digest"),
        "got {error}"
    );
    Ok(())
}

/// The deferred handoff transports answer with their documented typed refusal.
#[tokio::test]
async fn deferred_handoff_transports_are_typed_unsupported() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let identity = identity()?;
    let log_dir = root.path().join("logs").join(identity.directory_name());
    let log =
        ExecutionLog::open_with_clock(&log_dir, identity.clone(), LogCapacity::default(), clock())
            .await?;
    let entry = log.append(EntryPayload::note("handoff payload")).await?;
    ensure!(entry.seq == 1, "the journal numbers from one");
    ensure!(log.cursor().await.next_seq == 2, "and advances");

    let transfer = openjiuwen_harness::HandoffTransfer {
        from: identity,
        to: binding()?,
        record: PersistentSessionRecord::new(&binding()?)?,
        cursor: openjiuwen_harness::ContinuationCursor {
            recovery_ref: Some("exec-1".to_string()),
            predecessor_id: None,
            relation: Some(LinkRelation::First),
            resume_seq: 1,
            last_group_message_id: None,
        },
        window: WindowAssembler::new(Capacity::default()).assemble(&[]),
        next_goal: "summarise the run".to_string(),
        shared_files: Vec::new(),
    };

    let remote = hand_off(&RemoteHandoff, transfer.clone())
        .await
        .expect_err("the remote transport is deferred");
    match remote {
        Error::Unsupported(message) => ensure!(
            message
                == "persistent agent handoff over a remote transport is not implemented; use an \
                    in-process HandoffTarget",
            "unexpected refusal message: {message}"
        ),
        other => anyhow::bail!("expected a typed refusal, got {other:?}"),
    }

    let distributed = hand_off(&DistributedHandoff, transfer)
        .await
        .expect_err("the distributed transport is deferred");
    ensure!(
        distributed.kind() == ErrorKind::Unsupported,
        "the refusal carries the Unsupported kind"
    );
    match distributed {
        Error::Unsupported(message) => ensure!(
            message
                == "persistent agent handoff over a distributed transport is not implemented; use \
                    an in-process HandoffTarget",
            "unexpected refusal message: {message}"
        ),
        other => anyhow::bail!("expected a typed refusal, got {other:?}"),
    }
    Ok(())
}

// --- host restart: journal -> restart -> rediscover -> continue -------------

/// CLI session id the restart scenario continues (the kernel session and the record share it).
const CLI_SESSION_ID: &str = "session-restart";

/// Host namespace the journalled execution lives in.
const NAMESPACE: &str = "host";

/// Agent id of the CLI session, i.e. the agent half of its [`SessionBinding`].
const AGENT_ID: &str = "cli-agent";

/// Execution id of the journalled first turn.
const EXECUTION_ID: &str = "exec-cli-1";

/// Request id of the journalled first turn.
const FIRST_REQUEST: &str = "req-1";

/// Request id of the continued turn.
const SECOND_REQUEST: &str = "req-2";

/// User text of the first turn.
const FIRST_QUESTION: &str = "journal the resume path";

/// Assistant text of the first turn.
const FIRST_ANSWER: &str = "the first turn is journalled";

/// User text of the continued turn.
const SECOND_QUESTION: &str = "continue after the restart";

/// Assistant text of the continued turn.
const SECOND_ANSWER: &str = "continued from the journal";

/// [`LineReader`] that is never consulted: both turns carry their prompt in the arguments.
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
///
/// It carries no credentials and the same defaults the binary resolves with an empty environment,
/// so the run is offline and re-resolving it is what a restarted process does.
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

/// The `(agent, session)` pair the restart scenario persists and resumes.
fn cli_binding() -> Result<SessionBinding> {
    Ok(SessionBinding::new(AGENT_ID, CLI_SESSION_ID)?)
}

/// The journalled execution of the first turn.
fn cli_identity() -> Result<ExecutionIdentity> {
    Ok(ExecutionIdentity::new(
        NAMESPACE,
        AGENT_ID,
        CLI_SESSION_ID,
        FIRST_REQUEST,
        EXECUTION_ID,
        "cli",
    )?)
}

/// Runs the first turn through the CLI and journals it as a persistent execution.
///
/// Every handle — the agent, the model, the journal, the session store and the link store — is
/// dropped when this returns, which is the simulated process restart the scenario reopens from.
async fn run_first_turn(root: &std::path::Path, layout: &PersistentAgentLayout) -> Result<()> {
    let options = cli_options(root)?;
    let io = silent_io();
    let args = RunArgs {
        prompt: Some(FIRST_QUESTION.to_owned()),
        output_format: OutputFormat::Text,
    };
    let outcome = run_with_model_resumed(
        &options,
        &args,
        &io,
        Arc::new(ScriptedModel::text(FIRST_ANSWER)),
        ResumedTurn::new(CLI_SESSION_ID),
    )
    .await?;
    ensure!(
        outcome.answer == FIRST_ANSWER && outcome.session_id == CLI_SESSION_ID,
        "unexpected first-turn outcome: {outcome:?}"
    );

    // The journal a host keeps for that turn: the same two texts, in order, so the window it
    // rebuilds is the conversation the CLI record holds.
    let identity = cli_identity()?;
    let log = ExecutionLog::open_with_clock(
        layout.logs_root().join(identity.directory_name()),
        identity.clone(),
        LogCapacity::default(),
        clock(),
    )
    .await?;
    log.append(EntryPayload::execution_started(FIRST_QUESTION))
        .await?;
    log.append(EntryPayload::model_turn(
        "mc_1",
        FIRST_ANSWER,
        ModelUsage::known(12, 6),
    ))
    .await?;
    log.append(EntryPayload::execution_final(FinalStatus::Completed, None))
        .await?;

    let binding = cli_binding()?;
    let store = FileSessionStore::new(layout.sessions_root());
    let mut record = PersistentSessionRecord::new(&binding)?;
    record.messages = vec![BaseMessage::user(FIRST_QUESTION)];
    let record = record.with_ending(RecordEnding {
        outcome: "completed".to_owned(),
        request_id: FIRST_REQUEST.to_owned(),
        execution_id: EXECUTION_ID.to_owned(),
        ..RecordEnding::default()
    })?;
    store.save(&record).await?;

    let links = FileLinkStore::new(layout.links_root());
    links
        .write_link(&ExecutionLink::from_identity(
            &identity,
            None,
            LinkRelation::First,
        )?)
        .await?;
    Ok(())
}

/// What a restarted host rediscovers from disk alone.
struct Rediscovered {
    /// The catalogued execution of the first turn.
    execution: ExecutionSummary,
    /// The catalogued session record of the conversation.
    session: SessionSummary,
    /// The plan rebuilt from the journal, the record and the link chain.
    plan: ResumePlan,
}

/// Enumerates the executions and session records of `layout` and resolves the resume plan.
async fn rediscover(
    layout: &PersistentAgentLayout,
    binding: &SessionBinding,
) -> Result<Rediscovered> {
    let catalog =
        ExecutionCatalog::new(layout.logs_root(), LogCapacity::default()).with_clock(clock());
    let log_dir = layout.logs_root().join(cli_identity()?.directory_name());
    ensure!(
        catalog.path_for(EXECUTION_ID) == log_dir,
        "the catalogue must use the journal layout the host wrote: {}",
        catalog.path_for(EXECUTION_ID).display()
    );
    let executions = catalog.list().await?;
    ensure!(
        executions.len() == 1,
        "the restart must rediscover one execution, got {}",
        executions.len()
    );
    let execution = executions
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("the catalogued execution must be listed"))?;

    let store = FileSessionStore::new(layout.sessions_root());
    let sessions = store.list().await?;
    ensure!(
        sessions.len() == 1,
        "the restart must rediscover one session record, got {}",
        sessions.len()
    );
    let session = sessions
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("the persisted session record must be listed"))?;

    let plan = resume(
        &store,
        &RootLogSource::new(layout.logs_root(), LogCapacity::default()).with_clock(clock()),
        &FileLinkStore::new(layout.links_root()),
        binding,
        SECOND_REQUEST,
        &Continuation::new(EXECUTION_ID, Some("finish the job"), 4)?,
        &WindowAssembler::new(Capacity::default()),
    )
    .await?;
    Ok(Rediscovered {
        execution,
        session,
        plan,
    })
}

/// Returns the index of the first message whose text contains `needle`.
fn position_of(messages: &[BaseMessage], needle: &str) -> Result<usize> {
    messages
        .iter()
        .position(|message| message.text().contains(needle))
        .ok_or_else(|| anyhow!("no request message carries {needle:?}"))
}

/// Asserts the continued request carries the restored turn ahead of the new query.
fn assert_restored_turn_precedes_query(messages: &[BaseMessage]) -> Result<()> {
    let question = position_of(messages, FIRST_QUESTION)?;
    let answer = position_of(messages, FIRST_ANSWER)?;
    let query = position_of(messages, SECOND_QUESTION)?;
    ensure!(
        question < answer,
        "the restored turn keeps its oldest-first order"
    );
    ensure!(
        answer < query,
        "the restored transcript must precede the new query"
    );
    Ok(())
}

/// Asserts the restarted host rediscovers the execution and the record, and returns the plan
/// parsed back from its JSON form.
fn assert_rediscovered(
    discovered: &Rediscovered,
    layout: &PersistentAgentLayout,
    binding: &SessionBinding,
) -> Result<ResumePlan> {
    let execution = &discovered.execution;
    ensure!(
        execution.execution_id == EXECUTION_ID
            && execution.namespace_id == NAMESPACE
            && execution.agent_id == AGENT_ID
            && execution.session_id == CLI_SESSION_ID
            && execution.request_id == FIRST_REQUEST,
        "the summary must carry the execution identity: {execution:?}"
    );
    ensure!(
        execution.usable && execution.entries == 3 && execution.last_seq == Some(3),
        "the reopened journal must stay readable: {execution:?}"
    );
    ensure!(
        execution.created_at == "2023-11-14T22:13:20.000Z",
        "the journal is stamped by the frozen clock, got {}",
        execution.created_at
    );

    let session = &discovered.session;
    ensure!(
        session.binding == *binding,
        "the record must belong to the continued session: {session:?}"
    );
    ensure!(
        session.has_ending && !session.waiting,
        "the first turn ended instead of waiting: {session:?}"
    );
    ensure!(
        session.state_path == FileSessionStore::new(layout.sessions_root()).path_for(binding),
        "the summary must name the record file it was read from: {session:?}"
    );

    let plan = &discovered.plan;
    ensure!(
        plan.cursor.resume_seq == 4,
        "the plan continues the three-entry journal, got {}",
        plan.cursor.resume_seq
    );
    ensure!(
        plan.digest.entries == 3,
        "the plan carries the journal digest: {:?}",
        plan.digest
    );
    ensure!(
        plan.record.continuation.is_some(),
        "the resumed record keeps the continuation cursor"
    );
    ensure!(
        plan.background.contains(EXECUTION_ID),
        "the continuation prompt names the recovered execution"
    );

    // The plan survives the restart as JSON and still rebuilds turn one.
    let restored: ResumePlan = serde_json::from_str(&serde_json::to_string(plan)?)?;
    ensure!(restored == *plan, "the plan round-trips through JSON");
    let history = restored.seed_history();
    ensure!(
        history
            .iter()
            .any(|message| message.text().contains(FIRST_QUESTION)),
        "the rebuilt window carries turn one's question"
    );
    ensure!(
        history
            .iter()
            .any(|message| message.text().contains(FIRST_ANSWER)),
        "the rebuilt window carries turn one's answer"
    );
    Ok(restored)
}

/// A host restart rediscovers the journalled execution and continues the CLI session.
#[tokio::test]
async fn a_host_restart_resumes_the_journalled_cli_session() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let layout = PersistentAgentLayout::for_workspace(&Workspace::for_language(
        root.path().join("workspace"),
        "en",
    ));

    // --- process one: one CLI turn, journalled and persisted -------------------
    run_first_turn(root.path(), &layout).await?;

    // --- the restart: only the directory tree survives -------------------------
    let options = cli_options(root.path())?;
    let binding = cli_binding()?;
    let discovered = rediscover(&layout, &binding).await?;
    // The host's own persisted copy of the plan, parsed back from its JSON form.
    let plan = assert_rediscovered(&discovered, &layout, &binding)?;

    // --- process two: continue the session from the parsed plan ---------------
    let model = ScriptedModel::text(SECOND_ANSWER);
    let recorded = model.clone();
    let io = silent_io();
    let args = RunArgs {
        prompt: Some(SECOND_QUESTION.to_owned()),
        output_format: OutputFormat::Text,
    };
    let outcome = run_with_model_resumed(
        &options,
        &args,
        &io,
        Arc::new(model),
        ResumedTurn::new(CLI_SESSION_ID).with_history(plan.seed_history()),
    )
    .await?;
    ensure!(
        outcome.answer == SECOND_ANSWER && outcome.session_id == CLI_SESSION_ID,
        "unexpected continued-turn outcome: {outcome:?}"
    );

    // (a) the model request carries the restored turn ahead of the new query.
    let call = recorded
        .last_call()
        .ok_or_else(|| anyhow!("the continued turn must call the model"))?;
    assert_restored_turn_precedes_query(&call.messages)?;

    // (b) the CLI record holds both turns: continued, never truncated.
    let record = options
        .sessions()
        .load(CLI_SESSION_ID)?
        .ok_or_else(|| anyhow!("the CLI record must survive both turns"))?;
    let contents: Vec<&str> = record
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    ensure!(
        contents == [FIRST_QUESTION, FIRST_ANSWER, SECOND_QUESTION, SECOND_ANSWER],
        "the record must continue both turns, got {contents:?}"
    );
    Ok(())
}
