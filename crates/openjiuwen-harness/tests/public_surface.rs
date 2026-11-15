//! Smoke coverage of the flat crate-root re-exports.
//!
//! Every module of `openjiuwen_harness` is also flattened into the crate root, mirroring the
//! Python package's `__init__`. A flat `pub use` is invisible to the module's own suite, so a
//! wrong path, a renamed type or a dropped export compiles here for the first time.
//!
//! Each test therefore *names* the item through `openjiuwen_harness::…` only (never
//! `openjiuwen_harness::moa::…`) and then makes one behavioural assertion on it, so the export is
//! proven to resolve *and* to be the working type rather than a placeholder:
//!
//! - [`MoaConfig`] / [`MoaRail`] publish the mixture-of-agents consultation protocol text;
//! - [`ExecutionLog`] / [`WindowAssembler`] append, reopen and rebuild a window;
//! - [`WorkspaceWatcher`] accepts a plan without the feature, observes a real write (and publishes
//!   nothing after shutdown) with it, and reports the typed `Unsupported` when `notify` is off;
//! - [`HeartbeatRail`] fires on its injected clock and queues one event;
//! - [`PersonalContextStore`] derives its paths from a temporary home;
//! - [`WorktreeLock`] / [`SparseConfig`] acquire a repository lock and refuse a hostile path;
//! - [`ActivityThrottle`] clamps its tunables;
//! - [`drain_rail_events`] takes the goal channel out of a [`HarnessRailContext`];
//! - [`DiagnosticRegistry`] buffers and clears push diagnostics;
//! - [`ExecutionCatalog`] / [`FileSessionStore`] enumerate a journalled execution and its session
//!   record through the [`PersistentAgentLayout`] constants, [`seed_state`] replaces the
//!   `deepagent` lane, and the resume entry points ([`ResumedTurn`], `run_with_model_resumed`,
//!   `Repl::seed_history`, `DeepAgent::seed_history`) are pinned to their exact signatures.
//!
//! Everything runs offline against `tempfile` directories, scripted models and injected clocks.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use openjiuwen_agent::RailEvent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{BaseMessage, Model};
// `prompt_sections` is the read side of the prompt-section channel and is only reachable through
// its module; every type below is spelled through the flat crate root.
use openjiuwen_harness::cli::{CliIo, CliOptions, RunArgs, RunOutcome};
use openjiuwen_harness::rails::prompt_sections;
use openjiuwen_harness::{
    drain_rail_events, percent_decode_identifier, percent_encode_identifier, run_once_resumed,
    run_with_model_resumed, seed_state, ActivityThrottle, Capacity, DeepAgent, Diagnostic,
    DiagnosticRegistry, DiagnosticsConfig, DiagnosticsEvent, EntryPayload, ExecutionCatalog,
    ExecutionIdentity, ExecutionLog, ExecutionSummary, FileSessionStore, GoalRail, GoalRailConfig,
    GoalStore, HarnessRail, HarnessRailContext, HeartbeatRail, HeartbeatRailConfig,
    HeartbeatReason, InteractionEvent, LogCapacity, ManualClock, MoaConfig, MoaRail, ModelUsage,
    PersistentAgentLayout, PersistentSessionRecord, PersonalContextStore, Range, Repl, ResumedTurn,
    RuntimeSubAgentSpawner, SessionBinding, SessionStateStore, SessionSummary, SparseConfig,
    SubAgentRuntime, WatchTargetKind, WindowAssembler, Workspace, WorkspaceWatcher, WorktreeLock,
    WorktreeLockConfig, GOAL_EVENT_EXTRA_KEY, LINKS_DIR, LOGS_DIR, PERSISTENT_AGENT_NODE,
    SESSIONS_DIR, SESSION_STATE_KEY,
};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use tempfile::TempDir;

/// A frozen instant every fixture clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;

/// Wraps a scripted model as an injected [`Model`] handle.
fn model(text: &str) -> Arc<dyn Model> {
    Arc::new(ScriptedModel::text(text))
}

/// The production subagent seam resolves from the crate root and keeps its exact signature.
///
/// The seam is `crate::rails::SubAgentSpawner`; this pins that `RuntimeSubAgentSpawner`
/// implements it (so it is usable as `Arc<dyn SubAgentSpawner>`) and that the flat
/// constructors and accessors keep the documented shapes. Behaviour is covered end to end by
/// `tests/subagent_spawner.rs`.
#[test]
fn runtime_subagent_spawner_resolves_from_the_crate_root() {
    fn assert_spawner<T: openjiuwen_harness::rails::SubAgentSpawner>() {}
    assert_spawner::<RuntimeSubAgentSpawner>();
    let new: fn(Arc<SubAgentRuntime>) -> RuntimeSubAgentSpawner = RuntimeSubAgentSpawner::new;
    let with_limit: fn(Arc<SubAgentRuntime>, usize) -> RuntimeSubAgentSpawner =
        RuntimeSubAgentSpawner::with_limit;
    let runtime: fn(&RuntimeSubAgentSpawner) -> &Arc<SubAgentRuntime> =
        RuntimeSubAgentSpawner::runtime;
    let limit: fn(&RuntimeSubAgentSpawner) -> usize = RuntimeSubAgentSpawner::limit;
    let active: fn(&RuntimeSubAgentSpawner) -> usize = RuntimeSubAgentSpawner::active_permits;
    let tracked: fn(&RuntimeSubAgentSpawner) -> usize = RuntimeSubAgentSpawner::tracked;
    // Keep the pins alive so the signatures are part of the compiled test, not dead bindings.
    std::hint::black_box((new, with_limit, runtime, limit, active, tracked));
}

/// The `MoA` configuration and rail resolve from the crate root and publish their section.
#[tokio::test]
async fn moa_config_and_rail_come_from_the_crate_root() -> Result<()> {
    let config = MoaConfig::new(vec![model("first proposal"), model("second proposal")]);
    ensure!(
        config.reference_models.len() == 2,
        "both reference models must survive construction"
    );
    let rail = MoaRail::new(config)?;
    let section = rail.prompt_section("en");
    ensure!(
        section.name == "moa",
        "unexpected section name: {section:?}"
    );
    // Assert on the published *text*, not merely on a non-empty string: the default mode is
    // `AggregatorOnDemand`, whose English protocol must name the consultation tool.
    let rendered = section.render("en");
    ensure!(
        rendered.contains("Reference Consultation")
            && rendered.contains("consult_reference_models"),
        "the published MoA section must carry the consultation protocol, got: {rendered}"
    );
    ensure!(
        section.render("cn").contains("参考模型咨询"),
        "the section must stay bilingual, got: {}",
        section.render("cn")
    );

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    ctx.base.messages = vec![BaseMessage::user("fix the failing build")];
    rail.before_model_call(&mut ctx).await?;
    let sections = prompt_sections(&ctx);
    ensure!(
        sections.len() == 1 && sections[0].name == "moa",
        "the rail must publish exactly its own section: {sections:?}"
    );
    Ok(())
}

/// The execution journal and the window assembler resolve from the crate root and round-trip.
#[tokio::test]
async fn execution_log_and_window_assembler_come_from_the_crate_root() -> Result<()> {
    let temp = TempDir::new()?;
    let identity = ExecutionIdentity::new("ns", "agent", "session", "req", "exec", "src")?;
    let dir = temp.path().join("logs");
    let log = ExecutionLog::open_with_clock(
        &dir,
        identity.clone(),
        LogCapacity::default(),
        Arc::new(ManualClock::new(NOW_MS)),
    )
    .await?;
    log.append(EntryPayload::execution_started("ship the surface"))
        .await?;
    log.append(EntryPayload::model_turn(
        "mc_1",
        "first answer",
        ModelUsage::known(10, 5),
    ))
    .await?;
    let snapshot = log.read().await?;
    ensure!(
        snapshot.is_usable() && snapshot.entries.len() == 2,
        "two appended entries must read back: {snapshot:?}"
    );

    let assembly = WindowAssembler::new(Capacity::default()).assemble(&snapshot.entries);
    ensure!(
        !assembly.messages.is_empty(),
        "the assembler must rebuild a non-empty window"
    );

    // Reopening the same directory is a fresh process reading disk.
    drop(log);
    let reopened = ExecutionLog::open_with_clock(
        &dir,
        identity,
        LogCapacity::default(),
        Arc::new(ManualClock::new(NOW_MS + 1_000)),
    )
    .await?;
    let digest = reopened.digest().await?;
    ensure!(
        digest.entries == 2 && digest.last_seq == Some(2),
        "the reopened journal must keep both entries: {digest:?}"
    );
    Ok(())
}

/// The heartbeat rail fires from its injected clock and reports one scheduled event.
#[tokio::test]
async fn heartbeat_rail_come_from_the_crate_root() -> Result<()> {
    let now = Arc::new(AtomicU64::new(1_000));
    let handle = Arc::clone(&now);
    let rail = HeartbeatRail::new(HeartbeatRailConfig::default())
        .with_clock(Arc::new(move || handle.load(Ordering::SeqCst)));

    ensure!(
        rail.observe("agent", "session", None).is_none(),
        "the first observation only records a baseline"
    );
    ensure!(
        !rail.is_due("agent", "session"),
        "a fresh baseline is not due yet"
    );

    now.store(1_000 + 60_000, Ordering::SeqCst);
    ensure!(rail.is_due("agent", "session"), "one interval must be due");
    let event = rail
        .observe("agent", "session", None)
        .ok_or_else(|| anyhow!("a due observation must fire"))?;
    ensure!(
        event.reason == HeartbeatReason::Scheduled,
        "the periodic path must report `Scheduled`, got {:?}",
        event.reason
    );
    ensure!(
        rail.drain_events().len() == 1 && rail.events().is_empty(),
        "exactly one event must be queued and draining clears it"
    );
    Ok(())
}

/// The personal-context store derives every path from the home it was built with.
#[test]
fn personal_context_store_comes_from_the_crate_root() -> Result<()> {
    let temp = TempDir::new()?;
    let store = PersonalContextStore::new(temp.path());
    ensure!(
        store.home() == temp.path(),
        "the store must keep its home: {}",
        store.home().display()
    );
    ensure!(
        store.context_root() == temp.path().join("workspace").join("context"),
        "unexpected context root: {}",
        store.context_root().display()
    );
    ensure!(
        store.entries()?.is_empty(),
        "a missing index must load as an empty ledger"
    );
    let error = store
        .entry("not-a-source-id")
        .err()
        .ok_or_else(|| anyhow!("an invalid source id must be refused"))?;
    ensure!(
        error.kind() == ErrorKind::Validation,
        "unexpected error kind: {error:?}"
    );
    Ok(())
}

/// The worktree lock excludes a second handle and the sparse config refuses traversal.
#[tokio::test]
async fn worktree_lock_and_sparse_config_come_from_the_crate_root() -> Result<()> {
    let temp = TempDir::new()?;
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(repo.join(".git"))?;
    let lock = WorktreeLock::new(&repo, temp.path(), WorktreeLockConfig::default());
    ensure!(
        lock.lock_path().starts_with(repo.join(".git")),
        "the sidecar must live in .git: {}",
        lock.lock_path().display()
    );

    let guard = lock
        .try_acquire()
        .await
        .map_err(|error| anyhow!("first acquisition failed: {error}"))?;
    ensure!(
        guard.lock_path() == lock.lock_path(),
        "the guard must name the lock file it holds"
    );
    if lock.is_enabled() {
        ensure!(
            lock.lock_path().exists(),
            "an active acquisition must create the lock file"
        );
        ensure!(
            lock.try_acquire().await.is_err(),
            "a second handle must not take a held repository lock"
        );
    }
    drop(guard);
    let reacquired = lock
        .try_acquire()
        .await
        .map_err(|error| anyhow!("the released lock must be takeable again: {error}"))?;
    drop(reacquired);

    let sparse = SparseConfig::cone(["crates/openjiuwen-harness/"])?;
    ensure!(
        sparse.normalized().paths == vec!["crates/openjiuwen-harness".to_string()],
        "normalisation must strip the trailing slash: {:?}",
        sparse.paths
    );
    let hostile = SparseConfig::cone(["../escape"])
        .err()
        .ok_or_else(|| anyhow!("a traversal path must be refused"))?;
    ensure!(
        hostile.kind() == ErrorKind::Security,
        "unexpected error kind: {hostile:?}"
    );
    Ok(())
}

/// The activity throttle clamps every tunable into its documented range.
#[test]
fn activity_throttle_comes_from_the_crate_root() -> Result<()> {
    let clamped = ActivityThrottle::new(0, 0, -5);
    ensure!(
        clamped.capacity() == 1 && clamped.text_max_len() == 1 && clamped.throttle_ms() == 0,
        "zero and negative tunables must clamp: {clamped:?}"
    );
    let configured = ActivityThrottle::new(64, 128, 750);
    ensure!(
        configured.with_capacity(9_000).capacity() == 4_096,
        "the ring capacity must cap at 4096"
    );
    ensure!(
        configured.with_text_max_len(200_000).text_max_len() == 65_536,
        "the summary cap must cap at 65536"
    );
    Ok(())
}

/// The goal channel is drained out of a rail context exactly once.
#[test]
fn drain_rail_events_comes_from_the_crate_root() -> Result<()> {
    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    let event = InteractionEvent::goal_updated(Some(json!({"id": "g-1"})));
    ctx.extra_mut().insert(
        GOAL_EVENT_EXTRA_KEY.to_string(),
        json!([event.to_output_value()]),
    );

    let drained = drain_rail_events(&mut ctx);
    ensure!(
        drained.len() == 1,
        "the queued goal event must be drained: {drained:?}"
    );
    ensure!(
        drained[0].payload.get("goal") == Some(&json!({"id": "g-1"})),
        "the payload must survive the round trip: {:?}",
        drained[0].payload
    );
    ensure!(
        drain_rail_events(&mut ctx).is_empty(),
        "a drained context has nothing left"
    );
    Ok(())
}

/// The goal rail and its store resolve from the crate root and publish an active goal.
#[tokio::test]
async fn goal_rail_comes_from_the_crate_root() -> Result<()> {
    let temp = TempDir::new()?;
    let store = GoalStore::new(temp.path().join("goal.json"));
    let rail = GoalRail::new(
        Arc::clone(&store),
        GoalRailConfig::default().with_prompt_language("en"),
    );
    store
        .start("ship the flat surface", chrono::Utc::now())
        .await?;

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    rail.before_task_iteration(&mut ctx).await?;
    let sections = prompt_sections(&ctx);
    ensure!(
        sections.len() == 1 && sections[0].name == "goal",
        "the rail must publish the goal section: {sections:?}"
    );
    ensure!(
        sections[0].render("en").contains("ship the flat surface"),
        "the reminder must carry the objective: {}",
        sections[0].render("en")
    );
    Ok(())
}

/// The diagnostic registry buffers a publish, caps it and clears on an empty publish.
#[test]
fn diagnostic_registry_comes_from_the_crate_root() -> Result<()> {
    let registry = DiagnosticRegistry::new(DiagnosticsConfig::default());
    let uri = "file:///workspace/demo.rs".to_string();
    registry.publish(DiagnosticsEvent {
        uri: uri.clone(),
        version: Some(1),
        diagnostics: vec![
            Diagnostic {
                message: "unused import".to_string(),
                severity: Some(2),
                range: Range::default(),
                source: Some("demo".to_string()),
                code: None,
                tags: Vec::new(),
                related_information: Vec::new(),
            },
            Diagnostic {
                message: "unused import".to_string(),
                severity: Some(2),
                range: Range::default(),
                source: Some("demo".to_string()),
                code: None,
                tags: Vec::new(),
                related_information: Vec::new(),
            },
        ],
    });
    ensure!(
        registry.pending_count() == 1,
        "one document must be buffered, got {}",
        registry.pending_count()
    );
    let buffered = registry
        .snapshot(&uri)
        .ok_or_else(|| anyhow!("the published document must be buffered"))?;
    ensure!(
        buffered.len() == 1,
        "identical diagnostics must deduplicate: {buffered:?}"
    );

    registry.publish(DiagnosticsEvent {
        uri: uri.clone(),
        version: Some(2),
        diagnostics: Vec::new(),
    });
    ensure!(
        registry.snapshot(&uri).is_none() && registry.pending_count() == 0,
        "an empty publish must clear the document"
    );
    Ok(())
}

/// Without the feature a watcher still builds its plan, but `start` is typed `Unsupported`.
#[cfg(not(feature = "notify"))]
#[tokio::test]
async fn workspace_watcher_reports_unsupported_without_notify() -> Result<()> {
    let temp = TempDir::new()?;
    let watcher = WorkspaceWatcher::new()
        .debounce(std::time::Duration::from_millis(50))
        .watch_dir(WatchTargetKind::Skills, temp.path());
    ensure!(
        watcher.targets().len() == 1,
        "the plan must exist without the feature: {:?}",
        watcher.targets()
    );

    let error = watcher
        .start()
        .await
        .err()
        .ok_or_else(|| anyhow!("start must fail without the notify feature"))?;
    ensure!(
        error.kind() == ErrorKind::Unsupported,
        "unexpected error kind: {error:?}"
    );
    ensure!(
        error.to_string().contains("notify"),
        "the reason must name the feature: {error}"
    );
    Ok(())
}

/// Delivery budget of one awaited watcher event in this smoke suite.
#[cfg(feature = "notify")]
const WATCH_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// Quiet window proving a shut-down watcher publishes nothing more.
#[cfg(feature = "notify")]
const WATCH_QUIET: std::time::Duration = std::time::Duration::from_millis(400);

/// Awaits the next event whose path is `path`, within `budget`.
///
/// This is the smoke-level observation helper: it proves the re-exported watcher really watches,
/// not merely that it constructs.
#[cfg(feature = "notify")]
async fn await_event_for(
    subscription: &mut openjiuwen_harness::Subscription,
    path: &std::path::Path,
    budget: std::time::Duration,
) -> Option<openjiuwen_harness::WatchEvent> {
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        match tokio::time::timeout_at(deadline, subscription.recv()).await {
            Ok(Some(openjiuwen_harness::WatchUpdate::Event(event))) if event.path == path => {
                return Some(event)
            }
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return None,
        }
    }
}

/// With the feature a watcher starts on a real directory, observes a write and shuts down
/// cleanly without publishing anything afterwards.
#[cfg(feature = "notify")]
#[tokio::test]
async fn workspace_watcher_starts_with_notify() -> Result<()> {
    let temp = TempDir::new()?;
    let skills = temp.path().join("skills");
    std::fs::create_dir_all(&skills)?;
    let handle = WorkspaceWatcher::new()
        .debounce(std::time::Duration::from_millis(50))
        .watch_dir(WatchTargetKind::Skills, skills.clone())
        .start()
        .await?;
    ensure!(
        handle.target_count() == 1,
        "a started watcher observes one target, got {}",
        handle.target_count()
    );
    ensure!(!handle.is_shutdown(), "a started watcher is running");
    ensure!(
        handle.dropped_notifications() == 0 && handle.dropped_dispatches() == 0,
        "nothing may be dropped before the first event"
    );

    // Registration completed inside `start`, so this write must be delivered.
    let mut subscription = handle.subscribe();
    let notes = skills.join("notes.md");
    std::fs::write(&notes, "notes\n")?;
    let event = await_event_for(&mut subscription, &notes, WATCH_BUDGET)
        .await
        .ok_or_else(|| {
            anyhow!(
                "the started watcher must publish the write to {}",
                notes.display()
            )
        })?;
    ensure!(
        event.kind == WatchTargetKind::Skills && event.sequence >= 1,
        "unexpected watcher event: {event:?}"
    );
    ensure!(
        handle.dropped_notifications() == 0 && handle.dropped_dispatches() == 0,
        "one notification must not fill either bounded queue"
    );

    handle.shutdown().await?;
    ensure!(handle.is_shutdown(), "shutdown must be observable");
    handle.shutdown().await?;
    ensure!(handle.is_shutdown(), "shutdown must stay idempotent");

    // `shutdown` joins both tasks and releases the OS watch, so a later write is never published.
    std::fs::write(&notes, "after shutdown\n")?;
    ensure!(
        await_event_for(&mut subscription, &notes, WATCH_QUIET)
            .await
            .is_none(),
        "a shut-down watcher must not publish another update"
    );
    Ok(())
}

// --- host resume ----------------------------------------------------------

/// Compile-time pin of `run_once_resumed`'s signature.
///
/// An `async fn` item cannot coerce to a `fn` pointer (its return type is an unnameable future),
/// so every async pin here is a one-line wrapper with the exact argument and return types: a
/// changed argument type, arity or error type fails to compile. The wrappers are referenced but
/// never called — this one would build a provider model and needs credentials.
async fn pinned_run_once_resumed(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    resumed: ResumedTurn,
) -> openjiuwen_common::error::Result<RunOutcome> {
    run_once_resumed(options, args, io, resumed).await
}

/// Compile-time pin of `run_with_model_resumed`'s signature (never called here).
async fn pinned_run_with_model_resumed(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    model: Arc<dyn Model>,
    resumed: ResumedTurn,
) -> openjiuwen_common::error::Result<RunOutcome> {
    run_with_model_resumed(options, args, io, model, resumed).await
}

/// Compile-time pin of `Repl::seed_history`'s signature (never called here).
async fn pinned_repl_seed_history(
    repl: &mut Repl,
    history: Vec<BaseMessage>,
) -> openjiuwen_common::error::Result<()> {
    repl.seed_history(history).await
}

/// Compile-time pin of `DeepAgent::seed_history`'s signature (never called here).
async fn pinned_deep_agent_seed_history(
    agent: &DeepAgent,
    session: &Arc<dyn Session>,
    history: Vec<BaseMessage>,
) -> openjiuwen_common::error::Result<()> {
    agent.seed_history(session, history).await
}

/// The host-resume surface resolves from the crate root and does real work.
#[tokio::test]
async fn resume_surface_comes_from_the_crate_root() -> Result<()> {
    let temp = TempDir::new()?;
    let workspace = Workspace::for_language(temp.path().join("workspace"), "en");
    let layout = PersistentAgentLayout::for_workspace(&workspace);
    ensure!(
        layout.base() == workspace.root_path.join(PERSISTENT_AGENT_NODE).as_path(),
        "the layout must fall back to the workspace node, got {}",
        layout.base().display()
    );
    ensure!(
        layout.logs_root() == layout.base().join(LOGS_DIR)
            && layout.sessions_root() == layout.base().join(SESSIONS_DIR)
            && layout.links_root() == layout.base().join(LINKS_DIR),
        "the layout roots must be the exported directory constants"
    );

    // `percent_decode_identifier` inverts the encoding the layout and the stores share.
    let encoded = percent_encode_identifier("agent/1");
    ensure!(
        encoded == "agent%2F1" && percent_decode_identifier(&encoded).as_deref() == Some("agent/1"),
        "unexpected identifier encoding: {encoded}"
    );

    // One journalled execution, discovered by the catalogue.
    let identity = ExecutionIdentity::new("ns", "agent", "session", "req", "exec", "src")?;
    let log = ExecutionLog::open_with_clock(
        layout.logs_root().join(identity.directory_name()),
        identity,
        LogCapacity::default(),
        Arc::new(ManualClock::new(NOW_MS)),
    )
    .await?;
    log.append(EntryPayload::execution_started("resume me"))
        .await?;
    let catalog = ExecutionCatalog::new(layout.logs_root(), LogCapacity::default())
        .with_clock(Arc::new(ManualClock::new(NOW_MS)));
    let summary: ExecutionSummary = catalog
        .list()
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("the journalled execution must be catalogued"))?;
    ensure!(
        summary.execution_id == "exec" && summary.entries == 1 && summary.usable,
        "unexpected execution summary: {summary:?}"
    );

    // One persisted session record, discovered by the store.
    let binding = SessionBinding::new("agent", "session")?;
    let store = FileSessionStore::new(layout.sessions_root());
    let mut record = PersistentSessionRecord::new(&binding)?;
    record.messages = vec![BaseMessage::user("resume me")];
    store.save(&record).await?;
    let session: SessionSummary = store
        .list()
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("the persisted session record must be catalogued"))?;
    ensure!(
        session.binding == binding && session.state_path == store.path_for(&binding),
        "unexpected session summary: {session:?}"
    );

    // The resumed-turn value keeps what its builders set.
    let turn = ResumedTurn::new("session")
        .with_history(vec![BaseMessage::user("earlier question")])
        .with_state(json!({"iteration": 1}));
    ensure!(
        turn.session_id == "session"
            && turn.history.len() == 1
            && turn.state == Some(json!({"iteration": 1})),
        "the builders must keep their fields: {turn:?}"
    );

    // `seed_state` really replaces the persisted `deepagent` lane of a session.
    let seeded: Arc<dyn Session> = Arc::new(AgentSession::new("session"));
    seed_state(seeded.as_ref(), json!({"iteration": 3})).await?;
    ensure!(
        seeded.get_state(SESSION_STATE_KEY).await? == Some(json!({"iteration": 3})),
        "the seeded payload must be the session state"
    );

    // Signature pins of the async surface: referencing the wrappers keeps them (and therefore
    // their signatures) compiled; nothing is executed.
    let _ = (
        pinned_run_once_resumed,
        pinned_run_with_model_resumed,
        pinned_repl_seed_history,
        pinned_deep_agent_seed_history,
    );
    Ok(())
}

/// The completion seam, the notification outcomes and their named methods resolve flat.
///
/// All of them were added after the flat surface was first pinned, so this test spells them
/// through `openjiuwen_harness::…` only: a dropped `pub use` fails to compile here while the
/// canonical module paths stay green.
#[test]
fn the_completion_seam_and_the_mcp_notification_surface_resolve_flat() {
    use openjiuwen_harness::{
        CompletionFn, CompletionSummarizer, NotificationOutcome, LIVE_CLIENT_SUBSCRIBE_UNSUPPORTED,
        LIVE_CLIENT_UNSUBSCRIBE_UNSUPPORTED, RESOURCE_LIST_CHANGED_METHOD,
        RESOURCE_LIST_CHANGED_METHOD_SHORT, RESOURCE_UPDATED_METHOD, RESOURCE_UPDATED_METHOD_SHORT,
    };
    // The summariser takes the documented synchronous seam and bounds its answer.
    let summariser = CompletionSummarizer::new(|prompt: &str| {
        Ok::<String, openjiuwen_common::error::Error>(format!("{} chars", prompt.len()))
    });
    assert_eq!(summariser.max_chars(), 4096);
    assert_ne!(summariser.prompt(&[]), "");
    // A host may also hold the shared seam type itself.
    let shared: Arc<CompletionFn> = Arc::new(|_prompt: &str| Ok("shared".to_string()));
    assert_eq!(shared("x").expect("the seam answers"), "shared");
    // The notification outcomes keep their accessors and the method constants agree.
    let ignored = NotificationOutcome::IgnoredMethod {
        method: RESOURCE_LIST_CHANGED_METHOD.to_string(),
    };
    assert!(!ignored.is_published());
    assert_eq!(ignored.uri(), None);
    let published = NotificationOutcome::Published {
        uri: "file:///notes.md".to_string(),
        sequence: 1,
        reason: RESOURCE_UPDATED_METHOD.to_string(),
    };
    assert!(published.is_published());
    assert_eq!(published.uri(), Some("file:///notes.md"));
    assert!(RESOURCE_UPDATED_METHOD.ends_with(RESOURCE_UPDATED_METHOD_SHORT));
    assert!(RESOURCE_LIST_CHANGED_METHOD.ends_with(RESOURCE_LIST_CHANGED_METHOD_SHORT));
    // The two live-client refusals name the concrete gap and stay distinct.
    assert!(LIVE_CLIENT_SUBSCRIBE_UNSUPPORTED.contains("resources/subscribe"));
    assert_ne!(
        LIVE_CLIENT_SUBSCRIBE_UNSUPPORTED,
        LIVE_CLIENT_UNSUBSCRIBE_UNSUPPORTED
    );
}
