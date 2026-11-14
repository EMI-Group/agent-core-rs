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
//! - [`MoaConfig`] / [`MoaRail`] publish the mixture-of-agents prompt section;
//! - [`ExecutionLog`] / [`WindowAssembler`] append, reopen and rebuild a window;
//! - [`WorkspaceWatcher`] accepts a plan without the feature, starts with it, and reports the
//!   typed `Unsupported` when `notify` is off;
//! - [`HeartbeatRail`] fires on its injected clock and queues one event;
//! - [`PersonalContextStore`] derives its paths from a temporary home;
//! - [`WorktreeLock`] / [`SparseConfig`] acquire a repository lock and refuse a hostile path;
//! - [`ActivityThrottle`] clamps its tunables;
//! - [`drain_rail_events`] takes the goal channel out of a [`HarnessRailContext`];
//! - [`DiagnosticRegistry`] buffers and clears push diagnostics.
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
use openjiuwen_harness::rails::prompt_sections;
use openjiuwen_harness::{
    drain_rail_events, ActivityThrottle, Capacity, Diagnostic, DiagnosticRegistry,
    DiagnosticsConfig, DiagnosticsEvent, EntryPayload, ExecutionIdentity, ExecutionLog, GoalRail,
    GoalRailConfig, GoalStore, HarnessRail, HarnessRailContext, HeartbeatRail, HeartbeatRailConfig,
    HeartbeatReason, InteractionEvent, LogCapacity, ManualClock, MoaConfig, MoaRail, ModelUsage,
    PersonalContextStore, Range, SparseConfig, WatchTargetKind, WindowAssembler, WorkspaceWatcher,
    WorktreeLock, WorktreeLockConfig, GOAL_EVENT_EXTRA_KEY,
};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use tempfile::TempDir;

/// A frozen instant every fixture clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;

/// Wraps a scripted model as an injected [`Model`] handle.
fn model(text: &str) -> Arc<dyn Model> {
    Arc::new(ScriptedModel::text(text))
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
    ensure!(
        !section.render("en").trim().is_empty(),
        "the published MoA section must carry text"
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

/// With the feature a watcher starts on a real directory and shuts down cleanly.
#[cfg(feature = "notify")]
#[tokio::test]
async fn workspace_watcher_starts_with_notify() -> Result<()> {
    let temp = TempDir::new()?;
    let skills = temp.path().join("skills");
    std::fs::create_dir_all(&skills)?;
    let handle = WorkspaceWatcher::new()
        .debounce(std::time::Duration::from_millis(50))
        .watch_dir(WatchTargetKind::Skills, skills)
        .start()
        .await?;
    ensure!(
        handle.target_count() == 1 && !handle.is_shutdown(),
        "a started watcher observes one target"
    );
    handle.shutdown().await?;
    ensure!(handle.is_shutdown(), "shutdown must be observable");
    handle.shutdown().await?;
    ensure!(handle.is_shutdown(), "shutdown must stay idempotent");
    Ok(())
}
