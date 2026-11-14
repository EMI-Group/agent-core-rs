//! End-to-end tests of the `notify`-backed workspace watcher.
//!
//! Every test here drives a real OS watcher on a `tempfile` directory, so the
//! suites are inherently timing-sensitive. Three rules keep them stable:
//!
//! - **every wait is bounded**: [`WAIT`] caps each await and each poll loop, so
//!   a lost notification fails the test instead of hanging it;
//! - **no exact timing assumptions**: the quiet period [`QUIET`] only separates
//!   two bursts, it never asserts that something did *not* happen within a fixed
//!   duration, and no test sleeps for a duration it then measures;
//! - **exact event counts only inside one debounce window**: the OS routinely
//!   splits one logical change into several notifications (`create` then
//!   `modify`), and only the debouncer's window promises a count. That is why
//!   [`bursty_writes_coalesce_per_debounce_window`] is the only test asserting a
//!   count — the single property the debouncer itself owns — while every other
//!   test waits for a *matching* event instead of counting events.
//!
//! The feature-off `Error::Unsupported` path of
//! [`WorkspaceWatcher::start`] is already covered by
//! `backend::tests::the_feature_off_body_answers_unsupported`, which is gated
//! the other way round (`#[cfg(not(feature = "notify"))]`), so it is not
//! duplicated here; this module only exists with the feature enabled.
#![cfg(feature = "notify")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::ErrorKind;
use tempfile::TempDir;
use tokio::time::Instant;

use crate::cli::{instruction::InstructionFiles, mcp_config::McpConfiguration, MapEnv};
use crate::skills::SkillLibrary;
use crate::watcher::{
    ChangeKind, InstructionReloadHandler, InstructionReloader, McpConfigReloadHandler,
    McpConfigReloader, SkillLibraryReloader, SkillsReloadHandler, Subscription, WatchEvent,
    WatchPlan, WatchTargetKind, WatchUpdate, WorkspaceWatcher,
};

/// Upper bound of every wait and every poll loop in this file.
///
/// A lost OS notification must fail the test within a bounded time rather than
/// block the suite, so this is the deadline of every await below.
const WAIT: Duration = Duration::from_secs(10);

/// Quiet period inserted between two bursts of writes.
///
/// It is deliberately longer than the 50 ms debounce window of the tests that
/// use it, so the next burst starts from an empty debouncer instead of merging
/// into the previous one. It is a *separation*, not an assertion about timing.
const QUIET: Duration = Duration::from_millis(150);

/// Poll interval of the condition loops.
const POLL: Duration = Duration::from_millis(25);

/// Quiet window used to prove that nothing more arrives for one path.
const SETTLE: Duration = Duration::from_millis(600);

/// Writes `contents` to `path`.
fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap();
}

/// Writes a skill directory `<root>/<name>/SKILL.md` with a front-matter
/// description, which is what [`SkillLibrary`] discovery keys on.
fn write_skill(root: &Path, name: &str, description: &str) {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory).unwrap();
    write(
        &directory.join("SKILL.md"),
        &format!("---\ndescription: {description}\n---\n# {name}\n"),
    );
}

/// Writes an `mcp.json` document declaring the `(name, command)` servers.
fn write_mcp(path: &Path, servers: &[(&str, &str)]) {
    let mut document = serde_json::Map::new();
    for (name, command) in servers {
        document.insert(
            (*name).to_owned(),
            serde_json::json!({ "command": command }),
        );
    }
    let text = serde_json::json!({ "mcpServers": document }).to_string();
    write(path, &text);
}

/// Awaits the next event matching `matches`, skipping every other update.
///
/// Non-event updates (`ReloadFailed`, `WatcherFailed`) are skipped rather than
/// treated as failures: this file asserts delivery, and the failure policies
/// have their own unit tests in `backend.rs`.
async fn next_event_for<F>(subscription: &mut Subscription, mut matches: F) -> WatchEvent
where
    F: FnMut(&WatchEvent) -> bool,
{
    let deadline = Instant::now() + WAIT;
    loop {
        match tokio::time::timeout_at(deadline, subscription.recv()).await {
            Ok(Some(WatchUpdate::Event(event))) if matches(&event) => return event,
            Ok(Some(_)) => {}
            Ok(None) => panic!("the watcher subscription closed before the expected event"),
            Err(elapsed) => {
                panic!("waited {elapsed:?} beyond {WAIT:?} for a matching watcher event")
            }
        }
    }
}

/// Collects every event for `path` that arrives within `window`.
async fn collect_events_for(
    subscription: &mut Subscription,
    path: &Path,
    window: Duration,
) -> Vec<WatchEvent> {
    collect_updates(subscription, window)
        .await
        .into_iter()
        .filter_map(|update| match update {
            WatchUpdate::Event(event) if event.path == path => Some(event),
            _ => None,
        })
        .collect()
}

/// Collects every update that arrives within `window`.
async fn collect_updates(subscription: &mut Subscription, window: Duration) -> Vec<WatchUpdate> {
    let mut updates = Vec::new();
    let deadline = Instant::now() + window;
    loop {
        match tokio::time::timeout_at(deadline, subscription.recv()).await {
            Ok(Some(update)) => updates.push(update),
            Ok(None) => return updates,
            Err(_deadline_passed) => return updates,
        }
    }
}

/// Polls `check` every [`POLL`] until it reports `true` or `deadline` passes.
async fn wait_until<F>(deadline: Instant, mut check: F) -> bool
where
    F: FnMut() -> bool,
{
    loop {
        if check() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Workspace/home layout the reload handlers watch.
///
/// Everything is created before a watcher starts, because a root that appears
/// later is not part of an already running plan.
struct ReloadFixture {
    workspace: PathBuf,
    home: OpenJiuwenHome,
    skills_root: PathBuf,
    identity: PathBuf,
    workspace_mcp: PathBuf,
    home_mcp: PathBuf,
}

/// Creates the [`ReloadFixture`] layout under `root`.
fn reload_fixture(root: &Path) -> ReloadFixture {
    let workspace = root.join("workspace");
    let home = OpenJiuwenHome::from_root(root.join("home"));
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(home.root()).unwrap();

    let skills_root = workspace.join("skills");
    write_skill(&skills_root, "demo", "demo skill");
    let identity = workspace.join("IDENTITY.md");
    write(&identity, "identity v1\n");
    let workspace_mcp = workspace.join(".openjiuwen").join("mcp.json");
    std::fs::create_dir_all(workspace_mcp.parent().unwrap()).unwrap();
    write_mcp(&workspace_mcp, &[("shared", "workspace-tool")]);
    let home_mcp = home.join(".openjiuwen").join("mcp.json");
    std::fs::create_dir_all(home_mcp.parent().unwrap()).unwrap();
    write_mcp(
        &home_mcp,
        &[("home_only", "home-tool"), ("shared", "home-tool")],
    );

    ReloadFixture {
        workspace,
        home,
        skills_root,
        identity,
        workspace_mcp,
        home_mcp,
    }
}

/// Asserts that `plan` covers the labels the reload handlers key on.
fn assert_reload_labels(plan: &WatchPlan) {
    for label in [
        "skills:workspace",
        "instruction:workspace:IDENTITY.md",
        "mcp:workspace",
    ] {
        assert!(
            plan.targets()
                .iter()
                .any(|target| target.label() == Some(label)),
            "the workspace plan lost the `{label}` target"
        );
    }
}

#[tokio::test]
async fn directory_watch_delivers_create_modify_remove() {
    let temp = TempDir::new().unwrap();
    let skills = temp.path().join("skills");
    std::fs::create_dir_all(&skills).unwrap();

    let handle = WorkspaceWatcher::new()
        .debounce(Duration::from_millis(50))
        .watch_dir(WatchTargetKind::Skills, &skills)
        .start()
        .await
        .unwrap();
    let mut subscription = handle.subscribe();

    let file = skills.join("a.md");
    write(&file, "first\n");

    let created = next_event_for(&mut subscription, |event| event.path == file).await;
    assert_eq!(created.kind, WatchTargetKind::Skills);
    assert_eq!(created.path, file);
    assert!(created.path.ends_with("a.md"));
    assert_eq!(created.change, ChangeKind::Created);
    assert!(created.sequence >= 1);
    let mut sequence = created.sequence;

    // Separation only: let the debouncer publish everything the creation burst
    // produced, so the append below is attributed to its own window.
    tokio::time::sleep(QUIET).await;

    write(&file, "first\nsecond\n");
    let modified = next_event_for(&mut subscription, |event| event.path == file).await;
    assert_eq!(modified.change, ChangeKind::Modified);
    assert!(modified.sequence > sequence);
    sequence = modified.sequence;

    tokio::time::sleep(QUIET).await;

    std::fs::remove_file(&file).unwrap();
    let removed = next_event_for(&mut subscription, |event| event.path == file).await;
    assert_eq!(removed.change, ChangeKind::Removed);
    assert!(removed.sequence > sequence);

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn bursty_writes_coalesce_per_debounce_window() {
    let temp = TempDir::new().unwrap();
    let skills = temp.path().join("skills");
    std::fs::create_dir_all(&skills).unwrap();

    let handle = WorkspaceWatcher::new()
        .debounce(Duration::from_millis(250))
        .watch_dir(WatchTargetKind::Skills, &skills)
        .start()
        .await
        .unwrap();
    let mut subscription = handle.subscribe();

    let file = skills.join("burst.md");
    for index in 0..5 {
        write(&file, &format!("write {index}\n"));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let first = next_event_for(&mut subscription, |event| event.path == file).await;
    // The file did not exist before the burst, but a backend that reports the
    // first notification as a modification must not fail this test; the property
    // under test is the count, not the kind of the coalesced event.
    assert!(matches!(
        first.change,
        ChangeKind::Created | ChangeKind::Modified
    ));
    assert!(first.sequence >= 1);

    // Everything else the same burst can still produce must share this window:
    // all five writes fall inside one 250 ms debounce window, so exactly the one
    // event above is published for them.
    let further = collect_events_for(&mut subscription, &file, SETTLE).await;
    assert!(
        further.is_empty(),
        "the burst produced further events: {further:?}"
    );

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn reload_handlers_observe_instruction_mcp_and_skills_changes() {
    let temp = TempDir::new().unwrap();
    let ReloadFixture {
        workspace,
        home,
        skills_root,
        identity,
        workspace_mcp,
        home_mcp,
    } = reload_fixture(temp.path());

    let skills = Arc::new(SkillLibraryReloader::new(SkillLibrary::with_roots([
        skills_root.clone(),
    ])));
    let instructions = Arc::new(InstructionReloader::load(&workspace, &home).unwrap());
    let mcp = Arc::new(McpConfigReloader::new(vec![home_mcp, workspace_mcp.clone()]).unwrap());

    let plan = WatchPlan::workspace(
        &workspace,
        &home,
        &MapEnv::from_pairs([] as [(&str, &str); 0]),
    )
    .existing_only();
    assert_reload_labels(&plan);

    let handle = WorkspaceWatcher::new()
        .debounce(Duration::from_millis(50))
        .watch_plan(plan)
        .on_reload(SkillsReloadHandler::new(Arc::clone(&skills)))
        .on_reload(InstructionReloadHandler::new(Arc::clone(&instructions)))
        .on_reload(McpConfigReloadHandler::new(Arc::clone(&mcp)))
        .start()
        .await
        .unwrap();

    // One deadline for every poll of this test: the whole reload chain has to
    // converge inside `WAIT`, not `WAIT` per condition.
    let deadline = Instant::now() + WAIT;

    // (a) instruction files: the handler re-runs `InstructionFiles::load`.
    let marker = "identity marker written after the watcher started";
    write(&identity, &format!("{marker}\n"));
    let observed = wait_until(deadline, || {
        instructions
            .current()
            .system_prompt()
            .is_some_and(|prompt| prompt.contains(marker))
    })
    .await;
    assert!(
        observed,
        "the instruction reload handler never observed the rewritten IDENTITY.md"
    );
    let files: InstructionFiles = instructions.current();
    assert!(files.identity.is_some());

    // (b) MCP configuration: the workspace file overrides the home-level value
    // of `shared` and adds a server of its own.
    write_mcp(
        &workspace_mcp,
        &[
            ("shared", "workspace-tool"),
            ("workspace_added", "added-tool"),
        ],
    );
    let observed = wait_until(deadline, || {
        mcp.current().is_some_and(|config| {
            config.mcp_servers.contains_key("workspace_added")
                && config.mcp_servers.contains_key("home_only")
                && config.mcp_servers.contains_key("shared")
        })
    })
    .await;
    assert!(
        observed,
        "the mcp reload handler never observed the rewritten mcp.json"
    );
    let merged: McpConfiguration = mcp.current().unwrap();
    assert_eq!(
        merged.mcp_servers["shared"].command.as_deref(),
        Some("workspace-tool"),
        "the workspace file must override the home-level entry"
    );

    // (c) skills: a new skill directory is discovered by the reload handler.
    write_skill(&skills_root, "added", "added skill");
    let mut discovered = false;
    while Instant::now() < deadline {
        if skills
            .current()
            .await
            .iter()
            .any(|skill| skill.name == "added")
        {
            discovered = true;
            break;
        }
        tokio::time::sleep(POLL).await;
    }
    assert!(
        discovered,
        "the skills reload handler never observed the added skill"
    );
    let reloaded = skills.reload().await.unwrap();
    assert!(
        reloaded.iter().any(|skill| skill.name == "added"),
        "a direct reload lost the added skill: {reloaded:?}"
    );

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn removed_directory_does_not_error_storm_and_watcher_survives() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("root");
    let nested = root.join("nested");
    std::fs::create_dir_all(&nested).unwrap();

    let handle = WorkspaceWatcher::new()
        .debounce(Duration::from_millis(50))
        .watch_dir(WatchTargetKind::Other, &root)
        .start()
        .await
        .unwrap();
    let mut subscription = handle.subscribe();

    let watched = nested.join("x.md");
    write(&watched, "nested\n");
    let created = next_event_for(&mut subscription, |event| event.path == watched).await;
    assert!(created.sequence >= 1, "sequence {}", created.sequence);

    std::fs::remove_dir_all(&nested).unwrap();
    let updates = collect_updates(&mut subscription, SETTLE).await;
    let failures = updates
        .iter()
        .filter(|update| matches!(update, WatchUpdate::WatcherFailed(_)))
        .count();
    assert!(
        failures <= 1,
        "the removed subtree produced {failures} watcher failures: {updates:?}"
    );

    // The watcher is still alive after the removal: a new file under the same
    // root is delivered.
    let after = root.join("after.md");
    write(&after, "after\n");
    let observed = next_event_for(&mut subscription, |event| event.path == after).await;
    assert_eq!(observed.change, ChangeKind::Created);

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_is_idempotent_and_closes_subscriptions() {
    let temp = TempDir::new().unwrap();
    let handle = WorkspaceWatcher::new()
        .watch_dir(WatchTargetKind::Other, temp.path())
        .start()
        .await
        .unwrap();
    let mut subscription = handle.subscribe();
    assert!(!handle.is_shutdown());

    handle.shutdown().await.unwrap();
    assert!(handle.is_shutdown());
    // A second shutdown finds no task left and still succeeds.
    handle.shutdown().await.unwrap();
    assert!(handle.is_shutdown());

    drop(handle);
    let closed = tokio::time::timeout(WAIT, subscription.recv()).await;
    assert!(
        closed.unwrap().is_none(),
        "the subscription must end once the handle is gone"
    );
}

#[tokio::test]
async fn empty_plan_is_a_typed_config_error() {
    let error = WorkspaceWatcher::new().start().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("watch target"),
        "unexpected message: {error}"
    );
}
