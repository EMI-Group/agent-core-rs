//! End-to-end coverage for the `notify`-backed workspace watcher, driven through the crate root.
//!
//! One debounced watcher is started over a `tempfile` workspace with a skill root, an
//! `IDENTITY.md` and an MCP configuration, and each target is then mutated so the matching
//! [`WatchUpdate`] has to arrive. Every await is bounded by [`WAIT`], and the only event count
//! asserted is the one the debouncer itself owns: a burst of writes to a single path inside one
//! window publishes exactly one event.
//!
//! The whole file is gated on the feature, so `--no-default-features` compiles it to an empty
//! suite; the feature-off `Unsupported` path is covered by `tests/public_surface.rs`.
#![cfg(feature = "notify")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{ensure, Result};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_harness::cli::MapEnv;
use openjiuwen_harness::{
    ChangeKind, Subscription, WatchEvent, WatchTargetKind, WatchUpdate, WatcherHandle,
    WorkspaceWatcher,
};
use tempfile::TempDir;
use tokio::time::Instant;

/// Upper bound of every wait in this file: a lost notification fails the test instead of hanging.
const WAIT: Duration = Duration::from_secs(10);

/// Debounce window of the watchers under test.
const DEBOUNCE: Duration = Duration::from_millis(100);

/// The workspace/home layout the watcher plan discovers.
struct Fixture {
    workspace: PathBuf,
    home: OpenJiuwenHome,
    skills: PathBuf,
    identity: PathBuf,
    mcp: PathBuf,
}

impl Fixture {
    /// Creates `skills/`, `IDENTITY.md` and `.openjiuwen/mcp.json` before any watcher starts.
    fn new(root: &Path) -> Result<Self> {
        let workspace = root.join("workspace");
        let home = OpenJiuwenHome::from_root(root.join("home"));
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(home.root())?;
        let skills = workspace.join("skills");
        std::fs::create_dir_all(&skills)?;
        let identity = workspace.join("IDENTITY.md");
        write(&identity, "identity v1\n")?;
        let mcp = workspace.join(".openjiuwen").join("mcp.json");
        std::fs::create_dir_all(mcp.parent().expect("the config has a parent"))?;
        write(&mcp, r#"{"mcpServers":{}}"#)?;
        Ok(Self {
            workspace,
            home,
            skills,
            identity,
            mcp,
        })
    }

    /// Starts a debounced watcher over the whole discovered workspace plan.
    async fn watcher(&self) -> Result<WatcherHandle> {
        Ok(
            WorkspaceWatcher::for_workspace(&self.workspace, &self.home, &MapEnv::new())
                .debounce(DEBOUNCE)
                .start()
                .await?,
        )
    }
}

/// Writes `contents` to `path`.
fn write(path: &Path, contents: &str) -> Result<()> {
    Ok(std::fs::write(path, contents)?)
}

/// Writes the skill directory `<root>/<name>/SKILL.md` with a front-matter description.
fn write_skill(root: &Path, name: &str, description: &str) -> Result<()> {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory)?;
    write(
        &directory.join("SKILL.md"),
        &format!("---\ndescription: {description}\n---\n# {name}\n"),
    )
}

/// Awaits the next `Event` update matching `matches`, skipping failures and other targets.
async fn next_event_for<F>(subscription: &mut Subscription, mut matches: F) -> Result<WatchEvent>
where
    F: FnMut(&WatchEvent) -> bool,
{
    let deadline = Instant::now() + WAIT;
    loop {
        match tokio::time::timeout_at(deadline, subscription.recv()).await {
            Ok(Some(WatchUpdate::Event(event))) if matches(&event) => return Ok(event),
            Ok(Some(_)) => {}
            Ok(None) => anyhow::bail!("the subscription closed before the expected event"),
            Err(elapsed) => {
                anyhow::bail!("waited {elapsed:?} beyond {WAIT:?} for a matching watcher event")
            }
        }
    }
}

/// Collects every event for `path` that arrives within `window`.
async fn events_for(
    subscription: &mut Subscription,
    path: &Path,
    window: Duration,
) -> Vec<WatchEvent> {
    let mut events = Vec::new();
    let deadline = Instant::now() + window;
    loop {
        match tokio::time::timeout_at(deadline, subscription.recv()).await {
            Ok(Some(WatchUpdate::Event(event))) if event.path == path => events.push(event),
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return events,
        }
    }
}

/// Every discovered workspace target delivers its own typed event when it changes.
#[tokio::test]
async fn each_workspace_target_delivers_its_change() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let watcher = fixture.watcher().await?;
    ensure!(
        watcher.target_count() >= 3,
        "the plan covers skills, the instruction file and the MCP config, got {}",
        watcher.target_count()
    );
    let mut subscription = watcher.subscribe();

    // A new skill directory arrives under the recursive `skills` target.
    write_skill(&fixture.skills, "demo", "demo skill")?;
    let skill_event = next_event_for(&mut subscription, |event| {
        event.kind == WatchTargetKind::Skills && event.path.ends_with("SKILL.md")
    })
    .await?;
    ensure!(
        skill_event
            .path
            .ends_with(Path::new("demo").join("SKILL.md")),
        "the event names the new skill file, got {:?}",
        skill_event.path
    );
    ensure!(
        skill_event.label.as_deref() == Some("skills:workspace"),
        "the workspace skill target keeps its label, got {:?}",
        skill_event.label
    );
    ensure!(skill_event.sequence >= 1, "sequences start at one");

    // The instruction file is a single-file target, so a rewrite is `Modified`.
    write(&fixture.identity, "identity v2\n")?;
    let identity_event =
        next_event_for(&mut subscription, |event| event.path == fixture.identity).await?;
    ensure!(
        identity_event.kind == WatchTargetKind::InstructionFile,
        "got {:?}",
        identity_event.kind
    );
    ensure!(
        identity_event.change == ChangeKind::Modified,
        "an existing file rewrite is Modified, got {:?}",
        identity_event.change
    );
    ensure!(
        identity_event.label.as_deref() == Some("instruction:workspace:IDENTITY.md"),
        "got {:?}",
        identity_event.label
    );

    // The MCP configuration is watched as a single file too.
    write(
        &fixture.mcp,
        r#"{"mcpServers":{"demo":{"command":"demo"}}}"#,
    )?;
    let mcp_event = next_event_for(&mut subscription, |event| event.path == fixture.mcp).await?;
    ensure!(
        mcp_event.kind == WatchTargetKind::McpConfig,
        "got {:?}",
        mcp_event.kind
    );
    ensure!(
        mcp_event.label.as_deref() == Some("mcp:workspace"),
        "got {:?}",
        mcp_event.label
    );
    ensure!(
        identity_event.sequence < mcp_event.sequence,
        "sequences are monotonic across targets"
    );

    watcher.shutdown().await?;
    ensure!(watcher.is_shutdown(), "the handle reports the shutdown");
    Ok(())
}

/// Writes to one path inside a single debounce window coalesce into exactly one event.
#[tokio::test]
async fn a_burst_of_writes_coalesces_into_one_event() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let watcher = WorkspaceWatcher::new()
        .debounce(DEBOUNCE)
        .watch_dir(WatchTargetKind::Skills, &fixture.skills)
        .start()
        .await?;
    let mut subscription = watcher.subscribe();
    let path = fixture.skills.join("notes.md");
    for revision in 0..5 {
        write(&path, &format!("revision {revision}\n"))?;
    }
    let events = events_for(&mut subscription, &path, WAIT).await;
    ensure!(
        events.len() == 1,
        "one debounce window publishes one event, got {events:?}"
    );
    ensure!(
        events[0].change == ChangeKind::Created,
        "the first observation is the creation, got {:?}",
        events[0].change
    );
    ensure!(
        events[0].kind == WatchTargetKind::Skills,
        "got {:?}",
        events[0].kind
    );
    watcher.shutdown().await?;
    Ok(())
}

/// Shutdown is clean and idempotent, and a single-file target still watches a missing file.
#[tokio::test]
async fn shutdown_is_idempotent_and_file_targets_survive_creation() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let missing = fixture.workspace.join("OPENJIUWEN.md");
    ensure!(!missing.exists(), "the fixture starts without the file");
    let watcher = WorkspaceWatcher::new()
        .debounce(DEBOUNCE)
        .watch_file(WatchTargetKind::InstructionFile, &missing)
        .start()
        .await?;
    let mut subscription = watcher.subscribe();
    write(&missing, "workspace instructions\n")?;
    let event = next_event_for(&mut subscription, |event| event.path == missing).await?;
    ensure!(
        event.change == ChangeKind::Created,
        "a file created after start is Created, got {:?}",
        event.change
    );
    ensure!(
        event.kind == WatchTargetKind::InstructionFile,
        "got {:?}",
        event.kind
    );
    watcher.shutdown().await?;
    watcher.shutdown().await?;
    ensure!(watcher.is_shutdown(), "the second shutdown is a no-op");
    Ok(())
}
