//! End-to-end coverage for the `notify`-backed workspace watcher, driven through the crate root.
//!
//! Every test builds its own `tempfile` workspace, starts one debounced watcher and mutates
//! exactly one target, so a slow or lost notification can only fail the step that produced it.
//! Each await carries its own delivery budget ([`WAIT`]); no test shares one deadline across
//! several targets, and no assertion depends on the order in which two independent targets are
//! mutated.
//!
//! # Why the fixture pre-creates the skill directory
//!
//! The flake this file used to have came from `write_skill` creating `skills/demo/` *after* the
//! watcher had started and then immediately writing `skills/demo/SKILL.md`:
//!
//! - a recursive target registers one inotify watch per **existing** directory, and `notify`
//!   walks the tree synchronously while `Watcher::watch` blocks until its event loop
//!   acknowledges the registration — so everything that exists before `start()` returns is
//!   watched, with no extra readiness handshake to wait for;
//! - a directory created later is only watched after that same event loop thread processes the
//!   `Create(IsDir)` notification and adds the new watch (`add_watch_by_event` in
//!   `notify/src/inotify.rs`). That happens asynchronously on the backend's own thread, so a
//!   write issued immediately afterwards can reach the kernel **before** the watch exists, and
//!   the kernel then has no notification to deliver: the event is not delayed, it is lost.
//! - the directory creation itself is still reported by the parent watch, but as an event whose
//!   path is `skills/demo` — which the `SKILL.md` predicate rejects — so the test simply waited
//!   out its whole budget and failed.
//!
//! Under a loaded `--all-features` run (many binaries and unit tests competing for CPUs) the
//! backend thread is scheduled late and the window widens, which is why the suite passed alone
//! and failed intermittently in CI. [`Fixture::new`] therefore creates every watched parent
//! directory *before* `start()`, and the tests only write files below them; never re-introduce a
//! create-directory-then-immediately-write-inside-it step here.
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

/// Delivery budget of **one** awaited event.
///
/// The budget is deliberately generous and applied per await rather than once for the whole
/// test: a single inotify notification is delivered in microseconds, but a heavily loaded
/// `--all-features` run can deschedule the backend, the intake task and this task for a while,
/// so anything tighter risks a second, different flake. Reaching the budget is still a failure
/// (a lost notification must not turn into a hang), and every test awaits at most one event per
/// mutation, so the suite time stays bounded by a small multiple of this value.
const WAIT: Duration = Duration::from_secs(10);

/// Debounce window of the watchers under test.
const DEBOUNCE: Duration = Duration::from_millis(100);

/// Debounce window of the coalescing test.
///
/// Wider than [`DEBOUNCE`] because that test asserts *how many* events a burst produces: the
/// five writes are synchronous, but a page-cache stall of one scheduling slice on a loaded
/// machine must not be able to split them across two windows. The crate's own unit test uses the
/// same window for the same reason. [`SETTLE`] still exceeds it, so a second window would be
/// observed rather than missed.
const BURST_DEBOUNCE: Duration = Duration::from_millis(250);

/// Quiet window that proves no further event is due for one path: four debounce windows, so a
/// second pending window would already have been published.
const SETTLE: Duration = Duration::from_millis(400);

/// The workspace/home layout the watcher plan discovers.
struct Fixture {
    workspace: PathBuf,
    home: OpenJiuwenHome,
    skills: PathBuf,
    skill_dir: PathBuf,
    identity: PathBuf,
    mcp: PathBuf,
}

impl Fixture {
    /// Creates `skills/demo/`, `IDENTITY.md` and `.openjiuwen/mcp.json` before any watcher
    /// starts, so registration covers every path a test later mutates.
    ///
    /// `skills/demo` (not only `skills`) exists up front on purpose: see the module docs — a
    /// directory created after `start()` is watched asynchronously, and a file written into it
    /// right away can be missed outright.
    fn new(root: &Path) -> Result<Self> {
        let workspace = root.join("workspace");
        let home = OpenJiuwenHome::from_root(root.join("home"));
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(home.root())?;
        let skills = workspace.join("skills");
        let skill_dir = skills.join("demo");
        std::fs::create_dir_all(&skill_dir)?;
        let identity = workspace.join("IDENTITY.md");
        write(&identity, "identity v1\n")?;
        let mcp = workspace.join(".openjiuwen").join("mcp.json");
        std::fs::create_dir_all(mcp.parent().expect("the config has a parent"))?;
        write(&mcp, r#"{"mcpServers":{}}"#)?;
        Ok(Self {
            workspace,
            home,
            skills,
            skill_dir,
            identity,
            mcp,
        })
    }

    /// The skill file the skills-target tests write: `skills/demo/SKILL.md`.
    fn skill_file(&self) -> PathBuf {
        self.skill_dir.join("SKILL.md")
    }

    /// Starts a debounced watcher over the whole discovered workspace plan.
    ///
    /// `start()` returns only once the backend has acknowledged every watch registration, so
    /// returning means "every path that existed at this point is watched".
    async fn watcher(&self) -> Result<WatcherHandle> {
        let watcher = WorkspaceWatcher::for_workspace(&self.workspace, &self.home, &MapEnv::new())
            .debounce(DEBOUNCE)
            .start()
            .await?;
        ensure!(
            watcher.target_count() >= 3,
            "the plan covers skills, the instruction file and the MCP config, got {}",
            watcher.target_count()
        );
        ensure!(
            !watcher.is_shutdown(),
            "a freshly started watcher is running"
        );
        Ok(watcher)
    }
}

/// Writes `contents` to `path`.
fn write(path: &Path, contents: &str) -> Result<()> {
    Ok(std::fs::write(path, contents)?)
}

/// Writes the skill file `<root>/<name>/SKILL.md` with a front-matter description.
///
/// The skill directory itself must already exist when the watcher runs; see [`Fixture::new`].
fn write_skill(root: &Path, name: &str, description: &str) -> Result<()> {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory)?;
    write(
        &directory.join("SKILL.md"),
        &format!("---\ndescription: {description}\n---\n# {name}\n"),
    )
}

/// Awaits the next `Event` update matching `matches`, skipping failures and other targets.
///
/// Only this one await is bounded by [`WAIT`]; the caller issues the mutation immediately before
/// it, so a passing call proves that mutation was delivered.
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
///
/// Used only after the expected event was already observed, to prove that no *further* event is
/// published (a coalescing window or a shutdown must not leak one).
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

/// The recursive `skills` target delivers a new skill file inside an existing directory.
#[tokio::test]
async fn skills_target_delivers_a_new_skill_file() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let watcher = fixture.watcher().await?;
    let mut subscription = watcher.subscribe();

    // `skills/demo/` exists since the fixture, so the recursive registration already watches it
    // and this write is delivered; creating the directory here instead would race registration.
    write_skill(&fixture.skills, "demo", "demo skill")?;
    let event = next_event_for(&mut subscription, |event| {
        event.kind == WatchTargetKind::Skills && event.path.ends_with("SKILL.md")
    })
    .await?;

    let expected = fixture.skill_file();
    ensure!(
        event.path == expected,
        "the event names the new skill file, expected {expected:?}, got {:?}",
        event.path
    );
    ensure!(
        event.change == ChangeKind::Created,
        "a file that appears after start is Created, got {:?}",
        event.change
    );
    ensure!(
        event.label.as_deref() == Some("skills:workspace"),
        "the workspace skill target keeps its label, got {:?}",
        event.label
    );
    ensure!(event.sequence >= 1, "sequences start at one");
    ensure!(
        watcher.dropped_notifications() == 0 && watcher.dropped_dispatches() == 0,
        "a single notification must not fill either bounded queue"
    );
    watcher.shutdown().await?;
    ensure!(watcher.is_shutdown(), "the handle reports the shutdown");
    Ok(())
}

/// The single-file `IDENTITY.md` target delivers a rewrite as `Modified`.
#[tokio::test]
async fn instruction_file_target_delivers_a_rewrite() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let watcher = fixture.watcher().await?;
    let mut subscription = watcher.subscribe();

    write(&fixture.identity, "identity v2\n")?;
    let event = next_event_for(&mut subscription, |event| event.path == fixture.identity).await?;

    ensure!(
        event.kind == WatchTargetKind::InstructionFile,
        "got {:?}",
        event.kind
    );
    ensure!(
        event.change == ChangeKind::Modified,
        "an existing file rewrite is Modified, got {:?}",
        event.change
    );
    ensure!(
        event.label.as_deref() == Some("instruction:workspace:IDENTITY.md"),
        "got {:?}",
        event.label
    );
    watcher.shutdown().await?;
    ensure!(watcher.is_shutdown(), "the handle reports the shutdown");
    Ok(())
}

/// The single-file MCP configuration target delivers a rewrite of the existing file.
#[tokio::test]
async fn mcp_config_target_delivers_a_rewrite() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let watcher = fixture.watcher().await?;
    let mut subscription = watcher.subscribe();

    write(
        &fixture.mcp,
        r#"{"mcpServers":{"demo":{"command":"demo"}}}"#,
    )?;
    let event = next_event_for(&mut subscription, |event| event.path == fixture.mcp).await?;

    ensure!(
        event.kind == WatchTargetKind::McpConfig,
        "got {:?}",
        event.kind
    );
    ensure!(
        event.change == ChangeKind::Modified,
        "an existing config rewrite is Modified, got {:?}",
        event.change
    );
    ensure!(
        event.label.as_deref() == Some("mcp:workspace"),
        "got {:?}",
        event.label
    );
    watcher.shutdown().await?;
    ensure!(watcher.is_shutdown(), "the handle reports the shutdown");
    Ok(())
}

/// One watcher numbers its events monotonically across targets, one mutation at a time.
///
/// Every step awaits its own event under its own [`WAIT`] budget before the next mutation is
/// issued, so the sequence order is causal rather than a timing guess, and no deadline is
/// shared between the three targets.
#[tokio::test]
async fn sequences_increase_across_targets_of_one_watcher() -> Result<()> {
    let temp = TempDir::new()?;
    let fixture = Fixture::new(temp.path())?;
    let watcher = fixture.watcher().await?;
    let mut subscription = watcher.subscribe();

    write(&fixture.identity, "identity v2\n")?;
    let identity =
        next_event_for(&mut subscription, |event| event.path == fixture.identity).await?;

    write(
        &fixture.mcp,
        r#"{"mcpServers":{"demo":{"command":"demo"}}}"#,
    )?;
    let mcp = next_event_for(&mut subscription, |event| event.path == fixture.mcp).await?;

    write_skill(&fixture.skills, "demo", "demo skill")?;
    let skill = next_event_for(&mut subscription, |event| {
        event.kind == WatchTargetKind::Skills && event.path == fixture.skill_file()
    })
    .await?;

    ensure!(
        identity.sequence < mcp.sequence && mcp.sequence < skill.sequence,
        "sequences are monotonic across targets, got {identity:?} < {mcp:?} < {skill:?}"
    );
    ensure!(
        mcp.kind == WatchTargetKind::McpConfig && skill.kind == WatchTargetKind::Skills,
        "each event keeps the kind of its own target"
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
        .debounce(BURST_DEBOUNCE)
        .watch_dir(WatchTargetKind::Skills, &fixture.skills)
        .start()
        .await?;
    let mut subscription = watcher.subscribe();
    let path = fixture.skills.join("notes.md");
    for revision in 0..5 {
        write(&path, &format!("revision {revision}\n"))?;
    }

    // The burst is written synchronously, so every write falls inside one debounce window; the
    // awaited event is the coalesced one, and the settle window proves no second one follows.
    let first = next_event_for(&mut subscription, |event| event.path == path).await?;
    ensure!(
        matches!(first.change, ChangeKind::Created | ChangeKind::Modified),
        "the first observation is the creation (or an in-place rewrite), got {:?}",
        first.change
    );
    ensure!(
        first.kind == WatchTargetKind::Skills,
        "got {:?}",
        first.kind
    );
    let further = events_for(&mut subscription, &path, SETTLE).await;
    ensure!(
        further.is_empty(),
        "one debounce window publishes one event, got {further:?}"
    );
    watcher.shutdown().await?;
    Ok(())
}

/// Shutdown is clean and idempotent, publishes nothing afterwards, and a single-file target
/// still watches a file that appears after start.
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

    // `shutdown` awaits both tasks, so no update may be produced afterwards; the OS watch is
    // gone too, so this rewrite cannot reach the pipeline at all.
    write(&missing, "workspace instructions, revised\n")?;
    let late = events_for(&mut subscription, &missing, SETTLE).await;
    ensure!(
        late.is_empty(),
        "a shut-down watcher publishes nothing, got {late:?}"
    );
    Ok(())
}
