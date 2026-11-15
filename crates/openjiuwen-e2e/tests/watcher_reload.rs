//! Harness watcher reload: instruction files, MCP configuration and skills through
//! [`ReloadHub`].
//!
//! This binary pins the headless half of the workspace watcher — the part a CLI or a hosting
//! process drives after its own change detection: a [`ReloadHub`] keeps the reload handlers in
//! registration order, runs only the handlers interested in an event's
//! [`WatchTargetKind`](openjiuwen::harness::watcher::WatchTargetKind), keeps going after a
//! failing body and records each failure as a
//! [`ReloadFailure`]. The three shipped bodies are exercised against a real filesystem: the
//! instruction reloader re-reads `IDENTITY.md`/`OPENJIUWEN.md`, the MCP reloader re-merges the
//! home and workspace `mcp.json` files, and the skills reloader re-discovers `SKILL.md` trees
//! and re-applies `skills_state.json`.
//!
//! # What the scenarios avoid
//!
//! The OS watcher itself is deliberately out of scope: nothing here starts
//! `WorkspaceWatcher::start` (the feature-gated `notify` backend), waits on a debounce window or
//! observes a filesystem notification. Each scenario builds its own [`WatchEvent`] literal,
//! mutates the disk and calls [`ReloadHub::dispatch`], so the file is fully offline, needs no
//! feature beyond the default build, and has no timing dependency at all — a reload is a direct
//! call, not a notification race.
//!
//! Every path lives inside one `tempfile::TempDir` per test; the home/workspace layout mirrors
//! the harness discovery rules (`<home>/IDENTITY.md`, `<home>/OPENJIUWEN.md`,
//! `<workspace>/IDENTITY.md`, `<workspace>/OPENJIUWEN.md`, `<workspace>/skills/<name>/SKILL.md`,
//! `<workspace>/.openjiuwen/mcp.json`). No assertion depends on a timestamp, a generated id or
//! the order in which two independent targets were touched.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use openjiuwen::core::common::config::OpenJiuwenHome;
use openjiuwen::core::common::error::Error;
use openjiuwen::core::foundation::tool::mcp::{McpClientType, McpServerConfig};
use openjiuwen::harness::cli::instruction::InstructionSource;
use openjiuwen::harness::skills::SKILLS_STATE_FILENAME;
use openjiuwen::harness::watcher::{
    ChangeKind, FnReloadHandler, InstructionReloadHandler, InstructionReloader,
    McpConfigReloadHandler, McpConfigReloader, ReloadFailure, ReloadHub, SkillLibraryReloader,
    SkillsReloadHandler, WatchEvent, WatchTargetKind,
};
use openjiuwen::harness::{Skill, SkillLibrary};
use tempfile::TempDir;

// --- fixtures ---------------------------------------------------------------

/// The home/workspace layout every scenario discovers through.
///
/// The `TempDir` is kept alive for the whole test body, so every path handed out by this struct
/// is owned by the scenario and removed on drop.
struct Fixture {
    /// Owner of every path below.
    temp: TempDir,
}

impl Fixture {
    /// Creates the temp root plus an empty `workspace/` and `home/` directory.
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("workspace")).unwrap();
        std::fs::create_dir_all(temp.path().join("home")).unwrap();
        Self { temp }
    }

    /// Returns `<temp>/workspace`, the project directory the reloaders read.
    fn workspace(&self) -> PathBuf {
        self.temp.path().join("workspace")
    }

    /// Returns `<temp>/home`, the `OpenJiuwenHome` root.
    fn home_root(&self) -> PathBuf {
        self.temp.path().join("home")
    }

    /// Wraps the home root in the harness `OpenJiuwenHome` handle.
    fn home(&self) -> OpenJiuwenHome {
        OpenJiuwenHome::from_root(self.home_root())
    }

    /// Returns `<home>/mcp.json`, the user-level MCP file.
    fn home_mcp_path(&self) -> PathBuf {
        self.home_root().join("mcp.json")
    }

    /// Returns `<workspace>/.openjiuwen/mcp.json`, the project-level MCP file.
    fn workspace_mcp_path(&self) -> PathBuf {
        self.workspace().join(".openjiuwen").join("mcp.json")
    }

    /// Returns the MCP precedence order: home first, workspace last (it wins).
    fn mcp_paths(&self) -> Vec<PathBuf> {
        vec![self.home_mcp_path(), self.workspace_mcp_path()]
    }

    /// Returns `<workspace>/skills`, the skill library root.
    fn skills_root(&self) -> PathBuf {
        self.workspace().join("skills")
    }
}

/// Writes `text` to `path`, creating the parent directories when needed.
fn write_file(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, text).unwrap();
}

/// Writes `<root>/<name>/SKILL.md` with a parseable front matter block.
fn write_skill(root: &Path, name: &str, description: &str) {
    write_file(
        &root.join(name).join("SKILL.md"),
        &format!("---\nname: {name}\ndescription: {description}\n---\n# {name}\n"),
    );
}

/// Returns the sorted skill names of `skills`.
fn skill_names(skills: &[Skill]) -> Vec<String> {
    let mut names: Vec<String> = skills.iter().map(|skill| skill.name.clone()).collect();
    names.sort();
    names
}

/// Returns the server names of `servers` in the order the reloader emitted them.
fn server_names(servers: &[McpServerConfig]) -> Vec<String> {
    servers
        .iter()
        .map(|server| server.server_name.clone())
        .collect()
}

/// Builds one watcher event with an explicit sequence number.
fn event(
    sequence: u64,
    kind: WatchTargetKind,
    path: impl Into<PathBuf>,
    change: ChangeKind,
) -> WatchEvent {
    WatchEvent {
        sequence,
        kind,
        path: path.into(),
        change,
        previous_path: None,
        label: Some(format!("{}:watcher_reload", kind.as_str())),
    }
}

// --- dispatch routing -------------------------------------------------------

/// One hub holding every shipped handler routes by kind, never aborts on a failure and treats an
/// unknown kind as a no-op.
///
/// The hub registers `instruction`, `skills`, `broken` (a `FnReloadHandler` that always errors),
/// `recorder` (a `FnReloadHandler` that logs) and `mcp`, in that order. Dispatching an
/// `InstructionFile` event must run `instruction`, `broken` and `recorder` — in registration
/// order, with the failure surfacing as a [`ReloadFailure`] — while leaving the skills and MCP
/// snapshots untouched. Dispatching `Other` must run nothing at all.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn dispatch_routes_by_target_kind_and_survives_a_failing_handler() {
    let fixture = Fixture::new();
    let identity_path = fixture.workspace().join("IDENTITY.md");
    write_file(&identity_path, "initial identity");
    // A discoverable skill makes "the skills reloader did not run" observable on disk, not just
    // as an empty cache.
    write_skill(&fixture.skills_root(), "alpha", "The alpha skill");

    let instruction =
        Arc::new(InstructionReloader::load(&fixture.workspace(), &fixture.home()).unwrap());
    let mut library = SkillLibrary::new();
    library.add_root(fixture.skills_root());
    let skills = Arc::new(SkillLibraryReloader::new(library));
    let mcp = Arc::new(McpConfigReloader::new(fixture.mcp_paths()).unwrap());

    // Shared log of the closure handlers: proves the failing body did not stop the next one.
    let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let broken = FnReloadHandler::new("broken", vec![WatchTargetKind::InstructionFile], |_event| {
        Box::pin(async { Err(Error::sys_op("broken cannot reload instruction files")) })
    });
    let recorder = {
        let log = Arc::clone(&log);
        FnReloadHandler::new(
            "recorder",
            vec![WatchTargetKind::InstructionFile],
            move |_event| {
                let log = Arc::clone(&log);
                Box::pin(async move {
                    log.lock().unwrap().push("recorder");
                    Ok(())
                })
            },
        )
    };

    let mut hub = ReloadHub::new();
    assert!(hub.is_empty());
    assert_eq!(hub.len(), 0);
    assert_eq!(hub.handler_names(), Vec::<&str>::new());
    hub = hub.with_handler(InstructionReloadHandler::new(Arc::clone(&instruction)));
    hub = hub.with_handler(SkillsReloadHandler::new(Arc::clone(&skills)));
    hub.add_handler(Arc::new(broken));
    hub.add_handler(Arc::new(recorder));
    hub = hub.with_handler(McpConfigReloadHandler::new(Arc::clone(&mcp)));

    assert!(!hub.is_empty());
    assert_eq!(hub.len(), 5);
    assert_eq!(
        hub.handler_names(),
        vec!["instruction", "skills", "broken", "recorder", "mcp"],
        "handlers report in registration order"
    );
    assert_eq!(
        format!("{hub:?}"),
        "ReloadHub { handlers: [\"instruction\", \"skills\", \"broken\", \"recorder\", \"mcp\"] }"
    );

    // An instruction event reaches the instruction body, the failing body and the recorder —
    // and nothing else.
    write_file(&identity_path, "routed identity v2");
    let instruction_event = event(
        3,
        WatchTargetKind::InstructionFile,
        identity_path.clone(),
        ChangeKind::Modified,
    );
    let failures = hub.dispatch(&instruction_event).await;
    assert_eq!(
        instruction.system_prompt().as_deref(),
        Some("routed identity v2"),
        "the instruction handler re-read the rewritten identity"
    );
    assert_eq!(*log.lock().unwrap(), vec!["recorder"]);
    assert_eq!(failures.len(), 1, "only the always-failing body failed");
    let failure = &failures[0];
    assert_eq!(failure.sequence, 3);
    assert_eq!(failure.handler, "broken");
    assert_eq!(failure.kind, WatchTargetKind::InstructionFile);
    assert_eq!(failure.path, identity_path);
    assert_eq!(failure.error_kind, "sys_op");
    assert!(failure
        .message
        .contains("broken cannot reload instruction files"));
    assert!(
        skills.current().await.is_empty(),
        "an instruction event must not run the skills reloader"
    );
    assert!(
        mcp.current().is_none(),
        "an instruction event must not run the mcp reloader"
    );

    // An unknown kind matches no handler: no reload, no failure.
    let cached_instruction = instruction.current();
    write_file(&identity_path, "unrouted identity v3");
    let other_event = event(
        4,
        WatchTargetKind::Other,
        fixture.workspace().join("notes.txt"),
        ChangeKind::Created,
    );
    let failures = hub.dispatch(&other_event).await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    assert_eq!(*log.lock().unwrap(), vec!["recorder"]);
    assert_eq!(
        instruction.current(),
        cached_instruction,
        "no handler is interested in an `Other` event"
    );
    assert_eq!(skills.current().await, Vec::<Skill>::new());
    assert!(mcp.current().is_none());
}

// --- instruction files ------------------------------------------------------

/// The instruction body re-reads a rewritten identity and appends new memory layers after it.
///
/// The scenario starts with `<workspace>/IDENTITY.md`, rewrites it with a marker and dispatches a
/// `Modified` event; the marker must reach `system_prompt()`. It then adds `<home>/OPENJIUWEN.md`
/// and `<workspace>/OPENJIUWEN.md` and dispatches again: the documents must read identity,
/// user-level memory, project-level memory, and the composed prompt must place the project memory
/// after the identity text.
#[tokio::test]
async fn instruction_reload_picks_up_a_rewritten_identity_and_new_memory() {
    let fixture = Fixture::new();
    let identity_path = fixture.workspace().join("IDENTITY.md");
    write_file(&identity_path, "stale identity");

    let reloader =
        Arc::new(InstructionReloader::load(&fixture.workspace(), &fixture.home()).unwrap());
    assert_eq!(
        reloader.system_prompt().as_deref(),
        Some("stale identity"),
        "`load` caches the initial state"
    );

    let hub = ReloadHub::new().with_handler(InstructionReloadHandler::new(Arc::clone(&reloader)));

    // A rewritten identity is picked up by one dispatched event.
    write_file(&identity_path, "identity marker v2");
    let failures = hub
        .dispatch(&event(
            1,
            WatchTargetKind::InstructionFile,
            identity_path.clone(),
            ChangeKind::Modified,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    assert!(reloader
        .system_prompt()
        .unwrap()
        .contains("identity marker v2"));

    let current = reloader.current();
    assert_eq!(current.len(), 1);
    let identity = current.identity.as_ref().unwrap();
    assert_eq!(identity.source, InstructionSource::WorkspaceIdentity);
    assert_eq!(identity.content, "identity marker v2");
    assert_eq!(identity.path, identity_path);

    // New memory files join the identity, home memory first then the project layer.
    write_file(
        &fixture.home_root().join("OPENJIUWEN.md"),
        "user memory marker",
    );
    let memory_path = fixture.workspace().join("OPENJIUWEN.md");
    write_file(&memory_path, "project memory marker");
    let failures = hub
        .dispatch(&event(
            2,
            WatchTargetKind::InstructionFile,
            memory_path.clone(),
            ChangeKind::Created,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());

    let current = reloader.current();
    assert_eq!(current.len(), 3);
    let documents: Vec<(InstructionSource, &str)> = current
        .documents()
        .map(|doc| (doc.source, doc.content.as_str()))
        .collect();
    assert_eq!(
        documents,
        vec![
            (InstructionSource::WorkspaceIdentity, "identity marker v2"),
            (InstructionSource::HomeMemory, "user memory marker"),
            (InstructionSource::WorkspaceMemory, "project memory marker"),
        ]
    );
    assert_eq!(
        current.documents().last().unwrap().path,
        memory_path,
        "the project memory document points at the file the event named"
    );

    let prompt = reloader.system_prompt().unwrap();
    let identity_at = prompt.find("identity marker v2").unwrap();
    let user_at = prompt.find("### User-level memory").unwrap();
    let project_at = prompt.find("### Project-level memory").unwrap();
    assert!(identity_at < user_at, "identity comes before memory");
    assert!(
        user_at < project_at,
        "home memory comes before project memory"
    );
    assert!(prompt.contains("project memory marker"));
}

// --- skills -----------------------------------------------------------------

/// The skills body re-discovers the root: additions appear, removals disappear and
/// `skills_state.json` disables a skill on the next reload.
///
/// The two pre-existing skill directories are only discovered by the first dispatch, which pins
/// that `SkillLibraryReloader::current()` is empty until a reload runs. A third directory then
/// appears through a `Created` event, one is removed through a `Removed` event, and finally the
/// state file disables a skill while discovery keeps reporting it (the filter is `select`).
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn skills_reload_tracks_additions_removals_and_the_state_file() {
    let fixture = Fixture::new();
    let root = fixture.skills_root();
    write_skill(&root, "alpha", "The alpha skill");
    write_skill(&root, "beta", "The beta skill");

    let mut library = SkillLibrary::new();
    assert!(library.add_root(&root));
    let reloader = Arc::new(SkillLibraryReloader::new(library));
    assert!(
        reloader.current().await.is_empty(),
        "nothing is discovered before the first reload"
    );

    let hub = ReloadHub::new().with_handler(SkillsReloadHandler::new(Arc::clone(&reloader)));

    let failures = hub
        .dispatch(&event(
            1,
            WatchTargetKind::Skills,
            root.clone(),
            ChangeKind::Created,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    assert_eq!(
        skill_names(&reloader.current().await),
        vec!["alpha", "beta"]
    );
    let discovered = reloader.current().await;
    assert_eq!(discovered[0].description, "The alpha skill");

    // A skill directory written after the initial load appears on the next dispatch.
    write_skill(&root, "gamma", "The gamma skill");
    let failures = hub
        .dispatch(&event(
            2,
            WatchTargetKind::Skills,
            root.join("gamma"),
            ChangeKind::Created,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    assert_eq!(
        skill_names(&reloader.current().await),
        vec!["alpha", "beta", "gamma"]
    );

    // Removing a directory drops it from the snapshot.
    std::fs::remove_dir_all(root.join("beta")).unwrap();
    let failures = hub
        .dispatch(&event(
            3,
            WatchTargetKind::Skills,
            root.join("beta"),
            ChangeKind::Removed,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    assert_eq!(
        skill_names(&reloader.current().await),
        vec!["alpha", "gamma"]
    );

    // The library state file switches a skill off; the reloader re-reads it.
    write_file(
        &root.join(SKILLS_STATE_FILENAME),
        r#"{"skill_configs": {"gamma": {"enabled": false}}}"#,
    );
    let failures = hub
        .dispatch(&event(
            4,
            WatchTargetKind::Skills,
            root.join(SKILLS_STATE_FILENAME),
            ChangeKind::Modified,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    let library = reloader.library().await;
    assert!(
        library.is_disabled("gamma"),
        "the state file disabled gamma"
    );
    assert!(!library.is_disabled("alpha"));
    assert_eq!(
        skill_names(&reloader.current().await),
        vec!["alpha", "gamma"],
        "discovery keeps disabled skills; `select` applies the disabled set"
    );
}

// --- MCP configuration ------------------------------------------------------

/// The MCP body re-merges the home and workspace files: workspace wins, a rewrite adds a server,
/// deleting the workspace file falls back to home, and corruption surfaces one typed failure
/// while the last good configuration stays cached.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn mcp_reload_merges_precedence_falls_back_and_reports_corruption() {
    let fixture = Fixture::new();
    write_file(
        &fixture.home_mcp_path(),
        r#"{"mcpServers": {
            "shared": {"command": "home-tool"},
            "home_only": {"command": "home-only-tool"}
        }}"#,
    );
    let workspace_path = fixture.workspace_mcp_path();
    write_file(
        &workspace_path,
        r#"{"mcpServers": {
            "shared": {"command": "workspace-tool"},
            "remote": {"transport": "sse", "url": "https://example.test/sse"}
        }}"#,
    );

    let home_path = fixture.home_mcp_path();
    let reloader =
        Arc::new(McpConfigReloader::new(vec![home_path.clone(), workspace_path.clone()]).unwrap());

    // The workspace file overrides the home entry of the same name.
    let servers = reloader.servers().unwrap();
    assert_eq!(
        server_names(&servers),
        vec!["home_only", "remote", "shared"]
    );
    let shared = &servers[2];
    assert_eq!(
        shared.server_path, "workspace-tool",
        "the workspace file wins by server name"
    );
    assert_eq!(shared.client_type, McpClientType::Stdio);
    assert_eq!(servers[1].client_type, McpClientType::Sse);
    assert_eq!(servers[1].server_path, "https://example.test/sse");

    let hub = ReloadHub::new().with_handler(McpConfigReloadHandler::new(Arc::clone(&reloader)));

    // A rewritten workspace file is re-merged and brings a new server.
    write_file(
        &workspace_path,
        r#"{"mcpServers": {
            "shared": {"command": "workspace-tool-v2"},
            "remote": {"transport": "sse", "url": "https://example.test/sse"},
            "added": {"command": "added-tool"}
        }}"#,
    );
    let failures = hub
        .dispatch(&event(
            1,
            WatchTargetKind::McpConfig,
            workspace_path.clone(),
            ChangeKind::Modified,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    let servers = reloader.servers().unwrap();
    assert_eq!(
        server_names(&servers),
        vec!["added", "home_only", "remote", "shared"]
    );
    assert_eq!(servers[3].server_path, "workspace-tool-v2");

    // Deleting the workspace file falls back to the home configuration.
    std::fs::remove_file(&workspace_path).unwrap();
    let failures = hub
        .dispatch(&event(
            2,
            WatchTargetKind::McpConfig,
            workspace_path.clone(),
            ChangeKind::Removed,
        ))
        .await;
    assert_eq!(failures, Vec::<ReloadFailure>::new());
    let servers = reloader.servers().unwrap();
    assert_eq!(server_names(&servers), vec!["home_only", "shared"]);
    assert_eq!(
        servers[1].server_path, "home-tool",
        "the home entry is visible again once the override is gone"
    );

    // Corruption is one typed failure; the last good configuration is still served.
    write_file(&home_path, "{ not json");
    let failures = hub
        .dispatch(&event(
            3,
            WatchTargetKind::McpConfig,
            home_path.clone(),
            ChangeKind::Modified,
        ))
        .await;
    assert_eq!(failures.len(), 1);
    let failure = &failures[0];
    assert_eq!(failure.handler, "mcp");
    assert_eq!(failure.error_kind, "config");
    assert_eq!(failure.kind, WatchTargetKind::McpConfig);
    assert_eq!(failure.sequence, 3);
    assert_eq!(failure.path, home_path);
    assert!(failure.message.contains("mcp.json"));
    let servers = reloader.servers().unwrap();
    assert_eq!(
        server_names(&servers),
        vec!["home_only", "shared"],
        "a failed reload keeps the last good snapshot"
    );
    assert_eq!(servers[1].server_path, "home-tool");
}
