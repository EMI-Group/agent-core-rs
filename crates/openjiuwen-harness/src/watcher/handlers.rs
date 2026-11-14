//! The concrete reload bodies the watcher ships: skills, instruction files and
//! MCP configuration.
//!
//! Each body wraps one piece of harness state and refreshes it from disk on
//! demand:
//!
//! - [`SkillLibraryReloader`] re-reads `skills_state.json` and re-discovers the
//!   `SKILL.md` trees of a [`SkillLibrary`];
//! - [`InstructionReloader`] re-reads `IDENTITY.md` and `OPENJIUWEN.md` through
//!   [`InstructionFiles::load`];
//! - [`McpConfigReloader`] re-reads and merges the `mcp.json` files, later
//!   paths overriding earlier ones by server name.
//!
//! Each body has a matching [`ReloadHandler`] adapter ([`SkillsReloadHandler`],
//! [`InstructionReloadHandler`], [`McpConfigReloadHandler`]) so a caller can
//! register it with [`ReloadHub`](crate::watcher::ReloadHub) and then read the
//! refreshed snapshot without touching the filesystem again.
//!
//! The bodies live here, instead of beside the state they refresh, because the
//! watcher node owns this module and composes the public `skills`, `cli` and
//! `spec` APIs only. Nothing in this file duplicates their parsing rules.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::mcp::McpServerConfig;

use crate::cli::instruction::InstructionFiles;
use crate::cli::mcp_config::McpConfiguration;
use crate::skills::{Skill, SkillLibrary};
use crate::watcher::event::{WatchEvent, WatchTargetKind};
use crate::watcher::hub::ReloadHandler;

/// Re-discovers a [`SkillLibrary`] and caches the resulting snapshot.
///
/// The library is refreshed under a write lock — [`SkillLibrary::load_all_state`]
/// merges every root's `skills_state.json` into the disabled set — and the
/// discovery result is cached so a caller can read the same snapshot without a
/// second filesystem walk.
pub struct SkillLibraryReloader {
    library: tokio::sync::RwLock<SkillLibrary>,
    skills: tokio::sync::RwLock<Vec<Skill>>,
}

impl SkillLibraryReloader {
    /// Creates a reloader over `library` with an empty cache.
    ///
    /// The library's own roots are used; nothing is discovered until
    /// [`Self::reload`] runs.
    #[must_use]
    pub fn new(library: SkillLibrary) -> Self {
        Self {
            library: tokio::sync::RwLock::new(library),
            skills: tokio::sync::RwLock::new(Vec::new()),
        }
    }

    /// Re-reads the library state, re-discovers the skills and caches them.
    ///
    /// Disabled skills stay in the returned snapshot (discovery never filters
    /// them); use [`Self::library`] plus [`SkillLibrary::select`] to apply the
    /// disabled set.
    ///
    /// # Errors
    ///
    /// Reserved for API consistency, as in [`SkillLibrary::discover`]: IO
    /// problems are logged and skipped rather than reported.
    pub async fn reload(&self) -> Result<Vec<Skill>> {
        let library = {
            let mut library = self.library.write().await;
            library.load_all_state().await;
            library.clone()
        };
        let skills = library.discover().await?;
        *self.skills.write().await = skills.clone();
        tracing::debug!(
            roots = library.roots.len(),
            disabled = library.disabled_names().len(),
            skills = skills.len(),
            "reloaded the skill library"
        );
        Ok(skills)
    }

    /// Returns the cached snapshot, empty until the first successful reload.
    pub async fn current(&self) -> Vec<Skill> {
        self.skills.read().await.clone()
    }

    /// Returns a copy of the library, including its refreshed disabled set.
    pub async fn library(&self) -> SkillLibrary {
        self.library.read().await.clone()
    }
}

impl std::fmt::Debug for SkillLibraryReloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillLibraryReloader")
            .finish_non_exhaustive()
    }
}

/// [`ReloadHandler`] adapter for a [`SkillLibraryReloader`].
pub struct SkillsReloadHandler {
    reloader: Arc<SkillLibraryReloader>,
    name: String,
    kinds: [WatchTargetKind; 1],
}

impl SkillsReloadHandler {
    /// Creates the handler under its default name `skills`.
    #[must_use]
    pub fn new(reloader: Arc<SkillLibraryReloader>) -> Self {
        Self {
            reloader,
            name: "skills".to_owned(),
            kinds: [WatchTargetKind::Skills],
        }
    }

    /// Overrides the handler name reported in reload failures.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl std::fmt::Debug for SkillsReloadHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillsReloadHandler")
            .field("name", &self.name)
            .field("kinds", &self.kinds)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ReloadHandler for SkillsReloadHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn target_kinds(&self) -> &[WatchTargetKind] {
        &self.kinds
    }

    async fn reload(&self, _event: &WatchEvent) -> Result<()> {
        self.reloader.reload().await.map(|_skills| ())
    }
}

/// Re-reads the identity preamble and memory layers of a workspace.
pub struct InstructionReloader {
    workspace: PathBuf,
    home: OpenJiuwenHome,
    state: parking_lot::RwLock<InstructionFiles>,
}

impl InstructionReloader {
    /// Loads the instruction files once and returns a reloader around them.
    ///
    /// # Errors
    ///
    /// Propagates [`InstructionFiles::load`]; discovery itself logs and skips
    /// unreadable documents.
    pub fn load(workspace: &Path, home: &OpenJiuwenHome) -> Result<Self> {
        let state = InstructionFiles::load(workspace, home)?;
        Ok(Self {
            workspace: workspace.to_path_buf(),
            home: home.clone(),
            state: parking_lot::RwLock::new(state),
        })
    }

    /// Re-reads every instruction document and caches the result.
    ///
    /// # Errors
    ///
    /// Propagates [`InstructionFiles::load`].
    // The body is synchronous (`InstructionFiles::load` reads files directly), but
    // the reload seam is async so every body can be awaited uniformly.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn reload(&self) -> Result<InstructionFiles> {
        let files = InstructionFiles::load(&self.workspace, &self.home)?;
        tracing::debug!(
            workspace = %self.workspace.display(),
            documents = files.len(),
            "reloaded the instruction files"
        );
        let mut state = self.state.write();
        *state = files;
        Ok(state.clone())
    }

    /// Returns the cached instruction files.
    #[must_use]
    pub fn current(&self) -> InstructionFiles {
        self.state.read().clone()
    }

    /// Returns the cached system-prompt fragment, `None` when nothing was found.
    #[must_use]
    pub fn system_prompt(&self) -> Option<String> {
        self.state.read().system_prompt()
    }
}

impl std::fmt::Debug for InstructionReloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstructionReloader")
            .field("workspace", &self.workspace)
            .field("home", &self.home)
            .finish_non_exhaustive()
    }
}

/// [`ReloadHandler`] adapter for an [`InstructionReloader`].
pub struct InstructionReloadHandler {
    reloader: Arc<InstructionReloader>,
    name: String,
    kinds: [WatchTargetKind; 1],
}

impl InstructionReloadHandler {
    /// Creates the handler under its default name `instruction`.
    #[must_use]
    pub fn new(reloader: Arc<InstructionReloader>) -> Self {
        Self {
            reloader,
            name: "instruction".to_owned(),
            kinds: [WatchTargetKind::InstructionFile],
        }
    }

    /// Overrides the handler name reported in reload failures.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl std::fmt::Debug for InstructionReloadHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstructionReloadHandler")
            .field("name", &self.name)
            .field("kinds", &self.kinds)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ReloadHandler for InstructionReloadHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn target_kinds(&self) -> &[WatchTargetKind] {
        &self.kinds
    }

    async fn reload(&self, _event: &WatchEvent) -> Result<()> {
        self.reloader.reload().await.map(|_files| ())
    }
}

/// Loads and merges the `mcp.json` files a workspace uses.
///
/// Paths are read in order — the precedence of
/// [`mcp_config_paths`](crate::cli::paths::mcp_config_paths) — and later files
/// override earlier ones by server name, exactly like the one-shot CLI load.
pub struct McpConfigReloader {
    paths: Vec<PathBuf>,
    state: parking_lot::RwLock<Option<McpConfiguration>>,
}

impl McpConfigReloader {
    /// Performs the initial merged load and returns a reloader around it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) naming
    /// the path when a file cannot be read or parsed.
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let state = parking_lot::RwLock::new(merge_files(&paths)?);
        Ok(Self { paths, state })
    }

    /// Re-reads and re-merges every path, caching the result.
    ///
    /// Returns `None` when none of the paths exists.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) naming
    /// the path when a file cannot be read or parsed.
    // The body is synchronous (`McpConfiguration::load` reads files directly), but
    // the reload seam is async so every body can be awaited uniformly.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn reload(&self) -> Result<Option<McpConfiguration>> {
        let merged = merge_files(&self.paths)?;
        tracing::debug!(
            paths = self.paths.len(),
            servers = merged.as_ref().map_or(0, |config| config.mcp_servers.len()),
            "reloaded the mcp configuration"
        );
        let mut state = self.state.write();
        *state = merged;
        Ok(state.clone())
    }

    /// Returns the cached configuration, `None` when no watched file exists.
    #[must_use]
    pub fn current(&self) -> Option<McpConfiguration> {
        self.state.read().clone()
    }

    /// Converts the cached configuration into foundation server configurations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) naming
    /// the server when an entry is invalid; the message comes from
    /// [`McpConfiguration::servers`].
    pub fn servers(&self) -> Result<Vec<McpServerConfig>> {
        match &*self.state.read() {
            Some(config) => config.servers(),
            None => Ok(Vec::new()),
        }
    }
}

impl std::fmt::Debug for McpConfigReloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpConfigReloader")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

/// Reads `paths` in order and merges them by server name.
///
/// Returns `None` when none of the paths exists, so an empty configuration is
/// never confused with an absent one.
fn merge_files(paths: &[PathBuf]) -> Result<Option<McpConfiguration>> {
    let mut merged: BTreeMap<String, crate::cli::mcp_config::McpServerSpec> = BTreeMap::new();
    let mut found = false;
    for path in paths {
        if let Some(config) = McpConfiguration::load(path)? {
            found = true;
            merged.extend(config.mcp_servers);
        }
    }
    Ok(if found {
        Some(McpConfiguration {
            mcp_servers: merged,
        })
    } else {
        None
    })
}

/// [`ReloadHandler`] adapter for a [`McpConfigReloader`].
pub struct McpConfigReloadHandler {
    reloader: Arc<McpConfigReloader>,
    name: String,
    kinds: [WatchTargetKind; 1],
}

impl McpConfigReloadHandler {
    /// Creates the handler under its default name `mcp`.
    #[must_use]
    pub fn new(reloader: Arc<McpConfigReloader>) -> Self {
        Self {
            reloader,
            name: "mcp".to_owned(),
            kinds: [WatchTargetKind::McpConfig],
        }
    }

    /// Overrides the handler name reported in reload failures.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl std::fmt::Debug for McpConfigReloadHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpConfigReloadHandler")
            .field("name", &self.name)
            .field("kinds", &self.kinds)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ReloadHandler for McpConfigReloadHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn target_kinds(&self) -> &[WatchTargetKind] {
        &self.kinds
    }

    async fn reload(&self, _event: &WatchEvent) -> Result<()> {
        self.reloader.reload().await.map(|_config| ())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::skills::{SkillBudget, SKILLS_STATE_FILENAME};
    use openjiuwen_foundation::tool::mcp::McpClientType;
    use tempfile::TempDir;

    use crate::watcher::event::ChangeKind;

    /// Writes `<root>/<name>/SKILL.md` with a description in its front matter.
    fn write_skill(root: &Path, name: &str, description: &str) {
        let directory = root.join(name);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n# {name}\n"),
        )
        .unwrap();
    }

    /// Builds a reloader over a temporary skills root holding one skill.
    fn skills_fixture() -> (TempDir, Arc<SkillLibraryReloader>, PathBuf) {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).unwrap();
        write_skill(&root, "alpha", "The alpha skill");
        let mut library = SkillLibrary::new();
        library.add_root(&root);
        (temp, Arc::new(SkillLibraryReloader::new(library)), root)
    }

    /// Returns the sorted names of `skills`.
    fn names(skills: &[Skill]) -> Vec<String> {
        let mut names: Vec<String> = skills.iter().map(|skill| skill.name.clone()).collect();
        names.sort();
        names
    }

    fn event(kind: WatchTargetKind) -> WatchEvent {
        WatchEvent {
            sequence: 4,
            kind,
            path: PathBuf::from("/w/skills"),
            change: ChangeKind::Modified,
            previous_path: None,
            label: Some("skills:workspace".to_owned()),
        }
    }

    #[tokio::test]
    async fn skills_are_discovered_and_new_skills_appear_on_reload() {
        let (_temp, reloader, root) = skills_fixture();
        assert_eq!(reloader.current().await, Vec::<Skill>::new());

        let skills = reloader.reload().await.unwrap();
        assert_eq!(names(&skills), vec!["alpha"]);
        assert_eq!(names(&reloader.current().await), vec!["alpha"]);
        assert_eq!(skills[0].description, "The alpha skill");

        write_skill(&root, "beta", "The beta skill");
        let skills = reloader.reload().await.unwrap();
        assert_eq!(names(&skills), vec!["alpha", "beta"]);
        assert_eq!(names(&reloader.current().await), vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn a_removed_skill_disappears_on_reload() {
        let (_temp, reloader, root) = skills_fixture();
        write_skill(&root, "beta", "The beta skill");
        assert_eq!(reloader.reload().await.unwrap().len(), 2);

        std::fs::remove_dir_all(root.join("beta")).unwrap();
        assert_eq!(names(&reloader.reload().await.unwrap()), vec!["alpha"]);
    }

    #[tokio::test]
    async fn the_state_file_disables_a_skill() {
        let (_temp, reloader, root) = skills_fixture();
        write_skill(&root, "beta", "The beta skill");
        std::fs::write(
            root.join(SKILLS_STATE_FILENAME),
            r#"{"skill_configs": {"beta": {"enabled": false}}}"#,
        )
        .unwrap();

        let skills = reloader.reload().await.unwrap();
        let library = reloader.library().await;
        assert!(library.is_disabled("beta"));
        assert!(!library.is_disabled("alpha"));
        // Discovery keeps both; the library filter selects only the enabled one.
        assert_eq!(names(&skills), vec!["alpha", "beta"]);
        assert_eq!(
            names(&library.select(&skills, SkillBudget::unlimited())),
            vec!["alpha"]
        );
    }

    #[tokio::test]
    async fn the_skills_handler_delegates_to_the_reloader() {
        let (_temp, reloader, root) = skills_fixture();
        let handler = SkillsReloadHandler::new(reloader);
        assert_eq!(handler.name(), "skills");
        assert_eq!(handler.target_kinds(), &[WatchTargetKind::Skills]);

        handler
            .reload(&event(WatchTargetKind::Skills))
            .await
            .unwrap();
        write_skill(&root, "beta", "The beta skill");
        handler
            .reload(&event(WatchTargetKind::Skills))
            .await
            .unwrap();

        let renamed =
            SkillsReloadHandler::new(Arc::new(SkillLibraryReloader::new(SkillLibrary::new())))
                .with_name("skills-pinned");
        assert_eq!(renamed.name(), "skills-pinned");
    }

    #[tokio::test]
    async fn instruction_reload_reflects_a_rewritten_identity_file() {
        let temp = TempDir::new().unwrap();
        let workspace = temp.path().join("workspace");
        let home_root = temp.path().join("home");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&home_root).unwrap();
        std::fs::write(workspace.join("IDENTITY.md"), "first identity").unwrap();

        let home = OpenJiuwenHome::from_root(&home_root);
        let reloader = Arc::new(InstructionReloader::load(&workspace, &home).unwrap());
        assert_eq!(
            reloader.system_prompt().as_deref(),
            Some("first identity"),
            "initial state is loaded by `load`"
        );

        std::fs::write(workspace.join("IDENTITY.md"), "second identity").unwrap();
        let files = reloader.reload().await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(reloader.system_prompt().as_deref(), Some("second identity"));
        assert_eq!(reloader.current(), files);

        let handler = InstructionReloadHandler::new(Arc::clone(&reloader));
        assert_eq!(handler.name(), "instruction");
        assert_eq!(handler.target_kinds(), &[WatchTargetKind::InstructionFile]);
        std::fs::write(workspace.join("IDENTITY.md"), "third identity").unwrap();
        handler
            .reload(&event(WatchTargetKind::InstructionFile))
            .await
            .unwrap();
        assert_eq!(reloader.system_prompt().as_deref(), Some("third identity"));
    }

    #[tokio::test]
    async fn instruction_reload_picks_up_a_new_memory_file() {
        let temp = TempDir::new().unwrap();
        let workspace = temp.path().join("workspace");
        let home_root = temp.path().join("home");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&home_root).unwrap();
        let home = OpenJiuwenHome::from_root(&home_root);
        let reloader = InstructionReloader::load(&workspace, &home).unwrap();
        assert!(reloader.current().is_empty());
        assert_eq!(reloader.system_prompt(), None);

        std::fs::write(workspace.join("OPENJIUWEN.md"), "project memory").unwrap();
        reloader.reload().await.unwrap();
        let prompt = reloader.system_prompt().unwrap();
        assert!(prompt.contains("### Project-level memory"));
        assert!(prompt.contains("project memory"));
    }

    /// Builds the home and workspace `mcp.json` files of the merge fixtures.
    fn mcp_fixture() -> (TempDir, Vec<PathBuf>) {
        let temp = TempDir::new().unwrap();
        let home_root = temp.path().join("home");
        let workspace = temp.path().join("workspace");
        std::fs::create_dir_all(&home_root).unwrap();
        std::fs::create_dir_all(workspace.join(".openjiuwen")).unwrap();

        let home_path = home_root.join("mcp.json");
        std::fs::write(
            &home_path,
            r#"{"mcpServers": {
                "shared": {"command": "home-tool"},
                "home_only": {"command": "home-only-tool"}
            }}"#,
        )
        .unwrap();
        let workspace_path = workspace.join(".openjiuwen").join("mcp.json");
        std::fs::write(
            &workspace_path,
            r#"{"mcpServers": {
                "shared": {"command": "workspace-tool"},
                "remote": {"transport": "sse", "url": "https://example.test/sse"}
            }}"#,
        )
        .unwrap();
        (temp, vec![home_path, workspace_path])
    }

    #[tokio::test]
    async fn mcp_reload_merges_home_and_workspace_with_workspace_winning() {
        let (_temp, paths) = mcp_fixture();
        let reloader = McpConfigReloader::new(paths).unwrap();

        let current = reloader.current().unwrap();
        let mut names: Vec<&str> = current.mcp_servers.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["home_only", "remote", "shared"]);

        let servers = reloader.servers().unwrap();
        let names: Vec<&str> = servers
            .iter()
            .map(|server| server.server_name.as_str())
            .collect();
        assert_eq!(names, vec!["home_only", "remote", "shared"]);
        let shared = &servers[2];
        assert_eq!(shared.server_path, "workspace-tool");
        assert_eq!(shared.client_type, McpClientType::Stdio);
        assert_eq!(servers[1].client_type, McpClientType::Sse);
        assert_eq!(servers[1].server_path, "https://example.test/sse");

        // `reload` returns the same merge as the initial load.
        let merged_again = reloader.reload().await.unwrap().unwrap();
        assert_eq!(merged_again.mcp_servers.len(), 3);
        assert_eq!(
            reloader.current().unwrap().mcp_servers.len(),
            3,
            "the cache follows the reload"
        );
    }

    #[tokio::test]
    async fn mcp_reload_sees_a_rewritten_file() {
        let (_temp, paths) = mcp_fixture();
        let reloader = Arc::new(McpConfigReloader::new(paths.clone()).unwrap());
        let handler = McpConfigReloadHandler::new(Arc::clone(&reloader));
        assert_eq!(handler.name(), "mcp");
        assert_eq!(handler.target_kinds(), &[WatchTargetKind::McpConfig]);

        std::fs::remove_file(&paths[1]).unwrap();
        handler
            .reload(&event(WatchTargetKind::McpConfig))
            .await
            .unwrap();
        let servers = reloader.servers().unwrap();
        let names: Vec<&str> = servers
            .iter()
            .map(|server| server.server_name.as_str())
            .collect();
        assert_eq!(names, vec!["home_only", "shared"]);
        assert_eq!(servers[1].server_path, "home-tool");

        std::fs::remove_file(&paths[0]).unwrap();
        reloader.reload().await.unwrap();
        assert!(reloader.current().is_none());
        assert_eq!(reloader.servers().unwrap(), Vec::<McpServerConfig>::new());
    }

    #[tokio::test]
    async fn mcp_reload_reports_an_invalid_document() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("mcp.json");
        std::fs::write(&path, "{ not json").unwrap();
        let error = McpConfigReloader::new(vec![path.clone()]).unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
        assert!(error.to_string().contains("mcp.json"));

        // A valid initial load keeps working, and a later corruption surfaces on
        // `reload` while the last good configuration stays cached.
        std::fs::write(&path, r#"{"mcpServers": {"tool": {"command": "x"}}}"#).unwrap();
        let reloader = McpConfigReloader::new(vec![path.clone()]).unwrap();
        assert_eq!(reloader.servers().unwrap().len(), 1);

        std::fs::write(&path, "{ not json").unwrap();
        let error = reloader.reload().await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
        assert_eq!(reloader.servers().unwrap()[0].server_name, "tool");
    }

    #[test]
    fn merge_files_returns_none_when_no_path_exists() {
        let temp = TempDir::new().unwrap();
        let missing = vec![
            temp.path().join("home").join("mcp.json"),
            temp.path().join("ws").join("mcp.json"),
        ];
        assert!(merge_files(&missing).unwrap().is_none());
    }
}
