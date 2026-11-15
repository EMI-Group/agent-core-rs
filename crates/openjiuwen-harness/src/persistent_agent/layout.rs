//! The on-disk layout of persistent-agent state.
//!
//! Every persistent-agent store takes an explicit root directory, so the caller decides where the
//! state lives. [`PersistentAgentLayout`] names the standard roots for one base directory and the
//! single workspace rule that resolves that base, so [`ExecutionLog::for_workspace`] and
//! [`FileSessionStore::for_workspace`] cannot disagree about where a journal or a session record
//! lives.
//!
//! [`ExecutionLog::for_workspace`]: super::log::ExecutionLog::for_workspace
//! [`FileSessionStore::for_workspace`]: super::session::FileSessionStore::for_workspace
use std::path::{Path, PathBuf};

use crate::workspace::Workspace;

/// Workspace node (or `<root_path>` fallback) holding persistent-agent state.
pub const PERSISTENT_AGENT_NODE: &str = "persistent_agent";

/// Directory under the persistent-agent node holding execution journals.
pub const LOGS_DIR: &str = "logs";

/// Directory under the persistent-agent node holding session records.
pub const SESSIONS_DIR: &str = "sessions";

/// Directory under the persistent-agent node holding execution links.
///
/// [`FileLinkStore`](super::continuation::FileLinkStore) takes an explicit root, so this is the
/// conventional location a host seeds and the one the catalogue/resume recipe uses, not a
/// path the store derives on its own.
pub const LINKS_DIR: &str = "links";

/// The standard roots of the persistent-agent state for one base directory.
///
/// The base is the workspace's `persistent_agent` node when the workspace schema has one, else
/// `<root_path>/persistent_agent`, matching every other workspace-scoped store in the crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentAgentLayout {
    /// Directory all three roots live under.
    base: PathBuf,
}

impl PersistentAgentLayout {
    /// Builds a layout under `base`, the directory that directly holds `logs/`, `sessions/` and
    /// `links/`.
    #[must_use]
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// Builds the layout the workspace-scoped stores use.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Self {
        let base = workspace
            .get_node_path(PERSISTENT_AGENT_NODE)
            .unwrap_or_else(|| workspace.root_path.join(PERSISTENT_AGENT_NODE));
        Self::new(base)
    }

    /// Returns the directory holding `logs/`, `sessions/` and `links/`.
    #[must_use]
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Returns the root of the execution journals.
    #[must_use]
    pub fn logs_root(&self) -> PathBuf {
        self.base.join(LOGS_DIR)
    }

    /// Returns the root of the persisted session records.
    #[must_use]
    pub fn sessions_root(&self) -> PathBuf {
        self.base.join(SESSIONS_DIR)
    }

    /// Returns the conventional root of the execution links.
    #[must_use]
    pub fn links_root(&self) -> PathBuf {
        self.base.join(LINKS_DIR)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use super::PersistentAgentLayout;
    use crate::workspace::Workspace;

    #[test]
    fn roots_are_the_directory_names_the_stores_use() {
        let layout = PersistentAgentLayout::new("/tmp/pa");
        assert_eq!(layout.base(), std::path::Path::new("/tmp/pa"));
        assert_eq!(layout.logs_root(), PathBuf::from("/tmp/pa/logs"));
        assert_eq!(layout.sessions_root(), PathBuf::from("/tmp/pa/sessions"));
        assert_eq!(layout.links_root(), PathBuf::from("/tmp/pa/links"));
    }

    #[test]
    fn workspace_layout_matches_the_store_fallback() {
        let dir = tempfile::TempDir::new().unwrap();
        let workspace = Workspace::for_language(dir.path(), "en");
        let layout = PersistentAgentLayout::for_workspace(&workspace);
        // The default schema has no `persistent_agent` node, so both the layout and the stores
        // fall back to `<root_path>/persistent_agent`.
        assert_eq!(
            layout.base(),
            dir.path().join("persistent_agent").as_path(),
            "the fallback must match ExecutionLog::for_workspace"
        );
        assert_eq!(layout.logs_root(), dir.path().join("persistent_agent/logs"));
        assert_eq!(
            layout.sessions_root(),
            dir.path().join("persistent_agent/sessions")
        );
    }
}
