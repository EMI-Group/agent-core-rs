//! Shared fixtures of the `spec` tests.
//!
//! Every fixture is a per-test `tempfile::TempDir`; nothing touches the network,
//! the real home directory, or wall-clock ordering.

use std::path::Path;

use crate::workspace::Workspace;
use tempfile::TempDir;

/// Creates an isolated workspace rooted in a fresh temporary directory.
pub(super) fn temp_workspace() -> (TempDir, Workspace) {
    let temp = TempDir::new().expect("temp workspace directory");
    let workspace = Workspace::for_language(temp.path(), "en");
    (temp, workspace)
}

/// Writes `contents` to `path`, creating parent directories.
pub(super) fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent directories");
    }
    std::fs::write(path, contents).expect("write fixture file");
}

/// The `.openjiuwen/agents` directory of `root`.
pub(super) fn agents_dir(root: &Path) -> std::path::PathBuf {
    root.join(".openjiuwen").join("agents")
}

/// The `.openjiuwen/plugins` directory of `root`.
pub(super) fn plugins_dir(root: &Path) -> std::path::PathBuf {
    root.join(".openjiuwen").join("plugins")
}

/// Returns a minimal valid YAML template named `name`.
pub(super) fn minimal_spec(name: &str) -> String {
    format!("name: {name}\ndescription: fixture template\ntools: [read_file]\n")
}
