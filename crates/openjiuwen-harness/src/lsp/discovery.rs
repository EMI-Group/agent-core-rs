//! Per-extension language-server discovery and workspace-root inference.
//!
//! Ports the registry half of `openjiuwen/harness/lsp/servers/`: a server definition lists the
//! file extensions it handles, a priority, and the marker files whose nearest ancestor is the
//! project root. This module turns such a definition list into a [`DiscoveryIndex`] that maps
//! one file path to the [`ResolvedServer`] that should serve it, plus the built-in definition
//! table mirrored from `servers/servers/*.py`.
//!
//! # Resolution rules
//!
//! - Extension matching is case-insensitive and accepts entries with or without a leading
//!   dot, so `"rs"`, `".rs"` and `".RS"` all match `main.rs`.
//! - Candidates are ordered by `(priority, registration index)`: the lowest priority wins and
//!   registration order breaks ties. The first candidate whose root inference succeeds wins.
//! - A file whose extension matches no server resolves to `None`; the caller decides the
//!   fallback (Python's manager reports "no LSP server for this file").
//! - Root inference walks from the file's parent directory up to the workspace root
//!   (inclusive) and picks the nearest ancestor containing a `root_markers` entry. With no
//!   markers, or no match, it uses `config.root` when that is inside the workspace root and
//!   the workspace root otherwise.
//! - Rust keeps Python's Cargo-workspace special case: when the server targets `.rs` (or is
//!   named `rust`), a nearest `Cargo.toml` without a `[workspace]` table makes the search
//!   continue upward, still bounded by the workspace root, for a `Cargo.toml` that has one.
//! - `resolve` returns `None` for a path outside the workspace root. Paths are compared
//!   lexically after making them absolute against the workspace root, so callers should pass
//!   one consistent spelling (the `lsp` tool canonicalizes before asking).
//!
//! Nothing here starts a process: [`builtin_server_configs`] only describes commands, so an
//! operator without the binary gets the typed spawn error from
//! [`StdioTransport`](crate::lsp::StdioTransport) instead of a silent absence.
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::config::LspServerConfig;

/// Identifies one running server: its id and the workspace root it was started for.
///
/// Python keys its instance map by `(server_id, root)`; the same server can run once per
/// monorepo sub-project.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub struct ServerKey {
    /// Server id; taken from [`LspServerConfig::name`], falling back to `language_id`.
    pub name: String,
    /// Workspace root the server was started for.
    pub root: PathBuf,
}

/// A server configuration paired with the root it should serve one file from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedServer {
    /// Identity of the instance: server id plus root.
    pub key: ServerKey,
    /// The configuration to launch; `root` is already set to [`ResolvedServer::root`].
    pub config: LspServerConfig,
}

impl ResolvedServer {
    /// The LSP language id this server uses.
    #[must_use]
    pub fn language_id(&self) -> &str {
        &self.config.language_id
    }

    /// The workspace root this server should be started for.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.key.root
    }
}

/// An immutable discovery table built once per tool and reused for every request.
#[derive(Clone, Debug)]
pub struct DiscoveryIndex {
    /// Absolute workspace root every resolved root must stay inside.
    workspace_root: PathBuf,
    /// Server definitions in registration order.
    servers: Vec<LspServerConfig>,
}

impl DiscoveryIndex {
    /// Builds an index over `servers` for files under `workspace_root`.
    ///
    /// `workspace_root` is made absolute once; the definitions are kept in the order given so
    /// registration order can break priority ties.
    #[must_use]
    pub fn new(workspace_root: PathBuf, servers: Vec<LspServerConfig>) -> Self {
        Self {
            workspace_root: normalize(
                &std::path::absolute(&workspace_root).unwrap_or(workspace_root),
            ),
            servers,
        }
    }

    /// Whether no server definitions are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// The configured definitions, in registration order.
    #[must_use]
    pub fn servers(&self) -> &[LspServerConfig] {
        &self.servers
    }

    /// Resolves `path` to the server that should handle it.
    ///
    /// Returns `None` when the path is outside the workspace root, has no file extension, or
    /// no configured server claims that extension.
    #[must_use]
    pub fn resolve(&self, path: &Path) -> Option<ResolvedServer> {
        let absolute = self.absolute(path);
        if !absolute.starts_with(&self.workspace_root) {
            return None;
        }
        let extension = extension_of(&absolute)?;
        for config in self.candidates_for_extension(&extension) {
            if let Some(root) = self.infer_root(config, &absolute) {
                return Some(Self::resolved(config, root));
            }
        }
        None
    }

    /// Resolves a server by its LSP `language_id`, ignoring extensions.
    ///
    /// The first definition with that language wins; the root is `config.root` when set and
    /// the workspace root otherwise.
    #[must_use]
    pub fn resolve_by_language(&self, language_id: &str) -> Option<ResolvedServer> {
        let config = self
            .servers
            .iter()
            .find(|candidate| candidate.language_id == language_id)?;
        let root = config
            .root
            .as_ref()
            .map_or_else(|| self.workspace_root.clone(), |root| self.absolute(root));
        Some(Self::resolved(config, root))
    }

    /// Every server claiming `extension`, ordered by priority then registration order.
    ///
    /// The argument is normalized like a configured entry: a leading dot is ignored and the
    /// comparison is case-insensitive.
    #[must_use]
    pub fn candidates_for_extension(&self, extension: &str) -> Vec<&LspServerConfig> {
        let wanted = normalize_extension(extension);
        if wanted.is_empty() {
            return Vec::new();
        }
        let mut candidates: Vec<(u32, usize, &LspServerConfig)> = self
            .servers
            .iter()
            .enumerate()
            .filter(|(_, config)| {
                config
                    .extensions
                    .iter()
                    .any(|candidate| normalize_extension(candidate) == wanted)
            })
            .map(|(index, config)| (config.priority, index, config))
            .collect();
        candidates.sort_by_key(|(priority, index, _)| (*priority, *index));
        candidates
            .into_iter()
            .map(|(_, _, config)| config)
            .collect()
    }

    /// Pairs a definition with `root`, cloning the config and fixing its `root` field.
    fn resolved(config: &LspServerConfig, root: PathBuf) -> ResolvedServer {
        let name = server_name(config).to_owned();
        let mut config = config.clone();
        config.root = Some(root.clone());
        ResolvedServer {
            key: ServerKey { name, root },
            config,
        }
    }

    /// Makes `path` absolute against the workspace root and normalizes it lexically.
    fn absolute(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            normalize(path)
        } else {
            normalize(&self.workspace_root.join(path))
        }
    }

    /// Infers the root a server should use for `file`; `None` only when `file` escaped the
    /// workspace root.
    fn infer_root(&self, config: &LspServerConfig, file: &Path) -> Option<PathBuf> {
        if !file.starts_with(&self.workspace_root) {
            return None;
        }
        let start = file.parent().map_or_else(
            || self.workspace_root.clone(),
            |parent| {
                if parent.starts_with(&self.workspace_root) {
                    parent.to_path_buf()
                } else {
                    self.workspace_root.clone()
                }
            },
        );
        let root = self
            .nearest_marker_dir(&start, &config.root_markers)
            .unwrap_or_else(|| self.fallback_root(config));
        Some(self.rust_workspace_root(config, root))
    }

    /// Returns the nearest ancestor of `start` (workspace root inclusive) holding a marker.
    fn nearest_marker_dir(&self, start: &Path, markers: &[String]) -> Option<PathBuf> {
        if markers.is_empty() {
            return None;
        }
        let mut current = Some(start);
        while let Some(dir) = current {
            if markers.iter().any(|marker| dir.join(marker).exists()) {
                return Some(dir.to_path_buf());
            }
            if dir == self.workspace_root {
                break;
            }
            current = dir.parent();
        }
        None
    }

    /// The configured root when it stays inside the workspace, else the workspace root.
    fn fallback_root(&self, config: &LspServerConfig) -> PathBuf {
        config.root.as_ref().map_or_else(
            || self.workspace_root.clone(),
            |root| {
                let candidate = self.absolute(root);
                if candidate.starts_with(&self.workspace_root) {
                    candidate
                } else {
                    self.workspace_root.clone()
                }
            },
        )
    }

    /// Applies Python's Cargo-workspace rule for `rust-analyzer`.
    fn rust_workspace_root(&self, config: &LspServerConfig, root: PathBuf) -> PathBuf {
        if !targets_rust(config) {
            return root;
        }
        let cargo = root.join("Cargo.toml");
        // A missing manifest at the inferred root means the marker search matched something
        // else, so there is no manifest chain to follow; a `[workspace]` manifest is final.
        if read_contains_workspace(&cargo) != Some(false) {
            return root;
        }
        let mut current = root.parent();
        while let Some(dir) = current {
            if !dir.starts_with(&self.workspace_root) {
                break;
            }
            if read_contains_workspace(&dir.join("Cargo.toml")) == Some(true) {
                return dir.to_path_buf();
            }
            if dir == self.workspace_root {
                break;
            }
            current = dir.parent();
        }
        root
    }
}

/// The id a server is known by: explicit `name`, else its language id, else `lsp`.
pub(super) fn server_name(config: &LspServerConfig) -> &str {
    if !config.name.is_empty() {
        &config.name
    } else if !config.language_id.is_empty() {
        &config.language_id
    } else {
        "lsp"
    }
}

/// The definitions Python registers in `servers/servers/*.py`, opt-in and unprobed.
///
/// Every entry has priority 10. Commands are not checked against `PATH`: an operator without
/// the binary gets the typed spawn error when the server is first used.
#[must_use]
pub fn builtin_server_configs() -> Vec<LspServerConfig> {
    vec![
        LspServerConfig::new("pyright-langserver")
            .with_name("pyright")
            .with_args(["--stdio"])
            .with_language_id("python")
            .with_extensions([".py", ".pyi"])
            .with_priority(10)
            .with_root_markers([
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                "requirements.txt",
                "Pipfile",
                "pyrightconfig.json",
            ]),
        LspServerConfig::new("rust-analyzer")
            .with_name("rust")
            .with_language_id("rust")
            .with_extensions([".rs"])
            .with_priority(10)
            .with_root_markers(["Cargo.toml"]),
        LspServerConfig::new("typescript-language-server")
            .with_name("typescript")
            .with_args(["--stdio"])
            .with_language_id("typescript")
            .with_extensions([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"])
            .with_priority(10)
            .with_root_markers([
                "package.json",
                "package-lock.json",
                "pnpm-lock.yaml",
                "yarn.lock",
            ]),
        LspServerConfig::new("gopls")
            .with_name("gopls")
            .with_language_id("go")
            .with_extensions([".go"])
            .with_priority(10)
            .with_root_markers(["go.work", "go.mod", "go.sum"])
            .with_startup_timeout(std::time::Duration::from_secs(60)),
        LspServerConfig::new("jdtls")
            .with_name("jdtls")
            .with_language_id("java")
            .with_extensions([".java"])
            .with_priority(10)
            .with_root_markers(["pom.xml", "build.gradle", "build.gradle.kts", ".project"]),
    ]
}

/// Normalizes one extension for comparison: no leading dots, lower case.
fn normalize_extension(extension: &str) -> String {
    extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase()
}

/// The lower-cased extension of `path`, or `None` when it has none.
fn extension_of(path: &Path) -> Option<String> {
    path.extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
        .filter(|extension| !extension.is_empty())
}

/// Whether `config` serves Rust files.
fn targets_rust(config: &LspServerConfig) -> bool {
    config.name == "rust"
        || config
            .extensions
            .iter()
            .any(|extension| normalize_extension(extension) == "rs")
}

/// Whether `path` is a readable file whose text contains a `[workspace]` table.
///
/// An unreadable manifest yields `None` (the caller treats that as "not a workspace root"), so
/// the downgrade is logged at debug level: a permissions problem or a `Cargo.toml` that is a
/// directory silently changes root inference, and that must be diagnosable.
fn read_contains_workspace(path: &Path) -> Option<bool> {
    match fs::read_to_string(path) {
        Ok(text) => Some(text.contains("[workspace]")),
        Err(error) => {
            tracing::debug!(
                target: "openjiuwen_harness::lsp",
                path = %path.display(),
                error = %error,
                "cannot read Cargo.toml; not treating it as a workspace root"
            );
            None
        }
    }
}

/// Lexically normalizes `path`: drops `.` components and resolves `..` against the prefix.
///
/// No filesystem access happens here, so a path that does not exist yet still compares
/// correctly with an existing workspace root.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() && !out.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}
