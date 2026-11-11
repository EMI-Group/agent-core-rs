//! Path resolution and traversal protection.
//!
//! Every filesystem and shell call resolves its paths through [`PathPolicy`], which
//! reproduces Python's `_resolve_path` (`local/fs_operation.py`): a relative path is
//! joined onto the workspace root, the result is normalized, symbolic links are
//! resolved, and — when the configuration sets `restrict_to_sandbox` — the resolved path
//! must live under one of the configured roots. A violation is reported with Python's
//! `Access denied: Path ... outside sandbox [...]` message.
//!
//! Security properties:
//!
//! - `..` segments are resolved lexically *before* any filesystem access.
//! - Symbolic links are resolved for the longest existing prefix of the path, so a link
//!   pointing outside the workspace cannot be used to escape it.
//! - A path equal to a root is allowed; a sibling with a shared textual prefix
//!   (`/root-other` vs `/root`) is not.
//!
//! ```
//! use std::path::Path;
//! use openjiuwen_sysop::workspace::{clean_path, is_within};
//!
//! assert_eq!(clean_path(Path::new("/a/b/../c")), Path::new("/a/c"));
//! assert!(is_within(Path::new("/a/b/c"), Path::new("/a/b")));
//! assert!(!is_within(Path::new("/a/bc"), Path::new("/a/b")));
//! ```

use std::path::{Component, Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use crate::config::LocalWorkConfig;
use crate::status::fs_execution_message;

/// The resolved filesystem policy of a local backend.
#[derive(Debug, Clone)]
pub struct PathPolicy {
    workspace: PathBuf,
    roots: Vec<PathBuf>,
    restricted: bool,
    max_file_size: Option<u64>,
    config: LocalWorkConfig,
}

impl PathPolicy {
    /// Builds the policy described by `config`.
    ///
    /// Roots and the workspace are canonicalized eagerly (a non-existent root keeps its
    /// lexically cleaned form), so containment checks compare real paths.
    #[must_use]
    pub fn new(config: &LocalWorkConfig) -> Self {
        let workspace = canonical_or_clean(&config.resolved_workspace_root());
        let roots = match &config.sandbox_root {
            Some(configured) if !configured.is_empty() => configured
                .iter()
                .map(|root| {
                    let path = PathBuf::from(root);
                    let path = if path.is_absolute() {
                        path
                    } else {
                        workspace.join(path)
                    };
                    canonical_or_clean(&path)
                })
                .collect(),
            _ => vec![workspace.clone()],
        };
        Self {
            workspace,
            roots,
            restricted: config.restrict_to_sandbox,
            max_file_size: config.max_file_size_bytes,
            config: config.clone(),
        }
    }

    /// Builds an unrestricted policy rooted at `workspace`.
    #[must_use]
    pub fn from_workspace(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        Self::new(&LocalWorkConfig::unrestricted(format_path(&workspace)))
    }

    /// The directory relative paths are resolved against.
    #[must_use]
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// The containment roots.
    #[must_use]
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Whether paths outside [`Self::roots`] are rejected.
    #[must_use]
    pub const fn is_restricted(&self) -> bool {
        self.restricted
    }

    /// The configured maximum readable file size, in bytes.
    #[must_use]
    pub const fn max_file_size(&self) -> Option<u64> {
        self.max_file_size
    }

    /// The configuration this policy was built from.
    #[must_use]
    pub fn config(&self) -> &LocalWorkConfig {
        &self.config
    }

    /// Resolves `path` against the workspace, enforcing the sandbox roots.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` carrying Python's filesystem execution message
    /// (code 199003) when `path` is blank, when it escapes the sandbox roots, or when a
    /// parent directory cannot be created.
    pub async fn resolve(&self, path: &str, create_parent: bool) -> Result<PathBuf> {
        if path.trim().is_empty() {
            return Err(resolve_error("path must not be empty"));
        }
        let raw = Path::new(path);
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            self.workspace.join(raw)
        };
        let cleaned = clean_path(&joined);
        let normalized = canonicalize_containing(&cleaned).await;
        self.ensure_contained(path, &normalized)?;
        if create_parent {
            if let Some(parent) = normalized.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|error| resolve_error(&format!("create parent dirs: {error}")))?;
            }
        }
        Ok(normalized)
    }

    /// Checks that an already-resolved path is inside the sandbox roots.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (code 199003) with Python's `Access denied` message.
    pub fn ensure_contained(&self, raw: &str, path: &Path) -> Result<()> {
        if !self.restricted {
            return Ok(());
        }
        if self.roots.iter().any(|root| is_within(path, root)) {
            return Ok(());
        }
        let roots: Vec<String> = self
            .roots
            .iter()
            .map(|root| format!("'{}'", format_path(root)))
            .collect();
        Err(resolve_error(&format!(
            "Access denied: Path {raw} outside sandbox [{}]",
            roots.join(", ")
        )))
    }

    /// Resolves `path` without creating parents (convenience for read paths).
    ///
    /// # Errors
    ///
    /// See [`Self::resolve`].
    pub async fn resolve_existing(&self, path: &str) -> Result<PathBuf> {
        self.resolve(path, false).await
    }

    /// Resolves `path` for writing, creating parent directories.
    ///
    /// # Errors
    ///
    /// See [`Self::resolve`].
    pub async fn resolve_for_write(&self, path: &str) -> Result<PathBuf> {
        self.resolve(path, true).await
    }
}

/// Normalizes `.` and `..` segments lexically, without touching the filesystem.
///
/// Unlike [`std::path::Path::canonicalize`] this works for paths that do not exist, which
/// is required for write targets.
#[must_use]
pub fn clean_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let popped = out.pop();
                if !popped {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// Returns `true` when `path` equals `root` or lives beneath it.
#[must_use]
pub fn is_within(path: &Path, root: &Path) -> bool {
    path == root || path.starts_with(root)
}

/// Canonicalizes the longest existing prefix of `path` and re-appends the rest.
///
/// This resolves symbolic links for the part of the path that exists, so a symlink
/// created inside the workspace cannot redirect a write outside it, while still
/// supporting write targets whose leaf does not exist yet.
pub async fn canonicalize_containing(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(canonical) = tokio::fs::canonicalize(&current).await {
            let mut resolved = canonical;
            for segment in suffix.iter().rev() {
                resolved.push(segment);
            }
            return clean_path(&resolved);
        }
        let Some(name) = current.file_name().map(std::ffi::OsStr::to_os_string) else {
            return clean_path(path);
        };
        let Some(parent) = current.parent().map(Path::to_path_buf) else {
            return clean_path(path);
        };
        suffix.push(name);
        current = parent;
    }
}

/// Canonicalizes `path`, falling back to its lexically cleaned form.
fn canonical_or_clean(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| clean_path(path))
}

/// Builds a Python-compatible `resolve_path` filesystem error.
fn resolve_error(reason: &str) -> Error {
    Error::sys_op(fs_execution_message("resolve_path", reason))
}

/// Formats a path as a lossy string.
fn format_path(path: &Path) -> String {
    crate::util::format_path(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn clean_path_collapses_segments() {
        assert_eq!(clean_path(Path::new("a/./b")), Path::new("a/b"));
        assert_eq!(clean_path(Path::new("a/b/../c")), Path::new("a/c"));
        assert_eq!(clean_path(Path::new("a/../..")), Path::new(".."));
        assert_eq!(clean_path(Path::new("")), Path::new("."));
    }

    #[test]
    fn containment_is_not_textual_prefix_matching() {
        assert!(is_within(Path::new("/a/b"), Path::new("/a/b")));
        assert!(is_within(Path::new("/a/b/c"), Path::new("/a/b")));
        assert!(!is_within(Path::new("/a/bc"), Path::new("/a/b")));
        assert!(!is_within(Path::new("/a"), Path::new("/a/b")));
    }

    #[tokio::test]
    async fn resolve_joins_workspace_and_rejects_escapes() {
        let dir = tempfile::tempdir().unwrap();
        let policy = PathPolicy::new(
            &LocalWorkConfig::default()
                .with_workspace(format_path(dir.path()))
                .with_sandbox_roots([format_path(dir.path())]),
        );
        let inside = policy.resolve("a/b.txt", true).await.unwrap();
        assert!(inside.ends_with("b.txt"));
        assert!(is_within(&inside, &canonical_or_clean(dir.path())));

        let error = policy.resolve("../outside.txt", false).await.unwrap_err();
        assert!(
            error.message().contains("Access denied"),
            "{}",
            error.message()
        );
        assert!(error.message().contains("resolve_path"));

        let absolute = format_path(&dir.path().join("../escape"));
        assert!(policy.resolve(&absolute, false).await.is_err());
        assert!(policy.resolve("   ", false).await.is_err());
    }

    #[tokio::test]
    async fn unrestricted_policy_allows_any_path() {
        let dir = tempfile::tempdir().unwrap();
        let policy = PathPolicy::new(
            &LocalWorkConfig::default()
                .with_workspace(format_path(dir.path()))
                .with_sandbox_roots([format_path(dir.path())]),
        );
        assert!(policy.is_restricted());

        let loose = PathPolicy::new(&LocalWorkConfig::unrestricted(format_path(dir.path())));
        assert!(!loose.is_restricted());
        assert!(loose.resolve("../escape", false).await.is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_escape_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();

        let policy = PathPolicy::new(
            &LocalWorkConfig::default()
                .with_workspace(format_path(root.path()))
                .with_sandbox_roots([format_path(root.path())]),
        );
        let error = policy.resolve("link/secret.txt", false).await.unwrap_err();
        assert!(
            error.message().contains("Access denied"),
            "{}",
            error.message()
        );
    }
}
