//! Team and worktree link management.
//!
//! Python's `Workspace` keeps teams and git worktrees side by side without nesting them by
//! creating symbolic links under `.team/` and `.worktree/`. These are **synchronous**
//! filesystem setup helpers (Python parity): they are called while preparing a workspace,
//! not on a hot path, and the harness never holds one across an `.await`.
//!
//! When the platform refuses to create a symbolic link (Windows without developer mode, a
//! filesystem without symlink support), the target is copied instead and a
//! [`LINK_MARKER`] file records the original target. Python's copy fallback silently hides
//! the copied link from its listing; the marker keeps a degraded link visible to the agent.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::workspace::workspace::Workspace;

/// Subdirectory holding team links.
pub const TEAM_LINKS_DIR: &str = ".team";
/// Subdirectory holding worktree links.
pub const WORKTREE_LINKS_DIR: &str = ".worktree";
/// Marker file written into a copy-fallback link, containing the original target path.
pub const LINK_MARKER: &str = ".workspace-link";

/// How a link entry is represented on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// A real symbolic link.
    Symlink,
    /// A recursive copy created when symlinks are unavailable.
    Directory,
}

/// A discovered team or worktree link.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WorkspaceLink {
    /// Entry name inside the link directory.
    pub name: String,
    /// Absolute (canonical when possible) link target.
    #[schemars(with = "String")]
    pub target: PathBuf,
    /// How the entry is represented on disk.
    pub kind: LinkKind,
}

impl Workspace {
    /// Links `team_id` to the team workspace at `target`, returning the link path.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for an unsafe `team_id`,
    /// [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound) when `target`
    /// is missing, and [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp) for
    /// a filesystem failure.
    pub fn link_team(&self, team_id: &str, target: impl AsRef<Path>) -> Result<PathBuf> {
        link(&self.root_path, TEAM_LINKS_DIR, team_id, target.as_ref())
    }

    /// Removes the team link `team_id`, returning `true` when something was removed.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for
    /// an unsafe `team_id` and [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp)
    /// when removal fails.
    pub fn unlink_team(&self, team_id: &str) -> Result<bool> {
        unlink(&self.root_path, TEAM_LINKS_DIR, team_id)
    }

    /// Lists the team links, sorted by name.
    #[must_use]
    pub fn list_team_links(&self) -> Vec<WorkspaceLink> {
        list(&self.root_path, TEAM_LINKS_DIR)
    }

    /// Links `slug` to the worktree at `target`, returning the link path.
    ///
    /// # Errors
    ///
    /// See [`Self::link_team`].
    pub fn link_worktree(&self, slug: &str, target: impl AsRef<Path>) -> Result<PathBuf> {
        link(&self.root_path, WORKTREE_LINKS_DIR, slug, target.as_ref())
    }

    /// Removes the worktree link `slug`, returning `true` when something was removed.
    ///
    /// # Errors
    ///
    /// See [`Self::unlink_team`].
    pub fn unlink_worktree(&self, slug: &str) -> Result<bool> {
        unlink(&self.root_path, WORKTREE_LINKS_DIR, slug)
    }

    /// Lists the worktree links, sorted by name.
    #[must_use]
    pub fn list_worktree_links(&self) -> Vec<WorkspaceLink> {
        list(&self.root_path, WORKTREE_LINKS_DIR)
    }
}

/// Creates a link `name` under `root/subdir` pointing at `target`.
fn link(root: &Path, subdir: &str, name: &str, target: &Path) -> Result<PathBuf> {
    validate_link_name(name)?;
    std::fs::metadata(target)
        .map_err(|error| Error::not_found(format!("link target {}: {error}", target.display())))?;
    let links_dir = root.join(subdir);
    std::fs::create_dir_all(&links_dir).map_err(|error| {
        Error::sys_op(format!(
            "create link directory {}: {error}",
            links_dir.display()
        ))
    })?;
    let entry = links_dir.join(name);
    if std::fs::symlink_metadata(&entry).is_ok() {
        return Ok(entry);
    }
    match create_symlink(target, &entry) {
        Ok(()) => Ok(entry),
        Err(error) if is_copy_fallback(&error) => {
            tracing::warn!(
                link = %entry.display(),
                %error,
                "symbolic link denied; copying the link target instead"
            );
            copy_tree(target, &entry).map_err(|error| {
                Error::sys_op(format!(
                    "copy link target {} to {}: {error}",
                    target.display(),
                    entry.display()
                ))
            })?;
            std::fs::write(entry.join(LINK_MARKER), target.display().to_string()).map_err(
                |error| {
                    Error::sys_op(format!(
                        "write link marker {}: {error}",
                        entry.join(LINK_MARKER).display()
                    ))
                },
            )?;
            Ok(entry)
        }
        Err(error) => Err(Error::sys_op(format!(
            "create link {}: {error}",
            entry.display()
        ))),
    }
}

/// Removes the link `name` under `root/subdir`.
fn unlink(root: &Path, subdir: &str, name: &str) -> Result<bool> {
    validate_link_name(name)?;
    let entry = root.join(subdir).join(name);
    let metadata = match std::fs::symlink_metadata(&entry) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "inspect link {}: {error}",
                entry.display()
            )));
        }
    };
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        remove_symlink(&entry)?;
        Ok(true)
    } else if file_type.is_dir() {
        std::fs::remove_dir_all(&entry)
            .map_err(|error| Error::sys_op(format!("remove link {}: {error}", entry.display())))?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Lists the links under `root/subdir`, sorted by name.
///
/// Unreadable entries are logged and skipped: listing is diagnostics and must never fail.
fn list(root: &Path, subdir: &str) -> Vec<WorkspaceLink> {
    let links_dir = root.join(subdir);
    let Ok(entries) = std::fs::read_dir(&links_dir) else {
        return Vec::new();
    };
    let mut links = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(dir = %links_dir.display(), %error, "skipping unreadable link entry");
                continue;
            }
        };
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            tracing::warn!(entry = %path.display(), "skipping link entry with unreadable type");
            continue;
        };
        if file_type.is_symlink() {
            let target = std::fs::canonicalize(&path)
                .unwrap_or_else(|_| resolve_read_link(&path, &links_dir));
            links.push(WorkspaceLink {
                name,
                target,
                kind: LinkKind::Symlink,
            });
        } else if file_type.is_dir() {
            let marker = path.join(LINK_MARKER);
            if !marker.is_file() {
                continue;
            }
            let target = std::fs::read_to_string(&marker)
                .ok()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
                .map_or_else(|| path.clone(), PathBuf::from);
            links.push(WorkspaceLink {
                name,
                target,
                kind: LinkKind::Directory,
            });
        }
    }
    links.sort_by(|left, right| left.name.cmp(&right.name));
    links
}

/// Rejects a name that is not a single safe path component.
fn validate_link_name(name: &str) -> Result<()> {
    let invalid = name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0');
    if invalid {
        return Err(Error::validation(format!("invalid link name: {name:?}")));
    }
    Ok(())
}

/// Returns `true` for the errors Python handles by falling back to a copy.
fn is_copy_fallback(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
    )
}

/// Creates a directory symlink, when the platform supports one.
fn create_symlink(target: &Path, entry: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, entry)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_dir(target, entry)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, entry);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "symbolic links are not supported on this platform",
        ))
    }
}

/// Removes a symbolic link.
fn remove_symlink(entry: &Path) -> Result<()> {
    #[cfg(unix)]
    let result = std::fs::remove_file(entry);
    #[cfg(windows)]
    let result = std::fs::remove_dir(entry).or_else(|_| std::fs::remove_file(entry));
    #[cfg(not(any(unix, windows)))]
    let result = std::fs::remove_file(entry);
    result.map_err(|error| Error::sys_op(format!("remove link {}: {error}", entry.display())))
}

/// Resolves a link that cannot be canonicalized (for example a dangling link).
fn resolve_read_link(path: &Path, links_dir: &Path) -> PathBuf {
    std::fs::read_link(path).map_or_else(
        |_| path.to_path_buf(),
        |target| {
            if target.is_absolute() {
                target
            } else {
                links_dir.join(target)
            }
        },
    )
}

/// Copies a directory tree; used only by the copy fallback.
fn copy_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    let metadata = std::fs::metadata(source)?;
    if metadata.is_dir() {
        std::fs::create_dir_all(destination)?;
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            copy_tree(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(source, destination)?;
    }
    Ok(())
}
