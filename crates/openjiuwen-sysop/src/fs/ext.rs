//! Extension filesystem operations: `stat`, `exists`, `mkdir`, `delete`, `copy`, and
//! `move_path`.
//!
//! These six helpers are **Rust additions**: Python's `core/sys_operation` package stops at
//! `read_file`, `write_file`, the upload/download pair, `list_files`/`list_directories`,
//! and `search_files`, so there is no Python counterpart to port. They exist so
//! [`crate::operation::FsExtOperation`] can offer the everyday file management an agent
//! needs without shelling out, and they are deliberately absent from
//! [`crate::SysOperation::tool_specs`] so the generated tool cards stay
//! Python-compatible.
//!
//! # Path resolution
//!
//! Every caller-supplied string goes through [`PathPolicy`] first. Sources and read paths
//! use [`PathPolicy::resolve_existing`]; targets use [`PathPolicy::resolve_for_write`],
//! which creates missing parents while still applying the containment and symlink checks.
//! No function here touches the filesystem with an unresolved argument.
//!
//! # Blocking work
//!
//! [`walkdir`] traversal and entry counting are blocking APIs, so they run in
//! [`tokio::task::spawn_blocking`]; the reads, writes, and copies themselves are awaited
//! `tokio::fs` calls on the async task.
//!
//! # Failures
//!
//! A failure is a Python-compatible [`OpResult`] value with code `199003`, built by
//! [`crate::result::fs_failure`] with the function name in the `execution` slot
//! (`"stat"`, `"exists"`, `"mkdir"`, `"delete"`, `"copy"`, `"move_path"`). Nothing panics,
//! and every reported path is the resolved absolute one.
//!
//! # Cross-device moves
//!
//! [`move_path`] renames first, which is atomic and cheap on one filesystem. A move across
//! filesystems fails with `EXDEV`, which stable [`std::io::ErrorKind`] does not expose (it
//! surfaces as `ErrorKind::Other`), so any rename failure that is *not* one of the named
//! kinds (`NotFound`, `PermissionDenied`, `AlreadyExists`) is retried as a copy followed
//! by deleting the source. A genuine problem the fallback cannot solve resurfaces from the
//! copy and is reported as a normal failure.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::fs::types::{
    CopyData, CopyParams, DeleteData, DeleteParams, FileStat, MkdirData, MkdirParams, MoveData,
    MoveParams,
};
use crate::result::{fs_failure, OpResult};
use crate::util::{format_path, octal_permissions};
use crate::workspace::PathPolicy;

/// Method name reported by `stat` in failure messages.
const STAT: &str = "stat";
/// Method name reported by `exists` in failure messages.
const EXISTS: &str = "exists";
/// Method name reported by `mkdir` in failure messages.
const MKDIR: &str = "mkdir";
/// Method name reported by `delete` in failure messages.
const DELETE: &str = "delete";
/// Method name reported by `copy` in failure messages.
const COPY: &str = "copy";
/// Method name reported by `move_path` in failure messages.
const MOVE: &str = "move_path";

/// Returns metadata for `path`, inspecting a symbolic link rather than its target.
///
/// `is_symlink`, `size`, `modified_time`, and `permissions` come from the link itself,
/// while `is_file`/`is_directory` describe the link *target* (both `false` when the target
/// is missing). Note that [`PathPolicy`] canonicalizes the longest existing prefix, so a
/// link whose target exists resolves to the target and is reported as one; only a dangling
/// link is reported as a link.
///
/// # Errors
///
/// Returns code `199003` when the path is blank or outside the sandbox roots, when it does
/// not exist (`"no such file or directory: <resolved>"`), or when its metadata cannot be
/// read.
pub async fn stat(policy: &PathPolicy, path: &str) -> OpResult<FileStat> {
    let resolved = match resolve_read(policy, path).await {
        Ok(resolved) => resolved,
        Err(reason) => return fs_failure(STAT, &reason),
    };
    let link_metadata = match tokio::fs::symlink_metadata(&resolved).await {
        Ok(metadata) => metadata,
        Err(error) => return io_failure(STAT, &resolved, &error),
    };
    let target = match tokio::fs::metadata(&resolved).await {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        // A dangling link is a normal case (both kinds are `false`); any other error means
        // the link's target could not be inspected, which the caller should be able to see.
        Err(error) => {
            tracing::debug!(
                path = %format_path(&resolved),
                %error,
                "cannot inspect the stat target's metadata; reporting it as a link"
            );
            None
        }
    };
    OpResult::success(FileStat {
        path: format_path(&resolved),
        is_file: target.as_ref().is_some_and(std::fs::Metadata::is_file),
        is_directory: target.as_ref().is_some_and(std::fs::Metadata::is_dir),
        is_symlink: link_metadata.file_type().is_symlink(),
        size: link_metadata.len(),
        modified_time: crate::util::modified_time_of(&link_metadata, &resolved),
        permissions: octal_permissions(&link_metadata),
    })
}

/// Returns whether `path` exists, following symbolic links.
///
/// A missing path is a successful `false`, not an error; a broken symbolic link therefore
/// reports `false` as well. Any *other* metadata failure (a permission failure on a parent
/// directory, for instance) is also reported as `false`, matching Python's `Path.exists()`,
/// and logged so the caller is not left guessing.
///
/// # Errors
///
/// Returns code `199003` when the path is blank or escapes the sandbox roots.
pub async fn exists(policy: &PathPolicy, path: &str) -> OpResult<bool> {
    let resolved = match resolve_read(policy, path).await {
        Ok(resolved) => resolved,
        Err(reason) => return fs_failure(EXISTS, &reason),
    };
    match tokio::fs::metadata(&resolved).await {
        Ok(_) => OpResult::success(true),
        Err(error) if error.kind() == ErrorKind::NotFound => OpResult::success(false),
        Err(error) => {
            tracing::debug!(
                path = %format_path(&resolved),
                %error,
                "cannot read the metadata of an existence check; reporting it as absent"
            );
            OpResult::success(false)
        }
    }
}

/// Creates a directory.
///
/// `recursive` creates missing parents (`create_dir_all`); otherwise a missing parent is a
/// failure. An existing directory is a success with `created = false`, matching `mkdir -p`.
///
/// # Errors
///
/// Returns code `199003` when the path is blank or outside the sandbox roots, when it
/// exists but is not a directory (`"path exists and is not a directory: <resolved>"`), or
/// when the operating system refuses the creation (a missing parent for a non-recursive
/// call carries the I/O error text).
pub async fn mkdir(policy: &PathPolicy, params: &MkdirParams) -> OpResult<MkdirData> {
    let resolved = match resolve_read(policy, &params.path).await {
        Ok(resolved) => resolved,
        Err(reason) => return fs_failure(MKDIR, &reason),
    };
    let path = format_path(&resolved);
    match tokio::fs::metadata(&resolved).await {
        Ok(metadata) if metadata.is_dir() => {
            return OpResult::success(MkdirData {
                path,
                created: false,
            });
        }
        Ok(_) => {
            return fs_failure(
                MKDIR,
                &format!("path exists and is not a directory: {path}"),
            );
        }
        Err(_) => {}
    }
    let created = if params.recursive {
        tokio::fs::create_dir_all(&resolved).await
    } else {
        tokio::fs::create_dir(&resolved).await
    };
    match created {
        Ok(()) => OpResult::success(MkdirData {
            path,
            created: true,
        }),
        Err(error) => fs_failure(MKDIR, &error.to_string()),
    }
}

/// Deletes a file, a symbolic link, or a directory.
///
/// A non-recursive directory delete only works on an empty directory. A recursive delete
/// removes the whole tree and reports the number of entries it contained (`1` for an empty
/// directory, so a successful delete never reports `0`).
///
/// # Errors
///
/// Returns code `199003` when the path is blank, outside the sandbox roots, or missing
/// (`"no such file or directory: <resolved>"`), and when the operating system refuses the
/// removal (a non-empty directory without `recursive` carries the I/O error text).
pub async fn delete(policy: &PathPolicy, params: &DeleteParams) -> OpResult<DeleteData> {
    let resolved = match resolve_read(policy, &params.path).await {
        Ok(resolved) => resolved,
        Err(reason) => return fs_failure(DELETE, &reason),
    };
    let path = format_path(&resolved);
    let metadata = match tokio::fs::symlink_metadata(&resolved).await {
        Ok(metadata) => metadata,
        Err(error) => return io_failure(DELETE, &resolved, &error),
    };
    if !metadata.is_dir() {
        return match tokio::fs::remove_file(&resolved).await {
            Ok(()) => OpResult::success(DeleteData { path, deleted: 1 }),
            Err(error) => fs_failure(DELETE, &error.to_string()),
        };
    }
    if !params.recursive {
        return match tokio::fs::remove_dir(&resolved).await {
            Ok(()) => OpResult::success(DeleteData { path, deleted: 1 }),
            Err(error) => fs_failure(DELETE, &error.to_string()),
        };
    }
    let counted = resolved.clone();
    let counted = match tokio::task::spawn_blocking(move || count_entries(&counted)).await {
        Ok(count) => count,
        Err(error) => return fs_failure(DELETE, &error.to_string()),
    };
    match tokio::fs::remove_dir_all(&resolved).await {
        Ok(()) => OpResult::success(DeleteData {
            path,
            // The directory itself is one removed entry, matching the non-recursive
            // branch that reports `1` for an empty directory.
            deleted: counted.max(1),
        }),
        Err(error) => fs_failure(DELETE, &error.to_string()),
    }
}

/// Copies `params.source_path` to `params.target_path`.
///
/// A directory requires `recursive`, in which case the whole tree is recreated below the
/// target. Symbolic links inside a copied tree are skipped rather than followed, so a link
/// cannot pull content from outside the tree. An existing target is preserved unless
/// `overwrite` is set. `size` is the number of bytes copied (`0` for an empty tree).
///
/// # Errors
///
/// Returns code `199003` when either path is blank or outside the sandbox roots, when the
/// source is missing, when the source is a directory and `recursive` is `false`
/// (`"directory copy requires recursive: <source>"`), when the target is the source or
/// lives inside it, when the target exists without `overwrite`, or when the operating
/// system refuses the copy.
pub async fn copy(policy: &PathPolicy, params: &CopyParams) -> OpResult<CopyData> {
    let source = match resolve_read(policy, &params.source_path).await {
        Ok(source) => source,
        Err(reason) => return fs_failure(COPY, &reason),
    };
    let metadata = match tokio::fs::metadata(&source).await {
        Ok(metadata) => metadata,
        Err(error) => return io_failure(COPY, &source, &error),
    };
    let target = match resolve_write(policy, &params.target_path).await {
        Ok(target) => target,
        Err(reason) => return fs_failure(COPY, &reason),
    };
    let is_directory = metadata.is_dir();
    if is_directory && !params.recursive {
        return fs_failure(
            COPY,
            &format!(
                "directory copy requires recursive: {}",
                format_path(&source)
            ),
        );
    }
    if let Some(reason) = into_itself_reason(&source, &target, is_directory, "copy") {
        return fs_failure(COPY, &reason);
    }
    match target_exists(&target).await {
        Ok(false) => {}
        Ok(true) => {
            if !params.overwrite {
                return fs_failure(COPY, &format!("target exists: {}", format_path(&target)));
            }
            if let Err(error) = remove_existing(&target).await {
                return fs_failure(COPY, &error.to_string());
            }
        }
        Err(error) => return io_failure(COPY, &target, &error),
    }
    let copied = if is_directory {
        copy_tree(&source, &target).await
    } else {
        tokio::fs::copy(&source, &target).await
    };
    match copied {
        Ok(size) => OpResult::success(CopyData {
            source_path: format_path(&source),
            target_path: format_path(&target),
            size,
        }),
        Err(error) => fs_failure(COPY, &error.to_string()),
    }
}

/// Moves or renames `params.source_path` to `params.target_path`.
///
/// The rename is attempted first; when it fails for a reason that may be a cross-device
/// move, the tree is copied and the source deleted (see the module docs). An existing
/// target is preserved unless `overwrite` is set.
///
/// # Errors
///
/// Returns code `199003` when either path is blank or outside the sandbox roots, when the
/// source is missing (`"no such file or directory: <resolved>"`), when the target is the
/// source or lives inside it, when the target exists without `overwrite`, or when the
/// operating system refuses the move.
pub async fn move_path(policy: &PathPolicy, params: &MoveParams) -> OpResult<MoveData> {
    let source = match resolve_read(policy, &params.source_path).await {
        Ok(source) => source,
        Err(reason) => return fs_failure(MOVE, &reason),
    };
    let metadata = match tokio::fs::symlink_metadata(&source).await {
        Ok(metadata) => metadata,
        Err(error) => return io_failure(MOVE, &source, &error),
    };
    let target = match resolve_write(policy, &params.target_path).await {
        Ok(target) => target,
        Err(reason) => return fs_failure(MOVE, &reason),
    };
    let is_directory = metadata.is_dir();
    if let Some(reason) = into_itself_reason(&source, &target, is_directory, "move") {
        return fs_failure(MOVE, &reason);
    }
    match target_exists(&target).await {
        Ok(false) => {}
        Ok(true) => {
            if !params.overwrite {
                return fs_failure(MOVE, &format!("target exists: {}", format_path(&target)));
            }
            if let Err(error) = remove_existing(&target).await {
                return fs_failure(MOVE, &error.to_string());
            }
        }
        Err(error) => return io_failure(MOVE, &target, &error),
    }
    match tokio::fs::rename(&source, &target).await {
        Ok(()) => {}
        Err(error) if should_fallback(&error) => {
            if let Err(error) = relocate(&source, &target, is_directory).await {
                return fs_failure(MOVE, &error.to_string());
            }
        }
        Err(error) => return fs_failure(MOVE, &error.to_string()),
    }
    OpResult::success(MoveData {
        source_path: format_path(&source),
        target_path: format_path(&target),
    })
}

/// Resolves `path` for reading, mapping a policy rejection to its message.
async fn resolve_read(policy: &PathPolicy, path: &str) -> Result<PathBuf, String> {
    policy
        .resolve_existing(path)
        .await
        .map_err(|error| error.message().to_string())
}

/// Resolves `path` for writing (creating missing parents), mapping a rejection to its
/// message.
async fn resolve_write(policy: &PathPolicy, path: &str) -> Result<PathBuf, String> {
    policy
        .resolve_for_write(path)
        .await
        .map_err(|error| error.message().to_string())
}

/// Builds the failure for an I/O error on `path`, using Python's missing-path wording.
fn io_failure<T>(execution: &str, path: &Path, error: &std::io::Error) -> OpResult<T> {
    if error.kind() == ErrorKind::NotFound {
        return fs_failure(
            execution,
            &format!("no such file or directory: {}", format_path(path)),
        );
    }
    fs_failure(execution, &error.to_string())
}

/// Returns whether `path` exists, distinguishing "absent" from "cannot be inspected".
///
/// # Errors
///
/// Returns the OS error when the path's existence cannot be determined at all (a permission
/// failure or a symlink loop), so a rejected overwrite is not silently skipped.
async fn target_exists(path: &Path) -> std::io::Result<bool> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

/// Returns the reason a target is invalid relative to its source, if any.
///
/// A directory can never be copied or moved into itself (which includes the directory
/// itself), and a file target equal to the source is rejected before `tokio::fs::copy`
/// could truncate the file it is reading.
fn into_itself_reason(
    source: &Path,
    target: &Path,
    is_directory: bool,
    verb: &str,
) -> Option<String> {
    let target_text = format_path(target);
    if is_directory && target.starts_with(source) {
        return Some(format!(
            "cannot {verb} a directory into itself: {target_text}"
        ));
    }
    if target == source {
        return Some(format!(
            "source and target are the same path: {target_text}"
        ));
    }
    None
}

/// Removes an existing target, using the removal matching its kind.
async fn remove_existing(path: &Path) -> std::io::Result<()> {
    let metadata = tokio::fs::symlink_metadata(path).await?;
    if metadata.is_dir() {
        tokio::fs::remove_dir_all(path).await
    } else {
        tokio::fs::remove_file(path).await
    }
}

/// Returns `true` when a failed rename should be retried as a copy and delete.
///
/// `EXDEV` (cross-device link) is the case this exists for; stable Rust has no
/// `ErrorKind::CrossesDevices`, so it arrives as `ErrorKind::Other`. Every error the
/// standard kinds do name is a genuine failure and is reported as-is.
fn should_fallback(error: &std::io::Error) -> bool {
    !matches!(
        error.kind(),
        ErrorKind::NotFound | ErrorKind::PermissionDenied | ErrorKind::AlreadyExists
    )
}

/// Copies `source` to `target` and deletes the source, for a move a rename cannot perform.
async fn relocate(source: &Path, target: &Path, is_directory: bool) -> std::io::Result<()> {
    if is_directory {
        copy_tree(source, target).await?;
        tokio::fs::remove_dir_all(source).await
    } else {
        tokio::fs::copy(source, target).await?;
        tokio::fs::remove_file(source).await
    }
}

/// A directory tree split into the entries a copy has to recreate.
struct TreePlan {
    /// Subdirectories, relative to the tree root.
    directories: Vec<PathBuf>,
    /// Regular files, relative to the tree root.
    files: Vec<PathBuf>,
}

/// Recursively copies the tree at `source` into `target`, returning the bytes copied.
///
/// The walk runs on a blocking thread; the directory creation and file copies are awaited.
async fn copy_tree(source: &Path, target: &Path) -> std::io::Result<u64> {
    let root = source.to_path_buf();
    let plan = tokio::task::spawn_blocking(move || collect_tree(&root))
        .await
        .map_err(|error| std::io::Error::other(error.to_string()))??;
    tokio::fs::create_dir_all(target).await?;
    for relative in &plan.directories {
        tokio::fs::create_dir_all(target.join(relative)).await?;
    }
    let mut size = 0_u64;
    for relative in &plan.files {
        let destination = target.join(relative);
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        size += tokio::fs::copy(source.join(relative), &destination).await?;
    }
    Ok(size)
}

/// Walks `root` and collects the directories and regular files below it.
///
/// Symbolic links are neither followed nor copied, so a link cannot redirect the copy
/// outside the tree.
fn collect_tree(root: &Path) -> std::io::Result<TreePlan> {
    let mut plan = TreePlan {
        directories: Vec::new(),
        files: Vec::new(),
    };
    for entry in walkdir::WalkDir::new(root).min_depth(1).follow_links(false) {
        let entry = entry.map_err(std::io::Error::from)?;
        let Ok(relative) = entry.path().strip_prefix(root) else {
            continue;
        };
        let file_type = entry.file_type();
        if file_type.is_symlink() {
            tracing::debug!(
                path = %entry.path().display(),
                "skipping symbolic link while copying a directory tree"
            );
        } else if file_type.is_dir() {
            plan.directories.push(relative.to_path_buf());
        } else if file_type.is_file() {
            plan.files.push(relative.to_path_buf());
        }
    }
    Ok(plan)
}

/// Counts the entries below `root` (files, directories, and links).
///
/// The count is informational, so unreadable entries are logged and skipped instead of
/// failing the delete that follows.
fn count_entries(root: &Path) -> usize {
    let mut count = 0_usize;
    for entry in walkdir::WalkDir::new(root).min_depth(1).follow_links(false) {
        match entry {
            Ok(_) => count += 1,
            Err(error) => {
                tracing::debug!(error = %error, "skipping unreadable directory entry");
            }
        }
    }
    count
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::config::LocalWorkConfig;

    /// Builds a policy rooted at `dir` and accepting every path below it.
    fn policy(dir: &tempfile::TempDir) -> PathPolicy {
        PathPolicy::new(
            &LocalWorkConfig::default().with_workspace(crate::util::format_path(dir.path())),
        )
    }

    /// The canonical form of the temporary directory, which is the policy workspace.
    fn root(dir: &tempfile::TempDir) -> PathBuf {
        std::fs::canonicalize(dir.path()).unwrap()
    }

    #[tokio::test]
    async fn stat_describes_files_and_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        let policy = policy(&dir);

        let file = stat(&policy, "a.txt").await.into_result().unwrap();
        assert!(file.is_file);
        assert!(!file.is_directory);
        assert!(!file.is_symlink);
        assert_eq!(file.size, 5);
        assert_eq!(file.path, format_path(&root(&dir).join("a.txt")));
        assert_eq!(file.modified_time.len(), 26, "{}", file.modified_time);
        #[cfg(unix)]
        assert!(file.permissions.is_some());

        let directory = stat(&policy, "sub").await.into_result().unwrap();
        assert!(directory.is_directory);
        assert!(!directory.is_file);
        assert!(!directory.is_symlink);
        assert_eq!(directory.path, format_path(&root(&dir).join("sub")));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stat_reports_a_dangling_symlink_as_one() {
        let dir = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(dir.path().join("gone.txt"), dir.path().join("broken")).unwrap();
        let policy = policy(&dir);

        let broken = stat(&policy, "broken").await.into_result().unwrap();
        assert!(broken.is_symlink);
        assert!(
            !broken.is_file,
            "a broken link is neither file nor directory"
        );
        assert!(!broken.is_directory);
        let link_metadata = std::fs::symlink_metadata(dir.path().join("broken")).unwrap();
        assert_eq!(broken.size, link_metadata.len(), "size comes from the link");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stat_resolves_a_live_symlink_to_its_target() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("target.txt"), b"hello").unwrap();
        std::os::unix::fs::symlink(dir.path().join("target.txt"), dir.path().join("link")).unwrap();
        let policy = policy(&dir);

        // The policy canonicalizes the longest existing prefix, so a link whose target
        // exists reports the target: the link itself is never inspected.
        let link = stat(&policy, "link").await.into_result().unwrap();
        assert!(!link.is_symlink);
        assert!(link.is_file);
        assert_eq!(link.size, 5);
        assert_eq!(link.path, format_path(&root(&dir).join("target.txt")));
    }

    #[tokio::test]
    async fn stat_missing_path_is_a_failure() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let result = stat(&policy, "missing.txt").await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("no such file or directory"),
            "{}",
            result.message
        );
        assert!(
            result.message.contains("execution: stat"),
            "{}",
            result.message
        );
    }

    #[tokio::test]
    async fn exists_reports_presence_and_rejects_blank_paths() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        let policy = policy(&dir);

        assert!(exists(&policy, "a.txt").await.into_result().unwrap());
        assert!(!exists(&policy, "missing.txt").await.into_result().unwrap());

        let blank = exists(&policy, "   ").await;
        assert_eq!(blank.code, 199_003);
        assert!(
            blank.message.contains("path must not be empty"),
            "{}",
            blank.message
        );
        assert!(blank.message.contains("execution: exists"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn exists_is_false_for_a_broken_symlink() {
        let dir = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(dir.path().join("gone.txt"), dir.path().join("broken")).unwrap();
        let policy = policy(&dir);
        assert!(!exists(&policy, "broken").await.into_result().unwrap());
    }

    #[tokio::test]
    async fn mkdir_recursive_creates_missing_parents() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let data = mkdir(&policy, &MkdirParams::new("a/b/c"))
            .await
            .into_result()
            .unwrap();
        assert!(data.created);
        assert_eq!(data.path, format_path(&root(&dir).join("a/b/c")));
        assert!(root(&dir).join("a/b/c").is_dir());
    }

    #[tokio::test]
    async fn mkdir_non_recursive_requires_an_existing_parent() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let nested = MkdirParams {
            path: "a/b".to_string(),
            recursive: false,
        };
        let result = mkdir(&policy, &nested).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("execution: mkdir"),
            "{}",
            result.message
        );
        assert!(
            !root(&dir).join("a").exists(),
            "the missing parent must not be created"
        );

        let flat = MkdirParams {
            path: "alpha".to_string(),
            recursive: false,
        };
        assert!(mkdir(&policy, &flat).await.into_result().unwrap().created);
    }

    #[tokio::test]
    async fn mkdir_reports_existing_directory_and_file_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("file.txt"), b"x").unwrap();
        let policy = policy(&dir);

        let existing = mkdir(&policy, &MkdirParams::new("sub"))
            .await
            .into_result()
            .unwrap();
        assert!(!existing.created);

        let conflict = mkdir(&policy, &MkdirParams::new("file.txt")).await;
        assert_eq!(conflict.code, 199_003);
        assert!(
            conflict
                .message
                .contains("path exists and is not a directory"),
            "{}",
            conflict.message
        );
    }

    #[tokio::test]
    async fn delete_removes_files_and_empty_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        std::fs::create_dir(dir.path().join("empty")).unwrap();
        let policy = policy(&dir);

        let file = delete(&policy, &DeleteParams::new("a.txt"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(file.deleted, 1);
        assert_eq!(file.path, format_path(&root(&dir).join("a.txt")));
        assert!(!root(&dir).join("a.txt").exists());

        let empty = delete(&policy, &DeleteParams::new("empty"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(empty.deleted, 1);
        assert!(!root(&dir).join("empty").exists());

        let missing = delete(&policy, &DeleteParams::new("missing")).await;
        assert_eq!(missing.code, 199_003);
        assert!(
            missing.message.contains("no such file or directory"),
            "{}",
            missing.message
        );
    }

    #[tokio::test]
    async fn delete_non_empty_directory_requires_recursive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/child.txt"), b"x").unwrap();
        let policy = policy(&dir);

        let result = delete(&policy, &DeleteParams::new("sub")).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("execution: delete"),
            "{}",
            result.message
        );
        assert!(
            root(&dir).join("sub/child.txt").exists(),
            "the tree must be left intact"
        );
    }

    #[tokio::test]
    async fn delete_recursive_removes_the_whole_tree() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("tree/sub")).unwrap();
        std::fs::write(dir.path().join("tree/a.txt"), b"a").unwrap();
        std::fs::write(dir.path().join("tree/b.txt"), b"b").unwrap();
        std::fs::write(dir.path().join("tree/sub/c.txt"), b"c").unwrap();
        let policy = policy(&dir);

        let params = DeleteParams {
            path: "tree".to_string(),
            recursive: true,
        };
        let data = delete(&policy, &params).await.into_result().unwrap();
        assert_eq!(data.deleted, 4, "three files plus one subdirectory");
        assert!(!root(&dir).join("tree").exists());
    }

    #[tokio::test]
    async fn copy_copies_a_file_and_requires_overwrite_for_an_existing_target() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"old").unwrap();
        let policy = policy(&dir);

        let data = copy(&policy, &CopyParams::new("a.txt", "c.txt"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(data.size, 5);
        assert_eq!(data.source_path, format_path(&root(&dir).join("a.txt")));
        assert_eq!(data.target_path, format_path(&root(&dir).join("c.txt")));
        assert_eq!(
            std::fs::read(root(&dir).join("c.txt")).unwrap(),
            b"hello".to_vec()
        );

        let refused = copy(&policy, &CopyParams::new("a.txt", "b.txt")).await;
        assert_eq!(refused.code, 199_003);
        assert!(
            refused.message.contains("target exists"),
            "{}",
            refused.message
        );
        assert_eq!(
            std::fs::read(root(&dir).join("b.txt")).unwrap(),
            b"old".to_vec()
        );

        let missing = copy(&policy, &CopyParams::new("missing.txt", "d.txt")).await;
        assert_eq!(missing.code, 199_003);
        assert!(
            missing.message.contains("no such file or directory"),
            "{}",
            missing.message
        );
    }

    #[tokio::test]
    async fn copy_copies_a_directory_tree_recursively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        std::fs::create_dir(dir.path().join("src/empty")).unwrap();
        std::fs::write(dir.path().join("src/a.txt"), b"aaa").unwrap();
        std::fs::write(dir.path().join("src/nested/b.txt"), b"bb").unwrap();
        let policy = policy(&dir);

        let params = CopyParams {
            source_path: "src".to_string(),
            target_path: "dst".to_string(),
            overwrite: false,
            recursive: true,
        };
        let data = copy(&policy, &params).await.into_result().unwrap();
        assert_eq!(data.size, 5);
        assert_eq!(
            std::fs::read(root(&dir).join("dst/a.txt")).unwrap(),
            b"aaa".to_vec()
        );
        assert_eq!(
            std::fs::read(root(&dir).join("dst/nested/b.txt")).unwrap(),
            b"bb".to_vec()
        );
        assert!(root(&dir).join("dst/empty").is_dir());

        let empty = CopyParams {
            source_path: "src/empty".to_string(),
            target_path: "copy-of-empty".to_string(),
            overwrite: false,
            recursive: true,
        };
        let data = copy(&policy, &empty).await.into_result().unwrap();
        assert_eq!(data.size, 0, "an empty tree copies zero bytes");
        assert!(root(&dir).join("copy-of-empty").is_dir());

        let refused = copy(&policy, &CopyParams::new("src", "other")).await;
        assert_eq!(refused.code, 199_003);
        assert!(
            refused
                .message
                .contains("directory copy requires recursive"),
            "{}",
            refused.message
        );
    }

    #[tokio::test]
    async fn copy_overwrite_replaces_the_target() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"new content").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"old").unwrap();
        let policy = policy(&dir);

        let params = CopyParams {
            source_path: "a.txt".to_string(),
            target_path: "b.txt".to_string(),
            overwrite: true,
            recursive: false,
        };
        let data = copy(&policy, &params).await.into_result().unwrap();
        assert_eq!(data.size, 11);
        assert_eq!(
            std::fs::read(root(&dir).join("b.txt")).unwrap(),
            b"new content".to_vec()
        );
    }

    #[tokio::test]
    async fn copy_refuses_to_copy_a_directory_into_itself() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/inner")).unwrap();
        let policy = policy(&dir);

        let params = CopyParams {
            source_path: "src".to_string(),
            target_path: "src/inner/copy".to_string(),
            overwrite: false,
            recursive: true,
        };
        let result = copy(&policy, &params).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result
                .message
                .contains("cannot copy a directory into itself"),
            "{}",
            result.message
        );
        assert!(!root(&dir).join("src/inner/copy").exists());
    }

    #[tokio::test]
    async fn move_path_renames_and_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"old").unwrap();
        let policy = policy(&dir);

        let data = move_path(&policy, &MoveParams::new("a.txt", "c.txt"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(data.source_path, format_path(&root(&dir).join("a.txt")));
        assert_eq!(data.target_path, format_path(&root(&dir).join("c.txt")));
        assert!(root(&dir).join("c.txt").exists());
        assert!(!root(&dir).join("a.txt").exists());

        let refused = move_path(&policy, &MoveParams::new("c.txt", "b.txt")).await;
        assert_eq!(refused.code, 199_003);
        assert!(
            refused.message.contains("target exists"),
            "{}",
            refused.message
        );
        assert_eq!(
            std::fs::read(root(&dir).join("b.txt")).unwrap(),
            b"old".to_vec()
        );

        let params = MoveParams {
            source_path: "c.txt".to_string(),
            target_path: "b.txt".to_string(),
            overwrite: true,
        };
        move_path(&policy, &params).await.into_result().unwrap();
        assert_eq!(
            std::fs::read(root(&dir).join("b.txt")).unwrap(),
            b"hello".to_vec()
        );
        assert!(!root(&dir).join("c.txt").exists());
    }

    #[tokio::test]
    async fn move_path_moves_a_directory_tree_and_rejects_missing_sources() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        std::fs::write(dir.path().join("src/nested/a.txt"), b"a").unwrap();
        let policy = policy(&dir);

        let data = move_path(&policy, &MoveParams::new("src", "dst"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(data.target_path, format_path(&root(&dir).join("dst")));
        assert!(root(&dir).join("dst/nested/a.txt").exists());
        assert!(!root(&dir).join("src").exists());

        let missing = move_path(&policy, &MoveParams::new("missing", "x")).await;
        assert_eq!(missing.code, 199_003);
        assert!(
            missing.message.contains("no such file or directory"),
            "{}",
            missing.message
        );
        assert!(missing.message.contains("execution: move_path"));

        let into_itself = move_path(&policy, &MoveParams::new("dst", "dst/nested/inside")).await;
        assert_eq!(into_itself.code, 199_003);
        assert!(
            into_itself
                .message
                .contains("cannot move a directory into itself"),
            "{}",
            into_itself.message
        );
        assert!(root(&dir).join("dst/nested/a.txt").exists());
    }
    /// A target whose existence cannot be determined is an error, not "absent".
    #[cfg(unix)]
    #[tokio::test]
    async fn copy_reports_an_unreadable_target_instead_of_overwriting_it() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::write(locked.join("b.txt"), b"old").unwrap();
        let policy = policy(&dir);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = copy(&policy, &CopyParams::new("a.txt", "locked/b.txt")).await;
        let enforced = std::fs::metadata(locked.join("b.txt")).is_err();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        if !enforced {
            eprintln!("skipping: permission bits are not enforced for this user");
            return;
        }
        assert_eq!(result.code, 199_003, "{}", result.message);
        assert!(
            !result.message.contains("target exists"),
            "an inspection failure must not be reported as an existing target: {}",
            result.message
        );
        assert_eq!(
            std::fs::read(root(&dir).join("locked/b.txt")).unwrap(),
            b"old".to_vec(),
            "the unreadable target must be left untouched"
        );
    }
}
