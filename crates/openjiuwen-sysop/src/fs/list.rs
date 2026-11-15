//! Directory listing for the local filesystem backend.
//!
//! Ports `list_files` / `list_directories` (and the `_walk_path`, `_create_fs_item`,
//! `_sort_items`, `_list_items_internal_sync` helpers) from Python
//! `openjiuwen/core/sys_operation/local/fs_operation.py`.
//!
//! # Depth mapping
//!
//! Python walks with `os.walk` when `recursive` is set and `max_depth` is not `None`.
//! There the root's direct children are at depth `0`, so `max_depth = n` returns `n`
//! levels below the root (`0` = direct children only, `1` = children plus
//! grandchildren). With [`walkdir`] the same window is
//! `.min_depth(1).max_depth(n + 1)`; `max_depth = None` means unlimited depth (Python's
//! `Path.rglob("*")`).
//!
//! # Symlinks
//!
//! Recursion never follows symbolic links (`follow_links(false)`), so a directory symlink
//! cannot send the walk outside the workspace. The link entry itself is still listed, and
//! its type is derived from the link *target* because the per-entry `metadata` call follows
//! symlinks — mirroring Python's `p.is_dir()`.
//!
//! # Blocking work
//!
//! `walkdir` and `std::fs` are blocking APIs and a directory tree can be arbitrarily
//! large, so the entire traversal and the per-entry `metadata` calls run inside
//! [`tokio::task::spawn_blocking`]. Only the already-resolved root is touched on the async
//! task.

use std::path::{Path, PathBuf};

use crate::fs::types::{FileSystemItem, ListDirsParams, ListFilesData, ListFilesParams, SortBy};
use crate::result::{fs_failure, OpResult};
use crate::workspace::PathPolicy;

/// Parameters of one listing run, shared by `list_files` and `list_directories`.
struct ListingSpec<'a> {
    /// Method name reported in failure messages (`"list_files"` / `"list_directories"`).
    execution: &'static str,
    /// Caller-supplied directory path.
    path: &'a str,
    /// Recurse into subdirectories.
    recursive: bool,
    /// Maximum recursion depth, `None` for unlimited.
    max_depth: Option<usize>,
    /// Sort key.
    sort_by: SortBy,
    /// Reverse the sort order.
    sort_descending: bool,
    /// Suffix filter (files only); `None` or an empty list keeps every file.
    file_types: Option<Vec<String>>,
    /// Keep directories instead of files.
    directories_only: bool,
}

/// Owned copy of [`ListingSpec`] handed to the blocking traversal.
struct ListRequest {
    root: PathBuf,
    recursive: bool,
    max_depth: Option<usize>,
    file_types: Option<Vec<String>>,
    directories_only: bool,
    sort_by: SortBy,
    sort_descending: bool,
}

/// Lists files under `params.path`.
///
/// Non-recursive listings return the direct children; `recursive` with `max_depth`
/// returns at most `max_depth` levels below the root (see the module docs). Entries whose
/// metadata cannot be read are dropped rather than failing the call, and recursion never
/// follows directory symlinks.
///
/// Failures (a blank or disallowed path, a missing path, or one that is not a directory)
/// are returned as Python-compatible `OpResult` values with code `199003`.
pub async fn list_files(policy: &PathPolicy, params: &ListFilesParams) -> OpResult<ListFilesData> {
    let spec = ListingSpec {
        execution: "list_files",
        path: &params.path,
        recursive: params.recursive,
        max_depth: params.max_depth,
        sort_by: params.sort_by,
        sort_descending: params.sort_descending,
        file_types: params.file_types.clone(),
        directories_only: false,
    };
    collect_listing(policy, spec)
        .await
        .map(|(items, root)| ListFilesData {
            total_count: items.len(),
            list_items: items,
            root_path: crate::util::format_path(&root),
            recursive: params.recursive,
            max_depth: params.max_depth,
        })
}

/// Lists directories under `params.path`.
///
/// The same traversal rules as [`list_files`] apply; only directory entries are kept.
/// Failures are returned as `OpResult` values with code `199003`.
pub async fn list_directories(
    policy: &PathPolicy,
    params: &ListDirsParams,
) -> OpResult<ListFilesData> {
    let spec = ListingSpec {
        execution: "list_directories",
        path: &params.path,
        recursive: params.recursive,
        max_depth: params.max_depth,
        sort_by: params.sort_by,
        sort_descending: params.sort_descending,
        file_types: None,
        directories_only: true,
    };
    collect_listing(policy, spec)
        .await
        .map(|(items, root)| ListFilesData {
            total_count: items.len(),
            list_items: items,
            root_path: crate::util::format_path(&root),
            recursive: params.recursive,
            max_depth: params.max_depth,
        })
}

/// Builds the Python `FileSystemItem` for `path`.
///
/// `name` is the lossy file name, `path` the lossy absolute path, `size` is `0` for
/// directories, and `file_type` is the Python `Path.suffix` (including the dot) for files
/// and `None` for directories. A name without a suffix maps to `Some("")`, matching
/// Python's `Path("README").suffix == ""`.
#[must_use]
pub fn file_system_item(path: &Path, metadata: &std::fs::Metadata) -> FileSystemItem {
    let name = file_name_lossy(path);
    let is_directory = metadata.is_dir();
    let file_type = if is_directory {
        None
    } else {
        Some(python_suffix(&name))
    };
    FileSystemItem {
        name,
        path: crate::util::format_path(path),
        size: if is_directory { 0 } else { metadata.len() },
        modified_time: crate::util::modified_time_of(metadata, path),
        is_directory,
        file_type,
    }
}

/// Resolves, validates, walks, and sorts a listing, returning the items and the root.
async fn collect_listing(
    policy: &PathPolicy,
    spec: ListingSpec<'_>,
) -> OpResult<(Vec<FileSystemItem>, PathBuf)> {
    let root = match policy.resolve_existing(spec.path).await {
        Ok(root) => root,
        Err(error) => return fs_failure(spec.execution, &error.message()),
    };
    let is_directory = match tokio::fs::metadata(&root).await {
        Ok(metadata) => metadata.is_dir(),
        // A missing root keeps Python's "not a directory" wording; a root that cannot be
        // stat'ed at all (a permission failure, an I/O error) is not the same thing.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            tracing::debug!(path = %crate::util::format_path(&root), %error, "cannot stat the listing root");
            return fs_failure(spec.execution, &error.to_string());
        }
    };
    if !is_directory {
        let root_path = crate::util::format_path(&root);
        return fs_failure(
            spec.execution,
            &format!("Path is not a directory: {root_path}"),
        );
    }

    let request = ListRequest {
        root: root.clone(),
        recursive: spec.recursive,
        max_depth: spec.max_depth,
        file_types: spec.file_types,
        directories_only: spec.directories_only,
        sort_by: spec.sort_by,
        sort_descending: spec.sort_descending,
    };
    match tokio::task::spawn_blocking(move || collect(&request)).await {
        Ok(Ok(items)) => OpResult::success((items, root)),
        Ok(Err(error)) => fs_failure(spec.execution, &error.to_string()),
        Err(error) => fs_failure(spec.execution, &error.to_string()),
    }
}

/// Walks the tree, filters entries, and sorts the result.
///
/// A failure to read the *root* is returned: it would otherwise turn into an empty listing
/// that looks like "the directory has no entries". An unreadable subdirectory keeps
/// Python's `os.walk` behavior (the walk continues past it) and is logged with its path.
fn collect(request: &ListRequest) -> std::io::Result<Vec<FileSystemItem>> {
    let mut items = Vec::new();
    if request.recursive {
        for entry in walk(&request.root, request.max_depth) {
            match entry {
                Ok(entry) => push_entry(&mut items, entry.path(), request),
                Err(error) if is_root_error(&error, &request.root) => {
                    return Err(root_error(&error))
                }
                Err(error) => {
                    let path = error.path().map_or_else(
                        || crate::util::format_path(&request.root),
                        crate::util::format_path,
                    );
                    tracing::warn!(
                        path,
                        error = %error,
                        "skipping an unreadable directory; its entries are missing from this listing"
                    );
                }
            }
        }
    } else {
        for entry in std::fs::read_dir(&request.root)? {
            match entry {
                Ok(entry) => push_entry(&mut items, &entry.path(), request),
                Err(error) => {
                    tracing::debug!(error = %error, "skipping unreadable directory entry");
                }
            }
        }
    }
    sort_items(&mut items, request.sort_by, request.sort_descending);
    Ok(items)
}

/// Whether `error` is the walk failing on its own root rather than on a nested directory.
fn is_root_error(error: &walkdir::Error, root: &Path) -> bool {
    error
        .path()
        .is_some_and(|path| path == root && error.depth() == 0)
}

/// Converts a walk error on the root into the I/O error the listing must report.
fn root_error(error: &walkdir::Error) -> std::io::Error {
    match error.io_error() {
        Some(io) => std::io::Error::new(io.kind(), io.to_string()),
        None => std::io::Error::other(error.to_string()),
    }
}

/// Builds the recursive walker for `root`.
fn walk(root: &Path, max_depth: Option<usize>) -> walkdir::IntoIter {
    let builder = walkdir::WalkDir::new(root).min_depth(1).follow_links(false);
    match max_depth {
        // `max_depth` counts levels below the root starting at 0, while walkdir counts
        // the root itself as depth 0.
        Some(depth) => builder.max_depth(depth.saturating_add(1)).into_iter(),
        None => builder.into_iter(),
    }
}

/// Stats one entry, applies the kind/suffix filters, and appends it.
fn push_entry(items: &mut Vec<FileSystemItem>, path: &Path, request: &ListRequest) {
    let Ok(metadata) = std::fs::metadata(path) else {
        tracing::debug!(path = %path.display(), "skipping entry whose metadata could not be read");
        return;
    };
    let is_directory = metadata.is_dir();
    if request.directories_only {
        if !is_directory {
            return;
        }
    } else if is_directory {
        return;
    }
    if !is_directory && !matches_file_type(path, request.file_types.as_deref()) {
        return;
    }
    items.push(file_system_item(path, &metadata));
}

/// Returns `true` when `path` passes the exact, case-sensitive suffix filter.
fn matches_file_type(path: &Path, file_types: Option<&[String]>) -> bool {
    let Some(types) = file_types.filter(|types| !types.is_empty()) else {
        return true;
    };
    let suffix = python_suffix(&file_name_lossy(path));
    types.iter().any(|entry| entry == &suffix)
}

/// Sorts with Python's stable `list.sort` semantics.
fn sort_items(items: &mut [FileSystemItem], sort_by: SortBy, descending: bool) {
    match sort_by {
        SortBy::Name if descending => items.sort_by(|left, right| right.name.cmp(&left.name)),
        SortBy::Name => items.sort_by(|left, right| left.name.cmp(&right.name)),
        SortBy::ModifiedTime if descending => {
            items.sort_by(|left, right| right.modified_time.cmp(&left.modified_time));
        }
        SortBy::ModifiedTime => {
            items.sort_by(|left, right| left.modified_time.cmp(&right.modified_time));
        }
        SortBy::Size if descending => {
            items.sort_by_key(|item| std::cmp::Reverse(item.size));
        }
        SortBy::Size => items.sort_by_key(|item| item.size),
    }
}

/// Returns the lossy file name of `path`, or an empty string at a filesystem root.
fn file_name_lossy(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Reproduces Python's `Path.suffix`, including the leading dot.
///
/// Leading dots are not themselves suffixes (`""` → `""`, `.bashrc` → `""`), and the
/// final dot wins (`a.tar.gz` → `.gz`). A trailing dot is its own suffix (`a.` → `.`),
/// matching the Python 3.14 implementation.
fn python_suffix(name: &str) -> String {
    let trimmed = name.trim_start_matches('.');
    match trimmed.rfind('.') {
        Some(index) => trimmed[index..].to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use std::time::{Duration, UNIX_EPOCH};

    use crate::config::LocalWorkConfig;

    fn policy(dir: &tempfile::TempDir) -> PathPolicy {
        PathPolicy::new(
            &LocalWorkConfig::default().with_workspace(crate::util::format_path(dir.path())),
        )
    }

    fn build_tree(root: &Path) {
        std::fs::create_dir_all(root.join("sub/deep")).unwrap();
        std::fs::write(root.join("a.txt"), b"aaa").unwrap();
        std::fs::write(root.join("b.md"), b"b").unwrap();
        std::fs::write(root.join("sub/c.txt"), b"cc").unwrap();
        std::fs::write(root.join("sub/deep/d.txt"), b"dddd").unwrap();
    }

    fn files(dir: &tempfile::TempDir) -> PathBuf {
        std::fs::canonicalize(dir.path()).unwrap()
    }

    fn relative_paths(root: &Path, items: &[FileSystemItem]) -> Vec<String> {
        let mut paths: Vec<String> = items
            .iter()
            .map(|item| {
                Path::new(&item.path).strip_prefix(root).map_or_else(
                    |_| item.path.clone(),
                    |path| path.to_string_lossy().into_owned(),
                )
            })
            .collect();
        paths.sort();
        paths
    }

    #[tokio::test]
    async fn non_recursive_lists_direct_children_only() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let result = list_files(&policy, &ListFilesParams::new(".")).await;
        let data = result.into_result().unwrap();
        assert_eq!(data.total_count, 2);
        assert!(!data.recursive);
        assert_eq!(data.root_path, crate::util::format_path(&files(&dir)));
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["a.txt", "b.md"]);
    }

    #[tokio::test]
    async fn recursive_without_max_depth_lists_every_level() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let params = ListFilesParams::new(".").recursive(None);
        let data = list_files(&policy, &params).await.into_result().unwrap();
        assert_eq!(
            relative_paths(&files(&dir), &data.list_items),
            ["a.txt", "b.md", "sub/c.txt", "sub/deep/d.txt"]
        );
        assert!(data.recursive);
        assert_eq!(data.max_depth, None);
    }

    #[tokio::test]
    async fn max_depth_zero_lists_direct_children_only() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let params = ListFilesParams::new(".").recursive(Some(0));
        let data = list_files(&policy, &params).await.into_result().unwrap();
        assert_eq!(
            relative_paths(&files(&dir), &data.list_items),
            ["a.txt", "b.md"]
        );
        assert_eq!(data.max_depth, Some(0));
    }

    #[tokio::test]
    async fn max_depth_one_lists_two_levels() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let params = ListFilesParams::new(".").recursive(Some(1));
        let data = list_files(&policy, &params).await.into_result().unwrap();
        assert_eq!(
            relative_paths(&files(&dir), &data.list_items),
            ["a.txt", "b.md", "sub/c.txt"]
        );
    }

    #[tokio::test]
    async fn file_types_filter_is_exact_and_case_sensitive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), b"a").unwrap();
        std::fs::write(dir.path().join("b.md"), b"b").unwrap();
        std::fs::write(dir.path().join("upper.TXT"), b"c").unwrap();
        std::fs::write(dir.path().join("sub/c.txt"), b"d").unwrap();
        let policy = policy(&dir);
        let mut params = ListFilesParams::new(".").recursive(None);
        params.file_types = Some(vec![".txt".to_string()]);
        let data = list_files(&policy, &params).await.into_result().unwrap();
        assert_eq!(
            relative_paths(&files(&dir), &data.list_items),
            ["a.txt", "sub/c.txt"]
        );
    }

    #[tokio::test]
    async fn empty_file_types_keeps_every_file() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let mut params = ListFilesParams::new(".");
        params.file_types = Some(Vec::new());
        let data = list_files(&policy, &params).await.into_result().unwrap();
        assert_eq!(data.total_count, 2);
    }

    #[test]
    fn file_system_item_reports_python_suffix_and_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"hello").unwrap();
        std::fs::write(dir.path().join("README"), b"readme").unwrap();

        let file_path = dir.path().join("notes.txt");
        let file = file_system_item(&file_path, &std::fs::metadata(&file_path).unwrap());
        assert_eq!(file.name, "notes.txt");
        assert_eq!(file.file_type.as_deref(), Some(".txt"));
        assert_eq!(file.size, 5);
        assert!(!file.is_directory);
        assert!(Path::new(&file.path).is_absolute());

        let readme_path = dir.path().join("README");
        let readme = file_system_item(&readme_path, &std::fs::metadata(&readme_path).unwrap());
        assert_eq!(readme.file_type.as_deref(), Some(""));

        let dir_path = dir.path().join("sub");
        let directory = file_system_item(&dir_path, &std::fs::metadata(&dir_path).unwrap());
        assert_eq!(directory.file_type, None);
        assert_eq!(directory.size, 0);
        assert!(directory.is_directory);
    }

    #[test]
    fn python_suffix_matches_pathlib() {
        assert_eq!(python_suffix("README"), "");
        assert_eq!(python_suffix(".bashrc"), "");
        assert_eq!(python_suffix("a.txt"), ".txt");
        assert_eq!(python_suffix("a.tar.gz"), ".gz");
        assert_eq!(python_suffix("a."), ".");
        assert_eq!(python_suffix(".."), "");
        assert_eq!(python_suffix(""), "");
    }

    #[tokio::test]
    async fn sort_by_name_ascending_and_descending() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("c.rs"), b"ccc").unwrap();
        std::fs::write(dir.path().join("a.rs"), b"a").unwrap();
        std::fs::write(dir.path().join("b.rs"), b"bb").unwrap();
        let policy = policy(&dir);

        let mut params = ListFilesParams::new(".");
        params.sort_by = SortBy::Name;
        let data = list_files(&policy, &params).await.into_result().unwrap();
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["a.rs", "b.rs", "c.rs"]);

        params.sort_descending = true;
        let data = list_files(&policy, &params).await.into_result().unwrap();
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["c.rs", "b.rs", "a.rs"]);
    }

    #[tokio::test]
    async fn sort_by_size_ascending_and_descending() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("medium"), b"1234").unwrap();
        std::fs::write(dir.path().join("small"), b"1").unwrap();
        std::fs::write(dir.path().join("large"), b"123456789").unwrap();
        let policy = policy(&dir);

        let mut params = ListFilesParams::new(".");
        params.sort_by = SortBy::Size;
        let data = list_files(&policy, &params).await.into_result().unwrap();
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["small", "medium", "large"]);

        params.sort_descending = true;
        let data = list_files(&policy, &params).await.into_result().unwrap();
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["large", "medium", "small"]);
    }

    #[tokio::test]
    async fn sort_by_modified_time_ascending_and_descending() {
        let dir = tempfile::tempdir().unwrap();
        for (name, seconds) in [("old", 1_000_u64), ("middle", 2_000), ("new", 3_000)] {
            let path = dir.path().join(name);
            std::fs::write(&path, name.as_bytes()).unwrap();
            let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_modified(UNIX_EPOCH + Duration::from_secs(seconds))
                .unwrap();
        }
        let policy = policy(&dir);

        let mut params = ListFilesParams::new(".");
        params.sort_by = SortBy::ModifiedTime;
        let data = list_files(&policy, &params).await.into_result().unwrap();
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["old", "middle", "new"]);

        params.sort_descending = true;
        let data = list_files(&policy, &params).await.into_result().unwrap();
        let names: Vec<&str> = data
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["new", "middle", "old"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn directory_symlink_is_listed_but_not_descended() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("real/sub")).unwrap();
        std::fs::write(dir.path().join("real/sub/x.txt"), b"x").unwrap();
        std::os::unix::fs::symlink(dir.path().join("real"), dir.path().join("link")).unwrap();
        let policy = policy(&dir);
        let root = files(&dir);

        let dirs = list_directories(&policy, &ListDirsParams::new("."))
            .await
            .into_result()
            .unwrap();
        let names: Vec<&str> = dirs
            .list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["link", "real"]);
        let link = dirs
            .list_items
            .iter()
            .find(|item| item.name == "link")
            .unwrap();
        assert!(link.is_directory, "a symlink to a directory reports as one");

        let recursive_dirs = list_directories(
            &policy,
            &ListDirsParams {
                path: ".".to_string(),
                recursive: true,
                ..ListDirsParams::default()
            },
        )
        .await
        .into_result()
        .unwrap();
        assert_eq!(
            relative_paths(&root, &recursive_dirs.list_items),
            ["link", "real", "real/sub"],
            "the walk must not descend through the symlink"
        );

        let params = ListFilesParams::new(".").recursive(None);
        let files_found = list_files(&policy, &params).await.into_result().unwrap();
        assert_eq!(
            relative_paths(&root, &files_found.list_items),
            ["real/sub/x.txt"]
        );
    }

    #[tokio::test]
    async fn list_directories_returns_only_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("alpha/nested")).unwrap();
        std::fs::create_dir(dir.path().join("beta")).unwrap();
        std::fs::write(dir.path().join("file.txt"), b"file").unwrap();
        let policy = policy(&dir);
        let root = files(&dir);

        let direct = list_directories(&policy, &ListDirsParams::new("."))
            .await
            .into_result()
            .unwrap();
        assert_eq!(relative_paths(&root, &direct.list_items), ["alpha", "beta"]);
        assert!(direct.list_items.iter().all(|item| item.is_directory));

        let recursive = list_directories(
            &policy,
            &ListDirsParams {
                path: ".".to_string(),
                recursive: true,
                ..ListDirsParams::default()
            },
        )
        .await
        .into_result()
        .unwrap();
        assert_eq!(
            relative_paths(&root, &recursive.list_items),
            ["alpha", "alpha/nested", "beta"]
        );
    }

    #[tokio::test]
    async fn file_path_is_not_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("file.txt"), b"file").unwrap();
        let policy = policy(&dir);
        let result = list_files(&policy, &ListFilesParams::new("file.txt")).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("Path is not a directory"),
            "{}",
            result.message
        );
        assert!(result.message.contains("list_files"));
    }

    #[tokio::test]
    async fn missing_path_is_not_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let result = list_directories(&policy, &ListDirsParams::new("missing")).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("Path is not a directory"),
            "{}",
            result.message
        );
        assert!(result.message.contains("list_directories"));
    }

    #[tokio::test]
    async fn blank_path_is_rejected_by_the_policy() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let result = list_files(&policy, &ListFilesParams::new("   ")).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("path must not be empty"),
            "{}",
            result.message
        );
    }
    /// A root that cannot be read is a real failure, not an empty listing.
    #[cfg(unix)]
    #[tokio::test]
    async fn unreadable_root_is_an_error_not_an_empty_listing() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::write(locked.join("a.txt"), b"a").unwrap();
        let policy = policy(&dir);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = list_files(&policy, &ListFilesParams::new("locked").recursive(None)).await;
        let enforced = std::fs::read_dir(&locked).is_err();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        if !enforced {
            eprintln!("skipping: permission bits are not enforced for this user");
            return;
        }
        assert_eq!(result.code, 199_003, "{}", result.message);
        assert!(
            result.message.contains("ermission denied") || result.message.contains("os error 13"),
            "{}",
            result.message
        );
        assert!(
            result.data.is_none(),
            "a failed listing must not carry an (empty) item list"
        );
    }

    /// An unreadable *subdirectory* keeps Python's `os.walk` behavior: the walk continues.
    #[cfg(unix)]
    #[tokio::test]
    async fn unreadable_subdirectory_does_not_hide_the_readable_entries() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("locked")).unwrap();
        std::fs::write(dir.path().join("visible.txt"), b"v").unwrap();
        let policy = policy(&dir);
        let locked = dir.path().join("locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = list_files(&policy, &ListFilesParams::new(".").recursive(None)).await;
        let enforced = std::fs::read_dir(&locked).is_err();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        let data = result.into_result().unwrap();
        assert_eq!(data.total_count, 1, "the readable file is still listed");
        assert_eq!(data.list_items[0].name, "visible.txt");
        if !enforced {
            eprintln!("note: permission bits are not enforced for this user");
        }
    }
}
