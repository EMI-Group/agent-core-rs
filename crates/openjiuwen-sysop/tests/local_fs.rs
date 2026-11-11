//! Ports `tests/unit_tests/core/sys_operation/local/test_fs_operation.py`
//! (read/write/list/search/transfer round-trips).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::Arc;

use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::fs::local_fs;
use openjiuwen_sysop::fs::types::{
    DownloadFileParams, FileContent, FileSystemItem, ListDirsParams, ListFilesParams,
    ReadFileParams, ReadMode, SearchFilesParams, SortBy, UploadFileParams, WriteFileParams,
};
use openjiuwen_sysop::operation::FsOperation;
use openjiuwen_sysop::status::SysOpStatus;
use tempfile::TempDir;

/// The failure code every local filesystem error carries (Python `199003`).
const FS_ERROR: i32 = SysOpStatus::FsExecutionError.code();

/// Builds the local filesystem backend rooted at `dir`.
fn backend(dir: &TempDir) -> Arc<dyn FsOperation> {
    local_fs(LocalWorkConfig::unrestricted(path_string(dir))).expect("local filesystem backend")
}

/// The absolute path of `dir` rendered the way the backend renders paths.
fn path_string(dir: &TempDir) -> String {
    dir.path().to_string_lossy().to_string()
}

/// Writes `files` (workspace-relative paths) below `root`, creating parent directories.
fn write_tree(root: &Path, files: &[(&str, &str)]) {
    for (relative, content) in files {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create fixture directory");
        }
        std::fs::write(&path, content).expect("write fixture file");
    }
}

/// The names of `items`, sorted so set-like assertions are order independent.
fn sorted_names(items: &[FileSystemItem]) -> Vec<String> {
    let mut names: Vec<String> = items.iter().map(|item| item.name.clone()).collect();
    names.sort();
    names
}

#[tokio::test]
async fn write_then_read_round_trip() {
    let dir = TempDir::new().unwrap();
    let fs = backend(&dir);

    let written = fs
        .write_file(WriteFileParams::raw("hello.txt", "hello"))
        .await;
    assert_eq!(written.code, 0, "write failed: {}", written.message);
    assert_eq!(
        written.data.as_ref().map(|data| data.size),
        Some(5),
        "the report size is the number of bytes written"
    );

    let read = fs.read_file(ReadFileParams::text("hello.txt")).await;
    assert_eq!(read.code, 0, "read failed: {}", read.message);
    let data = read.data.expect("read payload");
    assert_eq!(data.content, FileContent::Text("hello".to_string()));
    assert_eq!(data.mode, ReadMode::Text);
    assert_eq!(data.total_bytes, 5);
    assert!(!data.truncated, "a full read is never truncated");
}

#[tokio::test]
async fn write_defaults_to_a_leading_newline() {
    let dir = TempDir::new().unwrap();
    let fs = backend(&dir);

    let written = fs.write_file(WriteFileParams::new("a.txt", "hello")).await;
    assert_eq!(written.code, 0, "write failed: {}", written.message);

    let read = fs.read_file(ReadFileParams::text("a.txt")).await;
    assert_eq!(read.code, 0, "read failed: {}", read.message);
    let data = read.data.expect("read payload");
    assert_eq!(
        data.content,
        FileContent::Text("\nhello".to_string()),
        "Python's prepend_newline default is true"
    );
    assert_eq!(data.total_bytes, 6);
}

#[tokio::test]
async fn append_does_not_insert_newline_and_creates_missing_file() {
    let dir = TempDir::new().unwrap();
    let fs = backend(&dir);

    let written = fs.write_file(WriteFileParams::raw("a.txt", "a")).await;
    assert_eq!(written.code, 0, "write failed: {}", written.message);

    let appended = fs
        .write_file(WriteFileParams::raw("a.txt", "b").with_append())
        .await;
    assert_eq!(appended.code, 0, "append failed: {}", appended.message);

    let read = fs.read_file(ReadFileParams::text("a.txt")).await;
    let data = read.data.expect("read payload");
    assert_eq!(
        data.content,
        FileContent::Text("ab".to_string()),
        "append must not insert a newline"
    );

    let created = fs
        .write_file(WriteFileParams::raw("missing.txt", "x").with_append())
        .await;
    assert_eq!(created.code, 0, "append failed: {}", created.message);
    let read = fs.read_file(ReadFileParams::text("missing.txt")).await;
    let data = read.data.expect("read payload");
    assert_eq!(
        data.content,
        FileContent::Text("x".to_string()),
        "appending to a missing file creates it"
    );
}

#[tokio::test]
async fn write_with_create_if_not_exist_false_rejects_missing_file() {
    let dir = TempDir::new().unwrap();
    let fs = backend(&dir);

    let mut params = WriteFileParams::raw("missing.txt", "x");
    params.create_if_not_exist = false;
    let result = fs.write_file(params).await;

    assert_eq!(result.code, FS_ERROR);
    assert!(
        result.message.contains("File does not exist"),
        "unexpected message: {}",
        result.message
    );
    assert!(result.data.is_none());
}

#[tokio::test]
async fn list_files_non_recursive_returns_direct_children() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[("a.txt", "a"), ("sub/b.txt", "b"), ("sub/deep/c.txt", "c")],
    );
    let fs = backend(&dir);

    let result = fs.list_files(ListFilesParams::new(path_string(&dir))).await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    assert_eq!(data.total_count, 1);
    assert_eq!(sorted_names(&data.list_items), vec!["a.txt"]);
}

#[tokio::test]
async fn list_files_recursive_returns_all_depths() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[("a.txt", "a"), ("sub/b.txt", "b"), ("sub/deep/c.txt", "c")],
    );
    let fs = backend(&dir);

    let params = ListFilesParams::new(path_string(&dir)).recursive(None);
    let result = fs.list_files(params).await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    assert_eq!(data.total_count, 3);
    assert_eq!(
        sorted_names(&data.list_items),
        vec!["a.txt", "b.txt", "c.txt"]
    );
    assert!(data.recursive);
}

#[tokio::test]
async fn list_files_max_depth_zero_is_direct_children_only() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[("a.txt", "a"), ("sub/b.txt", "b"), ("sub/deep/c.txt", "c")],
    );
    let fs = backend(&dir);

    let params = ListFilesParams::new(path_string(&dir)).recursive(Some(0));
    let result = fs.list_files(params).await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    assert_eq!(data.total_count, 1);
    assert_eq!(sorted_names(&data.list_items), vec!["a.txt"]);
    assert_eq!(data.max_depth, Some(0));
}

#[tokio::test]
async fn list_files_suffix_filter_is_exact() {
    let dir = TempDir::new().unwrap();
    write_tree(dir.path(), &[("a.txt", "a"), ("b.md", "b"), ("c.TXT", "c")]);
    let fs = backend(&dir);

    let mut params = ListFilesParams::new(path_string(&dir));
    params.file_types = Some(vec![".txt".to_string()]);
    let result = fs.list_files(params).await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    assert_eq!(
        sorted_names(&data.list_items),
        vec!["a.txt"],
        "the suffix filter is exact and case-sensitive"
    );
    assert_eq!(data.list_items[0].file_type.as_deref(), Some(".txt"));
}

#[tokio::test]
async fn list_files_sort_by_name_descending() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c")],
    );
    let fs = backend(&dir);

    let mut params = ListFilesParams::new(path_string(&dir));
    params.sort_by = SortBy::Name;
    params.sort_descending = true;
    let result = fs.list_files(params).await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    let names: Vec<String> = data
        .list_items
        .iter()
        .map(|item| item.name.clone())
        .collect();
    assert_eq!(names, vec!["c.txt", "b.txt", "a.txt"]);
}

#[tokio::test]
async fn list_files_sort_by_size() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[("a.txt", "1"), ("b.txt", "22"), ("c.txt", "333")],
    );
    let fs = backend(&dir);

    let mut params = ListFilesParams::new(path_string(&dir));
    params.sort_by = SortBy::Size;
    let result = fs.list_files(params).await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    let names: Vec<String> = data
        .list_items
        .iter()
        .map(|item| item.name.clone())
        .collect();
    let sizes: Vec<u64> = data.list_items.iter().map(|item| item.size).collect();
    assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    assert_eq!(sizes, vec![1, 2, 3]);
}

#[tokio::test]
async fn list_directories_returns_only_directories() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[("a.txt", "a"), ("sub/b.txt", "b"), ("sub/deep/c.txt", "c")],
    );
    std::fs::create_dir_all(dir.path().join("empty")).unwrap();
    let fs = backend(&dir);

    let result = fs
        .list_directories(ListDirsParams::new(path_string(&dir)))
        .await;
    assert_eq!(result.code, 0, "list failed: {}", result.message);
    let data = result.data.expect("listing");
    assert_eq!(sorted_names(&data.list_items), vec!["empty", "sub"]);
    assert!(
        data.list_items.iter().all(|item| item.is_directory),
        "only directories are returned"
    );
    assert!(
        data.list_items.iter().all(|item| item.file_type.is_none()),
        "directories carry no suffix"
    );
}

#[tokio::test]
async fn search_files_glob_matches_at_any_depth() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[
            ("root.txt", "r"),
            ("sub/nested.txt", "n"),
            ("sub/deep/deeper.txt", "d"),
            ("notes.md", "m"),
        ],
    );
    let fs = backend(&dir);

    let result = fs
        .search_files(SearchFilesParams::new(path_string(&dir), "*.txt"))
        .await;
    assert_eq!(result.code, 0, "search failed: {}", result.message);
    let data = result.data.expect("search results");
    assert_eq!(data.total_matches, 3);
    assert_eq!(
        sorted_names(&data.matching_files),
        vec!["deeper.txt", "nested.txt", "root.txt"],
        "a pattern without '/' matches the file name at any depth"
    );
}

#[tokio::test]
async fn search_files_double_star_matches_zero_or_more_components() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[
            ("root.txt", "r"),
            ("sub/nested.txt", "n"),
            ("sub/deep/deeper.txt", "d"),
            ("notes.md", "m"),
        ],
    );
    let fs = backend(&dir);

    let result = fs
        .search_files(SearchFilesParams::new(path_string(&dir), "**/*.txt"))
        .await;
    assert_eq!(result.code, 0, "search failed: {}", result.message);
    let data = result.data.expect("search results");
    assert_eq!(data.total_matches, 3);
    assert_eq!(
        sorted_names(&data.matching_files),
        vec!["deeper.txt", "nested.txt", "root.txt"],
        "** matches zero or more path components"
    );
    assert!(
        sorted_names(&data.matching_files).contains(&"root.txt".to_string()),
        "a root-level file needs zero components"
    );
}

#[tokio::test]
async fn search_files_exclusions_remove_matches() {
    let dir = TempDir::new().unwrap();
    write_tree(
        dir.path(),
        &[
            ("keep_root.txt", "k"),
            ("skip_root.txt", "s"),
            ("sub/keep.txt", "k"),
            ("sub/skip_nested.txt", "s"),
        ],
    );
    let fs = backend(&dir);

    let params = SearchFilesParams::new(path_string(&dir), "*.txt").with_excludes(["**/skip*"]);
    let result = fs.search_files(params).await;
    assert_eq!(result.code, 0, "search failed: {}", result.message);
    let data = result.data.expect("search results");
    assert_eq!(data.total_matches, 2);
    assert_eq!(
        sorted_names(&data.matching_files),
        vec!["keep.txt", "keep_root.txt"],
        "matching files are removed by the exclude globs"
    );
}

#[tokio::test]
async fn search_files_no_match_is_success() {
    let dir = TempDir::new().unwrap();
    write_tree(dir.path(), &[("a.txt", "a")]);
    let fs = backend(&dir);

    let result = fs
        .search_files(SearchFilesParams::new(path_string(&dir), "*.rst"))
        .await;
    assert_eq!(result.code, 0, "no match is a success: {}", result.message);
    let data = result.data.expect("search results");
    assert_eq!(data.total_matches, 0);
    assert!(
        data.matching_files.is_empty(),
        "an unmatched pattern returns no files"
    );
}

#[tokio::test]
async fn upload_file_copies_host_file_and_rejects_existing_destination() {
    let workspace = TempDir::new().unwrap();
    let host = TempDir::new().unwrap();
    std::fs::write(host.path().join("source.bin"), b"uploaded").unwrap();
    let fs = backend(&workspace);

    let source = host.path().join("source.bin").to_string_lossy().to_string();
    let target = workspace
        .path()
        .join("data")
        .join("target.bin")
        .to_string_lossy()
        .to_string();

    let uploaded = fs
        .upload_file(UploadFileParams {
            local_path: source.clone(),
            target_path: target.clone(),
            overwrite: false,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        })
        .await;
    assert_eq!(uploaded.code, 0, "upload failed: {}", uploaded.message);
    assert_eq!(uploaded.data.as_ref().map(|data| data.size), Some(8));

    let read = fs.read_file(ReadFileParams::bytes("data/target.bin")).await;
    let data = read.data.expect("read payload");
    assert_eq!(data.content, FileContent::Bytes(b"uploaded".to_vec()));

    let rejected = fs
        .upload_file(UploadFileParams {
            local_path: source.clone(),
            target_path: target.clone(),
            overwrite: false,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        })
        .await;
    assert_eq!(rejected.code, FS_ERROR);
    assert!(
        rejected.message.contains("Target exists"),
        "unexpected message: {}",
        rejected.message
    );

    let overwritten = fs
        .upload_file(UploadFileParams {
            local_path: source,
            target_path: target,
            overwrite: true,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        })
        .await;
    assert_eq!(
        overwritten.code, 0,
        "overwrite failed: {}",
        overwritten.message
    );
}

#[tokio::test]
async fn download_file_copies_to_host_and_rejects_existing_destination() {
    let workspace = TempDir::new().unwrap();
    let host = TempDir::new().unwrap();
    write_tree(workspace.path(), &[("payload.txt", "payload")]);
    let fs = backend(&workspace);

    let source = workspace
        .path()
        .join("payload.txt")
        .to_string_lossy()
        .to_string();
    let destination = host.path().join("out.txt").to_string_lossy().to_string();

    let downloaded = fs
        .download_file(DownloadFileParams {
            source_path: source.clone(),
            local_path: destination.clone(),
            overwrite: false,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        })
        .await;
    assert_eq!(
        downloaded.code, 0,
        "download failed: {}",
        downloaded.message
    );
    assert_eq!(downloaded.data.as_ref().map(|data| data.size), Some(7));
    assert_eq!(
        std::fs::read(host.path().join("out.txt")).unwrap(),
        b"payload".to_vec(),
        "the host target is written even though it is outside the workspace"
    );

    let rejected = fs
        .download_file(DownloadFileParams {
            source_path: source.clone(),
            local_path: destination.clone(),
            overwrite: false,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        })
        .await;
    assert_eq!(rejected.code, FS_ERROR);
    assert!(
        rejected.message.contains("Target exists"),
        "unexpected message: {}",
        rejected.message
    );

    let overwritten = fs
        .download_file(DownloadFileParams {
            source_path: source,
            local_path: destination,
            overwrite: true,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        })
        .await;
    assert_eq!(
        overwritten.code, 0,
        "overwrite failed: {}",
        overwritten.message
    );
}

#[tokio::test]
async fn read_missing_file_is_a_199003_failure() {
    let dir = TempDir::new().unwrap();
    let fs = backend(&dir);

    let result = fs.read_file(ReadFileParams::text("missing.txt")).await;
    assert_eq!(result.code, FS_ERROR);
    assert!(result.data.is_none());
    assert!(
        result.message.contains("File does not exist"),
        "unexpected message: {}",
        result.message
    );
}
