//! Workspace and path-safety integration suite: default schema seeding (`cn` vs `en`),
//! traversal / absolute / drive-letter rejection through `Workspace::resolve`, top-level
//! node lookup, and idempotent schema rebuilds through `DirectoryBuilder`.
//!
//! Every test owns a `tempfile::TempDir`, touches only that directory, and runs offline.
//! Only the public `openjiuwen_harness::workspace` API is used.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_harness::workspace::{
    is_safe_relative_path, DirectoryBuilder, DirectoryNode, Workspace, WorkspaceNode,
    WORKSPACE_MARKER,
};
use openjiuwen_sysop::{LocalWorkConfig, SysOperation};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Creates a fresh temporary workspace root.
fn temp_root() -> TempDir {
    tempfile::tempdir().expect("temp dir")
}

/// Builds a local sys-operation rooted at `root`.
fn local_operation(root: &Path) -> SysOperation {
    SysOperation::local(LocalWorkConfig::unrestricted(
        root.to_string_lossy().to_string(),
    ))
    .expect("local sys operation")
}

/// Materializes `workspace`'s directory schema under `root` through the local backend.
async fn seed(workspace: &Workspace, root: &Path) {
    DirectoryBuilder::new(local_operation(root), root)
        .build(&workspace.directories)
        .await
        .expect("seed workspace schema");
}

/// Whether `relative` exists under `root` and is a regular file.
fn has_file(root: &Path, relative: &str) -> bool {
    root.join(relative).is_file()
}

/// Whether `relative` exists under `root` and is a directory.
fn has_dir(root: &Path, relative: &str) -> bool {
    root.join(relative).is_dir()
}

/// Reads a workspace file as UTF-8.
fn read_text(root: &Path, relative: &str) -> String {
    std::fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

/// Recursively lists every path under `root` as sorted `/`-separated relative strings.
fn sorted_listing(root: &Path) -> Vec<String> {
    fn walk(base: &Path, current: &Path, out: &mut Vec<String>) {
        let entries = std::fs::read_dir(current).expect("read directory");
        for entry in entries {
            let path = entry.expect("directory entry").path();
            let relative = path.strip_prefix(base).expect("relative entry");
            out.push(relative.to_string_lossy().replace('\\', "/"));
            if path.is_dir() {
                walk(base, &path, out);
            }
        }
    }

    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// Asserts that `resolve(path)` rejects `path` with a validation error naming it.
async fn assert_unsafe_path(workspace: &Workspace, path: &str) {
    let error = workspace
        .resolve(path)
        .await
        .expect_err("unsafe path must be rejected");
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "path {path:?}: {error}"
    );
    let message = error.message();
    assert!(
        message.contains("unsafe workspace path"),
        "path {path:?}: {message}"
    );
    assert!(message.contains(path), "path {path:?}: {message}");
}

// ---------------------------------------------------------------------------
// default schema seeding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn seeds_cn_schema_with_files_directories_and_markers() {
    let temp = temp_root();
    let root = temp.path();
    let workspace = Workspace::for_language(root, "cn");
    seed(&workspace, root).await;

    assert_eq!(workspace.language(), "cn");
    assert!(!workspace.is_english());
    assert_eq!(workspace.directories.len(), 12);

    for file in [
        "AGENT.md",
        "SOUL.md",
        "HEARTBEAT.md",
        "IDENTITY.md",
        "USER.md",
    ] {
        assert!(has_file(root, file), "missing seeded file {file}");
        assert!(
            !read_text(root, file).trim().is_empty(),
            "seeded file {file} is empty"
        );
    }

    for dir in [
        "memory",
        "coding_memory",
        "todo",
        "messages",
        "skills",
        "agents",
        "context",
    ] {
        assert!(has_dir(root, dir), "missing seeded directory {dir}");
        let marker = format!("{dir}/{WORKSPACE_MARKER}");
        assert!(has_file(root, &marker), "missing marker {marker}");
        assert_eq!(read_text(root, &marker), "", "marker {marker} is not empty");
    }

    // File nodes are plain files, never directories carrying a marker.
    assert!(!has_dir(root, "AGENT.md"));
    assert!(!has_file(root, &format!("AGENT.md/{WORKSPACE_MARKER}")));

    // Nested schema nodes.
    assert!(has_file(root, "memory/MEMORY.md"));
    assert!(has_dir(root, "memory/daily_memory"));
    assert!(has_file(
        root,
        &format!("memory/daily_memory/{WORKSPACE_MARKER}")
    ));
    assert!(has_file(root, "coding_memory/MEMORY.md"));
    assert_eq!(
        read_text(root, "coding_memory/MEMORY.md"),
        "",
        "coding memory seed content is empty"
    );
    assert!(has_file(root, "context/session_memory.md"));
    assert_ne!(read_text(root, "context/session_memory.md").trim(), "");
}

#[tokio::test]
async fn seeds_en_schema_without_coding_memory() {
    let temp = temp_root();
    let root = temp.path();
    let workspace = Workspace::for_language(root, "en");
    seed(&workspace, root).await;

    assert_eq!(workspace.language(), "en");
    assert!(workspace.is_english());
    assert_eq!(workspace.directories.len(), 11);

    // The English schema drops `coding_memory` entirely.
    assert!(!has_dir(root, "coding_memory"));
    assert!(!has_file(root, "coding_memory/MEMORY.md"));
    assert!(workspace
        .get_node_path_for(WorkspaceNode::CodingMemory)
        .is_none());

    // Everything else is seeded the same way as the Chinese schema.
    assert!(has_file(root, "AGENT.md"));
    assert!(has_dir(root, "memory"));
    assert!(has_file(root, "memory/MEMORY.md"));
    assert!(has_dir(root, "memory/daily_memory"));
    assert!(has_dir(root, "todo"));
    assert!(has_dir(root, "skills"));
    assert!(has_file(root, "context/session_memory.md"));
}

// ---------------------------------------------------------------------------
// path traversal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rejects_relative_paths_that_escape_the_root() {
    let temp = temp_root();
    let root = temp.path();
    let workspace = Workspace::for_language(root, "cn");

    for path in [
        "../escape",
        "a/../../b",
        "..",
        "a/../..",
        "memory/../../escape",
        "..\\escape",
        "",
    ] {
        assert_unsafe_path(&workspace, path).await;
    }

    // A safe relative path still resolves inside the workspace root.
    seed(&workspace, root).await;
    let canonical_root = std::fs::canonicalize(root).expect("canonical root");
    let resolved = workspace
        .resolve("memory/MEMORY.md")
        .await
        .expect("safe path resolves");
    assert!(resolved.starts_with(&canonical_root), "{resolved:?}");
    assert!(resolved.ends_with("MEMORY.md"), "{resolved:?}");
}

#[tokio::test]
async fn rejects_absolute_paths_even_inside_the_root() {
    let temp = temp_root();
    let root = temp.path();
    let workspace = Workspace::for_language(root, "cn");

    let inside = root.join("AGENT.md").to_string_lossy().into_owned();
    let root_string = root.to_string_lossy().into_owned();
    for path in ["/etc/passwd", inside.as_str(), root_string.as_str()] {
        assert_unsafe_path(&workspace, path).await;
    }

    // The same file is reachable when named relatively.
    let resolved = workspace.resolve("AGENT.md").await.expect("relative form");
    assert_eq!(
        resolved,
        std::fs::canonicalize(root)
            .expect("canonical root")
            .join("AGENT.md")
    );
}

#[tokio::test]
async fn rejects_windows_drive_and_unc_paths() {
    let temp = temp_root();
    let root = temp.path();
    let workspace = Workspace::for_language(root, "cn");

    for path in [
        "C:\\Windows\\system32",
        "c:/tmp/x",
        "C:",
        "z:file.txt",
        "\\\\server\\share\\file.txt",
    ] {
        assert_unsafe_path(&workspace, path).await;
    }
}

// ---------------------------------------------------------------------------
// node lookup
// ---------------------------------------------------------------------------

#[test]
fn get_node_path_resolves_only_top_level_nodes() {
    let temp = temp_root();
    let root = temp.path();
    let root_path: PathBuf = root.to_path_buf();
    let workspace = Workspace::for_language(root, "cn");

    for name in [
        "AGENT.md",
        "SOUL.md",
        "memory",
        "coding_memory",
        "todo",
        "skills",
        "context",
    ] {
        let path = workspace
            .get_node_path(name)
            .unwrap_or_else(|| panic!("missing top-level node {name}"));
        assert_eq!(
            path,
            root_path.join(name),
            "node {name} is not a plain join"
        );
        assert!(path.starts_with(&root_path), "node {name}: {path:?}");
        assert_eq!(path.parent(), Some(root_path.as_path()));
        assert!(
            !path.to_string_lossy().contains(".."),
            "node {name} escaped: {path:?}"
        );
    }

    for nested in [
        "memory/daily_memory",
        "daily_memory",
        "MEMORY.md",
        "session_memory.md",
        "coding_memory/MEMORY.md",
        "missing",
    ] {
        assert_eq!(
            workspace.get_node_path(nested),
            None,
            "nested name {nested} must not resolve"
        );
    }

    assert_eq!(
        workspace.get_node_path_for(WorkspaceNode::Memory),
        Some(root_path.join("memory"))
    );
    assert_eq!(
        workspace.get_node_path_for(WorkspaceNode::CodingMemory),
        Some(root_path.join("coding_memory"))
    );
    assert_eq!(workspace.get_node_path_for(WorkspaceNode::TeamLinks), None);
}

// ---------------------------------------------------------------------------
// idempotency
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rebuild_preserves_authored_content_and_stays_idempotent() {
    let temp = temp_root();
    let root = temp.path();
    let workspace = Workspace::for_language(root, "cn");
    seed(&workspace, root).await;

    let authored = "# Authored agent configuration\n\nDo not overwrite me.\n";
    std::fs::write(root.join("AGENT.md"), authored).expect("write custom AGENT.md");
    let skill = "---\nname: my_skill\ndescription: custom skill\n---\n";
    std::fs::create_dir_all(root.join("skills/my_skill")).expect("create skill dir");
    std::fs::write(root.join("skills/my_skill/SKILL.md"), skill).expect("write skill file");

    let before = sorted_listing(root);
    seed(&workspace, root).await;
    let after = sorted_listing(root);

    assert_eq!(before, after, "rebuild must not add or remove paths");
    assert_eq!(
        read_text(root, "AGENT.md"),
        authored,
        "authored content clobbered"
    );
    assert_eq!(read_text(root, "skills/my_skill/SKILL.md"), skill);

    // A rebuild restores a marker an agent deleted, without touching neighbours.
    std::fs::remove_file(root.join(format!("todo/{WORKSPACE_MARKER}"))).expect("remove marker");
    assert!(!has_file(root, &format!("todo/{WORKSPACE_MARKER}")));
    seed(&workspace, root).await;
    assert!(
        has_file(root, &format!("todo/{WORKSPACE_MARKER}")),
        "rebuild did not restore the todo marker"
    );
    assert_eq!(read_text(root, "AGENT.md"), authored);
    assert_eq!(
        sorted_listing(root),
        after,
        "marker restore changed the tree"
    );
}

// ---------------------------------------------------------------------------
// unsupported node paths
// ---------------------------------------------------------------------------

#[test]
fn is_safe_relative_path_truth_table() {
    for safe in [
        "a/b",
        "./a/b",
        "a\\b",
        "a/../b",
        "AGENT.md",
        "memory/MEMORY.md",
        ".hidden",
        "a/./b",
        "a/b/",
    ] {
        assert!(is_safe_relative_path(safe), "{safe:?} should be safe");
    }

    for unsafe_path in [
        "",
        ".",
        "..",
        "x/..",
        "a/..",
        "/abs",
        "\\abs",
        "\\\\server\\share",
        "C:/x",
        "c:\\x",
        "../x",
        "a/../../b",
        "a\0b",
    ] {
        assert!(
            !is_safe_relative_path(unsafe_path),
            "{unsafe_path:?} should be unsafe"
        );
    }
}

#[tokio::test]
async fn build_rejects_unsafe_node_paths_before_touching_disk() {
    let temp = temp_root();
    let root = temp.path();
    let builder = DirectoryBuilder::new(local_operation(root), root);

    let directory_node = DirectoryNode::dir("escape", "../escape");
    let error = builder
        .build(std::slice::from_ref(&directory_node))
        .await
        .expect_err("traversal node must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
    assert!(
        error.message().contains("Unsafe path detected: ../escape"),
        "{error}"
    );
    assert!(
        std::fs::read_dir(root).expect("read root").next().is_none(),
        "rejected node still wrote to disk"
    );

    let file_node = DirectoryNode::file("abs", "/tmp/abs-escape.txt", "body");
    let error = builder
        .build(std::slice::from_ref(&file_node))
        .await
        .expect_err("absolute node must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
    assert!(
        error
            .message()
            .contains("Unsafe path detected: /tmp/abs-escape.txt"),
        "{error}"
    );
    assert!(!Path::new("/tmp/abs-escape.txt").exists());
}
