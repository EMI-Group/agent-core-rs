//! Ports `tests/unit_tests/core/sys_operation/local/test_fs_operation.py` (path containment
//! and sandbox escape attempts).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::fs::local_fs;
use openjiuwen_sysop::fs::types::{
    FileContent, ListFilesParams, ReadFileData, ReadFileParams, WriteFileParams,
};
use openjiuwen_sysop::result::OpResult;
use openjiuwen_sysop::status::SysOpStatus;
use openjiuwen_sysop::FsOperation;

/// A throwaway workspace plus a file that lives one level above it.
struct Fixture {
    base: tempfile::TempDir,
    workspace: PathBuf,
    outside_file: PathBuf,
}

impl Fixture {
    /// Creates `<base>/workspace` and `<base>/outside.txt`.
    fn new() -> Self {
        let base = tempfile::tempdir().expect("temp dir");
        let workspace = base.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("create workspace");
        let outside_file = base.path().join("outside.txt");
        std::fs::write(&outside_file, "outside").expect("write outside file");
        Self {
            base,
            workspace,
            outside_file,
        }
    }

    /// The absolute workspace path as a UTF-8 (lossy) string.
    fn workspace_str(&self) -> String {
        self.workspace.to_string_lossy().into_owned()
    }

    /// The sandbox root directory that contains the workspace.
    fn base_path(&self) -> &Path {
        self.base.path()
    }

    /// A backend rooted at the workspace and restricted to it.
    fn restricted(&self) -> Arc<dyn FsOperation> {
        let workspace = self.workspace_str();
        local_fs(LocalWorkConfig::unrestricted(workspace.clone()).with_sandbox_roots([workspace]))
            .expect("restricted backend")
    }
}

/// Returns the text of a successful read.
fn text_of(data: &ReadFileData) -> String {
    match &data.content {
        FileContent::Text(text) => text.clone(),
        FileContent::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

/// Asserts that `result` is a filesystem failure naming `raw` as an access violation.
fn assert_denied<T: std::fmt::Debug>(result: &OpResult<T>, raw: &str) {
    assert_eq!(
        result.code,
        SysOpStatus::FsExecutionError.code(),
        "expected a 199003 denial, got {} ({:?})",
        result.code,
        result.message
    );
    assert!(
        result.message.contains("Access denied: Path"),
        "message: {}",
        result.message
    );
    assert!(
        result.message.contains(raw),
        "message `{}` does not name `{raw}`",
        result.message
    );
    assert!(result.data.is_none(), "a denial must not carry data");
}

#[tokio::test]
async fn relative_dot_dot_escape_is_rejected_for_read() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();

    let result = fs.read_file(ReadFileParams::text("../outside.txt")).await;

    assert_denied(&result, "../outside.txt");
}

#[tokio::test]
async fn absolute_path_outside_root_is_rejected() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();
    let outside = fixture.outside_file.to_string_lossy().into_owned();

    let result = fs.read_file(ReadFileParams::text(outside.clone())).await;

    assert_eq!(result.code, SysOpStatus::FsExecutionError.code());
    assert!(
        result.message.contains("Access denied: Path"),
        "message: {}",
        result.message
    );
    assert!(
        result.message.contains(&outside),
        "message `{}` does not name the absolute path",
        result.message
    );
}

#[tokio::test]
async fn write_escape_is_rejected_and_creates_nothing() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();

    let result = fs
        .write_file(WriteFileParams::raw("../evil.txt", "x"))
        .await;

    assert_denied(&result, "../evil.txt");
    assert!(
        !fixture.base_path().join("evil.txt").exists(),
        "a rejected write must not create a file outside the sandbox"
    );
    assert!(
        !fixture.workspace.join("evil.txt").exists(),
        "a rejected write must not create a file inside the sandbox either"
    );
}

#[tokio::test]
async fn path_equal_to_root_is_allowed() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();

    let result = fs
        .list_files(ListFilesParams::new(fixture.workspace_str()))
        .await;

    assert_eq!(
        result.code, 0,
        "listing the sandbox root itself must succeed: {}",
        result.message
    );
    assert_eq!(result.data.expect("listing data").total_count, 0);
}

#[tokio::test]
async fn relative_path_inside_root_works() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();

    let written = fs
        .write_file(WriteFileParams::raw("notes.txt", "inside"))
        .await;
    assert_eq!(
        written.code, 0,
        "an in-root write must succeed: {}",
        written.message
    );

    let read = fs.read_file(ReadFileParams::text("notes.txt")).await;
    assert_eq!(
        read.code, 0,
        "an in-root read must succeed: {}",
        read.message
    );
    assert_eq!(text_of(&read.data.expect("read data")), "inside");
}

#[tokio::test]
async fn dot_dot_resolving_back_inside_is_allowed() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();
    std::fs::create_dir_all(fixture.workspace.join("sub")).expect("create sub");

    let written = fs
        .write_file(WriteFileParams::raw("sub/../notes.txt", "still inside"))
        .await;
    assert_eq!(
        written.code, 0,
        "lexical `..` that stays inside the root must be allowed: {}",
        written.message
    );
    assert!(
        fixture.workspace.join("notes.txt").is_file(),
        "the write must land on the normalized path"
    );

    let read = fs.read_file(ReadFileParams::text("sub/../notes.txt")).await;
    assert_eq!(
        read.code, 0,
        "reading through an in-root `..` must succeed: {}",
        read.message
    );
    assert_eq!(text_of(&read.data.expect("read data")), "still inside");
}

#[tokio::test]
async fn sibling_with_shared_textual_prefix_is_not_inside() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();
    let sibling = fixture.base_path().join("workspace-other");
    std::fs::create_dir_all(&sibling).expect("create sibling");
    std::fs::write(sibling.join("secret.txt"), "secret").expect("write secret");
    let secret = sibling.join("secret.txt").to_string_lossy().into_owned();

    let result = fs.read_file(ReadFileParams::text(secret.clone())).await;

    assert_eq!(
        result.code,
        SysOpStatus::FsExecutionError.code(),
        "a sibling sharing the root's textual prefix is outside the sandbox"
    );
    assert!(
        result.message.contains("Access denied: Path"),
        "message: {}",
        result.message
    );
    assert!(
        result.message.contains(&secret),
        "message `{}` does not name the sibling path",
        result.message
    );
}

#[tokio::test]
async fn escape_message_names_the_raw_path() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();

    let result = fs.read_file(ReadFileParams::text("../outside.txt")).await;

    assert_eq!(result.code, SysOpStatus::FsExecutionError.code());
    assert!(
        result.message.contains("Access denied: Path"),
        "message: {}",
        result.message
    );
    assert!(
        result.message.contains("../outside.txt"),
        "Python reports the raw path, not the resolved one: {}",
        result.message
    );
    assert!(
        result.message.contains("resolve_path"),
        "the failure keeps Python's execution slot: {}",
        result.message
    );
}

#[tokio::test]
async fn unrestricted_config_allows_reads_outside_the_workspace() {
    let fixture = Fixture::new();
    let fs = local_fs(LocalWorkConfig::unrestricted(fixture.workspace_str()))
        .expect("unrestricted backend");
    let outside = fixture.outside_file.to_string_lossy().into_owned();

    let result = fs.read_file(ReadFileParams::text(outside)).await;

    assert_eq!(
        result.code, 0,
        "without restrict_to_sandbox any path is readable: {}",
        result.message
    );
    assert_eq!(text_of(&result.data.expect("read data")), "outside");
}

#[cfg(unix)]
#[tokio::test]
async fn symlink_to_outside_is_rejected_for_read() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();
    std::os::unix::fs::symlink(&fixture.outside_file, fixture.workspace.join("link.txt"))
        .expect("create symlink");

    let result = fs.read_file(ReadFileParams::text("link.txt")).await;

    assert_denied(&result, "link.txt");
}

#[cfg(unix)]
#[tokio::test]
async fn symlink_to_outside_directory_is_rejected_for_write() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();
    std::os::unix::fs::symlink(fixture.base_path(), fixture.workspace.join("escape"))
        .expect("create dir symlink");

    let result = fs
        .write_file(WriteFileParams::raw("escape/pwn.txt", "pwn"))
        .await;

    assert_denied(&result, "escape/pwn.txt");
    assert!(
        !fixture.base_path().join("pwn.txt").exists(),
        "a write through a directory symlink must not escape the sandbox"
    );
}

#[tokio::test]
async fn directory_listing_of_an_outside_path_is_rejected() {
    let fixture = Fixture::new();
    let fs = fixture.restricted();
    let outside_dir = fixture.base_path().join("outside_dir");
    std::fs::create_dir_all(&outside_dir).expect("create outside dir");
    std::fs::write(outside_dir.join("data.txt"), "data").expect("write data");
    let outside_dir = outside_dir.to_string_lossy().into_owned();

    let result = fs
        .list_files(ListFilesParams::new(outside_dir.clone()))
        .await;

    assert_eq!(
        result.code,
        SysOpStatus::FsExecutionError.code(),
        "listing outside the sandbox must be denied"
    );
    assert!(
        result.message.contains("Access denied: Path"),
        "message: {}",
        result.message
    );
    assert!(
        result.message.contains(&outside_dir),
        "message `{}` does not name the outside directory",
        result.message
    );
}
