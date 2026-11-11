//! Temporary directory tree used as a workspace root by tests.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

/// An owned temporary directory tree with the layout `openJiuwen` tests expect.
///
/// The tree is created under the system temporary directory with a random name
/// and removed recursively when the value is dropped, so tests never share (or
/// leak) filesystem state. The initial layout is:
///
/// ```text
/// <root>/
///   home/
///   workspace/
///   logs/
/// ```
///
/// This type never reads or mutates the process environment; pair it with
/// [`TempHomeGuard`](super::TempHomeGuard) when a test needs `OPENJIUWEN_HOME`
/// pointed at [`TempWorkspace::home`].
#[derive(Debug)]
pub struct TempWorkspace {
    root: tempfile::TempDir,
}

impl TempWorkspace {
    /// Creates a fresh temporary root with the `home/`, `workspace/`, and
    /// `logs/` subdirectories.
    ///
    /// Only filesystem failures are reported; a failure while creating one of
    /// the three well-known subdirectories is propagated because the layout is
    /// part of the contract.
    pub fn new() -> io::Result<Self> {
        let workspace = Self {
            root: tempfile::TempDir::new()?,
        };
        for name in ["home", "workspace", "logs"] {
            fs::create_dir_all(workspace.root().join(name))?;
        }
        Ok(workspace)
    }

    /// The absolute path of the temporary root.
    pub fn root(&self) -> &Path {
        self.root.path()
    }

    /// Joins `rel` onto [`TempWorkspace::root`] without touching the filesystem.
    ///
    /// Nothing is created: a path returned here may not exist until
    /// [`TempWorkspace::write`] or [`TempWorkspace::create_dir`] runs.
    pub fn path(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.root().join(rel)
    }

    /// The `home/` subdirectory, created on call.
    ///
    /// The directory is created eagerly so callers can pass it straight to an
    /// environment variable. Creation failures are ignored because this getter
    /// cannot fail; a caller that needs the directory to exist should use
    /// [`TempWorkspace::create_dir`] and handle the error.
    pub fn home(&self) -> PathBuf {
        self.dir("home")
    }

    /// The `workspace/` subdirectory, created on call (failures ignored, see
    /// [`TempWorkspace::home`]).
    pub fn workspace(&self) -> PathBuf {
        self.dir("workspace")
    }

    /// The `logs/` subdirectory, created on call (failures ignored, see
    /// [`TempWorkspace::home`]).
    pub fn logs(&self) -> PathBuf {
        self.dir("logs")
    }

    /// Creates `rel` (and any missing parents) under the root.
    ///
    /// Returns the absolute path that was created.
    pub fn create_dir(&self, rel: impl AsRef<Path>) -> io::Result<PathBuf> {
        let path = self.path(rel);
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Writes `contents` to `rel`, creating missing parent directories.
    ///
    /// Returns the absolute path that was written.
    pub fn write(&self, rel: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> io::Result<PathBuf> {
        let path = self.path(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, contents)?;
        Ok(path)
    }

    /// Reads `rel` as UTF-8 text.
    pub fn read(&self, rel: impl AsRef<Path>) -> io::Result<String> {
        fs::read_to_string(self.path(rel))
    }

    /// Serializes `value` as pretty JSON into `rel`, creating missing parents.
    ///
    /// I/O failures surface as [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp)
    /// and serialization failures as
    /// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation);
    /// both messages name the relative path.
    pub fn write_json(&self, rel: impl AsRef<Path>, value: &Value) -> Result<PathBuf> {
        let rel = rel.as_ref();
        let text = serde_json::to_string_pretty(value)
            .map_err(|error| Error::validation(format!("write_json {}: {error}", rel.display())))?;
        let path = self.path(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| Error::sys_op(format!("write_json {}: {error}", rel.display())))?;
        }
        fs::write(&path, text)
            .map_err(|error| Error::sys_op(format!("write_json {}: {error}", rel.display())))?;
        Ok(path)
    }

    /// Parses `rel` as JSON.
    ///
    /// Read failures surface as [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp)
    /// and malformed documents as
    /// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation);
    /// both messages name the relative path.
    pub fn read_json(&self, rel: impl AsRef<Path>) -> Result<Value> {
        let rel = rel.as_ref();
        let path = self.path(rel);
        let text = fs::read_to_string(&path)
            .map_err(|error| Error::sys_op(format!("read_json {}: {error}", rel.display())))?;
        serde_json::from_str(&text)
            .map_err(|error| Error::validation(format!("read_json {}: {error}", rel.display())))
    }

    /// Reports whether `rel` exists under the root.
    pub fn exists(&self, rel: impl AsRef<Path>) -> bool {
        self.path(rel).exists()
    }

    /// Every regular file under the root, relative to it and lexicographically
    /// sorted.
    ///
    /// Directories are never returned; unreadable entries are skipped rather
    /// than reported because this is a diagnostics helper for assertions.
    pub fn files(&self) -> Vec<PathBuf> {
        let root = self.root();
        let mut found = Vec::new();
        collect_files(root, root, &mut found);
        found.sort();
        found
    }

    /// Returns `<root>/<name>`, creating the directory and ignoring failures.
    fn dir(&self, name: &str) -> PathBuf {
        let path = self.root().join(name);
        let _ = fs::create_dir_all(&path);
        path
    }
}

/// Recursively collects regular files below `dir`, relative to `root`.
fn collect_files(root: &Path, dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, found);
        } else if let Ok(relative) = path.strip_prefix(root) {
            found.push(relative.to_path_buf());
        }
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    #[test]
    fn new_creates_the_expected_layout() {
        let workspace = TempWorkspace::new().expect("temp workspace");

        assert!(workspace.home().is_dir());
        assert!(workspace.workspace().is_dir());
        assert!(workspace.logs().is_dir());
        assert_eq!(workspace.home(), workspace.path("home"));
        assert_eq!(workspace.workspace(), workspace.path("workspace"));
        assert_eq!(workspace.logs(), workspace.path("logs"));
    }

    #[test]
    fn nested_relative_paths_round_trip() {
        let workspace = TempWorkspace::new().expect("temp workspace");

        let written = workspace.write("a/b/c.txt", "hello").expect("write");

        assert_eq!(written, workspace.path("a/b/c.txt"));
        assert!(written.is_absolute());
        assert_eq!(workspace.read("a/b/c.txt").expect("read"), "hello");
        assert!(workspace.exists("a/b/c.txt"));
    }

    #[test]
    fn path_does_not_create_anything() {
        let workspace = TempWorkspace::new().expect("temp workspace");

        let path = workspace.path("missing/deep/file.txt");

        assert!(!path.exists());
        assert!(!workspace.exists("missing"));
    }

    #[test]
    fn read_reports_missing_files() {
        let workspace = TempWorkspace::new().expect("temp workspace");

        let error = workspace.read("nope.txt").expect_err("missing file");

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn create_dir_creates_nested_directories() {
        let workspace = TempWorkspace::new().expect("temp workspace");

        let path = workspace.create_dir("x/y/z").expect("create_dir");

        assert!(path.is_dir());
        assert_eq!(path, workspace.path("x/y/z"));
    }

    #[test]
    fn json_round_trips_through_pretty_output() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let value = json!({"name": "agent", "nested": {"count": 2}, "tags": ["a", "b"]});

        let path = workspace
            .write_json("state/state.json", &value)
            .expect("write");
        let raw = workspace.read("state/state.json").expect("read raw");

        assert!(raw.contains('\n'), "pretty output is multi-line: {raw}");
        assert_eq!(
            workspace.read_json("state/state.json").expect("read"),
            value
        );
        assert_eq!(path, workspace.path("state/state.json"));
    }

    #[test]
    fn read_json_reports_missing_files_as_sys_op() {
        let workspace = TempWorkspace::new().expect("temp workspace");

        let error = workspace
            .read_json("absent.json")
            .expect_err("missing file");

        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error.to_string().contains("absent.json"));
    }

    #[test]
    fn read_json_reports_malformed_documents_as_validation() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        workspace.write("bad.json", "{not json").expect("write");

        let error = workspace.read_json("bad.json").expect_err("malformed");

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("bad.json"));
    }

    #[test]
    fn files_are_relative_sorted_and_file_only() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        workspace.write("b.txt", "b").expect("write");
        workspace.write("a/2.txt", "2").expect("write");
        workspace.write("a/1.txt", "1").expect("write");
        workspace.create_dir("empty").expect("create_dir");

        let files = workspace.files();

        assert_eq!(
            files,
            vec![
                PathBuf::from("a/1.txt"),
                PathBuf::from("a/2.txt"),
                PathBuf::from("b.txt"),
            ]
        );
        assert!(files.iter().all(|file| !file.is_absolute()));
    }

    #[test]
    fn drop_removes_the_root() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let root = workspace.root().to_path_buf();
        workspace.write("file.txt", "x").expect("write");
        assert!(root.exists());

        drop(workspace);

        assert!(!root.exists());
    }
}
