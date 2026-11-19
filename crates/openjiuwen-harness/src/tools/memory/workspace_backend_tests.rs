//! Unit tests of [`WorkspaceMemoryBackend`](super::WorkspaceMemoryBackend).
//!
//! Kept beside `workspace_backend.rs` (through a `#[path]` module) so the implementation file
//! stays within the crate's per-file size budget.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use openjiuwen_sysop::fs::types::{
    DownloadFileData, DownloadFileParams, ListDirsParams, ListFilesData, ListFilesParams,
    ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData, SearchFilesParams,
    TransferChunkData, UploadFileData, UploadFileParams, WriteFileData,
};
use openjiuwen_sysop::{OpResult, OpStream, SysOpStatus};

use super::*;

/// Builds a backend over a throwaway English workspace.
fn backend() -> (tempfile::TempDir, WorkspaceMemoryBackend) {
    let dir = tempfile::tempdir().expect("temp workspace");
    let workspace = Workspace::for_language(dir.path(), "en");
    (dir, WorkspaceMemoryBackend::new(workspace))
}

/// How [`SpyFs`] answers a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpyMode {
    /// Create the temporary file at the requested path (a normal backend).
    Write,
    /// Fail the write without touching the disk (a refusing seam).
    Fail,
    /// Report success without creating the file, so the caller's rename cannot find it.
    Skip,
}

/// A filesystem backend that records every write and can refuse or hide it.
///
/// The spy is the assertion that a write actually goes through the [`FsOperation`] seam: a
/// counting backend sees the temporary-sibling write, a refusing one makes the whole write fail,
/// and a skipping one makes the rename step fail.
struct SpyFs {
    mode: SpyMode,
    root: PathBuf,
    writes: AtomicUsize,
    observed: Mutex<Vec<(String, String)>>,
}

impl SpyFs {
    /// Builds a spy over `root` answering writes with `mode`.
    fn new(mode: SpyMode, root: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            mode,
            root,
            writes: AtomicUsize::new(0),
            observed: Mutex::new(Vec::new()),
        })
    }

    /// Number of `write_file` calls the backend made through the seam.
    fn writes(&self) -> usize {
        self.writes.load(Ordering::SeqCst)
    }

    /// The `(path, text)` pairs the spy was asked to write.
    fn observed(&self) -> Vec<(String, String)> {
        self.observed.lock().expect("spy lock").clone()
    }
}

fn unsupported<T>() -> OpResult<T> {
    OpResult::failure(SysOpStatus::Error, "unsupported by the spy")
}

#[async_trait]
impl FsOperation for SpyFs {
    async fn read_file(&self, _params: ReadFileParams) -> OpResult<ReadFileData> {
        unsupported()
    }
    fn read_file_stream(self: Arc<Self>, _params: ReadFileParams) -> OpStream<ReadChunkData> {
        Box::pin(futures::stream::empty())
    }
    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        if self.mode == SpyMode::Fail {
            return OpResult::failure(SysOpStatus::FsExecutionError, "seam refused the write");
        }
        let text = params.content.as_text().unwrap_or_default().to_owned();
        self.observed
            .lock()
            .expect("spy lock")
            .push((params.path.clone(), text.clone()));
        if self.mode == SpyMode::Write {
            let path = self.root.join(&params.path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("spy parent directory");
            }
            std::fs::write(&path, &text).expect("spy temporary file");
        }
        OpResult::success(WriteFileData {
            path: params.path,
            size: text.len() as u64,
            mode: openjiuwen_sysop::fs::types::ReadMode::Text,
        })
    }
    async fn upload_file(&self, _params: UploadFileParams) -> OpResult<UploadFileData> {
        unsupported()
    }
    fn upload_file_stream(
        self: Arc<Self>,
        _params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(futures::stream::empty())
    }
    async fn download_file(&self, _params: DownloadFileParams) -> OpResult<DownloadFileData> {
        unsupported()
    }
    fn download_file_stream(
        self: Arc<Self>,
        _params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(futures::stream::empty())
    }
    async fn list_files(&self, _params: ListFilesParams) -> OpResult<ListFilesData> {
        unsupported()
    }
    async fn list_directories(&self, _params: ListDirsParams) -> OpResult<ListFilesData> {
        unsupported()
    }
    async fn search_files(&self, _params: SearchFilesParams) -> OpResult<SearchFilesData> {
        unsupported()
    }
}

/// Builds a backend over `dir` whose writes go through `spy`.
fn spied_backend(dir: &tempfile::TempDir, spy: Arc<SpyFs>) -> WorkspaceMemoryBackend {
    WorkspaceMemoryBackend::new(Workspace::for_language(dir.path(), "en")).with_fs_operation(spy)
}

/// Every file name below `root` that contains `.tmp`.
fn temp_siblings(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(temp_siblings(&path));
            continue;
        }
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.contains(".tmp") {
            found.push(name);
        }
    }
    found
}

#[tokio::test]
async fn writes_go_through_the_filesystem_seam() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let spy = SpyFs::new(SpyMode::Write, dir.path().to_path_buf());
    let backend = spied_backend(&dir, Arc::clone(&spy));

    let data = backend
        .write(MemoryScope::Memory, "MEMORY.md", "seam text", false)
        .await
        .expect("the spied write succeeds");

    assert_eq!(data.path, "memory/MEMORY.md");
    assert_eq!(spy.writes(), 1, "the backend must not bypass the seam");
    let observed = spy.observed();
    assert_eq!(observed.len(), 1);
    assert!(
        observed[0].0.contains(".tmp"),
        "the seam is handed the temporary sibling, got {}",
        observed[0].0
    );
    assert_eq!(observed[0].1, "seam text");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("memory/MEMORY.md")).expect("written file"),
        "seam text"
    );
    assert!(
        temp_siblings(dir.path()).is_empty(),
        "the temporary sibling is renamed away"
    );
}

#[tokio::test]
async fn a_refused_seam_write_keeps_the_previous_content() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let target = dir.path().join("memory/MEMORY.md");
    std::fs::create_dir_all(target.parent().expect("parent")).expect("memory directory");
    std::fs::write(&target, "original").expect("fixture");

    let spy = SpyFs::new(SpyMode::Fail, dir.path().to_path_buf());
    let backend = spied_backend(&dir, Arc::clone(&spy));
    let error = backend
        .write(MemoryScope::Memory, "MEMORY.md", "replacement", false)
        .await
        .expect_err("a refusing seam fails the write");

    assert!(
        error.message().contains("seam refused the write"),
        "the seam error surfaces, got {error}"
    );
    assert_eq!(
        spy.writes(),
        1,
        "the backend reached the seam before failing"
    );
    assert_eq!(
        std::fs::read_to_string(&target).expect("unchanged file"),
        "original"
    );
    assert!(
        temp_siblings(dir.path()).is_empty(),
        "no .tmp sibling survives"
    );
}

#[tokio::test]
async fn a_failed_rename_keeps_the_previous_content_and_removes_no_temp() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let target = dir.path().join("memory/MEMORY.md");
    std::fs::create_dir_all(target.parent().expect("parent")).expect("memory directory");
    std::fs::write(&target, "original").expect("fixture");

    // The seam reports success but never creates the temporary sibling, so the rename that
    // commits the write fails and the old content must survive.
    let spy = SpyFs::new(SpyMode::Skip, dir.path().to_path_buf());
    let backend = spied_backend(&dir, Arc::clone(&spy));
    let error = backend
        .write(MemoryScope::Memory, "MEMORY.md", "replacement", false)
        .await
        .expect_err("a missing temporary file fails the rename");

    assert!(
        error.message().contains("failed to write memory file"),
        "the rename failure surfaces, got {error}"
    );
    assert_eq!(spy.writes(), 1);
    assert_eq!(
        std::fs::read_to_string(&target).expect("unchanged file"),
        "original"
    );
    assert!(
        temp_siblings(dir.path()).is_empty(),
        "no .tmp sibling survives"
    );
}

#[test]
fn memory_paths_follow_the_python_mapping() {
    let (_dir, backend) = backend();

    let cases = [
        ("USER.md", "USER.md"),
        ("MEMORY.md", "memory/MEMORY.md"),
        ("memory/MEMORY.md", "memory/MEMORY.md"),
        ("2026-11-15.md", "memory/daily_memory/2026-11-15.md"),
        ("notes.md", "memory/notes.md"),
        ("nested/notes.md", "memory/notes.md"),
    ];
    for (path, expected) in cases {
        assert_eq!(
            backend
                .relative_target(MemoryScope::Memory, path)
                .expect("valid memory path"),
            expected,
            "{path}"
        );
    }
}

#[test]
fn coding_paths_require_a_markdown_name() {
    let (_dir, backend) = backend();

    assert_eq!(
        backend
            .relative_target(MemoryScope::CodingMemory, "projects/rust.md")
            .expect("valid coding path"),
        "coding_memory/rust.md"
    );
    let error = backend
        .relative_target(MemoryScope::CodingMemory, "notes.txt")
        .expect_err("txt is rejected");
    assert_eq!(error.message(), "Path must end with .md");
}

#[test]
fn traversal_and_empty_paths_are_validation_errors() {
    let (_dir, backend) = backend();

    for path in ["../x.md", "/etc/passwd", "\\evil.md", "a/../../b.md"] {
        let error = backend
            .relative_target(MemoryScope::Memory, path)
            .expect_err("unsafe path");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation,
            "{path}"
        );
    }
    assert_eq!(
        backend
            .relative_target(MemoryScope::Memory, "  ")
            .expect_err("blank path")
            .message(),
        "path is required"
    );
}

#[test]
fn line_windows_follow_python_offsets() {
    let lines = ["one", "two", "three", "four"];

    let window = LineWindow::of(&lines, 0, None);
    assert_eq!(window.text, "one\ntwo\nthree\nfour");
    assert_eq!((window.start_line, window.end_line), (1, 4));

    let window = LineWindow::of(&lines, 1, Some(2));
    assert_eq!(window.text, "one\ntwo");
    assert_eq!((window.start_line, window.end_line), (1, 2));

    let window = LineWindow::of(&lines, 3, Some(2));
    assert_eq!(window.text, "three\nfour");
    assert_eq!((window.start_line, window.end_line), (3, 4));

    let window = LineWindow::of(&lines, 9, Some(2));
    assert_eq!(window.text, "");
    assert_eq!((window.start_line, window.end_line), (5, 4));
}

#[test]
fn query_tokens_drop_short_words_and_duplicates() {
    assert_eq!(query_tokens("go to the SHOP"), ["the", "shop"]);
    assert_eq!(query_tokens("build BUILD steps"), ["build", "steps"]);
    assert_eq!(query_tokens("go"), ["go"]);
    assert_eq!(query_tokens(" ... "), [] as [String; 0]);
}

#[test]
fn coding_front_matter_round_trips() {
    let content = "---\nname: build\ndescription: how to build\ntype: project\n---\n\ncargo build";
    let chunk = CodingChunk::parse(content).expect("valid chunk");

    assert_eq!(chunk.front_matter.get("name"), Some("build"));
    assert_eq!(chunk.body, "cargo build");
    assert!(chunk.front_matter.get("created_at").is_some());
    let rendered = render_coding_document(&chunk.front_matter, &chunk.body);
    assert!(rendered.starts_with("---\nname: build\ndescription: how to build\ntype: project\n"));
    assert!(rendered.ends_with("---\n\ncargo build"));
}

#[test]
fn coding_front_matter_reports_the_python_messages() {
    let error = CodingChunk::parse("no front matter").expect_err("missing front matter");
    assert_eq!(error.message(), FRONT_MATTER_REQUIRED);

    let error = CodingChunk::parse("---\nname: x\ntype: project\n---\n\nbody")
        .expect_err("missing description");
    assert_eq!(error.message(), "Missing required field: description");

    let error = CodingChunk::parse("---\nname: x\ndescription: y\ntype: other\n---\n\nbody")
        .expect_err("bad type");
    assert_eq!(error.message(), CODING_TYPES_MESSAGE);

    let error = CodingChunk::parse("---\nname: x\ndescription: y\ntype: user\n---\n\n")
        .expect_err("empty body");
    assert_eq!(error.message(), EMPTY_BODY);
}

#[test]
fn daily_names_match_only_iso_dates() {
    assert!(is_daily_name("2026-11-15.md"));
    assert!(!is_daily_name("2026-11-15.MD"));
    assert!(!is_daily_name("26-11-15.md"));
    assert!(!is_daily_name("2026-11-15.md.bak"));
}
#[tokio::test]
async fn search_scopes_the_tree_and_filters_session_logs() {
    let dir = tempfile::tempdir().expect("temp workspace");
    for (path, text) in [
        (
            "memory/MEMORY.md",
            "# Memory\n\nArgoCD deploys the service.",
        ),
        (
            "memory/sessions/s1/log.jsonl",
            "{\"text\":\"ArgoCD rollout\"}",
        ),
        (
            "memory/sessions/s2/log.jsonl",
            "{\"text\":\"ArgoCD rollback\"}",
        ),
        (
            "coding_memory/build.md",
            "---\nname: build\ndescription: d\ntype: project\n---\n\nArgoCD",
        ),
        ("USER.md", "# User\n\nArgoCD owner."),
    ] {
        let path = dir.path().join(path);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
        std::fs::write(path, text).expect("fixture");
    }
    let workspace = Workspace::for_language(dir.path(), "en");
    let backend = WorkspaceMemoryBackend::new(workspace);

    let request = |scope: MemoryScope, session_key: Option<&str>| MemorySearchRequest {
        query: "argocd".to_owned(),
        max_results: 20,
        min_score: 0.0,
        session_key: session_key.map(str::to_owned),
        scope,
    };

    let memory = backend
        .search(request(MemoryScope::Memory, None))
        .await
        .expect("memory search");
    let paths: Vec<&str> = memory["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|hit| hit["path"].as_str())
        .collect();
    assert_eq!(
        paths,
        [
            "USER.md",
            "memory/MEMORY.md",
            "memory/sessions/s1/log.jsonl",
            "memory/sessions/s2/log.jsonl",
        ],
        "the long-term scope searches the memory tree plus USER.md"
    );

    let scoped = backend
        .search(request(MemoryScope::Memory, Some("s1")))
        .await
        .expect("scoped search");
    let paths: Vec<&str> = scoped["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|hit| hit["path"].as_str())
        .collect();
    assert_eq!(
        paths,
        [
            "USER.md",
            "memory/MEMORY.md",
            "memory/sessions/s1/log.jsonl"
        ],
        "session_key keeps only its own transcript"
    );

    let coding = backend
        .search(request(MemoryScope::CodingMemory, None))
        .await
        .expect("coding search");
    let paths: Vec<&str> = coding["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|hit| hit["path"].as_str())
        .collect();
    assert_eq!(paths, ["coding_memory/build.md"]);
}

#[tokio::test]
async fn search_of_a_missing_tree_is_empty_not_an_error() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let backend = WorkspaceMemoryBackend::new(Workspace::for_language(dir.path(), "en"));

    for scope in [MemoryScope::Memory, MemoryScope::CodingMemory] {
        let data = backend
            .search(MemorySearchRequest {
                query: "anything".to_owned(),
                max_results: 5,
                min_score: 0.0,
                session_key: None,
                scope,
            })
            .await
            .expect("search of a missing tree");
        assert_eq!(data["results"].as_array().map(Vec::len), Some(0));
        assert_eq!(data["disabled"], serde_json::json!(false));
    }
}
