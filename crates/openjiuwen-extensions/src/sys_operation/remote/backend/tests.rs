//! Unit tests of the provider-neutral backend wrappers.
//!
//! Python counterparts: the host-side behavior the three providers in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` share —
//! `read_file` line selection, newline transforms, `write_file` guards, transfer I/O,
//! listing/search shaping, and the shell/code output and timeout rules. The wire
//! contracts live in the per-provider suites.
// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;

use openjiuwen_core::sys_operation::code::types::CodeExecParams;
use openjiuwen_core::sys_operation::fs::types::{
    DownloadFileParams, FileContent, ListDirsParams, ListFilesParams, ReadChunkData,
    ReadFileParams, ReadMode, SearchFilesParams, SortBy, UploadFileParams, WriteFileParams,
};
use openjiuwen_core::sys_operation::operation::{
    CodeOperation, FsOperation, OpStream, ShellOperation,
};
use openjiuwen_core::sys_operation::result::OpResult;
use openjiuwen_core::sys_operation::shell::types::{ShellArgvParams, ShellExecParams, StreamKind};

use super::*;

use tokio::fs;
use tokio::sync::Mutex;

/// An in-memory backend with scripted executions.
#[derive(Default)]
struct FakeBackend {
    conditional_write: bool,
    files: Mutex<BTreeMap<String, Vec<u8>>>,
    entries: Mutex<Vec<RemoteEntry>>,
    search_entries: Mutex<Vec<RemoteEntry>>,
    executions: Mutex<Vec<ExecResult>>,
    requests: Mutex<Vec<ExecRequest>>,
}

impl FakeBackend {
    fn new() -> Self {
        Self::default()
    }

    /// A backend that advertises the conditional-write capability.
    fn conditional() -> Self {
        Self {
            conditional_write: true,
            ..Self::default()
        }
    }

    async fn put(&self, path: &str, data: &[u8]) {
        self.files
            .lock()
            .await
            .insert(path.to_string(), data.to_vec());
    }

    async fn contents(&self, path: &str) -> Option<Vec<u8>> {
        self.files.lock().await.get(path).cloned()
    }

    async fn set_entries(&self, entries: Vec<RemoteEntry>) {
        *self.entries.lock().await = entries;
    }

    async fn set_search_entries(&self, entries: Vec<RemoteEntry>) {
        *self.search_entries.lock().await = entries;
    }

    async fn script(&self, results: Vec<ExecResult>) {
        *self.executions.lock().await = results;
    }

    async fn requests(&self) -> Vec<ExecRequest> {
        self.requests.lock().await.clone()
    }

    async fn last_request(&self) -> ExecRequest {
        self.requests.lock().await.last().cloned().unwrap()
    }
}

#[async_trait]
impl SandboxBackend for FakeBackend {
    fn provider_name(&self) -> &'static str {
        "fake"
    }

    fn supports_conditional_write(&self) -> bool {
        self.conditional_write
    }

    async fn read_bytes(&self, path: &str) -> std::result::Result<Vec<u8>, RemoteError> {
        self.files
            .lock()
            .await
            .get(path)
            .cloned()
            .ok_or_else(|| RemoteError::NotFound(format!("{path} not found")))
    }

    async fn write_bytes(
        &self,
        path: &str,
        data: &[u8],
        append: bool,
    ) -> std::result::Result<(), RemoteError> {
        let mut files = self.files.lock().await;
        if append {
            files
                .entry(path.to_string())
                .or_default()
                .extend_from_slice(data);
        } else {
            files.insert(path.to_string(), data.to_vec());
        }
        Ok(())
    }

    async fn list(
        &self,
        request: &ListRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
        Ok(self
            .entries
            .lock()
            .await
            .iter()
            .filter(|entry| {
                if entry.is_directory {
                    request.include_dirs
                } else {
                    request.include_files
                }
            })
            .cloned()
            .collect())
    }

    async fn search(
        &self,
        _request: &SearchRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
        Ok(self.search_entries.lock().await.clone())
    }

    async fn execute(&self, request: &ExecRequest) -> std::result::Result<ExecResult, RemoteError> {
        self.requests.lock().await.push(request.clone());
        let mut scripted = self.executions.lock().await;
        if scripted.is_empty() {
            Ok(ExecResult::default())
        } else {
            Ok(scripted.remove(0))
        }
    }
}

fn fs_backend(backend: &Arc<FakeBackend>) -> RemoteFs<FakeBackend> {
    RemoteFs::new(Arc::clone(backend))
}

fn entry(name: &str, size: u64, modified_time: &str, file_type: Option<&str>) -> RemoteEntry {
    RemoteEntry {
        name: name.to_string(),
        path: format!("/work/{name}"),
        size,
        modified_time: modified_time.to_string(),
        is_directory: file_type.is_none(),
        file_type: file_type.map(str::to_string),
    }
}

fn chunk_texts(chunks: &[OpResult<ReadChunkData>]) -> Vec<String> {
    chunks
        .iter()
        .map(|chunk| match &chunk.data.as_ref().unwrap().chunk {
            FileContent::Text(text) => text.clone(),
            FileContent::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        })
        .collect()
}

async fn collect<T>(stream: OpStream<T>) -> Vec<OpResult<T>> {
    stream.collect().await
}

#[tokio::test]
async fn text_read_applies_head_tail_and_line_range() {
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/a.txt", b"one\ntwo\nthree\nfour\n").await;
    let fs = fs_backend(&backend);

    let head = fs
        .read_file(ReadFileParams::text("/work/a.txt").with_head(2))
        .await;
    assert_eq!(head.code, 0);
    assert_eq!(
        head.data.unwrap().content,
        FileContent::Text("one\ntwo\n".to_string())
    );

    let tail = fs
        .read_file(ReadFileParams::text("/work/a.txt").with_tail(1))
        .await;
    assert_eq!(
        tail.data.unwrap().content,
        FileContent::Text("four\n".to_string())
    );

    let range = fs
        .read_file(ReadFileParams::text("/work/a.txt").with_line_range(2, 3))
        .await;
    let data = range.data.unwrap();
    assert_eq!(data.content, FileContent::Text("two\nthree\n".to_string()));
    assert_eq!(data.total_bytes, 19);
    assert_eq!(data.encoding, None);
}

#[tokio::test]
async fn text_read_reports_a_non_default_encoding() {
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/a.txt", b"hi").await;
    let fs = fs_backend(&backend);
    let result = fs
        .read_file(ReadFileParams::text("/work/a.txt").with_encoding("gbk"))
        .await;
    assert_eq!(result.data.unwrap().encoding, Some("gbk".to_string()));
}

#[tokio::test]
async fn read_validation_errors_use_the_fs_status() {
    let backend = Arc::new(FakeBackend::new());
    let fs = fs_backend(&backend);

    let bytes_head = fs
        .read_file(ReadFileParams::bytes("/work/a.bin").with_head(1))
        .await;
    assert_eq!(bytes_head.code, 199_003);
    assert!(bytes_head
        .message
        .contains("Parameters 'head', 'tail', and 'line_range' are only supported in text mode"));

    let both = fs
        .read_file(
            ReadFileParams::text("/work/a.txt")
                .with_head(1)
                .with_tail(1),
        )
        .await;
    assert_eq!(both.code, 199_003);
    assert!(both
        .message
        .contains("head and tail cannot be specified simultaneously"));
}

#[tokio::test]
async fn a_missing_file_is_a_gateway_failure() {
    let backend = Arc::new(FakeBackend::new());
    let fs = fs_backend(&backend);
    let result = fs
        .read_file(ReadFileParams::text("/work/missing.txt"))
        .await;
    assert_eq!(result.code, 199_007);
    assert_eq!(
        result.message,
        "sandbox gateway error, operation: read_file, reason: /work/missing.txt not found"
    );
}

#[tokio::test]
async fn bytes_read_returns_the_raw_payload() {
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/a.bin", &[0_u8, 1, 2]).await;
    let fs = fs_backend(&backend);
    let result = fs.read_file(ReadFileParams::bytes("/work/a.bin")).await;
    let data = result.data.unwrap();
    assert_eq!(data.mode, ReadMode::Bytes);
    assert_eq!(data.content, FileContent::Bytes(vec![0, 1, 2]));
    assert_eq!(data.total_bytes, 3);
    assert!(!data.truncated);
}

#[tokio::test]
async fn read_stream_emits_one_chunk_per_selected_line() {
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/a.txt", b"one\ntwo\nthree\n").await;
    let fs = Arc::new(fs_backend(&backend));
    let chunks =
        collect(fs.read_file_stream(ReadFileParams::text("/work/a.txt").with_head(2))).await;

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunk_texts(&chunks), vec!["one\n", "two\n"]);
    let first = chunks[0].data.as_ref().unwrap();
    assert_eq!(first.chunk_index, 0);
    assert_eq!(first.chunk_size, 4);
    assert!(!first.is_last_chunk);
    let last = chunks[1].data.as_ref().unwrap();
    assert_eq!(last.chunk_index, 1);
    assert!(last.is_last_chunk);
}

#[tokio::test]
async fn read_stream_splits_bytes_by_chunk_size() {
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/a.bin", b"abcde").await;
    let fs = Arc::new(fs_backend(&backend));
    let mut params = ReadFileParams::bytes("/work/a.bin");
    params.chunk_size = 2;
    let chunks = collect(fs.read_file_stream(params)).await;

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunk_texts(&chunks), vec!["ab", "cd", "e"]);
    let sizes: Vec<u64> = chunks
        .iter()
        .map(|chunk| chunk.data.as_ref().unwrap().chunk_size)
        .collect();
    assert_eq!(sizes, vec![2, 2, 1]);
    assert!(chunks[2].data.as_ref().unwrap().is_last_chunk);
    assert_eq!(chunks[0].data.as_ref().unwrap().mode, ReadMode::Bytes);
}

#[tokio::test]
async fn an_empty_stream_always_yields_one_terminal_chunk() {
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/empty.txt", b"").await;
    backend.put("/work/empty.bin", b"").await;
    let fs = Arc::new(fs_backend(&backend));

    let text = collect(
        fs.clone()
            .read_file_stream(ReadFileParams::text("/work/empty.txt")),
    )
    .await;
    assert_eq!(text.len(), 1);
    let data = text[0].data.as_ref().unwrap();
    assert!(data.is_last_chunk);
    assert_eq!(data.chunk, FileContent::Text(String::new()));

    let bytes = collect(fs.read_file_stream(ReadFileParams::bytes("/work/empty.bin"))).await;
    assert_eq!(bytes.len(), 1);
    let data = bytes[0].data.as_ref().unwrap();
    assert!(data.is_last_chunk);
    assert_eq!(data.chunk, FileContent::Bytes(Vec::new()));
}

#[tokio::test]
async fn a_failed_stream_yields_exactly_one_terminal_chunk() {
    let backend = Arc::new(FakeBackend::new());
    let fs = Arc::new(fs_backend(&backend));
    let chunks = collect(fs.read_file_stream(ReadFileParams::text("/work/missing.txt"))).await;

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].code, 199_007);
    let data = chunks[0].data.as_ref().unwrap();
    assert_eq!(data.path, "/work/missing.txt");
    assert!(data.is_last_chunk);
    assert_eq!(data.mode, ReadMode::Text);
}

#[tokio::test]
async fn write_file_applies_python_newline_defaults_and_append() {
    let backend = Arc::new(FakeBackend::new());
    let fs = fs_backend(&backend);

    let written = fs
        .write_file(WriteFileParams::new("/work/a.txt", "hello"))
        .await;
    let data = written.data.unwrap();
    assert_eq!(data.size, 6);
    assert_eq!(data.mode, ReadMode::Text);
    assert_eq!(backend.contents("/work/a.txt").await.unwrap(), b"\nhello");

    let appended = fs
        .write_file(WriteFileParams::raw("/work/a.txt", "more").with_append())
        .await;
    assert_eq!(appended.code, 0);
    assert_eq!(
        backend.contents("/work/a.txt").await.unwrap(),
        b"\nhellomore"
    );

    let trailing = fs
        .write_file({
            let mut params = WriteFileParams::raw("/work/b.txt", "x");
            params.append_newline = true;
            params
        })
        .await;
    assert_eq!(trailing.data.unwrap().size, 2);
    assert_eq!(backend.contents("/work/b.txt").await.unwrap(), b"x\n");
}

#[tokio::test]
async fn write_file_enforces_create_if_not_exist() {
    let backend = Arc::new(FakeBackend::new());
    let fs = fs_backend(&backend);
    let params = {
        let mut params = WriteFileParams::raw("/work/new.txt", "x");
        params.create_if_not_exist = false;
        params
    };
    let result = fs.write_file(params).await;
    assert_eq!(result.code, 199_003);
    assert!(result
        .message
        .contains("File does not exist: /work/new.txt"));
    assert!(backend.contents("/work/new.txt").await.is_none());
}

#[tokio::test]
async fn conditional_writes_are_rejected_with_the_provider_name() {
    let backend = Arc::new(FakeBackend::new());
    let fs = fs_backend(&backend);

    let sha = fs
        .write_file(WriteFileParams::raw("/work/a.txt", "x").with_expected_sha256("deadbeef"))
        .await;
    assert_eq!(sha.code, 199_003);
    assert_eq!(
        sha.message,
        "file system operation execution error, execution: write_file, reason: \
         expected_content_sha256 is not supported by the fake provider (remote sandbox \
         transports have no compare-and-swap)"
    );

    let absent = fs
        .write_file({
            let mut params = WriteFileParams::raw("/work/a.txt", "x");
            params.create_if_absent = true;
            params
        })
        .await;
    assert_eq!(absent.code, 199_003);
    assert!(absent
        .message
        .contains("create_if_absent is not supported by the fake provider"));
}

#[tokio::test]
async fn upload_and_download_round_trip_through_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    fs::write(&local, b"payload").await.unwrap();
    let backend = Arc::new(FakeBackend::new());
    let fs_operation = Arc::new(fs_backend(&backend));

    let upload = fs_operation
        .upload_file(UploadFileParams {
            local_path: local.to_string_lossy().into_owned(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;
    assert_eq!(upload.code, 0);
    assert_eq!(upload.data.as_ref().unwrap().size, 7);
    assert_eq!(
        backend.contents("/work/payload.bin").await.unwrap(),
        b"payload"
    );

    let target = dir.path().join("out.bin");
    let download = fs_operation
        .download_file(DownloadFileParams {
            source_path: "/work/payload.bin".to_string(),
            local_path: target.to_string_lossy().into_owned(),
            ..DownloadFileParams::default()
        })
        .await;
    assert_eq!(download.code, 0);
    assert_eq!(fs::read(&target).await.unwrap(), b"payload");

    let chunk = collect(
        fs_operation.download_file_stream(DownloadFileParams {
            source_path: "/work/payload.bin".to_string(),
            local_path: dir
                .path()
                .join("nested/stream.bin")
                .to_string_lossy()
                .into_owned(),
            create_parent_dirs: true,
            overwrite: true,
            ..DownloadFileParams::default()
        }),
    )
    .await;
    assert_eq!(chunk.len(), 1);
    let data = chunk[0].data.as_ref().unwrap();
    assert_eq!(data.chunk_size, 7);
    assert!(data.is_last_chunk);
    assert_eq!(
        fs::read(dir.path().join("nested/stream.bin"))
            .await
            .unwrap(),
        b"payload"
    );
}

#[tokio::test]
async fn transfers_refuse_existing_destinations() {
    let dir = tempfile::tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    fs::write(&local, b"payload").await.unwrap();
    let backend = Arc::new(FakeBackend::new());
    backend.put("/work/payload.bin", b"remote").await;
    let fs_operation = fs_backend(&backend);

    let upload = fs_operation
        .upload_file(UploadFileParams {
            local_path: local.to_string_lossy().into_owned(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;
    assert_eq!(upload.code, 199_003);
    assert!(upload
        .message
        .contains("File already exists: /work/payload.bin"));

    let existing = dir.path().join("existing.bin");
    fs::write(&existing, b"old").await.unwrap();
    let download = fs_operation
        .download_file(DownloadFileParams {
            source_path: "/work/payload.bin".to_string(),
            local_path: existing.to_string_lossy().into_owned(),
            ..DownloadFileParams::default()
        })
        .await;
    assert_eq!(download.code, 199_003);
    assert!(download.message.contains("File already exists"));
    assert_eq!(fs::read(&existing).await.unwrap(), b"old");
}

#[tokio::test]
async fn upload_reports_an_unreadable_local_file() {
    let backend = Arc::new(FakeBackend::new());
    let fs_operation = fs_backend(&backend);
    let result = fs_operation
        .upload_file(UploadFileParams {
            local_path: "/definitely/missing/payload.bin".to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;
    assert_eq!(result.code, 199_003);
    assert!(result.message.contains(
        "failed to read local file '/definitely/missing/payload.bin': \
         failed to read host file '/definitely/missing/payload.bin'"
    ));
}

#[tokio::test]
async fn list_files_sorts_and_filters() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .set_entries(vec![
            entry("b.rs", 30, "2024-01-02 00:00:00", Some(".rs")),
            entry("a.txt", 10, "2024-01-03 00:00:00", Some(".txt")),
            entry("sub", 512, "2024-01-01 00:00:00", None),
        ])
        .await;
    let fs_operation = fs_backend(&backend);

    let mut params = ListFilesParams::new("/work");
    params.sort_by = SortBy::Size;
    params.sort_descending = true;
    let listed = fs_operation.list_files(params).await;
    let data = listed.data.unwrap();
    assert_eq!(data.total_count, 2, "directories are excluded");
    assert_eq!(
        data.list_items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        vec!["b.rs", "a.txt"]
    );

    let filtered = fs_operation
        .list_files({
            let mut params = ListFilesParams::new("/work");
            params.file_types = Some(vec![".txt".to_string()]);
            params
        })
        .await;
    let data = filtered.data.unwrap();
    assert_eq!(data.total_count, 1);
    assert_eq!(data.list_items[0].path, "/work/a.txt");
    assert_eq!(data.root_path, "/work");
}

#[tokio::test]
async fn list_directories_returns_only_directories() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .set_entries(vec![
            entry("a.txt", 10, "2024-01-03 00:00:00", Some(".txt")),
            entry("sub", 512, "2024-01-01 00:00:00", None),
        ])
        .await;
    let fs_operation = fs_backend(&backend);
    let listed = fs_operation
        .list_directories(ListDirsParams::new("/work"))
        .await;
    let data = listed.data.unwrap();
    assert_eq!(data.total_count, 1);
    assert!(data.list_items[0].is_directory);
    assert_eq!(data.list_items[0].size, 512);
    assert_eq!(data.list_items[0].file_type, None);
}

#[tokio::test]
async fn search_files_maps_entries_and_sorts_by_name() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .set_search_entries(vec![
            entry("b.txt", 2, "2024-01-01 00:00:00", Some(".txt")),
            entry("a.txt", 1, "2024-01-01 00:00:00", Some(".txt")),
        ])
        .await;
    let fs_operation = fs_backend(&backend);
    let result = fs_operation
        .search_files(SearchFilesParams::new("/work", "*.txt").with_excludes(["*.rs"]))
        .await;
    let data = result.data.unwrap();
    assert_eq!(data.total_matches, 2);
    assert_eq!(
        data.matching_files
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        vec!["a.txt", "b.txt"]
    );
    assert_eq!(data.search_path, "/work");
    assert_eq!(data.search_pattern, "*.txt");
    assert_eq!(data.exclude_patterns, Some(vec!["*.rs".to_string()]));
}

#[tokio::test]
async fn execute_cmd_reports_success_and_non_zero_exits_as_values() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 7,
            stdout: "out".to_string(),
            stderr: "err".to_string(),
            timed_out: false,
        }])
        .await;
    let shell = RemoteShell::new(Arc::clone(&backend));
    let params = ShellExecParams::new("false")
        .with_cwd(".")
        .with_timeout_secs(12);
    let result = shell.execute_cmd(params).await;

    assert_eq!(result.code, 0);
    let output = result.data.unwrap();
    assert_eq!(output.exit_code, 7);
    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
    assert_eq!(output.cwd, ".");
    assert!(!output.timed_out);
    assert!(!output.truncated);
    assert!(!output.cancelled);

    let request = backend.last_request().await;
    assert_eq!(request.command, ExecCommand::Shell("false".to_string()));
    assert_eq!(request.cwd, None, "'.' is normalized away");
    assert_eq!(request.timeout_secs, Some(12));
}

#[tokio::test]
async fn execute_cmd_rejects_an_empty_command() {
    let backend = Arc::new(FakeBackend::new());
    let shell = RemoteShell::new(Arc::clone(&backend));
    let result = shell.execute_cmd(ShellExecParams::new("   ")).await;
    assert_eq!(result.code, 199_004);
    assert!(result.message.contains("command can not be empty"));
    assert!(result.data.is_none());
    assert!(backend.requests().await.is_empty());
}

#[tokio::test]
async fn a_timeout_is_a_shell_failure_with_data() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 124,
            stdout: "partial".to_string(),
            stderr: String::new(),
            timed_out: true,
        }])
        .await;
    let shell = RemoteShell::new(Arc::clone(&backend));
    let result = shell
        .execute_cmd(ShellExecParams::new("sleep 5").with_timeout_secs(5))
        .await;

    assert_eq!(result.code, 199_004);
    assert_eq!(result.message, "execution timeout after 5 seconds");
    let output = result.data.unwrap();
    assert!(output.timed_out);
    assert_eq!(output.exit_code, 124);
    assert_eq!(output.stdout, "partial");
}

#[tokio::test]
async fn execute_argv_quotes_the_command_line() {
    let backend = Arc::new(FakeBackend::new());
    let shell = RemoteShell::new(Arc::clone(&backend));
    let result = shell
        .execute_argv(ShellArgvParams::new(["echo", "hello world"]))
        .await;

    assert_eq!(result.code, 0);
    assert_eq!(result.data.unwrap().command, "echo 'hello world'");
    let request = backend.last_request().await;
    assert_eq!(
        request.command,
        ExecCommand::Argv(vec!["echo".to_string(), "hello world".to_string()])
    );

    let empty = shell.execute_argv(ShellArgvParams::default()).await;
    assert_eq!(empty.code, 199_004);
    assert!(empty.message.contains("argv can not be empty"));
}

#[tokio::test]
async fn execute_cmd_background_parses_the_pid() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 0,
            stdout: "4321\n".to_string(),
            stderr: String::new(),
            timed_out: false,
        }])
        .await;
    let shell = RemoteShell::new(Arc::clone(&backend));
    let result = shell
        .execute_cmd_background(ShellExecParams::new("sleep 30"))
        .await;

    assert_eq!(result.code, 0);
    let output = result.data.unwrap();
    assert_eq!(output.pid, Some(4321));
    assert_eq!(output.command, "sleep 30");
    assert_eq!(output.cwd, ".");
    assert_eq!(output.error, None);

    let request = backend.last_request().await;
    assert_eq!(request.timeout_secs, Some(10));
    let ExecCommand::Shell(command) = request.command else {
        panic!("expected a shell command");
    };
    assert!(command.contains("echo $!"));
    assert!(command.contains("sleep 30"));
}

#[tokio::test]
async fn execute_cmd_background_surfaces_failures() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 3,
            stdout: String::new(),
            stderr: "boom".to_string(),
            timed_out: false,
        }])
        .await;
    let shell = RemoteShell::new(Arc::clone(&backend));
    let failed = shell
        .execute_cmd_background(ShellExecParams::new("bad"))
        .await;
    assert_eq!(failed.code, 199_004);
    assert!(failed.message.contains("exit code 3"));
    assert!(failed.message.contains("boom"));

    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 0,
            stdout: "not-a-pid".to_string(),
            stderr: String::new(),
            timed_out: false,
        }])
        .await;
    let shell = RemoteShell::new(Arc::clone(&backend));
    let unparsable = shell
        .execute_cmd_background(ShellExecParams::new("weird"))
        .await;
    assert_eq!(unparsable.code, 199_004);
    assert!(unparsable.message.contains("background pid"));

    let empty = shell
        .execute_cmd_background(ShellExecParams::new(" "))
        .await;
    assert_eq!(empty.code, 199_004);
    assert!(empty.message.contains("command can not be empty"));
}

#[tokio::test]
async fn shell_stream_orders_stdout_stderr_and_the_terminal_chunk() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 2,
            stdout: "a\nb\n".to_string(),
            stderr: "e\n".to_string(),
            timed_out: false,
        }])
        .await;
    let shell = Arc::new(RemoteShell::new(Arc::clone(&backend)));
    let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("cmd"))).await;

    assert_eq!(chunks.len(), 4);
    let data: Vec<_> = chunks
        .iter()
        .map(|chunk| chunk.data.as_ref().unwrap())
        .collect();
    assert_eq!(data[0].text, "a\n");
    assert_eq!(data[0].stream, Some(StreamKind::Stdout));
    assert_eq!(data[1].text, "b\n");
    assert_eq!(data[1].stream, Some(StreamKind::Stdout));
    assert_eq!(data[2].text, "e\n");
    assert_eq!(data[2].stream, Some(StreamKind::Stderr));
    assert_eq!(data[3].chunk_index, 3);
    assert_eq!(data[3].exit_code, Some(2));
    assert_eq!(
        data.iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[tokio::test]
async fn an_empty_shell_stream_still_terminates() {
    let backend = Arc::new(FakeBackend::new());
    let shell = Arc::new(RemoteShell::new(Arc::clone(&backend)));
    let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("true"))).await;

    assert_eq!(chunks.len(), 1);
    let data = chunks[0].data.as_ref().unwrap();
    assert_eq!(data.chunk_index, 0);
    assert_eq!(data.exit_code, Some(0));
}

#[tokio::test]
async fn a_failed_shell_stream_yields_one_terminal_chunk() {
    let backend = Arc::new(FakeBackend::new());
    let shell = Arc::new(RemoteShell::new(Arc::clone(&backend)));
    let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new(""))).await;

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].code, 199_004);
    assert!(chunks[0].message.contains("command can not be empty"));
    let data = chunks[0].data.as_ref().unwrap();
    assert_eq!(data.exit_code, Some(-1));
    assert_eq!(data.chunk_index, 0);
}

#[tokio::test]
async fn execute_code_uses_inline_argv_and_forced_environment_defaults() {
    let backend = Arc::new(FakeBackend::new());
    let code = RemoteCode::new(Arc::clone(&backend));
    let params = CodeExecParams::new("print(1)").with_cwd("/custom");
    let mut params = params;
    params
        .environment
        .insert("PYTHONUTF8".to_string(), Some("0".to_string()));
    params
        .environment
        .insert("EXTRA".to_string(), Some("1".to_string()));
    let result = code.execute_code(params).await;

    assert_eq!(result.code, 0);
    assert_eq!(result.data.unwrap().language, "python");

    let request = backend.last_request().await;
    assert_eq!(
        request.command,
        ExecCommand::Argv(vec![
            "python3".to_string(),
            "-u".to_string(),
            "-c".to_string(),
            "print(1)".to_string()
        ])
    );
    assert_eq!(request.cwd, Some("/tmp".to_string()));
    assert_eq!(request.timeout_secs, Some(300));
    assert_eq!(
        request.environment.get("PYTHONIOENCODING"),
        Some(&Some("utf-8".to_string()))
    );
    assert_eq!(
        request.environment.get("PYTHONUTF8"),
        Some(&Some("0".to_string())),
        "the caller wins over the forced default"
    );
    assert_eq!(
        request.environment.get("EXTRA"),
        Some(&Some("1".to_string()))
    );
}

#[tokio::test]
async fn execute_code_rejects_an_empty_snippet() {
    let backend = Arc::new(FakeBackend::new());
    let code = RemoteCode::new(Arc::clone(&backend));
    let result = code.execute_code(CodeExecParams::new(" \n")).await;

    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "code can not be empty");
    let output = result.data.unwrap();
    assert_eq!(output.exit_code, -1);
    assert_eq!(output.language, "python");
    assert!(backend.requests().await.is_empty());
}

#[tokio::test]
async fn force_file_writes_a_temporary_script() {
    let backend = Arc::new(FakeBackend::new());
    let code = RemoteCode::new(Arc::clone(&backend));
    let params = CodeExecParams::javascript("console.log(1)").with_force_file(true);
    code.execute_code(params).await;

    let request = backend.last_request().await;
    let ExecCommand::Shell(command) = request.command else {
        panic!("force_file must use a shell command");
    };
    assert!(command.contains("/tmp/ojw_code_XXXXXX.js"), "{command}");
    assert!(command.contains("base64 -d"), "{command}");
    assert!(command.contains("node \"$tmp\""), "{command}");
    let encoded = command
        .split("printf %s ")
        .nth(1)
        .unwrap()
        .split(" | base64 -d")
        .next()
        .unwrap();
    assert_eq!(base64_decode_for_test(encoded), "console.log(1)");
}

fn base64_decode_for_test(value: &str) -> String {
    use base64::Engine as _;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value.trim_matches('\''))
        .unwrap();
    String::from_utf8(decoded).unwrap()
}

#[tokio::test]
async fn a_code_timeout_is_a_code_failure_with_data() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 124,
            stdout: "partial".to_string(),
            stderr: String::new(),
            timed_out: true,
        }])
        .await;
    let code = RemoteCode::new(Arc::clone(&backend));
    let result = code
        .execute_code(CodeExecParams::new("while True: pass").with_timeout_secs(4))
        .await;

    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "execution timeout after 4 seconds");
    assert!(result.data.unwrap().timed_out);
}

#[tokio::test]
async fn code_stream_orders_chunks_and_terminates() {
    let backend = Arc::new(FakeBackend::new());
    backend
        .script(vec![ExecResult {
            exit_code: 0,
            stdout: "x\ny\n".to_string(),
            stderr: "z\n".to_string(),
            timed_out: false,
        }])
        .await;
    let code = Arc::new(RemoteCode::new(Arc::clone(&backend)));
    let chunks = collect(code.execute_code_stream(CodeExecParams::new("print(1)"))).await;

    assert_eq!(chunks.len(), 4);
    let data: Vec<_> = chunks
        .iter()
        .map(|chunk| chunk.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        data.iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(data[0].stream, Some(StreamKind::Stdout));
    assert_eq!(data[2].stream, Some(StreamKind::Stderr));
    assert_eq!(data[3].exit_code, Some(0));
}

#[tokio::test]
async fn a_failed_code_stream_yields_one_terminal_chunk() {
    let backend = Arc::new(FakeBackend::new());
    let code = Arc::new(RemoteCode::new(Arc::clone(&backend)));
    let chunks = collect(code.execute_code_stream(CodeExecParams::new(""))).await;

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].code, 199_005);
    assert_eq!(chunks[0].message, "code can not be empty");
    assert_eq!(chunks[0].data.as_ref().unwrap().exit_code, Some(-1));
}

#[tokio::test]
async fn provider_set_from_registers_three_factories_and_reports_the_backend_capability() {
    let set = provider_set_from(
        |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| Ok(FakeBackend::new()),
    );
    assert!(set.fs.is_some());
    assert!(set.shell.is_some());
    assert!(set.code.is_some());
    // The capability is the backend's: the gateway flag does not gate it,
    // because the precondition is evaluated over the backend's own read and
    // write primitives rather than a gateway compare-and-swap.
    let conditional = SandboxGatewayConfig {
        supports_conditional_write: false,
        ..SandboxGatewayConfig::default()
    };
    let capable = provider_set_from(
        |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| Ok(FakeBackend::conditional()),
    );
    let fs = capable.fs.as_ref().unwrap();
    let fs = fs(&SandboxEndpoint::default(), &conditional).unwrap();
    assert!(fs.supports_conditional_write());
    let plain = set.fs.as_ref().unwrap();
    let plain = plain(
        &SandboxEndpoint::default(),
        &SandboxGatewayConfig::default(),
    )
    .unwrap();
    assert!(!plain.supports_conditional_write());
}
