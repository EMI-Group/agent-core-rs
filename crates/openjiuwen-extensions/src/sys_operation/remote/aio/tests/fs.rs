//! Filesystem wire-contract tests of the `aio` provider.
//!
//! Python counterpart: `tests/unit_tests/extensions/sys_operation/sandbox/test_aio_*.py`
//! (registration only); these pin the envelope `params` of the Python `fs`
//! signatures, the mapping of `data` back into the kernel structs, the host-side
//! transfer I/O, and the chunking of streamed reads.
use super::*;

// ─── fs parameters ──────────────────────────────────────────────────────────

#[tokio::test]
async fn read_file_sends_the_python_signature_parameters() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": "hi"})).await;
    let mut params = ReadFileParams::text("/work/a.txt");
    params.line_range = Some(LineRange::new(2, 4));

    let result = fs(&server).read_file(params).await;

    assert_eq!(result.code, 0);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work/a.txt",
            "mode": "text",
            "head": null,
            "tail": null,
            "line_range": [2, 4],
            "encoding": "utf-8",
            "chunk_size": 0,
        })
    );
}

#[tokio::test]
async fn read_file_validates_the_line_selection_before_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = fs(&server)
        .read_file(ReadFileParams::bytes("/work/a.bin").with_head(2))
        .await;

    assert_eq!(result.code, 199_003);
    assert!(result.message.contains("only supported in text mode"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn write_file_sends_base64_bytes_and_the_newline_flags() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"path": "/work/a.bin", "size": 3, "mode": "bytes"}),
    )
    .await;

    let result = fs(&server)
        .write_file(WriteFileParams::new(
            "/work/a.bin",
            FileContent::Bytes(vec![1, 2, 250]),
        ))
        .await;

    assert_eq!(result.code, 0);
    let data: WriteFileData = result.data.unwrap();
    assert_eq!(data.path, "/work/a.bin");
    assert_eq!(data.size, 3);
    assert_eq!(data.mode, ReadMode::Bytes);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work/a.bin",
            "content": "AQL6",
            "mode": "bytes",
            "prepend_newline": true,
            "append_newline": false,
            "append": false,
            "create_if_not_exist": true,
            "permissions": "644",
            "encoding": "base64",
        })
    );
}

#[tokio::test]
async fn write_file_sends_text_content_and_the_request_encoding() {
    let server = MockServer::start().await;
    respond_data(&server, json!({})).await;
    let mut params = WriteFileParams::raw("/work/a.txt", "hello");
    params.append_newline = true;
    params.append = true;
    params.create_if_not_exist = false;

    let result = fs(&server).write_file(params).await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["params"]["content"], "hello");
    assert_eq!(envelope["params"]["encoding"], "utf-8");
    assert_eq!(envelope["params"]["append_newline"], true);
    assert_eq!(envelope["params"]["append"], true);
    assert_eq!(envelope["params"]["create_if_not_exist"], false);
    assert_eq!(envelope["params"]["prepend_newline"], false);
}

#[tokio::test]
async fn conditional_writes_are_rejected_before_calling_the_gateway() {
    let server = MockServer::start().await;
    let fs = fs(&server);

    let digest = fs
        .write_file(WriteFileParams::raw("/work/a.txt", "hi").with_expected_sha256("ab"))
        .await;
    assert_eq!(digest.code, 199_003);
    assert_eq!(
        digest.message,
        "file system operation execution error, execution: write_file, \
         reason: expected_content_sha256 is not supported by the aio provider \
         (remote sandbox transports have no compare-and-swap)"
    );

    let mut absent = WriteFileParams::raw("/work/a.txt", "hi");
    absent.create_if_absent = true;
    let absent = fs.write_file(absent).await;
    assert_eq!(absent.code, 199_003);
    assert!(absent
        .message
        .contains("create_if_absent is not supported by the aio provider"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn listing_sends_the_sort_fields_and_the_type_filter() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"total_count": 0, "list_items": []})).await;
    let mut params = ListFilesParams::new("/work").recursive(Some(2));
    params.sort_by = SortBy::ModifiedTime;
    params.sort_descending = true;
    params.file_types = Some(vec![".txt".to_string()]);

    let result = fs(&server).list_files(params).await;

    assert_eq!(result.code, 0);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work",
            "recursive": true,
            "max_depth": 2,
            "sort_by": "modified_time",
            "sort_descending": true,
            "file_types": [".txt"],
        })
    );
}

#[tokio::test]
async fn listing_directories_omits_the_type_filter() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"list_items": []})).await;

    let result = fs(&server)
        .list_directories(ListDirsParams::new("/work"))
        .await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["method"], "list_directories");
    assert_eq!(
        envelope["params"],
        json!({
            "path": "/work",
            "recursive": false,
            "max_depth": null,
            "sort_by": "name",
            "sort_descending": false,
        })
    );
    assert!(envelope["params"].get("file_types").is_none());
}

#[tokio::test]
async fn searching_sends_the_pattern_and_the_exclusions() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"matching_files": []})).await;

    let result = fs(&server)
        .search_files(SearchFilesParams::new("/work", "*.rs").with_excludes(["target/*"]))
        .await;

    assert_eq!(result.code, 0);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work",
            "pattern": "*.rs",
            "exclude_patterns": ["target/*"],
        })
    );
}

// ─── fs data mapping ────────────────────────────────────────────────────────

#[tokio::test]
async fn read_file_maps_text_content() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"path": "/work/a.txt", "content": "one\ntwo\n", "mode": "text"}),
    )
    .await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;

    assert_eq!(result.code, 0);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.path, "/work/a.txt");
    assert_eq!(data.content, FileContent::Text("one\ntwo\n".to_string()));
    assert_eq!(data.mode, ReadMode::Text);
    assert_eq!(data.encoding, None);
    assert!(!data.truncated);
    assert_eq!(data.total_bytes, 8);
}

#[tokio::test]
async fn read_file_maps_a_byte_array_and_falls_back_to_the_request_mode() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": [1, 2, 255]})).await;

    let result = fs(&server)
        .read_file(ReadFileParams::bytes("/work/a.bin"))
        .await;

    assert_eq!(result.code, 0);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.content, FileContent::Bytes(vec![1, 2, 255]));
    assert_eq!(data.mode, ReadMode::Bytes);
    assert_eq!(data.path, "/work/a.bin", "the request path is the fallback");
    assert_eq!(data.total_bytes, 3);
}

#[tokio::test]
async fn a_response_without_data_is_tolerated() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": 0})).await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;

    assert_eq!(result.code, 0, "a missing code defaults to success");
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.content, FileContent::Text(String::new()));
    assert_eq!(data.total_bytes, 0);
}

#[tokio::test]
async fn write_file_falls_back_to_the_payload_size_and_the_request_mode() {
    let server = MockServer::start().await;
    respond_data(&server, json!({})).await;

    let result = fs(&server)
        .write_file(WriteFileParams::raw("/work/a.txt", "abc"))
        .await;

    assert_eq!(result.code, 0);
    let data: WriteFileData = result.data.unwrap();
    assert_eq!(data.path, "/work/a.txt");
    assert_eq!(data.size, 3, "the text payload length");
    assert_eq!(data.mode, ReadMode::Text);
}

#[tokio::test]
async fn list_files_maps_entries_filters_directories_and_sorts() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "total_count": 4,
            "root_path": "/work",
            "recursive": false,
            "max_depth": null,
            "list_items": [
                {"name": "b.txt", "path": "/work/b.txt", "size": 2,
                 "modified_time": "2024-01-02 00:00:00"},
                {"name": "a.rs", "path": "/work/a.rs", "size": 1},
                {"name": "dir", "path": "/work/dir", "is_directory": true},
                {"name": "c.md", "path": "/work/c.md", "size": 3, "type": ".md",
                 "modified_time": "2024-01-01 00:00:00"},
            ],
        }),
    )
    .await;

    let result = fs(&server).list_files(ListFilesParams::new("/work")).await;

    assert_eq!(result.code, 0);
    let data: ListFilesData = result.data.unwrap();
    assert_eq!(data.total_count, 3, "the directory is dropped");
    assert_eq!(data.root_path, "/work");
    assert!(!data.recursive);
    assert_eq!(data.max_depth, None);
    assert_eq!(names(&data.list_items), vec!["a.rs", "b.txt", "c.md"]);
    assert_eq!(
        data.list_items[0].file_type,
        Some(".rs".to_string()),
        "the extension falls back from the name"
    );
    assert_eq!(
        data.list_items[0].modified_time, "0",
        "a missing modification time falls back to '0'"
    );
    assert!(!data.list_items[0].is_directory);
}

#[tokio::test]
async fn list_files_accepts_the_items_shape_and_the_type_filter() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"items": [
            {"name": "b.txt", "path": "/work/b.txt", "size": 2},
            {"name": "a.rs", "path": "/work/a.rs", "size": 1},
            {"name": "dir", "path": "/work/dir", "is_directory": true},
        ]}),
    )
    .await;
    let mut params = ListFilesParams::new("/work");
    params.file_types = Some(vec![".txt".to_string()]);

    let result = fs(&server).list_files(params).await;

    assert_eq!(result.code, 0);
    let data: ListFilesData = result.data.unwrap();
    assert_eq!(data.total_count, 1);
    assert_eq!(names(&data.list_items), vec!["b.txt"]);
}

#[tokio::test]
async fn list_files_sorts_descending_on_request() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"list_items": [
            {"name": "a.txt", "size": 1},
            {"name": "b.txt", "size": 3},
            {"name": "c.txt", "size": 2},
        ]}),
    )
    .await;
    let mut params = ListFilesParams::new("/work");
    params.sort_by = SortBy::Size;
    params.sort_descending = true;

    let result = fs(&server).list_files(params).await;

    let data: ListFilesData = result.data.unwrap();
    assert_eq!(names(&data.list_items), vec!["b.txt", "c.txt", "a.txt"]);
}

#[tokio::test]
async fn list_directories_keeps_only_directories_with_a_zero_size() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"list_items": [
            {"name": "b", "path": "/work/b", "is_directory": true, "size": 10, "type": ".x"},
            {"name": "a", "path": "/work/a", "is_directory": true, "size": 12},
            {"name": "f.txt", "path": "/work/f.txt", "size": 4},
        ]}),
    )
    .await;

    let result = fs(&server)
        .list_directories(ListDirsParams::new("/work"))
        .await;

    assert_eq!(result.code, 0);
    let data: ListFilesData = result.data.unwrap();
    assert_eq!(data.total_count, 2);
    assert_eq!(names(&data.list_items), vec!["a", "b"]);
    assert_eq!(data.list_items[0].size, 0);
    assert_eq!(data.list_items[0].file_type, None);
    assert!(data.list_items.iter().all(|item| item.is_directory));
}

#[tokio::test]
async fn search_files_maps_and_sorts_the_matches() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"matching_files": [
            {"name": "z.rs", "path": "/work/z.rs", "size": 9},
            {"name": "a.rs", "path": "/work/a.rs", "size": 1},
        ]}),
    )
    .await;

    let result = fs(&server)
        .search_files(SearchFilesParams::new("/work", "*.rs").with_excludes(["target/*"]))
        .await;

    assert_eq!(result.code, 0);
    let data: SearchFilesData = result.data.unwrap();
    assert_eq!(data.total_matches, 2);
    assert_eq!(names(&data.matching_files), vec!["a.rs", "z.rs"]);
    assert_eq!(data.search_path, "/work");
    assert_eq!(data.search_pattern, "*.rs");
    assert_eq!(data.exclude_patterns, Some(vec!["target/*".to_string()]));
}

// ─── transfers ──────────────────────────────────────────────────────────────

/// Mounts a missing-path answer for the upload probe and a write answer.
async fn mock_upload(server: &MockServer, probe: Value, written: Value) {
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .and(body_partial_json(json!({"method": "read_file"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(probe))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .and(body_partial_json(json!({"method": "write_file"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(written))
        .mount(server)
        .await;
}

#[tokio::test]
async fn upload_reads_the_host_file_and_sends_base64_bytes() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, [1_u8, 2, 250]).await.unwrap();
    let server = MockServer::start().await;
    mock_upload(
        &server,
        json!({"code": -1, "message": "No such file or directory"}),
        json!({"code": 0, "data": {"size": 3}}),
    )
    .await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    let data: UploadFileData = result.data.unwrap();
    assert_eq!(data.local_path, local.to_str().unwrap());
    assert_eq!(data.target_path, "/work/payload.bin");
    assert_eq!(data.size, 3);

    let envelopes = envelopes(&server).await;
    assert_eq!(envelopes.len(), 2, "one probe then one write");
    assert_eq!(envelopes[0]["method"], "read_file");
    assert_eq!(envelopes[0]["params"]["mode"], "bytes");
    assert_eq!(envelopes[1]["method"], "write_file");
    assert_eq!(envelopes[1]["params"]["content"], "AQL6");
    assert_eq!(envelopes[1]["params"]["encoding"], "base64");
    assert_eq!(envelopes[1]["params"]["mode"], "bytes");
    assert_eq!(envelopes[1]["params"]["prepend_newline"], false);
    assert_eq!(envelopes[1]["params"]["append_newline"], false);
    assert_eq!(envelopes[1]["params"]["append"], false);
    assert_eq!(envelopes[1]["params"]["create_if_not_exist"], true);
}

#[tokio::test]
async fn upload_skips_the_probe_when_overwriting() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, b"hi").await.unwrap();
    let server = MockServer::start().await;
    respond_data(&server, json!({"size": 2})).await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            overwrite: true,
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    let envelopes = envelopes(&server).await;
    assert_eq!(envelopes.len(), 1);
    assert_eq!(envelopes[0]["method"], "write_file");
}

#[tokio::test]
async fn upload_refuses_an_existing_destination() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, b"hi").await.unwrap();
    let server = MockServer::start().await;
    mock_upload(
        &server,
        json!({"code": 0, "data": {"content": [1]}}),
        json!({"code": 0, "data": {}}),
    )
    .await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 199_003);
    assert_eq!(
        result.message,
        "file system operation execution error, execution: upload_file, \
         reason: File already exists: /work/payload.bin"
    );
    assert_eq!(envelopes(&server).await.len(), 1, "only the probe ran");
}

#[tokio::test]
async fn upload_reports_a_missing_host_file_without_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: "/definitely/not/here.bin".to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 199_003);
    assert!(
        result.message.contains("failed to read host file"),
        "{result:?}"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn download_writes_the_local_file_and_creates_the_parents() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("nested/dir/out.bin");
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": [104, 105], "mode": "bytes"})).await;

    let result = fs(&server)
        .download_file(DownloadFileParams {
            source_path: "/work/out.bin".to_string(),
            local_path: local.to_str().unwrap().to_string(),
            ..DownloadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    let data: DownloadFileData = result.data.unwrap();
    assert_eq!(data.source_path, "/work/out.bin");
    assert_eq!(data.local_path, local.to_str().unwrap());
    assert_eq!(data.size, 2);
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"hi");
    assert_eq!(
        last_envelope(&server).await["params"]["mode"],
        "bytes",
        "a download reads the remote file in bytes mode"
    );
}

#[tokio::test]
async fn download_decodes_encoded_content_and_honours_overwrite() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("out.txt");
    tokio::fs::write(&local, b"old").await.unwrap();
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"content": "6869", "content_encoding": "hex"}),
    )
    .await;
    let params = DownloadFileParams {
        source_path: "/work/out.txt".to_string(),
        local_path: local.to_str().unwrap().to_string(),
        ..DownloadFileParams::default()
    };

    let refused = fs(&server).download_file(params.clone()).await;
    assert_eq!(refused.code, 199_003);
    assert!(refused
        .message
        .contains(&format!("File already exists: {}", local.display())));
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"old");

    let overwritten = fs(&server)
        .download_file(DownloadFileParams {
            overwrite: true,
            ..params
        })
        .await;
    assert_eq!(overwritten.code, 0);
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"hi");
}

#[tokio::test]
async fn download_decodes_base64_content() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("out.txt");
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"content": "aGVsbG8=", "content_encoding": "base64"}),
    )
    .await;

    let result = fs(&server)
        .download_file(DownloadFileParams {
            source_path: "/work/out.txt".to_string(),
            local_path: local.to_str().unwrap().to_string(),
            ..DownloadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"hello");
}

#[tokio::test]
async fn the_streaming_transfers_emit_exactly_one_terminal_chunk() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, b"abc").await.unwrap();
    let server = MockServer::start().await;
    mock_upload(
        &server,
        json!({"code": -1, "message": "not found"}),
        json!({"code": 0, "data": {"size": 7}}),
    )
    .await;

    let chunks: Vec<OpResult<TransferChunkData>> =
        collect(fs_arc(&server).upload_file_stream(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        }))
        .await;

    assert_eq!(chunks.len(), 1);
    let data = chunks[0].data.as_ref().unwrap();
    assert_eq!(data.source_path, local.to_str().unwrap());
    assert_eq!(data.target_path, "/work/payload.bin");
    assert_eq!(data.chunk_size, 3, "the real size of the uploaded bytes");
    assert_eq!(data.chunk_index, 0);
    assert!(data.is_last_chunk);
}

#[tokio::test]
async fn a_failing_transfer_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": []})).await;

    let chunks: Vec<OpResult<TransferChunkData>> =
        collect(fs_arc(&server).download_file_stream(DownloadFileParams {
            source_path: "/work/missing.bin".to_string(),
            local_path: "/definitely/not/here.bin".to_string(),
            ..DownloadFileParams::default()
        }))
        .await;

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].code, 199_003);
    let data = chunks[0].data.as_ref().unwrap();
    assert!(data.is_last_chunk);
    assert_eq!(data.chunk_size, 0);
    assert_eq!(data.source_path, "/work/missing.bin");
    assert_eq!(data.target_path, "/definitely/not/here.bin");
}

// ─── streamed reads ─────────────────────────────────────────────────────────

#[tokio::test]
async fn a_text_read_stream_emits_one_chunk_per_line() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": "one\ntwo\nthree"})).await;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(ReadFileParams::text("/work/a.txt"))).await;

    assert_eq!(items.len(), 3);
    assert!(items.iter().all(|item| item.code == 0));
    let chunks: Vec<&ReadChunkData> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| match &chunk.chunk {
                FileContent::Text(text) => text.as_str(),
                FileContent::Bytes(_) => panic!("expected text chunks"),
            })
            .collect::<Vec<_>>(),
        vec!["one\n", "two\n", "three"]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.is_last_chunk)
            .collect::<Vec<_>>(),
        vec![false, false, true]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(chunks[0].path, "/work/a.txt");
    assert_eq!(chunks[0].mode, ReadMode::Text);
}

#[tokio::test]
async fn a_byte_read_stream_honors_the_chunk_size() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"content": [1, 2, 3, 4, 5], "mode": "bytes"}),
    )
    .await;
    let mut params = ReadFileParams::bytes("/work/a.bin");
    params.chunk_size = 2;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(params)).await;

    assert_eq!(items.len(), 3);
    let chunks: Vec<&ReadChunkData> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_size)
            .collect::<Vec<_>>(),
        vec![2, 2, 1]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(chunks[2].is_last_chunk);
}

#[tokio::test]
async fn an_empty_read_stream_still_terminates() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": ""})).await;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(ReadFileParams::text("/work/empty.txt"))).await;

    assert_eq!(items.len(), 1);
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.chunk_size, 0);
    assert!(chunk.is_last_chunk);
    assert_eq!(chunk.path, "/work/empty.txt");
}

#[tokio::test]
async fn a_failing_read_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "boom"})).await;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(ReadFileParams::text("/work/a.txt"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 199_007);
    assert!(items[0].message.contains("aio gateway code -1: boom"));
    let chunk = items[0].data.as_ref().unwrap();
    assert!(chunk.is_last_chunk);
    assert_eq!(chunk.chunk_size, 0);
    assert_eq!(chunk.chunk_index, 0);
}
