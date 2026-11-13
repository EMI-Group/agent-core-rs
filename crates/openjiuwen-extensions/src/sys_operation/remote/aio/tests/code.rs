//! Code wire-contract tests of the `aio` provider.
//!
//! Python counterpart: `tests/unit_tests/extensions/sys_operation/sandbox/test_aio_*.py`
//! (registration only); these pin the `execute_code` envelope, the `199005`
//! validation/timeout rules, `199007` for gateway failures, and the host-side
//! chunking.
use super::*;

// ─── code ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_code_sends_the_python_signature_parameters() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let result = code(&server)
        .execute_code(
            CodeExecParams::new("print(1)")
                .with_timeout_secs(9)
                .with_cwd("/work")
                .with_force_file(true),
        )
        .await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["op_type"], "code");
    assert_eq!(envelope["method"], "execute_code");
    assert_eq!(
        envelope["params"],
        json!({
            "code": "print(1)",
            "language": "python",
            "timeout": 9,
            "environment": {},
            "cwd": "/work",
            "options": {"force_file": true},
        })
    );
}

#[tokio::test]
async fn an_empty_code_snippet_maps_to_199005_with_the_output() {
    let server = MockServer::start().await;

    let result = code(&server)
        .execute_code(CodeExecParams::javascript("   "))
        .await;

    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "code can not be empty");
    let data: CodeOutput = result.data.unwrap();
    assert_eq!(data.code_content, "   ");
    assert_eq!(data.language, "javascript");
    assert_eq!(data.exit_code, -1);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn code_output_maps_the_payload_and_falls_back_to_the_request() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "code_content": "print(1)",
            "language": "python",
            "exit_code": 0,
            "stdout": "1\n",
            "duration_ms": 21,
        }),
    )
    .await;

    let result = code(&server)
        .execute_code(CodeExecParams::new("print(1)"))
        .await;

    assert_eq!(result.code, 0);
    let data: CodeOutput = result.data.unwrap();
    assert_eq!(data.code_content, "print(1)");
    assert_eq!(data.language, "python");
    assert_eq!(data.exit_code, 0);
    assert_eq!(data.stdout, "1\n");
    assert_eq!(data.stderr, "");
    assert_eq!(data.duration_ms, 21);
    assert!(!data.timed_out);
    assert!(!data.truncated);

    let server = MockServer::start().await;
    respond_data(&server, json!({})).await;
    let result = code(&server)
        .execute_code(CodeExecParams::javascript("console.log(1)"))
        .await;
    let data: CodeOutput = result.data.unwrap();
    assert_eq!(data.code_content, "console.log(1)");
    assert_eq!(data.language, "javascript");
    assert_eq!(data.exit_code, -1);
}

#[tokio::test]
async fn a_code_gateway_failure_maps_to_199007() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "no interpreter"})).await;

    let result = code(&server)
        .execute_code(CodeExecParams::new("print(1)"))
        .await;

    assert_eq!(result.code, 199_007);
    assert_eq!(
        result.message,
        "sandbox gateway error, operation: execute_code, \
         reason: aio gateway code -1: no interpreter"
    );
    assert!(result.data.is_none());
}

#[tokio::test]
async fn a_code_timeout_is_detected_from_the_exit_code_or_stderr() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"stdout": "", "exit_code": 124})).await;

    let result = code(&server)
        .execute_code(CodeExecParams::new("import time").with_timeout_secs(2))
        .await;

    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "execution timeout after 2 seconds");
    let data: CodeOutput = result.data.unwrap();
    assert!(data.timed_out);
    assert_eq!(data.exit_code, 124);

    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"exit_code": 1, "stderr": "Command Timed Out\n"}),
    )
    .await;
    let result = code(&server)
        .execute_code(CodeExecParams::new("import time").with_timeout_secs(3))
        .await;
    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "execution timeout after 3 seconds");
    let data: CodeOutput = result.data.unwrap();
    assert!(data.timed_out);
    assert_eq!(data.exit_code, 1, "the reported status is kept");
}

#[tokio::test]
async fn the_code_stream_chunks_the_output_and_terminates() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"stdout": "one\n", "stderr": "bad\n", "exit_code": 4}),
    )
    .await;

    let items: Vec<OpResult<CodeChunk>> =
        collect(code_arc(&server).execute_code_stream(CodeExecParams::new("print(1)"))).await;

    assert_eq!(items.len(), 3);
    let chunks: Vec<&CodeChunk> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>(),
        vec!["one\n", "bad\n", ""]
    );
    assert_eq!(chunks[0].stream, Some(StreamKind::Stdout));
    assert_eq!(chunks[1].stream, Some(StreamKind::Stderr));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(chunks[2].exit_code, Some(4));
}

#[tokio::test]
async fn an_empty_code_stream_still_terminates() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let items: Vec<OpResult<CodeChunk>> =
        collect(code_arc(&server).execute_code_stream(CodeExecParams::new("pass"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].data.as_ref().unwrap().chunk_index, 0);
    assert_eq!(items[0].data.as_ref().unwrap().exit_code, Some(0));
}

#[tokio::test]
async fn a_failing_code_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "nope"})).await;

    let items: Vec<OpResult<CodeChunk>> =
        collect(code_arc(&server).execute_code_stream(CodeExecParams::new("pass"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 199_007);
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.exit_code, Some(-1));
}
