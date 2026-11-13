//! Shell wire-contract tests of the `aio` provider.
//!
//! Python counterpart: `tests/unit_tests/extensions/sys_operation/sandbox/test_aio_*.py`
//! (registration only); these pin the `execute_cmd` envelope, the background and
//! argv forms, the `199004` validation/timeout rules, `199007` for gateway
//! failures, and the host-side chunking.
use super::*;

// ─── shell ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_cmd_sends_the_python_signature_parameters() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"stdout": "hi\n", "exit_code": 0})).await;
    let mut params = ShellExecParams::new("echo hi")
        .with_timeout_secs(7)
        .with_cwd("/work")
        .with_shell_type(ShellType::Bash);
    params
        .environment
        .insert("A".to_string(), Some("1".to_string()));
    params.environment.insert("B".to_string(), None);

    let result = shell(&server).execute_cmd(params).await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["op_type"], "shell");
    assert_eq!(envelope["method"], "execute_cmd");
    assert_eq!(
        envelope["params"],
        json!({
            "command": "echo hi",
            "cwd": "/work",
            "timeout": 7,
            "environment": {"A": "1", "B": null},
            "shell_type": "bash",
        })
    );
}

#[tokio::test]
async fn execute_cmd_defaults_the_timeout_and_drops_a_dot_cwd() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("echo hi").with_cwd("."))
        .await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["params"]["timeout"], 300);
    assert!(envelope["params"]["cwd"].is_null());
    assert_eq!(envelope["params"]["shell_type"], "auto");
}

#[tokio::test]
async fn an_empty_command_is_rejected_before_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("   "))
        .await;

    assert_eq!(result.code, 199_004);
    assert_eq!(
        result.message,
        "shell operation execution error, execution: execute_cmd, \
         reason: command can not be empty"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn shell_output_prefers_stdout_and_falls_back_to_output() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "command": "make all",
            "cwd": "/work",
            "stdout": "one\n",
            "stderr": "warn\n",
            "exit_code": 1,
            "duration_ms": 12,
        }),
    )
    .await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("make all"))
        .await;

    assert_eq!(result.code, 0, "a non-zero exit code is still a success");
    let data: ShellOutput = result.data.unwrap();
    assert_eq!(data.command, "make all");
    assert_eq!(data.cwd, "/work");
    assert_eq!(data.exit_code, 1);
    assert_eq!(data.stdout, "one\n");
    assert_eq!(data.stderr, "warn\n");
    assert_eq!(data.duration_ms, 12);
    assert!(!data.timed_out);
    assert!(!data.truncated);
    assert!(!data.cancelled);

    let server = MockServer::start().await;
    respond_data(&server, json!({"output": "merged\n", "exit_code": 2})).await;
    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("make all").with_cwd("/work"))
        .await;
    let data: ShellOutput = result.data.unwrap();
    assert_eq!(data.stdout, "merged\n", "a merged output field is stdout");
    assert_eq!(data.stderr, "");
    assert_eq!(
        data.command, "make all",
        "the request command is the fallback"
    );
    assert_eq!(data.exit_code, 2);
}

#[tokio::test]
async fn a_shell_timeout_maps_to_199004_with_output() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"stdout": "partial\n", "exit_code": 124})).await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("sleep 99").with_timeout_secs(5))
        .await;

    assert_eq!(result.code, 199_004);
    assert_eq!(result.message, "execution timeout after 5 seconds");
    let data: ShellOutput = result.data.unwrap();
    assert!(data.timed_out);
    assert_eq!(data.exit_code, 124);
    assert_eq!(data.stdout, "partial\n");
}

#[tokio::test]
async fn execute_cmd_background_sends_its_own_envelope() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"command": "sleep 1", "cwd": "/work", "pid": 4242}),
    )
    .await;

    let result = shell(&server)
        .execute_cmd_background(ShellExecParams::new("sleep 1").with_cwd("/work"))
        .await;

    assert_eq!(result.code, 0);
    let data: ShellBackgroundOutput = result.data.unwrap();
    assert_eq!(data.command, "sleep 1");
    assert_eq!(data.cwd, "/work");
    assert_eq!(data.pid, Some(4242));
    assert_eq!(data.error, None);

    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["method"], "execute_cmd_background");
    assert_eq!(
        envelope["params"],
        json!({
            "command": "sleep 1",
            "cwd": "/work",
            "environment": {},
            "shell_type": "auto",
        })
    );
}

#[tokio::test]
async fn a_background_launch_error_is_reported_inside_the_output() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "command": "nope",
            "cwd": ".",
            "pid": null,
            "error": "process exited early with code 2: command not found",
        }),
    )
    .await;

    let result = shell(&server)
        .execute_cmd_background(ShellExecParams::new("nope"))
        .await;

    assert_eq!(result.code, 0);
    let data: ShellBackgroundOutput = result.data.unwrap();
    assert_eq!(data.pid, None);
    assert_eq!(
        data.error.as_deref(),
        Some("process exited early with code 2: command not found")
    );
}

#[tokio::test]
async fn execute_argv_quotes_the_argument_vector_into_a_command_line() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let result = shell(&server)
        .execute_argv(ShellArgvParams::new(["printf", "%s\\n", "hello world"]))
        .await;

    assert_eq!(result.code, 0);
    let data: ShellOutput = result.data.unwrap();
    assert_eq!(data.command, "printf '%s\\n' 'hello world'");
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["method"], "execute_cmd");
    assert_eq!(
        envelope["params"]["command"],
        "printf '%s\\n' 'hello world'"
    );
}

#[tokio::test]
async fn an_empty_argv_is_rejected_before_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = shell(&server)
        .execute_argv(ShellArgvParams::default())
        .await;

    assert_eq!(result.code, 199_004);
    assert!(result.message.contains("argv can not be empty"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn the_shell_stream_chunks_stdout_then_stderr_and_terminates() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"stdout": "a\nb\n", "stderr": "e\n", "exit_code": 3}),
    )
    .await;

    let items: Vec<OpResult<ShellChunk>> =
        collect(shell_arc(&server).execute_cmd_stream(ShellExecParams::new("echo a"))).await;

    assert_eq!(items.len(), 4);
    assert!(items.iter().all(|item| item.code == 0));
    let chunks: Vec<&ShellChunk> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>(),
        vec!["a\n", "b\n", "e\n", ""]
    );
    assert_eq!(
        chunks.iter().map(|chunk| chunk.stream).collect::<Vec<_>>(),
        vec![
            Some(StreamKind::Stdout),
            Some(StreamKind::Stdout),
            Some(StreamKind::Stderr),
            None,
        ]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(chunks[3].exit_code, Some(3));
}

#[tokio::test]
async fn an_empty_shell_stream_still_terminates() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let items: Vec<OpResult<ShellChunk>> =
        collect(shell_arc(&server).execute_cmd_stream(ShellExecParams::new("true"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 0);
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.exit_code, Some(0));
    assert!(chunk.stream.is_none());
}

#[tokio::test]
async fn a_failing_shell_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "nope"})).await;

    let items: Vec<OpResult<ShellChunk>> =
        collect(shell_arc(&server).execute_cmd_stream(ShellExecParams::new("echo a"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 199_007, "gateway failures report 199007");
    assert!(
        items[0].message.contains("aio gateway code -1: nope"),
        "{}",
        items[0].message
    );
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.exit_code, Some(-1));
}
