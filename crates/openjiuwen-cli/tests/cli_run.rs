//! Offline, process-level `run` scenarios of the `openjiuwen` binary.
//!
//! The prompt layer of the CLI is exercised end to end through a real child
//! process and a real loopback model server:
//!
//! 1. the mock server from `openjiuwen-test-support` binds `127.0.0.1:0` and
//!    speaks the `OpenAI` chat-completions wire format, so the binary's actual
//!    HTTP client, SSE decoder and model plumbing run — nothing is stubbed;
//! 2. the answer is a scripted **fallback**, which the server reuses for every
//!    request without consuming it, so any number of agent-loop turns succeed
//!    and the suite never depends on how many calls a run makes;
//! 3. the model connection is passed as flags (`--provider`, `--api-base`,
//!    `--api-key`, `-m`), never through the environment, so the case stays
//!    self-describing and the child inherits nothing;
//! 4. the child runs in a temporary working directory with a temporary
//!    `OPENJIUWEN_HOME` and a temporary `-w` workspace ([`support::Sandbox`]),
//!    and every ambient `OPENJIUWEN_*` / `OPENAI_*` / `API_KEY` / `API_BASE` /
//!    `MODEL_*` variable is removed before it starts.
//!
//! Covered: the three `-f` output formats (`text`, `json`, `stream-json`), the
//! documented JSON shapes from
//! [`openjiuwen_harness::cli::RunOutcome::to_json`] and
//! [`openjiuwen_harness::cli::stream_json_line`], `--no-color` accepted through a
//! real run, and the `chat` REPL's plain help page.
//!
//! No test writes colour: `stdout` is asserted to be escape-free because it is a
//! machine-readable stream for the JSON formats.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use openjiuwen_test_support::{MockLlmServer, ScriptedResponse, TEST_API_KEY};
use serde_json::Value;
use support::{assert_no_ansi, assert_success, non_empty_lines, parse_json, Sandbox};

/// The answer the scripted model returns for every request.
const ANSWER: &str = "hello from the scripted model";

/// Starts a mock model server whose reusable fallback answers [`ANSWER`].
async fn scripted_server() -> MockLlmServer {
    let server = MockLlmServer::start().await;
    server.set_fallback(ScriptedResponse::text(ANSWER));
    server
}

/// Builds the model-connection and workspace flags of one invocation.
///
/// The globals are placed before the subcommand on purpose: clap treats them as
/// global, and this is the spelling a user copies out of the README.
fn connection_args(sandbox: &Sandbox, server: &MockLlmServer) -> Vec<String> {
    vec![
        "--provider".to_owned(),
        "openai".to_owned(),
        "--api-base".to_owned(),
        server.base_url().to_owned(),
        "--api-key".to_owned(),
        TEST_API_KEY.to_owned(),
        "-m".to_owned(),
        "mock-model".to_owned(),
        "-w".to_owned(),
        sandbox.workspace().display().to_string(),
    ]
}

/// Appends `tail` to `args` (a tiny helper that keeps the cases readable).
fn with(mut args: Vec<String>, tail: &[&str]) -> Vec<String> {
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

/// Asserts the child used the mock server at least once.
fn assert_server_was_used(server: &MockLlmServer) {
    assert!(
        server.request_count() >= 1,
        "the run never reached the mock model server"
    );
}

#[tokio::test]
async fn json_format_prints_one_object_describing_the_run() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;
    let args = with(
        connection_args(&sandbox, &server),
        &["run", "hello", "-f", "json"],
    );
    let output = sandbox.run(&args).await;
    let text = assert_success(&output);
    assert_no_ansi(&text);

    // Parsing the whole trimmed stream proves exactly one JSON document was
    // printed: a second line (a log line, a banner) would break this.
    let value = parse_json(&text);
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("expected a JSON object, got {value}"));
    for key in ["answer", "result_type", "usage", "session_id"] {
        assert!(object.contains_key(key), "missing key `{key}`: {value}");
    }
    assert_eq!(value["answer"], ANSWER);
    assert_eq!(value["result_type"], "answer");
    let session_id = value["session_id"]
        .as_str()
        .expect("session_id is a string");
    assert!(
        session_id.starts_with("cli-"),
        "the session id carries the `cli-` prefix: {session_id}"
    );
    if !value["usage"].is_null() {
        assert!(
            value["usage"].is_object(),
            "usage is null or an object: {value}"
        );
    }
    assert_server_was_used(&server);
    server.shutdown().await;
}

#[tokio::test]
async fn text_format_streams_the_scripted_answer() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;
    let args = with(
        connection_args(&sandbox, &server),
        &["run", "hello", "-f", "text"],
    );
    let output = sandbox.run(&args).await;
    let text = assert_success(&output);
    assert_no_ansi(&text);
    assert!(
        text.contains(ANSWER),
        "the answer must be rendered: {text:?}"
    );
    assert_server_was_used(&server);
    server.shutdown().await;
}

#[tokio::test]
async fn stream_json_format_prints_json_lines_and_a_terminal_result() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;
    let args = with(
        connection_args(&sandbox, &server),
        &["run", "hello", "-f", "stream-json"],
    );
    let output = sandbox.run(&args).await;
    let text = assert_success(&output);
    assert_no_ansi(&text);

    let lines = non_empty_lines(&text);
    assert!(
        lines.len() >= 2,
        "expected chunk lines plus a result line, got {lines:?}"
    );
    let values: Vec<Value> = lines.iter().map(|line| parse_json(line)).collect();
    let (chunks, last) = values.split_at(values.len() - 1);
    for chunk in chunks {
        assert!(
            chunk["type"].is_string(),
            "a streamed line needs a string type: {chunk}"
        );
        assert!(chunk.get("index").is_some(), "missing index: {chunk}");
        assert!(chunk.get("payload").is_some(), "missing payload: {chunk}");
        assert_ne!(chunk["type"], "result", "only the last line is the result");
    }
    let result = &last[0];
    assert_eq!(
        result["type"], "result",
        "the last line is the result frame"
    );
    assert_eq!(result["answer"], ANSWER);
    assert!(
        result["session_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("cli-")),
        "the result frame carries the session id: {result}"
    );
    assert_server_was_used(&server);
    server.shutdown().await;
}

#[tokio::test]
async fn no_color_is_accepted_through_a_real_run() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;
    // `--no-color` after the subcommand: it reaches `Cli::to_harness` and is
    // resolved by the harness options like any other global.
    let args = with(
        connection_args(&sandbox, &server),
        &["run", "--no-color", "hello"],
    );
    let output = sandbox.run(&args).await;
    let text = assert_success(&output);
    assert!(
        text.contains(ANSWER),
        "the answer must be rendered: {text:?}"
    );
    assert_no_ansi(&text);
    assert_server_was_used(&server);
    server.shutdown().await;
}

#[test]
fn chat_help_describes_the_repl() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["chat", "--help"]);
    let text = assert_success(&output);
    assert!(
        text.to_lowercase().contains("repl"),
        "the chat help describes the REPL: {text:?}"
    );
}
