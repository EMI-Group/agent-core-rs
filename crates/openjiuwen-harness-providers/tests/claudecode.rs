//! Offline, deterministic suite of the Claude Code subprocess adapter.
//!
//! Every case starts a real child process from [`support::FakeCli`], a POSIX shell script that
//! speaks the CLI's stream-json protocol: it records its argv and every stdin line in a transcript
//! and emits scripted NDJSON frames. The suite therefore exercises the true transport — pipes,
//! argv, flushing, EOF — without the `claude` binary, a vendor SDK or a network.
//!
//! Covered: provider creation (fake binary, missing binary, invalid mapping), a full turn through
//! [`HarnessIOAdapter`] (init/session capture, text output, tool call/result pair, turn lifecycle),
//! the permission round trip (allow and deny), the interrupt path, `--session-id` / `--resume`
//! argv, and the missing-result failure.
//!
//! Python counterpart: `tests/unit_tests/harness_providers/claudecode/`.
#![cfg(feature = "vendor")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value
)]
mod support;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::AgentChunk;
use openjiuwen_harness_protocol::{
    json_object_from_value, HarnessCheckpoint, HarnessContext, HarnessError, HarnessErrorKind,
    HarnessInteractionHandler, HarnessInteractionRequest, HarnessInteractionResponse,
    HarnessProtocol, HarnessProvider, HarnessResult, HarnessState, HostCapability,
    InteractionCancelReason, JsonObject, JsonValue, ResumePolicy, ToolApprovalDecision,
    ToolApprovalResponse,
};
use openjiuwen_harness_providers::providers::claudecode;
use openjiuwen_harness_providers::{HarnessIOAdapter, HarnessIOAdapterConfig};
use serde_json::json;
use support::FakeCli;

/// Timeout applied to every await that must not hang the suite.
const WAIT: Duration = Duration::from_secs(10);

/// One successful turn's frames: init, assistant text + tool call, tool result, terminal result.
const TURN_FRAMES: &[&str] = &[
    r#"{"type":"system","subtype":"init","session_id":"sess-1"}"#,
    r#"{"type":"assistant","message":{"model":"claude-sonnet-4","content":[{"type":"text","text":"hello"},{"type":"tool_use","id":"tool-1","name":"Read","input":{"path":"a.txt"}}],"usage":{"input_tokens":3,"output_tokens":5}}}"#,
    r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tool-1","content":"file body"}]}}"#,
    r#"{"type":"result","subtype":"success","is_error":false,"result":"done","session_id":"sess-1"}"#,
];

/// Converts a JSON object value into the protocol's ordered object type.
fn object(value: JsonValue) -> JsonObject {
    json_object_from_value(value).expect("the test value is a JSON object")
}

/// Creates the provider's harness over `config`, unstarted.
fn create(config: JsonValue) -> Arc<dyn HarnessProtocol> {
    let harness = claudecode::provider()
        .create(object(config))
        .expect("the provider creates a harness");
    Arc::from(harness)
}

/// Creates the provider's harness over the fake CLI at `cli`.
fn harness_for(cli: &FakeCli) -> Arc<dyn HarnessProtocol> {
    create(json!({"cli_path": cli.path().to_string_lossy()}))
}

/// Builds the host context used by every turn.
fn context() -> HarnessContext {
    HarnessContext::new("claude", "agent-1", "host-session", "system prompt")
}

/// Wraps `harness` in an IO adapter that projects turn lifecycle and failures.
fn adapter(harness: Arc<dyn HarnessProtocol>) -> Arc<HarnessIOAdapter> {
    HarnessIOAdapter::new(
        harness,
        HarnessIOAdapterConfig {
            emit_turn_lifecycle: true,
            emit_turn_failure: true,
            ..HarnessIOAdapterConfig::default()
        },
    )
}

/// A fake CLI that logs its argv and the first request, emits `frames`, then drains stdin.
fn frame_script(frames: &[&str]) -> FakeCli {
    let mut body = String::from(
        "printf 'ARGV %s\\n' \"$*\" >> \"$FAKE_CLI_LOG\"\n\
         IFS= read -r line\n\
         printf '%s\\n' \"$line\" >> \"$FAKE_CLI_LOG\"\n",
    );
    for frame in frames {
        body.push_str("printf '%s\\n' '");
        body.push_str(frame);
        body.push_str("'\n");
    }
    body.push_str("cat > /dev/null\n");
    FakeCli::script(&body)
}

/// Returns the argv lines the fake CLI logged.
fn argv(cli: &FakeCli) -> Vec<String> {
    cli.lines()
        .into_iter()
        .filter(|line| line.starts_with("ARGV "))
        .collect()
}

/// Returns the transcript lines that carry a stdin request.
fn requests(cli: &FakeCli) -> Vec<String> {
    cli.lines()
        .into_iter()
        .filter(|line| !line.starts_with("ARGV ") && line.starts_with('{'))
        .collect()
}

/// Returns the `text` field of a projected answer/reasoning output chunk.
fn output_text(chunk: &AgentChunk) -> String {
    let content = chunk.payload["content"]
        .as_str()
        .expect("a structured output chunk carries its content as compact JSON");
    let value: JsonValue = serde_json::from_str(content).expect("the output content is JSON");
    value["text"]
        .as_str()
        .expect("the output content carries a text field")
        .to_owned()
}

/// Awaits the next projected chunk.
async fn next_chunk(stream: &mut BoxStream<'static, AgentChunk>) -> AgentChunk {
    tokio::time::timeout(WAIT, stream.next())
        .await
        .expect("the adapter emits the chunk before the deadline")
        .expect("the chunk stream is still open")
}

/// Reads `count` chunks in order.
async fn read_chunks(stream: &mut BoxStream<'static, AgentChunk>, count: usize) -> Vec<AgentChunk> {
    let mut chunks = Vec::with_capacity(count);
    for _ in 0..count {
        chunks.push(next_chunk(stream).await);
    }
    chunks
}

/// Host interaction handler that answers every tool approval with one fixed decision.
struct Approving {
    /// Decision returned for each request.
    decision: ToolApprovalDecision,
    /// Number of requests answered.
    calls: AtomicUsize,
}

impl Approving {
    /// Builds a handler that answers `decision`.
    fn new(decision: ToolApprovalDecision) -> Arc<Self> {
        Arc::new(Self {
            decision,
            calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl HarnessInteractionHandler for Approving {
    async fn handle(
        &self,
        request: &HarnessInteractionRequest,
    ) -> HarnessResult<HarnessInteractionResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match request {
            HarnessInteractionRequest::ToolApproval(approval) => Ok(
                HarnessInteractionResponse::ToolApproval(ToolApprovalResponse {
                    request_id: approval.request_id.clone(),
                    decision: self.decision,
                    updated_arguments: None,
                    reason: Some("host decision".to_owned()),
                    interrupt: false,
                    provider_data: JsonObject::new(),
                }),
            ),
            other => Err(HarnessError::state(format!(
                "unexpected interaction: {other:?}"
            ))),
        }
    }

    async fn cancel(
        &self,
        _request_id: &str,
        _reason: InteractionCancelReason,
    ) -> HarnessResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

/// A fake binary resolves and yields an unstarted harness.
#[tokio::test]
async fn create_with_a_fake_binary_returns_an_unstarted_harness() {
    let cli = FakeCli::script("cat > /dev/null\n");
    let harness = harness_for(&cli);
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert!(harness.provider_session_id().is_none());
    assert!(argv(&cli).is_empty(), "create never spawns the CLI");
}

/// A missing executable is an operator error, reported as `Protocol`, never as `Unsupported`.
#[tokio::test]
async fn create_without_a_binary_reports_the_cli_path() {
    let error = claudecode::provider()
        .create(object(json!({"cli_path": "/nonexistent/claude"})))
        .err()
        .expect("the explicit path does not exist");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(error.message().contains("claude"), "{}", error.message());
    assert!(
        error.message().contains("not executable"),
        "{}",
        error.message()
    );
    assert!(
        error.message().contains("install Claude Code"),
        "{}",
        error.message()
    );
}

/// An invalid mapping fails validation before the binary is resolved.
#[tokio::test]
async fn create_with_an_invalid_mapping_is_validation() {
    let error = claudecode::provider()
        .create(object(
            json!({"cli_path": "/nonexistent/claude", "unknown_field": 1}),
        ))
        .err()
        .expect("unknown fields are rejected");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert!(
        error.message().contains("unknown_field"),
        "{}",
        error.message()
    );
}

// ---------------------------------------------------------------------------
// Turn projection
// ---------------------------------------------------------------------------

/// One turn projects init capture, text, a tool call/result pair, usage and the lifecycle.
#[tokio::test]
async fn a_turn_projects_outputs_tool_calls_and_lifecycle() {
    let cli = frame_script(TURN_FRAMES);
    let adapter = adapter(create(
        json!({"cli_path": cli.path().to_string_lossy(), "session_id": "sess-1"}),
    ));
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("hello"), true)
        .await
        .expect("the input is accepted");

    let chunks = read_chunks(&mut stream, 5).await;
    let kinds: Vec<&str> = chunks.iter().map(|chunk| chunk.kind.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "turn_lifecycle",
            "llm_output",
            "tool_call",
            "tool_result",
            "turn_lifecycle"
        ]
    );
    assert_eq!(chunks[0].payload["kind"], json!("started"));
    assert_eq!(output_text(&chunks[1]), "hello");
    assert_eq!(chunks[1].payload["channel"], json!("answer"));
    assert_eq!(chunks[1].payload["operation"], json!("final"));
    assert_eq!(chunks[2].payload["name"], json!("Read"));
    assert_eq!(chunks[2].payload["tool_call_id"], json!("tool-1"));
    assert_eq!(chunks[3].payload["tool_call_id"], json!("tool-1"));
    assert_eq!(chunks[3].payload["result"], json!("file body"));
    assert_eq!(chunks[4].payload["kind"], json!("finished"));
    assert_eq!(chunks[4].payload["final_output"], json!("done"));
    assert_eq!(
        adapter.session_id(),
        Some("sess-1".to_owned()),
        "the init frame's session id is the provider session"
    );

    let requests = requests(&cli);
    let user: JsonValue = serde_json::from_str(&requests[0]).expect("the user frame is JSON");
    assert_eq!(user["type"], json!("user"));
    assert_eq!(user["message"]["content"][0]["text"], json!("hello"));
    assert_eq!(user["session_id"], json!("sess-1"));
    assert!(
        user["uuid"].as_str().is_some_and(|uuid| !uuid.is_empty()),
        "the user frame carries the harness message id"
    );

    adapter.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Permissions
// ---------------------------------------------------------------------------

/// Builds a CLI that asks permission for one tool, logs the answer, then completes.
fn permission_script(answer_frame: &str) -> FakeCli {
    let body = format!(
        "printf 'ARGV %s\\n' \"$*\" >> \"$FAKE_CLI_LOG\"\n\
         IFS= read -r line\n\
         printf '%s\\n' \"$line\" >> \"$FAKE_CLI_LOG\"\n\
         printf '%s\\n' '{request}'\n\
         IFS= read -r answer\n\
         printf '%s\\n' \"$answer\" >> \"$FAKE_CLI_LOG\"\n\
         printf '%s\\n' '{answer_frame}'\n\
         cat > /dev/null\n",
        request = r#"{"type":"control_request","request_id":"req-1","request":{"subtype":"can_use_tool","tool_name":"Bash","tool_use_id":"tool-9","input":{"command":"ls"}}}"#,
    );
    FakeCli::script(&body)
}

/// Runs one permission round trip and returns the control response the CLI received.
async fn permission_round_trip(decision: ToolApprovalDecision) -> (String, usize) {
    let cli = permission_script(
        r#"{"type":"result","subtype":"success","is_error":false,"result":"done"}"#,
    );
    let handler = Approving::new(decision);
    let context = context()
        .with_host_capabilities(BTreeSet::from([HostCapability::ToolApproval]))
        .with_interactions(handler.clone());
    let adapter = adapter(create(json!({"cli_path": cli.path().to_string_lossy()})));
    adapter.start(context).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("go"), true)
        .await
        .expect("the input is accepted");
    let chunks = read_chunks(&mut stream, 2).await;
    assert_eq!(chunks[1].payload["kind"], json!("finished"));
    let response = cli
        .lines()
        .into_iter()
        .find(|line| line.contains("\"control_response\""))
        .expect("the CLI records the control response");
    adapter.stop().await.expect("the harness stops");
    (response, handler.calls.load(Ordering::SeqCst))
}

/// A host approval reaches the CLI as an allow response with the request id.
#[tokio::test]
async fn a_host_approval_is_written_as_an_allow_response() {
    let (response, calls) = permission_round_trip(ToolApprovalDecision::Allow).await;
    assert_eq!(calls, 1, "the host handler answers the permission prompt");
    let value: JsonValue = serde_json::from_str(&response).expect("the response is JSON");
    assert_eq!(value["type"], json!("control_response"));
    assert_eq!(value["response"]["request_id"], json!("req-1"));
    assert_eq!(value["response"]["response"]["behavior"], json!("allow"));
}

/// A host denial reaches the CLI as a deny response.
#[tokio::test]
async fn a_host_denial_is_written_as_a_deny_response() {
    let (response, calls) = permission_round_trip(ToolApprovalDecision::Deny).await;
    assert_eq!(calls, 1, "the host handler answers the permission prompt");
    let value: JsonValue = serde_json::from_str(&response).expect("the response is JSON");
    assert_eq!(value["response"]["response"]["behavior"], json!("deny"));
}

/// The CLI is told to consult the stdio permission tool when the host can approve.
#[tokio::test]
async fn an_interactive_host_switches_the_cli_to_the_stdio_permission_tool() {
    let cli = frame_script(TURN_FRAMES);
    let context = context().with_host_capabilities(BTreeSet::from([HostCapability::ToolApproval]));
    let adapter = adapter(create(json!({
        "cli_path": cli.path().to_string_lossy(),
        "session_id": "sess-1",
        "permission_mode": "bypassPermissions",
    })));
    adapter.start(context).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("hello"), true)
        .await
        .expect("the input is accepted");
    let _ = read_chunks(&mut stream, 5).await;
    let argv = argv(&cli).join(" ");
    assert!(argv.contains("--permission-prompt-tool stdio"), "{argv}");
    assert!(argv.contains("--permission-mode default"), "{argv}");
    adapter.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Interrupt
// ---------------------------------------------------------------------------

/// Aborting a turn writes the CLI's interrupt request and ends the turn aborted.
#[tokio::test]
async fn aborting_a_turn_writes_the_interrupt_request() {
    let cli = FakeCli::script(
        "printf 'ARGV %s\\n' \"$*\" >> \"$FAKE_CLI_LOG\"\n\
         IFS= read -r line\n\
         printf '%s\\n' \"$line\" >> \"$FAKE_CLI_LOG\"\n\
         printf '%s\\n' '{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"first\"}]}}'\n\
         IFS= read -r request\n\
         printf '%s\\n' \"$request\" >> \"$FAKE_CLI_LOG\"\n\
         printf '%s\\n' '{\"type\":\"result\",\"subtype\":\"error_during_execution\",\"is_error\":true,\"result\":\"interrupted\"}'\n\
         cat > /dev/null\n",
    );
    let adapter = adapter(harness_for(&cli));
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("go"), true)
        .await
        .expect("the input is accepted");
    let started = next_chunk(&mut stream).await;
    assert_eq!(started.payload["kind"], json!("started"));
    let output = next_chunk(&mut stream).await;
    assert_eq!(output_text(&output), "first");

    adapter.abort(false).await.expect("the abort is accepted");
    let aborted = next_chunk(&mut stream).await;
    assert_eq!(aborted.kind, "turn_lifecycle");
    assert_eq!(aborted.payload["kind"], json!("aborted"));
    assert_eq!(aborted.payload["termination_kind"], json!("user_abort"));
    assert!(
        cli.lines()
            .iter()
            .any(|line| line.contains(r#""subtype":"interrupt""#)),
        "the CLI receives the interrupt request"
    );
    adapter.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// A configured session id is passed as `--session-id`.
#[tokio::test]
async fn a_configured_session_id_is_passed_as_session_id() {
    let cli = frame_script(TURN_FRAMES);
    let adapter = adapter(create(json!({
        "cli_path": cli.path().to_string_lossy(),
        "session_id": "sess-1",
    })));
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("hello"), true)
        .await
        .expect("the input is accepted");
    let _ = read_chunks(&mut stream, 5).await;
    let argv = argv(&cli).join(" ");
    assert!(argv.contains("--session-id sess-1"), "{argv}");
    assert!(!argv.contains("--resume"), "{argv}");
    adapter.stop().await.expect("the harness stops");
}

/// A checkpoint under `ResumeIfAvailable` is passed as `--resume`.
#[tokio::test]
async fn a_checkpoint_is_passed_as_resume() {
    let cli = frame_script(&[
        r#"{"type":"system","subtype":"init","session_id":"sess-2"}"#,
        r#"{"type":"result","subtype":"success","is_error":false,"result":"done"}"#,
    ]);
    let checkpoint = HarnessCheckpoint {
        provider: "claude-code".to_owned(),
        schema_version: "1".to_owned(),
        agent_id: "agent-1".to_owned(),
        host_session_id: "host-session".to_owned(),
        checkpoint_id: "checkpoint-1".to_owned(),
        sequence: 1,
        data: object(json!({"session_id": "sess-2"})),
        provider_session_id: Some("sess-2".to_owned()),
        revision: None,
    };
    let context = context()
        .with_resume_policy(ResumePolicy::ResumeIfAvailable)
        .with_checkpoint(checkpoint);
    let adapter = adapter(harness_for(&cli));
    adapter.start(context).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("hello"), true)
        .await
        .expect("the input is accepted");
    let _ = read_chunks(&mut stream, 2).await;
    let argv = argv(&cli).join(" ");
    assert!(argv.contains("--resume sess-2"), "{argv}");
    assert_eq!(adapter.session_id(), Some("sess-2".to_owned()));
    adapter.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

/// A stream that ends without a `result` frame fails with `CLAUDE_MISSING_RESULT`.
#[tokio::test]
async fn a_stream_without_a_result_fails_the_turn() {
    let cli = FakeCli::script(
        "printf 'ARGV %s\\n' \"$*\" >> \"$FAKE_CLI_LOG\"\n\
         IFS= read -r line\n\
         printf '%s\\n' \"$line\" >> \"$FAKE_CLI_LOG\"\n\
         printf '%s\\n' '{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"partial\"}]}}'\n\
         exit 0\n",
    );
    let adapter = adapter(harness_for(&cli));
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("go"), true)
        .await
        .expect("the input is accepted");
    let chunks = read_chunks(&mut stream, 3).await;
    assert_eq!(output_text(&chunks[1]), "partial");
    assert_eq!(chunks[2].kind, "turn_lifecycle");
    assert_eq!(chunks[2].payload["kind"], json!("failed"));
    assert_eq!(
        chunks[2].payload["error_code"],
        json!("CLAUDE_MISSING_RESULT")
    );
    assert!(
        chunks[2].payload["error_message"]
            .as_str()
            .is_some_and(|message| message.contains("without a result")),
        "{}",
        chunks[2].payload["error_message"]
    );
    adapter.stop().await.expect("the harness stops");
}

/// A `result` frame with `is_error` fails the turn with the CLI's subtype and text.
#[tokio::test]
async fn an_error_result_fails_the_turn_with_its_category() {
    let cli = frame_script(&[
        r#"{"type":"result","subtype":"invalid_request","is_error":true,"api_error_status":400,"result":"bad tool schema"}"#,
    ]);
    let adapter = adapter(harness_for(&cli));
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("go"), true)
        .await
        .expect("the input is accepted");
    let chunks = read_chunks(&mut stream, 2).await;
    assert_eq!(chunks[1].payload["kind"], json!("failed"));
    assert_eq!(chunks[1].payload["error_code"], json!("invalid_request"));
    assert_eq!(chunks[1].payload["error_message"], json!("bad tool schema"));
    adapter.stop().await.expect("the harness stops");
}
