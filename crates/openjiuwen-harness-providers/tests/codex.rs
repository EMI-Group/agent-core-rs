//! Offline, deterministic suite of the Codex app-server adapter.
//!
//! Every case starts a real child process from [`support::FakeCli`], a POSIX shell script that
//! speaks the app-server's newline-delimited JSON-RPC protocol: it records its argv and every stdin
//! line in a transcript, answers the handshake and the thread/turn requests, and emits scripted
//! notifications. The suite therefore exercises the true transport — pipes, argv, id correlation,
//! flushing, EOF — without the `codex` binary, a vendor SDK or a network.
//!
//! Covered: provider creation (fake binary, missing binary, invalid mapping), a full turn through
//! [`HarnessIOAdapter`] (answer delta, tool call/result pair, turn lifecycle, provider session id),
//! the argv and `-c` overrides, the approval round trip (accept and decline), steering, the
//! interrupt path, the resumed thread, and a stream that ends without `turn/completed`.
//!
//! Python counterpart: `tests/unit_tests/harness_providers/codex/`.
#![cfg(feature = "vendor")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::similar_names
)]
mod support;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::AgentChunk;
use openjiuwen_harness_protocol::{
    json_object_from_value, DeliveryMode, HarnessCheckpoint, HarnessContext, HarnessError,
    HarnessErrorKind, HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessInput,
    HarnessInteractionHandler, HarnessInteractionRequest, HarnessInteractionResponse,
    HarnessProtocol, HarnessProvider, HarnessResult, HarnessState, HostCapability,
    InteractionCancelReason, JsonObject, JsonValue, ResumePolicy, ToolApprovalDecision,
    ToolApprovalResponse, TurnStatus, TurnTerminationKind,
};
use openjiuwen_harness_providers::providers::codex;
use openjiuwen_harness_providers::{HarnessIOAdapter, HarnessIOAdapterConfig};
use serde_json::json;
use support::FakeCli;

/// Timeout applied to every await that must not hang the suite.
const WAIT: Duration = Duration::from_secs(10);

/// Loop header of every fake app-server: records argv and stdin, answers the handshake, and opens
/// or resumes the thread.
const HEAD: &str = r#"printf 'ARGV %s\n' "$*" >> "$FAKE_CLI_LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$FAKE_CLI_LOG"
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"userAgent":"fake-codex"}}\n' "$id" ;;
    *'"method":"thread/start"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"thread":{"id":"th-1"}}}\n' "$id" ;;
    *'"method":"thread/resume"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"thread":{"id":"th-9"}}}\n' "$id" ;;
"#;

/// Loop tail: answers everything else with an empty result, then drains stdin.
const TAIL: &str = r#"    *)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id" ;;
  esac
done
cat > /dev/null
"#;

/// The `turn/start` case arm: answers the turn and emits `notifications`.
fn turn_arm(notifications: &[(&str, &str)]) -> String {
    let mut arm = String::from(
        r#"    *'"method":"turn/start"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"turn":{"id":"tu-1"}}}\n' "$id"
"#,
    );
    for (method, params) in notifications {
        let _ = writeln!(
            arm,
            "      printf '%s\\n' '{{\"jsonrpc\":\"2.0\",\"method\":\"{method}\",\"params\":{params}}}'"
        );
    }
    arm.push_str("      ;;\n");
    arm
}

/// The `turn/steer` case arm: acknowledges the steer, then completes the turn.
const STEER_ARM: &str = r#"    *'"method":"turn/steer"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      printf '%s\n' '{"jsonrpc":"2.0","method":"turn/completed","params":{"turn":{"id":"tu-1","status":"completed"}}}'
      ;;
"#;

/// The `turn/interrupt` case arm: acknowledges the interrupt, then reports the interruption.
const INTERRUPT_ARM: &str = r#"    *'"method":"turn/interrupt"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      printf '%s\n' '{"jsonrpc":"2.0","method":"turn/completed","params":{"turn":{"id":"tu-1","status":"interrupted"}}}'
      ;;
"#;

/// The `turn/start` case arm that asks the host to approve one shell command.
const APPROVAL_ARM: &str = r#"    *'"method":"turn/start"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"turn":{"id":"tu-1"}}}\n' "$id"
      printf '%s\n' '{"jsonrpc":"2.0","id":9001,"method":"item/commandExecution/requestApproval","params":{"itemId":"cmd-1","threadId":"th-1","turnId":"tu-1","command":"ls"}}'
      IFS= read -r answer
      printf '%s\n' "$answer" >> "$FAKE_CLI_LOG"
      printf '%s\n' '{"jsonrpc":"2.0","method":"turn/completed","params":{"turn":{"id":"tu-1","status":"completed"}}}'
      ;;
"#;

/// The terminal `turn/completed` notification of a successful turn.
const COMPLETED: &str = r#"{"turn":{"id":"tu-1","status":"completed"}}"#;

/// Composes a fake app-server out of the shared header, `arms` and the shared tail.
fn script(arms: &[&str]) -> FakeCli {
    let mut body = String::from(HEAD);
    for arm in arms {
        body.push_str(arm);
    }
    body.push_str(TAIL);
    FakeCli::script(&body)
}

/// Converts a JSON object value into the protocol's ordered object type.
fn object(value: JsonValue) -> JsonObject {
    json_object_from_value(value).expect("the test value is a JSON object")
}

/// Creates the provider's harness over `config`, unstarted.
fn create(config: JsonValue) -> Arc<dyn HarnessProtocol> {
    let harness = codex::provider()
        .create(object(config))
        .expect("the provider creates a harness");
    Arc::from(harness)
}

/// Creates the provider's harness over the fake CLI at `cli`.
fn harness_for(cli: &FakeCli) -> Arc<dyn HarnessProtocol> {
    create(json!({"codex_bin": cli.path().to_string_lossy()}))
}

/// Builds the host context used by every turn.
fn context() -> HarnessContext {
    HarnessContext::new("agent", "agent-1", "host-session", "system prompt")
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

/// Returns the argv lines the fake CLI logged.
fn argv(cli: &FakeCli) -> Vec<String> {
    cli.lines()
        .into_iter()
        .filter(|line| line.starts_with("ARGV "))
        .collect()
}

/// Returns every stdin line the fake CLI logged, including the request lines.
fn transcript(cli: &FakeCli) -> String {
    cli.lines().join("\n")
}

/// Returns the last chunk of an adapter stream.
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

/// Reads cycle events until one carries an output, or the cursor ends.
async fn next_output(cursor: &mut dyn HarnessEventCursor) -> HarnessEvent {
    tokio::time::timeout(WAIT, async {
        while let Some(event) = cursor
            .next_event()
            .await
            .expect("the cycle cursor is healthy")
        {
            if matches!(event.event, HarnessEventPayload::Output(_)) {
                return Some(event);
            }
        }
        None
    })
    .await
    .expect("the harness publishes the output in time")
    .expect("the turn publishes an output")
}

/// Reads cycle events until the turn reaches a terminal lifecycle event.
async fn until_terminal(cursor: &mut dyn HarnessEventCursor) -> Vec<HarnessEvent> {
    tokio::time::timeout(WAIT, async {
        let mut events = Vec::new();
        while let Some(event) = cursor
            .next_event()
            .await
            .expect("the cycle cursor is healthy")
        {
            let terminal = matches!(
                &event.event,
                HarnessEventPayload::TurnLifecycle(lifecycle) if lifecycle.kind.is_terminal()
            );
            events.push(event);
            if terminal {
                break;
            }
        }
        events
    })
    .await
    .expect("the turn reaches its terminal event in time")
}

/// Returns the terminal turn result the events carry.
fn terminal_result(events: &[HarnessEvent]) -> &openjiuwen_harness_protocol::TurnResult {
    events
        .iter()
        .find_map(|event| match &event.event {
            HarnessEventPayload::TurnLifecycle(lifecycle) if lifecycle.kind.is_terminal() => {
                lifecycle.result.as_ref()
            }
            _ => None,
        })
        .expect("a terminal event carries its result")
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
    assert!(argv(&cli).is_empty(), "create never spawns the app-server");
}

/// A missing executable is an operator error, reported as `Protocol`, never as `Unsupported`.
#[tokio::test]
async fn create_without_a_binary_reports_codex() {
    let error = codex::provider()
        .create(object(json!({"codex_bin": "/nonexistent/codex"})))
        .err()
        .expect("the explicit path does not exist");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(error.message().contains("codex"), "{}", error.message());
    assert!(
        error.message().contains("install the Codex CLI"),
        "{}",
        error.message()
    );
}

/// An invalid mapping fails validation before the binary is resolved.
#[tokio::test]
async fn create_with_an_invalid_mapping_is_validation() {
    let error = codex::provider()
        .create(object(
            json!({"codex_bin": "/nonexistent/codex", "unknown_field": 1}),
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

/// One turn projects the answer delta, a shell call pair, the final text and the lifecycle.
#[tokio::test]
async fn a_turn_projects_outputs_tool_calls_and_lifecycle() {
    let cli = script(&[&turn_arm(&[
        (
            "item/agentMessage/delta",
            r#"{"itemId":"msg-1","delta":"hello"}"#,
        ),
        (
            "item/started",
            r#"{"item":{"id":"cmd-1","type":"commandExecution","command":["ls"],"cwd":"/tmp"}}"#,
        ),
        (
            "item/completed",
            r#"{"item":{"id":"cmd-1","type":"commandExecution","command":["ls"],"cwd":"/tmp","aggregatedOutput":"a.txt","exitCode":0}}"#,
        ),
        (
            "item/completed",
            r#"{"item":{"id":"msg-1","type":"agentMessage","text":"hello"}}"#,
        ),
        (
            "thread/tokenUsage/updated",
            r#"{"tokenUsage":{"last":{"inputTokens":3,"outputTokens":5,"totalTokens":8}}}"#,
        ),
        ("turn/completed", COMPLETED),
    ])]);
    let adapter = adapter(create(json!({
        "codex_bin": cli.path().to_string_lossy(),
        "model": {
            "model": "gpt-5",
            "provider": "openai",
            "api_base": "https://example.test/v1",
            "api_key": "sk-test",
        },
    })));
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("hello"), true)
        .await
        .expect("the input is accepted");
    let chunks = read_chunks(&mut stream, 6).await;
    let kinds: Vec<&str> = chunks.iter().map(|chunk| chunk.kind.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "turn_lifecycle",
            "llm_output",
            "tool_call",
            "tool_result",
            "llm_output",
            "turn_lifecycle"
        ]
    );
    assert_eq!(chunks[0].payload["kind"], json!("started"));
    assert_eq!(output_text(&chunks[1]), "hello");
    assert_eq!(chunks[1].payload["channel"], json!("answer"));
    assert_eq!(chunks[1].payload["operation"], json!("delta"));
    assert_eq!(chunks[2].payload["name"], json!("shell"));
    assert_eq!(chunks[2].payload["tool_call_id"], json!("cmd-1"));
    assert_eq!(chunks[3].payload["tool_call_id"], json!("cmd-1"));
    assert_eq!(output_text(&chunks[4]), "hello");
    assert_eq!(chunks[4].payload["operation"], json!("final"));
    assert_eq!(chunks[5].payload["kind"], json!("finished"));
    assert_eq!(chunks[5].payload["final_output"], json!("hello"));
    assert_eq!(
        adapter.session_id(),
        Some("th-1".to_owned()),
        "the thread id is the provider session"
    );
    let user: JsonValue = serde_json::from_str(
        transcript(&cli)
            .lines()
            .find(|line| line.contains(r#""method":"turn/start""#))
            .expect("the CLI records the turn request"),
    )
    .expect("the turn request is JSON");
    assert_eq!(user["params"]["threadId"], json!("th-1"));
    assert_eq!(user["params"]["input"][0]["type"], json!("text"));
    assert_eq!(user["params"]["input"][0]["text"], json!("hello"));
    adapter.stop().await.expect("the harness stops");
}

/// The argv selects the stdio app-server and renders the endpoint overrides.
#[tokio::test]
async fn the_argv_selects_the_app_server_and_renders_overrides() {
    let cli = script(&[&turn_arm(&[("turn/completed", COMPLETED)])]);
    let harness = create(json!({
        "codex_bin": cli.path().to_string_lossy(),
        "model": {
            "model": "gpt-5",
            "provider": "my-endpoint",
            "api_base": "https://example.test/v1",
            "api_key": "sk-test",
        },
    }));
    harness.start(context()).await.expect("the harness starts");
    let argv = argv(&cli).join(" ");
    assert!(argv.contains("app-server --listen stdio://"), "{argv}");
    assert!(argv.contains(r#"model_provider="my-endpoint""#), "{argv}");
    assert!(
        argv.contains(r#"model_providers.my_endpoint.base_url="https://example.test/v1""#),
        "{argv}"
    );
    assert!(
        argv.contains(r#"model_providers.my_endpoint.env_key="OPENJIUWEN_CODEX_API_KEY""#),
        "{argv}"
    );
    harness.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Permissions
// ---------------------------------------------------------------------------

/// Runs one approval round trip and returns the transcript and the handler's call count.
async fn approval_round_trip(decision: ToolApprovalDecision) -> (String, usize) {
    let cli = script(&[APPROVAL_ARM]);
    let handler = Approving::new(decision);
    let context = context()
        .with_host_capabilities(BTreeSet::from([HostCapability::ToolApproval]))
        .with_interactions(handler.clone());
    let adapter = adapter(create(json!({"codex_bin": cli.path().to_string_lossy()})));
    adapter.start(context).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("go"), true)
        .await
        .expect("the input is accepted");
    let chunks = read_chunks(&mut stream, 2).await;
    assert_eq!(chunks[1].payload["kind"], json!("finished"));
    let transcript = transcript(&cli);
    let calls = handler.calls.load(Ordering::SeqCst);
    adapter.stop().await.expect("the harness stops");
    (transcript, calls)
}

/// A host approval reaches the CLI as `{"decision":"accept"}`.
#[tokio::test]
async fn a_host_approval_is_written_as_accept() {
    let (transcript, calls) = approval_round_trip(ToolApprovalDecision::Allow).await;
    assert_eq!(calls, 1, "the host handler answers the approval prompt");
    assert!(
        transcript.contains(r#""decision":"accept""#),
        "{transcript}"
    );
}

/// A host denial reaches the CLI as `{"decision":"decline"}`.
#[tokio::test]
async fn a_host_denial_is_written_as_decline() {
    let (transcript, calls) = approval_round_trip(ToolApprovalDecision::Deny).await;
    assert_eq!(calls, 1, "the host handler answers the approval prompt");
    assert!(
        transcript.contains(r#""decision":"decline""#),
        "{transcript}"
    );
}

// ---------------------------------------------------------------------------
// Steering and interruption
// ---------------------------------------------------------------------------

/// Steering an active turn reaches the CLI as `turn/steer` with the native turn id.
#[tokio::test]
async fn steering_reaches_the_cli_as_turn_steer() {
    let cli = script(&[
        &turn_arm(&[(
            "item/agentMessage/delta",
            r#"{"itemId":"msg-1","delta":"first"}"#,
        )]),
        STEER_ARM,
    ]);
    let harness = harness_for(&cli);
    harness.start(context()).await.expect("the harness starts");
    let mut cursor = harness.events();
    let receipt = harness
        .send(HarnessInput::text("go"), DeliveryMode::FollowUp)
        .await
        .expect("the input is accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::FollowUp);
    let _ = next_output(cursor.as_mut()).await;
    let steer = harness
        .send(HarnessInput::text("more"), DeliveryMode::Steer)
        .await
        .expect("the steer is accepted");
    assert_eq!(steer.accepted_mode, DeliveryMode::Steer);
    let events = until_terminal(cursor.as_mut()).await;
    assert_eq!(terminal_result(&events).status, TurnStatus::Completed);
    let transcript = transcript(&cli);
    assert!(
        transcript.contains(r#""method":"turn/steer""#),
        "{transcript}"
    );
    assert!(transcript.contains(r#""threadId":"th-1""#), "{transcript}");
    assert!(transcript.contains(r#""turnId":"tu-1""#), "{transcript}");
    assert!(transcript.contains(r#""text":"more""#), "{transcript}");
    harness.stop().await.expect("the harness stops");
}

/// Aborting a turn writes the CLI's interrupt request and ends the turn aborted.
#[tokio::test]
async fn aborting_a_turn_writes_the_interrupt_request() {
    let cli = script(&[
        &turn_arm(&[(
            "item/agentMessage/delta",
            r#"{"itemId":"msg-1","delta":"first"}"#,
        )]),
        INTERRUPT_ARM,
    ]);
    let harness = harness_for(&cli);
    harness.start(context()).await.expect("the harness starts");
    let mut cursor = harness.events();
    harness
        .send(HarnessInput::text("go"), DeliveryMode::FollowUp)
        .await
        .expect("the input is accepted");
    let _ = next_output(cursor.as_mut()).await;
    harness
        .abort_graceful()
        .await
        .expect("the abort is accepted");
    let events = until_terminal(cursor.as_mut()).await;
    let result = terminal_result(&events);
    assert_eq!(result.status, TurnStatus::Interrupted);
    assert_eq!(
        result
            .termination
            .as_ref()
            .map(|termination| termination.kind),
        Some(TurnTerminationKind::UserAbort)
    );
    let transcript = transcript(&cli);
    assert!(
        transcript.contains(r#""method":"turn/interrupt""#),
        "{transcript}"
    );
    assert!(transcript.contains(r#""turnId":"tu-1""#), "{transcript}");
    harness.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// A checkpoint under `ResumeIfAvailable` resumes its thread instead of starting a new one.
#[tokio::test]
async fn a_checkpoint_resumes_its_thread() {
    let cli = script(&[&turn_arm(&[("turn/completed", COMPLETED)])]);
    let checkpoint = HarnessCheckpoint {
        provider: "codex".to_owned(),
        schema_version: "1".to_owned(),
        agent_id: "agent-1".to_owned(),
        host_session_id: "host-session".to_owned(),
        checkpoint_id: "checkpoint-1".to_owned(),
        sequence: 1,
        data: object(json!({"thread_id": "th-9", "resumed": true})),
        provider_session_id: Some("th-9".to_owned()),
        revision: None,
    };
    let context = context()
        .with_resume_policy(ResumePolicy::ResumeIfAvailable)
        .with_checkpoint(checkpoint);
    let harness = harness_for(&cli);
    harness.start(context).await.expect("the harness starts");
    let mut cursor = harness.events();
    harness
        .send(HarnessInput::text("hello"), DeliveryMode::FollowUp)
        .await
        .expect("the input is accepted");
    let events = until_terminal(cursor.as_mut()).await;
    assert_eq!(terminal_result(&events).status, TurnStatus::Completed);
    let transcript = transcript(&cli);
    assert!(
        transcript.contains(r#""method":"thread/resume""#),
        "{transcript}"
    );
    assert!(transcript.contains(r#""threadId":"th-9""#), "{transcript}");
    assert!(
        !transcript.contains(r#""method":"thread/start""#),
        "{transcript}"
    );
    assert_eq!(harness.provider_session_id(), Some("th-9"));
    harness.stop().await.expect("the harness stops");
}

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

/// A stream that ends without `turn/completed` fails with `CODEX_MISSING_TURN_COMPLETED`.
#[tokio::test]
async fn a_stream_without_turn_completed_fails_the_turn() {
    let mut body = String::from(HEAD);
    body.push_str(
        r#"    *'"method":"turn/start"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"turn":{"id":"tu-1"}}}\n' "$id"
      printf '%s\n' '{"jsonrpc":"2.0","method":"item/agentMessage/delta","params":{"itemId":"msg-1","delta":"partial"}}'
      exit 0
      ;;
"#,
    );
    body.push_str(TAIL);
    let cli = FakeCli::script(&body);
    let harness = harness_for(&cli);
    harness.start(context()).await.expect("the harness starts");
    let mut cursor = harness.events();
    harness
        .send(HarnessInput::text("go"), DeliveryMode::FollowUp)
        .await
        .expect("the input is accepted");
    let events = until_terminal(cursor.as_mut()).await;
    let result = terminal_result(&events);
    assert_eq!(result.status, TurnStatus::Failed);
    let error = result.error.as_ref().expect("a normalized error");
    assert_eq!(error.code.as_deref(), Some("CODEX_MISSING_TURN_COMPLETED"));
    assert_eq!(
        error.message,
        "Codex ended the stream without completing the turn"
    );
    harness.stop().await.expect("the harness stops");
}
