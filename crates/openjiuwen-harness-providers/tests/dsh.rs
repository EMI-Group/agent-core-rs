#![cfg(all(feature = "vendor", unix))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value
)]
//! End-to-end suite of the DeepSeek Harness (DSH) provider.
//!
//! Every case runs the real adapter over a real child process: [`FakeCli`] writes a POSIX shell
//! script that speaks the DSH JSON-RPC stdio transport (it answers `initialize` and
//! `session/prompt`, then publishes `session.event` notifications). The suite therefore covers the
//! whole path — configuration validation, binary resolution, `DSH_HOME`, argv/env propagation,
//! frame projection, turn endings, unsupported commands and child sessions — offline and
//! deterministically.
//!
//! No network, no vendor SDK, no `tokio::time::sleep`: ordering is observed through the harness
//! cursor or the adapter's chunk stream, and every await that could hang is bounded.
//!
//! Python counterpart: `tests/unit_tests/harness_providers/dsh/`.
mod support;

use std::path::Path;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::LLM_OUTPUT;
use openjiuwen_agent::AgentChunk;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, HarnessCheckpoint, HarnessContext, HarnessErrorKind,
    HarnessEventCursor, HarnessEventPayload, HarnessInput, HarnessProtocol, HarnessState,
    ItemEventKind, JsonObject, JsonValue, ResumePolicy, TurnEventKind, TurnStatus,
    TurnTerminationKind,
};
use openjiuwen_harness_providers::{
    create_harness, HarnessIOAdapter, HarnessIOAdapterConfig, TURN_LIFECYCLE_KIND,
};
use serde_json::json;

use support::FakeCli;

/// Timeout applied to every await that must not hang the suite.
const BOUND: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Fake runtime
// ---------------------------------------------------------------------------

/// Builds the body of a fake `dsh` runtime that emits `events` for a prompt, then keeps serving.
///
/// The script records its argv, `DSH_HOME` and content of every request line in the transcript, so
/// a test can assert the whole launch contract. `events` is spliced verbatim into the
/// `session/prompt` branch and may use the `respond`/`emit`/`notify` helpers and `$sid`.
fn runtime(events: &str) -> String {
    const TEMPLATE: &str = r#"
printf 'ARGV %s\n' "$*" >> "$FAKE_CLI_LOG"
printf 'HOME %s\n' "${DSH_HOME:-}" >> "$FAKE_CLI_LOG"
printf 'CWD %s\n' "$PWD" >> "$FAKE_CLI_LOG"
sid=""
respond() { printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$1"; }
emit() { printf '{"jsonrpc":"2.0","method":"session.event","params":{"sessionId":"%s","event":%s}}\n' "$sid" "$1"; }
notify() { printf '{"jsonrpc":"2.0","method":"%s","params":%s}\n' "$1" "$2"; }
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$FAKE_CLI_LOG"
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"session/prompt"'*)
      sid=$(printf '%s' "$line" | sed -n 's/.*"sessionId":"\([^"]*\)".*/\1/p')
      respond "$id"
@EVENTS@
      ;;
    *'"method":"shutdown"'*)
      respond "$id"
      exit 0
      ;;
    *)
      respond "$id"
      ;;
  esac
done
"#;
    TEMPLATE.replace("@EVENTS@", events)
}

/// A chunk projected from one `assistant/chunk` text delta.
fn delta(index: u64, text: &str) -> String {
    format!(
        "      emit '{{\"type\":\"assistant/chunk\",\"data\":{{\"turn\":1,\"step\":1,\
         \"chunk\":{{\"type\":\"text-delta\",\"index\":{index},\"text\":\"{text}\"}}}}}}'"
    )
}

/// A chunk closing the text block at `index`.
fn block_end(index: u64, text: &str) -> String {
    format!(
        "      emit '{{\"type\":\"assistant/chunk\",\"data\":{{\"turn\":1,\"step\":1,\
         \"chunk\":{{\"type\":\"block-end\",\"index\":{index},\
         \"block\":{{\"type\":\"text\",\"text\":\"{text}\"}}}}}}}}'"
    )
}

/// A usage chunk of the current step.
fn usage(input: u64, output: u64) -> String {
    format!(
        "      emit '{{\"type\":\"assistant/chunk\",\"data\":{{\"turn\":1,\"step\":1,\
         \"chunk\":{{\"type\":\"usage\",\"usage\":{{\"inputTokens\":{input},\
         \"outputTokens\":{output},\"cacheReadTokens\":0,\"cacheWriteTokens\":0}}}}}}}}'"
    )
}

/// A `turn/end` event carrying `reason` (a JSON object literal).
fn turn_end(reason: &str) -> String {
    format!("      emit '{{\"type\":\"turn/end\",\"data\":{{\"reason\":{reason}}}}}'")
}

/// One `tool/call` event.
fn tool_call(call_id: &str, name: &str, arguments: &str) -> String {
    format!(
        "      emit '{{\"type\":\"tool/call\",\"data\":{{\"callId\":\"{call_id}\",\
         \"name\":\"{name}\",\"arguments\":\"{arguments}\"}}}}'"
    )
}

/// One `tool/result` event for `call_id`.
fn tool_result(call_id: &str, content: &str) -> String {
    format!(
        "      emit '{{\"type\":\"tool/result\",\"data\":{{\"message\":{{\
         \"source\":{{\"callId\":\"{call_id}\"}},\"content\":\"{content}\"}}}}}}'"
    )
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Builds the configuration every case starts from: the fake binary plus `home` as `DSH_HOME`.
fn base_config(cli: &FakeCli, home: &Path) -> serde_json::Map<String, JsonValue> {
    let mut config = serde_json::Map::new();
    config.insert("dsh_bin".to_owned(), json!(cli.path().to_string_lossy()));
    config.insert("dsh_home".to_owned(), json!(home.to_string_lossy()));
    config
}

/// Creates a DSH harness through the registry, never starting it.
fn create(config: serde_json::Map<String, JsonValue>) -> Result<Box<dyn HarnessProtocol>, String> {
    let object: JsonObject = config.into_iter().collect();
    create_harness("dsh", object).map_err(|error| error.to_string())
}

/// Returns the host context every case starts from.
fn context() -> HarnessContext {
    HarnessContext::new("agent", "agent-1", "host-session", "system prompt")
}

/// Starts `harness` and returns its cycle cursor, consuming the `Terminated → Idle` transition.
async fn started(harness: &dyn HarnessProtocol) -> Box<dyn HarnessEventCursor> {
    harness
        .start(context())
        .await
        .expect("the fake runtime accepts the launch");
    let mut cursor = harness.events();
    let _ = until_turn_event(cursor.as_mut(), |_| true).await;
    cursor
}

/// Reads events until `stop` matches, returning everything read.
async fn until_turn_event(
    cursor: &mut dyn HarnessEventCursor,
    stop: impl Fn(&openjiuwen_harness_protocol::HarnessEvent) -> bool,
) -> Vec<openjiuwen_harness_protocol::HarnessEvent> {
    tokio::time::timeout(BOUND, async {
        let mut events = Vec::new();
        while let Some(event) = cursor
            .next_event()
            .await
            .expect("the cycle cursor is healthy")
        {
            let done = stop(&event);
            events.push(event);
            if done {
                break;
            }
        }
        events
    })
    .await
    .expect("the harness must publish events in time")
}

/// Reads the cycle cursor until the turn reaches a terminal lifecycle event.
async fn until_terminal(
    cursor: &mut dyn HarnessEventCursor,
) -> Vec<openjiuwen_harness_protocol::HarnessEvent> {
    until_turn_event(cursor, |event| {
        matches!(&event.event, HarnessEventPayload::TurnLifecycle(lifecycle) if lifecycle.kind.is_terminal())
    })
    .await
}

/// Returns the terminal turn result of `events`.
fn terminal(
    events: &[openjiuwen_harness_protocol::HarnessEvent],
) -> openjiuwen_harness_protocol::TurnResult {
    events
        .iter()
        .find_map(|event| match &event.event {
            HarnessEventPayload::TurnLifecycle(lifecycle) if lifecycle.kind.is_terminal() => {
                lifecycle.result.clone()
            }
            _ => None,
        })
        .expect("the turn published its terminal result")
}

/// Runs one turn on `harness` and returns the events up to its terminal lifecycle.
async fn one_turn(
    harness: &dyn HarnessProtocol,
    cursor: &mut dyn HarnessEventCursor,
    text: &str,
) -> Vec<openjiuwen_harness_protocol::HarnessEvent> {
    harness
        .send(HarnessInput::text(text), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    until_terminal(cursor).await
}

/// Returns the sequence of `(provider, event_type)` pairs of the provider events in `events`.
fn provider_events(events: &[openjiuwen_harness_protocol::HarnessEvent]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::Provider(provider) => {
                Some((provider.provider.clone(), provider.event_type.clone()))
            }
            _ => None,
        })
        .collect()
}

/// Returns the answer-delivering output payloads of `events`, in order.
fn outputs(events: &[openjiuwen_harness_protocol::HarnessEvent]) -> Vec<JsonValue> {
    events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::Output(output) if output.channel.as_str() == "answer" => {
                Some(output.content.clone())
            }
            _ => None,
        })
        .collect()
}

/// Reads chunks from `stream` until the turn lifecycle reports a terminal state.
async fn chunks_until_terminal(stream: &mut BoxStream<'static, AgentChunk>) -> Vec<AgentChunk> {
    tokio::time::timeout(BOUND, async {
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            let terminal = chunk.kind == TURN_LIFECYCLE_KIND
                && matches!(
                    chunk.payload.get("kind").and_then(JsonValue::as_str),
                    Some("finished" | "failed" | "aborted")
                );
            chunks.push(chunk);
            if terminal {
                break;
            }
        }
        chunks
    })
    .await
    .expect("the adapter must publish the turn in time")
}

/// Returns the chunk kinds of `chunks`, in order.
fn kinds(chunks: &[AgentChunk]) -> Vec<&str> {
    chunks.iter().map(|chunk| chunk.kind.as_str()).collect()
}

// ---------------------------------------------------------------------------
// Launch contract
// ---------------------------------------------------------------------------

/// A fake runtime starts the cycle, reports a `dsh-` session id, serves a prompt and stops cleanly.
#[tokio::test]
async fn create_and_start_open_a_dsh_session() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"completed"}"#)));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(harness.provider_session_id(), None);

    let mut cursor = started(harness.as_ref()).await;
    assert_eq!(harness.state(), HarnessState::Idle);
    let session = harness
        .provider_session_id()
        .expect("the cycle reports its session")
        .to_owned();
    assert!(session.starts_with("dsh-"), "{session}");

    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    assert_eq!(terminal(&events).status, TurnStatus::Completed);

    // The script dumps an `ARGV`/`HOME`/`CWD` header before it reads stdin, so the wire requests
    // are located by content instead of by position.
    let transcript = cli.lines();
    let initialize = transcript
        .iter()
        .position(|line| line.contains(r#""method":"initialize""#))
        .unwrap_or_else(|| panic!("the runtime is initialized: {transcript:?}"));
    assert!(
        transcript[initialize].contains(r#""model":""#),
        "initialize names the model: {transcript:?}"
    );
    let prompt = transcript
        .iter()
        .position(|line| line.contains(r#""method":"session/prompt""#))
        .unwrap_or_else(|| panic!("the prompt is written: {transcript:?}"));
    assert!(
        prompt > initialize,
        "the prompt follows the initialization: {transcript:?}"
    );
    assert!(
        transcript[prompt].contains(&format!(r#""sessionId":"{session}""#)),
        "the prompt carries the runtime session: {}",
        transcript[prompt]
    );
    assert!(
        transcript[prompt].contains(r#""contentBlocks":[{"type":"text","text":"hello"}]"#),
        "the prompt carries the input: {}",
        transcript[prompt]
    );

    harness.stop().await.expect("the runtime stops");
    assert_eq!(harness.state(), HarnessState::Terminated);
}

/// A configured binary that cannot be executed names `dsh` instead of reporting a missing SDK.
#[tokio::test]
async fn a_missing_runtime_is_a_protocol_error() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let mut config = base_config(&FakeCli::script("# unused\n"), home.path());
    config.insert("dsh_bin".to_owned(), json!("/nonexistent/dsh"));
    let Err(message) = create(config) else {
        panic!("a missing runtime must not create a harness");
    };
    assert!(message.contains("dsh"), "{message}");
    assert!(!message.contains("SDK required"), "{message}");
}

/// The home directory is mandatory: the adapter never guesses `~/.dsh`.
#[tokio::test]
async fn a_missing_home_is_a_protocol_error() {
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"completed"}"#)));
    let mut config = base_config(&cli, Path::new(""));
    config.remove("dsh_home");
    // An explicitly empty value overrides a stray `DSH_HOME` of the test process.
    config.insert("env".to_owned(), json!({"DSH_HOME": ""}));
    let Err(message) = create(config) else {
        panic!("a missing DSH_HOME must not create a harness");
    };
    assert!(message.contains("DSH_HOME"), "{message}");
    assert!(
        message.contains("set `dsh_home` or the DSH_HOME environment variable"),
        "{message}"
    );
}

/// An unknown configuration key is rejected before anything is spawned.
#[tokio::test]
async fn an_unknown_configuration_field_is_a_validation_error() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"completed"}"#)));
    let mut config = base_config(&cli, home.path());
    config.insert("nope".to_owned(), json!(true));
    let Err(message) = create(config) else {
        panic!("an unknown key must not create a harness");
    };
    assert!(
        message.contains("unknown DSH configuration fields"),
        "{message}"
    );
    assert!(cli.lines().is_empty(), "nothing may be spawned");
}

/// The profile, the patches, the working directory and `DSH_HOME` reach the child.
#[tokio::test]
async fn the_launch_contract_reaches_the_runtime() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let project = tempfile::tempdir().expect("a temporary project");
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"completed"}"#)));
    let mut config = base_config(&cli, home.path());
    config.insert("profile".to_owned(), json!("fast"));
    config.insert("patches".to_owned(), json!(["one.patch"]));
    config.insert("cwd".to_owned(), json!(project.path().to_string_lossy()));

    let harness = create(config).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let _ = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;

    let transcript = cli.lines();
    let argv = transcript
        .iter()
        .find(|line| line.starts_with("ARGV "))
        .unwrap_or_else(|| panic!("the script dumps its argv: {transcript:?}"));
    let patch = project.path().join("one.patch");
    assert_eq!(
        argv,
        &format!("ARGV --profile fast --patch {}", patch.display())
    );
    assert!(
        transcript
            .iter()
            .any(|line| line == &format!("HOME {}", home.path().display())),
        "{transcript:?}"
    );
    assert!(
        transcript
            .iter()
            .any(|line| line == &format!("CWD {}", project.path().display())),
        "{transcript:?}"
    );
    harness.stop().await.expect("the runtime stops");
}

// ---------------------------------------------------------------------------
// Turn projection
// ---------------------------------------------------------------------------

/// One turn projects deltas, a finalized answer, the tool pair and the terminal lifecycle.
#[tokio::test]
async fn a_turn_projects_outputs_and_the_tool_pair_through_the_io_adapter() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let events = [
        delta(0, "Hel"),
        delta(0, "lo"),
        block_end(0, "Hello"),
        usage(10, 2),
        tool_call("call-1", "shell", "{\\\"cmd\\\":\\\"ls\\\"}"),
        tool_result("call-1", "ok"),
        turn_end(r#"{"kind":"completed"}"#),
    ]
    .join("\n");
    let cli = FakeCli::script(&runtime(&events));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let adapter = HarnessIOAdapter::new(
        std::sync::Arc::from(harness),
        HarnessIOAdapterConfig {
            emit_turn_lifecycle: true,
            ..HarnessIOAdapterConfig::default()
        },
    );
    adapter.start(context()).await.expect("the runtime starts");
    let session = adapter.session_id().expect("the cycle reports its session");
    assert!(session.starts_with("dsh-"), "{session}");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    adapter
        .send(json!("hello"), false)
        .await
        .expect("the input is accepted");
    let chunks = chunks_until_terminal(&mut stream).await;

    assert_eq!(
        kinds(&chunks),
        [
            TURN_LIFECYCLE_KIND,
            LLM_OUTPUT,
            LLM_OUTPUT,
            LLM_OUTPUT,
            "tool_call",
            "tool_result",
            TURN_LIFECYCLE_KIND,
        ]
    );
    assert_eq!(chunks[0].payload["kind"], json!("started"));
    assert_eq!(chunks[1].payload["content"], json!("Hel"));
    assert_eq!(chunks[1].payload["operation"], json!("delta"));
    assert_eq!(chunks[1].payload["channel"], json!("answer"));
    assert_eq!(
        chunks[1].payload["output_id"],
        json!(format!("dsh-output:1:1:0:answer"))
    );
    assert_eq!(chunks[2].payload["content"], json!("lo"));
    assert_eq!(chunks[3].payload["content"], json!("Hello"));
    assert_eq!(chunks[3].payload["operation"], json!("final"));
    assert_eq!(chunks[4].payload["name"], json!("shell"));
    assert_eq!(chunks[4].payload["tool_call_id"], json!("call-1"));
    assert_eq!(chunks[4].payload["arguments"], json!(r#"{"cmd":"ls"}"#));
    assert_eq!(chunks[5].payload["tool_name"], json!("shell"));
    assert_eq!(chunks[5].payload["result"], json!("ok"));
    assert_eq!(chunks[6].payload["kind"], json!("finished"));
    assert_eq!(chunks[6].payload["final_output"], json!("Hello"));
    adapter.stop().await.expect("the runtime stops");
}

/// The prompt carries the session id and one text block; usage reaches the turn result.
#[tokio::test]
async fn the_prompt_and_usage_follow_the_wire_contract() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let events = [
        delta(0, "done"),
        usage(11, 3),
        turn_end(r#"{"kind":"completed"}"#),
    ]
    .join("\n");
    let cli = FakeCli::script(&runtime(&events));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let session = harness
        .provider_session_id()
        .expect("the cycle reports its session")
        .to_owned();
    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    let result = terminal(&events);
    assert_eq!(result.status, TurnStatus::Completed);
    assert_eq!(outputs(&events), vec![json!("done")]);
    let usage = result.usage.expect("the runtime reported usage");
    assert_eq!(usage.input_tokens, Some(11));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.total_tokens, Some(14));

    let prompt = cli
        .lines()
        .into_iter()
        .find(|line| line.contains(r#""method":"session/prompt""#))
        .expect("the prompt was written");
    assert!(
        prompt.contains(&format!(r#""sessionId":"{session}""#)),
        "{prompt}"
    );
    assert!(
        prompt.contains(r#""contentBlocks":[{"type":"text","text":"hello"}]"#),
        "{prompt}"
    );
    harness.stop().await.expect("the runtime stops");
}

// ---------------------------------------------------------------------------
// Turn endings
// ---------------------------------------------------------------------------

/// A blocked turn is a policy abort.
#[tokio::test]
async fn a_blocked_turn_is_a_policy_abort() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"blocked"}"#)));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    let result = terminal(&events);
    assert_eq!(result.status, TurnStatus::Interrupted);
    let termination = result
        .termination
        .expect("the turn reports its termination");
    assert_eq!(termination.kind, TurnTerminationKind::Policy);
    assert!(
        events.iter().any(|event| matches!(
            &event.event,
            HarnessEventPayload::TurnLifecycle(lifecycle)
                if lifecycle.kind == TurnEventKind::Aborted
        )),
        "the turn publishes an aborted lifecycle"
    );
}

/// A failed turn keeps the provider code and redacts the provider diagnostics.
#[tokio::test]
async fn a_failed_turn_keeps_the_provider_code() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let reason = r#"{"kind":"error","error":{"code":"rate_limit","status":429,"requestId":"req-1","message":"secret-provider-detail"}}"#;
    let cli = FakeCli::script(&runtime(&turn_end(reason)));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    let result = terminal(&events);
    assert_eq!(result.status, TurnStatus::Failed);
    let error = result.error.expect("the turn reports its error");
    assert_eq!(error.message, "DeepSeek Harness reported a failed turn");
    assert_eq!(error.code.as_deref(), Some("rate_limit"));
    assert_eq!(error.category.as_deref(), Some("provider"));
    assert_eq!(
        error.provider_data.get("error"),
        Some(&json!({"code": "rate_limit", "status": 429, "requestId": "req-1"}))
    );
    let surfaced = format!("{events:?}");
    assert!(
        !surfaced.contains("secret-provider-detail"),
        "the provider message is redacted"
    );
    harness.stop().await.expect("the runtime stops");
}

/// A stream that ends without a native turn ending is a protocol failure.
#[tokio::test]
async fn a_stream_without_turn_end_fails() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let events = [delta(0, "partial"), "      exit 0".to_owned()].join("\n");
    let cli = FakeCli::script(&runtime(&events));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    let result = terminal(&events);
    assert_eq!(result.status, TurnStatus::Failed);
    let error = result.error.expect("the turn reports its error");
    assert_eq!(
        error.message,
        "DeepSeek Harness became idle without ending a native turn"
    );
    assert_eq!(error.code.as_deref(), Some("DSH_MISSING_TURN_END"));
    assert_eq!(error.category.as_deref(), Some("protocol"));
    let _ = harness.stop().await;
}

/// An ending the adapter does not recognize is a protocol failure, not a silent completion.
#[tokio::test]
async fn an_unknown_turn_ending_fails() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"teleported"}"#)));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    let error = terminal(&events).error.expect("the turn reports its error");
    assert_eq!(error.code.as_deref(), Some("DSH_UNKNOWN_FINISH_REASON"));
    let _ = harness.stop().await;
}

/// A subagent's lifetime is an item lifecycle; a child session's event is forwarded as a provider
/// event and never joins the root turn.
#[tokio::test]
async fn subagent_lifecycles_become_items_and_child_events_are_forwarded() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let child = r#"{"type":"assistant/chunk","data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"child text"}}}"#;
    let events = [
        format!("      notify 'subagent.started' '{{\"childSessionId\":\"child-1\"}}'"),
        format!(
            "      printf '{{\"jsonrpc\":\"2.0\",\"method\":\"session.event\",\
             \"params\":{{\"sessionId\":\"child-1\",\"event\":{child}}}}}\\n'"
        ),
        format!("      notify 'subagent.finished' '{{\"childSessionId\":\"child-1\"}}'"),
        delta(0, "root"),
        turn_end(r#"{"kind":"completed"}"#),
    ]
    .join("\n");
    let cli = FakeCli::script(&runtime(&events));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;
    let events = one_turn(harness.as_ref(), cursor.as_mut(), "hello").await;
    assert_eq!(terminal(&events).status, TurnStatus::Completed);
    assert_eq!(outputs(&events), vec![json!("root")]);

    // `subagent.*` notifications describe one child session's lifetime, so the accumulator turns
    // them into subagent items instead of provider events.
    let items: Vec<(String, ItemEventKind, JsonValue)> = events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::ItemLifecycle(item) if item.item_type == "subagent" => Some((
                item.item_type.clone(),
                item.kind,
                item.data
                    .get("childSessionId")
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        items,
        vec![
            (
                "subagent".to_owned(),
                ItemEventKind::Started,
                json!("child-1")
            ),
            (
                "subagent".to_owned(),
                ItemEventKind::Completed,
                json!("child-1")
            ),
        ]
    );

    // Only the child's own `session.event` frames are forwarded as provider events; the root
    // turn additionally publishes its native `turn/end`.
    let forwarded = provider_events(&events);
    assert!(
        forwarded.contains(&("dsh".to_owned(), "child.session.event".to_owned())),
        "{forwarded:?}"
    );
    assert!(
        !forwarded
            .iter()
            .any(|(_, event_type)| event_type.starts_with("subagent.")),
        "subagent notifications are items, not provider events: {forwarded:?}"
    );
    let surfaced = format!("{events:?}");
    assert!(
        surfaced.contains("child text"),
        "the child event is surfaced"
    );
    harness.stop().await.expect("the runtime stops");
}

// ---------------------------------------------------------------------------
// Unsupported commands
// ---------------------------------------------------------------------------

/// The pinned protocol cannot restore a checkpoint, so a required resume is refused unspawned.
#[tokio::test]
async fn a_required_resume_is_unsupported() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let cli = FakeCli::script(&runtime(&turn_end(r#"{"kind":"completed"}"#)));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let error = harness
        .start(context().with_resume_policy(ResumePolicy::RequireResume))
        .await
        .expect_err("the SDK server cannot resume");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        error.message(),
        "the DSH SDK server cannot restore protocol checkpoints"
    );
    assert!(cli.lines().is_empty(), "nothing may be spawned");

    let checkpoint = HarnessCheckpoint {
        provider: "dsh".to_owned(),
        schema_version: "1".to_owned(),
        agent_id: "agent-1".to_owned(),
        host_session_id: "host-session".to_owned(),
        checkpoint_id: "cp-1".to_owned(),
        sequence: 1,
        data: JsonObject::new(),
        provider_session_id: Some("dsh-session".to_owned()),
        revision: None,
    };
    let error = harness
        .start(context().with_checkpoint(checkpoint))
        .await
        .expect_err("the SDK server cannot restore a checkpoint");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert!(cli.lines().is_empty(), "nothing may be spawned");
}

/// Abort, pause and checkpoint export stay unimplemented because the card declares no capability.
#[tokio::test]
async fn abort_pause_and_export_are_unsupported() {
    let home = tempfile::tempdir().expect("a temporary DSH home");
    // The runtime answers the prompt, publishes one delta and then blocks on stdin, so the turn
    // stays open while the commands are exercised.
    let cli = FakeCli::script(&runtime(&delta(0, "partial")));
    let harness = create(base_config(&cli, home.path())).expect("the runtime resolves");
    let mut cursor = started(harness.as_ref()).await;

    // With no active turn there is nothing to interrupt: `abort` is a documented no-op.
    harness
        .abort(AbortMode::Graceful)
        .await
        .expect("aborting an idle harness is a no-op");

    harness
        .send(HarnessInput::text("hello"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    let opened = until_turn_event(cursor.as_mut(), |event| {
        matches!(&event.event, HarnessEventPayload::Output(_))
    })
    .await;
    assert!(
        opened
            .iter()
            .any(|event| matches!(&event.event, HarnessEventPayload::Output(_))),
        "the turn is running before the commands are issued"
    );
    assert_eq!(harness.state(), HarnessState::Running);

    // The capability gate only applies to a running turn, so every mode is refused here.
    for mode in [AbortMode::Graceful, AbortMode::Force] {
        let error = harness.abort(mode).await.expect_err("DSH cannot abort");
        assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    }
    let error = harness.pause().await.expect_err("DSH cannot pause");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    let error = harness
        .export_checkpoint()
        .await
        .expect_err("DSH exports no checkpoint");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        harness.state(),
        HarnessState::Running,
        "the refused commands leave the turn alone"
    );
    harness.stop().await.expect("the runtime stops");
    assert_eq!(harness.state(), HarnessState::Terminated);
}
