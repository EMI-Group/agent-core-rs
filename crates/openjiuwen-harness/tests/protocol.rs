//! Offline integration suite of the native harness protocol.
//!
//! Every test drives the real [`NativeHarness`] over a kernel `DeepAgent` built on
//! `openjiuwen_test_support::ScriptedModel` and a temporary workspace, then reads
//! the host-facing `openjiuwen_harness_protocol` surface: lifecycle, delivery
//! modes, the observation cursor, checkpoint export, capability gating, and the
//! provider create mapping. No network is touched: provider-created harnesses
//! are never started (starting one would resolve a live model), and every wait
//! is bounded by a short `tokio::time::timeout`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::AgentCard;
use openjiuwen_foundation::llm::Model;
use openjiuwen_harness::agent::DeepAgentBuilder;
use openjiuwen_harness::protocol::{
    native_card, NativeHarness, NativeHarnessOptions, NativeHarnessProvider, HARNESS_NAME,
    PROVIDER_VERSION,
};
use openjiuwen_harness::schema::DeepAgentConfig;
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_harness_protocol::{
    DeliveryMode, HarnessCapability, HarnessCheckpoint, HarnessContext, HarnessError,
    HarnessErrorKind, HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessInput,
    HarnessProtocol, HarnessProvider, HarnessState, JsonObject, JsonValue, McpServerConfig,
    McpTransport, MessageRole, ModelChoice, ModelConfigProvider, ModelEndpoint, OutputChannel,
    TurnEventKind, TurnLifecycleEvent, TurnStatus,
};
use openjiuwen_test_support::ScriptedModel;
use tempfile::TempDir;

/// Safety net applied to every await that must not hang the suite.
const WAIT: Duration = Duration::from_secs(5);

/// Native checkpoint payload keys the host treats as opaque but tests inspect.
const CHECKPOINT_KEYS: [&str; 10] = [
    "schema_version",
    "session_id",
    "agent_id",
    "host_session_id",
    "iteration",
    "pending_follow_ups",
    "plan_mode",
    "turn_counter",
    "model",
    "effort",
];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One harness under test plus the temporary workspace it is rooted in.
struct HarnessFixture {
    /// Keeps the workspace directory alive for the whole test.
    _workspace: TempDir,
    /// Harness under test.
    harness: NativeHarness,
}

/// Builds a scripted-model harness and returns it with a handle to its model.
///
/// The handle shares the script and the call recording with the model owned by
/// the agent, so a test can assert that the model really ran.
fn harness_with(scripted: ScriptedModel) -> (HarnessFixture, ScriptedModel) {
    let workspace = TempDir::new().expect("temporary workspace");
    let config = DeepAgentConfig {
        model_name: Some("scripted-model".to_owned()),
        ..DeepAgentConfig::default()
    };
    let agent = DeepAgentBuilder::new(
        AgentCard::new("deep_agent").with_description("protocol integration suite"),
    )
    .config(config)
    .model(Arc::new(scripted.clone()) as Arc<dyn Model>)
    .workspace(Workspace::new(workspace.path()))
    .build()
    .expect("the deep agent builds");
    let harness = NativeHarness::new(agent, NativeHarnessOptions::new());
    (
        HarnessFixture {
            _workspace: workspace,
            harness,
        },
        scripted,
    )
}

/// The minimal host context: no system prompt, no host services, no MCP servers.
fn context() -> HarnessContext {
    HarnessContext::new("deep_agent", "agent-1", "host-1", "")
}

// ---------------------------------------------------------------------------
// Event helpers
// ---------------------------------------------------------------------------

/// Reads the next cursor event, failing fast on timeout or a cursor error.
async fn next_event(cursor: &mut Box<dyn HarnessEventCursor>) -> Option<HarnessEvent> {
    tokio::time::timeout(WAIT, cursor.next_event())
        .await
        .expect("timed out waiting for a protocol event")
        .expect("the observation cursor reported an error")
}

/// Reads the error the cursor reports on its next call.
async fn next_error(cursor: &mut Box<dyn HarnessEventCursor>) -> HarnessError {
    tokio::time::timeout(WAIT, cursor.next_event())
        .await
        .expect("timed out waiting for the cursor error")
        .expect_err("expected the cursor to report an error")
}

/// Reads events until `predicate` accepts one, returning everything read.
async fn drain_until(
    cursor: &mut Box<dyn HarnessEventCursor>,
    mut predicate: impl FnMut(&HarnessEvent) -> bool,
) -> Vec<HarnessEvent> {
    let mut events = Vec::new();
    loop {
        let Some(event) = next_event(cursor).await else {
            panic!("the observation cursor ended before the expected event");
        };
        let matched = predicate(&event);
        events.push(event);
        if matched {
            return events;
        }
    }
}

/// Returns the turn lifecycle payload `event` carries, when it is one.
fn lifecycle(event: &HarnessEvent) -> Option<&TurnLifecycleEvent> {
    match &event.event {
        HarnessEventPayload::TurnLifecycle(payload) => Some(payload),
        _ => None,
    }
}

/// Returns whether `event` is the `kind` lifecycle event of `turn_id`.
fn is_turn_event(event: &HarnessEvent, turn_id: &str, kind: TurnEventKind) -> bool {
    event.turn_id.as_deref() == Some(turn_id)
        && matches!(lifecycle(event), Some(payload) if payload.kind == kind)
}

/// Returns the terminal result of the single finished turn in `events`.
fn finished_result(events: &[HarnessEvent]) -> &openjiuwen_harness_protocol::TurnResult {
    let finished = events
        .iter()
        .find(|event| is_turn_event(event, "turn-1", TurnEventKind::Finished))
        .expect("the run finished its first turn");
    lifecycle(finished)
        .and_then(|payload| payload.result.as_ref())
        .expect("the terminal event carries a result")
}

/// Runs one scripted turn and returns every observation it produced.
async fn run_one_turn(fixture: &HarnessFixture, text: &str) -> Vec<HarnessEvent> {
    let mut cursor = fixture.harness.events();
    fixture
        .harness
        .send(HarnessInput::text(text), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;
    cursor.close().await.expect("the cursor closes");
    events
}

/// Waits until the latest exported checkpoint reaches `minimum` in sequence.
///
/// The `TurnCompleted` checkpoint is published after the terminal turn event,
/// so the test polls the (infallible) export instead of assuming the writer has
/// already run.
async fn checkpoint_at_least(harness: &NativeHarness, minimum: u64) -> HarnessCheckpoint {
    let waiting = async {
        loop {
            if let Some(checkpoint) = harness
                .export_checkpoint()
                .await
                .expect("checkpoint export is infallible")
            {
                if checkpoint.sequence >= minimum {
                    return checkpoint;
                }
            }
            tokio::task::yield_now().await;
        }
    };
    tokio::time::timeout(WAIT, waiting)
        .await
        .expect("the harness never published the expected checkpoint")
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_lifecycle_streams_the_scripted_answer_and_terminates_on_stop() {
    let (fixture, model) = harness_with(ScriptedModel::text("hello-world"));
    let harness = &fixture.harness;

    assert_eq!(harness.state(), HarnessState::Terminated);
    assert!(!harness.started());
    assert_eq!(harness.provider_session_id(), None);

    harness.start(context()).await.expect("start succeeds");
    assert_eq!(harness.state(), HarnessState::Idle);
    assert!(harness.started());
    assert_eq!(harness.provider_session_id(), Some("host-1"));

    let mut cursor = harness.events();
    let receipt = harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    assert_eq!(receipt.turn_id, "turn-1");
    assert_eq!(receipt.message_id, "message-1");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Auto);

    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;

    let started = events
        .iter()
        .find(|event| is_turn_event(event, "turn-1", TurnEventKind::Started))
        .expect("the turn starts");
    assert_eq!(started.correlation_id.as_deref(), Some("message-1"));
    assert_eq!(started.provider_session_id.as_deref(), Some("host-1"));

    let answer: String = events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::Output(output) if output.channel == OutputChannel::Answer => {
                output.content.as_str()
            }
            _ => None,
        })
        .collect();
    assert_eq!(answer, "hello-world", "the answer deltas spell the answer");

    let result = finished_result(&events);
    assert_eq!(result.status, TurnStatus::Completed);
    assert!(result.stop_reason.is_some(), "the stop reason is reported");
    assert_eq!(
        result.provider_data.get("result_type"),
        Some(&JsonValue::String("answer".to_owned()))
    );
    let assistant = result
        .messages
        .iter()
        .find(|message| message.role == MessageRole::Assistant)
        .expect("the assistant message is retained");
    let text = assistant
        .content
        .iter()
        .find(|block| block.kind == "text")
        .expect("the assistant message has a text block");
    assert_eq!(text.content, JsonValue::String("hello-world".to_owned()));

    assert_eq!(model.call_count(), 1, "the scripted model really ran");
    cursor.close().await.expect("the cursor closes");

    harness.stop().await.expect("stop succeeds");
    harness.stop().await.expect("stop is idempotent");
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert!(!harness.started());
    assert_eq!(harness.provider_session_id(), None);

    let error = harness
        .send(HarnessInput::text("again"), DeliveryMode::Auto)
        .await
        .expect_err("a stopped harness accepts no input");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "cannot send to a stopped native harness");
}

// ---------------------------------------------------------------------------
// Capability gating
// ---------------------------------------------------------------------------

#[tokio::test]
async fn start_rejects_context_mcp_servers_as_an_undeclared_capability() {
    // The only reachable gating branch of `NativeHarness` for an operation the
    // card does not declare: the card advertises `mcp_tools` but the native
    // harness never mounts context-supplied servers. The steer/abort/model
    // gating branches are unreachable by design because `NativeHarness::new`
    // hard-wires `native_card`, which declares every one of those capabilities.
    let (fixture, _model) = harness_with(ScriptedModel::text("unused"));
    let server = McpServerConfig {
        name: "search".to_owned(),
        transport: McpTransport::Stdio,
        command: vec!["mcp-search".to_owned()],
        url: None,
        instance: None,
        env: BTreeMap::new(),
        headers: BTreeMap::new(),
    };
    let error = fixture
        .harness
        .start(context().with_mcp_servers(vec![server]))
        .await
        .expect_err("context MCP servers are not mounted");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        error.message(),
        "declare MCP servers in the agent manifest; the native harness does not mount context MCP servers"
    );
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
    assert!(!fixture.harness.started());

    // The rejected attempt left the harness usable for a plain start.
    fixture
        .harness
        .start(context())
        .await
        .expect("the harness still starts without context MCP servers");
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
}

#[test]
fn model_config_gates_endpoint_and_auth_fallback_choices() {
    let provider = NativeHarnessProvider::new();
    let endpoint_choice = ModelChoice {
        model: Some("gpt-4o".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: Some("sk-test".to_owned()),
            provider: Some("openai".to_owned()),
        }),
    };
    let error = provider
        .model_config(&endpoint_choice, None)
        .expect_err("the card does not declare endpoint models");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        error.message(),
        "openjiuwen-native does not take endpoint models"
    );

    let builtin = ModelChoice {
        model: Some("gpt-4o".to_owned()),
        effort: None,
        endpoint: None,
    };
    let error = provider
        .model_config(&builtin, Some(&builtin))
        .expect_err("the card does not declare an auth fallback");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        error.message(),
        "openjiuwen-native has no authentication fallback"
    );

    // The declared path still works, so the gate is narrow rather than blanket.
    let fragment = provider
        .model_config(&builtin, None)
        .expect("a built-in model choice is supported");
    assert_eq!(fragment.get("model"), Some(&JsonValue::from("gpt-4o")));
}

// ---------------------------------------------------------------------------
// Observation cursors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn events_cursor_is_single_consumer_and_needs_an_active_cycle() {
    let (fixture, _model) = harness_with(ScriptedModel::text("unused"));

    let mut inactive = fixture.harness.events();
    let error = next_error(&mut inactive).await;
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "native harness has no active event cycle");

    fixture
        .harness
        .start(context())
        .await
        .expect("start succeeds");

    let mut first = fixture.harness.events();
    let mut second = fixture.harness.events();
    let error = next_error(&mut second).await;
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(
        error.message(),
        "the harness observation stream already has an active consumer"
    );
    // The rejected cursor never owned the lease, so closing it is a safe no-op.
    second.close().await.expect("a rejected cursor closes");

    let event = next_event(&mut first)
        .await
        .expect("the start transition is buffered");
    match &event.event {
        HarnessEventPayload::StateChanged(change) => {
            assert_eq!(change.old, HarnessState::Terminated);
            assert_eq!(change.new, HarnessState::Idle);
        }
        other => panic!("unexpected first event: {other:?}"),
    }
    first.close().await.expect("the cursor closes");

    // Closing released the lease, so a fresh cursor observes the same channel.
    let mut reopened = fixture.harness.events();
    fixture
        .harness
        .send(HarnessInput::text("again"), DeliveryMode::Auto)
        .await
        .expect("the reopened cycle accepts input");
    let events = drain_until(&mut reopened, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;
    assert_eq!(
        finished_result(&events).status,
        TurnStatus::Completed,
        "the reopened cursor observed the whole turn"
    );
    reopened.close().await.expect("the cursor closes");

    fixture.harness.stop().await.expect("stop succeeds");
    let mut after_stop = fixture.harness.events();
    let error = next_error(&mut after_stop).await;
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "native harness has no active event cycle");
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn export_checkpoint_covers_activation_and_a_completed_turn() {
    let (fixture, _model) = harness_with(ScriptedModel::text("checkpoint-answer"));
    let harness = &fixture.harness;
    harness.start(context()).await.expect("start succeeds");

    let activated = harness
        .export_checkpoint()
        .await
        .expect("checkpoint export is infallible")
        .expect("start publishes a checkpoint");
    assert_eq!(activated.provider, HARNESS_NAME);
    assert_eq!(activated.schema_version, "1");
    assert_eq!(activated.agent_id, "agent-1");
    assert_eq!(activated.host_session_id, "host-1");
    assert_eq!(activated.provider_session_id.as_deref(), Some("host-1"));
    assert_ne!(activated.checkpoint_id, "");
    assert!(
        activated.checkpoint_id.starts_with("checkpoint-"),
        "unexpected checkpoint id: {}",
        activated.checkpoint_id
    );
    assert!(activated.sequence >= 1, "the first sequence is 1 or later");
    for key in CHECKPOINT_KEYS {
        assert!(
            activated.data.contains_key(key),
            "the native payload is missing {key}"
        );
    }
    assert_eq!(
        activated.data.get("agent_id").and_then(JsonValue::as_str),
        Some("agent-1")
    );
    assert_eq!(
        activated.data.get("plan_mode").and_then(JsonValue::as_str),
        Some("normal")
    );
    assert_eq!(
        activated
            .data
            .get("turn_counter")
            .and_then(JsonValue::as_u64),
        Some(0)
    );

    let events = run_one_turn(&fixture, "hi").await;
    assert_eq!(finished_result(&events).status, TurnStatus::Completed);

    let completed = checkpoint_at_least(harness, activated.sequence + 1).await;
    assert_eq!(completed.provider, HARNESS_NAME);
    assert!(
        completed.sequence > activated.sequence,
        "the checkpoint sequence is monotonic: {} then {}",
        activated.sequence,
        completed.sequence
    );
    assert_eq!(
        completed
            .data
            .get("turn_counter")
            .and_then(JsonValue::as_u64),
        Some(1),
        "the completed checkpoint records the finished turn"
    );

    // Exporting is a pure read: the second call returns the same snapshot and
    // never rewinds or advances the sequence.
    let again = harness
        .export_checkpoint()
        .await
        .expect("checkpoint export is infallible")
        .expect("the completed checkpoint stays exportable");
    assert_eq!(again.checkpoint_id, completed.checkpoint_id);
    assert_eq!(again.sequence, completed.sequence);
}

// ---------------------------------------------------------------------------
// Provider create mapping
// ---------------------------------------------------------------------------

/// Builds a create mapping that names every key the native provider documents.
fn full_create_config(workspace: &std::path::Path) -> JsonObject {
    serde_json::from_value(serde_json::json!({
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "api_base": "https://example.invalid/v1",
        "temperature": 0.5,
        "workspace": workspace.display().to_string(),
        "enable_task_loop": true,
        "max_iterations": 3,
        "system_prompt": "be brief",
        "permissions": {"enabled": true},
        "language": "en",
    }))
    .expect("the create mapping is a JSON object")
}

#[tokio::test]
async fn provider_create_builds_an_unstarted_harness_without_contacting_a_model() {
    // `create` only constructs: the model named by the mapping is resolved
    // lazily in `start`, which this test never calls. `example.invalid` is
    // reserved by RFC 2606 and cannot resolve, so no request can escape even if
    // a future change made construction eager.
    let workspace = TempDir::new().expect("temporary workspace");
    let provider = NativeHarnessProvider::new();
    let harness = provider
        .create(full_create_config(workspace.path()))
        .expect("the documented create mapping is accepted");

    assert_eq!(harness.card().name, HARNESS_NAME);
    assert_eq!(harness.card().implementation_version, PROVIDER_VERSION);
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(harness.provider_session_id(), None);
    assert_eq!(harness.event_buffer_config().capacity, 1024);
    assert!(
        harness
            .export_checkpoint()
            .await
            .expect("checkpoint export is infallible")
            .is_none(),
        "an unstarted harness has no checkpoint"
    );

    harness
        .stop()
        .await
        .expect("stopping an unstarted harness is a no-op");
    assert_eq!(harness.state(), HarnessState::Terminated);

    let error = harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect_err("an unstarted harness accepts no input");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "cannot send to a stopped native harness");
}

#[test]
fn provider_create_rejects_unknown_and_mistyped_keys() {
    let provider = NativeHarnessProvider::new();
    let cases = [
        (
            serde_json::json!({"zeta": 1, "alpha": true}),
            "unknown native harness configuration fields: alpha, zeta",
        ),
        (
            serde_json::json!({"model": " "}),
            "native harness string options must be non-empty strings",
        ),
        (
            serde_json::json!({"max_iterations": 0}),
            "max_iterations must be a positive integer",
        ),
        (
            serde_json::json!({"temperature": 3.0}),
            "temperature must be a number between 0 and 2",
        ),
    ];
    for (config, expected) in cases {
        let mapping: JsonObject =
            serde_json::from_value(config.clone()).expect("the case is a JSON object");
        // `Box<dyn HarnessProtocol>` is not `Debug`, so `expect_err` cannot be
        // used on the success arm here.
        let Err(error) = provider.create(mapping) else {
            panic!("{config} must be rejected")
        };
        assert_eq!(error.kind(), HarnessErrorKind::Validation, "for {config}");
        assert_eq!(error.message(), expected, "for {config}");
    }
}

// ---------------------------------------------------------------------------
// Capability card
// ---------------------------------------------------------------------------

#[test]
fn native_card_declares_the_ten_documented_capabilities() {
    let (fixture, _model) = harness_with(ScriptedModel::text("unused"));
    let card = fixture.harness.card();
    assert_eq!(card.name, HARNESS_NAME);
    assert_eq!(card.implementation_version, PROVIDER_VERSION);
    assert_eq!(
        card.protocol_version,
        openjiuwen_harness_protocol::PROTOCOL_VERSION
    );

    let expected = native_card(HARNESS_NAME, PROVIDER_VERSION);
    assert_eq!(card.capabilities, expected.capabilities);
    assert_eq!(card.capabilities.len(), 10);
    for capability in [
        HarnessCapability::Steer,
        HarnessCapability::GracefulAbort,
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::PersistentSession,
        HarnessCapability::Checkpoint,
        HarnessCapability::NativeTools,
        HarnessCapability::McpTools,
        HarnessCapability::ModelSelection,
        HarnessCapability::ModelDiscovery,
    ] {
        assert!(
            card.supports(capability),
            "the card must declare {capability:?}"
        );
    }
    // Explicitly required by the protocol contract: a host may export
    // checkpoints, pause/resume, and call native tools against this harness.
    assert!(card.supports(HarnessCapability::Checkpoint));
    assert!(card.supports(HarnessCapability::PauseResume));
    assert!(card.supports(HarnessCapability::NativeTools));
    // The native provider has neither an endpoint model source nor a fallback.
    assert!(!card.supports(HarnessCapability::EndpointModels));
    assert!(!card.supports(HarnessCapability::AuthFallback));

    assert!(card.required_host_capabilities.is_empty());
    assert!(card.fork_modes.is_empty());
    assert_eq!(
        card.optional_host_capabilities.len(),
        4,
        "checkpoint sink, tool approval, user input, and telemetry are optional"
    );
}
