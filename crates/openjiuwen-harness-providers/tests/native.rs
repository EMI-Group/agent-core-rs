#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Offline, deterministic end-to-end suite of the native provider.
//!
//! Group A exercises this crate's provider wiring (`providers::native`, the registry descriptor and
//! the `create` validation) and is compiled only with the `native` feature. Group B drives a real
//! `openjiuwen_harness::NativeHarness` over a kernel `DeepAgent` built on
//! [`openjiuwen_test_support::ScriptedModel`] and a temporary workspace, so it runs in every
//! feature set — it exercises the harness crate, not this crate's `native` feature.
//!
//! No network, no real provider, no `tokio::time::sleep`: every await that could hang is wrapped in
//! a five-second timeout purely as a safety net, and ordering that matters (a finished turn before
//! the harness settles idle) is observed through the protocol event stream instead of timing
//! assumptions.
//!
//! Python counterpart: `tests/system_tests/harness_providers/test_native_e2e.py`.

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::AgentCard;
use openjiuwen_foundation::llm::Model;
use openjiuwen_harness::{
    DeepAgentBuilder, DeepAgentConfig, NativeHarness, NativeHarnessOptions, Workspace,
};
use openjiuwen_harness_protocol::{
    DeliveryMode, HarnessContext, HarnessErrorKind, HarnessEvent, HarnessEventCursor,
    HarnessEventPayload, HarnessInput, HarnessModelControl, HarnessProtocol, HarnessState,
    ModelSelection, TurnEventKind,
};
use openjiuwen_test_support::ScriptedModel;
use tempfile::TempDir;

#[cfg(feature = "native")]
use std::collections::BTreeSet;

#[cfg(feature = "native")]
use openjiuwen_harness::protocol::PROVIDER_VERSION;
#[cfg(feature = "native")]
use openjiuwen_harness_protocol::{
    json_object_from_value, HarnessCapability, HarnessCard, HarnessProvider, HostCapability,
    JsonObject, PROTOCOL_VERSION,
};
#[cfg(feature = "native")]
use openjiuwen_harness_providers::{describe_provider, providers::native};
#[cfg(feature = "native")]
use serde_json::json;

/// Timeout applied to every await that must not hang the suite.
const WAIT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Group A — provider wiring (`native` feature)
// ---------------------------------------------------------------------------

/// Builds the JSON object a `create` mapping needs.
#[cfg(feature = "native")]
fn object(value: serde_json::Value) -> JsonObject {
    json_object_from_value(value).expect("the case is a JSON object")
}

/// Returns the wire names of the harness capabilities `card` declares.
#[cfg(feature = "native")]
fn capability_names(card: &HarnessCard) -> Vec<&str> {
    card.capabilities
        .iter()
        .map(|capability| capability.as_str())
        .collect()
}

/// Returns the wire names of the optional host capabilities `card` declares.
#[cfg(feature = "native")]
fn optional_host_names(card: &HarnessCard) -> Vec<&str> {
    card.optional_host_capabilities
        .iter()
        .map(|capability| capability.as_str())
        .collect()
}

/// Port of `test_native_e2e.py::test_native_provider_identity`.
#[cfg(feature = "native")]
#[test]
fn native_identity_matches_the_harness() {
    assert_eq!(native::PROVIDER_NAME, "native");
    assert_eq!(native::CARD_NAME, "openjiuwen-native");
    let card = native::card();
    assert_eq!(card.name, "openjiuwen-native");
    assert_eq!(card.implementation_version, PROVIDER_VERSION);
    assert_eq!(card.protocol_version, PROTOCOL_VERSION);
    assert_eq!(card.protocol_version, "1.0");
    assert_eq!(
        card.compatible_protocol_versions,
        BTreeSet::from(["1.0".to_owned()])
    );
}

/// Port of `test_native_e2e.py::test_native_card_capabilities`.
#[cfg(feature = "native")]
#[test]
fn native_card_declares_exactly_the_native_capabilities() {
    let card = native::card();
    assert_eq!(
        card.capabilities,
        BTreeSet::from([
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
        ])
    );
    assert_eq!(
        capability_names(&card),
        vec![
            "steer",
            "graceful_abort",
            "force_abort",
            "pause_resume",
            "persistent_session",
            "checkpoint",
            "native_tools",
            "mcp_tools",
            "model_selection",
            "model_discovery",
        ]
    );
    assert!(card.required_host_capabilities.is_empty());
    assert_eq!(
        card.optional_host_capabilities,
        BTreeSet::from([
            HostCapability::CheckpointSink,
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::Telemetry,
        ])
    );
    assert_eq!(
        optional_host_names(&card),
        vec![
            "checkpoint_sink",
            "tool_approval",
            "user_input",
            "telemetry"
        ]
    );
    assert!(card.fork_modes.is_empty());
}

/// Port of `test_native_e2e.py::test_native_provider_descriptor`.
#[cfg(feature = "native")]
#[test]
fn native_provider_reports_the_shared_card_and_descriptor() {
    assert_eq!(native::provider().card(), &native::card());
    let descriptor = describe_provider("native").expect("native is a declared provider");
    assert_eq!(descriptor.name, "native");
    assert_eq!(descriptor.card_name, "openjiuwen-native");
    assert_eq!(descriptor.implementation_version, PROVIDER_VERSION);
    assert!(descriptor.builtin);
    assert!(descriptor.implemented);
    assert!(descriptor.configures_models);
    assert_eq!(descriptor.requires, None);
}

/// Port of `test_native_e2e.py::test_native_create_rejects_invalid_configuration`.
#[cfg(feature = "native")]
#[test]
fn native_create_rejects_invalid_configuration() {
    let provider = native::provider();
    let cases = [
        (
            json!({"zeta": 1, "alpha": true}),
            "unknown native harness configuration fields: alpha, zeta",
        ),
        (
            json!({"model": " "}),
            "native harness string options must be non-empty strings",
        ),
        (
            json!({"temperature": 2.5}),
            "temperature must be a number between 0 and 2",
        ),
        (
            json!({"max_iterations": 0}),
            "max_iterations must be a positive integer",
        ),
    ];
    for (config, expected) in cases {
        let error = provider
            .create(object(config.clone()))
            .err()
            .unwrap_or_else(|| panic!("{config} must be rejected"));
        assert_eq!(error.kind(), HarnessErrorKind::Validation, "for {config}");
        assert_eq!(error.message(), expected, "for {config}");
    }
}

/// Port of `test_native_e2e.py::test_native_create_returns_an_unstarted_harness`.
#[cfg(feature = "native")]
#[test]
fn native_create_returns_an_unstarted_harness() {
    let harness = native::provider()
        .create(object(json!({"model": "scripted"})))
        .expect("a model name is a valid create mapping");
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(harness.card().name, "openjiuwen-native");
    assert_eq!(harness.event_buffer_config().capacity, 1024);
}

// ---------------------------------------------------------------------------
// Group B — end-to-end over a real `NativeHarness`
// ---------------------------------------------------------------------------

/// A harness over a scripted-model agent in a temporary workspace.
struct Fixture {
    /// Kept alive so the temporary workspace outlives the harness.
    _tmp: TempDir,
    /// Harness under test.
    harness: NativeHarness,
}

impl Fixture {
    /// Builds a harness whose agent answers every turn with `hello`.
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("temporary workspace");
        let agent = DeepAgentBuilder::new(AgentCard::new("deep_agent"))
            .config(DeepAgentConfig {
                model_name: Some("scripted-model".to_owned()),
                ..DeepAgentConfig::default()
            })
            .model(Arc::new(ScriptedModel::text("hello")) as Arc<dyn Model>)
            .workspace(Workspace::new(tmp.path()))
            .build()
            .expect("the deep agent builds");
        Self {
            _tmp: tmp,
            harness: NativeHarness::new(agent, NativeHarnessOptions::new()),
        }
    }

    /// Starts the harness with the suite context and asserts it succeeded.
    async fn start(&self) {
        tokio::time::timeout(WAIT, self.harness.start(context()))
            .await
            .expect("start must not hang")
            .expect("the harness starts");
    }
}

/// The standard host context of the suite.
fn context() -> HarnessContext {
    HarnessContext::new(
        "deep_agent",
        "deep_agent",
        "host-session-1",
        "You are a test agent.",
    )
}

/// Reads the next event of `cursor`, failing fast on timeout or cursor error.
async fn next_event(cursor: &mut Box<dyn HarnessEventCursor>) -> Option<HarnessEvent> {
    tokio::time::timeout(WAIT, cursor.next_event())
        .await
        .expect("timed out waiting for a protocol event")
        .expect("the observation cursor reported an error")
}

/// Returns whether `event` is the `kind` lifecycle event of `turn_id`.
fn is_turn_event(event: &HarnessEvent, turn_id: &str, kind: TurnEventKind) -> bool {
    event.turn_id.as_deref() == Some(turn_id)
        && matches!(
            &event.event,
            HarnessEventPayload::TurnLifecycle(payload) if payload.kind == kind
        )
}

/// Drains `cursor` until `turn_id` finished and the harness settled back in `Idle`.
async fn drain_finished_turn(cursor: &mut Box<dyn HarnessEventCursor>, turn_id: &str) {
    let mut finished = false;
    loop {
        let event = next_event(cursor)
            .await
            .expect("the observation cursor ended before the turn finished");
        finished |= is_turn_event(&event, turn_id, TurnEventKind::Finished);
        let settled = matches!(
            &event.event,
            HarnessEventPayload::StateChanged(change) if change.new == HarnessState::Idle
        );
        if finished && settled {
            return;
        }
    }
}

/// Port of `test_native_e2e.py::test_native_starts_and_stops`.
#[tokio::test]
async fn start_settles_idle_and_stop_is_idempotent() {
    let fixture = Fixture::new();
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
    fixture.start().await;
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
    assert_eq!(fixture.harness.event_buffer_config().capacity, 1024);
    tokio::time::timeout(WAIT, fixture.harness.stop())
        .await
        .expect("stop must not hang")
        .expect("stop succeeds");
    tokio::time::timeout(WAIT, fixture.harness.stop())
        .await
        .expect("the second stop must not hang")
        .expect("stop is idempotent");
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
}

/// Port of `test_native_e2e.py::test_native_turn_completes`.
#[tokio::test]
async fn a_turn_finishes_and_leaves_an_exportable_checkpoint() {
    let fixture = Fixture::new();
    fixture.start().await;
    let mut cursor = fixture.harness.events();
    let receipt = tokio::time::timeout(WAIT, fixture.harness.send_auto(HarnessInput::text("hi")))
        .await
        .expect("send must not hang")
        .expect("the input is accepted");
    assert!(!receipt.message_id.is_empty(), "a message id is returned");
    assert!(!receipt.turn_id.is_empty(), "a turn id is returned");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Auto);
    drain_finished_turn(&mut cursor, &receipt.turn_id).await;
    drop(cursor);
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
    tokio::time::timeout(WAIT, fixture.harness.export_checkpoint())
        .await
        .expect("export must not hang")
        .expect("exporting after a finished turn succeeds");
}

/// Port of `test_native_e2e.py::test_native_model_control`.
#[tokio::test]
async fn model_control_lists_and_selects_the_configured_model() {
    let fixture = Fixture::new();
    fixture.start().await;
    let models = tokio::time::timeout(WAIT, fixture.harness.list_models())
        .await
        .expect("model discovery must not hang")
        .expect("model discovery works");
    assert!(!models.is_empty(), "the catalog reports the model");
    let first = models.first().expect("the catalog is non-empty");
    tokio::time::timeout(
        WAIT,
        fixture.harness.set_model(ModelSelection {
            model: Some(first.model_id.clone()),
            effort: None,
        }),
    )
    .await
    .expect("model selection must not hang")
    .expect("the catalog model is selectable");
    let error = tokio::time::timeout(
        WAIT,
        fixture.harness.set_model(ModelSelection {
            model: None,
            effort: Some("extreme".to_owned()),
        }),
    )
    .await
    .expect("the rejected selection must not hang")
    .expect_err("unknown efforts are rejected");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(error.message(), "unsupported reasoning effort 'extreme'");
}

/// Port of `test_native_e2e.py::test_native_send_before_start_is_rejected`.
#[tokio::test]
async fn send_before_start_is_a_state_error() {
    let fixture = Fixture::new();
    let error = tokio::time::timeout(WAIT, fixture.harness.send_auto(HarnessInput::text("hi")))
        .await
        .expect("the rejected send must not hang")
        .expect_err("an unstarted harness accepts no input");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "cannot send to a stopped native harness");
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
    tokio::time::timeout(WAIT, fixture.harness.stop())
        .await
        .expect("stop must not hang")
        .expect("stopping an unstarted harness is a no-op");
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
    tokio::time::timeout(WAIT, fixture.harness.export_checkpoint())
        .await
        .expect("export must not hang")
        .expect("exporting an unstarted harness succeeds");
}
