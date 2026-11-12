//! Start/stop lifecycle of the native harness.

use super::*;

#[tokio::test]
async fn state_before_start_is_terminated_and_has_no_cycle() {
    let fixture = fixture();

    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
    assert!(!fixture.harness.started());
    assert_eq!(fixture.harness.provider_session_id(), None);
    assert!(fixture.harness.export_checkpoint().await.unwrap().is_none());
    assert_eq!(fixture.harness.card().name, HARNESS_NAME);

    let mut cursor = fixture.harness.events();
    let error = next_error(&mut cursor).await;
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "native harness has no active event cycle");
}

#[tokio::test]
async fn start_rejects_an_unsupported_protocol_version() {
    let fixture = fixture();

    let error = fixture
        .harness
        .start(context().with_protocol_version("2.0"))
        .await
        .expect_err("protocol 2.0 is not supported");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(
        error.message().contains("is not supported"),
        "unexpected message: {}",
        error.message()
    );
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
}

#[tokio::test]
async fn start_without_host_capabilities_settles_idle_and_emits_the_first_transition() {
    let fixture = fixture();
    let standard = context();
    assert_eq!(standard.host_capabilities.len(), 0);

    fixture.start_with(standard).await;

    assert_eq!(fixture.harness.state(), HarnessState::Idle);
    assert!(fixture.harness.started());
    assert_eq!(fixture.harness.provider_session_id(), Some("host-1"));

    let mut cursor = fixture.harness.events();
    let event = next_event(&mut cursor).await.expect("an event is buffered");
    assert_eq!(event.sequence, 1);
    assert_eq!(event.host_session_id, "host-1");
    assert_eq!(event.agent_id, "agent-1");
    assert_eq!(event.provider_session_id.as_deref(), Some("host-1"));
    let change = state_change(&event).expect("the first event is a state change");
    assert_eq!(change.old, HarnessState::Terminated);
    assert_eq!(change.new, HarnessState::Idle);
}

#[tokio::test]
async fn double_start_is_a_state_error() {
    let fixture = fixture();
    fixture.start().await;

    let error = fixture
        .harness
        .start(context())
        .await
        .expect_err("a started harness cannot be started again");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "native harness is already started");
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
}

#[tokio::test]
async fn start_rejects_context_mcp_servers() {
    let fixture = fixture();
    let server = McpServerConfig {
        name: "search".to_owned(),
        transport: McpTransport::Stdio,
        command: vec!["mcp-search".to_owned()],
        url: None,
        instance: None,
        env: std::collections::BTreeMap::new(),
        headers: std::collections::BTreeMap::new(),
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
}

#[tokio::test]
async fn stop_is_idempotent_and_closes_the_cycle() {
    let fixture = fixture();
    fixture.start().await;
    let mut cursor = fixture.harness.events();

    fixture.harness.stop().await.expect("stop succeeds");
    fixture.harness.stop().await.expect("stop is idempotent");

    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
    assert!(!fixture.harness.started());
    assert_eq!(fixture.harness.provider_session_id(), None);

    let mut terminated = false;
    while let Some(event) = next_event(&mut cursor).await {
        if let Some(change) = state_change(&event) {
            terminated |= change.new == HarnessState::Terminated;
        }
    }
    assert!(terminated, "the cursor missed the terminal transition");

    let error = fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect_err("a stopped harness accepts no input");
    assert_eq!(error.message(), "cannot send to a stopped native harness");

    let error = next_error(&mut fixture.harness.events()).await;
    assert_eq!(error.message(), "native harness has no active event cycle");
}
