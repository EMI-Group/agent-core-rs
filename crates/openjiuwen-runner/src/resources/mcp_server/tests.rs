//! Unit tests for the MCP-server manager.

use super::*;
use async_trait::async_trait;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::tool::mcp::{McpRemoteTool, McpToolResult};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex as StdMutex;
use std::time::Duration;

/// A deterministic `McpClient`; failures are configured through flags.
struct FakeClient {
    server_id: String,
    server_name: String,
    tools: Vec<McpRemoteTool>,
    connects: AtomicUsize,
    disconnects: AtomicUsize,
    connected: AtomicBool,
    connect_error: StdMutex<Option<String>>,
    disconnect_error: StdMutex<Option<String>>,
}

impl FakeClient {
    fn new(server_id: &str, server_name: &str, tools: Vec<McpRemoteTool>) -> Arc<Self> {
        Arc::new(Self {
            server_id: server_id.to_owned(),
            server_name: server_name.to_owned(),
            tools,
            connects: AtomicUsize::new(0),
            disconnects: AtomicUsize::new(0),
            connected: AtomicBool::new(false),
            connect_error: StdMutex::new(None),
            disconnect_error: StdMutex::new(None),
        })
    }

    fn fail_connect(&self, reason: &str) {
        *self.connect_error.lock().unwrap() = Some(reason.to_owned());
    }

    fn fail_disconnect(&self, reason: &str) {
        *self.disconnect_error.lock().unwrap() = Some(reason.to_owned());
    }

    fn connect_count(&self) -> usize {
        self.connects.load(Ordering::SeqCst)
    }

    fn disconnect_count(&self) -> usize {
        self.disconnects.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl McpClient for FakeClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn server_id(&self) -> &str {
        &self.server_id
    }

    async fn connect(&self) -> Result<()> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        // Yield so two concurrent callers really interleave in the manager.
        tokio::task::yield_now().await;
        if let Some(reason) = self.connect_error.lock().unwrap().clone() {
            return Err(Error::Mcp(reason));
        }
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        self.disconnects.fetch_add(1, Ordering::SeqCst);
        if let Some(reason) = self.disconnect_error.lock().unwrap().clone() {
            return Err(Error::Mcp(reason));
        }
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
        Ok(self.tools.clone())
    }

    async fn call_tool_with(
        &self,
        _tool_name: &str,
        _arguments: Value,
        _timeout: Option<Duration>,
        _include_image_content: bool,
    ) -> Result<McpToolResult> {
        Err(Error::unsupported("the fake client does not call tools"))
    }
}

/// A connector returning a shared fake client (or a fixed failure).
struct FakeConnector {
    client: Arc<FakeClient>,
    builds: AtomicUsize,
    error: StdMutex<Option<String>>,
}

impl FakeConnector {
    fn new(client: Arc<FakeClient>) -> Arc<Self> {
        Arc::new(Self {
            client,
            builds: AtomicUsize::new(0),
            error: StdMutex::new(None),
        })
    }

    fn fail(&self, reason: &str) {
        *self.error.lock().unwrap() = Some(reason.to_owned());
    }

    fn builds(&self) -> usize {
        self.builds.load(Ordering::SeqCst)
    }
}

impl McpConnector for FakeConnector {
    fn build(&self, _config: &McpServerConfig) -> Result<Arc<dyn McpClient>> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        if let Some(reason) = self.error.lock().unwrap().clone() {
            return Err(Error::Mcp(reason));
        }
        Ok(Arc::clone(&self.client) as Arc<dyn McpClient>)
    }
}

/// One manager plus the fakes it talks to.
struct Fixture {
    mgr: McpServerMgr,
    client: Arc<FakeClient>,
    connector: Arc<FakeConnector>,
}

fn fixture() -> Fixture {
    let client = FakeClient::new(
        "s1",
        "alpha",
        vec![McpRemoteTool::new(
            "echo",
            "echoes its input",
            json!({"type": "object"}),
        )],
    );
    let connector = FakeConnector::new(Arc::clone(&client));
    let mgr = McpServerMgr::with_connector(Arc::clone(&connector) as Arc<dyn McpConnector>);
    Fixture {
        mgr,
        client,
        connector,
    }
}

fn config(server_id: &str, server_name: &str) -> McpServerConfig {
    McpServerConfig::stdio(server_name, "fake-mcp").with_server_id(server_id)
}

#[test]
fn states_round_trip_with_their_tagged_shape() {
    let cases = [
        (McpServerState::Registered, json!({"state": "registered"})),
        (McpServerState::Connected, json!({"state": "connected"})),
        (
            McpServerState::Failed {
                reason: "boom".to_owned(),
            },
            json!({"state": "failed", "reason": "boom"}),
        ),
        (
            McpServerState::Disconnected,
            json!({"state": "disconnected"}),
        ),
    ];
    for (state, expected) in cases {
        let value = serde_json::to_value(&state).unwrap();
        assert_eq!(value, expected);
        let back: McpServerState = serde_json::from_value(value).unwrap();
        assert_eq!(back, state);
    }
}

#[test]
fn server_info_round_trips() {
    let info = McpServerInfo {
        server_id: "s1".to_owned(),
        server_name: "alpha".to_owned(),
        state: McpServerState::Failed {
            reason: "nope".to_owned(),
        },
    };
    let value = serde_json::to_value(&info).unwrap();
    assert_eq!(
        value,
        json!({
            "server_id": "s1",
            "server_name": "alpha",
            "state": {"state": "failed", "reason": "nope"},
        })
    );
    let back: McpServerInfo = serde_json::from_value(value).unwrap();
    assert_eq!(back, info);
}

#[tokio::test]
async fn register_lists_servers_in_registration_order() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.mgr.register(config("s2", "beta")).unwrap();

    assert_eq!(f.mgr.ids(), vec!["s1", "s2"]);
    assert_eq!(f.mgr.len(), 2);
    assert!(!f.mgr.is_empty());
    assert_eq!(f.mgr.revision(), 2);
    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Registered));
    assert!(f.mgr.client("s1").is_none());
    assert_eq!(f.mgr.config("s2").unwrap().server_name, "beta");

    let infos = f.mgr.list();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].server_id, "s1");
    assert_eq!(infos[0].server_name, "alpha");
    assert_eq!(infos[0].state, McpServerState::Registered);
    assert_eq!(infos[1].server_id, "s2");
    assert_eq!(infos[1].server_name, "beta");
}

#[tokio::test]
async fn duplicate_registration_fails_under_fail_policy() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    let error = f.mgr.register(config("s1", "other")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: mcp server 's1' is already registered"
    );
    assert_eq!(f.mgr.config("s1").unwrap().server_name, "alpha");
    assert_eq!(f.mgr.revision(), 1);
}

#[tokio::test]
async fn keep_and_replace_policies_update_the_entry_as_documented() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    let kept = f
        .mgr
        .register_with(config("s1", "replacement"), RegisterPolicy::Keep)
        .unwrap();
    assert!(!kept.replaced);
    assert_eq!(kept.name, "alpha");
    assert_eq!(f.mgr.config("s1").unwrap().server_name, "alpha");
    assert_eq!(f.mgr.revision(), 1);

    f.mgr.connect("s1").await.unwrap();
    assert!(f.mgr.client("s1").is_some());

    let replaced = f
        .mgr
        .register_with(config("s1", "replacement"), RegisterPolicy::Replace)
        .unwrap();
    assert!(replaced.replaced);
    assert_eq!(replaced.name, "replacement");
    assert_eq!(f.mgr.config("s1").unwrap().server_name, "replacement");
    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Registered));
    assert!(f.mgr.client("s1").is_none());
    assert_eq!(f.mgr.revision(), 3);
}

#[tokio::test]
async fn empty_server_ids_are_rejected() {
    let f = fixture();
    for id in ["", "   ", "\t\n"] {
        let error = f.mgr.register(config(id, "ghost")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: mcp server id cannot be empty or whitespace only"
        );
    }
    assert!(f.mgr.is_empty());
    assert_eq!(f.mgr.revision(), 0);
}

#[tokio::test]
async fn connect_transitions_to_connected_and_exposes_the_client() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    f.mgr.connect("s1").await.unwrap();

    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
    assert!(f.mgr.client("s1").is_some());
    assert_eq!(f.client.connect_count(), 1);
    assert_eq!(f.connector.builds(), 1);
    assert_eq!(f.mgr.revision(), 2);
}

#[tokio::test]
async fn connect_is_idempotent() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    f.mgr.connect("s1").await.unwrap();
    f.mgr.connect("s1").await.unwrap();

    assert_eq!(f.client.connect_count(), 1);
    assert_eq!(f.connector.builds(), 1);
    assert_eq!(f.mgr.revision(), 2);
}

#[tokio::test]
async fn a_failing_connector_surfaces_the_error_and_marks_the_server_failed() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.connector.fail("cannot build");

    let error = f.mgr.connect("s1").await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Mcp);
    assert_eq!(error.to_string(), "mcp: cannot build");
    assert_eq!(
        f.mgr.state("s1"),
        Some(McpServerState::Failed {
            reason: "mcp: cannot build".to_owned()
        })
    );
    assert!(f.mgr.client("s1").is_none());
    assert_eq!(f.client.connect_count(), 0);
}

#[tokio::test]
async fn a_failing_connect_surfaces_the_error_and_marks_the_server_failed() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.client.fail_connect("transport down");

    let error = f.mgr.connect("s1").await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Mcp);
    assert_eq!(error.to_string(), "mcp: transport down");
    assert_eq!(
        f.mgr.state("s1"),
        Some(McpServerState::Failed {
            reason: "mcp: transport down".to_owned()
        })
    );
    assert!(f.mgr.client("s1").is_none());
    assert_eq!(f.client.connect_count(), 1);
}

#[tokio::test]
async fn a_failed_server_can_be_connected_after_the_cause_is_cleared() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.client.fail_connect("transport down");
    assert!(f.mgr.connect("s1").await.is_err());

    *f.client.connect_error.lock().unwrap() = None;
    f.mgr.connect("s1").await.unwrap();

    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
    assert!(f.mgr.client("s1").is_some());
    assert_eq!(f.client.connect_count(), 2);
}

#[tokio::test]
async fn load_tools_connects_implicitly() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    let tools = f.mgr.load_tools("s1").await.unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].card().id, "s1.alpha.echo");
    assert_eq!(tools[0].card().name, "echo");
    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
    assert_eq!(f.client.connect_count(), 1);
}

#[tokio::test]
async fn register_tools_publishes_the_cards_into_the_resource_mgr() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    let resources = crate::ResourceMgr::new();

    let ids = f.mgr.register_tools(&resources, "s1").await.unwrap();

    assert_eq!(ids, vec!["s1.alpha.echo".to_owned()]);
    assert_eq!(resources.tool_count(), 1);
    assert_eq!(resources.tool_card("s1.alpha.echo").unwrap().name, "echo");
    assert_eq!(
        resources.tool("s1.alpha.echo").unwrap().card().id,
        "s1.alpha.echo"
    );

    // Keep semantics: republishing is a no-op rather than a duplicate error.
    let again = f.mgr.register_tools(&resources, "s1").await.unwrap();
    assert_eq!(again, vec!["s1.alpha.echo".to_owned()]);
    assert_eq!(resources.tool_count(), 1);
}

#[tokio::test]
async fn disconnect_drops_the_client_and_marks_the_server_disconnected() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.mgr.connect("s1").await.unwrap();

    f.mgr.disconnect("s1").await.unwrap();

    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Disconnected));
    assert!(f.mgr.client("s1").is_none());
    assert_eq!(f.client.disconnect_count(), 1);

    // Already disconnected: no client, still Ok.
    f.mgr.disconnect("s1").await.unwrap();
    assert_eq!(f.client.disconnect_count(), 1);
}

#[tokio::test]
async fn disconnect_on_a_never_connected_server_is_ok() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    f.mgr.disconnect("s1").await.unwrap();

    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Disconnected));
    assert_eq!(f.client.disconnect_count(), 0);
}

#[tokio::test]
async fn a_failing_disconnect_still_marks_the_server_disconnected() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.mgr.connect("s1").await.unwrap();
    f.client.fail_disconnect("socket closed");

    let error = f.mgr.disconnect("s1").await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Mcp);
    assert_eq!(error.to_string(), "mcp: socket closed");
    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Disconnected));
    assert!(f.mgr.client("s1").is_none());
}

#[tokio::test]
async fn remove_disconnects_and_unregisters() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.mgr.connect("s1").await.unwrap();

    let config = f.mgr.remove("s1").await.unwrap();

    assert_eq!(config.server_id, "s1");
    assert_eq!(config.server_name, "alpha");
    assert!(f.mgr.is_empty());
    assert!(f.mgr.config("s1").is_none());
    assert_eq!(f.client.disconnect_count(), 1);
}

#[tokio::test]
async fn remove_still_removes_when_the_disconnect_fails() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.mgr.connect("s1").await.unwrap();
    f.client.fail_disconnect("socket closed");

    let config = f.mgr.remove("s1").await.unwrap();

    assert_eq!(config.server_name, "alpha");
    assert!(f.mgr.is_empty());
}

#[tokio::test]
async fn missing_ids_are_not_found() {
    let f = fixture();

    for error in [
        f.mgr.connect("missing").await.unwrap_err(),
        f.mgr.load_tools("missing").await.unwrap_err(),
        f.mgr.disconnect("missing").await.unwrap_err(),
        f.mgr.remove("missing").await.unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: mcp server 'missing' is not registered"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_connects_open_exactly_one_session() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();

    let (first, second) = tokio::join!(f.mgr.connect("s1"), f.mgr.connect("s1"));
    first.unwrap();
    second.unwrap();

    assert_eq!(f.client.connect_count(), 1);
    assert_eq!(f.connector.builds(), 1);
    assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
    assert!(f.mgr.client("s1").is_some());
}

#[tokio::test]
async fn revision_tracks_mutations_and_clear_reports_the_count() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    f.mgr.register(config("s2", "beta")).unwrap();
    assert_eq!(f.mgr.revision(), 2);

    assert_eq!(f.mgr.clear(), 2);
    assert!(f.mgr.is_empty());
    assert_eq!(f.mgr.revision(), 3);

    assert_eq!(f.mgr.clear(), 0);
    assert_eq!(f.mgr.revision(), 3, "clearing an empty registry is a no-op");
}

#[tokio::test]
async fn reset_clears_and_zeroes_the_revision() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    assert_eq!(f.mgr.reset(), 1);
    assert!(f.mgr.is_empty());
    assert_eq!(f.mgr.revision(), 0);
}

#[tokio::test]
async fn debug_output_lists_ids_and_the_revision() {
    let f = fixture();
    f.mgr.register(config("s1", "alpha")).unwrap();
    let rendered = format!("{:?}", f.mgr);
    assert!(rendered.contains("\"s1\""), "{rendered}");
    assert!(rendered.contains("revision: 1"), "{rendered}");
}
