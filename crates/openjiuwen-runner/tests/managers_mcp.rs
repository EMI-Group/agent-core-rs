#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration suite for the runner's MCP-server manager.
//!
//! The transports are local fakes built through `McpConnector`: registration,
//! connection state, tool loading and publication into the tool registry,
//! failure surfacing, disconnect/remove teardown, the per-server async lock, and
//! the serde shape of the state projection. One case proves the default
//! connector surfaces an unreachable binary as a typed error.
//!
//! Nothing here touches the network or the filesystem: the fake connector builds
//! an in-process `McpClient`, and the unreachable case fails while spawning a
//! child process for a path that does not exist.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::tool::mcp::{McpClient, McpRemoteTool, McpServerConfig, McpToolResult};
use openjiuwen_foundation::tool::Tool;
use openjiuwen_runner::resources::{
    McpConnector, McpServerInfo, McpServerMgr, McpServerState, ResourceMgr,
};
use serde_json::{json, Value};

/// Builds owned ids from `values`.
fn ids(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

// ------------------------------------------------------------ MCP servers --

/// A deterministic `McpClient` whose failures and hangs are flag-driven.
struct FakeMcpClient {
    server_id: String,
    server_name: String,
    tools: Vec<McpRemoteTool>,
    connects: AtomicUsize,
    disconnects: AtomicUsize,
    connected: AtomicBool,
    fail_connect: AtomicBool,
    fail_disconnect: AtomicBool,
    hang_connect: AtomicBool,
}

/// A fake client advertising one `echo` tool under `server_id`/`server_name`.
fn fake_mcp_client(server_id: &str, server_name: &str) -> Arc<FakeMcpClient> {
    Arc::new(FakeMcpClient {
        server_id: server_id.to_owned(),
        server_name: server_name.to_owned(),
        tools: vec![McpRemoteTool::new(
            "echo",
            "echoes its input",
            json!({"type": "object"}),
        )],
        connects: AtomicUsize::new(0),
        disconnects: AtomicUsize::new(0),
        connected: AtomicBool::new(false),
        fail_connect: AtomicBool::new(false),
        fail_disconnect: AtomicBool::new(false),
        hang_connect: AtomicBool::new(false),
    })
}

impl FakeMcpClient {
    /// Number of `connect` calls the manager made.
    fn connect_count(&self) -> usize {
        self.connects.load(Ordering::SeqCst)
    }

    /// Number of `disconnect` calls the manager made.
    fn disconnect_count(&self) -> usize {
        self.disconnects.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl McpClient for FakeMcpClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn server_id(&self) -> &str {
        &self.server_id
    }

    async fn connect(&self) -> Result<()> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        if self.hang_connect.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
        if self.fail_connect.load(Ordering::SeqCst) {
            return Err(Error::transport("fake transport is down"));
        }
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        self.disconnects.fetch_add(1, Ordering::SeqCst);
        if self.fail_disconnect.load(Ordering::SeqCst) {
            return Err(Error::transport("fake socket closed"));
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

/// A connector returning one shared fake client, or a fixed build failure.
struct FakeConnector {
    client: Arc<FakeMcpClient>,
    builds: AtomicUsize,
    build_error: StdMutex<Option<String>>,
}

impl FakeConnector {
    /// Number of `build` calls the manager made.
    fn builds(&self) -> usize {
        self.builds.load(Ordering::SeqCst)
    }

    /// Makes every subsequent `build` fail with a validation error.
    fn fail(&self, reason: &str) {
        *self.build_error.lock().unwrap() = Some(reason.to_owned());
    }
}

impl McpConnector for FakeConnector {
    fn build(&self, _config: &McpServerConfig) -> Result<Arc<dyn McpClient>> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        if let Some(reason) = self.build_error.lock().unwrap().clone() {
            return Err(Error::validation(reason));
        }
        Ok(Arc::clone(&self.client) as Arc<dyn McpClient>)
    }
}

/// One manager plus the fakes it talks to.
struct McpFixture {
    mgr: McpServerMgr,
    client: Arc<FakeMcpClient>,
    connector: Arc<FakeConnector>,
}

/// Builds a standalone MCP manager wired to a fake client and connector.
fn mcp_fixture() -> McpFixture {
    let client = fake_mcp_client("a", "alpha");
    let connector = Arc::new(FakeConnector {
        client: Arc::clone(&client),
        builds: AtomicUsize::new(0),
        build_error: StdMutex::new(None),
    });
    let mgr = McpServerMgr::with_connector(Arc::clone(&connector) as Arc<dyn McpConnector>);
    McpFixture {
        mgr,
        client,
        connector,
    }
}

/// A stdio server configuration with an explicit id.
fn mcp_config(server_id: &str, server_name: &str) -> McpServerConfig {
    McpServerConfig::stdio(server_name, "fake-mcp").with_server_id(server_id)
}

#[test]
fn mcp_servers_register_and_list_without_connecting() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();
    fixture.mgr.register(mcp_config("b", "beta")).unwrap();

    assert_eq!(fixture.mgr.ids(), ids(&["a", "b"]));
    assert_eq!(fixture.mgr.len(), 2);
    assert!(!fixture.mgr.is_empty());
    let listed: Vec<McpServerInfo> = fixture.mgr.list();
    assert_eq!(
        listed
            .iter()
            .map(|info| info.server_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"],
        "registration order is preserved"
    );
    assert!(listed
        .iter()
        .all(|info| info.state == McpServerState::Registered));
    assert_eq!(fixture.mgr.revision(), 2);
    assert!(
        fixture.mgr.client("a").is_none(),
        "registration never connects"
    );
    assert_eq!(
        fixture.connector.builds(),
        0,
        "no transport is built until connect"
    );

    let error = fixture.mgr.register(mcp_config("a", "other")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: mcp server 'a' is already registered"
    );
    assert_eq!(fixture.mgr.config("a").unwrap().server_name, "alpha");
    assert_eq!(fixture.mgr.revision(), 2, "the rejection did not mutate");
}

#[tokio::test]
async fn connecting_loads_and_publishes_the_fake_tools() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();

    fixture.mgr.connect("a").await.unwrap();
    assert_eq!(fixture.mgr.state("a"), Some(McpServerState::Connected));
    assert!(fixture.mgr.client("a").is_some());
    assert_eq!(fixture.client.connect_count(), 1);
    assert_eq!(fixture.connector.builds(), 1);

    let tools = fixture.mgr.load_tools("a").await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].card().id, "a.alpha.echo");

    let resources = ResourceMgr::new();
    let published = fixture.mgr.register_tools(&resources, "a").await.unwrap();
    assert_eq!(published, ids(&["a.alpha.echo"]));
    assert!(resources.tool_ids().contains(&"a.alpha.echo".to_owned()));
    assert_eq!(resources.tool_card("a.alpha.echo").unwrap().name, "echo");
    assert_eq!(
        resources.tool("a.alpha.echo").unwrap().card().id,
        "a.alpha.echo"
    );

    // Keep semantics: republishing neither duplicates nor fails.
    let again = fixture.mgr.register_tools(&resources, "a").await.unwrap();
    assert_eq!(again, published);
    assert_eq!(resources.tool_count(), 1);
}

#[tokio::test]
async fn a_failing_connector_marks_the_server_failed() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();
    fixture.connector.fail("cannot build");

    let error = fixture.mgr.connect("a").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: cannot build");

    let state = fixture.mgr.state("a").expect("registered");
    assert!(
        matches!(&state, McpServerState::Failed { reason } if reason == "validation: cannot build"),
        "{state:?}"
    );
    assert!(fixture.mgr.client("a").is_none());
    assert_eq!(fixture.client.connect_count(), 0, "nothing was connected");
}

#[tokio::test]
async fn a_failing_client_connect_surfaces_the_transport_failure() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();
    fixture.client.fail_connect.store(true, Ordering::SeqCst);

    let error = fixture.mgr.connect("a").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(error.to_string(), "transport: fake transport is down");

    let state = fixture.mgr.state("a").expect("registered");
    assert!(
        matches!(
            &state,
            McpServerState::Failed { reason }
                if reason.starts_with("transport:") && reason.contains("fake transport is down")
        ),
        "the failure reason names the transport error: {state:?}"
    );
    assert!(fixture.mgr.client("a").is_none());
    assert_eq!(fixture.client.connect_count(), 1);
}

#[tokio::test]
async fn disconnect_and_remove_tear_the_session_down() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();
    fixture.mgr.connect("a").await.unwrap();

    fixture.mgr.disconnect("a").await.unwrap();
    assert_eq!(fixture.mgr.state("a"), Some(McpServerState::Disconnected));
    assert!(fixture.mgr.client("a").is_none());
    assert_eq!(fixture.client.disconnect_count(), 1);

    // A second disconnect has no client to close and stays Ok (documented).
    fixture.mgr.disconnect("a").await.unwrap();
    assert_eq!(fixture.client.disconnect_count(), 1);
    assert_eq!(fixture.mgr.state("a"), Some(McpServerState::Disconnected));

    fixture.mgr.connect("a").await.unwrap();
    assert_eq!(fixture.mgr.state("a"), Some(McpServerState::Connected));

    let removed = fixture.mgr.remove("a").await.unwrap();
    assert_eq!(removed.server_id, "a");
    assert_eq!(removed.server_name, "alpha");
    assert_eq!(fixture.mgr.list(), Vec::<McpServerInfo>::new());
    assert_eq!(fixture.mgr.ids(), Vec::<String>::new());
    assert_eq!(
        fixture.client.disconnect_count(),
        2,
        "remove disconnects the held session"
    );

    let error = fixture.mgr.remove("a").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: mcp server 'a' is not registered"
    );
}

#[tokio::test]
async fn states_and_infos_serialize_with_their_tag() {
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
        assert_eq!(serde_json::to_value(&state).unwrap(), expected);
        let restored: McpServerState = serde_json::from_value(expected).unwrap();
        assert_eq!(restored, state);
    }

    // The projection the manager lists carries the same tagged state.
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();
    fixture.mgr.connect("a").await.unwrap();
    let info = fixture.mgr.list().remove(0);
    assert_eq!(
        serde_json::to_value(&info).unwrap(),
        json!({
            "server_id": "a",
            "server_name": "alpha",
            "state": {"state": "connected"},
        })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_connects_open_exactly_one_session() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();

    let (first, second) = tokio::join!(fixture.mgr.connect("a"), fixture.mgr.connect("a"));
    first.unwrap();
    second.unwrap();

    assert_eq!(
        fixture.client.connect_count(),
        1,
        "the per-server lock serialises the two callers"
    );
    assert_eq!(fixture.connector.builds(), 1);
    assert_eq!(fixture.mgr.state("a"), Some(McpServerState::Connected));
    assert!(fixture.mgr.client("a").is_some());
}

#[tokio::test]
async fn a_hung_connect_times_out_and_leaves_the_server_unconnected() {
    let fixture = mcp_fixture();
    fixture.mgr.register(mcp_config("a", "alpha")).unwrap();
    fixture.client.hang_connect.store(true, Ordering::SeqCst);

    let outcome = tokio::time::timeout(Duration::from_secs(1), fixture.mgr.connect("a")).await;
    assert!(
        outcome.is_err(),
        "the outer timeout fires before the fake wakes"
    );

    // A cancelled connect never records a state change: the entry stays as it
    // was registered, with no client (`McpServerMgr::connect` records only on a
    // completed build/connect).
    assert_eq!(fixture.mgr.state("a"), Some(McpServerState::Registered));
    assert_ne!(fixture.mgr.state("a"), Some(McpServerState::Connected));
    assert!(fixture.mgr.client("a").is_none());
}

#[cfg(unix)]
#[tokio::test]
async fn the_default_connector_surfaces_an_unreachable_transport() {
    let manager = McpServerMgr::new();
    manager
        .register(
            McpServerConfig::stdio("dead", "/nonexistent-openjiuwen-mcp-binary")
                .with_server_id("dead"),
        )
        .unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(10), manager.connect("dead")).await;
    let error = outcome
        .expect("spawning a missing binary fails fast")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    // The stdio transport fails while spawning the child, before any session is
    // established, so the message is the spawn failure rather than the
    // `MCP connect failed:` wrapper `spawn_session` adds for a protocol error.
    assert!(error.to_string().contains("failed to spawn"), "{error}");

    let state = manager.state("dead").expect("registered");
    assert!(
        matches!(&state, McpServerState::Failed { reason } if reason.contains("failed to spawn")),
        "{state:?}"
    );
    assert!(manager.client("dead").is_none());
}
