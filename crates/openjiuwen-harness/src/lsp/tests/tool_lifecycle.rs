//! `LspTool` lifecycle tests: per-extension discovery, crash retry through the tool, budget
//! exhaustion and explicit shutdown.
//!
//! Every server is an in-process fake behind a [`ChannelTransport`]; the retry tests use a
//! scripted server that drops its stream mid-request and the deterministic [`InstantTimer`], so
//! nothing waits on wall-clock time.
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::io::{AsyncWriteExt, DuplexStream};

use openjiuwen_foundation::tool::{Tool, ToolContext};

use crate::lsp::config::LspServerConfig;
use crate::lsp::instance::{InstantTimer, RestartPolicy, ServerFactory, ServerState};
use crate::lsp::protocol::{write_message, MessageReader};
use crate::lsp::tool::{LspTool, LspToolConfig};
use crate::lsp::transport::{ChannelTransport, LspTransport};

use super::fake_server;

/// Invokes the tool once and returns the raw JSON payload.
async fn call(tool: &LspTool, args: Value) -> Value {
    tool.invoke(args, &ToolContext::new()).await.unwrap()
}

/// Writes `content` to `workspace/src/<name>` and returns the relative path.
async fn write_file(workspace: &Path, name: &str, content: &str) -> String {
    let path = workspace.join("src");
    tokio::fs::create_dir_all(&path).await.unwrap();
    tokio::fs::write(path.join(name), content).await.unwrap();
    format!("src/{name}")
}

/// Hovers line 1, column 1 of `path`.
async fn hover(tool: &LspTool, path: &str) -> Value {
    call(
        tool,
        json!({"action": "hover", "path": path, "line": 1, "column": 1}),
    )
    .await
}

/// The configuration a discovery test registers for Rust files.
fn rust_server() -> LspServerConfig {
    LspServerConfig::new("rust-analyzer")
        .with_name("rust")
        .with_language_id("rust")
        .with_extensions([".rs"])
        .with_priority(10)
}

/// The configuration a discovery test registers for Python files.
fn python_server() -> LspServerConfig {
    LspServerConfig::new("pyright-langserver")
        .with_name("pyright")
        .with_args(["--stdio"])
        .with_language_id("python")
        .with_extensions([".py"])
        .with_priority(10)
}

/// A read-only copy of a fake server's message log, captured after the test spawned it.
struct FakeServerSnapshot {
    /// Every recorded message.
    messages: Vec<Value>,
}

impl FakeServerSnapshot {
    /// Wraps a recorded message list.
    fn new(messages: Vec<Value>) -> Self {
        Self { messages }
    }

    /// The `method` of every message, in order.
    fn methods(&self) -> Vec<String> {
        self.messages
            .iter()
            .filter_map(|message| message.get("method").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .collect()
    }

    /// The number of messages with `method`.
    fn count(&self, method: &str) -> usize {
        self.methods()
            .iter()
            .filter(|candidate| candidate.as_str() == method)
            .count()
    }

    /// The `textDocument.languageId` of the first `didOpen`.
    fn opened_language(&self) -> Option<String> {
        self.messages
            .iter()
            .find(|message| {
                message.get("method").and_then(Value::as_str) == Some("textDocument/didOpen")
            })
            .and_then(|message| message.get("params"))
            .and_then(|params| params.get("textDocument"))
            .and_then(|document| document.get("languageId"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    }
}

#[tokio::test]
async fn discovery_routes_each_file_to_its_own_server_and_reuses_it() {
    let dir = tempfile::tempdir().unwrap();
    let rust_file = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let python_file = write_file(dir.path(), "app.py", "print('hi')\n").await;
    let configs = Arc::new(Mutex::new(Vec::new()));
    let logs = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&configs);
    let capture = Arc::clone(&logs);
    let factory: ServerFactory = Arc::new(move |config: &LspServerConfig| {
        record.lock().push(config.clone());
        let (transport, peer) = ChannelTransport::new(64 * 1024);
        capture.lock().push(fake_server::spawn(peer));
        Ok(Box::new(transport) as Box<dyn LspTransport>)
    });
    let tool = LspTool::new(
        LspToolConfig::default()
            .with_language("en")
            .with_workspace_root(dir.path())
            .with_servers([rust_server(), python_server()])
            .with_request_timeout(Duration::from_secs(2)),
    )
    .with_transport_factory(factory);

    assert_eq!(hover(&tool, &rust_file).await["success"], json!(true));
    assert_eq!(hover(&tool, &python_file).await["success"], json!(true));
    assert_eq!(
        configs.lock().len(),
        2,
        "one server per discovered language"
    );
    let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
    let statuses = tool.server_status();
    assert_eq!(statuses.len(), 2);
    for status in &statuses {
        assert_eq!(status.key.root, root);
        assert_eq!(status.state, ServerState::Running);
        assert!(status.healthy);
        assert_eq!(status.attempts, 0);
    }

    // A second call on the same file reuses the same key and never re-initializes.
    assert_eq!(hover(&tool, &rust_file).await["success"], json!(true));
    assert_eq!(configs.lock().len(), 2, "the supervisor is reused");
    let initializes = {
        let started = logs.lock();
        FakeServerSnapshot::new(started[0].messages()).count("initialize")
    };
    assert_eq!(initializes, 1);
    tool.shutdown().await;
}

#[tokio::test]
async fn discovery_sends_the_matching_language_id_on_did_open() {
    let dir = tempfile::tempdir().unwrap();
    let rust_file = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let python_file = write_file(dir.path(), "app.py", "print('hi')\n").await;
    let configs = Arc::new(Mutex::new(Vec::new()));
    let logs = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&configs);
    let capture = Arc::clone(&logs);
    let factory: ServerFactory = Arc::new(move |config: &LspServerConfig| {
        record.lock().push(config.clone());
        let (transport, peer) = ChannelTransport::new(64 * 1024);
        capture.lock().push(fake_server::spawn(peer));
        Ok(Box::new(transport) as Box<dyn LspTransport>)
    });
    let tool = LspTool::new(
        LspToolConfig::default()
            .with_language("en")
            .with_workspace_root(dir.path())
            .with_servers([rust_server(), python_server()])
            .with_request_timeout(Duration::from_secs(2)),
    )
    .with_transport_factory(factory);

    assert_eq!(hover(&tool, &rust_file).await["success"], json!(true));
    assert_eq!(hover(&tool, &python_file).await["success"], json!(true));
    assert_eq!(hover(&tool, &rust_file).await["success"], json!(true));

    let recorded = configs.lock().clone();
    let names: Vec<&str> = recorded.iter().map(|config| config.name.as_str()).collect();
    assert_eq!(names, ["rust", "pyright"]);

    // Each fake saw exactly one `initialize` and one `didOpen`, with its own language id.
    let started = logs.lock();
    let rust = FakeServerSnapshot::new(started[0].messages());
    let python = FakeServerSnapshot::new(started[1].messages());
    assert_eq!(rust.count("initialize"), 1);
    assert_eq!(python.count("initialize"), 1);
    assert_eq!(rust.opened_language().as_deref(), Some("rust"));
    assert_eq!(python.opened_language().as_deref(), Some("python"));
    assert_eq!(rust.count("textDocument/didOpen"), 1);
    assert_eq!(python.count("textDocument/didOpen"), 1);
    assert_eq!(
        rust.count("textDocument/didChange"),
        1,
        "the repeated hover re-synchronizes instead of re-opening"
    );
}

#[tokio::test]
async fn server_status_lists_every_started_key() {
    let dir = tempfile::tempdir().unwrap();
    let rust_file = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let _server = fake_server::spawn(peer);
    let tool = LspTool::new(
        LspToolConfig::default()
            .with_language("en")
            .with_workspace_root(dir.path())
            .with_language_id("rust")
            .with_request_timeout(Duration::from_secs(2)),
    )
    .with_transport(Box::new(transport));
    assert!(tool.server_status().is_empty(), "nothing starts lazily");
    assert_eq!(hover(&tool, &rust_file).await["success"], json!(true));
    let statuses = tool.server_status();
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0].key.name, "rust");
    assert_eq!(statuses[0].language_id, "rust");
    assert_eq!(statuses[0].state, ServerState::Running);
    assert_eq!(tool.diagnostics().pending_count(), 1);
    tool.shutdown().await;
    assert_eq!(tool.server_status()[0].state, ServerState::Stopped);
}

/// How the scripted server answers `textDocument/hover`.
#[derive(Clone, Copy)]
enum HoverMode {
    /// Drop the stream without answering, simulating a crash mid-request.
    Crash,
    /// Answer with a hover result.
    Answer,
}

/// Reads framed requests from `stream` until the peer goes away.
async fn scripted(stream: DuplexStream, mode: HoverMode) {
    let (read, mut write) = tokio::io::split(stream);
    let mut reader = MessageReader::new(read);
    loop {
        let message = match reader.next().await {
            Ok(Some(message)) => message,
            Ok(None) | Err(_) => break,
        };
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let Some(id) = message.get("id").cloned() else {
            if method == "exit" {
                let _ = write.shutdown().await;
                break;
            }
            continue;
        };
        let response = match method.as_str() {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "capabilities": { "hoverProvider": true } },
            }),
            "textDocument/hover" => match mode {
                HoverMode::Crash => break,
                HoverMode::Answer => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "contents": { "kind": "markdown", "value": "second server" } },
                }),
            },
            _ => json!({ "jsonrpc": "2.0", "id": id, "result": null }),
        };
        if write_message(&mut write, &response).await.is_err() {
            break;
        }
    }
}

/// Builds a factory that counts starts and scripts each server by its index.
fn scripted_factory(starts: Arc<AtomicUsize>, mode_for: fn(usize) -> HoverMode) -> ServerFactory {
    Arc::new(move |_config: &LspServerConfig| {
        let index = starts.fetch_add(1, Ordering::SeqCst);
        let (transport, peer) = ChannelTransport::new(64 * 1024);
        tokio::spawn(scripted(peer, mode_for(index)));
        Ok(Box::new(transport) as Box<dyn LspTransport>)
    })
}

/// The restart policy used by the retry tests: fast, bounded and deterministic.
fn policy(max_attempts: u32) -> RestartPolicy {
    RestartPolicy {
        max_attempts,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(4),
    }
}

/// Builds a tool with no explicit server, so every started server uses the injected factory.
fn retry_tool(
    workspace: &Path,
    factory: ServerFactory,
    timer: Arc<InstantTimer>,
    restart_policy: RestartPolicy,
) -> LspTool {
    LspTool::new(
        LspToolConfig::default()
            .with_language("en")
            .with_workspace_root(workspace)
            .with_language_id("rust")
            .with_request_timeout(Duration::from_secs(2))
            .with_restart_policy(restart_policy),
    )
    .with_transport_factory(factory)
    .with_backoff_timer(timer)
}

#[tokio::test]
async fn a_crashed_server_is_restarted_and_the_tool_call_retried() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let starts = Arc::new(AtomicUsize::new(0));
    let timer = Arc::new(InstantTimer::new());
    let tool = retry_tool(
        dir.path(),
        scripted_factory(Arc::clone(&starts), |index| {
            if index == 0 {
                HoverMode::Crash
            } else {
                HoverMode::Answer
            }
        }),
        Arc::clone(&timer),
        policy(2),
    );

    let payload = hover(&tool, &path).await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert!(payload["data"]["result"]
        .as_str()
        .unwrap()
        .contains("second server"));
    assert_eq!(starts.load(Ordering::SeqCst), 2, "one restart");
    assert_eq!(
        timer.delays(),
        vec![Duration::from_millis(1)],
        "the first back-off is the policy base delay"
    );
    let statuses = tool.server_status();
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0].state, ServerState::Running);
    assert_eq!(
        statuses[0].attempts, 0,
        "a successful restart resets the budget"
    );
    tool.shutdown().await;
}

#[tokio::test]
async fn an_exhausted_recovery_budget_is_a_payload_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let starts = Arc::new(AtomicUsize::new(0));
    let tool = retry_tool(
        dir.path(),
        scripted_factory(Arc::clone(&starts), |_| HoverMode::Crash),
        Arc::new(InstantTimer::new()),
        policy(1),
    );
    let payload = hover(&tool, &path).await;
    assert_eq!(payload["success"], json!(false), "{payload}");
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .contains("exceeded max crash recovery attempts"),
        "{payload}"
    );
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    tool.shutdown().await;
}

#[tokio::test]
async fn shutdown_fails_later_calls_without_starting_another_server() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let starts = Arc::new(AtomicUsize::new(0));
    let tool = retry_tool(
        dir.path(),
        scripted_factory(Arc::clone(&starts), |_| HoverMode::Answer),
        Arc::new(InstantTimer::new()),
        RestartPolicy::default(),
    );
    assert_eq!(hover(&tool, &path).await["success"], json!(true));
    assert_eq!(starts.load(Ordering::SeqCst), 1);

    tool.shutdown().await;
    let payload = hover(&tool, &path).await;
    assert_eq!(payload["success"], json!(false), "{payload}");
    assert!(
        payload["error"].as_str().unwrap().contains("shut down"),
        "{payload}"
    );
    assert_eq!(
        starts.load(Ordering::SeqCst),
        1,
        "a shut-down tool never starts another server"
    );
    tool.shutdown().await;
}
