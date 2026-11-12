//! `07_mcp_tool` — an MCP-exposed tool mounted on a `ReAct` agent.
//!
//! An MCP tool is two halves: a `ToolCard` (what the model sees, projected from
//! the server's `McpRemoteTool`) and an `McpClient` (the transport reaching the
//! server). Part A builds both by hand so the example runs offline:
//!
//! - `LocalMcpClient` is the in-process `McpClient` — the seam
//!   `connect_mcp_server` fills with a real stdio/HTTP transport. It advertises
//!   `echo` and `word_count` with real JSON schemas, and reports an unknown tool
//!   as a failed result rather than an error.
//! - `load_mcp_tools` projects them into `McpTool`s (schema field:
//!   `input_params`); `ReActAgent::add_tool` mounts them, so the scripted model's
//!   `echo` call is served by the MCP client and comes back as an observation.
//!
//! Part B is opt-in: `OPENJIUWEN_MCP_SERVER_CMD="npx -y
//! @modelcontextprotocol/server-everything"` connects the same recipe to a real
//! stdio server via `connect_mcp_server`; without it the example explains how to
//! enable that and still exits 0. Python counterpart:
//! `examples/mcp/stdio/client_as_tool.py`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use openjiuwen::core::common::error::{Error, Result as SdkResult};
use openjiuwen::core::foundation::async_trait::async_trait;
use openjiuwen::core::foundation::tool::mcp::{
    connect_mcp_server, load_mcp_tools, McpClient, McpRemoteTool, McpServerConfig, McpToolOptions,
    McpToolResult,
};
use openjiuwen::core::foundation::tool::Tool;
use openjiuwen::core::single_agent::{Agent, AgentCard, ReActAgent, ReActAgentConfig};
use serde_json::{json, Map, Value};

/// Model name written into the agent config; the scripted model ignores it.
const MODEL_NAME: &str = "gpt-4o-mini";
/// Provider written into the agent config.
const MODEL_PROVIDER: &str = "openai";
/// Server name and id of the offline stand-in (the two id segments of its tools).
const LOCAL_SERVER_NAME: &str = "local";
const LOCAL_SERVER_ID: &str = "example";
/// The two tools the offline stand-in advertises, and the text handed to `echo`.
const ECHO_TOOL: &str = "echo";
const WORD_COUNT_TOOL: &str = "word_count";
const PING: &str = "ping from an MCP client";
/// Upper bounds: one MCP operation, the whole agent run, and a real stdio server.
const CALL_TIMEOUT: Duration = Duration::from_secs(10);
const RUN_TIMEOUT: Duration = Duration::from_secs(30);
const LIVE_TIMEOUT: Duration = Duration::from_secs(15);
/// Environment variable naming the stdio server part B may start.
const MCP_SERVER_ENV: &str = "OPENJIUWEN_MCP_SERVER_CMD";

/// An in-process MCP client, standing in for a stdio server.
///
/// It implements the same `McpClient` trait `RmcpClient` implements: only the
/// transport is local, so the code below is what a real deployment runs.
struct LocalMcpClient {
    /// Configured server name.
    name: String,
    /// Whether `connect` succeeded and `disconnect` has not run yet.
    connected: AtomicBool,
}

impl LocalMcpClient {
    /// Creates a disconnected client named `name`.
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            connected: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl McpClient for LocalMcpClient {
    fn server_name(&self) -> &str {
        &self.name
    }

    fn server_id(&self) -> &str {
        LOCAL_SERVER_ID
    }

    async fn connect(&self) -> SdkResult<()> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> SdkResult<()> {
        if !self.connected.swap(false, Ordering::SeqCst) {
            return Err(Error::mcp(format!("`{}` is not connected", self.name)));
        }
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn list_tools(&self) -> SdkResult<Vec<McpRemoteTool>> {
        // Real servers send a full JSON Schema per tool; one string argument is
        // enough to show the projection here.
        let schema = |about: &str| json!({ "type": "object", "properties": { "text": { "type": "string", "description": about } }, "required": ["text"] });
        Ok(vec![
            McpRemoteTool::new(
                ECHO_TOOL,
                "Echo the given text back to the caller.",
                schema("Text to echo back."),
            ),
            McpRemoteTool::new(
                WORD_COUNT_TOOL,
                "Count the whitespace-separated words in a string.",
                schema("Text whose words are counted."),
            ),
        ])
    }

    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: Value,
        _timeout: Option<Duration>,
        _include_image_content: bool,
    ) -> SdkResult<McpToolResult> {
        if !matches!(tool_name, ECHO_TOOL | WORD_COUNT_TOOL) {
            return Ok(McpToolResult::failure(format!(
                "unknown tool `{tool_name}`; `{}` exposes `{ECHO_TOOL}` and `{WORD_COUNT_TOOL}`",
                self.name
            )));
        }
        let Some(text) = arguments.get("text").and_then(Value::as_str) else {
            return Ok(McpToolResult::failure(format!(
                "`{tool_name}` needs a string `text` argument"
            )));
        };
        let content = if tool_name == WORD_COUNT_TOOL {
            format!("{} words", text.split_whitespace().count())
        } else {
            text.to_owned()
        };
        // MCP servers answer with content blocks; a lone text block is normalized
        // into `data.content`, which is what `McpTool` renders for the model.
        let mut data = Map::new();
        data.insert("content".to_owned(), Value::String(content));
        Ok(McpToolResult::success(data))
    }
}

/// Disconnects `client`, reporting the outcome instead of failing the example.
async fn disconnect(client: &Arc<dyn McpClient>, label: &str) {
    match tokio::time::timeout(CALL_TIMEOUT, client.disconnect()).await {
        Ok(Ok(())) => common::banner::note(&format!("{label} client disconnected")),
        Ok(Err(error)) => common::banner::note(&format!("{label} disconnect reported: {error}")),
        Err(_) => common::banner::note(&format!("disconnecting the {label} client timed out")),
    }
}

/// Part A: the local client, its tools, and one agent round over them.
async fn offline_round() -> Result<()> {
    common::banner::step("part A: a local MCP client standing in for a stdio server");
    let client: Arc<dyn McpClient> = Arc::new(LocalMcpClient::new(LOCAL_SERVER_NAME));
    tokio::time::timeout(CALL_TIMEOUT, client.connect())
        .await
        .context("connecting the local MCP client timed out")??;
    common::banner::note(&format!(
        "connected to `{}` (is_connected = {})",
        client.server_name(),
        client.is_connected().await
    ));

    let tools = tokio::time::timeout(
        CALL_TIMEOUT,
        load_mcp_tools(Arc::clone(&client), McpToolOptions::default()),
    )
    .await
    .context("listing the local MCP tools timed out")??;
    common::banner::step(&format!("the server advertises {} tool(s)", tools.len()));
    for tool in &tools {
        let card = tool.card();
        let schema = serde_json::to_string_pretty(&card.input_params)
            .context("the MCP input schema is not JSON-serializable")?;
        common::banner::note(&format!(
            "{} (id {}) — {}",
            card.name, card.id, card.description
        ));
        common::banner::note(&format!("  input_params: {schema}"));
    }

    common::banner::step("one direct call, to a tool the server does not advertise");
    let unknown = tokio::time::timeout(
        CALL_TIMEOUT,
        client.call_tool_with("does_not_exist", json!({}), None, false),
    )
    .await
    .context("the direct MCP call timed out")??;
    common::banner::note(&format!(
        "`does_not_exist` success={} error={}",
        unknown.success,
        unknown.error_msg().unwrap_or("none")
    ));

    common::banner::step("mounting the MCP tools on a ReAct agent");
    let agent = ReActAgent::new(AgentCard::new("mcp-agent"));
    let mut config = ReActAgentConfig::new();
    config
        .configure_model(MODEL_NAME, MODEL_PROVIDER)
        .configure_max_iterations(4);
    agent.configure(config);
    for tool in tools {
        let outcome = agent.add_tool(tool.boxed());
        common::banner::note(&format!("mounted {} ({})", outcome.name, outcome.reason));
    }

    common::banner::step(&format!("asking the agent to echo: \"{PING}\""));
    agent.set_model(common::scripted_tool_round(
        ECHO_TOOL,
        json!({ "text": PING }),
        format!("The MCP server echoed \"{PING}\"."),
    ));
    let output = tokio::time::timeout(
        RUN_TIMEOUT,
        agent.invoke(json!(format!("Echo \"{PING}\" back to me.")), None),
    )
    .await
    .context("the agent run timed out")??;
    if output.observations.is_empty() {
        bail!("the scripted model asked for `{ECHO_TOOL}` but the agent recorded no observation");
    }
    for observation in &output.observations {
        common::banner::note(&format!(
            "observation from {}: {}",
            observation.name.as_deref().unwrap_or("unknown tool"),
            observation.content
        ));
    }
    common::banner::note(&format!(
        "iterations used: {}",
        output.iterations_used.unwrap_or_default()
    ));
    common::banner::answer(&output.output);
    disconnect(&client, "local").await;
    Ok(())
}

/// Part B: the same recipe against a real stdio MCP server, when configured.
async fn live_stdio_round() -> Result<()> {
    common::banner::step("part B: a real stdio MCP server (optional)");
    let command_line = std::env::var(MCP_SERVER_ENV).unwrap_or_default();
    let mut words = command_line.split_whitespace();
    let Some(program) = words.next() else {
        return common::skip(
            &format!("{MCP_SERVER_ENV} is not set or empty, so no real MCP server was contacted"),
            &format!("{MCP_SERVER_ENV}=\"<program and args>\""),
        );
    };
    let args: Vec<String> = words.map(ToOwned::to_owned).collect();
    common::banner::note(&format!(
        "starting `{program}` with {} argument(s)",
        args.len()
    ));

    // A stdio config is the executable plus its arguments; the client spawns the
    // process on its first operation and speaks MCP over the child's stdio.
    let mut config = McpServerConfig::stdio(LOCAL_SERVER_NAME, program);
    config.params = json!({ "args": args });
    let client = connect_mcp_server(&config).context("the MCP server config is unusable")?;

    let listing = tokio::time::timeout(
        LIVE_TIMEOUT,
        load_mcp_tools(Arc::clone(&client), McpToolOptions::default()),
    )
    .await;
    match listing {
        Ok(Ok(tools)) => {
            common::banner::note(&format!(
                "the live server advertises {} tool(s):",
                tools.len()
            ));
            for tool in &tools {
                common::banner::note(&format!("  {}", tool.card().name));
            }
        }
        Ok(Err(error)) => {
            common::banner::note(&format!("the live server could not be listed: {error}"));
        }
        Err(_) => common::banner::note("connecting to the live MCP server timed out"),
    }
    disconnect(&client, "live").await;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _mode = common::setup()?;
    common::banner::title("07 mcp tool");

    offline_round().await?;
    live_stdio_round().await?;
    Ok(())
}
