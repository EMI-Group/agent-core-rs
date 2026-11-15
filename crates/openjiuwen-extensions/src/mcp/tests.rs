//! Tests for the MCP tool gateway.
//!
//! The MCP tools under test are real `McpTool`s over a scripted in-process `McpClient`, so the
//! kernel projection (`load_mcp_tools`), argument validation, and result normalization all run
//! for real; only the JSON-RPC transport is faked. The custom tools cover the kernel outcomes an
//! `McpTool` never produces as a value (`ToolOutput` with `success: false`) and the kernel error
//! kinds the gateway maps onto typed harness errors.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result as KernelResult};
use openjiuwen_core::foundation::tool::mcp::{
    load_mcp_tools, McpClient, McpRemoteTool, McpTool, McpToolOptions, McpToolResult,
};
use openjiuwen_core::foundation::tool::{Tool, ToolCard, ToolContext, EMPTY_SUCCESS_TEXT};
use openjiuwen_harness_protocol::{
    json_object_from_value, HarnessErrorKind, JsonValue, McpServerConfig, McpTransport,
    ToolGateway, ToolInvocation,
};
use serde_json::json;

use super::{McpToolGateway, PermissionFilter};

/// A scripted `McpClient`: no transport, no network, every reply pre-set.
struct FakeClient {
    server_id: String,
    server_name: String,
    tools: Vec<McpRemoteTool>,
    replies: Mutex<HashMap<String, McpToolResult>>,
    failures: Mutex<HashMap<String, String>>,
    calls: Mutex<Vec<(String, JsonValue)>>,
}

impl FakeClient {
    fn new(server_id: &str, server_name: &str, tools: Vec<McpRemoteTool>) -> Self {
        Self {
            server_id: server_id.to_owned(),
            server_name: server_name.to_owned(),
            tools,
            replies: Mutex::new(HashMap::new()),
            failures: Mutex::new(HashMap::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Scripts a successful reply whose `data` is `payload`.
    fn script_success(&self, tool_name: &str, payload: JsonValue) {
        let data = payload.as_object().cloned().unwrap_or_default();
        self.replies
            .lock()
            .unwrap()
            .insert(tool_name.to_owned(), McpToolResult::success(data));
    }

    /// Scripts a server-side failure for `tool_name`.
    fn script_failure(&self, tool_name: &str, message: &str) {
        self.failures
            .lock()
            .unwrap()
            .insert(tool_name.to_owned(), message.to_owned());
    }

    /// Returns the calls this client received, in order.
    fn calls(&self) -> Vec<(String, JsonValue)> {
        self.calls.lock().unwrap().clone()
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

    async fn connect(&self) -> KernelResult<()> {
        Ok(())
    }

    async fn disconnect(&self) -> KernelResult<()> {
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        true
    }

    async fn list_tools(&self) -> KernelResult<Vec<McpRemoteTool>> {
        Ok(self.tools.clone())
    }

    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: JsonValue,
        _timeout: Option<Duration>,
        _include_image_content: bool,
    ) -> KernelResult<McpToolResult> {
        self.calls
            .lock()
            .unwrap()
            .push((tool_name.to_owned(), arguments));
        if let Some(message) = self.failures.lock().unwrap().get(tool_name) {
            return Err(Error::mcp(message.clone()));
        }
        self.replies
            .lock()
            .unwrap()
            .get(tool_name)
            .cloned()
            .ok_or_else(|| Error::tool(format!("no scripted reply for `{tool_name}`")))
    }
}

/// The two remote tools the fake server advertises, deliberately unsorted.
fn remote_tools() -> Vec<McpRemoteTool> {
    vec![
        McpRemoteTool::new(
            "search_docs",
            "Search the team documentation",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }),
        ),
        McpRemoteTool::new(
            "add_numbers",
            "Add two integers",
            json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            }),
        ),
    ]
}

/// Builds the fixture gateway over two real `McpTool`s from one fake server.
async fn fixture() -> (Arc<FakeClient>, McpToolGateway) {
    let client = Arc::new(FakeClient::new("srv-1", "docs", remote_tools()));
    client.script_success("search_docs", json!({"content": "found 2 docs"}));
    let tools = load_mcp_tools(client.clone(), McpToolOptions::default())
        .await
        .unwrap();
    let gateway = McpToolGateway::new(tools.into_iter().map(McpTool::boxed)).unwrap();
    (client, gateway)
}

/// Builds an invocation for `tool_name` with `arguments`.
fn invocation(tool_name: &str, arguments: JsonValue) -> ToolInvocation {
    ToolInvocation {
        call_id: format!("call-{tool_name}"),
        name: tool_name.to_owned(),
        arguments: json_object_from_value(arguments).unwrap(),
    }
}

/// A kernel error a tool returns instead of a value.
#[derive(Debug, Clone)]
enum Failure {
    Tool(String),
    Unsupported(String),
    Config(String),
}

/// A tool with a scripted outcome, for kernel paths an `McpTool` cannot produce.
struct FixtureTool {
    card: ToolCard,
    outcome: Result<JsonValue, Failure>,
}

#[async_trait]
impl Tool for FixtureTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, _inputs: JsonValue, _ctx: &ToolContext) -> KernelResult<JsonValue> {
        match &self.outcome {
            Ok(value) => Ok(value.clone()),
            Err(Failure::Tool(message)) => Err(Error::tool(message.clone())),
            Err(Failure::Unsupported(message)) => Err(Error::unsupported(message.clone())),
            Err(Failure::Config(message)) => Err(Error::config(message.clone())),
        }
    }
}

/// Builds one tool with a scripted outcome under `name`.
fn custom_tool(name: &str, outcome: Result<JsonValue, Failure>) -> Arc<dyn Tool> {
    Arc::new(FixtureTool {
        card: ToolCard::new(name, "fixture tool", json!({"type": "object"})),
        outcome,
    })
}

/// Builds a valid streamable-HTTP server config.
fn http_server(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_owned(),
        transport: McpTransport::Http,
        command: Vec::new(),
        url: Some("https://mcp.example.test".to_owned()),
        instance: None,
        env: BTreeMap::new(),
        headers: BTreeMap::new(),
    }
}

/// Returns the definition names of `gateway`, in order.
async fn definition_names(gateway: &McpToolGateway) -> Vec<String> {
    gateway
        .definitions()
        .await
        .unwrap()
        .into_iter()
        .map(|definition| definition.name)
        .collect()
}

#[tokio::test]
async fn definitions_are_sorted_and_keep_name_description_and_schema() {
    let (_client, gateway) = fixture().await;
    let definitions = gateway.definitions().await.unwrap();

    assert_eq!(
        definition_names(&gateway).await,
        ["add_numbers", "search_docs"]
    );
    assert_eq!(definitions[0].name, "add_numbers");
    assert_eq!(definitions[0].description, "Add two integers");
    assert_eq!(definitions[0].input_schema["required"], json!(["a", "b"]));
    assert_eq!(definitions[1].description, "Search the team documentation");
    assert_eq!(definitions[1].input_schema["required"], json!(["query"]));
    assert_eq!(
        definitions[1].input_schema["properties"]["query"]["type"],
        json!("string")
    );
}

#[tokio::test]
async fn entries_carry_the_mcp_provenance_the_definition_cannot() {
    let (_client, gateway) = fixture().await;
    let entries: Vec<_> = gateway.entries().collect();
    assert_eq!(entries.len(), 2);

    let search = entries
        .iter()
        .find(|entry| entry.name() == "search_docs")
        .unwrap();
    assert!(search.is_mcp_served());
    assert_eq!(search.server_name(), Some("docs"));
    assert_eq!(search.server_id(), Some("srv-1"));
    assert_eq!(search.card_id, "srv-1.docs.search_docs");
    assert!(search.parallel_safe);
    assert_eq!(search.metadata["mcp"], json!(true));
    assert!(gateway.servers().is_empty());
    assert!(!gateway.is_empty());
}

#[tokio::test]
async fn invoke_round_trips_a_successful_call() {
    let (client, gateway) = fixture().await;
    let result = gateway
        .invoke(&invocation("search_docs", json!({"query": "rust"})))
        .await
        .unwrap();

    assert!(!result.is_error);
    assert_eq!(result.content["text"], json!("found 2 docs"));
    assert_eq!(
        result.content["structured"],
        json!({"content": "found 2 docs"})
    );
    assert_eq!(
        client.calls(),
        vec![("search_docs".to_owned(), json!({"query": "rust"}))]
    );
}

#[tokio::test]
async fn unknown_tool_is_a_validation_error_listing_the_served_tools() {
    let (_client, gateway) = fixture().await;
    let error = gateway
        .invoke(&invocation("missing_tool", json!({})))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert!(error.message().contains("missing_tool"));
    assert!(error.message().contains("add_numbers, search_docs"));
}

#[tokio::test]
async fn blank_invocation_is_a_validation_error() {
    let (_client, gateway) = fixture().await;
    let blank = ToolInvocation {
        call_id: String::new(),
        name: String::new(),
        arguments: BTreeMap::new(),
    };
    let error = gateway.invoke(&blank).await.unwrap_err();

    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(
        error.message(),
        "tool invocation call_id and name must not be empty"
    );
}

#[tokio::test]
async fn failing_server_call_becomes_an_error_result_naming_the_tool() {
    let (client, gateway) = fixture().await;
    client.script_failure("add_numbers", "server rejected the call");

    let result = gateway
        .invoke(&invocation("add_numbers", json!({"a": 1, "b": 2})))
        .await
        .unwrap();

    assert!(result.is_error);
    let message = result.content["error"].as_str().unwrap();
    assert!(message.starts_with("MCP tool `add_numbers` failed: "));
    assert!(message.contains("server rejected the call"));
}

#[tokio::test]
async fn invalid_arguments_are_a_validation_error_and_never_reach_the_server() {
    let (client, gateway) = fixture().await;
    let error = gateway
        .invoke(&invocation("search_docs", json!({})))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert!(error
        .message()
        .starts_with("MCP tool `search_docs` failed: "));
    assert!(error.message().contains("input does not match schema"));
    assert!(client.calls().is_empty());
}

#[tokio::test]
async fn duplicate_and_blank_tool_names_are_rejected() {
    let duplicate = McpToolGateway::new(vec![
        custom_tool("same", Ok(json!({"success": true}))),
        custom_tool("same", Ok(json!({"success": true}))),
    ])
    .expect_err("duplicate names are ambiguous");
    assert_eq!(duplicate.kind(), HarnessErrorKind::Validation);
    assert!(duplicate
        .message()
        .contains("duplicate MCP tool name `same`"));

    let blank = McpToolGateway::new(vec![custom_tool("   ", Ok(json!({"success": true})))])
        .expect_err("a blank name cannot be invoked by name");
    assert_eq!(blank.kind(), HarnessErrorKind::Validation);
    assert!(blank.message().contains("non-empty tool name"));
}

#[tokio::test]
async fn servers_are_validated_and_sorted() {
    let (_client, gateway) = fixture().await;
    let gateway = gateway
        .with_servers(vec![http_server("zeta"), http_server("alpha")])
        .unwrap();

    let names: Vec<&str> = gateway
        .servers()
        .iter()
        .map(|server| server.name.as_str())
        .collect();
    assert_eq!(names, ["alpha", "zeta"]);

    let mut invalid = http_server("broken");
    invalid.url = None;
    let error = gateway
        .with_servers(vec![invalid])
        .expect_err("an HTTP server needs a url");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert!(error.message().contains("invalid MCP server `broken`"));
    assert!(error.message().contains("HTTP MCP server requires url"));
}

#[tokio::test]
async fn permission_filter_hides_and_denies_tools() {
    let (_client, gateway) = fixture().await;
    let filter: PermissionFilter = Arc::new(|name: &str| name != "add_numbers");
    let gateway = gateway.with_permission(filter);

    assert_eq!(definition_names(&gateway).await, ["search_docs"]);
    assert_eq!(gateway.entries().count(), 2, "entries are unfiltered");

    let denied = gateway
        .invoke(&invocation("add_numbers", json!({"a": 1, "b": 2})))
        .await
        .unwrap_err();
    assert_eq!(denied.kind(), HarnessErrorKind::State);
    assert!(denied.message().contains("add_numbers"));
    assert!(denied.message().contains("denied"));

    let allowed = gateway
        .invoke(&invocation("search_docs", json!({"query": "rust"})))
        .await;
    assert!(allowed.is_ok(), "the allowed tool still runs");
}

#[tokio::test]
async fn tool_output_failure_becomes_an_error_result() {
    let gateway = McpToolGateway::new(vec![custom_tool(
        "flaky",
        Ok(json!({"success": false, "error": "boom"})),
    )])
    .unwrap();

    let result = gateway
        .invoke(&invocation("flaky", json!({})))
        .await
        .unwrap();
    assert!(result.is_error);
    assert_eq!(
        result.content["error"],
        json!("MCP tool `flaky` failed: boom")
    );
}

#[tokio::test]
async fn tool_output_without_a_payload_uses_the_kernel_placeholder() {
    let gateway = McpToolGateway::new(vec![custom_tool(
        "silent",
        Ok(json!({"success": true, "data": null})),
    )])
    .unwrap();

    let result = gateway
        .invoke(&invocation("silent", json!({})))
        .await
        .unwrap();
    assert!(!result.is_error);
    assert_eq!(result.content["text"], json!(EMPTY_SUCCESS_TEXT));
    assert_eq!(result.content["structured"], JsonValue::Null);
}

#[tokio::test]
async fn plain_values_project_onto_text_and_structured_content() {
    let gateway = McpToolGateway::new(vec![custom_tool("plain", Ok(json!("hi")))]).unwrap();

    let result = gateway
        .invoke(&invocation("plain", json!({})))
        .await
        .unwrap();
    assert!(!result.is_error);
    assert_eq!(result.content["text"], json!("hi"));
    assert_eq!(result.content["structured"], json!("hi"));
}

#[tokio::test]
async fn unsupported_and_config_errors_map_to_typed_harness_errors() {
    let gateway = McpToolGateway::new(vec![
        custom_tool(
            "deferred",
            Err(Failure::Unsupported("no browser runtime".to_owned())),
        ),
        custom_tool(
            "misfiled",
            Err(Failure::Config("server path is empty".to_owned())),
        ),
    ])
    .unwrap();

    let unsupported = gateway
        .invoke(&invocation("deferred", json!({})))
        .await
        .unwrap_err();
    assert_eq!(unsupported.kind(), HarnessErrorKind::UnsupportedCapability);
    assert!(unsupported.message().contains("deferred"));
    assert!(unsupported.message().contains("no browser runtime"));

    let config = gateway
        .invoke(&invocation("misfiled", json!({})))
        .await
        .unwrap_err();
    assert_eq!(config.kind(), HarnessErrorKind::Protocol);
    assert!(config.message().contains("misfiled"));
    assert!(config.message().contains("server path is empty"));
}

#[tokio::test]
async fn tool_errors_become_error_results_and_long_details_are_truncated() {
    let gateway = McpToolGateway::new(vec![custom_tool(
        "verbose",
        Err(Failure::Tool("x".repeat(2_000))),
    )])
    .unwrap();

    let result = gateway
        .invoke(&invocation("verbose", json!({})))
        .await
        .unwrap();
    assert!(result.is_error);
    let message = result.content["error"].as_str().unwrap();
    assert!(message.starts_with("MCP tool `verbose` failed: "));
    assert_eq!(
        message.chars().count(),
        "MCP tool `verbose` failed: ".chars().count() + 512 + 1
    );
    assert!(message.ends_with('…'));
}
