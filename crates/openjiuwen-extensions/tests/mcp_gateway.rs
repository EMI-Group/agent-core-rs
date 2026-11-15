//! Crate-boundary suite for the MCP tool gateway: `openjiuwen_extensions::mcp` public paths only.
//!
//! The in-crate suite (`src/mcp/tests.rs`) covers the projection and outcome tables in depth; this
//! file proves the same surface is usable by an external consumer with plain kernel `Tool`
//! objects, without reaching for any crate-private item.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result as KernelResult;
use openjiuwen_core::foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_extensions::mcp::McpToolGateway;
use openjiuwen_harness_protocol::{
    json_object_from_value, HarnessErrorKind, JsonValue, McpServerConfig, McpTransport,
    ToolGateway, ToolInvocation,
};
use serde_json::json;

/// A kernel tool that echoes its arguments back through a `ToolOutput` payload.
struct EchoTool {
    card: ToolCard,
}

#[async_trait]
impl Tool for EchoTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: JsonValue, _ctx: &ToolContext) -> KernelResult<JsonValue> {
        Ok(json!({
            "success": true,
            "data": {"content": format!("echo: {}", inputs["text"].as_str().unwrap_or_default())},
        }))
    }
}

/// Builds one echo tool named `name`.
fn echo_tool(name: &str) -> Arc<dyn Tool> {
    Arc::new(EchoTool {
        card: ToolCard::new(
            name,
            "echoes the text argument",
            json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }),
        ),
    })
}

/// Builds the fixture gateway over two tools declared in reverse name order.
fn gateway() -> McpToolGateway {
    McpToolGateway::new(vec![echo_tool("second"), echo_tool("first")]).expect("distinct names")
}

#[tokio::test]
async fn definitions_are_sorted_and_invocations_round_trip() {
    let gateway = gateway();
    let definitions = gateway.definitions().await.expect("definitions");

    assert_eq!(
        definitions
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(definitions[0].description, "echoes the text argument");
    assert_eq!(definitions[0].input_schema["required"], json!(["text"]));

    let invocation = ToolInvocation {
        call_id: "call-1".to_owned(),
        name: "second".to_owned(),
        arguments: json_object_from_value(json!({"text": "hello"})).expect("object"),
    };
    let result = gateway.invoke(&invocation).await.expect("invocation");
    assert!(!result.is_error);
    assert_eq!(result.content["text"], json!("echo: hello"));
    assert_eq!(
        result.content["structured"]["content"],
        json!("echo: hello")
    );
}

#[tokio::test]
async fn unknown_tool_is_a_typed_validation_error() {
    let gateway = gateway();
    let invocation = ToolInvocation {
        call_id: "call-1".to_owned(),
        name: "third".to_owned(),
        arguments: BTreeMap::new(),
    };
    let error = gateway.invoke(&invocation).await.expect_err("unknown tool");

    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert!(error.message().contains("third"));
}

#[tokio::test]
async fn server_metadata_and_permissions_are_opt_in() {
    let gateway = gateway()
        .with_servers(vec![McpServerConfig {
            name: "docs".to_owned(),
            transport: McpTransport::Http,
            command: Vec::new(),
            url: Some("https://mcp.example.test".to_owned()),
            instance: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
        }])
        .expect("valid server config");
    assert_eq!(gateway.servers().len(), 1);

    let gateway = gateway.with_permission(Arc::new(|name: &str| name == "first"));
    let names: Vec<String> = gateway
        .definitions()
        .await
        .expect("definitions")
        .into_iter()
        .map(|definition| definition.name)
        .collect();
    assert_eq!(names, ["first"]);
}
