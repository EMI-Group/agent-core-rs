//! Unit tests for MCP result normalization, server configuration, transport
//! validation, the reconnect policy, the SSE parser, and [`McpTool`].
//!
//! Ports the MCP unit tests of `openjiuwen/core/foundation/tool/mcp/`. Every
//! transport is replaced by [`FakeMcpClient`] and every protocol value is built
//! by hand: no test spawns a process, opens a socket, or touches the network.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use rmcp::model::{
    Annotated, CallToolResult, Content, RawAudioContent, RawContent, ResourceContents,
};
use serde_json::{json, Map, Value};

use crate::tool::card::ToolCard;
use crate::tool::context::ToolContext;
use crate::tool::traits::{Tool, EMPTY_SUCCESS_TEXT};

use super::client::{
    connect_mcp_server, reconnect_attempt_budget, reconnect_delay, FakeMcpClient,
    FakeSubscriptionAction, FakeSubscriptionCall, McpClient, McpClientType, McpRemoteTool,
    McpServerConfig,
};
use super::http::{SseFrame, SseParser};
use super::result::{
    extract_content, normalize_call_tool_result, McpToolResult, DEFAULT_MCP_ERROR_TEXT,
};
use super::tool::{load_mcp_tools, McpTool, McpToolOptions};

// --- fixtures ---------------------------------------------------------------

/// Builds a text content block.
fn text(value: &str) -> Content {
    Annotated::new(RawContent::text(value), None)
}

/// Builds an image content block with `data` as its base64 payload.
fn image(data: &str, mime_type: &str) -> Content {
    Annotated::new(RawContent::image(data, mime_type), None)
}

/// Builds an audio content block with `data` as its base64 payload.
fn audio(data: &str, mime_type: &str) -> Content {
    Annotated::new(
        RawContent::Audio(RawAudioContent {
            data: data.to_owned(),
            mime_type: mime_type.to_owned(),
        }),
        None,
    )
}

/// Builds an embedded text resource block for `uri`.
fn embedded_resource(uri: &str) -> Content {
    Annotated::new(
        RawContent::resource(ResourceContents::text("body", uri)),
        None,
    )
}

/// Builds a raw `CallToolResult` carrying `content`.
fn call_result(content: Vec<Content>, is_error: Option<bool>) -> CallToolResult {
    CallToolResult {
        content,
        structured_content: None,
        is_error,
        meta: None,
    }
}

/// A card whose schema requires `city`.
fn weather_card() -> ToolCard {
    ToolCard::new(
        "get_weather",
        "Get the weather",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
    )
}

/// A fake client with a fixed identity.
fn fake_client() -> Arc<FakeMcpClient> {
    Arc::new(FakeMcpClient::new("srv-id", "files"))
}

/// Wraps `client` as the weather tool with `options`.
fn weather_tool(client: Arc<dyn McpClient>, options: McpToolOptions) -> McpTool {
    McpTool::new(weather_card(), client, "get_weather").with_options(options)
}

// --- result normalization ---------------------------------------------------

// port of openjiuwen/core/foundation/tool/mcp/base.py extract_mcp_tool_result_content

#[test]
fn single_text_block_lands_under_content_as_a_string() {
    let result = extract_content(&[text("hello")], false);

    assert!(result.success);
    assert_eq!(result.data["content"], json!("hello"));
    assert_eq!(result.data["content"].as_str(), Some("hello"));
}

#[test]
fn two_text_blocks_are_joined_with_a_blank_line() {
    let result = extract_content(&[text("hello"), text("world")], false);

    assert_eq!(result.data["content"], json!("hello\n\nworld"));
}

#[test]
fn image_without_inline_data_becomes_a_text_placeholder() {
    let result = extract_content(&[image("abcdef", "image/png")], false);

    assert_eq!(
        result.data["content"],
        json!("[image content: image/png, 6 base64 chars]")
    );
    assert!(result.data.get("multimodal").is_none());
}

#[test]
fn image_with_inline_data_becomes_a_multimodal_entry() {
    let result = extract_content(&[image("abcdef", "image/png")], true);
    let multimodal = result.data["multimodal"].as_array().unwrap();

    assert_eq!(multimodal.len(), 1);
    assert_eq!(multimodal[0]["type"], json!("image"));
    assert_eq!(multimodal[0]["source"], json!("mcp"));
    assert_eq!(multimodal[0]["mime_type"], json!("image/png"));
    assert_eq!(
        multimodal[0]["data_url"],
        json!("data:image/png;base64,abcdef")
    );
    assert!(
        result.data["content"]
            .as_str()
            .unwrap()
            .contains("1 image(s) attached as multimodal input."),
        "unexpected content note: {:?}",
        result.data["content"]
    );
}

#[test]
fn audio_block_becomes_a_text_placeholder() {
    let result = extract_content(&[audio("abcdef", "audio/wav")], true);

    assert_eq!(
        result.data["content"],
        json!("[audio content: audio/wav, 6 base64 chars]")
    );
    assert!(result.data.get("multimodal").is_none());
}

#[test]
fn single_embedded_resource_lands_under_content() {
    let result = extract_content(&[embedded_resource("file:///tmp/notes.txt")], false);

    assert!(result.data["content"].is_object());
    assert_eq!(
        result.data["content"]["uri"],
        json!("file:///tmp/notes.txt")
    );
    assert!(result.data.get("resources").is_none());
}

#[test]
fn two_embedded_resources_land_under_a_resources_array() {
    let result = extract_content(
        &[
            embedded_resource("file:///tmp/one.txt"),
            embedded_resource("file:///tmp/two.txt"),
        ],
        false,
    );
    let resources = result.data["resources"].as_array().unwrap();

    assert_eq!(resources.len(), 2);
    assert_eq!(resources[0]["uri"], json!("file:///tmp/one.txt"));
    assert!(result.data.get("content").is_none());
}

#[test]
fn empty_content_list_yields_an_empty_payload() {
    let result = extract_content(&[], false);

    assert!(result.success);
    assert!(result.data.is_empty());
}

// port of openjiuwen/core/foundation/tool/mcp/base.py preserve_mcp_tool_result_status

#[test]
fn server_error_flag_uses_the_text_block_as_error() {
    let result = normalize_call_tool_result(&call_result(vec![text("boom")], Some(true)), false);

    assert!(!result.success);
    assert_eq!(result.error.as_deref(), Some("boom"));
}

#[test]
fn server_error_flag_without_text_uses_the_default_error_text() {
    let result = normalize_call_tool_result(
        &call_result(vec![image("abcdef", "image/png")], Some(true)),
        false,
    );

    assert!(!result.success);
    assert_eq!(result.error.as_deref(), Some(DEFAULT_MCP_ERROR_TEXT));
    assert_eq!(
        result.data["content"],
        json!("[image content: image/png, 6 base64 chars]")
    );
}

// --- server configuration ---------------------------------------------------

#[test]
fn default_config_uses_streamable_http_and_a_fresh_hex_id() {
    let first = McpServerConfig::default();
    let second = McpServerConfig::default();

    assert_eq!(first.client_type, McpClientType::StreamableHttp);
    assert_eq!(first.server_id.len(), 32);
    assert!(first
        .server_id
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_ne!(first.server_id, second.server_id);
}

#[test]
fn client_type_accepts_every_python_spelling() {
    for (spelling, expected) in [
        ("stdio", McpClientType::Stdio),
        ("sse", McpClientType::Sse),
        ("streamable-http", McpClientType::StreamableHttp),
        ("streamable_http", McpClientType::StreamableHttp),
        ("http", McpClientType::StreamableHttp),
        ("HTTP", McpClientType::StreamableHttp),
    ] {
        let config: McpServerConfig =
            serde_json::from_value(json!({ "client_type": spelling })).unwrap();
        assert_eq!(config.client_type, expected, "spelling `{spelling}`");
    }
}

#[test]
fn unknown_client_type_fails_deserialization() {
    let parsed: Result<McpServerConfig, _> =
        serde_json::from_value(json!({ "client_type": "carrier-pigeon" }));

    assert!(parsed.is_err());
}

#[test]
fn command_prefers_params_over_server_path_and_falls_back() {
    let mut config = McpServerConfig::stdio("files", "npx");

    config.params = json!({"command": "  node  "});
    assert_eq!(config.command(), Some("node"));

    config.params = json!({});
    assert_eq!(config.command(), Some("npx"));

    config.params = json!({"command": "   "});
    config.server_path = String::new();
    assert_eq!(config.command(), None);
}

#[test]
fn args_skips_non_string_entries() {
    let mut config = McpServerConfig::stdio("files", "npx");

    config.params = json!({"args": ["-y", 5, "@scope/server", null]});
    assert_eq!(config.args(), vec!["-y", "@scope/server"]);

    config.params = json!({"args": "not an array"});
    assert_eq!(config.args(), Vec::<String>::new());
}

#[test]
fn env_reads_string_values_only() {
    let mut config = McpServerConfig::stdio("files", "npx");

    config.params = json!({"env": {"TOKEN": "abc", "RETRIES": 3, "FLAG": true}});
    let env: BTreeMap<String, String> = config.env();
    assert_eq!(env.len(), 1);
    assert_eq!(env["TOKEN"], "abc");

    config.params = json!({"env": ["TOKEN=abc"]});
    assert!(config.env().is_empty());
}

#[test]
fn cwd_comes_from_params_only() {
    let mut config = McpServerConfig::stdio("files", "npx");

    config.params = json!({"cwd": "/tmp/work"});
    assert_eq!(config.cwd().as_deref(), Some("/tmp/work"));

    config.params = json!({});
    assert_eq!(config.cwd(), None);
}

#[test]
fn empty_params_yield_no_command_args_env_or_cwd() {
    let config = McpServerConfig {
        server_path: String::new(),
        params: json!({}),
        ..McpServerConfig::default()
    };

    assert_eq!(config.command(), None);
    assert_eq!(config.args(), Vec::<String>::new());
    assert!(config.env().is_empty());
    assert_eq!(config.cwd(), None);
}

// --- transport validation ---------------------------------------------------

#[test]
fn connect_mcp_server_accepts_a_stdio_command() {
    let client = connect_mcp_server(&McpServerConfig::stdio("files", "npx")).unwrap();

    assert_eq!(client.server_name(), "files");
}

#[test]
fn stdio_without_a_command_is_a_validation_error() {
    let config = McpServerConfig::stdio("files", "");

    let error = connect_mcp_server(&config).err().unwrap();

    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn http_without_an_absolute_url_is_a_validation_error() {
    let error = connect_mcp_server(&McpServerConfig::http("files", "not-a-url"))
        .err()
        .unwrap();

    assert_eq!(error.kind(), ErrorKind::Validation);
}

// --- reconnect policy -------------------------------------------------------

// port of openjiuwen/core/foundation/tool/mcp/client/reconnect.py

#[test]
fn reconnect_delay_follows_the_python_backoff() {
    for (attempt, expected) in [
        (1, 5_u64),
        (2, 10),
        (3, 20),
        (4, 40),
        (5, 80),
        (6, 120),
        (7, 120),
    ] {
        assert_eq!(
            reconnect_delay(attempt),
            Duration::from_secs(expected),
            "attempt {attempt}"
        );
    }
}

#[test]
fn reconnect_attempt_budget_is_never_below_one() {
    assert!(reconnect_attempt_budget() >= 1);
}

// --- SSE parser -------------------------------------------------------------

#[test]
fn sse_parser_reads_one_frame_from_one_chunk() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"data: hello\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].data, "hello");
    assert_eq!(frames[0].event, None);
}

#[test]
fn sse_parser_reads_two_frames_from_one_chunk() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"data: one\n\ndata: two\n\n");

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].data, "one");
    assert_eq!(frames[1].data, "two");
}

#[test]
fn sse_parser_reassembles_data_split_across_chunks() {
    let mut parser = SseParser::default();

    assert_eq!(parser.push(b"data: hel"), Vec::new());
    let frames = parser.push(b"lo\n\n");

    assert_eq!(
        frames,
        vec![SseFrame {
            event: None,
            data: "hello".to_owned(),
            id: None,
            retry: None,
        }]
    );
}

#[test]
fn sse_parser_joins_multiline_data_with_newlines() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"data: first\ndata: second\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].data, "first\nsecond");
}

#[test]
fn sse_parser_accepts_crlf_line_endings() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"data: hello\r\n\r\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].data, "hello");
}

#[test]
fn sse_parser_ignores_comment_lines() {
    let mut parser = SseParser::default();

    let frames = parser.push(b": keep-alive\n\ndata: hello\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].data, "hello");
    assert_eq!(frames[0].event, None);
}

#[test]
fn sse_parser_reads_event_and_data_fields() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"event: endpoint\ndata: /messages\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].event.as_deref(), Some("endpoint"));
    assert_eq!(frames[0].data, "/messages");
}

#[test]
fn sse_parser_reassembles_a_multibyte_character_split_across_chunks() {
    let mut parser = SseParser::default();
    let body = "data: {\"result\":\"你好\"}\n\n".as_bytes();
    // Split in the middle of the 3-byte encoding of `你`.
    let split = body
        .windows(3)
        .position(|window| window == "你".as_bytes())
        .expect("the character is in the body")
        + 1;
    assert_eq!(parser.push(&body[..split]), Vec::new());
    let frames = parser.push(&body[split..]);
    assert_eq!(
        frames.len(),
        1,
        "the split character was dropped: {frames:?}"
    );
    assert_eq!(frames[0].data, "{\"result\":\"你好\"}");
}

#[test]
fn sse_parser_tolerates_genuinely_invalid_utf8() {
    let mut parser = SseParser::default();
    let frames = parser.push(b"data: a\xFFb\n\n");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].data, "a\u{FFFD}b");
}

// --- McpTool ----------------------------------------------------------------

#[tokio::test]
async fn invoke_returns_the_normalized_result_and_forwards_call_options() {
    let client = fake_client();
    let mut data = Map::new();
    data.insert("content".to_owned(), json!("sunny"));
    client
        .set_result("get_weather", McpToolResult::success(data))
        .await;
    let tool = weather_tool(
        client.clone(),
        McpToolOptions::default().with_include_image_content(true),
    );
    let ctx = ToolContext::new().with_timeout(Duration::from_secs(7));

    let output = tool.invoke(json!({"city": "London"}), &ctx).await.unwrap();

    assert_eq!(
        output,
        json!({"success": true, "data": {"content": "sunny"}, "error": null})
    );
    let calls = client.calls().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_name, "get_weather");
    assert_eq!(calls[0].arguments, json!({"city": "London"}));
    assert!(calls[0].include_image_content);
    assert_eq!(calls[0].timeout, Some(Duration::from_secs(7)));
}

#[tokio::test]
async fn invoke_rejects_inputs_that_do_not_match_the_card_schema() {
    let client = fake_client();
    let tool = weather_tool(client.clone(), McpToolOptions::default());

    let error = tool
        .invoke(json!({}), &ToolContext::new())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(client.calls().await, Vec::new());
}

#[tokio::test]
async fn invoke_honors_a_cancelled_context() {
    let client = fake_client();
    let tool = weather_tool(client.clone(), McpToolOptions::default());
    let ctx = ToolContext::new();
    ctx.cancellation.cancel();

    let error = tool
        .invoke(json!({"city": "London"}), &ctx)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(client.calls().await, Vec::new());
}

#[tokio::test]
async fn invoke_maps_a_client_failure_to_mcp() {
    let client = fake_client();
    client.set_failure("get_weather", "boom").await;
    let tool = weather_tool(client.clone(), McpToolOptions::default());

    let error = tool
        .invoke(json!({"city": "London"}), &ToolContext::new())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Mcp);
}

#[tokio::test]
async fn invoke_sends_non_object_inputs_as_an_empty_object() {
    let client = fake_client();
    let tool = McpTool::new(
        ToolCard::new("echo", "Echo", json!({"type": "object"})),
        client.clone(),
        "echo",
    );

    tool.invoke(json!([1, 2]), &ToolContext::new())
        .await
        .unwrap();

    let calls = client.calls().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, json!({}));
}

#[tokio::test]
async fn context_timeout_wins_over_the_tool_option_timeout() {
    let client = fake_client();
    let tool = weather_tool(
        client.clone(),
        McpToolOptions::default().with_timeout(Duration::from_secs(30)),
    );
    let ctx = ToolContext::new().with_timeout(Duration::from_secs(5));

    tool.invoke(json!({"city": "London"}), &ctx).await.unwrap();

    let calls = client.calls().await;
    assert_eq!(calls[0].timeout, Some(Duration::from_secs(5)));
}

// port of openjiuwen/core/foundation/tool/mcp/base.py MCPTool.render_for_llm

#[test]
fn render_for_llm_renders_the_content_text() {
    let tool = weather_tool(fake_client(), McpToolOptions::default());

    let rendered = tool
        .render_for_llm(&json!({"success": true, "data": {"content": "London"}, "error": null}));

    assert_eq!(rendered, "London");
}

#[test]
fn render_for_llm_ignores_the_success_flag_and_the_error_field() {
    let tool = weather_tool(fake_client(), McpToolOptions::default());

    let rendered = tool
        .render_for_llm(&json!({"success": false, "data": {"content": "oops"}, "error": "bad"}));

    assert_eq!(rendered, "oops");
}

#[test]
fn render_for_llm_unwraps_a_plain_result_object() {
    let tool = weather_tool(fake_client(), McpToolOptions::default());

    assert_eq!(tool.render_for_llm(&json!({"result": "London"})), "London");
}

#[test]
fn render_for_llm_falls_back_to_the_empty_success_text() {
    let tool = weather_tool(fake_client(), McpToolOptions::default());

    let rendered =
        tool.render_for_llm(&json!({"success": true, "data": {"content": null}, "error": null}));

    assert_eq!(rendered, EMPTY_SUCCESS_TEXT);
}

// --- load_mcp_tools ---------------------------------------------------------

#[tokio::test]
async fn load_mcp_tools_projects_cards_and_copies_options() {
    let client = fake_client();
    let schema = json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    });
    client
        .set_tools(vec![
            McpRemoteTool::new("read_file", "Read a file", schema.clone()),
            McpRemoteTool::new("write_file", "Write a file", json!({})),
        ])
        .await;
    let options = McpToolOptions::default()
        .with_include_image_content(true)
        .with_timeout(Duration::from_secs(9));
    let shared: Arc<dyn McpClient> = client.clone();

    let tools = load_mcp_tools(shared, options.clone()).await.unwrap();

    assert_eq!(tools.len(), 2);
    let card = tools[0].card();
    assert_eq!(card.id, "srv-id.files.read_file");
    assert_eq!(card.name, "read_file");
    assert_eq!(card.description, "Read a file");
    assert_eq!(card.input_params, schema);
    assert_eq!(card.metadata["mcp"], json!(true));
    assert_eq!(card.metadata["server_name"], json!("files"));
    assert_eq!(card.metadata["server_id"], json!("srv-id"));
    assert_eq!(tools[0].tool_name(), "read_file");
    assert_eq!(tools[0].options(), &options);
    assert_eq!(tools[1].card().id, "srv-id.files.write_file");
    assert_eq!(tools[1].options(), &options);
}

#[tokio::test]
async fn load_mcp_tools_returns_an_empty_vec_for_an_empty_tool_list() {
    let client = fake_client();
    client.set_tools(vec![]).await;
    let shared: Arc<dyn McpClient> = client.clone();

    let tools = load_mcp_tools(shared, McpToolOptions::default())
        .await
        .unwrap();

    assert!(tools.is_empty());
}

// --- FakeMcpClient ----------------------------------------------------------

#[tokio::test]
async fn fake_client_counts_a_single_connection_across_repeated_connects() {
    let client = FakeMcpClient::new("srv-id", "files");

    client.connect().await.unwrap();
    client.connect().await.unwrap();

    assert_eq!(client.connect_count(), 1);
}

#[tokio::test]
async fn fake_client_connection_state_follows_disconnect() {
    let client = FakeMcpClient::new("srv-id", "files");
    client.connect().await.unwrap();

    assert!(client.is_connected().await);

    client.disconnect().await.unwrap();

    assert!(!client.is_connected().await);
}

// --- resource subscriptions -------------------------------------------------

/// A client that implements only the required [`McpClient`] methods.
struct BareClient;

#[async_trait]
impl McpClient for BareClient {
    fn server_name(&self) -> &'static str {
        "bare"
    }

    async fn connect(&self) -> Result<()> {
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        false
    }

    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
        Ok(Vec::new())
    }

    async fn call_tool_with(
        &self,
        _tool_name: &str,
        _arguments: Value,
        _timeout: Option<Duration>,
        _include_image_content: bool,
    ) -> Result<McpToolResult> {
        Ok(McpToolResult::success(Map::new()))
    }
}

#[tokio::test]
async fn default_subscription_methods_report_unsupported_naming_the_transport() {
    let client = BareClient;

    assert!(!client.supports_resource_subscriptions());
    let subscribe = client
        .subscribe_resource("file:///tmp/report.md")
        .await
        .expect_err("the default must not subscribe");
    let unsubscribe = client
        .unsubscribe_resource("file:///tmp/report.md")
        .await
        .expect_err("the default must not unsubscribe");

    assert_eq!(subscribe.kind(), ErrorKind::Unsupported);
    assert_eq!(unsubscribe.kind(), ErrorKind::Unsupported);
    assert!(subscribe.message().contains("transport"), "{subscribe}");
    assert!(
        subscribe.message().contains("resources/subscribe"),
        "{subscribe}"
    );
    assert!(
        unsubscribe.message().contains("resources/unsubscribe"),
        "{unsubscribe}"
    );
}

#[tokio::test]
async fn fake_client_reports_the_scripted_subscription_capability() {
    let client = FakeMcpClient::new("srv-id", "files");

    assert!(!client.supports_resource_subscriptions());

    client.set_resource_subscriptions_supported(true);

    assert!(client.supports_resource_subscriptions());
}

#[tokio::test]
async fn fake_client_records_subscription_requests() {
    let client = FakeMcpClient::new("srv-id", "files");
    client.set_resource_subscriptions_supported(true);

    client
        .subscribe_resource("file:///tmp/report.md")
        .await
        .unwrap();
    client
        .unsubscribe_resource("file:///tmp/report.md")
        .await
        .unwrap();

    assert_eq!(
        client.subscription_calls().await,
        vec![
            FakeSubscriptionCall {
                action: FakeSubscriptionAction::Subscribe,
                uri: "file:///tmp/report.md".to_owned(),
            },
            FakeSubscriptionCall {
                action: FakeSubscriptionAction::Unsubscribe,
                uri: "file:///tmp/report.md".to_owned(),
            },
        ]
    );
}

#[tokio::test]
async fn fake_client_reports_a_scripted_subscription_failure() {
    let client = FakeMcpClient::new("srv-id", "files");
    client.set_resource_subscriptions_supported(true);
    client
        .set_subscription_failure("resource is write-only")
        .await;

    let error = client
        .subscribe_resource("file:///tmp/report.md")
        .await
        .expect_err("scripted failure");

    assert_eq!(error.kind(), ErrorKind::Mcp);
    assert!(error.message().contains("write-only"), "{error}");

    client.clear_subscription_failure().await;
    client
        .subscribe_resource("file:///tmp/report.md")
        .await
        .unwrap();

    assert_eq!(client.subscription_calls().await.len(), 2);
}
