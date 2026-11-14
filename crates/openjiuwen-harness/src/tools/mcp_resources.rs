//! MCP resource tools: `list_mcp_resources` and `read_mcp_resource`.
//!
//! Ports `openjiuwen/harness/tools/mcp_tools.py`. The tools work through a narrow
//! [`McpResourceProvider`] seam so they stay independent of the MCP client transport; with no
//! provider wired they return a typed `success=false` payload instead of degrading silently.
//!
//! # Live client adapter
//!
//! The foundation [`openjiuwen_foundation::tool::mcp::McpClient`] trait exposes
//! `list_resources`/`read_resource` (raw `rmcp` JSON) next to tool discovery and tool calls, so
//! [`McpClientResourceProvider`] adapts it to this seam: list and read are delegated and
//! normalised, and every subscription entry point answers [`Error::Unsupported`] because the
//! trait exposes no `resources/subscribe` (see [`super::mcp_subscriptions`] for the registry the
//! deferred capability would feed). [`FakeMcpResourceProvider`] is the in-crate double tests
//! (and upper layers) use, following the precedent of
//! `openjiuwen_foundation::tool::mcp::FakeMcpClient`.
//!
//! # Model-facing contract
//!
//! - `list_mcp_resources` takes an empty argument object and returns the envelope with
//!   `data = { "resources": [{ "uri", "name", "mime_type", "description" }] }`.
//! - `read_mcp_resource` takes `{ "uri": <string> }` and returns
//!   `data = { "resource": { "uri", "mime_type", "text", "blob_base64", "byte_size" } }`.
//!
//! Both report the failure message of the provider unchanged as the payload `error`, and
//! report `mcp resource provider is not configured` when no provider is wired in.
//!
//! Differences from Python, all deliberate:
//!
//! - Python requires a `server_id` argument on both tools because its resource manager is
//!   keyed by server; here the provider is bound to one server at construction, so the tools
//!   take no `server_id` and any supplied value is ignored.
//! - Python's `data` is a bare list of resource/content dicts; here it is wrapped in
//!   `{"resources": ...}` / `{"resource": ...}` so a future multi-server provider can extend
//!   the payload without changing the shape.
//! - Python uses the wire spellings `mimeType`/`text`; the JSON here is `snake_case`
//!   (`mime_type`) per the crate's serialization rule, and `blob_base64`/`byte_size` are
//!   Rust-only additions so binary resources survive the round trip.
//! - Python's per-tool `render_for_llm` (one resource per line, joined text blocks) is not
//!   ported: the payload renders through the default [`Tool`] rendering.
//! - A read that returns several `contents` entries (a multi-part resource) keeps the first and
//!   logs the rest at `debug`, because this seam models one content per URI.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::mcp::McpClient;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::descriptions::description;
use super::mcp_subscriptions::McpResourceSubscriptionProvider;
use super::support::{parse_args, schema_for, tool_card, ToolPayload};

/// The failure reported when no MCP resource provider is wired into a tool.
const NO_PROVIDER_MESSAGE: &str = "mcp resource provider is not configured";

/// The model-facing name of the resource listing tool.
const LIST_TOOL_NAME: &str = "list_mcp_resources";

/// The model-facing name of the resource reading tool.
const READ_TOOL_NAME: &str = "read_mcp_resource";

/// One resource exposed by an MCP server.
///
/// Python parity: the `uri`/`name`/`mimeType`/`description` fields of an MCP resource, with
/// the `snake_case` `mime_type` spelling used throughout this crate. Unknown JSON fields are
/// tolerated and every field defaults, so a partial server response still deserializes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct McpResource {
    /// Stable resource URI, as the server reports it.
    pub uri: String,
    /// Human-readable resource name.
    pub name: String,
    /// MIME type of the resource content, when the server declares one.
    pub mime_type: Option<String>,
    /// Optional server-side description of the resource.
    pub description: Option<String>,
}

/// The content of one MCP resource.
///
/// Exactly one of `text` and `blob_base64` is normally set: MCP resource contents are either
/// text or base64-encoded binary.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct McpResourceContent {
    /// URI this content was read from.
    pub uri: String,
    /// MIME type of the content, when the server declares one.
    pub mime_type: Option<String>,
    /// UTF-8 text content, for text resources.
    pub text: Option<String>,
    /// Base64-encoded content, for binary resources.
    pub blob_base64: Option<String>,
    /// Size of the content in bytes; `0` when the provider does not report it.
    pub byte_size: u64,
}

/// The seam between the MCP resource tools and an MCP server connection.
///
/// Python parity: `Runner.resource_mgr.list_mcp_resources` / `read_mcp_resource`, which this
/// trait replaces because the foundation `McpClient` has no resource methods yet. Implement it
/// over a live transport to enable the tools; [`FakeMcpResourceProvider`] is the test double.
#[async_trait]
pub trait McpResourceProvider: Send + Sync {
    /// Lists the resources the bound server exposes.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the server cannot be reached or
    /// rejects the request; the tool forwards the message as a `success=false` payload.
    async fn list_resources(&self) -> Result<Vec<McpResource>>;

    /// Reads the content of the resource named `uri`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the resource does not exist, the
    /// server cannot be reached, or the request is rejected; the tool forwards the message as
    /// a `success=false` payload.
    async fn read_resource(&self, uri: &str) -> Result<McpResourceContent>;
}

/// A scriptable [`McpResourceProvider`] for tests and offline wiring.
///
/// Holds canned resources and contents, records every call, and can be told to fail listing
/// or reading a specific URI. All state is behind a mutex, so one instance can be shared by
/// both tools through an `Arc`. Precedent: `openjiuwen_foundation::tool::mcp::FakeMcpClient`.
#[derive(Debug, Default)]
pub struct FakeMcpResourceProvider {
    /// Resources returned by [`McpResourceProvider::list_resources`].
    resources: Mutex<Vec<McpResource>>,
    /// Canned read results, keyed by URI.
    contents: Mutex<BTreeMap<String, McpResourceContent>>,
    /// When set, listing fails with this message.
    list_failure: Mutex<Option<String>>,
    /// Per-URI read failures.
    read_failures: Mutex<BTreeMap<String, String>>,
    /// Number of listing calls seen.
    list_calls: Mutex<usize>,
    /// URIs passed to read calls, in call order.
    read_calls: Mutex<Vec<String>>,
}

impl FakeMcpResourceProvider {
    /// Creates a provider with no resources and no failures.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a provider that lists `resources`.
    #[must_use]
    pub fn with_resources(resources: Vec<McpResource>) -> Self {
        let provider = Self::new();
        provider.set_resources(resources);
        provider
    }

    /// Replaces the resources returned by listing.
    pub fn set_resources(&self, resources: Vec<McpResource>) {
        *self.resources.lock() = resources;
    }

    /// Registers a canned read result, keyed by its `uri`.
    pub fn set_resource(&self, content: McpResourceContent) {
        self.contents.lock().insert(content.uri.clone(), content);
    }

    /// Makes [`McpResourceProvider::list_resources`] fail with `message`.
    pub fn fail_list(&self, message: impl Into<String>) {
        *self.list_failure.lock() = Some(message.into());
    }

    /// Makes reading `uri` fail with `message`.
    pub fn fail_read(&self, uri: &str, message: impl Into<String>) {
        self.read_failures
            .lock()
            .insert(uri.to_owned(), message.into());
    }

    /// Number of listing calls seen so far.
    #[must_use]
    pub fn list_calls(&self) -> usize {
        *self.list_calls.lock()
    }

    /// URIs passed to read calls so far, in call order.
    #[must_use]
    pub fn read_calls(&self) -> Vec<String> {
        self.read_calls.lock().clone()
    }
}

#[async_trait]
impl McpResourceProvider for FakeMcpResourceProvider {
    async fn list_resources(&self) -> Result<Vec<McpResource>> {
        *self.list_calls.lock() += 1;
        let failure = self.list_failure.lock().clone();
        if let Some(message) = failure {
            return Err(Error::tool(message));
        }
        Ok(self.resources.lock().clone())
    }

    async fn read_resource(&self, uri: &str) -> Result<McpResourceContent> {
        self.read_calls.lock().push(uri.to_owned());
        let failure = self.read_failures.lock().get(uri).cloned();
        if let Some(message) = failure {
            return Err(Error::tool(message));
        }
        self.contents
            .lock()
            .get(uri)
            .cloned()
            .ok_or_else(|| Error::tool(format!("unknown MCP resource: {uri}")))
    }
}

/// An [`McpResourceProvider`] over a live foundation [`McpClient`].
///
/// List and read are delegated to
/// [`McpClient::list_resources`]/[`McpClient::read_resource`] and normalised from the raw
/// `rmcp` JSON (`mimeType`, `contents[].blob`) to this crate's `snake_case` types. Resources
/// without a usable `uri` are skipped with a `debug` log, exactly like Python's
/// `getattr(r, "uri", str(r))` tolerance.
///
/// The adapter also implements [`McpResourceSubscriptionProvider`], but only through that
/// trait's defaults: the foundation client has no `resources/subscribe` call and no
/// notification callback for `resources/updated`, so subscribing answers
/// [`super::mcp_subscriptions::SUBSCRIBE_UNSUPPORTED`] instead of pretending. A transport that
/// gains the capability implements the trait directly.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use openjiuwen_foundation::tool::mcp::McpClient;
/// # use openjiuwen_harness::tools::McpClientResourceProvider;
/// # fn wire(client: Arc<dyn McpClient>) {
/// let provider = Arc::new(McpClientResourceProvider::new(client));
/// # let _ = provider;
/// # }
/// ```
pub struct McpClientResourceProvider {
    /// The live client the two resource calls are delegated to.
    client: Arc<dyn McpClient>,
}

impl std::fmt::Debug for McpClientResourceProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpClientResourceProvider")
            .field("server", &self.client.server_name())
            .field("server_id", &self.client.server_id())
            .finish()
    }
}

impl McpClientResourceProvider {
    /// Builds the adapter over `client`.
    #[must_use]
    pub fn new(client: Arc<dyn McpClient>) -> Self {
        Self { client }
    }

    /// The wrapped client.
    #[must_use]
    pub fn client(&self) -> &Arc<dyn McpClient> {
        &self.client
    }
}

#[async_trait]
impl McpResourceProvider for McpClientResourceProvider {
    async fn list_resources(&self) -> Result<Vec<McpResource>> {
        let values = self.client.list_resources().await?;
        Ok(values
            .iter()
            .filter_map(|value| {
                let resource = resource_from_wire(value);
                if resource.is_none() {
                    tracing::debug!(
                        target: "openjiuwen_harness::tools::mcp_resources",
                        "skipping an MCP resource without a usable uri"
                    );
                }
                resource
            })
            .collect())
    }

    async fn read_resource(&self, uri: &str) -> Result<McpResourceContent> {
        let value = self.client.read_resource(uri).await?;
        content_from_wire(uri, &value)
    }
}

#[async_trait]
impl McpResourceSubscriptionProvider for McpClientResourceProvider {}

/// Normalises one `rmcp` resource (`{"uri", "name", "mimeType", "description", …}`) — or a bare
/// URI string, as Python tolerates — into [`McpResource`].
fn resource_from_wire(value: &Value) -> Option<McpResource> {
    if let Some(uri) = value.as_str() {
        let uri = uri.trim();
        return (!uri.is_empty()).then(|| McpResource {
            uri: uri.to_owned(),
            ..McpResource::default()
        });
    }
    let object = value.as_object()?;
    let uri = wire_string(object, &["uri"])?;
    Some(McpResource {
        uri,
        name: wire_string(object, &["name"]).unwrap_or_default(),
        mime_type: wire_string(object, &["mimeType", "mime_type"]),
        description: wire_string(object, &["description"]),
    })
}

/// Normalises one `rmcp` `ReadResourceResult` (`{"contents": [...]}`) into
/// [`McpResourceContent`].
///
/// A payload without a `contents` array is treated as a bare content object so a custom
/// transport can answer with either shape.
fn content_from_wire(requested: &str, value: &Value) -> Result<McpResourceContent> {
    let entry = match value.get("contents") {
        Some(Value::Array(entries)) => {
            let Some(first) = entries.first() else {
                return Err(Error::not_found(format!(
                    "MCP resource `{requested}` has no content"
                )));
            };
            if entries.len() > 1 {
                tracing::debug!(
                    target: "openjiuwen_harness::tools::mcp_resources",
                    uri = requested,
                    contents = entries.len(),
                    "MCP resource returned several contents; keeping the first"
                );
            }
            first
        }
        Some(_) => {
            return Err(Error::mcp(
                "MCP read_resource returned a non-array 'contents' field",
            ))
        }
        None => value,
    };
    let object = entry
        .as_object()
        .ok_or_else(|| Error::mcp("MCP read_resource returned a non-object content"))?;
    let text = wire_string(object, &["text"]);
    let blob_base64 = wire_string(object, &["blob", "blob_base64", "blobBase64"]);
    let byte_size = text.as_ref().map_or_else(
        || blob_base64.as_deref().map_or(0, base64_byte_len),
        |text| u64::try_from(text.len()).unwrap_or(u64::MAX),
    );
    Ok(McpResourceContent {
        uri: wire_string(object, &["uri"]).unwrap_or_else(|| requested.to_owned()),
        mime_type: wire_string(object, &["mimeType", "mime_type"]),
        text,
        blob_base64,
        byte_size,
    })
}

/// Returns the first non-blank string among `keys`.
fn wire_string(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        let value = object.get(*key)?.as_str()?.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

/// The decoded size of a base64 payload, without decoding it.
fn base64_byte_len(encoded: &str) -> u64 {
    let trimmed = encoded.trim_end();
    let padding = trimmed
        .chars()
        .rev()
        .take_while(|character| *character == '=')
        .count();
    let length = u64::try_from(trimmed.chars().count()).unwrap_or(u64::MAX);
    let padding = u64::try_from(padding).unwrap_or(0);
    (length.saturating_mul(3) / 4).saturating_sub(padding)
}

/// Lists the resources of the bound MCP server (`list_mcp_resources`).
///
/// The tool never returns [`Err`] for an expected failure: a missing provider, a malformed
/// argument object and a provider error all become a `success: false` payload.
pub struct ListMcpResourcesTool {
    /// The provider, when one is wired in.
    provider: Option<Arc<dyn McpResourceProvider>>,
    /// The model-facing card, built once from the description language.
    card: ToolCard,
}

impl ListMcpResourcesTool {
    /// Builds the tool over `provider` with the description language `language`.
    ///
    /// A `None` provider is legal: [`Tool::invoke`] then reports the "not configured"
    /// failure instead of silently returning an empty resource list.
    #[must_use]
    pub fn new(
        provider: Option<Arc<dyn McpResourceProvider>>,
        language: impl Into<String>,
    ) -> Self {
        let language = language.into();
        Self {
            provider,
            card: Self::card_for(&language),
        }
    }

    /// Builds the model-facing card for the description language `language`.
    ///
    /// The description text comes from [`super::descriptions`]; until that table gains a
    /// `list_mcp_resources` row the card carries an empty description.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            LIST_TOOL_NAME,
            description(LIST_TOOL_NAME, language),
            json!({"type": "object", "properties": {}}),
            true,
        )
    }
}

#[async_trait]
impl Tool for ListMcpResourcesTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        // `list_mcp_resources` takes no arguments; this only rejects a non-object argument
        // payload, which is a tool-call protocol violation rather than a tool failure.
        if let Err(payload) = parse_args::<Value>(&inputs) {
            return Ok(payload.to_json());
        }
        let Some(provider) = self.provider.clone() else {
            return Ok(ToolPayload::fail(NO_PROVIDER_MESSAGE).to_json());
        };
        match provider.list_resources().await {
            Ok(resources) => Ok(ToolPayload::ok(json!({"resources": resources})).to_json()),
            Err(error) => Ok(ToolPayload::fail(error.message().into_owned()).to_json()),
        }
    }
}

/// Arguments of the `read_mcp_resource` tool.
///
/// Python parity: the `uri` parameter of the Python schema; the `server_id` parameter is not
/// present because the provider is already bound to one server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReadMcpResourceArgs {
    /// URI of the resource to read.
    pub uri: String,
}

/// Reads one resource of the bound MCP server (`read_mcp_resource`).
///
/// The tool never returns [`Err`] for an expected failure: a missing provider, a missing
/// `uri`, a malformed argument object and a provider error all become a `success: false`
/// payload.
pub struct ReadMcpResourceTool {
    /// The provider, when one is wired in.
    provider: Option<Arc<dyn McpResourceProvider>>,
    /// The model-facing card, built once from the description language.
    card: ToolCard,
}

impl ReadMcpResourceTool {
    /// Builds the tool over `provider` with the description language `language`.
    ///
    /// A `None` provider is legal: [`Tool::invoke`] then reports the "not configured"
    /// failure instead of silently returning empty content.
    #[must_use]
    pub fn new(
        provider: Option<Arc<dyn McpResourceProvider>>,
        language: impl Into<String>,
    ) -> Self {
        let language = language.into();
        Self {
            provider,
            card: Self::card_for(&language),
        }
    }

    /// Builds the model-facing card for the description language `language`.
    ///
    /// The description text comes from [`super::descriptions`]; until that table gains a
    /// `read_mcp_resource` row the card carries an empty description.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            READ_TOOL_NAME,
            description(READ_TOOL_NAME, language),
            schema_for::<ReadMcpResourceArgs>(),
            true,
        )
    }
}

#[async_trait]
impl Tool for ReadMcpResourceTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        // Python checks the field before touching the server, and reports the same message
        // for an absent and an empty `uri`; do that before serde so the message matches.
        if is_blank(&inputs, "uri") {
            return Ok(ToolPayload::fail("uri is required").to_json());
        }
        let args = match parse_args::<ReadMcpResourceArgs>(&inputs) {
            Ok(args) => args,
            Err(payload) => return Ok(payload.to_json()),
        };
        let Some(provider) = self.provider.clone() else {
            return Ok(ToolPayload::fail(NO_PROVIDER_MESSAGE).to_json());
        };
        match provider.read_resource(&args.uri).await {
            Ok(content) => Ok(ToolPayload::ok(json!({"resource": content})).to_json()),
            Err(error) => Ok(ToolPayload::fail(error.message().into_owned()).to_json()),
        }
    }
}

/// Whether the raw `field` is absent, `null`, or an empty string.
fn is_blank(inputs: &Value, field: &str) -> bool {
    match inputs.get(field) {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.trim().is_empty(),
        Some(_) => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A resource with the given uri and name.
    fn resource(uri: &str, name: &str) -> McpResource {
        McpResource {
            uri: uri.to_owned(),
            name: name.to_owned(),
            mime_type: Some("text/markdown".to_owned()),
            description: Some(format!("{name} description")),
        }
    }

    /// Wires a fake provider into the tools as the trait object they accept.
    fn wired(fake: &Arc<FakeMcpResourceProvider>) -> Arc<dyn McpResourceProvider> {
        fake.clone()
    }
    /// Runs `list_mcp_resources` over `provider` and returns the payload JSON.
    async fn list(provider: Option<Arc<dyn McpResourceProvider>>) -> Value {
        let tool = ListMcpResourcesTool::new(provider, "en");
        tool.invoke(json!({}), &ToolContext::new())
            .await
            .expect("list_mcp_resources never fails for expected input")
    }

    /// Runs `read_mcp_resource` over `provider` with `inputs`.
    async fn read(provider: Option<Arc<dyn McpResourceProvider>>, inputs: Value) -> Value {
        let tool = ReadMcpResourceTool::new(provider, "en");
        tool.invoke(inputs, &ToolContext::new())
            .await
            .expect("read_mcp_resource never fails for expected input")
    }

    #[tokio::test]
    async fn list_without_a_provider_is_a_typed_failure() {
        let payload = list(None).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], NO_PROVIDER_MESSAGE);
        assert_eq!(payload["data"], Value::Null);
    }

    #[tokio::test]
    async fn read_without_a_provider_is_a_typed_failure() {
        let payload = read(None, json!({"uri": "mem://notes"})).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], NO_PROVIDER_MESSAGE);
    }

    #[tokio::test]
    async fn uri_validation_precedes_the_provider_check() {
        // Python validates `uri` first; matching that order keeps the message stable for both
        // unwired and malformed callers.
        let payload = read(None, json!({})).await;
        assert_eq!(payload["error"], "uri is required");
    }

    #[tokio::test]
    async fn list_round_trip_returns_the_canned_resources() {
        let provider = Arc::new(FakeMcpResourceProvider::with_resources(vec![
            resource("mem://a", "alpha"),
            resource("mem://b", "beta"),
        ]));

        let payload = list(Some(wired(&provider))).await;
        assert_eq!(payload["success"], true, "payload: {payload}");
        let resources = payload["data"]["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 2);
        assert_eq!(resources[0]["uri"], "mem://a");
        assert_eq!(resources[0]["name"], "alpha");
        assert_eq!(resources[0]["mime_type"], "text/markdown");
        assert_eq!(resources[0]["description"], "alpha description");
        assert_eq!(provider.list_calls(), 1);
    }

    #[tokio::test]
    async fn list_accepts_and_ignores_extra_arguments() {
        let provider = Arc::new(FakeMcpResourceProvider::new());
        let tool = ListMcpResourcesTool::new(Some(wired(&provider)), "en");
        let payload = tool
            .invoke(json!({"server_id": "python-style"}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(payload["success"], true);
        assert_eq!(payload["data"]["resources"], json!([]));
    }

    #[tokio::test]
    async fn read_round_trip_returns_the_canned_content() {
        let provider = Arc::new(FakeMcpResourceProvider::new());
        provider.set_resource(McpResourceContent {
            uri: "mem://a".to_owned(),
            mime_type: Some("text/plain".to_owned()),
            text: Some("hello".to_owned()),
            blob_base64: None,
            byte_size: 5,
        });

        let payload = read(Some(wired(&provider)), json!({"uri": "mem://a"})).await;
        assert_eq!(payload["success"], true, "payload: {payload}");
        let content = &payload["data"]["resource"];
        assert_eq!(content["uri"], "mem://a");
        assert_eq!(content["mime_type"], "text/plain");
        assert_eq!(content["text"], "hello");
        assert_eq!(content["blob_base64"], Value::Null);
        assert_eq!(content["byte_size"], 5);
        assert_eq!(provider.read_calls(), vec!["mem://a".to_owned()]);
    }

    #[tokio::test]
    async fn unknown_uri_becomes_a_failure_payload_carrying_the_provider_message() {
        let provider = Arc::new(FakeMcpResourceProvider::new());
        let payload = read(Some(wired(&provider)), json!({"uri": "mem://missing"})).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], "unknown MCP resource: mem://missing");
    }

    #[tokio::test]
    async fn missing_uri_argument_is_a_failure_payload() {
        let provider = Arc::new(FakeMcpResourceProvider::new());

        let absent = read(Some(wired(&provider)), json!({})).await;
        assert_eq!(absent["success"], false);
        assert_eq!(absent["error"], "uri is required");

        let blank = read(Some(wired(&provider)), json!({"uri": "   "})).await;
        assert_eq!(blank["error"], "uri is required");

        let wrong_type = read(Some(wired(&provider)), json!({"uri": 7})).await;
        assert_eq!(wrong_type["success"], false);
        assert!(
            wrong_type["error"]
                .as_str()
                .unwrap()
                .starts_with("invalid tool arguments: "),
            "payload: {wrong_type}"
        );
    }

    #[tokio::test]
    async fn provider_list_failure_becomes_a_failure_payload() {
        let provider = Arc::new(FakeMcpResourceProvider::new());
        provider.fail_list("server unavailable");
        let payload = list(Some(wired(&provider))).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], "server unavailable");
    }

    #[tokio::test]
    async fn provider_read_failure_is_injected_per_uri() {
        let provider = Arc::new(FakeMcpResourceProvider::new());
        provider.fail_read("mem://a", "permission denied");

        let failed = read(Some(wired(&provider)), json!({"uri": "mem://a"})).await;
        assert_eq!(failed["success"], false);
        assert_eq!(failed["error"], "permission denied");

        provider.set_resource(McpResourceContent {
            uri: "mem://b".to_owned(),
            text: Some("ok".to_owned()),
            ..McpResourceContent::default()
        });
        let succeeded = read(Some(wired(&provider)), json!({"uri": "mem://b"})).await;
        assert_eq!(succeeded["success"], true);
        assert_eq!(succeeded["data"]["resource"]["text"], "ok");
    }

    #[tokio::test]
    async fn non_object_arguments_are_a_protocol_failure() {
        let provider = Arc::new(FakeMcpResourceProvider::new());
        let tool = ListMcpResourcesTool::new(Some(wired(&provider)), "en");
        let payload = tool.invoke(json!([]), &ToolContext::new()).await.unwrap();
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], "tool arguments must be an object");
    }

    #[test]
    fn cards_declare_the_python_names_and_schemas() {
        let list_card = ListMcpResourcesTool::card_for("en");
        assert_eq!(list_card.name, "list_mcp_resources");
        assert!(list_card.parallel_safe);
        assert_eq!(list_card.input_params["type"], "object");
        assert_eq!(
            list_card.description,
            description("list_mcp_resources", "en")
        );

        let read_card = ReadMcpResourceTool::card_for("cn");
        assert_eq!(read_card.name, "read_mcp_resource");
        assert!(read_card.parallel_safe);
        assert_eq!(read_card.input_params["required"], json!(["uri"]));
        assert_eq!(
            read_card.input_params["properties"]["uri"]["type"],
            "string"
        );
        assert_eq!(
            read_card.description,
            description("read_mcp_resource", "cn")
        );
    }

    #[test]
    fn resources_round_trip_through_json_and_tolerate_unknown_fields() {
        let resource: McpResource = serde_json::from_value(json!({
            "uri": "mem://a",
            "name": "alpha",
            "mime_type": "text/plain",
            "description": "notes",
            "unknown_field": {"nested": true},
        }))
        .unwrap();
        assert_eq!(resource.uri, "mem://a");
        assert_eq!(resource.mime_type.as_deref(), Some("text/plain"));
        assert_eq!(resource.description.as_deref(), Some("notes"));

        let minimal: McpResource = serde_json::from_value(json!({"uri": "mem://b"})).unwrap();
        assert_eq!(minimal.name, "");
        assert_eq!(minimal.mime_type, None);
        assert_eq!(minimal.description, None);

        let content = McpResourceContent {
            uri: "mem://a".to_owned(),
            text: Some("hello".to_owned()),
            ..McpResourceContent::default()
        };
        let rendered = serde_json::to_value(&content).unwrap();
        assert_eq!(rendered["uri"], "mem://a");
        assert_eq!(rendered["text"], "hello");
        assert_eq!(rendered["byte_size"], 0);
        assert_eq!(rendered["blob_base64"], Value::Null);
    }

    #[test]
    fn fake_provider_records_calls_and_exposes_its_state() {
        let provider = FakeMcpResourceProvider::new();
        assert_eq!(provider.list_calls(), 0);
        assert_eq!(provider.read_calls(), Vec::<String>::new());
        provider.set_resources(vec![resource("mem://a", "alpha")]);
        assert_eq!(provider.resources.lock().len(), 1);
        assert_eq!(FakeMcpResourceProvider::default().list_calls(), 0);
    }

    /// A minimal live-client stand-in: canned `rmcp` payloads and one canned failure.
    #[derive(Debug, Default)]
    struct StubClient {
        /// Payload of `list_resources`; `None` answers with a transport failure.
        resources: Option<Vec<Value>>,
        /// Payload of `read_resource`, when it succeeds.
        read: Option<Value>,
        /// Failure `read_resource` reports instead of `read`.
        read_failure: Option<String>,
        /// URIs passed to `read_resource`.
        read_calls: Mutex<Vec<String>>,
    }

    impl StubClient {
        /// A client answering with `resources` and `read`.
        fn new(resources: Vec<Value>, read: Value) -> Arc<Self> {
            Arc::new(Self {
                resources: Some(resources),
                read: Some(read),
                read_failure: None,
                read_calls: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl McpClient for StubClient {
        fn server_name(&self) -> &'static str {
            "stub"
        }

        async fn connect(&self) -> Result<()> {
            Ok(())
        }

        async fn disconnect(&self) -> Result<()> {
            Ok(())
        }

        async fn is_connected(&self) -> bool {
            true
        }

        async fn list_tools(&self) -> Result<Vec<openjiuwen_foundation::tool::mcp::McpRemoteTool>> {
            Err(Error::unsupported("the stub client exposes no tools"))
        }

        async fn call_tool_with(
            &self,
            _tool_name: &str,
            _arguments: Value,
            _timeout: Option<std::time::Duration>,
            _include_image_content: bool,
        ) -> Result<openjiuwen_foundation::tool::mcp::McpToolResult> {
            Err(Error::unsupported("the stub client calls no tools"))
        }

        async fn list_resources(&self) -> Result<Vec<Value>> {
            self.resources
                .clone()
                .ok_or_else(|| Error::transport("socket closed"))
        }

        async fn read_resource(&self, uri: &str) -> Result<Value> {
            self.read_calls.lock().push(uri.to_owned());
            match (self.read.clone(), self.read_failure.as_deref()) {
                (_, Some(message)) => Err(Error::tool(message.to_owned())),
                (Some(value), None) => Ok(value),
                (None, None) => Err(Error::transport("socket closed")),
            }
        }
    }

    #[tokio::test]
    async fn the_live_client_adapter_normalises_list_and_read() {
        let client = StubClient::new(
            vec![
                json!({
                    "uri": "mem://a",
                    "name": "alpha",
                    "mimeType": "text/markdown",
                    "description": "notes",
                    "size": 42,
                }),
                json!("mem://bare"),
                json!({"name": "no uri"}),
            ],
            json!({"contents": [{"uri": "mem://a", "mimeType": "text/markdown", "text": "hello"}]}),
        );
        let provider: Arc<dyn McpResourceProvider> =
            Arc::new(McpClientResourceProvider::new(client.clone()));

        let listed = list(Some(provider.clone())).await;
        assert_eq!(listed["success"], true, "payload: {listed}");
        let resources = listed["data"]["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 2);
        assert_eq!(resources[0]["uri"], "mem://a");
        assert_eq!(resources[0]["name"], "alpha");
        assert_eq!(resources[0]["mime_type"], "text/markdown");
        assert_eq!(resources[1]["uri"], "mem://bare");
        assert_eq!(resources[1]["name"], "");

        let read = read(Some(provider), json!({"uri": "mem://a"})).await;
        assert_eq!(read["success"], true, "payload: {read}");
        assert_eq!(read["data"]["resource"]["uri"], "mem://a");
        assert_eq!(read["data"]["resource"]["text"], "hello");
        assert_eq!(read["data"]["resource"]["byte_size"], 5);
        assert_eq!(read["data"]["resource"]["blob_base64"], Value::Null);
        assert_eq!(client.read_calls.lock().clone(), vec!["mem://a".to_owned()]);
        assert_eq!(
            McpClientResourceProvider::new(client)
                .client()
                .server_name(),
            "stub"
        );
    }

    #[tokio::test]
    async fn the_live_client_adapter_reports_empty_and_malformed_reads() {
        let empty = Arc::new(McpClientResourceProvider::new(StubClient::new(
            Vec::new(),
            json!({"contents": []}),
        )));
        let error = empty.read_resource("mem://a").await.unwrap_err();
        assert!(matches!(error, Error::NotFound(_)), "error: {error}");
        assert_eq!(empty.list_resources().await.unwrap().len(), 0);

        let binary = Arc::new(McpClientResourceProvider::new(StubClient::new(
            Vec::new(),
            json!({"contents": [{"uri": "mem://b", "blob": "aGVsbG8="}]}),
        )));
        let content = binary.read_resource("mem://b").await.unwrap();
        assert_eq!(content.blob_base64.as_deref(), Some("aGVsbG8="));
        assert_eq!(content.text, None);
        assert_eq!(content.byte_size, 5);

        let malformed = Arc::new(McpClientResourceProvider::new(StubClient::new(
            Vec::new(),
            json!({"contents": "nope"}),
        )));
        let error = malformed.read_resource("mem://c").await.unwrap_err();
        assert!(matches!(error, Error::Mcp(_)), "error: {error}");

        let failing = Arc::new(McpClientResourceProvider::new(Arc::new(StubClient {
            resources: None,
            read: Some(json!({})),
            read_failure: None,
            read_calls: Mutex::new(Vec::new()),
        })));
        let error = failing.list_resources().await.unwrap_err();
        assert!(matches!(error, Error::Transport(_)), "error: {error}");
    }

    #[tokio::test]
    async fn the_live_client_adapter_cannot_subscribe() {
        let provider = McpClientResourceProvider::new(StubClient::new(Vec::new(), json!({})));
        assert!(!provider.supports_subscriptions());
        let registry = crate::tools::mcp_subscriptions::McpResourceSubscriptions::new()
            .with_provider(Arc::new(provider));
        let error = registry.subscribe("mem://a").await.unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "error: {error}");
        assert!(
            error.message().contains("resources/subscribe"),
            "error: {error}"
        );
        assert!(!registry.is_subscribed("mem://a"));
    }

    #[test]
    fn base64_sizes_are_computed_without_decoding() {
        assert_eq!(base64_byte_len("aGVsbG8="), 5);
        assert_eq!(base64_byte_len("aGVsbG8h"), 6);
        assert_eq!(base64_byte_len(""), 0);
    }
}
