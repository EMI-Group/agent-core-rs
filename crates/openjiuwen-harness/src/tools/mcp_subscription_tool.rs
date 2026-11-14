//! The model-facing `mcp_resource_subscription` tool.
//!
//! One tool with four actions over a [`McpResourceSubscriptions`] registry, mirroring the
//! single-entry-point shape of the Python `cron`/`worktree` tools:
//!
//! - `subscribe` — `uri` → the subscription snapshot (capacity, buffer occupancy, counters),
//! - `unsubscribe` — `uri` → the closed subscription snapshot,
//! - `list` — every open subscription, ordered by URI,
//! - `poll` — `uri` plus an optional `limit` → the buffered update events, oldest first.
//!
//! The tool never returns [`Err`] for an expected failure: a missing registry, a blank or
//! unsupported URI, a provider without `resources/subscribe` support and a provider failure all
//! become `success: false` payloads carrying the typed reason. Only cancellation (through
//! `ToolContext::check_cancelled`) produces an [`Err`](openjiuwen_common::error::Error).
//!
//! # Why a Rust-only tool
//!
//! Python's `mcp_tools.py` exposes only `list_mcp_resources`/`read_mcp_resource`; the resource
//! manager has no subscription surface. This tool is the model-facing half of the Rust
//! subscription registry and is documented as an intentional extension, not a port.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use crate::tools::descriptions;
use crate::tools::mcp_subscriptions::McpResourceSubscriptions;
use crate::tools::support::{parse_args, schema_for, tool_card, ToolPayload};

/// The model-facing tool name.
pub const TOOL_NAME: &str = "mcp_resource_subscription";

/// Number of events `poll` drains when the call does not ask for a count.
pub const DEFAULT_POLL_LIMIT: usize = 10;

/// Upper bound of the `poll` event count.
pub const MAX_POLL_LIMIT: usize = 50;

/// The failure reported when the tool was built without a registry.
const NO_REGISTRY_MESSAGE: &str =
    "mcp resource subscription is not available: no subscription registry is configured \
     (ToolRegistry::with_mcp_subscriptions)";

/// The action of one `mcp_resource_subscription` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptionAction {
    /// Open a subscription for one resource URI.
    Subscribe,
    /// Close the subscription of one resource URI.
    Unsubscribe,
    /// List the open subscriptions.
    List,
    /// Drain the buffered update events of one subscription.
    Poll,
}

impl SubscriptionAction {
    /// The action name as the model writes it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Subscribe => "subscribe",
            Self::Unsubscribe => "unsubscribe",
            Self::List => "list",
            Self::Poll => "poll",
        }
    }

    /// Parses an action name, accepting surrounding whitespace only.
    #[must_use]
    pub fn parse(action: &str) -> Option<Self> {
        match action.trim() {
            "subscribe" => Some(Self::Subscribe),
            "unsubscribe" => Some(Self::Unsubscribe),
            "list" => Some(Self::List),
            "poll" => Some(Self::Poll),
            _ => None,
        }
    }
}

/// Arguments of the `mcp_resource_subscription` tool.
///
/// One flat object for all four actions: `uri` belongs to `subscribe`/`unsubscribe`/`poll` and
/// `limit` only to `poll`. Every field is optional at the serde level so the tool reports its own
/// validation text instead of a deserialization error; `action` is the only required field in the
/// card schema.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct McpResourceSubscriptionArgs {
    /// The action to perform: `subscribe`, `unsubscribe`, `list` or `poll`.
    pub action: String,
    /// URI of the resource, as `list_mcp_resources` returned it.
    pub uri: String,
    /// Maximum number of events `poll` returns (clamped to 1..=50, default 10).
    #[serde(default)]
    #[schemars(with = "u64")]
    pub limit: Option<u64>,
}

/// Opens, closes, lists and drains MCP resource subscriptions.
pub struct McpResourceSubscriptionTool {
    /// The registry; without one every action reports a failure payload.
    registry: Option<Arc<McpResourceSubscriptions>>,
    /// The model-facing card built from the description language.
    card: ToolCard,
}

impl McpResourceSubscriptionTool {
    /// Builds the tool over `registry` with the card description language `language`.
    ///
    /// A `None` registry is legal: [`Tool::invoke`] then reports the "not available" failure
    /// instead of pretending the action succeeded.
    #[must_use]
    pub fn new(registry: Option<Arc<McpResourceSubscriptions>>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            registry,
        }
    }

    /// Builds the model-facing card for the description language `language`.
    ///
    /// The description text comes from [`crate::tools::descriptions`]; the per-property hints are
    /// inlined because they are action-shaped rather than a copy of a Python prompt constant.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<McpResourceSubscriptionArgs>();
        install_input_params(&mut input_params, language);
        tool_card(
            TOOL_NAME,
            descriptions::description(TOOL_NAME, language),
            input_params,
            false,
        )
    }

    /// The registry backing the tool, when one is wired.
    #[must_use]
    pub fn registry(&self) -> Option<&Arc<McpResourceSubscriptions>> {
        self.registry.as_ref()
    }

    /// Runs one action and renders the payload.
    async fn run(&self, inputs: &Value) -> ToolPayload {
        let Some(registry) = self.registry.as_ref() else {
            return ToolPayload::fail(NO_REGISTRY_MESSAGE);
        };
        let args: McpResourceSubscriptionArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.action.trim().is_empty() {
            return ToolPayload::fail(
                "action is required (expected subscribe, unsubscribe, list or poll)",
            );
        }
        match SubscriptionAction::parse(&args.action) {
            Some(SubscriptionAction::Subscribe) => subscribe(registry, args.uri.trim()).await,
            Some(SubscriptionAction::Unsubscribe) => unsubscribe(registry, args.uri.trim()).await,
            Some(SubscriptionAction::List) => {
                let subscriptions = registry.list();
                ToolPayload::ok(json!({
                    "subscriptions": subscriptions,
                    "count": subscriptions.len(),
                    "capacity": registry.capacity(),
                }))
            }
            Some(SubscriptionAction::Poll) => {
                poll(registry, args.uri.trim(), resolve_limit(args.limit))
            }
            None => ToolPayload::fail(format!(
                "invalid action: {:?} (expected subscribe, unsubscribe, list or poll)",
                args.action
            )),
        }
    }
}

#[async_trait]
impl Tool for McpResourceSubscriptionTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.run(&inputs).await.to_json())
    }
}

/// Handles `action=subscribe`.
async fn subscribe(registry: &McpResourceSubscriptions, uri: &str) -> ToolPayload {
    if uri.is_empty() {
        return ToolPayload::fail("uri is required for action=subscribe");
    }
    match registry.subscribe(uri).await {
        Ok(info) => ToolPayload::ok(json!({
            "subscription": info,
            "message": format!("subscribed to {uri}"),
        })),
        Err(error) => failure(&error),
    }
}

/// Handles `action=unsubscribe`.
async fn unsubscribe(registry: &McpResourceSubscriptions, uri: &str) -> ToolPayload {
    if uri.is_empty() {
        return ToolPayload::fail("uri is required for action=unsubscribe");
    }
    match registry.unsubscribe(uri).await {
        Ok(info) => ToolPayload::ok(json!({
            "subscription": info,
            "message": format!("unsubscribed from {uri}"),
        })),
        Err(error) => failure(&error),
    }
}

/// Handles `action=poll`.
fn poll(registry: &McpResourceSubscriptions, uri: &str, limit: usize) -> ToolPayload {
    if uri.is_empty() {
        return ToolPayload::fail("uri is required for action=poll");
    }
    match registry.poll(uri, limit) {
        Ok(events) => ToolPayload::ok(json!({
            "uri": uri,
            "events": events,
            "count": events.len(),
            "subscription": registry.info(uri).ok(),
        })),
        Err(error) => failure(&error),
    }
}

/// Renders a registry error: an [`Error::Unsupported`] payload carries its bare reason so the
/// configuration guidance stays readable, everything else keeps the crate's `kind: message` form.
fn failure(error: &Error) -> ToolPayload {
    match error {
        Error::Unsupported(message) => ToolPayload::fail(message.clone()),
        other => ToolPayload::fail(other.message().into_owned()),
    }
}

/// Clamps the requested `poll` count to `1..=MAX_POLL_LIMIT`.
fn resolve_limit(value: Option<u64>) -> usize {
    value.map_or(DEFAULT_POLL_LIMIT, |value| {
        usize::try_from(value)
            .unwrap_or(DEFAULT_POLL_LIMIT)
            .clamp(1, MAX_POLL_LIMIT)
    })
}

/// Installs the action-shaped property hints the generated schema cannot express.
fn install_input_params(input_params: &mut Value, language: &str) {
    let english = crate::tools::is_english(language);
    let Some(properties) = input_params.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    let limit = if english {
        format!(
            "Maximum number of buffered update events `poll` returns (clamped to 1..={MAX_POLL_LIMIT}, default {DEFAULT_POLL_LIMIT})."
        )
    } else {
        format!(
            "`poll` 返回的缓冲更新事件数量上限（限制在 1..={MAX_POLL_LIMIT}，默认 {DEFAULT_POLL_LIMIT}）。"
        )
    };
    let uri = if english {
        "Resource URI exactly as `list_mcp_resources` returned it."
    } else {
        "资源 URI，必须与 `list_mcp_resources` 返回的值完全一致。"
    };
    if let Some(schema) = properties.get_mut("uri").and_then(Value::as_object_mut) {
        schema.insert("description".to_owned(), Value::String(uri.to_owned()));
    }
    if let Some(schema) = properties.get_mut("limit").and_then(Value::as_object_mut) {
        schema.insert("description".to_owned(), Value::String(limit));
    }
    properties.insert(
        "action".to_owned(),
        json!({
            "type": "string",
            "enum": ["subscribe", "unsubscribe", "list", "poll"],
            "description": if english {
                "Action to perform: subscribe, unsubscribe, list or poll."
            } else {
                "要执行的操作：subscribe、unsubscribe、list 或 poll。"
            },
        }),
    );
    if let Some(object) = input_params.as_object_mut() {
        object.insert("required".to_owned(), json!(["action"]));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::tools::mcp_resources::McpResourceContent;
    use crate::tools::mcp_subscriptions::{
        FakeMcpResourceSubscriptionProvider, DEFAULT_BUFFER_CAPACITY,
    };

    /// Invokes the tool with a fresh context and unwraps the payload envelope.
    async fn invoke(tool: &McpResourceSubscriptionTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    /// A registry with the default capacity over `provider`.
    fn registry(
        provider: Arc<FakeMcpResourceSubscriptionProvider>,
    ) -> Arc<McpResourceSubscriptions> {
        Arc::new(McpResourceSubscriptions::new().with_provider(provider))
    }

    #[test]
    fn card_declares_the_action_and_arguments() {
        let card = McpResourceSubscriptionTool::card_for("en");
        assert_eq!(card.name, TOOL_NAME);
        assert!(!card.parallel_safe);
        assert_eq!(card.input_params["required"], json!(["action"]));
        assert_eq!(
            card.description,
            descriptions::description(TOOL_NAME, "en")
        );
        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["action", "limit", "uri"]);
        assert!(!properties["limit"]["description"]
            .as_str()
            .unwrap()
            .is_empty());

        let chinese = McpResourceSubscriptionTool::card_for("cn");
        assert_eq!(chinese.description, descriptions::description(TOOL_NAME, "cn"));
        assert!(chinese.input_params["properties"]["uri"]["description"]
            .as_str()
            .unwrap()
            .starts_with("资源 URI"));
    }

    #[test]
    fn actions_parse_and_render() {
        for action in [
            SubscriptionAction::Subscribe,
            SubscriptionAction::Unsubscribe,
            SubscriptionAction::List,
            SubscriptionAction::Poll,
        ] {
            assert_eq!(SubscriptionAction::parse(action.as_str()), Some(action));
        }
        assert_eq!(SubscriptionAction::parse(" poll "), Some(SubscriptionAction::Poll));
        assert_eq!(SubscriptionAction::parse("get"), None);
    }

    #[test]
    fn poll_limits_are_clamped() {
        assert_eq!(resolve_limit(None), DEFAULT_POLL_LIMIT);
        assert_eq!(resolve_limit(Some(0)), 1);
        assert_eq!(resolve_limit(Some(7)), 7);
        assert_eq!(resolve_limit(Some(9_999)), MAX_POLL_LIMIT);
    }

    #[tokio::test]
    async fn subscribe_list_poll_and_unsubscribe_round_trip() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let tool = McpResourceSubscriptionTool::new(Some(registry(provider.clone())), "en");

        let subscribed = invoke(&tool, json!({"action": "subscribe", "uri": "mem://a"})).await;
        assert_eq!(subscribed["success"], json!(true));
        assert_eq!(subscribed["data"]["message"], json!("subscribed to mem://a"));
        assert_eq!(subscribed["data"]["subscription"]["uri"], "mem://a");
        assert_eq!(
            subscribed["data"]["subscription"]["capacity"],
            json!(DEFAULT_BUFFER_CAPACITY)
        );

        assert!(provider.push_content(
            "mem://a",
            McpResourceContent {
                uri: "mem://a".to_owned(),
                text: Some("fresh".to_owned()),
                ..McpResourceContent::default()
            }
        ));

        let listed = invoke(&tool, json!({"action": "list"})).await;
        assert_eq!(listed["data"]["count"], json!(1));
        assert_eq!(listed["data"]["subscriptions"][0]["received"], json!(1));

        let polled = invoke(&tool, json!({"action": "poll", "uri": "mem://a"})).await;
        assert_eq!(polled["success"], json!(true));
        assert_eq!(polled["data"]["count"], json!(1));
        assert_eq!(polled["data"]["events"][0]["content"]["text"], "fresh");
        assert_eq!(polled["data"]["subscription"]["buffered"], json!(0));

        let closed = invoke(&tool, json!({"action": "unsubscribe", "uri": "mem://a"})).await;
        assert_eq!(closed["success"], json!(true));
        assert_eq!(
            closed["data"]["message"],
            json!("unsubscribed from mem://a")
        );
        assert_eq!(
            invoke(&tool, json!({"action": "list"})).await["data"]["count"],
            json!(0)
        );
    }

    #[tokio::test]
    async fn poll_clamps_the_limit_and_returns_oldest_first() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = Arc::new(
            McpResourceSubscriptions::new()
                .with_provider(provider.clone())
                .with_capacity(8),
        );
        let tool = McpResourceSubscriptionTool::new(Some(subscriptions), "en");
        invoke(&tool, json!({"action": "subscribe", "uri": "mem://a"})).await;
        for text in ["one", "two", "three"] {
            assert!(provider.push_notification("mem://a", text));
        }

        let polled = invoke(&tool, json!({"action": "poll", "uri": "mem://a", "limit": 2})).await;
        assert_eq!(polled["data"]["count"], json!(2));
        assert_eq!(polled["data"]["events"][0]["reason"], "one");
        assert_eq!(polled["data"]["events"][1]["reason"], "two");

        let polled = invoke(&tool, json!({"action": "poll", "uri": "mem://a"})).await;
        assert_eq!(polled["data"]["count"], json!(1));
    }

    #[tokio::test]
    async fn failures_are_typed_payloads() {
        let bare = McpResourceSubscriptionTool::new(None, "en");
        let payload = invoke(&bare, json!({"action": "list"})).await;
        assert_eq!(payload["success"], json!(false));
        assert_eq!(
            payload["error"],
            json!(NO_REGISTRY_MESSAGE)
        );

        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::unsupported());
        let tool = McpResourceSubscriptionTool::new(Some(registry(provider)), "en");
        let payload = invoke(&tool, json!({"action": "subscribe", "uri": "mem://a"})).await;
        assert_eq!(payload["success"], json!(false));
        let error = payload["error"].as_str().unwrap();
        assert!(error.contains("resources/subscribe"), "error: {error}");

        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        provider.fail_subscribe("mem://bad", "not subscribable");
        let tool = McpResourceSubscriptionTool::new(Some(registry(provider)), "en");
        let payload = invoke(&tool, json!({"action": "subscribe", "uri": "mem://bad"})).await;
        assert_eq!(payload["error"], json!("not subscribable"));
    }

    #[tokio::test]
    async fn argument_validation_is_reported_per_action() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let tool = McpResourceSubscriptionTool::new(Some(registry(provider)), "en");

        assert_eq!(
            invoke(&tool, json!({"action": "subscribe"})).await["error"],
            json!("uri is required for action=subscribe")
        );
        assert_eq!(
            invoke(&tool, json!({"action": "unsubscribe"})).await["error"],
            json!("uri is required for action=unsubscribe")
        );
        assert_eq!(
            invoke(&tool, json!({"action": "poll"})).await["error"],
            json!("uri is required for action=poll")
        );
        assert_eq!(
            invoke(&tool, json!({})).await["error"],
            json!("action is required (expected subscribe, unsubscribe, list or poll)")
        );
        assert_eq!(
            invoke(&tool, json!({"action": "get"})).await["error"],
            json!("invalid action: \"get\" (expected subscribe, unsubscribe, list or poll)")
        );
        assert_eq!(
            invoke(&tool, json!({"action": "poll", "uri": "mem://missing"})).await["error"],
            json!("no MCP resource subscription for `mem://missing`")
        );
        assert_eq!(
            invoke(&tool, json!([])).await["error"],
            json!("tool arguments must be an object")
        );
    }
}
