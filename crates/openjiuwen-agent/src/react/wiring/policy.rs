//! Tool-policy wiring: what the model sees versus what executes.
//!
//! [`ToolExposurePolicy`] and the [`McpToolAllowlist`] filter only the
//! model-facing projection of a run, so a hidden tool still executes when the
//! model asks for it by name; [`ToolResiliencePolicy`] retries a failing tool
//! up to its per-tool attempt budget before the failure becomes the tool result.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{FunctionTool, Tool, ToolCard, ToolContext};
use openjiuwen_test_support::{MockCompletion, ModelCall, ScriptedModel};
use serde_json::{json, Value};

use crate::agent::Agent;
use crate::policy::{McpToolAllowlist, ToolExposurePolicy, ToolResilience, ToolResiliencePolicy};

use super::agent;

/// A `FunctionTool` named `name` that counts its invocations and answers `"ok"`.
fn counting_tool(name: &str, calls: &Arc<AtomicUsize>) -> Arc<dyn Tool> {
    let calls = Arc::clone(calls);
    let tool = FunctionTool::builder()
        .name(name)
        .description("counting tool")
        .input_schema(json!({"type": "object"}))
        .handler(move |_inputs: Value, _ctx: ToolContext| {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Value::String("ok".to_owned()))
            }
        })
        .build()
        .expect("the counting tool fixture builds");
    Arc::new(tool)
}

/// A mounted tool with an arbitrary card; its body is never reached here.
struct StubTool {
    card: ToolCard,
}

#[async_trait::async_trait]
impl Tool for StubTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(inputs)
    }
}

/// A tool loaded from the MCP `server`, carrying the wire metadata
/// [`openjiuwen_foundation::load_mcp_tools`] attaches.
fn mcp_tool(name: &str, server: &str) -> Arc<dyn Tool> {
    Arc::new(StubTool {
        card: ToolCard::new(name, "an MCP tool", json!({"type": "object"}))
            .with_metadata_entry("mcp", json!(true))
            .with_metadata_entry("server_name", json!(server)),
    })
}

/// A tool that fails its first `failures` invocations with `message`, then
/// answers `"ok"`; every invocation counts in `calls`.
struct FlakyTool {
    card: ToolCard,
    calls: Arc<AtomicUsize>,
    failures: usize,
    message: &'static str,
}

#[async_trait::async_trait]
impl Tool for FlakyTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt <= self.failures {
            return Err(Error::tool(self.message));
        }
        Ok(Value::String("ok".to_owned()))
    }
}

/// A [`FlakyTool`] named `name` that fails `failures` times with `message`.
fn flaky_tool(
    name: &str,
    calls: &Arc<AtomicUsize>,
    failures: usize,
    message: &'static str,
) -> Arc<dyn Tool> {
    Arc::new(FlakyTool {
        card: ToolCard::new(name, "flaky tool", json!({"type": "object"})),
        calls: Arc::clone(calls),
        failures,
        message,
    })
}

/// The tool schemas `call` sent, in request order.
fn tool_schemas(call: &ModelCall) -> Vec<Value> {
    call.request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The model-facing tool names `call` sent, in request order.
fn tool_names(call: &ModelCall) -> Vec<String> {
    tool_schemas(call)
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned))
        .collect()
}

/// The first recorded model call of `model`.
fn first_call(model: &ScriptedModel) -> ModelCall {
    let mut calls = model.calls();
    assert!(!calls.is_empty(), "the model was called at least once");
    calls.remove(0)
}

#[tokio::test]
async fn allowlist_hides_a_tool_from_the_model_but_execution_still_works() {
    let echo_calls = Arc::new(AtomicUsize::new(0));
    let hidden_calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("done")
        .build();
    let agent = agent(Arc::new(model.clone()));
    agent.add_tool(counting_tool("echo", &echo_calls));
    agent.add_tool(counting_tool("hidden", &hidden_calls));
    agent.set_tool_exposure_policy(ToolExposurePolicy::allowlist(["echo"]));

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");

    let schemas = tool_schemas(&first_call(&model));
    assert_eq!(schemas.len(), 1, "only one schema reaches the model");
    assert_eq!(schemas[0]["function"]["name"], json!("echo"));

    let mounted = agent
        .tools()
        .into_iter()
        .map(|card| card.name)
        .collect::<Vec<_>>();
    assert_eq!(mounted, vec!["echo".to_owned(), "hidden".to_owned()]);

    // The hidden tool is still executable, but this run never asked for it.
    assert_eq!(echo_calls.load(Ordering::SeqCst), 1);
    assert_eq!(hidden_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn denied_tool_is_hidden_even_when_the_mode_allows_all() {
    let echo_calls = Arc::new(AtomicUsize::new(0));
    let hidden_calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("hidden", json!({})))
        .respond_text("done")
        .build();
    let restricted = agent(Arc::new(model.clone()));
    restricted.add_tool(counting_tool("echo", &echo_calls));
    restricted.add_tool(counting_tool("hidden", &hidden_calls));
    restricted.set_tool_exposure_policy(ToolExposurePolicy::allow_all().deny("echo"));

    let output = restricted
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");
    assert_eq!(
        tool_names(&first_call(&model)),
        vec!["hidden".to_owned()],
        "the denied tool is the only hidden one"
    );

    // `deny_all` exposes no tool at all: the request carries no schema list.
    let denied_model = ScriptedModel::builder().respond_text("done").build();
    let denied = agent(Arc::new(denied_model.clone()));
    denied.add_tool(counting_tool("echo", &Arc::new(AtomicUsize::new(0))));
    denied.set_tool_exposure_policy(ToolExposurePolicy::deny_all());
    let output = denied
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");
    assert!(
        tool_names(&first_call(&denied_model)).is_empty(),
        "deny_all exposes no tool"
    );
}

#[tokio::test]
async fn resilience_retries_a_flaky_tool_and_then_succeeds() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("flaky", json!({})))
        .respond_text("done")
        .build();
    let agent = agent(Arc::new(model));
    agent.add_tool(flaky_tool("flaky", &calls, 2, "flaky"));
    let mut resilience = ToolResiliencePolicy::new();
    resilience.set("flaky", ToolResilience::new().with_attempts(3));
    agent.set_tool_resilience(resilience);

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 3, "three attempts were made");

    let observation = output.observations.first().expect("one observation");
    assert_eq!(observation.name.as_deref(), Some("flaky"));
    assert_eq!(observation.content, "ok");
}

#[tokio::test]
async fn resilience_gives_up_after_the_configured_attempts() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("boom", json!({})))
        .respond_text("done")
        .build();
    let agent = agent(Arc::new(model));
    agent.add_tool(flaky_tool("boom", &calls, usize::MAX, "boom"));
    let mut resilience = ToolResiliencePolicy::new();
    resilience.set("boom", ToolResilience::new().with_attempts(2));
    agent.set_tool_resilience(resilience);

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the budget is two attempts"
    );

    let observation = output.observations.first().expect("one observation");
    assert_eq!(observation.name.as_deref(), Some("boom"));
    assert!(
        observation.content.contains("boom"),
        "the failure reaches the model: {}",
        observation.content
    );
}

#[tokio::test]
async fn mcp_allowlist_hides_a_scoped_tool() {
    let model = ScriptedModel::builder().respond_text("done").build();
    let scoped = agent(Arc::new(model.clone()));
    scoped.add_tool(mcp_tool("write", "files"));
    let mut allowlist = McpToolAllowlist::new();
    allowlist.allow("files", "read");
    scoped.set_mcp_tool_allowlist(allowlist);

    let output = scoped
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");
    assert!(
        tool_names(&first_call(&model)).is_empty(),
        "the unlisted tool of a restricted server is hidden"
    );

    // Listing the tool on its server exposes it again.
    let allowed_model = ScriptedModel::builder().respond_text("done").build();
    let allowed = agent(Arc::new(allowed_model.clone()));
    allowed.add_tool(mcp_tool("write", "files"));
    let mut allowlist = McpToolAllowlist::new();
    allowlist.allow("files", "write");
    allowed.set_mcp_tool_allowlist(allowlist);

    let output = allowed
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "done");
    assert_eq!(
        tool_names(&first_call(&allowed_model)),
        vec!["write".to_owned()],
        "the allowed tool is visible again"
    );
}
