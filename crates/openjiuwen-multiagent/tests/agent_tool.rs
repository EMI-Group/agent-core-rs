#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for [`AgentTool`], the agent-as-a-tool adapter.
//!
//! The suite registers [`common::FakeAgent`] members on a [`TeamRuntime`] and
//! drives the tool directly: the model-facing card shape, the child session one
//! dispatch builds (parent-scoped and invocation-scoped ids), the runtime's
//! dispatch, the mapping of a dropped runtime and of a failing member onto
//! tool errors, and the Python `str()` rendering of a result. Nothing touches
//! the network.

mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_multiagent::{AgentProvider, AgentTool, RuntimeConfig, TeamAgent, TeamRuntime};
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, FakeAgent};

/// The session one dispatch ran in, as `(session id, entity id)`.
type SeenSession = Arc<Mutex<Option<(String, Option<String>)>>>;

/// Registers an echoing member and returns the runtime plus the recorded session.
fn runtime_with_member(name: &str) -> (Arc<TeamRuntime>, SeenSession) {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let seen: SeenSession = Arc::new(Mutex::new(None));
    let recorded = Arc::clone(&seen);
    let card = agent_card(name);
    let agent: Arc<dyn openjiuwen_agent::Agent> = FakeAgent::new(card, move |_inputs, session| {
        let observed = session.map(|session| {
            (
                session.id().to_owned(),
                session.metadata().entity_id.clone(),
            )
        });
        *recorded.lock() = observed;
        Ok(openjiuwen_agent::AgentOutput::answer("done"))
    });
    let provider: AgentProvider = Arc::new(move || TeamAgent::new(Arc::clone(&agent)));
    runtime.register_agent(agent_card(name), provider);
    (runtime, seen)
}

#[tokio::test]
async fn card_shape() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let card = AgentCard::new("researcher").with_id("researcher-id");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));

    assert_eq!(tool.card().id, "researcher");
    assert_eq!(tool.card().name, "researcher");
    assert_eq!(tool.card().description, "");
    assert_eq!(
        tool.card().input_params,
        json!({"type": "object", "properties": {}})
    );
    assert!(tool.card().parallel_safe);
    assert_eq!(tool.agent_id(), "researcher-id");

    // A declared schema and description are forwarded verbatim.
    let described = AgentCard::new("writer")
        .with_id("writer-id")
        .with_description("writes prose")
        .with_input_params(json!({"type": "object", "properties": {"topic": {"type": "string"}}}));
    let tool = AgentTool::new(&described, Arc::downgrade(&runtime));
    assert_eq!(tool.card().id, "writer");
    assert_eq!(tool.card().description, "writes prose");
    assert_eq!(
        tool.card().input_params,
        json!({"type": "object", "properties": {"topic": {"type": "string"}}})
    );
}

#[tokio::test]
async fn invoke_dispatches_through_runtime() {
    let (runtime, seen) = runtime_with_member("worker");
    let card = agent_card("worker");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));

    let ctx = ToolContext::new()
        .with_session("parent")
        .with_metadata_entry("tool_call_id", json!("call-1"));
    let result = tool.invoke(json!({"query": "hi"}), &ctx).await.unwrap();

    assert_eq!(result, json!({"output": "done", "result_type": "answer"}));
    let observed = seen.lock().clone().expect("the member ran in a session");
    assert_eq!(observed.0, "parent:call-1");
    assert_eq!(observed.1.as_deref(), Some("worker"));
}

#[tokio::test]
async fn invocation_id_falls_back_for_tool_call_id() {
    let (runtime, seen) = runtime_with_member("worker");
    let card = agent_card("worker");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));

    let ctx = ToolContext::new().with_session("parent");
    let invocation_id = ctx.invocation_id.clone();
    tool.invoke(json!({"query": "hi"}), &ctx).await.unwrap();

    let observed = seen.lock().clone().expect("the member ran in a session");
    assert_eq!(observed.0, format!("parent:{invocation_id}"));
    assert_eq!(observed.1.as_deref(), Some("worker"));
}

#[tokio::test]
async fn dropped_runtime_is_a_tool_error() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let card = agent_card("worker");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));
    drop(runtime);

    let error = tool
        .invoke(json!({"query": "hi"}), &ToolContext::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(
        error.message().starts_with("Agent execution error:"),
        "{error}"
    );
    assert!(error.message().contains("team runtime dropped"), "{error}");
}

#[tokio::test]
async fn agent_failure_becomes_tool_error() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let card = agent_card("worker");
    let agent: Arc<dyn openjiuwen_agent::Agent> =
        FakeAgent::new(card.clone(), |_inputs, _session| {
            Err(Error::validation("boom"))
        });
    let provider: AgentProvider = Arc::new(move || TeamAgent::new(Arc::clone(&agent)));
    runtime.register_agent(card.clone(), provider);

    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));
    let error = tool
        .invoke(json!({"query": "hi"}), &ToolContext::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.message().contains("boom"), "{error}");
    assert!(
        error.message().starts_with("Agent execution error:"),
        "{error}"
    );
}

#[tokio::test]
async fn unknown_member_becomes_tool_error() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let card = AgentCard::new("ghost").with_id("ghost");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));

    let error = tool
        .invoke(json!({"query": "hi"}), &ToolContext::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.message().contains("ghost"), "{error}");
}

#[tokio::test]
async fn limiter_bounds_concurrent_dispatches() {
    /// In-flight and peak-in-flight sub-agent dispatches.
    #[derive(Default)]
    struct Concurrency {
        current: usize,
        peak: usize,
    }

    /// An agent that counts its concurrent invocations.
    struct CountingAgent {
        card: AgentCard,
        state: Arc<Mutex<Concurrency>>,
    }

    #[async_trait::async_trait]
    impl openjiuwen_agent::Agent for CountingAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> openjiuwen_common::error::Result<openjiuwen_agent::AgentOutput> {
            {
                let mut guard = self.state.lock();
                guard.current += 1;
                guard.peak = guard.peak.max(guard.current);
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            self.state.lock().current -= 1;
            Ok(openjiuwen_agent::AgentOutput::answer("done"))
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> futures::stream::BoxStream<
            'static,
            openjiuwen_common::error::Result<openjiuwen_agent::AgentChunk>,
        > {
            futures::stream::empty().boxed()
        }
    }

    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let card = agent_card("worker");
    let state = Arc::new(Mutex::new(Concurrency::default()));
    let agent: Arc<dyn openjiuwen_agent::Agent> = Arc::new(CountingAgent {
        card: card.clone(),
        state: Arc::clone(&state),
    });
    let provider: AgentProvider = Arc::new(move || TeamAgent::new(Arc::clone(&agent)));
    runtime.register_agent(card.clone(), provider);

    let tool = Arc::new(
        AgentTool::new(&card, Arc::downgrade(&runtime))
            .with_limiter(Arc::new(tokio::sync::Semaphore::new(1))),
    );
    let first = {
        let tool = Arc::clone(&tool);
        tokio::spawn(async move { tool.invoke(json!({}), &ToolContext::new()).await })
    };
    let second = {
        let tool = Arc::clone(&tool);
        tokio::spawn(async move { tool.invoke(json!({}), &ToolContext::new()).await })
    };
    assert_eq!(first.await.unwrap().unwrap()["output"], "done");
    assert_eq!(second.await.unwrap().unwrap()["output"], "done");
    assert_eq!(state.lock().peak, 1);
}

#[tokio::test]
async fn cancelled_context_is_reported() {
    let (runtime, _seen) = runtime_with_member("worker");
    let card = agent_card("worker");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));

    let ctx = ToolContext::new();
    ctx.cancellation.cancel();
    let error = tool.invoke(json!({}), &ctx).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

#[tokio::test]
async fn cancelled_context_emits_no_task_events() {
    use openjiuwen_multiagent::events::{TeamEvent, TeamEventFilter};

    let (runtime, _seen) = runtime_with_member("worker");
    let recorded: Arc<Mutex<Vec<TeamEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&recorded);
    runtime.events().subscribe(
        TeamEventFilter::new(),
        Arc::new(move |event: &TeamEvent| sink.lock().push(event.clone())),
    );

    let card = agent_card("worker");
    let tool = AgentTool::new(&card, Arc::downgrade(&runtime));
    let ctx = ToolContext::new();
    ctx.cancellation.cancel();
    let error = tool.invoke(json!({}), &ctx).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert!(recorded.lock().is_empty());
}

// ---------------------------------------------------------------------------
// Result rendering (Python `str(result)`)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn render_keeps_a_bare_string_bare() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let tool = AgentTool::new(&agent_card("worker"), Arc::downgrade(&runtime));

    assert_eq!(tool.render_for_llm(&json!("plain text")), "plain text");
    assert_eq!(tool.render_for_llm(&json!("")), "");
}

#[tokio::test]
async fn render_unwraps_a_string_content_object() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let tool = AgentTool::new(&agent_card("worker"), Arc::downgrade(&runtime));

    assert_eq!(
        tool.render_for_llm(&json!({"content": "the answer", "trace": 7})),
        "the answer",
        "the foundation rule keeps only the string content"
    );
}

#[tokio::test]
async fn render_uses_python_repr_for_containers() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let tool = AgentTool::new(&agent_card("worker"), Arc::downgrade(&runtime));

    assert_eq!(
        tool.render_for_llm(&json!({"a": 1, "ok": true, "missing": null})),
        "{'a': 1, 'ok': True, 'missing': None}",
        "booleans and null follow Python's spelling"
    );
    assert_eq!(
        tool.render_for_llm(&json!([1, true, null, "x"])),
        "[1, True, None, 'x']"
    );
    assert_eq!(tool.render_for_llm(&json!(true)), "True");
    assert_eq!(tool.render_for_llm(&json!(7)), "7");
    assert_eq!(
        tool.render_for_llm(&json!({"content": {"nested": true}})),
        "{'content': {'nested': True}}",
        "a non-string content stays part of the rendered object"
    );
}

#[tokio::test]
async fn render_keeps_the_empty_text_for_a_null_result() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let tool = AgentTool::new(&agent_card("worker"), Arc::downgrade(&runtime));

    assert_eq!(tool.render_for_llm(&Value::Null), "");
}
