use super::*;
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::interrupt::InterruptionState;
use crate::output::ResultType;
use crate::rails::{AskUserRail, ConfirmInterruptRail};
use openjiuwen_context::ContextConfig;
use openjiuwen_foundation::{MessageContent, Role, ToolContext};
use openjiuwen_session::AgentSession;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};

#[test]
fn request_for_inherits_the_configured_model_name() {
    let runtime_request = ModelRequestConfig::default();
    let mut config = ReActAgentConfig::new();
    config.model_name = "gpt-4o".to_owned();
    let runtime = AgentRuntime {
        card_id: "agent".to_owned(),
        card_name: "Agent".to_owned(),
        config,
        model: Arc::new(openjiuwen_test_support::ScriptedModel::text("x")),
        request_config: runtime_request,
        rails: Arc::new(RailRegistry::new()),
        engine: Arc::new(ContextEngine::new(ContextConfig::default())),
        tools: IndexMap::new(),
        tool_cards: Vec::new(),
        visible_tool_info: Vec::new(),
        visible_cards: Vec::new(),
        callbacks: Arc::new(crate::callbacks::CallbackFramework::new()),
        steering: crate::steering::SteeringQueue::new(),
        exposure: crate::policy::ToolExposurePolicy::allow_all(),
        mcp_allowlist: crate::policy::McpToolAllowlist::new(),
        client_tools: Arc::new(crate::policy::ClientToolRegistry::new()),
        resilience: crate::policy::ToolResiliencePolicy::new(),
        final_check: None,
        memory_provider: None,
        usage_telemetry: Arc::new(UsageTelemetry::new()),
    };
    assert_eq!(request_for(&runtime, &[]).model_name, "gpt-4o");
}

#[test]
fn reset_ctx_clears_event_fields_but_keeps_the_invocation_ones() {
    let mut ctx = RailContext::new(RailEvent::BeforeInvoke)
        .with_agent_id("agent")
        .with_input(AgentInput::from_text("hi"))
        .with_iteration(3);
    ctx.extra.insert("kept".to_owned(), Value::Bool(true));
    ctx.messages.push(BaseMessage::user("m"));
    ctx.response = Some(crate::react::model_step::ModelReply::default());
    ctx.request_retry(None);
    ctx.skip_tool_with("skip");

    reset_ctx(&mut ctx, RailEvent::AfterModelCall);

    assert_eq!(ctx.event, RailEvent::AfterModelCall);
    assert_eq!(ctx.agent_id, "agent");
    assert_eq!(ctx.iteration, 3);
    assert!(ctx.extra.contains_key("kept"));
    assert_eq!(ctx.messages.len(), 0);
    assert!(ctx.response.is_none());
    assert!(ctx.retry.is_none());
    assert!(ctx.skip_tool.is_none());
}

#[test]
fn with_progress_keeps_existing_values() {
    let observations = vec![Observation {
        content: "out".to_owned(),
        ..Observation::default()
    }];
    let output = with_progress(AgentOutput::answer("done"), 2, &observations);
    assert_eq!(output.iterations_used, Some(2));
    assert_eq!(output.observations, observations);
    let output = with_progress(
        AgentOutput::answer("done").with_iterations_used(9),
        2,
        &observations,
    );
    assert_eq!(output.iterations_used, Some(9));
}

/// Tool that echoes its arguments, used by the loop tests below.
struct Echo(ToolCard);

#[async_trait::async_trait]
impl Tool for Echo {
    fn card(&self) -> &ToolCard {
        &self.0
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(serde_json::json!({"echoed": inputs}))
    }
}

fn agent_with(model: ScriptedModel) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("react"));
    let model: Arc<dyn Model> = Arc::new(model);
    agent.set_model(model);
    agent.add_tool(Arc::new(Echo(ToolCard::new(
        "echo",
        "echoes its arguments",
        serde_json::json!({"type": "object"}),
    ))));
    agent
}

#[tokio::test]
async fn tool_calls_execute_and_feed_the_answer() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "echo",
            serde_json::json!({"x": 1}),
        ))
        .respond_text("done")
        .build();
    let agent = agent_with(model);
    let output = agent
        .invoke(Value::String("hi".to_owned()), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    assert_eq!(output.iterations_used, Some(2));
    assert_eq!(output.observations.len(), 1);
    assert_eq!(output.observations[0].name.as_deref(), Some("echo"));
    assert_eq!(
        output.observations[0].tool_call_id.as_deref(),
        Some("call_echo")
    );
}

#[tokio::test]
async fn empty_response_is_retried_once() {
    let model = ScriptedModel::builder()
        .respond_text("")
        .respond_text("second attempt")
        .build();
    let agent = agent_with(model);
    let output = agent
        .invoke(Value::String("hi".to_owned()), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "second attempt");
    assert_eq!(output.iterations_used, Some(2));
}

#[tokio::test]
async fn iteration_limit_reports_max_iterations() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
        .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
        .build();
    let agent = agent_with(model);
    agent.configure(ReActAgentConfig {
        max_iterations: 2,
        ..ReActAgentConfig::new()
    });
    let output = agent
        .invoke(Value::String("hi".to_owned()), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Error);
    assert_eq!(output.output, MAX_ITERATIONS_MSG);
    assert_eq!(output.termination_reason.as_deref(), Some("max_iterations"));
    assert!(output.truncated_by_max_iterations);
    assert_eq!(output.iterations_used, Some(2));
    assert_eq!(output.observations.len(), 2);
}

#[tokio::test]
async fn final_answer_policy_makes_one_tool_free_call() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
        .respond_text("final answer")
        .build();
    let agent = agent_with(model.clone());
    agent.configure(ReActAgentConfig {
        max_iterations: 1,
        on_max_iterations: MaxIterationsPolicy::FinalAnswer,
        ..ReActAgentConfig::new()
    });
    let output = agent
        .invoke(Value::String("hi".to_owned()), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "final answer");
    assert!(output.finalization_error.is_none());
    let last = model
        .last_call()
        .expect("the finalization call is recorded");
    assert!(last.request.tools.is_none());
}

#[tokio::test]
async fn confirm_interrupt_pauses_and_resumes() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "echo",
            serde_json::json!({"x": 1}),
        ))
        .respond_text("approved")
        .build();
    let agent = agent_with(model);
    agent.register_rail(Arc::new(ConfirmInterruptRail::new()));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s1"));

    let paused = agent
        .invoke(
            Value::String("do it".to_owned()),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
    assert!(paused.is_interrupt());
    assert_eq!(paused.interrupt_ids.len(), 1);
    assert_eq!(paused.state.len(), 1);
    let interrupt_id = paused.interrupt_ids[0].clone();

    let resumed = agent
        .invoke(
            serde_json::json!({
                "query": {"interrupt_answers": {interrupt_id.clone(): {"approved": true}}},
            }),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "approved");
    assert_eq!(resumed.iterations_used, Some(2));
    assert_eq!(resumed.observations.len(), 1);
    assert_eq!(
        InterruptionState::load(&*session).await.unwrap(),
        None,
        "a resumed turn clears the persisted interruption"
    );
}

/// Answers with a text part plus an image part, like a screenshot tool.
struct Shot(ToolCard);

#[async_trait::async_trait]
impl Tool for Shot {
    fn card(&self) -> &ToolCard {
        &self.0
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(serde_json::json!({"content": [
            {"type": "text", "text": "ok"},
            {"type": "image", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]}))
    }
}

#[tokio::test]
async fn multimodal_tool_output_reaches_the_observation_and_the_next_request() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("shot", serde_json::json!({})))
        .respond_text("done")
        .build();
    let agent = ReActAgent::new(AgentCard::new("react"));
    let model_arc: Arc<dyn Model> = Arc::new(model.clone());
    agent.set_model(model_arc);
    agent.add_tool(Arc::new(Shot(ToolCard::new(
        "shot",
        "captures the screen",
        serde_json::json!({"type": "object"}),
    ))));

    let output = agent
        .invoke(Value::String("hi".to_owned()), None)
        .await
        .unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    assert_eq!(output.observations.len(), 1);
    assert!(output.observations[0].content.contains("ok"));
    assert_eq!(output.observations[0].parts.len(), 2);
    assert_eq!(
        output.observations[0].parts[0],
        openjiuwen_foundation::ContentPart::Text {
            text: "ok".to_owned()
        }
    );
    assert_eq!(
        output.observations[0].parts[1]
            .image_url()
            .map(|image| image.url.as_str()),
        Some("data:image/png;base64,AAA")
    );

    let last = model.last_call().expect("the answer call is recorded");
    let tool_message = last
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .expect("the request carries the tool message");
    let MessageContent::Parts(parts) = &tool_message.content else {
        panic!("a multimodal tool message carries parts");
    };
    assert!(parts.iter().any(|part| part.image_url().is_some()));
}

#[tokio::test]
async fn question_rail_answers_become_the_tool_result() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
        .respond_text("settled")
        .build();
    let agent = agent_with(model);
    agent.register_rail(Arc::new(AskUserRail::new()));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s2"));

    let paused = agent
        .invoke(Value::String("ask".to_owned()), Some(Arc::clone(&session)))
        .await
        .unwrap();
    assert!(paused.is_interrupt());
    let resume = serde_json::json!({
        "query": {"user_inputs": {"call_echo": "Paris"}},
    });
    let resumed = agent
        .invoke(resume, Some(Arc::clone(&session)))
        .await
        .unwrap();
    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "settled");
    assert_eq!(resumed.observations[0].content, "Paris");
}

/// A `BeforeInvoke` rail that publishes memory prompt variables the way
/// `MemoryRail` does.
struct MemoryVariablesRail;

#[async_trait::async_trait]
impl crate::rails::AgentRail for MemoryVariablesRail {
    fn name(&self) -> &'static str {
        "memory-variables"
    }

    async fn before_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        ctx.extra.insert(
            MEMORY_VARIABLES_KEY.to_owned(),
            serde_json::json!({
                "sys_memory_variables": "{\"name\":\"Alice\"}",
                "sys_long_term_memory": "[\"sticky\"]",
            }),
        );
        Ok(())
    }
}

#[tokio::test]
async fn memory_variables_reach_every_model_call() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
        .respond_text("done")
        .build();
    let agent = ReActAgent::new(AgentCard::new("react"));
    agent.set_model(Arc::new(model.clone()));
    agent.configure(ReActAgentConfig {
        prompt_template: vec![BaseMessage::system(
            "identity {{sys_long_term_memory}} vars {{sys_memory_variables}}",
        )],
        ..ReActAgentConfig::new()
    });
    agent.add_tool(Arc::new(Echo(ToolCard::new(
        "echo",
        "echoes its arguments",
        serde_json::json!({"type": "object"}),
    ))));
    agent.register_rail(Arc::new(MemoryVariablesRail));

    let output = agent
        .invoke(Value::String("hi".to_owned()), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
    let calls = model.calls();
    assert_eq!(calls.len(), 2, "one tool turn plus one answer turn");
    for call in calls {
        let system = call
            .messages
            .iter()
            .find(|message| message.role == Role::System)
            .expect("every request carries the system prompt");
        assert_eq!(
            system.text(),
            "identity [\"sticky\"] vars {\"name\":\"Alice\"}"
        );
    }
}
