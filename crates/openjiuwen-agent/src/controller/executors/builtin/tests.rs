//! Suites for the concrete task executors of the controller stack.
//!
//! Every suite is offline and deterministic: agents are fakes, models are
//! `openjiuwen_test_support::ScriptedModel`s, and each await is bounded.
use super::*;
use crate::output::AgentChunk;
use crate::react::ReActAgent;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_session::AgentSession;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use openjiuwen_workflow::{ComponentOptions, EndComponent, StartComponent, WorkflowCard};
use serde_json::json;

/// Agent that returns a canned output.
struct FakeAgent {
    card: AgentCard,
    output: AgentOutput,
}

#[async_trait]
impl Agent for FakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        Ok(self.output.clone())
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty().boxed()
    }
}

fn task(task_id: &str, task_type: &str, description: &str) -> Task {
    Task::submitted("s1", task_id, task_type, description)
}

fn ctx() -> TaskContext<'static> {
    TaskContext::new(Some(Arc::new(AgentSession::new("s1"))), "ctl")
}

fn agent_card() -> AgentCard {
    AgentCard::new("a")
}
#[tokio::test]
async fn controller_agent_maps_answers_and_errors() {
    let answer = ControllerAgent::new(
        agent_card(),
        Arc::new(FakeAgent {
            card: agent_card(),
            output: AgentOutput::answer("done"),
        }),
    );
    assert_eq!(answer.task_type(), AGENT_TASK_TYPE);
    assert_eq!(answer.description(), "runs a controller agent");
    assert_eq!(answer.card().name, "a");
    let result = answer
        .execute(&task("t1", AGENT_TASK_TYPE, "hi"), &ctx())
        .await
        .expect("answer");
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.display(), "done");
    assert_eq!(answer.agent().card().name, "a");

    let failure = ControllerAgent::new(
        agent_card(),
        Arc::new(FakeAgent {
            card: agent_card(),
            output: AgentOutput::error(""),
        }),
    );
    let result = failure
        .execute(&task("t2", AGENT_TASK_TYPE, "hi"), &ctx())
        .await
        .expect("error output");
    assert_eq!(result.status, TaskStatus::Failed);
    assert_eq!(result.error.as_deref(), Some("agent returned an error"));
}

#[tokio::test]
async fn controller_agent_maps_interrupts_onto_input_required() {
    let mut output = AgentOutput::interrupt(
        vec![crate::output::OutputFrame::interaction(
            0,
            "i1",
            json!({"kind": "confirm"}),
        )],
        vec!["i1".to_owned()],
    );
    output.output = "please confirm".to_owned();
    let executor = ControllerAgent::new(
        agent_card(),
        Arc::new(FakeAgent {
            card: agent_card(),
            output,
        }),
    );

    let result = executor
        .execute(&task("t1", AGENT_TASK_TYPE, "hi"), &ctx())
        .await
        .expect("interrupt");

    assert_eq!(result.status, TaskStatus::InputRequired);
    assert_eq!(result.data.len(), 2);
    assert_eq!(result.data[0].kind(), "json");
    assert_eq!(result.data[1], DataFrame::text("please confirm"));
}
/// Agent whose `stream` replays `frames`, then an optional error.
struct StreamAgent {
    card: AgentCard,
    frames: Vec<AgentChunk>,
    failure: Option<&'static str>,
}
#[async_trait]
impl Agent for StreamAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }
    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        Err(Error::validation("the stream agent is not invokable"))
    }
    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        let mut items: Vec<Result<AgentChunk>> = self.frames.iter().cloned().map(Ok).collect();
        if let Some(message) = self.failure {
            items.push(Err(Error::llm(message)));
        }
        futures::stream::iter(items).boxed()
    }
}
/// The scripted turn both reasoning tests drive the model with.
fn reasoning_completion() -> MockCompletion {
    MockCompletion::text("the answer").with_reasoning("the reasoning")
}
/// A `ReActAgent` whose model answers with `completion`.
fn reacting_agent(completion: &MockCompletion) -> Arc<ReActAgent> {
    let agent = Arc::new(ReActAgent::new(AgentCard::new("react")));
    let model: Arc<dyn Model> = Arc::new(
        ScriptedModel::builder()
            .chunks(completion.to_chunks())
            .build(),
    );
    agent.set_model(model);
    agent
}
#[tokio::test]
async fn controller_agent_captures_reasoning_only_when_exposed() {
    let exposing = ControllerAgent::new(agent_card(), reacting_agent(&reasoning_completion()));
    let result = exposing
        .execute(
            &task("t1", AGENT_TASK_TYPE, "hi"),
            &ctx().with_expose_reasoning(true),
        )
        .await
        .expect("reasoning run");
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.display(), "the answer");
    assert_eq!(result.reasoning.as_deref(), Some("the reasoning"));
    let plain = ControllerAgent::new(agent_card(), reacting_agent(&reasoning_completion()));
    let result = plain
        .execute(&task("t2", AGENT_TASK_TYPE, "hi"), &ctx())
        .await
        .expect("plain run");
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.display(), "the answer");
    assert_eq!(result.reasoning, None);
}
#[tokio::test]
async fn controller_agent_streams_interrupts_into_input_required() {
    let mut output = AgentOutput::interrupt(
        vec![crate::output::OutputFrame::interaction(
            0,
            "i1",
            json!({"kind": "confirm"}),
        )],
        vec!["i1".to_owned()],
    );
    output.output = "please confirm".to_owned();
    let executor = ControllerAgent::new(
        agent_card(),
        Arc::new(StreamAgent {
            card: agent_card(),
            frames: vec![
                AgentChunk::llm_reasoning(0, "thinking"),
                AgentChunk::answer(&output),
            ],
            failure: None,
        }),
    );
    let result = executor
        .execute(
            &task("t1", AGENT_TASK_TYPE, "hi"),
            &ctx().with_expose_reasoning(true),
        )
        .await
        .expect("streamed interrupt");
    assert_eq!(result.status, TaskStatus::InputRequired);
    assert_eq!(result.data.len(), 2);
    assert_eq!(result.data[1], DataFrame::text("please confirm"));
    assert_eq!(result.reasoning.as_deref(), Some("thinking"));
}
#[tokio::test]
async fn controller_agent_reports_stream_failures() {
    let empty = ControllerAgent::new(
        agent_card(),
        Arc::new(StreamAgent {
            card: agent_card(),
            frames: vec![AgentChunk::llm_reasoning(0, "thinking")],
            failure: None,
        }),
    );
    let error = empty
        .execute(
            &task("t1", AGENT_TASK_TYPE, "hi"),
            &ctx().with_expose_reasoning(true),
        )
        .await
        .expect_err("no answer frame");
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
    assert!(error.to_string().contains("without an answer frame"));
    let failing = ControllerAgent::new(
        agent_card(),
        Arc::new(StreamAgent {
            card: agent_card(),
            frames: vec![AgentChunk::answer(&AgentOutput::answer("done"))],
            failure: Some("stream broke"),
        }),
    );
    let error = failing
        .execute(
            &task("t2", AGENT_TASK_TYPE, "hi"),
            &ctx().with_expose_reasoning(true),
        )
        .await
        .expect_err("stream error propagates");
    assert!(error.to_string().contains("stream broke"));
    // The flag off path never touches the stream.
    let plain = failing
        .execute(&task("t3", AGENT_TASK_TYPE, "hi"), &ctx())
        .await
        .expect_err("the invoke path is used instead");
    assert!(plain.to_string().contains("not invokable"));
}

#[test]
fn workflow_map_output_covers_every_state() {
    let completed = WorkflowController::map_output(
        "wf",
        WorkflowOutput {
            result: json!({"answer": "hi"}),
            state: WorkflowExecutionState::Completed,
        },
    );
    assert_eq!(completed.status, TaskStatus::Completed);
    assert_eq!(completed.output, json!({"answer": "hi"}));
    assert_eq!(completed.metadata["workflow_id"], json!("wf"));

    let raw = WorkflowController::map_output(
        "wf",
        WorkflowOutput {
            result: Value::String("plain".to_owned()),
            state: WorkflowExecutionState::Completed,
        },
    );
    assert_eq!(raw.display(), "plain");

    let interrupted = WorkflowController::map_output(
        "wf",
        WorkflowOutput {
            result: json!([{"id": 1}, {"id": 2}]),
            state: WorkflowExecutionState::InputRequired,
        },
    );
    assert_eq!(interrupted.status, TaskStatus::InputRequired);
    assert_eq!(interrupted.data.len(), 2);
    assert!(interrupted.data.iter().all(|frame| frame.kind() == "json"));

    let single = WorkflowController::map_output(
        "wf",
        WorkflowOutput {
            result: json!({"id": 1}),
            state: WorkflowExecutionState::InputRequired,
        },
    );
    assert_eq!(single.data.len(), 1);

    let failed = WorkflowController::map_output(
        "wf",
        WorkflowOutput {
            result: Value::Null,
            state: WorkflowExecutionState::Error,
        },
    );
    assert_eq!(failed.status, TaskStatus::Failed);
    assert_eq!(
        failed.error.as_deref(),
        Some("workflow wf finished in error state")
    );
}

/// `start → end`, projecting the end output to `{"answer": <start.query>}`.
fn echo_workflow(id: &str) -> Arc<Workflow> {
    let mut workflow = Workflow::new(WorkflowCard::new(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"answer": "${start.query}"}))
                .with_outputs_schema(json!({"answer": "${output.output.answer}"})),
        )
        .add_connection("start", "end");
    Arc::new(workflow)
}

#[test]
fn workflow_controller_registry_helpers() {
    let controller = WorkflowController::new();
    assert!(controller.workflows().is_empty());
    assert!(controller.add_workflow(echo_workflow("a")).is_none());
    assert!(controller.add_workflow(echo_workflow("a")).is_some());
    assert_eq!(controller.workflows().len(), 1);
    assert!(controller.get("a").is_some());
    assert!(controller.get("missing").is_none());
    controller.add_workflows([echo_workflow("b"), echo_workflow("c")]);
    assert_eq!(controller.workflows().len(), 3);
    assert!(controller.remove_workflow("b").is_some());
    assert!(controller.remove_workflow("b").is_none());
}

#[tokio::test]
async fn workflow_controller_executes_the_selected_workflow() {
    let controller = WorkflowController::new();
    controller.add_workflows([echo_workflow("first"), echo_workflow("second")]);
    assert_eq!(controller.task_type(), WORKFLOW_TASK_TYPE);
    assert_eq!(controller.description(), "runs a workflow");

    // An explicit selection in the task metadata wins.
    let selected =
        task("t1", WORKFLOW_TASK_TYPE, "ignored").with_metadata("workflow_id", json!("second"));
    let result = controller.execute(&selected, &ctx()).await.expect("run");
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.metadata["workflow_id"], json!("second"));
    assert_ne!(result.display(), "");

    // A description naming a card id selects it; otherwise the only
    // workflow is used.
    let single = WorkflowController::new();
    single.add_workflow(echo_workflow("only"));
    let result = single
        .execute(&task("t2", WORKFLOW_TASK_TYPE, "only"), &ctx())
        .await
        .expect("run");
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.metadata["workflow_id"], json!("only"));
    assert_ne!(result.display(), "");

    let result = single
        .execute(&task("t3", WORKFLOW_TASK_TYPE, "unrelated"), &ctx())
        .await
        .expect("single workflow fallback");
    assert_eq!(result.metadata["workflow_id"], json!("only"));
}

#[tokio::test]
async fn workflow_controller_reports_missing_workflow_and_session() {
    let empty = WorkflowController::new();
    let error = empty
        .execute(&task("t1", WORKFLOW_TASK_TYPE, "hi"), &ctx())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
    assert!(error
        .to_string()
        .contains("no workflow registered for task t1"));

    let controller = WorkflowController::new();
    controller.add_workflow(echo_workflow("only"));
    let error = controller
        .execute(
            &task("t2", WORKFLOW_TASK_TYPE, "hi"),
            &TaskContext::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    assert!(error.to_string().contains("requires a session"));
}

fn info(task_type: &str, name: &str) -> ExecutorInfo {
    ExecutorInfo::new(task_type, name, format!("{name} executor"))
}

#[tokio::test]
async fn llm_controller_skips_the_model_for_one_executor() {
    let scripted = ScriptedModel::builder().build();
    let handle = scripted.clone();
    let controller = LLMController::new(
        Arc::new(scripted) as Arc<dyn Model>,
        vec![info("agent", "agent")],
    );

    assert_eq!(controller.select("hi", &[]).await.expect("select"), "agent");
    assert_eq!(handle.call_count(), 0);
    assert_eq!(controller.default_task_type(), "agent");
    assert_eq!(controller.executors().len(), 1);
}

#[tokio::test]
async fn llm_controller_parses_the_model_reply() {
    let scripted = ScriptedModel::builder().respond_text("workflow").build();
    let handle = scripted.clone();
    let controller = LLMController::new(
        Arc::new(scripted) as Arc<dyn Model>,
        vec![info("agent", "agent"), info("workflow", "workflow")],
    );

    let selected = controller.select("run it", &[Intent::default()]).await;
    assert_eq!(selected.expect("select"), "workflow");
    assert_eq!(handle.call_count(), 1);
}

#[tokio::test]
async fn llm_controller_falls_back_to_the_first_executor() {
    let scripted = ScriptedModel::builder().respond_text("huh?").build();
    let handle = scripted.clone();
    let controller = LLMController::new(
        Arc::new(scripted) as Arc<dyn Model>,
        vec![info("agent", "agent"), info("workflow", "workflow")],
    );

    let selected = controller.select("run it", &[]).await;
    assert_eq!(selected.expect("select"), "agent");
    assert_eq!(handle.call_count(), 1);
}

#[tokio::test]
async fn llm_controller_without_executors_uses_the_default() {
    let scripted = ScriptedModel::builder().build();
    let controller = LLMController::new(Arc::new(scripted) as Arc<dyn Model>, vec![])
        .with_default_task_type("fallback");

    assert_eq!(controller.default_task_type(), "fallback");
    assert_eq!(
        controller.select("hi", &[]).await.expect("select"),
        "fallback"
    );

    // The default is `llm` when nothing is configured.
    let bare = LLMController::new(
        Arc::new(ScriptedModel::builder().build()) as Arc<dyn Model>,
        vec![],
    );
    assert_eq!(bare.default_task_type(), LLM_TASK_TYPE);
}

#[tokio::test]
async fn llm_controller_builders_and_prompt() {
    let scripted = ScriptedModel::builder().respond_text("named").build();
    let handle = scripted.clone();
    let request = ModelRequestConfig::new("custom-model");
    let controller = LLMController::new(
        Arc::new(scripted) as Arc<dyn Model>,
        vec![info("agent", "agent")],
    )
    .with_prompt("pick carefully")
    .with_request(request)
    .add_executor(ExecutorInfo::new("named", "named", "the named one"));

    assert_eq!(controller.executors().len(), 2);
    let selected = controller.select("hi", &[]).await.expect("select");
    assert_eq!(selected, "named");
    assert_eq!(handle.call_count(), 1);

    let kinds = controller
        .build_messages("hi", &[Intent::default()])
        .iter()
        .map(|message| message.role.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            openjiuwen_foundation::Role::System,
            openjiuwen_foundation::Role::User
        ]
    );
}

#[tokio::test]
async fn llm_controller_is_usable_as_a_task_router_object() {
    let scripted = ScriptedModel::builder().respond_text("workflow").build();
    let router: Arc<dyn TaskRouter> = Arc::new(LLMController::new(
        Arc::new(scripted) as Arc<dyn Model>,
        vec![info("agent", "agent"), info("workflow", "workflow")],
    ));

    assert_eq!(router.name(), "llm");
    assert_eq!(
        router.select("hi", &[]).await.expect("select"),
        Some("workflow".to_owned())
    );
}

#[test]
fn executor_info_round_trips() {
    let info = ExecutorInfo::new("agent", "agent", "runs a controller agent");
    let wire = serde_json::to_value(&info).unwrap();
    assert_eq!(
        wire,
        json!({
            "task_type": "agent",
            "name": "agent",
            "description": "runs a controller agent",
        })
    );
    let parsed: ExecutorInfo = serde_json::from_value(wire).unwrap();
    assert_eq!(parsed, info);
    assert_eq!(ExecutorInfo::default(), ExecutorInfo::default());
}
