//! Workflow-component scenario: `http_request`, `sub_workflow`, and `questioner`.
//!
//! This binary drives the three components that reach outside a single workflow
//! through their real seams: an HTTP request against a loopback server, a nested
//! workflow invoked as one node, and a human-in-the-loop questioner that pauses a
//! checkpointed run and resumes it with an `InteractiveInput`. The last scenario
//! hands the same HTTP workflow to the `Runner` facade, so the component topology
//! is also proven through the runner layer that hosts it in production.
//!
//! Everything is offline and deterministic: the HTTP server is a `wiremock`
//! instance bound to loopback (its `uri()` is the only URL used), the child
//! workflow's model is a `ScriptedModel`, checkpointers are in-memory, no global
//! `Runner` is taken, and nothing reads the clock. Assertions are made against
//! recorded evidence — the workflow output, the mock server's received requests,
//! the model's call history, the checkpointer's state — never against timing,
//! timestamps or generated ids. Every await that can hang runs under the scenario
//! deadline from [`support`].
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::sync::Arc;

use openjiuwen::core::common::error::ErrorKind;
use openjiuwen::core::foundation::Model;
use openjiuwen::core::runner::{RunOptions, Runner, WorkflowRef};
use openjiuwen::core::session::{
    InteractiveInput, MemoryCheckpointer, Session, WorkflowSessionOptions,
};
use openjiuwen::core::workflow::components::questioner::{QuestionerComponent, QuestionerConfig};
use openjiuwen::core::workflow::components::sub_workflow::SubWorkflowComponent;
use openjiuwen::core::workflow::components::{HttpComponentConfig, HttpRequestComponent};
use openjiuwen::core::workflow::{
    ComponentOptions, EndComponent, LlmCompConfig, LlmComponent, StartComponent, Workflow,
    WorkflowCard, WorkflowExecutionState,
};
use openjiuwen_test_support::wiremock::matchers::{method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use openjiuwen_test_support::{workflow_session, workflow_session_with, ScriptedModel};
use serde_json::{json, Value};

use support::within;

/// The questioner node id, which is also the interaction id its wire carries.
const QUESTIONER_NODE: &str = "ask";

/// Builds an `http_request` component from its `request_params` object, the way a
/// workflow card would deserialize it.
fn http_component(params: &Value) -> HttpRequestComponent {
    let config = serde_json::from_value::<HttpComponentConfig>(json!({"request_params": params}))
        .expect("the request params describe a valid http request config");
    HttpRequestComponent::new(config)
}

/// `start → http → end`; the request url is the run's `base` input plus `url_path`,
/// so the loopback address never has to be baked into the card.
fn http_workflow(id: &str, url_path: &str, end_inputs: Value) -> Workflow {
    http_workflow_with(id, url_path, &json!({}), end_inputs)
}

/// The same topology with extra `request_params` merged into the http node, which is
/// how a card sets `response_handling` (e.g. `throw_on_http_error`).
fn http_workflow_with(
    id: &str,
    url_path: &str,
    extra_params: &Value,
    end_inputs: Value,
) -> Workflow {
    let mut params = json!({"url": format!("{{{{base}}}}{url_path}"), "method": "GET"});
    if let (Some(params), Some(extra)) = (params.as_object_mut(), extra_params.as_object()) {
        for (key, value) in extra {
            params.insert(key.clone(), value.clone());
        }
    }
    let mut workflow = Workflow::new(WorkflowCard::new(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"base": "${inputs.base}"})),
        )
        .add_workflow_comp(
            "http",
            http_component(&params),
            ComponentOptions::new().with_inputs_schema(json!({"base": "${start.base}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(end_inputs),
        )
        .add_connection("start", "http")
        .add_connection("http", "end");
    workflow
}

/// Mounts one `GET path` mock on `server` answering `status` with `body`.
async fn mount_get(server: &MockServer, url_path: &str, status: u16, body: Value) {
    within(
        "mounting the loopback mock",
        Mock::given(method("GET"))
            .and(path(url_path))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(server),
    )
    .await;
}

/// The paths of the requests the loopback server recorded, in order.
async fn recorded_paths(server: &MockServer) -> Vec<String> {
    let requests = within(
        "reading the loopback server's recording",
        server.received_requests(),
    )
    .await
    .expect("the mock server records every request");
    requests
        .iter()
        .map(|request| request.url.path().to_owned())
        .collect()
}

/// A workflow session carrying `checkpointer`, which `resume` requires.
fn checkpointed(id: &str, checkpointer: &MemoryCheckpointer) -> Arc<dyn Session> {
    workflow_session_with(WorkflowSessionOptions {
        session_id: Some(id.to_owned()),
        checkpointer: Some(Arc::new(checkpointer.clone())),
        ..WorkflowSessionOptions::default()
    })
}

/// `start → llm → end` as a child workflow: the model renders `Answer: {{topic}}`
/// and the child's end reports the answer under `answer`.
fn child_workflow(model: Arc<dyn Model>) -> Workflow {
    let mut child = Workflow::new(WorkflowCard::new("child"));
    child
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"topic": "${inputs.topic}"})),
        )
        .add_workflow_comp(
            "llm",
            LlmComponent::new(LlmCompConfig {
                model: Some(model),
                user_prompt_template: Some("Answer: {{topic}}".into()),
                ..Default::default()
            })
            .expect("the llm component fixture builds"),
            ComponentOptions::new().with_inputs_schema(json!({"topic": "${start.topic}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${llm.result}"})),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");
    child
}

// The response wire object is the component's contract: the end node reads the
// nested body field and the status out of it, so a caller never handles the raw
// response.
#[tokio::test]
async fn start_http_end_resolves_the_body_field_and_status() {
    let server = within("the loopback server start", MockServer::start()).await;
    mount_get(&server, "/item", 200, json!({"value": 7})).await;

    let workflow = http_workflow(
        "http-body",
        "/item",
        json!({"value": "${http.body.value}", "status": "${http.statusCode}"}),
    );

    let output = within(
        "the http workflow run",
        workflow.invoke(json!({"base": server.uri()}), workflow_session("http-body")),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"value": 7, "status": 200}})
    );
    assert_eq!(recorded_paths(&server).await, vec!["/item".to_owned()]);
}

// With `response_handling.throw_on_http_error` explicitly off, a non-2xx is data: the
// run completes and the end node can read `ok`, the status and the error body.
#[tokio::test]
async fn a_non_2xx_response_is_data_with_its_status() {
    let server = within("the loopback server start", MockServer::start()).await;
    mount_get(&server, "/broken", 503, json!({"error": "boom"})).await;

    let workflow = http_workflow_with(
        "http-error",
        "/broken",
        &json!({"response_handling": {"throw_on_http_error": false}}),
        json!({
            "ok": "${http.ok}",
            "status": "${http.statusCode}",
            "error": "${http.body.error}",
        }),
    );

    let output = within(
        "the failing http workflow run",
        workflow.invoke(
            json!({"base": server.uri()}),
            workflow_session("http-error"),
        ),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"ok": false, "status": 503, "error": "boom"}})
    );
    assert_eq!(recorded_paths(&server).await, vec!["/broken".to_owned()]);
}

// `throw_on_http_error` defaults to `true` (Python's default), so a failed status never
// reaches the end node: the run fails with the component's coded `[102002]` error naming
// the node, the status and `ok = false`.
#[tokio::test]
async fn a_non_2xx_response_throws_by_default() {
    let server = within("the loopback server start", MockServer::start()).await;
    mount_get(&server, "/broken", 503, json!({"error": "boom"})).await;

    let workflow = http_workflow(
        "http-throw",
        "/broken",
        json!({"status": "${http.statusCode}"}),
    );

    let error = within(
        "the throwing http workflow run",
        workflow.invoke(
            json!({"base": server.uri()}),
            workflow_session("http-throw"),
        ),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    let text = error.to_string();
    assert!(
        text.contains("node `http` failed at superstep 2"),
        "the error names the failing node: {text}"
    );
    assert!(text.contains("[102002]"), "unexpected error text: {text}");
    assert!(
        text.contains("statusCode=503"),
        "unexpected error text: {text}"
    );
    assert!(text.contains("ok=false"), "unexpected error text: {text}");
    assert_eq!(recorded_paths(&server).await, vec!["/broken".to_owned()]);
}

// A `sub_workflow` node runs the child against a fresh child session and returns
// the child's end output unchanged, so the parent reads `${sub.output.answer}`.
// The child's model call proves the parent's resolved inputs reached it.
#[tokio::test]
async fn a_parent_reads_the_child_workflow_answer() {
    let model = ScriptedModel::builder()
        .respond_text("the child answer")
        .build();
    let child = child_workflow(Arc::new(model.clone()));
    let sub = SubWorkflowComponent::new(&child).expect("the child workflow compiles");

    let mut parent = Workflow::new(WorkflowCard::new("parent"));
    parent
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"topic": "${inputs.topic}"})),
        )
        .add_workflow_comp(
            "sub",
            sub,
            ComponentOptions::new().with_inputs_schema(json!({"topic": "${start.topic}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${sub.output.answer}"})),
        )
        .add_connection("start", "sub")
        .add_connection("sub", "end");

    let output = within(
        "the nested workflow run",
        parent.invoke(
            json!({"topic": "widgets"}),
            workflow_session("nested-workflow"),
        ),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"answer": "the child answer"}})
    );
    assert_eq!(model.call_count(), 1, "the child ran exactly once");
    let call = model
        .last_call()
        .expect("the child's llm node called the model");
    assert_eq!(
        call.last_user_text().as_deref(),
        Some("Answer: widgets"),
        "the parent's inputs reached the child through the sub node"
    );
}

// The questioner pauses the run with an `__interaction__` wire whose id is the
// node id, keeps the snapshot on the session's checkpointer, and completes on
// `resume`. `question_content` mode needs no model at all.
#[tokio::test]
async fn the_questioner_pauses_and_resumes_with_the_answer() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-hitl", &checkpointer);
    let config = QuestionerConfig::default()
        .with_question_content("请问{{topic}}是什么？")
        .with_extract_fields_from_response(false);
    let component = QuestionerComponent::new(config).expect("the questioner config is valid");

    let mut workflow = Workflow::new(WorkflowCard::new("questioner-hitl"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"topic": "${inputs.topic}"})),
        )
        .add_workflow_comp(
            QUESTIONER_NODE,
            component,
            ComponentOptions::new().with_inputs_schema(json!({"topic": "${start.topic}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "question": "${ask.question}",
                "user_response": "${ask.user_response}",
            })),
        )
        .add_connection("start", QUESTIONER_NODE)
        .add_connection(QUESTIONER_NODE, "end");

    let paused = within(
        "the paused questioner run",
        workflow.invoke(json!({"topic": "天气"}), Arc::clone(&session)),
    )
    .await
    .unwrap();

    assert_eq!(paused.state, WorkflowExecutionState::InputRequired);
    let wire = &paused.result[0];
    assert_eq!(wire["type"], json!("__interaction__"));
    assert_eq!(
        wire["payload"]["id"],
        json!(QUESTIONER_NODE),
        "the interaction id is the component id"
    );
    assert_eq!(wire["payload"]["question"], json!("请问天气是什么？"));
    assert_eq!(
        checkpointer.len(session.id()),
        1,
        "the pause kept a snapshot for the resume"
    );

    let mut interactive = InteractiveInput::new();
    interactive.update(QUESTIONER_NODE, json!("今天晴朗"));
    let resumed = within(
        "the resumed questioner run",
        workflow.resume(interactive, Arc::clone(&session)),
    )
    .await
    .unwrap();

    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(
        resumed.result,
        json!({
            "output": {
                "question": "请问天气是什么？",
                "user_response": "今天晴朗",
            }
        })
    );
    assert_eq!(
        checkpointer.len(session.id()),
        0,
        "a completed run drops its snapshot"
    );
}

// The `Runner` facade resolves a live `WorkflowRef` and runs the same component
// topology; the end output contract is unchanged at that layer.
#[tokio::test]
async fn the_runner_drives_the_http_workflow() {
    let server = within("the loopback server start", MockServer::start()).await;
    mount_get(&server, "/runner-item", 200, json!({"value": 42})).await;

    let workflow = http_workflow(
        "runner-http",
        "/runner-item",
        json!({"value": "${http.body.value}", "status": "${http.statusCode}"}),
    );
    let runner = Runner::new();

    let output = within(
        "the runner workflow run",
        runner.run_workflow(
            WorkflowRef::of(Arc::new(workflow)),
            json!({"base": server.uri()}),
            RunOptions::default(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"value": 42, "status": 200}})
    );
    assert_eq!(
        recorded_paths(&server).await,
        vec!["/runner-item".to_owned()]
    );

    // `stop` clears the runner's registries, so it runs once every assertion is done.
    within("the runner shutdown", runner.stop()).await.unwrap();
}
