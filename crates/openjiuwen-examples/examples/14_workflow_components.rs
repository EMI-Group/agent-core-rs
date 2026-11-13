//! `14_workflow_components` — a sub-workflow, an HTTP request, and a human-in-the-loop
//! question, wired into one ticket-triage family.
//!
//! Python counterparts: `openjiuwen/core/workflow/components/flow/workflow_comp.py`
//! (`SubWorkflowComponent`), `…/components/tool/http/http_request_component.py`
//! (`HttpRequestComponent`) and `…/components/llm/questioner_comp.py`
//! (`QuestionerComponent`).
//!
//! **One story, three seams.** `start → quote → status → confirm → end` ends a *ticket*:
//! the child workflow prices it, an HTTP request reads its approval state from a server
//! this example starts, and a questioner asks a human to confirm that state. Every value
//! the end node reports crossed one of those boundaries.
//!
//! *Sub-workflow.* `SubWorkflowComponent::new(&child)` compiles the child and drives it as
//! one parent node. The child runs against a fresh child session (its own state store), so
//! its node ids never collide with the parent's, and the child's `end` output is returned
//! unchanged — which is why the parent reads `${quote.output.quote_id}`: the value is
//! produced inside the child and consumed outside it.
//!
//! *HTTP request.* `HttpRequestComponent` is configured with Python's configuration tree
//! (`request_params`) and templates `{{key}}` placeholders from its resolved inputs, so the
//! url is `{{base}}/quotes/{{quote_id}}` where `base` is the loopback server's address and
//! `quote_id` is the child's output. The component answers `{statusCode, headers, body,
//! url, ok}`; HTTP failures are data (`ok = false`), and only transport failures become the
//! coded `[102000]` workflow error.
//!
//! *Questioner.* The questioner renders its `question_content` from the node inputs and
//! asks through `ComponentContext::interact`; an open question halts the run as
//! `INPUT_REQUIRED` with an `__interaction__` wire whose id is the node id. `Workflow::resume`
//! re-runs only the interrupted superstep with the seeded answers and needs a checkpointer,
//! which is why the session carries a `MemoryCheckpointer`. The component is configured in
//! `question_content` mode; enabling `extract_fields_from_response` adds a model-driven
//! extraction round (`QuestionerConfig::with_model` + `with_field_names`), covered by
//! `openjiuwen-workflow`'s `questioner_component` suite.
//!
//! Everything is offline and deterministic: the only network is loopback, the server answers
//! a fixed body, and no model is involved. The narrative never contains the ephemeral port.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::Arc;

use anyhow::{anyhow, ensure, Context};
use async_trait::async_trait;
use openjiuwen::core::common::error::{Error, Result};
use openjiuwen::core::session::interaction::INTERACTION_FRAME_TYPE;
use openjiuwen::core::session::{
    Checkpointer, InteractiveInput, MemoryCheckpointer, Session, WorkflowSession,
    WorkflowSessionOptions,
};
use openjiuwen::core::workflow::{
    Component, ComponentCard, ComponentContext, ComponentOptions, EndComponent,
    HttpComponentConfig, HttpRequestComponent, QuestionerComponent, QuestionerConfig,
    StartComponent, SubWorkflowComponent, Workflow, WorkflowCard, WorkflowExecutionState,
    WorkflowOutput,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use common::banner;

/// The ticket the family triages; also the id the child quotes.
const TICKET: &str = "T-1001";

/// How many seats the ticket asks for.
const SEATS: u64 = 3;

/// The child workflow's price per seat, in cents.
const UNIT_PRICE_CENTS: u64 = 2_500;

/// The component type tag of the child's pricing node.
const ESTIMATOR: &str = "estimate";

/// The questioner node id, which is also its interaction id on a suspended run.
const QUESTION_NODE: &str = "confirm";

/// The answer the suspended run is resumed with.
const ANSWER: &str = "approved: T-1001";

/// The canned JSON body the loopback server answers every request with.
const STATUS_BODY: &str = r#"{"status":"approved","approver":"ada"}"#;

/// Answers every loopback request with [`STATUS_BODY`] and a fixed `HTTP/1.1 200` head.
///
/// The bytes are hand-written because the crate carries no HTTP test-server dependency;
/// one accept loop plus one canned response exercises the component's real wire path.
async fn serve_status(listener: TcpListener) -> std::io::Result<()> {
    loop {
        let (mut socket, _peer) = listener.accept().await?;
        tokio::spawn(async move {
            // A GET carries no body, so the blank line ends the request head; the client
            // sends the head in one segment, but looping keeps a short read from cutting
            // the response off.
            let mut head = Vec::new();
            let mut buffer = [0_u8; 512];
            loop {
                match socket.read(&mut buffer).await {
                    Ok(0) => break,
                    Err(error) => {
                        tracing::debug!(%error, "the loopback server could not read a request");
                        break;
                    }
                    Ok(read) => {
                        head.extend_from_slice(&buffer[..read]);
                        if head.windows(4).any(|window| window == b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                STATUS_BODY.len(),
                STATUS_BODY
            );
            if let Err(error) = socket.write_all(response.as_bytes()).await {
                // The component reports a closed connection as an `[102000]` error, so a
                // failed write only explains where that error came from.
                tracing::debug!(%error, "the loopback server could not answer");
            }
            // The socket is dropped right after this, so a shutdown failure changes nothing.
            let _ = socket.shutdown().await;
        });
    }
}

/// Prices a ticket inside the child workflow.
///
/// This is the value that crosses the sub-workflow boundary: the child produces it, the
/// parent reads it back through `${quote.output.…}`.
struct Estimator {
    /// Price per seat, in cents.
    unit_price_cents: u64,
}

#[async_trait]
impl Component for Estimator {
    fn card(&self) -> ComponentCard {
        ComponentCard::new(ESTIMATOR, ESTIMATOR).with_description("prices a ticket")
    }

    fn component_type(&self) -> &str {
        ESTIMATOR
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let ticket = inputs
            .get("ticket")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if ticket.is_empty() {
            return Err(Error::workflow("the estimator needs a `ticket` input"));
        }
        let seats = inputs
            .get("seats")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::workflow("the estimator needs an integer `seats` input"))?;
        let amount_cents = seats.saturating_mul(self.unit_price_cents);
        tracing::info!(
            ticket,
            seats,
            amount_cents,
            "the child workflow priced the ticket"
        );
        Ok(json!({
            "quote_id": format!("Q-{}", ticket.to_uppercase()),
            "seats": seats,
            "amount_cents": amount_cents,
        }))
    }
}

/// The child workflow: `start → estimate → end`, ending with the stock flat `end`.
///
/// The flat `end` projects its resolved inputs into `{"output": …}`, so the child answers
/// `{"output": {"quote_id": …, "amount_cents": …}}` and the parent's references start at
/// `${quote.output.…}`.
fn quote_child() -> Workflow {
    let mut child = Workflow::new(WorkflowCard::new("quote-child"));
    child
        .set_start_comp(
            "start",
            StartComponent::new(),
            // A child run's `inputs` are exactly what the sub-workflow node resolved.
            ComponentOptions::new().with_inputs_schema(
                json!({"ticket": "${inputs.ticket}", "seats": "${inputs.seats}"}),
            ),
        )
        .add_workflow_comp(
            ESTIMATOR,
            Estimator {
                unit_price_cents: UNIT_PRICE_CENTS,
            },
            ComponentOptions::new().with_inputs_schema(
                json!({"ticket": "${start.ticket}", "seats": "${start.seats}"}),
            ),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "quote_id": "${estimate.quote_id}",
                "amount_cents": "${estimate.amount_cents}",
            })),
        )
        .add_connection("start", ESTIMATOR)
        .add_connection(ESTIMATOR, "end");
    child
}

/// The parent workflow: `start → quote → status → confirm → end`.
fn family() -> anyhow::Result<Workflow> {
    let sub =
        SubWorkflowComponent::new(&quote_child()).context("the child workflow does not compile")?;
    // The configuration is Python's `HttpComponentConfig` wire shape: `request_params`
    // holds the url template and every other field keeps its Python default.
    let http_config: HttpComponentConfig = serde_json::from_value(json!({
        "request_params": {
            "url": "{{base}}/quotes/{{quote_id}}",
            "method": "GET",
        }
    }))
    .context("the http_request configuration is not valid")?;
    let questioner = QuestionerComponent::new(
        QuestionerConfig::default()
            .with_question_content(
                "Quote {{quote_id}} is {{approval_state}}. \
                 Reply with the approval reference to continue.",
            )
            .with_extract_fields_from_response(false)
            .with_accept_language("en"),
    )
    .context("the questioner configuration is not valid")?;

    let mut workflow = Workflow::new(WorkflowCard::new("ticket-triage"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "ticket": "${inputs.ticket}",
                "seats": "${inputs.seats}",
                "base": "${inputs.base}",
            })),
        )
        .add_workflow_comp(
            "quote",
            sub,
            ComponentOptions::new().with_inputs_schema(
                json!({"ticket": "${start.ticket}", "seats": "${start.seats}"}),
            ),
        )
        .add_workflow_comp(
            "status",
            HttpRequestComponent::new(http_config),
            // `{{base}}`/`{{quote_id}}` are filled from these resolved inputs; the quote id
            // is the child's output, read across the sub-workflow boundary.
            ComponentOptions::new().with_inputs_schema(json!({
                "base": "${start.base}",
                "quote_id": "${quote.output.quote_id}",
            })),
        )
        .add_workflow_comp(
            QUESTION_NODE,
            questioner,
            // The question template renders these two fields into the asked question.
            ComponentOptions::new().with_inputs_schema(json!({
                "quote_id": "${quote.output.quote_id}",
                "approval_state": "${status.body.status}",
            })),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "quote_id": "${quote.output.quote_id}",
                "amount_cents": "${quote.output.amount_cents}",
                "http_status": "${status.statusCode}",
                "approval_state": "${status.body.status}",
                "approver": "${status.body.approver}",
                "question": "${confirm.question}",
                "answer": "${confirm.user_response}",
            })),
        )
        // One linear chain, so the single `end` node never joins a barrier group.
        .add_connection("start", "quote")
        .add_connection("quote", "status")
        .add_connection("status", QUESTION_NODE)
        .add_connection(QUESTION_NODE, "end");
    Ok(workflow)
}

/// The id and question of the first `__interaction__` wire a suspended run published.
fn interaction(output: &WorkflowOutput) -> anyhow::Result<(String, String)> {
    let payload = output
        .result
        .as_array()
        .and_then(|wires| {
            wires.iter().find(|wire| {
                wire.get("type").and_then(Value::as_str) == Some(INTERACTION_FRAME_TYPE)
            })
        })
        .and_then(|wire| wire.get("payload"))
        .ok_or_else(|| anyhow!("no interaction wire in {}", output.result))?;
    let field = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("the interaction wire carries no `{key}`: {payload}"))
    };
    Ok((field("id")?, field("question")?))
}

/// Starts the loopback status server and returns its task plus the base URL to call.
///
/// The ephemeral port lives only in the returned URL: the narrative states that the
/// server is loopback without printing an address that changes between runs.
async fn start_status_server(
) -> anyhow::Result<(tokio::task::JoinHandle<std::io::Result<()>>, String)> {
    banner::step("bind a loopback HTTP server and keep its ephemeral address in memory");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("binding the loopback status server failed")?;
    let address = listener
        .local_addr()
        .context("reading the loopback server's address failed")?;
    let server = tokio::spawn(serve_status(listener));
    banner::note(&format!(
        "the server is loopback-only and answers every request with {STATUS_BODY}; \
         the port never reaches the narrative"
    ));
    Ok((server, format!("http://{address}")))
}

/// Checks that each end-node field crossed the boundary it should have, then reports
/// where the fields came from.
fn report(result: &Value, snapshots: usize) -> anyhow::Result<()> {
    banner::answer(&serde_json::to_string(result)?);
    let output = result
        .get("output")
        .ok_or_else(|| anyhow!("the end node returned no `output` object: {result}"))?;
    // One entry per seam: the child workflow, the HTTP response, and the resumed answer.
    let expected = [
        ("quote_id", json!(format!("Q-{TICKET}"))),
        ("amount_cents", json!(SEATS * UNIT_PRICE_CENTS)),
        ("http_status", json!(200)),
        ("approval_state", json!("approved")),
        ("answer", json!(ANSWER)),
    ];
    for (field, value) in expected {
        ensure!(
            output[field] == value,
            "the `{field}` field did not reach the output as {value}: {output}"
        );
    }
    ensure!(
        snapshots == 0,
        "the completed run should have released its snapshot"
    );
    banner::note(&format!(
        "quote_id {quote_id} and amount {amount} cents came out of the child workflow; \
         status {status} came out of the loopback HTTP response; the answer {answer:?} \
         came from the resumed question",
        quote_id = output["quote_id"],
        amount = output["amount_cents"],
        status = output["http_status"],
        answer = ANSWER
    ));
    banner::note(
        "the parent run never saw a child node id: the sub-workflow session owns its own \
         state lanes",
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;

    banner::title("14 workflow components");
    banner::note(&format!(
        "no model is involved: a sub-workflow prices the ticket, an HTTP request reads its \
         approval state, and a questioner asks a human to confirm (run mode: {})",
        mode.describe()
    ));
    banner::step(
        "topology: start -> quote (sub_workflow) -> status (http_request) -> \
         confirm (questioner) -> end",
    );

    let (server, base) = start_status_server().await?;
    let workflow = family()?;
    let checkpointer = MemoryCheckpointer::new();
    // The checkpointer is what makes a suspend/resume possible: the interrupted run parks
    // one graph snapshot under the session id, and `resume` replays from it.
    let session: Arc<dyn Session> =
        Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
            session_id: Some("workflow-components".to_owned()),
            checkpointer: Some(Arc::new(checkpointer.clone()) as Arc<dyn Checkpointer>),
            ..WorkflowSessionOptions::default()
        }));

    banner::step(&format!(
        "invoke the family with ticket {TICKET}, {SEATS} seats, and the loopback base url"
    ));
    let suspended = workflow
        .invoke(
            json!({"ticket": TICKET, "seats": SEATS, "base": base}),
            Arc::clone(&session),
        )
        .await
        .context("the family failed before it asked its question")?;
    ensure!(
        suspended.state == WorkflowExecutionState::InputRequired,
        "the run finished in state {} instead of waiting for the answer",
        suspended.state
    );
    let (node, question) =
        interaction(&suspended).context("the suspended run published no question")?;
    banner::step(&format!(
        "the run suspended in state {} on node {node:?}",
        suspended.state
    ));
    banner::note(&format!("the workflow asked: {question}"));
    banner::note(&format!(
        "the session's checkpointer holds {} snapshot(s) for the resume",
        checkpointer.len(session.id())
    ));
    ensure!(
        question.contains(&format!("Q-{TICKET}")),
        "the question did not render the child workflow's quote id: {question}"
    );

    banner::step(&format!(
        "resume the same session with the answer {ANSWER:?}"
    ));
    let mut answers = InteractiveInput::new();
    answers.update(&node, json!(ANSWER));
    let completed = workflow
        .resume(answers, Arc::clone(&session))
        .await
        .context("resuming the family failed")?;
    ensure!(
        completed.state == WorkflowExecutionState::Completed,
        "the resumed run finished in state {} instead of COMPLETED",
        completed.state
    );
    banner::step(&format!("state {}", completed.state));
    report(&completed.result, checkpointer.len(session.id()))?;

    server.abort();
    Ok(())
}
