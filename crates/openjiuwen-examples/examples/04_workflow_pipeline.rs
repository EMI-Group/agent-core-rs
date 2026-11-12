//! `04_workflow_pipeline` — a conditional edge plus `${node.field}` data flow.
//!
//! Python counterparts: `examples/intelli_router/intelliRouter_demo.py` (a classifier
//! picks which path a request takes) and `examples/mcp/*/client_as_workflow.py` (steps
//! wired as workflow components whose data travels by `${node.field}` references).
//!
//! ```text
//! start ──▶ classify ──(conditional)──▶ short_answer ──▶ forward_short ──┐
//!                              └────────▶ long_answer  ──▶ forward_long ──┴──▶ end
//! ```
//!
//! `set_start_comp` / `add_workflow_comp` / `set_end_comp` register components with a
//! [`ComponentOptions`] input template, and `add_connection` draws the linear edges. The
//! conditional edge is `add_conditional_connection`, which attaches a
//! [`openjiuwen::core::graph::Router`] to `classify`: the router runs once that node
//! completes and returns the node ids to activate, so it decides *which* successor runs.
//! One connection is registered per candidate target and the engine activates the union of
//! every answer; each returned id must be a declared node, and an empty answer routes
//! nowhere.
//!
//! Each `inputs_schema` is a JSON template whose pure `"${node.field}"` string is replaced
//! by the value at that path of the run's state before the component runs — that is how one
//! component reads another's output. A missing path resolves to `null` instead of failing
//! the run, and the end component's flat projection drops `null` entries, so the branch
//! that did not run is simply absent from `end`'s result.
//!
//! Two invocations (a short prompt and a long one) take both branches of the same compiled
//! topology, offline: the pipeline is pure data flow, so it needs no model and no network.
//! The `forward_*` nodes exist because the graph engine's fan-in is a conjunction: a node
//! with incoming edges from two alternative branches waits for *both* and never runs when
//! only one is taken (see `openjiuwen-workflow`'s Known Issues). A component returning
//! targets from [`Component::route`] activates its successor without declaring an edge, so
//! that activation carries no barrier group and the single `end` node runs whichever branch
//! got there.

#![allow(
    // `Component::component_type` declares `&str`; every implementation here returns a
    // literal type tag, which is exactly the contract the SDK asks for.
    clippy::unnecessary_literal_bound
)]

mod common;

use std::sync::Arc;

use anyhow::{anyhow, bail, Context};
use async_trait::async_trait;
use openjiuwen::core::common::error::{Error, Result};
use openjiuwen::core::graph::{FnRouter, Router, RouterContext};
use openjiuwen::core::session::{Session, WorkflowSession};
use openjiuwen::core::workflow::{
    Component, ComponentCard, ComponentContext, ComponentOptions, EndComponent, StartComponent,
    Workflow, WorkflowCard, WorkflowExecutionState,
};
use serde_json::{json, Value};

use common::banner;

/// Word count below which a prompt counts as "short".
const SHORT_LIMIT: usize = 6;

/// A prompt under [`SHORT_LIMIT`] words.
const SHORT_PROMPT: &str = "how do I roll back";

/// A prompt at or over [`SHORT_LIMIT`] words.
const LONG_PROMPT: &str =
    "explain step by step how the workflow engine resolves node field references at run time";

/// Reads a string field of a component's resolved inputs, defaulting to `""`.
fn text_field<'a>(inputs: &'a Value, key: &str) -> &'a str {
    inputs.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// Labels a prompt `short` or `long` from its word count.
struct Classifier {
    /// Word count below which a prompt is short.
    limit: usize,
}

#[async_trait]
impl Component for Classifier {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("classify", "classify").with_description("labels a prompt by length")
    }

    fn component_type(&self) -> &str {
        "classify"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let words = text_field(&inputs, "text").split_whitespace().count();
        let label = if words < self.limit { "short" } else { "long" };
        tracing::info!(words, label, "classified the prompt");
        Ok(json!({"label": label, "words": words, "limit": self.limit}))
    }
}

/// Renders one branch's answer; it refuses a prompt the router sent to the wrong branch.
struct Answer {
    /// The label this branch answers for (`"short"` or `"long"`).
    label: &'static str,
    /// Prefix identifying the branch in the rendered answer.
    prefix: &'static str,
}

#[async_trait]
impl Component for Answer {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("answer", "answer").with_description("renders one branch's answer")
    }

    fn component_type(&self) -> &str {
        "answer"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let label = text_field(&inputs, "label");
        if label != self.label {
            return Err(Error::workflow(format!(
                "the {} answer component received a {label:?} prompt",
                self.label
            )));
        }
        let answer = format!("{}: {}", self.prefix, text_field(&inputs, "text"));
        tracing::info!(label, "rendering the branch answer");
        Ok(json!({"answer": answer, "label": label}))
    }
}

/// Activates `target` through [`Component::route`] instead of producing a value.
///
/// Routing is how a branch reaches the single `end` node without giving it a second
/// incoming edge, which would join the node's CNF barrier group and make it wait for every
/// branch.
struct Forward {
    /// The node id to activate.
    target: &'static str,
}

#[async_trait]
impl Component for Forward {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("forward", "forward").with_description("activates the end node")
    }

    fn component_type(&self) -> &str {
        "forward"
    }

    fn route(&self, _source: &Value) -> Option<Result<Vec<String>>> {
        Some(Ok(vec![self.target.to_owned()]))
    }
}

/// Builds the pipeline: `start → classify → {short_answer | long_answer} → end`.
fn pipeline() -> Workflow {
    // One router, registered once per candidate target: both registrations see the same
    // state (the `classify` output is already merged) and answer the same id, so the union
    // is exactly the branch the classifier labelled.
    let router = branch_router();
    let answer_inputs = ComponentOptions::new()
        .with_inputs_schema(json!({"text": "${start.text}", "label": "${classify.label}"}));

    let mut workflow = Workflow::new(WorkflowCard::new("prompt-pipeline"));
    workflow
        // The start node echoes a run input, so downstream templates can use
        // `${start.text}` as well as the root-level `${inputs.text}`.
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"text": "${inputs.text}"})),
        )
        .add_workflow_comp(
            "classify",
            Classifier { limit: SHORT_LIMIT },
            ComponentOptions::new().with_inputs_schema(json!({"text": "${start.text}"})),
        )
        .add_workflow_comp(
            "short_answer",
            Answer {
                label: "short",
                prefix: "one-liner",
            },
            answer_inputs.clone(),
        )
        .add_workflow_comp(
            "long_answer",
            Answer {
                label: "long",
                prefix: "walkthrough",
            },
            answer_inputs,
        )
        .add_workflow_comp(
            "forward_short",
            Forward { target: "end" },
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "forward_long",
            Forward { target: "end" },
            ComponentOptions::new(),
        )
        // The end template reads one field from the classifier and one from each branch;
        // the branch that did not run resolves to `null` and is dropped by the projection.
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "label": "${classify.label}",
                "words": "${classify.words}",
                "short_answer": "${short_answer.answer}",
                "long_answer": "${long_answer.answer}",
            })),
        )
        .add_connection("start", "classify")
        .add_connection("short_answer", "forward_short")
        .add_connection("long_answer", "forward_long")
        .add_conditional_connection("classify", "short_answer", Arc::clone(&router))
        .add_conditional_connection("classify", "long_answer", router);
    workflow
}

/// Routes to the answer component that matches the classifier's label.
fn branch_router() -> Arc<dyn Router> {
    Arc::new(FnRouter::new("by_label", |ctx: &RouterContext| {
        let label = ctx
            .state()
            .get_path("classify.label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        Box::pin(async move {
            match label.as_str() {
                "short" => Ok(vec!["short_answer".to_owned()]),
                "long" => Ok(vec!["long_answer".to_owned()]),
                // An unknown label fails the run instead of routing nowhere.
                other => Err(Error::graph(format!(
                    "the classifier produced the unknown label {other:?}"
                ))),
            }
        })
    }))
}

/// Reads the taken branch's answer out of the end node's result.
///
/// The end node projects its resolved inputs into `{"output": …}`; the untaken branch's
/// reference resolved to `null` and that projection drops `null` entries, so the untaken
/// branch is absent rather than `null`.
fn resolve_branch(result: &Value, label: &str) -> anyhow::Result<String> {
    let output = result
        .get("output")
        .ok_or_else(|| anyhow!("the end node returned {result} without its `output` wrapper"))?;
    let taken = format!("{label}_answer");
    let answer = output
        .get(taken.as_str())
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("`{taken}` did not resolve to a string: {output}"))?;
    let other = if label == "short" { "long" } else { "short" };
    if output.get(format!("{other}_answer").as_str()).is_some() {
        bail!("the {other} branch's field unexpectedly resolved on the {label} run: {output}");
    }
    Ok(answer.to_owned())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;

    banner::title("04 workflow pipeline");
    banner::note(&format!(
        "no model and no network needed: this pipeline is pure data flow (run mode: {})",
        mode.describe()
    ));

    let workflow = pipeline();
    banner::step("topology: start -> classify -(conditional)-> short_answer | long_answer -> end");
    banner::note("the end node's template resolves ${node.field} against the run's state");

    for (index, (prompt, expected)) in [(SHORT_PROMPT, "short"), (LONG_PROMPT, "long")]
        .into_iter()
        .enumerate()
    {
        let run = index + 1;
        let words = prompt.split_whitespace().count();
        banner::step(&format!(
            "run {run}: invoke a {words}-word prompt, expecting the {expected} branch: {prompt:?}"
        ));

        tracing::info!(run, prompt, "invoking the pipeline");
        let session: Arc<dyn Session> =
            Arc::new(WorkflowSession::new(format!("pipeline-run-{run}")));
        let output = workflow
            .invoke(json!({"text": prompt}), session)
            .await
            .with_context(|| format!("run {run} of the pipeline failed"))?;
        if output.state != WorkflowExecutionState::Completed {
            bail!(
                "run {run} finished in state {} instead of COMPLETED",
                output.state
            );
        }
        banner::step(&format!("run {run}: state {}", output.state));

        let answer = resolve_branch(&output.result, expected)
            .with_context(|| format!("run {run} did not resolve the {expected} branch"))?;
        banner::answer(&serde_json::to_string(&output.result)?);
        banner::note(&format!(
            "the {expected} branch answered {answer:?}; the other branch's field resolved to \
             null and the flat end projection dropped it"
        ));
    }

    banner::note("both branches ran; the topology and the router never changed");
    Ok(())
}
