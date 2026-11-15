//! Steering wiring: a queued message lands in the next model request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::BaseMessage;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use parking_lot::Mutex;
use serde_json::json;

use crate::agent::Agent;
use crate::rails::{AgentRail, RailContext};
use crate::steering::SteeringQueue;

use super::{agent, echo_tool};

/// Upper bound on a loop run, so a regression fails instead of hanging.
const BOUND: Duration = Duration::from_secs(10);

/// Whether `messages` holds a user message whose text is exactly `text`.
fn contains_user(messages: &[BaseMessage], text: &str) -> bool {
    messages
        .iter()
        .any(|message| message.role.is_user() && message.text().as_ref() == text)
}

/// The text of every user message in `messages`, oldest first.
fn user_texts(messages: &[BaseMessage]) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.role.is_user())
        .map(|message| message.text().into_owned())
        .collect()
}

/// A model script of two iterations: one `echo` tool call, then an answer.
fn tool_then_answer() -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::builder()
            .completion(MockCompletion::tool_call("echo", json!({})))
            .respond_text("done")
            .build(),
    )
}

/// Pushes one steering message during the first model call of a run.
struct SteerOnFirstStep {
    steering: SteeringQueue,
    pushed: Arc<Mutex<bool>>,
}

#[async_trait::async_trait]
impl AgentRail for SteerOnFirstStep {
    fn name(&self) -> &'static str {
        "steer-on-first-step"
    }

    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        let should_push = {
            let mut pushed = self.pushed.lock();
            let first = ctx.iteration == 1 && !*pushed;
            if first {
                *pushed = true;
            }
            first
        };
        if should_push {
            self.steering.push_text("stop after this step");
        }
        Ok(())
    }
}

#[tokio::test]
async fn steering_pushed_before_the_run_reaches_the_model() {
    let model = ScriptedModel::builder().respond_text("done").build();
    let agent = agent(Arc::new(model.clone()));
    agent.steer("focus on the refund policy");

    let output = agent.invoke(json!("hi"), None).await.expect("run succeeds");
    assert_eq!(output.output, "done");

    let calls = model.calls();
    assert!(
        contains_user(&calls[0].messages, "focus on the refund policy"),
        "the pre-run message is in the first request: {:?}",
        calls[0].messages
    );
}

#[tokio::test]
async fn steering_pushed_mid_run_lands_in_the_next_request() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("done")
        .build();
    let agent = agent(Arc::new(model.clone()));
    agent.add_tool(echo_tool());
    agent.register_rail(Arc::new(SteerOnFirstStep {
        steering: agent.steering(),
        pushed: Arc::new(Mutex::new(false)),
    }));

    let output = agent.invoke(json!("hi"), None).await.expect("run succeeds");
    assert_eq!(output.output, "done");

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert!(
        !contains_user(&calls[0].messages, "stop after this step"),
        "the first request predates the push"
    );
    assert!(
        contains_user(&calls[1].messages, "stop after this step"),
        "the second request observes the drained message"
    );
}

#[tokio::test]
async fn empty_queue_leaves_the_request_unchanged() {
    let baseline_model = ScriptedModel::builder().respond_text("done").build();
    let baseline_agent = agent(Arc::new(baseline_model.clone()));
    let baseline = baseline_agent
        .invoke(json!("hi"), None)
        .await
        .expect("baseline run");

    let steered_model = ScriptedModel::builder().respond_text("done").build();
    let steered_agent = agent(Arc::new(steered_model.clone()));
    assert_eq!(steered_agent.steer("   "), 0, "a blank message is dropped");
    let steered = steered_agent
        .invoke(json!("hi"), None)
        .await
        .expect("steered run");

    assert_eq!(
        baseline_model.calls()[0].messages,
        steered_model.calls()[0].messages
    );
    assert_eq!(baseline.output, steered.output);
}

/// Records the `steering_pending` count seen by every steering-drain firing.
struct RecordDrain {
    seen: Arc<Mutex<Vec<usize>>>,
}

#[async_trait::async_trait]
impl AgentRail for RecordDrain {
    fn name(&self) -> &'static str {
        "record-drain"
    }

    async fn before_steering_drain(&self, ctx: &mut RailContext) -> Result<()> {
        self.seen.lock().push(ctx.steering_pending);
        Ok(())
    }
}

/// One `(steering_pending, steering_limit)` pair seen at a model call.
type DrainFields = (usize, Option<usize>);

/// Caps the steering batch and records the drain fields at the next model call.
struct CapDrain {
    limit: Option<usize>,
    at_model_call: Arc<Mutex<Vec<DrainFields>>>,
}

#[async_trait::async_trait]
impl AgentRail for CapDrain {
    fn name(&self) -> &'static str {
        "cap-drain"
    }

    async fn before_steering_drain(&self, ctx: &mut RailContext) -> Result<()> {
        ctx.steering_limit = self.limit;
        Ok(())
    }

    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        self.at_model_call
            .lock()
            .push((ctx.steering_pending, ctx.steering_limit));
        Ok(())
    }
}

/// Refuses the steering drain.
struct RefuseDrain;

#[async_trait::async_trait]
impl AgentRail for RefuseDrain {
    fn name(&self) -> &'static str {
        "refuse-drain"
    }

    async fn before_steering_drain(&self, _ctx: &mut RailContext) -> Result<()> {
        Err(Error::config("steering refused"))
    }
}

#[tokio::test]
async fn the_drain_hook_fires_once_with_the_backlog_size() {
    let model = tool_then_answer();
    let react = agent(model.clone());
    react.add_tool(echo_tool());
    let seen = Arc::new(Mutex::new(Vec::new()));
    react.register_rail(Arc::new(RecordDrain {
        seen: Arc::clone(&seen),
    }));
    react.steer("first");
    react.steer("second");
    react.steer("third");

    let output = tokio::time::timeout(BOUND, react.invoke(json!("hi"), None))
        .await
        .expect("run finishes")
        .expect("run succeeds");
    assert_eq!(output.output, "done");

    assert_eq!(
        *seen.lock(),
        vec![3],
        "the hook fires once, on the first iteration, with the whole backlog"
    );
}

#[tokio::test]
async fn a_rail_caps_the_batch_and_leaves_the_surplus_queued() {
    let model = tool_then_answer();
    let react = agent(model.clone());
    react.add_tool(echo_tool());
    let at_model_call = Arc::new(Mutex::new(Vec::new()));
    react.register_rail(Arc::new(CapDrain {
        limit: Some(1),
        at_model_call: Arc::clone(&at_model_call),
    }));
    react.steer("first");
    react.steer("second");
    react.steer("third");

    let output = tokio::time::timeout(BOUND, react.invoke(json!("hi"), None))
        .await
        .expect("run finishes")
        .expect("run succeeds");
    assert_eq!(output.output, "done");

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        user_texts(&calls[0].messages),
        vec!["hi".to_owned(), "first".to_owned()],
        "the cap keeps the second and third message out of the first request"
    );
    assert_eq!(
        user_texts(&calls[1].messages),
        vec!["hi".to_owned(), "first".to_owned(), "second".to_owned()],
        "the next iteration drains the next message only"
    );
    let queued: Vec<String> = react
        .steering()
        .snapshot()
        .into_iter()
        .map(|message| message.text)
        .collect();
    assert_eq!(queued, vec!["third".to_owned()], "the surplus stays queued");
    assert_eq!(react.steering().len(), 1);
}

#[tokio::test]
async fn without_a_rail_the_whole_backlog_lands_in_the_first_request() {
    let model = tool_then_answer();
    let react = agent(model.clone());
    react.add_tool(echo_tool());
    react.steer("first");
    react.steer("second");
    react.steer("third");

    tokio::time::timeout(BOUND, react.invoke(json!("hi"), None))
        .await
        .expect("run finishes")
        .expect("run succeeds");

    let calls = model.calls();
    assert_eq!(
        user_texts(&calls[0].messages),
        vec![
            "hi".to_owned(),
            "first".to_owned(),
            "second".to_owned(),
            "third".to_owned(),
        ],
        "no rail means the whole backlog is drained as one batch"
    );
    assert!(react.steering().is_empty());
}

#[tokio::test]
async fn a_rail_error_aborts_the_run() {
    let model = tool_then_answer();
    let react = agent(model.clone());
    react.add_tool(echo_tool());
    react.register_rail(Arc::new(RefuseDrain));
    react.steer("first");

    let error = tokio::time::timeout(BOUND, react.invoke(json!("hi"), None))
        .await
        .expect("run finishes")
        .expect_err("a rail error aborts the run");
    assert!(
        error.to_string().contains("steering refused"),
        "the rail error surfaces: {error}"
    );
    assert!(model.calls().is_empty(), "the model is never called");
}

#[tokio::test]
async fn the_drain_hook_does_not_fire_on_an_empty_queue() {
    let model = tool_then_answer();
    let react = agent(model.clone());
    react.add_tool(echo_tool());
    let seen = Arc::new(Mutex::new(Vec::new()));
    react.register_rail(Arc::new(RecordDrain {
        seen: Arc::clone(&seen),
    }));

    tokio::time::timeout(BOUND, react.invoke(json!("hi"), None))
        .await
        .expect("run finishes")
        .expect("run succeeds");

    assert!(
        seen.lock().is_empty(),
        "an unsteered run never fires the hook"
    );
}

#[tokio::test]
async fn the_drain_fields_are_reset_after_the_hook() {
    let model = tool_then_answer();
    let react = agent(model.clone());
    react.add_tool(echo_tool());
    let at_model_call = Arc::new(Mutex::new(Vec::new()));
    react.register_rail(Arc::new(CapDrain {
        limit: Some(1),
        at_model_call: Arc::clone(&at_model_call),
    }));
    react.steer("first");
    react.steer("second");

    tokio::time::timeout(BOUND, react.invoke(json!("hi"), None))
        .await
        .expect("run finishes")
        .expect("run succeeds");

    let observed = at_model_call.lock().clone();
    assert_eq!(observed.len(), 2, "one observation per model call");
    assert!(
        observed.iter().all(|fields| *fields == (0, None)),
        "pending/limit are restored before the next firing point: {observed:?}"
    );
}
