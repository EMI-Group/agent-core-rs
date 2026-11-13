//! Steering wiring: a queued message lands in the next model request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::BaseMessage;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use parking_lot::Mutex;
use serde_json::json;

use crate::agent::Agent;
use crate::rails::{AgentRail, RailContext};
use crate::steering::SteeringQueue;

use super::{agent, echo_tool};

/// Whether `messages` holds a user message whose text is exactly `text`.
fn contains_user(messages: &[BaseMessage], text: &str) -> bool {
    messages
        .iter()
        .any(|message| message.role.is_user() && message.text().as_ref() == text)
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
