//! Final-request check wiring: reject-and-iterate, finish, and the default.
//!
//! The check runs before a no-tool-call answer. A rejection injects its notice
//! as a user message and runs another iteration, a finish returns the check's
//! own output with the run's progress fields filled, and an allow lets the
//! normal answer through.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_test_support::{ModelCall, ScriptedModel};
use serde_json::json;

use crate::agent::Agent;
use crate::output::AgentOutput;
use crate::policy::{
    DefaultFinalRequestCheck, FinalRequestCheck, FinalRequestContext, FinalRequestDecision,
};

use super::agent;

/// What a [`CountingCheck`] does while its budget lasts.
enum Mode {
    /// Reject the first `times` requests with `reason`/`inject`, then allow.
    Reject {
        /// How many leading requests are rejected.
        times: usize,
        /// Reason reported for a rejection.
        reason: String,
        /// Notice the loop appends as a user message.
        inject: Option<String>,
    },
    /// Finish the first request with `output` instead of calling the model.
    Finish(Box<AgentOutput>),
}

/// A final-request check that counts its calls and applies `mode`.
struct CountingCheck {
    mode: Mode,
    calls: Arc<AtomicUsize>,
}

impl CountingCheck {
    /// Builds the check and a handle to its call counter.
    fn new(mode: Mode) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                mode,
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait::async_trait]
impl FinalRequestCheck for CountingCheck {
    async fn check(&self, ctx: &FinalRequestContext<'_>) -> Result<FinalRequestDecision> {
        assert_eq!(ctx.agent_id, "react-test");
        let seen = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(match &self.mode {
            Mode::Reject {
                times,
                reason,
                inject,
            } if seen < *times => FinalRequestDecision::Reject {
                reason: reason.clone(),
                inject: inject.clone(),
            },
            Mode::Finish(output) if seen == 0 => FinalRequestDecision::Finish(output.clone()),
            Mode::Reject { .. } | Mode::Finish(_) => FinalRequestDecision::Allow,
        })
    }
}

/// Whether `call` sent a user message containing `needle`.
fn has_user_notice(call: &ModelCall, needle: &str) -> bool {
    call.messages
        .iter()
        .any(|message| message.role.is_user() && message.text().contains(needle))
}

#[tokio::test]
async fn reject_injects_a_message_and_continues() {
    let model = ScriptedModel::builder()
        .respond_text("draft")
        .respond_text("final")
        .build();
    let agent = agent(Arc::new(model.clone()));
    let (check, calls) = CountingCheck::new(Mode::Reject {
        times: 1,
        reason: "draft".to_owned(),
        inject: Some("please tighten the answer".to_owned()),
    });
    agent.set_final_request_check(Arc::new(check));

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "final");
    assert_eq!(output.iterations_used, Some(2));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the check saw both answers"
    );

    let second = model.calls().get(1).cloned().expect("a second model call");
    assert!(
        has_user_notice(&second, "please tighten the answer"),
        "the injected notice reaches the next request"
    );
}

#[tokio::test]
async fn finish_returns_the_custom_output() {
    let model = ScriptedModel::builder().respond_text("draft").build();
    let agent = agent(Arc::new(model.clone()));
    let (check, calls) = CountingCheck::new(Mode::Finish(Box::new(AgentOutput::answer("forced"))));
    agent.set_final_request_check(Arc::new(check));

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "forced");
    assert_eq!(output.iterations_used, Some(1));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.call_count(), 1, "the round stops before a model call");
}

#[tokio::test]
async fn default_check_rejects_an_empty_final_response() {
    let model = ScriptedModel::builder()
        .respond_text("")
        .respond_text("real answer")
        .build();
    let agent = agent(Arc::new(model.clone()));
    agent.set_final_request_check(Arc::new(DefaultFinalRequestCheck::new()));

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "real answer");
    assert_eq!(output.iterations_used, Some(2));

    let second = model.calls().get(1).cloned().expect("a second model call");
    assert!(
        has_user_notice(&second, "Reply with a final answer or call a tool"),
        "the default notice reaches the next request"
    );
}
