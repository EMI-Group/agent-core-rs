//! The pre-final-request check.
//!
//! Python parity: `FinalRequestCheck` / `set_final_request_check` in
//! `openjiuwen/core/single_agent/agents/react_agent.py`. The check runs inside
//! every model-call attempt, after the final system prompt, context window, and
//! tool list are built and before the provider is called. It lets a host
//! reject a request that must not reach the model, steer another iteration, or
//! finish the round with its own output.
use crate::output::AgentOutput;
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, ToolCard};
/// Everything the check may inspect about the pending final request.
#[derive(Debug, Clone, Copy)]
pub struct FinalRequestContext<'a> {
    /// Id of the agent making the request.
    pub agent_id: &'a str,
    /// 1-based index of the `ReAct` iteration making the request.
    pub iteration: usize,
    /// The answer text the loop currently holds for this round.
    pub output_text: &'a str,
    /// The context-window messages about to be sent to the model.
    pub messages: &'a [BaseMessage],
    /// The model-visible tools about to be sent to the model.
    pub tools: &'a [ToolCard],
}
impl<'a> FinalRequestContext<'a> {
    /// Builds a context from borrowed request data.
    #[must_use]
    pub fn new(
        agent_id: &'a str,
        iteration: usize,
        output_text: &'a str,
        messages: &'a [BaseMessage],
        tools: &'a [ToolCard],
    ) -> Self {
        Self {
            agent_id,
            iteration,
            output_text,
            messages,
            tools,
        }
    }
}
/// What the loop does with the pending final request.
#[derive(Debug, Clone, PartialEq)]
pub enum FinalRequestDecision {
    /// Let the model call go ahead.
    Allow,
    /// Do not call the model; append `inject` as a user message (when set) and
    /// run another iteration.
    Reject {
        /// Human-readable reason, used for logging and diagnostics.
        reason: String,
        /// Optional notice the loop appends as a user message.
        inject: Option<String>,
    },
    /// Do not call the model and return this output instead of its answer.
    Finish(Box<AgentOutput>),
}
/// Decides what happens to one pending final model request.
#[async_trait]
pub trait FinalRequestCheck: Send + Sync {
    /// Checks the pending request.
    ///
    /// # Errors
    ///
    /// Returns an error to abort the run; the loop does not call the model.
    async fn check(&self, ctx: &FinalRequestContext<'_>) -> Result<FinalRequestDecision>;
}
/// The default check: an empty or whitespace-only answer is rejected with a
/// notice that asks the model for a real answer or a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultFinalRequestCheck {
    inject: String,
}
/// The default rejection notice.
const DEFAULT_INJECT: &str =
    "Your previous response was empty. Reply with a final answer or call a tool.";
impl Default for DefaultFinalRequestCheck {
    fn default() -> Self {
        Self {
            inject: DEFAULT_INJECT.to_owned(),
        }
    }
}
impl DefaultFinalRequestCheck {
    /// Creates the default check.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Replaces the injected notice; a consuming builder.
    #[must_use]
    pub fn with_inject(mut self, inject: impl Into<String>) -> Self {
        self.inject = inject.into();
        self
    }
    /// The notice appended when the answer is empty.
    #[must_use]
    pub fn inject(&self) -> &str {
        &self.inject
    }
}
#[async_trait]
impl FinalRequestCheck for DefaultFinalRequestCheck {
    async fn check(&self, ctx: &FinalRequestContext<'_>) -> Result<FinalRequestDecision> {
        if ctx.output_text.trim().is_empty() {
            Ok(FinalRequestDecision::Reject {
                reason: "empty final response".to_owned(),
                inject: Some(self.inject.clone()),
            })
        } else {
            Ok(FinalRequestDecision::Allow)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::ToolCard;
    use serde_json::json;
    async fn decide(check: &dyn FinalRequestCheck, output_text: &str) -> FinalRequestDecision {
        let context = FinalRequestContext::new("agent", 2, output_text, &[], &[]);
        check.check(&context).await.expect("check")
    }
    #[tokio::test]
    async fn default_check_allows_a_non_empty_answer() {
        let check = DefaultFinalRequestCheck::new();
        assert_eq!(decide(&check, "42").await, FinalRequestDecision::Allow);
    }
    #[tokio::test]
    async fn default_check_rejects_empty_and_whitespace_answers() {
        let check = DefaultFinalRequestCheck::new();
        for text in ["", "   ", "\n\t"] {
            match decide(&check, text).await {
                FinalRequestDecision::Reject { reason, inject } => {
                    assert_eq!(reason, "empty final response");
                    assert_eq!(inject.as_deref(), Some(DEFAULT_INJECT));
                }
                other => panic!("expected Reject, got {other:?}"),
            }
        }
    }
    #[tokio::test]
    async fn default_check_can_replace_its_notice() {
        let check = DefaultFinalRequestCheck::new().with_inject("say something");
        assert_eq!(check.inject(), "say something");
        match decide(&check, "").await {
            FinalRequestDecision::Reject { inject, .. } => {
                assert_eq!(inject.as_deref(), Some("say something"));
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }
    #[tokio::test]
    async fn a_custom_check_can_finish_the_round() {
        struct FinishCheck;
        #[async_trait]
        impl FinalRequestCheck for FinishCheck {
            async fn check(&self, ctx: &FinalRequestContext<'_>) -> Result<FinalRequestDecision> {
                assert_eq!(ctx.agent_id, "agent");
                assert_eq!(ctx.iteration, 3);
                Ok(FinalRequestDecision::Finish(Box::new(AgentOutput::answer(
                    "stopped before the model",
                ))))
            }
        }
        let tools = [ToolCard::new("read", "read a file", json!({}))];
        let messages = [BaseMessage::user("hi")];
        let context = FinalRequestContext::new("agent", 3, "", &messages, &tools);
        let decision = FinishCheck.check(&context).await.expect("check");
        match decision {
            FinalRequestDecision::Finish(output) => {
                assert_eq!(output.output, "stopped before the model");
            }
            other => panic!("expected Finish, got {other:?}"),
        }
    }
}
