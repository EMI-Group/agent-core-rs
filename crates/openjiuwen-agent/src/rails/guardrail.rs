//! The security rail: scans user input, model output, and tool traffic through
//! a [`GuardrailPipeline`].
//!
//! Python parity: the guardrail hook of the harness rails, re-expressed on the
//! kernel [`AgentRail`] seam.
//!
//! Implemented hooks:
//!
//! - `BeforeInvoke` scans the user query (`scan_user_input`) when
//!   `stages.user_input` is enabled; a missing or empty query is skipped.
//! - `AfterModelCall` scans the reply text (`scan_model_output`) when
//!   `stages.model_outputs` is enabled; a missing or empty reply is skipped.
//! - `BeforeToolCall` scans the tool arguments (`scan_tool_input`) when
//!   `stages.tool_inputs` is enabled. Unparseable arguments are scanned as raw
//!   JSON text, so a malformed call never escapes the scan.
//! - `AfterToolCall` scans the tool result (`scan_tool_output`) when
//!   `stages.tool_outputs` is enabled: the raw output value, or the rendered
//!   text when the tool produced no value.
//!
//! `stages.prompts` is kept for parity with
//! [`ScanTargets`](openjiuwen_security::ScanTargets) but deliberately left
//! unwired: `BeforeModelCall` does not call `scan_prompt`, because the model
//! window is prepared by the context engine and the user and tool content in it
//! is already scanned at its own stage.
//!
//! Every scan finishes with
//! [`GuardrailOutcome::enforce`](openjiuwen_security::GuardrailOutcome::enforce),
//! and the resulting [`Error::Security`](openjiuwen_common::error::Error::security)
//! names the stage and the detected risk, so a blocking verdict aborts the run
//! with an actionable message.
use super::{AgentRail, RailContext};
use crate::input::AgentInput;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_security::{GuardrailOutcome, GuardrailPipeline};
use serde_json::Value;
use std::sync::Arc;
/// The pipeline stages the rail scans.
///
/// Reuses [`openjiuwen_security::ScanTargets`] so the pipeline and the rail
/// cannot disagree about what a stage means.
pub type StageFlags = openjiuwen_security::ScanTargets;
/// Scans the security-sensitive stages of an agent run.
pub struct GuardrailRail {
    pipeline: Arc<GuardrailPipeline>,
    stages: StageFlags,
    name: String,
}
impl GuardrailRail {
    /// Builds a rail over `pipeline` with every stage enabled.
    #[must_use]
    pub fn new(pipeline: Arc<GuardrailPipeline>) -> Self {
        Self {
            pipeline,
            stages: StageFlags::default(),
            name: "guardrail".to_owned(),
        }
    }
    /// Restricts the scanned stages.
    #[must_use]
    pub fn with_stages(mut self, stages: StageFlags) -> Self {
        self.stages = stages;
        self
    }
    /// Returns the pipeline the rail scans with.
    #[must_use]
    pub fn pipeline(&self) -> &Arc<GuardrailPipeline> {
        &self.pipeline
    }
    /// Returns the scanned stages.
    #[must_use]
    pub fn stages(&self) -> StageFlags {
        self.stages
    }
}
/// Converts a scan verdict into a rail result.
///
/// An allowed outcome passes; a blocking one becomes
/// [`Error::Security`](openjiuwen_common::error::Error::security) naming `stage`
/// plus the blocking risk type and level reported by
/// [`GuardrailOutcome::violation`].
fn enforce(outcome: &GuardrailOutcome, stage: &str) -> Result<()> {
    match outcome.enforce() {
        Ok(()) => Ok(()),
        Err(error) => {
            let (risk_type, level) = outcome.violation().map_or_else(
                || ("unknown".to_owned(), outcome.risk_level),
                |violation| (violation.risk_type, violation.risk_level),
            );
            // Only the finding metadata is logged, never the scanned content.
            tracing::warn!(
                target: "openjiuwen_agent::rails::guardrail",
                stage = %stage,
                risk_type = %risk_type,
                level = %level,
                error = %error,
                "guardrail blocked the run"
            );
            Err(Error::security(format!(
                "guardrail blocked {stage}: {risk_type} (level={level})"
            )))
        }
    }
}
#[async_trait]
impl AgentRail for GuardrailRail {
    fn name(&self) -> &str {
        &self.name
    }
    async fn before_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        if !self.stages.user_input {
            return Ok(());
        }
        let Some(query) = ctx.input.as_ref().and_then(AgentInput::query_text) else {
            return Ok(());
        };
        if query.is_empty() {
            return Ok(());
        }
        let outcome = self.pipeline.scan_user_input(query).await;
        enforce(&outcome, "user input")
    }
    async fn after_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        if !self.stages.model_outputs {
            return Ok(());
        }
        let Some(reply) = ctx.response.as_ref() else {
            return Ok(());
        };
        let text = reply.message.text();
        if text.is_empty() {
            return Ok(());
        }
        let outcome = self.pipeline.scan_model_output(text.into_owned()).await;
        enforce(&outcome, "model output")
    }
    async fn before_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        if !self.stages.tool_inputs {
            return Ok(());
        }
        let Some(call) = ctx.tool_call.as_ref() else {
            return Ok(());
        };
        let arguments = crate::abilities::parse_tool_arguments(&call.arguments)
            .unwrap_or_else(|_| Value::String(call.arguments.clone()));
        let outcome = self
            .pipeline
            .scan_tool_input(call.name.clone(), &arguments)
            .await;
        enforce(&outcome, "tool input")
    }
    async fn after_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        if !self.stages.tool_outputs {
            return Ok(());
        }
        let Some(result) = ctx.tool_result.as_ref() else {
            return Ok(());
        };
        let output = result
            .output
            .clone()
            .unwrap_or_else(|| Value::String(result.text.clone()));
        let outcome = self
            .pipeline
            .scan_tool_output(result.call.name.clone(), &output)
            .await;
        enforce(&outcome, "tool output")
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::abilities::ToolCallResult;
    use crate::rails::RailEvent;
    use crate::react::model_step::ModelReply;
    use async_trait::async_trait;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::prompt::ToolCall;
    use openjiuwen_foundation::BaseMessage;
    use openjiuwen_security::{Guardrail, GuardrailContext, GuardrailResult, RiskLevel};
    use serde_json::json;

    /// A guardrail that blocks every scan at [`RiskLevel::Critical`].
    struct Blocking;

    #[async_trait]
    impl Guardrail for Blocking {
        fn name(&self) -> &'static str {
            "test-blocking"
        }

        async fn check(&self, _ctx: &GuardrailContext) -> Result<GuardrailResult> {
            Ok(GuardrailResult::block(
                RiskLevel::Critical,
                "test_risk",
                Some(json!({"k": 1})),
                None,
            ))
        }
    }

    /// A guardrail that allows every scan.
    struct Passing;

    #[async_trait]
    impl Guardrail for Passing {
        fn name(&self) -> &'static str {
            "test-passing"
        }

        async fn check(&self, _ctx: &GuardrailContext) -> Result<GuardrailResult> {
            Ok(GuardrailResult::pass())
        }
    }

    /// Builds a rail over one guardrail with `stages` enabled.
    fn rail_with(guardrail: impl Guardrail + 'static, stages: StageFlags) -> GuardrailRail {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(guardrail);
        GuardrailRail::new(Arc::new(pipeline)).with_stages(stages)
    }

    /// Builds a rail over one guardrail with every stage enabled.
    fn rail(guardrail: impl Guardrail + 'static) -> GuardrailRail {
        rail_with(guardrail, StageFlags::default())
    }

    /// Asserts `error` is a security error naming `stage`, `risk_type`, and level.
    fn assert_blocked(error: &openjiuwen_common::error::Error, stage: &str) {
        assert_eq!(error.kind(), ErrorKind::Security, "{error}");
        let message = error.to_string();
        assert!(message.contains(stage), "{message}");
        assert!(message.contains("test_risk"), "{message}");
        assert!(message.contains("CRITICAL"), "{message}");
    }

    /// A context carrying a text query.
    fn invoke_ctx(query: &str) -> RailContext {
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        ctx.input = Some(AgentInput::parse(&json!({"query": query})).unwrap());
        ctx
    }

    /// A context carrying a model reply with `text`.
    fn model_ctx(text: &str) -> RailContext {
        let mut ctx = RailContext::new(RailEvent::AfterModelCall);
        ctx.response = Some(ModelReply {
            message: BaseMessage::assistant(text),
            ..ModelReply::default()
        });
        ctx
    }

    /// A context carrying the tool call `bash`.
    fn tool_call_ctx(arguments: Value) -> RailContext {
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        ctx.tool_call = Some(ToolCall::new("bash").with_arguments(arguments));
        ctx
    }

    /// A context carrying a successful result for the tool call `bash`.
    fn tool_result_ctx(output: Option<Value>, text: &str) -> RailContext {
        let call = ToolCall::new("bash");
        let mut ctx = RailContext::new(RailEvent::AfterToolCall);
        ctx.tool_call = Some(call.clone());
        ctx.tool_result = Some(ToolCallResult {
            output,
            text: text.to_owned(),
            success: true,
            error: None,
            parts: Vec::new(),
            call,
        });
        ctx
    }

    #[test]
    fn defaults_name_and_stage_flags() {
        let pipeline = Arc::new(GuardrailPipeline::new());
        let rail = GuardrailRail::new(Arc::clone(&pipeline));
        assert_eq!(rail.name(), "guardrail");
        assert_eq!(rail.stages(), StageFlags::default());
        assert!(rail.stages().user_input);
        assert!(rail.stages().tool_outputs);
        assert!(Arc::ptr_eq(rail.pipeline(), &pipeline));
    }

    #[test]
    fn stages_can_be_restricted() {
        let stages = StageFlags {
            user_input: false,
            ..StageFlags::default()
        };
        let rail = GuardrailRail::new(Arc::new(GuardrailPipeline::new())).with_stages(stages);
        assert!(!rail.stages().user_input);
        assert!(rail.stages().prompts);
    }

    #[tokio::test]
    async fn empty_pipeline_and_context_allow_every_hook() {
        let rail = GuardrailRail::new(Arc::new(GuardrailPipeline::new()));
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        rail.before_invoke(&mut ctx).await.unwrap();
        rail.after_model_call(&mut ctx).await.unwrap();
        rail.before_tool_call(&mut ctx).await.unwrap();
        rail.after_tool_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.interrupts, Vec::new());
    }

    #[tokio::test]
    async fn before_invoke_blocks_the_user_input() {
        let rail = rail(Blocking);
        let mut ctx = invoke_ctx("hello");
        let error = rail.before_invoke(&mut ctx).await.unwrap_err();
        assert_blocked(&error, "user input");
    }

    #[tokio::test]
    async fn after_model_call_blocks_the_model_output() {
        let rail = rail(Blocking);
        let mut ctx = model_ctx("bad");
        let error = rail.after_model_call(&mut ctx).await.unwrap_err();
        assert_blocked(&error, "model output");
    }

    #[tokio::test]
    async fn before_tool_call_blocks_the_tool_arguments() {
        let rail = rail(Blocking);
        let mut ctx = tool_call_ctx(json!({"cmd": "rm -rf /"}));
        let error = rail.before_tool_call(&mut ctx).await.unwrap_err();
        assert_blocked(&error, "tool input");
    }

    #[tokio::test]
    async fn before_tool_call_scans_unparseable_arguments_as_text() {
        let rail = rail(Blocking);
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        ctx.tool_call = Some(ToolCall {
            arguments: "{not json".to_owned(),
            ..ToolCall::new("bash")
        });
        let error = rail.before_tool_call(&mut ctx).await.unwrap_err();
        assert_blocked(&error, "tool input");
    }

    #[tokio::test]
    async fn after_tool_call_blocks_the_tool_output() {
        let rail = rail(Blocking);
        let mut ctx = tool_result_ctx(Some(json!("out")), "out");
        let error = rail.after_tool_call(&mut ctx).await.unwrap_err();
        assert_blocked(&error, "tool output");
    }

    #[tokio::test]
    async fn after_tool_call_scans_the_rendered_text_without_a_value() {
        let rail = rail(Blocking);
        let mut ctx = tool_result_ctx(None, "out");
        let error = rail.after_tool_call(&mut ctx).await.unwrap_err();
        assert_blocked(&error, "tool output");
    }

    #[tokio::test]
    async fn passing_guardrail_allows_every_hook_and_leaves_the_context_alone() {
        let rail = rail(Passing);
        let mut ctx = invoke_ctx("hello");
        ctx.messages = vec![BaseMessage::user("hello")];
        rail.before_invoke(&mut ctx).await.unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        rail.after_model_call(&mut ctx).await.unwrap();
        rail.before_tool_call(&mut ctx).await.unwrap();
        rail.after_tool_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages, vec![BaseMessage::user("hello")]);
        assert_eq!(
            ctx.input.as_ref().and_then(AgentInput::query_text),
            Some("hello")
        );
        assert_eq!(ctx.interrupts, Vec::new());
        assert!(ctx.force_finish.is_none());
        assert!(ctx.skip_tool.is_none());
    }

    #[tokio::test]
    async fn disabled_stages_skip_the_scan() {
        let no_input = StageFlags {
            user_input: false,
            ..StageFlags::default()
        };
        let mut ctx = invoke_ctx("hello");
        rail_with(Blocking, no_input)
            .before_invoke(&mut ctx)
            .await
            .unwrap();

        let no_model_output = StageFlags {
            model_outputs: false,
            ..StageFlags::default()
        };
        let mut ctx = model_ctx("bad");
        rail_with(Blocking, no_model_output)
            .after_model_call(&mut ctx)
            .await
            .unwrap();

        let no_tool_inputs = StageFlags {
            tool_inputs: false,
            ..StageFlags::default()
        };
        let mut ctx = tool_call_ctx(json!({"cmd": "rm -rf /"}));
        rail_with(Blocking, no_tool_inputs)
            .before_tool_call(&mut ctx)
            .await
            .unwrap();

        let no_tool_outputs = StageFlags {
            tool_outputs: false,
            ..StageFlags::default()
        };
        let mut ctx = tool_result_ctx(Some(json!("out")), "out");
        rail_with(Blocking, no_tool_outputs)
            .after_tool_call(&mut ctx)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn missing_or_empty_context_fields_skip_the_scan() {
        let rail = rail(Blocking);
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        rail.before_invoke(&mut ctx).await.unwrap();
        ctx.input = Some(AgentInput::from_text(""));
        rail.before_invoke(&mut ctx).await.unwrap();

        rail.after_model_call(&mut ctx).await.unwrap();
        ctx.response = Some(ModelReply::default());
        rail.after_model_call(&mut ctx).await.unwrap();

        rail.before_tool_call(&mut ctx).await.unwrap();
        rail.after_tool_call(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn before_model_call_does_not_scan_the_prompt() {
        let rail = rail(Blocking);
        let mut ctx = RailContext::new(RailEvent::BeforeModelCall);
        ctx.messages = vec![BaseMessage::user("bad")];
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages, vec![BaseMessage::user("bad")]);
    }
}
