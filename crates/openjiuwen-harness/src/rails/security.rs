//! The content-safety rail.
//!
//! Ports the hook body of `openjiuwen/harness/rails/security/safety_prompt_rail.py`
//! (`SafetyPromptRail`): every user input, prompt, model reply, tool call and tool
//! result passes through an [`openjiuwen_security::GuardrailPipeline`] before the
//! loop moves on. The rail owns no state beyond the pipeline and the
//! [`ScanTargets`] switches, so one instance can serve every invocation.
//!
//! Blocking semantics mirror the Python rail while staying unambiguous in Rust:
//! a blocked user input, prompt, model output or tool output aborts the firing
//! point with [`openjiuwen_common::error::Error::Security`]; a blocked tool
//! input only skips that tool call (the loop turns the skip message into the
//! tool result, so the model can react instead of the whole run dying).
//! Guardrail *failures* (as opposed to findings) follow the pipeline policy:
//! under the default `FailOpen` policy they are logged and the run continues.
//!
//! Never log prompt text, tool arguments/outputs, or memory content: the log
//! lines here carry only guardrail names, risk types, and counts.
use std::fmt;
use std::sync::Arc;

use openjiuwen_agent::InputQuery;
use openjiuwen_common::error::Result;
use openjiuwen_security::{GuardrailOutcome, GuardrailPipeline, ScanTargets, SecurityConfig};
use serde_json::Value;

use super::{rail_priority, HarnessRail, HarnessRailContext};

/// Scans the model-visible traffic of an invocation with a guardrail pipeline.
///
/// Dispatch priority is [`rail_priority::SAFETY_PROMPT`] (85), the slot of the
/// Python `SafetyPromptRail`.
pub struct SecurityRail {
    pipeline: Arc<GuardrailPipeline>,
    targets: ScanTargets,
}

impl SecurityRail {
    /// Builds a rail that runs `pipeline` at the default [`ScanTargets`].
    #[must_use]
    pub fn new(pipeline: GuardrailPipeline) -> Self {
        Self::from_arc(Arc::new(pipeline))
    }

    /// Builds a rail from a shared pipeline, the form a builder stores.
    #[must_use]
    pub fn from_arc(pipeline: Arc<GuardrailPipeline>) -> Self {
        Self {
            pipeline,
            targets: ScanTargets::default(),
        }
    }

    /// Builds a rail from an umbrella [`SecurityConfig`].
    ///
    /// The pipeline is rebuilt from `config`, so detector-configuration
    /// problems surface here instead of at the first scan.
    ///
    /// # Errors
    ///
    /// Propagates the [`openjiuwen_common::error::ErrorKind::Validation`] error
    /// of [`SecurityConfig::pipeline`] when a detector rejects its
    /// configuration (unknown mode, missing endpoint, invalid regular
    /// expression).
    pub fn from_config(config: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            pipeline: Arc::new(config.pipeline()?),
            targets: config.targets,
        })
    }

    /// Returns this rail with `targets` as its scan switches.
    #[must_use]
    pub fn with_targets(mut self, targets: ScanTargets) -> Self {
        self.targets = targets;
        self
    }

    /// The shared guardrail pipeline.
    #[must_use]
    pub fn pipeline(&self) -> &Arc<GuardrailPipeline> {
        &self.pipeline
    }

    /// The enabled scan stages.
    #[must_use]
    pub fn targets(&self) -> ScanTargets {
        self.targets
    }

    /// Records guardrail *failures* when the outcome still allows the call.
    ///
    /// Under the `FailOpen` policy the pipeline records an error instead of
    /// blocking; the rail logs the failing guardrail names (never their
    /// content) and lets the run continue. A blocked outcome is left to the
    /// caller, which converts it into the stage-specific signal.
    fn note_failures(outcome: &GuardrailOutcome, stage: &str) {
        if outcome.errors.is_empty() || outcome.is_blocked() {
            return;
        }
        let guardrails: Vec<&str> = outcome
            .evaluations
            .iter()
            .filter(|evaluation| evaluation.error.is_some())
            .map(|evaluation| evaluation.guardrail.as_str())
            .collect();
        tracing::warn!(
            stage,
            failed = guardrails.len(),
            guardrails = ?guardrails,
            "guardrail evaluation failed; continuing (fail-open)"
        );
    }
}

impl Default for SecurityRail {
    /// An empty pipeline with every scan target enabled: a valid no-op rail.
    fn default() -> Self {
        Self {
            pipeline: Arc::new(GuardrailPipeline::new()),
            targets: ScanTargets::default(),
        }
    }
}

impl fmt::Debug for SecurityRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecurityRail")
            .field("targets", &self.targets)
            .field("guardrails", &self.pipeline.guardrails().len())
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for SecurityRail {
    fn name(&self) -> &'static str {
        "security"
    }

    fn priority(&self) -> i32 {
        rail_priority::SAFETY_PROMPT
    }

    async fn before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.targets.user_input {
            return Ok(());
        }
        let Some(text) = ctx
            .base
            .input
            .as_ref()
            .and_then(|input| match &input.query {
                InputQuery::Text(text) if !text.trim().is_empty() => Some(text.clone()),
                _ => None,
            })
        else {
            return Ok(());
        };
        let outcome = self.pipeline.scan_user_input(text).await;
        Self::note_failures(&outcome, "user_input");
        outcome.enforce()
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.targets.prompts {
            return Ok(());
        }
        let outcome = self.pipeline.scan_prompt(&ctx.base.messages).await;
        Self::note_failures(&outcome, "prompt");
        outcome.enforce()
    }

    async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.targets.model_outputs {
            return Ok(());
        }
        let Some(response) = ctx.base.response.as_ref() else {
            return Ok(());
        };
        let text = response.message.content.to_plain_text();
        let outcome = self.pipeline.scan_model_output(text).await;
        Self::note_failures(&outcome, "model_output");
        outcome.enforce()
    }

    async fn before_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.targets.tool_inputs {
            return Ok(());
        }
        let Some(call) = ctx.base.tool_call.as_ref() else {
            return Ok(());
        };
        let arguments = serde_json::from_str::<Value>(&call.arguments)
            .unwrap_or_else(|_| Value::String(call.arguments.clone()));
        let outcome = self
            .pipeline
            .scan_tool_input(call.name.clone(), &arguments)
            .await;
        Self::note_failures(&outcome, "tool_input");
        if outcome.is_blocked() {
            let risk_type = outcome
                .violation()
                .map_or_else(|| "unknown".to_string(), |violation| violation.risk_type);
            ctx.request_skip_tool(format!("blocked by the security rail ({risk_type})"));
        }
        Ok(())
    }

    async fn after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.targets.tool_outputs {
            return Ok(());
        }
        let Some(result) = ctx.base.tool_result.as_ref() else {
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
        Self::note_failures(&outcome, "tool_output");
        outcome.enforce()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::sync::atomic::{AtomicUsize, Ordering};

    use openjiuwen_agent::react::model_step::ModelReply;
    use openjiuwen_agent::{AgentInput, RailEvent, ToolCallResult};
    use openjiuwen_common::error::{Error, ErrorKind};
    use openjiuwen_foundation::{BaseMessage, ToolCall};
    use openjiuwen_security::{Guardrail, GuardrailContext, GuardrailResult, RiskLevel, ScanStage};
    use serde_json::json;

    use super::*;

    /// What the scripted guardrail answers.
    enum Verdict {
        Pass,
        Block(&'static str),
        Fail,
    }

    /// Counts checks so a test can prove the pipeline was (not) consulted.
    struct ScriptedGuardrail {
        stages: Vec<ScanStage>,
        verdict: Verdict,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Guardrail for ScriptedGuardrail {
        fn name(&self) -> &'static str {
            "scripted"
        }

        fn stages(&self) -> &[ScanStage] {
            &self.stages
        }

        async fn check(&self, _ctx: &GuardrailContext) -> Result<GuardrailResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.verdict {
                Verdict::Pass => Ok(GuardrailResult::pass()),
                Verdict::Block(risk_type) => Ok(GuardrailResult::block(
                    RiskLevel::Critical,
                    risk_type,
                    None,
                    None,
                )),
                Verdict::Fail => Err(Error::security("scripted guardrail unavailable")),
            }
        }
    }

    fn scripted(stages: Vec<ScanStage>, verdict: Verdict) -> (Arc<SecurityRail>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(ScriptedGuardrail {
            stages,
            verdict,
            calls: calls.clone(),
        });
        (Arc::new(SecurityRail::new(pipeline)), calls)
    }

    fn input(text: &str) -> AgentInput {
        AgentInput {
            query: InputQuery::Text(text.to_string()),
            ..AgentInput::default()
        }
    }

    #[test]
    fn name_priority_and_defaults() {
        let rail = SecurityRail::default();
        assert_eq!(rail.name(), "security");
        assert_eq!(rail.priority(), rail_priority::SAFETY_PROMPT);
        assert_eq!(rail.priority(), 85);
        assert!(rail.pipeline().guardrails().is_empty());
        assert_eq!(rail.targets(), ScanTargets::default());
    }

    #[test]
    fn from_config_builds_the_pipeline_and_keeps_the_targets() {
        let mut config = SecurityConfig::protected();
        config.targets.model_outputs = false;
        let rail = SecurityRail::from_config(&config).unwrap();
        assert!(!rail.targets().model_outputs);
        assert!(!rail.pipeline().guardrails().is_empty());
    }

    #[tokio::test]
    async fn user_input_block_is_a_security_error() {
        let (rail, calls) = scripted(vec![ScanStage::UserInput], Verdict::Block("jailbreak"));
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.input = Some(input("scanned secret text"));

        let error = rail.before_invoke(&mut ctx).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(!error.to_string().contains("scanned secret text"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn blank_user_input_skips_the_pipeline() {
        let (rail, calls) = scripted(vec![ScanStage::UserInput], Verdict::Block("jailbreak"));
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.input = Some(input("   "));
        rail.before_invoke(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        rail.before_invoke(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn prompt_block_is_a_security_error() {
        let (rail, calls) = scripted(
            vec![ScanStage::LlmInvokeInput],
            Verdict::Block("prompt_injection"),
        );
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = vec![
            BaseMessage::system("system instructions"),
            BaseMessage::user("scanned prompt text"),
        ];

        let error = rail.before_model_call(&mut ctx).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(!error.to_string().contains("scanned prompt text"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn model_output_block_is_a_security_error() {
        let (rail, calls) = scripted(
            vec![ScanStage::LlmInvokeOutput],
            Verdict::Block("unsafe_output"),
        );
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        ctx.base.response = Some(ModelReply {
            message: BaseMessage::assistant("scanned model text"),
            ..ModelReply::default()
        });

        let error = rail.after_model_call(&mut ctx).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(!error.to_string().contains("scanned model text"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn missing_response_skips_the_model_output_scan() {
        let (rail, calls) = scripted(
            vec![ScanStage::LlmInvokeOutput],
            Verdict::Block("unsafe_output"),
        );
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        rail.after_model_call(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn tool_input_block_skips_the_tool_instead_of_failing() {
        let (rail, calls) = scripted(
            vec![ScanStage::ToolInvokeInput],
            Verdict::Block("dangerous_tool"),
        );
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeToolCall);
        ctx.base.tool_call =
            Some(ToolCall::new("bash").with_arguments(json!({"command": "scanned arguments"})));

        rail.before_tool_call(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let skip = ctx.base.take_skip_tool().unwrap();
        assert!(skip.contains("dangerous_tool"));
        assert!(!skip.contains("scanned arguments"));
    }

    #[tokio::test]
    async fn invalid_tool_arguments_fall_back_to_a_string_value() {
        let (rail, calls) = scripted(vec![ScanStage::ToolInvokeInput], Verdict::Pass);
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeToolCall);
        let mut call = ToolCall::new("bash");
        call.arguments = "not-json".to_string();
        ctx.base.tool_call = Some(call);

        rail.before_tool_call(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(ctx.base.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn tool_output_block_is_a_security_error() {
        let (rail, calls) = scripted(
            vec![ScanStage::ToolInvokeOutput],
            Verdict::Block("leaked_secret"),
        );
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterToolCall);
        ctx.base.tool_result = Some(ToolCallResult::success(
            ToolCall::new("read_file"),
            json!({"content": "scanned tool output"}),
            "scanned tool output",
        ));

        let error = rail.after_tool_call(&mut ctx).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(!error.to_string().contains("scanned tool output"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn tool_output_without_a_value_scans_the_rendered_text() {
        let (rail, calls) = scripted(vec![ScanStage::ToolInvokeOutput], Verdict::Pass);
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterToolCall);
        ctx.base.tool_result = Some(ToolCallResult::failure(
            ToolCall::new("read_file"),
            "text-only failure",
        ));
        rail.after_tool_call(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn disabled_targets_never_call_the_pipeline() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(ScriptedGuardrail {
            stages: vec![
                ScanStage::UserInput,
                ScanStage::LlmInvokeInput,
                ScanStage::LlmInvokeOutput,
                ScanStage::ToolInvokeInput,
                ScanStage::ToolInvokeOutput,
            ],
            verdict: Verdict::Block("whatever"),
            calls: calls.clone(),
        });
        let rail = SecurityRail::new(pipeline).with_targets(ScanTargets {
            prompts: false,
            user_input: false,
            model_outputs: false,
            tool_inputs: false,
            tool_outputs: false,
        });

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.input = Some(input("text"));
        ctx.base.messages = vec![BaseMessage::user("text")];
        ctx.base.response = Some(ModelReply {
            message: BaseMessage::assistant("text"),
            ..ModelReply::default()
        });
        ctx.base.tool_call = Some(ToolCall::new("bash"));
        ctx.base.tool_result = Some(ToolCallResult::success(
            ToolCall::new("bash"),
            json!({"ok": true}),
            "ok",
        ));

        rail.before_invoke(&mut ctx).await.unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        rail.after_model_call(&mut ctx).await.unwrap();
        rail.before_tool_call(&mut ctx).await.unwrap();
        rail.after_tool_call(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(ctx.base.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn failing_guardrail_is_swallowed_under_fail_open() {
        let (rail, calls) = scripted(vec![ScanStage::UserInput], Verdict::Fail);
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.input = Some(input("scanned secret text"));

        rail.before_invoke(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_missing_input_leaves_before_invoke_alone() {
        let (rail, calls) = scripted(vec![ScanStage::UserInput], Verdict::Block("jailbreak"));
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        rail.before_invoke(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.input = Some(AgentInput {
            query: InputQuery::default(),
            ..AgentInput::default()
        });
        rail.before_invoke(&mut ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn debug_reports_targets_and_guardrail_count() {
        let rail = SecurityRail::default();
        assert_eq!(
            format!("{rail:?}"),
            "SecurityRail { targets: ScanTargets { prompts: true, user_input: true, model_outputs: true, tool_inputs: true, tool_outputs: true }, guardrails: 0 }"
        );
    }
}
