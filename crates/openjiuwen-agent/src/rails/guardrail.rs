//! The security rail: scans user input, model output, and tool traffic through
//! a [`GuardrailPipeline`].
//!
//! Python parity: the guardrail hook of the harness rails, re-expressed on the
//! kernel [`AgentRail`] seam.
//!
//! Intended behaviour (hooked in a later round):
//!
//! - `BeforeInvoke` scans the user input (`scan_user_input`) when
//!   `stages.user_input` is enabled.
//! - `BeforeModelCall` scans the prepared prompt messages (`scan_prompt`) when
//!   `stages.prompts` is enabled.
//! - `AfterModelCall` scans the model output (`scan_model_output`) when
//!   `stages.model_outputs` is enabled.
//! - `BeforeToolCall` scans the tool arguments (`scan_tool_input`) and
//!   `AfterToolCall` scans the tool result (`scan_tool_output`), gated by
//!   `stages.tool_inputs` / `stages.tool_outputs`.
//!
//! Every scan finishes with
//! [`GuardrailOutcome::enforce`](openjiuwen_security::GuardrailOutcome::enforce),
//! so a blocking verdict surfaces as
//! [`Error::Security`](openjiuwen_common::error::Error::security) and aborts the
//! run.
use super::{AgentRail, RailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_security::GuardrailPipeline;
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
#[async_trait]
impl AgentRail for GuardrailRail {
    fn name(&self) -> &str {
        &self.name
    }
    /// TODO: scan the user input with `scan_user_input` when
    /// `stages.user_input`, then `enforce()`.
    async fn before_invoke(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
    /// TODO: scan `ctx.messages` with `scan_prompt` when `stages.prompts`, then
    /// `enforce()`.
    async fn before_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
    /// TODO: scan the reply text with `scan_model_output` when
    /// `stages.model_outputs`, then `enforce()`.
    async fn after_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
    /// TODO: scan the parsed arguments with `scan_tool_input` when
    /// `stages.tool_inputs`, then `enforce()`.
    async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
    /// TODO: scan the tool result with `scan_tool_output` when
    /// `stages.tool_outputs`, then `enforce()`.
    async fn after_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rails::RailEvent;

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
    async fn hooks_are_placeholders_that_allow_the_run() {
        let rail = GuardrailRail::new(Arc::new(GuardrailPipeline::new()));
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        rail.before_invoke(&mut ctx).await.unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        rail.after_model_call(&mut ctx).await.unwrap();
        rail.before_tool_call(&mut ctx).await.unwrap();
        rail.after_tool_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.interrupts, Vec::new());
    }
}
