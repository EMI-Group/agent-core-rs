//! The guardrail abstraction and the backend adapter.
//!
//! Ports `openjiuwen/core/security/guardrail/guardrail.py`: Python's
//! `BaseGuardrail` operates on untyped `*args/**kwargs` extracted from a
//! callback event, here a [`Guardrail`] receives a typed
//! [`GuardrailContext`] instead.
//!
//! Python builds a guardrail with a backend and fails at *call* time with
//! `ValueError("No backend configured for ...")` when it has none. Rust has no
//! null backend, so that mistake is a construction error: a
//! [`BackendGuardrail`] always owns one. A backend that fails at call time
//! surfaces through the pipeline's failure policy like every other guardrail
//! error.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;

use crate::{GuardrailContext, GuardrailResult, RiskAssessment, ScanStage};

/// Priority used by a guardrail that does not choose one.
pub const DEFAULT_PRIORITY: i32 = 100;

/// One content-safety check over a [`GuardrailContext`].
///
/// Implementations are object-safe and `Send + Sync`, so a pipeline stores them
/// as `Arc<dyn Guardrail>`. The pipeline runs guardrails in descending
/// [`Guardrail::priority`] order and only those whose [`Guardrail::handles`]
/// returns `true` for the context's stage.
#[async_trait]
pub trait Guardrail: Send + Sync {
    /// Stable name used in findings, errors, and reports.
    fn name(&self) -> &str;

    /// Ordering weight; higher runs earlier. Ties keep registration order.
    fn priority(&self) -> i32 {
        DEFAULT_PRIORITY
    }

    /// The stages this guardrail is interested in; empty means every stage.
    fn stages(&self) -> &[ScanStage] {
        &[]
    }

    /// Whether this guardrail runs at `stage`.
    fn handles(&self, stage: ScanStage) -> bool {
        self.stages().is_empty() || self.stages().contains(&stage)
    }

    /// Inspects the context and returns a verdict.
    ///
    /// # Errors
    ///
    /// Any failure is treated as a *guardrail* failure, not a finding: the
    /// pipeline records it and applies its [`crate::FailurePolicy`].
    async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult>;
}

/// A moderation service or local model that assessments risk.
///
/// This is the seam for vendor adapters: the backend owns transport and model
/// inference, the surrounding [`BackendGuardrail`] adapts it to [`Guardrail`].
#[async_trait]
pub trait GuardrailBackend: Send + Sync {
    /// Stable name of the backend, mirrored by its [`BackendGuardrail`].
    fn name(&self) -> &str;

    /// Classifies the context.
    ///
    /// # Errors
    ///
    /// Transport, decoding, and model failures, reported to the pipeline as a
    /// guardrail failure.
    async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment>;
}

/// Adapts a [`GuardrailBackend`] into a [`Guardrail`].
///
/// The guardrail name is independent of the backend name so the same backend
/// can be registered twice under different names and stages.
pub struct BackendGuardrail {
    name: String,
    backend: Arc<dyn GuardrailBackend>,
    priority: i32,
    stages: Vec<ScanStage>,
}

impl BackendGuardrail {
    /// Wraps `backend` under `name`, at [`DEFAULT_PRIORITY`] and every stage.
    #[must_use]
    pub fn new(name: impl Into<String>, backend: Arc<dyn GuardrailBackend>) -> Self {
        Self {
            name: name.into(),
            backend,
            priority: DEFAULT_PRIORITY,
            stages: Vec::new(),
        }
    }

    /// Overrides the pipeline priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Restricts the guardrail to `stages`; empty scans every stage.
    #[must_use]
    pub fn with_stages(mut self, stages: Vec<ScanStage>) -> Self {
        self.stages = stages;
        self
    }

    /// The wrapped backend.
    #[must_use]
    pub fn backend(&self) -> &Arc<dyn GuardrailBackend> {
        &self.backend
    }
}

impl fmt::Debug for BackendGuardrail {
    /// Prints the name, priority, and stages; the backend has no `Debug` bound.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendGuardrail")
            .field("name", &self.name)
            .field("backend", &self.backend.name())
            .field("priority", &self.priority)
            .field("stages", &self.stages)
            .finish()
    }
}

#[async_trait]
impl Guardrail for BackendGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn stages(&self) -> &[ScanStage] {
        &self.stages
    }

    async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult> {
        self.backend
            .analyze(ctx)
            .await
            .map(RiskAssessment::into_result)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{GuardrailContentType, RiskLevel};

    struct FakeBackend {
        assessment: RiskAssessment,
    }

    #[async_trait]
    impl GuardrailBackend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }

        async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment> {
            assert_eq!(ctx.content_type, GuardrailContentType::Text);
            Ok(self.assessment.clone())
        }
    }

    fn risky_backend() -> Arc<FakeBackend> {
        Arc::new(FakeBackend {
            assessment: RiskAssessment {
                has_risk: true,
                risk_level: RiskLevel::High,
                risk_type: Some("toxicity".to_owned()),
                confidence: 0.9,
                details: Some(json!({"score": 0.9})),
            },
        })
    }

    struct PlainGuardrail;

    #[async_trait]
    impl Guardrail for PlainGuardrail {
        fn name(&self) -> &'static str {
            "plain"
        }

        async fn check(&self, _ctx: &GuardrailContext) -> Result<GuardrailResult> {
            Ok(GuardrailResult::pass())
        }
    }

    #[tokio::test]
    async fn backend_guardrail_copies_the_assessment() {
        let backend: Arc<dyn GuardrailBackend> = risky_backend();
        let guardrail = BackendGuardrail::new("toxicity", Arc::clone(&backend));
        assert_eq!(guardrail.name(), "toxicity");
        assert_eq!(guardrail.priority(), DEFAULT_PRIORITY);
        assert_eq!(guardrail.backend().name(), "fake");

        let ctx = GuardrailContext::text("hello", ScanStage::UserInput);
        let result = guardrail.check(&ctx).await.unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::High);
        assert_eq!(result.risk_type.as_deref(), Some("toxicity"));
        assert_eq!(result.details, Some(json!({"score": 0.9})));
        assert_eq!(result.modified_data, None);
    }

    #[test]
    fn handles_defaults_to_every_stage() {
        let guardrail = BackendGuardrail::new("g", risky_backend());
        assert_eq!(guardrail.stages().len(), 0);
        for stage in [
            ScanStage::LlmInvokeInput,
            ScanStage::LlmInvokeOutput,
            ScanStage::ToolInvokeInput,
            ScanStage::ToolInvokeOutput,
            ScanStage::UserInput,
            ScanStage::Other("custom".to_owned()),
        ] {
            assert!(guardrail.handles(stage), "empty stages must accept all");
        }
    }

    #[test]
    fn handles_filters_configured_stages() {
        let guardrail = BackendGuardrail::new("g", risky_backend())
            .with_stages(vec![ScanStage::ToolInvokeInput]);
        assert!(guardrail.handles(ScanStage::ToolInvokeInput));
        assert!(!guardrail.handles(ScanStage::UserInput));
        assert_eq!(guardrail.stages(), &[ScanStage::ToolInvokeInput]);
    }

    #[test]
    fn builders_override_priority_and_stages() {
        let guardrail = BackendGuardrail::new("g", risky_backend())
            .with_priority(7)
            .with_stages(vec![ScanStage::LlmInvokeOutput]);
        assert_eq!(guardrail.priority(), 7);
        assert_eq!(guardrail.stages(), &[ScanStage::LlmInvokeOutput]);
        let debug = format!("{guardrail:?}");
        assert!(debug.contains("BackendGuardrail"), "{debug}");
        assert!(debug.contains("stages"), "{debug}");
    }

    #[test]
    fn trait_default_priority_and_stages() {
        assert_eq!(PlainGuardrail.priority(), DEFAULT_PRIORITY);
        assert_eq!(PlainGuardrail.stages().len(), 0);
        assert!(PlainGuardrail.handles(ScanStage::UserInput));
        assert_eq!(DEFAULT_PRIORITY, 100);
    }
}
