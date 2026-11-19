//! Guardrails and the ordered content-safety pipeline for the openJiuwen
//! runtime.
//!
//! Ports the Python package `openjiuwen/core/security/guardrail/` (v0.1.20) and
//! extends it with the detectors Python advertises but never shipped. This is a
//! self-contained layer-2 crate: it owns the models, the guardrail abstraction,
//! the pipeline, the parsers, and the rule engine, and never imports
//! `openjiuwen-runner`, so any caller can run a scan in-process.
//!
//! # Security model
//!
//! Detection is two-tier, exactly like Python: a guardrail returns a
//! [`GuardrailResult`], and only findings at or above
//! [`SecurityPolicy::block_level`] (default [`RiskLevel::Critical`]) block the
//! caller. Lower severities are recorded as findings and execution continues.
//! Guardrail *failures* are fail-closed by default
//! ([`FailurePolicy::FailClosed`]): a detector error, an unrecognized backend
//! reply, or a payload no detector can read becomes a blocking finding whose
//! message names the reason. [`FailurePolicy::FailOpen`] — Python's behavior, in
//! which the callback framework swallows guardrail errors — is an explicit
//! opt-in for callers that prefer availability over detection.
//!
//! A run produces one [`GuardrailOutcome`]: the evaluations, the findings, the
//! decision, and the merged sanitized content. [`GuardrailOutcome::enforce`]
//! turns a blocked outcome into an [`openjiuwen_common::error::Error`];
//! [`GuardrailOutcome::redacted`] is the content-free view that is safe to log.
//!
//! # Module map
//!
//! - [`level`] — [`RiskLevel`] and [`GuardrailContentType`].
//! - [`context`] — [`ScanStage`], the event-name constants, and
//!   [`GuardrailContext`] with its text projection.
//! - [`result`] — [`GuardrailResult`] and [`RiskAssessment`].
//! - [`finding`] — [`Finding`], [`GuardrailEvaluation`], [`SecurityDecision`],
//!   [`GuardrailOutcome`], and [`GuardrailViolation`].
//! - [`guardrail`] — the [`Guardrail`] and [`GuardrailBackend`] traits plus
//!   [`BackendGuardrail`].
//! - [`pipeline`] — the ordered [`GuardrailPipeline`] and its scan helpers.
//! - [`report`] — [`RedactedOutcome`].
//! - [`config`] — [`SecurityConfig`], [`SecurityPolicy`], [`FailurePolicy`], and
//!   [`ScanTargets`].
//! - [`rules`] — the shared regex rule engine ([`CompiledRule`],
//!   [`TextMatch`], Luhn, redaction).
//! - [`parsers`] — [`ModelOutputParser`] and the built-in
//!   [`BertBinaryParser`] and [`QwenGuardParser`] with their [`Thresholds`].
//! - [`backends`] — detection backends
//!   ([`RuleBasedPromptInjectionBackend`], [`LlmPromptInjectionBackend`],
//!   [`ApiModelBackend`], [`LocalModelBackend`]) with their configs and the
//!   [`ModerationTransport`] seam.
//! - [`guardrails`] — the built-in detectors ([`PromptInjectionGuardrail`],
//!   [`SensitiveDataGuardrail`], [`ContentRuleGuardrail`],
//!   [`ToolArgumentGuardrail`]) with their configs.
//!
//! # Example
//!
//! ```
//! use async_trait::async_trait;
//! use openjiuwen_common::error::Result;
//! use openjiuwen_security::{
//!     Guardrail, GuardrailContext, GuardrailPipeline, GuardrailResult, RiskLevel,
//! };
//!
//! struct ContainsWord(&'static str);
//!
//! #[async_trait]
//! impl Guardrail for ContainsWord {
//!     fn name(&self) -> &str {
//!         "contains_word"
//!     }
//!
//!     async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult> {
//!         // The checked projection: a payload no detector can read is a typed
//!         // failure for the pipeline's failure policy, never a silent pass.
//!         let text = ctx.text_view_checked()?;
//!         if text.contains(self.0) {
//!             Ok(GuardrailResult::block(
//!                 RiskLevel::High,
//!                 "banned_word",
//!                 None,
//!                 None,
//!             ))
//!         } else {
//!             Ok(GuardrailResult::pass())
//!         }
//!     }
//! }
//!
//! let runtime = tokio::runtime::Builder::new_current_thread()
//!     .enable_all()
//!     .build()
//!     .expect("a tokio runtime");
//!
//! let mut pipeline = GuardrailPipeline::new();
//! pipeline.add(ContainsWord("secret"));
//!
//! let outcome = runtime.block_on(pipeline.scan_user_input("all clear"));
//! assert!(outcome.is_safe());
//! assert!(outcome.findings.is_empty());
//!
//! let flagged = runtime.block_on(pipeline.scan_user_input("a secret plan"));
//! assert!(flagged.is_safe(), "HIGH findings do not block by default");
//! assert_eq!(flagged.risk_level, RiskLevel::High);
//! assert_eq!(flagged.findings.len(), 1);
//! ```

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod backends;
pub mod config;
pub mod context;
pub mod finding;
pub mod guardrail;
pub mod guardrails;
pub mod level;
pub mod parsers;
pub mod pipeline;
pub mod report;
pub mod result;
pub mod rules;

pub use backends::{
    AliyunGreenAuth, AliyunGreenBackend, AliyunGreenConfig, ApiModelBackend, ApiModelBackendConfig,
    AzureContentSafetyBackend, AzureContentSafetyConfig, HttpAuth, HttpModerationConfig,
    HttpModerationTransport, LlmPromptInjectionBackend, LlmPromptInjectionBackendConfig,
    LocalModelBackend, LocalModelBackendConfig, ModelInfo, ModerationRequest, ModerationTransport,
    OpenAiModerationBackend, OpenAiModerationConfig, RuleBasedBackendConfig,
    RuleBasedPromptInjectionBackend, SUPPORTED_DEVICES,
};
pub use config::{FailurePolicy, ScanTargets, SecurityConfig, SecurityPolicy};
pub use context::{
    GuardrailContext, ScanStage, EVENT_LLM_INVOKE_INPUT, EVENT_LLM_INVOKE_OUTPUT,
    EVENT_TOOL_INVOKE_INPUT, EVENT_TOOL_INVOKE_OUTPUT, EVENT_USER_INPUT,
};
pub use finding::{
    Finding, GuardrailEvaluation, GuardrailOutcome, GuardrailViolation, SecurityDecision,
};
pub use guardrail::{BackendGuardrail, Guardrail, GuardrailBackend, DEFAULT_PRIORITY};
pub use guardrails::{
    ContentRule, ContentRuleGuardrail, ContentRuleGuardrailConfig, PromptInjectionGuardrail,
    PromptInjectionGuardrailConfig, RedactionOutcome, SensitiveDataGuardrail,
    SensitiveDataGuardrailConfig, SensitiveMatch, SensitivePattern, SensitiveValidator,
    ToolArgumentGuardrail, ToolArgumentGuardrailConfig,
};
pub use level::{GuardrailContentType, RiskLevel};
pub use parsers::{BertBinaryParser, ModelOutputParser, QwenGuardParser, Thresholds};
pub use pipeline::GuardrailPipeline;
pub use report::RedactedOutcome;
pub use result::{GuardrailResult, RiskAssessment};
pub use rules::{CompiledRule, TextMatch};
