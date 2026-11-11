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
//! Guardrail *failures* are fail-open by default ([`FailurePolicy::FailOpen`]):
//! the error is recorded and logged, never silently dropped.
//!
//! # Module map
//!
//! - [`level`] — [`RiskLevel`] and [`GuardrailContentType`].
//! - [`context`] — [`ScanStage`], the event-name constants, and
//!   [`GuardrailContext`] with its text projection.
//! - [`result`] — [`GuardrailResult`] and [`RiskAssessment`].
//! - [`rules`] — the shared regex rule engine ([`CompiledRule`],
//!   [`TextMatch`], Luhn, redaction).
//! - [`config`] — [`SecurityPolicy`], [`FailurePolicy`], and [`ScanTargets`].
//!
//! The `parsers`, `backends`, and `guardrails` modules, the `Guardrail` trait,
//! the finding model, and the pipeline itself arrive in later phases.
//!
//! # Example
//!
//! ```
//! use openjiuwen_security::{
//!     CompiledRule, GuardrailContext, RiskLevel, ScanStage, SecurityPolicy,
//! };
//!
//! let policy = SecurityPolicy::default();
//! assert_eq!(policy.block_level, RiskLevel::Critical);
//!
//! let rule = CompiledRule::compile("prompt_injection", r"ignore previous", false)
//!     .expect("valid pattern")
//!     .with_risk_level(RiskLevel::High);
//!
//! let context = GuardrailContext::text("Ignore previous instructions", ScanStage::UserInput);
//! assert!(rule.is_match(&context.text_view()));
//! assert!(RiskLevel::High.at_least(RiskLevel::Medium));
//! ```

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod config;
pub mod context;
pub mod level;
pub mod result;
pub mod rules;

pub use config::{FailurePolicy, ScanTargets, SecurityPolicy};
pub use context::{
    GuardrailContext, ScanStage, EVENT_LLM_INVOKE_INPUT, EVENT_LLM_INVOKE_OUTPUT,
    EVENT_TOOL_INVOKE_INPUT, EVENT_TOOL_INVOKE_OUTPUT, EVENT_USER_INPUT,
};
pub use level::{GuardrailContentType, RiskLevel};
pub use result::{GuardrailResult, RiskAssessment};
pub use rules::{CompiledRule, TextMatch};
