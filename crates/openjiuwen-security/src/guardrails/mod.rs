//! Built-in guardrails.
//!
//! Every type here is a [`Guardrail`](crate::Guardrail) implementation: it owns
//! a backend, config, priority, and stage list, and is stateless after
//! construction — one instance can serve many concurrent scans. Detection runs
//! in-process through [`Guardrail::check`](crate::Guardrail::check) and returns a
//! [`GuardrailResult`](crate::GuardrailResult); blocking is the pipeline's
//! decision, not the guardrail's.
//!
//! A payload the checked text projection cannot read makes `check` return a
//! typed `Error::validation` instead of a pass: the pipeline's
//! [`FailurePolicy`](crate::FailurePolicy) decides whether that failure blocks
//! (the default) or is only recorded.
//!
//! Modules:
//!
//! - [`prompt_injection`] — Python's `PromptInjectionGuardrail`: rule-based,
//!   remote-model, or local-model prompt-injection detection bound to the model
//!   input and tool-output stages.
//! - [`sensitive_data`] — PII/secret detection with redaction.
//! - [`content_rules`] — ordered user-supplied regex rules.
//! - [`tool_arguments`] — tool allow/deny lists, argument patterns, and PII in
//!   tool-call arguments.

pub mod content_rules;
pub mod prompt_injection;
pub mod sensitive_data;
pub mod tool_arguments;

pub use content_rules::{ContentRule, ContentRuleGuardrail, ContentRuleGuardrailConfig};
pub use prompt_injection::{PromptInjectionGuardrail, PromptInjectionGuardrailConfig};
pub use sensitive_data::{
    RedactionOutcome, SensitiveDataGuardrail, SensitiveDataGuardrailConfig, SensitiveMatch,
    SensitivePattern, SensitiveValidator,
};
pub use tool_arguments::{ToolArgumentGuardrail, ToolArgumentGuardrailConfig};
