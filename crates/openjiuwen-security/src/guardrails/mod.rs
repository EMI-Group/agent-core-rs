//! Built-in guardrails.
//!
//! Every type here is a [`Guardrail`](crate::Guardrail) implementation: it owns
//! a backend, config, priority, and stage list, and is stateless after
//! construction — one instance can serve many concurrent scans. Detection runs
//! in-process through [`Guardrail::check`](crate::Guardrail::check) and returns a
//! [`GuardrailResult`](crate::GuardrailResult); blocking is the pipeline's
//! decision, not the guardrail's.
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

pub use prompt_injection::{PromptInjectionGuardrail, PromptInjectionGuardrailConfig};
