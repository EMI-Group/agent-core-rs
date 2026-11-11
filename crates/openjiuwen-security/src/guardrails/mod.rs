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
//!
//! Detectors that are not implemented yet are declared by their own files when
//! they land; this module only re-exports what exists.

pub mod prompt_injection;

pub use prompt_injection::{PromptInjectionGuardrail, PromptInjectionGuardrailConfig};
