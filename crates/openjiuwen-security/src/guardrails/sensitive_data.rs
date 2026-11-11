//! Sensitive-data/PII detection with redaction.
//!
//! Detection patterns, validators, and the [`SensitiveDataGuardrail`] live here;
//! the guardrail is a [`Guardrail`](crate::Guardrail) implementation whose
//! redaction primitive is reused by tool-argument validation.
