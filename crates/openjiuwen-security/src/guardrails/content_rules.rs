//! Configurable regex content rules.
//!
//! Ports Python's rule-backend semantics into a standalone [`ContentRuleGuardrail`]:
//! an ordered rule table where the first matching rule decides the verdict.
