//! Detection backends.
//!
//! Backends own *how* content is judged; guardrails own *what* to scan and how
//! to react. Ports `openjiuwen/core/security/guardrail/backends.py` (v0.1.20):
//! the rule-based and `LLM` prompt-injection backends, plus the remote and local
//! model backends with the transport seams Python inlined.
//!
//! Every backend implements [`GuardrailBackend`](crate::guardrail::GuardrailBackend)
//! and returns a [`RiskAssessment`]; a backend never decides blocking. The
//! pipeline turns assessments into findings and applies the
//! [`SecurityPolicy`](crate::SecurityPolicy).
//!
//! Backends never log or return the scanned text: diagnostics carry backend and
//! pattern names, stages, levels, and lengths only.
#![allow(clippy::module_name_repetitions)]

use crate::{RiskAssessment, RiskLevel};

pub mod api;
pub mod http;
pub mod llm;
pub mod local;
pub mod rule_based;
pub mod vendors;

pub use api::{ApiModelBackend, ApiModelBackendConfig, ModerationRequest, ModerationTransport};
pub use llm::{LlmPromptInjectionBackend, LlmPromptInjectionBackendConfig};
pub use local::{LocalModelBackend, LocalModelBackendConfig, ModelInfo};
pub use rule_based::{RuleBasedBackendConfig, RuleBasedPromptInjectionBackend};

/// The assessment a backend reports when it finds no risk.
///
/// Every backend short-circuits empty text to `has_risk = false`,
/// [`RiskLevel::Safe`], and `confidence = 1.0` (Python parity), which differs
/// from [`RiskAssessment::default`] only in its zero confidence.
pub(crate) fn safe_assessment() -> RiskAssessment {
    RiskAssessment {
        has_risk: false,
        risk_level: RiskLevel::Safe,
        risk_type: None,
        confidence: 1.0,
        details: None,
    }
}
