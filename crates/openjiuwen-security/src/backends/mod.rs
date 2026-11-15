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
//! Two rules keep a broken detector from reading as "clean":
//!
//! - the payload is projected with the *checked*
//!   [`GuardrailContext::text_view_checked`](crate::GuardrailContext::text_view_checked),
//!   so an unscannable payload is an [`Error`](openjiuwen_common::error::Error)
//!   rather than a pass;
//! - an unusable reply (an empty body, a non-JSON body, a non-object body, an
//!   unrecognized vendor schema, or an invalid judge verdict) is an error too,
//!   which under the default
//!   [`FailurePolicy::FailClosed`](crate::FailurePolicy::FailClosed) becomes a
//!   blocking `Critical` finding.
//!
//! Only a blank *scannable* payload short-circuits to `safe_assessment`.
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
pub use http::{HttpAuth, HttpModerationConfig, HttpModerationTransport};
pub use llm::{LlmPromptInjectionBackend, LlmPromptInjectionBackendConfig};
pub use local::{LocalModelBackend, LocalModelBackendConfig, ModelInfo, SUPPORTED_DEVICES};
pub use rule_based::{RuleBasedBackendConfig, RuleBasedPromptInjectionBackend};
pub use vendors::{
    AliyunGreenAuth, AliyunGreenBackend, AliyunGreenConfig, AzureContentSafetyBackend,
    AzureContentSafetyConfig, OpenAiModerationBackend, OpenAiModerationConfig,
};

/// The assessment a backend reports when there is nothing to scan.
///
/// This is the verdict for a **blank scannable** payload only: the checked text
/// projection ([`GuardrailContext::text_view_checked`](crate::GuardrailContext::text_view_checked))
/// succeeded and produced nothing but whitespace (the empty string, or an empty
/// message array). It reports `has_risk = false`, [`RiskLevel::Safe`], and
/// `confidence = 1.0`, which differs from [`RiskAssessment::default`] only in
/// its zero confidence.
///
/// It is never the answer to a failure: a payload the projection cannot read
/// (see `text_view_checked`) and a backend that cannot reach a verdict both
/// return [`Error`](openjiuwen_common::error::Error) instead, which the pipeline
/// resolves through its
/// [`FailurePolicy`](crate::FailurePolicy) (blocking by default). Callers must
/// therefore never use this value to paper over an error.
pub(crate) fn safe_assessment() -> RiskAssessment {
    RiskAssessment {
        has_risk: false,
        risk_level: RiskLevel::Safe,
        risk_type: None,
        confidence: 1.0,
        details: None,
    }
}
