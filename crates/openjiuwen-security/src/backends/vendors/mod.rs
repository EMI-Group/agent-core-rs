//! Vendor moderation adapters.
//!
//! Each adapter speaks one vendor's moderation `HTTP` API (request building and
//! response parsing) and reports a [`crate::RiskAssessment`]; all three are
//! `GuardrailBackend` implementations, so they plug into
//! [`BackendGuardrail`](crate::guardrail::BackendGuardrail) and
//! [`GuardrailPipeline`](crate::GuardrailPipeline) like any other backend.
//!
//! No vendor SDK is involved: the adapters use the crate-shared
//! [`super::http`] helpers, so they stay offline-testable against recorded
//! wire fixtures or a loopback server.
//!
//! Adapters never place raw matched text in `details`; vendor fields carrying
//! the offending words (Aliyun `RiskWords`, blocklist terms) are dropped.

pub mod aliyun;
pub mod azure;
pub mod openai;
