//! Local-model backend seam.
//!
//! Ports `LocalModelBackend` and `LocalModelBackendConfig` from
//! `openjiuwen/core/security/guardrail/backends.py` (v0.1.20). Python lazily
//! imports `torch`/`transformers` and runs the classifier in-process; this crate
//! ships no inference runtime, so configuration and introspection are preserved
//! (`model_info` reports a model that is never loaded) and `analyze` returns
//! [`Error::unsupported`] after the empty-text short-circuit.
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::safe_assessment;
use crate::guardrail::GuardrailBackend;
use crate::parsers::ModelOutputParser;
use crate::{GuardrailContext, RiskAssessment};

/// Python's default device (`device: str = "auto"`).
const DEFAULT_DEVICE: &str = "auto";

/// Python's default risk type (`risk_type: str = "model_detection"`).
const DEFAULT_RISK_TYPE: &str = "model_detection";

/// Stable backend name, mirrored in findings and logs.
const NAME: &str = "local_model";

/// Configuration for [`LocalModelBackend`].
///
/// Field names and defaults mirror Python's `LocalModelBackendConfig`:
/// `model_path`, `device` (`"auto"`, one of `auto`/`cpu`/`cuda` in Python), and
/// `risk_type` (`"model_detection"`). Python's `parser` field is not part of the
/// serialized config; inject a parser with [`LocalModelBackend::with_parser`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct LocalModelBackendConfig {
    /// Filesystem path of the local classifier.
    pub model_path: String,
    /// Device the classifier would run on (`"auto"` by default).
    pub device: String,
    /// Risk name reported by the parser's assessment.
    pub risk_type: String,
}

impl Default for LocalModelBackendConfig {
    /// The Python dataclass defaults: empty path, `"auto"`, `"model_detection"`.
    fn default() -> Self {
        Self {
            model_path: String::new(),
            device: DEFAULT_DEVICE.to_owned(),
            risk_type: DEFAULT_RISK_TYPE.to_owned(),
        }
    }
}

/// Load state of the local model, mirroring Python's introspection fields.
///
/// A [`LocalModelBackend`] in this crate never loads anything, so every field is
/// `false`; the type exists so callers written against the Python API can still
/// ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "snake_case")]
pub struct ModelInfo {
    /// Whether the model has been loaded into memory.
    pub model_loaded: bool,
    /// Whether a model object is present.
    pub has_model: bool,
    /// Whether a tokenizer is present.
    pub has_tokenizer: bool,
}

/// Backend that would run a moderation model locally.
///
/// The inference runtime is deferred: [`LocalModelBackend::analyze`] only
/// short-circuits empty text and otherwise reports that the runtime is missing.
/// Configuration and an optional [`ModelOutputParser`] are preserved so the
/// adapter can be completed without an API change.
pub struct LocalModelBackend {
    config: LocalModelBackendConfig,
    parser: Option<Arc<dyn ModelOutputParser>>,
}

impl LocalModelBackend {
    /// A backend with `cfg`, no parser, and no loaded model.
    #[must_use]
    pub fn new(cfg: LocalModelBackendConfig) -> Self {
        Self {
            config: cfg,
            parser: None,
        }
    }

    /// Uses `parser` once a model runtime exists instead of the default one.
    #[must_use]
    pub fn with_parser(mut self, parser: Arc<dyn ModelOutputParser>) -> Self {
        self.parser = Some(parser);
        self
    }

    /// The load state of the model: nothing is ever loaded here, so all `false`.
    #[must_use]
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo::default()
    }
}

impl fmt::Debug for LocalModelBackend {
    /// Prints the configuration plus parser presence; `dyn ModelOutputParser`
    /// has no `Debug` bound.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalModelBackend")
            .field("model_path", &self.config.model_path)
            .field("device", &self.config.device)
            .field("risk_type", &self.config.risk_type)
            .field("has_parser", &self.parser.is_some())
            .finish()
    }
}

#[async_trait]
impl GuardrailBackend for LocalModelBackend {
    fn name(&self) -> &str {
        NAME
    }

    async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment> {
        let text = ctx.text_view();
        if text.trim().is_empty() {
            return Ok(safe_assessment());
        }
        Err(Error::unsupported(
            "local model backend requires an inference runtime (deferred adapter)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::parsers::BertBinaryParser;
    use crate::{RiskLevel, ScanStage};

    fn ctx(text: &str) -> GuardrailContext {
        GuardrailContext::text(text, ScanStage::LlmInvokeInput)
    }

    #[test]
    fn config_defaults_match_python() {
        let default = LocalModelBackendConfig::default();
        assert_eq!(default.model_path, "");
        assert_eq!(default.device, "auto");
        assert_eq!(default.risk_type, "model_detection");

        let empty: LocalModelBackendConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, default);

        let partial: LocalModelBackendConfig =
            serde_json::from_value(json!({"model_path": "/models/bert", "device": "cuda"}))
                .unwrap();
        assert_eq!(partial.model_path, "/models/bert");
        assert_eq!(partial.device, "cuda");
        assert_eq!(partial.risk_type, DEFAULT_RISK_TYPE);
    }

    #[test]
    fn config_serde_round_trips_and_omits_the_parser() {
        let config = LocalModelBackendConfig {
            model_path: "/models/bert".to_owned(),
            device: "cpu".to_owned(),
            risk_type: "prompt_injection".to_owned(),
        };
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            value,
            json!({
                "model_path": "/models/bert",
                "device": "cpu",
                "risk_type": "prompt_injection",
            })
        );
        let back: LocalModelBackendConfig = serde_json::from_value(value).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn model_info_is_never_loaded() {
        let backend = LocalModelBackend::new(LocalModelBackendConfig::default());
        let info = backend.model_info();
        assert!(!info.model_loaded);
        assert!(!info.has_model);
        assert!(!info.has_tokenizer);
        assert_eq!(info, ModelInfo::default());

        let value = serde_json::to_value(info).unwrap();
        assert_eq!(
            value,
            json!({"model_loaded": false, "has_model": false, "has_tokenizer": false})
        );
        let back: ModelInfo = serde_json::from_value(json!({})).unwrap();
        assert_eq!(back, ModelInfo::default());

        let with_parser = backend.with_parser(Arc::new(BertBinaryParser::new()));
        assert_eq!(with_parser.model_info(), ModelInfo::default());
    }

    #[tokio::test]
    async fn empty_text_is_safe_without_a_runtime() {
        let backend = LocalModelBackend::new(LocalModelBackendConfig::default());
        for text in ["", " \n\t "] {
            let assessment = backend.analyze(&ctx(text)).await.unwrap();
            assert!(!assessment.has_risk);
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
            assert_eq!(assessment.risk_type, None);
            assert_eq!(assessment.details, None);
            assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
        }
    }

    #[tokio::test]
    async fn non_empty_text_reports_the_missing_runtime() {
        let backend = LocalModelBackend::new(LocalModelBackendConfig {
            model_path: "/models/bert".to_owned(),
            ..LocalModelBackendConfig::default()
        })
        .with_parser(Arc::new(BertBinaryParser::new()));
        let error = backend.analyze(&ctx("some text")).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("inference runtime"), "{error}");
    }

    #[tokio::test]
    async fn name_and_debug_are_stable() {
        let backend = LocalModelBackend::new(LocalModelBackendConfig {
            model_path: "/models/bert".to_owned(),
            device: "cpu".to_owned(),
            risk_type: "model_detection".to_owned(),
        });
        assert_eq!(backend.name(), "local_model");

        let debug = format!("{backend:?}");
        assert!(debug.contains("/models/bert"), "{debug}");
        assert!(debug.contains("has_parser: false"), "{debug}");
        assert!(debug.contains("cpu"), "{debug}");

        let with_parser = backend.with_parser(Arc::new(BertBinaryParser::new()));
        assert!(format!("{with_parser:?}").contains("has_parser: true"));
    }
}
