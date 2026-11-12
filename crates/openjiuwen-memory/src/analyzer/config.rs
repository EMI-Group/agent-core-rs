//! [`AnalyzerConfig`]: the knobs of the memory analyzer, extractor, and classifier.
//!
//! The config is deliberately its own type rather than a section of
//! [`crate::config::MemoryEngineConfig`]: the analyzer is an optional seam layered on
//! top of the engine's [`ExtractionConfig`], and callers may tune it per call site
//! (for example a harness that wants longer inputs or more facts) without touching the
//! engine contract. [`AnalyzerConfig::from_engine`] derives the defaults a
//! `MemoryEngineConfig` implies, so the two never drift.
use crate::config::{ExtractionConfig, MemoryEngineConfig};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Importance of a fact introduced by an explicit directive such as `remember ...`.
pub const DIRECTIVE_IMPORTANCE: f32 = 0.9;
/// Importance of a user-identity fact (`my name is ...`, `I work at ...`).
pub const IDENTITY_IMPORTANCE: f32 = 0.85;
/// Importance of a stated preference (`I prefer ...`, `I hate ...`).
pub const PREFERENCE_IMPORTANCE: f32 = 0.75;
/// Importance of a project/technology fact (`the service uses ...`).
pub const PROJECT_IMPORTANCE: f32 = 0.6;
/// Importance of a generic declarative statement with no stronger signal.
pub const STATEMENT_IMPORTANCE: f32 = 0.4;

/// Which [`crate::analyzer::FactExtractor`] the analyzer builds by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractorKind {
    /// Deterministic, model-free rule extraction (the default).
    #[default]
    Rule,
    /// Model-backed extraction with a deterministic fallback.
    Llm,
}

/// Which [`crate::analyzer::ConflictClassifier`] the analyzer builds by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierKind {
    /// Deterministic, model-free conflict classification (the default).
    #[default]
    Rule,
    /// Model-backed conflict classification with a deterministic fallback.
    Llm,
}

/// Configuration of the memory analyzer, extractor, and conflict classifier.
///
/// Every field has a documented default, JSON names are `snake_case`, and unknown
/// fields are ignored so configuration can be forward-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AnalyzerConfig {
    /// Whether extraction runs at all (`true`); `false` yields an empty result.
    pub enabled: bool,
    /// Extractor kind the analyzer builds when a model is supplied (`rule`).
    pub extractor: ExtractorKind,
    /// Classifier kind the analyzer builds when a model is supplied (`rule`).
    pub classifier: ClassifierKind,
    /// Maximum number of facts one extraction returns (`8`).
    pub max_facts: usize,
    /// Turns shorter than this many characters are skipped (`8`); `0` disables the
    /// minimum.
    pub min_message_len: usize,
    /// Maximum characters of conversation text fed to an extractor (`8192`).
    pub max_input_chars: usize,
    /// Maximum history turns rendered alongside the current turns (`2`).
    pub max_history_turns: usize,
    /// Retries after the first model call (`3`).
    pub model_retries: u32,
    /// Importance assigned to a fact that does not carry one (`0.5`).
    pub default_importance: f32,
    /// Facts below this importance are dropped (`0.0`).
    pub min_importance: f32,
    /// Maximum characters of a single fact's content (`512`).
    pub max_content_chars: usize,
    /// Maximum existing records sent to a conflict classifier (`8`).
    pub max_candidates: usize,
    /// Importance of an explicit directive fact (`0.9`, [`DIRECTIVE_IMPORTANCE`]).
    pub directive_importance: f32,
    /// Importance of an identity fact (`0.85`, [`IDENTITY_IMPORTANCE`]).
    pub identity_importance: f32,
    /// Importance of a preference fact (`0.75`, [`PREFERENCE_IMPORTANCE`]).
    pub preference_importance: f32,
    /// Importance of a project/technology fact (`0.6`, [`PROJECT_IMPORTANCE`]).
    pub project_importance: f32,
    /// Importance of a generic statement fact (`0.4`, [`STATEMENT_IMPORTANCE`]).
    pub statement_importance: f32,
}

impl Default for AnalyzerConfig {
    /// Returns the documented defaults.
    fn default() -> Self {
        Self {
            enabled: true,
            extractor: ExtractorKind::Rule,
            classifier: ClassifierKind::Rule,
            max_facts: 8,
            min_message_len: 8,
            max_input_chars: 8192,
            max_history_turns: 2,
            model_retries: 3,
            default_importance: 0.5,
            min_importance: 0.0,
            max_content_chars: 512,
            max_candidates: 8,
            directive_importance: DIRECTIVE_IMPORTANCE,
            identity_importance: IDENTITY_IMPORTANCE,
            preference_importance: PREFERENCE_IMPORTANCE,
            project_importance: PROJECT_IMPORTANCE,
            statement_importance: STATEMENT_IMPORTANCE,
        }
    }
}

impl AnalyzerConfig {
    /// Derives the analyzer settings an [`ExtractionConfig`] and an input limit imply.
    ///
    /// The shared knobs (`enabled`, `max_facts`, `model_retries`, `min_message_len`,
    /// history depth, input length) come from `extraction`/`input_msg_max_len`; the
    /// analyzer-only knobs keep their defaults.
    #[must_use]
    pub fn from_extraction(extraction: &ExtractionConfig, input_msg_max_len: usize) -> Self {
        Self {
            enabled: extraction.enabled,
            max_facts: extraction.max_facts,
            min_message_len: extraction.min_message_len,
            max_input_chars: input_msg_max_len,
            max_history_turns: extraction.use_history_messages,
            model_retries: extraction.model_retries,
            ..Self::default()
        }
    }

    /// Derives the analyzer settings from a whole [`MemoryEngineConfig`].
    ///
    /// Equivalent to [`AnalyzerConfig::from_extraction`] on the engine's extraction
    /// section plus the engine's `input_msg_max_len` and `importance_default`.
    #[must_use]
    pub fn from_engine(config: &MemoryEngineConfig) -> Self {
        Self {
            default_importance: config.importance_default,
            ..Self::from_extraction(&config.extraction, config.input_msg_max_len)
        }
    }

    /// Validates the limits and importance tiers.
    ///
    /// Zero `max_facts`/`max_input_chars`/`max_content_chars`/`max_candidates` and
    /// non-finite or out-of-range importances are `Error::Validation` naming the field
    /// and the offending value. `min_message_len == 0` is legal and disables the
    /// minimum-length filter.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("max_facts", self.max_facts),
            ("max_input_chars", self.max_input_chars),
            ("max_content_chars", self.max_content_chars),
            ("max_candidates", self.max_candidates),
        ] {
            if value == 0 {
                return Err(Error::validation(format!(
                    "analyzer `{field}` must be greater than zero (got {value})"
                )));
            }
        }
        for (field, value) in [
            ("default_importance", self.default_importance),
            ("min_importance", self.min_importance),
            ("directive_importance", self.directive_importance),
            ("identity_importance", self.identity_importance),
            ("preference_importance", self.preference_importance),
            ("project_importance", self.project_importance),
            ("statement_importance", self.statement_importance),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::validation(format!(
                    "analyzer `{field}` must be within 0.0..=1.0 (got {value})"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(config: &AnalyzerConfig) -> String {
        match config.validate() {
            Err(Error::Validation(message)) => message,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_match_the_specification() {
        let config = AnalyzerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.extractor, ExtractorKind::Rule);
        assert_eq!(config.classifier, ClassifierKind::Rule);
        assert_eq!(config.max_facts, 8);
        assert_eq!(config.min_message_len, 8);
        assert_eq!(config.max_input_chars, 8192);
        assert_eq!(config.max_history_turns, 2);
        assert_eq!(config.model_retries, 3);
        assert!((config.default_importance - 0.5).abs() < f32::EPSILON);
        assert!(config.min_importance.abs() < f32::EPSILON);
        assert_eq!(config.max_content_chars, 512);
        assert_eq!(config.max_candidates, 8);
        assert!((config.directive_importance - DIRECTIVE_IMPORTANCE).abs() < f32::EPSILON);
        assert!((config.identity_importance - IDENTITY_IMPORTANCE).abs() < f32::EPSILON);
        assert!((config.preference_importance - PREFERENCE_IMPORTANCE).abs() < f32::EPSILON);
        assert!((config.project_importance - PROJECT_IMPORTANCE).abs() < f32::EPSILON);
        assert!((config.statement_importance - STATEMENT_IMPORTANCE).abs() < f32::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn from_extraction_copies_the_shared_knobs() {
        let extraction = ExtractionConfig {
            enabled: false,
            max_facts: 3,
            model_retries: 1,
            use_history_messages: 4,
            min_message_len: 2,
        };
        let config = AnalyzerConfig::from_extraction(&extraction, 256);
        assert!(!config.enabled);
        assert_eq!(config.max_facts, 3);
        assert_eq!(config.model_retries, 1);
        assert_eq!(config.max_history_turns, 4);
        assert_eq!(config.min_message_len, 2);
        assert_eq!(config.max_input_chars, 256);
        assert_eq!(config.max_content_chars, 512);
    }

    #[test]
    fn from_engine_copies_the_input_limit_and_importance() {
        let engine = MemoryEngineConfig {
            input_msg_max_len: 1024,
            importance_default: 0.7,
            ..Default::default()
        };
        let config = AnalyzerConfig::from_engine(&engine);
        assert_eq!(config.max_input_chars, 1024);
        assert!((config.default_importance - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.max_facts, engine.extraction.max_facts);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_zero_limits() {
        for (field, config) in [
            (
                "max_facts",
                AnalyzerConfig {
                    max_facts: 0,
                    ..Default::default()
                },
            ),
            (
                "max_input_chars",
                AnalyzerConfig {
                    max_input_chars: 0,
                    ..Default::default()
                },
            ),
            (
                "max_content_chars",
                AnalyzerConfig {
                    max_content_chars: 0,
                    ..Default::default()
                },
            ),
            (
                "max_candidates",
                AnalyzerConfig {
                    max_candidates: 0,
                    ..Default::default()
                },
            ),
        ] {
            let text = message(&config);
            assert!(text.contains(field), "{text} must name {field}");
            assert!(text.contains('0'), "{text} must name the value");
        }
    }

    #[test]
    fn rejects_out_of_range_and_non_finite_importance() {
        for (field, config) in [
            (
                "default_importance",
                AnalyzerConfig {
                    default_importance: 1.5,
                    ..Default::default()
                },
            ),
            (
                "min_importance",
                AnalyzerConfig {
                    min_importance: -0.1,
                    ..Default::default()
                },
            ),
            (
                "statement_importance",
                AnalyzerConfig {
                    statement_importance: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "directive_importance",
                AnalyzerConfig {
                    directive_importance: f32::INFINITY,
                    ..Default::default()
                },
            ),
        ] {
            assert!(message(&config).contains(field), "{field} must be rejected");
        }
    }

    #[test]
    fn zero_min_message_len_is_legal() {
        let config = AnalyzerConfig {
            min_message_len: 0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = AnalyzerConfig {
            enabled: false,
            extractor: ExtractorKind::Llm,
            classifier: ClassifierKind::Llm,
            max_facts: 2,
            min_message_len: 1,
            max_input_chars: 64,
            max_history_turns: 0,
            model_retries: 0,
            default_importance: 0.4,
            min_importance: 0.2,
            max_content_chars: 32,
            max_candidates: 3,
            directive_importance: 0.95,
            identity_importance: 0.8,
            preference_importance: 0.7,
            project_importance: 0.5,
            statement_importance: 0.3,
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "enabled",
            "extractor",
            "classifier",
            "max_facts",
            "min_message_len",
            "max_input_chars",
            "max_history_turns",
            "model_retries",
            "default_importance",
            "min_importance",
            "max_content_chars",
            "max_candidates",
            "directive_importance",
            "identity_importance",
            "preference_importance",
            "project_importance",
            "statement_importance",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["extractor"], serde_json::json!("llm"));
        assert_eq!(json["classifier"], serde_json::json!("llm"));
        let restored: AnalyzerConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn unknown_fields_and_kind_strings_are_tolerated() {
        let config: AnalyzerConfig = serde_json::from_value(serde_json::json!({
            "max_facts": 4,
            "extractor": "rule",
            "classifier": "llm",
            "not_a_knob": [1, 2],
        }))
        .unwrap();
        assert_eq!(config.max_facts, 4);
        assert_eq!(config.classifier, ClassifierKind::Llm);
        assert!(config.enabled);
    }
}
