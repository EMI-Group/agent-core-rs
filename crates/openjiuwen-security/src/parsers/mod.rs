//! Model-output parsers.
//!
//! Turn the raw output of a moderation or jailbreak model into a
//! [`RiskAssessment`]. Ports `ModelOutputParser`, `BertBinaryParser`, and
//! `QwenGuardParser` from `openjiuwen/core/security/guardrail/context.py`
//! (v0.1.20) with the same accepted dialects and thresholds, because
//! deployments point these parsers at fixed model endpoints.
//!
//! Parsing is pure and synchronous: it performs no I/O and never panics.
//! [`ModelOutputParser::parse`] is *fallible*: an input that matches none of
//! the documented shapes (an empty object, an unrelated key set, `null`, a
//! boolean, a bare number or array, free text with no verdict marker, ...)
//! returns [`openjiuwen_common::error::Error::validation`] naming the parser and its expected dialects,
//! so a backend can never grade an unrecognizable reply as `Safe`.
//!
//! A *recognized* shape whose level value is unknown is a different case: the
//! value maps onto the parser's configured default level
//! ([`QwenGuardParser::with_default_risk_level`], `RiskLevel::High` in a
//! fail-safe deployment), and no error is returned.

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::RiskAssessment;

pub mod bert;
pub mod qwen;

pub use bert::BertBinaryParser;
pub use qwen::QwenGuardParser;

/// Converts raw model output into a [`RiskAssessment`].
///
/// Implementations are `Send + Sync` so a backend can share one parser across
/// async tasks. `parse` is fallible: input outside the accepted dialects is an
/// [`openjiuwen_common::error::Error::validation`], never a silent "safe" grade and never a panic.
pub trait ModelOutputParser: Send + Sync {
    /// Stable parser name (`"bert_binary"`, `"qwen_guard"`), used in logs and
    /// backend descriptions.
    fn name(&self) -> &str;

    /// Interprets `output` as this model's response and grades it.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] naming this parser and the dialects it accepts
    /// when `output` matches none of them. The message never echoes the
    /// payload.
    ///
    /// # Examples
    ///
    /// ```
    /// use openjiuwen_security::{BertBinaryParser, ModelOutputParser};
    /// use serde_json::json;
    ///
    /// let parser = BertBinaryParser::new();
    /// let assessment = parser
    ///     .parse(&json!({"predicted_class": 1, "confidence": 0.97}))
    ///     .unwrap();
    /// assert!(assessment.has_risk);
    /// assert_eq!(assessment.risk_level, openjiuwen_security::RiskLevel::High);
    /// ```
    fn parse(&self, output: &Value) -> Result<RiskAssessment>;
}

/// Confidence cut-offs for the `BERT` binary classifier.
///
/// The parser only escalates a predicted attack when the model is confident
/// enough, which keeps borderline predictions from producing false positives.
/// A confidence below [`Thresholds::low`] is treated as safe, then the value
/// falls into `low`/`medium`/`high` bands.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    /// Lowest confidence that reports [`crate::RiskLevel::Low`] (Python: `0.7`).
    pub low: f64,
    /// Lowest confidence that reports [`crate::RiskLevel::Medium`] (Python: `0.85`).
    pub medium: f64,
    /// Lowest confidence that reports [`crate::RiskLevel::High`] (Python: `0.95`).
    pub high: f64,
}

impl Default for Thresholds {
    /// The Python `BertBinaryParser.DEFAULT_THRESHOLDS` values (`0.7`, `0.85`,
    /// `0.95`).
    fn default() -> Self {
        Self {
            low: 0.7,
            medium: 0.85,
            high: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn default_thresholds_match_python() {
        let thresholds = Thresholds::default();
        assert!((thresholds.low - 0.7).abs() < f64::EPSILON);
        assert!((thresholds.medium - 0.85).abs() < f64::EPSILON);
        assert!((thresholds.high - 0.95).abs() < f64::EPSILON);
        assert_eq!(Thresholds::default(), thresholds);
    }

    #[test]
    fn thresholds_serde_defaults_and_round_trips() {
        let partial: Thresholds = serde_json::from_value(json!({"low": 0.5})).unwrap();
        assert!((partial.low - 0.5).abs() < f64::EPSILON);
        assert_eq!(partial.medium, Thresholds::default().medium);
        assert_eq!(partial.high, Thresholds::default().high);

        let empty: Thresholds = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, Thresholds::default());

        let value = serde_json::to_value(Thresholds::default()).unwrap();
        assert_eq!(value, json!({"low": 0.7, "medium": 0.85, "high": 0.95}));
        let back: Thresholds = serde_json::from_value(value).unwrap();
        assert_eq!(back, Thresholds::default());
    }

    #[test]
    fn parser_names_are_stable() {
        assert_eq!(BertBinaryParser::new().name(), "bert_binary");
        assert_eq!(QwenGuardParser::new().name(), "qwen_guard");
    }
}
