//! Result types shared by every guardrail.
//!
//! Ports `openjiuwen/core/security/guardrail/models.py`: a guardrail produces a
//! [`GuardrailResult`], while a backend produces a [`RiskAssessment`] that is
//! converted into a result.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::RiskLevel;

/// The verdict of one guardrail for one [`crate::GuardrailContext`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct GuardrailResult {
    /// Whether the inspected content passed the guardrail.
    pub is_safe: bool,
    /// Severity of the detected risk; [`RiskLevel::Safe`] when none.
    pub risk_level: RiskLevel,
    /// Machine-readable risk name (`"prompt_injection"`, `"pii"`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_type: Option<String>,
    /// Non-sensitive evidence: pattern names, rule names, counts, scores.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Sanitized content a guardrail proposes to the caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_data: Option<Value>,
}

impl Default for GuardrailResult {
    /// A safe pass without details.
    fn default() -> Self {
        Self::pass()
    }
}

impl GuardrailResult {
    /// A safe result without details.
    #[must_use]
    pub fn pass() -> Self {
        Self {
            is_safe: true,
            risk_level: RiskLevel::Safe,
            risk_type: None,
            details: None,
            modified_data: None,
        }
    }

    /// A safe result carrying evidence, such as the patterns that were checked.
    #[must_use]
    pub fn pass_with(details: Value) -> Self {
        Self {
            details: Some(details),
            ..Self::pass()
        }
    }

    /// An unsafe result at `risk_level`.
    #[must_use]
    pub fn block(
        risk_level: RiskLevel,
        risk_type: impl Into<String>,
        details: Option<Value>,
        modified_data: Option<Value>,
    ) -> Self {
        Self {
            is_safe: false,
            risk_level,
            risk_type: Some(risk_type.into()),
            details,
            modified_data,
        }
    }

    /// Whether the inspected content passed the guardrail.
    #[must_use]
    pub fn is_safe(&self) -> bool {
        self.is_safe
    }

    /// The risk name, or the literal `"unknown"` when it is missing or empty
    /// (Python parity).
    #[must_use]
    pub fn risk_type_or_unknown(&self) -> &str {
        match self.risk_type.as_deref() {
            Some(risk_type) if !risk_type.is_empty() => risk_type,
            _ => "unknown",
        }
    }

    /// The message describing the detection.
    ///
    /// A [`RiskLevel::Critical`] detection reproduces Python's `AbortError`
    /// reason, every lower level reproduces Python's `GuardrailError` message.
    #[must_use]
    pub fn message(&self) -> String {
        let risk_type = self.risk_type_or_unknown();
        if self.risk_level == RiskLevel::Critical {
            format!("Critical security risk detected: {risk_type}")
        } else {
            format!("Guardrail blocked: {risk_type} risk detected")
        }
    }
}

/// The verdict of a backend (moderation service or local model) for one scan.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct RiskAssessment {
    /// Whether the backend considers the content risky.
    pub has_risk: bool,
    /// Severity reported by the backend.
    pub risk_level: RiskLevel,
    /// Machine-readable risk name reported by the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_type: Option<String>,
    /// Backend confidence in `0.0..=1.0`.
    pub confidence: f64,
    /// Non-sensitive evidence reported by the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl RiskAssessment {
    /// Converts the assessment into a [`GuardrailResult`], dropping `confidence`.
    #[must_use]
    pub fn into_result(self) -> GuardrailResult {
        GuardrailResult {
            is_safe: !self.has_risk,
            risk_level: self.risk_level,
            risk_type: self.risk_type,
            details: self.details,
            modified_data: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn pass_is_safe_and_empty() {
        let result = GuardrailResult::pass();
        assert!(result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Safe);
        assert_eq!(result.risk_type, None);
        assert_eq!(result.details, None);
        assert_eq!(result.modified_data, None);
        assert_eq!(GuardrailResult::default(), result);
    }

    #[test]
    fn pass_with_keeps_details_only() {
        let result = GuardrailResult::pass_with(json!({"checked": 3}));
        assert!(result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Safe);
        assert_eq!(result.details, Some(json!({"checked": 3})));
        assert_eq!(result.risk_type, None);
    }

    #[test]
    fn block_marks_unsafe() {
        let result = GuardrailResult::block(
            RiskLevel::High,
            "prompt_injection",
            Some(json!({"matched_pattern": "ignore"})),
            None,
        );
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::High);
        assert_eq!(result.risk_type.as_deref(), Some("prompt_injection"));
        assert_eq!(result.details, Some(json!({"matched_pattern": "ignore"})));
        assert_eq!(result.modified_data, None);
    }

    #[test]
    fn message_matches_python_wording() {
        let critical = GuardrailResult::block(RiskLevel::Critical, "jailbreak", None, None);
        assert_eq!(
            critical.message(),
            "Critical security risk detected: jailbreak"
        );

        let high = GuardrailResult::block(RiskLevel::High, "pii", None, None);
        assert_eq!(high.message(), "Guardrail blocked: pii risk detected");

        let unknown = GuardrailResult::block(RiskLevel::High, "", None, None);
        assert_eq!(
            unknown.message(),
            "Guardrail blocked: unknown risk detected"
        );

        let missing = GuardrailResult {
            is_safe: false,
            risk_level: RiskLevel::Critical,
            ..GuardrailResult::pass()
        };
        assert_eq!(
            missing.message(),
            "Critical security risk detected: unknown"
        );
    }

    #[test]
    fn risk_assessment_defaults_are_safe() {
        let assessment = RiskAssessment::default();
        assert!(!assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert_eq!(assessment.risk_type, None);
        assert!((assessment.confidence - 0.0).abs() < f64::EPSILON);
        assert_eq!(assessment.details, None);
        assert!(assessment.clone().into_result().is_safe());
    }

    #[test]
    fn into_result_drops_confidence() {
        let assessment = RiskAssessment {
            has_risk: true,
            risk_level: RiskLevel::Medium,
            risk_type: Some("toxicity".to_owned()),
            confidence: 0.87,
            details: Some(json!({"score": 0.87})),
        };
        let result = assessment.into_result();
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Medium);
        assert_eq!(result.risk_type.as_deref(), Some("toxicity"));
        assert_eq!(result.details, Some(json!({"score": 0.87})));
        assert_eq!(result.modified_data, None);
    }

    #[test]
    fn json_field_names_and_optional_fields() {
        let safe = serde_json::to_value(GuardrailResult::pass()).unwrap();
        assert_eq!(
            safe,
            json!({"is_safe": true, "risk_level": "safe"}),
            "None fields must be omitted"
        );

        let blocked = serde_json::to_value(GuardrailResult::block(
            RiskLevel::Critical,
            "secret",
            Some(json!({"pattern": "aws_key"})),
            Some(json!({"sanitized_text": "<REDACTED>"})),
        ))
        .unwrap();
        assert_eq!(blocked["is_safe"], json!(false));
        assert_eq!(blocked["risk_level"], json!("critical"));
        assert_eq!(blocked["risk_type"], json!("secret"));
        assert_eq!(
            blocked["modified_data"]["sanitized_text"],
            json!("<REDACTED>")
        );

        let back: GuardrailResult = serde_json::from_value(safe).unwrap();
        assert_eq!(back, GuardrailResult::pass());
    }

    #[test]
    fn risk_assessment_serde_round_trip() {
        let assessment = RiskAssessment {
            has_risk: true,
            risk_level: RiskLevel::Low,
            risk_type: Some("spam".to_owned()),
            confidence: 0.5,
            details: None,
        };
        let value = serde_json::to_value(&assessment).unwrap();
        assert_eq!(value["confidence"], json!(0.5));
        assert_eq!(value.get("details"), None);
        let back: RiskAssessment = serde_json::from_value(value).unwrap();
        assert_eq!(back, assessment);

        let empty: RiskAssessment = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, RiskAssessment::default());
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let result: GuardrailResult =
            serde_json::from_value(json!({"is_safe": true, "future": 1})).unwrap();
        assert!(result.is_safe());
    }
}
