//! `BERT` binary-classification output parser.
//!
//! Ports `BertBinaryParser` from
//! `openjiuwen/core/security/guardrail/context.py`. The model has two classes:
//! class `0` is safe and the configurable attack class (default `1`) is an
//! attack. The predicted class is the primary signal and the confidence only
//! escalates it when the model is sure, which keeps false positives down.

use serde_json::{json, Value};

use super::{ModelOutputParser, Thresholds};
use crate::{RiskAssessment, RiskLevel};

/// Parser for a two-class `BERT` attack classifier.
///
/// The default configuration reproduces the Python constructor
/// (`risk_type = "attack_detected"`, `attack_class_id = 1`, default
/// [`Thresholds`]).
#[derive(Debug, Clone)]
pub struct BertBinaryParser {
    /// Risk name reported when the attack class is predicted
    /// (`"attack_detected"` by default).
    pub risk_type: String,
    /// Confidence bands that map an attack prediction to a risk level.
    pub confidence_thresholds: Thresholds,
    /// Class id that denotes an attack; class `0` is always safe.
    pub attack_class_id: i64,
}

impl Default for BertBinaryParser {
    /// The Python default configuration.
    fn default() -> Self {
        Self::new()
    }
}

impl BertBinaryParser {
    /// A parser with the Python defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            risk_type: "attack_detected".to_owned(),
            confidence_thresholds: Thresholds::default(),
            attack_class_id: 1,
        }
    }

    /// Uses `risk_type` instead of the default `"attack_detected"`.
    #[must_use]
    pub fn with_risk_type(mut self, risk_type: impl Into<String>) -> Self {
        self.risk_type = risk_type.into();
        self
    }

    /// Replaces the confidence bands.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.confidence_thresholds = thresholds;
        self
    }

    /// Uses `attack_class_id` as the attack class instead of `1`.
    ///
    /// Setting it to `0` flips the meaning of the positive class.
    #[must_use]
    pub fn with_attack_class_id(mut self, attack_class_id: i64) -> Self {
        self.attack_class_id = attack_class_id;
        self
    }

    /// Maps a predicted class and confidence to a risk level.
    fn determine_risk_level(&self, predicted_class: i64, confidence: f64) -> RiskLevel {
        if predicted_class != self.attack_class_id {
            return RiskLevel::Safe;
        }
        let confidence = if confidence.is_finite() {
            confidence
        } else {
            0.0
        };
        let thresholds = self.confidence_thresholds;
        if confidence < thresholds.low {
            RiskLevel::Safe
        } else if confidence < thresholds.medium {
            RiskLevel::Low
        } else if confidence < thresholds.high {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        }
    }

    /// Builds the assessment Python's `BertBinaryParser.parse` returns.
    fn assess(&self, predicted_class: i64, confidence: f64) -> RiskAssessment {
        let risk_level = self.determine_risk_level(predicted_class, confidence);
        let has_risk = risk_level != RiskLevel::Safe;
        let attack_confidence = if predicted_class == self.attack_class_id {
            confidence
        } else {
            1.0 - confidence
        };
        RiskAssessment {
            has_risk,
            risk_level,
            risk_type: has_risk.then(|| self.risk_type.clone()),
            confidence,
            details: Some(json!({
                "predicted_class": predicted_class,
                "attack_confidence": attack_confidence,
            })),
        }
    }
}

impl ModelOutputParser for BertBinaryParser {
    // The trait fixes the return type at `&str`; the name is a literal.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "bert_binary"
    }

    fn parse(&self, output: &Value) -> RiskAssessment {
        let (predicted_class, confidence) = extract_prediction(output, self.attack_class_id);
        self.assess(predicted_class, confidence)
    }
}

/// Extracts `(predicted_class, confidence)` from the first matching shape.
///
/// The order mirrors the Python `if` chain exactly, so a payload that carries
/// several keys is interpreted the same way.
fn extract_prediction(output: &Value, attack_class_id: i64) -> (i64, f64) {
    if let Value::Object(map) = output {
        if let Some(value) = map.get("predicted_class") {
            return (integer_of(value), float_or_zero(map.get("confidence")));
        }
        if let Some(value) = map.get("label") {
            let confidence = float_or_zero(map.get("score").or_else(|| map.get("confidence")));
            return (integer_of(value), confidence);
        }
        if let Some(Value::Array(probabilities)) = map.get("probabilities") {
            if probabilities.len() >= 2 {
                let values: Vec<f64> = probabilities.iter().map(float_of).collect();
                return argmax(&values);
            }
        }
        if let Some(Value::Array(logits)) = map.get("logits") {
            if logits.len() >= 2 {
                let values: Vec<f64> = logits.iter().map(float_of).collect();
                return argmax(&softmax(&values));
            }
        }
    }

    if let Value::Array(values) = output {
        if values.len() >= 2 {
            let numbers: Vec<f64> = values.iter().map(float_of).collect();
            return argmax(&numbers);
        }
    }

    if let Value::Number(number) = output {
        let confidence = number
            .as_f64()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        let predicted_class = if confidence > 0.5 { attack_class_id } else { 0 };
        return (predicted_class, confidence);
    }

    (0, 0.0)
}

/// The Python rule: class `1` when the second score beats the first, and the
/// maximum score as the confidence.
fn argmax(values: &[f64]) -> (i64, f64) {
    let first = values.first().copied().unwrap_or(0.0);
    let second = values.get(1).copied().unwrap_or(0.0);
    let predicted_class = i64::from(second > first);
    let confidence = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let confidence = if confidence.is_finite() {
        confidence
    } else {
        0.0
    };
    (predicted_class, confidence)
}

/// Numerically stable softmax over every logit.
///
/// Subtracting the maximum before `exp` keeps large logits from overflowing.
/// A degenerate sum (non-positive or non-finite) falls back to `[0.5, 0.5]`,
/// matching Python's guard.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exponentials.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        exponentials.into_iter().map(|value| value / sum).collect()
    } else {
        vec![0.5, 0.5]
    }
}

/// Reads a JSON number, or a numeric string, truncating toward zero.
///
/// Missing, non-numeric, and non-finite values degrade to `0`.
#[allow(clippy::cast_possible_truncation)]
fn integer_of(value: &Value) -> i64 {
    if let Some(integer) = value.as_i64() {
        return integer;
    }
    let parsed = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    };
    parsed
        .filter(|number| number.is_finite())
        .map_or(0, |number| number.trunc() as i64)
}

/// Reads a JSON number, or a numeric string, as a finite `f64`.
///
/// Missing, non-numeric, and non-finite values degrade to `0.0`.
fn float_of(value: &Value) -> f64 {
    let parsed = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    };
    parsed.filter(|number| number.is_finite()).unwrap_or(0.0)
}

/// [`float_of`] for an optional value, defaulting to `0.0`.
fn float_or_zero(value: Option<&Value>) -> f64 {
    value.map_or(0.0, float_of)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[allow(clippy::needless_pass_by_value)]
    fn parse(parser: &BertBinaryParser, output: Value) -> RiskAssessment {
        parser.parse(&output)
    }

    fn attack_confidence(assessment: &RiskAssessment) -> f64 {
        assessment.details.as_ref().unwrap()["attack_confidence"]
            .as_f64()
            .unwrap()
    }

    fn predicted_class(assessment: &RiskAssessment) -> i64 {
        assessment.details.as_ref().unwrap()["predicted_class"]
            .as_i64()
            .unwrap()
    }

    #[test]
    fn predicted_class_and_confidence_bands() {
        let parser = BertBinaryParser::new();

        let safe = parse(&parser, json!({"predicted_class": 1, "confidence": 0.6}));
        assert!(!safe.has_risk);
        assert_eq!(safe.risk_level, RiskLevel::Safe);
        assert_eq!(safe.risk_type, None);

        let low = parse(&parser, json!({"predicted_class": 1, "confidence": 0.75}));
        assert!(low.has_risk);
        assert_eq!(low.risk_level, RiskLevel::Low);
        assert_eq!(low.risk_type.as_deref(), Some("attack_detected"));

        let medium = parse(&parser, json!({"predicted_class": 1, "confidence": 0.9}));
        assert!(medium.has_risk);
        assert_eq!(medium.risk_level, RiskLevel::Medium);

        let high = parse(&parser, json!({"predicted_class": 1, "confidence": 0.97}));
        assert!(high.has_risk);
        assert_eq!(high.risk_level, RiskLevel::High);
        assert!((high.confidence - 0.97).abs() < f64::EPSILON);
    }

    #[test]
    fn predicted_class_safe_is_always_safe() {
        let parser = BertBinaryParser::new();
        let assessment = parse(&parser, json!({"predicted_class": 0, "confidence": 0.99}));
        assert!(!assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert_eq!(assessment.risk_type, None);
        assert_eq!(predicted_class(&assessment), 0);
        assert!((attack_confidence(&assessment) - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn predicted_class_boundaries_are_inclusive_low() {
        let parser = BertBinaryParser::new();
        for (confidence, expected) in [
            (0.69, RiskLevel::Safe),
            (0.7, RiskLevel::Low),
            (0.84, RiskLevel::Low),
            (0.85, RiskLevel::Medium),
            (0.94, RiskLevel::Medium),
            (0.95, RiskLevel::High),
        ] {
            let assessment = parse(
                &parser,
                json!({"predicted_class": 1, "confidence": confidence}),
            );
            assert_eq!(assessment.risk_level, expected, "confidence {confidence}");
        }
    }

    #[test]
    fn custom_thresholds_reband() {
        let parser = BertBinaryParser::new().with_thresholds(Thresholds {
            low: 0.8,
            medium: 0.9,
            high: 0.98,
        });
        let assessment = parse(&parser, json!({"predicted_class": 1, "confidence": 0.85}));
        assert_eq!(assessment.risk_level, RiskLevel::Low);

        let below = parse(&parser, json!({"predicted_class": 1, "confidence": 0.75}));
        assert_eq!(below.risk_level, RiskLevel::Safe);
    }

    #[test]
    fn attack_class_id_zero_flips_the_positive_class() {
        // ports tests/unit_tests/core/security/guardrail/test_parsers_and_backends.py
        let parser = BertBinaryParser::new().with_attack_class_id(0);
        let assessment = parse(&parser, json!({"predicted_class": 0, "confidence": 0.95}));
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert!((attack_confidence(&assessment) - 0.95).abs() < f64::EPSILON);

        let flipped = parse(&parser, json!({"predicted_class": 1, "confidence": 0.99}));
        assert!(!flipped.has_risk);
        assert_eq!(flipped.risk_level, RiskLevel::Safe);
    }

    #[test]
    fn label_and_score_shape() {
        // ports ...::test_parse_with_label_attack
        let parser = BertBinaryParser::new();
        let assessment = parse(&parser, json!({"label": 1, "score": 0.97}));
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(predicted_class(&assessment), 1);

        let falls_back_to_confidence = parse(&parser, json!({"label": 1, "confidence": 0.9}));
        assert_eq!(falls_back_to_confidence.risk_level, RiskLevel::Medium);

        let missing_score = parse(&parser, json!({"label": 1}));
        assert_eq!(missing_score.risk_level, RiskLevel::Safe);
        assert!(missing_score.confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn probabilities_shape() {
        let parser = BertBinaryParser::new();

        let attack = parse(&parser, json!({"probabilities": [0.02, 0.98]}));
        assert!(attack.has_risk);
        assert_eq!(attack.risk_level, RiskLevel::High);
        assert_eq!(predicted_class(&attack), 1);
        assert!((attack.confidence - 0.98).abs() < f64::EPSILON);

        let safe = parse(&parser, json!({"probabilities": [0.9, 0.1]}));
        assert!(!safe.has_risk);
        assert_eq!(safe.risk_level, RiskLevel::Safe);
        assert_eq!(predicted_class(&safe), 0);
        assert!((attack_confidence(&safe) - 0.1).abs() < f64::EPSILON);

        let too_short = parse(&parser, json!({"probabilities": [0.9]}));
        assert!(!too_short.has_risk);
        assert!(too_short.confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn logits_shape_uses_softmax() {
        let parser = BertBinaryParser::new();

        let attack = parse(&parser, json!({"logits": [-2.0, 5.0]}));
        assert!(attack.has_risk);
        assert_eq!(attack.risk_level, RiskLevel::High);
        assert_eq!(predicted_class(&attack), 1);

        let safe = parse(&parser, json!({"logits": [2.0, 0.5]}));
        assert!(!safe.has_risk);
        assert_eq!(safe.risk_level, RiskLevel::Safe);
        assert_eq!(predicted_class(&safe), 0);

        // Equal logits give a `[0.5, 0.5]` distribution: no attack, 0.5
        // confidence.
        let tied = parse(&parser, json!({"logits": [1.0, 1.0]}));
        assert!(!tied.has_risk);
        assert!((tied.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let probabilities = softmax(&[1000.0, 1001.0]);
        assert!((probabilities[0] + probabilities[1] - 1.0).abs() < 1e-9);
        assert!(probabilities[1] > probabilities[0]);

        // Degenerate input falls back to a uniform distribution.
        assert_eq!(softmax(&[]), vec![0.5, 0.5]);
    }

    #[test]
    fn top_level_array_shape() {
        // ports ...::test_parse_with_list_attack
        let parser = BertBinaryParser::new();

        let attack = parse(&parser, json!([0.2, 0.8]));
        assert!(attack.has_risk);
        assert_eq!(attack.risk_level, RiskLevel::Low);
        assert!((attack.confidence - 0.8).abs() < f64::EPSILON);

        let safe = parse(&parser, json!([0.8, 0.2]));
        assert!(!safe.has_risk);
        assert_eq!(safe.risk_level, RiskLevel::Safe);

        let short = parse(&parser, json!([0.9]));
        assert!(!short.has_risk);
        assert!(short.confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn bare_number_shape_uses_strict_greater_than() {
        let parser = BertBinaryParser::new();

        let at_half = parse(&parser, json!(0.5));
        assert!(!at_half.has_risk);
        assert_eq!(predicted_class(&at_half), 0);
        assert!((at_half.confidence - 0.5).abs() < f64::EPSILON);

        let above_half = parse(&parser, json!(0.6));
        assert_eq!(
            predicted_class(&above_half),
            1,
            "0.6 selects the attack class"
        );
        assert!(
            !above_half.has_risk,
            "0.6 is still below the low confidence band"
        );
        assert_eq!(above_half.risk_level, RiskLevel::Safe);

        let confident = parse(&parser, json!(0.97));
        assert!(confident.has_risk);
        assert_eq!(confident.risk_level, RiskLevel::High);
    }

    #[test]
    fn unrecognized_inputs_are_safe() {
        let parser = BertBinaryParser::new();
        for output in [
            json!({}),
            Value::Null,
            json!("definitely an attack"),
            json!(true),
            json!([0.5]),
        ] {
            let assessment = parse(&parser, output.clone());
            assert!(!assessment.has_risk, "{output}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
            assert!(assessment.confidence.abs() < f64::EPSILON, "{output}");
            assert_eq!(assessment.risk_type, None);
        }
    }

    #[test]
    fn numeric_strings_are_accepted() {
        let parser = BertBinaryParser::new();
        let assessment = parse(
            &parser,
            json!({"predicted_class": "1", "confidence": "0.97"}),
        );
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(predicted_class(&assessment), 1);
    }

    #[test]
    fn attack_confidence_is_inverted_for_the_safe_class() {
        let parser = BertBinaryParser::new();
        let safe = parse(&parser, json!({"predicted_class": 0, "confidence": 0.9}));
        assert!((attack_confidence(&safe) - 0.1).abs() < 1e-9);

        let attack = parse(&parser, json!({"predicted_class": 1, "confidence": 0.9}));
        assert!((attack_confidence(&attack) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn custom_risk_type_is_reported_only_when_risky() {
        let parser = BertBinaryParser::new().with_risk_type("prompt_injection");
        let risky = parse(&parser, json!({"predicted_class": 1, "confidence": 0.95}));
        assert_eq!(risky.risk_type.as_deref(), Some("prompt_injection"));

        let safe = parse(&parser, json!({"predicted_class": 0, "confidence": 0.95}));
        assert_eq!(safe.risk_type, None);
    }

    #[test]
    fn constructor_defaults_match_python() {
        let parser = BertBinaryParser::default();
        assert_eq!(parser.risk_type, "attack_detected");
        assert_eq!(parser.attack_class_id, 1);
        assert_eq!(parser.confidence_thresholds, Thresholds::default());
    }
}
