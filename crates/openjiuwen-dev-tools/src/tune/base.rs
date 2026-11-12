//! The tuning data models: [`Case`] and [`EvaluatedCase`].
//!
//! Ports `Case` and `EvaluatedCase` from the Python
//! `openjiuwen/dev_tools/tune/base.py`. Both types are plain data: the
//! constructors enforce the Python `pydantic` constraints (non-empty
//! `inputs`/`label`, a score inside `0.0..=1.0`) and report violations as
//! [`Error::validation`] instead of panicking, while deserialization stays
//! permissive so that malformed-but-parseable JSON still round-trips.
//!
//! The JSON field names are Python's: `inputs`, `label`, `tools`, `case_id` for
//! [`Case`] and `case`, `answer`, `score`, `reason` for [`EvaluatedCase`].

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::ToolInfo;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Generates the default [`Case::case_id`]: a fresh v4 UUID without hyphens.
///
/// Python defaults to `str(uuid.uuid4())`; the hyphen-free form keeps the
/// identifier convenient as a file name or map key without quoting, and stays
/// unique per deserialized case.
fn new_case_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// One tuning example: the model input plus the expected label.
///
/// Python: `openjiuwen.dev_tools.tune.base.Case`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct Case {
    /// The inputs handed to the agent; must be non-empty for a valid case.
    #[serde(default)]
    pub inputs: BTreeMap<String, Value>,
    /// The expected outputs used for scoring; must be non-empty for a valid case.
    #[serde(default)]
    pub label: BTreeMap<String, Value>,
    /// Optional tool declarations available to the agent while solving the case.
    #[serde(default)]
    pub tools: Option<Vec<ToolInfo>>,
    /// The case identifier; a fresh v4 UUID when absent from the JSON payload.
    #[serde(default = "new_case_id")]
    pub case_id: String,
}

impl Case {
    /// Creates a case from non-empty `inputs` and `label`.
    ///
    /// The `case_id` is generated. Empty maps are rejected with
    /// [`Error::validation`], matching Python's `Field(..., min_length=1)`.
    pub fn new(inputs: BTreeMap<String, Value>, label: BTreeMap<String, Value>) -> Result<Self> {
        let case = Self {
            inputs,
            label,
            tools: None,
            case_id: new_case_id(),
        };
        case.validate()?;
        Ok(case)
    }

    /// Attaches tool declarations, replacing any previously set list.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolInfo>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Replaces the generated identifier, useful for deterministic tests and
    /// for re-reading fixtures.
    #[must_use]
    pub fn with_case_id(mut self, id: impl Into<String>) -> Self {
        self.case_id = id.into();
        self
    }

    /// Checks the Python field constraints: non-empty `inputs`, non-empty
    /// `label`, and a non-blank `case_id`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the offending field.
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err(Error::validation("Case.inputs must not be empty"));
        }
        if self.label.is_empty() {
            return Err(Error::validation("Case.label must not be empty"));
        }
        if self.case_id.trim().is_empty() {
            return Err(Error::validation("Case.case_id must not be empty"));
        }
        Ok(())
    }
}

/// The outcome of running one [`Case`]: the answer, its score, and the reason.
///
/// Python: `openjiuwen.dev_tools.tune.base.EvaluatedCase`. The accessors mirror
/// Python's properties, which forward to the wrapped case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct EvaluatedCase {
    /// The case that was evaluated.
    pub case: Case,
    /// The agent answer; empty when nothing was produced.
    #[serde(default)]
    pub answer: BTreeMap<String, Value>,
    /// The score in `0.0..=1.0`; deserialization does not enforce the bounds.
    #[serde(default)]
    pub score: f64,
    /// The human-readable justification of `score`.
    #[serde(default)]
    pub reason: String,
}

impl EvaluatedCase {
    /// Creates an evaluated case, validating the score bounds.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `score` is not finite or outside `0.0..=1.0`.
    pub fn new(
        case: Case,
        answer: BTreeMap<String, Value>,
        score: f64,
        reason: impl Into<String>,
    ) -> Result<Self> {
        let evaluated = Self {
            case,
            answer,
            score,
            reason: reason.into(),
        };
        evaluated.validate()?;
        Ok(evaluated)
    }

    /// Checks that the score is finite and inside `0.0..=1.0`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] describing the rejected score.
    pub fn validate(&self) -> Result<()> {
        if !self.score.is_finite() || !(0.0..=1.0).contains(&self.score) {
            return Err(Error::validation(format!(
                "EvaluatedCase.score must be finite and between 0.0 and 1.0, got {}",
                self.score
            )));
        }
        Ok(())
    }

    /// The inputs of the wrapped case.
    #[must_use]
    pub fn inputs(&self) -> &BTreeMap<String, Value> {
        &self.case.inputs
    }

    /// The expected label of the wrapped case.
    #[must_use]
    pub fn label(&self) -> &BTreeMap<String, Value> {
        &self.case.label
    }

    /// The tool declarations of the wrapped case, when any were set.
    #[must_use]
    pub fn tools(&self) -> Option<&[ToolInfo]> {
        self.case.tools.as_deref()
    }

    /// The identifier of the wrapped case.
    #[must_use]
    pub fn case_id(&self) -> &str {
        &self.case.case_id
    }
}

impl From<Case> for EvaluatedCase {
    /// Wraps an unevaluated case with a zero score and no answer or reason.
    fn from(case: Case) -> Self {
        Self {
            case,
            answer: BTreeMap::new(),
            score: 0.0,
            reason: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-entry map used for the `inputs`/`label` fixtures.
    fn map(key: &str, value: &str) -> BTreeMap<String, Value> {
        let mut map = BTreeMap::new();
        map.insert(key.to_owned(), Value::String(value.to_owned()));
        map
    }

    /// A valid case with a fixed identifier.
    fn sample_case(id: &str) -> Case {
        Case::new(map("question", "hi"), map("answer", "hello"))
            .unwrap()
            .with_case_id(id)
    }

    fn tool() -> ToolInfo {
        ToolInfo::new("lookup", "look something up", Value::Null)
    }

    #[test]
    fn new_generates_an_identifier_and_keeps_the_maps() {
        let case = Case::new(map("question", "hi"), map("answer", "hello")).unwrap();
        assert_eq!(case.inputs, map("question", "hi"));
        assert_eq!(case.label, map("answer", "hello"));
        assert!(case.tools.is_none());
        assert_ne!(case.case_id, "");
        assert!(!case.case_id.contains('-'), "id is hyphen-free");
        assert_eq!(case.case_id.len(), 32, "simple v4 uuid");
    }

    #[test]
    fn new_rejects_empty_inputs() {
        let error = Case::new(BTreeMap::new(), map("answer", "hello")).unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("inputs")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn new_rejects_empty_label() {
        let error = Case::new(map("question", "hi"), BTreeMap::new()).unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("label")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn builders_set_tools_and_case_id() {
        let case = sample_case("case-1").with_tools(vec![tool()]);
        assert_eq!(case.case_id, "case-1");
        assert_eq!(case.tools.as_deref(), Some([tool()].as_slice()));
    }

    #[test]
    fn validate_accepts_a_well_formed_case() {
        assert!(sample_case("case-1").validate().is_ok());
    }

    #[test]
    fn validate_rejects_an_empty_case_id() {
        let case = sample_case("   ");
        let error = case.validate().unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("case_id")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn json_round_trip_uses_python_field_names() {
        let case = sample_case("case-1").with_tools(vec![tool()]);
        let value = serde_json::to_value(&case).unwrap();
        let object = value.as_object().unwrap();
        for key in ["inputs", "label", "tools", "case_id"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: Case = serde_json::from_value(value).unwrap();
        assert_eq!(restored, case);
    }

    #[test]
    fn deserialization_generates_a_missing_case_id() {
        let case: Case =
            serde_json::from_str(r#"{"inputs": {"q": "hi"}, "label": {"a": "hello"}}"#).unwrap();
        assert_eq!(case.case_id.len(), 32);
        assert!(case.tools.is_none());

        let other: Case =
            serde_json::from_str(r#"{"inputs": {"q": "hi"}, "label": {"a": "hello"}}"#).unwrap();
        assert_ne!(case.case_id, other.case_id, "ids are unique per payload");
    }

    #[test]
    fn deserialization_ignores_unknown_fields_and_defaults_missing_maps() {
        let case: Case =
            serde_json::from_str(r#"{"case_id": "case-9", "extra": {"nested": [1, 2]}}"#).unwrap();
        assert_eq!(case.case_id, "case-9");
        assert!(case.inputs.is_empty());
        assert!(case.label.is_empty());
        assert!(case.validate().is_err(), "empty maps stay invalid");
    }

    #[test]
    fn evaluated_new_accepts_the_bounds() {
        for score in [0.0, 0.5, 1.0] {
            let evaluated =
                EvaluatedCase::new(sample_case("case-1"), BTreeMap::new(), score, "ok").unwrap();
            assert!((evaluated.score - score).abs() < f64::EPSILON);
            assert_eq!(evaluated.reason, "ok");
        }
    }

    #[test]
    fn evaluated_new_rejects_out_of_range_scores() {
        for score in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = EvaluatedCase::new(sample_case("case-1"), BTreeMap::new(), score, "bad")
                .unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("score")),
                "unexpected error for {score}: {error}"
            );
        }
    }

    #[test]
    fn evaluated_json_round_trip_uses_python_field_names() {
        let evaluated =
            EvaluatedCase::new(sample_case("case-1"), map("answer", "hello"), 0.75, "close")
                .unwrap();
        let value = serde_json::to_value(&evaluated).unwrap();
        let object = value.as_object().unwrap();
        for key in ["case", "answer", "score", "reason"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: EvaluatedCase = serde_json::from_value(value).unwrap();
        assert_eq!(restored, evaluated);
    }

    #[test]
    fn evaluated_deserialization_is_permissive() {
        let evaluated: EvaluatedCase = serde_json::from_str(
            r#"{
                "case": {"inputs": {"q": "hi"}, "label": {"a": "hello"}, "case_id": "c"},
                "score": 4.5,
                "unknown": true
            }"#,
        )
        .unwrap();
        assert!(
            (evaluated.score - 4.5).abs() < f64::EPSILON,
            "score kept as-is"
        );
        assert!(evaluated.answer.is_empty());
        assert_eq!(evaluated.reason, "");
        assert!(evaluated.validate().is_err(), "but validation rejects it");
    }

    #[test]
    fn evaluated_deserialization_defaults_the_score() {
        let evaluated: EvaluatedCase = serde_json::from_str(
            r#"{"case": {"inputs": {"q": "hi"}, "label": {"a": "hello"}, "case_id": "c"}}"#,
        )
        .unwrap();
        assert!(evaluated.score.abs() < f64::EPSILON);
        assert!(evaluated.validate().is_ok());
    }

    #[test]
    fn from_case_yields_a_zero_score_and_mirrored_accessors() {
        let case = sample_case("case-1").with_tools(vec![tool()]);
        let inputs = case.inputs.clone();
        let label = case.label.clone();
        let evaluated = EvaluatedCase::from(case);

        assert!(evaluated.score.abs() < f64::EPSILON);
        assert!(evaluated.answer.is_empty());
        assert_eq!(evaluated.reason, "");
        assert_eq!(evaluated.inputs(), &inputs);
        assert_eq!(evaluated.label(), &label);
        assert_eq!(evaluated.tools(), Some([tool()].as_slice()));
        assert_eq!(evaluated.case_id(), "case-1");
        assert!(evaluated.validate().is_ok());
    }

    #[test]
    fn missing_tools_accessor_is_none() {
        let evaluated = EvaluatedCase::from(sample_case("case-1"));
        assert!(evaluated.tools().is_none());
    }

    #[test]
    fn case_schema_exposes_the_python_fields() {
        let schema = serde_json::to_value(schemars::schema_for!(Case)).unwrap();
        let properties = schema["properties"].as_object().unwrap();
        for key in ["inputs", "label", "tools", "case_id"] {
            assert!(properties.contains_key(key), "missing schema field {key}");
        }
    }
}
