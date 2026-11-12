//! Case datasets: [`Case`], [`EvaluatedCase`], and the [`CaseLoader`].
//!
//! Ports the pure part of Python `openjiuwen.agent_evolving.dataset`:
//!
//! - [`Case`] — one example: the [`inputs`](Case::inputs), the expected
//!   [`label`](Case::label), optional [`tools`](Case::tools), and a
//!   [`case_id`](Case::case_id).
//! - [`EvaluatedCase`] — a [`Case`] plus the [`answer`](EvaluatedCase::answer),
//!   [`score`](EvaluatedCase::score), [`reason`](EvaluatedCase::reason), and the
//!   optional per-metric breakdown [`per_metric`](EvaluatedCase::per_metric).
//! - [`CaseLoader`] — an in-memory case collection with deterministic
//!   [`shuffle`](CaseLoader::shuffle) and [`split`](CaseLoader::split).
//! - JSON/JSONL loading through [`parse_cases_json`], [`parse_cases_jsonl`],
//!   [`parse_cases_str`], and [`load_cases_from_path`].
//!
//! The constructors enforce the Python `pydantic` constraints (non-empty
//! `inputs`/`label`, a score inside `0.0..=1.0`) and report violations as
//! [`openjiuwen_common::error::Error::validation`] instead of panicking, while
//! deserialization stays permissive so malformed-but-parseable JSON still
//! round-trips.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//!
//! use openjiuwen_agent_evolving::dataset::{Case, CaseLoader, parse_cases_json};
//! use serde_json::Value;
//!
//! let mut inputs = BTreeMap::new();
//! inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
//! let mut label = BTreeMap::new();
//! label.insert("answer".to_owned(), Value::String("4".to_owned()));
//!
//! let case = Case::new(inputs, label)?;
//! let loader = CaseLoader::new([case]);
//!
//! let (train, test) = loader.split(0.5)?;
//! assert_eq!(train.size() + test.size(), 1);
//!
//! let parsed = parse_cases_json(r#"[{"inputs": {"q": "hi"}, "label": {"a": "hello"}}]"#)?;
//! assert_eq!(parsed.len(), 1);
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

mod loader;

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::tool::ToolInfo;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use loader::{
    load_cases_from_path, parse_cases_json, parse_cases_jsonl, parse_cases_str, shuffle_cases,
    split_cases, CaseFormat, CaseLoader,
};

/// Generates the default [`Case::case_id`]: a fresh v4 UUID without hyphens.
///
/// Python defaults to `str(uuid.uuid4())`; the hyphen-free form keeps the
/// identifier convenient as a file name or map key without quoting, and stays
/// unique per deserialized case.
fn new_case_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// One evaluation example: the model input plus the expected label.
///
/// Python: `openjiuwen.agent_evolving.dataset.Case`.
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

    /// Checks the Python field constraints: non-empty `inputs` and non-empty
    /// `label`.
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
        Ok(())
    }
}

/// The outcome of running one [`Case`]: the answer, its score, the reason, and
/// an optional per-metric breakdown.
///
/// Python: `openjiuwen.agent_evolving.dataset.EvaluatedCase`. The accessors
/// mirror Python's properties, which forward to the wrapped case. `answer` and
/// `per_metric` serialize as JSON `null` when absent, matching Python
/// `model_dump`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct EvaluatedCase {
    /// The case that was evaluated.
    pub case: Case,
    /// The agent answer; `None` when nothing was produced.
    #[serde(default)]
    pub answer: Option<BTreeMap<String, Value>>,
    /// The score in `0.0..=1.0`; deserialization does not enforce the bounds.
    #[serde(default)]
    pub score: f64,
    /// The human-readable justification of `score`.
    #[serde(default)]
    pub reason: String,
    /// The optional per-metric scores that produced `score`.
    #[serde(default)]
    pub per_metric: Option<BTreeMap<String, f64>>,
}

impl EvaluatedCase {
    /// Creates an evaluated case, clamping the score into `0.0..=1.0`.
    ///
    /// Python's constructor clamps finite scores; only non-finite values are
    /// rejected.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `score` is not finite.
    pub fn new(
        case: Case,
        answer: Option<BTreeMap<String, Value>>,
        score: f64,
        reason: impl Into<String>,
    ) -> Result<Self> {
        if !score.is_finite() {
            return Err(Error::validation(format!(
                "EvaluatedCase.score must be finite, got {score}"
            )));
        }
        Ok(Self {
            case,
            answer,
            score: score.clamp(0.0, 1.0),
            reason: reason.into(),
            per_metric: None,
        })
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
            answer: None,
            score: 0.0,
            reason: String::new(),
            per_metric: None,
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
    fn new_rejects_empty_inputs_and_label() {
        let error = Case::new(BTreeMap::new(), map("answer", "hello")).unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("inputs")),
            "unexpected error: {error}"
        );

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
        assert!(case.validate().is_ok());
    }

    #[test]
    fn evaluated_new_clamps_and_rejects_non_finite_scores() {
        let evaluated = EvaluatedCase::new(sample_case("case-1"), None, 1.5, "high").unwrap();
        assert!((evaluated.score - 1.0).abs() < f64::EPSILON);
        assert!(evaluated.answer.is_none());
        assert!(evaluated.per_metric.is_none());

        let evaluated = EvaluatedCase::new(sample_case("case-1"), None, -2.0, "low").unwrap();
        assert!(evaluated.score.abs() < f64::EPSILON);

        for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = EvaluatedCase::new(sample_case("case-1"), None, score, "bad").unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("score")),
                "unexpected error for {score}: {error}"
            );
        }
    }

    #[test]
    fn evaluated_validate_rejects_out_of_bounds_scores() {
        let mut evaluated = EvaluatedCase::from(sample_case("case-1"));
        assert!(evaluated.validate().is_ok());
        evaluated.score = 1.5;
        assert!(evaluated.validate().is_err());
        evaluated.score = f64::NAN;
        assert!(evaluated.validate().is_err());
    }

    #[test]
    fn from_case_yields_a_zero_score_and_mirrored_accessors() {
        let case = sample_case("case-1").with_tools(vec![tool()]);
        let inputs = case.inputs.clone();
        let label = case.label.clone();
        let evaluated = EvaluatedCase::from(case);

        assert!(evaluated.score.abs() < f64::EPSILON);
        assert!(evaluated.answer.is_none());
        assert!(evaluated.per_metric.is_none());
        assert_eq!(evaluated.reason, "");
        assert_eq!(evaluated.inputs(), &inputs);
        assert_eq!(evaluated.label(), &label);
        assert_eq!(evaluated.tools(), Some([tool()].as_slice()));
        assert_eq!(evaluated.case_id(), "case-1");
    }

    #[test]
    fn json_round_trips_use_python_field_names() {
        let case = sample_case("case-1").with_tools(vec![tool()]);
        let value = serde_json::to_value(&case).unwrap();
        let object = value.as_object().unwrap();
        for key in ["inputs", "label", "tools", "case_id"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: Case = serde_json::from_value(value).unwrap();
        assert_eq!(restored, case);

        let evaluated = EvaluatedCase::new(
            sample_case("case-1"),
            Some(map("answer", "hello")),
            0.75,
            "close",
        )
        .unwrap();
        let value = serde_json::to_value(&evaluated).unwrap();
        let object = value.as_object().unwrap();
        for key in ["case", "answer", "score", "reason", "per_metric"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        assert_eq!(value["per_metric"], Value::Null);
        let restored: EvaluatedCase = serde_json::from_value(value).unwrap();
        assert_eq!(restored, evaluated);
    }
}
