//! Evaluation cases and per-case status.
//!
//! An [`EvalCase`] is one prompt plus the expectations a skill's answer must
//! satisfy. Python's `EvalCase` dataclass carried the same fields plus loose
//! `Any` metadata; the Rust port keeps the fields that drive evaluation and
//! drops the unconstrained payload.
//!
//! Cases are plain data: construction goes through the builders, and
//! [`EvalCase::validate`] is the only fallible operation.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// One skill-evaluation case: a prompt and the expectations on its answer.
///
/// A case is runnable once it has an identity (non-blank [`id`](Self::id) and
/// [`name`](Self::name)), a non-blank [`prompt`](Self::prompt), and at least one
/// [`expectation`](Self::expectations); see [`validate`](Self::validate).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvalCase {
    /// Stable case identifier, unique inside one evaluation run.
    pub id: String,
    /// Human-readable case name.
    pub name: String,
    /// Case category, e.g. `functional` or `safety`; empty when uncategorized.
    pub case_category: String,
    /// The user prompt handed to the skill.
    pub prompt: String,
    /// Optional reference answer, informational only.
    pub expected_output: String,
    /// Input files the case needs, as skill-relative paths.
    pub files: Vec<String>,
    /// Expressed expectations, parsed by [`crate::skill_evaluator::Assertion::parse`].
    pub expectations: Vec<String>,
}

impl EvalCase {
    /// Creates a case with an id, a name, and a prompt.
    ///
    /// The category, expected output, files, and expectations start empty.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            prompt: prompt.into(),
            case_category: String::new(),
            expected_output: String::new(),
            files: Vec::new(),
            expectations: Vec::new(),
        }
    }

    /// Appends one expectation.
    #[must_use]
    pub fn with_expectation(mut self, expectation: impl Into<String>) -> Self {
        self.expectations.push(expectation.into());
        self
    }

    /// Appends every expectation of an iterator, preserving order.
    #[must_use]
    pub fn with_expectations(
        mut self,
        expectations: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.expectations
            .extend(expectations.into_iter().map(Into::into));
        self
    }

    /// Sets the reference answer.
    #[must_use]
    pub fn with_expected_output(mut self, output: impl Into<String>) -> Self {
        self.expected_output = output.into();
        self
    }

    /// Sets the case category.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.case_category = category.into();
        self
    }

    /// Appends every input file path of an iterator, preserving order.
    #[must_use]
    pub fn with_files(mut self, files: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.files.extend(files.into_iter().map(Into::into));
        self
    }

    /// Checks that the case can be evaluated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when `id`, `name`, or `prompt` is blank, or when the case has no
    /// expectation.
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(Error::validation("evaluation case id must not be blank"));
        }
        if self.name.trim().is_empty() {
            return Err(Error::validation("evaluation case name must not be blank"));
        }
        if self.prompt.trim().is_empty() {
            return Err(Error::validation(format!(
                "evaluation case {:?}: prompt must not be blank",
                self.id
            )));
        }
        if self.expectations.is_empty() {
            return Err(Error::validation(format!(
                "evaluation case {:?}: at least one expectation is required",
                self.id
            )));
        }
        Ok(())
    }
}

/// The outcome of running one evaluation case.
///
/// Serialized as `"passed"`, `"partial"`, or `"failed"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    /// Every assertion passed.
    #[default]
    Passed,
    /// Some assertions passed and some failed.
    Partial,
    /// No assertion passed.
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runnable() -> EvalCase {
        EvalCase::new("case-1", "First case", "Summarize the document.")
            .with_expectation("contains:summary")
    }

    #[test]
    fn builders_set_every_field() {
        let case = EvalCase::new("case-1", "First case", "Summarize the document.")
            .with_category("functional")
            .with_expected_output("A summary.")
            .with_files(["references/a.md", "references/b.md"])
            .with_expectation("contains:summary")
            .with_expectations(["equals:x", "regex:^a+$"]);

        assert_eq!(case.id, "case-1");
        assert_eq!(case.name, "First case");
        assert_eq!(case.case_category, "functional");
        assert_eq!(case.prompt, "Summarize the document.");
        assert_eq!(case.expected_output, "A summary.");
        assert_eq!(case.files, vec!["references/a.md", "references/b.md"]);
        assert_eq!(
            case.expectations,
            vec!["contains:summary", "equals:x", "regex:^a+$"]
        );
    }

    #[test]
    fn validate_accepts_runnable_case() {
        assert!(runnable().validate().is_ok());
    }

    #[test]
    fn validate_rejects_missing_expectation() {
        let case = EvalCase::new("case-1", "First case", "Summarize the document.");
        let error = case.validate().expect_err("missing expectation");
        assert!(error.to_string().contains("expectation"));
    }

    #[test]
    fn validate_rejects_blank_id() {
        let case = runnable();
        let case = EvalCase {
            id: "   ".to_string(),
            ..case
        };
        let error = case.validate().expect_err("blank id");
        assert!(error.to_string().contains("id"));
    }

    #[test]
    fn validate_rejects_blank_name() {
        let case = EvalCase {
            name: String::new(),
            ..runnable()
        };
        let error = case.validate().expect_err("blank name");
        assert!(error.to_string().contains("name"));
    }

    #[test]
    fn validate_rejects_blank_prompt() {
        let case = EvalCase {
            prompt: " \n ".to_string(),
            ..runnable()
        };
        let error = case.validate().expect_err("blank prompt");
        assert!(error.to_string().contains("prompt"));
    }

    #[test]
    fn status_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_value(CaseStatus::Passed).expect("serialize"),
            serde_json::json!("passed")
        );
        assert_eq!(
            serde_json::to_value(CaseStatus::Partial).expect("serialize"),
            serde_json::json!("partial")
        );
        assert_eq!(
            serde_json::to_value(CaseStatus::Failed).expect("serialize"),
            serde_json::json!("failed")
        );
        assert_eq!(CaseStatus::default(), CaseStatus::Passed);
    }

    #[test]
    fn deserialize_tolerates_unknown_fields() {
        let case: EvalCase = serde_json::from_value(serde_json::json!({
            "id": "case-1",
            "name": "First case",
            "prompt": "Summarize.",
            "expectations": ["contains:x"],
            "unknown": {"nested": true}
        }))
        .expect("deserialize");
        assert_eq!(case.id, "case-1");
        assert_eq!(case.case_category, "");
        assert_eq!(case.files, Vec::<String>::new());
    }

    #[test]
    fn case_round_trips_through_json() {
        let case = runnable();
        let json = serde_json::to_string(&case).expect("serialize");
        let back: EvalCase = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, case);
    }
}
