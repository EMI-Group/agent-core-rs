//! Report models for design, functional, safety, and final scoring.
//!
//! The report types mirror the JSON artifacts of the Python evaluator, so field
//! names and nesting are kept verbatim: `design_score_pct`,
//! `ready_for_functional_testing`, `e1_llm_judge`, `pipeline_status`, and so on.
//! Every type is `serde`-serializable with `snake_case` field names and tolerates
//! unknown fields, so persisted reports from older revisions still deserialize.
//!
//! [`EvaluationReport::to_json`] and [`EvaluationReport::to_json_string`] are the
//! serialization entry points; [`grade_for`] is the shared A–F mapping.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::judge::VerdictSource;

use super::assertion::AssertionOutcome;
use super::case::CaseStatus;

/// One rubric dimension: the points awarded and the points available.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ScoreDimension {
    /// Points awarded by this dimension.
    pub score: f64,
    /// Points available to this dimension.
    pub max: f64,
}

/// The structural design review of a skill.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DesignReport {
    /// The reviewed skill's name.
    pub skill_name: String,
    /// Weighted points awarded across all dimensions.
    pub design_score: f64,
    /// Total points available.
    pub design_score_max: f64,
    /// [`design_score`](Self::design_score) as a percentage, rounded to 2 decimals.
    pub design_score_pct: f64,
    /// Letter grade for [`design_score_pct`](Self::design_score_pct).
    pub grade: String,
    /// Fraction of the rubric satisfied, rounded to 2 decimals.
    pub knowledge_delta: f64,
    /// Whether the design clears the functional-testing gate.
    pub ready_for_functional_testing: bool,
    /// Per-dimension scores keyed by dimension name.
    pub dimensions: BTreeMap<String, ScoreDimension>,
}

/// Aggregate counts over the functional cases.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseSummary {
    /// Total number of executed cases.
    pub total: usize,
    /// Cases whose assertions all passed.
    pub passed: usize,
    /// Cases with a mix of passing and failing assertions.
    pub partial: usize,
    /// Cases with no passing assertion.
    pub failed: usize,
    /// `(passed + 0.5 * partial) / total`, in `0.0..=1.0`.
    pub pass_rate: f64,
}

/// The result of one executed case.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseResult {
    /// The case identifier.
    pub id: String,
    /// The case name.
    pub name: String,
    /// The Overall status derived from the assertion outcomes.
    pub status: CaseStatus,
    /// One outcome per checked expectation.
    pub assertions: Vec<AssertionOutcome>,
}

/// The functional (assertion-based) evaluation of a skill.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FunctionalReport {
    /// The evaluated skill's name.
    pub skill_name: String,
    /// Where the skill bundle lives.
    pub skill_path: String,
    /// Unix timestamp of the run, supplied by the caller.
    pub timestamp: i64,
    /// Aggregate counts over [`results`](Self::results).
    pub summary: CaseSummary,
    /// One entry per executed case, in execution order.
    pub results: Vec<CaseResult>,
    /// Free-form next steps for the skill author.
    pub recommendations: Vec<String>,
}

/// One hazard found by the safety scan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SafetyFinding {
    /// The safety dimension the rule belongs to.
    pub dimension: String,
    /// The finding severity, e.g. `info`, `warning`, or `critical`.
    pub severity: String,
    /// The human-readable explanation.
    pub message: String,
}

/// The regex-driven safety review of a skill document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SafetyReport {
    /// The scanned skill's name.
    pub skill_name: String,
    /// Safety score after the per-finding penalty, clamped at zero.
    pub score_pct: f64,
    /// The gate threshold expressed as a fraction of `1.0`.
    pub gate_threshold: f64,
    /// Whether the gate is closed (score below threshold or a critical finding).
    pub blocked: bool,
    /// Whether the block is an automatic failure; always mirrors
    /// [`blocked`](Self::blocked).
    pub auto_fail: bool,
    /// Per-dimension scores; `None` because the scan is not dimension-scored.
    pub dimensions: Option<BTreeMap<String, ScoreDimension>>,
    /// One entry per matching rule, in rule order.
    pub findings: Vec<SafetyFinding>,
}

/// One evaluator's contribution to the final score.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluatorScore {
    /// The evaluator's own score, as a percentage.
    pub score_pct: f64,
    /// The weight applied to [`score_pct`](Self::score_pct), as a fraction.
    pub weight: f64,
    /// `weight * score_pct`, rounded to 2 decimals.
    pub contribution: f64,
}

/// The per-evaluator scores behind a [`FinalScore`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluatorScores {
    /// The design (LLM-judge style) evaluator; absent when it did not run.
    pub e1_llm_judge: Option<EvaluatorScore>,
    /// The functional (case-runner) evaluator.
    pub e2_functional: EvaluatorScore,
}

/// The weighted final verdict over a skill.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FinalScore {
    /// The scored skill's name.
    pub skill_name: String,
    /// Where the pipeline stopped: `COMPLETED`, `BLOCKED_AT_E1_LLM_JUDGE`, or
    /// `BLOCKED_AT_SAFETY`.
    pub pipeline_status: String,
    /// The weighted final score in percentage points.
    pub final_score_pct: f64,
    /// Letter grade for [`final_score_pct`](Self::final_score_pct).
    pub grade: String,
    /// The evaluator contributions.
    pub evaluators: EvaluatorScores,
    /// The gates that passed, in evaluation order.
    pub gates_passed: Vec<String>,
    /// The release recommendation.
    pub recommendation: String,
}

/// Where the design and safety scores of a judge-driven run came from.
///
/// A deterministic run carries no record at all; a judge-driven run records one
/// [`VerdictSource`] per judged stage so a degraded (fallback) judgement is
/// visible in the persisted report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeRecord {
    /// The source of the design judgement.
    pub design_source: VerdictSource,
    /// The source of the safety judgement.
    pub safety_source: VerdictSource,
}

/// Every report produced by one evaluation run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationReport {
    /// The design review.
    pub design: DesignReport,
    /// The functional evaluation.
    pub functional: FunctionalReport,
    /// The safety scan.
    pub safety: SafetyReport,
    /// The weighted final verdict.
    pub final_score: FinalScore,
    /// Judge provenance; `None` for the deterministic pipeline and omitted from
    /// the serialized report when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeRecord>,
}

/// Derives the per-case status from its assertion outcomes.
///
/// All assertions passing is [`CaseStatus::Passed`], none passing is
/// [`CaseStatus::Failed`], and anything in between is [`CaseStatus::Partial`].
pub(crate) fn status_from(assertions: &[AssertionOutcome]) -> CaseStatus {
    let passed = assertions.iter().filter(|outcome| outcome.passed).count();
    if passed == assertions.len() {
        CaseStatus::Passed
    } else if passed == 0 {
        CaseStatus::Failed
    } else {
        CaseStatus::Partial
    }
}

/// Aggregates per-case results into the shared functional summary.
///
/// The pass rate is `(passed + 0.5 * partial) / total`; an empty result list
/// yields `0.0`. Both the deterministic evaluator and the evaluation harness use
/// this function so their summaries cannot drift apart.
pub(crate) fn summarize(results: &[CaseResult]) -> CaseSummary {
    let total = results.len();
    let passed = results
        .iter()
        .filter(|result| result.status == CaseStatus::Passed)
        .count();
    let partial = results
        .iter()
        .filter(|result| result.status == CaseStatus::Partial)
        .count();
    let failed = results
        .iter()
        .filter(|result| result.status == CaseStatus::Failed)
        .count();
    let pass_rate = if total == 0 {
        0.0
    } else {
        (passed as f64 + 0.5 * partial as f64) / total as f64
    };
    CaseSummary {
        total,
        passed,
        partial,
        failed,
        pass_rate,
    }
}

impl EvaluationReport {
    /// Serializes the report into a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::external`](openjiuwen_common::error::Error::external)
    /// when the report cannot be converted to JSON.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(Error::external)
    }

    /// Serializes the report into a compact JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::external`](openjiuwen_common::error::Error::external)
    /// when the report cannot be serialized.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Error::external)
    }
}

/// Maps a percentage score to a letter grade.
///
/// `A` at 90.0, `B` at 80.0, `C` at 70.0, `D` at 60.0, otherwise `F`. The
/// comparisons are ordered so that a negative or NaN score yields `F`.
#[must_use]
pub fn grade_for(score_pct: f64) -> String {
    let grade = if score_pct >= 90.0 {
        "A"
    } else if score_pct >= 80.0 {
        "B"
    } else if score_pct >= 70.0 {
        "C"
    } else if score_pct >= 60.0 {
        "D"
    } else {
        "F"
    };
    grade.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_boundaries() {
        assert_eq!(grade_for(100.0), "A");
        assert_eq!(grade_for(90.0), "A");
        assert_eq!(grade_for(89.99), "B");
        assert_eq!(grade_for(80.0), "B");
        assert_eq!(grade_for(79.9), "C");
        assert_eq!(grade_for(70.0), "C");
        assert_eq!(grade_for(69.99), "D");
        assert_eq!(grade_for(60.0), "D");
        assert_eq!(grade_for(59.9), "F");
        assert_eq!(grade_for(0.0), "F");
        assert_eq!(grade_for(-1.0), "F");
        assert_eq!(grade_for(f64::NAN), "F");
    }

    #[test]
    fn report_round_trips_through_json() {
        let mut report = EvaluationReport::default();
        report.design.skill_name = "demo".to_string();
        report.design.design_score_pct = 80.0;
        report.design.ready_for_functional_testing = true;
        report.functional.summary.pass_rate = 0.5;
        report.safety.score_pct = 100.0;
        report.final_score.pipeline_status = "COMPLETED".to_string();

        let json = report.to_json().expect("to_json");
        let back: EvaluationReport = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, report);

        let text = report.to_json_string().expect("to_json_string");
        assert!(text.contains("\"design\""));
    }

    #[test]
    fn to_json_has_the_top_level_keys() {
        let report = EvaluationReport::default();
        let json = report.to_json().expect("to_json");
        let object = json.as_object().expect("object");
        for key in ["design", "functional", "safety", "final_score"] {
            assert!(object.contains_key(key), "missing key {key}");
        }
    }

    #[test]
    fn safety_dimensions_serializes_null() {
        let report = SafetyReport::default();
        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(json["dimensions"], serde_json::Value::Null);
    }

    #[test]
    fn score_dimension_round_trips() {
        let dimension = ScoreDimension {
            score: 7.5,
            max: 15.0,
        };
        let json = serde_json::to_string(&dimension).expect("serialize");
        let back: ScoreDimension = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, dimension);
    }

    #[test]
    fn deserialize_tolerates_unknown_fields() {
        let report: DesignReport = serde_json::from_value(serde_json::json!({
            "skill_name": "demo",
            "design_score": 100.0,
            "extra_field": [1, 2, 3]
        }))
        .expect("deserialize");
        assert_eq!(report.skill_name, "demo");
        assert!((report.design_score - 100.0).abs() < 1e-9);
        assert!(report.dimensions.is_empty());
    }

    fn outcome(passed: bool) -> AssertionOutcome {
        AssertionOutcome {
            text: "contains:x".to_string(),
            passed,
            evidence: String::new(),
        }
    }

    #[test]
    fn status_from_covers_all_and_none_and_mixed() {
        assert_eq!(status_from(&[outcome(true), outcome(true)]), CaseStatus::Passed);
        assert_eq!(status_from(&[outcome(false)]), CaseStatus::Failed);
        assert_eq!(status_from(&[outcome(true), outcome(false)]), CaseStatus::Partial);
        assert_eq!(status_from(&[]), CaseStatus::Passed);
    }

    #[test]
    fn summarize_credits_partial_cases_with_half_a_pass() {
        let results = [
            CaseResult {
                id: "a".to_string(),
                status: CaseStatus::Passed,
                ..CaseResult::default()
            },
            CaseResult {
                id: "b".to_string(),
                status: CaseStatus::Partial,
                ..CaseResult::default()
            },
            CaseResult {
                id: "c".to_string(),
                status: CaseStatus::Failed,
                ..CaseResult::default()
            },
        ];
        let summary = summarize(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.partial, 1);
        assert_eq!(summary.failed, 1);
        assert!((summary.pass_rate - 0.5).abs() < 1e-9);

        let empty = summarize(&[]);
        assert_eq!(empty.total, 0);
        assert!((empty.pass_rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn judge_record_is_omitted_when_absent() {
        let report = EvaluationReport::default();
        let json = report.to_json().expect("to_json");
        assert!(json.get("judge").is_none(), "{json}");

        let judged = EvaluationReport {
            judge: Some(JudgeRecord {
                design_source: VerdictSource::Model,
                safety_source: VerdictSource::Deterministic,
            }),
            ..EvaluationReport::default()
        };
        let json = judged.to_json().expect("to_json");
        assert_eq!(json["judge"]["design_source"], "model");
        assert_eq!(json["judge"]["safety_source"], "deterministic");

        let back: EvaluationReport = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, judged);
    }
}
