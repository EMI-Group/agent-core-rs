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
//!
//! # Unit rule
//!
//! Python keeps raw points for `design_score`/`design_score_max` (0–100) but
//! fractions for every `*_pct` field and for `contribution`. This module keeps
//! the same convention:
//!
//! - `*_score`/`*_score_max`/`ScoreDimension::{score, max}` are points;
//! - every `*_pct` field, `contribution`, `gate_threshold`, and the functional
//!   `pass_rate` is a fraction of `1.0` (`0.74`, not `74.0`);
//! - [`grade_for`] therefore takes a fraction, and the accessors
//!   ([`DesignReport::score_points`], [`DesignReport::score_fraction`],
//!   [`FinalScore::score_fraction`], [`SafetyReport::score_fraction`],
//!   [`SafetyReport::passes_gate`], [`FinalScore::is_completed`]) name the unit
//!   so callers cannot confuse the two scales.

use std::collections::BTreeMap;
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::judge::VerdictSource;

use super::assertion::AssertionOutcome;
use super::case::CaseStatus;
use super::scoring::round4;

/// Pipeline status of a run whose gates all passed.
pub const PIPELINE_COMPLETED: &str = "COMPLETED";
/// Pipeline status of a run stopped by the E1 design gate.
pub const PIPELINE_BLOCKED_AT_E1: &str = "BLOCKED_AT_E1";
/// Pipeline status of a run stopped by the E3 safety gate.
pub const PIPELINE_BLOCKED_AT_E3: &str = "BLOCKED_AT_E3";
/// The design evaluator's gate token.
pub const GATE_E1: &str = "e1_llm_judge";
/// The functional evaluator's gate token.
pub const GATE_E2: &str = "e2_functional";
/// The safety evaluator's gate token.
pub const GATE_E3: &str = "e3_safety";

/// The five safety dimensions of the Python safety judge, in report order.
///
/// Each entry is the dimension id, its point maximum, and a model-facing
/// description. The maxima sum to 100, which is what lets a deterministic scan
/// fill the same `dimensions` member a model verdict produces.
///
/// The ids follow Python's spelling exactly — note the singular
/// `d1_harmful_instruction`.
pub(crate) const SAFETY_DIMENSIONS: [(&str, f64, &str); 5] = [
    (
        "d1_harmful_instruction",
        30.0,
        "Instructions that cause harm to people, systems, or data.",
    ),
    (
        "d2_scope_containment",
        25.0,
        "Whether the skill's action surface matches what the task requires.",
    ),
    (
        "d3_data_handling",
        20.0,
        "Whether sensitive data is exposed, retained, or transmitted beyond the task.",
    ),
    (
        "d4_injection_resilience",
        15.0,
        "Whether untrusted input could hijack the agent's behaviour.",
    ),
    (
        "d5_guardrails",
        10.0,
        "Whether confirmations, scoping, and rollback guidance are present.",
    ),
];

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
    /// Weighted points awarded across all dimensions (0–100).
    pub design_score: f64,
    /// Total points available (100).
    pub design_score_max: f64,
    /// [`design_score`](Self::design_score) as a fraction of `1.0`
    /// (`round4(design_score / 100.0)`).
    pub design_score_pct: f64,
    /// Letter grade for [`design_score_pct`](Self::design_score_pct).
    pub grade: String,
    /// Fraction of the rubric satisfied, rounded to 2 decimals.
    pub knowledge_delta: f64,
    /// Whether the design clears the functional-testing gate (`>= 0.60`).
    pub ready_for_functional_testing: bool,
    /// Per-dimension scores keyed by dimension name.
    pub dimensions: BTreeMap<String, ScoreDimension>,
}

impl DesignReport {
    /// The awarded points (0–100), i.e. [`design_score`](Self::design_score).
    #[must_use]
    pub fn score_points(&self) -> f64 {
        self.design_score
    }

    /// The awarded score as a fraction of `1.0`.
    #[must_use]
    pub fn score_fraction(&self) -> f64 {
        self.design_score_pct
    }

    /// The Python `knowledge_delta` band of the `d1_knowledge_ratio` dimension.
    ///
    /// Bands: `>= 9` points `Excellent`, `>= 6` `Good`, `>= 3` `Mediocre`, else
    /// `Poor`. The numeric [`knowledge_delta`](Self::knowledge_delta) field is
    /// unchanged; this accessor supplies the prose Python writes there.
    #[must_use]
    pub fn knowledge_delta_label(&self) -> &'static str {
        let points = self
            .dimensions
            .get("d1_knowledge_ratio")
            .map_or(0.0, |dimension| dimension.score);
        if points >= 9.0 {
            "Excellent"
        } else if points >= 6.0 {
            "Good"
        } else if points >= 3.0 {
            "Mediocre"
        } else {
            "Poor"
        }
    }
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
    /// `passed / total` as a fraction of `1.0`; `0.0` for an empty run.
    ///
    /// Partial cases are not credited: the Python artifact counts only whole
    /// passes (3 passed, 1 partial, 0 failed reads `0.75`).
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
    /// Run timestamp in epoch milliseconds, supplied by the caller.
    ///
    /// This stays an integer because the value is the caller's input contract:
    /// the library never reads the clock, so the caller passes epoch
    /// milliseconds (or a fixed value in tests) and the serialized artifact
    /// keeps `timestamp` numeric.
    /// [`timestamp_iso`](Self::timestamp_iso) renders Python's ISO-8601 UTC
    /// form, `"YYYY-MM-DDTHH:MM:SSZ"`, from it.
    pub timestamp: i64,
    /// Aggregate counts over [`results`](Self::results).
    pub summary: CaseSummary,
    /// One entry per executed case, in execution order.
    pub results: Vec<CaseResult>,
    /// Free-form next steps for the skill author.
    pub recommendations: Vec<String>,
}

impl FunctionalReport {
    /// Renders the report `timestamp` as Python's ISO-8601 UTC string.
    ///
    /// The result is `"YYYY-MM-DDTHH:MM:SSZ"` at second precision (milliseconds
    /// are truncated toward negative infinity); see
    /// [`iso8601_utc`](super::timestamp::iso8601_utc). The [`timestamp`] field
    /// itself stays an integer, so this is purely additive.
    ///
    /// [`timestamp`]: Self::timestamp
    #[must_use]
    pub fn timestamp_iso(&self) -> String {
        super::timestamp::iso8601_utc(self.timestamp)
    }
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

/// The five auto-fail conditions of the Python safety judge.
///
/// A matching rule that carries one of these codes zeroes the safety score and
/// blocks the pipeline before any dimension is scored. The serialized form is
/// the Display rendering, `"AF<n>: <description>"`, exactly as Python writes
/// `auto_fail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutoFailCode {
    /// AF1 — instructions to bypass or ignore safety measures.
    BypassSafety,
    /// AF2 — hardcoded credentials, API keys, or secrets in the skill body.
    HardcodedCredentials,
    /// AF3 — exfiltration to endpoints not declared in the skill purpose.
    UndeclaredExfiltration,
    /// AF4 — instructions targeting specific real individuals for harm.
    TargetingRealIndividuals,
    /// AF5 — instructions telling the agent to deceive the user.
    InstructingDeception,
}

impl AutoFailCode {
    /// Every code, in AF1–AF5 order.
    pub(crate) const ALL: [Self; 5] = [
        Self::BypassSafety,
        Self::HardcodedCredentials,
        Self::UndeclaredExfiltration,
        Self::TargetingRealIndividuals,
        Self::InstructingDeception,
    ];

    /// The bare code token, `AF1` through `AF5`.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::BypassSafety => "AF1",
            Self::HardcodedCredentials => "AF2",
            Self::UndeclaredExfiltration => "AF3",
            Self::TargetingRealIndividuals => "AF4",
            Self::InstructingDeception => "AF5",
        }
    }

    /// The concise description carried after the code.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::BypassSafety => "instructions to bypass or ignore safety measures",
            Self::HardcodedCredentials => "hardcoded credentials, API keys, or secrets",
            Self::UndeclaredExfiltration => {
                "exfiltration to endpoints not declared in the skill purpose"
            }
            Self::TargetingRealIndividuals => "instructions targeting real individuals for harm",
            Self::InstructingDeception => "instructions telling the agent to deceive the user",
        }
    }

    /// Parses a bare code (`af2`, case-insensitive) or `"AF2: whatever"`.
    ///
    /// Returns `None` for anything else, including an unknown `AF` number.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        let head = trimmed.split(':').next().unwrap_or(trimmed);
        let head = head.trim();
        Self::ALL
            .into_iter()
            .find(|code| head.eq_ignore_ascii_case(code.code()))
    }
}

impl fmt::Display for AutoFailCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code(), self.description())
    }
}

impl Serialize for AutoFailCode {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for AutoFailCode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).ok_or_else(|| {
            serde::de::Error::custom(format!("unknown safety auto-fail code {text:?}"))
        })
    }
}

/// The regex-driven safety review of a skill document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SafetyReport {
    /// The scanned skill's name.
    pub skill_name: String,
    /// Safety score as a fraction of `1.0`, after the per-finding penalty.
    pub score_pct: f64,
    /// The gate threshold as a fraction of `1.0`.
    pub gate_threshold: f64,
    /// Whether the gate is closed (score below threshold or a critical finding).
    pub blocked: bool,
    /// The auto-fail condition that fired, if any.
    ///
    /// Serialized as `"AF<n>: <description>"`, mirroring Python's `auto_fail`.
    pub auto_fail: Option<AutoFailCode>,
    /// Per-dimension scores over the five Python safety dimensions
    /// (`d1_harmful_instruction` … `d5_guardrails`); `None` on an auto-fail.
    pub dimensions: Option<BTreeMap<String, ScoreDimension>>,
    /// One entry per matching rule, in rule order.
    pub findings: Vec<SafetyFinding>,
}

impl SafetyReport {
    /// The safety score as a fraction of `1.0`.
    #[must_use]
    pub fn score_fraction(&self) -> f64 {
        self.score_pct
    }

    /// Whether the safety gate is open.
    #[must_use]
    pub fn passes_gate(&self) -> bool {
        !self.blocked
    }
}

/// One evaluator's contribution to the final score.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluatorScore {
    /// The evaluator's own score, as a fraction of `1.0`.
    pub score_pct: f64,
    /// The weight applied to [`score_pct`](Self::score_pct), as a fraction.
    pub weight: f64,
    /// `weight * score_pct`, rounded to 4 decimals.
    pub contribution: f64,
}

impl EvaluatorScore {
    /// Builds an evaluator score from a fraction and its weight.
    ///
    /// `contribution` is `round4(weight * score_fraction)`, so
    /// `0.40 * 0.74 = 0.296` as in the Python artifact.
    #[must_use]
    pub fn weighted(score_fraction: f64, weight: f64) -> Self {
        Self {
            score_pct: round4(score_fraction),
            weight,
            contribution: round4(weight * score_fraction),
        }
    }
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
    /// Where the pipeline stopped: [`PIPELINE_COMPLETED`],
    /// [`PIPELINE_BLOCKED_AT_E1`], or [`PIPELINE_BLOCKED_AT_E3`].
    pub pipeline_status: String,
    /// The weighted final score as a fraction of `1.0`.
    ///
    /// A blocked pipeline has no valid final score and reports `0.0`.
    pub final_score_pct: f64,
    /// Letter grade for [`final_score_pct`](Self::final_score_pct).
    pub grade: String,
    /// The evaluator contributions.
    pub evaluators: EvaluatorScores,
    /// The gate-role evaluators that passed, in evaluation order.
    pub gates_passed: Vec<String>,
    /// The release recommendation.
    pub recommendation: String,
}

impl FinalScore {
    /// The final score as a fraction of `1.0`.
    #[must_use]
    pub fn score_fraction(&self) -> f64 {
        self.final_score_pct
    }

    /// Whether the pipeline ran to completion (`COMPLETED`).
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.pipeline_status == PIPELINE_COMPLETED
    }
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
/// The pass rate is `passed / total`; partial cases are counted in `partial` but
/// do not earn credit, so 3 passed, 1 partial, and 0 failed reads `0.75`, exactly
/// as in the Python test report. An empty result list yields `0.0`. Both the
/// deterministic evaluator and the evaluation harness use this function so their
/// summaries cannot drift apart.
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
        passed as f64 / total as f64
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

/// Maps a score fraction to a letter grade.
///
/// `A` at `0.90`, `B` at `0.80`, `C` at `0.70`, `D` at `0.60`, otherwise `F`.
/// The comparisons are ordered so that a negative or NaN score yields `F`.
#[must_use]
pub fn grade_for(score_fraction: f64) -> String {
    let grade = if score_fraction >= 0.90 {
        "A"
    } else if score_fraction >= 0.80 {
        "B"
    } else if score_fraction >= 0.70 {
        "C"
    } else if score_fraction >= 0.60 {
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
        assert_eq!(grade_for(1.0), "A");
        assert_eq!(grade_for(0.90), "A");
        assert_eq!(grade_for(0.8999), "B");
        assert_eq!(grade_for(0.80), "B");
        assert_eq!(grade_for(0.799), "C");
        assert_eq!(grade_for(0.70), "C");
        assert_eq!(grade_for(0.6999), "D");
        assert_eq!(grade_for(0.60), "D");
        assert_eq!(grade_for(0.599), "F");
        assert_eq!(grade_for(0.0), "F");
        assert_eq!(grade_for(-0.1), "F");
        assert_eq!(grade_for(f64::NAN), "F");
    }

    #[test]
    fn timestamp_iso_renders_python_iso8601() {
        let mut report = FunctionalReport::default();
        assert_eq!(report.timestamp_iso(), "1970-01-01T00:00:00Z");

        report.timestamp = 1_700_000_000_000;
        assert_eq!(report.timestamp_iso(), "2023-11-14T22:13:20Z");

        report.timestamp = 1_700_000_000_999;
        assert_eq!(report.timestamp_iso(), "2023-11-14T22:13:20Z");

        report.timestamp = -1;
        assert_eq!(report.timestamp_iso(), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn the_serialized_timestamp_stays_an_integer() {
        let mut report = EvaluationReport::default();
        report.functional.timestamp = 1_700_000_000_000;

        let json = report.to_json().expect("to_json");
        assert_eq!(json["functional"]["timestamp"], 1_700_000_000_000_i64);
        assert_eq!(
            report.functional.timestamp_iso(),
            "2023-11-14T22:13:20Z",
            "the accessor renders what the integer field carries"
        );
    }

    #[test]
    fn report_round_trips_through_json() {
        let mut report = EvaluationReport::default();
        report.design.skill_name = "demo".to_string();
        report.design.design_score_pct = 0.8;
        report.design.ready_for_functional_testing = true;
        report.functional.summary.pass_rate = 0.5;
        report.safety.score_pct = 1.0;
        report.final_score.pipeline_status = PIPELINE_COMPLETED.to_string();

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
    fn safety_dimensions_serialize_as_null_without_a_scan() {
        let report = SafetyReport::default();
        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(json["dimensions"], serde_json::Value::Null);
        assert_eq!(json["auto_fail"], serde_json::Value::Null);
    }

    #[test]
    fn auto_fail_codes_round_trip_through_display_and_parse() {
        for code in AutoFailCode::ALL {
            let rendered = code.to_string();
            assert!(rendered.starts_with(code.code()), "{rendered}");
            assert!(rendered.contains(code.description()), "{rendered}");
            assert_eq!(AutoFailCode::parse(&rendered), Some(code));
            assert_eq!(AutoFailCode::parse(code.code()), Some(code));
            assert_eq!(AutoFailCode::parse(&code.code().to_lowercase()), Some(code));
        }
        assert_eq!(
            AutoFailCode::parse(" AF2: anything "),
            Some(AutoFailCode::HardcodedCredentials)
        );
        assert_eq!(AutoFailCode::parse("AF6: nope"), None);
        assert_eq!(AutoFailCode::parse(""), None);
        assert_eq!(AutoFailCode::parse("harmful"), None);
    }

    #[test]
    fn auto_fail_codes_serialize_as_the_display_string() {
        let code = AutoFailCode::HardcodedCredentials;
        let json = serde_json::to_value(code).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!("AF2: hardcoded credentials, API keys, or secrets")
        );

        let back: AutoFailCode = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, code);

        let bare: AutoFailCode = serde_json::from_str("\"af3\"").expect("bare code");
        assert_eq!(bare, AutoFailCode::UndeclaredExfiltration);

        let error = serde_json::from_str::<AutoFailCode>("\"AF9: unknown\"").expect_err("unknown");
        assert!(error.to_string().contains("auto-fail"), "{error}");
    }

    #[test]
    fn safety_report_serializes_the_auto_fail_string() {
        let report = SafetyReport {
            skill_name: "demo".to_string(),
            score_pct: 0.0,
            gate_threshold: 0.8,
            blocked: true,
            auto_fail: Some(AutoFailCode::BypassSafety),
            dimensions: None,
            findings: Vec::new(),
        };
        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(
            json["auto_fail"],
            serde_json::json!("AF1: instructions to bypass or ignore safety measures")
        );
        assert_eq!(json["dimensions"], serde_json::Value::Null);
        assert_eq!(report.score_fraction(), 0.0);
        assert!(!report.passes_gate());

        let back: SafetyReport = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, report);
    }

    #[test]
    fn safety_dimension_ids_are_the_python_ids() {
        let ids: Vec<&str> = SAFETY_DIMENSIONS.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(
            ids,
            vec![
                "d1_harmful_instruction",
                "d2_scope_containment",
                "d3_data_handling",
                "d4_injection_resilience",
                "d5_guardrails",
            ]
        );
        let maxima: f64 = SAFETY_DIMENSIONS.iter().map(|(_, max, _)| *max).sum();
        assert!((maxima - 100.0).abs() < 1e-9);
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
    fn design_accessors_name_their_unit() {
        let mut report = DesignReport {
            design_score: 74.0,
            design_score_max: 100.0,
            design_score_pct: 0.74,
            ..DesignReport::default()
        };
        assert!((report.score_points() - 74.0).abs() < 1e-9);
        assert!((report.score_fraction() - 0.74).abs() < 1e-9);

        report.dimensions.insert(
            "d1_knowledge_ratio".to_string(),
            ScoreDimension {
                score: 10.0,
                max: 10.0,
            },
        );
        assert_eq!(report.knowledge_delta_label(), "Excellent");

        for (points, expected) in [
            (9.0, "Excellent"),
            (6.0, "Good"),
            (3.0, "Mediocre"),
            (0.0, "Poor"),
        ] {
            report.dimensions.insert(
                "d1_knowledge_ratio".to_string(),
                ScoreDimension {
                    score: points,
                    max: 10.0,
                },
            );
            assert_eq!(report.knowledge_delta_label(), expected, "{points}");
        }
        assert_eq!(DesignReport::default().knowledge_delta_label(), "Poor");
    }

    #[test]
    fn final_score_accessors_name_their_unit() {
        let completed = FinalScore {
            pipeline_status: PIPELINE_COMPLETED.to_string(),
            final_score_pct: 0.81,
            ..FinalScore::default()
        };
        assert!((completed.score_fraction() - 0.81).abs() < 1e-9);
        assert!(completed.is_completed());

        let blocked = FinalScore {
            pipeline_status: PIPELINE_BLOCKED_AT_E3.to_string(),
            ..FinalScore::default()
        };
        assert!(!blocked.is_completed());
    }

    #[test]
    fn evaluator_score_weighted_computes_the_python_contribution() {
        let score = EvaluatorScore::weighted(0.74, 0.40);
        assert!((score.score_pct - 0.74).abs() < 1e-9);
        assert!((score.weight - 0.40).abs() < 1e-9);
        assert!((score.contribution - 0.296).abs() < 1e-9);

        let rounded = EvaluatorScore::weighted(2.0 / 3.0, 0.6);
        assert!((rounded.score_pct - 0.6667).abs() < 1e-9);
        assert!((rounded.contribution - 0.4).abs() < 1e-9);
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
        assert_eq!(
            status_from(&[outcome(true), outcome(true)]),
            CaseStatus::Passed
        );
        assert_eq!(status_from(&[outcome(false)]), CaseStatus::Failed);
        assert_eq!(
            status_from(&[outcome(true), outcome(false)]),
            CaseStatus::Partial
        );
        assert_eq!(status_from(&[]), CaseStatus::Passed);
    }

    fn case(id: &str, status: CaseStatus) -> CaseResult {
        CaseResult {
            id: id.to_string(),
            status,
            ..CaseResult::default()
        }
    }

    #[test]
    fn summarize_counts_only_whole_passes() {
        let results = [
            case("a", CaseStatus::Passed),
            case("b", CaseStatus::Partial),
            case("c", CaseStatus::Failed),
        ];
        let summary = summarize(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.partial, 1);
        assert_eq!(summary.failed, 1);
        assert!((summary.pass_rate - 1.0 / 3.0).abs() < 1e-9);

        let worked_example = [
            case("a", CaseStatus::Passed),
            case("b", CaseStatus::Passed),
            case("c", CaseStatus::Passed),
            case("d", CaseStatus::Partial),
        ];
        let summary = summarize(&worked_example);
        assert_eq!(summary.total, 4);
        assert_eq!(summary.passed, 3);
        assert_eq!(summary.partial, 1);
        assert_eq!(summary.failed, 0);
        assert!((summary.pass_rate - 0.75).abs() < 1e-9);

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
