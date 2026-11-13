//! The end-to-end evaluation pipeline.
//!
//! [`SkillEvaluator`] wires the four stages together: it validates the skill and
//! every [`EvalCase`], runs the cases through a [`CaseRunner`], checks their
//! expectations, and combines the structural design review, the functional
//! report, and the safety scan into one [`EvaluationReport`].
//!
//! Python drove the same pipeline from a script with a shell harness; here the
//! runner is a trait, so a scripted run is fully offline and a model-backed run
//! uses the same assertions and report types.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::skill_creator::SkillDefinition;

use super::assertion::Assertion;
use super::case::{CaseStatus, EvalCase};
use super::report::{
    status_from, summarize, CaseResult, EvaluationReport, FunctionalReport,
};
use super::runner::{CaseRunner, ScriptedCaseRunner};
use super::scoring::{aggregate, scan_safety, score_design, SafetyRule};

/// Tunables and rule set of one evaluation run.
///
/// The defaults match the Python formula: design weighted at `0.40`, functional
/// at `0.60`, and a safety gate at `0.80`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluatorConfig {
    /// Weight applied to the design score.
    pub design_weight: f64,
    /// Weight applied to the functional pass rate.
    pub functional_weight: f64,
    /// Fraction of the safety score required to keep the gate open.
    pub safety_gate_threshold: f64,
    /// Regex rules scanned against the rendered skill document.
    pub safety_rules: Vec<SafetyRule>,
    /// Partial-credit threshold, exposed for parity and future use.
    ///
    /// Per-case status is derived from all/none of the assertions passing, not
    /// from this threshold.
    pub partial_threshold: f64,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            design_weight: 0.40,
            functional_weight: 0.60,
            safety_gate_threshold: 0.80,
            safety_rules: Vec::new(),
            partial_threshold: 0.5,
        }
    }
}

/// Runs the full evaluation pipeline over a skill and its cases.
pub struct SkillEvaluator {
    /// Executes each case and supplies the answer text.
    runner: Arc<dyn CaseRunner>,
    /// Weights, safety rules, and thresholds.
    config: EvaluatorConfig,
}

impl SkillEvaluator {
    /// Creates an evaluator over `runner` with the default configuration.
    #[must_use]
    pub fn new(runner: Arc<dyn CaseRunner>) -> Self {
        Self {
            runner,
            config: EvaluatorConfig::default(),
        }
    }

    /// Replaces the configuration.
    #[must_use]
    pub fn with_config(mut self, config: EvaluatorConfig) -> Self {
        self.config = config;
        self
    }

    /// Creates an evaluator with an empty [`ScriptedCaseRunner`].
    ///
    /// Every case must have a registered response, so this is the strict
    /// offline default: an unregistered case is a validation error.
    #[must_use]
    pub fn scripted() -> Self {
        Self::new(Arc::new(ScriptedCaseRunner::new()))
    }

    /// The runner consulted for each case.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn CaseRunner> {
        &self.runner
    }

    /// The active configuration.
    #[must_use]
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }

    /// Evaluates `skill` over `cases`, timestamping the report with the current
    /// UNIX epoch milliseconds.
    ///
    /// # Errors
    ///
    /// Propagates every error of [`evaluate_at`](Self::evaluate_at).
    pub async fn evaluate(
        &self,
        skill: &SkillDefinition,
        cases: &[EvalCase],
    ) -> Result<EvaluationReport> {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| {
                i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
            });
        self.evaluate_at(skill, cases, timestamp_ms).await
    }

    /// Evaluates `skill` over `cases` with an explicit report timestamp.
    ///
    /// The skill is validated first, then every case, so the first invalid input
    /// fails before any case runs. Cases are executed sequentially in input
    /// order; a runner error aborts immediately. Each expectation is parsed into
    /// an [`Assertion`] and checked against the answer:
    ///
    /// - all assertions pass: [`CaseStatus::Passed`];
    /// - none pass: [`CaseStatus::Failed`];
    /// - otherwise: [`CaseStatus::Partial`].
    ///
    /// The pass rate is `(passed + 0.5 * partial) / total`, crediting partial
    /// cases with half a pass (a deviation from Python's strict counts); an
    /// empty case list yields `0.0`. Recommendations list the names of failed
    /// and partial cases in input order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid skill or case, for an invalid expectation, or for a runner
    /// without an answer; a model-backed runner propagates its own errors.
    pub async fn evaluate_at(
        &self,
        skill: &SkillDefinition,
        cases: &[EvalCase],
        timestamp_ms: i64,
    ) -> Result<EvaluationReport> {
        skill.validate()?;
        for case in cases {
            case.validate()?;
        }

        let mut results = Vec::with_capacity(cases.len());
        for case in cases {
            let output = self.runner.run(case, skill).await?;
            let mut assertions = Vec::with_capacity(case.expectations.len());
            for expectation in &case.expectations {
                let assertion = Assertion::parse(expectation)?;
                assertions.push(assertion.check(&output)?);
            }
            results.push(CaseResult {
                id: case.id.clone(),
                name: case.name.clone(),
                status: status_from(&assertions),
                assertions,
            });
        }

        let summary = summarize(&results);
        let recommendations = results
            .iter()
            .filter(|result| matches!(result.status, CaseStatus::Failed | CaseStatus::Partial))
            .map(|result| result.name.clone())
            .collect();
        let functional = FunctionalReport {
            skill_name: skill.name.clone(),
            skill_path: format!("skills/{}", skill.name),
            timestamp: timestamp_ms,
            summary,
            results,
            recommendations,
        };

        let design = score_design(skill);
        let safety = scan_safety(
            skill,
            &self.config.safety_rules,
            self.config.safety_gate_threshold,
        )?;
        let final_score = aggregate(
            &design,
            &functional,
            &safety,
            self.config.design_weight,
            self.config.functional_weight,
        );

        Ok(EvaluationReport {
            design,
            functional,
            safety,
            final_score,
            judge: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use openjiuwen_common::error::ErrorKind;

    fn skill() -> SkillDefinition {
        let body = concat!(
            "# Demo Skill\n\n",
            "## Overview\n\n",
            "This skill summarizes documents end to end. It explains the input ",
            "format, the processing steps, and the expected result structure so a ",
            "reader can reproduce the work without extra guidance.\n\n",
            "## Steps\n\n",
            "1. Read the input document.\n",
            "2. Extract the key facts.\n",
            "3. Write the summary.\n\n",
            "## Output\n\n",
            "The result format is plain text.\n\n",
            "- summary"
        );
        SkillDefinition::new("demo", "Summarize documents", body)
    }

    fn passing_case(id: &str, word: &str) -> EvalCase {
        EvalCase::new(id, format!("Case {id}"), format!("Respond with {word}"))
            .with_expectation(format!("contains:{word}"))
    }

    fn evaluator(responses: &[(&str, &str)]) -> SkillEvaluator {
        let mut runner = ScriptedCaseRunner::new();
        for (id, output) in responses {
            runner = runner.respond(*id, *output);
        }
        SkillEvaluator::new(Arc::new(runner))
    }

    #[test]
    fn default_config_matches_the_python_weights() {
        let config = EvaluatorConfig::default();
        assert!((config.design_weight - 0.4).abs() < 1e-9);
        assert!((config.functional_weight - 0.6).abs() < 1e-9);
        assert!((config.safety_gate_threshold - 0.8).abs() < 1e-9);
        assert_eq!(config.safety_rules, Vec::<SafetyRule>::new());
        assert!((config.partial_threshold - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn scripted_evaluator_has_an_empty_runner_and_default_config() {
        let evaluator = SkillEvaluator::scripted();
        assert_eq!(evaluator.config(), &EvaluatorConfig::default());
        let error = evaluator
            .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect_err("no scripted response");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("no scripted response"));
    }

    #[tokio::test]
    async fn evaluate_at_reports_every_status() {
        let cases = [
            passing_case("case-1", "alpha"),
            passing_case("case-2", "beta"),
            passing_case("case-3", "gamma"),
        ];
        let evaluator = evaluator(&[
            ("case-1", "alpha answer"),
            ("case-2", "beta answer"),
            ("case-3", "delta answer"),
        ]);

        let report = evaluator
            .evaluate_at(&skill(), &cases, 12_345)
            .await
            .expect("evaluate");

        assert_eq!(report.functional.summary.total, 3);
        assert_eq!(report.functional.summary.passed, 2);
        assert_eq!(report.functional.summary.partial, 0);
        assert_eq!(report.functional.summary.failed, 1);
        assert!((report.functional.summary.pass_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(
            report
                .functional
                .results
                .iter()
                .map(|result| result.status)
                .collect::<Vec<_>>(),
            vec![CaseStatus::Passed, CaseStatus::Passed, CaseStatus::Failed]
        );
        assert_eq!(report.functional.skill_name, "demo");
        assert_eq!(report.functional.skill_path, "skills/demo");
        assert_eq!(report.functional.timestamp, 12_345);
        assert_eq!(report.functional.recommendations, vec!["Case case-3"]);
        assert!(!report.safety.blocked);
        assert!(report.design.design_score_pct >= 50.0);
        assert!(report.design.ready_for_functional_testing);
        assert_eq!(report.final_score.pipeline_status, "COMPLETED");
        assert_eq!(
            report.final_score.gates_passed,
            vec!["e1_llm_judge", "e2_functional", "e3_safety"]
        );
    }

    #[tokio::test]
    async fn mixed_expectations_yield_a_partial_case() {
        let case = EvalCase::new("case-1", "Mixed case", "Do both")
            .with_expectations(["contains:alpha", "contains:beta"]);
        let evaluator = evaluator(&[("case-1", "alpha only")]);

        let report = evaluator
            .evaluate_at(&skill(), &[case], 1)
            .await
            .expect("evaluate");

        assert_eq!(report.functional.summary.total, 1);
        assert_eq!(report.functional.summary.partial, 1);
        assert_eq!(report.functional.summary.passed, 0);
        assert_eq!(report.functional.summary.failed, 0);
        assert!((report.functional.summary.pass_rate - 0.5).abs() < 1e-9);
        assert_eq!(report.functional.results[0].status, CaseStatus::Partial);
        assert_eq!(report.functional.results[0].assertions.len(), 2);
        assert_eq!(report.functional.recommendations, vec!["Mixed case"]);
    }

    #[tokio::test]
    async fn invalid_assertion_is_a_validation_error() {
        let case = EvalCase::new("case-1", "Bad regex", "Prompt").with_expectation("regex:(");
        let evaluator = evaluator(&[("case-1", "answer")]);

        let error = evaluator
            .evaluate_at(&skill(), &[case], 1)
            .await
            .expect_err("invalid regex");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("invalid assertion regex"));
    }

    #[tokio::test]
    async fn invalid_case_fails_before_the_runner_runs() {
        let case = EvalCase::new("case-1", "No expectations", "Prompt");
        let error = SkillEvaluator::scripted()
            .evaluate_at(&skill(), &[case], 1)
            .await
            .expect_err("missing expectation");

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("expectation"));
    }

    #[tokio::test]
    async fn invalid_skill_fails_before_the_runner_runs() {
        let skill = SkillDefinition::new("demo", "", "body");
        let error = SkillEvaluator::scripted()
            .evaluate_at(&skill, &[passing_case("case-1", "alpha")], 1)
            .await
            .expect_err("blank description");

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("description"));
    }

    #[tokio::test]
    async fn runner_error_propagates() {
        let error = SkillEvaluator::scripted()
            .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect_err("missing response");

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("case-1"));
    }

    #[tokio::test]
    async fn critical_safety_rule_blocks_the_pipeline() {
        let config = EvaluatorConfig {
            safety_rules: vec![SafetyRule::new(
                "demo",
                "data_leakage",
                "critical",
                "leaks the name",
            )],
            ..EvaluatorConfig::default()
        };
        let evaluator = evaluator(&[("case-1", "alpha answer")]).with_config(config);

        let report = evaluator
            .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("evaluate");

        assert!(report.safety.blocked);
        assert_eq!(report.final_score.pipeline_status, "BLOCKED_AT_SAFETY");
        assert!((report.final_score.final_score_pct - 0.0).abs() < 1e-9);
        assert_eq!(report.final_score.grade, "F");
    }

    #[tokio::test]
    async fn evaluate_uses_the_wall_clock_timestamp() {
        let evaluator = evaluator(&[("case-1", "alpha answer")]);
        let report = evaluator
            .evaluate(&skill(), &[passing_case("case-1", "alpha")])
            .await
            .expect("evaluate");
        assert!(report.functional.timestamp > 0);
    }

    async fn end_to_end_report() -> EvaluationReport {
        let cases = [
            passing_case("case-1", "alpha"),
            passing_case("case-2", "beta"),
            passing_case("case-3", "gamma"),
        ];
        evaluator(&[
            ("case-1", "alpha answer"),
            ("case-2", "beta answer"),
            ("case-3", "delta answer"),
        ])
        .evaluate_at(&skill(), &cases, 12_345)
        .await
        .expect("evaluate")
    }

    #[tokio::test]
    async fn report_json_exposes_the_python_shape() {
        let report = end_to_end_report().await;
        let json = report.to_json().expect("to_json");
        let object = json.as_object().expect("object");

        for key in ["design", "functional", "safety", "final_score"] {
            assert!(object.contains_key(key), "missing {key}");
        }

        let design = &json["design"];
        assert!(design["design_score_pct"].is_number());
        assert_eq!(design["ready_for_functional_testing"], true);
        let dimensions = design["dimensions"].as_object().expect("dimensions");
        assert_eq!(dimensions.len(), 7);
        for key in [
            "d1_knowledge_ratio",
            "d2_expert_knowledge_craft",
            "d3_spec_compliance",
            "d4_progressive_disclosure",
            "d5_pattern_freedom_fit",
            "d6_practical_usability",
            "d7_output_specification",
        ] {
            assert!(dimensions.contains_key(key), "missing {key}");
        }

        assert!(json["functional"]["summary"]["pass_rate"].is_number());
        assert!(json["final_score"]["final_score_pct"].is_number());
        assert_eq!(json["final_score"]["pipeline_status"], "COMPLETED");
        assert!(json["final_score"]["gates_passed"].is_array());
        assert!(json["final_score"]["evaluators"]["e1_llm_judge"].is_object());
        assert!(json["final_score"]["evaluators"]["e2_functional"].is_object());
    }

    #[tokio::test]
    async fn report_round_trips_through_json() {
        let report = end_to_end_report().await;
        let json = report.to_json_string().expect("to_json_string");
        let back: EvaluationReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);
    }
}
