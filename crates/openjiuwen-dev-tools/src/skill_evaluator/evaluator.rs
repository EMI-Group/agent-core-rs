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
//!
//! # Deterministic and judged runs
//!
//! `evaluate` and `evaluate_at` stay purely deterministic: they never call a
//! model and always leave [`EvaluationReport::judge`] as `None`. Calling
//! [`SkillEvaluator::with_judge`] attaches a `SkillJudge` and unlocks
//! `evaluate_with_judge`/`evaluate_with_judge_at`, which score the design and
//! safety stages with the rubric-driven judge and record the provenance of each
//! judgement in the report. A model failure degrades to the deterministic score
//! rather than failing the run.
//!
//! Everything is async-first over the existing tokio runtime and offline by
//! default: the only side channels are the injected [`CaseRunner`] and the
//! injected judge model.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};

use crate::skill_creator::SkillDefinition;

use super::assertion::Assertion;
use super::case::{CaseStatus, EvalCase};
use super::judge_scoring::SkillJudge;
use super::report::{
    status_from, summarize, CaseResult, EvaluationReport, FunctionalReport, JudgeRecord,
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
    ///
    /// Consumed by both paths: [`evaluate_at`](SkillEvaluator::evaluate_at)
    /// passes it to the deterministic scan, and
    /// [`evaluate_with_judge_at`](SkillEvaluator::evaluate_with_judge_at) hands it
    /// to the judge, so a judged run reports this fraction in
    /// `SafetyReport::gate_threshold` and gates on it.
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
    /// The optional rubric-driven judge for the design and safety stages.
    judge: Option<Arc<SkillJudge>>,
}

impl SkillEvaluator {
    /// Creates an evaluator over `runner` with the default configuration and no
    /// judge.
    #[must_use]
    pub fn new(runner: Arc<dyn CaseRunner>) -> Self {
        Self {
            runner,
            config: EvaluatorConfig::default(),
            judge: None,
        }
    }

    /// Replaces the configuration.
    #[must_use]
    pub fn with_config(mut self, config: EvaluatorConfig) -> Self {
        self.config = config;
        self
    }

    /// Attaches a model-backed judge over `model`.
    ///
    /// Only [`evaluate_with_judge`](Self::evaluate_with_judge) and
    /// [`evaluate_with_judge_at`](Self::evaluate_with_judge_at) consult the judge;
    /// [`evaluate`](Self::evaluate) and [`evaluate_at`](Self::evaluate_at) stay
    /// deterministic.
    #[must_use]
    pub fn with_judge(mut self, model: Arc<dyn Model>) -> Self {
        self.judge = Some(Arc::new(SkillJudge::new(model)));
        self
    }

    /// Attaches a preconfigured judge.
    #[must_use]
    pub fn with_skill_judge(mut self, judge: SkillJudge) -> Self {
        self.judge = Some(Arc::new(judge));
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

    /// The attached judge, when one was configured.
    #[must_use]
    pub fn judge(&self) -> Option<&Arc<SkillJudge>> {
        self.judge.as_ref()
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
        self.evaluate_at(skill, cases, now_ms()).await
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
    /// The pass rate is `passed / total` as a fraction of `1.0` (partial cases
    /// are counted but earn no credit); an empty case list yields `0.0`.
    /// Recommendations list the names of failed and partial cases in input
    /// order.
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
        let functional = self.functional_report(skill, cases, timestamp_ms).await?;

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

    /// Evaluates `skill` over `cases` with the attached judge, timestamping the
    /// report with the current UNIX epoch milliseconds.
    ///
    /// # Errors
    ///
    /// Propagates every error of
    /// [`evaluate_with_judge_at`](Self::evaluate_with_judge_at).
    pub async fn evaluate_with_judge(
        &self,
        skill: &SkillDefinition,
        cases: &[EvalCase],
    ) -> Result<EvaluationReport> {
        self.evaluate_with_judge_at(skill, cases, now_ms()).await
    }

    /// Evaluates `skill` over `cases` with the attached judge and an explicit
    /// report timestamp.
    ///
    /// The functional half is identical to [`evaluate_at`](Self::evaluate_at),
    /// including validation order and errors. The design and safety stages are
    /// scored by the judge: a model verdict replaces the deterministic score,
    /// while a model failure degrades to it. The [`JudgeRecord`] records one
    /// [`VerdictSource`](crate::judge::VerdictSource) per judged stage, so a
    /// degraded judgement stays visible in the persisted report. A safety rule
    /// hit always closes the gate, whatever the model answers.
    ///
    /// The safety stage gates on
    /// [`EvaluatorConfig::safety_gate_threshold`], so the configured fraction is
    /// the one reported in the safety report's `gate_threshold` and the one the
    /// judged gate applies — the same fraction the deterministic path uses.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when no judge is configured, the skill, a
    /// case, a safety rule, or the configured safety gate threshold is invalid,
    /// and propagates the runner's and the judge's own errors.
    pub async fn evaluate_with_judge_at(
        &self,
        skill: &SkillDefinition,
        cases: &[EvalCase],
        timestamp_ms: i64,
    ) -> Result<EvaluationReport> {
        let judge = self.judge.as_ref().ok_or_else(|| {
            Error::validation(
                "no skill judge configured; call SkillEvaluator::with_judge before \
                 evaluate_with_judge",
            )
        })?;

        let functional = self.functional_report(skill, cases, timestamp_ms).await?;
        let design = judge.design(skill).await?;
        let safety = judge
            .safety_with_threshold(
                skill,
                &self.config.safety_rules,
                self.config.safety_gate_threshold,
            )
            .await?;
        let final_score = aggregate(
            &design.report,
            &functional,
            &safety.report,
            self.config.design_weight,
            self.config.functional_weight,
        );

        Ok(EvaluationReport {
            design: design.report,
            functional,
            safety: safety.report,
            final_score,
            judge: Some(JudgeRecord {
                design_source: design.verdict.source,
                safety_source: safety.verdict.source,
            }),
        })
    }

    /// Validates the skill and its cases, runs them, and builds the functional
    /// report.
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
    /// The pass rate is `passed / total` as a fraction of `1.0` (partial cases
    /// are counted but earn no credit); an empty case list yields `0.0`.
    /// Recommendations list the names of failed and partial cases in input
    /// order.
    async fn functional_report(
        &self,
        skill: &SkillDefinition,
        cases: &[EvalCase],
        timestamp_ms: i64,
    ) -> Result<FunctionalReport> {
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
        Ok(FunctionalReport {
            skill_name: skill.name.clone(),
            skill_path: format!("skills/{}", skill.name),
            timestamp: timestamp_ms,
            summary,
            results,
            recommendations,
        })
    }
}

/// The current UNIX epoch milliseconds, or `0` when the clock is unusable.
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    use crate::judge::VerdictSource;
    use crate::skill_evaluator::report::{GATE_E3, PIPELINE_BLOCKED_AT_E3, PIPELINE_COMPLETED};

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
        assert!(report.design.design_score_pct >= 0.6);
        assert!(report.design.ready_for_functional_testing);
        assert_eq!(report.final_score.pipeline_status, PIPELINE_COMPLETED);
        assert_eq!(report.final_score.gates_passed, vec![GATE_E3]);
        assert!(report.safety.passes_gate());
        assert!((report.safety.score_fraction() - 1.0).abs() < 1e-9);
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
        assert!((report.functional.summary.pass_rate - 0.0).abs() < 1e-9);
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
        assert_eq!(report.final_score.pipeline_status, PIPELINE_BLOCKED_AT_E3);
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
        assert_eq!(json["final_score"]["pipeline_status"], PIPELINE_COMPLETED);
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

    fn fenced_verdict(score: f64, passed: bool) -> String {
        format!("```json\n{{\"score\": {score}, \"passed\": {passed}, \"reasoning\": \"ok\"}}\n```")
    }

    fn judged_evaluator(turns: impl IntoIterator<Item = ScriptedTurn>) -> SkillEvaluator {
        evaluator(&[("case-1", "alpha answer")]).with_judge(Arc::new(ScriptedModel::new(turns)))
    }

    #[tokio::test]
    async fn a_judged_run_records_model_provenance() {
        let evaluator = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(0.95, true)),
        ]);

        let report = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 7)
            .await
            .expect("judge");

        let record = report.judge.as_ref().expect("judge record");
        assert_eq!(record.design_source, VerdictSource::Model);
        assert_eq!(record.safety_source, VerdictSource::Model);
        assert!((report.design.design_score - 80.0).abs() < 1e-9);
        assert!((report.design.design_score_pct - 0.8).abs() < 1e-9);
        assert!((report.safety.score_pct - 0.95).abs() < 1e-9);
        assert_eq!(report.final_score.pipeline_status, PIPELINE_COMPLETED);
        assert_eq!(report.functional.timestamp, 7);
        assert_eq!(report.functional.skill_path, "skills/demo");
        assert!(!report.safety.blocked);
    }

    #[tokio::test]
    async fn a_malformed_design_answer_mixes_the_provenance() {
        let evaluator = judged_evaluator([
            ScriptedTurn::text("no structure"),
            ScriptedTurn::text(fenced_verdict(0.9, true)),
        ]);

        let report = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("judge");

        let record = report.judge.as_ref().expect("judge record");
        assert_eq!(record.design_source, VerdictSource::Deterministic);
        assert_eq!(record.safety_source, VerdictSource::Model);
        assert_eq!(report.design, score_design(&skill()));
    }

    #[tokio::test]
    async fn failing_turns_leave_both_stages_deterministic() {
        let evaluator = judged_evaluator([ScriptedTurn::fail("boom"), ScriptedTurn::fail("boom")]);

        let report = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("judge");

        let record = report.judge.as_ref().expect("judge record");
        assert_eq!(record.design_source, VerdictSource::Deterministic);
        assert_eq!(record.safety_source, VerdictSource::Deterministic);
        assert_eq!(report.design, score_design(&skill()));
        assert!((report.safety.score_pct - 1.0).abs() < 1e-9);
        assert_eq!(report.final_score.pipeline_status, PIPELINE_COMPLETED);
    }

    #[tokio::test]
    async fn a_critical_rule_blocks_a_passing_safety_verdict() {
        let config = EvaluatorConfig {
            safety_rules: vec![SafetyRule::new(
                "demo",
                "data_leakage",
                "critical",
                "leaks the name",
            )],
            ..EvaluatorConfig::default()
        };
        let evaluator = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(1.0, true)),
        ])
        .with_config(config);

        let report = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("judge");

        assert!(report.safety.blocked);
        assert_eq!(report.final_score.pipeline_status, PIPELINE_BLOCKED_AT_E3);
        assert_eq!(report.safety.findings.len(), 1);
        let record = report.judge.as_ref().expect("judge record");
        assert_eq!(record.safety_source, VerdictSource::Model);
    }

    #[tokio::test]
    async fn a_judged_run_without_an_answer_is_a_validation_error() {
        let evaluator = judged_evaluator([ScriptedTurn::text(fenced_verdict(0.8, true))]);
        let cases = [
            passing_case("case-1", "alpha"),
            passing_case("case-2", "beta"),
        ];

        let error = evaluator
            .evaluate_with_judge_at(&skill(), &cases, 1)
            .await
            .expect_err("no scripted response");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("case-2"));
    }

    #[tokio::test]
    async fn a_deterministic_run_rejects_the_judged_entry_point_without_a_judge() {
        let evaluator = evaluator(&[("case-1", "alpha answer")]);
        assert!(evaluator.judge().is_none());

        let error = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect_err("no judge");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("no skill judge configured"));
    }

    #[tokio::test]
    async fn evaluate_ignores_an_attached_judge() {
        let model = ScriptedModel::new([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(0.9, true)),
        ]);
        let evaluator =
            evaluator(&[("case-1", "alpha answer")]).with_judge(Arc::new(model.clone()));
        assert!(evaluator.judge().is_some());

        let report = evaluator
            .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("evaluate");

        assert!(report.judge.is_none());
        assert_eq!(model.call_count(), 0);
        assert_eq!(report.design, score_design(&skill()));
    }

    #[tokio::test]
    async fn evaluate_with_judge_uses_the_wall_clock_timestamp() {
        let evaluator = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(0.9, true)),
        ]);

        let report = evaluator
            .evaluate_with_judge(&skill(), &[passing_case("case-1", "alpha")])
            .await
            .expect("judge");
        assert!(report.functional.timestamp > 0);
        assert!(report.judge.is_some());
    }

    #[tokio::test]
    async fn the_judged_report_json_adds_the_judge_record() {
        let evaluator = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(0.9, true)),
        ]);
        let report = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("judge");

        let json = report.to_json().expect("to_json");
        let object = json.as_object().expect("object");
        for key in ["design", "functional", "safety", "final_score", "judge"] {
            assert!(object.contains_key(key), "missing {key}");
        }
        assert_eq!(json["judge"]["design_source"], "model");
        assert_eq!(json["judge"]["safety_source"], "model");
        assert!((json["design"]["design_score"].as_f64().expect("number") - 80.0).abs() < 1e-9);

        let back: EvaluationReport =
            serde_json::from_str(&report.to_json_string().expect("json")).expect("deserialize");
        assert_eq!(back, report);
    }

    /// A rule that matches the demo skill body with a non-critical severity.
    fn warning_rule() -> SafetyRule {
        SafetyRule::new(
            "demo",
            "d3_data_handling",
            "warning",
            "mentions the skill name",
        )
    }

    #[tokio::test]
    async fn a_judged_run_honours_the_configured_safety_gate() {
        let config = EvaluatorConfig {
            safety_gate_threshold: 0.90,
            ..EvaluatorConfig::default()
        };
        let raised = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(0.875, true)),
        ])
        .with_config(config)
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
        .await
        .expect("judge");

        assert!((raised.safety.score_pct - 0.875).abs() < 1e-9);
        assert!(raised.safety.blocked);
        assert!((raised.safety.gate_threshold - 0.9).abs() < 1e-9);
        assert_eq!(raised.final_score.pipeline_status, PIPELINE_BLOCKED_AT_E3);

        let defaulted = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(0.875, true)),
        ])
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
        .await
        .expect("judge");

        assert!(!defaulted.safety.blocked);
        assert!((defaulted.safety.gate_threshold - 0.8).abs() < 1e-9);
        assert_eq!(defaulted.final_score.pipeline_status, PIPELINE_COMPLETED);
    }

    #[tokio::test]
    async fn the_deterministic_and_judged_paths_share_the_configured_gate() {
        for (threshold, blocked, status) in [
            (0.70, false, PIPELINE_COMPLETED),
            (0.90, true, PIPELINE_BLOCKED_AT_E3),
        ] {
            let config = EvaluatorConfig {
                safety_gate_threshold: threshold,
                safety_rules: vec![warning_rule()],
                ..EvaluatorConfig::default()
            };

            let deterministic = evaluator(&[("case-1", "alpha answer")])
                .with_config(config.clone())
                .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], 1)
                .await
                .expect("evaluate");
            assert!((deterministic.safety.score_pct - 0.75).abs() < 1e-9);
            assert_eq!(deterministic.safety.blocked, blocked, "{threshold}");
            assert!((deterministic.safety.gate_threshold - threshold).abs() < 1e-9);
            assert_eq!(deterministic.final_score.pipeline_status, status);

            let judged = judged_evaluator([
                ScriptedTurn::text(fenced_verdict(0.8, true)),
                ScriptedTurn::text(fenced_verdict(1.0, true)),
            ])
            .with_config(config)
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect("judge");
            assert!((judged.safety.score_pct - 0.75).abs() < 1e-9);
            assert_eq!(judged.safety.blocked, blocked, "{threshold}");
            assert!((judged.safety.gate_threshold - threshold).abs() < 1e-9);
            assert_eq!(judged.final_score.pipeline_status, status);
        }
    }

    #[tokio::test]
    async fn an_invalid_configured_gate_threshold_is_a_validation_error() {
        let config = EvaluatorConfig {
            safety_gate_threshold: 1.5,
            ..EvaluatorConfig::default()
        };
        let evaluator = judged_evaluator([
            ScriptedTurn::text(fenced_verdict(0.8, true)),
            ScriptedTurn::text(fenced_verdict(1.0, true)),
        ])
        .with_config(config);

        let error = evaluator
            .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], 1)
            .await
            .expect_err("invalid threshold");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("safety gate threshold"),
            "{error}"
        );
    }
}
