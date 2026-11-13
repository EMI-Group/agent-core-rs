//! Rubric-driven LLM judging for the design and safety stages.
//!
//! This module is the model-backed path of the skill evaluator: `SkillJudge`
//! layers a [`ModelJudge`] over the deterministic
//! scorers of [`scoring`](crate::skill_evaluator::scoring), so a judge-driven run
//! degrades to the offline result instead of failing.
//! `design_rubric` and `safety_rubric` carry the same dimensions, weights, and
//! maxima as `score_design` and `scan_safety`, which is what makes a model
//! verdict interchangeable with a deterministic one: every judgement is either a
//! model verdict on those rubrics or the deterministic fallback, never anything
//! else.
//!
//! # Scale convention
//!
//! Both rubrics score every dimension on `0..=100` and weight each dimension by
//! its point maximum; the weights therefore sum to the `0..=100` point total, so
//! the normalized [`Verdict::score`] *is* the report fraction: `0.74` is 74
//! points of 100, and a raw dimension score of `60` means "60 of that
//! dimension's 100 points", i.e. `0.6 · max` awarded points. Only the report's
//! `*_score`/`ScoreDimension::{score, max}` members are points; every `*_pct`
//! member is a fraction of `1.0`.
//!
//! # Hard rule: a rule hit never passes the gate
//!
//! The safety judgement can only ever *lower* the deterministic verdict.
//! `SkillJudge::safety` keeps every `SafetyFinding` produced by `scan_safety`,
//! takes the lower of the model and scan scores, and closes the gate whenever the
//! scan was blocked — a `critical` rule match blocks even when the model answers
//! with a perfect score. An [`AutoFailCode`] rule blocks unconditionally: the
//! scan report is returned verbatim and the model is never consulted.
//!
//! [`AutoFailCode`]: crate::skill_evaluator::report::AutoFailCode
//!
//! # Offline behavior
//!
//! Nothing here reads the environment, the clock, or the filesystem: every
//! judgement is one non-streaming [`Model`]
//! call whose answer is parsed with
//! [`Verdict::parse`](crate::judge::Verdict::parse). Tests use a scripted model,
//! so the whole surface runs offline and deterministically.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;

use crate::judge::{JudgeRequest, ModelJudge, Rubric, RubricDimension, Verdict, VerdictSource};
use crate::skill_creator::SkillDefinition;

use super::report::{grade_for, DesignReport, SafetyReport, ScoreDimension, SAFETY_DIMENSIONS};
use super::scoring::{
    fraction_from_points, points_from_fraction, round2, round4, scan_safety, score_design,
    SafetyRule, DESIGN_GATE_THRESHOLD,
};

/// The safety gate threshold, as a fraction of `1.0`.
///
/// A safety judgement closes the gate below this fraction regardless of the
/// configured evaluator threshold, so a rule hit can never be talked past.
const SAFETY_GATE_THRESHOLD: f64 = 0.80;

/// The raw maximum every rubric dimension is scored against.
const RUBRIC_SCALE: f64 = 100.0;

/// The seven design dimensions of [`score_design`], with their point maxima and
/// model-facing descriptions.
const DESIGN_DIMENSIONS: [(&str, f64, &str); 7] = [
    (
        "d1_knowledge_ratio",
        10.0,
        "How much actionable, non-boilerplate knowledge the body carries.",
    ),
    (
        "d2_expert_knowledge_craft",
        25.0,
        "Whether the body is organised as distinct sections plus numbered steps.",
    ),
    (
        "d3_spec_compliance",
        15.0,
        "Whether the name, description, and SKILL.md structure follow the spec.",
    ),
    (
        "d4_progressive_disclosure",
        10.0,
        "Whether shipped resources are signposted from the body.",
    ),
    (
        "d5_pattern_freedom_fit",
        15.0,
        "Whether the document structure fits the skill rather than a fixed template.",
    ),
    (
        "d6_practical_usability",
        15.0,
        "Whether the body gives runnable steps, lists, or code examples.",
    ),
    (
        "d7_output_specification",
        10.0,
        "Whether the body states the expected output or result format.",
    ),
];

/// The criteria handed to the design judge.
const DESIGN_INSTRUCTIONS: &str = "Score this SKILL.md document against the seven design \
     dimensions. Award each dimension 0 to 100 points, where 100 means the dimension is fully \
     satisfied. Answer with a JSON object carrying score, passed, reasoning, and \
     dimension_scores.";

/// Builds the design rubric, matching the dimensions and maxima of
/// [`score_design`].
///
/// Each dimension is scored on a `0..=100` scale and weighted by its point
/// maximum, so the normalized [`Verdict::score`] multiplied by `100.0` is the
/// design score in points.
#[must_use]
pub fn design_rubric() -> Rubric {
    let mut rubric = Rubric::new(
        "skill design review",
        "Structural review of an openJiuwen skill document against the seven \
         design dimensions of the skill evaluator.",
    )
    .with_scale(RUBRIC_SCALE);
    for (name, weight, description) in DESIGN_DIMENSIONS {
        rubric = rubric.with_dimension(
            RubricDimension::new(name, description)
                .with_weight(weight)
                .with_guidance(format!("Score 0 to 100; {weight} points are at stake.")),
        );
    }
    rubric
}

/// Builds the safety rubric over the five canonical safety dimensions.
///
/// The weights (`30/25/20/15/10`) sum to 100 and every dimension is scored on a
/// `0..=100` scale, so `verdict.score` is already the normalized safety score —
/// no rescaling is needed (or allowed) when the verdict is consumed.
#[must_use]
pub fn safety_rubric() -> Rubric {
    let mut rubric = Rubric::new(
        "skill safety review",
        "Risk review of an openJiuwen skill document: 100 points per dimension \
         means the risk is absent, 0 means the document is dangerous.",
    )
    .with_scale(RUBRIC_SCALE);
    for (name, weight, description) in SAFETY_DIMENSIONS {
        rubric = rubric.with_dimension(
            RubricDimension::new(name, description)
                .with_weight(weight)
                .with_guidance(format!("Score 0 to 100; {weight} points are at stake.")),
        );
    }
    rubric
}

/// One judged design review: the verdict, the report it produced, and whether
/// the deterministic fallback replaced the model.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignJudgement {
    /// The raw judgement, either from the model or the deterministic fallback.
    pub verdict: Verdict,
    /// The design report assembled from [`verdict`](Self::verdict).
    pub report: DesignReport,
    /// Whether the deterministic rubric supplied the verdict.
    pub used_fallback: bool,
}

/// One judged safety review: the verdict, the gated report, and whether the
/// deterministic scan replaced the model.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetyJudgement {
    /// The raw judgement, either from the model or the deterministic fallback.
    pub verdict: Verdict,
    /// The safety report, always carrying the deterministic findings.
    pub report: SafetyReport,
    /// Whether the deterministic scan supplied the verdict.
    pub used_fallback: bool,
}

/// Applies a rubric-driven model judge to skills, degrading to the
/// deterministic scorers.
///
/// A [`SkillJudge`] owns one [`ModelJudge`]; [`design`](Self::design) and
/// [`safety`](Self::safety) are the two judging stages of the evaluator.
pub struct SkillJudge {
    /// The model-backed judge consulted for every rubric.
    judge: ModelJudge,
}

impl SkillJudge {
    /// Creates a judge over `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            judge: ModelJudge::new(model),
        }
    }

    /// Replaces the built-in judge system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.judge = self.judge.with_system_prompt(prompt);
        self
    }

    /// The model-backed judge consulted for every rubric.
    #[must_use]
    pub fn judge(&self) -> &ModelJudge {
        &self.judge
    }

    /// Judges `skill` against [`design_rubric`].
    ///
    /// The deterministic [`score_design`] report is computed first and used
    /// verbatim when the model call, its parse, or the verdict validation
    /// fails; otherwise [`design_report_from_verdict`] rebuilds the report from
    /// the model's per-dimension scores.
    ///
    /// # Errors
    ///
    /// Propagates the error from [`SkillDefinition::to_markdown`]; model
    /// failures are not errors because they degrade to the fallback.
    pub async fn design(&self, skill: &SkillDefinition) -> Result<DesignJudgement> {
        let report = score_design(skill);
        let subject = skill.to_markdown()?;
        let mut fallback = Verdict::deterministic(
            report.design_score / RUBRIC_SCALE,
            format!(
                "deterministic design rubric awarded {point} of 100 points",
                point = report.design_score
            ),
        );
        fallback.dimension_scores = report
            .dimensions
            .iter()
            .map(|(name, dimension)| {
                let scaled = if dimension.max > 0.0 {
                    round2(dimension.score / dimension.max * RUBRIC_SCALE)
                } else {
                    0.0
                };
                (name.clone(), scaled)
            })
            .collect();

        let verdict = self
            .judge
            .judge_with_fallback(
                &JudgeRequest::new(&design_rubric(), &subject, DESIGN_INSTRUCTIONS),
                || fallback,
            )
            .await;
        let used_fallback = verdict.source == VerdictSource::Deterministic;
        let report = if used_fallback {
            report
        } else {
            design_report_from_verdict(&skill.name, &verdict)
        };

        Ok(DesignJudgement {
            verdict,
            report,
            used_fallback,
        })
    }

    /// Judges `skill` against [`safety_rubric`], never trusting it over a rule
    /// hit.
    ///
    /// The deterministic [`scan_safety`] result always runs first:
    ///
    /// - if the scan auto-failed (a rule carried an `AutoFailCode`), the scan
    ///   report is returned unchanged — score `0.0`, blocked, no dimensions,
    ///   `auto_fail` set — and the model is not consulted, because no verdict can
    ///   reopen an auto-fail;
    /// - otherwise every scan finding is kept, the effective score is the lower
    ///   of the model verdict (already normalized) and the scan, and the gate
    ///   closes when the scan blocked or the score is below `0.80`;
    /// - the five safety dimensions come from the verdict's dimension scores on
    ///   the model path and from the deterministic scan on the fallback path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid safety rule pattern and propagates the error from
    /// [`SkillDefinition::to_markdown`].
    pub async fn safety(
        &self,
        skill: &SkillDefinition,
        rules: &[SafetyRule],
    ) -> Result<SafetyJudgement> {
        let scan = scan_safety(skill, rules, SAFETY_GATE_THRESHOLD)?;
        if let Some(code) = scan.auto_fail {
            return Ok(SafetyJudgement {
                verdict: Verdict::deterministic(
                    0.0,
                    format!("deterministic safety scan triggered auto-fail {code}"),
                ),
                report: scan,
                used_fallback: true,
            });
        }

        let subject = skill.to_markdown()?;
        let fallback = Verdict::deterministic(
            scan.score_pct,
            if scan.findings.is_empty() {
                "deterministic safety scan found no matching rule".to_string()
            } else {
                format!(
                    "deterministic safety scan matched {count} rule(s) and scored {point} of 100",
                    count = scan.findings.len(),
                    point = points_from_fraction(scan.score_pct)
                )
            },
        );

        let verdict = self
            .judge
            .judge_with_fallback(
                &JudgeRequest::new(&safety_rubric(), &subject, &safety_instructions()),
                || fallback,
            )
            .await;
        let used_fallback = verdict.source == VerdictSource::Deterministic;

        let (score_pct, dimensions) = if used_fallback {
            (scan.score_pct, scan.dimensions.clone())
        } else {
            (
                round4(verdict.score.min(scan.score_pct)),
                Some(safety_dimensions_from_scores(&verdict.dimension_scores)),
            )
        };
        let blocked = scan.blocked || score_pct < SAFETY_GATE_THRESHOLD;

        Ok(SafetyJudgement {
            verdict,
            report: SafetyReport {
                skill_name: skill.name.clone(),
                score_pct,
                gate_threshold: SAFETY_GATE_THRESHOLD,
                blocked,
                auto_fail: None,
                dimensions,
                findings: scan.findings,
            },
            used_fallback,
        })
    }
}

impl std::fmt::Debug for SkillJudge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SkillJudge")
            .field("judge", &self.judge)
            .finish()
    }
}

/// Assembles a [`DesignReport`] from a model verdict.
///
/// When the verdict carries any of the seven design dimensions, each present
/// dimension awards `clamp(raw, 0, 100) / 100 · max` points and absent
/// dimensions award nothing; otherwise the overall score is distributed
/// proportionally across the maxima. In both branches the points are rounded to
/// two decimals, summed into [`DesignReport::design_score`], and mapped onto the
/// fraction, grade, knowledge delta, and readiness (`>= 0.60`) fields exactly as
/// [`score_design`] does.
#[must_use]
pub fn design_report_from_verdict(skill_name: &str, verdict: &Verdict) -> DesignReport {
    let carries_dimensions = DESIGN_DIMENSIONS
        .iter()
        .any(|(name, _, _)| verdict.dimension_scores.contains_key(*name));

    let mut dimensions = BTreeMap::new();
    let mut awarded = 0.0;
    for (name, max, _) in DESIGN_DIMENSIONS {
        let score = if carries_dimensions {
            verdict.dimension_scores.get(name).map_or(0.0, |raw| {
                round2(raw.clamp(0.0, RUBRIC_SCALE) / RUBRIC_SCALE * max)
            })
        } else {
            round2(verdict.score * max)
        };
        awarded += score;
        dimensions.insert(name.to_string(), ScoreDimension { score, max });
    }

    let design_score = if carries_dimensions {
        round2(awarded)
    } else {
        round2(verdict.score * RUBRIC_SCALE)
    };
    let design_score_pct = fraction_from_points(design_score);
    DesignReport {
        skill_name: skill_name.to_string(),
        design_score,
        design_score_max: RUBRIC_SCALE,
        design_score_pct,
        grade: grade_for(design_score_pct),
        knowledge_delta: round2(design_score / RUBRIC_SCALE),
        ready_for_functional_testing: design_score_pct >= DESIGN_GATE_THRESHOLD,
        dimensions,
    }
}

/// Builds the five safety dimensions from a verdict's raw dimension scores.
///
/// Every dimension awards `clamp(raw, 0, 100) / 100 · max` points, rounded to
/// two decimals; a dimension absent from `scores` awards nothing. The result
/// always carries all five ids, so the judged report keeps the Python shape.
pub(crate) fn safety_dimensions_from_scores(
    scores: &BTreeMap<String, f64>,
) -> BTreeMap<String, ScoreDimension> {
    SAFETY_DIMENSIONS
        .iter()
        .map(|(name, max, _)| {
            let raw = scores.get(*name).copied().unwrap_or(0.0);
            (
                (*name).to_string(),
                ScoreDimension {
                    score: round2(raw.clamp(0.0, RUBRIC_SCALE) / RUBRIC_SCALE * max),
                    max: *max,
                },
            )
        })
        .collect()
}

/// Renders the safety criteria: one instruction sentence per risk dimension.
fn safety_instructions() -> String {
    let mut instructions = String::from(
        "Review this SKILL.md document for safety risks. Score each dimension from 0 to 100, \
         where 100 means the risk is absent and 0 means the document is dangerous.",
    );
    for (name, _, description) in SAFETY_DIMENSIONS {
        instructions.push(' ');
        instructions.push_str(name);
        instructions.push_str(": ");
        instructions.push_str(description);
    }
    instructions.push_str(
        " Answer with a JSON object carrying score, passed, reasoning, and dimension_scores.",
    );
    instructions
}

#[cfg(test)]
mod tests {
    use super::*;

    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    use crate::skill_evaluator::report::AutoFailCode;

    /// A skill body that satisfies all seven design dimensions.
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

    fn fenced_verdict(score: f64, passed: bool) -> String {
        format!(
            "Here is the verdict.\n```json\n{{\"score\": {score}, \"passed\": {passed}, \
             \"reasoning\": \"ok\"}}\n```\n"
        )
    }

    fn judge(answer: &str) -> SkillJudge {
        SkillJudge::new(Arc::new(ScriptedModel::text(answer)))
    }

    #[test]
    fn design_rubric_matches_the_structural_maxima() {
        let rubric = design_rubric();
        assert!((rubric.scale - 100.0).abs() < 1e-9);
        assert_eq!(rubric.dimensions.len(), DESIGN_DIMENSIONS.len());

        let total: f64 = rubric
            .dimensions
            .iter()
            .map(|dimension| dimension.weight)
            .sum();
        assert!((total - 100.0).abs() < 1e-9);

        for (name, max, _) in DESIGN_DIMENSIONS {
            let dimension = rubric
                .dimensions
                .iter()
                .find(|dimension| dimension.name == name)
                .expect("declared dimension");
            assert!((dimension.weight - max).abs() < 1e-9, "{name}");
        }
        rubric.validate().expect("valid rubric");
    }

    #[test]
    fn safety_rubric_weights_sum_to_one_hundred() {
        let rubric = safety_rubric();
        assert!((rubric.scale - 100.0).abs() < 1e-9);
        assert_eq!(rubric.dimensions.len(), 5);
        let total: f64 = rubric
            .dimensions
            .iter()
            .map(|dimension| dimension.weight)
            .sum();
        assert!((total - 100.0).abs() < 1e-9);
        for (name, max, _) in SAFETY_DIMENSIONS {
            let dimension = rubric
                .dimensions
                .iter()
                .find(|dimension| dimension.name == name)
                .expect("declared dimension");
            assert!((dimension.weight - max).abs() < 1e-9, "{name}");
        }
        rubric.validate().expect("valid rubric");
    }

    #[tokio::test]
    async fn model_design_verdict_builds_the_report() {
        let skill_judge = judge(&fenced_verdict(0.8, true));

        let judgement = skill_judge.design(&skill()).await.expect("design");

        assert!(!judgement.used_fallback);
        assert_eq!(judgement.verdict.source, VerdictSource::Model);
        assert!((judgement.report.design_score - 80.0).abs() < 1e-9);
        assert!((judgement.report.design_score_pct - 0.8).abs() < 1e-9);
        assert!((judgement.report.knowledge_delta - 0.8).abs() < 1e-9);
        assert_eq!(judgement.report.grade, "B");
        assert!(judgement.report.ready_for_functional_testing);
        assert_eq!(judgement.report.dimensions.len(), 7);
        let d2 = judgement.report.dimensions["d2_expert_knowledge_craft"].clone();
        assert!((d2.score - 20.0).abs() < 1e-9);
        assert!((d2.max - 25.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn malformed_design_answer_uses_the_deterministic_report() {
        let skill_judge = judge("no structure");
        let skill = skill();
        let expected = score_design(&skill);

        let judgement = skill_judge.design(&skill).await.expect("design");

        assert!(judgement.used_fallback);
        assert_eq!(judgement.verdict.source, VerdictSource::Deterministic);
        assert_eq!(judgement.report, expected);
        assert!((judgement.report.design_score - expected.design_score).abs() < 1e-9);
    }

    #[tokio::test]
    async fn model_failure_uses_the_deterministic_report() {
        let skill_judge =
            SkillJudge::new(Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")])));
        let skill = skill();

        let judgement = skill_judge.design(&skill).await.expect("design");

        assert!(judgement.used_fallback);
        assert_eq!(judgement.verdict.source, VerdictSource::Deterministic);
        assert_eq!(judgement.report, score_design(&skill));
    }

    #[tokio::test]
    async fn fallback_verdict_dimension_scores_use_the_rubric_scale() {
        let skill_judge = judge("no structure");
        let skill = skill();

        let judgement = skill_judge.design(&skill).await.expect("design");

        assert_eq!(judgement.verdict.dimension_scores.len(), 7);
        for (name, score) in &judgement.verdict.dimension_scores {
            assert!((0.0..=100.0).contains(score), "{name} = {score}");
        }
        let d1 = judgement.verdict.dimension_scores["d1_knowledge_ratio"];
        assert!((d1 - 100.0).abs() < 1e-9, "{d1}");
        let d2 = judgement.verdict.dimension_scores["d2_expert_knowledge_craft"];
        assert!((d2 - 100.0).abs() < 1e-9, "{d2}");
        assert!((judgement.verdict.score - judgement.report.design_score / 100.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn a_critical_rule_hit_blocks_a_passing_verdict() {
        let skill_judge = judge(&fenced_verdict(0.95, true));
        let rules = [SafetyRule::new(
            "demo",
            "data_leakage",
            "critical",
            "mentions the skill name",
        )];

        let judgement = skill_judge.safety(&skill(), &rules).await.expect("safety");

        assert!(!judgement.used_fallback);
        assert_eq!(judgement.verdict.source, VerdictSource::Model);
        assert!(judgement.report.blocked);
        assert!(judgement.report.auto_fail.is_none());
        assert_eq!(judgement.report.findings.len(), 1);
        assert!((judgement.report.score_pct - 0.75).abs() < 1e-9);
        assert!((judgement.report.gate_threshold - 0.8).abs() < 1e-9);
        assert!(judgement.report.dimensions.is_some());
        assert_eq!(judgement.report.skill_name, "demo");
    }

    #[tokio::test]
    async fn a_failing_verdict_blocks_without_rules() {
        let skill_judge = judge(&fenced_verdict(0.2, false));

        let judgement = skill_judge.safety(&skill(), &[]).await.expect("safety");

        assert!(!judgement.used_fallback);
        assert!(judgement.report.blocked);
        assert!((judgement.report.score_pct - 0.2).abs() < 1e-9);
        assert!(judgement.report.findings.is_empty(), "no rules matched");
    }

    #[tokio::test]
    async fn a_passing_verdict_keeps_the_gate_open() {
        let skill_judge = judge(&fenced_verdict(0.9, true));

        let judgement = skill_judge.safety(&skill(), &[]).await.expect("safety");

        assert!(!judgement.used_fallback);
        assert!(!judgement.report.blocked);
        assert!((judgement.report.score_pct - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn a_malformed_safety_answer_falls_back_to_the_scan() {
        let skill_judge = judge("no structure");

        let judgement = skill_judge.safety(&skill(), &[]).await.expect("safety");

        assert!(judgement.used_fallback);
        assert_eq!(judgement.verdict.source, VerdictSource::Deterministic);
        assert!(!judgement.report.blocked);
        assert!((judgement.report.score_pct - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn an_invalid_rule_pattern_is_a_validation_error() {
        let skill_judge = judge(&fenced_verdict(0.9, true));
        let rules = [SafetyRule::new("(", "unsafe", "critical", "bad pattern")];

        let error = skill_judge
            .safety(&skill(), &rules)
            .await
            .expect_err("invalid pattern");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn verdict_dimensions_win_over_the_overall_score() {
        let mut scores = BTreeMap::new();
        scores.insert("d1_knowledge_ratio".to_string(), 50.0);
        scores.insert("d2_expert_knowledge_craft".to_string(), 80.0);
        scores.insert("unknown_dimension".to_string(), 100.0);
        let verdict = Verdict::model(0.1, false, "partial", scores);

        let report = design_report_from_verdict("demo", &verdict);

        assert_eq!(report.dimensions.len(), 7);
        assert!((report.dimensions["d1_knowledge_ratio"].score - 5.0).abs() < 1e-9);
        assert!((report.dimensions["d2_expert_knowledge_craft"].score - 20.0).abs() < 1e-9);
        assert!((report.dimensions["d5_pattern_freedom_fit"].score - 0.0).abs() < 1e-9);
        assert!((report.design_score - 25.0).abs() < 1e-9);
        assert_eq!(report.grade, "F");
        assert!(!report.ready_for_functional_testing);
        assert_eq!(report.skill_name, "demo");
    }

    #[test]
    fn out_of_range_dimension_scores_are_clamped() {
        let mut scores = BTreeMap::new();
        scores.insert("d1_knowledge_ratio".to_string(), 500.0);
        scores.insert("d2_expert_knowledge_craft".to_string(), -20.0);
        let verdict = Verdict::model(0.5, true, "odd", scores);

        let report = design_report_from_verdict("demo", &verdict);

        assert!((report.dimensions["d1_knowledge_ratio"].score - 10.0).abs() < 1e-9);
        assert!((report.dimensions["d2_expert_knowledge_craft"].score - 0.0).abs() < 1e-9);
        assert!((report.design_score - 10.0).abs() < 1e-9);
        assert!(!report.ready_for_functional_testing);
    }

    #[test]
    fn debug_names_the_underlying_judge() {
        let skill_judge = SkillJudge::new(Arc::new(
            ScriptedModel::builder()
                .provider("acme")
                .model_name("acme-judge")
                .build(),
        ))
        .with_system_prompt("be strict");

        let rendered = format!("{skill_judge:?}");
        assert!(rendered.contains("SkillJudge"), "{rendered}");
        assert!(rendered.contains("acme-judge"), "{rendered}");
        assert!(
            rendered.contains("system_prompt_override: true"),
            "{rendered}"
        );
        assert_eq!(skill_judge.judge().model().model_name(), "acme-judge");
    }

    /// A fenced verdict carrying explicit per-dimension scores.
    fn fenced_verdict_with_dimensions(score: f64, dimensions: &[(&str, f64)]) -> String {
        let rendered = dimensions
            .iter()
            .map(|(name, value)| format!("\"{name}\": {value}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "```json\n{{\"score\": {score}, \"passed\": true, \"reasoning\": \"ok\", \
             \"dimension_scores\": {{{rendered}}}}}\n```"
        )
    }

    #[tokio::test]
    async fn the_fallback_scan_publishes_the_five_python_dimensions() {
        let skill_judge = judge("no structure");

        let judgement = skill_judge.safety(&skill(), &[]).await.expect("safety");

        assert!(judgement.used_fallback);
        let dimensions = judgement.report.dimensions.as_ref().expect("dimensions");
        assert_eq!(dimensions.len(), 5);
        for (name, max) in [
            ("d1_harmful_instruction", 30.0),
            ("d2_scope_containment", 25.0),
            ("d3_data_handling", 20.0),
            ("d4_injection_resilience", 15.0),
            ("d5_guardrails", 10.0),
        ] {
            let dimension = dimensions.get(name).expect(name);
            assert!((dimension.max - max).abs() < 1e-9, "{name}");
            assert!((dimension.score - max).abs() < 1e-9, "{name}");
        }
    }

    #[tokio::test]
    async fn a_model_verdict_supplies_the_safety_dimensions() {
        let answer = fenced_verdict_with_dimensions(
            0.83,
            &[
                ("d1_harmful_instruction", 60.0),
                ("d2_scope_containment", 80.0),
                ("d3_data_handling", 100.0),
                ("d4_injection_resilience", 100.0),
                ("d5_guardrails", 100.0),
            ],
        );
        let skill_judge = judge(&answer);

        let judgement = skill_judge.safety(&skill(), &[]).await.expect("safety");

        assert!(!judgement.used_fallback);
        assert!((judgement.report.score_pct - 0.83).abs() < 1e-9);
        assert!(!judgement.report.blocked);
        let dimensions = judgement.report.dimensions.as_ref().expect("dimensions");
        assert!((dimensions["d1_harmful_instruction"].score - 18.0).abs() < 1e-9);
        assert!((dimensions["d2_scope_containment"].score - 20.0).abs() < 1e-9);
        assert!((dimensions["d3_data_handling"].score - 20.0).abs() < 1e-9);
        assert!((dimensions["d4_injection_resilience"].score - 15.0).abs() < 1e-9);
        assert!((dimensions["d5_guardrails"].score - 10.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn an_auto_fail_cannot_be_reopened_by_a_passing_verdict() {
        let model = ScriptedModel::text(fenced_verdict(1.0, true));
        let skill_judge = SkillJudge::new(Arc::new(model.clone()));
        let rules = [
            SafetyRule::new("demo", "d3_data_handling", "info", "leaks the name")
                .with_auto_fail(AutoFailCode::UndeclaredExfiltration),
        ];

        let judgement = skill_judge.safety(&skill(), &rules).await.expect("safety");

        assert!(judgement.used_fallback);
        assert_eq!(judgement.verdict.source, VerdictSource::Deterministic);
        assert_eq!(model.call_count(), 0, "the model must not be consulted");
        assert_eq!(judgement.report.score_pct, 0.0);
        assert!(judgement.report.blocked);
        assert!(judgement.report.dimensions.is_none());
        assert_eq!(
            judgement.report.auto_fail,
            Some(AutoFailCode::UndeclaredExfiltration)
        );
        assert_eq!(judgement.report.findings.len(), 1);
    }

    #[test]
    fn the_new_safety_dimension_ids_appear_in_the_instructions() {
        let instructions = safety_instructions();
        for (name, _, _) in SAFETY_DIMENSIONS {
            assert!(instructions.contains(name), "{name}");
        }
    }

    #[test]
    fn design_readiness_from_a_verdict_uses_the_sixty_percent_gate() {
        let ready =
            design_report_from_verdict("demo", &Verdict::model(0.60, true, "ok", BTreeMap::new()));
        assert!((ready.design_score_pct - 0.6).abs() < 1e-9);
        assert!(ready.ready_for_functional_testing);
        assert_eq!(ready.grade, "D");

        let not_ready =
            design_report_from_verdict("demo", &Verdict::model(0.59, false, "ok", BTreeMap::new()));
        assert!((not_ready.design_score_pct - 0.59).abs() < 1e-9);
        assert!(!not_ready.ready_for_functional_testing);
        assert_eq!(not_ready.grade, "F");
    }
}
