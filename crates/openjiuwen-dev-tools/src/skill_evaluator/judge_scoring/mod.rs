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
//! # Safety gate threshold
//!
//! The fraction the safety score must reach is configurable: it defaults to
//! `0.80`, can be set once with [`SkillJudge::with_gate_threshold`], and can be
//! overridden per judgement with [`SkillJudge::safety_with_threshold`], which is
//! what lets `EvaluatorConfig::safety_gate_threshold` gate a judged run. The
//! applied threshold gates the score, is stored in
//! [`SafetyReport::gate_threshold`], and is handed to
//! [`scan_safety`](crate::skill_evaluator::scoring::scan_safety), so the
//! deterministic scan and the model verdict agree.
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

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;

use crate::judge::{JudgeRequest, ModelJudge, Rubric, RubricDimension, Verdict, VerdictSource};
use crate::skill_creator::SkillDefinition;

use super::report::{grade_for, DesignReport, SafetyReport, ScoreDimension, SAFETY_DIMENSIONS};
use super::scoring::{
    fraction_from_points, points_from_fraction, round2, round4, scan_safety, score_design,
    SafetyRule, DESIGN_GATE_THRESHOLD,
};

/// The safety gate threshold a fresh [`SkillJudge`] starts with, as a fraction
/// of `1.0`.
///
/// A judgement closes the gate below its threshold, so a rule hit can never be
/// talked past; callers raise or lower the fraction with
/// [`SkillJudge::with_gate_threshold`] or override it per judgement with
/// [`SkillJudge::safety_with_threshold`].
const DEFAULT_SAFETY_GATE_THRESHOLD: f64 = 0.80;

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
/// [`safety`](Self::safety) are the two judging stages of the evaluator. The
/// safety gate threshold starts at `0.80` and can be changed with
/// [`with_gate_threshold`](Self::with_gate_threshold) or overridden for a single
/// judgement with [`safety_with_threshold`](Self::safety_with_threshold).
pub struct SkillJudge {
    /// The model-backed judge consulted for every rubric.
    judge: ModelJudge,
    /// The fraction the safety score must reach to keep the gate open.
    gate_threshold: f64,
}

impl SkillJudge {
    /// Creates a judge over `model` with the default safety gate (`0.80`).
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            judge: ModelJudge::new(model),
            gate_threshold: DEFAULT_SAFETY_GATE_THRESHOLD,
        }
    }

    /// Replaces the built-in judge system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.judge = self.judge.with_system_prompt(prompt);
        self
    }

    /// Replaces the safety gate threshold used by [`safety`](Self::safety).
    ///
    /// The threshold is a fraction of `1.0`; it is validated when a safety
    /// judgement runs, so an out-of-range value is reported by
    /// [`safety`](Self::safety)/[`safety_with_threshold`](Self::safety_with_threshold)
    /// rather than by this builder.
    #[must_use]
    pub fn with_gate_threshold(mut self, threshold: f64) -> Self {
        self.gate_threshold = threshold;
        self
    }

    /// The safety gate threshold used by [`safety`](Self::safety), as a
    /// fraction of `1.0`.
    #[must_use]
    pub fn gate_threshold(&self) -> f64 {
        self.gate_threshold
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

    /// Judges `skill` against [`safety_rubric`] at the configured
    /// [`gate_threshold`](Self::gate_threshold), never trusting the model over a
    /// rule hit.
    ///
    /// This is [`safety_with_threshold`](Self::safety_with_threshold) with
    /// `self.gate_threshold`, which defaults to `0.80`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid safety rule pattern or a configured gate threshold that is
    /// not a finite fraction of `1.0`, and propagates the error from
    /// [`SkillDefinition::to_markdown`].
    pub async fn safety(
        &self,
        skill: &SkillDefinition,
        rules: &[SafetyRule],
    ) -> Result<SafetyJudgement> {
        self.safety_with_threshold(skill, rules, self.gate_threshold)
            .await
    }

    /// Judges `skill` against [`safety_rubric`], gating the result at
    /// `threshold`.
    ///
    /// The deterministic [`scan_safety`] result always runs first with the same
    /// `threshold`, so the scan and the model cannot disagree about the gate:
    ///
    /// - if the scan auto-failed (a rule carried an `AutoFailCode`), the scan
    ///   report is returned unchanged — score `0.0`, blocked, no dimensions,
    ///   `auto_fail` set — and the model is not consulted, because no verdict can
    ///   reopen an auto-fail;
    /// - otherwise every scan finding is kept, the effective score is the lower
    ///   of the model verdict (already normalized) and the scan, and the gate
    ///   closes when the scan blocked or the score is below `threshold`;
    /// - the five safety dimensions come from the verdict's dimension scores on
    ///   the model path and from the deterministic scan on the fallback path;
    /// - the report always carries `threshold` in
    ///   [`SafetyReport::gate_threshold`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when `threshold` is not a finite fraction of `1.0`, for an invalid safety
    /// rule pattern, and propagates the error from
    /// [`SkillDefinition::to_markdown`].
    pub async fn safety_with_threshold(
        &self,
        skill: &SkillDefinition,
        rules: &[SafetyRule],
        threshold: f64,
    ) -> Result<SafetyJudgement> {
        validate_gate_threshold(threshold)?;

        let scan = scan_safety(skill, rules, threshold)?;
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
        let blocked = scan.blocked || score_pct < threshold;

        Ok(SafetyJudgement {
            verdict,
            report: SafetyReport {
                skill_name: skill.name.clone(),
                score_pct,
                gate_threshold: threshold,
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
            .field("gate_threshold", &self.gate_threshold)
            .finish()
    }
}

/// Rejects a gate threshold that is not a finite fraction of `1.0`.
///
/// # Errors
///
/// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
/// naming the offending value for `NaN`, infinities, and values outside
/// `0.0..=1.0`.
fn validate_gate_threshold(threshold: f64) -> Result<()> {
    if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "safety gate threshold must be a finite fraction between 0.0 and 1.0, got {threshold}"
        )))
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
mod tests;
