//! Deterministic scoring: the design rubric, the safety scan, and aggregation.
//!
//! Three pure functions compose the evaluator's scoring surface:
//!
//! - [`score_design`] applies a seven-dimension structural rubric to a
//!   [`SkillDefinition`] and returns a [`DesignReport`].
//! - [`scan_safety`] scans the rendered `SKILL.md` for caller-supplied regex
//!   rules and returns a [`SafetyReport`] gated at a threshold.
//! - [`aggregate`] weights the design and functional results and returns the
//!   [`FinalScore`] for the run.
//!
//! Python graded design by asking a model to apply a prose rubric; this port
//! replaces that with deterministic structure checks (headings, lists, code
//! fences, resource references, output wording) so results are reproducible
//! offline. The report field names and the `0.40·design + 0.60·functional`
//! weights are preserved.
//!
//! # Unit rule
//!
//! Points are 0–100 and only ever live in `*_score`, `*_score_max`, and
//! [`ScoreDimension`] members. Every `*_pct` field, the safety
//! `gate_threshold`, the functional `pass_rate`, and `contribution` are
//! fractions of `1.0`. `fraction_from_points` and `points_from_fraction` are the
//! only conversions between the two scales, and `round4` keeps fraction
//! arithmetic byte-stable.
//!
//! # Design rubric
//!
//! Each dimension awards `0.0`, half its maximum, or its full maximum:
//!
//! | Dimension | Max | Full when |
//! |---|---|---|
//! | `d1_knowledge_ratio` | 10 | body has at least 200 characters |
//! | `d2_expert_knowledge_craft` | 25 | at least 6 distinct section headings plus numbered steps |
//! | `d3_spec_compliance` | 15 | name, description, and markdown round trip all hold |
//! | `d4_progressive_disclosure` | 10 | a resource exists and the body references it |
//! | `d5_pattern_freedom_fit` | 15 | the body has both a `#` and a `##` heading |
//! | `d6_practical_usability` | 15 | the body has a list item or a fenced code block |
//! | `d7_output_specification` | 10 | the body describes the output/result format |
//!
//! The dimension maxima sum to 100; [`DesignReport::design_score`] is the sum of
//! the awarded points, so an all-full rubric reads as `100.0` points
//! (`design_score_pct = 1.0`) and a half-full one as `50.0` points (`0.5`).
//!
//! # Safety contract
//!
//! The scan starts at 100 points and loses 25 per finding. A rule carrying an
//! [`AutoFailCode`] short-circuits everything: the score becomes `0.0`, the run
//! is blocked, and no dimensions are reported. Otherwise the five
//! [`SAFETY_DIMENSIONS`](crate::skill_evaluator::report) are filled
//! proportionally, so a scan-only run has the same shape as a model-judged one.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::skill_creator::definition::NAME_PATTERN;
use crate::skill_creator::SkillDefinition;

use super::report::{
    grade_for, AutoFailCode, DesignReport, EvaluatorScore, EvaluatorScores, FinalScore,
    FunctionalReport, SafetyFinding, SafetyReport, ScoreDimension, GATE_E1, GATE_E3,
    PIPELINE_BLOCKED_AT_E1, PIPELINE_BLOCKED_AT_E3, PIPELINE_COMPLETED, SAFETY_DIMENSIONS,
};

/// Regex detecting a reference to a resource-like file inside instruction text.
const RESOURCE_REFERENCE_PATTERN: &str = r"(?i)[\w./-]+\.(md|txt|json|ya?ml|py|sh|toml|rs)";

/// The safety penalty applied per finding, in points.
const SAFETY_PENALTY_PER_FINDING: f64 = 25.0;

/// The design fraction required to keep the E1 gate open.
///
/// Single source of truth: the design readiness flag and the aggregation gate
/// both read it.
pub(crate) const DESIGN_GATE_THRESHOLD: f64 = 0.60;

/// The final-score fraction at or above which a skill is release-ready.
const RELEASE_THRESHOLD: f64 = 0.80;

/// One regex-based safety rule.
///
/// A rule is a pattern, the dimension it belongs to, a severity (`info`,
/// `warning`, `critical`, …), the message reported when it matches, and an
/// optional [`AutoFailCode`] that turns a match into an unconditional failure.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SafetyRule {
    /// The regular expression matched against the rendered skill document.
    pub pattern: String,
    /// The safety dimension the rule covers, e.g. `data_handling`.
    pub dimension: String,
    /// The finding severity; exactly `critical` blocks unconditionally.
    pub severity: String,
    /// The message copied into a matched [`SafetyFinding`].
    pub message: String,
    /// The auto-fail condition a match triggers, if any.
    pub auto_fail: Option<AutoFailCode>,
}

impl SafetyRule {
    /// Creates a rule from its pattern, dimension, severity, and message.
    #[must_use]
    pub fn new(
        pattern: impl Into<String>,
        dimension: impl Into<String>,
        severity: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            dimension: dimension.into(),
            severity: severity.into(),
            message: message.into(),
            auto_fail: None,
        }
    }

    /// Marks the rule as an auto-fail condition.
    #[must_use]
    pub fn with_auto_fail(mut self, code: AutoFailCode) -> Self {
        self.auto_fail = Some(code);
        self
    }
}

/// Scores a skill against the structural design rubric.
///
/// The result is deterministic: no model is consulted and only
/// [`SkillDefinition::instructions`], the skill identity, and its resources are
/// inspected. `design_score` is in points (0–100), `design_score_pct` is the
/// same value as a fraction of `1.0`, and the readiness flag needs at least
/// `0.60`.
#[must_use]
pub fn score_design(skill: &SkillDefinition) -> DesignReport {
    let body = skill.instructions.as_str();
    let levels = [
        ("d1_knowledge_ratio", 10.0, d1_level(body)),
        ("d2_expert_knowledge_craft", 25.0, d2_level(body)),
        ("d3_spec_compliance", 15.0, d3_level(skill)),
        ("d4_progressive_disclosure", 10.0, d4_level(skill)),
        ("d5_pattern_freedom_fit", 15.0, d5_level(body)),
        ("d6_practical_usability", 15.0, d6_level(body)),
        ("d7_output_specification", 10.0, d7_level(body)),
    ];

    let mut dimensions = BTreeMap::new();
    let mut awarded = 0.0;
    for (name, max, level) in levels {
        let score = match level {
            0 => 0.0,
            1 => max / 2.0,
            _ => max,
        };
        awarded += score;
        dimensions.insert(name.to_string(), ScoreDimension { score, max });
    }

    let design_score = round2(awarded);
    let design_score_pct = fraction_from_points(design_score);
    DesignReport {
        skill_name: skill.name.clone(),
        design_score,
        design_score_max: 100.0,
        design_score_pct,
        grade: grade_for(design_score_pct),
        knowledge_delta: round2(design_score / 100.0),
        ready_for_functional_testing: design_score_pct >= DESIGN_GATE_THRESHOLD,
        dimensions,
    }
}

/// Scans a skill document for safety hazards.
///
/// The rendered `SKILL.md` is scanned; every rule that matches contributes one
/// [`SafetyFinding`] (duplicates are not collapsed). The score starts at 100
/// points and loses 25 per finding, clamped at zero, and is reported as a
/// fraction of `1.0`.
///
/// The first matching rule carrying an [`AutoFailCode`] short-circuits the
/// report: `score_pct` becomes `0.0`, `blocked` is `true`, `dimensions` is
/// `None`, and `auto_fail` names the code. Otherwise the report is blocked when
/// the score fraction falls below `gate_threshold` or when any finding has the
/// exact severity `critical`, and the five safety dimensions are filled
/// proportionally to their maxima.
///
/// # Errors
///
/// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
/// when a rule pattern does not compile, and propagates the error from
/// [`SkillDefinition::to_markdown`] when the skill cannot be rendered.
pub fn scan_safety(
    skill: &SkillDefinition,
    rules: &[SafetyRule],
    gate_threshold: f64,
) -> Result<SafetyReport> {
    let markdown = skill.to_markdown()?;
    let mut findings = Vec::new();
    let mut auto_fail = None;
    for rule in rules {
        let regex = Regex::new(&rule.pattern).map_err(|error| {
            Error::validation(format!(
                "invalid safety rule pattern {:?}: {error}",
                rule.pattern
            ))
        })?;
        if regex.is_match(&markdown) {
            if auto_fail.is_none() {
                auto_fail = rule.auto_fail;
            }
            findings.push(SafetyFinding {
                dimension: rule.dimension.clone(),
                severity: rule.severity.clone(),
                message: rule.message.clone(),
            });
        }
    }

    if let Some(code) = auto_fail {
        return Ok(SafetyReport {
            skill_name: skill.name.clone(),
            score_pct: 0.0,
            gate_threshold,
            blocked: true,
            auto_fail: Some(code),
            dimensions: None,
            findings,
        });
    }

    let points = (100.0 - SAFETY_PENALTY_PER_FINDING * findings.len() as f64).max(0.0);
    let score_pct = fraction_from_points(points);
    let blocked = score_pct < gate_threshold
        || findings
            .iter()
            .any(|finding| finding.severity == "critical");

    Ok(SafetyReport {
        skill_name: skill.name.clone(),
        score_pct,
        gate_threshold,
        blocked,
        auto_fail: None,
        dimensions: Some(safety_dimensions_from_fraction(score_pct)),
        findings,
    })
}

/// Combines the design and functional results into the weighted final score.
///
/// The gate order matches the Python pipeline — E1, then E3:
///
/// 1. a design fraction below `DESIGN_GATE_THRESHOLD` (`0.60`) stops the run with
///    [`PIPELINE_BLOCKED_AT_E1`];
/// 2. otherwise a blocked safety report stops it with [`PIPELINE_BLOCKED_AT_E3`];
/// 3. otherwise the run is [`PIPELINE_COMPLETED`] and
///    `final_score_pct = design*design_weight + pass_rate*functional_weight`
///    (a fraction of `1.0`).
///
/// A blocked run has no valid final score, so `final_score_pct` is `0.0`, the
/// grade is `F`, and no gate is reported as passed — even though the evaluator
/// contributions are still filled in, because the author needs the full picture.
/// A completed run reports [`GATE_E3`] as its only passed gate: it is the only
/// gate-role evaluator and it passed.
#[must_use]
pub fn aggregate(
    design: &DesignReport,
    functional: &FunctionalReport,
    safety: &SafetyReport,
    design_weight: f64,
    functional_weight: f64,
) -> FinalScore {
    let design_fraction = design.design_score_pct;
    let pass_rate = functional.summary.pass_rate;
    let evaluators = EvaluatorScores {
        e1_llm_judge: Some(EvaluatorScore::weighted(design_fraction, design_weight)),
        e2_functional: EvaluatorScore::weighted(pass_rate, functional_weight),
    };

    if design_fraction < DESIGN_GATE_THRESHOLD {
        return FinalScore {
            skill_name: design.skill_name.clone(),
            pipeline_status: PIPELINE_BLOCKED_AT_E1.to_string(),
            final_score_pct: 0.0,
            grade: "F".to_string(),
            evaluators,
            gates_passed: Vec::new(),
            recommendation: format!(
                "blocked at the {GATE_E1} gate: design score {} is below \
                 {DESIGN_GATE_THRESHOLD}, fix the design findings before testing",
                design.design_score_pct
            ),
        };
    }

    if safety.blocked {
        return FinalScore {
            skill_name: design.skill_name.clone(),
            pipeline_status: PIPELINE_BLOCKED_AT_E3.to_string(),
            final_score_pct: 0.0,
            grade: "F".to_string(),
            evaluators,
            gates_passed: Vec::new(),
            recommendation: format!(
                "blocked at the {GATE_E3} gate: safety score {} with {} finding(s) did not \
                 pass the gate",
                safety.score_pct,
                safety.findings.len()
            ),
        };
    }

    let final_score_pct = round4(design_weight * design_fraction + functional_weight * pass_rate);
    let recommendation = if final_score_pct >= RELEASE_THRESHOLD {
        "ready for release"
    } else if final_score_pct >= DESIGN_GATE_THRESHOLD {
        "usable, refine failing cases"
    } else {
        "not ready for release"
    };

    FinalScore {
        skill_name: design.skill_name.clone(),
        pipeline_status: PIPELINE_COMPLETED.to_string(),
        final_score_pct,
        grade: grade_for(final_score_pct),
        evaluators,
        gates_passed: vec![GATE_E3.to_string()],
        recommendation: recommendation.to_string(),
    }
}

/// Rounds a score to two decimal places.
///
/// This is the single two-decimal rounding helper for the module: point values
/// use it, fractions use [`round4`].
pub(crate) fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Rounds a fraction to four decimal places.
///
/// Used for every `*_pct`/`contribution` value so `0.4 * 0.74` reads exactly
/// `0.296` and two runs of the same inputs serialize byte-identically.
pub(crate) fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// Converts points (0–100) into a fraction of `1.0`, rounded to 4 decimals.
pub(crate) fn fraction_from_points(points: f64) -> f64 {
    round4(points / 100.0)
}

/// Converts a fraction of `1.0` into points (0–100), rounded to 2 decimals.
pub(crate) fn points_from_fraction(fraction: f64) -> f64 {
    round2(fraction * 100.0)
}

/// Builds the five safety dimensions from a score fraction.
///
/// Every dimension receives `round2(fraction * max)` points, so the reported
/// points always add up to the score itself. This is what lets a scan-only run
/// publish the same `dimensions` shape a model verdict produces.
pub(crate) fn safety_dimensions_from_fraction(fraction: f64) -> BTreeMap<String, ScoreDimension> {
    SAFETY_DIMENSIONS
        .iter()
        .map(|(name, max, _)| {
            (
                (*name).to_string(),
                ScoreDimension {
                    score: round2(fraction * max),
                    max: *max,
                },
            )
        })
        .collect()
}

/// `d1_knowledge_ratio`: body length in characters.
fn d1_level(body: &str) -> u8 {
    let length = body.chars().count();
    if length >= 200 {
        2
    } else {
        u8::from(length >= 80)
    }
}

/// `d2_expert_knowledge_craft`: distinct section headings plus numbered steps.
fn d2_level(body: &str) -> u8 {
    let mut headings = BTreeSet::new();
    let mut steps = 0usize;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") || trimmed.starts_with("### ") {
            headings.insert(trimmed);
        }
        if is_numbered_step(trimmed) {
            steps += 1;
        }
    }
    let craft = headings.len() + steps;
    if craft >= 6 {
        2
    } else {
        u8::from(craft >= 3)
    }
}

/// `d3_spec_compliance`: name pattern, description, and markdown round trip.
fn d3_level(skill: &SkillDefinition) -> u8 {
    let name_ok = Regex::new(NAME_PATTERN).is_ok_and(|regex| regex.is_match(&skill.name));
    let description_ok = !skill.description.trim().is_empty();
    let round_trip_ok = skill
        .to_markdown()
        .ok()
        .and_then(|markdown| SkillDefinition::from_markdown(&markdown).ok())
        .is_some_and(|parsed| {
            parsed.name == skill.name
                && parsed.description == skill.description
                && parsed.instructions == skill.instructions
        });
    let satisfied = usize::from(name_ok) + usize::from(description_ok) + usize::from(round_trip_ok);
    match satisfied {
        3 => 2,
        2 => 1,
        _ => 0,
    }
}

/// `d4_progressive_disclosure`: resources exist and are discoverable from the body.
///
/// Resources count as referenced when the body mentions the resource path or its
/// last path component. Without resources, a resource-like mention (a token with
/// a known file extension, e.g. `references/guide.md`) still earns the half score
/// because the author signposted progressive disclosure even though no asset was
/// shipped.
fn d4_level(skill: &SkillDefinition) -> u8 {
    let body = skill.instructions.as_str();
    let referenced = skill.resources.iter().any(|resource| {
        let path = resource.path.as_str();
        let file_name = path.rsplit('/').next().unwrap_or(path);
        (!path.is_empty() && body.contains(path))
            || (!file_name.is_empty() && body.contains(file_name))
    });
    if !skill.resources.is_empty() && referenced {
        2
    } else {
        u8::from(!skill.resources.is_empty() || has_resource_reference(body))
    }
}

/// `d5_pattern_freedom_fit`: both top heading levels are present.
fn d5_level(body: &str) -> u8 {
    let has_title = body.lines().any(|line| line.trim().starts_with("# "));
    let has_section = body.lines().any(|line| line.trim().starts_with("## "));
    match (has_title, has_section) {
        (true, true) => 2,
        (true, false) | (false, true) => 1,
        (false, false) => 0,
    }
}

/// `d6_practical_usability`: runnable lists or code blocks.
fn d6_level(body: &str) -> u8 {
    let has_block = body.lines().any(|line| {
        let trimmed = line.trim();
        is_list_item(trimmed) || trimmed.starts_with("```")
    });
    if has_block {
        2
    } else {
        u8::from(has_inline_code(body))
    }
}

/// `d7_output_specification`: the body describes the expected output.
fn d7_level(body: &str) -> u8 {
    let lower = body.to_lowercase();
    let describes_output = ["output", "result", "format", "输出", "结果"]
        .iter()
        .any(|word| lower.contains(word));
    let mentions_response = ["return", "response", "returns"]
        .iter()
        .any(|word| lower.contains(word));
    match (describes_output, mentions_response) {
        (true, _) => 2,
        (false, true) => 1,
        (false, false) => 0,
    }
}

/// Whether the trimmed line opens an ordered step (`1.`, `2)`, …).
fn is_numbered_step(trimmed: &str) -> bool {
    let digits = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if digits == 0 {
        return false;
    }
    let mut rest = trimmed[digits..].chars();
    let marker = rest.next();
    if !matches!(marker, Some('.' | ')')) {
        return false;
    }
    rest.next().is_some_and(char::is_whitespace)
}

/// Whether the trimmed line is a bullet item (`- `, `* `, or `+ `).
fn is_list_item(trimmed: &str) -> bool {
    let mut chars = trimmed.chars();
    if !matches!(chars.next(), Some('-' | '*' | '+')) {
        return false;
    }
    chars.next().is_some_and(char::is_whitespace)
}

/// Whether the body holds at least one inline code span (a backtick pair).
fn has_inline_code(body: &str) -> bool {
    body.matches('`').count() >= 2
}

/// Whether the body references a resource-like file with a known extension.
fn has_resource_reference(body: &str) -> bool {
    Regex::new(RESOURCE_REFERENCE_PATTERN).is_ok_and(|regex| regex.is_match(body))
}

#[cfg(test)]
mod tests;
