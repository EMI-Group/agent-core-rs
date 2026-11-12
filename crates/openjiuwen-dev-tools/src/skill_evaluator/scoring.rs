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
//! the awarded points, so an all-full rubric reads as `100.0` and a half-full one
//! as `50.0`.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::skill_creator::definition::NAME_PATTERN;
use crate::skill_creator::SkillDefinition;

use super::report::{
    grade_for, DesignReport, EvaluatorScore, EvaluatorScores, FinalScore, FunctionalReport,
    SafetyFinding, SafetyReport, ScoreDimension,
};

/// Regex detecting a reference to a resource-like file inside instruction text.
const RESOURCE_REFERENCE_PATTERN: &str = r"(?i)[\w./-]+\.(md|txt|json|ya?ml|py|sh|toml|rs)";

/// The safety penalty applied per finding, in percentage points.
const SAFETY_PENALTY_PER_FINDING: f64 = 25.0;

/// One regex-based safety rule.
///
/// A rule is a pattern, the dimension it belongs to, a severity (`info`,
/// `warning`, `critical`, …), and the message reported when it matches.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SafetyRule {
    /// The regular expression matched against the rendered skill document.
    pub pattern: String,
    /// The safety dimension the rule covers, e.g. `data_leakage`.
    pub dimension: String,
    /// The finding severity; exactly `critical` blocks unconditionally.
    pub severity: String,
    /// The message copied into a matched [`SafetyFinding`].
    pub message: String,
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
        }
    }
}

/// Scores a skill against the structural design rubric.
///
/// The result is deterministic: no model is consulted and only
/// [`SkillDefinition::instructions`], the skill identity, and its resources are
/// inspected.
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
    let design_score_pct = round2(design_score);
    DesignReport {
        skill_name: skill.name.clone(),
        design_score,
        design_score_max: 100.0,
        design_score_pct,
        grade: grade_for(design_score_pct),
        knowledge_delta: round2(design_score / 100.0),
        ready_for_functional_testing: design_score_pct >= 50.0,
        dimensions,
    }
}

/// Scans a skill document for safety hazards.
///
/// The rendered `SKILL.md` is scanned; every rule that matches contributes one
/// [`SafetyFinding`] (duplicates are not collapsed). The score starts at 100 and
/// loses 25 points per finding, clamped at zero.
///
/// The report is blocked when the score fraction falls below `gate_threshold` or
/// when any finding has the exact severity `critical`.
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
    for rule in rules {
        let regex = Regex::new(&rule.pattern).map_err(|error| {
            Error::validation(format!(
                "invalid safety rule pattern {:?}: {error}",
                rule.pattern
            ))
        })?;
        if regex.is_match(&markdown) {
            findings.push(SafetyFinding {
                dimension: rule.dimension.clone(),
                severity: rule.severity.clone(),
                message: rule.message.clone(),
            });
        }
    }

    let score_pct = (100.0 - SAFETY_PENALTY_PER_FINDING * findings.len() as f64).max(0.0);
    let blocked = score_pct / 100.0 < gate_threshold
        || findings
            .iter()
            .any(|finding| finding.severity == "critical");

    Ok(SafetyReport {
        skill_name: skill.name.clone(),
        score_pct,
        gate_threshold,
        blocked,
        auto_fail: blocked,
        dimensions: None,
        findings,
    })
}

/// Combines the design and functional results into the weighted final score.
///
/// The functional pass rate is expressed as a percentage and weighted by
/// `functional_weight`; the design percentage is weighted by `design_weight`.
/// The pipeline stops at the first failing gate:
///
/// 1. a blocked safety report yields `BLOCKED_AT_SAFETY`;
/// 2. a design score that is not ready for functional testing yields
///    `BLOCKED_AT_E1_LLM_JUDGE`;
/// 3. otherwise the run `COMPLETED` and both contributions are summed.
#[must_use]
pub fn aggregate(
    design: &DesignReport,
    functional: &FunctionalReport,
    safety: &SafetyReport,
    design_weight: f64,
    functional_weight: f64,
) -> FinalScore {
    let evaluators = EvaluatorScores {
        e1_llm_judge: Some(EvaluatorScore {
            score_pct: round2(design.design_score_pct),
            weight: design_weight,
            contribution: round2(design_weight * design.design_score_pct),
        }),
        e2_functional: EvaluatorScore {
            score_pct: round2(functional.summary.pass_rate * 100.0),
            weight: functional_weight,
            contribution: round2(functional_weight * functional.summary.pass_rate * 100.0),
        },
    };

    if safety.blocked {
        return FinalScore {
            skill_name: design.skill_name.clone(),
            pipeline_status: "BLOCKED_AT_SAFETY".to_string(),
            final_score_pct: 0.0,
            grade: "F".to_string(),
            evaluators,
            gates_passed: Vec::new(),
            recommendation: "not ready for release".to_string(),
        };
    }

    if !design.ready_for_functional_testing {
        return FinalScore {
            skill_name: design.skill_name.clone(),
            pipeline_status: "BLOCKED_AT_E1_LLM_JUDGE".to_string(),
            final_score_pct: 0.0,
            grade: "F".to_string(),
            evaluators,
            gates_passed: vec!["e3_safety".to_string()],
            recommendation: "not ready for release".to_string(),
        };
    }

    let gates = vec![
        "e1_llm_judge".to_string(),
        "e2_functional".to_string(),
        "e3_safety".to_string(),
    ];
    let final_score_pct = round2(
        design_weight * design.design_score_pct
            + functional_weight * functional.summary.pass_rate * 100.0,
    );
    let recommendation = if final_score_pct >= 80.0 {
        "ready for release"
    } else if final_score_pct >= 60.0 {
        "usable, refine failing cases"
    } else {
        "not ready for release"
    };

    FinalScore {
        skill_name: design.skill_name.clone(),
        pipeline_status: "COMPLETED".to_string(),
        final_score_pct,
        grade: grade_for(final_score_pct),
        evaluators,
        gates_passed: gates,
        recommendation: recommendation.to_string(),
    }
}

/// Rounds a score to two decimal places.
///
/// This is the single rounding helper for the module: the runner and evaluator
/// layers reuse it so report arithmetic stays consistent.
pub(crate) fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
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
mod tests {
    use super::*;
    use crate::skill_creator::SkillResource;

    use super::super::report::CaseSummary;

    fn rich_skill() -> SkillDefinition {
        let body = concat!(
            "# Demo Skill\n\n",
            "## Overview\n\n",
            "This skill explains how to summarize a document end to end. It covers the input ",
            "format, the processing steps, and the expected result structure so that a reader ",
            "can reproduce the work without any extra guidance.\n\n",
            "## Steps\n\n",
            "1. Read the input document carefully.\n",
            "2. Extract the key facts and figures.\n",
            "3. Write the summary using the format below.\n\n",
            "## Output\n\n",
            "The result format is:\n\n",
            "```text\nname: summary\n```\n\n",
            "See references/guide.md for the full checklist."
        );
        SkillDefinition::new("demo", "Summarize documents", body)
            .with_resource(SkillResource::new("references/guide.md", "# Checklist"))
    }

    fn minimal_skill() -> SkillDefinition {
        SkillDefinition::new("min", "A minimal skill", "x")
    }

    fn report_with(design_pct: f64, ready: bool) -> DesignReport {
        DesignReport {
            skill_name: "demo".to_string(),
            design_score: design_pct,
            design_score_max: 100.0,
            design_score_pct: design_pct,
            grade: grade_for(design_pct),
            knowledge_delta: round2(design_pct / 100.0),
            ready_for_functional_testing: ready,
            dimensions: BTreeMap::new(),
        }
    }

    fn functional_with(pass_rate: f64) -> FunctionalReport {
        FunctionalReport {
            skill_name: "demo".to_string(),
            summary: CaseSummary {
                total: 2,
                passed: 1,
                partial: 0,
                failed: 1,
                pass_rate,
            },
            ..FunctionalReport::default()
        }
    }

    fn safety_with(blocked: bool) -> SafetyReport {
        SafetyReport {
            skill_name: "demo".to_string(),
            score_pct: if blocked { 50.0 } else { 100.0 },
            gate_threshold: 0.8,
            blocked,
            auto_fail: blocked,
            ..SafetyReport::default()
        }
    }

    #[test]
    fn rich_skill_scores_every_dimension() {
        let report = score_design(&rich_skill());
        assert_eq!(report.dimensions.len(), 7);
        for (name, dimension) in &report.dimensions {
            assert!(
                (dimension.score - dimension.max).abs() < 1e-9,
                "dimension {name} not full: {dimension:?}"
            );
        }
        assert!((report.design_score - 100.0).abs() < 1e-9);
        assert!((report.design_score_pct - 100.0).abs() < 1e-9);
        assert!((report.design_score_max - 100.0).abs() < 1e-9);
        assert_eq!(report.grade, "A");
        assert!((report.knowledge_delta - 1.0).abs() < 1e-9);
        assert!(report.ready_for_functional_testing);
        assert_eq!(report.skill_name, "demo");
    }

    #[test]
    fn rubric_maxima_sum_to_one_hundred() {
        let report = score_design(&rich_skill());
        let total: f64 = report
            .dimensions
            .values()
            .map(|dimension| dimension.max)
            .sum();
        assert!((total - 100.0).abs() < 1e-9, "maxima sum to {total}");
        assert!((report.dimensions["d5_pattern_freedom_fit"].max - 15.0).abs() < 1e-9);
    }

    #[test]
    fn minimal_skill_scores_low_and_is_not_ready() {
        let report = score_design(&minimal_skill());
        assert!(report.design_score_pct < 50.0, "{report:?}");
        assert!(!report.ready_for_functional_testing);
        assert_eq!(report.dimensions["d1_knowledge_ratio"].score, 0.0);
        assert_eq!(report.dimensions["d5_pattern_freedom_fit"].score, 0.0);
    }

    #[test]
    fn invalid_name_lowers_spec_compliance_without_panicking() {
        let skill = SkillDefinition::new("bad name!", "Has a bad name", "body");
        let report = score_design(&skill);
        let compliance = &report.dimensions["d3_spec_compliance"];
        assert!(compliance.score < compliance.max);
        // Only the non-blank description holds; the name and the markdown round
        // trip both fail because `to_markdown` validates the name.
        assert!((compliance.score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn knowledge_ratio_has_three_levels() {
        assert_eq!(d1_level(&"a".repeat(199)), 1);
        assert_eq!(d1_level(&"a".repeat(200)), 2);
        assert_eq!(d1_level(&"a".repeat(79)), 0);
        assert_eq!(d1_level(&"a".repeat(80)), 1);
    }

    #[test]
    fn numbered_and_bulleted_lines_are_detected() {
        assert!(is_numbered_step("1. first"));
        assert!(is_numbered_step("12) twelfth"));
        assert!(!is_numbered_step("1.first"));
        assert!(!is_numbered_step(".first"));
        assert!(!is_numbered_step("no digits"));
        assert!(is_list_item("- item"));
        assert!(is_list_item("* item"));
        assert!(is_list_item("+ item"));
        assert!(!is_list_item("-item"));
    }

    #[test]
    fn quad_hashes_do_not_count_as_sections() {
        let body = "#### Deep\n\n#### Also\n\n#### Third";
        assert_eq!(d2_level(body), 0);
        let body = "## A\n\n## B\n\n### C";
        assert_eq!(d2_level(body), 1);
    }

    #[test]
    fn resource_reference_detection() {
        assert!(has_resource_reference("see references/guide.md"));
        assert!(has_resource_reference("run install.sh"));
        assert!(!has_resource_reference("no files here"));
    }

    #[test]
    fn half_scores_when_only_some_criteria_hold() {
        // Has a title but no section heading.
        assert_eq!(d5_level("# Title only"), 1);
        // "return" alone is the half level for output specification.
        assert_eq!(d7_level("You must return an answer"), 1);
    }

    #[test]
    fn safety_without_rules_is_clean() {
        let report = scan_safety(&rich_skill(), &[], 0.8).expect("scan");
        assert!((report.score_pct - 100.0).abs() < 1e-9);
        assert!(!report.blocked);
        assert!(!report.auto_fail);
        assert_eq!(report.findings.len(), 0);
        assert!(report.dimensions.is_none());
        assert!((report.gate_threshold - 0.8).abs() < 1e-9);
        assert_eq!(report.skill_name, "demo");
    }

    #[test]
    fn soft_rule_match_reports_one_finding() {
        let rules = [SafetyRule::new(
            "checklist",
            "content",
            "warning",
            "mentions a checklist",
        )];
        let report = scan_safety(&rich_skill(), &rules, 0.5).expect("scan");
        assert_eq!(report.findings.len(), 1);
        assert!((report.score_pct - 75.0).abs() < 1e-9);
        assert!(!report.blocked);
        assert!((report.gate_threshold - 0.5).abs() < 1e-9);
        assert_eq!(report.findings[0].dimension, "content");
        assert_eq!(report.findings[0].severity, "warning");
    }

    #[test]
    fn critical_rule_blocks_unconditionally() {
        let rules = [SafetyRule::new("checklist", "secrets", "critical", "leak")];
        let report = scan_safety(&rich_skill(), &rules, 0.0).expect("scan");
        assert!(report.blocked);
        assert!(report.auto_fail);
        assert!((report.score_pct - 75.0).abs() < 1e-9);
    }

    #[test]
    fn two_matching_rules_halve_the_score() {
        let rules = [
            SafetyRule::new("checklist", "a", "info", "one"),
            SafetyRule::new("(?i)SUMMARY", "b", "warning", "two"),
        ];
        let report = scan_safety(&rich_skill(), &rules, 0.8).expect("scan");
        assert_eq!(report.findings.len(), 2);
        assert!((report.score_pct - 50.0).abs() < 1e-9);
        assert!(report.blocked);
    }

    #[test]
    fn non_matching_rule_leaves_the_score_intact() {
        let rules = [SafetyRule::new("does-not-appear", "a", "warning", "unused")];
        let report = scan_safety(&rich_skill(), &rules, 0.8).expect("scan");
        assert_eq!(report.findings.len(), 0);
        assert!((report.score_pct - 100.0).abs() < 1e-9);
        assert!(!report.blocked);
    }

    #[test]
    fn invalid_rule_pattern_is_a_validation_error() {
        let rules = [SafetyRule::new("[", "a", "warning", "broken")];
        let error = scan_safety(&rich_skill(), &rules, 0.8).expect_err("invalid pattern");
        assert!(error.to_string().contains("invalid safety rule pattern"));
    }

    #[test]
    fn unrenderable_skill_propagates_the_error() {
        let skill = SkillDefinition::new("demo", "", "body");
        let error = scan_safety(&skill, &[], 0.8).expect_err("blank description");
        assert!(error.to_string().contains("description"));
    }

    #[test]
    fn aggregate_completes_with_both_contributions() {
        let design = report_with(80.0, true);
        let functional = functional_with(0.5);
        let safety = safety_with(false);
        let final_score = aggregate(&design, &functional, &safety, 0.4, 0.6);

        assert_eq!(final_score.pipeline_status, "COMPLETED");
        assert!((final_score.final_score_pct - 62.0).abs() < 1e-9);
        assert_eq!(final_score.grade, "D");
        assert_eq!(
            final_score.gates_passed,
            vec!["e1_llm_judge", "e2_functional", "e3_safety"]
        );
        assert_eq!(final_score.recommendation, "usable, refine failing cases");
        assert_eq!(final_score.skill_name, "demo");

        let judge = final_score.evaluators.e1_llm_judge.expect("judge score");
        assert!((judge.score_pct - 80.0).abs() < 1e-9);
        assert!((judge.weight - 0.4).abs() < 1e-9);
        assert!((judge.contribution - 32.0).abs() < 1e-9);

        let functional_score = final_score.evaluators.e2_functional;
        assert!((functional_score.score_pct - 50.0).abs() < 1e-9);
        assert!((functional_score.weight - 0.6).abs() < 1e-9);
        assert!((functional_score.contribution - 30.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_stops_at_safety() {
        let design = report_with(90.0, true);
        let functional = functional_with(1.0);
        let final_score = aggregate(&design, &functional, &safety_with(true), 0.4, 0.6);

        assert_eq!(final_score.pipeline_status, "BLOCKED_AT_SAFETY");
        assert!((final_score.final_score_pct - 0.0).abs() < 1e-9);
        assert_eq!(final_score.grade, "F");
        assert_eq!(final_score.gates_passed.len(), 0);
        assert_eq!(final_score.recommendation, "not ready for release");
    }

    #[test]
    fn aggregate_stops_at_design() {
        let design = report_with(40.0, false);
        let functional = functional_with(1.0);
        let final_score = aggregate(&design, &functional, &safety_with(false), 0.4, 0.6);

        assert_eq!(final_score.pipeline_status, "BLOCKED_AT_E1_LLM_JUDGE");
        assert!((final_score.final_score_pct - 0.0).abs() < 1e-9);
        assert_eq!(final_score.grade, "F");
        assert_eq!(final_score.gates_passed, vec!["e3_safety"]);
        assert_eq!(final_score.recommendation, "not ready for release");
    }

    #[test]
    fn aggregate_recommendation_boundaries() {
        let high = aggregate(
            &report_with(90.0, true),
            &functional_with(1.0),
            &safety_with(false),
            0.4,
            0.6,
        );
        assert!((high.final_score_pct - 96.0).abs() < 1e-9);
        assert_eq!(high.grade, "A");
        assert_eq!(high.recommendation, "ready for release");

        let low = aggregate(
            &report_with(55.0, true),
            &functional_with(0.5),
            &safety_with(false),
            0.4,
            0.6,
        );
        assert!((low.final_score_pct - 52.0).abs() < 1e-9);
        assert_eq!(low.recommendation, "not ready for release");
    }

    #[test]
    fn round2_rounds_half_up_to_cents() {
        assert!((round2(62.004) - 62.0).abs() < 1e-9);
        assert!((round2(62.006) - 62.01).abs() < 1e-9);
        assert!((round2(0.0) - 0.0).abs() < 1e-9);
    }
}
