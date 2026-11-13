//! Unit tests for the deterministic scorers.
//!
//! Kept beside `scoring.rs` (rather than inline) because the rubric, safety-scan,
//! and aggregation matrices together outgrow the implementation file.

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

/// A design report from points (0–100), with the fraction derived from it.
fn report_with(points: f64, ready: bool) -> DesignReport {
    let fraction = fraction_from_points(points);
    DesignReport {
        skill_name: "demo".to_string(),
        design_score: points,
        design_score_max: 100.0,
        design_score_pct: fraction,
        grade: grade_for(fraction),
        knowledge_delta: round2(points / 100.0),
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
        score_pct: if blocked { 0.5 } else { 1.0 },
        gate_threshold: 0.8,
        blocked,
        ..SafetyReport::default()
    }
}

/// A safety report whose gate decision follows the `0.80` threshold.
fn safety_at(score_pct: f64) -> SafetyReport {
    SafetyReport {
        skill_name: "demo".to_string(),
        score_pct,
        gate_threshold: 0.8,
        blocked: score_pct < 0.8,
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
    assert!((report.design_score_pct - 1.0).abs() < 1e-9);
    assert!((report.design_score_max - 100.0).abs() < 1e-9);
    assert_eq!(report.grade, "A");
    assert!((report.knowledge_delta - 1.0).abs() < 1e-9);
    assert_eq!(report.knowledge_delta_label(), "Excellent");
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
    assert!(
        report.design_score_pct < DESIGN_GATE_THRESHOLD,
        "{report:?}"
    );
    assert!(!report.ready_for_functional_testing);
    assert_eq!(report.grade, "F");
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
fn unit_helpers_convert_between_points_and_fractions() {
    assert!((fraction_from_points(74.0) - 0.74).abs() < 1e-9);
    assert!((fraction_from_points(100.0) - 1.0).abs() < 1e-9);
    assert!((points_from_fraction(0.74) - 74.0).abs() < 1e-9);
    assert!((round4(0.4 * 0.74) - 0.296).abs() < 1e-9);
    assert!((round4(2.0 / 3.0) - 0.6667).abs() < 1e-9);
    assert!((round4(0.0) - 0.0).abs() < 1e-9);
}

#[test]
fn safety_without_rules_is_clean() {
    let report = scan_safety(&rich_skill(), &[], 0.8).expect("scan");
    assert!((report.score_pct - 1.0).abs() < 1e-9);
    assert!(!report.blocked);
    assert!(report.auto_fail.is_none());
    assert!(report.passes_gate());
    assert_eq!(report.findings.len(), 0);
    assert!((report.gate_threshold - 0.8).abs() < 1e-9);
    assert_eq!(report.skill_name, "demo");

    let dimensions = report.dimensions.as_ref().expect("dimensions");
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
    assert!((report.score_pct - 0.75).abs() < 1e-9);
    assert!(!report.blocked);
    assert!((report.gate_threshold - 0.5).abs() < 1e-9);
    assert_eq!(report.findings[0].dimension, "content");
    assert_eq!(report.findings[0].severity, "warning");

    let dimensions = report.dimensions.as_ref().expect("dimensions");
    assert!((dimensions["d1_harmful_instruction"].score - 22.5).abs() < 1e-9);
    assert!((dimensions["d5_guardrails"].score - 7.5).abs() < 1e-9);
}

#[test]
fn critical_rule_blocks_unconditionally() {
    let rules = [SafetyRule::new("checklist", "secrets", "critical", "leak")];
    let report = scan_safety(&rich_skill(), &rules, 0.0).expect("scan");
    assert!(report.blocked);
    assert!(report.auto_fail.is_none());
    assert!(!report.passes_gate());
    assert!((report.score_pct - 0.75).abs() < 1e-9);
}

#[test]
fn two_matching_rules_halve_the_score() {
    let rules = [
        SafetyRule::new("checklist", "a", "info", "one"),
        SafetyRule::new("(?i)SUMMARY", "b", "warning", "two"),
    ];
    let report = scan_safety(&rich_skill(), &rules, 0.8).expect("scan");
    assert_eq!(report.findings.len(), 2);
    assert!((report.score_pct - 0.5).abs() < 1e-9);
    assert!(report.blocked);
}

#[test]
fn non_matching_rule_leaves_the_score_intact() {
    let rules = [SafetyRule::new("does-not-appear", "a", "warning", "unused")];
    let report = scan_safety(&rich_skill(), &rules, 0.8).expect("scan");
    assert_eq!(report.findings.len(), 0);
    assert!((report.score_pct - 1.0).abs() < 1e-9);
    assert!(!report.blocked);
}

#[test]
fn an_auto_fail_rule_zeroes_and_blocks_the_scan() {
    let rules = [
        SafetyRule::new("checklist", "d3_data_handling", "info", "leaks")
            .with_auto_fail(AutoFailCode::UndeclaredExfiltration),
    ];
    let report = scan_safety(&rich_skill(), &rules, 0.0).expect("scan");
    assert_eq!(report.score_pct, 0.0);
    assert!(report.blocked);
    assert_eq!(report.auto_fail, Some(AutoFailCode::UndeclaredExfiltration));
    assert!(report.dimensions.is_none());
    assert_eq!(report.findings.len(), 1);
    // An auto-fail blocks even with a zero threshold.
    assert!(!report.passes_gate());
}

#[test]
fn the_first_auto_fail_rule_wins_and_non_matching_rules_are_ignored() {
    let rules = [
        SafetyRule::new("checklist", "a", "warning", "one"),
        SafetyRule::new("does-not-appear", "b", "warning", "two")
            .with_auto_fail(AutoFailCode::BypassSafety),
        SafetyRule::new("(?i)SUMMARY", "c", "warning", "three")
            .with_auto_fail(AutoFailCode::HardcodedCredentials),
    ];
    let report = scan_safety(&rich_skill(), &rules, 0.8).expect("scan");
    assert_eq!(report.auto_fail, Some(AutoFailCode::HardcodedCredentials));
    assert_eq!(report.findings.len(), 2);
    assert_eq!(report.score_pct, 0.0);
}

#[test]
fn an_auto_fail_rule_and_a_critical_rule_still_report_the_code() {
    let rules = [
        SafetyRule::new("checklist", "a", "critical", "critical hit"),
        SafetyRule::new("checklist", "b", "info", "auto fail")
            .with_auto_fail(AutoFailCode::InstructingDeception),
    ];
    let report = scan_safety(&rich_skill(), &rules, 0.8).expect("scan");
    assert_eq!(report.auto_fail, Some(AutoFailCode::InstructingDeception));
    assert!(report.blocked);
    assert_eq!(report.findings.len(), 2);
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
    let final_score = aggregate(&design, &functional, &safety_with(false), 0.4, 0.6);

    assert_eq!(final_score.pipeline_status, PIPELINE_COMPLETED);
    assert!((final_score.final_score_pct - 0.62).abs() < 1e-9);
    assert!((final_score.score_fraction() - 0.62).abs() < 1e-9);
    assert!(final_score.is_completed());
    assert_eq!(final_score.grade, "D");
    assert_eq!(final_score.gates_passed, vec![GATE_E3]);
    assert_eq!(final_score.recommendation, "usable, refine failing cases");
    assert_eq!(final_score.skill_name, "demo");

    let judge = final_score.evaluators.e1_llm_judge.expect("judge score");
    assert!((judge.score_pct - 0.8).abs() < 1e-9);
    assert!((judge.weight - 0.4).abs() < 1e-9);
    assert!((judge.contribution - 0.32).abs() < 1e-9);

    let functional_score = final_score.evaluators.e2_functional;
    assert!((functional_score.score_pct - 0.5).abs() < 1e-9);
    assert!((functional_score.weight - 0.6).abs() < 1e-9);
    assert!((functional_score.contribution - 0.3).abs() < 1e-9);
}

#[test]
fn aggregate_matches_the_python_worked_example() {
    // 74 points of design and an 87% pass rate: 0.40*0.74 + 0.60*0.87 = 0.818.
    let design = report_with(74.0, true);
    let functional = functional_with(0.87);
    let final_score = aggregate(&design, &functional, &safety_with(false), 0.4, 0.6);

    assert_eq!(final_score.pipeline_status, PIPELINE_COMPLETED);
    assert!((final_score.final_score_pct - 0.818).abs() < 1e-9);
    assert_eq!(final_score.grade, "B");
    assert_eq!(final_score.recommendation, "ready for release");

    let judge = final_score.evaluators.e1_llm_judge.expect("judge score");
    assert!((judge.score_pct - 0.74).abs() < 1e-9);
    assert!((judge.contribution - 0.296).abs() < 1e-9);
    let functional_score = final_score.evaluators.e2_functional;
    assert!((functional_score.score_pct - 0.87).abs() < 1e-9);
    assert!((functional_score.contribution - 0.522).abs() < 1e-9);
}

#[test]
fn aggregate_stops_at_safety() {
    let design = report_with(90.0, true);
    let functional = functional_with(1.0);
    let final_score = aggregate(&design, &functional, &safety_with(true), 0.4, 0.6);

    assert_eq!(final_score.pipeline_status, PIPELINE_BLOCKED_AT_E3);
    assert!((final_score.final_score_pct - 0.0).abs() < 1e-9);
    assert_eq!(final_score.grade, "F");
    assert_eq!(final_score.gates_passed, Vec::<String>::new());
    assert!(!final_score.is_completed());
    assert!(
        final_score.recommendation.contains(GATE_E3),
        "{final_score:?}"
    );
    // The contributions are still filled in for the author's benefit.
    let judge = final_score.evaluators.e1_llm_judge.expect("judge score");
    assert!((judge.score_pct - 0.9).abs() < 1e-9);
    assert!((judge.contribution - 0.36).abs() < 1e-9);
}

#[test]
fn aggregate_stops_at_the_design_gate() {
    let design = report_with(40.0, false);
    let functional = functional_with(1.0);
    let final_score = aggregate(&design, &functional, &safety_with(false), 0.4, 0.6);

    assert_eq!(final_score.pipeline_status, PIPELINE_BLOCKED_AT_E1);
    assert!((final_score.final_score_pct - 0.0).abs() < 1e-9);
    assert_eq!(final_score.grade, "F");
    assert_eq!(final_score.gates_passed, Vec::<String>::new());
    assert!(
        final_score.recommendation.contains(GATE_E1),
        "{final_score:?}"
    );
}

#[test]
fn the_design_gate_boundary_is_sixty_percent() {
    for (points, status) in [
        (60.0, PIPELINE_COMPLETED),
        (61.0, PIPELINE_COMPLETED),
        (59.0, PIPELINE_BLOCKED_AT_E1),
    ] {
        let final_score = aggregate(
            &report_with(points, true),
            &functional_with(1.0),
            &safety_with(false),
            0.4,
            0.6,
        );
        assert_eq!(final_score.pipeline_status, status, "{points} points");
    }
}

#[test]
fn the_safety_gate_boundary_is_eighty_percent() {
    let design = report_with(80.0, true);
    let functional = functional_with(1.0);

    let open = safety_at(0.80);
    assert!(!open.blocked);
    assert_eq!(
        aggregate(&design, &functional, &open, 0.4, 0.6).pipeline_status,
        PIPELINE_COMPLETED
    );

    let closed = safety_at(0.79);
    assert!(closed.blocked);
    let blocked = aggregate(&design, &functional, &closed, 0.4, 0.6);
    assert_eq!(blocked.pipeline_status, PIPELINE_BLOCKED_AT_E3);
    assert_eq!(blocked.gates_passed, Vec::<String>::new());
}

#[test]
fn the_design_gate_is_checked_before_the_safety_gate() {
    let final_score = aggregate(
        &report_with(40.0, false),
        &functional_with(1.0),
        &safety_with(true),
        0.4,
        0.6,
    );
    assert_eq!(final_score.pipeline_status, PIPELINE_BLOCKED_AT_E1);
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
    assert!((high.final_score_pct - 0.96).abs() < 1e-9);
    assert_eq!(high.grade, "A");
    assert_eq!(high.recommendation, "ready for release");

    let middle = aggregate(
        &report_with(60.0, true),
        &functional_with(0.6),
        &safety_with(false),
        0.4,
        0.6,
    );
    assert!((middle.final_score_pct - 0.6).abs() < 1e-9);
    assert_eq!(middle.recommendation, "usable, refine failing cases");

    let low = aggregate(
        &report_with(60.0, true),
        &functional_with(0.5),
        &safety_with(false),
        0.4,
        0.6,
    );
    assert!((low.final_score_pct - 0.54).abs() < 1e-9);
    assert_eq!(low.grade, "F");
    assert_eq!(low.recommendation, "not ready for release");
}

#[test]
fn round2_rounds_half_up_to_cents() {
    assert!((round2(62.004) - 62.0).abs() < 1e-9);
    assert!((round2(62.006) - 62.01).abs() < 1e-9);
    assert!((round2(0.0) - 0.0).abs() < 1e-9);
}
