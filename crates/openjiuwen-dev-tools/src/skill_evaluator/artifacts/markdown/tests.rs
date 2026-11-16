//! Renderer tests for the four Markdown reports.
//!
//! Every fixture is a report value: rendering is a pure function, so the whole
//! document (or the whole section that matters) is pinned verbatim. Shared
//! hygiene is asserted through [`assert_clean_markdown`] — no trailing
//! whitespace, single blank lines between sections, exactly one final newline.

use std::collections::BTreeMap;

use super::*;
use crate::skill_evaluator::report::{
    AutoFailCode, CaseSummary, EvaluatorScore, EvaluatorScores, FinalScore, FunctionalReport,
    SafetyFinding, ScoreDimension,
};

/// A design review whose rubric awards 74 of 100 points.
fn design_report() -> DesignReport {
    let awarded = [
        ("d1_knowledge_ratio", 7.0, 10.0),
        ("d2_expert_knowledge_craft", 20.0, 25.0),
        ("d3_spec_compliance", 15.0, 15.0),
        ("d4_progressive_disclosure", 5.0, 10.0),
        ("d5_pattern_freedom_fit", 12.0, 15.0),
        ("d6_practical_usability", 10.0, 15.0),
        ("d7_output_specification", 5.0, 10.0),
    ];
    let dimensions = awarded
        .into_iter()
        .map(|(id, score, max)| (id.to_owned(), ScoreDimension { score, max }))
        .collect();
    DesignReport {
        skill_name: "demo".to_owned(),
        design_score: 74.0,
        design_score_max: 100.0,
        design_score_pct: 0.74,
        grade: grade_for(0.74),
        knowledge_delta: 0.74,
        ready_for_functional_testing: true,
        dimensions,
    }
}

/// The seven canonical design dimensions filled in from a per-dimension score.
fn design_with(scores: &[(&str, f64, f64)]) -> DesignReport {
    let mut report = design_report();
    report.dimensions = scores
        .iter()
        .map(|&(id, score, max)| (id.to_owned(), ScoreDimension { score, max }))
        .collect();
    report
}

/// A clean safety scan with one warning finding.
fn safety_report() -> SafetyReport {
    let mut dimensions = BTreeMap::new();
    for (id, score, max) in [
        ("d1_harmful_instruction", 26.0, 30.0),
        ("d2_scope_containment", 25.0, 25.0),
        ("d3_data_handling", 16.0, 20.0),
        ("d4_injection_resilience", 15.0, 15.0),
        ("d5_guardrails", 10.0, 10.0),
    ] {
        dimensions.insert(id.to_owned(), ScoreDimension { score, max });
    }
    SafetyReport {
        skill_name: "demo".to_owned(),
        score_pct: 0.83,
        gate_threshold: 0.8,
        blocked: false,
        auto_fail: None,
        dimensions: Some(dimensions),
        findings: vec![SafetyFinding {
            dimension: "d1_harmful_instruction".to_owned(),
            severity: "warning".to_owned(),
            message: "the skill body mentions bypassing the sandbox".to_owned(),
        }],
    }
}

/// An auto-failed safety scan: no dimension scores, one critical finding.
fn auto_fail_report() -> SafetyReport {
    SafetyReport {
        skill_name: "demo".to_owned(),
        score_pct: 0.0,
        gate_threshold: 0.8,
        blocked: true,
        auto_fail: Some(AutoFailCode::HardcodedCredentials),
        dimensions: None,
        findings: vec![SafetyFinding {
            dimension: "d3_data_handling".to_owned(),
            severity: "critical".to_owned(),
            message: "the skill body ships an API key".to_owned(),
        }],
    }
}

/// A completed run carrying one value in every report member.
fn completed_report() -> EvaluationReport {
    EvaluationReport {
        design: design_report(),
        functional: FunctionalReport {
            skill_name: "demo".to_owned(),
            skill_path: "skills/demo".to_owned(),
            timestamp: 1_700_000_000_000,
            summary: CaseSummary {
                total: 4,
                passed: 3,
                partial: 1,
                failed: 0,
                pass_rate: 0.75,
            },
            results: Vec::new(),
            recommendations: vec!["Edge cases".to_owned()],
        },
        safety: safety_report(),
        final_score: FinalScore {
            skill_name: "demo".to_owned(),
            pipeline_status: crate::skill_evaluator::report::PIPELINE_COMPLETED.to_owned(),
            final_score_pct: 0.818,
            grade: grade_for(0.818),
            evaluators: EvaluatorScores {
                e1_llm_judge: Some(EvaluatorScore::weighted(0.74, 0.4)),
                e2_functional: EvaluatorScore::weighted(0.87, 0.6),
            },
            gates_passed: vec!["e3_safety".to_owned()],
            recommendation: "ready for release".to_owned(),
        },
        judge: None,
    }
}

/// The completed fixture with its safety gate closed (blocked at E3).
fn blocked_report() -> EvaluationReport {
    let mut report = completed_report();
    report.safety = auto_fail_report();
    report.final_score.pipeline_status =
        crate::skill_evaluator::report::PIPELINE_BLOCKED_AT_E3.to_owned();
    report.final_score.final_score_pct = 0.0;
    report.final_score.grade = grade_for(0.0);
    report.final_score.gates_passed = Vec::new();
    report.final_score.recommendation = "fix the hardcoded credential".to_owned();
    report
}

/// Asserts the formatting invariants every rendered document must keep.
fn assert_clean_markdown(document: &str) {
    assert!(
        document.ends_with('\n'),
        "the document must end with one newline"
    );
    assert!(
        !document.ends_with("\n\n"),
        "the document must not end with a blank line"
    );
    assert!(
        !document.contains("\n\n\n"),
        "sections must be separated by exactly one blank line:\n{document}"
    );
    for (index, line) in document.lines().enumerate() {
        assert_eq!(
            line.trim_end(),
            line,
            "line {} carries trailing whitespace: {line:?}",
            index + 1
        );
    }
}

#[test]
fn llm_judge_report_renders_the_python_document() {
    let document = llm_judge_report(&design_report());

    let expected = "\
# Skill Evaluation Report: demo

## Summary
- **Total Score**: 74/100
- **Grade**: C
- **Knowledge Delta**: Good
- **Ready for functional testing**: Yes

## Dimension Scores
| Dimension | Score | Max |
| --- | --- | --- |
| D1: Knowledge Ratio | 7 | 10 |
| D2: Expert Knowledge Craft | 20 | 25 |
| D3: Specification Compliance | 15 | 15 |
| D4: Progressive Disclosure | 5 | 10 |
| D5: Pattern + Freedom Fit | 12 | 15 |
| D6: Practical Usability | 10 | 15 |
| D7: Output Specification | 5 | 10 |

## Critical Issues
None.

## Detailed Analysis
- D1: Knowledge Ratio: 7/10 points (70%).
- D4: Progressive Disclosure: 5/10 points (50%).
- D6: Practical Usability: 10/15 points (66.67%).
- D7: Output Specification: 5/10 points (50%).
";
    assert_eq!(document, expected);
    assert_clean_markdown(&document);
}

#[test]
fn the_design_report_humanises_all_seven_dimension_ids() {
    let document = llm_judge_report(&design_report());

    for label in [
        "D1: Knowledge Ratio",
        "D2: Expert Knowledge Craft",
        "D3: Specification Compliance",
        "D4: Progressive Disclosure",
        "D5: Pattern + Freedom Fit",
        "D6: Practical Usability",
        "D7: Output Specification",
    ] {
        assert!(document.contains(label), "missing dimension row {label}");
    }
    for id in DESIGN_DIMENSIONS.iter().map(|(id, _, _)| *id) {
        assert!(
            !document.contains(id),
            "the raw dimension id {id} must not leak into the report"
        );
    }
}

#[test]
fn a_closed_design_gate_is_a_critical_issue_and_readiness_reads_no() {
    let mut design = design_with(&[
        ("d1_knowledge_ratio", 5.0, 10.0),
        ("d2_expert_knowledge_craft", 10.0, 25.0),
        ("d3_spec_compliance", 0.0, 15.0),
    ]);
    design.design_score = 15.0;
    design.design_score_pct = 0.15;
    design.grade = grade_for(0.15);
    design.ready_for_functional_testing = false;

    let document = llm_judge_report(&design);

    assert!(
        document.contains("- **Ready for functional testing**: No"),
        "{document}"
    );
    assert!(
        document.contains(
            "- Design gate closed: 15 of 100 points is below the 60-point readiness threshold."
        ),
        "{document}"
    );
    assert!(
        document.contains("- D3: Specification Compliance: 0/15 points — no rubric credit."),
        "{document}"
    );
    assert_clean_markdown(&document);
}

#[test]
fn dimensions_the_report_omits_render_a_dash_with_the_canonical_maximum() {
    let design = design_with(&[]);

    let document = llm_judge_report(&design);

    assert!(
        document.contains("| D1: Knowledge Ratio | - | 10 |"),
        "{document}"
    );
    assert!(
        document.contains("| D2: Expert Knowledge Craft | - | 25 |"),
        "{document}"
    );
    assert!(
        document.contains("## Detailed Analysis\nNone.\n"),
        "{document}"
    );
    assert_clean_markdown(&document);
}

#[test]
fn skill_safety_report_renders_the_python_document() {
    let document = skill_safety_report(&safety_report());

    let expected = "\
# Safety Evaluation Report: demo

## Verdict
- **Score**: 83/100 (83%)
- **Gate**: PASS
- **Blocked**: No
- **Auto-fail triggered**: None

## Dimension Scores
| Dimension | Score | Max |
| --- | --- | --- |
| D1: Harmful Instruction Potential | 26 | 30 |
| D2: Scope Containment | 25 | 25 |
| D3: Data Handling Safety | 16 | 20 |
| D4: Injection Resilience | 15 | 15 |
| D5: Guardrail Presence | 10 | 10 |

## Findings
- D1: Harmful Instruction Potential (warning): the skill body mentions bypassing the sandbox

## Cleared Dimensions
- D2: Scope Containment (25/25)
- D4: Injection Resilience (15/15)
- D5: Guardrail Presence (10/10)
";
    assert_eq!(document, expected);
    assert_clean_markdown(&document);
}

#[test]
fn an_auto_fail_renders_the_code_and_zeroes_the_table() {
    let document = skill_safety_report(&auto_fail_report());

    assert!(
        document.starts_with("# Safety Evaluation Report: demo\n"),
        "{document}"
    );
    assert!(document.contains("- **Score**: 0/100 (0%)"), "{document}");
    assert!(document.contains("- **Gate**: FAIL"), "{document}");
    assert!(document.contains("- **Blocked**: Yes"), "{document}");
    assert!(
        document.contains(
            "- **Auto-fail triggered**: AF2: hardcoded credentials, API keys, or secrets"
        ),
        "{document}"
    );
    for row in [
        "| D1: Harmful Instruction Potential | - | 30 |",
        "| D2: Scope Containment | - | 25 |",
        "| D3: Data Handling Safety | - | 20 |",
        "| D4: Injection Resilience | - | 15 |",
        "| D5: Guardrail Presence | - | 10 |",
    ] {
        assert!(document.contains(row), "missing row {row}");
    }
    assert!(
        document.contains("- D3: Data Handling Safety (critical): the skill body ships an API key"),
        "{document}"
    );
    assert!(
        document.contains("## Cleared Dimensions\nNone.\n"),
        "{document}"
    );
    assert_clean_markdown(&document);
}

#[test]
fn a_clean_safety_scan_reports_the_empty_findings_state() {
    let mut safety = safety_report();
    safety.findings.clear();

    let document = skill_safety_report(&safety);

    assert!(document.contains("## Findings\nNone.\n"), "{document}");
    assert!(
        document.contains("## Cleared Dimensions\n- D2: Scope Containment (25/25)\n"),
        "{document}"
    );
    assert_clean_markdown(&document);
}

#[test]
fn a_finding_dimension_outside_the_rubric_keeps_its_raw_id() {
    let mut safety = safety_report();
    safety.findings = vec![SafetyFinding {
        dimension: "data_leakage".to_owned(),
        severity: "critical".to_owned(),
        message: "the skill body leaks the internal codename".to_owned(),
    }];

    let document = skill_safety_report(&safety);

    assert!(
        document.contains("- data_leakage (critical): the skill body leaks the internal codename"),
        "{document}"
    );
}

#[test]
fn final_report_renders_the_python_document_for_a_completed_run() {
    let document = final_report(&completed_report());

    let expected = "\
# Eval Pipeline Report: demo

## Final Verdict
- **Score**: 0.818 (B)
- **Status**: COMPLETED
- **Ship-ready**: Yes — ready for release

## Evaluator Breakdown
| Evaluator | Score | Weight | Contribution |
| --- | --- | --- | --- |
| E1 Design (skill_llm_judge) | 74% | 0.40 | 0.296 |
| E2 Functional (skill_tester) | 87% | 0.60 | 0.522 |
| E3 Safety (skill_safety_judge) | PASS | gate | - |

## Key Issues to Fix
- Functional: Edge cases
- Design: D1: Knowledge Ratio — 7/10 points (70%)
- Design: D4: Progressive Disclosure — 5/10 points (50%)
- Design: D6: Practical Usability — 10/15 points (66.67%)
- Design: D7: Output Specification — 5/10 points (50%)
- Safety: D1: Harmful Instruction Potential (warning): the skill body mentions bypassing the sandbox

## Full Reports
- [llm_judge_score.json](llm_judge_score.json)
- [skill_safety_score.json](skill_safety_score.json)
- [skill_tester_report.json](skill_tester_report.json)
- [final_score.json](final_score.json)

**Run timestamp**: 2023-11-14T22:13:20Z
";
    assert_eq!(document, expected);
    assert_clean_markdown(&document);
}

#[test]
fn a_completed_run_that_is_not_release_ready_reads_no() {
    let mut report = completed_report();
    report.final_score.final_score_pct = 0.62;
    report.final_score.grade = grade_for(0.62);
    report.final_score.recommendation = "usable, refine failing cases".to_owned();

    let document = final_report(&report);

    assert!(document.contains("- **Score**: 0.62 (D)"), "{document}");
    assert!(
        document.contains("- **Ship-ready**: No — usable, refine failing cases"),
        "{document}"
    );
}

#[test]
fn a_missing_design_evaluator_renders_a_dash_row() {
    let mut report = completed_report();
    report.final_score.evaluators.e1_llm_judge = None;

    let document = final_report(&report);

    assert!(
        document.contains("| E1 Design (skill_llm_judge) | - | - | - |"),
        "{document}"
    );
}

#[test]
fn pipeline_report_renders_the_python_document_for_a_blocked_run() {
    let document = pipeline_report(&blocked_report());

    let expected = "\
# Eval Pipeline Report: demo

## Final Verdict
- **Status**: BLOCKED_AT_E3
- **Blocked by**: E3 safety gate (safety score below its gate threshold or a critical finding)
- **Design score**: 74/100 (74%)
- **Safety score**: 0/100 (0%), gate FAIL

## Key Issues to Fix
- Gate: stopped at BLOCKED_AT_E3 — E3 safety gate (safety score below its gate threshold or a critical finding).
- Functional: Edge cases
- Design: D1: Knowledge Ratio — 7/10 points (70%)
- Design: D4: Progressive Disclosure — 5/10 points (50%)
- Design: D6: Practical Usability — 10/15 points (66.67%)
- Design: D7: Output Specification — 5/10 points (50%)
- Safety: D3: Data Handling Safety (critical): the skill body ships an API key

## Full Reports
- [llm_judge_score.json](llm_judge_score.json)
- [skill_safety_score.json](skill_safety_score.json)
- [skill_tester_report.json](skill_tester_report.json)

**Run timestamp**: 2023-11-14T22:13:20Z
";
    assert_eq!(document, expected);
    assert_clean_markdown(&document);
}

#[test]
fn a_design_gate_block_names_the_e1_gate() {
    let mut report = blocked_report();
    report.final_score.pipeline_status =
        crate::skill_evaluator::report::PIPELINE_BLOCKED_AT_E1.to_owned();
    report.design.design_score_pct = 0.4;
    report.design.design_score = 40.0;
    report.design.ready_for_functional_testing = false;
    report.safety = safety_report();

    let document = pipeline_report(&report);

    assert!(
        document.contains("- **Status**: BLOCKED_AT_E1"),
        "{document}"
    );
    assert!(
        document.contains("- **Blocked by**: E1 design gate (design score below 60 points)"),
        "{document}"
    );
    assert!(
        document.contains("- **Design score**: 40/100 (40%)"),
        "{document}"
    );
    assert!(
        document.contains("- **Safety score**: 83/100 (83%), gate PASS"),
        "{document}"
    );
    assert!(
        document.contains(
            "- Gate: stopped at BLOCKED_AT_E1 — E1 design gate (design score below 60 points)."
        ),
        "{document}"
    );
}

#[test]
fn rendered_numbers_are_trimmed_of_trailing_zeros() {
    assert_eq!(format_percent(0.875), "87.5%");
    assert_eq!(format_percent(1.0), "100%");
    assert_eq!(format_percent(0.8), "80%");
    assert_eq!(format_percent(0.0), "0%");
    assert_eq!(format_fraction(0.5), "0.5");
    assert_eq!(format_fraction(0.818), "0.818");
    assert_eq!(format_fraction(0.296), "0.296");
    assert_eq!(format_points(7.0), "7");
    assert_eq!(format_points(12.5), "12.5");
    assert_eq!(format_points(66.666), "66.67");
    assert_eq!(format_weight(0.4), "0.40");
    assert_eq!(format_ratio_percent(1.0, 0.0), "-");
}

#[test]
fn a_blank_skill_name_and_grade_fall_back_without_panicking() {
    let mut design = design_report();
    design.skill_name = "   ".to_owned();
    design.grade = String::new();

    let document = llm_judge_report(&design);

    assert!(
        document.starts_with("# Skill Evaluation Report:\n"),
        "{document}"
    );
    assert!(document.contains("- **Grade**: C"), "{document}");
    assert_clean_markdown(&document);
}

#[test]
fn a_report_without_a_name_borrows_one_from_another_member() {
    let mut report = completed_report();
    report.final_score.skill_name = String::new();

    let document = final_report(&report);

    assert!(
        document.starts_with("# Eval Pipeline Report: demo\n"),
        "{document}"
    );
}

#[test]
fn documents_carry_only_the_python_headings() {
    let documents = [
        ("design", llm_judge_report(&design_report())),
        ("safety", skill_safety_report(&safety_report())),
        ("final", final_report(&completed_report())),
        ("pipeline", pipeline_report(&blocked_report())),
    ];
    let expected = [
        (
            "design",
            vec![
                "# Skill Evaluation Report: demo",
                "## Summary",
                "## Dimension Scores",
                "## Critical Issues",
                "## Detailed Analysis",
            ],
        ),
        (
            "safety",
            vec![
                "# Safety Evaluation Report: demo",
                "## Verdict",
                "## Dimension Scores",
                "## Findings",
                "## Cleared Dimensions",
            ],
        ),
        (
            "final",
            vec![
                "# Eval Pipeline Report: demo",
                "## Final Verdict",
                "## Evaluator Breakdown",
                "## Key Issues to Fix",
                "## Full Reports",
            ],
        ),
        (
            "pipeline",
            vec![
                "# Eval Pipeline Report: demo",
                "## Final Verdict",
                "## Key Issues to Fix",
                "## Full Reports",
            ],
        ),
    ];

    for ((name, document), (expected_name, headings)) in documents.iter().zip(expected) {
        assert_eq!(*name, expected_name);
        let found: Vec<&str> = document
            .lines()
            .filter(|line| line.starts_with('#'))
            .collect();
        assert_eq!(found, headings, "unexpected headings in the {name} report");
        assert_clean_markdown(document);
    }
}

#[test]
fn every_section_uses_the_empty_state_for_an_empty_report() {
    let design = DesignReport::default();
    let safety = SafetyReport::default();
    let report = EvaluationReport::default();

    let design_document = llm_judge_report(&design);
    assert!(
        design_document.contains(
            "## Critical Issues\n- Design gate closed: 0 of 100 points is below the 60-point readiness threshold.\n"
        ),
        "{design_document}"
    );
    assert!(
        design_document.contains("## Detailed Analysis\nNone.\n"),
        "{design_document}"
    );

    let ready = DesignReport {
        ready_for_functional_testing: true,
        ..DesignReport::default()
    };
    assert!(
        llm_judge_report(&ready).contains("## Critical Issues\nNone.\n"),
        "a design report with an open gate and no zero-credit dimension has no critical issue"
    );

    let safety_document = skill_safety_report(&safety);
    assert!(
        safety_document.contains("## Findings\nNone.\n"),
        "{safety_document}"
    );
    assert!(
        safety_document.contains("## Cleared Dimensions\nNone.\n"),
        "{safety_document}"
    );

    let clean_completed = EvaluationReport {
        final_score: FinalScore {
            pipeline_status: crate::skill_evaluator::report::PIPELINE_COMPLETED.to_owned(),
            ..FinalScore::default()
        },
        ..EvaluationReport::default()
    };
    assert!(
        final_report(&clean_completed).contains("## Key Issues to Fix\nNone.\n"),
        "a clean completed run has no issue to list"
    );

    let pipeline_document = pipeline_report(&report);
    assert!(
        pipeline_document.contains("- **Blocked by**: pipeline gate"),
        "{pipeline_document}"
    );

    for document in [
        design_document,
        safety_document,
        final_report(&report),
        pipeline_document,
    ] {
        assert!(!document.contains("None.None."), "{document}");
        assert_clean_markdown(&document);
    }
}
