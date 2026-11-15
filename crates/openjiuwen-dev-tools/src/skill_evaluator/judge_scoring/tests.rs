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
    let skill_judge = SkillJudge::new(Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")])));
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
async fn a_raised_gate_blocks_a_judged_score_the_default_lets_through() {
    let blocked = judge(&fenced_verdict(0.875, true))
        .safety_with_threshold(&skill(), &[], 0.90)
        .await
        .expect("safety");
    assert!((blocked.report.score_pct - 0.875).abs() < 1e-9);
    assert!(blocked.report.blocked);
    assert!((blocked.report.gate_threshold - 0.9).abs() < 1e-9);

    let lowered = judge(&fenced_verdict(0.875, true))
        .safety_with_threshold(&skill(), &[], 0.70)
        .await
        .expect("safety");
    assert!((lowered.report.score_pct - 0.875).abs() < 1e-9);
    assert!(!lowered.report.blocked);
    assert!((lowered.report.gate_threshold - 0.7).abs() < 1e-9);

    let default = judge(&fenced_verdict(0.875, true))
        .safety(&skill(), &[])
        .await
        .expect("safety");
    assert!((default.report.score_pct - 0.875).abs() < 1e-9);
    assert!(!default.report.blocked);
    assert!((default.report.gate_threshold - 0.8).abs() < 1e-9);
}

#[tokio::test]
async fn a_non_critical_rule_hit_follows_the_threshold() {
    let rules = [warning_rule()];

    let default = judge(&fenced_verdict(1.0, true))
        .safety(&skill(), &rules)
        .await
        .expect("safety");
    assert!((default.report.score_pct - 0.75).abs() < 1e-9);
    assert!(default.report.blocked);
    assert!((default.report.gate_threshold - 0.8).abs() < 1e-9);

    let lowered = judge(&fenced_verdict(1.0, true))
        .safety_with_threshold(&skill(), &rules, 0.70)
        .await
        .expect("safety");
    assert!((lowered.report.score_pct - 0.75).abs() < 1e-9);
    assert!(!lowered.report.blocked);
    assert!((lowered.report.gate_threshold - 0.7).abs() < 1e-9);

    let raised = judge(&fenced_verdict(0.9, true))
        .safety_with_threshold(&skill(), &rules, 0.90)
        .await
        .expect("safety");
    assert!((raised.report.score_pct - 0.75).abs() < 1e-9);
    assert!(raised.report.blocked);
    assert!((raised.report.gate_threshold - 0.9).abs() < 1e-9);
}

#[tokio::test]
async fn a_configured_gate_threshold_reaches_the_plain_safety_call() {
    let skill_judge = judge(&fenced_verdict(1.0, true)).with_gate_threshold(0.90);
    assert!((skill_judge.gate_threshold() - 0.9).abs() < 1e-9);

    let judgement = skill_judge
        .safety(&skill(), &[warning_rule()])
        .await
        .expect("safety");

    assert!((judgement.report.score_pct - 0.75).abs() < 1e-9);
    assert!(judgement.report.blocked);
    assert!((judgement.report.gate_threshold - 0.9).abs() < 1e-9);
}

#[tokio::test]
async fn an_out_of_range_gate_threshold_is_a_validation_error() {
    for threshold in [1.5, -0.1, f64::NAN] {
        let model = ScriptedModel::text(fenced_verdict(1.0, true));
        let skill_judge = SkillJudge::new(Arc::new(model.clone()));

        let error = skill_judge
            .safety_with_threshold(&skill(), &[], threshold)
            .await
            .expect_err("invalid threshold");

        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(
            error.to_string().contains("safety gate threshold"),
            "{error}"
        );
        assert_eq!(model.call_count(), 0, "the model must not be consulted");
    }
}

#[tokio::test]
async fn an_auto_fail_still_short_circuits_a_raised_gate() {
    let model = ScriptedModel::text(fenced_verdict(1.0, true));
    let skill_judge = SkillJudge::new(Arc::new(model.clone()));
    let rules = [warning_rule().with_auto_fail(AutoFailCode::UndeclaredExfiltration)];

    let judgement = skill_judge
        .safety_with_threshold(&skill(), &rules, 0.90)
        .await
        .expect("safety");

    assert!(judgement.used_fallback);
    assert_eq!(model.call_count(), 0, "the model must not be consulted");
    assert_eq!(judgement.report.score_pct, 0.0);
    assert!(judgement.report.blocked);
    assert!(judgement.report.dimensions.is_none());
    assert!((judgement.report.gate_threshold - 0.9).abs() < 1e-9);
}
