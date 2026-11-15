//! The staged analyzer pipeline (`StagedAnalyzer`) over the plan, reasoning, proactive, and
//! conflict stages.
//!
//! These tests pin the staged surface rather than the analyzer internals: parity with the
//! plain [`MemoryAnalyzer`] for both `analyze` and `consolidate`, the executed stage order,
//! the reasoning metadata each fact carries, plan gating and focused turns, the proactive
//! write/skip decisions with their documented reasons, the `StageKind::Llm` fallback and
//! its degradation reporting, `StageKind::Llm` without a model, the per-scope
//! [`ScopeAnalyzerOverrides`] plumbing, and cancellation with zero model calls.
//!
//! Everything runs offline: facts come from the deterministic rule extractor and every
//! model-backed stage is driven by an `openjiuwen_test_support::ScriptedModel`, so no
//! assertion depends on the network, credentials, or the wall clock.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::analyzer::{
    AnalyzerConfig, ConsolidationAction, ExtractedFact, ExtractionInput, ExtractionPlan,
    ExtractionStage, ExtractorKind, MemoryAnalyzer, PlanIntent, PlanOperation, ProactiveConfig,
    ProactiveDecision, ProactiveSignal, ReasoningBasis, RulePlanStage, ScopeAnalyzerOverrides,
    StageConfig, StageInput, StageKind, StageOutput, StagedAnalyzer, DEFAULT_REASONING_KEY,
};
use openjiuwen_memory::config::{DedupeConfig, MemoryEngineConfig};
use openjiuwen_memory::record::{CallOptions, DedupePolicy, MemoryRecord, MemoryScope, MemoryType};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The three-rule conversation every parity test analyzes: identity, preference, project.
fn conversation() -> ExtractionInput {
    let mut input = ExtractionInput::default();
    input.push_user("my name is Alice.");
    input.push_user("I prefer dark mode.");
    input.push_user("The service uses Postgres.");
    input
}

/// A stored record the conflict stage can classify against.
fn record(id: &str, content: &str) -> MemoryRecord {
    MemoryRecord::new(
        MemoryScope::new("alice", "ws"),
        content,
        MemoryType::UserProfile,
    )
    .with_id(id)
}

/// The fact contents of a batch, for order-sensitive assertions.
fn contents(facts: &[ExtractedFact]) -> Vec<&str> {
    facts.iter().map(|fact| fact.content.as_str()).collect()
}

/// Uncancelled call options.
fn options() -> CallOptions {
    CallOptions::new()
}

/// Every optional stage switched to its model-backed implementation.
fn llm_stages() -> StageConfig {
    StageConfig {
        plan: StageKind::Llm,
        reasoning: StageKind::Llm,
        proactive: StageKind::Llm,
        ..StageConfig::default()
    }
}

/// The stage names the built-in sequence always runs, in order.
fn built_in_stages() -> Vec<String> {
    ["plan", "extract", "reasoning", "proactive", "conflict"]
        .iter()
        .map(|name| (*name).to_owned())
        .collect()
}

/// Runs the one-transcript pipeline and returns the proactive decision.
async fn proactive_decision(transcript: &str, stages: StageConfig) -> ProactiveDecision {
    let result = StagedAnalyzer::new(AnalyzerConfig::default())
        .with_stage_config(stages)
        .analyze(&ExtractionInput::from_text(transcript), &options())
        .await
        .expect("staged analysis");
    result.decision.expect("the proactive stage always decides")
}

/// Reads the deterministic skip/fallback message out of a validation error.
fn validation_message(error: Error) -> String {
    match error {
        Error::Validation(message) => message,
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_analysis_matches_the_plain_analyzer() {
    let config = AnalyzerConfig::default();
    let conversation = conversation();
    let plain = MemoryAnalyzer::rule(config.clone())
        .analyze(&conversation, &options())
        .await
        .unwrap();
    let staged = StagedAnalyzer::new(config.clone())
        .analyze(&conversation, &options())
        .await
        .unwrap();
    let from_engine = StagedAnalyzer::from_engine(&MemoryEngineConfig::default(), None)
        .expect("the engine default asks for rules")
        .analyze(&conversation, &options())
        .await
        .unwrap();

    assert_eq!(
        contents(plain.facts.as_slice()),
        vec![
            "my name is Alice",
            "I prefer dark mode",
            "The service uses Postgres"
        ]
    );
    assert_eq!(staged.facts().len(), plain.facts.len());
    for (staged_fact, plain_fact) in staged.facts().iter().zip(plain.facts.iter()) {
        assert_eq!(staged_fact.content, plain_fact.content);
        assert_eq!(staged_fact.kind, plain_fact.kind);
        assert_eq!(staged_fact.tags, plain_fact.tags);
        assert!(
            (staged_fact.importance - plain_fact.importance).abs() < f32::EPSILON,
            "{}: {} vs {}",
            staged_fact.content,
            staged_fact.importance,
            plain_fact.importance
        );
    }
    // `from_engine` derives the same analyzer knobs as the default config.
    assert_eq!(from_engine.facts(), staged.facts());
    assert_eq!(from_engine.plan, staged.plan);
    assert_eq!(from_engine.result.extractor, "rule");
    assert_eq!(staged.result.extractor, plain.extractor);
    assert!(!staged.is_degraded());
    assert_eq!(staged.result.degraded, None);
}

#[tokio::test]
async fn staged_consolidate_matches_the_plain_analyzer() {
    let config = AnalyzerConfig::default();
    let staged = StagedAnalyzer::new(config.clone());
    let incoming = vec![
        ExtractedFact::semantic("The service uses Postgres"),
        ExtractedFact::semantic("the release train leaves on fridays"),
    ];
    let existing = vec![record("id-1", "The service uses Postgres")];

    let dedupe = DedupeConfig::default();
    let staged_plan = staged.consolidate(&incoming, &existing, &dedupe);
    let plain_plan = MemoryAnalyzer::rule(config).consolidate(&incoming, &existing, &dedupe);
    assert_eq!(staged_plan, plain_plan);
    assert_eq!(staged_plan.actions.len(), 2);
    match &staged_plan.actions[0] {
        ConsolidationAction::Merge {
            fact_index,
            target_id,
            merged_content,
        } => {
            assert_eq!(*fact_index, 0);
            assert_eq!(target_id, "id-1");
            assert!(
                merged_content.eq_ignore_ascii_case("The service uses Postgres"),
                "{merged_content}"
            );
        }
        other => panic!("expected a merge into id-1, got {other:?}"),
    }
    assert_eq!(
        staged_plan.actions[1],
        ConsolidationAction::Add { fact_index: 1 }
    );
    assert_eq!(staged_plan.adds(), 1);

    // The policy travels through the staged facade unchanged.
    let skip = DedupeConfig {
        policy: DedupePolicy::Skip,
        ..DedupeConfig::default()
    };
    let skipped = staged.consolidate(&incoming, &existing, &skip);
    assert_eq!(
        skipped.actions,
        vec![
            ConsolidationAction::Skip {
                fact_index: 0,
                target_id: "id-1".to_owned(),
            },
            ConsolidationAction::Add { fact_index: 1 },
        ]
    );
}

#[tokio::test]
async fn stage_order_and_reasoning_metadata_follow_the_documented_contract() {
    let staged = StagedAnalyzer::new(AnalyzerConfig::default())
        .analyze(&conversation(), &options())
        .await
        .unwrap();

    assert_eq!(staged.stages, built_in_stages());
    assert_eq!(staged.facts().len(), 3);
    assert_eq!(staged.reasoning.len(), 3);
    for (index, entry) in staged.reasoning.iter().enumerate() {
        assert_eq!(entry.fact_index, index);
        assert_eq!(staged.rationale(index), Some(entry.rationale.as_str()));
        assert_eq!(
            staged.facts()[index].metadata.get(DEFAULT_REASONING_KEY),
            Some(&json!(entry.rationale)),
            "fact {index} carries its rationale under the default key"
        );
        assert_eq!(
            staged.facts()[index].metadata.get("plan_intent"),
            Some(&json!(staged.plan.intent.as_str()))
        );
    }
    assert_eq!(staged.reasoning[0].basis, ReasoningBasis::Identity);
    assert_eq!(staged.reasoning[1].basis, ReasoningBasis::Preference);
    assert_eq!(staged.reasoning[2].basis, ReasoningBasis::Project);
    assert!(staged.rationale(0).unwrap().contains("identity statement"));
    assert!(staged.rationale(1).unwrap().contains("stated preference"));
    // No candidates: the conflict stage runs but has nothing to classify.
    assert_eq!(
        staged.verdicts.len(),
        0,
        "no candidate records were supplied"
    );

    let against = StagedAnalyzer::new(AnalyzerConfig::default())
        .analyze_against(
            &conversation(),
            &[record("id-1", "I prefer dark mode")],
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(against.stages, built_in_stages());
    assert_eq!(against.verdicts.len(), against.facts().len());
    assert_eq!(
        against
            .verdicts
            .iter()
            .map(|verdict| verdict.fact_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let redundant = against
        .verdicts
        .iter()
        .find(|verdict| verdict.fact_index == 1)
        .expect("the preference fact is classified");
    assert_eq!(
        redundant.verdict.target_id.as_deref(),
        Some("id-1"),
        "{redundant:?}"
    );
}

/// A custom stage that replaces extraction with one fixed fact.
struct MarkerStage;

#[async_trait]
impl ExtractionStage for MarkerStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "marker"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        assert_eq!(input.config, AnalyzerConfig::default());
        Ok(StageOutput::of(self.name())
            .with_facts(vec![ExtractedFact::semantic("marker fact")])
            .with_extractor("marker"))
    }
}

#[tokio::test]
async fn a_custom_stage_sequence_replaces_the_built_in_order() {
    let config = AnalyzerConfig::default();
    let with_plan = StagedAnalyzer::new(config.clone()).with_stages(vec![
        Arc::new(RulePlanStage::new(config.clone())),
        Arc::new(MarkerStage),
    ]);
    let result = with_plan
        .analyze(&conversation(), &options())
        .await
        .unwrap();
    assert_eq!(result.stages, vec!["plan", "marker"]);
    // The plan stage ran first and filled the state the marker stage received.
    assert_eq!(result.plan.intent, PlanIntent::Preference);
    assert_eq!(result.plan.focus_turns, vec![0, 1, 2]);
    assert!(result.plan.extracts_facts());
    assert_eq!(contents(result.facts()), vec!["marker fact"]);
    assert_eq!(result.result.extractor, "marker");
    assert_eq!(result.reasoning.len(), 0);
    assert!(result.decision.is_none());

    // The custom list is used verbatim: a sequence without the plan stage leaves the
    // default (non-actionable) plan in place.
    let without_plan = StagedAnalyzer::new(config).with_stages(vec![Arc::new(MarkerStage)]);
    let result = without_plan
        .analyze(&conversation(), &options())
        .await
        .unwrap();
    assert_eq!(result.stages, vec!["marker"]);
    assert_eq!(result.plan, ExtractionPlan::default());
    assert_eq!(contents(result.facts()), vec!["marker fact"]);
}

#[tokio::test]
async fn a_plan_without_extract_facts_yields_no_facts() {
    // A transcript whose turns are all below `min_message_len` produces the empty plan.
    let mut short = ExtractionInput::default();
    short.push_user("ok");
    short.push_user("hm");
    let planned_out = StagedAnalyzer::new(AnalyzerConfig::default())
        .analyze(&short, &options())
        .await
        .unwrap();
    assert_eq!(planned_out.plan.intent, PlanIntent::None);
    assert!(!planned_out.plan.extracts_facts());
    assert_eq!(planned_out.plan.operations.len(), 0);
    assert_eq!(planned_out.facts().len(), 0);
    assert_eq!(
        planned_out.decision.unwrap().reason,
        "no candidate facts were extracted"
    );

    // A model plan that never asks for `extract_facts` gates the extract stage off.
    let model = ScriptedModel::builder()
        .respond_text(
            "{\"intent\":\"general\",\"operations\":[\"extract_summary\"],\
             \"rationale\":\"only a summary is needed\"}",
        )
        .build();
    let handle = model.clone();
    let pipeline = StagedAnalyzer::from_config(AnalyzerConfig::default(), Some(Arc::new(model)))
        .expect("a model satisfies the model-backed stages")
        .with_stage_config(StageConfig {
            plan: StageKind::Llm,
            ..StageConfig::default()
        });
    let gated = pipeline.analyze(&conversation(), &options()).await.unwrap();
    assert_eq!(handle.call_count(), 1);
    assert_eq!(gated.plan.intent, PlanIntent::General);
    assert_eq!(
        gated.plan.operations,
        vec![PlanOperation::ExtractSummary],
        "{:?}",
        gated.plan
    );
    assert!(!gated.plan.extracts_facts());
    assert_eq!(gated.facts().len(), 0);
    // The extract stage was gated off before it ran, so no extractor name is recorded.
    assert_eq!(gated.result.extractor, "");
    assert!(!gated.is_degraded());
    assert_eq!(
        gated.decision.unwrap().reason,
        "no candidate facts were extracted"
    );
    // The rule pipeline extracts from the same conversation, proving the gate did it.
    let ungated = StagedAnalyzer::new(AnalyzerConfig::default())
        .analyze(&conversation(), &options())
        .await
        .unwrap();
    assert_eq!(ungated.facts().len(), 3);
}

#[tokio::test]
async fn a_focused_plan_narrows_the_turns_the_extract_stage_sees() {
    let mut input = ExtractionInput::default();
    input.push_user("I prefer dark mode.");
    input.push_user("The service uses Postgres.");

    let focused_plan = ExtractionPlan {
        intent: PlanIntent::Preference,
        operations: vec![PlanOperation::ExtractFacts],
        focus_turns: vec![0],
        ..ExtractionPlan::default()
    };
    assert_eq!(focused_plan.selected_turns(&input), vec![0]);
    let unfocused_plan = ExtractionPlan {
        focus_turns: Vec::new(),
        ..focused_plan.clone()
    };
    assert_eq!(unfocused_plan.selected_turns(&input), vec![0, 1]);

    let model = ScriptedModel::builder()
        .respond_text(
            "{\"intent\":\"preference\",\"topic\":\"theme\",\
             \"operations\":[\"extract_facts\"],\"focus_turns\":[0],\
             \"rationale\":\"only the theme preference is memory\"}",
        )
        .build();
    let pipeline = StagedAnalyzer::from_config(AnalyzerConfig::default(), Some(Arc::new(model)))
        .expect("a model satisfies the model-backed stages")
        .with_stage_config(StageConfig {
            plan: StageKind::Llm,
            ..StageConfig::default()
        });
    let focused = pipeline.analyze(&input, &options()).await.unwrap();
    assert_eq!(focused.plan.focus_turns, vec![0]);
    assert_eq!(focused.plan.selected_turns(&input), vec![0]);
    assert_eq!(contents(focused.facts()), vec!["I prefer dark mode"]);
    // The narrowing is the plan's doing: the same input yields both facts with rules.
    let unfocused = StagedAnalyzer::new(AnalyzerConfig::default())
        .analyze(&input, &options())
        .await
        .unwrap();
    assert_eq!(
        contents(unfocused.facts()),
        vec!["I prefer dark mode", "The service uses Postgres"]
    );
}

#[tokio::test]
async fn the_proactive_stage_writes_directive_correction_and_preference_turns() {
    let directive =
        proactive_decision("remember to rotate the deploy key", StageConfig::default()).await;
    assert!(directive.write);
    assert_eq!(
        directive.signals,
        vec![
            ProactiveSignal::ExplicitDirective,
            ProactiveSignal::HighImportance
        ]
    );
    assert_eq!(directive.fact_indexes, vec![0]);
    assert!((directive.importance - 0.9).abs() < f32::EPSILON);
    assert!(
        directive.reason.contains("explicit_directive"),
        "{}",
        directive.reason
    );
    assert!(
        directive.reason.contains("writing 1 of 1 fact(s)"),
        "{}",
        directive.reason
    );
    assert!(
        directive.reason.contains("strongest importance 0.90"),
        "{}",
        directive.reason
    );
    assert!(directive.degraded.is_none());

    let correction = proactive_decision(
        "I no longer use Vim. I prefer dark mode instead.",
        StageConfig::default(),
    )
    .await;
    assert!(correction.write);
    assert!(
        correction.signals.contains(&ProactiveSignal::Correction),
        "{:?}",
        correction.signals
    );
    assert!(
        correction.signals.contains(&ProactiveSignal::Preference),
        "{:?}",
        correction.signals
    );
    assert!(
        correction.reason.contains("correction"),
        "{}",
        correction.reason
    );
    assert_eq!(correction.fact_indexes, vec![0]);

    let preference = proactive_decision("I prefer dark mode.", StageConfig::default()).await;
    assert!(preference.write);
    assert_eq!(
        preference.signals,
        vec![ProactiveSignal::Preference, ProactiveSignal::HighImportance]
    );
    assert!(
        preference.reason.contains("preference"),
        "{}",
        preference.reason
    );
}

#[tokio::test]
async fn the_proactive_stage_skips_with_the_documented_reasons() {
    let disabled = proactive_decision(
        "remember to rotate the deploy key",
        StageConfig {
            proactive_config: ProactiveConfig {
                enabled: false,
                ..ProactiveConfig::default()
            },
            ..StageConfig::default()
        },
    )
    .await;
    assert!(!disabled.write);
    assert_eq!(disabled.reason, "proactive memory is disabled");
    assert_eq!(disabled.signals.len(), 0);
    assert_eq!(disabled.fact_indexes.len(), 0);

    let empty = proactive_decision("ok", StageConfig::default()).await;
    assert!(!empty.write);
    assert_eq!(empty.reason, "no candidate facts were extracted");
    assert_eq!(empty.importance, 0.0);

    let short = proactive_decision("I like it", StageConfig::default()).await;
    assert!(!short.write);
    assert_eq!(short.reason, "transcript is shorter than 12 characters");
    assert_eq!(short.signals.len(), 0);

    let unremarkable = proactive_decision("The build is green today", StageConfig::default()).await;
    assert!(!unremarkable.write);
    assert!(
        unremarkable.reason.contains("no proactive-memory signal"),
        "{}",
        unremarkable.reason
    );
    assert!(
        unremarkable
            .reason
            .contains("0.40 below the 0.70 threshold"),
        "{}",
        unremarkable.reason
    );
    assert_eq!(unremarkable.signals.len(), 0);
    assert_eq!(unremarkable.fact_indexes.len(), 0);
    assert!((unremarkable.importance - 0.4).abs() < f32::EPSILON);
}

#[test]
fn proactive_signals_promote_the_batch_as_documented() {
    for signal in [
        ProactiveSignal::ExplicitDirective,
        ProactiveSignal::Correction,
        ProactiveSignal::Decision,
        ProactiveSignal::Preference,
    ] {
        assert!(signal.promotes_batch(), "{signal:?}");
        assert!(signal.promotes_batch_write(), "{signal:?}");
    }
    assert!(!ProactiveSignal::Repetition.promotes_batch());
    assert!(ProactiveSignal::Repetition.promotes_batch_write());
    assert!(!ProactiveSignal::HighImportance.promotes_batch());
    assert!(!ProactiveSignal::HighImportance.promotes_batch_write());
}

#[tokio::test]
async fn llm_stages_fall_back_to_rules_and_report_the_degradation() {
    let config = AnalyzerConfig::default();
    // An empty script serves the prose fallback, which no stage can parse.
    let model = ScriptedModel::builder().build();
    let handle = model.clone();
    let pipeline = StagedAnalyzer::from_config(config.clone(), Some(Arc::new(model)))
        .expect("a model satisfies the model-backed stages")
        .with_stage_config(llm_stages());
    let fell_back = pipeline.analyze(&conversation(), &options()).await.unwrap();

    let attempts = config.model_retries as usize + 1;
    assert_eq!(
        handle.call_count(),
        3 * attempts,
        "each of the three model-backed stages spends its retry budget"
    );
    assert!(fell_back.is_degraded());
    let degraded = fell_back
        .result
        .degraded
        .as_deref()
        .expect("the result records the degradation");
    for reason in [
        "llm plan fell back to rules",
        "llm reasoning fell back to rules",
        "llm proactive decision fell back to rules",
    ] {
        assert!(degraded.contains(reason), "missing {reason}: {degraded}");
    }
    assert!(fell_back
        .plan
        .degraded
        .as_deref()
        .unwrap()
        .starts_with("llm plan fell back to rules"));
    let decision = fell_back.decision.clone().expect("a decision");
    assert!(decision
        .degraded
        .as_deref()
        .unwrap()
        .contains("llm proactive decision fell back to rules"));

    // The deterministic stages still produced the rule output.
    let rule_only = StagedAnalyzer::new(config)
        .analyze(&conversation(), &options())
        .await
        .unwrap();
    assert_eq!(fell_back.facts(), rule_only.facts());
    assert_eq!(fell_back.plan.intent, rule_only.plan.intent);
    assert_eq!(fell_back.plan.operations, rule_only.plan.operations);
    assert_eq!(fell_back.plan.focus_turns, rule_only.plan.focus_turns);
    assert_eq!(fell_back.reasoning, rule_only.reasoning);
    let rule_decision = rule_only.decision.clone().expect("a decision");
    assert_eq!(decision.write, rule_decision.write);
    assert_eq!(decision.reason, rule_decision.reason);
    assert_eq!(decision.signals, rule_decision.signals);
    assert_eq!(decision.fact_indexes, rule_decision.fact_indexes);
    assert!(rule_decision.degraded.is_none());
}

#[tokio::test]
async fn llm_stages_use_valid_answers_without_degradation() {
    let model = ScriptedModel::builder()
        .respond_text(
            "{\"intent\":\"preference\",\"topic\":\"theme\",\
             \"operations\":[\"extract_facts\"],\"focus_turns\":[0,1,2],\
             \"rationale\":\"the user shares profile facts\"}",
        )
        .respond_text(
            "{\"reasons\":[\
             {\"index\":0,\"rationale\":\"an identity statement\",\"basis\":\"identity\"},\
             {\"index\":1,\"rationale\":\"a durable theme preference\",\"basis\":\"preference\"},\
             {\"index\":2,\"rationale\":\"a project detail\",\"basis\":\"project\"}]}",
        )
        .respond_text(
            "{\"write\":true,\"reason\":\"identity and a durable preference\",\
             \"signals\":[\"preference\"],\"fact_indexes\":[0,1,2],\"importance\":0.85}",
        )
        .build();
    let handle = model.clone();
    let pipeline = StagedAnalyzer::from_config(AnalyzerConfig::default(), Some(Arc::new(model)))
        .expect("a model satisfies the model-backed stages")
        .with_stage_config(llm_stages());
    let result = pipeline.analyze(&conversation(), &options()).await.unwrap();

    assert_eq!(handle.call_count(), 3);
    assert!(!result.is_degraded());
    assert_eq!(result.result.degraded, None);
    assert_eq!(result.plan.topic.as_deref(), Some("theme"));
    assert_eq!(result.plan.rationale, "the user shares profile facts");
    assert_eq!(result.plan.focus_turns, vec![0, 1, 2]);
    assert_eq!(result.facts().len(), 3);
    assert_eq!(result.reasoning.len(), 3);
    assert_eq!(result.reasoning[1].rationale, "a durable theme preference");
    assert_eq!(result.reasoning[1].basis, ReasoningBasis::Preference);
    assert_eq!(result.rationale(1), Some("a durable theme preference"));
    assert_eq!(
        result.facts()[1].metadata.get(DEFAULT_REASONING_KEY),
        Some(&json!("a durable theme preference"))
    );
    let decision = result.decision.clone().expect("a decision");
    assert!(decision.write);
    assert_eq!(decision.reason, "identity and a durable preference");
    assert_eq!(decision.signals, vec![ProactiveSignal::Preference]);
    assert_eq!(decision.fact_indexes, vec![0, 1, 2]);
    assert!((decision.importance - 0.85).abs() < f32::EPSILON);
    assert!(decision.degraded.is_none());
}

#[tokio::test]
async fn llm_stage_kinds_without_a_model_still_analyze_with_the_rule_stages() {
    let configured = StagedAnalyzer::new(AnalyzerConfig::default()).with_stage_config(llm_stages());
    assert_eq!(configured.stage_config().plan, StageKind::Llm);
    assert_eq!(configured.stage_config().reasoning, StageKind::Llm);
    assert_eq!(configured.stage_config().proactive, StageKind::Llm);

    let result = configured
        .analyze(&conversation(), &options())
        .await
        .unwrap();
    assert_eq!(result.stages, built_in_stages());
    assert_eq!(result.facts().len(), 3);
    // The rule stages ran (same output as the deterministic pipeline), but the fallback is
    // reported on the result rather than only logged.
    assert!(result.is_degraded());
    let degraded = result
        .result
        .degraded
        .as_deref()
        .expect("the no-model fallback is reported");
    for stage in ["plan", "reasoning", "proactive"] {
        let reason = format!("llm {stage} stage requested without a model");
        assert!(degraded.contains(&reason), "missing {reason}: {degraded}");
    }
    assert_eq!(result.decision.as_ref().unwrap().degraded, None);

    let rule_only = StagedAnalyzer::new(AnalyzerConfig::default())
        .analyze(&conversation(), &options())
        .await
        .unwrap();
    assert_eq!(result.facts(), rule_only.facts());
    assert_eq!(result.plan, rule_only.plan);
    assert_eq!(result.reasoning, rule_only.reasoning);
    assert_eq!(result.decision, rule_only.decision);
}

#[tokio::test]
async fn overrides_change_the_thresholds_and_the_fact_cap() {
    let base = AnalyzerConfig::default();
    let conversation = conversation();
    let unfiltered = StagedAnalyzer::new(base.clone())
        .analyze(&conversation, &options())
        .await
        .unwrap();
    assert_eq!(unfiltered.facts().len(), 3);

    let overrides = ScopeAnalyzerOverrides::new().with_min_importance(0.8);
    assert!(!overrides.is_empty());
    assert_eq!(overrides.min_importance(), Some(0.8));
    let staged = StagedAnalyzer::new(base.clone()).with_overrides(overrides.clone());
    assert_eq!(staged.overrides().min_importance(), Some(0.8));
    assert!((staged.effective_config().min_importance - 0.8).abs() < f32::EPSILON);
    assert_eq!(staged.effective_config(), overrides.effective_config(&base));
    let filtered = staged.analyze(&conversation, &options()).await.unwrap();
    assert_eq!(contents(filtered.facts()), vec!["my name is Alice"]);
    assert_eq!(filtered.facts()[0].kind, MemoryType::UserProfile);
    assert_eq!(filtered.facts()[0].tags, vec!["identity"]);
    assert!((filtered.facts()[0].importance - 0.85).abs() < f32::EPSILON);

    let capped =
        StagedAnalyzer::new(base).with_overrides(ScopeAnalyzerOverrides::new().with_max_facts(2));
    assert_eq!(capped.effective_config().max_facts, 2);
    let result = capped.analyze(&conversation, &options()).await.unwrap();
    assert_eq!(
        contents(result.facts()),
        vec!["my name is Alice", "I prefer dark mode"]
    );
}

#[tokio::test]
async fn overrides_switch_the_extractor_and_the_stage_list() {
    let base = AnalyzerConfig::default();
    let model = ScriptedModel::builder()
        .respond_text(
            "{\"intent\":\"preference\",\"topic\":\"theme\",\
             \"operations\":[\"extract_facts\"],\"focus_turns\":[0],\
             \"rationale\":\"model plan\"}",
        )
        .respond_text(
            "{\"facts\":[{\"content\":\"Alice prefers dark mode\",\
             \"type\":\"user_profile\",\"importance\":0.9}]}",
        )
        .build();
    let handle = model.clone();
    let overrides = ScopeAnalyzerOverrides::new()
        .with_model(Arc::new(model))
        .with_extractor(ExtractorKind::Llm)
        .with_stages(StageConfig {
            plan: StageKind::Llm,
            ..StageConfig::default()
        });
    let staged = StagedAnalyzer::new(base.clone()).with_overrides(overrides.clone());
    assert_eq!(staged.effective_config().extractor, ExtractorKind::Llm);
    assert_eq!(staged.effective_config(), overrides.effective_config(&base));
    // The override's stage selection wins over the pipeline's own.
    assert_eq!(staged.stage_config().plan, StageKind::Llm);
    assert_eq!(staged.stage_config().reasoning, StageKind::Rule);
    assert_eq!(staged.stage_config().proactive, StageKind::Rule);

    let result = staged
        .analyze(
            &ExtractionInput::from_text("I prefer dark mode."),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(handle.call_count(), 2, "one plan call, one extraction call");
    assert_eq!(result.result.extractor, "llm");
    assert_eq!(result.plan.topic.as_deref(), Some("theme"));
    assert_eq!(result.plan.rationale, "model plan");
    assert_eq!(contents(result.facts()), vec!["Alice prefers dark mode"]);
    assert_eq!(result.result.degraded, None);
}

#[tokio::test]
async fn overrides_report_emptiness_and_validate_their_ranges() {
    let base = AnalyzerConfig::default();
    let empty = ScopeAnalyzerOverrides::new();
    assert!(empty.is_empty());
    assert!(empty.validate().is_ok());
    assert_eq!(empty.effective_config(&base), base);

    let invalid_threshold = ScopeAnalyzerOverrides::new().with_min_importance(1.5);
    assert!(!invalid_threshold.is_empty());
    let message = validation_message(invalid_threshold.validate().unwrap_err());
    assert!(message.contains("min_importance"), "{message}");
    assert!(message.contains("1.5"), "{message}");
    let error = StagedAnalyzer::new(base.clone())
        .with_overrides(invalid_threshold)
        .analyze(&conversation(), &options())
        .await
        .unwrap_err();
    assert!(validation_message(error).contains("min_importance"));

    let invalid_importance = ScopeAnalyzerOverrides::new().with_default_importance(f32::NAN);
    let message = validation_message(invalid_importance.validate().unwrap_err());
    assert!(message.contains("default_importance"), "{message}");

    let zero_cap = ScopeAnalyzerOverrides::new().with_max_facts(0);
    let message = validation_message(zero_cap.validate().unwrap_err());
    assert!(message.contains("max_facts"), "{message}");

    let empty_key = ScopeAnalyzerOverrides::new().with_stages(StageConfig {
        reasoning_key: "  ".to_owned(),
        ..StageConfig::default()
    });
    let message = validation_message(empty_key.validate().unwrap_err());
    assert!(message.contains("reasoning_key"), "{message}");
}

#[tokio::test]
async fn a_cancelled_analysis_returns_cancelled_with_zero_model_calls() {
    let model = ScriptedModel::builder().build();
    let handle = model.clone();
    let pipeline = StagedAnalyzer::from_config(AnalyzerConfig::default(), Some(Arc::new(model)))
        .expect("a model satisfies the model-backed stages")
        .with_stage_config(llm_stages());
    let token = CancellationToken::new();
    token.cancel();

    let error = pipeline
        .analyze(&conversation(), &options().with_cancellation(token.clone()))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert!(token.is_cancelled());
    assert_eq!(handle.call_count(), 0);

    // The same pipeline without the cancelled token does reach the model.
    let result = pipeline.analyze(&conversation(), &options()).await.unwrap();
    assert!(result.is_degraded());
    assert_eq!(
        handle.call_count(),
        3 * (AnalyzerConfig::default().model_retries as usize + 1)
    );
}
