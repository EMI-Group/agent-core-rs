//! The analyzer wired into [`openjiuwen_memory::facade::LongTermMemory`].
//!
//! These tests pin the facade integration surface — `remember`, `consolidate`, and
//! `analyzer()` — rather than the analyzer internals (which are covered by the module
//! unit tests): rule-fact extraction into stored records, per-fact dedupe on a repeated
//! `remember`, the degraded LLM path surfaced on `RememberReport::degraded`,
//! cancellation before any write, `consolidate` plans against real engine records for
//! the redundant/conflicting/unrelated cases, dedupe-policy variation, and the
//! validation contract for blank input.
//!
//! Everything runs offline against in-memory stores: facts come from the deterministic
//! `RuleExtractor` (or an emptied `ScriptedModel` for the degraded path), so no
//! assertion depends on the wall clock, the network, or a live provider.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::analyzer::{
    AnalyzerConfig, ClassifierKind, ConflictKind, ConsolidationAction, ExtractedFact,
    ExtractionInput, ExtractionResult, ExtractorKind, FactExtractor, LlmExtractor, MemoryAnalyzer,
};
use openjiuwen_memory::config::{DedupeConfig, MemoryEngineConfig};
use openjiuwen_memory::facade::{LongTermMemory, ScopeAnalyzerOverrideSpec};
use openjiuwen_memory::record::{
    AddOptions, CallOptions, DedupePolicy, MemoryRecord, MemoryScope, MemoryType, NewMemory,
    SearchOptions,
};
use openjiuwen_test_support::ScriptedModel;
use tokio_util::sync::CancellationToken;

/// Builds an empty in-memory facade over `config`.
async fn facade(config: MemoryEngineConfig) -> Arc<LongTermMemory> {
    LongTermMemory::in_memory(config)
        .await
        .expect("in-memory facade")
}

/// Builds an in-memory facade around an explicitly supplied analyzer.
fn facade_with_analyzer(
    config: MemoryEngineConfig,
    analyzer: Arc<MemoryAnalyzer>,
) -> Arc<LongTermMemory> {
    LongTermMemory::builder(config)
        .analyzer(analyzer)
        .build()
        .expect("facade with analyzer")
}

/// The scope every test owns its records under.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// Reads the facade's `tags` metadata convention back out of a stored record.
fn tags_of(record: &MemoryRecord) -> Vec<String> {
    record
        .metadata
        .get("tags")
        .and_then(|tags| tags.as_array())
        .map(|tags| {
            tags.iter()
                .filter_map(|tag| tag.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// A fixed extractor that returns one caller-supplied fact, ignoring the conversation.
///
/// Used to prove the facade keeps every write field (`importance`, `source`, tags) of a
/// fact that the deterministic rules cannot produce on their own.
struct FixedExtractor {
    /// The single fact every extraction returns.
    fact: ExtractedFact,
}

#[async_trait]
impl FactExtractor for FixedExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "fixed"
    }

    async fn extract(
        &self,
        input: &ExtractionInput,
        options: &CallOptions,
    ) -> Result<ExtractionResult> {
        options.check()?;
        input.validate()?;
        Ok(ExtractionResult::new(vec![self.fact.clone()], self.name()))
    }
}

#[tokio::test]
async fn remember_extracts_preference_identity_and_project_facts() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();
    // The default analyzer is the deterministic rule extractor over the engine's knobs.
    assert_eq!(ltm.analyzer().extractor_name(), "rule");

    let mut conversation = ExtractionInput::default();
    conversation.push_user("my name is Alice.");
    conversation.push_user("I prefer dark mode.");
    conversation.push_user("The service uses Postgres.");

    let report = ltm
        .remember(&scope, &conversation, &AddOptions::default())
        .await
        .expect("remember");
    assert_eq!(report.extracted, 3, "{report:?}");
    assert_eq!(
        report.extracted,
        report.added + report.merged + report.skipped,
        "{report:?}"
    );
    assert!(report.added + report.merged > 0, "{report:?}");
    assert_eq!(report.degraded, None);

    let records = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(records.len(), 3);
    // Identity and preference are `user_profile`; the project statement is `semantic_memory`.
    let profiles = records
        .iter()
        .filter(|record| record.kind == MemoryType::UserProfile)
        .count();
    assert_eq!(profiles, 2, "{records:?}");
    assert!(records.iter().any(|record| {
        record.content == "my name is Alice" && tags_of(record).contains(&"identity".to_owned())
    }));
    assert!(records.iter().any(|record| {
        record.content == "I prefer dark mode" && tags_of(record).contains(&"preference".to_owned())
    }));
    let project = records
        .iter()
        .find(|record| record.kind == MemoryType::SemanticMemory)
        .expect("the project fact is stored");
    assert_eq!(project.content, "The service uses Postgres");
    assert!(tags_of(project).contains(&"project".to_owned()));
}

#[tokio::test]
async fn remembered_facts_keep_their_content_importance_source_and_tags() {
    let fact = ExtractedFact::semantic("the release train leaves on fridays")
        .with_importance(0.42)
        .with_source("conversation")
        .with_tag("custom");
    let analyzer = MemoryAnalyzer::with_extractor(
        AnalyzerConfig::default(),
        Arc::new(FixedExtractor { fact }),
    );
    assert_eq!(analyzer.extractor_name(), "fixed");
    let ltm = facade_with_analyzer(MemoryEngineConfig::default(), Arc::new(analyzer));
    // The facade exposes the analyzer it actually writes through.
    assert_eq!(ltm.analyzer().extractor_name(), "fixed");

    let scope = scope();
    let report = ltm
        .remember(
            &scope,
            &ExtractionInput::from_text("when does the release train leave?"),
            &AddOptions::default(),
        )
        .await
        .expect("remember");
    assert_eq!(
        (
            report.extracted,
            report.added,
            report.merged,
            report.skipped
        ),
        (1, 1, 0, 0),
        "{report:?}"
    );

    let records = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(records.len(), 1);
    let stored = &records[0];
    assert_eq!(stored.content, "the release train leaves on fridays");
    assert_eq!(stored.kind, MemoryType::SemanticMemory);
    assert!((stored.importance - 0.42).abs() < f32::EPSILON);
    assert_eq!(stored.source.as_deref(), Some("conversation"));
    assert_eq!(tags_of(stored), vec!["custom".to_owned()]);

    // The same fact is reachable through the recall pipeline, not just through `list`.
    let hits = ltm
        .engine()
        .search(
            &scope,
            "release train",
            &SearchOptions::default().with_threshold(0.0),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].record.id, stored.id);
    assert_eq!(hits[0].record.content, stored.content);
}

#[tokio::test]
async fn remembering_the_same_conversation_twice_merges_instead_of_duplicating() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();
    let conversation = ExtractionInput::from_text("I prefer dark mode.");

    let first = ltm
        .remember(&scope, &conversation, &AddOptions::default())
        .await
        .expect("first remember");
    assert_eq!(
        (first.extracted, first.added, first.merged, first.skipped),
        (1, 1, 0, 0),
        "{first:?}"
    );
    let before = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(before.len(), 1);

    let second = ltm
        .remember(&scope, &conversation, &AddOptions::default())
        .await
        .expect("second remember");
    assert_eq!(second.extracted, 1, "{second:?}");
    assert_eq!(second.added, 0, "the duplicate must not be stored again");
    // The engine's default dedupe policy is `Merge`.
    assert_eq!(second.merged, 1, "{second:?}");
    assert_eq!(second.skipped, 0, "{second:?}");
    assert_eq!(
        second.extracted,
        second.added + second.merged + second.skipped,
        "{second:?}"
    );

    let after = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(after.len(), 1, "the per-scope record count must not grow");
    assert_eq!(after[0].id, before[0].id);
}

#[tokio::test]
async fn degraded_llm_extraction_is_surfaced_while_the_rules_still_write() {
    // An empty script makes every model call unusable, so the LLM extractor exhausts its
    // retry budget (`model_retries + 1 = 4` calls) and falls back to rules.
    let model = ScriptedModel::builder().build();
    let handle = model.clone();
    let config = AnalyzerConfig {
        extractor: ExtractorKind::Llm,
        ..AnalyzerConfig::default()
    };
    let extractor = LlmExtractor::new(Arc::new(model), config.clone());
    let analyzer = MemoryAnalyzer::with_extractor(config, Arc::new(extractor));
    assert_eq!(analyzer.extractor_name(), "llm");
    let ltm = facade_with_analyzer(MemoryEngineConfig::default(), Arc::new(analyzer));
    assert_eq!(ltm.analyzer().extractor_name(), "llm");

    let scope = scope();
    let conversation = ExtractionInput::from_text("my name is Alice. I prefer dark mode.");
    let report = ltm
        .remember(&scope, &conversation, &AddOptions::default())
        .await
        .expect("remember");
    let degraded = report
        .degraded
        .as_deref()
        .expect("the fallback is recorded");
    assert!(
        degraded.contains("fell back to rules"),
        "unexpected reason: {degraded}"
    );
    assert_eq!(report.extracted, 2, "{report:?}");
    assert_eq!(
        report.extracted,
        report.added + report.merged + report.skipped,
        "{report:?}"
    );
    assert_eq!(report.added, 2, "{report:?}");
    assert_eq!(handle.call_count(), 4, "one call per retry attempt");

    // The rule fallback still produced real records.
    let records = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(records.len(), 2);
    assert!(records
        .iter()
        .any(|record| record.content == "my name is Alice"));
    assert!(records
        .iter()
        .any(|record| record.content == "I prefer dark mode"));
}

#[tokio::test]
async fn cancelled_calls_return_early_and_write_nothing() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();
    let token = CancellationToken::new();
    token.cancel();

    let error = ltm
        .remember(
            &scope,
            &ExtractionInput::from_text("I prefer dark mode."),
            &AddOptions::default().with_cancellation(token.clone()),
        )
        .await
        .expect_err("a cancelled remember must fail");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);

    let error = ltm
        .consolidate(
            &scope,
            &NewMemory::semantic("I prefer dark mode"),
            &CallOptions::new().with_cancellation(token),
        )
        .await
        .expect_err("a cancelled consolidate must fail");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);
}

#[tokio::test]
async fn consolidate_plans_a_merge_for_a_redundant_memory() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();
    let stored = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("seed add")
        .added
        .remove(0);

    // A case/punctuation variant normalizes to the same text, so the plan merges.
    let incoming = NewMemory::semantic("alice prefers DARK mode.");
    let plan = ltm
        .consolidate(&scope, &incoming, &CallOptions::new())
        .await
        .expect("consolidate");
    assert_eq!(plan.actions.len(), 1);
    match &plan.actions[0] {
        ConsolidationAction::Merge {
            fact_index,
            target_id,
            merged_content,
        } => {
            assert_eq!(*fact_index, 0);
            assert_eq!(target_id, &stored.id, "the plan names the stored record");
            assert_eq!(merged_content, "alice prefers DARK mode.");
        }
        other => panic!("expected a merge, got {other:?}"),
    }
    assert_eq!(plan.adds(), 0);

    let verdicts = ltm
        .analyzer()
        .classifier(None)
        .expect("the default config asks for the rule classifier")
        .classify(
            &incoming,
            std::slice::from_ref(&stored),
            &ltm.config().dedupe,
        )
        .await
        .expect("classify");
    assert_eq!(verdicts.len(), 1);
    assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
    assert_eq!(verdicts[0].target_id.as_deref(), Some(stored.id.as_str()));
    assert!(verdicts[0].degraded.is_none());

    // A plan describes writes only: nothing was stored, replaced, or deleted.
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn consolidate_adds_conflicting_and_unrelated_memories() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();
    let stored = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice is a developer"),
            &AddOptions::default(),
        )
        .await
        .expect("seed add")
        .added
        .remove(0);
    let classifier = ltm
        .analyzer()
        .classifier(None)
        .expect("the default config asks for the rule classifier");

    // Same subject, different value: a conflict, never a silent delete or merge.
    let conflicting = NewMemory::semantic("Alice is a manager");
    let plan = ltm
        .consolidate(&scope, &conflicting, &CallOptions::new())
        .await
        .expect("consolidate conflicting");
    assert!(
        matches!(
            plan.actions[..],
            [ConsolidationAction::Add { fact_index: 0 }]
        ),
        "{plan:?}"
    );
    assert_eq!(plan.adds(), 1);
    let verdicts = classifier
        .classify(
            &conflicting,
            std::slice::from_ref(&stored),
            &ltm.config().dedupe,
        )
        .await
        .expect("classify conflicting");
    assert_eq!(verdicts[0].kind, ConflictKind::Conflicting);
    assert_eq!(verdicts[0].target_id.as_deref(), Some(stored.id.as_str()));

    // No relation at all: also an add, and the verdict points at nothing.
    let unrelated = NewMemory::semantic("the build pipeline has three stages");
    let plan = ltm
        .consolidate(&scope, &unrelated, &CallOptions::new())
        .await
        .expect("consolidate unrelated");
    assert_eq!(plan.adds(), 1);
    assert!(matches!(
        plan.actions[..],
        [ConsolidationAction::Add { fact_index: 0 }]
    ));
    let verdicts = classifier
        .classify(
            &unrelated,
            std::slice::from_ref(&stored),
            &ltm.config().dedupe,
        )
        .await
        .expect("classify unrelated");
    assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
    assert!(verdicts[0].target_id.is_none());

    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn consolidate_follows_the_configured_dedupe_policy() {
    for (policy, expected) in [
        (DedupePolicy::Skip, "skip"),
        (DedupePolicy::Replace, "replace"),
    ] {
        let config = MemoryEngineConfig {
            dedupe: DedupeConfig {
                policy,
                ..DedupeConfig::default()
            },
            ..MemoryEngineConfig::default()
        };
        let ltm = facade(config).await;
        let scope = scope();
        let stored = ltm
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode"),
                &AddOptions::default(),
            )
            .await
            .expect("seed add")
            .added
            .remove(0);

        let plan = ltm
            .consolidate(
                &scope,
                &NewMemory::semantic("alice prefers DARK mode."),
                &CallOptions::new(),
            )
            .await
            .expect("consolidate");
        let json = serde_json::to_value(&plan).expect("plan json");
        assert_eq!(
            json["actions"][0]["action"],
            serde_json::json!(expected),
            "{policy:?}: {plan:?}"
        );
        assert_eq!(
            json["actions"][0]["target_id"],
            serde_json::json!(stored.id)
        );
        assert_eq!(plan.adds(), 0);
        // Neither policy applies anything here: the stored record is untouched.
        assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
    }
}

#[tokio::test]
async fn rejected_inputs_are_validation_errors_and_store_nothing() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let error = ltm
        .consolidate(&scope, &NewMemory::semantic("   "), &CallOptions::new())
        .await
        .expect_err("blank incoming content");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");

    // A conversation with no turns is valid input that extracts nothing: the analyzer
    // validates turn roles, not emptiness, so `remember` succeeds with a zero report.
    let report = ltm
        .remember(&scope, &ExtractionInput::default(), &AddOptions::default())
        .await
        .expect("empty conversation");
    assert_eq!(
        (
            report.extracted,
            report.added,
            report.merged,
            report.skipped
        ),
        (0, 0, 0, 0),
        "{report:?}"
    );
    assert_eq!(report.degraded, None);

    // A blank turn role is what the analyzer rejects before any write.
    let mut invalid = ExtractionInput::default();
    invalid.push_turn("  ", "I prefer dark mode.");
    let error = ltm
        .remember(&scope, &invalid, &AddOptions::default())
        .await
        .expect_err("blank role");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");

    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);
}

#[tokio::test]
async fn a_scope_override_asking_for_the_llm_extractor_without_a_model_is_unsupported() {
    // The facade holds no model, so the scope's `ExtractorKind::Llm` request cannot be
    // honoured: it is a typed error naming the field, not a silent run of the rules.
    let ltm = facade(MemoryEngineConfig::default()).await;
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new().with_extractor(ExtractorKind::Llm),
    )
    .await
    .expect("override");
    let error = ltm
        .remember(
            &scope(),
            &ExtractionInput::from_text("my name is Alice."),
            &AddOptions::default(),
        )
        .await
        .expect_err("a model-backed extractor needs a model");
    let Error::Unsupported(message) = error else {
        panic!("expected Unsupported, got {error:?}");
    };
    assert!(message.contains("extractor"), "{message}");
    assert!(message.contains("no model was supplied"), "{message}");
    assert!(
        message.contains("ExtractorKind::Rule"),
        "the fix is named: {message}"
    );
    assert_eq!(ltm.list(&scope(), &[]).await.expect("list").len(), 0);
}

#[tokio::test]
async fn a_scope_override_asking_for_the_llm_classifier_without_a_model_is_unsupported() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new().with_classifier(ClassifierKind::Llm),
    )
    .await
    .expect("override");
    let error = ltm
        .remember(
            &scope(),
            &ExtractionInput::from_text("my name is Alice."),
            &AddOptions::default(),
        )
        .await
        .expect_err("a model-backed classifier needs a model");
    let Error::Unsupported(message) = error else {
        panic!("expected Unsupported, got {error:?}");
    };
    assert!(message.contains("classifier"), "{message}");
    assert!(message.contains("no model was supplied"), "{message}");
    assert!(message.contains("ClassifierKind::Rule"), "{message}");
    assert_eq!(ltm.list(&scope(), &[]).await.expect("list").len(), 0);
}
