//! [`MemoryAnalyzer`]: the extract → classify → consolidate entry point.
//!
//! The analyzer owns one [`FactExtractor`] (rule-based by default, model-backed when a
//! [`Model`] is supplied and [`AnalyzerConfig::extractor`] asks for it) and builds the
//! matching [`ConflictClassifier`] on demand. [`MemoryAnalyzer::consolidate`] turns a
//! batch of facts into a [`ConsolidationPlan`] against already-stored records, using the
//! engine's [`DedupeConfig`] so the plan agrees with the write path's policy.
//!
//! The analyzer never writes storage itself: it produces facts and a plan, and the
//! caller (engine, harness, or workflow) applies them.
use crate::analyzer::config::{AnalyzerConfig, ClassifierKind, ExtractorKind};
use crate::analyzer::conflict::{ConflictClassifier, RuleClassifier};
use crate::analyzer::fact::{ExtractedFact, ExtractionInput, ExtractionResult, FactExtractor};
use crate::analyzer::llm::LlmExtractor;
use crate::analyzer::llm_classifier::LlmClassifier;
use crate::analyzer::rule::RuleExtractor;
use crate::config::DedupeConfig;
use crate::recall::{find_duplicate, merge_records, Candidate};
use crate::record::{CallOptions, DedupePolicy, MemoryRecord, MemoryScope};
use chrono::Utc;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// One write the caller should perform for an extracted fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum ConsolidationAction {
    /// Store the fact as a new record.
    Add {
        /// Index of the fact in the batch passed to
        /// [`MemoryAnalyzer::consolidate`].
        fact_index: usize,
    },
    /// Fold the fact into an existing record and store the merged text under its id.
    Merge {
        /// Index of the fact in the batch.
        fact_index: usize,
        /// Id of the record the fact folds into.
        target_id: String,
        /// The text that should be stored for `target_id`.
        merged_content: String,
    },
    /// Replace the existing record's content with the fact's content.
    Replace {
        /// Index of the fact in the batch.
        fact_index: usize,
        /// Id of the record to replace.
        target_id: String,
    },
    /// Drop the fact because it duplicates an existing record.
    Skip {
        /// Index of the fact in the batch.
        fact_index: usize,
        /// Id of the record it duplicates.
        target_id: String,
    },
}

/// The writes a caller should perform for one extracted batch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConsolidationPlan {
    /// One action per fact, in the input order of the batch.
    pub actions: Vec<ConsolidationAction>,
}

impl ConsolidationPlan {
    /// Returns the number of actions that create a new record.
    #[must_use]
    pub fn adds(&self) -> usize {
        self.actions
            .iter()
            .filter(|action| matches!(action, ConsolidationAction::Add { .. }))
            .count()
    }
}

/// Turns conversation input into facts and consolidates them against stored records.
pub struct MemoryAnalyzer {
    /// The analyzer settings.
    config: AnalyzerConfig,
    /// The extractor this analyzer runs.
    extractor: Arc<dyn FactExtractor>,
}

impl MemoryAnalyzer {
    /// Builds an analyzer that always extracts with deterministic rules.
    #[must_use]
    pub fn rule(config: AnalyzerConfig) -> Self {
        let extractor = Arc::new(RuleExtractor::new(config.clone()));
        Self { config, extractor }
    }

    /// Builds the analyzer `config` asks for.
    ///
    /// [`ExtractorKind::Llm`] with a model yields an [`LlmExtractor`] (whose own fallback
    /// is the rule extractor); every other combination yields rules, so a config that
    /// requests the model without supplying one degrades deterministically instead of
    /// failing.
    #[must_use]
    pub fn from_config(config: AnalyzerConfig, model: Option<Arc<dyn Model>>) -> Self {
        let extractor: Arc<dyn FactExtractor> = match (config.extractor, model) {
            (ExtractorKind::Llm, Some(model)) => Arc::new(LlmExtractor::new(model, config.clone())),
            _ => Arc::new(RuleExtractor::new(config.clone())),
        };
        Self { config, extractor }
    }

    /// Builds an analyzer around an externally constructed extractor.
    #[must_use]
    pub fn with_extractor(config: AnalyzerConfig, extractor: Arc<dyn FactExtractor>) -> Self {
        Self { config, extractor }
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Returns the extractor's name (`rule`/`llm`/a custom one).
    #[must_use]
    pub fn extractor_name(&self) -> &str {
        self.extractor.name()
    }

    /// Extracts facts from `input`.
    ///
    /// The analyzer validates its own config and the input before delegating, so an
    /// invalid call fails with `Error::Validation` instead of reaching the model.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an invalid config or input,
    /// [`openjiuwen_common::error::Error::Cancelled`] when `options` is cancelled, and
    /// the extractor's own error otherwise.
    pub async fn analyze(
        &self,
        input: &ExtractionInput,
        options: &CallOptions,
    ) -> Result<ExtractionResult> {
        self.config.validate()?;
        input.validate()?;
        self.extractor.extract(input, options).await
    }

    /// Builds the conflict classifier `config` asks for.
    ///
    /// [`ClassifierKind::Llm`] with a model yields an [`LlmClassifier`]; every other
    /// combination yields [`RuleClassifier`].
    #[must_use]
    pub fn classifier(&self, model: Option<Arc<dyn Model>>) -> Arc<dyn ConflictClassifier> {
        match (self.config.classifier, model) {
            (ClassifierKind::Llm, Some(model)) => {
                Arc::new(LlmClassifier::new(model, self.config.clone()))
            }
            _ => Arc::new(RuleClassifier::new(self.config.clone())),
        }
    }

    /// Plans the writes for `incoming` against `existing` under `dedupe`.
    ///
    /// With `dedupe.enabled == false` every fact becomes
    /// [`ConsolidationAction::Add`]. Otherwise each fact is compared with the stored
    /// records using [`crate::recall::dedupe::find_duplicate`] and the configured
    /// [`DedupePolicy`] decides between merge, replace, skip, and add. The plan is
    /// computed against the records as supplied; it does not fold earlier actions of the
    /// same batch into each other.
    #[must_use]
    pub fn consolidate(
        &self,
        incoming: &[ExtractedFact],
        existing: &[MemoryRecord],
        dedupe: &DedupeConfig,
    ) -> ConsolidationPlan {
        let candidates: Vec<Candidate> = existing
            .iter()
            .cloned()
            .map(|record| Candidate::new(record, None))
            .collect();
        let mut actions = Vec::with_capacity(incoming.len());
        for (fact_index, fact) in incoming.iter().enumerate() {
            if !dedupe.enabled {
                actions.push(ConsolidationAction::Add { fact_index });
                continue;
            }
            let record = record_from_fact(fact);
            let matched = find_duplicate(&record, &candidates, dedupe);
            let action = match matched {
                Some(matched) => {
                    let target = &existing[matched.index];
                    match dedupe.policy {
                        DedupePolicy::Merge => ConsolidationAction::Merge {
                            fact_index,
                            target_id: target.id.clone(),
                            merged_content: merge_records(target, &record, Utc::now()).content,
                        },
                        DedupePolicy::Replace => ConsolidationAction::Replace {
                            fact_index,
                            target_id: target.id.clone(),
                        },
                        DedupePolicy::Skip => ConsolidationAction::Skip {
                            fact_index,
                            target_id: target.id.clone(),
                        },
                        DedupePolicy::Add => ConsolidationAction::Add { fact_index },
                    }
                }
                None => ConsolidationAction::Add { fact_index },
            };
            actions.push(action);
        }
        ConsolidationPlan { actions }
    }
}

/// Builds a throw-away record for a fact so the dedupe helpers can be reused.
fn record_from_fact(fact: &ExtractedFact) -> MemoryRecord {
    let memory = fact.to_new_memory();
    let mut record = MemoryRecord::new(MemoryScope::default(), memory.content, memory.kind);
    if let Some(importance) = memory.importance {
        record.importance = importance;
    }
    record.metadata = memory.metadata;
    record.source = memory.source;
    record
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MemoryType, NewMemory};
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::ScriptedModel;

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::SemanticMemory,
        )
        .with_id(id)
    }

    fn fact(content: &str) -> ExtractedFact {
        ExtractedFact::semantic(content)
    }

    #[tokio::test]
    async fn rule_analyzer_extracts_deterministically() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        assert_eq!(analyzer.extractor_name(), "rule");
        let result = analyzer
            .analyze(
                &ExtractionInput::from_text("I prefer dark mode"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].content, "I prefer dark mode");
        assert_eq!(analyzer.config().max_facts, 8);
    }

    #[test]
    fn from_config_picks_the_requested_extractor() {
        let analyzer = MemoryAnalyzer::from_config(AnalyzerConfig::default(), None);
        assert_eq!(analyzer.extractor_name(), "rule");

        let config = AnalyzerConfig {
            extractor: ExtractorKind::Llm,
            ..Default::default()
        };
        let analyzer = MemoryAnalyzer::from_config(config.clone(), None);
        assert_eq!(analyzer.extractor_name(), "rule");

        let model = ScriptedModel::builder().build();
        let analyzer = MemoryAnalyzer::from_config(config, Some(Arc::new(model)));
        assert_eq!(analyzer.extractor_name(), "llm");
    }

    #[test]
    fn classifier_follows_the_classifier_kind() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        assert_eq!(analyzer.classifier(None).name(), "rule");
        let config = AnalyzerConfig {
            classifier: ClassifierKind::Llm,
            ..Default::default()
        };
        let analyzer = MemoryAnalyzer::rule(config.clone());
        assert_eq!(analyzer.classifier(None).name(), "rule");
        let model = ScriptedModel::builder().build();
        assert_eq!(analyzer.classifier(Some(Arc::new(model))).name(), "llm");
    }

    #[tokio::test]
    async fn with_extractor_uses_the_supplied_extractor() {
        let analyzer = MemoryAnalyzer::with_extractor(
            AnalyzerConfig::default(),
            Arc::new(RuleExtractor::new(AnalyzerConfig::default())),
        );
        assert_eq!(analyzer.extractor_name(), "rule");
        let result = analyzer
            .analyze(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts[0].content, "my name is Alice");
    }

    #[tokio::test]
    async fn analyze_validates_config_and_input() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig {
            max_facts: 0,
            ..Default::default()
        });
        let error = analyzer
            .analyze(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));

        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        let mut input = ExtractionInput::from_text("my name is Alice");
        input.turns[0].role.clear();
        let error = analyzer
            .analyze(&input, &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
    }

    #[test]
    fn consolidate_merges_duplicates() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        let existing = vec![record("id-1", "Alice prefers dark mode")];
        let plan = analyzer.consolidate(
            &[fact("alice prefers DARK mode.")],
            &existing,
            &DedupeConfig::default(),
        );
        assert_eq!(plan.actions.len(), 1);
        match &plan.actions[0] {
            ConsolidationAction::Merge {
                fact_index,
                target_id,
                merged_content,
            } => {
                assert_eq!(*fact_index, 0);
                assert_eq!(target_id, "id-1");
                assert_eq!(merged_content, "alice prefers DARK mode.");
            }
            other => panic!("expected a merge, got {other:?}"),
        }
        assert_eq!(plan.adds(), 0);
    }

    #[test]
    fn consolidate_applies_every_policy() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        let existing = vec![record("id-1", "Alice prefers dark mode")];
        let incoming = [fact("alice prefers DARK mode.")];
        for (policy, expected) in [
            (DedupePolicy::Merge, "merge"),
            (DedupePolicy::Replace, "replace"),
            (DedupePolicy::Skip, "skip"),
            (DedupePolicy::Add, "add"),
        ] {
            let dedupe = DedupeConfig {
                policy,
                ..Default::default()
            };
            let plan = analyzer.consolidate(&incoming, &existing, &dedupe);
            let json = serde_json::to_value(&plan).unwrap();
            assert_eq!(json["actions"][0]["action"], serde_json::json!(expected));
            assert_eq!(json["actions"][0]["fact_index"], serde_json::json!(0));
        }
    }

    #[test]
    fn consolidate_is_add_everything_when_dedupe_is_disabled() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        let existing = vec![record("id-1", "Alice prefers dark mode")];
        let dedupe = DedupeConfig {
            enabled: false,
            ..Default::default()
        };
        let plan = analyzer.consolidate(
            &[fact("Alice prefers dark mode"), fact("new fact")],
            &existing,
            &dedupe,
        );
        assert_eq!(plan.actions.len(), 2);
        assert!(plan
            .actions
            .iter()
            .all(|action| matches!(action, ConsolidationAction::Add { .. })));
        assert_eq!(plan.adds(), 2);
    }

    #[test]
    fn consolidate_adds_unmatched_facts_in_order() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        let existing = vec![record("id-1", "Alice prefers dark mode")];
        let incoming = [
            fact("the service uses postgres"),
            fact("alice prefers DARK mode."),
            fact("the build pipeline has three stages"),
        ];
        let plan = analyzer.consolidate(&incoming, &existing, &DedupeConfig::default());
        assert_eq!(plan.actions.len(), 3);
        assert!(matches!(
            plan.actions[0],
            ConsolidationAction::Add { fact_index: 0 }
        ));
        assert!(matches!(
            plan.actions[1],
            ConsolidationAction::Merge { fact_index: 1, .. }
        ));
        assert!(matches!(
            plan.actions[2],
            ConsolidationAction::Add { fact_index: 2 }
        ));
        assert_eq!(plan.adds(), 2);
    }

    #[test]
    fn consolidate_with_no_existing_records_adds_everything() {
        let analyzer = MemoryAnalyzer::rule(AnalyzerConfig::default());
        let plan = analyzer.consolidate(&[fact("one"), fact("two")], &[], &DedupeConfig::default());
        assert_eq!(plan.adds(), 2);
        assert_eq!(
            plan,
            ConsolidationPlan {
                actions: vec![
                    ConsolidationAction::Add { fact_index: 0 },
                    ConsolidationAction::Add { fact_index: 1 },
                ]
            }
        );
    }

    #[test]
    fn plan_types_round_trip_and_tolerate_unknown_fields() {
        let plan = ConsolidationPlan {
            actions: vec![
                ConsolidationAction::Add { fact_index: 0 },
                ConsolidationAction::Merge {
                    fact_index: 1,
                    target_id: "id-1".to_owned(),
                    merged_content: "merged".to_owned(),
                },
                ConsolidationAction::Replace {
                    fact_index: 2,
                    target_id: "id-2".to_owned(),
                },
                ConsolidationAction::Skip {
                    fact_index: 3,
                    target_id: "id-3".to_owned(),
                },
            ],
        };
        let json = serde_json::to_value(&plan).unwrap();
        assert_eq!(json["actions"][3]["action"], serde_json::json!("skip"));
        assert_eq!(json["actions"][3]["target_id"], serde_json::json!("id-3"));
        let restored: ConsolidationPlan = serde_json::from_value(json).unwrap();
        assert_eq!(restored, plan);
        let tolerant: ConsolidationPlan =
            serde_json::from_value(serde_json::json!({"actions": [], "unknown": 1})).unwrap();
        assert_eq!(tolerant.actions.len(), 0);
        assert_eq!(ConsolidationPlan::default(), tolerant);
    }

    #[test]
    fn record_from_fact_maps_the_write_fields() {
        let fact = ExtractedFact::new("port = 8080", MemoryType::Variable)
            .with_importance(0.4)
            .with_source("conversation");
        let record = record_from_fact(&fact);
        assert_eq!(record.content, "port = 8080");
        assert_eq!(record.kind, MemoryType::SemanticMemory);
        assert!((record.importance - 0.4).abs() < f32::EPSILON);
        assert_eq!(record.source.as_deref(), Some("conversation"));
        let _: NewMemory = fact.to_new_memory();
    }
}
