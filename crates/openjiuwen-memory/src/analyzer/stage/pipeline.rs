//! The staged analyzer: assemble the stages and run them as one pipeline.
//!
//! [`StagedAnalyzer`] wires the default sequence (plan → extract → reason → proactive →
//! conflict) around the existing [`MemoryAnalyzer`], so callers that only need facts keep
//! using `MemoryAnalyzer::analyze` while callers that need the plan, the rationale, or the
//! proactive decision use the staged pipeline. The stage list is replaceable through
//! [`StagedAnalyzer::with_stages`], and every stage is built from the *effective*
//! configuration — the base [`AnalyzerConfig`] with the per-scope
//! [`ScopeAnalyzerOverrides`] applied — so a scope override really changes the extractor,
//! the thresholds, and the stage kinds.
//!
//! Construction and every run refuse to degrade silently: a configuration that asks for
//! the model-backed extractor or classifier without a model is
//! [`openjiuwen_common::error::Error::Unsupported`], and a model-backed *stage kind*
//! requested without a model runs the rule stage while reporting the fallback on
//! [`StagedResult::result`](crate::analyzer::StagedResult::result)'s
//! [`ExtractionResult::degraded`](crate::analyzer::ExtractionResult::degraded) field.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::conflict::{ConflictClassifier, ConflictVerdict};
use crate::analyzer::fact::{ExtractedFact, ExtractionInput, ExtractionResult};
use crate::analyzer::stage::plan::{LlmPlanStage, RulePlanStage};
use crate::analyzer::stage::proactive::{
    LlmProactiveStage, ProactiveConfig, ProactiveDecision, RuleProactiveStage,
};
use crate::analyzer::stage::reasoning::{
    LlmReasoningStage, RuleReasoningStage, DEFAULT_REASONING_KEY,
};
use crate::analyzer::stage::{
    ExtractionStage, FactReasoning, ScopeAnalyzerOverrides, StageInput, StageOutput,
};
use crate::analyzer::MemoryAnalyzer;
use crate::config::{DedupeConfig, MemoryEngineConfig};
use crate::record::{CallOptions, MemoryRecord};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Which implementation of a stage the pipeline builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    /// Deterministic, model-free implementation (the default).
    #[default]
    Rule,
    /// Model-backed implementation with the deterministic one as fallback.
    Llm,
}

/// The pipeline's stage selection and proactive thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StageConfig {
    /// Implementation of the plan stage.
    pub plan: StageKind,
    /// Implementation of the reasoning stage.
    pub reasoning: StageKind,
    /// Implementation of the proactive stage.
    pub proactive: StageKind,
    /// The proactive stage's own thresholds.
    pub proactive_config: ProactiveConfig,
    /// Metadata key the reasoning stage stores a rationale under.
    pub reasoning_key: String,
}

impl Default for StageConfig {
    /// Returns the documented defaults: every stage deterministic.
    fn default() -> Self {
        Self {
            plan: StageKind::Rule,
            reasoning: StageKind::Rule,
            proactive: StageKind::Rule,
            proactive_config: ProactiveConfig::default(),
            reasoning_key: DEFAULT_REASONING_KEY.to_owned(),
        }
    }
}

impl StageConfig {
    /// Validates the proactive thresholds and the metadata key.
    pub fn validate(&self) -> Result<()> {
        if self.reasoning_key.trim().is_empty() {
            return Err(openjiuwen_common::error::Error::validation(
                "stage config `reasoning_key` must not be empty",
            ));
        }
        self.proactive_config.validate()
    }
}

/// One fact's classification against the candidate records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FactVerdict {
    /// Index of the fact in the batch.
    pub fact_index: usize,
    /// The classifier's verdict for this fact against one candidate record.
    pub verdict: ConflictVerdict,
}

impl Default for FactVerdict {
    /// An unrelated verdict attributed to the first fact.
    fn default() -> Self {
        Self {
            fact_index: 0,
            verdict: ConflictVerdict::default(),
        }
    }
}

/// The extraction stage: runs the configured [`crate::analyzer::FactExtractor`] over the planned turns.
pub struct ExtractStage {
    /// The analyzer that owns the extractor and validates the call.
    analyzer: Arc<MemoryAnalyzer>,
}

impl ExtractStage {
    /// Builds the stage around `analyzer`.
    #[must_use]
    pub fn new(analyzer: Arc<MemoryAnalyzer>) -> Self {
        Self { analyzer }
    }

    /// Returns the analyzer in use.
    #[must_use]
    pub fn analyzer(&self) -> &Arc<MemoryAnalyzer> {
        &self.analyzer
    }
}

#[async_trait]
impl ExtractionStage for ExtractStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "extract"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        if !input.plan.extracts_facts() {
            return Ok(StageOutput::of(self.name()).with_facts(Vec::new()));
        }
        let selected = input.plan.selected_turns(&input.conversation);
        let conversation = focus(&input.conversation, &selected);
        let extraction = self.analyzer.analyze(&conversation, options).await?;
        let facts: Vec<ExtractedFact> = extraction
            .facts
            .into_iter()
            .filter(|fact| {
                !fact.content.trim().is_empty()
                    && fact.importance.is_finite()
                    && fact.importance >= input.config.min_importance
            })
            .take(input.config.max_facts)
            .collect();
        let mut output = StageOutput::of(self.name())
            .with_facts(facts)
            .with_extractor(extraction.extractor);
        if let Some(reason) = extraction.degraded {
            output = output.degraded(reason);
        }
        Ok(output)
    }
}

/// Returns a conversation holding only the selected current turns.
fn focus(conversation: &ExtractionInput, selected: &[usize]) -> ExtractionInput {
    if selected.len() == conversation.turns.len() {
        return conversation.clone();
    }
    let turns = selected
        .iter()
        .filter_map(|index| conversation.turns.get(*index).cloned())
        .collect();
    ExtractionInput {
        turns,
        history: conversation.history.clone(),
    }
}

/// The conflict stage: classifies the extracted facts against candidate records.
pub struct ConflictStage {
    /// The classifier (rule-based or model-backed).
    classifier: Arc<dyn ConflictClassifier>,
    /// The dedupe thresholds the classifier uses.
    dedupe: DedupeConfig,
    /// Maximum candidate records sent to the classifier.
    max_candidates: usize,
}

impl ConflictStage {
    /// Builds the stage from a classifier, the dedupe thresholds, and the candidate cap.
    #[must_use]
    pub fn new(
        classifier: Arc<dyn ConflictClassifier>,
        dedupe: DedupeConfig,
        max_candidates: usize,
    ) -> Self {
        Self {
            classifier,
            dedupe,
            max_candidates,
        }
    }

    /// Returns the classifier name in use.
    #[must_use]
    pub fn classifier_name(&self) -> &str {
        self.classifier.name()
    }
}

#[async_trait]
impl ExtractionStage for ConflictStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "conflict"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        if input.facts.is_empty() || input.candidates.is_empty() {
            return Ok(StageOutput::of(self.name()).with_verdicts(Vec::new()));
        }
        let cap = self.max_candidates.max(1);
        let candidates: Vec<MemoryRecord> = input.candidates.iter().take(cap).cloned().collect();
        let mut verdicts = Vec::new();
        let mut degraded: Vec<String> = Vec::new();
        for (fact_index, fact) in input.facts.iter().enumerate() {
            if fact.content.trim().is_empty() {
                // The extract stage filters blank content, so this only happens with a
                // caller-supplied stage sequence; the skip is reported rather than only
                // logged so the caller can tell "no conflict" from "never classified".
                let reason = format!(
                    "fact #{fact_index} skipped in conflict classification: its content is empty"
                );
                tracing::warn!(target: "openjiuwen_memory::analyzer", fact_index, "%reason");
                degraded.push(reason);
                continue;
            }
            let incoming = fact.to_new_memory();
            let classified = self
                .classifier
                .classify(&incoming, &candidates, &self.dedupe)
                .await?;
            for verdict in classified {
                if let Some(reason) = &verdict.degraded {
                    degraded.push(reason.clone());
                }
                verdicts.push(FactVerdict {
                    fact_index,
                    verdict,
                });
            }
        }
        let mut output = StageOutput::of(self.name()).with_verdicts(verdicts);
        if let Some(reason) =
            crate::analyzer::stage::degraded_reasons(degraded.iter().map(String::as_str))
        {
            output = output.degraded(reason);
        }
        Ok(output)
    }
}

/// The result of one staged analysis.
///
/// [`StagedResult::result`] has the same shape [`MemoryAnalyzer::analyze`] returns (with
/// the rationale attached as fact metadata), and the remaining fields expose the plan,
/// the proactive decision, the reasoning, and the conflict verdicts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StagedResult {
    /// The extracted facts and any degradation.
    pub result: ExtractionResult,
    /// The plan the plan stage produced.
    pub plan: crate::analyzer::ExtractionPlan,
    /// The proactive-write decision, when the proactive stage ran.
    pub decision: Option<ProactiveDecision>,
    /// The rationale per fact.
    pub reasoning: Vec<FactReasoning>,
    /// The conflict verdicts.
    pub verdicts: Vec<FactVerdict>,
    /// Names of the stages that ran, in order.
    pub stages: Vec<String>,
}

impl StagedResult {
    /// Returns the extracted facts.
    #[must_use]
    pub fn facts(&self) -> &[ExtractedFact] {
        &self.result.facts
    }

    /// Returns whether the proactive stage recommends writing memory.
    #[must_use]
    pub fn should_write(&self) -> bool {
        self.decision
            .as_ref()
            .is_some_and(|decision| decision.write)
    }

    /// Returns the facts the proactive decision selected (all facts when no decision ran).
    #[must_use]
    pub fn writable_facts(&self) -> Vec<&ExtractedFact> {
        match &self.decision {
            Some(decision) if decision.write => decision
                .fact_indexes
                .iter()
                .filter_map(|index| self.result.facts.get(*index))
                .collect(),
            Some(_) => Vec::new(),
            None => self.result.facts.iter().collect(),
        }
    }

    /// Returns `true` when any stage degraded.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.result.is_degraded()
            || self.plan.degraded.is_some()
            || self
                .decision
                .as_ref()
                .is_some_and(ProactiveDecision::is_degraded)
    }

    /// Returns the rationale attached to the fact at `index`.
    #[must_use]
    pub fn rationale(&self, index: usize) -> Option<&str> {
        self.reasoning
            .iter()
            .find(|entry| entry.fact_index == index)
            .map(|entry| entry.rationale.as_str())
    }
}

/// Runs the staged pipeline: plan → extract → reason → proactive → conflict.
pub struct StagedAnalyzer {
    /// The analyzer the extract stage and `consolidate` delegate to.
    analyzer: MemoryAnalyzer,
    /// The model the model-backed stages use when no scope override supplies one.
    model: Option<Arc<dyn Model>>,
    /// Stage selection and proactive thresholds.
    stages: StageConfig,
    /// Per-scope overrides applied to every run.
    overrides: ScopeAnalyzerOverrides,
    /// Dedupe thresholds the conflict stage classifies with.
    dedupe: DedupeConfig,
    /// A caller-supplied stage sequence that replaces the built-in one.
    custom: Option<Vec<Arc<dyn ExtractionStage>>>,
}

impl StagedAnalyzer {
    /// Builds the deterministic pipeline over `config`.
    ///
    /// The pipeline always runs the rule extractor and the rule conflict classifier; use
    /// [`StagedAnalyzer::from_config`] when the configuration may ask for the model-backed
    /// extractor. A `config` that keeps [`ExtractorKind::Llm`](crate::analyzer::ExtractorKind::Llm)
    /// (or [`ClassifierKind::Llm`](crate::analyzer::ClassifierKind::Llm)) is not silently
    /// downgraded here: [`StagedAnalyzer::analyze`]/[`StagedAnalyzer::analyze_against`] and
    /// [`StagedAnalyzer::consolidate`] resolve the effective configuration again and return
    /// [`openjiuwen_common::error::Error::Unsupported`] when the model it asks for is
    /// missing.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            analyzer: MemoryAnalyzer::rule(config),
            model: None,
            stages: StageConfig::default(),
            overrides: ScopeAnalyzerOverrides::default(),
            dedupe: DedupeConfig::default(),
            custom: None,
        }
    }

    /// Builds the pipeline `config` asks for.
    ///
    /// The extract stage follows [`MemoryAnalyzer::from_config`] (`ExtractorKind::Llm`
    /// with a model, rules when the configuration asks for rules); the
    /// plan/reasoning/proactive stages follow [`StageConfig`] and fall back to their rule
    /// implementations when `Llm` is requested without a model, recording that fallback in
    /// [`StagedResult::result`]'s [`ExtractionResult::degraded`](crate::analyzer::ExtractionResult::degraded)
    /// field *and* in a `tracing::warn!`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] when `config` asks for the
    /// model-backed extractor and `model` is `None` (see [`MemoryAnalyzer::from_config`]).
    pub fn from_config(config: AnalyzerConfig, model: Option<Arc<dyn Model>>) -> Result<Self> {
        let analyzer = MemoryAnalyzer::from_config(config, model.clone())?;
        Ok(Self {
            analyzer,
            model,
            stages: StageConfig::default(),
            overrides: ScopeAnalyzerOverrides::default(),
            dedupe: DedupeConfig::default(),
            custom: None,
        })
    }

    /// Builds the pipeline an engine configuration implies.
    ///
    /// # Errors
    ///
    /// The same [`openjiuwen_common::error::Error::Unsupported`] as
    /// [`StagedAnalyzer::from_config`].
    pub fn from_engine(config: &MemoryEngineConfig, model: Option<Arc<dyn Model>>) -> Result<Self> {
        Self::from_config(AnalyzerConfig::from_engine(config), model)
    }

    /// Returns the pipeline with `stages` as its stage selection.
    #[must_use]
    pub fn with_stage_config(mut self, stages: StageConfig) -> Self {
        self.stages = stages;
        self
    }

    /// Returns the pipeline with `overrides` applied to every run.
    #[must_use]
    pub fn with_overrides(mut self, overrides: ScopeAnalyzerOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Returns the pipeline with `dedupe` as the conflict stage's thresholds.
    #[must_use]
    pub fn with_dedupe(mut self, dedupe: DedupeConfig) -> Self {
        self.dedupe = dedupe;
        self
    }

    /// Returns the pipeline with a caller-supplied stage sequence.
    ///
    /// The sequence replaces the built-in one entirely (including the extract and
    /// conflict stages), so the caller owns the ordering; `StageInput::config` still
    /// carries the effective configuration.
    #[must_use]
    pub fn with_stages(mut self, stages: Vec<Arc<dyn ExtractionStage>>) -> Self {
        self.custom = Some(stages);
        self
    }

    /// Returns the base configuration.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        self.analyzer.config()
    }

    /// Returns the per-scope overrides.
    #[must_use]
    pub fn overrides(&self) -> &ScopeAnalyzerOverrides {
        &self.overrides
    }

    /// Returns the stage selection in use.
    #[must_use]
    pub fn stage_config(&self) -> &StageConfig {
        self.overrides.stages().unwrap_or(&self.stages)
    }

    /// Returns the configuration a run resolves: the base config plus the overrides.
    #[must_use]
    pub fn effective_config(&self) -> AnalyzerConfig {
        self.overrides.effective_config(self.analyzer.config())
    }

    /// Extracts facts through the staged pipeline.
    ///
    /// Equivalent to [`StagedAnalyzer::analyze_against`] with no candidate records.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an invalid configuration or
    /// input, [`openjiuwen_common::error::Error::Cancelled`] when `options` is cancelled,
    /// [`openjiuwen_common::error::Error::Unsupported`] when the effective configuration
    /// asks for the model-backed extractor or classifier and no model is available, and
    /// the stage's own error otherwise.
    pub async fn analyze(
        &self,
        input: &ExtractionInput,
        options: &CallOptions,
    ) -> Result<StagedResult> {
        self.analyze_against(input, &[], options).await
    }

    /// Extracts facts and classifies them against `existing`.
    ///
    /// # Errors
    ///
    /// The same errors as [`StagedAnalyzer::analyze`].
    pub async fn analyze_against(
        &self,
        input: &ExtractionInput,
        existing: &[MemoryRecord],
        options: &CallOptions,
    ) -> Result<StagedResult> {
        options.check()?;
        self.overrides.validate()?;
        self.stage_config().validate()?;
        let config = self.effective_config();
        config.validate()?;
        input.validate()?;
        let model = self
            .overrides
            .model()
            .cloned()
            .or_else(|| self.model.clone());
        let built = self.build_stages(&config, model)?;
        let mut degradations = built.degradations;
        let mut state = StageInput::new(input.clone(), config).with_candidates(existing.to_vec());
        let mut ran: Vec<String> = Vec::with_capacity(built.stages.len());
        for stage in built.stages {
            options.check()?;
            let output = stage.run(&state, options).await?;
            ran.push(output.stage.clone());
            if let Some(reason) = &output.degraded {
                degradations.push(reason.clone());
            }
            output.apply(&mut state);
        }
        if let Some(reason) = state.plan.degraded.as_ref() {
            degradations.push(reason.clone());
        }
        if let Some(reason) = state
            .decision
            .as_ref()
            .and_then(|decision| decision.degraded.clone())
        {
            degradations.push(reason);
        }
        let degradation =
            crate::analyzer::stage::degraded_reasons(degradations.iter().map(String::as_str));
        let result = ExtractionResult {
            facts: state.facts,
            degraded: degradation,
            extractor: state.extractor,
        };
        Ok(StagedResult {
            result,
            plan: state.plan,
            decision: state.decision,
            reasoning: state.reasoning,
            verdicts: state.verdicts,
            stages: ran,
        })
    }

    /// Plans the writes for `incoming` against `existing` under `dedupe`.
    ///
    /// Delegates to [`MemoryAnalyzer::consolidate`], so the staged pipeline and the plain
    /// analyzer agree on the plan. The plan is a function of the facts, the stored records,
    /// and `dedupe` alone — consolidation never consults the extractor or the classifier —
    /// so the effective (override-resolved) configuration cannot change it.
    #[must_use]
    pub fn consolidate(
        &self,
        incoming: &[ExtractedFact],
        existing: &[MemoryRecord],
        dedupe: &DedupeConfig,
    ) -> crate::analyzer::ConsolidationPlan {
        self.analyzer.consolidate(incoming, existing, dedupe)
    }

    /// Builds the stage sequence for one run.
    ///
    /// Returns the stages plus the degradations that hold for the whole run, in the order
    /// the stages run: a [`StageKind::Llm`] stage configured without a model is reported
    /// here (and logged) so the fallback reaches
    /// [`StagedResult::result`]'s `degraded` field instead of staying in the log.
    fn build_stages(
        &self,
        config: &AnalyzerConfig,
        model: Option<Arc<dyn Model>>,
    ) -> Result<BuiltStages> {
        if let Some(custom) = &self.custom {
            return Ok(BuiltStages {
                stages: custom.clone(),
                degradations: Vec::new(),
            });
        }
        let selection = self.stage_config().clone();
        let mut degradations: Vec<String> = Vec::new();
        let planner: Arc<dyn ExtractionStage> = match (selection.plan, &model) {
            (StageKind::Llm, Some(model)) => {
                Arc::new(LlmPlanStage::new(model.clone(), config.clone()))
            }
            (StageKind::Llm, None) => {
                let reason = "llm plan stage requested without a model; using the rule plan stage";
                tracing::warn!(target: "openjiuwen_memory::analyzer", "{reason}");
                degradations.push(reason.to_owned());
                Arc::new(RulePlanStage::new(config.clone()))
            }
            (StageKind::Rule, _) => Arc::new(RulePlanStage::new(config.clone())),
        };
        let analyzer = Arc::new(MemoryAnalyzer::from_config(config.clone(), model.clone())?);
        let reasoning: Arc<dyn ExtractionStage> = match (selection.reasoning, &model) {
            (StageKind::Llm, Some(model)) => Arc::new(
                LlmReasoningStage::new(model.clone(), config.clone()).with_fallback(
                    RuleReasoningStage::new(config.clone())
                        .with_metadata_key(selection.reasoning_key.clone()),
                ),
            ),
            (StageKind::Llm, None) => {
                let reason = "llm reasoning stage requested without a model; \
                              using the rule reasoning stage";
                tracing::warn!(target: "openjiuwen_memory::analyzer", "{reason}");
                degradations.push(reason.to_owned());
                Arc::new(
                    RuleReasoningStage::new(config.clone())
                        .with_metadata_key(selection.reasoning_key.clone()),
                )
            }
            (StageKind::Rule, _) => Arc::new(
                RuleReasoningStage::new(config.clone())
                    .with_metadata_key(selection.reasoning_key.clone()),
            ),
        };
        let proactive: Arc<dyn ExtractionStage> = match (selection.proactive, &model) {
            (StageKind::Llm, Some(model)) => Arc::new(LlmProactiveStage::new(
                model.clone(),
                config.clone(),
                selection.proactive_config.clone(),
            )),
            (StageKind::Llm, None) => {
                let reason = "llm proactive stage requested without a model; \
                              using the rule proactive stage";
                tracing::warn!(target: "openjiuwen_memory::analyzer", "{reason}");
                degradations.push(reason.to_owned());
                Arc::new(RuleProactiveStage::new(
                    config.clone(),
                    selection.proactive_config.clone(),
                ))
            }
            (StageKind::Rule, _) => Arc::new(RuleProactiveStage::new(
                config.clone(),
                selection.proactive_config.clone(),
            )),
        };
        let conflict = ConflictStage::new(
            analyzer.classifier(model.clone())?,
            self.dedupe.clone(),
            config.max_candidates,
        );
        Ok(BuiltStages {
            stages: vec![
                planner,
                Arc::new(ExtractStage::new(analyzer)),
                reasoning,
                proactive,
                Arc::new(conflict),
            ],
            degradations,
        })
    }
}

/// The stage sequence one run executes, plus its run-wide degradations.
struct BuiltStages {
    /// The stages to run, in order.
    stages: Vec<Arc<dyn ExtractionStage>>,
    /// Why a configured model-backed stage ran its rule implementation instead.
    degradations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{
        AnalyzerConfig, ClassifierKind, ConflictKind, ConsolidationAction, ExtractionStage,
        ExtractorKind, MemoryAnalyzer, PlanIntent, ScopeAnalyzerOverrides, StageInput, StageOutput,
    };
    use crate::record::{CallOptions, MemoryScope, MemoryType};
    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, Result as CommonResult};
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    fn conversation() -> ExtractionInput {
        let mut input = ExtractionInput::default();
        input.push_user("my name is Alice.");
        input.push_user("I prefer dark mode.");
        input.push_user("The service uses Postgres.");
        input
    }

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::UserProfile,
        )
        .with_id(id)
    }

    #[tokio::test]
    async fn staged_pipeline_matches_the_plain_analyzer() {
        let config = AnalyzerConfig::default();
        let conversation = conversation();
        let plain = MemoryAnalyzer::rule(config.clone())
            .analyze(&conversation, &CallOptions::new())
            .await
            .unwrap();
        let staged = StagedAnalyzer::new(config.clone())
            .analyze(&conversation, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(
            staged.stages,
            vec!["plan", "extract", "reasoning", "proactive", "conflict"]
        );
        assert_eq!(staged.plan.intent, PlanIntent::Preference);
        assert!(staged.plan.extracts_facts());
        assert_eq!(staged.facts().len(), plain.facts.len());
        for (left, right) in staged.facts().iter().zip(plain.facts.iter()) {
            assert_eq!(left.content, right.content);
            assert_eq!(left.kind, right.kind);
            assert!((left.importance - right.importance).abs() < f32::EPSILON);
            assert_eq!(left.tags, right.tags);
            assert!(left.metadata.contains_key("rationale"));
        }
        assert_eq!(staged.reasoning.len(), staged.facts().len());
        assert!(
            staged.verdicts.is_empty(),
            "no candidate records were supplied"
        );
        assert!(!staged.is_degraded());
        assert_eq!(staged.result.extractor, "rule");
        assert!(staged.should_write());
        assert_eq!(staged.writable_facts().len(), staged.facts().len());
        assert!(staged
            .rationale(0)
            .is_some_and(|text| text.contains("preference")));
    }

    #[tokio::test]
    async fn staged_pipeline_skips_short_turns() {
        let mut input = ExtractionInput::default();
        input.push_user("ok");
        input.push_user("I prefer dark mode.");
        let config = AnalyzerConfig::default();
        let plain = MemoryAnalyzer::rule(config.clone())
            .analyze(&input, &CallOptions::new())
            .await
            .unwrap();
        let staged = StagedAnalyzer::new(config)
            .analyze(&input, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(staged.plan.focus_turns, vec![1]);
        assert_eq!(staged.facts().len(), plain.facts.len());
        assert_eq!(staged.facts()[0].content, "I prefer dark mode");
    }

    #[tokio::test]
    async fn staged_conflict_stage_classifies_against_existing_records() {
        let existing = vec![record("id-1", "I prefer dark mode")];
        let staged = StagedAnalyzer::new(AnalyzerConfig::default())
            .analyze_against(&conversation(), &existing, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(staged.verdicts.len(), staged.facts().len());
        let redundant = staged
            .verdicts
            .iter()
            .find(|entry| entry.fact_index == 1)
            .expect("the preference fact has a verdict");
        assert_eq!(redundant.verdict.kind, ConflictKind::Redundant);
        assert_eq!(redundant.verdict.target_id.as_deref(), Some("id-1"));
        assert!(!staged.is_degraded());
    }

    #[tokio::test]
    async fn the_conflict_stage_reports_facts_it_cannot_classify() {
        let conflict = ConflictStage::new(
            MemoryAnalyzer::rule(AnalyzerConfig::default())
                .classifier(None)
                .expect("the default config asks for the rule classifier"),
            DedupeConfig::default(),
            8,
        );
        let mut state = StageInput::new(
            ExtractionInput::from_text("anything"),
            AnalyzerConfig::default(),
        )
        .with_candidates(vec![record("id-1", "I prefer dark mode")]);
        state.facts = vec![ExtractedFact::semantic("   ")];
        let output = conflict.run(&state, &CallOptions::new()).await.unwrap();
        assert_eq!(output.verdicts.len(), 0);
        let degraded = output
            .degraded
            .as_deref()
            .expect("the skipped fact is reported, not just logged");
        assert!(degraded.contains("fact #0"), "{degraded}");
        assert!(degraded.contains("empty"), "{degraded}");
    }

    #[tokio::test]
    async fn override_threshold_drops_low_importance_facts() {
        let overrides = ScopeAnalyzerOverrides::new()
            .with_min_importance(0.8)
            .with_max_facts(2);
        let staged = StagedAnalyzer::new(AnalyzerConfig::default()).with_overrides(overrides);
        assert!((staged.effective_config().min_importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(staged.effective_config().max_facts, 2);
        let result = staged
            .analyze(
                &ExtractionInput::from_text("I prefer dark mode."),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts().len(), 0);
        assert!(!result.should_write());
        assert!(!result.is_degraded());
    }

    #[tokio::test]
    async fn override_extractor_builds_the_model_backed_stage() {
        let model = ScriptedModel::builder()
            .respond_text(
                "{\"facts\":[{\"content\":\"Alice prefers dark mode\",\
                 \"type\":\"user_profile\",\"importance\":0.9}]}",
            )
            .build();
        let handle = model.clone();
        let overrides = ScopeAnalyzerOverrides::new()
            .with_model(Arc::new(model))
            .with_extractor(ExtractorKind::Llm);
        let staged = StagedAnalyzer::new(AnalyzerConfig::default()).with_overrides(overrides);
        assert_eq!(staged.effective_config().extractor, ExtractorKind::Llm);
        let result = staged
            .analyze(
                &ExtractionInput::from_text("I prefer dark mode."),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.result.extractor, "llm");
        assert_eq!(result.facts().len(), 1);
        assert_eq!(result.facts()[0].content, "Alice prefers dark mode");
        assert!(result.facts()[0].metadata.contains_key("rationale"));
        assert!(result.should_write());
        assert_eq!(handle.call_count(), 1);
    }

    #[tokio::test]
    async fn llm_stage_kinds_run_through_the_pipeline() {
        let model = ScriptedModel::builder()
            .respond_text(
                "{\"intent\":\"preference\",\"topic\":\"theme\",\
                 \"operations\":[\"extract_facts\"],\"focus_turns\":[0],\
                 \"rationale\":\"the user states a preference\"}",
            )
            .respond_text(
                "{\"reasons\":[{\"index\":0,\"rationale\":\"a durable theme preference\",\
                 \"basis\":\"preference\"}]}",
            )
            .respond_text(
                "{\"write\":true,\"reason\":\"durable preference\",\
                 \"signals\":[\"preference\"],\"fact_indexes\":[0],\"importance\":0.75}",
            )
            .build();
        let handle = model.clone();
        let stages = StageConfig {
            plan: StageKind::Llm,
            reasoning: StageKind::Llm,
            proactive: StageKind::Llm,
            ..StageConfig::default()
        };
        let pipeline =
            StagedAnalyzer::from_config(AnalyzerConfig::default(), Some(Arc::new(model)))
                .expect("a model satisfies the model-backed stages")
                .with_stage_config(stages);
        assert_eq!(pipeline.stage_config().plan, StageKind::Llm);
        let result = pipeline
            .analyze(
                &ExtractionInput::from_text("I prefer dark mode."),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(handle.call_count(), 3);
        assert_eq!(result.plan.topic.as_deref(), Some("theme"));
        assert_eq!(result.plan.rationale, "the user states a preference");
        assert_eq!(result.facts()[0].content, "I prefer dark mode");
        assert_eq!(result.rationale(0), Some("a durable theme preference"));
        assert_eq!(
            result.facts()[0].metadata.get("rationale"),
            Some(&json!("a durable theme preference"))
        );
        let decision = result.decision.clone().unwrap();
        assert!(decision.write);
        assert_eq!(decision.reason, "durable preference");
        assert!(!result.is_degraded());
    }

    #[tokio::test]
    async fn llm_stage_kinds_without_a_model_use_the_rule_stages() {
        let stages = StageConfig {
            plan: StageKind::Llm,
            reasoning: StageKind::Llm,
            proactive: StageKind::Llm,
            ..StageConfig::default()
        };
        let with_llm = StagedAnalyzer::new(AnalyzerConfig::default()).with_stage_config(stages);
        let rule = StagedAnalyzer::new(AnalyzerConfig::default());
        let conversation = conversation();
        let left = with_llm
            .analyze(&conversation, &CallOptions::new())
            .await
            .unwrap();
        let right = rule
            .analyze(&conversation, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(left.facts().len(), right.facts().len());
        assert_eq!(left.plan.intent, right.plan.intent);
        assert_eq!(left.reasoning, right.reasoning);
        // The rule stages produce the same output, but the fallback is no longer silent:
        // the result names every stage kind that degraded.
        assert!(left.is_degraded());
        let degraded = left
            .result
            .degraded
            .as_deref()
            .expect("the no-model fallback is reported");
        for reason in [
            "llm plan stage requested without a model",
            "llm reasoning stage requested without a model",
            "llm proactive stage requested without a model",
        ] {
            assert!(degraded.contains(reason), "missing {reason}: {degraded}");
        }
        assert!(!right.is_degraded());
    }

    #[tokio::test]
    async fn a_model_backed_extractor_without_a_model_is_rejected() {
        let config = AnalyzerConfig {
            extractor: ExtractorKind::Llm,
            ..AnalyzerConfig::default()
        };
        // Construction refuses the request instead of silently running rules.
        let error = StagedAnalyzer::from_config(config.clone(), None)
            .err()
            .expect("the llm extractor needs a model");
        let Error::Unsupported(message) = error else {
            panic!("expected Unsupported, got {error:?}");
        };
        assert!(message.contains("extractor"), "{message}");
        assert!(message.contains("no model was supplied"), "{message}");
        // `new` builds the deterministic pipeline, but a run still refuses to downgrade the
        // configuration the caller asked for.
        let error = StagedAnalyzer::new(config)
            .analyze(&conversation(), &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_model_backed_classifier_without_a_model_is_rejected() {
        let config = AnalyzerConfig {
            classifier: ClassifierKind::Llm,
            ..AnalyzerConfig::default()
        };
        // The extractor stays rule-based, so construction succeeds.
        let staged = StagedAnalyzer::new(config);
        let error = staged
            .analyze(&conversation(), &CallOptions::new())
            .await
            .unwrap_err();
        let Error::Unsupported(message) = error else {
            panic!("expected Unsupported, got {error:?}");
        };
        assert!(message.contains("classifier"), "{message}");
        assert!(message.contains("no model was supplied"), "{message}");
        // With a model the same configuration runs end to end.
        let model = ScriptedModel::builder()
            .respond_text(r#"[{"info_id": "id-1", "result": "none"}]"#)
            .build();
        let staged = StagedAnalyzer::from_config(
            AnalyzerConfig {
                classifier: ClassifierKind::Llm,
                ..AnalyzerConfig::default()
            },
            Some(Arc::new(model)),
        )
        .expect("the model satisfies the classifier");
        let existing = vec![record("id-1", "the service uses postgres")];
        let result = staged
            .analyze_against(
                &ExtractionInput::from_text("the service uses mysql"),
                &existing,
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts().len(), 1);
        assert_eq!(result.verdicts.len(), 1);
        assert_eq!(result.verdicts[0].verdict.kind, ConflictKind::Unrelated);
        assert!(!result.is_degraded());
    }

    #[tokio::test]
    async fn a_custom_stage_sequence_replaces_the_built_in_one() {
        struct FixedStage;

        #[async_trait]
        impl ExtractionStage for FixedStage {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "fixed"
            }

            async fn run(
                &self,
                _input: &StageInput,
                options: &CallOptions,
            ) -> CommonResult<StageOutput> {
                options.check()?;
                Ok(StageOutput::of(self.name())
                    .with_facts(vec![ExtractedFact::semantic("fixed fact")])
                    .with_extractor("fixed"))
            }
        }

        let staged =
            StagedAnalyzer::new(AnalyzerConfig::default()).with_stages(vec![Arc::new(FixedStage)]);
        let result = staged
            .analyze(&ExtractionInput::from_text("anything"), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(result.stages, vec!["fixed"]);
        assert_eq!(result.facts().len(), 1);
        assert_eq!(result.result.extractor, "fixed");
        assert!(result.decision.is_none());
        assert_eq!(result.reasoning.len(), 0);
    }

    #[tokio::test]
    async fn cancelled_pipeline_makes_zero_model_calls() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let stages = StageConfig {
            plan: StageKind::Llm,
            reasoning: StageKind::Llm,
            proactive: StageKind::Llm,
            ..StageConfig::default()
        };
        let pipeline =
            StagedAnalyzer::from_config(AnalyzerConfig::default(), Some(Arc::new(model)))
                .expect("a model satisfies the model-backed stages")
                .with_stage_config(stages);
        let token = CancellationToken::new();
        token.cancel();
        let error = pipeline
            .analyze(
                &ExtractionInput::from_text("I prefer dark mode."),
                &CallOptions::new().with_cancellation(token),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn consolidate_matches_the_plain_analyzer() {
        let config = AnalyzerConfig::default();
        let staged = StagedAnalyzer::new(config.clone());
        let incoming = vec![ExtractedFact::semantic("the service uses postgres")];
        let existing = vec![record("id-1", "the service uses postgres")];
        let dedupe = DedupeConfig::default();
        let planned = staged.consolidate(&incoming, &existing, &dedupe);
        let plain = MemoryAnalyzer::rule(config).consolidate(&incoming, &existing, &dedupe);
        assert_eq!(planned, plain);
        assert!(matches!(
            planned.actions[0],
            ConsolidationAction::Merge { fact_index: 0, .. }
        ));
    }

    #[tokio::test]
    async fn staged_result_round_trips_through_json() {
        let staged = StagedAnalyzer::from_engine(&MemoryEngineConfig::default(), None)
            .expect("the engine default asks for rules")
            .analyze(&conversation(), &CallOptions::new())
            .await
            .unwrap();
        let json = serde_json::to_value(&staged).unwrap();
        for key in [
            "result",
            "plan",
            "decision",
            "reasoning",
            "verdicts",
            "stages",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["stages"][0], json!("plan"));
        let restored: StagedResult = serde_json::from_value(json).unwrap();
        assert_eq!(restored.facts().len(), staged.facts().len());
        assert_eq!(restored.stages, staged.stages);
        assert_eq!(restored.plan, staged.plan);
        let tolerant: StagedResult = serde_json::from_value(json!({"unknown": 1})).unwrap();
        assert_eq!(tolerant.facts().len(), 0);
        assert!(tolerant.decision.is_none());

        let verdict = FactVerdict {
            fact_index: 1,
            verdict: ConflictVerdict {
                kind: ConflictKind::Conflicting,
                target_id: Some("id-1".to_owned()),
                reason: Some("same subject".to_owned()),
                merged_content: Some("merged".to_owned()),
                degraded: None,
            },
        };
        let json = serde_json::to_value(&verdict).unwrap();
        let restored: FactVerdict = serde_json::from_value(json).unwrap();
        assert_eq!(restored, verdict);
        assert_eq!(FactVerdict::default().fact_index, 0);
    }

    #[test]
    fn stage_config_validation() {
        let config = StageConfig::default();
        assert_eq!(config.plan, StageKind::Rule);
        assert_eq!(config.reasoning, StageKind::Rule);
        assert!(config.validate().is_ok());
        assert!(StageConfig {
            reasoning_key: " ".to_owned(),
            ..StageConfig::default()
        }
        .validate()
        .is_err());
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["plan"], json!("rule"));
        let threshold = json["proactive_config"]["importance_threshold"]
            .as_f64()
            .expect("a numeric threshold");
        assert!((threshold - 0.7).abs() < 1e-6);
        let restored: StageConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }
}
