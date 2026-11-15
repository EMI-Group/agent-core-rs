//! Staged memory extraction: plan → extract → reason → proactive → conflict.
//!
//! The Python `process/extract` pipeline is a fixed sequence: `MemoryAnalyzer` decides
//! whether the conversation holds key information, which variables and summary to
//! capture; `LongTermMemoryExtractor` then runs the staged prompt (anaphora resolution,
//! subject locking, time reasoning, content extraction) and `MemUpdateChecker`
//! classifies the result against what is already stored. This module keeps that shape as
//! composable [`ExtractionStage`]s so callers can inspect, replace, or extend each step
//! while the deterministic implementations stay the default:
//!
//! 1. [`RulePlanStage`](crate::analyzer::RulePlanStage) / `LlmPlanStage` — decide which
//!    operations to run, from which turns, and why ([`ExtractionPlan`]);
//! 2. `ExtractStage` — run the configured [`FactExtractor`](crate::analyzer::FactExtractor)
//!    over the planned turns;
//! 3. `RuleReasoningStage` / `LlmReasoningStage` — attach a rationale to every fact
//!    ([`FactReasoning`]);
//! 4. `RuleProactiveStage` / `LlmProactiveStage` — decide whether memory should be written
//!    proactively ([`ProactiveDecision`]);
//! 5. `ConflictStage` — classify the facts against stored records (`FactVerdict`).
//!
//! [`StagedAnalyzer`](crate::analyzer::StagedAnalyzer) assembles the default sequence and
//! returns a [`StagedResult`](crate::analyzer::StagedResult); the plain
//! [`MemoryAnalyzer`](crate::analyzer::MemoryAnalyzer) API is unchanged, and its
//! `analyze`/`consolidate` results remain the reference the staged pipeline reproduces.
//!
//! # Degradation contract
//!
//! Every model-backed stage takes a deterministic implementation as its fallback. A
//! provider error, an empty answer, or an unparseable answer is retried
//! `AnalyzerConfig::model_retries + 1` times; afterwards the deterministic stage runs and
//! the stage output carries a `degraded` reason ("… fell back to rules: …"). A cancelled
//! [`CallOptions`] is checked before the first model call, so an already-cancelled call
//! performs zero model invocations.
//!
//! A *missing* model is reported too, never just logged: a [`StageKind::Llm`] stage whose
//! model is absent runs its rule implementation and the pipeline records `llm <stage>
//! stage requested without a model` in
//! [`StagedResult::result`](crate::analyzer::StagedResult::result)'s
//! [`ExtractionResult::degraded`](crate::analyzer::ExtractionResult::degraded) field, while
//! a configuration whose *extractor* or *classifier* asks for the model returns
//! [`openjiuwen_common::error::Error::Unsupported`] from
//! [`StagedAnalyzer::from_config`](crate::analyzer::StagedAnalyzer::from_config) or from
//! the run itself.
mod overrides;
mod pipeline;
mod plan;
mod proactive;
mod reasoning;

pub use overrides::ScopeAnalyzerOverrides;
pub use pipeline::{
    ConflictStage, ExtractStage, FactVerdict, StageConfig, StageKind, StagedAnalyzer, StagedResult,
};
pub use plan::{
    parse_plan_response, ExtractionPlan, LlmPlanStage, PlanIntent, PlanOperation, RulePlanStage,
    PLAN_JSON_SCHEMA, PLAN_SYSTEM_PROMPT, SUMMARY_TOKEN_THRESHOLD,
};
pub use proactive::{
    parse_proactive_response, LlmProactiveStage, ProactiveConfig, ProactiveDecision,
    ProactiveSignal, RuleProactiveStage, PROACTIVE_JSON_SCHEMA, PROACTIVE_SYSTEM_PROMPT,
};
pub use reasoning::{
    parse_reasoning_response, FactReasoning, LlmReasoningStage, ReasoningBasis, RuleReasoningStage,
    DEFAULT_REASONING_KEY, REASONING_JSON_SCHEMA, REASONING_SYSTEM_PROMPT,
};

use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::fact::{ExtractedFact, ExtractionInput};
use crate::record::{CallOptions, MemoryRecord};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

/// The state a stage receives and returns.
///
/// A stage reads the conversation, the effective configuration, the plan accumulated so
/// far, and any facts/candidates; it returns a [`StageOutput`] holding only what it
/// changed, which the pipeline merges back into the state. The type is plain `serde`
/// data apart from the config, so a pipeline state can be logged or replayed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StageInput {
    /// The conversation being analyzed (newest turn last).
    pub conversation: ExtractionInput,
    /// The effective analyzer configuration (base config plus scope overrides).
    pub config: AnalyzerConfig,
    /// The plan the plan stage produced; the empty plan before that.
    pub plan: ExtractionPlan,
    /// Name of the extractor that produced [`StageInput::facts`] (`rule`/`llm`).
    pub extractor: String,
    /// Facts extracted so far, in extractor order.
    pub facts: Vec<ExtractedFact>,
    /// The rationale attached to each fact, in fact order.
    pub reasoning: Vec<FactReasoning>,
    /// Existing records the conflict stage classifies against.
    pub candidates: Vec<MemoryRecord>,
    /// One verdict per fact/candidate pair, in fact order then candidate order.
    pub verdicts: Vec<FactVerdict>,
    /// The proactive-write decision, once the proactive stage ran.
    pub decision: Option<ProactiveDecision>,
}

impl StageInput {
    /// Builds a state for `conversation` under `config`.
    #[must_use]
    pub fn new(conversation: ExtractionInput, config: AnalyzerConfig) -> Self {
        Self {
            conversation,
            config,
            ..Self::default()
        }
    }

    /// Returns the state with the conflict stage's candidate records replaced.
    #[must_use]
    pub fn with_candidates(mut self, candidates: Vec<MemoryRecord>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Returns the number of facts accumulated so far.
    #[must_use]
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }
}

/// What one stage changed, plus any documented degradation.
///
/// A stage only fills the fields it owns: the plan stage returns `plan`, the extraction
/// stage `facts`, the reasoning stage `facts` and `reasoning`, the proactive stage
/// `decision`, and the conflict stage `verdicts`. Unset fields leave the pipeline state
/// untouched.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StageOutput {
    /// The stage's name, as reported by [`ExtractionStage::name`].
    pub stage: String,
    /// The plan this stage produced.
    pub plan: Option<ExtractionPlan>,
    /// The facts this stage produced.
    pub facts: Option<Vec<ExtractedFact>>,
    /// Name of the extractor that produced the facts.
    pub extractor: Option<String>,
    /// The rationale per fact.
    pub reasoning: Vec<FactReasoning>,
    /// The proactive-write decision.
    pub decision: Option<ProactiveDecision>,
    /// The conflict verdicts.
    pub verdicts: Vec<FactVerdict>,
    /// Why a deterministic fallback ran, when one did.
    pub degraded: Option<String>,
}

impl StageOutput {
    /// Builds an empty output attributed to `stage`.
    #[must_use]
    pub fn of(stage: impl Into<String>) -> Self {
        Self {
            stage: stage.into(),
            ..Self::default()
        }
    }

    /// Returns the output with `plan` set.
    #[must_use]
    pub fn with_plan(mut self, plan: ExtractionPlan) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Returns the output with `facts` set.
    #[must_use]
    pub fn with_facts(mut self, facts: Vec<ExtractedFact>) -> Self {
        self.facts = Some(facts);
        self
    }

    /// Returns the output with `reasoning` set.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: Vec<FactReasoning>) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Returns the output with `extractor` set.
    #[must_use]
    pub fn with_extractor(mut self, extractor: impl Into<String>) -> Self {
        self.extractor = Some(extractor.into());
        self
    }

    /// Returns the output with `decision` set.
    #[must_use]
    pub fn with_decision(mut self, decision: ProactiveDecision) -> Self {
        self.decision = Some(decision);
        self
    }

    /// Returns the output with `verdicts` set.
    #[must_use]
    pub fn with_verdicts(mut self, verdicts: Vec<FactVerdict>) -> Self {
        self.verdicts = verdicts;
        self
    }

    /// Returns the output marked as a deterministic fallback.
    #[must_use]
    pub fn degraded(mut self, reason: impl Into<String>) -> Self {
        self.degraded = Some(reason.into());
        self
    }

    /// Returns `true` when a deterministic fallback produced this output.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded.is_some()
    }

    /// Merges this output into `state`.
    ///
    /// Only the fields the stage set are copied; `degraded` is not stored on the state
    /// (the pipeline records it on the result).
    pub fn apply(self, state: &mut StageInput) {
        if let Some(plan) = self.plan {
            state.plan = plan;
        }
        if let Some(facts) = self.facts {
            state.facts = facts;
        }
        if let Some(extractor) = self.extractor {
            state.extractor = extractor;
        }
        if !self.reasoning.is_empty() {
            state.reasoning = self.reasoning;
        }
        if let Some(decision) = self.decision {
            state.decision = Some(decision);
        }
        if !self.verdicts.is_empty() {
            state.verdicts = self.verdicts;
        }
    }
}

/// One step of a staged memory analysis.
///
/// Implementations are `Send + Sync`, must check [`CallOptions::check`] before any model
/// call, and must never touch storage: a stage only transforms the state it is given.
#[async_trait]
pub trait ExtractionStage: Send + Sync {
    /// The stage name used in diagnostics and [`StageOutput::stage`].
    fn name(&self) -> &str;

    /// Runs the stage over `input`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Cancelled`] when `options` is cancelled,
    /// [`openjiuwen_common::error::Error::Validation`] for an invalid config or input, and
    /// the model provider's error when a model-backed stage cannot recover.
    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput>;
}

/// Returns a verdict list's degradation reasons, joined in order.
pub(crate) fn degraded_reasons<'a>(reasons: impl IntoIterator<Item = &'a str>) -> Option<String> {
    let mut collected: Vec<&str> = Vec::new();
    for reason in reasons {
        if !reason.trim().is_empty() {
            collected.push(reason);
        }
    }
    if collected.is_empty() {
        None
    } else {
        Some(collected.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::fact::ConversationTurn;
    use crate::record::{MemoryScope, MemoryType};

    fn fact(content: &str) -> ExtractedFact {
        ExtractedFact::semantic(content)
    }

    fn state() -> StageInput {
        let mut conversation = ExtractionInput::default();
        conversation.push_turn("user", "I prefer dark mode");
        StageInput::new(conversation, AnalyzerConfig::default())
    }

    #[test]
    fn stage_input_round_trips_and_tolerates_unknown_fields() {
        let mut state = state();
        state.plan = ExtractionPlan::default();
        state.facts = vec![fact("I prefer dark mode")];
        state.candidates = vec![MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "Alice prefers light mode",
            MemoryType::UserProfile,
        )];
        let json = serde_json::to_value(&state).unwrap();
        assert!(json.get("conversation").is_some());
        assert!(json.get("config").is_some());
        let restored: StageInput = serde_json::from_value(json).unwrap();
        assert_eq!(restored.fact_count(), 1);
        assert_eq!(restored.candidates.len(), 1);
        let tolerant: StageInput =
            serde_json::from_value(serde_json::json!({"unknown": 1})).unwrap();
        assert_eq!(tolerant.fact_count(), 0);
        assert!(tolerant.decision.is_none());
    }

    #[test]
    fn stage_output_applies_only_what_it_sets() {
        let mut state = state();
        state.facts = vec![fact("kept")];
        let output = StageOutput::of("extract")
            .with_facts(vec![fact("one"), fact("two")])
            .with_reasoning(vec![FactReasoning::rule(0, "why")])
            .degraded("fell back");
        assert_eq!(output.stage, "extract");
        assert!(output.is_degraded());
        let output_clone = output.clone();
        output.apply(&mut state);
        assert_eq!(state.fact_count(), 2);
        assert_eq!(state.reasoning.len(), 1);
        assert_eq!(state.plan, ExtractionPlan::default());
        // An output that sets nothing leaves the state untouched.
        let mut untouched = state.clone();
        StageOutput::of("noop").apply(&mut untouched);
        assert_eq!(untouched.fact_count(), 2);
        assert_eq!(untouched.reasoning.len(), 1);
        // The plan is only replaced when the output carries one.
        let mut stated =
            StageInput::new(ExtractionInput::from_text("x"), AnalyzerConfig::default());
        let plan = ExtractionPlan::default();
        StageOutput::of("plan")
            .with_plan(plan.clone())
            .apply(&mut stated);
        assert_eq!(stated.plan, plan);
        // `StageOutput` itself is plain serde data.
        let json = serde_json::to_value(&output_clone).unwrap();
        assert_eq!(json["stage"], serde_json::json!("extract"));
        let restored: StageOutput = serde_json::from_value(json).unwrap();
        assert_eq!(restored.facts.as_ref().map(Vec::len), Some(2));
        assert_eq!(restored.degraded.as_deref(), Some("fell back"));
    }

    #[test]
    fn reason_joining() {
        assert_eq!(degraded_reasons(std::iter::empty()), None);
        assert_eq!(
            degraded_reasons(["a fell back", "", "b fell back"]),
            Some("a fell back; b fell back".to_owned())
        );
    }

    #[test]
    fn with_candidates_replaces_the_records() {
        let candidate = MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "the service uses Postgres",
            MemoryType::SemanticMemory,
        );
        let state = state().with_candidates(vec![candidate]);
        assert_eq!(state.candidates.len(), 1);
        let _ = ConversationTurn::new("user", "x");
    }
}
