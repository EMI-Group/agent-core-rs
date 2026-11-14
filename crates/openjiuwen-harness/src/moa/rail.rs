//! The mixture-of-agents harness rail: publishes the consultation protocol and keeps the
//! per-invocation consultation state, the last outcome and the bounded history.
//!
//! Ports the lifecycle half of `openjiuwen/harness/rails/moa_rail.py`. Python registers a
//! `consult_reference_models` ability on the agent and routes the consultation through a
//! `ContextVar`; the Rust port has no ability manager, so [`MoaRail`] is a pure context
//! mutator and [`MoaRail::consult`] is the explicit host call that replaces the Python tool.
//! The state that Python keeps in a `ContextVar` is owned by the rail here: one
//! [`MoaState`] per outer invocation, installed by `before_invoke` and dropped by
//! `after_invoke`.
//!
//! # Hooks
//!
//! - [`HarnessRail::before_invoke`] installs a fresh [`MoaState`] for the invocation,
//!   publishes [`prompts::moa_section`] into the prompt-section channel
//!   ([`crate::rails::publish_prompt_section`]) and records the routing snapshot plus the
//!   last outcome under [`MOA_STATE_EXTRA_KEY`] and [`MOA_OUTCOME_EXTRA_KEY`].
//! - [`HarnessRail::before_model_call`] re-publishes the section (upsert by name) and the
//!   current outcome, so the outcome survives every `ReAct` iteration of the invocation.
//! - [`HarnessRail::after_invoke`] retires the installed state: it marks the state closed,
//!   closes the proposal store with `outer_end`, and clears the installed state.
//!
//! The rail never calls a model: the hooks only read configuration and consultation state.
//! [`MoaRail::consult`] is where the reference models are asked, and it always runs because
//! the host asked for it, mirroring Python's ability-manager tool. The remaining hooks keep
//! their defaults.
//!
//! # Failure policy
//!
//! A disabled configuration (`MoaConfig::enabled == false`) publishes nothing and writes no
//! diagnostics, and `consult` answers [`Error::Config`]. No hook takes a lock across an
//! `.await`: every lock guard lives in a synchronous statement and is dropped before the
//! next suspension point.
// The public names (`MoaRail`, `MOA_PROMPT_SECTION`, ...) mirror the Python contract and
// intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::path::Path;

use parking_lot::RwLock;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::BaseMessage;

use crate::prompts::PromptSection;
use crate::rails::{last_user_text, publish_prompt_section, HarnessRail, HarnessRailContext};

use super::aggregate::{AggregationStrategy, Aggregator};
use super::config::MoaConfig;
use super::consultation;
use super::prompts;
use super::state::{
    ConsultationHistory, ConsultationOutcome, ConsultationRecord, MoaSnapshot, MoaState,
};
use super::verifier;

/// Stable name of the rail; registering two rails with one name replaces the first.
pub const MOA_RAIL_NAME: &str = "moa";

/// Name of the prompt section the rail publishes.
///
/// Equal to [`prompts::MOA_PROMPT_SECTION_NAME`], so a caller collecting the assembled
/// prompt finds the consultation protocol under the same key in both paths.
pub const MOA_PROMPT_SECTION: &str = "moa";

/// Key in [`HarnessRailContext::extra_mut`] holding the invocation [`MoaSnapshot`].
///
/// The value is the JSON encoding of [`MoaSnapshot`]; it carries routing metadata only and
/// never reference text.
pub const MOA_STATE_EXTRA_KEY: &str = "harness.moa_state";

/// Key in [`HarnessRailContext::extra_mut`] holding the last [`ConsultationOutcome`].
///
/// The value is the JSON encoding of the outcome. A host drains it after firing the hooks to
/// render the ranking or the integrated answer; the rail itself never emits it anywhere.
pub const MOA_OUTCOME_EXTRA_KEY: &str = "harness.moa_outcome";

/// Language a run uses until [`MoaRail::with_language`] replaces it.
const DEFAULT_LANGUAGE: &str = "en";

/// Harness rail that publishes the `MoA` prompt section and owns the consultation state.
///
/// Build one with [`MoaRail::new`], register it on a
/// [`crate::rails::HarnessRailRegistry`] (priority [`crate::rails::rail_priority::MOA`] by
/// default), and call [`MoaRail::consult`] when the host decides the references should be
/// asked. The rail is `Send + Sync`, so it can be shared across tasks.
pub struct MoaRail {
    /// Reference models, routing policy and budgets.
    config: MoaConfig,
    /// Invocation-local consultation state, installed by `before_invoke`.
    state: RwLock<Option<MoaState>>,
    /// Outcome of the latest consultation, kept across invocations.
    outcome: RwLock<Option<ConsultationOutcome>>,
    /// Bounded consultation history, oldest first.
    history: RwLock<ConsultationHistory>,
    /// Aggregation strategy applied by [`Self::consult`].
    aggregator: Aggregator,
    /// Preferred prompt language.
    language: String,
}

impl fmt::Debug for MoaRail {
    /// Reports the configuration (which redacts model handles) and the retained record count.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoaRail")
            .field("config", &self.config)
            .field("language", &self.language)
            .field("history_records", &self.history.read().len())
            .finish_non_exhaustive()
    }
}

impl MoaRail {
    /// Builds a rail for `config` after applying the Python `__post_init__` checks.
    ///
    /// The history retains `config.history_limit` records and the prompt language starts at
    /// `"en"`; the aggregation strategy follows the configuration: an
    /// [`MoaConfig::aggregator_model`] selects [`AggregationStrategy::Synthesis`], otherwise
    /// the deterministic, fully offline [`AggregationStrategy::Labelled`] applies.
    ///
    /// # Errors
    ///
    /// Returns the [`MoaConfig::validate`] error when the configuration is invalid (no
    /// reference model, conflicting routing pieces, out-of-range budgets, or an unusable
    /// consultation skill directory).
    pub fn new(config: MoaConfig) -> Result<Self> {
        config.validate()?;
        let strategy = if config.aggregator_model.is_some() {
            AggregationStrategy::Synthesis
        } else {
            AggregationStrategy::Labelled
        };
        let history = ConsultationHistory::new(config.history_limit);
        Ok(Self {
            config,
            state: RwLock::new(None),
            outcome: RwLock::new(None),
            history: RwLock::new(history),
            aggregator: Aggregator::new(strategy),
            language: DEFAULT_LANGUAGE.to_owned(),
        })
    }

    /// Returns this rail preferring `language` for the prompt section.
    ///
    /// The section itself is bilingual, so the language only documents the caller's intent;
    /// it is never used to select text.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// The configuration this rail was built from.
    #[must_use]
    pub fn config(&self) -> &MoaConfig {
        &self.config
    }

    /// The consultation protocol section for `language` and the configured mode.
    #[must_use]
    pub fn prompt_section(&self, language: &str) -> PromptSection {
        prompts::moa_section(language, self.config.mode)
    }

    /// The latest consultation outcome, when one ran through this rail.
    #[must_use]
    pub fn last_outcome(&self) -> Option<ConsultationOutcome> {
        self.outcome.read().clone()
    }

    /// The routing diagnostics of the installed invocation state, when one is installed.
    #[must_use]
    pub fn snapshot(&self) -> Option<MoaSnapshot> {
        self.state.read().as_ref().map(MoaState::snapshot)
    }

    /// The retained consultation records, oldest first.
    #[must_use]
    pub fn history(&self) -> Vec<ConsultationRecord> {
        self.history.read().records().to_vec()
    }

    /// Asks every configured reference model for an independent proposal about the last user
    /// message of `messages`, ranks the retained proposals and aggregates one answer.
    ///
    /// The `problem` handed to the ranking and the aggregator is
    /// [`last_user_text`] (empty when the conversation carries no user message); the
    /// proposers receive the flattened transcript produced by
    /// [`crate::moa::prompts::consultation_transcript`], exactly like
    /// [`consultation::consult`].
    ///
    /// The call updates the installed invocation state (`reference_calls` grows by the number
    /// of asked reference models), stores the outcome as [`Self::last_outcome`] and appends
    /// one record to the bounded history.
    ///
    /// # Errors
    ///
    /// - [`Error::Config`] with `MoA consultation is disabled` when
    ///   [`MoaConfig::enabled`] is `false`.
    /// - [`Error::Llm`] when every reference model fails, or when
    ///   [`AggregationStrategy::Synthesis`] is configured but the aggregator fails or answers
    ///   with nothing.
    /// - [`Error::Cancelled`] when `cancel` fires before the batch completes.
    /// - [`Error::Timeout`] when a synthesis aggregator exceeds
    ///   [`MoaConfig::reference_timeout`]; a single reference timing out is recorded as a
    ///   [`super::state::ProposerFailure`] instead.
    pub async fn consult(
        &self,
        messages: &[BaseMessage],
        cancel: &CancellationToken,
    ) -> Result<ConsultationOutcome> {
        if !self.config.enabled {
            return Err(Error::config("MoA consultation is disabled"));
        }
        let problem = last_user_text(messages).unwrap_or_default();
        let batch = consultation::consult(&self.config, messages, cancel).await?;
        let ranking = verifier::rank_with_config(&self.config, &problem, &batch.sources).await?;
        let aggregate = self
            .aggregator
            .aggregate(&self.config, &problem, &batch.sources, cancel)
            .await?;

        let outcome = ConsultationOutcome {
            proposals: batch.sources,
            success_count: batch.success_count,
            failed_count: batch.failures.len(),
            failures: batch.failures,
            ranking,
            aggregate: Some(aggregate),
            verifier_score: None,
        };

        {
            let mut state = self.state.write();
            if let Some(state) = state.as_mut() {
                state.reference_calls += outcome.success_count + outcome.failed_count;
            }
        }
        *self.outcome.write() = Some(outcome.clone());
        self.history
            .write()
            .push(ConsultationRecord::new(problem, outcome.clone()));
        tracing::debug!(
            success = outcome.success_count,
            failed = outcome.failed_count,
            ranked = outcome.ranking.len(),
            "stored MoA consultation outcome"
        );
        Ok(outcome)
    }

    /// Writes the retained history to `path` as pretty JSON.
    ///
    /// The records are cloned under a brief lock, which is released before the file is
    /// written, so no guard is held across the `.await`.
    ///
    /// # Errors
    ///
    /// Returns an error when the history cannot be serialized or the file cannot be written.
    pub async fn save_history(&self, path: &Path) -> Result<()> {
        let history = self.history.read().clone();
        history.save(path).await
    }

    /// Replaces the retained records with a history written by
    /// [`ConsultationHistory::save`].
    ///
    /// The configured [`MoaConfig::history_limit`] of this rail is preserved; only the
    /// records are adopted from the file. The file is read before the lock is taken, so no
    /// guard is held across the `.await`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `path` does not exist, an [`Error::External`] I/O error
    /// when it cannot be read, and [`Error::Validation`] when the file is not a consultation
    /// history.
    pub async fn load_history(&self, path: &Path) -> Result<()> {
        let loaded = ConsultationHistory::load(path).await?;
        let mut history = self.history.write();
        history.records = loaded.records;
        Ok(())
    }

    /// Installs a fresh invocation state for `problem` and returns its snapshot.
    fn install_state(&self, problem: impl Into<String>) -> MoaSnapshot {
        let state = MoaState::new(problem);
        let snapshot = state.snapshot();
        *self.state.write() = Some(state);
        snapshot
    }

    /// Publishes the consultation section into the context's prompt-section channel.
    fn publish_section(&self, ctx: &mut HarnessRailContext) {
        publish_prompt_section(ctx, &self.prompt_section(&self.language));
    }

    /// Records the last outcome under [`MOA_OUTCOME_EXTRA_KEY`], when one exists.
    fn publish_outcome(&self, ctx: &mut HarnessRailContext) {
        let Some(outcome) = self.last_outcome() else {
            return;
        };
        if let Ok(value) = serde_json::to_value(&outcome) {
            ctx.extra_mut()
                .insert(MOA_OUTCOME_EXTRA_KEY.to_owned(), value);
        }
    }
}

#[async_trait::async_trait]
impl HarnessRail for MoaRail {
    fn name(&self) -> &str {
        MOA_RAIL_NAME
    }

    fn priority(&self) -> i32 {
        self.config.rail_priority
    }

    async fn before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let problem = last_user_text(&ctx.base.messages).unwrap_or_default();
        let snapshot = self.install_state(problem);
        self.publish_section(ctx);
        if let Ok(value) = serde_json::to_value(snapshot) {
            ctx.extra_mut()
                .insert(MOA_STATE_EXTRA_KEY.to_owned(), value);
        }
        self.publish_outcome(ctx);
        Ok(())
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        // Upsert by section name: one section per invocation, never one per iteration.
        self.publish_section(ctx);
        self.publish_outcome(ctx);
        Ok(())
    }

    async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        // `take` ends the guard with the statement, so no lock survives the hook.
        let state = self.state.write().take();
        if let Some(mut state) = state {
            state.closed = true;
            state.proposals.close("outer_end");
        }
        // Keep the context tidy for a caller that reuses it across invocations.
        ctx.extra_mut().remove(MOA_STATE_EXTRA_KEY);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use openjiuwen_agent::RailEvent;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::Model;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    use crate::rails::{prompt_sections, rail_priority};

    use super::*;
    use crate::moa::config::{MoaMode, DEFAULT_HISTORY_LIMIT};
    use crate::moa::state::ConsultationOutcome;

    /// Wraps a scripted model as a reference handle.
    fn reference(model: ScriptedModel) -> Arc<dyn Model> {
        Arc::new(model)
    }

    /// Builds a valid configuration over one answering reference.
    fn config() -> MoaConfig {
        MoaConfig::new(vec![reference(ScriptedModel::text("proposal"))])
    }

    /// Builds a context carrying one user message.
    fn context(text: &str) -> HarnessRailContext {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.messages = vec![BaseMessage::user(text)];
        ctx
    }

    #[test]
    fn name_and_priority_come_from_the_configuration() {
        let rail = MoaRail::new(config()).unwrap();
        assert_eq!(rail.name(), MOA_RAIL_NAME);
        assert_eq!(rail.name(), "moa");
        assert_eq!(rail.priority(), rail_priority::MOA);
        assert_eq!(rail.priority(), 10);
        assert_eq!(rail.config().mode, MoaMode::AggregatorOnDemand);
        assert_eq!(MOA_PROMPT_SECTION, prompts::MOA_PROMPT_SECTION_NAME);

        let custom = MoaRail::new(config().with_rail_priority(70)).unwrap();
        assert_eq!(custom.priority(), 70);
        assert_eq!(custom.config().history_limit, DEFAULT_HISTORY_LIMIT);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let error = MoaRail::new(MoaConfig::new(Vec::new())).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.message(), "MoA requires at least one reference model");
    }

    // `PromptSection` is not `PartialEq`, so emptiness is the only available assertion.
    #[allow(clippy::assert_is_empty)]
    #[tokio::test]
    async fn disabled_rail_publishes_nothing() {
        let rail = MoaRail::new(config().with_enabled(false)).unwrap();
        let mut ctx = context("fix the build");

        rail.before_invoke(&mut ctx).await.unwrap();
        assert!(prompt_sections(&ctx).is_empty());
        assert!(ctx.extra().get(MOA_STATE_EXTRA_KEY).is_none());
        assert!(ctx.extra().get(MOA_OUTCOME_EXTRA_KEY).is_none());
        assert!(rail.snapshot().is_none());

        rail.before_model_call(&mut ctx).await.unwrap();
        assert!(prompt_sections(&ctx).is_empty());
        assert!(ctx.extra().get(MOA_OUTCOME_EXTRA_KEY).is_none());

        rail.after_invoke(&mut ctx).await.unwrap();
        assert!(rail.snapshot().is_none());
    }

    #[tokio::test]
    async fn enabled_rail_publishes_one_bilingual_section_and_a_snapshot() {
        let rail = MoaRail::new(config()).unwrap();
        let mut ctx = context("fix the build");

        rail.before_invoke(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, MOA_PROMPT_SECTION);
        let languages: Vec<&str> = sections[0].content.keys().map(String::as_str).collect();
        assert_eq!(languages, ["cn", "en"]);
        assert!(sections[0].content["en"].contains("Reference Consultation"));
        assert!(sections[0].content["cn"].contains("参考模型咨询"));

        let value = ctx.extra().get(MOA_STATE_EXTRA_KEY).unwrap();
        let snapshot: MoaSnapshot = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(snapshot.reference_calls, 0);
        assert_eq!(snapshot.completed_turns, 0);
        assert_eq!(rail.snapshot(), Some(snapshot));
        assert!(rail.last_outcome().is_none());
    }

    #[tokio::test]
    async fn before_model_call_upserts_the_section_and_outcome() {
        let rail = MoaRail::new(config()).unwrap();
        let mut ctx = context("fix the build");

        rail.before_invoke(&mut ctx).await.unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(prompt_sections(&ctx).len(), 1);
        assert!(ctx.extra().get(MOA_OUTCOME_EXTRA_KEY).is_none());

        let outcome = rail
            .consult(&ctx.base.messages, &CancellationToken::new())
            .await
            .unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(prompt_sections(&ctx).len(), 1);
        let stored: ConsultationOutcome =
            serde_json::from_value(ctx.extra()[MOA_OUTCOME_EXTRA_KEY].clone()).unwrap();
        assert_eq!(stored, outcome);
    }

    #[tokio::test]
    async fn consult_records_counts_ranking_history_and_outcome() {
        let rail = MoaRail::new(MoaConfig::new(vec![
            reference(ScriptedModel::text(
                "1. Fix src/main.rs\n2. Run cargo test to verify",
            )),
            reference(ScriptedModel::text("Rebuild the project.")),
            reference(ScriptedModel::new([ScriptedTurn::fail("boom")])),
        ]))
        .unwrap();
        let mut ctx = context("fix the failing build");
        rail.before_invoke(&mut ctx).await.unwrap();

        let outcome = rail
            .consult(&ctx.base.messages, &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(outcome.success_count, 2);
        assert_eq!(outcome.failed_count, 1);
        assert_eq!(outcome.failures.len(), 1);
        assert!(outcome.failures[0].error.contains("boom"));
        assert_eq!(outcome.proposals.len(), 2);
        assert_eq!(outcome.ranking.len(), outcome.proposals.len());
        let aggregate = outcome.aggregate.as_deref().unwrap();
        assert!(aggregate.contains("Integrated answer"));
        assert!(aggregate.contains("Fix src/main.rs"));

        assert_eq!(rail.history().len(), 1);
        assert_eq!(rail.history()[0].problem, "fix the failing build");
        assert_eq!(rail.history()[0].outcome, outcome);
        assert_eq!(rail.last_outcome(), Some(outcome));
        assert_eq!(rail.snapshot().unwrap().reference_calls, 3);
    }

    #[tokio::test]
    async fn disabled_consult_is_a_config_error() {
        let rail = MoaRail::new(config().with_enabled(false)).unwrap();
        let error = rail
            .consult(&[BaseMessage::user("fix it")], &CancellationToken::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.message(), "MoA consultation is disabled");
        assert!(rail.last_outcome().is_none());
        assert_eq!(rail.history(), Vec::<ConsultationRecord>::new());
    }

    #[tokio::test]
    async fn pre_cancelled_consult_stops_before_any_proposal() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let rail = MoaRail::new(config()).unwrap();
        let error = rail
            .consult(&[BaseMessage::user("fix it")], &cancel)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(rail.history(), Vec::<ConsultationRecord>::new());
    }

    #[tokio::test]
    async fn after_invoke_retires_the_installed_state() {
        let rail = MoaRail::new(config()).unwrap();
        let mut ctx = context("fix the build");
        rail.before_invoke(&mut ctx).await.unwrap();
        assert!(rail.snapshot().is_some());
        assert!(ctx.extra().contains_key(MOA_STATE_EXTRA_KEY));

        rail.after_invoke(&mut ctx).await.unwrap();
        assert!(rail.snapshot().is_none());
        assert!(!ctx.extra().contains_key(MOA_STATE_EXTRA_KEY));
    }

    #[tokio::test]
    async fn history_round_trips_through_a_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("moa-history.json");

        let rail = MoaRail::new(config()).unwrap();
        rail.consult(
            &[BaseMessage::user("fix the build")],
            &CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(rail.history().len(), 1);
        rail.save_history(&path).await.unwrap();

        let restored = MoaRail::new(config()).unwrap();
        assert_eq!(restored.history(), Vec::<ConsultationRecord>::new());
        restored.load_history(&path).await.unwrap();
        assert_eq!(restored.history(), rail.history());
        assert_eq!(restored.history()[0].problem, "fix the build");
        assert_eq!(restored.config().history_limit, DEFAULT_HISTORY_LIMIT);

        let missing = MoaRail::new(config()).unwrap();
        let error = missing
            .load_history(&temp.path().join("absent.json"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
    }
}
