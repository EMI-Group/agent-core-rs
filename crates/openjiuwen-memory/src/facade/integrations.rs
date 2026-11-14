//! Facade entry points for the subsystems that sit beside the record path.
//!
//! [`LongTermMemory`] wires record CRUD, recall, messages, variables, and scope config;
//! this module adds the analyzer ([`LongTermMemory::remember`] /
//! [`LongTermMemory::consolidate`]), the provider SPI ([`LongTermMemory::provider`]),
//! dreaming, team memory, schema migrations, and the local coding stack over the same
//! injected stores. Nothing here starts a background task or reads a clock of its own:
//! scheduling, sweeps, purges, and the application of a plan stay with the caller.
//!
//! `remember` and `consolidate` run the staged pipeline
//! ([`StagedAnalyzer`](crate::analyzer::StagedAnalyzer): plan → extract → reason →
//! proactive → conflict) over the engine configuration plus the scope's persisted
//! analyzer overrides
//! ([`LongTermMemory::resolve_scope_overrides`](super::LongTermMemory::resolve_scope_overrides)).
//! Two documented rules keep the behaviour stable:
//!
//! - an analyzer supplied through
//!   [`LongTermMemoryBuilder::analyzer`](super::LongTermMemoryBuilder::analyzer) keeps
//!   owning extraction while the scope carries no overrides, because the staged pipeline
//!   builds its extract stage from the configuration and cannot reach a caller's custom
//!   extractor;
//! - the proactive stage's decision is advisory unless the scope carries explicit
//!   overrides (the pipeline documents it as advisory); with overrides in effect the
//!   decision gates the write and [`RememberReport::gated`] records why.
//!
//! [`LongTermMemory::analyze_for_write`] is that whole analysis pass; `remember` writes
//! its facts as records, while
//! [`LongTermMemory::remember_with_assignments`](crate::facade::LongTermMemory::remember_with_assignments)
//! writes the variable-assignment facts as real
//! [`MemoryType::Variable`](crate::record::MemoryType::Variable) projections. Both go
//! through the facade's single [`record_input`] conversion, which keeps a fact's tags.
use std::path::Path;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::analyzer::{
    AnalyzerConfig, ConsolidationPlan, ExtractedFact, ExtractionInput, ExtractorKind,
    MemoryAnalyzer, StagedAnalyzer,
};
use crate::dreaming::{DreamConfig, DreamPlanner, NoopConsolidator};
use crate::external::{EngineProvider, MemoryProvider};
use crate::lite::local::LocalCodingMemory;
use crate::record::{AddOptions, CallOptions, MemoryScope, NewMemory, SearchMode, SearchOptions};
use crate::team::TeamMemory;

#[cfg(feature = "sqlite")]
use crate::config::MemoryEngineConfig;
#[cfg(feature = "sqlite")]
use crate::store::MemoryStores;
#[cfg(feature = "sqlite")]
use openjiuwen_common::error::Error;

use super::record_input;
use super::LongTermMemory;

/// The tracing target of facade diagnostics.
const TARGET: &str = "openjiuwen_memory::facade";

/// What one [`LongTermMemory::remember`] call did (`extracted = added + merged + skipped`
/// when no write failed and no override gated the batch, aggregated from each fact's
/// [`crate::record::AddOutcome`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RememberReport {
    /// Number of facts the analyzer produced.
    pub extracted: usize,
    /// Number of facts stored as new records.
    pub added: usize,
    /// Number of facts folded into (or replacing) an existing record.
    pub merged: usize,
    /// Number of facts dropped by the dedupe policy.
    pub skipped: usize,
    /// Why a deterministic fallback ran instead of the model, when one did.
    pub degraded: Option<String>,
    /// Why the proactive stage withheld the write, when it did.
    ///
    /// Only set when the scope carries analyzer overrides, which turn the pipeline's
    /// advisory proactive decision into a gate: the report then names the decision's
    /// reason and nothing is written.
    pub gated: Option<String>,
}

impl LongTermMemory {
    /// The analyzer used by [`LongTermMemory::remember`].
    ///
    /// Built from the validated engine configuration and the attached model, unless
    /// [`LongTermMemoryBuilder::analyzer`](super::LongTermMemoryBuilder::analyzer)
    /// supplied one.
    #[must_use]
    pub fn analyzer(&self) -> &Arc<MemoryAnalyzer> {
        &self.analyzer
    }

    /// Extracts facts from `input` and writes each one under `scope`.
    ///
    /// `scope` is a parameter because [`ExtractionInput`] is pure conversation data (turns
    /// and history, no tenant), so the caller decides where the facts go. Each fact is its
    /// own engine write, so the dedupe policy applies per fact and earlier facts are
    /// visible to later ones.
    ///
    /// Extraction runs the staged pipeline over the engine configuration and the scope's
    /// persisted analyzer overrides (see the module docs). With no overrides the call is
    /// the same extraction-and-write pass the two-seam analyzer performed: every fact is
    /// written and the proactive decision is advisory. With overrides in effect the
    /// proactive decision gates the batch — `RememberReport::gated` names the reason and
    /// nothing is written when the stage says skip.
    ///
    /// # Errors
    ///
    /// `Error::Validation` for an invalid scope or input (and for a malformed stored
    /// override document), `Error::Unsupported` when a stored override names a
    /// `model`/`embedding` this facade does not hold, `Error::Cancelled` when `options.call`
    /// fires (including between facts — earlier writes stay stored), and the analyzer's or
    /// engine's error otherwise. A failed write is never dropped silently: the error names
    /// the fact index and content plus the engine's message.
    pub async fn remember(
        &self,
        scope: &MemoryScope,
        input: &ExtractionInput,
        options: &AddOptions,
    ) -> Result<RememberReport> {
        scope.validate()?;
        let (facts, mut report) = self.analyze_for_write(scope, input, options).await?;
        if report.gated.is_some() {
            return Ok(report);
        }
        for (index, fact) in facts.iter().enumerate() {
            options.call.check()?;
            let outcome = self
                .engine
                .add(scope, record_input(fact), options)
                .await
                .map_err(|error| {
                    error.with_context(format!("writing fact {index} (`{}`)", fact.content))
                })?;
            report.added += outcome.added.len();
            report.merged += outcome.merged.len();
            report.skipped += outcome.skipped.len();
        }
        Ok(report)
    }

    /// Runs `remember`'s analysis pass and returns the facts to write.
    ///
    /// The returned [`RememberReport`] carries `extracted`, `degraded`, and `gated`; its
    /// write counters stay at zero for the caller to fill from each
    /// [`AddOutcome`](crate::record::AddOutcome), so the same analysis feeds both
    /// [`LongTermMemory::remember`] and
    /// [`LongTermMemory::remember_with_assignments`]. A gated batch returns no facts at
    /// all, and a builder-injected analyzer keeps owning extraction while the scope
    /// carries no overrides (see the module docs).
    pub(super) async fn analyze_for_write(
        &self,
        scope: &MemoryScope,
        input: &ExtractionInput,
        options: &AddOptions,
    ) -> Result<(Vec<ExtractedFact>, RememberReport)> {
        let overrides = self.resolve_scope_overrides(scope).await?;
        if overrides.is_empty() && self.uses_injected_analyzer() {
            let extraction = self.analyzer.analyze(input, &options.call).await?;
            let report = RememberReport {
                extracted: extraction.facts.len(),
                degraded: extraction.degraded,
                gated: None,
                ..RememberReport::default()
            };
            return Ok((extraction.facts, report));
        }
        let staged = StagedAnalyzer::from_engine(self.config(), self.engine.model().cloned())
            .with_overrides(overrides.clone())
            .with_dedupe(self.config().dedupe.clone());
        let result = staged.analyze(input, &options.call).await?;
        let mut report = RememberReport {
            extracted: result.facts().len(),
            degraded: result.result.degraded.clone(),
            ..RememberReport::default()
        };
        let facts: Vec<ExtractedFact> = if overrides.is_empty() {
            result.facts().to_vec()
        } else if result.should_write() {
            result.writable_facts().into_iter().cloned().collect()
        } else {
            let reason = result.decision.as_ref().map_or_else(
                || "the proactive stage withheld the write".to_owned(),
                |decision| decision.reason.clone(),
            );
            tracing::debug!(
                target: TARGET,
                scope_id = scope.scope_id.as_str(),
                reason = reason.as_str(),
                facts = report.extracted,
                "remember withheld the write under the scope override gate"
            );
            report.gated = Some(reason);
            return Ok((Vec::new(), report));
        };
        Ok((facts, report))
    }

    /// Whether [`LongTermMemoryBuilder::analyzer`](super::LongTermMemoryBuilder::analyzer)
    /// supplied the analyzer.
    ///
    /// The default analyzer is built with [`MemoryAnalyzer::from_engine`], so "injected"
    /// means "the configuration or the extractor name the engine configuration implies
    /// does not match". A supplied analyzer that happens to match is indistinguishable
    /// from the default and behaves identically (both run the engine configuration).
    fn uses_injected_analyzer(&self) -> bool {
        let derived = AnalyzerConfig::from_engine(self.config());
        let implied_name = match (derived.extractor, self.engine.model().is_some()) {
            (ExtractorKind::Llm, true) => "llm",
            _ => "rule",
        };
        self.analyzer.config() != &derived || self.analyzer.extractor_name() != implied_name
    }

    /// Plans the write `incoming` would cause, without applying anything.
    ///
    /// Candidates are recalled with the dedupe candidate budget
    /// ([`DedupeConfig::candidate_top_k`](crate::config::DedupeConfig::candidate_top_k))
    /// and no score floor, like the engine's own dedupe scan: a blended recall score is not
    /// comparable to the dedupe thresholds (without an embedding model even an exact
    /// duplicate scores about `0.4`), so the [`DedupeConfig`](crate::config::DedupeConfig)
    /// similarity/Jaccard thresholds are applied by the plan itself. The plan comes from
    /// [`StagedAnalyzer::consolidate`] over the engine configuration, the scope's analyzer
    /// overrides, and the engine's `DedupeConfig`, so it agrees with the write path's
    /// policy instead of re-deriving one, and it only describes writes: nothing is stored,
    /// merged, replaced, or deleted.
    ///
    /// # Errors
    ///
    /// `Error::Validation` for an invalid scope or a blank/unsupported content (and for a
    /// malformed stored override document), `Error::Unsupported` when a stored override
    /// names a handle this facade does not hold, `Error::Cancelled` when `options` fires,
    /// and the engine's recall error otherwise.
    pub async fn consolidate(
        &self,
        scope: &MemoryScope,
        incoming: &NewMemory,
        options: &CallOptions,
    ) -> Result<ConsolidationPlan> {
        scope.validate()?;
        incoming.validate()?;
        options.check()?;
        let overrides = self.resolve_scope_overrides(scope).await?;
        let dedupe = self.config().dedupe.clone();
        let search = SearchOptions {
            top_k: dedupe.candidate_top_k,
            // The dedupe thresholds are applied by the plan; the recall stage must not
            // filter candidates out on a blended score.
            threshold: 0.0,
            kinds: self.config().searchable_kinds.clone(),
            mode: SearchMode::Hybrid,
            token_budget: None,
            call: options.clone(),
        };
        let hits = self
            .engine
            .search(scope, &incoming.content, &search)
            .await?;
        let records: Vec<_> = hits.into_iter().map(|hit| hit.record).collect();
        let fact = ExtractedFact::from(incoming);
        let staged = StagedAnalyzer::from_engine(self.config(), self.engine.model().cloned())
            .with_overrides(overrides)
            .with_dedupe(dedupe.clone());
        Ok(staged.consolidate(&[fact], &records, &dedupe))
    }

    /// The record surface of this instance through the stable [`MemoryProvider`] SPI.
    ///
    /// A view over the same [`MemoryEngine`](crate::engine::MemoryEngine), not a copy, so
    /// writes through either path see each other. Its capabilities come from
    /// [`crate::external::ProviderCapabilities::local`], so `semantic_search` is `true`
    /// only when an embedding model is attached.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn MemoryProvider> {
        Arc::new(EngineProvider::new(Arc::clone(&self.engine)))
    }

    /// A team-scoped memory facade over an in-process store.
    ///
    /// Team memory has its own store (it is not written through the engine), so this
    /// returns a fresh instance per call; use `LongTermMemory::team_memory_sqlite` when it
    /// must outlive the process.
    #[must_use]
    pub fn team_memory(&self) -> TeamMemory {
        TeamMemory::in_memory()
    }

    /// A team-scoped memory facade over the SQLite database at `path`.
    ///
    /// # Errors
    ///
    /// The store's construction error: `Error::Memory` when the parent directory cannot be
    /// created and `Error::External` when the database cannot be opened or its schema
    /// applied.
    #[cfg(feature = "sqlite")]
    pub async fn team_memory_sqlite(&self, path: impl AsRef<Path>) -> Result<TeamMemory> {
        TeamMemory::sqlite(path).await
    }

    /// A dream planner over an in-process job queue ([`NoopConsolidator`],
    /// [`DreamConfig::default`]).
    ///
    /// **Nothing starts a background loop or timer**: the caller owns scheduling and calls
    /// [`DreamPlanner::run_due`] or [`DreamPlanner::run_once`], nothing enforces retention
    /// automatically (`purge` is the caller's job), and no cross-process lease exists — the
    /// in-memory queue is process-local.
    ///
    /// # Errors
    ///
    /// The planner's config validation error (the defaults pass, but the constructor is
    /// fallible and this method never panics).
    pub fn dreaming_in_memory(&self) -> Result<DreamPlanner> {
        DreamPlanner::with_in_memory(Arc::new(NoopConsolidator), DreamConfig::default())
    }

    /// A dream planner over the SQLite job queue at `path` ([`NoopConsolidator`],
    /// [`DreamConfig::default`]).
    ///
    /// As with [`LongTermMemory::dreaming_in_memory`], **nothing starts a background loop
    /// or timer**: the caller drives `run_due`/`run_once`, calls `purge` for retention, and
    /// owns the sweep cadence. The queue is durable across processes and its claim is one
    /// atomic statement, so concurrent workers cannot claim the same job twice — that is
    /// single-statement concurrency control, not a cross-process lease.
    ///
    /// # Errors
    ///
    /// `Error::Memory` when the parent directory cannot be created, the database error when
    /// the queue cannot be opened or its schema applied, and the planner's config validation
    /// error.
    #[cfg(feature = "sqlite")]
    pub async fn dreaming_sqlite(&self, path: impl AsRef<Path>) -> Result<DreamPlanner> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::memory(format!(
                    "dreaming sqlite: cannot create directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        let url = format!("sqlite:{}", path.display());
        let repository = crate::dreaming::SqliteDreamRepository::connect(&url).await?;
        DreamPlanner::new(
            Arc::new(repository),
            Arc::new(NoopConsolidator),
            DreamConfig::default(),
        )
    }

    /// Builds a facade over a SQLite database whose schema is migrated first.
    ///
    /// [`MemoryStores::sqlite_migrated`] runs the versioned migrations and then opens the
    /// backends; the steps are forward-only and idempotent, so a migrated database is safe
    /// to reopen.
    ///
    /// # Errors
    ///
    /// The migration error annotated with `path` when a step fails, the storage error when
    /// the backends cannot be opened, and `Error::Config` when `config` is invalid.
    #[cfg(feature = "sqlite")]
    pub async fn sqlite_migrated(
        path: impl AsRef<Path>,
        config: MemoryEngineConfig,
    ) -> Result<Arc<Self>> {
        let stores = MemoryStores::sqlite_migrated(path).await?;
        Self::builder(config).stores(stores).build()
    }

    /// Opens the local coding-memory stack rooted at `root`.
    ///
    /// The local stack uses this instance's
    /// [`CodingMemoryConfig`](crate::config::CodingMemoryConfig) but keeps **its own source
    /// of truth**: a `MEMORY.md` file, a derived `FTS5` index, a cross-process advisory
    /// lock, and a debounced watcher. Those facts are *not* mirrored into the engine record
    /// path automatically, so pick one path per workspace —
    /// [`LongTermMemory::coding`](super::LongTermMemory::coding) for engine-backed facts,
    /// or this handle for the file-backed one.
    ///
    /// # Errors
    ///
    /// The filesystem, lock, or index error when the store cannot be initialized.
    pub async fn open_local_coding(&self, root: impl AsRef<Path>) -> Result<LocalCodingMemory> {
        LocalCodingMemory::open(root, self.config().coding.clone()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{AnalyzerConfig, ClassifierKind, ConsolidationAction};
    use crate::config::MemoryEngineConfig;
    use crate::dreaming::{DreamJob, DreamState};
    use crate::record::MemoryType;
    use crate::team::TeamMemoryFilter;
    use openjiuwen_common::error::Error;

    use super::super::ScopeAnalyzerOverrideSpec;
    use crate::facade::TAGS_KEY;

    async fn ltm() -> Arc<LongTermMemory> {
        LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("in-memory facade")
    }

    fn team_filter(team_id: &str) -> TeamMemoryFilter {
        TeamMemoryFilter {
            team_id: team_id.to_owned(),
            ..TeamMemoryFilter::default()
        }
    }

    #[tokio::test]
    async fn remember_writes_the_rule_facts_and_counts_them() {
        let ltm = ltm().await;
        // The default analyzer is the rule extractor over the engine's shared knobs.
        assert_eq!(ltm.analyzer().extractor_name(), "rule");
        assert_eq!(
            ltm.analyzer().config(),
            &AnalyzerConfig::from_engine(ltm.config())
        );
        assert!(format!("{ltm:?}").contains("analyzer"));
        let scope = MemoryScope::new("alice", "ws");
        let input = ExtractionInput::from_text(
            "I prefer dark mode. Remember that the deploy uses postgres.",
        );
        let report = ltm
            .remember(&scope, &input, &AddOptions::default())
            .await
            .expect("remember");
        assert_eq!(
            (
                report.extracted,
                report.added,
                report.merged,
                report.skipped
            ),
            (2, 2, 0, 0),
            "{report:?}"
        );
        assert_eq!(report.degraded, None);
        let records = ltm.list(&scope, &[]).await.expect("list");
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|r| r.kind == MemoryType::UserProfile));
        // Rule facts carry tags (`preference`, `explicit`); the facade keeps them under the
        // shared `tags` metadata key.
        assert!(records.iter().all(|record| record
            .metadata
            .get(TAGS_KEY)
            .and_then(|tags| tags.as_array())
            .is_some_and(|tags| !tags.is_empty())));
    }

    #[tokio::test]
    async fn rejected_inputs_are_errors_and_store_nothing() {
        let ltm = ltm().await;
        let scope = MemoryScope::new("alice", "ws");
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let cancelled = AddOptions::default().with_cancellation(token);
        let input = ExtractionInput::from_text("I prefer dark mode.");
        let error = ltm
            .remember(&scope, &input, &cancelled)
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");

        let mut invalid = ExtractionInput::default();
        invalid.push_turn("", "I prefer dark mode.");
        let error = ltm
            .remember(&scope, &invalid, &AddOptions::default())
            .await
            .expect_err("empty role");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        let error = ltm
            .consolidate(&scope, &NewMemory::semantic("   "), &CallOptions::new())
            .await
            .expect_err("blank content");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);
    }

    #[tokio::test]
    async fn a_scope_override_changes_what_remember_writes() {
        let conversation = {
            let mut input = ExtractionInput::default();
            input.push_user("my name is Alice.");
            input.push_user("I prefer dark mode.");
            input.push_user("The service uses Postgres.");
            input
        };
        let scope = MemoryScope::new("alice", "ws");
        // Without an override the staged pipeline writes every extracted fact.
        let plain = ltm().await;
        let baseline = plain
            .remember(&scope, &conversation, &AddOptions::default())
            .await
            .expect("remember");
        assert_eq!(
            (
                baseline.extracted,
                baseline.added,
                baseline.gated.as_deref()
            ),
            (3, 3, None),
            "{baseline:?}"
        );

        // `max_facts = 1` is a scope policy: one fact is extracted and written.
        let capped = ltm().await;
        capped
            .set_scope_overrides("ws", &ScopeAnalyzerOverrideSpec::new().with_max_facts(1))
            .await
            .expect("override");
        let report = capped
            .remember(&scope, &conversation, &AddOptions::default())
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
        assert_eq!(report.degraded, None);
        assert_eq!(report.gated, None);
        let records = capped.list(&scope, &[]).await.expect("list");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "my name is Alice");
    }

    #[tokio::test]
    async fn an_empty_spec_matches_the_unconfigured_behaviour() {
        let conversation = ExtractionInput::from_text("I prefer dark mode.");
        let scope = MemoryScope::new("alice", "ws");
        let plain = ltm().await;
        let configured = ltm().await;
        configured
            .set_scope_overrides("ws", &ScopeAnalyzerOverrideSpec::default())
            .await
            .expect("empty spec");
        let left = plain
            .remember(&scope, &conversation, &AddOptions::default())
            .await
            .expect("plain");
        let right = configured
            .remember(&scope, &conversation, &AddOptions::default())
            .await
            .expect("configured");
        // An all-`None` document resolves to empty overrides, so the call is unchanged.
        assert_eq!(left, right);
        assert_eq!((left.extracted, left.added), (1, 1));
        assert_eq!(left.degraded, None);
        assert_eq!(left.gated, None);
    }

    #[tokio::test]
    async fn the_scope_override_gate_withholds_a_write_and_reports_it() {
        // A plain project statement carries no proactive signal and sits below the
        // importance threshold, so the rule proactive stage says "skip".
        let plain = ltm().await;
        let scope = MemoryScope::new("alice", "ws");
        let input = ExtractionInput::from_text("The service uses Postgres.");
        let ungated = plain
            .remember(&scope, &input, &AddOptions::default())
            .await
            .expect("remember");
        assert_eq!((ungated.extracted, ungated.added), (1, 1), "{ungated:?}");
        assert_eq!(ungated.gated, None);
        assert_eq!(plain.list(&scope, &[]).await.expect("list").len(), 1);

        // Under an explicit scope policy the decision becomes a gate.
        let gated = ltm().await;
        gated
            .set_scope_overrides("ws", &ScopeAnalyzerOverrideSpec::new().with_max_facts(1))
            .await
            .expect("override");
        let report = gated
            .remember(&scope, &input, &AddOptions::default())
            .await
            .expect("remember");
        assert_eq!(
            (
                report.extracted,
                report.added,
                report.merged,
                report.skipped
            ),
            (1, 0, 0, 0),
            "{report:?}"
        );
        let reason = report.gated.as_deref().expect("the gate is reported");
        assert!(
            reason.contains("threshold") || reason.contains("no candidate"),
            "{reason}"
        );
        assert_eq!(gated.list(&scope, &[]).await.expect("list").len(), 0);
    }

    #[tokio::test]
    async fn remember_through_the_staged_pipeline_surfaces_a_model_fallback() {
        // An emptied script makes every model call unusable, so the scope's `llm`
        // extractor exhausts its retry budget and falls back to the rule extractor.
        let model = openjiuwen_test_support::ScriptedModel::builder().build();
        let handle = model.clone();
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(Arc::new(model))
            .build()
            .expect("facade with a model");
        ltm.set_scope_overrides(
            "ws",
            &ScopeAnalyzerOverrideSpec::new().with_extractor(ExtractorKind::Llm),
        )
        .await
        .expect("override");
        let scope = MemoryScope::new("alice", "ws");
        let report = ltm
            .remember(
                &scope,
                &ExtractionInput::from_text("my name is Alice. I prefer dark mode."),
                &AddOptions::default(),
            )
            .await
            .expect("remember");
        let degraded = report
            .degraded
            .as_deref()
            .expect("the fallback is recorded");
        assert!(degraded.contains("fell back to rules"), "{degraded}");
        assert_eq!(report.extracted, 2, "{report:?}");
        assert_eq!(report.added, 2, "{report:?}");
        assert_eq!(report.gated, None);
        assert_eq!(handle.call_count(), 4, "one call per retry attempt");
        let records = ltm.list(&scope, &[]).await.expect("list");
        assert_eq!(records.len(), 2);
        assert!(records
            .iter()
            .any(|record| record.content == "I prefer dark mode"));
    }

    #[tokio::test]
    async fn a_cancelled_staged_call_writes_nothing() {
        let ltm = ltm().await;
        ltm.set_scope_overrides("ws", &ScopeAnalyzerOverrideSpec::new().with_max_facts(1))
            .await
            .expect("override");
        let scope = MemoryScope::new("alice", "ws");
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let error = ltm
            .remember(
                &scope,
                &ExtractionInput::from_text("I prefer dark mode."),
                &AddOptions::default().with_cancellation(token),
            )
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);
    }

    #[tokio::test]
    async fn consolidate_honours_a_scope_classifier_override_and_applies_nothing() {
        let model = openjiuwen_test_support::ScriptedModel::text("{}");
        let handle = model.clone();
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(Arc::new(model))
            .build()
            .expect("facade with a model");
        let scope = MemoryScope::new("alice", "ws");
        let stored = ltm
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode"),
                &AddOptions::default(),
            )
            .await
            .expect("add")
            .added
            .remove(0);
        ltm.set_scope_overrides(
            "ws",
            &ScopeAnalyzerOverrideSpec::new().with_classifier(ClassifierKind::Llm),
        )
        .await
        .expect("override");
        // The scope really resolves to the model-backed classifier.
        let resolved = ltm
            .resolve_scope_overrides(&scope)
            .await
            .expect("resolve overrides");
        assert_eq!(resolved.classifier(), Some(ClassifierKind::Llm));

        let plan = ltm
            .consolidate(
                &scope,
                &NewMemory::semantic("alice prefers DARK mode."),
                &CallOptions::new(),
            )
            .await
            .expect("consolidate");
        assert!(matches!(
            plan.actions[..],
            [ConsolidationAction::Merge { fact_index: 0, .. }]
        ));
        assert_eq!(plan.adds(), 0);
        // A plan is dedupe policy work: the classifier is never consulted and nothing is
        // applied.
        assert_eq!(handle.call_count(), 0);
        let records = ltm.list(&scope, &[]).await.expect("list");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, stored.id);
        assert_eq!(records[0].content, "Alice prefers dark mode");
    }

    #[tokio::test]
    async fn consolidate_plans_a_merge_against_a_stored_record() {
        let ltm = ltm().await;
        let scope = MemoryScope::new("alice", "ws");
        let stored = ltm
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode"),
                &AddOptions::default(),
            )
            .await
            .expect("add")
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
        assert!(matches!(
            &plan.actions[..],
            [ConsolidationAction::Merge { fact_index: 0, target_id, .. }] if target_id == &stored.id
        ));
        // A plan describes writes only: nothing was stored.
        assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
    }

    #[tokio::test]
    async fn the_provider_view_round_trips_an_add_and_a_search() {
        let ltm = ltm().await;
        let provider = ltm.provider();
        assert_eq!(provider.name(), "local");
        assert_eq!(
            provider.capabilities().semantic_search,
            ltm.engine().has_embedding()
        );
        assert!(provider.capabilities().keyword_search);

        let scope = MemoryScope::new("alice", "ws");
        provider
            .add(
                &scope,
                vec![NewMemory::semantic("alice prefers dark mode")],
                &AddOptions::default(),
            )
            .await
            .expect("provider add");
        // The view shares the engine, so the facade sees the record too.
        assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
        let hits = provider
            .search(
                &scope,
                "dark mode",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("provider search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.content, "alice prefers dark mode");
    }

    #[tokio::test]
    async fn team_memory_and_dreaming_start_empty() {
        let ltm = ltm().await;
        let team = ltm.team_memory();
        assert_eq!(
            team.list(&team_filter("team-1")).await.expect("list").len(),
            0
        );

        let planner = ltm.dreaming_in_memory().expect("planner");
        assert_eq!(
            planner.config().batch_size,
            DreamConfig::default().batch_size
        );
        let job = DreamJob::default();
        let id = job.id.clone();
        planner.repository().enqueue(job).await.expect("enqueue");
        let summary = planner
            .run_once(chrono::Utc::now(), planner.config().batch_size)
            .await
            .expect("sweep");
        assert_eq!(summary.scanned, 1);
        let stored = planner
            .repository()
            .get(&id)
            .await
            .expect("get")
            .expect("job");
        assert_eq!(stored.state, DreamState::Skipped);
    }

    #[cfg(feature = "sqlite")]
    mod sqlite {
        use super::*;
        use crate::team::NewTeamMemory;

        #[tokio::test]
        async fn sqlite_migrated_opens_and_reopens_a_migrated_database() {
            let directory = tempfile::tempdir().expect("tempdir");
            let path = directory.path().join("memory.db");
            let ltm = LongTermMemory::sqlite_migrated(&path, MemoryEngineConfig::default())
                .await
                .expect("migrated facade");
            let scope = MemoryScope::new("alice", "ws");
            ltm.add(
                &scope,
                NewMemory::semantic("likes tea"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
            let report = crate::store::migration::migrate_sqlite(&path)
                .await
                .expect("re-migrate");
            assert!(report.is_noop(), "{report:?}");
            let reopened = LongTermMemory::sqlite(MemoryEngineConfig::default(), &path)
                .await
                .expect("reopen");
            assert_eq!(reopened.list(&scope, &[]).await.expect("list").len(), 1);
        }

        #[tokio::test]
        async fn team_memory_and_dreaming_sqlite_persist_across_handles() {
            let directory = tempfile::tempdir().expect("tempdir");
            let ltm = ltm().await;

            let team_path = directory.path().join("team.db");
            ltm.team_memory_sqlite(&team_path)
                .await
                .expect("team sqlite")
                .remember(
                    NewTeamMemory::shared_fact("team-1", "the build uses sqlite"),
                    &CallOptions::new(),
                )
                .await
                .expect("remember");
            let team = ltm.team_memory_sqlite(&team_path).await.expect("reopen");
            assert_eq!(
                team.list(&team_filter("team-1")).await.expect("list").len(),
                1
            );

            let dream_path = directory.path().join("dream.db");
            let planner = ltm.dreaming_sqlite(&dream_path).await.expect("dream");
            let job = DreamJob::default();
            let id = job.id.clone();
            planner.repository().enqueue(job).await.expect("enqueue");
            drop(planner);
            let reopened = ltm.dreaming_sqlite(&dream_path).await.expect("reopen");
            let stored = reopened
                .repository()
                .get(&id)
                .await
                .expect("get")
                .expect("job");
            assert_eq!(stored.state, DreamState::Pending);
        }
    }
}
