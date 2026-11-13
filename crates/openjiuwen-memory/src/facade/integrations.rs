//! Facade entry points for the subsystems that sit beside the record path.
//!
//! [`LongTermMemory`] wires record CRUD, recall, messages, variables, and scope config;
//! this module adds the analyzer ([`LongTermMemory::remember`] /
//! [`LongTermMemory::consolidate`]), the provider SPI ([`LongTermMemory::provider`]),
//! dreaming, team memory, schema migrations, and the local coding stack over the same
//! injected stores. Nothing here starts a background task or reads a clock of its own:
//! scheduling, sweeps, purges, and the application of a plan stay with the caller.
use std::path::Path;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::analyzer::{ConsolidationPlan, ExtractedFact, ExtractionInput, MemoryAnalyzer};
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

use super::LongTermMemory;

/// The metadata key the crate uses for free-form tags.
const TAGS_KEY: &str = "tags";

/// What one [`LongTermMemory::remember`] call did (`extracted = added + merged + skipped`
/// when no write failed, aggregated from each fact's [`crate::record::AddOutcome`]).
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
    /// # Errors
    ///
    /// `Error::Validation` for an invalid scope or input, `Error::Cancelled` when
    /// `options.call` fires (including between facts — earlier writes stay stored), and the
    /// analyzer's or engine's error otherwise. A failed write is never dropped silently: the
    /// error names the fact index and content plus the engine's message.
    pub async fn remember(
        &self,
        scope: &MemoryScope,
        input: &ExtractionInput,
        options: &AddOptions,
    ) -> Result<RememberReport> {
        scope.validate()?;
        let extraction = self.analyzer.analyze(input, &options.call).await?;
        let mut report = RememberReport {
            extracted: extraction.facts.len(),
            degraded: extraction.degraded,
            ..RememberReport::default()
        };
        for (index, fact) in extraction.facts.iter().enumerate() {
            options.call.check()?;
            let outcome = self
                .engine
                .add(scope, write_memory(fact), options)
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

    /// Plans the write `incoming` would cause, without applying anything.
    ///
    /// Candidates are recalled with the dedupe candidate budget
    /// ([`DedupeConfig::candidate_top_k`](crate::config::DedupeConfig::candidate_top_k))
    /// and no score floor, like the engine's own dedupe scan: a blended recall score is not
    /// comparable to the dedupe thresholds (without an embedding model even an exact
    /// duplicate scores about `0.4`), so the [`DedupeConfig`](crate::config::DedupeConfig)
    /// similarity/Jaccard thresholds are applied by the plan itself. The plan comes from
    /// [`MemoryAnalyzer::consolidate`] under the engine's `DedupeConfig`, so it agrees with
    /// the write path's policy instead of re-deriving one, and it only describes writes:
    /// nothing is stored, merged, replaced, or deleted.
    ///
    /// # Errors
    ///
    /// `Error::Validation` for an invalid scope or a blank/unsupported content,
    /// `Error::Cancelled` when `options` fires, and the engine's recall error otherwise.
    pub async fn consolidate(
        &self,
        scope: &MemoryScope,
        incoming: &NewMemory,
        options: &CallOptions,
    ) -> Result<ConsolidationPlan> {
        scope.validate()?;
        incoming.validate()?;
        options.check()?;
        let dedupe = &self.config().dedupe;
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
        Ok(self.analyzer.consolidate(&[fact], &records, dedupe))
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

/// Converts an extracted fact into an engine write input, keeping its tags.
///
/// [`ExtractedFact::to_new_memory`] already carries content, kind, importance, source, and
/// metadata. Tags are not a [`NewMemory`] field, so they go under the crate's `tags`
/// metadata key (the convention `lite` uses) unless the fact supplies that key itself.
fn write_memory(fact: &ExtractedFact) -> NewMemory {
    let mut memory = fact.to_new_memory();
    if !fact.tags.is_empty() && !memory.metadata.contains_key(TAGS_KEY) {
        let tags = fact.tags.iter().cloned().map(serde_json::Value::String);
        memory.metadata.insert(
            TAGS_KEY.to_owned(),
            serde_json::Value::Array(tags.collect()),
        );
    }
    memory
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{AnalyzerConfig, ConsolidationAction};
    use crate::config::MemoryEngineConfig;
    use crate::dreaming::{DreamJob, DreamState};
    use crate::record::MemoryType;
    use crate::team::TeamMemoryFilter;
    use openjiuwen_common::error::Error;

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
