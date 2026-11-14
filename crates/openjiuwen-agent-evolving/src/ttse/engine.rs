//! The TTSE engine: observe → induce → consult → dream, with persistence.
//!
//! Wires the tip bank, the inducer, the dreamer and the consultation renderer
//! into one offline pipeline. Ticks are monotonic counters owned by the bank, so
//! no clock is read and repeated runs over identical inputs are identical.
//!
//! Persistence is a plain [`TipBank`] JSON document written atomically
//! (`<path>.tmp` plus rename), so a reader never observes a partial file. The
//! document holds the tips and the tick only: engine-side bookkeeping such as
//! the consultation counter restarts at `0` across a save/load.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result, ResultExt};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::retrieval::embedding::Embedding;

use crate::signal::EvolutionSignal;
use crate::trajectory::Trajectory;
use crate::ttse::bank::{AddOutcome, TipBank, TipBankStats};
use crate::ttse::config::TtseConfig;
use crate::ttse::dream::Dreamer;
use crate::ttse::index::EmbeddingIndex;
use crate::ttse::induce::TipInducer;
use crate::ttse::models::{ConsultResult, DreamReport, TipRecord};
use crate::ttse::retrieve::{self, RetrieveMode};
use crate::ttse::success::SuccessDetector;

/// The suffix appended to the store path for the in-flight write.
const TEMP_SUFFIX: &str = ".tmp";

/// Offline TTSE pipeline over trajectories and evolution signals.
///
/// The engine owns the [`TipBank`] every pass reads and writes, the
/// [`TipInducer`] that proposes tips, and the [`Dreamer`] that merges and prunes
/// them. It never reads a clock: [`TtseEngine::observe`] advances the bank tick,
/// which is the only time-like value in the pipeline.
#[derive(Clone)]
pub struct TtseEngine {
    /// The shared tunables every pass reads.
    config: TtseConfig,
    /// The stored tips.
    bank: TipBank,
    /// The tip producer, offline or model-driven.
    inducer: TipInducer,
    /// The merge/prune pass.
    dreamer: Dreamer,
    /// The optional embedding index backing semantic retrieval.
    index: Option<EmbeddingIndex>,
    /// How many consultations this engine has served.
    consult_count: u64,
}

impl fmt::Debug for TtseEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TtseEngine")
            .field("config", &self.config)
            .field("bank", &self.bank)
            .field("inducer", &self.inducer)
            .field("dreamer", &self.dreamer)
            .field("index", &self.index)
            .field("consult_count", &self.consult_count)
            .finish()
    }
}

impl TtseEngine {
    /// Builds an engine from validated `config`.
    ///
    /// The bank starts empty, the inducer is offline and the dreamer follows the
    /// config; nothing is read from disk until [`TtseEngine::load`] is called.
    ///
    /// # Errors
    ///
    /// [`TtseConfig::validate`]'s error, so an unusable capacity or threshold
    /// fails at construction instead of on the first observation.
    pub fn new(config: TtseConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            bank: TipBank::new(config.capacity, config.dedup_threshold),
            inducer: TipInducer::new(config.clone()),
            dreamer: Dreamer::new(config.clone()),
            index: None,
            config,
            consult_count: 0,
        })
    }

    /// Installs the model used by the induction prompt path and by dreaming.
    ///
    /// The inducer and the dreamer are rebuilt from the engine config, so calling
    /// this twice replaces the model instead of stacking calls. Dreaming through
    /// the model is opt-in: [`TtseEngine::dream`] stays deterministic while
    /// [`TtseEngine::dream_async`] uses it.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>, request: ModelRequestConfig) -> Self {
        self.inducer =
            TipInducer::new(self.config.clone()).with_model(Arc::clone(&model), request.clone());
        self.dreamer = Dreamer::new(self.config.clone()).with_model(model, request);
        self
    }

    /// Installs an embedding provider, creating an empty index over it.
    ///
    /// The index is cold: call [`TtseEngine::sync_index`] (or
    /// [`TtseEngine::dream_async`], which syncs) before relying on semantic
    /// retrieval. Without an index the retrieval modes fall back to the lexical
    /// ranking.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.index = Some(EmbeddingIndex::new(embedding));
        self
    }

    /// Installs the embedding index persisted at `path`, creating an empty one
    /// when the file is absent.
    ///
    /// # Errors
    ///
    /// [`EmbeddingIndex::open`]'s error: a stored index built by another model or
    /// dimension is [`Error::Validation`], as is a malformed document.
    pub fn with_embedding_index(
        mut self,
        path: impl Into<PathBuf>,
        embedding: Arc<dyn Embedding>,
    ) -> Result<Self> {
        self.index = Some(EmbeddingIndex::open(path, embedding)?);
        Ok(self)
    }

    /// The embedding index, when one is installed.
    #[must_use]
    pub fn embedding_index(&self) -> Option<&EmbeddingIndex> {
        self.index.as_ref()
    }

    /// Cold-fills and refreshes the index against the bank.
    ///
    /// Returns how many vectors were embedded; `0` when no index is installed.
    ///
    /// # Errors
    ///
    /// The embedder's errors from [`EmbeddingIndex::sync`].
    pub async fn sync_index(&mut self) -> Result<usize> {
        match self.index.as_mut() {
            Some(index) => index.sync(&self.bank).await,
            None => Ok(0),
        }
    }

    /// The configuration this engine applies.
    #[must_use]
    pub fn config(&self) -> &TtseConfig {
        &self.config
    }

    /// The tip bank.
    #[must_use]
    pub fn bank(&self) -> &TipBank {
        &self.bank
    }

    /// The tip bank, mutably.
    ///
    /// Exposed for callers that restore a bank or seed one by hand; prefer
    /// [`TtseEngine::observe`] so the tick stays monotonic.
    pub fn bank_mut(&mut self) -> &mut TipBank {
        &mut self.bank
    }

    /// The tip producer.
    #[must_use]
    pub fn inducer(&self) -> &TipInducer {
        &self.inducer
    }

    /// The merge/prune pass.
    #[must_use]
    pub fn dreamer(&self) -> &Dreamer {
        &self.dreamer
    }

    /// How many consultations this engine has served since construction or load.
    #[must_use]
    pub fn consult_count(&self) -> u64 {
        self.consult_count
    }

    /// Learns from `trajectories` and `signals`, returning the tips that changed.
    ///
    /// The pass is skipped entirely when [`TtseConfig::evolve_enabled`] is off:
    /// the bank keeps its tips and its tick. Otherwise the bank tick advances,
    /// the trajectories go through [`SuccessDetector`] (successful runs are not
    /// lessons and are dropped, while signals are always passed), the inducer
    /// proposes tips, and each proposal is stored through [`TipBank::add`]. The
    /// returned records are the bank's own copies — inserted or merged into —
    /// deduplicated by [`TipRecord::tip_id`] in input order.
    ///
    /// When dreaming is enabled and the dreamer reports the bank is due
    /// ([`Dreamer::should_dream`]), one dream pass runs afterwards; its report is
    /// discarded because the added tips are the caller's answer.
    ///
    /// # Errors
    ///
    /// The inducer's error; the offline path never fails on user input.
    pub async fn observe(
        &mut self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
    ) -> Result<Vec<TipRecord>> {
        if !self.config.evolve_enabled {
            return Ok(Vec::new());
        }
        self.bank.increment_tick();

        let detector = SuccessDetector::from_config(&self.config);
        let kept: Vec<Trajectory> = trajectories
            .iter()
            .filter(|trajectory| !detector.detect(trajectory).successful)
            .cloned()
            .collect();

        let induced = self.inducer.induce(&kept, signals).await?;
        let mut added: Vec<TipRecord> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for tip in induced {
            let tip_id = match self.bank.add(tip) {
                AddOutcome::Inserted(tip_id) | AddOutcome::Merged { into: tip_id, .. } => tip_id,
            };
            if !seen.insert(tip_id.clone()) {
                continue;
            }
            if let Some(record) = self.bank.get(&tip_id) {
                added.push(record.clone());
            }
        }

        if self.config.dream_enabled && self.dreamer.should_dream(&self.bank) {
            self.dream();
        }
        Ok(added)
    }

    /// Selects and renders the tips relevant to `query`.
    ///
    /// Returns an empty result when [`TtseConfig::inject_enabled`] is off, so
    /// nothing is injected into the prompt. Each served consultation counts as
    /// one hit per returned tip, recorded as a success through
    /// [`TipBank::observe`]; the returned result keeps the rendering produced
    /// before the hit was recorded, so the text depends only on the bank the
    /// caller saw.
    pub fn consult(&mut self, query: &str) -> ConsultResult {
        if !self.config.inject_enabled {
            return ConsultResult::default();
        }
        self.consult_count = self.consult_count.saturating_add(1);
        let result = crate::ttse::consult::consult(&self.bank, query, &self.config);
        for tip in &result.tips {
            self.bank.observe(&tip.tip_id, true);
        }
        result
    }

    /// Selects and renders the tips relevant to `query` through the retrieval
    /// modes, using [`RetrieveMode::Hybrid`].
    ///
    /// This is [`TtseEngine::consult`] with the ranked retrieval path: the index
    /// contributes a cosine branch, the bank a lexical one, and both are fused with
    /// Reciprocal Rank Fusion. Hit bookkeeping and the
    /// [`TtseConfig::inject_enabled`] switch behave exactly as in `consult`.
    ///
    /// # Errors
    ///
    /// The embedder's errors from [`retrieve::rank`].
    pub async fn consult_async(&mut self, query: &str) -> Result<ConsultResult> {
        self.consult_with(query, RetrieveMode::Hybrid).await
    }

    /// [`TtseEngine::consult_async`] with an explicit retrieval mode.
    ///
    /// # Errors
    ///
    /// The embedder's errors from [`retrieve::rank`].
    pub async fn consult_with(&mut self, query: &str, mode: RetrieveMode) -> Result<ConsultResult> {
        if !self.config.inject_enabled {
            return Ok(ConsultResult::default());
        }
        self.consult_count = self.consult_count.saturating_add(1);
        let result =
            retrieve::consult_async(&self.bank, self.index.as_ref(), query, &self.config, mode)
                .await?;
        for tip in &result.tips {
            self.bank.observe(&tip.tip_id, true);
        }
        Ok(result)
    }

    /// Runs one dream pass over the bank and reports what changed.
    pub fn dream(&mut self) -> DreamReport {
        self.dreamer.dream(&mut self.bank)
    }

    /// Runs one dream pass, using the installed model when there is one.
    ///
    /// The index is synced first, so cosine clustering sees a vector for every tip,
    /// and again afterwards, so merged-away tips leave the index. Without a model
    /// this is [`TtseEngine::dream`].
    ///
    /// # Errors
    ///
    /// The embedder's errors from syncing the index.
    pub async fn dream_async(&mut self) -> Result<DreamReport> {
        self.sync_index().await?;
        let report = self
            .dreamer
            .dream_with_model(&mut self.bank, self.index.as_ref())
            .await?;
        self.sync_index().await?;
        Ok(report)
    }

    /// Aggregate counts over the stored tips.
    #[must_use]
    pub fn stats(&self) -> TipBankStats {
        self.bank.stats()
    }

    /// Writes the bank to `path`, creating the parent directory when needed.
    ///
    /// The document is `serde_json` pretty output of [`TipBank`], so a Python or
    /// Rust reader can share the file. The write is atomic: the bytes go to
    /// `<path>.tmp` first and are then renamed onto `path`, so readers never see
    /// a partial store, and a failed rename removes the temporary file.
    ///
    /// # Errors
    ///
    /// The serialization error, or [`Error::External`] wrapping the failing
    /// filesystem operation (the shared `std::io::Error` conversion).
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let encoded = serde_json::to_vec_pretty(&self.bank)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await.context(format!(
                "creating ttse store directory {}",
                parent.display()
            ))?;
        }
        let temp = temp_path(path);
        tokio::fs::write(&temp, &encoded)
            .await
            .context(format!("writing ttse temporary store {}", temp.display()))?;
        if let Err(error) = tokio::fs::rename(&temp, path).await {
            // Best effort: a leftover `.tmp` must not shadow a real store.
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(Error::from(error).with_context(format!(
                "renaming ttse store {} to {}",
                temp.display(),
                path.display()
            )));
        }
        Ok(())
    }

    /// Writes the bank to `path` and the embedding index next to it.
    ///
    /// The bank goes to `path`; the index goes to the path it was opened from or
    /// bound to, or to `<path>.embedding.json` when it has none. A missing index
    /// makes this [`TtseEngine::save`].
    ///
    /// # Errors
    ///
    /// [`TtseEngine::save`]'s errors, plus the index write errors.
    pub async fn save_with_index(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.save(path).await?;
        let Some(index) = self.index.as_ref() else {
            return Ok(());
        };
        let index_path = index.path().map_or_else(
            || {
                let mut sibling = path.as_os_str().to_owned();
                sibling.push(".embedding.json");
                PathBuf::from(sibling)
            },
            Path::to_path_buf,
        );
        index.save(index_path).await
    }

    /// Reads a bank from `path`, using the default configuration otherwise.
    ///
    /// The store path of the returned config is `path`, so a later
    /// [`TtseEngine::save`] to [`TtseConfig::store_path`] writes back where the
    /// bank came from. The consultation counter of the loaded engine starts at
    /// `0`: it is engine bookkeeping and is not persisted.
    ///
    /// # Errors
    ///
    /// [`Error::External`] when the file cannot be read, [`Error::Validation`]
    /// when it is not a valid bank document, and [`TipBank::validate`]'s error for
    /// a well-formed but inconsistent bank.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = TtseConfig {
            store_path: path.to_string_lossy().into_owned(),
            ..TtseConfig::default()
        };
        Self::load_with_config(path, config).await
    }

    /// Reads a bank from `path` and applies `config` to the loaded engine.
    ///
    /// The bank keeps the capacity, threshold and tick it was saved with; the
    /// configuration governs later passes (consultation limits, dreaming and
    /// induction).
    ///
    /// # Errors
    ///
    /// [`TtseConfig::validate`]'s error for an unusable `config`, plus the read,
    /// parse and bank-validation errors of [`TtseEngine::load`].
    pub async fn load_with_config(path: impl AsRef<Path>, config: TtseConfig) -> Result<Self> {
        let path = path.as_ref();
        config.validate()?;
        let bytes = tokio::fs::read(path)
            .await
            .context(format!("reading ttse store {}", path.display()))?;
        let bank = serde_json::from_slice::<TipBank>(&bytes).map_err(|error| {
            Error::validation(format!("invalid ttse store {}: {error}", path.display()))
        })?;
        bank.validate()?;
        let mut engine = Self::new(config)?;
        engine.bank = bank;
        Ok(engine)
    }
}

/// The sibling temporary path a save writes before renaming.
fn temp_path(path: &Path) -> PathBuf {
    let mut temp = path.as_os_str().to_owned();
    temp.push(TEMP_SUFFIX);
    PathBuf::from(temp)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::Error;
    use openjiuwen_core::foundation::llm::Model;
    use openjiuwen_test_support::ScriptedModel;

    use super::*;
    use crate::trajectory::{StepKind, StepStatus, TrajectoryOutcome, TrajectoryStep};
    use crate::ttse::models::TipCategory;

    fn trajectory(id: &str, outcome: TrajectoryOutcome, content: &str) -> Trajectory {
        let mut trajectory = Trajectory::new(id).expect("valid id").with_outcome(outcome);
        let status = if outcome == TrajectoryOutcome::Failure {
            StepStatus::Error
        } else {
            StepStatus::Ok
        };
        trajectory.push(TrajectoryStep::new(StepKind::ToolResult, content).with_status(status));
        trajectory
    }

    fn engine_with_tips() -> TtseEngine {
        let mut engine = TtseEngine::new(TtseConfig::default()).expect("valid config");
        engine
            .bank_mut()
            .add(TipRecord::new("Prefer rg over grep", TipCategory::Tool).expect("valid rule"));
        engine.bank_mut().add(
            TipRecord::new("Run the tests offline", TipCategory::Workflow).expect("valid rule"),
        );
        engine
    }

    #[test]
    fn new_rejects_an_invalid_config() {
        let config = TtseConfig {
            capacity: 0,
            ..TtseConfig::default()
        };
        let error = TtseEngine::new(config).expect_err("capacity 0 is invalid");
        assert!(matches!(error, Error::Validation(_)), "got {error:?}");

        let engine = TtseEngine::new(TtseConfig::default()).expect("valid config");
        assert_eq!(engine.consult_count(), 0);
        assert!(engine.bank().is_empty());
        assert_eq!(engine.stats().count, 0);
        assert!(!engine.inducer().has_model());
        assert_ne!(engine.dreamer().config().store_path, "");
    }

    #[tokio::test]
    async fn observe_is_a_no_op_when_evolving_is_disabled() {
        let config = TtseConfig {
            evolve_enabled: false,
            ..TtseConfig::default()
        };
        let mut engine = TtseEngine::new(config).expect("valid config");
        engine.bank_mut().add(
            TipRecord::new("Keep the build green", TipCategory::Workflow).expect("valid rule"),
        );
        let before = engine.bank().clone();

        let failed = trajectory("t-1", TrajectoryOutcome::Failure, "the build is broken");
        let signals = Vec::new();
        let added = engine
            .observe(std::slice::from_ref(&failed), &signals)
            .await
            .expect("observe succeeds");

        assert_eq!(added, Vec::<TipRecord>::new());
        assert_eq!(engine.bank(), &before, "the bank is untouched");
        assert_eq!(engine.bank().tick(), 0, "no tick is spent");
    }

    #[tokio::test]
    async fn observe_learns_from_failures_and_skips_successes() {
        let mut engine = TtseEngine::new(TtseConfig::default()).expect("valid config");
        let failed = trajectory("t-1", TrajectoryOutcome::Failure, "the build is broken");
        let succeeded = trajectory("t-2", TrajectoryOutcome::Success, "all green");

        let added = engine
            .observe(&[failed, succeeded], &[])
            .await
            .expect("observe succeeds");

        assert_eq!(added.len(), 1);
        assert_eq!(added[0].rule, "Avoid repeating: the build is broken");
        assert_eq!(added[0].source, "trajectory");
        assert_eq!(engine.bank().len(), 1);
        assert_eq!(engine.bank().tick(), 1);
        // A second observation of the same failure merges instead of duplicating.
        let failed = trajectory("t-3", TrajectoryOutcome::Failure, "the build is broken");
        let added = engine
            .observe(std::slice::from_ref(&failed), &[])
            .await
            .expect("observe succeeds");
        assert_eq!(added.len(), 1, "the merged record is returned once");
        assert_eq!(engine.bank().len(), 1);
        assert_eq!(engine.bank().tick(), 2);

        let mut engine = TtseEngine::new(TtseConfig::default())
            .expect("valid config")
            .with_model(
                Arc::new(ScriptedModel::text("nope")) as Arc<dyn Model>,
                ModelRequestConfig::default(),
            );
        assert!(engine.inducer().has_model());
        let added = engine.observe(&[], &[]).await.expect("observe");
        assert_eq!(added, Vec::<TipRecord>::new());
    }

    #[tokio::test]
    async fn consult_renders_tips_and_records_hits() {
        let mut engine = engine_with_tips();
        let tip_id = engine.bank().tips()[0].tip_id.clone();

        let result = engine.consult("search the sources");

        assert_ne!(result.tips, Vec::<TipRecord>::new());
        assert!(result.rendered.starts_with("- ["));
        assert_eq!(engine.consult_count(), 1);
        let hits: u64 = engine.bank().tips().iter().map(|tip| tip.hits).sum();
        assert_eq!(hits, result.tips.len() as u64);
        let successes: u64 = engine.bank().tips().iter().map(|tip| tip.successes).sum();
        assert_eq!(successes, result.tips.len() as u64);
        let consulted = engine.bank().get(&tip_id).expect("tip kept");
        assert!(consulted.hits >= 1);
        assert_eq!(consulted.successes, consulted.hits);

        let disabled = TtseConfig {
            inject_enabled: false,
            ..TtseConfig::default()
        };
        let mut engine = TtseEngine::new(disabled).expect("valid config");
        engine
            .bank_mut()
            .add(TipRecord::new("Prefer rg over grep", TipCategory::Tool).expect("valid rule"));
        let result = engine.consult("search");
        assert_eq!(result, ConsultResult::default());
        assert_eq!(
            engine.consult_count(),
            0,
            "disabled injection does not count"
        );
        assert_eq!(engine.bank().tips()[0].hits, 0);
    }

    #[tokio::test]
    async fn save_and_load_round_trip() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("nested").join("bank.json");
        let engine = engine_with_tips();

        engine.save(&path).await.expect("save succeeds");

        assert!(path.exists());
        assert!(!temp_path(&path).exists(), "the temp file is gone");
        let mut loaded = TtseEngine::load(&path).await.expect("load succeeds");
        assert_eq!(loaded.bank(), engine.bank());
        assert_eq!(loaded.consult_count(), 0);
        assert_eq!(loaded.config().store_path, path.to_string_lossy());
        assert_eq!(loaded.stats(), engine.stats());
        assert_ne!(loaded.consult("").rendered, "");

        let custom = TtseConfig {
            consult_top_k: 2,
            dream_enabled: false,
            ..TtseConfig::default()
        };
        let loaded = TtseEngine::load_with_config(&path, custom.clone())
            .await
            .expect("load with config succeeds");
        assert_eq!(loaded.config(), &custom);
        assert_eq!(loaded.bank(), engine.bank());

        let broken = TtseConfig {
            capacity: 0,
            ..TtseConfig::default()
        };
        assert!(TtseEngine::load_with_config(&path, broken).await.is_err());
        let missing = directory.path().join("absent.json");
        let error = TtseEngine::load(&missing).await.expect_err("missing store");
        assert!(matches!(error, Error::External(_)), "got {error:?}");

        tokio::fs::write(directory.path().join("bad.json"), b"not json")
            .await
            .expect("write the bad store");
        let error = TtseEngine::load(directory.path().join("bad.json"))
            .await
            .expect_err("malformed store");
        assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    }
}
