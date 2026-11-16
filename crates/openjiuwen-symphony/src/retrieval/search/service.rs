//! The public search service: initialization configs, result rows and the retriever facade.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.service` (v0.1.20) — `models.py`'s
//! public configs and `runtime_retriever_config_from_config`, and `retriever.py`'s
//! `Retriever` (`from_index`, `search`, `search_details`, `prepare_root`, the publicization
//! of a raw candidate and the trim of a public result). Python's `serving.py` is a demo
//! script and is deliberately not ported.
//!
//! # Two config layers
//!
//! [`RetrieverConfig`] is the caller-facing shape (Python `RetrieverConfig`), grouped into
//! [`TraversalConfig`], [`RenderConfig`] and [`GenerationConfig`];
//! [`runtime_config_from_config`] flattens it into the engine's
//! [`ProgressiveSearchConfig`],
//! clamping every bound the way Python's translator does. [`SearchRequest`] carries the
//! only per-query override, `top_k`.
//!
//! # What one search does
//!
//! [`Retriever::search_details_with`] validates the query, resolves the effective `top_k`,
//! runs the progressive walk
//! ([`ProgressiveRetriever`])
//! and then *publicizes* the raw candidates: each candidate's `payload` (the leaf `cid`) is
//! resolved through the index catalog to a [`CatalogRecord`], so a caller sees the public
//! skill name (`worker_id`), the record's name/description/branch path and its category
//! instead of internal ids. The rows are deduped (first occurrence wins), re-ranked from
//! `1`, and trimmed to the effective `top_k`. A payload with no catalog record falls back
//! to the candidate's own ids, so a stale catalog never turns a hit into an error.
//!
//! [`RetrievalResult::item_ids`] is what [`Retriever::search`] returns: the public
//! `worker_id`s in rank order, falling back to the candidate's item id when a record has no
//! worker id, so the list never holds an empty string.
//! [`RetrievalResult::summary_lines`] renders one line per candidate as
//! `"{rank}. {worker_id} (source={method})"`; this crate's candidates carry no score, so
//! Python's optional `score=` component is never emitted.
//!
//! # Deviations from Python
//!
//! - **The index is a local directory.** [`Retriever::load`] reads it with
//!   [`load_index`], the crate's three-artifact
//!   contract; Python's S3/GCS index materialization has no counterpart here.
//! - **No cached LLM client.** The model and the embedder are handed in as `Arc`s
//!   ([`Retriever::with_model`], [`Retriever::with_embedding`]); there is no client cache,
//!   no OAuth/OpenAI client construction and no `close()` — dropping the facade releases
//!   everything.
//! - **No thread parallelism.** [`TraversalConfig::max_parallel_branches`] and
//!   [`TraversalConfig::enable_parallel_branches`] are carried for configuration parity but
//!   ignored: the walk is sequential and deterministic.
//! - **A per-request `top_k` is clamped, never rejected.** Python raises for a request
//!   `top_k` that differs from the initialized one when a fixed-prefix-cache backend is
//!   configured; this crate has no such backend, so a larger request is clamped to the
//!   configured value and `0` keeps the crate's unlimited sentinel.
//! - **The result rows are typed.** Python's `candidate_records` are dicts; here they are
//!   [`CandidateRecord`] values, and no `elapsed_ms` is reported (the crate reads no wall
//!   clock).
//! - **The progressive runtime is bound on first use.** [`RetrieverConfig`] is translated
//!   eagerly in the constructor, while the [`ProgressiveRetriever`] itself is built (and
//!   cached) by the first [`Retriever::prepare`] or search, so an unusable embedder is
//!   reported by the call that needs it instead of being swallowed at construction time.
//!
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, PoisonError, RwLock};

use openjiuwen_retrieval::Embedding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::llm::{CompletionModel, FallbackMode};
use crate::retrieval::artifacts::load_index;
use crate::retrieval::search::codebooks::default_compact_boundary_codebook;
use crate::retrieval::search::config::{
    DisclosureConfig, EmbeddingScoringConfig, ProgressiveSearchConfig, SelectionMode,
};
use crate::retrieval::search::flat::FlatRetriever;
use crate::retrieval::search::flow::SelectionProvenance;
use crate::retrieval::search::progressive::{
    validate_query, ProgressiveRetriever, ProgressiveSearchResult, METHOD_LEXICAL_FALLBACK,
    METHOD_PROGRESSIVE,
};
use crate::retrieval::search::selector::{trace_degraded_selections, trace_selection_provenance};
use crate::retrieval::search::truncation::ScoreTruncationConfig;
use crate::retrieval::types::{
    CatalogRecord, LoadedIndex, RetrieverCandidate, RetrieverTraceEvent,
};

/// The `top_k` of the public retriever facade (Python `RetrieverConfig.top_k`).
///
/// Re-exported from `crate::retrieval::search::config::DEFAULT_TOP_K`, so the service and
/// the runtime config share one default.
pub use crate::retrieval::search::config::DEFAULT_TOP_K;

/// The progressive traversal and branch-search controls (Python `TraversalConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraversalConfig {
    /// How many child branches one node may select.
    pub max_branch_choices: usize,
    /// Carried for parity and **ignored**: the port searches branches sequentially.
    pub max_parallel_branches: usize,
    /// Carried for parity and **ignored**: there is no parallel branch search.
    pub enable_parallel_branches: bool,
    /// How many extra branches a selection may pick beyond `top_k`.
    pub branch_choice_slack: usize,
    /// How many extra candidates one branch may contribute beyond its share.
    pub branch_candidate_slack: usize,
    /// Reduce branch results round-robin so no branch monopolizes the list.
    pub round_robin_branch_reduce: bool,
}

impl Default for TraversalConfig {
    /// The Python `TraversalConfig` defaults.
    fn default() -> Self {
        Self {
            max_branch_choices: 6,
            max_parallel_branches: 3,
            enable_parallel_branches: true,
            branch_choice_slack: 2,
            branch_candidate_slack: 1,
            round_robin_branch_reduce: true,
        }
    }
}

/// The prompt rendering and tree disclosure controls (Python `RenderConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RenderConfig {
    /// Address candidates by compact code-book codes instead of names.
    pub compact_codes_enabled: bool,
    /// The code book to draw compact codes from; empty falls back to numeric codes `1`, …
    pub compact_codebook: Vec<String>,
    /// Render the whole candidate tree as one flat prompt payload.
    pub flatten_tree: bool,
    /// How many levels below the current node one call may expose.
    pub max_exposure_depth: usize,
    /// Expand a child when its subtree holds at most this many items.
    pub exposure_threshold: usize,
}

impl Default for RenderConfig {
    /// The Python `RenderConfig` production defaults, with
    /// [`default_compact_boundary_codebook`] as the code book.
    fn default() -> Self {
        Self {
            compact_codes_enabled: true,
            compact_codebook: default_compact_boundary_codebook().codes().to_vec(),
            flatten_tree: true,
            max_exposure_depth: 99,
            exposure_threshold: 1_000_000_000,
        }
    }
}

/// The candidate-generation controls (Python `GenerationConfig`).
///
/// Python's logit-selection knobs (`logit_*`) and its HTTP timeout have no counterpart in
/// this crate's runtime, so they are not carried; [`GenerationConfig::mode`] and
/// [`GenerationConfig::fallback_mode`] are the two that change behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GenerationConfig {
    /// How candidates are selected: ask a model, or rank locally.
    pub mode: SelectionMode,
    /// The advisory generation cap handed to a model.
    pub max_tokens: usize,
    /// Carried for parity: no constrained decoder is wired, so this changes nothing.
    pub trie_constrained_decoding_enabled: bool,
    /// What a stage does when the model fails.
    pub fallback_mode: FallbackMode,
}

impl Default for GenerationConfig {
    /// The Python `GenerationConfig` defaults: generate mode, 96 tokens, deterministic
    /// fallback.
    fn default() -> Self {
        Self {
            mode: SelectionMode::Generate,
            max_tokens: 96,
            trie_constrained_decoding_enabled: false,
            fallback_mode: FallbackMode::Deterministic,
        }
    }
}

/// The public retriever initialization configuration (Python `RetrieverConfig`).
///
/// A [`Retriever`] is built from one of these; changing it means building a new retriever,
/// while a per-query override goes through [`SearchRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverConfig {
    /// How many results one search returns by default.
    pub top_k: usize,
    /// The traversal controls.
    pub traversal: TraversalConfig,
    /// The rendering controls.
    pub render: RenderConfig,
    /// The generation controls.
    pub generation: GenerationConfig,
    /// The embedding blend weight: `0.0` is pure lexical (the default).
    ///
    /// Only used when an embedder is wired with [`Retriever::with_embedding`]; the value is
    /// clamped into `[0, 1]` (a non-finite weight becomes `0.0`) while translating.
    pub embedding_weight: f64,
    /// The score-truncation decision applied to a ranked list.
    pub truncation: ScoreTruncationConfig,
}

impl Default for RetrieverConfig {
    /// The Python `RetrieverConfig` defaults: ten results, the production traversal, render
    /// and generation shapes, a purely lexical ranking and the runtime truncation defaults.
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            traversal: TraversalConfig::default(),
            render: RenderConfig::default(),
            generation: GenerationConfig::default(),
            embedding_weight: 0.0,
            truncation: ScoreTruncationConfig::default(),
        }
    }
}

/// The per-query overrides that do not rebuild the runtime (Python `RequestConfig`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchRequest {
    /// The requested result count: `None` uses [`RetrieverConfig::top_k`], `0` means
    /// unlimited, and a larger value is clamped to the configured `top_k`.
    pub top_k: Option<usize>,
}

/// Flattens a public [`RetrieverConfig`] into the engine's runtime config.
///
/// Mirrors Python `runtime_retriever_config_from_config`:
///
/// - `top_k` is floored at `1` (Python's `max(1, top_k)`), so an unset or zero `top_k`
///   still returns one result;
/// - `max_tokens` is floored at `1`, `max_branch_choices` at `1`;
/// - the disclosure knobs come from [`RenderConfig`], the Rust-only bounds
///   ([`DisclosureConfig::max_chars`], `max_items_per_branch`, `max_description_chars`,
///   `code_width`) keep their conservative defaults because the public config does not
///   expose them;
/// - the embedding blend weight is clamped into `[0, 1]`;
/// - [`TraversalConfig::max_parallel_branches`],
///   [`TraversalConfig::enable_parallel_branches`] and
///   [`GenerationConfig::trie_constrained_decoding_enabled`] are carried for configuration
///   parity only: this crate's runtime searches sequentially and has no constrained
///   decoder.
///
/// `None` translates [`RetrieverConfig::default`].
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error) naming the offending field when the
/// translated config is unusable (an illegal compact code book, an unusable truncation
/// bound, …).
pub fn runtime_config_from_config(
    config: Option<&RetrieverConfig>,
) -> Result<ProgressiveSearchConfig> {
    let config = config.cloned().unwrap_or_default();
    let runtime = ProgressiveSearchConfig {
        top_k: config.top_k.max(1),
        max_tokens: config.generation.max_tokens.max(1),
        max_depth: 0,
        max_steps: 0,
        max_branch_choices: config.traversal.max_branch_choices.max(1),
        branch_choice_slack: config.traversal.branch_choice_slack,
        branch_candidate_slack: config.traversal.branch_candidate_slack,
        round_robin_branch_reduce: config.traversal.round_robin_branch_reduce,
        selection_mode: config.generation.mode,
        fallback_mode: config.generation.fallback_mode,
        disclosure: DisclosureConfig {
            max_exposure_depth_per_call: config.render.max_exposure_depth,
            exposure_threshold: config.render.exposure_threshold,
            compact_codes_enabled: config.render.compact_codes_enabled,
            compact_codebook: config.render.compact_codebook.clone(),
            flatten_full_tree_in_prompt: config.render.flatten_tree,
            ..DisclosureConfig::default()
        },
        embedding: EmbeddingScoringConfig::default().with_weight(config.embedding_weight),
        truncation: config.truncation,
    };
    runtime.validate()?;
    Ok(runtime)
}

/// One publicized candidate row of a [`RetrievalResult`].
///
/// Python keeps these as dicts; the port makes them typed. Every identity field is
/// public: `worker_id` is the catalog's public skill name, `cid` the internal leaf id it
/// came from, and `item_id` the candidate's own id inside its tree node.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CandidateRecord {
    /// The one-based rank after dedupe and trim.
    pub rank: usize,
    /// The candidate's own item id inside its tree node.
    pub item_id: String,
    /// The public skill name from the catalog (the candidate's item id when unresolved).
    pub worker_id: String,
    /// The catalog `cid` the candidate's payload named.
    pub cid: String,
    /// The catalog name, or the candidate's label when the catalog has none.
    pub name: String,
    /// The catalog description, or the candidate's own description.
    pub description: String,
    /// The ancestor `cid` chain of the item's node, root first.
    pub branch_path: Vec<String>,
    /// The catalog category, or the branch path dot-joined when the catalog has none.
    pub category: String,
    /// Whether this row is the result's first (selected) candidate.
    pub selected: bool,
}

/// One publicized search result.
///
/// Python's `SearchResult`, minus `elapsed_ms` (the crate reads no wall clock) and with
/// typed candidate rows instead of dicts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalResult {
    /// The backend that produced the result: `"progressive"` or `"lexical_fallback"`.
    pub method: String,
    /// The public `worker_id`s in rank order; this is what [`Retriever::search`] returns.
    pub item_ids: Vec<String>,
    /// The publicized candidate rows, best first.
    pub candidates: Vec<CandidateRecord>,
    /// One `"{rank}. {worker_id} (source={method})"` line per candidate.
    pub summary_lines: Vec<String>,
    /// The first candidate's public id, if any.
    pub selected_item_id: Option<String>,
    /// The first candidate's rank (`1`), if any.
    pub selected_rank: Option<usize>,
    /// The walk's trace events, in call order.
    pub trace_events: Vec<RetrieverTraceEvent>,
    /// How many nodes the walk visited.
    pub steps: usize,
    /// The provenance of the walk's last selection ([`SelectionProvenance::Lexical`] when the
    /// walk recorded no selection), so a caller that never reads logs still learns whether the
    /// configured model produced the answer.
    pub selection_provenance: SelectionProvenance,
    /// How many selections degraded away from the configured model (a missing model in
    /// generate mode, a failed model call or an unusable reply).
    pub degraded_selections: usize,
    /// Whether an over-large `SearchRequest::top_k` was clamped to
    /// [`RetrieverConfig::top_k`].
    pub top_k_clamped: bool,
}

/// A model bound to a [`Retriever`].
#[derive(Clone)]
struct ModelSelection {
    /// The completion model to select with.
    model: Arc<dyn CompletionModel>,
    /// The identity the trace and the model request carry.
    name: String,
}

/// The public retrieval facade over one loaded index.
///
/// Construction translates the config once ([`runtime_config_from_config`]) and keeps the
/// loaded index (plus a `cid` lookup over its catalog records); the progressive runtime is
/// bound (and cached) by the first [`Retriever::prepare`] or search. Every entry point is
/// `&self`-only, so one retriever serves concurrent searches; the only interior mutability
/// is that runtime cache and no lock is held across an `.await`.
pub struct Retriever {
    /// The loaded index (tree, choices, catalog, manifest, item paths).
    index: LoadedIndex,
    /// The catalog records keyed by `cid`, for publicization.
    records_by_cid: BTreeMap<String, CatalogRecord>,
    /// The caller-facing config.
    config: RetrieverConfig,
    /// The translated runtime config.
    runtime_config: ProgressiveSearchConfig,
    /// The model to select with, when one was wired.
    model: Option<ModelSelection>,
    /// The embedder to blend with, when one was wired.
    embedding: Option<Arc<dyn Embedding>>,
    /// The one-shot retriever (rebuilt when an embedder is wired).
    flat: FlatRetriever,
    /// The progressive runtime, built on first use and invalidated by the builders.
    progressive: RwLock<Option<Arc<ProgressiveRetriever>>>,
}

impl Retriever {
    /// Loads the index at `dir` and builds a retriever over it.
    ///
    /// # Errors
    ///
    /// Whatever [`load_index`] reports (a missing artifact is
    /// [`Error::NotFound`](crate::error::Error), an inconsistent one
    /// [`Error::Validation`](crate::error::Error)), and the config's validation errors.
    pub async fn load(dir: impl AsRef<Path>, config: RetrieverConfig) -> Result<Self> {
        let index = load_index(dir).await?;
        Self::from_loaded_index(index, config).await
    }

    /// Builds a retriever over an already-loaded index.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) when the translated config is unusable.
    // Kept `async` so both constructors share one awaitable shape and a future async
    // embedding warm-up does not change this signature.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn from_loaded_index(index: LoadedIndex, config: RetrieverConfig) -> Result<Self> {
        let runtime_config = runtime_config_from_config(Some(&config))?;
        let records_by_cid = index
            .records
            .iter()
            .map(|record| (record.cid.clone(), record.clone()))
            .collect();
        let flat = FlatRetriever::new(runtime_config.clone());
        Ok(Self {
            index,
            records_by_cid,
            config,
            runtime_config,
            model: None,
            embedding: None,
            flat,
            progressive: RwLock::new(None),
        })
    }

    /// Wires the model that drives the generate path.
    #[must_use]
    pub fn with_model(
        mut self,
        model: Arc<dyn CompletionModel>,
        model_name: impl Into<String>,
    ) -> Self {
        self.model = Some(ModelSelection {
            model,
            name: model_name.into(),
        });
        self.invalidate_runtime();
        self
    }

    /// Wires an embedder, turning the fallback ranking into a lexical/embedding blend.
    ///
    /// The blend weight is [`RetrieverConfig::embedding_weight`]; it is applied by both the
    /// progressive fallback selector and the flat retriever.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.flat =
            FlatRetriever::new(self.runtime_config.clone()).with_embedding(Arc::clone(&embedding));
        self.embedding = Some(embedding);
        self.invalidate_runtime();
        self
    }

    /// Warms one exposure per reachable node and reports how many were warmed.
    ///
    /// The warmed fragments are cached by the runtime, so a later [`Retriever::search`]
    /// costs no rendering and produces exactly the result an unwarmed search would.
    ///
    /// # Errors
    ///
    /// The seams' failures, and [`Error::Validation`](crate::error::Error) when the
    /// embedding blend is unusable.
    pub async fn prepare(&self) -> Result<usize> {
        self.progressive()?.prepare_root(None).await
    }

    /// Searches for `query` and returns the public `worker_id`s, best first.
    ///
    /// # Errors
    ///
    /// See [`Retriever::search_details_with`].
    pub async fn search(&self, query: &str) -> Result<Vec<String>> {
        self.search_with(query, &SearchRequest::default()).await
    }

    /// Searches for `query` with `request`'s overrides and returns the public ids.
    ///
    /// # Errors
    ///
    /// See [`Retriever::search_details_with`].
    pub async fn search_with(&self, query: &str, request: &SearchRequest) -> Result<Vec<String>> {
        Ok(self.search_details_with(query, request).await?.item_ids)
    }

    /// Searches for `query` and returns the publicized result.
    ///
    /// # Errors
    ///
    /// See [`Retriever::search_details_with`].
    pub async fn search_details(&self, query: &str) -> Result<RetrievalResult> {
        self.search_details_with(query, &SearchRequest::default())
            .await
    }

    /// Searches for `query` with `request`'s overrides and returns the publicized result.
    ///
    /// The effective `top_k` is [`RetrieverConfig::top_k`] unless the request overrides it:
    /// `Some(0)` keeps the crate's unlimited sentinel, and a larger request is clamped to
    /// the configured value (documented deviation: Python rejects that only for its
    /// fixed-prefix-cache backends, which this crate does not have).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `query` for a blank query,
    /// [`Error::Unsupported`](crate::error::Error) when generate mode needs a model that is
    /// not wired under [`FallbackMode::Error`], and whatever the walk's seams report.
    pub async fn search_details_with(
        &self,
        query: &str,
        request: &SearchRequest,
    ) -> Result<RetrievalResult> {
        validate_query(query)?;
        let requested = request.top_k;
        let top_k = self.resolve_top_k(requested);
        let top_k_clamped = matches!(
            requested,
            Some(value) if value != 0 && value > self.runtime_config.top_k
        );
        if top_k_clamped {
            tracing::warn!(
                requested = ?requested,
                configured = self.runtime_config.top_k,
                "the requested top_k exceeds the configured bound; clamping it"
            );
        }
        let result = self.progressive()?.search(query, Some(top_k)).await?;
        Ok(self.publicize(&result, top_k, top_k_clamped))
    }

    /// The loaded index the retriever serves.
    #[must_use]
    pub fn index(&self) -> &LoadedIndex {
        &self.index
    }

    /// The caller-facing config the retriever was built with.
    #[must_use]
    pub fn config(&self) -> &RetrieverConfig {
        &self.config
    }

    /// The translated runtime config, as the engine sees it.
    #[must_use]
    pub fn runtime_config(&self) -> &ProgressiveSearchConfig {
        &self.runtime_config
    }

    /// The flat one-shot retriever over the same runtime config and embedder.
    ///
    /// It ranks the whole catalog in one pass
    /// ([`FlatRetriever::retrieve`]/[`FlatRetriever::retrieve_choices`]) instead of
    /// walking the tree, which is the shape Python's legacy direct-recursion path served.
    #[must_use]
    pub fn flat(&self) -> &FlatRetriever {
        &self.flat
    }

    /// The backend the next search will use: [`METHOD_PROGRESSIVE`] with a model in generate
    /// mode, [`METHOD_LEXICAL_FALLBACK`] otherwise.
    #[must_use]
    pub fn method(&self) -> &'static str {
        if self.model.is_some() && self.config.generation.mode == SelectionMode::Generate {
            METHOD_PROGRESSIVE
        } else {
            METHOD_LEXICAL_FALLBACK
        }
    }

    /// The provenance the next search's selections will report.
    ///
    /// [`SelectionProvenance::LexicalNoModel`] means
    /// [`SelectionMode::Generate`](crate::retrieval::search::config::SelectionMode) is
    /// configured but no model is wired, so every selection will degrade to the lexical
    /// scorer (and warn once per runtime). [`SelectionProvenance::EmbeddingBlend`] means the
    /// lexical ranking is blended with the wired embedder and its positive weight.
    #[must_use]
    pub fn selection_provenance(&self) -> SelectionProvenance {
        match self.config.generation.mode {
            SelectionMode::Generate => {
                if self.model.is_some() {
                    SelectionProvenance::Model
                } else {
                    SelectionProvenance::LexicalNoModel
                }
            }
            SelectionMode::Lexical => {
                if self.embedding.is_some() && self.config.embedding_weight > 0.0 {
                    SelectionProvenance::EmbeddingBlend
                } else {
                    SelectionProvenance::Lexical
                }
            }
        }
    }

    /// How many exposures the bound runtime currently caches (`0` before it is bound).
    #[must_use]
    pub fn warmed_fragments(&self) -> usize {
        let cached = {
            let guard = self
                .progressive
                .read()
                .unwrap_or_else(PoisonError::into_inner);
            guard.clone()
        };
        cached.map_or(0, |runtime| runtime.warmed_fragments())
    }

    /// The effective `top_k` of one request.
    ///
    /// `None` uses the configured value, `Some(0)` stays the unlimited sentinel, and any
    /// other value is clamped to the configured `top_k`.
    fn resolve_top_k(&self, requested: Option<usize>) -> usize {
        let configured = self.runtime_config.top_k;
        match requested {
            None => configured,
            Some(0) => 0,
            Some(value) => value.min(configured),
        }
    }

    /// Drops the cached progressive runtime so the next search rebinds it.
    fn invalidate_runtime(&self) {
        let mut guard = self
            .progressive
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        *guard = None;
    }

    /// The progressive runtime, built once over the index root and the wired seams.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) from the runtime's own construction or
    /// from an unusable embedding blend.
    fn progressive(&self) -> Result<Arc<ProgressiveRetriever>> {
        let cached = {
            let guard = self
                .progressive
                .read()
                .unwrap_or_else(PoisonError::into_inner);
            guard.clone()
        };
        if let Some(runtime) = cached {
            return Ok(runtime);
        }
        let mut runtime = ProgressiveRetriever::new(
            Arc::new(self.index.root.clone()),
            self.runtime_config.clone(),
        )?;
        if let Some(selection) = &self.model {
            runtime = runtime.with_model(Arc::clone(&selection.model), selection.name.clone());
        }
        if let Some(embedding) = &self.embedding {
            runtime = runtime.with_embedding(Arc::clone(embedding))?;
        }
        let runtime = Arc::new(runtime);
        {
            let mut guard = self
                .progressive
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            *guard = Some(Arc::clone(&runtime));
        }
        Ok(runtime)
    }

    /// Turns a raw walk result into the public result, applying `top_k`.
    ///
    /// The rules are documented at the module level: resolve the payload through the
    /// catalog, dedupe by payload (then worker id), re-rank from `1`, mark the first row
    /// selected, trim to `top_k` (`0` = unlimited) and derive the ids, summary lines and
    /// selected row from the trimmed rows. The selection provenance is carried over from the
    /// walk's trace, so a degraded selection is visible on the public result too.
    fn publicize(
        &self,
        result: &ProgressiveSearchResult,
        top_k: usize,
        top_k_clamped: bool,
    ) -> RetrievalResult {
        let mut rows: Vec<CandidateRecord> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for candidate in &result.candidates {
            let row = self.publicize_candidate(candidate);
            let payload = candidate.payload.trim();
            let dedupe_key = if payload.is_empty() {
                format!("worker:{}", row.worker_id)
            } else {
                format!("payload:{payload}")
            };
            if !seen.insert(dedupe_key) {
                continue;
            }
            rows.push(row);
        }
        let limit = if top_k == 0 { usize::MAX } else { top_k };
        rows.truncate(limit);
        for (offset, row) in rows.iter_mut().enumerate() {
            row.rank = offset + 1;
            row.selected = offset == 0;
        }
        let method = result.method.clone();
        let item_ids: Vec<String> = rows
            .iter()
            .map(public_id)
            .filter(|id| !id.is_empty())
            .collect();
        let summary_lines = rows
            .iter()
            .map(|row| {
                format!(
                    "{}. {} (source={})",
                    row.rank,
                    public_id(row),
                    source_label(&method)
                )
            })
            .collect();
        RetrievalResult {
            method,
            selected_item_id: item_ids.first().cloned(),
            selected_rank: if item_ids.is_empty() { None } else { Some(1) },
            item_ids,
            candidates: rows,
            summary_lines,
            trace_events: result.trace.events().to_vec(),
            steps: result.steps,
            selection_provenance: trace_selection_provenance(&result.trace)
                .unwrap_or_else(|| self.selection_provenance()),
            degraded_selections: trace_degraded_selections(&result.trace),
            top_k_clamped,
        }
    }

    /// Publicizes one raw candidate through the catalog.
    fn publicize_candidate(&self, candidate: &RetrieverCandidate) -> CandidateRecord {
        let payload = candidate.payload.trim();
        let record = self.records_by_cid.get(payload);
        let worker_id = record
            .map(|record| record.worker_id.trim())
            .filter(|worker_id| !worker_id.is_empty())
            .map(str::to_owned)
            .or_else(|| non_blank(&candidate.item_id))
            .or_else(|| non_blank(&candidate.label))
            .unwrap_or_default();
        let name = record
            .and_then(|record| non_blank(&record.name))
            .or_else(|| non_blank(&candidate.label))
            .unwrap_or_else(|| worker_id.clone());
        let description = record
            .and_then(|record| non_blank(&record.description))
            .or_else(|| non_blank(&candidate.description))
            .unwrap_or_default();
        let branch_path = match record {
            Some(record) if !record.branch_path.is_empty() => record.branch_path.clone(),
            _ => candidate.branch_path.clone(),
        };
        let category = record
            .and_then(|record| non_blank(&record.category))
            .unwrap_or_else(|| branch_path.join("."));
        CandidateRecord {
            rank: 0,
            item_id: candidate.item_id.clone(),
            worker_id,
            cid: candidate.payload.clone(),
            name,
            description,
            branch_path,
            category,
            selected: false,
        }
    }
}

impl std::fmt::Debug for Retriever {
    /// Renders the config, the wired model/embedder and the runtime cache state.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Retriever")
            .field("records", &self.records_by_cid.len())
            .field("config", &self.config)
            .field(
                "model",
                &self.model.as_ref().map(|model| model.name.as_str()),
            )
            .field("embedding", &self.embedding.is_some())
            .field("warmed_fragments", &self.warmed_fragments())
            .finish_non_exhaustive()
    }
}

/// The public id of a row: its worker id, falling back to its own item id.
fn public_id(row: &CandidateRecord) -> String {
    non_blank(&row.worker_id)
        .or_else(|| non_blank(&row.item_id))
        .unwrap_or_default()
}

/// The `source` a summary line reports for `method`.
///
/// Python falls back to `unknown` for a missing source; a method is always present here, so
/// the fallback only guards an empty string.
fn source_label(method: &str) -> String {
    non_blank(method).unwrap_or_else(|| "unknown".to_owned())
}

/// `text` trimmed, or `None` when it is blank.
fn non_blank(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}
