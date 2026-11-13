//! The `context_evolver` service layer: shared configuration, ranking, and the
//! deterministic (offline) [`TaskMemoryService`].
//!
//! Ports the service half of Python's
//! `openjiuwen/extensions/context_evolver/service/task_memory_service.py`
//! (`TaskMemoryService.retrieve`/`summarize`/`add_memory`/`load_memories`) with
//! the model calls factored out behind seams:
//!
//! * [`TaskMemoryConfig`] holds the knobs every service shares (`top_k`,
//!   `candidate_limit`, ACE's playbook ceiling, the score floor).
//! * [`MemoryEmbedding`] is the optional embedding seam; [`HashingEmbedding`] is
//!   the offline implementer used for the cosine signal.
//! * [`MemoryBackend`] selects the implementation, and
//!   [`create_task_memory_service`] builds it.
//! * [`deterministic::DeterministicTaskMemoryService`] runs retrieve/summarize
//!   with lexical ranking only, so the crate is usable without any model.
//! * [`llm::LlmTaskMemoryService`] drives the same flows through an injected
//!   completion backend: prompt → completion → parse → persist.
//!
//! The model-driven service consumes exactly the same configuration, ranking
//! helpers ([`ranking`]), and response helpers (`memories_json`,
//! `finalize_hits`, `collapse`) that the deterministic service uses, so
//! switching backends does not change the shape of a response card.
//!
//! # Deviations from Python
//!
//! * Python builds one `TaskMemoryService` per algorithm and dispatches through
//!   `BaseOp` pipelines bound to a `ServiceContext` (LLM, embedding model,
//!   Milvus store). This port keeps one configuration plus an explicit backend
//!   enum, and persistence goes through [`super::MemoryStore`].
//! * Python's `retrieve` returns algorithm-specific `retrieved_memories`
//!   objects; this port always returns the shared
//!   `{"memories": [{"id", "content", "score", …}]}` card described on
//!   `memories_json`, because the persisted record is family-agnostic.
//! * Python derives a memory's embedding from separate `title`/`description`/
//!   `content` fields and stores it in Milvus; this port embeds
//!   [`ranking::scoring_text`] into an in-process [`super::MemoryVectorStore`].
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/`
//! (`test_retrieve_flow.py`, `test_summary_flow.py`); the live flows are skipped
//! in Python without an API key.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tracing::debug;

use super::completion::MemoryCompletion;
use super::prompt::RetrievedHit;
use super::record::MemoryRecord;
use super::store::MemoryStore;
use super::{Algo, TaskMemoryService, UnsupportedTaskMemoryService};

pub mod deterministic;
pub mod llm;
pub mod ranking;

pub use deterministic::DeterministicTaskMemoryService;
pub use llm::LlmTaskMemoryService;

/// The tracing target shared by this module tree.
pub(crate) const LOG_TARGET: &str = "openjiuwen_extensions::context_evolver::service";

/// Default for [`TaskMemoryConfig::top_k`] (Python retrieves a handful of
/// memories, not the whole store).
const DEFAULT_TOP_K: usize = 5;
/// Default for [`TaskMemoryConfig::candidate_limit`].
const DEFAULT_CANDIDATE_LIMIT: usize = 20;
/// Default for [`TaskMemoryConfig::ace_bullet_limit`]; Python's ACE playbook
/// ceiling.
const DEFAULT_ACE_BULLET_LIMIT: usize = 50;
/// Default for [`TaskMemoryConfig::min_score`]; keep every candidate.
const DEFAULT_MIN_SCORE: f32 = 0.0;
/// Dimensions of [`HashingEmbedding::default`].
const DEFAULT_HASHING_DIMS: usize = 256;
/// Weight of the lexical score in the blended ranking.
pub(crate) const LEXICAL_WEIGHT: f32 = 0.7;
/// Weight of the (clamped) cosine score in the blended ranking.
pub(crate) const COSINE_WEIGHT: f32 = 0.3;

/// Knobs shared by every `context_evolver` service.
///
/// [`TaskMemoryConfig::new`] applies the documented defaults; the `with_*`
/// builders override one knob each and keep the invariants (`top_k`,
/// `candidate_limit`, and `ace_bullet_limit` are at least `1`, `min_score` is
/// finite).
///
/// ```
/// use openjiuwen_extensions::context_evolver::{Algo, TaskMemoryConfig};
///
/// let config = TaskMemoryConfig::new(Algo::Reme).with_top_k(3).with_min_score(0.1);
/// assert_eq!(config.algo, Algo::Reme);
/// assert_eq!(config.top_k, 3);
/// assert_eq!(config.candidate_limit, 20);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskMemoryConfig {
    /// The memory algorithm whose prompts and rendering the service uses.
    pub algo: Algo,
    /// How many memories are injected into `memory_string` (default `5`).
    pub top_k: usize,
    /// How many records are offered to the ranking step (default `20`).
    pub candidate_limit: usize,
    /// The ACE playbook ceiling (Python's `50`).
    pub ace_bullet_limit: usize,
    /// Hits scoring below this value are dropped (default `0.0`).
    pub min_score: f32,
}

impl TaskMemoryConfig {
    /// Builds a configuration for `algo` with the documented defaults.
    #[must_use]
    pub fn new(algo: Algo) -> Self {
        Self {
            algo,
            top_k: DEFAULT_TOP_K,
            candidate_limit: DEFAULT_CANDIDATE_LIMIT,
            ace_bullet_limit: DEFAULT_ACE_BULLET_LIMIT,
            min_score: DEFAULT_MIN_SCORE,
        }
    }

    /// Sets [`TaskMemoryConfig::top_k`], clamped to at least `1`.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = clamp_limit(top_k);
        self
    }

    /// Sets [`TaskMemoryConfig::candidate_limit`], clamped to at least `1`.
    #[must_use]
    pub fn with_candidate_limit(mut self, candidate_limit: usize) -> Self {
        self.candidate_limit = clamp_limit(candidate_limit);
        self
    }

    /// Sets [`TaskMemoryConfig::ace_bullet_limit`], clamped to at least `1`.
    #[must_use]
    pub fn with_ace_bullet_limit(mut self, ace_bullet_limit: usize) -> Self {
        self.ace_bullet_limit = clamp_limit(ace_bullet_limit);
        self
    }

    /// Sets the score floor; a non-finite value resets it to the default.
    ///
    /// Keeping [`TaskMemoryConfig::min_score`] finite means the filter behaves
    /// like an ordinary `score >= min_score` comparison even when a host passes
    /// `f32::NAN`.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = if min_score.is_finite() {
            min_score
        } else {
            DEFAULT_MIN_SCORE
        };
        self
    }
}

impl Default for TaskMemoryConfig {
    /// Python's default algorithm is ACE, with the documented knobs.
    fn default() -> Self {
        Self::new(Algo::Ace)
    }
}

/// Coerces a configured size to at least `1`.
fn clamp_limit(limit: usize) -> usize {
    limit.max(1)
}

/// The scoring seam a service uses for an optional cosine signal.
///
/// Implementations are `Send + Sync + Debug` so a service can hold them behind
/// an [`Arc`] and still derive/clone its own state; the sibling
/// [`HashingEmbedding`] is the offline implementer, and a host may plug in a
/// real embedding model.
#[async_trait]
pub trait MemoryEmbedding: Send + Sync + fmt::Debug {
    /// Embeds `text`.
    ///
    /// # Errors
    ///
    /// Implementations report model or transport failures. Every returned
    /// vector must have a stable dimension per instance, because the service
    /// compares vectors with cosine similarity (mismatched dimensions score
    /// `0.0`).
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// A deterministic feature-hashing embedder.
///
/// Each token of [`ranking::tokenize`] is hashed with SHA-256; the first eight
/// digest bytes pick a bucket and the low bit of the ninth byte picks the sign,
/// so the vector is a signed bag-of-tokens sketch. The result is L2-normalised,
/// which makes cosine similarity independent of the text length.
///
/// Properties that matter for tests and offline runs: the same text always maps
/// to the same vector, no model or network is involved, and two texts sharing
/// tokens usually have a positive cosine even when their token sets differ in
/// size. The cost is that unrelated tokens may share a bucket (a "collision"
/// that looks like weak similarity); with the default 256 dimensions and short
/// memories this stays negligible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashingEmbedding {
    /// Number of buckets; always at least `1`.
    dims: usize,
}

impl HashingEmbedding {
    /// Builds an embedder with `dims` buckets, clamped to at least `1`.
    #[must_use]
    pub fn new(dims: usize) -> Self {
        Self {
            dims: clamp_limit(dims),
        }
    }

    /// The number of buckets this embedder produces per vector.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// The feature-hashing vector for `text`, before normalisation.
    fn hash(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dims];
        for token in ranking::tokenize(text) {
            let digest = Sha256::digest(token.as_bytes());
            let index = bucket_index(digest.as_slice(), self.dims);
            let sign = if digest.get(8).is_some_and(|byte| byte & 1 == 0) {
                1.0
            } else {
                -1.0
            };
            vector[index] += sign;
        }
        let norm = vector
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 && norm.is_finite() {
            for component in &mut vector {
                *component /= norm;
            }
        }
        vector
    }
}

impl Default for HashingEmbedding {
    /// The default embedder: `DEFAULT_HASHING_DIMS` dimensions (256).
    fn default() -> Self {
        Self::new(DEFAULT_HASHING_DIMS)
    }
}

/// Picks the bucket for `digest` without ever panicking on a short digest.
fn bucket_index(digest: &[u8], dims: usize) -> usize {
    let mut bytes = [0_u8; 8];
    if let Some(prefix) = digest.get(..8) {
        bytes.copy_from_slice(prefix);
    }
    let bucket = u64::from_be_bytes(bytes) % u64::try_from(dims).unwrap_or(u64::MAX);
    usize::try_from(bucket).unwrap_or_default()
}

#[async_trait]
impl MemoryEmbedding for HashingEmbedding {
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)] // No I/O: the hash is pure.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash(text))
    }
}

/// Which implementation backs a [`TaskMemoryService`].
///
/// [`MemoryBackend::Llm`] carries the completion backend the model-driven
/// service calls; construct it with [`Arc::new`] over any
/// [`MemoryCompletion`], for example
/// `MemoryBackend::Llm(Arc::new(ScriptedCompletion::new(["{}"])))`.
#[derive(Clone)]
pub enum MemoryBackend {
    /// The offline, model-free service ([`DeterministicTaskMemoryService`]).
    Deterministic,
    /// The model-driven service over the supplied completion backend.
    Llm(Arc<dyn MemoryCompletion>),
    /// The documented deferred default
    /// ([`super::UnsupportedTaskMemoryService`]).
    Unsupported,
}

impl fmt::Debug for MemoryBackend {
    /// Reports the variant only: a completion backend's fields are its own
    /// business (and may hold credentials or prompts), so they are never
    /// printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deterministic => formatter.write_str("MemoryBackend::Deterministic"),
            Self::Llm(_) => formatter.write_str("MemoryBackend::Llm(<completion backend>)"),
            Self::Unsupported => formatter.write_str("MemoryBackend::Unsupported"),
        }
    }
}

/// Builds the [`TaskMemoryService`] selected by `backend`.
///
/// [`MemoryBackend::Deterministic`] builds a
/// [`DeterministicTaskMemoryService`] over `config` and `store`;
/// [`MemoryBackend::Llm`] builds a [`LlmTaskMemoryService`] over `config`, the
/// supplied completion backend, and `store`, so retrieve/summarize become model
/// calls; [`MemoryBackend::Unsupported`] builds the documented deferred default.
#[must_use]
pub fn create_task_memory_service(
    config: TaskMemoryConfig,
    backend: MemoryBackend,
    store: Arc<dyn MemoryStore>,
) -> Arc<dyn TaskMemoryService> {
    match backend {
        MemoryBackend::Deterministic => Arc::new(DeterministicTaskMemoryService::new(
            config.algo,
            config,
            store,
        )),
        MemoryBackend::Llm(completion) => {
            Arc::new(LlmTaskMemoryService::new(config, completion, store))
        }
        MemoryBackend::Unsupported => Arc::new(UnsupportedTaskMemoryService::new(config.algo)),
    }
}

/// Renders the `retrieved_memory` card for the injected `hits`.
///
/// The shape is the one this crate documents for every service:
///
/// ```json
/// {"memories": [{"id": "…", "content": "…", "score": 0.5, "section": "general"}]}
/// ```
///
/// Each entry adds `when_to_use`, `title`, and `description` when the record
/// carries them, and omits them otherwise (never `null`). Hits keep their order;
/// a hit naming an id that `records` does not carry is skipped with a
/// `tracing::debug!`, which can only happen when a caller ranks records other
/// than the ones it passes in.
#[must_use]
pub(crate) fn memories_json(hits: &[RetrievedHit], records: &[MemoryRecord]) -> Value {
    let mut memories = Vec::with_capacity(hits.len());
    for hit in hits {
        let Some(record) = records.iter().find(|record| record.id == hit.id) else {
            debug!(
                target: LOG_TARGET,
                id = %hit.id,
                "retrieved hit names no record; skipping it"
            );
            continue;
        };
        let mut memory = Map::new();
        memory.insert("id".to_string(), Value::from(record.id.clone()));
        memory.insert("content".to_string(), Value::from(record.content.clone()));
        memory.insert("score".to_string(), Value::from(hit.score));
        for (key, value) in [
            ("when_to_use", &record.when_to_use),
            ("title", &record.title),
            ("description", &record.description),
        ] {
            if let Some(value) = value {
                memory.insert(key.to_string(), Value::from(value.clone()));
            }
        }
        memory.insert("section".to_string(), Value::from(record.section.clone()));
        memories.push(Value::Object(memory));
    }
    let mut card = Map::new();
    card.insert("memories".to_string(), Value::Array(memories));
    Value::Object(card)
}

/// Applies the shared post-ranking policy to `hits`.
///
/// Hits are ordered by descending score with an ascending-id tie-break, dropped
/// when they score below [`TaskMemoryConfig::min_score`], and truncated to
/// [`TaskMemoryConfig::top_k`]. Every service returns its hits through this
/// function, so the ordering and truncation rules cannot drift between backends.
#[must_use]
pub(crate) fn finalize_hits(
    mut hits: Vec<RetrievedHit>,
    config: &TaskMemoryConfig,
) -> Vec<RetrievedHit> {
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits.retain(|hit| hit.score >= config.min_score);
    hits.truncate(config.top_k);
    hits
}

/// Collapses whitespace runs in `text` and truncates it to `limit` characters.
///
/// Leading and trailing whitespace is dropped, every inner run becomes one
/// space, and the result is cut at a character boundary to at most `limit`
/// characters; a cut text ends with `…`, so the ellipsis counts towards the
/// limit. `limit == 0` yields an empty string.
#[must_use]
pub(crate) fn collapse(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if limit == 0 {
        return String::new();
    }
    let characters: Vec<char> = collapsed.chars().collect();
    if characters.len() <= limit {
        return collapsed;
    }
    let mut truncated: String = characters[..limit - 1].iter().collect();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    // Python counterparts: `extensions/context_evolver/service/task_memory_service.py`
    // (constructor defaults) and `tests/unit_tests/extensions/context_evolver/`
    // (retrieve/summary flows, skipped in Python without an API key).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::completion::ScriptedCompletion;
    use crate::context_evolver::prompt::RetrievedHit;
    use crate::context_evolver::{AddMemoryRequest, InMemoryMemoryStore, RetrieveRequest};
    use openjiuwen_common::error::ErrorKind;

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: content.to_string(),
            ..MemoryRecord::default()
        }
    }

    #[test]
    fn config_new_applies_the_documented_defaults() {
        let config = TaskMemoryConfig::new(Algo::Reme);
        assert_eq!(config.algo, Algo::Reme);
        assert_eq!(config.top_k, 5);
        assert_eq!(config.candidate_limit, 20);
        assert_eq!(config.ace_bullet_limit, 50);
        assert_eq!(config.min_score, 0.0);
        assert_eq!(TaskMemoryConfig::default().algo, Algo::Ace);
    }

    #[test]
    fn config_builders_clamp_and_round_trip_through_json() {
        let config = TaskMemoryConfig::new(Algo::Cognition)
            .with_top_k(0)
            .with_candidate_limit(0)
            .with_ace_bullet_limit(0)
            .with_min_score(0.25);
        assert_eq!(config.top_k, 1);
        assert_eq!(config.candidate_limit, 1);
        assert_eq!(config.ace_bullet_limit, 1);
        assert_eq!(config.min_score, 0.25);

        let encoded = serde_json::to_string(&config).unwrap();
        let decoded: TaskMemoryConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, config);

        let partial: TaskMemoryConfig =
            serde_json::from_str(r#"{"algo": "refcon", "top_k": 2}"#).unwrap();
        assert_eq!(partial.algo, Algo::Refcon);
        assert_eq!(partial.top_k, 2);
        assert_eq!(partial.candidate_limit, 20);
    }

    #[test]
    fn config_min_score_resets_non_finite_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                TaskMemoryConfig::new(Algo::Ace)
                    .with_min_score(value)
                    .min_score,
                0.0
            );
        }
        assert_eq!(
            TaskMemoryConfig::new(Algo::Ace)
                .with_min_score(-1.0)
                .min_score,
            -1.0
        );
    }

    #[tokio::test]
    async fn hashing_embedding_is_deterministic_and_normalised() {
        let embedder = HashingEmbedding::new(32);
        assert_eq!(embedder.dims(), 32);
        let first = embedder.embed("prefer ripgrep over grep").await.unwrap();
        let second = embedder.embed("prefer ripgrep over grep").await.unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
        let norm = first.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");

        let empty = embedder.embed("").await.unwrap();
        assert_eq!(empty, vec![0.0; 32]);
        assert_eq!(HashingEmbedding::new(0).dims(), 1);
        assert_eq!(HashingEmbedding::default().dims(), 256);
    }

    #[tokio::test]
    async fn hashing_embedding_ignores_token_order_and_case() {
        let embedder = HashingEmbedding::default();
        assert_eq!(
            embedder.embed("alpha beta").await.unwrap(),
            embedder.embed("BETA, alpha!").await.unwrap()
        );
        assert_ne!(
            embedder.embed("alpha").await.unwrap(),
            embedder.embed("beta").await.unwrap()
        );
    }

    #[test]
    fn memory_backend_debug_hides_the_completion_backend() {
        let backend = MemoryBackend::Llm(Arc::new(ScriptedCompletion::new(["{}"])));
        assert_eq!(
            format!("{backend:?}"),
            "MemoryBackend::Llm(<completion backend>)"
        );
        assert_eq!(
            format!("{:?}", MemoryBackend::Deterministic),
            "MemoryBackend::Deterministic"
        );
        assert_eq!(
            format!("{:?}", MemoryBackend::Unsupported),
            "MemoryBackend::Unsupported"
        );
    }

    #[tokio::test]
    async fn factory_builds_the_deterministic_service() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = create_task_memory_service(
            TaskMemoryConfig::new(Algo::ReasoningBank),
            MemoryBackend::Deterministic,
            store,
        );
        service
            .add_memory(
                "user",
                AddMemoryRequest {
                    content: "prefer ripgrep".into(),
                    ..AddMemoryRequest::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(service.load_memories("user").unwrap().len(), 1);

        let response = service
            .retrieve(RetrieveRequest {
                user_id: "user".into(),
                query: "ripgrep".into(),
            })
            .await
            .unwrap();
        assert!(response.memory_string.contains("prefer ripgrep"));
    }

    #[tokio::test]
    async fn factory_keeps_the_deferred_default_for_the_unsupported_backend() {
        let service = create_task_memory_service(
            TaskMemoryConfig::new(Algo::Refcon),
            MemoryBackend::Unsupported,
            Arc::new(InMemoryMemoryStore::new()),
        );
        let error = service.load_memories("user").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message().as_ref(),
            "context_evolver 'refcon' pipeline is deferred; inject a TaskMemoryService"
        );
    }

    #[tokio::test]
    async fn factory_wires_the_llm_backend_to_the_model_driven_service() {
        let service = create_task_memory_service(
            TaskMemoryConfig::new(Algo::Refcon),
            MemoryBackend::Llm(Arc::new(ScriptedCompletion::new([
                r#"{"ranked_indices": [0]}"#,
            ]))),
            Arc::new(InMemoryMemoryStore::new()),
        );
        // `Refcon` shares the `ReMe` prompts; an empty store answers without a
        // model call, which is enough to prove the arm is no longer deferred.
        let response = service
            .retrieve(crate::context_evolver::RetrieveRequest {
                user_id: "user".into(),
                query: "query".into(),
            })
            .await
            .unwrap();
        assert_eq!(response.status, "success");
        assert_eq!(response.memory_string, "");
    }

    #[test]
    fn memories_json_keeps_order_and_skips_absent_optional_fields() {
        let records = [
            record("b", "second"),
            MemoryRecord {
                title: Some("Title".into()),
                when_to_use: Some("When".into()),
                description: Some("Description".into()),
                ..record("a", "first")
            },
        ];
        let hits = vec![RetrievedHit::new("a", 0.75), RetrievedHit::new("b", 0.25)];
        let card = memories_json(&hits, &records);
        let memories = card["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0]["id"], serde_json::json!("a"));
        assert_eq!(memories[0]["content"], serde_json::json!("first"));
        assert_eq!(memories[0]["score"], serde_json::json!(0.75));
        assert_eq!(memories[0]["title"], serde_json::json!("Title"));
        assert_eq!(memories[0]["when_to_use"], serde_json::json!("When"));
        assert_eq!(memories[0]["description"], serde_json::json!("Description"));
        assert_eq!(memories[0]["section"], serde_json::json!("general"));
        assert!(!memories[1].as_object().unwrap().contains_key("title"));
        assert!(!memories[1].as_object().unwrap().contains_key("when_to_use"));
        assert!(!memories[1].as_object().unwrap().contains_key("description"));
        assert_eq!(
            memories_json(&[], &records)["memories"],
            serde_json::json!([])
        );

        // A hit naming no record is skipped instead of panicking.
        let orphan = memories_json(&[RetrievedHit::new("ghost", 1.0)], &records);
        assert_eq!(orphan["memories"], serde_json::json!([]));
    }

    #[test]
    fn finalize_hits_sorts_filters_and_truncates() {
        let config = TaskMemoryConfig::new(Algo::Reme)
            .with_top_k(2)
            .with_min_score(0.5);
        let hits = vec![
            RetrievedHit::new("d", 0.1),
            RetrievedHit::new("c", 0.9),
            RetrievedHit::new("b", 0.9),
            RetrievedHit::new("a", 0.6),
        ];
        let kept = finalize_hits(hits, &config);
        let ids: Vec<&str> = kept.iter().map(|hit| hit.id.as_str()).collect();
        assert_eq!(ids, ["b", "c"]);

        let config = TaskMemoryConfig::new(Algo::Ace).with_min_score(0.0);
        let kept = finalize_hits(vec![RetrievedHit::new("a", 0.0)], &config);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn collapse_collapses_and_truncates() {
        assert_eq!(collapse("  a\n\tb   c ", 10), "a b c");
        assert_eq!(collapse("", 10), "");
        assert_eq!(collapse("anything", 0), "");
        assert_eq!(collapse("abcdef", 6), "abcdef");
        assert_eq!(collapse("abcdefg", 6), "abcde…");
        assert_eq!(collapse("héllo wörld", 4), "hél…");
    }
}
