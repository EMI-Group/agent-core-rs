//! The offline, model-free [`TaskMemoryService`] implementation.
//!
//! Python's `TaskMemoryService.retrieve`/`summarize`
//! (`openjiuwen/extensions/context_evolver/service/task_memory_service.py`) both
//! drive a `BaseOp` pipeline whose stages call an LLM, an embedding model, or
//! Milvus. This module keeps the *contract* those methods satisfy and replaces
//! the calls with deterministic heuristics, so the crate is fully usable
//! offline:
//!
//! * **retrieve** ranks the user's stored memories lexically
//!   ([`super::ranking`]) and, when an embedding backend is configured, blends
//!   in a cosine signal (`0.7 * lexical + 0.3 * cosine`) computed over an
//!   in-process [`MemoryVectorStore`]. ACE is
//!   unranked exactly like Python's playbook load: id order, every hit `1.0`,
//!   capped at [`TaskMemoryConfig::ace_bullet_limit`].
//! * **summarize** is *extractive*: every trajectory becomes one memory whose
//!   `content` is the (whitespace-collapsed, truncated) trajectory response and
//!   whose `when_to_use` is the query. Python distills new strategies with an
//!   LLM; this port records what happened, deduplicates within the batch, and
//!   persists through [`MemoryStore`]. The extractive shape is deliberate: it is
//!   deterministic, it never invents advice, and it is enough to exercise the
//!   retrieve path end to end.
//! * **`add_memory`** reuses the id of a stored record with the same
//!   [`MemoryRecord::identity`], so re-adding one memory updates it instead of
//!   duplicating it, and **`load_memories`** projects the stored records back
//!   onto [`AddMemoryRequest`].
//!
//! # Deviations from Python
//!
//! * No model calls, no Milvus, and no `MaTTS` trajectory generation; the
//!   LLM-driven service lives in a sibling module that reuses this module's
//!   configuration and ranking helpers.
//! * Python scores retrieval with vector similarity alone; this port keeps a
//!   lexical fallback so retrieval works with no embedder, and clamps negative
//!   cosines to `0.0` before blending (a dissimilar memory never loses lexical
//!   ground to an anti-correlated one).
//! * Python's summarize returns algorithm-specific memory objects; this port
//!   returns the `AddMemoryRequest` shapes documented on
//!   [`create_task_memory_service`](super::create_task_memory_service) and in
//!   `super::memories_json`.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/`
//! (`test_retrieve_flow.py`, `test_summary_flow.py`); the live flows need an API
//! key in Python and are skipped there, which is why this module exists.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde_json::{json, Value};
use tracing::debug;

use super::super::prompt::{parse_trajectories, render_memory_string, RetrievedHit, Trajectory};
use super::super::record::{generate_memory_id, MemoryDraft, MemoryRecord};
use super::super::store::MemoryStore;
use super::super::{
    AddMemoryRequest, Algo, MemoryVectorStore, RetrieveRequest, RetrieveResponse, SummarizeRequest,
    SummarizeResponse, TaskMemoryService,
};
use super::ranking;
use super::{
    collapse, finalize_hits, memories_json, MemoryEmbedding, TaskMemoryConfig, COSINE_WEIGHT,
    LEXICAL_WEIGHT, LOG_TARGET,
};

/// The `status` every successful response card carries (Python's literal).
const SUCCESS: &str = "success";
/// Maximum characters kept from a trajectory query in the extractive draft.
const QUERY_LIMIT: usize = 200;
/// Maximum characters kept from a trajectory response in the extractive draft.
const CONTENT_LIMIT: usize = 400;
/// The section Cognition memories are filed under.
const COGNITION_SECTION: &str = "cognition";
/// The section every other algorithm files its extractive draft under.
const GENERAL_SECTION: &str = "general";

/// Offline [`TaskMemoryService`]: lexical ranking plus an optional embedding
/// cosine, with no model calls.
///
/// ```no_run
/// # async fn demo() -> Result<(), openjiuwen_common::error::Error> {
/// use std::sync::Arc;
/// use openjiuwen_extensions::context_evolver::{
///     create_task_memory_service, Algo, InMemoryMemoryStore, MemoryBackend, RetrieveRequest,
///     TaskMemoryConfig,
/// };
///
/// let service = create_task_memory_service(
///     TaskMemoryConfig::new(Algo::Reme).with_top_k(3),
///     MemoryBackend::Deterministic,
///     Arc::new(InMemoryMemoryStore::new()),
/// );
/// let response = service
///     .retrieve(RetrieveRequest { user_id: "user".into(), query: "search".into() })
///     .await?;
/// assert!(response.memory_string.is_empty());
/// # Ok(())
/// # }
/// ```
///
/// Behavior worth relying on:
///
/// * `retrieve` loads the user's records, ranks them (see the module docs), drops
///   hits below [`TaskMemoryConfig::min_score`], keeps at most
///   [`TaskMemoryConfig::top_k`], and renders `memory_string` with the
///   algorithm family's Python format
///   ([`render_memory_string`]);
///   `retrieved_memory` is the `{"memories": [...]}` card from
///   `super::memories_json`. An empty store answers `success` with an empty
///   string and `{"memories": []}`.
/// * `summarize` never calls a model: it parses the trajectories, builds one
///   extractive [`MemoryDraft`] per trajectory, deduplicates by identity, upserts
///   every surviving record, and returns the created `AddMemoryRequest` shapes.
///   An empty trajectory list answers `success` with `{"memories": []}` and
///   writes nothing.
/// * `add_memory` validates the request-derived record (`Error::Validation` for
///   blank content), reuses the id of a stored record with the same identity, and
///   upserts.
/// * `load_memories` projects the stored records back onto `AddMemoryRequest`.
///
/// The service holds the store behind an [`Arc`], so several services (or a host)
/// can share one backend. No lock guard is held across an `.await`: the store
/// owns its own synchronisation.
#[derive(Debug)]
pub struct DeterministicTaskMemoryService {
    /// The algorithm whose rendering and id prefix this service uses.
    algo: Algo,
    /// The knobs every service shares.
    config: TaskMemoryConfig,
    /// Persistence seam.
    store: Arc<dyn MemoryStore>,
    /// Optional cosine signal; `None` means lexical ranking only.
    embedding: Option<Arc<dyn MemoryEmbedding>>,
}

impl DeterministicTaskMemoryService {
    /// Builds a lexical-only service for `algo` over `config` and `store`.
    ///
    /// The `algo` argument wins over [`TaskMemoryConfig::algo`], so
    /// `algo`/`config.algo` never disagree; every other knob comes from `config`.
    /// Call [`DeterministicTaskMemoryService::with_embedding`] to add the cosine
    /// signal.
    #[must_use]
    pub fn new(algo: Algo, mut config: TaskMemoryConfig, store: Arc<dyn MemoryStore>) -> Self {
        config.algo = algo;
        Self {
            algo,
            config,
            store,
            embedding: None,
        }
    }

    /// Attaches an embedding backend, enabling the blended ranking.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn MemoryEmbedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// The algorithm this service renders and prefixes ids with.
    #[must_use]
    pub const fn algo(&self) -> Algo {
        self.algo
    }

    /// Ranks `records` for `query`, before the shared score floor and `top_k`.
    async fn rank(&self, query: &str, records: &[MemoryRecord]) -> Result<Vec<RetrievedHit>> {
        if self.algo == Algo::Ace {
            return Ok(ranking::rank_candidates(
                Algo::Ace,
                query,
                records,
                self.config.ace_bullet_limit,
            ));
        }
        let hits = ranking::rank_candidates(self.algo, query, records, self.config.candidate_limit);
        if hits.is_empty() {
            return Ok(hits);
        }
        match self.embedding.as_deref() {
            Some(embedding) => self.blend(query, hits, records, embedding).await,
            None => Ok(hits),
        }
    }

    /// Blends the lexical scores in `hits` with embedding cosine similarity.
    ///
    /// One vector per record (the embedding of [`ranking::scoring_text`]) is
    /// upserted into a fresh [`MemoryVectorStore`], the query is embedded once,
    /// and every hit is rescored:
    ///
    /// * `0.7 * lexical + 0.3 * cosine.max(0.0)` normally;
    /// * `cosine.max(0.0)` when every lexical score is `0.0`, because the query
    ///   shares no token with any candidate and the lexical scale carries no
    ///   information.
    ///
    /// The result keeps the input order; [`super::finalize_hits`] sorts it.
    #[allow(clippy::float_cmp)] // Zero is the documented "no lexical overlap" sentinel.
    async fn blend(
        &self,
        query: &str,
        hits: Vec<RetrievedHit>,
        records: &[MemoryRecord],
        embedding: &dyn MemoryEmbedding,
    ) -> Result<Vec<RetrievedHit>> {
        let index = MemoryVectorStore::new();
        let mut nodes = Vec::with_capacity(records.len());
        for record in records {
            let text = ranking::scoring_text(record);
            nodes.push(record.to_vector_node(embedding.embed(&text).await?));
        }
        index.upsert(nodes).await?;
        let query_vector = embedding.embed(query).await?;
        let scored = index.search(&query_vector, records.len()).await?;
        let cosines: BTreeMap<&str, f32> = scored
            .iter()
            .map(|scored| (scored.node.id.as_str(), scored.score))
            .collect();
        let flat_lexical = hits.iter().all(|hit| hit.score == 0.0);
        Ok(hits
            .into_iter()
            .map(|hit| {
                let similarity = cosines
                    .get(hit.id.as_str())
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0);
                let score = if flat_lexical {
                    similarity
                } else {
                    LEXICAL_WEIGHT * hit.score + COSINE_WEIGHT * similarity
                };
                RetrievedHit { score, ..hit }
            })
            .collect())
    }

    /// Builds the extractive draft for one trajectory, or `None` when the
    /// response carries no text.
    fn draft_for(&self, trajectory: &Trajectory) -> Option<MemoryDraft> {
        let content = collapse(&trajectory.response, CONTENT_LIMIT);
        if content.is_empty() {
            debug!(
                target: LOG_TARGET,
                algo = self.algo.as_str(),
                "trajectory has no response text; skipping it"
            );
            return None;
        }
        let query = collapse(&trajectory.query, QUERY_LIMIT);
        let feedback = trajectory.feedback.as_str();
        let subject = if query.is_empty() {
            "(unspecified task)"
        } else {
            query.as_str()
        };
        let description = format!("{feedback} outcome of a task like: {subject}");
        let mut attributes = BTreeMap::new();
        attributes.insert("deterministic".to_string(), Value::Bool(true));
        attributes.insert("feedback".to_string(), Value::from(feedback));
        Some(MemoryDraft {
            content,
            when_to_use: (!query.is_empty()).then_some(query),
            title: None,
            description: Some(description),
            section: if self.algo == Algo::Cognition {
                COGNITION_SECTION
            } else {
                GENERAL_SECTION
            }
            .to_string(),
            label: None,
            attributes,
            source_type: Some(feedback.to_string()),
        })
    }
}

#[async_trait]
impl TaskMemoryService for DeterministicTaskMemoryService {
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse> {
        let records = self.store.load(&request.user_id).await?;
        if records.is_empty() {
            debug!(
                target: LOG_TARGET,
                user = %request.user_id,
                "no stored memories; answering with the empty card"
            );
            return Ok(RetrieveResponse {
                status: SUCCESS.to_string(),
                memory_string: String::new(),
                retrieved_memory: json!({"memories": []}),
            });
        }
        let hits = finalize_hits(self.rank(&request.query, &records).await?, &self.config);
        let memory_string = render_memory_string(self.algo, &hits, &records);
        debug!(
            target: LOG_TARGET,
            user = %request.user_id,
            algo = self.algo.as_str(),
            hits = hits.len(),
            "retrieved memories"
        );
        Ok(RetrieveResponse {
            status: SUCCESS.to_string(),
            memory_string,
            retrieved_memory: memories_json(&hits, &records),
        })
    }

    async fn summarize(&self, request: SummarizeRequest) -> Result<SummarizeResponse> {
        let trajectories = parse_trajectories(&request.trajectories)?;
        let mut seen = BTreeSet::new();
        let mut created: Vec<AddMemoryRequest> = Vec::with_capacity(trajectories.len());
        for trajectory in &trajectories {
            let Some(draft) = self.draft_for(trajectory) else {
                continue;
            };
            let record = MemoryRecord::new(request.user_id.as_str(), draft.clone())
                .with_id(generate_memory_id(self.algo, &request.user_id, &draft));
            record.validate()?;
            if !seen.insert(record.identity()) {
                debug!(
                    target: LOG_TARGET,
                    id = %record.id,
                    "duplicate memory in the batch; keeping the first"
                );
                continue;
            }
            self.store.upsert(record.clone()).await?;
            created.push(record.to_request());
        }
        debug!(
            target: LOG_TARGET,
            user = %request.user_id,
            algo = self.algo.as_str(),
            trajectories = trajectories.len(),
            created = created.len(),
            "summarized trajectories"
        );
        Ok(SummarizeResponse {
            status: SUCCESS.to_string(),
            memories: json!({"memories": created}),
        })
    }

    async fn add_memory(&self, user_id: &str, request: AddMemoryRequest) -> Result<()> {
        let record = MemoryRecord::from_request(user_id, &request);
        let identity = record.identity();
        let id = self
            .store
            .load(user_id)
            .await?
            .iter()
            .find(|stored| stored.identity() == identity)
            .map_or_else(
                || generate_memory_id(self.algo, user_id, &record.to_draft()),
                |stored| stored.id.clone(),
            );
        let record = record.with_id(id);
        record.validate()?;
        self.store.upsert(record).await
    }

    fn load_memories(&self, user_id: &str) -> Result<Vec<AddMemoryRequest>> {
        // The seam's `load_memories` is synchronous (Python reads persisted
        // memories eagerly), so the store round trip is driven to completion on
        // the caller's thread. Neither bundled store blocks on a network: the
        // in-memory store resolves immediately, and the JSON store's
        // `tokio::fs` work runs on the blocking pool, which completes
        // independently of the waiting thread.
        let records = futures::executor::block_on(self.store.load(user_id))?;
        Ok(records.iter().map(MemoryRecord::to_request).collect())
    }
}

#[cfg(test)]
mod tests {
    // Python counterparts: `tests/unit_tests/extensions/context_evolver/`
    // (`test_retrieve_flow.py`, `test_summary_flow.py`). Python skips these
    // without an API key; the deterministic service is what runs offline.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::{InMemoryMemoryStore, MemoryStore};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, user_id: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: user_id.to_string(),
            content: content.to_string(),
            ..MemoryRecord::default()
        }
    }

    async fn service_with(
        algo: Algo,
        config: TaskMemoryConfig,
        records: Vec<MemoryRecord>,
    ) -> (DeterministicTaskMemoryService, Arc<InMemoryMemoryStore>) {
        let store = Arc::new(InMemoryMemoryStore::new());
        for record in records {
            store.upsert(record).await.unwrap();
        }
        (
            DeterministicTaskMemoryService::new(algo, config, store.clone()),
            store,
        )
    }

    fn retrieved_ids(response: &RetrieveResponse) -> Vec<String> {
        response.retrieved_memory["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|memory| memory["id"].as_str().unwrap().to_string())
            .collect()
    }

    fn request(user_id: &str, query: &str) -> RetrieveRequest {
        RetrieveRequest {
            user_id: user_id.to_string(),
            query: query.to_string(),
        }
    }

    #[tokio::test]
    async fn retrieve_orders_by_lexical_score_and_honours_top_k() {
        let (service, _) = service_with(
            Algo::ReasoningBank,
            TaskMemoryConfig::new(Algo::ReasoningBank).with_top_k(2),
            vec![
                record("a", "user", "alpha beta gamma"),
                record("b", "user", "alpha beta"),
                record("c", "user", "unrelated text"),
            ],
        )
        .await;
        assert_eq!(service.algo(), Algo::ReasoningBank);

        let response = service
            .retrieve(request("user", "alpha beta"))
            .await
            .unwrap();
        assert_eq!(response.status, "success");
        assert_eq!(retrieved_ids(&response), ["b", "a"]);
        assert!(response.memory_string.contains("alpha beta"));
        assert!(!response.memory_string.contains("unrelated text"));
        let first = &response.retrieved_memory["memories"][0];
        assert_eq!(first["content"], json!("alpha beta"));
        assert_eq!(first["score"], json!(1.0));
        assert_eq!(first["section"], json!("general"));
        assert!(first.as_object().unwrap().get("title").is_none());
    }

    #[tokio::test]
    async fn retrieve_drops_hits_below_min_score() {
        let (service, _) = service_with(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme).with_min_score(0.8),
            vec![
                record("a", "user", "alpha beta"),
                record("b", "user", "alpha"),
                record("c", "user", "gamma"),
            ],
        )
        .await;

        let response = service
            .retrieve(request("user", "alpha beta"))
            .await
            .unwrap();
        assert_eq!(retrieved_ids(&response), ["a"]);
        assert!(!response.memory_string.contains("gamma"));
    }

    #[tokio::test]
    async fn retrieve_returns_the_empty_card_for_an_empty_store() {
        let (service, _) = service_with(
            Algo::Cognition,
            TaskMemoryConfig::new(Algo::Cognition),
            Vec::new(),
        )
        .await;

        let response = service
            .retrieve(request("nobody", "anything"))
            .await
            .unwrap();
        assert_eq!(response.status, "success");
        assert_eq!(response.memory_string, "");
        assert_eq!(response.retrieved_memory, json!({"memories": []}));
    }

    #[tokio::test]
    async fn retrieve_ace_returns_the_playbook_in_id_order_up_to_the_ceiling() {
        let (service, _) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace).with_ace_bullet_limit(2),
            vec![
                record("c", "user", "third bullet"),
                record("a", "user", "first bullet"),
                record("b", "user", "second bullet"),
            ],
        )
        .await;

        let response = service
            .retrieve(request("user", "no overlap at all"))
            .await
            .unwrap();
        assert_eq!(retrieved_ids(&response), ["a", "b"]);
        assert!(response.retrieved_memory["memories"]
            .as_array()
            .unwrap()
            .iter()
            .all(|memory| memory["score"] == json!(1.0)));
        assert!(response.memory_string.contains("first bullet"));
        assert!(!response.memory_string.contains("third bullet"));
    }

    #[tokio::test]
    async fn retrieve_embedding_signal_reranks_a_frequency_variant() {
        let records = vec![
            record("a", "user", "alpha beta gamma"),
            record("b", "user", "alpha alpha alpha beta gamma"),
        ];
        let config = TaskMemoryConfig::new(Algo::ReasoningBank).with_top_k(2);
        let (lexical, store) =
            service_with(Algo::ReasoningBank, config.clone(), records.clone()).await;
        let blended = DeterministicTaskMemoryService::new(Algo::ReasoningBank, config, store)
            .with_embedding(Arc::new(super::super::HashingEmbedding::default()));

        // Both memories carry the same token set, so lexical ranking ties and
        // falls back to id order; the embedder sees the query-token frequency
        // and prefers `b`.
        let lexical_ids = retrieved_ids(
            &lexical
                .retrieve(request("user", "alpha beta"))
                .await
                .unwrap(),
        );
        assert_eq!(lexical_ids, ["a", "b"]);

        let response = blended
            .retrieve(request("user", "alpha beta"))
            .await
            .unwrap();
        assert_eq!(retrieved_ids(&response), ["b", "a"]);
        let scores: Vec<f64> = response.retrieved_memory["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|memory| memory["score"].as_f64().unwrap())
            .collect();
        assert!(scores[0] > scores[1], "{scores:?}");
        assert!(scores.iter().all(|score| (0.0..=1.0).contains(score)));
        assert!(scores.iter().all(|score| !score.is_nan()));
    }

    #[tokio::test]
    async fn retrieve_with_an_embedder_keeps_the_order_when_no_tokens_match() {
        let records = vec![
            record("b", "user", "gamma delta"),
            record("a", "user", "gamma"),
        ];
        let service = DeterministicTaskMemoryService::new(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme).with_top_k(5),
            Arc::new(InMemoryMemoryStore::new()),
        )
        .with_embedding(Arc::new(super::super::HashingEmbedding::default()));
        for record in records {
            service.store.upsert(record).await.unwrap();
        }

        let response = service
            .retrieve(request("user", "nothing in common"))
            .await
            .unwrap();
        // No shared tokens and no shared buckets: every score is `0.0`, so the
        // final ordering is the documented id tie-break.
        assert_eq!(retrieved_ids(&response), ["a", "b"]);
        assert!(response.retrieved_memory["memories"]
            .as_array()
            .unwrap()
            .iter()
            .all(|memory| memory["score"] == json!(0.0)));
    }

    #[tokio::test]
    async fn summarize_builds_extractive_drafts_and_persists_them() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = DeterministicTaskMemoryService::new(
            Algo::ReasoningBank,
            TaskMemoryConfig::new(Algo::ReasoningBank),
            store.clone(),
        );

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".to_string(),
                trajectories: json!([
                    {"query": "  find   a file ", "response": "use ripgrep", "feedback": "success"},
                    {"query": "search logs", "response": "pipe through jq", "feedback": "failure"}
                ]),
            })
            .await
            .unwrap();

        assert_eq!(response.status, "success");
        let created = response.memories["memories"].as_array().unwrap();
        assert_eq!(created.len(), 2);
        assert_eq!(created[0]["content"], json!("use ripgrep"));
        assert_eq!(created[0]["when_to_use"], json!("find a file"));
        assert_eq!(
            created[0]["description"],
            json!("helpful outcome of a task like: find a file")
        );
        assert_eq!(created[0]["section"], json!("general"));
        assert_eq!(created[0]["attributes"]["deterministic"], json!(true));
        assert_eq!(created[0]["attributes"]["feedback"], json!("helpful"));
        assert_eq!(created[1]["attributes"]["feedback"], json!("harmful"));

        let stored = store.load("user").await.unwrap();
        assert_eq!(stored.len(), 2);
        // `load` orders by id (a content hash), not by insertion order.
        let mut source_types: Vec<&str> = stored
            .iter()
            .map(|record| record.source_type.as_deref().unwrap())
            .collect();
        source_types.sort_unstable();
        assert_eq!(source_types, ["harmful", "helpful"]);
        assert!(stored
            .iter()
            .all(|record| record.id.starts_with("reasoning_bank_user_")));
        assert!(stored.iter().all(|record| !record.id.is_empty()));
    }

    #[tokio::test]
    async fn summarize_files_cognition_drafts_under_cognition() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = DeterministicTaskMemoryService::new(
            Algo::Cognition,
            TaskMemoryConfig::new(Algo::Cognition),
            store.clone(),
        );

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".to_string(),
                trajectories: json!([{"query": "q", "response": "r", "feedback": "neutral"}]),
            })
            .await
            .unwrap();
        assert_eq!(
            response.memories["memories"][0]["section"],
            json!("cognition")
        );
        assert_eq!(
            response.memories["memories"][0]["description"],
            json!("neutral outcome of a task like: q")
        );
        assert_eq!(store.load("user").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn summarize_deduplicates_within_the_batch() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = DeterministicTaskMemoryService::new(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            store.clone(),
        );

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".to_string(),
                trajectories: json!([
                    {"query": "same task", "response": "same answer", "feedback": "success"},
                    {"query": "same task", "response": "same answer", "feedback": "success"},
                    {"query": "other task", "response": "same answer", "feedback": "success"}
                ]),
            })
            .await
            .unwrap();

        assert_eq!(response.memories["memories"].as_array().unwrap().len(), 2);
        assert_eq!(store.load("user").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn summarize_writes_nothing_for_empty_or_blank_trajectories() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = DeterministicTaskMemoryService::new(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            store.clone(),
        );

        for trajectories in [
            json!([]),
            json!({"trajectories": []}),
            json!([{"query": "q", "response": "   "}]),
        ] {
            let response = service
                .summarize(SummarizeRequest {
                    user_id: "user".to_string(),
                    trajectories,
                })
                .await
                .unwrap();
            assert_eq!(response.status, "success");
            assert_eq!(response.memories, json!({"memories": []}));
        }
        assert_eq!(store.load("user").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn summarize_rejects_malformed_trajectories() {
        let service = DeterministicTaskMemoryService::new(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            Arc::new(InMemoryMemoryStore::new()),
        );
        let error = service
            .summarize(SummarizeRequest {
                user_id: "user".to_string(),
                trajectories: json!("not a trajectory list"),
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn add_memory_is_idempotent_for_identical_content() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = DeterministicTaskMemoryService::new(
            Algo::ReasoningBank,
            TaskMemoryConfig::new(Algo::ReasoningBank),
            store.clone(),
        );
        let request = || AddMemoryRequest {
            content: "prefer ripgrep over grep".to_string(),
            when_to_use: Some("searching a repo".to_string()),
            ..AddMemoryRequest::default()
        };

        service.add_memory("user", request()).await.unwrap();
        let first = store.load("user").await.unwrap();
        service.add_memory("user", request()).await.unwrap();
        let second = store.load("user").await.unwrap();

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(first[0].id, second[0].id);
        assert!(second[0].id.starts_with("reasoning_bank_user_"));

        // A different memory gets its own id.
        service
            .add_memory(
                "user",
                AddMemoryRequest {
                    content: "prefer rg for speed".to_string(),
                    ..AddMemoryRequest::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(store.load("user").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn add_memory_validates_blank_content_and_user() {
        let service = DeterministicTaskMemoryService::new(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            Arc::new(InMemoryMemoryStore::new()),
        );

        let error = service
            .add_memory("user", AddMemoryRequest::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("blank content"),
            "{}",
            error.message()
        );

        let error = service
            .add_memory(
                "  ",
                AddMemoryRequest {
                    content: "body".to_string(),
                    ..AddMemoryRequest::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("blank user_id"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn load_memories_round_trips_through_the_store() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let service = DeterministicTaskMemoryService::new(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            store.clone(),
        );
        service
            .add_memory(
                "user",
                AddMemoryRequest {
                    content: "reset the cache".to_string(),
                    when_to_use: Some("after a schema change".to_string()),
                    title: Some("Cache".to_string()),
                    description: Some("How to reset".to_string()),
                    section: "tooling".to_string(),
                    label: Some("cache".to_string()),
                    ..AddMemoryRequest::default()
                },
            )
            .await
            .unwrap();

        let loaded = service.load_memories("user").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "reset the cache");
        assert_eq!(
            loaded[0].when_to_use.as_deref(),
            Some("after a schema change")
        );
        assert_eq!(loaded[0].title.as_deref(), Some("Cache"));
        assert_eq!(loaded[0].description.as_deref(), Some("How to reset"));
        assert_eq!(loaded[0].section, "tooling");
        assert_eq!(loaded[0].label.as_deref(), Some("cache"));
        assert_eq!(loaded[0].query, None);

        assert_eq!(service.load_memories("nobody").unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn load_memories_reads_the_json_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::context_evolver::JsonFileMemoryStore::new(dir.path()));
        let service = DeterministicTaskMemoryService::new(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            store,
        );
        service
            .add_memory(
                "user",
                AddMemoryRequest {
                    content: "persisted".to_string(),
                    ..AddMemoryRequest::default()
                },
            )
            .await
            .unwrap();

        let loaded = service.load_memories("user").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "persisted");
    }
}
