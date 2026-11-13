//! The model-driven [`TaskMemoryService`]: prompt → completion → parse →
//! persist.
//!
//! Ports the LLM half of Python's
//! `openjiuwen/extensions/context_evolver/service/task_memory_service.py`
//! (`TaskMemoryService.retrieve`/`summarize`/`add_memory`/`load_memories` over
//! the `retrieve/task/*` and `summary/task/*` operation pipelines).
//!
//! Every flow is the same three phases: build a prompt from the stored records
//! ([`super::super::prompt`]), call the injected
//! [`MemoryCompletion`], and either
//! persist the parsed reply or answer with the rendered memory card. Ranking,
//! response shaping, and the score floor are shared with the deterministic
//! service ([`super::ranking`], `finalize_hits`, `memories_json`), so switching
//! backends does not change the shape of a response card.
//!
//! * **retrieve** pre-ranks the user's records lexically
//!   ([`ranking::rank_candidates`]), keeps the top
//!   [`TaskMemoryConfig::candidate_limit`], asks the model to re-order exactly
//!   those candidates, and injects the re-ordered hits. [`Algo::Ace`] has no
//!   ranking prompt (Python loads the playbook unranked), so it injects up to
//!   [`TaskMemoryConfig::ace_bullet_limit`] records in id order with no model
//!   call at all.
//! * **summarize** sends one prompt built from the trajectories plus the stored
//!   records and applies the returned operations (`Add`/`Update`, `Remove`,
//!   `Tag`) to the store in reply order.
//! * **`add_memory`** writes one record without touching the model (it is a
//!   manual write, not a distillation), and **`load_memories`** projects the
//!   stored records back onto [`AddMemoryRequest`].
//!
//! # Malformed-output policy (deliberately asymmetric)
//!
//! * A retrieve reply that cannot be parsed **degrades**: the error kind is
//!   logged at `warn` — never the reply and never a memory — and the request
//!   still answers `success` with the pre-ranked candidate order. Python's
//!   rerank step does the same ("Failed to parse rerank response, using
//!   original order"): a broken ranking must not lose the memories entirely.
//!   This is an intentional deviation from failing strictly.
//! * A summarize reply that cannot be parsed is a **hard error**: Python raises
//!   `TOOLCHAIN_EVOLVING_MEMORY_SUMMARIZE_EXECUTION_ERROR` there, and silently
//!   writing nothing would hide a pipeline that produces no memory. The typed
//!   [`Error::Validation`](openjiuwen_common::error::Error::Validation) naming
//!   the algorithm's summarize phase comes from the prompt layer. An empty
//!   operation list is *not* an error: it means "nothing worth remembering" and
//!   answers `success` without writing anything.
//! * Inside a summarize batch one unusable `Add`/`Update` draft (blank content)
//!   is skipped with a `debug!` instead of failing the batch, and a `Remove`/
//!   `Tag` naming an id that is not stored is skipped the same way. One bad
//!   item never discards the items that are fine.
//!
//! # Deviations from Python
//!
//! * One completion call per phase: Python's separate rewrite/classify/rerank
//!   round trips are folded into the single prompt each family module builds.
//! * Persistence goes through [`MemoryStore`] instead of the vector store, so an
//!   `Add`/`Update` resolves its id by `MemoryRecord::identity` rather than by a
//!   vector-node id, and faces are the shared `{"memories": [...]}` card.
//! * Retrieval scores come from the reply's rank position and the pre-ranking is
//!   lexical, where Python scores by embedding similarity in Milvus.
//!
//! Python counterparts: `service/task_memory_service.py`, `retrieve/task/*/`
//! (`run.py`, `prompt.py`), and `summary/task/*/update.py`. The live flows need
//! an API key in Python and are exercised here with
//! [`ScriptedCompletion`](super::super::completion::ScriptedCompletion).
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde_json::{json, Value};
use tracing::{debug, warn};

use super::super::completion::MemoryCompletion;
use super::super::prompt::{
    self, render_memory_string, RetrievedHit, SummaryDraft, SummaryOperation,
};
use super::super::record::{generate_memory_id, MemoryRecord};
use super::super::store::MemoryStore;
use super::super::{
    AddMemoryRequest, Algo, RetrieveRequest, RetrieveResponse, SummarizeRequest, SummarizeResponse,
    TaskMemoryService,
};
use super::{finalize_hits, memories_json, ranking, TaskMemoryConfig, LOG_TARGET};

/// The `status` every successful response card carries (Python's literal).
const SUCCESS: &str = "success";

/// Model-driven [`TaskMemoryService`]: prompt → completion → parse → persist.
///
/// Build it with [`LlmTaskMemoryService::new`] or through
/// [`create_task_memory_service`](super::create_task_memory_service) and
/// [`MemoryBackend::Llm`](super::MemoryBackend::Llm); the service holds the
/// completion backend and the store behind [`Arc`]s, so several services (or a
/// host) can share one backend. No lock guard is ever held across an `.await`.
///
/// ```
/// # async fn demo() -> Result<(), openjiuwen_common::error::Error> {
/// use std::sync::Arc;
/// use openjiuwen_extensions::context_evolver::completion::ScriptedCompletion;
/// use openjiuwen_extensions::context_evolver::{
///     create_task_memory_service, AddMemoryRequest, Algo, InMemoryMemoryStore, MemoryBackend,
///     RetrieveRequest, TaskMemoryConfig,
/// };
///
/// let service = create_task_memory_service(
///     TaskMemoryConfig::new(Algo::Reme),
///     MemoryBackend::Llm(Arc::new(ScriptedCompletion::new([r#"{"ranked_indices": [0]}"#]))),
///     Arc::new(InMemoryMemoryStore::new()),
/// );
/// service
///     .add_memory(
///         "user",
///         AddMemoryRequest { content: "prefer ripgrep".into(), ..AddMemoryRequest::default() },
///     )
///     .await?;
/// let response = service
///     .retrieve(RetrieveRequest { user_id: "user".into(), query: "search".into() })
///     .await?;
/// assert!(response.memory_string.contains("prefer ripgrep"));
/// # Ok(())
/// # }
/// ```
///
/// Behavior worth relying on:
///
/// * `retrieve` loads the user's records, pre-ranks them, asks the model to
///   re-order the candidates, drops hits below [`TaskMemoryConfig::min_score`],
///   keeps at most [`TaskMemoryConfig::top_k`], renders `memory_string` in the
///   family's Python format, and answers `retrieved_memory` with the same
///   `{"memories": [...]}` card the deterministic service produces. An empty
///   store answers `success` with an empty string and `{"memories": []}` without
///   calling the model.
/// * `summarize` calls the model once and applies the returned operations in
///   order; the response lists the written records (not the removed ones).
/// * `add_memory` validates the request-derived record (`Error::Validation` for
///   blank content), reuses the id of a stored record with the same identity, and
///   upserts — all without a model call.
/// * `load_memories` projects the stored records back onto [`AddMemoryRequest`].
pub struct LlmTaskMemoryService {
    /// The algorithm whose prompts, rendering, and id prefix this service uses.
    algo: Algo,
    /// The knobs every service shares.
    config: TaskMemoryConfig,
    /// The injected completion backend.
    completion: Arc<dyn MemoryCompletion>,
    /// Persistence seam.
    store: Arc<dyn MemoryStore>,
}

impl fmt::Debug for LlmTaskMemoryService {
    /// Reports the algorithm, the knobs, and the store, and replaces the
    /// completion backend with a placeholder: its fields are its own business
    /// (and may hold credentials or prompts), so they are never printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmTaskMemoryService")
            .field("algo", &self.algo)
            .field("config", &self.config)
            .field("completion", &"<completion backend>")
            .field("store", &self.store)
            .finish()
    }
}

impl LlmTaskMemoryService {
    /// Builds a model-driven service over `config`, `completion`, and `store`.
    ///
    /// The algorithm comes from [`TaskMemoryConfig::algo`], so the prompts, the
    /// rendering, and the generated id prefix always agree.
    #[must_use]
    pub fn new(
        config: TaskMemoryConfig,
        completion: Arc<dyn MemoryCompletion>,
        store: Arc<dyn MemoryStore>,
    ) -> Self {
        Self {
            algo: config.algo,
            config,
            completion,
            store,
        }
    }

    /// The algorithm this service prompts and renders with.
    #[must_use]
    pub const fn algo(&self) -> Algo {
        self.algo
    }

    /// The injected completion backend.
    #[must_use]
    pub fn completion(&self) -> &Arc<dyn MemoryCompletion> {
        &self.completion
    }

    /// The candidate records and the pre-ranked hits the model may re-order.
    ///
    /// [`Algo::Ace`] keeps id order (Python's unranked playbook load) capped at
    /// [`TaskMemoryConfig::ace_bullet_limit`] with every hit scored `1.0`; every
    /// other algorithm takes the lexical ranking
    /// ([`ranking::rank_candidates`]) capped at
    /// [`TaskMemoryConfig::candidate_limit`]. The hits mirror the candidate
    /// order, so they are also the fallback when the model's reply is unusable.
    fn candidates(
        &self,
        query: &str,
        records: &[MemoryRecord],
    ) -> (Vec<MemoryRecord>, Vec<RetrievedHit>) {
        if self.algo == Algo::Ace {
            let mut ordered = records.to_vec();
            ordered.sort_by(|left, right| left.id.cmp(&right.id));
            ordered.truncate(self.config.ace_bullet_limit);
            let hits = ranking::rank_candidates(Algo::Ace, query, &ordered, ordered.len());
            return (ordered, hits);
        }
        let hits = ranking::rank_candidates(self.algo, query, records, self.config.candidate_limit);
        let candidates = hits
            .iter()
            .filter_map(|hit| records.iter().find(|record| record.id == hit.id).cloned())
            .collect();
        (candidates, hits)
    }

    /// Resolves the id a summarize `Add`/`Update` writes under.
    ///
    /// The reply's own id wins when it has one; otherwise the id of a stored
    /// record with the same `MemoryRecord::identity` is reused (so re-summarizing
    /// the same insight updates it instead of duplicating it); otherwise a fresh
    /// deterministic id is generated. `live` holds the user's records as the
    /// batch has left them, so two drafts with the same identity in one reply
    /// still collapse onto one record.
    fn resolve_id(
        &self,
        user_id: &str,
        operation: &SummaryDraft,
        live: &BTreeMap<String, MemoryRecord>,
    ) -> String {
        if let Some(id) = target_id(operation) {
            return id.to_string();
        }
        let identity = MemoryRecord::new(user_id, operation.draft.clone()).identity();
        live.values()
            .find(|record| record.identity() == identity)
            .map_or_else(
                || generate_memory_id(self.algo, user_id, &operation.draft),
                |record| record.id.clone(),
            )
    }

    /// Applies one parsed operation to the store, returning the record it wrote.
    ///
    /// `Remove` writes nothing (its return is `None`); `Add`/`Update` and `Tag`
    /// return the record they upserted. An unusable draft, a missing target id,
    /// and an unknown target id are logged at `debug` and skipped instead of
    /// failing the surrounding batch.
    async fn apply(
        &self,
        user_id: &str,
        operation: &SummaryDraft,
        live: &mut BTreeMap<String, MemoryRecord>,
    ) -> Result<Option<MemoryRecord>> {
        match operation.operation {
            SummaryOperation::Add | SummaryOperation::Update => {
                let id = self.resolve_id(user_id, operation, live);
                let record = MemoryRecord::new(user_id, operation.draft.clone()).with_id(id);
                if let Err(error) = record.validate() {
                    debug!(
                        target: LOG_TARGET,
                        algo = self.algo.as_str(),
                        kind = ?error.kind(),
                        "summarize draft is unusable; skipping it"
                    );
                    return Ok(None);
                }
                self.store.upsert(record.clone()).await?;
                live.insert(record.id.clone(), record.clone());
                Ok(Some(record))
            }
            SummaryOperation::Remove => {
                let Some(id) = target_id(operation) else {
                    debug!(
                        target: LOG_TARGET,
                        algo = self.algo.as_str(),
                        "remove operation names no memory; skipping it"
                    );
                    return Ok(None);
                };
                let removed = self.store.remove(user_id, id).await?;
                live.remove(id);
                if !removed {
                    debug!(
                        target: LOG_TARGET,
                        algo = self.algo.as_str(),
                        id,
                        "remove operation names an unknown memory; nothing to delete"
                    );
                }
                Ok(None)
            }
            SummaryOperation::Tag => {
                let Some(id) = target_id(operation) else {
                    debug!(
                        target: LOG_TARGET,
                        algo = self.algo.as_str(),
                        "tag operation names no memory; skipping it"
                    );
                    return Ok(None);
                };
                let Some(record) = live.get(id).cloned() else {
                    debug!(
                        target: LOG_TARGET,
                        algo = self.algo.as_str(),
                        id,
                        "tag operation names an unknown memory; skipping it"
                    );
                    return Ok(None);
                };
                let updated = record.with_feedback(
                    counter(&operation.draft.attributes, "helpful"),
                    counter(&operation.draft.attributes, "harmful"),
                );
                self.store.upsert(updated.clone()).await?;
                live.insert(updated.id.clone(), updated.clone());
                Ok(Some(updated))
            }
        }
    }
}

/// The trimmed target id a summarize operation names, when it names one.
fn target_id(operation: &SummaryDraft) -> Option<&str> {
    operation
        .target_id()
        .map(str::trim)
        .filter(|id| !id.is_empty())
}

/// Reads a non-negative feedback delta from `attributes`.
///
/// JSON numbers and numeric strings are both accepted (models emit either),
/// fractional, negative, and non-numeric values are ignored (they yield `0`),
/// and a value that does not fit [`u32`] saturates at [`u32::MAX`].
fn counter(attributes: &BTreeMap<String, Value>, key: &str) -> u32 {
    let raw = match attributes.get(key) {
        Some(Value::Number(number)) => number.as_u64(),
        Some(Value::String(text)) => text.trim().parse::<u64>().ok(),
        _ => None,
    };
    raw.map_or(0, |value| u32::try_from(value).unwrap_or(u32::MAX))
}

#[async_trait]
impl TaskMemoryService for LlmTaskMemoryService {
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse> {
        let records = self.store.load(&request.user_id).await?;
        if records.is_empty() {
            debug!(
                target: LOG_TARGET,
                user = %request.user_id,
                algo = self.algo.as_str(),
                "no stored memories; answering with the empty card"
            );
            return Ok(RetrieveResponse {
                status: SUCCESS.to_string(),
                memory_string: String::new(),
                retrieved_memory: json!({"memories": []}),
            });
        }
        let (candidates, fallback) = self.candidates(&request.query, &records);
        // `candidates` is never empty here: the store returned records and both
        // limits are clamped to at least `1`, so every algorithm ranks one.
        let hits = match prompt::retrieve_prompt(self.algo, &request.query, &candidates) {
            None => fallback,
            Some(pair) => {
                let reply = self.completion.complete(&pair.system, &pair.user).await?;
                match prompt::parse_retrieve(self.algo, &reply, &candidates) {
                    Ok(hits) => hits,
                    Err(error) => {
                        // Python's rerank degrades the same way; only the error
                        // kind leaves this frame, never the reply or a memory.
                        warn!(
                            target: LOG_TARGET,
                            algo = self.algo.as_str(),
                            kind = ?error.kind(),
                            "ranking reply could not be parsed; using the pre-ranked candidate order"
                        );
                        fallback
                    }
                }
            }
        };
        let hits = finalize_hits(hits, &self.config);
        let memory_string = render_memory_string(self.algo, &hits, &candidates);
        debug!(
            target: LOG_TARGET,
            user = %request.user_id,
            algo = self.algo.as_str(),
            candidates = candidates.len(),
            hits = hits.len(),
            "retrieved memories"
        );
        Ok(RetrieveResponse {
            status: SUCCESS.to_string(),
            memory_string,
            retrieved_memory: memories_json(&hits, &candidates),
        })
    }

    async fn summarize(&self, request: SummarizeRequest) -> Result<SummarizeResponse> {
        let trajectories = prompt::parse_trajectories(&request.trajectories)?;
        if trajectories.is_empty() {
            debug!(
                target: LOG_TARGET,
                user = %request.user_id,
                algo = self.algo.as_str(),
                "no trajectories; nothing to summarize"
            );
            return Ok(SummarizeResponse {
                status: SUCCESS.to_string(),
                memories: json!({"memories": []}),
            });
        }
        let records = self.store.load(&request.user_id).await?;
        let pair = prompt::summarize_prompt(self.algo, &trajectories, &records);
        let reply = self.completion.complete(&pair.system, &pair.user).await?;
        // A malformed summarize reply is fatal: Python raises
        // `TOOLCHAIN_EVOLVING_MEMORY_SUMMARIZE_EXECUTION_ERROR` here, and a
        // silent no-op would hide a pipeline that produces no memory.
        let operations = prompt::parse_summarize(self.algo, &reply)?;
        let mut live: BTreeMap<String, MemoryRecord> = records
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect();
        let mut written: Vec<MemoryRecord> = Vec::new();
        for operation in &operations {
            if let Some(record) = self.apply(&request.user_id, operation, &mut live).await? {
                written.push(record);
            }
        }
        debug!(
            target: LOG_TARGET,
            user = %request.user_id,
            algo = self.algo.as_str(),
            trajectories = trajectories.len(),
            operations = operations.len(),
            written = written.len(),
            "summarized trajectories"
        );
        // The card reuses the retrieve shape; `score` carries the record's
        // feedback score (`get_score`), the only merit figure a freshly written
        // record has.
        let hits: Vec<RetrievedHit> = written
            .iter()
            .map(|record| RetrievedHit::new(record.id.clone(), record.feedback_score()))
            .collect();
        Ok(SummarizeResponse {
            status: SUCCESS.to_string(),
            memories: memories_json(&hits, &written),
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
        // the caller's thread, exactly like the deterministic service. Neither
        // bundled store blocks on a network: the in-memory store resolves
        // immediately, and the JSON store's `tokio::fs` work runs on the
        // blocking pool, which completes independently of the waiting thread.
        let records = futures::executor::block_on(self.store.load(user_id))?;
        Ok(records.iter().map(MemoryRecord::to_request).collect())
    }
}

#[cfg(test)]
mod tests {
    // Python counterparts: `extensions/context_evolver/service/task_memory_service.py`
    // (`retrieve`, `summarize`, `add_memory`, `load_memories`), plus
    // `retrieve/task/*/run.py` (candidate ranking, ACE's unranked playbook) and
    // `summary/task/*/update.py` (the add/update/tag/remove operations).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::completion::ScriptedCompletion;
    use crate::context_evolver::service::{create_task_memory_service, MemoryBackend};
    use crate::context_evolver::{InMemoryMemoryStore, MemoryDraft};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: content.to_string(),
            ..MemoryRecord::default()
        }
    }

    async fn service_with(
        algo: Algo,
        config: TaskMemoryConfig,
        records: Vec<MemoryRecord>,
        replies: impl IntoIterator<Item = &'static str>,
    ) -> (
        LlmTaskMemoryService,
        Arc<ScriptedCompletion>,
        Arc<InMemoryMemoryStore>,
    ) {
        assert_eq!(config.algo, algo, "fixtures must agree on the algorithm");
        let store = Arc::new(InMemoryMemoryStore::new());
        for record in records {
            store.upsert(record).await.unwrap();
        }
        let completion = Arc::new(ScriptedCompletion::new(replies));
        let service = LlmTaskMemoryService::new(config, completion.clone(), store.clone());
        (service, completion, store)
    }

    fn request(user_id: &str, query: &str) -> RetrieveRequest {
        RetrieveRequest {
            user_id: user_id.to_string(),
            query: query.to_string(),
        }
    }

    fn trajectories(query: &str) -> Value {
        json!([{"query": query, "response": "did it", "feedback": "success"}])
    }

    fn retrieved_ids(response: &RetrieveResponse) -> Vec<String> {
        response.retrieved_memory["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|memory| memory["id"].as_str().unwrap().to_string())
            .collect()
    }

    fn summarized_ids(response: &SummarizeResponse) -> Vec<String> {
        response.memories["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|memory| memory["id"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn retrieve_prompts_the_model_with_the_query_and_the_candidates() {
        let (service, completion, _) = service_with(
            Algo::ReasoningBank,
            TaskMemoryConfig::new(Algo::ReasoningBank),
            vec![
                MemoryRecord {
                    title: Some("Paginate".into()),
                    description: Some("Loop until empty".into()),
                    ..record("a", "alpha beta loop")
                },
                record("b", "unrelated text"),
            ],
            [r#"{"ranked_ids": ["b", "a"]}"#],
        )
        .await;
        assert_eq!(service.algo(), Algo::ReasoningBank);
        let as_dyn: Arc<dyn MemoryCompletion> = completion.clone();
        assert!(Arc::ptr_eq(service.completion(), &as_dyn));

        let response = service
            .retrieve(request("user", "alpha beta"))
            .await
            .unwrap();
        assert_eq!(completion.call_count(), 1);
        let call = &completion.calls()[0];
        assert!(call.user.contains("alpha beta"), "{}", call.user);
        assert!(call.user.contains("alpha beta loop"), "{}", call.user);
        assert!(call.user.contains("unrelated text"), "{}", call.user);

        assert_eq!(response.status, "success");
        // The reply's order wins over the lexical pre-ranking.
        assert_eq!(retrieved_ids(&response), ["b", "a"]);
        assert!(response.memory_string.contains("Title: Paginate"));
        assert!(response.memory_string.contains("Content: unrelated text"));
        let first = &response.retrieved_memory["memories"][0];
        assert_eq!(first["id"], json!("b"));
        assert_eq!(first["content"], json!("unrelated text"));
        assert_eq!(first["section"], json!("general"));
        assert!(first["score"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn retrieve_truncates_to_top_k() {
        let (service, _, _) = service_with(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme).with_top_k(1),
            vec![record("a", "alpha beta"), record("b", "alpha beta")],
            [r#"{"ranked_indices": [1, 0]}"#],
        )
        .await;

        let response = service
            .retrieve(request("user", "alpha beta"))
            .await
            .unwrap();
        assert_eq!(retrieved_ids(&response), ["b"]);
        assert!(!response.memory_string.contains("Memory 2:"));
        assert_eq!(
            response.retrieved_memory["memories"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn retrieve_ace_injects_the_playbook_in_id_order_without_a_model_call() {
        let (service, completion, _) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            vec![
                record("c", "zeta"),
                record("a", "alpha"),
                record("b", "beta"),
            ],
            std::iter::empty(),
        )
        .await;

        let response = service.retrieve(request("user", "alpha")).await.unwrap();
        assert_eq!(completion.call_count(), 0);
        assert_eq!(retrieved_ids(&response), ["a", "b", "c"]);
        assert!(response
            .memory_string
            .contains("[a] helpful=0 harmful=0 neutral=0"));
        assert!(response
            .memory_string
            .contains("[c] helpful=0 harmful=0 neutral=0"));

        let (capped, _, _) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace).with_ace_bullet_limit(2),
            vec![
                record("c", "zeta"),
                record("a", "alpha"),
                record("b", "beta"),
            ],
            std::iter::empty(),
        )
        .await;
        let response = capped.retrieve(request("user", "alpha")).await.unwrap();
        assert_eq!(retrieved_ids(&response), ["a", "b"]);
    }

    #[tokio::test]
    async fn retrieve_falls_back_to_the_candidate_order_on_a_malformed_reply() {
        let (service, completion, _) = service_with(
            Algo::Cognition,
            TaskMemoryConfig::new(Algo::Cognition),
            vec![
                record("a", "alpha beta"),
                record("b", "alpha"),
                record("c", "gamma"),
            ],
            ["not json at all"],
        )
        .await;

        let response = service
            .retrieve(request("user", "alpha beta"))
            .await
            .unwrap();
        assert_eq!(completion.call_count(), 1);
        assert_eq!(response.status, "success");
        // Lexical pre-ranking: "alpha beta" scores 1.0, "alpha" ~0.707, "gamma" 0.
        assert_eq!(retrieved_ids(&response), ["a", "b", "c"]);
        assert!(response.memory_string.contains("Content: alpha beta"));
    }

    #[tokio::test]
    async fn retrieve_with_an_empty_store_answers_the_empty_card() {
        let (service, completion, _) = service_with(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            Vec::new(),
            std::iter::empty(),
        )
        .await;

        let response = service.retrieve(request("user", "anything")).await.unwrap();
        assert_eq!(completion.call_count(), 0);
        assert_eq!(response.status, "success");
        assert_eq!(response.memory_string, "");
        assert_eq!(response.retrieved_memory, json!({"memories": []}));
    }

    #[tokio::test]
    async fn summarize_writes_each_family_reply_shape() {
        let cases = [
            (
                Algo::Ace,
                r#"{"operations": [{"type": "ADD", "section": "general", "content": "ace insight"}]}"#,
                "ace insight",
                "general",
            ),
            (
                Algo::ReasoningBank,
                r#"[{"title": "t", "description": "d", "content": "bank insight"}]"#,
                "bank insight",
                "general",
            ),
            (
                Algo::Reme,
                r#"{"when_to_use": "when paginating", "experience": "reme insight"}"#,
                "reme insight",
                "general",
            ),
            (
                Algo::Cognition,
                r#"{"description": "d", "experience": ["cognition insight"]}"#,
                "cognition insight",
                "cognition",
            ),
        ];
        for (algo, reply, content, section) in cases {
            let (service, completion, store) =
                service_with(algo, TaskMemoryConfig::new(algo), Vec::new(), [reply]).await;
            let response = service
                .summarize(SummarizeRequest {
                    user_id: "user".into(),
                    trajectories: trajectories("count the playlists"),
                })
                .await
                .unwrap_or_else(|error| panic!("{algo:?}: {}", error.message()));

            assert_eq!(completion.call_count(), 1, "{algo:?}");
            assert!(
                completion.calls()[0].user.contains("count the playlists"),
                "{algo:?}"
            );
            assert_eq!(response.status, "success", "{algo:?}");
            let memories = response.memories["memories"].as_array().unwrap();
            assert_eq!(memories.len(), 1, "{algo:?}");
            assert_eq!(memories[0]["content"], json!(content), "{algo:?}");
            assert_eq!(memories[0]["section"], json!(section), "{algo:?}");

            let stored = store.load("user").await.unwrap();
            assert_eq!(stored.len(), 1, "{algo:?}");
            assert_eq!(stored[0].content, content, "{algo:?}");
            assert_eq!(stored[0].section, section, "{algo:?}");
            assert_eq!(
                stored[0].id,
                memories[0]["id"].as_str().unwrap(),
                "{algo:?}"
            );
        }
    }

    #[tokio::test]
    async fn summarize_reuses_the_target_id_and_the_identity() {
        let (service, _, store) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            vec![record("b-1", "old rule"), record("seed-1", "same body")],
            [r#"{"operations": [
                {"type": "UPDATE", "bullet_id": "b-1", "content": "refined rule"},
                {"type": "ADD", "content": "same body"}
            ]}"#],
        )
        .await;

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: trajectories("q"),
            })
            .await
            .unwrap();
        assert_eq!(summarized_ids(&response), ["b-1", "seed-1"]);

        let stored = store.load("user").await.unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].id, "b-1");
        assert_eq!(stored[0].content, "refined rule");
        // The id-less draft matched an existing record by identity.
        assert_eq!(stored[1].id, "seed-1");
    }

    #[tokio::test]
    async fn summarize_remove_deletes_a_seeded_record() {
        let (service, _, store) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            vec![record("b-1", "stale"), record("b-2", "keep")],
            [r#"{"operations": [{"type": "REMOVE", "bullet_id": "b-1"}]}"#],
        )
        .await;

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: trajectories("q"),
            })
            .await
            .unwrap();
        assert_eq!(response.status, "success");
        assert_eq!(summarized_ids(&response), Vec::<String>::new());
        let stored = store.load("user").await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, "b-2");
    }

    #[tokio::test]
    async fn summarize_tag_bumps_the_feedback_counters() {
        let (service, _, store) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            vec![record("b-1", "rule")],
            [r#"{"operations": [
                {"type": "TAG", "bullet_id": "b-1", "metadata": {"helpful": 2, "harmful": 1}}
            ]}"#],
        )
        .await;

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: trajectories("q"),
            })
            .await
            .unwrap();
        assert_eq!(summarized_ids(&response), ["b-1"]);
        // The card carries the feedback score `(helpful + 1) / (harmful + 1)`.
        assert_eq!(response.memories["memories"][0]["score"], json!(1.5_f32));
        let stored = store.load("user").await.unwrap();
        assert_eq!(stored[0].helpful, 2);
        assert_eq!(stored[0].harmful, 1);
    }

    #[tokio::test]
    async fn summarize_skips_unknown_ids_and_unusable_drafts() {
        let (service, _, store) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            vec![record("b-1", "rule")],
            [r#"{"operations": [
                {"type": "TAG", "bullet_id": "ghost", "metadata": {"helpful": 1}},
                {"type": "REMOVE", "bullet_id": "ghost"},
                {"type": "ADD", "content": "fresh"}
            ]}"#],
        )
        .await;

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: trajectories("q"),
            })
            .await
            .unwrap();
        assert_eq!(response.status, "success");
        let memories = response.memories["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0]["content"], json!("fresh"));

        // A blank draft is skipped by `apply` itself, never failing the batch.
        let mut live: BTreeMap<String, MemoryRecord> = store
            .load("user")
            .await
            .unwrap()
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect();
        let blank = SummaryDraft::add(MemoryDraft::new("   "));
        assert_eq!(
            service.apply("user", &blank, &mut live).await.unwrap(),
            None
        );
        assert_eq!(store.load("user").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn summarize_with_empty_trajectories_makes_no_model_call() {
        let (service, completion, store) = service_with(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            Vec::new(),
            std::iter::empty(),
        )
        .await;

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: json!([]),
            })
            .await
            .unwrap();
        assert_eq!(completion.call_count(), 0);
        assert_eq!(response.status, "success");
        assert_eq!(response.memories, json!({"memories": []}));
        assert_eq!(
            store.load("user").await.unwrap(),
            Vec::<MemoryRecord>::new()
        );
    }

    #[tokio::test]
    async fn summarize_fails_on_a_malformed_reply_naming_the_phase() {
        let (service, _, store) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            Vec::new(),
            ["this is not a delta"],
        )
        .await;

        let error = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: trajectories("q"),
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("ace summarize"),
            "{}",
            error.message()
        );
        assert_eq!(
            store.load("user").await.unwrap(),
            Vec::<MemoryRecord>::new()
        );
    }

    #[tokio::test]
    async fn summarize_with_no_operations_writes_nothing() {
        let (service, completion, store) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            Vec::new(),
            [r#"{"operations": []}"#],
        )
        .await;

        let response = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: trajectories("q"),
            })
            .await
            .unwrap();
        assert_eq!(completion.call_count(), 1);
        assert_eq!(response.status, "success");
        assert_eq!(response.memories, json!({"memories": []}));
        assert_eq!(
            store.load("user").await.unwrap(),
            Vec::<MemoryRecord>::new()
        );
    }

    #[tokio::test]
    async fn add_memory_is_idempotent_and_validates_blank_content() {
        let (service, completion, store) = service_with(
            Algo::Reme,
            TaskMemoryConfig::new(Algo::Reme),
            Vec::new(),
            std::iter::empty(),
        )
        .await;
        let memory = AddMemoryRequest {
            content: "reset the cache".into(),
            when_to_use: Some("after a schema change".into()),
            ..AddMemoryRequest::default()
        };
        service.add_memory("user", memory.clone()).await.unwrap();
        service.add_memory("user", memory).await.unwrap();
        assert_eq!(completion.call_count(), 0);

        let stored = store.load("user").await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].user_id, "user");

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
    }

    #[tokio::test]
    async fn load_memories_round_trips_through_the_store() {
        let (service, _, _) = service_with(
            Algo::ReasoningBank,
            TaskMemoryConfig::new(Algo::ReasoningBank),
            vec![record("m-1", "remember me")],
            std::iter::empty(),
        )
        .await;

        let loaded = service.load_memories("user").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "remember me");
        assert_eq!(loaded[0].query, None);
        assert_eq!(service.load_memories("nobody").unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn factory_builds_a_working_llm_service() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let completion = Arc::new(ScriptedCompletion::new([r#"{"ranked_indices": [0]}"#]));
        let service = create_task_memory_service(
            TaskMemoryConfig::new(Algo::Reme),
            MemoryBackend::Llm(completion.clone()),
            store.clone(),
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

        let response = service.retrieve(request("user", "search")).await.unwrap();
        assert_eq!(completion.call_count(), 1);
        assert_eq!(response.status, "success");
        assert!(response.memory_string.contains("prefer ripgrep"));
        assert_eq!(retrieved_ids(&response).len(), 1);
    }

    #[tokio::test]
    async fn debug_replaces_the_completion_backend_with_a_placeholder() {
        let (service, _, _) = service_with(
            Algo::Ace,
            TaskMemoryConfig::new(Algo::Ace),
            vec![record("a", "secret memory body")],
            ["secret scripted reply"],
        )
        .await;

        let debug = format!("{service:?}");
        assert!(debug.contains("<completion backend>"), "{debug}");
        assert!(!debug.contains("secret scripted reply"), "{debug}");
        assert!(debug.contains("ace"), "{debug}");
    }
}
