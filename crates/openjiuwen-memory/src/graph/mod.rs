//! Entity/relation memory: extraction, persistence, and neighbor recall.
//!
//! Ports the clearly scoped subset of Python `core/memory/graph/`: deterministic
//! heuristic extraction plus optional LLM-assisted extraction, node/edge
//! persistence on [`openjiuwen_foundation::store::GraphStore`], and
//! neighbor-expanded recall.
//!
//! # Pipeline
//!
//! 1. **Extract** — [`GraphMemory::add`] turns one text into an [`Extraction`]
//!    through an [`EntityExtractor`]: the caller's extractor when
//!    [`GraphAddOptions::extractor`] is set, else [`LlmExtractor`] when a model is
//!    attached ([`GraphMemory::with_model`]), else [`HeuristicExtractor`].
//! 2. **Persist** — [`GraphMemory::add_extracted`] writes every extracted entity with
//!    [`store::upsert_entity`] (a repeat add merges into the existing node and keeps its
//!    id) and then every extracted relation with [`store::upsert_relation`]. Relations
//!    are written last, so extraction never leaves a dangling edge.
//! 3. **Search seeds** — [`GraphMemory::search`] scores the scope's entities against
//!    the query with [`recall::seed_score`] (embedding similarity plus keyword
//!    coverage), drops everything below the threshold, and keeps the top-k.
//! 4. **BFS expansion** — the surviving seeds are expanded
//!    breadth-first with [`recall::expand`], so recall also surfaces the entities the
//!    seeds are connected to, scored down by `0.5^depth`.
//!
//! # Storage mapping
//!
//! - Entities are [`openjiuwen_foundation::store::GraphNode`]s labelled
//!   `["Entity", <type>]` under the derived id `entity:{user}:{scope}:{name}`
//!   ([`store::entity_node_id`]); the `name`, `entity_type`, `user_id`, `scope_id`,
//!   `session_id`, `agent_id`, `importance`, and `embedding` properties are written by
//!   [`Entity::to_node`].
//! - Relations are persisted twice by [`store::upsert_relation`]: a mirror node
//!   (label `Relation`) that makes them enumerable, and an edge whose id is the same
//!   `rel:{user}:{scope}:{from}:{type}:{to}` string that
//!   [`store::relation_id`] derives, which makes them traversable for expansion.
//! - Scope filtering is *user plus scope* (`session_id`/`agent_id` do not narrow it), so
//!   every session and agent of one workspace shares its graph.
//!
//! # Deviations from the Python graph memory
//!
//! - One [`GraphMemory`] instance per application over an injected
//!   `Arc<dyn GraphStore>`, not a process-global singleton: tests and multi-tenant
//!   processes stay isolated.
//! - [`GraphSearchOptions::default`] uses zero knobs that fall back to the
//!   [`GraphMemoryConfig`] (depth, neighbors, minimum score) and sets
//!   [`GraphSearchOptions::include_relations`] to `true`, so a default search still
//!   expands neighbors; pass `false` for seeds only.
//! - The graph is enabled/disabled by the caller (the `LongTermMemory` facade decides
//!   whether to wire graph memory at all); [`GraphMemory`] itself never consults
//!   [`GraphMemoryConfig::enabled`].
//! - LLM entity dedupe/merge, time validity, episodes, and rerankers are deferred; this
//!   module keeps deterministic identity, merging, and scoring.
pub mod entity;
pub mod extract;
pub mod recall;
pub mod store;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::store::GraphStore;
use openjiuwen_retrieval::embedding::Embedding;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::GraphMemoryConfig;
use crate::record::{CallOptions, MemoryScope, RecallHit};

pub use entity::{Entity, Relation, ENTITY_LABEL, RELATION_LABEL};
pub use extract::{
    EntityExtractor, ExtractedEntity, ExtractedRelation, Extraction, HeuristicExtractor,
    LlmExtractor, EXTRACTION_SYSTEM_PROMPT,
};
pub use recall::GraphRecallHit;

/// Entity type used when extraction leaves it blank.
const DEFAULT_ENTITY_TYPE: &str = "Concept";
/// Entity property holding the salience in `0.0..=1.0`.
const IMPORTANCE_KEY: &str = "importance";
/// Entity property holding the stored embedding vector.
const EMBEDDING_KEY: &str = "embedding";

/// Describes why `vector` cannot index an entity, or `None` when it can.
///
/// An empty or non-finite vector carries no similarity signal, so storing it would
/// quietly downgrade the entity to keyword-only recall.
fn unusable_vector(vector: &[f32]) -> Option<&'static str> {
    if vector.is_empty() {
        Some("the vector is empty")
    } else if !vector.iter().all(|component| component.is_finite()) {
        Some("the vector contains a non-finite component")
    } else {
        None
    }
}

/// Per-call options for [`GraphMemory::add`].
///
/// `extractor` overrides the instance default (an attached model, else the heuristic
/// extractor) for this call only. Debug output reports the extractor's
/// [`EntityExtractor::name`] because the trait object has no `Debug` bound.
#[derive(Clone, Default)]
pub struct GraphAddOptions {
    /// Extractor to use instead of the instance default.
    pub extractor: Option<Arc<dyn EntityExtractor>>,
    /// Cancellation token and optional deadline for the whole call.
    pub call: CallOptions,
}

impl fmt::Debug for GraphAddOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extractor = self.extractor.as_ref().map(|extractor| extractor.name());
        f.debug_struct("GraphAddOptions")
            .field("extractor", &extractor)
            .field("call", &self.call)
            .finish()
    }
}

/// Per-call options for [`GraphMemory::search`].
///
/// The zero values defer to the instance's [`GraphMemoryConfig`]: `top_k == 0` means
/// every qualifying seed, `threshold == 0.0` means [`GraphMemoryConfig::min_score`],
/// `max_depth == 0` means [`GraphMemoryConfig::max_depth`], and `max_neighbors == 0`
/// means [`GraphMemoryConfig::max_neighbors`]. `include_relations` defaults to `true`,
/// so a default search expands the seeds to their graph neighbors; set it to `false`
/// to return the scored seeds only.
#[derive(Debug, Clone)]
pub struct GraphSearchOptions {
    /// Maximum number of scored seeds; `0` keeps every seed above the threshold.
    pub top_k: usize,
    /// Minimum seed score in `0.0..=1.0`; `0.0` uses [`GraphMemoryConfig::min_score`].
    pub threshold: f32,
    /// Expansion depth from the seeds; `0` uses [`GraphMemoryConfig::max_depth`].
    pub max_depth: usize,
    /// Neighbors expanded per seed entity; `0` uses
    /// [`GraphMemoryConfig::max_neighbors`].
    pub max_neighbors: usize,
    /// Whether the seeds are expanded to their neighbors (`true` by default).
    pub include_relations: bool,
    /// Cancellation token and optional deadline for the whole call.
    pub call: CallOptions,
}

impl Default for GraphSearchOptions {
    /// Returns the documented permissive defaults: zero knobs that defer to the graph
    /// config, and neighbor expansion on.
    fn default() -> Self {
        Self {
            top_k: 0,
            threshold: 0.0,
            max_depth: 0,
            max_neighbors: 0,
            include_relations: true,
            call: CallOptions::default(),
        }
    }
}

/// How many entities and relations one scope holds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphMemoryStats {
    /// Number of entity nodes in the scope.
    pub entities: usize,
    /// Number of relations (mirror nodes) in the scope.
    pub relations: usize,
}

/// Memories and graph entities recalled by one `LongTermMemory::search_all` call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CombinedRecall {
    /// Record hits, ordered by the record recall blend.
    pub memories: Vec<RecallHit>,
    /// Entity hits, ordered by graph recall (`score` descending, then `depth`).
    pub entities: Vec<GraphRecallHit>,
    /// Why graph recall produced no entities, when it failed rather than found none.
    ///
    /// Graph recall stays best-effort — [`CombinedRecall::memories`] is still returned —
    /// but a failure is reported here instead of masquerading as "the graph had no
    /// match". The field is omitted from JSON when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_error: Option<String>,
}

impl CombinedRecall {
    /// Returns `true` when graph recall failed and only the memory hits are complete.
    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.graph_error.is_some()
    }
}

/// Entity/relation memory over one graph store.
///
/// Build one instance per application (or per tenant) and share it behind an `Arc`:
/// all methods take `&self` and the graph store owns the synchronization. The attached
/// embedding and model are optional; without them persistence and recall stay fully
/// offline (keyword-only scoring, heuristic extraction).
pub struct GraphMemory {
    /// Backing node/edge store.
    store: Arc<dyn GraphStore>,
    /// Graph limits used for extraction and as the default search knobs.
    config: GraphMemoryConfig,
    /// Optional embedder for entity vectors and query vectors.
    embedding: Option<Arc<dyn Embedding>>,
    /// Optional model used by the default [`LlmExtractor`].
    model: Option<Arc<dyn Model>>,
}

impl GraphMemory {
    /// Builds graph memory over `store` with `config`.
    ///
    /// [`GraphMemoryConfig::enabled`] is **not** consulted: the facade decides whether
    /// graph memory is wired at all, so an instance built here always reads and writes.
    #[must_use]
    pub fn new(store: Arc<dyn GraphStore>, config: GraphMemoryConfig) -> Self {
        Self {
            store,
            config,
            embedding: None,
            model: None,
        }
    }

    /// Returns the instance with `embedding` attached.
    ///
    /// An embedder adds a vector to every persisted entity and turns query similarity
    /// into part of the seed score; without one, recall is keyword-only.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Returns the instance with `model` attached.
    ///
    /// An attached model makes [`GraphMemory::add`] default to [`LlmExtractor`]
    /// (heuristic fallback included) instead of [`HeuristicExtractor`].
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Returns the graph limits this instance was built with.
    #[must_use]
    pub fn config(&self) -> &GraphMemoryConfig {
        &self.config
    }

    /// Extracts `text`, persists the result, and returns the extraction.
    ///
    /// The extractor is [`GraphAddOptions::extractor`] when set, else [`LlmExtractor`]
    /// when a model is attached, else [`HeuristicExtractor`]. Cancellation is checked
    /// before the extraction, after it (before any write), and inside the extractor.
    /// A degraded extraction (the model path falling back to heuristics) is returned
    /// with [`Extraction::degraded`] set and logged at `warn`; the entities are still
    /// persisted, so the caller sees both the result and the quality change.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope or blank `text`,
    /// `Error::Cancelled` when the options' token has been triggered, and the
    /// extractor's own error kind (including embedding failures raised while
    /// persisting).
    pub async fn add(
        &self,
        scope: &MemoryScope,
        text: &str,
        options: &GraphAddOptions,
    ) -> Result<Extraction> {
        scope.validate()?;
        if text.trim().is_empty() {
            return Err(Error::validation(
                "text must not be blank (got an empty or whitespace-only argument)",
            ));
        }
        options.call.check()?;
        let extractor: Arc<dyn EntityExtractor> = match &options.extractor {
            Some(extractor) => extractor.clone(),
            None => match &self.model {
                Some(model) => Arc::new(LlmExtractor::new(model.clone(), &self.config)),
                None => Arc::new(HeuristicExtractor::new(&self.config)),
            },
        };
        tracing::debug!(
            target: "openjiuwen_memory::graph",
            extractor = extractor.name(),
            "extracting graph memory"
        );
        let extraction = extractor.extract(text, scope, &options.call).await?;
        if let Some(reason) = extraction.degraded_reason() {
            tracing::warn!(
                target: "openjiuwen_memory::graph",
                extractor = extractor.name(),
                reason,
                "graph extraction degraded to a fallback extractor"
            );
        }
        options.call.check()?;
        let (entities, relations) = self.add_extracted(scope, &extraction).await?;
        tracing::debug!(
            target: "openjiuwen_memory::graph",
            entities = entities.len(),
            relations = relations.len(),
            "stored extracted graph memory"
        );
        Ok(extraction)
    }

    /// Persists an already-built [`Extraction`] and returns what was stored.
    ///
    /// Entities are written first (one [`Entity::new`] per non-blank name, blank types
    /// become `Concept`, a finite `importance` is clamped to `0.0..=1.0`), then the
    /// relations whose two endpoints were among the stored entities. A relation is
    /// skipped when an endpoint is missing, both endpoints are the same entity, or its
    /// type is blank, so extraction never creates a dangling edge or a self loop.
    ///
    /// When an embedder is attached, the entity texts ([`recall::entity_text`]) are
    /// embedded in one [`Embedding::embed_documents`] call and the vectors are stored
    /// as the `embedding` property; an embedding failure aborts the call before any
    /// entity is written.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope, a provider that returns the
    /// wrong number of vectors, or a vector that is empty or non-finite (an entity is
    /// never stored without a usable embedding), and the store/embedding error kind
    /// otherwise.
    pub async fn add_extracted(
        &self,
        scope: &MemoryScope,
        extraction: &Extraction,
    ) -> Result<(Vec<Entity>, Vec<Relation>)> {
        scope.validate()?;
        let mut entities = Vec::with_capacity(extraction.entities.len());
        for candidate in &extraction.entities {
            let name = candidate.name.trim();
            if name.is_empty() {
                continue;
            }
            let entity_type = candidate.entity_type.trim();
            let entity_type = if entity_type.is_empty() {
                DEFAULT_ENTITY_TYPE
            } else {
                entity_type
            };
            let mut entity = Entity::new(scope, name, entity_type);
            if let Some(importance) = candidate.importance.filter(|value| value.is_finite()) {
                entity.properties.insert(
                    IMPORTANCE_KEY.to_owned(),
                    Value::from(importance.clamp(0.0, 1.0)),
                );
            }
            entities.push(entity);
        }
        self.embed_entities(&mut entities).await?;

        let mut stored_entities = Vec::with_capacity(entities.len());
        let mut by_name: BTreeMap<String, Entity> = BTreeMap::new();
        for entity in entities {
            let stored = store::upsert_entity(self.store.as_ref(), entity).await?;
            by_name.insert(store::normalize_name(&stored.name), stored.clone());
            stored_entities.push(stored);
        }

        let mut stored_relations = Vec::with_capacity(extraction.relations.len());
        for candidate in &extraction.relations {
            let relation_type = candidate.relation_type.trim();
            if relation_type.is_empty() {
                continue;
            }
            let from_key = store::normalize_name(candidate.from.trim());
            let to_key = store::normalize_name(candidate.to.trim());
            if from_key.is_empty() || from_key == to_key {
                continue;
            }
            let (Some(from), Some(to)) = (by_name.get(&from_key), by_name.get(&to_key)) else {
                tracing::debug!(
                    target: "openjiuwen_memory::graph",
                    from = from_key.as_str(),
                    to = to_key.as_str(),
                    "skipping a relation whose endpoints were not extracted as entities"
                );
                continue;
            };
            let relation = Relation::new(scope, &from.name, &to.name, relation_type);
            stored_relations.push(store::upsert_relation(self.store.as_ref(), relation).await?);
        }
        Ok((stored_entities, stored_relations))
    }

    /// Embeds `entities` in place when an embedder is attached.
    ///
    /// The whole batch goes through one [`Embedding::embed_documents`] call; a provider
    /// returning the wrong number of vectors is a validation error rather than a silent
    /// truncation, and so is a vector that cannot index anything (empty or holding a
    /// non-finite component). Either way the call fails before any entity is written, so
    /// no unembedded entity is ever stored behind a successful return.
    async fn embed_entities(&self, entities: &mut [Entity]) -> Result<()> {
        let Some(embedding) = &self.embedding else {
            return Ok(());
        };
        if entities.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = entities.iter().map(recall::entity_text).collect();
        let vectors = embedding.embed_documents(&texts, None).await?;
        if vectors.len() != entities.len() {
            return Err(Error::validation(format!(
                "embedding provider returned {} vectors for {} entity texts",
                vectors.len(),
                entities.len()
            )));
        }
        for (entity, vector) in entities.iter_mut().zip(vectors) {
            if let Some(reason) = unusable_vector(&vector) {
                tracing::warn!(
                    target: "openjiuwen_memory::graph",
                    entity = %entity.name,
                    reason,
                    "embedding provider returned an unusable vector; refusing to store an unembedded entity"
                );
                return Err(Error::validation(format!(
                    "embedding provider returned an unusable vector for entity `{}`: {reason}",
                    entity.name
                )));
            }
            let values = vector.into_iter().map(Value::from).collect();
            entity
                .properties
                .insert(EMBEDDING_KEY.to_owned(), Value::Array(values));
        }
        Ok(())
    }

    /// Recalls the scope's entities for `query`, expanded to their neighbors.
    ///
    /// Knob resolution: a positive `threshold` is clamped to `0.0..=1.0`, otherwise
    /// [`GraphMemoryConfig::min_score`] applies; `max_depth == 0` and
    /// `max_neighbors == 0` fall back to the graph config. `top_k == 0` keeps every
    /// seed above the threshold; a positive `top_k` also bounds the candidate scan to
    /// `top_k * 8` (at least `16`) entities so a large graph is not fully loaded.
    ///
    /// Seeds are scored with [`recall::seed_score`] (query similarity when an embedder
    /// is attached, plus keyword coverage), sorted by score, and truncated to `top_k`.
    /// With [`GraphSearchOptions::include_relations`] they are then expanded
    /// breadth-first by [`recall::expand`]; otherwise the seeds are returned as is.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope or blank `query`,
    /// `Error::Cancelled` when the options' token has been triggered (or fires during
    /// the query embedding), and the store/embedding error kind otherwise.
    pub async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &GraphSearchOptions,
    ) -> Result<Vec<GraphRecallHit>> {
        scope.validate()?;
        if query.trim().is_empty() {
            return Err(Error::validation(
                "query must not be blank (got an empty or whitespace-only argument)",
            ));
        }
        options.call.check()?;
        let threshold = if options.threshold > 0.0 {
            options.threshold.clamp(0.0, 1.0)
        } else {
            self.config.min_score
        };
        let max_depth = if options.max_depth == 0 {
            self.config.max_depth
        } else {
            options.max_depth
        };
        let max_neighbors = if options.max_neighbors == 0 {
            self.config.max_neighbors
        } else {
            options.max_neighbors
        };
        let top_k = options.top_k;
        let limit = if top_k == 0 {
            None
        } else {
            Some(top_k.saturating_mul(8).max(16))
        };
        let query_embedding = match &self.embedding {
            Some(embedding) => Some(options.call.run(embedding.embed_query(query)).await??),
            None => None,
        };

        let candidates = store::query_entities(self.store.as_ref(), scope, limit).await?;
        let mut seeds = Vec::with_capacity(candidates.len());
        for entity in candidates {
            let hit_score = recall::seed_score(query, &entity, query_embedding.as_deref());
            if !hit_score.is_finite() || hit_score < threshold {
                continue;
            }
            seeds.push(GraphRecallHit {
                entity,
                score: hit_score,
                depth: 0,
            });
        }
        recall::sort_hits(&mut seeds);
        if top_k > 0 {
            seeds.truncate(top_k);
        }
        if options.include_relations && max_depth > 0 && max_neighbors > 0 {
            return recall::expand(self.store.as_ref(), seeds, max_depth, max_neighbors).await;
        }
        Ok(seeds)
    }

    /// Returns the scope's entities sorted by node id; `limit == 0` means every entity.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope and the store's error kind on
    /// a failed query.
    pub async fn entities(&self, scope: &MemoryScope, limit: usize) -> Result<Vec<Entity>> {
        scope.validate()?;
        let limit = if limit == 0 { None } else { Some(limit) };
        store::query_entities(self.store.as_ref(), scope, limit).await
    }

    /// Returns the scope's relations sorted by mirror node id.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope and the store's error kind on
    /// a failed query.
    pub async fn relations(&self, scope: &MemoryScope) -> Result<Vec<Relation>> {
        scope.validate()?;
        store::query_relations(self.store.as_ref(), scope).await
    }

    /// Looks up the entity named `name` in `scope`.
    ///
    /// The lookup is normalized (case and whitespace folded), so `"  ALICE "` finds
    /// `"Alice"`; `Ok(None)` when the scope has no such entity.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope and the store's error kind on
    /// a failed lookup.
    pub async fn entity_named(&self, scope: &MemoryScope, name: &str) -> Result<Option<Entity>> {
        scope.validate()?;
        store::get_entity(self.store.as_ref(), scope, name).await
    }

    /// Deletes the entity node `id` and every relation that touches it.
    ///
    /// Returns `false` without mutating anything when the scope has no node under
    /// `id`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope and the store's error kind on
    /// a failed delete.
    pub async fn delete_entity(&self, scope: &MemoryScope, id: &str) -> Result<bool> {
        scope.validate()?;
        store::delete_entity(self.store.as_ref(), scope, id).await
    }

    /// Deletes every entity and relation of `scope`, returning the deleted entity count.
    ///
    /// Relations are removed before their entities, so no dangling edge survives; other
    /// scopes are untouched.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope and the store's error kind on
    /// a failed delete.
    pub async fn clear(&self, scope: &MemoryScope) -> Result<usize> {
        scope.validate()?;
        store::clear_entities(self.store.as_ref(), scope).await
    }

    /// Counts the scope's entities and relations.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid scope and the store's error kind on
    /// a failed query.
    pub async fn stats(&self, scope: &MemoryScope) -> Result<GraphMemoryStats> {
        scope.validate()?;
        Ok(GraphMemoryStats {
            entities: store::count_entities(self.store.as_ref(), scope).await?,
            relations: store::count_relations(self.store.as_ref(), scope).await?,
        })
    }
}

impl fmt::Debug for GraphMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphMemory")
            .field("config", &self.config)
            .field("embedding", &self.embedding.is_some())
            .field("model", &self.model.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::store::InMemoryGraphStore;
    use openjiuwen_retrieval::embedding::HashEmbedding;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn memory() -> GraphMemory {
        GraphMemory::new(
            Arc::new(InMemoryGraphStore::new()),
            GraphMemoryConfig::default(),
        )
    }

    fn extracted(name: &str, entity_type: &str, importance: Option<f32>) -> ExtractedEntity {
        ExtractedEntity {
            name: name.to_owned(),
            entity_type: entity_type.to_owned(),
            importance,
        }
    }

    #[test]
    fn add_options_debug_reports_the_extractor_name() {
        let config = GraphMemoryConfig::default();
        let options = GraphAddOptions {
            extractor: Some(Arc::new(HeuristicExtractor::new(&config))),
            call: CallOptions::new(),
        };
        assert!(format!("{options:?}").contains("heuristic"));
        assert!(format!("{:?}", GraphAddOptions::default()).contains("None"));
    }

    #[test]
    fn search_options_default_defers_to_the_graph_config() {
        let options = GraphSearchOptions::default();
        assert_eq!(options.top_k, 0);
        assert_eq!(options.max_depth, 0);
        assert_eq!(options.max_neighbors, 0);
        assert!((options.threshold - 0.0).abs() < f32::EPSILON);
        assert!(options.include_relations);
    }

    #[test]
    fn stats_and_combined_recall_round_trip() {
        let stats = GraphMemoryStats {
            entities: 3,
            relations: 2,
        };
        let value = serde_json::to_value(&stats).expect("stats json");
        assert_eq!(value, serde_json::json!({"entities": 3, "relations": 2}));
        let restored: GraphMemoryStats = serde_json::from_value(value).expect("stats");
        assert_eq!(restored, stats);

        let combined = CombinedRecall::default();
        let value = serde_json::to_value(&combined).expect("combined json");
        assert_eq!(value, serde_json::json!({"memories": [], "entities": []}));
        assert!(!combined.is_partial());

        // A graph failure is serialized (and therefore caller-visible) rather than
        // collapsing into an empty entity list.
        let partial = CombinedRecall {
            graph_error: Some("graph store unavailable".to_owned()),
            ..CombinedRecall::default()
        };
        let value = serde_json::to_value(&partial).expect("partial json");
        assert_eq!(
            value,
            serde_json::json!({
                "memories": [],
                "entities": [],
                "graph_error": "graph store unavailable",
            })
        );
        assert!(partial.is_partial());
        let restored: CombinedRecall = serde_json::from_value(value).expect("tolerant decode");
        assert_eq!(restored, partial);
        let legacy: CombinedRecall =
            serde_json::from_value(serde_json::json!({"memories": [], "entities": []}))
                .expect("legacy decode");
        assert_eq!(legacy.graph_error, None);
    }

    #[tokio::test]
    async fn add_rejects_blank_text_and_invalid_scopes() {
        let memory = memory();
        let error = memory
            .add(&scope(), "   ", &GraphAddOptions::default())
            .await
            .expect_err("blank text");
        assert!(matches!(error, Error::Validation(_)));

        let error = memory
            .add(
                &MemoryScope::new("alice", "bad/scope"),
                "Alice",
                &GraphAddOptions::default(),
            )
            .await
            .expect_err("invalid scope");
        assert!(matches!(error, Error::Validation(_)));
    }

    #[tokio::test]
    async fn search_rejects_blank_queries() {
        let memory = memory();
        let error = memory
            .search(&scope(), "  ", &GraphSearchOptions::default())
            .await
            .expect_err("blank query");
        assert!(matches!(error, Error::Validation(_)));
    }

    #[tokio::test]
    async fn add_extracted_defaults_blank_types_and_clamps_importance() {
        let memory = memory();
        let extraction = Extraction {
            entities: vec![
                extracted("  Alice ", "", Some(2.0)),
                extracted("   ", "Person", None),
            ],
            relations: vec![ExtractedRelation {
                from: "Alice".into(),
                to: "Acme".into(),
                relation_type: "works_at".into(),
            }],
            degraded: None,
        };
        let (entities, relations) = memory
            .add_extracted(&scope(), &extraction)
            .await
            .expect("add");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[0].entity_type, DEFAULT_ENTITY_TYPE);
        assert!((entities[0].importance().expect("importance") - 1.0).abs() < 1e-6);
        // The relation names an entity that was not extracted.
        assert_eq!(relations.len(), 0);
        assert!(entities[0].embedding().is_none());
    }

    #[tokio::test]
    async fn add_extracted_without_entities_skips_embedding() {
        let memory = memory().with_embedding(Arc::new(HashEmbedding::new(8)));
        let (entities, relations) = memory
            .add_extracted(&scope(), &Extraction::default())
            .await
            .expect("add");
        assert_eq!(entities.len(), 0);
        assert_eq!(relations.len(), 0);
        assert_eq!(memory.stats(&scope()).await.expect("stats").entities, 0);
    }

    /// An embedder that answers every text with the same vector.
    struct FixedEmbedding(Vec<f32>);

    #[async_trait::async_trait]
    impl Embedding for FixedEmbedding {
        fn model_name(&self) -> &'static str {
            "fixed"
        }

        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(self.0.clone())
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| self.0.clone()).collect())
        }
    }

    #[tokio::test]
    async fn unusable_vectors_are_validation_errors_not_silent_gaps() {
        for (vector, expected) in [
            (Vec::<f32>::new(), "the vector is empty"),
            (
                vec![0.5_f32, f32::NAN, 0.5],
                "the vector contains a non-finite component",
            ),
        ] {
            let memory = memory().with_embedding(Arc::new(FixedEmbedding(vector)));
            let extraction = Extraction {
                entities: vec![extracted("Alice", "Person", None)],
                ..Extraction::default()
            };
            let error = memory
                .add_extracted(&scope(), &extraction)
                .await
                .expect_err("an unusable vector must not be stored silently");
            match &error {
                Error::Validation(message) => {
                    assert!(
                        message.contains("Alice"),
                        "the error must name the entity: {message}"
                    );
                    assert!(
                        message.contains(expected),
                        "the error must name the defect `{expected}`: {message}"
                    );
                }
                other => panic!("expected a validation error, got {other:?}"),
            }
            // Nothing was written: no entity survives with a missing embedding.
            assert_eq!(memory.stats(&scope()).await.expect("stats").entities, 0);
            assert_eq!(
                memory.entities(&scope(), 0).await.expect("entities").len(),
                0
            );
        }
    }

    #[tokio::test]
    async fn a_usable_vector_is_still_stored_on_every_entity() {
        let memory = memory().with_embedding(Arc::new(FixedEmbedding(vec![0.25, 0.5, 0.75])));
        let extraction = Extraction {
            entities: vec![
                extracted("Alice", "Person", None),
                extracted("Acme", "Project", None),
            ],
            ..Extraction::default()
        };
        let (entities, _) = memory
            .add_extracted(&scope(), &extraction)
            .await
            .expect("add");
        assert_eq!(entities.len(), 2);
        for entity in &entities {
            let vector = entity
                .embedding()
                .expect("every entity keeps its embedding");
            assert_eq!(vector, vec![0.25, 0.5, 0.75]);
        }
    }

    #[tokio::test]
    async fn debug_reports_attached_capabilities() {
        let memory = memory()
            .with_embedding(Arc::new(HashEmbedding::new(8)))
            .with_model(Arc::new(
                openjiuwen_test_support::ScriptedModel::builder()
                    .respond_text("{}")
                    .build(),
            ));
        let rendered = format!("{memory:?}");
        assert!(rendered.contains("embedding: true"));
        assert!(rendered.contains("model: true"));
    }
}
