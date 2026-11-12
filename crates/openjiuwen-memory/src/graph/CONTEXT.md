# graph

## Intent

Basic entity/relation memory on `openjiuwen_foundation::store::GraphStore`: extraction, persistence, and neighbor-expanded recall.
Ports a clearly scoped subset of Python `memory/graph/` (`GraphMemory`, `extraction/`): deterministic heuristic extraction plus optional LLM-assisted extraction, node/edge persistence, and BFS neighbor expansion. Milvus-specific BFS, LLM dedupe/merge, episodes, and time-validity are deferred.
One `GraphMemory` instance runs per application over an injected `Arc<dyn GraphStore>`; the graph is user/scope partitioned and shared across that workspace's sessions and agents.

## API Surface

```rust
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Entity {
    pub id: String,
    pub name: String,
    #[serde(rename = "type", alias = "entity_type")]
    pub entity_type: String,             // "Person", "Project", "Concept", ...
    pub properties: BTreeMap<String, Value>,
    #[serde(flatten)]
    pub scope: MemoryScope,              // user_id/scope_id (+ optional session_id/agent_id) at top level
}
impl Entity { pub fn new(scope: &MemoryScope, name: impl Into<String>, entity_type: impl Into<String>) -> Self;
              pub fn from_node(node: &GraphNode) -> Option<Self>; pub fn to_node(&self) -> GraphNode;
              pub fn importance(&self) -> Option<f32>; pub fn embedding(&self) -> Option<Vec<f32>>;
              pub fn merge(&self, incoming: &Entity) -> Entity; }

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Relation {
    pub id: String, pub from: String, pub to: String,   // entity names, not node ids
    #[serde(rename = "type", alias = "relation_type")]
    pub relation_type: String,
    pub properties: BTreeMap<String, Value>,
    #[serde(flatten)] pub scope: MemoryScope,
}
impl Relation { pub fn new(scope: &MemoryScope, from: impl Into<String>, to: impl Into<String>,
                          relation_type: impl Into<String>) -> Self;
                pub fn from_node(node: &GraphNode) -> Option<Self>;
                pub fn to_node(&self) -> GraphNode; pub fn to_edge(&self) -> GraphEdge; }

pub const ENTITY_LABEL: &str = "Entity";
pub const RELATION_LABEL: &str = "Relation";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedEntity { pub name: String, pub entity_type: String, pub importance: Option<f32> }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedRelation { pub from: String, pub to: String, pub relation_type: String }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct Extraction { pub entities: Vec<ExtractedEntity>, pub relations: Vec<ExtractedRelation> }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRecallHit { #[serde(flatten)] pub entity: Entity, pub score: f32, pub depth: usize }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphMemoryStats { pub entities: usize, pub relations: usize }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CombinedRecall { pub memories: Vec<RecallHit>, pub entities: Vec<GraphRecallHit> }  // `LongTermMemory::search_all`

#[derive(Clone, Default)]                 // custom `Debug` reports `EntityExtractor::name`
pub struct GraphAddOptions { pub extractor: Option<Arc<dyn EntityExtractor>>, pub call: CallOptions }

#[derive(Debug, Clone)]
pub struct GraphSearchOptions { pub top_k: usize, pub threshold: f32, pub max_depth: usize,
                                pub max_neighbors: usize, pub include_relations: bool, pub call: CallOptions }
impl Default for GraphSearchOptions {
    fn default() -> Self { Self { top_k: 0, threshold: 0.0, max_depth: 0, max_neighbors: 0,
                                  include_relations: true, call: CallOptions::default() } }
}

#[async_trait]
pub trait EntityExtractor: Send + Sync {
    fn name(&self) -> &str;
    async fn extract(&self, text: &str, scope: &MemoryScope, options: &CallOptions) -> Result<Extraction>;
}

pub struct HeuristicExtractor { /* config: max_entities, extract_relations */ }
impl HeuristicExtractor { pub fn new(config: &GraphMemoryConfig) -> Self; }

pub struct LlmExtractor { /* model, retries, max_entities */ }
impl LlmExtractor { pub fn new(model: Arc<dyn Model>, config: &GraphMemoryConfig) -> Self; }

pub struct GraphMemory { /* store: Arc<dyn GraphStore>, config, embedding: Option<Arc<dyn Embedding>>,
                            model: Option<Arc<dyn Model>> */ }
impl GraphMemory {
    pub fn new(store: Arc<dyn GraphStore>, config: GraphMemoryConfig) -> Self;
    pub fn with_embedding(self, embedding: Arc<dyn Embedding>) -> Self;
    pub fn with_model(self, model: Arc<dyn Model>) -> Self;
    pub fn config(&self) -> &GraphMemoryConfig;

    pub async fn add(&self, scope: &MemoryScope, text: &str, options: &GraphAddOptions) -> Result<Extraction>;
    pub async fn add_extracted(&self, scope: &MemoryScope, extraction: &Extraction) -> Result<(Vec<Entity>, Vec<Relation>)>;
    pub async fn search(&self, scope: &MemoryScope, query: &str, options: &GraphSearchOptions) -> Result<Vec<GraphRecallHit>>;
    pub async fn entities(&self, scope: &MemoryScope, limit: usize) -> Result<Vec<Entity>>;   // limit 0 = all
    pub async fn relations(&self, scope: &MemoryScope) -> Result<Vec<Relation>>;
    pub async fn entity_named(&self, scope: &MemoryScope, name: &str) -> Result<Option<Entity>>;
    pub async fn delete_entity(&self, scope: &MemoryScope, id: &str) -> Result<bool>;
    pub async fn clear(&self, scope: &MemoryScope) -> Result<usize>;                          // returns deleted entities
    pub async fn stats(&self, scope: &MemoryScope) -> Result<GraphMemoryStats>;
}
```

`GraphMemory` itself never consults `GraphMemoryConfig::enabled`; the `LongTermMemory` facade decides whether to wire graph memory.

## Persistence Mapping

- Entity node id: `entity:{sanitized_user}:{sanitized_scope}:{normalized_name}` from `store::entity_node_id` (name lowercased, whitespace collapsed, spaces to `-`); repeating an add with the same name updates the node (merge properties, keep id).
- Entity node labels: `["Entity", entity_type]` (type label omitted when empty); properties include `name`, `entity_type`, `user_id`, `scope_id`, `session_id`/`agent_id` when the scope has them, plus the entity's own properties (`importance`, `embedding`, `content`, ...).
- Relation id: `rel:{sanitized_user}:{sanitized_scope}:{from_node}:{relation_type}:{to_node}` from `store::relation_id`; endpoints are the entity node ids from `entity_node_id`, and the type is normalized with `_` between its words (`"Works At"` becomes `works_at`). The mirror node and the edge share this id.
- Relation mirror node (`Relation::to_node`): a `GraphNode` labelled `["Relation", relation_type]` whose properties carry the entity names `from`/`to`, the derived `from_node`/`to_node` entity ids, `relation_type`, and the scope properties (`user_id`, `scope_id`, `session_id`/`agent_id` when present).
- Relation edge (`Relation::to_edge`): endpoints are the entity node ids from `entity_node_id`, `label` is the relation type, and properties carry the entity names `from`/`to` plus `relation_type` and the scope properties.
- Why twice: a `GraphStore` cannot query edges, so the mirror node makes relations enumerable via `query_relations` (and deletable/statable), while the edge makes them traversable for BFS expansion in `GraphStore::neighbors`.
- `store::upsert_relation` writes the mirror node first and the edge second, so a failed edge write leaves a recoverable, enumerable relation rather than an untraversable edge.
- Relations whose endpoints are not in `extraction.entities` are skipped, as are blank relation types and self loops (no dangling edges or self loops from extraction).
- Scope filtering uses `GraphQuery { labels: [...], filter: {"user_id": ..., "scope_id": ...} }` from `store::scope_query`; `session_id`/`agent_id` deliberately do not narrow a graph, so every session and agent of one workspace shares its graph.
- Nodes that do not convert to an `Entity`/`Relation` (relation mirror nodes seen by `query_entities`, foreign nodes in the same scope) are skipped by `filter_map`, never treated as errors.

## Stats and Ordering

- `GraphMemoryStats.entities` counts entity nodes; `GraphMemoryStats.relations` counts relation **mirror nodes** (not edges), via `count_relations` → `query_relations`.
- `entities` returns entities sorted by node id with `limit` applied after the store's id sort; `limit == 0` means every entity.
- `relations` returns relations sorted by mirror node id.
- `entity_named` looks up the derived node id with a normalized (case/whitespace folded) name and returns `Ok(None)` when no entity exists there.

## Extraction

`HeuristicExtractor` (default when no model): deterministic, no regex crate:
- Candidate entities: quoted substrings (`"..."`/`'...'`), runs of capitalized words not at sentence start, `@mentions`, `backticked` identifiers, and file/path-like tokens; drop stopwords (`The`, `A`, `An`, `This`, `It`, `I`, `We`, `You`, ...), 1-char tokens, and duplicates (case-insensitive); cap at `config.max_entities`.
- Entity type inference: `Person` for capitalized names followed by a verb-like word, `Project` for path/repo tokens, `Concept` otherwise.
- Relations when `extract_relations`: patterns `X is a Y`, `X is the Y`, `X works at Y`, `X uses Y`, `X depends on Y`, `X owns Y`, `X is named Y` (case-insensitive), producing `relation_type` = lowercased pattern verb (`is_a`, `works_at`, `uses`, `depends_on`, `owns`, `named`). Only relations whose endpoints survived entity filtering are emitted.
- Deterministic ordering: entities in first-seen order; relations in first-seen order.

`LlmExtractor`: one `Model::invoke` with a system prompt requesting JSON:
`{"entities":[{"name":"...","type":"...","importance":0.0}],"relations":[{"from":"...","to":"...","type":"..."}]}`.
Parse tolerantly (accept a fenced ```json block, an object or an array under `entities`/`relations`); retry up to `model_retries`; on exhausted retries return the heuristic result and log a warning (graceful degradation, Python parity). Cancellation wraps the call.
Default selection in `GraphMemory::add`: `GraphAddOptions::extractor` when set, else `LlmExtractor` when a model is attached, else `HeuristicExtractor`.

## Recall

1. Load scope entities via `query_entities` (no limit when `top_k == 0`, else a candidate scan bounded to `top_k * 8`, at least `16`).
2. Score seeds: `similarity = normalized cosine` when both query and entity embeddings exist, else 0; `keyword = keyword_score(query, entity_text)` over name + `content` + `description`; `score = 0.7 * similarity + 0.3 * keyword` clamped to `[0, 1]`; keep `score >= threshold`.
3. Sort by score desc (ties by entity id), then `truncate(top_k)` when `top_k > 0`.
4. BFS expansion for `max_depth` rounds: `store.neighbors(entity.id, Direction::Both, None)`, skip already visited, score a neighbor as `parent_score * 0.5^depth`, cap `max_neighbors` per parent per round.
5. Return unique `GraphRecallHit`s sorted by `(score desc, depth asc, id asc)`.

Knob resolution in `search`: a positive `threshold` is clamped to `0.0..=1.0`, otherwise `GraphMemoryConfig::min_score` (default `0.3`) applies; `max_depth == 0` uses `GraphMemoryConfig::max_depth` (default `1`); `max_neighbors == 0` uses `GraphMemoryConfig::max_neighbors` (default `8`); `top_k == 0` keeps every seed above the threshold.
`GraphSearchOptions::include_relations` defaults to `true`, so a default search expands neighbors; `false` returns the scored seeds only.
Expansion is gated by `include_relations && max_depth > 0 && max_neighbors > 0`, so after config resolution a positive depth and neighbor budget are required even when relations are included.

## Deletion

- `delete_entity` returns `false` without mutating anything when no node exists under `id`.
- Otherwise it deletes the entity node, scans the scope's relations, and deletes the incident mirror nodes and edges explicitly.
- `clear` deletes relation edges first, then the mirror nodes, then the entity nodes, so no dangling edge survives; it returns the number of entity nodes removed and leaves other scopes untouched.

## Constraints

- Errors: invalid scope or blank text/query `Error::Validation`; store/embedding failures keep their kind; LLM extraction failures fall back to heuristics (warn) but cancellation returns `Error::Cancelled`.
- Graph is disabled by default (`GraphMemoryConfig::enabled = false`); `LongTermMemory::search_all` only calls it when enabled and a graph store exists.
- No `unsafe`; async-trait extractors; graph store calls are async.
- LLM extraction must tolerate non-JSON output without panicking.

## Routing Table

- `./mod.rs` → `GraphMemory`, `GraphAddOptions`/`GraphSearchOptions`, `GraphMemoryStats`, `CombinedRecall`, re-exports
- `./entity.rs` → `Entity`, `Relation`, labels, node/edge conversion, merge
- `./extract/` → extraction seam: extraction data types, `EntityExtractor` trait, `HeuristicExtractor`, `LlmExtractor`, prompt and tolerant JSON parsing (`extract/CONTEXT.md`)
- `./recall.rs` → seed scoring, BFS expansion, hit ordering
- `./store.rs` → id derivation, `scope_query`, graph-store CRUD/stats helpers scoped by user/scope

## Test Strategy

- Unit tests: heuristic extraction (quotes/capitals/patterns, stopwords), LLM JSON parsing (fenced/tolerant/malformed), entity/relation id derivation, merge on repeat add, BFS scoring/ordering, `scope_query` filtering, delete/clear.
- `tests/graph_memory.rs`: `GraphMemory` directly — extraction wiring, persistence, seed scoring with and without an embedding, neighbor expansion on/off, scope isolation, delete/clear/stats, validation and cancellation.
- `tests/graph.rs`: through the `LongTermMemory` facade — `ltm.graph()` built only for enabled config plus a graph backend, `search_all` merging memory hits and graph entities, and facade-level persistence/recall.
