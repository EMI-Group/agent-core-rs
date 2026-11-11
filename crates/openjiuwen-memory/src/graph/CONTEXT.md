# graph

## Intent

Basic entity/relation memory on `openjiuwen_foundation::store::GraphStore`: extraction, persistence, and neighbor-expanded recall.
Ports a clearly scoped subset of Python `memory/graph/` (`GraphMemory`, `extraction/`): deterministic heuristic extraction plus optional LLM-assisted extraction, node/edge persistence, and BFS neighbor expansion. Milvus-specific BFS, LLM dedupe/merge, episodes, and time-validity are deferred.

## API Surface

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Entity {
    pub id: String,
    pub name: String,
    #[serde(rename = "type", alias = "entity_type")]
    pub entity_type: String,             // "Person", "Project", "Concept", ...
    pub properties: BTreeMap<String, Value>,
    #[serde(flatten)]
    pub scope: MemoryScope,
}
impl Entity { pub fn new(scope: &MemoryScope, name: impl Into<String>, entity_type: impl Into<String>) -> Self;
              pub fn from_node(node: &GraphNode) -> Option<Self>; pub fn to_node(&self) -> GraphNode; }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Relation {
    pub id: String, pub from: String, pub to: String,   // entity names
    #[serde(rename = "type", alias = "relation_type")]
    pub relation_type: String,
    pub properties: BTreeMap<String, Value>,
    #[serde(flatten)] pub scope: MemoryScope,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExtractedEntity { pub name: String, pub entity_type: String, pub importance: Option<f32> }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExtractedRelation { pub from: String, pub to: String, pub relation_type: String }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Extraction { pub entities: Vec<ExtractedEntity>, pub relations: Vec<ExtractedRelation> }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRecallHit { #[serde(flatten)] pub entity: Entity, pub score: f32, pub depth: usize }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphMemoryStats { pub entities: usize, pub relations: usize }

#[derive(Debug, Clone, Default)]
pub struct GraphAddOptions { pub extractor: Option<Arc<dyn EntityExtractor>>, pub call: CallOptions }
#[derive(Debug, Clone, Default)]
pub struct GraphSearchOptions { pub top_k: usize, pub threshold: f32, pub max_depth: usize,
                                pub max_neighbors: usize, pub include_relations: bool, pub call: CallOptions }

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
    pub async fn entities(&self, scope: &MemoryScope, limit: usize) -> Result<Vec<Entity>>;
    pub async fn relations(&self, scope: &MemoryScope) -> Result<Vec<Relation>>;
    pub async fn entity_named(&self, scope: &MemoryScope, name: &str) -> Result<Option<Entity>>;
    pub async fn delete_entity(&self, scope: &MemoryScope, id: &str) -> Result<bool>;
    pub async fn clear(&self, scope: &MemoryScope) -> Result<usize>;
    pub async fn stats(&self, scope: &MemoryScope) -> Result<GraphMemoryStats>;
}
```

## Persistence Mapping

- Node id: `entity:{sanitized_user}:{sanitized_scope}:{normalized_name}`; repeating an add with the same name updates the node (merge properties, keep id).
- Node labels: `["Entity", entity_type]`; properties include `name`, `entity_type`, `user_id`, `scope_id`, and optional `importance` / `embedding` (array).
- Edge id: `rel:{sanitized_user}:{sanitized_scope}:{from}:{relation_type}:{to}`; `from`/`to` are node ids; `label` is the relation type; properties include `user_id`, `scope_id`.
- Relations whose endpoints are not in `extraction.entities` are skipped (no dangling edges from extraction).
- Scope filtering uses `GraphQuery { labels: ["Entity"], filter: {"user_id": ..., "scope_id": ...} }`.

## Extraction

`HeuristicExtractor` (default when no model): deterministic, no regex crate:
- Candidate entities: quoted substrings (`"..."`/`'...'`), runs of capitalized words not at sentence start, `@mentions`, `backticked` identifiers, and file/path-like tokens; drop stopwords (`The`, `A`, `An`, `This`, `It`, `I`, `We`, `You`, ...), 1-char tokens, and duplicates (case-insensitive); cap at `config.max_entities`.
- Entity type inference: `Person` for capitalized names followed by a verb-like word, `Project` for path/repo tokens, `Concept` otherwise.
- Relations when `extract_relations`: patterns `X is a Y`, `X is the Y`, `X works at Y`, `X uses Y`, `X depends on Y`, `X owns Y`, `X is named Y` (case-insensitive), producing `relation_type` = lowercased pattern verb (`is_a`, `works_at`, `uses`, `depends_on`, `owns`, `named`). Only relations whose endpoints survived entity filtering are emitted.
- Deterministic ordering: entities in first-seen order; relations in first-seen order.

`LlmExtractor`: one `Model::invoke` with a system prompt requesting JSON:
`{"entities":[{"name":"...","type":"...","importance":0.0}],"relations":[{"from":"...","to":"...","type":"..."}]}`.
Parse tolerantly (accept a fenced ```json block, an object or an array under `entities`/`relations`); retry up to `model_retries`; on exhausted retries return the heuristic result and log a warning (graceful degradation, Python parity). Cancellation wraps the call.
Default selection in `GraphMemory::add`: `LlmExtractor` when a model is attached, else `HeuristicExtractor`.

## Recall

1. Load scope entities via `query_nodes` (no limit when `top_k == 0`, else a bounded candidate set).
2. Score seeds: `similarity = embedding cosine` when both query and entity embedding exist, else 0; `keyword = keyword_score(query, name + " " + properties.content)`; `score = 0.7 * similarity + 0.3 * keyword`; keep `score >= threshold`.
3. Sort by score desc (ties by entity id), take `top_k` seeds.
4. BFS expansion for `max_depth` rounds: `store.neighbors(entity.id, Direction::Both, None)`, skip already visited, score a neighbor as `parent_score * 0.5^depth`, cap `max_neighbors` per round.
5. Return unique `GraphRecallHit`s sorted by `(score desc, depth asc, id asc)`.

## Constraints

- Errors: invalid scope/input `Error::Validation`; missing graph store `Error::Unsupported`; LLM failures fall back to heuristics (warn) but cancellation returns `Error::Cancelled`.
- Graph is disabled by default (`GraphMemoryConfig::enabled = false`); `LongTermMemory::search_all` only calls it when enabled and a graph store exists.
- No `unsafe`; async-trait extractors; graph store calls are async.
- LLM extraction must tolerate non-JSON output without panicking.

## Routing Table

- `./mod.rs` → `GraphMemory`, types, re-exports
- `./entity.rs` → `Entity`, `Relation`, node/edge conversion, id derivation
- `./extract.rs` → `EntityExtractor`, `HeuristicExtractor`, `LlmExtractor`, JSON parsing/prompt
- `./recall.rs` → seed scoring, BFS expansion, hit ordering
- `./store.rs` → graph-store CRUD helpers scoped by user/scope

## Test Strategy

- Unit tests: heuristic extraction (quotes/capitals/patterns, stopwords), LLM JSON parsing (fenced/tolerant/malformed), entity/relation id derivation, merge on repeat add, BFS scoring/ordering.
- `tests/graph.rs`: add+search with `HeuristicExtractor`, LLM path with `ScriptedModel`, neighbor expansion, scope isolation, delete/clear/stats, `LongTermMemory::search_all` merging memories and entities.
