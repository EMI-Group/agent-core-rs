# openjiuwen-symphony / experience

## Intent

The experience layer: query patterns bound to the skill sets that solved them, mapping to Python `openjiuwen.symphony.experience`.
It records how a query was answered, identifies a duplicate observation, ranks stored experience against a new query so the planner can reuse a proven skill composition, and grades the traces it learns from.
Two catalogs share one rule set: `ExperienceBank` is the store-agnostic in-memory catalog that re-derives embeddings at search time, and `ExperienceIndex` is the persisted embedding catalog that keeps its vectors (and, for IVF, its trained coarse quantiser) next to the items and rehydrates them on open.
Ranking is embedding-optional: a configured `Embedding` ranks by cosine similarity, and without one a deterministic lexical term-coverage score keeps the bank usable offline.
The persisted index answers exact flat cosine search by default and, when its algorithm name selects IVF, probe search over a deterministic coarse quantiser trained from the stored vectors.
Factory names are parsed by one grammar, and the `pq<m>` part of a name is recorded only: no product quantisation is applied and no vector is compressed, so probe search is a recall trade-off and never an accuracy one.
Dedupe is an intentional difference from Python: Python appends blindly and keys on `id`, while this crate merges on the normalized query pattern plus the sorted skill set, so repeated observations reinforce one record instead of fragmenting the catalog.
The input half is implemented too: `trace/` parses agent-runtime session exports into `TraceRecord`s and caches them incrementally, `cluster.rs` groups records by skill set and then by a deterministic k-means over the query embeddings, and `collector.rs` distils each cluster into an item and writes it through an `ExperienceSink` into a bank or the persisted index.
Python's LLM distillation is replaced by the cluster's centroid query and a deterministic token-similarity merge, so ingestion works offline with no model configured and a caller with a model installs it on the collector.
Three support modules complete the surface: `embed.rs` is a batching, cache-bounded client over the `Embedding` seam, `retriever.rs` answers a request over one or both catalogs with skill filtering and explicit ranking provenance, and `evaluator.rs` grades traces with a deterministic rubric plus an opt-in model-judged pass.
Everything here is offline, deterministic, and free of external services: ordering comes from scores, ids and centroids, never from a clock, an RNG or a hash-map iteration order.

## API Surface

```rust
pub const ID_PREFIX: &str = "exp_";
pub const ID_DIGITS: usize = 4;
pub const DEFAULT_SUCCESS_COUNT: u32 = 1;

pub struct ExperienceItem {
    pub id: String,                                   // generated `exp_NNNN` when blank
    pub query_pattern: String, pub query_examples: Vec<String>, pub skill_ids: Vec<String>,
    pub success_count: u32,                           // default 1
    #[serde(default, skip_serializing)] pub embedding: Vec<f32>,
    pub created_at: f64, pub last_hit_at: f64,       // metadata only, never used for ordering
}
impl ExperienceItem {
    pub fn new(query_pattern: impl Into<String>) -> Self;
    pub fn with_id(self, id: impl Into<String>) -> Self;
    pub fn with_query_examples(self, examples: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_skill_ids(self, skill_ids: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_success_count(self, success_count: u32) -> Self;
    pub fn with_embedding(self, embedding: Vec<f32>) -> Self;
    pub fn with_created_at(self, created_at: f64) -> Self;
    pub fn with_last_hit_at(self, last_hit_at: f64) -> Self;
    pub fn normalized_pattern(&self) -> String;       // lowercased, whitespace-collapsed
    pub fn dedupe_key(&self) -> String;               // sha256(normalized pattern + sorted deduped skill ids)
    pub fn search_text(&self) -> String;              // pattern + examples + skill ids, one per line
    pub fn skill_set(&self) -> BTreeSet<&str>;
    pub fn has_skill(&self, skill_id: &str) -> bool;
}
```

`embedding` is skipped on serialization, so persisted items carry no vectors and a configured embedder re-derives them at search time; `ExperienceIndex` stores the vectors in a separate matrix file instead.
`merge_item` (crate-private, in `mod.rs`) is the one dedupe rule shared by `ExperienceBank::add`, `ExperienceIndex::add` and the collector, and `ordinal_of` (also in `mod.rs`) parses `exp_NNNN` for both generated-id counters.

```rust
pub struct ExperienceBank;
impl ExperienceBank {
    pub fn new() -> Self;                                                // in-memory store
    pub fn with_store(store: Arc<dyn ExperienceStore>) -> Self;          // starts empty; use `load` to hydrate
    pub async fn load(store: Arc<dyn ExperienceStore>) -> Result<Self>;
    pub async fn open_json_file(path: impl AsRef<Path>) -> Result<Self>;
    #[cfg(feature = "sqlite")] pub async fn open_sqlite(url: &str) -> Result<Self>;
    pub fn with_embedding(self, embedding: Arc<dyn Embedding>) -> Self;
    pub fn store(&self) -> &Arc<dyn ExperienceStore>;
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>>;
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn get(&self, id: &str) -> Option<ExperienceItem>;
    pub fn list(&self) -> Vec<ExperienceItem>;
    pub fn exists_for_skills(&self, skill_ids: &[String]) -> bool;
    pub fn generate_id(&self) -> String;                                 // `exp_0001`, zero-padded to ID_DIGITS (4)
    pub async fn add(&self, item: ExperienceItem) -> Result<AddOutcome>;
    pub async fn add_batch(&self, items: Vec<ExperienceItem>) -> Result<Vec<AddOutcome>>;
    pub async fn remove(&self, id: &str) -> Result<bool>;
    pub async fn clear(&self) -> Result<()>;
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_options(&self, query: &str, options: &SearchOptions) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_ranking(&self, query: &str, ranking: &ExperienceRanking) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_skill_ids(&self, query: &str, top_k: usize) -> Result<Vec<String>>;
}
pub struct AddOutcome { pub id: String, pub merged: bool }

pub struct ExperienceHit { pub score: f64, pub item: ExperienceItem }
pub struct ExperienceRanking { pub top_k: usize, pub min_score: Option<f64>, pub require_all_skills: bool }
pub struct SearchOptions { pub top_k: usize, pub min_score: Option<f64>, pub required_skills: Vec<String>, pub excluded_skills: Vec<String>, pub require_all_skills: bool }
pub const DEFAULT_TOP_K: usize = 10;
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f64;   // 0.0 on empty/mismatched/zero-norm, clamped to [-1, 1]
```

`add` is deterministic: a blank normalized `query_pattern` is `Error::Validation` naming `query_pattern`; a blank id becomes the next free `exp_NNNN`; an explicit id is kept trimmed; and the item is persisted before it becomes visible in memory.

A duplicate (same normalized pattern and same skill set, regardless of skill order) is merged instead of inserted: success counters add up, example lists union in first-seen order, the later `last_hit_at` wins, an existing embedding survives an empty one, the stored id is preserved, and `AddOutcome.merged` is `true`.

The generated-id counter continues past the highest `exp_NNNN` loaded from a store and is never reset by `clear`, so ids are not reused.

Ranking uses cosine similarity when an embedder is configured and the query is non-blank, and a deterministic lexical term-coverage score otherwise: a distinct query term found in the normalized pattern scores `1.0`, one found only in the examples or skill ids scores `0.5`, and the sum is divided by the number of distinct query terms so scores land in `[0, 1]`.

Both paths filter `excluded_skills` and `required_skills` (all of them by default, any when `require_all_skills` is false), sort by score descending with ties on id ascending, truncate to `top_k` (`0` = unlimited), and only then apply `min_score`, matching Python's `search_by_embedding` order.

```rust
#[async_trait]
pub trait ExperienceStore: Send + Sync {
    async fn list(&self) -> Result<Vec<ExperienceItem>>;    // sorted by id; malformed entries warned and skipped
    async fn get(&self, id: &str) -> Result<Option<ExperienceItem>>;
    async fn put(&self, item: &ExperienceItem) -> Result<()>;
    async fn remove(&self, id: &str) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
}
pub struct MemoryExperienceStore;
pub struct JsonFileExperienceStore;                         // ::open(path), ::path(), ::reload()
#[cfg(feature = "sqlite")] pub struct SqliteExperienceStore;  // ::connect(url), ::with_kv(kv), ::prefix()
pub const DEFAULT_KEY_PREFIX: &str = "experience/";
```

`JsonFileExperienceStore` rewrites one JSONL document through `<file>.tmp` plus `rename`, accepts JSONL or a JSON array on load, and warns-and-skips blank, malformed, or id-less entries; a missing file is an empty bank.

`SqliteExperienceStore` (feature `sqlite`) stores each item's JSON under `experience/<id>` in a foundation `SqliteKvStore`, the same key prefix layout as capability descriptors.

```rust
pub const MANIFEST_VERSION: u32 = 1;
pub const FLAT_VECTOR_ALGORITHM: &str = "flat";
pub struct ExperienceIndexMeta {
    pub version: u32, pub vector_count: usize, pub vector_algorithm: String,
    pub vector_dimension: usize,
    pub embeddings_sha256: String,          // of vector/embeddings.json
    pub scalar_sha256: String,              // of scalar/metadata.jsonl
    pub ivf_sha256: Option<String>,         // of vector/ivf.json; None for a flat index
}
pub struct ExperienceIndex;
impl ExperienceIndex {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self;                       // in-memory, flat
    pub fn new_with_algorithm(embedding: Arc<dyn Embedding>, algorithm: impl Into<String>) -> Result<Self>;
    pub async fn open(dir: impl AsRef<Path>, embedding: Arc<dyn Embedding>) -> Result<Self>;
    pub async fn open_with_algorithm(dir: impl AsRef<Path>, embedding: Arc<dyn Embedding>, algorithm: impl Into<String>) -> Result<Self>;
    pub fn embedding(&self) -> &Arc<dyn Embedding>;
    pub fn dir(&self) -> Option<&Path>;
    pub fn algorithm(&self) -> &str;                                         // canonical: "flat" or an "ivf…" name
    pub fn nlist(&self) -> usize;                                            // effective lists; 0 for flat/empty
    pub fn nprobe(&self) -> usize;                                           // 8 unless set
    pub fn set_nprobe(&self, nprobe: usize);
    pub fn with_nprobe(self, nprobe: usize) -> Self;
    pub fn meta(&self) -> ExperienceIndexMeta;                               // last load/persist manifest
    pub fn count(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn items(&self) -> Vec<ExperienceItem>;                              // sorted by id, with embeddings
    pub fn exist(&self, skill_ids: &[&str]) -> bool;                         // whole skill set, order-free
    pub fn generate_id(&self) -> String;                                     // `exp_0001`, zero-padded
    pub async fn add(&self, item: ExperienceItem) -> Result<AddOutcome>;
    pub async fn add_batch(&self, items: Vec<ExperienceItem>) -> Result<Vec<AddOutcome>>;
    pub async fn remove(&self, id: &str) -> Result<bool>;
    pub async fn persist(&self) -> Result<()>;                               // idempotent
    pub async fn search(&self, query: &str, ranking: &ExperienceRanking) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_skill_ids(&self, query: &str, ranking: &ExperienceRanking) -> Result<Vec<String>>;
}
```

`add` embeds `search_text()` with `embed_query` (a caller-supplied `embedding` is replaced: the index owns its vector space) before taking the lock, merges duplicates exactly like the bank, persists, and rolls the touched entry back if the write fails; a merge into an already-embedded duplicate skips the embedding call.
`add_batch` makes one `embed_documents` call, applies every item, persists once, and restores every touched entry on any failure, so the batch is all-or-nothing.
`search` embeds the query and ranks with the shared `rank_by_embedding` helper (ties on id, `top_k == 0` unlimited, `min_score` after truncation), and falls back to lexical ranking for a blank query.
`add`, `add_batch` and `remove` only mark the touched ids dirty and drop the frozen search quantiser, so the next IVF search retrains from the current catalog while `persist()` reuses its own cached quantiser under the documented rebuild policy.

### Vector algorithms

`new_with_algorithm` and `open_with_algorithm` parse one grammar: the name is case- and whitespace-insensitive, and `flat`, `ivf`, `ivfflat`, `ivfpq`, `ivf<nlist>` and `ivf<nlist>,pq<m>` are accepted.
The canonical form is lowercase and whitespace-free with the counts re-rendered as integers, so `" IVF 0008 , PQ 16 "` and `"ivf8,pq16"` name the same index; `ivf`, `ivfflat`, `ivfpq` and `ivf<nlist>,pq` canonicalize to `ivf`/`ivf<nlist>`.
A blank name, an unimplemented name (`hnsw`, `lsh`, …), or a zero `nlist`/`pq` count is `Error::Validation` naming `vector_algorithm`.
The `pq<m>` suffix is recorded in the canonical name and in `meta.json` only: nothing is compressed, and every probed candidate is scored against its exact stored vector.
`algorithm()` reports the canonical name; `FLAT_VECTOR_ALGORITHM` is `new`'s default.

### IVF training and search

The effective list count is the requested one (`DEFAULT_IVF_NLIST`, 64, when the name carries none) clamped to the item count, and at least one whenever the index is non-empty; an empty or flat index reports `0`.
`nlist()` reports the trained list count when a quantiser is cached or loaded, and otherwise the count the next search or `persist()` will train.
`nprobe()` defaults to 8 (`DEFAULT_IVF_NPROBE`) and lives in an atomic behind `&self`, so `set_nprobe`/`with_nprobe` tune a shared index; `0` is accepted and probes one list, and a value at or above `nlist` probes every list.
The coarse quantiser is the deterministic k-means in `index/ivf.rs`: vectors are L2-normalised, the first seed is the smallest id and each further seed is the point farthest from its nearest chosen centroid (ties by id), Lloyd runs at most `MAX_LLOYD_ITERATIONS` (25) times and stops once the largest centroid shift is at most `LLOYD_EPSILON` (`1e-6`), assignment ties keep the lowest list index, and an empty list is reseeded from the member farthest from its centroid among lists that still hold more than one (ties by id).
An IVF `search` embeds and normalises the query, scores it against every centroid, probes `clamp(nprobe, 1, nlist)` lists with ties on list index, and ranks exactly the gathered candidates with the flat helper — so the skill filters, `top_k` (`0` = unlimited), `min_score`-after-truncation and tie-on-id rules are unchanged, and `nprobe >= nlist` reproduces the flat neighbour list in the same order.
A flat index ignores `nlist` and `nprobe` entirely and scores every stored vector.
The trained `CoarseIndex` is cached behind its own lock, built lazily on the first IVF query after a mutation and reused by the persist that follows; a catalog whose vectors cannot be quantised (a dimension mismatch or a missing embedding) is `Error::Validation`.

### Persistence layout

```text
<dir>/
├── meta.json              # integrity manifest (ExperienceIndexMeta)
├── scalar/
│   └── metadata.jsonl     # one ExperienceItem JSON per line, sorted by id, no embedding field
└── vector/
    ├── embeddings.json    # {"dimension", "ids", "rows"}
    └── ivf.json           # IVF only: {"algorithm", "dimension", "nlist", "ids", "centroids", "lists"}
```

`persist()` on an in-memory index refreshes the manifest without writing files.
### Incremental encoding and the rebuild policy
`persist()` encodes only the ids that changed since the last publish: a `Mutex<EncodeCache>` holds one `CacheEntry` (the scalar JSON line plus the embedding row) per id plus a dirty `BTreeSet`, drained before the catalog is read so a mutation racing the encode re-marks itself.
The cache is seeded on `load()` from the parsed catalog, so the first mutation after a reopen encodes one item rather than the file.
`persist_counters()` reports `(items encoded, publishes completed)` — diagnostic only, never a result; `n` appends move the first counter by `n` and the second by `n`.
The CPU-bound encode and quantiser training run on `spawn_blocking`, so a publish never stalls a tokio worker; the per-index `persist_lock` serialises two `persist()` calls and each memory mutation is applied before its publish.
The cached coarse quantiser (`CachedQuantiser`: the `CoarseIndex`, the serialized `ivf.json` bytes and the trained item count) is reused unless the policy says otherwise: rebuild when no quantiser is cached, when the effective `nlist` differs from the trained one, when a known id disappeared, or when the catalog grew by `max(trained / 4, 64)` items (`IVF_REBUILD_GROWTH_DIVISOR`, `IVF_REBUILD_MIN_NEW_ITEMS`).
Otherwise every new id is assigned to its nearest existing centroid (`CoarseIndex::assign_all`) and the known lists are kept, so a bounded number of untrained adds costs an assignment instead of a full k-means, rebuilds total `O(log n)` over a growing catalog, and republishing an unchanged catalog reuses the cached bytes.
### Atomic publication
Every write is a unique temp sibling (`<path>.<pid>.<counter>.tmp`) created, written and `sync_all`ed, and the directory is fsynced after the renames, so a concurrent writer never shares a temp file and no rename is lost on a crash.
A publish writes every temp first and then renames in dependency order — vectors → scalar → coarse quantiser → manifest — with `meta.json` last as the commit point, so an interrupted publish leaves a directory whose manifest still describes the previous snapshot.
Publishes into one directory are serialised by an async mutex keyed on the canonical path (`publish_lock`), and a failed publish removes its temps; `add`/`add_batch` additionally restore the touched entries in memory.
A fault at any point of the rename sequence therefore leaves a loadable directory holding either the old or the new complete snapshot, never a mixture the loader rejects.
`meta.json` carries `MANIFEST_VERSION` (still 1) plus the digests of the three data files; `ivf_sha256` is `None` for a flat index and omitted from the document.
Load is all-or-nothing and loud: an empty/absent directory is an empty index with `next_ordinal = 1`; a partial file set is `Error::External`; and a version, algorithm-grammar, digest, count, dimension, duplicate-id, row-length, id-set or coarse-geometry problem is `Error::Validation` naming the offending file.
Opening an IVF directory as `flat` (or as a different IVF name) is a validation error naming `meta.json`, and a flat manifest that carries an IVF digest is rejected rather than repaired.
Ids resume past the highest `exp_NNNN`, stored vectors are reused instead of re-embedded, and a persisted quantiser is rehydrated instead of retrained.

### Embedding client (`embed.rs`)

```rust
pub const DEFAULT_BATCH_SIZE: usize = 32;
pub const MAX_BATCH_SIZE: usize = 256;
pub const DEFAULT_CACHE_CAPACITY: usize = 1024;
pub struct EmbeddingClientStats { pub calls: u64, pub batches: u64, pub embedded: u64, pub cache_hits: u64, pub cache_misses: u64 }
pub struct EmbeddingClient;
impl EmbeddingClient {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self;
    pub fn hash() -> Self;                                   // deterministic `HashEmbedding::default`
    pub fn offline() -> Self;                                // alias of `hash`
    pub fn with_batch_size(self, batch_size: usize) -> Self; // clamped to 1..=MAX_BATCH_SIZE
    pub fn with_cache_capacity(self, capacity: usize) -> Self; // 0 disables; shrinking drops oldest
    pub fn embedding(&self) -> &Arc<dyn Embedding>;
    pub fn model_name(&self) -> &str;
    pub fn batch_size(&self) -> usize;
    pub fn cache_len(&self) -> usize;
    pub fn cache_capacity(&self) -> usize;
    pub fn stats(&self) -> EmbeddingClientStats;
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

`embed_documents` splits the request into chunks of at most `batch_size` texts, issues exactly one provider call per chunk, preserves input order, and embeds each distinct text at most once — a repeat inside the call and a cached text both reuse their vector.
An empty request is answered with an empty list and no provider call.
Every vector is cached under `"<model_name>:<sha256(text)>"` in a bounded FIFO keyed by insertion sequence, so identical call sequences always produce identical cache contents.
The cache mutex is only held for a read or a write, never across an `.await`.
Provider errors propagate unchanged, and a provider that returns a different number of vectors than it was asked for is `Error::Retrieval`, because re-aligning rows would attach the wrong vector to a document.
`calls` counts public entry points, `batches` the provider calls they made, `embedded` the texts handed to the provider, and `cache_hits`/`cache_misses` the per-distinct-text lookups.

### Query-time retrieval (`retriever.rs`)

```rust
pub struct RetrieveRequest {
    pub query: String,
    pub top_k: usize,                        // DEFAULT_TOP_K; 0 = unlimited
    pub required_skills: Vec<String>, pub excluded_skills: Vec<String>,
    pub require_all_skills: bool,            // true
    pub min_score: Option<f64>,              // applied after truncation
}
impl RetrieveRequest {
    pub fn new(query: impl Into<String>) -> Self;
    pub fn unlimited(query: impl Into<String>) -> Self;
    pub fn with_top_k(self, top_k: usize) -> Self;
    pub fn with_required_skills(self, skills: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_excluded_skills(self, skills: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_require_all_skills(self, require_all_skills: bool) -> Self;
    pub fn with_min_score(self, min_score: f64) -> Self;
}
pub enum RetrieveSource { Bank, Index }                       // declaration order is report order
pub enum RankingProvenance { Embedding, Lexical }             // `as_str`, `is_lexical`
pub struct RetrievedExperience { pub item: ExperienceItem, pub score: f64, pub sources: Vec<RetrieveSource>, pub ranking: RankingProvenance }
pub struct RetrievalReport {
    pub query: String, pub hits: Vec<RetrievedExperience>, pub sources: Vec<RetrieveSource>,
    pub ranking: RankingProvenance, pub total_candidates: usize,
}
impl RetrievalReport { pub fn ids(&self) -> Vec<&str>; pub fn is_empty(&self) -> bool; }
pub struct ExperienceRetriever;
impl ExperienceRetriever {
    pub fn new() -> Self;                                    // no catalog; `retrieve` rejects
    pub fn for_bank(bank: Arc<ExperienceBank>) -> Self;
    pub fn for_index(index: Arc<ExperienceIndex>) -> Self;
    pub fn with_bank(self, bank: Arc<ExperienceBank>) -> Self;
    pub fn with_index(self, index: Arc<ExperienceIndex>) -> Self;
    pub fn bank(&self) -> Option<&Arc<ExperienceBank>>;
    pub fn index(&self) -> Option<&Arc<ExperienceIndex>>;
    pub fn has_source(&self) -> bool;
    pub fn sources(&self) -> Vec<RetrieveSource>;             // deterministic source order
    pub async fn retrieve(&self, request: &RetrieveRequest) -> Result<RetrievalReport>;
}
```

A request's skill filters reach both catalogs by two routes: the bank applies them inside the `SearchOptions` it receives, and the retriever applies them to the index hits itself with the same predicate, because `ExperienceIndex::search` has no skill-id filter seam.
That predicate rejects a hit listing any excluded skill first, then requires every required skill — or, with `require_all_skills` false, at least one of them — and keeps every hit when both filter lists are empty.
Skill matching is the case-sensitive exact comparison of `ExperienceItem::has_skill`, and repeated required ids do not change the verdict.
A request with any skill filter in effect queries the index with `top_k = 0` (unlimited), so filtering happens before the retriever's own `top_k` truncation and a rejected index hit never consumes the caller's budget.
A request with no skill filter keeps one bounded index request at its own `top_k`.
The index request is an `ExperienceRanking` built from that budget and `require_all_skills`.
Hits are merged on `ExperienceItem::dedupe_key`: the highest score wins, an exact tie keeps the smaller item id, the `sources` of both hits are unioned in declaration order, and the winner's provenance travels with its score.
The merged list is ordered by score descending then item id ascending, truncated to `top_k` (`0` = unlimited), and only then filtered by `min_score`.
`RankingProvenance::Embedding` is reported per hit and for the report only when every contributing catalog scored with an embedder; the index always ranks with its own embedder, so a bank-only report is `Lexical` exactly when the bank has no configured embedding.
`total_candidates` counts the hits each configured catalog returned before the retriever's client-side index filter, the merge, the truncation and `min_score`, so a request that filters every index hit away still reports the index's hit count while the bank's contribution is already filtered.
A blank `query` and a retriever with no catalog are both `Error::Validation`, naming `query` and `retriever` respectively.

### Trace grading (`evaluator.rs`)

```rust
pub const DEFAULT_MAX_STEPS: usize = 20;
pub const DEFAULT_MAX_TRACES: usize = 100;          // 0 = unlimited
pub const DEFAULT_OUTPUT_EXCERPT_CHARS: usize = 400;
pub const DEFAULT_MODEL_WEIGHT: f64 = 0.5;
pub const PASS_THRESHOLD: f64 = 0.6;
pub const REPEAT_THRESHOLD: usize = 3;
pub const FAILED_PENALTY: f64 = 0.5;
pub const EMPTY_OUTPUT_PENALTY: f64 = 0.35;
pub const TOOL_ERROR_PENALTY: f64 = 0.3;
pub const EXCESSIVE_STEPS_PENALTY: f64 = 0.15;
pub const REPEATED_TOOL_PENALTY: f64 = 0.1;
pub const REPEATED_TOOL_PENALTY_CAP: f64 = 0.2;
pub const REPEATED_QUERY_PENALTY: f64 = 0.05;
pub const ERROR_MARKERS: &[&str] = &["error", "failed", "exception", "traceback", "panic"];

pub struct EvaluationConfig {
    pub max_steps: usize, pub max_traces: usize, pub output_excerpt_chars: usize,
    pub model_judgement: bool, pub model_weight: f64,
}
pub enum QualitySignal {
    Failed { errors: Vec<String> }, Completed, EmptyOutput,
    ToolErrors { errors: usize, calls: usize },
    ExcessiveSteps { steps: usize, limit: usize },
    RepeatedToolCalls { name: String, count: usize },
    RepeatedQueryTokens { token: String, count: usize },
}
pub enum JudgementProvenance { Deterministic, Model, Fallback { reason: String } }
pub struct TraceGrade { pub trace_id: String, pub success: bool, pub score: f64, pub signals: Vec<QualitySignal>, pub judged: JudgementProvenance }
pub struct EvaluationSummary {
    pub traces: usize, pub mean_score: f64, pub success_rate: f64, pub failed: usize,
    pub signals: BTreeMap<String, usize>, pub judged_by_model: usize,
    pub provenance: JudgementProvenance,
}
pub struct TraceEvaluator;
impl TraceEvaluator {
    pub fn new() -> Self;
    pub fn with_config(config: EvaluationConfig) -> Self;
    pub fn with_judge(self, judge: LlmJudgeMetric) -> Self;
    pub fn config(&self) -> &EvaluationConfig;
    pub fn judge(&self) -> Option<&LlmJudgeMetric>;
    pub fn has_judge(&self) -> bool;
    pub fn judges(&self) -> bool;                             // toggle AND a judge
    pub fn excerpt(&self, text: &str) -> String;              // char-bounded, `...` suffix
    pub fn grade_record(&self, record: &TraceRecord) -> TraceGrade;
    pub async fn grade_traces(&self, records: &[TraceRecord]) -> Result<Vec<TraceGrade>>;
    pub fn summarise_grades(&self, grades: &[TraceGrade]) -> EvaluationSummary;
    pub async fn summarise(&self, records: &[TraceRecord]) -> Result<EvaluationSummary>;
    pub async fn grade_clusters(&self, clusters: &[ClusteredQuery]) -> Result<Vec<TraceGrade>>;
}
```

Every trace starts at `1.0` and loses documented penalties: `Failed` `-0.50`, `EmptyOutput` `-0.35`, `Completed` `0.00`, `ToolErrors` `-0.30 × clamp(errors / max(calls, 1))`, `ExcessiveSteps` `-0.15`, each `RepeatedToolCalls` `-0.10` capped at `-0.20` in total, and `RepeatedQueryTokens` `-0.05`; the result is clamped to `[0, 1]`.
`errors` counts the trace's recorded error messages plus the tool calls whose recorded result is empty or contains one of `ERROR_MARKERS`; a tool call with no recorded result is absence of evidence and is not counted.
The most repeated query token is reported with ties broken by the lexicographically smallest token, so the signal list never depends on hash order.
A grade's `success` is the pass verdict — the trace recorded success **and** the score reached `PASS_THRESHOLD` — while `score` stays the quality measure, so a technically successful trace with an empty answer is not a pass here.
`grade_traces` grades the first `max_traces` records after sorting by trace id (`0` = all), so the window and the grade order are reproducible.
`summarise_grades` guards every division, counts signal labels into a `BTreeMap`, and reports `Deterministic` when no model verdict was used, `Model` when every graded trace used one, and `Fallback` otherwise (naming the first fallback reason, or how many traces the model judged).
`grade_clusters` synthesises one trace per cluster from the pattern, the first member id, the skills and a nonzero success count; grade the underlying `TraceRecord`s when failure evidence matters.
The judged pass runs only when `model_judgement` is on and a `LlmJudgeMetric` is attached: it wraps a one-line trace summary in a synthetic `CapabilityDescriptor` plus a one-case `EvaluationInput` and delegates to `crate::llm::LlmJudgeMetric`.
Only a verdict that actually came from the model moves the score — `(1 - w) × deterministic + w × model`, clamped to `[0, 1]`, with `w = clamp(model_weight)` — and the pass verdict is recomputed from the blended score.
A judge with no model, or a judgement that fell back to the metric's static rubric, leaves the deterministic score untouched and only records the provenance, because a fallback verdict is not evidence about the trace.
With `FallbackMode::Error` a failed judgement propagates as the metric's typed error instead of degrading.

## Ingestion Surface (trace, cluster, collector)

```rust
pub struct TraceRecord {
    pub trace_id: String,      // "{session_id}/{request_id}"
    pub session_id: String, pub request_id: String,
    pub query: String,         // first non-blank user text of the segment
    pub final_output: String,  // last chat.final, else accumulated chat.delta
    pub skills: Vec<String>,   // sorted, deduped
    pub success: bool,         // no error event and no metadata failure marker
    pub duration_ms: u64,      // max duration_ms (or duration seconds) on the segment
    pub tool_calls: Vec<TraceToolCall>, pub errors: Vec<String>,
    pub timestamp: f64,        // first numeric event timestamp in epoch seconds; metadata only
}
impl TraceRecord { pub fn is_experience_candidate(&self) -> bool; }  // success && skills && query
pub struct TraceToolCall { pub name: String, pub arguments: Value, pub result: Option<String> }

pub struct SessionPaths { /* root */ }
impl SessionPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self;
    pub fn root(&self) -> &Path;
    pub fn sessions_dir(&self) -> PathBuf;          // <root>/sessions
    pub fn session_dir(&self, session_id: &str) -> PathBuf;
    pub fn trace_store_dir(&self) -> PathBuf;       // <root>/trace_store
    pub fn records_path(&self) -> PathBuf;          // <root>/trace_store/records.jsonl
    pub fn processed_index_path(&self) -> PathBuf;  // <root>/trace_store/processed_index.json
}
pub struct TraceStore;
impl TraceStore {
    pub fn new(root: impl Into<PathBuf>) -> Self;                 // no I/O
    pub async fn open(root: impl Into<PathBuf>) -> Result<Self>;  // creates trace_store/
    pub fn root(&self) -> &Path;
    pub fn paths(&self) -> &SessionPaths;
    pub fn trace_store_dir(&self) -> PathBuf;
    pub fn records_path(&self) -> PathBuf;
    pub fn processed_index_path(&self) -> PathBuf;
    pub async fn parse_all_sessions(&self) -> Result<Vec<TraceRecord>>;  // only new sessions; idempotent
    pub async fn load_all_records(&self) -> Result<Vec<TraceRecord>>;
    pub async fn clear(&self) -> Result<()>;
}
pub const SESSIONS_DIR: &str = "sessions";
pub const TRACE_STORE_DIR: &str = "trace_store";
pub const RECORDS_FILE: &str = "records.jsonl";
pub const PROCESSED_INDEX_FILE: &str = "processed_index.json";

pub struct ClusteredQuery {
    pub pattern: String, pub members: Vec<String>, pub query_examples: Vec<String>,
    pub skill_ids: Vec<String>, pub success_count: u32,
}
pub const DEFAULT_MAX_CLUSTERS: usize = 8;
pub const DEFAULT_MIN_CLUSTERS: usize = 1;
pub const DEFAULT_MAX_ITERATIONS: usize = 25;
pub const DEFAULT_EPSILON: f64 = 1e-6;
pub struct ClusterConfig {
    pub max_clusters: usize,                    // 8
    pub min_clusters: usize,                    // 1
    pub max_iterations: usize,                  // 25
    pub epsilon: f64,                           // 1e-6
    pub centroid_merge_threshold: Option<f64>,  // None: no post-pass merge
}
impl ClusterConfig { pub fn validate(&self) -> Result<()>; }
pub fn cluster_traces(records: &[TraceRecord]) -> Vec<ClusteredQuery>;
pub async fn cluster_traces_with_embedding(
    records: &[TraceRecord], embedding: &dyn Embedding, config: &ClusterConfig,
) -> Result<Vec<ClusteredQuery>>;

pub const PENDING_FLUSH_THRESHOLD: usize = 20;
pub const QUERY_EXAMPLES_COUNT: usize = 5;
pub const PATTERN_MERGE_THRESHOLD: f64 = 0.9;
#[async_trait] pub trait ExperienceSink: Send + Sync { async fn accept(&self, items: Vec<ExperienceItem>) -> Result<usize>; }
// implemented for ExperienceBank and ExperienceIndex
pub struct CollectorStats { pub accepted: usize, pub dropped: usize, pub merged: usize, pub flushed: usize }
pub struct ExperienceCollector;
impl ExperienceCollector {
    pub fn new(sink: Arc<dyn ExperienceSink>) -> Self;
    pub fn for_bank(bank: Arc<ExperienceBank>) -> Self;
    pub fn for_index(index: Arc<ExperienceIndex>) -> Self;
    pub fn with_embedding(self, embedding: Arc<dyn Embedding>) -> Self;    // default: HashEmbedding
    pub fn with_cluster_config(self, config: ClusterConfig) -> Self;       // default: ClusterConfig::default()
    pub fn sink(&self) -> &Arc<dyn ExperienceSink>;
    pub fn embedding(&self) -> &Arc<dyn Embedding>;
    pub fn cluster_config(&self) -> &ClusterConfig;
    pub async fn add(&self, records: &[TraceRecord]) -> Result<usize>;  // items produced; auto-flushes at 20
    pub async fn flush(&self) -> Result<usize>;                         // 0 when empty
    pub fn pending(&self) -> usize;
    pub fn stats(&self) -> CollectorStats;
}
```

### Parsing rules

`<root>/sessions/<session_id>/{metadata.json,history.json}` is the input; `heartbeat_*` sibling directories are skipped.
A missing session directory or metadata file yields no records; malformed metadata is warned about and the session is skipped; a missing or corrupt `history.json` is treated as an empty history (warning).
The flat event stream is folded into one record per `request_id`; a missing/blank `request_id` segment is dropped because it has no identity.
Recognized assistant events are `chat.reasoning`, `chat.delta`, `chat.final`, `chat.tool_call`, `chat.tool_result`, `chat.error`, and the same names without the `chat.` prefix; unknown types are dropped.
`query` is the first non-blank user text (Python keeps the last), `final_output` is the last `chat.final` else the concatenated deltas.
Skills come from `skill_tool.arguments.skill_name` and `symphony_compose_graph` (`arguments.candidate_skill_ids`, else `raw_output.plan.steps[].skill_id`/`name`, including a top-level `raw_output` on the tool-result event); they are sorted and deduped.
`success` is false when an error event was seen or the metadata carries `success: false`, a failed `status`/`state`, or a non-empty `error`/`error_type`/`error_detail`.
Tool payloads are decoded as JSON first, then by a small Python-literal reader (single- or double-quoted strings, `{...}`, `[...]`, numbers, `True`/`False`/`None`); an undecodable payload becomes `Value::Null` and the record survives (`ast.literal_eval` has no Rust equivalent).
A `tool_result` attaches to the most recent open tool call with the same `tool_name`; otherwise it is recorded as its own call.

### Store rules

`parse_all_sessions` filters session ids through `processed_index.json`, parses the rest, sorts them by `(timestamp, trace_id)`, skips records whose `trace_id` is already in `records.jsonl`, appends the survivors, and only then updates the index — so a crash between the two writes cannot duplicate a record.
A session that yields no records still becomes processed (Python behavior).
Both files are replaced atomically through `write_atomic` (temp sibling plus rename); existing record bytes keep their insertion order, and newly appended records are ordered by `(timestamp, trace_id)`.
`load_all_records` preserves file order and warns about/skips blank, malformed, and trace-id-less lines; a corrupt processed index degrades to an empty set with a warning.
`clear` removes both files (missing ones are fine) and keeps the `trace_store/` directory.

### Clustering rules

Stage 1 groups records by sorted, deduped skill set (a skill-less record forms its own group); groups are visited in skill-key order and their members in trace-id order (timestamp and query break exact id ties).
Stage 2 embeds every distinct non-blank query once — in trace-id order, with a single `Embedding::embed_documents` call — L2-normalises the vectors, and runs k-means inside the group: `k = ceil(sqrt(n / 2))` clamped to `ClusterConfig::min_clusters` and `min(max_clusters, n)`, and further capped by the number of members that carry a direction.
Seeding is deterministic: the first seed is the earliest member and every further seed is the point farthest (squared euclidean) from its nearest chosen seed, with ties on the earliest trace id.
Lloyd iterates at most `max_iterations` times and stops once the assignments are stable and the largest centroid shift is at most `epsilon`; an assignment tie keeps the lowest centroid index.
An empty cluster is reseeded with the point farthest from its own centroid among the clusters that still hold more than one member (ties by trace id); when no candidate is left the centroid is kept and the empty list is dropped.
A member with no direction — a blank query or a zero vector from the embedder — joins the group's first list, because its cosine similarity is `0.0` against every centroid.
When `centroid_merge_threshold` is set (a finite cosine in `[-1, 1]`), cluster pairs whose L2-normalised mean vectors reach it are merged, lowest list index absorbing the higher one, repeated until no pair qualifies, and the merged cluster's pattern is re-derived from all its members.
The centroid pattern is the member whose vector has the highest cosine similarity to the cluster's L2-normalised mean vector, with ties on trace id; a mean that cancels to zero keeps the cluster's previous centroid.
`members` is sorted and deduped, `query_examples` holds distinct non-blank queries in member order, and `success_count` counts successful members.
Clusters include failures; the collector filters them out before clustering.
Output is sorted by `(skill_ids, pattern, members)`, so two calls over the same input always agree.
`cluster_traces` is the synchronous offline path: the same k-means over `HashEmbedding::default` vectors with `ClusterConfig::default`, driven through a single-poll helper built on `Waker::noop` and `Box::pin`.
`HashEmbedding` futures are ready on their first poll, so the single poll completes; a pending poll or an embedding error — neither of which `HashEmbedding` can produce — is logged and the clusters are built over zero vectors, so the sync path never panics and never substitutes a lexical measure.
`cluster_traces_with_embedding` validates the config first and returns `Error::Validation` naming the offending field, the embedder's own error when a call fails, or `Error::Retrieval` when the embedder answers a different number of vectors than it was given.
`ClusterConfig::validate` rejects zero `max_clusters`/`min_clusters`/`max_iterations`, `min_clusters > max_clusters`, a non-finite or negative `epsilon`, and a `centroid_merge_threshold` outside `[-1, 1]`.

### Collector rules

`add` drops traces that are not `is_experience_candidate` (not successful, no skills, blank query), clusters the rest through `cluster_traces_with_embedding` with the collector's embedder and `ClusterConfig`, and builds one item per cluster with the centroid pattern, up to `QUERY_EXAMPLES_COUNT` examples, the shared skills, and the success count (at least 1).
A new item merges into the first pending item with the same skill set and a normalized-pattern token similarity of at least `PATTERN_MERGE_THRESHOLD`, using the shared `merge_item` rule (counters add, examples union); `stats.merged` counts merges.
`add` returns the number of items produced (merges included) and auto-flushes once `PENDING_FLUSH_THRESHOLD` items are buffered; `flush` writes the buffer through the sink, returns 0 when empty, and is idempotent.
A clustering failure (a model error or a rejected config) leaves the collector untouched — no counter and no buffer entry moves — so the caller can retry the same batch.
A failed sink write requeues the items ahead of newer pending items, so evidence is never silently dropped.
The clustering embedder defaults to `HashEmbedding` and the k-means tunables to `ClusterConfig::default`; `with_embedding` installs a model and `with_cluster_config` narrows the bounds.
Everything is clock-free: items keep `created_at`/`last_hit_at` at `0.0`.

## Intentional differences

- `vector/embeddings.json` replaces `vector/faiss_index.bin` + `vector/embeddings.npy`: an exact flat index is the matrix itself, and an offline build has neither numpy nor faiss.
- An IVF index adds `vector/ivf.json` and the deterministic coarse quantiser in `index/ivf.rs` in place of faiss' randomized k-means and binary inverted lists; no RNG, clock or hash-map iteration order reaches the centroids, so two builds over the same vectors produce byte-identical centroids, lists and digests.
- `pq<m>` is parsed, canonicalized and recorded in the manifest, but no product quantisation is applied and no vector is compressed, so a probe never degrades a score.
- Corruption is rejected instead of restored from Python's `_backup/` snapshot; the error names the file so a caller can rebuild or delete the directory.
- The embedded text is the whole `search_text()` record, where Python's `create_item` embeds only `query_pattern`.
- Duplicates merge instead of appending, using the same rule as `ExperienceBank`.
- `meta.json` carries `embeddings_sha256`, `scalar_sha256` and (for IVF) `ivf_sha256`; Python's `vector_sha256` has no counterpart because there is no separate faiss file, and `MANIFEST_VERSION` stays 1 so a flat manifest written before IVF support still parses.
- Clustering replaces faiss K-Means with the crate's own k-means over the `Embedding` seam, and `ClusterConfig::centroid_merge_threshold` is a Rust-only post-pass that Python does not have.
- `cluster_traces` is a synchronous convenience over `HashEmbedding` rather than a separate algorithm: the only difference from `cluster_traces_with_embedding` is the embedder, so there is no silent quality switch between the two.
- `ExperienceRetriever` generalises Python's one-method retriever: it can query a bank, the persisted index, or both, applies the request's skill filters to index hits itself because the index has no filter seam, and reports which catalogs contributed and whether the ranking was embedding-backed.
- `TraceEvaluator` grades deterministically first and treats a model verdict as an opt-in blend, where Python asks a model and marks a trace `unevaluated` without one; a fallback verdict never moves a score.

## Constraints

- Offline and deterministic: no network, no RNG, no clock-dependent ordering; ordering comes from score then id, from list index then id, or from sorted skill keys.
- `created_at` and `last_hit_at` are metadata only and never decide behavior.
- A store is written before the in-memory view changes, so a failed write leaves the bank unchanged; both catalogs roll a touched entry back when a persist fails.
- All persistence writes a temp sibling and renames, so a reader never observes a partial document; the IVF write order keeps the manifest describing a consistent snapshot.
- A lock is never held across an `.await`: catalogs snapshot under their read lock, mutations hold the write lock only for map work, and the coarse cache and the embedding cache have their own locks.
- The embedding cache is bounded and FIFO by insertion; a cache mutex poisoned by an unrelated panic is recovered rather than propagated.
- The effective `nlist` never exceeds the item count and `nprobe` is always clamped to `1..=nlist`, so a probe can never address a list that does not exist.
- The only vector data an IVF index can train on are the stored item vectors, so an item without an embedding is a validation error rather than a silently dropped list member.
- Every public item is documented; no `unsafe`, no `println!`, no panics on user input.
- `sqlite` is the only feature and it only enables `SqliteExperienceStore` via `openjiuwen-foundation/sqlite`.

## Test Strategy

- Unit tests live beside every module: `mod.rs` (item serde, dedupe keys, `ordinal_of`, `merge_item`), `bank.rs`, `ranking.rs`, `store.rs`, `cluster.rs`, `collector.rs`, `trace/{models,parser,store}.rs`, `index/{mod,algorithm,ivf,persist}.rs`, `embed.rs`, `retriever.rs` and `evaluator.rs`.
- Cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.
- `tests/experience.rs` — generated/explicit ids, dedupe merging, lexical and cosine ranking, skill filters, JSONL/SQLite persistence, id-counter continuity.
- `tests/experience_index.rs` — `ExperienceIndex` add/search/dedupe, `top_k`/threshold ordering, skill-id flattening, `exist`, digest/count/dimension corruption, empty-directory open, persist round-trip and idempotence, reopen id continuity, determinism, and the algorithm-name handling.
- `tests/experience_trace.rs` — two-session fixture with exact records, idempotent re-parse and append order, malformed/missing/heartbeat session tolerance, single-quoted tool payloads, `clear`, skill-set/topic clustering and centroid ties, collector drops/auto-flush/similarity merge/stats, the end-to-end `TraceStore` → `cluster_traces` → collector → `ExperienceBank::search` run, and the persisted `ExperienceIndex` sink with `HashEmbedding`.
- The algorithm grammar, the coarse quantiser and the persisted layout are pinned by their own module suites: canonical names and rejections, seeding/Lloyd/probe determinism and geometry validation, and the four-file write/read order with digests and loud corruption.
- The support modules pin their own rules: batching/cache/FIFO stats for the embedder, client-side index skill filtering plus multi-catalog merge order and provenance for the retriever, and every signal, penalty, window, summary and judged blend for the evaluator.
- Everything is offline and deterministic: `tempfile` roots, explicit numeric timestamps, `HashEmbedding`, no clock-dependent assertions, no hash-map iteration order in outputs.

## Routing Table

- `./mod.rs` → module docs, re-exports, `ExperienceItem`, id constants, normalization/dedupe helpers
- `./bank.rs` → `ExperienceBank`, `AddOutcome`, generated ids, merge rules, search entry points
- `./ranking.rs` → `ExperienceHit`, `ExperienceRanking`, `SearchOptions`, lexical/cosine scorers, `cosine_similarity`
- `./store.rs` → `ExperienceStore`, memory/JSON-file/SQLite backends, `DEFAULT_KEY_PREFIX`
- `./index/` → the persisted embedding catalog
  - `./index/mod.rs` → `ExperienceIndex`, `ExperienceIndexMeta`, manifest constants, the flat and IVF search paths, the coarse-quantiser cache, `nlist`/`nprobe`
  - `./index/algorithm.rs` → the vector-algorithm grammar (`AlgorithmSpec`, `normalize_algorithm`, `effective_nlist`)
  - `./index/ivf.rs` → the deterministic coarse quantiser (`CoarseIndex`, `CoarseConfig`, seeding/Lloyd/probe)
  - `./index/persist.rs` → the directory layout, digests, `vector/ivf.json`, atomic writes and load validation
- `./embed.rs` → `EmbeddingClient`, `EmbeddingClientStats`, batching, dedupe and the bounded FIFO vector cache
- `./retriever.rs` → `ExperienceRetriever`, `RetrieveRequest`, `RetrieveSource`, `RankingProvenance`, the index-side skill filter and index budget rule, merged hits and the report
- `./evaluator.rs` → `TraceEvaluator`, `EvaluationConfig`, `QualitySignal`, `TraceGrade`, `EvaluationSummary`, the rubric and the judged pass
- `./cluster.rs` → `ClusteredQuery`, `ClusterConfig`, `cluster_traces`, `cluster_traces_with_embedding`, deterministic k-means and centroid selection
- `./collector.rs` → `ExperienceCollector`, `ExperienceSink`, `CollectorStats`, embedder/config wiring, buffer/merge/flush rules
- `./trace/` → session parsing and the incremental trace cache
  - `./trace/models.rs` → `TraceRecord`, `TraceToolCall`, experience-candidate rule
  - `./trace/parser.rs` → metadata/history reading, event folding, skill extraction, lenient literals
  - `./trace/store.rs` → `TraceStore`, `SessionPaths`, atomic `records.jsonl`/`processed_index.json`
- `../tests/experience.rs`, `../tests/experience_index.rs`, `../tests/experience_trace.rs` → catalog, index and ingestion behavior
