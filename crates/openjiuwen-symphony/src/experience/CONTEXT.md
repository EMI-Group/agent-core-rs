# openjiuwen-symphony / experience

## Intent

The experience bank: query patterns bound to the skill sets that solved them, mapping to Python `openjiuwen.symphony.experience`.
The bank records how a query was answered, identifies a duplicate observation, and ranks stored experience against a new query so the planner can reuse a proven skill composition.
Ranking is embedding-optional: a configured `Embedding` ranks by cosine similarity, and without one a deterministic lexical term-coverage score keeps the bank usable offline.
A second catalog, `ExperienceIndex`, is the persisted embedding index: it keeps one vector per item, writes Python's `index_dir` layout (a manifest, a JSONL scalar file, and a JSON vector matrix), rehydrates the vectors on open instead of re-embedding, and verifies the manifest before it trusts a row.
Dedupe is an intentional difference from Python: Python appends blindly and keys on `id`, while this crate merges on the normalized query pattern plus the sorted skill set, so repeated observations reinforce one record instead of fragmenting the catalog.
The faiss experience index is no longer deferred: only IVF/faiss *training* stays behind `Error::Unsupported` (`open_with_algorithm` with an `ivf*` name), because exact flat cosine search returns the same neighbours without a GPU stack.
The input half of Python's pipeline is implemented too: `trace/` parses agent-runtime session exports into `TraceRecord`s and caches them incrementally, `cluster.rs` groups records by skill set and query topic, and `collector.rs` distils each cluster into an item and writes it through an `ExperienceSink` into a bank or the persisted index.
Python's LLM distillation is replaced by the cluster's centroid query and a deterministic token-similarity merge, so ingestion works offline with no model configured.

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
`merge_item` (crate-private, in `mod.rs`) is the one dedupe rule shared by `ExperienceBank::add` and `ExperienceIndex::add`, and `ordinal_of` (also in `mod.rs`) parses `exp_NNNN` for both generated-id counters.

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
    pub embeddings_sha256: String,   // of vector/embeddings.json
    pub scalar_sha256: String,       // of scalar/metadata.jsonl
}
pub struct ExperienceIndex;
impl ExperienceIndex {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self;                       // in-memory, no directory
    pub async fn open(dir: impl AsRef<Path>, embedding: Arc<dyn Embedding>) -> Result<Self>;
    pub async fn open_with_algorithm(dir: impl AsRef<Path>, embedding: Arc<dyn Embedding>, algorithm: impl Into<String>) -> Result<Self>;
    pub fn embedding(&self) -> &Arc<dyn Embedding>;
    pub fn dir(&self) -> Option<&Path>;
    pub fn algorithm(&self) -> &str;                                         // "flat"
    pub fn meta(&self) -> ExperienceIndexMeta;                               // last load/persist manifest
    pub fn count(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn items(&self) -> Vec<ExperienceItem>;                              // sorted by id, with embeddings
    pub fn exist(&self, skill_ids: &[&str]) -> bool;                          // whole skill set, order-free
    pub fn generate_id(&self) -> String;                                      // `exp_0001`, zero-padded
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
`open`/`open_with_algorithm` accept `"flat"` (any case); `ivf`/`ivfpq`/any `ivf*` name is `Error::Unsupported` (`faiss IVF index training`), anything else is `Error::Validation` naming `vector_algorithm`.
`search` embeds the query and ranks with the shared `rank_by_embedding` helper (ties on id, `top_k == 0` unlimited, `min_score` after truncation), and falls back to lexical ranking for a blank query.
Persistence: `<dir>/meta.json` (the manifest), `<dir>/scalar/metadata.jsonl` (one item JSON per line, no embedding field), `<dir>/vector/embeddings.json` (`{"dimension","ids","rows"}`), each written temp-sibling + rename in the order vectors → scalar → meta; `persist()` on an in-memory index refreshes the manifest without writing files.
Load is all-or-nothing and loud: an empty/absent directory is an empty index with `next_ordinal = 1`; a partial file set is `Error::External`, and a version, algorithm, digest, count, dimension, duplicate-id, row-length, or unknown-id problem is `Error::Validation` naming the offending file. Ids resume past the highest `exp_NNNN`; stored vectors are reused, never re-embedded.

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

pub const CLUSTER_SIMILARITY_THRESHOLD: f64 = 0.6;
pub struct ClusteredQuery {
    pub pattern: String, pub members: Vec<String>, pub query_examples: Vec<String>,
    pub skill_ids: Vec<String>, pub success_count: u32,
}
pub fn cluster_traces(records: &[TraceRecord]) -> Vec<ClusteredQuery>;

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
    pub fn sink(&self) -> &Arc<dyn ExperienceSink>;
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

Stage 1 groups records by sorted, deduped skill set (a skill-less record forms its own group).
Stage 2 visits a group in trace-id order and assigns each record to the first cluster whose seed query reaches `CLUSTER_SIMILARITY_THRESHOLD` token Jaccard similarity (from `crate::search::tokenize`), else starts a new cluster — deterministic, no RNG.
The centroid pattern is the member whose token set is closest (Jaccard) to the mean set of tokens present in at least half of the members; ties keep the earliest trace id.
`members` is sorted and deduped, `query_examples` holds distinct non-blank queries in member order, `success_count` counts successful members.
Clusters include failures; the collector filters them out before clustering. Output is sorted by `(skill_ids, pattern, members)`.

### Collector rules

`add` drops traces that are not `is_experience_candidate` (not successful, no skills, blank query), clusters the rest, and builds one item per cluster with the centroid pattern, up to `QUERY_EXAMPLES_COUNT` examples, the shared skills, and the success count (at least 1).
A new item merges into the first pending item with the same skill set and a normalized-pattern token similarity of at least `PATTERN_MERGE_THRESHOLD`, using the shared `merge_item` rule (counters add, examples union); `stats.merged` counts merges.
`add` returns the number of items produced (merges included) and auto-flushes once `PENDING_FLUSH_THRESHOLD` items are buffered; `flush` writes the buffer through the sink, returns 0 when empty, and is idempotent.
A failed sink write requeues the items ahead of newer pending items, so evidence is never silently dropped.
Everything is clock-free: items keep `created_at`/`last_hit_at` at `0.0`.

## Intentional differences (persisted index)

- `vector/embeddings.json` replaces `vector/faiss_index.bin` + `vector/embeddings.npy`: an exact flat index is the matrix itself, and an offline build has neither numpy nor faiss.
- Corruption is rejected instead of restored from Python's `_backup/` snapshot; the error names the file so a caller can rebuild or delete the directory.
- The embedded text is the whole `search_text()` record, where Python's `create_item` embeds only `query_pattern`.
- Duplicates merge instead of appending, using the same rule as `ExperienceBank`.
- `meta.json` carries `embeddings_sha256` and `scalar_sha256` only; Python's `vector_sha256` has no counterpart because there is no separate index file.

## Constraints

- Offline and deterministic: no network; ordering comes from score then id, never from timestamps or hash-map iteration.
- `created_at` and `last_hit_at` are metadata only and never decide behavior.
- A store is written before the in-memory view changes, so a failed write leaves the bank unchanged.
- All persistence writes a temp sibling and renames, so a reader never observes a partial document.
- Every public item is documented; no `unsafe`, no `println!`, no panics on user input.
- `sqlite` is the only feature and it only enables `SqliteExperienceStore` via `openjiuwen-foundation/sqlite`.

## Test Strategy
- Unit tests live beside the modules (`trace/models.rs`, `trace/parser.rs`, `trace/store.rs`, `cluster.rs`, `collector.rs`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.
- `tests/experience.rs` — generated/explicit ids, dedupe merging, lexical and cosine ranking, skill filters, JSONL/SQLite persistence, id-counter continuity.
- `tests/experience_index.rs` — `ExperienceIndex` add/search/dedupe, `top_k`/threshold ordering, skill-id flattening, `exist`, digest/count/dimension corruption, empty-directory open, persist round-trip and idempotence, reopen id continuity, and the IVF/unknown-algorithm errors.
- `tests/experience_trace.rs` — two-session fixture with exact records, idempotent re-parse and append order, malformed/missing/heartbeat session tolerance, single-quoted tool payloads, `clear`, skill-set/topic clustering and centroid ties, collector drops/auto-flush/similarity merge/stats, the end-to-end `TraceStore` → `cluster_traces` → collector → `ExperienceBank::search` run, and the persisted `ExperienceIndex` sink with `HashEmbedding`.
- Everything is offline and deterministic: `tempfile` roots, explicit numeric timestamps, `HashEmbedding`, no clock-dependent assertions, no hash-map iteration order in outputs.

## Routing Table

- `./mod.rs` → module docs, re-exports, `ExperienceItem`, id constants, normalization/dedupe helpers
- `./bank.rs` → `ExperienceBank`, `AddOutcome`, generated ids, merge rules, search entry points
- `./ranking.rs` → `ExperienceHit`, `ExperienceRanking`, `SearchOptions`, lexical/cosine scorers, `cosine_similarity`
- `./store.rs` → `ExperienceStore`, memory/JSON-file/SQLite backends, `DEFAULT_KEY_PREFIX`
- `./cluster.rs` → `ClusteredQuery`, `cluster_traces`, token Jaccard, centroid selection
- `./collector.rs` → `ExperienceCollector`, `ExperienceSink`, `CollectorStats`, buffer/merge/flush rules
- `./trace/` → session parsing and the incremental trace cache
  - `./trace/models.rs` → `TraceRecord`, `TraceToolCall`, experience-candidate rule
  - `./trace/parser.rs` → metadata/history reading, event folding, skill extraction, lenient literals
  - `./trace/store.rs` → `TraceStore`, `SessionPaths`, atomic `records.jsonl`/`processed_index.json`
- `../tests/experience_trace.rs` → ingestion, clustering, and collector behavior
