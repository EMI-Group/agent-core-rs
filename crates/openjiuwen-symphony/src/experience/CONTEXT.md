# openjiuwen-symphony / experience

## Intent

The experience bank: query patterns bound to the skill sets that solved them, mapping to Python `openjiuwen.symphony.experience`.
The bank records how a query was answered, identifies a duplicate observation, and ranks stored experience against a new query so the planner can reuse a proven skill composition.
Ranking is embedding-optional: a configured `Embedding` ranks by cosine similarity, and without one a deterministic lexical term-coverage score keeps the bank usable offline.
A second catalog, `ExperienceIndex`, is the persisted embedding index: it keeps one vector per item, writes Python's `index_dir` layout (a manifest, a JSONL scalar file, and a JSON vector matrix), rehydrates the vectors on open instead of re-embedding, and verifies the manifest before it trusts a row.
Dedupe is an intentional difference from Python: Python appends blindly and keys on `id`, while this crate merges on the normalized query pattern plus the sorted skill set, so repeated observations reinforce one record instead of fragmenting the catalog.
The faiss experience index is no longer deferred: only IVF/faiss *training* stays behind `Error::Unsupported` (`open_with_algorithm` with an `ivf*` name), because exact flat cosine search returns the same neighbours without a GPU stack.

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

## Routing Table

- `./mod.rs` → module docs, re-exports, `ExperienceItem`, id constants, normalization/dedupe helpers
- `./bank.rs` → `ExperienceBank`, `AddOutcome`, generated ids, merge rules, search entry points
- `./ranking.rs` → `ExperienceHit`, `ExperienceRanking`, `SearchOptions`, lexical/cosine scorers, `cosine_similarity`
- `./store.rs` → `ExperienceStore`, memory/JSON-file/SQLite backends, `DEFAULT_KEY_PREFIX`
