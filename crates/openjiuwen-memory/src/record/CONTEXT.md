# record

## Intent

The crate's data model and call-shape types: memory kinds, scopes, records, write inputs, filters, per-call options, and result structs.
Everything here is plain serde data; no I/O, no async, no storage.
JSON field names must stay compatible with the Python `MemoryDoc`/`MemResult` wire shapes.

## API Surface

`mod.rs` re-exports the submodules below.

### `scope.rs`
```rust
pub const DEFAULT_ID: &str = "__default__";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct MemoryScope {
    pub user_id: String,                 // required on decode; `Default`/`new` use DEFAULT_ID
    pub scope_id: String,                // required on decode; `Default`/`new` use DEFAULT_ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}
impl MemoryScope {
    pub fn new(user_id: impl Into<String>, scope_id: impl Into<String>) -> Self;
    pub fn user(user_id: impl Into<String>) -> Self;              // scope_id = DEFAULT_ID
    pub fn session(self, session_id: impl Into<String>) -> Self;  // builder
    pub fn agent(self, agent_id: impl Into<String>) -> Self;      // builder
    pub fn validate(&self) -> Result<()>;
    pub fn matches(&self, other: &Self) -> bool; // user+scope always; session/agent only when set on `self`
}
```
`user_id` and `scope_id` carry no serde default: a decoded payload that omits either fails with a `serde` "missing field" error instead of silently re-tenanting the record to `__default__`, while `Default`, `new` and `user` keep supplying `DEFAULT_ID` in code.
`validate`: user_id and scope_id non-empty, no NUL; scope_id additionally has no `/`, no `\\`, length ≤ 128 (Python `_validate_id` parity). Errors are `Error::Validation` naming the field. `session_id`/`agent_id`, when present, must be non-empty and contain no NUL; `/` is rejected for session ids used in KV keys.

### `kind.rs`
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType { UserProfile, SemanticMemory, EpisodicMemory, Variable, Summary, Coding, #[default] Unknown }
impl MemoryType {
    pub const ALL: [MemoryType; 7];
    pub const SEARCHABLE: [MemoryType; 5]; // UserProfile, SemanticMemory, EpisodicMemory, Summary, Coding
    pub const fn as_str(self) -> &'static str;   // Python value: "user_profile", ...
    pub const fn is_searchable(self) -> bool;
}
impl std::str::FromStr for MemoryType { type Err = Error; }
```
Serde values are exactly the Python enum values; accept unknown values as `Unknown` (implement `Deserialize` via a tolerant visitor or `#[serde(other)]` on `Unknown`). `Variable` and `Unknown` are never searchable.

### `record.rs`
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    #[serde(rename = "text", alias = "content", alias = "mem")]
    pub content: String,
    #[serde(rename = "type", alias = "kind", alias = "mem_type")]
    pub kind: MemoryType,
    #[serde(default = "default_importance")]
    pub importance: f32,                 // 0.5
    #[serde(default = "unix_epoch")]
    pub timestamp: DateTime<Utc>,        // decoded default: UNIX epoch
    #[serde(default = "unix_epoch")]
    pub updated_at: DateTime<Utc>,       // decoded default: UNIX epoch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_access: Option<DateTime<Utc>>,
    #[serde(default)]
    pub access_count: u64,
    #[serde(rename = "fields", alias = "metadata", default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(flatten)]
    pub scope: MemoryScope,
}
impl MemoryRecord {
    pub fn new(scope: MemoryScope, content: impl Into<String>, kind: MemoryType) -> Self;
    pub fn with_id(self, id: impl Into<String>) -> Self;
    pub fn with_importance(self, importance: f32) -> Self;
    pub fn with_metadata(self, key: impl Into<String>, value: Value) -> Self;
    pub fn with_source(self, source: impl Into<String>) -> Self;
    pub fn with_timestamp(self, timestamp: DateTime<Utc>) -> Self;
    pub fn with_embedding(self, embedding: Vec<f32>) -> Self;
    pub fn validate(&self) -> Result<()>; // non-empty id/content, 0..=1 importance, scope.validate
    pub fn metadata_str(&self, key: &str) -> Option<&str>;
    pub fn touch(&mut self, now: DateTime<Utc>); // last_access = now, access_count += 1, updated_at = now
}
```
`id.rs`: `pub fn new_memory_id(user_id: &str, now: DateTime<Utc>) -> String` producing 24 lowercase hex chars, mirroring Python: 6-byte big-endian millisecond timestamp + 3 random bytes (`Uuid::new_v4`) + 3-byte FNV-1a hash of `user_id`. Also `pub fn normalize_text(text: &str) -> String` (lowercase, collapse whitespace, strip surrounding punctuation) used by dedupe.
An absent `timestamp`/`updated_at` decodes to the UNIX epoch, never to "now": re-dating an undated row would fabricate a perfect recency score, and `MemoryRecord::new` is the only constructor that stamps the clock.

### `input.rs`
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NewMemory {
    pub content: String,
    #[serde(alias = "mem_type")]
    pub kind: MemoryType,
    pub importance: Option<f32>,
    pub metadata: BTreeMap<String, Value>,
    pub source: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
}
impl NewMemory { pub fn new(content, kind) -> Self; pub fn semantic/episodic/user_profile/summary/coding(content) -> Self;
                pub fn with_importance(self, f32) -> Self; pub fn with_metadata(self, key, Value) -> Self;
                pub fn with_source(self, impl Into<String>) -> Self; pub fn with_timestamp(self, DateTime<Utc>) -> Self;
                pub fn validate(&self) -> Result<()>; }   // Non-empty content; kind neither Variable nor Unknown

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryUpdate {
    pub content: Option<String>,
    pub kind: Option<MemoryType>,
    pub importance: Option<f32>,
    pub metadata: BTreeMap<String, Value>,  // merge-by-key, incoming wins
    pub source: Option<String>,
}
impl MemoryUpdate { pub fn content(impl Into<String>) -> Self; pub fn with_importance; pub fn with_metadata(key, Value); pub fn with_source; pub fn with_kind; pub fn is_empty(&self) -> bool; }
```
`MemoryUpdate::content` is a constructor returning `Self`; `with_*` are builders. `validate` rejects empty content, kind `Variable` or `Unknown`, importance outside `0..=1`.
`Variable` is the projection-only namespace and `Unknown` is what a decode produces for an absent or unrecognized kind, so neither is writable through `add`/`update`; both are `Error::Validation` naming the field and value.

### `filter.rs`
```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryFilter {
    pub user_id: Option<String>,
    pub scope_id: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub kinds: Vec<MemoryType>,          // empty = every kind
    pub ids: Vec<String>,                // empty = every id
    pub text_contains: Option<String>,   // case-insensitive substring on content
    pub min_importance: Option<f32>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
}
impl MemoryFilter { pub fn new() -> Self; pub fn for_scope(scope: &MemoryScope) -> Self; pub fn kind(self, MemoryType) -> Self;
                    pub fn matches(&self, record: &MemoryRecord) -> bool; pub fn is_empty(&self) -> bool;
                    pub fn scope_prefix(&self) -> Option<(&str, &str)>; }
```
`for_scope` copies user/scope/session/agent. `scope_prefix` returns `Some((user, scope))` only when both are set (allows targeted prefix scans).

### `options.rs`
```rust
#[derive(Debug, Clone, Default)]
pub struct CallOptions { pub cancellation: CancellationToken, pub timeout: Option<Duration> }
impl CallOptions { pub fn new() -> Self; pub fn with_cancellation(self, CancellationToken) -> Self; pub fn with_timeout(self, Duration) -> Self;
                   pub fn check(&self) -> Result<()>;           // Err(Error::Cancelled) when cancelled
                   pub async fn run<F: Future>(&self, future: F) -> Result<F::Output>; } // select! cancellation/timeout

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub enum SearchMode { #[default] Hybrid, Semantic, Keyword }

#[derive(Debug, Clone)]
pub struct AddOptions { pub dedupe: Option<DedupePolicy>, pub skip_embedding: bool, pub call: CallOptions }
impl Default for AddOptions; impl AddOptions { pub fn with_dedupe; pub fn skip_embedding(self) -> Self; pub fn with_cancellation; }

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: usize,                 // 5
    pub threshold: f32,               // 0.3
    pub kinds: Vec<MemoryType>,       // empty = engine.searchable_kinds
    pub mode: SearchMode,
    pub token_budget: Option<usize>,
    pub call: CallOptions,
}
impl Default for SearchOptions; impl SearchOptions { pub fn with_top_k; pub fn with_threshold; pub fn with_kinds; pub fn with_mode; pub fn with_token_budget; pub fn with_cancellation; }
```
`CallOptions` deliberately does not derive serde (a cancellation token is runtime state).

### `results.rs`
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallHit { #[serde(flatten)] pub record: MemoryRecord, pub score: f32,
                       pub similarity: f32, pub keyword: f32, pub importance: f32, pub recency: f32 }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AddOutcome { pub added: Vec<MemoryRecord>, pub merged: Vec<MemoryRecord>,
                        pub deleted: Vec<String>, pub skipped: Vec<String>,
                        pub degraded: Option<String> }   // why the dedupe scan did not run
impl AddOutcome { pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool; }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryStats { pub total: usize, pub by_kind: BTreeMap<String, usize>,
                         pub oldest: Option<DateTime<Utc>>, pub newest: Option<DateTime<Utc>>, pub total_tokens: usize }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AddMemResult { pub variables: BTreeMap<String, String>, pub user_profile: Vec<MemoryRecord>,
                          pub semantic_memory: Vec<MemoryRecord>, pub episodic_memory: Vec<MemoryRecord>,
                          pub summary: Vec<MemoryRecord>, pub coding: Vec<MemoryRecord>,
                          pub skipped: usize, pub deleted: usize, pub degraded: Option<String> }
impl AddMemResult { pub fn records(&self) -> impl Iterator<Item = &MemoryRecord>; pub fn len(&self) -> usize; }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CombinedRecall { pub memories: Vec<RecallHit>, pub entities: Vec<graph::GraphRecallHit>,
                            pub graph_error: Option<String> }
impl CombinedRecall { pub fn is_partial(&self) -> bool; }
```
`AddOutcome::degraded` names why duplicate detection could not run (the candidate scan failed), so a write that stored unchecked duplicates is distinguishable from a clean one; it is a diagnostic and is not counted by `len`/`is_empty`. `AddOutcome::deleted` is populated only by `Replace`: it names the row removed before the replacement was stored under that same id, so a replace counts once in `merged` and once in `deleted`.
`AddMemResult::skipped`/`deleted`/`degraded` carry the write bookkeeping the records cannot: dedupe-skipped inputs, rows a replace removed, and why model extraction fell back to the heuristics.
`CombinedRecall` lives in `graph::mod` as `{ memories, entities, graph_error }` and is re-exported; `graph_error` is `Some` when graph recall failed while the memory hits stayed complete, and `is_partial()` reports exactly that.

## Constraints

- Pure data + pure helpers only: no storage, no embedding, no `Model`, no async except `CallOptions::run` (a generic future selector).
- `MemoryRecord` JSON must round-trip and must accept Python shapes: `{"id","text","type","timestamp","fields","user_id","scope_id"}` and the `MemResult` aliases `"mem"`/`"mem_type"`.
- Use `BTreeMap` for metadata/variables so JSON output is deterministic.
- `MemoryType` `Variable` is the projection-only namespace and `Unknown` is the decode fallback; `NewMemory::validate`/`MemoryUpdate::validate` reject both.
- No panics: every fallible constructor returns `Result`.

## Routing Table

- `./scope.rs` → `MemoryScope`, `DEFAULT_ID`, validation/matching
- `./kind.rs` → `MemoryType`, searchable kinds, tolerant serde
- `./record.rs` → `MemoryRecord`, `new_memory_id`, `normalize_text`
- `./input.rs` → `NewMemory`, `MemoryUpdate`
- `./filter.rs` → `MemoryFilter`
- `./options.rs` → `CallOptions`, `AddOptions`, `SearchOptions`, `SearchMode`
- `./results.rs` → `RecallHit`, `AddOutcome`, `MemoryStats`, `AddMemResult`
- `./mod.rs` → module declarations and re-exports

## Test Strategy

- Unit tests in each file: scope validation matrix; scope decode requiring both tenant ids (a payload without `user_id`/`scope_id` fails, while `new`/`Default` still supply `DEFAULT_ID`); `MemoryType` serde incl. unknown; record JSON round-trip incl. Python `text`/`mem` and `type`/`mem_type` aliases; flattened scope fields; absent `timestamp`/`updated_at` decoding to the UNIX epoch while `MemoryRecord::new` stamps now; write-input validation rejecting `Variable` and `Unknown`; id shape (24 lowercase hex, deterministic length); update/filter matching; `AddOutcome`/`AddMemResult` degraded/skipped serialization; `CallOptions::check` after cancellation; `CallOptions::run` timeout.
- Integration JSON assertions live in `tests/crud.rs`; do not duplicate them here.
