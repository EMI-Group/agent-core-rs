# lite

## Intent

Lite coding memory: typed project/coding facts, served either from a `MemoryEngine` or from a local store rooted at one directory.
The crate has two paths.
(a) `CodingMemory` stores facts as ordinary `MemoryType::Coding` records on the shared engine, so they inherit scoping, dedupe/merge, recall, and every storage backend; `render_index`/`citation` are pure string builders.
(b) `LocalCodingMemory` is the port of Python `memory/lite/MemoryIndexManager`: a human-editable `MEMORY.md` is the source of truth, an FTS5 database under `.memory/` is the derived index, a cross-process advisory lock serializes mutations, and a debounced watcher picks up external edits.
On top of that base the local store adds, all opt-in or additive: a chunked embedding index with a content-hash cache (real local semantic/hybrid recall), a per-fact file layout with the `MEMORY.md`/`USER.md` index split, JSONL session recall, caller-driven sync triggers, `daily_memory` rollups, and write modes with a bounded conflict-retry checker.
Python's per-fact `.md` files plus its `MEMORY.md` summary index are mapped onto one canonical Markdown document per index level, with one file per fact when the per-fact layout is enabled.

## API Surface

### `CodingMemory` (engine-backed, unchanged)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CodingFactType {
    User,
    Feedback,
    #[default]
    Project,
    Reference,
}
impl CodingFactType { pub const ALL: [Self; 4]; pub const fn as_str(self) -> &'static str; }
impl std::str::FromStr for CodingFactType { type Err = Error; }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct CodingFact {
    pub name: String,                 // unique per scope; used as the lookup key
    pub description: String,          // one-line summary for the index
    #[serde(rename = "type", alias = "fact_type")]
    pub fact_type: CodingFactType,
    #[serde(flatten)]
    pub scope: MemoryScope,
    pub content: String,              // Markdown body
    pub tags: Vec<String>,
    pub path: Option<String>,         // optional source path for citations
    pub updated_at: DateTime<Utc>,
}
impl CodingFact { pub fn new(scope: &MemoryScope, name, content) -> Self;
                  pub fn with_description(self, impl Into<String>) -> Self;
                  pub fn with_type(self, CodingFactType) -> Self;
                  pub fn with_tags(self, Vec<String>) -> Self;
                  pub fn with_path(self, impl Into<String>) -> Self;
                  pub fn validate(&self) -> Result<()>; pub fn to_record(&self) -> MemoryRecord; pub fn from_record(record: &MemoryRecord) -> Option<Self>; }

#[derive(Clone)]
pub struct CodingMemory { /* engine: Arc<MemoryEngine>, config: CodingMemoryConfig */ }
impl CodingMemory {
    pub fn new(engine: Arc<MemoryEngine>) -> Self;               // uses engine.config().coding
    pub fn config(&self) -> &CodingMemoryConfig;
    pub async fn write(&self, fact: CodingFact, options: &AddOptions) -> Result<MemoryRecord>;
    pub async fn read(&self, scope: &MemoryScope, name: &str) -> Result<Option<CodingFact>>;
    pub async fn edit(&self, scope: &MemoryScope, name: &str, content: impl Into<String>, options: &CallOptions) -> Result<CodingFact>;
    pub async fn delete(&self, scope: &MemoryScope, name: &str) -> Result<bool>;
    pub async fn list(&self, scope: &MemoryScope) -> Result<Vec<CodingFact>>;   // updated_at desc, name asc
    pub async fn search(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    pub async fn render_index(&self, scope: &MemoryScope) -> Result<String>;    // `MEMORY.md`-shaped, max_index_lines
    pub fn citation(&self, hit: &RecallHit) -> String;                          // "path#L1-L5", "coding:<name>", or "coding:<id>"
}
```

### `local::{LocalCodingMemory, LocalCodingMemoryBuilder}`

`LocalCodingMemory` is the file-backed store. Clone it freely: every clone shares the document cache, the lock file, the derived indexes, and the watcher/interval-task slots.

```rust
impl LocalCodingMemory {
    pub fn builder(root: impl Into<PathBuf>, config: CodingMemoryConfig) -> LocalCodingMemoryBuilder;
    pub async fn open(root: impl AsRef<Path>, config: CodingMemoryConfig) -> Result<Self>;               // default scope, keyword-only, single-document layout
    pub async fn open_in_scope(root, config, scope: MemoryScope) -> Result<Self>;
    pub fn root(&self) -> &Path;  pub fn memory_path(&self) -> &Path;  pub fn user_path(&self) -> PathBuf;
    pub fn lock_path(&self) -> &Path;  pub fn config(&self) -> &CodingMemoryConfig;  pub fn scope(&self) -> &MemoryScope;
    pub fn reload_count(&self) -> u64;
    pub async fn markdown(&self) -> Result<MarkdownMemory>;

    // Facts (name-keyed upsert semantics, identical to the engine-backed path)
    pub async fn write(&self, fact: CodingFact) -> Result<CodingFact>;
    pub async fn read(&self, name: &str) -> Result<Option<CodingFact>>;
    pub async fn edit(&self, name: &str, content: impl Into<String>) -> Result<CodingFact>;
    pub async fn delete(&self, name: &str) -> Result<bool>;
    pub async fn list(&self) -> Result<Vec<CodingFact>>;
    pub async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    pub async fn render_index(&self) -> Result<String>;     // compact prompt summary, not the on-disk document
    pub async fn reindex(&self) -> Result<usize>;
    #[cfg(feature = "sqlite")] pub async fn indexed_count(&self) -> Result<usize>;
    pub async fn start_watching(&self) -> Result<()>;
    pub async fn stop_watching(&self);
    pub async fn is_watching(&self) -> bool;

    // Embedding index
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>>;
    pub fn has_embedding(&self) -> bool;
    pub fn chunking(&self) -> &ChunkingConfig;
    pub async fn embedding_chunks(&self) -> usize;
    pub async fn refresh_embeddings(&self) -> Result<EmbeddingStats>;
    pub async fn embedding_cache_len(&self) -> Result<usize>;

    // Per-fact files + MEMORY.md/USER.md split
    pub fn fact_files_enabled(&self) -> bool;
    pub fn facts_dir(&self) -> PathBuf;                     // <root>/facts
    pub async fn export_index(&self) -> Result<usize>;      // facts -> fact files + both index documents
    pub async fn import_index(&self) -> Result<usize>;      // hand-edited index documents -> facts + fact files

    // Sync triggers
    pub async fn sync_now(&self) -> Result<SyncReport>;
    pub async fn on_trigger(&self, trigger: SyncTrigger) -> Result<SyncReport>;
    pub async fn start_interval_sync(&self, interval: Duration) -> Result<()>;
    pub async fn stop_interval_sync(&self);
    pub async fn is_interval_sync_running(&self) -> bool;

    // Sessions
    pub fn sessions_dir(&self) -> PathBuf;                  // <root>/sessions
    pub async fn append_session(&self, session_id: &str, date: NaiveDate, records: &[SessionRecord]) -> Result<PathBuf>;
    pub async fn session_records(&self) -> Result<Vec<SessionChunk>>;
    pub async fn search_sessions(&self, query: &str, options: &SearchOptions) -> Result<Vec<SessionHit>>;

    // Daily rollups
    pub fn daily_dir(&self) -> PathBuf;                     // <root>/daily_memory
    pub async fn rollup_day(&self, date: NaiveDate) -> Result<DailyRollup>;        // renders only
    pub async fn write_daily_rollup(&self, date: NaiveDate) -> Result<DailyRollup>; // render + idempotent write

    // Write modes
    pub async fn write_mode(&self, fact: CodingFact, mode: WriteMode, policy: WritePolicy) -> Result<WriteOutcome>;
}

pub struct LocalCodingMemoryBuilder { /* root, config, scope, embedding, chunking, fact_files */ }
impl LocalCodingMemoryBuilder {
    pub fn new(root: impl Into<PathBuf>, config: CodingMemoryConfig) -> Self;
    pub fn scope(self, scope: MemoryScope) -> Self;
    pub fn embedding(self, embedding: Arc<dyn Embedding>) -> Self;
    pub fn chunking(self, chunking: ChunkingConfig) -> Self;
    pub fn fact_files(self, enabled: bool) -> Self;          // default false
    pub async fn open(self) -> Result<LocalCodingMemory>;
}
```

### `markdown::{MarkdownMemory, FrontMatter}`

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrontMatter { /* ordered key/value pairs + a verbatim raw tail */ }
impl FrontMatter {
    pub fn new() -> Self;
    pub fn get(&self, key: &str) -> Option<&str>;
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> Option<String>; // keeps an existing key's position
    pub fn remove(&mut self, key: &str) -> Option<String>;
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarkdownMemory { /* front_matter, title, blocks, scope */ }
impl MarkdownMemory {
    pub fn new() -> Self;                                       // schema front matter + `# Memory`
    pub fn parse(text: &str) -> Result<Self>;
    pub fn render(&self) -> String;                             // ends with exactly one "\n"
    pub fn scope(&self) -> &MemoryScope;
    pub fn with_scope(self, scope: MemoryScope) -> Self;        // rebinds every fact's scope
    pub fn set_scope(&mut self, scope: MemoryScope);
    pub fn front_matter(&self) -> &FrontMatter;
    pub fn with_front_matter(self, key: impl Into<String>, value: impl Into<String>) -> Self;
    pub fn title(&self) -> Option<&str>;
    pub fn with_title(self, title: impl Into<String>) -> Self;  // `# Memory` by default, `# User` for USER.md
    pub fn set_title(&mut self, title: impl Into<String>);
    pub fn upsert_fact(&mut self, fact: &CodingFact) -> bool;   // true when it replaced an entry
    pub fn remove(&mut self, name: &str) -> bool;
    pub fn fact(&self, name: &str) -> Option<CodingFact>;
    pub fn facts(&self) -> Vec<CodingFact>;                     // document order
    pub fn is_empty(&self) -> bool;
}
```

### `embed::{chunk, cache, index}`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ChunkingConfig { pub target_tokens: usize /*256*/, pub overlap: usize /*32*/ }
impl ChunkingConfig { pub fn validate(&self) -> Result<()>; }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemChunk { pub text: String, pub start_line: usize, pub end_line: usize }   // 1-based inclusive
pub fn estimate_tokens(text: &str) -> usize;                                    // chars / 4
pub fn chunk_markdown(content: &str, config: &ChunkingConfig) -> Vec<MemChunk>;

pub fn content_hash(text: &str) -> String;                                      // 16 lowercase hex, FNV-1a
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord)]
pub struct CacheKey { pub model: String, pub hash: String }
#[derive(Clone)]
pub struct EmbeddingCache { /* Arc inner: BTreeMap and/or SqlitePool */ }
impl EmbeddingCache {
    pub fn in_memory() -> Self;
    #[cfg(feature = "sqlite")] pub async fn sqlite(pool: SqlitePool) -> Result<Self>;
    pub fn is_persistent(&self) -> bool;
    pub async fn get(&self, model: &str, hash: &str) -> Result<Option<Vec<f32>>>;
    pub async fn put(&self, model: &str, hash: &str, vector: &[f32]) -> Result<()>;
    pub async fn len(&self) -> Result<usize>;
    pub async fn clear(&self) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedChunk { pub id: String, pub document: String, pub chunk: MemChunk, pub vector: Option<Vec<f32>> }  // id = "<document>:<start>:<end>"
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmbeddingStats { pub chunks: usize, pub embedded: usize, pub cached: usize, pub dimension: Option<usize> }
#[derive(Clone)]
pub struct EmbeddingIndex { /* chunks + cache + model */ }
impl EmbeddingIndex {
    pub fn new(model: impl Into<String>, cache: EmbeddingCache) -> Self;
    pub fn model(&self) -> &str;  pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn chunks(&self) -> &[IndexedChunk];
    pub async fn reindex(&mut self, embedding: &dyn Embedding, documents: &[(String, String)], config: &ChunkingConfig) -> Result<EmbeddingStats>;
    pub async fn search(&self, embedding: &dyn Embedding, query: &str, limit: usize) -> Result<Vec<ScoredChunk>>;
    pub fn clear(&mut self);
}
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredChunk { pub id: String, pub document: String, pub chunk: MemChunk, pub similarity: f32 }
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32;   // 0.0 on a dimension mismatch or a zero-norm vector
pub fn similarity_from_cosine(cosine: f32) -> f32;               // (cos + 1) / 2, clamped 0.0..=1.0
pub fn blend(semantic: f32, keyword: f32, vector_weight: f32, text_weight: f32) -> f32;  // weights renormalized by their sum
```

### `facts::{FactFiles, FactDocument, IndexLevel}`

```rust
pub const FACTS_DIR: &str = "facts";
pub const MEMORY_FILE: &str = "MEMORY.md";
pub const USER_FILE: &str = "USER.md";
pub fn slug(name: &str) -> String;                 // lowercase alphanumerics, other runs -> '-', 48-char cap
pub fn file_name_for(name: &str) -> String;        // "<slug>-<8 hex of the exact name>.md"

pub struct FactDocument { pub front_matter: FrontMatter, pub body: String }
impl FactDocument {
    pub fn new() -> Self;  pub fn parse(text: &str) -> Result<Self>;  pub fn render(&self) -> String;
    pub fn from_fact(fact: &CodingFact) -> Self;
    pub fn to_fact(&self, scope: &MemoryScope) -> Result<CodingFact>;
    pub fn name(&self) -> Option<&str>;  pub fn fact_type(&self) -> Option<CodingFactType>;
    pub fn description(&self) -> Option<&str>;  pub fn tags(&self) -> Vec<String>;
    pub fn path(&self) -> Option<&str>;  pub fn updated_at(&self) -> Option<DateTime<Utc>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexLevel { Memory, User }
impl IndexLevel {
    pub const ALL: [Self; 2];
    pub fn file_name(self) -> &'static str;   // "MEMORY.md" / "USER.md"
    pub fn title(self) -> &'static str;       // "Memory" / "User"
    pub fn contains(self, fact_type: CodingFactType) -> bool;
}
pub fn level_of(fact_type: CodingFactType) -> IndexLevel;    // User -> User, everything else -> Memory
pub fn render_level(facts: &[CodingFact], level: IndexLevel) -> String;   // deterministic (type order, then name)
pub fn parse_level(text: &str) -> Result<Vec<CodingFact>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirStamp { pub files: usize, pub total_len: u64, pub newest_ms: u128 }
#[derive(Debug, Clone)]
pub struct FactFiles { /* root, scope */ }
impl FactFiles {
    pub fn new(root: impl AsRef<Path>, scope: MemoryScope) -> Self;
    pub fn root(&self) -> &Path;  pub fn dir(&self) -> PathBuf;  pub fn path_for(&self, name: &str) -> PathBuf;
    pub async fn write(&self, fact: &CodingFact) -> Result<PathBuf>;      // atomic temp file + rename
    pub async fn read(&self, name: &str) -> Result<Option<CodingFact>>;
    pub async fn read_all(&self) -> Result<Vec<CodingFact>>;              // sorted by name; non-fact Markdown skipped with a warn
    pub async fn remove(&self, name: &str) -> Result<bool>;
    pub async fn contains(&self, name: &str) -> Result<bool>;
    pub async fn stamp(&self) -> Result<DirStamp>;
}
```

### `session::{SessionStore, SessionRecord, SessionChunk, SessionHit}`

```rust
pub const SESSIONS_DIR: &str = "sessions";
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionRecord { pub timestamp: Option<DateTime<Utc>>, pub role: Option<String>, pub content: String }
impl SessionRecord {
    pub fn new(content: impl Into<String>) -> Self;  pub fn with_role(self, role) -> Self;  pub fn with_timestamp(self, ts) -> Self;
    pub fn is_empty(&self) -> bool;  pub fn text(&self) -> String;             // "role: content" when a role is set
    pub fn parse_line(line: &str) -> Option<Self>;                             // tolerant, never fails
    pub fn to_json_line(&self) -> String;
}
pub struct SessionChunk { pub path: String, pub line: usize, pub record: SessionRecord }
pub struct SessionHit { pub path: String, pub line: usize, pub score: f32, pub similarity: f32, pub keyword: f32, pub record: SessionRecord }
pub fn keyword_score(query: &str, text: &str) -> f32;
pub fn rank_chunks(chunks: &[SessionChunk], query: &str, threshold: f32, limit: usize) -> Vec<SessionHit>;
pub fn sanitize_session_id(session_id: &str) -> Result<String>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirStamp { pub files: usize, pub total_len: u64, pub newest_ms: u128 }   // local to this module
pub struct SessionStore { /* root */ }
impl SessionStore {
    pub fn new(root: impl AsRef<Path>) -> Self;
    pub fn root(&self) -> &Path;  pub fn dir(&self) -> PathBuf;
    pub fn path_for(&self, session_id: &str, date: NaiveDate) -> Result<PathBuf>;   // sessions/<YYYY-MM-DD>/<id>.jsonl
    pub async fn append(&self, session_id: &str, date: NaiveDate, records: &[SessionRecord]) -> Result<PathBuf>;
    pub async fn files(&self) -> Result<Vec<PathBuf>>;          // recursive *.jsonl, sorted
    pub async fn load(&self, path: &Path) -> Result<Vec<SessionRecord>>;
    pub async fn chunks(&self) -> Result<Vec<SessionChunk>>;    // deterministic (path, line)
    pub async fn stamp(&self) -> Result<DirStamp>;
}
```

### `sync::{SyncTrigger, SyncReport, IntervalScheduler}`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncTrigger { OnSessionStart, OnInterval(Duration), OnWrite }
impl SyncTrigger {
    pub const ALL_KINDS: [&'static str; 3];
    pub fn label(&self) -> &'static str;      // "on_session_start" / "on_interval" / "on_write"
    pub fn interval(&self) -> Option<Duration>;
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncReport { pub ran: bool, pub trigger: &'static str, pub facts: usize, pub chunks: usize, pub embedded: usize, pub sessions: usize }
impl SyncReport {
    pub fn skipped(trigger: &'static str) -> Self;
    pub fn new(trigger: &'static str, facts: usize, chunks: usize, embedded: usize, sessions: usize) -> Self;
}

#[derive(Debug, Clone)]
pub struct IntervalScheduler { /* interval, last */ }
impl IntervalScheduler {
    pub fn new(interval: Duration) -> Result<Self>;   // zero -> Validation
    pub fn interval(&self) -> Duration;
    pub fn is_due(&self, now: Instant) -> bool;       // never run, or now - last >= interval; a backwards clock is not due
    pub fn mark(&mut self, now: Instant);
    pub fn poll(&mut self, now: Instant) -> bool;     // due -> mark + true
    pub fn last(&self) -> Option<Instant>;
    pub fn reset(&mut self);
}
```

### `daily::{DailyRollup, DailySessionLine, render_rollup, write_rollup, select_rollup_inputs}`

```rust
pub const DAILY_DIR: &str = "daily_memory";
pub fn rollup_id(date: NaiveDate) -> String;                  // "daily:YYYY-MM-DD"
pub fn file_name(date: NaiveDate) -> String;                  // "YYYY-MM-DD.md"
pub fn path_for(root: &Path, date: NaiveDate) -> PathBuf;     // <root>/daily_memory/YYYY-MM-DD.md
pub struct DailySessionLine { pub path: String, pub line: usize, pub date: Option<NaiveDate>, pub text: String }
pub struct DailyRollup {
    pub id: String, pub date: NaiveDate, pub facts: Vec<String>, pub sessions: Vec<String>,
    pub content: String, pub written: bool,
}
pub fn render_rollup(date: NaiveDate, facts: &[CodingFact], session_lines: &[DailySessionLine]) -> String;
pub fn select_rollup_inputs(date, facts, session_lines) -> (Vec<String>, Vec<String>);   // same filters/order the renderer uses
pub async fn write_rollup(root: &Path, date: NaiveDate, content: &str) -> Result<(PathBuf, bool)>;  // (path, changed)
```

### `write::{WriteMode, WritePolicy, WritePlan, WriteOutcome, MemUpdateChecker}`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteMode { Create, Append, #[default] Merge, Replace, Skip }
impl WriteMode { pub const ALL: [Self; 5]; pub fn as_str(self) -> &'static str; }
impl std::str::FromStr for WriteMode { type Err = Error; }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritePolicy { pub max_attempts: usize /*2*/, pub backoff: Duration /*5ms*/, pub max_backoff: Duration /*80ms*/ }
impl WritePolicy { pub fn validate(&self) -> Result<()>; pub fn with_max_attempts(self, n: usize) -> Self; pub fn with_backoff(self, b: Duration, max: Duration) -> Self; }

#[derive(Debug, Clone, PartialEq)]
pub enum WritePlan { Created(CodingFact), Replaced(CodingFact), Merged(CodingFact), Appended(CodingFact), Skipped(Option<CodingFact>) }

#[derive(Debug, Clone, PartialEq)]
pub struct WriteOutcome {
    pub fact: CodingFact, pub mode: WriteMode, pub created: bool,
    pub attempts: usize, pub conflict: bool, pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MemUpdateChecker { /* policy */ }
impl MemUpdateChecker {
    pub fn new(policy: WritePolicy) -> Result<Self>;
    pub fn policy(&self) -> &WritePolicy;
    pub fn backoff_for(&self, attempt: usize) -> Duration;      // backoff * 2^attempt, capped
    pub fn plan(&self, mode: WriteMode, incoming: &CodingFact, existing: Option<&CodingFact>, now: DateTime<Utc>) -> Result<WritePlan>;
    pub async fn run<T, F, Fut>(&self, what: &str, op: F) -> Result<(T, usize)>      // Ok(None) = conflict -> retry
    where F: FnMut(usize) -> Fut, Fut: Future<Output = Result<Option<T>>>;
}
```

### `lock::FileLock`, `watch::{MemoryWatcher, WatchEvent, WatchKind}`, `fts::{FtsIndex, FtsHit}`

- `lock::FileLock`: `acquire` (waits for the holder), `try_acquire` (`Ok(None)` when held elsewhere), `path`; `#[cfg(unix)]` `flock` via `nix`, other platforms `Error::Unsupported`.
- `watch::{WatchKind, WatchEvent, MemoryWatcher}`: debounced coalescing watcher on `notify`; `MemoryWatcher::start(paths, debounce, sender)`, `stop`, `is_running`.
- `fts::{FtsIndex, FtsHit}` (feature `sqlite`): `open`/`open_path`/`from_pool`/`in_memory`/`pool`, `upsert`, `remove`, `search`, `rebuild`, `count`, `clear`.

## On-disk layout

Default single-document layout (unchanged):

```text
<root>/
  MEMORY.md            the source of truth (fact sections + bodies)
  .memory/index.db     derived SQLite FTS5 index + the embedding cache (feature `sqlite`)
  .memory/index.lock   cross-process advisory lock file
```

Per-fact layout (`LocalCodingMemoryBuilder::fact_files(true)`):

```text
<root>/
  facts/<slug>-<hash>.md   one fact per file, authoritative for content
  MEMORY.md                rendered index of the non-user facts
  USER.md                  rendered index of the `user`-level facts
  .memory/index.db         derived index, unchanged
  .memory/index.lock       advisory lock, unchanged
```

Independent of the layout, three further surfaces live under the same root:

```text
<root>/
  sessions/<YYYY-MM-DD>/<session-id>.jsonl   append-only JSONL session logs
  daily_memory/<YYYY-MM-DD>.md               derived daily rollups
```

- Both fact layouts share the same lock file, derived index, watcher, and search path; only the authoritative representation and the staleness check differ.
- The default layout is byte-for-byte what it was before the per-fact layout existed.
- `open`/`open_in_scope` (default layout) create the root and `.memory/`, write an empty `MEMORY.md` when missing, take the file lock, load the document, then rebuild the index.
- With the per-fact layout, a fresh open adopts `facts/` when it holds at least one readable fact, otherwise it exports the loaded `MEMORY.md` into fact files plus both index documents. Both index documents are always created.

## Consistency model

- The Markdown representation wins over every derived index; the FTS table and the chunk index may always be rebuilt (`reindex`).
- In the per-fact layout, per-fact files are authoritative for content, and a changed index document wins over them (a hand edit is caller intent, `import_index`).
- A mutation takes the file lock, reloads when a stamp changed, applies the change, writes atomically (temporary file plus rename), and updates the indices; a reader applies the same staleness check, so another process's committed fact is visible without an explicit refresh.
- Staleness in the per-fact layout compares `MEMORY.md`/`USER.md` stamps and the `facts/` `DirStamp`; index-document changes trigger an import, fact-file changes trigger a re-render, and when both changed the index wins with a `tracing::warn!`.
- Mutations are additionally serialized inside the process by an async mutex; the file lock alone would work but is a blocking system call.
- A read path that observes a changed file without holding the file lock updates the cache and marks the projections dirty; every locked path flushes them, which preserves the file-lock-then-state-guard order.
- That read-path reload rebuilds only the document and the FTS index (`store::ensure_fresh`); the chunk index is refreshed solely by mutations, `reindex`, `sync_now`, `export_index`/`import_index` and `open`, so a `search` after an external edit scores changed facts against stale vectors.
- A failed `EmbeddingIndex::reindex` keeps the previously committed chunk list, but the first refresh after open inserts the empty index into its slot before reindexing, so a failure there leaves an empty index that later `search` calls read as `similarity = 0.0` with no error.
- `start_watching` reloads (and reindexes) on every debounced `.md` change; `reload_count` is the observable reload counter.
- `write`/`edit`/`delete`/`reindex` refresh the chunk index after the state guard is dropped, so no guard is ever held across an embedding call.

## Markdown document format

- The front matter is a `---`-fenced block of `key: value` lines; a fresh document carries `schema: openjiuwen-memory-lite` and `version: 1`.
- Front matter keys keep document order, an existing key keeps its position on insert, and lines that are not a `key: value` pair are preserved verbatim in a raw tail.
- The title is the first `# …` line (`# Memory` in a fresh document, `# User` for `USER.md`); a document without one renders no title line.
- Fact sections are `## <Type>` in canonical order (user, feedback, project, reference); an empty section renders nothing.
- Each fact is a `### <name>` entry followed by a bullet attribute block and then the body.
- Interpreted attributes: `- description:`, `- tags: ["a","b"]` (a comma-separated value is also accepted on parse), `- path:`, `- updated_at:` (RFC 3339).
- Unknown entry attributes are preserved across an edit, and a replaced fact keeps them.
- A missing or unparsable `updated_at` decodes to `DateTime<Utc>::default` (the Unix epoch), never "now".
- Input is normalized to LF; `render` emits LF and exactly one trailing newline, with one blank line between blocks.
- A body line that would look like a heading is escaped with a backslash on render (`\## note`) and unescaped on parse.
- Body boundaries are normalized: leading/trailing blank lines are dropped, and blank lines between heading, attributes, and body are canonical.
- `description` and `path` are single-line by construction (embedded newlines fold to spaces).
- The only parse error is an unterminated front-matter block whose content holds `:`-bearing or attribute lines (`Error::Validation`); every other input parses tolerantly.
- `parse(render(doc)) == doc`, so the renderer output is a fixpoint.

## Per-fact file format and the USER.md split

- One file per fact at `<root>/facts/<slug>-<hash>.md`: `slug` is the lowercased name with non-alphanumeric runs folded to `-` (48-char cap), and the 8-hex hash of the exact name keeps distinct names distinct.
- The front matter keys are `name`, `description`, `type`, `tags` (JSON array), `path` (only when set), `updated_at` (RFC 3339); the body is the fact content. `FactDocument::{from_fact, to_fact}` round trip every `CodingFact` field.
- Python's `created_at` is not represented because `CodingFact` has no creation timestamp.
- `IndexLevel::User` holds `CodingFactType::User`; every other type belongs to `IndexLevel::Memory`, which is rendered into `MEMORY.md`; `USER.md` renders the user level only.
- `render_level` filters to the level, sorts by the fixed type order then name ascending, and renders through `MarkdownMemory::new().with_title(level.title())`, so output is byte-deterministic.
- `parse_level` accepts hand-edited documents (unknown front-matter keys, extra sections, loose text) without error.
- Import merges both documents; a name present in both keeps the `USER.md` entry and logs a warning.
- `export_index` removes fact files whose fact no longer exists, so a delete propagates to disk.
- Unknown keys inside a *fact file's* front matter are preserved on parse into `FactDocument` but a rewrite re-renders the canonical key set; index-level unknown attributes are preserved by `MarkdownMemory`.

## FTS5 index

- Schema: `CREATE VIRTUAL TABLE IF NOT EXISTS coding_facts_fts USING fts5(name UNINDEXED, content, description, tags, tokenize='unicode61')`, applied idempotently by every constructor.
- `tags` is stored space-joined (`["build", "ci"]` → `build ci`), so each tag is an individual FTS term.
- `upsert` is name-keyed: delete plus insert in one transaction.
- `rebuild` clears and reinserts in one transaction and returns the indexed row count.
- Results are ordered by `bm25()` ascending (more negative is better), with a `snippet()` excerpt per hit.
- The MATCH expression is safe by construction: `\w+` tokens, at most 10 with an alphanumeric character, each double-quoted with `"` doubled, joined with `OR`, bound as a statement parameter; a query with no usable token returns no hits without querying.
- The `name` column is UNINDEXED, so name-only matches come from the deterministic keyword scan.

## Embedding index and cache

- `ChunkingConfig` defaults to `target_tokens = 256`, `overlap = 32`; `estimate_tokens` is `chars / 4`.
- `chunk_markdown` is line-granular: it flushes when adding a line would exceed `target_tokens` (so one oversized line becomes its own chunk), carries the maximal suffix of the flushed chunk whose tokens fit in `overlap` into the next chunk, and reports 1-based inclusive `start_line`/`end_line`.
- Documents are `(fact name, fact content)` for facts and `("<path>:<line>", record text)` for sessions; a chunk id is `"<document>:<start>:<end>"` and `IndexedChunk::document`/`ScoredChunk::document` carry the key back.
- The cache is keyed by `(embedding.model_name(), content_hash(chunk text))`; `content_hash` is 16 hex characters of an FNV-1a digest, chosen because the workspace has no `sha2` dependency (Python uses `sha256[:16]`).
- `reindex` re-chunks every document, serves every cache hit, embeds all misses in ONE `embed_documents(&texts, None)` batch, and only then commits the new chunk list.
- Query embeddings (`embed_query`) are never cached.
- The in-memory cache is a `parking_lot` map (never held across `.await`); with the `sqlite` feature and an embedding configured, the store opens `EmbeddingCache::sqlite(fts.pool().clone())`, which persists vectors in `lite_embedding_cache(model, hash, dims, vector BLOB)` so a reopen embeds nothing.
- Fact vectors live in memory per store handle (rebuilt on open and after every mutation); only the cache is durable.
- `SearchMode::Keyword` never calls the model, not even for the query.

## Search scoring

### Facts: keyword and FTS

- The keyword score is the sum over query tokens of the strongest field the token matches (name 3, description or tags 2, body 1), divided by three times the token count, capped at `1.0`.
- An FTS hit contributes `1.0 - FtsHit::score` (the FTS score is higher for weaker matches); a fact keeps the maximum of its two signals.
- `top_k`, `threshold`, `kinds`, `mode`, and the call's cancellation state are honoured; `token_budget` is ignored (there is no context window).
- `kinds` empty means every kind; a list without `MemoryType::Coding` yields no hit.

### Facts: semantic and hybrid

- A fact's semantic score is the maximum `similarity_from_cosine(cosine)` over its own chunks, normalized to `0.0..=1.0`; a fact without vectors contributes `0.0`.
- The blend is `(0.7 * semantic + 0.3 * keyword) / 1.0` (`VECTOR_WEIGHT = 0.7`, `TEXT_WEIGHT = 0.3`, matching Python's `vectorWeight`/`textWeight`).
- `SearchMode::Semantic` scores by the semantic signal alone and is `Error::Unsupported` (naming `builder(..).embedding(..)`) when no embedding is configured.
- `SearchMode::Hybrid` (the default) blends the two signals with an embedding and degrades to keyword-only without one.
- `threshold` applies to the final score; hits are ordered by score descending then name ascending, then truncated to `top_k.max(1)`.
- `RecallHit` reports `similarity` = semantic component, `keyword` = keyword component, `importance` = `0.5`, `recency` = `0.5^(age / 30 days)` clamped to `0.0..=1.0`.

### Sessions

- `search_sessions` scores each JSONL record with `keyword_score` (query `\w+` tokens, at most 10, matched as substrings of the lowercased `record.text()`, divided by the token count).
- With an embedding configured the semantic side reuses the store's cache through a transient `EmbeddingIndex`, so an unchanged line is never embedded twice; `Hybrid` blends `0.7`/`0.3`, `Semantic` requires an embedding, `Keyword` makes no model call.
- `threshold` applies to the final score; hits are ordered by score descending, then path ascending, then line ascending, and truncated to `top_k.max(1)`.
- An empty `kinds` recalls sessions; a non-empty list recalls them only when it contains `MemoryType::EpisodicMemory`.
- Sessions are never copied into `MEMORY.md`/`USER.md`; they are their own recall surface.

## Write modes and conflict retries

- `WriteMode`: `create` (fail when the name exists), `append` (join bodies with one blank line; a no-op append becomes a skip), `merge` (the default: incoming non-blank content/description/path win, tags union in order, stored name/type survive), `replace` (the incoming fact wins wholesale, a `None` path clears the stored one), `skip` (no write when the fact exists).
- Python defines only `create`/`append`/`skip` (`conflict_types.py`); `merge`/`replace` are Rust extensions, documented as such.
- `WritePolicy` defaults to `max_attempts = 2`, `backoff = 5 ms`, `max_backoff = 80 ms`; `backoff_for(attempt)` is `backoff * 2^attempt`, saturating and capped.
- `MemUpdateChecker::run` retries `Ok(None)` (a conflict) with the backoff and returns the value plus the 1-based attempt count.
- `LocalCodingMemory::write_mode` treats a `create` collision as `Ok(None)`, so the checker re-reads under the lock and retries, and exhaustion is a typed `Error::Memory` naming the fact and the attempt count. This deliberately deviates from Python's last-writer-wins degradation.
- `WriteOutcome` reports the applied mode, whether the fact was created, the attempts used, whether a conflict was retried, and a note for a skip.
- `LocalCodingMemory::write` keeps its original full-replacement (upsert) semantics and is unaffected by the write-mode surface.

## Sync triggers

- `SyncTrigger` has three caller-driven forms: `OnSessionStart`, `OnInterval(Duration)`, `OnWrite`; nothing starts a background task implicitly.
- `sync_now` refreshes the document (including the per-fact reconciliation), rebuilds the FTS index, refreshes the chunk index when an embedding is configured, and counts indexed session records.
- `on_trigger(OnWrite)` only reloads when the file changed (`ran` reports whether it did); `OnSessionStart` always runs a full sync; `OnInterval(d)` polls an `IntervalScheduler` and returns `SyncReport::skipped("on_interval")` when not due.
- `IntervalScheduler` takes `now` from the caller, so it is deterministic in tests; a zero interval is `Error::Validation` and a backwards clock is not due.
- `start_interval_sync(interval)` spawns one task per store handle holding a `CancellationToken`; it syncs on each tick, logs a failure with `tracing::warn!`, replaces any previous task, and `stop_interval_sync` cancels, aborts, and joins it. No global or static state.
- Python's `sync.onSessionStart` setting is read by nothing in the Python tree; here the trigger is an explicit API the harness must call.

## Daily rollups

- Rollups live at `<root>/daily_memory/YYYY-MM-DD.md` with the id `daily:YYYY-MM-DD`.
- Front matter: `id`, `date`, `facts` (count), `sessions` (count); title `# Daily Memory <YYYY-MM-DD>`; sections `## Facts` and `## Sessions`, each rendering `_None._` when empty.
- Facts are included when `updated_at.date_naive() == date`, ordered by the fixed type order then name; session lines are included when their record timestamp's date equals `date`, ordered by `(path, line)`.
- A session line renders as `- <path>:<line> — <text>` with newlines folded to spaces and a 300-character truncation on a char boundary with a trailing `…`.
- `write_rollup` writes atomically and reports `changed = false` for identical bytes, so a rerun is idempotent.
- Rollups are derived documents: they are never imported as facts, and nothing injects them into a prompt.

## Session log format

- One JSON object per line at `<root>/sessions/<YYYY-MM-DD>/<sanitized id>.jsonl`; `sanitize_session_id` keeps ASCII alphanumerics plus `-`, `_`, `.`, folds everything else to `-`, and rejects an empty result (`Error::Validation`), so no separator survives and traversal is impossible.
- Decoding is tolerant per line: `content` from `content`/`text`/`message`/`output`, `role` from `role`/`type`, `timestamp` from `timestamp`/`ts`/`time` as RFC 3339 or an epoch number (values ≥ 1e11 are milliseconds); a malformed, non-object, or content-less line becomes a plain-text record of the raw line; a blank line is skipped.
- `append` skips whitespace-only records and rejects a set with no non-empty record (`Error::Validation`).
- `files` walks the tree recursively and sorts; `chunks` numbers lines 1-based with store-root-relative `/` paths.
- Nothing in this workspace writes transcripts yet; the store exposes `append_session` for a caller that does.

## Record Encoding

- `kind = MemoryType::Coding`; `content` = fact Markdown body; `updated_at` mirrors `CodingFact::updated_at`.
- metadata keys: `fact_type` (string), `name`, `description`, `tags` (array of strings, always present), `path` (only when set), `coding_fact` = `true`.
- `CodingFact::from_record` accepts only records whose kind is `MemoryType::Coding`, whose `coding_fact` metadata is exactly `true`, and whose metadata `name` is a non-blank string; decoding is otherwise tolerant (unknown fact type falls back to `project`, non-string tags are dropped, null/missing path is `None`).
- `id` is a normal 24-hex memory id; lookup by name lists scoped coding records and compares metadata `name` (case-sensitive).
- `write` upserts by name: an existing fact with the same name in the scope keeps its id, and a cleared path is stored as JSON `null` so the merged metadata really drops the old path.
- `write` validates the name (non-empty, at most 128 characters, no `/` or `\`) and resolves the fact type against `config.fact_types`; a type absent from that list is replaced by `config.default_fact_type` with a debug log and never fails the write.
- `edit` updates the body and `updated_at`; a missing fact is `Error::NotFound`.
- `read`/`delete` return `None`/`false` for a missing fact.
- `list` orders by `updated_at` descending, then name ascending; `search` forces `SearchOptions::kinds` to `[MemoryType::Coding]` and passes every other knob through.

## Index Rendering

`index::render_index` (reachable as `CodingMemory::render_index` and `LocalCodingMemory::render_index`) is a pure string builder; nothing is written to disk.
- The first line is always `# Memory`.
- One bullet per fact is formatted exactly `- [<Type>] <name> — <description>`, with ` (`<path>`)` appended when the fact has a path.
- Facts are grouped in the fixed declaration order user, feedback, project, reference and sorted name-ascending inside each group.
- There is no blank line between groups, and the result always ends with a trailing newline.
- `max_lines` counts the title line, so `max_lines <= 1` yields the title alone; an empty scope yields exactly `"# Memory\n"`.
- The cap is `CodingMemoryConfig::max_index_lines` (default 200).
- `render_index` is the compact prompt-facing summary, not the on-disk document.

`index::citation` (reachable as `CodingMemory::citation(&RecallHit)`) returns `<path>#L1-L<n>` when the record has `path` metadata, where `n` is the content's newline count + 1; otherwise `coding:<name>`, falling back to `coding:<id>` when the metadata name is missing or blank.

## Concurrency and errors

- Lock order is file lock before state guard everywhere; the watcher task and the interval task only take the state guard (through the public API), so no cycle exists.
- Mutations are additionally serialized by a per-store async mutex, which keeps concurrent tasks from occupying blocking threads on the file lock.
- No blocking call runs on the async reactor: `flock` (and the lock-file open) runs in `spawn_blocking`.
- Filesystem work uses `tokio::fs`; the watcher's debounce task and the interval task live on the caller's tokio runtime and are owned by the store handle.
- Per-fact writes are atomic (hidden temp file plus rename in the same directory); index-document writes reuse the same helper.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for bad input (invalid fact, blank query or edit body, invalid scope, bad SQLite URL/limit, zero interval, empty session id), `NotFound` for a missing fact on `edit`, `Unsupported` for semantic search without an embedding, per-fact APIs without the flag, and non-Unix locking, `Memory` for an exhausted write conflict, and `External` (with the failing path as context) for filesystem, lock, and index failures.
- Nothing panics on user input: no `unwrap`/`expect` outside tests.

## Constraints

- `CodingMemory` uses only the public `MemoryEngine` API; no direct store access, no separate persistence.
- Coding facts are returned by `LongTermMemory::search` only when `MemoryType::Coding` is in the searched kinds (it is in the default `searchable_kinds`).
- `CodingMemory::render_index`/`citation` write nothing; `LocalCodingMemory` owns the file and is the only writer of `MEMORY.md`, `USER.md`, `facts/`, `sessions/`, `daily_memory/`, and `.memory/`.
- The derived indexes are never the source of truth: both may be dropped and rebuilt from the Markdown representation.
- Default builds work without the `sqlite` feature: `LocalCodingMemory` then reloads and keyword-searches the Markdown document with no FTS index and keeps the embedding cache in memory only. Every new module compiles and is tested in both builds.
- Every new feature is opt-in or additive: the default `open` keeps today's single-document, keyword-only behaviour, and `tests/lite.rs`/`tests/lite_local.rs` are unchanged.
- No new dependencies: the workspace has no `sha2`, `sqlite_vec`, or ANN crate, so content hashing is FNV-1a and vector search is a brute-force cosine scan.
- Errors follow the crate strategy; no panics.
- File sizes: the two previously oversized files are gone (`markdown.rs` and `local.rs` are now directories). `embed/index.rs` (~895 lines) and `write.rs` (~845) sit above the ~800-line target because each carries its full in-file test module; their implementations are ~390 and ~490 lines.

## Design Decisions

- **The per-fact layout is opt-in.** The single-document layout stays the default so every existing caller and test is byte-for-byte unaffected; `LocalCodingMemoryBuilder::fact_files(true)` switches the authoritative representation.
- **A hand-edited index wins.** When both the index documents and the fact files changed, the index is imported. Editing `MEMORY.md`/`USER.md` is an explicit human action, and the alternative (silently discarding it) would lose work.
- **Two index documents, one format.** `MEMORY.md` and `USER.md` use the same `MarkdownMemory` format with only the title and the level filter differing, so one parser/renderer serves both.
- **Embedding search is a cache-backed brute-force index.** The vector index is in memory per handle and the cache is the only durable part; with local corpora this is exact, and it keeps vector search working without a native extension. Python's `sqlite_vec`/`chunks_vec` ANN table is deferred.
- **The cache key is `(model name, content hash)`.** Python adds a provider fingerprint (`id:model:base_url:sha256(api_key)[:16]`); the `Embedding` trait exposes only `model_name`, so a provider change under the same model name reuses vectors.
- **Retry exhaustion is an error.** Python logs a warning and writes with the last-writer-wins state; this crate returns `Error::Memory`, because a silent overwrite is a data-loss path.
- **Sessions are their own recall surface.** `search_sessions` is separate from fact search and never filters through `MemoryType::Coding`; a synthesized record kind would have made the two indexes indistinguishable.
- **Rollups are derived documents.** `render_rollup` is deterministic and `write_rollup` is idempotent, so a rerun cannot churn files; there is no background scheduler.
- **Sync is caller-driven.** The trigger enum, `sync_now`, and the deterministic scheduler give the harness explicit control; the one background task is opt-in, cancellation-aware, and per handle.
- **No guard across an embedding call.** Mutations persist and drop the state guard before re-chunking, so a slow model can never block readers of the document.

## Routing Table

- `./mod.rs` → `CodingMemory`, `CodingFact`, `CodingFactType`, module declarations, module docs
- `./fact.rs` → fact model, record encoding/decoding, validation
- `./index.rs` → `render_index`, `citation`, grouping/truncation
- `./markdown/` → `MarkdownMemory`, `FrontMatter`, parse/render round trip, fact entries (`markdown/CONTEXT.md`)
- `./embed/` → chunking, content-hash cache, chunked vector index, cosine/blend helpers (`embed/CONTEXT.md`)
- `./facts/` → per-fact documents, slugs/file names, `MEMORY.md`/`USER.md` level split (`facts/CONTEXT.md`)
- `./session/` → session records, JSONL store, keyword scoring (`session/CONTEXT.md`)
- `./sync.rs` → `SyncTrigger`, `SyncReport`, `IntervalScheduler`
- `./daily.rs` → `DailyRollup`, `render_rollup`, `write_rollup`, `select_rollup_inputs`
- `./write.rs` → `WriteMode`, `WritePolicy`, `WritePlan`, `WriteOutcome`, `MemUpdateChecker`
- `./local/` → `LocalCodingMemory`, builder, layout/state/store/search, facts mode, sync mode, session mode, daily mode, write mode (`local/mod.rs` documents the store)
- `./lock.rs` → `FileLock`, `flock` semantics, non-Unix `Unsupported`
- `./watch.rs` → `MemoryWatcher`, `WatchEvent`, `WatchKind`, debounce/coalescing
- `./fts.rs` → `FtsIndex`, `FtsHit`, schema, safe MATCH expression (feature `sqlite`)

## Test Strategy

- Unit tests live beside each module; per-module counts (default build): `markdown` 25, `local` 47, `embed` 45, `facts` 51, `session` 36, `sync` 11, `daily` 14, `write` 17, plus 36 across `fact`/`index`/`lock`/`watch` — 282 in the `lite` subtree (359 with `--features sqlite`).
- `markdown/`: parse/render idempotence, front-matter ordering/raw lines, section order, unknown attributes and raw blocks, body boundary normalization, heading escaping, tolerant hostile input, `with_title`.
- `embed/`: chunk boundaries and overlap, `estimate_tokens`, `content_hash` stability, cache get/put/clear/len, the sqlite round trip, cosine/similarity/blend edge cases, and a counting `Embedding` wrapper proving a second `reindex` makes zero `embed_documents` calls and that misses are one batch.
- `facts/`: slug/file-name determinism, per-fact document round trip and hostile-input tolerance, `to_fact` validation, `FactFiles` write/read/read_all/remove/stamp, non-fact Markdown skipping, scope rebinding, and the three round trips (`facts → files → read_all`, `render_level → parse_level`, and a byte-for-byte hand-edit re-import).
- `session/`: `parse_line`/`to_json_line` round trip, malformed-line tolerance, timestamp string and epoch decoding, id sanitization, `path_for` layout, append accumulation, recursive `files`, tolerant `load`, `chunks` numbering, keyword scoring.
- `sync.rs`/`daily.rs`: scheduler due/not-due determinism from explicit `Instant`s, labels and report constructors; rollup shape, filters, ordering, truncation, and idempotent writes.
- `write.rs`: serde/`FromStr` for every mode, policy validation, `backoff_for` capping, `plan` for every mode with and without an existing fact, and `run` first-attempt/retry/exhaustion/error-propagation behaviour.
- `local/`: the original 13 store tests, plus the moved extras, plus the new suites — semantic vs keyword ordering, configured vs unconfigured `Semantic`, `RecallHit` components, cache hits across a refresh and a reopen, fact-file export/import, per-fact edits, the `USER.md` precedence rule, delete propagation, `sync_now` idempotency, trigger gating, the interval-task lifecycle, session recall in all three modes, rollup idempotency, and every write mode including retry exhaustion.
- Integration binaries under `tests/`: `lite.rs` (10 tests, engine-backed `CodingMemory`) and `lite_local.rs` (8 tests, default `LocalCodingMemory`) are unchanged and must stay green.
- Offline and deterministic: `tempfile` directories, `HashEmbedding` plus counting wrappers, explicit `Instant`s, no network, no credentials, bounded awaits (≲5 s in `watch.rs`).
- Gates (all must be green): `cargo fmt -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --offline` (971 tests), the same with `--all-features` (1048), `cargo clippy -p openjiuwen-memory --all-targets --all-features --offline -- -D warnings`, and `cargo doc -p openjiuwen-memory --no-deps` with and without the feature.
- Module-level (`//!`) doc links in this directory must be absolute (`crate::lite::<module>::<Item>`): `//!` docs resolve in the parent module's scope.

## Deferred (Python `core.memory.lite` features not ported)

- `sqlite_vec`/native ANN search and Python's `files`/`chunks`/`chunks_fts`/`chunks_vec`/`meta` schema: vectors live in memory and the content-hash cache is the only durable part, so recall is an exact brute-force cosine scan. The workspace has no `sqlite_vec` extension or ANN dependency and adding one would need a native build.
- The provider fingerprint (`provider_key` = `id:model:base_url:sha256(api_key)[:16]`) and the `meta`-row config-change rebuild rule: the `Embedding` trait exposes only `model_name`, so the cache key is `(model, content hash)`.
- Python's LLM redundancy/conflict checker (`MemUpdateChecker` is a model prompt in Python) and its "skip the redundant write" decision: the Rust checker is deterministic (fact lookup plus bounded retry) and makes no model call.
- Python's `created_at` front-matter field (`CodingFact` has no creation timestamp) and its caller-supplied free-form fact paths (filenames here are derived as `<slug>-<hash>.md`).
- Chunk-level trigram FTS over documents: the FTS5 table is fact-keyed and `unicode61`-tokenized; session and rollup text is not in FTS (sessions are scanned keyword-wise, rollups are not searched).
- Session transcripts written by the harness/CLI, watcher-driven session sync, session recency boosts, per-session scoping, and durable session vectors (the transient session index is cache-backed).
- An automatic session-lifecycle hook for `SyncTrigger::OnSessionStart`: the trigger is an explicit API and no caller in the workspace invokes it yet.
- Prompt-rail integration for `daily_memory` (Python's `MemoryRail` injects today's and yesterday's file) and agent-authored daily files outside the rendered rollup format.
- Interval-sync scheduling configuration (`sync.intervalMinutes`); the interval is caller-supplied and no task starts unless `start_interval_sync` is called.
- Python's optimistic snapshot retries over `WriteMode` (`WriteMode` itself is `create`/`append`/`skip` there; `merge`/`replace` are Rust extensions) and its last-writer-wins degradation, replaced here by a typed error.
- Platform parity for the lock: only the `#[cfg(unix)]` `flock` path exists.
