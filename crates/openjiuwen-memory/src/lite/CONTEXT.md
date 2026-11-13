# lite

## Intent

Lite coding memory: typed project/coding facts, served either from a `MemoryEngine` or from a local Markdown file with a derived index.
The crate has two lite paths.
(a) `CodingMemory` stores facts as ordinary `MemoryType::Coding` records on the shared engine, so they inherit scoping, dedupe/merge, recall, and every storage backend, and `render_index`/`citation` are pure string builders.
(b) `LocalCodingMemory` is the port of Python `memory/lite/MemoryIndexManager`: a human-editable `MEMORY.md` is the source of truth, an FTS5 database under `.memory/` is the derived index, a cross-process advisory lock serializes mutations, and a debounced watcher picks up external edits.
Python's per-fact `.md` files plus its `MEMORY.md` summary index are mapped onto ONE `MEMORY.md` that holds a section per fact type with the fact bodies inline.

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
    pub fn upsert_fact(&mut self, fact: &CodingFact) -> bool;   // true when it replaced an entry
    pub fn remove(&mut self, name: &str) -> bool;
    pub fn fact(&self, name: &str) -> Option<CodingFact>;
    pub fn facts(&self) -> Vec<CodingFact>;                     // document order
    pub fn is_empty(&self) -> bool;
}
```

### `local::{LocalCodingMemory, …}`

```rust
pub const MEMORY_FILE: &str = "MEMORY.md";
pub const STATE_DIR: &str = ".memory";
pub const INDEX_FILE: &str = "index.db";
pub const LOCK_FILE: &str = "index.lock";
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(200);

#[derive(Clone)]
pub struct LocalCodingMemory { /* inner: Arc<Inner>, fts: FtsIndex (cfg sqlite), watch */ }
impl LocalCodingMemory {
    pub async fn open(root: impl AsRef<Path>, config: CodingMemoryConfig) -> Result<Self>;
    pub async fn open_in_scope(root: impl AsRef<Path>, config: CodingMemoryConfig, scope: MemoryScope) -> Result<Self>;
    pub fn root(&self) -> &Path;
    pub fn memory_path(&self) -> &Path;
    pub fn lock_path(&self) -> &Path;
    pub fn config(&self) -> &CodingMemoryConfig;
    pub fn scope(&self) -> &MemoryScope;
    pub fn reload_count(&self) -> u64;
    pub async fn markdown(&self) -> Result<MarkdownMemory>;      // snapshot (reloads when stale)
    pub async fn write(&self, fact: CodingFact) -> Result<CodingFact>;   // upsert by name, updated_at = now
    pub async fn read(&self, name: &str) -> Result<Option<CodingFact>>;
    pub async fn edit(&self, name: &str, content: impl Into<String>) -> Result<CodingFact>;
    pub async fn delete(&self, name: &str) -> Result<bool>;
    pub async fn list(&self) -> Result<Vec<CodingFact>>;                 // updated_at desc, name asc
    pub async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    pub async fn render_index(&self) -> Result<String>;                  // compact summary, max_index_lines
    pub async fn reindex(&self) -> Result<usize>;                        // reload file + rebuild index
    #[cfg(feature = "sqlite")]
    pub async fn indexed_count(&self) -> Result<usize>;
    pub async fn start_watching(&self) -> Result<()>;
    pub async fn stop_watching(&self);
    pub async fn is_watching(&self) -> bool;
}
```

### `lock::FileLock`

```rust
#[derive(Debug)]
pub struct FileLock { /* path, locked: Flock<File> (cfg unix) */ }
impl FileLock {
    pub async fn acquire(path: impl AsRef<Path>) -> Result<Self>;              // waits for the holder
    pub async fn try_acquire(path: impl AsRef<Path>) -> Result<Option<Self>>;  // Ok(None) when held elsewhere
    pub fn path(&self) -> &Path;
}
```

### `watch::{MemoryWatcher, WatchEvent, WatchKind}`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WatchKind { Created, Modified, Removed, Renamed, Other }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchEvent { pub paths: Vec<PathBuf>, pub kinds: Vec<WatchKind> }   // deduplicated, sorted

pub struct MemoryWatcher { /* watcher, cancel, task, debounce */ }
impl MemoryWatcher {
    pub fn start(paths: impl IntoIterator<Item = PathBuf>, debounce: Duration, sender: UnboundedSender<WatchEvent>) -> Result<Self>;
    pub async fn stop(&mut self);
    pub fn is_running(&self) -> bool;
}
```

### `fts::{FtsIndex, FtsHit}` (feature `sqlite`)

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FtsHit { pub name: String, pub bm25: f64, pub score: f32, pub snippet: String }

#[derive(Debug, Clone)]
pub struct FtsIndex { /* pool: SqlitePool */ }
impl FtsIndex {
    pub async fn open(url: &str) -> Result<Self>;
    pub async fn open_path(path: &Path) -> Result<Self>;
    pub async fn from_pool(pool: SqlitePool) -> Result<Self>;
    pub async fn in_memory() -> Result<Self>;
    pub fn pool(&self) -> &SqlitePool;
    pub async fn upsert(&self, fact: &CodingFact) -> Result<()>;
    pub async fn remove(&self, name: &str) -> Result<bool>;
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<FtsHit>>;
    pub async fn rebuild<'a>(&self, facts: impl IntoIterator<Item = &'a CodingFact>) -> Result<usize>;
    pub async fn count(&self) -> Result<usize>;
    pub async fn clear(&self) -> Result<()>;
}
```

## On-disk layout and consistency model

```text
<root>/
  MEMORY.md            the source of truth (fact sections + bodies)
  .memory/index.db     derived SQLite FTS5 index (feature `sqlite`)
  .memory/index.lock   cross-process advisory lock file
```

- `<root>/MEMORY.md` is authoritative; `<root>/.memory/index.db` is derived and may always be rebuilt from it (`LocalCodingMemory::reindex`).
- `<root>/.memory/index.lock` is the advisory `flock` file (`#[cfg(unix)]`); on other platforms `FileLock::acquire`/`try_acquire` return `Error::Unsupported` naming the platform and create nothing.
- `open`/`open_in_scope` create the root and `.memory/`, write an empty `MEMORY.md` when it is missing, take the file lock, load the document, then rebuild the index.
- A mutation takes the file lock, reloads the Markdown file when its stamp (modification time plus length) changed, applies the change, writes atomically (temporary file plus rename), and then updates the index.
- A reader applies the same stamp check, so a fact committed by another process is visible without an explicit refresh.
- `start_watching` reloads (and reindexes) on every debounced `.md` change under the root, ignoring the `.memory/` directory; calling it again replaces the previous watcher.
- `reload_count` is the observable counter of document reloads, and it is what the watcher test observes.
- `render_index` is the compact prompt-facing summary (`- [Project] name — description`, `max_index_lines` cap), not the on-disk document.

## Markdown document format

- The front matter is a `---`-fenced block of `key: value` lines; a fresh document carries `schema: openjiuwen-memory-lite` and `version: 1`.
- Front matter keys keep document order, an existing key keeps its position on insert, and lines that are not a `key: value` pair are preserved verbatim in a raw tail.
- The title is the first `# …` line at the top of the document (`# Memory` in a fresh document); a document without one renders no title line.
- Fact sections are `## <Type>` in canonical order (user, feedback, project, reference); an empty section renders nothing.
- Each fact is a `### <name>` entry followed by a bullet attribute block and then the body.
- The interpreted attributes are `- description:`, `- tags: ["a","b"]` (a comma-separated value is also accepted on parse), `- path:`, and `- updated_at:` (RFC 3339).
- Unknown entry attributes are preserved across an edit, and a replaced fact keeps them.
- A missing or unparsable `updated_at` decodes to `DateTime<Utc>::default` (the Unix epoch), never "now".
- Input is normalized to LF; `render` emits LF and exactly one newline at the end, with one blank line between blocks.
- A body line that would look like a heading is escaped with a backslash on render (`\## note`) and unescaped on parse.
- Body boundaries are normalized: leading/trailing blank lines of a body are dropped, and the blank lines between heading, attributes, and body are canonical.
- `description` and `path` are single-line by construction (embedded newlines are folded to spaces on render).
- The only parse error is an unterminated front-matter block whose content holds `:`-bearing or attribute lines, which is `Error::Validation`; every other input parses tolerantly (unknown sections, loose text, hostile Markdown).
- `render` is the shape `parse` round-trips byte for byte, so `parse(render(doc)) == doc`.

## FTS5 index

- Schema: `CREATE VIRTUAL TABLE IF NOT EXISTS coding_facts_fts USING fts5(name UNINDEXED, content, description, tags, tokenize='unicode61')`, applied idempotently by every constructor.
- `tags` is stored space-joined (`["build", "ci"]` → `build ci`), so each tag is an individual FTS term.
- `upsert` is name-keyed: delete plus insert in one transaction, so a reader never sees the row missing.
- `rebuild` clears and reinserts in one transaction and returns the indexed row count.
- Results are ordered by `bm25()` ascending (more negative is better), with a `snippet()` excerpt per hit.
- The MATCH expression is safe by construction: the query is tokenized with `\w+`, at most 10 tokens with an alphanumeric character are kept, each is double-quoted with `"` doubled, and the tokens are joined with `OR` and bound as a statement parameter.
- A query with no usable token returns no hits without querying the database, so FTS5 syntax in user input is inert.
- The `name` column is UNINDEXED, so name-only matches come from the store's deterministic keyword scan, not from `MATCH`.

## Search scoring

`LocalCodingMemory::search` merges two signals per fact.
- The keyword score is the sum over query tokens of the strongest field the token matches (name 3, description or tags 2, body 1), divided by three times the token count, capped at `1.0`.
- An FTS hit contributes `1.0 - FtsHit::score` (the FTS score is higher for weaker matches); the store queries at most `top_k * 4` rows.
- A fact keeps the maximum of its two signals, and the merged score is clamped to `0.0..=1.0`.
- Hits are filtered by `threshold`, ordered by score descending then name ascending, and truncated to `top_k` (raised to at least 1).
- Only `top_k`, `threshold`, `kinds`, and `mode` are honoured: `kinds` empty means every kind and a list without `MemoryType::Coding` yields no hit.
- `SearchMode::Semantic` returns `Error::Unsupported` (the local store holds no embeddings), a blank query is `Error::Validation`, and `token_budget` is ignored.
- The returned `RecallHit` reports `similarity = 0.0`, `keyword = score`, `importance = 0.5`, and `recency = 0.5^(age / 30 days)` clamped to `0.0..=1.0`; ranking uses the merged keyword/FTS score, not a blend with those components.

## Record Encoding

- `kind = MemoryType::Coding`; `content` = fact Markdown body; `updated_at` mirrors `CodingFact::updated_at`.
- metadata keys: `fact_type` (string), `name`, `description`, `tags` (array of strings, always present), `path` (only when set), `coding_fact` = `true`.
- `CodingFact::from_record` accepts only records whose kind is `MemoryType::Coding`, whose `coding_fact` metadata is exactly `true`, and whose metadata `name` is a non-blank string; decoding is otherwise tolerant (unknown fact type falls back to `project`, non-string tags are dropped, null/missing path is `None`).
- `id` is a normal 24-hex memory id; lookup by name lists scoped coding records and compares metadata `name` (case-sensitive).
- `write` upserts by name: an existing fact with the same name in the scope is updated with its id preserved, and a cleared path is stored as JSON `null` so the merged metadata really drops the old path.
- `write` validates the name (non-empty, at most 128 characters, no `/` or `\`) and resolves the fact type against `config.fact_types`; a type absent from that list is replaced by `config.default_fact_type` with a debug log and never fails the write.
- `edit` updates the body and `updated_at` and returns the updated fact; a missing fact is `Error::NotFound`.
- `read`/`delete` return `None`/`false` for a missing fact.
- `list` orders by `updated_at` descending, then name ascending; `search` forces `SearchOptions::kinds` to `[MemoryType::Coding]` and passes every other knob through to the engine.

## Index Rendering

`index::render_index` (reachable as `CodingMemory::render_index` and `LocalCodingMemory::render_index`) is a pure string builder; nothing is written to disk.
- The first line is always `# Memory`.
- One bullet per fact is formatted exactly `- [<Type>] <name> — <description>`, with ` (`<path>`)` appended when the fact has a path.
- Facts are grouped in the fixed declaration order user, feedback, project, reference and sorted name-ascending inside each group.
- There is no blank line between groups, and the result always ends with a trailing newline.
- `max_lines` counts the title line, so `max_lines <= 1` yields the title alone.
- An empty scope yields exactly `"# Memory\n"`.
- The cap is `CodingMemoryConfig::max_index_lines` (default 200).

```text
# Memory
- [User] gamma — the user
- [Project] build — how to build (`docs/BUILD.md`)
```

`index::citation` (reachable as `CodingMemory::citation(&RecallHit)`) returns `<path>#L1-L<n>` when the record has `path` metadata, where `n` is the content's newline count + 1; otherwise it returns `coding:<name>`, falling back to `coding:<id>` when the metadata name is missing or blank.

## Concurrency and errors

- Lock order is file lock before state guard; the watcher task only takes the state guard, so the two can never deadlock.
- Mutations are additionally serialized by a per-store async mutex, which keeps concurrent tasks from occupying blocking threads on the file lock.
- No blocking call runs on the async reactor: `flock` (and the lock-file open) runs in `spawn_blocking`.
- Filesystem work uses `tokio::fs`; the watcher's debounce task lives on the caller's tokio runtime.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for bad input (invalid fact, blank query, blank edit body, invalid scope, bad SQLite URL/limit), `NotFound` for a missing fact on `edit`, `Unsupported` for semantic search and non-Unix locking, and `External` (with the failing path as context) for filesystem, lock, and index failures.
- Nothing panics on user input: no `unwrap`/`expect` outside tests.

## Constraints

- `CodingMemory` uses only the public `MemoryEngine` API; no direct store access, no separate persistence.
- Coding facts are returned by `LongTermMemory::search` only when `MemoryType::Coding` is in the searched kinds (it is in the default `searchable_kinds`).
- `CodingMemory::render_index`/`citation` write nothing; `LocalCodingMemory` owns the file and is the only writer of `MEMORY.md` and `.memory/`.
- The derived index is never the source of truth: it can be dropped and rebuilt from `MEMORY.md` at any time.
- Default builds work without the `sqlite` feature: `LocalCodingMemory` then reloads and keyword-searches the Markdown document with no FTS index.
- Errors follow the crate strategy; no panics.
- `markdown.rs` (~1130 lines) and `local.rs` (~1130 lines) sit slightly above the ~1000-line baseline because each carries its own full unit-test module in-file; their implementation halves are ~600 and ~780 lines, so they are kept whole rather than split.

## Routing Table

- `./mod.rs` → `CodingMemory`, `CodingFact`, `CodingFactType`, re-exports, module docs
- `./fact.rs` → fact model, record encoding/decoding, validation
- `./index.rs` → `render_index`, `citation`, grouping/truncation
- `./markdown.rs` → `MarkdownMemory`, `FrontMatter`, parse/render round trip, fact entries
- `./local.rs` → `LocalCodingMemory`, on-disk layout constants, consistency model, keyword scoring
- `./lock.rs` → `FileLock`, `flock` semantics, non-Unix `Unsupported`
- `./watch.rs` → `MemoryWatcher`, `WatchEvent`, `WatchKind`, debounce/coalescing
- `./fts.rs` → `FtsIndex`, `FtsHit`, schema, safe MATCH expression (feature `sqlite`)

## Test Strategy

- Unit tests in `fact.rs` and `index.rs`: fact↔record round-trip and tolerant decoding, name validation, index grouping/order/truncation/paths, citation forms.
- `markdown.rs`: 24 unit tests — parse/render idempotence, front-matter ordering/raw lines, section order, unknown attributes and raw blocks, body boundary normalization, heading escaping, tolerant hostile input.
- `fts.rs`: 15 unit tests — schema/constructor variants, name-keyed upsert and removal, `bm25` ordering, snippet extraction, safe MATCH normalization (syntax inert, empty query), rebuild/count/clear.
- `lock.rs`: 5 unit tests, all `#[cfg(unix)]` in their entirety — acquire creates the file and parents, a held lock refuses a contender, drop releases it, distinct paths do not interfere.
- `watch.rs`: 8 tests (`#[cfg(all(test, unix))]`) — markdown events emitted, non-Markdown and `.memory/` edits ignored, burst coalescing/deduplication, clean stop, `is_running`.
- `local.rs`: 13 tests — open creates document/state dir, write/read/edit/delete round trip, typed errors for missing facts and blank input, keyword search across name/body/description/tag, blank-query/semantic rejection, summary index shape, cross-instance visibility, external edit reload, reindex, document round trip through the store, concurrent writers serialized, watcher reload (`#[cfg(unix)]`), `Send + Sync` handle; the `sqlite` assertions are `#[cfg(feature = "sqlite")]`.
- `tests/lite.rs`: 10 integration tests over the engine-backed `CodingMemory` — write/read/edit/delete/list ordering/search kinds, upsert-by-name, scope isolation, index grouping and cap including the title, citation.
- Offline and deterministic: `tempfile` directories, real filesystem writes, no credentials, no network; watcher awaits are bounded (≲5 s `BOUND` in `watch.rs`, 50 ms polling capped at 100 iterations in `local.rs`).
- Gates (all must be green): `cargo fmt -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --offline --lib lite`, the same with `--features sqlite`, `cargo clippy -p openjiuwen-memory --offline --all-targets -- -D warnings` with and without the feature, and `cargo doc -p openjiuwen-memory --no-deps` with and without the feature.
- Module-level (`//!`) doc links in this directory must be absolute (`crate::lite::<module>::<Item>`): `//!` docs resolve in the parent module's scope, so a bare item link does not resolve.

## Deferred (Python `core.memory.lite` features not ported)

- `sqlite_vec`/cosine vector index and embeddings: local search is keyword plus FTS5 only, and `SearchMode::Semantic` is `Error::Unsupported`.
- LLM-driven conflict detection and retry (`MemUpdateChecker`, `WriteMode`/`WriteResult`, optimistic snapshot retries); the local store applies the last write under the lock.
- Python's per-fact `.md` files and the `USER.md` split; this crate keeps one `MEMORY.md`.
- `chunk_markdown` chunking with embeddings.
- The SQL `files`/`chunks`/`meta`/`embedding_cache` tables and provider-fingerprint rebuild tracking.
- Interval and on-session-start sync triggers.
- Session `.jsonl` indexing under `sessions/`.
- Markdown `daily_memory` rollups and `Dreaming` orchestration.
- Platform parity for the lock: only the `#[cfg(unix)]` `flock` path exists, so Windows/macOS specifics beyond it are not covered.
