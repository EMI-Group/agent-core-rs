# lite

## Intent

Lite coding memory: typed project/coding facts stored as ordinary `MemoryType::Coding` records on the same engine, plus a rendered Markdown index.
Ports the behavior the Python harness relies on (`memory/lite/`: coding-memory read/write/edit, frontmatter types, `MEMORY.md` index cap) without the Markdown source-of-truth, SQLite FTS5/`sqlite_vec` index, watchers, or cross-process rebuild machinery.

## API Surface

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub fn citation(&self, hit: &RecallHit) -> String;                          // "path#L1-L5" or "coding:<name>"
}
```

## Record Encoding

- `kind = MemoryType::Coding`; `content` = fact Markdown body.
- metadata keys: `fact_type` (string), `name`, `description`, `tags` (array of strings), `path` (when set), `coding_fact` = `true`.
- `id` is a normal 24-hex memory id; lookup by name lists scoped coding records and compares metadata `name` (case-sensitive). `write` upserts by name: an existing fact with the same name is updated (id preserved) rather than duplicated.
- `write` validates the name (non-empty, ≤ 128 chars, no path separators) and fact type against `config.fact_types` (falling back to `default_fact_type` when the configured list is empty/does not contain the type — never error on type, so harness writes never fail).
- `edit` updates the body and `updated_at`; missing fact is `Error::NotFound`.
- `read`/`delete` return `None`/`false` for a missing fact.

## Index Rendering

`render_index` produces:
```text
# Memory

- [Project] name — description (`path`)
...
```
one line per fact, grouped by `fact_type` (user, feedback, project, reference), truncated to `max_index_lines` (default 200); always emits the `# Memory` title. Empty scope yields `"# Memory\n"`.
`citation` returns `path#L1-L<line_count>` when `path` metadata exists, else `coding:<name>`; line count is the content's newline count + 1.

## Constraints

- Only the public `MemoryEngine` API; no direct store access, no separate persistence.
- Coding facts are returned by `LongTermMemory::search` only when `MemoryType::Coding` is in the searched kinds (it is in the default `searchable_kinds`).
- No filesystem writes: `render_index`/`citation` return strings; the harness decides where to materialize `MEMORY.md`.
- Errors follow the crate strategy; no panics.

## Routing Table

- `./mod.rs` → `CodingMemory`, `CodingFact`, `CodingFactType`, re-exports
- `./fact.rs` → fact model, record encoding/decoding, validation
- `./index.rs` → `render_index`, `citation`, grouping/truncation

## Test Strategy

- Unit tests: fact↔record round-trip, name validation, upsert-by-name, type fallback, index rendering/truncation, citation forms.
- `tests/lite.rs`: write/read/edit/delete/list/search, scope isolation, engine dedupe interaction, index cap.
