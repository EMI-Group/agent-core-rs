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

`index::render_index` (reachable as `CodingMemory::render_index`) is a pure string builder; nothing is written to disk.
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

- Unit tests in `fact.rs` and `index.rs`: fact↔record round-trip and tolerant decoding, name validation, index grouping/order/truncation/paths, citation forms.
- `tests/lite.rs`: write/read/edit/delete/list ordering/search kinds, upsert-by-name, scope isolation, index grouping and cap including the title, citation.
