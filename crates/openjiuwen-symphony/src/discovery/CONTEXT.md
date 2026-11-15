# openjiuwen-symphony / discovery

## Intent

SKILL.md discovery and the lexical skill catalog, mapping to Python `openjiuwen.symphony.discovery`.
A scan turns a directory of skill folders into ordinary `CapabilityDescriptor`s, so discovered skills flow straight into the registry, graph, and planner without a plugin protocol.
The module ports both Python scan modes (immediate-child and bounded recursive), reports per-skill problems as diagnostics instead of failing the scan, and derives a stable, order-independent inventory fingerprint.
Skill search reuses the crate's own BM25 `LexicalIndex` rather than duplicating retrieval semantics.
The model-facing surface is completed by a deterministic prompt snapshot, the visible skill-name list, an incremental skill-notice cursor with Python's reminder wording, a session-level notice layer, and the `skill_index` tool (list/search/read over a virtual category tree).
Everything here is offline, deterministic, and filesystem-only.
The `skill_index` tool is Python's single read-only catalog tool: `list` a category's subcategories and skills, `search` names, aliases, descriptions and bodies, `read` a skill's metadata card; it is a virtual tree, never a filesystem tool.

## API Surface

Front matter is an optional YAML mapping between a first line holding exactly `---` (CRLF tolerated) and a closing `---` or `...`, parsed with `serde_yaml_ng`.

```rust
pub const DELIMITER: &str = "---";
pub const ALTERNATE_DELIMITER: &str = "...";
pub enum FrontMatterStatus { Ok, Missing, Unterminated }     // Missing default
pub struct FrontMatterOutcome { pub status: FrontMatterStatus, pub metadata: Option<Value>, pub body: String, pub error: Option<String> }
impl FrontMatterOutcome {
    pub fn fences_ok(&self) -> bool;
    pub fn mapping(&self) -> Option<&Map<String, Value>>;
    pub fn get(&self, key: &str) -> Option<&Value>;
    pub fn get_str(&self, key: &str) -> Option<&str>;
    pub fn first_str(&self, keys: &[&str]) -> Option<&str>;
    pub fn get_bool(&self, key: &str) -> Option<bool>;
}
pub fn parse_front_matter(raw: &str) -> FrontMatterOutcome;
pub fn boolean_value(value: &Value) -> Option<bool>;
pub fn normalize_capability_type(raw: &str) -> String;
pub fn tag_values(value: &Value) -> Vec<String>;
pub fn io_values(value: &Value) -> Vec<CapabilityIo>;
```

A null, non-mapping, or malformed front-matter document yields `metadata = None` plus an `error` note instead of failing or panicking; `body` always holds the text after the closing fence.

```rust
pub struct SkillFolderScanner;
impl SkillFolderScanner {
    pub fn new(root: impl Into<PathBuf>) -> Self;
    pub fn with_config(root: impl Into<PathBuf>, config: SkillScanConfig) -> Self;
    pub fn root(&self) -> &Path;
    pub fn config(&self) -> &SkillScanConfig;
    pub async fn scan(&self) -> Result<Vec<CapabilityDescriptor>>;
    pub async fn scan_skills(&self) -> Result<Vec<ScannedSkill>>;
    pub async fn scan_result(&self) -> Result<SkillScanResult>;
    pub async fn discover(&self) -> Result<SkillDiscovery>;
}
pub struct SkillScanConfig { pub max_depth: Option<usize>, pub source: String, pub disabled_skills: BTreeSet<String>, pub recursive: bool }
impl SkillScanConfig {
    pub fn recursive(max_depth: Option<usize>) -> Self;
    pub fn with_source(self, source: impl Into<String>) -> Self;
    pub fn with_max_depth(self, max_depth: usize) -> Self;
    pub fn with_disabled_skills(self, skills: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn effective_max_depth(&self) -> usize;              // max_depth or DEFAULT_MAX_DEPTH (4)
    pub fn is_disabled(&self, folder_name: &str, descriptor: &CapabilityDescriptor) -> bool;
}
pub enum DiagnosticSeverity { Warning, Error }               // Warning default
pub struct SkillDiagnostic { pub code: String, pub severity: DiagnosticSeverity, pub path: String, pub message: String }
pub struct ScannedSkill { pub descriptor: CapabilityDescriptor, pub body: String }
pub struct SkillDiscovery { pub skills: Vec<ScannedSkill>, pub diagnostics: Vec<SkillDiagnostic>, pub source: String }
pub struct SkillScanResult { pub capabilities: Vec<CapabilityDescriptor>, pub diagnostics: Vec<SkillDiagnostic>, pub source: String }
impl SkillScanResult { pub fn error_codes(&self) -> Vec<&str>; pub fn capability_ids(&self) -> Vec<&str>; }
```

The module also exports the scan constants (`SKILL_CAPABILITY_TYPE` = `skill`, `SKILL_FILE_NAMES` = `["SKILL.md", "skill.md", "Skill.md"]`, `RECURSIVE_SKILL_FILE` = `SKILL.md`, `DEFAULT_MAX_DEPTH` = 4, `MAX_DESCRIPTION_CHARS` = 500), the `DIAGNOSTIC_*` codes (`missing_front_matter`, `unterminated_front_matter`, `malformed_front_matter`, `unreadable_file`, `unreadable_directory`, `symlink_skipped`, `duplicate_skill`, `invalid_skill`), and `DEFAULT_SOURCE` = `local`.

The default scan mode ports `scan_skill_directories`: every immediate child directory is probed for `SKILL.md`/`skill.md`/`Skill.md` in that priority order, with no recursion below a skill folder, and names starting with `.` or `_` are skipped.

Recursive mode walks to `effective_max_depth()` for an exact `SKILL.md` match and does not descend into a folder that holds one; symlinks are skipped with a diagnostic.

A scan never fails for one bad skill: every problem becomes a `SkillDiagnostic` and the scan continues.

Warnings keep the skill and explain the gap: missing front matter (descriptor derived from the body), malformed YAML (metadata dropped), a skipped symlink, and a duplicate capability id (first skill kept).

Errors skip the affected skill: unreadable file or directory, non-UTF-8 content, unterminated front matter, and a parsed skill without a usable identity.

Only an unusable root is a scan error: `Error::Validation` naming `root`.

A descriptor's capability id is the front-matter `capability_id` or `id`, otherwise a hyphenated `normalize_name_key` slug of the folder name; its name is `name`, else the folder name, else the id; `capability_type` is normalized (`skills` → `skill`, `subagent`/`sub-agent` → `agent`).

A missing description falls back to the first body paragraph, whitespace-collapsed and capped at `MAX_DESCRIPTION_CHARS` (500) with `…`.

The descriptor metadata carries `entrypoint` (the `SKILL.md` path relative to the root with `/` separators), `version`, `author`, `source_root`, and optional `aliases`; its `content_hash` is the SHA-256 hex digest of the raw `SKILL.md` bytes.

Unknown front-matter keys never reach the descriptor or its retrieval text.

```rust
pub struct SkillRecord { pub worker_id: String, pub name: String, pub description: String, pub skill_file: String, pub source: String, pub version: String, pub author: String, pub content_hash: String, pub source_root: String, pub aliases: Vec<String> }
pub struct SkillInventory { pub items: Vec<SkillRecord>, pub fingerprint: String, pub count: usize }
impl SkillInventory { pub fn new(items: Vec<SkillRecord>) -> Self; pub fn is_empty(&self) -> bool; pub fn get(&self, worker_id: &str) -> Option<&SkillRecord>; pub fn worker_ids(&self) -> Vec<&str>; }
pub fn sort_records(records: &mut [SkillRecord]);
pub fn inventory_fingerprint(records: &[SkillRecord]) -> String;
```

`SkillRecord`s reuse the descriptor fields, and the inventory fingerprint is the SHA-256 hex digest of the canonical JSON array of records sorted by `(worker_id, skill_file)` with object keys sorted, so it is independent of scan order and vector order.

```rust
pub struct SkillFs;
impl SkillFs {
    pub async fn open(root: impl AsRef<Path>) -> Result<Self>;
    pub async fn open_with(config: SkillScanConfig, root: impl AsRef<Path>) -> Result<Self>;
    pub fn root(&self) -> &Path;
    pub fn config(&self) -> &SkillScanConfig;
    pub fn inventory(&self) -> &SkillInventory;
    pub fn records(&self) -> &[SkillRecord];
    pub fn diagnostics(&self) -> &[SkillDiagnostic];
    pub fn read_body(&self, worker_id: &str) -> Option<&str>;
    pub fn selection_cards(&self) -> Vec<SelectionCard>;
    pub fn visible_skill_names(&self) -> Vec<String>;
    pub fn prompt_snapshot(&self, options: &PromptSnapshotOptions) -> PromptSnapshot;
    pub fn notice_state(&self) -> SkillNoticeState;
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SkillHit>>;   // 0 = unlimited
}
pub struct SkillHit { pub worker_id: String, pub name: String, pub description: String, pub score: f32, pub matched_terms: Vec<String>, pub excerpt: String }
pub struct SelectionCard { pub worker_id: String, pub name: String, pub description: String, pub tags: Vec<String>, pub version: String }
pub const EXCERPT_WIDTH: usize = 160;
pub const EXCERPT_LEAD: usize = 60;
```

`SkillFs` indexes every descriptor with `semantic_content = body` through `LexicalIndex`, ranks hits by BM25 descending and ties on `worker_id`, and treats `top_k == 0` as unlimited.

An empty or all-punctuation query is a filter-only request returning every record in `worker_id` order with score `0.0` and an empty excerpt.

An excerpt is the whitespace-collapsed window of `EXCERPT_WIDTH` characters (`EXCERPT_LEAD` kept before the term) around the earliest matched term of whichever of the description and body matches more terms, with `…` on a truncated side.

### Prompt snapshot and visible skills

```rust
pub const DEFAULT_PROMPT_MAX_CHARS: usize = 12_000;   // DiscoverySettings.max_output_chars default
pub const PROMPT_HEADER: &str = "# Skills";
pub const PROMPT_EMPTY_TEXT: &str = "# Skills\n\nNo skills are available for this task.";
pub const UNCATEGORIZED_CATEGORY: &str = "Uncategorized";
pub const PROMPT_TRUNCATION_MARKER: &str = "[truncated]";

pub struct PromptSnapshotOptions { pub max_chars: usize, pub include_categories: bool, pub worker_id: Option<String> }
impl Default for PromptSnapshotOptions;              // 12_000, false, None
impl PromptSnapshotOptions {
    pub fn new(max_chars: usize) -> Self;
    pub fn with_categories(self, include_categories: bool) -> Self;
    pub fn with_worker_id(self, worker_id: impl Into<String>) -> Self;
}

pub struct PromptSnapshot { pub rendered: String, pub included: Vec<String>, pub omitted: Vec<String>, pub truncated: bool, pub fingerprint: String }
impl Default for PromptSnapshot;                      // PROMPT_EMPTY_TEXT, empty lists, fingerprint of []
```

`visible_skill_names` returns the enabled worker ids in Python's prompt order `(worker_id.casefold(), worker_id)`; skills dropped by `SkillScanConfig::disabled_skills` never reach the catalog, so they are absent.

The rendered text is Python's markdown skill section:

- header line `# Skills`, then a blank line before the body;
- one line per skill: `- <worker_id>: <whitespace-collapsed description>` (Python's exact `prompt_snapshot` line);
- `include_categories` groups lines under `## <category>` headings separated by blank lines, with the descriptor's `classification` (front-matter `category`) as the label and `Uncategorized` when blank; categories sort by `(casefold, label)` and skills within a category by prompt order;
- an empty (or fully filtered) scope renders `PROMPT_EMPTY_TEXT`, never an empty string;
- when `max_chars` leaves part of the scope out, the last fitting line may be shortened on a `char` boundary to its first characters plus `...`, and the snapshot ends with a blank line and `[truncated]`. Lines are never split inside a code point, and a category heading is only emitted when at least one of its skills fits.

`included` is render order; `omitted` is the prompt order of in-scope worker ids the budget left out (worker ids excluded by `worker_id` are out of scope, not omitted). `fingerprint` is `inventory_fingerprint` over the included records, so equal included content yields the same digest regardless of render order or category layout. `max_chars == 0` omits every skill but still renders the header and `[truncated]`.

`notice_state` seeds a [`SkillNoticeState`](notice.rs) with the current inventory as baseline; callers may construct it directly from `SkillFs::inventory()`.

### Incremental skill notices

```rust
pub const SKILL_INDEX_TOOL_NAME: &str = "skill_index";
pub const DEFAULT_NOTICE_MAX_CHARS: usize = 4_000;
pub const NOTICE_DESCRIPTION_CHARS: usize = 300;

pub struct SkillNoticeState { pub seen: BTreeMap<String, String>, pub max_chars: usize, pub discovery_tool_name: String }
impl Default for SkillNoticeState;                    // empty seen, 4_000, "skill_index"
impl SkillNoticeState {
    pub fn new(inventory: &SkillInventory) -> Self;   // baseline, consumes no delta
    pub fn with_max_chars(self, max_chars: usize) -> Self;
    pub fn with_discovery_tool_name(self, tool_name: impl Into<String>) -> Self;
    pub async fn consume(&mut self, inventory: &SkillInventory) -> SkillNotice;
}
pub struct SkillNotice { pub added: Vec<String>, pub removed: Vec<String>, pub changed: Vec<String>, pub reminder: Option<String> }
```

`SkillNoticeState` is Python's `initialize_incremental_skill_notice_state` / `_NoticeState` as plain, serde-persistable data (one cursor owned by the caller instead of the process-global per-session map).

`seen` maps each worker id to `inventory_fingerprint` of its record, so a description edit is detected even when a caller supplies no content hash.

`consume` diffs against `seen` and updates it to the new inventory, so a second call with the same inventory reports no changes. `added`, `removed` and `changed` all sort by `(worker_id.casefold(), worker_id)`.

`reminder` follows Python's `_render_reminder` literally: for additions it renders `<system-reminder>` / `The following skills are newly available for use with the Skill tool:` / blank line / `- <name>: <description>` rows sorted by `name.casefold()` / `</system-reminder>`, with whitespace collapsed, `<`/`>` escaped, and descriptions shortened to `NOTICE_DESCRIPTION_CHARS` (`textwrap.shorten`, `...` placeholder). When the block exceeds `max_chars` it becomes `<system-reminder>` / `<N> new skills are available. Use \`<tool_name>\` to search the current catalog.` / `</system-reminder>`. Python's reminder only announces additions, so `reminder` is `None` when nothing was added even if skills were removed or changed; those deltas stay in the notice fields.

```rust
pub struct DiscoverySettings { pub max_results: usize, pub max_output_chars: usize, pub candidate_budget_ratio: f64, pub context_window_tokens: usize, pub use_existing_index: bool }
impl DiscoverySettings {
    pub fn candidate_budget_tokens(&self) -> usize;          // context_window_tokens * clamp(ratio, 0, 1)
    pub fn from_value(value: Value) -> Self;                 // lenient: every unusable field keeps its default
}
pub const DEFAULT_MAX_RESULTS: usize = 10;
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 12_000;
pub const DEFAULT_CANDIDATE_BUDGET_RATIO: f64 = 0.01;
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 200_000;
```
### Notice session layer (`toolkit/`)
```rust
pub const MAX_SESSION_STATES: usize = 1_024;
pub const TOOL_ID_DOMAIN: &str = "openjiuwen.skill-index.v1";
pub struct SkillCard { pub name: Option<String>, pub description: Option<String> }
pub type SkillCards = BTreeMap<String, SkillCard>;
pub struct SkillCardSnapshot { pub worker_id: String, pub name: String, pub description: String }
pub fn tool_id(session_scope: &str, tool_name: &str) -> Result<String>;    // "<name>__<sha256 hex>"
pub fn normalize_cards(cards: &SkillCards) -> BTreeMap<String, SkillCardSnapshot>;
pub fn cards_from_inventory(inventory: &SkillInventory) -> SkillCards;
pub fn render_reminder(added: &[SkillCardSnapshot], max_chars: usize, tool_name: &str) -> Option<String>;
pub struct IncrementalNoticeRegistry;   // new / with_capacity / len / is_empty / capacity / clear / initialize / consume / tool_ids
pub struct IncrementalNoticeSession;    // new / tool_id / max_chars / seen / append(output, cards, output_budget)
```
`tool_id` is Python's domain-separated SHA-256 identity, so two scopes or two tool names never share a cursor.
`normalize_cards` skips a card with a blank worker id, falls back `name` → worker id and `description` → name, collapses whitespace, escapes `<`/`>`, and shortens a description to `NOTICE_DESCRIPTION_CHARS` on a word boundary.
`render_reminder` is Python's block for `(name, description)` rows sorted by name, with the compact `<N> new skills…` fallback when the full block exceeds `max_chars`.
The registry is the owned substitute for Python's process-global `_STATES` map: a `Mutex` over a `BTreeMap` plus a monotonic recency counter, bounded at `capacity` (default `MAX_SESSION_STATES`) and evicting the least-recently-used cursor. A poisoned lock is recovered, never panics.
`IncrementalNoticeSession::append` is Python's rollback protocol: a reminder that does not fit `output_budget` is not acknowledged, so the next call offers it again; a call with no additions still advances the cursor.
### The `skill_index` tool (`directory_toolkit/`)
```rust
// view: the virtual category tree (see view.rs)
pub struct SkillDirectoryView;   // from_catalog / from_records / normalize_path / node / node_paths / children /
                                 // skill_count / record / record_ids / record_path / entries / tree_entries /
                                 // searchable_entries / scoped_records / metadata_path / resolve_metadata_path /
                                 // category_from_path / skill_category / category_text / category_path
pub struct DirectoryEntry { pub path: String, pub kind: EntryKind, pub label: String, pub description: String, pub worker_id: String, pub depth: usize }
pub enum EntryKind { Dir, Skill }
pub const ROOT_PATH: &str = "/";   // label "ROOT", description "Installed Skills"
// render: pinned rows and budgeting (see render.rs)
pub struct Row { pub text: String, pub worker_id: String }
pub enum ListView { Names, Details, Tree }
pub fn list_row(entry: &DirectoryEntry, view: ListView, directory: &SkillDirectoryView) -> Row;
pub fn search_row(record: &SkillRecord, directory: &SkillDirectoryView, indexes: &[usize], snippet: Option<&str>) -> Row;
pub fn search_directory_row(entry: &DirectoryEntry, directory: &SkillDirectoryView, indexes: &[usize]) -> Row;
pub fn fit_rows(rows: &[Row], summary: &ResponseSummary, budget: Option<usize>, count_value: Option<usize>, show_shortened_marker: bool) -> FittedRows;
pub fn summary_line(summary: &ResponseSummary) -> String;   pub fn summary_line / bounded_summary_line / compact / append_note
pub fn metadata_card(record: &SkillRecord) -> String;       pub fn empty_message(operation: &str, category: Option<&str>) -> String;
pub fn sanitize_model_text(value: &str) -> String;          pub fn inline_text(value: &str) -> String;
// validate: the typed request and the validation pass (see validate.rs)
pub struct SkillIndexRequest;   // operation, category, skills, paths, query, view, recursive, directory_entry,
                                // directories_only, match, result, case_insensitive, fixed_strings, max_depth,
                                // read_mode, line_count, start_line, end_line, per_query_limit, pipeline,
                                // output_mode, max_output_chars, disable_output_truncation, limit
pub struct ValidatedRequest;    // the above, parsed into enums with validated scalars
pub struct RequestIssue { pub field: String, pub message: String }
pub fn validate(request: &SkillIndexRequest) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>>;
pub fn apply_simple_defaults(request: &SkillIndexRequest) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>>;
// ops: execution (see ops.rs)
pub fn execute_list(view: &SkillDirectoryView, paths: &[String], request: &ValidatedRequest) -> Result<Execution>;
pub fn execute_search(catalog: &SkillFs, view: &SkillDirectoryView, paths: &[String], request: &ValidatedRequest) -> Result<Execution>;
pub fn execute_read(view: &SkillDirectoryView, skills: &[String], request: &ValidatedRequest) -> Result<Execution>;
pub fn apply_pipeline(rows: &[Row], pipeline: &[PipelineStage], output_mode: OutputMode) -> Result<PipelineOutcome>;
// toolkit: the model-facing tool (see toolkit.rs)
pub struct SkillIndexConfig { pub session_scope: String, pub max_output_chars: usize, pub incremental_notice_max_chars: usize }
pub struct SkillIndexToolkit { /* catalog, view, config, tool_id, notice, cursors, observed paths */ }
impl SkillIndexToolkit {
    pub fn new(catalog: SkillFs, config: SkillIndexConfig) -> Result<Self>;
    pub fn tool_id(&self) -> &str;   pub fn catalog(&self) -> &SkillFs;   pub fn view(&self) -> &SkillDirectoryView;
    pub fn card(&self) -> SkillIndexToolCard;
    pub fn invoke(&mut self, arguments: &Value) -> Result<SkillIndexResult>;
    pub fn execute(&mut self, request: &SkillIndexRequest) -> Result<SkillIndexResult>;
    pub fn close(&mut self);   pub fn cursor_count(&self) -> usize;   pub fn reload(&mut self) -> Result<()>;
}
pub struct SkillIndexToolCard { pub id: String, pub name: String, pub description: String, pub input_schema: Value, pub parallel_safe: bool, pub stateless: bool }
pub struct SkillIndexResult { /* content, model_content, operation, rows, observed_skill_ids, truncated,
                                 truncation_reason, total_count, shown_count, result_complete, next_cursor,
                                 effective_max_output_chars, skillfs_layout, index_state, candidate_count,
                                 estimated_candidate_tokens, candidate_budget_tokens */ }
pub const MAX_CURSORS: usize = 32;   pub const CURSOR_MAX_CHARS: usize = 128;
```
Categories are derived from each skill's `CapabilityDescriptor::classification` — split on `>` (else `/`), trimmed, blank segments dropped, depth capped at 32 — because Python's retriever-node taxonomy does not exist in this crate.
Sibling segments that collide case-insensitively (or a label colliding with a skill id) get `~2`, `~3`, … in `(to_lowercase, exact)` order, so the tree is a pure function of the catalog.
`invoke` is the model-facing shape and accepts only `operation`, `category`, `query` and `cursor`; `execute` takes the full typed request.
Search reuses the crate's BM25 (`SkillFs::search`) for content mode and `LexicalIndex` for category hints; `match = name|path` uses `matcher::glob_match` (`*`, `?`) with `literal_match_rank` ordering.
Pagination cursors are deterministic: the token is the unpadded URL-safe base64 of the first 16 SHA-256 bytes over a canonical JSON state (operation, category, queries, offset, page size, inventory fingerprint), so it is ≤ 24 characters; the state lives in a 32-entry LRU on the toolkit.
A cursor is rejected when the token is unknown, the operation/category/query differ from the original request, or the catalog fingerprint changed.
Rendering is pinned: `fit_rows` adds rows while `header + blank line + body` fits the char budget, then pops rows while the recomputed summary still overflows, then falls back to a compacted first row, then appends `SHORTENED_MARKER`.
`output_mode = count` and the `limit`/`slice`/`filter`/`count` pipeline stages run over the rendered lines, after flattening multi-line rows.
The notice reminder is appended last and never lets the response exceed its budget.

## Constraints

- Offline and deterministic: no network, no wall-clock ordering; skill and record ordering comes from ids, Python's casefold prompt order, or BM25 scores.
- A single bad skill never aborts a scan; diagnostics are the failure report and only an unusable root is `Err`.
- Blocking filesystem work uses tokio's `fs`; no `unsafe`, no `println!`, no panics on user input.
- Every public item is documented; files stay around or under 1000 lines and split by responsibility.
- Descriptors produced here must round-trip through serde and remain valid `CapabilityDescriptor`s.
- Front-matter keys outside the documented set are dropped, never leaked into metadata or retrieval text.
- Prompt rendering never splits a code point and never renders an empty string; notice ordering is casefold-then-exact and clock-free.
- **No process-global mutable state.** Python's per-session notice map and the tool's cursor map become owned values (`IncrementalNoticeRegistry`, `SkillIndexToolkit` cursors), so two toolkits never share state and tests need no reset hook.
- **The `skill_index` tool never panics on bad arguments.** A rejected request is `Vec<RequestIssue>` (each naming its field, including `pipeline[i].field`) from the validation pass, or `Error::Validation` from the execution path; both are `tracing`-friendly and model-readable.
- **No regex engine exists offline.** Content search is BM25 plus a case-(in)sensitive substring/glob matcher (`matcher::{glob_match, literal_match_rank}`); `fixed_strings` is accepted for schema compatibility and never changes the matcher.
- **Every budget is counted in `chars().count()`**, never bytes, so CJK text truncates at the documented width.
- **`ops.rs` is pure**: it reads a `SkillDirectoryView`/`SkillFs` and returns rows; only `toolkit.rs` owns cursors, observed paths and the notice cursor.

## Tests

- `tests/discovery.rs` — scans, front matter, diagnostics, descriptors, inventory, catalog search, excerpts, settings.
- `tests/skill_prompt.rs` — visible-name ordering and disabled-skill exclusion, exact pinned snapshot text (plain, category sections, empty), multi-byte `max_chars` truncation with `included`/`omitted`/`truncated` and fingerprint bookkeeping, `worker_id` filtering, fingerprint stability/content sensitivity, added/removed/changed notices with the exact Python reminder, reminder fallback and custom tool name, deterministic ordering, idempotent second consume, serde persistence, and `SkillFs::notice_state`.
- `tests/discovery_toolkit_notice.rs` — the notice session layer: the tool-id digest and its validation, card normalization (fallbacks, 300-char word-boundary shortening, escaping, ordering), pinned reminder text and its budget fallback, registry baseline/one-shot delta/LRU eviction/capacity clamp/clear, session rollback then later delivery, serde round-trip, byte-identical reruns.
- `tests/discovery_toolkit.rs` — the tool: category-tree building (`~n` collisions, depth cap), path normalization, `children`/`skill_count`/`entries`/`tree_entries`/`searchable_entries`/`scoped_records`, metadata paths, category lookups (unknown and ambiguous), every pinned row/summary/message rendering, `fit_rows` truncation with multi-byte budgets, the full validation matrix, then end-to-end `invoke`/`execute` for list/search/read, multi-query interleaving, `result = matches` snippets, pagination without gaps or duplicates, cursor mismatch/stale/unknown errors and the 32-entry bound, pipelines with `output_mode = count`, output-budget shortening and `disable_output_truncation`, notice delivery, the pinned tool card, and byte-identical repeated calls.
- Unit tests beside `catalog.rs` pin the plain/category/empty renderings; beside `notice.rs` the reminder wording, the `textwrap.shorten` behavior and state round-tripping; beside `ops.rs` the pipeline and search-assembly helpers.

## Routing Table

- `./mod.rs` → module docs and re-exports of the whole discovery surface
- `./frontmatter.rs` → `parse_front_matter`, `FrontMatterStatus`/`FrontMatterOutcome`, tag/IO/boolean coercion, capability-type normalization
- `./scanner.rs` → `SkillFolderScanner`, `SkillScanConfig`, descriptor derivation, diagnostics, scan constants
- `./models.rs` → `SkillRecord`, `SkillInventory`, `sort_records`, inventory fingerprint, `DEFAULT_SOURCE`
- `./catalog.rs` → `SkillFs`, `SkillHit`, `SelectionCard`, BM25 search, excerpts, `visible_skill_names`, `prompt_snapshot`
- `./notice.rs` → `SkillNoticeState`, `SkillNotice`, incremental reminder rendering, notice constants
- `./toolkit.rs` → `IncrementalNoticeRegistry`, `IncrementalNoticeSession`, `SkillCard` normalization, `tool_id`, reminder rendering
- `./directory_toolkit/` → the model-facing `skill_index` tool; see [`directory_toolkit/mod.rs`](directory_toolkit/mod.rs)
  - `./directory_toolkit/view.rs` → `SkillDirectoryView`, `DirectoryEntry`/`DirectoryNode`, category paths and skill counts
  - `./directory_toolkit/render.rs` → pinned `list_row`/`search_row` rows, summary lines, `fit_rows`, `compact`, `metadata_card`
  - `./directory_toolkit/validate.rs` → `SkillIndexRequest`, the six enums, `validate`/`apply_simple_defaults`, limits and `RequestIssue`
  - `./directory_toolkit/ops.rs` → `execute_list`/`execute_search`/`execute_read`, `apply_pipeline`, search assembly and snippets
  - `./directory_toolkit/toolkit.rs` → `SkillIndexToolkit`, `SkillIndexConfig`, `SkillIndexToolCard`, `SkillIndexResult`, cursors
  - `./directory_toolkit/matcher.rs` → `case_fold`, `glob_match`, `literal_match_rank`
- `./settings.rs` → `DiscoverySettings`, token budget, settings defaults

## Design Decisions

- **The notice state is owned, not global.** Python keeps a process-wide `_STATES` `OrderedDict` keyed by a session-scoped tool id because its tool call path has no session object; the Rust port puts the same `Mutex`-guarded, LRU-bounded map in `IncrementalNoticeRegistry`, and `SkillIndexToolkit` holds its own `IncrementalNoticeSession`, so state travel with the value that owns it and tests need no reset hook.
- **Categories come from the front-matter category, not a taxonomy index.** Python projects a prebuilt `RetrieverNode` tree; this crate has no taxonomy artifact, so the virtual tree is built from each skill's `CapabilityDescriptor::classification` and is therefore a pure function of the catalog.
- **Cursors are deterministic and instance-local.** Python mints a random `token_urlsafe` and keeps a dict; here the token is a truncated SHA-256 of the canonical cursor state, so a repeated request yields the same token, and the state still lives in a 32-entry LRU because the token is a key, not a container.
- **One matcher, no regex.** Both the search modes and the `filter` pipeline stage use case-folded substring matching with `*`/`?` globbing, so the offline build has no regex dependency and the same query text behaves identically everywhere.
- **Search reuses the crate's own scorer.** Content mode delegates to `SkillFs`'s BM25 index (with the hit excerpt as the evidence snippet) and category hints use `LexicalIndex`, so discovery never grows a second scoring implementation.
- **The escape replaces the untrusted-metadata prefix.** Python's `sanitize_model_text` prefixes `[untrusted Skill metadata] ` when it detects injection phrasing; here `<`/`>` are escaped and control characters dropped instead, which neutralizes the same vectors without leaking the prefix into legitimate descriptions and snippets.
- **The summary header is the paged contract.** A paged result's `content` equals `model_content` (header included); an unpaged result's `content` is the row body only and the header travels in `model_content`, mirroring Python's `SkillDCICommandResult`/`.data["content"]` split.

## Known Issues

- `SkillIndexResult::skillfs_layout` is always `flat` and `index_state` always `flat`/`missing`: no taxonomy index is built, so Python's `tree`/`fresh`/`stale` states have no counterpart.
- The category-hint fallback is mostly unreachable end-to-end: a label appears in its skills' own searchable text (the category is part of the view, not of the descriptor retrieval text, but the label usually matches the description), so a label query normally becomes a content hit first. `ops::search_categories` is implemented, exported and unit-tested directly.
- Python's `DiscoverySettings.max_results` is still unused by the tool; the model-facing `limit` is the only result cap.
- `read` resolves the legacy path form only for `META.md` paths an earlier result in the same toolkit instance returned; a fresh toolkit rejects them by design.
- Drop-in filesystem caching (Python's per-inode parse cache, single-flight scans and `id()`-keyed inventory cache) is absent: `SkillFs` is an immutable snapshot and `SkillIndexToolkit::reload` rescans explicitly.

