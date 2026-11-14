# openjiuwen-symphony / discovery

## Intent

SKILL.md discovery and the lexical skill catalog, mapping to Python `openjiuwen.symphony.discovery`.
A scan turns a directory of skill folders into ordinary `CapabilityDescriptor`s, so discovered skills flow straight into the registry, graph, and planner without a plugin protocol.
The module ports both Python scan modes (immediate-child and bounded recursive), reports per-skill problems as diagnostics instead of failing the scan, and derives a stable, order-independent inventory fingerprint.
Skill search reuses the crate's own BM25 `LexicalIndex` rather than duplicating retrieval semantics.
The model-facing surface is completed by a deterministic prompt snapshot, the visible skill-name list, and an incremental skill-notice cursor with Python's reminder wording.
Everything here is offline, deterministic, and filesystem-only.

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

## Constraints

- Offline and deterministic: no network, no wall-clock ordering; skill and record ordering comes from ids, Python's casefold prompt order, or BM25 scores.
- A single bad skill never aborts a scan; diagnostics are the failure report and only an unusable root is `Err`.
- Blocking filesystem work uses tokio's `fs`; no `unsafe`, no `println!`, no panics on user input.
- Every public item is documented; files stay around or under 1000 lines and split by responsibility.
- Descriptors produced here must round-trip through serde and remain valid `CapabilityDescriptor`s.
- Front-matter keys outside the documented set are dropped, never leaked into metadata or retrieval text.
- Prompt rendering never splits a code point and never renders an empty string; notice ordering is casefold-then-exact and clock-free.

## Tests

- `tests/discovery.rs` — scans, front matter, diagnostics, descriptors, inventory, catalog search, excerpts, settings.
- `tests/skill_prompt.rs` — visible-name ordering and disabled-skill exclusion, exact pinned snapshot text (plain, category sections, empty), multi-byte `max_chars` truncation with `included`/`omitted`/`truncated` and fingerprint bookkeeping, `worker_id` filtering, fingerprint stability/content sensitivity, added/removed/changed notices with the exact Python reminder, reminder fallback and custom tool name, deterministic ordering, idempotent second consume, serde persistence, and `SkillFs::notice_state`.
- Unit tests beside `catalog.rs` pin the plain/category/empty renderings; unit tests beside `notice.rs` pin the reminder wording, the `textwrap.shorten` behavior, and state round-tripping.

## Routing Table

- `./mod.rs` → module docs and re-exports of the whole discovery surface
- `./frontmatter.rs` → `parse_front_matter`, `FrontMatterStatus`/`FrontMatterOutcome`, tag/IO/boolean coercion, capability-type normalization
- `./scanner.rs` → `SkillFolderScanner`, `SkillScanConfig`, descriptor derivation, diagnostics, scan constants
- `./models.rs` → `SkillRecord`, `SkillInventory`, `sort_records`, inventory fingerprint, `DEFAULT_SOURCE`
- `./catalog.rs` → `SkillFs`, `SkillHit`, `SelectionCard`, BM25 search, excerpts, `visible_skill_names`, `prompt_snapshot`
- `./notice.rs` → `SkillNoticeState`, `SkillNotice`, incremental reminder rendering, notice constants
- `./settings.rs` → `DiscoverySettings`, token budget, settings defaults

