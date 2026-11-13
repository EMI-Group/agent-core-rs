# openjiuwen-symphony / discovery

## Intent

SKILL.md discovery and the lexical skill catalog, mapping to Python `openjiuwen.symphony.discovery`.
A scan turns a directory of skill folders into ordinary `CapabilityDescriptor`s, so discovered skills flow straight into the registry, graph, and planner without a plugin protocol.
The module ports both Python scan modes (immediate-child and bounded recursive), reports per-skill problems as diagnostics instead of failing the scan, and derives a stable, order-independent inventory fingerprint.
Skill search reuses the crate's own BM25 `LexicalIndex` rather than duplicating retrieval semantics.
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

- Offline and deterministic: no network, no wall-clock ordering; skill and record ordering comes from ids or BM25 scores.
- A single bad skill never aborts a scan; diagnostics are the failure report and only an unusable root is `Err`.
- Blocking filesystem work uses tokio's `fs`; no `unsafe`, no `println!`, no panics on user input.
- Every public item is documented; files stay around or under 1000 lines and split by responsibility.
- Descriptors produced here must round-trip through serde and remain valid `CapabilityDescriptor`s.
- Front-matter keys outside the documented set are dropped, never leaked into metadata or retrieval text.

## Routing Table

- `./mod.rs` → module docs and re-exports of the whole discovery surface
- `./frontmatter.rs` → `parse_front_matter`, `FrontMatterStatus`/`FrontMatterOutcome`, tag/IO/boolean coercion, capability-type normalization
- `./scanner.rs` → `SkillFolderScanner`, `SkillScanConfig`, descriptor derivation, diagnostics, scan constants
- `./models.rs` → `SkillRecord`, `SkillInventory`, `sort_records`, inventory fingerprint, `DEFAULT_SOURCE`
- `./catalog.rs` → `SkillFs`, `SkillHit`, `SelectionCard`, BM25 search, excerpts
- `./settings.rs` → `DiscoverySettings`, token budget, settings defaults
