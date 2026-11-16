# src/agent — agent-facing tools over the capability-tree index

## Intent

Ports the agent surface of Python `openjiuwen.symphony.agent` (v0.1.20): the read-only browsing and search tool a harness advertises to a model so it can explore the capability tree and read one skill's routing description.
One toolkit, four operations (`list`, `browse`, `search`, `read`) over the query-time runtime in [`crate::retrieval::search`](../retrieval/search/CONTEXT.md), which owns index loading, the progressive walk and the lexical fallback.
The toolkit never builds an index (that is [`crate::retrieval::build`](../retrieval/CONTEXT.md)), never calls a model, and never fabricates a hit: an unusable query or a zero-hit search abstains with a pinned reason and a suggestion.
Everything here is offline, deterministic and session scoped; identical requests render byte-identical content.

## API Surface

All paths are crate-relative. Every public item is documented (`missing_docs` is denied at the crate level); `cargo clippy -- -D warnings` includes `missing_docs`.

### `agent` (`src/agent/mod.rs`)

```rust
pub mod retrieval_toolkit;
pub use retrieval_toolkit::{
    render_retrieval_prompt, root_cid, tool_input_schema, RetrievalEntry, RetrievalOperation,
    RetrievalPromptSnapshot, RetrievalRequest, RetrievalResponse, RetrievalToolCard,
    RetrievalToolkit, RetrievalToolkitConfig, BLANK_QUERY_SUGGESTION, NO_RESULTS_SUGGESTION,
    SKILL_RETRIEVAL_TOOL_NAME,
};
```

### `agent::retrieval_toolkit` (`src/agent/retrieval_toolkit/mod.rs`)

```rust
pub const SKILL_RETRIEVAL_TOOL_NAME: &str = "skill_retrieval";

pub const MAX_CURSORS: usize = 32;            // the per-toolkit LRU cap
pub const CURSOR_MAX_CHARS: usize = 128;      // the longest accepted cursor token
pub const CURSOR_FOOTER_CHARS: usize = 96;    // the `[next] …` footer reserve
pub const CURSOR_STATE_VERSION: u64 = 1;      // hashed into every token
pub const CURSOR_TOKEN_BYTES: usize = 16;     // SHA-256 prefix length (22 base64 chars)

pub const MIN_OUTPUT_CHARS: usize = 512;
pub const MAX_OUTPUT_CHARS: usize = 48_000;
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 12_000;
pub const DEFAULT_PAGE_SIZE: usize = 10;
pub const MAX_PAGE_SIZE: usize = 100;
pub const DEFAULT_TOP_K: usize = 10;
pub const DEFAULT_MAX_TOP_K: usize = 20;
pub const MAX_TOP_K: usize = 100;

pub const ABSTAIN_BLANK_QUERY: &str = "blank_query";
pub const ABSTAIN_NO_RESULTS: &str = "no_results";
pub const MODEL_OPERATIONS: [RetrievalOperation; 4] = [List, Browse, Search, Read];

pub enum RetrievalOperation { List /* default */, Browse, Search, Read }
impl RetrievalOperation {
    pub fn as_str(&self) -> &'static str;          // `list`/`browse`/`search`/`read`
}
// Display, FromStr (trims; `Error::Validation` naming `operation`), Serialize as snake_case.

pub struct RetrievalRequest {                       // Default = `{"operation":"list"}`
    pub operation: RetrievalOperation,
    pub cid: Option<String>,
    pub query: Option<String>,
    pub cursor: Option<String>,
    pub page_size: Option<usize>,
    pub top_k: Option<usize>,
}

pub struct RetrievalEntry {                         // kind: `branch` | `item` | `search_hit`
    pub kind: String,
    pub cid: String,
    pub worker_id: String,       // empty for branches
    pub label: String,
    pub description: String,
    pub path: Vec<String>,       // ancestor cids, root excluded
    pub category: String,
    pub rank: usize,             // 1-based search rank; 0 otherwise
    pub item_count: usize,       // branch subtree totals; 0 for items
    pub branch_count: usize,
}

pub struct RetrievalResponse {
    pub operation: RetrievalOperation,
    pub content: String,          // title + rows + any `[next]` footer
    pub model_content: String,    // `[skill_retrieval] …` header + content
    pub entries: Vec<RetrievalEntry>,  // only the rows this page shows
    pub cid: String,
    pub path: Vec<String>,
    pub total_count: usize,       // matches before paging
    pub shown_count: usize,
    pub truncated: bool,
    pub truncation_reason: Option<String>, // `max_output_chars` | `page`
    pub next_cursor: Option<String>,
    pub result_complete: bool,
    pub abstained: bool,
    pub abstain_reason: Option<String>,
    pub suggestion: Option<String>,
    pub method: String,           // `Retriever::method()`
    pub effective_top_k: usize,   // search only, else 0
    pub page_size: usize,         // read => 0
}

pub struct RetrievalToolkitConfig {
    pub session_scope: String,
    pub max_output_chars: usize,
    pub default_page_size: usize,
    pub max_page_size: usize,
    pub max_top_k: usize,
}
impl RetrievalToolkitConfig {
    pub fn new(session_scope: impl Into<String>) -> Self;  // defaults: 12_000 / 10 / 100 / 20
}

pub struct RetrievalToolCard {
    pub id: String,               // `skill_retrieval__<sha256 hex>`
    pub name: String,             // `skill_retrieval`
    pub description: String,
    pub input_schema: serde_json::Value,
    pub parallel_safe: bool,      // false
    pub stateless: bool,          // false — it keeps page cursors
}
impl openjiuwen_common::card::Card for RetrievalToolCard { /* id/name/description */ }

pub fn tool_input_schema() -> serde_json::Value;  // operation required; 5 optional; additionalProperties: false
pub fn root_cid() -> &'static str;                // `"ROOT"`
pub fn render_retrieval_prompt(index: &LoadedIndex, max_chars: usize) -> RetrievalPromptSnapshot;

pub struct RetrievalPromptSnapshot {
    pub text: String,
    pub included: Vec<String>,    // first-level cids shown, tree order
    pub omitted: Vec<String>,     // first-level cids not shown (cap or budget)
    pub truncated: bool,          // the budget, not the category cap, cut the text
    pub max_chars: usize,
}
```

`RetrievalToolkit` (`src/agent/retrieval_toolkit/toolkit.rs`):

```rust
impl RetrievalToolkit {
    pub fn new(retriever: Retriever, config: RetrievalToolkitConfig) -> Result<Self>;
    pub async fn from_loaded_index(index: LoadedIndex, config: RetrievalToolkitConfig) -> Result<Self>;
    pub fn card(&self) -> RetrievalToolCard;
    pub fn tool_id(&self) -> &str;
    pub fn index(&self) -> &LoadedIndex;
    pub fn config(&self) -> &RetrievalToolkitConfig;
    pub fn retriever(&self) -> &Retriever;
    pub fn cursor_count(&self) -> usize;
    pub fn close(&mut self);                                  // drops every cursor
    pub fn prompt_snapshot(&self, max_chars: usize) -> RetrievalPromptSnapshot;
    pub async fn invoke(&mut self, arguments: &serde_json::Value) -> Result<RetrievalResponse>;
    pub async fn execute(&mut self, request: &RetrievalRequest) -> Result<RetrievalResponse>;
}
```

`from_loaded_index` builds the retriever with `RetrieverConfig::top_k = config.max_top_k` and the retriever defaults, so the lexical fallback answers with no model configured.
`invoke` is the model-facing path (`parse_invoke_arguments` + `execute`); `execute` is the typed path.

### Rules the toolkit guarantees

- **Abstain, never fabricate.** A blank/whitespace/missing `query` ⇒ `abstained = true`, `abstain_reason = "blank_query"`, `BLANK_QUERY_SUGGESTION`; a search with zero ranked hits ⇒ `"no_results"`, `NO_RESULTS_SUGGESTION`. Both are `Ok`, with `total_count = 0`, `result_complete = true` and no cursor.
- **Cursors.** The token is URL-safe unpadded base64 of the first 16 bytes of SHA-256 over canonical JSON `{v,operation,cid,query,offset,page_size,top_k,fingerprint}`, where `fingerprint` is SHA-256 hex over the canonical catalog+tree content. Tokens are 22 characters, stored in an LRU of `MAX_CURSORS`, and consumed on resume (a second use is `cursor: unknown or expired …`). A page mints a cursor only when `offset + shown < total && shown > 0`, and the footer is `[next] {"operation":"…","cursor":"…"}`.
- **Resume checks.** A cursor is rejected when the operation, cid, query, page size or `top_k` differs from the minting request, or when the index fingerprint changed; every case is `Error::Validation` naming `cursor`.
- **Clamping, not rejection.** `page_size`/`top_k` `0` selects `default_page_size`/`DEFAULT_TOP_K`; larger values clamp to `max_page_size`/`max_top_k` (`effective_top_k` and `page_size` in the response report the effective values).
- **Deterministic order.** Branches by `node_id`, own items by `item_id`, search hits in retriever rank order; read resolves a leaf cid, then worker id, then tree item id/payload.
- **Bounded output.** `content` and `model_content` stay within `max_output_chars`; `truncated`/`truncation_reason` say whether the budget or the page shortened the answer, and every shortening is character based (no split multi-byte character).
- **Unknown cids.** An unknown branch or item is `Error::NotFound` (never an empty success); a leaf cid passed to `list`/`browse` is `NotFound` naming `operation=read`.

## Validation Matrix

| Rejection | Error | Field |
| --- | --- | --- |
| blank `session_scope` | `Validation` | `session_scope` |
| `max_output_chars < MIN_OUTPUT_CHARS` or `> MAX_OUTPUT_CHARS` | `Validation` | `max_output_chars` |
| `max_page_size == 0` or `> MAX_PAGE_SIZE` | `Validation` | `max_page_size` |
| `max_top_k == 0` or `> MAX_TOP_K` | `Validation` | `max_top_k` |
| `default_page_size == 0` or `> max_page_size` | `Validation` | `default_page_size` |
| arguments not a JSON object, or an unknown key | `Validation` | `arguments` |
| missing/`null`/non-string `operation` | `Validation` | `operation` |
| unknown operation name | `Validation` | `operation` (lists the four names) |
| a field of the wrong JSON type (string/number/`null` mismatch) | `Validation` | that field |
| blank `cid` | `Validation` | `cid` |
| `cid` with `search` | `Validation` | `cid` |
| missing `cid` for `read` | `Validation` | `cid` (required) |
| `query` with a non-`search` operation | `Validation` | `query` |
| `page_size` with `read` | `Validation` | `page_size` |
| `top_k` with a non-`search` operation | `Validation` | `top_k` |
| `cursor` with `read`, blank, or longer than `CURSOR_MAX_CHARS` | `Validation` | `cursor` |
| unknown/expired cursor | `Validation` | `cursor` |
| cursor reused with a different operation/cid/query/page size/`top_k` | `Validation` | `cursor` |
| cursor minted over a different index | `Validation` | `cursor` |
| unknown branch cid (`list`/`browse`) | `NotFound` | — |
| leaf cid passed to `list`/`browse` | `NotFound` (points at `operation=read`) | — |
| unknown item cid/worker id (`read`) | `NotFound` | — |

No malformed request panics; `page_size`/`top_k` are clamped rather than rejected.

## Constraints

- Cargo: no new dependency. `sha2` (cursor tokens, index fingerprint, tool id), `base64::engine::general_purpose::URL_SAFE_NO_PAD` + `base64::Engine as _` (tokens), `serde`, `serde_json`, `schemars` (`JsonSchema` on every public data type), `tracing` only if a warning is warranted — never `println!`.
- Read-only over `crate::retrieval::types::{LoadedIndex, RetrieverNode, RetrieverItem, CatalogRecord}`; the toolkit does not mutate a tree, a catalog or an index.
- No unsafe, no `unwrap`/`expect` outside `#[cfg(test)]`, no panics on user input, no clock/RNG/network — cursors and fingerprints are hashes, never timestamps.
- `std::fmt::Write as _` + `let _ = write!(…)` for string building (the crate's `clippy::format_push_string` policy); `push_str` for literals.
- `#[serde(rename_all = "snake_case", default)]` on every public data type; unknown JSON fields are tolerated on deserialization.
- One toolkit per session: `invoke`/`execute` take `&mut self` because the toolkit owns the page-cursor store; `index`, `config`, `retriever`, `card` and `prompt_snapshot` are `&self`.
- Files stay ≲1000 lines; `render.rs`, `ops.rs`, `validate.rs`, `toolkit.rs` split by responsibility, with `tests.rs` holding the in-file suite.

## Routing Table

- `./mod.rs` → module docs, constants, `RetrievalOperation`, `RetrievalRequest`, `RetrievalEntry`, `RetrievalResponse`, `RetrievalToolkitConfig`, `RetrievalToolCard` + `Card` impl, `tool_input_schema`, `root_cid`, re-exports
- `./toolkit.rs` → `RetrievalToolkit` (constructors, `invoke`/`execute`, resume, per-operation run, abstain, paging + cursor minting), `BLANK_QUERY_SUGGESTION`, `NO_RESULTS_SUGGESTION`
- `./ops.rs` → the four operation builders over `LoadedIndex`/`RetrievalResult`: `list`, `browse`, `search`, `read`, `node_stats`, branch/item row construction and cid resolution
- `./render.rs` → page/read/abstain rendering, the `[skill_retrieval]` summary header, `render_retrieval_prompt` + `RetrievalPromptSnapshot`, and the character-budget helpers (`compact`, `truncate_chars`, `append_note`)
- `./validate.rs` → `validate_config`, `parse_invoke_arguments`, `validate_request`, the `CursorState`/`CursorStore` LRU, `validate_resume`, `index_fingerprint`
- `./tests.rs` → the behaviour suite (see below)
- `../mod.rs` → `agent` module docs and the public re-export list

## Test Strategy

Unit tests live in `src/agent/retrieval_toolkit/tests.rs` (`#[cfg(test)] mod tests;`), offline over a hand-built `LoadedIndex` with the lexical fallback (no model wired).

| Group | Pins |
|---|---|
| list | root/nested/empty branch rows, branch before item order, item/branch counts, `path`/`category`, `ROOT` sentinel vs omitted cid, leaf-cid and unknown-branch rejection |
| browse | sub-branches then own items, catalog identity (`worker_id`, label, description), empty node message, unknown cid |
| search | ranked hits in retriever order, `method`/`effective_top_k`, zero-hit abstain, blank/missing-query abstain, `top_k` clamp |
| read | by leaf cid, worker id and tree item id, catalog body (`META_CONTENT`), `SKILL.md` line, fallback without a catalog, unknown item, missing cid |
| paging | `[next]` footer and cursor resume, cursor consumption/expiry, page-size `0` and clamping, mismatched-cursor matrix, `close()` |
| budget | wide-tree page shortened by `max_output_chars` with a resumable cursor, multi-byte `truncate_chars`/`compact` behaviour |
| determinism | repeated `invoke`/`execute` calls are byte-identical (content, model_content, entries, cursor) |
| invoke | unknown keys, wrong JSON types, unknown operation, misused fields, `invoke`/`execute` parity |
| config/card | the five configuration rejections, `Card` id/name/description, session-scoped tool id stability |
| prompt | category lines, ordering, the 30-category cap and `more categories` note, budget cuts, empty tree, multi-byte budget |
| ops | `node_stats` subtree totals, search rows keeping rank |

## Deferred Scope

Python's index-BUILD machinery is deliberately absent, not stubbed: this crate's index building already lives in [`crate::retrieval::build`](../retrieval/CONTEXT.md), and the toolkit is the read-only browse/search surface.

| Python surface | Why it is not ported |
|---|---|
| `build_index`, `build_index_async`, `check_build_status`, `cancel_build`, `load_index_status` | Duplicates `retrieval::build`; the toolkit takes a `LoadedIndex` or a `Retriever`, so a build tool here would be a second, drifting builder |
| publish locks, commit/state files, temp dirs, `portalocker`, wall-clock timing | The crate is clock-free and single-process by design; `retrieval::build` already publishes its three artifacts atomically |
| `update_skills` / `update_visible_skills` live refresh | Requires mutating the index and the notice registry from a long-lived host; the Rust toolkit is an immutable snapshot over a loaded index (rebuild and construct a new toolkit) |
| S3/HTTP index materialization, `service/serving.py` | The query-time runtime reads a local index directory; remote materialization is a documented gap of `retrieval::search` |
| `load_index_tree` / raw tree dumping | Replaced by `list`/`browse`, which render scored, counted, paged rows instead of the raw node tree |
| Python's `AgenticRetrievalConfig` tool-name injection and its zh locale prompts | The Rust surface pins one tool name (`skill_retrieval`) and one English prompt; a host that wants another name renames the card before registration |
| Python's three separate tools (`skill_index_build`, `skill_branch_explore`, `skill_branch_peek`) | One tool with an `operation` discriminator covers all browsing, so the model sees one card and one argument schema |

Nothing in this module is a silent no-op: every operation either answers with rows, abstains with a pinned reason, or fails with a typed error naming the field.
