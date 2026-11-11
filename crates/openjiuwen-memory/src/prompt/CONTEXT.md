# prompt

## Intent

Turn recall results into prompt material without depending on `openjiuwen-context` (no cycle, no window policy): a rendered Markdown/JSON string for template placeholders and `BaseMessage` values for callers that append messages.
Ports the Python behavior where memory results are JSON-dumped into `sys_long_term_memory`/`sys_memory_variables` and rendered by `PromptTemplate.format`.

## API Surface

```rust
pub const MEMORY_SECTION_TITLE: &str = "Long-term memory";
pub const MEMORY_SECTION_TAG: &str = "[MEMORY]";
pub const MEMORY_PLACEHOLDER: &str = "sys_long_term_memory";
pub const VARIABLES_PLACEHOLDER: &str = "sys_memory_variables";

pub fn render_records(records: &[MemoryRecord], max_tokens: usize) -> String;
pub fn render_snippets(hits: &[RecallHit], max_tokens: usize) -> String;
pub fn hits_to_system_message(hits: &[RecallHit], max_tokens: usize) -> Option<BaseMessage>;
pub fn hits_to_messages(hits: &[RecallHit], max_tokens: usize) -> Vec<BaseMessage>;
pub fn variables_to_system_message(variables: &BTreeMap<String, String>) -> Option<BaseMessage>;
pub fn to_memory_json(hits: &[RecallHit]) -> Result<String>;   // Python template parity
pub fn to_memory_values(hits: &[RecallHit]) -> Vec<Value>;     // [{id, mem, mem_type, timestamp, score}]
```

## Rendering Rules

- `render_records` / `render_snippets` emit nothing for empty input (`""`). Otherwise:
  ```text
  ## Long-term memory
  - [<type>] <content>
  - [<type>] <content> (score: 0.87)
  ```
  `render_snippets` includes the score (3 decimals) and uses `hits` order; `render_records` omits scores and uses `MemoryType::as_str`.
- Token budget: `ApproxTokenCounter` (foundation); records are added while `used + count_text(line) <= max_tokens`. A budget of `0` returns `""` (header included only when at least one line fits). Never split a line.
- `hits_to_system_message` / `variables_to_system_message` return `None` when the rendered body is empty; otherwise `BaseMessage::system(body)`.
- `hits_to_messages` returns `Vec` of zero or one system message (kept as a Vec so callers can extend it uniformly).
- `to_memory_values` serializes Python `MemResult` shape: `{"id", "mem", "mem_type", "timestamp" (RFC 3339), "score"}`; `to_memory_json` is `serde_json::to_string` of that array (`Error::Validation` on serialization failure, which cannot happen for these types).
- Content is included verbatim (no HTML escaping); multi-line content is collapsed to one line with spaces for the bullet list.

## Constraints

- Depends only on `openjiuwen_foundation::prompt::{BaseMessage, ApproxTokenCounter, TokenCounter}` and this crate's `record` types.
- No `println!`; no context-window manipulation; no message-role policy beyond `system`.
- Deterministic output for identical input.

## Routing Table

- `./mod.rs` → constants and public functions
- `./render.rs` → Markdown rendering and token trimming
- `./messages.rs` → `BaseMessage` construction and Python `MemResult` JSON projection

## Test Strategy

- Unit tests: empty input, header/line shape, score formatting, token trimming (including zero budget), multi-line collapse, Python JSON field names, variables rendering, `None` behavior for empty.
- `tests/recall.rs` asserts `render_snippets` ordering matches hit ordering.
