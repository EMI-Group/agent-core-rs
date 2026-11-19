# prompt — messages, content, tool calls, templates, tokens

## Intent

Provider-neutral prompt data types and the `{{slot}}` template engine.
Ports `openjiuwen/core/foundation/prompt/` (`template.py`, `assemble/*`) and the message schema of `openjiuwen/core/foundation/llm/schema/` (`message.py`, `tool_call.py`, `message_chunk.py`).
Everything above this module — the provider clients, tools, agents, workflow, memory — builds and inspects prompts through these types.
The module is async-free and performs no I/O except `PromptTemplate::load_yaml`.

## API Surface

- `Role`: `System | User | Assistant | Tool | Other(String)`; serializes as the bare lowercase role string (`"system"`, `"tool"`, or the raw `Other` payload), deserializes any string (unknown → `Other`), rejects non-string JSON.
  `as_str`, `is_system`/`is_user`/`is_assistant`/`is_tool`, `FromStr` (infallible), `Display`, `Default` (`User`), `From<&str>`/`From<String>`, `Eq + Hash`.
- `MessageContent`: `Text(String) | Parts(Vec<ContentPart>)`; accepts a JSON string, a JSON array, or `null` (→ empty text) and rejects other shapes.
  `as_text() -> Option<&str>`, `to_plain_text() -> String` (joins text parts with `\n`), `parts() -> &[ContentPart]` (`Text` → `&[]`), `is_empty()`, `len()` (content parts: `Text` → 0 when empty else 1, `Parts` → length).
  `From<&str>`, `From<String>`, `From<Vec<ContentPart>>`, `Default` (empty text).
- `ContentPart`: `Text { text }`, `ImageUrl { image_url }`, `Audio(AudioPart { data, format })`, `File(FilePart { url, data, mime_type, filename })`, `Other(Value)`.
  Each variant serializes as a tagged object (`{"type":"text","text":...}`, `{"type":"image_url","image_url":{"url":...,"detail":...}}`, ...); unknown or missing `type` and a bare string array element deserialize into `Text`/`Other` without error, preserving provider-specific blocks verbatim.
  `text() -> Option<&str>`, `image_url() -> Option<&ImageUrl>`, `part_type() -> &str`.
  `ImageUrl { url, detail }`, `AudioPart { data, format }`, `FilePart { url, data, mime_type, filename }` skip `None` fields.
- `BaseMessage { role, content, name, tool_calls, tool_call_id, reasoning_content, metadata: BTreeMap<String, Value> }`; snake_case serde, missing fields defaulted, unknown fields ignored, `None`/empty fields omitted.
  Constructors `BaseMessage::system/user/assistant(content)` and `tool(tool_call_id, content)`, builders `with_name`, `with_tool_calls`, `with_reasoning`, `with_metadata(key, value)`, mutator `push_part`, accessor `text() -> Cow<'_, str>`, `From<&str>`/`From<String>` (user message), `Default` (empty user message).
- `ToolCall { id, kind (serde "type", default "function"), name, arguments, index }` with `new(name)`, `with_id`, `with_arguments(Value)`, `with_index`, `arguments_value() -> Result<Value>` (empty arguments → `Value::Null`, malformed → `Error::validation`), `to_openai()` / `from_openai(&Value)` / `from_openai_with_index(&Value, u32)` for the nested `{id,type,function:{name,arguments}}` wire shape (flat `{name,arguments}` also accepted on the way in).
- `ToolCallDelta { index, id, name, arguments }` and `ToolCallAccumulator` (`new`, `is_empty`, `calls`, `push_delta`, `push_call`, `into_calls`).
- `Usage { input_tokens, output_tokens, total_tokens, cache_tokens, reasoning_tokens, extra }` with `new`, `add`, `AddAssign`, `is_empty`, `normalized` (total = input+output when 0); unknown provider counters land in `extra` (serde `flatten`) and re-serialize flat.
- `Usage` has no `skip_serializing_if`, so every counter is always present on the wire: `to_value(&Usage::new(3, 4)) == {"input_tokens":3,"output_tokens":4,"total_tokens":7,"cache_tokens":0,"reasoning_tokens":0}` (frames such as `llm_usage` bake these keys in).
- `PromptTemplate { name, content: TemplateContent, placeholder_prefix: String /* "{{" */, placeholder_suffix: String /* "}}" */ }` and `TemplateContent { Text(String), Messages(Vec<BaseMessage>) }` (untagged serde).
  `new`, `named`, `from_messages`, `with_placeholders`, `placeholders`, `input_keys`, `format(&BTreeMap<String, Value>) -> PromptTemplate`, `try_format(&BTreeMap<String, Value>) -> Result<PromptTemplate>`, `resolved_keys(&BTreeMap<String, Value>) -> Vec<String>`, `unresolved_keys(&BTreeMap<String, Value>) -> Vec<String>`, `format_one(key, value)`, `to_messages() -> Vec<BaseMessage>`, `from_yaml_str`, `load_yaml`.
- `TokenCounter` trait (`count_text`, `count_messages` with a framing-aware default, `is_exact` defaulting to `false`), `ApproxTokenCounter` (`DEFAULT_CHARS_PER_TOKEN = 4`, `new`, `with_chars_per_token`, `chars_per_token`, `is_exact() == false`), `TiktokenCounter` (feature `tiktoken`, cl100k_base with approx fallback, `is_exact()` true only while the BPE is loaded), `count_message_tokens`, `truncate_messages(&mut Vec<BaseMessage>, max_tokens, &dyn TokenCounter) -> usize`.
- `mod.rs` re-exports every public type above at `openjiuwen_foundation::prompt::*`.

## Constraints

- Depends only on `openjiuwen-common` (`Error`, `Result`); never on `llm`, `tool`, or `store`.
- No panics: malformed JSON produces `Error::Validation`; missing template slots are never errors (see below).
- Rendering is single-pass: substituted values are appended literally and never re-scanned (no injection cascade).
- Files ≤1000 lines: `content`/`message`/`tool_call`/`usage` are independent files; the assembler is a private module.
- `#[cfg(feature = "tiktoken")]` wraps the `tiktoken-rs` dependency, so the crate builds and tests with `--no-default-features`.

## Design Decisions

- **Template rendering contract** (Python parity): only `{{name}}` / `{{a.b}}` with the configured delimiters are placeholders; the name is whitespace-stripped; names may not contain `{`/`}`; an empty name (`{{}}`) and single braces stay literal.
  Extra slots are ignored, and a missing simple slot keeps the placeholder text verbatim (Python back-fills `"{{key}}"`), so partially formatted templates stay usable.
  That lenient path is not silent: `format` emits one `tracing::warn!` naming the template and the unresolved placeholder paths, `unresolved_keys`/`resolved_keys` expose the same split to code, and `try_format` turns it into an `Error::Validation` listing the missing placeholders — the variant to use wherever a rendered prompt is sent to a provider.
  `to_messages` warns too, because it copies any remaining placeholder text straight into the messages a model reads.
  A dotted path resolves through JSON objects; a missing head/intermediate, a non-object intermediate, or `null` leaf keeps the placeholder verbatim and logs a `tracing::warn` (Python raises on a missing head — deliberately softened to avoid panics).
  A `null` slot value therefore never renders as `"null"`.
  The never-mutating `format` returns a new template and copies `name` and both delimiters; an empty value map is a clone.
- **Scan recovery** (assembler): after a rejected match (empty name or a name containing braces) the cursor advances one character past the opening delimiter, so `"{{a{{b}}"` resolves the inner `{{b}}`; empty delimiters mean "no placeholders".
- **Value stringification**: `Value::String` renders raw (no JSON quotes); numbers/bools use their JSON text; objects and arrays use compact JSON — a deliberate deviation from Python's `str()`/`repr()` so templates are language- and provider-neutral.
- **Message-list templates**: only each message's `content` is rendered (for `Parts`, only `Text` parts; image/audio/file/`Other` parts are byte-identical); `role`, `name`, `tool_calls`, `tool_call_id`, `reasoning_content`, `metadata` are preserved. A list of messages is returned as a deep copy.
  Python's `DictableVariable` also renders `image_url.url`; this module deliberately renders text parts only.
- **`to_messages()`**: text content → `[BaseMessage::user(content)]`; message content → the messages themselves; empty content → `[]` (Python parity).
- **Tool-call accumulation** (Python `AssistantMessageChunk.__add__`): a delta merges into the previous call when both ids are non-empty and equal, or when either id is empty, and both kinds are `"function"`; `arguments` concatenate, the first non-empty `id`/`name`/`index` wins; otherwise a new call is appended with `index` set to its array position. Non-`"function"` kinds never merge.
- **`kind` is a `String`**, not an enum: providers only ever send `"function"`, but a permissive field avoids rejecting vendor extensions. `to_openai()` omits `index` (OpenAI carries position on the enclosing array).
- **Content tolerance**: a malformed payload for a known part tag falls back to `Other(raw object)` instead of failing, so vendor variants are never rewritten or lost.
- **YAML**: `from_yaml_str` also accepts a bare string document (→ `content` with defaults); `load_yaml` maps both I/O and parse failures to `Error::Validation` with the path as context.
- **Tokens**: Python keeps real counters in `context_engine`; this module only offers a minimal counter for prompt construction so the feature flag `tiktoken` has a home. `context_engine` owns compression/trim policies.
  `ApproxTokenCounter::count_text` = `chars().count().div_ceil(chars_per_token)` (empty → 0, chars_per_token clamped to ≥1); the default `count_messages` charges `4 + text tokens` per message plus name/argument tokens per tool call.
  The approximation is documented as an estimate, not a tokenizer (English prose within roughly ±20–30 % of a real BPE count; symbol-dense code and CJK under-counted), and `TokenCounter::is_exact` says which kind a counter is: `false` for the heuristic, `true` for `TiktokenCounter` only while its vocabulary is really loaded.
  `truncate_messages` is the crate's one budget decision, so it logs one `tracing::warn!` per process (`std::sync::Once`) when the counter it was given is inexact.
  `truncate_messages` drops the oldest non-system messages until the budget fits, preserving system messages in order, and returns the dropped count.
  `TiktokenCounter` shares one `Lazy` cl100k_base BPE and falls back to the approx counter (one `tracing::warn`) if it cannot be loaded.

## Known Issues
- `truncate_messages` evicts the oldest non-system message one at a time, so it can leave a `tool` result whose assistant `tool_calls` message was removed (and a list starting with `assistant`), which provider APIs reject.
- `truncate_messages` recomputes `count_messages` and rescans the list after every single removal, so truncating a long conversation is quadratic in message count.
- `assembler::render` emits one `warn` per unresolved placeholder occurrence in addition to `PromptTemplate::format`'s single summary warning, so a partially formatted template logs N+1 events per render.
- `assembler::scan` re-searches for the suffix from scratch after every rejected match, so brace-heavy template text makes placeholder scanning quadratic.

## Test Strategy

- Unit tests live beside the module (`#[cfg(test)] mod tests`), one behavior per test, no network.
- Integration tests live in `crates/openjiuwen-foundation/tests/`: `messages.rs` (message/content/tool-call/usage JSON shapes) and `prompt_template.rs` (rendering matrix, YAML, serde); each starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` and names the Python test it mirrors.
- `cargo test -p openjiuwen-foundation prompt` and `--no-default-features prompt` must both pass offline.

## Routing Table

- `./role.rs` → `Role`
- `./content.rs` → `MessageContent`, `ContentPart`, `ImageUrl`, `AudioPart`, `FilePart` and their serde
- `./message.rs` → `BaseMessage` and constructors
- `./tool_call.rs` → `ToolCall`, `ToolCallDelta`, `ToolCallAccumulator`, OpenAI wire conversion
- `./usage.rs` → `Usage`
- `./template.rs` → `PromptTemplate`, `TemplateContent`, `to_messages`
- `./assembler.rs` → private placeholder scanning/substitution engine
- `./tokens.rs` → `TokenCounter`, `ApproxTokenCounter`, `TiktokenCounter`, `truncate_messages`
