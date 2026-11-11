# prompt — messages, content, tool calls, templates, tokens

## Intent

Provider-neutral prompt data types and the `{{slot}}` template engine.
Ports `openjiuwen/core/foundation/prompt/` (`template.py`, `assemble/*`) and the message schema of `openjiuwen/core/foundation/llm/schema/` (`message.py`, `tool_call.py`, `message_chunk.py`).
Everything above this module — the provider clients, tools, agents, workflow, memory — builds and inspects prompts through these types.

## API Surface

- `Role`: `System | User | Assistant | Tool | Other(String)`; serializes as the lowercase role string (`"system"`, `"tool"`, ...), deserializes any string (unknown → `Other`), implements `FromStr`/`Display`/`Default` (`User`).
- `MessageContent`: `Text(String) | Parts(Vec<ContentPart>)`; accepts a JSON string, a JSON array, or `null` (→ empty text); `as_text()`, `to_plain_text()` (joins text parts with `\n`), `parts()`, `is_empty()`; `From<&str>`/`From<String>`/`From<Vec<ContentPart>>`.
- `ContentPart`: `Text { text }`, `ImageUrl { image_url: ImageUrl }`, `Audio(AudioPart { data, format })`, `File(FilePart { url, data, mime_type, filename })`, `Other(Value)`.
  Each variant serializes as a tagged object (`{"type":"text","text":...}`, `{"type":"image_url","image_url":{"url":...,"detail":...}}`, ...); an unknown `type` deserializes into `Other`, preserving the raw object so provider-specific blocks round-trip.
- `BaseMessage { role, content, name, tool_calls: Vec<ToolCall>, tool_call_id, reasoning_content, metadata: BTreeMap<String, Value> }`; snake_case serde, unknown fields ignored, missing collections empty.
  Constructors `BaseMessage::system/user/assistant/tool(tool_call_id, content)` plus `with_name`, `with_tool_calls`, `with_reasoning`, `with_metadata`, `push_part`, `text()`.
- `ToolCall { id: Option<String>, kind: String /* "function" */, name, arguments: String /* JSON text */, index: Option<u32> }`; `new(name)`, `with_id`, `with_arguments(Value)`, `arguments_value() -> Result<Value>`, `to_openai()` / `from_openai(&Value)` for the nested `{id,type,function:{name,arguments}}` wire shape.
- `ToolCallDelta { index, id, name, arguments }` and `ToolCallAccumulator` (`push_delta`, `push_call`, `into_calls`, `is_empty`).
- `Usage { input_tokens, output_tokens, total_tokens, cache_tokens, reasoning_tokens, extra }` with `new`, `add`, `is_empty`, `normalized`.
- `PromptTemplate { name, content: TemplateContent, placeholder_prefix: String /* "{{" */, placeholder_suffix: String /* "}}" */ }` with `new`, `named`, `from_messages`, `with_placeholders`, `placeholders`, `input_keys`, `format(&BTreeMap<String, Value>)`, `format_one`, `to_messages`, `from_yaml_str`, `load_yaml`.
- `TokenCounter` trait, `ApproxTokenCounter` (≈4 chars/token), `TiktokenCounter` (feature `tiktoken`, cl100k_base), `count_message_tokens`, `truncate_messages`.

## Constraints

- Depends only on `openjiuwen-common` (`Error`, `Result`, redaction helpers); never on `llm`, `tool`, or `store`.
- No panics: malformed JSON produces `Error::Validation`; missing template slots are never errors (see below).
- Rendering is single-pass: substituted values are appended literally and never re-scanned (no injection cascade).
- Files ≤1000 lines: `content`/`message`/`tool_call`/`usage` are independent files; the assembler is a private module.

## Design Decisions

- **Template rendering contract** (Python parity): only `{{name}}` / `{{a.b}}` with the configured delimiters are placeholders; the name is whitespace-stripped; names may not contain `{`/`}`; an empty name (`{{}}`) and single braces stay literal.
  Extra slots are ignored, and a missing simple slot keeps the placeholder text verbatim (Python back-fills `"{{key}}"`), so partially formatted templates stay usable.
  A dotted path resolves through JSON objects; a missing intermediate or `null` leaf keeps the placeholder verbatim and logs a `tracing::warn` (Python raises on a missing head — deliberately softened to avoid panics).
  A `null` slot value therefore never renders as `"null"`.
- **Value stringification**: `Value::String` renders raw (no JSON quotes); numbers/bools use their JSON text; objects and arrays use compact JSON — a deliberate deviation from Python's `str()`/`repr()` so templates are language- and provider-neutral.
- **Message-list templates**: only each message's `content` is rendered (for `Parts`, only `Text` parts); `role`, `name`, `tool_calls`, `tool_call_id`, `metadata` are preserved. A list of messages is returned as a deep copy.
- **`to_messages()`**: text content → `[BaseMessage::user(content)]`; message content → the messages themselves; empty content → `[]` (Python parity).
- **Tool-call accumulation** (Python `AssistantMessageChunk.__add__`): a delta merges into the previous call when its id matches a non-empty id, or when either id is empty; `arguments` concatenate, the first non-empty `id`/`name`/`index` wins; otherwise a new call is appended.
- **`kind` is a `String`**, not an enum: providers only ever send `"function"`, but a permissive field avoids rejecting vendor extensions.
- **Tokens**: Python keeps real counters in `context_engine`; this module only offers a minimal counter for prompt construction so the feature flag `tiktoken` has a home. `context_engine` owns compression/trim policies.

## Routing Table

- `./role.rs` → `Role`
- `./content.rs` → `MessageContent`, `ContentPart`, `ImageUrl`, `AudioPart`, `FilePart` and their serde
- `./message.rs` → `BaseMessage` and constructors
- `./tool_call.rs` → `ToolCall`, `ToolCallDelta`, `ToolCallAccumulator`, OpenAI wire conversion
- `./usage.rs` → `Usage`
- `./template.rs` → `PromptTemplate`, `TemplateContent`, `to_messages`
- `./assembler.rs` → private placeholder scanning/substitution engine
- `./tokens.rs` → `TokenCounter`, `ApproxTokenCounter`, `TiktokenCounter`, `truncate_messages`
