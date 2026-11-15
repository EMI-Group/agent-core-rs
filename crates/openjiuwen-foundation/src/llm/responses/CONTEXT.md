# responses — OpenAI Responses API transport

## Intent

The third model protocol of `openjiuwen-foundation`: the `OpenAI` Responses API (`POST {base}/responses`), used by the `openai_account` provider and by any `openai`-family provider configured with `api_mode: responses`.
Ports the Responses branch of `openjiuwen/core/foundation/llm/model_clients/openai_model_client.py` plus `.../utils/responses_utils.py`, `.../utils/responses_transport.py`, and `.../model_clients/openai_account_model_client.py`.
The module mirrors `../openai/`: pure request projection, pure response parsing, an event-stream state machine, and a thin `Model` implementation over the shared HTTP client.

## API Surface

`request`:
- `normalize_api_base(&str) -> String` — trims whitespace, trailing slashes, and a pasted `/responses` suffix; never invents a `/v1` segment.
- `convert_messages(&[BaseMessage]) -> Result<(String, Vec<Value>)>` — the `(instructions, input)` pair; `Error::Validation` on an empty conversation.
- `convert_tools(&[Value]) -> Vec<Value>` — OpenAI-shaped tool declarations flattened to the Responses shape.
- `build_responses_body(&[BaseMessage], &ModelRequestConfig, stream: bool) -> Result<Value>` — the full wire body; `Error::Config` on a blank `model_name`, `Error::Validation` for a malformed `max_tool_calls`/`previous_response_id`/`conversation_id`, `Error::Unsupported` for `background` mode and hosted tools.
- `EXTRA_BODY_KEY` (`extra_body`) — shared with `state` so the client-side-key filter and the nested flattening agree.

`response`:
- `parse_responses_response(&Value) -> Result<ChatResponse>` — parses the non-streaming response object (message `output_text`/`text` parts, `reasoning.summary[].text`, top-level `output_text`, `function_call` items, `id`).
- `parse_responses_usage(&Value) -> Usage` — `input_tokens|prompt_tokens`, `output_tokens|completion_tokens`, cache and reasoning detail counters.

`state`:
- `ResponsesSessionStore` (`last_response_id(&self, &str) -> Option<String>`, `remember_response_id(&self, &str, &str)`) — the conversation seam mapping a caller-supplied conversation key to the last response id.
- `InMemoryResponsesSessionStore` (`new`, `with_capacity`, `capacity`, `Default`) — bounded LRU store of `DEFAULT_CONVERSATION_CAPACITY` (256) conversations, capacity clamped to at least one, least-recently-used conversation evicted first (a lookup counts as use).
- `NoopResponsesSessionStore` — remembers nothing, so no id is ever resolved or recorded.
- `previous_response_id(&ModelRequestConfig) -> Result<Option<String>>`, `conversation_key(&ModelRequestConfig) -> Result<Option<String>>`, `max_tool_calls(&ModelRequestConfig) -> Result<Option<u32>>` — typed accessors over `ModelRequestConfig::extra` using the keys `PREVIOUS_RESPONSE_ID_KEY` (`previous_response_id`), `CONVERSATION_KEY` (`conversation_id`) and `MAX_TOOL_CALLS_KEY` (`max_tool_calls`).
- `ConversationTurn` (`request`, `key`, `record(&dyn ResponsesSessionStore, Option<&str>)`), `resolve(&ModelRequestConfig, Option<&str>, &dyn ResponsesSessionStore)`, `apply_body_fields`, `is_client_side_key` and `reject_unsupported` are crate-internal.

`stream`:
- `ResponsesStreamState` (`new`, `handle_data(&str) -> Result<Vec<StreamChunk>>`, `finish() -> Vec<StreamChunk>`, `response_id() -> Option<&str>`) — one state machine per stream; tracks emitted text/reasoning deltas, streamed function calls by item id, the provider response id carried by any `response` object, and the terminal flag. The client emits exactly one `StreamChunk::Done`.

`mod`:
- `OpenAiResponsesClient` (`new(ModelClientConfig, ModelRequestConfig)`, `with_http_client`, `with_session_store`, `with_conversation_key`, `client_config`, `request_config`, `session_store`, `conversation_key`, `base_url`, `chat_path`, `impl Model`).
- Re-exports `build_responses_body`, `convert_messages`, `convert_tools`, `normalize_api_base`, `parse_responses_response`, `parse_responses_usage`, `ResponsesStreamState`, and the `state` surface (`ResponsesSessionStore`, `InMemoryResponsesSessionStore`, `NoopResponsesSessionStore`, `DEFAULT_CONVERSATION_CAPACITY`, `CONVERSATION_KEY`, `MAX_TOOL_CALLS_KEY`, `PREVIOUS_RESPONSE_ID_KEY`, `previous_response_id`, `conversation_key`, `max_tool_calls`).

## Constraints

- Only `openjiuwen_common::http::HttpClient` performs I/O; `SseDecoder`, `join_url`, `error_for_status`, `redacted_body`, `carries_payload`, `next_chunk`, `poll_budget`, and `read_error_body` are reused from the sibling modules, so the streaming timeout pattern matches `../openai/mod.rs` exactly.
- Every body is built and parsed by pure functions; tests assert exact JSON without a server.
- `api_key` stays a `SecretString`; provider error bodies are redacted and truncated before they reach an error message or a log.
- No new dependency and no feature gate: the module compiles and tests offline in every feature combination.
- Non-streaming `invoke` still streams: the account endpoint only answers event streams, so `invoke` forces `stream: true`, decodes the SSE events, and aggregates them with `StreamAccumulator` (Python `create_response` parity).
- Conversation state is per-client: `new` mounts a fresh `InMemoryResponsesSessionStore`, `with_session_store` swaps it (share an `Arc` to share history, use `NoopResponsesSessionStore` to keep none), and no process global exists. The store is in-memory and lost on restart; its capacity bounds the conversation keys it tracks.
- Request-level hints beat client-level ones: `extra["conversation_id"]` wins over `with_conversation_key`, and an explicit `extra["previous_response_id"]` always wins over the store. A call without any conversation key never touches the store.
- `max_tool_calls` is forwarded, never enforced: the service owns the limit. The transport validates the value so a bad one is a typed `Error::Validation` instead of a silently dropped body field.

## Wire contract

- `POST {base}/responses`; `Authorization: Bearer <key>`, `Accept: application/json`, `Content-Type: application/json`, plus `session_id` and `x-client-request-id` when `request.extra["session_id"]` is a non-blank string.
- Body: `model`, `instructions` (system/developer text joined with `"\n\n"`), `input` items, `store: false`; `temperature`/`top_p`/`max_output_tokens`/`stop` only when set; `tools`/`parallel_tool_calls: true`/`tool_choice` (default `"auto"`) only when tools are present; `include: ["reasoning.encrypted_content"]` when `extra.include_reasoning_encrypted_content` is `true`; `stream: true` for streaming.
- Conversation fields: `max_tool_calls` (a positive integer, from `extra["max_tool_calls"]`, a JSON number or numeric string) and `previous_response_id` (a non-blank string, an explicit `extra["previous_response_id"]` or the id the client's store holds for the resolved conversation key) are present only when configured. `conversation_id` and the client-wide conversation key are client-side hints and never reach the body, at either `extra` level (a hint nested in `extra_body` is dropped with a warning).
- `background: true` and hosted tools (`web_search*`, `file_search`, `computer`/`computer_use_preview`, `code_interpreter`, `image_generation`, `local_shell`/`shell`, `mcp`) are typed `Error::Unsupported` naming the live OpenAI Responses service; `background: false` and unfamiliar tool types (`custom`, vendor extensions) still pass through.
- Input items: user text → `{"role":"user","content":[{"type":"input_text",...}]}` (adjacent text parts merge); images → `{"type":"input_image","image_url","detail"?}`; assistant text → `{"role":"assistant","content":[{"type":"output_text",...}]}`; assistant tool calls → `{"type":"function_call","id","call_id","name","arguments"}`; tool turns → `{"type":"function_call_output","call_id","output"}` and are dropped without a call id; `File`/`Audio`/unknown parts are dropped.
- Tools: `{"type":"function","function":{name,description,parameters}}` → `{"type":"function","name","description","parameters"}` (no `strict`); other shapes pass through.
- Extras: `extra.extra_body` is flattened one level into the body root (outer keys win); `response_format` moves to `text.format` (an existing `text.format` wins); a consumed reasoning intent is not re-forwarded, and all remaining caller `extra` keys are merged last and win.
- Streaming events handled: `response.output_text.delta`/`response.refusal.delta` → content; `response.reasoning_text.delta`/`response.reasoning_summary_text.delta` → reasoning; `response.output_item.added` (`function_call`) → `ToolCallDelta` (id, name, initial arguments, index); `response.function_call_arguments.delta` → accumulated argument diff; `response.output_item.done` (`function_call`) → remaining argument suffix and `finish_reason = ToolCalls`; `response.completed`/`response.incomplete` → usage, `finish_reason` (`Stop`, `Length` when `status == "incomplete"`), then `Done`; `response.failed`/`error` → `Error::Llm`. Every other event is ignored, and malformed JSON frames are logged and skipped.
- Usage: `input_tokens|prompt_tokens`, `output_tokens|completion_tokens`, a summed total when the provider omits it, `*_tokens_details.cached_tokens` → `cache_tokens`, `*_tokens_details.reasoning_tokens` → `reasoning_tokens`.
- Errors: message precedence `error.message` → `error` string → `detail` → summarized body → `...failed with status N.`; the body is redacted and truncated; `401`/`403` → `Error::Config`, `408`/`504` → `Error::Timeout`, other non-2xx → `Error::Llm`.

## Test Strategy

- `tests.rs` (declared from `mod.rs`) drives the client end to end through `wiremock`: exact URL/headers/body for a simple user message, session headers, a pasted `/responses` suffix, streamed text/reasoning/usage assembly, a full function-call sequence, `response.failed`, a malformed frame, `401` → `Config`, `500` → `Llm`, `504` → `Timeout`, and factory dispatch (`api_mode: responses` on `openai`, the `openai_account` default base).
- The conversation suite in `tests.rs` covers `max_tool_calls` on the wire (present when set, absent by default, typed `Validation` with zero requests sent for a bad value), the two-call `previous_response_id` flow (`id` captured from the stream, follow-up send, explicit-id override of a seeded store), the streamed path recording its id for a client-wide conversation key, the conversation key never appearing on the wire (at either `extra` level), and `background`/hosted tools as typed `Unsupported`.
- `state.rs` unit tests cover the accessors' validation, the LRU bound/eviction and overwrite/blank-input rules, the noop store, key precedence, and recording rules.
- Pure conversion tests live beside their functions in `request.rs`, `response.rs`, and `stream.rs`: instructions/input projection with dropped parts and tool turns, tool flattening and `tool_choice`, reasoning/include/`extra_body`/`response_format` handling, every stream event, the captured response id, and both response/usage parsers.
- Everything is offline and deterministic: loopback `wiremock` servers, no credentials, no wall-clock ordering; environment mutations are serialized by the module mutex and restored.

## Known Gaps

- The non-streaming response parser (`parse_responses_response`) is public but no transport path uses it: `invoke` always streams, so a provider that ignores `stream: true` and answers one JSON object fails to decode.
- That parser maps every `status` other than `"incomplete"` to `FinishReason::Stop`, so a `failed`/`cancelled`/`queued` payload parses into an empty assistant message with no error (only the streaming path turns `response.failed` into `Error::Llm`).
- No support for `previous_response_id`/conversation state, `background` mode, `max_tool_calls`, built-in hosted tools (web search, file search, computer use), or structured `text.format` validation.
- Reasoning replay of `reasoning.encrypted_content` is not implemented: `include` is written, but returned encrypted items are dropped and never sent back on the next turn.
- Input parts other than text and images (`File`, `Audio`, unknown) are dropped rather than encoded, matching the Python projection.
- Multi-part image detail defaults to the provider default; no per-part `detail` normalization beyond passing the caller's value through.
