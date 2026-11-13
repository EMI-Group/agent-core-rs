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
- `build_responses_body(&[BaseMessage], &ModelRequestConfig, stream: bool) -> Result<Value>` — the full wire body; `Error::Config` on a blank `model_name`.

`response`:
- `parse_responses_response(&Value) -> Result<ChatResponse>` — parses the non-streaming response object (message `output_text`/`text` parts, `reasoning.summary[].text`, top-level `output_text`, `function_call` items).
- `parse_responses_usage(&Value) -> Usage` — `input_tokens|prompt_tokens`, `output_tokens|completion_tokens`, cache and reasoning detail counters.

`stream`:
- `ResponsesStreamState` (`new`, `handle_data(&str) -> Result<Vec<StreamChunk>>`, `finish() -> Vec<StreamChunk>`) — one state machine per stream; tracks emitted text/reasoning deltas, streamed function calls by item id, and the terminal flag. The client emits exactly one `StreamChunk::Done`.

`mod`:
- `OpenAiResponsesClient` (`new(ModelClientConfig, ModelRequestConfig)`, `with_http_client`, `client_config`, `request_config`, `base_url`, `chat_path`, `impl Model`).
- Re-exports `build_responses_body`, `convert_messages`, `convert_tools`, `normalize_api_base`, `parse_responses_response`, `parse_responses_usage`, `ResponsesStreamState`.

## Constraints

- Only `openjiuwen_common::http::HttpClient` performs I/O; `SseDecoder`, `join_url`, `error_for_status`, `redacted_body`, `carries_payload`, `next_chunk`, `poll_budget`, and `read_error_body` are reused from the sibling modules, so the streaming timeout pattern matches `../openai/mod.rs` exactly.
- Every body is built and parsed by pure functions; tests assert exact JSON without a server.
- `api_key` stays a `SecretString`; provider error bodies are redacted and truncated before they reach an error message or a log.
- No new dependency and no feature gate: the module compiles and tests offline in every feature combination.
- Non-streaming `invoke` still streams: the account endpoint only answers event streams, so `invoke` forces `stream: true`, decodes the SSE events, and aggregates them with `StreamAccumulator` (Python `create_response` parity).

## Wire contract

- `POST {base}/responses`; `Authorization: Bearer <key>`, `Accept: application/json`, `Content-Type: application/json`, plus `session_id` and `x-client-request-id` when `request.extra["session_id"]` is a non-blank string.
- Body: `model`, `instructions` (system/developer text joined with `"\n\n"`), `input` items, `store: false`; `temperature`/`top_p`/`max_output_tokens`/`stop` only when set; `tools`/`parallel_tool_calls: true`/`tool_choice` (default `"auto"`) only when tools are present; `include: ["reasoning.encrypted_content"]` when `extra.include_reasoning_encrypted_content` is `true`; `stream: true` for streaming.
- Input items: user text → `{"role":"user","content":[{"type":"input_text",...}]}` (adjacent text parts merge); images → `{"type":"input_image","image_url","detail"?}`; assistant text → `{"role":"assistant","content":[{"type":"output_text",...}]}`; assistant tool calls → `{"type":"function_call","id","call_id","name","arguments"}`; tool turns → `{"type":"function_call_output","call_id","output"}` and are dropped without a call id; `File`/`Audio`/unknown parts are dropped.
- Tools: `{"type":"function","function":{name,description,parameters}}` → `{"type":"function","name","description","parameters"}` (no `strict`); other shapes pass through.
- Extras: `extra.extra_body` is flattened one level into the body root (outer keys win); `response_format` moves to `text.format` (an existing `text.format` wins); a consumed reasoning intent is not re-forwarded, and all remaining caller `extra` keys are merged last and win.
- Streaming events handled: `response.output_text.delta`/`response.refusal.delta` → content; `response.reasoning_text.delta`/`response.reasoning_summary_text.delta` → reasoning; `response.output_item.added` (`function_call`) → `ToolCallDelta` (id, name, initial arguments, index); `response.function_call_arguments.delta` → accumulated argument diff; `response.output_item.done` (`function_call`) → remaining argument suffix and `finish_reason = ToolCalls`; `response.completed`/`response.incomplete` → usage, `finish_reason` (`Stop`, `Length` when `status == "incomplete"`), then `Done`; `response.failed`/`error` → `Error::Llm`. Every other event is ignored, and malformed JSON frames are logged and skipped.
- Usage: `input_tokens|prompt_tokens`, `output_tokens|completion_tokens`, a summed total when the provider omits it, `*_tokens_details.cached_tokens` → `cache_tokens`, `*_tokens_details.reasoning_tokens` → `reasoning_tokens`.
- Errors: message precedence `error.message` → `error` string → `detail` → summarized body → `...failed with status N.`; the body is redacted and truncated; `401`/`403` → `Error::Config`, `408`/`504` → `Error::Timeout`, other non-2xx → `Error::Llm`.

## Test Strategy

- `tests.rs` (declared from `mod.rs`) drives the client end to end through `wiremock`: exact URL/headers/body for a simple user message, session headers, a pasted `/responses` suffix, streamed text/reasoning/usage assembly, a full function-call sequence, `response.failed`, a malformed frame, `401` → `Config`, `500` → `Llm`, `504` → `Timeout`, and factory dispatch (`api_mode: responses` on `openai`, the `openai_account` default base).
- Pure conversion tests live beside their functions in `request.rs`, `response.rs`, and `stream.rs`: instructions/input projection with dropped parts and tool turns, tool flattening and `tool_choice`, reasoning/include/`extra_body`/`response_format` handling, every stream event, and both response/usage parsers.
- Everything is offline and deterministic: loopback `wiremock` servers, no credentials, no wall-clock ordering; environment mutations are serialized by the module mutex and restored.

## Known Gaps

- The non-streaming response parser (`parse_responses_response`) is public but no transport path uses it: `invoke` always streams, so a provider that ignores `stream: true` and answers one JSON object fails to decode.
- No support for `previous_response_id`/conversation state, `background` mode, `max_tool_calls`, built-in hosted tools (web search, file search, computer use), or structured `text.format` validation.
- Reasoning replay of `reasoning.encrypted_content` is not implemented: `include` is written, but returned encrypted items are dropped and never sent back on the next turn.
- Input parts other than text and images (`File`, `Audio`, unknown) are dropped rather than encoded, matching the Python projection.
- Multi-part image detail defaults to the provider default; no per-part `detail` normalization beyond passing the caller's value through.
