# anthropic — Anthropic Messages transport

## Intent

The Anthropic Messages implementation of the [`Model`](../mod.rs) seam: one client, the OpenAI-shape → Anthropic request projection, the response parser, and the event-oriented SSE stream.
Ports `openjiuwen/core/foundation/llm/model_clients/anthropic_model_client.py`.

## API Surface

- `AnthropicClient` — `new(ModelClientConfig, ModelRequestConfig)`, `with_http_client`, `client_config`, `request_config`, `base_url`, `stream_with_state`, `complete_turn`, `impl Model` (`invoke`, `stream`, `provider`, `model_name`).
- `AnthropicClient::stream_with_state(messages, request) -> (BoxStream<'static, Result<StreamChunk>>, Arc<Mutex<AnthropicStreamState>>)` — the stream plus the accumulator that preserves its thinking blocks; `Model::stream` is `.0` only.
- `AnthropicClient::complete_turn(&[BaseMessage], &ModelRequestConfig) -> Result<AssistantTurn>` — drains the stream, accumulates it, and attaches the preserved thinking blocks to the message.
- `AnthropicStreamState` — `new`, `handle_event`, `finish`, `message_id`, `model`, `thinking_blocks`, `apply_thinking_blocks`.
- `build_messages_body(&[BaseMessage], &ModelRequestConfig, stream) -> Result<Value>` — pure request projection for the `anthropic` provider.
- `build_messages_body_with_provider(&[BaseMessage], &ModelRequestConfig, stream, provider, api_base: Option<&str>) -> Result<Value>` — the same projection with the client's provider name and endpoint, which select the reasoning profile.
- `build_messages_body_with_endpoint(&[BaseMessage], &ModelRequestConfig, stream, provider, endpoint_profile: Option<&str>, api_base: Option<&str>) -> Result<Value>` — the same projection with the client's `endpoint_profile` as well; the client calls this one, and `_with_provider` delegates with no profile.
- `normalize_api_base(&str) -> String` — strips a pasted `/v1/messages` or `/v1` suffix.
- `parse_messages_response`, `parse_anthropic_usage`, `AnthropicStreamState` — response and stream conversion.
- `THINKING_BLOCKS_METADATA_KEY` (`"anthropic_thinking_blocks"`) — the `BaseMessage::metadata` key holding the raw `thinking`/`redacted_thinking` blocks preserved from a response.

## Extended Thinking (`response.rs`, `stream.rs`, `convert.rs`)

A `thinking` block appends its `thinking` text to the assistant message's `reasoning_content` (adjacent blocks concatenate in order); `reasoning_content` stays `None` when no thinking block carried text.
Every replayable block — a `thinking` block with a non-empty `signature`, a `redacted_thinking` block with non-empty `data` — is preserved verbatim, in response order, as a JSON array under `BaseMessage::metadata[THINKING_BLOCKS_METADATA_KEY]`.
A block without that provider proof is dropped with a `tracing::warn!` and never stored: the client never invents a signature.
One rule decides replayability for both paths: `response::replayable_block` validates and returns the verbatim block, and `response::attach_thinking_blocks` writes the metadata key, so a streamed turn and an invoked turn carry identical metadata.
The assistant projection reads the same key and emits those blocks first, verbatim and in order, ahead of the text and `tool_use` blocks, because the Messages API requires the original signed blocks on a follow-up `tool_use` turn.
A turn without survivable blocks keeps the plain string `content` it always had; a turn with them always sends a block array.
The key is caller-writable, so the projection re-validates every entry (`type` plus `signature`/`data`) and silently drops anything else instead of forwarding a block the API would reject.

## Streamed Thinking (`stream.rs`)

`AnthropicStreamState` buffers one block per content-block index: `content_block_start` opens a `thinking`/`redacted_thinking` block (taking any `thinking`, `signature` or `data` it already carries), `thinking_delta` appends text, `signature_delta` appends the signature, a `data` delta appends redacted data, and `content_block_stop` seals the block.
A sealed block is validated with the shared `replayable_block` rule and kept only when the provider vouched for it; `finish` (and `message_stop`) seals anything the server never stopped, so a fully signed block still survives a truncated event list while an unsigned one is dropped with the same warning.
`thinking_blocks()` returns the preserved blocks by block index (stream order, not stop order) and `apply_thinking_blocks(&mut BaseMessage)` writes them under `THINKING_BLOCKS_METADATA_KEY`; `AnthropicClient::complete_turn` is `StreamAccumulator` + that call, so a streamed thinking turn is replayed exactly like an invoked one.
The thinking text keeps flowing to `StreamChunk::Delta::reasoning` (the stream chunk surface is unchanged and carries no metadata); the shared state lives in the returned `Arc<Mutex<AnthropicStreamState>>` instead.

## Request Projection (`request.rs`)

`system` is always a block array, one `{"type": "text", "text": …}` per `system`/developer turn, in order.
A single system turn yields a one-element array; a request with no (or only empty) system turn omits `system` entirely.
Turns are never joined into one string, never `<system>`-prefixed, and never dropped; non-text parts on a system turn have no destination and are skipped.
Assistant `tool_calls` become `tool_use` blocks with a parsed JSON `input` (`{"_raw_arguments": …}` for unparseable text), consecutive `tool` turns merge into one `user` message of `tool_result` blocks (flushed before every assistant/user/system turn), images become base64 or URL `image` blocks, and OpenAI function schemas become `{name, description, input_schema}`.
`request.extra` is merged last and wins, except the reasoning intent and the two cache-control knobs, which are consumed and never forwarded.

## Reasoning Plan (`request.rs`)

A provider-neutral intent in `request.extra["reasoning"]` (`{mode, effort, budget_tokens}`, also `true`/`false` or a mode/effort string) is parsed with `ReasoningConfig::from_value` and resolved with `resolve_reasoning_plan_with_endpoint(ReasoningProtocol::AnthropicMessages, provider, endpoint_profile, api_base, model, &intent)`.
An `endpoint_profile` of `vllm` bypasses the profile table and writes `enable_thinking` plus `chat_template_kwargs.enable_thinking` (with an optional `thinking_budget`) instead of the `thinking` object.
`provider`, `endpoint_profile`, and `api_base` are the client's resolved values (`AnthropicClient::invoke`/`stream` pass `client_config.provider`, `client_config.endpoint_profile`, and the normalized base URL), so a generic provider name defers to an API-base host hint (a DashScope-hosted Claude resolves the DashScope Anthropic wire) and a non-generic profile wins.
`plan.sdk_params` are merged first, then `plan.extra_body`, both at the body top level because the Messages API takes `thinking`, `output_config`, `reasoning_effort` there; `apply_reasoning_plan` keeps a caller key that is already present.
`request.extra` is merged afterwards, so a caller's own keys still win, and `reasoning` is never forwarded.
Examples: `{"mode":"enabled","effort":"medium"}` on `claude-sonnet-4-5*` → `thinking: {"type":"enabled","budget_tokens":8192}` (the `anthropic_manual` effort map); `{"mode":"disabled"}` → `thinking: {"type":"disabled"}`; an unlisted model with `{"mode":"enabled","budget_tokens":2048}` → `thinking: {"type":"enabled","budget_tokens":2048}`; `{"mode":"enabled","effort":"high"}` on an unlisted model → `thinking: {"type":"adaptive"}` plus `output_config: {"effort":"high"}`.

Legacy precedence: when `extra` already carries any of `reasoning_effort`, `thinking`, `enable_thinking`, `thinking_budget`, `thinking_strategy`, `chat_template_kwargs` — at the top level or inside a nested `extra_body` object — no plan is resolved or applied, so the caller's explicit vendor control is the only encoding sent (a `tracing::debug!` records the skip).
A present `reasoning` key that is not an intent (`null` means "no intent"; a number, an array, or an object with foreign keys like `{"thinking": …}`) is `Error::Validation`, never a panic and never a silently dropped key.
Without a `reasoning` key the produced body is unchanged, and `build_messages_body` is exactly `build_messages_body_with_provider(…, "anthropic", None)`, which is `build_messages_body_with_endpoint(…, "anthropic", None, None)`.

## Cache Breakpoints (`cache.rs`)

Breakpoints are always on: no config flag, no beta header.
Exactly three anchors, each `cache_control: {"type": "ephemeral"}`:
1. the last entry of the top-level `tools` array;
2. the last cacheable block (`text`, `image`, `document`, `tool_use`, `tool_result`) of the `system` block array;
3. the last cacheable block of the last *stable* message — the walk starts at the tail, skips a `metadata["transient"] == true` tail message (read from the original `BaseMessage`s) when another message precedes it, skips inline `system` messages, and continues backwards over messages with no cacheable block.

A chosen anchor whose `content` is a plain, non-empty string is materialized as a one-element `text` block array so the marker has somewhere to live; an empty string is not an anchor.
Because the neutral projection keeps text content as strings, this conversion is what lets a plain text conversation carry a breakpoint at all.

Opt-out seam (`ModelRequestConfig::extra`, no signature change): `disable_cache_control: true` writes no anchor (the system array stays a block array), and `cache_control_ttl: "1h"` puts `"ttl": "1h"` on all three markers (default: no `ttl` key).

## Mid-Conversation `system`

The Messages API rejects `role: "system"` inside `messages`, so system turns are laid out in place first and partitioned afterwards.
A turn stays inline only when the model id contains (case-insensitive) `claude-opus-5`, `claude-opus-4-8`, `claude-fable-5` or `claude-mythos-5`, it is not `messages[0]`, the previous message is a `user` turn, and it is the last message or followed by an `assistant` turn; otherwise it is hoisted, appended in order to the top-level array.
Inline turns carry plain text as their `content`; neighbours are read from the pre-partition list so adjacent system turns both fall back.

## Constraints

- All HTTP goes through `openjiuwen_common::http::HttpClient`; headers are `x-api-key` (never `authorization`) and `anthropic-version`.
- `max_tokens` is required: `request.max_tokens` else 32768, capped at 8192 for `claude-3-5*`/`claude-3.5*` and 4096 for `claude-3-opus|sonnet|haiku`.
- Errors are `openjiuwen_common::error::{Error, Result}`: 401/403 → `Config`, 408/504 → `Timeout`, other non-2xx → `Llm` with a redacted body; an unrepresentable message is `Validation`.
- The projection is pure: no I/O, no panics on caller input, and no provider-specific extra fields beyond `request.extra` and the resolved reasoning plan.
- Files stay ≤1000 lines; the projection lives in `convert.rs`, the anchors in `cache.rs`, the body assembly in `request.rs`.

## Routing Table

- `./mod.rs` → client, `Model` impl, streaming loop, module wiring
- `./request.rs` → body assembly, reasoning-plan wiring and legacy-control precedence, `max_tokens` caps, `normalize_api_base`
- `./convert.rs` → message projection, mid-conversation system partition, tools/tool_choice, images
- `./cache.rs` → cache-control marker, the three anchors, transient-tail detection, `extra` knobs
- `./response.rs` → non-streaming response parsing (`content` blocks → `BaseMessage`, preserved thinking blocks, `stop_reason`, usage)
- `./stream.rs` → SSE event state machine (`content_block_*`, `message_*`, tool-input buffering)
- `./stream/thinking.rs` → extended-thinking block buffering and sealing (`ThinkingBlocks`) plus its unit tests
- `./request/tests.rs` → body projection tests
- `./request/cache_tests.rs` → cache-anchor tests
- `./request/reasoning_tests.rs` → reasoning-plan wiring tests (`thinking`/`output_config` shapes, legacy precedence, validation)
- `./tests.rs` → wiremock client tests (headers, body, error mapping, streaming, streamed thinking-block replay)

## Test Strategy

`request/tests.rs` and `request/cache_tests.rs` assert exact JSON for every projection and anchor: single/multiple system turns, marker placement on tool_use/tool_result/image/plain-text anchors, the transient-tail skip, `disable_cache_control`, `cache_control_ttl`, and inline-vs-hoisted system placement.
`request/reasoning_tests.rs` asserts the reasoning body shapes for a profiled model (`claude-sonnet-4-5*`), an unlisted model, and a host-hint-routed vendor, plus every legacy control key, the nested `extra_body` form, the consumed keys, and the validation errors for malformed intents.
`cache.rs` unit-tests the marker/walk-back primitives; `stream/thinking.rs` unit-tests thinking-block accumulation (signed, unsigned, redacted `data` fragments, stop order vs. block order, sealing at `finish`); `tests.rs` asserts the wire body, headers, retry/auth error mapping, streamed assembly, that `invoke`/`stream` send the resolved reasoning plan, and that a streamed thinking turn reaches the next request body with its signature (and that an unsigned one reaches it not at all) against wiremock.
No test needs a network or a credential.

## Known Gaps

- No `cache_control` breakpoints beyond the three automatic anchors (no per-message caller control, no `clear`/`auto` caching modes).
- Mid-conversation system turns are text-only, and the supported-family list is a fixed substring match (no capability query).
