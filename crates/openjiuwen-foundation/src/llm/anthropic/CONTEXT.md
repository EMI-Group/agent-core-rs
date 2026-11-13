# anthropic — Anthropic Messages transport

## Intent

The Anthropic Messages implementation of the [`Model`](../mod.rs) seam: one client, the OpenAI-shape → Anthropic request projection, the response parser, and the event-oriented SSE stream.
Ports `openjiuwen/core/foundation/llm/model_clients/anthropic_model_client.py`.

## API Surface

- `AnthropicClient` — `new(ModelClientConfig, ModelRequestConfig)`, `with_http_client`, `client_config`, `request_config`, `base_url`, `impl Model` (`invoke`, `stream`, `provider`, `model_name`).
- `build_messages_body(&[BaseMessage], &ModelRequestConfig, stream) -> Result<Value>` — pure request projection.
- `normalize_api_base(&str) -> String` — strips a pasted `/v1/messages` or `/v1` suffix.
- `parse_messages_response`, `parse_anthropic_usage`, `AnthropicStreamState` — response and stream conversion.

## Request Projection (`request.rs`)

`system` is always a block array, one `{"type": "text", "text": …}` per `system`/developer turn, in order.
A single system turn yields a one-element array; a request with no (or only empty) system turn omits `system` entirely.
Turns are never joined into one string, never `<system>`-prefixed, and never dropped; non-text parts on a system turn have no destination and are skipped.
Assistant `tool_calls` become `tool_use` blocks with a parsed JSON `input` (`{"_raw_arguments": …}` for unparseable text), consecutive `tool` turns merge into one `user` message of `tool_result` blocks (flushed before every assistant/user/system turn), images become base64 or URL `image` blocks, and OpenAI function schemas become `{name, description, input_schema}`.
`request.extra` is merged last and wins, except the two cache-control knobs, which are consumed and never forwarded.

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
- The projection is pure: no I/O, no panics on caller input, no provider-specific extra fields beyond `request.extra`.
- Files stay ≤1000 lines; the projection lives in `convert.rs`, the anchors in `cache.rs`, the body assembly in `request.rs`.

## Routing Table

- `./mod.rs` → client, `Model` impl, streaming loop, module wiring
- `./request.rs` → body assembly, `max_tokens` caps, `normalize_api_base`
- `./convert.rs` → message projection, mid-conversation system partition, tools/tool_choice, images
- `./cache.rs` → cache-control marker, the three anchors, transient-tail detection, `extra` knobs
- `./response.rs` → non-streaming response parsing (`content` blocks → `BaseMessage`, `stop_reason`, usage)
- `./stream.rs` → SSE event state machine (`content_block_*`, `message_*`, tool-input buffering)
- `./request/tests.rs` → body projection tests
- `./request/cache_tests.rs` → cache-anchor tests
- `./tests.rs` → wiremock client tests (headers, body, error mapping, streaming)

## Test Strategy

`request/tests.rs` and `request/cache_tests.rs` assert exact JSON for every projection and anchor: single/multiple system turns, marker placement on tool_use/tool_result/image/plain-text anchors, the transient-tail skip, `disable_cache_control`, `cache_control_ttl`, and inline-vs-hoisted system placement; `cache.rs` unit-tests the marker/walk-back primitives; `tests.rs` asserts the wire body, headers, retry/auth error mapping, and streamed assembly against wiremock. No test needs a network or a credential.

## Known Gaps

- No `cache_control` breakpoints beyond the three automatic anchors (no per-message caller control, no `clear`/`auto` caching modes).
- `thinking` blocks are ignored in non-streaming responses; only streaming maps `thinking_delta` onto `reasoning`.
- Mid-conversation system turns are text-only, and the supported-family list is a fixed substring match (no capability query).
