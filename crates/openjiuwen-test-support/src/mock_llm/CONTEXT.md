# mock_llm — loopback OpenAI-compatible server

## Intent

An axum server bound to `127.0.0.1:0` that speaks the OpenAI chat-completions wire format so the real `OpenAiCompatibleClient` (transport, SSE decoder, error mapping, retries) can be tested offline.
It is the Rust replacement for the `httpx.MockTransport` handler pattern in the Python tests, plus the scripted-queue idea of `tests/unit_tests/fixtures/mock_llm.py` (`set_responses` → `push`).
The public API is exactly the `MockLlmServer`, `MockLlmServerBuilder`, `ScriptedResponse`, `MockHttpError`, and `RecordedRequest` items specified in the crate `CONTEXT.md`; re-export them from `mod.rs` and from the crate root.

## API Surface

- Server handle: `MockLlmServer` — `start`/`try_start`/`with_script`/`builder`, `base_url`, `endpoint`, `addr`, `client_config`, `request_config`, `openai_client`, `push`/`push_many`/`clear_script`/`set_fallback`/`queued`, `request_count`/`requests`/`last_request`, `wait_for_requests`/`wait_for_requests_timeout`, `shutdown`.
- Builder: `MockLlmServerBuilder` — `script`, `item`, `fallback`, `id_prefix`, `default_model`, `chunk_delay`, `start`/`try_start`.
- Script entries: `ScriptedResponse::{Completion, Chunks, Error}` with `text`, `completion`, `chunks`, `error`.
- Error payloads: `MockHttpError { status, body }` with `new`, `with_body`, `rate_limited`, `unauthorized`, `server_error`, `not_found`.
- Recording: `RecordedRequest` with `header`, `messages`, `texts_of_role`, `user_texts`.

## Constraints

- Bind loopback only; never fall back to a fixed port; the OS assigns the port and `addr()`/`base_url()` report it.
- No `unwrap`/`expect`/`panic` outside `MockLlmServer::start`/`try_start` bind failure, where `start` panics with a clear message and `try_start` returns `Result`.
- No `MutexGuard` held across `.await`; queue pops, recording pushes, and counter updates happen in short synchronous blocks.
- The handler must never fail the whole server task on a bad request: malformed JSON is a 400, unknown route is a 404 JSON envelope.
- Deterministic output only: `created: 0`, counter-based ids `chatcmpl-{id_prefix}{seq:04}`, no clocks, no randomness.
- SSE frames must be exactly `data: <json>\n\n`, terminated by `data: [DONE]\n\n`; `StreamChunk::Done` never becomes its own JSON frame.
- Usage is emitted as a dedicated frame with `choices: []` when the scripted usage is non-empty; the usage object always uses `prompt_tokens`/`completion_tokens`/`total_tokens` plus `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` when non-zero.
- `MockLlmServer::openai_client` builds an `OpenAiCompatibleClient` with provider `openai`, api key `test-api-key`, `max_retries = 0`, so a scripted 5xx fails immediately.

## Module Layout

- `mod.rs` — handle (`MockLlmServer`), builder, re-exports; `Drop` aborts the server task when `shutdown` was not awaited.
- `state.rs` — `ServerState`: script queue, fallback, recordings, sequence counter, `Notify`, config (id prefix, default model, chunk delay); plus the synchronous pop/record/pending helpers.
- `wire.rs` — OpenAI request parsing (tolerant serde) and response projection: completion body, SSE frame generation from `StreamChunk`, error envelope, health body.
- `routes.rs` — axum `Router`, `chat_completions` handler (read `Bytes` → parse → pop → record → respond), `health`, JSON 404 fallback, SSE body construction with the optional per-frame delay.
- `tests.rs` — unit tests for wire projection and state behavior (`#[cfg(test)]`).

## Test Strategy

Unit tests in `tests.rs` cover: completion body shape, delta/finish/usage/`[DONE]` frame order, tool-call delta shape, queue exhaustion fallback, deterministic ids, and error bodies.
End-to-end coverage (real client, real socket) lives in `../../tests/mock_server.rs`; do not bind sockets in unit tests unless the test is explicitly an async server test.
