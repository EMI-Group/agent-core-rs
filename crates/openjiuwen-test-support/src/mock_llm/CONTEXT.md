# mock_llm — loopback OpenAI-compatible server

## Intent

An axum server bound to `127.0.0.1:0` that speaks the OpenAI chat-completions wire format so the real `OpenAiCompatibleClient` (transport, SSE decoder, error mapping, retries) can be tested offline.
It is the Rust replacement for the `httpx.MockTransport` handler pattern in the Python tests, plus the scripted-queue idea of `tests/unit_tests/fixtures/mock_llm.py` (`set_responses` → `push`).
The public API is the `MockLlmServer`, `MockLlmServerBuilder`, `ScriptedResponse`, `MockHttpError`, and `RecordedRequest` items specified in the crate `CONTEXT.md`, plus `TEST_API_KEY`; re-export them from `mod.rs` and from the crate root.

## API Surface

- Server handle: `MockLlmServer` — `start`/`try_start`/`with_script`/`builder`, `base_url`, `endpoint`, `addr`, `client_config`, `request_config`, `openai_client`, `push`/`push_many`/`clear_script`/`set_fallback`/`queued`, `request_count`/`requests`/`last_request`, `wait_for_requests` (bounded, `Result`) / `wait_for_requests_timeout`, `connections`/`live_connections`, `shutdown`.
- Connection tasks: `ConnectionTasks` with `live`/`spawned`; the handle `MockLlmServer::connections` returns outlives the server, so a test can assert that dropping it left no task running.
- Builder: `MockLlmServerBuilder` — `script`, `item`, `fallback`, `id_prefix`, `default_model`, `chunk_delay`, `start`/`try_start`.
- Script entries: `ScriptedResponse::{Completion, Chunks, Error}` with `text`, `completion`, `chunks`, `error`.
- Error payloads: `MockHttpError { status, body }` with `new`, `with_body`, `rate_limited`, `unauthorized`, `server_error`, `not_found`.
- Recording: `RecordedRequest` with `header`, `messages`, `texts_of_role`, `user_texts`.
- API key: `pub const TEST_API_KEY: &str = "test-api-key"` — the key `client_config()` configures every client with.

## Constraints

- Bind loopback only; never fall back to a fixed port; the OS assigns the port and `addr()`/`base_url()` report it.
- No `unwrap`/`expect`/`panic` outside the documented panic surface: `MockLlmServer::start`, `MockLlmServer::with_script`, and `MockLlmServerBuilder::start` panic with a clear bind-failure message and each carries a `# Panics` section; `MockLlmServer::try_start` and `MockLlmServerBuilder::try_start` return `Result` instead.
- No `MutexGuard` held across `.await`; queue pops, recording pushes, and counter updates happen in short synchronous blocks.
- The handler must never fail the whole server task on a bad request: a body that is not a JSON object is a 400 that pops nothing but is still recorded (`body: Value::Null`, no response id), and an unknown route is a 404 JSON envelope.
- A wrong method on a known route is answered with the same JSON 404 (never a 405) and is not recorded.
- Only chat-completion traffic is recorded: `/health` and the JSON 404s never appear in `requests()`/`request_count()`.
- Deterministic output only: `created: 0`, counter-based ids `chatcmpl-{id_prefix}{seq:04}`, no clocks, no randomness.
- The sequence counter advances only for popped script entries (the first popped answer is `chatcmpl-{id_prefix}0001`); a fallback answer reuses the current value without advancing it, so an empty script answers `chatcmpl-{id_prefix}0000` and a fallback after N pops repeats the N-th id.
- `MockLlmServerBuilder::script` replaces the queued script while `item` appends, so `.script([a, b]).item(c)` serves `a`, `b`, `c`.
- SSE frames must be exactly `data: <json>\n\n`, terminated by `data: [DONE]\n\n`; `StreamChunk::Done` never becomes its own JSON frame.
- Usage is emitted as a dedicated frame with `choices: []` when the scripted usage is non-empty; the usage object always uses `prompt_tokens`/`completion_tokens`/`total_tokens` plus `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` when non-zero.
- `MockLlmServer::openai_client` builds an `OpenAiCompatibleClient` with provider `openai`, api key `test-api-key`, `max_retries = 0`, so a scripted 5xx fails immediately.
- The server owns its accept loop (a plain `TcpListener` loop feeding `hyper_util`'s auto connection builder) instead of `axum::serve`, because `axum::serve` spawns one task per connection with no handle; every connection task is recorded in `ConnectionTasks`.
- `shutdown` flips the graceful signal the accept loop forwards to each connection, awaits the accept task with a 5 s bound, waits up to 5 s for the connections to drain and aborts whatever is left; `Drop` aborts the accept task and every connection task, so no task outlives the server.

## Module Layout

- `mod.rs` — handle (`MockLlmServer`), `ConnectionTasks`, accept loop and per-connection serving, builder, re-exports; `Drop` aborts the accept task and every connection task when `shutdown` was not awaited.
- `state.rs` — `ServerState`: script queue, fallback, recordings, sequence counter, `Notify`, config (id prefix, default model, chunk delay); plus the synchronous pop/record/pending helpers.
- `wire.rs` — OpenAI request parsing (tolerant serde) and response projection: completion body, SSE frame generation from `StreamChunk`, error envelope, health body; `ChatCompletionRequest` parses `messages`/`tools`/`tool_choice`/`stream_options` for tolerance only (only `model` and `stream` are read), hence the documented `#[allow(dead_code)]`.
- `routes.rs` — axum `Router`, `chat_completions` handler (read `Bytes` → parse → pop → record → respond), `health`, JSON 404 fallback, SSE body construction with the optional per-frame delay.
- `../internal.rs` — crate-private helpers shared with the scripted model: `lock` (poison-recovering mutex), `FUNCTION_KIND`, and `DEFAULT_FALLBACK_TEXT`; no public API.
- `tests.rs` — unit tests for wire projection and state behavior (`#[cfg(test)]`).

## Test Strategy

Unit tests in `tests.rs` cover the completion body shape (content, reasoning, tool calls, every finish reason), the delta/finish/usage/`[DONE]` frame order, tool-call deltas, `usage_json` detail merging, the queue/fallback rules, the deterministic and fallback ids, the error and 404/malformed envelopes, the recording accessors, and a socket-free `wait_for_requests` wake-up test.
End-to-end coverage through the real client on a real socket lives in `../../tests/mock_server.rs`, so unit tests bind no sockets unless they are explicitly async server tests.
