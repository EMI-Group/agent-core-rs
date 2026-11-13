# dashscope — image, speech, and video generation over the vendor HTTP API

## Intent

Port the Deferred DashScope media generation of the Python SDK (`generate_image`, `generate_speech`, `generate_video` on the OpenAI-compatible client) without the vendor `dashscope` SDK: the same HTTP surface is implemented directly on `openjiuwen_common::http::HttpClient`.
The client is DashScope-specific; the Python `endpoint_profile == "dashscope"` gate lives at the call site in Rust (construct it for a `ModelClientConfig` whose provider is `qwen`/`dashscope`).

## API Surface

- `DashScopeConfig` — `api_base` (`https://dashscope.aliyuncs.com`), `api_key: SecretString`, `timeout` (360 s), `poll_interval` (2 s), `task_timeout` (10 min), with builders, `from_env` (`OPENJIUWEN_DASHSCOPE_API_KEY` → `DASHSCOPE_API_KEY`, `OPENJIUWEN_DASHSCOPE_API_BASE` → `DASHSCOPE_API_BASE`) and `from_client_config(&ModelClientConfig)`.
- `DashScopeMediaClient` — `new`, `with_http_client`, `config`, `generate_image`, `generate_speech`, `generate_video`.
- Requests: `ImageRequest`, `SpeechRequest`, `VideoRequest` (field-per-vendor-knob, builders, documented Python-shaped defaults).
- Responses: `ImageGenerationResponse` (URLs split from inline base64), `AudioGenerationResponse`, `VideoGenerationResponse`, `VideoTask` + `VideoTaskStatus` (tolerant manual serde).
- Pure wire functions: `build_image_body`, `build_speech_body`, `build_video_body`, `parse_image_response`, `parse_speech_response`, `parse_video_task`, `parse_video_response`, `last_user_text`, `task_path`, and the path/constant set.

## Constraints

- All I/O through `HttpClient` (bearer auth, per-request timeout); never `reqwest` directly.
- Errors follow the `llm` module: blank key/base → `Config`, bad parameters → `Validation`, 401/403 → `Config`, 408/504 and task budget → `Timeout`, everything else → `Llm`.
- Vendor `code`/`message` are surfaced truncated to 500 characters; an unrecognized body is rendered through `security::redact_str` and never echoed.
- Prompts are user content and are never logged; logs are `tracing` with target `openjiuwen_foundation::llm::dashscope` and carry only model/task id/status.
- Video polling treats every non-terminal status (including `Unknown`) as pending and is bounded by `task_timeout`.

## Design Decisions

- Builders own the validation (blank model, blank prompt, blank image, video `size` XOR `resolution`) so the client stays a transport wrapper and tests assert bodies without a server.
- Video sizing follows the vendor split: image-to-video (`img_url`) accepts `resolution` only, text-to-video accepts `size` only.
- A payload carrying `output.task_status` is never treated as a general vendor error; a `FAILED` task is reported by the task path so the error names the task id.
- Attachment fields (`images_base64`, `audio_data`, `video_data`) hold inline payloads only; result URLs are never downloaded.

## Test Strategy

- `tests.rs` covers exact request bodies, response parsing, the full async video flow (`PENDING` → `RUNNING` → `SUCCEEDED`), a `FAILED` task, a bounded task timeout, non-2xx mapping, redacted bodies, and construction validation, all against `wiremock` on loopback.
- `config.rs` owns the `from_env` precedence and serde round-trip tests behind a process-environment mutex with restore.

## Known Gaps

- No streaming media, no result-URL download, no `X-DashScope-WorkSpace` scoping, no voice/language allow-list, no local-file or `oss://` inputs, no per-call `**kwargs` escape hatch.

## Routing Table

- `./config.rs` → `DashScopeConfig`, env precedence, `ModelClientConfig` bridge
- `./types.rs` → request/response types, `VideoTaskStatus`
- `./wire.rs` → body builders, response parsers, failure rendering
- `./client.rs` → `DashScopeMediaClient` and the video polling loop
- `./tests.rs` → wiremock, parser, and construction suites
