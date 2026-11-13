# otel/langfuse

## Intent

Langfuse integration compiled in every feature mode: an OTLP preset for `{host}/api/public/otel` and a REST ingestion client for `POST {host}/api/public/ingestion`.
Ports the Langfuse target of `openjiuwen/extensions/observability/exporters/langfuse.py` (v0.1.20) onto the Rust OTLP/HTTP JSON exporter plus the v2/v3 ingestion API.
Requires a Langfuse project (public + secret key) and network access to the host; Langfuse Cloud (`https://cloud.langfuse.com`) is the default, a self-hosted instance is set with `with_host` or `LANGFUSE_HOST`/`LANGFUSE_BASEURL`.
Only OTLP/HTTP **JSON** is available offline (`gRPC` needs `tonic`/`prost`); the REST ingestion API is plain JSON over HTTPS.

## API Surface

- `mod.rs` — `LangfuseConfig`: `new(public_key, secret_key)`, `with_host`, `with_service_name`, `host()`, `service_name()`, `endpoint()`, `ingestion_url()`, `authorization()`, `validate()`, `otel_config()`, `from_env()`, `exporter()` behind `feature = "otel"`; constants `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_BASEURL`, `DEFAULT_HOST`.
- `events.rs` — `IngestionEventType` (`trace-create`/`span-create`/`generation-create`, `as_str`, `Display`), `IngestionEvent` (`new`, `trace`, `span`, `generation`, `with_id`, `with_timestamp`, `to_json`), and the camelCase bodies `TraceBody`, `SpanBody`, `GenerationBody` (each `new` plus `with_*` builders).
- `batch.rs` — `IngestionBatch` (`new`, `with_capacity`, `push`, `extend`, `len`, `is_empty`, `events`, `into_events`, `max_size_hint`) and `DEFAULT_MAX_BATCH_SIZE` (1000 events per request).
- `client.rs` — `LangfuseClient` (`new`, `from_env`, `with_http_client`, `with_max_batch_size`, `config`, `max_batch_size`, `ingest`, `ingest_events`) and `IngestionSummary { events, requests }`.
- The default client times out each request after 30 s; `with_http_client` replaces it wholesale.

## Constraints

- An empty batch returns `Ok(IngestionSummary::default())` without any HTTP request; `with_max_batch_size(0)` is treated as `1`.
- A 2xx (or 207) response is a success only when its JSON has no `errors` entries; a non-empty `errors` array, a malformed `errors` field, a non-JSON 2xx body, and every non-2xx status become `Error::tool` naming the status and the truncated body or the failing ids/statuses/messages.
- Bodies are camelCase on the wire, tolerate unknown fields, and omit absent fields instead of sending `null`; ids are UUID v4 and timestamps are RFC 3339 UTC (`openjiuwen_common::utils::id::now_rfc3339`).
- `Debug` for the preset and the client redacts the secret key; no `unwrap`/`expect` outside tests, no panics on user input, `tracing` only.

## Test Strategy

`tests.rs` covers the preset (default and custom hosts, trailing-slash and pasted-REST-path normalisation, exact base64 credentials, `otel_config()` fields, blank-key and bad-host validation, `from_env()` precedence, `exporter().traces_url()`).
`client/tests.rs` drives `wiremock`: POST path, `content-type`, exact `Authorization`, the `batch` array shape per event type, camelCase body fields, the success summary, a 207 with `errors`, a 500, a non-JSON 2xx, 5-event chunking at size 2 (3 requests), the empty batch (zero requests), and `max_batch_size(0)`.
`events.rs` and `batch.rs` keep their own inline unit tests for the wire shape and the ordered buffer.
Environment tests hold a `static Mutex` and clear `LANGFUSE_*` on entry and on drop; every test stays offline.
