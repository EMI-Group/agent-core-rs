# otel/otlp

## Intent

OTLP/HTTP span exporter with JSON payloads, built for `ExporterType::Otlp` and shared by the Grafana and Langfuse presets.
Ports the exporter wiring of `openjiuwen/extensions/tracer_otel/setup.py` (v0.1.20), which hands an `OTLPSpanExporter` to the SDK; here the exporter is owned by this crate so the wire body can be asserted and the transport can run offline.
JSON over HTTP is the only transport available: gRPC and protobuf framing need `opentelemetry-otlp`, `prost`, and `tonic`, which the offline registry does not carry, while `http/json` is a first-class OTLP encoding every collector accepts on port 4318.
The whole subtree is compiled only with `feature = "otel"` and holds every SDK type of the export path.

## API Surface

- `OtlpHttpExporter::new(endpoint: impl Into<String>, headers: BTreeMap<String, String>, timeout: Duration) -> Result<Self>` — validates and normalizes the endpoint, converts the headers, and starts the worker thread; performs no I/O.
- `OtlpHttpExporter::traces_url(&self) -> &str` — the fully-qualified URL every batch is posted to, always ending in `/v1/traces`.
- `impl SpanExporter for OtlpHttpExporter` — `export(&self, batch: Vec<SpanData>)`, `shutdown_with_timeout(&mut self, Duration)`, `force_flush(&mut self)` (a no-op, nothing is buffered), and `set_resource(&mut self, &Resource)` (cached for the body encoder).
- `traces_url(endpoint: &str) -> String` — the public normalization helper from `endpoint.rs`.
- Private seams: `json::encode_body(resource, spans) -> Value` and `json::hex(bytes) -> String` for the encoder, `transport::{Job, JobSender, Worker}` for the worker, `header_map`/`lock` for header conversion and poison-recovering locks.

## Constraints

- The exporter owns its HTTP execution context: `opentelemetry_sdk` 0.31 drives `SpanExporter::export` from the batch processor's plain `std::thread` under `futures_executor::block_on`, where no tokio reactor exists, so a `reqwest` call made there panics with "there is no reactor running".
- `transport.rs` starts one `std::thread` named `openjiuwen-otlp-http` that builds a current-thread tokio runtime and the single `reqwest::Client`, then serves jobs until the channel closes.
- Jobs are pre-serialized bodies plus a `tokio::sync::oneshot` reply, sent over an unbounded tokio channel, so `export` awaits the outcome under whatever executor called it (the batch processor's `block_on`, a tokio task, or a plain test).
- Dropping the `Worker` without `shutdown` detaches the thread; it exits as soon as the job channel closes, so no exporter leaks a running thread.
- `shutdown_with_timeout` closes the channel, waits on the worker's `std::sync::mpsc` completion signal, and detaches it with a `warn` when the budget expires; the worker shuts its runtime down with a 100 ms grace period after the last job.
- Error policy is fail-open: every failure becomes `OTelSdkError::InternalFailure` plus a `tracing::warn!` on target `openjiuwen_extensions::otel::otlp`, and nothing panics.
- `new` answers `Error::Validation` for an empty endpoint, a non-absolute or non-`http`/`https` URL, or an invalid header name/value, and `Error::Config` when the worker thread cannot be spawned.
- An empty batch returns `Ok(())` without a request; an export after shutdown, a closed job channel, and a lost reply each report a distinct typed message.
- A non-2xx response reports the status plus a body snippet capped at 512 characters; transport failures are named as timeout, connect, or generic request errors.
- Header values may be credentials, so `Debug` prints the header names only; `content-type` is always `application/json` and is not overridable.
- The cached resource is read under a poison-recovering mutex, so a panic elsewhere cannot turn span export into a panic on the batch-processor thread.
- `endpoint::traces_url` trims trailing slashes and appends `/v1/traces` exactly once, so a base URL (`http://host:4318`) and a full traces URL (`http://host:4318/v1/traces`) both normalize to the same single path.

## Wire Shape

The body is the OTLP/JSON `ExportTraceServiceRequest`: one `resourceSpans` entry holding the provider's resource attributes (an empty `resource` object when none were set) and one `scopeSpans` entry per distinct `(name, version)` instrumentation scope, in first-seen order with span order preserved inside a scope.
Each span carries `traceId`, `spanId`, and (unless invalid) `parentSpanId` as lowercase hex, `name`, `kind` (`SPAN_KIND_INTERNAL`/`_SERVER`/`_CLIENT`/`_PRODUCER`/`_CONSUMER`), `startTimeUnixNano`/`endTimeUnixNano` as decimal strings (JSON numbers cannot carry 64-bit nanoseconds; pre-epoch times clamp to `"0"`), `attributes`, `droppedAttributesCount`, `events`, `links`, and `status`.
Status codes are `STATUS_CODE_UNSET`, `STATUS_CODE_OK`, and `STATUS_CODE_ERROR`, with `message` emitted only for a described error.
Events carry `timeUnixNano`, `name`, `attributes`, and `droppedAttributesCount`; links carry `traceId`, `spanId`, `attributes`, and `droppedAttributesCount`.
Attribute values encode as `boolValue`, `intValue` (decimal string), `doubleValue`, `stringValue`, or `arrayValue` with a `values` array; a non-exhaustive `Value` variant falls back to its `Display` text as `stringValue` instead of dropping the attribute.
Attribute lists are omitted when empty, except the span's own `attributes`, `events`, and `links`, which are always arrays so the span shape never depends on content.

## Test Strategy

`tests.rs` drives `export()` from plain `#[test]` functions with `futures::executor::block_on` — deliberately reactor-free, so any tokio dependency in the exporter would hang or panic — and mounts a loopback `wiremock` collector for the HTTP paths.
It pins endpoint normalization, constructor validation (empty, relative, `ftp`, bad header name/value), `Send + Sync + Debug`, the JSON body field by field (scope grouping, resource, hex ids, kinds, nanosecond timestamps, typed attributes, events, links, statuses, resource-less bodies), posting to the traces path, non-2xx handling, the client timeout budget, and resource/shutdown/post-shutdown-export behavior.
Every test stays offline and uses loopback addresses or `127.0.0.1:9` for the "nothing is posted" cases.

## Routing Table

- `./mod.rs` → `OtlpHttpExporter` (construction, endpoint/header validation, body assembly, `SpanExporter` impl, `Debug` redaction, doctest)
- `./json.rs` → OTLP/JSON encoder: `resourceSpans`/`scopeSpans` grouping, span/event/link/status encoding, `AnyValue` mapping, hex ids, nanosecond formatting
- `./transport.rs` → dedicated-thread worker: current-thread runtime, shared `reqwest::Client`, job channel, POST, response mapping, bounded shutdown
- `./endpoint.rs` → `/v1/traces` normalization (`traces_url`, doctest)
- `./tests.rs` → unit tests of the exporter, the encoder, and the transport
