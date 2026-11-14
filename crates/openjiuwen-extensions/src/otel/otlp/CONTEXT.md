# otel/otlp

## Intent

OTLP/HTTP span exporter with JSON and protobuf payloads, built for `ExporterType::Otlp` and shared by the Grafana and Langfuse presets.
Ports the exporter wiring of `openjiuwen/extensions/tracer_otel/setup.py` (v0.1.20), which hands an `OTLPSpanExporter` to the SDK; here the exporter is owned by this crate so the wire body can be asserted and the transport can run offline.
Both OTLP/HTTP encodings exist: `http/json` (`OtlpEncoding::Json`, the default, `Content-Type: application/json`) and `http/protobuf` (`OtlpEncoding::Protobuf`, `application/x-protobuf`).
The protobuf body comes from the hand-written encoder in `protobuf.rs`: `prost`, `tonic` and `opentelemetry-otlp` are unavailable to an offline build and none of them is needed for OTLP/HTTP.
Both encoders consume the span grouping from `model.rs`, so the scopes, their first-seen order, and the spans inside each scope are identical in the two bodies.
gRPC is the only deferred transport: it needs `tonic`/`prost` plus an HTTP/2 stack, and its error names OTLP/HTTP as the supported alternative.
The whole subtree is compiled only with `feature = "otel"` and holds every SDK type of the export path.

## API Surface

- `OtlpEncoding` — `Json` (`#[default]`) and `Protobuf`, deriving `Debug, Clone, Copy, PartialEq, Eq, Hash, Default`, with `content_type(self) -> &'static str`.
- `OtlpHttpExporter::new(endpoint, headers, timeout) -> Result<Self>` — the JSON exporter; validates and normalizes the endpoint, converts the headers, starts the worker thread, performs no I/O.
- `OtlpHttpExporter::with_encoding(endpoint, headers, timeout, OtlpEncoding) -> Result<Self>` — the same construction with an explicit encoding; `new` delegates to it.
- `OtlpHttpExporter::traces_url(&self) -> &str` — the fully-qualified URL every batch is posted to, always ending in `/v1/traces`.
- `OtlpHttpExporter::content_type(&self) -> &'static str` — the `Content-Type` every request carries, derived from the encoding.
- `impl SpanExporter for OtlpHttpExporter` — `export(&self, batch: Vec<SpanData>)`, `shutdown_with_timeout(&mut self, Duration)`, `force_flush(&mut self)` (a no-op, nothing is buffered), and `set_resource(&mut self, &Resource)` (cached for the body encoder).
- `traces_url(endpoint: &str) -> String` — the public normalization helper from `endpoint.rs`.
- Private seams: `model::scope_groups(spans) -> Vec<ScopeGroup<'_>>` with `ScopeGroup { scope, spans }`, `json::encode_body(resource, spans) -> Value`, `json::hex(bytes) -> String`, `protobuf::encode_body(resource, spans) -> Vec<u8>`, and `transport::{Job, JobSender, Worker}`.

## Constraints

- The exporter owns its HTTP execution context: `opentelemetry_sdk` 0.31 drives `SpanExporter::export` from the batch processor's plain `std::thread` under `futures_executor::block_on`, where no tokio reactor exists, so a `reqwest` call made there panics with "there is no reactor running".
- `transport.rs` starts one `std::thread` named `openjiuwen-otlp-http` that builds a current-thread tokio runtime and the single `reqwest::Client`, then serves jobs until the channel closes.
- A job carries the pre-encoded body as bytes plus its `Content-Type`, so the worker serves both encodings without knowing which one it is posting.
- A job also carries a `tokio::sync::oneshot` reply and travels over an unbounded tokio channel, so `export` awaits the outcome under whatever executor called it (the batch processor's `block_on`, a tokio task, or a plain test).
- Dropping the `Worker` without `shutdown` detaches the thread; it exits as soon as the job channel closes, so no exporter leaks a running thread.
- `shutdown_with_timeout` closes the channel, waits on the worker's `std::sync::mpsc` completion signal, and detaches it with a `warn` when the budget expires; the worker shuts its runtime down with a 100 ms grace period after the last job.
- Error policy is fail-open: every failure becomes `OTelSdkError::InternalFailure` plus a `tracing::warn!` on target `openjiuwen_extensions::otel::otlp`, and nothing panics.
- Body assembly names the encoding it failed on (`the OTLP JSON body could not be serialized`); the protobuf encoder is infallible and yields bytes directly.
- `new`/`with_encoding` answer `Error::Validation` for an empty endpoint, a non-absolute or non-`http`/`https` URL, or an invalid header name/value, and `Error::Config` when the worker thread cannot be spawned.
- An empty batch returns `Ok(())` without a request; an export after shutdown, a closed job channel, and a lost reply each report a distinct typed message.
- A non-2xx response reports the status plus a body snippet capped at 512 characters; transport failures are named as timeout, connect, or generic request errors.
- Header values may be credentials, so `Debug` prints the header names only (next to the encoding and the shutdown flag); `content-type` is always the encoding's and is not overridable.
- The cached resource is read under a poison-recovering mutex, so a panic elsewhere cannot turn span export into a panic on the batch-processor thread.
- `endpoint::traces_url` trims trailing slashes and appends `/v1/traces` exactly once, so a base URL and a full traces URL normalize to the same single path.
- The env layer accepts `http/json` and `http` as JSON and `http/protobuf`, `http_protobuf`, `http-protobuf` and `protobuf` as protobuf; `grpc` stays `Error::Unsupported` with the OTLP/HTTP alternative.

## Wire Shape — JSON

The body is the OTLP/JSON `ExportTraceServiceRequest`: one `resourceSpans` entry holding the provider's resource attributes (an empty `resource` object when none were set) and one `scopeSpans` entry per distinct `(name, version)` instrumentation scope, in first-seen order with span order preserved inside a scope.
Each span carries `traceId`, `spanId`, and (unless invalid) `parentSpanId` as lowercase hex, `name`, `kind` (`SPAN_KIND_INTERNAL`/`_SERVER`/`_CLIENT`/`_PRODUCER`/`_CONSUMER`), `startTimeUnixNano`/`endTimeUnixNano` as decimal strings (JSON numbers cannot carry 64-bit nanoseconds; pre-epoch times clamp to `"0"`), `attributes`, `droppedAttributesCount`, `events`, `links`, and `status`.
Events carry `timeUnixNano`, `name`, `attributes`, and `droppedAttributesCount`; links carry `traceId`, `spanId`, `attributes`, and `droppedAttributesCount`.
Status codes are `STATUS_CODE_UNSET`, `STATUS_CODE_OK`, and `STATUS_CODE_ERROR`, with `message` emitted only for a described error.
Attribute values encode as `boolValue`, `intValue` (decimal string), `doubleValue`, `stringValue`, or `arrayValue` with a `values` array; a non-exhaustive `Value` variant falls back to its `Display` text as `stringValue` instead of dropping the attribute.
Attribute lists are omitted when empty, except the span's own `attributes`, `events`, and `links`, which are always arrays so the span shape never depends on content.

## Wire Shape — protobuf

The body is `ExportTraceServiceRequest` with one `resource_spans` (field 1) holding the resource attributes and one `scope_spans` (field 2) per group from `model::scope_groups`.
`Resource.attributes` is field 1, `ScopeSpans.scope`/`spans` are fields 1/2, and `InstrumentationScope.name`/`version`/`attributes` are fields 1/2/3.
`Span` follows the OTLP proto numbering: ids as raw 16/8 bytes (1, 2, and 4 for a non-invalid parent), `name` 5, `kind` 6, `start_time_unix_nano`/`end_time_unix_nano` as `fixed64` 7/8, `attributes` 9, `dropped_attributes_count` 10, `events` 11, `links` 13, `status` 15.
`trace_state` (3), `dropped_events_count` (12), `dropped_links_count` (14), and the span/link `flags` (16) are omitted because the JSON body carries no counterpart; the parity test pins that rule.
`AnyValue` is a oneof: `string_value` 1, `bool_value` 2, `int_value` 3 (two's-complement varint, not zig-zag), `double_value` 4 (IEEE-754 bits in a `fixed64`), `array_value` 5; oneof members keep explicit presence, so `""`, `false`, `0`, and `0.0` are still written.
Proto3 default scalars outside the oneof are skipped, exactly as a `prost`-generated encoder would skip them.
`Status` writes `code` (3) as 1/2 for Ok/Error and `message` (2) only for a described error, so `UNSET` is the absent zero; `SpanKind` maps Internal, Server, Client, Producer, Consumer to 1–5.
A pre-epoch or unrepresentable `SystemTime` becomes 0 nanoseconds, an unknown `Value` variant falls back to `string_value`, and an unknown `Array` variant yields an empty `ArrayValue`, all matching the JSON encoder.
Ids, `fixed64` words, and `fixed32` words are little-endian; `write_fixed32` exists for the omitted `flags` field and is exercised by the unit tests only.

## Test Strategy

`tests.rs` drives `export()` from plain `#[test]` functions with `futures::executor::block_on` — deliberately reactor-free, so any tokio dependency in the exporter would hang or panic — and mounts a loopback `wiremock` collector for the HTTP paths.
It pins endpoint normalization, constructor validation, `Send + Sync + Debug`, the JSON body field by field (scope grouping, resource, hex ids, kinds, nanosecond timestamps, typed attributes, events, links, statuses, resource-less bodies), posting to the traces path, non-2xx handling, the client timeout budget, and resource/shutdown/post-shutdown-export behavior.
`protobuf/tests.rs` decodes the protobuf body with a hand-written reader (varint, `fixed64`, `fixed32`, length-delimited) into a field map and asserts the writer primitives (`varints_encode_the_unsigned_edges`, `tags_combine_field_number_and_wire_type`, `fixed_width_words_are_little_endian`), a full span (`one_span_encodes_every_field`), the `AnyValue` matrix (`any_values_encode_every_scalar_variant`, `known_any_values_never_take_the_text_fallback`), status mapping (`status_codes_and_messages_map_to_the_enum`), all five span kinds (`every_span_kind_has_its_wire_value`), scope grouping shared with JSON (`scopes_group_spans_in_first_seen_order`), the empty-batch shape (`an_empty_batch_keeps_the_request_shape`), and the cross-encoding parity of one rich batch (`json_and_protobuf_describe_the_same_batch`).
Fixtures are built through a real `SdkTracerProvider` plus `CaptureExporter`, so no message is hand-assembled.
Every test stays offline and uses loopback addresses or `127.0.0.1:9` for the "nothing is posted" cases.

## Routing Table

- `./mod.rs` → `OtlpEncoding`, `OtlpHttpExporter` (construction, validation, body dispatch per encoding, `SpanExporter` impl, `Debug` redaction, doctests)
- `./model.rs` → shared span grouping: `ScopeGroup`, `scope_groups`
- `./json.rs` → OTLP/JSON encoder: `resourceSpans`/`scopeSpans` rendering, span/event/link/status encoding, `AnyValue` mapping, hex ids, nanosecond formatting
- `./protobuf.rs` → OTLP/protobuf encoder: varint/tag/fixed/length-delimited writers, message encoders, `AnyValue` oneof
- `./protobuf/tests.rs` → protobuf unit tests with the hand-written decoder and the JSON parity test
- `./transport.rs` → dedicated-thread worker: current-thread runtime, shared `reqwest::Client`, byte jobs with their content type, POST, response mapping, bounded shutdown
- `./endpoint.rs` → `/v1/traces` normalization (`traces_url`, doctest)
- `./tests.rs` → unit tests of the exporter, the JSON encoder, and the transport
