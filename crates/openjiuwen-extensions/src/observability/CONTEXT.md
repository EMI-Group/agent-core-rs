# openjiuwen-extensions/src/observability

## Intent

Dependency-neutral observability utilities: the port of the pure half of Python's
`openjiuwen/extensions/observability/` (v0.1.20).
Python keeps that package importable without initializing an exporter — semantic conventions,
redaction, content addressing and tool/error classification are plain helpers — and this module
keeps the same split.
Everything here compiles into the default build except `file_exporter` and `transforming`, which
hand a `SpanData` batch to the SDK and are gated on the `otel` feature.
The live tracer lives in `../otel/`; this module adds the attribute/string plumbing and the two
export-path utilities around it.

## API Surface

- `project_semconv` — 151 constants: the `agentteam.*` (`AT_*`), `deepagent.*` (`DA_*`) and
  `openjiuwen.*` namespaces Python's `semconv.py` owns, the 39 `gen_ai.*` keys the pinned upstream
  registry defines but `../otel/semconv.rs` lacks, plus `TRAJECTORY_SPAN_SCHEMA_VERSION`,
  `TRAJECTORY_RECORD_KINDS`, `TRAJECTORY_EVENT_KINDS`, `STREAM_SPAN_EVENT_NAMES`,
  `STREAM_FRAME_KIND_*`, `GEN_AI_SEMCONV_REVISION`, `GEN_AI_SEMCONV_SCHEMA_URL`,
  `GEN_AI_CORE_SEMCONV_SCHEMA_URL`, `GEN_AI_SEMCONV_ATTRIBUTE_COUNT`.
  The 38 names `../otel/semconv.rs` already owns are re-exported, never redefined.
  Enumerable inventories: `GEN_AI_KEYS`, `OJ_KEYS`, `AT_KEYS`, `DA_KEYS`, `REEXPORTED_KEYS`.
- `redaction` — `MASK`, `BEARER_MASK`, `ERROR_SUMMARY_MAX_LENGTH`, `REDACTED_HASH_PREFIX`,
  `mask_secrets`, `mask_bearer_tokens`, `redact_error_summary`, `redact_error_stacktrace`, plus a
  re-export of `hash_value`.
  The prompt/completion half stays in `../otel/redaction.rs`.
- `content_addressing` — `SEQUENCE_REFERENCE_VERSION`, `SEQUENCE_REFERENCE_PREFIX`,
  `ADDRESSED_ATTRIBUTE_KEYS`, `SequenceNode`, `AddressedSequence`, `content_hash`,
  `sequence_reference`, `parse_sequence_reference`, `split_elements`, `rebuild_value`,
  `build_sequence`, `addressable_attributes`, `address_payload`.
- `tool_outcome` — `TOOL_REPORTED_FAILURE`, `DEFAULT_FAILURE_REASON`, `EXCEPTION_RESULT_PREFIX`,
  `tool_failure_reason`, `tool_result_for_exception`.
- `error_reporting` — `error_reason(exception_text, error_message, exception_type, default)`.
- `file_exporter` (`otel`) — `TraceFileExporter::{new, from_config}` implementing `SpanExporter`;
  behind `ExporterType::File` with `OtelTracerConfig::{traces_dir, file_retention_days}`.
- `transforming` (`otel`) — `SpanTransform` (blanket impl for closures), `clone_span_with_attributes`,
  `TransformingSpanExporter::{new, inner, transform}`.

## Constraints

- No SDK types outside the two `otel`-gated modules: the always-compiled modules use only
  `serde_json`, `sha2` and plain Rust, so the default build keeps its dependency set.
- No `regex` crate (offline, no new crates): the credential scanners are hand-rolled
  `char_indices` walkers — ASCII case-insensitive, never byte-slicing, never panicking.
- Wire keys are byte-for-byte Python literals; never "normalize" a key, and never invent one.
  Where `../otel/semconv.rs` already owns a key, re-export it instead of re-typing the literal.
- `logical_bytes` in `AddressedSequence` measures the addressed value as Python measures it, not
  the rebuilt string; the rebuilt length is a different number.
- `file_exporter` is the one sanctioned file-write seam: appends are serialized behind one mutex
  and every I/O failure is logged and swallowed, because telemetry must never fail the caller.
  The file name uses `chrono::Local`, matching Python's local-date rollover.
- `transforming` loses no telemetry: a failing transform logs and exports the original span.
- Files ≤ ~1000 lines; `project_semconv.rs` is at the baseline and is a flat constant table, which
  is why it is not split further.
- Known upstream gap: the pinned registry has no `gen_ai.client.*` metric names (Python only
  alludes to them in prose), so no metric constants are invented here;
  `ADDRESSED_ATTRIBUTE_KEYS`' `gen_ai.tool.definitions` literal is local because
  `../otel/semconv.rs` has no constant for it yet.

## Deferred

- `setup.py`, `demand.py`, `runtime.py` — a process-global `TracerProvider` with reference-counted
  acquire/release. This crate keeps one `TracerHandle` per ownership scope and installs no global
  exporter or background runtime.
- `span_record_processor.py`, `trajectory_events.py`, `span_context.py` — the async span-record and
  stream-frame fan-out plus the contextvar span tracker; they need frozen `ReadableSpan`s and a
  reader-side consumer registry that this workspace has no trajectory data plane for. Their pure
  parts (record kinds, key literals, `@oj-seq` addressing) are ported.
- `instrumentation.py` — monkey-patched `httpx`/`requests`/`aiohttp`/`FastAPI` instrumentation; Rust
  has no equivalent hook surface.
- `context_compression_handler.py` — the context-engine compaction to trajectory-event bridge; it
  belongs with the deferred data plane.

## Routing Table

- `./project_semconv.rs` → project semantic conventions, trajectory key sets/versions, pinned registry metadata
- `./redaction.rs` → credential masking and error summary/stack-trace redaction
- `./content_addressing.rs` → `@oj-seq` prefix chains and OTLP JSON attribute addressing
- `./tool_outcome.rs` → tool-failure classification over a tool result
- `./error_reporting.rs` → the `error_reason` priority chain
- `./file_exporter.rs` → the OTLP-JSON WAL behind `ExporterType::File` (`otel` feature)
- `./transforming.rs` → export-time span projection wrapper (`otel` feature)
