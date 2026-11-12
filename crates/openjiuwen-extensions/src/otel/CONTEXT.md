# otel/ — OpenTelemetry tracing over the session `TraceSink` SPI

## Intent

Port of `openjiuwen/extensions/tracer_otel/` (v0.1.20): turn session trace frames into OpenTelemetry spans with the Python span names, attributes, statuses, redaction, and parent/child links.
Python registers async handler callbacks on the session tracer registry; Rust sessions emit synchronous [`TraceSink`] frames instead, so the primary seam is a sink (`OtelTraceSink`) plus typed handlers (`OtelAgentHandler`, `OtelWorkflowHandler`) for callers that already hold a `TraceSpan`.
Everything here must degrade to a cheap no-op when the `otel` feature is off: the config, semantic conventions, redaction, and span-name helpers compile in both cases, and `init_otel_tracer` returns an inactive handle instead of failing.

## API Surface

```rust
// config.rs (always compiled)
pub enum ExporterType { None, Memory, Console, Otlp }   // serde snake_case; Default = Otlp (Python default)
pub enum OtlpProtocol { Grpc, Http }                    // Default = Grpc
pub struct OtelTracerConfig {
    pub tracer_name: String,            // "openjiuwen.tracer.otel"
    pub exporter_type: ExporterType,    // "otlp"
    pub exporter_endpoint: Option<String>,
    pub protocol: OtlpProtocol,         // "grpc"
    pub headers: BTreeMap<String, String>,
    pub service_name: String,           // "openjiuwen"
    pub service_version: Option<String>,// "unknown" when unset
    pub sample_rate: f64,               // 1.0; must be within 0.0..=1.0
    pub schedule_delay_millis: u64,     // 5000
    pub export_timeout_ms: u64,         // 30000
    pub max_export_batch_size: usize,   // 512
    pub redaction_enabled: bool,        // true
    pub redact_prompts: Option<bool>,   // None = follow redaction_enabled
    pub redact_completions: Option<bool>,
    pub max_attr_length: usize,         // 4096
    pub max_attributes: usize,          // 200 (SpanLimits)
    pub global_instrument_enable: bool, // false
}
impl OtelTracerConfig {
    pub fn validate(&self) -> Result<()>;                          // Error::Validation on sample_rate outside 0..=1
    pub fn with_exporter_type(self, kind: ExporterType) -> Self;
    pub fn with_endpoint(self, endpoint: impl Into<String>) -> Self;
    pub fn with_service(self, name: impl Into<String>, version: Option<String>) -> Self;
}
pub fn global_instrument_enabled(config: &OtelTracerConfig) -> bool; // OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE wins ("1"/"true"/"yes"/"on")

// redaction.rs (always compiled)
pub fn truncate(value: &str, max_length: usize) -> String;
pub fn hash_value(value: &str) -> String;                 // "sha256:<first 16 hex chars>"
pub struct Redactor<'a> { pub config: &'a OtelTracerConfig }
impl Redactor<'_> {
    pub fn redact(&self, value: Option<&Value>, field: Option<RedactionField>) -> String;
    pub fn redact_system_prompt(&self, value: Option<&Value>) -> String;
    pub fn redact_json(&self, value: &Value, field: Option<RedactionField>, system_prompt: bool) -> String;
}
pub enum RedactionField { Prompts, Completions }

// semconv.rs (always compiled) — attribute keys verbatim, `pub const` &str
// gen_ai.*: error.type, gen_ai.agent.description/id/name, gen_ai.conversation.id, gen_ai.data_source.id,
//   gen_ai.embeddings.dimension.count, gen_ai.input.messages, gen_ai.memory.record.count,
//   gen_ai.operation.name, gen_ai.output.messages, gen_ai.provider.name, gen_ai.request.max_tokens,
//   gen_ai.request.model, gen_ai.request.reasoning.level, gen_ai.request.stop_sequences,
//   gen_ai.request.temperature, gen_ai.request.top_k, gen_ai.request.top_p,
//   gen_ai.response.finish_reasons, gen_ai.response.model, gen_ai.response.time_to_first_chunk,
//   gen_ai.retrieval.documents, gen_ai.retrieval.query.text, gen_ai.retrieval.top_k,
//   gen_ai.system_instructions, gen_ai.tool.call.id, gen_ai.tool.name, gen_ai.tool.type,
//   gen_ai.usage.cache_read.input_tokens, gen_ai.usage.cache_write.input_tokens,
//   gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, gen_ai.usage.reasoning.output_tokens
// openjiuwen.*: llm.prev_message_count, gen_ai.usage.total_cost/input_cost/output_cost, gen_ai.user.id,
//   gen_ai.metadata, trace.name, gen_ai.response.inter_token_latency_ms, gen_ai.reasoning.duration_ms,
//   workflow.id/name/version/component.id/component.type/component.name/execution_id/loop.node_id/loop.index/
//   inputs/outputs/error_message/invoke_data, agent.invoke_type/name/inputs/outputs/error_message,
//   trace.id, session_id, invoke_id, parent_invoke_id, start_time, end_time, elapsed_time, status, error,
//   child_invoke_ids, parent_node_id, source_ids, inner_error, stream_inputs, stream_outputs, interactive_inputs
// values: GEN_AI_PROVIDER_OPENJIUWEN = "openjiuwen"; operation names chat/execute_tool/embeddings/retrieval;
//   status values start/running/finish/interrupted/error; ERROR_TYPE = "error.type".

// span_names.rs (always compiled)
pub fn chat_span_name(class_name: &str) -> String;              // "chat {class_name}"
pub fn chat_span_name_for_model(model: &str) -> String;         // "chat {model}" / "chat" when empty
pub fn tool_span_name(name: &str) -> String;                    // "execute_tool {name}"
pub fn legacy_tool_span_name(name: &str) -> String;             // "tool.{name}"
pub fn simple_span_name(invoke_type: InvokeType, name: &str) -> String; // "prompt.{name}", "chain.{name}", "retriever.{name}", "evaluator.{name}", "workflow.{name}", "memory.{name}"
pub fn embeddings_span_name(model: &str) -> String;             // "embeddings {model}"
pub fn workflow_component_span_name(invoke_id: &str) -> String; // "component.{invoke_id}"
pub fn is_llm_component(component_type: &str) -> bool;          // contains "LLM" | "IntentDetection" | "Questioner"
pub fn is_tool_component(component_type: &str) -> bool;         // contains "Tool"
pub fn is_retrieval_component(component_type: &str) -> bool;    // contains "Retrieval"

// handle.rs (always compiled)
pub struct TracerHandle { .. }
impl TracerHandle {
    pub fn is_active(&self) -> bool;
    pub fn tracer_name(&self) -> &str;
    pub fn force_flush(&self);
    pub fn shutdown(&self);
}
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle>;
// without `otel`: validates the config and returns an inactive handle (no error for any exporter type)
// with `otel`:      Memory/None/Console build a real provider; Otlp returns Error::Unsupported (deferred)

// feature "otel" only
pub struct OtelTraceSink { .. }  // impl openjiuwen_core::session::tracer::TraceSink
impl OtelTraceSink {
    pub fn new(handle: &TracerHandle) -> Result<Arc<Self>>;   // also accepts a tracer + config
    pub fn with_session(session_id: impl Into<String>) -> Result<Arc<Self>>;
    pub fn is_recording(&self) -> bool;
}
pub struct OtelAgentHandler { .. }    // typed callbacks, never panics, swallows and logs failures
pub struct OtelWorkflowHandler { .. }
pub fn tracing_layer<S>(tracer: opentelemetry_sdk::trace::SdkTracer)
    -> tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::SdkTracer>;
```

## Ported Behavior

- Redaction (`tracer_otel/redaction.py`): `truncate` returns the value unchanged when `max_length == 0` or `len <= max_length`, else `value[:max] + "...<OTel attribute truncated: {omitted} chars omitted>"`.
  `hash_value` = `"sha256:" + first 16 lowercase hex chars of sha256(value)`.
  Field overrides: `Prompts` uses `redact_prompts` when set else `redaction_enabled`; `Completions` uses `redact_completions` when set else `redaction_enabled`; `None` uses `redaction_enabled`.
  `redact` hashes when redaction applies, otherwise truncates to `max_attr_length`; `redact_system_prompt` never truncates (hash or full text).
  `redact_json` walks the value: keys exactly `content`, `arguments`, `response` get the field/system-prompt treatment; every other key/array element recurses; the result is compact JSON (`serde_json::to_string`, non-ASCII preserved).
- Span naming (`tracer_otel/handler.py`, `span_manager.py`): agent LLM span = `chat {class_name}` (`SpanKind::Client`, `gen_ai.operation.name = chat`); tool span = `execute_tool {name}` (`SpanKind::Internal`, `execute_tool`); prompt/chain/retriever/evaluator/workflow/memory = `{kind}.{name}`; embeddings = `embeddings {model}`; workflow root = the `invoke_id`; workflow components: LLM-ish → `chat {model}` (Client), tool-ish → `execute_tool {name}`, retrieval-ish → `retrieval {name}`, otherwise `component.{invoke_id}`.
- Agent span start attributes: `openjiuwen.trace.id`, `gen_ai.provider.name = "openjiuwen"`, `openjiuwen.session_id` (when non-empty), `openjiuwen.invoke_id`, `openjiuwen.parent_invoke_id` (when set), `openjiuwen.start_time` (`%Y-%m-%d %H:%M:%S%.6f` local time), plus the per-callback attributes listed in the crate's agent-handler docs; parent context comes from the `parent_invoke_id` state.
- Span end: `Status::Ok`, `openjiuwen.end_time`, `openjiuwen.elapsed_time` (`"{ms}ms"` below 1 s, else `"{s:.2}s"`), `openjiuwen.child_invoke_ids` (JSON array) when known, `openjiuwen.status` = span status (`finish` by default).
- Span error: `Status::Error`, `openjiuwen.status = "error"`, `openjiuwen.agent.error_message` (or `openjiuwen.workflow.error_message`), `openjiuwen.error` (JSON `{error_code, message}`), `error.type`; `record_exception` equivalent is the event `exception` with `exception.message`.
- Interruption: `SpanStatus::Interrupted` → `openjiuwen.status = "interrupted"` (workflow `openjiuwen.workflow.error_message` set), no error status, no `openjiuwen.error`.
- Tool-span dedup (`bug #2046`): `OtelAgentSpanManager::has_open_tool_span(tool_name, trace_id)` inspects open spans whose `gen_ai.operation.name` is `execute_tool`, compares `gen_ai.tool.name`, and (when `trace_id` is non-empty) `openjiuwen.trace.id`; a match means the tool rail already opened the span and the agent callback must not open a second one.
- Workflow spans keep per-invoke buffers for `openjiuwen.workflow.invoke_data`, `openjiuwen.stream_inputs`, `openjiuwen.stream_outputs`; they are flushed as JSON only when non-empty, on `on_call_done`/`on_invoke`.
- Every handler callback is fail-open: errors are logged at `warn`, never returned and never panicked.

## Constraints

- `otel` feature deps: `opentelemetry`, `opentelemetry_sdk`, `tracing-opentelemetry`, `tracing-subscriber`, `chrono`.
- The provider is per-handle, never the process-global `opentelemetry::global` provider (Python keeps a private provider too); `TracerHandle::shutdown` calls `SdkTracerProvider::shutdown` and is idempotent.
- OTLP export is deferred: `ExporterType::Otlp` → `Error::Unsupported("OTLP span export is deferred; use ExporterType::Memory/Console or inject a SpanExporter")`. `Console` uses an in-crate line-per-span exporter writing through `tracing` at `debug` (no stdout writes).
- Tests must not read the real clock for assertions: elapsed/start-time assertions only check presence/format.
- Files ≤ ~1000 lines, one concern per file; no `unsafe`; no blocking calls.

## Test Strategy

- `./tests/otel.rs` (`#![cfg(feature = "otel")]`): a `CaptureExporter` (`opentelemetry_sdk::trace::SpanExporter`) collects `SpanData`; asserts span names, kinds, `gen_ai.*`/`openjiuwen.*` attributes, OK/ERROR status, parent/child links, tool dedup, and the workflow buffer flush.
- `./tests/otel_seam.rs` (no feature requirement): `init_otel_tracer` with every exporter type returns a handle, `is_active()` matches `cfg!(feature = "otel")` for `Memory`, and `ExporterType::Otlp` under the feature returns `ErrorKind::Unsupported`.
- Unit tests beside `redaction.rs` (all `test_redaction.py` cases: exact truncation marker, hash formula, field override truth table, system-prompt non-truncation), `span_names.rs` (every naming rule), `config.rs` (sample-rate validation, env toggle precedence), `handle.rs` (no-op vs active).
- Python counterparts named in comments: `tests/unit_tests/extensions/tracer_otel/test_redaction.py`, `test_setup.py`, `test_handler.py`, `test_span_manager.py`, `test_otel_tool_span_dedup.py`.

## Routing Table

- `./config.rs` → `OtelTracerConfig`, exporter/protocol enums, validation, env toggle
- `./semconv.rs` → attribute key/value constants
- `./redaction.rs` → truncate/hash/redact helpers
- `./span_names.rs` → span naming rules for agent/workflow spans
- `./handle.rs` → `TracerHandle`, `init_otel_tracer`, no-op seam when the feature is off
- `./provider.rs` → SDK provider construction, in-memory + console exporters, `tracing_layer` (`otel`)
- `./span_manager.rs` → `OtelSpanState`, agent/workflow span managers (`otel`)
- `./sink.rs` → `OtelTraceSink` (`TraceSink` implementation) (`otel`)
- `./agent_handler.rs` → `OtelAgentHandler` callbacks (`otel`)
- `./workflow_handler.rs` → `OtelWorkflowHandler` callbacks (`otel`)
