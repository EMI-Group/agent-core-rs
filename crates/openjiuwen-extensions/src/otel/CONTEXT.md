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
    pub fn new(config: &OtelTracerConfig) -> Self;
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
    pub fn config(&self) -> &OtelTracerConfig;
    pub fn force_flush(&self);
    pub fn shutdown(&self);
    // feature "otel" only
    pub fn tracer(&self) -> Option<SdkTracer>;
    pub fn provider(&self) -> Option<&SdkTracerProvider>;
    pub fn capture(&self) -> Option<Arc<CaptureExporter>>;
}
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle>;
// without `otel`: validates the config and returns an inactive handle (no error for any exporter type)
// with `otel`:      Memory/None/Console build a real provider; Otlp returns Error::Unsupported (deferred)
pub fn init_otel_tracer_with_exporter<E: SpanExporter + 'static>(..) -> Result<TracerHandle>;

// sink/ (feature "otel") — the frame-level seam
pub struct OtelTraceSink { .. }  // Debug + Send + Sync; impl TraceSink for OtelTraceSink
impl OtelTraceSink {
    pub fn new(handle: &TracerHandle) -> Result<Arc<Self>>;   // inactive handle => Ok, emit is a no-op
    pub fn with_session(session_id: impl Into<String>) -> Result<Arc<Self>>;  // default config + ExporterType::Memory
    pub fn is_recording(&self) -> bool;                       // false for an inactive or shut-down handle
}

// agent_handler/ (feature "otel") — typed callbacks for callers holding a TraceSpan
pub struct OtelAgentHandler { .. }   // Debug + Send + Sync
impl OtelAgentHandler {
    pub fn new(tracer: SdkTracer, config: OtelTracerConfig, trace_id: Option<String>) -> Self;
    pub fn from_handle(handle: &TracerHandle) -> Result<Self>;   // Error::config when inactive
    pub fn set_trace_id(&mut self, trace_id: impl Into<String>);
    pub fn set_session_id(&mut self, session_id: impl Into<String>);
    pub fn has_open_tool_span(&self, tool_name: &str, trace_id: &str) -> bool;
    // async, fail-open, never return Result and never panic:
    pub async fn on_llm_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value);
    pub async fn on_llm_request(&self, span: &TraceSpan);
    pub async fn on_llm_end(&self, span: &TraceSpan, outputs: &Value);
    pub async fn on_llm_error(&self, span: &TraceSpan, error: &Error);
    // same start/end/error triple (`(span, inputs, instance_info)` / `(span, outputs)` / `(span, error)`) for:
    // plugin, prompt, chain, retriever, embedding, memory, evaluator, workflow
}

// workflow_handler/ (feature "otel") — typed callbacks for workflow engines
pub struct OtelWorkflowHandler { .. }   // Debug + Send + Sync
impl OtelWorkflowHandler {
    pub fn new(tracer: SdkTracer, config: OtelTracerConfig, trace_id: Option<String>) -> Self;
    pub fn from_handle(handle: &TracerHandle) -> Result<Self>;
    pub fn set_trace_id(&mut self, trace_id: impl Into<String>);
    pub fn set_session_id(&mut self, session_id: impl Into<String>);
    pub async fn on_call_start(&self, invoke_id: &str, metadata: &Value, inputs: Option<&Value>);
    pub async fn on_call_done(&self, invoke_id: &str, outputs: Option<&Value>);
    pub async fn on_pre_invoke(&self, invoke_id: &str, inputs: Option<&Value>);
    pub async fn on_pre_stream(&self, invoke_id: &str, chunk: &Value);
    pub async fn on_invoke(&self, invoke_id: &str, data: &Value);
    pub async fn on_post_invoke(&self, invoke_id: &str, outputs: &Value);
    pub async fn on_post_stream(&self, invoke_id: &str, chunk: &Value);
    pub async fn on_interact(&self, invoke_id: &str, inputs: &Value);
    pub async fn on_call_error(&self, invoke_id: &str, error: &Error);
    pub async fn on_call_interrupted(&self, invoke_id: &str, reason: Option<&str>);
}

// provider.rs (feature "otel")
pub struct CaptureExporter { .. }   // in-memory exporter: new/spans/len/is_empty/clear/names/span_named/resource
pub fn tracing_layer<S>(tracer: SdkTracer) -> tracing_opentelemetry::OpenTelemetryLayer<S, SdkTracer>;
```

`attrs`, `llm_attrs`, and `span_manager` are crate-private plumbing shared by the sink and both handlers (`apply_start_attributes`, `apply_end_attributes`, `apply_error_attributes{,_with}`, `apply_interrupt_attributes`, `AttributeScope`, `error_payload_for`, `local_timestamp`, `elapsed_string`; `standard_messages`, `set_llm_request_attributes`, `set_llm_response_attributes`, `unwrap_llm_payload`, `extract_retrieval_{query,documents}`, `redacted_json`; `OtelSpanState`, `OtelAgentSpanManager`, `OtelWorkflowSpanManager`).

## Ported Behavior

- Redaction (`tracer_otel/redaction.py`): `truncate` returns the value unchanged when `max_length == 0` or `len <= max_length`, else `value[:max] + "...<OTel attribute truncated: {omitted} chars omitted>"`.
  `hash_value` = `"sha256:" + first 16 lowercase hex chars of sha256(value)`.
  Field overrides: `Prompts` uses `redact_prompts` when set else `redaction_enabled`; `Completions` uses `redact_completions` when set else `redaction_enabled`; `None` uses `redaction_enabled`.
  `redact` hashes when redaction applies, otherwise truncates to `max_attr_length`; `redact_system_prompt` never truncates (hash or full text).
  `redact_json` walks the value: keys exactly `content`, `arguments`, `response` get the field/system-prompt treatment; every other key/array element recurses; the result is compact JSON (`serde_json::to_string`, non-ASCII preserved).
- Span naming (`tracer_otel/handler.py`, `span_manager.py`): agent LLM span = `chat {class_name}` (`SpanKind::Client`, `gen_ai.operation.name = chat`); tool span = `execute_tool {name}` (`SpanKind::Internal`, `execute_tool`); prompt/chain/retriever/evaluator/workflow/memory = `{kind}.{name}`; embeddings = `embeddings {model}`; workflow root = the `invoke_id`; workflow components: LLM-ish → `chat {model}` (Client), tool-ish → `execute_tool {name}`, retrieval-ish → `retrieval {name}`, otherwise `component.{invoke_id}`.
- Agent span start attributes: `openjiuwen.trace.id`, `gen_ai.provider.name = "openjiuwen"`, `openjiuwen.session_id` (when non-empty), `openjiuwen.invoke_id`, `openjiuwen.parent_invoke_id` (when set), `openjiuwen.start_time` (`%Y-%m-%d %H:%M:%S%.6f` local time), plus the per-callback attributes listed in the handler docs; parent context comes from the `parent_invoke_id` state.
- Span end: `Status::Ok`, `openjiuwen.end_time`, `openjiuwen.elapsed_time` (`"{ms}ms"` below 1 s, else `"{s:.2}s"`), `openjiuwen.child_invoke_ids` (JSON array) when known, `openjiuwen.status` = span status (`finish` by default).
- Span error: `Status::Error`, `openjiuwen.status = "error"`, `openjiuwen.agent.error_message` (or `openjiuwen.workflow.error_message`), `openjiuwen.error` (JSON `{error_code, message}`), `error.type`; the `record_exception` equivalent is the event `exception` with `exception.message`.
- Interruption: `SpanStatus::Interrupted` → `openjiuwen.status = "interrupted"` (workflow `openjiuwen.workflow.error_message` set), no error status, no `openjiuwen.error`.
- Tool-span dedup (`bug #2046`): `OtelAgentSpanManager::has_open_tool_span(tool_name, trace_id)` inspects open spans whose `gen_ai.operation.name` is `execute_tool`, compares `gen_ai.tool.name`, and (when `trace_id` is non-empty) `openjiuwen.trace.id`; a match means the tool rail already opened the span and the agent callback must not open a second one.
  The rail event is identified by `OtelSpanState::from_tool_rail` (set when `instance_info.type == "plugin"`), not by `gen_ai.tool.call.id` presence, so id-less calls are covered too.
- Workflow spans keep per-invoke buffers for `openjiuwen.workflow.invoke_data`, `openjiuwen.stream_inputs`, `openjiuwen.stream_outputs`; they are flushed as JSON only when non-empty, on `on_call_done`/`on_call_error`/the sink's terminal frame.
- Every handler callback and every `emit` path is fail-open: errors are logged at `warn`, never returned and never panicked.

## Frame-model adaptations (deviations forced by the Rust seams)

- `OtelTraceSink::emit` is the whole agent/workflow handler pair collapsed onto one frame: `payload.kind == Agent` maps the agent callbacks, `Workflow`/`Component` map the workflow callbacks, and `payload.status` selects start / running / terminal. `item_type` is ignored.
- The session tracer emits exactly one frame per span (its terminal `SpanHandle::complete`), so the sink opens spans lazily: an agent/workflow/component `running` or terminal frame whose invoke id has no open state first opens the span with the frame's `start_time` as the `OTel` start timestamp and the same name/kind/operation/parent/identity attributes as a `start` frame, then applies the running or terminal attributes. A `running`-only frame leaves the span open for its terminal frame; a lazily opened span ends at the frame's `end_time` (falling back to the local clock when the end is absent or precedes the start), while a span opened by a real `start` frame still ends at the local clock.
- A frame with no mappable span (the sink holds no tracer) registers nothing, so an inactive sink stays a no-op.
- The workflow handler has no per-event kwargs, so `parent_node_id`, `session_id`, and `source_ids` travel in the `metadata` object under those keys; `on_pre_invoke` has no `component_metadata`, so component/workflow attributes are written once on `on_call_start`.
- Python's exception branch of `on_invoke` is split into `on_call_error(&Error)` and `on_call_interrupted(reason)` because a session frame carries no exception object; `on_invoke` records `inner_error` when its payload carries one.
- Python's `on_invoke`/`on_pre_stream`/`on_post_stream` buffering becomes `meta_data` keys `on_invoke_data`, `stream_inputs`, `stream_outputs` on non-terminal frames (arrays append per element, scalars append once), flushed on the terminal frame.
- `otel_context.attach`/`detach` does not exist here. Parent links are rebuilt from the open parent's `SpanContext` (`Context::new().with_remote_span_context(..)`); `SpanRef { invoke_id, span_context, workflow_id, sequence }` in the workflow handler stands in for Python's `_layer_root_spans`/`_component_spans`, because `OtelSpanState` is not `Clone` and is owned by the span manager. "Last component span" uses an explicit monotonic sequence, not map order.
- Because the SDK span does not expose its attributes, anything a later callback re-reads (`openjiuwen.trace.id`, `gen_ai.tool.name`, `gen_ai.operation.name`, `openjiuwen.workflow.component.type`) must be written through `OtelSpanState::record_attribute`.
- `OtelTraceSink` stores a clone of the `SdkTracerProvider` alongside the tracer: `SdkTracer` only holds a weak provider reference, and `with_session` drops its local `TracerHandle`.
- `on_llm_request` adds a bare `llm.request` event (the Rust signature carries no attributes); `openjiuwen.agent.inputs` is redacted with `RedactionField::Prompts`; `openjiuwen.error`'s code is `ErrorKind::as_str()` with `attrs::DEFAULT_ERROR_CODE` as the frame-path fallback.

## Constraints

- `otel` feature deps: `opentelemetry`, `opentelemetry_sdk`, `tracing-opentelemetry`, `tracing-subscriber`, `chrono`; `serde_json`, `openjiuwen-core`, and `tracing` are unconditional crate dependencies.
- The provider is per-handle, never the process-global `opentelemetry::global` provider (Python keeps a private provider too); `TracerHandle::shutdown` calls `SdkTracerProvider::shutdown` and is idempotent.
- OTLP export is deferred: `ExporterType::Otlp` → `Error::Unsupported("OTLP span export is deferred; use ExporterType::Memory/Console or inject a SpanExporter")`. `Console` uses an in-crate line-per-span exporter writing through `tracing` at `debug` (no stdout writes).
- Tests must not read the real clock for assertions: elapsed/start-time assertions only check presence/format.
- Never take two span-manager locks at once: `OtelWorkflowSpanManager::with_state` holds the manager lock, so buffer snapshots (`on_invoke_data`/`stream_inputs`/`stream_outputs`) are read *before* entering `with_state`.
- Files ≤ ~1000 lines, one concern per file; no `unsafe`; no blocking calls; no `println!`.
- `agent_handler` and `workflow_handler` are private modules whose public types are re-exported at the `otel` root; each carries a documented `#![allow(clippy::unused_async, clippy::unused_async_trait_impl)]` because the callbacks keep Python's async signatures with synchronous bodies (clippy reports part of the family under each lint).

## Test Strategy

- `./sink/tests.rs` (feature-gated): frames driven through both the sink's `TraceSink::emit` and a real `Tracer::with_sink`; asserts span names/kinds, `gen_ai.*`/`openjiuwen.*` values, OK/ERROR/interrupted statuses, parent/child span ids, tool dedup, cross-dimension parenting, and buffer flushes.
- `./sink/tests/lazy_open.rs`: `Tracer::with_sink` + `start_span`/`start_child`/`start_workflow_span` + `finish`/`fail`/`interrupted` with no synthetic start frame; asserts the lazy open exports exactly one span with the frame's timestamps, the `running`-only frame stays open, and every terminal status/ordering (child first, child after the parent's state was popped) still exports.
- `./agent_handler/tests.rs`: the LLM/tool/prompt/chain/retriever/embedding/memory/evaluator/workflow callbacks over a `CaptureExporter`, tool dedup (`bug #2046`), error attributes, session fallback, `from_handle`.
- `./workflow_handler/tests.rs`: root/component naming and kinds, retrieval query/documents, layer-root and sub-workflow parenting, cross-workflow isolation, buffer flushes, error/interrupt paths, stale-span cleanup on trace-id change.
- Unit tests beside `redaction.rs` (all `test_redaction.py` cases: exact truncation marker, hash formula, field override truth table, system-prompt non-truncation), `span_names.rs` (every naming rule), `config.rs` (sample-rate validation, env toggle precedence), `handle.rs` (no-op vs active), `attrs.rs`, `llm_attrs.rs`.
- Every test module always calls `force_flush()` before reading captured `SpanData`; Python counterparts are named in comments.
- `./mod.rs` module docs carry three doctests that compile as external crates: the seam (`init_otel_tracer` active/inactive), the `TraceSink` integration (one exported `chat {model}` span from a terminal frame), and the typed handlers (LLM span attributes + `bug #2046` tool dedup, and the workflow buffer flush); the handler examples are wrapped in `# #[cfg(feature = "otel")] # { .. # }` so the default build stays green.
- The crate-boundary integration suites `../tests/otel.rs` (`#![cfg(feature = "otel")]`) and `../tests/otel_seam.rs` assert the public surface from outside the crate: the seam (`init_otel_tracer`, `is_active`, deferred OTLP), one exported `chat {model}` span through a session `Tracer`, the default error-status path, and the `bug #2046` tool dedup.

## Routing Table

- `./config.rs` → `OtelTracerConfig`, exporter/protocol enums, validation, env toggle
- `./semconv.rs` → attribute key/value constants
- `./redaction.rs` → truncate/hash/redact helpers
- `./span_names.rs` → span naming rules for agent/workflow spans
- `./handle.rs` → `TracerHandle`, `init_otel_tracer`, no-op seam when the feature is off
- `./provider.rs` → SDK provider construction, in-memory + console exporters, `CaptureExporter`, `tracing_layer`
- `./attrs.rs` → shared span start/end/error/interrupt attribute writers and string helpers
- `./llm_attrs.rs` → LLM payload normalization (`standard_messages`), GenAI request/response/usage attributes, retrieval extraction
- `./span_manager.rs` → `OtelSpanState`, agent/workflow span managers, workflow buffers, tool dedup probe
- `./sink/` → `OtelTraceSink` (`TraceSink` impl): `mod.rs` public type + emit dispatch/lifecycle, `frames.rs` agent-path callbacks plus the helpers shared with the workflow path, `frames/workflow.rs` workflow/component callbacks (classifier, buffers, lazy open), `tests.rs` + `tests/lazy_open.rs`
- `./agent_handler/` → `OtelAgentHandler`: `mod.rs` lifecycle + LLM callbacks, `non_llm.rs` non-LLM callback families, `tests.rs`
- `./workflow_handler/` → `OtelWorkflowHandler`: `mod.rs` lifecycle callbacks, `spans.rs` parent resolution + workflow attribute writers, `tests.rs`
