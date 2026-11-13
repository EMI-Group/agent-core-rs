# otel/ — OpenTelemetry tracing over the session `TraceSink` SPI

## Intent

Port of `openjiuwen/extensions/tracer_otel/` (v0.1.20): turn session trace frames into OpenTelemetry spans with the Python span names, attributes, statuses, redaction, and parent/child links.
Python registers async handler callbacks on the session tracer registry; Rust sessions emit synchronous [`TraceSink`] frames instead, so the primary seam is a sink (`OtelTraceSink`) plus typed handlers (`OtelAgentHandler`, `OtelWorkflowHandler`) for callers that already hold a `TraceSpan`.
Finished spans leave the process through one of four exporters: in-memory capture (`Memory`), one `tracing` line per span (`Console`), nothing (`None`), or OTLP/HTTP with JSON payloads (`Otlp`, in `otlp/`).
Two presets wrap the OTLP exporter for hosted backends: the Grafana Cloud gateway (`grafana/`) and Langfuse (`langfuse/`).
Everything that decides *what a span looks like* — configuration, environment resolution, semantic conventions, redaction, span names — plus both presets compile without the `otel` feature; only the SDK-backed exporter, sink, and handlers need it.
Without the feature, `init_otel_tracer` validates the configuration and returns an inactive handle for every exporter type instead of failing.

## API Surface

```rust
// config.rs (always compiled)
pub enum ExporterType { None, Memory, Console, Otlp }   // serde snake_case; Default = Otlp (Python default)
pub enum OtlpProtocol { Grpc, Http }                    // Default = Grpc (Python default); Http = JSON payloads
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
    pub fn from_env() -> Self;                                     // default() + apply_env(), infallible
    pub fn apply_env(self) -> Self;                                // fills only still-default fields
}
pub fn global_instrument_enabled(config: &OtelTracerConfig) -> bool; // OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE wins ("1"/"true"/"yes"/"on")

// env.rs (always compiled) — OTLP knob resolution, no SDK types
pub const OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT;   // "OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT"
pub const OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS;    // "OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS"
pub const OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL;   // "OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL"
pub const OPENJIUWEN_OTEL_SERVICE_NAME;             // "OPENJIUWEN_OTEL_SERVICE_NAME"
pub const OTEL_EXPORTER_OTLP_TRACES_ENDPOINT;       // trace-specific standard endpoint
pub const OTEL_EXPORTER_OTLP_ENDPOINT;              // all-signal standard endpoint
pub const OTEL_EXPORTER_OTLP_HEADERS;
pub const OTEL_EXPORTER_OTLP_PROTOCOL;
pub const OTEL_SERVICE_NAME;
pub const DEFAULT_OTLP_HTTP_ENDPOINT: &str;         // "http://localhost:4318" (no /v1/traces suffix)
pub fn parse_headers(raw: &str) -> BTreeMap<String, String>;      // "k=v,k2=v2"; last duplicate wins; split on the first '='
pub fn env_headers() -> Option<BTreeMap<String, String>>;         // OPENJIUWEN_* then OTEL_*; None when blank/unparsable
pub fn env_endpoint() -> Option<String>;                          // OPENJIUWEN_*, OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_EXPORTER_OTLP_ENDPOINT
pub fn env_protocol_value() -> Option<String>;                    // OPENJIUWEN_* then OTEL_*, returned verbatim for parse_protocol
pub fn env_service_name() -> Option<String>;                      // OPENJIUWEN_* then OTEL_SERVICE_NAME
pub fn resolve_endpoint(explicit: Option<&str>) -> String;        // explicit -> env -> DEFAULT_OTLP_HTTP_ENDPOINT
pub fn resolve_headers(explicit: &BTreeMap<String, String>) -> BTreeMap<String, String>; // explicit only when non-empty
pub fn resolve_protocol(explicit: OtlpProtocol) -> Result<OtlpProtocol>; // Http wins; else env; else explicit unchanged
pub fn parse_protocol(raw: &str) -> Result<OtlpProtocol>;         // case-insensitive; unsupported/validation errors
pub fn resolve_service_name(explicit: &str) -> String;            // "openjiuwen" counts as unset

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
    pub fn shutdown(&self);                             // flush, then provider shutdown within export_timeout_ms
    pub fn shutdown_with_timeout(&self, timeout: Duration);  // explicit budget; idempotent in both builds
    // feature "otel" only
    pub fn tracer(&self) -> Option<SdkTracer>;
    pub fn provider(&self) -> Option<&SdkTracerProvider>;
    pub fn capture(&self) -> Option<Arc<CaptureExporter>>;
}
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle>;
// without `otel`: validates the config and returns an inactive handle (no error for any exporter type)
// with `otel`:      every ExporterType builds a real provider; Otlp builds an OtlpHttpExporter and
//                   answers Error::Unsupported only when the protocol resolves to OtlpProtocol::Grpc
pub fn init_otel_tracer_with_exporter<E: SpanExporter + 'static>(config, exporter) -> Result<TracerHandle>;
// the injected exporter wins for every ExporterType, including Otlp (config protocol/endpoint unused)

// otlp/ (feature "otel")
pub struct OtlpHttpExporter { .. }   // Debug + Send + Sync; Debug shows header names, never values
impl OtlpHttpExporter {
    pub fn new(endpoint: impl Into<String>, headers: BTreeMap<String, String>, timeout: Duration) -> Result<Self>;
    pub fn traces_url(&self) -> &str;               // always ends in /v1/traces
}
// impl SpanExporter: export / shutdown_with_timeout / force_flush (no-op) / set_resource
pub fn traces_url(endpoint: &str) -> String;        // base or traces URL -> exactly one /v1/traces

// grafana/ (always compiled; `exporter()` is feature "otel")
pub struct GrafanaCloudConfig { .. }
impl GrafanaCloudConfig {
    pub fn new(instance_id: impl Into<String>, token: impl Into<String>, zone: impl Into<String>) -> Self;
    pub fn with_service_name(self, name: impl Into<String>) -> Self;
    pub fn with_zone(self, zone: impl Into<String>) -> Self;
    pub fn with_otlp_endpoint(self, endpoint: impl Into<String>) -> Self;
    pub fn zone(&self) -> &str;
    pub fn service_name(&self) -> &str;
    pub fn endpoint(&self) -> String;               // https://otlp-gateway-<zone>.grafana.net/otlp
    pub fn authorization(&self) -> String;          // "Basic base64(instance_id:token)"
    pub fn validate(&self) -> Result<()>;
    pub fn otel_config(&self) -> Result<OtelTracerConfig>;  // ExporterType::Otlp + OtlpProtocol::Http
    pub fn from_env() -> Result<Self>;
    #[cfg(feature = "otel")] pub fn exporter(&self) -> Result<OtlpHttpExporter>;
}
pub struct GrafanaAgentConfig { .. }   // new(endpoint), with_service_name, endpoint, service_name -> Option<&str>,
                                      // otel_config()/exporter() always Error::Unsupported (external collector)
pub const GRAFANA_INSTANCE_ID, GRAFANA_API_TOKEN, GRAFANA_ZONE, GRAFANA_OTLP_ZONE, GRAFANA_OTLP_ENDPOINT: &str;

// langfuse/ (always compiled; `exporter()` is feature "otel")
pub struct LangfuseConfig { .. }
impl LangfuseConfig {
    pub fn new(public_key: impl Into<String>, secret_key: impl Into<String>) -> Self;
    pub fn with_host(self, host: impl Into<String>) -> Self;
    pub fn with_service_name(self, name: impl Into<String>) -> Self;
    pub fn host(&self) -> &str;
    pub fn service_name(&self) -> &str;
    pub fn endpoint(&self) -> String;               // {host}/api/public/otel
    pub fn ingestion_url(&self) -> String;          // {host}/api/public/ingestion
    pub fn authorization(&self) -> String;          // "Basic base64(public_key:secret_key)"
    pub fn validate(&self) -> Result<()>;
    pub fn otel_config(&self) -> Result<OtelTracerConfig>;  // ExporterType::Otlp + OtlpProtocol::Http
    pub fn from_env() -> Result<Self>;
    #[cfg(feature = "otel")] pub fn exporter(&self) -> Result<OtlpHttpExporter>;
}
pub struct LangfuseClient { .. }   // new, from_env, with_http_client, with_max_batch_size, config,
                                   // max_batch_size, ingest(IngestionBatch), ingest_events(Vec<IngestionEvent>)
pub struct IngestionSummary { pub events: usize, pub requests: usize }  // Copy + Default
pub enum IngestionEventType { TraceCreate, SpanCreate, GenerationCreate }  // kebab-case wire strings
pub struct IngestionEvent { .. }   // new(type, body), trace/span/generation(body) -> Result, with_id,
                                   // with_timestamp, to_json
pub struct TraceBody / SpanBody / GenerationBody { .. }  // camelCase wire bodies, `new` plus `with_*` builders
pub struct IngestionBatch { .. }   // new, with_capacity, push, extend, len, is_empty, events, into_events, max_size_hint
pub const DEFAULT_MAX_BATCH_SIZE: usize = 1000;
pub const LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, LANGFUSE_BASEURL, DEFAULT_HOST: &str;

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

The crate root re-exports `ExporterType`, `OtelTracerConfig`, `OtlpProtocol`, `global_instrument_enabled`, `DEFAULT_OTLP_HTTP_ENDPOINT`, `GrafanaAgentConfig`, `GrafanaCloudConfig`, the Langfuse config/client/event/batch types, `TracerHandle`, `init_otel_tracer`, `RedactionField`, `Redactor`, `truncate`, `hash_value`, and the span-name helpers; `init_otel_tracer_with_exporter`, `OtlpHttpExporter`, `CaptureExporter`, `tracing_layer`, `OtelTraceSink`, the span managers, and both handlers are re-exported only with `otel`.
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

## OTLP export

`ExporterType::Otlp` installs `OtlpHttpExporter`, which posts finished spans to a collector over OTLP/HTTP with JSON payloads (`Content-Type: application/json`, the spec's `http/json`).
That is the only transport this crate speaks: gRPC and protobuf framing need `opentelemetry-otlp`, `prost`, and `tonic`, which the offline registry does not carry.
`OtlpProtocol::Grpc` is therefore the configuration default (Python parity) *and* unsupported: building the exporter answers `Error::Unsupported` with a message naming `http/json`, so nothing is downgraded silently.
`OtlpProtocol::Http` (or `OTEL_EXPORTER_OTLP_PROTOCOL=http/json`) selects the supported transport, and an explicitly configured `Http` always wins over the environment.

Protocol mapping (`env::parse_protocol`, ASCII case-insensitive): `http/json` → `Http`; `http` → `Http` — Python's alias, transported here as JSON rather than protobuf (the documented deviation; the endpoint and collector contract are identical); `http/protobuf` → `Error::Unsupported` (no protobuf encoder); `grpc` → `Error::Unsupported` (no gRPC transport); anything else → `Error::Validation` naming the accepted values.

Endpoint, headers, protocol, and `service.name` resolve in one order per setting: the value on `OtelTracerConfig`, then the `OPENJIUWEN_OTEL_*` variable, then the standard `OTEL_*` variable, then the built-in default (`http://localhost:4318`, `openjiuwen`, no extra headers).
A blank (empty or whitespace-only) value counts as unset at every stage.
Endpoints prefer `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` over `OTEL_EXPORTER_OTLP_ENDPOINT`; header lists follow the `k=v,k2=v2` grammar with both sides trimmed, the last duplicate key winning, and the split on the first `=` so base64 credentials such as `Basic abc==` survive.
`resolve_protocol` consults the environment only while the explicit value is `Grpc`, `resolve_headers` treats an empty configuration map as unset, and `resolve_service_name` treats the `openjiuwen` default as unset.
`OtelTracerConfig::from_env`/`apply_env` fill only still-default fields and are infallible: an unusable protocol string is ignored at configuration load and reported when the exporter is built.

`OtlpHttpExporter::new` accepts a base URL or a complete traces URL, rejects a non-`http`/`https` or empty endpoint and invalid header names/values with `Error::Validation`, and reports a failed worker spawn as `Error::Config`; `endpoint::traces_url` appends `/v1/traces` exactly once (trailing slashes dropped first).
Construction starts the worker thread but performs no I/O; the first export opens the connection.

The transport exists because of a hard SDK constraint: `opentelemetry_sdk` 0.31's batch processor calls `SpanExporter::export` on a plain `std::thread` driven by `futures_executor::block_on`, where no tokio reactor is installed — building or awaiting a `reqwest` request there panics ("there is no reactor running").
`otlp/transport.rs` therefore owns one `std::thread` (named `openjiuwen-otlp-http`) that builds a current-thread tokio runtime and the single `reqwest::Client`, drains an unbounded tokio channel with `blocking_recv`, and reports each job over a oneshot channel; `export` awaits that reply under whatever executor called it (the batch processor's `block_on`, a tokio task, or a plain test).
Headers are attached to every request, `content-type` is always `application/json`, and header *values* are never logged (`Debug` prints the names only).

`export_timeout_ms` sizes the per-request HTTP timeout (minimum 1 ms) and the default `TracerHandle::shutdown` budget; the batch processor keeps its SDK default because `BatchConfigBuilder::with_max_export_timeout` is gated behind the SDK's `experimental_trace_batch_span_processor_with_async_runtime` feature, which this crate does not enable (`OTEL_BSP_EXPORT_TIMEOUT` therefore governs it).
`schedule_delay_millis` and `max_export_batch_size` configure the processor as before.

`TracerHandle::shutdown_with_timeout` flushes the batch processor first and then stops the provider within the budget, so a pending OTLP batch is handed to the exporter instead of being dropped; both it and `shutdown` are idempotent and are no-ops for an inactive handle.
On the exporter side `shutdown_with_timeout` closes the job channel, waits for the worker, and detaches it with a `warn` when the budget expires; the worker exits after the job in flight and shuts its runtime down with a 100 ms grace period. `force_flush` on the exporter is a no-op because nothing is buffered there.

Every export failure is fail-open: it surfaces as `OTelSdkError::InternalFailure` and never panics, and the transport-level failures (non-2xx status, timeout, connect, post-shutdown, closed channel, lost reply) also emit a `tracing::warn!` on target `openjiuwen_extensions::otel::otlp`.
A non-2xx response names the status and a body snippet capped at 512 characters; timeouts and connect failures are named as such; a poisoned exporter lock is recovered rather than panicking on the batch-processor thread.

The body is the OTLP/JSON `ExportTraceServiceRequest`: one `resourceSpans` entry whose `resource` object carries the provider's resource attributes (empty object when it has none), and one `scopeSpans` entry per distinct `(name, version)` instrumentation scope, in first-seen order with span order preserved.
Each span carries `traceId`/`spanId`/`parentSpanId` as lowercase hex (the parent omitted when invalid), `name`, `kind` (`SPAN_KIND_*`), `startTimeUnixNano`/`endTimeUnixNano` as decimal strings (JSON numbers cannot hold 64-bit nanoseconds; pre-epoch clamps to `"0"`), `attributes`, `droppedAttributesCount`, `events` (time, name, attributes, dropped count), `links`, and `status` (`STATUS_CODE_UNSET`/`_OK`/`_ERROR`, `message` only for a described error).
Attribute values encode as `boolValue`, `intValue` (decimal string), `doubleValue`, `stringValue`, or `arrayValue` with a `values` array; a non-exhaustive `Value` variant falls back to its `Display` text as `stringValue` instead of dropping the attribute.

## Integrations

`grafana::GrafanaCloudConfig` turns a Grafana Cloud stack into an OTLP/HTTP JSON preset: the gateway endpoint `https://otlp-gateway-<zone>.grafana.net/otlp` plus the Basic `authorization` header `base64(instance_id:token)`, exposed through `otel_config()` and, with `otel`, `exporter()`.
`grafana::GrafanaAgentConfig` keeps the self-hosted Grafana Agent/Alloy push API-shaped and answers `Error::Unsupported`: a collector service has to exist and run outside this process, and this crate neither starts nor discovers one.

`langfuse::LangfuseConfig` addresses the OTLP endpoint `{host}/api/public/otel` with `base64(public_key:secret_key)` as Basic credentials, and `langfuse::LangfuseClient` posts batched `trace-create`/`span-create`/`generation-create` events to `POST {host}/api/public/ingestion`.
`IngestionBatch` buffers events in insertion order and `LangfuseClient::ingest` splits them into requests of at most `max_batch_size` events (`DEFAULT_MAX_BATCH_SIZE` = 1000) and returns an `IngestionSummary { events, requests }`.
Ingestion is partial: a 2xx (including 207) response succeeds only when its JSON reports no `errors`; a non-empty `errors` array, a malformed `errors` field, a non-JSON 2xx body, and every non-2xx status become `Error::Tool` naming the status and a truncated body or the failing ids/statuses/messages.

Neither preset contacts its backend at configuration time: requests happen when a span batch is exported, or when `LangfuseClient` is asked to ingest.
Both presets and their environment variables (`GRAFANA_*`, `LANGFUSE_*`) compile in every feature mode; only `exporter()` and the OTLP handle need `otel`.

## Constraints

- `otel` feature deps: `opentelemetry`, `opentelemetry_sdk`, `tracing-opentelemetry`, `tracing-subscriber`, `chrono`; `serde_json`, `reqwest`, `url`, `tokio`, `futures`, `base64`, `openjiuwen-core`, and `tracing` are unconditional crate dependencies (the OTLP transport, the presets, and `env` compile without the feature).
- The provider is per-handle, never the process-global `opentelemetry::global` provider (Python keeps a private provider too); `TracerHandle::shutdown` flushes and then stops the private provider with the timed shutdown, and is idempotent.
- `Console` uses an in-crate line-per-span exporter writing through `tracing` at `debug` (no stdout writes).
- The exporter's worker thread owns the only reactor on the export path; nothing calls `block_on` inside a tokio worker, and `export` only awaits a oneshot reply.
- Tests must not read the real clock for assertions: elapsed/start-time assertions only check presence/format.
- Never take two span-manager locks at once: `OtelWorkflowSpanManager::with_state` holds the manager lock, so buffer snapshots (`on_invoke_data`/`stream_inputs`/`stream_outputs`) are read *before* entering `with_state`.
- Files ≤ ~1000 lines, one concern per file; no `unsafe`; no blocking calls; no `println!`.
- `agent_handler` and `workflow_handler` are private modules whose public types are re-exported at the `otel` root; each carries a documented `#![allow(clippy::unused_async, clippy::unused_async_trait_impl)]` because the callbacks keep Python's async signatures with synchronous bodies (clippy reports part of the family under each lint).

## Test Strategy

- `./sink/tests.rs` (feature-gated): frames driven through both the sink's `TraceSink::emit` and a real `Tracer::with_sink`; asserts span names/kinds, `gen_ai.*`/`openjiuwen.*` values, OK/ERROR/interrupted statuses, parent/child span ids, tool dedup, cross-dimension parenting, and buffer flushes.
- `./sink/tests/lazy_open.rs`: `Tracer::with_sink` + `start_span`/`start_child`/`start_workflow_span` + `finish`/`fail`/`interrupted` with no synthetic start frame; asserts the lazy open exports exactly one span with the frame's timestamps, the `running`-only frame stays open, and every terminal status/ordering (child first, child after the parent's state was popped) still exports.
- `./agent_handler/tests.rs`: the LLM/tool/prompt/chain/retriever/embedding/memory/evaluator/workflow callbacks over a `CaptureExporter`, tool dedup (`bug #2046`), error attributes, session fallback, `from_handle`.
- `./workflow_handler/tests.rs`: root/component naming and kinds, retrieval query/documents, layer-root and sub-workflow parenting, cross-workflow isolation, buffer flushes, error/interrupt paths, stale-span cleanup on trace-id change.
- Unit tests beside `redaction.rs` (all `test_redaction.py` cases: exact truncation marker, hash formula, field override truth table, system-prompt non-truncation), `span_names.rs` (every naming rule), `config.rs` (sample-rate validation, env toggle precedence, `from_env`/`apply_env` default-only filling and ignored unusable protocols), `handle.rs` (no-op vs active, OTLP/HTTP builds an active handle without a capture API, post-shutdown deactivation), `attrs.rs`, `llm_attrs.rs`.
- `./env.rs` unit tests (port of the `test_setup.py` env cases): header grammar, `OPENJIUWEN_*` before `OTEL_*`, trace-specific before generic endpoint, blank-as-unset, protocol mapping including both `Unsupported` values and the validation error, and every `resolve_*` fallback; they share one `ENV_LOCK` and an `EnvGuard` that clears and restores the process environment.
- `./otlp/tests.rs` drives `export()` from plain `#[test]` functions with `futures::executor::block_on` — deliberately reactor-free, so a reactor dependency in the exporter would hang or panic — and covers endpoint normalization, constructor validation, `Send + Sync + Debug`, the JSON body field by field (scope grouping, hex ids, kinds, decimal nanosecond timestamps, typed attributes, events, links, statuses, resource-less bodies), posting to the traces path through `wiremock`, non-2xx handling, the client timeout, and resource/shutdown/post-shutdown behavior.
- `./grafana/tests.rs` covers endpoint derivation for every accepted zone form, the exact Basic credentials, `otel_config()` field by field, every validation error, `from_env()` precedence, the collector deferral, and `exporter().traces_url()` under `otel`.
- `./langfuse/tests.rs` covers the preset (default and custom hosts, path normalization, credentials, `otel_config()`, validation, `from_env()` precedence, `exporter()`), and `./langfuse/client/tests.rs` drives `wiremock` for the POST path, headers, the per-type batch body, the success summary, a 207 with `errors`, a 500, a non-JSON 2xx, chunking, the empty batch, and `max_batch_size(0)`.
- Every test module that reads captured `SpanData` calls `force_flush()` first; Python counterparts are named in comments, and environment tests serialize on a `static Mutex` and never contact the network.
- Doctests that compile as external crates: `./mod.rs` carries six (the seam, the `TraceSink` integration, the two typed-handler examples, the OTLP/HTTP handle, and the presets), while `./otlp/mod.rs` and `./otlp/endpoint.rs` document the exporter and `traces_url`; the SDK-backed examples are wrapped in `# #[cfg(feature = "otel")] # { .. # }` so the default build stays green.
- The crate-boundary suites assert the public surface from outside the crate: `../tests/otel.rs` (`#![cfg(feature = "otel")]`) pins one exported `chat {model}` span through a session `Tracer`, the default error-status path, and the `bug #2046` tool dedup; `../tests/otel_seam.rs` (both builds) pins `init_otel_tracer` for all four exporter types with `OtlpProtocol::Http`, the `Grpc` default answering `Error::Unsupported`, and idempotent shutdown; `../tests/otel_otlp.rs` (`otel`) posts a whole batch to a loopback `wiremock` collector and asserts the wire payload, environment precedence at both levels, the typed failure paths (status, timeout, post-shutdown), and the gRPC rejection; `../tests/otel_integrations.rs` (both builds, with `otel` branches) covers both presets and the Langfuse ingestion contract.

## Routing Table

- `./config.rs` → `OtelTracerConfig`, exporter/protocol enums, validation, `from_env`/`apply_env`, env toggle
- `./env.rs` → OTLP variable names, header grammar, endpoint/protocol/`service.name` precedence and resolution
- `./semconv.rs` → attribute key/value constants
- `./redaction.rs` → truncate/hash/redact helpers
- `./span_names.rs` → span naming rules for agent/workflow spans
- `./handle.rs` → `TracerHandle`, `init_otel_tracer`, exporter dispatch (including OTLP), timed shutdown, no-op seam when the feature is off
- `./otlp/` → OTLP/HTTP JSON exporter: `mod.rs` public type + `SpanExporter` impl, `json.rs` body encoder, `transport.rs` dedicated-thread runtime and job channel, `endpoint.rs` `/v1/traces` normalization, `tests.rs`
- `./grafana/` → Grafana Cloud gateway preset and the deferred Grafana Agent/Alloy preset (`mod.rs`, `tests.rs`)
- `./langfuse/` → Langfuse OTLP preset (`mod.rs`), `events.rs` wire bodies, `batch.rs` `IngestionBatch`, `client.rs` REST ingestion, `tests.rs`, `client/tests.rs`
- `./provider.rs` → SDK provider construction, in-memory + console exporters, `CaptureExporter`, `tracing_layer`
- `./attrs.rs` → shared span start/end/error/interrupt attribute writers and string helpers
- `./llm_attrs.rs` → LLM payload normalization (`standard_messages`), GenAI request/response/usage attributes, retrieval extraction
- `./span_manager.rs` → `OtelSpanState`, agent/workflow span managers, workflow buffers, tool dedup probe
- `./sink/` → `OtelTraceSink` (`TraceSink` impl): `mod.rs` public type + emit dispatch/lifecycle, `frames.rs` agent-path callbacks plus the helpers shared with the workflow path, `frames/workflow.rs` workflow/component callbacks (classifier, buffers, lazy open), `tests.rs` + `tests/lazy_open.rs`
- `./agent_handler/` → `OtelAgentHandler`: `mod.rs` lifecycle + LLM callbacks, `non_llm.rs` non-LLM callback families, `tests.rs`
- `./workflow_handler/` → `OtelWorkflowHandler`: `mod.rs` lifecycle callbacks, `spans.rs` parent resolution + workflow attribute writers, `tests.rs`
