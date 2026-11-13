//! OpenTelemetry tracing over the session [`TraceSink`] SPI.
//!
//! Ports `openjiuwen/extensions/tracer_otel/` (v0.1.20). Sessions emit
//! synchronous trace frames, so the primary seam is a sink (`OtelTraceSink`,
//! `otel` feature) plus typed handlers (`OtelAgentHandler`, `OtelWorkflowHandler`)
//! for callers that already hold a [`TraceSpan`].
//!
//! Everything that decides *what a span looks like* — configuration, semantic
//! conventions, redaction, span names — compiles with or without the `otel`
//! feature, so hosts can build configuration and assert naming rules without
//! pulling the SDK in. Without the feature, [`init_otel_tracer`] validates the
//! configuration and returns an inactive handle instead of failing.
//!
//! # Seam
//!
//! ```
//! use openjiuwen_extensions::otel::{init_otel_tracer, ExporterType, OtelTracerConfig};
//!
//! // The in-memory exporter needs no collector. Without the `otel` feature the
//! // handle is an inactive no-op seam, so the call still succeeds.
//! let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Memory);
//! let handle = init_otel_tracer(config).expect("the memory exporter always initializes");
//! assert_eq!(handle.is_active(), cfg!(feature = "otel"));
//! assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
//! handle.force_flush();
//! handle.shutdown();
//! handle.shutdown(); // idempotent
//! ```
//!
//! # Tracer integration
//!
//! ```
//! # #[cfg(feature = "otel")]
//! # {
//! use std::sync::Arc;
//!
//! use openjiuwen_core::session::tracer::{InvokeType, TraceSink, Tracer};
//! use openjiuwen_extensions::otel::{
//!     init_otel_tracer, ExporterType, OtelTraceSink, OtelTracerConfig,
//! };
//!
//! let handle = init_otel_tracer(
//!     OtelTracerConfig::default().with_exporter_type(ExporterType::Memory),
//! )
//! .expect("the memory exporter initializes");
//! let sink: Arc<dyn TraceSink> = OtelTraceSink::new(&handle).expect("a live handle builds a sink");
//! let tracer = Tracer::with_sink("session-1", "trace-1", sink);
//!
//! let span = tracer.start_span("Model", InvokeType::Llm);
//! span.set_inputs(serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}));
//! span.finish(serde_json::json!({"content": "hello"}));
//! handle.force_flush();
//!
//! // The session tracer emits one terminal frame per span; the sink opens the
//! // OTel span from that frame, so a finished span is still exported.
//! let exporter = handle.capture().expect("the memory exporter is attached");
//! assert_eq!(exporter.names(), vec!["chat Model".to_owned()]);
//! # }
//! ```
//!
//! # Handler callbacks
//!
//! The typed handlers are the seam for callers that already hold a
//! [`TraceSpan`]; both are driven synchronously, so any executor works.
//!
//! ```
//! # #[cfg(feature = "otel")]
//! # {
//! use futures::executor::block_on;
//! use openjiuwen_core::session::tracer::{TraceSpan, TraceSpanKind};
//! use openjiuwen_extensions::otel::{
//!     init_otel_tracer, ExporterType, OtelAgentHandler, OtelTracerConfig,
//! };
//! use opentelemetry::trace::{SpanKind, Status};
//! use serde_json::json;
//!
//! let config = OtelTracerConfig {
//!     redaction_enabled: false, // keep the attribute values readable
//!     ..OtelTracerConfig::default()
//! }
//! .with_exporter_type(ExporterType::Memory);
//! let handle = init_otel_tracer(config).expect("the memory exporter initializes");
//! let handler = OtelAgentHandler::from_handle(&handle).expect("the handle is active");
//!
//! // An LLM invocation: `chat {model}` with the GenAI conventions.
//! let frame = TraceSpan::new("trace-1", "", "invoke-1", "", TraceSpanKind::Agent);
//! let info = json!({"class_name": "TestModel"});
//! block_on(handler.on_llm_start(&frame, &json!("hello"), &info));
//! block_on(handler.on_llm_end(&frame, &json!({"content": "world"})));
//!
//! handle.force_flush();
//! let exporter = handle.capture().expect("the memory exporter is attached");
//! let span = exporter
//!     .span_named("chat TestModel")
//!     .expect("the LLM span is exported");
//! let attribute = |key: &str| {
//!     span.attributes
//!         .iter()
//!         .rev()
//!         .find(|pair| pair.key.as_str() == key)
//!         .map(|pair| pair.value.as_str().into_owned())
//! };
//! assert_eq!(span.name.as_ref(), "chat TestModel");
//! assert_eq!(span.span_kind, SpanKind::Client);
//! assert_eq!(attribute("gen_ai.operation.name").as_deref(), Some("chat"));
//! assert_eq!(attribute("gen_ai.provider.name").as_deref(), Some("openjiuwen"));
//! assert_eq!(span.status, Status::Ok);
//! assert_eq!(attribute("openjiuwen.status").as_deref(), Some("finish"));
//!
//! // `bug #2046`: the tool rail opens the span, its decorator twin is deduped.
//! let rail = TraceSpan::new("trace-1", "", "invoke-rail", "", TraceSpanKind::Agent);
//! let rail_info = json!({"class_name": "get_weather", "type": "plugin"});
//! block_on(handler.on_plugin_start(&rail, &json!(null), &rail_info));
//! assert!(handler.has_open_tool_span("get_weather", "trace-1"));
//!
//! let decorator = TraceSpan::new("trace-1", "", "invoke-decorator", "", TraceSpanKind::Agent);
//! let decorator_info = json!({"class_name": "get_weather", "type": "tool"});
//! block_on(handler.on_plugin_start(&decorator, &json!(null), &decorator_info));
//! block_on(handler.on_plugin_end(&rail, &json!("22C")));
//!
//! handle.force_flush();
//! assert_eq!(
//!     exporter.names(),
//!     ["chat TestModel", "execute_tool get_weather"],
//! );
//! # }
//! ```
//!
//! Workflow engines buffer their payloads and flush them on the terminal
//! callback:
//!
//! ```
//! # #[cfg(feature = "otel")]
//! # {
//! use futures::executor::block_on;
//! use openjiuwen_extensions::otel::{
//!     init_otel_tracer, ExporterType, OtelTracerConfig, OtelWorkflowHandler,
//! };
//! use serde_json::json;
//!
//! let config = OtelTracerConfig {
//!     redaction_enabled: false, // keep the attribute values readable
//!     ..OtelTracerConfig::default()
//! }
//! .with_exporter_type(ExporterType::Memory);
//! let handle = init_otel_tracer(config).expect("the memory exporter initializes");
//! let handler = OtelWorkflowHandler::from_handle(&handle).expect("the handle is active");
//!
//! let metadata = json!({"workflow_id": "wf1", "workflow_name": "MyWorkflow"});
//! block_on(handler.on_call_start("wf_root", &metadata, None));
//! block_on(handler.on_invoke("wf_root", &json!({"step": 1})));
//! block_on(handler.on_call_done("wf_root", None));
//!
//! handle.force_flush();
//! let exporter = handle.capture().expect("the memory exporter is attached");
//! let span = exporter
//!     .span_named("wf_root")
//!     .expect("the workflow span is exported");
//! let attribute = |key: &str| {
//!     span.attributes
//!         .iter()
//!         .rev()
//!         .find(|pair| pair.key.as_str() == key)
//!         .map(|pair| pair.value.as_str().into_owned())
//! };
//! assert_eq!(
//!     attribute("openjiuwen.workflow.invoke_data").as_deref(),
//!     Some(r#"[{"step":1}]"#),
//! );
//! assert_eq!(attribute("openjiuwen.status").as_deref(), Some("finish"));
//! # }
//! ```
//!
//! # OTLP export
//!
//! [`ExporterType::Otlp`] ships finished spans to a collector over OTLP/HTTP
//! with JSON payloads. That is the only transport this crate speaks: gRPC with
//! protobuf framing needs `opentelemetry-otlp`, `prost`, and `tonic`, which the
//! offline build cannot reach, so [`OtlpProtocol::Grpc`] — the configuration
//! default — answers
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) instead
//! of being downgraded silently. Select the supported transport with
//! [`OtlpProtocol::Http`] or `OTEL_EXPORTER_OTLP_PROTOCOL=http/json`.
//!
//! Every setting resolves in one order: the value on the configuration, then
//! the `OPENJIUWEN_OTEL_*` variable, then the standard `OTEL_*` variable, then
//! the built-in default ([`DEFAULT_OTLP_HTTP_ENDPOINT`],
//! `http/json`, no extra headers). The rules live in [`mod@env`].
//! `export_timeout_ms` bounds one HTTP request, `schedule_delay_millis` and
//! `max_export_batch_size` shape the batch processor, and a failed export is
//! logged at `warn` and dropped rather than propagated to the caller.
//!
//! Shutdown keeps the tail of the stream: [`TracerHandle::shutdown`] flushes the
//! batch processor and then stops the provider within `export_timeout_ms`, and
//! [`TracerHandle::shutdown_with_timeout`] takes an explicit budget; both are
//! idempotent and both are no-ops for an inactive handle.
//!
//! The doctest below builds a handle towards a local collector and never
//! exports a span, so it needs no running collector.
//!
//! ```
//! # #[cfg(feature = "otel")]
//! # {
//! use std::time::Duration;
//!
//! use openjiuwen_extensions::otel::{
//!     init_otel_tracer, ExporterType, OtelTracerConfig, OtlpProtocol,
//! };
//!
//! let config = OtelTracerConfig {
//!     protocol: OtlpProtocol::Http,
//!     exporter_endpoint: Some("http://localhost:4318".to_owned()),
//!     export_timeout_ms: 1_000,
//!     ..OtelTracerConfig::default()
//! }
//! .with_exporter_type(ExporterType::Otlp);
//! let handle = init_otel_tracer(config).expect("OTLP/HTTP initializes offline");
//! assert!(handle.is_active());
//! assert!(handle.capture().is_none(), "OTLP exposes no capture API");
//!
//! // No span was created, so the shutdown performs no HTTP request.
//! handle.shutdown_with_timeout(Duration::from_millis(250));
//! assert!(!handle.is_active());
//! # }
//! ```
//!
//! # Integrations
//!
//! [`ExporterType::Otlp`] with [`OtlpProtocol::Http`] is the transport the
//! presets below share: finished spans leave the process as OTLP/HTTP JSON
//! batches.
//!
//! - `grafana` — [`GrafanaCloudConfig`] turns a Grafana Cloud stack into the
//!   gateway preset (`https://otlp-gateway-<zone>.grafana.net/otlp`) plus the
//!   Basic `authorization` header built from the stack instance id and an
//!   access-policy token. `GrafanaAgentConfig` keeps the self-hosted Grafana
//!   Agent/Alloy push API-shaped and answers
//!   [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported): the
//!   collector service has to exist and run outside this process.
//! - `langfuse` — [`LangfuseConfig`] addresses the OTLP endpoint
//!   `{host}/api/public/otel` with the project keys as Basic credentials, and
//!   [`LangfuseClient`] posts batched trace/span/generation events to
//!   `POST {host}/api/public/ingestion` through [`IngestionBatch`].
//!
//! Neither preset contacts its backend at configuration time: requests happen
//! when a span batch is exported, or when [`LangfuseClient`] is asked to
//! ingest.
//!
//! ```
//! use openjiuwen_extensions::otel::{
//!     GrafanaAgentConfig, GrafanaCloudConfig, LangfuseConfig,
//! };
//!
//! let grafana = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0")
//!     .with_service_name("checkout");
//! assert_eq!(
//!     grafana.endpoint(),
//!     "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
//! );
//! assert!(grafana.authorization().starts_with("Basic "));
//! let otel = grafana.otel_config()?;
//! assert_eq!(
//!     otel.exporter_endpoint.as_deref(),
//!     Some(grafana.endpoint().as_str()),
//! );
//! assert_eq!(otel.headers["authorization"], grafana.authorization());
//!
//! let langfuse = LangfuseConfig::new("pk-lf-public", "sk-lf-secret")
//!     .with_host("https://langfuse.internal");
//! assert_eq!(
//!     langfuse.endpoint(),
//!     "https://langfuse.internal/api/public/otel",
//! );
//! assert_eq!(
//!     langfuse.ingestion_url(),
//!     "https://langfuse.internal/api/public/ingestion",
//! );
//! assert!(langfuse.otel_config()?.headers["authorization"].starts_with("Basic "));
//!
//! // The self-hosted Grafana Agent/Alloy push needs an external collector.
//! let error = GrafanaAgentConfig::new("http://localhost:4318")
//!     .otel_config()
//!     .expect_err("a collector has to run outside the process");
//! assert!(error.to_string().contains("collector"));
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```
//!
//! [`TraceSink`]: openjiuwen_core::session::tracer::TraceSink
//! [`TraceSpan`]: openjiuwen_core::session::tracer::TraceSpan
//! [`TracerHandle::shutdown`]: self::handle::TracerHandle::shutdown
//! [`TracerHandle::shutdown_with_timeout`]: self::handle::TracerHandle::shutdown_with_timeout

pub mod config;
pub mod env;
pub mod grafana;
pub mod handle;
pub mod langfuse;
pub mod redaction;
pub mod semconv;
pub mod span_names;

#[cfg(feature = "otel")]
mod agent_handler;
#[cfg(feature = "otel")]
mod attrs;
#[cfg(feature = "otel")]
mod llm_attrs;
#[cfg(feature = "otel")]
pub mod otlp;
#[cfg(feature = "otel")]
pub mod provider;
#[cfg(feature = "otel")]
pub mod sink;
#[cfg(feature = "otel")]
pub mod span_manager;
#[cfg(feature = "otel")]
mod workflow_handler;

pub use config::{global_instrument_enabled, ExporterType, OtelTracerConfig, OtlpProtocol};
pub use env::DEFAULT_OTLP_HTTP_ENDPOINT;
pub use grafana::{GrafanaAgentConfig, GrafanaCloudConfig};
pub use handle::{init_otel_tracer, TracerHandle};
pub use langfuse::{
    GenerationBody, IngestionBatch, IngestionEvent, IngestionEventType, IngestionSummary,
    LangfuseClient, LangfuseConfig, SpanBody, TraceBody,
};
pub use redaction::{hash_value, truncate, RedactionField, Redactor};
pub use span_names::{
    chat_span_name, chat_span_name_for_model, embeddings_span_name, is_llm_component,
    is_retrieval_component, is_tool_component, legacy_tool_span_name, simple_span_name,
    tool_span_name, workflow_component_span_name,
};

#[cfg(feature = "otel")]
pub use agent_handler::OtelAgentHandler;
#[cfg(feature = "otel")]
pub use handle::init_otel_tracer_with_exporter;
#[cfg(feature = "otel")]
pub use otlp::OtlpHttpExporter;
#[cfg(feature = "otel")]
pub use provider::{tracing_layer, CaptureExporter};
#[cfg(feature = "otel")]
pub use sink::OtelTraceSink;
#[cfg(feature = "otel")]
pub use span_manager::{OtelAgentSpanManager, OtelSpanState, OtelWorkflowSpanManager};
#[cfg(feature = "otel")]
pub use workflow_handler::OtelWorkflowHandler;
