//! Non-LLM agent callbacks: tool, prompt, chain, retriever, embedding, memory,
//! evaluator, and workflow frames.
//!
//! Port of the non-LLM half of `OtelAgentHandler` in
//! `openjiuwen/extensions/tracer_otel/handler.py` (v0.1.20). Every family is the
//! same three-step lifecycle — start a [`SpanKind::Internal`] span through
//! [`OtelAgentHandler::start_non_llm_span`], record the family-specific `gen_ai.*`
//! extras, then close it through
//! [`OtelAgentHandler::end_non_llm_span`] or
//! [`OtelAgentHandler::error_non_llm_span`].
//!
//! The tool family additionally carries the duplicate-event probe of
//! `bug #2046` (`_is_duplicate_tool_event`) and flags the rail's own span so the
//! decorator's nested twin is dropped exactly once.
//!
//! Every callback is fail-open: the bodies are infallible by construction (no
//! `unwrap` on payload data, no panics, no blocking work), so a malformed frame
//! can never abort the business flow.
//!
//! [`SpanKind::Internal`]: opentelemetry::trace::SpanKind::Internal

use opentelemetry::trace::Span as _;
use opentelemetry::KeyValue;
use serde_json::{Map, Value};

use openjiuwen_common::error::Error;
use openjiuwen_core::session::tracer::{InvokeType, TraceSpan};

use super::super::attrs;
use super::super::semconv::{
    GEN_AI_AGENT_NAME, GEN_AI_DATA_SOURCE_ID, GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_MEMORY_RECORD_COUNT, GEN_AI_OPERATION_EMBEDDINGS, GEN_AI_OPERATION_EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME, GEN_AI_REQUEST_MODEL, GEN_AI_RESPONSE_MODEL, GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME, GEN_AI_TOOL_TYPE, GEN_AI_USAGE_INPUT_TOKENS,
};
use super::{
    is_truthy, text_field, truthy_text, NonLlmSpan, OtelAgentHandler, LOG_TARGET, RAIL_PLUGIN_EVENT,
};

/// Prefix of the prompt span (`{prefix}.{class_name}`).
const PROMPT_PREFIX: &str = "prompt";

/// Prefix of the chain span.
const CHAIN_PREFIX: &str = "chain";

/// Prefix of the retriever span.
const RETRIEVER_PREFIX: &str = "retriever";

/// Prefix of the embedding span, which the explicit name overrides.
const EMBEDDING_PREFIX: &str = "embedding";

/// Prefix of the memory span.
const MEMORY_PREFIX: &str = "memory";

/// Prefix of the evaluator span.
const EVALUATOR_PREFIX: &str = "evaluator";

/// Prefix of the agent-level workflow span.
const WORKFLOW_PREFIX: &str = "workflow";

/// The `openjiuwen.agent.invoke_type` of embedding frames.
///
/// Python passes the bare string `"embedding"` (there is no `InvokeType`
/// variant for it), so the family declares its own value.
const EMBEDDING_INVOKE_TYPE: &str = "embedding";

/// The `openjiuwen.agent.invoke_type` of memory frames.
///
/// Python passes the bare string `"memory"` for the same reason as embedding.
const MEMORY_INVOKE_TYPE: &str = "memory";

/// Model name reported when an embedding frame forwards none.
///
/// Python counterpart: `str(instance_info.get("model_name") or
/// instance_info.get("class_name") or "unknown")`.
const UNKNOWN_MODEL_NAME: &str = "unknown";

/// The tool callbacks: a tool invocation is an `execute_tool {name}` span.
impl OtelAgentHandler {
    /// Starts the span of a tool (plugin) invocation.
    ///
    /// Python counterpart: `on_plugin_start`. A decorator event that duplicates
    /// an open rail span is dropped (`bug #2046`); otherwise the span carries the
    /// `GenAI` tool identity (`gen_ai.operation.name = execute_tool`,
    /// `gen_ai.tool.name`, the optional call id/type, and the executing agent)
    /// plus the redacted inputs. A rail event (`instance_info["type"] ==
    /// "plugin"`) flags the state so the nested decorator event is deduped.
    pub async fn on_plugin_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value) {
        // Dedupe is a best-effort optimization and must fail open: the probe is
        // infallible here (no panics), and a non-duplicate event always creates
        // its span.
        if self.is_duplicate_tool_event(span, instance_info) {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %span.invoke_id,
                "otel agent handler: duplicate tool event skipped"
            );
            return;
        }
        let class_name = text_field(instance_info, "class_name").unwrap_or_default();
        let mut spec = NonLlmSpan::new(InvokeType::Plugin.as_str(), "tool");
        spec.extra_attrs.push((
            GEN_AI_OPERATION_NAME,
            GEN_AI_OPERATION_EXECUTE_TOOL.to_owned(),
        ));
        spec.extra_attrs
            .push((GEN_AI_TOOL_NAME, class_name.to_owned()));
        if let Some(tool_type) = truthy_text(instance_info.get("tool_type")) {
            spec.extra_attrs.push((GEN_AI_TOOL_TYPE, tool_type));
        }
        // Agent executing the tool; the rail forwards it, and direct callers
        // fall back to the tool name.
        let agent_name =
            truthy_text(instance_info.get("agent_name")).unwrap_or_else(|| class_name.to_owned());
        spec.extra_attrs.push((GEN_AI_AGENT_NAME, agent_name));
        if let Some(tool_call_id) = truthy_text(instance_info.get("tool_call_id")) {
            spec.extra_attrs.push((GEN_AI_TOOL_CALL_ID, tool_call_id));
        }
        self.start_non_llm_span(span, inputs, instance_info, &spec);
        if text_field(instance_info, "type") == Some(RAIL_PLUGIN_EVENT) {
            // The rail's own event (the decorator's carries type="tool"): flag
            // the state so the dedup does not depend on the optional call id.
            self.span_manager.with_state(&span.invoke_id, |state| {
                state.from_tool_rail = true;
            });
        }
    }

    /// Closes the span of a successful tool invocation.
    ///
    /// Python counterpart: `on_plugin_end` → `_end_non_llm_span`. A call whose
    /// start was deduped has no open span, so this is a no-op.
    pub async fn on_plugin_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed tool invocation.
    ///
    /// Python counterpart: `on_plugin_error`.
    pub async fn on_plugin_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The prompt callbacks: a `prompt.{name}` span.
impl OtelAgentHandler {
    /// Starts the span of a prompt invocation.
    ///
    /// Python counterpart: `on_prompt_start`.
    pub async fn on_prompt_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value) {
        self.start_non_llm_span(
            span,
            inputs,
            instance_info,
            &NonLlmSpan::new(InvokeType::Prompt.as_str(), PROMPT_PREFIX),
        );
    }

    /// Closes the span of a successful prompt invocation.
    ///
    /// Python counterpart: `on_prompt_end`.
    pub async fn on_prompt_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed prompt invocation.
    ///
    /// Python counterpart: `on_prompt_error`.
    pub async fn on_prompt_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The chain callbacks: a `chain.{name}` span.
impl OtelAgentHandler {
    /// Starts the span of a chain invocation.
    ///
    /// Python counterpart: `on_chain_start`.
    pub async fn on_chain_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value) {
        self.start_non_llm_span(
            span,
            inputs,
            instance_info,
            &NonLlmSpan::new(InvokeType::Chain.as_str(), CHAIN_PREFIX),
        );
    }

    /// Closes the span of a successful chain invocation.
    ///
    /// Python counterpart: `on_chain_end`.
    pub async fn on_chain_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed chain invocation.
    ///
    /// Python counterpart: `on_chain_error`.
    pub async fn on_chain_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The retriever callbacks: a `retriever.{name}` span plus the retrieval facts.
impl OtelAgentHandler {
    /// Starts the span of a retrieval invocation.
    ///
    /// Python counterpart: `on_retriever_start` — after the shared start, the
    /// inputs supply `gen_ai.retrieval.top_k` (`top_k`, then `k`) and
    /// `gen_ai.data_source.id` (`data_source_id`, then `data_source`).
    pub async fn on_retriever_start(
        &self,
        span: &TraceSpan,
        inputs: &Value,
        instance_info: &Value,
    ) {
        self.start_non_llm_span(
            span,
            inputs,
            instance_info,
            &NonLlmSpan::new(InvokeType::Retriever.as_str(), RETRIEVER_PREFIX),
        );
        let Some(payload) = inputs.as_object() else {
            return;
        };
        let top_k = first_truthy(payload, &["top_k", "k"]).and_then(json_i64);
        let data_source =
            first_truthy(payload, &["data_source_id", "data_source"]).map(attrs::serialize_value);
        self.with_recorded_state(&span.invoke_id, |state| {
            if let Some(top_k) = top_k {
                state
                    .span
                    .set_attribute(KeyValue::new(GEN_AI_RETRIEVAL_TOP_K, top_k));
            }
            if let Some(data_source) = data_source.as_deref() {
                state.record_attribute(GEN_AI_DATA_SOURCE_ID, data_source);
            }
        });
    }

    /// Closes the span of a successful retrieval invocation.
    ///
    /// Python counterpart: `on_retriever_end`.
    pub async fn on_retriever_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed retrieval invocation.
    ///
    /// Python counterpart: `on_retriever_error`.
    pub async fn on_retriever_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The embedding callbacks: an `embeddings {model}` span (issue #1833).
impl OtelAgentHandler {
    /// Starts the span of an embedding invocation.
    ///
    /// Python counterpart: `on_embedding_start` — the `GenAI` naming rule gives
    /// `embeddings {gen_ai.request.model}` with the model taken from
    /// `model_name`, then `class_name`, then `"unknown"`; the inputs supply
    /// `gen_ai.embeddings.dimension.count` (`dimension`, then `dimensions`).
    pub async fn on_embedding_start(
        &self,
        span: &TraceSpan,
        inputs: &Value,
        instance_info: &Value,
    ) {
        let model_name = truthy_text(instance_info.get("model_name"))
            .or_else(|| truthy_text(instance_info.get("class_name")))
            .unwrap_or_else(|| UNKNOWN_MODEL_NAME.to_owned());
        let mut spec = NonLlmSpan::new(EMBEDDING_INVOKE_TYPE, EMBEDDING_PREFIX);
        spec.span_name = Some(format!("embeddings {model_name}"));
        spec.extra_attrs.push((
            GEN_AI_OPERATION_NAME,
            GEN_AI_OPERATION_EMBEDDINGS.to_owned(),
        ));
        spec.extra_attrs.push((GEN_AI_REQUEST_MODEL, model_name));
        self.start_non_llm_span(span, inputs, instance_info, &spec);
        let dimension = inputs
            .as_object()
            .and_then(|payload| first_truthy(payload, &["dimension", "dimensions"]))
            .and_then(json_i64);
        self.with_recorded_state(&span.invoke_id, |state| {
            if let Some(dimension) = dimension {
                state
                    .span
                    .set_attribute(KeyValue::new(GEN_AI_EMBEDDINGS_DIMENSION_COUNT, dimension));
            }
        });
    }

    /// Closes the span of a successful embedding invocation.
    ///
    /// Python counterpart: `on_embedding_end` — the service-side facts the core
    /// emitter captured from the raw response (`response_model`, `dimension`,
    /// `usage_input_tokens`) are projected onto the span before the shared end.
    pub async fn on_embedding_end(&self, span: &TraceSpan, outputs: &Value) {
        if let Some(payload) = outputs.as_object() {
            let response_model = truthy_text(payload.get("response_model"));
            let dimension = payload.get("dimension").and_then(json_i64);
            // Python requires a real `int` here, not a numeric string.
            let usage_input_tokens = payload.get("usage_input_tokens").and_then(Value::as_i64);
            self.with_recorded_state(&span.invoke_id, |state| {
                if let Some(response_model) = response_model.as_deref() {
                    state.record_attribute(GEN_AI_RESPONSE_MODEL, response_model);
                }
                if let Some(dimension) = dimension {
                    state
                        .span
                        .set_attribute(KeyValue::new(GEN_AI_EMBEDDINGS_DIMENSION_COUNT, dimension));
                }
                if let Some(usage_input_tokens) = usage_input_tokens {
                    state.span.set_attribute(KeyValue::new(
                        GEN_AI_USAGE_INPUT_TOKENS,
                        usage_input_tokens,
                    ));
                }
            });
        }
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed embedding invocation.
    ///
    /// Python counterpart: `on_embedding_error`.
    pub async fn on_embedding_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The memory callbacks: a `memory.{name}` span.
impl OtelAgentHandler {
    /// Starts the span of a memory invocation.
    ///
    /// Python counterpart: `on_memory_start` — the inputs supply
    /// `gen_ai.memory.record.count` (`record_count`, then `count`).
    pub async fn on_memory_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value) {
        self.start_non_llm_span(
            span,
            inputs,
            instance_info,
            &NonLlmSpan::new(MEMORY_INVOKE_TYPE, MEMORY_PREFIX),
        );
        let record_count = inputs
            .as_object()
            .and_then(|payload| first_truthy(payload, &["record_count", "count"]))
            .and_then(json_i64);
        self.with_recorded_state(&span.invoke_id, |state| {
            if let Some(record_count) = record_count {
                state
                    .span
                    .set_attribute(KeyValue::new(GEN_AI_MEMORY_RECORD_COUNT, record_count));
            }
        });
    }

    /// Closes the span of a successful memory invocation.
    ///
    /// Python counterpart: `on_memory_end`.
    pub async fn on_memory_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed memory invocation.
    ///
    /// Python counterpart: `on_memory_error` (absent upstream; the family
    /// exposes the same lifecycle as its siblings).
    pub async fn on_memory_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The evaluator callbacks: an `evaluator.{name}` span.
impl OtelAgentHandler {
    /// Starts the span of an evaluator invocation.
    ///
    /// Python counterpart: `on_evaluator_start`.
    pub async fn on_evaluator_start(
        &self,
        span: &TraceSpan,
        inputs: &Value,
        instance_info: &Value,
    ) {
        self.start_non_llm_span(
            span,
            inputs,
            instance_info,
            &NonLlmSpan::new(InvokeType::Evaluator.as_str(), EVALUATOR_PREFIX),
        );
    }

    /// Closes the span of a successful evaluator invocation.
    ///
    /// Python counterpart: `on_evaluator_end`.
    pub async fn on_evaluator_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed evaluator invocation.
    ///
    /// Python counterpart: `on_evaluator_error`.
    pub async fn on_evaluator_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// The agent-level workflow callbacks: a `workflow.{name}` span.
impl OtelAgentHandler {
    /// Starts the span of an agent-level workflow invocation.
    ///
    /// Python counterpart: `on_workflow_start`. The workflow *dimension* of a
    /// run is owned by the sibling `OtelWorkflowHandler`; this family covers the
    /// workflow frame an agent emits while it delegates.
    pub async fn on_workflow_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value) {
        self.start_non_llm_span(
            span,
            inputs,
            instance_info,
            &NonLlmSpan::new(InvokeType::Workflow.as_str(), WORKFLOW_PREFIX),
        );
    }

    /// Closes the span of a successful agent-level workflow invocation.
    ///
    /// Python counterpart: `on_workflow_end`.
    pub async fn on_workflow_end(&self, span: &TraceSpan, outputs: &Value) {
        self.end_non_llm_span(span, outputs);
    }

    /// Closes the span of a failed agent-level workflow invocation.
    ///
    /// Python counterpart: `on_workflow_error`.
    pub async fn on_workflow_error(&self, span: &TraceSpan, error: &Error) {
        self.error_non_llm_span(span, error);
    }
}

/// Returns the first key of `keys` whose value is truthy in Python's sense.
///
/// Python counterpart: the `info.get("top_k") or info.get("k")` chains of the
/// family-specific attribute blocks.
fn first_truthy<'a>(info: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|key| info.get(*key).filter(|value| is_truthy(value)))
}

/// Reads a JSON scalar as an `i64`, mirroring Python's `int()` coercion.
///
/// A JSON integer converts directly and a float truncates toward zero; a numeric
/// string parses after trimming. Any other shape yields `None` instead of
/// raising, so one malformed counter cannot abort the span finalization.
#[allow(clippy::cast_possible_truncation)]
fn json_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|number| number as i64)),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}
