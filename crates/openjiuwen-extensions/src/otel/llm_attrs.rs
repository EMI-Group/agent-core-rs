//! LLM payload normalization and GenAI request/response attributes.
//!
//! Port of the GenAI attribute helpers of
//! `openjiuwen/extensions/tracer_otel/handler.py` (v0.1.20): the payload
//! normalizer (`_standard_messages`, `_unwrap_llm_payload`, `_redacted_json`),
//! the request-attribute block of `OtelAgentHandler.on_llm_start`, the
//! response block of `OtelAgentHandler.on_llm_end`
//! (`_set_llm_response_attrs`), the retrieval projections
//! (`_extract_retrieval_query`, `_extract_retrieval_documents`), and the agent
//! identity block (`_set_agent_identity_attrs`).
//!
//! The helpers take the concrete [`Span`] and a [`serde_json::Value`] payload,
//! so the agent and workflow handlers that call them stay thin: they resolve
//! trace-span identity, open/close the span, and delegate the `gen_ai.*` /
//! `openjiuwen.*` value shape to this module. Everything is fail-open — a
//! payload of the wrong shape skips a single attribute, never panics, so a
//! malformed provider response cannot abort span finalization.
//!
//! # Deviations from Python
//!
//! - `_normalize_llm_payload` has no Rust counterpart: tracer payloads are
//!   already [`serde_json::Value`], so there is no Pydantic model to walk with
//!   `model_dump()`; message objects arrive as JSON objects.
//! - `openjiuwen.agent.invoke_type` / `openjiuwen.agent.name` are derived from
//!   a `TraceSpan` field that only the agent handler holds, so they stay there.
//! - Python stringifies a truthy non-string `role`; here only a non-empty
//!   string counts, and anything else falls back to `default_role`.
//! - A `null` payload yields no instructions and no messages (Python's caller
//!   guards `inputs is not None`, so the two agree on every real payload).
//! - Python renders scalars with `str()` (`True` → `"True"`); [`serialize_value`]
//!   emits their JSON text (`"true"`), matching the rest of the crate.
//! - Python's `int()`/`float()` coercions are mirrored with a numeric-string
//!   fallback: any other shape skips only that attribute instead of raising.
//!
//! Every attribute key comes from [`super::semconv`]; this module never
//! re-types a wire literal.

// The agent handler and the sink that call these helpers land in the
// immediately following commits; until then no crate code calls them, so the
// module opts out of the dead-code lint instead of hiding every helper behind
// an `#[allow]` (same precedent as `attrs.rs`). The unit tests below exercise
// every helper. The polish pass removes this attribute once the callers exist.
#![allow(dead_code)]

use opentelemetry::trace::Span as _;
use opentelemetry::{Array, KeyValue, StringValue, Value as OtelValue};
use opentelemetry_sdk::trace::Span;
use serde_json::{Map, Value};

use super::attrs::{serialize_value, set_text_attribute};
use super::config::OtelTracerConfig;
use super::redaction::{RedactionField, Redactor};
use super::semconv::{
    GEN_AI_AGENT_DESCRIPTION, GEN_AI_AGENT_ID, GEN_AI_AGENT_NAME, GEN_AI_CONVERSATION_ID,
    GEN_AI_INPUT_MESSAGES, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_NAME, GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL, GEN_AI_REQUEST_REASONING_LEVEL, GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE, GEN_AI_REQUEST_TOP_K, GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS, GEN_AI_RESPONSE_MODEL, GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_SYSTEM_INSTRUCTIONS, GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, OJ_GEN_AI_METADATA, OJ_GEN_AI_REASONING_DURATION_MS,
    OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS, OJ_GEN_AI_TRACE_NAME, OJ_GEN_AI_USAGE_INPUT_COST,
    OJ_GEN_AI_USAGE_OUTPUT_COST, OJ_GEN_AI_USAGE_TOTAL_COST, OJ_GEN_AI_USER_ID,
    OJ_LLM_PREV_MESSAGE_COUNT,
};

/// Role assumed for an LLM input that does not declare one.
///
/// Python counterpart: the `default_role="user"` argument of every
/// `_standard_messages` call site in `on_llm_start`.
const DEFAULT_INPUT_ROLE: &str = "user";

/// Object keys that wrap a message list, in precedence order.
///
/// Python counterpart: the `("messages", "inputs")` tuple of
/// `_standard_messages` and the `("messages", "outputs", "inputs")` tuple of
/// `_unwrap_llm_payload`; the first one applies to inputs, the second to
/// responses.
const INPUT_CONTAINERS: [&str; 2] = ["messages", "inputs"];

/// Response wrapper keys, in precedence order.
const RESPONSE_CONTAINERS: [&str; 3] = ["messages", "outputs", "inputs"];

/// Message keys whose value becomes content, in precedence order.
const CONTENT_KEYS: [&str; 3] = ["content", "response", "output"];

/// Tool-call fields copied onto a `tool_call` part, in emission order.
const TOOL_CALL_FIELDS: [&str; 3] = ["id", "name", "arguments"];

/// Request parameters written as `f64` attributes.
const FLOAT_REQUEST_PARAMETERS: [(&str, &str); 3] = [
    ("temperature", GEN_AI_REQUEST_TEMPERATURE),
    ("top_p", GEN_AI_REQUEST_TOP_P),
    ("top_k", GEN_AI_REQUEST_TOP_K),
];

/// Usage counters written as `i64` attributes, 0 when the payload omits them.
const INTEGER_USAGE_FIELDS: [(&str, &str); 3] = [
    ("input_tokens", GEN_AI_USAGE_INPUT_TOKENS),
    ("output_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
    ("reasoning_tokens", GEN_AI_USAGE_REASONING_OUTPUT_TOKENS),
];

/// Usage costs written as `f64` attributes, 0.0 when the payload omits them.
const COST_USAGE_FIELDS: [(&str, &str); 3] = [
    ("total_cost", OJ_GEN_AI_USAGE_TOTAL_COST),
    ("input_cost", OJ_GEN_AI_USAGE_INPUT_COST),
    ("output_cost", OJ_GEN_AI_USAGE_OUTPUT_COST),
];

/// Streaming/reasoning latencies written only when present and parseable.
const LATENCY_USAGE_FIELDS: [(&str, &str); 2] = [
    (
        "inter_token_latency_ms",
        OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS,
    ),
    ("reasoning_duration_ms", OJ_GEN_AI_REASONING_DURATION_MS),
];

/// Normalizes an LLM input payload into GenAI `instructions` and `messages`.
///
/// Python counterpart: `_standard_messages` in `tracer_otel/handler.py`. The
/// value is unwrapped from its `messages`/`inputs` container (or used as an
/// array, or wrapped as a single item) and every item becomes a message with a
/// `role` and a list of `parts`; `system`-role parts become `instructions` and
/// every other role becomes a message.
///
/// A `null` payload yields two empty vectors (see the module docs). A non-empty
/// string `role` wins over `default_role`; content is taken from `content`,
/// then `response`, then `output`; tool calls append `{"type": "tool_call"}`
/// parts.
#[must_use]
pub(crate) fn standard_messages(value: &Value, default_role: &str) -> (Vec<Value>, Vec<Value>) {
    if value.is_null() {
        return (Vec::new(), Vec::new());
    }
    let container = match value {
        Value::Object(map) => INPUT_CONTAINERS.iter().find_map(|key| match map.get(*key) {
            Some(Value::Array(items)) => Some(items.as_slice()),
            _ => None,
        }),
        _ => None,
    };
    let items = container.unwrap_or_else(|| match value {
        Value::Array(items) => items.as_slice(),
        single => std::slice::from_ref(single),
    });

    let mut instructions: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for item in items {
        // Python counterpart: `item if isinstance(item, dict) else {"content": item}`.
        let message: Map<String, Value> = match item {
            Value::Object(map) => map.clone(),
            other => {
                let mut map = Map::new();
                map.insert("content".to_owned(), other.clone());
                map
            }
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| !role.is_empty())
            .unwrap_or(default_role);
        let mut parts: Vec<Value> = Vec::new();
        if let Some(content) = content_value(&message) {
            parts.push(serde_json::json!({
                "type": "text",
                "content": serialize_value(content),
            }));
        }
        extend_tool_call_parts(&message, &mut parts);
        if role == "system" {
            instructions.extend(parts);
        } else {
            messages.push(serde_json::json!({ "role": role, "parts": parts }));
        }
    }
    (instructions, messages)
}

/// Resolves the content of a normalized message.
///
/// Python counterpart: `content = message.get("content")` followed by
/// `content = message.get("response", message.get("output"))` when content is
/// absent. `null` values are skipped, so a message without content yields no
/// text part at all.
fn content_value(message: &Map<String, Value>) -> Option<&Value> {
    CONTENT_KEYS
        .iter()
        .find_map(|key| message.get(*key).filter(|value| !value.is_null()))
}

/// Appends one `tool_call` part per entry of the message's `tool_calls` array.
///
/// Python counterpart: the `tool_calls` loop of `_standard_messages`. Each
/// field prefers the call's own key and falls back to the nested `function`
/// object; an absent or `null` value is skipped rather than emitted as `null`.
fn extend_tool_call_parts(message: &Map<String, Value>, parts: &mut Vec<Value>) {
    let Some(Value::Array(calls)) = message.get("tool_calls") else {
        return;
    };
    for call in calls {
        let Some(call) = call.as_object() else {
            continue;
        };
        let function = call.get("function").and_then(Value::as_object);
        let mut part = Map::new();
        part.insert("type".to_owned(), Value::String("tool_call".to_owned()));
        for field in TOOL_CALL_FIELDS {
            let value = match call.get(field) {
                Some(Value::Null) | None => function.and_then(|function| function.get(field)),
                Some(value) => Some(value),
            };
            if let Some(value) = value.filter(|value| !value.is_null()) {
                part.insert(field.to_owned(), value.clone());
            }
        }
        parts.push(Value::Object(part));
    }
}

/// Unwraps the single-key container an LLM rail wraps around a response.
///
/// Python counterpart: `_unwrap_llm_payload`. The first present
/// `messages`/`outputs`/`inputs` value is returned; anything else passes
/// through unchanged so the caller can decide whether it is an object.
fn unwrap_payload(value: &Value) -> &Value {
    let Some(map) = value.as_object() else {
        return value;
    };
    RESPONSE_CONTAINERS
        .iter()
        .find_map(|key| map.get(*key).filter(|inner| !inner.is_null()))
        .unwrap_or(value)
}

/// Returns an owned copy of the payload an LLM rail wrapped in a container.
///
/// Python counterpart: `_unwrap_llm_payload`. This is an additive wrapper over
/// the borrowed [`unwrap_payload`] for callers that outlive the borrow (the
/// agent handler's `on_llm_end` only holds the payload by reference while it
/// normalizes the completion messages).
#[must_use]
pub(crate) fn unwrap_llm_payload(value: &Value) -> Value {
    unwrap_payload(value).clone()
}

/// Serializes a payload with the configured redaction policy applied.
///
/// Python counterpart: `_redacted_json`. The object walk lives in
/// [`Redactor::redact_json`]; this wrapper pins the `field`/`system_prompt`
/// argument shape and documents that the result is compact JSON with non-ASCII
/// text preserved.
// The redactor is a two-field `Copy` view; it stays behind a reference to keep
// the call shape of the sibling helpers (and of Python's `_redacted_json`) and
// so a future non-`Copy` policy need not change every call site.
#[allow(clippy::trivially_copy_pass_by_ref)]
#[must_use]
pub(crate) fn redacted_json(
    value: &Value,
    redactor: &Redactor<'_>,
    field: RedactionField,
    system_prompt: bool,
) -> String {
    redactor.redact_json(value, Some(field), system_prompt)
}

/// Converts a JSON scalar into the typed OTel value of the same shape.
///
/// Objects, arrays, and `null` have no OTel scalar equivalent, so they yield
/// `None` and the caller skips that attribute. Numbers keep their JSON
/// integer/float distinction; string values are passed through verbatim.
fn scalar_value(value: &Value) -> Option<OtelValue> {
    match value {
        Value::Bool(flag) => Some(OtelValue::Bool(*flag)),
        Value::String(text) => Some(OtelValue::String(StringValue::from(text.clone()))),
        Value::Number(number) => number
            .as_i64()
            .map(OtelValue::I64)
            .or_else(|| number.as_f64().map(OtelValue::F64)),
        _ => None,
    }
}

/// Reads a JSON scalar as an `f64`, mirroring Python's `float()` coercion.
///
/// A JSON number converts directly (an integer widens); a numeric string
/// parses after trimming, which keeps the legacy latency fields that arrive as
/// text (`"0.35"`) working. Any other shape yields `None`.
// Widening an `i64` into an `f64` loses precision only past 2^53, far beyond
// every token count, cost, and latency this module writes.
#[allow(clippy::cast_precision_loss)]
fn scalar_f64(value: &Value) -> Option<f64> {
    match scalar_value(value)? {
        OtelValue::F64(value) => Some(value),
        OtelValue::I64(value) => Some(value as f64),
        OtelValue::String(value) => value.as_str().trim().parse().ok(),
        _ => None,
    }
}

/// Reads a JSON scalar as an `i64`, mirroring Python's `int()` coercion.
///
/// A JSON integer converts directly and a float truncates toward zero; a
/// numeric string parses after trimming. Any other shape yields `None` instead
/// of raising, so one malformed counter cannot abort the attribute block.
#[allow(clippy::cast_possible_truncation)]
fn scalar_i64(value: &Value) -> Option<i64> {
    match scalar_value(value)? {
        OtelValue::I64(value) => Some(value),
        OtelValue::F64(value) => Some(value as i64),
        OtelValue::String(value) => value.as_str().trim().parse().ok(),
        _ => None,
    }
}

/// Reports whether a JSON value is truthy in Python's `if value:` sense.
///
/// `null`, `false`, `0`, the empty string, and empty containers are falsy;
/// everything else is truthy. Python builds these blocks with bare truthiness
/// checks (`if info.get("metadata"):`, `if finish_reason:`).
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Returns a non-empty string field of an instance-info object.
fn text_field<'a>(info: Option<&'a Map<String, Value>>, key: &str) -> Option<&'a str> {
    info.and_then(|info| info.get(key))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

/// Writes the GenAI request attributes of an LLM span.
///
/// Python counterpart: the attribute block of `OtelAgentHandler.on_llm_start`
/// (without the `TraceSpan`-derived `openjiuwen.agent.*` pair, see the module
/// docs). `instance_info` is the rail-forwarded common info; a non-object is
/// treated as empty, so absent identity facts simply skip their attribute.
/// `inputs` is the raw LLM input payload; `config` drives redaction.
pub(crate) fn set_llm_request_attributes(
    span: &mut Span,
    inputs: &Value,
    instance_info: &Value,
    config: &OtelTracerConfig,
) {
    let info = instance_info.as_object();
    let class_name = info
        .and_then(|info| info.get("class_name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    set_text_attribute(span, GEN_AI_REQUEST_MODEL, class_name);
    set_text_attribute(span, GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_CHAT);
    // Trace display name: the executing agent, falling back to the model.
    let trace_name = text_field(info, "agent_name").unwrap_or(class_name);
    set_text_attribute(span, OJ_GEN_AI_TRACE_NAME, trace_name);
    set_identity_attributes(span, info);
    set_request_parameters(span, info);
    if let Some(count) = info
        .and_then(|info| info.get("message_count"))
        .and_then(scalar_i64)
    {
        span.set_attribute(KeyValue::new(OJ_LLM_PREV_MESSAGE_COUNT, count));
    }
    set_prompt_attributes(span, inputs, config);
}

/// Writes the agent identity / conversation attributes from rail common info.
///
/// Python counterpart: `_set_agent_identity_attrs`. Every identity fact is an
/// unbounded string (`agent_id`, `agent_name`, `agent_description`,
/// `conversation_id`, `user_id`) and is written only when non-empty; `metadata`
/// is serialized as JSON when truthy.
fn set_identity_attributes(span: &mut Span, info: Option<&Map<String, Value>>) {
    let Some(info) = info else {
        return;
    };
    for (source, key) in [
        ("agent_id", GEN_AI_AGENT_ID),
        ("agent_name", GEN_AI_AGENT_NAME),
        ("agent_description", GEN_AI_AGENT_DESCRIPTION),
        ("conversation_id", GEN_AI_CONVERSATION_ID),
        ("user_id", OJ_GEN_AI_USER_ID),
    ] {
        if let Some(text) = text_field(Some(info), source) {
            set_text_attribute(span, key, text);
        }
    }
    if let Some(metadata) = info.get("metadata").filter(|value| is_truthy(value)) {
        set_text_attribute(span, OJ_GEN_AI_METADATA, &serialize_value(metadata));
    }
}

/// Writes the GenAI request parameters forwarded by the rail.
///
/// Python counterpart: the `request_params` loop of `on_llm_start`. Numbers
/// keep their numeric type (`f64` for sampling knobs, `i64` for the token
/// budget), `stop_sequences` becomes an OTel string array, and
/// `reasoning_level` stays a string. A parameter of the wrong shape skips only
/// itself.
fn set_request_parameters(span: &mut Span, info: Option<&Map<String, Value>>) {
    let Some(params) = info
        .and_then(|info| info.get("request_params"))
        .and_then(Value::as_object)
    else {
        return;
    };
    for (name, key) in [
        ("temperature", GEN_AI_REQUEST_TEMPERATURE),
        ("top_p", GEN_AI_REQUEST_TOP_P),
        ("top_k", GEN_AI_REQUEST_TOP_K),
    ] {
        if let Some(value) = params.get(name).and_then(scalar_f64) {
            span.set_attribute(KeyValue::new(key, value));
        }
    }
    if let Some(value) = params.get("max_tokens").and_then(scalar_i64) {
        span.set_attribute(KeyValue::new(GEN_AI_REQUEST_MAX_TOKENS, value));
    }
    if let Some(Value::Array(sequences)) = params.get("stop_sequences") {
        let values: Vec<StringValue> = sequences
            .iter()
            .map(|sequence| StringValue::from(serialize_value(sequence)))
            .collect();
        span.set_attribute(KeyValue::new(
            GEN_AI_REQUEST_STOP_SEQUENCES,
            OtelValue::Array(Array::String(values)),
        ));
    }
    if let Some(level) = params.get("reasoning_level").and_then(Value::as_str) {
        span.set_attribute(KeyValue::new(
            GEN_AI_REQUEST_REASONING_LEVEL,
            level.to_owned(),
        ));
    }
}

/// Writes the GenAI system-instruction and input-message payloads.
///
/// Python counterpart: the `inputs is not None` block of `on_llm_start`.
/// Instructions use the system-prompt redaction policy (hashed or verbatim,
/// never truncated); messages use the prompt policy (hashed or truncated to
/// the configured attribute length). Empty halves write no attribute.
fn set_prompt_attributes(span: &mut Span, inputs: &Value, config: &OtelTracerConfig) {
    if inputs.is_null() {
        return;
    }
    let (instructions, messages) = standard_messages(inputs, DEFAULT_INPUT_ROLE);
    let redactor = Redactor::new(config);
    if !instructions.is_empty() {
        let text = redacted_json(
            &Value::Array(instructions),
            &redactor,
            RedactionField::Prompts,
            true,
        );
        set_text_attribute(span, GEN_AI_SYSTEM_INSTRUCTIONS, &text);
    }
    if !messages.is_empty() {
        let text = redacted_json(
            &Value::Array(messages),
            &redactor,
            RedactionField::Prompts,
            false,
        );
        set_text_attribute(span, GEN_AI_INPUT_MESSAGES, &text);
    }
}

/// Writes the GenAI response / usage / cost / latency attributes of an LLM span.
///
/// Python counterpart: `_set_llm_response_attrs`. The payload is unwrapped from
/// its `messages`/`outputs`/`inputs` container and must be an object;
/// `finish_reason` is written as a one-element array when it is a truthy,
/// non-`"null"` value, and every usage value requires a `usage_metadata`
/// object. Present-but-zero usage facts are recorded, because a recorded 0 is
/// a fact rather than an absence.
pub(crate) fn set_llm_response_attributes(span: &mut Span, outputs: &Value) {
    let Some(object) = unwrap_payload(outputs).as_object() else {
        return;
    };
    set_finish_reason(span, object);
    let Some(usage) = object.get("usage_metadata").and_then(Value::as_object) else {
        return;
    };
    set_usage_attributes(span, usage);
}

/// Writes `gen_ai.response.finish_reasons` when the payload reports one.
///
/// Python counterpart: the `finish_reason` branch of `_set_llm_response_attrs`.
/// `"null"` is the schema default for "the provider did not report one", so it
/// stays filtered along with the falsy shapes.
fn set_finish_reason(span: &mut Span, object: &Map<String, Value>) {
    let Some(reason) = object.get("finish_reason").filter(|value| is_truthy(value)) else {
        return;
    };
    let text = serialize_value(reason);
    if text == "null" {
        return;
    }
    span.set_attribute(KeyValue::new(
        GEN_AI_RESPONSE_FINISH_REASONS,
        OtelValue::Array(Array::String(vec![StringValue::from(text)])),
    ));
}

/// Writes the token, cache, cost, and latency facts of `usage_metadata`.
///
/// Python counterpart: the `usage` block of `_set_llm_response_attrs`. The
/// response model is always written (empty when absent); counters and costs
/// default to 0/0.0; the cache splits keep their legacy fallbacks; the latency
/// keys are schema-transparent and written only when present and parseable.
fn set_usage_attributes(span: &mut Span, usage: &Map<String, Value>) {
    let model = usage
        .get("model_name")
        .filter(|value| is_truthy(value))
        .map_or_else(String::new, serialize_value);
    set_text_attribute(span, GEN_AI_RESPONSE_MODEL, &model);

    for (name, key) in [
        ("input_tokens", GEN_AI_USAGE_INPUT_TOKENS),
        ("output_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
        ("reasoning_tokens", GEN_AI_USAGE_REASONING_OUTPUT_TOKENS),
    ] {
        let value = usage.get(name).and_then(scalar_i64).unwrap_or_default();
        span.set_attribute(KeyValue::new(key, value));
    }
    // Prefer the explicit cache_read_tokens split; fall back to cache_tokens.
    let cache_read = usage
        .get("cache_read_tokens")
        .filter(|value| is_truthy(value))
        .or_else(|| usage.get("cache_tokens").filter(|value| is_truthy(value)))
        .and_then(scalar_i64)
        .unwrap_or_default();
    span.set_attribute(KeyValue::new(
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        cache_read,
    ));
    let cache_write = usage
        .get("cache_write_tokens")
        .filter(|value| is_truthy(value))
        .or_else(|| {
            usage
                .get("cache_creation_input_tokens")
                .filter(|value| is_truthy(value))
        })
        .and_then(scalar_i64)
        .unwrap_or_default();
    span.set_attribute(KeyValue::new(
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        cache_write,
    ));

    for (name, key) in [
        ("total_cost", OJ_GEN_AI_USAGE_TOTAL_COST),
        ("input_cost", OJ_GEN_AI_USAGE_INPUT_COST),
        ("output_cost", OJ_GEN_AI_USAGE_OUTPUT_COST),
    ] {
        let value = usage.get(name).and_then(scalar_f64).unwrap_or_default();
        span.set_attribute(KeyValue::new(key, value));
    }

    set_time_to_first_chunk(span, usage);
    // Streaming/reasoning latencies are schema-transparent: written only when
    // present and parseable, never defaulted.
    for (name, key) in [
        (
            "inter_token_latency_ms",
            OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS,
        ),
        ("reasoning_duration_ms", OJ_GEN_AI_REASONING_DURATION_MS),
    ] {
        if let Some(value) = usage
            .get(name)
            .filter(|value| !value.is_null())
            .and_then(scalar_f64)
        {
            span.set_attribute(KeyValue::new(key, value));
        }
    }
}

/// Writes `gen_ai.response.time_to_first_chunk` from the latency fields.
///
/// Python counterpart: the TTFT branch of `_set_llm_response_attrs`.
/// `ttft_seconds` is the only semantic source; `first_token_time` remains a
/// legacy fallback that is consulted only when the explicit field is absent or
/// null. A zero latency is written as `0.0` (a fact), while an empty string or
/// a malformed value skips the attribute.
fn set_time_to_first_chunk(span: &mut Span, usage: &Map<String, Value>) {
    let ttft = match usage.get("ttft_seconds") {
        Some(value) if !value.is_null() => Some(value),
        _ => usage.get("first_token_time"),
    };
    if let Some(seconds) = ttft.and_then(scalar_f64) {
        span.set_attribute(KeyValue::new(GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK, seconds));
    }
}

/// Pulls the query text out of a retrieval component's inputs.
///
/// Python counterpart: `_extract_retrieval_query`. A non-empty string input is
/// the query; an object answers with its non-empty string `query`; any other
/// shape (including an empty string or an empty query) yields `None`.
#[must_use]
pub(crate) fn extract_retrieval_query(value: &Value) -> Option<String> {
    match value {
        Value::String(query) if !query.is_empty() => Some(query.clone()),
        Value::Object(map) => map
            .get("query")
            .and_then(Value::as_str)
            .filter(|query| !query.is_empty())
            .map(str::to_owned),
        _ => None,
    }
}

/// Pulls the retrieved documents out of a retrieval component's outputs.
///
/// Python counterpart: `_extract_retrieval_documents`. Only an object with a
/// non-empty `results` array yields documents; each item is rendered with
/// [`serialize_value`] (strings verbatim, other shapes as compact JSON rather
/// than Python's `str()` repr).
#[must_use]
pub(crate) fn extract_retrieval_documents(value: &Value) -> Option<Vec<String>> {
    let Value::Array(results) = value.as_object()?.get("results")? else {
        return None;
    };
    if results.is_empty() {
        return None;
    }
    Some(results.iter().map(serialize_value).collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! Port of the GenAI attribute assertions of
    //! `tests/unit_tests/extensions/tracer_otel/test_handler.py`:
    //! `TestGenAiSemconvAttrs`, the `_standard_messages` cases, the retrieval
    //! projection cases, and the `usage_metadata` latency cases.

    use super::*;
    use opentelemetry::trace::Tracer as _;
    use opentelemetry::{Context, Value as SpanValue};
    use opentelemetry_sdk::trace::SpanData;
    use serde_json::json;

    use crate::otel::config::{ExporterType, OtelTracerConfig};
    use crate::otel::handle::{init_otel_tracer, TracerHandle};

    /// Name of the span every export helper opens.
    const SPAN_NAME: &str = "chat Model";

    /// A configuration with redaction off: attribute text stays readable.
    fn plain_config() -> OtelTracerConfig {
        OtelTracerConfig {
            redaction_enabled: false,
            ..OtelTracerConfig::default()
        }
    }

    /// Runs `fill` on a fresh span, ends it, flushes, and returns its export.
    fn export(fill: impl FnOnce(&mut Span)) -> SpanData {
        let handle: TracerHandle =
            init_otel_tracer(OtelTracerConfig::default().with_exporter_type(ExporterType::Memory))
                .expect("the memory exporter always initializes");
        let mut span = handle
            .tracer()
            .expect("the memory handle is active")
            .start_with_context(SPAN_NAME, &Context::new());
        fill(&mut span);
        span.end();
        handle.force_flush();
        handle
            .capture()
            .expect("the memory handle captures")
            .span_named(SPAN_NAME)
            .expect("the span was exported")
    }

    /// The export of one LLM request span.
    fn request_span(inputs: &Value, info: &Value, config: &OtelTracerConfig) -> SpanData {
        export(|span| set_llm_request_attributes(span, inputs, info, config))
    }

    /// The export of one LLM response span.
    fn response_span(outputs: &Value) -> SpanData {
        export(|span| set_llm_response_attributes(span, outputs))
    }

    /// The typed OTel value of one attribute, when present.
    fn value_of<'a>(data: &'a SpanData, key: &str) -> Option<&'a SpanValue> {
        data.attributes
            .iter()
            .find(|pair| pair.key.as_str() == key)
            .map(|pair| &pair.value)
    }

    /// The string value of one attribute, when present.
    fn attribute(data: &SpanData, key: &str) -> Option<String> {
        value_of(data, key).map(|value| value.as_str().into_owned())
    }

    /// The numeric value of one attribute, whether stored as int or float.
    // Widening an `i64` attribute (token counts) into an `f64` is exact here.
    #[allow(clippy::cast_precision_loss)]
    fn number(data: &SpanData, key: &str) -> Option<f64> {
        match value_of(data, key)? {
            SpanValue::F64(value) => Some(*value),
            SpanValue::I64(value) => Some(*value as f64),
            _ => None,
        }
    }

    /// The integer value of one attribute, when stored as an `i64`.
    fn integer(data: &SpanData, key: &str) -> Option<i64> {
        match value_of(data, key)? {
            SpanValue::I64(value) => Some(*value),
            _ => None,
        }
    }

    /// The string-array value of one attribute, when present.
    fn strings(data: &SpanData, key: &str) -> Option<Vec<String>> {
        match value_of(data, key)? {
            SpanValue::Array(Array::String(values)) => Some(
                values
                    .iter()
                    .map(|value| value.as_str().to_owned())
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Parses a JSON-encoded attribute payload.
    fn json_attribute(data: &SpanData, key: &str) -> Value {
        let text = attribute(data, key).expect("the attribute is written");
        serde_json::from_str(&text).expect("the attribute holds JSON")
    }

    #[test]
    fn request_parameters_keep_their_json_types() {
        // test_handler.py::test_llm_start_sets_request_params_and_message_count
        let info = json!({
            "class_name": "TestModel",
            "request_params": {
                "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 4096,
                "stop_sequences": ["END"], "reasoning_level": "high",
            },
            "message_count": 3,
        });
        let data = request_span(&Value::Null, &info, &plain_config());
        let temperature = value_of(&data, GEN_AI_REQUEST_TEMPERATURE);
        assert!(matches!(temperature, Some(SpanValue::F64(_))));
        assert_eq!(number(&data, GEN_AI_REQUEST_TEMPERATURE), Some(0.7));
        assert_eq!(number(&data, GEN_AI_REQUEST_TOP_P), Some(0.9));
        // An integer knob widens to the float attribute the schema expects.
        assert_eq!(number(&data, GEN_AI_REQUEST_TOP_K), Some(40.0));
        assert_eq!(integer(&data, GEN_AI_REQUEST_MAX_TOKENS), Some(4096));
        assert_eq!(integer(&data, OJ_LLM_PREV_MESSAGE_COUNT), Some(3));
        let stops = strings(&data, GEN_AI_REQUEST_STOP_SEQUENCES);
        assert_eq!(stops, Some(vec!["END".to_owned()]));
        let model = attribute(&data, GEN_AI_REQUEST_MODEL);
        assert_eq!(model.as_deref(), Some("TestModel"));
        let level = attribute(&data, GEN_AI_REQUEST_REASONING_LEVEL);
        assert_eq!(level.as_deref(), Some("high"));
        let trace_name = attribute(&data, OJ_GEN_AI_TRACE_NAME);
        assert_eq!(trace_name.as_deref(), Some("TestModel"));
        let operation = attribute(&data, GEN_AI_OPERATION_NAME);
        assert_eq!(operation.as_deref(), Some(GEN_AI_OPERATION_CHAT));
    }

    #[test]
    fn request_parameters_skip_absent_and_malformed_values() {
        // test_handler.py::test_llm_start_without_params_sets_no_request_attrs
        let info = json!({
            "class_name": "Model",
            "request_params": {
                "temperature": "not-a-number", "max_tokens": ["bad"],
                "stop_sequences": "END", "reasoning_level": 5,
            },
        });
        let data = request_span(&Value::Null, &info, &plain_config());
        for key in [
            GEN_AI_REQUEST_TEMPERATURE,
            GEN_AI_REQUEST_MAX_TOKENS,
            GEN_AI_REQUEST_STOP_SEQUENCES,
            GEN_AI_REQUEST_REASONING_LEVEL,
            OJ_LLM_PREV_MESSAGE_COUNT,
        ] {
            assert!(value_of(&data, key).is_none(), "{key} must be skipped");
        }
        // A non-object instance_info behaves like an empty one.
        let data = request_span(&Value::Null, &json!("nope"), &plain_config());
        assert_eq!(attribute(&data, GEN_AI_REQUEST_MODEL).as_deref(), Some(""));
    }

    #[test]
    fn identity_attributes_follow_the_common_info() {
        // test_handler.py::test_llm_start_sets_agent_identity_attrs
        let info = json!({
            "class_name": "TestModel", "agent_id": "card-1", "agent_name": "HelperAgent",
            "agent_description": "A helper", "conversation_id": "sess-42", "user_id": "u-9",
            "metadata": {"user_id": "u-9", "channel": "web"},
        });
        let data = request_span(&Value::Null, &info, &plain_config());
        for (key, expected) in [
            (GEN_AI_AGENT_ID, "card-1"),
            (GEN_AI_AGENT_NAME, "HelperAgent"),
            (GEN_AI_AGENT_DESCRIPTION, "A helper"),
            (GEN_AI_CONVERSATION_ID, "sess-42"),
            (OJ_GEN_AI_USER_ID, "u-9"),
        ] {
            assert_eq!(attribute(&data, key).as_deref(), Some(expected), "{key}");
        }
        let metadata = json_attribute(&data, OJ_GEN_AI_METADATA);
        assert_eq!(metadata, json!({"user_id": "u-9", "channel": "web"}));
        // The executing agent names the trace, not the span entity.
        let trace_name = attribute(&data, OJ_GEN_AI_TRACE_NAME);
        assert_eq!(trace_name.as_deref(), Some("HelperAgent"));
        // Absent facts (and empty metadata) skip their attribute.
        let info = json!({"class_name": "Model", "metadata": {}});
        let data = request_span(&Value::Null, &info, &plain_config());
        for key in [
            GEN_AI_AGENT_ID,
            GEN_AI_AGENT_NAME,
            GEN_AI_AGENT_DESCRIPTION,
            GEN_AI_CONVERSATION_ID,
            OJ_GEN_AI_USER_ID,
            OJ_GEN_AI_METADATA,
        ] {
            assert!(value_of(&data, key).is_none(), "{key} must be absent");
        }
    }

    #[test]
    fn inputs_split_into_system_instructions_and_user_messages() {
        // test_handler.py::test_agent_llm_inputs_normalized_to_dict
        let inputs = json!({"inputs": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]});
        let data = request_span(&inputs, &json!({}), &plain_config());
        let instructions = json_attribute(&data, GEN_AI_SYSTEM_INSTRUCTIONS);
        assert_eq!(
            instructions,
            json!([{"type": "text", "content": "you are helpful"}])
        );
        let messages = json!([{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]);
        assert_eq!(json_attribute(&data, GEN_AI_INPUT_MESSAGES), messages);
        // A null payload writes neither attribute.
        let data = request_span(&Value::Null, &json!({}), &plain_config());
        assert!(value_of(&data, GEN_AI_INPUT_MESSAGES).is_none());
        assert!(value_of(&data, GEN_AI_SYSTEM_INSTRUCTIONS).is_none());
    }

    #[test]
    fn standard_messages_covers_every_payload_shape() {
        // ::test_agent_llm_inputs_normalized_to_dict and the tool-call path.
        let (instructions, messages) = standard_messages(&json!("hi"), "user");
        assert_eq!(instructions.len(), 0);
        let expected = vec![json!({"role": "user", "parts": [{"type": "text", "content": "hi"}]})];
        assert_eq!(messages, expected);
        // An array of scalars: each item becomes a content-only message.
        let (_, messages) = standard_messages(&json!(["a", "b"]), "assistant");
        let expected = vec![
            json!({"role": "assistant", "parts": [{"type": "text", "content": "a"}]}),
            json!({"role": "assistant", "parts": [{"type": "text", "content": "b"}]}),
        ];
        assert_eq!(messages, expected);
        // `messages` wins over `inputs`, a blank role takes the default, and a
        // `response`/`output` key stands in for content.
        let payload = json!({"messages": [{"role": "", "response": "x"}], "inputs": ["no"]});
        let (_, messages) = standard_messages(&payload, "user");
        let expected = vec![json!({"role": "user", "parts": [{"type": "text", "content": "x"}]})];
        assert_eq!(messages, expected);
        // A message with no content keeps an empty parts list; null yields nothing.
        let (_, messages) = standard_messages(&json!({"role": "user"}), "user");
        assert_eq!(messages, vec![json!({"role": "user", "parts": []})]);
        let (instructions, messages) = standard_messages(&Value::Null, "user");
        assert_eq!((instructions.len(), messages.len()), (0, 0));
        // A tool call copies its own keys, falls back to `function`, skips a
        // null/absent field, and ignores non-object calls.
        let inputs = json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call-1", "function": {"name": "weather", "arguments": "{\"city\":\"北京\"}"}},
                {"id": null, "name": "echo", "arguments": {"raw": true}},
                "not-an-object",
            ],
        }]);
        let (instructions, messages) = standard_messages(&inputs, "user");
        assert_eq!(instructions.len(), 0);
        assert_eq!(
            messages,
            vec![json!({
                "role": "assistant",
                "parts": [
                    {"type": "tool_call", "id": "call-1", "name": "weather",
                     "arguments": "{\"city\":\"北京\"}"},
                    {"type": "tool_call", "name": "echo", "arguments": {"raw": true}},
                ],
            })]
        );
    }

    #[test]
    fn prompt_redaction_hashes_content_and_honours_the_field_override() {
        // test_handler.py::test_agent_llm_prompt_completion_split_redaction
        let config = OtelTracerConfig {
            redaction_enabled: true,
            ..OtelTracerConfig::default()
        };
        let data = request_span(&json!("secret data"), &json!({}), &config);
        let messages = json_attribute(&data, GEN_AI_INPUT_MESSAGES);
        let content = messages[0]["parts"][0]["content"].as_str().unwrap();
        assert!(content.starts_with("sha256:"), "unexpected {content}");
        // `redacted_json` is the same walk; hashing replaces the raw text.
        let redactor = Redactor::new(&config);
        let hashed = redacted_json(&messages, &redactor, RedactionField::Prompts, false);
        assert!(hashed.contains("sha256:") && !hashed.contains("secret data"));
        // `redact_prompts = false` overrides `redaction_enabled = true`.
        let config = OtelTracerConfig {
            redaction_enabled: true,
            redact_prompts: Some(false),
            ..OtelTracerConfig::default()
        };
        let data = request_span(&json!("visible prompt"), &json!({}), &config);
        let messages = json_attribute(&data, GEN_AI_INPUT_MESSAGES);
        assert_eq!(messages[0]["parts"][0]["content"], json!("visible prompt"));
    }

    #[test]
    fn system_instructions_bypass_the_attribute_length_cap() {
        // test_handler.py::test_agent_llm_system_instructions_bypass_attribute_length_cap
        let config = OtelTracerConfig {
            redaction_enabled: false,
            max_attr_length: 8,
            ..OtelTracerConfig::default()
        };
        let system_prompt = "stable-system-prompt-".repeat(20);
        let inputs = json!({"inputs": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "long-user-message"},
        ]});
        let data = request_span(&inputs, &json!({}), &config);
        let instructions = json_attribute(&data, GEN_AI_SYSTEM_INSTRUCTIONS);
        assert_eq!(instructions[0]["content"], json!(system_prompt));
        let messages = json_attribute(&data, GEN_AI_INPUT_MESSAGES);
        let truncated = messages[0]["parts"][0]["content"].as_str().unwrap();
        assert!(
            truncated.contains("OTel attribute truncated"),
            "{truncated}"
        );
    }

    #[test]
    fn response_finish_reason_and_container_unwrapping() {
        // test_handler.py::test_llm_end_unwraps_outputs_container_into_content
        for container in ["outputs", "messages", "inputs"] {
            let payload = json!({
                container: {"finish_reason": "stop", "usage_metadata": {"input_tokens": 4}},
            });
            let data = response_span(&payload);
            let reasons = strings(&data, GEN_AI_RESPONSE_FINISH_REASONS);
            assert_eq!(reasons, Some(vec!["stop".to_owned()]), "{container}");
            assert_eq!(integer(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(4));
        }
        // The schema default "null", a null, a `false`, and an absent value all
        // write nothing; the usage block is unaffected.
        for reason in [
            json!({"finish_reason": "null", "usage_metadata": {"input_tokens": 1}}),
            json!({"finish_reason": null, "usage_metadata": {"input_tokens": 1}}),
            json!({"finish_reason": false, "usage_metadata": {"input_tokens": 1}}),
            json!({"usage_metadata": {"input_tokens": 1}}),
        ] {
            let data = response_span(&reason);
            let key = GEN_AI_RESPONSE_FINISH_REASONS;
            assert!(value_of(&data, key).is_none(), "{reason}");
            assert_eq!(integer(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(1));
        }
        // A scalar payload, a container holding one, a non-object, or a missing
        // `usage_metadata` object writes no usage attributes at all.
        for payload in [
            json!("plain"),
            json!({"outputs": "plain"}),
            json!([1]),
            json!({}),
        ] {
            let data = response_span(&payload);
            let model = value_of(&data, GEN_AI_RESPONSE_MODEL);
            assert!(model.is_none(), "{payload}");
            assert!(value_of(&data, GEN_AI_USAGE_INPUT_TOKENS).is_none());
        }
    }

    #[test]
    fn response_usage_cost_and_cache_attributes_are_recorded() {
        // test_handler.py::test_llm_end_extracts_response_usage_and_cost_attrs
        let payload = json!({
            "finish_reason": "stop",
            "usage_metadata": {
                "model_name": "claude-test", "input_tokens": 11, "output_tokens": 22,
                "cache_read_tokens": 5, "cache_creation_input_tokens": 7,
                "reasoning_tokens": 3, "input_cost": 0.001, "output_cost": 0.002,
                "total_cost": 0.003,
            },
        });
        let data = response_span(&payload);
        let model = attribute(&data, GEN_AI_RESPONSE_MODEL);
        assert_eq!(model.as_deref(), Some("claude-test"));
        for (key, expected) in [
            (GEN_AI_USAGE_INPUT_TOKENS, 11),
            (GEN_AI_USAGE_OUTPUT_TOKENS, 22),
            (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, 5),
            (GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, 7),
            (GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, 3),
        ] {
            assert_eq!(integer(&data, key), Some(expected), "{key}");
        }
        for (key, expected) in [
            (OJ_GEN_AI_USAGE_INPUT_COST, 0.001),
            (OJ_GEN_AI_USAGE_OUTPUT_COST, 0.002),
            (OJ_GEN_AI_USAGE_TOTAL_COST, 0.003),
        ] {
            assert_eq!(number(&data, key), Some(expected), "{key}");
        }
        // The legacy cache keys stand in when the explicit splits are absent.
        let payload = json!({"usage_metadata": {
            "input_tokens": 1, "cache_tokens": 9, "cache_creation_input_tokens": 6,
        }});
        let data = response_span(&payload);
        assert_eq!(
            integer(&data, GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS),
            Some(9)
        );
        assert_eq!(
            integer(&data, GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS),
            Some(6)
        );
    }

    #[test]
    fn response_usage_zeroes_are_recorded_as_facts() {
        // test_handler.py::test_llm_end_sets_zero_usage_values
        let data = response_span(&json!({"usage_metadata": {}}));
        assert_eq!(attribute(&data, GEN_AI_RESPONSE_MODEL).as_deref(), Some(""));
        for key in [
            GEN_AI_USAGE_INPUT_TOKENS,
            GEN_AI_USAGE_OUTPUT_TOKENS,
            GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
            GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        ] {
            assert_eq!(integer(&data, key), Some(0), "{key}");
        }
        for key in [
            OJ_GEN_AI_USAGE_TOTAL_COST,
            OJ_GEN_AI_USAGE_INPUT_COST,
            OJ_GEN_AI_USAGE_OUTPUT_COST,
        ] {
            assert_eq!(number(&data, key), Some(0.0), "{key}");
        }
        // Latencies have no default: absent stays absent.
        for key in [
            GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
            OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS,
            OJ_GEN_AI_REASONING_DURATION_MS,
        ] {
            assert!(value_of(&data, key).is_none(), "{key}");
        }
    }

    #[test]
    fn time_to_first_chunk_table() {
        // test_handler.py::test_llm_end_reads_ttft_from_the_explicit_ttft_seconds_field,
        // ::test_llm_end_prefers_ttft_seconds_over_legacy_first_token_time, and
        // ::test_llm_end_malformed_ttft_seconds_skips_attr_not_finalization
        for (usage, expected) in [
            (json!({"ttft_seconds": 0.42}), Some(0.42)),
            (json!({"ttft_seconds": 0.0}), Some(0.0)),
            (
                json!({"ttft_seconds": 0.35, "first_token_time": "9.99"}),
                Some(0.35),
            ),
            // The legacy fallback applies only when the explicit key is absent.
            (json!({"first_token_time": "0.42"}), Some(0.42)),
            (json!({"first_token_time": ""}), None),
            (
                json!({"ttft_seconds": null, "first_token_time": "0.5"}),
                Some(0.5),
            ),
            // A malformed explicit value skips instead of falling back.
            (
                json!({"ttft_seconds": "bad", "first_token_time": "9.99"}),
                None,
            ),
            (json!({"ttft_seconds": ["bad"]}), None),
        ] {
            let data = response_span(&json!({"usage_metadata": usage}));
            let ttfc = number(&data, GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK);
            assert_eq!(ttfc, expected, "{usage} is wrong");
        }
    }

    #[test]
    fn latency_attributes_are_written_only_when_parseable() {
        // test_handler.py::test_llm_end_sets_streaming_and_reasoning_latency and
        // ::test_llm_end_malformed_latency_skips_attrs_not_finalization
        let payload = json!({"usage_metadata": {
            "inter_token_latency_ms": 12.5, "reasoning_duration_ms": "300.0",
        }});
        let data = response_span(&payload);
        let inter = number(&data, OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS);
        assert_eq!(inter, Some(12.5));
        assert_eq!(number(&data, OJ_GEN_AI_REASONING_DURATION_MS), Some(300.0));
        let malformed = json!({"usage_metadata": {
            "input_tokens": 1, "inter_token_latency_ms": "bad",
            "reasoning_duration_ms": ["bad"],
        }});
        let data = response_span(&malformed);
        assert!(value_of(&data, OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS).is_none());
        assert!(value_of(&data, OJ_GEN_AI_REASONING_DURATION_MS).is_none());
        // Finalization survived: the other usage facts are still written.
        assert_eq!(integer(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(1));
    }

    #[test]
    fn retrieval_query_and_documents_extraction_table() {
        // test_handler.py::test_workflow_retrieval_component_tagged_as_retrieval
        let query = extract_retrieval_query(&json!("什么是向量检索"));
        assert_eq!(query, Some("什么是向量检索".to_owned()));
        let query = extract_retrieval_query(&json!({"query": "hello"}));
        assert_eq!(query, Some("hello".to_owned()));
        for absent in [
            json!(""),
            json!({"query": ""}),
            json!({"query": 7}),
            json!([]),
        ] {
            assert_eq!(extract_retrieval_query(&absent), None, "{absent}");
        }
        assert_eq!(extract_retrieval_query(&Value::Null), None);
        let documents = extract_retrieval_documents(&json!({"results": ["doc-1 内容", "x"]}));
        let expected = vec!["doc-1 内容".to_owned(), "x".to_owned()];
        assert_eq!(documents, Some(expected));
        let documents = extract_retrieval_documents(&json!({"results": [1, {"id": 2}]}));
        assert_eq!(
            documents,
            Some(vec!["1".to_owned(), r#"{"id":2}"#.to_owned()])
        );
        for absent in [json!({"results": []}), json!({"results": "doc"}), json!({})] {
            assert_eq!(extract_retrieval_documents(&absent), None, "{absent}");
        }
        assert_eq!(extract_retrieval_documents(&Value::Null), None);
    }
}
