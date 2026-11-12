//! LLM payload normalization and `GenAI` request/response attributes.
//!
//! Port of the `GenAI` attribute helpers of
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
// The attribute writers inline these pairs; the table keeps Python's field order.
#[allow(dead_code)]
const FLOAT_REQUEST_PARAMETERS: [(&str, &str); 3] = [
    ("temperature", GEN_AI_REQUEST_TEMPERATURE),
    ("top_p", GEN_AI_REQUEST_TOP_P),
    ("top_k", GEN_AI_REQUEST_TOP_K),
];

/// Usage counters written as `i64` attributes, 0 when the payload omits them.
// The attribute writers inline these pairs; the table keeps Python's field order.
#[allow(dead_code)]
const INTEGER_USAGE_FIELDS: [(&str, &str); 3] = [
    ("input_tokens", GEN_AI_USAGE_INPUT_TOKENS),
    ("output_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
    ("reasoning_tokens", GEN_AI_USAGE_REASONING_OUTPUT_TOKENS),
];

/// Usage costs written as `f64` attributes, 0.0 when the payload omits them.
// The attribute writers inline these pairs; the table keeps Python's field order.
#[allow(dead_code)]
const COST_USAGE_FIELDS: [(&str, &str); 3] = [
    ("total_cost", OJ_GEN_AI_USAGE_TOTAL_COST),
    ("input_cost", OJ_GEN_AI_USAGE_INPUT_COST),
    ("output_cost", OJ_GEN_AI_USAGE_OUTPUT_COST),
];

/// Streaming/reasoning latencies written only when present and parseable.
// The attribute writers inline these pairs; the table keeps Python's field order.
#[allow(dead_code)]
const LATENCY_USAGE_FIELDS: [(&str, &str); 2] = [
    (
        "inter_token_latency_ms",
        OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS,
    ),
    ("reasoning_duration_ms", OJ_GEN_AI_REASONING_DURATION_MS),
];

/// Normalizes an LLM input payload into `GenAI` `instructions` and `messages`.
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

/// Converts a JSON scalar into the typed `OTel` value of the same shape.
///
/// Objects, arrays, and `null` have no `OTel` scalar equivalent, so they yield
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

/// Writes the `GenAI` request attributes of an LLM span.
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

/// Writes the `GenAI` request parameters forwarded by the rail.
///
/// Python counterpart: the `request_params` loop of `on_llm_start`. Numbers
/// keep their numeric type (`f64` for sampling knobs, `i64` for the token
/// budget), `stop_sequences` becomes an `OTel` string array, and
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

/// Writes the `GenAI` system-instruction and input-message payloads.
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

/// Writes the `GenAI` response / usage / cost / latency attributes of an LLM span.
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
mod tests;
