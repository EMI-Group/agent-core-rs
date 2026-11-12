//! Semantic-convention attribute keys and values (verbatim wire names).
//!
//! Port of `openjiuwen/extensions/tracer_otel/semconv.py` (v0.1.20), which
//! defines the project-specific `openjiuwen.*` keys and re-exports the standard
//! `gen_ai.*` keys from `openjiuwen/extensions/observability/semconv.py`.
//! Keeping every key here avoids typo drift between the agent handler, the
//! workflow handler, and the span manager; the values are wire format and must
//! never change silently (see `AGENTS.md` in the Python reference).
//!
//! Layout mirrors the Python module:
//!
//! - standard `gen_ai.*` keys (plus [`ERROR_TYPE`]) — re-exported upstream names,
//! - the `openjiuwen.*` request/response extensions the pinned `GenAI` registry
//!   does not model (issue #1833: project extensions must not ride the
//!   `gen_ai.*` prefix, hence the `openjiuwen.gen_ai.*` namespace),
//! - workflow-level, agent-level, trace-bridge, base-span, and
//!   workflow-span attributes,
//! - attribute values: provider name, operation names, and span statuses.
//!
//! Everything is a `const &'static str`, so this module compiles with and
//! without the `otel` feature and pulls in no dependencies.

// ---------------------------------------------------------------------------
// gen_ai.* — standard GenAI keys
// Re-exported from `openjiuwen/extensions/observability/semconv.py`, which in
// turn re-exports the generated `gen_ai_semconv` registry
// (revision fee465db333bdd6a7d2faa320edab5cf3101a4f4). One authoritative
// definition per standard name, shared by every instrumentation path.
// ---------------------------------------------------------------------------

/// Wire name `gen_ai.agent.description` — free-form agent description.
pub const GEN_AI_AGENT_DESCRIPTION: &str = "gen_ai.agent.description";

/// Wire name `gen_ai.agent.id` — stable agent identifier.
pub const GEN_AI_AGENT_ID: &str = "gen_ai.agent.id";

/// Wire name `gen_ai.agent.name` — human-readable agent name (agent spans).
pub const GEN_AI_AGENT_NAME: &str = "gen_ai.agent.name";

/// Wire name `gen_ai.conversation.id` — session/conversation identity.
///
/// The single carrier for session identity: no `openjiuwen.*` mirror is written.
pub const GEN_AI_CONVERSATION_ID: &str = "gen_ai.conversation.id";

/// Wire name `gen_ai.data_source.id` — data source a retrieval span queried.
pub const GEN_AI_DATA_SOURCE_ID: &str = "gen_ai.data_source.id";

/// Wire name `gen_ai.embeddings.dimension.count` — embedding vector width.
pub const GEN_AI_EMBEDDINGS_DIMENSION_COUNT: &str = "gen_ai.embeddings.dimension.count";

/// Wire name `gen_ai.input.messages` — JSON-encoded request messages (LLM spans).
pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";

/// Wire name `gen_ai.memory.record.count` — records touched by a memory span.
pub const GEN_AI_MEMORY_RECORD_COUNT: &str = "gen_ai.memory.record.count";

/// Wire name `gen_ai.operation.name` — operation discriminator of a span
/// (`chat`, `execute_tool`, `embeddings`, `retrieval`; see the
/// [`GEN_AI_OPERATION_CHAT`] family).
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";

/// Wire name `gen_ai.output.messages` — JSON-encoded completion messages.
pub const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";

/// Wire name `gen_ai.provider.name` — model provider; [`GEN_AI_PROVIDER_OPENJIUWEN`]
/// on this stack.
pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";

/// Wire name `gen_ai.request.max_tokens` — requested output-token budget.
pub const GEN_AI_REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";

/// Wire name `gen_ai.request.model` — requested model name.
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";

/// Wire name `gen_ai.request.reasoning.level` — requested reasoning effort.
pub const GEN_AI_REQUEST_REASONING_LEVEL: &str = "gen_ai.request.reasoning.level";

/// Wire name `gen_ai.request.stop_sequences` — requested stop sequences.
pub const GEN_AI_REQUEST_STOP_SEQUENCES: &str = "gen_ai.request.stop_sequences";

/// Wire name `gen_ai.request.temperature` — requested sampling temperature.
pub const GEN_AI_REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";

/// Wire name `gen_ai.request.top_k` — requested top-k sampling parameter.
pub const GEN_AI_REQUEST_TOP_K: &str = "gen_ai.request.top_k";

/// Wire name `gen_ai.request.top_p` — requested nucleus sampling parameter.
pub const GEN_AI_REQUEST_TOP_P: &str = "gen_ai.request.top_p";

/// Wire name `gen_ai.response.finish_reasons` — provider finish reasons.
pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";

/// Wire name `gen_ai.response.model` — model that actually served the request.
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";

/// Wire name `gen_ai.response.time_to_first_chunk` — first-token latency in
/// **seconds** (the standard's unit; the `openjiuwen.gen_ai.*_ms` durations
/// below are deliberately in milliseconds).
pub const GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: &str = "gen_ai.response.time_to_first_chunk";

/// Wire name `gen_ai.retrieval.documents` — JSON-encoded retrieved documents.
pub const GEN_AI_RETRIEVAL_DOCUMENTS: &str = "gen_ai.retrieval.documents";

/// Wire name `gen_ai.retrieval.query.text` — retrieval query text.
pub const GEN_AI_RETRIEVAL_QUERY_TEXT: &str = "gen_ai.retrieval.query.text";

/// Wire name `gen_ai.retrieval.top_k` — retrieval top-k.
pub const GEN_AI_RETRIEVAL_TOP_K: &str = "gen_ai.retrieval.top_k";

/// Wire name `gen_ai.system_instructions` — JSON-encoded system prompt.
pub const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";

/// Wire name `gen_ai.tool.call.id` — provider-side tool call identifier.
pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";

/// Wire name `gen_ai.tool.name` — tool name (tool spans; tool-span dedup).
pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";

/// Wire name `gen_ai.tool.type` — tool kind (`function`, …).
pub const GEN_AI_TOOL_TYPE: &str = "gen_ai.tool.type";

/// Wire name `gen_ai.usage.cache_read.input_tokens` — prompt-cache read tokens.
pub const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

/// Wire name `gen_ai.usage.cache_write.input_tokens` — prompt-cache write tokens.
pub const GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: &str = "gen_ai.usage.cache_write.input_tokens";

/// Wire name `gen_ai.usage.input_tokens` — prompt tokens.
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";

/// Wire name `gen_ai.usage.output_tokens` — completion tokens.
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

/// Wire name `gen_ai.usage.reasoning.output_tokens` — reasoning tokens.
pub const GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: &str = "gen_ai.usage.reasoning.output_tokens";

/// Wire name `error.type` — error discriminator written on failed spans.
///
/// Re-exported by the Python tracer module from
/// `extensions/observability/semconv.py`; it is the one standard key that does
/// not live under the `gen_ai.*` namespace.
pub const ERROR_TYPE: &str = "error.type";

// ---------------------------------------------------------------------------
// Attribute values
// ---------------------------------------------------------------------------

/// Value of [`GEN_AI_PROVIDER_NAME`] on this stack (`GEN_AI_PROVIDER_NAME_VALUE`
/// in Python). The non-standard `gen_ai.system` key is deliberately not used.
pub const GEN_AI_PROVIDER_OPENJIUWEN: &str = "openjiuwen";

/// Value of [`GEN_AI_OPERATION_NAME`] for LLM spans (`chat {class_name}` /
/// `chat {model}`).
pub const GEN_AI_OPERATION_CHAT: &str = "chat";

/// Value of [`GEN_AI_OPERATION_NAME`] for tool spans (`execute_tool {name}`).
pub const GEN_AI_OPERATION_EXECUTE_TOOL: &str = "execute_tool";

/// Value of [`GEN_AI_OPERATION_NAME`] for embedding spans (`embeddings {model}`).
pub const GEN_AI_OPERATION_EMBEDDINGS: &str = "embeddings";

/// Value of [`GEN_AI_OPERATION_NAME`] for retrieval spans (`retrieval {name}`).
pub const GEN_AI_OPERATION_RETRIEVAL: &str = "retrieval";

/// Wire name `start` — [`SPAN_STATUS_START`]; the value of
/// [`OJ_STATUS`] while a span is open, mirroring
/// `openjiuwen_session::SpanStatus::Start`.
pub const SPAN_STATUS_START: &str = "start";

/// Wire name `running` — value of [`OJ_STATUS`], mirroring
/// `openjiuwen_session::SpanStatus::Running`.
pub const SPAN_STATUS_RUNNING: &str = "running";

/// Wire name `finish` — the default value of [`OJ_STATUS`] when a span ends
/// successfully, mirroring `openjiuwen_session::SpanStatus::Finish`.
pub const SPAN_STATUS_FINISH: &str = "finish";

/// Wire name `interrupted` — value of [`OJ_STATUS`] for a span interrupted by
/// human-in-the-loop or cancellation, mirroring
/// `openjiuwen_session::SpanStatus::Interrupted`.
pub const SPAN_STATUS_INTERRUPTED: &str = "interrupted";

/// Wire name `error` — value of [`OJ_STATUS`] for a failed span, mirroring
/// `openjiuwen_session::SpanStatus::Error`.
pub const SPAN_STATUS_ERROR: &str = "error";

// ---------------------------------------------------------------------------
// openjiuwen.* — request/response facts the pinned GenAI registry does not
// model (project extensions owned by this package; wire format. See AGENTS.md
// in the Python reference before renaming them.)
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.llm.prev_message_count` — conversation turns replayed
/// into the request.
pub const OJ_LLM_PREV_MESSAGE_COUNT: &str = "openjiuwen.llm.prev_message_count";

/// Wire name `openjiuwen.gen_ai.usage.total_cost` — total request cost.
pub const OJ_GEN_AI_USAGE_TOTAL_COST: &str = "openjiuwen.gen_ai.usage.total_cost";

/// Wire name `openjiuwen.gen_ai.usage.input_cost` — prompt cost.
pub const OJ_GEN_AI_USAGE_INPUT_COST: &str = "openjiuwen.gen_ai.usage.input_cost";

/// Wire name `openjiuwen.gen_ai.usage.output_cost` — completion cost.
pub const OJ_GEN_AI_USAGE_OUTPUT_COST: &str = "openjiuwen.gen_ai.usage.output_cost";

/// Wire name `openjiuwen.gen_ai.user.id` — session-carried user identity.
///
/// Conversation fact the registry does not model; follows the project
/// extension naming regime (`openjiuwen.*`, never `gen_ai.*`).
pub const OJ_GEN_AI_USER_ID: &str = "openjiuwen.gen_ai.user.id";

/// Wire name `openjiuwen.gen_ai.metadata` — raw source metadata as JSON.
pub const OJ_GEN_AI_METADATA: &str = "openjiuwen.gen_ai.metadata";

/// Wire name `openjiuwen.trace.name` — trace display name.
pub const OJ_GEN_AI_TRACE_NAME: &str = "openjiuwen.trace.name";

/// Wire name `openjiuwen.gen_ai.response.inter_token_latency_ms` — mean
/// inter-token latency in milliseconds.
pub const OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS: &str =
    "openjiuwen.gen_ai.response.inter_token_latency_ms";

/// Wire name `openjiuwen.gen_ai.reasoning.duration_ms` — reasoning duration in
/// milliseconds.
pub const OJ_GEN_AI_REASONING_DURATION_MS: &str = "openjiuwen.gen_ai.reasoning.duration_ms";

// ---------------------------------------------------------------------------
// openjiuwen.workflow.* — workflow-level custom attributes
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.workflow.id` — workflow identifier.
pub const OJ_WORKFLOW_ID: &str = "openjiuwen.workflow.id";

/// Wire name `openjiuwen.workflow.name` — workflow name.
pub const OJ_WORKFLOW_NAME: &str = "openjiuwen.workflow.name";

/// Wire name `openjiuwen.workflow.version` — workflow version.
pub const OJ_WORKFLOW_VERSION: &str = "openjiuwen.workflow.version";

/// Wire name `openjiuwen.workflow.component.id` — component identifier.
pub const OJ_WORKFLOW_COMPONENT_ID: &str = "openjiuwen.workflow.component.id";

/// Wire name `openjiuwen.workflow.component.type` — component type; drives span
/// naming (`is_llm_component` / `is_tool_component` / `is_retrieval_component`).
pub const OJ_WORKFLOW_COMPONENT_TYPE: &str = "openjiuwen.workflow.component.type";

/// Wire name `openjiuwen.workflow.component.name` — component name.
pub const OJ_WORKFLOW_COMPONENT_NAME: &str = "openjiuwen.workflow.component.name";

/// Wire name `openjiuwen.workflow.execution_id` — workflow execution identifier.
pub const OJ_WORKFLOW_EXECUTION_ID: &str = "openjiuwen.workflow.execution_id";

/// Wire name `openjiuwen.workflow.loop.node_id` — loop node identifier.
pub const OJ_WORKFLOW_LOOP_NODE_ID: &str = "openjiuwen.workflow.loop.node_id";

/// Wire name `openjiuwen.workflow.loop.index` — iteration index inside a loop.
pub const OJ_WORKFLOW_LOOP_INDEX: &str = "openjiuwen.workflow.loop.index";

// ---------------------------------------------------------------------------
// openjiuwen.agent.* — agent-level custom attributes (non-LLM agent types)
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.agent.invoke_type` — agent invoke type
/// (`InvokeType::as_str`).
pub const OJ_AGENT_INVOKE_TYPE: &str = "openjiuwen.agent.invoke_type";

/// Wire name `openjiuwen.agent.name` — agent name.
pub const OJ_AGENT_NAME: &str = "openjiuwen.agent.name";

/// Wire name `openjiuwen.agent.inputs` — JSON-encoded agent inputs.
pub const OJ_AGENT_INPUTS: &str = "openjiuwen.agent.inputs";

/// Wire name `openjiuwen.agent.outputs` — JSON-encoded agent outputs.
pub const OJ_AGENT_OUTPUTS: &str = "openjiuwen.agent.outputs";

/// Wire name `openjiuwen.agent.error_message` — agent failure message.
pub const OJ_AGENT_ERROR_MESSAGE: &str = "openjiuwen.agent.error_message";

// ---------------------------------------------------------------------------
// Trace ID bridge — links the OTel trace to the tracer UUID
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.trace.id` — tracer trace UUID; the bridge between the
/// `OTel` trace and the session tracer, and part of the tool-span dedup key.
pub const OJ_TRACE_ID: &str = "openjiuwen.trace.id";

/// Wire name `openjiuwen.session_id` — session identifier (written only when
/// non-empty).
pub const OJ_SESSION_ID: &str = "openjiuwen.session_id";

// ---------------------------------------------------------------------------
// openjiuwen.* — base span attributes (shared by both handlers)
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.invoke_id` — invoke identifier, also the span name of
/// a workflow root span.
pub const OJ_INVOKE_ID: &str = "openjiuwen.invoke_id";

/// Wire name `openjiuwen.parent_invoke_id` — parent invoke identifier; also the
/// key used to resolve the parent span context.
pub const OJ_PARENT_INVOKE_ID: &str = "openjiuwen.parent_invoke_id";

/// Wire name `openjiuwen.start_time` — span start, formatted
/// `%Y-%m-%d %H:%M:%S%.6f` in local time.
pub const OJ_START_TIME: &str = "openjiuwen.start_time";

/// Wire name `openjiuwen.end_time` — span end, same format as [`OJ_START_TIME`].
pub const OJ_END_TIME: &str = "openjiuwen.end_time";

/// Wire name `openjiuwen.elapsed_time` — duration, `"{ms}ms"` below one second
/// and `"{s:.2}s"` otherwise.
pub const OJ_ELAPSED_TIME: &str = "openjiuwen.elapsed_time";

/// Wire name `openjiuwen.status` — span status; one of the `SPAN_STATUS_*`
/// values or a session `SpanStatus` wire name.
pub const OJ_STATUS: &str = "openjiuwen.status";

/// Wire name `openjiuwen.error` — JSON `{error_code, message}` on failed spans.
pub const OJ_ERROR: &str = "openjiuwen.error";

/// Wire name `openjiuwen.child_invoke_ids` — JSON array of child invoke ids,
/// written when known.
pub const OJ_CHILD_INVOKE_IDS: &str = "openjiuwen.child_invoke_ids";

/// Wire name `openjiuwen.meta_data` — JSON-encoded trace metadata.
pub const OJ_META_DATA: &str = "openjiuwen.meta_data";

// ---------------------------------------------------------------------------
// openjiuwen.* — workflow-specific base attributes
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.parent_node_id` — parent node in the workflow graph.
pub const OJ_PARENT_NODE_ID: &str = "openjiuwen.parent_node_id";

/// Wire name `openjiuwen.source_ids` — JSON array of upstream node ids.
pub const OJ_SOURCE_IDS: &str = "openjiuwen.source_ids";

/// Wire name `openjiuwen.inner_error` — error raised by a nested component.
pub const OJ_INNER_ERROR: &str = "openjiuwen.inner_error";

/// Wire name `openjiuwen.stream_inputs` — buffered JSON stream inputs, flushed
/// when non-empty.
pub const OJ_STREAM_INPUTS: &str = "openjiuwen.stream_inputs";

/// Wire name `openjiuwen.stream_outputs` — buffered JSON stream outputs,
/// flushed when non-empty.
pub const OJ_STREAM_OUTPUTS: &str = "openjiuwen.stream_outputs";

/// Wire name `openjiuwen.interactive_inputs` — JSON-encoded human-in-the-loop
/// inputs.
pub const OJ_INTERACTIVE_INPUTS: &str = "openjiuwen.interactive_inputs";

/// Wire name `openjiuwen.workflow.inputs` — JSON-encoded workflow inputs.
pub const OJ_WORKFLOW_INPUTS: &str = "openjiuwen.workflow.inputs";

/// Wire name `openjiuwen.workflow.outputs` — JSON-encoded workflow outputs.
pub const OJ_WORKFLOW_OUTPUTS: &str = "openjiuwen.workflow.outputs";

/// Wire name `openjiuwen.workflow.error_message` — workflow failure message,
/// also set when a workflow span is interrupted.
pub const OJ_WORKFLOW_ERROR_MESSAGE: &str = "openjiuwen.workflow.error_message";

/// Wire name `openjiuwen.workflow.invoke_data` — buffered per-invoke JSON data,
/// flushed on `on_call_done` / `on_invoke`.
pub const OJ_WORKFLOW_INVOKE_DATA: &str = "openjiuwen.workflow.invoke_data";

#[cfg(test)]
mod tests {
    use super::*;

    /// Every standard `gen_ai.*` key constant with its constant name, so the
    /// test below can check the whole inventory rather than a sample.
    const GEN_AI_KEYS: &[(&str, &str)] = &[
        ("GEN_AI_AGENT_DESCRIPTION", GEN_AI_AGENT_DESCRIPTION),
        ("GEN_AI_AGENT_ID", GEN_AI_AGENT_ID),
        ("GEN_AI_AGENT_NAME", GEN_AI_AGENT_NAME),
        ("GEN_AI_CONVERSATION_ID", GEN_AI_CONVERSATION_ID),
        ("GEN_AI_DATA_SOURCE_ID", GEN_AI_DATA_SOURCE_ID),
        (
            "GEN_AI_EMBEDDINGS_DIMENSION_COUNT",
            GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
        ),
        ("GEN_AI_INPUT_MESSAGES", GEN_AI_INPUT_MESSAGES),
        ("GEN_AI_MEMORY_RECORD_COUNT", GEN_AI_MEMORY_RECORD_COUNT),
        ("GEN_AI_OPERATION_NAME", GEN_AI_OPERATION_NAME),
        ("GEN_AI_OUTPUT_MESSAGES", GEN_AI_OUTPUT_MESSAGES),
        ("GEN_AI_PROVIDER_NAME", GEN_AI_PROVIDER_NAME),
        ("GEN_AI_REQUEST_MAX_TOKENS", GEN_AI_REQUEST_MAX_TOKENS),
        ("GEN_AI_REQUEST_MODEL", GEN_AI_REQUEST_MODEL),
        (
            "GEN_AI_REQUEST_REASONING_LEVEL",
            GEN_AI_REQUEST_REASONING_LEVEL,
        ),
        (
            "GEN_AI_REQUEST_STOP_SEQUENCES",
            GEN_AI_REQUEST_STOP_SEQUENCES,
        ),
        ("GEN_AI_REQUEST_TEMPERATURE", GEN_AI_REQUEST_TEMPERATURE),
        ("GEN_AI_REQUEST_TOP_K", GEN_AI_REQUEST_TOP_K),
        ("GEN_AI_REQUEST_TOP_P", GEN_AI_REQUEST_TOP_P),
        (
            "GEN_AI_RESPONSE_FINISH_REASONS",
            GEN_AI_RESPONSE_FINISH_REASONS,
        ),
        ("GEN_AI_RESPONSE_MODEL", GEN_AI_RESPONSE_MODEL),
        (
            "GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK",
            GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
        ),
        ("GEN_AI_RETRIEVAL_DOCUMENTS", GEN_AI_RETRIEVAL_DOCUMENTS),
        ("GEN_AI_RETRIEVAL_QUERY_TEXT", GEN_AI_RETRIEVAL_QUERY_TEXT),
        ("GEN_AI_RETRIEVAL_TOP_K", GEN_AI_RETRIEVAL_TOP_K),
        ("GEN_AI_SYSTEM_INSTRUCTIONS", GEN_AI_SYSTEM_INSTRUCTIONS),
        ("GEN_AI_TOOL_CALL_ID", GEN_AI_TOOL_CALL_ID),
        ("GEN_AI_TOOL_NAME", GEN_AI_TOOL_NAME),
        ("GEN_AI_TOOL_TYPE", GEN_AI_TOOL_TYPE),
        (
            "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
            GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        ),
        (
            "GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS",
            GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        ),
        ("GEN_AI_USAGE_INPUT_TOKENS", GEN_AI_USAGE_INPUT_TOKENS),
        ("GEN_AI_USAGE_OUTPUT_TOKENS", GEN_AI_USAGE_OUTPUT_TOKENS),
        (
            "GEN_AI_USAGE_REASONING_OUTPUT_TOKENS",
            GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        ),
    ];

    /// Every `openjiuwen.*` project extension key constant.
    const OJ_KEYS: &[(&str, &str)] = &[
        ("OJ_LLM_PREV_MESSAGE_COUNT", OJ_LLM_PREV_MESSAGE_COUNT),
        ("OJ_GEN_AI_USAGE_TOTAL_COST", OJ_GEN_AI_USAGE_TOTAL_COST),
        ("OJ_GEN_AI_USAGE_INPUT_COST", OJ_GEN_AI_USAGE_INPUT_COST),
        ("OJ_GEN_AI_USAGE_OUTPUT_COST", OJ_GEN_AI_USAGE_OUTPUT_COST),
        ("OJ_GEN_AI_USER_ID", OJ_GEN_AI_USER_ID),
        ("OJ_GEN_AI_METADATA", OJ_GEN_AI_METADATA),
        ("OJ_GEN_AI_TRACE_NAME", OJ_GEN_AI_TRACE_NAME),
        (
            "OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS",
            OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS,
        ),
        (
            "OJ_GEN_AI_REASONING_DURATION_MS",
            OJ_GEN_AI_REASONING_DURATION_MS,
        ),
        ("OJ_WORKFLOW_ID", OJ_WORKFLOW_ID),
        ("OJ_WORKFLOW_NAME", OJ_WORKFLOW_NAME),
        ("OJ_WORKFLOW_VERSION", OJ_WORKFLOW_VERSION),
        ("OJ_WORKFLOW_COMPONENT_ID", OJ_WORKFLOW_COMPONENT_ID),
        ("OJ_WORKFLOW_COMPONENT_TYPE", OJ_WORKFLOW_COMPONENT_TYPE),
        ("OJ_WORKFLOW_COMPONENT_NAME", OJ_WORKFLOW_COMPONENT_NAME),
        ("OJ_WORKFLOW_EXECUTION_ID", OJ_WORKFLOW_EXECUTION_ID),
        ("OJ_WORKFLOW_LOOP_NODE_ID", OJ_WORKFLOW_LOOP_NODE_ID),
        ("OJ_WORKFLOW_LOOP_INDEX", OJ_WORKFLOW_LOOP_INDEX),
        ("OJ_AGENT_INVOKE_TYPE", OJ_AGENT_INVOKE_TYPE),
        ("OJ_AGENT_NAME", OJ_AGENT_NAME),
        ("OJ_AGENT_INPUTS", OJ_AGENT_INPUTS),
        ("OJ_AGENT_OUTPUTS", OJ_AGENT_OUTPUTS),
        ("OJ_AGENT_ERROR_MESSAGE", OJ_AGENT_ERROR_MESSAGE),
        ("OJ_TRACE_ID", OJ_TRACE_ID),
        ("OJ_SESSION_ID", OJ_SESSION_ID),
        ("OJ_INVOKE_ID", OJ_INVOKE_ID),
        ("OJ_PARENT_INVOKE_ID", OJ_PARENT_INVOKE_ID),
        ("OJ_START_TIME", OJ_START_TIME),
        ("OJ_END_TIME", OJ_END_TIME),
        ("OJ_ELAPSED_TIME", OJ_ELAPSED_TIME),
        ("OJ_STATUS", OJ_STATUS),
        ("OJ_ERROR", OJ_ERROR),
        ("OJ_CHILD_INVOKE_IDS", OJ_CHILD_INVOKE_IDS),
        ("OJ_META_DATA", OJ_META_DATA),
        ("OJ_PARENT_NODE_ID", OJ_PARENT_NODE_ID),
        ("OJ_SOURCE_IDS", OJ_SOURCE_IDS),
        ("OJ_INNER_ERROR", OJ_INNER_ERROR),
        ("OJ_STREAM_INPUTS", OJ_STREAM_INPUTS),
        ("OJ_STREAM_OUTPUTS", OJ_STREAM_OUTPUTS),
        ("OJ_INTERACTIVE_INPUTS", OJ_INTERACTIVE_INPUTS),
        ("OJ_WORKFLOW_INPUTS", OJ_WORKFLOW_INPUTS),
        ("OJ_WORKFLOW_OUTPUTS", OJ_WORKFLOW_OUTPUTS),
        ("OJ_WORKFLOW_ERROR_MESSAGE", OJ_WORKFLOW_ERROR_MESSAGE),
        ("OJ_WORKFLOW_INVOKE_DATA", OJ_WORKFLOW_INVOKE_DATA),
    ];

    /// Every value constant (provider, operation names, span statuses).
    const VALUES: &[(&str, &str)] = &[
        ("GEN_AI_PROVIDER_OPENJIUWEN", GEN_AI_PROVIDER_OPENJIUWEN),
        ("GEN_AI_OPERATION_CHAT", GEN_AI_OPERATION_CHAT),
        (
            "GEN_AI_OPERATION_EXECUTE_TOOL",
            GEN_AI_OPERATION_EXECUTE_TOOL,
        ),
        ("GEN_AI_OPERATION_EMBEDDINGS", GEN_AI_OPERATION_EMBEDDINGS),
        ("GEN_AI_OPERATION_RETRIEVAL", GEN_AI_OPERATION_RETRIEVAL),
        ("SPAN_STATUS_START", SPAN_STATUS_START),
        ("SPAN_STATUS_RUNNING", SPAN_STATUS_RUNNING),
        ("SPAN_STATUS_FINISH", SPAN_STATUS_FINISH),
        ("SPAN_STATUS_INTERRUPTED", SPAN_STATUS_INTERRUPTED),
        ("SPAN_STATUS_ERROR", SPAN_STATUS_ERROR),
    ];

    /// The exact wire value of every standard key the Python module re-exports.
    ///
    /// Python counterpart: `EXTENSIONS/observability/gen_ai_semconv.py`
    /// (revision fee465db333bdd6a7d2faa320edab5cf3101a4f4).
    #[test]
    fn standard_gen_ai_keys_carry_the_registry_wire_values() {
        let expected = [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.conversation.id",
            "gen_ai.data_source.id",
            "gen_ai.embeddings.dimension.count",
            "gen_ai.input.messages",
            "gen_ai.memory.record.count",
            "gen_ai.operation.name",
            "gen_ai.output.messages",
            "gen_ai.provider.name",
            "gen_ai.request.max_tokens",
            "gen_ai.request.model",
            "gen_ai.request.reasoning.level",
            "gen_ai.request.stop_sequences",
            "gen_ai.request.temperature",
            "gen_ai.request.top_k",
            "gen_ai.request.top_p",
            "gen_ai.response.finish_reasons",
            "gen_ai.response.model",
            "gen_ai.response.time_to_first_chunk",
            "gen_ai.retrieval.documents",
            "gen_ai.retrieval.query.text",
            "gen_ai.retrieval.top_k",
            "gen_ai.system_instructions",
            "gen_ai.tool.call.id",
            "gen_ai.tool.name",
            "gen_ai.tool.type",
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cache_write.input_tokens",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.reasoning.output_tokens",
        ];
        let actual: Vec<&str> = GEN_AI_KEYS.iter().map(|(_, value)| *value).collect();
        assert_eq!(actual, expected);
    }

    /// The exact wire value of every `openjiuwen.*` project extension.
    ///
    /// Python counterpart: `openjiuwen/extensions/tracer_otel/semconv.py`.
    #[test]
    fn openjiuwen_extension_keys_carry_the_python_wire_values() {
        let expected = [
            "openjiuwen.llm.prev_message_count",
            "openjiuwen.gen_ai.usage.total_cost",
            "openjiuwen.gen_ai.usage.input_cost",
            "openjiuwen.gen_ai.usage.output_cost",
            "openjiuwen.gen_ai.user.id",
            "openjiuwen.gen_ai.metadata",
            "openjiuwen.trace.name",
            "openjiuwen.gen_ai.response.inter_token_latency_ms",
            "openjiuwen.gen_ai.reasoning.duration_ms",
            "openjiuwen.workflow.id",
            "openjiuwen.workflow.name",
            "openjiuwen.workflow.version",
            "openjiuwen.workflow.component.id",
            "openjiuwen.workflow.component.type",
            "openjiuwen.workflow.component.name",
            "openjiuwen.workflow.execution_id",
            "openjiuwen.workflow.loop.node_id",
            "openjiuwen.workflow.loop.index",
            "openjiuwen.agent.invoke_type",
            "openjiuwen.agent.name",
            "openjiuwen.agent.inputs",
            "openjiuwen.agent.outputs",
            "openjiuwen.agent.error_message",
            "openjiuwen.trace.id",
            "openjiuwen.session_id",
            "openjiuwen.invoke_id",
            "openjiuwen.parent_invoke_id",
            "openjiuwen.start_time",
            "openjiuwen.end_time",
            "openjiuwen.elapsed_time",
            "openjiuwen.status",
            "openjiuwen.error",
            "openjiuwen.child_invoke_ids",
            "openjiuwen.meta_data",
            "openjiuwen.parent_node_id",
            "openjiuwen.source_ids",
            "openjiuwen.inner_error",
            "openjiuwen.stream_inputs",
            "openjiuwen.stream_outputs",
            "openjiuwen.interactive_inputs",
            "openjiuwen.workflow.inputs",
            "openjiuwen.workflow.outputs",
            "openjiuwen.workflow.error_message",
            "openjiuwen.workflow.invoke_data",
        ];
        let actual: Vec<&str> = OJ_KEYS.iter().map(|(_, value)| *value).collect();
        assert_eq!(actual, expected);
    }

    /// Representative values, spelled verbatim so a typo fails loudly.
    #[test]
    fn representative_values_are_verbatim() {
        assert_eq!(GEN_AI_PROVIDER_OPENJIUWEN, "openjiuwen");
        assert_eq!(GEN_AI_PROVIDER_NAME, "gen_ai.provider.name");
        assert_eq!(OJ_TRACE_ID, "openjiuwen.trace.id");
        assert_eq!(OJ_INVOKE_ID, "openjiuwen.invoke_id");
        assert_eq!(OJ_PARENT_INVOKE_ID, "openjiuwen.parent_invoke_id");
        assert_eq!(
            OJ_WORKFLOW_COMPONENT_TYPE,
            "openjiuwen.workflow.component.type"
        );
        assert_eq!(OJ_WORKFLOW_COMPONENT_ID, "openjiuwen.workflow.component.id");
        assert_eq!(OJ_META_DATA, "openjiuwen.meta_data");
        assert_eq!(OJ_GEN_AI_METADATA, "openjiuwen.gen_ai.metadata");
        assert_eq!(ERROR_TYPE, "error.type");
        assert_eq!(GEN_AI_OPERATION_NAME, "gen_ai.operation.name");
        assert_eq!(GEN_AI_OPERATION_CHAT, "chat");
        assert_eq!(GEN_AI_OPERATION_EXECUTE_TOOL, "execute_tool");
        assert_eq!(GEN_AI_OPERATION_EMBEDDINGS, "embeddings");
        assert_eq!(GEN_AI_OPERATION_RETRIEVAL, "retrieval");
        assert_eq!(SPAN_STATUS_START, "start");
        assert_eq!(SPAN_STATUS_RUNNING, "running");
        assert_eq!(SPAN_STATUS_FINISH, "finish");
        assert_eq!(SPAN_STATUS_INTERRUPTED, "interrupted");
        assert_eq!(SPAN_STATUS_ERROR, "error");
        // The status vocabulary mirrors the session's `SpanStatus` wire names.
        assert_eq!(
            SPAN_STATUS_FINISH,
            openjiuwen_session::SpanStatus::Finish.as_str()
        );
        assert_eq!(
            SPAN_STATUS_ERROR,
            openjiuwen_session::SpanStatus::Error.as_str()
        );
    }

    /// Every constant carries a non-empty value and no value is blank.
    #[test]
    fn all_constants_are_non_empty() {
        let all = [GEN_AI_KEYS, OJ_KEYS, VALUES].concat();
        assert!(all.len() > 1, "the constant inventory must not be empty");
        for (name, value) in all {
            assert!(!name.is_empty(), "constant {name} has an empty name");
            assert!(!value.is_empty(), "constant {name} has an empty value");
            assert_eq!(value, value.trim(), "constant {name} has padding");
        }
    }

    /// Prefix regime (issue #1833): standard keys ride `gen_ai.*`, project
    /// extensions ride `openjiuwen.*`, and the project's `gen_ai`-flavoured
    /// extensions stay under `openjiuwen.gen_ai.*` rather than `gen_ai.*`.
    #[test]
    fn names_use_the_correct_prefix_namespace() {
        for (name, value) in GEN_AI_KEYS {
            assert!(
                value.starts_with("gen_ai."),
                "standard key {name} = {value:?} must start with `gen_ai.`"
            );
        }
        for (name, value) in OJ_KEYS {
            assert!(
                value.starts_with("openjiuwen."),
                "project extension {name} = {value:?} must start with `openjiuwen.`"
            );
            assert!(
                !value.starts_with("gen_ai."),
                "project extension {name} = {value:?} must not ride the `gen_ai.*` prefix"
            );
        }
        assert_eq!(ERROR_TYPE, "error.type");
        for (name, value) in VALUES {
            assert!(
                !value.contains(' '),
                "value constant {name} = {value:?} must be a single token"
            );
        }
        // `openjiuwen.gen_ai.*` names are project keys, not standard ones.
        let gen_ai_extensions: Vec<&str> = OJ_KEYS
            .iter()
            .map(|(_, value)| *value)
            .filter(|value| value.contains(".gen_ai."))
            .collect();
        assert!(
            gen_ai_extensions.len() > 1,
            "the `openjiuwen.gen_ai.*` extensions must be present"
        );
        for value in gen_ai_extensions {
            assert!(
                value.starts_with("openjiuwen.gen_ai."),
                "{value:?} must be namespaced as `openjiuwen.gen_ai.*`"
            );
        }
    }

    /// Constant names are unique across the whole inventory (copy-paste drift).
    #[test]
    fn constant_names_and_values_are_unique() {
        let all = [GEN_AI_KEYS, OJ_KEYS, VALUES].concat();
        let mut names: Vec<&str> = all.iter().map(|(name, _)| *name).collect();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();
        assert_eq!(names.len(), unique, "duplicate constant name");

        let mut key_values: Vec<&str> = [GEN_AI_KEYS, OJ_KEYS]
            .concat()
            .iter()
            .map(|(_, value)| *value)
            .collect();
        key_values.sort_unstable();
        let unique = key_values.len();
        key_values.dedup();
        assert_eq!(key_values.len(), unique, "duplicate attribute key");
    }
}
