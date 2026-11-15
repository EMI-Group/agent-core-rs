//! Project-owned observability semantic conventions: attribute keys and closed value sets.
//!
//! Port of `openjiuwen/extensions/observability/semconv.py` and its generated companion
//! `openjiuwen/extensions/observability/gen_ai_semconv.py` (v0.1.20). Python keeps every
//! project-owned key in one importable module so the handlers cannot drift apart on a typo; this
//! module is that single place for the Rust handlers, and stays dependency-free so it compiles
//! into the default build.
//!
//! # Naming rule
//!
//! The constant name carries the namespace, and the wire name is the namespace plus the rest of the
//! name lowercased with `_` → `.`:
//!
//! - `OJ_*` → `openjiuwen.*` — additive project extensions (trajectory, correlation, streams, tools),
//! - `AT_*` → `agentteam.*` — team-collaboration attributes,
//! - `DA_*` → `deepagent.*` — DeepAgent task-loop attributes,
//! - `GEN_AI_*` → `gen_ai.*` — pinned upstream standard names.
//!
//! The mechanical part of the rule is the namespace and the dotted path; final compound nouns stay
//! one snake_case segment because the wire name is defined once, here, and never derived from the
//! constant name (`message_count`, `tool_call`, `recorded_at_unix_nano`, `request.number`,
//! `follow_up`, `loop_event`). Treat the literal as authoritative and the name as a hint.
//!
//! # No duplicated keys
//!
//! A key has exactly one definition across the observability trees. Where Python's `semconv.py`
//! re-exports a name that `openjiuwen/extensions/tracer_otel/semconv.py` already owns — the standard
//! `gen_ai.*` keys, `error.type`, and the four `openjiuwen.gen_ai.*` extension keys that module
//! declares — this module re-exports [`crate::otel::semconv`]'s constant instead of repeating the
//! literal, so a consumer importing `project_semconv` still finds the full set while no second
//! source of truth exists. [`REEXPORTED_KEYS`] enumerates those names; [`GEN_AI_KEYS`],
//! [`OJ_KEYS`], [`AT_KEYS`] and [`DA_KEYS`] enumerate the ones defined here.
//!
//! # Source and revision
//!
//! The standard names come from `open-telemetry/semantic-conventions-genai` revision
//! [`GEN_AI_SEMCONV_REVISION`]: [`GEN_AI_SEMCONV_ATTRIBUTE_COUNT`] attributes, GenAI schema
//! [`GEN_AI_SEMCONV_SCHEMA_URL`] layered over core schema
//! [`GEN_AI_CORE_SEMCONV_SCHEMA_URL`]. That pinned revision defines attribute names only — it
//! carries no `gen_ai.client.*` metric instruments (its token-usage and operation-duration metrics
//! are named nowhere in the registry, only alluded to in prose), so this module defines no metric
//! constants rather than inventing names the pinned upstream does not carry. Upgrading the registry
//! replaces this block as one unit.
//!
//! # Design rules kept from Python
//!
//! A project key exists only when the standard has no attribute for the same fact: session identity
//! is `gen_ai.conversation.id`, agent identity is `gen_ai.agent.name` / `gen_ai.agent.id`, tool-call
//! identity is `gen_ai.tool.call.id` / `gen_ai.tool.name`, and no `openjiuwen.*` mirror of any of
//! them is written, so no reader ever needs a fallback chain. `openjiuwen.gen_ai.*` is the additive
//! namespace for facts the standard does not model at all (cost, TPOT, logprobs, provider
//! payloads). Durations here are milliseconds and keep a load-bearing `_ms` suffix, because the
//! standard states durations in seconds (`gen_ai.response.time_to_first_chunk` is the one duration
//! attribute in the pinned registry, and the handler writes it in seconds).

// ---------------------------------------------------------------------------
// gen_ai.* — standard keys re-exported from the OTel semconv tree
// `crate::otel::semconv` already owns these literals (Python re-exports them from
// the same generated registry), so they are re-exported rather than redefined.
// ---------------------------------------------------------------------------

pub use crate::otel::semconv::{
    ERROR_TYPE, GEN_AI_AGENT_DESCRIPTION, GEN_AI_AGENT_ID, GEN_AI_AGENT_NAME,
    GEN_AI_CONVERSATION_ID, GEN_AI_DATA_SOURCE_ID, GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_INPUT_MESSAGES, GEN_AI_MEMORY_RECORD_COUNT, GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES, GEN_AI_PROVIDER_NAME, GEN_AI_REQUEST_MAX_TOKENS, GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_REASONING_LEVEL, GEN_AI_REQUEST_STOP_SEQUENCES, GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K, GEN_AI_REQUEST_TOP_P, GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL, GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK, GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT, GEN_AI_RETRIEVAL_TOP_K, GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME, GEN_AI_TOOL_TYPE, GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
};

// `openjiuwen.gen_ai.*` extension keys that the tracer-level semconv module already defines.
pub use crate::otel::semconv::{
    OJ_GEN_AI_REASONING_DURATION_MS, OJ_GEN_AI_USAGE_INPUT_COST, OJ_GEN_AI_USAGE_OUTPUT_COST,
    OJ_GEN_AI_USAGE_TOTAL_COST,
};

// ---------------------------------------------------------------------------
// Pinned upstream GenAI registry metadata
// Replacement granularity: the whole `gen_ai_semconv` block moves at once.
// ---------------------------------------------------------------------------

/// Upstream registry revision the standard names are pinned to.
pub const GEN_AI_SEMCONV_REVISION: &str = "fee465db333bdd6a7d2faa320edab5cf3101a4f4";

/// GenAI semantic-convention schema URL of the pinned revision.
pub const GEN_AI_SEMCONV_SCHEMA_URL: &str =
    "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev";

/// Core semantic-convention schema URL the GenAI schema layers over.
pub const GEN_AI_CORE_SEMCONV_SCHEMA_URL: &str = "https://opentelemetry.io/schemas/1.44.0";

/// Number of attributes the pinned registry defines.
pub const GEN_AI_SEMCONV_ATTRIBUTE_COUNT: usize = 72;

// ---------------------------------------------------------------------------
// gen_ai.* — standard keys the pinned registry defines and `otel::semconv` does not
// ---------------------------------------------------------------------------

/// Wire name `gen_ai.agent.version` — agent build/version identity.
pub const GEN_AI_AGENT_VERSION: &str = "gen_ai.agent.version";

/// Wire name `gen_ai.conversation.compacted` — conversation compaction marker.
pub const GEN_AI_CONVERSATION_COMPACTED: &str = "gen_ai.conversation.compacted";

/// Wire name `gen_ai.evaluation.explanation` — free-text rationale of an evaluation.
pub const GEN_AI_EVALUATION_EXPLANATION: &str = "gen_ai.evaluation.explanation";

/// Wire name `gen_ai.evaluation.name` — evaluation metric name.
pub const GEN_AI_EVALUATION_NAME: &str = "gen_ai.evaluation.name";

/// Wire name `gen_ai.evaluation.score.label` — categorical label of a score.
pub const GEN_AI_EVALUATION_SCORE_LABEL: &str = "gen_ai.evaluation.score.label";

/// Wire name `gen_ai.evaluation.score.value` — numeric evaluation score.
pub const GEN_AI_EVALUATION_SCORE_VALUE: &str = "gen_ai.evaluation.score.value";

/// Wire name `gen_ai.memory.query.text` — memory query text.
pub const GEN_AI_MEMORY_QUERY_TEXT: &str = "gen_ai.memory.query.text";

/// Wire name `gen_ai.memory.record.id` — identity of one memory record.
pub const GEN_AI_MEMORY_RECORD_ID: &str = "gen_ai.memory.record.id";

/// Wire name `gen_ai.memory.records` — JSON-encoded memory records.
pub const GEN_AI_MEMORY_RECORDS: &str = "gen_ai.memory.records";

/// Wire name `gen_ai.memory.store.id` — memory store identity.
pub const GEN_AI_MEMORY_STORE_ID: &str = "gen_ai.memory.store.id";

/// Wire name `gen_ai.output.type` — output modality (`text`, `json`, `image`, …).
pub const GEN_AI_OUTPUT_TYPE: &str = "gen_ai.output.type";

/// Wire name `gen_ai.prompt.name` — prompt template name.
pub const GEN_AI_PROMPT_NAME: &str = "gen_ai.prompt.name";

/// Wire name `gen_ai.prompt.variable` — JSON-encoded prompt template variables.
pub const GEN_AI_PROMPT_VARIABLE: &str = "gen_ai.prompt.variable";

/// Wire name `gen_ai.prompt.version` — prompt template version.
pub const GEN_AI_PROMPT_VERSION: &str = "gen_ai.prompt.version";

/// Wire name `gen_ai.request.choice.count` — requested completion count.
pub const GEN_AI_REQUEST_CHOICE_COUNT: &str = "gen_ai.request.choice.count";

/// Wire name `gen_ai.request.encoding_formats` — requested embedding encodings.
pub const GEN_AI_REQUEST_ENCODING_FORMATS: &str = "gen_ai.request.encoding_formats";

/// Wire name `gen_ai.request.frequency_penalty` — frequency penalty.
pub const GEN_AI_REQUEST_FREQUENCY_PENALTY: &str = "gen_ai.request.frequency_penalty";

/// Wire name `gen_ai.request.presence_penalty` — presence penalty.
pub const GEN_AI_REQUEST_PRESENCE_PENALTY: &str = "gen_ai.request.presence_penalty";

/// Wire name `gen_ai.request.previous_response.id` — previous response in a chain.
pub const GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: &str = "gen_ai.request.previous_response.id";

/// Wire name `gen_ai.request.seed` — determinism seed.
pub const GEN_AI_REQUEST_SEED: &str = "gen_ai.request.seed";

/// Wire name `gen_ai.request.stream` — whether the request asked for a stream.
pub const GEN_AI_REQUEST_STREAM: &str = "gen_ai.request.stream";

/// Wire name `gen_ai.request.stream_cursor` — resume cursor of a streamed request.
pub const GEN_AI_REQUEST_STREAM_CURSOR: &str = "gen_ai.request.stream_cursor";

/// Wire name `gen_ai.response.id` — provider response identity.
pub const GEN_AI_RESPONSE_ID: &str = "gen_ai.response.id";

/// Wire name `gen_ai.response.status` — provider-reported response status.
pub const GEN_AI_RESPONSE_STATUS: &str = "gen_ai.response.status";

/// Wire name `gen_ai.token.type` — token bucket a usage count belongs to.
pub const GEN_AI_TOKEN_TYPE: &str = "gen_ai.token.type";

/// Wire name `gen_ai.tool.call.arguments` — JSON-encoded tool-call arguments.
pub const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";

/// Wire name `gen_ai.tool.call.result` — JSON-encoded tool-call result.
pub const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";

/// Wire name `gen_ai.tool.definitions` — JSON-encoded tool definitions offered.
pub const GEN_AI_TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";

/// Wire name `gen_ai.tool.description` — tool description text.
pub const GEN_AI_TOOL_DESCRIPTION: &str = "gen_ai.tool.description";

/// Wire name `gen_ai.usage.audio.cache_read.input_tokens` — cached audio input tokens.
pub const GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS: &str =
    "gen_ai.usage.audio.cache_read.input_tokens";

/// Wire name `gen_ai.usage.audio.input_tokens` — audio input tokens.
pub const GEN_AI_USAGE_AUDIO_INPUT_TOKENS: &str = "gen_ai.usage.audio.input_tokens";

/// Wire name `gen_ai.usage.audio.output_tokens` — audio output tokens.
pub const GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS: &str = "gen_ai.usage.audio.output_tokens";

/// Wire name `gen_ai.usage.image.cache_read.input_tokens` — cached image input tokens.
pub const GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS: &str =
    "gen_ai.usage.image.cache_read.input_tokens";

/// Wire name `gen_ai.usage.image.input_tokens` — image input tokens.
pub const GEN_AI_USAGE_IMAGE_INPUT_TOKENS: &str = "gen_ai.usage.image.input_tokens";

/// Wire name `gen_ai.usage.image.output_tokens` — image output tokens.
pub const GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.image.output_tokens";

/// Wire name `gen_ai.usage.text.cache_read.input_tokens` — cached text input tokens.
pub const GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS: &str =
    "gen_ai.usage.text.cache_read.input_tokens";

/// Wire name `gen_ai.usage.text.input_tokens` — text input tokens.
pub const GEN_AI_USAGE_TEXT_INPUT_TOKENS: &str = "gen_ai.usage.text.input_tokens";

/// Wire name `gen_ai.usage.text.output_tokens` — text output tokens.
pub const GEN_AI_USAGE_TEXT_OUTPUT_TOKENS: &str = "gen_ai.usage.text.output_tokens";

/// Wire name `gen_ai.workflow.name` — workflow identity on workflow spans.
pub const GEN_AI_WORKFLOW_NAME: &str = "gen_ai.workflow.name";

// ---------------------------------------------------------------------------
// openjiuwen.* — additive trajectory and correlation attributes
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.trace.root` — marks the span that roots a trace.
pub const OJ_TRACE_ROOT: &str = "openjiuwen.trace.root";

/// Wire name `openjiuwen.trace.complete` — the trace finished without a forced close.
pub const OJ_TRACE_COMPLETE: &str = "openjiuwen.trace.complete";

/// Wire name `openjiuwen.trace.forced_close` — the trace was closed by the collector.
pub const OJ_TRACE_FORCED_CLOSE: &str = "openjiuwen.trace.forced_close";

/// Wire name `openjiuwen.span.forced_close` — the span was closed before it ended.
pub const OJ_SPAN_FORCED_CLOSE: &str = "openjiuwen.span.forced_close";

/// Wire name `openjiuwen.span.forced_close.reason` — why the span was force-closed.
pub const OJ_SPAN_FORCED_CLOSE_REASON: &str = "openjiuwen.span.forced_close.reason";

/// Wire name `openjiuwen.span.input` — backend-neutral JSON span input.
///
/// Used by records with no dedicated standard carrier (team/task/event/agent-tier spans); the
/// payload follows the same redaction and truncation policy as prompts.
pub const OJ_SPAN_INPUT: &str = "openjiuwen.span.input";

/// Wire name `openjiuwen.span.output` — backend-neutral JSON span output.
pub const OJ_SPAN_OUTPUT: &str = "openjiuwen.span.output";

/// Wire name `openjiuwen.request.id` — one LLM request.
pub const OJ_REQUEST_ID: &str = "openjiuwen.request.id";

/// Wire name `openjiuwen.request.message_count` — messages sent in the request.
pub const OJ_REQUEST_MESSAGE_COUNT: &str = "openjiuwen.request.message_count";

/// Wire name `openjiuwen.run.id` — one agent run.
pub const OJ_RUN_ID: &str = "openjiuwen.run.id";

/// Wire name `openjiuwen.turn.id` — one agent turn.
pub const OJ_TURN_ID: &str = "openjiuwen.turn.id";

/// Wire name `openjiuwen.turn.number` — ordinal of the turn inside its run.
pub const OJ_TURN_NUMBER: &str = "openjiuwen.turn.number";

/// Wire name `openjiuwen.step.id` — one agent step.
pub const OJ_STEP_ID: &str = "openjiuwen.step.id";

/// Wire name `openjiuwen.step.number` — ordinal of the step inside its turn.
pub const OJ_STEP_NUMBER: &str = "openjiuwen.step.number";

/// Wire name `openjiuwen.inference.id` — one model inference.
pub const OJ_INFERENCE_ID: &str = "openjiuwen.inference.id";

/// Wire name `openjiuwen.inference.input_observed` — whether the request was observed.
///
/// False when the inference was reported from the reply alone (a harness whose request logs never
/// arrived), so its input is unknown and its output may be partial; absent means observed.
pub const OJ_INFERENCE_INPUT_OBSERVED: &str = "openjiuwen.inference.input_observed";

/// Wire name `openjiuwen.request.number` — ordinal of the request for its subject.
pub const OJ_REQUEST_NUMBER: &str = "openjiuwen.request.number";

/// Wire name `openjiuwen.request.purpose` — why the request was issued.
pub const OJ_REQUEST_PURPOSE: &str = "openjiuwen.request.purpose";

/// Wire name `openjiuwen.request.retry_count` — provider-SDK retries inside one request.
///
/// Retry time is part of every request-relative duration, so a reader needs the count to tell a slow
/// provider from a retried one.
pub const OJ_REQUEST_RETRY_COUNT: &str = "openjiuwen.request.retry_count";

/// Wire name `openjiuwen.model_group.id` — logical model group of the request.
pub const OJ_MODEL_GROUP_ID: &str = "openjiuwen.model_group.id";

/// Wire name `openjiuwen.model_group.route_id` — deployment the group router picked.
pub const OJ_MODEL_GROUP_ROUTE_ID: &str = "openjiuwen.model_group.route_id";

/// Wire name `openjiuwen.context.operation.id` — one context operation.
pub const OJ_CONTEXT_OPERATION_ID: &str = "openjiuwen.context.operation.id";

/// Wire name `openjiuwen.compaction.number` — ordinal of this compaction for its subject.
///
/// Counts operations, not model calls: a throttled compaction is retried and every attempt states
/// the same number, so a reader counts what happened to the context.
pub const OJ_COMPACTION_NUMBER: &str = "openjiuwen.compaction.number";

/// Wire name `openjiuwen.trajectory.record.kind` — see [`TRAJECTORY_RECORD_KINDS`].
pub const OJ_TRAJECTORY_RECORD_KIND: &str = "openjiuwen.trajectory.record.kind";

/// Wire name `openjiuwen.trajectory.schema_version` — span-level contract version
/// ([`TRAJECTORY_SPAN_SCHEMA_VERSION`]).
pub const OJ_TRAJECTORY_SCHEMA_VERSION: &str = "openjiuwen.trajectory.schema_version";

/// Wire name `openjiuwen.trajectory.event_id` — identity of a v2 trajectory event.
pub const OJ_TRAJECTORY_EVENT_ID: &str = "openjiuwen.trajectory.event_id";

/// Wire name `openjiuwen.trajectory.event_kind` — see [`TRAJECTORY_EVENT_KINDS`].
pub const OJ_TRAJECTORY_EVENT_KIND: &str = "openjiuwen.trajectory.event_kind";

/// Wire name `openjiuwen.trajectory.subject_id` — subject a record belongs to.
pub const OJ_TRAJECTORY_SUBJECT_ID: &str = "openjiuwen.trajectory.subject_id";

/// Wire name `openjiuwen.trajectory.sequence_epoch` — epoch of the subject's sequence.
pub const OJ_TRAJECTORY_SEQUENCE_EPOCH: &str = "openjiuwen.trajectory.sequence_epoch";

/// Wire name `openjiuwen.trajectory.subject_sequence` — record ordinal within the epoch.
pub const OJ_TRAJECTORY_SUBJECT_SEQUENCE: &str = "openjiuwen.trajectory.subject_sequence";

/// Wire name `openjiuwen.trajectory.recorded_at_unix_nano` — record timestamp in unix nanos.
pub const OJ_TRAJECTORY_RECORDED_AT_UNIX_NANO: &str = "openjiuwen.trajectory.recorded_at_unix_nano";

/// Wire name `openjiuwen.trajectory.payload` — JSON-encoded record payload.
pub const OJ_TRAJECTORY_PAYLOAD: &str = "openjiuwen.trajectory.payload";

/// Wire name `openjiuwen.agent.mode` — agent execution mode on agent spans.
pub const OJ_AGENT_MODE: &str = "openjiuwen.agent.mode";

/// Wire name `openjiuwen.execution.subject.id` — subject of an execution record.
pub const OJ_EXECUTION_SUBJECT_ID: &str = "openjiuwen.execution.subject.id";

/// Wire name `openjiuwen.execution.subject.display_name` — human-readable subject name.
pub const OJ_EXECUTION_SUBJECT_DISPLAY_NAME: &str = "openjiuwen.execution.subject.display_name";

/// Wire name `openjiuwen.execution.subject.kind` — subject kind (agent, task, …).
pub const OJ_EXECUTION_SUBJECT_KIND: &str = "openjiuwen.execution.subject.kind";

/// Wire name `openjiuwen.execution.subject.parent_id` — parent subject identity.
pub const OJ_EXECUTION_SUBJECT_PARENT_ID: &str = "openjiuwen.execution.subject.parent_id";

/// Wire name `openjiuwen.execution.subject.session_id` — session a subject runs in.
pub const OJ_EXECUTION_SUBJECT_SESSION_ID: &str = "openjiuwen.execution.subject.session_id";

/// Wire name `openjiuwen.execution.subject.request.number` — request ordinal for the subject.
pub const OJ_EXECUTION_SUBJECT_REQUEST_NUMBER: &str = "openjiuwen.execution.subject.request.number";

// ---------------------------------------------------------------------------
// openjiuwen.gen_ai.* — additive extension namespace for facts the standard omits
// (cost, per-token identity, TPOT, logprobs, provider payloads). Project keys must not ride the
// `gen_ai.*` prefix, hence the nested namespace.
// ---------------------------------------------------------------------------

/// Wire name `openjiuwen.gen_ai.response.prompt_token_ids` — token identity of the prompt.
pub const OJ_GEN_AI_RESPONSE_PROMPT_TOKEN_IDS: &str = "openjiuwen.gen_ai.response.prompt_token_ids";

/// Wire name `openjiuwen.gen_ai.response.completion_token_ids` — token identity of the completion.
pub const OJ_GEN_AI_RESPONSE_COMPLETION_TOKEN_IDS: &str =
    "openjiuwen.gen_ai.response.completion_token_ids";

/// Wire name `openjiuwen.gen_ai.response.logprobs` — token log probabilities.
pub const OJ_GEN_AI_RESPONSE_LOGPROBS: &str = "openjiuwen.gen_ai.response.logprobs";

/// Wire name `openjiuwen.gen_ai.response.parser_result` — structured-output parser result.
pub const OJ_GEN_AI_RESPONSE_PARSER_RESULT: &str = "openjiuwen.gen_ai.response.parser_result";

/// Wire name `openjiuwen.gen_ai.response.provider_metadata` — provider-specific metadata.
pub const OJ_GEN_AI_RESPONSE_PROVIDER_METADATA: &str =
    "openjiuwen.gen_ai.response.provider_metadata";

/// Wire name `openjiuwen.gen_ai.response.provider_content` — raw provider content.
pub const OJ_GEN_AI_RESPONSE_PROVIDER_CONTENT: &str = "openjiuwen.gen_ai.response.provider_content";

/// Wire name `openjiuwen.gen_ai.input.message_provenance` — where an input message came from.
pub const OJ_GEN_AI_INPUT_MESSAGE_PROVENANCE: &str = "openjiuwen.gen_ai.input.message_provenance";

/// Wire name `openjiuwen.gen_ai.reasoning.timing` — reasoning duration or
/// [`OJ_GEN_AI_REASONING_TIMING_UNMEASURED`].
pub const OJ_GEN_AI_REASONING_TIMING: &str = "openjiuwen.gen_ai.reasoning.timing";

/// Wire name `openjiuwen.gen_ai.response.total_latency_ms` — whole call, in milliseconds.
pub const OJ_GEN_AI_RESPONSE_TOTAL_LATENCY_MS: &str = "openjiuwen.gen_ai.response.total_latency_ms";

/// Wire name `openjiuwen.gen_ai.response.tpot_ms` — mean time per output token, in milliseconds.
pub const OJ_GEN_AI_RESPONSE_TPOT_MS: &str = "openjiuwen.gen_ai.response.tpot_ms";

/// Wire name `openjiuwen.gen_ai.response.time_to_first_byte_ms` — until response headers arrive.
pub const OJ_GEN_AI_RESPONSE_TIME_TO_FIRST_BYTE_MS: &str =
    "openjiuwen.gen_ai.response.time_to_first_byte_ms";

/// Wire name `openjiuwen.gen_ai.response.time_to_first_token_ms` — until the first chunk carrying
/// model output.
pub const OJ_GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN_MS: &str =
    "openjiuwen.gen_ai.response.time_to_first_token_ms";

// Request-relative durations share one origin: the moment the LLM span opens, before the client
// builds the request. They are client-side end-to-end measurements including request preparation,
// network and gateway time. The nearest standard relative is `total_latency_ms`: the standard's
// `gen_ai.client.operation.duration` metric measures the same fact in seconds, under another name
// and as a metric rather than a span attribute, so neither is a duplicate.

/// Wire name `openjiuwen.event.sequence` — ordinal of an event frame.
pub const OJ_EVENT_SEQUENCE: &str = "openjiuwen.event.sequence";

/// Wire name `openjiuwen.stream.kind` — see [`STREAM_FRAME_KINDS`].
pub const OJ_STREAM_KIND: &str = "openjiuwen.stream.kind";

/// Wire name `openjiuwen.stream.text` — text carried by a stream frame.
pub const OJ_STREAM_TEXT: &str = "openjiuwen.stream.text";

/// Wire name `openjiuwen.stream.tool_call.arguments_delta` — tool-argument fragment.
pub const OJ_STREAM_TOOL_CALL_ARGUMENTS_DELTA: &str = "openjiuwen.stream.tool_call.arguments_delta";

/// Span event name `openjiuwen.stream.chunk` — one model-stream frame.
///
/// Frames ride the stream-frame channel rather than span events: an answer produces hundreds of
/// them and the SDK's per-span event limit evicts the oldest, which would drop the beginning of every
/// long answer.
pub const OJ_STREAM_FRAME_EVENT: &str = "openjiuwen.stream.chunk";

/// Span event name `openjiuwen.stream.open` — the stream started.
pub const OJ_STREAM_OPEN_EVENT: &str = "openjiuwen.stream.open";

/// Span event name `openjiuwen.stream.phase.open` — a stream phase opened.
pub const OJ_STREAM_PHASE_OPEN_EVENT: &str = "openjiuwen.stream.phase.open";

/// Span event name `openjiuwen.stream.phase.close` — a stream phase closed.
pub const OJ_STREAM_PHASE_CLOSE_EVENT: &str = "openjiuwen.stream.phase.close";

/// Span event name `openjiuwen.stream.close` — the stream ended.
pub const OJ_STREAM_CLOSE_EVENT: &str = "openjiuwen.stream.close";

/// Wire name `openjiuwen.stream.phase.first_sequence` — first frame a phase marker covers.
pub const OJ_STREAM_PHASE_FIRST_SEQUENCE: &str = "openjiuwen.stream.phase.first_sequence";

/// Wire name `openjiuwen.stream.phase.last_sequence` — last frame a phase marker covers.
pub const OJ_STREAM_PHASE_LAST_SEQUENCE: &str = "openjiuwen.stream.phase.last_sequence";

/// Wire name `openjiuwen.stream.frame_count` — frames a phase marker covers.
pub const OJ_STREAM_FRAME_COUNT: &str = "openjiuwen.stream.frame_count";

/// Wire name `openjiuwen.tool.resource_id` — MCP resource a tool acts on.
pub const OJ_TOOL_RESOURCE_ID: &str = "openjiuwen.tool.resource_id";

/// Wire name `openjiuwen.tool.protocol` — tool transport protocol.
pub const OJ_TOOL_PROTOCOL: &str = "openjiuwen.tool.protocol";

/// Wire name `openjiuwen.tool.authoritative` — the tool result is authoritative.
pub const OJ_TOOL_AUTHORITATIVE: &str = "openjiuwen.tool.authoritative";

// ---------------------------------------------------------------------------
// agentteam.* — team-level collaboration attributes (Monitor handler)
// ---------------------------------------------------------------------------

/// Wire name `agentteam.team.id` — team identity.
pub const AT_TEAM_ID: &str = "agentteam.team.id";

/// Wire name `agentteam.team.name` — team name.
pub const AT_TEAM_NAME: &str = "agentteam.team.name";

/// Wire name `agentteam.team.display_name` — team display name.
pub const AT_TEAM_DISPLAY_NAME: &str = "agentteam.team.display_name";

/// Wire name `agentteam.team.leader` — team leader identity.
pub const AT_TEAM_LEADER: &str = "agentteam.team.leader";

/// Wire name `agentteam.event_type` — team event discriminator.
pub const AT_EVENT_TYPE: &str = "agentteam.event_type";

/// Wire name `agentteam.agent.id` — team member agent identity.
pub const AT_AGENT_ID: &str = "agentteam.agent.id";

/// Wire name `agentteam.agent.role` — role a team member plays.
pub const AT_AGENT_ROLE: &str = "agentteam.agent.role";

/// Wire name `agentteam.member.name` — member name.
pub const AT_MEMBER_NAME: &str = "agentteam.member.name";

/// Wire name `agentteam.member.status.old` — member status before a transition.
pub const AT_MEMBER_STATUS_OLD: &str = "agentteam.member.status.old";

/// Wire name `agentteam.member.status.new` — member status after a transition.
pub const AT_MEMBER_STATUS_NEW: &str = "agentteam.member.status.new";

/// Wire name `agentteam.member.restart_reason` — why a member was restarted.
pub const AT_MEMBER_RESTART_REASON: &str = "agentteam.member.restart_reason";

/// Wire name `agentteam.member.restart_count` — restarts a member has taken.
pub const AT_MEMBER_RESTART_COUNT: &str = "agentteam.member.restart_count";

/// Wire name `agentteam.member.shutdown_force` — the member shutdown was forced.
pub const AT_MEMBER_SHUTDOWN_FORCE: &str = "agentteam.member.shutdown_force";

/// Wire name `agentteam.message.id` — team message identity.
pub const AT_MESSAGE_ID: &str = "agentteam.message.id";

/// Wire name `agentteam.message.from` — message sender.
pub const AT_MESSAGE_FROM: &str = "agentteam.message.from";

/// Wire name `agentteam.message.to` — message recipient.
pub const AT_MESSAGE_TO: &str = "agentteam.message.to";

/// Wire name `agentteam.message.broadcast` — the message was broadcast.
pub const AT_MESSAGE_BROADCAST: &str = "agentteam.message.broadcast";

/// Wire name `agentteam.task.id` — team task identity.
pub const AT_TASK_ID: &str = "agentteam.task.id";

/// Wire name `agentteam.task.status` — team task status.
pub const AT_TASK_STATUS: &str = "agentteam.task.status";

/// Wire name `agentteam.task.assignee` — member a task is assigned to.
pub const AT_TASK_ASSIGNEE: &str = "agentteam.task.assignee";

/// Wire name `agentteam.task.tag` — task tag.
pub const AT_TASK_TAG: &str = "agentteam.task.tag";

/// Wire name `agentteam.task.recovered` — the task span was rebuilt after a pause/resume cycle,
/// so the trace viewer can tell it apart from a task created inside the current trace.
pub const AT_TASK_RECOVERED: &str = "agentteam.task.recovered";

/// Wire name `agentteam.plan.approved` — the plan was approved.
pub const AT_PLAN_APPROVED: &str = "agentteam.plan.approved";

/// Wire name `agentteam.plan.submitted_by` — who submitted the plan.
pub const AT_PLAN_SUBMITTED_BY: &str = "agentteam.plan.submitted_by";

// ---------------------------------------------------------------------------
// deepagent.* — DeepAgent task-loop attributes (Rail)
// ---------------------------------------------------------------------------

/// Wire name `deepagent.task.iteration` — iteration of the task loop.
pub const DA_TASK_ITERATION: &str = "deepagent.task.iteration";

/// Wire name `deepagent.task.is_follow_up` — the iteration answers a follow-up.
pub const DA_TASK_IS_FOLLOW_UP: &str = "deepagent.task.is_follow_up";

/// Wire name `deepagent.task.loop_event` — task-loop event discriminator.
pub const DA_TASK_LOOP_EVENT: &str = "deepagent.task.loop_event";

// ---------------------------------------------------------------------------
// Attribute values — everything above this section is an attribute key
// ---------------------------------------------------------------------------

/// The one span-level trajectory contract version, the value of
/// [`OJ_TRAJECTORY_SCHEMA_VERSION`].
///
/// Every canonical span states it; v2 events are identified by [`OJ_TRAJECTORY_EVENT_KIND`], not by
/// this value.
pub const TRAJECTORY_SPAN_SCHEMA_VERSION: &str = "2";

/// Closed set of [`OJ_TRAJECTORY_RECORD_KIND`] values.
///
/// Readers generated from this module (the web trajectory viewer) reject anything outside it, so a
/// new record kind is added here first.
pub const TRAJECTORY_RECORD_KINDS: [&str; 7] = [
    "turn",
    "step",
    "inference",
    "reasoning",
    "tool",
    "agent",
    "event",
];

/// Closed set of [`OJ_TRAJECTORY_EVENT_KIND`] values a v2 event may state.
pub const TRAJECTORY_EVENT_KINDS: [&str; 4] = [
    "context.window.commit",
    "compaction.completed",
    "ask_user.requested",
    "ask_user.resolved",
];

/// Value of [`OJ_GEN_AI_REASONING_TIMING`] when no reasoning duration could be measured.
///
/// A zero would itself be a measurement, so the reason is recorded in place of a duration.
pub const OJ_GEN_AI_REASONING_TIMING_UNMEASURED: &str = "unmeasured: non-streaming call";

/// Span event names the stream instrumentation writes, in Python's declaration order.
pub const STREAM_SPAN_EVENT_NAMES: [&str; 5] = [
    OJ_STREAM_FRAME_EVENT,
    OJ_STREAM_OPEN_EVENT,
    OJ_STREAM_CLOSE_EVENT,
    OJ_STREAM_PHASE_OPEN_EVENT,
    OJ_STREAM_PHASE_CLOSE_EVENT,
];

/// [`OJ_STREAM_KIND`] value for a text fragment of an answer.
pub const STREAM_FRAME_KIND_TEXT_DELTA: &str = "text-delta";

/// [`OJ_STREAM_KIND`] value for a reasoning fragment.
pub const STREAM_FRAME_KIND_REASONING_DELTA: &str = "reasoning-delta";

/// [`OJ_STREAM_KIND`] value for a tool-call argument fragment.
pub const STREAM_FRAME_KIND_TOOL_CALL_DELTA: &str = "tool-call-delta";

/// [`OJ_STREAM_KIND`] value for a usage report frame.
pub const STREAM_FRAME_KIND_USAGE: &str = "usage";

/// Closed set of [`OJ_STREAM_KIND`] values.
///
/// Python writes these as bare literals in `callback_handler.py`; the Rust side names them so a
/// frame producer cannot misspell one.
pub const STREAM_FRAME_KINDS: [&str; 4] = [
    STREAM_FRAME_KIND_TEXT_DELTA,
    STREAM_FRAME_KIND_REASONING_DELTA,
    STREAM_FRAME_KIND_TOOL_CALL_DELTA,
    STREAM_FRAME_KIND_USAGE,
];

// ---------------------------------------------------------------------------
// Inventories — enumerable key sets for consumers and for the drift tests
// ---------------------------------------------------------------------------

/// Every `gen_ai.*` key this module defines (standard keys absent from [`crate::otel::semconv`]).
pub const GEN_AI_KEYS: &[&str] = &[
    GEN_AI_AGENT_VERSION,
    GEN_AI_CONVERSATION_COMPACTED,
    GEN_AI_EVALUATION_EXPLANATION,
    GEN_AI_EVALUATION_NAME,
    GEN_AI_EVALUATION_SCORE_LABEL,
    GEN_AI_EVALUATION_SCORE_VALUE,
    GEN_AI_MEMORY_QUERY_TEXT,
    GEN_AI_MEMORY_RECORD_ID,
    GEN_AI_MEMORY_RECORDS,
    GEN_AI_MEMORY_STORE_ID,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_PROMPT_NAME,
    GEN_AI_PROMPT_VARIABLE,
    GEN_AI_PROMPT_VERSION,
    GEN_AI_REQUEST_CHOICE_COUNT,
    GEN_AI_REQUEST_ENCODING_FORMATS,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_REQUEST_STREAM_CURSOR,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_STATUS,
    GEN_AI_TOKEN_TYPE,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_DESCRIPTION,
    GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
    GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
    GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
    GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_TEXT_INPUT_TOKENS,
    GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
    GEN_AI_WORKFLOW_NAME,
];

/// Every `openjiuwen.*` key this module defines.
pub const OJ_KEYS: &[&str] = &[
    OJ_TRACE_ROOT,
    OJ_TRACE_COMPLETE,
    OJ_TRACE_FORCED_CLOSE,
    OJ_SPAN_FORCED_CLOSE,
    OJ_SPAN_FORCED_CLOSE_REASON,
    OJ_SPAN_INPUT,
    OJ_SPAN_OUTPUT,
    OJ_REQUEST_ID,
    OJ_REQUEST_MESSAGE_COUNT,
    OJ_RUN_ID,
    OJ_TURN_ID,
    OJ_TURN_NUMBER,
    OJ_STEP_ID,
    OJ_STEP_NUMBER,
    OJ_INFERENCE_ID,
    OJ_INFERENCE_INPUT_OBSERVED,
    OJ_REQUEST_NUMBER,
    OJ_REQUEST_PURPOSE,
    OJ_REQUEST_RETRY_COUNT,
    OJ_MODEL_GROUP_ID,
    OJ_MODEL_GROUP_ROUTE_ID,
    OJ_CONTEXT_OPERATION_ID,
    OJ_COMPACTION_NUMBER,
    OJ_TRAJECTORY_RECORD_KIND,
    OJ_TRAJECTORY_SCHEMA_VERSION,
    OJ_TRAJECTORY_EVENT_ID,
    OJ_TRAJECTORY_EVENT_KIND,
    OJ_TRAJECTORY_SUBJECT_ID,
    OJ_TRAJECTORY_SEQUENCE_EPOCH,
    OJ_TRAJECTORY_SUBJECT_SEQUENCE,
    OJ_TRAJECTORY_RECORDED_AT_UNIX_NANO,
    OJ_TRAJECTORY_PAYLOAD,
    OJ_AGENT_MODE,
    OJ_EXECUTION_SUBJECT_ID,
    OJ_EXECUTION_SUBJECT_DISPLAY_NAME,
    OJ_EXECUTION_SUBJECT_KIND,
    OJ_EXECUTION_SUBJECT_PARENT_ID,
    OJ_EXECUTION_SUBJECT_SESSION_ID,
    OJ_EXECUTION_SUBJECT_REQUEST_NUMBER,
    OJ_GEN_AI_RESPONSE_PROMPT_TOKEN_IDS,
    OJ_GEN_AI_RESPONSE_COMPLETION_TOKEN_IDS,
    OJ_GEN_AI_RESPONSE_LOGPROBS,
    OJ_GEN_AI_RESPONSE_PARSER_RESULT,
    OJ_GEN_AI_RESPONSE_PROVIDER_METADATA,
    OJ_GEN_AI_RESPONSE_PROVIDER_CONTENT,
    OJ_GEN_AI_INPUT_MESSAGE_PROVENANCE,
    OJ_GEN_AI_REASONING_TIMING,
    OJ_GEN_AI_RESPONSE_TOTAL_LATENCY_MS,
    OJ_GEN_AI_RESPONSE_TPOT_MS,
    OJ_GEN_AI_RESPONSE_TIME_TO_FIRST_BYTE_MS,
    OJ_GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN_MS,
    OJ_EVENT_SEQUENCE,
    OJ_STREAM_KIND,
    OJ_STREAM_TEXT,
    OJ_STREAM_TOOL_CALL_ARGUMENTS_DELTA,
    OJ_STREAM_FRAME_EVENT,
    OJ_STREAM_OPEN_EVENT,
    OJ_STREAM_PHASE_OPEN_EVENT,
    OJ_STREAM_PHASE_CLOSE_EVENT,
    OJ_STREAM_CLOSE_EVENT,
    OJ_STREAM_PHASE_FIRST_SEQUENCE,
    OJ_STREAM_PHASE_LAST_SEQUENCE,
    OJ_STREAM_FRAME_COUNT,
    OJ_TOOL_RESOURCE_ID,
    OJ_TOOL_PROTOCOL,
    OJ_TOOL_AUTHORITATIVE,
];

/// Every `agentteam.*` key this module defines.
pub const AT_KEYS: &[&str] = &[
    AT_TEAM_ID,
    AT_TEAM_NAME,
    AT_TEAM_DISPLAY_NAME,
    AT_TEAM_LEADER,
    AT_EVENT_TYPE,
    AT_AGENT_ID,
    AT_AGENT_ROLE,
    AT_MEMBER_NAME,
    AT_MEMBER_STATUS_OLD,
    AT_MEMBER_STATUS_NEW,
    AT_MEMBER_RESTART_REASON,
    AT_MEMBER_RESTART_COUNT,
    AT_MEMBER_SHUTDOWN_FORCE,
    AT_MESSAGE_ID,
    AT_MESSAGE_FROM,
    AT_MESSAGE_TO,
    AT_MESSAGE_BROADCAST,
    AT_TASK_ID,
    AT_TASK_STATUS,
    AT_TASK_ASSIGNEE,
    AT_TASK_TAG,
    AT_TASK_RECOVERED,
    AT_PLAN_APPROVED,
    AT_PLAN_SUBMITTED_BY,
];

/// Every `deepagent.*` key this module defines.
pub const DA_KEYS: &[&str] = &[DA_TASK_ITERATION, DA_TASK_IS_FOLLOW_UP, DA_TASK_LOOP_EVENT];

/// Keys re-exported from [`crate::otel::semconv`] instead of redefined here.
///
/// Listed so a consumer sees the complete set and so the drift tests can prove no key is defined
/// twice.
pub const REEXPORTED_KEYS: &[&str] = &[
    ERROR_TYPE,
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_CONVERSATION_ID,
    GEN_AI_DATA_SOURCE_ID,
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_MEMORY_RECORD_COUNT,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_REASONING_LEVEL,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    OJ_GEN_AI_REASONING_DURATION_MS,
    OJ_GEN_AI_USAGE_INPUT_COST,
    OJ_GEN_AI_USAGE_OUTPUT_COST,
    OJ_GEN_AI_USAGE_TOTAL_COST,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// One pinned literal per project namespace (Python `semconv.py` parity).
    #[test]
    fn pinned_literals_match_python() {
        assert_eq!(OJ_TRACE_ROOT, "openjiuwen.trace.root");
        assert_eq!(
            OJ_GEN_AI_RESPONSE_TPOT_MS,
            "openjiuwen.gen_ai.response.tpot_ms"
        );
        assert_eq!(AT_TEAM_ID, "agentteam.team.id");
        assert_eq!(DA_TASK_ITERATION, "deepagent.task.iteration");
        assert_eq!(GEN_AI_WORKFLOW_NAME, "gen_ai.workflow.name");
        assert_eq!(ERROR_TYPE, "error.type");
    }

    /// The naming rule: namespace prefix, then the name lowercased with `_` → `.`.
    #[test]
    fn naming_rule_holds_for_samples() {
        fn mechanize(prefix: &str, namespace: &str, name: &str) -> String {
            let rest = name
                .strip_prefix(prefix)
                .unwrap_or_else(|| panic!("{name} lacks the {prefix} prefix"))
                .to_ascii_lowercase()
                .replace('_', ".");
            format!("{namespace}{rest}")
        }
        for (prefix, namespace, name, value) in [
            ("OJ_", "openjiuwen.", "OJ_RUN_ID", OJ_RUN_ID),
            ("OJ_", "openjiuwen.", "OJ_TURN_NUMBER", OJ_TURN_NUMBER),
            (
                "OJ_",
                "openjiuwen.",
                "OJ_TRAJECTORY_PAYLOAD",
                OJ_TRAJECTORY_PAYLOAD,
            ),
            ("AT_", "agentteam.", "AT_TEAM_ID", AT_TEAM_ID),
            ("AT_", "agentteam.", "AT_TASK_RECOVERED", AT_TASK_RECOVERED),
            ("DA_", "deepagent.", "DA_TASK_ITERATION", DA_TASK_ITERATION),
            (
                "GEN_AI_",
                "gen_ai.",
                "GEN_AI_WORKFLOW_NAME",
                GEN_AI_WORKFLOW_NAME,
            ),
            (
                "GEN_AI_",
                "gen_ai.",
                "GEN_AI_RESPONSE_ID",
                GEN_AI_RESPONSE_ID,
            ),
        ] {
            assert_eq!(
                mechanize(prefix, namespace, name),
                value,
                "naming rule broke for {name}"
            );
        }
        // Compound nouns are wire-defined, not mechanically derived: the literal wins.
        assert_eq!(OJ_REQUEST_MESSAGE_COUNT, "openjiuwen.request.message_count");
        assert_eq!(
            OJ_TRAJECTORY_RECORDED_AT_UNIX_NANO,
            "openjiuwen.trajectory.recorded_at_unix_nano"
        );
        assert_eq!(
            OJ_STREAM_TOOL_CALL_ARGUMENTS_DELTA,
            "openjiuwen.stream.tool_call.arguments_delta"
        );
        assert_eq!(DA_TASK_IS_FOLLOW_UP, "deepagent.task.is_follow_up");
    }

    /// Every defined key rides its namespace and nothing else.
    #[test]
    fn inventories_use_their_namespace_prefix() {
        for key in GEN_AI_KEYS {
            assert!(key.starts_with("gen_ai."), "{key} must ride `gen_ai.`");
        }
        for key in OJ_KEYS {
            assert!(
                key.starts_with("openjiuwen."),
                "{key} must ride `openjiuwen.`"
            );
            assert!(!key.starts_with("gen_ai."), "{key} must not ride `gen_ai.`");
        }
        for key in AT_KEYS {
            assert!(
                key.starts_with("agentteam."),
                "{key} must ride `agentteam.`"
            );
        }
        for key in DA_KEYS {
            assert!(
                key.starts_with("deepagent."),
                "{key} must ride `deepagent.`"
            );
        }
        for key in REEXPORTED_KEYS {
            assert!(!key.is_empty(), "empty re-exported key");
        }
    }

    /// No key string appears twice, in one inventory or across the re-exported set.
    #[test]
    fn keys_are_unique_across_the_module() {
        let all: Vec<&str> = [GEN_AI_KEYS, OJ_KEYS, AT_KEYS, DA_KEYS, REEXPORTED_KEYS].concat();
        let mut sorted = all.clone();
        sorted.sort_unstable();
        let total = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), total, "duplicate attribute key in the module");

        for key in &all {
            assert_eq!(*key, key.trim(), "{key} has padding");
            assert!(!key.contains(' '), "{key} contains a space");
            assert!(
                !key.chars().any(|c| c.is_ascii_uppercase()),
                "{key} is not lowercase"
            );
        }
    }

    /// Inventory sizes pin completeness: a new key must be added to its table too.
    #[test]
    fn inventory_sizes_are_pinned() {
        assert_eq!(GEN_AI_KEYS.len(), 39, "pinned registry keys defined here");
        assert_eq!(OJ_KEYS.len(), 66, "openjiuwen.* keys defined here");
        assert_eq!(AT_KEYS.len(), 24, "agentteam.* keys");
        assert_eq!(DA_KEYS.len(), 3, "deepagent.* keys");
        assert_eq!(REEXPORTED_KEYS.len(), 38, "keys owned by otel::semconv");
        // 39 + 33 re-exported standard keys = the 72 attributes of the pinned registry.
        assert_eq!(GEN_AI_KEYS.len() + 33, GEN_AI_SEMCONV_ATTRIBUTE_COUNT);
    }

    /// The closed value sets are exactly Python's tuples.
    #[test]
    fn closed_value_sets_match_python() {
        assert_eq!(TRAJECTORY_SPAN_SCHEMA_VERSION, "2");
        assert_eq!(
            TRAJECTORY_RECORD_KINDS,
            [
                "turn",
                "step",
                "inference",
                "reasoning",
                "tool",
                "agent",
                "event"
            ]
        );
        assert_eq!(
            TRAJECTORY_EVENT_KINDS,
            [
                "context.window.commit",
                "compaction.completed",
                "ask_user.requested",
                "ask_user.resolved"
            ]
        );
        assert_eq!(
            OJ_GEN_AI_REASONING_TIMING_UNMEASURED,
            "unmeasured: non-streaming call"
        );
    }

    /// Stream span-event names and frame kinds are the Python literals.
    #[test]
    fn stream_event_names_and_frame_kinds_match_python() {
        assert_eq!(
            STREAM_SPAN_EVENT_NAMES,
            [
                "openjiuwen.stream.chunk",
                "openjiuwen.stream.open",
                "openjiuwen.stream.close",
                "openjiuwen.stream.phase.open",
                "openjiuwen.stream.phase.close"
            ]
        );
        assert_eq!(
            STREAM_FRAME_KINDS,
            ["text-delta", "reasoning-delta", "tool-call-delta", "usage"]
        );
        // Frame kinds are values, not keys: they must stay out of the key inventories.
        for kind in STREAM_FRAME_KINDS {
            assert!(!OJ_KEYS.contains(&kind));
        }
    }

    /// Registry metadata is the pinned upstream revision.
    #[test]
    fn registry_metadata_is_pinned() {
        assert_eq!(GEN_AI_SEMCONV_REVISION.len(), 40);
        assert!(GEN_AI_SEMCONV_REVISION.starts_with("fee465db"));
        assert_eq!(
            GEN_AI_SEMCONV_SCHEMA_URL,
            "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev"
        );
        assert_eq!(
            GEN_AI_CORE_SEMCONV_SCHEMA_URL,
            "https://opentelemetry.io/schemas/1.44.0"
        );
        assert_eq!(GEN_AI_SEMCONV_ATTRIBUTE_COUNT, 72);
    }
}
