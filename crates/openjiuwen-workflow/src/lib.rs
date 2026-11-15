//! Workflow builder, components, and reference-based data flow.
//!
//! Maps to the Python package `core.workflow` in the openJiuwen reference implementation.
//! A workflow is a directed graph of components with one start and one end; the builder
//! records component configs and edges and compiling maps them onto `openjiuwen-graph`.
//! Data flow is textual: a component's `inputs_schema` is a JSON template whose pure
//! reference strings resolve against workflow state before the component runs.
//!
//! # Re-exports
//!
//! The crate root mirrors Python's `openjiuwen/core/workflow/__init__.py`: every built-in
//! component, the whole `Http*` configuration tree, the component-local Python status codes,
//! and the Python-name type aliases (`LLMComponent`, `LLMCompConfig`,
//! `ReActAgentCompConfig`) are re-exported here, so `use openjiuwen_workflow::…` reaches the
//! same surface as the Python package root.
//!
//! The per-component error *builders* are deliberately **not** re-exported at the root: names
//! such as `config_error`, `input_param_error`, or `runtime_error` would pollute the crate
//! namespace. They stay in their owning modules —
//! `components::questioner::{input_param_error, config_error, …}` and
//! `components::intent_detection::{intent_detection_config_error, …}` — while their status
//! codes (`COMPONENT_QUESTIONER_*`, `COMPONENT_INTENT_DETECTION_*`) are available both there
//! and at the crate root.

pub mod ability;
pub mod card;
pub mod component;
pub mod components;
pub mod condition;
pub mod context;
pub mod engine;
pub mod error;
pub mod interrupt;
pub mod output;
pub mod reference;
pub mod registry;
pub mod render;
pub mod single;
pub mod workflow;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-workflow"
}

pub use ability::ComponentAbility;
pub use card::{
    generate_workflow_key, validate_component_id, CompIoConfig, ExceptionConfig, NodeSpec,
    WorkflowCard, WorkflowConfig, WorkflowSpec, INVALID_ID_CHARS_REASON, MAX_COMPONENT_ID_LEN,
};
pub use component::{unsupported_ability, Component, ComponentCard};
// --- components (component + full config surface, as in the Python package root) ---
/// The `intent_detection` status codes (Python `COMPONENT_INTENT_DETECTION_*`).
///
/// The matching error builders stay in [`components::intent_detection`]; see the crate docs.
pub use components::intent_detection::{
    COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID,
    COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE,
    COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID, COMPONENT_INTENT_DETECTION_CONFIG_ERROR,
    COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID, COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR,
    COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR, COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED,
    COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN,
    COMPONENT_INTENT_DETECTION_KEYWORD_INVALID, COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED,
    COMPONENT_INTENT_DETECTION_PROMPT_INVALID, COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR,
};
/// The `questioner` status codes (Python `COMPONENT_QUESTIONER_*`).
///
/// The matching error builders stay in [`components::questioner`]; see the crate docs.
pub use components::questioner::{
    COMPONENT_QUESTIONER_CONFIG_ERROR, COMPONENT_QUESTIONER_EXECUTION_PROCESS_ERROR,
    COMPONENT_QUESTIONER_INPUT_INVALID, COMPONENT_QUESTIONER_INPUT_PARAM_ERROR,
    COMPONENT_QUESTIONER_INVOKE_CALL_FAILED, COMPONENT_QUESTIONER_RUNTIME_ERROR,
    COMPONENT_QUESTIONER_STATE_INIT_FAILED,
};
/// The `react_agent` `on_max_iterations` policy values.
pub use components::react_agent::{
    ON_MAX_ITERATIONS_ERROR, ON_MAX_ITERATIONS_FINAL_ANSWER, ON_MAX_ITERATIONS_RETURN_OBSERVATIONS,
};
pub use components::{
    AdvancedLoopComponent, AdvancedLoopConfig, AgentRunner, BranchComponent, BranchRouterAdapter,
    EndComponent, EndConfig, FieldInfo, HttpAdvancedOptionsConfig, HttpAuthConfig, HttpAuthIn,
    HttpAuthType, HttpBackoffType, HttpComponentConfig, HttpContentType, HttpHeadersConfig,
    HttpRateLimitConfig, HttpRateLimitUnit, HttpRequestBodyConfig, HttpRequestComponent,
    HttpRequestParamConfig, HttpResponseFormat, HttpResponseHandlingConfig, HttpResponseMode,
    HttpRetryConfig, IntentDetectionCompConfig, IntentDetectionComponent, IntentRouter,
    KnowledgeRetrievalComponent, LLMCompConfig, LLMComponent, LlmCompConfig, LlmComponent,
    LlmResponseFormat, LoopBound, LoopBreakComponent, LoopBreakConfig, LoopComponent, LoopConfig,
    LoopGroupComponent, LoopGroupConfig, LoopSetVariableComponent, LoopSetVariableConfig, LoopType,
    MemoryRecall, MemoryRecord, MemoryRetrievalComponent, MemoryWriteComponent, ModelReActRunner,
    QuestionerComponent, QuestionerConfig, ReActAgentCompConfig, ReActAgentComponent,
    ReActAgentConfig, StartComponent, SubWorkflowComponent, ToolComponent, LOOP_BREAK_MARKER,
    LOOP_SET_VARIABLES_MARKER, SUB_WORKFLOW_COMPONENT,
};
pub use condition::{Branch, BranchRouter, Condition};
pub use context::history::{history_corrupt_error, WORKFLOW_HISTORY_CORRUPT};
pub use context::{
    ComponentContext, HistoryCorruption, HistoryCorruptionReason, MessageHistory,
    SessionMessageHistory, MESSAGES_STATE_KEY,
};
pub use engine::{
    CompiledWorkflow, ConditionalSource, Edge, EdgeKind, Entry, NodeConfig, StreamDelivery,
    WorkflowDefinition, WorkflowRunOptions,
};
pub use error::{
    branch_execution_error, branch_param_invalid, compile_error, component_ability_invalid,
    component_execution_error, component_id_invalid, component_schema_invalid,
    component_tool_execution_error, component_tool_init_failed, component_tool_input_param_error,
    condition_edge_invalid, edge_invalid, end_param_invalid, expression_eval_error,
    expression_syntax_error, knowledge_retrieval_input_param_error,
    knowledge_retrieval_invoke_call_failed, llm_config_error, llm_execution_process_error,
    llm_init_failed, llm_invoke_call_failed, llm_response_config_invalid,
    llm_template_config_error, loop_execution_error, loop_param_invalid,
    loop_set_var_param_invalid, memory_retrieval_input_param_error,
    memory_retrieval_invoke_call_failed, memory_write_input_param_error,
    memory_write_invoke_call_failed, stream_edge_invalid, tool_execution_error,
    COMPONENT_BRANCH_EXECUTION_ERROR, COMPONENT_BRANCH_PARAM_INVALID, COMPONENT_END_PARAM_INVALID,
    COMPONENT_KNOWLEDGE_RETRIEVAL_INPUT_PARAM_ERROR,
    COMPONENT_KNOWLEDGE_RETRIEVAL_INVOKE_CALL_FAILED, COMPONENT_LLM_CONFIG_ERROR,
    COMPONENT_LLM_EXECUTION_PROCESS_ERROR, COMPONENT_LLM_INIT_FAILED,
    COMPONENT_LLM_INVOKE_CALL_FAILED, COMPONENT_LLM_RESPONSE_CONFIG_INVALID,
    COMPONENT_LLM_TEMPLATE_CONFIG_ERROR, COMPONENT_LOOP_EXECUTION_ERROR,
    COMPONENT_LOOP_PARAM_INVALID, COMPONENT_LOOP_SET_VAR_PARAM_INVALID,
    COMPONENT_MEMORY_RETRIEVAL_INPUT_PARAM_ERROR, COMPONENT_MEMORY_RETRIEVAL_INVOKE_CALL_FAILED,
    COMPONENT_MEMORY_WRITE_INPUT_PARAM_ERROR, COMPONENT_MEMORY_WRITE_INVOKE_CALL_FAILED,
    COMPONENT_TOOL_EXECUTION_ERROR, COMPONENT_TOOL_INIT_FAILED, COMPONENT_TOOL_INPUT_PARAM_ERROR,
    EXPRESSION_EVAL_ERROR, EXPRESSION_SYNTAX_ERROR, TOOL_EXECUTION_ERROR, WORKFLOW_COMPILE_ERROR,
    WORKFLOW_COMPONENT_ABILITY_INVALID, WORKFLOW_COMPONENT_EXECUTION_ERROR,
    WORKFLOW_COMPONENT_ID_INVALID, WORKFLOW_COMPONENT_SCHEMA_INVALID,
    WORKFLOW_CONDITION_EDGE_INVALID, WORKFLOW_EDGE_INVALID, WORKFLOW_STREAM_EDGE_INVALID,
};
pub use interrupt::{
    find_interrupt, interrupt_error, InteractionState, WorkflowInteraction, WorkflowInterrupt,
};
pub use output::{
    custom_chunk, frame_to_wire, output_chunk, trace_chunk, FrameCollector, FrameWriter,
    OutputFrame, WorkflowChunk, WorkflowChunkType, WorkflowExecutionState, WorkflowOutput,
    CUSTOM_FRAME_TYPE, END_NODE_STREAM, OUTPUT_FRAME_TYPE, TRACE_FRAME_TYPE, WORKFLOW_FINAL,
};
pub use reference::{
    extract_origin_key, is_ref_path, project, resolve_path, resolve_schema, split_path, PathSegment,
};
pub use registry::{ComponentFactory, ComponentRegistry};
pub use render::{render_svg, to_svg, DrawFormat, LayoutDirection, SvgOptions};
pub use single::{execute_single_component, execute_single_component_with, SingleComponentOptions};
pub use workflow::{ComponentOptions, Workflow};
