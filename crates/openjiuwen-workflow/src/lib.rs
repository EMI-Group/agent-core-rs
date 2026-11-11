//! Workflow builder, components, and reference-based data flow.
//!
//! Maps to the Python package `core.workflow` in the openJiuwen reference implementation.
//! A workflow is a directed graph of components with one start and one end; the builder
//! records component configs and edges and compiling maps them onto `openjiuwen-graph`.
//! Data flow is textual: a component's `inputs_schema` is a JSON template whose pure
//! reference strings resolve against workflow state before the component runs.

pub mod ability;
pub mod card;
pub mod component;
pub mod components;
pub mod condition;
pub mod context;
pub mod error;
pub mod interrupt;
pub mod output;
pub mod reference;
pub mod registry;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-workflow"
}

pub use ability::ComponentAbility;
pub use card::{
    validate_component_id, CompIoConfig, ExceptionConfig, NodeSpec, WorkflowCard, WorkflowConfig,
    WorkflowSpec,
};
pub use component::{unsupported_ability, Component, ComponentCard};
pub use components::{
    BranchComponent, BranchRouterAdapter, EndComponent, EndConfig, KnowledgeRetrievalComponent,
    LlmCompConfig, LlmComponent, LlmResponseFormat, LoopComponent, LoopConfig, LoopType,
    MemoryRecall, MemoryRecord, MemoryRetrievalComponent, MemoryWriteComponent, StartComponent,
    ToolComponent,
};
pub use condition::{Branch, BranchRouter, Condition};
pub use context::ComponentContext;
pub use error::{
    branch_execution_error, branch_param_invalid, compile_error, component_ability_invalid,
    component_execution_error, component_id_invalid, component_schema_invalid,
    component_tool_execution_error, component_tool_init_failed, component_tool_input_param_error,
    condition_edge_invalid, edge_invalid, end_param_invalid, expression_eval_error,
    expression_syntax_error, knowledge_retrieval_input_param_error,
    knowledge_retrieval_invoke_call_failed, llm_config_error, llm_execution_process_error,
    llm_init_failed, llm_invoke_call_failed, llm_response_config_invalid,
    llm_template_config_error, loop_execution_error, loop_param_invalid,
    memory_retrieval_input_param_error, memory_retrieval_invoke_call_failed,
    memory_write_input_param_error, memory_write_invoke_call_failed, stream_edge_invalid,
    tool_execution_error, COMPONENT_BRANCH_EXECUTION_ERROR, COMPONENT_BRANCH_PARAM_INVALID,
    COMPONENT_END_PARAM_INVALID, COMPONENT_KNOWLEDGE_RETRIEVAL_INPUT_PARAM_ERROR,
    COMPONENT_KNOWLEDGE_RETRIEVAL_INVOKE_CALL_FAILED, COMPONENT_LLM_CONFIG_ERROR,
    COMPONENT_LLM_EXECUTION_PROCESS_ERROR, COMPONENT_LLM_INIT_FAILED,
    COMPONENT_LLM_INVOKE_CALL_FAILED, COMPONENT_LLM_RESPONSE_CONFIG_INVALID,
    COMPONENT_LLM_TEMPLATE_CONFIG_ERROR, COMPONENT_LOOP_EXECUTION_ERROR,
    COMPONENT_LOOP_PARAM_INVALID, COMPONENT_MEMORY_RETRIEVAL_INPUT_PARAM_ERROR,
    COMPONENT_MEMORY_RETRIEVAL_INVOKE_CALL_FAILED, COMPONENT_MEMORY_WRITE_INPUT_PARAM_ERROR,
    COMPONENT_MEMORY_WRITE_INVOKE_CALL_FAILED, COMPONENT_TOOL_EXECUTION_ERROR,
    COMPONENT_TOOL_INIT_FAILED, COMPONENT_TOOL_INPUT_PARAM_ERROR, EXPRESSION_EVAL_ERROR,
    EXPRESSION_SYNTAX_ERROR, TOOL_EXECUTION_ERROR, WORKFLOW_COMPILE_ERROR,
    WORKFLOW_COMPONENT_ABILITY_INVALID, WORKFLOW_COMPONENT_EXECUTION_ERROR,
    WORKFLOW_COMPONENT_ID_INVALID, WORKFLOW_COMPONENT_SCHEMA_INVALID,
    WORKFLOW_CONDITION_EDGE_INVALID, WORKFLOW_EDGE_INVALID, WORKFLOW_STREAM_EDGE_INVALID,
};
pub use interrupt::{
    find_interrupt, interrupt_error, InteractionState, WorkflowInteraction, WorkflowInterrupt,
};
pub use output::{
    custom_chunk, output_chunk, trace_chunk, FrameWriter, OutputFrame, WorkflowChunk,
    WorkflowExecutionState, WorkflowOutput, END_NODE_STREAM, OUTPUT_FRAME_TYPE, WORKFLOW_FINAL,
};
pub use reference::{
    extract_origin_key, is_ref_path, project, resolve_path, resolve_schema, split_path, PathSegment,
};
pub use registry::{ComponentFactory, ComponentRegistry};
