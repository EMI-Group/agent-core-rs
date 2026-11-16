//! Python-compatible workflow status codes and their message builders.
//!
//! The Python SDK raises a single `BaseError` carrying a `StatusCode`; the numeric code
//! and the rendered message template are part of the observable contract (callers match
//! on the `[<code>] <message>` prefix of `str(error)`). This module keeps both: one
//! `pub const …: u32` per Python status code and one builder function per code that
//! renders the template byte-for-byte and returns an
//! [`openjiuwen_common::error::Error::Workflow`].
//!
//! Codes are grouped exactly like `openjiuwen/core/common/exception/codes.py`:
//! workflow definition errors (100000–100099), workflow execution errors
//! (100100–100199), built-in component errors (101000–102019), and the shared tool
//! error (182012).
//!
//! ```
//! use openjiuwen_workflow::error::component_execution_error;
//!
//! let err = component_execution_error("llm", "invoke", "boom", "wf");
//! assert_eq!(
//!     err.message(),
//!     "[100054] component 'llm' execute 'invoke' error, reason='boom', workflow='wf'",
//! );
//! ```

use openjiuwen_common::error::Error;

/// The Python status code of [`component_id_invalid`].
pub const WORKFLOW_COMPONENT_ID_INVALID: u32 = 100_010;
/// The Python status code of [`component_ability_invalid`].
pub const WORKFLOW_COMPONENT_ABILITY_INVALID: u32 = 100_011;
/// The Python status code of [`edge_invalid`].
pub const WORKFLOW_EDGE_INVALID: u32 = 100_012;
/// The Python status code of [`condition_edge_invalid`].
pub const WORKFLOW_CONDITION_EDGE_INVALID: u32 = 100_013;
/// The Python status code of [`component_schema_invalid`].
pub const WORKFLOW_COMPONENT_SCHEMA_INVALID: u32 = 100_014;
/// The Python status code of [`stream_edge_invalid`].
pub const WORKFLOW_STREAM_EDGE_INVALID: u32 = 100_015;
/// The Python status code of [`macro@compile_error`].
pub const WORKFLOW_COMPILE_ERROR: u32 = 100_100;
/// The Python status code of [`component_execution_error`].
pub const WORKFLOW_COMPONENT_EXECUTION_ERROR: u32 = 100_054;
/// The Python status code of [`end_param_invalid`].
///
/// Python reuses the workflow component-id code here (101010–101019 was never
/// allocated for the end component); the constant stays separate so call sites read
/// clearly.
pub const COMPONENT_END_PARAM_INVALID: u32 = 100_010;
/// The Python status code of [`llm_template_config_error`].
pub const COMPONENT_LLM_TEMPLATE_CONFIG_ERROR: u32 = 101_000;
/// The Python status code of [`llm_response_config_invalid`].
pub const COMPONENT_LLM_RESPONSE_CONFIG_INVALID: u32 = 101_001;
/// The Python status code of [`llm_config_error`].
pub const COMPONENT_LLM_CONFIG_ERROR: u32 = 101_002;
/// The Python status code of [`llm_invoke_call_failed`].
pub const COMPONENT_LLM_INVOKE_CALL_FAILED: u32 = 101_003;
/// The Python status code of [`llm_execution_process_error`].
pub const COMPONENT_LLM_EXECUTION_PROCESS_ERROR: u32 = 101_004;
/// The Python status code of [`llm_init_failed`].
pub const COMPONENT_LLM_INIT_FAILED: u32 = 101_005;
/// The Python status code of [`branch_param_invalid`].
pub const COMPONENT_BRANCH_PARAM_INVALID: u32 = 101_020;
/// The Python status code of [`branch_execution_error`].
pub const COMPONENT_BRANCH_EXECUTION_ERROR: u32 = 101_021;
/// The Python status code of [`expression_syntax_error`].
pub const EXPRESSION_SYNTAX_ERROR: u32 = 101_024;
/// The Python status code of [`expression_eval_error`].
pub const EXPRESSION_EVAL_ERROR: u32 = 101_025;
/// The Python status code of [`loop_param_invalid`].
pub const COMPONENT_LOOP_PARAM_INVALID: u32 = 101_030;
/// The Python status code of [`loop_set_var_param_invalid`].
pub const COMPONENT_LOOP_SET_VAR_PARAM_INVALID: u32 = 101_031;
/// The Python status code of [`loop_execution_error`].
pub const COMPONENT_LOOP_EXECUTION_ERROR: u32 = 101_040;
/// The Python status code of [`knowledge_retrieval_invoke_call_failed`].
pub const COMPONENT_KNOWLEDGE_RETRIEVAL_INVOKE_CALL_FAILED: u32 = 101_100;
/// The Python status code of [`knowledge_retrieval_input_param_error`].
pub const COMPONENT_KNOWLEDGE_RETRIEVAL_INPUT_PARAM_ERROR: u32 = 101_102;
/// The Python status code of [`memory_write_input_param_error`].
pub const COMPONENT_MEMORY_WRITE_INPUT_PARAM_ERROR: u32 = 101_150;
/// The Python status code of [`memory_write_invoke_call_failed`].
pub const COMPONENT_MEMORY_WRITE_INVOKE_CALL_FAILED: u32 = 101_151;
/// The Python status code of [`memory_retrieval_input_param_error`].
pub const COMPONENT_MEMORY_RETRIEVAL_INPUT_PARAM_ERROR: u32 = 101_200;
/// The Python status code of [`memory_retrieval_invoke_call_failed`].
pub const COMPONENT_MEMORY_RETRIEVAL_INVOKE_CALL_FAILED: u32 = 101_201;
/// The Python status code of [`component_tool_execution_error`].
pub const COMPONENT_TOOL_EXECUTION_ERROR: u32 = 102_000;
/// The Python status code of [`component_tool_input_param_error`].
pub const COMPONENT_TOOL_INPUT_PARAM_ERROR: u32 = 102_001;
/// The Python status code of [`component_tool_init_failed`].
pub const COMPONENT_TOOL_INIT_FAILED: u32 = 102_002;
/// The Python status code of [`tool_execution_error`].
pub const TOOL_EXECUTION_ERROR: u32 = 182_012;

/// Renders the Python `"[<code>] <message>"` form as a workflow error.
fn coded(code: u32, message: impl Into<String>) -> Error {
    Error::workflow(format!("[{code}] {}", message.into()))
}

/// The component id is invalid (Python `WORKFLOW_COMPONENT_ID_INVALID`).
pub fn component_id_invalid(comp_id: &str, reason: &str, workflow: &str) -> Error {
    coded(
        WORKFLOW_COMPONENT_ID_INVALID,
        format!(
            "the component id is invalid for component '{comp_id}', reason='{reason}', \
             workflow='{workflow}'"
        ),
    )
}

/// The declared ability is invalid (Python `WORKFLOW_COMPONENT_ABILITY_INVALID`).
pub fn component_ability_invalid(
    comp_id: &str,
    ability: &str,
    reason: &str,
    workflow: &str,
) -> Error {
    coded(
        WORKFLOW_COMPONENT_ABILITY_INVALID,
        format!(
            "the ability is invalid for component '{comp_id}', ability={ability}, \
             reason='{reason}', workflow='{workflow}'"
        ),
    )
}

/// An edge is invalid (Python `WORKFLOW_EDGE_INVALID`).
pub fn edge_invalid(reason: &str, src_cmp_id: &str, target_cmp_id: &str, workflow: &str) -> Error {
    coded(
        WORKFLOW_EDGE_INVALID,
        format!(
            "edge is invalid, reason='{reason}', source='{src_cmp_id}', \
             target='{target_cmp_id}', workflow='{workflow}'"
        ),
    )
}

/// A conditional edge is invalid (Python `WORKFLOW_CONDITION_EDGE_INVALID`).
pub fn condition_edge_invalid(reason: &str, src_cmp_id: &str, workflow: &str) -> Error {
    coded(
        WORKFLOW_CONDITION_EDGE_INVALID,
        format!(
            "condition edge is invalid, reason='{reason}'. source='{src_cmp_id}', \
             workflow='{workflow}'"
        ),
    )
}

/// A component input/output schema is invalid (Python `WORKFLOW_COMPONENT_SCHEMA_INVALID`).
pub fn component_schema_invalid(comp_id: &str, reason: &str, workflow: &str) -> Error {
    coded(
        WORKFLOW_COMPONENT_SCHEMA_INVALID,
        format!(
            "component input/output schema is invalid for component '{comp_id}', \
             reason='{reason}', workflow='{workflow}'"
        ),
    )
}

/// A stream edge is invalid (Python `WORKFLOW_STREAM_EDGE_INVALID`).
pub fn stream_edge_invalid(
    reason: &str,
    src_cmp_id: &str,
    target_cmp_id: &str,
    workflow: &str,
) -> Error {
    coded(
        WORKFLOW_STREAM_EDGE_INVALID,
        format!(
            "stream edge is invalid, reason='{reason}', source='{src_cmp_id}', \
             target='{target_cmp_id}', workflow='{workflow}'"
        ),
    )
}

/// Workflow compilation failed (Python `WORKFLOW_COMPILE_ERROR`).
pub fn compile_error(reason: &str, workflow: &str) -> Error {
    coded(
        WORKFLOW_COMPILE_ERROR,
        format!("workflow compilation has error, error='{reason}', workflow={workflow}"),
    )
}

/// A component failed while executing (Python `WORKFLOW_COMPONENT_EXECUTION_ERROR`).
pub fn component_execution_error(comp: &str, ability: &str, reason: &str, workflow: &str) -> Error {
    coded(
        WORKFLOW_COMPONENT_EXECUTION_ERROR,
        format!("component '{comp}' execute '{ability}' error, reason='{reason}', workflow='{workflow}'"),
    )
}

/// The end component parameters are invalid (Python `COMPONENT_END_PARAM_INVALID`).
pub fn end_param_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_END_PARAM_INVALID,
        format!("component end params is invalid, error='{reason}'"),
    )
}

/// The LLM component template configuration is invalid
/// (Python `COMPONENT_LLM_TEMPLATE_CONFIG_ERROR`).
pub fn llm_template_config_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_LLM_TEMPLATE_CONFIG_ERROR,
        format!("component llm_template config error, reason: {error_msg}"),
    )
}

/// The LLM response configuration is invalid
/// (Python `COMPONENT_LLM_RESPONSE_CONFIG_INVALID`).
pub fn llm_response_config_invalid(error_msg: &str) -> Error {
    coded(
        COMPONENT_LLM_RESPONSE_CONFIG_INVALID,
        format!("component llm_response_config is invalid, reason: {error_msg}"),
    )
}

/// The LLM component configuration is invalid (Python `COMPONENT_LLM_CONFIG_ERROR`).
pub fn llm_config_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_LLM_CONFIG_ERROR,
        format!("component llm config error, reason: {error_msg}"),
    )
}

/// The model invocation failed (Python `COMPONENT_LLM_INVOKE_CALL_FAILED`).
pub fn llm_invoke_call_failed(error_msg: &str) -> Error {
    coded(
        COMPONENT_LLM_INVOKE_CALL_FAILED,
        format!("component llm_invoke call failed, reason: {error_msg}"),
    )
}

/// The LLM component failed while processing its output
/// (Python `COMPONENT_LLM_EXECUTION_PROCESS_ERROR`).
pub fn llm_execution_process_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_LLM_EXECUTION_PROCESS_ERROR,
        format!("component llm_execution process error, reason: {error_msg}"),
    )
}

/// The LLM component could not be initialized (Python `COMPONENT_LLM_INIT_FAILED`).
pub fn llm_init_failed(error_msg: &str) -> Error {
    coded(
        COMPONENT_LLM_INIT_FAILED,
        format!("component llm initialization failed, reason: {error_msg}"),
    )
}

/// Branch parameters are invalid (Python `COMPONENT_BRANCH_PARAM_INVALID`).
pub fn branch_param_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_BRANCH_PARAM_INVALID,
        format!("component branch params is invalid, error='{reason}'"),
    )
}

/// Branch routing failed (Python `COMPONENT_BRANCH_EXECUTION_ERROR`).
pub fn branch_execution_error(reason: &str) -> Error {
    coded(
        COMPONENT_BRANCH_EXECUTION_ERROR,
        format!("component branch execution error, error='{reason}'"),
    )
}

/// A condition expression failed to parse (Python `EXPRESSION_SYNTAX_ERROR`).
pub fn expression_syntax_error() -> Error {
    coded(
        EXPRESSION_SYNTAX_ERROR,
        "expression syntax error".to_string(),
    )
}

/// A condition expression failed to evaluate (Python `EXPRESSION_EVAL_ERROR`).
pub fn expression_eval_error(error_msg: &str) -> Error {
    coded(
        EXPRESSION_EVAL_ERROR,
        format!("expression evaluation error, reason: {error_msg}"),
    )
}

/// Loop parameters are invalid (Python `COMPONENT_LOOP_PARAM_INVALID`).
pub fn loop_param_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_LOOP_PARAM_INVALID,
        format!("component loop_group params is invalid, error='{reason}'"),
    )
}

/// The loop-variable assignment of a `loop_set_variable` component is invalid
/// (Python `COMPONENT_LOOP_SET_VAR_PARAM_INVALID`).
pub fn loop_set_var_param_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_LOOP_SET_VAR_PARAM_INVALID,
        format!("loop set_var params invalid, error='{reason}'"),
    )
}
/// A loop iteration failed (Python `COMPONENT_LOOP_EXECUTION_ERROR`).
pub fn loop_execution_error(reason: &str, comp: &str) -> Error {
    coded(
        COMPONENT_LOOP_EXECUTION_ERROR,
        format!("loop execution error, error='{reason}', comp='{comp}'"),
    )
}

/// The knowledge-retrieval invocation failed
/// (Python `COMPONENT_KNOWLEDGE_RETRIEVAL_INVOKE_CALL_FAILED`).
pub fn knowledge_retrieval_invoke_call_failed(error_msg: &str) -> Error {
    coded(
        COMPONENT_KNOWLEDGE_RETRIEVAL_INVOKE_CALL_FAILED,
        format!("component knowledge_retrieval invoke call failed, reason: {error_msg}"),
    )
}

/// The knowledge-retrieval input parameter is invalid
/// (Python `COMPONENT_KNOWLEDGE_RETRIEVAL_INPUT_PARAM_ERROR`).
pub fn knowledge_retrieval_input_param_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_KNOWLEDGE_RETRIEVAL_INPUT_PARAM_ERROR,
        format!("component knowledge_retrieval input parameter error, reason: {error_msg}"),
    )
}

/// The memory-write input parameter is invalid
/// (Python `COMPONENT_MEMORY_WRITE_INPUT_PARAM_ERROR`).
pub fn memory_write_input_param_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_MEMORY_WRITE_INPUT_PARAM_ERROR,
        format!("component memory_write input parameter error, reason: {error_msg}"),
    )
}

/// The memory-write invocation failed (Python `COMPONENT_MEMORY_WRITE_INVOKE_CALL_FAILED`).
pub fn memory_write_invoke_call_failed(error_msg: &str) -> Error {
    coded(
        COMPONENT_MEMORY_WRITE_INVOKE_CALL_FAILED,
        format!("component memory_write invoke call failed, reason: {error_msg}"),
    )
}

/// The memory-retrieval input parameter is invalid
/// (Python `COMPONENT_MEMORY_RETRIEVAL_INPUT_PARAM_ERROR`).
pub fn memory_retrieval_input_param_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_MEMORY_RETRIEVAL_INPUT_PARAM_ERROR,
        format!("component memory_retrieval input parameter error, reason: {error_msg}"),
    )
}

/// The memory-retrieval invocation failed
/// (Python `COMPONENT_MEMORY_RETRIEVAL_INVOKE_CALL_FAILED`).
pub fn memory_retrieval_invoke_call_failed(error_msg: &str) -> Error {
    coded(
        COMPONENT_MEMORY_RETRIEVAL_INVOKE_CALL_FAILED,
        format!("component memory_retrieval invoke call failed, reason: {error_msg}"),
    )
}

/// The tool component failed while executing its tool
/// (Python `COMPONENT_TOOL_EXECUTION_ERROR`).
pub fn component_tool_execution_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_TOOL_EXECUTION_ERROR,
        format!("component tool execution error, reason: {error_msg}"),
    )
}

/// The tool component input parameter is invalid
/// (Python `COMPONENT_TOOL_INPUT_PARAM_ERROR`).
pub fn component_tool_input_param_error(error_msg: &str) -> Error {
    coded(
        COMPONENT_TOOL_INPUT_PARAM_ERROR,
        format!("component tool_input parameter error, reason: {error_msg}"),
    )
}

/// The tool component could not be initialized (Python `COMPONENT_TOOL_INIT_FAILED`).
pub fn component_tool_init_failed(error_msg: &str) -> Error {
    coded(
        COMPONENT_TOOL_INIT_FAILED,
        format!("component tool initialization failed, reason: {error_msg}"),
    )
}

/// A tool invocation failed (Python `TOOL_EXECUTION_ERROR`).
pub fn tool_execution_error(card: &str, reason: &str) -> Error {
    coded(
        TOOL_EXECUTION_ERROR,
        format!("tool execution error, too card={card}, reason={reason}"),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn code_constants_match_python() {
        assert_eq!(WORKFLOW_COMPONENT_ID_INVALID, 100_010);
        assert_eq!(WORKFLOW_COMPONENT_ABILITY_INVALID, 100_011);
        assert_eq!(WORKFLOW_EDGE_INVALID, 100_012);
        assert_eq!(WORKFLOW_CONDITION_EDGE_INVALID, 100_013);
        assert_eq!(WORKFLOW_COMPONENT_SCHEMA_INVALID, 100_014);
        assert_eq!(WORKFLOW_STREAM_EDGE_INVALID, 100_015);
        assert_eq!(WORKFLOW_COMPONENT_EXECUTION_ERROR, 100_054);
        assert_eq!(WORKFLOW_COMPILE_ERROR, 100_100);
        assert_eq!(COMPONENT_END_PARAM_INVALID, 100_010);
        assert_eq!(COMPONENT_LLM_TEMPLATE_CONFIG_ERROR, 101_000);
        assert_eq!(COMPONENT_LLM_RESPONSE_CONFIG_INVALID, 101_001);
        assert_eq!(COMPONENT_LLM_CONFIG_ERROR, 101_002);
        assert_eq!(COMPONENT_LLM_INVOKE_CALL_FAILED, 101_003);
        assert_eq!(COMPONENT_LLM_EXECUTION_PROCESS_ERROR, 101_004);
        assert_eq!(COMPONENT_LLM_INIT_FAILED, 101_005);
        assert_eq!(COMPONENT_BRANCH_PARAM_INVALID, 101_020);
        assert_eq!(COMPONENT_BRANCH_EXECUTION_ERROR, 101_021);
        assert_eq!(EXPRESSION_SYNTAX_ERROR, 101_024);
        assert_eq!(EXPRESSION_EVAL_ERROR, 101_025);
        assert_eq!(COMPONENT_LOOP_PARAM_INVALID, 101_030);
        assert_eq!(COMPONENT_LOOP_SET_VAR_PARAM_INVALID, 101_031);
        assert_eq!(COMPONENT_LOOP_EXECUTION_ERROR, 101_040);
        assert_eq!(COMPONENT_KNOWLEDGE_RETRIEVAL_INVOKE_CALL_FAILED, 101_100);
        assert_eq!(COMPONENT_KNOWLEDGE_RETRIEVAL_INPUT_PARAM_ERROR, 101_102);
        assert_eq!(COMPONENT_MEMORY_WRITE_INPUT_PARAM_ERROR, 101_150);
        assert_eq!(COMPONENT_MEMORY_WRITE_INVOKE_CALL_FAILED, 101_151);
        assert_eq!(COMPONENT_MEMORY_RETRIEVAL_INPUT_PARAM_ERROR, 101_200);
        assert_eq!(COMPONENT_MEMORY_RETRIEVAL_INVOKE_CALL_FAILED, 101_201);
        assert_eq!(COMPONENT_TOOL_EXECUTION_ERROR, 102_000);
        assert_eq!(COMPONENT_TOOL_INPUT_PARAM_ERROR, 102_001);
        assert_eq!(COMPONENT_TOOL_INIT_FAILED, 102_002);
        assert_eq!(TOOL_EXECUTION_ERROR, 182_012);
    }

    #[test]
    fn builders_render_python_templates() {
        assert_eq!(
            component_id_invalid("c1", "is None or empty", "wf").message(),
            "[100010] the component id is invalid for component 'c1', \
             reason='is None or empty', workflow='wf'"
        );
        assert_eq!(
            component_ability_invalid("c1", "stream", "bad edge", "wf").message(),
            "[100011] the ability is invalid for component 'c1', ability=stream, \
             reason='bad edge', workflow='wf'"
        );
        assert_eq!(
            edge_invalid("cycle", "a", "b", "wf").message(),
            "[100012] edge is invalid, reason='cycle', source='a', target='b', workflow='wf'"
        );
        assert_eq!(
            condition_edge_invalid("no router", "a", "wf").message(),
            "[100013] condition edge is invalid, reason='no router'. source='a', workflow='wf'"
        );
        assert_eq!(
            compile_error("no start", "wf").message(),
            "[100100] workflow compilation has error, error='no start', workflow=wf"
        );
        assert_eq!(
            component_execution_error("llm", "invoke", "boom", "wf").message(),
            "[100054] component 'llm' execute 'invoke' error, reason='boom', workflow='wf'"
        );
        assert_eq!(
            end_param_invalid("bad template").message(),
            "[100010] component end params is invalid, error='bad template'"
        );
        assert_eq!(
            branch_execution_error("no branch matched").message(),
            "[101021] component branch execution error, error='no branch matched'"
        );
        assert_eq!(
            expression_syntax_error().message(),
            "[101024] expression syntax error"
        );
        assert_eq!(
            expression_eval_error("bad operands").message(),
            "[101025] expression evaluation error, reason: bad operands"
        );
        assert_eq!(
            loop_execution_error("bounded", "loop_1").message(),
            "[101040] loop execution error, error='bounded', comp='loop_1'"
        );
        assert_eq!(
            tool_execution_error("calculator", "divide by zero").message(),
            "[182012] tool execution error, too card=calculator, reason=divide by zero"
        );
    }

    #[test]
    fn builders_return_workflow_errors() {
        for error in [
            component_id_invalid("c", "r", "w"),
            edge_invalid("r", "a", "b", "w"),
            expression_syntax_error(),
            tool_execution_error("t", "r"),
        ] {
            assert_eq!(error.kind(), ErrorKind::Workflow);
        }
    }
}
