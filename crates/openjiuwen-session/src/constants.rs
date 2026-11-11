//! Environment key names and default identifiers shared by the session layer.
//!
//! Mirrors `openjiuwen/core/session/constants.py` (v0.1.20): the `*_ENV_KEY`
//! constants name the process environment variables read by
//! [`crate::config::SessionConfig::from_env`], while the remaining constants are
//! the default node/workflow ids and stream framing values.

/// Environment key for the workflow execution timeout (`WORKFLOW_EXECUTE_TIMEOUT`).
pub const ENV_WORKFLOW_EXECUTE_TIMEOUT: &str = "WORKFLOW_EXECUTE_TIMEOUT";
/// Environment key for the inter-frame stream timeout (`WORKFLOW_STREAM_FRAME_TIMEOUT`).
pub const ENV_WORKFLOW_STREAM_FRAME_TIMEOUT: &str = "WORKFLOW_STREAM_FRAME_TIMEOUT";
/// Environment key for the first-frame stream timeout (`WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT`).
pub const ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT: &str = "WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT";
/// Environment key for the component stream-call timeout (`COMP_STREAM_CALL_TIMEOUT`).
pub const ENV_COMP_STREAM_CALL_TIMEOUT: &str = "COMP_STREAM_CALL_TIMEOUT";
/// Environment key for the stream input generator timeout (`STREAM_INPUT_GEN_TIMEOUT`).
pub const ENV_STREAM_INPUT_GEN_TIMEOUT: &str = "STREAM_INPUT_GEN_TIMEOUT";
/// Environment key for the loop component iteration cap (`LOOP_NUMBER_MAX_LIMIT`).
pub const ENV_LOOP_NUMBER_MAX_LIMIT: &str = "LOOP_NUMBER_MAX_LIMIT";
/// Environment key for deleting workflow state after a run (`FORCE_DEL_WORKFLOW_STATE`).
pub const ENV_FORCE_DEL_WORKFLOW_STATE: &str = "FORCE_DEL_WORKFLOW_STATE";

/// Default node id (`"default"`), the root node of a workflow state tree.
pub const DEFAULT_NODE_ID: &str = "default";
/// Default workflow id (`"workflow"`), the key used for workflow-level staged state.
pub const DEFAULT_WORKFLOW_ID: &str = "workflow";
/// Marker stored on single-component direct execution, which bypasses the graph engine.
pub const SINGLE_COMPONENT_EXECUTION_KEY: &str = "_single_component_execution";

/// Default bounded capacity of a [`crate::stream::StreamHub`] broadcast channel.
pub const DEFAULT_STREAM_CAPACITY: usize = 1024;
/// Sentinel payload announcing that all streaming outputs finished.
pub const END_FRAME: &str = "all streaming outputs finish";
