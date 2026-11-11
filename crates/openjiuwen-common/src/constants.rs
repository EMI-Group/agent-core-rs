//! Shared protocol constants and controller/task type enums.
//!
//! Port of Python `openjiuwen/core/common/constants/constant.py` and `enums.py`.
//! These names cross process boundaries: workflow `${node.field}` references,
//! interaction frames, and serialized card/enum values must stay byte-identical.

use serde::{Deserialize, Serialize};

/// Interaction frame key raised by interactive nodes (`"__interaction__"`).
pub const INTERACTION: &str = "__interaction__";

/// Dynamic interaction input key (`"__interactive_input__"`).
pub const INTERACTIVE_INPUT: &str = "__interactive_input__";

/// Workflow inputs key (`"inputs"`).
pub const INPUTS_KEY: &str = "inputs";

/// Workflow config key (`"config"`).
pub const CONFIG_KEY: &str = "config";

/// Sentinel emitted when all streaming outputs have finished.
pub const END_FRAME: &str = "all streaming outputs finish";

/// Sentinel marking the end of a node's stream.
pub const END_NODE_STREAM: &str = "end node stream";

/// System loop identifier key (`"__sys_loop_id"`).
pub const LOOP_ID: &str = "__sys_loop_id";

/// Index of a streaming chunk.
pub const INDEX: &str = "index";

/// Index at which streaming finishes.
pub const FINISH_INDEX: &str = "finish_index";

/// IR user fields key (`"userFields"`, Python parity — camelCase).
pub const USER_FIELDS: &str = "userFields";

/// IR system fields key (`"systemFields"`, Python parity — camelCase).
pub const SYSTEM_FIELDS: &str = "systemFields";

/// The user query key (`"query"`).
pub const QUERY: &str = "query";

/// Maximum allowed collection size (Python `MAX_COLLECTION_SIZE`).
pub const MAX_COLLECTION_SIZE: usize = 100_000;

/// Maximum allowed expression length (Python `MAX_EXPRESSION_LENGTH`).
pub const MAX_EXPRESSION_LENGTH: usize = 5_000;

/// Maximum allowed expression AST depth (Python `MAX_AST_DEPTH`).
pub const MAX_AST_DEPTH: usize = 50;

/// Maximum allowed nested loop depth; `1` means no nesting (Python `NESTED_LOOP_DEPTH`).
pub const NESTED_LOOP_DEPTH: usize = 1;

/// Shared redaction placeholder used by `logging` and `security`.
pub const REDACTED: &str = "<REDACTED>";

/// Controller selected by an agent card (Python `ControllerType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[must_use]
pub enum ControllerType {
    /// ReAct-based controller (`"react"`).
    React,
    /// Workflow-based controller (`"workflow"`).
    Workflow,
    /// No controller selected (`"undefined"`), the default.
    #[default]
    Undefined,
}

impl ControllerType {
    /// The Python enum value (`"react"`, `"workflow"`, `"undefined"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::React => "react",
            Self::Workflow => "workflow",
            Self::Undefined => "undefined",
        }
    }
}

impl std::fmt::Display for ControllerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Kind of task a workflow node runs (Python `TaskType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[must_use]
pub enum TaskType {
    /// Plugin task (`"plugin"`).
    Plugin,
    /// Workflow task (`"workflow"`).
    Workflow,
    /// MCP tool task (`"mcp"`).
    Mcp,
    /// Unknown task type (`"undefined"`), the default.
    #[default]
    Undefined,
}

impl TaskType {
    /// The Python enum value (`"plugin"`, `"workflow"`, `"mcp"`, `"undefined"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plugin => "plugin",
            Self::Workflow => "workflow",
            Self::Mcp => "mcp",
            Self::Undefined => "undefined",
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn protocol_constants_match_python() {
        assert_eq!(INTERACTION, "__interaction__");
        assert_eq!(INTERACTIVE_INPUT, "__interactive_input__");
        assert_eq!(INPUTS_KEY, "inputs");
        assert_eq!(CONFIG_KEY, "config");
        assert_eq!(END_FRAME, "all streaming outputs finish");
        assert_eq!(END_NODE_STREAM, "end node stream");
        assert_eq!(LOOP_ID, "__sys_loop_id");
        assert_eq!(INDEX, "index");
        assert_eq!(FINISH_INDEX, "finish_index");
        assert_eq!(USER_FIELDS, "userFields");
        assert_eq!(SYSTEM_FIELDS, "systemFields");
        assert_eq!(QUERY, "query");
        assert_eq!(REDACTED, "<REDACTED>");
    }

    #[test]
    fn safety_limits_match_python() {
        assert_eq!(MAX_COLLECTION_SIZE, 100_000);
        assert_eq!(MAX_EXPRESSION_LENGTH, 5_000);
        assert_eq!(MAX_AST_DEPTH, 50);
        assert_eq!(NESTED_LOOP_DEPTH, 1);
    }

    #[test]
    fn controller_type_serde_round_trip() {
        for (value, expected) in [
            (ControllerType::React, "react"),
            (ControllerType::Workflow, "workflow"),
            (ControllerType::Undefined, "undefined"),
        ] {
            let json = serde_json::to_string(&value).unwrap();
            assert_eq!(json, format!("\"{expected}\""));
            assert_eq!(value.as_str(), expected);
            assert_eq!(value.to_string(), expected);
            assert_eq!(
                serde_json::from_str::<ControllerType>(&json).unwrap(),
                value
            );
        }

        assert_eq!(ControllerType::default(), ControllerType::Undefined);
        assert_eq!(
            serde_json::from_str::<ControllerType>("\"react\"").unwrap(),
            ControllerType::React
        );
        assert!(serde_json::from_str::<ControllerType>("\"ReActController\"").is_err());
    }

    #[test]
    fn task_type_serde_round_trip() {
        for (value, expected) in [
            (TaskType::Plugin, "plugin"),
            (TaskType::Workflow, "workflow"),
            (TaskType::Mcp, "mcp"),
            (TaskType::Undefined, "undefined"),
        ] {
            let json = serde_json::to_string(&value).unwrap();
            assert_eq!(json, format!("\"{expected}\""));
            assert_eq!(value.as_str(), expected);
            assert_eq!(value.to_string(), expected);
            assert_eq!(serde_json::from_str::<TaskType>(&json).unwrap(), value);
        }

        assert_eq!(TaskType::default(), TaskType::Undefined);
        assert_eq!(
            serde_json::from_str::<TaskType>("\"mcp\"").unwrap(),
            TaskType::Mcp
        );
    }
}
