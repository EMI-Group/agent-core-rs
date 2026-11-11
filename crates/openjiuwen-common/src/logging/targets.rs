//! Logger names, expressed as [`tracing`] targets.
//!
//! Port of the Python module logger names (`logging.getLogger(__name__)`) used
//! across `openjiuwen`. In the Rust SDK the same dotted names are passed to the
//! `tracing` macros as an explicit `target`, so filtering by module still works:
//!
//! ```
//! use openjiuwen_common::logging::{info, targets};
//!
//! info!(target: targets::LLM, "request completed");
//! ```
//!
//! Keeping the Python names verbatim means existing log filters such as
//! `openjiuwen.agent=debug` behave the same way they did in Python.

/// Shared/common infrastructure logger (`openjiuwen.common`).
pub const COMMON: &str = "openjiuwen.common";
/// Public interface layer (`openjiuwen.interface`).
pub const INTERFACE: &str = "openjiuwen.interface";
/// Prompt builder (`openjiuwen.prompt_builder`).
pub const PROMPT_BUILDER: &str = "openjiuwen.prompt_builder";
/// Performance measurement (`openjiuwen.performance`).
pub const PERFORMANCE: &str = "openjiuwen.performance";
/// Single-agent runtime (`openjiuwen.agent`).
pub const AGENT: &str = "openjiuwen.agent";
/// Multi-agent teams (`openjiuwen.multi_agent`).
pub const MULTI_AGENT: &str = "openjiuwen.multi_agent";
/// Workflow engine (`openjiuwen.workflow`).
pub const WORKFLOW: &str = "openjiuwen.workflow";
/// Session management (`openjiuwen.session`).
pub const SESSION: &str = "openjiuwen.session";
/// Controller/runner control plane (`openjiuwen.controller`).
pub const CONTROLLER: &str = "openjiuwen.controller";
/// Runner and task dispatch (`openjiuwen.runner`).
pub const RUNNER: &str = "openjiuwen.runner";
/// System operations: filesystem, shell, code (`openjiuwen.sys_operation`).
pub const SYS_OPERATION: &str = "openjiuwen.sys_operation";
/// LLM clients (`openjiuwen.llm`).
pub const LLM: &str = "openjiuwen.llm";
/// Tool and MCP invocation (`openjiuwen.tool`).
pub const TOOL: &str = "openjiuwen.tool";
/// Prompt templates (`openjiuwen.prompt`).
pub const PROMPT: &str = "openjiuwen.prompt";
/// Storage backends (`openjiuwen.store`).
pub const STORE: &str = "openjiuwen.store";
/// Long-term and coding memory (`openjiuwen.memory`).
pub const MEMORY: &str = "openjiuwen.memory";
/// Retrieval, embeddings, vector stores (`openjiuwen.retrieval`).
pub const RETRIEVAL: &str = "openjiuwen.retrieval";
/// Context engine: windows, compression, offload (`openjiuwen.context_engine`).
pub const CONTEXT_ENGINE: &str = "openjiuwen.context_engine";
/// Graph execution engine (`openjiuwen.graph`).
pub const GRAPH: &str = "openjiuwen.graph";
/// Workflow operators (`openjiuwen.operator`).
pub const OPERATOR: &str = "openjiuwen.operator";
/// Model Context Protocol client (`openjiuwen.mcp`).
pub const MCP: &str = "openjiuwen.mcp";
/// Agent teams (`openjiuwen.team`).
pub const TEAM: &str = "openjiuwen.team";
/// Server/hosting layer (`openjiuwen.server`).
pub const SERVER: &str = "openjiuwen.server";
/// Structured event emission (`openjiuwen.events`).
pub const EVENTS: &str = "openjiuwen.events";

#[cfg(test)]
mod tests {
    use super::*;
    // port of openjiuwen/core/common/logging/logger.py (logger name parity)
    #[test]
    fn every_target_uses_the_openjiuwen_namespace() {
        let targets = [
            COMMON,
            INTERFACE,
            PROMPT_BUILDER,
            PERFORMANCE,
            AGENT,
            MULTI_AGENT,
            WORKFLOW,
            SESSION,
            CONTROLLER,
            RUNNER,
            SYS_OPERATION,
            LLM,
            TOOL,
            PROMPT,
            STORE,
            MEMORY,
            RETRIEVAL,
            CONTEXT_ENGINE,
            GRAPH,
            OPERATOR,
            MCP,
            TEAM,
            SERVER,
            EVENTS,
        ];
        for target in targets {
            assert!(
                target.starts_with("openjiuwen."),
                "unexpected target {target}"
            );
        }
    }
    #[test]
    fn target_names_match_python_loggers() {
        assert_eq!(COMMON, "openjiuwen.common");
        assert_eq!(MULTI_AGENT, "openjiuwen.multi_agent");
        assert_eq!(SYS_OPERATION, "openjiuwen.sys_operation");
        assert_eq!(CONTEXT_ENGINE, "openjiuwen.context_engine");
        assert_eq!(EVENTS, "openjiuwen.events");
    }
}
