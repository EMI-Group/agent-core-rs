//! Sub-agent configuration, the task-loop no-progress guard, and the
//! `on_max_iterations` policy re-export.
//!
//! Ports `SubAgentConfig` and `TaskLoopNoProgressGuardConfig` from
//! `openjiuwen/harness/schema/config.py`. Both are serde data only: live handles
//! (model, tools, rails, `SysOperation`) are supplied by the builder layer, so
//! `tools` holds tool names and `factory_kwargs` holds plain JSON.

use std::collections::BTreeMap;

use openjiuwen_agent::AgentCard;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::mode::PromptMode;

/// Exhaustion policy of the inner `ReAct` loop when it reaches its iteration cap.
///
/// Re-export of [`openjiuwen_agent::MaxIterationsPolicy`] under the Python-facing
/// name (`on_max_iterations`), so mapping a `DeepAgentConfig` onto a
/// `ReActAgentConfig` needs no conversion. Wire values are `error` (default),
/// `final_answer` and `return_observations`.
pub use openjiuwen_agent::MaxIterationsPolicy as OnMaxIterations;

/// Guard against repeated empty or near-empty no-tool answers in task-loop mode.
///
/// Python parity: `TaskLoopNoProgressGuardConfig`; every field defaults to the
/// Python value (`true`, `3`, `20`). The guard belongs to the task loop, not to
/// this type: this struct carries only the thresholds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskLoopNoProgressGuardConfig {
    /// Whether the guard runs at all.
    pub enabled: bool,
    /// Number of consecutive short answers that stops the loop.
    pub max_consecutive_empty_answers: usize,
    /// Answer length (in characters, after trimming) below which a no-tool
    /// answer counts as "no progress".
    pub min_answer_chars: usize,
}

impl Default for TaskLoopNoProgressGuardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_consecutive_empty_answers: 3,
            min_answer_chars: 20,
        }
    }
}

/// Configuration of one `DeepAgent` sub-agent.
///
/// Python parity: `SubAgentConfig`. The Rust port stores tool *names* rather than
/// `Tool`/`ToolCard` objects and keeps `factory_kwargs` as plain JSON; the live
/// objects are bound by [`crate::subagents`]. Fields that Python leaves `None`
/// (language, prompt mode, iteration cap) inherit the parent agent's setting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
// Python parity: the booleans are independent switches, not a state bitfield.
#[allow(clippy::struct_excessive_bools)]
pub struct SubAgentConfig {
    /// Identity card of the sub-agent.
    pub agent_card: AgentCard,
    /// System prompt of the sub-agent; `None` inherits the generated prompt.
    pub system_prompt: Option<String>,
    /// Names of the tools mounted on the sub-agent.
    pub tools: Vec<String>,
    /// Skill definitions mounted on the sub-agent.
    pub skills: Vec<String>,
    /// Maximum inner `ReAct` iterations; `None` inherits the parent setting.
    pub max_iterations: Option<u32>,
    /// Run the outer task loop inside the sub-agent.
    pub enable_task_loop: bool,
    /// Language of the sub-agent's prompts; `None` inherits the parent setting.
    pub language: Option<String>,
    /// Prompt verbosity of the sub-agent; `None` inherits the parent setting.
    pub prompt_mode: Option<PromptMode>,
    /// Factory name used to build the sub-agent instead of the default factory.
    pub factory_name: Option<String>,
    /// JSON keyword arguments passed to that factory.
    pub factory_kwargs: BTreeMap<String, Value>,
    /// Allow the sub-agent to switch into plan mode.
    pub enable_plan_mode: bool,
    /// Execute the sub-agent's tool calls of one turn in parallel when `true`.
    pub parallel_tool_calls: bool,
    /// Restrict the sub-agent's file operations to its workspace/project root.
    pub restrict_to_work_dir: bool,
    /// Human-readable name shown in prompts and transcripts.
    pub display_name: Option<String>,
    /// Role description given to the sub-agent in the parent's prompt.
    pub role: Option<String>,
    /// Sub-agent type used for preset lookup and runtime bookkeeping.
    pub subagent_type: Option<String>,
}

impl Default for SubAgentConfig {
    fn default() -> Self {
        Self {
            agent_card: AgentCard::default(),
            system_prompt: None,
            tools: Vec::new(),
            skills: Vec::new(),
            max_iterations: None,
            enable_task_loop: false,
            language: None,
            prompt_mode: None,
            factory_name: None,
            factory_kwargs: BTreeMap::new(),
            enable_plan_mode: false,
            parallel_tool_calls: true,
            restrict_to_work_dir: true,
            display_name: None,
            role: None,
            subagent_type: None,
        }
    }
}

impl SubAgentConfig {
    /// Creates a sub-agent configuration for `agent_card` with every other field
    /// at its default.
    #[must_use]
    pub fn new(agent_card: AgentCard) -> Self {
        Self {
            agent_card,
            ..Self::default()
        }
    }

    /// Overrides the identity card of the sub-agent.
    #[must_use]
    pub fn with_agent_card(mut self, agent_card: AgentCard) -> Self {
        self.agent_card = agent_card;
        self
    }

    /// Overrides the sub-agent's system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: Option<String>) -> Self {
        self.system_prompt = system_prompt;
        self
    }

    /// Overrides the names of the tools mounted on the sub-agent.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Overrides the skill definitions mounted on the sub-agent.
    #[must_use]
    pub fn with_skills(mut self, skills: Vec<String>) -> Self {
        self.skills = skills;
        self
    }

    /// Overrides the maximum number of inner `ReAct` iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: Option<u32>) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Overrides whether the sub-agent runs the outer task loop.
    #[must_use]
    pub fn with_enable_task_loop(mut self, enable_task_loop: bool) -> Self {
        self.enable_task_loop = enable_task_loop;
        self
    }

    /// Overrides the language of the sub-agent's prompts.
    #[must_use]
    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language;
        self
    }

    /// Overrides the prompt verbosity of the sub-agent.
    #[must_use]
    pub fn with_prompt_mode(mut self, prompt_mode: Option<PromptMode>) -> Self {
        self.prompt_mode = prompt_mode;
        self
    }

    /// Overrides the factory name used to build the sub-agent.
    #[must_use]
    pub fn with_factory_name(mut self, factory_name: Option<String>) -> Self {
        self.factory_name = factory_name;
        self
    }

    /// Overrides the keyword arguments passed to that factory.
    #[must_use]
    pub fn with_factory_kwargs(mut self, factory_kwargs: BTreeMap<String, Value>) -> Self {
        self.factory_kwargs = factory_kwargs;
        self
    }

    /// Overrides whether the sub-agent may switch into plan mode.
    #[must_use]
    pub fn with_enable_plan_mode(mut self, enable_plan_mode: bool) -> Self {
        self.enable_plan_mode = enable_plan_mode;
        self
    }

    /// Overrides whether the sub-agent's tool calls run in parallel.
    #[must_use]
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    /// Overrides whether the sub-agent's file operations stay inside its work dir.
    #[must_use]
    pub fn with_restrict_to_work_dir(mut self, restrict_to_work_dir: bool) -> Self {
        self.restrict_to_work_dir = restrict_to_work_dir;
        self
    }

    /// Overrides the human-readable name of the sub-agent.
    #[must_use]
    pub fn with_display_name(mut self, display_name: Option<String>) -> Self {
        self.display_name = display_name;
        self
    }

    /// Overrides the role description given to the sub-agent.
    #[must_use]
    pub fn with_role(mut self, role: Option<String>) -> Self {
        self.role = role;
        self
    }

    /// Overrides the sub-agent type used for preset lookup.
    #[must_use]
    pub fn with_subagent_type(mut self, subagent_type: Option<String>) -> Self {
        self.subagent_type = subagent_type;
        self
    }
}
