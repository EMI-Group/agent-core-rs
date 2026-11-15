//! The `DeepAgent` configuration and its validation.
//!
//! Ports `DeepAgentConfig`, `resolve_inner_react_max_iterations` and
//! `DeepAgentConfig.__post_init__` from `openjiuwen/harness/schema/config.py`.
//!
//! [`DeepAgentConfig`] is serde data only. Python's live handles (`Model`, rails,
//! `SysOperation`, `permission_host`, tool and MCP object lists) are supplied
//! through `DeepAgentBuilder`, so the model identity is stored as
//! `model_name`/`model_provider`/`api_key`/`api_base` plus optional
//! [`ModelRequestConfig`]/[`ModelClientConfig`].

use std::collections::BTreeMap;
use std::path::PathBuf;

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::llm::{ModelClientConfig, ModelRequestConfig};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::permission::PermissionsSection;
use crate::schema::mode::{AgentMode, PromptMode};
use crate::workspace::Workspace;

use super::{OnMaxIterations, SubAgentConfig, TaskLoopNoProgressGuardConfig};

/// Python `__post_init__` message: the ceiling needs single-round mode.
const ERR_CEILING_NEEDS_SINGLE_ROUND: &str =
    "max_iterations_hard_limit requires single-round mode (enable_task_loop=False)";
/// Python `__post_init__` message: the ceiling itself must be positive.
const ERR_CEILING_NOT_POSITIVE: &str = "max_iterations_hard_limit must be a positive integer";
/// Python `__post_init__` message: `max_iterations` must fit under the ceiling.
const ERR_ITERATIONS_OUTSIDE_CEILING: &str =
    "max_iterations must be positive and within max_iterations_hard_limit";
/// Shared message for a non-positive `max_iterations`.
const ERR_ITERATIONS_NOT_POSITIVE: &str = "max_iterations must be a positive integer";

/// Runtime configuration of the `DeepAgent`.
///
/// Python parity: `DeepAgentConfig`, ported as serde data only. Unknown JSON
/// fields are tolerated and dropped, every field has its Python default, and
/// `Debug` never prints the API key ([`SecretString`] redacts it).
///
/// # Examples
///
/// ```
/// use openjiuwen_harness::schema::DeepAgentConfig;
///
/// let config = DeepAgentConfig::default().with_language("en".to_string());
/// assert_eq!(config.language, "en");
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
// Python parity: the booleans are independent switches, not a state bitfield.
#[allow(clippy::struct_excessive_bools)]
pub struct DeepAgentConfig {
    /// Provider-facing model name of the inner `ReAct` agent.
    pub model_name: Option<String>,
    /// Model provider name (`openai`, `anthropic`, …); `None` resolves from
    /// [`Self::model_client_config`] or the environment.
    pub model_provider: Option<String>,
    /// API key of the model provider; [`SecretString`] keeps it out of logs.
    #[schemars(with = "Option<String>")]
    pub api_key: Option<SecretString>,
    /// Base URL of the model provider; `None` uses the provider default.
    pub api_base: Option<String>,
    /// Extra HTTP headers sent with every model request.
    pub custom_headers: BTreeMap<String, String>,
    /// Request-level model settings (`temperature`, `max_tokens`, …).
    #[schemars(with = "Option<serde_json::Value>")]
    pub model_config: Option<ModelRequestConfig>,
    /// Client-level model settings (timeouts, retries, transport headers).
    #[schemars(with = "Option<serde_json::Value>")]
    pub model_client_config: Option<ModelClientConfig>,
    /// Agent identity card; its id is the persistence identity of the session
    /// state, so it must stay stable across restarts.
    pub card: AgentCard,
    /// System prompt injected into the inner `ReAct` agent's prompt template.
    pub system_prompt: Option<String>,
    /// Maximum inner `ReAct` iterations per invoke; `None` means unbounded.
    pub max_iterations: Option<u32>,
    /// Optional positive ceiling for single-round execution; cannot be combined
    /// with the outer task loop.
    pub max_iterations_hard_limit: Option<u32>,
    /// What the inner loop does when it exhausts [`Self::max_iterations`].
    #[schemars(with = "String")]
    pub on_max_iterations: OnMaxIterations,
    /// Run the outer task loop instead of a single `ReAct` round.
    pub enable_task_loop: bool,
    /// Register session tools for async sub-agent spawning instead of the
    /// synchronous task tools.
    pub enable_async_subagent: bool,
    /// Expose the persistent sub-agent runtime tools (`subagent_spawn`,
    /// `subagent_wait`, `subagent_list`); takes precedence over
    /// [`Self::enable_async_subagent`].
    pub enable_subagent_runtime: bool,
    /// Add a general-purpose agent to the sub-agent set.
    pub add_general_purpose_agent: bool,
    /// Seconds to wait for one task-loop iteration to complete; `600.0` by
    /// default, also deserialized from the Python key `completion_timeout`.
    #[serde(alias = "completion_timeout")]
    pub completion_timeout_secs: f64,
    /// Workspace root for this agent's own artifacts (memory, skills view,
    /// produced files) — not the shell working directory, see [`Self::cwd`].
    pub workspace: Workspace,
    /// Directory shell commands run in and relative paths resolve against;
    /// `None` means the workspace root.
    pub cwd: Option<PathBuf>,
    /// Project identity anchor (repository root) used for access-boundary
    /// checks, not for path resolution; `None` means [`Self::cwd`].
    #[schemars(with = "Option<String>")]
    pub project_root: Option<PathBuf>,
    /// Skill definitions; a single JSON string deserializes as a one-element
    /// list.
    #[serde(deserialize_with = "deserialize_skills")]
    pub skills: Vec<String>,
    /// Discover skills from the workspace in addition to the configured list.
    pub enable_skill_discovery: bool,
    /// Language of the generated prompts and workspace schema; `"cn"` by
    /// default, `"en"` selects English.
    pub language: String,
    /// Verbosity of the generated system prompt.
    pub prompt_mode: PromptMode,
    /// Sub-agent specifications mounted on this agent.
    pub subagents: Vec<SubAgentConfig>,
    /// Execution mode the agent starts in.
    pub default_mode: AgentMode,
    /// Allow the agent to switch into plan mode.
    pub enable_plan_mode: bool,
    /// Execute the tool calls of one model turn in parallel when `true`.
    pub parallel_tool_calls: bool,
    /// Restrict file operations (and sub-agents) to the workspace/project root.
    pub restrict_to_work_dir: bool,
    /// Mount the bounded retry rail for retryable tool-call failures.
    pub enable_tool_resilience_rail: bool,
    /// Enable todo/task-planning tools and the plan prompt section.
    pub enable_task_planning: bool,
    /// Mount the security rail (input/output guardrails).
    pub enable_security_rail: bool,
    /// Tool permission policy; when enabled the permission interrupt rail is
    /// mounted automatically.
    pub permissions: Option<PermissionsSection>,
    /// Guard against repeated empty or near-empty no-tool answers in task-loop
    /// mode.
    pub task_loop_no_progress_guard: TaskLoopNoProgressGuardConfig,
    /// Maximum number of skills included in the prompt; whole low-ranked skills
    /// are dropped beyond it.
    pub skill_budget_max_skills: Option<usize>,
    /// Maximum total characters of the skill prompt section.
    pub skill_budget_max_total_chars: Option<usize>,
    /// Directory for bounded per-invocation logs.
    #[schemars(with = "Option<String>")]
    pub invocation_log_dir: Option<PathBuf>,
    /// Journal every turn as a resumable persistent-agent execution.
    ///
    /// Additive Rust switch (Python records executions through the persistent-agent host instead):
    /// when `true`, a turn appends its goal, answer and terminal status to an
    /// [`ExecutionLog`](crate::persistent_agent::ExecutionLog) under the workspace's
    /// `persistent_agent/logs` node and rewrites the persisted session record, so
    /// [`resume`](crate::persistent_agent::resume) can rebuild the window later. Recording
    /// failures are logged and never fail the turn.
    pub persistent_agent: bool,
}

impl Default for DeepAgentConfig {
    fn default() -> Self {
        Self {
            model_name: None,
            model_provider: None,
            api_key: None,
            api_base: None,
            custom_headers: BTreeMap::new(),
            model_config: None,
            model_client_config: None,
            card: AgentCard::default(),
            system_prompt: None,
            max_iterations: None,
            max_iterations_hard_limit: None,
            on_max_iterations: OnMaxIterations::default(),
            enable_task_loop: false,
            enable_async_subagent: false,
            enable_subagent_runtime: false,
            add_general_purpose_agent: false,
            completion_timeout_secs: 600.0,
            workspace: Workspace::default(),
            cwd: None,
            project_root: None,
            skills: Vec::new(),
            enable_skill_discovery: false,
            language: "cn".to_string(),
            prompt_mode: PromptMode::default(),
            subagents: Vec::new(),
            default_mode: AgentMode::default(),
            enable_plan_mode: false,
            parallel_tool_calls: true,
            restrict_to_work_dir: true,
            enable_tool_resilience_rail: true,
            enable_task_planning: false,
            enable_security_rail: true,
            permissions: None,
            task_loop_no_progress_guard: TaskLoopNoProgressGuardConfig::default(),
            skill_budget_max_skills: None,
            skill_budget_max_total_chars: None,
            invocation_log_dir: None,
            persistent_agent: false,
        }
    }
}

impl DeepAgentConfig {
    /// Validates the iteration configuration.
    ///
    /// The messages are the Python `__post_init__` texts, so a rejected
    /// configuration reports exactly what the Python SDK reports.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a hard limit is combined with the task
    /// loop, when the hard limit is not positive, when `max_iterations` is
    /// missing/non-positive/above the ceiling, or when a standalone
    /// `max_iterations` is `0`. That last check is stricter than Python's
    /// `__post_init__`, which only validates it under a ceiling.
    pub fn validate(&self) -> Result<()> {
        if let Some(ceiling) = self.max_iterations_hard_limit {
            if self.enable_task_loop {
                return Err(Error::config(ERR_CEILING_NEEDS_SINGLE_ROUND));
            }
            if ceiling == 0 {
                return Err(Error::config(ERR_CEILING_NOT_POSITIVE));
            }
            if !matches!(self.max_iterations, Some(iterations) if iterations >= 1 && iterations <= ceiling)
            {
                return Err(Error::config(ERR_ITERATIONS_OUTSIDE_CEILING));
            }
        } else if self.max_iterations == Some(0) {
            return Err(Error::config(ERR_ITERATIONS_NOT_POSITIVE));
        }
        Ok(())
    }

    /// Overrides the model name.
    #[must_use]
    pub fn with_model_name(mut self, model_name: Option<String>) -> Self {
        self.model_name = model_name;
        self
    }

    /// Overrides the model provider.
    #[must_use]
    pub fn with_model_provider(mut self, model_provider: Option<String>) -> Self {
        self.model_provider = model_provider;
        self
    }

    /// Overrides the model API key.
    #[must_use]
    pub fn with_api_key(mut self, api_key: Option<SecretString>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Overrides the model base URL.
    #[must_use]
    pub fn with_api_base(mut self, api_base: Option<String>) -> Self {
        self.api_base = api_base;
        self
    }

    /// Overrides the extra request headers.
    #[must_use]
    pub fn with_custom_headers(mut self, custom_headers: BTreeMap<String, String>) -> Self {
        self.custom_headers = custom_headers;
        self
    }

    /// Overrides the request-level model settings.
    #[must_use]
    pub fn with_model_config(mut self, model_config: Option<ModelRequestConfig>) -> Self {
        self.model_config = model_config;
        self
    }

    /// Overrides the client-level model settings.
    #[must_use]
    pub fn with_model_client_config(
        mut self,
        model_client_config: Option<ModelClientConfig>,
    ) -> Self {
        self.model_client_config = model_client_config;
        self
    }

    /// Overrides the agent identity card.
    #[must_use]
    pub fn with_card(mut self, card: AgentCard) -> Self {
        self.card = card;
        self
    }

    /// Overrides the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: Option<String>) -> Self {
        self.system_prompt = system_prompt;
        self
    }

    /// Overrides the inner iteration cap.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: Option<u32>) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Overrides the single-round iteration ceiling.
    #[must_use]
    pub fn with_max_iterations_hard_limit(
        mut self,
        max_iterations_hard_limit: Option<u32>,
    ) -> Self {
        self.max_iterations_hard_limit = max_iterations_hard_limit;
        self
    }

    /// Overrides the iteration-exhaustion policy.
    #[must_use]
    pub fn with_on_max_iterations(mut self, on_max_iterations: OnMaxIterations) -> Self {
        self.on_max_iterations = on_max_iterations;
        self
    }

    /// Overrides whether the outer task loop runs.
    #[must_use]
    pub fn with_enable_task_loop(mut self, enable_task_loop: bool) -> Self {
        self.enable_task_loop = enable_task_loop;
        self
    }

    /// Overrides whether async sub-agent tools are registered.
    #[must_use]
    pub fn with_enable_async_subagent(mut self, enable_async_subagent: bool) -> Self {
        self.enable_async_subagent = enable_async_subagent;
        self
    }

    /// Overrides whether the persistent sub-agent runtime tools are exposed.
    #[must_use]
    pub fn with_enable_subagent_runtime(mut self, enable_subagent_runtime: bool) -> Self {
        self.enable_subagent_runtime = enable_subagent_runtime;
        self
    }

    /// Overrides whether a general-purpose sub-agent is added.
    #[must_use]
    pub fn with_add_general_purpose_agent(mut self, add_general_purpose_agent: bool) -> Self {
        self.add_general_purpose_agent = add_general_purpose_agent;
        self
    }

    /// Overrides the per-iteration completion timeout.
    #[must_use]
    pub fn with_completion_timeout_secs(mut self, completion_timeout_secs: f64) -> Self {
        self.completion_timeout_secs = completion_timeout_secs;
        self
    }

    /// Overrides the workspace.
    #[must_use]
    pub fn with_workspace(mut self, workspace: Workspace) -> Self {
        self.workspace = workspace;
        self
    }

    /// Overrides the shell working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: Option<PathBuf>) -> Self {
        self.cwd = cwd;
        self
    }

    /// Overrides the project root.
    #[must_use]
    pub fn with_project_root(mut self, project_root: Option<PathBuf>) -> Self {
        self.project_root = project_root;
        self
    }

    /// Overrides the configured skills.
    #[must_use]
    pub fn with_skills(mut self, skills: Vec<String>) -> Self {
        self.skills = skills;
        self
    }

    /// Overrides whether skills are discovered from the workspace.
    #[must_use]
    pub fn with_enable_skill_discovery(mut self, enable_skill_discovery: bool) -> Self {
        self.enable_skill_discovery = enable_skill_discovery;
        self
    }

    /// Overrides the prompt language.
    #[must_use]
    pub fn with_language(mut self, language: String) -> Self {
        self.language = language;
        self
    }

    /// Overrides the prompt verbosity.
    #[must_use]
    pub fn with_prompt_mode(mut self, prompt_mode: PromptMode) -> Self {
        self.prompt_mode = prompt_mode;
        self
    }

    /// Overrides the sub-agent specifications.
    #[must_use]
    pub fn with_subagents(mut self, subagents: Vec<SubAgentConfig>) -> Self {
        self.subagents = subagents;
        self
    }

    /// Overrides the mode the agent starts in.
    #[must_use]
    pub fn with_default_mode(mut self, default_mode: AgentMode) -> Self {
        self.default_mode = default_mode;
        self
    }

    /// Overrides whether plan mode may be entered.
    #[must_use]
    pub fn with_enable_plan_mode(mut self, enable_plan_mode: bool) -> Self {
        self.enable_plan_mode = enable_plan_mode;
        self
    }

    /// Overrides whether tool calls run in parallel.
    #[must_use]
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    /// Overrides whether file operations are restricted to the work dir.
    #[must_use]
    pub fn with_restrict_to_work_dir(mut self, restrict_to_work_dir: bool) -> Self {
        self.restrict_to_work_dir = restrict_to_work_dir;
        self
    }

    /// Overrides whether the tool-call resilience rail is mounted.
    #[must_use]
    pub fn with_enable_tool_resilience_rail(mut self, enable_tool_resilience_rail: bool) -> Self {
        self.enable_tool_resilience_rail = enable_tool_resilience_rail;
        self
    }

    /// Overrides whether task planning is enabled.
    #[must_use]
    pub fn with_enable_task_planning(mut self, enable_task_planning: bool) -> Self {
        self.enable_task_planning = enable_task_planning;
        self
    }

    /// Overrides whether the security rail is mounted.
    #[must_use]
    pub fn with_enable_security_rail(mut self, enable_security_rail: bool) -> Self {
        self.enable_security_rail = enable_security_rail;
        self
    }

    /// Overrides the tool permission policy.
    #[must_use]
    pub fn with_permissions(mut self, permissions: Option<PermissionsSection>) -> Self {
        self.permissions = permissions;
        self
    }

    /// Overrides the task-loop no-progress guard.
    #[must_use]
    pub fn with_task_loop_no_progress_guard(
        mut self,
        task_loop_no_progress_guard: TaskLoopNoProgressGuardConfig,
    ) -> Self {
        self.task_loop_no_progress_guard = task_loop_no_progress_guard;
        self
    }

    /// Overrides the skill count budget.
    #[must_use]
    pub fn with_skill_budget_max_skills(mut self, skill_budget_max_skills: Option<usize>) -> Self {
        self.skill_budget_max_skills = skill_budget_max_skills;
        self
    }

    /// Overrides the skill character budget.
    #[must_use]
    pub fn with_skill_budget_max_total_chars(
        mut self,
        skill_budget_max_total_chars: Option<usize>,
    ) -> Self {
        self.skill_budget_max_total_chars = skill_budget_max_total_chars;
        self
    }

    /// Overrides the per-invocation log directory.
    #[must_use]
    pub fn with_invocation_log_dir(mut self, invocation_log_dir: Option<PathBuf>) -> Self {
        self.invocation_log_dir = invocation_log_dir;
        self
    }

    /// Overrides whether every turn is journaled as a resumable execution.
    #[must_use]
    pub fn with_persistent_agent(mut self, persistent_agent: bool) -> Self {
        self.persistent_agent = persistent_agent;
        self
    }
}

/// Maps the configured `DeepAgent` iteration cap onto the inner `ReAct` loop cap.
///
/// Python parity: `resolve_inner_react_max_iterations`. `None` (unconfigured)
/// keeps the inner loop unbounded, which Python expresses as `sys.maxsize`.
#[must_use]
pub const fn resolve_inner_react_max_iterations(max_iterations: Option<u32>) -> usize {
    match max_iterations {
        Option::None => usize::MAX,
        Option::Some(value) => value as usize,
    }
}

/// Deserializes `skills`, accepting Python's `Union[str, List[str]]` shape.
///
/// A single string becomes a one-element list, an array is decoded as a list of
/// strings, and any other JSON type is rejected.
fn deserialize_skills<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(skill) => Ok(vec![skill]),
        array @ Value::Array(_) => match serde_json::from_value::<Vec<String>>(array) {
            Ok(skills) => Ok(skills),
            Err(err) => Err(<D::Error as serde::de::Error>::custom(err)),
        },
        other => Err(<D::Error as serde::de::Error>::invalid_type(
            serde::de::Unexpected::Other(json_kind(&other)),
            &"a skill name or a list of skill names",
        )),
    }
}

/// Returns a human-readable JSON kind for error messages.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
