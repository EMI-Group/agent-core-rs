//! Default sub-agent construction for the `DeepAgent` runtime.
//!
//! The in-process [`SubAgentRuntime`] delegates child construction to a
//! [`SubAgentFactory`] seam so the runtime never depends on this module. The
//! harness implementation, [`DeepAgentSubAgentFactory`], builds a nested
//! `DeepAgent` from a [`SubAgentConfig`]: it re-roots the child workspace at
//! `<parent workspace>/sub-agents/<session_id>`, inherits the parent's model,
//! language and prompt mode unless the sub-agent overrides them, and never
//! mounts sub-agent tools again (one level of nesting).
//!
//! [`wire_subagents`] is the builder-side helper: it assembles the registry, the
//! runtime, the `task` tool and the persistent `subagent_*` tools.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::Agent;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::{Tool, ToolCard};

use crate::schema::{DeepAgentConfig, PromptMode, SubAgentConfig};
use crate::subagents::{
    general_purpose_config, subagent_tools_with_transcript, SubAgentFactory, SubAgentRegistry,
    SubAgentRuntime, SubAgentSpec, TaskTool,
};
use crate::workspace::Workspace;

use super::builder::DeepAgentBuilder;

/// Path segment under the parent workspace holding every child workspace.
const SUBAGENT_WORKSPACE_DIR: &str = "sub-agents";

/// Builds the nested `DeepAgent` of one sub-agent instance.
///
/// Python parity: the `factory_name` default path of
/// `openjiuwen/harness/subagent_runtime/`. The factory holds the immutable
/// parent resources every child inherits; cloning is cheap except for the
/// workspace value.
#[derive(Clone)]
pub struct DeepAgentSubAgentFactory {
    /// Parent model inherited by every child, when the parent has one.
    model: Option<Arc<dyn Model>>,
    /// Fallback language for children that do not set their own.
    language: String,
    /// Fallback prompt mode for children that do not set their own.
    prompt_mode: PromptMode,
    /// Parent workspace the child workspace is re-rooted under.
    parent_workspace: Workspace,
    /// Nesting depth of the parent agent; `0` is the root agent.
    depth: usize,
}

impl DeepAgentSubAgentFactory {
    /// Builds a factory for the children of a depth-`0` agent.
    #[must_use]
    pub fn new(
        model: Option<Arc<dyn Model>>,
        language: impl Into<String>,
        prompt_mode: PromptMode,
        parent_workspace: Workspace,
    ) -> Self {
        Self {
            model,
            language: language.into(),
            prompt_mode,
            parent_workspace,
            depth: 0,
        }
    }

    /// Returns this factory configured for an agent nested `depth` levels deep.
    #[must_use]
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Builds the configuration of the child agent for `config`.
    ///
    /// The nested `subagents` list and `add_general_purpose_agent` are always
    /// cleared: nested agents never mount sub-agent tools again.
    fn nested_config(&self, config: &SubAgentConfig) -> DeepAgentConfig {
        let language = config
            .language
            .clone()
            .unwrap_or_else(|| self.language.clone());
        let prompt_mode = config.prompt_mode.unwrap_or(self.prompt_mode);
        DeepAgentConfig::default()
            .with_card(config.agent_card.clone())
            .with_system_prompt(config.system_prompt.clone())
            .with_skills(config.skills.clone())
            .with_max_iterations(config.max_iterations)
            .with_enable_task_loop(config.enable_task_loop)
            .with_language(language)
            .with_prompt_mode(prompt_mode)
            .with_enable_plan_mode(config.enable_plan_mode)
            .with_parallel_tool_calls(config.parallel_tool_calls)
            .with_restrict_to_work_dir(config.restrict_to_work_dir)
            .with_subagents(Vec::new())
            .with_add_general_purpose_agent(false)
    }
}

#[async_trait]
impl SubAgentFactory for DeepAgentSubAgentFactory {
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>> {
        let root = self
            .parent_workspace
            .root_path
            .join(SUBAGENT_WORKSPACE_DIR)
            .join(session_id);
        let workspace = Workspace::for_language(root, self.language.clone());
        tracing::debug!(
            depth = self.depth,
            session_id,
            agent = %config.agent_card.name,
            "building a nested DeepAgent sub-agent"
        );
        let nested = self.nested_config(config);
        let mut builder = DeepAgentBuilder::new(config.agent_card.clone())
            .config(nested)
            .workspace(workspace);
        if let Some(model) = &self.model {
            builder = builder.model(model.clone());
        }
        let agent = builder.build()?;
        Ok(agent as Arc<dyn Agent>)
    }
}

/// Which sub-agent tool surface a configuration selects.
///
/// Python parity: `SubagentRail` has three modes and resolves them in this order —
/// `enable_subagent_runtime` wins over `enable_async_subagent`, which wins over the
/// synchronous `task_tool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubagentMode {
    /// The persistent runtime tools (`subagent_spawn`/`_wait`/`_list`/…), selected by
    /// [`DeepAgentConfig::enable_subagent_runtime`].
    Runtime,
    /// The asynchronous lifecycle subset needed to drive background sessions
    /// (`subagent_spawn`, `subagent_wait`, `subagent_list`, `subagent_close`),
    /// selected by [`DeepAgentConfig::enable_async_subagent`] without the runtime.
    ///
    /// Python's equivalent is `sessions_spawn`/`sessions_list`/`sessions_cancel`; the
    /// session toolkit itself is not ported, so the same operations run on the
    /// persistent runtime tools.
    AsyncSession,
    /// The synchronous `task_tool` only — the default, and the pre-flag behaviour of
    /// Python's rail.
    SyncTask,
}

/// The runtime tools the [`SubagentMode::AsyncSession`] surface mounts.
const ASYNC_SESSION_TOOL_NAMES: [&str; 4] = [
    "subagent_spawn",
    "subagent_wait",
    "subagent_list",
    "subagent_close",
];

/// Resolves the sub-agent tool mode of `config`.
#[must_use]
pub(crate) fn subagent_mode(config: &DeepAgentConfig) -> SubagentMode {
    if config.enable_subagent_runtime {
        SubagentMode::Runtime
    } else if config.enable_async_subagent {
        SubagentMode::AsyncSession
    } else {
        SubagentMode::SyncTask
    }
}

/// Assembles the sub-agent runtime and its model-facing tools.
///
/// Registers the built-in presets for `config.language`, every configured
/// [`SubAgentConfig`], and the general-purpose fallback when
/// [`DeepAgentConfig::add_general_purpose_agent`] is set. The runtime starts
/// with an empty parent session id: the per-spawn child session id is allocated
/// by the runtime itself.
///
/// The mounted tools follow [`subagent_mode`]: the synchronous `task_tool` is always
/// present, [`SubagentMode::Runtime`] adds the seven-tool
/// [`subagent_tools_with_transcript`] set, [`SubagentMode::AsyncSession`] adds its four
/// lifecycle tools, and [`SubagentMode::SyncTask`] adds none of them.
///
/// A wired sub-agent runtime with its model-facing tools and their cards.
pub(crate) type SubagentWiring = (Arc<SubAgentRuntime>, Vec<Arc<dyn Tool>>, Vec<ToolCard>);
///
/// # Errors
///
/// Currently infallible; the signature keeps room for registry validation and
/// mirrors the builder call site.
#[allow(clippy::unnecessary_wraps)] // kept fallible for the factory layer (next wave)
pub(crate) fn wire_subagents(
    config: &DeepAgentConfig,
    workspace: &Workspace,
    model: Option<Arc<dyn Model>>,
    factory: Option<Arc<dyn SubAgentFactory>>,
) -> Result<SubagentWiring> {
    let registry = Arc::new(SubAgentRegistry::with_builtin_presets(&config.language));
    for subagent in &config.subagents {
        registry.register(SubAgentSpec::new(subagent.clone()));
    }
    if config.add_general_purpose_agent {
        registry.register(SubAgentSpec::new(general_purpose_config()));
    }
    let factory: Arc<dyn SubAgentFactory> = match factory {
        Some(factory) => factory,
        None => Arc::new(DeepAgentSubAgentFactory::new(
            model,
            config.language.clone(),
            config.prompt_mode,
            workspace.clone(),
        )),
    };
    let runtime = Arc::new(SubAgentRuntime::new(registry, factory, ""));
    let mut tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TaskTool::new(runtime.clone()))];
    match subagent_mode(config) {
        SubagentMode::Runtime => tools.extend(subagent_tools_with_transcript(runtime.clone())),
        SubagentMode::AsyncSession => tools.extend(
            subagent_tools_with_transcript(runtime.clone())
                .into_iter()
                .filter(|tool| ASYNC_SESSION_TOOL_NAMES.contains(&tool.card().name.as_str())),
        ),
        SubagentMode::SyncTask => {}
    }
    let cards = tools.iter().map(|tool| tool.card().clone()).collect();
    Ok((runtime, tools, cards))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_test_support::TempWorkspace;

    use super::{subagent_mode, wire_subagents, DeepAgentSubAgentFactory, SubagentMode};
    use crate::schema::{DeepAgentConfig, PromptMode, SubAgentConfig};
    use crate::subagents::SubAgentFactory;
    use crate::workspace::Workspace;

    #[test]
    fn factory_new_and_with_depth() {
        let workspace = Workspace::new(".");
        let factory =
            DeepAgentSubAgentFactory::new(None, "en", PromptMode::Full, workspace).with_depth(2);
        assert_eq!(factory.depth, 2);
        assert_eq!(factory.language, "en");
    }

    #[test]
    fn nested_config_inherits_language_and_prompt_mode() {
        let factory =
            DeepAgentSubAgentFactory::new(None, "en", PromptMode::Minimal, Workspace::new("."));
        let subagent = SubAgentConfig::new(openjiuwen_agent::AgentCard::new("child"));
        let nested = factory.nested_config(&subagent);
        assert_eq!(nested.language, "en");
        assert_eq!(nested.prompt_mode, PromptMode::Minimal);
        assert_eq!(nested.subagents.len(), 0);
        assert!(!nested.add_general_purpose_agent);
    }

    #[test]
    fn nested_config_overrides_language_when_the_child_sets_one() {
        let factory =
            DeepAgentSubAgentFactory::new(None, "en", PromptMode::Full, Workspace::new("."));
        let subagent = SubAgentConfig::new(openjiuwen_agent::AgentCard::new("child"))
            .with_language(Some("cn".to_string()));
        assert_eq!(factory.nested_config(&subagent).language, "cn");
    }

    #[tokio::test]
    async fn factory_creates_a_nested_deep_agent_in_the_child_workspace() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let model: std::sync::Arc<dyn openjiuwen_foundation::llm::Model> =
            std::sync::Arc::new(openjiuwen_test_support::ScriptedModel::text("child answer"));
        let factory =
            DeepAgentSubAgentFactory::new(Some(model), "en", PromptMode::Full, workspace.clone());
        let subagent = SubAgentConfig::new(openjiuwen_agent::AgentCard::new("child"));
        let agent = factory.create(&subagent, "sub-1").await.unwrap();
        assert_eq!(agent.card().name, "child");
    }

    #[test]
    fn wire_subagents_builds_the_task_and_runtime_tools() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let config = DeepAgentConfig::default()
            .with_add_general_purpose_agent(true)
            .with_enable_subagent_runtime(true);
        let (runtime, tools, cards) = wire_subagents(&config, &workspace, None, None).unwrap();
        // Five built-in presets plus the general-purpose fallback.
        assert_eq!(runtime.registry().len(), 6);
        // `task_tool` plus the seven-tool `subagent_*` set.
        assert_eq!(tools.len(), 8);
        assert_eq!(cards.len(), 8);
        assert!(cards.iter().any(|card| card.name == "task_tool"));
        assert!(cards.iter().any(|card| card.name == "subagent_spawn"));
        assert!(cards.iter().any(|card| card.name == "subagent_transcript"));
        assert_eq!(
            cards
                .iter()
                .filter(|card| card.name.starts_with("subagent_"))
                .count(),
            7
        );
    }

    #[test]
    fn the_default_mode_mounts_only_the_synchronous_task_tool() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let config = DeepAgentConfig::default().with_add_general_purpose_agent(true);
        let (_, tools, cards) = wire_subagents(&config, &workspace, None, None).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].name, "task_tool");
    }

    #[test]
    fn the_async_mode_mounts_the_lifecycle_subset_and_runtime_wins() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let async_config = DeepAgentConfig::default()
            .with_add_general_purpose_agent(true)
            .with_enable_async_subagent(true);
        let (_, _, cards) = wire_subagents(&async_config, &workspace, None, None).unwrap();
        let names: Vec<&str> = cards.iter().map(|card| card.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "task_tool",
                "subagent_spawn",
                "subagent_wait",
                "subagent_list",
                "subagent_close"
            ]
        );

        let both = async_config.clone().with_enable_subagent_runtime(true);
        let (_, _, cards) = wire_subagents(&both, &workspace, None, None).unwrap();
        assert_eq!(cards.len(), 8, "the runtime mode wins over the async mode");
    }

    #[test]
    fn the_mode_precedence_matches_python() {
        let base = DeepAgentConfig::default();
        assert_eq!(subagent_mode(&base), SubagentMode::SyncTask);
        assert_eq!(
            subagent_mode(&base.clone().with_enable_async_subagent(true)),
            SubagentMode::AsyncSession
        );
        assert_eq!(
            subagent_mode(
                &base
                    .clone()
                    .with_enable_async_subagent(true)
                    .with_enable_subagent_runtime(true)
            ),
            SubagentMode::Runtime
        );
    }
}
