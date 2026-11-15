//! Assembly of the `DeepAgent` construction inputs and the async factory.
//!
//! Ports `openjiuwen/harness/factory.py`: [`resolve_deep_agent_parts`] is the
//! pure-assembly half (configuration validation, default rail selection, tool
//! pass-through), [`apply_deep_agent_parts`] materializes the parts on an
//! already built agent, and [`create_deep_agent`] is the one-call factory with
//! the Python keyword surface.
//!
//! Default rails are added in the Python factory order and only when the caller
//! did not supply a rail of the same [`HarnessRail::name`], so a caller rail
//! always wins:
//!
//! 1. `security` — when [`DeepAgentConfig::enable_security_rail`] is set; a
//!    pipeline that cannot be built is logged and skipped (Python logs and
//!    continues).
//! 2. `skills` — when skills are configured or discovery is enabled; the
//!    library is rooted at the workspace `skills` node plus every configured
//!    skill path that exists on disk.
//! 3. `subagent` — when sub-agents are configured or the general-purpose agent
//!    is enabled.
//! 4. `goal` — when the mounted tools include `get_current_goal` or
//!    `submit_goal_report`; the rail is rooted at the workspace goal store.
//!
//! Memory rails are never added here: they carry caller-supplied recall and
//! write-back seams, so they belong to
//! [`DeepAgentBuilder::memory`](super::DeepAgentBuilder::memory) (documented
//! deviation from Python, where the memory rail is built from config data).

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::Tool;
use openjiuwen_security::SecurityConfig;

use crate::permission::{PermissionHost, PermissionsSection};
use crate::rails::{HarnessRail, SecurityRail, SkillsRail, SkillsRailConfig, SubagentRail};
use crate::schema::{DeepAgentConfig, PromptMode, SubAgentConfig};
use crate::skills::{SkillBudget, SkillLibrary};
use crate::tools::goal::{GoalRail, GoalRailConfig, GoalStore};
use crate::tools::BuiltinTool;
use crate::workspace::Workspace;

use super::builder::DeepAgentBuilder;
use super::DeepAgent;

/// Name of the default security rail.
const SECURITY_RAIL_NAME: &str = "security";

/// Name of the default skills rail.
const SKILLS_RAIL_NAME: &str = "skills";

/// Name of the default subagent rail.
pub(crate) const SUBAGENT_RAIL_NAME: &str = "subagent";

/// Name of the default goal rail.
pub(crate) const GOAL_RAIL_NAME: &str = "goal";

/// Workspace node holding the skill library.
const SKILLS_WORKSPACE_NODE: &str = "skills";

/// Name of the card [`create_deep_agent`] builds when the caller passes none.
const DEFAULT_CARD_NAME: &str = "deep_agent";

/// Description of that default card.
const DEFAULT_CARD_DESCRIPTION: &str = "DeepAgent instance";

/// Root of the default workspace [`create_deep_agent`] builds.
const DEFAULT_WORKSPACE_ROOT: &str = "./";

/// Assembled [`DeepAgent`] construction inputs, decoupled from any instance.
///
/// Python parity: `DeepAgentParts`. [`resolve_deep_agent_parts`] produces these
/// from the raw inputs (pure assembly, no agent created);
/// [`apply_deep_agent_parts`] materializes them onto a target agent, which lets
/// upper layers configure an agent they already own instead of building a
/// throwaway template.
pub struct DeepAgentParts {
    /// The assembled configuration; rails travel separately and are applied
    /// through [`DeepAgent::add_rail`].
    pub config: DeepAgentConfig,
    /// Caller rails followed by the default rails that were not suppressed.
    pub rails: Vec<Arc<dyn HarnessRail>>,
    /// Tools to mount next to the built-in coding tools, unchanged.
    pub tools: Vec<Arc<dyn Tool>>,
}

impl fmt::Debug for DeepAgentParts {
    /// Prints the configuration plus the rail/tool counts; the rails and tools
    /// themselves are trait objects without a stable representation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepAgentParts")
            .field("config", &self.config)
            .field(
                "rails",
                &self
                    .rails
                    .iter()
                    .map(|rail| rail.name())
                    .collect::<Vec<_>>(),
            )
            .field("tools", &self.tools.len())
            .finish()
    }
}

/// Assembles the configuration, default rails and tools of a `DeepAgent`.
///
/// Rails are appended in the Python factory order (`security`, `skills`,
/// `subagent`, `goal`) and each default is skipped when `rails` already contains a
/// rail with the same [`HarnessRail::name`]. The caller rails are appended after the
/// defaults in their original order, so they keep precedence.
///
/// `model` is not embedded in the parts (the configuration carries only data);
/// it is logged for traceability and consumed by
/// [`DeepAgentBuilder::model`](super::DeepAgentBuilder::model) on the
/// [`create_deep_agent`] path.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::config`] when
/// [`DeepAgentConfig::validate`] rejects the configuration.
#[allow(clippy::needless_pass_by_value)] // the fixed Python factory signature takes the model by value
pub fn resolve_deep_agent_parts(
    config: DeepAgentConfig,
    model: Option<Arc<dyn Model>>,
    tools: Vec<Arc<dyn Tool>>,
    rails: Vec<Arc<dyn HarnessRail>>,
) -> Result<DeepAgentParts> {
    config.validate()?;
    if let Some(model) = &model {
        tracing::debug!(
            provider = model.provider(),
            model = model.model_name(),
            rails = rails.len(),
            tools = tools.len(),
            "resolving DeepAgent parts"
        );
    } else {
        tracing::debug!(
            rails = rails.len(),
            tools = tools.len(),
            "resolving DeepAgent parts without a model"
        );
    }

    let caller_names: Vec<&str> = rails.iter().map(|rail| rail.name()).collect();
    let mut all_rails: Vec<Arc<dyn HarnessRail>> = Vec::new();

    if config.enable_security_rail && !caller_names.contains(&SECURITY_RAIL_NAME) {
        match SecurityRail::from_config(&SecurityConfig::default()) {
            Ok(rail) => all_rails.push(Arc::new(rail)),
            Err(error) => tracing::warn!(
                error = %error,
                "skipping the default security rail: its pipeline could not be built"
            ),
        }
    }

    if (!config.skills.is_empty() || config.enable_skill_discovery)
        && !caller_names.contains(&SKILLS_RAIL_NAME)
    {
        all_rails.push(Arc::new(default_skills_rail(&config)));
    }

    if (!config.subagents.is_empty() || config.add_general_purpose_agent)
        && !caller_names.contains(&SUBAGENT_RAIL_NAME)
    {
        all_rails.push(Arc::new(SubagentRail::new()));
    }

    if tools
        .iter()
        .any(|tool| is_goal_tool_name(&tool.card().name))
        && !caller_names.contains(&GOAL_RAIL_NAME)
    {
        all_rails.push(default_goal_rail(&config.workspace, &config.language));
    }

    all_rails.extend(rails);
    Ok(DeepAgentParts {
        config,
        rails: all_rails,
        tools,
    })
}

/// Builds the default skills rail of `config`.
///
/// The library is rooted at the workspace `skills` node (added even when the
/// directory does not exist yet, like the Python rail) plus every configured
/// skill path that exists. The selection budget and description language come
/// from the configuration.
fn default_skills_rail(config: &DeepAgentConfig) -> SkillsRail {
    let mut library = SkillLibrary::new();
    if let Some(path) = config.workspace.get_node_path(SKILLS_WORKSPACE_NODE) {
        library.add_root(path);
    }
    for skill in &config.skills {
        if Path::new(skill).exists() {
            library.add_root(skill.clone());
        }
    }
    SkillsRail::new(library).with_config(SkillsRailConfig {
        budget: SkillBudget {
            max_skills: config.skill_budget_max_skills,
            max_total_chars: config.skill_budget_max_total_chars,
        },
        language: Some(config.language.clone()),
    })
}

/// Whether `name` is the model-facing name of one of the two goal tools.
///
/// A tool set without a goal tool cannot read or report a goal, so exactly those two names
/// gate the default goal rail.
pub(crate) fn is_goal_tool_name(name: &str) -> bool {
    [
        BuiltinTool::GetCurrentGoal.name(),
        BuiltinTool::SubmitGoalReport.name(),
    ]
    .contains(&name)
}

/// Builds the default goal rail over the workspace goal store.
pub(crate) fn default_goal_rail(workspace: &Workspace, language: &str) -> Arc<dyn HarnessRail> {
    Arc::new(GoalRail::new(
        GoalStore::for_workspace(workspace),
        GoalRailConfig::default().with_prompt_language(language.to_string()),
    ))
}

/// Applies resolved [`DeepAgentParts`] onto an already built agent.
///
/// `agent.configure(parts.config)` rebuilds the prompt sections and re-applies
/// the inner `ReActAgentConfig`, `agent.apply_extra_tools(parts.tools)` mounts
/// the extra tools and refreshes the tools section, and every part rail is
/// registered through [`DeepAgent::add_rail`] (harness registry *and* inner
/// agent). Use this when the caller assembles the parts itself — for example a
/// protocol host configuring an agent it already owns.
///
/// # Errors
///
/// Kept fallible for parity with the Python call site (which may fail while
/// registering tool resources); the current assembly performs no fallible step.
#[allow(clippy::unnecessary_wraps)] // parity with the Python call site
pub fn apply_deep_agent_parts(agent: &Arc<DeepAgent>, parts: DeepAgentParts) -> Result<()> {
    let DeepAgentParts {
        config,
        rails,
        tools,
    } = parts;
    agent.configure(config);
    agent.apply_extra_tools(tools);
    for rail in rails {
        agent.add_rail(rail);
    }
    Ok(())
}

/// Builds a fully wired `DeepAgent` from the Python keyword surface.
///
/// `config` (when supplied) is the base configuration: the keyword arguments
/// are applied onto it. The always-supplied booleans (`enable_task_loop`,
/// `restrict_to_work_dir`) overwrite the base value, so a caller starting from a
/// non-default configuration must pass the intended value explicitly. Lists and
/// options only override when supplied: an empty `skills`/`subagents` list and a
/// `None` `max_iterations`/`permissions` keep the base configuration's value.
/// `language` and `prompt_mode` win when supplied, otherwise the base
/// configuration keeps its own.
///
/// `card` defaults to `deep_agent` (`"DeepAgent instance"`) and `workspace` to
/// `Workspace::for_language("./", <language>)`.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::config`] when the assembled
/// configuration is invalid or the agent cannot be wired.
#[allow(clippy::too_many_arguments)] // the documented Python keyword surface
#[allow(clippy::unused_async)] // parity: the factory is awaited by every upper layer
pub async fn create_deep_agent(
    model: Arc<dyn Model>,
    card: Option<AgentCard>,
    config: Option<DeepAgentConfig>,
    tools: Vec<Arc<dyn Tool>>,
    rails: Vec<Arc<dyn HarnessRail>>,
    workspace: Option<Workspace>,
    skills: Vec<String>,
    subagents: Vec<SubAgentConfig>,
    enable_task_loop: bool,
    max_iterations: Option<u32>,
    language: Option<String>,
    prompt_mode: Option<PromptMode>,
    restrict_to_work_dir: bool,
    permissions: Option<PermissionsSection>,
    permission_host: Option<Arc<dyn PermissionHost>>,
) -> Result<Arc<DeepAgent>> {
    let mut config = config.unwrap_or_default();
    config.card = card.unwrap_or_else(|| {
        AgentCard::new(DEFAULT_CARD_NAME).with_description(DEFAULT_CARD_DESCRIPTION)
    });
    if !skills.is_empty() {
        config.skills = skills;
    }
    if !subagents.is_empty() {
        config.subagents = subagents;
    }
    config.enable_task_loop = enable_task_loop;
    if max_iterations.is_some() {
        config.max_iterations = max_iterations;
    }
    if let Some(language) = language {
        config.language = language;
    }
    if let Some(prompt_mode) = prompt_mode {
        config.prompt_mode = prompt_mode;
    }
    config.restrict_to_work_dir = restrict_to_work_dir;
    if permissions.is_some() {
        config.permissions = permissions;
    }
    let workspace = workspace.unwrap_or_else(|| {
        Workspace::for_language(DEFAULT_WORKSPACE_ROOT, config.language.clone())
    });
    config.workspace = workspace.clone();

    let parts = resolve_deep_agent_parts(config, Some(model.clone()), tools, rails)?;
    let DeepAgentParts {
        config,
        rails,
        tools,
    } = parts;

    let mut builder = DeepAgentBuilder::new(config.card.clone())
        .config(config)
        .model(model)
        .workspace(workspace)
        .tools(tools);
    for rail in rails {
        builder = builder.rail(rail);
    }
    if let Some(host) = permission_host {
        builder = builder.permission_host(host);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_agent::{Agent, AgentCard, ResultType};
    use openjiuwen_common::error::Result;
    use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
    use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
    use serde_json::{json, Value};

    use super::{apply_deep_agent_parts, create_deep_agent, resolve_deep_agent_parts};
    use crate::permission::{PermissionLevel, PermissionsSection};
    use crate::rails::{HarnessRail, HarnessRailContext};
    use crate::schema::{DeepAgentConfig, PromptMode, SubAgentConfig};
    use crate::tools::goal::{GetCurrentGoalTool, GoalStore};
    use crate::tools::tool_card;
    use crate::workspace::Workspace;

    /// A rail recording how often its `before_invoke` ran.
    struct CountingRail {
        hits: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl HarnessRail for CountingRail {
        fn name(&self) -> &'static str {
            "counting"
        }

        fn priority(&self) -> i32 {
            crate::rails::rail_priority::DEFAULT
        }

        async fn before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A rail standing in for a caller-supplied default of the same name.
    struct NamedRail {
        name: &'static str,
    }

    #[async_trait]
    impl HarnessRail for NamedRail {
        fn name(&self) -> &str {
            self.name
        }
    }

    /// A tool echoing its input and counting invocations.
    struct EchoTool {
        card: ToolCard,
        calls: Arc<AtomicUsize>,
    }

    impl EchoTool {
        fn new() -> Self {
            Self {
                card: tool_card(
                    "echo",
                    "Echo the input text back",
                    json!({"type": "object"}),
                    true,
                ),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"success": true, "echo": inputs}))
        }
    }

    fn names(patterns: &[&str]) -> Vec<String> {
        patterns.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn defaults_include_the_security_rail() {
        let parts =
            resolve_deep_agent_parts(DeepAgentConfig::default(), None, Vec::new(), Vec::new())
                .unwrap();
        assert_eq!(parts.rails.len(), 1);
        assert_eq!(parts.rails[0].name(), "security");
        assert!(parts.tools.is_empty());
    }

    #[test]
    fn security_rail_can_be_disabled() {
        let config = DeepAgentConfig::default().with_enable_security_rail(false);
        let parts = resolve_deep_agent_parts(config, None, Vec::new(), Vec::new()).unwrap();
        assert!(parts.rails.is_empty());
    }

    #[test]
    fn skills_rail_is_added_for_configured_skills_and_discovery() {
        let temp = TempWorkspace::new().unwrap();
        let root = temp.workspace().join("custom-skills");
        std::fs::create_dir_all(&root).unwrap();
        let config = DeepAgentConfig::default().with_skills(vec![
            root.display().to_string(),
            temp.workspace().join("missing").display().to_string(),
        ]);
        let parts = resolve_deep_agent_parts(config, None, Vec::new(), Vec::new()).unwrap();
        let names: Vec<&str> = parts.rails.iter().map(|rail| rail.name()).collect();
        assert_eq!(names, ["security", "skills"]);

        let discovery = DeepAgentConfig::default().with_enable_skill_discovery(true);
        let parts = resolve_deep_agent_parts(discovery, None, Vec::new(), Vec::new()).unwrap();
        assert_eq!(parts.rails.len(), 2);
    }

    #[test]
    fn subagent_rail_is_added_for_configured_subagents() {
        let config = DeepAgentConfig::default().with_add_general_purpose_agent(true);
        let parts = resolve_deep_agent_parts(config, None, Vec::new(), Vec::new()).unwrap();
        let names: Vec<&str> = parts.rails.iter().map(|rail| rail.name()).collect();
        assert_eq!(names, ["security", "subagent"]);

        let config = DeepAgentConfig::default()
            .with_subagents(vec![SubAgentConfig::new(AgentCard::new("helper"))]);
        let parts = resolve_deep_agent_parts(config, None, Vec::new(), Vec::new()).unwrap();
        assert!(parts.rails.iter().any(|rail| rail.name() == "subagent"));
    }

    #[test]
    fn caller_rails_suppress_defaults_and_are_appended_last() {
        let config = DeepAgentConfig::default()
            .with_skills(vec!["missing-skill".to_string()])
            .with_add_general_purpose_agent(true);
        let rails: Vec<Arc<dyn HarnessRail>> = vec![
            Arc::new(NamedRail { name: "security" }),
            Arc::new(NamedRail { name: "custom" }),
        ];
        let parts = resolve_deep_agent_parts(config, None, Vec::new(), rails).unwrap();
        let names: Vec<&str> = parts.rails.iter().map(|rail| rail.name()).collect();
        assert_eq!(names, ["skills", "subagent", "security", "custom"]);
    }

    #[test]
    fn an_invalid_configuration_is_rejected() {
        let config = DeepAgentConfig::default()
            .with_enable_task_loop(true)
            .with_max_iterations_hard_limit(Some(3));
        let error = resolve_deep_agent_parts(config, None, Vec::new(), Vec::new())
            .expect_err("an invalid configuration must fail");
        assert!(error.to_string().contains("max_iterations_hard_limit"));
    }

    #[tokio::test]
    async fn apply_parts_configures_mounts_and_raises_the_extra_rail() {
        let hits = Arc::new(AtomicUsize::new(0));
        let echo = Arc::new(EchoTool::new());
        let mut parts = resolve_deep_agent_parts(
            DeepAgentConfig::default().with_language("en".to_string()),
            None,
            vec![echo.clone() as Arc<dyn Tool>],
            Vec::new(),
        )
        .unwrap();
        parts
            .rails
            .push(Arc::new(CountingRail { hits: hits.clone() }));

        let agent = super::DeepAgentBuilder::new(AgentCard::new("apply-agent"))
            .model(Arc::new(ScriptedModel::text("done")))
            .build()
            .unwrap();
        apply_deep_agent_parts(&agent, parts).unwrap();

        assert!(agent
            .tool_cards
            .read()
            .iter()
            .any(|card| card.name == "echo"));
        assert_eq!(agent.config().language, "en");
        assert!(agent.rails().names().iter().any(|name| name == "counting"));

        let output = agent.invoke(json!({"query": "hi"}), None).await.unwrap();
        assert_eq!(output.result_type, ResultType::Answer);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn create_deep_agent_answers_end_to_end() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let agent = create_deep_agent(
            Arc::new(ScriptedModel::text("factory answer")),
            None,
            None,
            Vec::new(),
            Vec::new(),
            Some(workspace.clone()),
            Vec::new(),
            Vec::new(),
            false,
            Some(3),
            Some("en".to_string()),
            Some(PromptMode::Minimal),
            true,
            None,
            None,
        )
        .await
        .unwrap();

        let config = agent.config();
        assert_eq!(config.card.name, "deep_agent");
        assert_eq!(config.card.description, "DeepAgent instance");
        assert_eq!(config.language, "en");
        assert_eq!(config.prompt_mode, PromptMode::Minimal);
        assert_eq!(config.max_iterations, Some(3));
        assert!(config.restrict_to_work_dir);
        assert_eq!(config.workspace.root_path, workspace.root_path);

        let output = agent.invoke(json!({"query": "hi"}), None).await.unwrap();
        assert_eq!(output.result_type, ResultType::Answer);
        assert_eq!(output.output, "factory answer");
    }

    #[tokio::test]
    async fn create_deep_agent_keeps_the_base_config_when_a_keyword_is_absent() {
        let base = DeepAgentConfig::default()
            .with_language("cn".to_string())
            .with_max_iterations(Some(7))
            .with_skills(names(&["missing"]));
        let agent = create_deep_agent(
            Arc::new(ScriptedModel::text("unused")),
            Some(AgentCard::new("custom").with_description("Custom identity")),
            Some(base),
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
            Vec::new(),
            false,
            None,
            None,
            None,
            true,
            None,
            None,
        )
        .await
        .unwrap();

        let config = agent.config();
        assert_eq!(config.card.name, "custom");
        assert_eq!(config.card.description, "Custom identity");
        assert_eq!(config.language, "cn");
        assert_eq!(config.max_iterations, Some(7));
        assert_eq!(config.skills, names(&["missing"]));
        assert_eq!(config.workspace.language, "cn");
    }

    #[tokio::test]
    async fn create_deep_agent_applies_permissions_and_caller_rails() {
        let temp = TempWorkspace::new().unwrap();
        let mut permissions = PermissionsSection::default();
        permissions
            .tools
            .insert("bash".to_string(), PermissionLevel::Deny);
        let hits = Arc::new(AtomicUsize::new(0));
        let rails: Vec<Arc<dyn HarnessRail>> = vec![Arc::new(CountingRail { hits: hits.clone() })];
        let model = ScriptedModel::builder()
            .completion(openjiuwen_test_support::MockCompletion::tool_calls(vec![
                openjiuwen_test_support::MockToolCall::new("bash", json!({"command": "ls"})),
            ]))
            .respond_text("denied")
            .build();
        let agent = create_deep_agent(
            Arc::new(model),
            None,
            None,
            Vec::new(),
            rails,
            Some(Workspace::for_language(temp.workspace(), "en")),
            Vec::new(),
            Vec::new(),
            false,
            None,
            Some("en".to_string()),
            None,
            true,
            Some(permissions),
            None,
        )
        .await
        .unwrap();

        assert!(agent.config().permissions.is_some());
        assert!(agent.rails().names().iter().any(|name| name == "counting"));
        let output = agent
            .invoke(json!({"query": "run bash"}), None)
            .await
            .unwrap();
        assert_eq!(output.output, "denied");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn create_deep_agent_registers_one_goal_rail_for_a_goal_tool() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let goal_tool = GetCurrentGoalTool::new(GoalStore::for_workspace(&workspace), "en");
        let agent = create_deep_agent(
            Arc::new(ScriptedModel::text("hi")),
            None,
            None,
            vec![Arc::new(goal_tool)],
            Vec::new(),
            Some(workspace),
            Vec::new(),
            Vec::new(),
            false,
            None,
            Some("en".to_string()),
            None,
            true,
            None,
            None,
        )
        .await
        .unwrap();

        let names = agent.rails().names();
        assert_eq!(
            names.iter().filter(|name| name.as_str() == "goal").count(),
            1,
            "the factory and the builder register the goal rail once: {names:?}"
        );
    }

    #[tokio::test]
    async fn create_deep_agent_mounts_the_subagent_transcript_tool() {
        let temp = TempWorkspace::new().unwrap();
        let agent = create_deep_agent(
            Arc::new(ScriptedModel::text("hi")),
            None,
            None,
            Vec::new(),
            Vec::new(),
            Some(Workspace::for_language(temp.workspace(), "en")),
            Vec::new(),
            vec![SubAgentConfig::new(AgentCard::new("helper"))],
            false,
            None,
            Some("en".to_string()),
            None,
            true,
            None,
            None,
        )
        .await
        .unwrap();

        let names: Vec<String> = agent
            .tool_cards
            .read()
            .iter()
            .map(|card| card.name.clone())
            .collect();
        // The six Python parity tools stay mounted and the additive transcript reader is
        // mounted exactly once beside them.
        for expected in [
            "task_tool",
            "subagent_spawn",
            "subagent_wait",
            "subagent_list",
            "subagent_transcript",
            "subagent_send_input",
            "subagent_close",
            "subagent_resume",
        ] {
            assert!(
                names.iter().any(|name| name == expected),
                "missing tool {expected} in {names:?}"
            );
        }
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "subagent_transcript")
                .count(),
            1
        );
        assert!(agent.rails().names().iter().any(|name| name == "subagent"));
    }

    #[test]
    fn debug_reports_rail_names_and_tool_counts() {
        let parts =
            resolve_deep_agent_parts(DeepAgentConfig::default(), None, Vec::new(), Vec::new())
                .unwrap();
        let rendered = format!("{parts:?}");
        assert!(rendered.contains("security"));
        assert!(rendered.contains("tools: 0"));
    }
}
