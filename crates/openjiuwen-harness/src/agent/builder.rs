//! Construction and lazy model wiring of the `DeepAgent`.
//!
//! [`DeepAgentBuilder`] is the place where the live handles Python's
//! `DeepAgentConfig` mixes with data are supplied: the model, the sys-operation
//! facade, extra tools and rails, the permission host, the user asker, the
//! memory seams and the sub-agent factory. [`DeepAgentBuilder::build`] assembles
//! one fully wired [`DeepAgent`].
//!
//! Model resolution is lazy: when no model is supplied, [`ensure_agent_wired`]
//! builds one from the persisted model fields through
//! [`openjiuwen_foundation::llm::init_model`] on the first invocation. A
//! configuration without a model identity fails with [`Error::config`].

use std::sync::atomic::Ordering;
use std::sync::Arc;

use openjiuwen_agent::{AgentCard, ReActAgent};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{init_model, Model, ModelClientConfig, ModelRequestConfig};
use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_foundation::BaseMessage;
use openjiuwen_sysop::SysOperation;
use parking_lot::RwLock;

use crate::permission::{
    PermissionEngine, PermissionEngineConfig, PermissionHost, PermissionInterruptRail,
};
use crate::prompts::{
    identity, mode_instructions, safety, subagent_section, task_loop_section, tools_section,
    workspace_section, HarnessPromptBuilder, PromptCategory, PromptSection, SectionName,
};
use crate::rails::{
    HarnessRail, HarnessRailRegistry, MemoryRail, MemoryRecall, MemoryWriteBack, SubagentRail,
};
use crate::schema::{
    resolve_inner_react_max_iterations, AgentMode, DeepAgentConfig, PlanModeState,
};
use crate::subagents::SubAgentFactory;
use crate::task_loop::LoopQueues;
use crate::tools::{ToolRegistry, UserAsker};
use crate::workspace::Workspace;

use super::factory::{default_goal_rail, is_goal_tool_name, GOAL_RAIL_NAME};
use super::prompt_rail::{PromptRail, PromptState};
use super::rail_events::{RailEventBus, RailEventDrain};
use super::{subagent_factory, DeepAgent, Driver};

/// Prompt priority of the identity section (matches [`identity`]).
const IDENTITY_PRIORITY: i32 = 10;

/// Builder of a fully wired [`DeepAgent`].
///
/// Python parity: the keyword surface of `create_deep_agent` plus the live
/// handles `DeepAgentConfig` cannot carry. Every setter consumes and returns the
/// builder, so calls chain.
pub struct DeepAgentBuilder {
    /// Identity card of the agent being built.
    card: AgentCard,
    /// Data configuration; the builder merges its own live handles into it.
    config: DeepAgentConfig,
    /// Model supplied by the host, if any.
    model: Option<Arc<dyn Model>>,
    /// Sys-operation backend source; `None` derives a local one from the workspace.
    sys_operation: Option<Arc<SysOperation>>,
    /// Workspace override; `None` keeps the configured workspace.
    workspace: Option<Workspace>,
    /// Extra tools mounted next to the built-in coding tools.
    tools: Vec<Arc<dyn Tool>>,
    /// Extra harness rails registered before the built ones.
    rails: Vec<Arc<dyn HarnessRail>>,
    /// Permission host answering human-in-the-loop confirmations.
    permission_host: Option<Arc<dyn PermissionHost>>,
    /// User-interaction handler behind the `ask_user` tool.
    user_asker: Option<Arc<dyn UserAsker>>,
    /// Long-term memory recall and write-back seams.
    memory: Option<(Arc<dyn MemoryRecall>, Arc<dyn MemoryWriteBack>)>,
    /// Factory building the child agents of this agent's sub-agents.
    subagent_factory: Option<Arc<dyn SubAgentFactory>>,
}

impl DeepAgentBuilder {
    /// Starts a builder for `card` with the default configuration.
    #[must_use]
    pub fn new(card: AgentCard) -> Self {
        Self {
            config: DeepAgentConfig::default().with_card(card.clone()),
            card,
            model: None,
            sys_operation: None,
            workspace: None,
            tools: Vec::new(),
            rails: Vec::new(),
            permission_host: None,
            user_asker: None,
            memory: None,
            subagent_factory: None,
        }
    }

    /// Replaces the data configuration.
    #[must_use]
    pub fn config(mut self, config: DeepAgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Supplies the model directly, skipping lazy model resolution.
    #[must_use]
    pub fn model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Uses `operation` as the filesystem/shell backend source.
    #[must_use]
    pub fn sys_operation(mut self, operation: Arc<SysOperation>) -> Self {
        self.sys_operation = Some(operation);
        self
    }

    /// Overrides the workspace of the agent.
    #[must_use]
    pub fn workspace(mut self, workspace: Workspace) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// Mounts one extra tool next to the built-in coding tools.
    #[must_use]
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Mounts several extra tools next to the built-in coding tools.
    #[must_use]
    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Registers one extra harness rail.
    #[must_use]
    pub fn rail(mut self, rail: Arc<dyn HarnessRail>) -> Self {
        self.rails.push(rail);
        self
    }

    /// Supplies the permission host answering confirmation requests.
    #[must_use]
    pub fn permission_host(mut self, host: Arc<dyn PermissionHost>) -> Self {
        self.permission_host = Some(host);
        self
    }

    /// Supplies the handler behind the `ask_user` tool.
    #[must_use]
    pub fn user_asker(mut self, asker: Arc<dyn UserAsker>) -> Self {
        self.user_asker = Some(asker);
        self
    }

    /// Wires the long-term memory recall and write-back seams.
    #[must_use]
    pub fn memory(
        mut self,
        recall: Arc<dyn MemoryRecall>,
        write_back: Arc<dyn MemoryWriteBack>,
    ) -> Self {
        self.memory = Some((recall, write_back));
        self
    }

    /// Supplies the factory building this agent's sub-agent children.
    #[must_use]
    pub fn subagent_factory(mut self, factory: Arc<dyn SubAgentFactory>) -> Self {
        self.subagent_factory = Some(factory);
        self
    }

    /// Builds the fully wired [`DeepAgent`].
    ///
    /// The default rails are the [`PromptRail`], the memory and subagent rails of the supplied
    /// seams, and the goal rail ([`crate::tools::goal::GoalRail`]) when the mounted tool cards
    /// include a goal tool; a caller rail of the same name replaces a default one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the merged configuration is invalid, and
    /// propagates any error from sub-agent assembly.
    #[allow(clippy::too_many_lines)] // pure assembly of the documented wiring order
    pub fn build(self) -> Result<Arc<DeepAgent>> {
        let Self {
            card,
            mut config,
            model,
            sys_operation,
            workspace,
            tools: extra_tools,
            rails: extra_rails,
            permission_host,
            user_asker,
            memory,
            subagent_factory,
        } = self;

        config.card.clone_from(&card);
        if let Some(workspace) = workspace {
            config.workspace = workspace;
        }
        config.validate()?;

        let workspace = config.workspace.clone();
        let language = config.language.clone();

        let permission = config
            .permissions
            .as_ref()
            .filter(|section| section.enabled)
            .map(|section| {
                Arc::new(PermissionEngine::new(
                    PermissionEngineConfig::new(section.clone())
                        .with_workspace_root(workspace.root_path.clone()),
                ))
            });

        let mut registry = ToolRegistry::new(workspace.clone());
        if let Some(operation) = sys_operation {
            registry = registry.with_sys_operation(operation);
        }
        if let Some(engine) = &permission {
            registry = registry.with_permissions(engine.clone());
        }
        if let Some(asker) = user_asker {
            registry = registry.with_user_asker(asker);
        }
        registry = registry.with_language(language.clone());

        let mut tools = registry.build();
        let mut cards = registry.cards(&language);
        for tool in &extra_tools {
            tools.push(tool.clone());
            cards.push(tool.card().clone());
        }

        let mounts_subagents = !config.subagents.is_empty() || config.add_general_purpose_agent;
        let subagent_runtime = if mounts_subagents {
            let (runtime, subagent_tools, subagent_cards) = subagent_factory::wire_subagents(
                &config,
                &workspace,
                model.clone(),
                subagent_factory,
            )?;
            tools.extend(subagent_tools);
            cards.extend(subagent_cards);
            Some(runtime)
        } else {
            None
        };

        let prompt = build_prompt_state(&config, &cards);
        let queues = LoopQueues::default();

        let caller_names: Vec<&str> = extra_rails.iter().map(|rail| rail.name()).collect();
        let rails = HarnessRailRegistry::new();
        rails.register(Arc::new(PromptRail::new(prompt.clone(), queues.clone())));
        if let Some((recall, write_back)) = memory {
            rails.register(Arc::new(
                MemoryRail::new()
                    .with_recall(recall)
                    .with_write_back(write_back),
            ));
        }
        if mounts_subagents {
            rails.register(Arc::new(SubagentRail::new()));
        }
        if cards.iter().any(|card| is_goal_tool_name(&card.name))
            && !caller_names.contains(&GOAL_RAIL_NAME)
        {
            rails.register(default_goal_rail(&workspace, &language));
        }
        for rail in extra_rails {
            rails.register(rail);
        }
        // Registered last so it fires after every rail of equal (lowest) priority.
        let rail_events = Arc::new(RailEventBus::default());
        rails.register(Arc::new(RailEventDrain::new(rail_events.clone())));

        let inner = Arc::new(ReActAgent::new(card.clone()));
        let model_slot = Arc::new(RwLock::new(model.clone()));
        let initialized = model.is_some();
        if let Some(model) = &model {
            inner.set_model(model.clone());
        }

        let agent = DeepAgent {
            card,
            config: Arc::new(RwLock::new(config.clone())),
            inner,
            rails,
            prompt,
            tool_cards: Arc::new(RwLock::new(cards)),
            queues,
            controller: Arc::new(parking_lot::Mutex::new(None)),
            permission,
            permission_host: permission_host
                .unwrap_or_else(|| Arc::new(crate::permission::NoopPermissionHost)),
            subagent_runtime: Arc::new(RwLock::new(subagent_runtime)),
            model: model_slot,
            initialized: Arc::new(std::sync::atomic::AtomicBool::new(initialized)),
            wired: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            cancelled: Arc::new(parking_lot::Mutex::new(
                tokio_util::sync::CancellationToken::new(),
            )),
            state: crate::agent::RoundState::new(),
            rail_events,
            plan_mode: Arc::new(RwLock::new(plan_mode_for(&config))),
            last_session: Arc::new(parking_lot::Mutex::new(None)),
            last_state: Arc::new(parking_lot::Mutex::new(
                crate::schema::DeepAgentState::default(),
            )),
        };

        let driver = agent.driver();
        wire(&driver, tools)?;
        Ok(Arc::new(agent))
    }
}

/// Builds the initial prompt state for `config` and the mounted `cards`.
#[must_use]
pub(crate) fn build_prompt_state(config: &DeepAgentConfig, cards: &[ToolCard]) -> Arc<PromptState> {
    PromptState::new(build_prompt_builder(config, cards))
}

/// Assembles the static prompt sections of `config`.
///
/// Sections are added in the documented order; a section the builder rejects
/// (two sections with one name and different priorities) is logged and skipped,
/// never fatal.
pub(crate) fn build_prompt_builder(
    config: &DeepAgentConfig,
    cards: &[ToolCard],
) -> HarnessPromptBuilder {
    let mut builder = HarnessPromptBuilder::new(config.language.clone(), config.prompt_mode);
    let mut sections = vec![
        identity_section(config),
        safety(),
        tools_section(cards),
        workspace_section(&config.workspace),
    ];
    if config.enable_task_loop {
        sections.push(task_loop_section());
    }
    if config.default_mode == AgentMode::Plan {
        sections.push(mode_instructions(AgentMode::Plan));
    }
    if !config.subagents.is_empty() || config.add_general_purpose_agent {
        sections.push(subagent_section());
    }
    for section in sections {
        if let Err(error) = builder.add_section(section) {
            tracing::warn!(error = %error, "skipping an invalid prompt section");
        }
    }
    builder
}

/// Builds the identity section from the system prompt, the card, or the default.
fn identity_section(config: &DeepAgentConfig) -> PromptSection {
    let text = config
        .system_prompt
        .clone()
        .filter(|text| !text.trim().is_empty())
        .or_else(|| {
            (!config.card.description.trim().is_empty()).then(|| config.card.description.clone())
        });
    match text {
        Some(text) => {
            PromptSection::bilingual(SectionName::IDENTITY, text.clone(), text, IDENTITY_PRIORITY)
                .with_category(PromptCategory::SystemPrompt)
        }
        None => identity(),
    }
}

/// Derives the plan-mode bookkeeping from the configured default mode.
fn plan_mode_for(config: &DeepAgentConfig) -> PlanModeState {
    PlanModeState {
        mode: config.default_mode.as_str().to_string(),
        pre_plan_mode: Some(AgentMode::Normal.as_str().to_string()),
        plan_slug: None,
        prompt_context: None,
    }
}

/// Applies the data configuration to the inner kernel agent.
///
/// The whole prompt template becomes one `system` message carrying
/// `prompt_text`; [`PromptRail`] replaces that message before every model call
/// with the freshly rendered prompt.
pub(crate) fn apply_react_config(inner: &ReActAgent, config: &DeepAgentConfig, prompt_text: &str) {
    let mut react = inner.config();
    react.model_name = config.model_name.clone().unwrap_or_default();
    react.model_provider = config
        .model_provider
        .clone()
        .unwrap_or_else(|| "openai".to_string());
    react.api_key.clone_from(&config.api_key);
    react.api_base.clone_from(&config.api_base);
    react.custom_headers = config.custom_headers.clone();
    react.max_iterations = resolve_inner_react_max_iterations(config.max_iterations);
    react.max_iterations_hard_limit = config.max_iterations_hard_limit.map(|value| value as usize);
    react.on_max_iterations = config.on_max_iterations;
    react.parallel_tool_calls = config.parallel_tool_calls;
    react
        .model_client_config
        .clone_from(&config.model_client_config);
    react.model_config.clone_from(&config.model_config);
    react.configure_prompt_template(vec![BaseMessage::system(prompt_text)]);
    inner.configure(react);
}

/// Finishes the wiring of a freshly constructed agent or of a lazy one.
///
/// Mounts `tools`, registers every harness rail through a
/// [`HarnessRailAdapter`](crate::rails::HarnessRailAdapter), registers the
/// permission interrupt rail when a permission engine exists, and applies the
/// inner [`ReActAgentConfig`].
#[allow(clippy::unnecessary_wraps)] // kept fallible for the factory layer (next wave)
pub(crate) fn wire(driver: &Driver, tools: Vec<Arc<dyn Tool>>) -> Result<()> {
    let config = driver.config.read().clone();
    apply_react_config(&driver.inner, &config, &driver.prompt.system_text());
    if !tools.is_empty() {
        driver.inner.add_tools(tools);
    }
    for rail in driver.rails.to_agent_rails() {
        driver.inner.register_rail(rail);
    }
    if let Some(engine) = &driver.permission {
        driver.inner.register_rail(Arc::new(
            PermissionInterruptRail::new(engine.clone()).with_host(driver.permission_host.clone()),
        ));
    }
    driver.wired.store(true, Ordering::Release);
    Ok(())
}

/// Resolves a model lazily from the configuration when none was supplied.
///
/// A supplied model short-circuits the function. Otherwise the client and
/// request configurations come from the explicit `model_client_config` /
/// `model_config`, falling back to the flat `model_name`, `model_provider`,
/// `api_base` and `api_key` fields. A configuration with neither a model nor a
/// model identity fails with [`Error::config`].
///
/// # Errors
///
/// Returns [`Error::config`] when the model identity is missing or
/// [`init_model`] rejects the resolved provider/configuration.
pub(crate) async fn ensure_agent_wired(driver: &Driver) -> Result<()> {
    if driver.model.read().is_some() {
        driver.wired.store(true, Ordering::Release);
        return Ok(());
    }
    let config = driver.config.read().clone();
    let request = if let Some(request) = config.model_config.clone() {
        request
    } else {
        let name = config.model_name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(Error::config(
                "DeepAgent requires a model: supply one via the builder or configure \
                 model_name/model_provider.",
            ));
        }
        ModelRequestConfig::new(name)
    };
    let client = if let Some(client) = config.model_client_config.clone() {
        client
    } else {
        let provider = config
            .model_provider
            .clone()
            .unwrap_or_else(|| "openai".to_string());
        let mut client =
            ModelClientConfig::new(provider).with_headers(config.custom_headers.clone());
        if let Some(api_base) = &config.api_base {
            client = client.with_api_base(api_base.clone());
        }
        if let Some(api_key) = &config.api_key {
            client = client.with_secret(api_key.clone());
        }
        client
    };
    let model = init_model(client, request).await.map_err(|error| {
        Error::config(format!("failed to initialise the DeepAgent model: {error}"))
    })?;
    *driver.model.write() = Some(model.clone());
    driver.inner.set_model(model);
    driver.wired.store(true, Ordering::Release);
    Ok(())
}

/// Rebuilds the prompt sections of an already constructed agent.
///
/// Keeps the existing [`PromptState`] (and therefore its rail wiring) but
/// replaces its builder, so the language and prompt mode of `config` apply.
pub(crate) fn replace_prompt_state(
    prompt: &PromptState,
    config: &DeepAgentConfig,
    cards: &[ToolCard],
) {
    let builder = build_prompt_builder(config, cards);
    prompt.update(|current| *current = builder);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_agent::AgentCard;
    use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
    use serde_json::json;

    use super::{build_prompt_builder, DeepAgentBuilder};
    use crate::schema::{AgentMode, DeepAgentConfig};
    use crate::tools::goal::{GetCurrentGoalTool, GoalStore};
    use crate::tools::tool_card;
    use crate::workspace::Workspace;

    fn card() -> AgentCard {
        AgentCard::new("builder-agent").with_description("Builder identity")
    }

    #[test]
    fn prompt_builder_contains_identity_workspace_and_tools() {
        let workspace = Workspace::for_language("./work", "en");
        let config = DeepAgentConfig::default()
            .with_card(card())
            .with_language("en".to_string())
            .with_workspace(workspace.clone());
        let cards = vec![tool_card(
            "echo",
            "Echo the input back",
            json!({"type": "object"}),
            true,
        )];
        let text = build_prompt_builder(&config, &cards).build();
        assert!(text.contains("Builder identity"));
        assert!(text.contains("work"));
        assert!(text.contains("echo"));
    }

    #[test]
    fn prompt_builder_adds_the_plan_section_only_in_plan_mode() {
        let config = DeepAgentConfig::default()
            .with_language("en".to_string())
            .with_default_mode(AgentMode::Plan);
        assert!(build_prompt_builder(&config, &[])
            .build()
            .contains("Plan Mode"));
        let normal = DeepAgentConfig::default().with_language("en".to_string());
        assert!(!build_prompt_builder(&normal, &[])
            .build()
            .contains("Plan Mode"));
    }

    #[tokio::test]
    async fn build_sets_the_inner_prompt_and_tool_cards() {
        let workspace = Workspace::new(".");
        let agent = DeepAgentBuilder::new(card())
            .config(DeepAgentConfig::default().with_workspace(workspace))
            .model(std::sync::Arc::new(ScriptedModel::text("hi")))
            .build()
            .unwrap();
        let template = agent.inner.config().prompt_template;
        assert_eq!(template.len(), 1);
        assert!(template[0].text().contains("Builder identity"));
        assert!(!agent.tool_cards.read().is_empty());
        assert!(agent.wired.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn build_rejects_an_invalid_configuration() {
        let config = DeepAgentConfig::default()
            .with_enable_task_loop(true)
            .with_max_iterations_hard_limit(Some(3));
        let error = DeepAgentBuilder::new(card())
            .config(config)
            .build()
            .err()
            .expect("invalid configuration must fail");
        assert!(error.to_string().contains("max_iterations_hard_limit"));
    }

    #[tokio::test]
    async fn build_registers_the_goal_rail_when_a_goal_tool_is_mounted() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let goal_tool = GetCurrentGoalTool::new(GoalStore::for_workspace(&workspace), "en");
        let agent = DeepAgentBuilder::new(card())
            .config(DeepAgentConfig::default().with_workspace(workspace))
            .model(std::sync::Arc::new(ScriptedModel::text("hi")))
            .tool(std::sync::Arc::new(goal_tool))
            .build()
            .unwrap();

        assert!(
            agent.rails().names().iter().any(|name| name == "goal"),
            "the goal rail joins the default rails when a goal tool is mounted: {:?}",
            agent.rails().names()
        );
    }

    #[tokio::test]
    async fn build_skips_the_goal_rail_without_goal_tools() {
        let temp = TempWorkspace::new().unwrap();
        let workspace = Workspace::for_language(temp.workspace(), "en");
        let agent = DeepAgentBuilder::new(card())
            .config(DeepAgentConfig::default().with_workspace(workspace))
            .model(std::sync::Arc::new(ScriptedModel::text("hi")))
            .build()
            .unwrap();

        assert!(
            !agent.rails().names().iter().any(|name| name == "goal"),
            "no goal tool is mounted by default: {:?}",
            agent.rails().names()
        );
    }
}
