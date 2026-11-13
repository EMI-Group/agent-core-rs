//! Validation and materialization of a [`DeepAgentSpec`] into harness parts.
//!
//! [`resolve`] is the pure data step: it validates the template, builds the
//! [`AgentCard`], resolves the declared tools to [`BuiltinTool`] values, mounts
//! the enabled skill roots, turns sub-agent declarations into
//! [`SubAgentConfig`] values and normalizes the rail names.
//!
//! [`build_agent`] is the hand-off to the existing product surface: it assembles
//! the inputs of [`create_deep_agent`] (or [`ResolvedSpec::into_builder`] for
//! the builder path) without changing their signatures. Built-in `DEFAULT`
//! tools are always mounted by the builder itself, so only the template's
//! *extended* tools are materialized as extra tool instances; this is a
//! documented deviation from a fully exclusive tool list.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::tool::Tool;

use crate::agent::{create_deep_agent, resolve_deep_agent_parts, DeepAgent, DeepAgentBuilder};
use crate::permission::PermissionHost;
use crate::rails::HarnessRail;
use crate::schema::{DeepAgentConfig, SubAgentConfig};
use crate::skills::SkillRootMount;
use crate::subagents::{build_preset, SubAgentPreset};
use crate::tools::{BuiltinTool, ToolRegistry};
use crate::workspace::Workspace;

use super::spec::{normalize_rail, DeepAgentSpec, ModelSpec, SubAgentSpec};

/// A skill root declared by a template, after resolution against its base directory.
#[derive(Debug, Clone)]
pub struct ResolvedSkill {
    /// Absolute or base-resolved root path.
    pub root: PathBuf,
    /// Whether the template enabled this root.
    pub enabled: bool,
    /// Ownership handle of the mounted root; absent for a disabled root.
    pub mount: Option<SkillRootMount>,
}

/// A validated template together with every harness input it produces.
///
/// The value is plain data plus the owned skill-root mounts; no model, tool
/// instance or live rail is embedded. [`Self::workspace`] applies the template's
/// layout hints, [`Self::to_config`] maps the data onto a [`DeepAgentConfig`],
/// and [`build_agent`] hands the result to `create_deep_agent`.
#[derive(Debug, Clone)]
pub struct ResolvedSpec {
    /// The validated source template.
    pub spec: DeepAgentSpec,
    /// Identity card built from `id`/`name`/`description`.
    pub card: AgentCard,
    /// Prompt fragments in rendering order (ascending priority, stable).
    pub prompt_fragments: Vec<super::spec::PromptFragment>,
    /// Combined system prompt: the explicit prompt followed by the fragments.
    pub system_prompt: Option<String>,
    /// Declared tools in template order.
    pub tools: Vec<BuiltinTool>,
    /// Skill roots in template order, including disabled ones.
    pub skills: Vec<ResolvedSkill>,
    /// Roots actually mounted, in first-declaration order.
    pub skill_roots: Vec<PathBuf>,
    /// Sub-agent configurations in template order.
    pub subagents: Vec<SubAgentConfig>,
    /// Canonical rail names in template order.
    pub rails: Vec<String>,
    /// Effective prompt language.
    pub language: String,
    /// Model hints, when declared.
    pub model: Option<ModelSpec>,
}

impl ResolvedSpec {
    /// Builds the workspace of this template: the hinted root and language plus
    /// the extra directory nodes, layered over the default bilingual schema.
    #[must_use]
    pub fn workspace(&self) -> Workspace {
        let root = self
            .spec
            .workspace
            .root
            .clone()
            .unwrap_or_else(|| "./".to_string());
        let language = self
            .spec
            .workspace
            .language
            .clone()
            .unwrap_or_else(|| self.language.clone());
        let mut workspace = Workspace::for_language(root, language);
        for directory in &self.spec.workspace.directories {
            if !workspace
                .directories
                .iter()
                .any(|node| node.name == *directory)
            {
                workspace
                    .directories
                    .push(crate::workspace::DirectoryNode::dir(
                        directory.clone(),
                        directory.clone(),
                    ));
            }
        }
        workspace
    }

    /// Returns the mounted skill roots as string paths, the shape
    /// [`DeepAgentConfig::skills`] expects.
    #[must_use]
    pub fn enabled_skill_roots(&self) -> Vec<String> {
        self.skill_roots
            .iter()
            .map(|root| root.to_string_lossy().into_owned())
            .collect()
    }

    /// Maps the resolved data onto a [`DeepAgentConfig`].
    ///
    /// `workspace` is stored verbatim. The security rail is enabled when
    /// `security` is declared, disabled when a non-empty rail list omits it, and
    /// left at its default for an empty list; the `skills` and `subagent` rails
    /// are mounted by the factory from the configured skills/sub-agents.
    #[must_use]
    pub fn to_config(&self, workspace: Workspace) -> DeepAgentConfig {
        let mut config = DeepAgentConfig::default()
            .with_card(self.card.clone())
            .with_workspace(workspace)
            .with_language(self.language.clone())
            .with_system_prompt(self.system_prompt.clone())
            .with_skills(self.enabled_skill_roots())
            .with_subagents(self.subagents.clone())
            .with_enable_task_loop(self.spec.task_loop)
            .with_max_iterations(self.spec.max_iterations)
            .with_restrict_to_work_dir(self.spec.restrict_to_work_dir.unwrap_or(true));
        if let Some(mode) = self.spec.prompt_mode {
            config = config.with_prompt_mode(mode);
        }
        if !self.rails.is_empty() {
            config =
                config.with_enable_security_rail(self.rails.iter().any(|rail| rail == "security"));
        }
        if let Some(model) = &self.model {
            config = apply_model_hints(config, model);
        }
        config
    }

    /// Materializes the template's extended built-in tools as tool instances.
    ///
    /// The builder always mounts [`BuiltinTool::DEFAULT`]; only tools outside
    /// that set are returned here so the mounted prompt never lists a tool
    /// twice.
    #[must_use]
    pub fn supplemental_tools(&self, workspace: &Workspace) -> Vec<Arc<dyn Tool>> {
        let extras: Vec<BuiltinTool> = self
            .tools
            .iter()
            .copied()
            .filter(|tool| !BuiltinTool::DEFAULT.contains(tool))
            .collect();
        if extras.is_empty() {
            return Vec::new();
        }
        ToolRegistry::new(workspace.clone())
            .with_tools(extras)
            .with_language(self.language.clone())
            .build()
    }

    /// Builds a [`DeepAgentBuilder`] wired with the resolved configuration,
    /// default rails and supplemental tools.
    ///
    /// The caller supplies the live model and workspace; everything else comes
    /// from the template. Call [`DeepAgentBuilder::build`] to obtain the agent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the assembled configuration is invalid or
    /// the default rail set cannot be resolved.
    pub fn into_builder(
        self,
        model: Arc<dyn Model>,
        workspace: Workspace,
    ) -> Result<DeepAgentBuilder> {
        let config = self.to_config(workspace.clone());
        let tools = self.supplemental_tools(&workspace);
        let parts = resolve_deep_agent_parts(config, Some(model.clone()), tools, Vec::new())?;
        let mut builder = DeepAgentBuilder::new(self.card.clone())
            .config(parts.config)
            .model(model)
            .workspace(workspace)
            .tools(parts.tools);
        for rail in parts.rails {
            builder = builder.rail(rail);
        }
        Ok(builder)
    }
}

/// Options of [`build_agent`].
#[derive(Default)]
pub struct SpecBuildOptions {
    /// Workspace override; `None` uses [`ResolvedSpec::workspace`].
    pub workspace: Option<Workspace>,
    /// Host answering human-in-the-loop permission confirmations.
    pub permission_host: Option<Arc<dyn PermissionHost>>,
    /// Extra tools appended after the template's supplemental tools.
    pub extra_tools: Vec<Arc<dyn Tool>>,
    /// Extra rails registered after the factory's default rails.
    pub extra_rails: Vec<Arc<dyn HarnessRail>>,
}

/// Builds a `DeepAgent` from a resolved template through `create_deep_agent`.
///
/// The live model comes from the caller; the card, configuration, workspace,
/// skills, sub-agents, task-loop switch, iteration cap, language, prompt mode
/// and permission section come from the template. The helper only constructs
/// the documented `create_deep_agent` inputs — the factory signature is
/// unchanged.
///
/// # Errors
///
/// Propagates [`Error::config`] from the assembled configuration or the agent
/// wiring.
pub async fn build_agent(
    resolved: ResolvedSpec,
    model: Arc<dyn Model>,
    options: SpecBuildOptions,
) -> Result<Arc<DeepAgent>> {
    let SpecBuildOptions {
        workspace,
        permission_host,
        extra_tools,
        extra_rails,
    } = options;
    let workspace = workspace.unwrap_or_else(|| resolved.workspace());
    let config = resolved.to_config(workspace.clone());
    let mut tools = resolved.supplemental_tools(&workspace);
    tools.extend(extra_tools);
    let skills = resolved.enabled_skill_roots();
    let subagents = resolved.subagents.clone();
    let language = Some(resolved.language.clone());
    let prompt_mode = resolved.spec.prompt_mode;
    let max_iterations = resolved.spec.max_iterations;
    create_deep_agent(
        model,
        Some(resolved.card.clone()),
        Some(config),
        tools,
        extra_rails,
        Some(workspace),
        skills,
        subagents,
        resolved.spec.task_loop,
        max_iterations,
        language,
        prompt_mode,
        resolved.spec.restrict_to_work_dir.unwrap_or(true),
        None,
        permission_host,
    )
    .await
}

/// Resolves `spec` against `workspace`, using the template's own directory (or
/// the workspace root for a built-in template) as the base of relative paths.
///
/// # Errors
///
/// Propagates the validation errors of [`DeepAgentSpec::validate`] and
/// [`resolve`].
pub fn resolve_with_source(spec: &DeepAgentSpec, workspace: &Workspace) -> Result<ResolvedSpec> {
    let base = spec
        .source_path
        .as_deref()
        .and_then(Path::parent)
        .map_or_else(|| workspace.root_path.clone(), Path::to_path_buf);
    resolve(spec, workspace, &base)
}

/// Resolves `spec` against `workspace`, resolving relative skill roots against
/// `base_dir`.
///
/// # Errors
///
/// Returns [`Error::validation`] when the template is invalid, a sub-agent
/// preset is unknown or unsupported, or a sub-agent model hint is malformed.
pub fn resolve(
    spec: &DeepAgentSpec,
    workspace: &Workspace,
    base_dir: &Path,
) -> Result<ResolvedSpec> {
    spec.validate()?;
    let language = spec
        .language
        .clone()
        .unwrap_or_else(|| workspace.language.clone());
    let card = build_card(spec);
    let (prompt_fragments, system_prompt) = build_prompts(spec);
    let tools = spec
        .tools
        .iter()
        .filter_map(|name| BuiltinTool::from_name(name))
        .collect();
    let mut skill_roots: Vec<PathBuf> = Vec::new();
    let mut skills = Vec::new();
    for declared in &spec.skills {
        let root = resolve_path(base_dir, &declared.root);
        let mount = if declared.enabled {
            let mount = SkillRootMount::mount(&mut skill_roots, root.clone());
            Some(mount)
        } else {
            None
        };
        skills.push(ResolvedSkill {
            root,
            enabled: declared.enabled,
            mount,
        });
    }
    let mut subagents = Vec::new();
    for declared in &spec.subagents {
        subagents.push(resolve_subagent(declared, &language, spec)?);
    }
    let mut rails: Vec<String> = Vec::new();
    for rail in &spec.rails {
        let canonical = normalize_rail(rail)?;
        if !rails.iter().any(|known| known == canonical) {
            rails.push(canonical.to_string());
        }
    }
    Ok(ResolvedSpec {
        spec: spec.clone(),
        card,
        prompt_fragments,
        system_prompt,
        tools,
        skills,
        skill_roots,
        subagents,
        rails,
        language,
        model: spec.model.clone(),
    })
}

/// Builds the identity card of a template.
fn build_card(spec: &DeepAgentSpec) -> AgentCard {
    let card = AgentCard::new(spec.name.clone()).with_description(spec.description.clone());
    if spec.id.is_empty() {
        card
    } else {
        card.with_id(spec.id.clone())
    }
}

/// Sorts the prompt fragments and combines them with the explicit prompt.
fn build_prompts(spec: &DeepAgentSpec) -> (Vec<super::spec::PromptFragment>, Option<String>) {
    let mut fragments = spec.prompt_sections.clone();
    fragments.sort_by_key(|fragment| fragment.priority);
    let mut parts = Vec::new();
    if let Some(prompt) = spec.system_prompt.as_ref().filter(|text| !text.is_empty()) {
        parts.push(prompt.clone());
    }
    for fragment in &fragments {
        if !fragment.content.is_empty() {
            parts.push(fragment.content.clone());
        }
    }
    let system_prompt = if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    };
    (fragments, system_prompt)
}

/// Resolves one declared sub-agent into a [`SubAgentConfig`].
fn resolve_subagent(
    declared: &SubAgentSpec,
    language: &str,
    owner: &DeepAgentSpec,
) -> Result<SubAgentConfig> {
    let mut config = match &declared.preset {
        Some(preset) => {
            let parsed = parse_preset(preset)?;
            build_preset(parsed, declared.language.as_deref().unwrap_or(language))?
        }
        None => SubAgentConfig::new(
            AgentCard::new(declared.name.clone()).with_description(declared.description.clone()),
        )
        .with_language(Some(
            declared
                .language
                .clone()
                .unwrap_or_else(|| language.to_string()),
        )),
    };
    config.agent_card.name.clone_from(&declared.name);
    if !declared.description.trim().is_empty() {
        config
            .agent_card
            .description
            .clone_from(&declared.description);
        config.role = Some(declared.description.clone());
    }
    if let Some(prompt) = &declared.system_prompt {
        config.system_prompt = Some(prompt.clone());
    }
    if !declared.tools.is_empty() {
        config.tools.clone_from(&declared.tools);
    }
    if let Some(iterations) = declared.max_iterations {
        config.max_iterations = Some(iterations);
    }
    if let Some(language) = &declared.language {
        config.language = Some(language.clone());
    }
    if config.subagent_type.is_none() {
        config.subagent_type = Some(declared.name.clone());
    }
    if let Some(model) = &declared.model {
        let value = serde_json::to_value(model).map_err(|error| {
            Error::validation(format!(
                "sub-agent '{}' model hints of {} are not serializable: {error}",
                declared.name,
                owner.label()
            ))
        })?;
        config.factory_kwargs.insert("model".to_string(), value);
    }
    if !declared.metadata.is_empty() {
        config.factory_kwargs.insert(
            "metadata".to_string(),
            serde_json::Value::Object(declared.metadata.clone().into_iter().collect()),
        );
    }
    Ok(config)
}

/// Parses a built-in sub-agent preset name.
///
/// Both the bare name (`explore`) and the Python registry name
/// (`explore_agent`) are accepted. Feature-gated presets are only available
/// when their cargo feature is enabled; otherwise they are reported as unknown.
///
/// # Errors
///
/// Returns [`Error::validation`] listing the known preset names.
pub(crate) fn parse_preset(raw: &str) -> Result<SubAgentPreset> {
    match raw.trim() {
        "code" | "code_agent" => Ok(SubAgentPreset::Code),
        "explore" | "explore_agent" => Ok(SubAgentPreset::Explore),
        "plan" | "plan_agent" => Ok(SubAgentPreset::Plan),
        "research" | "research_agent" => Ok(SubAgentPreset::Research),
        "verification" | "verification_agent" => Ok(SubAgentPreset::Verification),
        other => {
            let known: Vec<&str> = SubAgentPreset::builtin()
                .iter()
                .map(SubAgentPreset::as_str)
                .collect();
            Err(Error::validation(format!(
                "unknown sub-agent preset '{other}'; known presets: {}",
                known.join(", ")
            )))
        }
    }
}

/// Resolves a possibly relative path against `base_dir`.
fn resolve_path(base_dir: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

/// Applies [`ModelSpec`] hints onto a configuration.
fn apply_model_hints(mut config: DeepAgentConfig, model: &ModelSpec) -> DeepAgentConfig {
    config.model_provider.clone_from(&model.provider);
    config.api_base.clone_from(&model.api_base);
    if model.model.is_some() {
        config.model_name.clone_from(&model.model);
    }
    if let Some(name) = &model.model {
        let mut request = ModelRequestConfig::new(name.clone());
        request.temperature = model.temperature;
        request.top_p = model.top_p;
        request.max_tokens = model.max_tokens;
        config = config.with_model_config(Some(request));
    }
    config
}
