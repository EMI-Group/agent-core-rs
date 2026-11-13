//! The prompt-pack schema and the apply/load/serialize methods.
//!
//! A [`PromptPack`] is plain data: [`PromptPack::from_yaml`] (or
//! [`PromptPack::from_yaml_file`]) parses one YAML document, and
//! [`PromptPack::apply_to`] copies it onto a live
//! [`ReActAgent`]. The pack never owns runtime state, so the
//! same value can be applied to several agents or serialized back with
//! [`PromptPack::to_yaml`].
use super::rail::{PromptSectionsRail, PROMPT_SECTIONS_RAIL_NAME};
use super::rails::RailSpec;
use crate::policy::{McpToolAllowlist, ToolExposurePolicy, ToolResiliencePolicy};
use crate::react::ReActAgent;
use crate::skills::SkillManager;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, Model};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Priority a [`SectionSpec`] gets when the pack does not set one.
pub const DEFAULT_SECTION_PRIORITY: i32 = 50;

/// One agent configuration document: prompts, tool policy, and rails.
///
/// Only [`PromptPack::id`] is required. The type is a plain data holder; see
/// the [module docs](self) for the YAML format and the apply semantics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptPack {
    /// Stable pack id; required.
    pub id: String,
    /// Human-readable pack name; also the identity prompt's subject.
    pub name: String,
    /// One-line description; appended to the identity prompt.
    pub description: String,
    /// The prompt messages and the extra system sections.
    pub prompts: PromptSpec,
    /// The tool policy the pack installs.
    pub tools: ToolSpec,
    /// The rails the pack asks the host to install.
    pub rails: Vec<RailSpec>,
    /// Free-form pack metadata; never applied to the agent.
    pub metadata: BTreeMap<String, Value>,
}

impl PromptPack {
    /// Creates an empty pack with `id`.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Parses one YAML document into a pack.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document is not valid YAML, is not
    /// a mapping, or has no `id`. The message names the problem (and the YAML
    /// parser's location when there is one).
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let pack: Self = serde_yaml_ng::from_str(yaml)
            .map_err(|error| Error::validation(format!("invalid prompt pack YAML: {error}")))?;
        pack.validate()?;
        Ok(pack)
    }

    /// Reads and parses the YAML document at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when the file cannot be read, and the
    /// [`PromptPack::from_yaml`] errors when its contents are not a valid pack.
    pub async fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let yaml = tokio::fs::read_to_string(path).await.map_err(|error| {
            Error::sys_op(format!(
                "cannot read prompt pack {}: {error}",
                path.display()
            ))
        })?;
        Self::from_yaml(&yaml)
    }

    /// Serializes the pack back to one YAML document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pack cannot be represented as YAML.
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml_ng::to_string(self).map_err(|error| {
            Error::validation(format!("cannot serialize prompt pack {}: {error}", self.id))
        })
    }

    /// Returns the rail specs the pack declares.
    #[must_use]
    pub fn rail_specs(&self) -> &[RailSpec] {
        &self.rails
    }

    /// The resolved tool-exposure policy: the pack's policy, or allow-all when
    /// the pack does not set one.
    #[must_use]
    pub fn to_tool_policy(&self) -> ToolExposurePolicy {
        self.tools
            .policy
            .clone()
            .unwrap_or_else(ToolExposurePolicy::allow_all)
    }

    /// The identity message synthesized from [`PromptPack::name`] and
    /// [`PromptPack::description`], or `None` when both are blank.
    ///
    /// A name alone yields `You are <name>.`, a description alone yields the
    /// description, and both yield `You are <name>.\n\n<description>`.
    #[must_use]
    pub fn identity_message(&self) -> Option<BaseMessage> {
        let name = self.name.trim();
        let description = self.description.trim();
        match (name.is_empty(), description.is_empty()) {
            (true, true) => None,
            (false, true) => Some(BaseMessage::system(format!("You are {name}."))),
            (true, false) => Some(BaseMessage::system(description.to_owned())),
            (false, false) => Some(BaseMessage::system(format!(
                "You are {name}.\n\n{description}"
            ))),
        }
    }

    /// Applies the pack to `agent` without host-supplied objects.
    ///
    /// Equivalent to [`PromptPack::apply_to_with`] with a default
    /// [`PromptPackContext`]: `skill`, `model_backup`, and `custom` rails are
    /// reported as deferred.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pack has no `id`.
    pub fn apply_to(&self, agent: &ReActAgent) -> Result<AppliedPack> {
        self.apply_to_with(agent, &PromptPackContext::default())
    }

    /// Applies the pack to `agent`, building the rails `ctx` can supply.
    ///
    /// The agent's prompt template, tool policies, and rails are updated; every
    /// other configuration field is preserved. The returned [`AppliedPack`]
    /// lists the registered rails and the specs that need a host object the
    /// context did not carry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pack has no `id`.
    pub fn apply_to_with(
        &self,
        agent: &ReActAgent,
        ctx: &PromptPackContext,
    ) -> Result<AppliedPack> {
        self.validate()?;

        let mut config = agent.config();
        if let Some(template) = self.resolved_template(&config.prompt_template) {
            config.prompt_template = template;
            agent.configure(config);
        }
        if let Some(policy) = &self.tools.policy {
            agent.set_tool_exposure_policy(policy.clone());
        }
        if let Some(allowlist) = &self.tools.mcp_allowlist {
            agent.set_mcp_tool_allowlist(allowlist.clone());
        }
        if let Some(resilience) = &self.tools.resilience {
            agent.set_tool_resilience(resilience.clone());
        }

        let mut applied = AppliedPack::default();
        if !self.prompts.sections.is_empty() {
            let rail = PromptSectionsRail::new(self.prompts.sections.clone());
            agent.register_rail(Arc::new(rail));
            applied
                .rails_registered
                .push(PROMPT_SECTIONS_RAIL_NAME.to_owned());
        }
        for spec in &self.rails {
            match spec.build(ctx) {
                Some(rail) => {
                    let name = rail.name().to_owned();
                    agent.register_rail(rail);
                    if !applied.rails_registered.contains(&name) {
                        applied.rails_registered.push(name);
                    }
                }
                None => applied.deferred_rails.push(spec.clone()),
            }
        }
        tracing::debug!(
            target: "openjiuwen_agent::prompt_pack",
            pack = %self.id,
            registered = applied.rails_registered.len(),
            deferred = applied.deferred_rails.len(),
            "applied the prompt pack"
        );
        Ok(applied)
    }

    /// The template the pack installs over `existing`: the pack's own template
    /// when it declares one, otherwise the synthesized identity message
    /// prepended to `existing`, otherwise `None` (leave the config alone).
    fn resolved_template(&self, existing: &[BaseMessage]) -> Option<Vec<BaseMessage>> {
        if !self.prompts.template.is_empty() {
            return Some(self.prompts.template.clone());
        }
        self.identity_message().map(|identity| {
            std::iter::once(identity)
                .chain(existing.iter().cloned())
                .collect()
        })
    }

    /// Rejects a pack without an identity.
    fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(Error::validation(
                "prompt pack is missing the required `id` field",
            ));
        }
        Ok(())
    }
}

/// The prompt half of a pack.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptSpec {
    /// Messages copied into
    /// [`ReActAgentConfig::prompt_template`](crate::react::ReActAgentConfig::prompt_template).
    pub template: Vec<BaseMessage>,
    /// Extra system sections injected before every model call.
    pub sections: Vec<SectionSpec>,
}

/// One extra system section of a pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SectionSpec {
    /// Section heading; rendered as `## <name>`.
    pub name: String,
    /// Section body.
    pub content: String,
    /// Ordering key; lower priorities render first.
    pub priority: i32,
}

impl Default for SectionSpec {
    /// An unnamed, empty section at [`DEFAULT_SECTION_PRIORITY`].
    fn default() -> Self {
        Self {
            name: String::new(),
            content: String::new(),
            priority: DEFAULT_SECTION_PRIORITY,
        }
    }
}

impl SectionSpec {
    /// Builds a section at [`DEFAULT_SECTION_PRIORITY`].
    #[must_use]
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            ..Self::default()
        }
    }

    /// Sets [`SectionSpec::priority`]; a consuming builder.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// The tool-policy half of a pack; every field is optional.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolSpec {
    /// Which tools the model may see.
    pub policy: Option<ToolExposurePolicy>,
    /// Per-MCP-server allowlists of underlying tool names.
    pub mcp_allowlist: Option<McpToolAllowlist>,
    /// Per-tool timeouts, attempt budgets, and backoff.
    pub resilience: Option<ToolResiliencePolicy>,
}

/// Host objects a pack may need to build its rails.
///
/// [`PromptPack::apply_to_with`] consumes the context; the context itself owns
/// nothing the pack mutates, so it can be reused across agents.
#[derive(Clone, Default)]
pub struct PromptPackContext {
    /// Skill library the `skill` rails advertise.
    pub skills: Option<SkillManager>,
    /// Backup models a `model_backup` rail tries in order.
    pub models: Vec<Arc<dyn Model>>,
}

impl PromptPackContext {
    /// Creates an empty context: every pack rail that needs a host object is
    /// deferred.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Supplies the skill library; a consuming builder.
    #[must_use]
    pub fn with_skills(mut self, skills: SkillManager) -> Self {
        self.skills = Some(skills);
        self
    }

    /// Supplies the backup models; a consuming builder.
    #[must_use]
    pub fn with_models(mut self, models: impl IntoIterator<Item = Arc<dyn Model>>) -> Self {
        self.models = models.into_iter().collect();
        self
    }
}

// `Arc<dyn Model>` is not `Debug`, so the models are summarized by count.
impl fmt::Debug for PromptPackContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptPackContext")
            .field("skills", &self.skills.as_ref().map(SkillManager::len))
            .field("models", &self.models.len())
            .finish()
    }
}

/// The outcome of applying a pack to an agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AppliedPack {
    /// Names of the rails that were registered, in registration order.
    pub rails_registered: Vec<String>,
    /// Specs that could not be built from the context and are the host's job.
    pub deferred_rails: Vec<RailSpec>,
}

impl AppliedPack {
    /// Whether every rail the pack declared was registered.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.deferred_rails.is_empty()
    }
}

#[cfg(test)]
mod tests;
