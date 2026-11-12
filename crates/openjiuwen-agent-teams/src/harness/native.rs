//! The native (in-process) member harness factory.
//!
//! Python parity source: `openjiuwen/agent_teams/harness/native_harness.py` construction and
//! `openjiuwen/agent_teams/harness/protocol_adapter.py`'s "build a `NativeHarness` from the cold
//! `DeepAgentSpec`" path. In this port [`openjiuwen_harness::NativeHarness`] already implements the
//! provider-neutral [`HarnessProtocol`], so the factory only has to materialize the
//! [`openjiuwen_harness::DeepAgent`] and wrap it.
//!
//! Model resolution is injected through [`ModelResolver`] so tests can run without a provider and
//! a future model-pool allocator can pick a model per member without changing the factory.
//!
//! ## Spec → harness mapping
//!
//! | Spec field | Harness counterpart |
//! | --- | --- |
//! | `card` | builder card, else `AgentCard::new(member_name)` |
//! | `system_prompt` | `DeepAgentConfig::system_prompt` |
//! | `model` | [`ModelResolver::resolve`] → `DeepAgentBuilder::model` |
//! | `enable_task_loop` | `DeepAgentConfig::enable_task_loop` |
//! | `max_iterations` | `DeepAgentConfig::max_iterations` |
//! | `workspace.root_path` | runtime `workspace_root` → `Workspace` |
//! | `cwd` / `project_root` | `DeepAgentConfig::{cwd, project_root}` |
//! | `skills`, `language`, `prompt_mode` | matching `DeepAgentConfig` fields |
//! | `enable_security_rail`, `completion_timeout` | matching `DeepAgentConfig` fields |
//! | team `teammate_mode` | `default_mode`/`enable_plan_mode` (`plan_mode` ⇒ plan-first) |
//!
//! Fields without a factory-level counterpart (`tools`, `mcps`, `subagents`, `rails`,
//! `approval_required_tools`, model `provider`/`params`) are ignored here and logged once at
//! `debug`; mounting tools/MCP servers/rails belongs to the runtime wiring that owns live
//! handles, not to this cold-blueprint factory.
use super::adapter::{MemberHarnessContext, MemberHarnessFactory};
use crate::schema::{DeepAgentSpec, MemberMode, ModelRef};
use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{init_model, Model, ModelClientConfig, ModelRequestConfig};
use openjiuwen_harness::{
    AgentMode, DeepAgentBuilder, DeepAgentConfig, NativeHarness, NativeHarnessOptions, PromptMode,
    Workspace,
};
use openjiuwen_harness_protocol::HarnessProtocol;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;
/// Static identity of the native member factory.
pub const NATIVE_FACTORY_NAME: &str = "native";
/// Resolves the model a member runs on.
///
/// The resolver is async because building a client may perform provider discovery; it takes the
/// spec's model name so a model-pool allocator can override the default.
#[async_trait]
pub trait ModelResolver: Send + Sync {
    /// Resolves `model_name` (or the resolver's default) to a ready model handle.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Config`] when no provider or model is
    /// configured, and the LLM layer's own error when the client cannot be built.
    async fn resolve(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>>;
}
/// Environment-driven [`ModelResolver`].
///
/// Reads `MODEL_PROVIDER` and `MODEL_NAME` here and delegates the rest to
/// [`ModelClientConfig::from_env`] (which covers `API_KEY`/`OPENAI_API_KEY` and
/// `API_BASE`/`OPENAI_BASE_URL`, plus the `OPENJIUWEN_*` spellings). Explicit values supplied
/// through the builder win over the environment.
#[derive(Debug, Clone, Default)]
pub struct EnvModelResolver {
    /// Provider override; `None` falls back to `MODEL_PROVIDER`.
    provider: Option<String>,
    /// Model-name override; `None` falls back to `MODEL_NAME` (then the spec's model name wins).
    model_name: Option<String>,
    /// API-key override; `None` falls back to the environment.
    api_key: Option<String>,
    /// API-base override; `None` falls back to the environment.
    api_base: Option<String>,
}
impl EnvModelResolver {
    /// Builds a resolver that reads only the environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self::default()
    }
    /// Overrides the provider name.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }
    /// Overrides the default model name.
    #[must_use]
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }
    /// Overrides the API key.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }
    /// Overrides the API base URL.
    #[must_use]
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = Some(api_base.into());
        self
    }
}
/// Returns a non-empty environment value, treating blanks as unset.
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}
#[async_trait]
impl ModelResolver for EnvModelResolver {
    async fn resolve(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let provider = self
            .provider
            .clone()
            .or_else(|| non_empty_env("MODEL_PROVIDER"))
            .ok_or_else(|| {
                Error::config("model provider is not configured; set MODEL_PROVIDER or pass one")
            })?;
        let name = model_name
            .map(str::to_owned)
            .or_else(|| self.model_name.clone())
            .or_else(|| non_empty_env("MODEL_NAME"))
            .ok_or_else(|| {
                Error::config(
                    "model name is not configured; set MODEL_NAME or name one in the spec",
                )
            })?;
        let mut client = ModelClientConfig::from_env();
        client.provider = provider;
        if let Some(api_key) = self
            .api_key
            .clone()
            .or_else(|| non_empty_env("API_KEY"))
            .or_else(|| non_empty_env("OPENAI_API_KEY"))
        {
            client = client.with_api_key(api_key);
        }
        if let Some(api_base) = self
            .api_base
            .clone()
            .or_else(|| non_empty_env("API_BASE"))
            .or_else(|| non_empty_env("OPENAI_BASE_URL"))
        {
            client = client.with_api_base(api_base);
        }
        init_model(client, ModelRequestConfig::new(name)).await
    }
}
/// [`MemberHarnessFactory`] building the in-process `NativeHarness`.
pub struct NativeMemberFactory {
    /// Model resolver shared by every member this factory builds.
    resolver: Arc<dyn ModelResolver>,
}
impl NativeMemberFactory {
    /// Builds a factory over an explicit resolver.
    #[must_use]
    pub fn new(resolver: Arc<dyn ModelResolver>) -> Self {
        Self { resolver }
    }
    /// Builds a factory that resolves models from the environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(Arc::new(EnvModelResolver::from_env()))
    }
}
/// Returns the agent card a member runs with: its spec card, else one named after the member.
fn build_card(spec: &DeepAgentSpec, ctx: &MemberHarnessContext) -> AgentCard {
    spec.card
        .clone()
        .unwrap_or_else(|| AgentCard::new(ctx.member_name.clone()))
}
/// Builds the `DeepAgent` configuration from a cold member spec.
fn build_config(spec: &DeepAgentSpec, ctx: &MemberHarnessContext) -> DeepAgentConfig {
    let mut config = DeepAgentConfig::default()
        .with_card(build_card(spec, ctx))
        .with_system_prompt(spec.system_prompt.clone())
        .with_max_iterations(
            spec.max_iterations
                .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        );
    config.enable_task_loop = spec.enable_task_loop;
    config.enable_security_rail = spec.enable_security_rail;
    config.completion_timeout_secs = spec.completion_timeout;
    config.language = spec
        .language
        .clone()
        .unwrap_or_else(|| ctx.language.clone());
    config.skills = spec.skills.clone().unwrap_or_default();
    config.cwd = spec.cwd.as_ref().map(PathBuf::from);
    config.project_root = spec.project_root.as_ref().map(PathBuf::from);
    config.workspace = Workspace::new(ctx.workspace_root.clone());
    config.default_mode = match ctx.mode {
        MemberMode::BuildMode => AgentMode::Normal,
        MemberMode::PlanMode => AgentMode::Plan,
    };
    config.enable_plan_mode = ctx.mode == MemberMode::PlanMode;
    if let Some(mode) = spec.prompt_mode.as_deref() {
        match mode {
            "full" => config.prompt_mode = PromptMode::Full,
            "minimal" => config.prompt_mode = PromptMode::Minimal,
            "none" => config.prompt_mode = PromptMode::None,
            other => debug!(
                prompt_mode = other,
                "member spec prompt_mode has no native harness counterpart; keeping the default"
            ),
        }
    }
    config
}
/// Logs spec entries the factory deliberately does not mount.
fn log_unmapped_spec_fields(spec: &DeepAgentSpec) {
    let unmapped = spec.tools.len()
        + spec.mcps.len()
        + spec.subagents.len()
        + spec.rails.len()
        + spec.approval_required_tools.len();
    if unmapped > 0 {
        debug!(
            tools = spec.tools.len(),
            mcps = spec.mcps.len(),
            subagents = spec.subagents.len(),
            rails = spec.rails.len(),
            approval_required_tools = spec.approval_required_tools.len(),
            "member spec entries without a native harness factory counterpart are not mounted"
        );
    }
}
#[async_trait]
impl MemberHarnessFactory for NativeMemberFactory {
    async fn create(
        &self,
        spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        log_unmapped_spec_fields(spec);
        let card = build_card(spec, ctx);
        let model_name = spec.model.as_ref().and_then(ModelRef::resolved_name);
        let model = self.resolver.resolve(model_name).await?;
        let config = build_config(spec, ctx);
        let agent = DeepAgentBuilder::new(card)
            .config(config)
            .model(model)
            .workspace(Workspace::new(ctx.workspace_root.clone()))
            .build()?;
        let harness: Arc<dyn HarnessProtocol> =
            Arc::new(NativeHarness::new(agent, NativeHarnessOptions::default()));
        Ok(harness)
    }
    fn name(&self) -> &'static str {
        NATIVE_FACTORY_NAME
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::{DeepAgentSpec, MemberMode, ToolSpec};
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_harness::AgentMode;
    /// Resolver that always fails, for the error path.
    struct FailingResolver;
    #[async_trait]
    impl ModelResolver for FailingResolver {
        async fn resolve(&self, _model_name: Option<&str>) -> Result<Arc<dyn Model>> {
            Err(Error::config("no model configured"))
        }
    }
    /// Build-mode runtime context rooted at a fixed directory.
    fn context() -> MemberHarnessContext {
        MemberHarnessContext::new(
            "team",
            "dev-1",
            "team:dev-1",
            "/work/dev-1",
            "en",
            MemberMode::BuildMode,
        )
    }
    #[test]
    fn factory_identity_is_native() {
        let factory = NativeMemberFactory::from_env();
        assert_eq!(factory.name(), NATIVE_FACTORY_NAME);
        assert_eq!(factory.name(), "native");
    }
    #[tokio::test]
    async fn resolver_errors_are_propagated() {
        let factory = NativeMemberFactory::new(Arc::new(FailingResolver));
        let error = factory
            .create(&DeepAgentSpec::default(), &context())
            .await
            .err()
            .expect("the failing resolver must reject the build");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("no model configured"));
    }
    #[test]
    fn config_maps_every_modelled_spec_field() {
        let spec = DeepAgentSpec {
            system_prompt: Some("be useful".to_owned()),
            max_iterations: Some(7),
            enable_task_loop: false,
            enable_security_rail: false,
            completion_timeout: 42.0,
            cwd: Some("/repo".to_owned()),
            project_root: Some("/repo".to_owned()),
            skills: Some(vec!["lint".to_owned()]),
            prompt_mode: Some("minimal".to_owned()),
            ..DeepAgentSpec::default()
        };
        let config = build_config(&spec, &context());
        assert_eq!(config.system_prompt.as_deref(), Some("be useful"));
        assert_eq!(config.max_iterations, Some(7));
        assert!(!config.enable_task_loop);
        assert!(!config.enable_security_rail);
        assert!((config.completion_timeout_secs - 42.0).abs() < f64::EPSILON);
        assert_eq!(config.cwd, Some(PathBuf::from("/repo")));
        assert_eq!(config.project_root, Some(PathBuf::from("/repo")));
        assert_eq!(config.skills, vec!["lint".to_owned()]);
        assert_eq!(config.language, "en");
        assert_eq!(config.prompt_mode, PromptMode::Minimal);
        assert_eq!(config.workspace.root_path, PathBuf::from("/work/dev-1"));
        assert_eq!(config.default_mode, AgentMode::Normal);
        assert!(!config.enable_plan_mode);
        assert_eq!(config.card.name, "dev-1");
    }
    #[test]
    fn spec_language_and_card_override_the_context() {
        let spec = DeepAgentSpec {
            language: Some("cn".to_owned()),
            card: Some(AgentCard::new("custom").with_id("card-1")),
            ..DeepAgentSpec::default()
        };
        let config = build_config(&spec, &context());
        assert_eq!(config.language, "cn");
        assert_eq!(config.card.id, "card-1");
        assert_eq!(config.card.name, "custom");
    }
    #[test]
    fn plan_mode_selects_plan_defaults() {
        let ctx = MemberHarnessContext::new(
            "team",
            "dev-1",
            "team:dev-1",
            "/work/dev-1",
            "cn",
            MemberMode::PlanMode,
        );
        let config = build_config(&DeepAgentSpec::default(), &ctx);
        assert_eq!(config.default_mode, AgentMode::Plan);
        assert!(config.enable_plan_mode);
    }
    #[test]
    fn unmapped_spec_fields_do_not_break_configuration() {
        let spec = DeepAgentSpec {
            tools: vec![ToolSpec {
                name: "shell".to_owned(),
                description: String::new(),
                input_schema: serde_json::json!({}),
            }],
            ..DeepAgentSpec::default()
        };
        let config = build_config(&spec, &context());
        assert_eq!(config.card.name, "dev-1");
    }
}
