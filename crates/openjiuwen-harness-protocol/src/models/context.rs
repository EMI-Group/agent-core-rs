//! Per-agent runtime context supplied when a third-party harness starts.
//!
//! Mirrors `openjiuwen.harness_protocol.models.HarnessContext`, [`ResumePolicy`], and
//! [`HarnessTelemetry`] (v0.1.20).
//!
//! [`HarnessContext`] is deliberately **not** serde-serializable: it carries live host services
//! (`Arc<dyn …>`), which have no wire shape. Python never serialized the context either — only
//! [`HarnessCard`](crate::HarnessCard) and the event/results models travel over the wire (see
//! `DESIGN.md` §4.4 and §7).
//!
//! Python's `MappingProxyType` freezing of `env`/`metadata` has no Rust analogue: owned
//! `BTreeMap` values replace deep immutability. `env`'s string-to-string shape is enforced by the
//! Rust type, so Python's `TypeError("harness context env must map strings to strings")` has no
//! analogue. Constructors do not validate implicitly; callers that need the Python invariants call
//! [`Validate::validate`].
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::checkpoints::{HarnessCheckpoint, HarnessCheckpointSink};
use crate::errors::HarnessResult;
use crate::hooks::HarnessHookDispatcher;
use crate::interactions::HarnessInteractionHandler;
use crate::json::JsonObject;
use crate::models::card::{HostCapability, PROTOCOL_VERSION};
use crate::tools::{McpServerConfig, ToolGateway};
use crate::validation::{require_non_empty, Validate};

/// How a harness should use the checkpoint supplied at start.
///
/// Wire values are `new`, `resume_if_available`, and `require_resume`.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ResumePolicy {
    /// Start a fresh provider session and ignore any supplied checkpoint (the default).
    #[default]
    New,
    /// Restore the supplied checkpoint when one is present, and start fresh when none is.
    ResumeIfAvailable,
    /// Restore the supplied checkpoint; `start` fails when the harness cannot restore it.
    RequireResume,
}

/// Minimal provider-neutral telemetry service supplied by the host.
///
/// `attributes` are already redacted by the caller: this trait must never receive raw secrets.
/// `record` is a synchronous callback and must not block the async executor; implementations hand
/// observations to an internal queue or channel instead of awaiting I/O inline.
pub trait HarnessTelemetry: Send + Sync {
    /// Records one already-redacted telemetry observation.
    fn record(&self, name: &str, attributes: &JsonObject);
}

/// Per-agent runtime context supplied when a third-party harness starts.
///
/// `env` may contain secrets and must not be logged or copied into events; the manual [`Debug`]
/// implementation therefore redacts `env` and `metadata`. `checkpoint` is opaque to `OpenJiuwen`:
/// only the owning implementation may interpret its contents. `interactions` is an awaited control
/// plane and must not be replaced by observation events.
///
/// The context is **not** serde-serializable — it carries live host services (`Arc<dyn …>`) with
/// no wire representation, exactly like Python, which never serialized it. There is no [`Default`]
/// implementation because identity is mandatory; build a context with [`HarnessContext::new`].
#[derive(Clone)]
pub struct HarnessContext {
    /// Human-readable name of the agent the harness runs.
    pub agent_name: String,
    /// Stable identifier of the agent the harness runs.
    pub agent_id: String,
    /// Host session the agent runs in; checkpoints are scoped to it.
    pub host_session_id: String,
    /// System prompt the harness should treat as the agent's instructions.
    pub system_prompt: String,
    /// Host protocol version, [`PROTOCOL_VERSION`] by default.
    pub protocol_version: String,
    /// Host services the harness may use; empty by default.
    pub host_capabilities: BTreeSet<HostCapability>,
    /// How the harness should use [`checkpoint`](Self::checkpoint); [`ResumePolicy::New`] by
    /// default.
    pub resume_policy: ResumePolicy,
    /// Working directory for the harness, when the host pins one.
    pub cwd: Option<String>,
    /// Environment variables for the harness; may contain secrets and must not be logged.
    pub env: BTreeMap<String, String>,
    /// Opaque checkpoint the harness may resume from.
    pub checkpoint: Option<HarnessCheckpoint>,
    /// Host service the harness persists checkpoints into.
    pub checkpoint_sink: Option<Arc<dyn HarnessCheckpointSink>>,
    /// Host tool gateway used for native tool calls.
    pub tools: Option<Arc<dyn ToolGateway>>,
    /// MCP servers the harness should connect to; empty by default.
    pub mcp_servers: Vec<McpServerConfig>,
    /// Host hook dispatcher the harness calls at control-plane seams.
    pub hooks: Option<Arc<dyn HarnessHookDispatcher>>,
    /// Host interaction handler answering awaited provider requests.
    pub interactions: Option<Arc<dyn HarnessInteractionHandler>>,
    /// Host telemetry service for already-redacted observations.
    pub telemetry: Option<Arc<dyn HarnessTelemetry>>,
    /// Free-form host metadata; must not be logged because it may carry secrets.
    pub metadata: JsonObject,
}

impl HarnessContext {
    /// Builds a context from the mandatory identity fields, with every other field defaulted.
    ///
    /// Defaults match Python: [`PROTOCOL_VERSION`], empty `host_capabilities`, empty `env`,
    /// [`ResumePolicy::New`], no checkpoint or host services, no MCP servers, and empty `metadata`.
    #[must_use]
    pub fn new(
        agent_name: impl Into<String>,
        agent_id: impl Into<String>,
        host_session_id: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            agent_name: agent_name.into(),
            agent_id: agent_id.into(),
            host_session_id: host_session_id.into(),
            system_prompt: system_prompt.into(),
            protocol_version: PROTOCOL_VERSION.to_owned(),
            host_capabilities: BTreeSet::new(),
            resume_policy: ResumePolicy::default(),
            cwd: None,
            env: BTreeMap::new(),
            checkpoint: None,
            checkpoint_sink: None,
            tools: None,
            mcp_servers: Vec::new(),
            hooks: None,
            interactions: None,
            telemetry: None,
            metadata: JsonObject::new(),
        }
    }

    /// Sets the host protocol version.
    #[must_use]
    pub fn with_protocol_version(mut self, protocol_version: impl Into<String>) -> Self {
        self.protocol_version = protocol_version.into();
        self
    }

    /// Sets the host capabilities available to the harness.
    #[must_use]
    pub fn with_host_capabilities(mut self, host_capabilities: BTreeSet<HostCapability>) -> Self {
        self.host_capabilities = host_capabilities;
        self
    }

    /// Sets how the harness should use the supplied checkpoint.
    #[must_use]
    pub fn with_resume_policy(mut self, resume_policy: ResumePolicy) -> Self {
        self.resume_policy = resume_policy;
        self
    }

    /// Sets the harness working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Sets the harness environment variables.
    ///
    /// The map may contain secrets; it is redacted by the [`Debug`] implementation.
    #[must_use]
    pub fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Sets the checkpoint the harness may resume from.
    #[must_use]
    pub fn with_checkpoint(mut self, checkpoint: HarnessCheckpoint) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Sets the host checkpoint sink the harness persists checkpoints into.
    #[must_use]
    pub fn with_checkpoint_sink(mut self, checkpoint_sink: Arc<dyn HarnessCheckpointSink>) -> Self {
        self.checkpoint_sink = Some(checkpoint_sink);
        self
    }

    /// Sets the host tool gateway used for native tool calls.
    #[must_use]
    pub fn with_tools(mut self, tools: Arc<dyn ToolGateway>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the MCP servers the harness should connect to.
    #[must_use]
    pub fn with_mcp_servers(mut self, mcp_servers: Vec<McpServerConfig>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Sets the host hook dispatcher.
    #[must_use]
    pub fn with_hooks(mut self, hooks: Arc<dyn HarnessHookDispatcher>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Sets the host interaction handler.
    #[must_use]
    pub fn with_interactions(mut self, interactions: Arc<dyn HarnessInteractionHandler>) -> Self {
        self.interactions = Some(interactions);
        self
    }

    /// Sets the host telemetry service.
    #[must_use]
    pub fn with_telemetry(mut self, telemetry: Arc<dyn HarnessTelemetry>) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    /// Sets the free-form host metadata.
    ///
    /// The object may carry secrets; it is redacted by the [`Debug`] implementation.
    #[must_use]
    pub fn with_metadata(mut self, metadata: JsonObject) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Validate for HarnessContext {
    /// Checks the identity fields in Python's order: `agent_name`, `agent_id`,
    /// `host_session_id`, then `protocol_version`.
    ///
    /// Nothing else is validated: `env`'s string-to-string shape is enforced by the Rust type, so
    /// Python's `TypeError` has no analogue, and `checkpoint`/`metadata` are opaque values.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("harness context agent_name", &self.agent_name)?;
        require_non_empty("harness context agent_id", &self.agent_id)?;
        require_non_empty("harness context host_session_id", &self.host_session_id)?;
        require_non_empty("harness context protocol_version", &self.protocol_version)?;
        Ok(())
    }
}

impl fmt::Debug for HarnessContext {
    /// Renders identity, wiring, and presence flags only.
    ///
    /// `env` and `metadata` may contain secrets and are always rendered as `"<redacted>"`; the
    /// checkpoint payload and the host services are never printed, only whether they are set.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HarnessContext")
            .field("agent_name", &self.agent_name)
            .field("agent_id", &self.agent_id)
            .field("host_session_id", &self.host_session_id)
            .field("system_prompt", &self.system_prompt)
            .field("protocol_version", &self.protocol_version)
            .field("host_capabilities", &self.host_capabilities)
            .field("resume_policy", &self.resume_policy)
            .field("cwd", &self.cwd)
            .field("env", &"<redacted>")
            .field("checkpoint", &self.checkpoint.is_some())
            .field("checkpoint_sink", &self.checkpoint_sink.is_some())
            .field("tools", &self.tools.is_some())
            .field("mcp_servers", &self.mcp_servers.len())
            .field("hooks", &self.hooks.is_some())
            .field("interactions", &self.interactions.is_some())
            .field("telemetry", &self.telemetry.is_some())
            .field("metadata", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::json;

    use super::{HarnessContext, HarnessTelemetry, ResumePolicy};
    use crate::checkpoints::{
        CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint, HarnessCheckpointSink,
    };
    use crate::hooks::{
        AfterToolContext, AfterToolResult, BeforePromptContext, BeforePromptResult,
        BeforeToolContext, HarnessHookDispatcher, StopHookContext, ToolDecision,
    };
    use crate::interactions::{
        HarnessInteractionHandler, HarnessInteractionRequest, HarnessInteractionResponse,
        InteractionCancelReason,
    };
    use crate::json::JsonObject;
    use crate::tools::{
        McpServerConfig, McpTransport, ToolDefinition, ToolExecutionResult, ToolGateway,
        ToolInvocation,
    };
    use crate::{HarnessError, HarnessResult, HostCapability, Validate, PROTOCOL_VERSION};

    /// Host services that a context may retain; never invoked, only stored.
    struct UnusedServices;

    #[async_trait]
    impl HarnessCheckpointSink for UnusedServices {
        async fn save(
            &self,
            _checkpoint: &HarnessCheckpoint,
            _reason: CheckpointReason,
            _expected_storage_revision: Option<&str>,
        ) -> HarnessResult<CheckpointSaveReceipt> {
            Err(HarnessError::protocol("unused in context tests"))
        }
    }

    #[async_trait]
    impl ToolGateway for UnusedServices {
        async fn definitions(&self) -> HarnessResult<Vec<ToolDefinition>> {
            Err(HarnessError::protocol("unused in context tests"))
        }

        async fn invoke(&self, _invocation: &ToolInvocation) -> HarnessResult<ToolExecutionResult> {
            Err(HarnessError::protocol("unused in context tests"))
        }
    }

    #[async_trait]
    impl HarnessHookDispatcher for UnusedServices {
        async fn before_prompt(
            &self,
            _context: &BeforePromptContext,
        ) -> HarnessResult<BeforePromptResult> {
            Err(HarnessError::protocol("unused in context tests"))
        }

        async fn before_tool(&self, _context: &BeforeToolContext) -> HarnessResult<ToolDecision> {
            Err(HarnessError::protocol("unused in context tests"))
        }

        async fn after_tool(&self, _context: &AfterToolContext) -> HarnessResult<AfterToolResult> {
            Err(HarnessError::protocol("unused in context tests"))
        }

        async fn on_stop(&self, _context: &StopHookContext) -> HarnessResult<()> {
            Err(HarnessError::protocol("unused in context tests"))
        }
    }

    #[async_trait]
    impl HarnessInteractionHandler for UnusedServices {
        async fn handle(
            &self,
            _request: &HarnessInteractionRequest,
        ) -> HarnessResult<HarnessInteractionResponse> {
            Err(HarnessError::protocol("unused in context tests"))
        }

        async fn cancel(
            &self,
            _request_id: &str,
            _reason: InteractionCancelReason,
        ) -> HarnessResult<()> {
            Err(HarnessError::protocol("unused in context tests"))
        }
    }

    impl HarnessTelemetry for UnusedServices {
        fn record(&self, _name: &str, _attributes: &JsonObject) {}
    }

    fn base_context() -> HarnessContext {
        HarnessContext::new("agent", "agent-1", "session-1", "you are helpful")
    }

    fn checkpoint() -> HarnessCheckpoint {
        HarnessCheckpoint {
            provider: "native".to_owned(),
            schema_version: "1".to_owned(),
            agent_id: "agent-1".to_owned(),
            host_session_id: "session-1".to_owned(),
            checkpoint_id: "cp-1".to_owned(),
            sequence: 1,
            data: JsonObject::new(),
            provider_session_id: None,
            revision: None,
        }
    }

    fn stdio_server() -> McpServerConfig {
        McpServerConfig {
            name: "files".to_owned(),
            transport: McpTransport::Stdio,
            command: vec!["mcp".to_owned()],
            url: None,
            instance: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
        }
    }

    #[test]
    fn new_fills_python_defaults() {
        let context = base_context();
        assert_eq!(context.agent_name, "agent");
        assert_eq!(context.agent_id, "agent-1");
        assert_eq!(context.host_session_id, "session-1");
        assert_eq!(context.system_prompt, "you are helpful");
        assert_eq!(context.protocol_version, PROTOCOL_VERSION);
        assert!(context.host_capabilities.is_empty());
        assert_eq!(context.resume_policy, ResumePolicy::New);
        assert!(context.cwd.is_none());
        assert!(context.env.is_empty());
        assert!(context.checkpoint.is_none());
        assert!(context.checkpoint_sink.is_none());
        assert!(context.tools.is_none());
        assert_eq!(context.mcp_servers.len(), 0);
        assert!(context.hooks.is_none());
        assert!(context.interactions.is_none());
        assert!(context.telemetry.is_none());
        assert!(context.metadata.is_empty());
        assert!(context.validate().is_ok());
    }

    #[test]
    fn builders_chain_and_retain_services_and_data() {
        let services = Arc::new(UnusedServices);
        let capabilities: BTreeSet<HostCapability> =
            [HostCapability::CheckpointSink, HostCapability::Telemetry]
                .into_iter()
                .collect();
        let env: BTreeMap<String, String> = [("API_KEY".to_owned(), "sk-live-secret".to_owned())]
            .into_iter()
            .collect();
        let mut metadata = JsonObject::new();
        metadata.insert("tenant".to_owned(), json!("acme-secret"));

        let context = base_context()
            .with_protocol_version("1.1")
            .with_host_capabilities(capabilities.clone())
            .with_resume_policy(ResumePolicy::RequireResume)
            .with_cwd("/tmp/agent")
            .with_env(env.clone())
            .with_checkpoint(checkpoint())
            .with_checkpoint_sink(services.clone())
            .with_tools(services.clone())
            .with_mcp_servers(vec![stdio_server()])
            .with_hooks(services.clone())
            .with_interactions(services.clone())
            .with_telemetry(services)
            .with_metadata(metadata.clone());

        assert_eq!(context.protocol_version, "1.1");
        assert_eq!(context.host_capabilities, capabilities);
        assert_eq!(context.resume_policy, ResumePolicy::RequireResume);
        assert_eq!(context.cwd.as_deref(), Some("/tmp/agent"));
        assert_eq!(context.env, env);
        assert_eq!(context.checkpoint, Some(checkpoint()));
        assert!(context.checkpoint_sink.is_some());
        assert!(context.tools.is_some());
        assert_eq!(context.mcp_servers, vec![stdio_server()]);
        assert!(context.hooks.is_some());
        assert!(context.interactions.is_some());
        assert!(context.telemetry.is_some());
        assert_eq!(context.metadata, metadata);
        assert!(context.validate().is_ok());
    }

    #[test]
    fn identity_validation_reports_fields_in_python_order() {
        let cases = [
            (
                HarnessContext::new("", "agent-1", "session-1", "prompt"),
                "harness context agent_name must not be empty",
            ),
            (
                HarnessContext::new("agent", "", "session-1", "prompt"),
                "harness context agent_id must not be empty",
            ),
            (
                HarnessContext::new("agent", "agent-1", "", "prompt"),
                "harness context host_session_id must not be empty",
            ),
            (
                base_context().with_protocol_version(""),
                "harness context protocol_version must not be empty",
            ),
        ];
        for (context, expected) in cases {
            let error = context.validate().expect_err("empty identity field");
            assert_eq!(error.message(), expected);
        }
    }

    #[test]
    fn resume_policy_wire_values_and_default() {
        assert_eq!(ResumePolicy::default(), ResumePolicy::New);
        let wire = [
            (ResumePolicy::New, "\"new\""),
            (ResumePolicy::ResumeIfAvailable, "\"resume_if_available\""),
            (ResumePolicy::RequireResume, "\"require_resume\""),
        ];
        for (policy, expected) in wire {
            assert_eq!(serde_json::to_string(&policy).unwrap(), expected);
            let decoded: ResumePolicy = serde_json::from_str(expected).unwrap();
            assert_eq!(decoded, policy);
        }
    }

    #[test]
    fn debug_redacts_env_and_metadata() {
        let mut metadata = JsonObject::new();
        metadata.insert("tenant".to_owned(), json!("acme-secret"));
        let context = base_context()
            .with_env(
                [("API_KEY".to_owned(), "sk-live-secret".to_owned())]
                    .into_iter()
                    .collect(),
            )
            .with_metadata(metadata);
        let rendered = format!("{context:?}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(!rendered.contains("sk-live-secret"), "{rendered}");
        assert!(!rendered.contains("acme-secret"), "{rendered}");
        assert!(!rendered.contains("API_KEY"), "{rendered}");
        assert!(rendered.contains("agent-1"), "{rendered}");
        assert!(rendered.contains("resume_policy"), "{rendered}");
    }
}
