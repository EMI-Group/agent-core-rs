//! Team members: an agent plus its optional tool surface and bind hook.
//!
//! Python parity: team members in `openjiuwen/core/multi_agent/team.py` are
//! plain `BaseAgent` instances that the runtime probes for an ability manager
//! and rebinds to the runner. Rust makes both capabilities explicit: a
//! [`TeamAgent`] pairs the agent with an optional [`ToolHost`] (the surface a
//! strategy mounts `AgentTool`s on) and an optional [`BindHook`] (called once a
//! member is resolved from its [`AgentProvider`]).
//!
//! Mounting only works for tool-hosting agents: a plain `dyn Agent` has no tool
//! surface, so [`TeamAgent::tool_host`] may be `None` and strategies silently
//! skip members that cannot host tools (Python parity: the attribute probe
//! fails and the mount is skipped).

use std::sync::Arc;

use openjiuwen_agent::{Agent, LlmAgent, ReActAgent};
use openjiuwen_foundation::tool::Tool;

use crate::runtime::TeamRuntime;

/// Builds the member instance for one resolution.
///
/// The runtime invokes the provider every time a member is resolved, matching
/// Python's per-dispatch provider. A strategy that wired state into one instance
/// can [`TeamRuntime::pin_agent`] that instance instead.
pub type AgentProvider = Arc<dyn Fn() -> TeamAgent + Send + Sync>;

/// Callback that binds a resolved member to its runtime before invocation.
///
/// `handoff` and `hierarchical_tools` use it to mount tools on the freshly
/// resolved instance; the hook receives the runtime and the member's agent id.
pub type BindHook = Arc<dyn Fn(&Arc<TeamRuntime>, &str) + Send + Sync>;

/// A team member: an agent plus the capabilities a strategy may need.
///
/// Cloning is cheap (every field is an `Arc`), so a provider can hand out the
/// same instance repeatedly.
#[derive(Clone)]
pub struct TeamAgent {
    agent: Arc<dyn Agent>,
    tool_host: Option<Arc<dyn ToolHost>>,
    bind_hook: Option<BindHook>,
}

impl std::fmt::Debug for TeamAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamAgent")
            .field("agent", &self.agent.card().id)
            .field("has_tool_host", &self.tool_host.is_some())
            .field("has_bind_hook", &self.bind_hook.is_some())
            .finish()
    }
}

impl TeamAgent {
    /// Builds a member around `agent` with no tool surface and no bind hook.
    #[must_use]
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        Self {
            agent,
            tool_host: None,
            bind_hook: None,
        }
    }

    /// Builds a member around `agent` whose tool surface is `host`.
    ///
    /// The host is usually the agent itself (see [`TeamAgent::react`]) but may
    /// be any object that mounts tools on the agent's behalf.
    #[must_use]
    pub fn with_tool_host(agent: Arc<dyn Agent>, host: Arc<dyn ToolHost>) -> Self {
        Self {
            agent,
            tool_host: Some(host),
            bind_hook: None,
        }
    }

    /// Sets the hook called after this member is resolved from its provider.
    #[must_use]
    pub fn with_bind(
        mut self,
        hook: impl Fn(&Arc<TeamRuntime>, &str) + Send + Sync + 'static,
    ) -> Self {
        self.bind_hook = Some(Arc::new(hook));
        self
    }

    /// Builds a member around a [`ReActAgent`] that is its own tool host.
    ///
    /// The same agent object answers invocations and accepts mounted tools.
    #[must_use]
    pub fn react(agent: Arc<ReActAgent>) -> Self {
        let tool_host: Arc<dyn ToolHost> = agent.clone();
        Self {
            agent,
            tool_host: Some(tool_host),
            bind_hook: None,
        }
    }

    /// Returns the agent driven by the runtime.
    #[must_use]
    pub fn agent(&self) -> Arc<dyn Agent> {
        Arc::clone(&self.agent)
    }

    /// Returns the tool surface of the member, when it has one.
    #[must_use]
    pub fn tool_host(&self) -> Option<Arc<dyn ToolHost>> {
        self.tool_host.clone()
    }

    /// Calls the bind hook, if any, with the runtime and `agent_id`.
    ///
    /// A member without a hook ignores the call, so callers never need to
    /// special-case it.
    pub fn bind(&self, runtime: &Arc<TeamRuntime>, agent_id: &str) {
        if let Some(hook) = &self.bind_hook {
            hook(runtime, agent_id);
        }
    }
}

impl From<Arc<dyn Agent>> for TeamAgent {
    fn from(agent: Arc<dyn Agent>) -> Self {
        Self::new(agent)
    }
}

/// The tool surface of a member agent.
///
/// Implemented for the agents that own an ability manager. A strategy mounts a
/// tool through [`mount_tool`](ToolHost::mount_tool) and treats `false` as "this
/// agent cannot host it" rather than an error.
pub trait ToolHost: Send + Sync {
    /// Mounts (or refreshes) a tool; returns `true` when it was added or refreshed.
    ///
    /// `false` means the tool was rejected — usually because a different tool
    /// with the same name is already mounted.
    fn mount_tool(&self, tool: Arc<dyn Tool>) -> bool;
}

impl ToolHost for ReActAgent {
    fn mount_tool(&self, tool: Arc<dyn Tool>) -> bool {
        self.add_tool(tool).added
    }
}

impl ToolHost for LlmAgent {
    fn mount_tool(&self, tool: Arc<dyn Tool>) -> bool {
        self.react().add_tool(tool).added
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use openjiuwen_agent::AgentCard;
    use openjiuwen_foundation::tool::ToolContext;
    use serde_json::Value;

    use super::*;
    use crate::RuntimeConfig;

    /// A tool with a fixed card and a trivial success result.
    struct StubTool {
        card: openjiuwen_foundation::tool::ToolCard,
    }

    impl StubTool {
        fn new(name: &str) -> Arc<Self> {
            Arc::new(Self {
                card: openjiuwen_foundation::tool::ToolCard::new(
                    name,
                    "a stub tool",
                    serde_json::json!({"type": "object"}),
                ),
            })
        }
    }

    #[async_trait::async_trait]
    impl Tool for StubTool {
        fn card(&self) -> &openjiuwen_foundation::tool::ToolCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _ctx: &ToolContext,
        ) -> openjiuwen_common::error::Result<Value> {
            Ok(serde_json::json!({"ok": true}))
        }
    }

    #[test]
    fn new_has_no_tool_host_and_no_hook() {
        let agent: Arc<dyn Agent> = Arc::new(ReActAgent::new(AgentCard::new("plain")));
        let member = TeamAgent::new(agent);
        assert!(member.tool_host().is_none());
        assert_eq!(member.agent().card().name, "plain");
    }

    #[test]
    fn from_arc_dyn_agent_matches_new() {
        let agent: Arc<dyn Agent> = Arc::new(ReActAgent::new(AgentCard::new("plain")));
        let member: TeamAgent = agent.into();
        assert!(member.tool_host().is_none());
    }

    #[test]
    fn react_exposes_the_agent_as_host_and_mounts_tools() {
        let react = Arc::new(ReActAgent::new(AgentCard::new("worker")));
        let member = TeamAgent::react(Arc::clone(&react));
        let host = member.tool_host().expect("react members host tools");
        assert_eq!(member.agent().card().name, "worker");

        let tool: Arc<dyn Tool> = StubTool::new("stub");
        assert!(host.mount_tool(Arc::clone(&tool)));
        // Re-mounting the same instance refreshes the entry and reports `true`.
        assert!(host.mount_tool(tool));
        assert_eq!(react.tools().len(), 1);
    }

    #[test]
    fn llm_agent_is_a_tool_host() {
        let llm = openjiuwen_agent::LlmAgent::new(openjiuwen_agent::LlmAgentConfig::new("llm"))
            .expect("an agent without a model still builds");
        let host: Arc<dyn ToolHost> = Arc::new(llm);
        assert!(host.mount_tool(StubTool::new("stub")));
    }

    #[test]
    fn bind_is_a_no_op_without_a_hook() {
        let runtime = TeamRuntime::new(RuntimeConfig::new());
        let member = TeamAgent::new(Arc::new(ReActAgent::new(AgentCard::new("plain"))));
        member.bind(&runtime, "plain");
    }

    #[test]
    fn bind_calls_the_hook_with_the_agent_id() {
        let runtime = TeamRuntime::new(RuntimeConfig::new());
        let called = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&called);
        let member = TeamAgent::new(Arc::new(ReActAgent::new(AgentCard::new("plain")))).with_bind(
            move |bound, agent_id| {
                assert_eq!(bound.team_id(), "default");
                assert_eq!(agent_id, "worker-1");
                flag.store(true, Ordering::SeqCst);
            },
        );
        member.bind(&runtime, "worker-1");
        assert!(called.load(Ordering::SeqCst));
    }
}
