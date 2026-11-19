//! Shared fakes for the sub-agent tool tests.
//!
//! [`FakeAgent`] answers `echo:<query>` (or an error result, or blocks on a [`Notify`]),
//! and [`FakeFactory`] records the configuration and child session id of every spawn.
//! [`echo_runtime`] wires a runtime over the built-in English presets plus the
//! `general-purpose` definition used by the tool tests.
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::Notify;

use super::{
    general_purpose_config, SubAgentConfig, SubAgentFactory, SubAgentQuota, SubAgentRegistry,
    SubAgentRuntime, SubAgentSpec,
};

/// How a [`FakeAgent`] answers one turn.
#[derive(Clone)]
pub(crate) enum Behavior {
    /// Answer with `echo:<query>`.
    Echo,
    /// Answer with an `error` result carrying `model failed`.
    ErrorResult,
    /// Block until the gate is notified, then answer with `echo:<query>`.
    Blocked(Arc<Notify>),
}

/// An [`Agent`] whose behaviour is fixed at construction.
pub(crate) struct FakeAgent {
    /// Card copied from the configuration the factory received.
    card: AgentCard,
    /// How turns are answered.
    behavior: Behavior,
}

#[async_trait]
impl Agent for FakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let query = inputs
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        match &self.behavior {
            Behavior::Echo => Ok(AgentOutput::answer(format!("echo:{query}"))),
            Behavior::ErrorResult => Ok(AgentOutput::error("model failed")),
            Behavior::Blocked(gate) => {
                gate.notified().await;
                Ok(AgentOutput::answer(format!("echo:{query}")))
            }
        }
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }
}

/// A [`SubAgentFactory`] building [`FakeAgent`]s and recording every request.
pub(crate) struct FakeFactory {
    /// Every `(configuration, child session id)` the runtime asked for.
    recorded: Mutex<Vec<(SubAgentConfig, String)>>,
    /// Behaviour of the agents this factory creates.
    behavior: Behavior,
}

impl FakeFactory {
    /// Creates a factory whose children behave like `behavior`.
    #[must_use]
    pub(crate) fn new(behavior: Behavior) -> Arc<Self> {
        Arc::new(Self {
            recorded: Mutex::new(Vec::new()),
            behavior,
        })
    }

    /// Returns a copy of the recorded `(configuration, child session id)` pairs.
    #[must_use]
    pub(crate) fn recorded(&self) -> Vec<(SubAgentConfig, String)> {
        self.recorded.lock().clone()
    }
}

#[async_trait]
impl SubAgentFactory for FakeFactory {
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>> {
        self.recorded
            .lock()
            .push((config.clone(), session_id.to_owned()));
        Ok(Arc::new(FakeAgent {
            card: config.agent_card.clone(),
            behavior: self.behavior.clone(),
        }))
    }
}

/// Builds an English registry with the built-in presets plus `general-purpose`.
#[must_use]
pub(crate) fn build_registry() -> Arc<SubAgentRegistry> {
    let registry = SubAgentRegistry::with_builtin_presets("en");
    registry.register(SubAgentSpec::new(general_purpose_config()));
    Arc::new(registry)
}

/// Builds a runtime over [`build_registry`] with parent session `parent-session`.
#[must_use]
pub(crate) fn build_runtime(factory: Arc<FakeFactory>) -> Arc<SubAgentRuntime> {
    Arc::new(SubAgentRuntime::new(
        build_registry(),
        factory,
        "parent-session",
    ))
}

/// Builds a runtime over [`build_registry`] with an explicit quota.
#[must_use]
pub(crate) fn build_runtime_with_quota(
    factory: Arc<FakeFactory>,
    quota: SubAgentQuota,
) -> Arc<SubAgentRuntime> {
    Arc::new(SubAgentRuntime::new(build_registry(), factory, "parent-session").with_quota(quota))
}
/// Builds an echoing runtime with an explicit quota, returning it with its factory.
#[must_use]
pub(crate) fn echo_runtime_with_quota(
    quota: SubAgentQuota,
) -> (Arc<SubAgentRuntime>, Arc<FakeFactory>) {
    let factory = FakeFactory::new(Behavior::Echo);
    let runtime = build_runtime_with_quota(Arc::clone(&factory), quota);
    (runtime, factory)
}
/// Builds an echoing runtime and returns it together with its factory.
#[must_use]
pub(crate) fn echo_runtime() -> (Arc<SubAgentRuntime>, Arc<FakeFactory>) {
    let factory = FakeFactory::new(Behavior::Echo);
    let runtime = build_runtime(Arc::clone(&factory));
    (runtime, factory)
}
