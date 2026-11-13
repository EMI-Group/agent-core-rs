//! Peer-to-peer ability dispatch: agent cards exposed to a supervisor model as
//! tools, plus a local executor for everything else.
//!
//! Python parity: `P2PAbilityManager` in
//! `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/p2p_ability_manager.py`.
//! Python subclasses `AbilityManager` and overrides `execute()`: it partitions a
//! batch of model tool calls into `AgentCard` calls and other calls, dispatches
//! the agent calls over the team runtime's P2P `send()` bounded by a semaphore,
//! forwards the rest to the base class, turns per-call exceptions into error tool
//! messages, and returns one result per call **in the original call order**.
//!
//! This module keeps that contract with Rust types:
//!
//! - [`AbilityCall`] is one model tool call (`call_id`, `name`, JSON
//!   `arguments`).
//! - [`P2PAbilityManager::register_agent_tool`] maps a member [`AgentCard`]'s
//!   **name** to its **id**: a call whose name is registered dispatches over
//!   [`TeamRuntime::send`], every other call is a *local* call.
//! - [`AbilityExecutor`] serves the local calls; without one a local call is an
//!   explicit error result rather than a silent no-op.
//! - [`AbilityResult`] reports the route each call took ([`AbilityRoute::P2p`],
//!   [`AbilityRoute::Local`] or [`AbilityRoute::Error`]) plus its output or error
//!   text, in input order.
//!
//! Agent dispatch is concurrent and capped by [`P2PAbilityManager::max_parallel`]
//! (Python's `max_parallel_sub_agents`, default `10`, clamped to at least `1`).
//! Only agent calls consume permits, so a slow sub-agent never blocks a local
//! tool call. The dispatch deadline is the runtime's
//! [`p2p_timeout`](TeamRuntime::p2p_timeout), and the emitted P2P `sender` is the
//! supervisor id, which keeps the runtime's trace validation happy.
//!
//! A failing call never aborts the batch: agent failures become
//! [`AbilityRoute::Error`] with `"P2P dispatch to '<tool>' failed: <error>"`, and
//! executor failures keep the executor's message.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use openjiuwen_common::utils::id::uuid_v4;
use openjiuwen_session::Session;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::runtime::TeamRuntime;

/// Default cap on concurrent agent dispatches per [`P2PAbilityManager::execute`].
pub const DEFAULT_MAX_PARALLEL_SUB_AGENTS: usize = 10;

/// One tool call from a model: an id, a tool name, and its JSON arguments.
///
/// A missing `call_id` in JSON becomes a fresh UUID v4 (so a hand-written config
/// still yields a usable call), and `arguments` default to `null`, which dispatch
/// treats like Python's `arguments or {}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct AbilityCall {
    /// The model's tool-call id, echoed back in [`AbilityResult::call_id`].
    pub call_id: String,
    /// The called tool name.
    pub name: String,
    /// The JSON arguments the model produced.
    pub arguments: Value,
}

impl Default for AbilityCall {
    fn default() -> Self {
        Self {
            call_id: uuid_v4(),
            name: String::new(),
            arguments: Value::Null,
        }
    }
}

impl AbilityCall {
    /// Builds a call to `name` with a fresh UUID v4 call id.
    #[must_use]
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            call_id: uuid_v4(),
            name: name.into(),
            arguments,
        }
    }

    /// Replaces the generated call id with the model's own.
    #[must_use]
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = call_id.into();
        self
    }
}

/// How one call was served.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AbilityRoute {
    /// Dispatched over the team runtime to a registered member agent.
    P2p,
    /// Served by the configured [`AbilityExecutor`].
    Local,
    /// Neither route succeeded; [`AbilityResult::error`] carries the reason.
    Error,
}

/// One call's outcome, in the input order of [`AbilityCall`]s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct AbilityResult {
    /// The call id this result belongs to.
    pub call_id: String,
    /// The called tool name.
    pub name: String,
    /// The route that served the call.
    pub route: AbilityRoute,
    /// The reply value; `null` when the call failed.
    #[serde(default)]
    pub output: Value,
    /// The failure text; `None` when the call succeeded.
    #[serde(default)]
    pub error: Option<String>,
}

impl AbilityResult {
    /// Returns `true` when the call failed.
    ///
    /// A result is an error when its route is [`AbilityRoute::Error`] or when it
    /// carries an error message, so a hand-built result cannot look successful
    /// while reporting a failure.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self.route, AbilityRoute::Error) || self.error.is_some()
    }

    /// Builds a successful result.
    fn succeeded(call: &AbilityCall, route: AbilityRoute, output: Value) -> Self {
        Self {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            route,
            output,
            error: None,
        }
    }

    /// Builds a failed result carrying `message`.
    fn failed(call: &AbilityCall, message: impl Into<String>) -> Self {
        Self {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            route: AbilityRoute::Error,
            output: Value::Null,
            error: Some(message.into()),
        }
    }
}

/// Executor of the non-agent (local) tool calls of a batch.
///
/// Implementations usually delegate to the supervisor's ordinary tool registry;
/// the crate only needs the seam so P2P dispatch can stay model-agnostic.
#[async_trait]
pub trait AbilityExecutor: Send + Sync {
    /// Runs one local call and returns its tool result value.
    ///
    /// # Errors
    ///
    /// Any failure becomes an [`AbilityRoute::Error`] result of that single call;
    /// the batch continues.
    async fn execute(&self, call: &AbilityCall, session: Option<Arc<dyn Session>>)
        -> Result<Value>;
}

/// The concurrency gate of agent dispatches.
///
/// The semaphore is created with the manager, so no permit machinery runs lazily
/// inside a task and no `OnceCell`/`unsafe` is needed.
struct ParallelGate {
    /// The configured cap, always at least one.
    max: usize,
    /// The semaphore enforcing it.
    semaphore: Arc<Semaphore>,
}

impl ParallelGate {
    /// Builds a gate capping dispatches at `max`, clamped to at least one.
    fn new(max: usize) -> Self {
        let max = max.max(1);
        Self {
            max,
            semaphore: Arc::new(Semaphore::new(max)),
        }
    }
}

/// Routes model tool calls to member agents over P2P and to everything else
/// locally.
///
/// See the `ability` module documentation for the dispatch contract.
pub struct P2PAbilityManager {
    /// The team runtime agent calls are dispatched through.
    runtime: Arc<TeamRuntime>,
    /// The supervisor id used as the P2P `sender`.
    supervisor_id: String,
    /// The concurrency gate of agent dispatches.
    gate: RwLock<ParallelGate>,
    /// The executor of local calls, when one was configured.
    executor: RwLock<Option<Arc<dyn AbilityExecutor>>>,
    /// Registered agent tools, keyed by tool name (sorted iteration).
    agents: RwLock<BTreeMap<String, AgentCard>>,
}

impl P2PAbilityManager {
    /// Builds a manager dispatching for `supervisor_id` through `runtime`.
    ///
    /// The parallel cap starts at [`DEFAULT_MAX_PARALLEL_SUB_AGENTS`] and no
    /// executor is configured, so local calls fail explicitly until
    /// [`with_executor`](Self::with_executor) supplies one.
    #[must_use]
    pub fn new(runtime: Arc<TeamRuntime>, supervisor_id: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            runtime,
            supervisor_id: supervisor_id.into(),
            gate: RwLock::new(ParallelGate::new(DEFAULT_MAX_PARALLEL_SUB_AGENTS)),
            executor: RwLock::new(None),
            agents: RwLock::new(BTreeMap::new()),
        })
    }

    /// Caps how many agent dispatches run concurrently, clamped to at least one.
    ///
    /// The semaphore is replaced, so call this before the first
    /// [`execute`](Self::execute): permits already handed out belong to the old
    /// gate and are not revoked.
    #[must_use]
    pub fn with_max_parallel(self: Arc<Self>, max_parallel: usize) -> Arc<Self> {
        *self.gate.write() = ParallelGate::new(max_parallel);
        self
    }

    /// Configures the executor of local (non-agent) calls.
    #[must_use]
    pub fn with_executor(self: Arc<Self>, executor: Arc<dyn AbilityExecutor>) -> Arc<Self> {
        *self.executor.write() = Some(executor);
        self
    }

    /// Returns the current cap on concurrent agent dispatches.
    #[must_use]
    pub fn max_parallel(&self) -> usize {
        self.gate.read().max
    }

    /// Returns the supervisor id used as the P2P `sender`.
    #[must_use]
    pub fn supervisor_id(&self) -> &str {
        &self.supervisor_id
    }

    /// Returns the team runtime agent calls are dispatched through.
    #[must_use]
    pub fn runtime(&self) -> Arc<TeamRuntime> {
        Arc::clone(&self.runtime)
    }

    /// Registers `card` so calls named after it dispatch to `card.id`; a later
    /// registration of the same name replaces the earlier one.
    ///
    /// A card without a name is ignored (with a warning): it could never match a
    /// tool call, and an empty registry key would leak into
    /// [`agent_tool_ids`](Self::agent_tool_ids).
    pub fn register_agent_tool(&self, card: &AgentCard) {
        if card.name.is_empty() {
            tracing::warn!(agent_id = %card.id, "agent tool card without a name was ignored");
            return;
        }
        self.agents.write().insert(card.name.clone(), card.clone());
    }

    /// Returns the registered tool names (the member card names), sorted.
    #[must_use]
    pub fn agent_tool_ids(&self) -> Vec<String> {
        self.agents.read().keys().cloned().collect()
    }

    /// Returns `true` when `name` is a registered agent tool.
    #[must_use]
    pub fn is_agent_tool(&self, name: &str) -> bool {
        self.agents.read().contains_key(name)
    }

    /// Returns the registered card of the tool `name`.
    #[must_use]
    pub fn agent_card(&self, name: &str) -> Option<AgentCard> {
        self.agents.read().get(name).cloned()
    }

    /// Executes `calls` and returns one result per call, in input order.
    ///
    /// Agent calls dispatch concurrently through the runtime, bounded by the
    /// parallel cap; local calls run through the configured executor at the same
    /// time. A failing call yields an [`AbilityRoute::Error`] result and never
    /// aborts the batch, so the result vector always has the shape of the input
    /// (an empty input yields an empty vector).
    ///
    /// # Errors
    ///
    /// The signature mirrors Python's coroutine and leaves room for a
    /// batch-level failure; per-call failures are reported in the results, so
    /// the current implementation never returns `Err`.
    pub async fn execute(
        &self,
        calls: Vec<AbilityCall>,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Vec<AbilityResult>> {
        if calls.is_empty() {
            return Ok(Vec::new());
        }
        let session_id = session.as_ref().map(|session| session.id().to_owned());
        let gate = {
            let guard = self.gate.read();
            Arc::clone(&guard.semaphore)
        };
        let pending = calls
            .iter()
            .map(|call| self.dispatch(call, session.clone(), session_id.as_deref(), &gate));
        Ok(futures::future::join_all(pending).await)
    }

    /// Routes one call to its agent or its local executor.
    async fn dispatch(
        &self,
        call: &AbilityCall,
        session: Option<Arc<dyn Session>>,
        session_id: Option<&str>,
        gate: &Arc<Semaphore>,
    ) -> AbilityResult {
        let card = {
            let agents = self.agents.read();
            agents.get(&call.name).cloned()
        };
        let Some(card) = card else {
            return self.execute_local(call, session).await;
        };
        let Ok(permit) = Arc::clone(gate).acquire_owned().await else {
            return AbilityResult::failed(call, "ability dispatch gate closed");
        };
        let arguments = if call.arguments.is_null() {
            json!({})
        } else {
            call.arguments.clone()
        };
        let result = self
            .runtime
            .send(arguments, &card.id, &self.supervisor_id, session_id, None)
            .await;
        drop(permit);
        match result {
            Ok(output) => AbilityResult::succeeded(call, AbilityRoute::P2p, output),
            Err(error) => {
                tracing::warn!(
                    tool = %call.name,
                    agent_id = %card.id,
                    %error,
                    "P2P ability dispatch failed"
                );
                AbilityResult::failed(
                    call,
                    format!("P2P dispatch to '{}' failed: {error}", call.name),
                )
            }
        }
    }

    /// Serves one non-agent call through the configured executor.
    async fn execute_local(
        &self,
        call: &AbilityCall,
        session: Option<Arc<dyn Session>>,
    ) -> AbilityResult {
        let executor = {
            let guard = self.executor.read();
            guard.as_ref().map(Arc::clone)
        };
        let Some(executor) = executor else {
            return AbilityResult::failed(
                call,
                format!(
                    "no ability executor configured: tool '{}' is not a registered agent tool",
                    call.name
                ),
            );
        };
        match executor.execute(call, session).await {
            Ok(output) => AbilityResult::succeeded(call, AbilityRoute::Local, output),
            Err(error) => {
                tracing::warn!(tool = %call.name, %error, "local ability execution failed");
                AbilityResult::failed(call, error.message().into_owned())
            }
        }
    }
}

impl std::fmt::Debug for P2PAbilityManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Every lock is read into a local first: a lock guard created inside a
        // `field(..)` argument lives until the end of the statement, so taking a
        // second lock there would deadlock a parked task.
        let max_parallel = self.max_parallel();
        let has_executor = self.executor.read().is_some();
        let agent_tools = self.agent_tool_ids();
        formatter
            .debug_struct("P2PAbilityManager")
            .field("supervisor_id", &self.supervisor_id)
            .field("max_parallel", &max_parallel)
            .field("has_executor", &has_executor)
            .field("agent_tools", &agent_tools)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn call_ids_are_uuid_v4() {
        let first = AbilityCall::new("search", json!({"q": "rust"}));
        let second = AbilityCall::new("search", Value::Null);
        assert_eq!(first.call_id.len(), 36);
        assert_ne!(first.call_id, second.call_id);
        assert_eq!(first.name, "search");
        assert_eq!(first.arguments, json!({"q": "rust"}));
    }

    #[test]
    fn with_call_id_replaces_the_generated_id() {
        let call = AbilityCall::new("search", json!({})).with_call_id("call-1");
        assert_eq!(call.call_id, "call-1");
    }

    #[test]
    fn call_deserializes_with_a_generated_id_and_tolerates_unknown_fields() {
        let call: AbilityCall =
            serde_json::from_value(json!({"name": "search", "arguments": {"q": 1}, "x": true}))
                .unwrap();
        assert_eq!(call.call_id.len(), 36);
        assert_eq!(call.name, "search");
        assert_eq!(call.arguments, json!({"q": 1}));

        let empty: AbilityCall = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty.name, "");
        assert!(empty.arguments.is_null());
    }

    #[test]
    fn results_round_trip_and_report_errors() {
        let call = AbilityCall::new("search", json!({})).with_call_id("c1");
        let ok = AbilityResult::succeeded(&call, AbilityRoute::Local, json!({"hits": 2}));
        assert!(!ok.is_error());
        assert_eq!(ok.call_id, "c1");
        assert_eq!(serde_json::to_value(&ok).unwrap()["route"], "local");

        let failed = AbilityResult::failed(&call, "boom");
        assert!(failed.is_error());
        assert_eq!(failed.error.as_deref(), Some("boom"));
        assert_eq!(serde_json::to_value(&failed).unwrap()["route"], "error");

        let decoded: AbilityResult = serde_json::from_value(json!({
            "call_id": "c2",
            "name": "search",
            "route": "p2p",
        }))
        .unwrap();
        assert_eq!(decoded.route, AbilityRoute::P2p);
        assert!(decoded.output.is_null());
        assert!(!decoded.is_error());
    }

    #[test]
    fn route_wire_names_are_snake_case() {
        for (route, wire) in [
            (AbilityRoute::P2p, "p2p"),
            (AbilityRoute::Local, "local"),
            (AbilityRoute::Error, "error"),
        ] {
            assert_eq!(serde_json::to_value(route).unwrap(), wire);
            let decoded: AbilityRoute = serde_json::from_value(json!(wire)).unwrap();
            assert_eq!(decoded, route);
        }
    }

    #[tokio::test]
    async fn registration_maps_tool_names_to_agent_ids() {
        let manager = P2PAbilityManager::new(
            TeamRuntime::new(crate::runtime::RuntimeConfig::new().with_team_id("unit")),
            "supervisor",
        );
        manager.register_agent_tool(&AgentCard::new("zeta").with_id("agent-z"));
        manager.register_agent_tool(&AgentCard::new("alpha").with_id("agent-a"));
        manager.register_agent_tool(&AgentCard::new("").with_id("agent-empty"));

        assert_eq!(manager.agent_tool_ids(), vec!["alpha", "zeta"]);
        assert!(manager.is_agent_tool("alpha"));
        assert!(!manager.is_agent_tool("agent-a"));
        assert!(!manager.is_agent_tool(""));
        assert_eq!(
            manager.agent_card("zeta").map(|card| card.id),
            Some("agent-z".to_owned())
        );
        assert!(manager.agent_card("missing").is_none());
    }

    #[tokio::test]
    async fn max_parallel_defaults_and_clamps() {
        let runtime = TeamRuntime::new(crate::runtime::RuntimeConfig::new().with_team_id("unit"));
        let manager = P2PAbilityManager::new(Arc::clone(&runtime), "supervisor");
        assert_eq!(manager.max_parallel(), DEFAULT_MAX_PARALLEL_SUB_AGENTS);
        assert_eq!(manager.supervisor_id(), "supervisor");
        assert_eq!(Arc::clone(&manager).with_max_parallel(4).max_parallel(), 4);
        assert_eq!(Arc::clone(&manager).with_max_parallel(0).max_parallel(), 1);
    }

    #[tokio::test]
    async fn a_local_call_without_an_executor_is_an_explicit_error() {
        let runtime = TeamRuntime::new(crate::runtime::RuntimeConfig::new().with_team_id("unit"));
        let manager = P2PAbilityManager::new(runtime, "supervisor");
        let call = AbilityCall::new("unknown-tool", json!({})).with_call_id("c1");
        let results = manager.execute(vec![call], None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].route, AbilityRoute::Error);
        let message = results[0].error.as_deref().unwrap();
        assert!(
            message.contains("no ability executor configured"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn an_empty_batch_needs_no_routes() {
        let runtime = TeamRuntime::new(crate::runtime::RuntimeConfig::new().with_team_id("unit"));
        let manager = P2PAbilityManager::new(runtime, "supervisor");
        assert_eq!(manager.execute(Vec::new(), None).await.unwrap().len(), 0);
    }
}
