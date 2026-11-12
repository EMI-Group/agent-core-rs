//! The per-chain coordinator: route approval, hop accounting, and completion.
//!
//! Python parity: `HandoffOrchestrator` in
//! `openjiuwen/core/multi_agent/teams/handoff/handoff_orchestrator.py`. One
//! orchestrator belongs to one handoff chain (one team-session invocation): it
//! decides whether a member may transfer control to another member, tracks how
//! many hops were taken, and exposes the chain's final result to the waiting
//! strategy through a oneshot channel.
//!
//! The completion channel is created in the constructor rather than lazily, so
//! `complete`/`error` work before anyone calls [`HandoffOrchestrator::wait`] —
//! the hops run while the strategy is still setting up, and a hop that finishes
//! first must not lose its result.
//!
//! [`HandoffRegistry`] maps a team session id to its live orchestrator, so the
//! per-hop `ContainerAgent` endpoints can find the coordinator of the chain they
//! belong to.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{AgentTeamSession, Session};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use super::{
    HandoffConfig, HandoffRoute, HandoffSnapshot, TerminationCondition, COORDINATOR_STATE_KEY,
};
use crate::error::TeamError;

/// The error text used when no completion result can be read from the chain.
const CLOSED_MESSAGE: &str = "handoff coordinator closed without a result";

/// The live orchestrators of a team, keyed by team session id.
///
/// A strategy inserts one orchestrator per invocation and removes it on
/// teardown; the key is the id of the team session the chain runs in.
#[derive(Default)]
pub struct HandoffRegistry {
    /// Session id to the orchestrator of that session's chain.
    orchestrators: RwLock<BTreeMap<String, Arc<HandoffOrchestrator>>>,
}

impl fmt::Debug for HandoffRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let session_ids: Vec<String> = self.orchestrators.read().keys().cloned().collect();
        f.debug_struct("HandoffRegistry")
            .field("session_ids", &session_ids)
            .finish()
    }
}

impl HandoffRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores the orchestrator of `session_id`, replacing any previous one.
    pub fn insert(&self, session_id: impl Into<String>, orchestrator: Arc<HandoffOrchestrator>) {
        self.orchestrators
            .write()
            .insert(session_id.into(), orchestrator);
    }

    /// Returns the orchestrator of `session_id`, when one is registered.
    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<Arc<HandoffOrchestrator>> {
        self.orchestrators.read().get(session_id).map(Arc::clone)
    }

    /// Removes and returns the orchestrator of `session_id`.
    pub fn remove(&self, session_id: &str) -> Option<Arc<HandoffOrchestrator>> {
        self.orchestrators.write().remove(session_id)
    }
}

/// The coordinator of one handoff chain.
pub struct HandoffOrchestrator {
    /// Id of the agent that runs the next hop.
    current_agent_id: Mutex<String>,
    /// Number of handoffs approved so far.
    handoff_count: AtomicUsize,
    /// Allowed `source -> targets` transitions.
    route_graph: BTreeMap<String, BTreeSet<String>>,
    /// Maximum number of handoff transfers after the initial hop.
    max_handoffs: usize,
    /// Optional early-termination predicate.
    termination_condition: Option<TerminationCondition>,
    /// Sending half of the completion channel; `None` once resolved.
    completion_sender: Mutex<Option<oneshot::Sender<Result<Value>>>>,
    /// Receiving half of the completion channel; `None` once awaited.
    completion_receiver: Mutex<Option<oneshot::Receiver<Result<Value>>>>,
}

impl fmt::Debug for HandoffOrchestrator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandoffOrchestrator")
            .field("current_agent_id", &self.current_agent_id())
            .field("handoff_count", &self.handoff_count())
            .field("max_handoffs", &self.max_handoffs)
            .field("routes", &self.route_graph)
            .field("has_termination", &self.termination_condition.is_some())
            .finish_non_exhaustive()
    }
}

impl HandoffOrchestrator {
    /// Creates a coordinator for a chain starting at `start_agent_id`.
    ///
    /// `members` seeds the route graph and `config` supplies the hop cap, the
    /// explicit routes, and the termination condition. The completion channel is
    /// created here so a hop may resolve the chain before anyone awaits it.
    #[must_use]
    pub fn new(
        start_agent_id: impl Into<String>,
        members: &[String],
        config: &HandoffConfig,
    ) -> Self {
        let (completion_sender, completion_receiver) = oneshot::channel();
        Self {
            current_agent_id: Mutex::new(start_agent_id.into()),
            handoff_count: AtomicUsize::new(0),
            route_graph: Self::build_route_graph(members, &config.routes),
            max_handoffs: config.max_handoffs,
            termination_condition: config.termination_condition.clone(),
            completion_sender: Mutex::new(Some(completion_sender)),
            completion_receiver: Mutex::new(Some(completion_receiver)),
        }
    }

    /// Builds the `source -> allowed targets` adjacency graph.
    ///
    /// Every member becomes a key. With explicit `routes` only those transitions
    /// are allowed (a target need not be a member); with no routes the graph is a
    /// full mesh minus self-transitions, matching Python.
    #[must_use]
    pub fn build_route_graph(
        members: &[String],
        routes: &[HandoffRoute],
    ) -> BTreeMap<String, BTreeSet<String>> {
        let mut graph: BTreeMap<String, BTreeSet<String>> = members
            .iter()
            .map(|member| (member.clone(), BTreeSet::new()))
            .collect();
        if routes.is_empty() {
            for source in members {
                let targets = graph.entry(source.clone()).or_default();
                for target in members {
                    if target != source {
                        targets.insert(target.clone());
                    }
                }
            }
        } else {
            for route in routes {
                graph
                    .entry(route.source.clone())
                    .or_default()
                    .insert(route.target.clone());
            }
        }
        graph
    }

    /// Returns the id of the agent that runs the next hop.
    #[must_use]
    pub fn current_agent_id(&self) -> String {
        self.current_agent_id.lock().clone()
    }

    /// Returns the number of handoffs approved so far.
    #[must_use]
    pub fn handoff_count(&self) -> usize {
        self.handoff_count.load(Ordering::SeqCst)
    }

    /// Returns the route graph this coordinator enforces.
    #[must_use]
    pub fn route_graph(&self) -> &BTreeMap<String, BTreeSet<String>> {
        &self.route_graph
    }

    /// Requests a handoff from the current agent to `target`.
    ///
    /// Checks run in the Python order and every rejection is reported as
    /// `Ok(false)` with the state untouched:
    ///
    /// 1. the hop cap (`handoff_count >= max_handoffs`);
    /// 2. the configured termination condition (a condition error propagates);
    /// 3. the route graph entry of the current agent.
    ///
    /// An approved handoff increments the count and moves the current agent.
    ///
    /// # Errors
    ///
    /// Propagates a failure of the termination condition.
    pub async fn request_handoff(&self, target: &str, reason: Option<&str>) -> Result<bool> {
        let count = self.handoff_count();
        if count >= self.max_handoffs {
            tracing::debug!(target, count, "handoff rejected: max_handoffs reached");
            return Ok(false);
        }
        if let Some(condition) = self.termination_condition.clone() {
            let snapshot = HandoffSnapshot {
                current_agent_id: self.current_agent_id(),
                handoff_count: count,
            };
            if condition(snapshot).await? {
                tracing::debug!(target, count, "handoff rejected: termination condition met");
                return Ok(false);
            }
        }
        let current = self.current_agent_id();
        let allowed = self
            .route_graph
            .get(&current)
            .is_some_and(|targets| targets.contains(target));
        if !allowed {
            tracing::warn!(%current, target, "handoff route not allowed");
            return Ok(false);
        }
        target.clone_into(&mut self.current_agent_id.lock());
        let approved = self.handoff_count.fetch_add(1, Ordering::SeqCst) + 1;
        tracing::debug!(
            target,
            count = approved,
            reason = reason.unwrap_or_default(),
            "handoff approved"
        );
        Ok(true)
    }

    /// Resolves the chain with `result`; returns `false` when already resolved.
    // `async` mirrors Python's awaitable `complete` and keeps the coordinator's
    // resolution API uniform with `request_handoff`/`wait`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn complete(&self, result: Value) -> bool {
        let sender = self.completion_sender.lock().take();
        let Some(sender) = sender else {
            tracing::debug!("handoff chain already resolved; ignoring completion");
            return false;
        };
        if sender.send(Ok(result)).is_err() {
            tracing::debug!("handoff completion has no waiting receiver");
        }
        true
    }

    /// Resolves the chain with `error`; returns `false` when already resolved.
    // `async` mirrors Python's awaitable `error` and keeps the coordinator's
    // resolution API uniform with `request_handoff`/`wait`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn error(&self, error: Error) -> bool {
        let sender = self.completion_sender.lock().take();
        let Some(sender) = sender else {
            tracing::debug!("handoff chain already resolved; ignoring error");
            return false;
        };
        if sender.send(Err(error)).is_err() {
            tracing::debug!("handoff error has no waiting receiver");
        }
        true
    }

    /// Awaits the chain's result.
    ///
    /// The receiver is taken exactly once, so a second `wait` cannot observe a
    /// stale result. `timeout` bounds the wait; the elapsed error is generic
    /// because the strategy rewrites it with the team context.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Timeout`] when the deadline
    /// elapses and [`openjiuwen_common::error::ErrorKind::External`] when the
    /// coordinator has no receiver left or its channel closed without a result.
    pub async fn wait(&self, timeout: Option<Duration>) -> Result<Value> {
        let receiver = self.completion_receiver.lock().take();
        let Some(receiver) = receiver else {
            return Err(Error::external(TeamError::execution(CLOSED_MESSAGE)));
        };
        let outcome = match timeout {
            Some(timeout) => tokio::time::timeout(timeout, receiver)
                .await
                .map_err(|_| Error::timeout("handoff chain timed out"))?,
            None => receiver.await,
        };
        match outcome {
            Ok(result) => result,
            Err(_) => Err(Error::external(TeamError::execution(CLOSED_MESSAGE))),
        }
    }

    /// Persists the coordinator state into the team session.
    ///
    /// The snapshot lives under [`COORDINATOR_STATE_KEY`] and is read back by
    /// [`HandoffOrchestrator::restore_from_session`] after an interrupt.
    ///
    /// # Errors
    ///
    /// Propagates session state write failures.
    pub async fn save_to_session(&self, session: &Arc<AgentTeamSession>) -> Result<()> {
        session
            .update_state(json!({
                COORDINATOR_STATE_KEY: {
                    "current_agent_id": self.current_agent_id(),
                    "handoff_count": self.handoff_count(),
                }
            }))
            .await
    }

    /// Rebuilds a coordinator, restoring the saved state when it is complete.
    ///
    /// A snapshot object must carry both `current_agent_id` (string) and
    /// `handoff_count` (unsigned integer); when it is absent or partial the
    /// coordinator keeps the start values and logs a warning. Python indexes the
    /// snapshot and raises `KeyError` on a partial one; starting over is the
    /// documented Rust deviation (a resumable chain should never fail to resume).
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` keeps the signature stable for callers
    /// that read the snapshot asynchronously before restoring.
    #[allow(clippy::unnecessary_wraps)]
    pub fn restore_from_session(
        snapshot: Option<&Value>,
        start_agent_id: impl Into<String>,
        members: &[String],
        config: &HandoffConfig,
    ) -> Result<Self> {
        let orchestrator = Self::new(start_agent_id, members, config);
        let restored = snapshot.and_then(Value::as_object).and_then(|object| {
            let current = object.get("current_agent_id").and_then(Value::as_str)?;
            let count = object.get("handoff_count").and_then(Value::as_u64)?;
            Some((current.to_owned(), count))
        });
        if let Some((current, count)) = restored {
            *orchestrator.current_agent_id.lock() = current;
            orchestrator.handoff_count.store(
                usize::try_from(count).unwrap_or(usize::MAX),
                Ordering::SeqCst,
            );
        } else {
            tracing::warn!(
                "handoff coordinator snapshot is absent or partial; starting from the start agent"
            );
        }
        Ok(orchestrator)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use futures::FutureExt;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    fn members() -> Vec<String> {
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
    }

    fn targets(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn route_graph_defaults_to_a_full_mesh() {
        let graph = HandoffOrchestrator::build_route_graph(&members(), &[]);
        assert_eq!(graph.len(), 3);
        assert_eq!(graph["a"], targets(&["b", "c"]));
        assert_eq!(graph["b"], targets(&["a", "c"]));
        assert_eq!(graph["c"], targets(&["a", "b"]));
    }

    #[test]
    fn route_graph_honours_explicit_routes() {
        let routes = vec![HandoffRoute::new("a", "b")];
        let graph = HandoffOrchestrator::build_route_graph(&members(), &routes);
        assert_eq!(graph["a"], targets(&["b"]));
        assert!(graph["b"].is_empty());
        assert!(graph["c"].is_empty());
    }

    #[test]
    fn route_graph_keeps_targets_outside_the_members() {
        let members = vec!["a".to_owned()];
        let graph =
            HandoffOrchestrator::build_route_graph(&members, &[HandoffRoute::new("a", "ghost")]);
        assert_eq!(graph["a"], targets(&["ghost"]));
    }

    #[tokio::test]
    async fn handoff_cap_is_checked_before_the_count_moves() {
        let config = HandoffConfig {
            max_handoffs: 1,
            ..HandoffConfig::default()
        };
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        assert_eq!(orchestrator.current_agent_id(), "a");
        assert_eq!(orchestrator.handoff_count(), 0);

        assert!(orchestrator
            .request_handoff("b", Some("needs billing"))
            .await
            .unwrap());
        assert_eq!(orchestrator.current_agent_id(), "b");
        assert_eq!(orchestrator.handoff_count(), 1);

        assert!(!orchestrator.request_handoff("c", None).await.unwrap());
        assert_eq!(orchestrator.current_agent_id(), "b");
        assert_eq!(orchestrator.handoff_count(), 1);
    }

    #[tokio::test]
    async fn rejected_routes_leave_the_state_untouched() {
        let config = HandoffConfig {
            routes: vec![HandoffRoute::new("a", "b")],
            ..HandoffConfig::default()
        };
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        assert!(!orchestrator.request_handoff("c", None).await.unwrap());
        assert_eq!(orchestrator.current_agent_id(), "a");
        assert_eq!(orchestrator.handoff_count(), 0);

        assert!(orchestrator.request_handoff("b", None).await.unwrap());
        assert_eq!(orchestrator.current_agent_id(), "b");
        // `b` has no outgoing routes once explicit routes are configured.
        assert!(!orchestrator.request_handoff("a", None).await.unwrap());
        assert_eq!(orchestrator.handoff_count(), 1);
    }

    #[tokio::test]
    async fn a_ready_termination_condition_blocks_the_handoff() {
        let condition: TerminationCondition = Arc::new(|snapshot| {
            assert_eq!(snapshot.current_agent_id, "a");
            assert_eq!(snapshot.handoff_count, 0);
            futures::future::ready(Ok(true)).boxed()
        });
        let config = HandoffConfig {
            termination_condition: Some(condition),
            ..HandoffConfig::default()
        };
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        assert!(!orchestrator.request_handoff("b", None).await.unwrap());
        assert_eq!(orchestrator.handoff_count(), 0);
        assert_eq!(orchestrator.current_agent_id(), "a");
    }

    #[tokio::test]
    async fn an_async_termination_condition_blocks_the_handoff() {
        let condition: TerminationCondition =
            Arc::new(|snapshot| async move { Ok(snapshot.handoff_count > 0) }.boxed());
        let config = HandoffConfig {
            termination_condition: Some(condition),
            ..HandoffConfig::default()
        };
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        // count is 0, so the condition allows the first hop...
        assert!(orchestrator.request_handoff("b", None).await.unwrap());
        // ...and blocks the next one.
        assert!(!orchestrator.request_handoff("c", None).await.unwrap());
        assert_eq!(orchestrator.handoff_count(), 1);
    }

    #[tokio::test]
    async fn a_failing_termination_condition_propagates() {
        let condition: TerminationCondition =
            Arc::new(|_snapshot| futures::future::ready(Err(Error::validation("nope"))).boxed());
        let config = HandoffConfig {
            termination_condition: Some(condition),
            ..HandoffConfig::default()
        };
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        let error = orchestrator.request_handoff("b", None).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(orchestrator.handoff_count(), 0);
    }

    #[tokio::test]
    async fn completion_is_first_wins() {
        let config = HandoffConfig::default();
        let orchestrator = Arc::new(HandoffOrchestrator::new("a", &members(), &config));
        assert!(orchestrator.complete(json!({"answer": 42})).await);
        assert!(!orchestrator.complete(json!({"answer": 43})).await);
        assert!(!orchestrator.error(Error::validation("late")).await);
        assert_eq!(
            orchestrator
                .wait(Some(Duration::from_secs(1)))
                .await
                .unwrap(),
            json!({"answer": 42})
        );
    }

    #[tokio::test]
    async fn an_error_resolution_wins_over_a_later_completion() {
        let config = HandoffConfig::default();
        let orchestrator = Arc::new(HandoffOrchestrator::new("a", &members(), &config));
        assert!(orchestrator.error(Error::validation("boom")).await);
        assert!(!orchestrator.complete(json!({"answer": 1})).await);
        let error = orchestrator.wait(None).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn wait_times_out_and_only_once() {
        let config = HandoffConfig::default();
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        let error = orchestrator
            .wait(Some(Duration::from_millis(5)))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(error.message().contains("handoff chain timed out"));

        let error = orchestrator.wait(None).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::External);
        assert!(error.message().contains(CLOSED_MESSAGE));
    }

    #[tokio::test]
    async fn session_state_round_trips_and_partial_snapshots_start_over() {
        let config = HandoffConfig::default();
        let orchestrator = HandoffOrchestrator::new("a", &members(), &config);
        assert!(orchestrator.request_handoff("b", None).await.unwrap());

        let session = Arc::new(AgentTeamSession::new("team"));
        orchestrator.save_to_session(&session).await.unwrap();
        let snapshot = session.get_state(COORDINATOR_STATE_KEY).await.unwrap();
        assert_eq!(
            snapshot,
            Some(json!({"current_agent_id": "b", "handoff_count": 1}))
        );

        let restored =
            HandoffOrchestrator::restore_from_session(snapshot.as_ref(), "a", &members(), &config)
                .unwrap();
        assert_eq!(restored.current_agent_id(), "b");
        assert_eq!(restored.handoff_count(), 1);

        let partial = json!({"current_agent_id": "c"});
        let restored =
            HandoffOrchestrator::restore_from_session(Some(&partial), "a", &members(), &config)
                .unwrap();
        assert_eq!(restored.current_agent_id(), "a");
        assert_eq!(restored.handoff_count(), 0);

        let restored =
            HandoffOrchestrator::restore_from_session(None, "a", &members(), &config).unwrap();
        assert_eq!(restored.current_agent_id(), "a");
        assert_eq!(restored.handoff_count(), 0);
    }

    #[tokio::test]
    async fn registry_tracks_orchestrators_by_session() {
        let config = HandoffConfig::default();
        let orchestrator = Arc::new(HandoffOrchestrator::new("a", &members(), &config));
        let registry = HandoffRegistry::new();
        registry.insert("s-1", Arc::clone(&orchestrator));

        assert!(registry.get("s-1").is_some());
        assert!(registry.get("s-2").is_none());
        assert!(format!("{registry:?}").contains("s-1"));

        assert!(registry.remove("s-1").is_some());
        assert!(registry.get("s-1").is_none());
        assert!(registry.remove("s-1").is_none());

        assert_eq!(orchestrator.route_graph().len(), 3);
    }
}
