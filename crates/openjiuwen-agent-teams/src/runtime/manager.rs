//! Runtime management: the pool owner, the activation decision fan-out, and the process-global
//! registry the CLI and the MCP server enter through.
//!
//! Python parity source: `openjiuwen/agent_teams/runtime/manager.py` (`TeamRuntimeManager`) and
//! `openjiuwen/agent_teams/runtime/dispatch.py` (`decide_run_action`, consumed here).
//!
//! A manager owns exactly two things: the injected [`TeamBuildDeps`] a team is built over, and the
//! [`TeamRuntimePool`] of live teams. It holds no parallel mirror of "which teams are active" — the
//! pool is the single source of truth, per Python ("pool entries are the sole source of truth for
//! which teams are currently active; the manager itself holds no parallel `_active_*` mirror").
//!
//! The run entry point is [`TeamRuntimeManager::activate`]: it validates the blueprint, decides
//! through [`decide_run_action`] whether the request creates, attaches to, resumes or is rejected
//! by the live team under that name, and then either builds a team or hands back the live handle.
//! The caller drives the returned handle (`invoke`/`stream`/`interact`); activation itself never
//! consumes the request `inputs`, which is why every branch logs them at `debug` and does nothing
//! else with them.
//!
//! ```
//! use openjiuwen_agent_teams::runtime::manager::{global_registry, try_global_registry};
//!
//! // The process-global manager is created on first use and never rebuilt afterwards.
//! let manager = global_registry();
//! assert!(try_global_registry().is_some());
//! assert_eq!(manager.pool().len(), 0);
//! ```
//!
//! # The global registry
//!
//! [`global_registry`] is the crate's **second documented process-global singleton** (the first is
//! the process expert catalog, [`crate::experts`]). It exists because the CLI and the
//! `openjiuwen-team-mcp` server start from separate entry points but must see the same live teams.
//! Tests construct local [`TeamRuntimeManager`] instances instead and call
//! [`clear_global_registry`] when they touch the global one.
use std::fmt;
use std::sync::{Arc, OnceLock};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use parking_lot::RwLock;
use serde_json::Value;
use tracing::{debug, warn};

use super::deps::TeamBuildDeps;
use super::dispatch::{decide_run_action, LiveTeamState, RunActionKind};
use super::pool::TeamRuntimePool;
use super::team_agent::{SpecBuild, TeamAgent};
use crate::interaction::{DeliverResult, InteractPayload};
use crate::schema::{MemberLifetime, TeamAgentSpec, TeamSnapshot};

/// Error text of a blueprint with a blank team name.
const EMPTY_TEAM_NAME_MESSAGE: &str = "team_name must not be empty";

/// Error text of an activation without a session.
const EMPTY_SESSION_MESSAGE: &str = "session_id must not be empty";

/// The process-global manager slot.
///
/// `OnceLock` guarantees the slot itself is created once; the `RwLock` around the `Option` lets
/// [`set_global_registry`] and [`clear_global_registry`] replace or drop the manager without
/// re-initialising the cell. The inner lock is held for one read or one write and never across an
/// `.await`.
static GLOBAL_REGISTRY: OnceLock<RwLock<Option<Arc<TeamRuntimeManager>>>> = OnceLock::new();

/// Returns the process-global registry cell, creating it empty on first use.
fn registry_cell() -> &'static RwLock<Option<Arc<TeamRuntimeManager>>> {
    GLOBAL_REGISTRY.get_or_init(|| RwLock::new(None))
}

/// Owns the live team pool and every operation the run entry points need.
///
/// Cheap to share (`Arc<TeamRuntimeManager>`): the only mutable state is the pool, which locks per
/// operation. A manager is either the process-global one ([`global_registry`]) or a local instance
/// a test (or an embedded host) drives itself.
pub struct TeamRuntimeManager {
    /// Everything a team is built over.
    deps: TeamBuildDeps,
    /// Live teams, grouped by activating session.
    pool: Arc<TeamRuntimePool>,
}

impl fmt::Debug for TeamRuntimeManager {
    /// Prints the build seams and how many teams are live, never the teams themselves.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamRuntimeManager")
            .field("deps", &self.deps)
            .field("teams", &self.pool.len())
            .finish_non_exhaustive()
    }
}

impl TeamRuntimeManager {
    /// Builds a manager over `deps` with an empty pool.
    #[must_use]
    pub fn new(deps: TeamBuildDeps) -> Self {
        Self {
            deps,
            pool: Arc::new(TeamRuntimePool::new()),
        }
    }

    /// Builds a manager over the in-process defaults ([`TeamBuildDeps::default`]).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TeamBuildDeps::default())
    }

    /// Returns the dependencies every team of this manager is built over.
    #[must_use]
    pub fn deps(&self) -> &TeamBuildDeps {
        &self.deps
    }

    /// Returns the live team pool.
    #[must_use]
    pub fn pool(&self) -> Arc<TeamRuntimePool> {
        Arc::clone(&self.pool)
    }

    /// Activates `spec` for `session_id`: creates, attaches to, resumes or rejects the run.
    ///
    /// Python parity: `TeamRuntimeManager.activate`. The steps are:
    ///
    /// 1. validate the blueprint and refuse a blank team name or session;
    /// 2. decide through [`decide_run_action`] with the live state of the team under that name;
    /// 3. serve the request:
    ///    * `create_team` — build over [`Self::deps`], start the team and register it in the pool;
    ///    * `attach_existing` / `resume_existing` — start the live team when it is not started
    ///      (a failure is logged, not propagated: the caller's run reports it) and return it;
    ///    * `resume_paused` — resume the live runtime (a failure is logged, not propagated);
    ///    * any `reject_*` — [`Error::Validation`] carrying the decision's reason.
    ///
    /// `inputs` is **not** consumed by any branch: the caller drives the returned handle with it.
    /// It is logged at `debug` so an operator can see which request activated the team.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid blueprint, a blank team name or session, and every
    /// rejection [`decide_run_action`] produces; the build errors of
    /// [`SpecBuild::build_with`] (unsupported transport,
    /// store, workspace, harness factory) and the leader runtime's start error on creation.
    pub async fn activate(
        &self,
        spec: &TeamAgentSpec,
        session_id: &str,
        inputs: &Value,
    ) -> Result<Arc<TeamAgent>> {
        spec.validate()?;
        let team_name = spec.team_name.clone();
        if team_name.trim().is_empty() {
            return Err(Error::validation(EMPTY_TEAM_NAME_MESSAGE));
        }
        if session_id.trim().is_empty() {
            return Err(Error::validation(EMPTY_SESSION_MESSAGE));
        }
        debug!(
            team = %team_name,
            session = session_id,
            inputs = %inputs,
            "activating a team; the caller drives the returned handle"
        );

        // The live handle is read once so the decision and the branch it selects agree even when a
        // concurrent task stops the team in between.
        let live = self.pool.find(&team_name);
        let live_state = live
            .as_ref()
            .map(|(session_id, agent)| live_state_of(session_id, agent));
        let action = decide_run_action(live_state.as_ref(), session_id, spec.lifecycle);
        if action.kind.is_reject() {
            return Err(Error::validation(action.reason.unwrap_or_else(|| {
                format!(
                    "team '{team_name}' cannot be activated ({})",
                    action.kind.as_str()
                )
            })));
        }
        if let Some((live_session, agent)) = live {
            debug!(
                team = %team_name,
                session = %live_session,
                action = action.kind.as_str(),
                "reusing the live team"
            );
            if action.kind == RunActionKind::ResumePaused {
                if let Err(error) = agent.runtime().resume(None).await {
                    warn!(
                        team = %team_name,
                        error = %error,
                        "failed to resume the paused team; returning it as-is"
                    );
                }
            } else {
                start_quietly(&agent).await;
            }
            return Ok(agent);
        }

        let agent = spec.build_with(self.deps.clone()).await?;
        agent.start().await?;
        self.pool.insert(session_id, Arc::clone(&agent));
        debug!(
            team = %team_name,
            session = session_id,
            "created and started a new team"
        );
        Ok(agent)
    }

    /// Returns the live team registered under `team_name`, regardless of its session.
    #[must_use]
    pub fn live(&self, team_name: &str) -> Option<Arc<TeamAgent>> {
        self.pool.find(team_name).map(|(_, agent)| agent)
    }

    /// Returns the live state of `team_name`: its session, lifecycle, paused flag and state.
    ///
    /// The session reported is the one the team was activated with (the pool key), not the session
    /// its members' harnesses run under — `TeamAgent::build_with` mints the latter itself.
    #[must_use]
    pub fn live_state(&self, team_name: &str) -> Option<LiveTeamState> {
        self.pool
            .find(team_name)
            .map(|(session_id, agent)| live_state_of(&session_id, &agent))
    }

    /// Routes one external payload to the leader mailbox of `team_name`.
    ///
    /// Python parity: `TeamRuntimeManager.interact`. An unknown team is
    /// [`DeliverResult::not_active`] and a closed gate is [`DeliverResult::gate_closed`]; otherwise
    /// the payload goes through [`TeamAgent::deliver`], which admits it through the gate and
    /// persists/routes it (a `HumanAgent`/`Operator` payload lands in the leader mailbox).
    pub async fn interact(&self, team_name: &str, payload: InteractPayload) -> DeliverResult {
        let Some(agent) = self.live(team_name) else {
            debug!(team = team_name, "interact for an unknown team");
            return DeliverResult::not_active();
        };
        if agent.gate().is_closed() {
            debug!(team = team_name, "interact while the gate is closed");
            return DeliverResult::gate_closed();
        }
        agent.deliver(payload).await
    }

    /// Stops the team of `session_id` and drops it from the pool.
    ///
    /// Returns `true` when a live team of that session was stopped and removed. A team live in a
    /// different session is left untouched and reports `false` — a temporary team never serves a
    /// second session, so a mismatched stop must not tear down the owner's runtime.
    ///
    /// # Errors
    ///
    /// This implementation logs teardown failures and returns `Ok`; the signature is fallible so a
    /// future store-backed teardown can report them.
    pub async fn stop_team(&self, team_name: &str, session_id: &str) -> Result<bool> {
        let Some((live_session, agent)) = self.pool.find(team_name) else {
            debug!(team = team_name, "stop_team for an unknown team");
            return Ok(false);
        };
        if live_session != session_id {
            debug!(
                team = team_name,
                live_session = %live_session,
                requested_session = session_id,
                "stop_team refused: the team belongs to another session"
            );
            return Ok(false);
        }
        if let Err(error) = agent.stop().await {
            warn!(team = team_name, error = %error, "failed to stop the team while removing it");
        }
        self.pool.remove(&live_session, team_name);
        debug!(team = team_name, session = %live_session, "team stopped and removed");
        Ok(true)
    }

    /// Retires one member of `team_name` and removes it from the team's roster.
    ///
    /// The member's runtime is stopped (failures are logged), its row settles on
    /// [`MemberStatus::ShutDown`](crate::schema::MemberStatus::ShutDown) through
    /// [`TeamMember::shutdown`](crate::member::TeamMember::shutdown), and the backend drops it from
    /// the roster and the activity registry.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown team or member, and the lifecycle machine's or the
    /// store's own errors while settling the row.
    pub async fn finalize_member(&self, team_name: &str, member_name: &str) -> Result<()> {
        let Some(agent) = self.live(team_name) else {
            return Err(Error::not_found(format!(
                "team '{team_name}' is not running"
            )));
        };
        let backend = agent.backend();
        let Some(member) = backend.member(member_name) else {
            return Err(Error::not_found(format!(
                "member '{member_name}' not found in team '{team_name}'"
            )));
        };
        if let Some(runtime) = member.runtime() {
            if let Err(error) = runtime.stop().await {
                warn!(
                    team = team_name,
                    member = member_name,
                    error = %error,
                    "failed to stop the member runtime while finalizing it"
                );
            }
        }
        member.shutdown(MemberLifetime::Shutdown, true).await?;
        backend.remove_member(member_name);
        debug!(team = team_name, member = member_name, "member finalized");
        Ok(())
    }

    /// Stops the team registered under `team_name` and drops it from the pool.
    ///
    /// Returns `true` when a live team was stopped and removed, `false` when no team was live.
    ///
    /// # Errors
    ///
    /// This implementation logs teardown failures and returns `Ok`; the signature is fallible so a
    /// future store-backed teardown can report them.
    pub async fn remove(&self, team_name: &str) -> Result<bool> {
        let Some((session_id, agent)) = self.pool.find(team_name) else {
            return Ok(false);
        };
        if let Err(error) = agent.stop().await {
            warn!(team = team_name, error = %error, "failed to stop the team while removing it");
        }
        self.pool.remove(&session_id, team_name);
        debug!(team = team_name, session = %session_id, "team removed");
        Ok(true)
    }

    /// Stops every live team and empties the pool.
    ///
    /// Failures are logged, never propagated: shutdown is a cleanup path and must reach every team.
    /// The process-global registry is **not** cleared here — a manager does not know whether it is
    /// the registered one, so the caller that owns the global clears it through
    /// [`clear_global_registry`].
    ///
    /// # Errors
    ///
    /// This implementation always returns `Ok` (see above).
    pub async fn shutdown(&self) -> Result<()> {
        for (session_id, team_names) in self.pool.snapshot() {
            for team_name in &team_names {
                if let Some(agent) = self.pool.get(&session_id, team_name) {
                    if let Err(error) = agent.stop().await {
                        warn!(
                            team = %team_name,
                            session = %session_id,
                            error = %error,
                            "failed to stop a team during shutdown"
                        );
                    }
                }
            }
            self.pool.remove_session(&session_id);
        }
        debug!("team runtime manager shut down; the pool is empty");
        Ok(())
    }

    /// Returns the roster and task-board snapshot of `team_name`, or `None` when it is not live.
    pub async fn snapshot(&self, team_name: &str) -> Option<TeamSnapshot> {
        let agent = self.live(team_name)?;
        Some(agent.snapshot().await)
    }
}

/// Projects the live handle and its session onto the decision input.
fn live_state_of(session_id: &str, agent: &Arc<TeamAgent>) -> LiveTeamState {
    let state = agent.state();
    LiveTeamState {
        team_name: agent.team_name().to_owned(),
        session_id: session_id.to_owned(),
        lifecycle: agent.blueprint().lifecycle,
        paused: matches!(state, HarnessState::Paused | HarnessState::Pausing),
        state,
    }
}

/// Starts `agent` when it is not started, logging (never propagating) a failure.
///
/// A resumed or attached team must be running before the caller drives it, but a start failure here
/// is not fatal to the activation: the caller's own run reports the same error with its context.
async fn start_quietly(agent: &TeamAgent) {
    if agent.is_started() {
        return;
    }
    if let Err(error) = agent.start().await {
        warn!(
            team = agent.team_name(),
            error = %error,
            "failed to start the live team; the caller's run will report it"
        );
    }
}

/// Returns the process-global manager, creating it over [`TeamBuildDeps::default`] on first use.
///
/// Python parity: the runner-scoped `TeamRuntimeManager` the CLI and the agent-team MCP tools share.
/// This is the crate's second documented process-global singleton; tests that touch it must call
/// [`clear_global_registry`] so the next test sees a clean slot.
#[must_use]
pub fn global_registry() -> Arc<TeamRuntimeManager> {
    let cell = registry_cell();
    let existing = cell.read().clone();
    if let Some(manager) = existing {
        return manager;
    }
    let mut slot = cell.write();
    // Another task may have created the manager while this one waited for the write lock.
    if let Some(manager) = slot.as_ref() {
        return Arc::clone(manager);
    }
    let manager = Arc::new(TeamRuntimeManager::with_defaults());
    *slot = Some(Arc::clone(&manager));
    manager
}

/// Replaces the process-global manager with `manager`.
///
/// A host that wants its own dependencies (a CLI with a configured store, a test) installs them
/// this way before the first [`global_registry`] call it cares about.
pub fn set_global_registry(manager: Arc<TeamRuntimeManager>) {
    *registry_cell().write() = Some(manager);
}

/// Returns the process-global manager when one was installed, without creating one.
///
/// Unlike [`global_registry`] this never builds a manager over the default dependencies, so a
/// caller that only wants to know whether a registry exists pays nothing.
#[must_use]
pub fn try_global_registry() -> Option<Arc<TeamRuntimeManager>> {
    registry_cell().read().clone()
}

/// Drops the process-global manager, if any.
///
/// The live teams of that manager are *not* stopped: call [`TeamRuntimeManager::shutdown`] first
/// when they must be torn down. Tests call this to keep the slot from leaking between cases.
pub fn clear_global_registry() {
    *registry_cell().write() = None;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::interaction::{GodViewMessage, OperatorMessage};
    use crate::member::TeamMember;
    use crate::runtime::team_agent::test_support::{
        blueprint, deps as test_deps, RecordingWorkspaceProvider, ScriptedFactory,
    };
    use crate::schema::{MemberStatus, TeamLifecycle};

    /// Builds a manager over fresh scripted fakes and returns it with them.
    fn manager() -> (
        TeamRuntimeManager,
        Arc<ScriptedFactory>,
        Arc<RecordingWorkspaceProvider>,
    ) {
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        let deps = test_deps(&factory, &workspace);
        (TeamRuntimeManager::new(deps), factory, workspace)
    }

    /// Builds a temporary team blueprint `alpha` led by `boss` with one teammate.
    fn alpha() -> TeamAgentSpec {
        blueprint("alpha", "boss", &["dev-1"])
    }

    /// Builds an operator payload asking for status.
    fn operator(body: &str) -> InteractPayload {
        InteractPayload::Operator(OperatorMessage {
            target: None,
            body: body.to_owned(),
            sender: None,
        })
    }

    #[tokio::test]
    async fn activate_creates_starts_and_registers_the_team() {
        let (manager, factory, _workspace) = manager();
        let agent = manager
            .activate(&alpha(), "session-a", &json!({"query": "ship it"}))
            .await
            .unwrap();

        assert_eq!(agent.team_name(), "alpha");
        assert!(agent.is_started());
        assert_eq!(manager.pool().len(), 1);
        assert!(Arc::ptr_eq(&manager.live("alpha").unwrap(), &agent));

        let state = manager.live_state("alpha").unwrap();
        assert_eq!(state.team_name, "alpha");
        assert_eq!(state.session_id, "session-a");
        assert_eq!(state.lifecycle, TeamLifecycle::Temporary);
        assert!(!state.paused);
        assert_eq!(state.state, HarnessState::Idle);
        assert_eq!(factory.calls().len(), 1, "one harness for the leader");

        assert!(manager.live("beta").is_none());
        assert!(manager.live_state("beta").is_none());
    }

    #[tokio::test]
    async fn activate_on_the_same_session_attaches_the_same_team() {
        let (manager, factory, _workspace) = manager();
        let spec = alpha();
        let first = manager
            .activate(&spec, "session-a", &Value::Null)
            .await
            .unwrap();
        let second = manager
            .activate(&spec, "session-a", &json!({"query": "again"}))
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second), "the live handle is reused");
        assert_eq!(manager.pool().len(), 1);
        assert_eq!(factory.calls().len(), 1, "no second build happened");
    }

    #[tokio::test]
    async fn activate_rejects_a_cross_session_temporary_team() {
        let (manager, _factory, _workspace) = manager();
        let spec = alpha();
        manager
            .activate(&spec, "session-a", &Value::Null)
            .await
            .unwrap();

        let error = manager
            .activate(&spec, "session-b", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("team 'alpha' is already running in session 'session-a'"),
            "{error}"
        );
        assert!(error.to_string().contains("session-b"), "{error}");
        assert_eq!(manager.pool().len(), 1);
    }

    #[tokio::test]
    async fn activate_resumes_a_persistent_team_across_sessions() {
        let (manager, factory, _workspace) = manager();
        let mut spec = alpha();
        spec.lifecycle = TeamLifecycle::Persistent;
        let first = manager
            .activate(&spec, "session-a", &Value::Null)
            .await
            .unwrap();
        let second = manager
            .activate(&spec, "session-b", &Value::Null)
            .await
            .unwrap();

        assert!(
            Arc::ptr_eq(&first, &second),
            "the persistent team is resumed"
        );
        assert_eq!(factory.calls().len(), 1, "no second build happened");
        assert_eq!(
            manager.live_state("alpha").unwrap().session_id,
            "session-a",
            "the pool keeps the activating session"
        );
    }

    #[tokio::test]
    async fn activate_rejects_a_stopped_team() {
        let (manager, _factory, _workspace) = manager();
        let spec = alpha();
        let agent = manager
            .activate(&spec, "session-a", &Value::Null)
            .await
            .unwrap();
        agent.stop().await.unwrap();

        let error = manager
            .activate(&spec, "session-a", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: team 'alpha' has stopped; start a new team run"
        );
    }

    #[tokio::test]
    async fn activate_validates_the_name_and_the_session() {
        let (manager, _factory, _workspace) = manager();
        let mut blank = alpha();
        blank.team_name = "   ".to_owned();
        let error = manager
            .activate(&blank, "session-a", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            format!("validation: {EMPTY_TEAM_NAME_MESSAGE}")
        );

        let error = manager
            .activate(&alpha(), "  ", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            format!("validation: {EMPTY_SESSION_MESSAGE}")
        );

        let mut invalid = alpha();
        invalid.agents.remove("leader");
        let error = manager
            .activate(&invalid, "session-a", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: agents dict must contain a 'leader' key"
        );
        assert!(manager.pool().is_empty(), "nothing was built");
    }

    #[tokio::test]
    async fn interact_routes_to_the_leader_mailbox() {
        let (manager, _factory, _workspace) = manager();
        let agent = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();

        let result = manager
            .interact("alpha", operator("what is the status?"))
            .await;
        assert!(result.ok, "{result:?}");

        let unread = agent
            .backend()
            .unread(agent.leader_member_name())
            .await
            .unwrap();
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].content, "what is the status?");
    }

    #[tokio::test]
    async fn interact_reports_not_active_and_gate_closed() {
        let (manager, _factory, _workspace) = manager();
        let mut result = manager.interact("unknown", operator("hello")).await;
        assert_eq!(result.reason.as_deref(), Some("not_active"));

        let agent = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();
        agent.gate().close();
        result = manager.interact("alpha", operator("hello")).await;
        assert_eq!(result.reason.as_deref(), Some("gate_closed"));

        assert!(manager.stop_team("alpha", "session-a").await.unwrap());
        result = manager.interact("alpha", operator("hello")).await;
        assert_eq!(result.reason.as_deref(), Some("not_active"));
        assert!(!result.ok);
    }

    #[tokio::test]
    async fn interact_accepts_a_god_view_payload() {
        let (manager, _factory, _workspace) = manager();
        let agent = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();

        let result = manager
            .interact(
                "alpha",
                InteractPayload::GodView(GodViewMessage {
                    body: "hello from ops".to_owned(),
                    sender: Some("ops".to_owned()),
                }),
            )
            .await;
        assert!(result.ok, "{result:?}");
        let unread = agent
            .backend()
            .unread(agent.leader_member_name())
            .await
            .unwrap();
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].content, "hello from ops");
        assert_eq!(unread[0].from_member_name, "ops");
    }

    #[tokio::test]
    async fn stop_team_of_another_session_is_a_no_op() {
        let (manager, _factory, _workspace) = manager();
        let agent = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();

        assert!(!manager.stop_team("alpha", "session-b").await.unwrap());
        assert!(agent.is_started(), "the owner's team is untouched");
        assert!(manager.live("alpha").is_some());

        assert!(manager.stop_team("alpha", "session-a").await.unwrap());
        assert!(!agent.is_started());
        assert!(manager.live("alpha").is_none());
        assert!(manager.pool().is_empty());
        assert!(!manager.stop_team("alpha", "session-a").await.unwrap());
    }

    #[tokio::test]
    async fn finalize_member_retires_and_removes_the_member() {
        let (manager, _factory, _workspace) = manager();
        let agent = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();
        agent.spawn_member("dev-1").await.unwrap();
        let member: Arc<TeamMember> = agent.backend().member("dev-1").unwrap();
        assert_eq!(member.status(), MemberStatus::Ready);

        manager.finalize_member("alpha", "dev-1").await.unwrap();
        assert_eq!(member.status(), MemberStatus::ShutDown);
        assert!(agent.backend().member("dev-1").is_none());
        assert_eq!(agent.roster(), vec!["boss".to_owned()]);
    }

    #[tokio::test]
    async fn finalize_member_reports_unknown_teams_and_members() {
        let (manager, _factory, _workspace) = manager();
        let error = manager
            .finalize_member("unknown", "dev-1")
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: team 'unknown' is not running"
        );

        manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();
        let error = manager.finalize_member("alpha", "ghost").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: member 'ghost' not found in team 'alpha'"
        );
    }

    #[tokio::test]
    async fn remove_and_shutdown_empty_the_pool() {
        let (manager, _factory, _workspace) = manager();
        let agent = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();
        assert!(manager.remove("alpha").await.unwrap());
        assert!(!agent.is_started());
        assert!(!manager.remove("alpha").await.unwrap());

        let alpha = manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();
        let beta = manager
            .activate(&blueprint("beta", "boss", &[]), "session-b", &Value::Null)
            .await
            .unwrap();
        assert_eq!(manager.pool().len(), 2);

        manager.shutdown().await.unwrap();
        assert!(manager.pool().is_empty());
        assert!(!alpha.is_started());
        assert!(!beta.is_started());
        assert!(manager.live("alpha").is_none());
    }

    #[tokio::test]
    async fn snapshot_reports_the_live_team_or_nothing() {
        let (manager, _factory, _workspace) = manager();
        assert!(manager.snapshot("alpha").await.is_none());

        manager
            .activate(&alpha(), "session-a", &Value::Null)
            .await
            .unwrap();
        let snapshot = manager.snapshot("alpha").await.unwrap();
        assert_eq!(snapshot.team_name, "alpha");
        assert_eq!(snapshot.state, HarnessState::Idle);
        let members: Vec<&str> = snapshot
            .members
            .iter()
            .map(|member| member.member_name.as_str())
            .collect();
        assert_eq!(members, vec!["boss", "dev-1"]);
        assert_eq!(snapshot.tasks, Vec::new());
    }

    #[test]
    fn debug_names_the_seams_and_the_team_count() {
        let manager = TeamRuntimeManager::with_defaults();
        let debug = format!("{manager:?}");
        assert!(debug.starts_with("TeamRuntimeManager"), "{debug}");
        assert!(debug.contains("teams: 0"), "{debug}");
    }

    #[test]
    fn global_registry_is_created_replaced_and_cleared() {
        clear_global_registry();
        assert!(try_global_registry().is_none());

        let created = global_registry();
        assert!(Arc::ptr_eq(&global_registry(), &created), "created once");
        assert!(Arc::ptr_eq(&try_global_registry().unwrap(), &created));

        let custom = Arc::new(TeamRuntimeManager::with_defaults());
        set_global_registry(Arc::clone(&custom));
        assert!(Arc::ptr_eq(&try_global_registry().unwrap(), &custom));
        assert!(Arc::ptr_eq(&global_registry(), &custom), "the slot is kept");

        clear_global_registry();
        assert!(try_global_registry().is_none());
        // A cleared slot is re-created over the defaults on the next call.
        assert!(!Arc::ptr_eq(&global_registry(), &custom));
        clear_global_registry();
    }
}
