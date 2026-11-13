//! [`TeamAgent`]: the team agent itself, one type for the leader and every teammate.
//!
//! Python parity source: `openjiuwen/agent_teams/agent/team_agent.py` (`TeamAgent`) and
//! `agent/agent_configurator.py` (the wiring `TeamAgent.__init__` performs). Python has one class
//! for both roles; this port keeps that: [`TeamAgent`] carries a
//! [`TeamRole`](crate::schema::TeamRole), a leader dispatches and steers while a teammate polls the
//! board and answers mail.
//!
//! The struct is a *cheap handle*: every field lives inside one `Arc<TeamAgentInner>`, so cloning
//! it (the run loop does, on every `invoke`) copies one pointer. Building is the only fallible,
//! `await`-heavy step; it lives in [`build`], which owns the blueprint →
//! runtime path and is pinned by `src/runtime/CONTEXT.md` §"Build path".
//!
//! # Roles
//!
//! [`SpecBuild::build`] always produces the *leader* handle: a teammate is not a second public
//! constructor but a [`TeamMember`](crate::member::TeamMember) whose runtime is materialized by
//! [`TeamAgent::spawn_member`] through the same harness-factory seam. The role field exists because
//! the shared run loop and the adapter must know which half of the contract they are executing.
//!
//! # Async build
//!
//! `team_agent`/`leader` must await the store, the workspace provider and the harness factory, so
//! [`SpecBuild`] is an `#[async_trait]` seam. The node contract sketches the trait with synchronous
//! signatures; the build path it pins is the authority (this deviation is reported by the node).
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use openjiuwen_multiagent::{TeamCard, TeamRuntime};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_util::sync::CancellationToken;

use super::attach_host::AttachHost;
use super::backend::TeamBackend;
use super::deps::TeamBuildDeps;
use super::health::TeamHealth;
use super::observability_bind::TeamSignals;
use super::routing::TeamModelRouter;
use super::spawner::TeamMemberSpawner;
use super::spec_meta::{AttachConfig, ForkDirective};
use crate::interaction::{HumanAgentInbox, InteractGate};
use crate::kv_cache::TeamKvCacheAffinity;
use crate::member::{MemberActivityRegistry, MemberRuntime, MemberSpawnOptions, TeamMember};
use crate::schema::{EventMessage, TeamAgentSpec, TeamRole, TeamSpec};
use crate::storage::TeamStore;
use crate::workspace::{IsolationMode, MemberWorkspace};

mod attach;
mod lifecycle;
mod members;
mod org_ops;
mod signals;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
pub(crate) mod test_support;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

/// Error text of a `create_task` call on a non-leader handle.
const LEADER_ONLY_TASKS_MESSAGE: &str = "only the leader can create tasks";

/// Builds a [`TeamAgent`] from a team blueprint.
///
/// Python parity: `TeamAgentSpec.build()` / `build_with(...)`. The trait is async because the
/// build path awaits the store, the workspace provider and the harness factory; Python's `build()`
/// is synchronous only because it defers the same work to the first run.
#[async_trait]
pub trait SpecBuild {
    /// Builds the leader agent from the in-process defaults ([`TeamBuildDeps::default`]).
    ///
    /// # Errors
    ///
    /// Returns every error [`SpecBuild::build_with`] returns.
    async fn build(&self) -> Result<Arc<TeamAgent>>;

    /// Builds the leader agent over the injected dependencies.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the blueprint is invalid (reserved names, missing leader
    /// blueprint, bad dispatch knobs), [`Error::Unsupported`] for a transport this build does not
    /// implement, and propagates store, workspace, allocator and harness-factory failures.
    async fn build_with(&self, deps: TeamBuildDeps) -> Result<Arc<TeamAgent>>;
}

/// One built, live team: the leader handle, its roster and its seams.
///
/// Every accessor is a cheap read over an [`Arc`]; the only mutable state is the started flag, the
/// cancellation token of the current run, the chunk fan-out and the single-consumer bus receiver.
pub struct TeamAgent {
    /// All team state; cloned by value, shared by pointer.
    inner: Arc<TeamAgentInner>,
}

impl Clone for TeamAgent {
    /// Clones the handle, not the team: both copies see the same roster and the same run token.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl fmt::Debug for TeamAgent {
    /// Prints the identity an operator needs: team, role, member, session and run state. The
    /// blueprint and the store are omitted (they carry prompts and potentially credentials).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamAgent")
            .field("team_name", &self.inner.spec.team_name)
            .field("role", &self.inner.role)
            .field("member_name", &self.inner.member_name)
            .field("session_id", &self.inner.session_id)
            .field("started", &self.is_started())
            .field("members", &self.inner.backend.len())
            .finish_non_exhaustive()
    }
}

/// Everything one `TeamAgent` handle owns.
pub(crate) struct TeamAgentInner {
    /// The cold blueprint the team was built from.
    pub(crate) blueprint: TeamAgentSpec,
    /// The runtime mirror the running team carries ([`TeamSpec::from_spec`]).
    pub(crate) spec: TeamSpec,
    /// Role of this handle (always [`TeamRole::Leader`] from [`SpecBuild`]).
    pub(crate) role: TeamRole,
    /// Wire name of this handle's member.
    pub(crate) member_name: String,
    /// Session id every harness of the team is opened under.
    pub(crate) session_id: String,
    /// Resolved prompt language (`cn` / `cf` / `en`).
    pub(crate) language: String,
    /// Card of this handle's member, named `<team>_<member>`.
    pub(crate) card: AgentCard,
    /// Member name of the leader, resolved from the blueprint.
    pub(crate) leader_member_name: String,
    /// Persisted state of the team.
    pub(crate) store: Arc<dyn TeamStore>,
    /// Organisation facade: members, mail, events, tasks and snapshots.
    pub(crate) backend: Arc<TeamBackend>,
    /// This handle's member row and lifecycle machine.
    pub(crate) member: Arc<TeamMember>,
    /// The harness-backed runtime attached to [`Self::member`] (the same `Arc`).
    pub(crate) runtime: Arc<dyn MemberRuntime>,
    /// Materializes teammate runtimes from the blueprint.
    pub(crate) spawner: Arc<TeamMemberSpawner>,
    /// Admission control for external payloads.
    pub(crate) gate: Arc<InteractGate>,
    /// Leader-side activity projection.
    pub(crate) activity: Arc<MemberActivityRegistry>,
    /// Human-agent inbox (disabled unless `enable_hitt`).
    pub(crate) inbox: Arc<HumanAgentInbox>,
    /// The `openjiuwen_multiagent` session/registry seam.
    pub(crate) team_runtime: Arc<TeamRuntime>,
    /// Model pool + router; the cursor advances per allocated member.
    pub(crate) routing: Arc<TeamModelRouter>,
    /// KV-cache affinity registry shared by every member of the run.
    pub(crate) kv_cache: Arc<TeamKvCacheAffinity>,
    /// Observability hooks and the optional monitor sink.
    pub(crate) signals: TeamSignals,
    /// Workspace isolation requested by the spec (`none` by default).
    pub(crate) isolation: IsolationMode,
    /// Workspaces this team allocated, keyed by member name (leader included).
    ///
    /// The build path records the leader's workspace and `spawn_member` records each teammate's;
    /// [`TeamAgent::shutdown_member`] and [`TeamAgent::stop`] take them back out and hand them to
    /// [`WorkspaceProvider::release`](crate::workspace::WorkspaceProvider::release), so a worktree
    /// is unregistered exactly once instead of leaking for the team's lifetime.
    pub(crate) workspaces: Mutex<BTreeMap<String, MemberWorkspace>>,
    /// Spawn options requested by the spec (in-process by default).
    pub(crate) spawn_options: MemberSpawnOptions,
    /// Fork directive, when this team forks a member from a parent.
    pub(crate) fork: Option<ForkDirective>,
    /// Attach-host settings, when a socket is configured.
    pub(crate) attach: Option<AttachConfig>,
    /// The bound attach server of the current run.
    pub(crate) attach_host: Mutex<Option<AttachHost>>,
    /// Whether [`TeamAgent::start`] completed and no [`TeamAgent::stop`] followed.
    pub(crate) started: AtomicBool,
    /// Token of the current run; replaced by every `start`, cancelled by `stop`.
    pub(crate) cancel: Mutex<CancellationToken>,
    /// Fan-out of this handle's runtime chunks to [`TeamAgent::subscribe_chunks`] callers.
    pub(crate) chunk_sinks: Arc<Mutex<Vec<UnboundedSender<Value>>>>,
    /// Single-consumer bus receiver, taken once by [`super::leader::run`].
    pub(crate) events: Mutex<Option<UnboundedReceiver<EventMessage>>>,
    /// Queued steering messages the leader takes per model call.
    pub(crate) steer_batch_size: usize,
}

/// Rejects a `create_task` call made by any role but the leader.
fn ensure_leader_role(role: TeamRole) -> Result<()> {
    if role == TeamRole::Leader {
        Ok(())
    } else {
        Err(Error::validation(LEADER_ONLY_TASKS_MESSAGE))
    }
}

impl TeamAgent {
    /// Assembles a handle from its parts; [`super::build`] is the only caller.
    ///
    /// Keeping the constructor here leaves the inner fields crate-visible without adding a second
    /// public way to build a team.
    pub(crate) fn assemble(inner: TeamAgentInner) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Returns this member's card, named `<team>_<member>`.
    #[must_use]
    pub fn card(&self) -> &AgentCard {
        &self.inner.card
    }

    /// Returns the team name every row and event is scoped to.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.inner.spec.team_name
    }

    /// Returns this handle's role.
    #[must_use]
    pub fn role(&self) -> TeamRole {
        self.inner.role
    }

    /// Returns this handle's member name.
    #[must_use]
    pub fn member_name(&self) -> &str {
        &self.inner.member_name
    }

    /// Returns the session id the team's harnesses run under.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// Returns the leader's member name.
    #[must_use]
    pub fn leader_member_name(&self) -> &str {
        &self.inner.leader_member_name
    }

    /// Returns the runtime mirror of the blueprint the team runs on.
    #[must_use]
    pub fn spec(&self) -> &TeamSpec {
        &self.inner.spec
    }

    /// Returns the cold blueprint the team was built from.
    #[must_use]
    pub fn blueprint(&self) -> &TeamAgentSpec {
        &self.inner.blueprint
    }

    /// Returns the resolved prompt language (`cn` / `cf` / `en`).
    #[must_use]
    pub fn language(&self) -> &str {
        &self.inner.language
    }

    /// Returns the attached runtime's lifecycle phase, or [`HarnessState::Idle`] before a start.
    #[must_use]
    pub fn state(&self) -> HarnessState {
        self.inner
            .member
            .runtime()
            .map_or(HarnessState::Idle, |runtime| runtime.state())
    }

    /// Returns `true` between a completed [`Self::start`] and the next [`Self::stop`].
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.inner.started.load(Ordering::SeqCst)
    }
    /// Returns the member-liveness projection of the team, with the run flag filled in.
    ///
    /// The roster projection comes from [`Self::backend`] (`TeamHealth::health`); `started` comes
    /// from this handle, because the backend does not track the run lifecycle. Purely a read of
    /// already-cached state: no store access, no probe, no `.await`.
    #[must_use]
    pub fn health(&self) -> TeamHealth {
        let mut health = self.backend().health();
        health.started = self.is_started();
        health
    }

    /// Returns the organisation facade of the team.
    #[must_use]
    pub fn backend(&self) -> Arc<TeamBackend> {
        Arc::clone(&self.inner.backend)
    }

    /// Returns the fork directive the team was built with, when one is configured.
    ///
    /// Later teammates inherit the directive through the spawner; this accessor exposes it to
    /// hosts inspecting the build.
    #[must_use]
    pub fn fork_directive(&self) -> Option<&super::spec_meta::ForkDirective> {
        self.inner.fork.as_ref()
    }
    /// Returns the interaction gate of the current run cycle.
    #[must_use]
    pub fn gate(&self) -> Arc<InteractGate> {
        Arc::clone(&self.inner.gate)
    }

    /// Returns the leader-side activity projection.
    #[must_use]
    pub fn activity(&self) -> Arc<MemberActivityRegistry> {
        Arc::clone(&self.inner.activity)
    }

    /// Returns the human-agent inbox.
    #[must_use]
    pub fn inbox(&self) -> Arc<HumanAgentInbox> {
        Arc::clone(&self.inner.inbox)
    }

    /// Returns this handle's member row.
    #[must_use]
    pub fn member(&self) -> Arc<TeamMember> {
        Arc::clone(&self.inner.member)
    }

    /// Returns this handle's runtime.
    ///
    /// The handle always has one: the build path attaches the leader's runtime and refuses to
    /// build without it.
    #[must_use]
    pub fn runtime(&self) -> Arc<dyn MemberRuntime> {
        Arc::clone(&self.inner.runtime)
    }

    /// Returns the teammate spawner of the team.
    #[must_use]
    pub fn spawner(&self) -> Arc<TeamMemberSpawner> {
        Arc::clone(&self.inner.spawner)
    }

    /// Returns the `openjiuwen_multiagent` runtime seam of the team.
    #[must_use]
    pub fn team_runtime(&self) -> Arc<TeamRuntime> {
        Arc::clone(&self.inner.team_runtime)
    }

    /// Returns the registry card of the whole team.
    ///
    /// The id and name are the team name and every current roster member contributes one card
    /// named `<team>_<member>`, in roster order (sorted).
    #[must_use]
    pub fn team_card(&self) -> TeamCard {
        let agent_cards = self
            .roster()
            .into_iter()
            .map(|member| AgentCard::new(format!("{}_{}", self.team_name(), member)))
            .collect();
        TeamCard::new(self.team_name())
            .with_id(self.team_name())
            .with_description(self.description())
            .with_agent_cards(agent_cards)
    }

    /// Returns the roster member names, sorted.
    #[must_use]
    pub fn roster(&self) -> Vec<String> {
        self.inner.backend.member_names()
    }

    /// Returns the team's steering batch size.
    pub(crate) fn steer_batch_size(&self) -> usize {
        self.inner.steer_batch_size
    }

    /// Returns the cancellation token of the current run.
    pub(crate) fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.lock().clone()
    }

    /// Takes the bus receiver of the leader.
    ///
    /// One receiver serves one run: a second concurrent run gets `None` and simply observes no bus
    /// events, which is why one run per team is supported (documented in `src/runtime/CONTEXT.md`).
    pub(crate) fn take_events(&self) -> Option<UnboundedReceiver<EventMessage>> {
        self.inner.events.lock().take()
    }

    /// Returns the description of the team: `team_desc`, else the leader's default blurb.
    fn description(&self) -> String {
        self.inner
            .blueprint
            .team_desc
            .clone()
            .unwrap_or_else(|| self.inner.blueprint.leader.desc_or_default().to_owned())
    }
}
