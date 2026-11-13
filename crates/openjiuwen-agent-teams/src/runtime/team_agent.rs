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
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_agent::{AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use openjiuwen_multiagent::{TeamCard, TeamRuntime};
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::attach_host::{AttachHost, BackendAttachHandler};
use super::backend::TeamBackend;
use super::build::teammate_record;
use super::deps::TeamBuildDeps;
use super::initial_input::parse_initial_input;
use super::leader;
use super::observability_bind::TeamSignals;
use super::observer::MemberStreamObserver;
use super::org_bind::OrgBinding;
use super::routing::TeamModelRouter;
use super::spawner::TeamMemberSpawner;
use super::spec_meta::{AttachConfig, ForkDirective};
use crate::constants::USER_PSEUDO_MEMBER_NAME;
use crate::interaction::{
    DeliverResult, HumanAgentInbox, HumanAgentMessage, InteractGate, InteractPayload,
};
use crate::kv_cache::{MemberCacheKey, TeamKvCacheAffinity};
use crate::member::{
    MemberActivityRegistry, MemberRuntime, MemberSpawnOptions, MemberSpawnRequest, MemberSpawner,
    TeamMember,
};
use crate::monitor::{BackpressurePolicy, MonitorReceiver};
use crate::org::{OrgTaskOpResult, OrgTaskSpec};
use crate::schema::{
    EventMessage, EventPayload, MemberLifetime, MemberStatus, MemberStatusChangedEvent,
    TaskGraphSpec, TaskRecord, TeamAgentSpec, TeamMemberSpec, TeamRole, TeamSnapshot, TeamSpec,
    TeamTopic,
};
use crate::storage::TeamStore;
use crate::workspace::{IsolationMode, WorkspaceRequest};

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

    /// Starts the team: the transport, the leader runtime and the interaction gate.
    ///
    /// Idempotent: a second call while started is a no-op. A failed start resets the started flag
    /// and propagates the error, so a caller may retry.
    ///
    /// # Errors
    ///
    /// Propagates transport start failures and the leader runtime's own start error (missing
    /// model, bad checkpoint). Starting a team whose leader row is already terminal fails the
    /// member lifecycle (`member is not in unstarted state`).
    pub async fn start(&self) -> Result<()> {
        if self.inner.started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        if let Err(error) = self.run_start().await {
            self.inner.started.store(false, Ordering::SeqCst);
            return Err(error);
        }
        Ok(())
    }

    /// Performs the actual start work, in order, so [`Self::start`] can roll the flag back.
    async fn run_start(&self) -> Result<()> {
        self.inner.backend.messager().start().await?;
        self.inner.member.start(None).await?;
        self.start_attach().await?;
        self.inner.signals.team_started(Some(self.session_id()));
        debug!(
            team = self.team_name(),
            member = self.member_name(),
            "team agent started"
        );
        Ok(())
    }

    /// Binds the attach server when the spec (or `OPENJIUWEN_TEAM_SOCKET`) names a socket.
    ///
    /// Bind failures are logged and ignored unless `attach.required` is set, in which case the
    /// error aborts the start (see `src/runtime/CONTEXT.md` §"Attach policy").
    async fn start_attach(&self) -> Result<()> {
        let Some(config) = self.inner.attach.clone() else {
            return Ok(());
        };
        if self.inner.attach_host.lock().is_some() {
            return Ok(());
        }
        let handler: Arc<dyn crate::attach::AttachHandler> = Arc::new(BackendAttachHandler::new(
            Arc::clone(&self.inner.backend),
            self.leader_member_name(),
            self.cancellation_token(),
        ));
        match AttachHost::start(&config, handler).await {
            Ok(host) => {
                debug!(
                    team = self.team_name(),
                    socket = %host.path().display(),
                    "serving the team over the attach socket"
                );
                *self.inner.attach_host.lock() = Some(host);
            }
            Err(error) if config.required => return Err(error),
            Err(error) => {
                warn!(
                    team = self.team_name(),
                    socket = %config.socket.display(),
                    %error,
                    "attach socket could not be bound; continuing without it"
                );
            }
        }
        Ok(())
    }
    /// Stops the team: cancels the run, stops the leader runtime and closes the gate.
    ///
    /// Idempotent and never fails: teardown problems are logged so a caller's cleanup path cannot
    /// be blocked by them. The member's row settles on the `shutdown` retire target.
    ///
    /// # Errors
    ///
    /// This implementation returns `Ok(())` even when teardown partially fails.
    pub async fn stop(&self) -> Result<()> {
        self.inner.started.store(false, Ordering::SeqCst);
        self.inner.cancel.lock().cancel();
        let attach_host = self.inner.attach_host.lock().take();
        if let Some(host) = attach_host {
            host.shutdown().await;
        }
        if let Some(runtime) = self.inner.member.runtime() {
            if let Err(error) = runtime.stop().await {
                warn!(team = self.team_name(), error = %error, "failed to stop the leader runtime");
            }
        }
        if let Err(error) = self
            .inner
            .member
            .shutdown(MemberLifetime::Shutdown, true)
            .await
        {
            warn!(team = self.team_name(), error = %error, "failed to settle the leader row");
        }
        if let Err(error) = self.inner.backend.messager().stop().await {
            warn!(team = self.team_name(), error = %error, "failed to stop the team transport");
        }
        self.inner.gate.close();
        if let Err(error) = self.inner.spawner.despawn(self.member_name()).await {
            debug!(team = self.team_name(), error = %error, "leader has no child process to reap");
        }
        let released = self.inner.kv_cache.release_team(self.session_id()).await;
        debug!(
            team = self.team_name(),
            released, "kv-cache affinity entries released"
        );
        self.inner.signals.team_finished();
        debug!(team = self.team_name(), "team agent stopped");
        Ok(())
    }

    /// Runs the team for one caller request and returns the leader's answer.
    ///
    /// The request is parsed once ([`parse_initial_input`]): routable `$`/`@` channels become
    /// deliveries that never reach the leader, everything else becomes the leader query.
    ///
    /// # Errors
    ///
    /// Propagates start failures, leader runtime failures and store failures from the run loop.
    pub async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let input = parse_initial_input(&inputs, &self.roster());
        leader::run(
            self.clone(),
            input.leader_query,
            session,
            input.deliveries,
            None,
        )
        .await
    }

    /// Streams the same run as [`Self::invoke`], frame by frame.
    ///
    /// The run is spawned into an unbounded channel, so frames are yielded as they are produced;
    /// a run error becomes the last stream item. Dropping the stream does not cancel the run —
    /// call [`Self::stop`] for that.
    // The caller hands the request in as one owned value; the stream owns it from then on.
    #[allow(clippy::needless_pass_by_value)]
    pub fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> futures::stream::BoxStream<'static, Result<AgentChunk>> {
        let input = parse_initial_input(&inputs, &self.roster());
        let agent = self.clone();
        let (tx, mut rx) = mpsc::unbounded_channel::<Result<AgentChunk>>();
        let run_sink = tx.clone();
        // The task owns the run: the stream only forwards frames, so a slow consumer never stalls
        // the coordination loop.
        tokio::spawn(async move {
            if let Err(error) = leader::run(
                agent,
                input.leader_query,
                session,
                input.deliveries,
                Some(run_sink),
            )
            .await
            {
                let _ = tx.send(Err(error));
            }
        });
        async_stream::stream! {
            while let Some(item) = rx.recv().await {
                yield item;
            }
        }
        .boxed()
    }

    /// Routes one external payload when the team is active.
    ///
    /// Python parity: `TeamAgent.interact`. The gate decides: closed, not active, or admitted.
    // The async signature is the seam (a future gate may await admission); the gate itself is a
    // lock-guarded queue.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn interact(&self, payload: InteractPayload) -> DeliverResult {
        self.inner.gate.dispatch(payload, self.is_started())
    }

    /// Delivers one payload into the team without touching the leader runtime.
    ///
    /// Routing (pinned by `src/runtime/CONTEXT.md`):
    ///
    /// * [`InteractPayload::HumanAgent`] — the inbox (when HITL is enabled) plus one persisted
    ///   mailbox row; the delivered message id is returned.
    /// * [`InteractPayload::Operator`] / [`InteractPayload::GroupChat`] /
    ///   [`InteractPayload::GodView`] — the interaction gate, then one persisted row addressed to
    ///   the leader.
    /// * [`InteractPayload::HumanAgentToolCall`] and [`InteractPayload::ExternalTeam`] — rejected as
    ///   unsupported capabilities.
    pub async fn deliver(&self, payload: InteractPayload) -> DeliverResult {
        match payload {
            InteractPayload::HumanAgent(message) => self.deliver_human_agent(message).await,
            InteractPayload::Operator(_)
            | InteractPayload::GroupChat(_)
            | InteractPayload::GodView(_) => self.deliver_leader_bound(payload).await,
            InteractPayload::HumanAgentToolCall(_) => {
                DeliverResult::unsupported("human_agent_tool_call")
            }
            InteractPayload::ExternalTeam(_) => DeliverResult::unsupported("external_team"),
        }
    }

    /// Persists a human-agent message (and queues it on the inbox when HITL is on).
    async fn deliver_human_agent(&self, message: HumanAgentMessage) -> DeliverResult {
        if self.inner.inbox.is_enabled() {
            if let Err(error) = self.inner.inbox.push(message.clone()).await {
                warn!(team = self.team_name(), error = %error, "failed to queue a human-agent message");
            }
        } else {
            debug!(
                team = self.team_name(),
                human = %message.human_name,
                "human agent is not enabled; persisting the message only"
            );
        }
        let sender = message.sender.as_deref().unwrap_or(USER_PSEUDO_MEMBER_NAME);
        match self
            .inner
            .backend
            .send_message(sender, message.target.as_deref(), &message.body)
            .await
        {
            Ok(record) => DeliverResult::ok(Some(record.message_id)),
            Err(error) => DeliverResult::failed(error.to_string()),
        }
    }

    /// Admits a leader-bound payload through the gate and forwards it to the leader mailbox.
    async fn deliver_leader_bound(&self, payload: InteractPayload) -> DeliverResult {
        let admitted = self.inner.gate.dispatch(payload.clone(), self.is_started());
        if !admitted.ok {
            return admitted;
        }
        let (sender, body) = match &payload {
            InteractPayload::GodView(message) => (message.sender.clone(), message.body.clone()),
            InteractPayload::Operator(message) => (message.sender.clone(), message.body.clone()),
            InteractPayload::GroupChat(message) => (message.sender.clone(), message.body.clone()),
            _ => (None, String::new()),
        };
        let from = sender.as_deref().unwrap_or(USER_PSEUDO_MEMBER_NAME);
        match self
            .inner
            .backend
            .send_message(from, Some(self.leader_member_name()), &body)
            .await
        {
            Ok(_) => DeliverResult::ok(None),
            Err(error) => DeliverResult::failed(format!(
                "send_failed:{}: {error}",
                self.leader_member_name()
            )),
        }
    }

    /// Returns the roster projection with the team state and its task board.
    ///
    /// # Errors
    ///
    /// Propagates store failures while reading the board.
    pub async fn snapshot(&self) -> TeamSnapshot {
        let mut snapshot = self
            .inner
            .backend
            .snapshot_with_tasks()
            .await
            .unwrap_or_else(|error| {
                warn!(team = self.team_name(), error = %error, "failed to read the task board");
                self.inner.backend.snapshot()
            });
        snapshot.state = self.state();
        snapshot
    }

    /// Spawns one teammate: allocate its workspace, build its runtime, start it.
    ///
    /// The leader name is a no-op when the team is started. An already-running member is a no-op.
    /// A failed spawn leaves an `unstarted` row behind and returns the factory's error.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a name on no roster, [`Error::Unsupported`] for a role that cannot
    /// run in-process, [`Error::Validation`] when the team leader is not running, and the
    /// workspace/factory/runtime errors of the spawn itself.
    pub async fn spawn_member(&self, name: &str) -> Result<()> {
        if name == self.leader_member_name() {
            return if self.is_started() {
                Ok(())
            } else {
                Err(Error::validation(format!(
                    "team leader '{name}' is not running; start the team first"
                )))
            };
        }
        let Some(member_spec) = self.inner.spawner.member_spec(name) else {
            return Err(Error::not_found(format!(
                "member '{name}' not found in team '{}'",
                self.team_name()
            )));
        };
        if member_spec.role_type != TeamRole::Teammate {
            return Err(Error::unsupported(format!(
                "member '{name}' has role '{}'; only teammate members can be spawned in-process",
                member_spec.role_type.as_str()
            )));
        }
        if let Some(existing) = self.inner.backend.member(name) {
            if matches!(
                existing.status(),
                MemberStatus::Starting
                    | MemberStatus::Ready
                    | MemberStatus::Busy
                    | MemberStatus::Paused
            ) {
                debug!(
                    team = self.team_name(),
                    member = name,
                    "member is already running"
                );
                return Ok(());
            }
        }

        let spawner = Arc::clone(&self.inner.spawner);
        let workspace = spawner
            .deps()
            .workspace
            .allocate(&WorkspaceRequest {
                team_name: self.team_name().to_owned(),
                session_id: self.session_id().to_owned(),
                member_name: name.to_owned(),
                spec: self.inner.blueprint.workspace.clone(),
                member_spec: spawner
                    .deep_agent_spec(name)
                    .and_then(|spec| spec.workspace.clone()),
                isolation: self.inner.isolation,
            })
            .await?;

        let request = MemberSpawnRequest::new(
            self.team_name(),
            name,
            self.session_id(),
            workspace.root.clone(),
        )
        .with_options(self.inner.spawn_options.clone());
        let runtime = match spawner.spawn(request).await {
            Ok(runtime) => runtime,
            Err(error) => {
                self.ensure_row(name, &member_spec).await;
                return Err(error);
            }
        };

        let member = if let Some(member) = self.inner.backend.member(name) {
            member
        } else {
            let record = teammate_record(&spawner, name, &self.inner.routing);
            self.inner.store.insert_member(record.clone()).await?;
            let created = Arc::new(TeamMember::new(self.inner.store.clone(), record));
            self.inner.backend.add_member(Arc::clone(&created));
            created
        };
        if member.runtime().is_none() {
            member.observe(Arc::new(MemberStreamObserver::new(
                Arc::clone(&self.inner.activity),
                Arc::downgrade(&member),
                member.role(),
                name.to_owned(),
                self.inner.signals.clone(),
                Arc::clone(&self.inner.chunk_sinks),
            )));
        }
        member.attach_runtime(runtime);
        let previous = member.status();
        member.start(None).await?;
        // Register the KV identity only after the start succeeded: a failed start would otherwise
        // leave a live affinity entry for a member that never ran.
        self.register_kv_affinity(name);
        self.inner.signals.member_spawned(name, member.role());
        self.publish_member_status(name, previous, member.status())
            .await;
        Ok(())
    }

    /// Ensures an `unstarted` row exists for `name`, logging (never failing) a store problem.
    async fn ensure_row(&self, name: &str, member_spec: &TeamMemberSpec) {
        if self.inner.backend.member(name).is_some() {
            return;
        }
        let record = teammate_record(&self.inner.spawner, name, &self.inner.routing);
        debug!(
            team = self.team_name(),
            member = name,
            role = member_spec.role_type.as_str(),
            "materializing an unstarted roster row after a failed spawn"
        );
        if let Err(error) = self.inner.store.insert_member(record.clone()).await {
            warn!(team = self.team_name(), member = name, error = %error, "failed to persist the roster row");
            return;
        }
        self.inner
            .backend
            .add_member(Arc::new(TeamMember::new(self.inner.store.clone(), record)));
    }

    /// Publishes a member status change, logging (never failing) a transport problem.
    async fn publish_member_status(
        &self,
        name: &str,
        previous: MemberStatus,
        status: MemberStatus,
    ) {
        self.inner
            .signals
            .member_status(name, self.role_of(name), status);
        if previous == status {
            return;
        }
        let payload = EventPayload::MemberStatusChanged(MemberStatusChangedEvent {
            team_name: self.team_name().to_owned(),
            member_name: name.to_owned(),
            status,
            previous: Some(previous),
        });
        if let Err(error) = self
            .inner
            .backend
            .post_event(TeamTopic::Team, payload)
            .await
        {
            warn!(team = self.team_name(), member = name, error = %error, "failed to publish a member status change");
        }
    }

    /// Retires one member and returns the status its row settles on.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown member, the member runtime's teardown error (logged, not
    /// propagated) and the lifecycle machine's own validation errors.
    pub async fn shutdown_member(
        &self,
        name: &str,
        lifetime: MemberLifetime,
        force: bool,
    ) -> Result<MemberStatus> {
        let Some(member) = self.inner.backend.member(name) else {
            return Err(Error::not_found(format!(
                "member '{name}' not found in team '{}'",
                self.team_name()
            )));
        };
        let previous = member.status();
        if let Some(runtime) = member.runtime() {
            if let Err(error) = runtime.stop().await {
                warn!(team = self.team_name(), member = name, error = %error, "failed to stop a member runtime");
            }
        }
        let status = member.shutdown(lifetime, force).await?;
        if let Err(error) = self.inner.spawner.despawn(name).await {
            debug!(team = self.team_name(), member = name, %error, "failed to reap a member child process");
        }
        let _ = self.inner.kv_cache.release(name).await;
        self.inner
            .signals
            .member_stopped(name, member.role(), "shutdown");
        self.publish_member_status(name, previous, status).await;
        Ok(status)
    }

    /// Spawns every `unstarted` non-leader member, returning the first error after the loop.
    ///
    /// # Errors
    ///
    /// Returns the first spawn error; every member the loop reached before it was still spawned.
    pub async fn auto_start_all(&self) -> Result<()> {
        let mut first_error = None;
        for member in self.inner.backend.members() {
            if member.role() == TeamRole::Leader || member.status() != MemberStatus::Unstarted {
                continue;
            }
            if let Err(error) = self.spawn_member(member.member_name()).await {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Creates one top-level task on the team's board and publishes its events.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] on a non-leader handle (`only the leader can create tasks`), the
    /// board's own validation (duplicate id, cycle, unknown dependency) and the store's errors.
    pub async fn create_task(&self, spec: TaskGraphSpec) -> Result<TaskRecord> {
        ensure_leader_role(self.role())?;
        let mutation = self.inner.backend.tasks().create_task(spec).await?;
        self.inner.backend.publish_mutation(&mutation).await?;
        Ok(mutation.task)
    }

    /// Sends `content` from this member, optionally addressed to `to`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an over-long body or an empty target list, and the store's or the
    /// transport's errors.
    pub async fn send_message(
        &self,
        to: Option<&str>,
        content: &str,
    ) -> Result<crate::storage::MessageRecord> {
        self.inner
            .backend
            .send_message(self.member_name(), to, content)
            .await
    }

    /// Returns the live roster handles, sorted by member name.
    ///
    /// # Errors
    ///
    /// This implementation never fails; the signature is async so a future store-backed roster
    /// does not change the seam.
    // Async for seam stability: a future store-backed roster reads asynchronously.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn members(&self) -> Vec<Arc<TeamMember>> {
        self.inner.backend.members()
    }

    /// Subscribes to this handle's runtime chunks.
    ///
    /// The fan-out is the run loop's input: every registered sender receives each chunk the
    /// member's harness observer reports, and a dropped receiver is pruned on the next chunk.
    #[must_use]
    pub fn subscribe_chunks(&self) -> UnboundedReceiver<Value> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.chunk_sinks.lock().push(tx);
        rx
    }

    /// Returns the team lifecycle signals (observability hooks + monitor sink).
    #[must_use]
    pub fn signals(&self) -> TeamSignals {
        self.inner.signals.clone()
    }
    /// Attaches a monitor stream to this team and returns its receiving end.
    ///
    /// Events are emitted best-effort from the lifecycle calls (start/stop, member spawn/stop,
    /// rounds, task transitions); installing a second stream replaces the first emitter.
    #[must_use]
    pub fn monitor_stream(
        &self,
        capacity: usize,
        policy: BackpressurePolicy,
    ) -> Option<MonitorReceiver> {
        self.inner.backend.monitor_stream(capacity, policy)
    }
    /// Returns the KV-cache affinity registry of the run.
    #[must_use]
    pub fn kv_cache(&self) -> Arc<TeamKvCacheAffinity> {
        Arc::clone(&self.inner.kv_cache)
    }
    /// Returns the KV-cache identity env a member session should inherit, if registered.
    #[must_use]
    pub fn kv_env(&self, member_name: &str) -> Option<std::collections::BTreeMap<String, String>> {
        self.inner.kv_cache.env(member_name)
    }
    /// Returns the organisation binding, when the spec configures one.
    #[must_use]
    pub fn org(&self) -> Option<Arc<OrgBinding>> {
        self.inner.backend.org()
    }
    /// Returns the bind path of the live attach server, if one is serving.
    #[must_use]
    pub fn attach_socket(&self) -> Option<std::path::PathBuf> {
        self.inner
            .attach_host
            .lock()
            .as_ref()
            .map(|host| host.path().to_path_buf())
    }
    /// Creates a cross-team organisation task owned by this team.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when no organisation is configured and the pool's own errors otherwise.
    pub async fn create_org_task(&self, spec: OrgTaskSpec) -> Result<crate::org::OrgTaskMutation> {
        let org = self
            .org()
            .ok_or_else(|| Error::config("team has no organisation configured"))?;
        let mutation = org.create_task(spec).await?;
        self.inner.signals.delegation(
            self.leader_member_name(),
            "",
            &mutation.task.task_id,
            "created",
        );
        Ok(mutation)
    }
    /// Delegates one org task after consulting the delegation router.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] without an organisation, [`Error::Validation`] when the router refuses and
    /// the pool's own errors otherwise.
    pub async fn delegate_org_task(
        &self,
        task_id: &str,
        to_team: &str,
        to_member: Option<&str>,
    ) -> Result<OrgTaskOpResult> {
        let org = self
            .org()
            .ok_or_else(|| Error::config("team has no organisation configured"))?;
        let result = org
            .delegate(
                task_id,
                self.team_name(),
                self.leader_member_name(),
                to_team,
                to_member,
            )
            .await?;
        self.inner.signals.delegation(
            self.leader_member_name(),
            to_member.unwrap_or(to_team),
            task_id,
            "allowed",
        );
        Ok(result)
    }
    /// Registers one member's KV-cache affinity, deriving a child identity for a fork.
    fn register_kv_affinity(&self, member_name: &str) {
        let key = MemberCacheKey::new(self.team_name(), self.session_id(), member_name.to_owned());
        let model = self
            .inner
            .routing
            .model_ref_for(member_name, None, self.role_of(member_name))
            .unwrap_or_default();
        let forked = self.inner.spawner.forked_member(member_name);
        let identity = match forked {
            Some(child) => self
                .inner
                .kv_cache
                .register_child(&child.parent_member, key, &model),
            None => Ok(self.inner.kv_cache.register_member(key, &model)),
        };
        match identity {
            Ok(identity) => debug!(
                team = self.team_name(),
                member = member_name,
                cache = %identity.session_id,
                "registered the member's KV-cache affinity"
            ),
            Err(error) => debug!(
                team = self.team_name(),
                member = member_name,
                %error,
                "KV-cache affinity not registered (disabled or unknown parent)"
            ),
        }
    }
    /// Returns a roster member's role, defaulting to teammate for an unknown name.
    fn role_of(&self, member_name: &str) -> TeamRole {
        self.inner
            .backend
            .member(member_name)
            .map_or(TeamRole::Teammate, |member| member.role())
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
pub(crate) mod test_support {
    //! Offline fakes shared by the `team_agent` and `leader` unit tests.
    //!
    //! The scripted harness answers every `send` with one answer chunk and immediately returns to
    //! `Idle` unless it is configured to hang (state stays `Running`, no output), which is what the
    //! settle and cancellation tests need. No model, socket or clock is involved.
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_agent::AgentCard;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_harness_protocol::{
        AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessContext, HarnessEvent,
        HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, HarnessState,
        OutputChannel, OutputEvent, OutputKind, OutputOperation, SendReceipt, VecEventCursor,
    };
    use parking_lot::Mutex;
    use serde_json::json;
    use tokio::sync::mpsc;

    use crate::harness::{MemberHarnessContext, MemberHarnessFactory};
    use crate::runtime::deps::TeamBuildDeps;
    use crate::schema::{DeepAgentSpec, TeamAgentSpec};
    use crate::workspace::{MemberWorkspace, WorkspaceProvider, WorkspaceRequest};

    use super::SpecBuild;

    /// One command a [`ScriptedHarness`] received.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum Call {
        /// The cycle was started.
        Start,
        /// The cycle was stopped.
        Stop,
        /// The cycle was paused.
        Pause,
        /// A warm round was resumed.
        Resume,
        /// An input was delivered; the payload is its text.
        Send(String),
    }

    /// Channel-backed cursor that yields the scripted observations in order.
    struct ScriptCursor {
        /// Event source; closed when the harness drops its sender.
        receiver: mpsc::UnboundedReceiver<HarnessEvent>,
    }

    #[async_trait]
    impl HarnessEventCursor for ScriptCursor {
        async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
            Ok(self.receiver.recv().await)
        }

        async fn close(&mut self) -> HarnessResult<()> {
            Ok(())
        }
    }

    /// Harness protocol handle with scripted answers: records commands, scripts one answer chunk.
    pub(crate) struct ScriptedHarness {
        /// State reported by [`HarnessProtocol::state`].
        state: Mutex<HarnessState>,
        /// Commands received, in order.
        calls: Arc<Mutex<Vec<Call>>>,
        /// Static capability card.
        card: HarnessCard,
        /// Observation buffer policy.
        buffer: EventBufferConfig,
        /// The single observation receiver handed to the first `events()` caller.
        events: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
        /// Sender that scripts the observation stream.
        sender: mpsc::UnboundedSender<HarnessEvent>,
        /// Answer text emitted by every `send`.
        answer: String,
        /// When set, `send` leaves the harness `Running` and emits nothing.
        hang: bool,
        /// Envelope sequence counter.
        sequence: AtomicU64,
    }

    impl ScriptedHarness {
        /// Builds a harness that answers every input with `answer`.
        fn new(answer: String, hang: bool) -> Arc<Self> {
            let (sender, receiver) = mpsc::unbounded_channel();
            Arc::new(Self {
                state: Mutex::new(HarnessState::Idle),
                calls: Arc::new(Mutex::new(Vec::new())),
                card: HarnessCard::new("scripted", "0.0.0"),
                buffer: EventBufferConfig::default(),
                events: Mutex::new(Some(receiver)),
                sender,
                answer,
                hang,
                sequence: AtomicU64::new(0),
            })
        }

        /// Returns the recorded commands.
        pub(crate) fn calls(&self) -> Vec<Call> {
            self.calls.lock().clone()
        }

        /// Returns the text of every delivered input, in order.
        pub(crate) fn sends(&self) -> Vec<String> {
            self.calls
                .lock()
                .iter()
                .filter_map(|call| match call {
                    Call::Send(text) => Some(text.clone()),
                    _ => None,
                })
                .collect()
        }

        /// Builds one answer observation.
        fn output_event(&self) -> HarnessEvent {
            let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
            HarnessEvent::new(
                sequence + 1,
                1.0,
                OutputEvent {
                    output_id: format!("out-{sequence}"),
                    kind: OutputKind::Text,
                    content: json!(self.answer),
                    operation: OutputOperation::Delta,
                    channel: OutputChannel::Answer,
                    content_index: 0,
                    content_type: None,
                    data: BTreeMap::new(),
                },
                "session:scripted",
                "scripted",
            )
            .with_turn_id("turn-1")
        }
    }

    #[async_trait]
    #[allow(clippy::needless_pass_by_value)]
    impl HarnessProtocol for ScriptedHarness {
        fn card(&self) -> &HarnessCard {
            &self.card
        }

        fn state(&self) -> HarnessState {
            *self.state.lock()
        }

        fn provider_session_id(&self) -> Option<&str> {
            None
        }

        fn event_buffer_config(&self) -> &EventBufferConfig {
            &self.buffer
        }

        async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
            self.calls.lock().push(Call::Start);
            *self.state.lock() = HarnessState::Idle;
            Ok(())
        }

        async fn stop(&self) -> HarnessResult<()> {
            self.calls.lock().push(Call::Stop);
            *self.state.lock() = HarnessState::Terminated;
            Ok(())
        }

        fn events(&self) -> Box<dyn HarnessEventCursor> {
            match self.events.lock().take() {
                Some(receiver) => Box::new(ScriptCursor { receiver }),
                None => Box::new(VecEventCursor::new(Vec::new())),
            }
        }

        fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
            Box::new(VecEventCursor::new(Vec::new()))
        }

        async fn send(
            &self,
            input: HarnessInput,
            mode: DeliveryMode,
        ) -> HarnessResult<SendReceipt> {
            let text = input.content.as_str().unwrap_or_default().to_owned();
            self.calls.lock().push(Call::Send(text));
            *self.state.lock() = HarnessState::Running;
            if !self.hang {
                let _ = self.sender.send(self.output_event());
                *self.state.lock() = HarnessState::Idle;
            }
            Ok(SendReceipt {
                message_id: "message-1".to_owned(),
                turn_id: "turn-1".to_owned(),
                accepted_mode: mode,
            })
        }

        async fn abort(&self, _mode: AbortMode) -> HarnessResult<()> {
            Ok(())
        }

        async fn pause(&self) -> HarnessResult<()> {
            self.calls.lock().push(Call::Pause);
            *self.state.lock() = HarnessState::Paused;
            Ok(())
        }

        async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
            self.calls.lock().push(Call::Resume);
            *self.state.lock() = HarnessState::Running;
            Ok(())
        }

        async fn export_checkpoint(
            &self,
        ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
            Ok(None)
        }
    }

    /// Harness factory that builds [`ScriptedHarness`]es, records contexts and can fail or hang.
    #[derive(Default)]
    pub(crate) struct ScriptedFactory {
        /// Contexts received by [`MemberHarnessFactory::create`], in call order.
        calls: Mutex<Vec<MemberHarnessContext>>,
        /// Members whose build must fail with `Error::config`.
        failing: Mutex<BTreeSet<String>>,
        /// Members whose harness never settles.
        hanging: Mutex<BTreeSet<String>>,
        /// Harness built per member, so a test can inspect what the leader received.
        harnesses: Mutex<BTreeMap<String, Arc<ScriptedHarness>>>,
    }

    impl ScriptedFactory {
        /// Builds an absent-minded factory: every member gets its scripted answer.
        pub(crate) fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        /// Makes `create` fail for `member`.
        pub(crate) fn failing_for(&self, member: &str) {
            self.failing.lock().insert(member.to_owned());
        }

        /// Makes the harness of `member` hang on every input.
        pub(crate) fn hanging_for(&self, member: &str) {
            self.hanging.lock().insert(member.to_owned());
        }

        /// Returns the recorded factory contexts, in call order.
        pub(crate) fn calls(&self) -> Vec<MemberHarnessContext> {
            self.calls.lock().clone()
        }

        /// Returns the harness built for `member`, when it was built.
        pub(crate) fn harness(&self, member: &str) -> Option<Arc<ScriptedHarness>> {
            self.harnesses.lock().get(member).cloned()
        }
    }

    #[async_trait]
    impl MemberHarnessFactory for ScriptedFactory {
        async fn create(
            &self,
            _spec: &DeepAgentSpec,
            ctx: &MemberHarnessContext,
        ) -> Result<Arc<dyn HarnessProtocol>> {
            self.calls.lock().push(ctx.clone());
            if self.failing.lock().contains(&ctx.member_name) {
                return Err(Error::config(format!(
                    "scripted factory failure for '{}'",
                    ctx.member_name
                )));
            }
            let harness = ScriptedHarness::new(
                format!("done: {}", ctx.member_name),
                self.hanging.lock().contains(&ctx.member_name),
            );
            self.harnesses
                .lock()
                .insert(ctx.member_name.clone(), Arc::clone(&harness));
            Ok(harness)
        }

        fn name(&self) -> &'static str {
            "scripted"
        }
    }

    /// Workspace provider that hands out one distinct directory per member and records requests.
    #[derive(Default)]
    pub(crate) struct RecordingWorkspaceProvider {
        /// Allocation requests, in call order.
        requests: Mutex<Vec<WorkspaceRequest>>,
    }

    impl RecordingWorkspaceProvider {
        /// Builds an empty recorder.
        pub(crate) fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        /// Returns the recorded requests.
        pub(crate) fn requests(&self) -> Vec<WorkspaceRequest> {
            self.requests.lock().clone()
        }
    }

    #[async_trait]
    impl WorkspaceProvider for RecordingWorkspaceProvider {
        async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
            self.requests.lock().push(request.clone());
            let root = std::env::temp_dir()
                .join("openjiuwen-agent-teams-tests")
                .join(&request.team_name)
                .join(&request.session_id)
                .join(&request.member_name);
            Ok(MemberWorkspace::new(
                request.team_name.clone(),
                request.session_id.clone(),
                request.member_name.clone(),
                root,
            ))
        }

        async fn release(&self, _workspace: &MemberWorkspace) -> Result<()> {
            Ok(())
        }
    }

    /// Builds one member blueprint with a card and a prompt.
    pub(crate) fn deep_spec(name: &str) -> DeepAgentSpec {
        DeepAgentSpec {
            card: Some(AgentCard::new(name)),
            system_prompt: Some(format!("prompt for {name}")),
            ..DeepAgentSpec::default()
        }
    }

    /// Builds a blueprint for `team` led by `leader` with one agent entry per teammate.
    pub(crate) fn blueprint(team: &str, leader: &str, teammates: &[&str]) -> TeamAgentSpec {
        let mut spec = TeamAgentSpec {
            team_name: team.to_owned(),
            ..TeamAgentSpec::default()
        };
        spec.leader.member_name = leader.to_owned();
        spec.agents.insert("leader".to_owned(), deep_spec("Leader"));
        for name in teammates {
            spec.agents.insert((*name).to_owned(), deep_spec(name));
        }
        spec
    }

    /// Builds the injected dependencies over the two fakes.
    pub(crate) fn deps(
        factory: &Arc<ScriptedFactory>,
        workspace: &Arc<RecordingWorkspaceProvider>,
    ) -> TeamBuildDeps {
        TeamBuildDeps::default()
            .with_harness_factory(Arc::clone(factory) as Arc<dyn MemberHarnessFactory>)
            .with_workspace(Arc::clone(workspace) as Arc<dyn WorkspaceProvider>)
    }

    /// Builds a leader agent for `blueprint` with fresh fakes and returns all three.
    pub(crate) async fn built(
        blueprint: &TeamAgentSpec,
    ) -> (
        Arc<super::TeamAgent>,
        Arc<ScriptedFactory>,
        Arc<RecordingWorkspaceProvider>,
    ) {
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        let agent = blueprint
            .build_with(deps(&factory, &workspace))
            .await
            .unwrap();
        (agent, factory, workspace)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::test_support::{blueprint, built, deps, ScriptedFactory};
    use super::*;
    use crate::interaction::{ExternalTeamEvent, HumanAgentMessage, OperatorMessage};
    use openjiuwen_common::error::ErrorKind;

    /// Builds a two-teammate blueprint for the shared tests.
    fn two_teammates() -> TeamAgentSpec {
        blueprint("alpha", "boss", &["dev-1", "dev-2"])
    }

    #[tokio::test]
    async fn build_produces_the_leader_handle_and_the_roster() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;

        assert_eq!(agent.card().name, "alpha_boss");
        assert_eq!(agent.role(), TeamRole::Leader);
        assert_eq!(agent.member_name(), "boss");
        assert_eq!(agent.leader_member_name(), "boss");
        assert_eq!(agent.team_name(), "alpha");
        assert_eq!(agent.spec().team_name, "alpha");
        assert!(agent.session_id().starts_with("session_"));
        assert_eq!(agent.roster(), vec!["boss", "dev-1", "dev-2"]);
        assert!(!agent.is_started());

        let leader = agent.backend().member("boss").unwrap();
        assert_eq!(leader.role(), TeamRole::Leader);
        assert_eq!(leader.status(), MemberStatus::Unstarted);
        assert_eq!(leader.record().prompt, "prompt for Leader");

        for name in ["dev-1", "dev-2"] {
            let member = agent.backend().member(name).unwrap();
            assert_eq!(member.role(), TeamRole::Teammate);
            assert_eq!(member.status(), MemberStatus::Unstarted);
            assert_eq!(member.record().prompt, format!("prompt for {name}"));
            assert!(
                member.runtime().is_none(),
                "teammates start without a brain"
            );
        }

        let card = agent.team_card();
        assert_eq!(card.id, "alpha");
        assert_eq!(card.name, "alpha");
        assert_eq!(card.agent_cards.len(), 3);
        assert_eq!(card.agent_cards[0].name, "alpha_boss");
        assert_eq!(agent.runtime().session_id(), agent.session_id());
    }

    #[tokio::test]
    async fn build_requires_the_leader_blueprint() {
        let spec = TeamAgentSpec {
            team_name: "alpha".to_owned(),
            ..TeamAgentSpec::default()
        };
        let error = spec
            .build_with(deps(
                &ScriptedFactory::new(),
                &super::test_support::RecordingWorkspaceProvider::new(),
            ))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("agents dict must contain a 'leader' key"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn language_comes_from_the_deps_else_the_blueprint() {
        let factory = ScriptedFactory::new();
        let workspace = super::test_support::RecordingWorkspaceProvider::new();
        let agent = blueprint("alpha", "boss", &[])
            .build_with(deps(&factory, &workspace).with_language("en"))
            .await
            .unwrap();
        assert_eq!(agent.language(), "en");
        assert_eq!(agent.spec().language.as_deref(), Some("en"));

        let mut declared = blueprint("beta", "boss", &[]);
        declared.language = Some("cf".to_owned());
        let agent = declared
            .build_with(deps(&factory, &workspace))
            .await
            .unwrap();
        assert_eq!(agent.language(), "cf");

        let agent = blueprint("gamma", "boss", &[])
            .build_with(deps(&factory, &workspace).with_language("bogus"))
            .await
            .unwrap();
        assert_eq!(agent.language(), "cn", "an unknown language falls back");
    }

    #[tokio::test]
    async fn start_is_idempotent_and_stop_settles_the_leader_row() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;
        agent.start().await.unwrap();
        agent.start().await.unwrap();
        assert!(agent.is_started());
        assert_eq!(agent.state(), HarnessState::Idle);
        assert_eq!(
            agent.backend().member("boss").unwrap().status(),
            MemberStatus::Ready
        );

        agent.stop().await.unwrap();
        agent.stop().await.unwrap();
        assert!(!agent.is_started());
        assert_eq!(
            agent.backend().member("boss").unwrap().status(),
            MemberStatus::ShutDown
        );
        assert!(agent.gate().is_closed());
    }

    #[tokio::test]
    async fn spawning_an_unknown_member_is_not_found() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;
        let error = agent.spawn_member("ghost").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error
            .to_string()
            .contains("member 'ghost' not found in team 'alpha'"));
    }

    #[tokio::test]
    async fn spawning_a_teammate_builds_and_starts_its_runtime() {
        let (agent, factory, _workspace) = built(&two_teammates()).await;
        agent.spawn_member("dev-1").await.unwrap();

        let member = agent.backend().member("dev-1").unwrap();
        assert_eq!(member.status(), MemberStatus::Ready);
        assert!(member.runtime().is_some());
        let calls: Vec<String> = factory
            .harness("dev-1")
            .unwrap()
            .calls()
            .iter()
            .map(|call| format!("{call:?}"))
            .collect();
        assert!(calls.contains(&"Start".to_owned()), "{calls:?}");

        // A second spawn of a running member is a no-op.
        agent.spawn_member("dev-1").await.unwrap();
        assert_eq!(
            agent.backend().member("dev-1").unwrap().status(),
            MemberStatus::Ready
        );
    }

    #[tokio::test]
    async fn a_failed_spawn_keeps_the_row_unstarted() {
        let (agent, factory, _workspace) = built(&two_teammates()).await;
        factory.failing_for("dev-1");
        let error = agent.spawn_member("dev-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            agent.backend().member("dev-1").unwrap().status(),
            MemberStatus::Unstarted
        );
        assert!(agent.backend().member("dev-1").unwrap().runtime().is_none());
    }

    #[tokio::test]
    async fn shutdown_member_settles_on_the_lifetime_target() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;
        agent.spawn_member("dev-1").await.unwrap();
        let status = agent
            .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
            .await
            .unwrap();
        assert_eq!(status, MemberStatus::ShutDown);

        agent.spawn_member("dev-2").await.unwrap();
        let status = agent
            .shutdown_member("dev-2", MemberLifetime::Dormant, false)
            .await
            .unwrap();
        assert_eq!(status, MemberStatus::Dormant);

        let error = agent
            .shutdown_member("ghost", MemberLifetime::Shutdown, true)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn auto_start_all_starts_every_unstarted_teammate() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;
        agent.auto_start_all().await.unwrap();
        for name in ["dev-1", "dev-2"] {
            assert_eq!(
                agent.backend().member(name).unwrap().status(),
                MemberStatus::Ready,
                "{name}"
            );
        }
        assert_eq!(
            agent.backend().member("boss").unwrap().status(),
            MemberStatus::Unstarted,
            "auto_start_all never starts the leader"
        );
    }

    #[tokio::test]
    async fn every_member_gets_its_own_workspace_root() {
        let (agent, factory, workspace) = built(&two_teammates()).await;
        agent.spawn_member("dev-1").await.unwrap();

        let contexts = factory.calls();
        assert_eq!(
            contexts.len(),
            2,
            "one harness for the leader, one for dev-1"
        );
        let leader_root = contexts[0].workspace_root.clone();
        let teammate_root = contexts[1].workspace_root.clone();
        assert_ne!(leader_root, teammate_root);
        assert!(leader_root.ends_with("boss"), "{leader_root:?}");
        assert!(teammate_root.ends_with("dev-1"), "{teammate_root:?}");

        let requests = workspace.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].session_id, requests[1].session_id,
            "both members run under the team session"
        );
    }

    #[tokio::test]
    async fn create_task_is_leader_only_and_publishes_its_events() {
        let error = ensure_leader_role(TeamRole::Teammate).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("only the leader can create tasks"),
            "{error}"
        );
        assert!(ensure_leader_role(TeamRole::Leader).is_ok());

        let (agent, _factory, _workspace) = built(&two_teammates()).await;
        let record = agent
            .create_task(TaskGraphSpec {
                title: "ship it".to_owned(),
                content: "do the thing".to_owned(),
                ..TaskGraphSpec::default()
            })
            .await
            .unwrap();
        assert_eq!(record.title, "ship it");
        assert!(!agent.backend().tasks().is_settled().await.unwrap());
    }

    #[tokio::test]
    async fn deliver_routes_every_channel() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;

        let human = agent
            .deliver(InteractPayload::HumanAgent(HumanAgentMessage {
                human_name: "alice".to_owned(),
                target: None,
                body: "the build is green".to_owned(),
                sender: Some("alice".to_owned()),
            }))
            .await;
        assert!(human.ok, "{human:?}");
        assert!(human.message_id.is_some(), "the delivered row is returned");
        assert!(!agent.inbox().is_enabled(), "HITL is off by default");

        let operator = agent
            .deliver(InteractPayload::Operator(OperatorMessage {
                target: None,
                body: "status?".to_owned(),
                sender: None,
            }))
            .await;
        assert_eq!(operator.reason.as_deref(), Some("not_active"));

        let external = agent
            .deliver(InteractPayload::ExternalTeam(ExternalTeamEvent {
                event: Value::Null,
            }))
            .await;
        assert_eq!(
            external.reason.as_deref(),
            Some("unsupported:external_team")
        );
    }

    #[tokio::test]
    async fn interact_reports_the_gate_state() {
        let (agent, _factory, _workspace) = built(&two_teammates()).await;
        let payload = InteractPayload::GodView(crate::interaction::GodViewMessage {
            body: "hello".to_owned(),
            sender: None,
        });
        assert_eq!(
            agent.interact(payload.clone()).await.reason.as_deref(),
            Some("not_active")
        );
        agent.start().await.unwrap();
        assert!(agent.interact(payload.clone()).await.ok);
        agent.stop().await.unwrap();
        assert_eq!(
            agent.interact(payload).await.reason.as_deref(),
            Some("gate_closed")
        );
    }
}
