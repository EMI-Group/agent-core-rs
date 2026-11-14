//! The harness-backed [`MemberSpawner`](crate::member::MemberSpawner): one member's cold spec in,
//! a live member runtime out.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/inprocess_spawn.py`, which builds a
//! teammate `TeamAgent` from the blueprint before wrapping it in an `asyncio.Task`. The Rust port
//! splits that in two: this spawner materializes the member's *brain* through the injected
//! [`MemberHarnessFactory`](crate::harness::MemberHarnessFactory) and returns it as a
//! [`MemberRuntime`](crate::member::MemberRuntime); owning the driving task is the caller's job.
//!
//! # Spawn modes
//!
//! The requested [`MemberSpawnOptions::kind`](crate::member::MemberSpawnOptions) decides the
//! backend:
//!
//! * `in_process` (default) — the harness factory builds the brain here, exactly as before;
//! * `process` / `external_cli` / `ssh` — the request is forwarded to the matching member spawner
//!   (`ProcessMemberSpawner`, `ExternalCliMemberSpawner`, `SshMemberSpawner`), which owns the child
//!   process and reaps it on [`MemberSpawner::despawn`](crate::member::MemberSpawner::despawn).
//!
//! The mode comes from the blueprint: `metadata["spawn"]` (see [`super::spec_meta`](crate::runtime::spec_meta)) with
//! `spawn_mode: process` as the fallback. Child spawners live for the spawner's lifetime, so a
//! re-spawn of the same member reuses and replaces its child.
//!
//! # Model identity and forks
//!
//! Before the factory is called, the member's [`DeepAgentSpec`](crate::schema::DeepAgentSpec) is run through the team's
//! [`TeamModelRouter`]: a pool entry rewrites `spec.model` to that endpoint, so the pool selects a
//! real model instead of only being recorded on the roster row. A fork directive
//! (`metadata["spawn"]["fork"]` / `metadata["fork"]`) replaces the blueprint with
//! [`crate::org::fork_member_context`]'s child spec (parent tools filtered, identity cleared).
//!
//! # Member identity and checkpoints
//!
//! A spawned member runs under its *derived* session id,
//! `MemberHarnessContext::member_session_id(team_session, member_name)`, so two members of one team
//! never share a harness session (Python keys member state the same way). The request's session id
//! stays the team session and reaches the harness as `host_session_id`.
//! The runtime is wired through `super::checkpoints::wire_runtime`: the card id of the effective
//! spec (else the member name), the `metadata["checkpoint"]` policy, the team-shared store the build
//! path resolved (or a per-member fallback) and the workspace root as cwd. A `{"enabled": false}`
//! policy attaches no store at all, so the member does no checkpoint work.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use tracing::{debug, warn};

use super::checkpoints::{member_card_id, wire_runtime};
use super::deps::TeamBuildDeps;
use super::routing::TeamModelRouter;
use super::spec_meta::{checkpoint_config, fork_directive, spawn_options, ForkDirective};
use crate::harness::{HarnessMemberRuntime, MemberHarnessContext};
use crate::member::{
    ExternalCliMemberSpawner, ExternalCliRuntimeConfig, MemberRuntime, MemberSpawnKind,
    MemberSpawnOptions, MemberSpawnRequest, MemberSpawner, ProcessMemberConfig,
    ProcessMemberSpawner, SshMemberSpawner, SshTransportSpec,
};
use crate::memory::TeamMemoryWiring;
use crate::org::{fork_member_context, ForkSource, ForkedMemberSpec};
use crate::schema::{DeepAgentSpec, TeamAgentSpec, TeamMemberSpec, TeamRole};

/// Key of the leader's blueprint in [`TeamAgentSpec::agents`].
///
/// Mirrors the private constant of `crate::schema::spec`; the leader is never a spawnable roster
/// name, so the roster projection here has to know the key without widening the schema surface.
const LEADER_AGENT_KEY: &str = "leader";

/// Spawns team members from the blueprint through the configured spawn backend.
///
/// One spawner serves one built team: it carries the roster (`spec`), the injection point for the
/// member's brain (`deps.harness_factory`), the model router, the child-process spawners and the
/// team session id every member's own session id is derived from.
pub struct TeamMemberSpawner {
    /// The blueprint the roster, the teammate mode and the team name come from.
    spec: TeamAgentSpec,
    /// The injected build dependencies; only `harness_factory`, `language`, `routing`, `kv_cache`
    /// and `checkpoints` are consulted.
    deps: TeamBuildDeps,
    /// Team session id every member derives its own session id from.
    session_id: String,
    /// Spawn options the blueprint asks for (in-process by default).
    options: MemberSpawnOptions,
    /// Fork directive, when this team forks from another member.
    fork: Option<ForkDirective>,
    /// The model router (shared with the build path when deps carries one).
    routing: Arc<TeamModelRouter>,
    /// Child-process spawners, built lazily from the request's options.
    process: ProcessMemberSpawner,
    /// External-CLI spawner (vendor CLI members).
    external_cli: ExternalCliMemberSpawner,
    /// SSH transport spawner.
    ssh: SshMemberSpawner,
    /// The backend each spawned member uses, so `despawn` reaches the right child.
    kinds: Mutex<BTreeMap<String, MemberSpawnKind>>,
    /// Forked child contexts produced by [`MemberSpawner::spawn`], for the caller to inspect.
    forks: Mutex<BTreeMap<String, ForkedMemberSpec>>,
    /// Team-memory wiring the spawned members mount their tools from.
    ///
    /// Seeded from [`TeamBuildDeps::memory`] and replaced by the build path once it has resolved
    /// the team's wiring (`install_memory`), because the document lives under the leader's
    /// workspace, which is allocated after this spawner is built.
    memory: Mutex<Option<Arc<TeamMemoryWiring>>>,
}
impl TeamMemberSpawner {
    /// Builds a spawner for `spec` over `deps`, opening member harnesses under `session_id`.
    #[must_use]
    pub fn new(
        spec: TeamAgentSpec,
        deps: TeamBuildDeps,
        session_id: impl Into<String>,
    ) -> Arc<Self> {
        let routing =
            deps.routing
                .clone()
                .unwrap_or_else(|| match TeamModelRouter::for_spec(&spec) {
                    Ok(routing) => Arc::new(routing),
                    Err(error) => {
                        warn!(%error, "model pool is invalid; spawning without routed models");
                        Arc::new(TeamModelRouter::empty())
                    }
                });
        let options = spawn_options(&spec);
        let fork = fork_directive(&spec);
        Self::with_parts(spec, deps, session_id.into(), options, fork, routing)
    }
    /// Builds the spawner from already-parsed parts (the build path parses once and shares).
    #[must_use]
    pub fn with_parts(
        spec: TeamAgentSpec,
        deps: TeamBuildDeps,
        session_id: String,
        options: MemberSpawnOptions,
        fork: Option<ForkDirective>,
        routing: Arc<TeamModelRouter>,
    ) -> Arc<Self> {
        let memory = deps.memory.clone();
        Arc::new(Self {
            spec,
            deps,
            session_id,
            options,
            fork,
            routing,
            process: ProcessMemberSpawner::new(ProcessMemberConfig::default()),
            external_cli: ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig::default()),
            ssh: SshMemberSpawner::new(SshTransportSpec::default()),
            kinds: Mutex::new(BTreeMap::new()),
            forks: Mutex::new(BTreeMap::new()),
            memory: Mutex::new(memory),
        })
    }
    /// Returns the blueprint this spawner materializes members from.
    #[must_use]
    pub fn spec(&self) -> &TeamAgentSpec {
        &self.spec
    }
    /// Returns the injected build dependencies.
    #[must_use]
    pub fn deps(&self) -> &TeamBuildDeps {
        &self.deps
    }
    /// Returns the team session id a spawned member derives its own session id from.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    /// Returns the team name, copied from the blueprint.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.spec.team_name
    }
    /// Returns the spawn options the blueprint asks for.
    #[must_use]
    pub fn options(&self) -> &MemberSpawnOptions {
        &self.options
    }
    /// Returns the fork directive, when this team forks from a parent member.
    #[must_use]
    pub fn fork(&self) -> Option<&ForkDirective> {
        self.fork.as_ref()
    }
    /// Returns the model router shared with the build path.
    #[must_use]
    pub fn routing(&self) -> &Arc<TeamModelRouter> {
        &self.routing
    }

    /// Returns the team-memory wiring a spawned member mounts its tools from, when enabled.
    #[must_use]
    pub fn memory(&self) -> Option<Arc<TeamMemoryWiring>> {
        self.memory.lock().clone()
    }

    /// Installs (or replaces) the team-memory wiring after the build path resolved it.
    ///
    /// The build path resolves the wiring once the leader's workspace exists and installs it here
    /// before the first spawn, so every teammate mounts the team's tools from the same wiring.
    pub fn install_memory(&self, memory: Arc<TeamMemoryWiring>) {
        *self.memory.lock() = Some(memory);
    }
    /// Returns the backend a member was spawned with (`None` before its first spawn).
    #[must_use]
    pub fn spawn_kind(&self, member_name: &str) -> Option<MemberSpawnKind> {
        self.kinds.lock().get(member_name).copied()
    }
    /// Returns the forked context produced for `member_name`, if it was forked.
    #[must_use]
    pub fn forked_member(&self, member_name: &str) -> Option<ForkedMemberSpec> {
        self.forks.lock().get(member_name).cloned()
    }
    /// Returns the blueprint of one member: `agents[name]`, else `agents["teammate"]`.
    #[must_use]
    pub fn deep_agent_spec(&self, member_name: &str) -> Option<&DeepAgentSpec> {
        self.spec
            .agents
            .get(member_name)
            .or_else(|| self.spec.teammate_defaults())
    }
    /// Resolves a roster name to its [`TeamMemberSpec`], predefined entries first.
    #[must_use]
    pub fn member_spec(&self, member_name: &str) -> Option<TeamMemberSpec> {
        self.spec.member_spec(member_name)
    }
    /// Returns the spawnable roster names, sorted and deduplicated.
    #[must_use]
    pub fn member_names(&self) -> Vec<String> {
        let leader_name = self.spec.leader_member_name();
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for member in &self.spec.predefined_members {
            names.insert(member.member_name());
        }
        for name in self.spec.agents.keys() {
            if name == LEADER_AGENT_KEY || name == leader_name {
                continue;
            }
            names.insert(name.as_str());
        }
        names.into_iter().map(str::to_owned).collect()
    }
    /// Builds the forked child context for `member_name`, when a fork directive applies.
    ///
    /// The parent is resolved from the blueprint (`agents[parent]` and the roster entry), the tool
    /// policy is applied by the org fork helper and the result is cached so callers can inspect the
    /// child's lineage and task override. A directive that names the member itself is ignored.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the parent has no blueprint or roster entry, and the fork helper's
    /// validation for an illegal child name.
    pub fn forked_context(&self, member_name: &str) -> Result<Option<ForkedMemberSpec>> {
        let Some(directive) = self.fork.as_ref() else {
            return Ok(None);
        };
        if directive.parent == member_name {
            return Ok(None);
        }
        if let Some(cached) = self.forks.lock().get(member_name) {
            return Ok(Some(cached.clone()));
        }
        let parent_spec = self.deep_agent_spec(&directive.parent).ok_or_else(|| {
            Error::not_found(format!(
                "fork parent '{}' has no blueprint in team '{}'",
                directive.parent,
                self.team_name()
            ))
        })?;
        let parent_member = self.member_spec(&directive.parent).ok_or_else(|| {
            Error::not_found(format!(
                "fork parent '{}' is not on the roster of team '{}'",
                directive.parent,
                self.team_name()
            ))
        })?;
        let child = self.member_spec(member_name).ok_or_else(|| {
            Error::not_found(format!(
                "member '{member_name}' not found in team '{}'",
                self.team_name()
            ))
        })?;
        let source = ForkSource::new(
            self.team_name(),
            directive.parent.clone(),
            self.session_id.clone(),
            parent_member,
            parent_spec.clone(),
        );
        let forked = fork_member_context(&source, child, directive.task.clone(), &directive.tools)?;
        debug!(
            team = self.team_name(),
            member = member_name,
            parent = %directive.parent,
            depth = forked.fork_depth,
            "forked member context from its parent"
        );
        self.forks
            .lock()
            .insert(member_name.to_owned(), forked.clone());
        Ok(Some(forked))
    }
    /// Resolves the effective blueprint for `member_name`: fork child spec, then routed model.
    fn effective_spec(
        &self,
        member_name: &str,
        blueprint: &DeepAgentSpec,
        forked: Option<&ForkedMemberSpec>,
    ) -> Result<DeepAgentSpec> {
        let base = forked.map_or_else(|| blueprint.clone(), |child| child.spec.clone());
        let requested = self
            .member_spec(member_name)
            .and_then(|member| member.base.model_name);
        let (effective, selection) = self.routing.effective_spec_for(
            member_name,
            &base,
            requested.as_deref(),
            TeamRole::Teammate,
        )?;
        debug!(
            team = self.team_name(),
            member = member_name,
            model = selection.model_name(),
            pooled = selection.entry.is_some(),
            "selected the member model"
        );
        Ok(effective)
    }
    /// Builds one member's in-process runtime: the harness, then its identity and checkpoint wiring.
    ///
    /// The member runs under its *derived* session id
    /// ([`MemberHarnessContext::member_session_id`]) while the team session stays the request's
    /// session id, which is what the harness sees as `host_session_id`. The checkpoint card id comes
    /// from the effective spec the factory just received, so a routed or forked blueprint checkpoints
    /// under the card it actually runs; the store and policy come from `metadata["checkpoint"]` and
    /// the team-shared store the build path resolved.
    async fn build_harness_runtime(
        &self,
        member_name: &str,
        spec: &DeepAgentSpec,
        workspace_root: &std::path::Path,
    ) -> Result<HarnessMemberRuntime> {
        let checkpoint = checkpoint_config(&self.spec);
        let card_id = member_card_id(spec, member_name);
        let member_session =
            MemberHarnessContext::member_session_id(self.session_id(), member_name);
        let context = MemberHarnessContext::new(
            self.team_name(),
            member_name,
            member_session.clone(),
            workspace_root,
            self.deps.resolved_language(),
            self.spec.teammate_mode,
        )
        .with_card_id(card_id.clone());
        let context = match self.memory() {
            Some(memory) => {
                context.with_memory_toolkit(memory.toolkit(member_name, Some(self.session_id()))?)
            }
            None => context,
        };
        let harness = self.deps.harness_factory.create(spec, &context).await?;
        Ok(wire_runtime(
            HarnessMemberRuntime::new(harness, member_session),
            &card_id,
            &checkpoint,
            self.deps.checkpoints.as_ref(),
            workspace_root,
        ))
    }
    /// Forwards a non-in-process request to the matching child spawner.
    async fn spawn_child(&self, mut request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
        let kind = request.options.kind;
        let member_name = request.member_name.clone();
        // The served member reads its checkpoint policy from the request's env, so the leader's own
        // `metadata["checkpoint"]` block has to travel with the spawn; without it the child falls
        // back to its defaults.
        request = request.with_checkpoint_config(checkpoint_config(&self.spec));
        self.apply_kv_env(&mut request);
        let runtime = match kind {
            MemberSpawnKind::InProcess => unreachable!("in-process is handled before dispatch"),
            MemberSpawnKind::Process => self.process.spawn(request.clone()).await?,
            MemberSpawnKind::ExternalCli => self.external_cli.spawn(request.clone()).await?,
            MemberSpawnKind::Ssh => self.ssh.spawn(request.clone()).await?,
        };
        self.kinds.lock().insert(member_name, kind);
        debug!(
            team = self.team_name(),
            member = %request.member_name,
            mode = kind.as_str(),
            "spawned a member through its child-process backend"
        );
        Ok(runtime)
    }
    /// Copies the KV-cache affinity env onto a child command (process/external-CLI only).
    ///
    /// The SSH transport carries no env bag: its identity travels in the remote command, which is
    /// built by the SSH spawner; affinity is therefore in-process/process-only (documented).
    fn apply_kv_env(&self, request: &mut MemberSpawnRequest) {
        let Some(kv) = self.deps.kv_cache.as_ref() else {
            return;
        };
        let Some(env) = kv.env(&request.member_name) else {
            return;
        };
        if let Some(process) = request.options.process.as_mut() {
            process.command.env.extend(env.clone());
        }
        if let Some(external) = request.options.external_cli.as_mut() {
            external.command.env.extend(env);
        }
    }
}
impl fmt::Debug for TeamMemberSpawner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamMemberSpawner")
            .field("team_name", &self.team_name())
            .field("session_id", &self.session_id)
            .field("factory", &self.deps.harness_factory.name())
            .field("spawn_kind", &self.options.kind.as_str())
            .field("fork", &self.fork.as_ref().map(|fork| fork.parent.clone()))
            .finish_non_exhaustive()
    }
}
#[async_trait]
impl MemberSpawner for TeamMemberSpawner {
    /// Materializes one member's runtime through the backend its options select.
    ///
    /// The in-process path builds the harness through the injected factory and returns it as an
    /// **unstarted** runtime ([`MemberRuntime::start`] belongs to the member's lifecycle). The
    /// process/external-CLI/SSH paths delegate to their spawners, which own the child and reap it
    /// on [`MemberSpawner::despawn`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the member has no blueprint, [`Error::Unsupported`] for a
    /// role that cannot run in-process, and propagates the factory's or the child spawner's error.
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
        if !request.options.kind.is_in_process() {
            return self.spawn_child(request).await;
        }
        let member_name = request.member_name.clone();
        let Some(blueprint) = self.deep_agent_spec(&member_name) else {
            return Err(Error::not_found(format!(
                "member '{member_name}' not found in team '{}'",
                self.team_name()
            )));
        };
        if let Some(member) = self.member_spec(&member_name) {
            if member.role_type != TeamRole::Teammate {
                return Err(Error::unsupported(format!(
                    "member '{member_name}' has role '{}'; \
                     only teammate members can be spawned in-process",
                    member.role_type.as_str()
                )));
            }
        }
        let forked = self.forked_context(&member_name)?;
        let spec = self.effective_spec(&member_name, blueprint, forked.as_ref())?;
        let runtime: Arc<dyn MemberRuntime> = Arc::new(
            self.build_harness_runtime(&member_name, &spec, &request.workspace_root)
                .await?,
        );
        self.kinds
            .lock()
            .insert(member_name.clone(), MemberSpawnKind::InProcess);
        debug!(
            team = self.team_name(),
            member = %member_name,
            session = runtime.session_id(),
            factory = self.deps.harness_factory.name(),
            "spawned a member harness"
        );
        Ok(runtime)
    }
    /// Stops the member's child process, when its backend owns one.
    ///
    /// In-process members have no process to reap, so the call is a logged no-op for them. Called by
    /// the runtime at finalize (`shutdown_member`/`stop`) so a child never outlives its team.
    async fn despawn(&self, member_name: &str) -> Result<()> {
        let kind = self.kinds.lock().remove(member_name);
        match kind {
            Some(MemberSpawnKind::Process) => self.process.despawn(member_name).await?,
            Some(MemberSpawnKind::ExternalCli) => self.external_cli.despawn(member_name).await?,
            Some(MemberSpawnKind::Ssh) => self.ssh.despawn(member_name).await?,
            Some(MemberSpawnKind::InProcess) | None => {
                debug!(
                    team = self.team_name(),
                    member = member_name,
                    "no child process to reap; the runner owns member teardown"
                );
            }
        }
        self.forks.lock().remove(member_name);
        Ok(())
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_harness_protocol::{
        AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessContext,
        HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, HarnessState,
        SendReceipt, VecEventCursor,
    };
    use parking_lot::Mutex;

    use super::*;
    use crate::harness::MemberHarnessFactory;
    use crate::schema::{MemberMode, MemberSpecBase, PredefinedMemberSpec};

    /// Harness factory that records every context it is asked to build and can fail on demand.
    #[derive(Default)]
    struct RecordingFactory {
        /// Contexts received by [`MemberHarnessFactory::create`], in call order.
        calls: Mutex<Vec<MemberHarnessContext>>,
        /// When set, `create` fails with `Error::config(<message>)`.
        failure: Mutex<Option<String>>,
    }

    impl RecordingFactory {
        /// Builds a factory that always succeeds.
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        /// Builds a factory whose `create` fails with `message`.
        fn failing(message: &str) -> Arc<Self> {
            let factory = Self::default();
            *factory.failure.lock() = Some(message.to_owned());
            Arc::new(factory)
        }

        /// Returns the recorded contexts.
        fn calls(&self) -> Vec<MemberHarnessContext> {
            self.calls.lock().clone()
        }
    }

    #[async_trait]
    impl MemberHarnessFactory for RecordingFactory {
        async fn create(
            &self,
            _spec: &DeepAgentSpec,
            ctx: &MemberHarnessContext,
        ) -> Result<Arc<dyn HarnessProtocol>> {
            self.calls.lock().push(ctx.clone());
            let failure = self.failure.lock().clone();
            if let Some(message) = failure {
                return Err(Error::config(message));
            }
            Ok(Arc::new(StubHarness::new()))
        }

        fn name(&self) -> &'static str {
            "recording"
        }
    }

    /// Minimal [`HarnessProtocol`] with no behavior: enough for the spawner to hand out a runtime.
    struct StubHarness {
        /// State reported by [`HarnessProtocol::state`].
        state: Mutex<HarnessState>,
        /// Static capability card.
        card: HarnessCard,
        /// Observation buffer policy.
        buffer: EventBufferConfig,
    }

    impl StubHarness {
        /// Builds an idle stub.
        fn new() -> Self {
            Self {
                state: Mutex::new(HarnessState::Idle),
                card: HarnessCard::new("stub", "0.0.0"),
                buffer: EventBufferConfig::default(),
            }
        }
    }

    #[async_trait]
    #[allow(clippy::needless_pass_by_value)]
    impl HarnessProtocol for StubHarness {
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
            *self.state.lock() = HarnessState::Idle;
            Ok(())
        }

        async fn stop(&self) -> HarnessResult<()> {
            *self.state.lock() = HarnessState::Terminated;
            Ok(())
        }

        fn events(&self) -> Box<dyn HarnessEventCursor> {
            Box::new(VecEventCursor::new(Vec::new()))
        }

        fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
            Box::new(VecEventCursor::new(Vec::new()))
        }

        async fn send(
            &self,
            _input: HarnessInput,
            mode: DeliveryMode,
        ) -> HarnessResult<SendReceipt> {
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
            *self.state.lock() = HarnessState::Paused;
            Ok(())
        }

        async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
            *self.state.lock() = HarnessState::Running;
            Ok(())
        }

        async fn export_checkpoint(
            &self,
        ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
            Ok(None)
        }
    }

    /// Builds a blueprint with `dev-1` (`agents` only), `predefined` and `agents["teammate"]`.
    fn spec() -> TeamAgentSpec {
        let mut agents = BTreeMap::new();
        agents.insert(LEADER_AGENT_KEY.to_owned(), DeepAgentSpec::default());
        agents.insert(
            "teammate".to_owned(),
            DeepAgentSpec {
                system_prompt: Some("shared teammate prompt".to_owned()),
                ..DeepAgentSpec::default()
            },
        );
        agents.insert(
            "dev-1".to_owned(),
            DeepAgentSpec {
                system_prompt: Some("dev-1 prompt".to_owned()),
                ..DeepAgentSpec::default()
            },
        );
        TeamAgentSpec {
            team_name: "dev_team".to_owned(),
            agents,
            teammate_mode: MemberMode::PlanMode,
            predefined_members: vec![PredefinedMemberSpec::Teammate(MemberSpecBase {
                member_name: "reviewer".to_owned(),
                display_name: "Reviewer".to_owned(),
                ..MemberSpecBase::default()
            })],
            ..TeamAgentSpec::default()
        }
    }

    /// Builds dependencies over `factory` at the `en` language.
    fn deps(factory: Arc<RecordingFactory>) -> TeamBuildDeps {
        TeamBuildDeps::new()
            .with_harness_factory(factory)
            .with_language("en")
    }

    /// Builds a spawn request rooted at `/tmp/ws`.
    fn request(member_name: &str) -> MemberSpawnRequest {
        MemberSpawnRequest::new("dev_team", member_name, "session-1", "/tmp/ws")
    }

    /// Unwraps the error of a spawn result; `dyn MemberRuntime` is not `Debug`, so `expect_err`
    /// cannot be used.
    fn spawn_error(result: Result<Arc<dyn MemberRuntime>>, context: &str) -> Error {
        match result {
            Ok(_) => panic!("expected a spawn error: {context}"),
            Err(error) => error,
        }
    }

    #[test]
    fn accessors_expose_the_injected_values() {
        let spawner = TeamMemberSpawner::new(spec(), deps(RecordingFactory::new()), "session-1");
        assert_eq!(spawner.team_name(), "dev_team");
        assert_eq!(spawner.session_id(), "session-1");
        assert_eq!(spawner.spec().team_name, "dev_team");
        assert_eq!(spawner.deps().resolved_language(), "en");
        assert_eq!(spawner.deps().harness_factory.name(), "recording");
        let debug = format!("{spawner:?}");
        assert!(debug.contains("dev_team"), "{debug}");
        assert!(debug.contains("session-1"), "{debug}");
        assert!(debug.contains("recording"), "{debug}");
    }

    #[test]
    fn member_names_lists_predefined_and_agent_keys_without_the_leader() {
        let spawner = TeamMemberSpawner::new(spec(), deps(RecordingFactory::new()), "session-1");
        assert_eq!(
            spawner.member_names(),
            vec![
                "dev-1".to_owned(),
                "reviewer".to_owned(),
                "teammate".to_owned()
            ]
        );
    }

    #[test]
    fn member_names_skips_the_leader_keyed_under_its_member_name() {
        let mut spec = spec();
        spec.leader.member_name = "boss".to_owned();
        spec.agents
            .insert("boss".to_owned(), DeepAgentSpec::default());
        let spawner = TeamMemberSpawner::new(spec, deps(RecordingFactory::new()), "session-1");
        assert_eq!(
            spawner.member_names(),
            vec![
                "dev-1".to_owned(),
                "reviewer".to_owned(),
                "teammate".to_owned()
            ]
        );
    }

    #[test]
    fn member_names_deduplicates_a_predefined_and_agent_name() {
        let mut spec = spec();
        spec.predefined_members = vec![PredefinedMemberSpec::Teammate(MemberSpecBase {
            member_name: "dev-1".to_owned(),
            ..MemberSpecBase::default()
        })];
        let spawner = TeamMemberSpawner::new(spec, deps(RecordingFactory::new()), "session-1");
        assert_eq!(
            spawner.member_names(),
            vec!["dev-1".to_owned(), "teammate".to_owned()]
        );
    }

    #[test]
    fn deep_agent_spec_prefers_the_named_agent_and_falls_back() {
        let spawner = TeamMemberSpawner::new(spec(), deps(RecordingFactory::new()), "session-1");
        assert_eq!(
            spawner
                .deep_agent_spec("dev-1")
                .and_then(|spec| spec.system_prompt.as_deref()),
            Some("dev-1 prompt")
        );
        assert_eq!(
            spawner
                .deep_agent_spec("reviewer")
                .and_then(|spec| spec.system_prompt.as_deref()),
            Some("shared teammate prompt")
        );
        let mut bare = spec();
        bare.agents.remove("teammate");
        let spawner = TeamMemberSpawner::new(bare, deps(RecordingFactory::new()), "session-1");
        assert!(spawner.deep_agent_spec("reviewer").is_none());
    }

    #[test]
    fn member_spec_prefers_a_predefined_entry_over_the_agent_blueprint() {
        let mut spec = spec();
        spec.predefined_members = vec![PredefinedMemberSpec::Teammate(MemberSpecBase {
            member_name: "dev-1".to_owned(),
            display_name: "Predefined Dev".to_owned(),
            ..MemberSpecBase::default()
        })];
        let spawner = TeamMemberSpawner::new(spec, deps(RecordingFactory::new()), "session-1");
        let member = spawner
            .member_spec("dev-1")
            .expect("dev-1 is on the roster");
        assert_eq!(member.base.display_name, "Predefined Dev");
        assert_eq!(member.role_type, TeamRole::Teammate);
    }

    #[test]
    fn member_spec_derives_an_unknown_name_from_the_teammate_blueprint() {
        let spawner = TeamMemberSpawner::new(spec(), deps(RecordingFactory::new()), "session-1");
        let member = spawner
            .member_spec("dev-9")
            .expect("the teammate fallback applies");
        assert_eq!(member.base.member_name, "dev-9");
        assert_eq!(member.base.prompt, "shared teammate prompt");
        assert_eq!(member.role_type, TeamRole::Teammate);
    }

    #[test]
    fn member_spec_is_none_when_no_blueprint_matches() {
        let mut bare = spec();
        bare.agents.remove("teammate");
        let spawner = TeamMemberSpawner::new(bare, deps(RecordingFactory::new()), "session-1");
        assert!(spawner.member_spec("ghost").is_none());
    }

    #[tokio::test]
    async fn spawn_builds_the_harness_from_the_request_context() {
        let factory = RecordingFactory::new();
        let spawner = TeamMemberSpawner::new(spec(), deps(Arc::clone(&factory)), "session-1");
        let runtime = spawner
            .spawn(request("dev-1"))
            .await
            .expect("spawn succeeds");
        assert_eq!(runtime.session_id(), "session-1:dev-1");
        let calls = factory.calls();
        assert_eq!(calls.len(), 1);
        let ctx = calls.first().expect("one call was recorded");
        assert_eq!(ctx.team_name, "dev_team");
        assert_eq!(ctx.member_name, "dev-1");
        assert_eq!(ctx.session_id, "session-1:dev-1");
        assert_eq!(ctx.workspace_root, PathBuf::from("/tmp/ws"));
        assert_eq!(ctx.language, "en");
        assert_eq!(ctx.mode, MemberMode::PlanMode);
        assert_eq!(
            ctx.card_id, "dev-1",
            "a spec without a card falls back to the member name"
        );
    }

    #[tokio::test]
    async fn spawn_derives_the_member_session_from_the_team_session() {
        let spawner = TeamMemberSpawner::new(spec(), deps(RecordingFactory::new()), "session-1");
        let runtime = spawner
            .spawn(MemberSpawnRequest::new(
                "dev_team",
                "reviewer",
                "member-session-9",
                "/tmp/ws",
            ))
            .await
            .expect("spawn succeeds");
        assert_eq!(
            runtime.session_id(),
            "session-1:reviewer",
            "the member derives its session from the team session the spawner holds"
        );
        assert_eq!(runtime.state(), HarnessState::Idle);
    }

    /// Builds the concrete runtime of `dev-1`, which the `dyn MemberRuntime` returned by `spawn`
    /// hides (no checkpoint accessor on the trait).
    async fn built_runtime(spawner: &TeamMemberSpawner) -> crate::harness::HarnessMemberRuntime {
        let spec = spawner
            .deep_agent_spec("dev-1")
            .cloned()
            .expect("dev-1 has a blueprint");
        spawner
            .build_harness_runtime("dev-1", &spec, std::path::Path::new("/tmp/ws"))
            .await
            .expect("the runtime builds")
    }

    #[tokio::test]
    async fn a_default_spec_persists_checkpoints_in_memory() {
        let spawner = TeamMemberSpawner::new(spec(), deps(RecordingFactory::new()), "session-1");
        let runtime = built_runtime(&spawner).await;
        assert_eq!(runtime.session_id(), "session-1:dev-1");
        assert_eq!(runtime.card_id(), "dev-1");
        assert_eq!(runtime.cwd(), Some("/tmp/ws"));
        let store = runtime
            .checkpoint_store()
            .expect("the default policy is enabled");
        assert_eq!(store.name(), "memory");
        assert!(runtime.checkpoint_policy().is_enabled());
    }

    #[tokio::test]
    async fn a_disabled_checkpoint_policy_attaches_no_store() {
        let mut spec = spec();
        spec.metadata.insert(
            crate::runtime::spec_meta::CHECKPOINT_METADATA_KEY.to_owned(),
            serde_json::json!({"enabled": false}),
        );
        let spawner = TeamMemberSpawner::new(spec, deps(RecordingFactory::new()), "session-1");
        let runtime = built_runtime(&spawner).await;
        assert!(runtime.checkpoint_store().is_none());
        assert!(!runtime.checkpoint_policy().is_enabled());
    }

    #[tokio::test]
    async fn a_file_checkpoint_config_roots_the_store_on_disk() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let mut spec = spec();
        spec.metadata.insert(
            crate::runtime::spec_meta::CHECKPOINT_METADATA_KEY.to_owned(),
            serde_json::json!({"store": "file", "dir": directory.path().to_string_lossy()}),
        );
        let spawner = TeamMemberSpawner::new(spec, deps(RecordingFactory::new()), "session-1");
        let runtime = built_runtime(&spawner).await;
        let store = runtime.checkpoint_store().expect("a file store");
        assert_eq!(store.name(), "file");
    }

    #[tokio::test]
    async fn an_injected_team_store_wins_over_the_metadata_backend() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let mut spec = spec();
        spec.metadata.insert(
            crate::runtime::spec_meta::CHECKPOINT_METADATA_KEY.to_owned(),
            serde_json::json!({"store": "file", "dir": directory.path().to_string_lossy()}),
        );
        let injected = crate::harness::InMemoryMemberCheckpointStore::shared();
        let deps = deps(RecordingFactory::new()).with_checkpoint_store(Arc::clone(&injected));
        let spawner = TeamMemberSpawner::new(spec, deps, "session-1");
        let runtime = built_runtime(&spawner).await;
        let store = runtime.checkpoint_store().expect("the shared store");
        assert!(Arc::ptr_eq(store, &injected));
        assert_eq!(store.name(), "memory");
    }

    #[tokio::test]
    async fn spawn_rejects_a_member_without_a_blueprint() {
        let mut bare = spec();
        bare.agents.remove("teammate");
        let factory = RecordingFactory::new();
        let spawner = TeamMemberSpawner::new(bare, deps(Arc::clone(&factory)), "session-1");
        let error = spawn_error(spawner.spawn(request("ghost")).await, "no blueprint");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.message(),
            "member 'ghost' not found in team 'dev_team'"
        );
        assert_eq!(factory.calls(), Vec::new(), "the factory was not called");
    }

    #[tokio::test]
    async fn spawn_rejects_a_non_teammate_role() {
        let mut spec = spec();
        spec.predefined_members = vec![PredefinedMemberSpec::HumanAgent(MemberSpecBase {
            member_name: "alice".to_owned(),
            ..MemberSpecBase::default()
        })];
        let factory = RecordingFactory::new();
        let spawner = TeamMemberSpawner::new(spec, deps(Arc::clone(&factory)), "session-1");
        let error = spawn_error(spawner.spawn(request("alice")).await, "a human");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "member 'alice' has role 'human_agent'; \
             only teammate members can be spawned in-process"
        );
        assert_eq!(factory.calls(), Vec::new(), "the factory was not called");
    }

    #[tokio::test]
    async fn spawn_propagates_a_factory_error_unchanged() {
        let factory = RecordingFactory::failing("model 'gpt-x' is not configured");
        let spawner = TeamMemberSpawner::new(spec(), deps(factory), "session-1");
        let error = spawn_error(spawner.spawn(request("dev-1")).await, "factory fails");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.message(), "model 'gpt-x' is not configured");
    }

    #[tokio::test]
    async fn despawn_is_a_no_op() {
        let factory = RecordingFactory::new();
        let spawner = TeamMemberSpawner::new(spec(), deps(Arc::clone(&factory)), "session-1");
        spawner.despawn("dev-1").await.expect("despawn succeeds");
        spawner
            .despawn("never-spawned")
            .await
            .expect("unknown members are not an error");
        assert_eq!(factory.calls(), Vec::new(), "the factory was not called");
    }
}
