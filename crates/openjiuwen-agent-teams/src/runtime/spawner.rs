//! The harness-backed [`MemberSpawner`](crate::member::MemberSpawner): one member's cold spec in,
//! a live member runtime out.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/inprocess_spawn.py`, which builds a
//! teammate `TeamAgent` from the blueprint before wrapping it in an `asyncio.Task`. The Rust port
//! splits that in two: this spawner materializes the member's *brain* through the injected
//! [`MemberHarnessFactory`](crate::harness::MemberHarnessFactory) and returns it as a
//! [`MemberRuntime`](crate::member::MemberRuntime); owning the driving task is the
//! caller's job (`InProcessSpawnHandle` / `TeamBackend`).
//!
//! The spawner is deliberately stateless beyond its three injected values — the blueprint, the
//! build dependencies and the session id — so it is cheap to share and adds no global state to
//! the runtime node.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_teams::runtime::deps::TeamBuildDeps;
//! use openjiuwen_agent_teams::runtime::spawner::TeamMemberSpawner;
//! use openjiuwen_agent_teams::schema::TeamAgentSpec;
//!
//! let spawner = TeamMemberSpawner::new(TeamAgentSpec::default(), TeamBuildDeps::new(), "s-1");
//! assert_eq!(spawner.session_id(), "s-1");
//! assert_eq!(spawner.team_name(), "agent_team");
//! assert!(spawner.member_names().is_empty());
//! ```
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use tracing::debug;

use super::deps::TeamBuildDeps;
use crate::harness::{HarnessMemberRuntime, MemberHarnessContext};
use crate::member::{MemberRuntime, MemberSpawnRequest, MemberSpawner};
use crate::schema::{DeepAgentSpec, TeamAgentSpec, TeamMemberSpec, TeamRole};

/// Key of the leader's blueprint in [`TeamAgentSpec::agents`].
///
/// Mirrors the private constant of `crate::schema::spec`; the leader is never a spawnable roster
/// name, so the roster projection here has to know the key without widening the schema surface.
const LEADER_AGENT_KEY: &str = "leader";

/// Spawns team members from the blueprint through the injected harness factory.
///
/// One spawner serves one built team: it carries the roster (`spec`), the injection point for the
/// member's brain (`deps.harness_factory`) and the session id every member's harness is opened
/// under. Building a runtime is a pure projection of those three values plus the request, which is
/// why the same spawner can spawn every member concurrently.
pub struct TeamMemberSpawner {
    /// The blueprint the roster, the teammate mode and the team name come from.
    spec: TeamAgentSpec,
    /// The injected build dependencies; only `harness_factory` and `language` are consulted.
    deps: TeamBuildDeps,
    /// Session id every spawned harness runs under.
    session_id: String,
}

impl TeamMemberSpawner {
    /// Builds a spawner for `spec` over `deps`, opening member harnesses under `session_id`.
    #[must_use]
    pub fn new(
        spec: TeamAgentSpec,
        deps: TeamBuildDeps,
        session_id: impl Into<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            spec,
            deps,
            session_id: session_id.into(),
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

    /// Returns the session id every spawned member harness is opened under.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the team name, copied from the blueprint.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.spec.team_name
    }

    /// Returns the blueprint of one member: `agents[name]`, else `agents["teammate"]`.
    ///
    /// The teammate fallback is the shared default every unnamed roster member runs on, so a team
    /// that declares only `agents["teammate"]` can still spawn any predefined member.
    #[must_use]
    pub fn deep_agent_spec(&self, member_name: &str) -> Option<&DeepAgentSpec> {
        self.spec
            .agents
            .get(member_name)
            .or_else(|| self.spec.teammate_defaults())
    }

    /// Resolves a roster name to its [`TeamMemberSpec`], predefined entries first.
    ///
    /// Delegates to [`TeamAgentSpec::member_spec`], so a predefined member always wins over the
    /// agent blueprint registered under the same name, and an unknown name derives from
    /// `agents["teammate"]`. `None` means the name is on no roster at all.
    #[must_use]
    pub fn member_spec(&self, member_name: &str) -> Option<TeamMemberSpec> {
        self.spec.member_spec(member_name)
    }

    /// Returns the spawnable roster names, sorted and deduplicated.
    ///
    /// The roster is the predefined members plus every `agents` key, minus `agents["leader"]` and
    /// the leader's member name: the leader is never spawned as a teammate. Sorted output keeps
    /// `TeamAgent`'s row materialization and the CLI listing deterministic.
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
}

impl fmt::Debug for TeamMemberSpawner {
    /// Prints the identity that matters when a spawn is traced: the team, the session and the
    /// factory name. The blueprint itself is omitted because it carries prompts and, through a
    /// model reference, potential credentials.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamMemberSpawner")
            .field("team_name", &self.team_name())
            .field("session_id", &self.session_id)
            .field("factory", &self.deps.harness_factory.name())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl MemberSpawner for TeamMemberSpawner {
    /// Materializes one member's harness and returns it as an unstarted member runtime.
    ///
    /// The caller owns the workspace root in the request: the spawner forwards it verbatim into
    /// [`MemberHarnessContext::workspace_root`] instead of allocating its own. The runtime is
    /// **not** started — [`MemberRuntime::start`] belongs to the member's lifecycle so a failed
    /// start can roll the roster row back.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the member has no blueprint (`agents[name]` and
    /// `agents["teammate"]` are both absent), [`Error::Unsupported`] when the resolved member role
    /// is not [`TeamRole::Teammate`] (human, bridge and external-CLI members are not built
    /// in-process), and propagates the harness factory's own error unchanged.
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
        let member_name = request.member_name;
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
        let context = MemberHarnessContext::new(
            self.team_name(),
            member_name,
            self.session_id(),
            request.workspace_root,
            self.deps.resolved_language(),
            self.spec.teammate_mode,
        );
        let harness = self
            .deps
            .harness_factory
            .create(blueprint, &context)
            .await?;
        let runtime: Arc<dyn MemberRuntime> =
            Arc::new(HarnessMemberRuntime::new(harness, request.session_id));
        debug!(
            team = self.team_name(),
            member = %context.member_name,
            session = runtime.session_id(),
            factory = self.deps.harness_factory.name(),
            "spawned a member harness"
        );
        Ok(runtime)
    }

    /// Does nothing and succeeds: teardown order belongs to the runner, not to the spawner.
    ///
    /// A spawned member is stopped by its owner (`TeamAgent`), which calls
    /// [`MemberRuntime::stop`] before dropping the spawn handle. There is no registry here to
    /// unregister from, so an unknown member is not an error.
    async fn despawn(&self, member_name: &str) -> Result<()> {
        debug!(
            team = self.team_name(),
            member = member_name,
            "despawn is a no-op; the runner owns member teardown"
        );
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
        assert_eq!(runtime.session_id(), "session-1");
        let calls = factory.calls();
        assert_eq!(calls.len(), 1);
        let ctx = calls.first().expect("one call was recorded");
        assert_eq!(ctx.team_name, "dev_team");
        assert_eq!(ctx.member_name, "dev-1");
        assert_eq!(ctx.session_id, "session-1");
        assert_eq!(ctx.workspace_root, PathBuf::from("/tmp/ws"));
        assert_eq!(ctx.language, "en");
        assert_eq!(ctx.mode, MemberMode::PlanMode);
    }

    #[tokio::test]
    async fn spawn_propagates_the_request_session_id_into_the_runtime() {
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
        assert_eq!(runtime.session_id(), "member-session-9");
        assert_eq!(runtime.state(), HarnessState::Idle);
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
