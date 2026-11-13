//! The blueprint → runtime build path: [`SpecBuild::build`] and [`SpecBuild::build_with`].
//!
//! Python parity source: `openjiuwen/agent_teams/agent/agent_configurator.py`, the wiring
//! `TeamAgent.__init__` performs. The build order is pinned by `src/runtime/CONTEXT.md`
//! §"Build path":
//!
//! 1. [`TeamAgentSpec::validate`](crate::schema::TeamAgentSpec::validate) — reserved names and the
//!    dispatch/bootstrap pairing.
//! 2. The runtime mirror ([`TeamSpec`](crate::schema::TeamSpec)), the resolved language, the parsed
//!    metadata knobs (spawn mode, isolation, fork, KV cache, attach).
//! 3. One store (injected, never a second one), one in-process transport, one activity registry,
//!    one signals bundle (observability hooks + monitor), one model router, one KV-cache registry.
//! 4. [`TeamBackend`] over the store, board, transport, spawner and workspace provider; the team
//!    row is written before any member row; the organisation layer is bound when configured.
//! 5. The leader row, its workspace and its harness (with the pool-selected model), then the
//!    observer that feeds status, rounds, chunks, observability and the monitor.
//! 6. One `unstarted` row per roster teammate, so `snapshot()` answers before the first spawn.
//! 7. One bus subscription for the leader (drained only by [`leader`]) and the interaction
//!    gate/inbox.
//!
//! The build always produces the *leader* handle: a teammate is not a second public constructor but
//! a [`TeamMember`](crate::member::TeamMember) whose runtime is materialized by
//! `TeamAgent::spawn_member` through the same harness-factory seam.
//!
//! # Async build
//!
//! The build path awaits the store, the workspace provider and the harness factory, so
//! [`SpecBuild`] is an `#[async_trait]` seam. The node contract sketches the trait with synchronous
//! signatures; the build path it pins is the authority (this deviation is reported by the node).
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;
use openjiuwen_multiagent::{RuntimeConfig, TeamRuntime};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio_util::sync::CancellationToken;
use tracing::debug;

use super::backend::TeamBackend;
use super::deps::TeamBuildDeps;
use super::observability_bind::TeamSignals;
use super::observer::MemberStreamObserver;
use super::org_bind::{org_task_sink, OrgBinding};
use super::routing::TeamModelRouter;
use super::spawner::TeamMemberSpawner;
use super::spec_meta::{
    attach_config, fork_directive, isolation_mode, kv_cache_config, organization_block,
    spawn_options,
};
use super::team_agent::{SpecBuild, TeamAgent, TeamAgentInner};
use crate::harness::{HarnessMemberRuntime, MemberHarnessContext};
use crate::interaction::{HumanAgentInbox, InteractGate};
use crate::kv_cache::{MemberCacheKey, TeamKvCacheAffinity};
use crate::member::{MemberActivityRegistry, MemberRuntime, MemberSpawner, TeamMember};
use crate::messager::create_messager;
use crate::schema::{
    DeepAgentSpec, Language, TeamAgentSpec, TeamMemberSpec, TeamRole, TeamSpec, TeamTopic,
};
use crate::storage::{MemberRecord, TeamInfoRecord, TeamStore};
use crate::taskboard::{TaskBoard, TaskBoardLimits};
use crate::workspace::WorkspaceRequest;

/// Key of the leader's blueprint in [`TeamAgentSpec::agents`].
const LEADER_KEY: &str = "leader";
/// Error text of a blueprint that has no `agents["leader"]` entry.
const LEADER_MISSING_MESSAGE: &str = "agents dict must contain a 'leader' key";
/// Prefix of the session id a built team mints for itself.
const SESSION_PREFIX: &str = "session_";

#[async_trait]
impl SpecBuild for TeamAgentSpec {
    /// Builds a leader agent over [`TeamBuildDeps::default`].
    async fn build(&self) -> Result<Arc<TeamAgent>> {
        self.build_with(TeamBuildDeps::default()).await
    }
    /// Builds a leader agent over `deps`, in the order pinned by the node contract.
    ///
    /// # Errors
    ///
    /// See [`SpecBuild::build_with`].
    // The build path is one documented linear sequence; splitting it would hide the order that
    // `src/runtime/CONTEXT.md` pins.
    #[allow(clippy::too_many_lines)]
    async fn build_with(&self, deps: TeamBuildDeps) -> Result<Arc<TeamAgent>> {
        self.validate()?;
        let leader_spec = self
            .agents
            .get(LEADER_KEY)
            .cloned()
            .ok_or_else(|| Error::validation(LEADER_MISSING_MESSAGE))?;
        let team_name = self.team_name.clone();
        let leader_name = self.leader_member_name().to_owned();
        let language = resolve_language(&deps, self.language.as_deref());
        let mut team_spec = TeamSpec::from_spec(self);
        team_spec.language = Some(language.clone());
        let session_id = format!("{SESSION_PREFIX}{}", uuid_hex());
        let messager = create_messager(&self.transport)?;
        let store = Arc::clone(&deps.store);
        let activity = Arc::new(MemberActivityRegistry::new());
        let routing = Arc::new(TeamModelRouter::for_spec(self)?);
        let kv_cache = deps
            .kv_cache
            .clone()
            .unwrap_or_else(|| Arc::new(TeamKvCacheAffinity::new(kv_cache_config(self))));
        let signals = TeamSignals::new(team_name.clone(), deps.observability_hooks.clone());
        let deps = deps
            .with_routing(Arc::clone(&routing))
            .with_kv_cache(Arc::clone(&kv_cache));
        let options = spawn_options(self);
        let fork = fork_directive(self);
        let isolation = isolation_mode(self);
        let attach = attach_config(self);
        let spawner = TeamMemberSpawner::with_parts(
            self.clone(),
            deps.clone(),
            session_id.clone(),
            options.clone(),
            fork.clone(),
            Arc::clone(&routing),
        );
        let limits = TaskBoardLimits::new()
            .with_default_max_review_rounds(
                u32::try_from(self.default_max_review_rounds).unwrap_or(u32::MAX),
            )
            .with_max_subtask_depth(self.max_subtask_depth);
        let taskboard = Arc::new(TaskBoard::new(store.clone(), team_name.clone(), limits));
        let backend = Arc::new(
            TeamBackend::new(
                team_spec.clone(),
                store.clone(),
                taskboard,
                messager.clone(),
                Arc::clone(&spawner) as Arc<dyn MemberSpawner>,
                deps.workspace.clone(),
                activity.clone(),
            )
            .with_signals(signals.clone()),
        );
        if let Some(block) = organization_block(self)? {
            let binding = Arc::new(OrgBinding::build(
                &block,
                store.clone(),
                Some(org_task_sink(Arc::clone(&backend))),
            )?);
            backend.install_org(binding);
        }
        backend
            .upsert_team_info(team_info(self, &team_spec, &leader_name, &leader_spec))
            .await?;
        let leader_record = leader_record(self, &leader_name, &leader_spec, &routing);
        store.insert_member(leader_record.clone()).await?;
        let leader_member = Arc::new(TeamMember::new(store.clone(), leader_record.clone()));
        let (leader_effective, leader_selection) =
            routing.effective_spec(&leader_spec, leader_spec.model_name(), TeamRole::Leader)?;
        let workspace = deps
            .workspace
            .allocate(&WorkspaceRequest {
                team_name: team_name.clone(),
                session_id: session_id.clone(),
                member_name: leader_name.clone(),
                spec: self.workspace.clone(),
                member_spec: leader_spec.workspace.clone(),
                isolation,
            })
            .await?;
        let harness = deps
            .harness_factory
            .create(
                &leader_effective,
                &MemberHarnessContext::new(
                    team_name.clone(),
                    leader_name.clone(),
                    session_id.clone(),
                    workspace.root.clone(),
                    language.clone(),
                    self.teammate_mode,
                ),
            )
            .await?;
        let runtime: Arc<dyn MemberRuntime> =
            Arc::new(HarnessMemberRuntime::new(harness, session_id.clone()));
        leader_member.attach_runtime(Arc::clone(&runtime));
        let model = leader_selection.model_name().to_owned();
        let _identity = kv_cache.register_member(
            MemberCacheKey::new(team_name.clone(), session_id.clone(), leader_name.clone()),
            &model,
        );
        let chunk_sinks: Arc<Mutex<Vec<UnboundedSender<Value>>>> = Arc::new(Mutex::new(Vec::new()));
        leader_member.observe(Arc::new(MemberStreamObserver::new(
            activity.clone(),
            Arc::downgrade(&leader_member),
            TeamRole::Leader,
            leader_name.clone(),
            signals.clone(),
            Arc::clone(&chunk_sinks),
        )));
        backend.add_member(Arc::clone(&leader_member));
        insert_teammate_rows(&store, &backend, &spawner, &routing, &leader_name).await?;
        let (tx, rx) = mpsc::unbounded_channel();
        messager.subscribe(
            &leader_name,
            &[TeamTopic::Team, TeamTopic::Task, TeamTopic::Message],
            Arc::new(move |event| {
                let _ = tx.send(event);
            }),
        )?;
        let gate = Arc::new(InteractGate::new());
        let inbox = Arc::new(HumanAgentInbox::new());
        inbox.set_enabled(self.enable_hitt);
        let card = AgentCard::new(format!("{team_name}_{leader_name}")).with_description(
            self.team_desc
                .clone()
                .unwrap_or_else(|| self.leader.desc_or_default().to_owned()),
        );
        let team_runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id(team_name));
        Ok(TeamAgent::assemble(TeamAgentInner {
            blueprint: self.clone(),
            spec: team_spec,
            role: TeamRole::Leader,
            member_name: leader_name,
            session_id,
            language,
            card,
            leader_member_name: self.leader_member_name().to_owned(),
            store,
            backend,
            member: leader_member,
            runtime,
            spawner,
            gate,
            activity,
            inbox,
            team_runtime,
            routing,
            kv_cache,
            signals,
            isolation,
            spawn_options: options,
            fork,
            attach,
            attach_host: Mutex::new(None),
            started: AtomicBool::new(false),
            cancel: Mutex::new(CancellationToken::new()),
            chunk_sinks,
            events: Mutex::new(Some(rx)),
            steer_batch_size: self.steer_batch_size,
        }))
    }
}

/// Resolves the effective language: the dependency override, else the blueprint's declaration.
fn resolve_language(deps: &TeamBuildDeps, declared: Option<&str>) -> String {
    let value = deps.language.as_deref().or(declared);
    Language::resolve(value).to_owned()
}

/// Builds the persisted team row from the blueprint and its runtime mirror.
fn team_info(
    blueprint: &TeamAgentSpec,
    team_spec: &TeamSpec,
    leader_name: &str,
    leader_spec: &DeepAgentSpec,
) -> TeamInfoRecord {
    TeamInfoRecord {
        team_name: team_spec.team_name.clone(),
        display_name: team_spec.display_name.clone(),
        leader_member_name: leader_name.to_owned(),
        desc: blueprint.team_desc.clone().unwrap_or_default(),
        prompt: leader_spec.system_prompt.clone().unwrap_or_default(),
        dispatch_mode: blueprint.dispatch_mode,
        enable_task_verification: blueprint.enable_task_verification,
        created: 0,
        updated_at: 0,
    }
}

/// Builds the leader's roster row.
fn leader_record(
    blueprint: &TeamAgentSpec,
    leader_name: &str,
    leader_spec: &DeepAgentSpec,
    routing: &TeamModelRouter,
) -> MemberRecord {
    MemberRecord {
        display_name: blueprint.leader.display_name.clone(),
        desc: blueprint.leader.desc_or_default().to_owned(),
        prompt: leader_spec.system_prompt.clone().unwrap_or_default(),
        mode: blueprint.teammate_mode,
        model_ref: routing.model_ref(leader_spec.model_name(), TeamRole::Leader),
        ..MemberRecord::new(&blueprint.team_name, leader_name, TeamRole::Leader)
    }
}

/// Builds one teammate's roster row from its resolved member spec and blueprint.
///
/// Also used by `TeamAgent::spawn_member` when a member first acquires a row.
pub(super) fn teammate_record(
    spawner: &TeamMemberSpawner,
    name: &str,
    routing: &TeamModelRouter,
) -> MemberRecord {
    let member_spec = spawner.member_spec(name);
    let blueprint = spawner.deep_agent_spec(name);
    let role = member_spec
        .as_ref()
        .map_or(TeamRole::Teammate, |spec| spec.role_type);
    let mut record = MemberRecord::new(spawner.team_name(), name, role);
    record.mode = spawner.spec().teammate_mode;
    record.model_ref = routing.model_ref(requested_model(member_spec.as_ref(), blueprint), role);
    if let Some(spec) = &member_spec {
        record.display_name.clone_from(&spec.base.display_name);
        record.desc.clone_from(&spec.base.desc);
        record.prompt.clone_from(&spec.base.prompt);
    }
    record
}

/// Returns the model a member asked for: its member spec first, then its blueprint.
fn requested_model<'a>(
    spec: Option<&'a TeamMemberSpec>,
    blueprint: Option<&'a DeepAgentSpec>,
) -> Option<&'a str> {
    spec.and_then(|spec| spec.base.model_name.as_deref())
        .or_else(|| blueprint.and_then(DeepAgentSpec::model_name))
}

/// Materializes one `unstarted` row per roster teammate.
async fn insert_teammate_rows(
    store: &Arc<dyn TeamStore>,
    backend: &Arc<TeamBackend>,
    spawner: &Arc<TeamMemberSpawner>,
    routing: &TeamModelRouter,
    leader_name: &str,
) -> Result<()> {
    for name in spawner.member_names() {
        if name == leader_name || backend.member(&name).is_some() {
            continue;
        }
        let record = teammate_record(spawner, &name, routing);
        store.insert_member(record.clone()).await?;
        backend.add_member(Arc::new(TeamMember::new(store.clone(), record)));
    }
    debug!(
        team = spawner.team_name(),
        members = spawner.member_names().len(),
        "materialized the teammate roster"
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::runtime::team_agent::test_support::{
        blueprint, built, deps, RecordingWorkspaceProvider, ScriptedFactory,
    };

    /// Builds a two-teammate blueprint for the build-path tests.
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
        assert_eq!(leader.status(), crate::schema::MemberStatus::Unstarted);
        assert_eq!(leader.record().prompt, "prompt for Leader");

        for name in ["dev-1", "dev-2"] {
            let member = agent.backend().member(name).unwrap();
            assert_eq!(member.role(), TeamRole::Teammate);
            assert_eq!(member.status(), crate::schema::MemberStatus::Unstarted);
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
                &RecordingWorkspaceProvider::new(),
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
        let workspace = RecordingWorkspaceProvider::new();
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
}
