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
//! 3. One store — an injected store wins over `spec.storage`, while a non-memory `spec.storage`
//!    opens its own backend through `crate::storage::open_store` (with a warning when an injected
//!    store overrode it) — plus one in-process transport, one activity registry, one signals bundle
//!    (observability hooks + monitor), one model router, one KV-cache registry, and the team-shared
//!    member checkpoint store resolved from `metadata["checkpoint"]` (injected into the dependencies
//!    so every member, leader included, persists into it).
//! 4. [`TeamBackend`] over the store, board, transport, spawner and workspace provider; the team
//!    row is written before any member row; the organisation layer is bound when configured.
//! 5. The leader row, its workspace and its harness (with the pool-selected model and its own
//!    checkpoint identity — derived member session id + card id, policy, store and cwd), then the
//!    observer that feeds status, rounds, chunks, observability and the monitor. The leader's
//!    workspace root also anchors the team-memory wiring: `memory_bind::resolve_memory` runs once
//!    per team (injected wiring → `metadata["memory"]` → none), installs the wiring on the backend
//!    (the turn seam) and on the spawner (the teammate toolkits), and mounts the leader's own
//!    `MemberMemoryToolkit` on its harness context. A resolution failure fails the build.
//! 6. One `unstarted` row per roster teammate, so `snapshot()` answers before the first spawn.
//! 7. One bus subscription for the leader (its receiver is leased by every [`leader`] run and
//!    returned to the handle when that run ends, so a second run is not blind) and the interaction
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
use std::collections::BTreeMap;
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
use tracing::{debug, warn};

use super::backend::TeamBackend;
use super::checkpoints::{member_card_id, team_checkpoint_store, wire_runtime};
use super::deps::TeamBuildDeps;
use super::memory_bind::resolve_memory;
use super::observability_bind::TeamSignals;
use super::observer::MemberStreamObserver;
use super::org_bind::{org_task_sink, OrgBinding};
use super::routing::TeamModelRouter;
use super::spawner::TeamMemberSpawner;
use super::spec_meta::{
    attach_config, checkpoint_config, fork_directive, isolation_mode, kv_cache_config,
    organization_block, spawn_options,
};
use super::team_agent::{SpecBuild, TeamAgent, TeamAgentInner};
use crate::harness::{HarnessMemberRuntime, MemberHarnessContext};
use crate::interaction::{HumanAgentInbox, InteractGate};
use crate::kv_cache::{MemberCacheKey, TeamKvCacheAffinity};
use crate::member::{MemberActivityRegistry, MemberRuntime, MemberSpawner, TeamMember};
use crate::messager::create_messager;
use crate::schema::{
    DeepAgentSpec, Language, StorageSpec, TeamAgentSpec, TeamMemberSpec, TeamRole, TeamSpec,
    TeamTopic,
};
use crate::storage::{open_store, MemberRecord, TeamInfoRecord, TeamStore, MEMORY_STORAGE_KIND};
use crate::taskboard::{TaskBoard, TaskBoardLimits};
use crate::workspace::WorkspaceRequest;

/// Key of the leader's blueprint in [`TeamAgentSpec::agents`].
const LEADER_KEY: &str = "leader";
/// Error text of a blueprint that has no `agents["leader"]` entry.
const LEADER_MISSING_MESSAGE: &str = "agents dict must contain a 'leader' key";
/// Prefix of the session id a built team mints for itself.
const SESSION_PREFIX: &str = "session_";
/// Storage kind of the SQLite backend (`crate::storage::open_store` accepts the same literal).
const SQLITE_KIND: &str = "sqlite";
/// `StorageSpec::params` keys that may name the SQLite database, in precedence order.
const SQLITE_TARGET_PARAMS: [&str; 3] = ["path", "url", "database"];

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
        let store = resolve_store(self, &deps)?;
        let activity = Arc::new(MemberActivityRegistry::new());
        let routing = Arc::new(TeamModelRouter::for_spec(self)?);
        let kv_cache = deps
            .kv_cache
            .clone()
            .unwrap_or_else(|| Arc::new(TeamKvCacheAffinity::new(kv_cache_config(self))));
        let signals = TeamSignals::new(team_name.clone(), deps.observability_hooks.clone());
        let checkpoint = checkpoint_config(self);
        let mut deps = deps
            .with_routing(Arc::clone(&routing))
            .with_kv_cache(Arc::clone(&kv_cache));
        // Resolve the team-shared checkpoint store once, before the spawner is built: the cloned
        // `deps` it receives carries the same store, so the leader and every teammate share it.
        if let Some(store) = team_checkpoint_store(&checkpoint, deps.checkpoints.as_ref()) {
            deps = deps.with_checkpoint_store(store);
        }
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
        let (leader_effective, leader_selection) = routing.effective_spec_for(
            &leader_name,
            &leader_spec,
            leader_spec.model_name(),
            TeamRole::Leader,
        )?;
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
        let card_id = member_card_id(&leader_effective, &leader_name);
        let member_session = MemberHarnessContext::member_session_id(&session_id, &leader_name);
        // Team memory is resolved once here, now that the leader's workspace gives the team its
        // document root; the spawner and the backend receive the same wiring, so the leader, every
        // teammate and the turn buffer share one team scope. Memory is a configured capability, so
        // a wiring that cannot be opened fails the build (the freshly allocated workspace goes back
        // rather than leaking for a team that never ran).
        let memory = match resolve_memory(self, &deps, &team_name, &workspace.root).await {
            Ok(memory) => memory,
            Err(error) => {
                abandon_workspace(&deps, &workspace, &team_name).await;
                return Err(error);
            }
        };
        if let Some(wiring) = memory.as_ref() {
            spawner.install_memory(Arc::clone(wiring));
            backend.install_memory(Arc::clone(wiring));
        }
        let context = MemberHarnessContext::new(
            team_name.clone(),
            leader_name.clone(),
            member_session.clone(),
            workspace.root.clone(),
            language.clone(),
            self.teammate_mode,
        )
        .with_card_id(card_id.clone());
        let context = match memory.as_ref() {
            Some(wiring) => match wiring.toolkit(&leader_name, Some(&session_id)) {
                Ok(toolkit) => context.with_memory_toolkit(toolkit),
                Err(error) => {
                    abandon_workspace(&deps, &workspace, &team_name).await;
                    return Err(error);
                }
            },
            None => context,
        };
        let harness = deps
            .harness_factory
            .create(&leader_effective, &context)
            .await?;
        let runtime: Arc<dyn MemberRuntime> = Arc::new(wire_runtime(
            HarnessMemberRuntime::new(harness, member_session),
            &card_id,
            &checkpoint,
            deps.checkpoints.as_ref(),
            &workspace.root,
        ));
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
        let sink_team = team_name.clone();
        messager.subscribe(
            &leader_name,
            &[TeamTopic::Team, TeamTopic::Task, TeamTopic::Message],
            Arc::new(move |event| {
                // The receiver lives on the handle between runs, so a closed channel only means the
                // handle itself is gone (or a run panicked while holding the lease). Never silent:
                // a blind leader is exactly the state this subscription exists to prevent.
                if let Err(error) = tx.send(event) {
                    warn!(
                        team = %sink_team,
                        event_type = %error.0.event_type,
                        "dropped a leader bus event: the team handle no longer holds a receiver"
                    );
                }
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
        let workspaces = BTreeMap::from([(leader_name.clone(), workspace)]);
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
            workspaces: Mutex::new(workspaces),
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

/// Hands an allocated leader workspace back after the build aborted before retaining it.
///
/// A memory failure is the only new abort point after the allocation, and a team that never runs
/// must not leave a directory (or a worktree) behind. A release failure is logged, never returned:
/// the caller is already reporting the build error.
async fn abandon_workspace(
    deps: &TeamBuildDeps,
    workspace: &crate::workspace::MemberWorkspace,
    team: &str,
) {
    if let Err(error) = deps.workspace.release(workspace).await {
        warn!(
            team,
            root = %workspace.root.display(),
            %error,
            "failed to release the leader workspace after an aborted build"
        );
    }
}

/// Resolves the effective language: the dependency override, else the blueprint's declaration.
fn resolve_language(deps: &TeamBuildDeps, declared: Option<&str>) -> String {
    let value = deps.language.as_deref().or(declared);
    Language::resolve(value).to_owned()
}

/// Resolves the team store, letting an injected store win over the blueprint's [`StorageSpec`].
///
/// Precedence, in the order the build path applies it:
///
/// 1. [`TeamBuildDeps::has_injected_store`] — the injected store is used verbatim. A spec that also
///    names a non-memory kind is logged (`warn!`) and ignored, so a host that supplied a store never
///    trips over a backend this build cannot open.
/// 2. A blank or `memory` spec kind — [`TeamBuildDeps::store`] is reused, exactly as the in-memory
///    default intends.
/// 3. Any other kind — [`open_store`] opens the named backend, so `postgresql`/`mysql`/unknown kinds
///    propagate their typed [`Error::Unsupported`].
///
/// A `sqlite` spec additionally gets the parent directory of its path-shaped target created
/// best-effort first (see [`ensure_sqlite_parent`]).
fn resolve_store(spec: &TeamAgentSpec, deps: &TeamBuildDeps) -> Result<Arc<dyn TeamStore>> {
    let kind = spec.storage.kind.trim();
    if deps.has_injected_store() {
        if !kind.is_empty() && kind != MEMORY_STORAGE_KIND {
            warn!(
                store = deps.store.name(),
                spec_kind = kind,
                "the injected team store wins over the blueprint's storage spec; the spec kind is \
                 ignored"
            );
        }
        return Ok(Arc::clone(&deps.store));
    }
    if kind.is_empty() || kind == MEMORY_STORAGE_KIND {
        return Ok(Arc::clone(&deps.store));
    }
    if kind == SQLITE_KIND {
        ensure_sqlite_parent(&spec.storage);
    }
    open_store(&spec.storage)
}

/// Creates the parent directory of a path-shaped SQLite target, best-effort.
///
/// SQLite creates the database file but not the directory holding it, so the first write of a
/// freshly configured team would fail on a nested path. Only a filesystem path is touched: an empty
/// target, an in-memory target and every `sqlite:` URL form are left to the backend. A creation
/// failure is logged and never propagated — [`open_store`] reports the real connect error.
fn ensure_sqlite_parent(spec: &StorageSpec) {
    let Some(raw) = SQLITE_TARGET_PARAMS
        .iter()
        .find_map(|key| spec.params.get(*key)?.as_str())
    else {
        return;
    };
    let target = raw.trim();
    if target.is_empty()
        || target == ":memory:"
        || target.contains("mode=memory")
        || target.starts_with("sqlite:")
    {
        return;
    }
    let Some(parent) = std::path::Path::new(target).parent() else {
        return;
    };
    if parent.as_os_str().is_empty() {
        return;
    }
    if let Err(error) = std::fs::create_dir_all(parent) {
        warn!(
            target,
            parent = %parent.display(),
            %error,
            "could not create the sqlite database directory; the store will report the connect error"
        );
    }
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
        model_ref: routing.model_ref_for(leader_name, leader_spec.model_name(), TeamRole::Leader),
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
    record.model_ref =
        routing.model_ref_for(name, requested_model(member_spec.as_ref(), blueprint), role);
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
    use crate::storage::InMemoryTeamStore;

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
        assert_eq!(
            agent.runtime().session_id(),
            format!("{}:{}", agent.session_id(), agent.member_name()),
            "the leader's runtime runs under its derived member session id"
        );
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

    #[tokio::test]
    async fn memory_spec_reuses_the_dependency_store() {
        let (agent, _factory, _workspace) = built(&blueprint("alpha", "boss", &[])).await;
        assert_eq!(agent.backend().store().name(), "memory");
    }

    #[tokio::test]
    async fn injected_store_wins_over_a_deferred_storage_backend() {
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
        let mut spec = blueprint("alpha", "boss", &[]);
        spec.storage.kind = "postgresql".to_owned();

        let agent = spec
            .build_with(deps(&factory, &workspace).with_store(Arc::clone(&store)))
            .await
            .expect("an injected store must bypass the spec's deferred backend");

        assert_eq!(agent.backend().store().name(), "memory");
        assert!(
            Arc::ptr_eq(&agent.backend().store(), &store),
            "the build must use the injected store itself"
        );
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_spec_opens_the_named_database_and_creates_its_directory() {
        let root = std::env::temp_dir().join(format!("openjiuwen-store-{}", uuid_hex()));
        let database = root.join("nested").join("team.db");
        let parent = database
            .parent()
            .expect("the database path has a parent")
            .to_path_buf();
        assert!(!parent.exists(), "the parent directory must not exist yet");

        let mut spec = blueprint("alpha", "boss", &[]);
        spec.storage.kind = "sqlite".to_owned();
        spec.storage.params.insert(
            "path".to_owned(),
            Value::String(database.to_string_lossy().into_owned()),
        );

        let agent = blueprint_build(&spec).await;
        assert_eq!(agent.backend().store().name(), "sqlite");
        assert!(
            parent.exists(),
            "the build creates the sqlite database directory"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn deferred_storage_backends_fail_with_unsupported() {
        for kind in ["postgresql", "mysql"] {
            let mut spec = blueprint("alpha", "boss", &[]);
            spec.storage.kind = kind.to_owned();
            let error = blueprint_build_error(&spec).await;
            assert_eq!(error.kind(), ErrorKind::Unsupported, "{kind}: {error}");
            assert!(error.to_string().contains(kind), "{kind}: {error}");
        }
    }

    #[tokio::test]
    async fn unknown_storage_kind_fails_with_unsupported() {
        let mut spec = blueprint("alpha", "boss", &[]);
        spec.storage.kind = "oracle".to_owned();
        let error = blueprint_build_error(&spec).await;
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
        assert!(error.to_string().contains("oracle"), "{error}");
    }

    /// Builds `spec` over fresh scripted fakes, panicking on a build error.
    #[cfg(feature = "sqlite")]
    async fn blueprint_build(spec: &TeamAgentSpec) -> Arc<TeamAgent> {
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        spec.build_with(deps(&factory, &workspace))
            .await
            .expect("the team did not build")
    }

    /// Builds `spec` over fresh scripted fakes and returns the build error.
    async fn blueprint_build_error(spec: &TeamAgentSpec) -> Error {
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        match spec.build_with(deps(&factory, &workspace)).await {
            Ok(_) => panic!("the build was expected to fail"),
            Err(error) => error,
        }
    }
}
