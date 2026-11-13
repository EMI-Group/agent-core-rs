//! Per-member checkpoint identity for a team run.
//!
//! Python parity: `openjiuwen/core/session/checkpointer/inmemory.py` keeps one
//! `AgentStorage` per team session (`_agent_stores[session.session_id()]`) and
//! keys each agent's state by `session.agent_id()`. `pre_agent_execute`
//! restores that state before a member runs, `post_agent_execute` saves it
//! afterwards, and the team's own state lives in a separate `AgentTeamStorage`
//! keyed by `session.team_id()`. Two members of one team therefore never
//! overwrite each other even though they share a single checkpointer.
//!
//! This module reproduces that identity on the Rust kernel seam: every member
//! checkpoint is persisted with `session_id = <team session id>`,
//! `namespace = agent`, and `entity_id = <member agent id>`. The member key
//! `"{team_session_id}:agent:{agent_id}"` ([`member_checkpoint_key`]) is the
//! prefix of the deterministic checkpoint id
//! `"{team_session_id}:agent:{agent_id}:{sequence:06}"`
//! ([`member_checkpoint_id`]), so a member's history is append-only and its
//! newest state is the checkpoint with the greatest
//! `(created_at, checkpoint_id)`.
//!
//! Divergences from Python (intentional):
//!
//! - Python reuses the team session id for child agent sessions, so a child
//!   session already carries the team id and its own `agent_id()`. The Rust
//!   `AgentSession` instead gets a fresh session id plus `parent_session_id`,
//!   so the member identity is built explicitly here: the team session id comes
//!   from `Session::parent_id()` and the agent id from
//!   `SessionMetadata::entity_id` (falling back to the `agent_id` metadata
//!   entry, which every `AgentSession` stamps).
//! - The Rust `Checkpointer` is storage-only (no `pre_agent_execute` /
//!   `post_agent_execute` hooks), so the caller performs the two steps
//!   explicitly: [`MemberCheckpointer::restore`] before a member invocation and
//!   [`MemberCheckpointer::save`] after it.
//! - Checkpoints are JSON documents, not pickles, and there is no process-global
//!   store: a [`MemberCheckpointer`] wraps the caller's shared
//!   `Arc<dyn Checkpointer>`.
//!
//! ```
//! use openjiuwen_multiagent::member_checkpoint::{
//!     member_checkpoint_id, member_checkpoint_key, MEMBER_CHECKPOINT_NAMESPACE,
//! };
//! use openjiuwen_session::CheckpointNamespace;
//!
//! assert_eq!(MEMBER_CHECKPOINT_NAMESPACE, CheckpointNamespace::Agent);
//! assert_eq!(member_checkpoint_key("team-1", "researcher"), "team-1:agent:researcher");
//! assert_eq!(
//!     member_checkpoint_id("team-1", "researcher", 7),
//!     "team-1:agent:researcher:000007",
//! );
//! ```
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{
    build_key_with_namespace, Checkpoint, CheckpointNamespace, Checkpointer, Session,
    SessionSnapshot,
};
use serde_json::Value;

/// The checkpoint namespace member state is stored under (`agent`).
///
/// Python parity: `AgentStorage` keeps agent state in the `agent` namespace
/// while the team's own state uses `agent-team`, so a member checkpoint never
/// shadows a team checkpoint in the same session key space.
pub const MEMBER_CHECKPOINT_NAMESPACE: CheckpointNamespace = CheckpointNamespace::Agent;

/// The metadata entry a session exposes its agent id under (`agent_id`).
///
/// [`openjiuwen_session::AgentSession`] always stamps it, so it is the fallback
/// for a session whose [`openjiuwen_session::SessionMetadata::entity_id`] is
/// empty.
const AGENT_ID_METADATA_KEY: &str = "agent_id";

/// The metadata entry recording which child session produced a member
/// checkpoint (`agent_session_id`).
const AGENT_SESSION_ID_METADATA_KEY: &str = "agent_session_id";

/// Builds the key one member's state is stored under:
/// `"{team_session_id}:agent:{agent_id}"`.
///
/// This is the Python agent key layout (`build_key_with_namespace`) and the
/// prefix of every checkpoint id [`MemberCheckpointer`] generates for the
/// member. Empty segments collapse, so an empty `agent_id` yields
/// `"{team_session_id}:agent"`; the store never calls it with an empty id.
#[must_use]
pub fn member_checkpoint_key(team_session_id: &str, agent_id: &str) -> String {
    build_key_with_namespace(
        team_session_id,
        MEMBER_CHECKPOINT_NAMESPACE.as_str(),
        agent_id,
        &[],
    )
}

/// Builds the deterministic checkpoint id of the `sequence`-th save of one
/// member: `"{team_session_id}:agent:{agent_id}:{sequence:06}"`.
///
/// The zero-padded, monotonically increasing sequence keeps a member's history
/// append-only and totally ordered by checkpoint id. [`MemberCheckpointer`]
/// takes `sequence` from a process-local `AtomicU64`, so ids are unique within
/// one store instance; separate instances (or processes) may reuse a sequence,
/// which the `created_at` stamp then disambiguates.
#[must_use]
pub fn member_checkpoint_id(team_session_id: &str, agent_id: &str, sequence: u64) -> String {
    let sequence = format!("{sequence:06}");
    build_key_with_namespace(
        team_session_id,
        MEMBER_CHECKPOINT_NAMESPACE.as_str(),
        agent_id,
        &[&sequence],
    )
}

/// A member-scoped checkpoint store over one shared [`Checkpointer`].
///
/// The store adds the member identity (`team session id`, `agent id`) to every
/// checkpoint it writes and filters on that identity when it reads, so several
/// members can share one checkpointer without overwriting each other. Saves are
/// append-only: each one gets a fresh checkpoint id from a store-local counter.
///
/// The wrapped checkpointer is the caller's; nothing is cached here, so another
/// instance over the same backend sees the same member state.
#[derive(Debug)]
pub struct MemberCheckpointer {
    checkpointer: Arc<dyn Checkpointer>,
    save_counter: AtomicU64,
}

impl MemberCheckpointer {
    /// Wraps `checkpointer` in a member-scoped view starting at sequence `0`.
    #[must_use]
    pub fn new(checkpointer: Arc<dyn Checkpointer>) -> Self {
        Self {
            checkpointer,
            save_counter: AtomicU64::new(0),
        }
    }

    /// Returns the wrapped checkpointer.
    #[must_use]
    pub fn checkpointer(&self) -> Arc<dyn Checkpointer> {
        Arc::clone(&self.checkpointer)
    }

    /// Saves `member`'s state under its own member identity.
    ///
    /// The team session id comes from [`Session::parent_id`] and the agent id
    /// from [`Session::metadata`] (`entity_id`, falling back to the `agent_id`
    /// metadata entry). Returns `Ok(None)` when the member has no team parent,
    /// because there is nothing to scope its state to.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// member has no resolvable agent id, and propagates
    /// [`Session::snapshot`] and [`Checkpointer::save`] failures unchanged.
    pub async fn save(&self, member: &dyn Session) -> Result<Option<String>> {
        match member_identity(member)? {
            Some((team_session_id, agent_id)) => self
                .save_in(&team_session_id, &agent_id, member)
                .await
                .map(Some),
            None => Ok(None),
        }
    }

    /// Saves `member` under an explicit team session and agent id.
    ///
    /// The persisted checkpoint carries `session_id = team_session_id`,
    /// `namespace = agent`, and `entity_id = agent_id`, which is the per-member
    /// identity. The checkpoint metadata also records the child session id
    /// under `agent_session_id` for diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when either
    /// id is empty, and propagates [`Session::snapshot`] and
    /// [`Checkpointer::save`] failures unchanged; the returned id is the one the
    /// checkpointer reports.
    pub async fn save_in(
        &self,
        team_session_id: &str,
        agent_id: &str,
        member: &dyn Session,
    ) -> Result<String> {
        validate_identity(team_session_id, agent_id)?;
        let sequence = self.save_counter.fetch_add(1, Ordering::Relaxed);
        let snapshot = member.snapshot().await?;
        let mut metadata = snapshot.metadata;
        metadata
            .entry(AGENT_SESSION_ID_METADATA_KEY.to_string())
            .or_insert_with(|| Value::String(member.id().to_string()));
        let checkpoint = Checkpoint::new(team_session_id, MEMBER_CHECKPOINT_NAMESPACE, agent_id)
            .with_checkpoint_id(member_checkpoint_id(team_session_id, agent_id, sequence))
            .with_global_state(snapshot.global)
            .with_component_state(Value::Object(snapshot.components.into_iter().collect()))
            .with_trace(snapshot.trace)
            .with_metadata(metadata);
        self.checkpointer.save(checkpoint).await
    }

    /// Loads the newest member checkpoint for `(team_session_id, agent_id)`.
    ///
    /// Candidate checkpoints are the `agent` checkpoints of `team_session_id`
    /// whose `entity_id` is `agent_id`; the newest is the one with the greatest
    /// `(created_at, checkpoint_id)`, mirroring [`Checkpointer::latest`].
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when either
    /// id is empty, and propagates [`Checkpointer::list`] and
    /// [`Checkpointer::load`] failures unchanged.
    pub async fn load(&self, team_session_id: &str, agent_id: &str) -> Result<Option<Checkpoint>> {
        validate_identity(team_session_id, agent_id)?;
        let Some(checkpoint_id) =
            newest_member_checkpoint_id(&self.checkpointer.list(team_session_id).await?, agent_id)
        else {
            return Ok(None);
        };
        self.checkpointer
            .load(team_session_id, &checkpoint_id)
            .await
    }

    /// Restores `member` from its newest member checkpoint; `true` when a
    /// checkpoint was restored.
    ///
    /// The identity is resolved like [`MemberCheckpointer::save`], so a member
    /// without a team parent restores nothing (`Ok(false)`). The checkpoint's
    /// `global_state` becomes the snapshot's `global` lane, its
    /// `component_state` object the snapshot's `components`, and its `trace` and
    /// `metadata` are carried over before [`Session::restore`] applies them.
    /// Restore is the session contract's merge, so it is additive on a member
    /// session that already holds state.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// member has no resolvable agent id or the stored component state is not a
    /// JSON object, and propagates [`Checkpointer::list`],
    /// [`Checkpointer::load`], and [`Session::restore`] failures unchanged.
    pub async fn restore(&self, member: &dyn Session) -> Result<bool> {
        let Some((team_session_id, agent_id)) = member_identity(member)? else {
            return Ok(false);
        };
        let Some(checkpoint) = self.load(&team_session_id, &agent_id).await? else {
            return Ok(false);
        };
        let snapshot = snapshot_from_checkpoint(member, &checkpoint)?;
        member.restore(snapshot).await?;
        Ok(true)
    }

    /// Saves every member of a team: one checkpoint per member, in the given
    /// order.
    ///
    /// Members are identified by [`Session::metadata`] (`entity_id`, falling
    /// back to the `agent_id` metadata entry), so two members must have
    /// distinct agent ids to get distinct identities.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// team session id or a member without an agent id, and propagates
    /// [`MemberCheckpointer::save_in`] failures unchanged (a failure aborts the
    /// remaining members).
    pub async fn save_all(
        &self,
        team_session_id: &str,
        members: &[Arc<dyn Session>],
    ) -> Result<Vec<String>> {
        validate_team_session_id(team_session_id)?;
        let mut checkpoint_ids = Vec::with_capacity(members.len());
        for member in members {
            let member = member.as_ref();
            let agent_id = agent_id_of(member)?;
            checkpoint_ids.push(self.save_in(team_session_id, &agent_id, member).await?);
        }
        Ok(checkpoint_ids)
    }

    /// Restores every member that has state, returning the agent ids that were
    /// restored in input order; members without a checkpoint are omitted.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// team session id or a member without an agent id, and propagates
    /// [`MemberCheckpointer::load`] and [`Session::restore`] failures unchanged.
    pub async fn restore_all(
        &self,
        team_session_id: &str,
        members: &[&dyn Session],
    ) -> Result<Vec<String>> {
        validate_team_session_id(team_session_id)?;
        let mut restored = Vec::with_capacity(members.len());
        for member in members {
            let member: &dyn Session = *member;
            let agent_id = agent_id_of(member)?;
            let Some(checkpoint) = self.load(team_session_id, &agent_id).await? else {
                continue;
            };
            member
                .restore(snapshot_from_checkpoint(member, &checkpoint)?)
                .await?;
            restored.push(agent_id);
        }
        Ok(restored)
    }

    /// Returns the distinct member agent ids that have state under
    /// `team_session_id`, sorted.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// team session id, and propagates [`Checkpointer::list`] failures unchanged.
    pub async fn saved_members(&self, team_session_id: &str) -> Result<Vec<String>> {
        validate_team_session_id(team_session_id)?;
        let members: BTreeSet<String> = self
            .checkpointer
            .list(team_session_id)
            .await?
            .into_iter()
            .filter(|meta| {
                meta.namespace == MEMBER_CHECKPOINT_NAMESPACE && !meta.entity_id.is_empty()
            })
            .map(|meta| meta.entity_id)
            .collect();
        Ok(members.into_iter().collect())
    }
}

/// Resolves the member identity `(team session id, agent id)` of `member`.
///
/// Returns `None` when the member has no (or an empty) parent session id, which
/// means it is not a team member and has nothing to scope its state to.
fn member_identity(member: &dyn Session) -> Result<Option<(String, String)>> {
    let Some(team_session_id) = member.parent_id().filter(|id| !id.is_empty()) else {
        return Ok(None);
    };
    let agent_id = agent_id_of(member)?;
    Ok(Some((team_session_id.to_string(), agent_id)))
}

/// Resolves the agent id of `member`, preferring `entity_id`.
fn agent_id_of(member: &dyn Session) -> Result<String> {
    let metadata = member.metadata();
    metadata
        .entity_id
        .filter(|id| !id.is_empty())
        .or_else(|| {
            metadata
                .metadata
                .get(AGENT_ID_METADATA_KEY)
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            Error::validation(format!(
                "member session '{}' has no agent id: state cannot be scoped to a member",
                member.id()
            ))
        })
}

/// Returns the newest member checkpoint id among `metas`, if any.
fn newest_member_checkpoint_id(
    metas: &[openjiuwen_session::CheckpointMeta],
    agent_id: &str,
) -> Option<String> {
    metas
        .iter()
        .filter(|meta| meta.namespace == MEMBER_CHECKPOINT_NAMESPACE && meta.entity_id == agent_id)
        .max_by(|left, right| {
            (&left.created_at, &left.checkpoint_id).cmp(&(&right.created_at, &right.checkpoint_id))
        })
        .map(|meta| meta.checkpoint_id.clone())
}

/// Builds the snapshot that restores `checkpoint` into `member`.
fn snapshot_from_checkpoint(
    member: &dyn Session,
    checkpoint: &Checkpoint,
) -> Result<SessionSnapshot> {
    let components: BTreeMap<String, Value> = match &checkpoint.component_state {
        Value::Null => BTreeMap::new(),
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        _ => {
            return Err(Error::validation(format!(
                "member checkpoint '{}' has a non-object component state",
                checkpoint.checkpoint_id
            )))
        }
    };
    Ok(SessionSnapshot {
        session_id: member.id().to_string(),
        kind: member.session_type(),
        global: checkpoint.global_state.clone(),
        components,
        trace: checkpoint.trace.clone(),
        metadata: checkpoint.metadata.clone(),
    })
}

/// Validates the two ids that make up a member identity.
fn validate_identity(team_session_id: &str, agent_id: &str) -> Result<()> {
    validate_team_session_id(team_session_id)?;
    if agent_id.is_empty() {
        return Err(Error::validation("member agent id must not be empty"));
    }
    Ok(())
}

/// Validates the team session id of a member identity.
fn validate_team_session_id(team_session_id: &str) -> Result<()> {
    if team_session_id.is_empty() {
        return Err(Error::validation("team session id must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, ErrorKind, Result};
    use openjiuwen_session::{
        AgentSession, AgentTeamSession, AgentTeamSessionOptions, Checkpoint, CheckpointMeta,
        CheckpointNamespace, Checkpointer, InteractionRequest, InteractionResponse,
        MemoryCheckpointer, NoopTraceSink, Session, SessionMetadata, SessionType, StreamItem,
        TraceSink,
    };
    use serde_json::{json, Value};

    use super::{
        member_checkpoint_id, member_checkpoint_key, MemberCheckpointer, AGENT_ID_METADATA_KEY,
        MEMBER_CHECKPOINT_NAMESPACE,
    };

    /// A team session id shared by the member fixtures.
    const TEAM_SESSION: &str = "team-session";

    /// A checkpointer every member shares, like a `TeamRuntime` would.
    fn shared_store() -> (MemoryCheckpointer, MemberCheckpointer) {
        let store = MemoryCheckpointer::new();
        let checkpointer = MemberCheckpointer::new(Arc::new(store.clone()));
        (store, checkpointer)
    }

    /// Creates a team session with a fixed id and the shared checkpointer.
    fn team_session(store: &MemoryCheckpointer) -> Arc<AgentTeamSession> {
        Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
            session_id: Some(TEAM_SESSION.to_string()),
            team_id: Some("crew".to_string()),
            checkpointer: Some(Arc::new(store.clone())),
            ..AgentTeamSessionOptions::default()
        }))
    }

    /// A session whose metadata is fully caller-controlled, for identity tests.
    struct MetaSession {
        id: String,
        parent: Option<String>,
        metadata: SessionMetadata,
        sink: NoopTraceSink,
    }

    impl MetaSession {
        fn new(
            id: &str,
            parent: Option<&str>,
            entity_id: Option<&str>,
            agent_id: Option<&str>,
        ) -> Self {
            let mut metadata = SessionMetadata::new(id, SessionType::Agent);
            metadata.parent_session_id = parent.map(str::to_string);
            metadata.entity_id = entity_id.map(str::to_string);
            if let Some(agent_id) = agent_id {
                metadata
                    .metadata
                    .insert(AGENT_ID_METADATA_KEY.to_string(), json!(agent_id));
            }
            Self {
                id: id.to_string(),
                parent: parent.map(str::to_string),
                metadata,
                sink: NoopTraceSink::new(),
            }
        }
    }

    #[async_trait]
    impl Session for MetaSession {
        fn id(&self) -> &str {
            &self.id
        }

        fn parent_id(&self) -> Option<&str> {
            self.parent.as_deref()
        }

        fn session_type(&self) -> SessionType {
            SessionType::Agent
        }

        async fn get_state(&self, _key: &str) -> Result<Option<Value>> {
            Ok(None)
        }

        async fn update_state(&self, _patch: Value) -> Result<()> {
            Ok(())
        }

        async fn get_component_state(&self, _component_id: &str) -> Result<Option<Value>> {
            Ok(None)
        }

        async fn update_component_state(&self, _component_id: &str, _patch: Value) -> Result<()> {
            Ok(())
        }

        async fn write_stream(&self, _item: StreamItem) -> Result<()> {
            Ok(())
        }

        async fn request_interaction(
            &self,
            _request: InteractionRequest,
        ) -> Result<InteractionResponse> {
            Err(Error::unsupported("no interaction broker"))
        }

        fn trace(&self) -> &dyn TraceSink {
            &self.sink
        }

        fn metadata(&self) -> SessionMetadata {
            self.metadata.clone()
        }
    }

    /// A checkpointer whose every read and write fails.
    #[derive(Debug)]
    struct FailingCheckpointer;

    #[async_trait]
    impl Checkpointer for FailingCheckpointer {
        async fn save(&self, _checkpoint: Checkpoint) -> Result<String> {
            Err(Error::session("backend unavailable"))
        }

        async fn load(
            &self,
            _session_id: &str,
            _checkpoint_id: &str,
        ) -> Result<Option<Checkpoint>> {
            Err(Error::session("backend unavailable"))
        }

        async fn list(&self, _session_id: &str) -> Result<Vec<CheckpointMeta>> {
            Err(Error::session("backend unavailable"))
        }

        async fn delete(&self, _session_id: &str, _checkpoint_id: &str) -> Result<()> {
            Ok(())
        }

        async fn release(&self, _session_id: &str) -> Result<()> {
            Ok(())
        }

        async fn session_exists(&self, _session_id: &str) -> Result<bool> {
            Ok(false)
        }
    }

    #[test]
    fn namespace_is_the_agent_namespace() {
        assert_eq!(MEMBER_CHECKPOINT_NAMESPACE, CheckpointNamespace::Agent);
        assert_eq!(MEMBER_CHECKPOINT_NAMESPACE.as_str(), "agent");
    }

    #[test]
    fn key_uses_the_python_agent_layout() {
        assert_eq!(
            member_checkpoint_key("team-session", "researcher"),
            "team-session:agent:researcher"
        );
    }

    #[test]
    fn ids_are_ordered_and_zero_padded() {
        assert_eq!(
            member_checkpoint_id("team-session", "researcher", 0),
            "team-session:agent:researcher:000000"
        );
        assert_eq!(
            member_checkpoint_id("team-session", "researcher", 41),
            "team-session:agent:researcher:000041"
        );
        assert!(member_checkpoint_id("s", "a", 41) > member_checkpoint_id("s", "a", 40));
        assert!(member_checkpoint_id("s", "a", 0).starts_with(&member_checkpoint_key("s", "a")));
    }

    #[tokio::test]
    async fn save_without_a_team_parent_returns_none() {
        let (store, checkpointer) = shared_store();
        let solo = AgentSession::new("solo");
        assert_eq!(checkpointer.save(&solo).await.unwrap(), None);
        assert!(!checkpointer.restore(&solo).await.unwrap());
        assert_eq!(store.len(TEAM_SESSION), 0);
    }

    #[tokio::test]
    async fn save_in_rejects_an_empty_agent_id() {
        let (_store, checkpointer) = shared_store();
        let team = team_session(&MemoryCheckpointer::new());
        let member = team.create_agent_session("researcher");
        let error = checkpointer
            .save_in(TEAM_SESSION, "", &member)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("agent id"));
        let error = checkpointer
            .save_in("", "researcher", &member)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("team session id"));
    }

    #[tokio::test]
    async fn checkpoint_carries_the_member_identity() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let member = team.create_agent_session("researcher");
        member.update_state(json!({ "step": 1 })).await.unwrap();

        let checkpoint_id = checkpointer.save(&member).await.unwrap().unwrap();
        assert!(checkpoint_id.starts_with(&member_checkpoint_key(TEAM_SESSION, "researcher")));

        let saved = store
            .load(TEAM_SESSION, &checkpoint_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.session_id, TEAM_SESSION);
        assert_eq!(saved.namespace, CheckpointNamespace::Agent);
        assert_eq!(saved.entity_id, "researcher");
        assert_eq!(saved.global_state, json!({ "step": 1 }));
        assert_eq!(saved.metadata["agent_session_id"], json!(member.id()));
    }

    #[tokio::test]
    async fn consecutive_saves_get_distinct_sequential_ids() {
        let (_store, checkpointer) = shared_store();
        let team = team_session(&MemoryCheckpointer::new());
        let member = team.create_agent_session("researcher");

        let first = checkpointer.save(&member).await.unwrap().unwrap();
        let second = checkpointer.save(&member).await.unwrap().unwrap();
        assert_eq!(first, member_checkpoint_id(TEAM_SESSION, "researcher", 0));
        assert_eq!(second, member_checkpoint_id(TEAM_SESSION, "researcher", 1));
        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn load_picks_the_newest_save_of_one_member() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let member = team.create_agent_session("researcher");
        member.update_state(json!({ "step": 1 })).await.unwrap();
        let older = checkpointer.save(&member).await.unwrap().unwrap();
        member.update_state(json!({ "step": 2 })).await.unwrap();
        let newer = checkpointer.save(&member).await.unwrap().unwrap();

        let loaded = checkpointer
            .load(TEAM_SESSION, "researcher")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.checkpoint_id, newer);
        assert_ne!(loaded.checkpoint_id, older);
        // The member's own state merges forward, so the newest save holds it.
        assert_eq!(loaded.global_state, json!({ "step": 2 }));
    }

    #[tokio::test]
    async fn restore_writes_global_and_component_state() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let member = team.create_agent_session("researcher");
        member.update_state(json!({ "step": 3 })).await.unwrap();
        member
            .update_component_state("memory", json!({ "hits": 2 }))
            .await
            .unwrap();
        checkpointer.save(&member).await.unwrap().unwrap();

        let fresh_team = team_session(&store);
        let fresh = fresh_team.create_agent_session("researcher");
        assert!(checkpointer.restore(&fresh).await.unwrap());
        assert_eq!(fresh.get_state("step").await.unwrap(), Some(json!(3)));
        assert_eq!(
            fresh.get_component_state("memory").await.unwrap(),
            Some(json!({ "hits": 2 }))
        );
    }

    #[tokio::test]
    async fn restore_of_an_unknown_member_returns_false() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let ghost = team.create_agent_session("ghost");
        assert!(!checkpointer.restore(&ghost).await.unwrap());
    }

    #[tokio::test]
    async fn restore_rejects_non_object_component_state() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let broken = Checkpoint::new(TEAM_SESSION, CheckpointNamespace::Agent, "researcher")
            .with_component_state(json!(7));
        store.save(broken).await.unwrap();

        let fresh = team.create_agent_session("researcher");
        let error = checkpointer.restore(&fresh).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("non-object component state"));
    }

    #[tokio::test]
    async fn saved_members_is_sorted_and_de_duplicated() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let writer = team.create_agent_session("writer");
        let researcher = team.create_agent_session("researcher");
        checkpointer.save(&writer).await.unwrap();
        checkpointer.save(&researcher).await.unwrap();
        checkpointer.save(&researcher).await.unwrap();
        // A team-namespace checkpoint is not a member checkpoint.
        store
            .save(Checkpoint::new(
                TEAM_SESSION,
                CheckpointNamespace::AgentTeam,
                "crew",
            ))
            .await
            .unwrap();

        assert_eq!(
            checkpointer.saved_members(TEAM_SESSION).await.unwrap(),
            vec!["researcher".to_string(), "writer".to_string()]
        );
        assert_eq!(
            checkpointer.saved_members("unknown").await.unwrap(),
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn save_and_restore_all_cover_every_member() {
        let (store, checkpointer) = shared_store();
        let team = team_session(&store);
        let researcher = team.create_agent_session("researcher");
        researcher
            .update_state(json!({ "notes": "r" }))
            .await
            .unwrap();
        let writer = team.create_agent_session("writer");
        writer.update_state(json!({ "draft": "w" })).await.unwrap();
        let members: Vec<Arc<dyn Session>> = vec![Arc::new(researcher), Arc::new(writer)];

        let ids = checkpointer.save_all(TEAM_SESSION, &members).await.unwrap();
        assert_eq!(ids.len(), 2);

        let fresh_team = team_session(&store);
        let fresh_researcher = fresh_team.create_agent_session("researcher");
        let fresh_writer = fresh_team.create_agent_session("writer");
        let restored = checkpointer
            .restore_all(TEAM_SESSION, &[&fresh_researcher, &fresh_writer])
            .await
            .unwrap();
        assert_eq!(
            restored,
            vec!["researcher".to_string(), "writer".to_string()]
        );
        assert_eq!(
            fresh_researcher.get_state("notes").await.unwrap(),
            Some(json!("r"))
        );
        assert_eq!(
            fresh_writer.get_state("draft").await.unwrap(),
            Some(json!("w"))
        );
    }

    #[tokio::test]
    async fn agent_id_falls_back_to_the_metadata_entry() {
        let (store, checkpointer) = shared_store();
        let member = MetaSession::new("child-1", Some(TEAM_SESSION), None, Some("researcher"));
        let checkpoint_id = checkpointer.save(&member).await.unwrap().unwrap();
        let saved = store
            .load(TEAM_SESSION, &checkpoint_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.entity_id, "researcher");
    }

    #[tokio::test]
    async fn a_member_without_an_agent_id_is_a_validation_error() {
        let (_store, checkpointer) = shared_store();
        let member = MetaSession::new("child-1", Some(TEAM_SESSION), None, None);
        let error = checkpointer.save(&member).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("no agent id"));
    }

    #[tokio::test]
    async fn checkpointer_errors_propagate_unchanged() {
        let checkpointer = MemberCheckpointer::new(Arc::new(FailingCheckpointer));
        let team = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
            session_id: Some(TEAM_SESSION.to_string()),
            checkpointer: Some(Arc::new(FailingCheckpointer)),
            ..AgentTeamSessionOptions::default()
        }));
        let member = team.create_agent_session("researcher");

        let error = checkpointer.save(&member).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Session);
        assert_eq!(error.message(), "backend unavailable");
        assert!(checkpointer.load(TEAM_SESSION, "researcher").await.is_err());
    }
}
