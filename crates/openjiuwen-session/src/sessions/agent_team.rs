//! `AgentTeamSession` — the per-team execution context.
//!
//! Ports `openjiuwen/core/session/internal/agent_team.py` (v0.1.20): the team
//! session owns an [`AgentState`] store and the team id, and creates child
//! [`AgentSession`]s that share its config, stream hub, checkpointer, and trace
//! id. Children are stamped with `source_agent_id`/`source_team_id` when source
//! metadata is enabled (Python `source_metadata_enabled`).
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use parking_lot::RwLock;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_common::utils::id::uuid_v4;

use super::agent::{AgentSession, AgentSessionOptions};
use super::shared::{resolve_shared, SessionInner, SessionInnerOptions};
use crate::checkpoint::Checkpointer;
use crate::config::SessionConfig;
use crate::interaction::{InteractionRequest, InteractionResponse};
use crate::session::{Session, SessionMetadata, SessionSnapshot, SessionType};
use crate::state::{AgentState, StateStore};
use crate::stream::{StreamEvent, StreamItem, StreamWriterManager};
use crate::tracer::{TraceSink, Tracer};

/// Default team id (`openjiuwen/core/session/agent_team.py`).
pub const DEFAULT_TEAM_ID: &str = "agent_team";

/// Process-local construction options for an [`AgentTeamSession`].
///
/// Holds trait objects, so deliberately not `serde`-serializable. Every field
/// is optional; the defaults mirror Python's `agent_team.Session(session_id)`.
#[derive(Clone)]
pub struct AgentTeamSessionOptions {
    /// Session identifier; generated with [`uuid_v4`] when absent or empty.
    pub session_id: Option<String>,
    /// Team identifier; defaults to [`DEFAULT_TEAM_ID`].
    pub team_id: Option<String>,
    /// Environment entries applied to a freshly created config.
    pub envs: BTreeMap<String, Value>,
    /// Checkpointer backing [`Session::save_checkpoint`] and [`Session::resume`].
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Bounded stream capacity when this session creates its own hub.
    pub stream_capacity: Option<usize>,
    /// Trace id; generated when absent.
    pub trace_id: Option<String>,
    /// Opaque metadata copied into [`SessionMetadata::metadata`].
    pub metadata: BTreeMap<String, Value>,
    /// Python `source_metadata_enabled` (default `true`): stamp
    /// `source_agent_id`/`source_team_id` on child agent sessions.
    pub source_metadata_enabled: bool,
}

impl Default for AgentTeamSessionOptions {
    fn default() -> Self {
        Self {
            session_id: None,
            team_id: None,
            envs: BTreeMap::new(),
            checkpointer: None,
            stream_capacity: None,
            trace_id: None,
            metadata: BTreeMap::new(),
            source_metadata_enabled: true,
        }
    }
}

impl std::fmt::Debug for AgentTeamSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTeamSessionOptions")
            .field("session_id", &self.session_id)
            .field("team_id", &self.team_id)
            .field("envs", &self.envs)
            .field("has_checkpointer", &self.checkpointer.is_some())
            .field("stream_capacity", &self.stream_capacity)
            .field("trace_id", &self.trace_id)
            .field("metadata", &self.metadata)
            .field("source_metadata_enabled", &self.source_metadata_enabled)
            .finish()
    }
}

/// An agent-team run.
///
/// Owns an [`AgentState`] store and the team id, and delegates everything else
/// to the crate-private `SessionInner`. Child agent sessions share the team's
/// config, stream hub, checkpointer, and trace id.
#[derive(Clone)]
pub struct AgentTeamSession {
    inner: Arc<SessionInner>,
    state: Arc<AgentState>,
    team_id: String,
    source_metadata_enabled: bool,
    post_run_done: Arc<AtomicBool>,
}

impl std::fmt::Debug for AgentTeamSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTeamSession")
            .field("session_id", &self.inner.id)
            .field("team_id", &self.team_id)
            .finish_non_exhaustive()
    }
}

impl AgentTeamSession {
    /// Creates a team session with `team_id`; an empty id uses
    /// [`DEFAULT_TEAM_ID`].
    #[must_use]
    pub fn new(team_id: impl Into<String>) -> Self {
        Self::with_options(AgentTeamSessionOptions {
            team_id: Some(team_id.into()),
            ..AgentTeamSessionOptions::default()
        })
    }

    /// Creates a team session from `options`.
    #[must_use]
    pub fn with_options(options: AgentTeamSessionOptions) -> Self {
        let AgentTeamSessionOptions {
            session_id,
            team_id,
            envs,
            checkpointer,
            stream_capacity,
            trace_id,
            mut metadata,
            source_metadata_enabled,
        } = options;

        let session_id = session_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(uuid_v4);
        let team_id = team_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| DEFAULT_TEAM_ID.to_string());
        let state = Arc::new(AgentState::new());
        let shared = resolve_shared(None, checkpointer, trace_id, stream_capacity, &envs);

        // Derived entries first, caller-supplied metadata wins.
        metadata
            .entry("team_id".to_string())
            .or_insert_with(|| Value::String(team_id.clone()));

        let inner = SessionInner::new(SessionInnerOptions {
            id: session_id,
            parent_id: None,
            kind: SessionType::AgentTeam,
            entity_id: Some(team_id.clone()),
            entity_name: Some(team_id.clone()),
            trace_id: shared.trace_id,
            created_at: None,
            config: shared.config,
            state: Arc::clone(&state) as Arc<dyn StateStore>,
            writers: Some(shared.writers),
            stream_capacity,
            sink: None,
            checkpointer: shared.checkpointer,
            parent: None,
            metadata,
        });

        Self {
            inner,
            state,
            team_id,
            source_metadata_enabled,
            post_run_done: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns the team identifier.
    #[must_use]
    pub fn team_id(&self) -> &str {
        &self.team_id
    }

    /// Returns the environment entry stored under `key`.
    #[must_use]
    pub fn env(&self, key: &str) -> Option<Value> {
        self.inner.config.read().get_env(key).cloned()
    }

    /// Returns a copy of every environment entry.
    #[must_use]
    pub fn envs(&self) -> BTreeMap<String, Value> {
        self.inner.config.read().get_envs()
    }

    /// Returns the team state store.
    #[must_use]
    pub fn state(&self) -> Arc<AgentState> {
        Arc::clone(&self.state)
    }

    /// Returns every committed lane of the team state store
    /// (`{"global_state": …, "agent_state": …}`).
    #[must_use]
    pub fn dump_state(&self) -> Value {
        self.state.dump()
    }

    /// Creates a child agent session that shares this team's config, stream
    /// hub, checkpointer, and trace id.
    ///
    /// An empty `agent_id` generates a new uuid. The child gets its own session
    /// id, `parent_session_id` set to the team session id, and a `post_run` that
    /// never closes the shared stream.
    #[must_use]
    pub fn create_agent_session(self: &Arc<Self>, agent_id: impl Into<String>) -> AgentSession {
        let agent_id = agent_id.into();
        let agent_id = if agent_id.is_empty() {
            uuid_v4()
        } else {
            agent_id
        };

        let mut metadata = BTreeMap::new();
        if self.source_metadata_enabled {
            metadata.insert(
                "source_agent_id".to_string(),
                Value::String(agent_id.clone()),
            );
            metadata.insert(
                "source_team_id".to_string(),
                Value::String(self.team_id.clone()),
            );
        }

        AgentSession::with_parent(
            AgentSessionOptions {
                session_id: None,
                agent_id: Some(agent_id),
                envs: self.envs(),
                parent_session_id: Some(self.inner.id.clone()),
                close_stream_on_post_run: false,
                metadata,
                ..AgentSessionOptions::default()
            },
            Some(Arc::clone(self) as Arc<dyn Session>),
        )
    }

    /// Saves a checkpoint and returns its id when a checkpointer is configured.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::save_checkpoint`] failures.
    pub async fn commit(&self) -> Result<Option<String>> {
        if self.inner.checkpointer.is_none() {
            return Ok(None);
        }
        let checkpoint_id = self.save_checkpoint(None).await?;
        Ok(Some(checkpoint_id))
    }

    /// Alias of [`AgentTeamSession::commit`] (Python `flush_checkpoint`).
    ///
    /// # Errors
    ///
    /// Propagates [`AgentTeamSession::commit`] failures.
    pub async fn flush_checkpoint(&self) -> Result<Option<String>> {
        self.commit().await
    }

    /// Ends the run: closes the stream and commits.
    ///
    /// Idempotent — only the first call does any work, and every later call
    /// returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Propagates [`AgentTeamSession::commit`] failures.
    pub async fn post_run(&self) -> Result<Option<String>> {
        if self.post_run_done.swap(true, Ordering::SeqCst) {
            return Ok(None);
        }
        self.inner.end_stream();
        self.commit().await
    }
}

#[async_trait]
impl Session for AgentTeamSession {
    fn id(&self) -> &str {
        &self.inner.id
    }

    fn parent_id(&self) -> Option<&str> {
        self.inner.parent_id.as_deref()
    }

    fn session_type(&self) -> SessionType {
        SessionType::AgentTeam
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        Ok(self.inner.get_state(key))
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        self.inner.update_state(patch)
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        Ok(self.inner.get_component_state(component_id))
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        self.inner.update_component_state(component_id, patch)
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.inner.write_stream(item).await
    }

    async fn request_interaction(
        &self,
        request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        self.inner.request_interaction(request).await
    }

    fn trace(&self) -> &dyn TraceSink {
        self.inner.trace_sink.as_ref()
    }

    fn metadata(&self) -> SessionMetadata {
        self.inner.metadata()
    }

    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.inner.checkpointer.clone()
    }

    fn tracer(&self) -> Option<&Tracer> {
        Some(&self.inner.tracer)
    }

    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> {
        Some(self.inner.config_handle())
    }

    fn writers(&self) -> Option<Arc<StreamWriterManager>> {
        Some(Arc::clone(&self.inner.writers))
    }

    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        self.inner.subscribe()
    }

    async fn resolve_interaction(&self, response: InteractionResponse) -> Result<()> {
        self.inner.resolve_interaction(response)
    }

    async fn end_stream(&self) -> Result<()> {
        self.inner.end_stream();
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.inner.close();
        Ok(())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        Ok(self.inner.snapshot())
    }

    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()> {
        self.inner.restore(snapshot)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use serde_json::json;

    use super::{AgentTeamSession, AgentTeamSessionOptions, DEFAULT_TEAM_ID};
    use crate::checkpoint::MemoryCheckpointer;
    use crate::session::{Session, SessionType};
    use crate::stream::{StreamEvent, StreamItem};

    /// Upper bound on how long a deterministic in-memory receive may take.
    const RECV_TIMEOUT: Duration = Duration::from_secs(1);

    #[tokio::test]
    async fn default_team_id_is_agent_team() {
        let team = AgentTeamSession::new("");
        assert_eq!(team.team_id(), DEFAULT_TEAM_ID);
        assert_eq!(team.team_id(), "agent_team");
        assert_eq!(team.session_type(), SessionType::AgentTeam);
        assert_ne!(team.id(), "");
    }

    #[tokio::test]
    async fn explicit_team_id_and_metadata_are_kept() {
        let team = AgentTeamSession::with_options(AgentTeamSessionOptions {
            session_id: Some("team-session".to_string()),
            team_id: Some("crew".to_string()),
            ..AgentTeamSessionOptions::default()
        });
        assert_eq!(team.team_id(), "crew");
        let metadata = team.metadata();
        assert_eq!(metadata.session_id, "team-session");
        assert_eq!(metadata.entity_id.as_deref(), Some("crew"));
        assert_eq!(metadata.entity_name.as_deref(), Some("crew"));
        assert_eq!(metadata.metadata["team_id"], json!("crew"));
    }

    #[tokio::test]
    async fn dump_state_has_global_and_agent_lanes() {
        let team = AgentTeamSession::new("crew");
        team.update_state(json!({ "round": 1 })).await.unwrap();
        team.update_component_state("planner", json!({ "done": true }))
            .await
            .unwrap();

        let dump = team.dump_state();
        assert_eq!(dump["global_state"], json!({ "round": 1 }));
        assert_eq!(dump["agent_state"], json!({ "planner": { "done": true } }));
    }

    #[tokio::test]
    async fn child_agent_carries_source_metadata() {
        let team = Arc::new(AgentTeamSession::new("crew"));
        let child = team.create_agent_session("researcher");

        let metadata = child.metadata();
        assert_eq!(metadata.kind, SessionType::Agent);
        assert_eq!(metadata.parent_session_id.as_deref(), Some(team.id()));
        assert_ne!(metadata.session_id, team.id());
        assert_eq!(metadata.entity_id.as_deref(), Some("researcher"));
        assert_eq!(metadata.metadata["source_agent_id"], json!("researcher"));
        assert_eq!(metadata.metadata["source_team_id"], json!("crew"));
    }

    #[tokio::test]
    async fn disabled_source_metadata_stamps_nothing() {
        let team = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
            team_id: Some("crew".to_string()),
            source_metadata_enabled: false,
            ..AgentTeamSessionOptions::default()
        }));
        let child = team.create_agent_session("researcher");

        let metadata = child.metadata();
        assert!(!metadata.metadata.contains_key("source_agent_id"));
        assert!(!metadata.metadata.contains_key("source_team_id"));
        assert_eq!(metadata.metadata["agent_id"], json!("researcher"));
    }

    #[tokio::test]
    async fn child_agent_shares_the_team_stream() {
        let team = Arc::new(AgentTeamSession::new("crew"));
        let mut events = team.subscribe();
        let child = team.create_agent_session("researcher");

        child
            .write_stream(StreamItem::output("llm_output", None, json!({ "x": 1 })))
            .await
            .unwrap();

        let event = tokio::time::timeout(RECV_TIMEOUT, events.next())
            .await
            .expect("a frame arrives before the timeout")
            .expect("the stream stays open")
            .expect("the stream yields Ok events");
        let StreamEvent::Item(item) = event else {
            panic!("expected an output frame");
        };
        assert_eq!(item.item_type, "llm_output");
        assert_eq!(item.payload, json!({ "x": 1 }));
    }

    #[tokio::test]
    async fn team_and_child_share_the_writer_manager() {
        let team = Arc::new(AgentTeamSession::new("crew"));
        let child = team.create_agent_session("researcher");
        let team_writers = team.writers().unwrap();
        let child_writers = child.writers().unwrap();
        assert!(Arc::ptr_eq(&team_writers, &child_writers));
    }

    #[tokio::test]
    async fn child_post_run_keeps_the_shared_stream_open() {
        let team = Arc::new(AgentTeamSession::new("crew"));
        let child = team.create_agent_session("researcher");

        assert_eq!(child.post_run().await.unwrap(), None);
        assert!(!team.writers().unwrap().is_closed());
    }

    #[tokio::test]
    async fn commit_and_post_run_are_idempotent() {
        let team = AgentTeamSession::with_options(AgentTeamSessionOptions {
            team_id: Some("crew".to_string()),
            checkpointer: Some(Arc::new(MemoryCheckpointer::new())),
            ..AgentTeamSessionOptions::default()
        });

        assert!(team.flush_checkpoint().await.unwrap().is_some());
        assert!(team.post_run().await.unwrap().is_some());
        assert_eq!(team.post_run().await.unwrap(), None);
    }
}
