//! `AgentSession` — the per-agent execution context.
//!
//! Ports `openjiuwen/core/session/internal/agent.py` (v0.1.20): the session
//! owns an [`AgentState`] store (a `global_state` lane plus an `agent_state`
//! lane), carries the agent id/name metadata, and shares its config, stream
//! hub, interaction broker, tracer, checkpointer, and trace id with the
//! workflow and node sessions it creates.
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

use super::shared::{resolve_shared, SessionInner, SessionInnerOptions};
use super::workflow::{WorkflowSession, WorkflowSessionOptions};
use crate::checkpoint::Checkpointer;
use crate::config::SessionConfig;
use crate::interaction::{InteractionRequest, InteractionResponse};
use crate::session::{Session, SessionMetadata, SessionSnapshot, SessionType};
use crate::state::{AgentState, StateStore};
use crate::stream::{StreamEvent, StreamItem, StreamWriterManager};
use crate::tracer::{TraceSink, Tracer};

/// Process-local construction options for an [`AgentSession`].
///
/// Holds trait objects, so deliberately not `serde`-serializable. Every field
/// is optional; the defaults mirror Python's `agent.Session(session_id)`.
#[derive(Clone)]
pub struct AgentSessionOptions {
    /// Session identifier; generated with [`uuid_v4`] when absent or empty.
    pub session_id: Option<String>,
    /// Agent identifier used for entity metadata; falls back to the session id.
    pub agent_id: Option<String>,
    /// Human-readable agent name carried in the metadata block.
    pub agent_name: Option<String>,
    /// Agent description carried in the metadata block.
    pub description: Option<String>,
    /// Environment entries applied to a freshly created config.
    pub envs: BTreeMap<String, Value>,
    /// Parent session id; the team passes its own id for child agent sessions.
    pub parent_session_id: Option<String>,
    /// Checkpointer backing [`Session::save_checkpoint`] and [`Session::resume`].
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Bounded stream capacity when this session creates its own hub.
    pub stream_capacity: Option<usize>,
    /// Trace id; generated when absent and no parent supplies one.
    pub trace_id: Option<String>,
    /// Opaque metadata copied into [`SessionMetadata::metadata`].
    pub metadata: BTreeMap<String, Value>,
    /// Whether `post_run` closes the session stream before committing (Python
    /// `close_stream_on_post_run`, default `true`).
    pub close_stream_on_post_run: bool,
}

impl Default for AgentSessionOptions {
    fn default() -> Self {
        Self {
            session_id: None,
            agent_id: None,
            agent_name: None,
            description: None,
            envs: BTreeMap::new(),
            parent_session_id: None,
            checkpointer: None,
            stream_capacity: None,
            trace_id: None,
            metadata: BTreeMap::new(),
            close_stream_on_post_run: true,
        }
    }
}

impl std::fmt::Debug for AgentSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSessionOptions")
            .field("session_id", &self.session_id)
            .field("agent_id", &self.agent_id)
            .field("agent_name", &self.agent_name)
            .field("description", &self.description)
            .field("envs", &self.envs)
            .field("parent_session_id", &self.parent_session_id)
            .field("has_checkpointer", &self.checkpointer.is_some())
            .field("stream_capacity", &self.stream_capacity)
            .field("trace_id", &self.trace_id)
            .field("metadata", &self.metadata)
            .field("close_stream_on_post_run", &self.close_stream_on_post_run)
            .finish()
    }
}

/// A single agent invocation.
///
/// Owns an [`AgentState`] store and delegates everything else to the
/// crate-private `SessionInner`. A session created by a team shares that
/// team's config, stream hub, checkpointer, and trace id, but keeps its own
/// session id and state.
#[derive(Clone)]
pub struct AgentSession {
    inner: Arc<SessionInner>,
    state: Arc<AgentState>,
    agent_id: String,
    agent_name: Option<String>,
    description: Option<String>,
    close_stream_on_post_run: bool,
    post_run_done: Arc<AtomicBool>,
}

impl std::fmt::Debug for AgentSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSession")
            .field("session_id", &self.inner.id)
            .field("agent_id", &self.agent_id)
            .finish_non_exhaustive()
    }
}

impl AgentSession {
    /// Creates a session with `session_id`; an empty id generates a new uuid.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_options(AgentSessionOptions {
            session_id: Some(session_id.into()),
            ..AgentSessionOptions::default()
        })
    }

    /// Creates a session from `options`.
    #[must_use]
    pub fn with_options(options: AgentSessionOptions) -> Self {
        Self::with_parent(options, None)
    }

    /// Builds the session, inheriting config, stream hub, checkpointer, and
    /// trace id from `parent` when one is given.
    ///
    /// [`AgentSession::with_options`] passes `None`; the team session passes
    /// itself so a child agent shares the team stream and configuration.
    pub(crate) fn with_parent(
        options: AgentSessionOptions,
        parent: Option<Arc<dyn Session>>,
    ) -> Self {
        let AgentSessionOptions {
            session_id,
            agent_id,
            agent_name,
            description,
            envs,
            parent_session_id,
            checkpointer,
            stream_capacity,
            trace_id,
            mut metadata,
            close_stream_on_post_run,
        } = options;

        let resolved_session_id = session_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(uuid_v4);
        let parent_id = parent_session_id.filter(|id| !id.is_empty());
        let agent_id = agent_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| resolved_session_id.clone());
        let state = Arc::new(AgentState::new());
        let shared = resolve_shared(
            parent.as_ref(),
            checkpointer,
            trace_id,
            stream_capacity,
            &envs,
        );

        // Caller-supplied metadata wins over the derived entries, matching the
        // workflow session's `entry(...).or_insert(...)` policy.
        metadata
            .entry("agent_id".to_string())
            .or_insert_with(|| Value::String(agent_id.clone()));
        if let Some(name) = &agent_name {
            metadata
                .entry("agent_name".to_string())
                .or_insert_with(|| Value::String(name.clone()));
        }
        if let Some(description) = &description {
            metadata
                .entry("description".to_string())
                .or_insert_with(|| Value::String(description.clone()));
        }
        if let Some(parent_id) = &parent_id {
            metadata
                .entry("parent_session_id".to_string())
                .or_insert_with(|| Value::String(parent_id.clone()));
        }

        let inner = SessionInner::new(SessionInnerOptions {
            id: resolved_session_id,
            parent_id,
            kind: SessionType::Agent,
            entity_id: Some(agent_id.clone()),
            entity_name: agent_name.clone(),
            trace_id: shared.trace_id,
            created_at: None,
            config: shared.config,
            state: Arc::clone(&state) as Arc<dyn StateStore>,
            writers: Some(shared.writers),
            stream_capacity,
            sink: None,
            checkpointer: shared.checkpointer,
            parent,
            metadata,
        });

        Self {
            inner,
            state,
            agent_id,
            agent_name,
            description,
            close_stream_on_post_run,
            post_run_done: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns the agent identifier.
    #[must_use]
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Returns the agent name, when one was supplied.
    #[must_use]
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// Returns the agent description, when one was supplied.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
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

    /// Returns the agent state store.
    #[must_use]
    pub fn state(&self) -> Arc<AgentState> {
        Arc::clone(&self.state)
    }

    /// Returns every committed lane of the agent state store
    /// (`{"global_state": …, "agent_state": …}`; the trace lane is not a key).
    #[must_use]
    pub fn dump_state(&self) -> Value {
        self.state.dump()
    }

    /// Creates a workflow session that shares this agent session's id, config,
    /// stream hub, checkpointer, and trace id.
    ///
    /// The child owns an independent [`crate::state::WorkflowState`], matching
    /// Python's production path where the workflow session has its own state
    /// store.
    #[must_use]
    pub fn create_workflow_session(self: &Arc<Self>) -> WorkflowSession {
        WorkflowSession::with_options(WorkflowSessionOptions {
            session_id: Some(self.inner.id.clone()),
            parent: Some(Arc::clone(self) as Arc<dyn Session>),
            trace_id: Some(self.inner.trace_id.clone()),
            ..WorkflowSessionOptions::default()
        })
    }

    /// Saves a checkpoint and returns its id when a checkpointer is configured.
    ///
    /// This is Python's `commit()` → `post_agent_execute` seam: a session
    /// without a checkpointer has nothing to persist, so it returns `Ok(None)`.
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

    /// Ends the run: closes the stream (when configured to) and commits.
    ///
    /// Idempotent — only the first call does any work, and every later call
    /// returns `Ok(None)`. No lock is held across the checkpoint await.
    ///
    /// # Errors
    ///
    /// Propagates [`AgentSession::commit`] failures.
    pub async fn post_run(&self) -> Result<Option<String>> {
        if self.post_run_done.swap(true, Ordering::SeqCst) {
            return Ok(None);
        }
        if self.close_stream_on_post_run {
            self.inner.end_stream();
        }
        self.commit().await
    }
}

#[async_trait]
impl Session for AgentSession {
    fn id(&self) -> &str {
        &self.inner.id
    }

    fn parent_id(&self) -> Option<&str> {
        self.inner.parent_id.as_deref()
    }

    fn session_type(&self) -> SessionType {
        SessionType::Agent
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

    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use serde_json::json;

    use super::{AgentSession, AgentSessionOptions};
    use crate::checkpoint::MemoryCheckpointer;
    use crate::session::{Session, SessionType};
    use crate::stream::{StreamEvent, StreamItem};

    /// Upper bound on how long a deterministic in-memory receive may take.
    const RECV_TIMEOUT: Duration = Duration::from_secs(1);

    #[tokio::test]
    async fn env_access_reads_the_shared_config() {
        let mut envs = BTreeMap::new();
        envs.insert("API_KEY".to_string(), json!("secret"));
        let session = AgentSession::with_options(AgentSessionOptions {
            session_id: Some("agent-1".to_string()),
            envs,
            ..AgentSessionOptions::default()
        });

        assert_eq!(session.env("API_KEY"), Some(json!("secret")));
        assert_eq!(session.env("missing"), None);
        assert_eq!(session.envs().get("API_KEY"), Some(&json!("secret")));
    }

    #[tokio::test]
    async fn dump_state_has_global_and_agent_lanes() {
        let session = AgentSession::new("agent-1");
        session.update_state(json!({ "step": 1 })).await.unwrap();
        session
            .update_component_state("memory", json!({ "hits": 2 }))
            .await
            .unwrap();

        let dump = session.dump_state();
        assert_eq!(dump["global_state"], json!({ "step": 1 }));
        assert_eq!(dump["agent_state"], json!({ "memory": { "hits": 2 } }));
        assert_eq!(dump.as_object().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn metadata_records_agent_identity() {
        let session = AgentSession::with_options(AgentSessionOptions {
            session_id: Some("agent-1".to_string()),
            agent_name: Some("researcher".to_string()),
            description: Some("looks things up".to_string()),
            parent_session_id: Some("team-1".to_string()),
            ..AgentSessionOptions::default()
        });

        let metadata = session.metadata();
        assert_eq!(metadata.session_id, "agent-1");
        assert_eq!(metadata.kind, SessionType::Agent);
        assert_eq!(metadata.entity_id.as_deref(), Some("agent-1"));
        assert_eq!(metadata.entity_name.as_deref(), Some("researcher"));
        assert_eq!(metadata.parent_session_id.as_deref(), Some("team-1"));
        assert_eq!(metadata.metadata["agent_id"], json!("agent-1"));
        assert_eq!(metadata.metadata["agent_name"], json!("researcher"));
        assert_eq!(metadata.metadata["description"], json!("looks things up"));
        assert_eq!(metadata.metadata["parent_session_id"], json!("team-1"));
    }

    #[tokio::test]
    async fn commit_without_checkpointer_returns_none() {
        let session = AgentSession::new("agent-1");
        assert_eq!(session.commit().await.unwrap(), None);
    }

    #[tokio::test]
    async fn commit_with_checkpointer_saves_a_checkpoint() {
        let session = AgentSession::with_options(AgentSessionOptions {
            session_id: Some("agent-1".to_string()),
            checkpointer: Some(Arc::new(MemoryCheckpointer::new())),
            ..AgentSessionOptions::default()
        });
        session.update_state(json!({ "step": 1 })).await.unwrap();

        let checkpoint_id = session.commit().await.unwrap();
        assert!(checkpoint_id.is_some());
        assert!(!session.resume("missing").await.unwrap());
    }

    #[tokio::test]
    async fn post_run_is_idempotent() {
        let session = AgentSession::with_options(AgentSessionOptions {
            session_id: Some("agent-1".to_string()),
            checkpointer: Some(Arc::new(MemoryCheckpointer::new())),
            ..AgentSessionOptions::default()
        });

        assert!(session.post_run().await.unwrap().is_some());
        assert_eq!(session.post_run().await.unwrap(), None);
        assert!(session.writers().unwrap().is_closed());
    }

    #[test]
    fn empty_session_id_generates_a_uuid() {
        let session = AgentSession::new("");
        assert_ne!(session.id(), "");
        assert_eq!(session.session_type(), SessionType::Agent);
    }

    #[tokio::test]
    async fn workflow_child_shares_the_agent_hub() {
        let agent = Arc::new(AgentSession::new("agent-1"));
        let mut events = agent.subscribe();
        let child = agent.create_workflow_session();
        assert_eq!(child.id(), "agent-1");
        assert_eq!(child.parent_id(), Some("agent-1"));

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
    async fn workflow_child_owns_independent_state() {
        let agent = Arc::new(AgentSession::new("agent-1"));
        let child = agent.create_workflow_session();

        child
            .update_state(json!({ "only_child": true }))
            .await
            .unwrap();
        assert_eq!(
            child.get_state("only_child").await.unwrap(),
            Some(json!(true))
        );
        assert_eq!(agent.get_state("only_child").await.unwrap(), None);
    }
}
