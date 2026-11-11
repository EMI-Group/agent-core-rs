//! The [`Session`] contract: the per-run execution context every graph node,
//! workflow component, agent, team, and runner receives as `Arc<dyn Session>`.
//!
//! Ports the state, streaming, interaction, and tracing seam of
//! `openjiuwen/core/session/session.py` (v0.1.20) without Python's ambient
//! session `ContextVar`: the session is always passed explicitly.
//!
//! # State semantics
//!
//! Every implementor must obey the same state rules, so workflows, agents, and
//! teams can share one session type:
//!
//! - [`Session::get_state`] reads the session's global lane; `key` is a
//!   dot/bracket path (`a.b[0].c`), and an empty key returns the whole global
//!   object.
//! - [`Session::update_state`] deep-merges an object patch into the global
//!   lane under one write lock: object-vs-object recurses, a `null` value
//!   deletes the path (deletes are applied after writes), and any other value
//!   replaces wholesale (arrays are never element-merged).
//! - [`Session::get_component_state`] and [`Session::update_component_state`]
//!   address the component lane by component id; ids are nested paths, so
//!   component `a.b` is stored at `{"a": {"b": ...}}` exactly like Python's
//!   `executable_id` nesting. An empty component id is a validation error.
//!
//! # Streaming and human-in-the-loop
//!
//! [`Session::write_stream`] publishes a [`StreamItem`] to the session hub
//! (`end_stream` publishes the completion marker), and
//! [`Session::request_interaction`] publishes an interaction frame and awaits a
//! [`InteractionResponse`]. Both are shared with child sessions so a consumer
//! sees one ordered stream per run.
//!
//! # Tracing and checkpoints
//!
//! [`Session::trace`] hands out the session's stable [`TraceSink`], which is
//! what [`Session::snapshot`] serializes as the trace position.
//! [`Session::save_checkpoint`] and [`Session::resume`] are provided on top of
//! the [`Checkpointer`] seam: they persist and restore global state, component
//! state, and the trace position, and are no-ops when the session has no
//! checkpointer configured.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::checkpoint::{Checkpoint, CheckpointNamespace, Checkpointer};
use crate::config::SessionConfig;
use crate::interaction::{InteractionRequest, InteractionResponse};
use crate::stream::{StreamEvent, StreamItem, StreamWriterManager};
use crate::tracer::{TracePosition, TraceSink, Tracer};

/// The kind of execution context a session represents.
///
/// Mirrors Python's four session flavours: a workflow run, an agent invocation,
/// an agent-team run, and a single graph node. The `snake_case` wire names are
/// `workflow`, `agent`, `agent_team`, and `node`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionType {
    /// A workflow (or graph) execution context; the default kind.
    #[default]
    Workflow,
    /// A single agent invocation.
    Agent,
    /// An agent-team run.
    AgentTeam,
    /// A component-scoped child view of a parent session.
    Node,
}

/// Descriptive metadata about a session, independent of its state.
///
/// The session id, parent id, and kind are always available; entity, trace, and
/// creation fields are filled in by concrete sessions when they know them.
/// Field names are the JSON contract (`snake_case`), and unknown fields are
/// tolerated so the shape can grow without breaking readers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionMetadata {
    /// The session identifier.
    pub session_id: String,
    /// The parent session's id for child sessions (`None` at the root).
    pub parent_session_id: Option<String>,
    /// The kind of execution context.
    pub kind: SessionType,
    /// The id of the entity the session runs for (agent id, workflow id, …).
    pub entity_id: Option<String>,
    /// The human-readable name of that entity.
    pub entity_name: Option<String>,
    /// The trace id shared by every frame emitted through this session.
    pub trace_id: Option<String>,
    /// Creation time as an RFC 3339 string, when known.
    pub created_at: Option<String>,
    /// Opaque additional metadata.
    pub metadata: BTreeMap<String, Value>,
}

impl SessionMetadata {
    /// Creates the default metadata for `session_id` of kind `kind`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, kind: SessionType) -> Self {
        Self {
            session_id: session_id.into(),
            kind,
            ..Self::default()
        }
    }
}

/// A point-in-time capture of a session used for checkpointing and resume.
///
/// `global` is the session's whole global lane (a JSON object), `components`
/// maps a component id to its component lane, and `trace` is the position of
/// the last emitted trace frame. Field names are the JSON contract
/// (`snake_case`), and unknown fields are tolerated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionSnapshot {
    /// The session identifier this snapshot belongs to.
    pub session_id: String,
    /// The kind of execution context, needed to restore a typed session.
    pub kind: SessionType,
    /// The session's global state lane.
    pub global: Value,
    /// Component-scoped state, keyed by component id.
    pub components: BTreeMap<String, Value>,
    /// The trace position at the time the snapshot was taken.
    pub trace: Option<TracePosition>,
    /// The session metadata captured with the snapshot.
    pub metadata: BTreeMap<String, Value>,
}

impl SessionSnapshot {
    /// Creates an empty snapshot for `session_id` of kind `kind`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, kind: SessionType) -> Self {
        Self {
            session_id: session_id.into(),
            kind,
            global: Value::Object(Map::new()),
            ..Self::default()
        }
    }
}

/// The stable per-run execution context shared by every layer above the kernel.
///
/// A session owns state lanes, the stream hub, the interaction broker, tracing,
/// and (optionally) a checkpointer. Concrete sessions ([`crate::sessions`]) add
/// their kind-specific views; higher layers must hold `Arc<dyn Session>` rather
/// than a concrete type so fakes and test doubles stay cheap.
///
/// The required methods are the contract (`docs/ARCHITECTURE.md` §2.4); the
/// provided methods below have defaults so an implementation only overrides
/// what it actually supports.
#[async_trait]
pub trait Session: Send + Sync {
    /// Returns the session identifier.
    fn id(&self) -> &str;

    /// Returns the parent session's id, or `None` at the root.
    fn parent_id(&self) -> Option<&str>;

    /// Returns the kind of execution context this session represents.
    fn session_type(&self) -> SessionType;

    /// Reads the global state lane at the dot/bracket path `key`.
    ///
    /// An empty `key` returns the whole global object, and a key that is not
    /// present returns `None`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for a
    /// malformed path and [`openjiuwen_common::error::ErrorKind::Session`] when
    /// the state store cannot be read.
    async fn get_state(&self, key: &str) -> Result<Option<Value>>;

    /// Deep-merges `patch` into the global state lane.
    ///
    /// Object-vs-object merges recurse, a `null` value deletes the path, and
    /// any other value replaces wholesale. The patch must be a JSON object.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// patch is not an object or a path is malformed, and
    /// [`openjiuwen_common::error::ErrorKind::Session`] on write failure.
    async fn update_state(&self, patch: Value) -> Result<()>;

    /// Reads the component lane of `component_id` (a nested path).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// or malformed component id and [`openjiuwen_common::error::ErrorKind::Session`]
    /// when the state store cannot be read.
    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>>;

    /// Deep-merges `patch` into the component lane of `component_id`.
    ///
    /// `component_id` is treated as a nested path, so `a.b` writes under
    /// `{"a": {"b": ...}}` exactly like Python's `executable_id` nesting.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// component id or a non-object patch, and
    /// [`openjiuwen_common::error::ErrorKind::Session`] on write failure.
    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()>;

    /// Publishes one frame to the session stream.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// frame payload is `null`, and
    /// [`openjiuwen_common::error::ErrorKind::Session`] when the stream is
    /// closed.
    async fn write_stream(&self, item: StreamItem) -> Result<()>;

    /// Publishes an interaction frame and awaits the matching response.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Timeout`] when the
    /// request times out and [`openjiuwen_common::error::ErrorKind::Cancelled`]
    /// when the run is cancelled while waiting.
    async fn request_interaction(&self, request: InteractionRequest)
        -> Result<InteractionResponse>;

    /// Returns the session's trace sink.
    ///
    /// Implementations must return a reference to a sink they own (never a
    /// temporary), so callers can hold the reference across calls.
    fn trace(&self) -> &dyn TraceSink;

    /// Returns descriptive metadata derived from the id, parent, and kind.
    ///
    /// Concrete sessions override this to add entity, trace, and creation
    /// fields.
    fn metadata(&self) -> SessionMetadata {
        SessionMetadata {
            session_id: self.id().to_string(),
            parent_session_id: self.parent_id().map(str::to_string),
            kind: self.session_type(),
            ..SessionMetadata::default()
        }
    }

    /// Returns the checkpointer backing [`Session::save_checkpoint`] and
    /// [`Session::resume`], when the session has one.
    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        None
    }

    /// Returns the span factory when the session owns one.
    fn tracer(&self) -> Option<&Tracer> {
        None
    }

    /// Returns the session configuration behind a shared lock.
    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> {
        None
    }

    /// Returns the mode-routed stream writers shared with child sessions.
    fn writers(&self) -> Option<Arc<StreamWriterManager>> {
        None
    }

    /// Subscribes to the session stream.
    ///
    /// The default implementation yields an immediately empty stream, which is
    /// what a session without a stream hub should behave like.
    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        futures::stream::empty().boxed()
    }

    /// Delivers a response to a pending interaction request.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Unsupported`] by default,
    /// and [`openjiuwen_common::error::ErrorKind::NotFound`] from sessions that
    /// accept responses but have no matching pending request.
    async fn resolve_interaction(&self, _response: InteractionResponse) -> Result<()> {
        Err(Error::unsupported(
            "session does not accept interaction responses",
        ))
    }

    /// Publishes the stream completion marker; idempotent.
    ///
    /// # Errors
    ///
    /// Propagates stream failures; completing an already-closed stream is not
    /// an error.
    async fn end_stream(&self) -> Result<()> {
        Ok(())
    }

    /// Releases the session's resources; idempotent.
    ///
    /// # Errors
    ///
    /// Propagates cleanup failures from the concrete session.
    async fn close(&self) -> Result<()> {
        Ok(())
    }

    /// Captures global state, component state, the trace position, and
    /// metadata.
    ///
    /// A missing global lane is captured as an empty object. The default
    /// implementation does not include component state; concrete sessions
    /// override it to enumerate their component lanes.
    ///
    /// # Errors
    ///
    /// Propagates state-read failures from [`Session::get_state`].
    async fn snapshot(&self) -> Result<SessionSnapshot> {
        let global = self
            .get_state("")
            .await?
            .unwrap_or_else(|| Value::Object(Map::new()));
        Ok(SessionSnapshot {
            session_id: self.id().to_string(),
            kind: self.session_type(),
            global,
            components: BTreeMap::new(),
            trace: self.trace().position(),
            metadata: self.metadata().metadata,
        })
    }

    /// Restores global state, component state, and the trace position from
    /// `snapshot`.
    ///
    /// A `global` lane that is not a JSON object is ignored (there is no
    /// sensible merged restore for a scalar), while a non-empty `components`
    /// map is applied entry by entry.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::update_state`] and
    /// [`Session::update_component_state`] failures.
    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()> {
        let SessionSnapshot {
            global,
            components,
            trace,
            ..
        } = snapshot;
        if global.is_object() {
            self.update_state(global).await?;
        }
        for (component_id, patch) in components {
            self.update_component_state(&component_id, patch).await?;
        }
        if let Some(position) = trace {
            self.trace().set_position(position);
        }
        Ok(())
    }

    /// Saves a checkpoint of this session and returns its checkpoint id.
    ///
    /// The namespace is derived from the session kind (`agent`, `agent-team`,
    /// or `workflow`), and the entity id is the session metadata's `entity_id`
    /// when present, else the session id.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the
    /// session has no checkpointer configured, and propagates
    /// [`Session::snapshot`] and [`Checkpointer::save`] failures.
    async fn save_checkpoint(&self, checkpoint_id: Option<&str>) -> Result<String> {
        let checkpointer = self.checkpointer().ok_or_else(|| {
            Error::session(format!(
                "no checkpointer configured for session {}",
                self.id()
            ))
        })?;
        let snapshot = self.snapshot().await?;
        let namespace = match self.session_type() {
            SessionType::Agent => CheckpointNamespace::Agent,
            SessionType::AgentTeam => CheckpointNamespace::AgentTeam,
            SessionType::Workflow | SessionType::Node => CheckpointNamespace::Workflow,
        };
        let entity_id = self
            .metadata()
            .entity_id
            .unwrap_or_else(|| self.id().to_string());
        let mut checkpoint = Checkpoint::new(self.id(), namespace, entity_id)
            .with_global_state(snapshot.global)
            .with_component_state(Value::Object(snapshot.components.into_iter().collect()))
            .with_metadata(snapshot.metadata);
        if let Some(position) = snapshot.trace {
            checkpoint = checkpoint.with_trace(Some(position));
        }
        if let Some(id) = checkpoint_id {
            checkpoint = checkpoint.with_checkpoint_id(id);
        }
        checkpointer.save(checkpoint).await
    }

    /// Restores this session from `checkpoint_id`, returning whether a
    /// checkpoint was found.
    ///
    /// Without a configured checkpointer there is nothing to resume, so this
    /// returns `Ok(false)` rather than an error.
    ///
    /// # Errors
    ///
    /// Propagates [`Checkpointer::load`] and [`Session::restore`] failures.
    async fn resume(&self, checkpoint_id: &str) -> Result<bool> {
        let Some(checkpointer) = self.checkpointer() else {
            return Ok(false);
        };
        let Some(checkpoint) = checkpointer.load(self.id(), checkpoint_id).await? else {
            return Ok(false);
        };
        let components = match checkpoint.component_state {
            Value::Object(map) => map.into_iter().collect(),
            _ => BTreeMap::new(),
        };
        let snapshot = SessionSnapshot {
            session_id: checkpoint.session_id,
            kind: self.session_type(),
            global: checkpoint.global_state,
            components,
            trace: checkpoint.trace,
            metadata: checkpoint.metadata,
        };
        self.restore(snapshot).await?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;

    use openjiuwen_common::error::{Error, Result};

    use super::{Session, SessionMetadata, SessionSnapshot, SessionType};
    use crate::checkpoint::MemoryCheckpointer;
    use crate::interaction::{InteractionRequest, InteractionResponse};
    use crate::stream::StreamItem;
    use crate::tracer::{MemoryTraceSink, TracePosition, TraceSink};

    #[derive(Debug, Default)]
    struct FakeSession {
        global: Mutex<serde_json::Value>,
        components: Mutex<std::collections::BTreeMap<String, serde_json::Value>>,
        sink: MemoryTraceSink,
    }

    #[async_trait]
    impl Session for FakeSession {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "s1"
        }

        fn parent_id(&self) -> Option<&str> {
            None
        }

        fn session_type(&self) -> SessionType {
            SessionType::Workflow
        }

        async fn get_state(&self, key: &str) -> Result<Option<serde_json::Value>> {
            if key.is_empty() {
                return Ok(Some(self.global.lock().clone()));
            }
            Ok(self.global.lock().get(key).cloned())
        }

        async fn update_state(&self, patch: serde_json::Value) -> Result<()> {
            if !patch.is_object() {
                return Err(Error::validation("patch must be an object"));
            }
            *self.global.lock() = patch;
            Ok(())
        }

        async fn get_component_state(
            &self,
            component_id: &str,
        ) -> Result<Option<serde_json::Value>> {
            Ok(self.components.lock().get(component_id).cloned())
        }

        async fn update_component_state(
            &self,
            component_id: &str,
            patch: serde_json::Value,
        ) -> Result<()> {
            self.components
                .lock()
                .insert(component_id.to_string(), patch);
            Ok(())
        }

        async fn write_stream(&self, _item: StreamItem) -> Result<()> {
            Ok(())
        }

        async fn request_interaction(
            &self,
            _request: InteractionRequest,
        ) -> Result<InteractionResponse> {
            Err(Error::unsupported("fake session"))
        }

        fn trace(&self) -> &dyn TraceSink {
            &self.sink
        }
    }

    impl FakeSession {
        fn with_state(global: serde_json::Value) -> Arc<Self> {
            let session = Self::default();
            *session.global.lock() = global;
            Arc::new(session)
        }
    }

    #[test]
    fn default_metadata_derives_from_id_and_kind() {
        let session = FakeSession::default();
        let metadata = session.metadata();
        assert_eq!(metadata.session_id, "s1");
        assert_eq!(metadata.parent_session_id, None);
        assert_eq!(metadata.kind, SessionType::Workflow);
    }

    #[test]
    fn metadata_constructor_sets_id_and_kind() {
        let metadata = SessionMetadata::new("a1", SessionType::Agent);
        assert_eq!(metadata.session_id, "a1");
        assert_eq!(metadata.kind, SessionType::Agent);
    }

    #[test]
    fn session_type_wire_names_are_snake_case() {
        assert_eq!(
            serde_json::to_value(SessionType::Workflow).unwrap(),
            json!("workflow")
        );
        assert_eq!(
            serde_json::to_value(SessionType::AgentTeam).unwrap(),
            json!("agent_team")
        );
        assert_eq!(
            serde_json::from_value::<SessionType>(json!("node")).unwrap(),
            SessionType::Node
        );
        assert_eq!(SessionType::default(), SessionType::Workflow);
    }

    #[test]
    fn snapshot_metadata_round_trips_and_tolerates_unknown_fields() {
        let raw = json!({
            "session_id": "s1",
            "kind": "agent_team",
            "entity_id": "team-1",
            "unknown": true,
        });
        let metadata: SessionMetadata = serde_json::from_value(raw).unwrap();
        assert_eq!(metadata.kind, SessionType::AgentTeam);
        assert_eq!(metadata.entity_id.as_deref(), Some("team-1"));
        let round_tripped = serde_json::to_value(&metadata).unwrap();
        assert_eq!(round_tripped["session_id"], json!("s1"));
        assert_eq!(round_tripped["kind"], json!("agent_team"));
    }

    #[test]
    fn snapshot_wire_shape_is_snake_case() {
        let snapshot = SessionSnapshot::new("s1", SessionType::Agent);
        let value = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(value["session_id"], json!("s1"));
        assert_eq!(value["global"], json!({}));
        assert_eq!(value["components"], json!({}));
    }

    #[tokio::test]
    async fn snapshot_and_restore_round_trip_global_state() {
        let session = FakeSession::with_state(json!({ "a": 1 }));
        let snapshot = session.snapshot().await.unwrap();
        assert_eq!(snapshot.global, json!({ "a": 1 }));
        assert!(snapshot.components.is_empty());

        let other = FakeSession::with_state(json!({}));
        other.restore(snapshot).await.unwrap();
        assert_eq!(other.get_state("").await.unwrap(), Some(json!({ "a": 1 })));
    }

    #[tokio::test]
    async fn restore_ignores_non_object_global_state() {
        let session = FakeSession::with_state(json!({ "a": 1 }));
        let mut snapshot = SessionSnapshot::new("s1", SessionType::Workflow);
        snapshot.global = json!("scalar");
        session.restore(snapshot).await.unwrap();
        assert_eq!(
            session.get_state("").await.unwrap(),
            Some(json!({ "a": 1 }))
        );
    }

    #[tokio::test]
    async fn restore_applies_component_state_and_trace_position() {
        let session = FakeSession::default();
        let mut snapshot = SessionSnapshot::new("s1", SessionType::Workflow);
        snapshot
            .components
            .insert("node".into(), json!({ "done": true }));
        snapshot.trace = Some(TracePosition {
            trace_id: "t1".into(),
            last_invoke_id: Some("i1".into()),
            frame_count: 2,
        });
        session.restore(snapshot).await.unwrap();
        assert_eq!(
            session.get_component_state("node").await.unwrap(),
            Some(json!({ "done": true }))
        );
        assert_eq!(
            session.trace().position().map(|position| position.trace_id),
            Some("t1".to_string())
        );
    }

    #[tokio::test]
    async fn save_checkpoint_requires_a_checkpointer() {
        let session = FakeSession::default();
        let error = session.save_checkpoint(None).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Session);
    }

    #[tokio::test]
    async fn resume_without_checkpointer_is_false() {
        let session = FakeSession::default();
        assert!(!session.resume("cp-1").await.unwrap());
    }

    #[tokio::test]
    async fn save_and_resume_checkpoint_round_trip() {
        #[derive(Debug)]
        struct CheckpointedSession {
            inner: FakeSession,
            checkpointer: Arc<dyn crate::checkpoint::Checkpointer>,
        }

        #[async_trait]
        impl Session for CheckpointedSession {
            fn id(&self) -> &str {
                self.inner.id()
            }
            fn parent_id(&self) -> Option<&str> {
                self.inner.parent_id()
            }
            fn session_type(&self) -> SessionType {
                self.inner.session_type()
            }
            async fn get_state(&self, key: &str) -> Result<Option<serde_json::Value>> {
                self.inner.get_state(key).await
            }
            async fn update_state(&self, patch: serde_json::Value) -> Result<()> {
                self.inner.update_state(patch).await
            }
            async fn get_component_state(
                &self,
                component_id: &str,
            ) -> Result<Option<serde_json::Value>> {
                self.inner.get_component_state(component_id).await
            }
            async fn update_component_state(
                &self,
                component_id: &str,
                patch: serde_json::Value,
            ) -> Result<()> {
                self.inner.update_component_state(component_id, patch).await
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
                self.inner.trace()
            }
            fn checkpointer(&self) -> Option<Arc<dyn crate::checkpoint::Checkpointer>> {
                Some(self.checkpointer.clone())
            }
            async fn snapshot(&self) -> Result<SessionSnapshot> {
                let mut snapshot = self.inner.snapshot().await?;
                snapshot.components = self.inner.components.lock().clone();
                Ok(snapshot)
            }
        }

        let checkpointer: Arc<dyn crate::checkpoint::Checkpointer> =
            Arc::new(MemoryCheckpointer::new());
        let session = CheckpointedSession {
            inner: FakeSession::default(),
            checkpointer: checkpointer.clone(),
        };
        session.update_state(json!({ "step": 1 })).await.unwrap();
        session
            .update_component_state("node", json!({ "value": 2 }))
            .await
            .unwrap();
        let checkpoint_id = session.save_checkpoint(Some("cp-1")).await.unwrap();
        assert_eq!(checkpoint_id, "cp-1");

        let target = CheckpointedSession {
            inner: FakeSession::default(),
            checkpointer,
        };
        assert!(target.resume("cp-1").await.unwrap());
        assert_eq!(
            target.get_state("").await.unwrap(),
            Some(json!({ "step": 1 }))
        );
        assert_eq!(
            target.get_component_state("node").await.unwrap(),
            Some(json!({ "value": 2 }))
        );
        assert!(!target.resume("missing").await.unwrap());
    }
}
