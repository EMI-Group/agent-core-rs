//! Crate-private plumbing shared by every concrete session.
//!
//! [`SessionInner`] holds everything the [`crate::session::Session`] contract
//! needs regardless of session flavour: the identifier block, the shared config,
//! the state store, the stream hub and writer manager, the interaction broker,
//! the tracer, and the optional checkpointer and parent. Concrete sessions
//! ([`crate::sessions::WorkflowSession`], [`crate::sessions::NodeSession`], and
//! the agent/team sessions that land next) wrap one `Arc<SessionInner>` and
//! override only their kind-specific views.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use parking_lot::RwLock;
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::now_rfc3339;

use crate::checkpoint::Checkpointer;
use crate::config::SessionConfig;
use crate::constants::DEFAULT_STREAM_CAPACITY;
use crate::interaction::{InteractionBroker, InteractionRequest, InteractionResponse};
use crate::session::{Session, SessionMetadata, SessionSnapshot, SessionType};
use crate::state::{ensure_object, StateStore};
use crate::stream::{StreamEvent, StreamHub, StreamItem, StreamWriterManager};
use crate::tracer::{StreamTraceSink, TraceSink, Tracer};

/// Everything a concrete session hands to [`SessionInner::new`].
pub(crate) struct SessionInnerOptions {
    pub(crate) id: String,
    pub(crate) parent_id: Option<String>,
    pub(crate) kind: SessionType,
    pub(crate) entity_id: Option<String>,
    pub(crate) entity_name: Option<String>,
    pub(crate) trace_id: Option<String>,
    pub(crate) created_at: Option<String>,
    pub(crate) config: Arc<RwLock<SessionConfig>>,
    pub(crate) state: Arc<dyn StateStore>,
    pub(crate) writers: Option<Arc<StreamWriterManager>>,
    pub(crate) stream_capacity: Option<usize>,
    /// `None` installs a [`StreamTraceSink`] over the writer manager; pass
    /// `Some(Arc::new(NoopTraceSink::new()))` to opt out of trace frames.
    pub(crate) sink: Option<Arc<dyn TraceSink>>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) parent: Option<Arc<dyn Session>>,
    pub(crate) metadata: BTreeMap<String, Value>,
}

/// The shared implementation behind every concrete session.
pub(crate) struct SessionInner {
    pub(crate) id: String,
    pub(crate) parent_id: Option<String>,
    pub(crate) kind: SessionType,
    pub(crate) entity_id: Option<String>,
    pub(crate) entity_name: Option<String>,
    pub(crate) trace_id: String,
    pub(crate) created_at: Option<String>,
    pub(crate) config: Arc<RwLock<SessionConfig>>,
    pub(crate) state: Arc<dyn StateStore>,
    pub(crate) writers: Arc<StreamWriterManager>,
    pub(crate) trace_sink: Arc<dyn TraceSink>,
    pub(crate) tracer: Tracer,
    pub(crate) interactions: Arc<InteractionBroker>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Parent session, kept for derived sessions that need parent lookups;
    /// the workflow/node sessions delegate through their own fields.
    #[allow(dead_code)]
    pub(crate) parent: Option<Arc<dyn Session>>,
    pub(crate) metadata: BTreeMap<String, Value>,
}

impl SessionInner {
    /// Builds the shared inner, wiring a hub, writer manager, and trace sink
    /// when the caller did not supply them.
    pub(crate) fn new(options: SessionInnerOptions) -> Arc<Self> {
        let SessionInnerOptions {
            id,
            parent_id,
            kind,
            entity_id,
            entity_name,
            trace_id,
            created_at,
            config,
            state,
            writers,
            stream_capacity,
            sink,
            checkpointer,
            parent,
            metadata,
        } = options;
        let writers = writers.unwrap_or_else(|| {
            let hub = StreamHub::with_capacity(stream_capacity.unwrap_or(DEFAULT_STREAM_CAPACITY));
            StreamWriterManager::new(hub)
        });
        let trace_sink =
            sink.unwrap_or_else(|| Arc::new(StreamTraceSink::new(Arc::clone(&writers))));
        let tracer = Tracer::with_sink(
            id.clone(),
            trace_id.unwrap_or_default(),
            Arc::clone(&trace_sink),
        );
        let trace_id = tracer.trace_id().to_string();
        Arc::new(Self {
            id,
            parent_id,
            kind,
            entity_id,
            entity_name,
            trace_id,
            created_at: created_at.or_else(|| Some(now_rfc3339())),
            config,
            state,
            writers,
            trace_sink,
            tracer,
            interactions: Arc::new(InteractionBroker::new()),
            checkpointer,
            parent,
            metadata,
        })
    }

    /// Derives the public metadata block from the inner fields.
    pub(crate) fn metadata(&self) -> SessionMetadata {
        SessionMetadata {
            session_id: self.id.clone(),
            parent_session_id: self.parent_id.clone(),
            kind: self.kind,
            entity_id: self.entity_id.clone(),
            entity_name: self.entity_name.clone(),
            trace_id: Some(self.trace_id.clone()),
            created_at: self.created_at.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Returns a cloned handle to the shared config lock.
    pub(crate) fn config_handle(&self) -> Arc<RwLock<SessionConfig>> {
        Arc::clone(&self.config)
    }

    /// Reads the committed global lane at `key`.
    pub(crate) fn get_state(&self, key: &str) -> Option<Value> {
        self.state.get_global(key)
    }

    /// Merges `patch` into the committed global lane, bypassing staging.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object.
    pub(crate) fn update_state(&self, patch: Value) -> Result<()> {
        let object = ensure_object(patch, "global state patch")?;
        self.state.write_global(Value::Object(object))
    }

    /// Reads component state by nested component id; an empty id yields `None`.
    pub(crate) fn get_component_state(&self, component_id: &str) -> Option<Value> {
        if component_id.is_empty() {
            return None;
        }
        self.state.get_component(component_id)
    }

    /// Merges `patch` into the component lane of `component_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `component_id` is empty or the write
    /// cannot be applied.
    pub(crate) fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        if component_id.is_empty() {
            return Err(Error::validation("component id must not be empty"));
        }
        self.state.update_component(component_id, patch)
    }

    /// Publishes `item` to the session stream.
    ///
    /// # Errors
    ///
    /// Propagates [`StreamWriterManager::write`] errors (a `Null` payload is
    /// [`Error::Validation`]).
    pub(crate) async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.writers.write(item).await
    }

    /// Publishes an interaction frame and awaits the matching response.
    ///
    /// # Errors
    ///
    /// Propagates [`InteractionBroker::request`] errors (validation, timeout,
    /// cancellation).
    pub(crate) async fn request_interaction(
        &self,
        request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        self.interactions.request(request, &self.writers).await
    }

    /// Delivers `response` to the pending interaction it answers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when no request with that id is pending.
    pub(crate) fn resolve_interaction(&self, response: InteractionResponse) -> Result<()> {
        self.interactions.resolve(response)
    }

    /// Subscribes to the session stream.
    pub(crate) fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        self.writers.subscribe()
    }

    /// Publishes the completion marker; idempotent.
    pub(crate) fn end_stream(&self) {
        self.writers.close();
    }

    /// Closes the shared hub; idempotent.
    pub(crate) fn close(&self) {
        self.writers.close();
    }

    /// Captures global state, component state, the trace position, and
    /// metadata.
    pub(crate) fn snapshot(&self) -> SessionSnapshot {
        let global = self
            .state
            .get_global("")
            .unwrap_or_else(|| Value::Object(Map::new()));
        let components = match self.state.components() {
            Value::Object(map) => map.into_iter().collect(),
            _ => BTreeMap::new(),
        };
        SessionSnapshot {
            session_id: self.id.clone(),
            kind: self.kind,
            global,
            components,
            trace: self.trace_sink.position(),
            metadata: self.metadata().metadata,
        }
    }

    /// Restores global state, component state, and the trace position.
    ///
    /// # Errors
    ///
    /// Propagates [`SessionInner::update_state`] and
    /// [`SessionInner::update_component_state`] failures.
    pub(crate) fn restore(&self, snapshot: SessionSnapshot) -> Result<()> {
        let SessionSnapshot {
            global,
            components,
            trace,
            ..
        } = snapshot;
        if global.is_object() {
            self.update_state(global)?;
        }
        for (component_id, patch) in components {
            self.update_component_state(&component_id, patch)?;
        }
        if let Some(position) = trace {
            self.trace_sink.set_position(position);
        }
        Ok(())
    }
}

/// The pieces a concrete session resolves once and shares with its children.
pub(crate) struct SharedParts {
    pub(crate) config: Arc<RwLock<SessionConfig>>,
    pub(crate) writers: Arc<StreamWriterManager>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) trace_id: Option<String>,
}

/// Resolves the config, stream writers, checkpointer, and trace id for a new
/// session: a parent's values win, and an explicit argument overrides a
/// parent's for the checkpointer and trace id.
pub(crate) fn resolve_shared(
    parent: Option<&Arc<dyn Session>>,
    checkpointer: Option<Arc<dyn Checkpointer>>,
    trace_id: Option<String>,
    stream_capacity: Option<usize>,
    envs: &BTreeMap<String, Value>,
) -> SharedParts {
    let parent_config = parent.and_then(|parent| parent.config());
    let parent_writers = parent.and_then(|parent| parent.writers());
    let parent_checkpointer = parent.and_then(|parent| parent.checkpointer());
    let parent_trace_id = parent.and_then(|parent| parent.metadata().trace_id);

    let config = parent_config.unwrap_or_else(|| {
        let mut config = SessionConfig::new();
        for (key, value) in envs {
            config.set_env(key.clone(), value.clone());
        }
        Arc::new(RwLock::new(config))
    });
    let writers = parent_writers.unwrap_or_else(|| {
        StreamWriterManager::new(StreamHub::with_capacity(
            stream_capacity.unwrap_or(DEFAULT_STREAM_CAPACITY),
        ))
    });

    SharedParts {
        config,
        writers,
        checkpointer: checkpointer.or(parent_checkpointer),
        trace_id: trace_id.or(parent_trace_id),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;
    use std::sync::Arc;

    use parking_lot::RwLock;
    use serde_json::json;

    use super::{SessionInner, SessionInnerOptions};
    use crate::config::SessionConfig;
    use crate::session::SessionType;
    use crate::state::WorkflowState;
    use crate::tracer::NoopTraceSink;

    fn build_inner() -> Arc<SessionInner> {
        SessionInner::new(SessionInnerOptions {
            id: "s1".to_string(),
            parent_id: None,
            kind: SessionType::Workflow,
            entity_id: Some("wf".to_string()),
            entity_name: Some("wf".to_string()),
            trace_id: Some("trace-1".to_string()),
            created_at: None,
            config: Arc::new(RwLock::new(SessionConfig::new())),
            state: Arc::new(WorkflowState::with_global_lane(false)),
            writers: None,
            stream_capacity: Some(8),
            sink: Some(Arc::new(NoopTraceSink::new())),
            checkpointer: None,
            parent: None,
            metadata: BTreeMap::new(),
        })
    }

    #[test]
    fn update_state_is_immediately_visible() {
        let inner = build_inner();
        inner.update_state(json!({ "a": 1 })).unwrap();
        assert_eq!(inner.get_state(""), Some(json!({ "a": 1 })));
        inner.update_state(json!({ "a": { "b": 2 } })).unwrap();
        assert_eq!(inner.get_state("a.b"), Some(json!(2)));
        assert!(inner.update_state(json!("nope")).is_err());
    }

    #[test]
    fn component_ids_nest() {
        let inner = build_inner();
        inner
            .update_component_state("a.b", json!({ "value": 1 }))
            .unwrap();
        assert_eq!(
            inner.get_component_state("a.b"),
            Some(json!({ "value": 1 }))
        );
        assert_eq!(
            inner.get_component_state("a"),
            Some(json!({ "b": { "value": 1 } }))
        );
        assert_eq!(inner.get_component_state(""), None);
        assert!(inner.update_component_state("", json!({})).is_err());
    }

    #[tokio::test]
    async fn end_stream_is_idempotent() {
        let inner = build_inner();
        inner.end_stream();
        inner.end_stream();
        assert!(inner.writers.is_closed());
    }

    #[test]
    fn metadata_carries_trace_and_creation_time() {
        let inner = build_inner();
        let metadata = inner.metadata();
        assert_eq!(metadata.session_id, "s1");
        assert_eq!(metadata.kind, SessionType::Workflow);
        assert_eq!(metadata.trace_id.as_deref(), Some("trace-1"));
        assert!(metadata.created_at.is_some());
    }
}
