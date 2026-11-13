//! The in-process backend of the spawn seam: an agent that lives in this
//! process, driven without a child or an endpoint.
//!
//! A host that binds an agent to its own runtime (a harness instance, a
//! long-lived session, a test double) has nothing to launch, so this backend
//! wraps it in an [`InProcessHandler`] and hands it out as a
//! [`crate::spawn::SpawnedAgent`] like any other backend. The seam stays uniform:
//! the same `send`/`stream`/`shutdown` calls, the same
//! [`crate::spawn::SpawnBackend::InProcess`] identity, the same
//! [`crate::spawn::SpawnHealth`] snapshot.
//!
//! Two shapes:
//!
//! - [`InProcessSpawner::new`] takes an [`InProcessFactory`], so every spawn
//!   builds its own handler from the [`crate::spawn::SpawnSpec`].
//! - [`InProcessSpawner::from_handler`] shares one already-built handler across
//!   every spawn, which is what a host that owns a single agent wants.
//!
//! An in-process handler is *stopped*, not signalled: [`SpawnedAgent::shutdown`]
//! and [`SpawnedAgent::force_kill`] both call [`InProcessHandler::shutdown`], and
//! there is no OS process to report (`pid` is always `None`).
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnSpec, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_common::utils::id::uuid_v4;
use serde_json::Value;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The hosted half of the in-process backend: the agent itself.
///
/// A handler reports liveness, answers one invocation, streams its frames, and
/// stops. It has no process and no transport, so the only failure it can report
/// is its own.
#[async_trait]
pub trait InProcessHandler: Send + Sync {
    /// Whether the handler can still serve calls.
    ///
    /// Defaults to `true`: a handler that cannot report its state is assumed up.
    fn is_alive(&self) -> bool {
        true
    }

    /// Handles one invocation and returns its terminal output.
    ///
    /// # Errors
    ///
    /// Whatever the hosted agent reports: a validation error for unusable
    /// inputs, a cancellation, or an agent-level failure.
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;

    /// Handles one invocation as a stream of [`AgentChunk`] frames.
    ///
    /// The stream reports a failure as its own item rather than at call time, so
    /// this method does not return a `Result`.
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;

    /// Stops the handler; idempotent.
    ///
    /// # Errors
    ///
    /// Whatever the hosted agent reports while releasing its resources.
    async fn shutdown(&self) -> Result<()>;
}

/// Builds the handler for one spawn, from the spawn spec it was asked to serve.
///
/// The factory form lets a spec select the bound agent (its `session_id`,
/// `init_kwargs`, or `payload`); a spawn that cannot build its handler fails with
/// the factory's error and launches nothing.
pub type InProcessFactory =
    Arc<dyn Fn(&SpawnSpec) -> Result<Arc<dyn InProcessHandler>> + Send + Sync>;

/// The in-process [`AgentSpawner`]: wraps a host-bound handler in a
/// [`SpawnedAgent`].
///
/// Cheap to clone (the factory is an `Arc` and is shared); every spawn builds a
/// fresh handle with its own id.
#[derive(Clone)]
pub struct InProcessSpawner {
    /// Builds the handler for each spawn.
    factory: InProcessFactory,
}

impl InProcessSpawner {
    /// A spawner that calls `factory` once per spawn.
    #[must_use]
    pub fn new(factory: InProcessFactory) -> Self {
        Self { factory }
    }

    /// A spawner that hands out `handler` to every spawn.
    ///
    /// The handle is new per spawn (its own id, its own alive flag), but the
    /// handler — and therefore the hosted agent's state — is shared.
    #[must_use]
    pub fn from_handler(handler: Arc<dyn InProcessHandler>) -> Self {
        Self::new(Arc::new(move |_spec: &SpawnSpec| Ok(Arc::clone(&handler))))
    }
}

impl fmt::Debug for InProcessSpawner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InProcessSpawner")
            .field("backend", &SpawnBackend::InProcess)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl AgentSpawner for InProcessSpawner {
    fn backend(&self) -> SpawnBackend {
        SpawnBackend::InProcess
    }

    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let handler = (self.factory)(&spec)?;
        Ok(Arc::new(InProcessAgent {
            id: format!("in-process-{}", uuid_v4()),
            spec,
            handler,
            alive: AtomicBool::new(true),
        }))
    }
}

/// A live in-process handle: one hosted handler plus this handle's own liveness.
///
/// Obtained from [`InProcessSpawner`]; see the module docs for why `pid` is
/// always `None` and why killing it stops the handler instead of signalling a
/// process.
pub struct InProcessAgent {
    /// Stable handle id (`in-process-<uuid>`), unique per spawn.
    id: String,
    /// The spec this handle was spawned from.
    spec: SpawnSpec,
    /// The hosted agent.
    handler: Arc<dyn InProcessHandler>,
    /// Whether this handle has been stopped.
    alive: AtomicBool,
}

impl InProcessAgent {
    /// The spec this handle was spawned from.
    #[must_use]
    pub fn spec(&self) -> &SpawnSpec {
        &self.spec
    }

    /// The hosted handler, for a caller that wants the concrete agent back.
    #[must_use]
    pub fn handler(&self) -> &Arc<dyn InProcessHandler> {
        &self.handler
    }

    /// Marks the handle dead; shared by `shutdown` and `force_kill`.
    fn mark_dead(&self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

impl fmt::Debug for InProcessAgent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InProcessAgent")
            .field("id", &self.id)
            .field("session_id", &self.spec.session_id)
            .field("alive", &self.is_alive())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SpawnedAgent for InProcessAgent {
    fn id(&self) -> &str {
        &self.id
    }

    /// Alive while this handle is open *and* the handler still serves calls.
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst) && self.handler.is_alive()
    }

    /// Always `None`: an in-process handler has no OS process.
    fn pid(&self) -> Option<u32> {
        None
    }

    fn backend(&self) -> SpawnBackend {
        SpawnBackend::InProcess
    }

    /// No probe exists in-process, so liveness is the answer.
    fn is_healthy(&self) -> bool {
        self.is_alive()
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        self.handler.send(inputs).await
    }

    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        self.handler.stream(inputs)
    }

    /// Stops the handler and marks the handle dead; a second call still reports
    /// `Ok(true)` — the handler's own `shutdown` is idempotent.
    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        self.handler.shutdown().await?;
        self.mark_dead();
        Ok(true)
    }

    /// An in-process handler is stopped, not signalled: there is no OS process to
    /// kill, so this is `shutdown` without the graceful deadline.
    async fn force_kill(&self) -> Result<()> {
        self.handler.shutdown().await?;
        self.mark_dead();
        Ok(())
    }

    /// There is no probe in-process: the answer is the current liveness.
    #[allow(clippy::unused_async_trait_impl)]
    async fn health(&self) -> Result<bool> {
        Ok(self.is_alive())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_common::error::{Error, ErrorKind};
    use parking_lot::Mutex;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;

    /// A handler that records the calls it served.
    struct RecordingHandler {
        alive: AtomicBool,
        shutdowns: AtomicUsize,
        seen: Mutex<Vec<Value>>,
    }

    impl RecordingHandler {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                alive: AtomicBool::new(true),
                shutdowns: AtomicUsize::new(0),
                seen: Mutex::new(Vec::new()),
            })
        }

        fn shutdowns(&self) -> usize {
            self.shutdowns.load(Ordering::SeqCst)
        }

        fn seen(&self) -> Vec<Value> {
            self.seen.lock().clone()
        }
    }

    #[async_trait]
    impl InProcessHandler for RecordingHandler {
        fn is_alive(&self) -> bool {
            self.alive.load(Ordering::SeqCst)
        }

        async fn send(&self, inputs: Value) -> Result<AgentOutput> {
            self.seen.lock().push(inputs.clone());
            Ok(AgentOutput::answer(format!("echo: {inputs}")))
        }

        fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::iter(vec![
                Ok(AgentChunk::llm_output(0, "first")),
                Ok(AgentChunk::llm_output(1, inputs.to_string())),
            ]))
        }

        async fn shutdown(&self) -> Result<()> {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            self.alive.store(false, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A handler whose every call fails.
    struct FailingHandler;

    #[async_trait]
    impl InProcessHandler for FailingHandler {
        async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
            Err(Error::validation("the bound agent is unavailable"))
        }

        fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::iter(vec![Err(Error::validation(
                "the bound agent is unavailable",
            ))]))
        }

        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    /// The trait object form of a recording handler, so a test can keep its own
    /// handle on the same object the spawner hosts.
    fn as_handler(handler: &Arc<RecordingHandler>) -> Arc<dyn InProcessHandler> {
        Arc::clone(handler) as Arc<dyn InProcessHandler>
    }

    #[tokio::test]
    async fn the_factory_runs_once_per_spawn_with_the_spec() {
        let sessions: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_by_factory = Arc::clone(&sessions);
        let spawner = InProcessSpawner::new(Arc::new(move |spec: &SpawnSpec| {
            seen_by_factory.lock().push(spec.session_id.clone());
            Ok(RecordingHandler::new() as Arc<dyn InProcessHandler>)
        }));
        let first = spawner
            .spawn(SpawnSpec::class_agent("mod", "A").with_session_id("s-1"))
            .await
            .unwrap();
        let second = spawner
            .spawn(SpawnSpec::class_agent("mod", "A").with_session_id("s-2"))
            .await
            .unwrap();
        assert_eq!(
            sessions.lock().clone(),
            vec![Some("s-1".to_owned()), Some("s-2".to_owned())]
        );
        assert_ne!(first.id(), second.id());
        assert!(first.id().starts_with("in-process-"));
        assert_eq!(first.backend(), SpawnBackend::InProcess);
        assert_eq!(spawner.backend(), SpawnBackend::InProcess);
    }

    #[tokio::test]
    async fn send_and_stream_delegate_to_the_handler() {
        let handler = RecordingHandler::new();
        let spawner = InProcessSpawner::from_handler(as_handler(&handler));
        let agent = spawner
            .spawn(SpawnSpec::class_agent("mod", "A"))
            .await
            .unwrap();
        let output = agent.send(json!({"query": "hi"})).await.unwrap();
        assert!(output.output.contains("echo:"), "{output:?}");
        assert_eq!(handler.seen(), vec![json!({"query": "hi"})]);
        let chunks: Vec<_> = agent.stream(json!({})).collect().await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_ref().unwrap().index, 0);
        assert_eq!(chunks[1].as_ref().unwrap().index, 1);
    }

    #[tokio::test]
    async fn handler_failures_reach_the_caller() {
        let spawner = InProcessSpawner::from_handler(Arc::new(FailingHandler));
        let agent = spawner.spawn(SpawnSpec::default()).await.unwrap();
        let error = agent.send(json!({})).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let chunks: Vec<_> = agent.stream(json!({})).collect().await;
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_err());
    }

    #[tokio::test]
    async fn shutdown_marks_the_handle_dead_and_is_idempotent() {
        let handler = RecordingHandler::new();
        let spawner = InProcessSpawner::from_handler(as_handler(&handler));
        let agent = spawner.spawn(SpawnSpec::default()).await.unwrap();
        assert_eq!(agent.backend(), SpawnBackend::InProcess);
        assert_eq!(agent.pid(), None);
        assert!(agent.is_alive());
        assert!(agent.is_healthy());
        assert!(agent.health().await.unwrap());

        assert!(agent.shutdown(Duration::from_secs(1)).await.unwrap());
        assert!(!agent.is_alive());
        assert!(!agent.is_healthy());
        assert!(!agent.health().await.unwrap());
        // A second call is still a success.
        assert!(agent.shutdown(Duration::from_secs(1)).await.unwrap());
        assert_eq!(handler.shutdowns(), 2);
    }

    #[tokio::test]
    async fn force_kill_stops_the_handler_without_a_process() {
        let handler = RecordingHandler::new();
        let spawner = InProcessSpawner::from_handler(as_handler(&handler));
        let agent = spawner.spawn(SpawnSpec::default()).await.unwrap();
        agent.force_kill().await.unwrap();
        assert!(!agent.is_alive());
        agent.force_kill().await.unwrap();
        assert_eq!(handler.shutdowns(), 2);
        assert!(!agent.is_alive());
    }

    #[tokio::test]
    async fn from_handler_shares_one_handler_but_each_handle_gets_an_id() {
        let handler = RecordingHandler::new();
        let spawner = InProcessSpawner::from_handler(as_handler(&handler));
        let first = spawner.spawn(SpawnSpec::default()).await.unwrap();
        let second = spawner
            .spawn(SpawnSpec::default().with_session_id("s"))
            .await
            .unwrap();
        assert_ne!(first.id(), second.id());
        first.send(json!({"a": 1})).await.unwrap();
        second.send(json!({"a": 2})).await.unwrap();
        assert_eq!(handler.seen().len(), 2);
        // Both handles point at the one hosted agent, so stopping it through
        // either closes the other's view too.
        first.force_kill().await.unwrap();
        assert!(!first.is_alive());
        assert!(!second.is_alive(), "the shared handler is stopped");
    }

    #[tokio::test]
    async fn a_factory_error_propagates() {
        let spawner = InProcessSpawner::new(Arc::new(|_spec: &SpawnSpec| {
            Err(Error::validation("no handler for this spec"))
        }));
        let Err(error) = spawner.spawn(SpawnSpec::default()).await else {
            panic!("the factory refused the spec");
        };
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("no handler for this spec"));
        assert_eq!(spawner.backend(), SpawnBackend::InProcess);
        assert!(format!("{spawner:?}").contains("InProcessSpawner"));
    }

    #[test]
    fn the_handle_exposes_its_spec_and_handler() {
        let handler: Arc<dyn InProcessHandler> = RecordingHandler::new();
        let agent = InProcessAgent {
            id: "in-process-test".to_owned(),
            spec: SpawnSpec::class_agent("mod", "A").with_session_id("s"),
            handler,
            alive: AtomicBool::new(true),
        };
        assert_eq!(agent.spec().session_id.as_deref(), Some("s"));
        assert_eq!(agent.spec().agent_class.as_deref(), Some("A"));
        assert!(agent.handler().is_alive());
        assert_eq!(agent.id(), "in-process-test");
        let rendered = format!("{agent:?}");
        assert!(rendered.contains("InProcessAgent"), "{rendered}");
        assert!(rendered.contains("in-process-test"), "{rendered}");
    }

    #[test]
    fn a_handler_that_does_not_report_liveness_is_assumed_up() {
        struct SilentHandler;

        #[async_trait]
        impl InProcessHandler for SilentHandler {
            async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
                Ok(AgentOutput::answer("ok"))
            }

            fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
                Box::pin(futures::stream::empty())
            }

            async fn shutdown(&self) -> Result<()> {
                Ok(())
            }
        }

        assert!(SilentHandler.is_alive());
    }
}
