//! The spawner registry: one place where a runner's spawn backends live, and the
//! routing that picks one for a [`crate::spawn::SpawnSpec`].
//!
//! A runner that installs a single `AgentSpawner` has one backend; a distributed
//! host needs several at once (a local subprocess, a vendor CLI, an in-process
//! harness, HTTP endpoints) and a rule for which one a spawn goes to. A
//! [`SpawnerRegistry`] is that rule:
//!
//! 1. [`crate::spawn::SpawnSpec::backend`], when the caller named one — a
//!    registered spawner, otherwise a [`DeferredSpawner`] naming the missing
//!    backend (never a silent fallback to another backend).
//! 2. the registry's explicit default ([`SpawnerRegistry::set_default`]).
//! 3. the sole registered spawner, when there is exactly one.
//! 4. nothing usable — an empty registry is `"no default spawn backend"` and an
//!    ambiguous one is `"ambiguous spawn backend"`, both as a
//!    [`DeferredSpawner`] so the failure is a typed
//!    [`openjiuwen_common::error::Error::Unsupported`] with the hint that fixes
//!    it.
//!
//! [`SpawnerRegistry::with_deferred`] pre-registers the deferred backends
//! ([`crate::spawn::SpawnBackend::RemoteMq`],
//! [`crate::spawn::SpawnBackend::RemoteA2a`]) so a spec that asks for them is
//! answered by name rather than by "nothing registered". Everything else is
//! registered on demand; the lock is a `parking_lot::RwLock` that is never held
//! across an `.await` (the spawner `Arc` is cloned out first).
use crate::error;
use crate::spawn::external::ExternalCommandSpawner;
use crate::spawn::in_process::InProcessSpawner;
use crate::spawn::remote::RemoteSpawner;
use crate::spawn::subprocess::SubprocessSpawner;
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnSpec, SpawnedAgent, SubprocessConfig};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// An [`AgentSpawner`] that reports why a backend cannot spawn.
///
/// The registry answers with one of these whenever a backend is named but not
/// registered, so a caller gets a typed
/// [`openjiuwen_common::error::Error::Unsupported`] that names the missing
/// transport instead of a generic "no spawner".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredSpawner {
    /// Backend this spawner stands in for.
    backend: SpawnBackend,
    /// What is missing, phrased as the unsupported feature.
    reason: String,
    /// What the caller should do instead.
    hint: String,
}

impl DeferredSpawner {
    /// The deferred spawner for `backend`, with this crate's standard wording.
    ///
    /// The wording names the missing transport: Pulsar/ZeroMQ reply-topic routing
    /// for [`SpawnBackend::RemoteMq`], the A2A transport for
    /// [`SpawnBackend::RemoteA2a`], and the registration method for a backend this
    /// crate implements but the registry does not host.
    #[must_use]
    pub fn for_backend(backend: SpawnBackend) -> Self {
        match backend {
            SpawnBackend::RemoteMq => Self::new(
                backend,
                "Pulsar/ZeroMQ spawn transport",
                "drunner reply-topic routing is deferred; register a `subprocess` or \
                 `remote_http` spawner",
            ),
            SpawnBackend::RemoteA2a => Self::new(
                backend,
                "a2a spawn transport",
                "the A2A transport lives in `openjiuwen-extensions` (feature `a2a`); \
                 register that spawner instead",
            ),
            SpawnBackend::Subprocess
            | SpawnBackend::ExternalCommand
            | SpawnBackend::InProcess
            | SpawnBackend::RemoteHttp => {
                let method = register_method(backend);
                Self::new(
                    backend,
                    format!("no {backend} spawner registered"),
                    format!(
                        "call `SpawnerRegistry::{method}` before spawning through the \
                         `{backend}` backend"
                    ),
                )
            }
            SpawnBackend::Unknown => Self::new(
                backend,
                "spawn backend `unknown`",
                "register a spawner for it or use `subprocess`/`in_process`",
            ),
        }
    }

    /// A deferred spawner with explicit wording.
    #[must_use]
    pub fn new(backend: SpawnBackend, reason: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            backend,
            reason: reason.into(),
            hint: hint.into(),
        }
    }

    /// The backend this spawner stands in for.
    #[must_use]
    pub fn backend_name(&self) -> SpawnBackend {
        self.backend
    }
}

#[async_trait]
impl AgentSpawner for DeferredSpawner {
    fn backend(&self) -> SpawnBackend {
        self.backend
    }

    async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        Err(error::unsupported(&self.reason, &self.hint))
    }
}

/// The `SpawnerRegistry` method that registers `backend`.
const fn register_method(backend: SpawnBackend) -> &'static str {
    match backend {
        SpawnBackend::Subprocess => "register_subprocess",
        SpawnBackend::ExternalCommand => "register_external_command",
        SpawnBackend::InProcess => "register_in_process",
        SpawnBackend::RemoteHttp => "register_remote",
        // Only the four implemented backends reach this helper.
        SpawnBackend::Unknown | SpawnBackend::RemoteMq | SpawnBackend::RemoteA2a => "register",
    }
}

/// The registry's mutable half: the spawners and the routing default.
#[derive(Default)]
struct RegistryState {
    /// One spawner per backend.
    spawners: BTreeMap<SpawnBackend, Arc<dyn AgentSpawner>>,
    /// The backend an unrouted spawn goes to, when a host set one.
    default: Option<SpawnBackend>,
}

/// A set of spawn backends hosted side by side, plus the routing between them.
///
/// See the module docs for the routing precedence. Every method but `spawn` and
/// `spawn_with` is synchronous, and no lock is held across an `.await`.
#[derive(Default)]
pub struct SpawnerRegistry {
    /// Spawners and default, behind one reader/writer lock.
    state: RwLock<RegistryState>,
}

impl SpawnerRegistry {
    /// An empty registry: every spawn is deferred until a backend is registered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry that already answers for the two deferred backends
    /// ([`SpawnBackend::RemoteMq`], [`SpawnBackend::RemoteA2a`]).
    #[must_use]
    pub fn with_deferred() -> Self {
        let registry = Self::new();
        for backend in [SpawnBackend::RemoteMq, SpawnBackend::RemoteA2a] {
            registry.register(backend, Arc::new(DeferredSpawner::for_backend(backend)));
        }
        registry
    }

    /// Registers `spawner` under `backend`, returning whatever it replaced.
    pub fn register(
        &self,
        backend: SpawnBackend,
        spawner: Arc<dyn AgentSpawner>,
    ) -> Option<Arc<dyn AgentSpawner>> {
        self.state.write().spawners.insert(backend, spawner)
    }

    /// Registers a [`SubprocessSpawner`] configured by `config`.
    pub fn register_subprocess(&self, config: SubprocessConfig) -> Option<Arc<dyn AgentSpawner>> {
        self.register(
            SpawnBackend::Subprocess,
            Arc::new(SubprocessSpawner::new(config)),
        )
    }

    /// Registers an [`ExternalCommandSpawner`].
    pub fn register_external_command(
        &self,
        spawner: ExternalCommandSpawner,
    ) -> Option<Arc<dyn AgentSpawner>> {
        self.register(SpawnBackend::ExternalCommand, Arc::new(spawner))
    }

    /// Registers an [`InProcessSpawner`].
    pub fn register_in_process(&self, spawner: InProcessSpawner) -> Option<Arc<dyn AgentSpawner>> {
        self.register(SpawnBackend::InProcess, Arc::new(spawner))
    }

    /// Registers a [`RemoteSpawner`].
    pub fn register_remote(&self, spawner: RemoteSpawner) -> Option<Arc<dyn AgentSpawner>> {
        self.register(SpawnBackend::RemoteHttp, Arc::new(spawner))
    }

    /// Removes the spawner for `backend`, returning it.
    ///
    /// A default that pointed at the removed backend is cleared, so the registry
    /// never routes to a backend it cannot reach.
    pub fn unregister(&self, backend: SpawnBackend) -> Option<Arc<dyn AgentSpawner>> {
        let mut state = self.state.write();
        let removed = state.spawners.remove(&backend);
        if state.default == Some(backend) {
            state.default = None;
        }
        removed
    }

    /// The spawner registered for `backend`, if any.
    #[must_use]
    pub fn get(&self, backend: SpawnBackend) -> Option<Arc<dyn AgentSpawner>> {
        self.state.read().spawners.get(&backend).map(Arc::clone)
    }

    /// Whether `backend` is registered.
    #[must_use]
    pub fn contains(&self, backend: SpawnBackend) -> bool {
        self.state.read().spawners.contains_key(&backend)
    }

    /// How many backends are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.read().spawners.len()
    }

    /// Whether no backend is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.read().spawners.is_empty()
    }

    /// The registered backends, sorted.
    #[must_use]
    pub fn backends(&self) -> Vec<SpawnBackend> {
        self.state.read().spawners.keys().copied().collect()
    }

    /// Routes every unaddressed spawn to `backend`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] naming an unregistered
    /// `backend`: a default the registry cannot honour is refused, not stored.
    pub fn set_default(&self, backend: SpawnBackend) -> Result<()> {
        let mut state = self.state.write();
        if !state.spawners.contains_key(&backend) {
            return Err(Error::validation(format!(
                "spawn backend '{backend}' is not registered, so it cannot be the default"
            )));
        }
        state.default = Some(backend);
        Ok(())
    }

    /// The backend unrouted spawns go to, when a host set one.
    #[must_use]
    pub fn default_backend(&self) -> Option<SpawnBackend> {
        self.state.read().default
    }

    /// The spawner a [`SpawnSpec`] routes to.
    ///
    /// Never `None`: an unroutable spec gets a [`DeferredSpawner`] that names the
    /// problem, so the failure happens at spawn time with the right wording.
    #[must_use]
    pub fn spawner_for(&self, spec: &SpawnSpec) -> Arc<dyn AgentSpawner> {
        let state = self.state.read();
        if let Some(backend) = spec.backend {
            return state.spawners.get(&backend).map_or_else(
                || Arc::new(DeferredSpawner::for_backend(backend)) as Arc<dyn AgentSpawner>,
                Arc::clone,
            );
        }
        if let Some(backend) = state.default {
            if let Some(spawner) = state.spawners.get(&backend) {
                return Arc::clone(spawner);
            }
        }
        if state.spawners.len() == 1 {
            if let Some(spawner) = state.spawners.values().next() {
                return Arc::clone(spawner);
            }
        }
        match state.spawners.len() {
            0 | 1 => Arc::new(DeferredSpawner::new(
                SpawnBackend::Unknown,
                "no default spawn backend",
                "register a spawner and call `SpawnerRegistry::set_default`, or set \
                 `SpawnSpec::backend`",
            )),
            _ => Arc::new(DeferredSpawner::new(
                SpawnBackend::Unknown,
                "ambiguous spawn backend",
                "call `SpawnerRegistry::set_default` or set `SpawnSpec::backend` to pick one",
            )),
        }
    }

    /// Spawns `spec` through the backend it routes to.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] when nothing usable is
    /// registered (with the routing problem named in the message), plus whatever
    /// the chosen backend reports.
    pub async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        self.spawner_for(&spec).spawn(spec).await
    }

    /// Spawns `spec` through exactly `backend`.
    ///
    /// Unlike [`Self::spawn`], this never falls back: a `backend` that is not
    /// registered is a [`DeferredSpawner`] failure naming it.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] when `backend` is not
    /// registered, plus whatever the chosen backend reports.
    pub async fn spawn_with(
        &self,
        backend: SpawnBackend,
        spec: SpawnSpec,
    ) -> Result<Arc<dyn SpawnedAgent>> {
        let spawner = self
            .get(backend)
            .unwrap_or_else(|| Arc::new(DeferredSpawner::for_backend(backend)));
        spawner.spawn(spec).await
    }
}

impl fmt::Debug for SpawnerRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read();
        formatter
            .debug_struct("SpawnerRegistry")
            .field("backends", &state.spawners.keys().collect::<Vec<_>>())
            .field("default", &state.default)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::spawn::in_process::{InProcessHandler, InProcessSpawner};
    use futures::stream::BoxStream;
    use openjiuwen_agent::output::{AgentChunk, AgentOutput};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::{json, Value};

    /// The smallest handler the in-process backend accepts.
    struct EchoHandler;

    #[async_trait]
    impl InProcessHandler for EchoHandler {
        async fn send(&self, inputs: Value) -> Result<AgentOutput> {
            Ok(AgentOutput::answer(format!("echo: {inputs}")))
        }

        fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }

        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    fn in_process() -> InProcessSpawner {
        InProcessSpawner::from_handler(Arc::new(EchoHandler))
    }

    /// The message of the error a deferred spawn produces.
    async fn deferred_message(spawner: Arc<dyn AgentSpawner>) -> String {
        match spawner.spawn(SpawnSpec::default()).await {
            Ok(_) => panic!("a deferred spawner never spawns"),
            Err(error) => {
                assert_eq!(error.kind(), ErrorKind::Unsupported);
                error.to_string()
            }
        }
    }

    #[test]
    fn registration_replace_and_removal() {
        let registry = SpawnerRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        assert!(registry.register_in_process(in_process()).is_none());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains(SpawnBackend::InProcess));
        assert!(!registry.contains(SpawnBackend::Subprocess));

        let replaced = registry.register_in_process(in_process());
        assert!(replaced.is_some(), "the previous spawner comes back");
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister(SpawnBackend::InProcess);
        assert!(removed.is_some());
        assert!(registry.is_empty());
        assert!(registry.unregister(SpawnBackend::InProcess).is_none());
    }

    #[test]
    fn backends_are_sorted_and_the_types_have_a_debug_form() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        registry.register_remote(RemoteSpawner::new("http://127.0.0.1:9"));
        registry.register_subprocess(SubprocessConfig::new("agent"));
        assert_eq!(
            registry.backends(),
            vec![
                SpawnBackend::Subprocess,
                SpawnBackend::InProcess,
                SpawnBackend::RemoteHttp
            ]
        );
        let rendered = format!("{registry:?}");
        assert!(rendered.contains("SpawnerRegistry"), "{rendered}");
        assert!(rendered.contains("Subprocess"), "{rendered}");
        let deferred = DeferredSpawner::for_backend(SpawnBackend::RemoteMq);
        assert_eq!(deferred.backend_name(), SpawnBackend::RemoteMq);
        assert!(format!("{deferred:?}").contains("RemoteMq"));
    }

    #[tokio::test]
    async fn deferred_backends_name_the_missing_transport() {
        let registry = SpawnerRegistry::with_deferred();
        assert_eq!(registry.len(), 2);

        let mq = registry
            .spawn(SpawnSpec::default().with_backend(SpawnBackend::RemoteMq))
            .await
            .err()
            .expect("the MQ transport is deferred");
        assert_eq!(mq.kind(), ErrorKind::Unsupported);
        assert!(mq.to_string().contains("Pulsar"), "{mq}");
        assert!(mq.to_string().contains("reply-topic"), "{mq}");

        let a2a = registry
            .spawn(SpawnSpec::default().with_backend(SpawnBackend::RemoteA2a))
            .await
            .err()
            .expect("the A2A transport is deferred");
        assert_eq!(a2a.kind(), ErrorKind::Unsupported);
        assert!(a2a.to_string().contains("a2a"), "{a2a}");
        assert!(a2a.to_string().contains("openjiuwen-extensions"), "{a2a}");
    }

    #[tokio::test]
    async fn a_named_unregistered_backend_names_its_registration_method() {
        let registry = SpawnerRegistry::new();
        let error = registry
            .spawn(SpawnSpec::default().with_backend(SpawnBackend::ExternalCommand))
            .await
            .err()
            .expect("nothing is registered");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error
                .to_string()
                .contains("no external_command spawner registered"),
            "{error}"
        );
        assert!(
            error.to_string().contains("register_external_command"),
            "{error}"
        );

        let message = deferred_message(Arc::new(DeferredSpawner::for_backend(
            SpawnBackend::Unknown,
        )))
        .await;
        assert!(message.contains("spawn backend `unknown`"), "{message}");
    }

    #[test]
    fn a_sole_spawner_needs_no_default() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        let spawner = registry.spawner_for(&SpawnSpec::default());
        assert_eq!(spawner.backend(), SpawnBackend::InProcess);
    }

    #[tokio::test]
    async fn an_ambiguous_registry_names_the_routing_knobs() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        registry.register_remote(RemoteSpawner::new("http://127.0.0.1:9"));
        let error = registry
            .spawn(SpawnSpec::default())
            .await
            .err()
            .expect("two backends and no default is ambiguous");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("ambiguous spawn backend"),
            "{error}"
        );
        assert!(error.to_string().contains("set_default"), "{error}");
        assert!(error.to_string().contains("SpawnSpec::backend"), "{error}");
    }

    #[tokio::test]
    async fn an_empty_registry_reports_no_default() {
        let registry = SpawnerRegistry::new();
        let error = registry
            .spawn(SpawnSpec::default())
            .await
            .err()
            .expect("nothing is registered");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("no default spawn backend"),
            "{error}"
        );
        assert!(error.to_string().contains("set_default"), "{error}");
        assert_eq!(
            registry.spawner_for(&SpawnSpec::default()).backend(),
            SpawnBackend::Unknown
        );
    }

    #[test]
    fn the_default_wins_over_the_sole_spawner_and_can_be_cleared() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        registry.register_remote(RemoteSpawner::new("http://127.0.0.1:9"));
        assert_eq!(registry.default_backend(), None);

        registry.set_default(SpawnBackend::RemoteHttp).unwrap();
        assert_eq!(registry.default_backend(), Some(SpawnBackend::RemoteHttp));
        assert_eq!(
            registry.spawner_for(&SpawnSpec::default()).backend(),
            SpawnBackend::RemoteHttp
        );

        // A spec's own hint beats the default.
        assert_eq!(
            registry
                .spawner_for(&SpawnSpec::default().with_backend(SpawnBackend::InProcess))
                .backend(),
            SpawnBackend::InProcess
        );

        registry.unregister(SpawnBackend::RemoteHttp);
        assert_eq!(registry.default_backend(), None);
    }

    #[test]
    fn set_default_refuses_an_unregistered_backend() {
        let registry = SpawnerRegistry::new();
        let error = registry.set_default(SpawnBackend::Subprocess).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("subprocess"), "{error}");
        assert_eq!(registry.default_backend(), None);
    }

    #[tokio::test]
    async fn spawn_with_never_falls_back() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        let error = registry
            .spawn_with(SpawnBackend::Subprocess, SpawnSpec::default())
            .await
            .err()
            .expect("no subprocess spawner is registered");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error
                .to_string()
                .contains("no subprocess spawner registered"),
            "{error}"
        );
        assert!(error.to_string().contains("register_subprocess"), "{error}");
    }

    #[tokio::test]
    async fn spawn_with_uses_the_named_backend() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        let agent = registry
            .spawn_with(SpawnBackend::InProcess, SpawnSpec::default())
            .await
            .unwrap();
        assert_eq!(agent.backend(), SpawnBackend::InProcess);
        let output = agent.send(json!({"query": "hi"})).await.unwrap();
        assert_eq!(output.output, "echo: {\"query\":\"hi\"}");
    }

    #[tokio::test]
    async fn spawn_routes_by_the_spec_hint() {
        let registry = SpawnerRegistry::new();
        registry.register_in_process(in_process());
        let agent = registry
            .spawn(SpawnSpec::default().with_backend(SpawnBackend::InProcess))
            .await
            .unwrap();
        assert_eq!(agent.backend(), SpawnBackend::InProcess);
        assert!(agent.id().starts_with("in-process-"));
    }
}
