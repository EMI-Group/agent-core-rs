//! The spawn facade of [`Runner`]: backend installation, registry routing,
//! introspection, and the monitor observation of every spawned child.
//!
//! A runner spawns through one of two seams:
//!
//! - [`Runner::set_spawner`] installs a single [`AgentSpawner`]; its `backend()`
//!   is [`Runner::spawn_backend`], and every [`Runner::spawn_agent`] goes there.
//! - [`Runner::set_spawner_registry`] installs a [`SpawnerRegistry`], which hosts
//!   several backends and routes a [`SpawnSpec`] by its `backend` hint, then the
//!   registry default, then a sole registered spawner. A registry takes
//!   precedence over the single spawner; the last installation wins, so
//!   [`Runner::set_spawner`] clears a registry and
//!   [`Runner::set_spawner_registry`] leaves the single spawner in place.
//!
//! Both entry points wrap a successful child in
//! [`MonitoredSpawnedAgent::track`], so [`Runner::active_spawns`] (and
//! [`Runner::monitor`]) report it as a [`RunKind::Spawn`] run from admission until
//! the run is reported finished — a health probe that finds the child gone, a
//! `shutdown`, a `force_kill`, or a drop of the handle. A failed spawn creates no
//! monitor entry.
//!
//! A [`SpawnSpec::backend`] hint is a registry address. With a
//! [`SpawnerRegistry`] installed it selects the backend (see
//! [`SpawnerRegistry::route`]); with only a single spawner it can only confirm
//! that spawner, so a hint naming a different backend is
//! [`openjiuwen_common::error::Error::Config`] instead of a silent misroute.

use crate::monitor::{RunKind, RunStatus};
use crate::runner::Runner;
use crate::spawn::{
    AgentSpawner, MonitoredSpawnedAgent, SpawnBackend, SpawnSpec, SpawnedAgent, SpawnerRegistry,
};
use openjiuwen_common::error::{Error, Result};
use serde_json::json;
use std::sync::Arc;

impl Runner {
    // -------------------------------------------------------------- spawn --

    /// The single spawn backend installed with [`Runner::set_spawner`].
    ///
    /// The default is [`crate::spawn::UnsupportedSpawner`], which reports
    /// [`SpawnBackend::Unknown`] and fails every spawn with
    /// [`openjiuwen_common::error::Error::Unsupported`]. A registry installed with
    /// [`Runner::set_spawner_registry`] does not replace this handle; routing for
    /// [`Runner::spawn_agent`] just takes precedence over it.
    #[must_use]
    pub fn spawner(&self) -> Arc<dyn AgentSpawner> {
        Arc::clone(&self.spawner.read())
    }

    /// The backend the single installed spawner launches.
    ///
    /// [`SpawnBackend::Unknown`] when a spawner does not name one — which is what
    /// the default `UnsupportedSpawner` reports. Use [`Runner::spawn_backends`]
    /// for the full routing set, including a registry's backends.
    #[must_use]
    pub fn spawn_backend(&self) -> SpawnBackend {
        self.spawner().backend()
    }

    /// Installs `registry` as the runner's multi-backend spawn host.
    ///
    /// Takes precedence over the single spawner for [`Runner::spawn_agent`]
    /// routing. The single spawner stays installed — [`Runner::spawn_backend`]
    /// keeps reporting it — until a later [`Runner::set_spawner`], which clears
    /// the registry (the last installation wins).
    pub fn set_spawner_registry(&self, registry: Arc<SpawnerRegistry>) {
        *self.spawner_registry.write() = Some(registry);
    }

    /// The installed [`SpawnerRegistry`], when a host installed one.
    #[must_use]
    pub fn spawner_registry(&self) -> Option<Arc<SpawnerRegistry>> {
        self.spawner_registry.read().clone()
    }

    /// Every backend this runner can route a spawn to.
    ///
    /// With a registry, its registered backends (sorted); otherwise exactly
    /// [`Runner::spawn_backend`].
    #[must_use]
    pub fn spawn_backends(&self) -> Vec<SpawnBackend> {
        self.spawner_registry().map_or_else(
            || vec![self.spawn_backend()],
            |registry| registry.backends(),
        )
    }

    /// Installs the single spawn backend [`Runner::spawn_agent`] delegates to.
    ///
    /// Clears any registry installed with [`Runner::set_spawner_registry`]: the
    /// last installation wins. The default
    /// [`crate::spawn::UnsupportedSpawner`] keeps its existing wording when a host
    /// never installs anything.
    pub fn set_spawner(&self, spawner: Arc<dyn AgentSpawner>) {
        *self.spawner.write() = spawner;
        *self.spawner_registry.write() = None;
    }

    /// Asks the installed spawner — or the registry a spec routes to — to launch
    /// the agent described by `spec`.
    ///
    /// With a registry installed the spec goes through [`SpawnerRegistry::route`],
    /// so its `backend` hint, the registry's default, and the sole-spawner rule
    /// apply, and the run is labelled with the *resolved* backend key. Without a
    /// registry the spec goes to the single installed spawner, and a
    /// [`SpawnSpec::backend`] hint must agree with that spawner: a contradiction is
    /// [`openjiuwen_common::error::Error::Config`], and an installed spawner that
    /// reports [`SpawnBackend::Unknown`] cannot contradict the hint, so the hint
    /// labels the run. The default [`crate::spawn::UnsupportedSpawner`] answers
    /// [`openjiuwen_common::error::Error::Unsupported`] naming the seam.
    ///
    /// A successful child is admitted into [`Runner::monitor`] as a
    /// [`RunKind::Spawn`] run bound to `spec.session_id` (see
    /// [`Runner::active_spawns`]); a failed spawn creates no entry.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] when a `spec.backend` hint
    /// contradicts the single installed spawner (install a [`SpawnerRegistry`]
    /// with [`Runner::set_spawner_registry`] or address one with
    /// [`Runner::spawn_agent_with`]), plus the resolved spawner's error (by default
    /// `Error::Unsupported`, or the registry's deferred-backend wording).
    pub async fn spawn_agent(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let session_id = spec.session_id.clone();
        let (backend, spawner) = if let Some(registry) = self.spawner_registry() {
            registry.route(&spec)
        } else {
            let spawner = self.spawner();
            let backend = single_spawn_backend(&spec, &spawner)?;
            (backend, spawner)
        };
        let child = spawner.spawn(spec).await?;
        Ok(self.observed_spawn(child, backend, session_id.as_deref()))
    }

    /// Spawns `spec` through exactly `backend` in the installed registry.
    ///
    /// Unlike [`Runner::spawn_agent`] this never falls back: a backend the
    /// registry does not host is the registry's deferred-spawner failure naming
    /// it, and no registry at all is a configuration error.
    ///
    /// A successful child is admitted into [`Runner::monitor`] as a
    /// [`RunKind::Spawn`] run under `backend`; a failed spawn creates no entry.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] naming
    /// [`Runner::set_spawner_registry`] when no registry is installed, plus the
    /// registry's own error (`Error::Unsupported` for an unregistered or deferred
    /// backend).
    pub async fn spawn_agent_with(
        &self,
        backend: SpawnBackend,
        spec: SpawnSpec,
    ) -> Result<Arc<dyn SpawnedAgent>> {
        let Some(registry) = self.spawner_registry() else {
            return Err(Error::config(
                "no spawner registry is installed; install one with \
                 `Runner::set_spawner_registry` before calling `Runner::spawn_agent_with`",
            ));
        };
        let session_id = spec.session_id.clone();
        let child = registry.spawn_with(backend, spec).await?;
        Ok(self.observed_spawn(child, backend, session_id.as_deref()))
    }

    /// The spawned children this runner has not reported finished: every
    /// [`RunKind::Spawn`] run admitted here whose end has not been observed.
    ///
    /// "Finished" is an observation, not a live check: a run leaves this list when
    /// a `health` probe finds the child gone, when it is shut down or killed, or
    /// when its handle is dropped. A child that dies without any of those still
    /// appears here until something probes it — this method does not poll the
    /// backends, so an idle dead child is listed until the host asks.
    ///
    /// A convenience over [`RunMonitor::list_active`](crate::monitor::RunMonitor::list_active)
    /// filtered to [`RunKind::Spawn`]; finished spawns stay readable through
    /// [`Runner::monitor`]'s `snapshot` until the monitor's retention bound evicts
    /// them.
    #[must_use]
    pub fn active_spawns(&self) -> Vec<RunStatus> {
        self.monitor()
            .list_active()
            .into_iter()
            .filter(|status| status.kind == RunKind::Spawn)
            .collect()
    }

    // ----------------------------------------------------------- internals --

    /// Admits a freshly spawned child into the monitor and wraps it.
    ///
    /// Called only after a backend returned a child, so a failed spawn never
    /// reaches the monitor. The detail records the resolved `backend` and the
    /// child's `pid` when it has one.
    fn observed_spawn(
        &self,
        child: Arc<dyn SpawnedAgent>,
        backend: SpawnBackend,
        session_id: Option<&str>,
    ) -> Arc<dyn SpawnedAgent> {
        let detail = json!({"backend": backend.as_str(), "pid": child.pid()});
        Arc::new(MonitoredSpawnedAgent::track(
            child,
            self.monitor(),
            session_id,
            detail,
        ))
    }
}

/// The backend a spec resolves to when only a single spawner is installed.
///
/// One spawner can only serve its own backend, so a [`SpawnSpec::backend`] hint is
/// honoured only when it agrees with [`AgentSpawner::backend`]; a contradiction is
/// [`Error::Config`] naming both sides rather than a silent misroute. An installed
/// spawner that reports [`SpawnBackend::Unknown`] names no backend, so it cannot
/// contradict the hint — the hint is kept and labels the run.
///
/// # Errors
///
/// [`Error::Config`] when the hint names a backend the installed spawner does not
/// launch; the message names both backends and both remedies.
fn single_spawn_backend(spec: &SpawnSpec, spawner: &Arc<dyn AgentSpawner>) -> Result<SpawnBackend> {
    let installed = spawner.backend();
    let Some(requested) = spec.backend else {
        return Ok(installed);
    };
    if installed == SpawnBackend::Unknown || installed == requested {
        return Ok(requested);
    }
    Err(Error::config(format!(
        "spawn spec requests backend '{requested}' but the installed single spawner launches \
         '{installed}'; install a `SpawnerRegistry` with `Runner::set_spawner_registry` to host \
         several backends, or address a registry backend with `Runner::spawn_agent_with`"
    )))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use crate::monitor::RunPhase;
    use crate::spawn::SpawnHealth;
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use openjiuwen_agent::output::{AgentChunk, AgentOutput};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// A deterministic child handle reporting a fixed identity and backend.
    struct FakeChild {
        /// Stable handle id, unique per spawn.
        id: String,
        /// Backend the handle reports.
        backend: SpawnBackend,
    }

    #[async_trait]
    impl SpawnedAgent for FakeChild {
        fn id(&self) -> &str {
            &self.id
        }

        fn is_alive(&self) -> bool {
            true
        }

        fn pid(&self) -> Option<u32> {
            Some(4242)
        }

        async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
            Ok(AgentOutput {
                output: "spawned".to_owned(),
                ..AgentOutput::default()
            })
        }

        fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }

        async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
            Ok(true)
        }

        async fn force_kill(&self) -> Result<()> {
            Ok(())
        }

        fn backend(&self) -> SpawnBackend {
            self.backend
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn health_snapshot(&self) -> SpawnHealth {
            SpawnHealth::healthy(self.backend, self.pid())
        }
    }

    /// A spawner that counts its spawns and hands out a fresh [`FakeChild`].
    struct FakeSpawner {
        /// Backend this spawner names.
        backend: SpawnBackend,
        /// Id prefix of the children it hands out.
        id_prefix: &'static str,
        /// How many times `spawn` was called.
        calls: Arc<AtomicUsize>,
    }

    impl FakeSpawner {
        /// A spawner naming `backend` with its own call counter.
        fn new(backend: SpawnBackend, id_prefix: &'static str) -> Self {
            Self {
                backend,
                id_prefix,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl AgentSpawner for FakeSpawner {
        async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakeChild {
                id: format!("{}-{call}", self.id_prefix),
                backend: self.backend,
            }))
        }

        fn backend(&self) -> SpawnBackend {
            self.backend
        }
    }

    /// A spawner whose every spawn fails.
    struct FailingSpawner;

    #[async_trait]
    impl AgentSpawner for FailingSpawner {
        async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
            Err(Error::transport("launch failed"))
        }
    }

    /// Re-types a concrete fake spawner as the trait object the seams take.
    fn as_spawner(spawner: &Arc<FakeSpawner>) -> Arc<dyn AgentSpawner> {
        Arc::<FakeSpawner>::clone(spawner)
    }

    /// A registry hosting one in-process fake and one subprocess fake.
    fn registry(
        in_process: &Arc<FakeSpawner>,
        subprocess: &Arc<FakeSpawner>,
    ) -> Arc<SpawnerRegistry> {
        let registry = Arc::new(SpawnerRegistry::new());
        registry.register(SpawnBackend::InProcess, as_spawner(in_process));
        registry.register(SpawnBackend::Subprocess, as_spawner(subprocess));
        registry
    }

    #[test]
    fn spawner_and_spawn_backend_reflect_set_spawner() {
        let runner = Runner::new();
        assert_eq!(runner.spawn_backend(), SpawnBackend::Unknown);
        assert_eq!(runner.spawn_backends(), vec![SpawnBackend::Unknown]);
        assert!(runner.spawner_registry().is_none());

        runner.set_spawner(Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in")));
        assert_eq!(runner.spawn_backend(), SpawnBackend::InProcess);
        assert_eq!(runner.spawner().backend(), SpawnBackend::InProcess);
        assert_eq!(runner.spawn_backends(), vec![SpawnBackend::InProcess]);
    }

    #[test]
    fn spawn_backends_lists_the_sorted_registry_backends() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let runner = Runner::new();
        runner.set_spawner(Arc::new(FakeSpawner::new(SpawnBackend::RemoteHttp, "http")));
        runner.set_spawner_registry(registry(&in_process, &subprocess));

        assert_eq!(
            runner.spawn_backends(),
            vec![SpawnBackend::Subprocess, SpawnBackend::InProcess],
            "registry order is sorted"
        );
        assert_eq!(
            runner.spawn_backend(),
            SpawnBackend::RemoteHttp,
            "the single spawner stays installed beside the registry"
        );
        assert!(runner.spawner_registry().is_some());
    }

    #[test]
    fn set_spawner_clears_the_registry() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let runner = Runner::new();
        runner.set_spawner_registry(registry(&in_process, &subprocess));
        assert!(runner.spawner_registry().is_some());

        runner.set_spawner(Arc::new(FakeSpawner::new(
            SpawnBackend::ExternalCommand,
            "cli",
        )));
        assert!(runner.spawner_registry().is_none());
        assert_eq!(runner.spawn_backends(), vec![SpawnBackend::ExternalCommand]);
    }

    #[tokio::test]
    async fn the_spec_backend_routes_through_the_registry() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let single = Arc::new(FakeSpawner::new(SpawnBackend::RemoteHttp, "http"));
        let runner = Runner::new();
        runner.set_spawner(as_spawner(&single));
        runner.set_spawner_registry(registry(&in_process, &subprocess));

        let child = runner
            .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::Subprocess))
            .await
            .unwrap();
        assert_eq!(subprocess.calls.load(Ordering::SeqCst), 1);
        assert_eq!(in_process.calls.load(Ordering::SeqCst), 0);
        assert_eq!(single.calls.load(Ordering::SeqCst), 0, "the registry wins");
        assert_eq!(child.backend(), SpawnBackend::Subprocess);
        assert_eq!(child.id(), "sub-0");
    }

    #[tokio::test]
    async fn spawn_agent_without_a_registry_uses_the_single_spawner() {
        let single = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let runner = Runner::new();
        runner.set_spawner(as_spawner(&single));

        let child = runner.spawn_agent(SpawnSpec::default()).await.unwrap();
        assert_eq!(single.calls.load(Ordering::SeqCst), 1);
        assert_eq!(child.backend(), SpawnBackend::InProcess);
        assert_eq!(child.id(), "in-0");
    }

    #[tokio::test]
    async fn spawn_agent_without_a_registry_is_a_config_error() {
        let runner = Runner::new();
        let error = runner
            .spawn_agent_with(SpawnBackend::InProcess, SpawnSpec::default())
            .await
            .err()
            .expect("a registry is required");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(
            error.to_string().contains("set_spawner_registry"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_successful_spawn_is_listed_and_its_shutdown_clears_it() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let runner = Runner::new();
        runner.set_spawner_registry(registry(&in_process, &subprocess));

        let child = runner
            .spawn_agent_with(
                SpawnBackend::InProcess,
                SpawnSpec::default().with_session_id("s-1"),
            )
            .await
            .unwrap();

        let active = runner.active_spawns();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].run_id, child.id());
        assert_eq!(active[0].kind, RunKind::Spawn);
        assert_eq!(active[0].phase, RunPhase::Running);
        assert_eq!(active[0].session_id.as_deref(), Some("s-1"));
        assert_eq!(active[0].detail["backend"], json!("in_process"));
        assert_eq!(active[0].detail["pid"], json!(4242));

        assert!(child.shutdown(Duration::from_secs(1)).await.unwrap());
        assert!(runner.active_spawns().is_empty());
        let finished = runner.monitor().snapshot(child.id()).unwrap();
        assert_eq!(finished.phase, RunPhase::Completed);
        assert_eq!(finished.detail, json!({"graceful": true}));
    }

    #[tokio::test]
    async fn dropping_a_spawn_handle_clears_it_from_active_spawns() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let runner = Runner::new();
        runner.set_spawner_registry(registry(&in_process, &subprocess));

        let child = runner
            .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::InProcess))
            .await
            .unwrap();
        let run_id = child.id().to_owned();
        assert_eq!(runner.active_spawns().len(), 1);

        drop(child);
        assert!(runner.active_spawns().is_empty());
        let finished = runner.monitor().snapshot(&run_id).unwrap();
        assert_eq!(finished.phase, RunPhase::Completed);
        assert_eq!(finished.detail, json!({"reason": "handle_dropped"}));
        assert_eq!(
            runner.monitor().history(&run_id).unwrap().len(),
            3,
            "queued, running, dropped"
        );
    }

    #[tokio::test]
    async fn a_deferred_backend_is_unsupported_and_names_pulsar() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let runner = Runner::new();
        runner.set_spawner_registry(registry(&in_process, &subprocess));

        let error = runner
            .spawn_agent_with(SpawnBackend::RemoteMq, SpawnSpec::default())
            .await
            .err()
            .expect("the MQ transport is deferred");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("Pulsar"), "{error}");
        assert!(runner.active_spawns().is_empty());
    }

    #[tokio::test]
    async fn a_failed_spawn_creates_no_active_spawn() {
        let runner = Runner::new();
        runner.set_spawner(Arc::new(FailingSpawner));

        let error = runner
            .spawn_agent(SpawnSpec::default())
            .await
            .err()
            .expect("the spawner fails");
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(runner.active_spawns().is_empty());
        assert!(runner.monitor().list_active().is_empty());
    }

    #[tokio::test]
    async fn a_finished_spawn_is_retained_as_its_terminal_outcome() {
        let in_process = Arc::new(FakeSpawner::new(SpawnBackend::InProcess, "in"));
        let subprocess = Arc::new(FakeSpawner::new(SpawnBackend::Subprocess, "sub"));
        let runner = Runner::new();
        runner.set_spawner_registry(registry(&in_process, &subprocess));

        let child = runner
            .spawn_agent_with(SpawnBackend::Subprocess, SpawnSpec::default())
            .await
            .unwrap();
        child.force_kill().await.unwrap();
        let status = runner.monitor().snapshot(child.id()).unwrap();
        assert_eq!(status.phase, RunPhase::Cancelled);
        assert_eq!(status.detail, json!({"reason": "force_kill"}));
    }
}
