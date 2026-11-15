//! Sandbox lifecycle: one `SandboxRecord` per isolation key plus the launcher calls that
//! keep it running.
//!
//! Python parity: `sandbox/gateway/gateway.py` (`SandboxGateway._get_endpoint`,
//! `_create_new_sandbox`, `_evict_idle`, `release_sandbox`, `pause_sandbox`,
//! `delete_sandbox`) and `sandbox/gateway/sandbox_store.py` (`SandboxRecord`,
//! `AbstractSandboxStore`, `InMemorySandboxStore`). The provider cache of Python's gateway
//! is not ported: providers are rebuilt from the resolved endpoint per
//! [`crate::SysOperation`], so only the container bookkeeping lives here — including the
//! `container_config_hash` and `metadata` fields that key that cache in Python (see
//! [`refresh_container_config`]).
//!
//! # State machine
//!
//! A record is `running` after a successful launch or resume, `paused` while the launcher
//! suspended it, and removed when [`SandboxLifecycle::release`] or
//! [`SandboxLifecycle::evict_idle`] destroys the container:
//!
//! ```text
//!            launch ok              resume ok
//!   (none) -------------> running ------------> running
//!                            |  ^                   |
//!      release(Pause)        |  |                   | release(Delete)
//!                            v  |                   v
//!                          paused                (removed)
//!                            |
//!                            +-- evict_idle / release(Delete) --> (removed)
//! ```
//!
//! [`SandboxLifecycle::release`] with [`OnStop::Pause`] keeps the record in the `paused`
//! state so the next [`SandboxLifecycle::ensure_endpoint`] resumes the same container, and
//! [`OnStop::Keep`] leaves a running record untouched: the container is externally managed,
//! so it stays usable.
//!
//! A record whose container disappeared meanwhile (`check_status` reports
//! [`SandboxStatus::Stopped`]) is dropped and relaunched, mirroring Python's
//! `SandboxGateway._get_endpoint`.
//!
//! # Failures
//!
//! A launcher that fails while doing its job is code 199008 ([`ErrorKind::SysOp`]). A
//! launcher that cannot perform the operation *at all* keeps its own
//! [`ErrorKind::Unsupported`] — the defaulted `pause`/`resume`/`delete` of
//! [`crate::sandbox::launcher::SandboxLauncher`] — and a launcher that rejects the request as
//! invalid keeps its [`ErrorKind::Validation`]/[`ErrorKind::Config`]; see
//! `map_launcher_failure`. Teardown is never reported as a success the launcher did not
//! perform, and the record is restored on failure so the call is retryable.
//!
//! # Time
//!
//! Every decision takes `now_epoch_secs` as an argument (or the injected [`Clock`]), so
//! idle-stop is deterministic in tests. Only [`SandboxLifecycle::new`] installs the
//! wall-clock [`SystemClock`]; use [`SandboxLifecycle::with_clock`] and [`FixedClock`] to
//! drive time by hand.
//!
//! # Concurrency
//!
//! One `tokio::sync::Mutex` guards the record map. The lock is never held across a
//! launcher call: the code clones what it needs, awaits the launcher, then re-locks. If two
//! callers launch the same isolation key concurrently, the first record wins and the loser
//! deletes the container it just created.
//!
//! # Layout
//!
//! `record` holds the bookkeeping struct plus the config hash/metadata helpers, `clock` the
//! injected time sources, and this module the [`SandboxLifecycle`] manager plus [`global`].
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_sysop::sandbox::launcher::{LauncherRegistry, LaunchedSandbox, SandboxLauncher};
//! use openjiuwen_sysop::sandbox::config::{SandboxLauncherConfig, SandboxGatewayConfig, OnStop};
//! use openjiuwen_sysop::sandbox::lifecycle::SandboxLifecycle;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! struct PreDeployed;
//!
//! #[async_trait::async_trait]
//! impl SandboxLauncher for PreDeployed {
//!     async fn launch(
//!         &self,
//!         _config: &SandboxLauncherConfig,
//!         _timeout_seconds: u64,
//!         _isolation_key: Option<&str>,
//!     ) -> openjiuwen_common::error::Result<LaunchedSandbox> {
//!         Ok(LaunchedSandbox::new("http://127.0.0.1:9000").with_sandbox_id("c-1"))
//!     }
//!
//!     // A launcher that can destroy its container must say so, or `release(Delete)`
//!     // reports the inherited `Unsupported` refusal instead of pretending success.
//!     async fn delete(&self, _sandbox_id: &str) -> openjiuwen_common::error::Result<()> {
//!         Ok(())
//!     }
//! }
//!
//! let launchers = LauncherRegistry::new();
//! launchers.register("fake", Arc::new(PreDeployed))?;
//! let lifecycle = SandboxLifecycle::new(launchers);
//! let config = SandboxGatewayConfig {
//!     launcher_config: Some(SandboxLauncherConfig {
//!         launcher_type: "fake".to_string(),
//!         ..SandboxLauncherConfig::default()
//!     }),
//!     ..SandboxGatewayConfig::default()
//! };
//!
//! let endpoint = lifecycle.ensure_endpoint(&config, "session_key", 1_000).await?;
//! assert_eq!(endpoint.base_url, "http://127.0.0.1:9000");
//! assert_eq!(endpoint.sandbox_id.as_deref(), Some("c-1"));
//!
//! // The record carries the hash and labels of the launcher configuration it was made with.
//! let record = lifecycle.record("session_key").await.unwrap();
//! assert_eq!(
//!     record.container_config_hash,
//!     openjiuwen_sysop::sandbox::lifecycle::container_config_hash(
//!         config.launcher_config.as_ref().unwrap()
//!     )
//! );
//! assert_eq!(record.metadata["launcher_type"], "fake");
//!
//! // Reused, not relaunched, on the next call.
//! let again = lifecycle.ensure_endpoint(&config, "session_key", 1_010).await?;
//! assert_eq!(again, endpoint);
//!
//! lifecycle.release("session_key", OnStop::Delete).await?;
//! assert!(lifecycle.records().await.is_empty());
//! # Ok(())
//! # }
//! ```

mod clock;
mod record;

pub use clock::{now_epoch_secs, Clock, FixedClock, SystemClock};
pub use record::{
    container_config_hash, container_metadata, refresh_container_config, SandboxRecord,
};

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, OnceLock},
};

use openjiuwen_common::error::{Error, ErrorKind, Result};
use tokio::sync::Mutex;

use crate::sandbox::config::{
    OnStop, SandboxEndpoint, SandboxGatewayConfig, SandboxLauncherConfig,
};
use crate::sandbox::launcher::{
    isolation_key_error, launcher_error, LauncherRegistry, SandboxLauncher, SandboxStatus,
};

/// Maps a launcher failure onto the lifecycle's error codes.
///
/// A launcher that exists and fails while doing its job is code 199008
/// ([`launcher_error`], [`ErrorKind::SysOp`]). Kinds that describe what the launcher *is*
/// rather than how a call went are passed through unchanged: [`ErrorKind::Unsupported`] (the
/// launcher cannot do this at all, for example destroying a pre-deployed container),
/// [`ErrorKind::Validation`] and [`ErrorKind::Config`] (the launcher rejects the request as
/// invalid). Flattening those into 199008 is what made a refused teardown look like a
/// transient failure; keeping them typed lets a caller tell "this runtime cannot do that"
/// from "the call failed, retry may help".
fn map_launcher_failure(process: &str, error: Error) -> Error {
    match error.kind() {
        ErrorKind::Unsupported | ErrorKind::Validation | ErrorKind::Config => error,
        _ => launcher_error(process, error.message().as_ref()),
    }
}

/// The lifecycle manager: resolves, reuses, pauses, resumes, and evicts sandbox containers.
///
/// Python parity: the endpoint half of `SandboxGateway` (`_get_endpoint`,
/// `_create_new_sandbox`, `_evict_idle`) over `InMemorySandboxStore`. One instance per
/// process is the norm ([`global`]); [`SandboxLifecycle::new`] builds an isolated one for a
/// test or a host-owned pool.
#[derive(Clone)]
pub struct SandboxLifecycle {
    launchers: LauncherRegistry,
    clock: Arc<dyn Clock>,
    records: Arc<Mutex<BTreeMap<String, SandboxRecord>>>,
}

impl fmt::Debug for SandboxLifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Records carry endpoints, which may embed credentials: only keys and states show.
        f.debug_struct("SandboxLifecycle")
            .field("launchers", &self.launchers.launcher_types())
            .field(
                "records",
                &self
                    .records
                    .try_lock()
                    .map(|records| {
                        records
                            .iter()
                            .map(|(key, record)| (key.clone(), record.status))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            )
            .finish_non_exhaustive()
    }
}

impl SandboxLifecycle {
    /// A lifecycle over `launchers`, driven by the wall clock.
    #[must_use]
    pub fn new(launchers: LauncherRegistry) -> Self {
        Self::with_clock(launchers, Arc::new(SystemClock))
    }

    /// A lifecycle over `launchers`, driven by `clock`.
    #[must_use]
    pub fn with_clock(launchers: LauncherRegistry, clock: Arc<dyn Clock>) -> Self {
        Self {
            launchers,
            clock,
            records: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// The launcher registry this lifecycle launches through.
    #[must_use]
    pub const fn launchers(&self) -> &LauncherRegistry {
        &self.launchers
    }

    /// The current time reported by the injected clock.
    #[must_use]
    pub fn now(&self) -> u64 {
        self.clock.now_epoch_secs()
    }

    /// The record stored for `isolation_key`, if any.
    pub async fn record(&self, isolation_key: &str) -> Option<SandboxRecord> {
        self.records.lock().await.get(isolation_key).cloned()
    }

    /// Every record, ordered by isolation key.
    pub async fn records(&self) -> Vec<SandboxRecord> {
        self.records.lock().await.values().cloned().collect()
    }

    /// The number of tracked containers.
    pub async fn len(&self) -> usize {
        self.records.lock().await.len()
    }

    /// Whether no container is tracked.
    pub async fn is_empty(&self) -> bool {
        self.records.lock().await.is_empty()
    }

    /// Resolves the endpoint an operation on `isolation_key` must target.
    ///
    /// A `running` record is reused and its `last_used_epoch_secs` refreshed. A `paused`
    /// record is resumed through its launcher. Anything else evicts the idle containers
    /// (when `launcher_config.idle_ttl_seconds` is set) and launches a new sandbox, storing
    /// a fresh record.
    ///
    /// # Errors
    ///
    /// - `ErrorKind::SysOp` (199008) when the launcher fails to launch or resume;
    /// - `ErrorKind::SysOp` (199010) when `isolation_key` is blank;
    /// - `ErrorKind::SysOp` (199002) when the configuration has no `launcher_config`;
    /// - `ErrorKind::Unsupported` when no launcher is registered for the configured
    ///   `launcher_type`.
    pub async fn ensure_endpoint(
        &self,
        config: &SandboxGatewayConfig,
        isolation_key: &str,
        now_epoch_secs: u64,
    ) -> Result<SandboxEndpoint> {
        let launcher_config = config.launcher_config.as_ref().ok_or_else(|| {
            crate::card::card_param_error("launcher_config is required when mode is 'sandbox'")
        })?;
        if isolation_key.trim().is_empty() {
            return Err(isolation_key_error(
                "isolation key must not be empty when resolving a sandbox endpoint",
            ));
        }
        // A launcher that exists but fails is 199008; only an unregistered type is
        // `Unsupported`, so the container runtime being absent stays distinguishable.
        let launcher = self.launchers.create(&launcher_config.launcher_type)?;

        if let Some(endpoint) = self
            .revive(&launcher, launcher_config, isolation_key, now_epoch_secs)
            .await?
        {
            return Ok(endpoint);
        }
        self.launch_container(
            &launcher,
            config,
            launcher_config,
            isolation_key,
            now_epoch_secs,
        )
        .await
    }

    /// Steps 1 and 2 of [`Self::ensure_endpoint`]: reuse a running container, or revive a
    /// recorded one.
    ///
    /// A paused container is resumed; one that is gone is forgotten so the caller launches a
    /// fresh sandbox. Both reuse paths merge `launcher_config`'s derived bookkeeping into the
    /// record once the launcher call has succeeded (see [`refresh_container_config`]), so a
    /// failure never loses metadata.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (199008) when the launcher fails to resume the container
    /// or cannot report its status, and passes a launcher's own
    /// [`ErrorKind::Unsupported`]/[`ErrorKind::Validation`]/[`ErrorKind::Config`] through
    /// unchanged (see [`map_launcher_failure`]) — a launcher that cannot resume surfaces its
    /// typed refusal instead of a 199008.
    async fn revive(
        &self,
        launcher: &Arc<dyn SandboxLauncher>,
        launcher_config: &SandboxLauncherConfig,
        isolation_key: &str,
        now_epoch_secs: u64,
    ) -> Result<Option<SandboxEndpoint>> {
        // 1. Reuse a running container.
        if let Some(endpoint) = self
            .touch_running(isolation_key, launcher_config, now_epoch_secs)
            .await
        {
            return Ok(Some(endpoint));
        }
        // 2. A stored record that is not running: ask the launcher what the container is.
        let stored = self.record(isolation_key).await;
        let Some(record) = stored.filter(|record| !record.status.is_running()) else {
            return Ok(None);
        };
        let Some(sandbox_id) = record.sandbox_id else {
            tracing::debug!(
                isolation_key,
                "recorded sandbox has no id to resume; relaunching"
            );
            return Ok(None);
        };
        match launcher.check_status(&sandbox_id).await {
            Ok(SandboxStatus::Running) => Ok(self
                .mark_running(isolation_key, &sandbox_id, launcher_config, now_epoch_secs)
                .await),
            // Python only knows running/paused/killed; a launcher that cannot tell is
            // treated as paused, because it may hold a suspended container that a resume
            // revives.
            Ok(SandboxStatus::Paused | SandboxStatus::Unknown) => {
                launcher
                    .resume(&sandbox_id)
                    .await
                    .map_err(|error| map_launcher_failure("resume", error))?;
                Ok(self
                    .mark_running(isolation_key, &sandbox_id, launcher_config, now_epoch_secs)
                    .await)
            }
            Ok(SandboxStatus::Stopped) => {
                tracing::debug!(
                    isolation_key,
                    sandbox_id,
                    "recorded sandbox is gone; relaunching"
                );
                self.drop_record(isolation_key, &sandbox_id).await;
                Ok(None)
            }
            Err(error) => Err(map_launcher_failure("check_status", error)),
        }
    }

    /// Steps 3 and 4 of [`Self::ensure_endpoint`]: evict the idle containers, launch a new
    /// sandbox, and store its record.
    ///
    /// The stored record carries the configuration hash and the labels derived from
    /// `launcher_config` (see [`container_config_hash`] / [`container_metadata`]), so a reuse
    /// can tell whether the running container still matches the configured launcher.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (199008) when the launcher fails to launch the container.
    async fn launch_container(
        &self,
        launcher: &Arc<dyn SandboxLauncher>,
        config: &SandboxGatewayConfig,
        launcher_config: &SandboxLauncherConfig,
        isolation_key: &str,
        now_epoch_secs: u64,
    ) -> Result<SandboxEndpoint> {
        // 3. Evict idle containers, then launch a new one.
        if let Some(idle_ttl_seconds) = launcher_config.idle_ttl_seconds {
            let evicted = self.evict_idle(idle_ttl_seconds, now_epoch_secs).await;
            if !evicted.is_empty() {
                tracing::debug!(evicted = ?evicted, "evicted idle sandboxes before launch");
            }
        }
        let descriptor = launcher
            .launch(launcher_config, config.timeout_seconds, Some(isolation_key))
            .await
            .map_err(|error| map_launcher_failure("launch", error))?;

        // 4. Store the record. The lock is never held across a launcher await.
        let record = SandboxRecord {
            isolation_key: isolation_key.to_string(),
            sandbox_id: descriptor.sandbox_id.clone(),
            endpoint: SandboxEndpoint {
                base_url: descriptor.base_url.clone(),
                sandbox_id: descriptor.sandbox_id.clone(),
                isolation_key: Some(isolation_key.to_string()),
            },
            status: SandboxStatus::Running,
            launcher_type: launcher_config.launcher_type.clone(),
            created_epoch_secs: now_epoch_secs,
            last_used_epoch_secs: now_epoch_secs,
            container_config_hash: container_config_hash(launcher_config),
            metadata: container_metadata(launcher_config),
        };
        let winner = {
            let mut records = self.records.lock().await;
            match records.get(isolation_key) {
                // A concurrent caller stored a running record first: keep theirs.
                Some(existing) if existing.status == SandboxStatus::Running => {
                    Some(existing.endpoint.clone())
                }
                _ => {
                    records.insert(isolation_key.to_string(), record.clone());
                    None
                }
            }
        };
        let Some(endpoint) = winner else {
            return Ok(record.endpoint);
        };
        // Destroy the container this call launched so it is not leaked. A launcher that
        // reuses the isolation key as its container identity reports the winner's id, in
        // which case there is nothing extra to delete.
        if let Some(sandbox_id) = descriptor.sandbox_id.as_deref() {
            if Some(sandbox_id) == endpoint.sandbox_id.as_deref() {
                tracing::debug!(
                    isolation_key,
                    sandbox_id,
                    "duplicate launch reused the same sandbox id; nothing to delete"
                );
            } else if let Err(error) = launcher.delete(sandbox_id).await {
                if error.kind() == ErrorKind::Unsupported {
                    tracing::warn!(
                        isolation_key,
                        sandbox_id,
                        launcher_type = %launcher_config.launcher_type,
                        error = %error,
                        "the duplicate sandbox launched for this isolation key was left behind \
                         (leaked): its launcher cannot delete containers"
                    );
                } else {
                    tracing::warn!(
                        sandbox_id,
                        error = %error,
                        "failed to delete a duplicate sandbox launched for the same isolation key"
                    );
                }
            }
        } else {
            tracing::debug!(
                isolation_key,
                base_url = %descriptor.base_url,
                "duplicate launch had no sandbox id; nothing to delete"
            );
        }
        Ok(endpoint)
    }

    /// [`Self::ensure_endpoint`] using the injected clock.
    ///
    /// # Errors
    ///
    /// See [`Self::ensure_endpoint`].
    pub async fn ensure_endpoint_now(
        &self,
        config: &SandboxGatewayConfig,
        isolation_key: &str,
    ) -> Result<SandboxEndpoint> {
        self.ensure_endpoint(config, isolation_key, self.now())
            .await
    }

    /// Releases the container tracked under `isolation_key` according to `on_stop`.
    ///
    /// [`OnStop::Delete`] destroys the container and drops the record,
    /// [`OnStop::Pause`] suspends it and keeps the record in the `paused` state so the next
    /// [`Self::ensure_endpoint`] resumes the same container instead of creating another
    /// one, and [`OnStop::Keep`] leaves the container running with its record in place,
    /// refreshing `last_used_epoch_secs` so the next sweep does not evict it. An unknown key
    /// is a successful no-op (Python returns "Sandbox record not found"; releasing something
    /// that was never launched is not an error here).
    ///
    /// # Errors
    ///
    /// - `ErrorKind::SysOp` (199008) when the launcher fails the `delete`/`pause` call;
    /// - the launcher's own [`ErrorKind::Unsupported`] when it cannot destroy or suspend the
    ///   container at all: teardown is *not* reported as a success, because the container
    ///   would be left running (see `map_launcher_failure`). A `pre_deploy` container is
    ///   the built-in example;
    /// - `ErrorKind::Unsupported` when the recorded `launcher_type` is no longer registered.
    ///
    /// The record is restored on every failure, so the call can be retried without losing
    /// its bookkeeping (`container_config_hash`, `metadata`).
    pub async fn release(&self, isolation_key: &str, on_stop: OnStop) -> Result<()> {
        if on_stop == OnStop::Keep {
            let mut records = self.records.lock().await;
            if let Some(record) = records.get_mut(isolation_key) {
                record.last_used_epoch_secs = self.now();
            }
            return Ok(());
        }

        let Some(record) = self.take(isolation_key).await else {
            tracing::debug!(isolation_key, "release of an untracked sandbox is a no-op");
            return Ok(());
        };
        let launcher = match self.launchers.create(&record.launcher_type) {
            Ok(launcher) => launcher,
            Err(error) => {
                self.restore(record).await;
                return Err(error);
            }
        };
        let sandbox_id = record.sandbox_id.clone().unwrap_or_default();
        let outcome = if on_stop == OnStop::Pause {
            launcher.pause(&sandbox_id).await
        } else {
            launcher.delete(&sandbox_id).await
        };
        match outcome {
            Ok(()) => {
                if on_stop == OnStop::Pause {
                    let paused = SandboxRecord {
                        status: SandboxStatus::Paused,
                        last_used_epoch_secs: self.now(),
                        ..record
                    };
                    self.restore(paused).await;
                }
                Ok(())
            }
            Err(error) => {
                let process = if on_stop == OnStop::Pause {
                    "pause"
                } else {
                    "delete"
                };
                if error.kind() == ErrorKind::Unsupported {
                    tracing::warn!(
                        isolation_key,
                        sandbox_id,
                        launcher_type = %record.launcher_type,
                        process,
                        error = %error,
                        "sandbox teardown was refused by its launcher; the container is left \
                         in place and the record is kept"
                    );
                }
                let mapped = map_launcher_failure(process, error);
                self.restore(record).await;
                Err(mapped)
            }
        }
    }

    /// Destroys every container whose `now - last_used >= idle_ttl_seconds` and returns
    /// the evicted isolation keys.
    ///
    /// Python parity: `SandboxGateway._evict_idle` over
    /// `InMemorySandboxStore.evict_expired`, except that the comparison is `>=` so a
    /// zero TTL evicts everything (Python uses `>`).
    ///
    /// Deletion happens regardless of the launcher configuration's `on_stop`: an idle
    /// container is always destroyed (Python semantics). A launcher failure does not stop
    /// the sweep; it is logged and the remaining records are still evicted, and the failed
    /// record is gone from the map like every other expired one.
    pub async fn evict_idle(&self, idle_ttl_seconds: u64, now_epoch_secs: u64) -> Vec<String> {
        let expired = {
            let mut records = self.records.lock().await;
            let keys: Vec<String> = records
                .iter()
                .filter(|(_, record)| {
                    now_epoch_secs.saturating_sub(record.last_used_epoch_secs) >= idle_ttl_seconds
                })
                .map(|(key, _)| key.clone())
                .collect();
            keys.into_iter()
                .filter_map(|key| records.remove(&key))
                .collect::<Vec<_>>()
        };

        let mut evicted = Vec::with_capacity(expired.len());
        for record in expired {
            let sandbox_id = record.sandbox_id.clone().unwrap_or_default();
            match self.launchers.create(&record.launcher_type) {
                Ok(launcher) => {
                    if let Err(error) = launcher.delete(&sandbox_id).await {
                        if error.kind() == ErrorKind::Unsupported {
                            // The sweep keeps Python's drop-and-continue semantics, so the
                            // record is gone either way; saying so explicitly is the only
                            // honest outcome for a container the launcher cannot destroy.
                            tracing::warn!(
                                isolation_key = %record.isolation_key,
                                launcher_type = %record.launcher_type,
                                sandbox_id,
                                error = %error,
                                "idle sandbox was left behind (leaked): its launcher cannot \
                                 delete containers; the record was dropped and the sweep \
                                 continues"
                            );
                        } else {
                            tracing::warn!(
                                isolation_key = %record.isolation_key,
                                launcher_type = %record.launcher_type,
                                sandbox_id,
                                error = %error,
                                "idle sandbox deletion failed; the record was dropped and the \
                                 sweep continues"
                            );
                        }
                    }
                }
                Err(error) => tracing::warn!(
                    isolation_key = %record.isolation_key,
                    launcher_type = %record.launcher_type,
                    error = %error,
                    "idle sandbox launcher is not registered; the record was dropped"
                ),
            }
            evicted.push(record.isolation_key);
        }
        evicted
    }

    /// [`Self::evict_idle`] using the injected clock.
    pub async fn evict_idle_now(&self, idle_ttl_seconds: u64) -> Vec<String> {
        self.evict_idle(idle_ttl_seconds, self.now()).await
    }

    /// Evicts idle containers using the TTL of `config`, if it sets one.
    ///
    /// Returns an empty list when the configuration has no launcher or no
    /// `idle_ttl_seconds`, mirroring Python's `_evict_idle`, which returns early when the
    /// TTL is `None`.
    pub async fn evict_idle_for(
        &self,
        config: &SandboxGatewayConfig,
        now_epoch_secs: u64,
    ) -> Vec<String> {
        match config
            .launcher_config
            .as_ref()
            .and_then(|launcher| launcher.idle_ttl_seconds)
        {
            Some(idle_ttl_seconds) => self.evict_idle(idle_ttl_seconds, now_epoch_secs).await,
            None => Vec::new(),
        }
    }

    /// Drops every record without calling a launcher (test and shutdown helper).
    pub async fn clear(&self) {
        self.records.lock().await.clear();
    }

    /// Refreshes a running record and returns its endpoint.
    ///
    /// The record absorbs `launcher_config`'s hash and labels on the way out, so a container
    /// that is reused under a changed configuration is recorded as such.
    async fn touch_running(
        &self,
        isolation_key: &str,
        launcher_config: &SandboxLauncherConfig,
        now_epoch_secs: u64,
    ) -> Option<SandboxEndpoint> {
        let mut records = self.records.lock().await;
        let record = records.get_mut(isolation_key)?;
        if !record.status.is_running() {
            return None;
        }
        refresh_container_config(record, launcher_config);
        record.last_used_epoch_secs = now_epoch_secs;
        Some(record.endpoint.clone())
    }

    /// Marks the resumed record running, when it is still the one that was resumed.
    ///
    /// Called only after the launcher reported the container running or resumed it, so the
    /// configuration merge never runs on a failed launcher call.
    async fn mark_running(
        &self,
        isolation_key: &str,
        sandbox_id: &str,
        launcher_config: &SandboxLauncherConfig,
        now_epoch_secs: u64,
    ) -> Option<SandboxEndpoint> {
        let mut records = self.records.lock().await;
        let record = records.get_mut(isolation_key)?;
        if record.sandbox_id.as_deref() != Some(sandbox_id) {
            return None;
        }
        record.status = SandboxStatus::Running;
        refresh_container_config(record, launcher_config);
        record.last_used_epoch_secs = now_epoch_secs;
        Some(record.endpoint.clone())
    }

    /// Puts `record` back unless another caller already stored a record for its key.
    async fn restore(&self, record: SandboxRecord) {
        let key = record.isolation_key.clone();
        self.records.lock().await.entry(key).or_insert(record);
    }

    /// Removes `isolation_key` when it still points at `sandbox_id`.
    async fn drop_record(&self, isolation_key: &str, sandbox_id: &str) {
        let mut records = self.records.lock().await;
        if records
            .get(isolation_key)
            .is_some_and(|record| record.sandbox_id.as_deref() == Some(sandbox_id))
        {
            records.remove(isolation_key);
        }
    }

    /// Removes and returns the record of `isolation_key`.
    async fn take(&self, isolation_key: &str) -> Option<SandboxRecord> {
        self.records.lock().await.remove(isolation_key)
    }
}

/// The process-global lifecycle, over [`crate::sandbox::launcher::global`].
///
/// Python parity: `SandboxGateway.get_instance()` — one store per process. A local
/// [`SandboxLifecycle`] should be used in tests and for isolation.
#[must_use]
pub fn global() -> &'static SandboxLifecycle {
    static LIFECYCLE: OnceLock<SandboxLifecycle> = OnceLock::new();
    LIFECYCLE.get_or_init(|| SandboxLifecycle::new(crate::sandbox::launcher::global().clone()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
