//! Sandbox lifecycle: one `SandboxRecord` per isolation key plus the launcher calls that
//! keep it running.
//!
//! Python parity: `sandbox/gateway/gateway.py` (`SandboxGateway._get_endpoint`,
//! `_create_new_sandbox`, `_evict_idle`, `release_sandbox`, `pause_sandbox`,
//! `delete_sandbox`) and `sandbox/gateway/sandbox_store.py` (`SandboxRecord`,
//! `AbstractSandboxStore`, `InMemorySandboxStore`). The provider cache of Python's gateway
//! is not ported: providers are rebuilt from the resolved endpoint per
//! [`crate::SysOperation`], so only the container bookkeeping lives here.
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
//! // Reused, not relaunched, on the next call.
//! let again = lifecycle.ensure_endpoint(&config, "session_key", 1_010).await?;
//! assert_eq!(again, endpoint);
//!
//! lifecycle.release("session_key", OnStop::Delete).await?;
//! assert!(lifecycle.records().await.is_empty());
//! # Ok(())
//! # }
//! ```
use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use openjiuwen_common::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::sandbox::config::{
    OnStop, SandboxEndpoint, SandboxGatewayConfig, SandboxLauncherConfig,
};
use crate::sandbox::launcher::{
    isolation_key_error, launcher_error, LauncherRegistry, SandboxLauncher, SandboxStatus,
};

/// The bookkeeping entry of one sandbox container.
///
/// Python parity: `SandboxRecord` (`sandbox/gateway/sandbox_store.py`). Python keys the
/// record by isolation key inside the store; this struct carries the key too, so a record
/// is meaningful on its own. Python's `container_config_hash` and `metadata` are not
/// ported (they served the provider cache, which this rewrite does not keep).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxRecord {
    /// Isolation key the record is stored under.
    pub isolation_key: String,
    /// Runtime-assigned sandbox id, when the launcher reports one.
    pub sandbox_id: Option<String>,
    /// Endpoint handed to the providers.
    pub endpoint: SandboxEndpoint,
    /// Current lifecycle state.
    pub status: SandboxStatus,
    /// Launcher type that created the container.
    pub launcher_type: String,
    /// Creation time, in seconds since the Unix epoch.
    pub created_epoch_secs: u64,
    /// Last time the endpoint was handed out, in seconds since the Unix epoch.
    pub last_used_epoch_secs: u64,
}

/// A source of Unix-epoch seconds.
///
/// The lifecycle never reads the wall clock directly: it asks its clock, which a test
/// replaces with [`FixedClock`].
pub trait Clock: Send + Sync {
    /// Seconds since the Unix epoch.
    fn now_epoch_secs(&self) -> u64;
}

/// The wall clock ([`SystemTime`]); the default clock of [`SandboxLifecycle::new`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_epoch_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs())
    }
}

/// A clock that only moves when a caller moves it, for deterministic idle-stop tests.
#[derive(Debug, Clone, Default)]
pub struct FixedClock {
    epoch_secs: Arc<AtomicU64>,
}

impl FixedClock {
    /// A clock pinned to `epoch_secs`.
    #[must_use]
    pub fn new(epoch_secs: u64) -> Self {
        Self {
            epoch_secs: Arc::new(AtomicU64::new(epoch_secs)),
        }
    }

    /// Moves the clock to `epoch_secs`.
    pub fn set(&self, epoch_secs: u64) {
        self.epoch_secs.store(epoch_secs, Ordering::SeqCst);
    }

    /// Advances the clock by `seconds`.
    pub fn advance(&self, seconds: u64) {
        let current = self.epoch_secs.load(Ordering::SeqCst);
        self.set(current.saturating_add(seconds));
    }
}

impl Clock for FixedClock {
    fn now_epoch_secs(&self) -> u64 {
        self.epoch_secs.load(Ordering::SeqCst)
    }
}

/// Seconds since the Unix epoch, for hosts that do not inject a [`Clock`].
#[must_use]
pub fn now_epoch_secs() -> u64 {
    SystemClock.now_epoch_secs()
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
            .revive(&launcher, isolation_key, now_epoch_secs)
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
    /// fresh sandbox.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (199008) when the launcher fails to resume the container
    /// or cannot report its status.
    async fn revive(
        &self,
        launcher: &Arc<dyn SandboxLauncher>,
        isolation_key: &str,
        now_epoch_secs: u64,
    ) -> Result<Option<SandboxEndpoint>> {
        // 1. Reuse a running container.
        if let Some(endpoint) = self.touch_running(isolation_key, now_epoch_secs).await {
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
                .mark_running(isolation_key, &sandbox_id, now_epoch_secs)
                .await),
            // Python only knows running/paused/killed; a launcher that cannot tell is
            // treated as paused, because it may hold a suspended container that a resume
            // revives.
            Ok(SandboxStatus::Paused | SandboxStatus::Unknown) => {
                launcher
                    .resume(&sandbox_id)
                    .await
                    .map_err(|error| launcher_error("resume", error.message().as_ref()))?;
                Ok(self
                    .mark_running(isolation_key, &sandbox_id, now_epoch_secs)
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
            Err(error) => Err(launcher_error("check_status", error.message().as_ref())),
        }
    }

    /// Steps 3 and 4 of [`Self::ensure_endpoint`]: evict the idle containers, launch a new
    /// sandbox, and store its record.
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
            .map_err(|error| launcher_error("launch", error.message().as_ref()))?;

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
                tracing::warn!(
                    sandbox_id,
                    error = %error,
                    "failed to delete a duplicate sandbox launched for the same isolation key"
                );
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
    /// Returns `ErrorKind::SysOp` (199008) when the launcher fails, and
    /// `ErrorKind::Unsupported` when the recorded `launcher_type` is no longer registered.
    /// The record is restored on failure, so the call can be retried.
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
                let mapped = launcher_error(process, error.message().as_ref());
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
                        tracing::warn!(
                            isolation_key = %record.isolation_key,
                            sandbox_id,
                            error = %error,
                            "idle sandbox deletion failed"
                        );
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
    async fn touch_running(
        &self,
        isolation_key: &str,
        now_epoch_secs: u64,
    ) -> Option<SandboxEndpoint> {
        let mut records = self.records.lock().await;
        let record = records.get_mut(isolation_key)?;
        if !record.status.is_running() {
            return None;
        }
        record.last_used_epoch_secs = now_epoch_secs;
        Some(record.endpoint.clone())
    }

    /// Marks the resumed record running, when it is still the one that was resumed.
    async fn mark_running(
        &self,
        isolation_key: &str,
        sandbox_id: &str,
        now_epoch_secs: u64,
    ) -> Option<SandboxEndpoint> {
        let mut records = self.records.lock().await;
        let record = records.get_mut(isolation_key)?;
        if record.sandbox_id.as_deref() != Some(sandbox_id) {
            return None;
        }
        record.status = SandboxStatus::Running;
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

/// The process-global lifecycle, over [`LauncherRegistry::global`].
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
mod tests {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Mutex as StdMutex,
    };

    use async_trait::async_trait;
    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::sandbox::config::SandboxLauncherConfig;
    use crate::sandbox::launcher::{LaunchedSandbox, SandboxLauncher};

    /// A launcher recording every lifecycle call it receives.
    #[derive(Default)]
    struct RecordingLauncher {
        launches: AtomicU32,
        calls: StdMutex<Vec<String>>,
        status: StdMutex<SandboxStatus>,
        fail_launch: bool,
        unique_ids: bool,
    }

    impl RecordingLauncher {
        fn new() -> Self {
            Self {
                status: StdMutex::new(SandboxStatus::Running),
                ..Self::default()
            }
        }

        /// A launcher whose containers get a distinct id per launch, like a real runtime.
        fn unique() -> Self {
            Self {
                unique_ids: true,
                ..Self::new()
            }
        }

        fn failing() -> Self {
            Self {
                fail_launch: true,
                ..Self::new()
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl SandboxLauncher for RecordingLauncher {
        async fn launch(
            &self,
            _config: &SandboxLauncherConfig,
            timeout_seconds: u64,
            isolation_key: Option<&str>,
        ) -> Result<LaunchedSandbox> {
            let launch_number = self.launches.fetch_add(1, Ordering::SeqCst) + 1;
            self.calls
                .lock()
                .unwrap()
                .push(format!("launch:{timeout_seconds}"));
            if self.fail_launch {
                return Err(openjiuwen_common::error::Error::transport(
                    "container runtime refused the request",
                ));
            }
            // A real launch awaits I/O; yield so two concurrent callers interleave.
            tokio::task::yield_now().await;
            let key = isolation_key.unwrap_or("none");
            let sandbox_id = if self.unique_ids {
                format!("sandbox-{key}-{launch_number}")
            } else {
                format!("sandbox-{key}")
            };
            *self.status.lock().unwrap() = SandboxStatus::Running;
            Ok(LaunchedSandbox::new(format!("http://container/{key}")).with_sandbox_id(sandbox_id))
        }

        async fn pause(&self, sandbox_id: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("pause:{sandbox_id}"));
            *self.status.lock().unwrap() = SandboxStatus::Paused;
            Ok(())
        }

        async fn resume(&self, sandbox_id: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("resume:{sandbox_id}"));
            *self.status.lock().unwrap() = SandboxStatus::Running;
            Ok(())
        }

        async fn delete(&self, sandbox_id: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("delete:{sandbox_id}"));
            *self.status.lock().unwrap() = SandboxStatus::Stopped;
            Ok(())
        }

        async fn check_status(&self, _sandbox_id: &str) -> Result<SandboxStatus> {
            Ok(*self.status.lock().unwrap())
        }
    }

    fn config(launcher_type: &str, idle_ttl_seconds: Option<u64>) -> SandboxGatewayConfig {
        SandboxGatewayConfig {
            launcher_config: Some(SandboxLauncherConfig {
                launcher_type: launcher_type.to_string(),
                idle_ttl_seconds,
                ..SandboxLauncherConfig::default()
            }),
            timeout_seconds: 42,
            ..SandboxGatewayConfig::default()
        }
    }

    fn lifecycle_with(launcher_type: &str, launcher: Arc<RecordingLauncher>) -> SandboxLifecycle {
        let launchers = LauncherRegistry::new();
        launchers.register(launcher_type, launcher).unwrap();
        SandboxLifecycle::with_clock(launchers, Arc::new(FixedClock::new(1_000)))
    }

    #[tokio::test]
    async fn first_call_launches_and_later_calls_reuse() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", None);

        let endpoint = lifecycle
            .ensure_endpoint(&config, "key", 1_000)
            .await
            .unwrap();
        assert_eq!(endpoint.base_url, "http://container/key");
        assert_eq!(endpoint.sandbox_id.as_deref(), Some("sandbox-key"));
        assert_eq!(endpoint.isolation_key.as_deref(), Some("key"));
        assert_eq!(launcher.calls(), vec!["launch:42".to_string()]);

        let record = lifecycle.record("key").await.unwrap();
        assert_eq!(record.status, SandboxStatus::Running);
        assert_eq!(record.created_epoch_secs, 1_000);
        assert_eq!(record.last_used_epoch_secs, 1_000);

        // The second call reuses the container and refreshes `last_used`.
        let again = lifecycle
            .ensure_endpoint(&config, "key", 1_120)
            .await
            .unwrap();
        assert_eq!(again, endpoint);
        assert_eq!(launcher.calls(), vec!["launch:42".to_string()]);
        let record = lifecycle.record("key").await.unwrap();
        assert_eq!(record.created_epoch_secs, 1_000);
        assert_eq!(record.last_used_epoch_secs, 1_120);
    }

    #[tokio::test]
    async fn paused_records_are_resumed() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", None);
        lifecycle
            .ensure_endpoint(&config, "key", 1_000)
            .await
            .unwrap();

        lifecycle.release("key", OnStop::Pause).await.unwrap();
        assert_eq!(launcher.calls(), vec!["launch:42", "pause:sandbox-key"]);
        let record = lifecycle.record("key").await.unwrap();
        assert_eq!(record.status, SandboxStatus::Paused);

        // The paused container is resumed instead of being launched again.
        let endpoint = lifecycle
            .ensure_endpoint(&config, "key", 1_200)
            .await
            .unwrap();
        assert_eq!(endpoint.sandbox_id.as_deref(), Some("sandbox-key"));
        assert_eq!(
            launcher.calls(),
            vec![
                "launch:42".to_string(),
                "pause:sandbox-key".to_string(),
                "resume:sandbox-key".to_string(),
            ]
        );
        let record = lifecycle.record("key").await.unwrap();
        assert_eq!(record.status, SandboxStatus::Running);
        assert_eq!(record.last_used_epoch_secs, 1_200);
        assert_eq!(record.created_epoch_secs, 1_000);
    }

    #[tokio::test]
    async fn a_vanished_container_is_dropped_and_relaunched() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", None);
        lifecycle
            .ensure_endpoint(&config, "key", 1_000)
            .await
            .unwrap();
        lifecycle.release("key", OnStop::Pause).await.unwrap();

        // The paused container was destroyed out of band.
        *launcher.status.lock().unwrap() = SandboxStatus::Stopped;
        lifecycle
            .ensure_endpoint(&config, "key", 1_300)
            .await
            .unwrap();
        assert_eq!(
            launcher.calls(),
            vec![
                "launch:42".to_string(),
                "pause:sandbox-key".to_string(),
                "launch:42".to_string(),
            ]
        );
        let record = lifecycle.record("key").await.unwrap();
        assert_eq!(record.status, SandboxStatus::Running);
        assert_eq!(record.created_epoch_secs, 1_300);
    }

    #[tokio::test]
    async fn release_honors_on_stop() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", None);

        // Keep leaves the record (and refreshes it) and calls nothing.
        lifecycle
            .ensure_endpoint(&config, "keep", 1_000)
            .await
            .unwrap();
        lifecycle.release("keep", OnStop::Keep).await.unwrap();
        assert_eq!(launcher.calls(), vec!["launch:42".to_string()]);
        let record = lifecycle.record("keep").await.unwrap();
        assert_eq!(record.status, SandboxStatus::Running);
        assert_eq!(
            record.last_used_epoch_secs, 1_000,
            "the fixed clock did not move"
        );

        // Delete destroys the container and drops the record.
        lifecycle
            .ensure_endpoint(&config, "delete", 1_000)
            .await
            .unwrap();
        lifecycle.release("delete", OnStop::Delete).await.unwrap();
        assert!(lifecycle.record("delete").await.is_none());

        // Pause suspends the container and keeps its record paused.
        lifecycle
            .ensure_endpoint(&config, "pause", 1_000)
            .await
            .unwrap();
        lifecycle.release("pause", OnStop::Pause).await.unwrap();
        assert_eq!(
            lifecycle.record("pause").await.unwrap().status,
            SandboxStatus::Paused
        );

        assert_eq!(
            launcher.calls(),
            vec![
                "launch:42".to_string(),
                "launch:42".to_string(),
                "delete:sandbox-delete".to_string(),
                "launch:42".to_string(),
                "pause:sandbox-pause".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn releasing_an_unknown_key_is_a_no_op() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        for on_stop in [OnStop::Delete, OnStop::Pause, OnStop::Keep] {
            lifecycle.release("missing", on_stop).await.unwrap();
        }
        assert_eq!(launcher.calls(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn evict_idle_uses_the_fixed_clock_and_deletes_regardless_of_on_stop() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", None);
        lifecycle
            .ensure_endpoint(&config, "old", 1_000)
            .await
            .unwrap();
        lifecycle
            .ensure_endpoint(&config, "fresh", 1_400)
            .await
            .unwrap();

        // TTL 300 at t=1_600: "old" is 600 seconds idle, "fresh" only 200.
        let evicted = lifecycle.evict_idle(300, 1_600).await;
        assert_eq!(evicted, vec!["old".to_string()]);
        assert_eq!(
            launcher.calls(),
            vec![
                "launch:42".to_string(),
                "launch:42".to_string(),
                "delete:sandbox-old".to_string(),
            ]
        );
        assert_eq!(lifecycle.len().await, 1);

        // The boundary is inclusive: exactly the TTL counts as expired.
        let evicted = lifecycle.evict_idle(200, 1_600).await;
        assert_eq!(evicted, vec!["fresh".to_string()]);
        assert!(lifecycle.is_empty().await);

        // A zero TTL evicts everything still tracked.
        lifecycle
            .ensure_endpoint(&config, "zero", 1_600)
            .await
            .unwrap();
        assert_eq!(
            lifecycle.evict_idle(0, 1_600).await,
            vec!["zero".to_string()]
        );
    }

    #[tokio::test]
    async fn ensure_endpoint_evicts_before_launching_when_a_ttl_is_configured() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", Some(100));
        lifecycle
            .ensure_endpoint(&config, "a", 1_000)
            .await
            .unwrap();
        // Launching "b" at t=1_200 evicts "a" (200 seconds idle).
        lifecycle
            .ensure_endpoint(&config, "b", 1_200)
            .await
            .unwrap();
        assert_eq!(
            launcher.calls(),
            vec![
                "launch:42".to_string(),
                "delete:sandbox-a".to_string(),
                "launch:42".to_string(),
            ]
        );
        assert_eq!(lifecycle.len().await, 1);
    }

    #[tokio::test]
    async fn evict_idle_for_requires_a_configured_ttl() {
        let launcher = Arc::new(RecordingLauncher::new());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        lifecycle
            .ensure_endpoint(&config("docker", None), "a", 1_000)
            .await
            .unwrap();
        assert_eq!(
            lifecycle
                .evict_idle_for(&config("docker", None), 9_999)
                .await,
            Vec::<String>::new()
        );
        assert_eq!(
            lifecycle
                .evict_idle_for(&config("docker", Some(10)), 9_999)
                .await,
            vec!["a".to_string()]
        );
        assert_eq!(
            lifecycle
                .evict_idle_for(&SandboxGatewayConfig::default(), 9_999)
                .await,
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn launcher_failure_is_199008_not_unsupported() {
        let launcher = Arc::new(RecordingLauncher::failing());
        let lifecycle = lifecycle_with("docker", launcher);
        let error = lifecycle
            .ensure_endpoint(&config("docker", None), "key", 1_000)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert_eq!(
            error.message(),
            "sandbox launcher error, process: launch, reason: container runtime refused the \
             request"
        );
        assert!(lifecycle.is_empty().await);
    }

    #[tokio::test]
    async fn unknown_launcher_type_is_unsupported() {
        let lifecycle = lifecycle_with("docker", Arc::new(RecordingLauncher::new()));
        let error = lifecycle
            .ensure_endpoint(&config("podman", None), "key", 1_000)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error
            .message()
            .contains("Unknown launcher_type: podman; registered launcher types: docker"));
    }

    #[tokio::test]
    async fn blank_isolation_key_and_missing_launcher_config_are_rejected() {
        let lifecycle = lifecycle_with("docker", Arc::new(RecordingLauncher::new()));
        let error = lifecycle
            .ensure_endpoint(&config("docker", None), "   ", 1_000)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert_eq!(
            error.message(),
            "sandbox isolation key error, reason: isolation key must not be empty when resolving \
             a sandbox endpoint"
        );

        let error = lifecycle
            .ensure_endpoint(&SandboxGatewayConfig::default(), "key", 1_000)
            .await
            .unwrap_err();
        assert!(error
            .message()
            .contains("launcher_config is required when mode is 'sandbox'"));
    }

    #[tokio::test]
    async fn concurrent_launches_keep_one_container_and_delete_the_duplicate() {
        let launcher = Arc::new(RecordingLauncher::unique());
        let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
        let config = config("docker", None);

        let (first, second) = tokio::join!(
            lifecycle.ensure_endpoint(&config, "key", 1_000),
            lifecycle.ensure_endpoint(&config, "key", 1_000),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first, second);
        assert_eq!(lifecycle.len().await, 1);

        let calls = launcher.calls();
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("launch"))
                .count(),
            2
        );
        let deletes: Vec<&String> = calls
            .iter()
            .filter(|call| call.starts_with("delete"))
            .collect();
        assert_eq!(
            deletes.len(),
            1,
            "the losing launch must be cleaned up: {calls:?}"
        );
        assert_ne!(
            Some(deletes[0].trim_start_matches("delete:")),
            first.sandbox_id.as_deref(),
            "the surviving container must not be deleted"
        );
        // The stored record is the winner's, not the deleted duplicate.
        let record = lifecycle.record("key").await.unwrap();
        assert_eq!(record.sandbox_id, first.sandbox_id);
    }

    #[tokio::test]
    async fn records_round_trip_through_serde() {
        let lifecycle = lifecycle_with("docker", Arc::new(RecordingLauncher::new()));
        lifecycle
            .ensure_endpoint(&config("docker", None), "key", 1_000)
            .await
            .unwrap();
        let record = lifecycle.record("key").await.unwrap();
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["isolation_key"], "key");
        assert_eq!(json["status"], "running");
        assert_eq!(json["endpoint"]["sandbox_id"], "sandbox-key");
        let back: SandboxRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back, record);

        let unknown: SandboxRecord = serde_json::from_str("{\"isolation_key\": \"x\"}").unwrap();
        assert_eq!(unknown.status, SandboxStatus::Unknown);
    }

    #[test]
    fn fixed_clock_moves_only_when_told() {
        let clock = FixedClock::new(100);
        assert_eq!(clock.now_epoch_secs(), 100);
        clock.advance(5);
        assert_eq!(clock.now_epoch_secs(), 105);
        clock.set(7);
        assert_eq!(clock.now_epoch_secs(), 7);
        assert!(now_epoch_secs() > 1_600_000_000);
    }
}
