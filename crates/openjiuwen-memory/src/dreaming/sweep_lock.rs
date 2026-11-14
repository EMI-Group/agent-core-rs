//! [`DREAM_SWEEP_LOCK_NAME`]: the caller-injected lock one dream sweep holds.
//!
//! Per-job leasing ([`crate::dreaming::DreamPlanner::run_due_leased`]) stops two workers from
//! consolidating the *same job*. A sweep lock stops two workers from sweeping the *same
//! queue* at the same time, which is what keeps two claim scans, two retention purges, and
//! two repository budgets from interleaving.
//!
//! The seam is [`DistributedKvLock`] (see `crate::store::kv_lock`):
//! [`InProcessKvLock`](crate::store::kv_lock::InProcessKvLock) excludes tasks of one process,
//! [`FileKvLock`](crate::store::kv_lock::FileKvLock) excludes other processes sharing one lock
//! root, and [`UnsupportedKvLock`](crate::store::kv_lock::UnsupportedKvLock) rejects a
//! multi-node broker loudly. The backend is injected with
//! [`DreamOrchestrator::with_sweep_lock`](crate::dreaming::DreamOrchestrator::with_sweep_lock),
//! so nothing here is global and an orchestrator without a lock behaves exactly as before.
//!
//! # Kinds of outcome
//!
//! [`acquire_sweep_lock`] never waits for anything itself: [`LockOptions`] decides the wait.
//! It returns one of two outcomes:
//!
//! - [`SweepLockOutcome::Acquired`] — the caller holds [`DREAM_SWEEP_LOCK_NAME`] for the
//!   critical section and releases it by dropping the [`KvLockGuard`];
//! - [`SweepLockOutcome::Skipped`] — a typed [`SweepLockSkip`]:
//!   [`SweepLockSkipKind::Held`] when another sweeper holds the lock and the caller's timeout
//!   expired, [`SweepLockSkipKind::Unsupported`] when the backend cannot lock at all (a
//!   vendor broker such as `UnsupportedKvLock::redis()`).
//!
//! Every other failure — a [`Validation`](openjiuwen_common::error::ErrorKind::Validation)
//! error for unusable options, cancellation, or the file backend failing to create its
//! sidecar — is returned as `Err`: only *lock unavailability* is normal operation.
//!
//! # Timeouts
//!
//! [`LockOptions::new`] waits forever, so a caller whose sweeps may overlap another worker
//! must bound the wait ([`LockOptions::with_timeout`]); otherwise the tick parks until the
//! current holder releases instead of skipping. Nothing polls in the background and no
//! heartbeat exists: the guard lives exactly as long as the tick that acquired it.

use std::fmt;

use openjiuwen_common::error::{ErrorKind, Result};
use serde::{Deserialize, Serialize};

use crate::store::kv_lock::{DistributedKvLock, KvLockGuard, LockOptions};

/// The tracing target for every event this module emits.
const TARGET: &str = "openjiuwen_memory::dreaming::sweep_lock";

/// The name of the lock a dream sweep holds: `"memory_dream_sweep"`.
///
/// One name for every dream queue, because the queue is the unit of exclusion; per-job
/// exclusion is [`DreamLease`](crate::dreaming::DreamLease)'s job, not the sweep lock's.
pub const DREAM_SWEEP_LOCK_NAME: &str = "memory_dream_sweep";

/// Why a sweep lock could not be obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepLockSkipKind {
    /// Another sweeper holds the lock and this caller's bounded wait expired.
    Held,
    /// The configured backend cannot lock at all (a typed `Unsupported` seam).
    Unsupported,
}

impl SweepLockSkipKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 2] = [Self::Held, Self::Unsupported];

    /// The `snake_case` name used in JSON and in log lines.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Held => "held",
            Self::Unsupported => "unsupported",
        }
    }
}

impl fmt::Display for SweepLockSkipKind {
    /// Prints [`SweepLockSkipKind::as_str`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A recorded sweep-lock skip: the typed kind, the lock, the backend, and its message.
///
/// JSON is `snake_case`, every field defaults on input (so a partial object decodes), unknown
/// fields are ignored, and the type round-trips — which matters because the reason is copied
/// into [`DreamOrchestratorStatus::last_skip_reason`](crate::dreaming::DreamOrchestratorStatus)
/// and onto the owning [`DreamRunSummary`](crate::dreaming::DreamRunSummary).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SweepLockSkip {
    /// Why the lock was unavailable.
    pub kind: SweepLockSkipKind,
    /// The lock name that could not be taken; [`DREAM_SWEEP_LOCK_NAME`] for a dream tick.
    pub lock: String,
    /// The backend label that refused (`"in-process"`, `"file"`, `"redis"`, ...).
    pub backend: String,
    /// The backend's own message, kept verbatim for diagnostics.
    pub reason: String,
}

impl Default for SweepLockSkipKind {
    /// The routine case: another sweeper holds the lock.
    fn default() -> Self {
        Self::Held
    }
}

impl Default for SweepLockSkip {
    /// The empty skip: the routine [`SweepLockSkipKind::Held`] kind on the documented lock,
    /// with no backend or message yet.
    fn default() -> Self {
        Self {
            kind: SweepLockSkipKind::default(),
            lock: DREAM_SWEEP_LOCK_NAME.to_owned(),
            backend: String::new(),
            reason: String::new(),
        }
    }
}

impl SweepLockSkip {
    /// A skip of `kind` on `backend` for `DREAM_SWEEP_LOCK_NAME`.
    #[must_use]
    pub fn new(
        kind: SweepLockSkipKind,
        backend: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            lock: DREAM_SWEEP_LOCK_NAME.to_owned(),
            backend: backend.into(),
            reason: reason.into(),
        }
    }

    /// A skip because another sweeper holds the lock.
    #[must_use]
    pub fn held(backend: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(SweepLockSkipKind::Held, backend, reason)
    }

    /// A skip because the backend cannot lock.
    #[must_use]
    pub fn unsupported(backend: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(SweepLockSkipKind::Unsupported, backend, reason)
    }

    /// Whether another sweeper held the lock.
    #[must_use]
    pub const fn is_held(&self) -> bool {
        matches!(self.kind, SweepLockSkipKind::Held)
    }

    /// Whether the backend is a typed `Unsupported` seam.
    #[must_use]
    pub const fn is_unsupported(&self) -> bool {
        matches!(self.kind, SweepLockSkipKind::Unsupported)
    }
}

impl fmt::Display for SweepLockSkip {
    /// Prints the lock, the backend, the kind, and the backend's message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the dream sweep lock {:?} on the {} backend was skipped ({}): {}",
            self.lock, self.backend, self.kind, self.reason
        )
    }
}

/// One sweep-lock acquisition: the held guard, or the typed reason it was unavailable.
#[derive(Debug)]
pub enum SweepLockOutcome {
    /// The lock is held; dropping the guard releases it immediately.
    Acquired(KvLockGuard),
    /// The lock was unavailable; a dream tick skips its sweep and its retention pass.
    Skipped(SweepLockSkip),
}

impl SweepLockOutcome {
    /// Whether the acquisition succeeded.
    #[must_use]
    pub const fn is_acquired(&self) -> bool {
        matches!(self, Self::Acquired(_))
    }

    /// Whether the acquisition was skipped.
    #[must_use]
    pub const fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped(_))
    }

    /// The skip reason, when the acquisition was skipped.
    #[must_use]
    pub const fn skip(&self) -> Option<&SweepLockSkip> {
        match self {
            Self::Skipped(skip) => Some(skip),
            Self::Acquired(_) => None,
        }
    }
}

/// Acquires [`DREAM_SWEEP_LOCK_NAME`] from `lock` with the caller's `options`.
///
/// One attempt, bounded only by [`LockOptions::timeout`]: no retry loop, no poll, no spawned
/// task. The guard is returned to the caller, which owns the critical section.
///
/// # Errors
///
/// Propagates everything that is *not* lock unavailability: `Validation` for an unusable lock
/// name or options, `Cancelled` from a backend that observes cancellation, and `External`
/// (for example the file backend being unable to create its sidecar).
pub async fn acquire_sweep_lock(
    lock: &dyn DistributedKvLock,
    options: &LockOptions,
) -> Result<SweepLockOutcome> {
    match lock.acquire(DREAM_SWEEP_LOCK_NAME, options).await {
        Ok(guard) => {
            tracing::debug!(
                target: TARGET,
                lock = DREAM_SWEEP_LOCK_NAME,
                owner = %options.owner,
                backend = guard.backend(),
                "acquired the dream sweep lock"
            );
            Ok(SweepLockOutcome::Acquired(guard))
        }
        Err(error) if error.kind() == ErrorKind::Timeout => {
            tracing::debug!(
                target: TARGET,
                lock = DREAM_SWEEP_LOCK_NAME,
                backend = lock.backend(),
                error = %error,
                "the dream sweep lock is held elsewhere; the tick will skip"
            );
            Ok(SweepLockOutcome::Skipped(SweepLockSkip::held(
                lock.backend(),
                error.to_string(),
            )))
        }
        Err(error) if error.kind() == ErrorKind::Unsupported => {
            tracing::debug!(
                target: TARGET,
                lock = DREAM_SWEEP_LOCK_NAME,
                backend = lock.backend(),
                error = %error,
                "the dream sweep lock backend cannot lock; the tick will skip"
            );
            Ok(SweepLockOutcome::Skipped(SweepLockSkip::unsupported(
                lock.backend(),
                error.to_string(),
            )))
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dreaming::{DreamOrchestratorStatus, DreamRunSummary};
    use crate::store::kv_lock::{InProcessKvLock, UnsupportedKvLock};
    use std::time::Duration;

    #[tokio::test]
    async fn a_free_lock_is_acquired_under_the_documented_name() {
        let locks = InProcessKvLock::new();
        let outcome = acquire_sweep_lock(&locks, &LockOptions::new("worker-a"))
            .await
            .unwrap();
        assert!(outcome.is_acquired());
        assert!(!outcome.is_skipped());
        assert!(outcome.skip().is_none());
        let SweepLockOutcome::Acquired(guard) = outcome else {
            panic!("expected an acquired guard");
        };
        assert_eq!(guard.name(), DREAM_SWEEP_LOCK_NAME);
        assert_eq!(guard.name(), "memory_dream_sweep");
        assert_eq!(guard.owner(), "worker-a");
        assert_eq!(guard.backend(), "in-process");
        assert_eq!(locks.held_names(), vec![DREAM_SWEEP_LOCK_NAME.to_owned()]);
        drop(guard);
        assert_eq!(locks.held_names(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn a_held_lock_skips_with_the_held_kind() {
        let locks = InProcessKvLock::new();
        let held = locks
            .acquire(DREAM_SWEEP_LOCK_NAME, &LockOptions::new("worker-a"))
            .await
            .unwrap();

        let options = LockOptions::new("worker-b").with_timeout(Duration::from_millis(50));
        let outcome = acquire_sweep_lock(&locks, &options).await.unwrap();
        assert!(outcome.is_skipped());
        let skip = outcome.skip().expect("a skip reason");
        assert_eq!(skip.kind, SweepLockSkipKind::Held);
        assert!(skip.is_held());
        assert!(!skip.is_unsupported());
        assert_eq!(skip.lock, DREAM_SWEEP_LOCK_NAME);
        assert_eq!(skip.backend, "in-process");
        assert!(
            skip.reason.contains(DREAM_SWEEP_LOCK_NAME),
            "{}",
            skip.reason
        );
        assert!(
            skip.to_string().contains("held"),
            "the display names the kind: {skip}"
        );

        // Releasing the first holder makes the same call succeed.
        held.release();
        let again = acquire_sweep_lock(&locks, &options).await.unwrap();
        assert!(again.is_acquired());
    }

    #[tokio::test]
    async fn an_unsupported_backend_skips_with_the_unsupported_kind() {
        let locks = UnsupportedKvLock::redis();
        let outcome = acquire_sweep_lock(&locks, &LockOptions::new("worker-a"))
            .await
            .unwrap();
        assert!(outcome.is_skipped());
        let skip = outcome.skip().expect("a skip reason");
        assert_eq!(skip.kind, SweepLockSkipKind::Unsupported);
        assert!(skip.is_unsupported());
        assert!(!skip.is_held());
        assert_eq!(skip.backend, "redis");
        assert!(skip.reason.contains("redis"), "{}", skip.reason);
    }

    #[tokio::test]
    async fn unusable_options_are_errors_rather_than_skips() {
        let locks = InProcessKvLock::new();
        let error = acquire_sweep_lock(&locks, &LockOptions::new(""))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
    }

    #[test]
    fn kinds_render_and_are_listed() {
        assert_eq!(
            SweepLockSkipKind::ALL.map(SweepLockSkipKind::as_str),
            ["held", "unsupported"]
        );
        assert_eq!(SweepLockSkipKind::Held.to_string(), "held");
        assert_eq!(SweepLockSkipKind::Unsupported.to_string(), "unsupported");
        assert_eq!(SweepLockSkipKind::default(), SweepLockSkipKind::Held);
    }

    #[test]
    fn skips_round_trip_and_decode_partial_json() {
        let skip = SweepLockSkip::held("file", "timed out");
        let json = serde_json::to_value(&skip).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "kind": "held",
                "lock": "memory_dream_sweep",
                "backend": "file",
                "reason": "timed out",
            })
        );
        assert_eq!(serde_json::from_value::<SweepLockSkip>(json).unwrap(), skip);

        let partial: SweepLockSkip = serde_json::from_value(serde_json::json!({
            "kind": "unsupported",
            "extra": true,
        }))
        .unwrap();
        assert_eq!(partial.kind, SweepLockSkipKind::Unsupported);
        assert_eq!(partial.lock, DREAM_SWEEP_LOCK_NAME);
        assert_eq!(partial.backend, "");
        assert_eq!(partial.reason, "");

        let empty: SweepLockSkip = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, SweepLockSkip::default());
        assert_eq!(empty.kind, SweepLockSkipKind::Held);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_second_handle_on_one_root_skips() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("locks");
        let first = crate::store::kv_lock::FileKvLock::new(&root).unwrap();
        let held = acquire_sweep_lock(&first, &LockOptions::new("worker-a"))
            .await
            .unwrap();
        assert!(held.is_acquired());

        let second = crate::store::kv_lock::FileKvLock::new(&root).unwrap();
        let options = LockOptions::new("worker-b")
            .with_timeout(Duration::from_millis(100))
            .with_poll_interval(Duration::from_millis(1));
        let outcome = acquire_sweep_lock(&second, &options).await.unwrap();
        let skip = outcome.skip().expect("a skip reason");
        assert_eq!(skip.kind, SweepLockSkipKind::Held);
        assert_eq!(skip.backend, "file");
    }

    #[test]
    fn the_sweep_lock_fields_stay_out_of_json_when_empty() {
        let status = DreamOrchestratorStatus::default();
        let json = serde_json::to_value(&status).unwrap();
        assert!(json.get("lock_skips").is_none(), "{json}");
        assert!(json.get("last_skip_reason").is_none(), "{json}");

        let summary = DreamRunSummary::new();
        let json = serde_json::to_value(&summary).unwrap();
        assert!(json.get("lock_skip").is_none(), "{json}");

        // Legacy JSON without the new fields still decodes.
        let legacy: DreamOrchestratorStatus = serde_json::from_value(serde_json::json!({
            "ticks": 2,
            "last_error": "boom",
        }))
        .unwrap();
        assert_eq!(legacy.ticks, 2);
        assert_eq!(legacy.lock_skips, 0);
        assert!(legacy.last_skip_reason.is_none());
        let legacy: DreamRunSummary =
            serde_json::from_value(serde_json::json!({"scanned": 1, "succeeded": 1})).unwrap();
        assert!(legacy.lock_skip.is_none());

        // A present skip round-trips through both types.
        let skip = crate::dreaming::SweepLockSkip::held("file", "timed out");
        let status = DreamOrchestratorStatus {
            lock_skips: 3,
            last_skip_reason: Some(skip.to_string()),
            ..DreamOrchestratorStatus::default()
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"lock_skips\":3"), "{json}");
        assert_eq!(
            serde_json::from_str::<DreamOrchestratorStatus>(&json).unwrap(),
            status
        );

        let summary = DreamRunSummary::lock_skipped(skip.clone());
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("memory_dream_sweep"), "{json}");
        let parsed: DreamRunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.lock_skip, Some(skip.clone()));
        assert!(parsed.is_lock_skip());
        assert!(parsed.is_empty());

        // The skip marker survives a merge.
        let mut total = DreamRunSummary::new();
        total.merge(DreamRunSummary::lock_skipped(skip.clone()));
        assert_eq!(total.lock_skip, Some(skip));
        assert_eq!(total.scanned, 0);
    }
}
