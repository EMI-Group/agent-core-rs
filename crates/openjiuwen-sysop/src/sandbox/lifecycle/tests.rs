//! Behaviour ports of the sandbox lifecycle manager: launch, reuse, resume, release,
//! idle-stop, and the concurrency and error paths.
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
    /// When set, `delete` reports this as a transport failure (a launcher that fails its job).
    delete_error: StdMutex<Option<String>>,
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
        if let Some(reason) = self.delete_error.lock().unwrap().clone() {
            return Err(openjiuwen_common::error::Error::transport(reason));
        }
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

/// `config` with an explicit `sandbox_type`, so two configurations hash differently.
fn config_for(launcher_type: &str, sandbox_type: &str) -> SandboxGatewayConfig {
    let mut config = config(launcher_type, None);
    if let Some(launcher) = config.launcher_config.as_mut() {
        launcher.sandbox_type = sandbox_type.to_string();
    }
    config
}

/// A launcher that can launch and report a state, but inherits the trait's refusals for
/// `pause`, `resume`, and `delete` — the shape of the built-in `pre_deploy` launcher.
#[derive(Default)]
struct LaunchOnlyLauncher {
    launches: AtomicU32,
    status: StdMutex<Option<SandboxStatus>>,
}

impl LaunchOnlyLauncher {
    fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SandboxLauncher for LaunchOnlyLauncher {
    async fn launch(
        &self,
        _config: &SandboxLauncherConfig,
        _timeout_seconds: u64,
        isolation_key: Option<&str>,
    ) -> Result<LaunchedSandbox> {
        self.launches.fetch_add(1, Ordering::SeqCst);
        let key = isolation_key.unwrap_or("none");
        let mut status = self.status.lock().unwrap();
        if status.is_none() {
            *status = Some(SandboxStatus::Running);
        }
        Ok(LaunchedSandbox::new(format!("http://container/{key}"))
            .with_sandbox_id(format!("sandbox-{key}")))
    }

    async fn check_status(&self, _sandbox_id: &str) -> Result<SandboxStatus> {
        Ok(self
            .status
            .lock()
            .unwrap()
            .unwrap_or(SandboxStatus::Unknown))
    }
}

/// A launcher that can suspend its container but never resume or destroy it: `pause`
/// succeeds and `resume`/`delete` keep the trait's typed refusals.
struct PauseOnlyLauncher {
    status: StdMutex<SandboxStatus>,
}

impl PauseOnlyLauncher {
    fn new() -> Self {
        Self {
            status: StdMutex::new(SandboxStatus::Running),
        }
    }
}

#[async_trait]
impl SandboxLauncher for PauseOnlyLauncher {
    async fn launch(
        &self,
        _config: &SandboxLauncherConfig,
        _timeout_seconds: u64,
        isolation_key: Option<&str>,
    ) -> Result<LaunchedSandbox> {
        *self.status.lock().unwrap() = SandboxStatus::Running;
        let key = isolation_key.unwrap_or("none");
        Ok(LaunchedSandbox::new(format!("http://container/{key}"))
            .with_sandbox_id(format!("sandbox-{key}")))
    }

    async fn pause(&self, _sandbox_id: &str) -> Result<()> {
        *self.status.lock().unwrap() = SandboxStatus::Paused;
        Ok(())
    }

    async fn check_status(&self, _sandbox_id: &str) -> Result<SandboxStatus> {
        Ok(*self.status.lock().unwrap())
    }
}

fn lifecycle_with_launcher(
    launcher_type: &str,
    launcher: Arc<dyn SandboxLauncher>,
) -> SandboxLifecycle {
    let launchers = LauncherRegistry::new();
    launchers.register(launcher_type, launcher).unwrap();
    SandboxLifecycle::with_clock(launchers, Arc::new(FixedClock::new(1_000)))
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

// ── typed launcher refusals ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn release_delete_on_a_launcher_without_delete_is_unsupported_and_keeps_the_record() {
    let lifecycle = lifecycle_with_launcher("pre_deploy_like", Arc::new(LaunchOnlyLauncher::new()));
    let config = config("pre_deploy_like", None);
    lifecycle
        .ensure_endpoint(&config, "key", 1_000)
        .await
        .unwrap();
    let stored = lifecycle.record("key").await.unwrap();

    let error = lifecycle
        .release("key", OnStop::Delete)
        .await
        .expect_err("a launcher that cannot destroy containers must not report success");
    assert_eq!(
        error.kind(),
        ErrorKind::Unsupported,
        "the typed refusal must not be flattened into 199008: {error}"
    );
    assert_eq!(
        error.message(),
        "sandbox launcher unsupported, launcher: custom, process: delete, reason: this \
         launcher cannot destroy containers; teardown would leak the sandbox"
    );

    // The record is restored, so teardown can be retried or swept later.
    assert_eq!(lifecycle.record("key").await.unwrap(), stored);
    assert_eq!(lifecycle.len().await, 1);
}

#[tokio::test]
async fn release_pause_on_a_launcher_without_pause_is_unsupported_and_keeps_the_record() {
    let lifecycle = lifecycle_with_launcher("launch_only", Arc::new(LaunchOnlyLauncher::new()));
    let config = config("launch_only", None);
    lifecycle
        .ensure_endpoint(&config, "key", 1_000)
        .await
        .unwrap();
    let stored = lifecycle.record("key").await.unwrap();

    let error = lifecycle
        .release("key", OnStop::Pause)
        .await
        .expect_err("a launcher that cannot suspend containers must not report success");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(
        error.message(),
        "sandbox launcher unsupported, launcher: custom, process: pause, reason: this \
         launcher cannot suspend containers; the sandbox keeps running"
    );
    assert_eq!(
        lifecycle.record("key").await.unwrap(),
        stored,
        "the restored record keeps its running state, hash, and metadata"
    );
    assert_eq!(
        lifecycle.record("key").await.unwrap().status,
        SandboxStatus::Running
    );
}

#[tokio::test]
async fn ensure_endpoint_on_a_paused_record_whose_launcher_cannot_resume_is_unsupported() {
    let lifecycle = lifecycle_with_launcher("pause_only", Arc::new(PauseOnlyLauncher::new()));
    let config = config("pause_only", None);
    lifecycle
        .ensure_endpoint(&config, "key", 1_000)
        .await
        .unwrap();
    lifecycle.release("key", OnStop::Pause).await.unwrap();
    let paused = lifecycle.record("key").await.unwrap();
    assert_eq!(paused.status, SandboxStatus::Paused);

    let error = lifecycle
        .ensure_endpoint(&config, "key", 1_100)
        .await
        .expect_err("resuming a container the launcher cannot resume must report the refusal");
    assert_eq!(
        error.kind(),
        ErrorKind::Unsupported,
        "the launcher's Unsupported must survive `revive`: {error}"
    );
    assert_eq!(
        error.message(),
        "sandbox launcher unsupported, launcher: custom, process: resume, reason: this \
         launcher cannot resume containers; the sandbox stays suspended"
    );
    // Nothing was mutated by the failed revive, and no replacement container was launched.
    assert_eq!(lifecycle.record("key").await.unwrap(), paused);
}

#[tokio::test]
async fn evict_idle_drops_a_container_the_launcher_cannot_delete_and_keeps_sweeping() {
    let launcher = Arc::new(RecordingLauncher::new());
    let launchers = LauncherRegistry::new();
    launchers.register("docker", launcher.clone()).unwrap();
    launchers
        .register("launch_only", Arc::new(LaunchOnlyLauncher::new()))
        .unwrap();
    let lifecycle = SandboxLifecycle::with_clock(launchers, Arc::new(FixedClock::new(1_000)));

    lifecycle
        .ensure_endpoint(&config("launch_only", None), "a_leaky", 1_000)
        .await
        .unwrap();
    lifecycle
        .ensure_endpoint(&config("docker", None), "b_deletable", 1_000)
        .await
        .unwrap();

    // Both records expired; the first cannot be destroyed, which must not abort the sweep.
    let evicted = lifecycle.evict_idle(300, 1_400).await;
    assert_eq!(
        evicted,
        vec!["a_leaky".to_string(), "b_deletable".to_string()]
    );
    assert!(lifecycle.is_empty().await);
    assert_eq!(
        launcher.calls(),
        vec![
            "launch:42".to_string(),
            "delete:sandbox-b_deletable".to_string()
        ],
        "the deletable record was still deleted after the refused one"
    );
}

#[tokio::test]
async fn a_failing_delete_is_still_199008_and_restores_the_record() {
    let launcher = Arc::new(RecordingLauncher::new());
    let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
    lifecycle
        .ensure_endpoint(&config("docker", None), "key", 1_000)
        .await
        .unwrap();
    launcher
        .delete_error
        .lock()
        .unwrap()
        .replace("runtime down".to_string());

    let error = lifecycle.release("key", OnStop::Delete).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::SysOp, "a failed call stays 199008");
    assert_eq!(
        error.message(),
        "sandbox launcher error, process: delete, reason: runtime down"
    );
    assert_eq!(
        lifecycle.record("key").await.unwrap().status,
        SandboxStatus::Running
    );
}

// ── container config hash and metadata ──────────────────────────────────────────────────

#[tokio::test]
async fn a_fresh_record_stores_the_config_hash_and_derived_metadata() {
    let lifecycle = lifecycle_with("docker", Arc::new(RecordingLauncher::new()));
    let config = config_for("docker", "aio");
    lifecycle
        .ensure_endpoint(&config, "key", 1_000)
        .await
        .unwrap();

    let record = lifecycle.record("key").await.unwrap();
    let launcher_config = config.launcher_config.as_ref().unwrap();
    assert_eq!(
        record.container_config_hash,
        container_config_hash(launcher_config)
    );
    assert_eq!(record.container_config_hash.len(), 64);
    assert_eq!(record.metadata["launcher_type"], "docker");
    assert_eq!(record.metadata["sandbox_type"], "aio");
    assert_eq!(record.metadata, container_metadata(launcher_config));
}

#[tokio::test]
async fn reuse_updates_the_hash_when_the_launcher_config_changes_and_merges_metadata() {
    let lifecycle = lifecycle_with("docker", Arc::new(RecordingLauncher::new()));
    let first = config_for("docker", "aio");
    lifecycle
        .ensure_endpoint(&first, "key", 1_000)
        .await
        .unwrap();
    let before = lifecycle.record("key").await.unwrap();

    // Same container, different configuration: the record follows the live configuration.
    let mut changed = config_for("docker", "jiuwenbox");
    changed
        .launcher_config
        .as_mut()
        .unwrap()
        .extra_params
        .insert("image".to_string(), serde_json::json!("alpine"));
    let endpoint = lifecycle
        .ensure_endpoint(&changed, "key", 1_100)
        .await
        .unwrap();
    assert_eq!(
        endpoint, before.endpoint,
        "the container is reused, not relaunched"
    );

    let after = lifecycle.record("key").await.unwrap();
    assert_ne!(after.container_config_hash, before.container_config_hash);
    assert_eq!(
        after.container_config_hash,
        container_config_hash(changed.launcher_config.as_ref().unwrap())
    );
    assert_eq!(
        after.metadata["sandbox_type"], "jiuwenbox",
        "new value wins"
    );
    assert_eq!(after.metadata["image"], "alpine");
    assert_eq!(after.created_epoch_secs, before.created_epoch_secs);
    assert_eq!(after.last_used_epoch_secs, 1_100);

    // A later configuration that drops a label leaves the stored one alone.
    let back = config_for("docker", "aio");
    lifecycle
        .ensure_endpoint(&back, "key", 1_200)
        .await
        .unwrap();
    let after = lifecycle.record("key").await.unwrap();
    assert_eq!(after.metadata["sandbox_type"], "aio");
    assert_eq!(
        after.metadata["image"], "alpine",
        "a label the new configuration does not mention is preserved"
    );
}

#[tokio::test]
async fn resume_merges_the_config_hash_and_metadata() {
    let launcher = Arc::new(RecordingLauncher::new());
    let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
    lifecycle
        .ensure_endpoint(&config_for("docker", "aio"), "key", 1_000)
        .await
        .unwrap();
    lifecycle.release("key", OnStop::Pause).await.unwrap();

    let changed = config_for("docker", "yuanrong");
    let endpoint = lifecycle
        .ensure_endpoint(&changed, "key", 1_200)
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
    assert_eq!(
        record.container_config_hash,
        container_config_hash(changed.launcher_config.as_ref().unwrap())
    );
    assert_eq!(record.metadata["sandbox_type"], "yuanrong");
}

#[tokio::test]
async fn failed_launcher_calls_never_lose_the_hash_or_metadata() {
    let launcher = Arc::new(RecordingLauncher::new());
    let lifecycle = lifecycle_with("docker", Arc::clone(&launcher));
    let config = config_for("docker", "aio");
    lifecycle
        .ensure_endpoint(&config, "key", 1_000)
        .await
        .unwrap();
    let stored = lifecycle.record("key").await.unwrap();
    assert_ne!(stored.container_config_hash, "");

    launcher
        .delete_error
        .lock()
        .unwrap()
        .replace("nope".to_string());
    let _ = lifecycle.release("key", OnStop::Delete).await;
    assert_eq!(
        lifecycle.record("key").await.unwrap().container_config_hash,
        stored.container_config_hash
    );
    assert_eq!(
        lifecycle.record("key").await.unwrap().metadata,
        stored.metadata
    );
}

// ── idle sweeper with the injected clock ───────────────────────────────────────────────

#[tokio::test]
async fn the_idle_sweeper_reads_the_injected_clock_and_spares_the_fresh_record() {
    let clock = Arc::new(FixedClock::new(2_000));
    let launcher = Arc::new(RecordingLauncher::new());
    let launchers = LauncherRegistry::new();
    launchers.register("docker", launcher.clone()).unwrap();
    let lifecycle = SandboxLifecycle::with_clock(launchers, clock.clone());
    // No TTL on the launch path, so only the explicit sweeps below decide anything.
    let plain = config("docker", None);
    let ttl_config = config("docker", Some(300));

    lifecycle
        .ensure_endpoint(&plain, "expired", 2_000)
        .await
        .unwrap();
    lifecycle
        .ensure_endpoint(&plain, "fresh", 2_500)
        .await
        .unwrap();

    // Nothing is past the TTL yet.
    assert_eq!(lifecycle.evict_idle_now(300).await, Vec::<String>::new());
    assert_eq!(lifecycle.len().await, 2);

    clock.advance(400);
    assert_eq!(
        lifecycle.evict_idle_now(300).await,
        vec!["expired".to_string()],
        "only the record past the TTL is evicted"
    );
    assert_eq!(launcher.calls().last().unwrap(), "delete:sandbox-expired");
    let fresh = lifecycle
        .record("fresh")
        .await
        .expect("the fresh record survives");
    assert_eq!(fresh.status, SandboxStatus::Running);

    // `evict_idle_for` honours the configuration's TTL with the same clock.
    clock.advance(1_000);
    assert_eq!(
        lifecycle
            .evict_idle_for(&ttl_config, clock.now_epoch_secs())
            .await,
        vec!["fresh".to_string()]
    );
    assert!(lifecycle.is_empty().await);
}

#[tokio::test]
async fn evict_idle_for_is_a_no_op_without_a_configured_ttl() {
    let clock = Arc::new(FixedClock::new(1_000));
    let launcher = Arc::new(RecordingLauncher::new());
    let launchers = LauncherRegistry::new();
    launchers.register("docker", launcher.clone()).unwrap();
    let lifecycle = SandboxLifecycle::with_clock(launchers, clock.clone());
    lifecycle
        .ensure_endpoint(&config("docker", None), "key", 1_000)
        .await
        .unwrap();

    clock.advance(100_000);
    assert_eq!(
        lifecycle
            .evict_idle_for(&config("docker", None), clock.now_epoch_secs())
            .await,
        Vec::<String>::new()
    );
    assert_eq!(
        lifecycle
            .evict_idle_for(&SandboxGatewayConfig::default(), clock.now_epoch_secs())
            .await,
        Vec::<String>::new()
    );
    assert_eq!(lifecycle.len().await, 1, "no TTL means no sweep");
    assert_eq!(launcher.calls(), vec!["launch:42".to_string()]);
}

/// The `Debug` rendering never claims "no containers" when it cannot read the map.
#[tokio::test]
async fn debug_marks_a_busy_record_map_instead_of_listing_nothing() {
    let lifecycle = SandboxLifecycle::new(LauncherRegistry::new());
    assert!(
        format!("{lifecycle:?}").contains("records: []"),
        "an idle lifecycle renders an empty list: {lifecycle:?}"
    );
    // Holding the map lock makes `try_lock` fail while the call is running.
    let guard = lifecycle.records.lock().await;
    let rendered = format!("{lifecycle:?}");
    assert!(
        rendered.contains("<locked>"),
        "a busy record map must be marked, not shown as empty: {rendered}"
    );
    drop(guard);
    assert!(
        format!("{lifecycle:?}").contains("records: []"),
        "the marker disappears once the map is readable again"
    );
}
