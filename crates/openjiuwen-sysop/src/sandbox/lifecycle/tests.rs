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
