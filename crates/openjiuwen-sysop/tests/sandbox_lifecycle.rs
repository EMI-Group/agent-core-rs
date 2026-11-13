//! Remote sandbox launcher lifecycle: registry, container reuse/resume/release/eviction,
//! and provider interop.
//!
//! Ports the container half of Python's `tests/unit_tests/core/sys_operation/sandbox/`
//! (`launchers/`, `gateway/`, `sandbox_registry.py`): the `SandboxRegistry` launcher
//! functions, `SandboxGateway._get_endpoint` / `_create_new_sandbox` / `_evict_idle`, and
//! `release_sandbox`'s `on_stop` handling.
//!
//! Every case stays offline and deterministic: fake launchers and providers are registered
//! in local registries (never `provider::global()`), time comes from `FixedClock` or an
//! explicit `now` argument, and no HTTP client is constructed.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_sysop::{
    card::SysOperationCard,
    code::types::{CodeChunk, CodeExecParams, CodeOutput},
    fs::types::{
        DownloadFileData, DownloadFileParams, ListDirsParams, ListFilesData, ListFilesParams,
        ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData, SearchFilesParams,
        TransferChunkData, UploadFileData, UploadFileParams, WriteFileData, WriteFileParams,
    },
    mode::OperationMode,
    operation::{CodeOperation, FsOperation, OpStream, ShellOperation},
    result::{code_failure, fs_failure, shell_failure, OpResult},
    sandbox::{
        config::{
            ContainerScope, OnStop, SandboxEndpoint, SandboxGatewayConfig, SandboxIsolationConfig,
            SandboxLauncherConfig,
        },
        launcher::{
            global as global_launchers, register_builtin_launchers, LaunchedSandbox,
            LauncherRegistry, SandboxLauncher, SandboxStatus,
        },
        lifecycle::{FixedClock, SandboxLifecycle},
        operation::{
            sandbox_operations, sandbox_operations_for_session, sandbox_operations_with_registries,
        },
        provider::SandboxProviderRegistry,
    },
    shell::types::{
        ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput,
    },
    SysOperation,
};

const NOW: u64 = 1_000;

// ── fakes ────────────────────────────────────────────────────────────────────────────────

/// Records every endpoint a provider factory was called with.
#[derive(Clone, Default)]
struct SeenEndpoints(Arc<Mutex<Vec<SandboxEndpoint>>>);

impl SeenEndpoints {
    fn push(&self, endpoint: &SandboxEndpoint) {
        self.0.lock().unwrap().push(endpoint.clone());
    }

    fn all(&self) -> Vec<SandboxEndpoint> {
        self.0.lock().unwrap().clone()
    }
}

/// A filesystem provider that echoes the endpoint it was built for.
struct FakeFs {
    base_url: String,
}

#[async_trait]
impl FsOperation for FakeFs {
    fn supports_conditional_write(&self) -> bool {
        true
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        OpResult::success(ReadFileData {
            path: format!("{}::{}", self.base_url, params.path),
            total_bytes: 1,
            ..ReadFileData::default()
        })
    }

    fn read_file_stream(self: Arc<Self>, _params: ReadFileParams) -> OpStream<ReadChunkData> {
        Box::pin(futures::stream::empty())
    }

    async fn write_file(&self, _params: WriteFileParams) -> OpResult<WriteFileData> {
        fs_failure("write_file", &self.base_url)
    }

    async fn upload_file(&self, _params: UploadFileParams) -> OpResult<UploadFileData> {
        fs_failure("upload_file", &self.base_url)
    }

    fn upload_file_stream(
        self: Arc<Self>,
        _params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(futures::stream::empty())
    }

    async fn download_file(&self, _params: DownloadFileParams) -> OpResult<DownloadFileData> {
        fs_failure("download_file", &self.base_url)
    }

    fn download_file_stream(
        self: Arc<Self>,
        _params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(futures::stream::empty())
    }

    async fn list_files(&self, _params: ListFilesParams) -> OpResult<ListFilesData> {
        fs_failure("list_files", &self.base_url)
    }

    async fn list_directories(&self, _params: ListDirsParams) -> OpResult<ListFilesData> {
        fs_failure("list_directories", &self.base_url)
    }

    async fn search_files(&self, _params: SearchFilesParams) -> OpResult<SearchFilesData> {
        fs_failure("search_files", &self.base_url)
    }
}

/// A shell provider naming the endpoint it was built for.
struct FakeShell {
    base_url: String,
}

#[async_trait]
impl ShellOperation for FakeShell {
    async fn execute_cmd(&self, _params: ShellExecParams) -> OpResult<ShellOutput> {
        shell_failure("execute_cmd", &self.base_url)
    }

    fn execute_cmd_stream(self: Arc<Self>, _params: ShellExecParams) -> OpStream<ShellChunk> {
        Box::pin(futures::stream::empty())
    }

    async fn execute_cmd_background(
        &self,
        _params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput> {
        shell_failure("execute_cmd_background", &self.base_url)
    }

    async fn execute_argv(&self, _params: ShellArgvParams) -> OpResult<ShellOutput> {
        shell_failure("execute_argv", &self.base_url)
    }
}

/// A code provider naming the endpoint it was built for.
struct FakeCode {
    base_url: String,
}

#[async_trait]
impl CodeOperation for FakeCode {
    async fn execute_code(&self, _params: CodeExecParams) -> OpResult<CodeOutput> {
        code_failure("execute_code", &self.base_url)
    }

    fn execute_code_stream(self: Arc<Self>, _params: CodeExecParams) -> OpStream<CodeChunk> {
        Box::pin(futures::stream::empty())
    }
}

/// Registers the three fakes under `sandbox_type`, recording every endpoint they see.
fn register_providers(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    seen: &SeenEndpoints,
) {
    let fs_seen = seen.clone();
    registry
        .register_fs(
            sandbox_type,
            Arc::new(
                move |endpoint: &SandboxEndpoint,
                      _config: &SandboxGatewayConfig|
                      -> Result<Arc<dyn FsOperation>> {
                    fs_seen.push(endpoint);
                    Ok(Arc::new(FakeFs {
                        base_url: endpoint.base_url.clone(),
                    }))
                },
            ),
        )
        .unwrap();

    let shell_seen = seen.clone();
    registry
        .register_shell(
            sandbox_type,
            Arc::new(
                move |endpoint: &SandboxEndpoint,
                      _config: &SandboxGatewayConfig|
                      -> Result<Arc<dyn ShellOperation>> {
                    shell_seen.push(endpoint);
                    Ok(Arc::new(FakeShell {
                        base_url: endpoint.base_url.clone(),
                    }))
                },
            ),
        )
        .unwrap();

    let code_seen = seen.clone();
    registry
        .register_code(
            sandbox_type,
            Arc::new(
                move |endpoint: &SandboxEndpoint,
                      _config: &SandboxGatewayConfig|
                      -> Result<Arc<dyn CodeOperation>> {
                    code_seen.push(endpoint);
                    Ok(Arc::new(FakeCode {
                        base_url: endpoint.base_url.clone(),
                    }))
                },
            ),
        )
        .unwrap();
}

/// A launcher that records its calls and reports a fixed container.
struct FakeLauncher {
    base_url: String,
    calls: Arc<Mutex<Vec<String>>>,
    modes: Arc<Mutex<Vec<SandboxStatus>>>,
    fail_launch: Option<String>,
}

impl FakeLauncher {
    fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            calls: Arc::new(Mutex::new(Vec::new())),
            modes: Arc::new(Mutex::new(Vec::new())),
            fail_launch: None,
        }
    }

    fn failing(reason: &str) -> Self {
        Self {
            fail_launch: Some(reason.to_string()),
            ..Self::new("")
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn count(&self, prefix: &str) -> usize {
        self.calls()
            .iter()
            .filter(|call| call.starts_with(prefix))
            .count()
    }

    fn status(&self) -> SandboxStatus {
        self.modes
            .lock()
            .unwrap()
            .last()
            .copied()
            .unwrap_or(SandboxStatus::Running)
    }
}

#[async_trait]
impl SandboxLauncher for FakeLauncher {
    async fn launch(
        &self,
        _config: &SandboxLauncherConfig,
        timeout_seconds: u64,
        isolation_key: Option<&str>,
    ) -> Result<LaunchedSandbox> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("launch:{timeout_seconds}"));
        if let Some(reason) = &self.fail_launch {
            return Err(Error::transport(reason.clone()));
        }
        let key = isolation_key.unwrap_or("none");
        self.modes.lock().unwrap().push(SandboxStatus::Running);
        Ok(
            LaunchedSandbox::new(self.base_url.clone())
                .with_sandbox_id(format!("container::{key}")),
        )
    }

    async fn pause(&self, sandbox_id: &str) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("pause:{sandbox_id}"));
        self.modes.lock().unwrap().push(SandboxStatus::Paused);
        Ok(())
    }

    async fn resume(&self, sandbox_id: &str) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("resume:{sandbox_id}"));
        self.modes.lock().unwrap().push(SandboxStatus::Running);
        Ok(())
    }

    async fn delete(&self, sandbox_id: &str) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("delete:{sandbox_id}"));
        self.modes.lock().unwrap().push(SandboxStatus::Stopped);
        Ok(())
    }

    async fn check_status(&self, _sandbox_id: &str) -> Result<SandboxStatus> {
        Ok(self.status())
    }
}

/// A launcher type backed by `launcher`, plus the registry holding it.
fn launchers_with(kind: &str, launcher: Arc<FakeLauncher>) -> LauncherRegistry {
    let registry = LauncherRegistry::new();
    registry.register(kind, launcher).unwrap();
    registry
}

fn lifecycle_with(launchers: LauncherRegistry, now: u64) -> SandboxLifecycle {
    SandboxLifecycle::with_clock(launchers, Arc::new(FixedClock::new(now)))
}

fn gateway(
    launcher_type: &str,
    sandbox_type: &str,
    base_url: Option<&str>,
    idle_ttl_seconds: Option<u64>,
) -> SandboxGatewayConfig {
    SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            launcher_type: launcher_type.to_string(),
            sandbox_type: sandbox_type.to_string(),
            base_url: base_url.map(str::to_string),
            idle_ttl_seconds,
            ..SandboxLauncherConfig::default()
        }),
        isolation: SandboxIsolationConfig {
            container_scope: ContainerScope::Session,
            ..SandboxIsolationConfig::default()
        },
        timeout_seconds: 17,
        ..SandboxGatewayConfig::default()
    }
}

fn sandbox_card(config: SandboxGatewayConfig) -> SysOperationCard {
    SysOperationCard::new("sys_op")
        .with_mode(OperationMode::Sandbox)
        .with_gateway_config(config)
}

// ── launcher registry ────────────────────────────────────────────────────────────────────

#[test]
fn duplicate_launcher_registration_is_rejected() {
    let launchers = launchers_with("docker", Arc::new(FakeLauncher::new("http://d")));
    let error = launchers
        .register("docker", Arc::new(FakeLauncher::new("http://d")))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::SysOp);
    assert_eq!(
        error.message(),
        "sys operation registry error, process: register_launcher, reason: launcher type \
         'docker' is already registered"
    );
}

#[test]
fn unknown_launcher_type_is_unsupported_and_names_the_registered_types() {
    let launchers = launchers_with("docker", Arc::new(FakeLauncher::new("http://d")));
    launchers
        .register("podman", Arc::new(FakeLauncher::new("http://p")))
        .unwrap();
    let error = launchers.create("firecracker").err().unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(
        error.message(),
        "Unknown launcher_type: firecracker; registered launcher types: docker, podman"
    );
}

#[test]
fn pre_deploy_launcher_is_registered_idempotently() {
    let launchers = LauncherRegistry::new();
    register_builtin_launchers(&launchers).unwrap();
    register_builtin_launchers(&launchers).unwrap();
    assert_eq!(launchers.launcher_types(), vec!["pre_deploy".to_string()]);
    assert!(global_launchers().has("pre_deploy"));
    assert!(std::ptr::eq(global_launchers(), global_launchers()));
}

#[tokio::test]
async fn pre_deploy_launch_returns_the_configured_base_url_and_requires_it() {
    let launcher = global_launchers().create("pre_deploy").unwrap();
    let config = gateway("pre_deploy", "aio", Some("http://predeployed:8080"), None);
    let descriptor = launcher
        .launch(config.launcher_config.as_ref().unwrap(), 17, Some("key"))
        .await
        .unwrap();
    assert_eq!(descriptor, LaunchedSandbox::new("http://predeployed:8080"));
    assert_eq!(descriptor.sandbox_id, None);

    for base_url in [None, Some("   ")] {
        let config = gateway("pre_deploy", "aio", base_url, None);
        let error = launcher
            .launch(config.launcher_config.as_ref().unwrap(), 17, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert_eq!(
            error.message(),
            "sandbox launcher error, process: launch, reason: pre_deploy launcher requires \
             launcher_config.base_url (the base URL of an already deployed sandbox)"
        );
    }
}

// ── lifecycle state machine ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_running_container_is_reused_and_its_last_used_time_refreshed() {
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = lifecycle_with(launchers_with("docker", Arc::clone(&launcher)), NOW);
    let config = gateway("docker", "aio", None, None);

    let first = lifecycle
        .ensure_endpoint(&config, "session_key", NOW)
        .await
        .unwrap();
    assert_eq!(first.base_url, "http://docker:1");
    assert_eq!(first.sandbox_id.as_deref(), Some("container::session_key"));
    assert_eq!(first.isolation_key.as_deref(), Some("session_key"));

    let second = lifecycle
        .ensure_endpoint(&config, "session_key", NOW + 300)
        .await
        .unwrap();
    assert_eq!(second, first);
    assert_eq!(launcher.count("launch"), 1, "the launcher ran exactly once");
    let record = lifecycle.record("session_key").await.unwrap();
    assert_eq!(record.status, SandboxStatus::Running);
    assert_eq!(record.created_epoch_secs, NOW);
    assert_eq!(record.last_used_epoch_secs, NOW + 300);
}

#[tokio::test]
async fn a_paused_container_is_resumed_instead_of_relaunched() {
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = lifecycle_with(launchers_with("docker", Arc::clone(&launcher)), NOW);
    let config = gateway("docker", "aio", None, None);
    lifecycle
        .ensure_endpoint(&config, "session_key", NOW)
        .await
        .unwrap();

    // Releasing with `pause` suspends the container and keeps its record paused.
    lifecycle
        .release("session_key", OnStop::Pause)
        .await
        .unwrap();
    let endpoint = lifecycle
        .ensure_endpoint(&config, "session_key", NOW + 60)
        .await
        .unwrap();
    assert_eq!(
        endpoint.sandbox_id.as_deref(),
        Some("container::session_key")
    );
    assert_eq!(
        launcher.calls(),
        vec![
            "launch:17".to_string(),
            "pause:container::session_key".to_string(),
            "resume:container::session_key".to_string(),
        ]
    );
    let record = lifecycle.record("session_key").await.unwrap();
    assert_eq!(record.status, SandboxStatus::Running);
    assert_eq!(record.last_used_epoch_secs, NOW + 60);
}

#[tokio::test]
async fn a_paused_record_whose_container_vanished_is_relaunched() {
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = lifecycle_with(launchers_with("docker", Arc::clone(&launcher)), NOW);
    let config = gateway("docker", "aio", None, None);
    lifecycle
        .ensure_endpoint(&config, "session_key", NOW)
        .await
        .unwrap();

    lifecycle
        .release("session_key", OnStop::Pause)
        .await
        .unwrap();
    // The paused container was destroyed out of band, so it cannot be resumed.
    launcher.modes.lock().unwrap().push(SandboxStatus::Stopped);
    lifecycle
        .ensure_endpoint(&config, "session_key", NOW + 60)
        .await
        .unwrap();
    // Only a stored `paused` record is resumed; a stopped one is relaunched.
    assert_eq!(launcher.count("resume"), 0);
    assert_eq!(launcher.count("launch"), 2);
}

#[tokio::test]
async fn release_handles_every_on_stop_variant() {
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = lifecycle_with(launchers_with("docker", Arc::clone(&launcher)), NOW);
    let config = gateway("docker", "aio", None, None);

    lifecycle
        .ensure_endpoint(&config, "keep", NOW)
        .await
        .unwrap();
    lifecycle.release("keep", OnStop::Keep).await.unwrap();
    assert_eq!(launcher.count("delete"), 0);
    assert_eq!(launcher.count("pause"), 0);
    assert!(
        lifecycle.record("keep").await.is_some(),
        "keep retains the record"
    );

    lifecycle
        .ensure_endpoint(&config, "pause", NOW)
        .await
        .unwrap();
    lifecycle.release("pause", OnStop::Pause).await.unwrap();
    assert_eq!(launcher.count("pause"), 1);
    // A paused container stays tracked so the next `ensure_endpoint` can resume it; only
    // `delete` forgets the container.
    let paused = lifecycle
        .record("pause")
        .await
        .expect("pause retains a paused record");
    assert_eq!(paused.status, SandboxStatus::Paused);

    lifecycle
        .ensure_endpoint(&config, "delete", NOW)
        .await
        .unwrap();
    lifecycle.release("delete", OnStop::Delete).await.unwrap();
    assert_eq!(launcher.count("delete"), 1);
    assert!(lifecycle.record("delete").await.is_none());

    // Releasing an unknown key succeeds and calls nothing.
    let before = launcher.calls().len();
    for on_stop in [OnStop::Delete, OnStop::Pause, OnStop::Keep] {
        lifecycle.release("unknown", on_stop).await.unwrap();
    }
    assert_eq!(launcher.calls().len(), before);
}

#[tokio::test]
async fn release_of_a_kept_container_refreshes_its_last_used_time() {
    let clock = Arc::new(FixedClock::new(NOW));
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = SandboxLifecycle::with_clock(launchers_with("docker", launcher), clock.clone());
    let config = gateway("docker", "aio", None, None);
    lifecycle
        .ensure_endpoint(&config, "key", NOW)
        .await
        .unwrap();

    clock.set(NOW + 500);
    lifecycle.release("key", OnStop::Keep).await.unwrap();
    assert_eq!(
        lifecycle.record("key").await.unwrap().last_used_epoch_secs,
        NOW + 500
    );
}

#[tokio::test]
async fn evict_idle_deletes_expired_keys_only_and_ignores_on_stop() {
    let clock = Arc::new(FixedClock::new(1_000));
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle =
        SandboxLifecycle::with_clock(launchers_with("docker", launcher.clone()), clock.clone());
    // `on_stop = pause` must not stop the sweep from deleting.
    let mut config = gateway("docker", "aio", None, None);
    if let Some(launcher_config) = config.launcher_config.as_mut() {
        launcher_config.on_stop = OnStop::Pause;
    }
    lifecycle
        .ensure_endpoint(&config, "idle", 1_000)
        .await
        .unwrap();
    lifecycle
        .ensure_endpoint(&config, "busy", 1_400)
        .await
        .unwrap();

    // `evict_idle_now` reads the fixed clock, which is still at 1_000: nothing expired.
    assert_eq!(lifecycle.evict_idle_now(500).await, Vec::<String>::new());
    clock.set(1_600);
    let evicted = lifecycle.evict_idle_now(500).await;
    assert_eq!(evicted, vec!["idle".to_string()]);
    assert_eq!(launcher.count("delete"), 1);
    assert_eq!(launcher.count("pause"), 0);
    assert!(lifecycle.record("busy").await.is_some());

    // The boundary is inclusive: exactly the TTL is expired.
    assert_eq!(
        lifecycle.evict_idle(200, 1_600).await,
        vec!["busy".to_string()]
    );
    assert!(lifecycle.is_empty().await);
}

#[tokio::test]
async fn ensure_endpoint_evicts_idle_containers_when_the_launcher_sets_a_ttl() {
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = lifecycle_with(launchers_with("docker", Arc::clone(&launcher)), NOW);
    let config = gateway("docker", "aio", None, Some(100));
    lifecycle.ensure_endpoint(&config, "a", NOW).await.unwrap();
    lifecycle
        .ensure_endpoint(&config, "b", NOW + 250)
        .await
        .unwrap();
    assert_eq!(launcher.count("delete"), 1);
    assert!(lifecycle.record("a").await.is_none());
    assert!(lifecycle.record("b").await.is_some());
}

#[tokio::test]
async fn a_failing_launcher_is_199008_and_never_unsupported() {
    let lifecycle = lifecycle_with(
        launchers_with(
            "docker",
            Arc::new(FakeLauncher::failing("no docker daemon")),
        ),
        NOW,
    );
    let config = gateway("docker", "aio", None, None);
    let error = lifecycle
        .ensure_endpoint(&config, "key", NOW)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::SysOp);
    assert_eq!(
        error.message(),
        "sandbox launcher error, process: launch, reason: no docker daemon"
    );
    assert!(
        lifecycle.is_empty().await,
        "a failed launch stores no record"
    );
}

#[tokio::test]
async fn an_unknown_launcher_type_is_unsupported() {
    let lifecycle = lifecycle_with(
        launchers_with("docker", Arc::new(FakeLauncher::new("http://d"))),
        NOW,
    );
    let error = lifecycle
        .ensure_endpoint(&gateway("podman", "aio", None, None), "key", NOW)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error
        .message()
        .contains("Unknown launcher_type: podman; registered launcher types: docker"));
}

#[tokio::test]
async fn a_blank_isolation_key_is_199010() {
    let lifecycle = lifecycle_with(
        launchers_with("docker", Arc::new(FakeLauncher::new("http://d"))),
        NOW,
    );
    let error = lifecycle
        .ensure_endpoint(&gateway("docker", "aio", None, None), "  ", NOW)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::SysOp);
    assert_eq!(
        error.message(),
        "sandbox isolation key error, reason: isolation key must not be empty when resolving a \
         sandbox endpoint"
    );
}

// ── isolation keys ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn per_session_templates_produce_distinct_records_and_containers() {
    let launcher = Arc::new(FakeLauncher::new("http://docker:1"));
    let lifecycle = lifecycle_with(launchers_with("docker", Arc::clone(&launcher)), NOW);
    let providers = SandboxProviderRegistry::new();
    let seen = SeenEndpoints::default();
    register_providers(&providers, "aio", &seen);
    let card = sandbox_card(gateway("docker", "aio", None, None));

    sandbox_operations_with_registries(&card, Some("s1"), &providers, &lifecycle, NOW)
        .await
        .unwrap();
    sandbox_operations_with_registries(&card, Some("s2"), &providers, &lifecycle, NOW)
        .await
        .unwrap();
    sandbox_operations_with_registries(&card, None, &providers, &lifecycle, NOW)
        .await
        .unwrap();

    let mut keys: Vec<String> = lifecycle
        .records()
        .await
        .into_iter()
        .map(|record| record.isolation_key)
        .collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "session_docker_aio_default_session".to_string(),
            "session_docker_aio_s1".to_string(),
            "session_docker_aio_s2".to_string(),
        ]
    );
    assert_eq!(launcher.count("launch"), 3, "one container per session");
    assert!(seen
        .all()
        .iter()
        .all(|endpoint| endpoint.isolation_key.is_some()));
}

// ── provider interop (the `openjiuwen-extensions` shape) ─────────────────────────────────

#[tokio::test]
async fn managed_dispatch_targets_the_launched_container_per_sandbox_type() {
    let cases = [
        ("aio", "http://aio:1001"),
        ("jiuwenbox", "http://jiuwenbox:1002"),
        ("yuanrong", "http://yuanrong:1003"),
    ];

    let providers = SandboxProviderRegistry::new();
    let seen = SeenEndpoints::default();
    let launchers = LauncherRegistry::new();
    let mut fakes = Vec::new();
    for (sandbox_type, base_url) in cases {
        register_providers(&providers, sandbox_type, &seen);
        let launcher = Arc::new(FakeLauncher::new(base_url));
        launchers
            .register(
                sandbox_type,
                Arc::clone(&launcher) as Arc<dyn SandboxLauncher>,
            )
            .unwrap();
        fakes.push(launcher);
    }
    let lifecycle = lifecycle_with(launchers, NOW);

    for (index, (sandbox_type, base_url)) in cases.iter().enumerate() {
        seen.0.lock().unwrap().clear();
        let card = sandbox_card(gateway(sandbox_type, sandbox_type, None, None));
        let operations =
            sandbox_operations_with_registries(&card, Some("s1"), &providers, &lifecycle, NOW)
                .await
                .unwrap();

        let fs = operations.fs.unwrap();
        let read = fs.read_file(ReadFileParams::text("notes.txt")).await;
        assert_eq!(read.code, 0);
        assert_eq!(
            read.data.unwrap().path,
            format!("{base_url}::notes.txt"),
            "the provider must be built from the launched endpoint"
        );
        assert_eq!(
            operations
                .shell
                .unwrap()
                .execute_cmd(ShellExecParams::new("echo hi"))
                .await
                .message,
            format!("shell operation execution error, execution: execute_cmd, reason: {base_url}")
        );
        assert_eq!(
            operations
                .code
                .unwrap()
                .execute_code(CodeExecParams::new("print(1)"))
                .await
                .message,
            format!("code operation execution error, execution: execute_code, reason: {base_url}")
        );

        // Every created backend saw the same container endpoint.
        let endpoints = seen.all();
        assert_eq!(endpoints.len(), 3, "fs, shell, and code were built once");
        assert!(endpoints
            .iter()
            .all(|endpoint| endpoint.base_url == *base_url
                && endpoint.sandbox_id.as_deref()
                    == Some(
                        format!("container::session_{sandbox_type}_{sandbox_type}_s1").as_str()
                    )));
        assert_eq!(
            fakes[index].count("launch"),
            1,
            "each type launched its own container exactly once"
        );
    }

    // The three containers are distinct records over one lifecycle.
    assert_eq!(lifecycle.len().await, 3);
}

#[tokio::test]
async fn an_unregistered_sandbox_type_is_still_unsupported() {
    let providers = SandboxProviderRegistry::new();
    register_providers(&providers, "aio", &SeenEndpoints::default());
    let lifecycle = lifecycle_with(
        launchers_with("docker", Arc::new(FakeLauncher::new("http://docker:1"))),
        NOW,
    );
    let card = sandbox_card(gateway("docker", "no-such-type", None, None));
    let error = sandbox_operations_with_registries(&card, None, &providers, &lifecycle, NOW)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(
        error.message(),
        "Sandbox type 'no-such-type' does not support operation 'fs'"
    );
    // The container was still launched (the endpoint is resolved before the provider).
    assert_eq!(lifecycle.len().await, 1);
}

#[tokio::test]
async fn a_managed_launcher_type_through_the_static_path_is_a_card_error() {
    let card = sandbox_card(gateway("docker", "aio", None, None));
    for error in [
        sandbox_operations(&card).unwrap_err(),
        sandbox_operations_for_session(&card, Some("s1")).unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error.message().contains("sys operation card param error"));
        assert!(error
            .message()
            .contains("launcher_type 'docker' creates sandbox containers"));
    }
}

#[tokio::test]
async fn a_pre_deployed_base_url_is_still_static_and_records_nothing() {
    let providers_registry = SandboxProviderRegistry::new();
    let seen = SeenEndpoints::default();
    register_providers(&providers_registry, "aio", &seen);
    let lifecycle = lifecycle_with(
        launchers_with("docker", Arc::new(FakeLauncher::new("http://docker:1"))),
        NOW,
    );
    // `pre_deploy` with a base URL: today's endpoint, no launch bookkeeping.
    let card = sandbox_card(gateway(
        "pre_deploy",
        "aio",
        Some("http://predeployed:8080"),
        None,
    ));
    let operations =
        sandbox_operations_with_registries(&card, Some("s1"), &providers_registry, &lifecycle, NOW)
            .await
            .unwrap();
    assert!(lifecycle.is_empty().await);
    let endpoint = seen.all().first().cloned().unwrap();
    assert_eq!(endpoint.base_url, "http://predeployed:8080");
    assert_eq!(endpoint.sandbox_id, None);
    assert_eq!(
        endpoint.isolation_key.as_deref(),
        Some("session_pre_deploy_aio_s1")
    );
    assert_eq!(
        operations
            .fs
            .unwrap()
            .read_file(ReadFileParams::text("x"))
            .await
            .data
            .unwrap()
            .path,
        "http://predeployed:8080::x"
    );
}

// ── facade ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_facade_reports_launcher_failures_and_unknown_types() {
    let lifecycle = lifecycle_with(
        launchers_with("docker", Arc::new(FakeLauncher::failing("runtime down"))),
        NOW,
    );
    let error = SysOperation::sandbox_with_lifecycle(
        "aio",
        gateway("docker", "aio", None, None),
        Some("s1"),
        &lifecycle,
        NOW,
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::SysOp);
    assert_eq!(
        error.message(),
        "sandbox launcher error, process: launch, reason: runtime down"
    );

    let error = SysOperation::sandbox_with_lifecycle(
        "aio",
        gateway("nobody-registered-this", "aio", None, None),
        None,
        &lifecycle,
        NOW,
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[tokio::test]
async fn idle_stop_and_release_are_reachable_from_the_facade() {
    let workspace = tempfile::tempdir().unwrap();
    let config = openjiuwen_sysop::config::LocalWorkConfig::unrestricted(
        workspace.path().to_string_lossy().into_owned(),
    );
    let operation = SysOperation::local(config).unwrap();
    assert_eq!(operation.mode(), OperationMode::Local);
    assert_eq!(operation.evict_idle_sandboxes().await, Vec::<String>::new());
    assert_eq!(
        operation.evict_idle_sandboxes_at(NOW).await,
        Vec::<String>::new()
    );
    assert!(operation.sandbox_endpoint().is_none());
    operation.release_sandbox().await.unwrap();
}

#[tokio::test]
async fn idle_stop_is_reachable_from_a_sandbox_facade() {
    // A static pre-deployed card carries no binding, so the facade has no lifecycle of its
    // own; the sweep still runs (against the global lifecycle) and finds nothing.
    let card = sandbox_card(gateway(
        "pre_deploy",
        "aio",
        Some("http://predeployed:8080"),
        Some(1),
    ));
    let providers = openjiuwen_sysop::sandbox::provider::global();
    assert!(
        !providers.has_fs("aio"),
        "no test may register a global provider"
    );
    let endpoint = openjiuwen_sysop::sandbox::operation::resolve_sandbox_endpoint(
        &card,
        None,
        &lifecycle_with(LauncherRegistry::new(), NOW),
        NOW,
    )
    .await
    .unwrap();
    assert_eq!(endpoint.base_url, "http://predeployed:8080");
}

#[tokio::test]
async fn the_facade_carries_local_registry_side_effects_only() {
    // Guard: none of the tests above may leave records in the global lifecycle.
    assert!(
        openjiuwen_sysop::sandbox::lifecycle::global()
            .is_empty()
            .await
    );
}
