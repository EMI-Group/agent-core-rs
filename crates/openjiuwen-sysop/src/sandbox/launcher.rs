//! The sandbox launcher seam: container lifecycle calls plus their registry.
//!
//! Python parity: `core/sys_operation/sandbox/launchers/base.py` (`LaunchedSandbox`,
//! `SandboxLauncher`), `sandbox/launchers/pre_deployment_launcher.py`
//! (`PreDeploymentLauncher`), and the launcher half of
//! `sandbox/sandbox_registry.py` (`register_launcher`, `get_launcher_cls`,
//! `create_launcher`, `unregister_launcher`).
//!
//! A launcher turns a [`SandboxLauncherConfig`] into a running container and is the only
//! component that knows how a deployment is started (`docker run`, an AIO create call, a
//! JiuwenBox/YuanRong environment request, ...). A launcher failure *while doing its job* is
//! mapped onto Python's `sandbox launcher error` (code 199008), while an operation the
//! launcher cannot perform at all stays
//! [`openjiuwen_common::error::ErrorKind::Unsupported`] (see
//! [`unsupported_launcher_error`]), so a caller can tell "this runtime cannot do that" from
//! "the runtime exists and failed".
//!
//! Only [`PreDeploymentLauncher`] ships with this crate: it points at an already deployed
//! sandbox (`launcher_config.base_url`) and cannot suspend, resume, or destroy that
//! container, so those calls report a typed refusal instead of a silent success.
//! Container-creating launchers are host extensions registered through
//! [`LauncherRegistry::register`] — [`global`] is the process-wide instance the
//! [`crate::SysOperation`] facade uses.
//!
//! ```
//! use openjiuwen_sysop::sandbox::launcher::{LauncherRegistry, LaunchedSandbox, SandboxLauncher};
//! use openjiuwen_sysop::sandbox::config::SandboxLauncherConfig;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! struct Fixed;
//!
//! #[async_trait::async_trait]
//! impl SandboxLauncher for Fixed {
//!     async fn launch(
//!         &self,
//!         _config: &SandboxLauncherConfig,
//!         _timeout_seconds: u64,
//!         _isolation_key: Option<&str>,
//!     ) -> openjiuwen_common::error::Result<LaunchedSandbox> {
//!         Ok(LaunchedSandbox::new("http://sandbox.internal:8080"))
//!     }
//! }
//!
//! let registry = LauncherRegistry::new();
//! registry.register("docker", std::sync::Arc::new(Fixed))?;
//! let launcher = registry.create("docker")?;
//! let launched = launcher
//!     .launch(&SandboxLauncherConfig::default(), 30, Some("session_key"))
//!     .await?;
//! assert_eq!(launched.base_url, "http://sandbox.internal:8080");
//! assert!(registry.create("nope").is_err());
//! # Ok(())
//! # }
//! ```
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, OnceLock, RwLock},
};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_LAUNCHER_TYPE;
use crate::result::OpResult;
use crate::sandbox::config::SandboxLauncherConfig;
use crate::status::{
    isolation_key_message, registry_message, sandbox_gateway_message, sandbox_launcher_message,
    sandbox_provider_message, unsupported_launcher_message, SysOpStatus,
};

/// Lifecycle state of one sandbox container.
///
/// Python parity: `SandboxStatus` (`sandbox/gateway/sandbox_store.py`), whose members are
/// `RUNNING`, `PAUSED`, and `KILLED`; `KILLED` is spelled [`Self::Stopped`] here because a
/// container can also have exited on its own, and [`Self::Unknown`] is added because a
/// launcher may be unable to report a state (Python's `check_status` never returns one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SandboxStatus {
    /// The container is running and can serve operations.
    Running,
    /// The container is suspended; `resume` makes it usable again.
    Paused,
    /// The container was deleted or exited (`SandboxStatus.KILLED` in Python).
    Stopped,
    /// The state could not be determined (for example a launcher that reports nothing).
    #[default]
    Unknown,
}

impl SandboxStatus {
    /// The serialized name (`"running"`, `"paused"`, `"stopped"`, `"unknown"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::Unknown => "unknown",
        }
    }

    /// Whether the container can serve operations without a resume.
    #[must_use]
    pub const fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }
}

impl fmt::Display for SandboxStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The handle a [`SandboxLauncher::launch`] returns for a started sandbox.
///
/// Python parity: `LaunchedSandbox` (`launchers/base.py`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LaunchedSandbox {
    /// HTTP base URL of the sandbox service (empty when the runtime manages the endpoint).
    pub base_url: String,
    /// Runtime-assigned sandbox identifier (a container id, an E2B sandbox id, ...).
    pub sandbox_id: Option<String>,
    /// Host-side mapped port (container launchers only).
    pub host_port: Option<u16>,
}

impl LaunchedSandbox {
    /// A descriptor with only a base URL, which is all the `pre_deploy` launcher reports.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            sandbox_id: None,
            host_port: None,
        }
    }

    /// Adds the runtime-assigned sandbox id.
    #[must_use]
    pub fn with_sandbox_id(mut self, sandbox_id: impl Into<String>) -> Self {
        self.sandbox_id = Some(sandbox_id.into());
        self
    }
}

/// Starts, suspends, resumes, and destroys sandbox containers.
///
/// Python parity: `SandboxLauncher` (`launchers/base.py`). Only [`Self::launch`] must be
/// implemented. The other calls default to *typed refusals* instead of silent successes:
/// [`Self::pause`], [`Self::resume`], and [`Self::delete`] return
/// [`ErrorKind::Unsupported`] naming the launcher and the operation, and
/// [`Self::check_status`] reports [`SandboxStatus::Unknown`]. A launcher overrides exactly
/// the calls its runtime can perform; one that leaves a default in place makes a teardown
/// fail loudly instead of leaking a container behind an `Ok(())`.
#[async_trait]
pub trait SandboxLauncher: Send + Sync {
    /// The `launcher_type` this implementation manages (`"custom"` when not overridden).
    ///
    /// Defaulted so host implementations keep compiling. It appears only in diagnostics —
    /// the unsupported-operation messages and the lifecycle's leaked-container warnings —
    /// so a launcher registered under a specific key should override it (as
    /// [`PreDeploymentLauncher`] does with `"pre_deploy"`).
    fn launcher_type(&self) -> &'static str {
        "custom"
    }

    /// Starts (or resumes) a sandbox and returns its descriptor.
    ///
    /// `timeout_seconds` bounds how long the deployment may take and
    /// `isolation_key` identifies the container so a later `launch` for the same key can
    /// find it again instead of creating a duplicate.
    ///
    /// # Errors
    ///
    /// Returns the launcher's own failure; [`crate::sandbox::lifecycle::SandboxLifecycle`]
    /// maps it onto code 199008.
    async fn launch(
        &self,
        config: &SandboxLauncherConfig,
        timeout_seconds: u64,
        isolation_key: Option<&str>,
    ) -> Result<LaunchedSandbox>;

    /// Suspends the sandbox, keeping its state without consuming compute.
    ///
    /// # Errors
    ///
    /// The default returns [`ErrorKind::Unsupported`]
    /// ([`unsupported_launcher_error`]) because a launcher that cannot suspend a container
    /// must not report a suspension it did not perform; the sandbox keeps running. An
    /// override returns its own failure, which the caller maps onto code 199008.
    async fn pause(&self, _sandbox_id: &str) -> Result<()> {
        Err(unsupported_launcher_error(
            self.launcher_type(),
            "pause",
            "this launcher cannot suspend containers; the sandbox keeps running",
        ))
    }

    /// Resumes a previously paused sandbox.
    ///
    /// # Errors
    ///
    /// The default returns [`ErrorKind::Unsupported`]
    /// ([`unsupported_launcher_error`]) because a launcher that cannot resume a container
    /// must not report one it did not revive. An override returns its own failure, which
    /// the caller maps onto code 199008.
    async fn resume(&self, _sandbox_id: &str) -> Result<()> {
        Err(unsupported_launcher_error(
            self.launcher_type(),
            "resume",
            "this launcher cannot resume containers; the sandbox stays suspended",
        ))
    }

    /// Destroys the sandbox and releases its resources.
    ///
    /// # Errors
    ///
    /// The default returns [`ErrorKind::Unsupported`]
    /// ([`unsupported_launcher_error`]) because a launcher that cannot destroy a container
    /// must not report a teardown it did not perform — the container would leak. An
    /// override returns its own failure, which the caller maps onto code 199008.
    async fn delete(&self, _sandbox_id: &str) -> Result<()> {
        Err(unsupported_launcher_error(
            self.launcher_type(),
            "delete",
            "this launcher cannot destroy containers; teardown would leak the sandbox",
        ))
    }

    /// Reports the current state of the sandbox.
    ///
    /// The default reports [`SandboxStatus::Unknown`], which is what a launcher that cannot
    /// inspect its runtime must say; it is never a claim that the container is running.
    /// Override it to report a real state: [`crate::sandbox::lifecycle::SandboxLifecycle`]
    /// resumes a `Paused`/`Unknown` record and relaunches a [`SandboxStatus::Stopped`] one.
    ///
    /// # Errors
    ///
    /// An override returns its own failure, which the caller maps onto code 199008.
    async fn check_status(&self, _sandbox_id: &str) -> Result<SandboxStatus> {
        Ok(SandboxStatus::Unknown)
    }
}

/// The `pre_deploy` launcher: the sandbox already exists at `launcher_config.base_url`.
///
/// Python parity: `PreDeploymentLauncher` (`launchers/pre_deployment_launcher.py`). Python's
/// `delete` tears the container down through the extension providers (`YuanRong` /
/// `JiuwenBox` shared-scope keys); that teardown is deferred, and this launcher owns no
/// handle on the container at all. It therefore deliberately inherits the trait's typed
/// refusals: `pause`, `resume`, and `delete` return
/// [`ErrorKind::Unsupported`] and `check_status` reports [`SandboxStatus::Unknown`],
/// instead of claiming a teardown or a state it cannot know.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreDeploymentLauncher;

#[async_trait]
impl SandboxLauncher for PreDeploymentLauncher {
    fn launcher_type(&self) -> &'static str {
        DEFAULT_LAUNCHER_TYPE
    }

    async fn launch(
        &self,
        config: &SandboxLauncherConfig,
        timeout_seconds: u64,
        isolation_key: Option<&str>,
    ) -> Result<LaunchedSandbox> {
        let base_url = config.base_url.as_deref().unwrap_or_default().trim();
        if base_url.is_empty() {
            return Err(launcher_error(
                "launch",
                "pre_deploy launcher requires launcher_config.base_url (the base URL of an \
                 already deployed sandbox)",
            ));
        }
        tracing::debug!(
            base_url,
            timeout_seconds,
            isolation_key,
            "resolved a pre-deployed sandbox endpoint"
        );
        Ok(LaunchedSandbox::new(base_url))
    }
}

/// Registry of sandbox launchers, keyed by `launcher_type`.
///
/// Python parity: `SandboxRegistry._launchers` with `register_launcher`,
/// `get_launcher_cls`, `create_launcher`, and `unregister_launcher`. Python stores classes
/// and instantiates per call; this registry stores ready instances (launchers are stateless
/// seams), which is cheaper and lets a host inject configuration. The registry is cheap to
/// clone and safe to share; the process-global instance is [`global`].
#[derive(Clone, Default)]
pub struct LauncherRegistry {
    launchers: Arc<RwLock<BTreeMap<String, Arc<dyn SandboxLauncher>>>>,
}

impl fmt::Debug for LauncherRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LauncherRegistry")
            .field("launchers", &self.launcher_types())
            .finish()
    }
}

impl LauncherRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `launcher` under `launcher_type`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (code 199006) when the launcher type is already
    /// registered, matching the provider registry's duplicate handling.
    pub fn register(
        &self,
        launcher_type: impl Into<String>,
        launcher: Arc<dyn SandboxLauncher>,
    ) -> Result<()> {
        let launcher_type = launcher_type.into();
        let mut map = self
            .launchers
            .write()
            .map_err(|_| poisoned("register_launcher"))?;
        if map.contains_key(&launcher_type) {
            return Err(Error::sys_op(registry_message(
                "register_launcher",
                &format!("launcher type '{launcher_type}' is already registered"),
            )));
        }
        map.insert(launcher_type, launcher);
        Ok(())
    }

    /// Whether a launcher is registered under `launcher_type`.
    #[must_use]
    pub fn has(&self, launcher_type: &str) -> bool {
        self.launchers
            .read()
            .is_ok_and(|map| map.contains_key(launcher_type))
    }

    /// The launcher registered under `launcher_type`, if any.
    #[must_use]
    pub fn get(&self, launcher_type: &str) -> Option<Arc<dyn SandboxLauncher>> {
        self.launchers
            .read()
            .ok()
            .and_then(|map| map.get(launcher_type).cloned())
    }

    /// The launcher registered under `launcher_type`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Unsupported` when no launcher type matches, naming the
    /// registered types (Python raises `ValueError("Unknown launcher_type: ...")`).
    pub fn create(&self, launcher_type: &str) -> Result<Arc<dyn SandboxLauncher>> {
        let map = self
            .launchers
            .read()
            .map_err(|_| poisoned("create_launcher"))?;
        map.get(launcher_type).cloned().ok_or_else(|| {
            unknown_launcher_error(launcher_type, &map.keys().cloned().collect::<Vec<_>>())
        })
    }

    /// Removes the launcher registered under `launcher_type`, if any.
    pub fn unregister(&self, launcher_type: &str) {
        if let Ok(mut map) = self.launchers.write() {
            map.remove(launcher_type);
        }
    }

    /// The registered launcher types, sorted.
    #[must_use]
    pub fn launcher_types(&self) -> Vec<String> {
        self.launchers
            .read()
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// The process-global launcher registry, pre-populated with [`register_builtin_launchers`].
///
/// Python parity: `SandboxRegistry._launchers` is class-level state and
/// `SandboxGateway.__init__` registers `PreDeploymentLauncher` under `"pre_deploy"`. A
/// local [`LauncherRegistry`] should be used in tests and for isolation.
#[must_use]
pub fn global() -> &'static LauncherRegistry {
    static REGISTRY: OnceLock<LauncherRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry = LauncherRegistry::new();
        if let Err(error) = register_builtin_launchers(&registry) {
            tracing::warn!(error = %error, "failed to register the built-in sandbox launchers");
        }
        registry
    })
}

/// Registers [`PreDeploymentLauncher`] under `"pre_deploy"`.
///
/// Idempotent: a registry that already holds a `"pre_deploy"` launcher (including a host
/// implementation) is left untouched, so repeated calls succeed.
///
/// # Errors
///
/// Propagates registration failures other than an existing `"pre_deploy"` entry.
pub fn register_builtin_launchers(registry: &LauncherRegistry) -> Result<()> {
    if registry.has(DEFAULT_LAUNCHER_TYPE) {
        return Ok(());
    }
    match registry.register(DEFAULT_LAUNCHER_TYPE, Arc::new(PreDeploymentLauncher)) {
        Ok(()) => Ok(()),
        Err(error) if registry.has(DEFAULT_LAUNCHER_TYPE) => {
            // A concurrent registration won the race: the outcome is the same.
            tracing::debug!(error = %error, "pre_deploy launcher was registered concurrently");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

/// Builds the `Unsupported` error for an unregistered launcher type.
///
/// Names the registered types so a caller can fix the configuration from the message
/// alone.
pub fn unknown_launcher_error(launcher_type: &str, registered: &[String]) -> Error {
    let registered = if registered.is_empty() {
        "none".to_string()
    } else {
        registered.join(", ")
    };
    Error::unsupported(format!(
        "Unknown launcher_type: {launcher_type}; registered launcher types: {registered}"
    ))
}

/// A sandbox launcher failure: `ErrorKind::SysOp` with Python's code 199008 text.
pub fn launcher_error(process: &str, reason: &str) -> Error {
    Error::sys_op(sandbox_launcher_message(process, reason))
}

/// An operation the launcher cannot perform at all:
/// `ErrorKind::Unsupported` naming the launcher and the operation.
///
/// Deliberately *not* code 199008: 199008 means "the launcher exists and the call failed",
/// which is a runtime problem a caller may retry. A capability the launcher does not have
/// (destroying a pre-deployed container, suspending a runtime without pause, ...) is
/// permanent, and pretending otherwise is how a teardown silently leaks a sandbox. The
/// message is
/// `sandbox launcher unsupported, launcher: {launcher}, process: {process}, reason: {reason}`.
pub fn unsupported_launcher_error(launcher: &str, process: &str, reason: &str) -> Error {
    Error::unsupported(unsupported_launcher_message(launcher, process, reason))
}

/// A sandbox provider failure: `ErrorKind::SysOp` with Python's code 199009 text.
pub fn provider_error(process: &str, reason: &str) -> Error {
    Error::sys_op(sandbox_provider_message(process, reason))
}

/// An isolation-key failure: `ErrorKind::SysOp` with Python's code 199010 text.
pub fn isolation_key_error(reason: &str) -> Error {
    Error::sys_op(isolation_key_message(reason))
}

/// A sandbox gateway failure: `ErrorKind::SysOp` with Python's code 199007 text.
pub fn gateway_error(operation: &str, reason: &str) -> Error {
    Error::sys_op(sandbox_gateway_message(operation, reason))
}

/// Renders a failure as a code 199008 [`OpResult`] value (never panics).
///
/// [`OpResult`] has no `Unsupported` code, so a genuinely missing transport must stay an
/// [`Error`]: use [`launcher_op_result`] rather than calling this directly when the failure
/// may be [`ErrorKind::Unsupported`].
#[must_use]
pub fn launcher_op_failure<T>(operation: &str, error: &Error) -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::SandboxLauncherError,
        sandbox_launcher_message(operation, error.message().as_ref()),
    )
}

/// Converts a launcher outcome into an [`OpResult`] value.
///
/// Launcher failures become code 199008 values, because a sys-operation reports failures as
/// values. A missing transport ([`ErrorKind::Unsupported`]) is returned as an error, so it
/// can never be flattened into a value a caller might mistake for a provider answer.
///
/// # Errors
///
/// Returns the original error when it is [`ErrorKind::Unsupported`].
pub fn launcher_op_result<T>(operation: &str, result: Result<T>) -> Result<OpResult<T>> {
    match result {
        Ok(value) => Ok(OpResult::success(value)),
        Err(error) if error.kind() == ErrorKind::Unsupported => Err(error),
        Err(error) => Ok(launcher_op_failure(operation, &error)),
    }
}

/// Maps a poisoned launcher-registry lock onto the registry error (code 199006).
fn poisoned(operation: &str) -> Error {
    Error::sys_op(registry_message(
        "sandbox_launcher",
        &format!("registry lock poisoned while accessing '{operation}'"),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// A launcher that records how often it was called and reports a fixed descriptor.
    struct CountingLauncher {
        launches: AtomicU32,
        status: SandboxStatus,
    }

    #[async_trait]
    impl SandboxLauncher for CountingLauncher {
        fn launcher_type(&self) -> &'static str {
            "counting"
        }

        async fn launch(
            &self,
            config: &SandboxLauncherConfig,
            _timeout_seconds: u64,
            isolation_key: Option<&str>,
        ) -> Result<LaunchedSandbox> {
            self.launches.fetch_add(1, Ordering::SeqCst);
            Ok(
                LaunchedSandbox::new(config.base_url.clone().unwrap_or_default())
                    .with_sandbox_id(isolation_key.unwrap_or("none")),
            )
        }

        async fn pause(&self, _sandbox_id: &str) -> Result<()> {
            Ok(())
        }

        async fn resume(&self, _sandbox_id: &str) -> Result<()> {
            Ok(())
        }

        async fn delete(&self, _sandbox_id: &str) -> Result<()> {
            Ok(())
        }

        async fn check_status(&self, _sandbox_id: &str) -> Result<SandboxStatus> {
            Ok(self.status)
        }
    }

    /// A launcher that implements only [`SandboxLauncher::launch`], so every other call is
    /// the trait default.
    struct Minimal;

    #[async_trait]
    impl SandboxLauncher for Minimal {
        async fn launch(
            &self,
            _config: &SandboxLauncherConfig,
            _timeout_seconds: u64,
            _isolation_key: Option<&str>,
        ) -> Result<LaunchedSandbox> {
            Ok(LaunchedSandbox::new("http://minimal"))
        }
    }

    fn counting(status: SandboxStatus) -> Arc<CountingLauncher> {
        Arc::new(CountingLauncher {
            launches: AtomicU32::new(0),
            status,
        })
    }

    fn config_with_base_url(base_url: &str) -> SandboxLauncherConfig {
        SandboxLauncherConfig {
            base_url: Some(base_url.to_string()),
            ..SandboxLauncherConfig::default()
        }
    }

    #[tokio::test]
    async fn pre_deploy_launcher_returns_the_configured_base_url() {
        let launcher = PreDeploymentLauncher;
        let descriptor = launcher
            .launch(
                &config_with_base_url(" http://10.0.0.1:8080 "),
                30,
                Some("key"),
            )
            .await
            .unwrap();
        assert_eq!(descriptor, LaunchedSandbox::new("http://10.0.0.1:8080"));
        assert_eq!(launcher.launcher_type(), "pre_deploy");
    }

    #[tokio::test]
    async fn pre_deploy_launcher_refuses_the_calls_it_cannot_perform() {
        // The `pre_deploy` launcher owns no handle on its container: suspending, resuming,
        // or destroying it is impossible, so every call is a typed refusal rather than a
        // silent success that would leak the container.
        let launcher = PreDeploymentLauncher;
        for (process, outcome) in [
            ("pause", launcher.pause("any").await),
            ("resume", launcher.resume("any").await),
            ("delete", launcher.delete("any").await),
        ] {
            let error = outcome.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported, "process: {process}");
            assert!(
                error.message().starts_with(&format!(
                    "sandbox launcher unsupported, launcher: pre_deploy, process: {process}, \
                     reason: "
                )),
                "unexpected message for {process}: {}",
                error.message()
            );
        }
        assert_eq!(
            launcher.check_status("any").await.unwrap(),
            SandboxStatus::Unknown,
            "the pre_deploy launcher cannot report a state"
        );
    }

    #[tokio::test]
    async fn defaulted_lifecycle_calls_are_unsupported_and_name_the_launcher() {
        // `Minimal` overrides nothing but `launch`: the defaults must refuse, never succeed.
        let launcher = Minimal;
        assert_eq!(launcher.launcher_type(), "custom");

        let delete = launcher.delete("c-1").await.unwrap_err();
        assert_eq!(delete.kind(), ErrorKind::Unsupported);
        assert_eq!(
            delete.message(),
            "sandbox launcher unsupported, launcher: custom, process: delete, reason: this \
             launcher cannot destroy containers; teardown would leak the sandbox"
        );

        let pause = launcher.pause("c-1").await.unwrap_err();
        assert_eq!(pause.kind(), ErrorKind::Unsupported);
        assert_eq!(
            pause.message(),
            "sandbox launcher unsupported, launcher: custom, process: pause, reason: this \
             launcher cannot suspend containers; the sandbox keeps running"
        );

        let resume = launcher.resume("c-1").await.unwrap_err();
        assert_eq!(resume.kind(), ErrorKind::Unsupported);
        assert_eq!(
            resume.message(),
            "sandbox launcher unsupported, launcher: custom, process: resume, reason: this \
             launcher cannot resume containers; the sandbox stays suspended"
        );

        assert_eq!(
            launcher.check_status("c-1").await.unwrap(),
            SandboxStatus::Unknown,
            "a launcher that cannot inspect its runtime reports unknown, not running"
        );
    }

    #[tokio::test]
    async fn an_explicit_override_still_wins_over_every_default() {
        let launcher = CountingLauncher {
            launches: AtomicU32::new(0),
            status: SandboxStatus::Paused,
        };
        launcher.delete("c-1").await.unwrap();
        launcher.pause("c-1").await.unwrap();
        launcher.resume("c-1").await.unwrap();
        assert_eq!(
            launcher.check_status("c-1").await.unwrap(),
            SandboxStatus::Paused
        );
        assert_eq!(launcher.launcher_type(), "counting");
    }

    #[tokio::test]
    async fn pre_deploy_launcher_requires_a_base_url() {
        for base_url in [None, Some(String::new()), Some("   ".to_string())] {
            let config = SandboxLauncherConfig {
                base_url,
                ..SandboxLauncherConfig::default()
            };
            let error = PreDeploymentLauncher
                .launch(&config, 30, None)
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

    #[test]
    fn registration_duplicates_and_lookups() {
        let registry = LauncherRegistry::new();
        assert_eq!(registry.launcher_types(), Vec::<String>::new());
        assert!(!registry.has("custom"));

        registry
            .register("custom", counting(SandboxStatus::Running))
            .unwrap();
        assert!(registry.has("custom"));
        assert!(registry.get("custom").is_some());
        assert_eq!(registry.launcher_types(), vec!["custom".to_string()]);

        let error = registry
            .register("custom", counting(SandboxStatus::Running))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error
            .message()
            .contains("launcher type 'custom' is already registered"));
        assert!(error
            .message()
            .starts_with("sys operation registry error, process: register_launcher"));

        registry.unregister("custom");
        assert!(!registry.has("custom"));
        assert!(registry.get("custom").is_none());
    }

    #[test]
    fn unknown_launcher_type_is_unsupported_and_names_the_registered_types() {
        let registry = LauncherRegistry::new();
        registry
            .register("docker", counting(SandboxStatus::Running))
            .unwrap();
        registry
            .register("podman", counting(SandboxStatus::Running))
            .unwrap();

        let error = registry.create("nope").err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "Unknown launcher_type: nope; registered launcher types: docker, podman"
        );

        let empty = LauncherRegistry::new().create("nope").err().unwrap();
        assert!(empty.message().contains("registered launcher types: none"));
    }

    #[test]
    fn builtin_registration_is_idempotent() {
        let registry = LauncherRegistry::new();
        register_builtin_launchers(&registry).unwrap();
        assert!(registry.has(DEFAULT_LAUNCHER_TYPE));
        register_builtin_launchers(&registry).unwrap();
        assert_eq!(
            registry.launcher_types(),
            vec![DEFAULT_LAUNCHER_TYPE.to_string()]
        );
    }

    #[test]
    fn builtin_registration_keeps_a_host_implementation() {
        let registry = LauncherRegistry::new();
        registry
            .register(DEFAULT_LAUNCHER_TYPE, counting(SandboxStatus::Paused))
            .unwrap();
        register_builtin_launchers(&registry).unwrap();
        let launcher = registry.create(DEFAULT_LAUNCHER_TYPE).unwrap();
        let status = futures::executor::block_on(launcher.check_status("x")).unwrap();
        assert_eq!(status, SandboxStatus::Paused, "the host launcher must win");
    }

    #[test]
    fn global_registry_is_stable_and_preloaded() {
        assert!(std::ptr::eq(global(), global()));
        assert!(global().has(DEFAULT_LAUNCHER_TYPE));
    }

    #[test]
    fn error_helpers_carry_python_codes() {
        assert_eq!(
            launcher_error("launch", "boom").message(),
            "sandbox launcher error, process: launch, reason: boom"
        );
        assert_eq!(
            provider_error("create", "boom").message(),
            "sandbox provider error, process: create, reason: boom"
        );
        assert_eq!(
            isolation_key_error("blank").message(),
            "sandbox isolation key error, reason: blank"
        );
        assert_eq!(
            gateway_error("fs", "boom").message(),
            "sandbox gateway error, operation: fs, reason: boom"
        );
        for error in [
            launcher_error("launch", "boom"),
            provider_error("create", "boom"),
            isolation_key_error("blank"),
            gateway_error("fs", "boom"),
        ] {
            assert_eq!(error.kind(), ErrorKind::SysOp);
        }
    }

    #[test]
    fn launcher_failures_become_199008_values_but_unsupported_stays_an_error() {
        let failure: Result<u32> = Err(launcher_error("launch", "boom"));
        let value = launcher_op_result("ensure_endpoint", failure).unwrap();
        assert_eq!(value.code, 199_008);
        assert_eq!(
            value.message,
            "sandbox launcher error, process: ensure_endpoint, reason: sandbox launcher error, \
             process: launch, reason: boom"
        );

        let success: Result<u32> = Ok(7);
        assert_eq!(launcher_op_result("launch", success).unwrap().data, Some(7));

        let unsupported: Result<u32> = Err(unknown_launcher_error("nope", &[]));
        let error = launcher_op_result("launch", unsupported).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}
