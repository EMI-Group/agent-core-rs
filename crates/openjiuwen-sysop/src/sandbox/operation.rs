//! Sandbox operation dispatch: turns a sandbox card into the three operation backends.
//!
//! The dispatch resolves the card's isolation-key template, asks the
//! [`SandboxProviderRegistry`] for a provider for the configured `sandbox_type`, and
//! returns backends that forward every call to that provider. When no provider is
//! registered the whole set is rejected with `ErrorKind::Unsupported` (Python raises
//! `NotImplementedError` only on the first call; failing at construction is stricter and
//! reported as a documented deviation).
//!
//! # Dispatch design
//!
//! [`sandbox_operations`] and [`sandbox_operations_for_session`] only select the
//! process-global [`crate::sandbox::provider::global`] registry; the entire decision logic
//! lives in a private core that takes the registry as an argument, so the behavior can be
//! exercised against a local [`SandboxProviderRegistry`] without polluting process state:
//!
//! 1. `card.validate()` — a sandbox card without `gateway_config` or `launcher_config`
//!    fails with the Python `sys operation card param error` (code 199002).
//! 2. The isolation-key template is derived from the gateway configuration
//!    (`{scope}_{launcher_type}_{sandbox_type}_{prefix}{identity}`) and every
//!    `{session_id}` placeholder is replaced with the caller's session id (or
//!    [`crate::sandbox::config::DEFAULT_SESSION_ID`] when none is given).
//! 3. The **static** path uses `launcher_config.base_url` as the endpoint: that is a
//!    pre-deployed sandbox and no lifecycle bookkeeping is involved.
//! 4. The **managed** path ([`sandbox_operations_with_registries`],
//!    [`sandbox_operations_for_session_async`], [`resolve_sandbox_endpoint`]) asks the
//!    [`SandboxLifecycle`] for an endpoint, which reuses a running container, resumes a
//!    paused one, or launches a new one — see [`crate::sandbox::lifecycle`].
//! 5. [`register_builtin_providers`] installs the provider shipped with this crate (the
//!    HTTP gateway behind the `sandbox` feature); extension crates register their own.
//! 6. `create_fs` / `create_shell` / `create_code` build the provider backends for the
//!    resolved endpoint, and each is wrapped in the forwarding type below so the caller
//!    sees one uniform [`OperationSet`] carrying the endpoint and isolation key.
//!
//! A missing provider is reported when the backend set is built, not on the first call:
//! Python's sandbox backends raise `NotImplementedError` lazily, which this rewrite
//! deliberately makes eager so a misconfigured card fails before any I/O.
//!
//! # Forwarding
//!
//! The wrappers are wire-format agnostic: they never serialize, re-map, or inspect the
//! parameters or the provider's [`OpResult`], so a provider may speak HTTP, a unix socket,
//! or call an in-process fake. Failures stay values (a non-zero code with the Python
//! message text), never panics.
//!
//! ```
//! use openjiuwen_sysop::card::SysOperationCard;
//! use openjiuwen_sysop::mode::OperationMode;
//! use openjiuwen_sysop::sandbox::operation::sandbox_operations;
//!
//! let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Sandbox);
//! // A sandbox card without a gateway configuration is rejected before any I/O.
//! assert!(sandbox_operations(&card).is_err());
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};

use crate::card::SysOperationCard;
use crate::code::types::{CodeChunk, CodeExecParams, CodeOutput};
use crate::constants::DEFAULT_LAUNCHER_TYPE;
use crate::fs::types::{
    DownloadFileData, DownloadFileParams, ListDirsParams, ListFilesData, ListFilesParams,
    ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData, SearchFilesParams,
    TransferChunkData, UploadFileData, UploadFileParams, WriteFileData, WriteFileParams,
};
use crate::operation::{CodeOperation, FsOperation, OpStream, OperationSet, ShellOperation};
use crate::result::OpResult;
use crate::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig, SandboxLauncherConfig};
use crate::sandbox::lifecycle::SandboxLifecycle;
use crate::sandbox::provider::SandboxProviderRegistry;
use crate::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput,
};

/// Builds the sandbox backends for `card`, using no session id for the isolation key.
///
/// # Errors
///
/// Returns `ErrorKind::SysOp` when the card lacks a usable gateway configuration and
/// `ErrorKind::Unsupported` when no provider is registered for the configured
/// `sandbox_type`.
pub fn sandbox_operations(card: &SysOperationCard) -> Result<OperationSet> {
    sandbox_operations_for_session(card, None)
}

/// Builds the sandbox backends for `card`, resolving `{session_id}` in the isolation key.
///
/// This is the *static* path: the endpoint is `launcher_config.base_url`, exactly how a
/// pre-deployed sandbox is addressed. A launcher that creates containers is resolved by
/// [`sandbox_operations_for_session_async`] (or [`sandbox_operations_with_registries`]),
/// which drives the async [`SandboxLifecycle`]; this function reports Python's
/// card-parameter error (199002) for such a card instead of silently targeting an empty
/// endpoint.
///
/// Uses the process-global provider registry; see [`sandbox_operations`] for the errors.
///
/// # Errors
///
/// See [`sandbox_operations`].
pub fn sandbox_operations_for_session(
    card: &SysOperationCard,
    session_id: Option<&str>,
) -> Result<OperationSet> {
    operations_for_session_in(card, session_id, crate::sandbox::provider::global())
}

/// Builds the sandbox backends for `card`, resolving a managed container through `lifecycle`.
///
/// The endpoint is resolved exactly once, through [`resolve_sandbox_endpoint`], and the
/// providers are then created from it, so a launched container's `base_url`/`sandbox_id`
/// reach the backends (and [`SandboxFsOperation::endpoint`] reports them).
///
/// # Errors
///
/// Returns the [`resolve_sandbox_endpoint`] failures (199002, 199008, 199010,
/// `Unsupported`) plus the provider-registration and provider-construction failures of
/// [`sandbox_operations`].
pub async fn sandbox_operations_with_registries(
    card: &SysOperationCard,
    session_id: Option<&str>,
    providers: &SandboxProviderRegistry,
    lifecycle: &SandboxLifecycle,
    now_epoch_secs: u64,
) -> Result<OperationSet> {
    let endpoint = resolve_sandbox_endpoint(card, session_id, lifecycle, now_epoch_secs).await?;
    operations_for_endpoint(card, endpoint, providers)
}

/// [`sandbox_operations_for_session`] over the process-global provider registry, launcher
/// registry, and lifecycle.
///
/// This is the asynchronous entry point for a host that manages containers; it reads the
/// wall clock, so pass an explicit `now` through [`sandbox_operations_with_registries`] when
/// idle-stop must be deterministic.
///
/// # Errors
///
/// See [`sandbox_operations_with_registries`].
pub async fn sandbox_operations_for_session_async(
    card: &SysOperationCard,
    session_id: Option<&str>,
) -> Result<OperationSet> {
    sandbox_operations_with_registries(
        card,
        session_id,
        crate::sandbox::provider::global(),
        crate::sandbox::lifecycle::global(),
        crate::sandbox::lifecycle::now_epoch_secs(),
    )
    .await
}

/// Resolves the endpoint operations on `card` must target for `session_id`.
///
/// Three cases, in order:
///
/// 1. A `base_url` is configured → that URL, verbatim. This is the pre-deploy case and is
///    today's behavior: no launch, no bookkeeping.
/// 2. `launcher_type` is `pre_deploy` without a `base_url` → the empty base URL, again
///    exactly as before (the provider then decides whether it can serve it).
/// 3. Any other launcher type → [`SandboxLifecycle::ensure_endpoint`] with the resolved
///    isolation key, which reuses, resumes, or launches a container.
///
/// # Errors
///
/// Returns `ErrorKind::SysOp` for an invalid card (199002) or a failed launch/resume
/// (199008), Python's isolation-key error (199010) for an unusable key, and
/// `ErrorKind::Unsupported` when the launcher type in case 3 is not registered.
pub async fn resolve_sandbox_endpoint(
    card: &SysOperationCard,
    session_id: Option<&str>,
    lifecycle: &SandboxLifecycle,
    now_epoch_secs: u64,
) -> Result<SandboxEndpoint> {
    card.validate()?;
    let gateway = card.gateway_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("gateway_config is required when mode is 'sandbox'")
    })?;
    let launcher = gateway.launcher_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("launcher_config is required when mode is 'sandbox'")
    })?;
    let template = crate::sandbox::config::isolation_key_template_from_config(gateway)?;
    let isolation_key = crate::sandbox::config::resolve_isolation_key(&template, session_id);
    if is_static_launcher(launcher) {
        return Ok(static_endpoint(launcher, isolation_key));
    }
    let launchers = lifecycle.launchers();
    if !launchers.has(&launcher.launcher_type) {
        return Err(crate::sandbox::launcher::unknown_launcher_error(
            &launcher.launcher_type,
            &launchers.launcher_types(),
        ));
    }
    lifecycle
        .ensure_endpoint(gateway, &isolation_key, now_epoch_secs)
        .await
}

/// Builds the backends of `card` for an already resolved `endpoint`.
///
/// # Errors
///
/// Returns `ErrorKind::SysOp` (199002) when the card is invalid and propagates the
/// provider-registration and provider-construction failures.
pub(crate) fn operations_for_endpoint(
    card: &SysOperationCard,
    endpoint: SandboxEndpoint,
    registry: &SandboxProviderRegistry,
) -> Result<OperationSet> {
    card.validate()?;
    let gateway = card.gateway_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("gateway_config is required when mode is 'sandbox'")
    })?;
    let launcher = gateway.launcher_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("launcher_config is required when mode is 'sandbox'")
    })?;
    let (fs, shell, code) = build_backends(gateway, launcher, endpoint, registry)?;
    Ok(OperationSet {
        fs: Some(Arc::new(fs)),
        shell: Some(Arc::new(shell)),
        code: Some(Arc::new(code)),
    })
}

/// Whether `launcher` addresses a sandbox that already exists.
///
/// A configured `base_url` always wins, so behavior for a pre-deployed sandbox is unchanged
/// whether or not this crate knows its launcher type. `pre_deploy` is static without a
/// `base_url` too, because that launcher never creates a container.
fn is_static_launcher(launcher: &SandboxLauncherConfig) -> bool {
    let base_url = launcher.base_url.as_deref().unwrap_or_default().trim();
    !base_url.is_empty() || launcher.launcher_type == DEFAULT_LAUNCHER_TYPE
}

/// The static endpoint of a pre-deployed sandbox (today's dispatch endpoint).
///
/// A `pre_deploy` launcher without a `base_url` yields the empty base URL, which is kept for
/// compatibility (the provider decides whether it can serve such an endpoint; the built-in
/// HTTP gateway rejects it with a typed error). It is logged, because an empty endpoint is
/// otherwise indistinguishable from a real one until the first call fails.
fn static_endpoint(launcher: &SandboxLauncherConfig, isolation_key: String) -> SandboxEndpoint {
    let base_url = launcher.base_url.clone().unwrap_or_default();
    if base_url.trim().is_empty() {
        tracing::warn!(
            launcher_type = %launcher.launcher_type,
            sandbox_type = %launcher.sandbox_type,
            "the static sandbox endpoint has no base_url; the provider must reject it"
        );
    }
    SandboxEndpoint {
        base_url,
        sandbox_id: None,
        isolation_key: Some(isolation_key),
    }
}

/// The card-parameter error that points a managed-launcher card at the async seam.
fn managed_launcher_error(launcher_type: &str) -> Error {
    crate::card::card_param_error(&format!(
        "launcher_type '{launcher_type}' creates sandbox containers; resolve its endpoint with \
         sandbox_operations_with_registries (or SysOperation::sandbox_with_lifecycle) so the \
         launch runs on the async lifecycle"
    ))
}

/// Builds the sandbox backends for `card` against an explicit provider registry.
///
/// This is the single dispatch implementation of the static path; the public entry points
/// only choose the registry. A missing provider for the configured `sandbox_type` is
/// reported here, so a misconfigured card fails at construction instead of on its first
/// operation.
///
/// # Errors
///
/// Returns `ErrorKind::SysOp` (Python code 199002) when the card or its gateway
/// configuration is invalid or the launcher creates containers, `ErrorKind::SysOp` (199010)
/// when a custom-scope isolation key lacks a `custom_id`, and `ErrorKind::Unsupported` when
/// no provider is registered for the configured `sandbox_type`; provider registration
/// errors are propagated.
fn operations_for_session_in(
    card: &SysOperationCard,
    session_id: Option<&str>,
    registry: &SandboxProviderRegistry,
) -> Result<OperationSet> {
    let (fs, shell, code) = dispatch_in(card, session_id, registry)?;
    Ok(OperationSet {
        fs: Some(Arc::new(fs)),
        shell: Some(Arc::new(shell)),
        code: Some(Arc::new(code)),
    })
}

/// Creates the concrete forwarding backends for `card` from `registry`.
///
/// Returns the wrapper types (rather than an erased [`OperationSet`]) so the resolved
/// endpoint and isolation key stay inspectable.
///
/// # Errors
///
/// See [`operations_for_session_in`].
fn dispatch_in(
    card: &SysOperationCard,
    session_id: Option<&str>,
    registry: &SandboxProviderRegistry,
) -> Result<(
    SandboxFsOperation,
    SandboxShellOperation,
    SandboxCodeOperation,
)> {
    card.validate()?;
    let gateway = card.gateway_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("gateway_config is required when mode is 'sandbox'")
    })?;
    let launcher = gateway.launcher_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("launcher_config is required when mode is 'sandbox'")
    })?;
    let template = crate::sandbox::config::isolation_key_template_from_config(gateway)?;
    let isolation_key = crate::sandbox::config::resolve_isolation_key(&template, session_id);
    if !is_static_launcher(launcher) {
        return Err(managed_launcher_error(&launcher.launcher_type));
    }
    build_backends(
        gateway,
        launcher,
        static_endpoint(launcher, isolation_key),
        registry,
    )
}

/// Installs the crate's built-in provider and creates the three forwarding backends.
///
/// # Errors
///
/// Propagates [`register_builtin_providers`] and the provider-construction failures.
fn build_backends(
    gateway: &SandboxGatewayConfig,
    launcher: &SandboxLauncherConfig,
    endpoint: SandboxEndpoint,
    registry: &SandboxProviderRegistry,
) -> Result<(
    SandboxFsOperation,
    SandboxShellOperation,
    SandboxCodeOperation,
)> {
    register_builtin_providers(registry, gateway)?;
    let fs = registry.create_fs(&launcher.sandbox_type, &endpoint, gateway)?;
    let shell = registry.create_shell(&launcher.sandbox_type, &endpoint, gateway)?;
    let code = registry.create_code(&launcher.sandbox_type, &endpoint, gateway)?;
    Ok((
        SandboxFsOperation::new(endpoint.clone(), gateway, fs),
        SandboxShellOperation::new(endpoint.clone(), gateway, shell),
        SandboxCodeOperation::new(endpoint, gateway, code),
    ))
}

/// Installs the provider shipped with this crate for the configured sandbox type.
///
/// Without the `sandbox` feature nothing is installed, so a sandbox card is reported as
/// unsupported. With the feature, the HTTP gateway provider is registered; it
/// validates the configuration and returns `Error::Unsupported` for remote calls that
/// need the external AIO container protocol.
///
/// Registration is idempotent: it is skipped when the gateway has no launcher or no
/// non-blank `base_url`, and `crate::sandbox::http::install` ignores kinds that are
/// already registered, so repeated [`crate::SysOperation::new`] calls succeed.
///
/// # Errors
///
/// Propagates registration failures (except "already registered", which is treated as
/// success).
pub fn register_builtin_providers(
    registry: &SandboxProviderRegistry,
    gateway: &SandboxGatewayConfig,
) -> Result<()> {
    #[cfg(not(feature = "sandbox"))]
    {
        let _ = (registry, gateway);
        Ok(())
    }
    #[cfg(feature = "sandbox")]
    {
        let Some(launcher) = gateway.launcher_config.as_ref() else {
            return Ok(());
        };
        let Some(base_url) = launcher.base_url.as_deref().map(str::trim) else {
            return Ok(());
        };
        if base_url.is_empty() {
            return Ok(());
        }
        let prototype = SandboxEndpoint {
            base_url: base_url.to_string(),
            sandbox_id: None,
            isolation_key: None,
        };
        crate::sandbox::http::install(registry, &launcher.sandbox_type, &prototype, gateway)
    }
}

/// Reports that a sandbox operation is unavailable in this build.
pub fn unsupported_sandbox(operation: &str) -> Error {
    Error::unsupported(format!(
        "sandbox operation '{operation}' is not available: no provider registered for this sandbox type"
    ))
}

/// Forwards filesystem operations to a provider, carrying the resolved endpoint.
///
/// The wrapper keeps the endpoint (base URL, sandbox id, isolation key) so callers and
/// diagnostics can tell which container a backend targets, while every operation is
/// forwarded verbatim to `inner` — no parameter inspection, no result remapping.
pub struct SandboxFsOperation {
    endpoint: SandboxEndpoint,
    supports_conditional_write: bool,
    inner: Arc<dyn FsOperation>,
}

impl SandboxFsOperation {
    /// Wraps `inner`, capturing the endpoint and the card's conditional-write claim.
    ///
    /// `config.supports_conditional_write` is what the card *claims*; the answer of
    /// [`FsOperation::supports_conditional_write`] delegates to the provider and wins.
    #[must_use]
    pub fn new(
        endpoint: SandboxEndpoint,
        config: &SandboxGatewayConfig,
        inner: Arc<dyn FsOperation>,
    ) -> Self {
        Self {
            endpoint,
            supports_conditional_write: config.supports_conditional_write,
            inner,
        }
    }

    /// The endpoint this backend forwards to.
    #[must_use]
    pub const fn endpoint(&self) -> &SandboxEndpoint {
        &self.endpoint
    }

    /// The resolved isolation key of the target container.
    #[must_use]
    pub fn isolation_key(&self) -> Option<&str> {
        self.endpoint.isolation_key.as_deref()
    }

    /// The conditional-write support claimed by the card's gateway configuration.
    ///
    /// This is the configured claim, not the effective answer: use
    /// [`FsOperation::supports_conditional_write`] to ask the provider.
    #[must_use]
    pub const fn configured_conditional_write(&self) -> bool {
        self.supports_conditional_write
    }
}

impl fmt::Debug for SandboxFsOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only the endpoint and the provider marker are shown: the gateway configuration
        // (which may carry auth headers) is deliberately absent.
        f.debug_struct("SandboxFsOperation")
            .field("endpoint", &self.endpoint)
            .field("inner", &"Arc<dyn FsOperation>")
            .field(
                "supports_conditional_write",
                &self.supports_conditional_write,
            )
            .finish()
    }
}

#[async_trait]
impl FsOperation for SandboxFsOperation {
    fn supports_conditional_write(&self) -> bool {
        self.inner.supports_conditional_write()
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        self.inner.read_file(params).await
    }

    fn read_file_stream(self: Arc<Self>, params: ReadFileParams) -> OpStream<ReadChunkData> {
        let inner = Arc::clone(&self.inner);
        inner.read_file_stream(params)
    }

    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData> {
        self.inner.write_file(params).await
    }

    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData> {
        self.inner.upload_file(params).await
    }

    fn upload_file_stream(
        self: Arc<Self>,
        params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        let inner = Arc::clone(&self.inner);
        inner.upload_file_stream(params)
    }

    async fn download_file(&self, params: DownloadFileParams) -> OpResult<DownloadFileData> {
        self.inner.download_file(params).await
    }

    fn download_file_stream(
        self: Arc<Self>,
        params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        let inner = Arc::clone(&self.inner);
        inner.download_file_stream(params)
    }

    async fn list_files(&self, params: ListFilesParams) -> OpResult<ListFilesData> {
        self.inner.list_files(params).await
    }

    async fn list_directories(&self, params: ListDirsParams) -> OpResult<ListFilesData> {
        self.inner.list_directories(params).await
    }

    async fn search_files(&self, params: SearchFilesParams) -> OpResult<SearchFilesData> {
        self.inner.search_files(params).await
    }
}

/// Forwards shell operations to a provider, carrying the resolved endpoint.
pub struct SandboxShellOperation {
    endpoint: SandboxEndpoint,
    inner: Arc<dyn ShellOperation>,
}

impl SandboxShellOperation {
    /// Wraps `inner`, capturing the endpoint it forwards to.
    #[must_use]
    pub fn new(
        endpoint: SandboxEndpoint,
        _config: &SandboxGatewayConfig,
        inner: Arc<dyn ShellOperation>,
    ) -> Self {
        Self { endpoint, inner }
    }

    /// The endpoint this backend forwards to.
    #[must_use]
    pub const fn endpoint(&self) -> &SandboxEndpoint {
        &self.endpoint
    }

    /// The resolved isolation key of the target container.
    #[must_use]
    pub fn isolation_key(&self) -> Option<&str> {
        self.endpoint.isolation_key.as_deref()
    }
}

impl fmt::Debug for SandboxShellOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The gateway configuration is never printed: it may carry auth data.
        f.debug_struct("SandboxShellOperation")
            .field("endpoint", &self.endpoint)
            .field("inner", &"Arc<dyn ShellOperation>")
            .finish()
    }
}

#[async_trait]
impl ShellOperation for SandboxShellOperation {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        self.inner.execute_cmd(params).await
    }

    fn execute_cmd_stream(self: Arc<Self>, params: ShellExecParams) -> OpStream<ShellChunk> {
        let inner = Arc::clone(&self.inner);
        inner.execute_cmd_stream(params)
    }

    async fn execute_cmd_background(
        &self,
        params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput> {
        self.inner.execute_cmd_background(params).await
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        self.inner.execute_argv(params).await
    }
}

/// Forwards code execution to a provider, carrying the resolved endpoint.
pub struct SandboxCodeOperation {
    endpoint: SandboxEndpoint,
    inner: Arc<dyn CodeOperation>,
}

impl SandboxCodeOperation {
    /// Wraps `inner`, capturing the endpoint it forwards to.
    #[must_use]
    pub fn new(
        endpoint: SandboxEndpoint,
        _config: &SandboxGatewayConfig,
        inner: Arc<dyn CodeOperation>,
    ) -> Self {
        Self { endpoint, inner }
    }

    /// The endpoint this backend forwards to.
    #[must_use]
    pub const fn endpoint(&self) -> &SandboxEndpoint {
        &self.endpoint
    }

    /// The resolved isolation key of the target container.
    #[must_use]
    pub fn isolation_key(&self) -> Option<&str> {
        self.endpoint.isolation_key.as_deref()
    }
}

impl fmt::Debug for SandboxCodeOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The gateway configuration is never printed: it may carry auth data.
        f.debug_struct("SandboxCodeOperation")
            .field("endpoint", &self.endpoint)
            .field("inner", &"Arc<dyn CodeOperation>")
            .finish()
    }
}

#[async_trait]
impl CodeOperation for SandboxCodeOperation {
    async fn execute_code(&self, params: CodeExecParams) -> OpResult<CodeOutput> {
        self.inner.execute_code(params).await
    }

    fn execute_code_stream(self: Arc<Self>, params: CodeExecParams) -> OpStream<CodeChunk> {
        let inner = Arc::clone(&self.inner);
        inner.execute_code_stream(params)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::mode::OperationMode;
    use crate::result::{code_failure, fs_failure, shell_failure};
    use crate::sandbox::config::SandboxLauncherConfig;
    use crate::sandbox::provider::{CodeProviderFactory, FsProviderFactory, ShellProviderFactory};

    /// A filesystem provider whose results are recognizable and never touch the disk.
    struct FakeFs;

    impl FakeFs {
        fn read_data() -> ReadFileData {
            ReadFileData {
                path: "fake.txt".to_string(),
                total_bytes: 4,
                ..ReadFileData::default()
            }
        }

        fn chunk() -> ReadChunkData {
            ReadChunkData {
                path: "fake.txt".to_string(),
                chunk_index: 7,
                is_last_chunk: true,
                ..ReadChunkData::default()
            }
        }
    }

    #[async_trait]
    impl FsOperation for FakeFs {
        fn supports_conditional_write(&self) -> bool {
            true
        }

        async fn read_file(&self, _params: ReadFileParams) -> OpResult<ReadFileData> {
            OpResult::success(Self::read_data())
        }

        fn read_file_stream(self: Arc<Self>, _params: ReadFileParams) -> OpStream<ReadChunkData> {
            Box::pin(futures::stream::once(async {
                OpResult::success(Self::chunk())
            }))
        }

        async fn write_file(&self, _params: WriteFileParams) -> OpResult<WriteFileData> {
            fs_failure("write_file", "fake fs")
        }

        async fn upload_file(&self, _params: UploadFileParams) -> OpResult<UploadFileData> {
            fs_failure("upload_file", "fake fs")
        }

        fn upload_file_stream(
            self: Arc<Self>,
            _params: UploadFileParams,
        ) -> OpStream<TransferChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn download_file(&self, _params: DownloadFileParams) -> OpResult<DownloadFileData> {
            fs_failure("download_file", "fake fs")
        }

        fn download_file_stream(
            self: Arc<Self>,
            _params: DownloadFileParams,
        ) -> OpStream<TransferChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn list_files(&self, _params: ListFilesParams) -> OpResult<ListFilesData> {
            fs_failure("list_files", "fake fs")
        }

        async fn list_directories(&self, _params: ListDirsParams) -> OpResult<ListFilesData> {
            fs_failure("list_directories", "fake fs")
        }

        async fn search_files(&self, _params: SearchFilesParams) -> OpResult<SearchFilesData> {
            fs_failure("search_files", "fake fs")
        }
    }

    /// A shell provider that always fails with a recognizable message.
    struct FakeShell;

    #[async_trait]
    impl ShellOperation for FakeShell {
        async fn execute_cmd(&self, _params: ShellExecParams) -> OpResult<ShellOutput> {
            shell_failure("execute_cmd", "fake shell")
        }

        fn execute_cmd_stream(self: Arc<Self>, _params: ShellExecParams) -> OpStream<ShellChunk> {
            Box::pin(futures::stream::empty())
        }

        async fn execute_cmd_background(
            &self,
            _params: ShellExecParams,
        ) -> OpResult<ShellBackgroundOutput> {
            shell_failure("execute_cmd_background", "fake shell")
        }

        async fn execute_argv(&self, _params: ShellArgvParams) -> OpResult<ShellOutput> {
            shell_failure("execute_argv", "fake shell")
        }
    }

    /// A code provider that always fails with a recognizable message.
    struct FakeCode;

    #[async_trait]
    impl CodeOperation for FakeCode {
        async fn execute_code(&self, _params: CodeExecParams) -> OpResult<CodeOutput> {
            code_failure("execute_code", "fake code")
        }

        fn execute_code_stream(self: Arc<Self>, _params: CodeExecParams) -> OpStream<CodeChunk> {
            Box::pin(futures::stream::empty())
        }
    }

    /// Registers the in-process fakes for `sandbox_type` on a local registry.
    fn register_fakes(registry: &SandboxProviderRegistry, sandbox_type: &str) {
        let fs: FsProviderFactory = Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Ok(Arc::new(FakeFs) as Arc<dyn FsOperation>)
            },
        );
        let shell: ShellProviderFactory = Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Ok(Arc::new(FakeShell) as Arc<dyn ShellOperation>)
            },
        );
        let code: CodeProviderFactory = Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Ok(Arc::new(FakeCode) as Arc<dyn CodeOperation>)
            },
        );
        registry.register_fs(sandbox_type, fs).unwrap();
        registry.register_shell(sandbox_type, shell).unwrap();
        registry.register_code(sandbox_type, code).unwrap();
    }

    /// A sandbox card for `sandbox_type`, optionally with a launcher base URL.
    fn sandbox_card(sandbox_type: &str, base_url: Option<&str>) -> SysOperationCard {
        SysOperationCard::new("sys_op")
            .with_mode(OperationMode::Sandbox)
            .with_gateway_config(SandboxGatewayConfig {
                launcher_config: Some(SandboxLauncherConfig {
                    launcher_type: "pre_deploy".to_string(),
                    sandbox_type: sandbox_type.to_string(),
                    base_url: base_url.map(str::to_string),
                    ..SandboxLauncherConfig::default()
                }),
                ..SandboxGatewayConfig::default()
            })
    }

    fn endpoint_with_key(isolation_key: Option<&str>) -> SandboxEndpoint {
        SandboxEndpoint {
            base_url: "http://127.0.0.1:1".to_string(),
            sandbox_id: Some("sandbox-1".to_string()),
            isolation_key: isolation_key.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn forwarding_delegates_to_provider() {
        let config = SandboxGatewayConfig::default();
        assert!(!config.supports_conditional_write);
        let fs = Arc::new(SandboxFsOperation::new(
            endpoint_with_key(Some("session_pre_deploy_mock_s1")),
            &config,
            Arc::new(FakeFs),
        ));

        assert_eq!(fs.endpoint().base_url, "http://127.0.0.1:1");
        assert_eq!(fs.isolation_key(), Some("session_pre_deploy_mock_s1"));

        let read = fs.read_file(ReadFileParams::text("fake.txt")).await;
        assert_eq!(read, OpResult::success(FakeFs::read_data()));

        // The provider's answer wins; the card's claim is reported separately.
        assert!(fs.supports_conditional_write());
        assert!(!fs.configured_conditional_write());

        let chunks: Vec<OpResult<ReadChunkData>> = Arc::clone(&fs)
            .read_file_stream(ReadFileParams::text("fake.txt"))
            .collect()
            .await;
        assert_eq!(chunks, vec![OpResult::success(FakeFs::chunk())]);

        // Non-streamed forwarding of an unimplemented method reaches the provider too.
        let write = fs.write_file(WriteFileParams::raw("fake.txt", "x")).await;
        assert_eq!(write.code, 199_003);
        assert!(write.message.contains("fake fs"));

        // Debug never shows the gateway auth configuration.
        let debug = format!("{fs:?}");
        assert!(debug.contains("SandboxFsOperation"));
        assert!(debug.contains("session_pre_deploy_mock_s1"));
        assert!(!debug.contains("auth_headers"));
    }

    #[tokio::test]
    async fn dispatch_wraps_provider_and_resolves_isolation_key() {
        let registry = SandboxProviderRegistry::new();
        register_fakes(&registry, "mock");
        let card = sandbox_card("mock", None);

        let (fs, _shell, _code) = dispatch_in(&card, Some("s1"), &registry).unwrap();
        assert_eq!(fs.isolation_key(), Some("session_pre_deploy_mock_s1"));
        assert_eq!(
            fs.read_file(ReadFileParams::text("fake.txt")).await,
            OpResult::success(FakeFs::read_data())
        );

        let (fs_default, _, _) = dispatch_in(&card, None, &registry).unwrap();
        assert_eq!(
            fs_default.isolation_key(),
            Some("session_pre_deploy_mock_default_session")
        );

        // The erased set carries all three backends and works end to end.
        let operations = operations_for_session_in(&card, Some("s1"), &registry).unwrap();
        let operations_fs = operations.fs.unwrap();
        assert!(operations.shell.is_some());
        assert!(operations.code.is_some());
        assert_eq!(
            operations_fs
                .read_file(ReadFileParams::text("fake.txt"))
                .await,
            OpResult::success(FakeFs::read_data())
        );
        let shell = operations.shell.unwrap();
        let output = shell.execute_cmd(ShellExecParams::new("echo hi")).await;
        assert_eq!(output.code, 199_004);
        assert!(output.message.contains("fake shell"));
    }

    #[tokio::test]
    async fn missing_provider_is_unsupported() {
        let registry = SandboxProviderRegistry::new();
        let card = sandbox_card("nope", None);
        let error = operations_for_session_in(&card, None, &registry).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().contains("does not support operation 'fs'"));
    }

    #[tokio::test]
    async fn missing_gateway_config_is_card_param_error() {
        let registry = SandboxProviderRegistry::new();
        let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Sandbox);
        let error = operations_for_session_in(&card, None, &registry).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error.message().contains("sys operation card param error"));
        assert!(error.message().contains("gateway_config is required"));

        let card = SysOperationCard::new("sys_op")
            .with_mode(OperationMode::Sandbox)
            .with_gateway_config(SandboxGatewayConfig::default());
        let error = operations_for_session_in(&card, None, &registry).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error.message().contains("sys operation card param error"));
        assert!(error.message().contains("launcher_config is required"));
    }

    #[tokio::test]
    async fn register_builtin_providers_is_inert_without_base_url() {
        let registry = SandboxProviderRegistry::new();
        let gateway = SandboxGatewayConfig {
            launcher_config: Some(SandboxLauncherConfig::default()),
            ..SandboxGatewayConfig::default()
        };
        assert_eq!(gateway.launcher_config.as_ref().unwrap().base_url, None);
        assert!(register_builtin_providers(&registry, &gateway).is_ok());
        assert_eq!(registry.sandbox_types(), Vec::<String>::new());

        // A gateway without a launcher is inert as well.
        let gateway = SandboxGatewayConfig::default();
        assert!(register_builtin_providers(&registry, &gateway).is_ok());
        assert_eq!(registry.sandbox_types(), Vec::<String>::new());
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn register_builtin_providers_installs_http_providers() {
        let registry = SandboxProviderRegistry::new();
        let gateway = SandboxGatewayConfig {
            launcher_config: Some(SandboxLauncherConfig {
                launcher_type: "pre_deploy".to_string(),
                sandbox_type: "aio_http_test".to_string(),
                base_url: Some("http://127.0.0.1:1".to_string()),
                ..SandboxLauncherConfig::default()
            }),
            ..SandboxGatewayConfig::default()
        };

        assert!(register_builtin_providers(&registry, &gateway).is_ok());
        assert!(registry.has_fs("aio_http_test"));
        assert!(registry.has_shell("aio_http_test"));
        assert!(registry.has_code("aio_http_test"));

        // Repeated calls are idempotent.
        assert!(register_builtin_providers(&registry, &gateway).is_ok());

        let endpoint = endpoint_with_key(Some("session_pre_deploy_aio_http_test_s1"));
        let fs = registry
            .create_fs("aio_http_test", &endpoint, &gateway)
            .unwrap();
        let result = fs.read_file(ReadFileParams::text("x")).await;
        assert_eq!(result.code, 199_012);
        assert!(result.message.contains("sandbox gateway error"));
    }
}
