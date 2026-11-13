//! Remote sandbox providers and the registration facade for them.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/`.
//! In the reference implementation, importing `providers` registers the `aio`, `jiuwenbox`, and `yuanrong` sandbox providers into `SandboxRegistry` at import time (`@SandboxRegistry.provider("<type>", "fs" | "shell" | "code")`).
//!
//! # Providers
//!
//! [`remote`] implements those three sandbox types as real HTTP transports:
//!
//! - `aio` drives an AIO container through the kernel gateway envelope (`GatewayInvokeRequest` / `GatewayResponse`).
//! - `jiuwenbox` drives the `JiuwenBox` container through its native REST API under `/api/v1/...`.
//! - `yuanrong` drives the `YuanRong` ADX API under `/api/environment/v1/...` plus `/direct/{id}/invoke`.
//!
//! [`REMOTE_SANDBOX_TYPES`] lists them.
//! A host installs all three, in order, with [`register_remote_providers`], or a single type with [`register_remote`]; [`remote_provider_set`] returns a factory set without registering it.
//!
//! # Registration
//!
//! Registration is the kernel seam: [`register`] installs any [`SandboxProviderSet`] under one sandbox type, keyed by `(sandbox_type, operation)`.
//! [`SandboxProviderSet::deferred`] remains available for a host that wants an explicit `Unsupported` set, for example to reserve a sandbox type before its transport exists.
//! [`is_deferred`] reports whether a sandbox type has no remote provider.
//!
//! The kernel resolves the providers of a sandbox-mode `SysOperation` only from its process-global registry (`openjiuwen_sysop::sandbox::provider::global()`), while [`register_remote_providers`] takes any registry, so a host that drives `SysOperation` registers on `global()`.
//! [`registered_types`] reports what a given registry holds.
//!
//! [`register_deferred`] used to install placeholder providers for the Python types; it is now a compatibility no-op that registers nothing and returns `Ok(())`.
//!
//! # Errors
//!
//! An unknown sandbox type is [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported), naming the supported types.
//! A duplicate registration is the kernel's [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp).
//!
//! # Example
//!
//! ```
//! use openjiuwen_core::sys_operation::sandbox::provider::SandboxProviderRegistry;
//! use openjiuwen_extensions::sys_operation::{
//!     SandboxProviderSet, is_deferred, register, remote_provider_set,
//! };
//!
//! let registry = SandboxProviderRegistry::new();
//! register(&registry, "custom", SandboxProviderSet::deferred("custom")).unwrap();
//! assert!(registry.has_fs("custom"));
//! assert!(is_deferred("custom"));
//! assert!(registry
//!     .create_fs("custom", &Default::default(), &Default::default())
//!     .is_err());
//!
//! // `aio`, `jiuwenbox`, and `yuanrong` are real providers now: registration needs
//! // no base URL and contacts nothing.
//! assert!(remote_provider_set("aio").is_ok());
//! assert!(!is_deferred("aio"));
//! ```
#![allow(clippy::module_name_repetitions)] // `FsProviderFactory`/`CodeProviderFactory` are the contract's names.
pub mod remote;
use openjiuwen_common::error::{Error, ErrorKind, Result};
pub use openjiuwen_core::sys_operation::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};
use openjiuwen_core::sys_operation::sandbox::provider::{
    CodeProviderFactory, FsProviderFactory, SandboxProviderRegistry, ShellProviderFactory,
};
pub use remote::{SandboxSelector, TeardownFailure, TeardownReport};
use std::sync::Arc;
/// The sandbox types with a real HTTP transport in [`remote`].
///
/// Python's `extensions/sys_operation/sandbox/providers/` package registers exactly these three names, so configuration written for Python keeps working.
pub const REMOTE_SANDBOX_TYPES: [&str; 3] = ["aio", "jiuwenbox", "yuanrong"];
/// Sandbox types Python registers but this build cannot provide: none.
///
/// Every type Python ships has a transport in [`remote`]; see [`REMOTE_SANDBOX_TYPES`].
/// The constant stays so call sites that iterate the unsupported set keep compiling.
pub const DEFERRED_SANDBOX_TYPES: [&str; 0] = [];
/// The factories Python registers per sandbox type (any subset may be present).
///
/// A set is the unit of registration: [`register`] installs each present factory under
/// one sandbox type, mirroring the three `@SandboxRegistry.provider(<type>, <operation>)`
/// decorators of the reference implementation.
#[derive(Clone, Default)]
pub struct SandboxProviderSet {
    /// Filesystem factory (`@SandboxRegistry.provider(type, "fs")`).
    pub fs: Option<FsProviderFactory>,
    /// Shell factory (`@SandboxRegistry.provider(type, "shell")`).
    pub shell: Option<ShellProviderFactory>,
    /// Code factory (`@SandboxRegistry.provider(type, "code")`).
    pub code: Option<CodeProviderFactory>,
}
impl std::fmt::Debug for SandboxProviderSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxProviderSet")
            .field("fs", &self.fs.is_some())
            .field("shell", &self.shell.is_some())
            .field("code", &self.code.is_some())
            .finish()
    }
}
impl SandboxProviderSet {
    /// An empty set: nothing is registered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Whether the set has no factory at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fs.is_none() && self.shell.is_none() && self.code.is_none()
    }
    /// Adds the filesystem factory.
    #[must_use]
    pub fn with_fs(mut self, factory: FsProviderFactory) -> Self {
        self.fs = Some(factory);
        self
    }
    /// Adds the shell factory.
    #[must_use]
    pub fn with_shell(mut self, factory: ShellProviderFactory) -> Self {
        self.shell = Some(factory);
        self
    }
    /// Adds the code factory.
    #[must_use]
    pub fn with_code(mut self, factory: CodeProviderFactory) -> Self {
        self.code = Some(factory);
        self
    }
    /// A complete set whose three factories answer `Unsupported` naming `sandbox_type`.
    ///
    /// This keeps a sandbox type registered (so an existing card still resolves) while
    /// making the missing transport explicit at call time.
    /// The three Python sandbox types no longer need it; use [`remote_provider_set`] for them.
    #[must_use]
    pub fn deferred(sandbox_type: impl Into<String>) -> Self {
        let sandbox_type: Arc<str> = Arc::from(sandbox_type.into());
        let fs_type = Arc::clone(&sandbox_type);
        let shell_type = Arc::clone(&sandbox_type);
        let code_type = sandbox_type;
        Self {
            fs: Some(Arc::new(
                move |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                    Err(deferred_error(&fs_type))
                },
            )),
            shell: Some(Arc::new(
                move |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                    Err(deferred_error(&shell_type))
                },
            )),
            code: Some(Arc::new(
                move |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                    Err(deferred_error(&code_type))
                },
            )),
        }
    }
}
/// A sandbox type without a transport is documented as deferred rather than silently missing.
fn deferred_error(sandbox_type: &str) -> Error {
    Error::unsupported(format!(
        "sandbox provider '{sandbox_type}' is deferred; register a provider with sys_operation::register"
    ))
}
/// The factory set of one remote sandbox type.
///
/// This is the in-memory half of [`register_remote`]: build (or inspect) the set without
/// touching a registry.
///
/// # Errors
///
/// Returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) when
/// `sandbox_type` is not one of [`REMOTE_SANDBOX_TYPES`], with a message naming them.
pub fn remote_provider_set(sandbox_type: &str) -> Result<SandboxProviderSet> {
    remote::provider_set(sandbox_type)
}
/// Registers the remote provider set of one sandbox type.
///
/// Registration is synchronous and offline: the provider contacts its container only when
/// an operation is created through the kernel registry.
///
/// # Errors
///
/// Propagates [`remote_provider_set`]'s `Unsupported` error for an unknown sandbox type and
/// [`register`]'s errors (a duplicate registration is
/// [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp)).
pub fn register_remote(registry: &SandboxProviderRegistry, sandbox_type: &str) -> Result<()> {
    register(registry, sandbox_type, remote_provider_set(sandbox_type)?)
}
/// Registers the remote providers of every type in [`REMOTE_SANDBOX_TYPES`].
///
/// Types are installed in declaration order, `aio`, `jiuwenbox`, `yuanrong`, and
/// registration stops at the first failure.
/// As with [`register`], registration is not transactional: the types installed before the
/// failure stay registered.
///
/// # Errors
///
/// Returns [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp) when a type is
/// already registered in `registry`.
pub fn register_remote_providers(registry: &SandboxProviderRegistry) -> Result<()> {
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        register_remote(registry, sandbox_type)?;
    }
    Ok(())
}
/// Registers every factory present in `providers` under `sandbox_type`.
///
/// Factories are installed in the order filesystem, shell, code; registration stops at
/// the first failure, so a set that fails part-way leaves the earlier factories
/// registered (the kernel offers no rollback).
///
/// # Errors
///
/// Returns [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp) when a
/// factory is already registered for `sandbox_type` (Python raises on duplicates) and
/// propagates any kernel registration error.
pub fn register(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    providers: SandboxProviderSet,
) -> Result<()> {
    let SandboxProviderSet { fs, shell, code } = providers;
    if let Some(factory) = fs {
        registry.register_fs(sandbox_type, factory)?;
    }
    if let Some(factory) = shell {
        registry.register_shell(sandbox_type, factory)?;
    }
    if let Some(factory) = code {
        registry.register_code(sandbox_type, factory)?;
    }
    Ok(())
}
/// Compatibility no-op kept for call sites that used to install the deferred providers.
///
/// The Python types are real providers now, so registration belongs to
/// [`register_remote_providers`] / [`register_remote`].
/// This function registers nothing, logs one `debug` line under
/// `openjiuwen_extensions::sys_operation`, and returns `Ok(())`, so calling it repeatedly
/// (or after [`register_remote_providers`]) never fails.
///
/// The function stays `async` to mirror the Python async setup path that registers
/// providers; it performs no I/O.
///
/// # Errors
///
/// Never returns an error; the `Result` keeps the historical signature.
#[allow(clippy::unused_async)] // Python parity: registration happens in the async setup path.
pub async fn register_deferred(_registry: &SandboxProviderRegistry) -> Result<()> {
    tracing::debug!(
        target: "openjiuwen_extensions::sys_operation",
        "register_deferred is a compatibility no-op; use register_remote_providers"
    );
    Ok(())
}
/// The sandbox types with at least one registered provider, sorted.
///
/// The kernel seam enumerates registered types through its filesystem map, so a set
/// registered with only a shell or code factory is not listed here (its providers are
/// still usable through [`SandboxProviderRegistry::create_shell`] /
/// [`SandboxProviderRegistry::create_code`]).
#[must_use]
pub fn registered_types(registry: &SandboxProviderRegistry) -> Vec<String> {
    registry.sandbox_types()
}
/// Whether the sandbox type is deferred (documented, not implemented) rather than
/// available.
///
/// The three Python sandbox types are remote providers now, so they are not deferred;
/// any other name is.
#[must_use]
pub fn is_deferred(sandbox_type: &str) -> bool {
    !REMOTE_SANDBOX_TYPES.contains(&sandbox_type)
}
/// Registers a provider set bound to a shared [`SandboxGatewayConfig`].
///
/// Hosts that implement the gateway transport elsewhere build their factories once and
/// hand them over with the gateway configuration; each registered factory is called with
/// that shared configuration regardless of the `SandboxGatewayConfig` a caller passes to
/// `create_fs`/`create_shell`/`create_code`. Absent factories stay unregistered.
///
/// # Errors
///
/// Propagates [`register`]'s errors (duplicate registration is
/// [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp)).
pub fn register_gateway_provider(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    gateway: SandboxGatewayConfig,
    providers: SandboxProviderSet,
) -> Result<()> {
    let gateway = Arc::new(gateway);
    let SandboxProviderSet { fs, shell, code } = providers;
    let fs = fs.map(|factory| {
        let gateway = Arc::clone(&gateway);
        Arc::new(
            move |endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                factory(endpoint, &gateway)
            },
        ) as FsProviderFactory
    });
    let shell = shell.map(|factory| {
        let gateway = Arc::clone(&gateway);
        Arc::new(
            move |endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                factory(endpoint, &gateway)
            },
        ) as ShellProviderFactory
    });
    let code = code.map(|factory| {
        let gateway = Arc::clone(&gateway);
        Arc::new(
            move |endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                factory(endpoint, &gateway)
            },
        ) as CodeProviderFactory
    });
    register(
        registry,
        sandbox_type,
        SandboxProviderSet { fs, shell, code },
    )
}
/// Releases the cached remote containers of one sandbox type.
///
/// Python parity: `delete_jiuwenbox_sandbox` / `delete_yuanrong_sandbox`. The
/// kernel never calls a provider on release, so a host tears containers down
/// itself through this entry point: the selected entries leave the process-wide
/// cache and the provider issues its container-delete request (`2xx` and `404`
/// both count as released).
///
/// `aio` has no container-delete API — the AIO gateway owns the container
/// lifetime and reaps idle containers — so it answers
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// naming the gateway instead of inventing an endpoint.
///
/// # Errors
///
/// Returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for the `aio` sandbox type and for any name outside
/// [`REMOTE_SANDBOX_TYPES`], and
/// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// for a [`SandboxSelector`] that names nothing to release.
/// A container whose delete fails is reported in [`TeardownReport::failed`]
/// rather than failing the whole call.
///
/// # Example
///
/// ```no_run
/// use openjiuwen_extensions::sys_operation::{shutdown_remote, SandboxSelector};
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let selector = SandboxSelector::scope("http://127.0.0.1:8080", Some("agent-1"))
///     .with_reason("shutdown");
/// let report = shutdown_remote("jiuwenbox", &selector).await?;
/// assert!(report.failed.is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn shutdown_remote(
    sandbox_type: &str,
    selector: &SandboxSelector,
) -> Result<TeardownReport> {
    remote::teardown_remote(sandbox_type, selector).await
}

/// Drains the cached containers of every remote sandbox type.
///
/// Reports arrive in [`REMOTE_SANDBOX_TYPES`] order. A sandbox type without a
/// teardown path (today `aio`) does not fail the call: its report carries the
/// reason in [`TeardownReport::unsupported`], so an exit-time cleanup can drain
/// whatever can be drained.
///
/// # Errors
///
/// Returns the underlying error when a provider fails for a reason other than
/// the absence of a teardown path. Per-container delete failures stay in the
/// reports.
pub async fn shutdown_remote_all() -> Result<Vec<TeardownReport>> {
    let mut reports = Vec::with_capacity(REMOTE_SANDBOX_TYPES.len());
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        match remote::teardown_remote(sandbox_type, &SandboxSelector::all()).await {
            Ok(report) => reports.push(report),
            Err(error) if error.kind() == ErrorKind::Unsupported => {
                reports.push(TeardownReport::not_supported(sandbox_type, error.message()));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(reports)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, ErrorKind};
    use openjiuwen_core::sys_operation::{
        fs::types::{
            DownloadFileData, DownloadFileParams, ListDirsParams, ListFilesData, ListFilesParams,
            ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData, SearchFilesParams,
            TransferChunkData, UploadFileData, UploadFileParams, WriteFileData, WriteFileParams,
        },
        operation::{FsOperation, OpStream},
        result::{fs_failure, OpResult},
    };
    /// A filesystem backend that only identifies itself, used to prove a registered
    /// factory is the one the registry returns.
    struct MockFs;
    #[async_trait]
    impl FsOperation for MockFs {
        fn supports_conditional_write(&self) -> bool {
            true
        }
        async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
            fs_failure("read_file", &format!("mock: {}", params.path))
        }
        fn read_file_stream(self: Arc<Self>, _params: ReadFileParams) -> OpStream<ReadChunkData> {
            Box::pin(futures::stream::empty())
        }
        async fn write_file(&self, _params: WriteFileParams) -> OpResult<WriteFileData> {
            fs_failure("write_file", "mock")
        }
        async fn upload_file(&self, _params: UploadFileParams) -> OpResult<UploadFileData> {
            fs_failure("upload_file", "mock")
        }
        fn upload_file_stream(
            self: Arc<Self>,
            _params: UploadFileParams,
        ) -> OpStream<TransferChunkData> {
            Box::pin(futures::stream::empty())
        }
        async fn download_file(&self, _params: DownloadFileParams) -> OpResult<DownloadFileData> {
            fs_failure("download_file", "mock")
        }
        fn download_file_stream(
            self: Arc<Self>,
            _params: DownloadFileParams,
        ) -> OpStream<TransferChunkData> {
            Box::pin(futures::stream::empty())
        }
        async fn list_files(&self, _params: ListFilesParams) -> OpResult<ListFilesData> {
            fs_failure("list_files", "mock")
        }
        async fn list_directories(&self, _params: ListDirsParams) -> OpResult<ListFilesData> {
            fs_failure("list_directories", "mock")
        }
        async fn search_files(&self, _params: SearchFilesParams) -> OpResult<SearchFilesData> {
            fs_failure("search_files", "mock")
        }
    }
    fn mock_factory() -> FsProviderFactory {
        Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Ok(Arc::new(MockFs) as Arc<dyn FsOperation>)
            },
        )
    }
    fn expect_unsupported<T>(result: Result<T>) -> Error {
        match result {
            Ok(_) => panic!("expected the call to be refused as unsupported"),
            Err(error) => error,
        }
    }
    #[test]
    fn deferred_types_are_empty() {
        assert_eq!(DEFERRED_SANDBOX_TYPES, [] as [&str; 0]);
    }
    #[test]
    fn remote_types_match_the_python_provider_package() {
        assert_eq!(REMOTE_SANDBOX_TYPES, ["aio", "jiuwenbox", "yuanrong"]);
    }
    #[test]
    fn every_remote_type_yields_a_complete_set() {
        for sandbox_type in REMOTE_SANDBOX_TYPES {
            let set = remote_provider_set(sandbox_type).unwrap();
            assert!(!set.is_empty(), "{sandbox_type} must provide factories");
            assert!(
                set.fs.is_some(),
                "{sandbox_type} needs a filesystem factory"
            );
            assert!(set.shell.is_some(), "{sandbox_type} needs a shell factory");
            assert!(set.code.is_some(), "{sandbox_type} needs a code factory");
        }
    }
    #[test]
    fn empty_set_registers_nothing() {
        assert!(SandboxProviderSet::new().is_empty());
        assert!(!SandboxProviderSet::new().with_fs(mock_factory()).is_empty());
        let registry = SandboxProviderRegistry::new();
        register(&registry, "empty", SandboxProviderSet::new()).unwrap();
        assert!(!registry.has_fs("empty"));
        assert!(!registry.has_shell("empty"));
        assert!(!registry.has_code("empty"));
    }
    #[test]
    fn deferred_set_reports_unsupported_for_every_operation() {
        // Python counterpart: the `@SandboxRegistry.provider("<type>", ...)` classes of
        // tests/unit_tests/extensions/sys_operation/sandbox/ are registered but their
        // behavior is integration-gated; registration itself is the ported surface.
        let registry = SandboxProviderRegistry::new();
        register(&registry, "custom", SandboxProviderSet::deferred("custom")).unwrap();
        assert!(registry.has_fs("custom"));
        assert!(registry.has_shell("custom"));
        assert!(registry.has_code("custom"));
        assert!(!registry.has_fs("aio"));
        let endpoint = SandboxEndpoint::default();
        let config = SandboxGatewayConfig::default();
        for error in [
            expect_unsupported(registry.create_fs("custom", &endpoint, &config)),
            expect_unsupported(registry.create_shell("custom", &endpoint, &config)),
            expect_unsupported(registry.create_code("custom", &endpoint, &config)),
        ] {
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error
                .message()
                .contains("sandbox provider 'custom' is deferred"));
            assert!(error.message().contains("sys_operation::register"));
        }
    }
    #[test]
    fn register_remote_providers_installs_every_type() {
        let registry = SandboxProviderRegistry::new();
        register_remote_providers(&registry).unwrap();
        for sandbox_type in REMOTE_SANDBOX_TYPES {
            assert!(registry.has_fs(sandbox_type));
            assert!(registry.has_shell(sandbox_type));
            assert!(registry.has_code(sandbox_type));
        }
        assert_eq!(
            registered_types(&registry),
            vec![
                "aio".to_string(),
                "jiuwenbox".to_string(),
                "yuanrong".to_string()
            ]
        );
    }
    #[test]
    fn register_remote_providers_stops_at_the_first_error() {
        let registry = SandboxProviderRegistry::new();
        registry.register_fs("jiuwenbox", mock_factory()).unwrap();
        let error = register_remote_providers(&registry).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error.message().contains("jiuwenbox"));
        assert!(registry.has_fs("aio"));
        assert!(!registry.has_fs("yuanrong"));
    }
    #[test]
    fn register_remote_rejects_unknown_types() {
        let registry = SandboxProviderRegistry::new();
        let error = register_remote(&registry, "local").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error
            .message()
            .contains("supported types: aio, jiuwenbox, yuanrong"));
        assert!(!registry.has_fs("local"));
    }
    #[test]
    fn remote_provider_set_reports_the_unsupported_message() {
        let error = expect_unsupported(remote_provider_set("local"));
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "sandbox provider 'local' is not a remote provider; supported types: aio, jiuwenbox, yuanrong"
        );
    }
    #[tokio::test]
    async fn register_deferred_is_a_no_op() {
        let registry = SandboxProviderRegistry::new();
        register_deferred(&registry).await.unwrap();
        assert_eq!(registry.sandbox_types(), Vec::<String>::new());
        // Idempotent, and it never collides with a previous registration.
        register_deferred(&registry).await.unwrap();
        assert_eq!(registry.sandbox_types(), Vec::<String>::new());
        register_remote(&registry, "aio").unwrap();
        register_deferred(&registry).await.unwrap();
        assert_eq!(registered_types(&registry), vec!["aio".to_string()]);
    }
    #[test]
    fn duplicate_remote_registration_surfaces_sys_op() {
        let registry = SandboxProviderRegistry::new();
        register_remote(&registry, "aio").unwrap();
        let error = register_remote(&registry, "aio").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
        assert!(error.message().contains("aio"));
    }
    #[test]
    fn duplicate_factory_registration_surfaces_sys_op() {
        let registry = SandboxProviderRegistry::new();
        register(
            &registry,
            "mock",
            SandboxProviderSet::new().with_fs(mock_factory()),
        )
        .unwrap();
        let error = register(
            &registry,
            "mock",
            SandboxProviderSet::new().with_shell(Arc::new(
                |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                    Err(Error::unsupported("unused"))
                },
            )),
        );
        // The shell slot is free, so registration succeeds; re-registering fs fails.
        assert!(error.is_ok());
        let error = register(
            &registry,
            "mock",
            SandboxProviderSet::new().with_fs(mock_factory()),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::SysOp);
    }
    #[test]
    fn mock_provider_set_creates_the_registered_backend() {
        let registry = SandboxProviderRegistry::new();
        register(
            &registry,
            "mock",
            SandboxProviderSet::new().with_fs(mock_factory()),
        )
        .unwrap();
        let fs = registry
            .create_fs(
                "mock",
                &SandboxEndpoint::default(),
                &SandboxGatewayConfig::default(),
            )
            .unwrap();
        assert!(fs.supports_conditional_write());
        assert_eq!(registered_types(&registry), vec!["mock".to_string()]);
        registry.unregister("mock");
        assert_eq!(registered_types(&registry), Vec::<String>::new());
    }
    #[test]
    fn is_deferred_marks_everything_outside_the_remote_set() {
        for sandbox_type in REMOTE_SANDBOX_TYPES {
            assert!(!is_deferred(sandbox_type));
        }
        assert!(is_deferred("local"));
        assert!(is_deferred(""));
    }
    #[test]
    fn a_remote_provider_needs_a_base_url() {
        // The provider implementations land separately; only the failure is asserted here.
        let registry = SandboxProviderRegistry::new();
        register_remote(&registry, "aio").unwrap();
        assert!(registry
            .create_fs(
                "aio",
                &SandboxEndpoint::default(),
                &SandboxGatewayConfig::default()
            )
            .is_err());
    }
    #[test]
    fn gateway_provider_binds_the_shared_configuration() {
        let gateway = SandboxGatewayConfig {
            timeout_seconds: 99,
            ..SandboxGatewayConfig::default()
        };
        let factory: FsProviderFactory = Arc::new(
            |_endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                if config.timeout_seconds == 99 {
                    Ok(Arc::new(MockFs) as Arc<dyn FsOperation>)
                } else {
                    Err(Error::unsupported("gateway configuration was not bound"))
                }
            },
        );
        let registry = SandboxProviderRegistry::new();
        register_gateway_provider(
            &registry,
            "gateway",
            gateway,
            SandboxProviderSet::new().with_fs(factory),
        )
        .unwrap();
        assert!(registry.has_fs("gateway"));
        assert!(!registry.has_shell("gateway"));
        let fs = registry
            .create_fs(
                "gateway",
                &SandboxEndpoint::default(),
                // The caller's configuration is overridden by the bound gateway.
                &SandboxGatewayConfig::default(),
            )
            .unwrap();
        assert!(fs.supports_conditional_write());
    }
}
