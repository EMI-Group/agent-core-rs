//! Sandbox provider registration facade.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/`. In the
//! reference implementation, importing `providers` registers the `aio`, `jiuwenbox`, and
//! `yuanrong` sandbox providers into `SandboxRegistry` at import time
//! (`@SandboxRegistry.provider("<type>", "fs" | "shell" | "code")`).
//!
//! This module keeps that registration seam and drops the protocol implementations: a
//! host builds a [`SandboxProviderSet`] (any subset of filesystem, shell, and code
//! factories) and registers it with [`register`]; Python's three providers are available
//! as documented deferred sets through [`register_deferred`], whose factories answer
//! [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) with a
//! message naming the sandbox type.
//!
//! The provider seam itself is the kernel's
//! (`openjiuwen_sysop::sandbox::provider`, reachable as
//! `openjiuwen_core::sys_operation::sandbox::provider`), so a backend registered here is
//! the same one a `SysOperation` card resolves.
//!
//! ```
//! use openjiuwen_core::sys_operation::sandbox::provider::SandboxProviderRegistry;
//! use openjiuwen_extensions::sys_operation::{SandboxProviderSet, is_deferred, register};
//!
//! let registry = SandboxProviderRegistry::new();
//! register(&registry, "aio", SandboxProviderSet::deferred("aio")).unwrap();
//! assert!(registry.has_fs("aio"));
//! assert!(is_deferred("aio"));
//! assert!(registry
//!     .create_fs("aio", &Default::default(), &Default::default())
//!     .is_err());
//! ```
#![allow(clippy::module_name_repetitions)] // `FsProviderFactory`/`CodeProviderFactory` are the contract's names.
use openjiuwen_common::error::{Error, Result};
pub use openjiuwen_core::sys_operation::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};
use openjiuwen_core::sys_operation::sandbox::provider::{
    CodeProviderFactory, FsProviderFactory, SandboxProviderRegistry, ShellProviderFactory,
};
use std::sync::Arc;
/// The sandbox types Python registers from `extensions/sys_operation/sandbox/providers/`.
///
/// Every name stays usable in configuration written for Python; the providers behind
/// them are deferred, so creation reports `Unsupported` instead of contacting a sandbox.
pub const DEFERRED_SANDBOX_TYPES: [&str; 3] = ["aio", "jiuwenbox", "yuanrong"];
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
    /// This keeps Python's provider names registered (so `sandbox_type = "aio"` in a card
    /// still resolves) while making the missing protocol explicit at call time.
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
/// Python's provider is documented as deferred rather than silently missing.
fn deferred_error(sandbox_type: &str) -> Error {
    Error::unsupported(format!(
        "sandbox provider '{sandbox_type}' is deferred; register a provider with sys_operation::register"
    ))
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
/// Registers the deferred Python providers ([`DEFERRED_SANDBOX_TYPES`]).
///
/// This is the Rust stand-in for importing the Python `providers` package: every type
/// keeps a filesystem, shell, and code factory that answers `Unsupported`.
///
/// The function is `async` to mirror the Python async setup path that registers
/// providers; it performs no I/O.
///
/// # Errors
///
/// Returns [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp) when any of
/// the types is already registered, leaving the types registered earlier in
/// [`DEFERRED_SANDBOX_TYPES`] in place.
#[allow(clippy::unused_async)] // Python parity: registration happens in the async setup path.
pub async fn register_deferred(registry: &SandboxProviderRegistry) -> Result<()> {
    for sandbox_type in DEFERRED_SANDBOX_TYPES {
        register(
            registry,
            sandbox_type,
            SandboxProviderSet::deferred(sandbox_type),
        )?;
    }
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
#[must_use]
pub fn is_deferred(sandbox_type: &str) -> bool {
    DEFERRED_SANDBOX_TYPES.contains(&sandbox_type)
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
            Ok(_) => panic!("expected the deferred provider to refuse the call"),
            Err(error) => error,
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
        register(&registry, "aio", SandboxProviderSet::deferred("aio")).unwrap();
        assert!(registry.has_fs("aio"));
        assert!(registry.has_shell("aio"));
        assert!(registry.has_code("aio"));
        assert!(!registry.has_fs("jiuwenbox"));
        let endpoint = SandboxEndpoint::default();
        let config = SandboxGatewayConfig::default();
        for error in [
            expect_unsupported(registry.create_fs("aio", &endpoint, &config)),
            expect_unsupported(registry.create_shell("aio", &endpoint, &config)),
            expect_unsupported(registry.create_code("aio", &endpoint, &config)),
        ] {
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error
                .message()
                .contains("sandbox provider 'aio' is deferred"));
            assert!(error.message().contains("sys_operation::register"));
        }
    }
    #[tokio::test]
    async fn register_deferred_populates_the_python_types() {
        let registry = SandboxProviderRegistry::new();
        register_deferred(&registry).await.unwrap();
        for sandbox_type in DEFERRED_SANDBOX_TYPES {
            assert!(is_deferred(sandbox_type));
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
        assert!(!is_deferred("local"));
    }
    #[tokio::test]
    async fn duplicate_registration_surfaces_sys_op() {
        let registry = SandboxProviderRegistry::new();
        register_deferred(&registry).await.unwrap();
        let error = register_deferred(&registry).await.unwrap_err();
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
