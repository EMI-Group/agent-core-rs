//! The sandbox provider seam.
//!
//! Python parity: `core/sys_operation/protocal/*.py` (`BaseFsProtocal`,
//! `BaseShellProtocal`, `BaseCodeProtocal`, spelled `protocal` in the reference) plus
//! `sandbox/sandbox_registry.py`. A provider is a factory that, given a resolved
//! [`SandboxEndpoint`] and the gateway configuration, produces a backend implementing the
//! same [`FsOperation`] / [`ShellOperation`] / [`CodeOperation`] traits the local backend
//! implements. Registration is keyed by `(sandbox_type, operation)`; `openjiuwen-extensions`
//! registers the concrete AIO/JiuwenBox/YuanRong providers at startup.
//!
//! ```
//! use openjiuwen_sysop::sandbox::provider::SandboxProviderRegistry;
//!
//! let registry = SandboxProviderRegistry::new();
//! assert!(!registry.has_fs("aio"));
//! assert!(registry.create_fs("aio", &Default::default(), &Default::default()).is_err());
//! ```

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock, RwLock},
};

use openjiuwen_common::error::{Error, Result};

use crate::{
    operation::{CodeOperation, FsOperation, ShellOperation},
    sandbox::config::{SandboxEndpoint, SandboxGatewayConfig},
    status::{registry_message, sandbox_provider_message},
};

/// Builds a sandbox filesystem backend for one endpoint.
pub type FsProviderFactory = Arc<
    dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<Arc<dyn FsOperation>> + Send + Sync,
>;

/// Builds a sandbox shell backend for one endpoint.
pub type ShellProviderFactory = Arc<
    dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<Arc<dyn ShellOperation>>
        + Send
        + Sync,
>;

/// Builds a sandbox code backend for one endpoint.
pub type CodeProviderFactory = Arc<
    dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<Arc<dyn CodeOperation>> + Send + Sync,
>;

/// Registry of sandbox providers, keyed by `sandbox_type`.
///
/// Python parity: `SandboxRegistry` class-level registries (`register_provider`,
/// `create_provider`, `unregister_provider`). The registry is cheap to clone and safe to
/// share; the process-global instance is [`global`].
#[derive(Clone, Default)]
pub struct SandboxProviderRegistry {
    fs: Arc<RwLock<BTreeMap<String, FsProviderFactory>>>,
    shell: Arc<RwLock<BTreeMap<String, ShellProviderFactory>>>,
    code: Arc<RwLock<BTreeMap<String, CodeProviderFactory>>>,
}

impl std::fmt::Debug for SandboxProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A poisoned provider lock is rendered explicitly: an empty list would claim that
        // no provider is registered, which is a different state.
        fn types<T>(map: &RwLock<BTreeMap<String, T>>) -> Vec<String> {
            match map.read() {
                Ok(map) => map.keys().cloned().collect(),
                Err(_) => vec!["<poisoned>".to_string()],
            }
        }
        f.debug_struct("SandboxProviderRegistry")
            .field("fs", &types(&self.fs))
            .field("shell", &types(&self.shell))
            .field("code", &types(&self.code))
            .finish()
    }
}

impl SandboxProviderRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a filesystem provider factory for `sandbox_type`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` when a factory is already registered for that
    /// `sandbox_type` (Python raises on duplicates).
    pub fn register_fs(
        &self,
        sandbox_type: impl Into<String>,
        factory: FsProviderFactory,
    ) -> Result<()> {
        let sandbox_type = sandbox_type.into();
        let mut map = self.fs.write().map_err(|_| poisoned("fs"))?;
        if map.contains_key(&sandbox_type) {
            return Err(Error::sys_op(sandbox_provider_message(
                "register_fs",
                &format!("sandbox type '{sandbox_type}' is already registered"),
            )));
        }
        map.insert(sandbox_type, factory);
        Ok(())
    }

    /// Registers a shell provider factory for `sandbox_type`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` when a factory is already registered.
    pub fn register_shell(
        &self,
        sandbox_type: impl Into<String>,
        factory: ShellProviderFactory,
    ) -> Result<()> {
        let sandbox_type = sandbox_type.into();
        let mut map = self.shell.write().map_err(|_| poisoned("shell"))?;
        if map.contains_key(&sandbox_type) {
            return Err(Error::sys_op(sandbox_provider_message(
                "register_shell",
                &format!("sandbox type '{sandbox_type}' is already registered"),
            )));
        }
        map.insert(sandbox_type, factory);
        Ok(())
    }

    /// Registers a code provider factory for `sandbox_type`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` when a factory is already registered.
    pub fn register_code(
        &self,
        sandbox_type: impl Into<String>,
        factory: CodeProviderFactory,
    ) -> Result<()> {
        let sandbox_type = sandbox_type.into();
        let mut map = self.code.write().map_err(|_| poisoned("code"))?;
        if map.contains_key(&sandbox_type) {
            return Err(Error::sys_op(sandbox_provider_message(
                "register_code",
                &format!("sandbox type '{sandbox_type}' is already registered"),
            )));
        }
        map.insert(sandbox_type, factory);
        Ok(())
    }

    /// Removes every factory registered for `sandbox_type`.
    ///
    /// A poisoned lock on any of the three maps is reported at `warn!` instead of being
    /// ignored: that kind's factory then stays registered and still builds backends.
    pub fn unregister(&self, sandbox_type: &str) {
        let fs_removed = self
            .fs
            .write()
            .map(|mut map| map.remove(sandbox_type))
            .is_ok();
        let shell_removed = self
            .shell
            .write()
            .map(|mut map| map.remove(sandbox_type))
            .is_ok();
        let code_removed = self
            .code
            .write()
            .map(|mut map| map.remove(sandbox_type))
            .is_ok();
        for (kind, removed) in [
            ("fs", fs_removed),
            ("shell", shell_removed),
            ("code", code_removed),
        ] {
            if !removed {
                tracing::warn!(
                    sandbox_type,
                    kind,
                    "the provider registry lock is poisoned; that provider was not removed"
                );
            }
        }
    }

    /// Whether a filesystem provider is registered for `sandbox_type`.
    #[must_use]
    pub fn has_fs(&self, sandbox_type: &str) -> bool {
        if let Ok(map) = self.fs.read() {
            map.contains_key(sandbox_type)
        } else {
            tracing::warn!(
                sandbox_type,
                kind = "fs",
                "the provider registry lock is poisoned; reporting no such provider"
            );
            false
        }
    }

    /// Whether a shell provider is registered for `sandbox_type`.
    #[must_use]
    pub fn has_shell(&self, sandbox_type: &str) -> bool {
        if let Ok(map) = self.shell.read() {
            map.contains_key(sandbox_type)
        } else {
            tracing::warn!(
                sandbox_type,
                kind = "shell",
                "the provider registry lock is poisoned; reporting no such provider"
            );
            false
        }
    }

    /// Whether a code provider is registered for `sandbox_type`.
    #[must_use]
    pub fn has_code(&self, sandbox_type: &str) -> bool {
        if let Ok(map) = self.code.read() {
            map.contains_key(sandbox_type)
        } else {
            tracing::warn!(
                sandbox_type,
                kind = "code",
                "the provider registry lock is poisoned; reporting no such provider"
            );
            false
        }
    }

    /// The registered sandbox types, sorted.
    ///
    /// A poisoned registry lock yields only the types of the readable maps (an infallible
    /// accessor cannot report the failure), and that gap is logged at `warn!`.
    #[must_use]
    pub fn sandbox_types(&self) -> Vec<String> {
        let mut types = Vec::new();
        if let Ok(map) = self.fs.read() {
            types.extend(map.keys().cloned());
        } else {
            tracing::warn!(
                "the provider registry lock is poisoned; the reported sandbox types are incomplete"
            );
        }
        types.sort();
        types.dedup();
        types
    }

    /// Builds a filesystem backend for `endpoint`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Unsupported` with Python's `NotImplementedError` text when no
    /// provider is registered, and propagates factory errors.
    pub fn create_fs(
        &self,
        sandbox_type: &str,
        endpoint: &SandboxEndpoint,
        config: &SandboxGatewayConfig,
    ) -> Result<Arc<dyn FsOperation>> {
        let factory = {
            let map = self.fs.read().map_err(|_| poisoned("fs"))?;
            map.get(sandbox_type).cloned()
        };
        let factory = factory.ok_or_else(|| unsupported(sandbox_type, "fs"))?;
        factory(endpoint, config)
    }

    /// Builds a shell backend for `endpoint`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Unsupported` when no provider is registered.
    pub fn create_shell(
        &self,
        sandbox_type: &str,
        endpoint: &SandboxEndpoint,
        config: &SandboxGatewayConfig,
    ) -> Result<Arc<dyn ShellOperation>> {
        let factory = {
            let map = self.shell.read().map_err(|_| poisoned("shell"))?;
            map.get(sandbox_type).cloned()
        };
        let factory = factory.ok_or_else(|| unsupported(sandbox_type, "shell"))?;
        factory(endpoint, config)
    }

    /// Builds a code backend for `endpoint`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Unsupported` when no provider is registered.
    pub fn create_code(
        &self,
        sandbox_type: &str,
        endpoint: &SandboxEndpoint,
        config: &SandboxGatewayConfig,
    ) -> Result<Arc<dyn CodeOperation>> {
        let factory = {
            let map = self.code.read().map_err(|_| poisoned("code"))?;
            map.get(sandbox_type).cloned()
        };
        let factory = factory.ok_or_else(|| unsupported(sandbox_type, "code"))?;
        factory(endpoint, config)
    }
}

/// The process-global provider registry.
///
/// Python parity: `SandboxRegistry` is class-level state shared by the process. A local
/// [`SandboxProviderRegistry`] should be used in tests and for isolation.
#[must_use]
pub fn global() -> &'static SandboxProviderRegistry {
    static REGISTRY: OnceLock<SandboxProviderRegistry> = OnceLock::new();
    REGISTRY.get_or_init(SandboxProviderRegistry::new)
}

/// Python's `NotImplementedError` text for a missing provider.
fn unsupported(sandbox_type: &str, operation: &str) -> Error {
    Error::unsupported(format!(
        "Sandbox type '{sandbox_type}' does not support operation '{operation}'"
    ))
}

/// Maps a poisoned registry lock onto the registry error.
fn poisoned(operation: &str) -> Error {
    Error::sys_op(registry_message(
        "sandbox_provider",
        &format!("registry lock poisoned while accessing '{operation}'"),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{
        fs::types::{ReadFileData, ReadFileParams},
        result::{fs_failure, OpResult},
    };

    struct NullFs;

    #[async_trait::async_trait]
    impl FsOperation for NullFs {
        fn supports_conditional_write(&self) -> bool {
            true
        }

        async fn read_file(&self, _params: ReadFileParams) -> OpResult<ReadFileData> {
            fs_failure("read_file", "null provider")
        }

        fn read_file_stream(
            self: Arc<Self>,
            _params: ReadFileParams,
        ) -> crate::operation::OpStream<crate::fs::types::ReadChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn write_file(
            &self,
            _params: crate::fs::types::WriteFileParams,
        ) -> OpResult<crate::fs::types::WriteFileData> {
            fs_failure("write_file", "null provider")
        }

        async fn upload_file(
            &self,
            _params: crate::fs::types::UploadFileParams,
        ) -> OpResult<crate::fs::types::UploadFileData> {
            fs_failure("upload_file", "null provider")
        }

        fn upload_file_stream(
            self: Arc<Self>,
            _params: crate::fs::types::UploadFileParams,
        ) -> crate::operation::OpStream<crate::fs::types::TransferChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn download_file(
            &self,
            _params: crate::fs::types::DownloadFileParams,
        ) -> OpResult<crate::fs::types::DownloadFileData> {
            fs_failure("download_file", "null provider")
        }

        fn download_file_stream(
            self: Arc<Self>,
            _params: crate::fs::types::DownloadFileParams,
        ) -> crate::operation::OpStream<crate::fs::types::TransferChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn list_files(
            &self,
            _params: crate::fs::types::ListFilesParams,
        ) -> OpResult<crate::fs::types::ListFilesData> {
            fs_failure("list_files", "null provider")
        }

        async fn list_directories(
            &self,
            _params: crate::fs::types::ListDirsParams,
        ) -> OpResult<crate::fs::types::ListFilesData> {
            fs_failure("list_directories", "null provider")
        }

        async fn search_files(
            &self,
            _params: crate::fs::types::SearchFilesParams,
        ) -> OpResult<crate::fs::types::SearchFilesData> {
            fs_failure("search_files", "null provider")
        }
    }

    fn null_factory() -> FsProviderFactory {
        Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Ok(Arc::new(NullFs) as Arc<dyn FsOperation>)
            },
        )
    }

    #[test]
    fn registration_and_creation() {
        let registry = SandboxProviderRegistry::new();
        assert!(!registry.has_fs("mock"));
        match registry.create_fs(
            "mock",
            &SandboxEndpoint::default(),
            &SandboxGatewayConfig::default(),
        ) {
            Ok(_) => panic!("expected an unsupported-provider error"),
            Err(error) => {
                assert_eq!(
                    error.kind(),
                    openjiuwen_common::error::ErrorKind::Unsupported
                );
                assert!(error.message().contains("does not support operation 'fs'"));
            }
        }

        registry.register_fs("mock", null_factory()).unwrap();
        assert!(registry.has_fs("mock"));
        assert_eq!(registry.sandbox_types(), vec!["mock".to_string()]);
        let fs = registry
            .create_fs(
                "mock",
                &SandboxEndpoint::default(),
                &SandboxGatewayConfig::default(),
            )
            .unwrap();
        assert!(fs.supports_conditional_write());

        registry.unregister("mock");
        assert!(!registry.has_fs("mock"));
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let registry = SandboxProviderRegistry::new();
        registry.register_fs("mock", null_factory()).unwrap();
        let error = registry.register_fs("mock", null_factory()).unwrap_err();
        assert!(error.message().contains("already registered"));
    }

    #[test]
    fn global_registry_is_stable() {
        assert!(std::ptr::eq(global(), global()));
    }
}
