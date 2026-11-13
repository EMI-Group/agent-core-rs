//! Vendor memory-provider adapters.
//!
//! Every adapter in this module is a typed handle whose transport — a vendor SDK, an
//! HTTP client, or a database client — is not part of this build. Construction therefore
//! succeeds and every operation returns
//! [`openjiuwen_common::error::Error::Unsupported`] whose message names the missing SDK
//! or transport verbatim, matching the Python package's lazy client acquisition (a
//! missing extra surfaces at first use, not at import).
//!
//! The five adapters share one `VendorProvider` helper, so the error text, the
//! conservative capability flags, and the health behaviour cannot drift between them.
//! No wire format is invented here: adding a real transport means replacing the helper
//! with a client and keeping the public handle unchanged.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

use super::provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
use super::registry::{ProviderConfig, ProviderFactory, ProviderKind};

/// The shared backend of every vendor stub.
///
/// `id` is the registry name (the Python provider id), `display` is the human-readable
/// vendor name used in error messages, and `missing` names the exact SDK or transport
/// this build does not include.
struct VendorProvider {
    /// Registry name of the adapter (`agentarts`, `mem0`, …).
    id: &'static str,
    /// Human-readable vendor name used in diagnostics.
    display: &'static str,
    /// The missing SDK or transport, named verbatim.
    missing: &'static str,
    /// The configuration the handle was built from.
    config: ProviderConfig,
}

impl VendorProvider {
    /// Builds the helper for one vendor.
    fn new(
        config: &ProviderConfig,
        id: &'static str,
        display: &'static str,
        missing: &'static str,
    ) -> Self {
        Self {
            id,
            display,
            missing,
            config: config.clone(),
        }
    }

    /// The single typed error every operation returns.
    fn error(&self) -> Error {
        Error::unsupported(format!(
            "the {} memory provider requires {}, which this build does not provide",
            self.display, self.missing
        ))
    }
}

#[async_trait]
impl MemoryProvider for VendorProvider {
    fn name(&self) -> &str {
        self.id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // A stub has no transport, so it declares no capability at all.
        ProviderCapabilities::none()
    }

    async fn initialize(&self) -> Result<()> {
        Err(self.error())
    }

    async fn add(
        &self,
        _scope: &MemoryScope,
        _memories: Vec<NewMemory>,
        _options: &AddOptions,
    ) -> Result<AddOutcome> {
        Err(self.error())
    }

    async fn search(
        &self,
        _scope: &MemoryScope,
        _query: &str,
        _options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        Err(self.error())
    }

    async fn get(&self, _scope: &MemoryScope, _id: &str) -> Result<Option<MemoryRecord>> {
        Err(self.error())
    }

    async fn update(
        &self,
        _scope: &MemoryScope,
        _id: &str,
        _update: MemoryUpdate,
        _options: &CallOptions,
    ) -> Result<MemoryRecord> {
        Err(self.error())
    }

    async fn delete(&self, _scope: &MemoryScope, _id: &str) -> Result<bool> {
        Err(self.error())
    }

    async fn list(&self, _filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        Err(self.error())
    }

    async fn clear(&self, _filter: &MemoryFilter) -> Result<usize> {
        Err(self.error())
    }

    async fn health(&self) -> Result<ProviderHealth> {
        Err(self.error())
    }
}

/// Generates one vendor adapter around [`VendorProvider`].
///
/// Every generated adapter has the same shape: an infallible `new`, the `config`/`kind`
/// accessors, a redacting `Debug`, and a `MemoryProvider` impl whose every operation is
/// the same typed `Unsupported` error naming the missing SDK or transport.
macro_rules! vendor_adapter {
    (
        $(#[$meta:meta])*
        $adapter:ident, $kind:expr, $id:literal, $display:literal, $missing:literal $(,)?
    ) => {
        $(#[$meta])*
        pub struct $adapter {
            /// The shared unsupported backend.
            inner: VendorProvider,
        }

        impl $adapter {
            /// Builds the adapter handle for `config`.
            ///
            /// Construction never fails: the vendor SDK or transport is required at
            /// [`MemoryProvider::initialize`], mirroring the lazy client acquisition of
            /// the Python adapters.
            #[must_use]
            pub fn new(config: &ProviderConfig) -> Self {
                Self {
                    inner: VendorProvider::new(config, $id, $display, $missing),
                }
            }

            /// The configuration this handle was built from.
            #[must_use]
            pub fn config(&self) -> &ProviderConfig {
                &self.inner.config
            }

            /// The kind this adapter implements.
            #[must_use]
            pub fn kind(&self) -> ProviderKind {
                $kind
            }
        }

        impl fmt::Debug for $adapter {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($adapter))
                    .field("name", &$id)
                    .field("missing", &$missing)
                    .finish_non_exhaustive()
            }
        }

        #[async_trait::async_trait]
        impl MemoryProvider for $adapter {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn capabilities(&self) -> ProviderCapabilities {
                self.inner.capabilities()
            }

            async fn initialize(&self) -> Result<()> {
                self.inner.initialize().await
            }

            async fn add(
                &self,
                scope: &MemoryScope,
                memories: Vec<NewMemory>,
                options: &AddOptions,
            ) -> Result<AddOutcome> {
                self.inner.add(scope, memories, options).await
            }

            async fn search(
                &self,
                scope: &MemoryScope,
                query: &str,
                options: &SearchOptions,
            ) -> Result<Vec<RecallHit>> {
                self.inner.search(scope, query, options).await
            }

            async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
                self.inner.get(scope, id).await
            }

            async fn update(
                &self,
                scope: &MemoryScope,
                id: &str,
                update: MemoryUpdate,
                options: &CallOptions,
            ) -> Result<MemoryRecord> {
                self.inner.update(scope, id, update, options).await
            }

            async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool> {
                self.inner.delete(scope, id).await
            }

            async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
                self.inner.list(filter).await
            }

            async fn clear(&self, filter: &MemoryFilter) -> Result<usize> {
                self.inner.clear(filter).await
            }

            async fn health(&self) -> Result<ProviderHealth> {
                self.inner.health().await
            }
        }
    };
}

vendor_adapter! {
    /// Handle for the `AgentArts` memory service (`agentarts`).
    ///
    /// The Python adapter drives `agentarts.sdk.memory.MemoryClient` and an HTTP data
    /// plane; this build ships neither.
    AgentArtsMemoryProvider,
    ProviderKind::AgentArts,
    "agentarts",
    "AgentArts",
    "the `agentarts-sdk` package (`MemoryClient`) and its HTTP transport",
}

vendor_adapter! {
    /// Handle for the `Jiuwen` memory server or SDK (`jiuwen_memory`).
    ///
    /// The Python adapter speaks the Jiuwen `/v1/*` HTTP API (`httpx.AsyncClient`) or
    /// imports the in-process `jiuwen_memory` kernel; this build ships neither.
    JiuwenMemoryProvider,
    ProviderKind::Jiuwen,
    "jiuwen_memory",
    "Jiuwen",
    "the `jiuwen_memory` SDK and its HTTP server transport",
}

vendor_adapter! {
    /// Handle for the `LakeBase`/`DBay` memory service (`lakebase`).
    ///
    /// The Python adapter speaks the `LakeBase` REST API over `httpx.AsyncClient`; this
    /// build ships no HTTP transport.
    LakeBaseMemoryProvider,
    ProviderKind::LakeBase,
    "lakebase",
    "LakeBase",
    "the LakeBase REST transport (`httpx.AsyncClient`)",
}

vendor_adapter! {
    /// Handle for the `mem0` hosted memory service (`mem0`).
    ///
    /// The Python adapter drives `mem0ai`'s `AsyncMemoryClient`; this build ships no
    /// `mem0ai` package.
    Mem0MemoryProvider,
    ProviderKind::Mem0,
    "mem0",
    "mem0",
    "the `mem0ai` SDK (`AsyncMemoryClient`)",
}

vendor_adapter! {
    /// Handle for the `OpenViking` context database (`openviking`).
    ///
    /// The Python adapter drives the `OpenViking` REST API through a sync `httpx` client;
    /// this build ships neither the client nor the transport.
    OpenVikingMemoryProvider,
    ProviderKind::OpenViking,
    "openviking",
    "OpenViking",
    "the OpenViking API client and its HTTP transport",
}

/// Builds a provider from `config`, the constructor behind one vendor factory.
type VendorBuilder = fn(&ProviderConfig) -> Arc<dyn MemoryProvider>;

/// The factory of one vendor adapter.
struct VendorFactory {
    /// The kind this factory builds.
    kind: ProviderKind,
    /// The adapter constructor.
    builder: VendorBuilder,
}

impl VendorFactory {
    /// Builds a factory for `kind` around `builder`.
    fn new(kind: ProviderKind, builder: VendorBuilder) -> Self {
        Self { kind, builder }
    }
}

impl ProviderFactory for VendorFactory {
    fn kind(&self) -> ProviderKind {
        self.kind.clone()
    }

    fn build(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>> {
        if config.kind != self.kind {
            return Err(Error::validation(format!(
                "the `{}` factory cannot build kind `{}`",
                self.kind, config.kind
            )));
        }
        Ok((self.builder)(config))
    }
}

/// Builds an `AgentArts` adapter.
fn build_agentarts(config: &ProviderConfig) -> Arc<dyn MemoryProvider> {
    Arc::new(AgentArtsMemoryProvider::new(config))
}

/// Builds a `Jiuwen` adapter.
fn build_jiuwen(config: &ProviderConfig) -> Arc<dyn MemoryProvider> {
    Arc::new(JiuwenMemoryProvider::new(config))
}

/// Builds a `LakeBase` adapter.
fn build_lakebase(config: &ProviderConfig) -> Arc<dyn MemoryProvider> {
    Arc::new(LakeBaseMemoryProvider::new(config))
}

/// Builds a `mem0` adapter.
fn build_mem0(config: &ProviderConfig) -> Arc<dyn MemoryProvider> {
    Arc::new(Mem0MemoryProvider::new(config))
}

/// Builds an `OpenViking` adapter.
fn build_openviking(config: &ProviderConfig) -> Arc<dyn MemoryProvider> {
    Arc::new(OpenVikingMemoryProvider::new(config))
}

/// The five vendor factories, in [`ProviderKind::BUILTIN`] vendor order.
///
/// [`ProviderRegistry::with_vendor_stubs`](crate::external::ProviderRegistry::with_vendor_stubs)
/// registers exactly this list.
#[must_use]
pub fn vendor_factories() -> Vec<Arc<dyn ProviderFactory>> {
    vec![
        Arc::new(VendorFactory::new(ProviderKind::AgentArts, build_agentarts)),
        Arc::new(VendorFactory::new(ProviderKind::Jiuwen, build_jiuwen)),
        Arc::new(VendorFactory::new(ProviderKind::LakeBase, build_lakebase)),
        Arc::new(VendorFactory::new(ProviderKind::Mem0, build_mem0)),
        Arc::new(VendorFactory::new(
            ProviderKind::OpenViking,
            build_openviking,
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(kind: ProviderKind) -> ProviderConfig {
        ProviderConfig::new(kind)
            .with_endpoint("https://example.invalid")
            .with_api_key("sk-secret")
    }

    /// Asserts that `result` is a typed `Unsupported` naming `token`.
    fn assert_unsupported<T>(result: Result<T>, token: &str) {
        let error = result.err().expect("the vendor stub must fail");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        assert!(
            error.to_string().contains(token),
            "message `{error}` must name `{token}`"
        );
    }

    #[test]
    fn adapters_report_their_python_provider_ids() {
        assert_eq!(
            AgentArtsMemoryProvider::new(&config(ProviderKind::AgentArts)).name(),
            "agentarts"
        );
        assert_eq!(
            JiuwenMemoryProvider::new(&config(ProviderKind::Jiuwen)).name(),
            "jiuwen_memory"
        );
        assert_eq!(
            LakeBaseMemoryProvider::new(&config(ProviderKind::LakeBase)).name(),
            "lakebase"
        );
        assert_eq!(
            Mem0MemoryProvider::new(&config(ProviderKind::Mem0)).name(),
            "mem0"
        );
        assert_eq!(
            OpenVikingMemoryProvider::new(&config(ProviderKind::OpenViking)).name(),
            "openviking"
        );
    }

    #[test]
    fn adapters_expose_their_kind_and_configuration() {
        let adapter = Mem0MemoryProvider::new(&config(ProviderKind::Mem0));
        assert_eq!(adapter.kind(), ProviderKind::Mem0);
        assert_eq!(
            adapter.config().endpoint.as_deref(),
            Some("https://example.invalid")
        );
        assert_eq!(adapter.capabilities(), ProviderCapabilities::none());
        assert_eq!(adapter.name(), "mem0");
    }

    #[tokio::test]
    async fn every_operation_returns_the_same_unsupported_error() {
        let adapters: Vec<(Arc<dyn MemoryProvider>, &str)> = vec![
            (
                Arc::new(AgentArtsMemoryProvider::new(&config(
                    ProviderKind::AgentArts,
                ))),
                "agentarts-sdk",
            ),
            (
                Arc::new(JiuwenMemoryProvider::new(&config(ProviderKind::Jiuwen))),
                "jiuwen_memory",
            ),
            (
                Arc::new(LakeBaseMemoryProvider::new(&config(ProviderKind::LakeBase))),
                "httpx.AsyncClient",
            ),
            (
                Arc::new(Mem0MemoryProvider::new(&config(ProviderKind::Mem0))),
                "mem0ai",
            ),
            (
                Arc::new(OpenVikingMemoryProvider::new(&config(
                    ProviderKind::OpenViking,
                ))),
                "OpenViking API client",
            ),
        ];
        let scope = MemoryScope::new("alice", "ws");
        let filter = MemoryFilter::for_scope(&scope);
        for (provider, token) in adapters {
            assert_unsupported(provider.initialize().await, token);
            assert_unsupported(
                provider
                    .add(
                        &scope,
                        vec![NewMemory::semantic("anything")],
                        &AddOptions::default(),
                    )
                    .await,
                token,
            );
            assert_unsupported(
                provider
                    .search(&scope, "anything", &SearchOptions::default())
                    .await,
                token,
            );
            assert_unsupported(provider.get(&scope, "id").await, token);
            assert_unsupported(
                provider
                    .update(
                        &scope,
                        "id",
                        MemoryUpdate::content("x"),
                        &CallOptions::default(),
                    )
                    .await,
                token,
            );
            assert_unsupported(provider.delete(&scope, "id").await, token);
            assert_unsupported(provider.list(&filter).await, token);
            assert_unsupported(provider.clear(&filter).await, token);
            assert_unsupported(provider.health().await, token);
        }
    }

    #[test]
    fn adapter_debug_does_not_leak_the_credential() {
        let adapter = AgentArtsMemoryProvider::new(&config(ProviderKind::AgentArts));
        let debug = format!("{adapter:?}");
        assert!(debug.contains("AgentArtsMemoryProvider"));
        assert!(debug.contains("agentarts-sdk"));
        assert!(!debug.contains("sk-secret"), "{debug}");
    }

    #[test]
    fn vendor_factories_cover_the_five_vendors() {
        let factories = vendor_factories();
        let kinds: Vec<ProviderKind> = factories.iter().map(|factory| factory.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                ProviderKind::AgentArts,
                ProviderKind::Jiuwen,
                ProviderKind::LakeBase,
                ProviderKind::Mem0,
                ProviderKind::OpenViking,
            ]
        );

        let error = factories[0]
            .build(&ProviderConfig::new(ProviderKind::Mem0))
            .err()
            .expect("kind mismatch");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        for factory in &factories {
            let kind = factory.kind();
            let provider = factory
                .build(&ProviderConfig::new(kind.clone()))
                .unwrap_or_else(|error| panic!("`{kind}` must build: {error}"));
            assert_eq!(provider.name(), kind.as_str());
        }
    }
}
