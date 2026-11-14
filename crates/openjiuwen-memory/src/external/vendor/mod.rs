//! Vendor memory-provider adapters, one module per vendor.
//!
//! The five vendor adapters keep the public handle surface of the previous single-module
//! layout — an infallible `new(&ProviderConfig)`, the `config`/`kind` accessors, a
//! redacting `Debug`, and a [`MemoryProvider`] impl —
//! while moving the vendor-specific HTTP shape into one file per vendor.
//!
//! # Two backends
//!
//! - **Real REST** (`mem0`, `jiuwen_memory`, `lakebase`, `openviking`) — each file
//!   defines a private `WireSpec` with the vendor's static shape (id, display name,
//!   missing dependency, capability flags, default auth scheme, static headers, fallback
//!   scope) and wraps it in a shared [`RestProvider`](super::rest::RestProvider). The
//!   transport owns validation, authentication, retries, failover, status mapping, and
//!   cancellation, so the vendor files stay declarative.
//! - **SDK-only** (`agentarts`) — the Python adapter drives
//!   `agentarts.sdk.memory.MemoryClient` and never speaks raw HTTP, so this build has no
//!   transport to fall back on: every operation is a typed
//!   [`openjiuwen_common::error::Error::Unsupported`] naming the missing SDK, whether or
//!   not an `endpoint` is configured.
//!
//! # Endpoint-less handles
//!
//! Construction never fails. A REST handle without an `endpoint` (and without
//! `options["failover_endpoints"]`) is inert: [`MemoryProvider::capabilities`] reports
//! nothing and every operation returns the same typed `Unsupported` error naming the
//! vendor's missing dependency and stating that no `endpoint` is configured. That keeps
//! [`ProviderRegistry::with_vendor_stubs`](super::registry::ProviderRegistry::with_vendor_stubs)
//! able to build a provider from a configuration document that has not been filled in
//! yet, mirroring the lazy "missing SDK surfaces at first use" behaviour of the Python
//! adapters.
//!
//! The modules are private; every handle is re-exported from this module, so callers keep
//! using `openjiuwen_memory::external::{Mem0MemoryProvider, …}` unchanged.

/// Generates the [`MemoryProvider`] impl of one adapter.
///
/// The twelve core methods forward to `self.$field` unchanged, so the adapter inherits the
/// backend's semantics (the REST transport's validation, cancellation, and error typing,
/// or the SDK-only helper's single typed `Unsupported`). The caller's extra items —
/// `$($rail:item)*` — are spliced into the same impl block, so a vendor file supplies only
/// its overrides and never repeats the forwarding boilerplate. Rail overrides are written
/// exactly like trait methods (they belong to the trait), so the list may be empty while
/// the rail surface is still being added to the trait.
macro_rules! implement_vendor_provider {
    ($adapter:ident, $field:ident; $($rail:item)*) => {
        #[async_trait::async_trait]
        impl $crate::external::provider::MemoryProvider for $adapter {
            fn name(&self) -> &str {
                self.$field.name()
            }

            fn capabilities(&self) -> $crate::external::provider::ProviderCapabilities {
                self.$field.capabilities()
            }

            async fn initialize(&self) -> openjiuwen_common::error::Result<()> {
                self.$field.initialize().await
            }

            async fn add(
                &self,
                scope: &$crate::record::MemoryScope,
                memories: Vec<$crate::record::NewMemory>,
                options: &$crate::record::AddOptions,
            ) -> openjiuwen_common::error::Result<$crate::record::AddOutcome> {
                self.$field.add(scope, memories, options).await
            }

            async fn search(
                &self,
                scope: &$crate::record::MemoryScope,
                query: &str,
                options: &$crate::record::SearchOptions,
            ) -> openjiuwen_common::error::Result<Vec<$crate::record::RecallHit>> {
                self.$field.search(scope, query, options).await
            }

            async fn get(
                &self,
                scope: &$crate::record::MemoryScope,
                id: &str,
            ) -> openjiuwen_common::error::Result<Option<$crate::record::MemoryRecord>> {
                self.$field.get(scope, id).await
            }

            async fn update(
                &self,
                scope: &$crate::record::MemoryScope,
                id: &str,
                update: $crate::record::MemoryUpdate,
                options: &$crate::record::CallOptions,
            ) -> openjiuwen_common::error::Result<$crate::record::MemoryRecord> {
                self.$field.update(scope, id, update, options).await
            }

            async fn delete(
                &self,
                scope: &$crate::record::MemoryScope,
                id: &str,
            ) -> openjiuwen_common::error::Result<bool> {
                self.$field.delete(scope, id).await
            }

            async fn list(
                &self,
                filter: &$crate::record::MemoryFilter,
            ) -> openjiuwen_common::error::Result<Vec<$crate::record::MemoryRecord>> {
                self.$field.list(filter).await
            }

            async fn clear(
                &self,
                filter: &$crate::record::MemoryFilter,
            ) -> openjiuwen_common::error::Result<usize> {
                self.$field.clear(filter).await
            }

            async fn health(
                &self,
            ) -> openjiuwen_common::error::Result<$crate::external::provider::ProviderHealth>
            {
                self.$field.health().await
            }

            $($rail)*
        }
    };
}

mod agentarts;
mod jiuwen;
mod lakebase;
mod mem0;
mod openviking;

pub use agentarts::AgentArtsMemoryProvider;
pub use jiuwen::JiuwenMemoryProvider;
pub use lakebase::LakeBaseMemoryProvider;
pub use mem0::Mem0MemoryProvider;
pub use openviking::OpenVikingMemoryProvider;

use std::sync::Arc;

use openjiuwen_common::error::Error;
use serde_json::Value;

use crate::record::MemoryScope;

use super::provider::MemoryProvider;
use super::registry::{ProviderConfig, ProviderFactory, ProviderKind};

/// Reads one trimmed, non-blank string option.
///
/// Vendor specs share this accessor so `options["user_id"]`-style keys are read
/// identically everywhere; a blank or non-string value is treated as absent.
pub(super) fn option_text(config: &ProviderConfig, key: &str) -> Option<String> {
    config
        .option(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// The fallback scope a vendor uses when a call does not name one.
///
/// Reads `options["user_id"]`, `options["scope_id"]`, `options["session_id"]`, and
/// `options["agent_id"]`; an absent or blank required id falls back to
/// `MemoryScope::default` ([`DEFAULT_ID`](crate::record::DEFAULT_ID)).
pub(super) fn scope_from_options(config: &ProviderConfig) -> MemoryScope {
    let fallback = MemoryScope::default();
    let mut scope = MemoryScope::new(
        option_text(config, "user_id").unwrap_or(fallback.user_id),
        option_text(config, "scope_id").unwrap_or(fallback.scope_id),
    );
    if let Some(session_id) = option_text(config, "session_id") {
        scope = scope.session(session_id);
    }
    if let Some(agent_id) = option_text(config, "agent_id") {
        scope = scope.agent(agent_id);
    }
    scope
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

    fn build(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>, Error> {
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
/// [`ProviderRegistry::with_vendor_stubs`](super::registry::ProviderRegistry::with_vendor_stubs)
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
    use openjiuwen_common::error::Result;
    use serde_json::json;

    use super::super::provider::ProviderCapabilities;
    use super::super::registry::ProviderRegistry;
    use super::*;
    use crate::record::{
        AddOptions, CallOptions, MemoryFilter, MemoryUpdate, NewMemory, SearchOptions,
    };

    /// The credential every redaction assertion looks for.
    const SECRET: &str = "sk-vendor-secret";
    /// A configured endpoint no test ever connects to.
    const ENDPOINT: &str = "https://vendor.invalid";
    /// The vendor set: kind, provider id, and the dependency token the errors must name.
    const VENDORS: [(ProviderKind, &str, &str); 5] = [
        (
            ProviderKind::AgentArts,
            "agentarts",
            "the `agentarts-sdk` package (`MemoryClient`) and its HTTP transport",
        ),
        (
            ProviderKind::Jiuwen,
            "jiuwen_memory",
            "the `jiuwen_memory` SDK and its HTTP server transport",
        ),
        (
            ProviderKind::LakeBase,
            "lakebase",
            "the LakeBase REST transport (`httpx.AsyncClient`)",
        ),
        (
            ProviderKind::Mem0,
            "mem0",
            "the `mem0ai` SDK (`AsyncMemoryClient`)",
        ),
        (
            ProviderKind::OpenViking,
            "openviking",
            "the OpenViking API client",
        ),
    ];

    /// A configuration for `kind` with a credential and no endpoint.
    fn config(kind: ProviderKind) -> ProviderConfig {
        ProviderConfig::new(kind).with_api_key(SECRET)
    }

    /// Asserts that `result` is a typed `Unsupported` naming `token`, the vendor, and the
    /// absence of the dependency.
    fn assert_unsupported<T>(result: Result<T>, token: &str, id: &str) {
        let error = result.err().expect("the vendor handle must fail");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(token), "`{message}` must name `{token}`");
        assert!(
            message.contains("does not provide"),
            "`{message}` must state the dependency is absent"
        );
        assert!(
            message.to_lowercase().contains(&id.to_lowercase()),
            "`{message}` must name the vendor `{id}`"
        );
    }

    /// Runs every SPI operation of a vendor handle and asserts each one is `Unsupported`.
    async fn assert_vendor_surface_unsupported(
        provider: &dyn MemoryProvider,
        id: &str,
        token: &str,
    ) {
        let scope = MemoryScope::new("alice", "ws");
        let filter = MemoryFilter::for_scope(&scope);

        assert_unsupported(provider.initialize().await, token, id);
        assert_unsupported(
            provider
                .add(
                    &scope,
                    vec![NewMemory::semantic("a vendor write")],
                    &AddOptions::default(),
                )
                .await,
            token,
            id,
        );
        assert_unsupported(
            provider
                .search(&scope, "a vendor query", &SearchOptions::default())
                .await,
            token,
            id,
        );
        assert_unsupported(
            provider.get(&scope, "0123456789abcdef01234567").await,
            token,
            id,
        );
        assert_unsupported(
            provider
                .update(
                    &scope,
                    "0123456789abcdef01234567",
                    MemoryUpdate::content("a vendor update"),
                    &CallOptions::new(),
                )
                .await,
            token,
            id,
        );
        assert_unsupported(
            provider.delete(&scope, "0123456789abcdef01234567").await,
            token,
            id,
        );
        assert_unsupported(provider.list(&filter).await, token, id);
        assert_unsupported(provider.clear(&filter).await, token, id);
        assert_unsupported(provider.health().await, token, id);
    }

    /// Asserts the redacting `Debug` keeps the identity and endpoint but drops the key.
    fn assert_debug_redacted(debug: &str, adapter: &str, id: &str, token: &str) {
        assert!(debug.contains(adapter), "{debug}");
        assert!(debug.contains(id), "{debug}");
        assert!(debug.contains(token), "{debug}");
        assert!(debug.contains(ENDPOINT), "{debug}");
        assert!(!debug.contains(SECRET), "{debug}");
    }

    #[tokio::test]
    async fn unconfigured_handles_are_inert_typed_stubs() {
        let registry = ProviderRegistry::with_vendor_stubs();
        for (kind, id, token) in VENDORS {
            let provider = registry
                .build(&ProviderConfig::new(kind))
                .unwrap_or_else(|error| panic!("{id} must build: {error}"));
            assert_eq!(provider.name(), id);
            assert_eq!(
                provider.capabilities(),
                ProviderCapabilities::none(),
                "{id} declares no capability without an endpoint"
            );
            assert_vendor_surface_unsupported(provider.as_ref(), id, token).await;
        }
    }

    #[test]
    fn configured_handles_report_their_spec_capabilities() {
        let mem0 = Mem0MemoryProvider::new(&config(ProviderKind::Mem0).with_endpoint(ENDPOINT));
        assert_eq!(mem0.capabilities(), mem0::SPEC.capabilities());
        assert_eq!(
            mem0.capabilities(),
            ProviderCapabilities {
                semantic_search: true,
                ..ProviderCapabilities::none()
            }
        );
        assert_eq!(mem0.spec().id(), "mem0");

        let jiuwen =
            JiuwenMemoryProvider::new(&config(ProviderKind::Jiuwen).with_endpoint(ENDPOINT));
        assert_eq!(jiuwen.capabilities(), jiuwen::SPEC.capabilities());
        assert!(jiuwen.capabilities().semantic_search);
        assert!(jiuwen.capabilities().message_history);
        assert!(!jiuwen.capabilities().pagination);

        let lakebase =
            LakeBaseMemoryProvider::new(&config(ProviderKind::LakeBase).with_endpoint(ENDPOINT));
        assert_eq!(lakebase.capabilities(), lakebase::SPEC.capabilities());
        assert!(lakebase.capabilities().pagination);
        assert!(!lakebase.capabilities().message_history);

        let openviking = OpenVikingMemoryProvider::new(
            &config(ProviderKind::OpenViking).with_endpoint(ENDPOINT),
        );
        assert_eq!(openviking.capabilities(), openviking::SPEC.capabilities());
        assert_eq!(
            openviking.capabilities(),
            ProviderCapabilities {
                semantic_search: true,
                ..ProviderCapabilities::none()
            }
        );

        // The SDK-only adapter never declares a capability, configured or not.
        let agentarts =
            AgentArtsMemoryProvider::new(&config(ProviderKind::AgentArts).with_endpoint(ENDPOINT));
        assert_eq!(agentarts.capabilities(), ProviderCapabilities::none());
    }

    #[test]
    fn configured_handles_order_their_endpoints_for_failover() {
        let config = ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint("https://primary.invalid/")
            .with_option(
                "failover_endpoints",
                json!(["https://secondary.invalid", " https://primary.invalid ", ""]),
            );
        let handle = JiuwenMemoryProvider::new(&config);
        assert_eq!(
            handle.inner.endpoints().to_vec(),
            vec![
                "https://primary.invalid".to_owned(),
                "https://secondary.invalid".to_owned(),
            ]
        );
    }

    #[test]
    fn adapter_debug_reports_identity_but_not_the_credential() {
        let configured = |kind| config(kind).with_endpoint(ENDPOINT);
        assert_debug_redacted(
            &format!(
                "{:?}",
                Mem0MemoryProvider::new(&configured(ProviderKind::Mem0))
            ),
            "Mem0MemoryProvider",
            "mem0",
            "mem0ai",
        );
        assert_debug_redacted(
            &format!(
                "{:?}",
                JiuwenMemoryProvider::new(&configured(ProviderKind::Jiuwen))
            ),
            "JiuwenMemoryProvider",
            "jiuwen_memory",
            "jiuwen_memory",
        );
        assert_debug_redacted(
            &format!(
                "{:?}",
                LakeBaseMemoryProvider::new(&configured(ProviderKind::LakeBase))
            ),
            "LakeBaseMemoryProvider",
            "lakebase",
            "httpx.AsyncClient",
        );
        assert_debug_redacted(
            &format!(
                "{:?}",
                OpenVikingMemoryProvider::new(&configured(ProviderKind::OpenViking))
            ),
            "OpenVikingMemoryProvider",
            "openviking",
            "OpenViking API client",
        );
        assert_debug_redacted(
            &format!(
                "{:?}",
                AgentArtsMemoryProvider::new(&configured(ProviderKind::AgentArts))
            ),
            "AgentArtsMemoryProvider",
            "agentarts",
            "agentarts-sdk",
        );
    }

    #[tokio::test]
    async fn agentarts_stays_unsupported_even_with_an_endpoint() {
        let handle =
            AgentArtsMemoryProvider::new(&config(ProviderKind::AgentArts).with_endpoint(ENDPOINT));
        assert_eq!(handle.name(), "agentarts");
        assert_eq!(handle.kind(), ProviderKind::AgentArts);
        let scope = MemoryScope::new("alice", "ws");
        let error = handle
            .search(&scope, "anything", &SearchOptions::default())
            .await;
        assert_unsupported(error, VENDORS[0].2, "agentarts");
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
