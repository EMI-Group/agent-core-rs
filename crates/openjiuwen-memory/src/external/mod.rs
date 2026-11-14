//! External memory-provider SPI and registry.
//!
//! The in-process engine ([`crate::engine::MemoryEngine`]) is the default memory backend;
//! this module is the seam that lets a deployment delegate the same record surface to a
//! remote service, a vendor SDK, or any custom backend.
//!
//! - [`crate::external::provider`] — the [`crate::external::MemoryProvider`] trait,
//!   [`crate::external::ProviderCapabilities`], and [`crate::external::ProviderHealth`], with the full error and cancel-safety contract.
//! - [`crate::external::local`] — [`crate::external::EngineProvider`], the reference implementation
//!   over the shared engine, plus [`crate::external::LocalProviderFactory`].
//! - [`crate::external::registry`] — [`crate::external::ProviderKind`], [`crate::external::ProviderConfig`],
//!   [`crate::external::ProviderFactory`], and the application-scoped
//!   [`crate::external::ProviderRegistry`].
//! - [`crate::external::vendor`] — the `AgentArts`, `Jiuwen`, `LakeBase`, `mem0`, and
//!   `OpenViking` adapter handles.
//! - [`crate::external::rest`] — the reusable REST transport vendor adapters are built on:
//!   [`crate::external::WireSpec`], [`crate::external::RestCall`],
//!   [`crate::external::AuthScheme`], and [`crate::external::RestProvider`].
//! - [`crate::external::rest_json`] — the tolerant JSON accessors every
//!   [`crate::external::WireSpec`] uses to decode vendor bodies (re-exported by
//!   [`crate::external::rest`]).
//!
//! # Registry semantics
//!
//! A registry is one instance per application, injected as an `Arc`; it holds no
//! process-global state. Factories are keyed by [`ProviderKind`], live providers by name.
//! Registering the same kind or name twice **replaces** the previous entry and returns it
//! (last registration wins). The first registered provider becomes the default;
//! [`crate::external::ProviderRegistry::select`] builds a provider, registers it under
//! [`crate::external::ProviderConfig::provider_name`], and makes it the new default.
//! Unknown kind strings deserialize to [`crate::external::ProviderKind::Custom`] instead
//! of failing, so newer configuration
//! files keep working on older builds.
//!
//! # Vendor stubs
//!
//! [`crate::external::ProviderRegistry::with_vendor_stubs`] registers the five vendor
//! factories. Their
//! adapters construct successfully but every operation returns
//! [`openjiuwen_common::error::Error::Unsupported`] naming the missing SDK or transport
//! verbatim — no HTTP client, wire format, or vendor SDK is embedded in this crate.
//!
//! # Parity note
//!
//! The Python `openjiuwen.core.memory.external` SPI is tool-schema oriented
//! (`get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`,
//! `system_prompt_block`). The Rust SPI is record oriented (`add`, `search`, `get`,
//! `update`, `delete`, `list`, `clear`, `health`) because it adapts this crate's
//! `MemoryScope`/`MemoryRecord` types. The provider ids, the vendor set, and the lazy
//! "missing SDK surfaces at first use" behaviour are preserved; the tool rail and the
//! vendor transports remain deferred.

pub mod local;
pub mod provider;
pub mod registry;
pub mod rest;
pub mod rest_json;
pub mod vendor;

pub use local::{EngineProvider, LocalProviderFactory};
pub use provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
pub use registry::{ProviderConfig, ProviderFactory, ProviderKind, ProviderRegistry};
pub use rest::{resolve_auth, AuthScheme, Op, RestCall, RestProvider, SpecContext, WireSpec};
pub use vendor::{
    vendor_factories, AgentArtsMemoryProvider, JiuwenMemoryProvider, LakeBaseMemoryProvider,
    Mem0MemoryProvider, OpenVikingMemoryProvider,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    use crate::config::MemoryEngineConfig;
    use crate::engine::MemoryEngine;
    use crate::record::{AddOptions, MemoryScope, NewMemory, SearchOptions};
    use crate::store::MemoryStores;

    #[tokio::test]
    async fn a_registry_serves_the_local_engine_and_reports_vendor_stubs() {
        let engine = MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory())
            .expect("engine");
        let registry = ProviderRegistry::with_local(Arc::clone(&engine));

        let provider = registry
            .select(&ProviderConfig::new(ProviderKind::Local))
            .expect("select local");
        provider.initialize().await.expect("initialize");
        let scope = MemoryScope::new("alice", "ws");
        provider
            .add(
                &scope,
                vec![NewMemory::semantic("alice prefers dark mode")],
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let hits = provider
            .search(
                &scope,
                "dark mode",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(registry.names(), vec!["local"]);

        let vendor = registry
            .select(&ProviderConfig::new(ProviderKind::Mem0))
            .expect("the vendor factory builds a handle");
        assert_eq!(vendor.name(), "mem0");
        assert_eq!(registry.names(), vec!["local", "mem0"]);
        let error = vendor.initialize().await.expect_err("vendor stub");
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Unsupported(_)
        ));
        assert!(error.to_string().contains("mem0ai"), "{error}");
    }
}
