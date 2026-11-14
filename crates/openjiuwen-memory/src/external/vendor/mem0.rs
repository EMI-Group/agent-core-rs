//! The `mem0` hosted memory-service adapter (`mem0`).
//!
//! The Python adapter drives `mem0ai`'s `AsyncMemoryClient`; this Rust adapter speaks the
//! same record surface over the mem0 platform's HTTP API through the shared
//! [`RestProvider`](super::super::rest::RestProvider). Only the static shape of the
//! vendor is described here — paths, bodies, and response decoding are added to
//! [`Mem0Spec`]'s `*_call`/`*_result` methods.
use std::fmt;

use super::super::provider::ProviderCapabilities;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{AuthScheme, RestProvider, WireSpec};
use crate::record::MemoryScope;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "mem0";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "mem0";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the `mem0ai` SDK (`AsyncMemoryClient`)";

/// The static HTTP shape of the mem0 memory service.
pub(super) struct Mem0Spec;

impl WireSpec for Mem0Spec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// mem0 recalls semantically only; it has no keyword mode, graph, variables, or
    /// message history, and its listing is not page-addressable.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            ..ProviderCapabilities::none()
        }
    }

    /// The mem0 platform authenticates with `Authorization: Token <api_key>`.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Token
    }

    /// mem0 has no static vendor headers beyond the transport's JSON ones.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        Vec::new()
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::scope_from_options(config)
    }
}

/// The one [`WireSpec`] instance every mem0 adapter shares.
pub(super) static SPEC: &dyn WireSpec = &Mem0Spec;

/// Handle for the `mem0` hosted memory service (`mem0`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct Mem0MemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl Mem0MemoryProvider {
    /// Builds the adapter handle for `config`.
    #[must_use]
    pub fn new(config: &ProviderConfig) -> Self {
        Self {
            inner: RestProvider::from_config(SPEC, config),
        }
    }

    /// The configuration this handle was built from.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        self.inner.config()
    }

    /// The kind this adapter implements.
    #[must_use]
    pub fn kind(&self) -> ProviderKind {
        ProviderKind::Mem0
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }
}

impl fmt::Debug for Mem0MemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mem0MemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

implement_vendor_provider!(Mem0MemoryProvider, inner;);
