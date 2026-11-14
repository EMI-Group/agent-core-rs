//! The `Jiuwen` memory server/SDK adapter (`jiuwen_memory`).
//!
//! The Python adapter speaks the Jiuwen `/v1/*` HTTP API (`httpx.AsyncClient`) or imports
//! the in-process `jiuwen_memory` kernel; this Rust adapter speaks the same record surface
//! over the shared [`RestProvider`](super::super::rest::RestProvider). Only the static
//! shape of the vendor is described here — paths, bodies, and response decoding are added
//! to [`JiuwenSpec`]'s `*_call`/`*_result` methods.
use std::fmt;

use super::super::provider::ProviderCapabilities;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{AuthScheme, RestProvider, WireSpec};
use crate::record::MemoryScope;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "jiuwen_memory";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "Jiuwen";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the `jiuwen_memory` SDK and its HTTP server transport";

/// The static HTTP shape of the Jiuwen memory server.
pub(super) struct JiuwenSpec;

impl WireSpec for JiuwenSpec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// The Jiuwen server keeps session message history alongside semantic recall; it has
    /// no keyword mode, graph, variables, or page-addressable listing.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            message_history: true,
            ..ProviderCapabilities::none()
        }
    }

    /// The Jiuwen server takes a bearer token.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Bearer
    }

    /// The transport supplies the JSON content type and `Accept` headers; Jiuwen adds no
    /// static vendor header of its own.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        Vec::new()
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::scope_from_options(config)
    }
}

/// The one [`WireSpec`] instance every Jiuwen adapter shares.
pub(super) static SPEC: &dyn WireSpec = &JiuwenSpec;

/// Handle for the `Jiuwen` memory server or SDK (`jiuwen_memory`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct JiuwenMemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl JiuwenMemoryProvider {
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
        ProviderKind::Jiuwen
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }
}

impl fmt::Debug for JiuwenMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JiuwenMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

implement_vendor_provider!(JiuwenMemoryProvider, inner;);
