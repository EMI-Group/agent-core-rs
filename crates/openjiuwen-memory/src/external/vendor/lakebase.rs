//! The `LakeBase`/`DBay` memory-service adapter (`lakebase`).
//!
//! The Python adapter speaks the `LakeBase` REST API over `httpx.AsyncClient`; this Rust
//! adapter speaks the same record surface through the shared
//! [`RestProvider`](super::super::rest::RestProvider). Only the static shape of the vendor
//! is described here — paths, bodies, and response decoding are added to
//! [`LakeBaseSpec`]'s `*_call`/`*_result` methods.
use std::fmt;

use super::super::provider::ProviderCapabilities;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{AuthScheme, RestProvider, WireSpec};
use crate::record::MemoryScope;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "lakebase";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "LakeBase";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the LakeBase REST transport (`httpx.AsyncClient`)";

/// The static HTTP shape of the `LakeBase` memory service.
pub(super) struct LakeBaseSpec;

impl WireSpec for LakeBaseSpec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// `LakeBase` recalls semantically and pages its listings; it has no keyword mode,
    /// graph, variables, or message history.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            pagination: true,
            ..ProviderCapabilities::none()
        }
    }

    /// The `LakeBase` API takes a bearer token.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Bearer
    }

    /// The transport supplies the JSON content type and `Accept` headers; `LakeBase` adds no
    /// static vendor header of its own.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        Vec::new()
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::scope_from_options(config)
    }
}

/// The one [`WireSpec`] instance every `LakeBase` adapter shares.
pub(super) static SPEC: &dyn WireSpec = &LakeBaseSpec;

/// Handle for the `LakeBase`/`DBay` memory service (`lakebase`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct LakeBaseMemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl LakeBaseMemoryProvider {
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
        ProviderKind::LakeBase
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }
}

impl fmt::Debug for LakeBaseMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LakeBaseMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

implement_vendor_provider!(LakeBaseMemoryProvider, inner;);
