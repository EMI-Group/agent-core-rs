//! The `OpenViking` context-database adapter (`openviking`).
//!
//! The Python adapter drives the `OpenViking` REST API through a sync `httpx` client; this
//! Rust adapter speaks the same record surface through the shared
//! [`RestProvider`](super::super::rest::RestProvider). Only the static shape of the vendor
//! is described here — paths, bodies, and response decoding are added to
//! [`OpenVikingSpec`]'s `*_call`/`*_result` methods.
use std::fmt;

use super::super::provider::ProviderCapabilities;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{AuthScheme, RestProvider, WireSpec};
use crate::record::MemoryScope;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "openviking";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "OpenViking";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the OpenViking API client";
/// Account sent when `options["account"]` names none.
const DEFAULT_ACCOUNT: &str = "default";
/// Agent sent when `options["agent_id"]` names none.
const DEFAULT_AGENT: &str = "hermes";

/// The static HTTP shape of the `OpenViking` context database.
pub(super) struct OpenVikingSpec;

impl WireSpec for OpenVikingSpec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// `OpenViking` performs semantic recall only; it has no keyword mode, graph, variables,
    /// message history, or page-addressable listing.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            ..ProviderCapabilities::none()
        }
    }

    /// `OpenViking` authenticates with a raw API key in `X-API-Key`.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Header {
            name: "X-API-Key".to_owned(),
        }
    }

    /// `OpenViking` routes a request by account, user, and agent headers.
    ///
    /// The account falls back to `default`, the user to the configured
    /// `options["user_id"]` (or the default scope's user), and the agent to `hermes` when
    /// `options["agent_id"]` names none.
    fn headers(&self, config: &ProviderConfig) -> Vec<(String, String)> {
        let account =
            super::option_text(config, "account").unwrap_or_else(|| DEFAULT_ACCOUNT.to_owned());
        let user = super::option_text(config, "user_id")
            .unwrap_or_else(|| super::scope_from_options(config).user_id);
        let agent =
            super::option_text(config, "agent_id").unwrap_or_else(|| DEFAULT_AGENT.to_owned());
        vec![
            ("X-OpenViking-Account".to_owned(), account),
            ("X-OpenViking-User".to_owned(), user),
            ("X-OpenViking-Agent".to_owned(), agent),
        ]
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::scope_from_options(config)
    }
}

/// The one [`WireSpec`] instance every `OpenViking` adapter shares.
pub(super) static SPEC: &dyn WireSpec = &OpenVikingSpec;

/// Handle for the `OpenViking` context database (`openviking`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct OpenVikingMemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl OpenVikingMemoryProvider {
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
        ProviderKind::OpenViking
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }
}

impl fmt::Debug for OpenVikingMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenVikingMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

implement_vendor_provider!(OpenVikingMemoryProvider, inner;);
