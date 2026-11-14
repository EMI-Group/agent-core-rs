//! [`Op`], [`SpecContext`], and the [`WireSpec`] trait: the per-vendor HTTP shape.
//!
//! A spec is a `&'static dyn WireSpec` value. Every shape method defaults to a typed
//! `Unsupported`, so a vendor implements only what its API documents; the provider in
//! [`super`] supplies validation, authentication, retries, failover, and status mapping.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;
use serde_json::Value;

use crate::record::{
    AddOptions, AddOutcome, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate, NewMemory,
    RecallHit, SearchOptions,
};

use super::super::provider::ProviderCapabilities;
use super::super::registry::ProviderConfig;
use super::auth::{resolve_auth, AuthScheme};
use super::call::RestCall;

/// Content type of every vendor request and of the bodies specs parse.
const JSON_CONTENT_TYPE: &str = "application/json";

/// A [`MemoryProvider`](super::super::provider::MemoryProvider) operation.
///
/// Every dispatch decision the transport makes — which side effect it may repeat, whether
/// a spec must implement it, which message an absent implementation produces — is keyed
/// on this enum instead of a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// Store one memory.
    Add,
    /// Recall memories matching a query.
    Search,
    /// Read one memory by id.
    Get,
    /// Modify one memory by id.
    Update,
    /// Remove one memory by id.
    Delete,
    /// List memories matching a filter.
    List,
    /// Remove every memory matching a filter.
    Clear,
    /// Probe the backend.
    Health,
}

impl Op {
    /// The lowercase name used in messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Search => "search",
            Self::Get => "get",
            Self::Update => "update",
            Self::Delete => "delete",
            Self::List => "list",
            Self::Clear => "clear",
            Self::Health => "health",
        }
    }

    /// Whether repeating the request is safe.
    ///
    /// Only [`Op::Add`] is not idempotent: a retried write could store the memory twice.
    #[must_use]
    pub const fn is_idempotent(self) -> bool {
        !matches!(self, Self::Add)
    }
}

/// Everything a spec method needs: the configuration and the scope of the call.
#[derive(Debug, Clone, Copy)]
pub struct SpecContext<'a> {
    /// The provider configuration (endpoint, options, per-request settings).
    pub config: &'a ProviderConfig,
    /// The scope the call belongs to.
    pub scope: &'a MemoryScope,
}

impl<'a> SpecContext<'a> {
    /// Builds a context from the configuration and the scope.
    #[must_use]
    pub const fn new(config: &'a ProviderConfig, scope: &'a MemoryScope) -> Self {
        Self { config, scope }
    }

    /// The user id of the scope.
    #[must_use]
    pub fn user_id(&self) -> &str {
        &self.scope.user_id
    }

    /// The scope id of the scope.
    #[must_use]
    pub fn scope_id(&self) -> &str {
        &self.scope.scope_id
    }

    /// The session id of the scope, when the scope is session-bound.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.scope.session_id.as_deref()
    }

    /// The agent id of the scope, when the scope is agent-bound.
    #[must_use]
    pub fn agent_id(&self) -> Option<&str> {
        self.scope.agent_id.as_deref()
    }

    /// Reads one vendor option.
    #[must_use]
    pub fn option(&self, key: &str) -> Option<&'a Value> {
        self.config.option(key)
    }
}

/// The static HTTP description of one vendor.
///
/// A spec is a `&'static dyn WireSpec` value: immutable, shared by every provider built
/// for that vendor, and free of per-deployment state. It describes *shapes* only —
/// request construction and response decoding — because validation, authentication,
/// retries, failover, status mapping, and error typing belong to [`RestProvider`](super::RestProvider).
///
/// Every `*_call` and `*_result` pair defaults to
/// [`Error::Unsupported`] built by
/// [`WireSpec::unsupported`], so a spec implements only what its vendor documents. Two
/// fallbacks in [`RestProvider`](super::RestProvider) cover the common gaps:
///
/// - Without `clear_call`, `clear` lists the filter and deletes each id one by one.
/// - Without `health_call`, `health` probes the vendor's `list` on the sentinel scope
///   `("__health__", "__health__")`.
pub trait WireSpec: Send + Sync {
    /// The registry id of the vendor (`"mem0"`, `"jiuwen_memory"`, …).
    fn id(&self) -> &'static str;

    /// The vendor name used in human-readable messages (`"mem0"`, `"LakeBase"`).
    fn display(&self) -> &'static str;

    /// The dependency token an unconfigured handle names (`"the `mem0ai` SDK"`).
    fn missing(&self) -> &'static str;

    /// The capabilities of a *configured* handle.
    fn capabilities(&self) -> ProviderCapabilities;

    /// The vendor default authentication scheme.
    fn default_auth(&self) -> AuthScheme;

    /// The effective authentication scheme, honouring `options["auth"]`.
    #[must_use]
    fn auth(&self, config: &ProviderConfig) -> AuthScheme {
        resolve_auth(self.default_auth(), config)
    }

    /// Static vendor headers sent with every request.
    ///
    /// Must not include the credential: the provider injects that from the resolved
    /// [`AuthScheme`], so a header list can be logged.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        vec![
            ("Content-Type".to_owned(), JSON_CONTENT_TYPE.to_owned()),
            ("Accept".to_owned(), JSON_CONTENT_TYPE.to_owned()),
        ]
    }

    /// The scope a vendor falls back to when a call does not name one.
    ///
    /// Vendors read `options["user_id"]`, `options["scope_id"]`, `options["agent_id"]`,
    /// and `options["session_id"]`; the default is [`MemoryScope::default`].
    fn default_scope(&self, _config: &ProviderConfig) -> MemoryScope {
        MemoryScope::default()
    }

    /// Whether the spec implements `op`.
    fn supports(&self, _op: Op) -> bool {
        true
    }

    /// The error reported by an operation the spec does not implement.
    fn unsupported(&self, op: Op) -> Error {
        Error::unsupported(format!(
            "the {} memory provider does not implement {} over HTTP",
            self.display(),
            op.as_str()
        ))
    }

    /// Builds the write request for one memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no write path, and
    /// [`Error::Validation`] when the input
    /// cannot be encoded.
    fn add_call(
        &self,
        _cx: &SpecContext<'_>,
        _memories: &[NewMemory],
        _options: &AddOptions,
    ) -> Result<RestCall> {
        Err(self.unsupported(Op::Add))
    }

    /// Decodes the write response.
    ///
    /// Called only for a `2xx` answer, with the single memory the request carried.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no write path, and a typed error when the body is unusable.
    fn add_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _memories: &[NewMemory],
    ) -> Result<AddOutcome> {
        Err(self.unsupported(Op::Add))
    }

    /// Builds the recall request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no recall path, and a typed error when the input cannot be encoded.
    fn search_call(
        &self,
        _cx: &SpecContext<'_>,
        _query: &str,
        _options: &SearchOptions,
    ) -> Result<RestCall> {
        Err(self.unsupported(Op::Search))
    }

    /// Decodes the recall response into scored hits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no recall path, and a typed error when the body is unusable.
    fn search_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        Err(self.unsupported(Op::Search))
    }

    /// Builds the read request for `id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no read path.
    fn get_call(&self, _cx: &SpecContext<'_>, _id: &str) -> Result<RestCall> {
        Err(self.unsupported(Op::Get))
    }

    /// Decodes the read response.
    ///
    /// A `404` never reaches this method: the provider turns it into `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no read path, and a typed error when the body is unusable.
    fn get_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _id: &str,
    ) -> Result<Option<MemoryRecord>> {
        Err(self.unsupported(Op::Get))
    }

    /// Builds the update request for `id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no update path.
    fn update_call(
        &self,
        _cx: &SpecContext<'_>,
        _id: &str,
        _update: &MemoryUpdate,
    ) -> Result<RestCall> {
        Err(self.unsupported(Op::Update))
    }

    /// Decodes the update response into the stored record.
    ///
    /// A `404` never reaches this method: the provider reports it as
    /// [`Error::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no update path, and a typed error when the body is unusable.
    fn update_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _id: &str,
    ) -> Result<MemoryRecord> {
        Err(self.unsupported(Op::Update))
    }

    /// Builds the delete request for `id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no delete path.
    fn delete_call(&self, _cx: &SpecContext<'_>, _id: &str) -> Result<RestCall> {
        Err(self.unsupported(Op::Delete))
    }

    /// Decodes the delete response.
    ///
    /// A `404` never reaches this method: the provider turns it into `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no delete path, and a typed error when the body is unusable.
    fn delete_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _id: &str,
    ) -> Result<bool> {
        Err(self.unsupported(Op::Delete))
    }

    /// Builds the listing request for `filter`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no listing path.
    fn list_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Err(self.unsupported(Op::List))
    }

    /// Decodes the listing response.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no listing path, and a typed error when the body is unusable.
    fn list_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _filter: &MemoryFilter,
    ) -> Result<Vec<MemoryRecord>> {
        Err(self.unsupported(Op::List))
    }

    /// Builds the bulk-delete request for `filter`.
    ///
    /// A spec that does not implement this keeps
    /// [`supports`](WireSpec::supports)`(Op::Clear) == false` semantics: the provider
    /// then lists the filter and deletes each id individually.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no bulk delete.
    fn clear_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Err(self.unsupported(Op::Clear))
    }

    /// Decodes the bulk-delete response into the number of removed memories.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no bulk delete, and a typed error when the body is unusable.
    fn clear_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _filter: &MemoryFilter,
    ) -> Result<usize> {
        Err(self.unsupported(Op::Clear))
    }

    /// Builds the health probe.
    ///
    /// A spec without a documented probe leaves this unimplemented; the provider then
    /// probes with `list` on the sentinel scope `("__health__", "__health__")`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when
    /// the vendor has no probe endpoint.
    fn health_call(&self, _cx: &SpecContext<'_>) -> Result<RestCall> {
        Err(self.unsupported(Op::Health))
    }
}
