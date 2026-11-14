//! The reusable REST transport that every vendor memory-provider adapter is built on.
//!
//! A vendor adapter is two things: a [`WireSpec`] — the static description of one
//! vendor's HTTP shape (paths, auth default, JSON encoding/decoding) — and a
//! [`RestProvider`], the [`MemoryProvider`] that runs
//! that spec over the shared [`HttpClient`]:
//! validation, cancellation/deadline handling, retries, failover, status mapping, and
//! error typing all live here, once.
//!
//! # Requests
//!
//! A spec never sees a base URL. It returns a [`RestCall`] — method, path, query, extra
//! headers, optional JSON body — relative to the endpoint the provider will use, and the
//! provider resolves it against one of [`RestProvider::endpoints`] (the configured
//! `endpoint` first, then `options["failover_endpoints"]`). That is what makes failover
//! possible without a spec knowing anything about it.
//!
//! # Authentication
//!
//! [`AuthScheme`] is the vendor default (for example
//! [`AuthScheme::Bearer`] for mem0) resolved with the configuration's `options["auth"]`
//! override by [`resolve_auth`]:
//!
//! | `options["auth"]` | Effect |
//! | --- | --- |
//! | `none` | no credential is attached |
//! | `bearer` | `Authorization: Bearer <api_key>` |
//! | `token` | `Authorization: Token <api_key>` |
//! | `header` | raw `api_key` in `Authorization` (name from `options["auth_header"]`) |
//! | `query` | raw `api_key` in the `api_key` query parameter (name from `options["auth_query_param"]`) |
//!
//! An unknown or blank value falls back to the vendor default instead of failing, and an
//! absent or blank `api_key` attaches nothing at all.
//!
//! # Writes, retries, and failover
//!
//! [`RestProvider::add`](super::provider::MemoryProvider::add) sends **one request per
//! memory**, because every documented vendor API (`mem0`, jiuwen `/v1/add`, `LakeBase`
//! ingest, `OpenViking`) accepts a single content per call; the spec's
//! `add_call`/`add_result` pair therefore always sees a one-element slice, and the
//! provider concatenates the outcomes. `Add` is the only non-idempotent operation: it is
//! attempted once per endpoint, while every other operation retries a transport failure
//! or a retryable status (`429`, `5xx` by default) up to `options["max_retries"]` times
//! (`2` by default, backoff from `options["retry_backoff_ms"]`, capped at 5 s).
//!
//! A definitive HTTP answer (`400`, `401`, `403`, `404`, `409`, `422`, …) is the vendor's
//! verdict and never fails over; a transport failure or an exhausted retryable status
//! moves to the next endpoint, and the first `2xx` wins. When every endpoint fails, the
//! returned error names the last URL and endpoint count.
//!
//! # Health
//!
//! A spec that implements `health_call` is probed with it: `2xx` is healthy, `401`/`403`,
//! any other status, and a transport failure are reported as degraded with the reason
//! attached (a configured vendor never fails
//! [`health`](super::provider::MemoryProvider::health); `Error::Unsupported` is reserved
//! for a provider with no endpoint at all). A spec without a health call is probed with
//! its own `list` on the sentinel scope `("__health__", "__health__")`.
//!
//! # Unconfigured handles
//!
//! [`RestProvider::from_config`] never fails: a handle without an `endpoint` is a valid,
//! inert handle whose every method returns the same typed
//! [`Error::Unsupported`] naming the
//! vendor's missing dependency and stating that no `endpoint` is configured. That keeps
//! a registry able to build a provider from a configuration document that has not been
//! filled in yet, and keeps the "missing SDK surfaces at first use" behaviour of the
//! Python adapters.
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchMode, SearchOptions,
};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{
    HttpClient, HttpClientConfig, HttpRequest, HttpResponse, Method, RetryPolicy,
};
use serde::Serialize;
use serde_json::Value;

use super::provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
use super::registry::ProviderConfig;

pub use super::rest_json::{
    body_value, error_detail, field_f32, field_f64, field_i64, field_str, field_value,
    parse_timestamp, results_array, scope_id, truncate_detail, user_id, vendor_hit, vendor_record,
};

/// `tracing` target of every event the transport emits.
const LOG_TARGET: &str = "openjiuwen_memory::external::rest";
/// Per-request timeout used when the configuration names none.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Retries after the first attempt when `options["max_retries"]` is absent.
const DEFAULT_MAX_RETRIES: u32 = 2;
/// Backoff before the first retry when `options["retry_backoff_ms"]` is absent.
const DEFAULT_RETRY_BACKOFF: Duration = Duration::from_millis(200);
/// Upper bound on the exponential retry backoff.
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(5);
/// Number of response-body characters quoted in an error message.
const BODY_SNIPPET_CHARS: usize = 512;
/// Vendor-level header a spec adds to every request.
const JSON_CONTENT_TYPE: &str = "application/json";
/// `User-Agent` sent with every request.
const USER_AGENT: &str = concat!("openjiuwen-memory/", env!("CARGO_PKG_VERSION"));
/// Scope used for the list-based health probe, mirroring the local provider.
const HEALTH_SCOPE: &str = "__health__";

/// One vendor request, relative to whichever base URL the provider is currently using.
///
/// The path is intentionally base-less: the same call is replayed against a failover
/// endpoint unchanged. Query entries and headers are ordered lists so a spec keeps full
/// control; the provider appends authentication and, for a JSON body, serializes it on
/// every attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct RestCall {
    /// HTTP verb, e.g. `Method::POST`.
    pub method: Method,
    /// Path relative to the endpoint (`"/items"` and `"items"` are equivalent).
    pub path: String,
    /// Query parameters appended in order.
    pub query: Vec<(String, String)>,
    /// Extra headers for this call only; the provider's vendor headers are added too.
    pub headers: Vec<(String, String)>,
    /// Structured body; serialized into a JSON payload when the request is sent.
    pub body: Option<Value>,
}

impl RestCall {
    /// Builds a call with `method` and `path` and nothing else.
    #[must_use]
    pub fn new(method: Method, path: impl Into<String>) -> Self {
        Self {
            method,
            path: path.into(),
            query: Vec::new(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Builds a `GET` call.
    #[must_use]
    pub fn get(path: impl Into<String>) -> Self {
        Self::new(Method::GET, path)
    }

    /// Builds a `POST` call.
    #[must_use]
    pub fn post(path: impl Into<String>) -> Self {
        Self::new(Method::POST, path)
    }

    /// Builds a `PUT` call.
    #[must_use]
    pub fn put(path: impl Into<String>) -> Self {
        Self::new(Method::PUT, path)
    }

    /// Builds a `PATCH` call.
    #[must_use]
    pub fn patch(path: impl Into<String>) -> Self {
        Self::new(Method::PATCH, path)
    }

    /// Builds a `DELETE` call.
    #[must_use]
    pub fn delete(path: impl Into<String>) -> Self {
        Self::new(Method::DELETE, path)
    }

    /// Appends one query parameter.
    #[must_use]
    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    /// Appends several query parameters.
    #[must_use]
    pub fn queries(mut self, params: impl IntoIterator<Item = (String, String)>) -> Self {
        self.query.extend(params);
        self
    }

    /// Appends one header for this call.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets the JSON body of this call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value cannot be serialized.
    pub fn json(mut self, body: &impl Serialize) -> Result<Self> {
        self.body = Some(serde_json::to_value(body).map_err(|error| {
            Error::validation(format!(
                "serializing a memory provider request body: {error}"
            ))
        })?);
        Ok(self)
    }

    /// Sets a body that is already a JSON value.
    #[must_use]
    pub fn with_body(mut self, body: Value) -> Self {
        self.body = Some(body);
        self
    }

    /// Joins `base` and `path` with exactly one `/`.
    ///
    /// A base may end with a `/` and a path may start with one; both are normalised. An
    /// already-absolute path is returned unchanged, and a blank base leaves the path as
    /// it is.
    #[must_use]
    pub fn absolute_url(base: &str, path: &str) -> String {
        let base = base.trim();
        let path = path.trim();
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_owned();
        }
        if base.is_empty() {
            return path.to_owned();
        }
        if path.is_empty() {
            return base.trim_end_matches('/').to_owned();
        }
        format!(
            "{}/{}",
            base.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}

/// How a vendor authenticates a request.
///
/// The variant is the vendor default or the caller's `options["auth"]` override; see
/// [`resolve_auth`]. The credential itself is never part of this value — it is read from
/// [`ProviderConfig::api_key`] at request time, so a scheme can be logged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AuthScheme {
    /// No credential is attached.
    #[default]
    None,
    /// `Authorization: Bearer <api_key>`.
    Bearer,
    /// `Authorization: Token <api_key>`.
    Token,
    /// The raw `api_key` in the named header.
    Header {
        /// Header name, for example `X-Api-Key`.
        name: String,
    },
    /// The raw `api_key` in the named query parameter.
    Query {
        /// Query parameter name, for example `api_key`.
        name: String,
    },
}

impl AuthScheme {
    /// The header this scheme injects for `api_key`, if any.
    #[must_use]
    pub fn header_for(&self, api_key: &str) -> Option<(String, String)> {
        if api_key.is_empty() {
            return None;
        }
        match self {
            Self::Bearer => Some(("Authorization".to_owned(), format!("Bearer {api_key}"))),
            Self::Token => Some(("Authorization".to_owned(), format!("Token {api_key}"))),
            Self::Header { name } => Some((name.clone(), api_key.to_owned())),
            Self::None | Self::Query { .. } => None,
        }
    }

    /// The query parameter this scheme injects for `api_key`, if any.
    #[must_use]
    pub fn query_for(&self, api_key: &str) -> Option<(String, String)> {
        if api_key.is_empty() {
            return None;
        }
        match self {
            Self::Query { name } => Some((name.clone(), api_key.to_owned())),
            Self::None | Self::Bearer | Self::Token | Self::Header { .. } => None,
        }
    }
}

/// Resolves the effective authentication scheme.
///
/// `default` is the vendor default; `config.option("auth")` overrides it
/// (`none`, `bearer`, `token`, `header`, `query`, ASCII-case-insensitive and trimmed).
/// `header` takes its name from `options["auth_header"]` (default `Authorization`) and
/// `query` from `options["auth_query_param"]` (default `api_key`). An unknown, blank, or
/// non-string `auth` value keeps `default`: a typo must never silently disable
/// authentication *and* never fail a config load.
#[must_use]
pub fn resolve_auth(default: AuthScheme, config: &ProviderConfig) -> AuthScheme {
    let Some(raw) = config.option("auth").and_then(Value::as_str) else {
        return default;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => AuthScheme::None,
        "bearer" => AuthScheme::Bearer,
        "token" => AuthScheme::Token,
        "header" => AuthScheme::Header {
            name: option_text(config, "auth_header").unwrap_or_else(|| "Authorization".to_owned()),
        },
        "query" => AuthScheme::Query {
            name: option_text(config, "auth_query_param").unwrap_or_else(|| "api_key".to_owned()),
        },
        _ => default,
    }
}

/// Reads one trimmed, non-blank string option.
fn option_text(config: &ProviderConfig, key: &str) -> Option<String> {
    config
        .option(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// A [`MemoryProvider`] operation.
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
/// retries, failover, status mapping, and error typing belong to [`RestProvider`].
///
/// Every `*_call` and `*_result` pair defaults to
/// [`Error::Unsupported`] built by
/// [`WireSpec::unsupported`], so a spec implements only what its vendor documents. Two
/// fallbacks in [`RestProvider`] cover the common gaps:
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

/// A [`MemoryProvider`] driven by a [`WireSpec`] over
/// HTTP.
///
/// The handle owns its configuration, its resolved [`AuthScheme`], and one shared
/// [`HttpClient`]. It is built by
/// [`RestProvider::from_config`], which never fails: a handle without an `endpoint` is
/// inert and reports the vendor's missing dependency from every method.
pub struct RestProvider {
    /// The vendor shape this provider speaks.
    spec: &'static dyn WireSpec,
    /// The configuration the handle was built from.
    config: ProviderConfig,
    /// The resolved authentication scheme.
    auth: AuthScheme,
    /// Primary endpoint first, then the normalised failover endpoints.
    endpoints: Vec<String>,
    /// The shared transport, absent only when the host's HTTP stack could not be built.
    client: Option<HttpClient>,
    /// Why `client` is absent, if it is.
    client_error: Option<String>,
    /// Whether [`MemoryProvider::initialize`] has probed the backend.
    initialized: AtomicBool,
}

impl RestProvider {
    /// Builds a provider for `spec` from `config`.
    ///
    /// Never fails: an absent or blank `endpoint` yields an inert handle whose every
    /// method reports the vendor's missing dependency, and a host HTTP stack that cannot
    /// be built is reported per call (see [`RestProvider::client`]).
    #[must_use]
    pub fn from_config(spec: &'static dyn WireSpec, config: &ProviderConfig) -> Self {
        let client_config = HttpClientConfig::default()
            .with_max_retries(0)
            .with_timeout(config.timeout.unwrap_or(DEFAULT_TIMEOUT))
            .with_user_agent(USER_AGENT);
        let (client, client_error) = match HttpClient::new(client_config) {
            Ok(client) => (Some(client), None),
            Err(error) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    vendor = spec.id(),
                    %error,
                    "the memory provider could not build its HTTP transport"
                );
                (None, Some(error.to_string()))
            }
        };
        Self {
            spec,
            config: config.clone(),
            auth: spec.auth(config),
            endpoints: collect_endpoints(config),
            client,
            client_error,
            initialized: AtomicBool::new(false),
        }
    }

    /// The vendor shape this provider speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.spec
    }

    /// The configuration the handle was built from.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// The resolved authentication scheme (never the credential itself).
    #[must_use]
    pub fn auth(&self) -> &AuthScheme {
        &self.auth
    }

    /// Whether an `endpoint` is configured, i.e. whether the handle can send anything.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        self.config
            .endpoint
            .as_deref()
            .is_some_and(|endpoint| !endpoint.trim().is_empty())
    }

    /// The endpoints in failover order: the configured `endpoint` first.
    ///
    /// `options["failover_endpoints"]` (an array of strings) adds the rest; blanks and
    /// duplicates are dropped and a trailing `/` is normalised away.
    #[must_use]
    pub fn endpoints(&self) -> &[String] {
        &self.endpoints
    }

    /// The shared transport.
    ///
    /// `None` only when the host environment's HTTP stack could not be built (a malformed
    /// proxy URL, for example); every call then returns that typed transport error, so a
    /// construction failure is never silently ignored.
    #[must_use]
    pub fn client(&self) -> Option<&HttpClient> {
        self.client.as_ref()
    }

    /// Returns the shared transport or the stored construction error.
    fn http(&self) -> Result<&HttpClient> {
        self.client.as_ref().ok_or_else(|| {
            let reason = self
                .client_error
                .as_deref()
                .unwrap_or("the HTTP transport is unavailable");
            Error::transport(format!(
                "the {} memory provider could not build its HTTP transport: {reason}",
                self.spec.display()
            ))
        })
    }

    /// Fails with the unconfigured error unless an endpoint is set.
    fn require_configured(&self) -> Result<()> {
        if self.is_configured() {
            Ok(())
        } else {
            Err(self.unconfigured_error())
        }
    }

    /// The single `Unsupported` error every method of an unconfigured handle returns.
    fn unconfigured_error(&self) -> Error {
        Error::unsupported(format!(
            "the {} memory provider requires {}, which this build does not provide: no `endpoint` is configured",
            self.spec.display(),
            self.spec.missing()
        ))
    }

    /// The retry schedule: `options["max_retries"]` and `options["retry_backoff_ms"]`
    /// over the documented defaults, without jitter so tests stay deterministic.
    fn retry_policy(&self) -> RetryPolicy {
        let max_retries = self
            .config
            .option("max_retries")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_MAX_RETRIES, |value| {
                u32::try_from(value).unwrap_or(u32::MAX)
            });
        let backoff = self
            .config
            .option("retry_backoff_ms")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_RETRY_BACKOFF, Duration::from_millis);
        RetryPolicy {
            max_retries,
            initial_backoff: backoff.min(MAX_RETRY_BACKOFF),
            max_backoff: MAX_RETRY_BACKOFF,
            retry_statuses: RetryPolicy::default().retry_statuses,
            jitter: false,
        }
    }

    /// Maps a definitive HTTP answer onto a typed error.
    ///
    /// `401`/`403` are security errors, `400`/`409`/`412`/`422` are validation errors
    /// carrying the vendor's own message, `404` is not-found, and everything else
    /// (including `429` and `5xx` after the retry budget) is a transport error. The body
    /// is quoted from the vendor's `error`/`message`/`detail` field when present, and is
    /// whitespace-collapsed and shortened otherwise.
    pub fn status_error(&self, op: Op, status: u16, url: &str, body: &str) -> Error {
        let vendor = self.spec.display();
        let detail =
            error_detail(body).unwrap_or_else(|| truncate_detail(body, BODY_SNIPPET_CHARS));
        match status {
            404 => Error::not_found(format!(
                "the {vendor} memory provider has no record for the {} request (HTTP 404 for {url})",
                op.as_str()
            )),
            401 | 403 => Error::security(format!(
                "the {vendor} memory provider rejected the credentials (HTTP {status} for {url})"
            )),
            400 | 409 | 412 | 422 => Error::validation(detailed(
                format!(
                    "the {vendor} memory provider rejected the {} request (HTTP {status} for {url})",
                    op.as_str()
                ),
                &detail,
            )),
            _ => Error::transport(detailed(
                format!(
                    "the {vendor} memory provider request for {} failed (HTTP {status} for {url})",
                    op.as_str()
                ),
                &detail,
            )),
        }
    }

    /// Builds the wire request for one call against `base`.
    fn request_for(
        &self,
        call: &RestCall,
        base: &str,
        options: &CallOptions,
    ) -> Result<HttpRequest> {
        let mut request = HttpRequest::new(
            call.method.clone(),
            RestCall::absolute_url(base, &call.path),
        )
        .queries(call.query.clone())
        .with_headers(self.spec.headers(&self.config))
        .with_headers(call.headers.clone())
        .cancel(options.cancellation.clone());
        let api_key = self.config.api_key.as_deref().map_or("", str::trim);
        if let Some((name, value)) = self.auth.header_for(api_key) {
            request = request.header(name, value);
        }
        if let Some((name, value)) = self.auth.query_for(api_key) {
            request = request.query(name, value);
        }
        if let Some(timeout) = options.timeout {
            request = request.timeout(timeout);
        }
        if let Some(body) = &call.body {
            request = request.json(body)?;
        }
        Ok(request)
    }

    /// Sleeps for the backoff before `attempt`, honouring cancellation.
    async fn backoff(
        &self,
        policy: &RetryPolicy,
        attempt: u32,
        options: &CallOptions,
    ) -> Result<()> {
        let delay = policy.backoff(attempt);
        tokio::select! {
            biased;
            () = options.cancellation.cancelled() => Err(Error::Cancelled),
            () = tokio::time::sleep(delay) => Ok(()),
        }
    }

    /// Runs `call` against every endpoint in turn, retrying idempotent operations.
    ///
    /// Returns the first `2xx` response, or the first definitive non-2xx answer (the
    /// caller maps it with [`RestProvider::status_error`]). A transport failure or an
    /// exhausted retryable status moves to the next endpoint; when none answers, the
    /// error names the last endpoint and how many were tried.
    async fn execute(
        &self,
        op: Op,
        call: &RestCall,
        options: &CallOptions,
    ) -> Result<HttpResponse> {
        options.check()?;
        let client = self.http()?;
        let policy = self.retry_policy();
        let mut failure: Option<(String, Error)> = None;

        for base in &self.endpoints {
            let url = RestCall::absolute_url(base, &call.path);
            let mut attempt = 0_u32;
            loop {
                let request = self.request_for(call, base, options)?;
                match client.execute(request).await {
                    Ok(response) => {
                        let status = response.status;
                        if response.is_success() {
                            return Ok(response);
                        }
                        if op.is_idempotent() && policy.should_retry(attempt, None, Some(status)) {
                            attempt = attempt.saturating_add(1);
                            tracing::warn!(
                                target: LOG_TARGET,
                                vendor = self.spec.id(),
                                op = op.as_str(),
                                status,
                                attempt,
                                "retrying the memory provider request"
                            );
                            self.backoff(&policy, attempt, options).await?;
                            continue;
                        }
                        if policy.is_retryable_status(status) {
                            tracing::warn!(
                                target: LOG_TARGET,
                                vendor = self.spec.id(),
                                op = op.as_str(),
                                status,
                                "the memory provider endpoint answered a transient status"
                            );
                            failure = Some((
                                url.clone(),
                                self.status_error(op, status, &url, &response.text()),
                            ));
                            break;
                        }
                        return Ok(response);
                    }
                    Err(error) => {
                        if error.is_cancelled() {
                            return Err(error);
                        }
                        if op.is_idempotent() && policy.should_retry(attempt, Some(&error), None) {
                            attempt = attempt.saturating_add(1);
                            tracing::warn!(
                                target: LOG_TARGET,
                                vendor = self.spec.id(),
                                op = op.as_str(),
                                attempt,
                                %error,
                                "retrying the memory provider request after a transport failure"
                            );
                            self.backoff(&policy, attempt, options).await?;
                            continue;
                        }
                        tracing::warn!(
                            target: LOG_TARGET,
                            vendor = self.spec.id(),
                            op = op.as_str(),
                            %url,
                            %error,
                            "the memory provider endpoint failed"
                        );
                        failure = Some((url.clone(), error));
                        break;
                    }
                }
            }
        }

        Err(match failure {
            Some((url, error)) => self.endpoint_failure(&url, self.endpoints.len(), error),
            None => Error::transport(format!(
                "the {} memory provider has no endpoint configured",
                self.spec.display()
            )),
        })
    }

    /// Wraps the last endpoint failure, keeping the transport/timeout kind.
    fn endpoint_failure(&self, url: &str, endpoints: usize, last: Error) -> Error {
        let message = format!(
            "the {} memory provider request to {url} failed after {endpoints} endpoint(s): {last}",
            self.spec.display()
        );
        if matches!(last, Error::Timeout(_)) {
            Error::timeout(message)
        } else {
            Error::transport(message)
        }
    }

    /// Turns a non-2xx answer into a typed error.
    fn ensure_success(&self, op: Op, response: &HttpResponse) -> Result<()> {
        if response.is_success() {
            Ok(())
        } else {
            Err(self.status_error(op, response.status, &response.url, &response.text()))
        }
    }

    /// Derives the vendor scope of a filter, falling back to the vendor default scope.
    fn scope_for(&self, filter: &MemoryFilter) -> MemoryScope {
        let fallback = self.spec.default_scope(&self.config);
        MemoryScope {
            user_id: filter.user_id.clone().unwrap_or(fallback.user_id),
            scope_id: filter.scope_id.clone().unwrap_or(fallback.scope_id),
            session_id: filter.session_id.clone().or(fallback.session_id),
            agent_id: filter.agent_id.clone().or(fallback.agent_id),
        }
    }

    /// Rejects a blank record id.
    fn require_id(id: &str) -> Result<&str> {
        if id.trim().is_empty() {
            Err(Error::validation("memory record `id` must not be blank"))
        } else {
            Ok(id)
        }
    }
}

impl fmt::Debug for RestProvider {
    /// Prints the vendor, endpoints, and state — never the configuration or credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestProvider")
            .field("spec", &self.spec.id())
            .field("endpoints", &self.endpoints)
            .field("auth", &self.auth)
            .field("config", &self.config)
            .field("configured", &self.is_configured())
            .field("client", &self.client.is_some())
            .field("client_error", &self.client_error)
            .field("initialized", &self.initialized.load(Ordering::Relaxed))
            .finish()
    }
}

/// Joins a message and its detail with `": "`, or returns the message alone.
fn detailed(message: String, detail: &str) -> String {
    if detail.is_empty() {
        message
    } else {
        format!("{message}: {detail}")
    }
}

/// Normalises one endpoint: trimmed, without a trailing `/`, or `None` when blank.
fn normalize_endpoint(endpoint: &str) -> Option<String> {
    let trimmed = endpoint.trim().trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// The endpoint list of a configuration: primary first, then the failover array.
fn collect_endpoints(config: &ProviderConfig) -> Vec<String> {
    let mut endpoints: Vec<String> = Vec::new();
    if let Some(primary) = config.endpoint.as_deref().and_then(normalize_endpoint) {
        endpoints.push(primary);
    }
    if let Some(Value::Array(entries)) = config.option("failover_endpoints") {
        for entry in entries {
            if let Some(endpoint) = entry.as_str().and_then(normalize_endpoint) {
                if !endpoints.contains(&endpoint) {
                    endpoints.push(endpoint);
                }
            }
        }
    }
    endpoints
}

#[async_trait]
impl MemoryProvider for RestProvider {
    /// The configured name when set, otherwise the vendor id.
    fn name(&self) -> &str {
        if self.config.name.trim().is_empty() {
            self.spec.id()
        } else {
            self.config.provider_name()
        }
    }

    /// The spec's capabilities for a configured handle, none for an unconfigured one.
    fn capabilities(&self) -> ProviderCapabilities {
        if self.is_configured() {
            self.spec.capabilities()
        } else {
            ProviderCapabilities::none()
        }
    }

    /// Marks the handle initialized and, when the spec has a health call, probes it.
    ///
    /// Any answer — including `4xx`/`5xx` — is success, mirroring the Python adapters,
    /// which only warn; a transport failure is returned so a caller can tell "wrong
    /// credentials" from "nothing is listening". Idempotent: the second call does no I/O.
    async fn initialize(&self) -> Result<()> {
        self.require_configured()?;
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }
        if self.spec.supports(Op::Health) {
            let scope = self.spec.default_scope(&self.config);
            let cx = SpecContext::new(&self.config, &scope);
            if let Ok(call) = self.spec.health_call(&cx) {
                self.execute(Op::Health, &call, &CallOptions::default())
                    .await?;
            }
        }
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Writes one memory per request, merging the per-memory outcomes.
    ///
    /// Every documented vendor API takes a single content per call, so the spec's
    /// `add_call`/`add_result` pair always sees a one-element slice.
    async fn add(
        &self,
        scope: &MemoryScope,
        memories: Vec<NewMemory>,
        options: &AddOptions,
    ) -> Result<AddOutcome> {
        self.require_configured()?;
        scope.validate()?;
        options.call.check()?;
        if memories.is_empty() {
            return Ok(AddOutcome::default());
        }
        let cx = SpecContext::new(&self.config, scope);
        let mut outcome = AddOutcome::default();
        for memory in &memories {
            memory.validate()?;
            options.call.check()?;
            let single = std::slice::from_ref(memory);
            let call = self.spec.add_call(&cx, single, options)?;
            let response = self.execute(Op::Add, &call, &options.call).await?;
            self.ensure_success(Op::Add, &response)?;
            let mut partial = self.spec.add_result(&cx, &response, single)?;
            outcome.added.append(&mut partial.added);
            outcome.merged.append(&mut partial.merged);
            outcome.deleted.append(&mut partial.deleted);
            outcome.skipped.append(&mut partial.skipped);
        }
        Ok(outcome)
    }

    /// Recalls memories, refusing a mode the spec does not declare.
    async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        self.require_configured()?;
        scope.validate()?;
        if query.trim().is_empty() {
            return Err(Error::validation(
                "memory search `query` must not be blank or whitespace",
            ));
        }
        options.call.check()?;
        let capabilities = self.spec.capabilities();
        match options.mode {
            SearchMode::Semantic if !capabilities.semantic_search => {
                return Err(Error::unsupported(format!(
                    "the {} memory provider has no semantic recall",
                    self.spec.display()
                )))
            }
            SearchMode::Keyword if !capabilities.keyword_search => {
                return Err(Error::unsupported(format!(
                    "the {} memory provider has no keyword recall",
                    self.spec.display()
                )))
            }
            SearchMode::Hybrid | SearchMode::Semantic | SearchMode::Keyword => {}
        }
        let cx = SpecContext::new(&self.config, scope);
        let call = self.spec.search_call(&cx, query, options)?;
        let response = self.execute(Op::Search, &call, &options.call).await?;
        self.ensure_success(Op::Search, &response)?;
        self.spec.search_result(&cx, &response, options)
    }

    /// Reads one record; a `404` is `Ok(None)`.
    async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
        self.require_configured()?;
        scope.validate()?;
        let id = Self::require_id(id)?;
        let cx = SpecContext::new(&self.config, scope);
        let call = self.spec.get_call(&cx, id)?;
        let response = self
            .execute(Op::Get, &call, &CallOptions::default())
            .await?;
        if response.status == 404 {
            return Ok(None);
        }
        self.ensure_success(Op::Get, &response)?;
        self.spec.get_result(&cx, &response, id)
    }

    /// Applies a patch; a `404` is `Error::NotFound`.
    async fn update(
        &self,
        scope: &MemoryScope,
        id: &str,
        update: MemoryUpdate,
        options: &CallOptions,
    ) -> Result<MemoryRecord> {
        self.require_configured()?;
        scope.validate()?;
        let id = Self::require_id(id)?;
        if update.is_empty() {
            return Err(Error::validation(
                "a memory update must change at least one field",
            ));
        }
        update.validate()?;
        options.check()?;
        let cx = SpecContext::new(&self.config, scope);
        let call = self.spec.update_call(&cx, id, &update)?;
        let response = self.execute(Op::Update, &call, options).await?;
        self.ensure_success(Op::Update, &response)?;
        self.spec.update_result(&cx, &response, id)
    }

    /// Deletes one record; a `404` is `Ok(false)`.
    async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool> {
        self.require_configured()?;
        scope.validate()?;
        let id = Self::require_id(id)?;
        let cx = SpecContext::new(&self.config, scope);
        let call = self.spec.delete_call(&cx, id)?;
        let response = self
            .execute(Op::Delete, &call, &CallOptions::default())
            .await?;
        if response.status == 404 {
            return Ok(false);
        }
        self.ensure_success(Op::Delete, &response)?;
        self.spec.delete_result(&cx, &response, id)
    }

    /// Lists the filter, using the filter's scope when it names one.
    async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        self.require_configured()?;
        let scope = self.scope_for(filter);
        scope.validate()?;
        let cx = SpecContext::new(&self.config, &scope);
        let call = self.spec.list_call(&cx, filter)?;
        let response = self
            .execute(Op::List, &call, &CallOptions::default())
            .await?;
        self.ensure_success(Op::List, &response)?;
        self.spec.list_result(&cx, &response, filter)
    }

    /// Removes every matching record.
    ///
    /// Without a `clear_call`, this lists the filter and deletes the ids one by one,
    /// counting the successes; a delete that answers `404` counts as "not removed" rather
    /// than as a failure.
    async fn clear(&self, filter: &MemoryFilter) -> Result<usize> {
        self.require_configured()?;
        let scope = self.scope_for(filter);
        scope.validate()?;
        let cx = SpecContext::new(&self.config, &scope);
        match self.spec.clear_call(&cx, filter) {
            Ok(call) => {
                let response = self
                    .execute(Op::Clear, &call, &CallOptions::default())
                    .await?;
                self.ensure_success(Op::Clear, &response)?;
                self.spec.clear_result(&cx, &response, filter)
            }
            Err(Error::Unsupported(_)) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    vendor = self.spec.id(),
                    "the memory provider has no bulk delete; clearing record by record"
                );
                let mut removed = 0;
                for record in self.list(filter).await? {
                    match self.delete(&scope, &record.id).await {
                        Ok(true) => removed += 1,
                        Ok(false) | Err(Error::NotFound(_)) => {}
                        Err(error) => return Err(error),
                    }
                }
                Ok(removed)
            }
            Err(error) => Err(error),
        }
    }

    /// Probes the backend, reporting a degraded handle instead of failing.
    async fn health(&self) -> Result<ProviderHealth> {
        self.require_configured()?;
        if self.spec.supports(Op::Health) {
            let scope = self.spec.default_scope(&self.config);
            let cx = SpecContext::new(&self.config, &scope);
            if let Ok(call) = self.spec.health_call(&cx) {
                return match self
                    .execute(Op::Health, &call, &CallOptions::default())
                    .await
                {
                    Ok(response) if response.is_success() => Ok(ProviderHealth::healthy()),
                    Ok(response) if response.status == 401 || response.status == 403 => {
                        Ok(ProviderHealth::degraded(format!(
                            "credentials rejected (HTTP {})",
                            response.status
                        )))
                    }
                    Ok(response) => Ok(ProviderHealth::degraded(format!(
                        "the {} memory provider answered HTTP {}",
                        self.spec.display(),
                        response.status
                    ))),
                    Err(error) if error.is_cancelled() => Err(error),
                    Err(error) => Ok(ProviderHealth::degraded(error.to_string())),
                };
            }
        }
        let probe = MemoryFilter::for_scope(&MemoryScope::new(HEALTH_SCOPE, HEALTH_SCOPE));
        match self.list(&probe).await {
            Ok(_) => Ok(ProviderHealth::healthy()),
            Err(error) if error.is_cancelled() => Err(error),
            Err(error) => Ok(ProviderHealth::degraded(error.to_string())),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::external::registry::ProviderKind;
    use crate::record::MemoryType;

    /// The vendor shape every test speaks: `/items` CRUD, `/search`, `/health`.
    struct TestSpec;

    impl WireSpec for TestSpec {
        fn id(&self) -> &'static str {
            "test"
        }

        fn display(&self) -> &'static str {
            "Test"
        }

        fn missing(&self) -> &'static str {
            "the `testspec` transport"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                semantic_search: true,
                keyword_search: true,
                ..ProviderCapabilities::none()
            }
        }

        fn default_auth(&self) -> AuthScheme {
            AuthScheme::Bearer
        }

        fn add_call(
            &self,
            _cx: &SpecContext<'_>,
            memories: &[NewMemory],
            _options: &AddOptions,
        ) -> Result<RestCall> {
            let memory = memories
                .first()
                .ok_or_else(|| Error::validation("the test spec needs one memory"))?;
            RestCall::post("/items").json(&json!({"memory": memory.content}))
        }

        fn add_result(
            &self,
            cx: &SpecContext<'_>,
            response: &HttpResponse,
            _memories: &[NewMemory],
        ) -> Result<AddOutcome> {
            let value = body_value(response)?;
            Ok(AddOutcome {
                added: vec![vendor_record(&value, cx.scope, MemoryType::SemanticMemory)],
                ..AddOutcome::default()
            })
        }

        fn search_call(
            &self,
            _cx: &SpecContext<'_>,
            query: &str,
            options: &SearchOptions,
        ) -> Result<RestCall> {
            RestCall::post("/search").json(&json!({"query": query, "top_k": options.top_k}))
        }

        fn search_result(
            &self,
            cx: &SpecContext<'_>,
            response: &HttpResponse,
            _options: &SearchOptions,
        ) -> Result<Vec<RecallHit>> {
            let value = body_value(response)?;
            Ok(results_array(&value)
                .iter()
                .map(|entry| {
                    let record = vendor_record(entry, cx.scope, MemoryType::SemanticMemory);
                    vendor_hit(record, field_f32(entry, &["score"]).unwrap_or(0.0))
                })
                .collect())
        }

        fn get_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
            Ok(RestCall::get(format!("/items/{id}")))
        }

        fn get_result(
            &self,
            cx: &SpecContext<'_>,
            response: &HttpResponse,
            _id: &str,
        ) -> Result<Option<MemoryRecord>> {
            let value = body_value(response)?;
            Ok(Some(vendor_record(
                &value,
                cx.scope,
                MemoryType::SemanticMemory,
            )))
        }

        fn update_call(
            &self,
            _cx: &SpecContext<'_>,
            id: &str,
            update: &MemoryUpdate,
        ) -> Result<RestCall> {
            RestCall::put(format!("/items/{id}")).json(&json!({"memory": update.content}))
        }

        fn update_result(
            &self,
            cx: &SpecContext<'_>,
            response: &HttpResponse,
            _id: &str,
        ) -> Result<MemoryRecord> {
            let value = body_value(response)?;
            Ok(vendor_record(&value, cx.scope, MemoryType::SemanticMemory))
        }

        fn delete_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
            Ok(RestCall::delete(format!("/items/{id}")))
        }

        fn delete_result(
            &self,
            _cx: &SpecContext<'_>,
            _response: &HttpResponse,
            _id: &str,
        ) -> Result<bool> {
            Ok(true)
        }

        fn list_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
            Ok(RestCall::get("/items"))
        }

        fn list_result(
            &self,
            cx: &SpecContext<'_>,
            response: &HttpResponse,
            _filter: &MemoryFilter,
        ) -> Result<Vec<MemoryRecord>> {
            let value = body_value(response)?;
            Ok(results_array(&value)
                .iter()
                .map(|entry| vendor_record(entry, cx.scope, MemoryType::SemanticMemory))
                .collect())
        }

        fn health_call(&self, _cx: &SpecContext<'_>) -> Result<RestCall> {
            Ok(RestCall::get("/health"))
        }
    }

    /// A spec with only a listing path and no health probe.
    struct ListOnlySpec;

    impl WireSpec for ListOnlySpec {
        fn id(&self) -> &'static str {
            "listonly"
        }

        fn display(&self) -> &'static str {
            "ListOnly"
        }

        fn missing(&self) -> &'static str {
            "the `listonly` transport"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::none()
        }

        fn default_auth(&self) -> AuthScheme {
            AuthScheme::None
        }

        fn list_call(&self, cx: &SpecContext<'_>, filter: &MemoryFilter) -> Result<RestCall> {
            SPEC.list_call(cx, filter)
        }

        fn list_result(
            &self,
            cx: &SpecContext<'_>,
            response: &HttpResponse,
            filter: &MemoryFilter,
        ) -> Result<Vec<MemoryRecord>> {
            SPEC.list_result(cx, response, filter)
        }
    }

    /// The one static spec every test shares.
    static SPEC: TestSpec = TestSpec;

    /// The listing-only spec.
    static LIST_ONLY: ListOnlySpec = ListOnlySpec;

    /// The static spec as the provider takes it.
    fn spec() -> &'static dyn WireSpec {
        &SPEC
    }

    /// A provider over `server`, with a fast retry backoff.
    fn provider(server: &MockServer) -> RestProvider {
        RestProvider::from_config(
            spec(),
            &ProviderConfig::new(ProviderKind::Mem0)
                .with_endpoint(format!("{}/", server.uri()))
                .with_api_key("test-key")
                .with_option("retry_backoff_ms", json!(1)),
        )
    }

    /// An unconfigured `mem0`-shaped handle.
    fn unconfigured() -> RestProvider {
        RestProvider::from_config(spec(), &ProviderConfig::new(ProviderKind::Mem0))
    }

    /// The scope every test writes under.
    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    /// Mounts one mock that answers every request with `status` and `body`.
    async fn mount(server: &MockServer, verb: &str, route: &str, response: ResponseTemplate) {
        let matcher = match verb {
            "GET" => Mock::given(method("GET")),
            "POST" => Mock::given(method("POST")),
            "PUT" => Mock::given(method("PUT")),
            "DELETE" => Mock::given(method("DELETE")),
            other => panic!("unsupported verb {other}"),
        };
        matcher
            .and(path(route))
            .respond_with(response)
            .mount(server)
            .await;
    }

    /// Mounts a `POST /search` mock answering `body`.
    async fn mount_search(server: &MockServer, body: serde_json::Value) {
        mount(
            server,
            "POST",
            "/search",
            ResponseTemplate::new(200).set_body_json(body),
        )
        .await;
    }

    #[test]
    fn absolute_url_joins_with_exactly_one_slash() {
        assert_eq!(
            RestCall::absolute_url("http://host/api/", "/items"),
            "http://host/api/items"
        );
        assert_eq!(
            RestCall::absolute_url("http://host/api", "items"),
            "http://host/api/items"
        );
        assert_eq!(
            RestCall::absolute_url("http://host", "items"),
            "http://host/items"
        );
        assert_eq!(RestCall::absolute_url("http://host/", ""), "http://host");
        assert_eq!(
            RestCall::absolute_url("", "/items"),
            "/items",
            "a blank base keeps the path"
        );
        assert_eq!(
            RestCall::absolute_url("http://host", "https://other/items"),
            "https://other/items"
        );
    }

    #[test]
    fn resolve_auth_reads_every_documented_option() {
        let config = ProviderConfig::new(ProviderKind::Mem0);
        assert_eq!(
            resolve_auth(AuthScheme::Bearer, &config),
            AuthScheme::Bearer
        );

        for (raw, expected) in [
            ("none", AuthScheme::None),
            (" None ", AuthScheme::None),
            ("BEARER", AuthScheme::Bearer),
            ("token", AuthScheme::Token),
            (
                "header",
                AuthScheme::Header {
                    name: "Authorization".to_owned(),
                },
            ),
            (
                "query",
                AuthScheme::Query {
                    name: "api_key".to_owned(),
                },
            ),
            ("nonsense", AuthScheme::Bearer),
            ("", AuthScheme::Bearer),
        ] {
            let config = config.clone().with_option("auth", json!(raw));
            assert_eq!(resolve_auth(AuthScheme::Bearer, &config), expected, "{raw}");
        }

        let named = config
            .clone()
            .with_option("auth", json!("header"))
            .with_option("auth_header", json!("X-Api-Key"));
        assert_eq!(
            resolve_auth(AuthScheme::None, &named),
            AuthScheme::Header {
                name: "X-Api-Key".to_owned()
            }
        );
        let queried = config
            .with_option("auth", json!("query"))
            .with_option("auth_query_param", json!("token"));
        assert_eq!(
            resolve_auth(AuthScheme::None, &queried),
            AuthScheme::Query {
                name: "token".to_owned()
            }
        );
    }

    #[test]
    fn auth_schemes_inject_the_credential_where_documented() {
        assert_eq!(
            AuthScheme::Bearer.header_for("key"),
            Some(("Authorization".to_owned(), "Bearer key".to_owned()))
        );
        assert_eq!(
            AuthScheme::Token.header_for("key"),
            Some(("Authorization".to_owned(), "Token key".to_owned()))
        );
        assert_eq!(
            AuthScheme::Header {
                name: "X-Api-Key".to_owned()
            }
            .header_for("key"),
            Some(("X-Api-Key".to_owned(), "key".to_owned()))
        );
        assert_eq!(
            AuthScheme::Query {
                name: "api_key".to_owned()
            }
            .query_for("key"),
            Some(("api_key".to_owned(), "key".to_owned()))
        );
        assert_eq!(AuthScheme::None.header_for("key"), None);
        assert_eq!(AuthScheme::Bearer.header_for(""), None);
        assert_eq!(
            AuthScheme::Query { name: "q".into() }.header_for("key"),
            None
        );
    }

    #[test]
    fn op_names_and_idempotence() {
        assert_eq!(Op::Add.as_str(), "add");
        assert_eq!(Op::Health.as_str(), "health");
        assert!(!Op::Add.is_idempotent());
        for op in [
            Op::Search,
            Op::Get,
            Op::Update,
            Op::Delete,
            Op::List,
            Op::Clear,
            Op::Health,
        ] {
            assert!(op.is_idempotent(), "{}", op.as_str());
        }
    }

    #[tokio::test]
    async fn an_unconfigured_handle_reports_the_missing_dependency_everywhere() {
        let provider = unconfigured();
        assert!(!provider.is_configured());
        assert_eq!(provider.endpoints(), [] as [String; 0]);
        assert!(provider.client().is_some());
        assert_eq!(provider.name(), "test");
        assert_eq!(provider.capabilities(), ProviderCapabilities::none());

        let scope = scope();
        let operations: Vec<Error> = vec![
            provider.initialize().await.expect_err("initialize"),
            provider
                .add(
                    &scope,
                    vec![NewMemory::semantic("a write")],
                    &AddOptions::default(),
                )
                .await
                .expect_err("add"),
            provider
                .search(&scope, "a query", &SearchOptions::default())
                .await
                .expect_err("search"),
            provider.get(&scope, "id-1").await.expect_err("get"),
            provider
                .update(
                    &scope,
                    "id-1",
                    MemoryUpdate::content("x"),
                    &CallOptions::new(),
                )
                .await
                .expect_err("update"),
            provider.delete(&scope, "id-1").await.expect_err("delete"),
            provider
                .list(&MemoryFilter::default())
                .await
                .expect_err("list"),
            provider
                .clear(&MemoryFilter::default())
                .await
                .expect_err("clear"),
            provider.health().await.expect_err("health"),
        ];
        for error in operations {
            assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
            let message = error.to_string();
            assert!(message.contains("testspec"), "{message}");
            assert!(message.contains("does not provide"), "{message}");
            assert!(message.contains("no `endpoint` is configured"), "{message}");
            assert!(message.contains("Test"), "{message}");
        }
        let _ = format!("{provider:?}");
    }

    #[tokio::test]
    async fn a_configured_handle_takes_its_capabilities_from_the_spec() {
        let server = MockServer::start().await;
        let provider = provider(&server);
        assert!(provider.is_configured());
        assert_eq!(provider.endpoints(), vec![server.uri()]);
        assert_eq!(
            provider.capabilities(),
            ProviderCapabilities {
                semantic_search: true,
                keyword_search: true,
                ..ProviderCapabilities::none()
            }
        );
        assert_eq!(provider.spec().id(), "test");
        assert_eq!(provider.auth(), &AuthScheme::Bearer);
        assert_eq!(provider.config().provider_name(), "mem0");
    }

    #[tokio::test]
    async fn add_posts_one_request_per_memory_with_the_bearer_credential() {
        let server = MockServer::start().await;
        mount(
            &server,
            "POST",
            "/items",
            ResponseTemplate::new(200).set_body_json(json!({"id": "mem-1", "memory": "hello"})),
        )
        .await;

        let outcome = provider(&server)
            .add(
                &scope(),
                vec![NewMemory::semantic("hello"), NewMemory::semantic("again")],
                &AddOptions::default(),
            )
            .await
            .expect("add");
        assert_eq!(outcome.added.len(), 2);
        assert_eq!(outcome.added[0].content, "hello");

        let received = server.received_requests().await.expect("requests");
        assert_eq!(received.len(), 2, "one request per memory");
        let first = &received[0];
        assert_eq!(first.method.as_str(), "POST");
        assert_eq!(first.url.path(), "/items");
        let body: serde_json::Value = serde_json::from_slice(&first.body).expect("body");
        assert_eq!(body, json!({"memory": "hello"}));
        assert_eq!(
            first
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        assert_eq!(
            first
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[tokio::test]
    async fn search_maps_results_ignoring_unknown_fields() {
        let server = MockServer::start().await;
        mount_search(
            &server,
            json!({"results": [{"id": "a", "memory": "hello", "score": 0.75, "extra": [1, 2]}]}),
        )
        .await;

        let hits = provider(&server)
            .search(&scope(), "hello", &SearchOptions::default())
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, "a");
        assert_eq!(hits[0].record.content, "hello");
        assert!((hits[0].score - 0.75).abs() < f32::EPSILON);

        let empty = MockServer::start().await;
        mount_search(&empty, json!({"results": []})).await;
        let hits = provider(&empty)
            .search(&scope(), "hello", &SearchOptions::default())
            .await
            .expect("empty search");
        assert_eq!(hits, [] as [RecallHit; 0]);
    }

    #[tokio::test]
    async fn a_missing_record_maps_to_none_false_and_not_found() {
        let server = MockServer::start().await;
        mount(&server, "GET", "/items/missing", ResponseTemplate::new(404)).await;
        mount(&server, "PUT", "/items/missing", ResponseTemplate::new(404)).await;
        mount(
            &server,
            "DELETE",
            "/items/missing",
            ResponseTemplate::new(404),
        )
        .await;
        let provider = provider(&server);

        assert_eq!(provider.get(&scope(), "missing").await.expect("get"), None);
        assert!(!provider.delete(&scope(), "missing").await.expect("delete"));
        let error = provider
            .update(
                &scope(),
                "missing",
                MemoryUpdate::content("changed"),
                &CallOptions::new(),
            )
            .await
            .expect_err("update");
        assert!(matches!(error, Error::NotFound(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_get_success_decodes_the_record() {
        let server = MockServer::start().await;
        mount(
            &server,
            "GET",
            "/items/a",
            ResponseTemplate::new(200)
                .set_body_json(json!({"id": "a", "memory": "hello", "created_at": 1_700_000_000})),
        )
        .await;
        let record = provider(&server)
            .get(&scope(), "a")
            .await
            .expect("get")
            .expect("some");
        assert_eq!(record.id, "a");
        assert_eq!(record.content, "hello");
        assert_eq!(record.scope, scope());
        assert_eq!(
            record.timestamp,
            chrono::DateTime::from_timestamp(1_700_000_000, 0).expect("timestamp")
        );
    }

    #[tokio::test]
    async fn status_answers_map_to_typed_errors() {
        let server = MockServer::start().await;
        mount(&server, "GET", "/items/denied", ResponseTemplate::new(401)).await;
        mount(
            &server,
            "GET",
            "/items/invalid",
            ResponseTemplate::new(422).set_body_json(json!({"error": "id must be a uuid"})),
        )
        .await;
        let provider = provider(&server);

        let error = provider.get(&scope(), "denied").await.expect_err("401");
        assert!(matches!(error, Error::Security(_)), "{error:?}");
        assert!(
            error.to_string().contains("rejected the credentials"),
            "{error}"
        );

        let error = provider.get(&scope(), "invalid").await.expect_err("422");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("id must be a uuid"), "{error}");
    }

    #[tokio::test]
    async fn a_transient_status_is_retried_once_then_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        mount(&server, "GET", "/health", ResponseTemplate::new(200)).await;

        let health = provider(&server).health().await.expect("health");
        assert!(health.healthy, "{health:?}");
        let received = server.received_requests().await.expect("requests");
        assert_eq!(received.len(), 2, "one retry");
    }

    #[tokio::test]
    async fn exhausted_retries_surface_as_a_transport_error() {
        let server = MockServer::start().await;
        mount(&server, "GET", "/health", ResponseTemplate::new(503)).await;
        mount(&server, "GET", "/items", ResponseTemplate::new(503)).await;
        let provider = RestProvider::from_config(
            spec(),
            &ProviderConfig::new(ProviderKind::Mem0)
                .with_endpoint(server.uri())
                .with_option("max_retries", json!(0))
                .with_option("retry_backoff_ms", json!(1)),
        );

        let health = provider
            .health()
            .await
            .expect("health is degraded, not failing");
        assert!(!health.healthy);
        assert!(
            health.detail.as_deref().unwrap_or_default().contains("503"),
            "{health:?}"
        );
        let list_error = provider
            .list(&MemoryFilter::default())
            .await
            .expect_err("list");
        assert!(matches!(list_error, Error::Transport(_)), "{list_error:?}");
        let received = server.received_requests().await.expect("requests");
        assert_eq!(received.len(), 2, "no retry with max_retries 0");
    }

    #[tokio::test]
    async fn a_failing_endpoint_fails_over_to_the_next() {
        let primary = MockServer::start().await;
        mount(&primary, "POST", "/search", ResponseTemplate::new(503)).await;
        let secondary = MockServer::start().await;
        mount_search(
            &secondary,
            json!({"results": [{"id": "a", "memory": "hello"}]}),
        )
        .await;

        let provider = RestProvider::from_config(
            spec(),
            &ProviderConfig::new(ProviderKind::Mem0)
                .with_endpoint(primary.uri())
                .with_option("failover_endpoints", json!([secondary.uri()]))
                .with_option("max_retries", json!(0))
                .with_option("retry_backoff_ms", json!(1)),
        );
        assert_eq!(provider.endpoints().len(), 2);

        let hits = provider
            .search(&scope(), "hello", &SearchOptions::default())
            .await
            .expect("failover search");
        assert_eq!(hits.len(), 1);
        assert_eq!(
            secondary.received_requests().await.expect("requests").len(),
            1
        );
    }

    #[tokio::test]
    async fn every_endpoint_failing_names_the_last_endpoint() {
        let primary = MockServer::start().await;
        mount(&primary, "GET", "/items", ResponseTemplate::new(503)).await;
        let secondary = MockServer::start().await;
        mount(&secondary, "GET", "/items", ResponseTemplate::new(503)).await;

        let provider = RestProvider::from_config(
            spec(),
            &ProviderConfig::new(ProviderKind::Mem0)
                .with_endpoint(primary.uri())
                .with_option("failover_endpoints", json!([secondary.uri()]))
                .with_option("max_retries", json!(0)),
        );
        let error = provider
            .list(&MemoryFilter::default())
            .await
            .expect_err("list");
        assert!(matches!(error, Error::Transport(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(&secondary.uri()), "{message}");
        assert!(message.contains("2 endpoint(s)"), "{message}");
    }

    #[tokio::test]
    async fn clear_without_a_bulk_delete_removes_record_by_record() {
        let server = MockServer::start().await;
        mount(
            &server,
            "GET",
            "/items",
            ResponseTemplate::new(200).set_body_json(json!({"items": [{"id": "a"}, {"id": "b"}]})),
        )
        .await;
        mount(&server, "DELETE", "/items/a", ResponseTemplate::new(200)).await;
        mount(&server, "DELETE", "/items/b", ResponseTemplate::new(404)).await;

        let removed = provider(&server)
            .clear(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("clear");
        assert_eq!(removed, 1, "a delete answering 404 is not counted");
    }

    #[tokio::test]
    async fn health_without_a_probe_falls_back_to_a_listing() {
        let server = MockServer::start().await;
        mount(
            &server,
            "GET",
            "/items",
            ResponseTemplate::new(200).set_body_json(json!({"items": []})),
        )
        .await;
        let provider = RestProvider::from_config(
            &LIST_ONLY,
            &ProviderConfig::new(ProviderKind::Custom("listonly".to_owned()))
                .with_endpoint(server.uri()),
        );

        let health = provider.health().await.expect("health");
        assert!(health.healthy, "{health:?}");
        let received = server.received_requests().await.expect("requests");
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].url.path(), "/items");
        assert_eq!(received[0].headers.get("authorization"), None);
    }

    #[tokio::test]
    async fn a_cancelled_call_sends_nothing() {
        let server = MockServer::start().await;
        mount_search(&server, json!({"results": []})).await;

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = SearchOptions::default().with_cancellation(cancellation);
        let error = provider(&server)
            .search(&scope(), "hello", &options)
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert!(server
            .received_requests()
            .await
            .expect("requests")
            .is_empty());
    }

    #[tokio::test]
    async fn validation_precedes_the_request() {
        let server = MockServer::start().await;
        let provider = provider(&server);

        assert!(matches!(
            provider.get(&scope(), "  ").await.expect_err("blank id"),
            Error::Validation(_)
        ));
        assert!(matches!(
            provider
                .search(&scope(), "   ", &SearchOptions::default())
                .await
                .expect_err("blank query"),
            Error::Validation(_)
        ));
        assert!(matches!(
            provider
                .update(
                    &scope(),
                    "id-1",
                    MemoryUpdate::default(),
                    &CallOptions::new()
                )
                .await
                .expect_err("empty update"),
            Error::Validation(_)
        ));
        assert!(server
            .received_requests()
            .await
            .expect("requests")
            .is_empty());
    }
}
