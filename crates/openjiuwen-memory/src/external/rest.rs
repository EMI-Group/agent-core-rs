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
//! [`initialize`](super::provider::MemoryProvider::initialize) runs the same probe but
//! honours its verdict: a rejected credential (`401`/`403`) or any other non-`2xx` answer
//! is a typed error and the handle stays uninitialized, so "wrong api_key" can never be
//! mistaken for a healthy handle.
//!
//! # Pagination
//!
//! [`list`](super::provider::MemoryProvider::list) follows a vendor's own paging when the
//! spec implements `next_list_call`: it collects every page (bounded by
//! [`MAX_LIST_PAGES`]) before decoding, so `list` and the list-then-delete fallback of
//! [`clear`](super::provider::MemoryProvider::clear) see the whole collection instead of
//! only its first page. A spec that pages in one request leaves the hook unimplemented.
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

pub mod auth;
pub mod call;
pub mod wire;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

pub use auth::{resolve_auth, AuthScheme};
pub use call::RestCall;
pub use wire::{Op, SpecContext, WireSpec};

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{
    HttpClient, HttpClientConfig, HttpRequest, HttpResponse, RetryPolicy,
};
use serde_json::Value;

use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchMode, SearchOptions,
};

use super::provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
use super::registry::ProviderConfig;

pub use super::rest_json::{
    body_value, error_detail, field_f32, field_f64, field_i64, field_str, field_value,
    listing_record, parse_timestamp, rank_hits, results_array, results_array_checked, scope_id,
    truncate_detail, user_id, vendor_hit, vendor_record,
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
/// Upper bound on the pages one `list` follows, so a vendor that never stops paging
/// cannot loop forever; reaching it is a typed error, never a silent partial listing.
const MAX_LIST_PAGES: usize = 200;
/// Number of response-body characters quoted in an error message.
const BODY_SNIPPET_CHARS: usize = 512;
/// `User-Agent` sent with every request.
const USER_AGENT: &str = concat!("openjiuwen-memory/", env!("CARGO_PKG_VERSION"));
/// Scope used for the list-based health probe, mirroring the local provider.
const HEALTH_SCOPE: &str = "__health__";

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

    /// Maps a failed health probe onto a typed error.
    ///
    /// Unlike [`RestProvider::status_error`], this always names the *initialization* probe
    /// as the failure, because that is what the caller asked for: `401`/`403` become a
    /// [`Error::Security`] stating the credentials were rejected, and every other status a
    /// [`Error::Transport`] carrying the vendor's own message when it sent one.
    fn probe_error(&self, response: &HttpResponse) -> Error {
        let vendor = self.spec.display();
        let message = format!(
            "the {vendor} memory provider failed its initialization probe (HTTP {} for {})",
            response.status, response.url
        );
        if matches!(response.status, 401 | 403) {
            return Error::security(detailed(
                message,
                "the credentials were rejected; check `api_key` and the auth scheme",
            ));
        }
        let detail = error_detail(&response.text())
            .unwrap_or_else(|| truncate_detail(&response.text(), BODY_SNIPPET_CHARS));
        Error::transport(detailed(message, &detail))
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

    /// Probes the backend and marks the handle initialized only when the probe succeeds.
    ///
    /// A spec with a health call is probed with it; that probe is the only verification a
    /// handle can perform, so its verdict is honoured:
    ///
    /// | Probe answer | Result |
    /// | --- | --- |
    /// | any `2xx` | `Ok(())`, `initialized = true` |
    /// | `401`/`403` | [`Error::Security`] naming the rejected credentials |
    /// | any other non-`2xx` | [`Error::Transport`] naming the status and URL |
    /// | transport failure | that error, unwrapped |
    ///
    /// A handle that fails its probe stays uninitialized, so a later call retries it. A
    /// spec whose `health_call` reports [`Error::Unsupported`] has no probe to run and is
    /// simply marked initialized; any other error from building the probe is returned.
    /// Idempotent: once initialized, a call does no I/O.
    async fn initialize(&self) -> Result<()> {
        self.require_configured()?;
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }
        if self.spec.supports(Op::Health) {
            let scope = self.spec.default_scope(&self.config);
            let cx = SpecContext::new(&self.config, &scope);
            match self.spec.health_call(&cx) {
                Ok(call) => {
                    let response = self
                        .execute(Op::Health, &call, &CallOptions::default())
                        .await?;
                    if !response.is_success() {
                        let error = self.probe_error(&response);
                        self.initialized.store(false, Ordering::SeqCst);
                        tracing::warn!(
                            target: LOG_TARGET,
                            vendor = self.spec.id(),
                            status = response.status,
                            %error,
                            "the memory provider failed its initialization probe"
                        );
                        return Err(error);
                    }
                }
                Err(Error::Unsupported(_)) => {
                    tracing::debug!(
                        target: LOG_TARGET,
                        vendor = self.spec.id(),
                        "the memory provider documents no health probe; initializing unverified"
                    );
                }
                Err(error) => return Err(error),
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

    /// Lists every record matching the filter, following the vendor's pages.
    ///
    /// The spec's `list_call` asks for the first page; when it also implements
    /// `next_list_call`, each answer is followed by the next request until the vendor says
    /// the collection is exhausted, so a paged vendor (mem0) never reports only its first
    /// page. `MAX_LIST_PAGES` bounds the walk: a vendor that keeps paging forever is a
    /// typed transport error, never a truncated listing. A spec that answers a whole
    /// collection in one request (every other vendor here) leaves the hook unimplemented
    /// and is called exactly once.
    async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        self.require_configured()?;
        let scope = self.scope_for(filter);
        scope.validate()?;
        let cx = SpecContext::new(&self.config, &scope);
        let mut call = self.spec.list_call(&cx, filter)?;
        let mut records: Vec<MemoryRecord> = Vec::new();
        for page in 1..=MAX_LIST_PAGES {
            let response = self
                .execute(Op::List, &call, &CallOptions::default())
                .await?;
            self.ensure_success(Op::List, &response)?;
            records.extend(self.spec.list_result(&cx, &response, filter)?);
            match self.spec.next_list_call(&cx, filter, &response, page)? {
                Some(next) => call = next,
                None => return Ok(records),
            }
        }
        Err(Error::transport(format!(
            "the {} memory provider listing did not finish within {MAX_LIST_PAGES} pages",
            self.spec.display()
        )))
    }

    /// Removes every matching record.
    ///
    /// Without a `clear_call`, this lists the filter — following every page, so a paged
    /// vendor loses nothing beyond the first page — and deletes the ids one by one,
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
