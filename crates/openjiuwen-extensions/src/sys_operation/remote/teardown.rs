//! Container teardown for the remote sandbox providers.
//!
//! Python parity: `delete_jiuwenbox_sandbox` and `delete_yuanrong_sandbox` in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/`. Both pop the
//! matching entries out of the process-wide container cache and then release
//! the containers one by one, treating "already gone" as success and logging —
//! not raising — a release failure.
//!
//! # Why a selector
//!
//! The kernel never calls a provider when a handle is released, so a host tears
//! containers down itself. [`SandboxSelector`] names what to release: one
//! scope (`shared_key` = `"{base_url}|{isolation_key}"`), one container id, every
//! cache entry of one base URL, or the whole process cache (`delete_all`).
//! Selection is deterministic (the cache is a `BTreeMap`, so entries are ordered
//! by key), and the report keeps that order.
//!
//! # What teardown can and cannot do
//!
//! - `jiuwenbox` deletes through `DELETE /api/v1/sandboxes/{id}`;
//! - `yuanrong` deletes through `DELETE /api/environment/v1/environments/{id}`;
//! - `aio` has **no** container-delete API — the AIO gateway owns the container
//!   lifetime and reaps idle containers — so [`aio_teardown_error`] is a typed
//!   `Error::Unsupported` naming the gateway instead of an invented endpoint.
//!
//! A backend instance that already resolved a container keeps its own copy of
//! the id, exactly as Python's per-instance `_sandbox_id` does: after a teardown
//! the next operation sees the container gone and recreates it (`JiuwenBox`
//! and `YuanRong` both already handle that).

use openjiuwen_common::error::{Error, Result};

use super::transport::{normalize_base_url, SandboxHttp};

/// Separates the base URL from the isolation key in a cache key.
pub(crate) const KEY_SEPARATOR: char = '|';

/// The reason a teardown records when the caller gives none (Python's default).
const DEFAULT_REASON: &str = "teardown";

/// Why the `aio` sandbox type cannot be torn down by this crate.
pub(crate) const AIO_TEARDOWN_UNSUPPORTED: &str =
    "the aio sandbox type has no container-delete API: \
     the AIO gateway owns the container lifetime and reaps idle containers, so this crate caches \
     nothing it could release; tear down jiuwenbox or yuanrong instead";

/// Builds the cache key both container caches use: `"{base_url}|{isolation_key}"`.
///
/// The base URL is normalized the way [`SandboxHttp`] normalizes it, so a
/// selector built from user configuration addresses the entry a backend wrote.
#[must_use]
pub(crate) fn shared_key(base_url: &str, isolation_key: Option<&str>) -> String {
    format!(
        "{}{KEY_SEPARATOR}{}",
        normalize_base_url(base_url),
        isolation_key.unwrap_or_default()
    )
}

/// The base-URL half of a cache key.
fn key_base_url(key: &str) -> &str {
    key.split(KEY_SEPARATOR).next().unwrap_or_default()
}

/// Which cached remote containers a teardown releases.
///
/// The fields are checked in Python's precedence order: `delete_all` first, then
/// `shared_key`, then `base_url`; `sandbox_id` selects by container identity for
/// the providers that can delete an id without a cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxSelector {
    /// Scope key of one cached container: `"{base_url}|{isolation_key}"`.
    pub shared_key: Option<String>,
    /// Remote container id (`sandbox_id` / `environmentId`).
    pub sandbox_id: Option<String>,
    /// Base URL whose cached containers should all be released.
    pub base_url: Option<String>,
    /// Drain the whole process-wide cache of the provider.
    pub delete_all: bool,
    /// Reason recorded in the teardown logs (Python's `reason` argument).
    pub reason: String,
}

impl Default for SandboxSelector {
    fn default() -> Self {
        Self {
            shared_key: None,
            sandbox_id: None,
            base_url: None,
            delete_all: false,
            reason: DEFAULT_REASON.to_string(),
        }
    }
}

impl SandboxSelector {
    /// A selector that drains the whole process-wide cache.
    #[must_use]
    pub fn all() -> Self {
        Self {
            delete_all: true,
            ..Self::default()
        }
    }

    /// A selector for one scope: `base_url` plus an optional isolation key.
    ///
    /// The base URL is normalized (`http://` added, trailing slashes stripped),
    /// so `SandboxSelector::scope("127.0.0.1:8080/", Some("agent"))` addresses
    /// the entry a backend created for `http://127.0.0.1:8080`.
    #[must_use]
    pub fn scope(base_url: &str, isolation_key: Option<&str>) -> Self {
        Self {
            shared_key: Some(shared_key(base_url, isolation_key)),
            ..Self::default()
        }
    }

    /// A selector for one cache key already in the `"{base_url}|{isolation_key}"` form.
    #[must_use]
    pub fn from_shared_key(key: impl Into<String>) -> Self {
        Self {
            shared_key: Some(key.into()),
            ..Self::default()
        }
    }

    /// A selector for one remote container id.
    #[must_use]
    pub fn from_sandbox_id(id: impl Into<String>) -> Self {
        Self {
            sandbox_id: Some(id.into()),
            ..Self::default()
        }
    }

    /// A selector for every cached container of one base URL.
    #[must_use]
    pub fn for_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: Some(base_url.into()),
            ..Self::default()
        }
    }

    /// Records `reason` in the teardown logs.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Whether the selector names nothing at all.
    #[must_use]
    pub fn selects_nothing(&self) -> bool {
        !self.delete_all
            && self.shared_key.is_none()
            && self.sandbox_id.is_none()
            && self.base_url.is_none()
    }

    /// Whether a cache entry of this key is selected by the key-based criteria.
    ///
    /// `sandbox_id` is not consulted here: it matches a cache *value*, which only
    /// the provider can see.
    #[must_use]
    pub(crate) fn matches_key(&self, key: &str) -> bool {
        if self.delete_all {
            return true;
        }
        if let Some(expected) = self.shared_key.as_deref() {
            return key == expected;
        }
        match self.base_url_filter() {
            Some(base_url) => key_base_url(key) == base_url,
            None => false,
        }
    }

    /// The normalized base URL filter, when one is configured.
    #[must_use]
    pub(crate) fn base_url_filter(&self) -> Option<String> {
        self.base_url
            .as_deref()
            .map(str::trim)
            .filter(|base_url| !base_url.is_empty())
            .map(normalize_base_url)
    }
}

/// One container a teardown failed to release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeardownFailure {
    /// Cache key of the entry.
    pub key: String,
    /// Remote container id the release targeted.
    pub container_id: String,
    /// Why the release failed, as the transport rendered it.
    pub reason: String,
}

/// What a teardown released.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TeardownReport {
    /// The sandbox type the report describes (`"jiuwenbox"`, `"yuanrong"`, …).
    pub sandbox_type: String,
    /// Cache keys of the released entries, in the order they were released.
    pub released_keys: Vec<String>,
    /// Remote container ids of the released entries, in the same order
    /// (Python's `sandbox_ids` / `environment` ids).
    pub deleted: Vec<String>,
    /// Entries whose release failed; they are already out of the cache.
    pub failed: Vec<TeardownFailure>,
    /// Why the sandbox type cannot be torn down at all, when that is the case.
    pub unsupported: Option<String>,
}

impl TeardownReport {
    /// An empty report for `sandbox_type`.
    #[must_use]
    pub fn new(sandbox_type: impl Into<String>) -> Self {
        Self {
            sandbox_type: sandbox_type.into(),
            ..Self::default()
        }
    }

    /// A report for a sandbox type that has no teardown path.
    #[must_use]
    pub fn not_supported(sandbox_type: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            unsupported: Some(reason.into()),
            ..Self::new(sandbox_type)
        }
    }

    /// Records one released container.
    pub(crate) fn record_released(&mut self, key: String, container_id: String) {
        self.released_keys.push(key);
        self.deleted.push(container_id);
    }

    /// Records one container whose release failed.
    pub(crate) fn record_failure(&mut self, key: String, container_id: String, reason: String) {
        self.failed.push(TeardownFailure {
            key,
            container_id,
            reason,
        });
    }

    /// Whether the report describes no work at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.released_keys.is_empty() && self.failed.is_empty() && self.unsupported.is_none()
    }
}

/// The typed rejection of [`super::teardown`] for the `aio` sandbox type.
pub(crate) fn aio_teardown_error() -> Error {
    Error::unsupported(AIO_TEARDOWN_UNSUPPORTED)
}

/// Rejects a selector that names nothing to release.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// naming the four ways to select containers.
pub(crate) fn require_selection(provider: &str, selector: &SandboxSelector) -> Result<()> {
    if selector.selects_nothing() {
        return Err(Error::validation(format!(
            "{provider} teardown needs a shared key, a sandbox id, a base URL, or delete_all"
        )));
    }
    Ok(())
}

/// Issues one container-delete request and interprets the answer.
///
/// `2xx` and `404` mean the container is gone (Python treats both as success);
/// any other status and any transport failure become the reason of a
/// [`TeardownFailure`], so one unreachable container never aborts a teardown.
pub(crate) async fn send_delete(
    http: &SandboxHttp,
    request: reqwest::RequestBuilder,
    operation: &str,
) -> std::result::Result<(), String> {
    match http.send_raw(request, operation).await {
        Ok(response) if response.is_success() || response.status == 404 => Ok(()),
        Ok(response) => {
            let detail = response.text();
            let detail = detail.trim();
            Err(if detail.is_empty() {
                format!("HTTP {} for {operation}", response.status)
            } else {
                format!("HTTP {} for {operation}: {detail}", response.status)
            })
        }
        Err(error) => Err(error.message()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn a_scope_selector_normalizes_the_base_url() {
        let selector = SandboxSelector::scope("127.0.0.1:8080/", Some("agent"));
        assert_eq!(
            selector.shared_key.as_deref(),
            Some("http://127.0.0.1:8080|agent")
        );
        assert!(selector.matches_key("http://127.0.0.1:8080|agent"));
        assert!(!selector.matches_key("http://127.0.0.1:8080|other"));
        assert!(!selector.matches_key("http://127.0.0.1:9090|agent"));
    }

    #[test]
    fn a_scope_without_an_isolation_key_ends_in_the_separator() {
        let selector = SandboxSelector::scope("http://host:1", None);
        assert_eq!(selector.shared_key.as_deref(), Some("http://host:1|"));
        assert!(selector.matches_key("http://host:1|"));
        assert!(!selector.matches_key("http://host:1|agent"));
    }

    #[test]
    fn a_base_url_selector_matches_every_scope_of_that_host() {
        let selector = SandboxSelector::for_base_url("host:1/");
        assert!(selector.matches_key("http://host:1|a"));
        assert!(selector.matches_key("http://host:1|"));
        assert!(!selector.matches_key("http://host:2|a"));
    }

    #[test]
    fn delete_all_matches_every_entry() {
        let selector = SandboxSelector::all();
        assert!(selector.matches_key("http://any|anything"));
        assert!(!selector.selects_nothing());
    }

    #[test]
    fn a_shared_key_wins_over_a_base_url() {
        let selector = SandboxSelector {
            shared_key: Some("http://host:1|a".to_string()),
            base_url: Some("http://host:1".to_string()),
            ..SandboxSelector::default()
        };
        assert!(selector.matches_key("http://host:1|a"));
        assert!(!selector.matches_key("http://host:1|b"));
    }

    #[test]
    fn a_sandbox_id_selector_matches_no_key() {
        let selector = SandboxSelector::from_sandbox_id("sb-1");
        assert!(!selector.selects_nothing());
        assert!(!selector.matches_key("http://host:1|"));
    }

    #[test]
    fn the_default_selector_selects_nothing_and_records_the_python_reason() {
        let selector = SandboxSelector::default();
        assert!(selector.selects_nothing());
        assert_eq!(selector.reason, "teardown");
        assert_eq!(
            SandboxSelector::default().with_reason("shutdown").reason,
            "shutdown"
        );
    }

    #[test]
    fn an_empty_selector_is_a_validation_error() {
        let error = require_selection("jiuwenbox", &SandboxSelector::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("jiuwenbox"), "{}", error.message());
        assert!(require_selection("jiuwenbox", &SandboxSelector::all()).is_ok());
    }

    #[test]
    fn the_aio_rejection_names_the_gateway() {
        let error = aio_teardown_error();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_string();
        assert!(message.contains("aio"), "{message}");
        assert!(message.contains("gateway"), "{message}");
        assert!(message.contains("no container-delete API"), "{message}");
    }

    #[test]
    fn a_report_records_releases_and_failures_in_order() {
        let mut report = TeardownReport::new("jiuwenbox");
        assert!(report.is_empty());
        report.record_released("http://host:1|a".to_string(), "sb-1".to_string());
        report.record_failure(
            "http://host:1|b".to_string(),
            "sb-2".to_string(),
            "HTTP 500 for delete_sandbox".to_string(),
        );
        assert!(!report.is_empty());
        assert_eq!(report.released_keys, vec!["http://host:1|a"]);
        assert_eq!(report.deleted, vec!["sb-1"]);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].container_id, "sb-2");
        assert_eq!(
            report.failed[0].reason,
            "HTTP 500 for delete_sandbox".to_string()
        );

        let unsupported = TeardownReport::not_supported("aio", AIO_TEARDOWN_UNSUPPORTED);
        assert!(!unsupported.is_empty());
        assert_eq!(unsupported.sandbox_type, "aio");
        assert_eq!(unsupported.deleted, Vec::<String>::new());
    }
}
