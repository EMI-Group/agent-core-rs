//! Configuration for the provider-agnostic HTTP client.
//!
//! [`HttpClientConfig`] carries transport settings only: timeouts, retry policy,
//! default headers, proxy, TLS verification, redirect handling, and a response
//! size cap. It deliberately has no provider-specific fields — authentication
//! headers are supplied per request through [`crate::http::HttpRequest`].
//!
//! Every timing/boolean field has an explicit serde default function so a partial
//! JSON object keeps the intended values instead of dropping to `Duration::ZERO`
//! or `false`.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::http::retry::RetryPolicy;

/// Default total timeout for one request (including the response body).
const fn default_timeout() -> Duration {
    Duration::from_secs(60)
}

/// Default first retry backoff.
const fn default_retry_initial_backoff() -> Duration {
    Duration::from_millis(200)
}

/// Default upper bound on the retry backoff.
const fn default_retry_max_backoff() -> Duration {
    Duration::from_secs(5)
}

/// HTTP statuses that are retried by default (rate limiting and server errors).
fn default_retry_statuses() -> Vec<u16> {
    vec![429, 500, 502, 503, 504]
}

/// TLS certificates are verified by default.
const fn default_verify_ssl() -> bool {
    true
}

/// Redirects are followed by default.
const fn default_follow_redirects() -> bool {
    true
}

/// Transport configuration for [`crate::http::HttpClient`].
///
/// The type is `serde`-serializable so callers can persist it alongside model
/// configuration and clone it into client instances.
///
/// ```
/// use std::time::Duration;
/// use openjiuwen_common::http::HttpClientConfig;
///
/// let config = HttpClientConfig::default()
///     .with_timeout(Duration::from_secs(30))
///     .with_max_retries(3)
///     .with_header("X-Client", "openjiuwen");
///
/// assert_eq!(config.timeout, Duration::from_secs(30));
/// assert_eq!(config.max_retries, 3);
/// assert_eq!(config.headers.get("X-Client").map(String::as_str), Some("openjiuwen"));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpClientConfig {
    /// Total timeout for a single request, including reading the response body.
    #[serde(default = "default_timeout")]
    pub timeout: Duration,
    /// Optional timeout for establishing the TCP/TLS connection.
    pub connect_timeout: Option<Duration>,
    /// Number of *additional* attempts after the first one; `0` disables retries.
    #[serde(default)]
    pub max_retries: u32,
    /// Backoff before the first retry.
    #[serde(default = "default_retry_initial_backoff")]
    pub retry_initial_backoff: Duration,
    /// Upper bound on the exponential retry backoff.
    #[serde(default = "default_retry_max_backoff")]
    pub retry_max_backoff: Duration,
    /// Statuses that trigger a retry.
    #[serde(default = "default_retry_statuses")]
    pub retry_statuses: Vec<u16>,
    /// Default headers sent with every request.
    ///
    /// Protected headers (notably `Authorization`) are dropped by
    /// [`crate::security::sanitize_headers`] before they reach the wire.
    pub headers: BTreeMap<String, String>,
    /// Explicit proxy URL; when unset, `http_proxy`/`https_proxy` are consulted.
    pub proxy: Option<String>,
    /// Verify TLS certificates.
    #[serde(default = "default_verify_ssl")]
    pub verify_ssl: bool,
    /// `User-Agent` sent with every request.
    pub user_agent: Option<String>,
    /// Reject responses whose body exceeds this many bytes.
    pub max_response_bytes: Option<u64>,
    /// Follow HTTP redirects (up to 10 hops).
    #[serde(default = "default_follow_redirects")]
    pub follow_redirects: bool,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            connect_timeout: None,
            max_retries: 0,
            retry_initial_backoff: default_retry_initial_backoff(),
            retry_max_backoff: default_retry_max_backoff(),
            retry_statuses: default_retry_statuses(),
            headers: BTreeMap::new(),
            proxy: None,
            verify_ssl: default_verify_ssl(),
            user_agent: None,
            max_response_bytes: None,
            follow_redirects: default_follow_redirects(),
        }
    }
}

impl HttpClientConfig {
    /// Sets the total request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the connection timeout; `None` leaves the client default.
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.connect_timeout = timeout.into();
        self
    }

    /// Sets the number of retries after the initial attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the initial and maximum retry backoff bounds.
    #[must_use]
    pub fn with_retry_backoff(mut self, initial: Duration, max: Duration) -> Self {
        self.retry_initial_backoff = initial;
        self.retry_max_backoff = max;
        self
    }

    /// Sets the statuses that trigger a retry.
    #[must_use]
    pub fn with_retry_statuses(mut self, statuses: impl IntoIterator<Item = u16>) -> Self {
        self.retry_statuses = statuses.into_iter().collect();
        self
    }

    /// Replaces the default request headers.
    #[must_use]
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = headers.into_iter().collect();
        self
    }

    /// Adds one default request header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Sets an explicit proxy URL.
    #[must_use]
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self
    }

    /// Enables or disables TLS certificate verification.
    #[must_use]
    pub fn with_verify_ssl(mut self, verify_ssl: bool) -> Self {
        self.verify_ssl = verify_ssl;
        self
    }

    /// Sets the `User-Agent` header.
    #[must_use]
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Caps the size of a response body in bytes.
    #[must_use]
    pub fn with_max_response_bytes(mut self, max_bytes: u64) -> Self {
        self.max_response_bytes = Some(max_bytes);
        self
    }

    /// Enables or disables following redirects.
    #[must_use]
    pub fn with_follow_redirects(mut self, follow_redirects: bool) -> Self {
        self.follow_redirects = follow_redirects;
        self
    }

    /// Builds the retry policy described by this configuration.
    #[must_use]
    pub fn retry_policy(&self) -> RetryPolicy {
        RetryPolicy {
            max_retries: self.max_retries,
            initial_backoff: self.retry_initial_backoff,
            max_backoff: self.retry_max_backoff,
            retry_statuses: self.retry_statuses.clone(),
            jitter: true,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_documented_values() {
        let config = HttpClientConfig::default();

        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.connect_timeout, None);
        assert_eq!(config.max_retries, 0);
        assert_eq!(config.retry_initial_backoff, Duration::from_millis(200));
        assert_eq!(config.retry_max_backoff, Duration::from_secs(5));
        assert_eq!(config.retry_statuses, vec![429, 500, 502, 503, 504]);
        assert!(config.headers.is_empty());
        assert_eq!(config.proxy, None);
        assert!(config.verify_ssl);
        assert_eq!(config.user_agent, None);
        assert_eq!(config.max_response_bytes, None);
        assert!(config.follow_redirects);
    }

    #[test]
    fn partial_deserialization_keeps_defaults() {
        let config: HttpClientConfig =
            serde_json::from_str(r#"{"max_retries": 2, "headers": {"X-A": "1"}}"#).unwrap();

        assert_eq!(config.max_retries, 2);
        assert_eq!(config.headers.get("X-A").map(String::as_str), Some("1"));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.retry_initial_backoff, Duration::from_millis(200));
        assert_eq!(config.retry_max_backoff, Duration::from_secs(5));
        assert_eq!(config.retry_statuses, vec![429, 500, 502, 503, 504]);
        assert!(config.verify_ssl);
        assert!(config.follow_redirects);
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let config: HttpClientConfig =
            serde_json::from_str(r#"{"timeout": {"secs": 1, "nanos": 0}, "extra": true}"#).unwrap();
        assert_eq!(config.timeout, Duration::from_secs(1));
    }

    #[test]
    fn round_trips_through_json() {
        let config = HttpClientConfig::default()
            .with_timeout(Duration::from_secs(7))
            .with_max_retries(4)
            .with_retry_backoff(Duration::from_millis(10), Duration::from_millis(80));

        let json = serde_json::to_string(&config).unwrap();
        let restored: HttpClientConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn builder_chain_sets_every_field() {
        let mut headers = BTreeMap::new();
        headers.insert("X-First".to_owned(), "one".to_owned());

        let config = HttpClientConfig::default()
            .with_timeout(Duration::from_secs(5))
            .with_connect_timeout(Some(Duration::from_secs(2)))
            .with_max_retries(3)
            .with_retry_backoff(Duration::from_millis(50), Duration::from_millis(500))
            .with_retry_statuses(vec![503])
            .with_headers(headers)
            .with_header("X-Second", "two")
            .with_proxy("http://proxy.local:8080")
            .with_verify_ssl(false)
            .with_user_agent("openjiuwen-test/1.0")
            .with_max_response_bytes(1024)
            .with_follow_redirects(false);

        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.connect_timeout, Some(Duration::from_secs(2)));
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_initial_backoff, Duration::from_millis(50));
        assert_eq!(config.retry_max_backoff, Duration::from_millis(500));
        assert_eq!(config.retry_statuses, vec![503]);
        assert_eq!(config.headers.len(), 2);
        assert_eq!(
            config.headers.get("X-Second").map(String::as_str),
            Some("two")
        );
        assert_eq!(config.proxy.as_deref(), Some("http://proxy.local:8080"));
        assert!(!config.verify_ssl);
        assert_eq!(config.user_agent.as_deref(), Some("openjiuwen-test/1.0"));
        assert_eq!(config.max_response_bytes, Some(1024));
        assert!(!config.follow_redirects);
    }

    #[test]
    fn retry_policy_mirrors_the_configuration() {
        let config = HttpClientConfig::default()
            .with_max_retries(2)
            .with_retry_backoff(Duration::from_millis(10), Duration::from_millis(40))
            .with_retry_statuses(vec![500]);

        let policy = config.retry_policy();
        assert_eq!(policy.max_retries, 2);
        assert_eq!(policy.initial_backoff, Duration::from_millis(10));
        assert_eq!(policy.max_backoff, Duration::from_millis(40));
        assert_eq!(policy.statuses(), &[500]);
        assert!(policy.jitter);
    }
}
