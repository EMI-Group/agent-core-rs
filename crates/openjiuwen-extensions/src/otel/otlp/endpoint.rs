//! OTLP/HTTP endpoint normalization.
//!
//! The collector endpoint may be given either as a base URL (`http://host:4318`)
//! or as the complete traces URL (`http://host:4318/v1/traces`). The exporter
//! always posts to the traces path, so the suffix is appended exactly once.

/// Path every OTLP/HTTP trace export is posted to.
const TRACES_PATH: &str = "/v1/traces";

/// Appends `/v1/traces` when `endpoint` does not already end with it.
///
/// Trailing slashes are dropped first, so `http://host:4318/` and
/// `http://host:4318` both normalize to `http://host:4318/v1/traces`, while
/// `http://host:4318/v1/traces/` stays a single traces URL. The result always
/// ends with `/v1/traces` unless `endpoint` is empty, in which case it is just
/// the path (callers validate the endpoint before they get here).
///
/// # Examples
///
/// ```
/// use openjiuwen_extensions::otel::otlp::traces_url;
///
/// assert_eq!(traces_url("http://localhost:4318"), "http://localhost:4318/v1/traces");
/// assert_eq!(traces_url("http://localhost:4318/"), "http://localhost:4318/v1/traces");
/// assert_eq!(
///     traces_url("http://localhost:4318/v1/traces"),
///     "http://localhost:4318/v1/traces"
/// );
/// ```
#[must_use]
pub fn traces_url(endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    if base.ends_with(TRACES_PATH) {
        base.to_owned()
    } else {
        format!("{base}{TRACES_PATH}")
    }
}
