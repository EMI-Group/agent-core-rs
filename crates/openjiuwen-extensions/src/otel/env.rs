//! Environment-variable resolution for the OTLP exporter.
//!
//! Port of the environment handling around `init_otel_tracer` in
//! `openjiuwen/extensions/tracer_otel/setup.py` (v0.1.20): Python reads the
//! OTLP knobs with `os.getenv(...)` inside `_build_provider` /
//! `_create_otlp_exporter` and hands `endpoint`/`headers` straight to the SDK
//! exporter. Rust funnels them through this module, so the variable names, the
//! precedence, the trimming rules, and the defaults live in exactly one place —
//! and so a host can resolve a collector endpoint without the SDK (this module
//! compiles in every feature mode and holds no SDK types).
//!
//! # Precedence
//!
//! For every setting the first non-blank value wins:
//!
//! 1. the value configured explicitly on [`OtelTracerConfig`],
//! 2. the `openJiuwen`-specific variable (`OPENJIUWEN_OTEL_*`),
//! 3. the standard OpenTelemetry variable (`OTEL_*`),
//! 4. the built-in default ([`DEFAULT_OTLP_HTTP_ENDPOINT`], `grpc`,
//!    `openjiuwen`).
//!
//! Blank (empty or whitespace-only) values count as unset, matching the
//! normalisation [`var`] applies (Python parity: `os.getenv(name) or None`).
//!
//! # Headers wire format
//!
//! [`parse_headers`] implements the `OTEL_EXPORTER_OTLP_HEADERS` grammar
//! (`k=v,k2=v2`) — the same string the Python SDK parses before passing it to
//! `OTLPSpanExporter(headers=...)`.
//!
//! # Protocol mapping
//!
//! Python selects between the gRPC and HTTP exporters with
//! `protocol in {"grpc", "http"}`. The Rust exporter speaks **OTLP/HTTP with
//! JSON payloads** only (`opentelemetry-otlp`, `prost`, and `tonic` are absent
//! from the offline registry), so the mapping is:
//!
//! | Value | Result |
//! | --- | --- |
//! | `http/json` | [`OtlpProtocol::Http`] (supported) |
//! | `http` | [`OtlpProtocol::Http`] — Python's alias, transported as JSON |
//! | `http/protobuf` | [`Error::unsupported`] (no protobuf encoder offline) |
//! | `grpc` | [`Error::unsupported`] (no gRPC transport offline) |
//! | anything else | [`Error::validation`] naming the accepted values |
//!
//! [`Error::unsupported`]: openjiuwen_common::error::Error::unsupported
//! [`Error::validation`]: openjiuwen_common::error::Error::validation
//! [`OtelTracerConfig`]: super::config::OtelTracerConfig
//! [`var`]: openjiuwen_common::config::env::var

use std::collections::BTreeMap;

use openjiuwen_common::config::env::var;
use openjiuwen_common::error::{Error, Result};

use super::config::OtlpProtocol;

/// `openJiuwen` collector endpoint: `OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT`.
pub const OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT";
/// `openJiuwen` OTLP request headers: `OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS`.
pub const OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS: &str = "OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS";
/// `openJiuwen` OTLP transport protocol: `OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL`.
pub const OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL: &str = "OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL";
/// `openJiuwen` `service.name` override: `OPENJIUWEN_OTEL_SERVICE_NAME`.
pub const OPENJIUWEN_OTEL_SERVICE_NAME: &str = "OPENJIUWEN_OTEL_SERVICE_NAME";
/// Standard OTLP trace endpoint: `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
pub const OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";
/// Standard OTLP endpoint covering every signal: `OTEL_EXPORTER_OTLP_ENDPOINT`.
pub const OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
/// Standard OTLP request headers: `OTEL_EXPORTER_OTLP_HEADERS`.
pub const OTEL_EXPORTER_OTLP_HEADERS: &str = "OTEL_EXPORTER_OTLP_HEADERS";
/// Standard OTLP transport protocol: `OTEL_EXPORTER_OTLP_PROTOCOL`.
pub const OTEL_EXPORTER_OTLP_PROTOCOL: &str = "OTEL_EXPORTER_OTLP_PROTOCOL";
/// Standard `service.name` resource attribute: `OTEL_SERVICE_NAME`.
pub const OTEL_SERVICE_NAME: &str = "OTEL_SERVICE_NAME";

/// Default OTLP/HTTP collector endpoint.
///
/// The OTLP/HTTP default port (`4318`), matching the endpoint the Python
/// examples use and the collector in `deploy/observability`. The `/v1/traces`
/// suffix is appended by the exporter, never stored here.
pub const DEFAULT_OTLP_HTTP_ENDPOINT: &str = "http://localhost:4318";

/// `service.name` used when neither the configuration nor the environment
/// names one (Python `OtelTracerConfig.service_name` default).
const DEFAULT_SERVICE_NAME: &str = "openjiuwen";

/// Protocol values accepted by [`parse_protocol`], named in validation errors.
const ACCEPTED_PROTOCOLS: &str = "\"http/json\" or \"http\"";

/// Parses a `k=v,k2=v2` header list (`OTEL_*_HEADERS` wire format).
///
/// Both sides of every pair are trimmed; the last duplicate key wins; empty
/// entries, entries without `=`, and entries with an empty key are skipped.
/// A value may itself contain `=` (the split happens on the first one), which
/// keeps base64 credentials such as `Basic abc==` intact.
#[must_use]
pub fn parse_headers(raw: &str) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::new();
    for entry in raw.split(',') {
        let Some((key, value)) = entry.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        headers.insert(key.to_owned(), value.trim().to_owned());
    }
    headers
}

/// Reads the OTLP headers from the environment.
///
/// `OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS` wins over
/// `OTEL_EXPORTER_OTLP_HEADERS`. Returns `None` when both are unset or blank,
/// and also when the value parses to no header at all, so callers never have to
/// special-case an empty map.
#[must_use]
pub fn env_headers() -> Option<BTreeMap<String, String>> {
    let raw =
        var(OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS).or_else(|| var(OTEL_EXPORTER_OTLP_HEADERS))?;
    let headers = parse_headers(&raw);
    if headers.is_empty() {
        None
    } else {
        Some(headers)
    }
}

/// Reads the collector endpoint from the environment.
///
/// Precedence: `OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT`, then
/// `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (the trace-specific standard variable),
/// then `OTEL_EXPORTER_OTLP_ENDPOINT`. Returns `None` when all are unset or
/// blank; no default is applied here (see [`resolve_endpoint`]).
#[must_use]
pub fn env_endpoint() -> Option<String> {
    var(OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT)
        .or_else(|| var(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT))
        .or_else(|| var(OTEL_EXPORTER_OTLP_ENDPOINT))
}

/// Reads the raw OTLP protocol string from the environment.
///
/// `OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL` wins over
/// `OTEL_EXPORTER_OTLP_PROTOCOL`. The value is returned verbatim (trimmed) and
/// still has to be interpreted by [`parse_protocol`], so a bad value can be
/// reported with its own error instead of being silently dropped here.
#[must_use]
pub fn env_protocol_value() -> Option<String> {
    var(OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL).or_else(|| var(OTEL_EXPORTER_OTLP_PROTOCOL))
}

/// Reads `service.name` from the environment.
///
/// `OPENJIUWEN_OTEL_SERVICE_NAME` wins over `OTEL_SERVICE_NAME`. Returns `None`
/// when both are unset or blank.
#[must_use]
pub fn env_service_name() -> Option<String> {
    var(OPENJIUWEN_OTEL_SERVICE_NAME).or_else(|| var(OTEL_SERVICE_NAME))
}

/// Resolves the collector endpoint: explicit, then environment, then
/// [`DEFAULT_OTLP_HTTP_ENDPOINT`].
///
/// A blank `explicit` endpoint counts as unset. The returned value is the
/// **base** endpoint (`http://host:4318`); the `/v1/traces` suffix is added by
/// the exporter, exactly as Python appends it in `_create_otlp_exporter`.
#[must_use]
pub fn resolve_endpoint(explicit: Option<&str>) -> String {
    if let Some(endpoint) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return endpoint.to_owned();
    }
    env_endpoint().unwrap_or_else(|| DEFAULT_OTLP_HTTP_ENDPOINT.to_owned())
}

/// Resolves the OTLP request headers: explicit, then environment, then empty.
///
/// The explicit map wins only when it carries at least one entry, so a
/// configuration built with [`Default::default`](std::default::Default) still
/// picks up the environment (Python's `headers or {}` fallback).
#[must_use]
pub fn resolve_headers(explicit: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    if explicit.is_empty() {
        env_headers().unwrap_or_default()
    } else {
        explicit.clone()
    }
}

/// Resolves the OTLP transport protocol.
///
/// An explicit [`OtlpProtocol::Http`] always wins. Otherwise the environment is
/// consulted, so [`OtlpProtocol::Grpc`] — the Python/Rust configuration default
/// — can be overridden by `OTEL_EXPORTER_OTLP_PROTOCOL=http/json`; when the
/// variable is unset the explicit value is returned unchanged.
///
/// # Errors
///
/// - [`Error::unsupported`] when the environment asks for `grpc` or
///   `http/protobuf`: neither transport is available offline, and the message
///   names `http/json` as the alternative.
/// - [`Error::validation`] for any other value, listing the accepted ones.
pub fn resolve_protocol(explicit: OtlpProtocol) -> Result<OtlpProtocol> {
    if explicit == OtlpProtocol::Http {
        return Ok(OtlpProtocol::Http);
    }
    match env_protocol_value() {
        Some(raw) => parse_protocol(&raw),
        None => Ok(explicit),
    }
}

/// Parses one raw OTLP protocol string into an [`OtlpProtocol`].
///
/// Matching is ASCII case-insensitive, so `HTTP/JSON` and `http/json` agree.
/// Python's `"http"` is accepted as an alias for the HTTP exporter; in Rust it
/// is transported as JSON rather than protobuf (documented deviation, see the
/// module docs).
///
/// # Errors
///
/// - [`Error::unsupported`] for `grpc` and `http/protobuf`: the exporter here
///   speaks OTLP/HTTP with JSON payloads, and no gRPC or protobuf encoder is
///   available offline.
/// - [`Error::validation`] for every other (including blank) value, naming the
///   accepted values.
pub fn parse_protocol(raw: &str) -> Result<OtlpProtocol> {
    let value = raw.trim();
    if value.eq_ignore_ascii_case("http/json") || value.eq_ignore_ascii_case("http") {
        return Ok(OtlpProtocol::Http);
    }
    if value.eq_ignore_ascii_case("http/protobuf") {
        return Err(Error::unsupported(
            "otlp protocol \"http/protobuf\" is not available: the exporter speaks OTLP/HTTP \
             with JSON payloads and no protobuf encoder is available offline; use \"http/json\"",
        ));
    }
    if value.eq_ignore_ascii_case("grpc") {
        return Err(Error::unsupported(
            "otlp protocol \"grpc\" is not available: OTLP/gRPC needs tonic and prost, which are \
             not available offline; use \"http/json\"",
        ));
    }
    Err(Error::validation(format!(
        "unknown otlp protocol {value:?}, supported: {ACCEPTED_PROTOCOLS}"
    )))
}

/// Resolves the `service.name` resource attribute.
///
/// A non-empty explicit name other than the built-in default (`openjiuwen`)
/// wins; otherwise `OPENJIUWEN_OTEL_SERVICE_NAME`, then `OTEL_SERVICE_NAME`,
/// then `openjiuwen`. The default is treated as unset so a configuration built
/// with [`Default::default`](std::default::Default) still honours the
/// environment (Python `os.getenv("OTEL_SERVICE_NAME", "openjiuwen")`).
#[must_use]
pub fn resolve_service_name(explicit: &str) -> String {
    let explicit = explicit.trim();
    if !explicit.is_empty() && explicit != DEFAULT_SERVICE_NAME {
        return explicit.to_owned();
    }
    env_service_name().unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_owned())
}

/// Test-only helpers shared by the environment tests of this module and
/// [`config`](super::config).
///
/// Both modules mutate the same process-global variables, so they must agree on
/// one lock; [`ENV_LOCK`] is that lock and [`EnvGuard`] is the RAII wrapper that
/// acquires it, clears every OTLP variable, and restores them on drop.
#[cfg(test)]
pub(crate) mod test_support {
    /// Serialises every test that reads or writes the OTLP process
    /// environment.
    pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Every variable [`super`] resolves.
    pub(crate) const ENV_VARS: [&str; 9] = [
        super::OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT,
        super::OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS,
        super::OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL,
        super::OPENJIUWEN_OTEL_SERVICE_NAME,
        super::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
        super::OTEL_EXPORTER_OTLP_ENDPOINT,
        super::OTEL_EXPORTER_OTLP_HEADERS,
        super::OTEL_EXPORTER_OTLP_PROTOCOL,
        super::OTEL_SERVICE_NAME,
    ];

    /// Holds [`ENV_LOCK`] for the lifetime of a test and keeps the OTLP
    /// variables clean.
    ///
    /// Creation clears every variable, so a test starts from the "unset" state;
    /// dropping removes them again, so no test leaks into the next one. Only one
    /// guard may be alive per test thread ([`ENV_LOCK`] is not reentrant); use
    /// [`clear_env`] to switch scenarios inside a single test.
    pub(crate) struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        /// Acquires the lock, recovering it when another test panicked.
        pub(crate) fn new() -> Self {
            let lock = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            clear_env();
            Self { _lock: lock }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            clear_env();
        }
    }

    /// Sets one OTLP variable for the rest of the test.
    pub(crate) fn set_env(name: &str, value: &str) {
        std::env::set_var(name, value);
    }

    /// Removes every OTLP variable, starting a new scenario inside one test.
    pub(crate) fn clear_env() {
        for name in ENV_VARS {
            std::env::remove_var(name);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::test_support::{clear_env, set_env, EnvGuard};
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    // port of tests/unit_tests/extensions/tracer_otel/test_setup.py (env cases)

    #[test]
    fn parse_headers_splits_pairs_and_trims() {
        let headers = parse_headers("a=1, b=2");
        assert_eq!(headers.len(), 2);
        assert_eq!(headers["a"], "1");
        assert_eq!(headers["b"], "2");
    }

    #[test]
    fn parse_headers_rejects_empty_input() {
        assert!(parse_headers("").is_empty());
        assert!(parse_headers("   ").is_empty());
        assert!(parse_headers(" , , ").is_empty());
    }

    #[test]
    fn parse_headers_skips_entries_without_a_separator() {
        assert!(parse_headers("k").is_empty());
        let headers = parse_headers("k, a=1");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers["a"], "1");
    }

    #[test]
    fn parse_headers_skips_entries_with_an_empty_key() {
        assert!(parse_headers("=value").is_empty());
        let headers = parse_headers(" =value, a=1");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers["a"], "1");
    }

    #[test]
    fn parse_headers_keeps_the_last_duplicate() {
        let headers = parse_headers("a=1,a=2");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers["a"], "2");
    }

    #[test]
    fn parse_headers_keeps_the_remainder_of_the_value() {
        let headers = parse_headers("authorization=Basic abc==");
        assert_eq!(headers["authorization"], "Basic abc==");
        let headers = parse_headers("tenant=x=y");
        assert_eq!(headers["tenant"], "x=y");
    }

    #[test]
    fn parse_headers_allows_an_empty_value() {
        let headers = parse_headers("a=");
        assert_eq!(headers["a"], "");
    }

    #[test]
    fn endpoint_falls_back_to_the_built_in_default() {
        let _guard = EnvGuard::new();
        assert_eq!(env_endpoint(), None);
        assert_eq!(resolve_endpoint(None), DEFAULT_OTLP_HTTP_ENDPOINT);
        assert_eq!(resolve_endpoint(Some("   ")), DEFAULT_OTLP_HTTP_ENDPOINT);
    }

    #[test]
    fn endpoint_precedence_follows_the_contract() {
        let _guard = EnvGuard::new();
        assert_eq!(resolve_endpoint(None), DEFAULT_OTLP_HTTP_ENDPOINT);
        set_env(OTEL_EXPORTER_OTLP_ENDPOINT, "http://otel-any:4318");
        assert_eq!(resolve_endpoint(None), "http://otel-any:4318");
        set_env(
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
            "http://otel-traces:4318",
        );
        assert_eq!(resolve_endpoint(None), "http://otel-traces:4318");
        set_env(
            OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT,
            "http://openjiuwen:4318",
        );
        assert_eq!(resolve_endpoint(None), "http://openjiuwen:4318");
        // An explicit endpoint outranks every variable.
        assert_eq!(
            resolve_endpoint(Some("http://explicit:4318")),
            "http://explicit:4318"
        );
    }

    #[test]
    fn endpoint_ignores_blank_values() {
        let _guard = EnvGuard::new();
        set_env(OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT, "   ");
        set_env(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, "");
        set_env(OTEL_EXPORTER_OTLP_ENDPOINT, "http://otel-any:4318");
        assert_eq!(env_endpoint().as_deref(), Some("http://otel-any:4318"));
        assert_eq!(resolve_endpoint(None), "http://otel-any:4318");

        clear_env();
        set_env(OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT, "  ");
        assert_eq!(env_endpoint(), None);
    }

    #[test]
    fn headers_precedence_follows_the_contract() {
        let _guard = EnvGuard::new();
        assert_eq!(env_headers(), None);
        assert!(resolve_headers(&BTreeMap::new()).is_empty());

        set_env(OTEL_EXPORTER_OTLP_HEADERS, "a=otel");
        assert_eq!(resolve_headers(&BTreeMap::new())["a"], "otel");

        set_env(OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS, "a=openjiuwen,b=2");
        let headers = env_headers().unwrap();
        assert_eq!(headers["a"], "openjiuwen");
        assert_eq!(headers["b"], "2");

        // An explicit non-empty map outranks every variable.
        let explicit = BTreeMap::from([("authorization".to_owned(), "Bearer x".to_owned())]);
        assert_eq!(resolve_headers(&explicit), explicit);

        // A variable that parses to nothing counts as unset.
        clear_env();
        set_env(OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS, "not-a-pair");
        assert_eq!(env_headers(), None);
        assert!(resolve_headers(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn protocol_http_is_always_supported() {
        let _guard = EnvGuard::new();
        set_env(OTEL_EXPORTER_OTLP_PROTOCOL, "grpc");
        assert_eq!(
            resolve_protocol(OtlpProtocol::Http).unwrap(),
            OtlpProtocol::Http
        );
    }

    #[test]
    fn protocol_env_overrides_the_default_config_value() {
        let _guard = EnvGuard::new();
        set_env(OTEL_EXPORTER_OTLP_PROTOCOL, "http/json");
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap(),
            OtlpProtocol::Http
        );
    }

    #[test]
    fn protocol_openjiuwen_variable_wins_over_the_standard_one() {
        let _guard = EnvGuard::new();
        set_env(OTEL_EXPORTER_OTLP_PROTOCOL, "grpc");
        set_env(OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL, "http/json");
        assert_eq!(env_protocol_value().as_deref(), Some("http/json"));
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap(),
            OtlpProtocol::Http
        );
    }

    #[test]
    fn protocol_falls_back_to_the_explicit_value_without_env() {
        let _guard = EnvGuard::new();
        assert_eq!(env_protocol_value(), None);
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap(),
            OtlpProtocol::Grpc
        );
        assert_eq!(
            resolve_protocol(OtlpProtocol::Http).unwrap(),
            OtlpProtocol::Http
        );
    }

    #[test]
    fn parse_protocol_accepts_http_json_and_the_python_alias() {
        assert_eq!(parse_protocol("http/json").unwrap(), OtlpProtocol::Http);
        assert_eq!(parse_protocol("  http  ").unwrap(), OtlpProtocol::Http);
        assert_eq!(parse_protocol("HTTP/JSON").unwrap(), OtlpProtocol::Http);
    }

    #[test]
    fn parse_protocol_rejects_grpc_with_the_alternative() {
        for raw in ["grpc", "GRPC", " grpc "] {
            let error = parse_protocol(raw).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported, "value {raw:?}");
            let message = error.to_string();
            assert!(message.contains("grpc"), "unexpected message: {message}");
            assert!(
                message.contains("http/json"),
                "the message must name the alternative: {message}"
            );
        }
    }

    #[test]
    fn parse_protocol_rejects_protobuf_with_the_alternative() {
        let error = parse_protocol("http/protobuf").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(
            message.contains("http/protobuf") && message.contains("http/json"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn parse_protocol_rejects_unknown_values_as_validation() {
        for raw in ["", "thrift", "otlp"] {
            let error = parse_protocol(raw).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "value {raw:?}");
            let message = error.to_string();
            assert!(
                message.contains("http/json"),
                "the message must list the accepted values: {message}"
            );
        }
    }

    #[test]
    fn protocol_errors_surface_through_resolve_protocol() {
        let _guard = EnvGuard::new();
        set_env(OTEL_EXPORTER_OTLP_PROTOCOL, "grpc");
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap_err().kind(),
            ErrorKind::Unsupported
        );

        set_env(OTEL_EXPORTER_OTLP_PROTOCOL, "http/protobuf");
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap_err().kind(),
            ErrorKind::Unsupported
        );

        set_env(OTEL_EXPORTER_OTLP_PROTOCOL, "carrier-pigeon");
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap_err().kind(),
            ErrorKind::Validation
        );

        clear_env();
        assert_eq!(
            resolve_protocol(OtlpProtocol::Grpc).unwrap(),
            OtlpProtocol::Grpc
        );
    }

    #[test]
    fn service_name_precedence_follows_the_contract() {
        let _guard = EnvGuard::new();
        assert_eq!(env_service_name(), None);
        assert_eq!(resolve_service_name(""), "openjiuwen");
        assert_eq!(resolve_service_name("openjiuwen"), "openjiuwen");

        set_env(OTEL_SERVICE_NAME, "otel-service");
        assert_eq!(resolve_service_name(""), "otel-service");
        assert_eq!(resolve_service_name("openjiuwen"), "otel-service");

        set_env(OPENJIUWEN_OTEL_SERVICE_NAME, "openjiuwen-service");
        assert_eq!(env_service_name().as_deref(), Some("openjiuwen-service"));

        // An explicit non-default name outranks every variable.
        assert_eq!(resolve_service_name("explicit-service"), "explicit-service");

        clear_env();
        set_env(OPENJIUWEN_OTEL_SERVICE_NAME, "   ");
        set_env(OTEL_SERVICE_NAME, "otel-service");
        assert_eq!(resolve_service_name("  "), "otel-service");
    }

    #[test]
    fn service_name_default_matches_the_config_default() {
        let _guard = EnvGuard::new();
        assert_eq!(
            resolve_service_name(""),
            super::super::config::OtelTracerConfig::default().service_name
        );
    }
}
