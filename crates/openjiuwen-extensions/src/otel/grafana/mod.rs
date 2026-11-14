//! Grafana Cloud OTLP gateway preset.
//!
//! Grafana Cloud ingests traces through its OTLP gateway
//! (`https://otlp-gateway-<zone>.grafana.net/otlp`) and authenticates every
//! request with HTTP Basic credentials built from the stack *instance id* and a
//! *Cloud access policy token* (`base64("<instance>:<token>")`). This module
//! turns those three inputs into an [`OtelTracerConfig`], so a host does not
//! have to spell out the endpoint or the header by hand.
//!
//! # What the integration needs from the outside world
//!
//! - A Grafana Cloud stack: the instance id, a token with the
//!   `traces:write` scope, and the zone the stack lives in
//!   (`prod-us-east-0`, `prod-eu-west-2`, …).
//! - Network access to the gateway. Nothing here contacts Grafana at
//!   configuration time; the first request happens when a span batch is
//!   exported.
//!
//! Both presets select [`OtlpProtocol::Http`] — the transport Grafana's gateway
//! and a local collector accept on the same endpoint — and the exporter encodes
//! every batch as OTLP/HTTP JSON.
//!
//! # Self-hosted Grafana
//!
//! A self-hosted Grafana Agent or Grafana Alloy deployment is an OTLP/HTTP
//! *collector*. [`GrafanaAgentConfig`] targets one that is already running: it
//! resolves the endpoint ([`DEFAULT_LOCAL_OTLP_ENDPOINT`] when none is
//! configured), carries optional headers, and produces the same
//! [`OtelTracerConfig`] and exporter the gateway path uses, so finished spans
//! are pushed to the local collector exactly like to any other collector.
//!
//! What stays outside this crate is the collector's *lifecycle*: it cannot
//! install, start, discover, or supervise an Agent/Alloy process, so that
//! process has to exist and listen before the first batch arrives.
//!
//! ```
//! use openjiuwen_extensions::otel::grafana::{GrafanaAgentConfig, DEFAULT_LOCAL_OTLP_ENDPOINT};
//!
//! let agent = GrafanaAgentConfig::default()
//!     .with_service_name("checkout")
//!     .with_header("x-tenant", "acme");
//! let otel = agent.otel_config()?;
//! assert_eq!(
//!     otel.exporter_endpoint.as_deref(),
//!     Some(DEFAULT_LOCAL_OTLP_ENDPOINT),
//! );
//! assert_eq!(otel.service_name, "checkout");
//! assert_eq!(otel.headers["x-tenant"], "acme");
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```
//!
//! ```
//! use openjiuwen_extensions::otel::grafana::GrafanaCloudConfig;
//!
//! let config = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0")
//!     .with_service_name("checkout");
//! assert_eq!(
//!     config.endpoint(),
//!     "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
//! );
//! let otel = config.otel_config()?;
//! assert_eq!(otel.exporter_endpoint.as_deref(), Some(config.endpoint().as_str()));
//! assert_eq!(otel.headers["authorization"], config.authorization());
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
#[cfg(feature = "otel")]
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::config::env::var;
use openjiuwen_common::error::{Error, Result};
use reqwest::header::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use url::Url;

use super::config::{ExporterType, OtelTracerConfig, OtlpProtocol};

/// Grafana Cloud stack instance id: `GRAFANA_INSTANCE_ID`.
pub const GRAFANA_INSTANCE_ID: &str = "GRAFANA_INSTANCE_ID";
/// Grafana Cloud access-policy token: `GRAFANA_API_TOKEN`.
pub const GRAFANA_API_TOKEN: &str = "GRAFANA_API_TOKEN";
/// Grafana Cloud zone: `GRAFANA_ZONE`.
pub const GRAFANA_ZONE: &str = "GRAFANA_ZONE";
/// Alias for [`GRAFANA_ZONE`] used by some Grafana tooling: `GRAFANA_OTLP_ZONE`.
pub const GRAFANA_OTLP_ZONE: &str = "GRAFANA_OTLP_ZONE";
/// Complete OTLP endpoint override: `GRAFANA_OTLP_ENDPOINT`.
///
/// Wins over the zone-derived endpoint, which is how a private or proxied
/// gateway is addressed.
pub const GRAFANA_OTLP_ENDPOINT: &str = "GRAFANA_OTLP_ENDPOINT";

/// Default endpoint of a local Grafana Agent / Grafana Alloy OTLP/HTTP
/// collector: `http://localhost:4318`.
///
/// This is the port both products listen on out of the box, and the same value
/// the generic exporter falls back to
/// ([`crate::otel::env::DEFAULT_OTLP_HTTP_ENDPOINT`]); the constant is re-used
/// rather than duplicated so the preset and the exporter cannot drift apart.
/// It is the **base** endpoint — the `/v1/traces` signal path is appended by
/// the exporter, never stored here.
pub const DEFAULT_LOCAL_OTLP_ENDPOINT: &str = crate::otel::env::DEFAULT_OTLP_HTTP_ENDPOINT;

/// `service.name` used when the host does not name one.
const DEFAULT_SERVICE_NAME: &str = "openjiuwen";
/// Host prefix of the Grafana Cloud OTLP gateway.
const ZONE_PREFIX: &str = "otlp-gateway-";
/// Domain suffix of the Grafana Cloud OTLP gateway.
const GRAFANA_DOMAIN: &str = ".grafana.net";
/// Path the OTLP/HTTP exporter appends its signal path to.
const OTLP_PATH: &str = "/otlp";

/// Typed preset for the Grafana Cloud OTLP gateway.
///
/// Build it with [`GrafanaCloudConfig::new`] or read it from the environment
/// with [`GrafanaCloudConfig::from_env`]; [`Self::validate`] (called by
/// [`Self::otel_config`]) rejects incomplete credentials before any span is
/// exported.
///
/// [`Debug`] never prints the token.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GrafanaCloudConfig {
    /// Stack instance id (HTTP Basic user).
    instance_id: String,
    /// Cloud access-policy token (HTTP Basic password).
    token: String,
    /// Zone slug or Grafana gateway host the endpoint is derived from.
    zone: String,
    /// `service.name` reported with every span.
    service_name: String,
    /// Complete endpoint override; wins over the zone-derived endpoint.
    otlp_endpoint: Option<String>,
}

impl Default for GrafanaCloudConfig {
    fn default() -> Self {
        Self {
            instance_id: String::new(),
            token: String::new(),
            zone: String::new(),
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
            otlp_endpoint: None,
        }
    }
}

impl std::fmt::Debug for GrafanaCloudConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GrafanaCloudConfig")
            .field("instance_id", &self.instance_id)
            .field("token", &"<redacted>")
            .field("zone", &self.zone)
            .field("service_name", &self.service_name)
            .field("otlp_endpoint", &self.otlp_endpoint)
            .finish()
    }
}

impl GrafanaCloudConfig {
    /// Builds a preset from the instance id, the access-policy token, and the
    /// zone.
    ///
    /// No validation happens here: an unparsable zone surfaces from
    /// [`Self::validate`] (or [`Self::otel_config`]) instead of making the
    /// constructor fallible. The zone is normalised when the endpoint is
    /// derived, so both `prod-us-east-0` and
    /// `otlp-gateway-prod-us-east-0.grafana.net` address the same gateway.
    #[must_use]
    pub fn new(
        instance_id: impl Into<String>,
        token: impl Into<String>,
        zone: impl Into<String>,
    ) -> Self {
        Self {
            instance_id: instance_id.into(),
            token: token.into(),
            zone: zone.into(),
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
            otlp_endpoint: None,
        }
    }

    /// Returns the preset with a different `service.name`.
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Returns the preset with a different zone (or Grafana gateway host).
    #[must_use]
    pub fn with_zone(mut self, zone: impl Into<String>) -> Self {
        self.zone = zone.into();
        self
    }

    /// Returns the preset with a complete endpoint override.
    ///
    /// The override wins over the zone-derived endpoint — the same precedence
    /// `GRAFANA_OTLP_ENDPOINT` has in [`Self::from_env`].
    #[must_use]
    pub fn with_otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Reports the configured zone as given (normalisation happens in
    /// [`Self::endpoint`]).
    #[must_use]
    pub fn zone(&self) -> &str {
        &self.zone
    }

    /// Reports the configured `service.name`.
    #[must_use]
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Returns the OTLP endpoint: the override when configured, otherwise the
    /// gateway URL derived from the zone.
    ///
    /// Trailing slashes are dropped. An unparsable zone yields an empty string
    /// here — [`Self::validate`] is the function that reports why.
    #[must_use]
    pub fn endpoint(&self) -> String {
        if let Some(endpoint) = &self.otlp_endpoint {
            return endpoint.trim().trim_end_matches('/').to_owned();
        }
        normalize_zone(&self.zone).map_or_else(
            |_| String::new(),
            |zone| format!("https://{ZONE_PREFIX}{zone}{GRAFANA_DOMAIN}{OTLP_PATH}"),
        )
    }

    /// Returns the `authorization` header value: `"Basic "` followed by the
    /// standard (padded) base64 of `"<instance_id>:<token>"`.
    #[must_use]
    pub fn authorization(&self) -> String {
        let credentials = format!("{}:{}", self.instance_id, self.token);
        format!("Basic {}", STANDARD.encode(credentials))
    }

    /// Checks that the preset can address the gateway.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the instance id, the token, or the
    /// `service.name` is blank, when the zone is not a Grafana Cloud zone slug
    /// (see [`Self::endpoint`] for the accepted forms), or when the endpoint
    /// override is not an absolute `http`/`https` URL.
    pub fn validate(&self) -> Result<()> {
        if self.instance_id.trim().is_empty() {
            return Err(Error::validation(format!(
                "Grafana `instance_id` is required: set it on the preset or in {GRAFANA_INSTANCE_ID}"
            )));
        }
        if self.token.trim().is_empty() {
            return Err(Error::validation(format!(
                "Grafana `token` is required: set it on the preset or in {GRAFANA_API_TOKEN}"
            )));
        }
        if self.service_name.trim().is_empty() {
            return Err(Error::validation(
                "Grafana `service_name` must not be empty",
            ));
        }
        match &self.otlp_endpoint {
            Some(endpoint) => {
                validate_endpoint(endpoint)?;
            }
            None => {
                normalize_zone(&self.zone)?;
            }
        }
        Ok(())
    }

    /// Builds the OTLP/HTTP tracer configuration for the gateway.
    ///
    /// The result selects [`ExporterType::Otlp`] with [`OtlpProtocol::Http`],
    /// carries the derived endpoint, sets the `authorization` header, and
    /// reports [`Self::service_name`] as `service.name`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::validate`].
    pub fn otel_config(&self) -> Result<OtelTracerConfig> {
        self.validate()?;
        let mut headers = BTreeMap::new();
        headers.insert("authorization".to_owned(), self.authorization());
        let mut config = OtelTracerConfig::default()
            .with_exporter_type(ExporterType::Otlp)
            .with_endpoint(self.endpoint())
            .with_service(self.service_name.clone(), None);
        config.protocol = OtlpProtocol::Http;
        config.headers = headers;
        Ok(config)
    }

    /// Builds an exporter that posts span batches to the gateway.
    ///
    /// Uses the configuration from [`Self::otel_config`] and the batch export
    /// timeout ([`OtelTracerConfig::export_timeout_ms`]) as the per-request
    /// timeout. Construction starts a worker thread but performs no I/O.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::validate`], plus [`Error::config`] when the
    /// exporter worker thread cannot be started.
    ///
    /// [`Error::config`]: openjiuwen_common::error::Error::config
    #[cfg(feature = "otel")]
    pub fn exporter(&self) -> Result<crate::otel::otlp::OtlpHttpExporter> {
        let config = self.otel_config()?;
        let endpoint = config.exporter_endpoint.clone().unwrap_or_default();
        crate::otel::otlp::OtlpHttpExporter::new(
            endpoint,
            config.headers.clone(),
            Duration::from_millis(config.export_timeout_ms),
        )
    }

    /// Reads the preset from the environment and validates it.
    ///
    /// | Variable | Meaning |
    /// | --- | --- |
    /// | `GRAFANA_INSTANCE_ID` | instance id (required) |
    /// | `GRAFANA_API_TOKEN` | access-policy token (required) |
    /// | `GRAFANA_ZONE` | zone slug (required unless an endpoint is configured) |
    /// | `GRAFANA_OTLP_ZONE` | zone alias, read when `GRAFANA_ZONE` is unset |
    /// | `GRAFANA_OTLP_ENDPOINT` | complete endpoint override (wins over the zone) |
    ///
    /// Blank values count as unset. `GRAFANA_ZONE` wins over
    /// `GRAFANA_OTLP_ZONE`; `GRAFANA_OTLP_ENDPOINT` wins over both, so a host
    /// behind a private gateway does not need a zone at all.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the missing variable, or the errors of
    /// [`Self::validate`].
    pub fn from_env() -> Result<Self> {
        let instance_id = var(GRAFANA_INSTANCE_ID).ok_or_else(|| {
            Error::validation(format!(
                "{GRAFANA_INSTANCE_ID} is required for Grafana Cloud"
            ))
        })?;
        let token = var(GRAFANA_API_TOKEN).ok_or_else(|| {
            Error::validation(format!("{GRAFANA_API_TOKEN} is required for Grafana Cloud"))
        })?;
        let zone = var(GRAFANA_ZONE)
            .or_else(|| var(GRAFANA_OTLP_ZONE))
            .unwrap_or_default();
        let preset = Self {
            instance_id,
            token,
            zone,
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
            otlp_endpoint: var(GRAFANA_OTLP_ENDPOINT),
        };
        preset.validate()?;
        Ok(preset)
    }
}

/// Preset for a local Grafana Agent / Grafana Alloy OTLP push.
///
/// The agent is an OTLP/HTTP collector process that has to be installed and
/// running before the first batch arrives; this preset *targets* it instead of
/// managing it. [`Self::otel_config`] resolves the collector endpoint
/// ([`DEFAULT_LOCAL_OTLP_ENDPOINT`] when none is configured) and validates it,
/// then returns an [`OtelTracerConfig`] selecting [`ExporterType::Otlp`] over
/// [`OtlpProtocol::Http`], so finished spans leave the process as OTLP/HTTP
/// JSON batches exactly as they do for [`GrafanaCloudConfig`].
///
/// Installing, starting, discovering, and supervising the Agent/Alloy process
/// remains with the host; nothing here contacts the collector at configuration
/// time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GrafanaAgentConfig {
    /// Collector endpoint the agent listens on; blank means
    /// [`DEFAULT_LOCAL_OTLP_ENDPOINT`].
    endpoint: String,
    /// `service.name` reported with every span.
    service_name: Option<String>,
    /// Extra headers sent with every OTLP request.
    headers: BTreeMap<String, String>,
}

impl GrafanaAgentConfig {
    /// Builds a preset for a local Grafana Agent / Alloy endpoint.
    ///
    /// Use [`Self::default`] to address [`DEFAULT_LOCAL_OTLP_ENDPOINT`]. No
    /// validation happens here; [`Self::otel_config`] reports an unusable
    /// endpoint.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            service_name: None,
            headers: BTreeMap::new(),
        }
    }

    /// Returns the preset with a different `service.name`.
    ///
    /// A blank name is rejected by [`Self::otel_config`] rather than silently
    /// replaced by the default, so a typo cannot hide.
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = Some(name.into());
        self
    }

    /// Returns the preset with one extra OTLP request header.
    ///
    /// The pair is stored verbatim; an invalid header name or value (including
    /// an empty name or a control character) is reported by [`Self::otel_config`]
    /// with [`Error::validation`].
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Returns the preset with every pair of `headers` added; an existing name
    /// is overwritten.
    ///
    /// Validation matches [`Self::with_header`].
    #[must_use]
    pub fn with_headers(mut self, headers: BTreeMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Reports the configured collector endpoint as given; an empty value means
    /// [`DEFAULT_LOCAL_OTLP_ENDPOINT`].
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Reports the configured `service.name`, if any.
    #[must_use]
    pub fn service_name(&self) -> Option<&str> {
        self.service_name.as_deref()
    }

    /// Reports the extra OTLP request headers.
    #[must_use]
    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    /// Builds the OTLP/HTTP tracer configuration for the local collector.
    ///
    /// The result selects [`ExporterType::Otlp`] with [`OtlpProtocol::Http`],
    /// carries the resolved endpoint, applies [`Self::headers`], and reports the
    /// preset's `service.name` (or `"openjiuwen"` when none is set) with no
    /// `service.version`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the resolved endpoint is not an absolute
    /// `http`/`https` URL, when an explicitly configured `service.name` is
    /// blank, or when a header name or value is not a valid HTTP header.
    pub fn otel_config(&self) -> Result<OtelTracerConfig> {
        let endpoint = self.resolved_endpoint()?;
        let service_name = self.resolved_service_name()?;
        validate_headers(&self.headers)?;
        let mut config = OtelTracerConfig::default()
            .with_exporter_type(ExporterType::Otlp)
            .with_endpoint(endpoint)
            .with_service(service_name, None);
        config.protocol = OtlpProtocol::Http;
        config.headers = self.headers.clone();
        Ok(config)
    }

    /// Builds an exporter that posts span batches to the local collector.
    ///
    /// Uses the configuration from [`Self::otel_config`] and the batch export
    /// timeout ([`OtelTracerConfig::export_timeout_ms`]) as the per-request
    /// timeout. Construction starts a worker thread but performs no I/O, so it
    /// needs no reachable collector.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::otel_config`], plus [`Error::config`] when the
    /// exporter worker thread cannot be started.
    ///
    /// [`Error::config`]: openjiuwen_common::error::Error::config
    #[cfg(feature = "otel")]
    pub fn exporter(&self) -> Result<crate::otel::otlp::OtlpHttpExporter> {
        let config = self.otel_config()?;
        let endpoint = config.exporter_endpoint.clone().unwrap_or_default();
        crate::otel::otlp::OtlpHttpExporter::new(
            endpoint,
            config.headers.clone(),
            Duration::from_millis(config.export_timeout_ms),
        )
    }

    /// Resolves the collector endpoint: the configured value, or
    /// [`DEFAULT_LOCAL_OTLP_ENDPOINT`] when it is blank.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the resolved value is not an absolute
    /// `http`/`https` URL (see [`validate_endpoint`]).
    fn resolved_endpoint(&self) -> Result<String> {
        let configured = self.endpoint.trim();
        let endpoint = if configured.is_empty() {
            DEFAULT_LOCAL_OTLP_ENDPOINT
        } else {
            configured
        };
        validate_endpoint(endpoint)
    }

    /// Resolves `service.name`: the configured name, or
    /// [`DEFAULT_SERVICE_NAME`] when none is set.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a name is set but blank.
    fn resolved_service_name(&self) -> Result<String> {
        match self.service_name.as_deref() {
            Some(name) => {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    return Err(Error::validation(
                        "Grafana Agent `service_name` must not be empty",
                    ));
                }
                Ok(trimmed.to_owned())
            }
            None => Ok(DEFAULT_SERVICE_NAME.to_owned()),
        }
    }
}

/// Validates every configured OTLP header pair.
///
/// Names must be HTTP tokens and values free of control characters, which is
/// exactly what the exporter itself accepts; checking here turns a late
/// exporter error into a configuration error naming the offending header.
///
/// # Errors
///
/// [`Error::validation`] naming the header and the requirement.
fn validate_headers(headers: &BTreeMap<String, String>) -> Result<()> {
    for (name, value) in headers {
        if HeaderName::from_bytes(name.as_bytes()).is_err() {
            return Err(Error::validation(format!(
                "Grafana Agent header name must be a non-empty HTTP token: {name:?}"
            )));
        }
        if HeaderValue::from_str(value).is_err() {
            return Err(Error::validation(format!(
                "Grafana Agent header value for {name:?} must be a valid HTTP header value"
            )));
        }
    }
    Ok(())
}

/// Normalises a zone into the bare Grafana Cloud zone slug.
///
/// Accepted forms, all mapping to `prod-us-east-0`:
///
/// - `prod-us-east-0` (the slug),
/// - `otlp-gateway-prod-us-east-0` (the host label),
/// - `otlp-gateway-prod-us-east-0.grafana.net` (the host),
/// - `https://otlp-gateway-prod-us-east-0.grafana.net` and the same URL with a
///   `/otlp` path or a trailing slash.
///
/// The slug is lowercased. A value given as a URL must name a `*.grafana.net`
/// gateway host. Everything else — a path, a query, a fragment, whitespace, a
/// foreign host, a scheme-only value such as `https://prod-us-east-0`, or an
/// empty `-` segment — is rejected, so a zone cannot smuggle a path into the
/// derived endpoint.
fn normalize_zone(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(Error::validation(
            "Grafana `zone` is required: set GRAFANA_ZONE or GRAFANA_OTLP_ENDPOINT (got an empty \
             zone)",
        ));
    }
    let (had_scheme, without_scheme) = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .map_or((false, trimmed), |rest| (true, rest));
    let without_slash = without_scheme.trim_end_matches('/');
    let without_path = without_slash
        .strip_suffix(OTLP_PATH)
        .unwrap_or(without_slash)
        .trim_end_matches('/');
    if had_scheme && !without_path.ends_with(GRAFANA_DOMAIN) {
        return Err(Error::validation(format!(
            "Grafana `zone` given as a URL must point at a `*{GRAFANA_DOMAIN}` gateway host: {raw:?}"
        )));
    }
    let host = without_path
        .strip_suffix(GRAFANA_DOMAIN)
        .unwrap_or(without_path);
    let zone = host.strip_prefix(ZONE_PREFIX).unwrap_or(host);
    if zone.is_empty() {
        return Err(Error::validation(format!(
            "Grafana `zone` is empty after normalisation: {raw:?}"
        )));
    }
    if let Some(invalid) = zone
        .chars()
        .find(|character| !character.is_ascii_alphanumeric() && *character != '-')
    {
        return Err(Error::validation(format!(
            "Grafana `zone` must be a zone slug such as `prod-us-east-0`, not a URL or path: \
             {raw:?} contains {invalid:?}"
        )));
    }
    if zone.split('-').any(str::is_empty) {
        return Err(Error::validation(format!(
            "Grafana `zone` must not contain an empty `-` segment: {raw:?}"
        )));
    }
    Ok(zone.to_ascii_lowercase())
}

/// Validates a complete endpoint override and returns it without trailing
/// slashes.
///
/// # Errors
///
/// [`Error::validation`] when the endpoint is blank or is not an absolute
/// `http`/`https` URL.
fn validate_endpoint(endpoint: &str) -> Result<String> {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(Error::validation(format!(
            "Grafana `endpoint` is required: set {GRAFANA_OTLP_ENDPOINT} to an absolute http(s) URL"
        )));
    }
    let parsed = Url::parse(trimmed).map_err(|error| {
        Error::validation(format!(
            "Grafana `endpoint` is not an absolute URL: {trimmed} ({error})"
        ))
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(Error::validation(format!(
            "Grafana `endpoint` must use http or https: {trimmed}"
        )));
    }
    Ok(trimmed.to_owned())
}
