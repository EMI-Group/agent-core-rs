//! Langfuse integration: OTLP preset and REST ingestion client.
//!
//! Langfuse accepts the same data through two surfaces, and this module covers
//! both:
//!
//! 1. **OTLP ingestion** — the trace pipeline installed by [`OtelTracerConfig`]
//!    posts OTLP/HTTP JSON to `{host}/api/public/otel` with HTTP Basic
//!    credentials built from the project's public and secret key. This is the
//!    path an `OtelTracerConfig` or an exporter wants.
//! 2. **REST ingestion** — [`LangfuseClient`] posts typed
//!    `trace-create`/`span-create`/`generation-create` events to
//!    `{host}/api/public/ingestion`, for callers that build observations
//!    themselves instead of emitting spans.
//!
//! # What the integration needs from the outside world
//!
//! - A Langfuse project: the public key (`pk-lf-…`) and the secret key
//!   (`sk-lf-…`). Langfuse Cloud is the default host
//!   (`https://cloud.langfuse.com`); a self-hosted instance is addressed with
//!   [`LangfuseConfig::with_host`] or `LANGFUSE_HOST`/`LANGFUSE_BASEURL`.
//! - Network access to that host. Nothing here contacts Langfuse at
//!   configuration time; requests happen when a span batch is exported or
//!   [`LangfuseClient::ingest`] is called.
//!
//! Both OTLP/HTTP encodings are available offline in this workspace: JSON
//! (the default) and protobuf through the crate's hand-written encoder. Only
//! `gRPC` is deferred, because it needs `tonic`/`prost`, which the local
//! registry does not carry. Langfuse accepts OTLP/HTTP on the same endpoint, and
//! the REST ingestion API is plain JSON over HTTPS.
//!
//! ```
//! use openjiuwen_extensions::otel::langfuse::LangfuseConfig;
//!
//! let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret")
//!     .with_host("https://langfuse.internal/");
//! assert_eq!(config.endpoint(), "https://langfuse.internal/api/public/otel");
//! assert_eq!(
//!     config.ingestion_url(),
//!     "https://langfuse.internal/api/public/ingestion",
//! );
//! let otel = config.otel_config()?;
//! assert!(otel.headers["authorization"].starts_with("Basic "));
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

pub mod batch;
pub mod client;
pub mod events;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
#[cfg(feature = "otel")]
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::config::env::var;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use url::Url;

use super::config::{ExporterType, OtelTracerConfig, OtlpProtocol};

pub use batch::{IngestionBatch, DEFAULT_MAX_BATCH_SIZE};
pub use client::{IngestionSummary, LangfuseClient};
pub use events::{GenerationBody, IngestionEvent, IngestionEventType, SpanBody, TraceBody};

/// Langfuse project public key: `LANGFUSE_PUBLIC_KEY`.
pub const LANGFUSE_PUBLIC_KEY: &str = "LANGFUSE_PUBLIC_KEY";
/// Langfuse project secret key: `LANGFUSE_SECRET_KEY`.
pub const LANGFUSE_SECRET_KEY: &str = "LANGFUSE_SECRET_KEY";
/// Langfuse host: `LANGFUSE_HOST`.
pub const LANGFUSE_HOST: &str = "LANGFUSE_HOST";
/// Langfuse host alias used by the Python SDK: `LANGFUSE_BASEURL`.
pub const LANGFUSE_BASEURL: &str = "LANGFUSE_BASEURL";

/// Langfuse Cloud host used when none is configured.
pub const DEFAULT_HOST: &str = "https://cloud.langfuse.com";

/// Path of the OTLP ingestion endpoint, appended to the host.
const OTLP_PATH: &str = "/api/public/otel";
/// Path of the REST ingestion endpoint, appended to the host.
const INGESTION_PATH: &str = "/api/public/ingestion";
/// Prefix of the REST paths, stripped from a host that already carries it.
const REST_PREFIX: &str = "/api/public";
/// `service.name` used when the host does not name one.
const DEFAULT_SERVICE_NAME: &str = "openjiuwen";

/// Langfuse connection preset.
///
/// Holds the project keys and the host; [`Self::otel_config`] turns it into a
/// tracer configuration and [`LangfuseClient::new`] into a REST ingestion
/// client. [`Debug`] never prints the secret key.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LangfuseConfig {
    /// Project public key (HTTP Basic user).
    public_key: String,
    /// Project secret key (HTTP Basic password).
    secret_key: String,
    /// Langfuse host, without a trailing slash once normalised.
    host: String,
    /// `service.name` reported with every span.
    service_name: String,
}

impl Default for LangfuseConfig {
    fn default() -> Self {
        Self {
            public_key: String::new(),
            secret_key: String::new(),
            host: DEFAULT_HOST.to_owned(),
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
        }
    }
}

impl std::fmt::Debug for LangfuseConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LangfuseConfig")
            .field("public_key", &self.public_key)
            .field("secret_key", &"<redacted>")
            .field("host", &self.host)
            .field("service_name", &self.service_name)
            .finish()
    }
}

impl LangfuseConfig {
    /// Builds a preset for the given project keys on Langfuse Cloud.
    ///
    /// No validation happens here: blank keys surface from [`Self::validate`]
    /// (or [`Self::otel_config`]) instead of making the constructor fallible.
    /// Call [`Self::with_host`] for a self-hosted instance.
    #[must_use]
    pub fn new(public_key: impl Into<String>, secret_key: impl Into<String>) -> Self {
        Self {
            public_key: public_key.into(),
            secret_key: secret_key.into(),
            host: DEFAULT_HOST.to_owned(),
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
        }
    }

    /// Returns the preset with a different host.
    ///
    /// Trailing slashes are ignored, and a host that already carries
    /// `/api/public` (or one of its REST paths) is accepted, so a copy-pasted
    /// endpoint URL keeps working.
    #[must_use]
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Returns the preset with a different `service.name`.
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Reports the configured host as given.
    #[must_use]
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Reports the configured `service.name`.
    #[must_use]
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Returns the OTLP endpoint: `{host}/api/public/otel`, with no duplicated
    /// or trailing slash.
    #[must_use]
    pub fn endpoint(&self) -> String {
        format!("{}{OTLP_PATH}", self.base())
    }

    /// Returns the REST ingestion URL: `{host}/api/public/ingestion`.
    #[must_use]
    pub fn ingestion_url(&self) -> String {
        format!("{}{INGESTION_PATH}", self.base())
    }

    /// Returns the `authorization` header value: `"Basic "` followed by the
    /// standard (padded) base64 of `"<public_key>:<secret_key>"`.
    #[must_use]
    pub fn authorization(&self) -> String {
        let credentials = format!("{}:{}", self.public_key, self.secret_key);
        format!("Basic {}", STANDARD.encode(credentials))
    }

    /// Checks that the preset can address Langfuse.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the public key, the secret key, or the
    /// `service.name` is blank, or when the host is not an absolute
    /// `http`/`https` URL.
    pub fn validate(&self) -> Result<()> {
        if self.public_key.trim().is_empty() {
            return Err(Error::validation(format!(
                "Langfuse `public_key` is required: set it on the preset or in {LANGFUSE_PUBLIC_KEY}"
            )));
        }
        if self.secret_key.trim().is_empty() {
            return Err(Error::validation(format!(
                "Langfuse `secret_key` is required: set it on the preset or in {LANGFUSE_SECRET_KEY}"
            )));
        }
        if self.service_name.trim().is_empty() {
            return Err(Error::validation(
                "Langfuse `service_name` must not be empty",
            ));
        }
        let base = self.base();
        if base.is_empty() {
            return Err(Error::validation(format!(
                "Langfuse `host` is required: set {LANGFUSE_HOST} to an absolute http(s) URL"
            )));
        }
        let parsed = Url::parse(&base).map_err(|error| {
            Error::validation(format!(
                "Langfuse `host` is not an absolute URL: {base} ({error}); set {LANGFUSE_HOST}"
            ))
        })?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(Error::validation(format!(
                "Langfuse `host` must use http or https: {base}"
            )));
        }
        Ok(())
    }

    /// Builds the OTLP/HTTP tracer configuration for Langfuse.
    ///
    /// The result selects [`ExporterType::Otlp`] with [`OtlpProtocol::Http`],
    /// targets [`Self::endpoint`], sets the `authorization` header, and reports
    /// [`Self::service_name`] as `service.name`.
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

    /// Builds an exporter that posts span batches to Langfuse.
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
    /// | `LANGFUSE_PUBLIC_KEY` | project public key (required) |
    /// | `LANGFUSE_SECRET_KEY` | project secret key (required) |
    /// | `LANGFUSE_HOST` | host, wins over the alias below |
    /// | `LANGFUSE_BASEURL` | host alias read when `LANGFUSE_HOST` is unset |
    ///
    /// Blank values count as unset; an unset host falls back to
    /// [`DEFAULT_HOST`].
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the missing variable, or the errors of
    /// [`Self::validate`].
    pub fn from_env() -> Result<Self> {
        let public_key = var(LANGFUSE_PUBLIC_KEY).ok_or_else(|| {
            Error::validation(format!("{LANGFUSE_PUBLIC_KEY} is required for Langfuse"))
        })?;
        let secret_key = var(LANGFUSE_SECRET_KEY).ok_or_else(|| {
            Error::validation(format!("{LANGFUSE_SECRET_KEY} is required for Langfuse"))
        })?;
        let host = var(LANGFUSE_HOST)
            .or_else(|| var(LANGFUSE_BASEURL))
            .unwrap_or_else(|| DEFAULT_HOST.to_owned());
        let preset = Self {
            public_key,
            secret_key,
            host,
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
        };
        preset.validate()?;
        Ok(preset)
    }

    /// Returns the host without whitespace, trailing slashes, or a copy of the
    /// REST prefix (`/api/public` and its known paths).
    fn base(&self) -> String {
        let trimmed = self.host.trim().trim_end_matches('/');
        let without_path = trimmed
            .strip_suffix(OTLP_PATH)
            .or_else(|| trimmed.strip_suffix(INGESTION_PATH))
            .or_else(|| trimmed.strip_suffix(REST_PREFIX))
            .unwrap_or(trimmed);
        without_path.trim_end_matches('/').to_owned()
    }
}
