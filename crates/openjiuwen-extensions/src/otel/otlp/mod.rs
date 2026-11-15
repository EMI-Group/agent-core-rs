//! OTLP/HTTP span exporter with JSON and protobuf payloads.
//!
//! Ports the OTLP exporter wiring of `openjiuwen/extensions/tracer_otel/setup.py`
//! (v0.1.20). Both OTLP/HTTP encodings are supported: `http/json`
//! (`Content-Type: application/json`, [`OtlpEncoding::Json`], the default) and
//! `http/protobuf` (`application/x-protobuf`, [`OtlpEncoding::Protobuf`]).
//! Protobuf bodies come from the hand-written encoder in `protobuf.rs`, so
//! neither `prost` nor `opentelemetry-otlp` is required. gRPC stays
//! unsupported: it needs `tonic`/`prost` plus an HTTP/2 stack that this offline
//! workspace does not carry.
//!
//! # Runtime model
//!
//! The SDK's `BatchSpanProcessor` calls [`SpanExporter::export`] on a dedicated
//! `std::thread` that drives the future with `futures_executor::block_on`, so
//! that thread has **no tokio reactor**. A plain async `reqwest` call made there
//! panics ("there is no reactor running"). [`OtlpHttpExporter`] therefore owns
//! its HTTP execution context: one `std::thread` (named
//! `openjiuwen-otlp-http`) builds a current-thread tokio runtime and the
//! `reqwest::Client`, and runs the POSTs on that runtime. Jobs travel over an
//! unbounded tokio channel with a oneshot reply, so `export` awaits the outcome
//! under whatever executor called it — the batch processor's `block_on`, a tokio
//! task, or a plain test.
//!
//! ```
//! use std::collections::BTreeMap;
//! use std::time::Duration;
//!
//! use openjiuwen_extensions::otel::otlp::{OtlpEncoding, OtlpHttpExporter};
//!
//! let exporter = OtlpHttpExporter::new(
//!     "http://localhost:4318",
//!     BTreeMap::from([(String::from("authorization"), String::from("Bearer token"))]),
//!     Duration::from_secs(5),
//! )?;
//! assert_eq!(exporter.traces_url(), "http://localhost:4318/v1/traces");
//! assert_eq!(exporter.content_type(), "application/json");
//!
//! let protobuf = OtlpHttpExporter::with_encoding(
//!     "http://localhost:4318",
//!     BTreeMap::new(),
//!     Duration::from_secs(5),
//!     OtlpEncoding::Protobuf,
//! )?;
//! assert_eq!(protobuf.content_type(), "application/x-protobuf");
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```
//!
//! [`SpanExporter::export`]: opentelemetry_sdk::trace::SpanExporter::export

mod endpoint;
// `pub(crate)` so the collector-free file exporter can reuse the same encoder
// instead of carrying a second OTLP JSON implementation (no public API change).
pub(crate) mod json;
mod model;
mod protobuf;
#[cfg(test)]
mod tests;
mod transport;

pub use endpoint::traces_url;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use opentelemetry_sdk::Resource;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use tracing::warn;
use url::Url;

use transport::{Job, JobSender, Worker};

/// `tracing` target for exporter-level failures.
const LOG_TARGET: &str = "openjiuwen_extensions::otel::otlp";

/// Message reported when a batch is exported after shutdown.
const SHUTDOWN_EXPORT: &str = "the OTLP HTTP exporter has already been shut down";

/// Message reported when the worker thread is gone.
const WORKER_GONE: &str = "the OTLP HTTP worker is not running";

/// Message reported when the worker dropped the reply channel.
const REPLY_LOST: &str = "the OTLP HTTP worker dropped the export result";

/// Body encoding of an OTLP/HTTP export.
///
/// [`OtlpEncoding::Json`] is the OTLP `http/json` encoding and the default;
/// [`OtlpEncoding::Protobuf`] is `http/protobuf`. The choice drives both the
/// `Content-Type` request header and the encoder used for every batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OtlpEncoding {
    /// `application/json`: the OTLP `http/json` encoding.
    #[default]
    Json,
    /// `application/x-protobuf`: the OTLP `http/protobuf` encoding.
    Protobuf,
}

impl OtlpEncoding {
    /// The HTTP `Content-Type` of a body in this encoding.
    #[must_use]
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Protobuf => "application/x-protobuf",
        }
    }
}

/// Exports finished spans to an OTLP/HTTP collector as JSON or protobuf.
///
/// Construction starts the worker thread but performs no I/O; the first export
/// opens the connection. The exporter is `Send + Sync`, so the SDK can own it
/// behind `&self` for the whole provider lifetime, and it never panics: every
/// failure surfaces as [`OTelSdkError::InternalFailure`] and a `tracing::warn!`
/// line.
pub struct OtlpHttpExporter {
    /// The fully-qualified traces URL every batch is posted to.
    traces_url: String,
    /// Headers sent with every request (values are never logged).
    headers: HeaderMap,
    /// Per-request timeout handed to the `reqwest::Client`.
    timeout: Duration,
    /// The body encoding, which also fixes the `Content-Type`.
    encoding: OtlpEncoding,
    /// The resource the provider handed over in `set_resource`.
    resource: Mutex<Option<Resource>>,
    /// Job channel; `None` once shut down closes the channel.
    sender: Option<JobSender>,
    /// The worker thread; behind a mutex so `&self` stays `Sync`.
    worker: Mutex<Option<Worker>>,
}

impl OtlpHttpExporter {
    /// Builds a JSON-encoding exporter for `endpoint` and starts its worker
    /// thread.
    ///
    /// `endpoint` may be a base URL (`http://host:4318`) or a complete traces
    /// URL; the `/v1/traces` suffix is appended when absent. `headers` are sent
    /// verbatim with every request, except `content-type`, which is always the
    /// encoding's (`application/json` here). `timeout` bounds each HTTP
    /// request. This is [`Self::with_encoding`] with [`OtlpEncoding::Json`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `endpoint` is empty, is not an absolute
    /// `http`/`https` URL, or when a header name or value is not a valid HTTP
    /// header. [`Error::Config`] when the worker thread cannot be started.
    ///
    /// [`Error::Validation`]: openjiuwen_common::error::Error::Validation
    /// [`Error::Config`]: openjiuwen_common::error::Error::Config
    pub fn new(
        endpoint: impl Into<String>,
        headers: BTreeMap<String, String>,
        timeout: Duration,
    ) -> Result<Self> {
        Self::with_encoding(endpoint, headers, timeout, OtlpEncoding::Json)
    }

    /// Builds an exporter that encodes its bodies as `encoding` and starts its
    /// worker thread.
    ///
    /// Endpoint and header validation, normalization, and worker startup are
    /// those of [`Self::new`]. `content-type` is always
    /// [`OtlpEncoding::content_type`] and cannot be overridden through
    /// `headers`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `endpoint` is empty, is not an absolute
    /// `http`/`https` URL, or when a header name or value is not a valid HTTP
    /// header. [`Error::Config`] when the worker thread cannot be started.
    ///
    /// [`Error::Validation`]: openjiuwen_common::error::Error::Validation
    /// [`Error::Config`]: openjiuwen_common::error::Error::Config
    pub fn with_encoding(
        endpoint: impl Into<String>,
        headers: BTreeMap<String, String>,
        timeout: Duration,
        encoding: OtlpEncoding,
    ) -> Result<Self> {
        let endpoint = endpoint.into();
        let endpoint = endpoint.trim();
        if endpoint.is_empty() {
            return Err(Error::validation(
                "the OTLP HTTP exporter requires a non-empty endpoint",
            ));
        }
        let parsed = Url::parse(endpoint).map_err(|error| {
            Error::validation(format!(
                "the OTLP HTTP exporter endpoint is not an absolute URL: {endpoint} ({error})"
            ))
        })?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(Error::validation(format!(
                "the OTLP HTTP exporter endpoint must use http or https: {endpoint}"
            )));
        }
        let traces_url = endpoint::traces_url(endpoint);
        let headers = header_map(headers)?;
        let (worker, sender) = Worker::spawn(traces_url.clone(), headers.clone(), timeout)
            .map_err(|error| {
                Error::config(format!(
                    "failed to start the OTLP HTTP worker thread: {error}"
                ))
            })?;
        Ok(Self {
            traces_url,
            headers,
            timeout,
            encoding,
            resource: Mutex::new(None),
            sender: Some(sender),
            worker: Mutex::new(Some(worker)),
        })
    }

    /// The fully-qualified URL this exporter posts to; always ends in
    /// `/v1/traces`.
    #[must_use]
    pub fn traces_url(&self) -> &str {
        &self.traces_url
    }

    /// The `Content-Type` sent with every request, taken from the encoding.
    #[must_use]
    pub fn content_type(&self) -> &'static str {
        self.encoding.content_type()
    }

    /// Encodes the batch with the cached resource into the exporter's encoding.
    fn body(&self, batch: &[SpanData]) -> std::result::Result<Vec<u8>, OTelSdkError> {
        let resource = lock(&self.resource).clone();
        match self.encoding {
            OtlpEncoding::Json => serde_json::to_vec(&json::encode_body(resource.as_ref(), batch))
                .map_err(|error| {
                    OTelSdkError::InternalFailure(format!(
                        "the OTLP JSON body could not be serialized: {error}"
                    ))
                }),
            OtlpEncoding::Protobuf => Ok(protobuf::encode_body(resource.as_ref(), batch)),
        }
    }

    /// Reports whether the worker still accepts batches.
    fn is_shutdown(&self) -> bool {
        self.sender.is_none()
    }
}

impl fmt::Debug for OtlpHttpExporter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header *values* may hold credentials, so only the names are shown.
        let header_names: Vec<&str> = self.headers.keys().map(HeaderName::as_str).collect();
        formatter
            .debug_struct("OtlpHttpExporter")
            .field("traces_url", &self.traces_url)
            .field("timeout", &self.timeout)
            .field("headers", &header_names)
            .field("encoding", &self.encoding)
            .field("shutdown", &self.is_shutdown())
            .finish_non_exhaustive()
    }
}

impl SpanExporter for OtlpHttpExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        if batch.is_empty() {
            return Ok(());
        }
        let Some(sender) = self.sender.clone() else {
            warn!(target: LOG_TARGET, "rejected a span batch: {SHUTDOWN_EXPORT}");
            return Err(OTelSdkError::InternalFailure(SHUTDOWN_EXPORT.to_owned()));
        };
        let body = self.body(&batch)?;
        let (reply, response) = tokio::sync::oneshot::channel();
        if sender
            .send(Job {
                body,
                content_type: self.encoding.content_type(),
                reply,
            })
            .is_err()
        {
            warn!(target: LOG_TARGET, "rejected a span batch: {WORKER_GONE}");
            return Err(OTelSdkError::InternalFailure(WORKER_GONE.to_owned()));
        }
        match response.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(message)) => {
                warn!(target: LOG_TARGET, "span export failed: {message}");
                Err(OTelSdkError::InternalFailure(message))
            }
            Err(_) => {
                warn!(target: LOG_TARGET, "span export failed: {REPLY_LOST}");
                Err(OTelSdkError::InternalFailure(REPLY_LOST.to_owned()))
            }
        }
    }

    /// Stops the worker, waiting at most `timeout` for it to finish.
    ///
    /// Idempotent: a second call returns `Ok(())` immediately because the job
    /// sender and the worker handle are already gone. If the worker does not
    /// stop in time it is detached; its in-flight request is still bounded by
    /// the client timeout.
    fn shutdown_with_timeout(&mut self, timeout: Duration) -> OTelSdkResult {
        // Dropping the sender closes the channel; the worker exits after the
        // job in flight, which is what the completion signal reports.
        self.sender = None;
        let worker = lock(&self.worker).take();
        if let Some(mut worker) = worker {
            worker.shutdown(timeout);
        }
        Ok(())
    }

    /// Nothing is buffered in the exporter, so a flush is always a no-op.
    fn force_flush(&mut self) -> OTelSdkResult {
        Ok(())
    }

    fn set_resource(&mut self, resource: &Resource) {
        *lock(&self.resource) = Some(resource.clone());
    }
}

/// Parses the configured headers, rejecting invalid names and values.
fn header_map(headers: BTreeMap<String, String>) -> Result<HeaderMap> {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
            Error::validation(format!(
                "the OTLP HTTP exporter header name is invalid: {name}"
            ))
        })?;
        let header_value = HeaderValue::from_str(&value).map_err(|_| {
            Error::validation(format!(
                "the OTLP HTTP exporter header value for {name} is invalid"
            ))
        })?;
        map.insert(header_name, header_value);
    }
    Ok(map)
}

/// Locks a mutex, recovering the data when a previous holder panicked.
///
/// Exporter callbacks are fail-open: a poisoned lock must not turn span export
/// into a panic on the batch-processor thread.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
