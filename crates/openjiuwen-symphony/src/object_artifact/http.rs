//! HTTP failure mapping shared by the artifact transports.
//!
//! The Google Cloud Storage transport speaks the GCS JSON API directly: the workspace ships
//! no cloud SDK, so the client is built on
//! [`openjiuwen_common::http`](openjiuwen_common::http), which reports a non-2xx status as an
//! ordinary [`HttpResponse`] and leaves the mapping to the caller. That mapping is the same
//! for every transport, so it lives here: a rejected response becomes an
//! [`Error::External`](openjiuwen_common::error::Error) naming the service, the operation, the
//! URL, the status, and a short excerpt of the body.
//!
//! The body is never echoed whole: an object-storage error document can be a multi-kilobyte
//! JSON payload (or a signed URL), and the message ends up in logs.
//!
//! Only the shared pieces live here. The GCS API is JSON-only, so no XML scanner is needed;
//! a transport that speaks XML keeps its own scanner beside its client.

use std::fmt;

use openjiuwen_common::error::Error;
use openjiuwen_common::http::HttpResponse;

/// Maximum number of response-body characters echoed in an error message.
const ERROR_BODY_EXCERPT: usize = 200;

/// Turns a rejected response into an [`Error::External`] for `service` and `operation`.
///
/// `service` is the short transport name used in diagnostics (`GCS`), and `operation` the
/// trait method that was running (`put`, `list`, …).
pub(super) fn failure(service: &str, operation: &str, response: &HttpResponse) -> Error {
    Error::external(HttpFailure {
        service: service.to_string(),
        operation: operation.to_string(),
        status: response.status,
        url: response.url.clone(),
        body: excerpt(&response.text()),
    })
}

/// Shortens a response body for an error message.
///
/// The excerpt is trimmed and truncated to [`ERROR_BODY_EXCERPT`] characters, with a trailing
/// ellipsis marking the truncation, so one line of the body is readable in a log.
fn excerpt(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "<empty body>".to_string();
    }
    let mut excerpt: String = trimmed.chars().take(ERROR_BODY_EXCERPT).collect();
    if trimmed.chars().count() > ERROR_BODY_EXCERPT {
        excerpt.push('…');
    }
    excerpt
}

/// The external error produced by a rejected object-storage request.
#[derive(Debug)]
struct HttpFailure {
    /// Short service name, for example `GCS`.
    service: String,
    /// Operation that failed, for example `put`.
    operation: String,
    /// HTTP status the service answered with.
    status: u16,
    /// URL that was requested.
    url: String,
    /// Body excerpt, which keeps the raw error document visible.
    body: String,
}

impl fmt::Display for HttpFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {} failed with HTTP {} for {}: {}",
            self.service, self.operation, self.status, self.url, self.body
        )
    }
}

impl std::error::Error for HttpFailure {}

#[cfg(test)]
// `Default::default()` builds the header map because `reqwest` is not a direct dependency of
// this crate, so its `HeaderMap` type cannot be named here.
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::default_trait_access)]
mod tests {
    use super::*;
    use bytes::Bytes;

    /// Builds a response with `status` and `body`.
    fn response(status: u16, body: &str) -> HttpResponse {
        HttpResponse {
            status,
            headers: Default::default(),
            url: "http://localhost/storage/v1/b/b/o/k".to_string(),
            body: Bytes::copy_from_slice(body.as_bytes()),
        }
    }

    #[test]
    fn an_empty_body_is_called_out() {
        assert_eq!(excerpt("   \n "), "<empty body>");
        assert_eq!(excerpt(""), "<empty body>");
    }

    #[test]
    fn long_bodies_are_truncated_with_an_ellipsis() {
        let long = "x".repeat(ERROR_BODY_EXCERPT + 50);
        let shortened = excerpt(&long);
        assert_eq!(shortened.chars().count(), ERROR_BODY_EXCERPT + 1);
        assert!(shortened.ends_with('…'));
        assert_eq!(excerpt("short"), "short");
    }

    #[test]
    fn the_failure_names_the_service_operation_status_and_url() {
        let error = failure("GCS", "get", &response(403, "{\"error\":\"denied\"}"));
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);
        let message = error.to_string();
        assert!(
            message.contains("GCS get failed with HTTP 403"),
            "{message}"
        );
        assert!(message.contains("/storage/v1/b/b/o/k"), "{message}");
        assert!(message.contains("denied"), "{message}");
    }
}
