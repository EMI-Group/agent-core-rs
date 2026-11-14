//! HTTP failure mapping and signing helpers shared by the artifact transports.
//!
//! The Google Cloud Storage and Azure Blob transports speak their service APIs directly: the
//! workspace ships no cloud SDK, so the clients are built on
//! [`openjiuwen_common::http`](openjiuwen_common::http), which reports a non-2xx status as an
//! ordinary [`HttpResponse`] and leaves the mapping to the caller. That mapping is the same for
//! every transport, so it lives here: a rejected response becomes an
//! [`Error::External`](openjiuwen_common::error::Error) naming the service, the operation, the
//! URL, the status, and a short excerpt of the body.
//!
//! The body is never echoed whole: an object-storage error document can be a multi-kilobyte
//! JSON payload (or a signed URL), and the message ends up in logs.
//!
//! Azure Blob Shared Key authorization needs an HMAC-SHA256, and the workspace deliberately
//! carries no HMAC crate (the only lock entry is transitive), so [`hmac_sha256`] builds one from
//! `sha2` — the same construction the S3 Signature Version 4 signer uses.
//!
//! Only the shared pieces live here. The GCS API is JSON-only, so no XML scanner is needed; the
//! Azure Blob API speaks XML, so its scanner sits beside its client.

use std::fmt;

use openjiuwen_common::error::Error;
use openjiuwen_common::http::HttpResponse;
use sha2::{Digest, Sha256};

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

/// Computes an HMAC-SHA256 over `data` with `key` (RFC 2104).
///
/// The workspace has no HMAC crate, so the construction is spelled out: the key is padded (or
/// hashed first when it is longer than one block), combined with the inner and outer pads, and the
/// two SHA-256 digests are chained. Azure Blob Shared Key authorization signs with exactly this.
pub(super) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;

    let mut normalized = [0_u8; BLOCK];
    if key.len() > BLOCK {
        normalized[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        normalized[..key.len()].copy_from_slice(key);
    }

    let mut inner_pad = [0x36_u8; BLOCK];
    let mut outer_pad = [0x5c_u8; BLOCK];
    for index in 0..BLOCK {
        inner_pad[index] ^= normalized[index];
        outer_pad[index] ^= normalized[index];
    }

    let mut inner = Sha256::new();
    inner.update(inner_pad);
    inner.update(data);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner_digest);
    let digest = outer.finalize();

    let mut result = [0_u8; 32];
    result.copy_from_slice(&digest);
    result
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

    /// Asserts one RFC 4231 test vector: key and data in hexadecimal, digest in hexadecimal.
    fn assert_vector(case: usize, key: &str, data: &str, expected: &str) {
        let key = decode_hex(key);
        let data = decode_hex(data);
        let digest = hmac_sha256(&key, &data);
        assert_eq!(encode_hex(&digest), expected, "RFC 4231 case {case}");
    }

    #[test]
    fn hmac_sha256_matches_the_rfc4231_test_vectors() {
        // Cases 1-4 of RFC 4231 §4 ("Test Cases for HMAC-SHA-256").
        assert_vector(
            1,
            "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
            "4869205468657265",
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
        );
        assert_vector(
            2,
            "4a656665",
            "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
        );
        assert_vector(
            3,
            &"aa".repeat(20),
            &"dd".repeat(50),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
        );
        assert_vector(
            4,
            "0102030405060708090a0b0c0d0e0f10111213141516171819",
            &"cd".repeat(50),
            "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
        );
    }

    /// Decodes a hexadecimal string used by the RFC 4231 vectors.
    fn decode_hex(text: &str) -> Vec<u8> {
        let bytes = text.as_bytes();
        let mut decoded = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            let high = char::from(chunk[0]).to_digit(16).expect("hex digit");
            let low = char::from(chunk[1]).to_digit(16).expect("hex digit");
            decoded.push(u8::try_from(high * 16 + low).expect("byte"));
        }
        decoded
    }

    /// Encodes bytes as lowercase hexadecimal.
    fn encode_hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        bytes.iter().fold(
            String::with_capacity(bytes.len() * 2),
            |mut encoded, byte| {
                let _ = write!(encoded, "{byte:02x}");
                encoded
            },
        )
    }
}
