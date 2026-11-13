//! AWS Signature Version 4, implemented in-crate and deterministically.
//!
//! [`sign_request`] turns one HTTP request description into the
//! `Authorization` header and the `x-amz-*` headers an S3-compatible service
//! expects. It is pure: no I/O, no clock of its own — the timestamp comes from
//! [`SigningInput::now`], so known-answer vectors reproduce exactly.
//!
//! # Algorithm
//!
//! Canonical request → string to sign → signing key → signature, exactly as the
//! AWS documentation describes it:
//!
//! ```text
//! <METHOD>\n<canonical URI>\n<canonical query>\n<canonical headers>\n<signed headers>\n<payload hash>
//! ```
//!
//! The canonical URI percent-encodes every path segment per RFC 3986 (keeping
//! `/` and the unreserved `A-Z a-z 0-9 - . _ ~` set, never `+` for a space) after
//! decoding the URL's own escaping, so a segment is encoded exactly once — S3
//! signs single-encoded paths, unlike the other AWS services.
//! The canonical query sorts the request's parameters by key and then by value
//! and RFC 3986-encodes both sides.
//!
//! # Which headers are signed
//!
//! The signed set is every header this module will send, lowercased and sorted:
//! the caller's `headers` (minus any `authorization`), plus `host`, plus
//! `x-amz-date`, plus `x-amz-security-token` when a session token is configured.
//! `x-amz-content-sha256` is always *emitted* (from [`SigningInput::payload_sha256_hex`]
//! unless the caller supplied its own value) but it only joins the signed set when
//! the caller lists it in `headers`: the AWS `get-vanilla` vector signs only
//! `host;x-amz-date`, whereas a real S3 request must sign the payload hash and
//! therefore passes it in `headers` (which [`S3ObjectStore`](super::S3ObjectStore)
//! always does).
//!
//! Header values are trimmed and empty ones are kept as empty, matching the AWS
//! canonicalization rules. Duplicate header names collapse to the last value.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use url::Url;

use openjiuwen_common::security::SecretString;

/// SHA-256 of the empty payload — the payload hash of every body-less request.
pub const EMPTY_PAYLOAD_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Name of the header carrying the payload hash.
const PAYLOAD_HASH_HEADER: &str = "x-amz-content-sha256";

/// Name of the header carrying the request timestamp.
const DATE_HEADER: &str = "x-amz-date";

/// Name of the header carrying a temporary-credential session token.
const TOKEN_HEADER: &str = "x-amz-security-token";

/// The algorithm identifier that prefixes both the scope and the signature.
const ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// Upper-case hexadecimal digits used by every encoder in this module.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// Long-form timestamp format of [`DATE_HEADER`] (`20150830T123600Z`).
const AMZ_DATE_FORMAT: &str = "%Y%m%dT%H%M%SZ";

/// Credentials of one S3 principal.
///
/// `secret_access_key` and `session_token` are [`SecretString`]s: they never
/// appear in `Debug` output, yet they serialize transparently so a configuration
/// round-trips through JSON.
///
/// ```
/// use openjiuwen_foundation::store::external::s3::Credentials;
///
/// let credentials = Credentials::new("AKIDEXAMPLE", "s3cr3t-key").with_session_token("token-value");
/// assert_eq!(credentials.access_key_id, "AKIDEXAMPLE");
/// // The secrets are redacted in `Debug` output.
/// assert!(!format!("{credentials:?}").contains("s3cr3t-key"));
/// assert!(!format!("{credentials:?}").contains("token-value"));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Credentials {
    /// Access key id, the first half of the `Credential=` scope.
    pub access_key_id: String,
    /// Secret access key used to derive the signing key.
    pub secret_access_key: SecretString,
    /// Optional session token of temporary credentials.
    pub session_token: Option<SecretString>,
}

impl Credentials {
    /// Builds a credential pair from an access key id and its secret.
    #[must_use]
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: SecretString::new(secret_access_key),
            session_token: None,
        }
    }

    /// Attaches a session token, sent as `x-amz-security-token`.
    #[must_use]
    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(SecretString::new(token));
        self
    }

    /// Reports whether no usable credential pair is configured.
    ///
    /// An empty access key id or secret means "anonymous", which a signature
    /// cannot express; the stores reject such a configuration instead of sending
    /// an unsigned request.
    #[must_use]
    pub fn is_anonymous(&self) -> bool {
        self.access_key_id.trim().is_empty() || self.secret_access_key.is_empty()
    }
}

/// Everything [`sign_request`] needs about one request.
///
/// `query` is the authoritative list of query parameters (the signer does not
/// re-parse [`SigningInput::url`]'s query string), and `headers` carries the
/// caller's own headers — the signer adds `host` when it is absent.
#[derive(Debug, Clone, Copy)]
pub struct SigningInput<'a> {
    /// HTTP method, for example `GET`; case is normalized to upper case.
    pub method: &'a str,
    /// Request URL; its path and authority feed the canonical request.
    pub url: &'a Url,
    /// Query parameters of the request, unencoded.
    pub query: &'a [(String, String)],
    /// Headers the caller will send, unencoded values.
    pub headers: &'a [(String, String)],
    /// Lower-case hexadecimal SHA-256 of the request body.
    pub payload_sha256_hex: &'a str,
    /// AWS region of the signature scope, for example `us-east-1`.
    pub region: &'a str,
    /// AWS service of the signature scope; `s3` for this backend.
    pub service: &'a str,
    /// The instant the request is signed for.
    pub now: DateTime<Utc>,
}

/// The signature and the headers that carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigningResult {
    /// Value of the `authorization` header.
    pub authorization: String,
    /// Every signed header to send, lowercased names: `authorization`,
    /// `x-amz-date`, `x-amz-content-sha256`, and — when a session token is
    /// configured — `x-amz-security-token`.
    pub headers: Vec<(String, String)>,
    /// The canonical request the signature covers.
    pub canonical_request: String,
    /// The string that was signed.
    pub string_to_sign: String,
    /// Lower-case hexadecimal signature.
    pub signature: String,
}

/// Signs one request and returns the headers that carry the signature.
///
/// Every input is a plain string, so the function cannot fail; a malformed URL
/// simply feeds whatever authority and path it has into the canonical request.
#[must_use]
pub fn sign_request(credentials: &Credentials, input: &SigningInput<'_>) -> SigningResult {
    let canonical = canonical_headers(credentials, input);
    let amz_date = canonical
        .get(DATE_HEADER)
        .cloned()
        .unwrap_or_else(|| input.now.format(AMZ_DATE_FORMAT).to_string());
    let short_date = short_date(&amz_date, input.now);

    let canonical_uri = canonical_uri(input.url);
    let canonical_query = canonical_query(input.query);

    let mut header_block = String::new();
    for (name, value) in &canonical {
        header_block.push_str(name);
        header_block.push(':');
        header_block.push_str(value);
        header_block.push('\n');
    }
    let signed_headers = canonical
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        input.method.trim().to_ascii_uppercase(),
        canonical_uri,
        canonical_query,
        header_block,
        signed_headers,
        input.payload_sha256_hex.trim(),
    );

    let scope = format!(
        "{short_date}/{}/{}/aws4_request",
        input.region, input.service
    );
    let string_to_sign = format!(
        "{ALGORITHM}\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let key = signing_key(
        credentials.secret_access_key.expose_secret(),
        &short_date,
        input.region,
        input.service,
    );
    let signature = hex_encode(&hmac_sha256(&key, string_to_sign.as_bytes()));
    let authorization = format!(
        "{ALGORITHM} Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        credentials.access_key_id,
    );

    let mut headers = vec![
        ("authorization".to_string(), authorization.clone()),
        (DATE_HEADER.to_string(), amz_date),
        (
            PAYLOAD_HASH_HEADER.to_string(),
            canonical
                .get(PAYLOAD_HASH_HEADER)
                .cloned()
                .unwrap_or_else(|| input.payload_sha256_hex.trim().to_string()),
        ),
    ];
    if let Some(token) = canonical.get(TOKEN_HEADER) {
        headers.push((TOKEN_HEADER.to_string(), token.clone()));
    }

    SigningResult {
        authorization,
        headers,
        canonical_request,
        string_to_sign,
        signature,
    }
}

/// Builds the canonical header map: caller headers plus the headers the signer
/// adds, lowercased and sorted.
fn canonical_headers(
    credentials: &Credentials,
    input: &SigningInput<'_>,
) -> BTreeMap<String, String> {
    let mut canonical: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in input.headers {
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() || name == "authorization" {
            continue;
        }
        canonical.insert(name, value.trim().to_string());
    }
    canonical
        .entry("host".to_string())
        .or_insert_with(|| host_header(input.url));
    canonical
        .entry(DATE_HEADER.to_string())
        .or_insert_with(|| input.now.format(AMZ_DATE_FORMAT).to_string());
    if let Some(token) = &credentials.session_token {
        if !token.is_empty() {
            canonical.insert(
                TOKEN_HEADER.to_string(),
                token.expose_secret().trim().to_string(),
            );
        }
    }
    canonical
}

/// The `host[:port]` value of `url`, with the default port elided.
fn host_header(url: &Url) -> String {
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    }
}

/// The date portion (`yyyyMMdd`) of the request timestamp.
fn short_date(amz_date: &str, now: DateTime<Utc>) -> String {
    match amz_date.get(..8) {
        Some(date) if date.chars().all(|c| c.is_ascii_digit()) => date.to_string(),
        _ => now.format("%Y%m%d").to_string(),
    }
}

/// Derives the signing key for one date/region/service triple.
fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> [u8; 32] {
    let initial = format!("AWS4{secret}");
    let date_key = hmac_sha256(initial.as_bytes(), date.as_bytes());
    let region_key = hmac_sha256(&date_key, region.as_bytes());
    let service_key = hmac_sha256(&region_key, service.as_bytes());
    hmac_sha256(&service_key, b"aws4_request")
}

/// HMAC-SHA256 of `data` under `key`.
///
/// The implementation is the standard RFC 2104 construction over two SHA-256
/// passes with a 64-byte block, built by hand because no `hmac` crate is
/// available offline.
#[must_use]
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
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
    let outer_digest = outer.finalize();

    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&outer_digest);
    digest
}

/// Lower-case hexadecimal SHA-256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

/// Lower-case hexadecimal encoding of `bytes`.
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
    const LOWER: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(LOWER[usize::from(byte >> 4)]));
        encoded.push(char::from(LOWER[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// The canonical (single-encoded) path of `url`.
fn canonical_uri(url: &Url) -> String {
    let decoded = percent_decode(url.path());
    let mut uri = String::with_capacity(decoded.len());
    let mut leading = true;
    for segment in decoded.split('/') {
        if leading {
            leading = false;
        } else {
            uri.push('/');
        }
        if segment.is_empty() {
            continue;
        }
        uri.push_str(&rfc3986_encode(segment));
    }
    if uri.is_empty() {
        uri.push('/');
    }
    uri
}

/// The canonical query string: parameters sorted by key then value, RFC 3986 encoded.
fn canonical_query(query: &[(String, String)]) -> String {
    let mut pairs: Vec<(String, String)> = query
        .iter()
        .map(|(key, value)| (rfc3986_encode(key), rfc3986_encode(value)))
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// RFC 3986 percent-encoding of `value` (unreserved characters stay literal).
pub(crate) fn rfc3986_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
    }
    encoded
}

/// Decodes `%XX` escapes, leaving malformed sequences untouched.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let (Some(high), Some(low)) =
                (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
            {
                decoded.push(high * 16 + low);
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// The numeric value of one hexadecimal digit.
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
