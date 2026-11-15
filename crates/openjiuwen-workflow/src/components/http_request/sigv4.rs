//! AWS Signature Version 4 for the `aws` authentication mode.
//!
//! The component signs in pre-flight: once the body bytes are known, the canonical
//! request → string to sign → signing key → signature chain produces the `Authorization`
//! header plus the `x-amz-*` headers the request must carry. The module is pure — no I/O,
//! no clock of its own (the timestamp is a parameter) — so every vector below reproduces
//! exactly.
//!
//! # Algorithm
//!
//! ```text
//! <METHOD>\n<canonical URI>\n<canonical query>\n<canonical headers>\n<signed headers>\n<payload hash>
//! ```
//!
//! The canonical URI percent-encodes every path segment (RFC 3986, keeping `/` and the
//! unreserved set) after decoding the URL's own escaping, so a segment is encoded once.
//! The canonical query sorts the parameters by key and then value and encodes both sides.
//!
//! # Signed set
//!
//! `host`, `x-amz-date`, `x-amz-content-sha256`, `x-amz-security-token` when a session
//! token is configured, and **every header the request actually sends** are signed —
//! lowercased, trimmed, and sorted, which is also the order of the `SignedHeaders`
//! parameter.
//!
//! # Region and service
//!
//! The region and service are supplied by [`super::auth`]: `AWS_REGION` /
//! `AWS_DEFAULT_REGION` / `us-east-1` and `AWS_SERVICE` / `execute-api`.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

/// The algorithm identifier that prefixes the credential scope and the signature.
pub(crate) const ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// Header carrying the request timestamp.
pub(crate) const DATE_HEADER: &str = "x-amz-date";

/// Header carrying the payload hash.
pub(crate) const PAYLOAD_HASH_HEADER: &str = "x-amz-content-sha256";

/// Header carrying a temporary-credential session token.
pub(crate) const TOKEN_HEADER: &str = "x-amz-security-token";

/// SHA-256 of the empty payload — the payload hash of every body-less request.
pub(crate) const EMPTY_PAYLOAD_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// The timestamp format of [`DATE_HEADER`].
const AMZ_DATE_FORMAT_SUFFIX: &str = "Z";

/// Seconds in one day.
const SECONDS_PER_DAY: u64 = 86_400;

/// Upper-case hexadecimal digits of the URL encoder.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// The credentials a signature is derived from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Credentials {
    /// The access key id.
    pub(crate) access_key: String,
    /// The secret access key.
    pub(crate) secret_key: String,
    /// An optional session token of temporary credentials.
    pub(crate) session_token: Option<String>,
}

/// Everything one signature needs about the request.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SigningRequest<'a> {
    /// HTTP method; normalized to upper case.
    pub(crate) method: &'a str,
    /// The full request URL (scheme, authority, path).
    pub(crate) url: &'a str,
    /// The resolved query parameters, unencoded.
    pub(crate) query: &'a [(String, String)],
    /// Every header the request will send that must be signed.
    pub(crate) headers: &'a [(String, String)],
    /// The exact request body bytes.
    pub(crate) payload: &'a [u8],
    /// The signature-scope region.
    pub(crate) region: &'a str,
    /// The signature-scope service.
    pub(crate) service: &'a str,
    /// The instant the request is signed for, in Unix seconds.
    pub(crate) timestamp: u64,
}

/// The signature and the headers that carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Signed {
    /// Value of the `authorization` header.
    pub(crate) authorization: String,
    /// Value of the `x-amz-date` header.
    pub(crate) amz_date: String,
    /// Value of the `x-amz-content-sha256` header.
    pub(crate) payload_hash: String,
    /// Value of the `x-amz-security-token` header, when a session token is configured.
    pub(crate) session_token: Option<String>,
}

/// Signs one request.
pub(crate) fn sign(credentials: &Credentials, request: &SigningRequest<'_>) -> Signed {
    let payload_hash = if request.payload.is_empty() {
        EMPTY_PAYLOAD_SHA256.to_owned()
    } else {
        sha256_hex(request.payload)
    };
    let amz_date = format_amz_date(request.timestamp);
    let short_date = amz_date.get(..8).unwrap_or_default().to_owned();

    let parts = split_url(request.url);
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in request.headers {
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() || name == "authorization" {
            continue;
        }
        headers.insert(name, value.trim().to_owned());
    }
    headers
        .entry("host".to_owned())
        .or_insert_with(|| parts.host.to_owned());
    headers.insert(DATE_HEADER.to_owned(), amz_date.clone());
    headers.insert(PAYLOAD_HASH_HEADER.to_owned(), payload_hash.clone());
    let session_token = credentials
        .session_token
        .as_deref()
        .map(str::trim)
        .filter(|token| !token.is_empty());
    if let Some(token) = session_token {
        headers.insert(TOKEN_HEADER.to_owned(), token.to_owned());
    }

    let canonical = canonical_request(
        request.method,
        &canonical_uri(parts.path),
        request.query,
        &headers,
        &payload_hash,
    );
    let scope = format!(
        "{short_date}/{}/{}/aws4_request",
        request.region, request.service
    );
    let to_sign = string_to_sign(&amz_date, &scope, &canonical.text);
    let key = signing_key(
        &credentials.secret_key,
        &short_date,
        request.region,
        request.service,
    );
    let signature = hex_encode(&hmac_sha256(&key, to_sign.as_bytes()));

    Signed {
        authorization: format!(
            "{ALGORITHM} Credential={}/{scope}, SignedHeaders={}, Signature={signature}",
            credentials.access_key, canonical.signed_headers
        ),
        amz_date,
        payload_hash,
        session_token: session_token.map(str::to_owned),
    }
}

/// A canonical request and the `SignedHeaders` parameter that lists its headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalRequest {
    /// The canonical request text.
    pub(crate) text: String,
    /// The lower-case header names, sorted and joined by `;`.
    pub(crate) signed_headers: String,
}

/// Builds the canonical request of one signed request.
pub(crate) fn canonical_request(
    method: &str,
    canonical_uri: &str,
    query: &[(String, String)],
    headers: &BTreeMap<String, String>,
    payload_hash: &str,
) -> CanonicalRequest {
    let mut header_block = String::new();
    for (name, value) in headers {
        header_block.push_str(name);
        header_block.push(':');
        header_block.push_str(value);
        header_block.push('\n');
    }
    let signed_headers = headers.keys().cloned().collect::<Vec<String>>().join(";");
    let text = format!(
        "{}\n{canonical_uri}\n{}\n{header_block}\n{signed_headers}\n{}",
        method.trim().to_ascii_uppercase(),
        canonical_query(query),
        payload_hash.trim(),
    );
    CanonicalRequest {
        text,
        signed_headers,
    }
}

/// Builds the string to sign for `scope`.
pub(crate) fn string_to_sign(amz_date: &str, scope: &str, canonical_request: &str) -> String {
    format!(
        "{ALGORITHM}\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    )
}

/// Derives the signing key of one date/region/service triple.
pub(crate) fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> [u8; 32] {
    let initial = format!("AWS4{secret}");
    let date_key = hmac_sha256(initial.as_bytes(), date.as_bytes());
    let region_key = hmac_sha256(&date_key, region.as_bytes());
    let service_key = hmac_sha256(&region_key, service.as_bytes());
    hmac_sha256(&service_key, b"aws4_request")
}

/// Lower-case hexadecimal HMAC-SHA256 of `data` under `key` (RFC 2104, hand-rolled).
pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
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
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&outer.finalize());
    digest
}

/// Lower-case hexadecimal SHA-256 of `bytes`.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

/// Lower-case hexadecimal encoding of `bytes`.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const LOWER: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(LOWER[usize::from(byte >> 4)]));
        encoded.push(char::from(LOWER[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// The host and path components of a request URL.
struct UrlParts<'a> {
    /// `host[:port]` of the authority, without any user info.
    host: &'a str,
    /// The path, always starting with `/`.
    path: &'a str,
}

/// Splits `url` into the parts the canonical request needs.
fn split_url(url: &str) -> UrlParts<'_> {
    let without_fragment = url.split('#').next().unwrap_or(url);
    let after_scheme = without_fragment
        .split_once("://")
        .map_or(without_fragment, |(_, rest)| rest);
    let (authority, path) = after_scheme.find('/').map_or((after_scheme, "/"), |index| {
        (&after_scheme[..index], &after_scheme[index..])
    });
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let path = path.split('?').next().unwrap_or(path);
    UrlParts {
        host,
        path: if path.is_empty() { "/" } else { path },
    }
}

/// The canonical (single-encoded) form of the request path.
pub(crate) fn canonical_uri(path: &str) -> String {
    let decoded = percent_decode(path);
    let mut uri = String::with_capacity(decoded.len());
    for (index, segment) in decoded.split('/').enumerate() {
        if index > 0 {
            uri.push('/');
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
        .collect::<Vec<String>>()
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

/// Formats `seconds` since the Unix epoch as an AWS timestamp (`20150830T123600Z`).
pub(crate) fn format_amz_date(seconds: u64) -> String {
    let days = i64::try_from(seconds / SECONDS_PER_DAY).unwrap_or(i64::MAX);
    let remainder = seconds % SECONDS_PER_DAY;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}{AMZ_DATE_FORMAT_SUFFIX}",
        remainder / 3_600,
        (remainder % 3_600) / 60,
        remainder % 60
    )
}

/// Converts days since the Unix epoch into a civil date (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    (
        if month <= 2 { year + 1 } else { year },
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The AWS documentation's example secret access key.
    const SECRET: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

    /// `20150830T123600Z` as Unix seconds — the instant every AWS example uses.
    const STAMP: u64 = 1_440_938_160;

    fn headers(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn timestamps_are_formatted_in_aws_form() {
        assert_eq!(format_amz_date(STAMP), "20150830T123600Z");
        assert_eq!(format_amz_date(0), "19700101T000000Z");
        assert_eq!(format_amz_date(1_609_459_199), "20201231T235959Z");
        assert_eq!(format_amz_date(951_782_400), "20000229T000000Z");
    }

    #[test]
    fn the_aws_get_vanilla_vector_reproduces() {
        // AWS SigV4 test suite: get-vanilla (only `host` and `x-amz-date` are signed).
        let canonical = canonical_request(
            "GET",
            &canonical_uri("/"),
            &[],
            &headers(&[
                ("host", "example.amazonaws.com"),
                ("x-amz-date", "20150830T123600Z"),
            ]),
            EMPTY_PAYLOAD_SHA256,
        );
        assert_eq!(
            canonical.text,
            "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\n\
             host;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(canonical.signed_headers, "host;x-amz-date");

        let scope = "20150830/us-east-1/service/aws4_request";
        let to_sign = string_to_sign("20150830T123600Z", scope, &canonical.text);
        assert_eq!(
            to_sign,
            "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/service/aws4_request\n\
             bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63"
        );

        let key = signing_key(SECRET, "20150830", "us-east-1", "service");
        assert_eq!(
            hex_encode(&key),
            "938127b5336810ddb6a5d6af445fcac9e371f9ed418ed386b022aed82901be75"
        );
        assert_eq!(
            hex_encode(&hmac_sha256(&key, to_sign.as_bytes())),
            "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }

    #[test]
    fn a_signed_get_matches_the_fixed_vector() {
        let credentials = Credentials {
            access_key: "AKIDEXAMPLE".to_owned(),
            secret_key: SECRET.to_owned(),
            session_token: None,
        };
        let query = [
            ("b".to_owned(), "2".to_owned()),
            ("a".to_owned(), "1".to_owned()),
        ];
        let sent = [
            ("content-type".to_owned(), "application/json".to_owned()),
            ("host".to_owned(), "api.example.com".to_owned()),
        ];
        let signed = sign(
            &credentials,
            &SigningRequest {
                method: "get",
                url: "https://api.example.com/items/7/x",
                query: &query,
                headers: &sent,
                payload: b"",
                region: "us-east-1",
                service: "execute-api",
                timestamp: STAMP,
            },
        );
        assert_eq!(signed.amz_date, "20150830T123600Z");
        assert_eq!(signed.payload_hash, EMPTY_PAYLOAD_SHA256);
        assert_eq!(signed.session_token, None);
        assert_eq!(
            signed.authorization,
            "AWS4-HMAC-SHA256 \
             Credential=AKIDEXAMPLE/20150830/us-east-1/execute-api/aws4_request, \
             SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, \
             Signature=192c901ffb1dbabe5aa5bfd8422139fae25e39e76544fe11bc39b7bddc3c46a1"
        );
    }

    #[test]
    fn a_signed_post_hashes_the_payload() {
        let credentials = Credentials {
            access_key: "AKIDEXAMPLE".to_owned(),
            secret_key: SECRET.to_owned(),
            session_token: None,
        };
        let sent = [
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("Host".to_owned(), "api.example.com".to_owned()),
        ];
        let signed = sign(
            &credentials,
            &SigningRequest {
                method: "POST",
                url: "https://api.example.com/upload",
                query: &[],
                headers: &sent,
                payload: br#"{"a":1}"#,
                region: "us-east-1",
                service: "execute-api",
                timestamp: STAMP,
            },
        );
        assert_eq!(
            signed.payload_hash,
            "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
        );
        assert!(
            signed.authorization.ends_with(
                "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, \
                 Signature=94ca0e96a57ff8baae575fde58f698b6fc5efbd8a00755f5f58c1a42277d3d74"
            ),
            "unexpected authorization: {}",
            signed.authorization
        );
        assert!(
            !signed.authorization.contains("Content-Type"),
            "header names are lower-cased"
        );
    }

    #[test]
    fn a_session_token_is_signed_and_returned() {
        let credentials = Credentials {
            access_key: "AKIDEXAMPLE".to_owned(),
            secret_key: SECRET.to_owned(),
            session_token: Some("token-value".to_owned()),
        };
        let signed = sign(
            &credentials,
            &SigningRequest {
                method: "GET",
                url: "https://api.example.com/",
                query: &[],
                headers: &[],
                payload: b"",
                region: "eu-west-1",
                service: "execute-api",
                timestamp: STAMP,
            },
        );
        assert_eq!(signed.session_token.as_deref(), Some("token-value"));
        assert!(signed
            .authorization
            .contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token"));
        assert!(signed
            .authorization
            .contains("/20150830/eu-west-1/execute-api/aws4_request"));
    }

    #[test]
    fn paths_and_queries_are_canonicalized() {
        assert_eq!(canonical_uri("/items/7/x"), "/items/7/x");
        assert_eq!(canonical_uri("/"), "/");
        assert_eq!(canonical_uri("/a b/é"), "/a%20b/%C3%A9");
        assert_eq!(canonical_uri("/a%20b"), "/a%20b", "no double encoding");
        assert_eq!(canonical_uri("/a//b"), "/a//b");
        assert_eq!(
            canonical_query(&[
                ("b".to_owned(), "2 1".to_owned()),
                ("a".to_owned(), "1".to_owned()),
                ("a".to_owned(), "0".to_owned()),
            ]),
            "a=0&a=1&b=2%201"
        );
    }

    #[test]
    fn urls_are_split_into_host_and_path() {
        let parts = split_url("https://api.example.com/v1/x?q=1#frag");
        assert_eq!(parts.host, "api.example.com");
        assert_eq!(parts.path, "/v1/x");
        let parts = split_url("https://user:pw@example.com:8443");
        assert_eq!(parts.host, "example.com:8443");
        assert_eq!(parts.path, "/");
        let parts = split_url("/relative/path");
        assert_eq!(parts.host, "");
        assert_eq!(parts.path, "/relative/path");
    }

    #[test]
    fn the_empty_payload_hash_is_the_documented_constant() {
        assert_eq!(sha256_hex(b""), EMPTY_PAYLOAD_SHA256);
    }
}
