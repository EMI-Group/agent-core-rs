//! The [`ObjectStore`] implementation over the S3 REST API.
//!
//! Every request is built here, signed by [`super::sigv4`], and executed through
//! [`openjiuwen_common::http::HttpClient`]; responses are mapped onto
//! [`ObjectMeta`] values or into [`Error`]s carrying the S3 error code.
//!
//! XML is parsed by a tolerant tag scan ([`tag_texts`]) rather than an XML crate:
//! the two S3 documents involved (`ListAllMyBucketsResult`, `ListBucketResult`)
//! are flat enough that a scan is smaller and cannot panic on malformed input.

use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use url::Url;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};

use super::sigv4::{self, SigningInput, EMPTY_PAYLOAD_SHA256};
use super::{parse_endpoint, S3Config};
use crate::store::object::{bucket_not_found, ObjectMeta, ObjectStore};

/// The AWS service name used in every signature scope.
const SERVICE: &str = "s3";

/// Header carrying the payload hash; signed with every request.
const PAYLOAD_HASH_HEADER: &str = "x-amz-content-sha256";

/// Maximum number of keys S3 returns in one `ListObjectsV2` page.
const MAX_KEYS_PER_PAGE: usize = 1000;

/// Hard cap on the pages one `list` call follows, so a misbehaving service
/// cannot make the call loop forever.
const MAX_PAGES: usize = 1000;

/// Maximum number of response-body characters echoed in an error message.
const ERROR_BODY_EXCERPT: usize = 200;

/// The HTTP verbs this backend uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verb {
    /// `GET`
    Get,
    /// `PUT`
    Put,
    /// `DELETE`
    Delete,
    /// `HEAD`
    Head,
}

impl Verb {
    /// The method name as it appears in the canonical request.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
        }
    }

    /// Starts a request for this verb.
    fn request(self, url: impl Into<String>) -> HttpRequest {
        match self {
            Self::Get => HttpRequest::get(url),
            Self::Put => HttpRequest::put(url),
            Self::Delete => HttpRequest::delete(url),
            Self::Head => HttpRequest::head(url),
        }
    }
}

/// An [`ObjectStore`] backed by an S3-compatible service.
///
/// The store holds one HTTP client and one base URL, so it is cheap to share
/// behind an `Arc` and every method takes `&self`.
///
/// # Errors
///
/// [`S3ObjectStore::new`] validates the endpoint; every operation returns the
/// errors documented on [`ObjectStore`], plus
/// [`Error::Config`](openjiuwen_common::error::Error) when credentials are
/// missing, and an external error naming the status, the URL and the body excerpt
/// when the service rejects a request.
#[derive(Debug, Clone)]
pub struct S3ObjectStore {
    /// HTTP client shared by every request.
    client: HttpClient,
    /// Configuration the store was built from.
    config: S3Config,
    /// Endpoint parsed once, so each request only sets its path and query.
    base: Url,
}

impl S3ObjectStore {
    /// Builds a store from `config`.
    ///
    /// Redirects are disabled: a Signature Version 4 signature covers the authority and the
    /// path, so following a redirect would only replay an invalid signature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint, a non-HTTP(S) scheme or a host-less URL, and propagates the
    /// HTTP client's configuration errors.
    pub fn new(config: S3Config) -> Result<Self> {
        let base = parse_endpoint(&config.endpoint)?;
        let client = HttpClient::new(
            HttpClientConfig::default()
                .with_timeout(config.timeout)
                .with_follow_redirects(false),
        )?;
        Ok(Self {
            client,
            config,
            base,
        })
    }

    /// Returns the configuration the store was built from.
    #[must_use]
    pub const fn config(&self) -> &S3Config {
        &self.config
    }

    /// The service root (`{endpoint}/`), used by `list_buckets`.
    fn root_url(&self) -> Url {
        let mut url = self.base.clone();
        url.set_query(None);
        match base_path(&self.base) {
            path if path.is_empty() => url.set_path("/"),
            path => url.set_path(&path),
        }
        url
    }

    /// The URL of `bucket`, or of `key` inside it.
    ///
    /// Visible to the parent module so the addressing tests can assert the URL
    /// shape without depending on how (or whether) a host name resolves.
    pub(super) fn url_for(&self, bucket: &str, key: Option<&str>) -> Result<Url> {
        validate_bucket(bucket)?;
        let prefix = base_path(&self.base);
        let mut url = self.base.clone();
        url.set_query(None);
        if self.config.path_style {
            let mut path = prefix;
            path.push('/');
            path.push_str(&sigv4::rfc3986_encode(bucket));
            if let Some(key) = key {
                validate_key(key)?;
                path.push('/');
                path.push_str(&encode_key_path(key));
            }
            url.set_path(&path);
            return Ok(url);
        }

        let host = self.base.host_str().unwrap_or_default().to_string();
        url.set_host(Some(&format!("{bucket}.{host}")))
            .map_err(|error| {
                Error::config(format!(
                    "S3 bucket {bucket:?} cannot be used as a virtual-host label: {error}"
                ))
            })?;
        match key {
            Some(key) => {
                validate_key(key)?;
                let mut path = prefix;
                path.push('/');
                path.push_str(&encode_key_path(key));
                url.set_path(&path);
            }
            None => {
                if prefix.is_empty() {
                    url.set_path("/");
                } else {
                    url.set_path(&prefix);
                }
            }
        }
        Ok(url)
    }

    /// Signs `method url?query` with `body` and executes it.
    ///
    /// The returned response is not interpreted: each caller decides whether a
    /// non-2xx status is a tolerated absence or a failure.
    async fn send(
        &self,
        method: Verb,
        mut url: Url,
        query: &[(String, String)],
        body: Option<Bytes>,
    ) -> Result<HttpResponse> {
        if self.config.credentials.is_anonymous() {
            return Err(Error::config(
                "S3 requests must be signed; set credentials.access_key_id and \
                 credentials.secret_access_key (or the matching AWS_* variables)",
            ));
        }
        if !query.is_empty() {
            url.set_query(Some(&encode_query(query)));
        }

        let payload_hash = body.as_ref().map_or_else(
            || EMPTY_PAYLOAD_SHA256.to_string(),
            |body| sigv4::sha256_hex(body),
        );
        let headers = vec![
            ("host".to_string(), host_header(&url)),
            (PAYLOAD_HASH_HEADER.to_string(), payload_hash.clone()),
        ];
        let signing = sigv4::sign_request(
            &self.config.credentials,
            &SigningInput {
                method: method.as_str(),
                url: &url,
                query,
                headers: &headers,
                payload_sha256_hex: &payload_hash,
                region: &self.config.region,
                service: SERVICE,
                now: Utc::now(),
            },
        );

        let mut request = method
            .request(url.as_str())
            .with_headers(signing.headers)
            .timeout(self.config.timeout);
        if let Some(body) = body {
            request = request.bytes(body);
        }
        self.client.execute(request).await
    }

    /// Signs and executes a request, mapping any non-2xx status to an error.
    async fn send_ok(
        &self,
        method: Verb,
        url: Url,
        query: &[(String, String)],
        body: Option<Bytes>,
        context: &str,
    ) -> Result<HttpResponse> {
        let response = self.send(method, url, query, body).await?;
        if response.is_success() {
            Ok(response)
        } else {
            Err(failure(&response, context))
        }
    }
}

#[async_trait]
impl ObjectStore for S3ObjectStore {
    async fn create_bucket(&self, bucket: &str) -> Result<()> {
        let url = self.url_for(bucket, None)?;
        let response = self.send(Verb::Put, url, &[], None).await?;
        // `BucketAlreadyOwnedByYou` is a success for an idempotent create.
        if response.is_success() || response.status == 409 {
            Ok(())
        } else {
            Err(failure(&response, "create_bucket"))
        }
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        let url = self.url_for(bucket, None)?;
        let response = self.send(Verb::Delete, url, &[], None).await?;
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(failure(&response, "delete_bucket"))
        }
    }

    async fn list_buckets(&self) -> Result<Vec<String>> {
        let response = self
            .send_ok(Verb::Get, self.root_url(), &[], None, "list_buckets")
            .await?;
        let mut names = tag_texts(&response.text(), "Name");
        names.sort();
        names.dedup();
        Ok(names)
    }

    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<ObjectMeta> {
        let url = self.url_for(bucket, Some(key))?;
        let body_len = data.len() as u64;
        let response = self.send_ok(Verb::Put, url, &[], Some(data), "put").await?;
        // A body-less `PUT` response carries `Content-Length: 0`; only a
        // non-zero length describes the stored object.
        let size = header_value(&response, "content-length")
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|size| *size > 0)
            .unwrap_or(body_len);
        let updated_at = header_value(&response, "last-modified")
            .as_deref()
            .and_then(parse_http_date)
            .unwrap_or_else(Utc::now);
        Ok(ObjectMeta {
            key: key.to_string(),
            size,
            updated_at,
        })
    }

    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Bytes>> {
        let url = self.url_for(bucket, Some(key))?;
        let response = self.send(Verb::Get, url, &[], None).await?;
        if response.status == 404 {
            return Ok(None);
        }
        if !response.is_success() {
            return Err(failure(&response, "get"));
        }
        Ok(Some(response.body.clone()))
    }

    async fn delete(&self, bucket: &str, key: &str) -> Result<()> {
        let url = self.url_for(bucket, Some(key))?;
        let response = self.send(Verb::Delete, url, &[], None).await?;
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(failure(&response, "delete"))
        }
    }

    async fn exists(&self, bucket: &str, key: &str) -> Result<bool> {
        let url = self.url_for(bucket, Some(key))?;
        let response = self.send(Verb::Head, url, &[], None).await?;
        if response.status == 404 {
            return Ok(false);
        }
        if response.is_success() {
            Ok(true)
        } else {
            Err(failure(&response, "exists"))
        }
    }

    async fn list(
        &self,
        bucket: &str,
        prefix: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectMeta>> {
        let url = self.url_for(bucket, None)?;
        let mut entries: Vec<ObjectMeta> = Vec::new();
        let mut token: Option<String> = None;

        for _page in 0..MAX_PAGES {
            let mut query = vec![
                ("list-type".to_string(), "2".to_string()),
                ("prefix".to_string(), prefix.to_string()),
            ];
            if let Some(token) = &token {
                query.push(("continuation-token".to_string(), token.clone()));
            }
            if let Some(limit) = limit {
                let remaining = limit
                    .saturating_sub(entries.len())
                    .clamp(1, MAX_KEYS_PER_PAGE);
                query.push(("max-keys".to_string(), remaining.to_string()));
            }

            let response = self.send(Verb::Get, url.clone(), &query, None).await?;
            if response.status == 404 {
                return Err(bucket_not_found(bucket));
            }
            if !response.is_success() {
                return Err(failure(&response, "list"));
            }
            let page = response.text();
            append_entries(&mut entries, &page);

            let truncated = first_tag(&page, "IsTruncated")
                .is_some_and(|value| value.eq_ignore_ascii_case("true"));
            if !truncated || limit.is_some_and(|limit| entries.len() >= limit) {
                break;
            }
            match first_tag(&page, "NextContinuationToken") {
                Some(next) if !next.is_empty() && token.as_deref() != Some(next.as_str()) => {
                    token = Some(next);
                }
                _ => break,
            }
        }

        entries.sort_by(|left, right| left.key.cmp(&right.key));
        if let Some(limit) = limit {
            entries.truncate(limit);
        }
        Ok(entries)
    }
}

/// The endpoint's path without its trailing slash (empty for a host-only endpoint).
fn base_path(base: &Url) -> String {
    base.path().trim_end_matches('/').to_string()
}

/// The `host[:port]` value of `url`, with the default port elided.
fn host_header(url: &Url) -> String {
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    }
}

/// Encodes query parameters into an RFC 3986 query string.
fn encode_query(query: &[(String, String)]) -> String {
    query
        .iter()
        .map(|(key, value)| {
            format!(
                "{}={}",
                sigv4::rfc3986_encode(key),
                sigv4::rfc3986_encode(value)
            )
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// Percent-encodes one object key segment by segment, keeping `/` separators.
fn encode_key_path(key: &str) -> String {
    key.split('/')
        .map(sigv4::rfc3986_encode)
        .collect::<Vec<_>>()
        .join("/")
}

/// Rejects a bucket name that cannot appear in a path or a host label.
fn validate_bucket(bucket: &str) -> Result<()> {
    if bucket.trim().is_empty() {
        return Err(Error::validation("S3 bucket name must not be empty"));
    }
    if bucket
        .chars()
        .any(|c| matches!(c, '/' | '?' | '#' | ' ') || c.is_control())
    {
        return Err(Error::validation(format!(
            "invalid S3 bucket name {bucket:?}"
        )));
    }
    Ok(())
}

/// Rejects an empty object key.
fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() {
        return Err(Error::validation("S3 object key must not be empty"));
    }
    Ok(())
}

/// Returns one response header as a trimmed string.
fn header_value(response: &HttpResponse, name: &str) -> Option<String> {
    response
        .headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
}

/// Parses an HTTP date (`Last-Modified`) or an ISO-8601 timestamp.
fn parse_http_date(value: &str) -> Option<DateTime<Utc>> {
    let value = value.trim();
    DateTime::parse_from_rfc2822(value)
        .or_else(|_| DateTime::parse_from_rfc3339(value))
        .ok()
        .map(|date| date.with_timezone(&Utc))
}

/// Appends every `<Contents>` entry of one `ListBucketResult` page.
fn append_entries(entries: &mut Vec<ObjectMeta>, page: &str) {
    for block in tag_texts(page, "Contents") {
        let Some(key) = first_tag(&block, "Key").filter(|key| !key.is_empty()) else {
            continue;
        };
        let size = first_tag(&block, "Size")
            .and_then(|size| size.parse::<u64>().ok())
            .unwrap_or(0);
        let updated_at = first_tag(&block, "LastModified")
            .as_deref()
            .and_then(parse_http_date)
            .unwrap_or_else(Utc::now);
        entries.push(ObjectMeta {
            key,
            size,
            updated_at,
        });
    }
}

/// Returns the first unescaped `<tag>…</tag>` text of a document.
fn first_tag(xml: &str, tag: &str) -> Option<String> {
    tag_texts(xml, tag).into_iter().next()
}

/// Returns the unescaped text of every `<tag>…</tag>` occurrence, in order.
///
/// The scan is tolerant: attributes are ignored, self-closing and unclosed tags
/// are skipped, and malformed input yields whatever was parsed so far instead of
/// panicking.
fn tag_texts(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut texts = Vec::new();
    let mut rest = xml;

    while let Some(start) = rest.find(&open) {
        let after_name = &rest[start + open.len()..];
        let Some(end_of_name) = after_name.find('>') else {
            break;
        };
        let boundary = after_name.chars().next();
        let is_this_element =
            matches!(boundary, Some('>' | '/')) || boundary.is_some_and(char::is_whitespace);
        let inner = &after_name[end_of_name + 1..];
        if !is_this_element {
            // A longer tag name (`<TagOther>`): keep scanning after its `>`.
            rest = inner;
            continue;
        }
        match inner.find(&close) {
            Some(end) => {
                texts.push(unescape(inner[..end].trim()));
                rest = &inner[end + close.len()..];
            }
            None => rest = inner,
        }
    }
    texts
}

/// Replaces the five XML entities S3 emits, leaving unknown entities untouched.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        out.push_str(&rest[..position]);
        let tail = &rest[position..];
        let Some(semicolon) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let entity = &tail[1..semicolon];
        let replacement = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => None,
        };
        if let Some(character) = replacement {
            out.push(character);
            rest = &tail[semicolon + 1..];
        } else {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Shortens a response body for an error message.
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

/// The external error produced by a rejected S3 request.
#[derive(Debug)]
struct S3Error {
    /// Operation that failed, for example `put`.
    context: String,
    /// HTTP status the service answered with.
    status: u16,
    /// URL that was requested.
    url: String,
    /// `<Code>` of the S3 error document, when present.
    code: Option<String>,
    /// `<Message>` of the S3 error document, when present.
    message: Option<String>,
    /// Body excerpt, which keeps the raw XML tags visible.
    body: String,
}

impl S3Error {
    /// Builds the error from a failed response.
    fn from_response(response: &HttpResponse, context: &str) -> Self {
        let text = response.text();
        Self {
            context: context.to_string(),
            status: response.status,
            url: response.url.clone(),
            code: first_tag(&text, "Code"),
            message: first_tag(&text, "Message"),
            body: excerpt(&text),
        }
    }
}

impl fmt::Display for S3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "S3 {} failed with HTTP {} for {}",
            self.context, self.status, self.url
        )?;
        if let Some(code) = &self.code {
            write!(formatter, ": {code}")?;
        }
        if let Some(message) = &self.message {
            write!(formatter, " ({message})")?;
        }
        write!(formatter, ": {}", self.body)
    }
}

impl std::error::Error for S3Error {}

/// Turns a failed response into the external S3 error.
fn failure(response: &HttpResponse, context: &str) -> Error {
    Error::external(S3Error::from_response(response, context))
}
