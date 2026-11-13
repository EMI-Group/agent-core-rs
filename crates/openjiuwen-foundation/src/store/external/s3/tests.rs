//! Offline tests for the S3 object store and the in-crate Signature Version 4 signer.
//!
//! The signer is pinned by known-answer vectors from the AWS signature test suite,
//! RFC 4231 and the S3 documentation; the store is exercised against a loopback
//! `wiremock` server, so the suite needs no S3 service, no network, and no
//! credentials. Every store test additionally re-derives the signature of the
//! request the mock received, so a header that changes without the signature
//! changing fails the suite.

use std::collections::BTreeMap;
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use url::Url;

use wiremock::matchers::{method, path, query_param, query_param_is_missing};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;

use super::sigv4::{self, Credentials, SigningInput, EMPTY_PAYLOAD_SHA256};
use super::{S3Config, S3ObjectStore};
use crate::store::object::ObjectStore;

// --- fixtures ---------------------------------------------------------------

/// Access key id used by the AWS signature test suite and by the store tests.
const ACCESS_KEY: &str = "AKIDEXAMPLE";

/// Secret access key used by the AWS signature test suite and by the store tests.
const SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

/// The region the store tests sign with.
const REGION: &str = "us-east-1";

/// The credentials shared by the vector and store tests.
fn credentials() -> Credentials {
    Credentials::new(ACCESS_KEY, SECRET_KEY)
}

/// Parses a URL for the signer tests.
fn url(raw: &str) -> Url {
    Url::parse(raw).expect("valid URL")
}

/// Builds an instant from a `yyyy-mm-ddThh:mm:ssZ` coordinate.
fn instant(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
        .single()
        .expect("valid timestamp")
}

/// Turns the emitted header list into a lookup table.
fn header_map(headers: &[(String, String)]) -> BTreeMap<String, String> {
    headers.iter().cloned().collect()
}

/// Reads one header of a received request as a string.
fn header(request: &wiremock::Request, name: &str) -> String {
    request
        .headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

/// Reads one query parameter of a received request.
fn query(request: &wiremock::Request, name: &str) -> Option<String> {
    request
        .url
        .query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// Removes and returns the single request the mock server received.
async fn only_request(server: &MockServer) -> wiremock::Request {
    let mut requests = server
        .received_requests()
        .await
        .expect("mock server is running");
    assert_eq!(requests.len(), 1, "expected exactly one request");
    requests.remove(0)
}

/// Builds a store pointing at `server` with path-style addressing.
fn store(server: &MockServer) -> S3ObjectStore {
    S3ObjectStore::new(
        S3Config::new(server.uri())
            .with_credentials(credentials())
            .with_timeout(Duration::from_secs(5)),
    )
    .expect("build store")
}

/// Builds the signable view of one request the mock server received.
///
/// The signed header set is taken from the `Authorization` header itself, so the
/// recomputed signature covers exactly the headers the client claims to sign.
fn restored_input<'a>(
    request: &'a wiremock::Request,
    signed: &'a [(String, String)],
    payload_hash: &'a str,
    now: DateTime<Utc>,
    query: &'a [(String, String)],
) -> SigningInput<'a> {
    SigningInput {
        method: request.method.as_str(),
        url: &request.url,
        query,
        headers: signed,
        payload_sha256_hex: payload_hash,
        region: REGION,
        service: "s3",
        now,
    }
}

/// Recomputes the signature of a request the mock server received.
fn recompute_signature(request: &wiremock::Request) -> String {
    let authorization = header(request, "authorization");
    let names = signed_headers(&authorization);
    let signed: Vec<(String, String)> = names
        .iter()
        .map(|name| (name.clone(), header(request, name)))
        .collect();
    let query: Vec<(String, String)> = request
        .url
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    let payload_hash = header(request, "x-amz-content-sha256");
    let amz_date = header(request, "x-amz-date");
    let now = chrono::NaiveDateTime::parse_from_str(
        amz_date.strip_suffix('Z').unwrap_or(&amz_date),
        "%Y%m%dT%H%M%S",
    )
    .expect("x-amz-date is a signature timestamp")
    .and_utc();
    let input = restored_input(request, &signed, &payload_hash, now, &query);
    sigv4::sign_request(&credentials(), &input).signature
}

/// The `SignedHeaders=` list of an `Authorization` header.
fn signed_headers(authorization: &str) -> Vec<String> {
    let field = authorization
        .split("SignedHeaders=")
        .nth(1)
        .unwrap_or_default();
    field
        .split(',')
        .next()
        .unwrap_or_default()
        .split(';')
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

/// Asserts the headers every signed request must carry, and that the signature
/// covers the request as it was sent.
fn assert_signed(request: &wiremock::Request, expected_signed: &str) -> String {
    let authorization = header(request, "authorization");
    assert!(
        authorization.starts_with(&format!("AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/")),
        "unexpected authorization header: {authorization}"
    );
    assert!(
        authorization.contains(&format!("/{REGION}/s3/aws4_request")),
        "unexpected scope: {authorization}"
    );
    assert_eq!(
        signed_headers(&authorization).join(";"),
        expected_signed,
        "unexpected signed headers in {authorization}"
    );
    assert!(
        header(request, "x-amz-date").len() == 16,
        "x-amz-date must be a signature timestamp"
    );
    assert!(
        !header(request, "x-amz-content-sha256").is_empty(),
        "x-amz-content-sha256 must be sent"
    );
    let signature = recompute_signature(request);
    assert!(
        authorization.ends_with(&format!("Signature={signature}")),
        "the signature does not cover the request as sent: {authorization}"
    );
    signature
}

// The pure signer tests live in their own file: the known-answer vectors are
// independent of the store and keep this file readable.
mod sigv4_vectors;

// --- configuration ----------------------------------------------------------

#[test]
fn config_defaults_and_round_trip() {
    let config = S3Config::default();
    assert_eq!(config.region, "us-east-1");
    assert!(config.path_style);
    assert_eq!(config.timeout, Duration::from_secs(30));
    assert!(config.bucket.is_none());

    let config: S3Config = serde_json::from_str(
        r#"{"endpoint":"http://localhost:9000","bucket":"uploads","path_style":false,"extra":1}"#,
    )
    .expect("config deserializes");
    assert_eq!(config.endpoint, "http://localhost:9000");
    assert_eq!(config.bucket.as_deref(), Some("uploads"));
    assert!(!config.path_style);
    // Omitted fields keep their documented defaults.
    assert_eq!(config.region, "us-east-1");
    assert_eq!(config.timeout, Duration::from_secs(30));

    let json = serde_json::to_string(&config).expect("config serializes");
    let restored: S3Config = serde_json::from_str(&json).expect("config round-trips");
    assert_eq!(restored.endpoint, config.endpoint);
}

#[test]
fn invalid_endpoints_are_rejected() {
    assert_eq!(
        S3ObjectStore::new(S3Config::default()).unwrap_err().kind(),
        ErrorKind::Config
    );
    assert_eq!(
        S3ObjectStore::new(S3Config::new("ftp://example.com"))
            .unwrap_err()
            .kind(),
        ErrorKind::Config
    );
    assert_eq!(
        S3ObjectStore::new(S3Config::new("not a url"))
            .unwrap_err()
            .kind(),
        ErrorKind::Config
    );
}

#[tokio::test]
async fn anonymous_requests_are_rejected_before_any_transfer() {
    let store = S3ObjectStore::new(S3Config::new("http://127.0.0.1:1")).expect("build store");
    let error = store
        .put("bucket", "key", bytes::Bytes::from_static(b"x"))
        .await
        .expect_err("anonymous requests are refused");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("must be signed"));
}

// --- addressing -------------------------------------------------------------

#[test]
fn path_style_urls_nest_the_bucket_and_encode_the_key() {
    let store = S3ObjectStore::new(
        S3Config::new("http://localhost:9000/prefix/").with_credentials(credentials()),
    )
    .expect("build store");

    assert_eq!(
        store
            .url_for("bucket", Some("dir/a b.txt"))
            .expect("url")
            .as_str(),
        "http://localhost:9000/prefix/bucket/dir/a%20b.txt"
    );
    assert_eq!(
        store.url_for("bucket", None).expect("url").as_str(),
        "http://localhost:9000/prefix/bucket"
    );
    assert_eq!(
        store
            .url_for("bucket", Some("nested/深い.txt"))
            .expect("url")
            .as_str(),
        "http://localhost:9000/prefix/bucket/nested/%E6%B7%B1%E3%81%84.txt"
    );

    assert_eq!(
        store.url_for("", Some("k")).unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        store.url_for("bucket", Some("")).unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        store.url_for("bad/bucket", None).unwrap_err().kind(),
        ErrorKind::Validation
    );
}

#[test]
fn virtual_host_addressing_moves_the_bucket_into_the_host() {
    let store = S3ObjectStore::new(
        S3Config::new("https://s3.example.com:8443")
            .with_credentials(credentials())
            .with_path_style(false),
    )
    .expect("build store");

    assert_eq!(
        store
            .url_for("uploads", Some("dir/a.txt"))
            .expect("url")
            .as_str(),
        "https://uploads.s3.example.com:8443/dir/a.txt"
    );
    assert_eq!(
        store.url_for("uploads", None).expect("url").as_str(),
        "https://uploads.s3.example.com:8443/"
    );
    // A dotted bucket name is used as-is; TLS against it needs a wildcard cert.
    assert_eq!(
        store
            .url_for("my.bucket", Some("k"))
            .expect("url")
            .host_str()
            .unwrap_or_default(),
        "my.bucket.s3.example.com"
    );
}

// --- object operations ------------------------------------------------------

#[tokio::test]
async fn put_sends_a_signed_path_style_request() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/bucket/dir/a.txt"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-length", "4")
                .insert_header("last-modified", "Fri, 24 May 2013 00:00:00 GMT"),
        )
        .mount(&server)
        .await;

    let store = store(&server);
    let meta = store
        .put("bucket", "dir/a.txt", bytes::Bytes::from_static(b"data"))
        .await
        .expect("put");

    assert_eq!(meta.key, "dir/a.txt");
    assert_eq!(meta.size, 4);
    assert_eq!(meta.updated_at.timestamp(), 1_369_353_600);

    let request = only_request(&server).await;
    assert_eq!(request.method.as_str(), "PUT");
    assert_eq!(request.url.path(), "/bucket/dir/a.txt");
    assert_eq!(request.body, b"data");
    assert_signed(&request, "host;x-amz-content-sha256;x-amz-date");
    assert_eq!(
        header(&request, "x-amz-content-sha256"),
        sigv4::sha256_hex(b"data")
    );
    assert_eq!(header(&request, "x-amz-security-token"), "");
}

#[tokio::test]
async fn put_without_metadata_falls_back_to_the_body_length() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/bucket/k.txt"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let before = Utc::now();
    let store = store(&server);
    let meta = store
        .put("bucket", "k.txt", bytes::Bytes::from_static(b"hello"))
        .await
        .expect("put");

    assert_eq!(meta.size, 5);
    assert!(meta.updated_at >= before);
}

#[tokio::test]
async fn put_with_a_session_token_sends_and_signs_it() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/bucket/k.txt"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let store = S3ObjectStore::new(
        S3Config::new(server.uri()).with_credentials(credentials().with_session_token("token-123")),
    )
    .expect("build store");
    store
        .put("bucket", "k.txt", bytes::Bytes::from_static(b"x"))
        .await
        .expect("put");

    let request = only_request(&server).await;
    assert_eq!(header(&request, "x-amz-security-token"), "token-123");
    assert_signed(
        &request,
        "host;x-amz-content-sha256;x-amz-date;x-amz-security-token",
    );
}

#[tokio::test]
async fn get_returns_the_payload_and_maps_404_to_none() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket/present.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"payload".to_vec()))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/bucket/missing.txt"))
        .respond_with(
            ResponseTemplate::new(404).set_body_string(
                "<Error><Code>NoSuchKey</Code><Message>not found</Message></Error>",
            ),
        )
        .mount(&server)
        .await;

    let store = store(&server);
    assert_eq!(
        store.get("bucket", "present.txt").await.expect("get"),
        Some(bytes::Bytes::from_static(b"payload"))
    );
    assert_eq!(store.get("bucket", "missing.txt").await.expect("get"), None);

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_signed(request, "host;x-amz-content-sha256;x-amz-date");
        assert_eq!(
            header(request, "x-amz-content-sha256"),
            EMPTY_PAYLOAD_SHA256
        );
    }
}

#[tokio::test]
async fn delete_tolerates_a_missing_object() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/bucket/present.txt"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/bucket/missing.txt"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let store = store(&server);
    store.delete("bucket", "present.txt").await.expect("delete");
    store.delete("bucket", "missing.txt").await.expect("delete");

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].url.path(), "/bucket/present.txt");
    assert_eq!(requests[1].url.path(), "/bucket/missing.txt");
    for request in &requests {
        assert_signed(request, "host;x-amz-content-sha256;x-amz-date");
    }
}

#[tokio::test]
async fn exists_maps_head_statuses() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/bucket/present.txt"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("HEAD"))
        .and(path("/bucket/missing.txt"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let store = store(&server);
    assert!(store.exists("bucket", "present.txt").await.expect("exists"));
    assert!(!store.exists("bucket", "missing.txt").await.expect("exists"));

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 2);
    assert_signed(&requests[0], "host;x-amz-content-sha256;x-amz-date");
}

#[tokio::test]
async fn exists_reports_a_forbidden_response_as_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/bucket/k.txt"))
        .respond_with(ResponseTemplate::new(403))
        .mount(&server)
        .await;

    let store = store(&server);
    let error = store
        .exists("bucket", "k.txt")
        .await
        .expect_err("403 fails");
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(error.to_string().contains("403"));
}

// --- bucket operations ------------------------------------------------------

#[tokio::test]
async fn create_bucket_is_idempotent_on_409() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/bucket"))
        .respond_with(ResponseTemplate::new(409).set_body_string(
            "<Error><Code>BucketAlreadyOwnedByYou</Code><Message>exists</Message></Error>",
        ))
        .mount(&server)
        .await;

    let store = store(&server);
    store.create_bucket("bucket").await.expect("create bucket");

    let request = only_request(&server).await;
    assert_eq!(request.url.path(), "/bucket");
    assert_eq!(request.url.query(), None);
    assert_signed(&request, "host;x-amz-content-sha256;x-amz-date");
}

#[tokio::test]
async fn delete_bucket_tolerates_404() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/bucket"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let store = store(&server);
    store.delete_bucket("bucket").await.expect("delete bucket");
    store.delete_bucket("bucket").await.expect("delete again");
}

#[tokio::test]
async fn list_buckets_parses_and_sorts_names() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
            r#"<?xml version="1.0" encoding="UTF-8"?>"#,
            r#"<ListAllMyBucketsResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">"#,
            r#"<Owner><ID>owner-id</ID><DisplayName>me</DisplayName></Owner>"#,
            r#"<Buckets>"#,
            r#"<Bucket><Name>zeta</Name><CreationDate>2019-12-11T23:32:47.000Z</CreationDate></Bucket>"#,
            r#"<Bucket><Name>alpha&amp;beta</Name><CreationDate>2019-12-11T23:32:47.000Z</CreationDate></Bucket>"#,
            r#"</Buckets>"#,
            r#"</ListAllMyBucketsResult>"#,
        )))
        .mount(&server)
        .await;

    let store = store(&server);
    assert_eq!(
        store.list_buckets().await.expect("list buckets"),
        vec!["alpha&beta".to_string(), "zeta".to_string()]
    );

    let request = only_request(&server).await;
    assert_eq!(request.url.path(), "/");
    assert_signed(&request, "host;x-amz-content-sha256;x-amz-date");
}

// --- listings ---------------------------------------------------------------

#[tokio::test]
async fn list_parses_entries_sizes_and_dates() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket"))
        .and(query_param("list-type", "2"))
        .and(query_param("prefix", "docs/"))
        .and(query_param_is_missing("continuation-token"))
        .and(query_param("max-keys", "10"))
        .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
            r#"<?xml version="1.0" encoding="UTF-8"?>"#,
            r#"<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">"#,
            r#"<Name>bucket</Name><Prefix>docs/</Prefix><KeyCount>2</KeyCount>"#,
            r#"<IsTruncated>false</IsTruncated>"#,
            r#"<Contents>"#,
            r#"<Key>docs/&lt;x&gt; &amp; y.txt</Key>"#,
            r#"<LastModified>2013-05-24T00:00:00.000Z</LastModified>"#,
            r#"<ETag>&quot;abc&quot;</ETag><Size>3</Size>"#,
            r#"</Contents>"#,
            r#"<Contents>"#,
            r#"<Key>docs/a.txt</Key>"#,
            r#"<LastModified>2013-05-24T01:00:00.000Z</LastModified>"#,
            r#"<Size>5</Size>"#,
            r#"</Contents>"#,
            r#"</ListBucketResult>"#,
        )))
        .mount(&server)
        .await;

    let store = store(&server);
    let entries = store.list("bucket", "docs/", Some(10)).await.expect("list");

    let names: Vec<&str> = entries.iter().map(|entry| entry.key.as_str()).collect();
    assert_eq!(names, vec!["docs/<x> & y.txt", "docs/a.txt"]);
    assert_eq!(entries[0].size, 3);
    assert_eq!(entries[1].size, 5);
    assert_eq!(entries[1].updated_at.timestamp(), 1_369_357_200);

    let request = only_request(&server).await;
    assert_eq!(request.url.path(), "/bucket");
    assert_eq!(query(&request, "list-type").as_deref(), Some("2"));
    assert_eq!(query(&request, "prefix").as_deref(), Some("docs/"));
    assert_eq!(query(&request, "max-keys").as_deref(), Some("10"));
    assert_signed(&request, "host;x-amz-content-sha256;x-amz-date");
}

#[tokio::test]
async fn list_follows_continuation_tokens_and_truncates() {
    let server = MockServer::start().await;
    // First page: two entries, truncated, with a token.
    Mock::given(method("GET"))
        .and(path("/bucket"))
        .and(query_param_is_missing("continuation-token"))
        .and(query_param("max-keys", "3"))
        .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
            r#"<ListBucketResult>"#,
            r#"<IsTruncated>true</IsTruncated>"#,
            r#"<Contents><Key>b.txt</Key><LastModified>2013-05-24T00:00:00.000Z</LastModified><Size>1</Size></Contents>"#,
            r#"<Contents><Key>c.txt</Key><LastModified>2013-05-24T00:00:00.000Z</LastModified><Size>2</Size></Contents>"#,
            r#"<NextContinuationToken>page-2</NextContinuationToken>"#,
            r#"</ListBucketResult>"#,
        )))
        .mount(&server)
        .await;
    // Second page: the remaining budget is one key.
    Mock::given(method("GET"))
        .and(path("/bucket"))
        .and(query_param("continuation-token", "page-2"))
        .and(query_param("max-keys", "1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
            r#"<ListBucketResult>"#,
            r#"<IsTruncated>false</IsTruncated>"#,
            r#"<Contents><Key>a.txt</Key><LastModified>2013-05-24T02:00:00.000Z</LastModified><Size>3</Size></Contents>"#,
            r#"<Contents><Key>d.txt</Key><LastModified>2013-05-24T03:00:00.000Z</LastModified><Size>4</Size></Contents>"#,
            r#"</ListBucketResult>"#,
        )))
        .mount(&server)
        .await;

    let store = store(&server);
    let entries = store.list("bucket", "", Some(3)).await.expect("list");

    let names: Vec<&str> = entries.iter().map(|entry| entry.key.as_str()).collect();
    assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(query(&requests[0], "max-keys").as_deref(), Some("3"));
    assert_eq!(query(&requests[0], "prefix").as_deref(), Some(""));
    assert_eq!(
        query(&requests[1], "continuation-token").as_deref(),
        Some("page-2")
    );
    assert_eq!(query(&requests[1], "max-keys").as_deref(), Some("1"));
    for request in &requests {
        assert_signed(request, "host;x-amz-content-sha256;x-amz-date");
    }
}

#[tokio::test]
async fn list_without_a_limit_stops_at_the_last_page() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket"))
        .and(query_param_is_missing("max-keys"))
        .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
            r#"<ListBucketResult>"#,
            r#"<IsTruncated>false</IsTruncated>"#,
            r#"<Contents><Key>only.txt</Key><LastModified>2013-05-24T00:00:00.000Z</LastModified><Size>0</Size></Contents>"#,
            r#"</ListBucketResult>"#,
        )))
        .mount(&server)
        .await;

    let store = store(&server);
    let entries = store.list("bucket", "", None).await.expect("list");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].key, "only.txt");
    assert_eq!(entries[0].size, 0);
}

#[tokio::test]
async fn list_maps_a_missing_bucket_to_not_found() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket"))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_string("<Error><Code>NoSuchBucket</Code><Message>gone</Message></Error>"),
        )
        .mount(&server)
        .await;

    let store = store(&server);
    let error = store
        .list("bucket", "", None)
        .await
        .expect_err("missing bucket");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("bucket"));
}

#[tokio::test]
async fn malformed_listing_bodies_do_not_panic() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string("<ListBucketResult><Contents><Key>a"),
        )
        .mount(&server)
        .await;

    let store = store(&server);
    assert_eq!(
        store.list("bucket", "", None).await.expect("list"),
        Vec::new()
    );
}

// --- errors -----------------------------------------------------------------

#[tokio::test]
async fn error_body_surfaces_the_s3_code_and_status() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket/secret.txt"))
        .respond_with(ResponseTemplate::new(403).set_body_string(concat!(
            r#"<?xml version="1.0" encoding="UTF-8"?>"#,
            r#"<Error><Code>AccessDenied</Code><Message>Access Denied</Message>"#,
            r#"<RequestId>656c76696e6727732072657175657374</RequestId></Error>"#,
        )))
        .mount(&server)
        .await;

    let store = store(&server);
    let error = store.get("bucket", "secret.txt").await.expect_err("403");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(message.contains("403"), "unexpected message: {message}");
    assert!(
        message.contains("AccessDenied"),
        "unexpected message: {message}"
    );
    assert!(message.contains("<Code>"), "unexpected message: {message}");
    assert!(
        message.contains(&format!("{}/bucket/secret.txt", server.uri())),
        "unexpected message: {message}"
    );
    // The external error keeps its cause reachable.
    assert!(std::error::Error::source(&error).is_some());
}

#[tokio::test]
async fn put_failure_reports_the_operation_and_status() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/bucket/k.txt"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    let store = store(&server);
    let error = store
        .put("bucket", "k.txt", bytes::Bytes::from_static(b"x"))
        .await
        .expect_err("500 fails");
    let message = error.to_string();
    assert!(message.contains("put"), "unexpected message: {message}");
    assert!(message.contains("500"), "unexpected message: {message}");
    assert!(message.contains("boom"), "unexpected message: {message}");
}
