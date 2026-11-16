use super::*;

#[test]
fn object_names_are_percent_encoded_as_one_segment() {
    assert_eq!(
        encode_object_name("graph/versions/v1/graph.json"),
        "graph%2Fversions%2Fv1%2Fgraph.json"
    );
    assert_eq!(encode_object_name("a b/c d.json"), "a%20b%2Fc%20d.json");
    assert_eq!(
        encode_object_name("gráf/图.json"),
        "gr%C3%A1f%2F%E5%9B%BE.json"
    );
    assert_eq!(encode_object_name("keep-me_1.2~3"), "keep-me_1.2~3");
    assert_eq!(encode_object_name("a?b#c%d"), "a%3Fb%23c%25d");
}

#[test]
fn the_default_configuration_targets_the_public_endpoint() {
    let config = GcsConfig::default();
    assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
    assert!(config.project.is_none());
    assert!(config.access_token.is_none());
    assert_eq!(config.timeout, Duration::from_secs(60));
}

#[test]
fn the_access_token_is_serialized_but_redacted_in_debug() {
    let config = GcsConfig::new("http://localhost:4443")
        .with_project("demo")
        .with_access_token("ya29.secret");
    let json = serde_json::to_value(&config).expect("serialize");
    assert_eq!(json["access_token"], "ya29.secret");
    assert!(!format!("{config:?}").contains("ya29.secret"));

    let decoded: GcsConfig = serde_json::from_value(json).expect("deserialize");
    assert_eq!(decoded.project.as_deref(), Some("demo"));
    assert_eq!(
        decoded
            .access_token
            .as_ref()
            .map(SecretString::expose_secret),
        Some("ya29.secret")
    );
}

#[test]
fn a_partial_document_keeps_every_default() {
    let config: GcsConfig =
        serde_json::from_str(r#"{"project": "demo"}"#).expect("decode a partial document");
    assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.project.as_deref(), Some("demo"));
}

#[test]
fn endpoints_are_validated() {
    assert_eq!(
        validate_endpoint(" http://localhost:4443/ ").expect("valid endpoint"),
        "http://localhost:4443"
    );
    assert_eq!(
        validate_endpoint("https://example.com/api/").expect("a path is kept"),
        "https://example.com/api"
    );
    for endpoint in ["", "  ", "localhost:4443", "ftp://example.com", "http://"] {
        let error = validate_endpoint(endpoint).expect_err("must reject");
        assert_eq!(
            error.kind(),
            crate::error::ErrorKind::Config,
            "{endpoint:?}"
        );
        assert!(error.message().contains("endpoint"), "{endpoint:?}");
    }
}

#[test]
fn bucket_names_and_keys_are_validated() {
    for bucket in ["", "  ", "a/b", "a?b", "a#b", "a%b"] {
        let error = validate_bucket(bucket).expect_err("must reject");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
    assert!(validate_bucket("my-bucket").is_ok());
    assert!(validate_key("").is_err());
    assert!(validate_key("k").is_ok());
}

#[test]
fn sizes_are_read_from_numbers_and_decimal_strings() {
    assert_eq!(json_u64(&serde_json::json!(12)), Some(12));
    assert_eq!(json_u64(&serde_json::json!("12")), Some(12));
    assert_eq!(json_u64(&serde_json::json!(" 12 ")), Some(12));
    assert_eq!(json_u64(&serde_json::json!(-3)), None);
    assert_eq!(json_u64(&serde_json::json!(1.5)), None);
    assert_eq!(json_u64(&serde_json::json!(null)), None);
}

#[test]
fn object_metadata_parses_the_rfc3339_timestamp_and_falls_back() {
    let resource = ObjectResource {
        name: "graph/current.json".to_string(),
        size: Some(serde_json::json!("7")),
        updated: Some("2024-01-02T03:04:05.678Z".to_string()),
    };
    let meta = meta_from_resource("graph/current.json", &resource, 99);
    assert_eq!(meta.key, "graph/current.json");
    assert_eq!(meta.size, 7);
    assert_eq!(meta.updated_at.timestamp(), 1_704_164_645);

    let bare = ObjectResource::default();
    let before = now_utc();
    let meta = meta_from_resource("k", &bare, 5);
    assert_eq!(meta.size, 5);
    assert!(meta.updated_at >= before);

    let malformed = ObjectResource {
        updated: Some("not a timestamp".to_string()),
        ..ObjectResource::default()
    };
    let meta = meta_from_resource("k", &malformed, 3);
    assert_eq!(meta.size, 3);
    assert!(meta.updated_at >= before);
}

/// The same 2048-bit PKCS#8 RSA key the token, key and signing modules' tests use.
const SERVICE_ACCOUNT_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDWEwu1E3LIrY/1
ukZXqTj+8wVh/tbtgPgivj104Ox3FLpvj3jhRu3QRibNMN6DM4Fi13J6IE2WRdxF
qiwyVX9lg98zpl6msySu3EYewibFHWPyW0OaKzm1GqNjzvFev/O7CqvA6KJUtgfj
rneYOGBqpl//ojbdO7d9/kq1ivXmaM6lvx+JSssHnYbvdfmH3PtSGfdtODuNSrrI
yDyAycSKmi6DIxojI3hJGXZm9oMX0Dh39sXOvTk+s66b59cDqq2TZfFoUD4Z9wtj
7b34kAZj4pGKk36OT+TDXzmypKvFbf7lHYSf+P4RN9EasunktpoRSQRlpXElNkKd
qgkZft/hAgMBAAECggEAds7TLIfCqIVK5EGhUH1mWHhfgGeCuQVesUmLqhY6pkwm
vEuP9tb6nSCR4EdC9uXa5RRVQFo1NTVIbpFnPfgdYCWz9mJbHYFG1pUBhYiTI8qP
8a2c8y8SM+S40WfEpaU6i4/8im+RyBnOfxoQ3bZMLGRx2toYQOgnihFbPS4GEv0L
8NTdj0XGNw09cy9ckIosHaZjF7MvNleeJjSR0l4Bl6HbCAS8DAxmrwXzEX6tRXdp
rYrjkqSP6qvC4+S7L4q6OpPNqRSl4urTC9ubFhtyfmo3FlMBC9w7uHLEghyfbSAF
WrHYjTxgRZnifeb9RfPfxButnQ4Y2lExmBysuaQzUQKBgQDmQ4tvP6wKHFwxljzG
ioyhJlWNsekJ306IVvLPd7fGfosJOIyVe/s1CJJa8hbNETZuW4TqyNVwBBKJYFHd
qBvJbOO5uxCb3kd9SoVR5GKM/XJm1344G1bBNil2fize+meza3YIs9bFP1DdJFeL
xxtIhfAAsehpsiaXLdO49/Cs/wKBgQDuAEc9DeIO5f6MuTijS2uvR8MrRcCap0PC
hYWjfeibYrg3jMAZuGmIEDY8UMwDIyTC5adsgnvbBwa9kuu1QJoOQkgpynSncdHq
rWRui1Vreqzwjwgo+vlPjP7UCNmmmsX5h4Lq/piKZRoNDHJuYbrsP4GLHYqXu708
nYriQn0THwKBgD+/s3q5RggGkf+oeGr/MK/szQLNnpb9G+DjMZX+rwJlRljnLwSz
rnigRuSm4Wkkus0SyVm3AGBiwD8IS6HySL8bRnDFOXPtth0Z5I8ni6E9Cc8vGeKu
rvYO6KvFgua4rL0foy4FnBA/10NiLFqmZQdPDYMjvhSFHen85gDvo9JBAoGAErlq
4V7KPinSEPgxgIdafXvD2w+ty30FU7R1gxtQxYjKvXQRWPStOMstuSkSI90L81lm
Bp9tZxnyac+5Q5VFiYGlc8Q90T6fKTti/19u5PnNyU83cSHto1iQY9YeaYEbujnz
DtHg5mJ13jjwd8awkg2c/3jfgnh+ttoeJxbcCDkCgYEA1wWLpUIeRchZHOqbYFoM
QEDRqvnHYqrd0cp9hTp+uyoCz9e8C8n54wKzgWEtkxsAfO3Pami4T+z+NiYeFpWn
+1SJIbJWbPf2elxD60UjEa8OeelSGiu5X5KlfPvyuVgX4qs9vUYQQiTHDZbzmDlV
I1Ru6xykeZLm6JI5YG+U0zg=
-----END PRIVATE KEY-----
";

/// The bearer token the mocked token endpoint mints.
const MINTED_TOKEN: &str = "ya29.minted-token";

/// The bucket the wiremock cases read from.
const BUCKET: &str = "demo-bucket";

/// The object key the wiremock cases read.
const KEY: &str = "graph.json";

/// Writes a complete service-account key file whose `token_uri` is `token_uri`.
fn write_key_file(token_uri: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("service-account.json");
    let document = serde_json::json!({
        "type": "service_account",
        "project_id": "demo-project",
        "private_key_id": "0123456789abcdef",
        "private_key": SERVICE_ACCOUNT_PEM,
        "client_email": "symphony@demo-project.iam.gserviceaccount.com",
        "token_uri": token_uri,
        "universe_domain": "googleapis.com",
    });
    std::fs::write(&path, serde_json::to_vec(&document).expect("serialize")).expect("write");
    (directory, path)
}

/// Mounts the token endpoint, minting [`MINTED_TOKEN`] with an hour of lifetime.
async fn mount_token_endpoint(server: &wiremock::MockServer) {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": MINTED_TOKEN,
            "expires_in": 3600,
            "token_type": "Bearer",
        })))
        .mount(server)
        .await;
}

/// Mounts a permissive object endpoint answering every request with a 200 `{}` reply.
async fn mount_object_endpoint(server: &wiremock::MockServer) {
    use wiremock::matchers::any;
    use wiremock::{Mock, ResponseTemplate};
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .mount(server)
        .await;
}

/// The `Authorization` header of a recorded request, when it carries one.
fn authorization(request: &wiremock::Request) -> Option<String> {
    request
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

/// The recorded requests that reached the object API (not the token endpoint).
fn object_requests(requests: &[wiremock::Request]) -> Vec<&wiremock::Request> {
    requests
        .iter()
        .filter(|request| request.url.path().starts_with("/storage/"))
        .collect()
}

#[tokio::test]
async fn a_service_account_store_sends_the_minted_bearer_token() {
    use wiremock::MockServer;
    let server = MockServer::start().await;
    mount_token_endpoint(&server).await;
    mount_object_endpoint(&server).await;

    let (_directory, path) = write_key_file(&format!("{}/token", server.uri()));
    let store = GcsObjectStore::from_service_account_file_with(GcsConfig::new(server.uri()), &path)
        .expect("build a store");
    // The project falls back to the key file's `project_id`.
    assert_eq!(store.config().project.as_deref(), Some("demo-project"));

    store.exists(BUCKET, KEY).await.expect("read an object");

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.url.path() == "/token")
            .count(),
        1,
        "the token is minted once"
    );
    let objects = object_requests(&requests);
    assert_eq!(objects.len(), 1);
    assert_eq!(
        authorization(objects[0]).as_deref(),
        Some(format!("Bearer {MINTED_TOKEN}").as_str())
    );
}

#[tokio::test]
async fn an_explicit_access_token_wins_over_the_key_file() {
    use wiremock::MockServer;
    let server = MockServer::start().await;
    mount_token_endpoint(&server).await;
    mount_object_endpoint(&server).await;

    let (_directory, path) = write_key_file(&format!("{}/token", server.uri()));
    let store = GcsObjectStore::from_service_account_file_with(
        GcsConfig::new(server.uri()).with_access_token("ya29.explicit"),
        &path,
    )
    .expect("build a store");

    store.exists(BUCKET, KEY).await.expect("read an object");

    let requests = server.received_requests().await.expect("recorded requests");
    assert!(
        requests
            .iter()
            .all(|request| request.url.path() != "/token"),
        "the token endpoint must not be called"
    );
    let objects = object_requests(&requests);
    assert_eq!(objects.len(), 1);
    assert_eq!(
        authorization(objects[0]).as_deref(),
        Some("Bearer ya29.explicit")
    );
}

#[test]
fn an_unreadable_key_file_is_reported_with_its_path() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("absent.json");
    let error = GcsObjectStore::from_service_account_file(&path)
        .expect_err("a missing key file must be rejected");
    assert_eq!(error.kind(), crate::error::ErrorKind::External);
    assert!(error.message().contains("absent.json"), "{error}");
}

#[tokio::test]
async fn a_failed_token_exchange_never_falls_back_to_anonymous_requests() {
    use wiremock::matchers::path;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "error": "invalid_grant",
            "error_description": "Invalid JWT",
        })))
        .mount(&server)
        .await;
    mount_object_endpoint(&server).await;

    let (_directory, path) = write_key_file(&format!("{}/token", server.uri()));
    let store = GcsObjectStore::from_service_account_file_with(GcsConfig::new(server.uri()), &path)
        .expect("build a store");

    let error = store
        .exists(BUCKET, KEY)
        .await
        .expect_err("a failed mint must fail the request");
    assert_eq!(error.kind(), crate::error::ErrorKind::External);
    assert!(error.message().contains("invalid_grant"), "{error}");
    let requests = server.received_requests().await.expect("recorded requests");
    assert!(
        object_requests(&requests).is_empty(),
        "no anonymous object request may be sent"
    );
}

#[test]
fn a_service_account_store_never_renders_its_key_or_a_token() {
    let (_directory, path) = write_key_file("https://oauth2.googleapis.com/token");
    let store = GcsObjectStore::from_service_account_file(&path).expect("build a store");
    let rendered = format!("{store:?}");
    assert!(rendered.contains("TokenSource"), "{rendered}");
    assert!(rendered.contains("demo-project"), "{rendered}");
    assert!(rendered.contains("cached"), "{rendered}");
    assert!(!rendered.contains(MINTED_TOKEN), "{rendered}");
    assert!(!rendered.contains("MII"), "{rendered}");
    assert!(!rendered.contains("BEGIN PRIVATE KEY"), "{rendered}");
}

#[test]
fn a_project_less_bucket_operation_names_the_project() {
    let store =
        GcsObjectStore::new(GcsConfig::new("http://localhost:4443")).expect("build a store");
    for error in [
        store.project("create").expect_err("create needs a project"),
        store.project("list").expect_err("list needs a project"),
    ] {
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("project"), "{}", error.message());
    }
}
