//! Behaviour tests for the Google Cloud Storage artifact transport.
//!
//! The transport speaks the GCS JSON API directly, so the suite drives it against a stateful
//! loopback emulator built on `wiremock`: one responder dispatches by method and path, keeps
//! buckets and objects in memory, and reports the metadata, `nextPageToken` pagination and
//! `notFound` error documents a real bucket would. Everything is offline and deterministic —
//! the endpoint is the mock server's URI, no credentials are sent to any real service, and no
//! assertion depends on the wall clock (the emulator reports a fixed `updated` timestamp).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use bytes::Bytes;
use serde_json::{json, Value};
use wiremock::matchers::{any, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use openjiuwen_foundation::store::ObjectStore;
use openjiuwen_symphony::artifact::{GraphArtifactStore, STATUS_READY};
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::graph::{CapabilityGraph, CapabilityNode};
use openjiuwen_symphony::object_artifact::{
    ArtifactTransport, GcsConfig, GcsObjectStore, ObjectGraphArtifactStore,
};

/// Bearer token every authenticated case configures.
const TOKEN: &str = "ya29.test-token";

/// The `updated` timestamp the emulator reports (2024-01-02T03:04:05.678Z).
const UPDATED: &str = "2024-01-02T03:04:05.678Z";

/// The Unix timestamp of [`UPDATED`].
const UPDATED_EPOCH: i64 = 1_704_164_645;

// --- the emulator -----------------------------------------------------------

/// The buckets and objects the emulator holds.
#[derive(Debug, Default)]
struct FakeState {
    /// Objects per bucket; the outer key is the bucket name.
    buckets: BTreeMap<String, BTreeMap<String, Bytes>>,
}

/// A minimal stateful Google Cloud Storage emulator.
///
/// It answers the subset of the JSON API the transport uses and honours a fixed page size, so
/// pagination is exercised even when the client sends no `maxResults`.
struct FakeGcs {
    /// Objects the emulator holds.
    state: Arc<Mutex<FakeState>>,
    /// Largest page the listing endpoint returns.
    page_size: usize,
}

impl FakeGcs {
    /// Locks the emulator state, ignoring a panic of another test thread.
    fn lock(&self) -> MutexGuard<'_, FakeState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Answers `POST /storage/v1/b?project=…`.
    fn create_bucket(&self, query: &HashMap<String, String>, body: &[u8]) -> ResponseTemplate {
        if query.get("project").is_none_or(String::is_empty) {
            return bad("creating a bucket needs a project");
        }
        let Ok(document) = serde_json::from_slice::<Value>(body) else {
            return bad("creating a bucket needs a JSON body");
        };
        let Some(name) = document
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            return bad("creating a bucket needs a name");
        };
        let mut state = self.lock();
        if state.buckets.contains_key(name) {
            return json_response(
                409,
                json!({
                    "error": {
                        "code": 409,
                        "message": "You already own this bucket.",
                        "errors": [{ "reason": "conflict" }],
                    }
                }),
            );
        }
        state.buckets.insert(name.to_string(), BTreeMap::new());
        json_response(200, json!({ "kind": "storage#bucket", "name": name }))
    }

    /// Answers `GET /storage/v1/b?project=…`.
    fn list_buckets(&self, query: &HashMap<String, String>) -> ResponseTemplate {
        if query.get("project").is_none_or(String::is_empty) {
            return bad("listing buckets needs a project");
        }
        let state = self.lock();
        let items: Vec<Value> = state
            .buckets
            .keys()
            .map(|name| json!({ "kind": "storage#bucket", "name": name }))
            .collect();
        json_response(200, json!({ "kind": "storage#buckets", "items": items }))
    }

    /// Answers `DELETE /storage/v1/b/{bucket}`.
    fn delete_bucket(&self, bucket: &str) -> ResponseTemplate {
        if self.lock().buckets.remove(bucket).is_none() {
            return bucket_missing();
        }
        ResponseTemplate::new(204)
    }

    /// Answers `POST /upload/storage/v1/b/{bucket}/o?uploadType=media&name=…`.
    fn upload(
        &self,
        bucket: &str,
        query: &HashMap<String, String>,
        body: &[u8],
    ) -> ResponseTemplate {
        if query.get("uploadType").map(String::as_str) != Some("media") {
            return bad("only `uploadType=media` is implemented");
        }
        let Some(key) = query.get("name").filter(|name| !name.is_empty()) else {
            return bad("an upload needs a name");
        };
        let payload = Bytes::copy_from_slice(body);
        let resource = object_resource(bucket, key, &payload);
        let mut state = self.lock();
        let Some(objects) = state.buckets.get_mut(bucket) else {
            return bucket_missing();
        };
        objects.insert(key.clone(), payload);
        json_response(200, resource)
    }

    /// Answers `GET /storage/v1/b/{bucket}/o/{key}`.
    ///
    /// `alt=media` streams the payload; without it the object resource is returned, which is
    /// how the transport implements `exists`.
    fn get_object(
        &self,
        bucket: &str,
        key: &str,
        query: &HashMap<String, String>,
    ) -> ResponseTemplate {
        let state = self.lock();
        let Some(objects) = state.buckets.get(bucket) else {
            return bucket_missing();
        };
        let Some(payload) = objects.get(key) else {
            return object_missing(bucket, key);
        };
        if query.get("alt").map(String::as_str) == Some("media") {
            return ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(payload.to_vec());
        }
        json_response(200, object_resource(bucket, key, payload))
    }

    /// Answers `DELETE /storage/v1/b/{bucket}/o/{key}`.
    fn delete_object(&self, bucket: &str, key: &str) -> ResponseTemplate {
        let mut state = self.lock();
        let Some(objects) = state.buckets.get_mut(bucket) else {
            return bucket_missing();
        };
        if objects.remove(key).is_none() {
            return object_missing(bucket, key);
        }
        ResponseTemplate::new(204)
    }

    /// Answers `GET /storage/v1/b/{bucket}/o?prefix=…&maxResults=…&pageToken=…`.
    fn list_objects(&self, bucket: &str, query: &HashMap<String, String>) -> ResponseTemplate {
        let state = self.lock();
        let Some(objects) = state.buckets.get(bucket) else {
            return bucket_missing();
        };
        let prefix = query.get("prefix").map_or("", String::as_str);
        let mut keys: Vec<&String> = objects
            .keys()
            .filter(|key| key.starts_with(prefix))
            .collect();
        keys.sort();

        let offset = query
            .get("pageToken")
            .and_then(|token| token.parse::<usize>().ok())
            .unwrap_or(0);
        let requested = query
            .get("maxResults")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(self.page_size);
        let size = requested.min(self.page_size).max(1);
        let start = offset.min(keys.len());
        let end = start.saturating_add(size).min(keys.len());

        let items: Vec<Value> = keys[start..end]
            .iter()
            .map(|key| object_resource(bucket, key, &objects[*key]))
            .collect();
        let mut document = json!({ "kind": "storage#objects", "items": items });
        if end < keys.len() {
            document["nextPageToken"] = json!(end.to_string());
        }
        json_response(200, document)
    }
}

impl Respond for FakeGcs {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let path = request.url.path().to_string();
        let query: HashMap<String, String> = request.url.query_pairs().into_owned().collect();
        let verb = request.method.as_str().to_string();

        if let Some(rest) = path.strip_prefix("/upload/storage/v1/b/") {
            let Some((bucket, tail)) = rest.split_once('/') else {
                return bad("an upload needs a bucket and an object path");
            };
            return if verb == "POST" && tail == "o" {
                self.upload(bucket, &query, &request.body)
            } else {
                bad(&format!("unsupported upload request {verb} {path}"))
            };
        }

        let Some(rest) = path.strip_prefix("/storage/v1/b") else {
            return bad(&format!("unsupported request {verb} {path}"));
        };
        if rest.is_empty() {
            return match verb.as_str() {
                "POST" => self.create_bucket(&query, &request.body),
                "GET" => self.list_buckets(&query),
                _ => bad(&format!("unsupported request {verb} {path}")),
            };
        }

        let rest = rest.strip_prefix('/').unwrap_or_default();
        let (bucket, tail) = rest.split_once('/').unwrap_or((rest, ""));
        match (verb.as_str(), tail) {
            ("DELETE", "") => self.delete_bucket(bucket),
            ("GET", "o") => self.list_objects(bucket, &query),
            ("GET", tail) if tail.starts_with("o/") => {
                self.get_object(bucket, &percent_decode(&tail[2..]), &query)
            }
            ("DELETE", tail) if tail.starts_with("o/") => {
                self.delete_object(bucket, &percent_decode(&tail[2..]))
            }
            _ => bad(&format!("unsupported request {verb} {path}")),
        }
    }
}

/// Builds a JSON response with the matching content type.
fn json_response(status: u16, body: Value) -> ResponseTemplate {
    ResponseTemplate::new(status)
        .insert_header("content-type", "application/json")
        .set_body_json(body)
}

/// The `notFound` document a real bucket answers with for a missing bucket.
fn bucket_missing() -> ResponseTemplate {
    json_response(
        404,
        json!({
            "error": {
                "code": 404,
                "message": "The specified bucket does not exist.",
                "errors": [{ "reason": "notFound", "message": "The specified bucket does not exist." }],
            }
        }),
    )
}

/// The `notFound` document a real bucket answers with for a missing object.
fn object_missing(bucket: &str, key: &str) -> ResponseTemplate {
    json_response(
        404,
        json!({
            "error": {
                "code": 404,
                "message": format!("No such object: {bucket}/{key}"),
                "errors": [{ "reason": "notFound", "message": format!("No such object: {bucket}/{key}") }],
            }
        }),
    )
}

/// The object resource GCS reports for a stored payload.
///
/// `size` is a decimal *string*, the way the JSON API actually encodes 64-bit integers, so the
/// client's string-and-number reader is exercised.
fn object_resource(bucket: &str, key: &str, payload: &Bytes) -> Value {
    json!({
        "kind": "storage#object",
        "bucket": bucket,
        "name": key,
        "size": payload.len().to_string(),
        "updated": UPDATED,
    })
}

/// The error document of an unsupported request, so a wrong path fails loudly.
fn bad(message: &str) -> ResponseTemplate {
    json_response(400, json!({ "error": { "code": 400, "message": message } }))
}

/// Decodes the percent-encoded object name of a request path.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = hex_digit(bytes[index + 1]);
            let low = hex_digit(bytes[index + 2]);
            if let (Some(high), Some(low)) = (high, low) {
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

/// Reads one hexadecimal digit of a percent escape.
fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

// --- fixtures ---------------------------------------------------------------

/// Starts a mock server whose every request is answered by a [`FakeGcs`].
async fn emulator(page_size: usize) -> (MockServer, Arc<Mutex<FakeState>>) {
    let server = MockServer::start().await;
    let state = Arc::new(Mutex::new(FakeState::default()));
    Mock::given(any())
        .respond_with(FakeGcs {
            state: Arc::clone(&state),
            page_size,
        })
        .mount(&server)
        .await;
    (server, state)
}

/// Builds a configuration pointing at `server`.
fn config(server: &MockServer, token: Option<&str>) -> GcsConfig {
    let config = GcsConfig::new(server.uri())
        .with_project("demo-project")
        .with_timeout(Duration::from_secs(5));
    match token {
        Some(token) => config.with_access_token(token),
        None => config,
    }
}

/// Builds a one-node graph so revision metadata is non-trivial.
fn graph(graph_id: &str) -> CapabilityGraph {
    let mut graph = CapabilityGraph::new(graph_id);
    graph.upsert_node(CapabilityNode {
        id: "tool:a".to_string(),
        label: "Tool A".to_string(),
        node_type: "tool".to_string(),
        metadata: BTreeMap::new(),
    });
    graph
}

/// Reads one request header as a string.
fn header(request: &Request, name: &str) -> String {
    request
        .headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

/// Reads one query parameter of a received request.
fn query(request: &Request, name: &str) -> Option<String> {
    request
        .url
        .query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// Removes and returns the single request the mock server received.
async fn only_request(server: &MockServer) -> Request {
    let mut requests = server
        .received_requests()
        .await
        .expect("mock server is running");
    assert_eq!(requests.len(), 1, "expected exactly one request");
    requests.remove(0)
}

/// The keys the emulator holds in `bucket`, ascending.
fn keys(state: &Arc<Mutex<FakeState>>, bucket: &str) -> Vec<String> {
    let state = state.lock().unwrap_or_else(PoisonError::into_inner);
    state
        .buckets
        .get(bucket)
        .map(|objects| objects.keys().cloned().collect())
        .unwrap_or_default()
}

// --- behaviour --------------------------------------------------------------

#[tokio::test]
async fn the_artifact_store_round_trips_over_the_json_api() {
    let (server, state) = emulator(2).await;
    let store = ObjectGraphArtifactStore::open(
        ArtifactTransport::Gcs(config(&server, Some(TOKEN))),
        "artifacts",
        "graph",
    )
    .expect("open the GCS transport");

    let first = store.publish(&graph("g1"), None).await.expect("publish v1");
    assert_eq!(first.version, "v1");
    assert_eq!(first.node_count, 1);
    let second = store.publish(&graph("g2"), None).await.expect("publish v2");
    assert_eq!(second.version, "v2");

    let current = store
        .read(None)
        .await
        .expect("read the current revision")
        .expect("a revision is published");
    assert_eq!(current.graph_id, "g2");
    let pinned = store
        .read(Some("v1"))
        .await
        .expect("read v1")
        .expect("v1 exists");
    assert_eq!(pinned.graph_id, "g1");
    assert!(store.read(Some("v3")).await.expect("read v3").is_none());

    let status = store.status().await.expect("status");
    assert_eq!(status.status, STATUS_READY);
    assert_eq!(status.current_version.as_deref(), Some("v2"));
    assert_eq!(status.node_count, 1);

    let versions = store.versions().await.expect("versions");
    assert_eq!(
        versions
            .iter()
            .map(|revision| revision.version.as_str())
            .collect::<Vec<_>>(),
        vec!["v1", "v2"]
    );

    assert_eq!(
        keys(&state, "artifacts"),
        vec![
            "graph/current.json".to_string(),
            "graph/versions/v1/graph.json".to_string(),
            "graph/versions/v2/graph.json".to_string(),
        ]
    );

    let requests = server.received_requests().await.expect("requests");
    assert!(!requests.is_empty(), "the round trip makes requests");
    for request in &requests {
        assert_eq!(
            header(request, "authorization"),
            format!("Bearer {TOKEN}"),
            "every request is authenticated: {}",
            request.url
        );
    }
}

#[tokio::test]
async fn a_store_without_a_token_sends_anonymous_requests() {
    let (server, _state) = emulator(2).await;
    let store = ObjectGraphArtifactStore::gcs(config(&server, None), "artifacts", "graph")
        .expect("open an anonymous transport");
    store
        .publish(&graph("g1"), None)
        .await
        .expect("publish anonymously");

    let requests = server.received_requests().await.expect("requests");
    assert!(!requests.is_empty());
    for request in &requests {
        assert_eq!(header(request, "authorization"), "", "{}", request.url);
    }
}

/// The same 2048-bit PKCS#8 RSA key the crate's GCS unit tests sign assertions with.
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

/// Writes a complete service-account key file whose `token_uri` is `token_uri`.
fn write_service_account_key(directory: &tempfile::TempDir, token_uri: &str) -> std::path::PathBuf {
    let path = directory.path().join("service-account.json");
    let document = json!({
        "type": "service_account",
        "project_id": "demo-project",
        "private_key_id": "0123456789abcdef",
        "private_key": SERVICE_ACCOUNT_PEM,
        "client_email": "symphony@demo-project.iam.gserviceaccount.com",
        "token_uri": token_uri,
        "universe_domain": "googleapis.com",
    });
    std::fs::write(
        &path,
        serde_json::to_vec(&document).expect("serialize the key file"),
    )
    .expect("write the key file");
    path
}

/// A service-account key file end to end: one minted token, sent on every artifact request.
///
/// The key file is real (the crate's RSA fixture), so the store signs an RS256 JWT assertion and
/// exchanges it at the key's `token_uri` — a loopback wiremock endpoint. The GCS endpoint is the
/// stateful emulator, which lets the test prove both object operations actually completed rather
/// than only that they carried a header. Two operations must mint exactly one token (the cached
/// one is reused) and every storage request must carry that token as a bearer credential.
#[tokio::test]
async fn a_service_account_key_mints_one_bearer_token_for_the_artifact_path() {
    // The key's `token_uri`, answering with the OAuth reply the token source expects.
    let token_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": MINTED_TOKEN,
            "expires_in": 3600,
            "token_type": "Bearer",
        })))
        .mount(&token_server)
        .await;

    // The GCS endpoint, with the bucket the test writes into already present.
    let (server, state) = emulator(2).await;
    state
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .buckets
        .insert("artifacts".to_string(), BTreeMap::new());

    let directory = tempfile::tempdir().expect("temp dir");
    let key_path = write_service_account_key(&directory, &format!("{}/token", token_server.uri()));

    let store =
        GcsObjectStore::from_service_account_file_with(GcsConfig::new(server.uri()), &key_path)
            .expect("build a store from the key file");
    // The key file's `project_id` fills in the configuration's missing project.
    assert_eq!(store.config().project.as_deref(), Some("demo-project"));

    store
        .put(
            "artifacts",
            "graph/versions/v1/graph.json",
            Bytes::from_static(b"{\"v\":1}"),
        )
        .await
        .expect("the first object is signed with the minted token");
    store
        .put(
            "artifacts",
            "graph/current.json",
            Bytes::from_static(b"{\"v\":1,\"current\":true}"),
        )
        .await
        .expect("the second object reuses the cached token");

    // Both writes really landed, through the authenticated path.
    assert_eq!(
        keys(&state, "artifacts"),
        vec![
            "graph/current.json".to_string(),
            "graph/versions/v1/graph.json".to_string(),
        ]
    );

    // (a) One mint serves both operations: the token source caches it.
    let token_requests: Vec<Request> = token_server
        .received_requests()
        .await
        .expect("token server is running");
    assert_eq!(
        token_requests.len(),
        1,
        "two object operations mint exactly one token: {:?}",
        token_requests
            .iter()
            .map(|request| request.url.path().to_string())
            .collect::<Vec<_>>()
    );
    let exchange = &token_requests[0];
    assert_eq!(exchange.url.path(), "/token");
    assert_eq!(
        header(exchange, "content-type"),
        "application/x-www-form-urlencoded",
        "the assertion travels as a form field"
    );
    let form = String::from_utf8_lossy(&exchange.body);
    assert!(
        form.contains("grant_type=") && form.contains("jwt-bearer"),
        "the exchange uses the JWT-bearer grant: {form}"
    );
    assert!(
        form.contains("assertion=") && !form.contains(SERVICE_ACCOUNT_PEM),
        "a signed assertion is posted, and the private key never leaves the process"
    );

    // (b) Every storage request carries the minted token, including both uploads.
    let storage_requests: Vec<Request> = server
        .received_requests()
        .await
        .expect("the emulator is running")
        .into_iter()
        .filter(|request| !request.url.path().starts_with("/token"))
        .collect();
    assert!(
        !storage_requests.is_empty(),
        "the object operations reach the GCS endpoint"
    );
    for request in &storage_requests {
        assert_eq!(
            header(request, "authorization"),
            format!("Bearer {MINTED_TOKEN}"),
            "every artifact request is authenticated: {} {}",
            request.method,
            request.url
        );
    }
    let uploads = storage_requests
        .iter()
        .filter(|request| request.url.path().starts_with("/upload/"))
        .count();
    assert_eq!(uploads, 2, "both `put`s reach the upload endpoint signed");
}

#[tokio::test]
async fn a_bucket_create_conflict_is_a_success_and_sends_the_project() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/storage/v1/b"))
        .respond_with(json_response(
            409,
            json!({ "error": { "code": 409, "message": "conflict" } }),
        ))
        .mount(&server)
        .await;

    let store = GcsObjectStore::new(config(&server, Some(TOKEN))).expect("build the store");
    store
        .create_bucket("artifacts")
        .await
        .expect("409 means the bucket already exists");

    let request = only_request(&server).await;
    assert_eq!(
        query(&request, "project").as_deref(),
        Some("demo-project"),
        "the project travels as a query parameter"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&request.body).expect("JSON body"),
        json!({ "name": "artifacts" })
    );
}

#[tokio::test]
async fn buckets_are_created_listed_and_deleted() {
    let (server, state) = emulator(2).await;
    let store = GcsObjectStore::new(config(&server, None)).expect("build the store");

    store.create_bucket("zeta").await.expect("create zeta");
    store.create_bucket("alpha").await.expect("create alpha");
    store
        .create_bucket("alpha")
        .await
        .expect("creating an existing bucket is a no-op");
    assert_eq!(
        store.list_buckets().await.expect("list buckets"),
        vec!["alpha".to_string(), "zeta".to_string()]
    );

    store.delete_bucket("alpha").await.expect("delete alpha");
    store
        .delete_bucket("alpha")
        .await
        .expect("deleting a missing bucket is a no-op");
    assert_eq!(keys(&state, "zeta"), Vec::<String>::new());
    assert!(state.lock().expect("state").buckets.contains_key("zeta"));
}

#[tokio::test]
async fn bucket_operations_require_a_project() {
    // No request can reach this endpoint, so a missing project must fail before any transfer.
    let store = GcsObjectStore::new(GcsConfig::new("http://127.0.0.1:9")).expect("build the store");
    let create = store
        .create_bucket("artifacts")
        .await
        .expect_err("creating a bucket needs a project");
    assert_eq!(create.kind(), ErrorKind::Validation);
    assert!(create.message().contains("project"), "{}", create.message());

    let list = store
        .list_buckets()
        .await
        .expect_err("listing buckets needs a project");
    assert_eq!(list.kind(), ErrorKind::Validation);
    assert!(list.message().contains("project"), "{}", list.message());
}

#[tokio::test]
async fn a_missing_object_is_tolerated_and_a_missing_bucket_is_not_found() {
    let (server, _state) = emulator(2).await;
    let store = GcsObjectStore::new(config(&server, Some(TOKEN))).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");

    assert!(store
        .get("artifacts", "graph/current.json")
        .await
        .expect("get a missing object")
        .is_none());
    store
        .delete("artifacts", "graph/current.json")
        .await
        .expect("deleting a missing object is a no-op");
    assert!(!store
        .exists("artifacts", "graph/current.json")
        .await
        .expect("exists on a missing object"));

    let error = store
        .get("unknown", "k")
        .await
        .expect_err("a missing bucket is an error");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("unknown"), "{}", error.message());

    let error = store
        .list("unknown", "", None)
        .await
        .expect_err("listing a missing bucket is an error");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("unknown"), "{}", error.message());
}

#[tokio::test]
async fn listings_apply_the_prefix_and_follow_page_tokens() {
    let (server, _state) = emulator(2).await;
    let store = GcsObjectStore::new(config(&server, None)).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");
    for (key, len) in [
        ("a/1", 1usize),
        ("a/2", 2),
        ("a/3", 3),
        ("b/1", 4),
        ("a/4", 5),
    ] {
        store
            .put("artifacts", key, Bytes::from(vec![b'x'; len]))
            .await
            .expect("put");
    }

    let listed = store.list("artifacts", "a/", None).await.expect("list");
    assert_eq!(
        listed
            .iter()
            .map(|meta| meta.key.as_str())
            .collect::<Vec<_>>(),
        vec!["a/1", "a/2", "a/3", "a/4"]
    );
    assert_eq!(
        listed.iter().map(|meta| meta.size).collect::<Vec<_>>(),
        vec![1, 2, 3, 5]
    );
    assert_eq!(
        listed[0].updated_at.timestamp(),
        UPDATED_EPOCH,
        "the emulator reports an RFC 3339 timestamp"
    );

    let page_tokens = server
        .received_requests()
        .await
        .expect("requests")
        .iter()
        .filter_map(|request| query(request, "pageToken"))
        .collect::<Vec<_>>();
    assert!(
        !page_tokens.is_empty(),
        "a page size of two must make the client follow `nextPageToken`"
    );

    let limited = store
        .list("artifacts", "a/", Some(2))
        .await
        .expect("limited list");
    assert_eq!(
        limited
            .iter()
            .map(|meta| meta.key.as_str())
            .collect::<Vec<_>>(),
        vec!["a/1", "a/2"]
    );
}

#[tokio::test]
async fn put_reports_the_size_and_timestamp_of_the_stored_object() {
    let (server, _state) = emulator(2).await;
    let store = GcsObjectStore::new(config(&server, None)).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");

    let meta = store
        .put(
            "artifacts",
            "graph/current.json",
            Bytes::from_static(b"seven!!"),
        )
        .await
        .expect("put");
    assert_eq!(meta.key, "graph/current.json");
    assert_eq!(meta.size, 7);
    assert_eq!(meta.updated_at.timestamp(), UPDATED_EPOCH);

    assert!(store
        .exists("artifacts", "graph/current.json")
        .await
        .expect("exists"));

    let upload = server
        .received_requests()
        .await
        .expect("requests")
        .into_iter()
        .find(|request| request.url.path().starts_with("/upload/"))
        .expect("an upload request");
    assert_eq!(query(&upload, "uploadType").as_deref(), Some("media"));
    assert_eq!(
        query(&upload, "name").as_deref(),
        Some("graph/current.json"),
        "the object name is sent as one escaped query value"
    );
    assert_eq!(upload.body, b"seven!!".to_vec());
}

#[tokio::test]
async fn object_names_with_separators_and_non_ascii_arrive_intact() {
    let (server, _state) = emulator(2).await;
    let store = GcsObjectStore::new(config(&server, None)).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");

    let key = "graph/a b/图.json";
    store
        .put("artifacts", key, Bytes::from_static(b"payload"))
        .await
        .expect("put");
    assert_eq!(
        store
            .get("artifacts", key)
            .await
            .expect("get")
            .expect("the object is stored"),
        Bytes::from_static(b"payload")
    );

    let paths: Vec<String> = server
        .received_requests()
        .await
        .expect("requests")
        .into_iter()
        .filter(|request| request.url.path().contains("/o/"))
        .map(|request| request.url.path().to_string())
        .collect();
    assert_eq!(
        paths.len(),
        1,
        "exactly one object read carries the name: {paths:?}"
    );
    for path in &paths {
        assert!(
            path.contains("%2F") || path.contains("a%20b"),
            "the object name is percent-encoded as one segment: {path}"
        );
    }
}

#[tokio::test]
async fn the_artifact_store_still_rejects_a_blank_bucket() {
    let (server, _state) = emulator(2).await;
    let store =
        ObjectGraphArtifactStore::gcs(config(&server, None), "   ", "graph").expect("build");
    let error = store
        .publish(&graph("g"), None)
        .await
        .expect_err("a blank bucket must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("bucket"), "{}", error.message());
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty(),
        "a blank bucket must not reach the network"
    );
}

#[tokio::test]
async fn a_rejected_request_names_the_operation_status_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/artifacts/o/k"))
        .respond_with(ResponseTemplate::new(500).set_body_string("backend exploded"))
        .mount(&server)
        .await;

    let store = GcsObjectStore::new(config(&server, None)).expect("build the store");
    let error = store
        .get("artifacts", "k")
        .await
        .expect_err("a 500 is an error");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(
        message.contains("GCS get failed with HTTP 500"),
        "{message}"
    );
    assert!(message.contains("backend exploded"), "{message}");
}

#[tokio::test]
async fn an_unreachable_endpoint_is_a_transport_error() {
    // Port 9 (discard) is closed, so the request fails before any GCS endpoint is reached.
    let store = GcsObjectStore::new(
        GcsConfig::new("http://127.0.0.1:9")
            .with_project("demo-project")
            .with_timeout(Duration::from_millis(500)),
    )
    .expect("build the store");
    let error = store
        .create_bucket("artifacts")
        .await
        .expect_err("a closed port must not look like a success");
    assert_ne!(error.kind(), ErrorKind::Validation);
    assert!(!error.message().is_empty());
}

#[test]
fn a_malformed_endpoint_is_a_config_error() {
    for endpoint in ["", "localhost:4443", "ftp://example.com", "http://"] {
        let error = GcsObjectStore::new(GcsConfig::new(endpoint))
            .expect_err("a malformed endpoint must be rejected");
        assert_eq!(error.kind(), ErrorKind::Config, "{endpoint:?}");
    }
}

#[test]
fn the_gcs_transport_descriptor_round_trips() {
    let transport =
        ArtifactTransport::Gcs(GcsConfig::new("http://localhost:4443").with_project("demo"));
    assert_eq!(transport.as_str(), "gcs");

    let document = serde_json::to_value(&transport).expect("serialize the descriptor");
    assert_eq!(document["gcs"]["project"], "demo");
    let decoded: ArtifactTransport =
        serde_json::from_value(document).expect("deserialize the descriptor");
    match decoded {
        ArtifactTransport::Gcs(config) => {
            assert_eq!(config.endpoint, "http://localhost:4443");
            assert_eq!(config.project.as_deref(), Some("demo"));
        }
        other => panic!("expected a GCS transport, got {}", other.as_str()),
    }
}
