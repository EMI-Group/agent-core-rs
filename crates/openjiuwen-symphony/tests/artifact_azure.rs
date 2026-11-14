//! Behaviour tests for the Azure Blob Storage artifact transport.
//!
//! The transport speaks the Blob REST API directly, so the suite drives it against a stateful
//! loopback emulator built on `wiremock`: one responder dispatches by method and path, keeps
//! containers and blobs in memory, and reports the `x-ms-*` exchange, the XML listings, the
//! `NextMarker` pagination and the `BlobNotFound` errors a real container would. Everything is
//! offline and deterministic — the endpoint is the mock server's URI, no credential leaves the
//! process, and no assertion depends on the wall clock (the emulator reports a fixed
//! `Last-Modified`).
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use sha2::{Digest, Sha256};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use openjiuwen_foundation::store::ObjectStore;
use openjiuwen_symphony::artifact::{GraphArtifactStore, STATUS_READY};
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::graph::{CapabilityGraph, CapabilityNode};
use openjiuwen_symphony::object_artifact::{
    ArtifactTransport, AzureBlobConfig, AzureBlobStore, ObjectGraphArtifactStore, API_VERSION,
};

/// Storage account every shared-key case signs with.
const ACCOUNT: &str = "demoaccount";
/// Base64 storage account key every shared-key case signs with.
const KEY: &str = "ZGVtby1rZXktbWF0ZXJpYWw=";
/// The `Last-Modified` timestamp the emulator reports (2026-11-14T08:49:00Z).
const LAST_MODIFIED: &str = "Fri, 14 Nov 2026 08:49:00 GMT";
/// The Unix timestamp of [`LAST_MODIFIED`].
const LAST_MODIFIED_EPOCH: i64 = 1_794_646_140;
/// The SAS token every SAS case configures.
const SAS: &str = "sv=2021-12-02&ss=b&sig=c2lnbmF0dXJl";

// --- the emulator -----------------------------------------------------------
/// The containers and blobs the emulator holds.
#[derive(Debug, Default)]
struct FakeState {
    /// Blobs per container; the outer key is the container name.
    containers: BTreeMap<String, BTreeMap<String, Bytes>>,
}

/// A minimal stateful Azure Blob emulator.
///
/// It answers the subset of the Blob REST API the transport uses and honours a fixed page size,
/// so pagination is exercised even when the client sends a larger `maxresults`.
struct FakeBlob {
    /// Containers and blobs the emulator holds.
    state: Arc<Mutex<FakeState>>,
    /// Largest page the listing endpoint returns.
    page_size: usize,
}

impl FakeBlob {
    /// Locks the emulator state, ignoring a panic of another test thread.
    fn lock(&self) -> MutexGuard<'_, FakeState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Answers `/` with a container listing.
    fn list_containers(&self, query: &HashMap<String, String>) -> ResponseTemplate {
        let state = self.lock();
        let names: Vec<&String> = state.containers.keys().collect();
        let offset = offset_of(query);
        let end = offset.saturating_add(self.page_size).min(names.len());
        let entries = names[offset.min(names.len())..end]
            .iter()
            .map(|name| {
                format!(
                    "<Container><Name>{}</Name><Properties><Last-Modified>{LAST_MODIFIED}\
                     </Last-Modified></Properties></Container>",
                    escape(name)
                )
            })
            .collect::<Vec<_>>()
            .concat();
        let marker = if end < names.len() {
            format!("page-{end}")
        } else {
            String::new()
        };
        xml_response(format!(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?><EnumerationResults><Containers>\
             {entries}</Containers><NextMarker>{marker}</NextMarker></EnumerationResults>"
        ))
    }

    /// Answers `PUT /{container}?restype=container`.
    fn create_container(&self, container: &str) -> ResponseTemplate {
        let mut state = self.lock();
        if state.containers.contains_key(container) {
            return error(409, "ContainerAlreadyExists");
        }
        state
            .containers
            .insert(container.to_string(), BTreeMap::new());
        ResponseTemplate::new(201)
    }

    /// Answers `DELETE /{container}?restype=container`.
    fn delete_container(&self, container: &str) -> ResponseTemplate {
        if self.lock().containers.remove(container).is_none() {
            return error(404, "ContainerNotFound");
        }
        ResponseTemplate::new(202)
    }

    /// Answers `GET /{container}?restype=container&comp=list`.
    fn list_blobs(&self, container: &str, query: &HashMap<String, String>) -> ResponseTemplate {
        let state = self.lock();
        let Some(blobs) = state.containers.get(container) else {
            return error(404, "ContainerNotFound");
        };
        let prefix = query.get("prefix").map_or("", String::as_str);
        let keys: Vec<&String> = blobs.keys().filter(|key| key.starts_with(prefix)).collect();
        let requested = query
            .get("maxresults")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(self.page_size);
        let size = requested.min(self.page_size).max(1);
        let offset = offset_of(query);
        let end = offset.saturating_add(size).min(keys.len());
        let entries = keys[offset.min(keys.len())..end]
            .iter()
            .map(|key| {
                let size = blobs.get(*key).map_or(0, Bytes::len);
                format!(
                    "<Blob><Name>{}</Name><Properties><Last-Modified>{LAST_MODIFIED}\
                     </Last-Modified><Content-Length>{size}</Content-Length></Properties></Blob>",
                    escape(key)
                )
            })
            .collect::<Vec<_>>()
            .concat();
        let marker = if end < keys.len() {
            format!("page-{end}")
        } else {
            String::new()
        };
        xml_response(format!(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?><EnumerationResults ContainerName=\"{}\">\
             <Prefix>{}</Prefix><Blobs>{entries}</Blobs><NextMarker>{marker}</NextMarker>\
             </EnumerationResults>",
            escape(container),
            escape(prefix)
        ))
    }

    /// Answers `PUT /{container}/{blob}`.
    fn put_blob(&self, container: &str, key: &str, body: &[u8]) -> ResponseTemplate {
        let mut state = self.lock();
        let Some(blobs) = state.containers.get_mut(container) else {
            return error(404, "ContainerNotFound");
        };
        let payload = Bytes::copy_from_slice(body);
        blobs.insert(key.to_string(), payload);
        // The real service answers with `Content-Length: 0` (the reply has no body), so the
        // transport reports the uploaded payload length; `Last-Modified` carries the timestamp.
        ResponseTemplate::new(201).insert_header("last-modified", LAST_MODIFIED)
    }

    /// Answers `GET` or `HEAD` on a blob.
    fn read_blob(&self, container: &str, key: &str, head: bool) -> ResponseTemplate {
        let state = self.lock();
        let Some(blobs) = state.containers.get(container) else {
            return error(404, "ContainerNotFound");
        };
        let Some(payload) = blobs.get(key) else {
            return error(404, "BlobNotFound");
        };
        let response = ResponseTemplate::new(200)
            .insert_header("content-length", payload.len().to_string())
            .insert_header("last-modified", LAST_MODIFIED);
        if head {
            response
        } else {
            response.set_body_bytes(payload.to_vec())
        }
    }

    /// Answers `DELETE /{container}/{blob}`.
    fn delete_blob(&self, container: &str, key: &str) -> ResponseTemplate {
        let mut state = self.lock();
        let Some(blobs) = state.containers.get_mut(container) else {
            return error(404, "ContainerNotFound");
        };
        if blobs.remove(key).is_none() {
            return error(404, "BlobNotFound");
        }
        ResponseTemplate::new(202)
    }
}

impl Respond for FakeBlob {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let path = request.url.path().to_string();
        let query: HashMap<String, String> = request.url.query_pairs().into_owned().collect();
        let verb = request.method.as_str().to_string();
        let rest = path.trim_start_matches('/');
        if rest.is_empty() {
            return match verb.as_str() {
                "GET" => self.list_containers(&query),
                _ => bad(&format!("unsupported request {verb} {path}")),
            };
        }
        let (container, tail) = rest.split_once('/').unwrap_or((rest, ""));
        let container = percent_decode(container);
        let container_request =
            query.get("restype").map(String::as_str) == Some("container") && tail.is_empty();
        if container_request {
            return match verb.as_str() {
                "PUT" => self.create_container(&container),
                "DELETE" => self.delete_container(&container),
                "GET" if query.get("comp").map(String::as_str) == Some("list") => {
                    self.list_blobs(&container, &query)
                }
                _ => bad(&format!("unsupported request {verb} {path}")),
            };
        }
        if tail.is_empty() {
            return bad(&format!("unsupported request {verb} {path}"));
        }
        let key = decode_blob_path(tail);
        match verb.as_str() {
            "PUT" => self.put_blob(&container, &key, &request.body),
            "GET" => self.read_blob(&container, &key, false),
            "HEAD" => self.read_blob(&container, &key, true),
            "DELETE" => self.delete_blob(&container, &key),
            _ => bad(&format!("unsupported request {verb} {path}")),
        }
    }
}

/// The page offset a `marker` names.
fn offset_of(query: &HashMap<String, String>) -> usize {
    query
        .get("marker")
        .and_then(|marker| marker.strip_prefix("page-"))
        .and_then(|offset| offset.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Builds an XML response with the matching content type.
fn xml_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "application/xml")
        .set_body_string(body)
}

/// The error document the service answers with for a missing container or blob.
fn error(status: u16, code: &str) -> ResponseTemplate {
    ResponseTemplate::new(status)
        .insert_header("content-type", "application/xml")
        .set_body_string(format!(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?><Error><Code>{code}</Code>\
             <Message>the emulator refused the request</Message></Error>"
        ))
}

/// The error document of an unsupported request, so a wrong path fails loudly.
fn bad(message: &str) -> ResponseTemplate {
    ResponseTemplate::new(400).set_body_string(format!("unsupported: {message}"))
}

/// Escapes the five XML entities a listing emits.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Decodes a `/`-separated percent-encoded blob path.
fn decode_blob_path(path: &str) -> String {
    path.split('/')
        .map(percent_decode)
        .collect::<Vec<_>>()
        .join("/")
}

/// Decodes one percent-encoded path segment.
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
/// Starts a mock server whose every request is answered by a [`FakeBlob`].
async fn emulator(page_size: usize) -> (MockServer, Arc<Mutex<FakeState>>) {
    let server = MockServer::start().await;
    let state = Arc::new(Mutex::new(FakeState::default()));
    Mock::given(any())
        .respond_with(FakeBlob {
            state: Arc::clone(&state),
            page_size,
        })
        .mount(&server)
        .await;
    (server, state)
}

/// Builds a configuration pointing at `server` with shared-key authorization.
fn shared_key_config(server: &MockServer) -> AzureBlobConfig {
    AzureBlobConfig::new(ACCOUNT)
        .with_endpoint(server.uri())
        .with_shared_key(ACCOUNT, KEY)
        .with_timeout(Duration::from_secs(5))
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

/// The keys the emulator holds in `container`, ascending.
fn keys(state: &Arc<Mutex<FakeState>>, container: &str) -> Vec<String> {
    let state = state.lock().unwrap_or_else(PoisonError::into_inner);
    state
        .containers
        .get(container)
        .map(|blobs| blobs.keys().cloned().collect())
        .unwrap_or_default()
}

/// The value the Shared Key signature must have for `request`, computed independently.
///
/// The test re-derives the documented layout (verb, the twelve fixed lines with the declared
/// `Content-Length`, the sorted `x-ms-*` headers, the canonicalized resource with its sorted
/// query parameters) and signs it with an HMAC built here from `sha2`, so a library regression
/// in the layout, the encoding or the header name shows up as a mismatch.
fn expected_authorization(request: &Request, account: &str, key: &str) -> String {
    let mut ms_headers: Vec<(String, String)> = request
        .headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str().to_ascii_lowercase();
            if name.starts_with("x-ms-") {
                Some((
                    name,
                    value.to_str().unwrap_or_default().replace(['\r', '\n'], ""),
                ))
            } else {
                None
            }
        })
        .collect();
    ms_headers.sort();

    let mut resource = format!("/{account}{}", request.url.path());
    let mut params: Vec<(String, String)> = request
        .url
        .query_pairs()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.into_owned()))
        .collect();
    params.sort();
    for (name, value) in &params {
        resource.push('\n');
        resource.push_str(name);
        resource.push(':');
        resource.push_str(value);
    }

    let length = header(request, "content-length");
    let length = if length == "0" { String::new() } else { length };
    let mut string = String::new();
    for line in [
        request.method.as_str(),
        "",
        "",
        length.as_str(),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ] {
        string.push_str(line);
        string.push('\n');
    }
    for (name, value) in &ms_headers {
        string.push_str(name);
        string.push(':');
        string.push_str(value.trim());
        string.push('\n');
    }
    string.push_str(&resource);

    let decoded = STANDARD.decode(key).expect("base64 key");
    format!(
        "SharedKey {account}:{}",
        STANDARD.encode(hmac(&decoded, string.as_bytes()))
    )
}

/// The HMAC-SHA256 of `data` under `key`, built from `sha2` for the test's own comparison.
fn hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
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
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner);
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&outer.finalize());
    digest
}

// --- behaviour --------------------------------------------------------------
#[tokio::test]
async fn the_artifact_store_round_trips_over_the_blob_api() {
    let (server, state) = emulator(2).await;
    let store = ObjectGraphArtifactStore::open(
        ArtifactTransport::AzureBlob(shared_key_config(&server)),
        "artifacts",
        "graph",
    )
    .expect("open the Azure Blob transport");

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
            header(request, "x-ms-version"),
            API_VERSION,
            "every request names the API version: {}",
            request.url
        );
        let date = header(request, "x-ms-date");
        assert!(
            date.ends_with(" GMT") && date.split_whitespace().count() == 6,
            "every request carries an RFC 1123 date: {date:?}"
        );
        assert_eq!(
            header(request, "authorization"),
            expected_authorization(request, ACCOUNT, KEY),
            "the Shared Key signature matches the documented layout: {}",
            request.url
        );
    }
}

#[tokio::test]
async fn a_sas_token_travels_in_the_query_string_without_signing() {
    let (server, _state) = emulator(2).await;
    let store = ObjectGraphArtifactStore::azure_blob(
        AzureBlobConfig::new(ACCOUNT)
            .with_endpoint(server.uri())
            .with_sas_token(format!("?{SAS}"))
            .with_timeout(Duration::from_secs(5)),
        "artifacts",
        "graph",
    )
    .expect("open a SAS transport");

    store.publish(&graph("g1"), None).await.expect("publish");
    assert!(store.read(None).await.expect("read").is_some());

    let requests = server.received_requests().await.expect("requests");
    assert!(!requests.is_empty());
    for request in &requests {
        assert_eq!(
            header(request, "authorization"),
            "",
            "a SAS token replaces the signature: {}",
            request.url
        );
        assert_eq!(
            query(request, "sig").as_deref(),
            Some("c2lnbmF0dXJl"),
            "the token travels as the query string: {}",
            request.url
        );
        assert_eq!(query(request, "sv").as_deref(), Some("2021-12-02"));
        assert_eq!(header(request, "x-ms-version"), API_VERSION);
    }
}

#[tokio::test]
async fn an_anonymous_store_sends_no_credentials() {
    let (server, _state) = emulator(2).await;
    let store = ObjectGraphArtifactStore::azure_blob(
        AzureBlobConfig::default().with_endpoint(server.uri()),
        "artifacts",
        "graph",
    )
    .expect("open an anonymous transport");
    store
        .publish(&graph("g1"), None)
        .await
        .expect("publish anonymously");
    for request in server.received_requests().await.expect("requests") {
        assert_eq!(header(&request, "authorization"), "", "{}", request.url);
        assert!(query(&request, "sig").is_none(), "{}", request.url);
    }
}

#[tokio::test]
async fn blobs_are_written_read_head_and_deleted() {
    let (server, state) = emulator(2).await;
    let store = AzureBlobStore::new(shared_key_config(&server)).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");
    store
        .create_bucket("artifacts")
        .await
        .expect("a 409 means the container already exists");

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
    assert_eq!(meta.updated_at.timestamp(), LAST_MODIFIED_EPOCH);
    assert_eq!(
        keys(&state, "artifacts"),
        vec!["graph/current.json".to_string()],
        "the emulator decodes the blob path back to the key"
    );
    assert!(store
        .exists("artifacts", "graph/current.json")
        .await
        .expect("exists"));
    assert_eq!(
        store
            .get("artifacts", "graph/current.json")
            .await
            .expect("get")
            .expect("the blob is stored"),
        Bytes::from_static(b"seven!!")
    );

    store
        .delete("artifacts", "graph/current.json")
        .await
        .expect("delete");
    assert!(!store
        .exists("artifacts", "graph/current.json")
        .await
        .expect("exists after delete"));
    assert!(store
        .get("artifacts", "graph/current.json")
        .await
        .expect("a missing blob is not an error")
        .is_none());
    store
        .delete("artifacts", "graph/current.json")
        .await
        .expect("deleting a missing blob is a no-op");
    store
        .delete_bucket("artifacts")
        .await
        .expect("delete the container");
    store
        .delete_bucket("artifacts")
        .await
        .expect("deleting a missing container is a no-op");

    let upload = server
        .received_requests()
        .await
        .expect("requests")
        .into_iter()
        .find(|request| request.method.as_str() == "PUT" && request.url.path().contains("/graph/"))
        .expect("a blob upload");
    assert_eq!(header(&upload, "x-ms-blob-type"), "BlockBlob");
    assert_eq!(header(&upload, "content-length"), "7");
    assert_eq!(header(&upload, "content-type"), "");
    assert_eq!(upload.body, b"seven!!".to_vec());
}

#[tokio::test]
async fn bucket_operations_create_list_and_delete() {
    let (server, _state) = emulator(1).await;
    let store = AzureBlobStore::new(shared_key_config(&server)).expect("build the store");
    for container in ["zeta", "alpha", "middle"] {
        store.create_bucket(container).await.expect("create");
    }
    assert_eq!(
        store.list_buckets().await.expect("list containers"),
        vec![
            "alpha".to_string(),
            "middle".to_string(),
            "zeta".to_string()
        ]
    );
    let requests = server.received_requests().await.expect("requests");
    let markers: Vec<String> = requests
        .iter()
        .filter_map(|request| query(request, "marker"))
        .collect();
    assert!(
        !markers.is_empty(),
        "a page size of one must make the client follow `NextMarker`"
    );
    assert!(requests
        .iter()
        .any(|request| query(request, "comp").as_deref() == Some("list")));
}

#[tokio::test]
async fn listings_apply_the_prefix_follow_markers_and_sort() {
    let (server, _state) = emulator(2).await;
    let store = AzureBlobStore::new(shared_key_config(&server)).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");
    for (key, len) in [
        ("b/1", 1usize),
        ("a/1", 2),
        ("a/3", 3),
        ("a&b.json", 4),
        ("a/2", 5),
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
        vec!["a/1", "a/2", "a/3"]
    );
    assert_eq!(
        listed.iter().map(|meta| meta.size).collect::<Vec<_>>(),
        vec![2, 5, 3]
    );
    assert_eq!(
        listed[0].updated_at.timestamp(),
        LAST_MODIFIED_EPOCH,
        "the emulator reports an RFC 1123 timestamp"
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

    let all = store.list("artifacts", "", None).await.expect("unprefixed");
    assert_eq!(
        all.iter().map(|meta| meta.key.as_str()).collect::<Vec<_>>(),
        vec!["a&b.json", "a/1", "a/2", "a/3", "b/1"],
        "the listing is sorted and the XML entity is decoded"
    );

    let markers: Vec<String> = server
        .received_requests()
        .await
        .expect("requests")
        .iter()
        .filter_map(|request| query(request, "marker"))
        .collect();
    assert!(
        markers.len() >= 2,
        "a page size of two must make the client follow `NextMarker`: {markers:?}"
    );
}

#[tokio::test]
async fn a_missing_container_is_not_found_for_a_listing_and_absent_for_a_read() {
    let (server, _state) = emulator(2).await;
    let store = AzureBlobStore::new(shared_key_config(&server)).expect("build the store");
    assert!(store.get("unknown", "k").await.expect("get").is_none());
    assert!(!store.exists("unknown", "k").await.expect("exists"));
    store
        .delete("unknown", "k")
        .await
        .expect("delete is a no-op");
    let error = store
        .list("unknown", "", None)
        .await
        .expect_err("listing a missing container is an error");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("unknown"), "{}", error.message());
}

#[tokio::test]
async fn blob_names_with_separators_and_non_ascii_arrive_intact() {
    let (server, state) = emulator(2).await;
    let store = AzureBlobStore::new(shared_key_config(&server)).expect("build the store");
    store.create_bucket("artifacts").await.expect("create");
    let key = "graph/a b/图.json";
    store
        .put("artifacts", key, Bytes::from_static(b"payload"))
        .await
        .expect("put");
    assert_eq!(
        keys(&state, "artifacts"),
        vec![key.to_string()],
        "the emulator decodes the path back to the original name"
    );
    assert_eq!(
        store
            .get("artifacts", key)
            .await
            .expect("get")
            .expect("the blob is stored"),
        Bytes::from_static(b"payload")
    );
    let paths: Vec<String> = server
        .received_requests()
        .await
        .expect("requests")
        .into_iter()
        .filter(|request| request.method.as_str() == "PUT")
        .map(|request| request.url.path().to_string())
        .collect();
    assert!(
        paths
            .iter()
            .any(|path| path.contains("a%20b") && path.contains("/graph/a%20b/")),
        "the separators stay while each segment is encoded: {paths:?}"
    );
}

#[tokio::test]
async fn the_artifact_store_still_rejects_a_blank_bucket() {
    let (server, _state) = emulator(2).await;
    let store = ObjectGraphArtifactStore::azure_blob(shared_key_config(&server), "   ", "graph")
        .expect("build");
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
    // Every request is answered with a 500 so the rejection mapping is exercised.
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(500).set_body_string("backend exploded"))
        .mount(&server)
        .await;
    let store = AzureBlobStore::new(shared_key_config(&server)).expect("build the store");

    let error = store
        .put("artifacts", "k", Bytes::from_static(b"x"))
        .await
        .expect_err("a 500 is an error");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(
        message.contains("Azure Blob put failed with HTTP 500"),
        "{message}"
    );
    assert!(message.contains("backend exploded"), "{message}");

    let error = store.list_buckets().await.expect_err("a 500 is an error");
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(
        error
            .to_string()
            .contains("Azure Blob list_buckets failed with HTTP 500"),
        "{error}"
    );

    // A container name that cannot be a path segment never reaches the service.
    let error = store
        .create_bucket("bad/name")
        .await
        .expect_err("a slash cannot appear in a container name");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn an_unreachable_endpoint_is_a_transport_error() {
    // Port 9 (discard) is closed, so the request fails before any endpoint is reached.
    let store = AzureBlobStore::new(
        AzureBlobConfig::new(ACCOUNT)
            .with_endpoint("http://127.0.0.1:9")
            .with_shared_key(ACCOUNT, KEY)
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
    for endpoint in ["", "localhost:10000", "ftp://example.com", "http://"] {
        let config = AzureBlobConfig::new(ACCOUNT).with_endpoint(endpoint);
        // An empty endpoint falls back to the account host, so only the others are rejected.
        if endpoint.is_empty() {
            assert!(AzureBlobStore::new(config).is_ok(), "{endpoint:?}");
            continue;
        }
        let error = AzureBlobStore::new(config).expect_err("a malformed endpoint is rejected");
        assert_eq!(error.kind(), ErrorKind::Config, "{endpoint:?}");
    }
}

#[test]
fn the_azure_blob_transport_descriptor_round_trips() {
    let transport = ArtifactTransport::AzureBlob(
        AzureBlobConfig::new(ACCOUNT)
            .with_endpoint("http://127.0.0.1:10000")
            .with_sas_token(SAS),
    );
    assert_eq!(transport.as_str(), "azure_blob");
    let document = serde_json::to_value(&transport).expect("serialize the descriptor");
    assert_eq!(document["azure_blob"]["account"], ACCOUNT);
    assert_eq!(document["azure_blob"]["auth"]["sas"]["token"], SAS);
    let decoded: ArtifactTransport =
        serde_json::from_value(document).expect("deserialize the descriptor");
    match decoded {
        ArtifactTransport::AzureBlob(config) => {
            assert_eq!(config.endpoint, "http://127.0.0.1:10000");
            assert_eq!(config.auth.as_str(), "sas");
        }
        other => panic!("expected an Azure Blob transport, got {}", other.as_str()),
    }
}

#[test]
fn client_credentials_are_a_typed_unsupported_error() {
    let error = AzureBlobStore::from_client_credentials("tenant", "client", "secret")
        .expect_err("the client-credential flow is deferred");
    assert!(openjiuwen_symphony::error::is_unsupported(&error));
    assert!(error.message().contains("client"), "{}", error.message());
    assert!(
        error.message().contains("shared key") || error.message().contains("SAS"),
        "the message must point at the supported credentials: {}",
        error.message()
    );
}
