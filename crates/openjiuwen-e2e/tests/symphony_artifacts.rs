#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]
//! Symphony artifact scenario: the GCS and Azure Blob transports over loopback endpoints.
//!
//! This binary is the cross-crate seam above `openjiuwen::symphony::object_artifact`: the
//! capability-graph artifact store is built over a transport descriptor, and every byte it
//! writes travels through the shared kernel HTTP client to a stateful `wiremock` emulator that
//! keeps buckets, containers and objects in memory. Nothing reaches a real cloud, no
//! credential leaves the process, and the endpoints are the mock servers' `127.0.0.1:0`
//! addresses.
//!
//! Three transports are exercised end to end — a Google Cloud Storage bucket with an OAuth
//! bearer token, an Azure Blob container with Shared Key authorization, and the same container
//! with a SAS token — plus the rejection paths: a non-2xx reply and a revision name that cannot
//! be an object-key segment.
//!
//! The evidence is recorded state and the wire itself: the objects the emulator holds, the
//! request paths and query strings, the `Authorization` and `x-ms-*` headers the production code
//! emits, the uploaded JSON envelopes, and the graph snapshot read back through the store. The
//! suite never re-derives a signature — it asserts the shape and presence of what the transport
//! signed — and it never asserts a generated id or timestamp (`revision.created_at` is written
//! by the store and only checked for being non-empty).

mod support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use openjiuwen::symphony::artifact::{GraphArtifactStore, STATUS_READY};
use openjiuwen::symphony::error::ErrorKind;
use openjiuwen::symphony::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode};
use openjiuwen::symphony::object_artifact::{
    ArtifactTransport, AzureBlobAuth, AzureBlobConfig, GcsConfig, ObjectGraphArtifactStore,
    API_VERSION,
};
use openjiuwen_test_support::wiremock::matchers::any;
use openjiuwen_test_support::wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};
use serde_json::{json, Value};
use support::within;

/// OAuth access token the authenticated GCS cases configure.
const TOKEN: &str = "ya29.e2e-gcs-token";
/// Bucket and container every case publishes into.
const BUCKET: &str = "artifacts";
/// Key prefix every case publishes below.
const PREFIX: &str = "graph";
/// Storage account every Shared Key case signs with.
const ACCOUNT: &str = "demoaccount";
/// Base64 storage account key every Shared Key case signs with.
const KEY: &str = "ZGVtby1rZXktbWF0ZXJpYWw=";
/// The SAS token the SAS case configures.
const SAS: &str = "sv=2021-12-02&ss=b&sig=c2lnbmF0dXJl";
/// The `updated` timestamp the GCS emulator reports (2024-01-02T03:04:05.678Z).
const GCS_UPDATED: &str = "2024-01-02T03:04:05.678Z";
/// The `Last-Modified` timestamp the Azure emulator reports (2026-11-14T08:49:00Z).
const AZURE_LAST_MODIFIED: &str = "Fri, 14 Nov 2026 08:49:00 GMT";
/// Path of the GCS media-upload endpoint for [`BUCKET`].
const GCS_UPLOAD_PATH: &str = "/upload/storage/v1/b/artifacts/o";
/// Path of the Azure container resource for [`BUCKET`].
const AZURE_CONTAINER_PATH: &str = "/artifacts";

// --- the shared object state ------------------------------------------------

/// What one emulator holds: bucket/container name → object key → payload.
type Objects = BTreeMap<String, BTreeMap<String, Vec<u8>>>;

// --- the GCS emulator -------------------------------------------------------

/// A minimal Google Cloud Storage emulator over the JSON API subset the transport uses.
///
/// One responder dispatches by method and path and keeps the buckets in memory, so a
/// publication is observable both as requests and as stored objects.
struct FakeGcs {
    /// Buckets and objects the emulator holds.
    state: Arc<Mutex<Objects>>,
    /// Status the media-upload endpoint answers instead of `200` (the rejection case).
    upload_status: u16,
}

impl FakeGcs {
    /// Locks the emulator state, ignoring a panic of another test thread.
    fn lock(&self) -> MutexGuard<'_, Objects> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Answers `POST /storage/v1/b?project=…`.
    fn create_bucket(&self, request: &Request) -> ResponseTemplate {
        if query(request, "project").is_none_or(|project| project.is_empty()) {
            return gcs_error(400, "Required parameter: project");
        }
        let Ok(document) = serde_json::from_slice::<Value>(&request.body) else {
            return gcs_error(400, "a bucket create needs a JSON body");
        };
        let Some(name) = document.get("name").and_then(Value::as_str) else {
            return gcs_error(400, "a bucket create needs a name");
        };
        if self.lock().contains_key(name) {
            return gcs_error(409, "You already own this bucket.");
        }
        self.lock().insert(name.to_string(), BTreeMap::new());
        json_response(200, json!({ "kind": "storage#bucket", "name": name }))
    }

    /// Answers `POST /upload/storage/v1/b/{bucket}/o?uploadType=media&name=…`.
    fn upload(&self, request: &Request, bucket: &str) -> ResponseTemplate {
        if self.upload_status != 200 {
            return json_response(
                self.upload_status,
                json!({
                    "error": {
                        "code": self.upload_status,
                        "message": "denied: the caller lacks storage.objects.create",
                    }
                }),
            );
        }
        let Some(key) = query(request, "name") else {
            return gcs_error(400, "an upload needs a name");
        };
        let payload = request.body.clone();
        let size = payload.len();
        self.lock()
            .entry(bucket.to_string())
            .or_default()
            .insert(key.clone(), payload);
        json_response(
            200,
            json!({
                "kind": "storage#object",
                "bucket": bucket,
                "name": key,
                "size": size.to_string(),
                "updated": GCS_UPDATED,
            }),
        )
    }

    /// Answers `GET /storage/v1/b/{bucket}/o/{object}?alt=media`.
    fn download(&self, bucket: &str, key: &str) -> ResponseTemplate {
        let payload = {
            let state = self.lock();
            state
                .get(bucket)
                .and_then(|objects| objects.get(key).cloned())
        };
        match payload {
            Some(bytes) => {
                ResponseTemplate::new(200).set_body_raw(bytes, "application/octet-stream")
            }
            // Deliberately never says "bucket does not exist": GCS uses that phrase to tell a
            // missing bucket from a missing object, and this reply is about the object.
            None => gcs_error(404, "no such object"),
        }
    }

    /// Answers `GET /storage/v1/b/{bucket}/o?prefix=…`.
    fn list(&self, request: &Request, bucket: &str) -> ResponseTemplate {
        let prefix = query(request, "prefix").unwrap_or_default();
        let items: Vec<Value> = {
            let state = self.lock();
            state
                .get(bucket)
                .map(|objects| {
                    objects
                        .iter()
                        .filter(|(key, _)| key.starts_with(&prefix))
                        .map(|(key, payload)| {
                            json!({
                                "kind": "storage#object",
                                "name": key,
                                "size": payload.len().to_string(),
                                "updated": GCS_UPDATED,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        json_response(200, json!({ "kind": "storage#objects", "items": items }))
    }
}

impl Respond for FakeGcs {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parts: Vec<&str> = request
            .url
            .path()
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        match (request.method.as_str(), parts.as_slice()) {
            ("POST", ["storage", "v1", "b"]) => self.create_bucket(request),
            ("POST", ["upload", "storage", "v1", "b", bucket, "o"]) => self.upload(request, bucket),
            ("GET", ["storage", "v1", "b", bucket, "o"]) => self.list(request, bucket),
            ("GET", ["storage", "v1", "b", bucket, "o", object]) => {
                self.download(bucket, &percent_decode(object))
            }
            _ => gcs_error(404, "unsupported request"),
        }
    }
}

// --- the Azure Blob emulator ------------------------------------------------

/// A minimal Azure Blob emulator over the REST subset the transport uses.
///
/// Containers and blobs stay in memory, listings are flat XML, and a rejected container create
/// is reported through `container_status` so the failure path is driven by the same emulator.
struct FakeBlob {
    /// Containers and blobs the emulator holds.
    state: Arc<Mutex<Objects>>,
    /// Status the container-create endpoint answers instead of `201` (the rejection case).
    container_status: u16,
}

impl FakeBlob {
    /// Locks the emulator state, ignoring a panic of another test thread.
    fn lock(&self) -> MutexGuard<'_, Objects> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Answers `PUT /{container}?restype=container`.
    fn create_container(&self, container: &str) -> ResponseTemplate {
        if self.container_status != 201 {
            return azure_error(self.container_status);
        }
        self.lock().entry(container.to_string()).or_default();
        ResponseTemplate::new(201)
    }

    /// Answers `GET /{container}?restype=container&comp=list&prefix=…`.
    fn list_blobs(&self, request: &Request, container: &str) -> ResponseTemplate {
        let prefix = query(request, "prefix").unwrap_or_default();
        let state = self.lock();
        let Some(blobs) = state.get(container) else {
            return azure_error_xml(404, "ContainerNotFound");
        };
        let entries: String = blobs
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .map(|(key, payload)| {
                format!(
                    "<Blob><Name>{key}</Name><Properties>\
                     <Last-Modified>{AZURE_LAST_MODIFIED}</Last-Modified>\
                     <Content-Length>{}</Content-Length></Properties></Blob>",
                    payload.len()
                )
            })
            .collect::<Vec<_>>()
            .concat();
        xml_response(format!(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\
             <EnumerationResults ContainerName=\"{container}\"><Prefix>{prefix}</Prefix>\
             <Blobs>{entries}</Blobs><NextMarker></NextMarker></EnumerationResults>"
        ))
    }

    /// Answers `PUT /{container}/{blob}`.
    fn put_blob(&self, container: &str, key: &str, body: &[u8]) -> ResponseTemplate {
        let mut state = self.lock();
        let Some(blobs) = state.get_mut(container) else {
            return azure_error_xml(404, "ContainerNotFound");
        };
        blobs.insert(key.to_string(), body.to_vec());
        // A block upload answers with an empty body; `Last-Modified` carries the timestamp.
        ResponseTemplate::new(201).insert_header("last-modified", AZURE_LAST_MODIFIED)
    }

    /// Answers `GET /{container}/{blob}`.
    fn read_blob(&self, container: &str, key: &str) -> ResponseTemplate {
        let state = self.lock();
        let Some(blobs) = state.get(container) else {
            return azure_error_xml(404, "ContainerNotFound");
        };
        match blobs.get(key) {
            Some(payload) => ResponseTemplate::new(200)
                .insert_header("content-length", payload.len().to_string())
                .insert_header("last-modified", AZURE_LAST_MODIFIED)
                .set_body_raw(payload.clone(), "application/json"),
            None => azure_error_xml(404, "BlobNotFound"),
        }
    }
}

impl Respond for FakeBlob {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parts: Vec<&str> = request
            .url
            .path()
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        let container_request = query(request, "restype").as_deref() == Some("container");
        match request.method.as_str() {
            "PUT" if container_request => match parts.first().copied() {
                Some(container) => self.create_container(&percent_decode(container)),
                None => azure_error_xml(400, "InvalidUri"),
            },
            "PUT" if parts.len() >= 2 => self.put_blob(
                &percent_decode(parts[0]),
                &decode_blob_path(&parts[1..]),
                &request.body,
            ),
            "GET" if container_request => match parts.first().copied() {
                Some(container) => self.list_blobs(request, &percent_decode(container)),
                None => azure_error_xml(400, "InvalidUri"),
            },
            "GET" if parts.len() >= 2 => {
                self.read_blob(&percent_decode(parts[0]), &decode_blob_path(&parts[1..]))
            }
            _ => azure_error_xml(400, "InvalidRequest"),
        }
    }
}

// --- fixtures ---------------------------------------------------------------

/// Starts a GCS endpoint whose every request is answered by a [`FakeGcs`].
async fn gcs_emulator(upload_status: u16) -> (MockServer, Arc<Mutex<Objects>>) {
    let server = within("the GCS loopback endpoint starts", MockServer::start()).await;
    let state = Arc::new(Mutex::new(Objects::new()));
    let responder = FakeGcs {
        state: Arc::clone(&state),
        upload_status,
    };
    within(
        "the GCS emulator mounts",
        Mock::given(any()).respond_with(responder).mount(&server),
    )
    .await;
    (server, state)
}

/// Starts an Azure Blob endpoint whose every request is answered by a [`FakeBlob`].
async fn azure_emulator(container_status: u16) -> (MockServer, Arc<Mutex<Objects>>) {
    let server = within("the Azure loopback endpoint starts", MockServer::start()).await;
    let state = Arc::new(Mutex::new(Objects::new()));
    let responder = FakeBlob {
        state: Arc::clone(&state),
        container_status,
    };
    within(
        "the Azure emulator mounts",
        Mock::given(any()).respond_with(responder).mount(&server),
    )
    .await;
    (server, state)
}

/// A GCS configuration pointing at `server` with a project and a bearer token.
fn gcs_config(server: &MockServer) -> GcsConfig {
    GcsConfig::new(server.uri())
        .with_project("demo-project")
        .with_access_token(TOKEN)
}

/// An Azure configuration pointing at `server` with Shared Key credentials.
fn azure_shared_key_config(server: &MockServer) -> AzureBlobConfig {
    AzureBlobConfig::new(ACCOUNT)
        .with_endpoint(server.uri())
        .with_shared_key(ACCOUNT, KEY)
}

/// A two-node, one-edge graph, so revision metadata is non-trivial.
fn graph(graph_id: &str) -> CapabilityGraph {
    let mut graph = CapabilityGraph::new(graph_id);
    graph.upsert_node(CapabilityNode::new("tool:a", "Tool A", "tool"));
    graph.upsert_node(CapabilityNode::new("tool:b", "Tool B", "tool"));
    graph
        .add_edge(CapabilityEdge::new("tool:a", "tool:b", "can_feed", 1.0))
        .expect("the edge links two inserted nodes");
    graph
}

/// The object keys the emulator holds in `bucket`, ascending.
fn keys(state: &Arc<Mutex<Objects>>, bucket: &str) -> Vec<String> {
    let state = state.lock().unwrap_or_else(PoisonError::into_inner);
    state
        .get(bucket)
        .map(|objects| objects.keys().cloned().collect())
        .unwrap_or_default()
}

/// The three keys one store publishes for `versions` `v1` and `v2`.
fn expected_keys() -> Vec<String> {
    vec![
        "graph/current.json".to_string(),
        "graph/versions/v1/graph.json".to_string(),
        "graph/versions/v2/graph.json".to_string(),
    ]
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

/// Reads one query parameter of a received request, already percent-decoded.
fn query(request: &Request, name: &str) -> Option<String> {
    request
        .url
        .query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// Percent-decodes one URL path segment.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let (Some(high), Some(low)) =
                (hex_digit(bytes[index + 1]), hex_digit(bytes[index + 2]))
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

/// Decodes one uppercase hexadecimal digit.
fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Decodes a `/`-separated blob name: each segment is percent-decoded, the separators stay.
fn decode_blob_path(segments: &[&str]) -> String {
    segments
        .iter()
        .map(|segment| percent_decode(segment))
        .collect::<Vec<_>>()
        .join("/")
}

/// A JSON reply with `status` and `body`.
fn json_response(status: u16, body: impl Into<Value>) -> ResponseTemplate {
    let document = body.into().to_string();
    ResponseTemplate::new(status).set_body_raw(document, "application/json")
}

/// A GCS error document with `status` and `message`.
fn gcs_error(status: u16, message: &str) -> ResponseTemplate {
    json_response(
        status,
        json!({ "error": { "code": status, "message": message } }),
    )
}

/// An XML reply with `body`.
fn xml_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "application/xml")
}

/// An Azure error document with `status` and a service `code`.
fn azure_error_xml(status: u16, code: &str) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_raw(
        format!("<?xml version=\"1.0\" encoding=\"utf-8\"?><Error><Code>{code}</Code></Error>"),
        "application/xml",
    )
}

/// The rejection the container-create endpoint answers with.
fn azure_error(status: u16) -> ResponseTemplate {
    azure_error_xml(status, "ServerBusy")
}

/// Asserts the Shared Key `Authorization` shape without re-deriving the signature.
///
/// Only the documented shape is checked: the `SharedKey {account}:{base64}` layout, the account
/// the configuration names, and a base64 HMAC-SHA256 digest (44 characters, one `=` pad).
fn assert_shared_key_signature(request: &Request) {
    let authorization = header(request, "authorization");
    let (scheme, rest) = authorization
        .split_once(' ')
        .unwrap_or((authorization.as_str(), ""));
    assert_eq!(scheme, "SharedKey", "{}: {authorization}", request.url);
    let (account, signature) = rest.split_once(':').unwrap_or((rest, ""));
    assert_eq!(account, ACCOUNT, "{}: {authorization}", request.url);
    assert_eq!(
        signature.len(),
        44,
        "a base64 HMAC-SHA256 is 44 characters: {signature}"
    );
    assert!(
        signature.ends_with('='),
        "a 32-byte digest needs one pad: {signature}"
    );
    assert!(
        signature
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "+/=".contains(character)),
        "the signature is standard base64: {signature}"
    );
}

// --- GCS --------------------------------------------------------------------

#[tokio::test]
async fn gcs_artifact_round_trips_over_a_loopback_endpoint() {
    let (server, state) = gcs_emulator(200).await;
    let store =
        ObjectGraphArtifactStore::open(ArtifactTransport::Gcs(gcs_config(&server)), BUCKET, PREFIX)
            .expect("open the GCS transport");

    let first = within("the v1 publication", store.publish(&graph("g1"), None))
        .await
        .expect("publish v1");
    assert_eq!(first.version, "v1");
    assert_eq!(first.graph_id, "g1");
    assert_eq!(first.node_count, 2);
    assert_eq!(first.edge_count, 1);
    assert!(!first.created_at.is_empty(), "the envelope is timestamped");
    let second = within("the v2 publication", store.publish(&graph("g2"), None))
        .await
        .expect("publish v2");
    assert_eq!(second.version, "v2");

    let current = within("the current revision reads back", store.read(None))
        .await
        .expect("read the current revision")
        .expect("a revision is published");
    assert_eq!(current.graph_id, "g2");
    assert_eq!(current.node_count(), 2);
    assert_eq!(current.edge_count(), 1);
    assert!(current.has_node("tool:a"));
    assert!(current.has_node("tool:b"));
    assert_eq!(current.to_value(), graph("g2").to_value());

    let pinned = within("v1 reads back", store.read(Some("v1")))
        .await
        .expect("read v1")
        .expect("v1 exists");
    assert_eq!(pinned, graph("g1"));
    let absent = within("v3 reads back", store.read(Some("v3")))
        .await
        .expect("read v3");
    assert!(absent.is_none(), "v3 was never published");

    let status = within("the graph status", store.status())
        .await
        .expect("status");
    assert_eq!(status.status, STATUS_READY);
    assert_eq!(status.current_version.as_deref(), Some("v2"));
    assert_eq!(status.node_count, 2);
    assert_eq!(status.edge_count, 1);
    let versions = within("the revision index", store.versions())
        .await
        .expect("versions");
    assert_eq!(
        versions
            .iter()
            .map(|revision| revision.version.as_str())
            .collect::<Vec<_>>(),
        vec!["v1", "v2"]
    );

    assert_eq!(
        keys(&state, BUCKET),
        expected_keys(),
        "the bucket layout mirrors the filesystem backend"
    );

    let requests = within("the recorded requests", server.received_requests())
        .await
        .expect("the mock server records requests");
    assert!(!requests.is_empty(), "the round trip makes requests");
    for request in &requests {
        assert_eq!(
            header(request, "authorization"),
            format!("Bearer {TOKEN}"),
            "every request carries the bearer token: {}",
            request.url
        );
    }

    // Bucket creation names the project and the bucket, and happens once.
    let creates: Vec<&Request> = requests
        .iter()
        .filter(|request| {
            request.method.as_str() == "POST" && request.url.path() == "/storage/v1/b"
        })
        .collect();
    assert_eq!(creates.len(), 1, "the bucket is created on first publish");
    assert_eq!(
        query(creates[0], "project").as_deref(),
        Some("demo-project")
    );
    let body = serde_json::from_slice::<Value>(&creates[0].body).expect("the create body is JSON");
    assert_eq!(body["name"], BUCKET);

    // Every upload uses the media form and stores the revision envelope.
    let uploads: Vec<&Request> = requests
        .iter()
        .filter(|request| {
            request.method.as_str() == "POST" && request.url.path() == GCS_UPLOAD_PATH
        })
        .collect();
    assert_eq!(
        uploads
            .iter()
            .map(|request| query(request, "name").expect("an upload names its object"))
            .collect::<Vec<_>>(),
        vec![
            "graph/versions/v1/graph.json".to_string(),
            "graph/current.json".to_string(),
            "graph/versions/v2/graph.json".to_string(),
            "graph/current.json".to_string(),
        ],
        "the revision object is written before the current pointer"
    );
    for request in &uploads {
        assert_eq!(query(request, "uploadType").as_deref(), Some("media"));
        assert_eq!(
            header(request, "content-type"),
            "application/octet-stream",
            "{}",
            request.url
        );
    }
    let envelope = serde_json::from_slice::<Value>(&uploads[0].body)
        .expect("the uploaded body is the revision envelope");
    assert_eq!(envelope["revision"]["version"], "v1");
    assert_eq!(envelope["revision"]["graph_id"], "g1");
    assert_eq!(envelope["revision"]["node_count"], 2);
    assert_eq!(envelope["revision"]["edge_count"], 1);
    assert_eq!(envelope["graph"]["graph"]["id"], "g1");
    assert_eq!(
        envelope["graph"]["graph"]["nodes"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(envelope["graph"]["graph"]["edges"][0]["source"], "tool:a");
    assert_eq!(envelope["graph"]["graph"]["edges"][0]["target"], "tool:b");

    // A download names the object as one percent-encoded segment, so `/` never splits the path.
    let downloads: Vec<&Request> = requests
        .iter()
        .filter(|request| {
            request.method.as_str() == "GET"
                && request.url.path()
                    == "/storage/v1/b/artifacts/o/graph%2Fversions%2Fv1%2Fgraph.json"
        })
        .collect();
    assert!(
        !downloads.is_empty(),
        "v1 is fetched by its percent-encoded name"
    );
    // The existence probe before a publication carries no query; every payload read asks for
    // the media representation.
    for request in &downloads {
        assert!(
            query(request, "alt").as_deref() == Some("media") || request.url.query().is_none(),
            "a GET of the object is either a probe or a media read: {}",
            request.url
        );
    }
    assert!(
        downloads
            .iter()
            .any(|request| query(request, "alt").as_deref() == Some("media")),
        "the payload is read through `alt=media`"
    );
}

// --- Azure Blob, Shared Key -------------------------------------------------

#[tokio::test]
async fn azure_blob_shared_key_requests_are_signed_over_the_block_blob_api() {
    let (server, state) = azure_emulator(201).await;
    let store =
        ObjectGraphArtifactStore::azure_blob(azure_shared_key_config(&server), BUCKET, PREFIX)
            .expect("open the Azure Blob transport");
    assert_eq!(store.bucket(), BUCKET);
    assert_eq!(store.prefix(), PREFIX);

    let first = within("the v1 publication", store.publish(&graph("g1"), None))
        .await
        .expect("publish v1");
    assert_eq!(first.version, "v1");
    assert_eq!(first.node_count, 2);
    assert_eq!(first.edge_count, 1);
    let second = within("the v2 publication", store.publish(&graph("g2"), None))
        .await
        .expect("publish v2");
    assert_eq!(second.version, "v2");

    let current = within("the current revision reads back", store.read(None))
        .await
        .expect("read the current revision")
        .expect("a revision is published");
    assert_eq!(current, graph("g2"));
    let pinned = within("v1 reads back", store.read(Some("v1")))
        .await
        .expect("read v1")
        .expect("v1 exists");
    assert_eq!(pinned, graph("g1"));
    let status = within("the graph status", store.status())
        .await
        .expect("status");
    assert_eq!(status.status, STATUS_READY);
    assert_eq!(status.current_version.as_deref(), Some("v2"));
    assert_eq!(
        keys(&state, BUCKET),
        expected_keys(),
        "blob names keep their `/` separators and the prefix"
    );

    let requests = within("the recorded requests", server.received_requests())
        .await
        .expect("the mock server records requests");
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
            "every request carries an RFC 1123 date: {date:?} for {}",
            request.url
        );
        assert_shared_key_signature(request);
    }

    // The container is created with the REST container form before any blob is written.
    let creates: Vec<&Request> = requests
        .iter()
        .filter(|request| {
            request.method.as_str() == "PUT" && request.url.path() == AZURE_CONTAINER_PATH
        })
        .collect();
    assert_eq!(
        creates.len(),
        1,
        "the container is created on first publish"
    );
    assert_eq!(query(creates[0], "restype").as_deref(), Some("container"));
    assert!(creates[0].body.is_empty(), "a container create has no body");

    // Every blob write is a block-blob PUT under the container path.
    let uploads: Vec<&Request> = requests
        .iter()
        .filter(|request| {
            request.method.as_str() == "PUT" && request.url.path().starts_with("/artifacts/graph/")
        })
        .collect();
    assert_eq!(
        uploads
            .iter()
            .map(|request| request.url.path().to_string())
            .collect::<Vec<_>>(),
        vec![
            "/artifacts/graph/versions/v1/graph.json".to_string(),
            "/artifacts/graph/current.json".to_string(),
            "/artifacts/graph/versions/v2/graph.json".to_string(),
            "/artifacts/graph/current.json".to_string(),
        ],
        "the blob path layout mirrors the filesystem backend"
    );
    for request in &uploads {
        assert_eq!(
            header(request, "x-ms-blob-type"),
            "BlockBlob",
            "{}",
            request.url
        );
        assert_eq!(
            header(request, "content-length"),
            request.body.len().to_string(),
            "the signed length is the payload length: {}",
            request.url
        );
    }
    let envelope = serde_json::from_slice::<Value>(&uploads[0].body)
        .expect("the uploaded blob is the revision envelope");
    assert_eq!(envelope["revision"]["version"], "v1");
    assert_eq!(envelope["revision"]["graph_id"], "g1");
    assert_eq!(envelope["revision"]["node_count"], 2);
    assert_eq!(envelope["revision"]["edge_count"], 1);
    assert_eq!(envelope["graph"]["graph"]["id"], "g1");
    assert_eq!(
        envelope["graph"]["graph"]["edges"][0]["relation"],
        "can_feed"
    );

    // A read is a plain GET on the blob path, still signed and versioned.
    assert!(
        requests.iter().any(|request| {
            request.method.as_str() == "GET"
                && request.url.path() == "/artifacts/graph/versions/v1/graph.json"
        }),
        "v1 is fetched by its blob path"
    );
}

// --- Azure Blob, SAS --------------------------------------------------------

#[tokio::test]
async fn azure_blob_sas_tokens_travel_in_the_query_string_without_signing() {
    let (server, state) = azure_emulator(201).await;
    let store = ObjectGraphArtifactStore::open(
        ArtifactTransport::AzureBlob(
            AzureBlobConfig::new(ACCOUNT)
                .with_endpoint(server.uri())
                .with_auth(AzureBlobAuth::sas(SAS)),
        ),
        BUCKET,
        PREFIX,
    )
    .expect("open the SAS transport");

    let published = within("the publication", store.publish(&graph("g1"), None))
        .await
        .expect("publish with a SAS token");
    assert_eq!(published.version, "v1");
    let reloaded = within("the publication reads back", store.read(None))
        .await
        .expect("read the current revision")
        .expect("a revision is published");
    assert_eq!(reloaded, graph("g1"));
    assert_eq!(
        keys(&state, BUCKET),
        vec![
            "graph/current.json".to_string(),
            "graph/versions/v1/graph.json".to_string(),
        ]
    );

    let requests = within("the recorded requests", server.received_requests())
        .await
        .expect("the mock server records requests");
    assert!(!requests.is_empty(), "the round trip makes requests");
    for request in &requests {
        assert!(
            header(request, "authorization").is_empty(),
            "a SAS request carries no signature header: {}",
            request.url
        );
        assert_eq!(
            query(request, "sig").as_deref(),
            Some("c2lnbmF0dXJl"),
            "the token travels in the query string: {}",
            request.url
        );
        assert_eq!(query(request, "sv").as_deref(), Some("2021-12-02"));
        assert_eq!(query(request, "ss").as_deref(), Some("b"));
        assert_eq!(
            header(request, "x-ms-version"),
            API_VERSION,
            "the version header is still sent: {}",
            request.url
        );
        assert!(
            !request.url.path().contains("sig"),
            "the token never lands in the path: {}",
            request.url
        );
    }

    // A request that already has query parameters keeps them next to the token.
    let create = requests
        .iter()
        .find(|request| {
            request.method.as_str() == "PUT" && request.url.path() == AZURE_CONTAINER_PATH
        })
        .expect("the container is created");
    assert_eq!(query(create, "restype").as_deref(), Some("container"));
    assert_eq!(query(create, "sig").as_deref(), Some("c2lnbmF0dXJl"));
}

// --- rejections -------------------------------------------------------------

#[tokio::test]
async fn rejected_requests_and_bad_revision_names_surface_typed_errors() {
    // A rejected media upload is an external error naming the operation, the status and a body
    // excerpt — and nothing is stored.
    let (server, state) = gcs_emulator(403).await;
    let store = ObjectGraphArtifactStore::gcs(gcs_config(&server), BUCKET, PREFIX)
        .expect("open the GCS transport");
    let error = within("the rejected upload", store.publish(&graph("g1"), None))
        .await
        .expect_err("a 403 must not look like a success");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(
        message.contains("GCS put failed with HTTP 403"),
        "the message names the operation and the status: {message}"
    );
    assert!(
        message.contains("storage.objects.create"),
        "the body excerpt is kept: {message}"
    );
    assert!(
        keys(&state, BUCKET).is_empty(),
        "a rejected upload stores no object"
    );

    // A rejected container create is reported the same way and leaves the container absent.
    let (server, state) = azure_emulator(500).await;
    let store =
        ObjectGraphArtifactStore::azure_blob(azure_shared_key_config(&server), BUCKET, PREFIX)
            .expect("open the Azure Blob transport");
    let error = within(
        "the rejected container create",
        store.publish(&graph("g1"), None),
    )
    .await
    .expect_err("a 500 must not look like a success");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(
        message.contains("Azure Blob create_bucket failed with HTTP 500"),
        "the message names the operation and the status: {message}"
    );
    assert!(
        message.contains("ServerBusy"),
        "the body excerpt is kept: {message}"
    );
    assert!(
        keys(&state, BUCKET).is_empty(),
        "a rejected container create stores nothing"
    );

    // A revision identifier that cannot be an object-key segment is refused before the wire.
    let (server, _state) = gcs_emulator(200).await;
    let store = ObjectGraphArtifactStore::gcs(gcs_config(&server), BUCKET, PREFIX)
        .expect("open the GCS transport");
    for version in ["../evil", "bad/version", "with space", "   "] {
        let error = within(
            "the rejected publication",
            store.publish(&graph("g1"), Some(version)),
        )
        .await
        .expect_err("a malformed version must be refused");
        assert_eq!(error.kind(), ErrorKind::Validation, "{version:?}");
        assert!(
            error.message().contains("version"),
            "{version:?} names the field: {error}"
        );
        let error = within("the rejected read", store.read(Some(version)))
            .await
            .expect_err("a malformed version must be refused");
        assert_eq!(error.kind(), ErrorKind::Validation, "{version:?}");
    }
    let requests = within("the recorded requests", server.received_requests())
        .await
        .expect("the mock server records requests");
    assert!(
        requests.is_empty(),
        "a rejected revision name never reaches the endpoint"
    );
}
