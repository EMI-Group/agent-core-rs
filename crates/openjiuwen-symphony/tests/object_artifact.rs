//! Behaviour tests for the object-storage graph artifact backend: numbered immutable
//! revisions, the `current` pointer, status reporting, reopen persistence, and the typed
//! unsupported transport. The Google Cloud Storage transport is covered by
//! `tests/artifact_gcs.rs`.
//!
//! Everything is offline and deterministic: the in-memory object store backs most cases, the
//! local backend publishes into a `tempfile::TempDir`, and no assertion depends on the wall
//! clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use openjiuwen_foundation::store::external::S3Config;
use openjiuwen_foundation::store::{InMemoryObjectStore, LocalObjectStore, ObjectStore};
use openjiuwen_symphony::artifact::{
    GraphArtifactStatus, GraphArtifactStore, STATUS_MISSING, STATUS_READY,
};
use openjiuwen_symphony::error::{is_unsupported, ErrorKind};
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode};
use openjiuwen_symphony::object_artifact::{
    ArtifactTransport, ObjectGraphArtifactStore, DEFAULT_BUCKET,
};

/// Builds a two-node, one-edge graph so revision metadata is non-trivial.
fn graph(graph_id: &str) -> CapabilityGraph {
    let mut graph = CapabilityGraph::new(graph_id);
    for (id, label) in [("tool:a", "Tool A"), ("tool:b", "Tool B")] {
        graph.upsert_node(CapabilityNode {
            id: id.to_string(),
            label: label.to_string(),
            node_type: "tool".to_string(),
            metadata: BTreeMap::new(),
        });
    }
    graph
        .add_edge(CapabilityEdge {
            source: "tool:a".to_string(),
            target: "tool:b".to_string(),
            relation: "can_feed".to_string(),
            weight: 1.0,
            metadata: BTreeMap::new(),
        })
        .expect("a can_feed edge between the two nodes");
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
    graph
}

/// Builds a store over a caller-owned in-memory object store, so a test can inspect objects.
fn store_over(objects: &Arc<InMemoryObjectStore>, prefix: &str) -> ObjectGraphArtifactStore {
    ObjectGraphArtifactStore::new(objects.clone(), DEFAULT_BUCKET, prefix)
}

#[tokio::test]
async fn numbered_revisions_are_published_and_served() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    let first = store.publish(&graph("g1"), None).await.expect("publish v1");
    let second = store.publish(&graph("g2"), None).await.expect("publish v2");

    assert_eq!(first.version, "v1");
    assert_eq!(first.graph_id, "g1");
    assert_eq!(first.node_count, 2);
    assert_eq!(first.edge_count, 1);
    assert!(!first.created_at.is_empty(), "created_at must be recorded");
    assert_eq!(second.version, "v2");

    let current = store
        .read(None)
        .await
        .expect("read current")
        .expect("current revision");
    assert_eq!(current.to_value(), graph("g2").to_value());

    let pinned = store
        .read(Some("v1"))
        .await
        .expect("read v1")
        .expect("v1 revision");
    assert_eq!(pinned.to_value(), graph("g1").to_value());
}

#[tokio::test]
async fn versions_list_in_order_and_status_tracks_the_current_revision() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    assert_eq!(
        store.status().await.expect("status before publishing"),
        GraphArtifactStatus::missing()
    );
    assert_eq!(store.versions().await.expect("versions"), Vec::new());

    store.publish(&graph("g1"), None).await.expect("publish v1");
    let first = store.status().await.expect("status after v1");
    assert_eq!(first.status, STATUS_READY);
    assert_eq!(first.current_version.as_deref(), Some("v1"));
    assert_eq!(first.graph_id.as_deref(), Some("g1"));
    assert_eq!(first.node_count, 2);
    assert_eq!(first.edge_count, 1);
    assert_eq!(first.snapshot["current_version"], "v1");
    assert_eq!(first.snapshot["graph_id"], "g1");

    store.publish(&graph("g2"), None).await.expect("publish v2");
    let second = store.status().await.expect("status after v2");
    assert_eq!(second.current_version.as_deref(), Some("v2"));
    assert_eq!(second.snapshot["graph_id"], "g2");

    let listed = store.versions().await.expect("versions");
    assert_eq!(
        listed
            .iter()
            .map(|revision| revision.version.as_str())
            .collect::<Vec<_>>(),
        vec!["v1", "v2"]
    );
    assert_eq!(listed[0].graph_id, "g1");
    assert_eq!(listed[1].graph_id, "g2");
}

#[tokio::test]
async fn published_revisions_are_immutable() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    store
        .publish(&graph("g"), Some("v1"))
        .await
        .expect("publish v1");

    let error = store
        .publish(&graph("g"), Some("v1"))
        .await
        .expect_err("a duplicate version must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("version"),
        "unexpected message: {}",
        error.message()
    );
    assert_eq!(store.versions().await.expect("versions").len(), 1);
}

#[tokio::test]
async fn illegal_versions_fail_on_read_and_write() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    for version in [".", "..", "../evil", "", "  ", "bad/version"] {
        let write = store
            .publish(&graph("g"), Some(version))
            .await
            .expect_err("an illegal version must not be published");
        assert_eq!(write.kind(), ErrorKind::Validation, "write {version:?}");
        assert!(write.message().contains("version"));

        let read = store
            .read(Some(version))
            .await
            .expect_err("an illegal version must not be read");
        assert_eq!(read.kind(), ErrorKind::Validation, "read {version:?}");
        assert!(read.message().contains("version"));
    }
    assert_eq!(store.versions().await.expect("versions"), Vec::new());
}

#[tokio::test]
async fn an_unknown_legal_version_reads_as_absent() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    store.publish(&graph("g"), None).await.expect("publish v1");
    assert!(
        store.read(Some("v9")).await.expect("read v9").is_none(),
        "a legal but unpublished version is absent, not an error"
    );
}

#[tokio::test]
async fn a_generated_version_skips_a_pinned_identifier() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    let pinned = store
        .publish(&graph("g1"), Some("v2"))
        .await
        .expect("pin v2");
    assert_eq!(pinned.version, "v2");

    let generated = store
        .publish(&graph("g2"), None)
        .await
        .expect("publish the next generated revision");
    assert_eq!(
        generated.version, "v3",
        "v2 is already published, so the next free ordinal wins"
    );
    assert_eq!(
        store
            .versions()
            .await
            .expect("versions")
            .iter()
            .map(|revision| revision.version.as_str())
            .collect::<Vec<_>>(),
        vec!["v2", "v3"]
    );
}

#[tokio::test]
async fn the_local_backend_survives_a_reopen() {
    let temp = tempfile::TempDir::new().expect("temp dir");
    let root = temp.path().join("objects");
    {
        let store = ObjectGraphArtifactStore::local(&root, "graph")
            .await
            .expect("open local store");
        store.publish(&graph("g1"), None).await.expect("publish v1");
        store.publish(&graph("g2"), None).await.expect("publish v2");
    }

    let reopened = ObjectGraphArtifactStore::local(&root, "graph")
        .await
        .expect("reopen local store");
    let status = reopened.status().await.expect("status");
    assert_eq!(status.status, STATUS_READY);
    assert_eq!(status.current_version.as_deref(), Some("v2"));
    assert_eq!(
        reopened
            .read(None)
            .await
            .expect("read current")
            .expect("current revision")
            .to_value(),
        graph("g2").to_value()
    );
    assert_eq!(
        reopened
            .read(Some("v1"))
            .await
            .expect("read v1")
            .expect("v1 revision")
            .to_value(),
        graph("g1").to_value()
    );
    assert_eq!(
        reopened
            .versions()
            .await
            .expect("versions")
            .iter()
            .map(|revision| revision.version.as_str())
            .collect::<Vec<_>>(),
        vec!["v1", "v2"]
    );
}

#[tokio::test]
async fn the_local_backend_writes_the_documented_object_keys() {
    let temp = tempfile::TempDir::new().expect("temp dir");
    let root = temp.path().join("objects");
    let store = ObjectGraphArtifactStore::local(&root, "graph")
        .await
        .expect("open local store");
    assert_eq!(store.bucket(), DEFAULT_BUCKET);
    assert_eq!(store.prefix(), "graph");
    store.publish(&graph("g1"), None).await.expect("publish v1");

    let objects = LocalObjectStore::new(&root)
        .await
        .expect("inspect the root");
    let keys = objects
        .list(DEFAULT_BUCKET, "graph/", None)
        .await
        .expect("list the graph prefix")
        .into_iter()
        .map(|meta| meta.key)
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![
            "graph/current.json".to_string(),
            "graph/versions/v1/graph.json".to_string(),
        ]
    );

    let pointer = objects
        .get(DEFAULT_BUCKET, "graph/current.json")
        .await
        .expect("get current")
        .expect("current object");
    let pointer: serde_json::Value = serde_json::from_slice(&pointer).expect("decode current");
    assert_eq!(pointer["version"], "v1");
    assert_eq!(pointer["graph_id"], "g1");
    assert_eq!(pointer["revision"]["version"], "v1");
    assert_eq!(pointer["revision"]["graph_id"], "g1");
    assert_eq!(pointer["graph"]["graph"]["id"], "g1");
}

#[tokio::test]
async fn the_local_backend_tolerates_a_prefix_free_layout() {
    let temp = tempfile::TempDir::new().expect("temp dir");
    let root = temp.path().join("objects");
    let store = ObjectGraphArtifactStore::local(&root, "")
        .await
        .expect("open local store");
    assert_eq!(store.prefix(), "");
    store.publish(&graph("g"), None).await.expect("publish v1");

    let objects = LocalObjectStore::new(&root)
        .await
        .expect("inspect the root");
    let keys = objects
        .list(DEFAULT_BUCKET, "", None)
        .await
        .expect("list the bucket")
        .into_iter()
        .map(|meta| meta.key)
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![
            "current.json".to_string(),
            "versions/v1/graph.json".to_string(),
        ]
    );
    assert!(store.read(None).await.expect("read current").is_some());
}

#[tokio::test]
async fn the_deferred_transport_is_a_typed_unsupported_error() {
    // Google Cloud Storage is a real transport (see `tests/artifact_gcs.rs`); Azure Blob is
    // still the one backend with no in-crate client.
    let error = ObjectGraphArtifactStore::open(ArtifactTransport::AzureBlob, "bucket", "graph")
        .expect_err("a deferred transport must not resolve to a backend");
    assert!(is_unsupported(&error));
    assert!(
        error.message().contains("azure_blob"),
        "the message must name the transport: {}",
        error.message()
    );

    let error = ObjectGraphArtifactStore::open(ArtifactTransport::ObjectStore, "bucket", "graph")
        .expect_err("an unresolvable store must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("new"),
        "the message must point at `new`: {}",
        error.message()
    );
}

#[tokio::test]
async fn a_blank_bucket_is_a_validation_error_on_every_operation() {
    let objects = Arc::new(InMemoryObjectStore::new());
    let store = ObjectGraphArtifactStore::new(objects, "  ", "graph");
    let publish = store
        .publish(&graph("g"), None)
        .await
        .expect_err("a blank bucket must not be published to");
    assert_eq!(publish.kind(), ErrorKind::Validation);
    assert!(publish.message().contains("bucket"));

    let read = store
        .read(None)
        .await
        .expect_err("a blank bucket must not be read");
    assert_eq!(read.kind(), ErrorKind::Validation);
    assert!(read.message().contains("bucket"));

    assert_eq!(
        store.status().await.expect_err("blank bucket").kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        store.versions().await.expect_err("blank bucket").kind(),
        ErrorKind::Validation
    );
}

#[tokio::test]
async fn status_is_missing_until_something_is_published() {
    let store = ObjectGraphArtifactStore::in_memory("graph");
    let status = store.status().await.expect("status");
    assert_eq!(status.status, STATUS_MISSING);
    assert!(!status.is_ready());
    assert!(status.snapshot.is_empty());
    assert!(status.current_version.is_none());
}

#[tokio::test]
async fn a_pointer_to_a_missing_revision_reports_missing() {
    let objects = Arc::new(InMemoryObjectStore::new());
    let store = store_over(&objects, "graph");
    objects
        .create_bucket(DEFAULT_BUCKET)
        .await
        .expect("create bucket");
    let pointer =
        br#"{"version":"v7","graph_id":"g7","revision":{"version":"v7","graph_id":"g7"}}"#;
    objects
        .put(
            DEFAULT_BUCKET,
            "graph/current.json",
            Bytes::from_static(pointer),
        )
        .await
        .expect("write a pointer-only document");

    assert_eq!(store.status().await.expect("status").status, STATUS_MISSING);
    assert!(store.read(None).await.expect("read").is_none());
}

#[tokio::test]
async fn malformed_current_documents_are_errors_not_silent_misses() {
    let objects = Arc::new(InMemoryObjectStore::new());
    let store = store_over(&objects, "graph");
    objects
        .create_bucket(DEFAULT_BUCKET)
        .await
        .expect("create bucket");
    objects
        .put(
            DEFAULT_BUCKET,
            "graph/current.json",
            Bytes::from_static(b"not json"),
        )
        .await
        .expect("write junk");

    let error = store
        .status()
        .await
        .expect_err("malformed JSON must not read as missing");
    assert_eq!(error.kind(), ErrorKind::External);
    let error = store
        .read(None)
        .await
        .expect_err("malformed JSON must not read as missing");
    assert_eq!(error.kind(), ErrorKind::External);
}

#[tokio::test]
async fn unrelated_objects_are_skipped_when_listing_versions() {
    let objects = Arc::new(InMemoryObjectStore::new());
    let store = store_over(&objects, "graph");
    store.publish(&graph("g1"), None).await.expect("publish v1");
    objects
        .put(
            DEFAULT_BUCKET,
            "graph/versions/notes.txt",
            Bytes::from_static(b"ignore me"),
        )
        .await
        .expect("write a stray object");

    let versions = store.versions().await.expect("versions");
    assert_eq!(
        versions
            .iter()
            .map(|revision| revision.version.as_str())
            .collect::<Vec<_>>(),
        vec!["v1"]
    );
}

#[tokio::test]
async fn an_s3_transport_without_an_endpoint_is_rejected() {
    let error = ObjectGraphArtifactStore::s3(S3Config::default(), "bucket", "graph")
        .expect_err("an empty endpoint must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
    let error = ObjectGraphArtifactStore::open(
        ArtifactTransport::S3(S3Config::default()),
        "bucket",
        "graph",
    )
    .expect_err("an empty endpoint must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
}
