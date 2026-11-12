//! Behaviour tests for the graph artifact store: immutable revisions, sequential and
//! pinned versions, the `current` pointer, status reporting, and atomic file writes.
//!
//! Everything is offline and deterministic: the file backend publishes into a
//! `tempfile::TempDir`, document contents are inspected directly, and no assertion
//! depends on the wall clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_symphony::artifact::{
    FileGraphArtifactStore, GraphArtifactStatus, GraphArtifactStore, GraphRevision,
    MemoryGraphArtifactStore, STATUS_MISSING, STATUS_READY,
};
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph};
use openjiuwen_symphony::model::CapabilityDescriptor;
use serde_json::Value;

/// Builds a two-node, one-edge graph so revision metadata is non-trivial.
fn graph(graph_id: &str) -> CapabilityGraph {
    let descriptors = vec![
        CapabilityDescriptor::new("tool", "a", "Tool A").with_description("produces a value"),
        CapabilityDescriptor::new("tool", "b", "Tool B").with_description("consumes a value"),
    ];
    let mut graph = CapabilityGraph::from_descriptors(graph_id, &descriptors);
    assert_eq!(
        graph.node_count(),
        descriptors.len(),
        "from_descriptors must create one node per descriptor"
    );
    graph
        .add_edge(CapabilityEdge {
            source: "tool:a".to_string(),
            target: "tool:b".to_string(),
            relation: "can_feed".to_string(),
            weight: 1.0,
            metadata: BTreeMap::new(),
        })
        .expect("a can_feed edge between the two nodes");
    assert_eq!(graph.edge_count(), 1);
    graph
}

/// Reads a JSON document from disk.
fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("decoding {}: {error}", path.display()))
}

/// Collects every `*.tmp` file below `root`, recursively.
fn temp_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).expect("read_dir") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "tmp") {
                found.push(path);
            }
        }
    }
    found
}

#[tokio::test]
async fn memory_store_numbers_revisions_and_serves_the_current_one() {
    let store = MemoryGraphArtifactStore::new();
    let first = store.publish(&graph("g1"), None).await.expect("publish v1");
    let second = store.publish(&graph("g2"), None).await.expect("publish v2");

    assert_eq!(first.version, "v1");
    assert_eq!(second.version, "v2");
    assert_eq!(first.graph_id, "g1");
    assert_eq!(first.node_count, 2);
    assert_eq!(first.edge_count, 1);
    assert!(!first.created_at.is_empty(), "created_at must be recorded");

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
async fn memory_store_rejects_duplicate_versions() {
    let store = MemoryGraphArtifactStore::new();
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
async fn memory_store_status_is_missing_then_ready() {
    let store = MemoryGraphArtifactStore::default();
    let missing = store.status().await.expect("status");
    assert_eq!(missing, GraphArtifactStatus::missing());
    assert_eq!(missing.status, STATUS_MISSING);
    assert!(missing.current_version.is_none());
    assert!(missing.graph_id.is_none());
    assert!(missing.updated_at.is_none());
    assert!(missing.snapshot.is_empty());
    assert_eq!((missing.node_count, missing.edge_count), (0, 0));

    let revision = store.publish(&graph("g"), None).await.expect("publish");
    let ready = store.status().await.expect("status");
    assert_eq!(ready.status, STATUS_READY);
    assert!(ready.is_ready());
    assert_eq!(ready.current_version.as_deref(), Some("v1"));
    assert_eq!(ready.graph_id.as_deref(), Some("g"));
    assert_eq!(ready.node_count, 2);
    assert_eq!(ready.edge_count, 1);
    assert_eq!(
        ready.updated_at.as_deref(),
        Some(revision.created_at.as_str())
    );
    assert_eq!(
        ready.snapshot.get("graph_id"),
        Some(&Value::String("g".to_string()))
    );
    assert_eq!(
        ready.snapshot.get("current_version"),
        Some(&Value::String("v1".to_string()))
    );
    assert_eq!(ready.snapshot.len(), 2);
}

#[tokio::test]
async fn memory_store_lists_versions_sorted_and_tracks_publication_order() {
    let store = MemoryGraphArtifactStore::new();
    store
        .publish(&graph("gb"), Some("b"))
        .await
        .expect("publish b");
    store
        .publish(&graph("ga"), Some("a"))
        .await
        .expect("publish a");

    let versions = store.versions().await.expect("versions");
    let listed: Vec<&str> = versions
        .iter()
        .map(|revision| revision.version.as_str())
        .collect();
    assert_eq!(listed, vec!["a", "b"]);
    assert_eq!(versions[0].graph_id, "ga");
    assert_eq!(versions[1].graph_id, "gb");

    // The current pointer follows publication order, not the sorted revision order.
    let current = store.read(None).await.expect("read").expect("current");
    assert_eq!(current.graph_id, "ga");
}

#[tokio::test]
async fn memory_store_generated_versions_skip_published_ones() {
    let store = MemoryGraphArtifactStore::new();
    store
        .publish(&graph("g"), Some("v2"))
        .await
        .expect("publish v2");

    let generated = store
        .publish(&graph("g"), None)
        .await
        .expect("publish generated");
    assert_eq!(
        generated.version, "v3",
        "the generated version must not collide with the pinned one"
    );
}

#[tokio::test]
async fn illegal_versions_are_rejected_without_publishing() {
    let store = MemoryGraphArtifactStore::new();
    for version in ["", "   ", ".", "..", "../evil", "bad/version", "with space"] {
        let error = store
            .publish(&graph("g"), Some(version))
            .await
            .expect_err("an illegal version must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "version {version:?}");
        assert!(
            error.message().contains("version"),
            "version {version:?}: {}",
            error.message()
        );
    }
    assert_eq!(store.versions().await.expect("versions"), Vec::new());
    assert_eq!(
        store.status().await.expect("status"),
        GraphArtifactStatus::missing()
    );
}

#[tokio::test]
async fn reading_an_unknown_version_is_none_and_an_illegal_one_is_an_error() {
    let store = MemoryGraphArtifactStore::new();
    assert!(store
        .read(Some("v1"))
        .await
        .expect("unknown read")
        .is_none());

    let error = store
        .read(Some("../evil"))
        .await
        .expect_err("an illegal version must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn memory_store_is_usable_as_a_trait_object() {
    let store: Box<dyn GraphArtifactStore> = Box::new(MemoryGraphArtifactStore::new());
    let revision = store.publish(&graph("g"), None).await.expect("publish");
    assert_eq!(revision.version, "v1");
    assert_eq!(store.versions().await.expect("versions"), vec![revision]);
    assert!(store.status().await.expect("status").is_ready());
}

#[tokio::test]
async fn file_store_publishes_atomically_and_reopens() {
    let directory = tempfile::tempdir().expect("tempdir");
    let root = directory.path();

    let (first, second) = {
        let store = FileGraphArtifactStore::new(root).await.expect("open");
        assert_eq!(store.root(), root);
        let first = store.publish(&graph("g1"), None).await.expect("publish v1");
        let second = store.publish(&graph("g2"), None).await.expect("publish v2");
        (first, second)
    };
    assert_eq!(
        (first.version.as_str(), second.version.as_str()),
        ("v1", "v2")
    );

    // On-disk layout: one envelope per revision, plus the current pointer.
    let v1_path = root.join("versions/v1/graph.json");
    let v2_path = root.join("versions/v2/graph.json");
    assert!(v1_path.is_file(), "{} must exist", v1_path.display());
    assert!(v2_path.is_file(), "{} must exist", v2_path.display());
    let current_path = root.join("current.json");
    assert!(
        current_path.is_file(),
        "{} must exist",
        current_path.display()
    );

    let v1 = read_json(&v1_path);
    assert_eq!(v1["revision"]["version"], Value::String("v1".to_string()));
    assert_eq!(v1["revision"]["graph_id"], Value::String("g1".to_string()));
    assert_eq!(v1["revision"]["node_count"], Value::Number(2.into()));
    assert_eq!(v1["revision"]["edge_count"], Value::Number(1.into()));
    assert!(
        v1["graph"].to_string().contains("\"g1\""),
        "the revision envelope must carry the graph: {v1}"
    );

    let current = read_json(&current_path);
    assert_eq!(
        current["revision"]["version"],
        Value::String("v2".to_string())
    );
    assert!(
        current["graph"].to_string().contains("\"g2\""),
        "the current pointer must carry the graph: {current}"
    );

    assert!(
        temp_files(root).is_empty(),
        "atomic writes must not leave temporary files: {:?}",
        temp_files(root)
    );

    // Reopening hydrates every revision and the pointer from disk.
    let reopened = FileGraphArtifactStore::new(root).await.expect("reopen");
    let versions: Vec<GraphRevision> = reopened.versions().await.expect("versions");
    assert_eq!(versions, vec![first.clone(), second.clone()]);

    let current_graph = reopened
        .read(None)
        .await
        .expect("read current")
        .expect("current revision");
    assert_eq!(current_graph.graph_id, "g2");
    assert_eq!(current_graph.node_count(), 2);
    let pinned = reopened
        .read(Some("v1"))
        .await
        .expect("read v1")
        .expect("v1 revision");
    assert_eq!(pinned.graph_id, "g1");
    assert_eq!(pinned.to_value(), graph("g1").to_value());

    let status = reopened.status().await.expect("status");
    assert_eq!(status, GraphArtifactStatus::ready(&second));
    assert_eq!(status.current_version.as_deref(), Some("v2"));
    assert_eq!(status.graph_id.as_deref(), Some("g2"));
    assert_eq!(
        status.updated_at.as_deref(),
        Some(second.created_at.as_str())
    );
}

#[tokio::test]
async fn file_store_status_on_an_empty_root_is_missing() {
    let directory = tempfile::tempdir().expect("tempdir");
    let store = FileGraphArtifactStore::new(directory.path())
        .await
        .expect("open");

    let status = store.status().await.expect("status");
    assert_eq!(status.status, STATUS_MISSING);
    assert!(status.current_version.is_none());
    assert!(status.graph_id.is_none());
    assert!(status.updated_at.is_none());
    assert!(status.snapshot.is_empty());
    assert_eq!((status.node_count, status.edge_count), (0, 0));
    assert!(store.read(None).await.expect("read").is_none());
    assert_eq!(store.versions().await.expect("versions"), Vec::new());
}

#[tokio::test]
async fn file_store_rejects_duplicate_and_illegal_versions_without_writing() {
    let directory = tempfile::tempdir().expect("tempdir");
    let root = directory.path();
    let store = FileGraphArtifactStore::new(root).await.expect("open");
    store
        .publish(&graph("g"), Some("v1"))
        .await
        .expect("publish v1");

    let duplicate = store
        .publish(&graph("g"), Some("v1"))
        .await
        .expect_err("a duplicate version must be rejected");
    assert_eq!(duplicate.kind(), ErrorKind::Validation);

    for version in ["../evil", "bad/version", "..", "."] {
        let error = store
            .publish(&graph("g"), Some(version))
            .await
            .expect_err("an illegal version must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "version {version:?}");
        assert!(
            error.message().contains("version"),
            "version {version:?}: {}",
            error.message()
        );
    }

    // Nothing escaped the root and no extra revision was published.
    assert_eq!(store.versions().await.expect("versions").len(), 1);
    assert!(!root.join("bad").exists());
    assert!(!root.join("..").join("evil").exists());
    assert_eq!(temp_files(root), Vec::<std::path::PathBuf>::new());
}

#[tokio::test]
async fn file_store_reads_unknown_versions_as_none() {
    let directory = tempfile::tempdir().expect("tempdir");
    let root = directory.path();
    let store = FileGraphArtifactStore::new(root).await.expect("open");
    store.publish(&graph("g"), None).await.expect("publish");

    assert!(store
        .read(Some("v7"))
        .await
        .expect("unknown read")
        .is_none());
    let error = store
        .read(Some("bad/version"))
        .await
        .expect_err("an illegal version must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn concurrent_publications_claim_distinct_versions() {
    let directory = tempfile::tempdir().expect("tempdir");
    let root = directory.path();
    let store = FileGraphArtifactStore::new(root).await.expect("open");

    let g1 = graph("g1");
    let g2 = graph("g2");
    let (first, second) = tokio::join!(store.publish(&g1, None), store.publish(&g2, None));
    let first = first.expect("publish g1");
    let second = second.expect("publish g2");
    let mut versions = vec![first.version, second.version];
    versions.sort();
    assert_eq!(versions, vec!["v1", "v2"]);

    let reopened = FileGraphArtifactStore::new(root).await.expect("reopen");
    assert_eq!(reopened.versions().await.expect("versions").len(), 2);
    assert_eq!(temp_files(root), Vec::<std::path::PathBuf>::new());
}

#[tokio::test]
async fn an_unusable_root_is_a_config_error() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("not-a-directory");
    std::fs::write(&path, b"occupied").expect("write the blocking file");

    let error = FileGraphArtifactStore::new(&path)
        .await
        .expect_err("a file cannot be a graph artifact root");
    assert_eq!(error.kind(), ErrorKind::Config);
}
