//! Behaviour tests for [`CapabilityGraph`]: node bookkeeping, edge validation and
//! upsert, deterministic reads (`outgoing`/`incoming`/`roots`/`leaves`), and the JGF v2
//! document round-trip.
//!
//! Everything here is offline and deterministic: no clock, no network, no shared state.
//! Artifact-store tests live in their own file; this one covers only the graph type.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use openjiuwen_symphony::error::{Error, ErrorKind};
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode, CAN_FEED};
use openjiuwen_symphony::model::CapabilityDescriptor;
use serde_json::json;

/// Builds a descriptor with just the fields the graph reads.
fn descriptor(capability_type: &str, id: &str, name: &str) -> CapabilityDescriptor {
    CapabilityDescriptor::new(capability_type, id, name)
}

/// Builds a graph whose nodes are `(id, label, node_type)`.
fn graph_with(nodes: &[(&str, &str, &str)]) -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("test-graph");
    for (id, label, node_type) in nodes {
        graph.upsert_node(CapabilityNode::new(*id, *label, *node_type));
    }
    graph
}

/// Asserts that `error` is a validation failure whose message mentions `field`.
fn assert_validation(error: &Error, field: &str) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "unexpected error kind for `{error}`"
    );
    assert!(
        error.message().contains(field),
        "message `{}` does not mention `{field}`",
        error.message()
    );
}

/// The edge identities `(source, target, relation)` in storage order.
fn edge_keys(graph: &CapabilityGraph) -> Vec<(String, String, String)> {
    graph
        .edges
        .iter()
        .map(|edge| {
            (
                edge.source.clone(),
                edge.target.clone(),
                edge.relation.clone(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Construction from descriptors
// ---------------------------------------------------------------------------

#[test]
fn from_descriptors_creates_one_node_per_capability() {
    let capabilities = vec![
        descriptor("tool", "sum", "Sum Numbers"),
        descriptor("workflow", "weather", "Weather Forecast"),
    ];
    let graph = CapabilityGraph::from_descriptors("catalog", &capabilities);

    assert_eq!(graph.graph_id, "catalog");
    assert_eq!(graph.graph_type, "capability_graph");
    assert!(graph.directed);
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 0);
    assert!(graph.metadata.is_empty());
    assert_eq!(
        graph.nodes.keys().cloned().collect::<Vec<_>>(),
        vec!["tool:sum".to_string(), "workflow:weather".to_string()]
    );

    let node = graph.node("tool:sum").expect("sum node");
    assert_eq!(node.label, "Sum Numbers");
    assert_eq!(node.node_type, "tool");
    assert_eq!(node.metadata["capability_id"], json!("sum"));
    assert_eq!(node.metadata["capability_type"], json!("tool"));
    assert!(graph.has_node("workflow:weather"));
    assert!(graph.node("tool:missing").is_none());
    assert!(!graph.has_node("tool:missing"));
}

#[test]
fn from_descriptors_collapses_duplicate_qualified_ids() {
    let capabilities = vec![
        descriptor("tool", "sum", "First"),
        descriptor("tool", "sum", "Second"),
    ];
    let graph = CapabilityGraph::from_descriptors("catalog", &capabilities);

    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.node("tool:sum").expect("sum node").label, "Second");
}

#[test]
fn an_empty_catalog_produces_an_empty_graph() {
    let graph = CapabilityGraph::from_descriptors("empty", &[]);
    assert!(graph.nodes.is_empty());
    assert!(graph.edges.is_empty());
    assert!(graph.roots().is_empty());
    assert!(graph.leaves().is_empty());
}

// ---------------------------------------------------------------------------
// Node bookkeeping
// ---------------------------------------------------------------------------

#[test]
fn upsert_node_replaces_the_node_in_place() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    graph.upsert_node(
        CapabilityNode::new("a", "A renamed", "skill").with_metadata("owner", json!("ops")),
    );

    assert_eq!(graph.node_count(), 2);
    let node = graph.node("a").expect("a");
    assert_eq!(node.label, "A renamed");
    assert_eq!(node.node_type, "skill");
    assert_eq!(node.metadata["owner"], json!("ops"));
}

#[test]
fn upsert_node_keeps_edges_of_the_replaced_node() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("add edge");
    graph.upsert_node(CapabilityNode::new("a", "A2", "tool"));

    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.outgoing("a").len(), 1);
}

#[test]
fn remove_node_drops_incident_edges_and_reports_existence() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool"), ("c", "C", "tool")]);
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("a->b");
    graph
        .add_edge(CapabilityEdge::can_feed("b", "c"))
        .expect("b->c");
    graph
        .add_edge(CapabilityEdge::can_feed("a", "c"))
        .expect("a->c");

    assert!(graph.remove_node("b"));
    assert!(!graph.has_node("b"));
    assert_eq!(graph.node_count(), 2);
    assert_eq!(
        edge_keys(&graph),
        vec![("a".to_string(), "c".to_string(), CAN_FEED.to_string())]
    );

    assert!(!graph.remove_node("b"));
    assert!(!graph.remove_node("missing"));
}

#[test]
fn remove_node_removes_dangling_edges_too() {
    // An edge can only be added between existing nodes, so build the dangling state by
    // removing the node through the public API before re-inserting a partial node.
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("add edge");
    graph.remove_node("b");
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.remove_incident_edges("b"), 0);
}

// ---------------------------------------------------------------------------
// Edge validation
// ---------------------------------------------------------------------------

#[test]
fn add_edge_rejects_a_missing_source() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    let error = graph
        .add_edge(CapabilityEdge::can_feed("missing", "b"))
        .unwrap_err();
    assert_validation(&error, "edge.source");
    assert!(error.message().contains("missing"));
    assert_eq!(graph.edge_count(), 0);

    let error = graph
        .add_edge(CapabilityEdge::can_feed("   ", "b"))
        .unwrap_err();
    assert_validation(&error, "edge.source");
    assert!(error.message().contains("blank"));
}

#[test]
fn add_edge_rejects_a_missing_target() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    let error = graph
        .add_edge(CapabilityEdge::can_feed("a", "missing"))
        .unwrap_err();
    assert_validation(&error, "edge.target");
    assert!(error.message().contains("missing"));
    assert_eq!(graph.edge_count(), 0);

    let error = graph
        .add_edge(CapabilityEdge::can_feed("a", " "))
        .unwrap_err();
    assert_validation(&error, "edge.target");
    assert!(error.message().contains("blank"));
}

#[test]
fn add_edge_rejects_self_loops() {
    let mut graph = graph_with(&[("a", "A", "tool")]);
    let error = graph
        .add_edge(CapabilityEdge::can_feed("a", "a"))
        .unwrap_err();
    assert_validation(&error, "edge.target");
    assert!(error.message().contains("self-loop"));
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn add_edge_rejects_a_blank_relation() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    let error = graph
        .add_edge(CapabilityEdge::new("a", "b", "  ", 0.5))
        .unwrap_err();
    assert_validation(&error, "edge.relation");
    assert!(error.message().contains("blank"));
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn add_edge_rejects_non_finite_weights() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = graph
            .add_edge(CapabilityEdge::new("a", "b", CAN_FEED, weight))
            .unwrap_err();
        assert_validation(&error, "edge.weight");
        assert!(error.message().contains("finite"));
    }
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn add_edge_rejects_weights_outside_the_unit_interval() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    for weight in [-0.1, 1.0 + f64::EPSILON, 2.0] {
        let error = graph
            .add_edge(CapabilityEdge::new("a", "b", CAN_FEED, weight))
            .unwrap_err();
        assert_validation(&error, "edge.weight");
        assert!(error.message().contains("[0, 1]"));
    }
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn add_edge_accepts_the_weight_boundaries() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool")]);
    graph
        .add_edge(CapabilityEdge::new("a", "b", CAN_FEED, 0.0))
        .expect("weight 0.0");
    graph
        .add_edge(CapabilityEdge::new("a", "b", CAN_FEED, 1.0))
        .expect("weight 1.0");
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.edges[0].weight, 1.0);
}

#[test]
fn add_edge_upserts_in_place_preserving_insertion_order() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool"), ("c", "C", "tool")]);
    graph
        .add_edge(CapabilityEdge::new("a", "b", CAN_FEED, 0.4))
        .expect("a->b");
    graph
        .add_edge(CapabilityEdge::new("a", "c", CAN_FEED, 0.8))
        .expect("a->c");
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("a->b upsert");

    assert_eq!(graph.edge_count(), 2);
    assert_eq!(
        edge_keys(&graph),
        vec![
            ("a".to_string(), "b".to_string(), CAN_FEED.to_string()),
            ("a".to_string(), "c".to_string(), CAN_FEED.to_string()),
        ]
    );
    assert_eq!(graph.edges[0].weight, 1.0);

    // A different relation is a different identity: it is appended, not upserted.
    graph
        .add_edge(CapabilityEdge::new("a", "b", "depends_on", 0.5))
        .expect("a->b depends_on");
    assert_eq!(graph.edge_count(), 3);
    assert_eq!(graph.edges[2].relation, "depends_on");
}

// ---------------------------------------------------------------------------
// Edge removal
// ---------------------------------------------------------------------------

#[test]
fn remove_edge_drops_every_relation_between_a_pair() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool"), ("c", "C", "tool")]);
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("a->b");
    graph
        .add_edge(CapabilityEdge::new("a", "b", "depends_on", 0.5))
        .expect("a->b depends_on");
    graph
        .add_edge(CapabilityEdge::can_feed("a", "c"))
        .expect("a->c");

    assert!(graph.remove_edge("a", "b"));
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(
        edge_keys(&graph),
        vec![("a".to_string(), "c".to_string(), CAN_FEED.to_string())]
    );
    assert!(!graph.remove_edge("a", "b"));
    assert!(!graph.remove_edge("c", "a"));
}

#[test]
fn remove_incident_edges_counts_only_what_it_removed() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool"), ("c", "C", "tool")]);
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("a->b");
    graph
        .add_edge(CapabilityEdge::can_feed("b", "c"))
        .expect("b->c");
    graph
        .add_edge(CapabilityEdge::can_feed("a", "c"))
        .expect("a->c");

    assert_eq!(graph.remove_incident_edges("b"), 2);
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.remove_incident_edges("b"), 0);
    assert_eq!(graph.remove_incident_edges("a"), 1);
    assert_eq!(graph.edge_count(), 0);
}

// ---------------------------------------------------------------------------
// Deterministic reads
// ---------------------------------------------------------------------------

#[test]
fn outgoing_and_incoming_are_sorted_by_source_target_relation() {
    let mut graph = graph_with(&[("a", "A", "tool"), ("b", "B", "tool"), ("c", "C", "tool")]);
    graph
        .add_edge(CapabilityEdge::new("a", "c", "depends_on", 0.2))
        .expect("a->c");
    graph
        .add_edge(CapabilityEdge::new("a", "b", "depends_on", 0.3))
        .expect("a->b depends_on");
    graph
        .add_edge(CapabilityEdge::new("a", "b", CAN_FEED, 0.9))
        .expect("a->b can_feed");

    let keys = |edges: Vec<&CapabilityEdge>| {
        edges
            .into_iter()
            .map(|edge| {
                (
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.relation.clone(),
                )
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(
        keys(graph.outgoing("a")),
        vec![
            ("a".to_string(), "b".to_string(), CAN_FEED.to_string()),
            ("a".to_string(), "b".to_string(), "depends_on".to_string()),
            ("a".to_string(), "c".to_string(), "depends_on".to_string()),
        ]
    );
    assert_eq!(
        keys(graph.incoming("b")),
        vec![
            ("a".to_string(), "b".to_string(), CAN_FEED.to_string()),
            ("a".to_string(), "b".to_string(), "depends_on".to_string()),
        ]
    );
    assert!(graph.outgoing("b").is_empty());
    assert!(graph.incoming("a").is_empty());
}

#[test]
fn roots_and_leaves_are_sorted_node_ids() {
    let mut graph = graph_with(&[("c", "C", "tool"), ("a", "A", "tool"), ("b", "B", "tool")]);
    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("a->b");
    graph
        .add_edge(CapabilityEdge::can_feed("b", "c"))
        .expect("b->c");

    assert_eq!(graph.roots(), vec!["a".to_string()]);
    assert_eq!(graph.leaves(), vec!["c".to_string()]);

    // Isolated nodes are both roots and leaves; the result stays sorted.
    let mut isolated = graph.clone();
    isolated.upsert_node(CapabilityNode::new("0", "Zero", "tool"));
    assert_eq!(isolated.roots(), vec!["0".to_string(), "a".to_string()]);
    assert_eq!(isolated.leaves(), vec!["0".to_string(), "c".to_string()]);

    // Closing the chain into a cycle leaves no roots or leaves.
    graph
        .add_edge(CapabilityEdge::can_feed("c", "a"))
        .expect("c->a");
    assert!(graph.roots().is_empty());
    assert!(graph.leaves().is_empty());
}

#[test]
fn counts_track_nodes_and_edges() {
    let mut graph = CapabilityGraph::new("counts");
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);

    graph.upsert_node(CapabilityNode::new("a", "A", "tool"));
    graph.upsert_node(CapabilityNode::new("b", "B", "tool"));
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 0);

    graph
        .add_edge(CapabilityEdge::can_feed("a", "b"))
        .expect("add edge");
    assert_eq!(graph.edge_count(), 1);
}

// ---------------------------------------------------------------------------
// JGF documents
// ---------------------------------------------------------------------------

/// Builds a three-node graph with two `can_feed` edges and graph metadata.
fn sample_graph() -> CapabilityGraph {
    let capabilities = vec![
        descriptor("tool", "sum", "Sum Numbers"),
        descriptor("tool", "translate", "Translate Text"),
        descriptor("workflow", "report", "Build Report"),
    ];
    let mut graph = CapabilityGraph::from_descriptors("test-graph", &capabilities);
    graph.metadata.insert("scope".to_string(), json!("demo"));
    graph
        .add_edge(CapabilityEdge::can_feed("tool:sum", "workflow:report"))
        .expect("sum->report");
    graph
        .add_edge(CapabilityEdge::can_feed(
            "tool:translate",
            "workflow:report",
        ))
        .expect("translate->report");
    graph
}

#[test]
fn to_value_renders_a_sorted_jgf_v2_document() {
    let value = sample_graph().to_value();
    let member = value["graph"].as_object().expect("graph member");

    assert_eq!(member["id"], json!("test-graph"));
    assert_eq!(member["type"], json!("capability_graph"));
    assert_eq!(member["directed"], json!(true));
    assert_eq!(member["metadata"]["scope"], json!("demo"));

    let nodes = member["nodes"].as_array().expect("nodes array");
    assert_eq!(nodes.len(), 3);
    assert_eq!(
        nodes
            .iter()
            .map(|node| node["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["tool:sum", "tool:translate", "workflow:report"]
    );
    assert_eq!(nodes[0]["label"], json!("Sum Numbers"));
    assert_eq!(nodes[0]["node_type"], json!("tool"));
    assert_eq!(nodes[0]["metadata"]["capability_id"], json!("sum"));

    let edges = member["edges"].as_array().expect("edges array");
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[0]["source"], json!("tool:sum"));
    assert_eq!(edges[1]["source"], json!("tool:translate"));
    assert_eq!(edges[0]["target"], json!("workflow:report"));
    assert_eq!(edges[0]["relation"], json!(CAN_FEED));
    assert_eq!(edges[0]["weight"], json!(1.0));
}

#[test]
fn jgf_documents_round_trip_exactly() {
    let graph = sample_graph();
    let value = graph.to_value();

    let parsed = CapabilityGraph::from_value(&value).expect("parse JGF");
    assert_eq!(parsed, graph);
    assert_eq!(parsed.to_value(), value);
}

#[test]
fn an_empty_graph_round_trips() {
    let graph = CapabilityGraph::new("empty");
    let value = graph.to_value();
    assert_eq!(value["graph"]["nodes"], json!([]));
    assert_eq!(value["graph"]["edges"], json!([]));
    assert_eq!(CapabilityGraph::from_value(&value).expect("parse"), graph);
}

#[test]
fn from_value_rejects_structurally_malformed_documents() {
    let cases = [
        (json!({}), "graph"),
        (json!([]), "graph"),
        (json!({"graph": 7}), "graph"),
        (json!({"graph": {"nodes": 7}}), "graph.nodes"),
        (json!({"graph": {"nodes": [42]}}), "graph.nodes[0]"),
        (
            json!({"graph": {"nodes": [{"id": 7}]}}),
            "graph.nodes[0].id",
        ),
        (
            json!({"graph": {"nodes": [{"id": "a", "metadata": "x"}]}}),
            "graph.nodes[0].metadata",
        ),
        (json!({"graph": {"edges": "nope"}}), "graph.edges"),
        (
            json!({"graph": {"edges": [{"source": 7}]}}),
            "graph.edges[0].source",
        ),
        (
            json!({"graph": {"edges": [{"source": "a", "weight": "heavy"}]}}),
            "graph.edges[0].weight",
        ),
        (json!({"graph": {"directed": "yes"}}), "graph.directed"),
    ];
    for (document, path) in cases {
        let error = CapabilityGraph::from_value(&document).unwrap_err();
        assert_validation(&error, path);
    }
}

#[test]
fn from_value_is_default_tolerant_and_ignores_unknown_members() {
    let document = json!({
        "unknown_root": true,
        "graph": {
            "id": "g",
            "unknown_graph_member": 1,
            "nodes": [{"id": "a", "unknown_node_member": 1}],
            "edges": [{"source": "a", "target": "b", "unknown_edge_member": true}]
        }
    });

    let graph = CapabilityGraph::from_value(&document).expect("parse tolerant document");
    assert_eq!(graph.graph_id, "g");
    assert_eq!(graph.graph_type, "capability_graph");
    assert!(graph.directed);
    assert!(graph.metadata.is_empty());

    let node = graph.node("a").expect("node a");
    assert_eq!(node.label, "");
    assert_eq!(node.node_type, "");
    assert!(node.metadata.is_empty());

    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.edges[0].relation, "");
    assert_eq!(graph.edges[0].weight, 0.0);
    assert!(graph.edges[0].metadata.is_empty());
}

#[test]
fn from_value_preserves_the_edge_array_order() {
    let document = json!({
        "graph": {
            "id": "g",
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "c", "target": "a", "relation": CAN_FEED},
                {"source": "a", "target": "b", "relation": CAN_FEED}
            ]
        }
    });
    let graph = CapabilityGraph::from_value(&document).expect("parse");
    assert_eq!(
        edge_keys(&graph),
        vec![
            ("c".to_string(), "a".to_string(), CAN_FEED.to_string()),
            ("a".to_string(), "b".to_string(), CAN_FEED.to_string()),
        ]
    );
    // Reads re-sort by (source, target, relation) regardless of storage order.
    assert_eq!(graph.outgoing("c")[0].target, "a");
}

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

#[test]
fn nodes_and_edges_deserialize_tolerantly() {
    let node: CapabilityNode =
        serde_json::from_value(json!({"id": "a", "extra": 1})).expect("node");
    assert_eq!(node.id, "a");
    assert_eq!(node.label, "");
    assert!(node.metadata.is_empty());

    let edge: CapabilityEdge = serde_json::from_value(json!({
        "source": "a",
        "target": "b",
        "relation": CAN_FEED,
        "weight": 0.25,
        "extra": true
    }))
    .expect("edge");
    assert_eq!(edge.weight, 0.25);
    assert_eq!(edge.relation, CAN_FEED);

    let graph: CapabilityGraph = serde_json::from_value(json!({
        "graph_id": "g",
        "nodes": {"a": {"id": "a", "label": "A", "node_type": "tool"}}
    }))
    .expect("graph");
    assert_eq!(graph.graph_type, "capability_graph");
    assert!(graph.directed);
    assert_eq!(graph.node("a").expect("a").label, "A");
}

#[test]
fn node_and_edge_constructors_populate_metadata() {
    let node = CapabilityNode::new("a", "A", "tool").with_metadata("k", json!(1));
    assert_eq!(node.metadata["k"], json!(1));

    let edge = CapabilityEdge::can_feed("a", "b").with_metadata("strategy", json!("exact_io"));
    assert_eq!(edge.source, "a");
    assert_eq!(edge.target, "b");
    assert_eq!(edge.relation, CAN_FEED);
    assert_eq!(edge.weight, 1.0);
    assert_eq!(edge.metadata["strategy"], json!("exact_io"));

    let weighted = CapabilityEdge::new("a", "b", "depends_on", 0.4);
    assert_eq!(weighted.relation, "depends_on");
    assert_eq!(weighted.weight, 0.4);
    assert!(weighted.metadata.is_empty());
}
