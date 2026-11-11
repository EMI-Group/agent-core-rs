//! `GraphSpec` serialization behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

use std::collections::BTreeMap;

use serde_json::json;

use openjiuwen_graph::{EdgeKind, GraphSpec, NodeOptions, NodeSpec};

#[test]
fn spec_round_trips_through_json() {
    let value = json!({
        "graph_id": "g",
        "entry": ["a"],
        "exit": ["b"],
        "nodes": [
            {"id": "a"},
            {"id": "b", "kind": "llm", "options": {"timeout_ms": 25, "max_retries": 2}}
        ],
        "edges": [{"sources": ["a", "c"], "target": "b", "kind": "barrier"}],
    });
    let spec: GraphSpec = serde_json::from_value(value).unwrap();
    assert_eq!(spec.graph_id, "g");
    assert_eq!(spec.entry, vec!["a".to_string()]);
    assert_eq!(spec.nodes[0].kind, "node");
    assert_eq!(spec.nodes[0].options, NodeOptions::default());
    assert_eq!(
        spec.nodes[1].options,
        NodeOptions {
            timeout_ms: Some(25),
            max_retries: 2
        }
    );
    assert_eq!(spec.edges[0].kind, EdgeKind::Barrier);
    assert!(spec.metadata.is_empty());

    let encoded = serde_json::to_value(&spec).unwrap();
    let decoded: GraphSpec = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, spec);
}

#[test]
fn unknown_fields_are_tolerated() {
    let spec: GraphSpec = serde_json::from_value(json!({
        "graph_id": "g",
        "nodes": [{"id": "a", "future_field": true}],
        "unknown": {"nested": 1},
    }))
    .unwrap();
    assert_eq!(
        spec.nodes,
        vec![NodeSpec {
            id: "a".to_string(),
            kind: "node".to_string(),
            options: NodeOptions::default(),
        }]
    );
}

#[test]
fn collections_default_to_empty() {
    let spec: GraphSpec = serde_json::from_value(json!({"graph_id": "g"})).unwrap();
    assert!(spec.entry.is_empty());
    assert!(spec.exit.is_empty());
    assert!(spec.nodes.is_empty());
    assert!(spec.edges.is_empty());
    assert!(spec.metadata.is_empty());
}

#[test]
fn spec_encode_uses_snake_case() {
    let spec = GraphSpec {
        graph_id: "g".to_string(),
        entry: vec!["a".to_string()],
        exit: Vec::new(),
        nodes: vec![NodeSpec {
            id: "a".to_string(),
            kind: "node".to_string(),
            options: NodeOptions::default(),
        }],
        edges: Vec::new(),
        metadata: BTreeMap::from([("owner".to_string(), json!("tests"))]),
    };
    let value = serde_json::to_value(&spec).unwrap();
    assert_eq!(value["graph_id"], json!("g"));
    assert_eq!(value["nodes"][0]["options"]["max_retries"], json!(0));
    assert_eq!(value["metadata"]["owner"], json!("tests"));
}

#[test]
fn node_outcome_uses_the_documented_shapes() {
    use openjiuwen_graph::NodeOutcome;

    assert_eq!(
        serde_json::to_value(NodeOutcome::Continue).unwrap(),
        json!("continue")
    );
    assert_eq!(
        serde_json::to_value(NodeOutcome::route(["a", "b"])).unwrap(),
        json!({"route": {"targets": ["a", "b"]}})
    );
    assert_eq!(
        serde_json::to_value(NodeOutcome::halt(Some(json!(1)))).unwrap(),
        json!({"halt": {"value": 1}})
    );
    assert_eq!(
        serde_json::from_value::<NodeOutcome>(json!({"halt": {}})).unwrap(),
        NodeOutcome::Halt { value: None }
    );
}
