#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Crate-level honesty tests for the remaining typed failure surfaces.
//!
//! Only capabilities that stay genuinely unported may report
//! `ErrorKind::Unsupported`; everything else in this crate performs its work.
//! This binary pins the two invariants that outlive individual modules:
//!
//! - the workflow DSL rejects the structural errors the builder and the
//!   declarative generator rely on (no nodes, duplicate ids, unresolvable edge
//!   endpoints) and names the offending id, while a well-formed document passes
//!   and round-trips;
//! Module-local suites own the checks for capabilities that stay deliberately
//! deferred (for example the archive formats `skill_creator::omni` cannot unpack
//! without a compression dependency), so this binary stays focused on the DSL.
//!
//! Everything here is offline and deterministic.
use std::collections::BTreeMap;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::agent_builder::{DslEdge, DslNode, WorkflowDsl};
use serde_json::json;

/// A node whose id is the only field the structural validation reads.
fn node(id: &str) -> DslNode {
    DslNode {
        id: id.to_owned(),
        node_type: "llm".to_owned(),
        title: id.to_owned(),
        ..DslNode::default()
    }
}

/// Asserts that `dsl` fails structural validation with `expected` in its
/// message, as a [`ErrorKind::Validation`] error.
fn assert_rejected(dsl: &WorkflowDsl, label: &str, expected: &str) {
    let error = dsl
        .validate()
        .expect_err(&format!("{label} must fail validation"));
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "{label} is a Validation error, got {:?}: {error}",
        error.kind()
    );
    let message = error.to_string();
    assert!(
        message.contains(expected),
        "{label}: the error message must contain {expected:?}: {message}"
    );
}

#[test]
fn workflow_dsl_validate_rejects_structural_errors() {
    // (a) A document without nodes cannot be executed.
    assert_rejected(
        &WorkflowDsl::default(),
        "a document without nodes",
        "no nodes",
    );

    // (b) Node ids address the graph, so they must be unique.
    let duplicate = WorkflowDsl {
        nodes: vec![node("start"), node("start")],
        ..WorkflowDsl::default()
    };
    assert_rejected(&duplicate, "a repeated node id", "duplicate node id: start");

    // (c) Every edge endpoint must resolve to a node, and the error names the
    // dangling id — for the source and for the target alike.
    for (label, (source, target), dangling) in [
        (
            "a dangling edge source",
            ("ghost-source", "start"),
            "ghost-source",
        ),
        (
            "a dangling edge target",
            ("start", "ghost-target"),
            "ghost-target",
        ),
    ] {
        let dsl = WorkflowDsl {
            nodes: vec![node("start")],
            edges: vec![DslEdge {
                source: source.to_owned(),
                target: target.to_owned(),
                condition: None,
            }],
            ..WorkflowDsl::default()
        };
        assert_rejected(
            &dsl,
            label,
            &format!("edge references unknown node: {dangling}"),
        );
    }

    // A well-formed two-node/one-edge document validates and round-trips.
    let valid = WorkflowDsl {
        name: "greeter".to_owned(),
        description: "start then end".to_owned(),
        nodes: vec![
            DslNode {
                id: "start".to_owned(),
                node_type: "start".to_owned(),
                title: "Start".to_owned(),
                inputs: BTreeMap::from([("query".to_owned(), json!({"type": "string"}))]),
                outputs: BTreeMap::from([("text".to_owned(), json!({"type": "string"}))]),
                ..DslNode::default()
            },
            DslNode {
                id: "end".to_owned(),
                node_type: "end".to_owned(),
                title: "End".to_owned(),
                ..DslNode::default()
            },
        ],
        edges: vec![DslEdge {
            source: "start".to_owned(),
            target: "end".to_owned(),
            condition: Some("score > 0.5".to_owned()),
        }],
    };
    valid
        .validate()
        .expect("two unique nodes joined by one resolvable edge form a valid document");
    assert_eq!(
        valid.node("end").map(|found| found.title.as_str()),
        Some("End"),
        "node lookup must resolve the ids the edges reference"
    );

    let text = valid
        .to_json_string()
        .expect("a workflow DSL document serializes to JSON");
    let reparsed = WorkflowDsl::from_json_str(&text)
        .expect("the serialized workflow DSL document parses back");
    assert_eq!(
        reparsed, valid,
        "a valid workflow DSL document must round-trip through to_json_string/from_json_str"
    );
    reparsed
        .validate()
        .expect("the round-tripped document must still validate");
}
