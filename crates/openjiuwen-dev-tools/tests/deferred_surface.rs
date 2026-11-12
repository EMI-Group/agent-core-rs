#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Crate-level honesty tests for the deferred surface and workflow DSL validation.
//!
//! Two invariants are checked here instead of in the module unit tests:
//!
//! - Every deferred `tune` entry point reports the *same* honest error shape:
//!   `ErrorKind::Unsupported`, a message naming the deferred pipeline, and a
//!   pointer at `docs/PARITY.md`. The stubs must never pretend to succeed.
//! - `WorkflowDsl::validate` rejects the structural errors the builder relies on
//!   (no nodes, duplicate ids, unresolvable edge endpoints) and names the
//!   offending id, while a well-formed document passes and round-trips.
//!
//! Everything here is offline and deterministic.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_dev_tools::agent_builder::{DslEdge, DslNode, WorkflowDsl};
use openjiuwen_dev_tools::tune::{
    optimize_prompt, Case, ExampleOptimizer, InstructionOptimizer, JointOptimizer,
    OptimizationResult, PromptSearchOptimizer, Trainer,
};
use serde_json::{json, Value};

/// A minimal valid case, enough to satisfy the stub signatures.
fn case(id: &str) -> Case {
    let mut inputs = BTreeMap::new();
    inputs.insert("question".to_owned(), Value::String(id.to_owned()));
    let mut label = BTreeMap::new();
    label.insert("answer".to_owned(), Value::String(id.to_owned()));
    Case::new(inputs, label)
        .expect("a non-empty inputs and label map make a valid case")
        .with_case_id(id)
}

/// A node whose id is the only field the structural validation reads.
fn node(id: &str) -> DslNode {
    DslNode {
        id: id.to_owned(),
        node_type: "llm".to_owned(),
        title: id.to_owned(),
        ..DslNode::default()
    }
}

/// Asserts that `error` is the honest deferred-pipeline error of `entry_point`.
fn assert_deferred(entry_point: &str, pipeline: &str, error: &Error) {
    assert_eq!(
        error.kind(),
        ErrorKind::Unsupported,
        "{entry_point} must report the deferred pipeline as Unsupported, got {:?}: {error}",
        error.kind()
    );
    let message = error.to_string();
    assert!(
        message.contains("not ported"),
        "{entry_point} must state that the pipeline is not ported: {message}"
    );
    assert!(
        message.contains(pipeline),
        "{entry_point} must name the deferred pipeline {pipeline:?}: {message}"
    );
    assert!(
        message.contains("docs/PARITY.md"),
        "{entry_point} must point at the parity document: {message}"
    );
}

#[tokio::test]
async fn deferred_tuning_entry_points_report_unsupported() {
    let prompt = "Answer the question.";
    let cases = vec![case("case-1")];

    // Every deferred entry point, labelled so a failure names the caller that
    // broke the contract. `optimize_prompt` delegates to the instruction
    // optimizer, so it reports that pipeline by name.
    let results: Vec<(&str, &str, Result<OptimizationResult>)> = vec![
        (
            "InstructionOptimizer::optimize",
            "InstructionOptimizer",
            InstructionOptimizer::new().optimize(prompt, &cases).await,
        ),
        (
            "ExampleOptimizer::optimize",
            "ExampleOptimizer",
            ExampleOptimizer::new().optimize(prompt, &cases, 2).await,
        ),
        (
            "JointOptimizer::optimize",
            "JointOptimizer",
            JointOptimizer::new().optimize(prompt, &cases).await,
        ),
        (
            "PromptSearchOptimizer::optimize",
            "PromptSearchOptimizer",
            PromptSearchOptimizer::new().optimize(prompt, &cases).await,
        ),
        (
            "Trainer::train",
            "Trainer",
            Trainer::new().train(prompt, &cases, &cases),
        ),
        (
            "optimize_prompt",
            "InstructionOptimizer",
            optimize_prompt(prompt, &cases).await,
        ),
    ];

    assert_eq!(
        results.len(),
        6,
        "the table must cover every deferred tuning entry point"
    );
    for (entry_point, pipeline, result) in results {
        let error = result.expect_err(&format!(
            "{entry_point} must fail as unsupported instead of returning a result"
        ));
        assert_deferred(entry_point, pipeline, &error);
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
