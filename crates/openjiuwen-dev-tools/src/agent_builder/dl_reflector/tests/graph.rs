use super::super::*;
use super::support::*;
use crate::agent_builder::cycle_checker::has_cycle;

#[test]
fn valid_document_passes_unchanged() {
    let dsl = valid_document();
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert!(reflection.passed);
    assert_eq!(reflection.issues.len(), 0);
    assert_eq!(reflection.repaired, dsl);
}

#[test]
fn dangling_edge_is_dropped() {
    let dsl = graph(
        vec![node("start", "start"), node("end", "end")],
        vec![edge("start", "end"), edge("start", "ghost")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::DanglingEdge]);
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("ghost"));
    assert!(reflection.issues[0].message.contains("`start` -> `ghost`"));
    assert_eq!(reflection.repaired.edges, vec![edge("start", "end")]);
    assert_eq!(dsl.edges.len(), 2, "the input document is untouched");
    assert_valid(&reflection);
}

#[test]
fn duplicate_node_id_is_renamed() {
    let dsl = graph(
        vec![
            node("start", "start"),
            node("dup", "llm"),
            node("dup", "llm"),
            node("end", "end"),
        ],
        vec![edge("start", "dup"), edge("dup", "end")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(
        kinds(&reflection),
        vec![
            ReflectionIssueKind::DuplicateNodeId,
            ReflectionIssueKind::UnreachableNode
        ]
    );
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("dup"));
    assert_eq!(reflection.repaired.nodes[1].id, "dup");
    assert_eq!(reflection.repaired.nodes[2].id, "dup_2");
    assert!(
        reflection.repaired.edges.contains(&edge("dup", "dup_2")),
        "the renamed copy is connected: {:?}",
        reflection.repaired.edges
    );
    assert_valid(&reflection);
}

#[test]
fn missing_start_is_inserted_at_the_front() {
    let dsl = graph(
        vec![node("llm", "llm"), node("end", "end")],
        vec![edge("llm", "end")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::MissingStart]);
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("start"));
    let inserted = &reflection.repaired.nodes[0];
    assert_eq!(inserted.id, "start");
    assert_eq!(inserted.node_type, "start");
    assert_eq!(inserted.title, "Start");
    assert!(inserted.inputs.contains_key("query"));
    assert!(reflection.repaired.edges.contains(&edge("start", "llm")));
    assert_valid(&reflection);
}

#[test]
fn missing_end_is_appended_and_sinks_connected() {
    let dsl = graph(
        vec![node("start", "start"), node("llm", "llm")],
        vec![edge("start", "llm")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::MissingEnd]);
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("end"));
    let appended = reflection.repaired.nodes.last().expect("a last node");
    assert_eq!(appended.id, "end");
    assert_eq!(appended.node_type, "end");
    assert!(reflection.repaired.edges.contains(&edge("llm", "end")));
    assert_valid(&reflection);
}

#[test]
fn cycle_back_edge_is_removed() {
    let dsl = graph(
        vec![node("a", "start"), node("b", "llm"), node("c", "end")],
        vec![edge("a", "b"), edge("b", "c"), edge("c", "a")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::Cycle]);
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("a"));
    assert!(
        reflection.issues[0].message.contains("a -> b -> c -> a"),
        "the message names the path: {}",
        reflection.issues[0].message
    );
    assert!(!has_cycle(&reflection.repaired));
    assert!(!reflection.repaired.edges.contains(&edge("c", "a")));
    assert_valid(&reflection);
}

#[test]
fn unreachable_node_is_connected() {
    let dsl = graph(
        vec![
            node("start", "start"),
            node("a", "llm"),
            node("end", "end"),
            node("b", "llm"),
        ],
        vec![edge("start", "a"), edge("a", "end")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(
        kinds(&reflection),
        vec![ReflectionIssueKind::UnreachableNode]
    );
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("b"));
    assert!(reflection.repaired.edges.contains(&edge("end", "b")));
    assert_valid(&reflection);
}

#[test]
fn blank_id_and_title_are_repaired() {
    let dsl = graph(
        vec![
            node("start", "start"),
            DslNode {
                id: "   ".to_string(),
                node_type: "llm".to_string(),
                ..DslNode::default()
            },
            node("end", "end"),
        ],
        vec![edge("start", "end")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(
        kinds(&reflection),
        vec![
            ReflectionIssueKind::InvalidNode,
            ReflectionIssueKind::UnreachableNode
        ]
    );
    assert_eq!(reflection.repaired.nodes[1].id, "node_2");
    assert_eq!(reflection.repaired.nodes[1].title, "node_2");
    assert_eq!(reflection.issues[0].node_id.as_deref(), Some("node_2"));
    assert!(reflection.issues[0].message.contains("id and title"));
    assert_valid(&reflection);

    let dsl = graph(
        vec![
            node("start", "start"),
            DslNode {
                id: "llm".to_string(),
                node_type: "llm".to_string(),
                ..DslNode::default()
            },
            node("end", "end"),
        ],
        vec![edge("start", "llm"), edge("llm", "end")],
    );
    let reflection = DlReflector::new().reflect(&dsl).expect("reflect");

    assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::InvalidNode]);
    assert_eq!(reflection.repaired.nodes[1].title, "llm");
    assert!(reflection.issues[0].message.contains("title"));
    assert_valid(&reflection);
}

#[test]
fn empty_document_gets_a_start_end_chain() {
    let reflection = DlReflector::new()
        .reflect(&WorkflowDsl::default())
        .expect("reflect");

    assert!(!reflection.passed);
    assert_eq!(
        kinds(&reflection),
        vec![
            ReflectionIssueKind::MissingStart,
            ReflectionIssueKind::MissingEnd
        ]
    );
    assert_eq!(reflection.repaired.nodes[0].id, "start");
    assert_eq!(reflection.repaired.nodes[1].id, "end");
    assert_eq!(reflection.repaired.edges, vec![edge("start", "end")]);
    assert_valid(&reflection);
}

#[test]
fn reflection_is_deterministic() {
    let dsl = graph(
        vec![
            node("start", "start"),
            node("dup", "llm"),
            node("dup", "llm"),
            node("loop", "llm"),
            node("end", "end"),
        ],
        vec![
            edge("start", "dup"),
            edge("dup", "end"),
            edge("loop", "start"),
            edge("start", "ghost"),
        ],
    );
    let reflector = DlReflector::new();
    let first = reflector.reflect(&dsl).expect("reflect");
    let second = reflector.reflect(&dsl).expect("reflect");

    assert_eq!(first, second);
    assert!(!first.passed);
    assert!(!has_cycle(&first.repaired));
    assert_valid(&first);
}
