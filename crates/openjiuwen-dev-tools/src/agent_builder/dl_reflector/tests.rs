//! Tests for the rule-based DL reflector and its optional model revision.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence.
use super::*;
use crate::agent_builder::cycle_checker::has_cycle;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

/// A node titled after its id.
fn node(id: &str, node_type: &str) -> DslNode {
    DslNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        title: id.to_string(),
        ..DslNode::default()
    }
}

/// A document with the given nodes and edges.
fn graph(nodes: Vec<DslNode>, edges: Vec<DslEdge>) -> WorkflowDsl {
    WorkflowDsl {
        name: "demo".to_string(),
        description: "a demo workflow".to_string(),
        nodes,
        edges,
    }
}

/// A valid `start` -> `end` document.
fn valid_document() -> WorkflowDsl {
    graph(
        vec![node("start", "start"), node("end", "end")],
        vec![edge("start", "end")],
    )
}

/// Coerces a scripted model into the trait object the reflector takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The issue kinds, in recorded order.
fn kinds(reflection: &Reflection) -> Vec<ReflectionIssueKind> {
    reflection.issues.iter().map(|issue| issue.kind).collect()
}

/// Asserts the repaired document validates.
fn assert_valid(reflection: &Reflection) {
    reflection.repaired.validate().expect("repaired validates");
}

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

#[test]
fn reflection_types_round_trip_as_json() {
    assert_eq!(
        ReflectionIssueKind::default(),
        ReflectionIssueKind::DanglingEdge
    );
    let issue = ReflectionIssue {
        kind: ReflectionIssueKind::Cycle,
        message: "cycle".to_string(),
        node_id: Some("a".to_string()),
    };
    let value = serde_json::to_value(&issue).expect("serialize");
    assert_eq!(value["kind"], json!("cycle"));
    assert_eq!(value["node_id"], json!("a"));

    let reflection = Reflection {
        passed: false,
        issues: vec![issue],
        repaired: valid_document(),
    };
    let text = serde_json::to_string(&reflection).expect("serialize");
    let parsed: Reflection = serde_json::from_str(&text).expect("parse");
    assert_eq!(parsed, reflection);
    assert_eq!(
        serde_json::from_str::<Reflection>("{}").expect("parse"),
        Reflection::default()
    );
    assert_eq!(
        serde_json::to_value(ReflectionIssueKind::UnreachableNode).expect("serialize"),
        json!("unreachable_node")
    );
}

#[test]
fn debug_reports_whether_a_model_is_configured() {
    let model = Arc::new(ScriptedModel::text("unused"));
    let reflectors = [
        (DlReflector::new(), "DlReflector { has_model: false }"),
        (DlReflector::default(), "DlReflector { has_model: false }"),
        (
            DlReflector::with_model(dynamic(&model)),
            "DlReflector { has_model: true }",
        ),
    ];
    for (reflector, expected) in reflectors {
        assert_eq!(format!("{reflector:?}"), expected);
    }
}

#[tokio::test]
async fn model_revision_is_accepted() {
    let answer = r#"```json
{"name": "demo", "description": "improved workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Begin"},
           {"id": "end", "type": "end", "title": "Finish"}],
 "edges": [{"source": "start", "target": "end"}]}
```"#;
    let model = Arc::new(ScriptedModel::text(answer));
    let reflector = DlReflector::with_model(dynamic(&model));
    let dsl = valid_document();

    let reflection = reflector
        .reflect_with_model(&dsl, "improve the titles")
        .await
        .expect("no error");

    assert!(reflection.passed);
    assert_eq!(reflection.issues.len(), 0);
    assert_eq!(reflection.repaired.description, "improved workflow");
    assert_eq!(reflection.repaired.nodes[0].title, "Begin");
    assert_eq!(reflection.repaired.nodes[1].title, "Finish");

    assert_eq!(model.call_count(), 1);
    let call = model.last_call().expect("one call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), REFLECTOR_SYSTEM_PROMPT);
    let user = call.messages[1].text().into_owned();
    assert!(user.contains("improve the titles"));
    assert!(user.contains(&dsl.to_json_string().expect("serialize")));
    assert_eq!(call.request.model_name, "scripted-model");
}

#[tokio::test]
async fn model_problems_are_rejected() {
    let answers = [
        (ScriptedModel::text("Looks good to me."), "code fence"),
        (ScriptedModel::new([ScriptedTurn::fail("boom")]), "boom"),
    ];
    for (scripted, expected) in answers {
        let model = Arc::new(scripted);
        let reflector = DlReflector::with_model(dynamic(&model));
        let dsl = valid_document();

        let reflection = reflector
            .reflect_with_model(&dsl, "improve")
            .await
            .expect("no error");

        assert_eq!(reflection.repaired, dsl);
        assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::InvalidNode]);
        assert_eq!(reflection.issues[0].node_id, None);
        assert!(reflection.issues[0]
            .message
            .starts_with("model revision rejected: "));
        assert!(reflection.issues[0].message.contains(expected));
        assert!(!reflection.passed);
        assert_eq!(model.call_count(), 1);
    }
}

#[tokio::test]
async fn model_revisions_outside_the_contract_are_rejected() {
    let answers = [
        (
            r#"```json
{"nodes": [{"id": "start", "type": "start"}, {"id": "finish", "type": "end"}],
 "edges": [{"source": "start", "target": "finish"}]}
```"#,
            "node ids",
        ),
        (
            r#"```json
{"nodes": [{"id": "start", "type": "start"}, {"id": "end", "type": "end"}],
 "edges": [{"source": "start", "target": "ghost"}]}
```"#,
            "invalid",
        ),
    ];
    for (answer, expected) in answers {
        let model = Arc::new(ScriptedModel::text(answer));
        let reflector = DlReflector::with_model(dynamic(&model));
        let dsl = valid_document();

        let reflection = reflector
            .reflect_with_model(&dsl, "improve")
            .await
            .expect("no error");

        assert_eq!(reflection.repaired, dsl);
        assert_eq!(kinds(&reflection), vec![ReflectionIssueKind::InvalidNode]);
        assert!(reflection.issues[0].message.contains(expected));
        assert!(!reflection.passed);
    }
}

#[tokio::test]
async fn rejection_is_appended_to_the_baseline_issues() {
    let model = Arc::new(ScriptedModel::text("no document here"));
    let reflector = DlReflector::with_model(dynamic(&model));
    let dsl = graph(
        vec![node("start", "start"), node("llm", "llm")],
        vec![edge("start", "llm")],
    );

    let baseline = reflector.reflect(&dsl).expect("reflect");
    let reflection = reflector
        .reflect_with_model(&dsl, "improve")
        .await
        .expect("no error");

    assert_eq!(reflection.issues.len(), baseline.issues.len() + 1);
    assert_eq!(
        reflection.issues[..baseline.issues.len()],
        baseline.issues[..]
    );
    assert_eq!(reflection.repaired, baseline.repaired);
    assert!(!reflection.passed);
}

#[tokio::test]
async fn reflector_without_a_model_returns_the_baseline() {
    let dsl = graph(
        vec![node("llm", "llm"), node("end", "end")],
        vec![edge("llm", "end")],
    );
    let reflector = DlReflector::new();

    let baseline = reflector.reflect(&dsl).expect("reflect");
    let reflection = reflector
        .reflect_with_model(&dsl, "ignored without a model")
        .await
        .expect("no error");

    assert_eq!(reflection, baseline);
}
