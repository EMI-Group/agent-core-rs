use super::super::*;
use super::support::*;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

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
