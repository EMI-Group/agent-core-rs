//! Tests for the rule-based DL reflector and its optional model revision.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence. The per-node-type validation cases build
//! documents from the Python-shaped DL model.
use super::*;
use crate::agent_builder::cycle_checker::has_cycle;
use crate::agent_builder::dl_transformer::model::{DlCondition, DlDocument, DlNode, DlVariable};
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

// ---------------------------------------------------------------------------
// Per-node-type DL validation (`DlReflector::validate_dl`)
// ---------------------------------------------------------------------------

/// A DL variable with an optional description.
fn dl_variable(name: &str, description: Option<&str>) -> DlVariable {
    DlVariable {
        name: name.to_string(),
        description: description.map(str::to_string),
        ..DlVariable::default()
    }
}

/// The Start node's required `query` output.
fn query_output() -> DlVariable {
    dl_variable("query", Some("用户输入"))
}

/// A DL node with a non-blank description.
fn dl_node(id: &str, node_type: &str) -> DlNode {
    DlNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        description: format!("{node_type} node"),
        ..DlNode::default()
    }
}

/// A Start node linking to `next` and declaring the `query` output.
fn dl_start(next: &str) -> DlNode {
    let mut node = dl_node("start", "Start");
    node.next = Some(next.to_string());
    node.parameters.outputs = vec![query_output()];
    node
}

/// An LLM node with variables, both prompts, and a `next` link.
fn dl_llm(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "LLM");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("result", Some("the answer"))];
    node.parameters.configs = BTreeMap::from([
        ("system_prompt".to_string(), json!("system")),
        ("user_prompt".to_string(), json!("user")),
    ]);
    node
}

/// A Questioner node with variables, `prompt`, and a `next` link.
fn dl_questioner(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Questioner");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("answer", Some("the answer"))];
    node.parameters.configs = BTreeMap::from([("prompt".to_string(), json!("Ask the user"))]);
    node
}

/// A Plugin node with variables, `tool_id`, and a `next` link.
fn dl_plugin(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Plugin");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("result", Some("the result"))];
    node.parameters.configs = BTreeMap::from([("tool_id".to_string(), json!("tool-1"))]);
    node
}

/// A Code node with variables, `code`, and a `next` link.
fn dl_code(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Code");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("result", Some("the result"))];
    node.parameters.configs = BTreeMap::from([("code".to_string(), json!("return 1"))]);
    node
}

/// An Output node with inputs, `template`, and a `next` link.
fn dl_output(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Output");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("result", None)];
    node.parameters.configs =
        BTreeMap::from([("template".to_string(), json!("answer: {{result}}"))]);
    node
}

/// An End node with inputs and `template`.
fn dl_end() -> DlNode {
    let mut node = dl_node("end", "End");
    node.parameters.inputs = vec![dl_variable("result", None)];
    node.parameters.configs = BTreeMap::from([("template".to_string(), json!("{{result}}"))]);
    node
}

/// One branch condition.
fn dl_condition(branch: &str, description: &str, next: &str, expression: &str) -> DlCondition {
    DlCondition {
        branch: branch.to_string(),
        description: description.to_string(),
        next: Some(next.to_string()),
        expression: Some(expression.to_string()),
        ..DlCondition::default()
    }
}

/// A Branch node: one matching condition over the LLM output plus a default.
fn dl_branch() -> DlNode {
    let mut node = dl_node("branch", "Branch");
    node.parameters.conditions = vec![
        dl_condition(
            "match",
            "the model matched",
            "end",
            "${llm.result} contain ${llm.result}",
        ),
        dl_condition("fallback", "anything else", "end", "default"),
    ];
    node
}

/// An `IntentDetection` node with a non-default and a default intent.
fn dl_intent(id: &str) -> DlNode {
    let mut node = dl_node(id, "IntentDetection");
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.configs = BTreeMap::from([("prompt".to_string(), json!("Detect the intent"))]);
    node.parameters.conditions = vec![
        dl_condition(
            "greet",
            "a greeting",
            "end",
            &format!("${{{id}.rawOutput}} contain hello"),
        ),
        dl_condition("fallback", "anything else", "end", "default"),
    ];
    node
}

/// The complete valid document used by the negative fixtures.
fn valid_dl() -> DlDocument {
    DlDocument {
        nodes: vec![
            dl_start("llm"),
            dl_llm("llm", "branch"),
            dl_branch(),
            dl_end(),
        ],
    }
}

/// A minimal `Start -> LLM -> End` document.
fn linear_dl() -> DlDocument {
    DlDocument {
        nodes: vec![dl_start("llm"), dl_llm("llm", "end"), dl_end()],
    }
}

/// The document `Start -> node -> End`, where `node` has the given id.
fn single_step_dl(node: DlNode) -> DlDocument {
    DlDocument {
        nodes: vec![dl_start(&node.id), node, dl_end()],
    }
}

/// The issue kinds of a validation, in recorded order.
fn dl_kinds(validation: &DlValidation) -> Vec<ReflectionIssueKind> {
    validation.issues.iter().map(|issue| issue.kind).collect()
}

/// Asserts the validation failed with exactly one issue of `kind`.
fn assert_issue(validation: &DlValidation, kind: ReflectionIssueKind, node_id: &str, needle: &str) {
    assert!(!validation.passed, "the document should not pass");
    assert_eq!(dl_kinds(validation), vec![kind]);
    let issue = &validation.issues[0];
    let message = &issue.message;
    assert_eq!(issue.node_id.as_deref(), Some(node_id));
    assert!(
        message.contains(needle),
        "message {message:?} lacks {needle:?}"
    );
}

/// Asserts the validation passed with no issue.
fn assert_passes(validation: &DlValidation) {
    assert_eq!(
        validation,
        &DlValidation {
            passed: true,
            issues: Vec::new(),
        }
    );
}

#[test]
fn valid_dl_document_passes_validation() {
    let validation = DlReflector::new().validate_dl(&valid_dl());

    assert_passes(&validation);
    assert_eq!(validation.issues, Vec::new());
}

#[test]
fn dl_validation_is_deterministic_and_never_calls_the_model() {
    let mut document = valid_dl();
    document.nodes[2].parameters.conditions[0].expression =
        Some("${ghost.result} frobnicate x".to_string());

    let model = Arc::new(ScriptedModel::text("unused"));
    let reflector = DlReflector::with_model(dynamic(&model));
    let first = reflector.validate_dl(&document);
    let second = reflector.validate_dl(&document);

    assert_eq!(first, second);
    assert_eq!(
        dl_kinds(&first),
        vec![
            ReflectionIssueKind::InvalidExpression,
            ReflectionIssueKind::InvalidReference
        ]
    );
    assert_eq!(model.call_count(), 0, "validation never calls the model");
    assert_eq!(document.nodes.len(), 4, "the document is untouched");
}

#[test]
fn dl_issues_are_reported_in_node_order() {
    let mut document = linear_dl();
    document.nodes[1].parameters.configs.remove("system_prompt");
    document.nodes[2].parameters.configs.remove("template");

    let validation = DlReflector::new().validate_dl(&document);

    assert_eq!(
        dl_kinds(&validation),
        vec![
            ReflectionIssueKind::MissingField,
            ReflectionIssueKind::MissingField
        ]
    );
    assert_eq!(validation.issues[0].node_id.as_deref(), Some("llm"));
    assert!(validation.issues[0].message.contains("system_prompt"));
    assert_eq!(validation.issues[1].node_id.as_deref(), Some("end"));
    assert!(validation.issues[1].message.contains("template"));
}

#[test]
fn dl_blank_basic_fields_are_reported() {
    let mut blank_id = dl_end();
    blank_id.id = String::new();
    let mut blank_type = dl_end();
    blank_type.node_type = String::new();
    let mut blank_description = dl_end();
    blank_description.description = "  ".to_string();

    let fixtures = [
        (blank_id, "", "id"),
        (blank_type, "end", "type"),
        (blank_description, "end", "description"),
    ];
    for (node, node_id, field) in fixtures {
        let document = DlDocument { nodes: vec![node] };
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            node_id,
            field,
        );
        assert!(validation.issues[0].message.contains("is blank"));
    }
}

#[test]
fn dl_duplicate_node_id_is_reported() {
    let document = DlDocument {
        nodes: vec![dl_end(), dl_end()],
    };

    let validation = DlReflector::new().validate_dl(&document);

    assert_issue(
        &validation,
        ReflectionIssueKind::DuplicateNodeId,
        "end",
        "duplicate node id",
    );
}

#[test]
fn dl_unknown_node_type_is_reported_and_skips_type_checks() {
    let mut node = dl_end();
    node.node_type = "Agent".to_string();
    node.parameters.inputs.clear();
    node.parameters.configs.clear();
    let document = DlDocument { nodes: vec![node] };

    let validation = DlReflector::new().validate_dl(&document);

    assert_issue(
        &validation,
        ReflectionIssueKind::UnknownNodeType,
        "end",
        "Agent",
    );
}

#[test]
fn dl_dangling_links_are_reported() {
    let mut document = linear_dl();
    document.nodes[1].next = Some("ghost".to_string());
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::DanglingEdge,
        "llm",
        "ghost",
    );

    let mut document = valid_dl();
    document.nodes[2].parameters.conditions[1].next = Some("ghost".to_string());
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::DanglingEdge,
        "branch",
        "ghost",
    );
}

#[test]
fn dl_start_requires_the_query_output_and_next() {
    let mut missing_query = dl_start("end");
    missing_query.parameters.outputs = vec![dl_variable("other", Some("用户输入"))];
    let mut wrong_description = dl_start("end");
    wrong_description.parameters.outputs = vec![dl_variable("query", Some("user input"))];
    let mut missing_next = dl_start("end");
    missing_next.next = None;

    for node in [missing_query, wrong_description, missing_next] {
        let document = DlDocument {
            nodes: vec![node, dl_end()],
        };
        let validation = DlReflector::new().validate_dl(&document);
        assert!(!validation.passed);
        assert_eq!(
            dl_kinds(&validation),
            vec![ReflectionIssueKind::MissingField]
        );
        assert_eq!(validation.issues[0].node_id.as_deref(), Some("start"));
    }

    let mut document = valid_dl();
    document.nodes[0].next = None;
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::MissingField,
        "start",
        "next",
    );
}

#[test]
fn dl_llm_requires_variables_prompts_and_next() {
    let mut missing_inputs = dl_llm("llm", "end");
    missing_inputs.parameters.inputs.clear();
    let mut missing_outputs = dl_llm("llm", "end");
    missing_outputs.parameters.outputs.clear();
    let mut missing_system = dl_llm("llm", "end");
    missing_system.parameters.configs.remove("system_prompt");
    let mut missing_user = dl_llm("llm", "end");
    missing_user.parameters.configs.remove("user_prompt");
    let mut missing_next = dl_llm("llm", "end");
    missing_next.next = Some("  ".to_string());

    let fixtures = [
        (missing_inputs, "parameters.inputs"),
        (missing_outputs, "parameters.outputs"),
        (missing_system, "parameters.configs.system_prompt"),
        (missing_user, "parameters.configs.user_prompt"),
        (missing_next, "next"),
    ];
    for (node, field) in fixtures {
        let document = single_step_dl(node);
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(&validation, ReflectionIssueKind::MissingField, "llm", field);
    }
}

#[test]
fn dl_questioner_requires_variables_prompt_and_next() {
    let mut missing_inputs = dl_questioner("asker", "end");
    missing_inputs.parameters.inputs.clear();
    let mut missing_outputs = dl_questioner("asker", "end");
    missing_outputs.parameters.outputs.clear();
    let mut missing_prompt = dl_questioner("asker", "end");
    missing_prompt.parameters.configs.remove("prompt");
    let mut missing_next = dl_questioner("asker", "end");
    missing_next.next = None;

    let fixtures = [
        (missing_inputs, "parameters.inputs"),
        (missing_outputs, "parameters.outputs"),
        (missing_prompt, "parameters.configs.prompt"),
        (missing_next, "next"),
    ];
    for (node, field) in fixtures {
        let document = single_step_dl(node);
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            "asker",
            field,
        );
    }
}

#[test]
fn dl_plugin_requires_variables_tool_id_and_next() {
    let mut missing_inputs = dl_plugin("tool", "end");
    missing_inputs.parameters.inputs.clear();
    let mut missing_outputs = dl_plugin("tool", "end");
    missing_outputs.parameters.outputs.clear();
    let mut missing_tool = dl_plugin("tool", "end");
    missing_tool.parameters.configs.remove("tool_id");
    let mut missing_next = dl_plugin("tool", "end");
    missing_next.next = None;

    let fixtures = [
        (missing_inputs, "parameters.inputs"),
        (missing_outputs, "parameters.outputs"),
        (missing_tool, "parameters.configs.tool_id"),
        (missing_next, "next"),
    ];
    for (node, field) in fixtures {
        let document = single_step_dl(node);
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            "tool",
            field,
        );
    }
}

#[test]
fn dl_code_requires_variables_code_and_next() {
    let mut missing_inputs = dl_code("runner", "end");
    missing_inputs.parameters.inputs.clear();
    let mut missing_outputs = dl_code("runner", "end");
    missing_outputs.parameters.outputs.clear();
    let mut missing_code = dl_code("runner", "end");
    missing_code.parameters.configs.remove("code");
    let mut missing_next = dl_code("runner", "end");
    missing_next.next = None;

    let fixtures = [
        (missing_inputs, "parameters.inputs"),
        (missing_outputs, "parameters.outputs"),
        (missing_code, "parameters.configs.code"),
        (missing_next, "next"),
    ];
    for (node, field) in fixtures {
        let document = single_step_dl(node);
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            "runner",
            field,
        );
    }
}

#[test]
fn dl_output_requires_inputs_template_and_next() {
    let mut missing_inputs = dl_output("show", "end");
    missing_inputs.parameters.inputs.clear();
    let mut missing_template = dl_output("show", "end");
    missing_template.parameters.configs.remove("template");
    let mut missing_next = dl_output("show", "end");
    missing_next.next = None;

    let fixtures = [
        (missing_inputs, "parameters.inputs"),
        (missing_template, "parameters.configs.template"),
        (missing_next, "next"),
    ];
    for (node, field) in fixtures {
        let document = single_step_dl(node);
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            "show",
            field,
        );
    }
}

#[test]
fn dl_end_requires_inputs_and_template() {
    let mut missing_inputs = dl_end();
    missing_inputs.parameters.inputs.clear();
    let mut missing_template = dl_end();
    missing_template.parameters.configs.remove("template");

    for (node, field) in [
        (missing_inputs, "parameters.inputs"),
        (missing_template, "parameters.configs.template"),
    ] {
        let document = DlDocument {
            nodes: vec![dl_start("end"), node],
        };
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(&validation, ReflectionIssueKind::MissingField, "end", field);
    }
}

#[test]
fn dl_intent_detection_requires_inputs_prompt_and_conditions() {
    let valid = single_step_dl(dl_intent("intent"));
    assert_passes(&DlReflector::new().validate_dl(&valid));

    let mut missing_inputs = dl_intent("intent");
    missing_inputs.parameters.inputs.clear();
    let mut missing_prompt = dl_intent("intent");
    missing_prompt.parameters.configs.remove("prompt");
    let mut missing_conditions = dl_intent("intent");
    missing_conditions.parameters.conditions.clear();

    let fixtures = [
        (missing_inputs, "parameters.inputs"),
        (missing_prompt, "parameters.configs.prompt"),
        (missing_conditions, "parameters.conditions"),
    ];
    for (node, field) in fixtures {
        let document = single_step_dl(node);
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            "intent",
            field,
        );
    }
}

#[test]
fn dl_intent_detection_checks_its_expressions() {
    let mut missing_raw_output = dl_intent("intent");
    missing_raw_output.parameters.conditions[0].expression = Some("contain hello".to_string());
    let mut missing_contain = dl_intent("intent");
    missing_contain.parameters.conditions[0].expression =
        Some("${intent.rawOutput} equals hello".to_string());
    let mut missing_default = dl_intent("intent");
    missing_default.parameters.conditions.pop();

    let document = single_step_dl(missing_raw_output);
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::MissingField,
        "intent",
        "rawOutput",
    );

    let document = single_step_dl(missing_contain);
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::InvalidExpression,
        "intent",
        "contain",
    );

    let document = single_step_dl(missing_default);
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::MissingDefaultBranch,
        "intent",
        "default",
    );

    let mut missing_fields = dl_intent("intent");
    missing_fields.parameters.conditions[0].branch = String::new();
    missing_fields.parameters.conditions[0].description = "   ".to_string();
    missing_fields.parameters.conditions[0].next = None;
    let document = single_step_dl(missing_fields);
    let validation = DlReflector::new().validate_dl(&document);
    assert_eq!(
        dl_kinds(&validation),
        vec![
            ReflectionIssueKind::MissingField,
            ReflectionIssueKind::MissingField,
            ReflectionIssueKind::MissingField,
        ]
    );
    assert!(validation.issues[0].message.contains("branch"));
    assert!(validation.issues[1].message.contains("description"));
    assert!(validation.issues[2].message.contains("next"));
}

#[test]
fn dl_branch_requires_route_fields() {
    let mut missing_branch = dl_branch();
    missing_branch.parameters.conditions[0].branch = String::new();
    let mut missing_description = dl_branch();
    missing_description.parameters.conditions[0].description = " ".to_string();
    let mut missing_next = dl_branch();
    missing_next.parameters.conditions[0].next = None;

    let fixtures = [
        (missing_branch, "branch"),
        (missing_description, "description"),
        (missing_next, "next"),
    ];
    for (branch, field) in fixtures {
        let mut document = valid_dl();
        document.nodes[2] = branch;
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingField,
            "branch",
            field,
        );
    }
}

#[test]
fn dl_branch_requires_one_expression_form() {
    let mut neither = dl_branch();
    neither.parameters.conditions[0].expression = None;
    let mut both = dl_branch();
    both.parameters.conditions[0].expressions = Some(Vec::new());

    for branch in [neither, both] {
        let mut document = valid_dl();
        document.nodes[2] = branch;
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::MissingConditionExpression,
            "branch",
            "`expression` or `expressions`",
        );
    }

    // A `default` expression still marks the fallback even when `expressions`
    // is present too, so only the exactly-one rule is reported.
    let mut default_and_list = dl_branch();
    default_and_list.parameters.conditions.truncate(1);
    default_and_list.parameters.conditions[0].expression = Some("default".to_string());
    default_and_list.parameters.conditions[0].expressions =
        Some(vec!["${llm.result} contain foo".to_string()]);
    let mut document = valid_dl();
    document.nodes[2] = default_and_list;
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::MissingConditionExpression,
        "branch",
        "`expression` or `expressions`",
    );
}

#[test]
fn dl_branch_checks_operators() {
    let mut unsupported = dl_branch();
    unsupported.parameters.conditions[0].expression =
        Some("${llm.result} frobnicate ${llm.result}".to_string());
    let mut no_operator = dl_branch();
    no_operator.parameters.conditions[0].expression = Some("${llm.result}".to_string());
    let mut valid_operator = dl_branch();
    valid_operator.parameters.conditions[0].expression =
        Some("${llm.result} not_contain foo".to_string());

    for (branch, needle) in [
        (unsupported, "frobnicate"),
        (no_operator, "has no operator"),
    ] {
        let mut document = valid_dl();
        document.nodes[2] = branch;
        let validation = DlReflector::new().validate_dl(&document);
        assert_issue(
            &validation,
            ReflectionIssueKind::InvalidExpression,
            "branch",
            needle,
        );
    }

    let mut document = valid_dl();
    document.nodes[2] = valid_operator;
    assert_passes(&DlReflector::new().validate_dl(&document));
}

#[test]
fn dl_branch_checks_variable_references() {
    let mut unknown = dl_branch();
    unknown.parameters.conditions[0].expression = Some("${ghost.result} contain foo".to_string());

    let mut document = valid_dl();
    document.nodes[2] = unknown;
    let validation = DlReflector::new().validate_dl(&document);

    assert_issue(
        &validation,
        ReflectionIssueKind::InvalidReference,
        "branch",
        "${ghost.result}",
    );

    let mut empty_placeholder = dl_branch();
    empty_placeholder.parameters.conditions[0].expression =
        Some("${llm.result} contain ${}".to_string());
    let mut document = valid_dl();
    document.nodes[2] = empty_placeholder;
    assert_passes(&DlReflector::new().validate_dl(&document));

    let mut nested = dl_branch();
    nested.parameters.conditions[0].expression =
        Some("${llm.result} contain ${ ${ghost.result}".to_string());
    let mut document = valid_dl();
    document.nodes[2] = nested;
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::InvalidReference,
        "branch",
        "${ghost.result}",
    );
}

#[test]
fn dl_branch_requires_a_default_branch() {
    let mut branch = dl_branch();
    branch.parameters.conditions.pop();

    let mut document = valid_dl();
    document.nodes[2] = branch;
    let validation = DlReflector::new().validate_dl(&document);

    assert_issue(
        &validation,
        ReflectionIssueKind::MissingDefaultBranch,
        "branch",
        "`default` branch",
    );
}

#[test]
fn dl_branch_checks_the_expressions_list() {
    let mut valid = dl_branch();
    valid.parameters.conditions[0].expression = None;
    valid.parameters.conditions[0].expressions =
        Some(vec!["${llm.result} contain foo".to_string()]);

    let mut document = valid_dl();
    document.nodes[2] = valid;
    assert_passes(&DlReflector::new().validate_dl(&document));

    let mut unknown = dl_branch();
    unknown.parameters.conditions[0].expression = None;
    unknown.parameters.conditions[0].expressions = Some(vec![
        "${llm.result} contain foo".to_string(),
        "${ghost.result} contain foo".to_string(),
    ]);
    let mut document = valid_dl();
    document.nodes[2] = unknown;
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::InvalidReference,
        "branch",
        "${ghost.result}",
    );

    let mut bad_operator = dl_branch();
    bad_operator.parameters.conditions[0].expression = None;
    bad_operator.parameters.conditions[0].expressions =
        Some(vec!["${llm.result} frobnicate foo".to_string()]);
    let mut document = valid_dl();
    document.nodes[2] = bad_operator;
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::InvalidExpression,
        "branch",
        "frobnicate",
    );
}

#[test]
fn dl_references_need_an_earlier_declaration() {
    let mut branch = dl_branch();
    branch.parameters.conditions[0].expression = Some("${llm.result} contain foo".to_string());
    let early_branch = DlDocument {
        nodes: vec![
            dl_start("branch"),
            branch.clone(),
            dl_llm("llm", "end"),
            dl_end(),
        ],
    };
    let validation = DlReflector::new().validate_dl(&early_branch);
    assert_issue(
        &validation,
        ReflectionIssueKind::InvalidReference,
        "branch",
        "${llm.result}",
    );

    let late_branch = DlDocument {
        nodes: vec![dl_start("llm"), dl_llm("llm", "branch"), branch, dl_end()],
    };
    assert_passes(&DlReflector::new().validate_dl(&late_branch));
}

#[test]
fn dl_validation_and_new_issue_kinds_round_trip_as_json() {
    let kinds = [
        (ReflectionIssueKind::MissingField, "missing_field"),
        (ReflectionIssueKind::InvalidExpression, "invalid_expression"),
        (
            ReflectionIssueKind::MissingDefaultBranch,
            "missing_default_branch",
        ),
        (ReflectionIssueKind::InvalidReference, "invalid_reference"),
        (
            ReflectionIssueKind::MissingConditionExpression,
            "missing_condition_expression",
        ),
        (ReflectionIssueKind::UnknownNodeType, "unknown_node_type"),
    ];
    for (kind, wire) in kinds {
        assert_eq!(serde_json::to_value(kind).expect("serialize"), json!(wire));
        assert_eq!(
            serde_json::from_value::<ReflectionIssueKind>(json!(wire)).expect("parse"),
            kind
        );
    }

    let validation = DlValidation {
        passed: false,
        issues: vec![ReflectionIssue {
            kind: ReflectionIssueKind::InvalidReference,
            message: "node `branch`: condition 0 references `x`".to_string(),
            node_id: Some("branch".to_string()),
        }],
    };
    let text = serde_json::to_string(&validation).expect("serialize");
    let parsed: DlValidation = serde_json::from_str(&text).expect("parse");
    assert_eq!(parsed, validation);
    assert_eq!(
        serde_json::from_str::<DlValidation>("{}").expect("parse"),
        DlValidation::default()
    );
    assert!(!DlValidation::default().passed);
}

#[test]
fn default_max_retries_matches_python() {
    assert_eq!(DEFAULT_MAX_RETRIES, 3);
}
