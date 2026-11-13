//! Shared fixtures for the DL reflector tests.

use super::super::*;
use crate::agent_builder::dl_transformer::model::{DlCondition, DlDocument, DlNode, DlVariable};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;

/// A node titled after its id.
pub(super) fn node(id: &str, node_type: &str) -> DslNode {
    DslNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        title: id.to_string(),
        ..DslNode::default()
    }
}

/// A document with the given nodes and edges.
pub(super) fn graph(nodes: Vec<DslNode>, edges: Vec<DslEdge>) -> WorkflowDsl {
    WorkflowDsl {
        name: "demo".to_string(),
        description: "a demo workflow".to_string(),
        nodes,
        edges,
    }
}

/// A valid `start` -> `end` document.
pub(super) fn valid_document() -> WorkflowDsl {
    graph(
        vec![node("start", "start"), node("end", "end")],
        vec![edge("start", "end")],
    )
}

/// Coerces a scripted model into the trait object the reflector takes.
pub(super) fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The issue kinds, in recorded order.
pub(super) fn kinds(reflection: &Reflection) -> Vec<ReflectionIssueKind> {
    reflection.issues.iter().map(|issue| issue.kind).collect()
}

/// Asserts the repaired document validates.
pub(super) fn assert_valid(reflection: &Reflection) {
    reflection.repaired.validate().expect("repaired validates");
}

/// A DL variable with an optional description.
pub(super) fn dl_variable(name: &str, description: Option<&str>) -> DlVariable {
    DlVariable {
        name: name.to_string(),
        description: description.map(str::to_string),
        ..DlVariable::default()
    }
}

/// The Start node's required `query` output.
pub(super) fn query_output() -> DlVariable {
    dl_variable("query", Some("用户输入"))
}

/// A DL node with a non-blank description.
pub(super) fn dl_node(id: &str, node_type: &str) -> DlNode {
    DlNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        description: format!("{node_type} node"),
        ..DlNode::default()
    }
}

/// A Start node linking to `next` and declaring the `query` output.
pub(super) fn dl_start(next: &str) -> DlNode {
    let mut node = dl_node("start", "Start");
    node.next = Some(next.to_string());
    node.parameters.outputs = vec![query_output()];
    node
}

/// An LLM node with variables, both prompts, and a `next` link.
pub(super) fn dl_llm(id: &str, next: &str) -> DlNode {
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
pub(super) fn dl_questioner(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Questioner");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("answer", Some("the answer"))];
    node.parameters.configs = BTreeMap::from([("prompt".to_string(), json!("Ask the user"))]);
    node
}

/// A Plugin node with variables, `tool_id`, and a `next` link.
pub(super) fn dl_plugin(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Plugin");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("result", Some("the result"))];
    node.parameters.configs = BTreeMap::from([("tool_id".to_string(), json!("tool-1"))]);
    node
}

/// A Code node with variables, `code`, and a `next` link.
pub(super) fn dl_code(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Code");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("query", None)];
    node.parameters.outputs = vec![dl_variable("result", Some("the result"))];
    node.parameters.configs = BTreeMap::from([("code".to_string(), json!("return 1"))]);
    node
}

/// An Output node with inputs, `template`, and a `next` link.
pub(super) fn dl_output(id: &str, next: &str) -> DlNode {
    let mut node = dl_node(id, "Output");
    node.next = Some(next.to_string());
    node.parameters.inputs = vec![dl_variable("result", None)];
    node.parameters.configs =
        BTreeMap::from([("template".to_string(), json!("answer: {{result}}"))]);
    node
}

/// An End node with inputs and `template`.
pub(super) fn dl_end() -> DlNode {
    let mut node = dl_node("end", "End");
    node.parameters.inputs = vec![dl_variable("result", None)];
    node.parameters.configs = BTreeMap::from([("template".to_string(), json!("{{result}}"))]);
    node
}

/// One branch condition.
pub(super) fn dl_condition(
    branch: &str,
    description: &str,
    next: &str,
    expression: &str,
) -> DlCondition {
    DlCondition {
        branch: branch.to_string(),
        description: description.to_string(),
        next: Some(next.to_string()),
        expression: Some(expression.to_string()),
        ..DlCondition::default()
    }
}

/// A Branch node: one matching condition over the LLM output plus a default.
pub(super) fn dl_branch() -> DlNode {
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
pub(super) fn dl_intent(id: &str) -> DlNode {
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
pub(super) fn valid_dl() -> DlDocument {
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
pub(super) fn linear_dl() -> DlDocument {
    DlDocument {
        nodes: vec![dl_start("llm"), dl_llm("llm", "end"), dl_end()],
    }
}

/// The document `Start -> node -> End`, where `node` has the given id.
pub(super) fn single_step_dl(node: DlNode) -> DlDocument {
    DlDocument {
        nodes: vec![dl_start(&node.id), node, dl_end()],
    }
}

/// The issue kinds of a validation, in recorded order.
pub(super) fn dl_kinds(validation: &DlValidation) -> Vec<ReflectionIssueKind> {
    validation.issues.iter().map(|issue| issue.kind).collect()
}

/// Asserts the validation failed with exactly one issue of `kind`.
pub(super) fn assert_issue(
    validation: &DlValidation,
    kind: ReflectionIssueKind,
    node_id: &str,
    needle: &str,
) {
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
pub(super) fn assert_passes(validation: &DlValidation) {
    assert_eq!(
        validation,
        &DlValidation {
            passed: true,
            issues: Vec::new(),
        }
    );
}
