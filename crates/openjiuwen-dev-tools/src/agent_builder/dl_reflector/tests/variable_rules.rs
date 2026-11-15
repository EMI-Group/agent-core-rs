use super::super::*;
use super::support::*;
use crate::agent_builder::dl_transformer::model::DlVariable;
use serde_json::json;

// ---------------------------------------------------------------------------
// `DlReflector::validate_dl` variable-list checks
// ---------------------------------------------------------------------------

/// A document `Start -> LLM -> End` whose LLM input references the start output.
fn referenced_dl() -> DlDocument {
    let mut llm = dl_llm("llm", "end");
    llm.parameters.inputs = vec![dl_reference("query", "${start.query}")];
    DlDocument {
        nodes: vec![dl_start("llm"), llm, dl_end()],
    }
}

#[test]
fn dl_referenced_variables_pass_validation() {
    assert_passes(&DlReflector::new().validate_dl(&referenced_dl()));
    assert_passes(&DlReflector::new().validate_dl(&valid_dl()));
    assert_passes(&DlReflector::new().validate_dl(&linear_dl()));
}

#[test]
fn dl_duplicate_input_names_are_reported() {
    let mut node = dl_llm("llm", "end");
    node.parameters.inputs = vec![dl_input("query"), dl_input("query")];

    let validation = DlReflector::new().validate_dl(&single_step_dl(node));

    assert_issue(
        &validation,
        ReflectionIssueKind::DuplicateVariable,
        "llm",
        "`parameters.inputs` declares the name `query` more than once",
    );
}

#[test]
fn dl_duplicate_output_names_are_reported() {
    let mut node = dl_llm("llm", "end");
    node.parameters.outputs = vec![
        dl_variable("result", Some("the answer")),
        dl_variable("result", Some("the answer again")),
    ];

    let validation = DlReflector::new().validate_dl(&single_step_dl(node));

    assert_issue(
        &validation,
        ReflectionIssueKind::DuplicateVariable,
        "llm",
        "`parameters.outputs` declares the name `result` more than once",
    );
}

#[test]
fn dl_inputs_require_a_value() {
    let mut node = dl_llm("llm", "end");
    node.parameters.inputs = vec![dl_variable("query", None)];

    let validation = DlReflector::new().validate_dl(&single_step_dl(node));

    assert_issue(
        &validation,
        ReflectionIssueKind::MissingField,
        "llm",
        "input `query`",
    );
    assert!(validation.issues[0].message.contains("value"));
}

#[test]
fn dl_non_string_values_are_accepted_without_reference_checks() {
    let mut node = dl_llm("llm", "end");
    node.parameters.inputs = vec![DlVariable {
        name: "query".to_string(),
        value: Some(json!(3)),
        ..DlVariable::default()
    }];

    assert_passes(&DlReflector::new().validate_dl(&single_step_dl(node)));
}

#[test]
fn dl_multi_placeholder_values_are_reported() {
    let mut node = dl_llm("llm", "end");
    node.parameters.inputs = vec![dl_reference("query", "${start.query} and ${start.query}")];

    let validation = DlReflector::new().validate_dl(&single_step_dl(node));

    assert_issue(
        &validation,
        ReflectionIssueKind::MultiPlaceholderValue,
        "llm",
        "2 reference variables",
    );
}

#[test]
fn dl_input_references_need_a_declaration() {
    let mut unknown = dl_llm("llm", "end");
    unknown.parameters.inputs = vec![dl_reference("query", "${ghost.result}")];

    let validation = DlReflector::new().validate_dl(&single_step_dl(unknown));
    assert_issue(
        &validation,
        ReflectionIssueKind::UndeclaredReference,
        "llm",
        "${ghost.result}",
    );

    // A reference to a later node's output is not declared yet.
    let mut early = dl_llm("llm", "later");
    early.parameters.inputs = vec![dl_reference("query", "${later.result}")];
    let document = DlDocument {
        nodes: vec![dl_start("llm"), early, dl_code("later", "end"), dl_end()],
    };
    let validation = DlReflector::new().validate_dl(&document);
    assert_issue(
        &validation,
        ReflectionIssueKind::UndeclaredReference,
        "llm",
        "${later.result}",
    );

    // The same reference resolves once the declaring node comes first.
    let mut late = dl_llm("llm", "end");
    late.parameters.inputs = vec![dl_reference("query", "${runner.result}")];
    let document = DlDocument {
        nodes: vec![dl_start("runner"), dl_code("runner", "llm"), late, dl_end()],
    };
    assert_passes(&DlReflector::new().validate_dl(&document));
}

#[test]
fn dl_outputs_require_a_description() {
    let mut node = dl_llm("llm", "end");
    node.parameters.outputs = vec![dl_variable("result", None)];

    let validation = DlReflector::new().validate_dl(&single_step_dl(node));

    assert_issue(
        &validation,
        ReflectionIssueKind::MissingField,
        "llm",
        "output `result`",
    );
    assert!(validation.issues[0].message.contains("description"));
}

#[test]
fn dl_start_outputs_are_checked_too() {
    let mut start = dl_start("end");
    start.parameters.outputs = vec![dl_variable("extra", None)];

    let document = DlDocument {
        nodes: vec![start, dl_end()],
    };
    let validation = DlReflector::new().validate_dl(&document);

    assert_eq!(
        dl_kinds(&validation),
        vec![
            ReflectionIssueKind::MissingField,
            ReflectionIssueKind::MissingField
        ]
    );
    assert_eq!(validation.issues[0].node_id.as_deref(), Some("start"));
    assert!(validation.issues[0].message.contains("output `extra`"));
    assert!(validation.issues[1].message.contains("`query`"));
}

#[test]
fn dl_variable_checks_are_deterministic() {
    let mut node = dl_llm("llm", "end");
    node.parameters.inputs = vec![
        dl_reference("query", "${ghost.result}"),
        dl_variable("query", None),
        dl_reference("extra", "${a.x} ${b.y}"),
    ];
    node.parameters.outputs = vec![dl_variable("result", None)];

    let document = single_step_dl(node);
    let first = DlReflector::new().validate_dl(&document);
    let second = DlReflector::new().validate_dl(&document);

    assert_eq!(first, second);
    assert_eq!(
        dl_kinds(&first),
        vec![
            ReflectionIssueKind::UndeclaredReference,
            ReflectionIssueKind::DuplicateVariable,
            ReflectionIssueKind::MissingField,
            ReflectionIssueKind::MultiPlaceholderValue,
            ReflectionIssueKind::MissingField,
        ]
    );
    assert_eq!(document.nodes.len(), 3, "the document is untouched");
}

#[test]
fn new_issue_kinds_round_trip_as_json() {
    let kinds = [
        (ReflectionIssueKind::DuplicateVariable, "duplicate_variable"),
        (
            ReflectionIssueKind::MultiPlaceholderValue,
            "multi_placeholder_value",
        ),
        (
            ReflectionIssueKind::UndeclaredReference,
            "undeclared_reference",
        ),
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
            kind: ReflectionIssueKind::UndeclaredReference,
            message: "node `llm`: input `query` `value` references `x`".to_string(),
            node_id: Some("llm".to_string()),
        }],
    };
    let text = serde_json::to_string(&validation).expect("serialize");
    let parsed: DlValidation = serde_json::from_str(&text).expect("parse");
    assert_eq!(parsed, validation);
}
