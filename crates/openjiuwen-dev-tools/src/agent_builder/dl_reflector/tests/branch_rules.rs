use super::super::*;
use super::support::*;
use serde_json::json;

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
