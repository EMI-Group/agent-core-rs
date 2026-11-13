use super::super::*;
use super::support::*;
use openjiuwen_test_support::ScriptedModel;

// ---------------------------------------------------------------------------
// Per-node-type DL validation (`DlReflector::validate_dl`)
// ---------------------------------------------------------------------------

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
