use super::super::*;
use super::support::*;
use crate::agent_builder::dl_transformer::model::DlParameters;
use serde_json::json;

#[test]
fn input_variables_are_constants_or_references_with_indices() {
    let nodes = vec![with_outputs(
        with_inputs(
            with_config(
                with_config(
                    node("llm", "LLM", "generate"),
                    "system_prompt",
                    json!("system"),
                ),
                "user_prompt",
                json!("user"),
            ),
            vec![
                input("one", None, json!("literal")),
                input("two", Some("Integer"), json!(5)),
                input("three", Some("String"), json!("${start.query}")),
            ],
        ),
        vec![
            output("first", Some("String"), None),
            output("second", Some("Boolean"), None),
        ],
    )];

    let data = node_data(nodes, "llm");
    assert_eq!(
        data["inputs"]["input_parameters"],
        json!({
            "one": {
                "type": "constant",
                "content": "literal",
                "schema": {"type": "string"},
                "extra": {"index": 0},
            },
            "two": {
                "type": "constant",
                "content": 5,
                "schema": {"type": "Integer"},
                "extra": {"index": 1},
            },
            "three": {
                "type": "ref",
                "content": ["start", "query"],
                "extra": {"index": 2},
            },
        })
    );
    assert_eq!(
        data["outputs"]["properties"]["first"]["extra"],
        json!({"index": 3})
    );
    assert_eq!(
        data["outputs"]["properties"]["second"]["extra"],
        json!({"index": 4})
    );
}

#[test]
fn reference_variables_split_on_of_suffix() {
    let nodes = vec![with_inputs(
        node("end", "End", "exit"),
        vec![input("result", None, json!("${a.b_of_c}"))],
    )];

    assert_eq!(
        node_data(nodes, "end")["inputs"]["input_parameters"]["result"],
        json!({
            "type": "ref",
            "content": ["a", "c", "b"],
            "extra": {"index": 0},
        })
    );
}

#[test]
fn nested_output_names_build_object_schemas() {
    let nodes = vec![with_outputs(
        node("start", "Start", "entry"),
        vec![
            output("b_of_a", Some("String"), Some("nested")),
            output("c_of_a", Some("String"), None),
        ],
    )];

    assert_eq!(
        node_data(nodes, "start")["outputs"],
        json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "String",
                            "description": "nested",
                            "extra": {"index": 0},
                        },
                        "c": {"type": "String", "extra": {"index": 1}},
                    },
                    "required": [],
                },
            },
            "required": ["a"],
        })
    );
}

#[test]
fn object_outputs_keep_properties_and_required() {
    let mut item = output("payload", Some("object"), Some("payload"));
    item.properties = Some(json!({"name": {"type": "string"}}));
    item.required = Some(vec!["name".to_string()]);
    let nodes = vec![with_outputs(node("start", "Start", "entry"), vec![item])];

    assert_eq!(
        node_data(nodes, "start")["outputs"]["properties"]["payload"],
        json!({
            "type": "object",
            "description": "payload",
            "extra": {"index": 0},
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
    );
}

#[test]
fn array_outputs_keep_items() {
    let mut item = output("rows", Some("Array"), Some("rows"));
    item.items = Some(json!({"type": "string"}));
    let nodes = vec![with_outputs(node("start", "Start", "entry"), vec![item])];

    assert_eq!(
        node_data(nodes, "start")["outputs"]["properties"]["rows"],
        json!({
            "type": "Array",
            "description": "rows",
            "extra": {"index": 0},
            "items": {"type": "string"},
        })
    );
}

#[test]
fn node_conversion_helpers_parse_variables() {
    assert_eq!(
        parse_variable("${node.variable}"),
        Some(("node".to_string(), "variable".to_string()))
    );
    assert_eq!(
        parse_variable("  ${ node.variable }  "),
        Some(("node".to_string(), "variable".to_string()))
    );
    assert_eq!(
        parse_variable("${用户.输入}"),
        Some(("用户".to_string(), "输入".to_string()))
    );
    assert_eq!(
        parse_variable("${a.b} trailing"),
        Some(("a".to_string(), "b".to_string()))
    );
    assert_eq!(parse_variable("${a.b.c}"), None);
    assert_eq!(parse_variable("${a.}"), None);
    assert_eq!(parse_variable("a.b"), None);
    assert_eq!(parse_variable("${.b}"), None);
}

#[test]
fn output_names_reverse_of_splits() {
    assert_eq!(output_names("plain"), vec!["plain".to_string()]);
    assert_eq!(
        output_names("b_of_a"),
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(
        output_names("c_of_b_of_a"),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn converter_utilities_are_exposed() {
    assert_eq!(
        llm_model_config(),
        json!({"id": "52", "name": "siliconf-qwen3-8b", "type": "Qwen/Qwen3-8B"})
    );
    assert_eq!(
        Value::Object(llm_param("system", "user")),
        json!({
            "systemPrompt": {"type": "template", "content": "system"},
            "prompt": {"type": "template", "content": "user"},
            "mode": {"id": "52", "name": "siliconf-qwen3-8b", "type": "Qwen/Qwen3-8B"},
        })
    );
    assert_eq!(template("hi"), json!({"type": "template", "content": "hi"}));
    assert_eq!(
        convert_reference("end", "${a.b_of_c}").expect("reference"),
        json!({"type": "ref", "content": ["a", "c", "b"]})
    );
    assert_eq!(
        DlTransformer::new()
            .transform(&DlDocument::default())
            .unwrap_err()
            .kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    let _ = DlParameters::default();
}
