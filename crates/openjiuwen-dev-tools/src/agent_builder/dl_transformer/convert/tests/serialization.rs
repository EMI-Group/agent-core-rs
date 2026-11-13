use super::super::*;
use super::support::*;
use serde_json::json;

#[test]
fn unknown_node_type_is_rejected() {
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![node("mystery", "Mystery", "unknown")],
        })
        .expect_err("unknown type");
    assert!(
        error.to_string().contains("unknown DL node type: Mystery"),
        "{error}"
    );
}

#[test]
fn missing_configurations_are_rejected() {
    fn error_for(nodes: Vec<DlNode>) -> String {
        DlTransformer::new()
            .transform(&DlDocument { nodes })
            .expect_err("conversion fails")
            .to_string()
    }

    let llm = |configs: Vec<(&str, Value)>| {
        let mut node = node("llm", "LLM", "generate");
        for (key, value) in configs {
            node = with_config(node, key, value);
        }
        vec![node]
    };

    assert!(error_for(llm(vec![("user_prompt", json!("u"))]))
        .contains("node llm: configs.system_prompt is required"));
    assert!(error_for(llm(vec![("system_prompt", json!("s"))]))
        .contains("node llm: configs.user_prompt is required"));
    assert!(error_for(llm(vec![
        ("system_prompt", json!("  ")),
        ("user_prompt", json!("u"))
    ]))
    .contains("node llm: configs.system_prompt must not be blank"));
    assert!(error_for(llm(vec![
        ("system_prompt", json!(5)),
        ("user_prompt", json!("u"))
    ]))
    .contains("node llm: configs.system_prompt must be a string"));

    assert!(error_for(vec![node("code", "Code", "run")])
        .contains("node code: configs.code is required"));
    assert!(error_for(vec![node("plugin", "Plugin", "call")])
        .contains("node plugin: configs.tool_id is required"));
    assert!(error_for(vec![node("out", "Output", "format")])
        .contains("node out: configs.template is required"));
    assert!(error_for(vec![node("ask", "Questioner", "ask")])
        .contains("node ask: configs.prompt is required"));
}

#[test]
fn invalid_input_values_are_rejected() {
    let missing = vec![with_inputs(
        node("end", "End", "exit"),
        vec![variable("result", None, None, None)],
    )];
    let error = DlTransformer::new()
        .transform(&DlDocument { nodes: missing })
        .expect_err("missing value");
    assert!(
        error
            .to_string()
            .contains("node end: input result has no value"),
        "{error}"
    );

    let invalid = vec![with_inputs(
        node("end", "End", "exit"),
        vec![input("result", None, json!("${oops}"))],
    )];
    let error = DlTransformer::new()
        .transform(&DlDocument { nodes: invalid })
        .expect_err("invalid reference");
    assert!(
        error
            .to_string()
            .contains("node end: invalid reference variable expression: ${oops}"),
        "{error}"
    );
}

#[test]
fn empty_documents_and_structural_errors_are_rejected() {
    let error = DlTransformer::new()
        .transform(&DlDocument::default())
        .expect_err("empty document");
    assert!(
        error.to_string().contains("DL document has no nodes"),
        "{error}"
    );

    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_next(node("start", "Start", "entry"), "ghost"),
                node("start", "Start", "entry"),
            ],
        })
        .expect_err("dangling next");
    assert!(
        error.to_string().contains("duplicate node id: start"),
        "{error}"
    );
}

#[test]
fn transform_str_parses_fenced_documents() {
    let text = r#"Here is the workflow:
```json
[
  {
    "id": "start",
    "type": "Start",
    "description": "entry",
    "next": "end",
    "parameters": {
      "outputs": [{"name": "query", "type": "String", "description": "q"}]
    }
  },
  {"id": "end", "type": "End", "description": "exit"}
]
```
"#;
    let workflow = DlTransformer::new()
        .transform_str(text)
        .expect("fenced document");
    let value = workflow.to_value().expect("serializes");
    assert_eq!(value["nodes"][0]["type"], json!("1"));
    assert_eq!(
        value["nodes"][0]["data"]["outputs"]["required"],
        json!(["query"])
    );
    assert_eq!(
        value["edges"],
        json!([{"source_node_id": "start", "target_node_id": "end"}])
    );

    let error = DlTransformer::new()
        .transform_str("```json\n{\"nodes\": []}\n```")
        .expect_err("object instead of array");
    assert!(
        error.to_string().contains("failed to parse DL JSON"),
        "{error}"
    );
}

#[test]
fn to_workflow_dsl_validates_multi_node_documents() {
    let nodes = vec![
        with_next(
            with_outputs(
                node("start", "Start", "entry"),
                vec![output("query", Some("String"), Some("user input"))],
            ),
            "llm",
        ),
        with_next(
            with_outputs(
                with_inputs(
                    with_config(
                        with_config(node("llm", "LLM", "generate"), "system_prompt", json!("s")),
                        "user_prompt",
                        json!("${\"start.query\"}"),
                    ),
                    vec![input("query", Some("String"), json!("${start.query}"))],
                ),
                vec![output("result", Some("String"), None)],
            ),
            "end",
        ),
        with_inputs(
            node("end", "End", "exit"),
            vec![input("result", Some("String"), json!("${llm.result}"))],
        ),
    ];

    let workflow = transform(nodes);
    let dsl = workflow.to_workflow_dsl().expect("valid workflow DSL");
    assert_eq!(dsl.nodes.len(), 3);
    assert_eq!(dsl.nodes[0].node_type, "1");
    assert_eq!(dsl.nodes[1].node_type, "3");
    assert_eq!(dsl.nodes[2].node_type, "2");
    assert_eq!(dsl.nodes[1].title, "generate");
    assert_eq!(dsl.nodes[1].description, "generate");
    assert_eq!(dsl.nodes[1].data, workflow.nodes[1].data);
    assert_eq!(dsl.edges.len(), 2);
    assert_eq!(dsl.edges[0].source, "start");
    assert_eq!(dsl.edges[0].target, "llm");
    assert_eq!(dsl.edges[0].condition, None);
    dsl.validate().expect("dsl validates");
}

#[test]
fn platform_document_keeps_falsy_values_and_drops_nulls() {
    let nodes = vec![with_config(
        with_config(
            node("ask", "Questioner", "ask user"),
            "prompt",
            json!("user"),
        ),
        "history_enable",
        json!(false),
    )];

    let text = transform(nodes).to_json_string().expect("serializes");
    assert!(!text.contains("null"), "{text}");
    assert!(text.contains("\"history_enable\": false"), "{text}");
    assert!(text.contains("\"max_response\": 3"), "{text}");
    assert!(text.contains("\"content\": \"\""), "{text}");
}
