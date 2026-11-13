use super::super::*;
use super::support::*;
use serde_json::json;

/// A start node declaring two outputs, followed by an end node.
fn start_fixture() -> Vec<DlNode> {
    vec![
        with_next(
            with_outputs(
                node("start", "Start", "entry"),
                vec![
                    output("query", Some("String"), Some("user query")),
                    output("用户输入", Some("String"), Some("用户输入")),
                ],
            ),
            "end",
        ),
        node("end", "End", "exit"),
    ]
}

#[test]
fn start_node_renders_outputs_and_position() {
    let json = node_json(start_fixture(), "start");

    assert_eq!(
        json,
        json!({
            "id": "start",
            "type": "1",
            "meta": {"position": {"x": 0, "y": 0}},
            "data": {
                "title": "entry",
                "outputs": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "String",
                            "description": "user query",
                            "extra": {"index": 0},
                        },
                        "用户输入": {
                            "type": "String",
                            "description": "用户输入",
                            "extra": {"index": 1},
                        },
                    },
                    "required": ["query", "用户输入"],
                },
            },
        })
    );
    assert_eq!(
        edges_json(start_fixture()),
        json!([{"source_node_id": "start", "target_node_id": "end"}])
    );
}

#[test]
fn end_node_renders_input_parameters() {
    let nodes = vec![
        with_next(node("start", "Start", "entry"), "end"),
        with_inputs(
            node("end", "End", "exit"),
            vec![input("result", Some("String"), json!("${start.query}"))],
        ),
    ];

    assert_eq!(
        node_json(nodes, "end"),
        json!({
            "id": "end",
            "type": "2",
            "meta": {"position": {"x": 20, "y": 20}},
            "data": {
                "title": "exit",
                "inputs": {
                    "input_parameters": {
                        "result": {
                            "type": "ref",
                            "content": ["start", "query"],
                            "extra": {"index": 0},
                        },
                    },
                },
            },
        })
    );
}

#[test]
fn llm_node_renders_templates_and_output_format() {
    let nodes = vec![
        node("start", "Start", "entry"),
        with_next(
            with_outputs(
                with_inputs(
                    with_config(
                        with_config(
                            node("llm", "LLM", "generate"),
                            "system_prompt",
                            json!("You are helpful."),
                        ),
                        "user_prompt",
                        json!("${start.query}"),
                    ),
                    vec![input("query", Some("String"), json!("${start.query}"))],
                ),
                vec![output("result", Some("String"), Some("model output"))],
            ),
            "end",
        ),
        node("end", "End", "exit"),
    ];

    assert_eq!(
        node_json(nodes.clone(), "llm"),
        json!({
            "id": "llm",
            "type": "3",
            "meta": {"position": {"x": 20, "y": 20}},
            "data": {
                "title": "generate",
                "output_format": "text",
                "inputs": {
                    "input_parameters": {
                        "query": {
                            "type": "ref",
                            "content": ["start", "query"],
                            "extra": {"index": 0},
                        },
                    },
                    "llm_param": {
                        "systemPrompt": {"type": "template", "content": "You are helpful."},
                        "prompt": {"type": "template", "content": "${start.query}"},
                        "mode": {
                            "id": "52",
                            "name": "siliconf-qwen3-8b",
                            "type": "Qwen/Qwen3-8B",
                        },
                        "response_format": {"type": "text"},
                    },
                },
                "outputs": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "String",
                            "description": "model output",
                            "extra": {"index": 1},
                        },
                    },
                    "required": [],
                },
            },
        })
    );
    assert_eq!(
        edges_json(nodes),
        json!([{"source_node_id": "llm", "target_node_id": "end"}])
    );
}

#[test]
fn llm_output_format_follows_python_rules() {
    fn llm_with(outputs: Vec<DlVariable>, format: Option<&str>) -> Vec<DlNode> {
        let mut llm = with_config(
            with_config(
                with_outputs(node("llm", "LLM", "generate"), outputs),
                "system_prompt",
                json!("system"),
            ),
            "user_prompt",
            json!("user"),
        );
        if let Some(format) = format {
            llm = with_config(llm, "output_format", json!(format));
        }
        vec![llm]
    }

    let two_text = llm_with(
        vec![
            output("a", Some("String"), None),
            output("b", Some("String"), None),
        ],
        None,
    );
    assert_eq!(
        node_data(two_text.clone(), "llm")["output_format"],
        json!("json")
    );

    let two_markdown = llm_with(
        vec![
            output("a", Some("String"), None),
            output("b", Some("String"), None),
        ],
        Some("markdown"),
    );
    assert_eq!(
        node_data(two_markdown, "llm")["output_format"],
        json!("json")
    );

    let one_unknown = llm_with(vec![output("a", Some("String"), None)], Some("xml"));
    assert_eq!(
        node_data(one_unknown, "llm")["output_format"],
        json!("text")
    );

    let one_json = llm_with(vec![output("a", Some("String"), None)], Some("json"));
    assert_eq!(node_data(one_json, "llm")["output_format"], json!("json"));

    let none = llm_with(Vec::new(), None);
    assert_eq!(node_data(none, "llm")["output_format"], json!("text"));

    let two_json_inputs = node_data(two_text, "llm");
    assert_eq!(
        two_json_inputs["inputs"]["llm_param"]["response_format"],
        json!({"type": "json"})
    );
}

#[test]
fn positions_step_by_twenty() {
    let nodes = vec![
        with_next(node("start", "Start", "entry"), "llm"),
        with_next(
            with_config(
                with_config(node("llm", "LLM", "generate"), "system_prompt", json!("s")),
                "user_prompt",
                json!("u"),
            ),
            "end",
        ),
        node("end", "End", "exit"),
    ];

    let value = transform_value(nodes);
    let positions: Vec<Value> = value["nodes"]
        .as_array()
        .expect("nodes")
        .iter()
        .map(|node| node["meta"]["position"].clone())
        .collect();
    assert_eq!(
        positions,
        vec![
            json!({"x": 0, "y": 0}),
            json!({"x": 20, "y": 20}),
            json!({"x": 40, "y": 40}),
        ]
    );
}
