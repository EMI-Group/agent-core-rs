use super::super::*;
use super::support::*;
use serde_json::json;

#[test]
fn questioner_node_renders_prompt_and_generated_outputs() {
    let nodes = vec![
        with_next(
            with_outputs(
                with_inputs(
                    with_config(
                        node("ask", "Questioner", "ask user"),
                        "prompt",
                        json!("Ask the user"),
                    ),
                    vec![input("query", Some("String"), json!("${start.query}"))],
                ),
                vec![output("answer", Some("String"), Some("answer"))],
            ),
            "end",
        ),
        node("end", "End", "exit"),
    ];

    assert_eq!(
        node_json(nodes.clone(), "ask"),
        json!({
            "id": "ask",
            "type": "7",
            "meta": {"position": {"x": 0, "y": 0}},
            "data": {
                "title": "ask user",
                "inputs": {
                    "input_parameters": {
                        "query": {
                            "type": "ref",
                            "content": ["start", "query"],
                            "extra": {"index": 0},
                        },
                    },
                    "llm_param": {
                        "systemPrompt": {"type": "template", "content": "Ask the user"},
                        "prompt": {"type": "template", "content": ""},
                        "mode": {
                            "id": "52",
                            "name": "siliconf-qwen3-8b",
                            "type": "Qwen/Qwen3-8B",
                        },
                    },
                    "system_prompt": {"type": "template", "content": "Ask the user"},
                    "history_enable": false,
                    "max_response": 3,
                },
                "outputs": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "String",
                            "description": "answer",
                            "extra": {"index": 1},
                        },
                        "user_response": {
                            "type": "string",
                            "description": "用户响应输出变量",
                        },
                        "output": {"type": "string", "description": "输出变量"},
                    },
                    "required": ["answer", "user_response"],
                },
            },
        })
    );
    assert_eq!(
        edges_json(nodes),
        json!([{"source_node_id": "ask", "target_node_id": "end"}])
    );
}

#[test]
fn questioner_without_inputs_or_outputs_adds_only_user_response() {
    let nodes = vec![with_config(
        node("ask", "Questioner", "ask user"),
        "prompt",
        json!("Ask the user"),
    )];

    let data = node_data(nodes, "ask");
    assert_eq!(
        data["inputs"]["input_parameters"],
        json!({
            "input": {
                "type": "ref",
                "content": "node_start.query",
                "extra": {"index": 0},
            },
        })
    );
    assert_eq!(
        data["outputs"],
        json!({
            "type": "object",
            "properties": {
                "user_response": {"type": "string", "description": "用户响应输出变量"},
            },
            "required": ["user_response"],
        })
    );
    assert!(data["outputs"]["properties"].get("output").is_none());
}

#[test]
fn code_node_renders_language_code_and_exception_config() {
    let nodes = vec![
        with_next(
            with_outputs(
                with_inputs(
                    with_config(node("code", "Code", "run"), "code", json!("return 1")),
                    vec![input("query", Some("String"), json!("${start.query}"))],
                ),
                vec![output("result", Some("String"), None)],
            ),
            "end",
        ),
        node("end", "End", "exit"),
    ];

    assert_eq!(
        node_json(nodes.clone(), "code"),
        json!({
            "id": "code",
            "type": "10",
            "meta": {"position": {"x": 0, "y": 0}},
            "data": {
                "title": "run",
                "inputs": {
                    "input_parameters": {
                        "query": {
                            "type": "ref",
                            "content": ["start", "query"],
                            "extra": {"index": 0},
                        },
                    },
                    "language": "python",
                    "code": "return 1",
                },
                "outputs": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "String", "extra": {"index": 1}},
                    },
                    "required": ["result"],
                },
                "exception_config": {
                    "retryTimes": 3,
                    "timeoutSeconds": 30,
                    "processType": "break",
                    "executeStep": {"defaultStep": "0", "errorStep": "1"},
                },
            },
        })
    );
    assert_eq!(
        edges_json(nodes),
        json!([{"source_node_id": "code", "target_node_id": "end", "source_port_id": "0"}])
    );
}

/// A plugin document whose single plugin node declares one input and one output.
fn plugin_fixture() -> Vec<DlNode> {
    vec![
        with_next(
            with_outputs(
                with_inputs(
                    with_config(
                        node("plugin", "Plugin", "call tool"),
                        "tool_id",
                        json!("tool-1"),
                    ),
                    vec![input("query", Some("String"), json!("${start.query}"))],
                ),
                vec![output("result", Some("String"), Some("tool result"))],
            ),
            "end",
        ),
        node("end", "End", "exit"),
    ]
}

/// The expected plugin `data` for `plugin_param`.
fn plugin_data(plugin_param: Value) -> Value {
    json!({
        "title": "call tool",
        "inputs": {
            "input_parameters": {
                "query": {
                    "type": "ref",
                    "content": ["start", "query"],
                    "extra": {"index": 0},
                },
            },
            "plugin_param": plugin_param,
        },
        "outputs": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "String",
                    "description": "tool result",
                    "extra": {"index": 1},
                },
                "error_code": {"type": "integer", "extra": {"index": 1}},
                "error_message": {"type": "string", "extra": {"index": 2}},
                "data": {"type": "object", "extra": {"index": 3}, "properties": {}},
            },
            "required": ["error_code", "error_message", "data"],
        },
    })
}

#[test]
fn plugin_node_renders_defaults_without_resources() {
    let nodes = plugin_fixture();
    assert_eq!(
        node_json(nodes.clone(), "plugin"),
        json!({
            "id": "plugin",
            "type": "19",
            "meta": {"position": {"x": 0, "y": 0}},
            "data": plugin_data(json!({
                "toolID": "",
                "toolName": "",
                "pluginID": "",
                "pluginName": "",
                "pluginVersion": "draft",
            })),
        })
    );
    assert_eq!(
        edges_json(nodes),
        json!([{"source_node_id": "plugin", "target_node_id": "end"}])
    );
}

#[test]
fn plugin_node_resolves_resources_and_falls_back_to_draft() {
    let resources = vec![PluginResource {
        tool_id: "tool-1".to_string(),
        tool_name: "search".to_string(),
        plugin_id: "plugin-9".to_string(),
        plugin_name: "Search".to_string(),
        plugin_version: String::new(),
    }];
    let workflow = DlTransformer::new()
        .transform_with_resources(
            &DlDocument {
                nodes: plugin_fixture(),
            },
            &resources,
        )
        .expect("transforms");
    let value = workflow.to_value().expect("serializes");
    let data = value["nodes"]
        .as_array()
        .expect("nodes")
        .iter()
        .find(|node| node["id"] == json!("plugin"))
        .expect("plugin node")["data"]
        .clone();

    assert_eq!(
        data,
        plugin_data(json!({
            "toolID": "tool-1",
            "toolName": "search",
            "pluginID": "plugin-9",
            "pluginName": "Search",
            "pluginVersion": "draft",
        }))
    );
}

#[test]
fn plugin_node_keeps_a_declared_resource_version() {
    let resources = vec![PluginResource {
        tool_id: "tool-1".to_string(),
        tool_name: "search".to_string(),
        plugin_id: "plugin-9".to_string(),
        plugin_name: "Search".to_string(),
        plugin_version: "1.0.0".to_string(),
    }];
    let workflow = DlTransformer::new()
        .transform_with_resources(
            &DlDocument {
                nodes: plugin_fixture(),
            },
            &resources,
        )
        .expect("transforms");
    let value = workflow.to_value().expect("serializes");
    let data = &value["nodes"][0]["data"];
    assert_eq!(
        data["inputs"]["plugin_param"]["pluginVersion"],
        json!("1.0.0")
    );
}

#[test]
fn output_node_renders_template() {
    let nodes = vec![with_inputs(
        with_config(
            node("out", "Output", "format"),
            "template",
            json!("Answer: ${llm.result}"),
        ),
        vec![input("answer", Some("String"), json!("${llm.result}"))],
    )];

    assert_eq!(
        node_json(nodes.clone(), "out"),
        json!({
            "id": "out",
            "type": "9",
            "meta": {"position": {"x": 0, "y": 0}},
            "data": {
                "title": "format",
                "inputs": {
                    "input_parameters": {
                        "answer": {
                            "type": "ref",
                            "content": ["llm", "result"],
                            "extra": {"index": 0},
                        },
                    },
                    "content": {"type": "template", "content": "Answer: ${llm.result}"},
                },
            },
        })
    );
    assert_eq!(edges_json(nodes), json!([]));
}
