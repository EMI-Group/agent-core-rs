//! Tests for the DL -> platform workflow transformer.
//!
//! Every test is offline and deterministic: the documents are built in memory,
//! no id or timestamp is generated, and the assertions compare whole JSON
//! objects (Python's wire shape) rather than serialized key order.

use super::*;
use crate::agent_builder::dl_transformer::model::{
    DlCondition, DlDocument, DlNode, DlParameters, DlVariable,
};
use serde_json::json;

/// Builds one variable.
fn variable(
    name: &str,
    var_type: Option<&str>,
    description: Option<&str>,
    value: Option<Value>,
) -> DlVariable {
    DlVariable {
        name: name.to_string(),
        var_type: var_type.map(str::to_string),
        description: description.map(str::to_string),
        value,
        ..DlVariable::default()
    }
}

/// Builds one input variable bound to `value`.
fn input(name: &str, var_type: Option<&str>, value: Value) -> DlVariable {
    variable(name, var_type, None, Some(value))
}

/// Builds one output variable.
fn output(name: &str, var_type: Option<&str>, description: Option<&str>) -> DlVariable {
    variable(name, var_type, description, None)
}

/// Builds a node without a successor or parameters.
fn node(id: &str, node_type: &str, description: &str) -> DlNode {
    DlNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        description: description.to_string(),
        ..DlNode::default()
    }
}

/// Sets a node's successor.
fn with_next(mut node: DlNode, next: &str) -> DlNode {
    node.next = Some(next.to_string());
    node
}

/// Sets a node's declared inputs.
fn with_inputs(mut node: DlNode, inputs: Vec<DlVariable>) -> DlNode {
    node.parameters.inputs = inputs;
    node
}

/// Sets a node's declared outputs.
fn with_outputs(mut node: DlNode, outputs: Vec<DlVariable>) -> DlNode {
    node.parameters.outputs = outputs;
    node
}

/// Sets one `parameters.configs` entry.
fn with_config(mut node: DlNode, key: &str, value: Value) -> DlNode {
    node.parameters.configs.insert(key.to_string(), value);
    node
}

/// Sets a node's branch conditions.
fn with_conditions(mut node: DlNode, conditions: Vec<DlCondition>) -> DlNode {
    node.parameters.conditions = conditions;
    node
}

/// Builds a condition with a branch name and a successor.
fn condition(branch: &str, next: &str) -> DlCondition {
    DlCondition {
        branch: branch.to_string(),
        next: Some(next.to_string()),
        ..DlCondition::default()
    }
}

/// Transforms a document, panicking on error.
fn transform(nodes: Vec<DlNode>) -> PlatformWorkflow {
    DlTransformer::new()
        .transform(&DlDocument { nodes })
        .expect("document transforms")
}

/// Transforms a document and returns its platform JSON.
fn transform_value(nodes: Vec<DlNode>) -> Value {
    transform(nodes).to_value().expect("document serializes")
}

/// The platform JSON of one node.
fn node_json(nodes: Vec<DlNode>, id: &str) -> Value {
    let value = transform_value(nodes);
    value["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .find(|node| node["id"] == json!(id))
        .unwrap_or_else(|| panic!("node {id} is present"))
        .clone()
}

/// The platform `data` of one node.
fn node_data(nodes: Vec<DlNode>, id: &str) -> Value {
    node_json(nodes, id)["data"].clone()
}

/// The platform edges of a document.
fn edges_json(nodes: Vec<DlNode>) -> Value {
    transform_value(nodes)["edges"].clone()
}

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
fn branch_node_renders_conditions_and_ports() {
    let mut positive = condition("positive", "plugin");
    positive.expression = Some("${llm.result} contain 'good'".to_string());
    positive.description = "positive case".to_string();

    let mut combined = condition("combined", "plugin");
    combined.expressions = Some(vec![
        "${llm.result} eq 'a'".to_string(),
        "${llm.result} eq 'b'".to_string(),
    ]);
    combined.operator = Some("or".to_string());

    let mut fallback = condition("fallback", "end");
    fallback.expression = Some("default".to_string());

    let branch = with_conditions(
        node("branch", "Branch", "route"),
        vec![positive, combined, fallback],
    );
    let llm = with_outputs(
        with_config(
            with_config(node("llm", "LLM", "generate"), "system_prompt", json!("s")),
            "user_prompt",
            json!("u"),
        ),
        vec![output("result", Some("String"), None)],
    );
    let nodes = vec![
        llm,
        branch,
        with_config(
            node("plugin", "Plugin", "call tool"),
            "tool_id",
            json!("tool-1"),
        ),
        node("end", "End", "exit"),
    ];

    assert_eq!(
        node_json(nodes.clone(), "branch"),
        json!({
            "id": "branch",
            "type": "4",
            "meta": {"position": {"x": 20, "y": 20}},
            "data": {
                "title": "route",
                "branches": [
                    {
                        "conditions": [{
                            "left": {"type": "ref", "content": ["llm", "result"]},
                            "operator": "contains",
                            "right": {
                                "type": "constant",
                                "content": "'good'",
                                "schema": {"type": "string", "extra": {"weak": true}},
                            },
                        }],
                        "branchId": "positive",
                    },
                    {
                        "conditions": [
                            {
                                "left": {"type": "ref", "content": ["llm", "result"]},
                                "operator": "eq",
                                "right": {
                                    "type": "constant",
                                    "content": "'a'",
                                    "schema": {"type": "string", "extra": {"weak": true}},
                                },
                            },
                            {
                                "left": {"type": "ref", "content": ["llm", "result"]},
                                "operator": "eq",
                                "right": {
                                    "type": "constant",
                                    "content": "'b'",
                                    "schema": {"type": "string", "extra": {"weak": true}},
                                },
                            },
                        ],
                        "logic": 1,
                        "branchId": "combined",
                    },
                    {"conditions": [], "branchId": "fallback"},
                ],
            },
        })
    );

    let edges = transform_value(nodes)["edges"].clone();
    assert_eq!(
        edges,
        json!([
            {"source_node_id": "branch", "target_node_id": "plugin", "source_port_id": "positive"},
            {"source_node_id": "branch", "target_node_id": "plugin", "source_port_id": "combined"},
            {"source_node_id": "branch", "target_node_id": "end", "source_port_id": "fallback"},
        ])
    );
}

#[test]
fn branch_expressions_map_every_operator() {
    let cases = [
        ("${a.b} eq 'x'", "eq", true),
        ("${a.b} not_eq 'x'", "neq", true),
        ("${a.b} contain 'x'", "contains", true),
        ("${a.b} not_contain 'x'", "not_contains", true),
        ("${a.b} longer_than 'x'", "gt", true),
        ("${a.b} longer_than_or_eq 'x'", "gte", true),
        ("${a.b} short_than 'x'", "lt", true),
        ("${a.b} short_than_or_eq 'x'", "lte", true),
        ("${a.b} is_empty", "is_empty", false),
        ("${a.b} is_not_empty", "is_not_empty", false),
    ];

    for (expression, operator, has_right) in cases {
        let mut entry = condition("branch-1", "end");
        entry.expression = Some(expression.to_string());
        let nodes = vec![
            with_conditions(node("branch", "Branch", "route"), vec![entry]),
            node("end", "End", "exit"),
        ];
        let data = node_data(nodes, "branch");
        let converted = &data["branches"][0]["conditions"][0];
        assert_eq!(converted["operator"], json!(operator), "{expression}");
        assert_eq!(
            converted["left"],
            json!({"type": "ref", "content": ["a", "b"]}),
            "{expression}"
        );
        if has_right {
            assert_eq!(
                converted["right"],
                json!({
                    "type": "constant",
                    "content": "'x'",
                    "schema": {"type": "string", "extra": {"weak": true}},
                }),
                "{expression}"
            );
        } else {
            assert!(converted.get("right").is_none(), "{expression}");
        }
    }
}

#[test]
fn branch_accepts_and_logic_and_empty_sides() {
    let mut entry = condition("combined", "end");
    entry.expressions = Some(vec![
        "${a.b} not_eq".to_string(),
        "${a.b} not_eq 'x'".to_string(),
    ]);
    entry.operator = Some("and".to_string());
    let nodes = vec![
        with_conditions(node("branch", "Branch", "route"), vec![entry]),
        node("end", "End", "exit"),
    ];

    let data = node_data(nodes, "branch");
    let branches = &data["branches"][0];
    assert_eq!(branches["logic"], json!(2));
    assert_eq!(branches["branchId"], json!("combined"));
    // A missing right side is omitted; the left side and operator stay.
    assert!(branches["conditions"][0].get("right").is_none());
    assert_eq!(
        branches["conditions"][0]["left"],
        json!({"type": "ref", "content": ["a", "b"]})
    );
    assert_eq!(branches["conditions"][1]["right"]["content"], json!("'x'"));

    // A single-token expression cannot be parsed.
    let mut single = condition("single", "end");
    single.expression = Some("${a.b}".to_string());
    let nodes = vec![
        with_conditions(node("branch", "Branch", "route"), vec![single]),
        node("end", "End", "exit"),
    ];
    let error = DlTransformer::new()
        .transform(&DlDocument { nodes })
        .expect_err("unparseable expression");
    assert!(
        error
            .to_string()
            .contains("cannot parse branch expression: ${a.b}"),
        "{error}"
    );

    // `or` maps to 1.
    let mut or_entry = condition("either", "end");
    or_entry.expressions = Some(vec!["${a.b} eq 'x'".to_string()]);
    or_entry.operator = Some("or".to_string());
    let nodes = vec![
        with_conditions(node("branch", "Branch", "route"), vec![or_entry]),
        node("end", "End", "exit"),
    ];
    assert_eq!(node_data(nodes, "branch")["branches"][0]["logic"], json!(1));
}

#[test]
fn branch_rejects_unknown_operator_logic_and_ambiguous_conditions() {
    let mut unknown_operator = condition("b", "end");
    unknown_operator.expression = Some("${a.b} weird 'x'".to_string());
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(node("branch", "Branch", "route"), vec![unknown_operator]),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("unknown operator");
    assert!(
        error.to_string().contains("unknown branch operator: weird"),
        "{error}"
    );

    let mut unknown_logic = condition("b", "end");
    unknown_logic.expressions = Some(vec!["${a.b} eq 'x'".to_string()]);
    unknown_logic.operator = Some("xor".to_string());
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(node("branch", "Branch", "route"), vec![unknown_logic]),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("unknown logic");
    assert!(
        error
            .to_string()
            .contains("condition b has unknown logic operator: xor"),
        "{error}"
    );

    let mut missing_logic = condition("b", "end");
    missing_logic.expressions = Some(vec!["${a.b} eq 'x'".to_string()]);
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(node("branch", "Branch", "route"), vec![missing_logic]),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("missing logic");
    assert!(
        error
            .to_string()
            .contains("unknown logic operator: <missing>"),
        "{error}"
    );

    let mut both = condition("b", "end");
    both.expression = Some("${a.b} eq 'x'".to_string());
    both.expressions = Some(vec!["${a.b} eq 'y'".to_string()]);
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(node("branch", "Branch", "route"), vec![both]),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("ambiguous condition");
    assert!(
        error
            .to_string()
            .contains("must have exactly one of expression or expressions"),
        "{error}"
    );

    let mut neither = condition("b", "end");
    neither.description = "no expressions".to_string();
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(node("branch", "Branch", "route"), vec![neither]),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("condition without expressions");
    assert!(
        error
            .to_string()
            .contains("must have exactly one of expression or expressions"),
        "{error}"
    );
}

#[test]
fn branch_requires_a_next_node() {
    let mut orphan = condition("b", "end");
    orphan.expression = Some("default".to_string());
    orphan.next = None;
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(node("branch", "Branch", "route"), vec![orphan]),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("condition without next");
    assert!(
        error.to_string().contains("condition b has no next node"),
        "{error}"
    );
}

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

/// An intent-detection document followed by one end node per branch.
fn intent_fixture() -> Vec<DlNode> {
    let mut greeting = condition("greet", "greet");
    greeting.expression = Some("${start.query} contain greeting".to_string());
    greeting.intent_id = Some("intent-greet".to_string());

    let mut question = condition("question", "question");
    question.expression = Some("${start.query} contain question".to_string());

    let mut fallback = condition("fallback", "end");
    fallback.expression = Some("default".to_string());

    vec![
        with_conditions(
            with_inputs(
                with_config(
                    node("intent", "IntentDetection", "classify"),
                    "prompt",
                    json!("Classify the query"),
                ),
                vec![input("query", Some("String"), json!("${start.query}"))],
            ),
            vec![greeting, question, fallback],
        ),
        node("greet", "End", "greet branch"),
        node("question", "End", "question branch"),
        node("end", "End", "fallback branch"),
    ]
}

#[test]
fn intent_detection_renders_intents_and_ports() {
    let nodes = intent_fixture();

    assert_eq!(
        node_json(nodes.clone(), "intent"),
        json!({
            "id": "intent",
            "type": "6",
            "meta": {"position": {"x": 0, "y": 0}},
            "data": {
                "title": "classify",
                "inputs": {
                    "input_parameters": {
                        "query": {
                            "type": "ref",
                            "content": ["start", "query"],
                            "extra": {"index": 0},
                        },
                    },
                    "llm_param": {
                        "systemPrompt": {"type": "template", "content": ""},
                        "prompt": {
                            "type": "template",
                            "content": "Classify the query",
                        },
                        "model": {
                            "id": "52",
                            "name": "siliconf-qwen3-8b",
                            "type": "Qwen/Qwen3-8B",
                        },
                    },
                    "intents": [
                        {"name": "greeting", "id": "intent-greet"},
                        {"name": "question", "id": "intent_intent_1"},
                    ],
                },
                "outputs": {
                    "type": "object",
                    "properties": {
                        "classification_id": {
                            "type": "integer",
                            "extra": {"index": 1},
                        },
                    },
                    "required": ["classification_id"],
                },
            },
        })
    );

    assert_eq!(
        edges_json(nodes),
        json!([
            {"source_node_id": "intent", "target_node_id": "greet", "source_port_id": "intent-greet"},
            {"source_node_id": "intent", "target_node_id": "question", "source_port_id": "intent_intent_1"},
            {"source_node_id": "intent", "target_node_id": "end", "source_port_id": "0"},
        ])
    );
}

#[test]
fn intent_detection_is_deterministic() {
    let first = transform_value(intent_fixture());
    let second = transform_value(intent_fixture());
    assert_eq!(first, second);
}

#[test]
fn intent_detection_without_inputs_leaves_query_empty() {
    let mut fallback = condition("fallback", "end");
    fallback.expression = Some("default".to_string());
    let nodes = vec![
        with_conditions(
            with_config(
                node("intent", "IntentDetection", "classify"),
                "prompt",
                json!("p"),
            ),
            vec![fallback],
        ),
        node("end", "End", "exit"),
    ];

    let data = node_data(nodes, "intent");
    assert_eq!(data["inputs"]["input_parameters"], json!({}));
    assert_eq!(data["inputs"]["intents"], json!([]));
    assert_eq!(
        data["outputs"]["properties"]["classification_id"],
        json!({"type": "integer", "extra": {"index": 0}})
    );
}

#[test]
fn intent_detection_rejects_missing_or_malformed_conditions() {
    let missing_prompt = || {
        let mut fallback = condition("fallback", "end");
        fallback.expression = Some("default".to_string());
        fallback.next = None;
        DlDocument {
            nodes: vec![with_conditions(
                node("intent", "IntentDetection", "classify"),
                vec![fallback],
            )],
        }
    };
    let error = DlTransformer::new()
        .transform(&missing_prompt())
        .expect_err("missing prompt");
    assert!(
        error
            .to_string()
            .contains("node intent: configs.prompt is required"),
        "{error}"
    );

    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![with_config(
                node("intent", "IntentDetection", "classify"),
                "prompt",
                json!("p"),
            )],
        })
        .expect_err("missing conditions");
    assert!(
        error.to_string().contains("conditions are required"),
        "{error}"
    );

    let mut no_expression = condition("fallback", "end");
    no_expression.next = None;
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![with_conditions(
                with_config(
                    node("intent", "IntentDetection", "classify"),
                    "prompt",
                    json!("p"),
                ),
                vec![no_expression],
            )],
        })
        .expect_err("missing expression");
    assert!(
        error
            .to_string()
            .contains("condition fallback has no expression"),
        "{error}"
    );

    let mut malformed = condition("greet", "end");
    malformed.expression = Some("${start.query} eq greeting".to_string());
    let error = DlTransformer::new()
        .transform(&DlDocument {
            nodes: vec![
                with_conditions(
                    with_config(
                        node("intent", "IntentDetection", "classify"),
                        "prompt",
                        json!("p"),
                    ),
                    vec![malformed],
                ),
                node("end", "End", "exit"),
            ],
        })
        .expect_err("not a contain expression");
    assert!(
        error
            .to_string()
            .contains("is not a \" contain \" expression"),
        "{error}"
    );
}

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
