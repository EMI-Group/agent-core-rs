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
