use super::super::*;
use super::support::*;
use serde_json::json;

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
