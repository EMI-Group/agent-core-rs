use super::super::*;
use super::support::*;
use serde_json::json;

/// The expected uuid4 shape: five hyphen-separated groups of 8-4-4-4-12
/// lowercase hex digits, with the version and variant nibbles of a v4 id.
fn is_uuid4(id: &str) -> bool {
    let groups: Vec<&str> = id.split('-').collect();
    let lengths = [8, 4, 4, 4, 12];
    if groups.len() != lengths.len() || id.len() != 36 {
        return false;
    }
    let hex = |group: &str| {
        group
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
    };
    let shapes = groups
        .iter()
        .zip(lengths)
        .all(|(group, length)| group.len() == length && hex(group));
    let version = groups[2].starts_with('4');
    let variant = matches!(groups[3].chars().next(), Some('8' | '9' | 'a' | 'b'));
    shapes && version && variant
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

/// An intent-detection document whose conditions carry no `intent_id`.
fn generated_fixture() -> Vec<DlNode> {
    let mut greet = condition("greet", "greet");
    greet.expression = Some("${start.query} contain greeting".to_string());

    let mut question = condition("question", "question");
    question.expression = Some("${start.query} contain question".to_string());

    let mut farewell = condition("farewell", "farewell");
    farewell.expression = Some("${start.query} contain farewell".to_string());

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
            vec![greet, question, farewell, fallback],
        ),
        node("greet", "End", "greet branch"),
        node("question", "End", "question branch"),
        node("farewell", "End", "farewell branch"),
        node("end", "End", "fallback branch"),
    ]
}

/// The platform node with id `id` inside one transformed document.
fn node_of(value: &Value, id: &str) -> Value {
    value["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .find(|node| node["id"] == json!(id))
        .unwrap_or_else(|| panic!("node {id} is present"))
        .clone()
}

/// The `(name, id)` pairs of the rendered intent list.
fn intents_of(value: &Value) -> Vec<(String, String)> {
    node_of(value, "intent")["data"]["inputs"]["intents"]
        .as_array()
        .expect("intent array")
        .iter()
        .map(|intent| {
            (
                intent["name"].as_str().expect("intent name").to_string(),
                intent["id"].as_str().expect("intent id").to_string(),
            )
        })
        .collect()
}

/// The `source_port_id` of every edge, in edge order.
fn ports_of(value: &Value) -> Vec<Option<String>> {
    value["edges"]
        .as_array()
        .expect("edge array")
        .iter()
        .map(|edge| edge["source_port_id"].as_str().map(str::to_string))
        .collect()
}

/// Replaces every generated intent id with `intent-<position>`, in list order.
fn normalise_intent_ids(mut value: Value) -> Value {
    let mut mapping: Vec<(String, String)> = Vec::new();
    if let Some(nodes) = value["nodes"].as_array_mut() {
        for node in nodes.iter_mut() {
            let intents = &mut node["data"]["inputs"]["intents"];
            if let Some(intents) = intents.as_array_mut() {
                for (position, intent) in intents.iter_mut().enumerate() {
                    let generated = intent["id"].as_str().unwrap_or_default().to_string();
                    let placeholder = format!("intent-{position}");
                    intent["id"] = json!(placeholder.clone());
                    mapping.push((generated, placeholder));
                }
            }
        }
    }
    if let Some(edges) = value["edges"].as_array_mut() {
        for edge in edges.iter_mut() {
            let Some(port) = edge["source_port_id"].as_str() else {
                continue;
            };
            let found = mapping
                .iter()
                .find(|(generated, _)| generated == port)
                .map(|(_, placeholder)| placeholder.clone());
            if let Some(placeholder) = found {
                edge["source_port_id"] = json!(placeholder);
            }
        }
    }
    value
}

#[test]
fn intent_detection_renders_intents_and_ports() {
    let value = transform_value(intent_fixture());
    let intents = intents_of(&value);
    assert_eq!(intents[0].0, "greeting");
    assert_eq!(intents[0].1, "intent-greet");
    assert_eq!(intents[1].0, "question");
    assert!(is_uuid4(&intents[1].1), "{intents:?}");
    let generated = intents[1].1.clone();

    assert_eq!(
        node_of(&value, "intent"),
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
                        {"name": "question", "id": generated},
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

    let ports = ports_of(&value);
    assert_eq!(ports.len(), 3);
    assert_eq!(ports[0].as_deref(), Some("intent-greet"));
    assert_eq!(ports[1].as_deref(), Some(generated.as_str()));
    assert_eq!(ports[2].as_deref(), Some("0"));
}

#[test]
fn intent_detection_generates_distinct_uuid4_ids() {
    let value = transform_value(generated_fixture());
    let intents = intents_of(&value);

    assert_eq!(
        intents
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["greeting", "question", "farewell"]
    );
    for (_, id) in &intents {
        assert!(is_uuid4(id), "{id}");
    }
    let ids: Vec<&str> = intents.iter().map(|(_, id)| id.as_str()).collect();
    for (position, id) in ids.iter().enumerate() {
        assert!(
            !ids[position + 1..].contains(id),
            "intent ids are unique: {ids:?}"
        );
    }

    let ports = ports_of(&value);
    assert_eq!(ports[0].as_deref(), Some(ids[0]));
    assert_eq!(ports[1].as_deref(), Some(ids[1]));
    assert_eq!(ports[2].as_deref(), Some(ids[2]));
    assert_eq!(ports[3].as_deref(), Some("0"));
}

#[test]
fn intent_generation_only_changes_ids_between_conversions() {
    let first = transform_value(generated_fixture());
    let second = transform_value(generated_fixture());

    fn ids(value: &Value) -> Vec<String> {
        value["nodes"][0]["data"]["inputs"]["intents"]
            .as_array()
            .expect("intents array")
            .iter()
            .map(|intent| intent["id"].as_str().expect("intent id").to_string())
            .collect()
    }

    let first_ids = ids(&first);
    let second_ids = ids(&second);
    assert_eq!(first_ids.len(), 3);
    assert_eq!(second_ids.len(), 3);
    for id in &first_ids {
        assert!(
            !second_ids.contains(id),
            "two conversions generate different ids: {first_ids:?} vs {second_ids:?}"
        );
    }

    assert_eq!(
        normalise_intent_ids(first),
        normalise_intent_ids(second.clone())
    );
    assert_eq!(
        normalise_intent_ids(second)["edges"],
        json!([
            {"source_node_id": "intent", "target_node_id": "greet", "source_port_id": "intent-0"},
            {"source_node_id": "intent", "target_node_id": "question", "source_port_id": "intent-1"},
            {"source_node_id": "intent", "target_node_id": "farewell", "source_port_id": "intent-2"},
            {"source_node_id": "intent", "target_node_id": "end", "source_port_id": "0"},
        ])
    );
}

#[test]
fn intent_detection_with_explicit_ids_is_deterministic() {
    let mut fixture = intent_fixture();
    fixture[0]
        .parameters
        .conditions
        .get_mut(1)
        .expect("question condition")
        .intent_id = Some("intent-question".to_string());

    let first = transform_value(fixture.clone());
    assert_eq!(first, transform_value(fixture));
    assert_eq!(
        first["nodes"][0]["data"]["inputs"]["intents"],
        json!([
            {"name": "greeting", "id": "intent-greet"},
            {"name": "question", "id": "intent-question"},
        ])
    );
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
