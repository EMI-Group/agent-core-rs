use super::super::*;
use super::support::*;
use serde_json::json;

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
