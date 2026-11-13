#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for Python-literal parsing and Python-style rendering.
//!
//! These cases mirror the two call sites the module exists for: recovering a
//! handoff payload from a tool message that was routed through `str()`
//! (`json.loads` first, `ast.literal_eval` second) and rendering a tool result
//! for a model prompt. Everything is pure string work, so the suite is offline
//! and deterministic.
use openjiuwen_multiagent::pyrepr::{
    literal_or_json, parse_python_literal, render_python_value, MAX_LITERAL_DEPTH,
};
use serde_json::json;

#[test]
fn recovers_a_handoff_payload_from_python_repr() {
    let content = "{'__handoff_to__': 'reviewer', '__handoff_message__': \"looks good\", \
                   '__handoff_reason__': None, 'meta': {'count': 2, 'tags': ['a', 'b']}}";
    let parsed = literal_or_json(content).expect("python repr parses");
    assert_eq!(parsed["__handoff_to__"], json!("reviewer"));
    assert_eq!(parsed["__handoff_message__"], json!("looks good"));
    assert_eq!(parsed["__handoff_reason__"], json!(null));
    assert_eq!(parsed["meta"]["tags"], json!(["a", "b"]));
}

#[test]
fn prefers_json_when_the_payload_is_valid_json() {
    assert_eq!(
        literal_or_json(r#"{"__handoff_to__": "reviewer"}"#),
        Some(json!({"__handoff_to__": "reviewer"}))
    );
    // JSON is accepted unchanged, including escapes Python would spell differently.
    assert_eq!(literal_or_json(r#""a\u0041""#), Some(json!("aA")));
}

#[test]
fn non_literal_text_yields_none() {
    assert_eq!(
        literal_or_json("I have handed the task to the reviewer."),
        None
    );
    assert_eq!(literal_or_json(""), None);
    assert_eq!(parse_python_literal("handoff -> reviewer"), None);
    assert_eq!(parse_python_literal("['a', 'b'] trailing"), None);
}

#[test]
fn renders_tool_results_the_way_python_str_does() {
    let result = json!({
        "status": "ok",
        "count": 3,
        "ratio": 0.5,
        "done": true,
        "error": null,
        "items": ["first", "second"]
    });
    let rendered = render_python_value(&result);
    assert_eq!(
        rendered,
        "{'status': 'ok', 'count': 3, 'ratio': 0.5, 'done': True, 'error': None, \
         'items': ['first', 'second']}"
    );
    assert_eq!(render_python_value(&json!("plain text")), "plain text");
    assert_eq!(
        render_python_value(&json!({"quote": "it's"})),
        "{'quote': 'it\\'s'}"
    );
}

#[test]
fn rendering_then_parsing_round_trips_nested_values() {
    let value = json!({
        "nested": {"list": [1, 2.5, false, null, "x'y"], "empty": {}, "none": []},
        "unicode": "中文",
        "escaped": "a\\b\nc\td"
    });
    let rendered = render_python_value(&value);
    assert_eq!(parse_python_literal(&rendered), Some(value));
}

#[test]
fn tuples_and_sets_become_arrays() {
    assert_eq!(parse_python_literal("(1, 2, 3)"), Some(json!([1, 2, 3])));
    assert_eq!(parse_python_literal("(1,)"), Some(json!([1])));
    assert_eq!(parse_python_literal("(1)"), Some(json!(1)));
    assert_eq!(parse_python_literal("{'a', 'b'}"), Some(json!(["a", "b"])));
    assert_eq!(parse_python_literal("{}"), Some(json!({})));
}

#[test]
fn numbers_keep_their_kind() {
    assert_eq!(parse_python_literal("-0"), Some(json!(0)));
    assert_eq!(parse_python_literal("1_000"), Some(json!(1000)));
    assert_eq!(parse_python_literal("1.0"), Some(json!(1.0)));
    assert_eq!(parse_python_literal("2.5e-3"), Some(json!(0.0025)));
    assert_eq!(parse_python_literal("inf"), None);
    assert_eq!(parse_python_literal("nan"), None);
    assert_eq!(parse_python_literal("1e999"), None);
}

#[test]
fn depth_limit_is_enforced_at_the_boundary() {
    fn at_depth(levels: usize) -> String {
        format!("{}'x'{}", "[".repeat(levels), "]".repeat(levels))
    }
    fn nested(levels: usize) -> serde_json::Value {
        let mut value = json!("x");
        for _ in 0..levels {
            value = json!([value]);
        }
        value
    }
    assert_eq!(
        parse_python_literal(&at_depth(MAX_LITERAL_DEPTH)),
        Some(nested(MAX_LITERAL_DEPTH))
    );
    assert_eq!(parse_python_literal(&at_depth(MAX_LITERAL_DEPTH + 1)), None);
}

#[test]
fn non_string_scalars_survive_a_render_parse_cycle() {
    for value in [
        json!(null),
        json!(true),
        json!(false),
        json!(0),
        json!(-17),
        json!(1.25),
    ] {
        let rendered = render_python_value(&value);
        assert_eq!(
            parse_python_literal(&rendered),
            Some(value),
            "rendered {rendered}"
        );
    }
}

#[test]
fn a_top_level_string_is_bare_and_not_a_literal() {
    // Python's `str("py")` is `py`, which is not a literal: only the bare form
    // matches Python, so a top-level string is deliberately not round-trippable.
    assert_eq!(render_python_value(&json!("py")), "py");
    assert_eq!(parse_python_literal("py"), None);
    assert_eq!(render_python_value(&json!(["py"])), "['py']");
    assert_eq!(parse_python_literal("['py']"), Some(json!(["py"])));
}
