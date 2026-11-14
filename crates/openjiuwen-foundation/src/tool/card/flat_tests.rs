//! Flat/nested wire-conversion tests for [`ToolInfo`](super::ToolInfo).
//!
//! These live beside `src/tool/card.rs` (declared as `mod flat_tests;` there) so
//! the module file stays under the workspace file-size baseline. They cover
//! Python's flat `ToolInfo` contract, the nested `OpenAI` shape, the
//! `from_wire` auto-detection, the parameter normalization rules, and the batch
//! helpers — the serde-shape tests stay in `card.rs` itself.

use openjiuwen_common::error::ErrorKind;
use serde_json::json;

use super::{flat_tools, from_wire_tools, ToolCard, ToolInfo};

/// The exact literal Python's `ToolInfo` serializes to.
fn python_flat_literal() -> serde_json::Value {
    json!({
        "type": "function",
        "name": "calc",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    })
}

#[test]
fn nested_flat_nested_round_trip_is_identity() {
    let nested = ToolInfo::new("calc", "does math", json!({"type": "object"}));

    let flat = nested.flat();
    assert_eq!(
        flat,
        json!({
            "type": "function",
            "name": "calc",
            "description": "does math",
            "parameters": {"type": "object"},
        })
    );

    let parsed = ToolInfo::from_flat(flat).unwrap();
    assert_eq!(parsed, nested);
    assert_eq!(parsed.openai(), nested.openai());
    assert_eq!(
        serde_json::to_value(&parsed).unwrap(),
        nested.openai(),
        "openai() must stay the canonical serde serialization"
    );
}

#[test]
fn from_flat_parses_the_python_literal() {
    let literal = python_flat_literal();
    let info = ToolInfo::from_flat(literal.clone()).unwrap();

    assert_eq!(info.kind, "function");
    assert_eq!(info.function.name, "calc");
    assert_eq!(info.function.description, "d");
    assert_eq!(
        info.function.parameters,
        json!({"type": "object", "properties": {}})
    );
    assert_eq!(
        info.flat(),
        literal,
        "flat -> ToolInfo -> flat must be the identity for the Python literal"
    );
    assert_eq!(
        info.openai(),
        json!({
            "type": "function",
            "function": {
                "name": "calc",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        })
    );
}

#[test]
fn from_wire_accepts_both_shapes() {
    let flat = python_flat_literal();
    let from_flat = ToolInfo::from_wire(flat).unwrap();
    assert_eq!(from_flat.function.name, "calc");

    let nested = json!({
        "type": "function",
        "function": {
            "name": "calc",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        },
    });
    let from_nested = ToolInfo::from_wire(nested.clone()).unwrap();
    assert_eq!(from_nested, from_flat);
    assert_eq!(from_nested.openai(), nested);
}

#[test]
fn from_wire_rejects_non_objects_and_ambiguous_objects() {
    for value in [
        json!("calc"),
        json!(null),
        json!([{"type": "function", "name": "calc"}]),
        json!(42),
    ] {
        let error = ToolInfo::from_wire(value.clone()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "value: {value}");
        assert!(
            error.to_string().contains("must be a JSON object"),
            "unexpected message: {error}"
        );
    }

    // Neither a nested `function` object nor a top-level `name`.
    let error = ToolInfo::from_wire(json!({"type": "function"})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("neither shape"),
        "unexpected message: {error}"
    );
}

#[test]
fn from_flat_defaults_missing_fields() {
    let info = ToolInfo::from_flat(json!({"name": "calc", "unexpected": {"x": 1}})).unwrap();
    assert_eq!(info.kind, "function", "a missing `type` defaults");
    assert_eq!(info.function.description, "");
    assert_eq!(
        info.function.parameters,
        json!({}),
        "missing parameters -> {{}}"
    );

    let bare = ToolInfo::from_flat(json!({})).unwrap();
    assert_eq!(bare.kind, "function");
    assert_eq!(bare.function.name, "");
    assert_eq!(bare.function.description, "");
    assert_eq!(
        bare.function.parameters,
        json!({}),
        "from_flat normalizes the default `null` parameters to {{}}"
    );
    assert_eq!(
        bare,
        ToolInfo::new("", "", json!({})),
        "an empty object is the default info modulo the parameter normalization"
    );
}

#[test]
fn flat_normalizes_null_and_non_object_parameters() {
    for info in [
        ToolInfo::new("calc", "d", json!(null)),
        ToolInfo::new("calc", "d", json!("not a schema")),
        ToolInfo::new("calc", "d", json!([1, 2])),
    ] {
        assert_eq!(info.flat()["parameters"], json!({}));
        // The nested wire shape stays a faithful serialization.
        assert_eq!(
            info.openai()["function"]["parameters"],
            info.function.parameters
        );
    }

    let missing = ToolInfo::from_flat(json!({"name": "calc", "parameters": null})).unwrap();
    assert_eq!(missing.function.parameters, json!({}));
}

#[test]
fn from_flat_rejects_non_object_parameters() {
    for parameters in [json!("x"), json!(3), json!(true), json!([])] {
        let error =
            ToolInfo::from_flat(json!({"name": "calc", "parameters": parameters})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("`parameters`"),
            "the error must name the field: {error}"
        );
    }
}

#[test]
fn from_openai_requires_a_nested_function() {
    let error = ToolInfo::from_openai(json!({"type": "function", "name": "calc"})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("no `function` object"), "{message}");
    assert!(message.contains("from_flat"), "{message}");
    assert!(message.contains("from_wire"), "{message}");

    let error = ToolInfo::from_openai(json!([1, 2])).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn from_openai_defaults_the_type_and_drops_unknown_fields() {
    let info = ToolInfo::from_openai(json!({
        "function": {"name": "calc", "description": "d", "parameters": {"type": "object"}},
        "metadata": {"ignored": true},
    }))
    .unwrap();
    assert_eq!(info.kind, "function", "a missing `type` defaults");
    assert_eq!(info.function.name, "calc");

    let empty_kind = ToolInfo::from_openai(json!({
        "type": "",
        "function": {"name": "calc", "description": "d", "parameters": {}},
    }))
    .unwrap();
    assert_eq!(empty_kind.kind, "function", "an empty `type` normalizes");
}

#[test]
fn batch_helpers_project_and_parse_both_shapes() {
    let infos = vec![
        ToolInfo::new("a", "first", json!({"type": "object"})),
        ToolInfo::new("b", "second", json!({})),
    ];

    let flat = flat_tools(&infos);
    assert_eq!(flat, json!([infos[0].flat(), infos[1].flat()]));
    assert_eq!(flat[0]["name"], json!("a"));
    assert_eq!(flat_tools(&[]), json!([]));

    let parsed = from_wire_tools(flat).unwrap();
    assert_eq!(parsed, infos);
}

#[test]
fn from_wire_tools_mixes_shapes_and_tolerates_unknown_fields() {
    let mixed = json!([
        {"type": "function", "name": "flat", "description": "f", "parameters": {}, "extra": 1},
        {
            "type": "function",
            "function": {"name": "nested", "description": "n", "parameters": {}},
            "extra": {"deep": true},
        },
    ]);

    let parsed = from_wire_tools(mixed).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].function.name, "flat");
    assert_eq!(parsed[0].function.description, "f");
    assert_eq!(parsed[1].function.name, "nested");
}

#[test]
fn from_wire_tools_rejects_non_arrays_and_names_the_index() {
    let error = from_wire_tools(json!({"name": "calc"})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("must be a JSON array"),
        "unexpected message: {error}"
    );

    let error = from_wire_tools(json!([{"name": "ok"}, ["not a tool"]])).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("index 1"),
        "the error must name the element: {error}"
    );
}

#[test]
fn card_flat_tool_info_is_the_flat_projection_and_openai_tool_stays_nested() {
    let card = ToolCard::new("calc", "does math", json!({"type": "object"}));

    assert_eq!(card.flat_tool_info(), card.tool_info().flat());
    assert_eq!(
        card.flat_tool_info(),
        json!({
            "type": "function",
            "name": "calc",
            "description": "does math",
            "parameters": {"type": "object"},
        })
    );
    assert_eq!(
        card.openai_tool(),
        card.tool_info().openai(),
        "openai_tool must stay byte-identical to the nested projection"
    );
    assert_eq!(
        card.openai_tool(),
        serde_json::to_value(card.tool_info()).unwrap()
    );

    // An empty input schema is normalized on both projections.
    let empty = ToolCard::new("calc", "", json!({}));
    assert_eq!(
        empty.flat_tool_info()["parameters"],
        json!({"type": "object"})
    );
}
