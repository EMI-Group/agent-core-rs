#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Card and wire-shape tests for `AgentCard`, `AgentOutput`, and `AgentChunk`.
//!
//! These pin the Python-compatible JSON surface of the crate's data types: the
//! six `AgentCard` fields (with `null` for the unset optional params), the
//! `result_type`/`state`/`interrupt_ids` keys of an `AgentOutput` together with
//! the absence of `None`/empty keys, and the `StreamItem` bridge of
//! `AgentChunk`. Everything is pure serialization work, so the suite is offline
//! and deterministic.

use openjiuwen_agent::output::{ANSWER, INTERACTION, LLM_OUTPUT};
use openjiuwen_agent::{AgentCard, AgentChunk, AgentOutput, OutputFrame, ResultType};
use openjiuwen_common::card::Card;
use openjiuwen_session::StreamMode;
use serde_json::{json, Value};

/// The exact Python field set of an `AgentCard`, sorted.
const CARD_FIELDS: [&str; 6] = [
    "description",
    "id",
    "input_params",
    "interface_url",
    "name",
    "output_params",
];

/// Keys that a plain `AgentOutput::answer` must not emit, sorted.
const ABSENT_FOR_ANSWER: [&str; 12] = [
    "component_ids",
    "data",
    "extra",
    "finalization_error",
    "finish_reason",
    "interrupt_ids",
    "iterations_used",
    "observations",
    "state",
    "termination_reason",
    "truncated_by_max_iterations",
    "workflow_execution_state",
];

/// The keys of the JSON object `value`, sorted.
///
/// # Panics
///
/// Panics when `value` is not a JSON object.
fn sorted_keys(value: &Value) -> Vec<String> {
    let object = value.as_object().expect("the value is a JSON object");
    let mut keys: Vec<String> = object.keys().cloned().collect();
    keys.sort_unstable();
    keys
}

// --- AgentCard -------------------------------------------------------------

#[test]
fn card_json_uses_the_python_field_set_with_null_optionals() {
    let value = serde_json::to_value(AgentCard::new("assistant")).unwrap();

    assert_eq!(sorted_keys(&value), CARD_FIELDS.map(str::to_owned));
    assert_eq!(value["name"], "assistant");
    assert_eq!(value["description"], "");
    assert_eq!(value["input_params"], Value::Null);
    assert_eq!(value["output_params"], Value::Null);
    assert_eq!(value["interface_url"], Value::Null);
}

#[test]
fn card_drops_unknown_keys_and_round_trips() {
    let raw = json!({
        "id": "0123456789abcdef0123456789abcdef",
        "name": "assistant",
        "description": "helps",
        "input_params": {"type": "object"},
        "output_params": {"type": "string"},
        "interface_url": "http://localhost/agent",
        "surprise": {"nested": true},
        "extra_flag": true,
    });

    let card: AgentCard = serde_json::from_value(raw).unwrap();
    assert_eq!(card.id, "0123456789abcdef0123456789abcdef");
    assert_eq!(card.name, "assistant");
    assert_eq!(card.description, "helps");
    assert_eq!(card.input_params, Some(json!({"type": "object"})));
    assert_eq!(card.output_params, Some(json!({"type": "string"})));
    assert_eq!(
        card.interface_url.as_deref(),
        Some("http://localhost/agent")
    );

    let value = serde_json::to_value(&card).unwrap();
    assert_eq!(sorted_keys(&value), CARD_FIELDS.map(str::to_owned));
    assert!(value.get("surprise").is_none(), "{value}");
    assert!(value.get("extra_flag").is_none(), "{value}");

    let restored: AgentCard = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored, card);
    assert_eq!(serde_json::to_value(&restored).unwrap(), value);
}

#[test]
fn card_trait_exposes_id_name_and_optional_description() {
    let anonymous = AgentCard::new("worker");
    assert_eq!(Card::id(&anonymous), anonymous.id.as_str());
    assert_eq!(Card::name(&anonymous), "worker");
    assert_eq!(Card::description(&anonymous), None);

    let described = AgentCard::new("worker").with_description("does work");
    assert_eq!(Card::id(&described), described.id.as_str());
    assert_eq!(Card::name(&described), "worker");
    assert_eq!(Card::description(&described), Some("does work"));
}

#[test]
fn card_new_generates_a_32_char_lowercase_hex_id() {
    let card = AgentCard::new("assistant");
    assert_eq!(card.id.len(), 32);
    assert!(
        card.id
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "not a lowercase hex id: {}",
        card.id
    );
    assert_ne!(AgentCard::new("assistant").id, card.id);
    assert_eq!(card.name, "assistant");
    assert_eq!(card.description, "");
    assert!(card.input_params.is_none());
    assert!(card.output_params.is_none());
    assert!(card.interface_url.is_none());
}

#[test]
fn card_builders_set_every_field() {
    let card = AgentCard::new("initial")
        .with_id("id-1")
        .with_name("renamed")
        .with_description("desc")
        .with_input_params(json!({"type": "object"}))
        .with_output_params(json!({"type": "string"}))
        .with_interface_url("http://example.test/agent");

    assert_eq!(card.id, "id-1");
    assert_eq!(card.name, "renamed");
    assert_eq!(card.description, "desc");
    assert_eq!(card.input_params, Some(json!({"type": "object"})));
    assert_eq!(card.output_params, Some(json!({"type": "string"})));
    assert_eq!(
        card.interface_url.as_deref(),
        Some("http://example.test/agent")
    );

    let value = serde_json::to_value(&card).unwrap();
    assert_eq!(value["id"], "id-1");
    assert_eq!(value["name"], "renamed");
    assert_eq!(value["description"], "desc");
    assert_eq!(value["input_params"], json!({"type": "object"}));
    assert_eq!(value["output_params"], json!({"type": "string"}));
    assert_eq!(value["interface_url"], "http://example.test/agent");
}

// --- AgentOutput -----------------------------------------------------------

#[test]
fn output_result_types_use_python_wire_strings() {
    assert_eq!(AgentOutput::answer("a").to_value()["result_type"], "answer");
    assert_eq!(AgentOutput::error("e").to_value()["result_type"], "error");
    let interrupt = AgentOutput::interrupt(
        vec![OutputFrame::interaction(0, "intr_1", json!({}))],
        vec!["intr_1".to_owned()],
    );
    assert_eq!(interrupt.to_value()["result_type"], "interrupt");

    assert_eq!(serde_json::to_value(ResultType::Answer).unwrap(), "answer");
    assert_eq!(serde_json::to_value(ResultType::Error).unwrap(), "error");
    assert_eq!(
        serde_json::to_value(ResultType::Interrupt).unwrap(),
        "interrupt"
    );
}

#[test]
fn interrupt_output_serializes_state_frames_and_interrupt_ids() {
    let frames = vec![
        OutputFrame::interaction(0, "intr_a", json!({"approved": true})),
        OutputFrame::interaction(1, "intr_b", json!({"value": "asked"})),
    ];
    let ids = vec!["intr_a".to_owned(), "intr_b".to_owned()];
    let output = AgentOutput::interrupt(frames, ids.clone());

    let value = output.to_value();
    assert_eq!(value["result_type"], "interrupt");
    assert_eq!(value["output"].as_str().unwrap(), "");
    assert_eq!(value["interrupt_ids"], json!(ids));

    let state = value["state"].as_array().expect("state is an array");
    assert_eq!(state.len(), 2);
    assert_eq!(
        sorted_keys(&state[0]),
        ["index", "payload", "type"].map(str::to_owned)
    );
    assert_eq!(state[0]["type"], INTERACTION);
    assert_eq!(state[0]["index"], 0);
    assert_eq!(state[0]["payload"]["id"], "intr_a");
    assert_eq!(state[0]["payload"]["value"], json!({"approved": true}));
    assert_eq!(state[1]["type"], INTERACTION);
    assert_eq!(state[1]["index"], 1);
    assert_eq!(state[1]["payload"]["id"], "intr_b");
    assert_eq!(state[1]["payload"]["value"], json!({"value": "asked"}));

    let restored = AgentOutput::from_value(value).unwrap();
    assert_eq!(restored, output);
}

#[test]
fn none_and_empty_output_fields_are_absent_from_the_wire_object() {
    let value = AgentOutput::answer("hi").to_value();
    assert_eq!(value["output"], "hi");
    assert_eq!(value["result_type"], "answer");
    for key in ABSENT_FOR_ANSWER {
        assert!(value.get(key).is_none(), "{key} must be absent: {value}");
    }

    let error = AgentOutput::error("boom").to_value();
    assert_eq!(error["output"], "boom");
    assert_eq!(error["result_type"], "error");
    for key in ABSENT_FOR_ANSWER {
        assert!(error.get(key).is_none(), "{key} must be absent: {error}");
    }
}

/// Python only adds `truncated_by_max_iterations` to the result dict when the
/// run was cut off (`react_agent.py::_finish_max_iterations`), never as `false`,
/// so the answer projection omits the key.
#[test]
fn truncated_by_max_iterations_is_absent_unless_truncated() {
    let untruncated = AgentOutput::answer("hi").to_value();
    assert!(
        untruncated.get("truncated_by_max_iterations").is_none(),
        "{untruncated}"
    );

    let truncated = AgentOutput {
        truncated_by_max_iterations: true,
        ..AgentOutput::answer("hi")
    };
    assert_eq!(
        truncated.to_value()["truncated_by_max_iterations"],
        json!(true)
    );
}

#[test]
fn is_interrupt_is_true_only_for_interrupt_outputs() {
    assert!(!AgentOutput::answer("a").is_interrupt());
    assert!(!AgentOutput::error("e").is_interrupt());
    assert!(AgentOutput::interrupt(Vec::new(), Vec::new()).is_interrupt());
    assert!(AgentOutput::from_value(json!({"result_type": "interrupt"}))
        .unwrap()
        .is_interrupt());
}

#[test]
fn python_shaped_answer_dict_deserializes_with_defaults() {
    let output = AgentOutput::from_value(json!({"output": "x", "result_type": "answer"})).unwrap();

    assert_eq!(output.output, "x");
    assert_eq!(output.result_type, ResultType::Answer);
    assert!(!output.is_interrupt());
    assert!(!output.reasoning_present);
    assert!(!output.truncated_by_max_iterations);
    assert!(output.finish_reason.is_none());
    assert!(output.termination_reason.is_none());
    assert!(output.iterations_used.is_none());
    assert_eq!(output.observations, Vec::new());
    assert!(output.finalization_error.is_none());
    assert_eq!(output.state, Vec::new());
    assert_eq!(output.interrupt_ids, Vec::<String>::new());
    assert_eq!(output.component_ids, Vec::<String>::new());
    assert!(output.workflow_output.is_none());
    assert!(output.data.is_none());
    assert!(output.extra.is_empty());
}

// --- AgentChunk ------------------------------------------------------------

#[test]
fn chunk_kind_constants_match_python_wire_names() {
    assert_eq!(LLM_OUTPUT, "llm_output");
    assert_eq!(ANSWER, "answer");
    assert_eq!(INTERACTION, "__interaction__");

    assert_eq!(AgentChunk::llm_output(0, "x").kind, LLM_OUTPUT);
    assert_eq!(AgentChunk::answer(&AgentOutput::answer("x")).kind, ANSWER);
    assert_eq!(
        AgentChunk::interaction("intr_1", Value::Null).kind,
        INTERACTION
    );
}

#[test]
fn answer_and_interaction_chunks_carry_python_payloads() {
    let output = AgentOutput::answer("done");
    let answer = AgentChunk::answer(&output);
    assert_eq!(answer.index, 0);
    assert_eq!(answer.payload, output.to_value());
    assert_eq!(answer.payload["result_type"], "answer");

    let interaction = AgentChunk::interaction("intr_1", json!({"approved": true}));
    assert_eq!(interaction.index, 0);
    assert_eq!(
        interaction.payload,
        json!({"id": "intr_1", "value": {"approved": true}})
    );
}

#[test]
fn chunk_to_stream_item_keeps_kind_index_and_payload_in_output_mode() {
    let chunk = AgentChunk::new(LLM_OUTPUT, 7, json!({"content": "tok"}));

    let item = chunk.to_stream_item();

    assert_eq!(item.item_type, chunk.kind);
    assert_eq!(item.index, Some(7));
    assert_eq!(item.payload, chunk.payload);
    assert_eq!(item.mode, StreamMode::Output);
    assert_eq!(item.mode.as_str(), "output");
    assert_eq!(item.to_wire(), chunk.to_value());
}

#[test]
fn chunk_from_stream_item_round_trips_llm_output_answer_and_interaction() {
    let chunks = [
        AgentChunk::new(LLM_OUTPUT, 2, json!({"content": "hello"})),
        AgentChunk::answer(&AgentOutput::answer("done")),
        AgentChunk::interaction("intr_1", json!({"approved": true})),
    ];

    for chunk in chunks {
        let item = chunk.to_stream_item();

        assert_eq!(item.item_type, chunk.kind);
        assert_eq!(item.mode, StreamMode::Output);
        assert_eq!(item.index, Some(chunk.index));
        assert_eq!(item.payload, chunk.payload);
        assert_eq!(AgentChunk::from_stream_item(&item), chunk);
        assert_eq!(AgentChunk::from_value(chunk.to_value()).unwrap(), chunk);
    }
}
