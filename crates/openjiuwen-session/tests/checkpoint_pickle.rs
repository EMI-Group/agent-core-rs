//! Behavior tests for the Python-pickle checkpoint interop.
//!
//! Ports the payload shape of `openjiuwen/core/session/checkpointer/persistence.py`
//! (v0.1.20), which persists checkpoints as Python pickles, so a Python-written
//! checkpoint can be mapped onto the Rust `Checkpoint` model and back.
//!
//! Every fixture below is a hand-written byte literal whose comment lists the
//! opcodes it contains; the expected streams were checked once against `CPython`
//! while the fixtures were written, and nothing here shells out to Python.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::checkpoint::pickle::{self, BYTES_KEY};
use openjiuwen_session::checkpoint::{checkpoint_from_pickle, checkpoint_to_pickle};
use openjiuwen_session::{Checkpoint, CheckpointNamespace};
use serde_json::{json, Value};

/// Decodes a hex fixture written by hand from the documented `CPython` stream.
fn bytes(hex: &str) -> Vec<u8> {
    assert!(
        hex.len() % 2 == 0,
        "hex fixtures have an even number of digits: {hex}"
    );
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
        .collect()
}

// `{"session_id": "s1", "n": 42}` exactly as CPython writes it with protocol 4:
// PROTO 4, FRAME(29), EMPTY_DICT, MEMOIZE, MARK,
// SHORT_BINUNICODE "session_id" MEMOIZE, SHORT_BINUNICODE "s1" MEMOIZE,
// SHORT_BINUNICODE "n" MEMOIZE, BININT 42, SETITEMS, STOP.
const PROTO4_DICT: &str =
    "8004951d000000000000007d94288c0a73657373696f6e5f6964948c027331948c016e944b2a752e";

// The same dict as CPython writes it with protocol 2:
// PROTO 2, EMPTY_DICT, BINPUT 0, MARK, BINUNICODE keys plus BINPUT,
// BINUNICODE "s1" BINPUT 1, BINUNICODE "n" BINPUT 2, BININT 42, SETITEMS, STOP.
const PROTO2_DICT: &str =
    "80027d710028580a00000073657373696f6e5f6964710158020000007331710258010000006e71034b2a752e";

// `{"session_id": "s1", "entity_id": "s1", "n": 42}` with protocol 2, where the
// repeated "s1" is a BINGET (h 02) of the value memoized under index 2:
// PROTO 2, EMPTY_DICT, BINPUT 0, MARK, BINUNICODE pairs with BINPUT,
// BINGET 2, BININT 42, SETITEMS, STOP.
const PROTO2_SHARED_STRING: &str = "80027d710028580a00000073657373696f6e5f696471015802000000733171025809000000656e746974795f69647103680258010000006e71044b2a752e";

// A payload mixing every supported scalar and container kind (protocol 4):
// EMPTY_DICT, MARK, a list built with APPENDS (`e`), a TUPLE2 pair, a nested
// EMPTY_DICT, NEWTRUE/NEWFALSE/NONE, BINFLOAT 1.5, LONG1 4294967296,
// SHORT_BINBYTES 00ff, SETITEMS, STOP.
const PROTO4_MIXED: &str = "8004956a000000000000007d94288c046c697374945d94284b014b02658c0470616972948c0161948c01629486948c066e6573746564947d948c016b944e738c05666c616773945d942888894e658c016694473ff80000000000008c03626967948a0500000000018c0372617794430200ff94752e";

// `{"a": {"x": 1}, "b": <the same dict>}` with protocol 2, where the second dict
// is a BINGET (h 02) of the memoized inner container:
// PROTO 2, EMPTY_DICT, BINPUT 0, MARK, BINUNICODE "a" BINPUT 1,
// EMPTY_DICT BINPUT 2, BINUNICODE "x" BINPUT 3, BININT1 1, SETITEM,
// BINUNICODE "b" BINPUT 4, BINGET 2, SETITEMS, STOP.
const PROTO2_SHARED_CONTAINER: &str =
    "80027d71002858010000006171017d710258010000007871034b017358010000006271046802752e";

// One full Python-written checkpoint payload (protocol 4, 299 bytes):
// PROTO 4, FRAME(288), EMPTY_DICT, MARK, SHORT_BINUNICODE keys with MEMOIZE,
// nested dicts for `global_state`, `component_state`, `trace`, `metadata`,
// APPENDS for the `[1, 2]` list, NONE for `updates`/`last_invoke_id`,
// BININT1 3 for `frame_count`, SETITEM/SETITEMS, STOP.
const PROTO4_CHECKPOINT: &str = "80049520010000000000007d94288c0a73657373696f6e5f6964948c027331948c0d636865636b706f696e745f6964948c03637031948c096e616d657370616365948c08776f726b666c6f77948c09656e746974795f6964948c03776631948c0a637265617465645f6174948c14323032352d30312d30315430303a30303a30305a948c0c676c6f62616c5f7374617465947d948c016e944b2a738c0f636f6d706f6e656e745f7374617465947d948c046e6f6465947d948c0178945d94284b014b026573738c0775706461746573944e8c057472616365947d94288c0874726163655f6964948c027431948c0e6c6173745f696e766f6b655f6964944e8c0b6672616d655f636f756e74944b03758c086d65746164617461947d948c046e6f7465948c0268699473752e";

// `{"n": 42}` as CPython writes it with protocol 4; the encoder pins these bytes.
const DUMPS_DICT: &str = "8004950a000000000000007d948c016e944b2a732e";

#[test]
fn reads_a_protocol_4_dict() {
    // ports openjiuwen/core/session/checkpointer/persistence.py
    let payload = bytes(PROTO4_DICT);
    assert_eq!(
        pickle::loads(&payload).unwrap(),
        json!({"session_id": "s1", "n": 42})
    );
}

#[test]
fn reads_the_equivalent_protocol_2_dict() {
    let payload = bytes(PROTO2_DICT);
    assert_eq!(
        pickle::loads(&payload).unwrap(),
        json!({"session_id": "s1", "n": 42})
    );
}

#[test]
fn reads_text_protocol_0_streams() {
    // `(lp0\nI1\naI2\na.` — MARK, LIST, memo PUT 0, text INT 1, APPEND, text INT 2,
    // APPEND, STOP.
    assert_eq!(pickle::loads(b"(lp0\nI1\naI2\na.").unwrap(), json!([1, 2]));
    // `Vh\xe9llo\np0\n.` — raw-unicode-escape UNICODE, text memo PUT 0, STOP.
    assert_eq!(
        pickle::loads(b"Vh\xe9llo\np0\n.").unwrap(),
        json!("h\u{e9}llo")
    );
}

#[test]
fn reads_every_supported_scalar_and_container() {
    let payload = bytes(PROTO4_MIXED);
    let expected = json!({
        "list": [1, 2],
        "pair": ["a", "b"],
        "nested": {"k": null},
        "flags": [true, false, null],
        "f": 1.5,
        "big": 4_294_967_296_i64,
        "raw": {BYTES_KEY: "00ff"},
    });
    assert_eq!(pickle::loads(&payload).unwrap(), expected);
}

#[test]
fn memo_references_reuse_strings_and_containers() {
    assert_eq!(
        pickle::loads(&bytes(PROTO2_SHARED_STRING)).unwrap(),
        json!({"session_id": "s1", "entity_id": "s1", "n": 42})
    );
    assert_eq!(
        pickle::loads(&bytes(PROTO2_SHARED_CONTAINER)).unwrap(),
        json!({"a": {"x": 1}, "b": {"x": 1}})
    );
}

#[test]
fn dumps_round_trips_through_loads() {
    let long = "x".repeat(300);
    let values = [
        Value::Null,
        json!(true),
        json!(false),
        json!(0),
        json!(42),
        json!(-1),
        json!(255),
        json!(65536),
        json!(-4_294_967_296_i64),
        json!(i64::MIN),
        json!(i64::MAX),
        json!(1.5),
        json!(""),
        json!("h\u{e9}llo"),
        json!("\u{1f600}"),
        Value::String(long),
        json!([]),
        json!({}),
        json!([[], {}, [1, [2, [3]]]]),
        json!({"k": {"n": null}, "list": [true, 1.5, "s"]}),
        json!({BYTES_KEY: "00ff"}),
        json!({BYTES_KEY: ""}),
        json!(["a", "a", "b", "a"]),
    ];
    for value in values {
        let encoded = pickle::dumps(&value).unwrap();
        assert_eq!(pickle::loads(&encoded).unwrap(), value, "{value}");
    }
}

#[test]
fn dumps_pins_the_protocol_4_encoding() {
    let encoded = pickle::dumps(&json!({"n": 42})).unwrap();
    assert_eq!(encoded, bytes(DUMPS_DICT));
    assert!(encoded.starts_with(&[0x80, 0x04]), "PROTO 4 header");
    // The frame length covers the body and the trailing STOP, like CPython.
    assert_eq!(
        u64::from_le_bytes(encoded[3..11].try_into().unwrap()),
        u64::try_from(encoded.len() - 3).unwrap() - 8
    );
    assert_eq!(encoded.last().copied(), Some(b'.'));
}

#[test]
fn rejects_an_unknown_opcode() {
    // PROTO 4, then GLOBAL (`c`), which this codec deliberately does not decode.
    let error = pickle::loads(&[0x80, 0x04, b'c']).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("GLOBAL"), "{message}");
    assert!(message.contains("offset 2"), "{message}");
}

#[test]
fn rejects_a_truncated_stream() {
    // SHORT_BINUNICODE declares five bytes but only two remain.
    let error = pickle::loads(&[0x80, 0x04, 0x8c, 0x05, b'a', b'b']).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("truncated"), "{error}");
}

#[test]
fn rejects_a_non_string_dict_key() {
    // EMPTY_DICT, MEMOIZE, BININT1 1, BININT 2, SETITEM, STOP.
    let payload = bytes("80047d944b014b02732e");
    let error = pickle::loads(&payload).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("dict key"), "{message}");
    assert!(message.contains("integer"), "{message}");
}

#[test]
fn rejects_a_memo_reference_before_it_is_stored() {
    // PROTO 2, BINGET 5, STOP.
    let error = pickle::loads(&[0x80, 0x02, b'h', 0x05, b'.']).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("memo reference 5"), "{error}");
}

#[test]
fn rejects_values_the_writer_cannot_represent() {
    // An integer above i64 has no lossless pickle form that `loads` accepts.
    let error = pickle::dumps(&json!(u64::MAX)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("i64"), "{error}");
    // The reserved bytes shape must hold hex.
    let error = pickle::dumps(&json!({BYTES_KEY: "not-hex"})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains(BYTES_KEY), "{error}");
}

#[test]
fn reads_a_python_written_checkpoint() {
    let checkpoint = checkpoint_from_pickle(&bytes(PROTO4_CHECKPOINT)).unwrap();
    assert_eq!(checkpoint.session_id, "s1");
    assert_eq!(checkpoint.checkpoint_id, "cp1");
    assert_eq!(checkpoint.namespace, CheckpointNamespace::Workflow);
    assert_eq!(checkpoint.entity_id, "wf1");
    assert_eq!(checkpoint.created_at, "2025-01-01T00:00:00Z");
    assert_eq!(checkpoint.global_state, json!({"n": 42}));
    assert_eq!(checkpoint.component_state, json!({"node": {"x": [1, 2]}}));
    assert_eq!(checkpoint.updates, Value::Null);
    let trace = checkpoint.trace.expect("trace position");
    assert_eq!(trace.trace_id, "t1");
    assert_eq!(trace.last_invoke_id, None);
    assert_eq!(trace.frame_count, 3);
    assert_eq!(checkpoint.metadata.get("note"), Some(&json!("hi")));
}

#[test]
fn checkpoint_pickles_round_trip() {
    let expected = checkpoint_from_pickle(&bytes(PROTO4_CHECKPOINT)).unwrap();
    let encoded = checkpoint_to_pickle(&expected).unwrap();
    let decoded = checkpoint_from_pickle(&encoded).unwrap();
    assert_eq!(decoded.session_id, expected.session_id);
    assert_eq!(decoded.checkpoint_id, expected.checkpoint_id);
    assert_eq!(decoded.namespace, expected.namespace);
    assert_eq!(decoded.entity_id, expected.entity_id);
    assert_eq!(decoded.created_at, expected.created_at);
    assert_eq!(decoded.global_state, expected.global_state);
    assert_eq!(decoded.component_state, expected.component_state);
    assert_eq!(decoded.updates, expected.updates);
    assert_eq!(decoded.metadata, expected.metadata);
    assert_eq!(
        decoded.trace.map(|trace| trace.frame_count),
        expected.trace.map(|trace| trace.frame_count)
    );
    // The payload is the checkpoint's snake_case JSON shape.
    let payload = pickle::loads(&encoded).unwrap();
    assert_eq!(payload["session_id"], json!("s1"));
    assert_eq!(payload["component_state"], json!({"node": {"x": [1, 2]}}));
}

#[test]
fn checkpoint_decoding_generates_a_missing_id() {
    let payload = pickle::dumps(&json!({"session_id": "s1"})).unwrap();
    let checkpoint: Checkpoint = checkpoint_from_pickle(&payload).unwrap();
    assert_eq!(checkpoint.session_id, "s1");
    assert_ne!(checkpoint.checkpoint_id, "");
    assert_eq!(checkpoint.namespace, CheckpointNamespace::Agent);
}

#[test]
fn checkpoint_decoding_rejects_bad_payloads() {
    let payload = pickle::dumps(&json!(42)).unwrap();
    let error = checkpoint_from_pickle(&payload).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("dict"), "{error}");

    let payload = pickle::dumps(&json!({"checkpoint_id": "cp1"})).unwrap();
    let error = checkpoint_from_pickle(&payload).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("session_id"), "{error}");

    let payload = pickle::dumps(&json!({"session_id": "../escape"})).unwrap();
    let error = checkpoint_from_pickle(&payload).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}
