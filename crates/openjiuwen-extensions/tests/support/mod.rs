//! Shared harness for the OTLP crate-boundary suites.
//!
//! Two integration binaries decode a real OTLP/protobuf
//! `ExportTraceServiceRequest` with the small hand-written reader below (no
//! `prost`, no `opentelemetry-otlp` — neither is available offline):
//!
//! - `otel_otlp_protobuf.rs` pins the exporter's wire payload field by field;
//! - `otel_team_memory.rs` proves the team-memory → OTLP bridge in both
//!   OTLP/HTTP encodings and reads the span names, attributes, trace ids and
//!   statuses back out of the protobuf body.
//!
//! The reader lives here so both suites share one measuring instrument. Every
//! item below is used by at least one of them, but each binary only uses a
//! subset, so the module silences `dead_code` instead of duplicating the reader
//! (or trimming it in a way that weakens the protobuf suite's wire coverage).
//!
//! The reader understands every wire type the OTLP message set uses and is
//! itself pinned by `otel_otlp_protobuf.rs`'s
//! `the_decoder_round_trips_every_wire_type`.

#![allow(dead_code)]

use std::collections::BTreeMap;

/// Varint wire type: `int32`, `int64`, `uint32`, `uint64`, `bool`, enums.
pub(crate) const WIRE_VARINT: u8 = 0;
/// 64-bit wire type: `fixed64` and `double`.
pub(crate) const WIRE_FIXED64: u8 = 1;
/// Length-delimited wire type: strings, bytes, and sub-messages.
pub(crate) const WIRE_LEN: u8 = 2;
/// 32-bit wire type: `fixed32` and `float`.
pub(crate) const WIRE_FIXED32: u8 = 5;

/// One decoded protobuf value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Wire {
    /// Wire type 0.
    Varint(u64),
    /// Wire type 1.
    Fixed64(u64),
    /// Wire type 5.
    Fixed32(u32),
    /// Wire type 2: strings, bytes, and buffered sub-messages.
    Bytes(Vec<u8>),
}

/// The decoded fields of one message, keyed by field number.
pub(crate) type Fields = BTreeMap<u32, Vec<Wire>>;

/// Reads one base-128 varint, advancing `offset`.
pub(crate) fn read_varint(input: &[u8], offset: &mut usize) -> u64 {
    let mut value = 0_u64;
    let mut shift = 0_u32;
    loop {
        let byte = *input.get(*offset).expect("truncated varint");
        *offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

/// Decodes one message into its fields.
pub(crate) fn decode(input: &[u8]) -> Fields {
    let mut fields = Fields::new();
    let mut offset = 0;
    while offset < input.len() {
        let key = read_varint(input, &mut offset);
        let field = u32::try_from(key >> 3).expect("field number out of range");
        let wire_type = u8::try_from(key & 0x7).expect("wire type out of range");
        let value = match wire_type {
            WIRE_VARINT => Wire::Varint(read_varint(input, &mut offset)),
            WIRE_FIXED64 => {
                let bytes: [u8; 8] = input
                    .get(offset..offset + 8)
                    .expect("truncated fixed64")
                    .try_into()
                    .expect("fixed64 is eight bytes");
                offset += 8;
                Wire::Fixed64(u64::from_le_bytes(bytes))
            }
            WIRE_FIXED32 => {
                let bytes: [u8; 4] = input
                    .get(offset..offset + 4)
                    .expect("truncated fixed32")
                    .try_into()
                    .expect("fixed32 is four bytes");
                offset += 4;
                Wire::Fixed32(u32::from_le_bytes(bytes))
            }
            WIRE_LEN => {
                let length =
                    usize::try_from(read_varint(input, &mut offset)).expect("length out of range");
                let bytes = input
                    .get(offset..offset + length)
                    .expect("truncated length-delimited field")
                    .to_vec();
                offset += length;
                Wire::Bytes(bytes)
            }
            other => panic!("unsupported wire type {other}"),
        };
        fields.entry(field).or_default().push(value);
    }
    fields
}

/// The first varint stored under `field`.
pub(crate) fn varint(fields: &Fields, field: u32) -> Option<u64> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Varint(value)) => Some(*value),
        _ => None,
    })
}

/// The first `fixed64`/`double` word stored under `field`.
pub(crate) fn fixed64(fields: &Fields, field: u32) -> Option<u64> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Fixed64(value)) => Some(*value),
        _ => None,
    })
}

/// The first `fixed32`/`float` word stored under `field`.
pub(crate) fn fixed32(fields: &Fields, field: u32) -> Option<u32> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Fixed32(value)) => Some(*value),
        _ => None,
    })
}

/// The first length-delimited payload stored under `field`.
pub(crate) fn bytes(fields: &Fields, field: u32) -> Option<&[u8]> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Bytes(value)) => Some(value.as_slice()),
        _ => None,
    })
}

/// The first string stored under `field`.
pub(crate) fn string(fields: &Fields, field: u32) -> Option<String> {
    bytes(fields, field).map(|bytes| String::from_utf8(bytes.to_vec()).expect("UTF-8 string"))
}

/// Every sub-message stored under `field`, in wire order.
pub(crate) fn messages(fields: &Fields, field: u32) -> Vec<Fields> {
    fields.get(&field).map_or_else(Vec::new, |values| {
        values
            .iter()
            .filter_map(|value| match value {
                Wire::Bytes(bytes) => Some(decode(bytes)),
                _ => None,
            })
            .collect()
    })
}

/// Decodes a repeated `KeyValue` field into attribute name → `AnyValue` fields.
pub(crate) fn key_values(fields: &Fields, field: u32) -> BTreeMap<String, Fields> {
    messages(fields, field)
        .iter()
        .map(|pair| {
            let key = string(pair, 1).expect("every KeyValue has a key");
            let mut values = messages(pair, 2);
            assert_eq!(values.len(), 1, "every KeyValue has one AnyValue");
            (key, values.remove(0))
        })
        .collect()
}

/// The `string_value` of an `AnyValue`.
pub(crate) fn any_string(value: &Fields) -> String {
    string(value, 1).expect("the AnyValue carries a string_value")
}
