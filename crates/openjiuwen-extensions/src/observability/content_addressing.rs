//! Content-addressed sequences for the trajectory storage path.
//!
//! Port of `openjiuwen/extensions/observability/content_addressing.py`
//! (v0.1.20), plus the pure half of the addressing loop in
//! `openjiuwen/extensions/observability/otlp_codec.py` (see
//! [`address_payload`]).
//!
//! The `GenAI` convention has every model call state its whole input: the
//! messages, the tool definitions, the system instructions.
//! That is right for the protocol and wasteful for storage, because
//! consecutive calls of one conversation repeat almost all of it.
//! Measured on one real session, 803 calls carried three distinct tool
//! definitions, and 57,550 message slots held 1,991 distinct messages.
//!
//! So the wire keeps stating everything and storage stops repeating it.
//! Each such attribute is treated as a *sequence*, every element is addressed
//! by the hash of its content, and the sequence itself is a hash chain:
//!
//! ```text
//! seq(0) = H(""       || H(element 0))
//! seq(i) = H(seq(i-1) || H(element i))
//! ```
//!
//! Two properties follow, and both are load-bearing for the reader:
//!
//! - Equal `seq_hash` means every element is equal, in order.
//!   A reader holding the hash knows nothing changed without fetching anything.
//! - A shared prefix produces shared chain nodes, so the fork between two
//!   versions is where their chains stop agreeing — which is what a diff is.
//!
//! Nothing here holds state.
//! The chain of a sequence depends only on its content, so the same messages
//! always build the same nodes, whatever order spans are encoded in and
//! whichever thread encodes them.
//! Deduplication is the storage layer inserting a node it already has.
//!
//! # Element encoding
//!
//! Python serializes each element with `json.dumps(..., separators=(",", ":"))`
//! and reads it back the same way; this module uses [`serde_json::to_string`],
//! whose writer already emits compact JSON with no spaces after `,` or `:`.
//! Byte-identical round-tripping of the original attribute is deliberately not
//! a goal: the original spacing carries no meaning, and reproducing it would
//! make the hash depend on how a caller happened to serialize its messages.
//!
//! # What is not here
//!
//! The protobuf encode/decode half of `encode_span_with_addressed_sequences`
//! is SDK-only and stays on the export path (`crate::otel::otlp`'s JSON
//! encoder); [`address_payload`] is the storage-path transformation itself,
//! which operates on an already-decoded OTLP payload and is pure JSON.
use crate::otel::semconv::{
    GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES, GEN_AI_SYSTEM_INSTRUCTIONS,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
/// Version prefix of a stored reference.
///
/// It exists so a later change of hash algorithm or node shape stays
/// recognisable next to what is already stored.
pub const SEQUENCE_REFERENCE_VERSION: &str = "1";
/// Prefix that marks an attribute value as a sequence reference.
pub const SEQUENCE_REFERENCE_PREFIX: &str = "@oj-seq";
/// Wire name `gen_ai.tool.definitions` — the tool-definition payload an LLM
/// call restates on every request.
///
/// [`crate::otel::semconv`] re-exports the other three `GenAI` payload keys but
/// has no constant for this one yet, so the wire name lives here.
const GEN_AI_TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";
/// The attributes a conversation restates on every call, and which the storage
/// path therefore addresses.
///
/// The Python `otlp_codec.ADDRESSED_ATTRIBUTE_KEYS` set: `gen_ai.input.messages`,
/// `gen_ai.output.messages`, `gen_ai.system_instructions` and
/// `gen_ai.tool.definitions`.
pub const ADDRESSED_ATTRIBUTE_KEYS: [&str; 4] = [
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_DEFINITIONS,
];
/// Lowercase hexadecimal digits, used to encode a digest by hand.
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
/// Separator mixed into the chain hash so a prefix cannot be confused with an
/// element's own digest.
const CHAIN_SEPARATOR: u8 = 0x00;
/// One prefix of a sequence, addressed by the content of that prefix.
///
/// The Python `SequenceNode` dataclass; `prev_hash` is `None` for the first
/// element, and `depth` is one-based (the number of elements in the prefix).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceNode {
    /// Hash of the whole prefix up to and including this element.
    pub seq_hash: String,
    /// Hash of the prefix before this element, `None` for the first one.
    pub prev_hash: Option<String>,
    /// Hash of this element's compact JSON text.
    pub blob_hash: String,
    /// Number of elements in this prefix, starting at 1.
    pub depth: usize,
}
/// One attribute value expressed as a chain, plus what it introduced.
///
/// The Python `AddressedSequence` dataclass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddressedSequence {
    /// Attribute the sequence was built from.
    pub key: String,
    /// Hash of the complete sequence; the reference a record keeps.
    pub seq_hash: String,
    /// Number of elements.
    pub depth: usize,
    /// Length in bytes of the attribute value before it was replaced.
    ///
    /// The read path budgets pages by this, because a reader receives the
    /// rebuilt value rather than the reference.
    pub logical_bytes: usize,
    /// Chain nodes of the whole sequence, shallowest first.
    pub nodes: Vec<SequenceNode>,
    /// Element content by hash; a repeated element is stored once.
    pub blobs: BTreeMap<String, Vec<u8>>,
}
/// Returns the address of one piece of content: the full lowercase SHA-256
/// hex digest (64 characters).
#[must_use]
pub fn content_hash(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}
/// Returns the value stored in place of a sequence: `@oj-seq:1:<hash>:<depth>`.
#[must_use]
pub fn sequence_reference(seq_hash: &str, depth: usize) -> String {
    format!("{SEQUENCE_REFERENCE_PREFIX}:{SEQUENCE_REFERENCE_VERSION}:{seq_hash}:{depth}")
}
/// Returns the hash and depth a reference names, or `None` for a plain value.
///
/// A reference is exactly four colon-separated parts, must start with
/// [`SEQUENCE_REFERENCE_PREFIX`], carry [`SEQUENCE_REFERENCE_VERSION`], name a
/// non-empty hash, and state a depth that parses as a non-negative integer.
#[must_use]
pub fn parse_sequence_reference(value: &str) -> Option<(&str, usize)> {
    if !value.starts_with(SEQUENCE_REFERENCE_PREFIX) {
        return None;
    }
    let mut parts = value.split(':');
    let (prefix, version, seq_hash, depth) =
        (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some()
        || prefix != SEQUENCE_REFERENCE_PREFIX
        || version != SEQUENCE_REFERENCE_VERSION
        || seq_hash.is_empty()
    {
        return None;
    }
    Some((seq_hash, depth.parse().ok()?))
}
/// Splits one attribute value into the elements it states.
///
/// Only an array has elements.
/// Anything else — an object, a bare string, a value that does not parse —
/// states nothing to address here, and saying so is what keeps a chain
/// unambiguous: every chain is built from an array, so every chain rebuilds
/// into one, whatever its depth.
/// Treating a scalar as a sequence of one made it indistinguishable from a
/// single-element array, and left both rebuild paths guessing which they held.
/// A one-message answer — the ordinary case for an assistant turn — is exactly
/// where that guess went wrong.
///
/// Each element is re-serialized compactly, matching Python's
/// `separators=(",", ":")`; `serde_json::to_string` writes the same form.
#[must_use]
pub fn split_elements(value: &str) -> Vec<String> {
    let Ok(Value::Array(parsed)) = serde_json::from_str::<Value>(value) else {
        return Vec::new();
    };
    parsed
        .iter()
        .map(|element| serde_json::to_string(element).unwrap_or_default())
        .collect()
}
/// Returns the attribute value a sequence of elements states.
///
/// Always an array: a chain is built from one, so it rebuilds into one at any
/// depth. Nothing here inspects the count.
#[must_use]
pub fn rebuild_value(elements: &[String]) -> String {
    let mut value = String::with_capacity(elements.iter().map(String::len).sum::<usize>() + 2);
    value.push('[');
    value.push_str(&elements.join(","));
    value.push(']');
    value
}
/// Expresses one attribute value as a chain of content-addressed elements.
///
/// Returns `None` when the value states nothing to address — an empty value, an
/// empty array, or anything that is not an array at all.
/// A caller that gets `None` leaves the value as it is.
#[must_use]
pub fn build_sequence(key: impl Into<String>, value: &str) -> Option<AddressedSequence> {
    if value.is_empty() {
        return None;
    }
    let elements = split_elements(value);
    if elements.is_empty() {
        return None;
    }
    let mut nodes = Vec::with_capacity(elements.len());
    let mut blobs = BTreeMap::new();
    let mut previous: Option<String> = None;
    for (index, element) in elements.iter().enumerate() {
        let encoded = element.as_bytes().to_vec();
        let blob_hash = content_hash(&encoded);
        blobs.insert(blob_hash.clone(), encoded);
        let seq_hash = chain_hash(previous.as_deref(), &blob_hash);
        nodes.push(SequenceNode {
            seq_hash: seq_hash.clone(),
            prev_hash: previous.take(),
            blob_hash,
            depth: index + 1,
        });
        previous = Some(seq_hash);
    }
    let seq_hash = nodes.last()?.seq_hash.clone();
    Some(AddressedSequence {
        key: key.into(),
        seq_hash,
        depth: nodes.len(),
        // Python measures the value *before* it was replaced, not the rebuilt
        // text: the read path budgets pages by what a reader receives.
        logical_bytes: value.len(),
        nodes,
        blobs,
    })
}
/// Returns the `(key, value)` pairs of an OTLP payload that `keys` names.
///
/// `payload` is one decoded OTLP `ExportTraceServiceRequest`; the walk covers
/// `resourceSpans[].scopeSpans[].spans[].attributes[]` and tolerates missing
/// levels and non-object entries.
/// Only attributes whose `value.stringValue` is a string are reported, because
/// those are the only ones [`build_sequence`] can address.
/// Pairs are returned in document order.
#[must_use]
pub fn addressable_attributes(payload: &Value, keys: &[&str]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let Some(resource_spans) = payload.get("resourceSpans").and_then(Value::as_array) else {
        return found;
    };
    for resource_span in resource_spans {
        let Some(scope_spans) = resource_span.get("scopeSpans").and_then(Value::as_array) else {
            continue;
        };
        for scope_span in scope_spans {
            let Some(spans) = scope_span.get("spans").and_then(Value::as_array) else {
                continue;
            };
            for span in spans {
                let Some(attributes) = span.get("attributes").and_then(Value::as_array) else {
                    continue;
                };
                for attribute in attributes {
                    let Some(key) = attribute.get("key").and_then(Value::as_str) else {
                        continue;
                    };
                    if !keys.contains(&key) {
                        continue;
                    }
                    let Some(stated) = attribute
                        .get("value")
                        .and_then(|value| value.get("stringValue"))
                        .and_then(Value::as_str)
                    else {
                        continue;
                    };
                    found.push((key.to_owned(), stated.to_owned()));
                }
            }
        }
    }
    found
}
/// Replaces the restated attributes of one decoded OTLP payload with sequence
/// references, returning every sequence the payload now depends on.
///
/// This is the storage-path transformation of Python's
/// `encode_span_with_addressed_sequences`: the exporter path keeps receiving
/// the complete span, while this path — which has no obligation to repeat what
/// storage already holds — rewrites each addressed `stringValue` in place.
/// A reader needs the returned sequences to rebuild the span; `None` from
/// [`build_sequence`] (a non-array, an empty value) leaves the attribute as it
/// was.
#[must_use]
pub fn address_payload(payload: &mut Value, keys: &[&str]) -> Vec<AddressedSequence> {
    let mut sequences = Vec::new();
    let Some(resource_spans) = payload
        .get_mut("resourceSpans")
        .and_then(Value::as_array_mut)
    else {
        return sequences;
    };
    for resource_span in resource_spans {
        let Some(scope_spans) = resource_span
            .get_mut("scopeSpans")
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        for scope_span in scope_spans {
            let Some(spans) = scope_span.get_mut("spans").and_then(Value::as_array_mut) else {
                continue;
            };
            for span in spans {
                let Some(attributes) = span.get_mut("attributes").and_then(Value::as_array_mut)
                else {
                    continue;
                };
                for attribute in attributes {
                    let Some(key) = attribute.get("key").and_then(Value::as_str) else {
                        continue;
                    };
                    if !keys.contains(&key) {
                        continue;
                    }
                    let key = key.to_owned();
                    let Some(Value::Object(value)) = attribute.get_mut("value") else {
                        continue;
                    };
                    let Some(Value::String(stated)) = value.get("stringValue") else {
                        continue;
                    };
                    let stated = stated.clone();
                    let Some(sequence) = build_sequence(key, &stated) else {
                        continue;
                    };
                    let reference = sequence_reference(&sequence.seq_hash, sequence.depth);
                    value.insert("stringValue".to_owned(), Value::String(reference));
                    sequences.push(sequence);
                }
            }
        }
    }
    sequences
}
/// Returns the chain hash of one prefix: `H(previous || \x00 || blob_hash)`.
fn chain_hash(previous: Option<&str>, blob_hash: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(previous.unwrap_or_default().as_bytes());
    digest.update([CHAIN_SEPARATOR]);
    digest.update(blob_hash.as_bytes());
    hex_lower(&digest.finalize())
}
/// Returns the lowercase hexadecimal encoding of `bytes`.
fn hex_lower(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}
#[cfg(test)]
mod tests {
    //! Port of `tests/unit_tests/extensions/observability/test_content_addressing.py`.
    //!
    //! A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::collections::BTreeMap;

    use super::{
        address_payload, addressable_attributes, build_sequence, content_hash,
        parse_sequence_reference, rebuild_value, sequence_reference, split_elements,
        ADDRESSED_ATTRIBUTE_KEYS, SEQUENCE_REFERENCE_PREFIX, SEQUENCE_REFERENCE_VERSION,
    };
    use serde_json::{json, Value};
    /// A conversation of `count` user messages, as the convention states it.
    fn messages(count: usize) -> String {
        let turns: Vec<Value> = (0..count)
            .map(|index| json!({"role": "user", "content": format!("turn {index}")}))
            .collect();
        serde_json::to_string(&turns).expect("a message list serializes")
    }
    /// One decoded OTLP payload with one span carrying `attributes`.
    fn payload(attributes: &Value) -> Value {
        json!({
            "resourceSpans": [{
                "scopeSpans": [{
                    "spans": [{ "name": "chat model", "attributes": attributes }]
                }]
            }]
        })
    }
    #[test]
    fn constants_name_the_wire_format() {
        assert_eq!(SEQUENCE_REFERENCE_VERSION, "1");
        assert_eq!(SEQUENCE_REFERENCE_PREFIX, "@oj-seq");
        assert_eq!(
            ADDRESSED_ATTRIBUTE_KEYS,
            [
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.system_instructions",
                "gen_ai.tool.definitions",
            ]
        );
    }
    #[test]
    fn content_hash_is_the_full_lowercase_sha256() {
        // SHA-256 of the empty string, and of `"abc"`.
        assert_eq!(
            content_hash(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            content_hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(content_hash(b"abc").len(), 64);
    }
    #[test]
    fn the_same_content_always_builds_the_same_chain() {
        // Nothing is remembered between calls; the content decides the chain.
        // This is what lets any thread encode any span in any order and still
        // let storage deduplicate by simply inserting a node it already holds.
        let first = build_sequence("gen_ai.input.messages", &messages(5));
        let second = build_sequence("gen_ai.input.messages", &messages(5));
        let (first, second) = (first.unwrap(), second.unwrap());
        assert_eq!(first, second);
        let first_hashes: Vec<&String> = first.nodes.iter().map(|node| &node.seq_hash).collect();
        let second_hashes: Vec<&String> = second.nodes.iter().map(|node| &node.seq_hash).collect();
        assert_eq!(first_hashes, second_hashes);
    }
    #[test]
    fn a_shared_prefix_shares_its_chain_nodes() {
        // A longer conversation reuses every node of the shorter one it extends.
        let short = build_sequence("gen_ai.input.messages", &messages(5)).unwrap();
        let long = build_sequence("gen_ai.input.messages", &messages(8)).unwrap();
        let shared: Vec<&String> = long.nodes[..5].iter().map(|node| &node.seq_hash).collect();
        let shorter: Vec<&String> = short.nodes.iter().map(|node| &node.seq_hash).collect();
        assert_eq!(shared, shorter);
        // The head of the shorter sequence is an ancestor of the longer one,
        // which is what makes "has the prefix changed" answerable without
        // reading content.
        assert_eq!(long.nodes[5].prev_hash.as_ref(), Some(&short.seq_hash));
        assert_eq!(long.depth, 8);
    }
    #[test]
    fn changing_an_early_element_forks_the_chain() {
        // A rewritten prefix produces a different chain from the point it changed.
        let original = build_sequence("gen_ai.input.messages", &messages(5)).unwrap();
        let mut edited: Value = serde_json::from_str(&messages(5)).unwrap();
        edited.as_array_mut().unwrap()[1]["content"] = json!("compacted");
        let forked = build_sequence(
            "gen_ai.input.messages",
            &serde_json::to_string(&edited).unwrap(),
        )
        .unwrap();
        assert_eq!(original.nodes[0].seq_hash, forked.nodes[0].seq_hash);
        assert_ne!(original.nodes[1].seq_hash, forked.nodes[1].seq_hash);
        assert_ne!(original.seq_hash, forked.seq_hash);
        // The first element is unchanged, so its blob is shared.
        assert_eq!(original.nodes[0].blob_hash, forked.nodes[0].blob_hash);
    }
    #[test]
    fn only_an_array_is_addressed() {
        // A chain states an array, so nothing else becomes one.
        assert!(build_sequence("gen_ai.system_instructions", "you are an agent").is_none());
        assert!(build_sequence("gen_ai.output.messages", r#"{"role":"assistant"}"#).is_none());
        assert!(build_sequence("gen_ai.input.messages", "not json at all").is_none());
    }
    #[test]
    fn a_single_element_array_rebuilds_as_an_array() {
        // One assistant message is the ordinary case, and it must stay an
        // array. Losing the brackets here left every finished answer
        // unreadable.
        let value = r#"[{"role":"assistant","parts":[{"type":"text","content":"hi"}]}]"#;
        let sequence = build_sequence("gen_ai.output.messages", value).unwrap();
        assert_eq!(sequence.depth, 1);
        assert!(sequence.nodes[0].prev_hash.is_none());
        assert_eq!(sequence.seq_hash, sequence.nodes[0].seq_hash);
        let rebuilt = rebuild_value(&split_elements(value));
        assert_eq!(
            serde_json::from_str::<Value>(&rebuilt).unwrap(),
            serde_json::from_str::<Value>(value).unwrap()
        );
    }
    #[test]
    fn an_empty_value_addresses_nothing() {
        assert!(build_sequence("gen_ai.input.messages", "").is_none());
        assert!(build_sequence("gen_ai.input.messages", "[]").is_none());
    }
    #[test]
    fn a_reference_round_trips() {
        assert_eq!(
            parse_sequence_reference(&sequence_reference("abc", 7)),
            Some(("abc", 7))
        );
        assert_eq!(sequence_reference("abc", 7), "@oj-seq:1:abc:7");
        assert_eq!(parse_sequence_reference("not a reference"), None);
        assert_eq!(parse_sequence_reference("@oj-seq:99:abc:7"), None);
        assert_eq!(parse_sequence_reference("@oj-seq:1:abc:not-a-number"), None);
        assert_eq!(parse_sequence_reference("@oj-seq:1::7"), None);
        assert_eq!(parse_sequence_reference("@oj-seq:1:abc:7:extra"), None);
        assert_eq!(parse_sequence_reference(""), None);
    }
    #[test]
    fn elements_rebuild_the_value_they_came_from() {
        let value = messages(3);
        let elements = split_elements(&value);
        assert_eq!(elements.len(), 3);
        // Compact, no spaces after `,` or `:` (Python's `separators` pair), and
        // key order as stated (the workspace parses JSON with `preserve_order`).
        assert_eq!(elements[0], r#"{"role":"user","content":"turn 0"}"#);
        assert!(!elements[0].contains(": "));
        assert!(!elements[0].contains(", "));
        assert_eq!(
            serde_json::from_str::<Value>(&rebuild_value(&elements)).unwrap(),
            serde_json::from_str::<Value>(&value).unwrap()
        );
        assert_eq!(rebuild_value(&[]), "[]");
    }
    #[test]
    fn split_elements_keeps_non_ascii_text_unescaped() {
        let elements = split_elements(r#"[{"content":"你好"}]"#);
        assert_eq!(elements, vec![r#"{"content":"你好"}"#.to_owned()]);
    }
    #[test]
    fn logical_size_measures_the_stated_value_not_the_reference() {
        // A page is budgeted by what a reader receives, which is the rebuilt
        // span. Measuring the reference-carrying bytes would let one page
        // promise its budget and deliver many times it.
        let stated = messages(40);
        let sequence = build_sequence("gen_ai.input.messages", &stated).unwrap();
        assert_eq!(sequence.logical_bytes, stated.len());
        let reference = sequence_reference(&sequence.seq_hash, sequence.depth);
        assert!(sequence.logical_bytes > reference.len());
    }
    #[test]
    fn repeated_elements_are_stored_once() {
        let value = r#"[{"role":"user","content":"again"},{"role":"user","content":"again"}]"#;
        let sequence = build_sequence("gen_ai.input.messages", value).unwrap();
        assert_eq!(sequence.depth, 2);
        assert_eq!(sequence.blobs.len(), 1);
        assert_eq!(sequence.nodes[0].blob_hash, sequence.nodes[1].blob_hash);
        assert_ne!(sequence.nodes[0].seq_hash, sequence.nodes[1].seq_hash);
    }
    #[test]
    fn addressable_attributes_finds_only_named_string_attributes() {
        let payload = payload(&json!([
            {"key": "gen_ai.input.messages", "value": {"stringValue": messages(2)}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "model-x"}},
            {"key": "gen_ai.system_instructions", "value": {"intValue": "3"}},
            {"key": "gen_ai.output.messages"},
            {"value": {"stringValue": "no key"}}
        ]));
        let found = addressable_attributes(&payload, &ADDRESSED_ATTRIBUTE_KEYS);
        assert_eq!(
            found,
            vec![("gen_ai.input.messages".to_owned(), messages(2))]
        );
    }
    #[test]
    fn addressable_attributes_tolerates_missing_levels() {
        for payload in [
            json!(null),
            json!({}),
            json!({"resourceSpans": 3}),
            json!({"resourceSpans": [{"scopeSpans": [{"spans": [{"attributes": "no"}]}]}]}),
            json!({"resourceSpans": [{"scopeSpans": [{"spans": [{"attributes": [1, "x"]}]}]}]}),
        ] {
            let found = addressable_attributes(&payload, &ADDRESSED_ATTRIBUTE_KEYS);
            assert_eq!(found, Vec::new());
        }
    }
    #[test]
    fn addressing_replaces_restated_attributes_with_references() {
        // The storage path carries references; the sequences state the content.
        let instructions = r#"[{"type":"text","content":"be brief"}]"#;
        let mut payload = payload(&json!([
            {"key": "gen_ai.input.messages", "value": {"stringValue": messages(4)}},
            {"key": "gen_ai.system_instructions", "value": {"stringValue": instructions}},
            // An addressed key whose value is not an array stays as it is,
            // rather than becoming a chain that would rebuild as one.
            {"key": "gen_ai.tool.definitions", "value": {"stringValue": "no tools"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "model-x"}}
        ]));
        let sequences = address_payload(&mut payload, &ADDRESSED_ATTRIBUTE_KEYS);
        let by_key: BTreeMap<&str, &super::AddressedSequence> = sequences
            .iter()
            .map(|sequence| (sequence.key.as_str(), sequence))
            .collect();
        assert_eq!(
            by_key.keys().copied().collect::<Vec<&str>>(),
            vec!["gen_ai.input.messages", "gen_ai.system_instructions"]
        );
        let stated: BTreeMap<String, String> = addressable_attributes(
            &payload,
            &[
                "gen_ai.input.messages",
                "gen_ai.system_instructions",
                "gen_ai.tool.definitions",
                "gen_ai.request.model",
            ],
        )
        .into_iter()
        .collect();
        for (key, sequence) in &by_key {
            assert_eq!(
                parse_sequence_reference(&stated[*key]),
                Some((sequence.seq_hash.as_str(), sequence.depth))
            );
        }
        assert_eq!(stated["gen_ai.tool.definitions"], "no tools");
        assert_eq!(stated["gen_ai.request.model"], "model-x");
    }
}
