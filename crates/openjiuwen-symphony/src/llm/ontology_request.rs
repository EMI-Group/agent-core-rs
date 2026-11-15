//! The compact ontology-match request sent to the model.
//!
//! This module owns everything about *building* the request — the system instruction,
//! the direction tokens, the description budget and the pruning rules that keep a prompt
//! short — so [`OntologyMatcher`](crate::llm::ontology::OntologyMatcher) stays focused on
//! validating the reply, applying the fallback and reporting provenance.
//!
//! One request is one batch:
//!
//! ```json
//! {"candidates":[{"id":"c1","source":{"name":"…","description":"…"},
//!  "target":{"name":"…","description":"…"},
//!  "directions":{"forward":{"outputs":[…],"inputs":[…],"ports":[…]},
//!                "reverse":{"outputs":[…],"inputs":[…],"ports":[…]}}}]}
//! ```
//!
//! Blank text, empty arrays and a direction carrying no IO declaration are pruned, and a
//! description is truncated to `DESCRIPTION_LIMIT` characters on a character boundary, so
//! the prompt carries only what the model needs to answer.
use serde_json::{Map, Value};

use crate::model::{CapabilityDescriptor, CapabilityIo};

use super::{dedupe_strings, truncate_text};

/// How many characters of a description are sent to the model.
pub(super) const DESCRIPTION_LIMIT: usize = 240;
/// The system instruction sent with every match request.
pub(super) const SYSTEM_PROMPT: &str = "Validate whether each candidate Skill output can feed the target input. \
Return JSON only: {\"matches\":[{\"id\":\"c1\",\"direction\":\"forward|reverse\",\"confidence\":0.0,\"reason\":\"optional short reason\"}]}. \
Use \"forward\" when the source output feeds the target input and \"reverse\" when the target output feeds the source. \
Confidence must be between 0 and 1. Omit candidates that cannot feed each other.";
/// The forward direction token.
pub(super) const DIRECTION_FORWARD: &str = "forward";
/// The reverse direction token.
pub(super) const DIRECTION_REVERSE: &str = "reverse";

/// Builds the compact request batch from `labelled` `(id, source, target)` triples.
///
/// The id is the batch label the reply must name (`c1`, `c2`, …); the two descriptors are
/// the candidate's producing and consuming capabilities.
pub(super) fn request_json(
    labelled: &[(String, &CapabilityDescriptor, &CapabilityDescriptor)],
) -> Value {
    let candidates: Vec<Value> = labelled
        .iter()
        .map(|(id, source, target)| {
            let mut object = Map::new();
            object.insert("id".to_string(), Value::String(id.clone()));
            object.insert("source".to_string(), brief(source));
            object.insert("target".to_string(), brief(target));
            let directions = directions(source, target);
            if directions.as_object().is_some_and(|map| !map.is_empty()) {
                object.insert("directions".to_string(), directions);
            }
            Value::Object(object)
        })
        .collect();
    let mut root = Map::new();
    root.insert("candidates".to_string(), Value::Array(candidates));
    Value::Object(root)
}

/// The short source/target description sent to the model.
pub(super) fn brief(capability: &CapabilityDescriptor) -> Value {
    let mut object = Map::new();
    insert_text(&mut object, "name", &capability.name);
    let description = truncate_text(&capability.description, DESCRIPTION_LIMIT);
    insert_text(&mut object, "description", description);
    Value::Object(object)
}

/// Both directions of one candidate, each pruned to its non-empty parts.
pub(super) fn directions(source: &CapabilityDescriptor, target: &CapabilityDescriptor) -> Value {
    let mut object = Map::new();
    let forward = direction(&source.outputs, &target.inputs);
    if forward.as_object().is_some_and(|map| !map.is_empty()) {
        object.insert(DIRECTION_FORWARD.to_string(), forward);
    }
    let reverse = direction(&target.outputs, &source.inputs);
    if reverse.as_object().is_some_and(|map| !map.is_empty()) {
        object.insert(DIRECTION_REVERSE.to_string(), reverse);
    }
    Value::Object(object)
}

/// One direction: the produced outputs, the consumed inputs, and their port names.
pub(super) fn direction(outputs: &[CapabilityIo], inputs: &[CapabilityIo]) -> Value {
    let mut object = Map::new();
    let produced = io_values(outputs);
    if !produced.is_empty() {
        object.insert("outputs".to_string(), Value::Array(produced));
    }
    let consumed = io_values(inputs);
    if !consumed.is_empty() {
        object.insert("inputs".to_string(), Value::Array(consumed));
    }
    let ports = dedupe_strings(outputs.iter().chain(inputs).map(|io| io.name.clone()));
    if !ports.is_empty() {
        object.insert(
            "ports".to_string(),
            Value::Array(ports.into_iter().map(Value::String).collect()),
        );
    }
    Value::Object(object)
}

/// The pruned `{name, type}` list of one IO declaration list.
pub(super) fn io_values(ios: &[CapabilityIo]) -> Vec<Value> {
    ios.iter()
        .filter_map(|io| {
            let mut object = Map::new();
            insert_text(&mut object, "name", &io.name);
            insert_text(&mut object, "type", &io.io_type);
            if object.is_empty() {
                None
            } else {
                Some(Value::Object(object))
            }
        })
        .collect()
}

/// Inserts `text` under `key` unless it is blank.
pub(super) fn insert_text(object: &mut Map<String, Value>, key: &str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    object.insert(key.to_string(), Value::String(text.to_string()));
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn entry() -> Value {
        let source = CapabilityDescriptor::new("tool", "a", "Extract")
            .with_description("Extracts text from a document")
            .with_outputs(vec![CapabilityIo::new("text", "string")]);
        let target = CapabilityDescriptor::new("tool", "b", "Summarise")
            .with_inputs(vec![CapabilityIo::new("text", "string")]);
        directions(&source, &target)
    }

    #[test]
    fn brief_prunes_blank_fields_and_truncates_descriptions() {
        let capability = CapabilityDescriptor::new("tool", "a", "Name")
            .with_description("x".repeat(DESCRIPTION_LIMIT + 40));
        let value = brief(&capability);
        let object = value.as_object().unwrap();
        assert_eq!(object.len(), 2);
        assert_eq!(
            object["description"].as_str().unwrap().chars().count(),
            DESCRIPTION_LIMIT
        );
    }

    #[test]
    fn directions_carry_outputs_inputs_and_ports() {
        let value = entry();
        let object = value.as_object().unwrap();
        let forward = object[DIRECTION_FORWARD].as_object().unwrap();
        assert_eq!(forward["outputs"][0]["name"], "text");
        assert_eq!(forward["ports"][0], "text");
        // The reverse direction has no outputs and no inputs, so it is pruned.
        assert!(!object.contains_key(DIRECTION_REVERSE));

        let source = CapabilityDescriptor::new("tool", "a", "A")
            .with_inputs(vec![CapabilityIo::new("in", "string")]);
        let target = CapabilityDescriptor::new("tool", "b", "B")
            .with_outputs(vec![CapabilityIo::new("out", "string")]);
        let reverse_only = directions(&source, &target);
        let object = reverse_only.as_object().unwrap();
        assert!(!object.contains_key(DIRECTION_FORWARD));
        let reverse = object[DIRECTION_REVERSE].as_object().unwrap();
        assert_eq!(reverse["outputs"][0]["name"], "out");
        assert_eq!(reverse["ports"][0], "out");
    }

    #[test]
    fn insert_text_skips_blank_values() {
        let mut object = Map::new();
        insert_text(&mut object, "name", "  ");
        assert!(object.is_empty());
        insert_text(&mut object, "name", " value ");
        assert_eq!(object["name"], "value");
    }
}
