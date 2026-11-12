//! Conversion between the kernel agent card and the A2A agent card.
//!
//! Python parity: `A2AAgentCardAdapter` in
//! `openjiuwen/extensions/a2a/a2a_agentcard_adapter.py` (v0.1.20). The Python
//! adapter maps `openjiuwen.core.single_agent.schema.agent_card.AgentCard` to
//! the `a2a-sdk` `AgentCard` and back; this module owns the same two
//! directions over [`types::AgentCard`].
//!
//! # Description sections
//!
//! The A2A card has no field for the kernel card's `input_params` /
//! `output_params` schemas, so the adapter embeds them in the free-form
//! description, one section per payload:
//!
//! ```text
//! <trimmed base description>
//! [input_params] {"type": "object"}
//! [output_params] {"type": "string"}
//! ```
//!
//! Sections are joined with `"\n"` and empty sections are dropped, so a card
//! without a base description yields only the parameter sections, and a card
//! with neither parameters yields just the trimmed description. See
//! [`build_description`].
//!
//! # Payload formatting
//!
//! A payload is formatted exactly like Python's
//! `json.dumps(value, ensure_ascii=False, sort_keys=True)`
//! ([`serialize_param_payload`]): object keys are sorted recursively at every
//! nesting level, separators are `": "` and `", "` (a space after each `:` and
//! `,`, none inside empty `{}` / `[]`), non-ASCII characters are emitted
//! verbatim rather than `\uXXXX`-escaped, and `null` / `true` / `false` are
//! literal. This workspace enables `serde_json`'s `preserve_order` feature, so
//! `serde_json::to_string` keeps insertion order and uses compact separators;
//! that is why this module writes the Python shape with a small recursive
//! serializer instead of delegating to it.
//!
//! # Lossiness
//!
//! [`AgentCardAdapter::from_a2a`] is deliberately lossy, matching Python: only
//! `name` and `description` are copied, and a fresh kernel card id is generated
//! by `AgentCard::default()`. Interfaces, capabilities, skills, provider, and
//! security metadata from the A2A card are dropped, and a description that
//! carries embedded `[input_params]` / `[output_params]` sections is copied
//! as-is (the sections are not parsed back into the kernel fields).
//!
//! [`AgentCardAdapter::to_a2a`] likewise leaves `version` empty and `skills`
//! empty, and sets no `provider`: Python's adapter does not populate them
//! either, and this port keeps that shape so both runtimes emit the same card
//! JSON for the same kernel card.

use openjiuwen_core::single_agent::AgentCard as KernelAgentCard;
use serde_json::Value;

use crate::a2a::types;

/// The media types an A2A card advertises as accepted by default.
///
/// Python parity: `A2AAgentCardAdapter.DEFAULT_INPUT_MODES`.
pub const DEFAULT_INPUT_MODES: [&str; 2] = ["text/plain", "application/json"];

/// The media types an A2A card advertises as produced by default.
///
/// Python parity: `A2AAgentCardAdapter.DEFAULT_OUTPUT_MODES`. The Python value
/// is identical to `DEFAULT_INPUT_MODES`.
pub const DEFAULT_OUTPUT_MODES: [&str; 2] = ["text/plain", "application/json"];

/// Writes one JSON string literal with Python's `ensure_ascii=False` escaping.
///
/// Python's `json` encoder escapes exactly `"`, `\`, and the control characters
/// below `0x20` (`\b`, `\t`, `\n`, `\f`, `\r`, else `\u00XX` with lowercase
/// hex); everything else, including non-ASCII text and `0x7f`, is emitted
/// verbatim. `serde_json`'s encoder follows the same table, but this helper
/// keeps the serialization free of fallible conversions.
fn write_json_string(value: &str, out: &mut String) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0a}' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0d}' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                // Every remaining control character is `\u00XX`, lowercase hex.
                let code = c as u32;
                out.push_str("\\u00");
                if let Some(high) = char::from_digit(code >> 4, 16) {
                    out.push(high);
                }
                if let Some(low) = char::from_digit(code & 0xf, 16) {
                    out.push(low);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Appends `value` rendered like Python's `json.dumps(..., ensure_ascii=False,
/// sort_keys=True)` to `out`.
///
/// Object keys are sorted recursively; arrays keep their order; separators are
/// `": "` and `", "`; empty containers render as `{}` and `[]`.
fn write_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_json_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            // Python sorts strings by code point; UTF-8 byte order is the same.
            keys.sort_unstable();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_json_string(key, out);
                out.push_str(": ");
                if let Some(item) = map.get(key) {
                    write_python_json(item, out);
                }
            }
            out.push('}');
        }
    }
}

/// Serializes a parameter payload exactly like Python's
/// `json.dumps(value, ensure_ascii=False, sort_keys=True)`.
///
/// Object keys are sorted recursively, separators are `": "` and `", "`,
/// and non-ASCII text is preserved verbatim. Non-object values (arrays,
/// strings, numbers, booleans, `null`) are serialized as-is.
///
/// Python's `_serialize_param_payload` also coerces non-JSON parameter objects
/// (dataclasses, schemas, classes) into a JSON payload; here the kernel card
/// already stores `serde_json::Value`, so there is nothing to coerce. `None`
/// maps to the empty string in Python; in Rust the absence is modelled by
/// `Option` at the call sites ([`build_description`]).
///
/// # Examples
///
/// ```
/// use openjiuwen_extensions::a2a::card::serialize_param_payload;
/// use serde_json::json;
///
/// let payload = serialize_param_payload(&json!({"b": [1, 2], "a": "héllo"}));
/// assert_eq!(payload, r#"{"a": "héllo", "b": [1, 2]}"#);
/// ```
#[must_use]
pub fn serialize_param_payload(value: &Value) -> String {
    let mut out = String::new();
    write_python_json(value, &mut out);
    out
}

/// Builds the A2A card description from the base description and the
/// parameter schemas.
///
/// The trimmed base description comes first (dropped when empty), followed by
/// a `[input_params] {payload}` section when `input_params` is `Some`, then an
/// `[output_params] {payload}` section when `output_params` is `Some`. Sections
/// are joined with `"\n"` and the result is trimmed.
///
/// Python parity: `A2AAgentCardAdapter._build_description`.
///
/// # Examples
///
/// ```
/// use openjiuwen_extensions::a2a::card::build_description;
/// use serde_json::json;
///
/// let input = json!({"type": "object"});
/// let description = build_description("  translates  ", Some(&input), None);
/// assert_eq!(description, "translates\n[input_params] {\"type\": \"object\"}");
/// ```
#[must_use]
pub fn build_description(
    base_description: &str,
    input_params: Option<&Value>,
    output_params: Option<&Value>,
) -> String {
    let mut sections: Vec<String> = Vec::with_capacity(3);
    let base = base_description.trim();
    if !base.is_empty() {
        sections.push(base.to_owned());
    }
    if let Some(params) = input_params {
        sections.push(format!(
            "[input_params] {}",
            serialize_param_payload(params)
        ));
    }
    if let Some(params) = output_params {
        sections.push(format!(
            "[output_params] {}",
            serialize_param_payload(params)
        ));
    }
    sections.join("\n").trim().to_owned()
}

/// Adapts kernel agent cards to A2A agent cards and back.
///
/// Python parity: `A2AAgentCardAdapter` in
/// `openjiuwen/extensions/a2a/a2a_agentcard_adapter.py`.
pub struct AgentCardAdapter;

impl AgentCardAdapter {
    /// The media types an A2A card advertises as accepted by default.
    ///
    /// Python parity: `A2AAgentCardAdapter.DEFAULT_INPUT_MODES`.
    pub const DEFAULT_INPUT_MODES: [&str; 2] = DEFAULT_INPUT_MODES;

    /// The media types an A2A card advertises as produced by default.
    ///
    /// Python parity: `A2AAgentCardAdapter.DEFAULT_OUTPUT_MODES`.
    pub const DEFAULT_OUTPUT_MODES: [&str; 2] = DEFAULT_OUTPUT_MODES;

    /// Converts a kernel [`openjiuwen_core::single_agent::AgentCard`] into an
    /// A2A [`types::AgentCard`].
    ///
    /// Python parity: `A2AAgentCardAdapter.to_a2a_agent_card`.
    ///
    /// - `name` is copied verbatim (possibly empty).
    /// - `description` is [`build_description`] over the card's description and
    ///   parameter schemas.
    /// - `capabilities` advertise streaming and no push notifications.
    /// - `default_input_modes` / `default_output_modes` are
    ///   [`DEFAULT_INPUT_MODES`] / [`DEFAULT_OUTPUT_MODES`].
    /// - `supported_interfaces` holds exactly one entry when `interface_url` is
    ///   `Some` and non-empty, and is empty otherwise. `protocol_binding` and
    ///   `protocol_version` are copied verbatim — transport support is enforced
    ///   by the server, not by this adapter — and `tenant` is left unset.
    /// - `version`, `skills`, and `provider` stay at their defaults because
    ///   Python's adapter does not populate them.
    #[must_use]
    pub fn to_a2a(
        card: &KernelAgentCard,
        interface_url: Option<&str>,
        protocol_binding: &str,
        protocol_version: &str,
    ) -> types::AgentCard {
        let supported_interfaces = interface_url
            .filter(|url| !url.is_empty())
            .map(|url| {
                vec![types::AgentInterface {
                    url: url.to_owned(),
                    protocol_binding: protocol_binding.to_owned(),
                    protocol_version: protocol_version.to_owned(),
                    tenant: None,
                }]
            })
            .unwrap_or_default();

        types::AgentCard {
            name: card.name.clone(),
            description: build_description(
                &card.description,
                card.input_params.as_ref(),
                card.output_params.as_ref(),
            ),
            supported_interfaces,
            capabilities: types::AgentCapabilities {
                streaming: true,
                push_notifications: false,
            },
            default_input_modes: Self::DEFAULT_INPUT_MODES
                .iter()
                .map(|mode| (*mode).to_owned())
                .collect(),
            default_output_modes: Self::DEFAULT_OUTPUT_MODES
                .iter()
                .map(|mode| (*mode).to_owned())
                .collect(),
            ..types::AgentCard::default()
        }
    }

    /// Converts an A2A [`types::AgentCard`] into a kernel
    /// [`openjiuwen_core::single_agent::AgentCard`].
    ///
    /// Python parity: `A2AAgentCardAdapter.from_a2a_agent_card`.
    ///
    /// Only `name` and `description` are copied; every other kernel field takes
    /// its default, including a freshly generated uuid hex `id` and empty
    /// `input_params` / `output_params` / `interface_url`. Interfaces,
    /// capabilities, skills, and security metadata from the A2A card are
    /// dropped, and description-embedded parameter sections are kept as plain
    /// text. This lossiness matches Python and means the conversion is not
    /// round-trip safe.
    #[must_use]
    pub fn from_a2a(card: &types::AgentCard) -> KernelAgentCard {
        KernelAgentCard {
            name: card.name.clone(),
            description: card.description.clone(),
            ..KernelAgentCard::default()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    // Python counterpart: tests/unit_tests/extensions/a2a/test_a2a_agentcard_adapter.py
    #[test]
    fn description_without_params_is_trimmed_base_only() {
        assert_eq!(
            build_description("  a helpful agent  ", None, None),
            "a helpful agent"
        );
        assert_eq!(build_description("", None, None), "");
        assert_eq!(build_description("   ", None, None), "");
    }

    #[test]
    fn description_joins_sections_in_order() {
        let input = json!({"type": "object"});
        let output = json!({"type": "string"});
        assert_eq!(
            build_description("translate", Some(&input), Some(&output)),
            "translate\n[input_params] {\"type\": \"object\"}\n[output_params] {\"type\": \"string\"}"
        );
        assert_eq!(
            build_description("translate", None, Some(&output)),
            "translate\n[output_params] {\"type\": \"string\"}"
        );
    }

    #[test]
    fn empty_base_description_yields_only_sections() {
        let input = json!({"b": 2, "a": 1});
        assert_eq!(
            build_description("", Some(&input), None),
            "[input_params] {\"a\": 1, \"b\": 2}"
        );
        assert_eq!(
            build_description("   ", Some(&input), Some(&input)),
            "[input_params] {\"a\": 1, \"b\": 2}\n[output_params] {\"a\": 1, \"b\": 2}"
        );
    }

    #[test]
    fn payload_keys_are_sorted_recursively() {
        let value = json!({
            "z": {"b": 1, "a": [{"y": 1, "x": 2}]},
            "a": true,
        });
        assert_eq!(
            serialize_param_payload(&value),
            "{\"a\": true, \"z\": {\"a\": [{\"x\": 2, \"y\": 1}], \"b\": 1}}"
        );
    }

    #[test]
    fn payload_uses_python_separators_and_preserves_unicode() {
        let value = json!({"b": [1, 2], "a": "héllo", "c": null, "d": false});
        assert_eq!(
            serialize_param_payload(&value),
            "{\"a\": \"héllo\", \"b\": [1, 2], \"c\": null, \"d\": false}"
        );
        assert_eq!(serialize_param_payload(&json!({})), "{}");
        assert_eq!(serialize_param_payload(&json!([])), "[]");
        assert_eq!(
            serialize_param_payload(&json!("\"quoted\"")),
            "\"\\\"quoted\\\"\""
        );
        assert_eq!(
            serialize_param_payload(&json!("line\nbreak\ttab")),
            "\"line\\nbreak\\ttab\""
        );
    }

    #[test]
    fn payload_escapes_control_characters_like_python() {
        assert_eq!(
            serialize_param_payload(&json!("\u{1}\u{1f}")),
            "\"\\u0001\\u001f\""
        );
        assert_eq!(
            serialize_param_payload(&json!({"k": "back\\slash"})),
            "{\"k\": \"back\\\\slash\"}"
        );
    }

    #[test]
    fn non_object_payloads_are_serialized_as_is() {
        assert_eq!(
            serialize_param_payload(&json!([3, {"b": 2, "a": 1}])),
            "[3, {\"a\": 1, \"b\": 2}]"
        );
        assert_eq!(serialize_param_payload(&json!("plain")), "\"plain\"");
        assert_eq!(serialize_param_payload(&json!(42)), "42");
        assert_eq!(serialize_param_payload(&json!(true)), "true");
        assert_eq!(serialize_param_payload(&json!(null)), "null");
    }

    #[test]
    fn to_a2a_maps_name_description_capabilities_and_modes() {
        let card = KernelAgentCard::new("translator")
            .with_description("  translates text  ")
            .with_input_params(json!({"b": 2, "a": 1}))
            .with_output_params(json!({"type": "string"}))
            .with_interface_url("http://localhost/agent");
        let a2a = AgentCardAdapter::to_a2a(&card, Some("http://localhost/a2a"), "JSONRPC", "1.0");

        assert_eq!(a2a.name, "translator");
        assert_eq!(
            a2a.description,
            "translates text\n[input_params] {\"a\": 1, \"b\": 2}\n[output_params] {\"type\": \"string\"}"
        );
        assert!(a2a.capabilities.streaming);
        assert!(!a2a.capabilities.push_notifications);
        assert_eq!(
            a2a.default_input_modes,
            vec!["text/plain", "application/json"]
        );
        assert_eq!(
            a2a.default_output_modes,
            vec!["text/plain", "application/json"]
        );
        assert_eq!(a2a.supported_interfaces.len(), 1);
        let interface = &a2a.supported_interfaces[0];
        assert_eq!(interface.url, "http://localhost/a2a");
        assert_eq!(interface.protocol_binding, "JSONRPC");
        assert_eq!(interface.protocol_version, "1.0");
        assert!(interface.tenant.is_none());
        assert_eq!(a2a.version, "");
        assert!(
            a2a.skills.is_empty(),
            "skills are not populated by the adapter"
        );
        assert!(a2a.provider.is_none());
    }

    #[test]
    fn to_a2a_omits_interface_without_url() {
        let card = KernelAgentCard::new("n");
        assert!(
            AgentCardAdapter::to_a2a(&card, None, "JSONRPC", "1.0")
                .supported_interfaces
                .is_empty(),
            "no interface URL means no advertised interface"
        );
        assert!(
            AgentCardAdapter::to_a2a(&card, Some(""), "JSONRPC", "1.0")
                .supported_interfaces
                .is_empty(),
            "an empty interface URL means no advertised interface"
        );
    }

    #[test]
    fn to_a2a_copies_binding_verbatim_without_filtering() {
        let a2a =
            AgentCardAdapter::to_a2a(&KernelAgentCard::new("n"), Some("http://x"), "gRPC", "0.3");
        assert_eq!(a2a.supported_interfaces[0].protocol_binding, "gRPC");
        assert_eq!(a2a.supported_interfaces[0].protocol_version, "0.3");
    }

    #[test]
    fn from_a2a_copies_only_name_and_description() {
        let a2a = types::AgentCard {
            name: "remote".to_owned(),
            description: "remote agent".to_owned(),
            version: "9.9".to_owned(),
            supported_interfaces: vec![types::AgentInterface {
                url: "http://remote".to_owned(),
                protocol_binding: "JSONRPC".to_owned(),
                protocol_version: "1.0".to_owned(),
                tenant: None,
            }],
            skills: vec![types::AgentSkill {
                id: "s".to_owned(),
                name: "skill".to_owned(),
                ..types::AgentSkill::default()
            }],
            ..types::AgentCard::default()
        };
        let card = AgentCardAdapter::from_a2a(&a2a);
        assert_eq!(card.name, "remote");
        assert_eq!(card.description, "remote agent");
        assert_eq!(card.id.len(), 32);
        assert!(card.id.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(card.input_params.is_none());
        assert!(card.output_params.is_none());
        assert!(card.interface_url.is_none());
    }

    #[test]
    fn from_a2a_generates_distinct_ids() {
        let a2a = types::AgentCard {
            name: "n".to_owned(),
            ..types::AgentCard::default()
        };
        assert_ne!(
            AgentCardAdapter::from_a2a(&a2a).id,
            AgentCardAdapter::from_a2a(&a2a).id
        );
    }

    #[test]
    fn constants_match_python_defaults() {
        assert_eq!(
            AgentCardAdapter::DEFAULT_INPUT_MODES,
            ["text/plain", "application/json"]
        );
        assert_eq!(
            AgentCardAdapter::DEFAULT_OUTPUT_MODES,
            ["text/plain", "application/json"]
        );
    }
}
