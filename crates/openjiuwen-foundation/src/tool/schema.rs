//! JSON-Schema derivation, default filling, validation, and tool-list projection.
//!
//! Ports the JSON-Schema half of `openjiuwen/core/common/utils/schema_utils.py`
//! (`format_with_schema`, `validate_with_schema`) and the argument handling of
//! `openjiuwen/core/foundation/tool/function/function.py`.
//!
//! Schemas are produced by `schemars` from Rust types or written as literal JSON
//! by closures and remote tools; both paths meet in [`validate_inputs`] and
//! [`apply_defaults`] before a handler ever runs.
//!
//! ```
//! use openjiuwen_foundation::tool::schema::ensure_object_schema;
//! use serde_json::json;
//!
//! assert_eq!(ensure_object_schema(json!({})), json!({"type": "object"}));
//! ```

use jsonschema::validator_for;
use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde_json::{json, Value};

use super::card::ToolCard;

/// Derives the JSON Schema of a `JsonSchema` type as a plain JSON value.
///
/// This is the single schema entry point the `#[tool]` macro targets. The
/// top-level `$schema` key is removed (Python tools carry no dialect marker) and
/// `$defs` is preserved for types that need it. The result is normalized with
/// [`ensure_object_schema`], so it always describes an object.
///
/// A serialization failure — practically impossible, since a generated schema is
/// plain JSON — falls back to `{}` and is logged at debug level rather than
/// panicking.
#[must_use]
pub fn schema_for<T: JsonSchema>() -> Value {
    let schema = schemars::SchemaGenerator::default().into_root_schema_for::<T>();
    match serde_json::to_value(&schema) {
        Ok(mut value) => {
            if let Some(object) = value.as_object_mut() {
                object.remove("$schema");
            }
            ensure_object_schema(value)
        }
        Err(error) => {
            tracing::debug!(
                target: "openjiuwen_foundation::tool::schema",
                %error,
                "failed to serialize generated tool schema"
            );
            json!({})
        }
    }
}

/// Normalizes a schema so it describes an object.
///
/// A non-object schema (`true`, `false`, `null`, or a scalar) becomes
/// `{"type": "object"}`; an object schema without `"type"` gets
/// `"type": "object"` inserted. Existing `properties`, `required`, and `$defs`
/// are preserved, and a schema that already declares a `"type"` is returned
/// unchanged.
///
/// Deliberately no `additionalProperties: false` is added: Python rejects every
/// argument for an empty schema, but Rust closures and remote tools take
/// arbitrary `Value` input, so the default stays permissive (documented
/// deviation in `src/tool/CONTEXT.md`).
#[must_use]
pub fn ensure_object_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut object) => {
            object
                .entry("type")
                .or_insert_with(|| Value::String("object".to_owned()));
            Value::Object(object)
        }
        _ => json!({"type": "object"}),
    }
}

/// Fills schema defaults into `inputs`, in place.
///
/// For an object `inputs` and an object `schema` with `properties`, every
/// property carrying a `"default"` is inserted when `inputs` lacks the key.
/// Keys present in both recurse into the nested object, so nested defaults are
/// filled without overriding provided values; a nested key whose provided value
/// is not an object is left untouched. Non-object inputs and schemas are a
/// no-op.
pub fn apply_defaults(schema: &Value, inputs: &mut Value) {
    let Some(input_object) = inputs.as_object_mut() else {
        return;
    };
    let Some(properties) = schema
        .as_object()
        .and_then(|schema| schema.get("properties"))
        .and_then(Value::as_object)
    else {
        return;
    };

    for (key, property_schema) in properties {
        if let Some(existing) = input_object.get_mut(key) {
            apply_defaults(property_schema, existing);
            continue;
        }
        if let Some(default) = property_schema.get("default") {
            input_object.insert(key.clone(), default.clone());
        }
    }
}

/// Validates `inputs` against `schema`.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the schema itself cannot be compiled into
/// a validator (the message names the schema as invalid), or when `inputs` does
/// not match it. Mismatch messages include the failing instance path, rendered
/// as `/` for the root.
///
/// ```
/// use openjiuwen_foundation::tool::schema::validate_inputs;
/// use serde_json::json;
///
/// let schema = json!({"type": "object", "required": ["name"]});
/// assert!(validate_inputs(&schema, &json!({"name": "bob"})).is_ok());
/// assert!(validate_inputs(&schema, &json!({})).is_err());
/// ```
pub fn validate_inputs(schema: &Value, inputs: &Value) -> Result<()> {
    let validator = validator_for(schema)
        .map_err(|error| Error::validation(format!("invalid input schema: {error}")))?;

    validator.validate(inputs).map_err(|error| {
        let path = error.instance_path.as_str();
        let path = if path.is_empty() { "/" } else { path };
        Error::validation(format!("input does not match schema at {path}: {error}"))
    })
}

/// Projects a list of cards onto the `OpenAI` `tools` array, in order.
#[must_use]
pub fn openai_tools(cards: &[ToolCard]) -> Value {
    Value::Array(cards.iter().map(ToolCard::openai_tool).collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    // port of openjiuwen/core/common/utils/schema_utils.py

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Point {
        x: i64,
        y: i64,
        label: Option<String>,
    }

    #[test]
    fn schema_for_derives_an_object_schema_without_a_dialect_marker() {
        let schema = schema_for::<Point>();

        assert_eq!(schema["type"], "object");
        assert!(schema.get("$schema").is_none());
        assert_eq!(schema["properties"]["x"]["type"], "integer");
        assert!(schema["properties"].get("label").is_some());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("x")));
        assert!(required.contains(&json!("y")));
        assert!(
            !required.contains(&json!("label")),
            "an Option field must not be required"
        );
        assert!(
            schema.get("additionalProperties").is_none(),
            "the permissive default must stay untouched"
        );
    }

    #[test]
    fn ensure_object_schema_normalizes_non_object_schemas() {
        assert_eq!(ensure_object_schema(json!(true)), json!({"type": "object"}));
        assert_eq!(
            ensure_object_schema(json!(false)),
            json!({"type": "object"})
        );
        assert_eq!(ensure_object_schema(Value::Null), json!({"type": "object"}));
        assert_eq!(
            ensure_object_schema(json!("string")),
            json!({"type": "object"})
        );
        assert_eq!(ensure_object_schema(json!({})), json!({"type": "object"}));
    }

    #[test]
    fn ensure_object_schema_preserves_fields_and_declared_types() {
        let schema = json!({
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "$defs": {"Name": {"type": "string"}},
        });
        let ensured = ensure_object_schema(schema.clone());

        assert_eq!(ensured["type"], "object");
        assert_eq!(ensured["properties"], schema["properties"]);
        assert_eq!(ensured["required"], schema["required"]);
        assert_eq!(ensured["$defs"], schema["$defs"]);
        assert!(ensured.get("additionalProperties").is_none());

        let typed = json!({"type": "string", "minLength": 1});
        assert_eq!(ensure_object_schema(typed.clone()), typed);
    }

    #[test]
    fn apply_defaults_fills_nested_defaults_without_overriding_values() {
        let schema = json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "default": "fast"},
                "label": {"type": "string", "default": "none"},
                "options": {
                    "type": "object",
                    "properties": {
                        "retries": {"type": "integer", "default": 3},
                        "verbose": {"type": "boolean", "default": false},
                        "nested": {
                            "type": "object",
                            "properties": {"limit": {"type": "integer", "default": 10}}
                        }
                    }
                }
            }
        });
        let mut inputs = json!({
            "options": {"verbose": true, "nested": {}},
            "label": "given",
            "unknown": 1,
        });

        apply_defaults(&schema, &mut inputs);

        assert_eq!(inputs["mode"], "fast");
        assert_eq!(inputs["label"], "given");
        assert_eq!(inputs["options"]["retries"], 3);
        assert_eq!(inputs["options"]["verbose"], true);
        assert_eq!(inputs["options"]["nested"]["limit"], 10);
        assert_eq!(inputs["unknown"], 1);
    }

    #[test]
    fn apply_defaults_leaves_non_object_values_untouched() {
        let schema = json!({
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {"retries": {"type": "integer", "default": 3}}
                }
            }
        });

        let mut scalar = json!("not an object");
        apply_defaults(&schema, &mut scalar);
        assert_eq!(scalar, json!("not an object"));

        let mut nested_scalar = json!({"options": 5});
        apply_defaults(&schema, &mut nested_scalar);
        assert_eq!(nested_scalar, json!({"options": 5}));

        let mut no_properties = json!({});
        apply_defaults(&json!({"type": "object"}), &mut no_properties);
        assert_eq!(no_properties, json!({}));
    }

    #[test]
    fn validate_inputs_accepts_valid_and_reports_paths_for_invalid() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });

        assert!(validate_inputs(&schema, &json!({"name": "bob", "age": 3})).is_ok());

        let missing = validate_inputs(&schema, &json!({})).unwrap_err();
        assert_eq!(missing.kind(), ErrorKind::Validation);
        assert!(missing.to_string().contains('/'), "{missing}");

        let wrong_type = validate_inputs(&schema, &json!({"name": 5})).unwrap_err();
        assert_eq!(wrong_type.kind(), ErrorKind::Validation);
        assert!(
            wrong_type.to_string().contains("/name"),
            "the message must name the failing path: {wrong_type}"
        );

        let root =
            validate_inputs(&json!({"type": "object"}), &json!("not an object")).unwrap_err();
        assert!(root.to_string().contains("at /"), "{root}");
    }

    #[test]
    fn validate_inputs_reports_an_uncompilable_schema_as_validation() {
        let broken = json!({"$ref": "#/$defs/missing"});
        let error = validate_inputs(&broken, &json!({})).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid input schema"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn openai_tools_projects_cards_in_order() {
        let cards = [
            ToolCard::new("a", "first", json!({})),
            ToolCard::new("b", "second", json!({"type": "object"})),
        ];
        let tools = openai_tools(&cards);

        let array = tools.as_array().unwrap();
        assert_eq!(array.len(), 2);
        assert_eq!(array[0]["type"], "function");
        assert_eq!(array[0]["function"]["name"], "a");
        assert_eq!(
            array[0]["function"]["parameters"],
            json!({"type": "object"})
        );
        assert_eq!(array[1]["function"]["name"], "b");
        assert_eq!(openai_tools(&[]), json!([]));
    }
}
