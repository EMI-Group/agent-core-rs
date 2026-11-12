//! Card-schema construction for the todo tools.
//!
//! `card_schema` exists because `support::schema_for` strips `$defs`, which would leave the
//! generated `$ref`s to the named sub-types dangling; it inlines those definitions instead.

use schemars::JsonSchema;
use serde_json::{json, Map, Value};

use super::create::TodoCreateArgs;
use super::get::TodoGetArgs;
use super::list::TodoListArgs;
use super::modify::{TodoAction, TodoModifyArgs};
use crate::schema::TodoItem;

/// The four Python todo statuses, in card order.
const TODO_STATUS_VALUES: [&str; 4] = ["pending", "in_progress", "completed", "cancelled"];
/// Required item fields of a `todo_create` entry.
const CREATE_ITEM_REQUIRED: [&str; 4] = ["id", "content", "activeForm", "description"];
/// Required item fields of a `todo_modify` entry.
const MODIFY_ITEM_REQUIRED: [&str; 1] = ["id"];
/// Root keys under which `schemars` collects the named sub-schemas a `$ref` points at.
const DEFINITION_KEYS: [&str; 2] = ["$defs", "definitions"];

/// A JSON object as produced by `serde_json`.
pub(super) type JsonObject = Map<String, Value>;

/// Derives the model-facing schema of `T` with every `$ref` inlined.
///
/// `support::schema_for` drops `$defs`, which would leave the references to the named sub-types
/// ([`TodoItem`], [`TodoStatus`], [`TodoData`]) dangling. The todo cards therefore inline
/// those definitions — a `$ref`'s sibling keys, such as a property description, win — then
/// drop the schema-only `default` values Python never publishes, rewrite every `status`
/// property as Python's four-value enum, and apply `support::schema_for`'s top-level normalization.
fn card_schema<T: JsonSchema>() -> Value {
    let generated = schemars::SchemaGenerator::default().into_root_schema_for::<T>();
    let Ok(mut schema) = serde_json::to_value(&generated) else {
        tracing::debug!("failed to serialize generated todo schema");
        return json!({"type": "object", "properties": {}});
    };
    let definitions = take_definitions(&mut schema);
    let mut schema = inline_refs(schema, &definitions, &mut Vec::new());
    strip_defaults(&mut schema);
    normalize_statuses(&mut schema);
    normalize_root(&mut schema);
    schema
}

/// Removes the named sub-schemas of a generated schema and returns them as one map.
fn take_definitions(schema: &mut Value) -> JsonObject {
    let mut definitions = JsonObject::new();
    let Some(root) = schema.as_object_mut() else {
        return definitions;
    };
    for key in DEFINITION_KEYS {
        if let Some(Value::Object(named)) = root.remove(key) {
            definitions.extend(named);
        }
    }
    definitions
}

/// Replaces every `$ref` of `value` with the definition it names.
fn inline_refs(value: Value, definitions: &JsonObject, visiting: &mut Vec<String>) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| inline_refs(item, definitions, visiting))
                .collect(),
        ),
        Value::Object(object) => inline_object(object, definitions, visiting),
        other => other,
    }
}

/// Inlines the `$ref` of one JSON object, if it carries one.
///
/// A reference that cannot be resolved (unknown name, or a cycle back into a definition that
/// is already being expanded) collapses to a plain object schema merged with the `$ref`'s
/// sibling keys, so a card never keeps a dangling `$ref`.
fn inline_object(
    mut object: JsonObject,
    definitions: &JsonObject,
    visiting: &mut Vec<String>,
) -> Value {
    let Some(reference) = object.remove("$ref") else {
        return Value::Object(
            object
                .into_iter()
                .map(|(key, value)| (key, inline_refs(value, definitions, visiting)))
                .collect(),
        );
    };
    let siblings: JsonObject = object
        .into_iter()
        .map(|(key, value)| (key, inline_refs(value, definitions, visiting)))
        .collect();
    let name = reference.as_str().and_then(definition_name);
    let Some(name) = name.filter(|name| !visiting.iter().any(|seen| seen == name)) else {
        return plain_object(siblings);
    };
    let Some(definition) = definitions.get(name).cloned() else {
        return plain_object(siblings);
    };
    visiting.push(name.to_owned());
    let resolved = inline_refs(definition, definitions, visiting);
    visiting.pop();
    match resolved {
        Value::Object(mut resolved) => {
            resolved.extend(siblings);
            Value::Object(resolved)
        }
        resolved if siblings.is_empty() => resolved,
        resolved => {
            let mut wrapper = siblings;
            wrapper.insert("allOf".to_owned(), Value::Array(vec![resolved]));
            Value::Object(wrapper)
        }
    }
}

/// Returns the definition name of a local reference, e.g. `TodoItem` for `#/$defs/TodoItem`.
fn definition_name(reference: &str) -> Option<&str> {
    ["#/$defs/", "#/definitions/"]
        .into_iter()
        .find_map(|prefix| reference.strip_prefix(prefix))
        .filter(|name| !name.is_empty())
}

/// The fallback for a reference that cannot be inlined: a plain object schema.
fn plain_object(mut siblings: JsonObject) -> Value {
    siblings
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("object".to_owned()));
    Value::Object(siblings)
}

/// Removes every `default` from `value`; the Python todo cards publish none.
fn strip_defaults(value: &mut Value) {
    match value {
        Value::Object(object) => {
            let _ = object.remove("default");
            for child in object.values_mut() {
                strip_defaults(child);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                strip_defaults(item);
            }
        }
        _ => {}
    }
}

/// Rewrites every `status` property of `value` as Python's four-value string enum.
fn normalize_statuses(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::Object(properties)) = object.get_mut("properties") {
                for (name, property) in properties.iter_mut() {
                    if name == "status" {
                        *property = status_property(property);
                    }
                }
            }
            for child in object.values_mut() {
                normalize_statuses(child);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                normalize_statuses(item);
            }
        }
        _ => {}
    }
}

/// Builds Python's `status` property, keeping the description of the generated one.
fn status_property(property: &Value) -> Value {
    let mut normalized = JsonObject::new();
    if let Some(description) = property.get("description").cloned() {
        normalized.insert("description".to_owned(), description);
    }
    normalized.insert("type".to_owned(), Value::String("string".to_owned()));
    normalized.insert("enum".to_owned(), json!(TODO_STATUS_VALUES));
    Value::Object(normalized)
}

/// Applies the top-level normalization of `support::schema_for` to `value`.
fn normalize_root(schema: &mut Value) {
    let Some(root) = schema.as_object_mut() else {
        *schema = json!({"type": "object", "properties": {}});
        return;
    };
    for key in ["$schema", "title"] {
        root.remove(key);
    }
    root.entry("type".to_owned())
        .or_insert_with(|| Value::String("object".to_owned()));
    root.entry("properties".to_owned())
        .or_insert_with(|| json!({}));
    if matches!(root.get("required"), Some(Value::Array(required)) if required.is_empty()) {
        root.remove("required");
    }
}

/// Returns property `name` of a card schema.
fn property_mut<'a>(schema: &'a mut Value, name: &str) -> Option<&'a mut Value> {
    schema.get_mut("properties")?.as_object_mut()?.get_mut(name)
}

/// Sets the `items` sub-schema of property `property` inside one object schema.
fn set_items_in_object(object: &mut JsonObject, property: &str, items: Value) {
    if let Some(Value::Object(target)) = object.get_mut(property) {
        target.insert("items".to_owned(), items);
    }
}

/// Sets the `items` sub-schema of property `property` of a card schema.
fn set_property_items(schema: &mut Value, property: &str, items: Value) {
    if let Some(Value::Object(target)) = property_mut(schema, property) {
        target.insert("items".to_owned(), items);
    }
}

/// Replaces the top-level `required` list of a card schema.
fn require(schema: &mut Value, fields: &[&str]) {
    if let Some(object) = schema.as_object_mut() {
        object.insert("required".to_owned(), json!(fields));
    }
}

/// The shared `TodoItem` sub-schema of the `todos` and `todo_data.items` arrays.
///
/// Python publishes every item field and only the `required` list differs between
/// `todo_create` (`id`, `content`, `activeForm`, `description`) and `todo_modify` (`id`).
fn item_schema(required: &[&str]) -> Value {
    let mut schema = card_schema::<TodoItem>();
    if let Some(object) = schema.as_object_mut() {
        // Python's item schema has no description of its own; the property descriptions stay.
        let _ = object.remove("description");
        object.insert("required".to_owned(), json!(required));
    }
    schema
}

/// Card schema of `todo_create`: `todos` holds items requiring the four creation fields.
pub(super) fn todo_create_schema() -> Value {
    let mut schema = card_schema::<TodoCreateArgs>();
    set_property_items(&mut schema, "todos", item_schema(&CREATE_ITEM_REQUIRED));
    schema
}

/// Card schema of `todo_list`, which takes no arguments.
pub(super) fn todo_list_schema() -> Value {
    card_schema::<TodoListArgs>()
}

/// Card schema of `todo_get`.
pub(super) fn todo_get_schema() -> Value {
    card_schema::<TodoGetArgs>()
}

/// Card schema of `todo_modify`: Python's `action` enum and id-only item objects.
pub(super) fn todo_modify_schema() -> Value {
    let mut schema = card_schema::<TodoModifyArgs>();
    set_property_items(&mut schema, "todos", item_schema(&MODIFY_ITEM_REQUIRED));
    set_todo_data_items(&mut schema, item_schema(&MODIFY_ITEM_REQUIRED));
    set_action_enum(&mut schema);
    require(&mut schema, &["action"]);
    schema
}

/// Sets the `todo_data.items` sub-schema, which lives inside the property's `anyOf` branches.
fn set_todo_data_items(schema: &mut Value, items: Value) {
    let Some(todo_data) = property_mut(schema, "todo_data") else {
        return;
    };
    let Some(object) = todo_data.as_object_mut() else {
        return;
    };
    if let Some(Value::Array(branches)) = object.get_mut("anyOf") {
        for branch in branches.iter_mut().filter(|branch| branch.is_object()) {
            set_property_items(branch, "items", items.clone());
        }
    } else {
        set_items_in_object(object, "items", items);
    }
}

/// Adds Python's six-value `action` enum to the `todo_modify` schema, keeping its description.
fn set_action_enum(schema: &mut Value) {
    let Some(Value::Object(action)) = property_mut(schema, "action") else {
        return;
    };
    let mut normalized = JsonObject::new();
    if let Some(description) = action.remove("description") {
        normalized.insert("description".to_owned(), description);
    }
    normalized.insert("type".to_owned(), Value::String("string".to_owned()));
    normalized.insert(
        "enum".to_owned(),
        Value::Array(
            TodoAction::ALL
                .iter()
                .map(|action| Value::String(action.as_str().to_owned()))
                .collect(),
        ),
    );
    *action = normalized;
}
