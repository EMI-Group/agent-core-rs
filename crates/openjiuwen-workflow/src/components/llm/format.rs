//! Prompt formatting and answer shaping for the `llm` component.
//!
//! This is the formatting half of
//! `openjiuwen/core/workflow/components/llm/llm_comp.py`:
//!
//! - [`format_prompt`] is `LLMPromptFormatter.format_prompt`: it rewrites the
//!   **last** user message with a format instruction while leaving the rest of the
//!   conversation byte-identical.
//! - [`generate_json_schema`] is `SchemaGenerator.generate_json_schema`: it turns an
//!   `output_config` map into the JSON Schema embedded in the prompt.
//! - [`output_config_object`] normalizes the `None`/empty/whole-schema shapes the
//!   two halves have to agree on.
//! - [`shape_output`] is `OutputFormatter.format_response` (with
//!   `_format_text_response` / `_format_json_response`). JSON replies go through
//!   `JsonParser.parse_json_content` and `_clean_markdown_blocks` first, so a fenced
//!   ` ```json … ``` ` answer parses.
//!
//! Nothing here sends a provider-side `response_format` parameter: Python shapes the
//! answer text instead of asking the provider, and so does this module.
//!
//! # Deviations from Python
//!
//! - The generated schema is serialized compactly with [`serde_json::to_string`];
//!   Python's `json.dumps` inserts spaces after `:` and `,`. The schema only ever
//!   appears inside a prompt, so the wire text differs but the contract does not.
//! - `_format_text_response` raises `COMPONENT_LLM_CONFIG_INVALID` (`[101002]`),
//!   which Python's `_create_output` re-wraps as
//!   `COMPONENT_LLM_EXECUTION_PROCESS_ERROR` (`[101004]`). Here the documented
//!   configuration error is returned unchanged.
//! - A field's declared `"type"` is validated only when the field config declares
//!   one. Python validates against the generated schema, which defaults a missing
//!   type to `"string"`.
//! - An object value is filtered by a field's `"properties"` only when those
//!   properties are declared. Python drops every sub-key of an object whose field
//!   config omits `"properties"`.
//! - The whole-schema branch keeps Python's required-field check and property
//!   projection, but not the full recursive validation `SchemaUtils` performs.

use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, MessageContent, Role};

use super::LlmResponseFormat;
use crate::error::{llm_config_error, llm_execution_process_error};

/// Default markdown instruction (Python's `_DEFAULT_MARKDOWN_INSTRUCTION`).
pub(crate) const DEFAULT_MARKDOWN_INSTRUCTION: &str = concat!(
    "Please return the answer in markdown format.\n",
    "- For headings, use number signs (#).\n",
    "- For list items, start with dashes (-).\n",
    "- To emphasize text, wrap it with asterisks (*).\n",
    "- For code or commands, surround them with backticks (`).\n",
    "- For quoted text, use greater than signs (>).\n",
    "- For links, wrap the text in square brackets [], followed by the URL in parentheses ().\n",
    "- For images, use square brackets [] for the alt text, followed by the image URL in parentheses ().\n",
    "The question is: ${query}.",
);

/// Default JSON instruction (Python's `_DEFAULT_JSON_INSTRUCTION`).
pub(crate) const DEFAULT_JSON_INSTRUCTION: &str = concat!(
    "Carefully consider the user's question to ensure your answer is logical and makes sense.\n",
    "- Make sure your explanation is concise and easy to understand, not verbose.\n",
    "- Strictly return the answer in valid JSON format only, and ",
    "\"DO NOT ADD ANY COMMENTS BEFORE OR AFTER IT\" to ensure it could be formatted ",
    "as a JSON instance that conforms to the JSON schema below.\n",
    "Here is the JSON schema: ${json_schema}.\n",
    "The question is: ${query}.",
);

/// Rewrites the last user message with the instruction for `format`.
///
/// Port of `LLMPromptFormatter.format_prompt`. `Text` returns `messages` untouched;
/// `Markdown` and `Json` replace only the content of the **last** [`Role::User`]
/// message (when one exists) and leave every other message byte-identical.
/// `${query}` is replaced by that message's text, and `${json_schema}` by the
/// compactly serialized schema of `output_config`. The schema is substituted before
/// the query, exactly like Python, so a `${query}`-looking substring inside the
/// schema is not corrupted.
pub(crate) fn format_prompt(
    mut messages: Vec<BaseMessage>,
    format: LlmResponseFormat,
    markdown_instruction: Option<&str>,
    json_instruction: Option<&str>,
    output_config: Option<&Value>,
) -> Vec<BaseMessage> {
    if format == LlmResponseFormat::Text {
        return messages;
    }
    let Some(index) = messages
        .iter()
        .rposition(|message| message.role == Role::User)
    else {
        return messages;
    };
    let query = messages[index].text().into_owned();
    let prompt = match format {
        LlmResponseFormat::Markdown => markdown_instruction
            .unwrap_or(DEFAULT_MARKDOWN_INSTRUCTION)
            .replace("${query}", &query),
        LlmResponseFormat::Json => {
            let schema = generate_json_schema(&output_config_object(output_config));
            let schema_text = serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_owned());
            json_instruction
                .unwrap_or(DEFAULT_JSON_INSTRUCTION)
                .replace("${json_schema}", &schema_text)
                .replace("${query}", &query)
        }
        LlmResponseFormat::Text => return messages,
    };
    messages[index].content = MessageContent::Text(prompt);
    messages
}

/// Builds the JSON Schema embedded in a JSON instruction.
///
/// Port of `SchemaGenerator.generate_json_schema`. A config that already is a whole
/// schema (an object whose `"type"` is `"object"`) is returned unchanged. Otherwise
/// every top-level key becomes a property whose type defaults to `"string"`; a
/// configured `"description"` is passed through, as are `"items"` (for arrays) and
/// `"properties"` (for objects). A key is required unless its config sets
/// `"required": false`. Non-object configs never panic: the config is treated as an
/// empty object and a non-object key config as `{}`.
pub(crate) fn generate_json_schema(output_config: &Value) -> Value {
    if is_whole_schema(output_config) {
        return output_config.clone();
    }
    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();
    if let Value::Object(config) = output_config {
        for (name, field_config) in config {
            let field = field_config.as_object();
            let field_type = field
                .and_then(|config| config.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("string");
            let mut property = Map::new();
            property.insert("type".to_owned(), Value::String(field_type.to_owned()));
            if let Some(description) = field.and_then(|config| config.get("description")) {
                property.insert("description".to_owned(), description.clone());
            }
            if field_type == "array" {
                if let Some(items) = field.and_then(|config| config.get("items")) {
                    property.insert("items".to_owned(), items.clone());
                }
            }
            if field_type == "object" {
                if let Some(nested) = field.and_then(|config| config.get("properties")) {
                    property.insert("properties".to_owned(), nested.clone());
                }
            }
            if field_is_required(field) {
                required.push(Value::String(name.clone()));
            }
            properties.insert(name.clone(), Value::Object(property));
        }
    }
    json!({
        "type": "object",
        "properties": Value::Object(properties),
        "required": Value::Array(required),
    })
}

/// Normalizes `output_config` to a JSON object.
///
/// `None`, a non-object value, and an object without keys all become
/// `{"result": {}}` — Python's `outputs_config` default, matching
/// [`super::LlmComponent::output_key`]. A whole-schema config is kept as is.
pub(crate) fn output_config_object(output_config: Option<&Value>) -> Value {
    match output_config {
        Some(Value::Object(map)) if !map.is_empty() => Value::Object(map.clone()),
        _ => json!({"result": {}}),
    }
}

/// The single output key a text/markdown answer is written under.
///
/// # Errors
///
/// Returns [`Error::Workflow`] with code `[101002]` when the normalized config does
/// not name exactly one field (Python's `_format_text_response` rule).
pub(crate) fn single_output_key(output_config: Option<&Value>) -> Result<String> {
    text_output_key(&output_config_object(output_config))
}

/// Turns the model answer into the configured output object.
///
/// Port of `OutputFormatter.format_response`. `Text` and `Markdown` return
/// `{<the single configured key>: content}`; `Json` parses `content` (accepting a
/// fenced code block) and projects it onto the configured fields.
///
/// # Errors
///
/// `[101002]` for the text/markdown one-field rule and `[101004]` for JSON parsing,
/// missing required fields, and type mismatches.
pub(crate) fn shape_output(
    content: &str,
    format: LlmResponseFormat,
    output_config: Option<&Value>,
) -> Result<Value> {
    let config = output_config_object(output_config);
    match format {
        LlmResponseFormat::Text | LlmResponseFormat::Markdown => {
            format_text_response(content, &config)
        }
        LlmResponseFormat::Json => format_json_response(content, &config),
    }
}

/// The one-field rule shared by `shape_output` and the stream pre-check.
fn text_output_key(config: &Value) -> Result<String> {
    let Some(map) = config.as_object() else {
        return Err(text_one_field_error());
    };
    if map.len() != 1 {
        return Err(text_one_field_error());
    }
    map.keys().next().cloned().ok_or_else(text_one_field_error)
}

/// The `[101002]` error for a text/markdown config that is not exactly one field.
fn text_one_field_error() -> Error {
    llm_config_error("text/markdown response type, outputs_config must contain only one field")
}

/// Port of `OutputFormatter._format_text_response`.
fn format_text_response(content: &str, config: &Value) -> Result<Value> {
    let key = text_output_key(config)?;
    Ok(super::LlmComponent::frame(&key, content))
}

/// Port of `OutputFormatter._format_json_response`.
fn format_json_response(content: &str, config: &Value) -> Result<Value> {
    let parsed = parse_json_content(content)?;
    let Value::Object(parsed) = parsed else {
        return Err(llm_execution_process_error(&format!(
            "Json parse error: {content}"
        )));
    };
    if is_whole_schema(config) {
        return project_schema_output(&parsed, config);
    }
    extract_configured_fields(&parsed, config)
}

/// Port of `JsonParser.parse_json_content`: strip a fenced block, then parse.
fn parse_json_content(content: &str) -> Result<Value> {
    let cleaned = clean_markdown_blocks(content);
    serde_json::from_str::<Value>(&cleaned)
        .map_err(|_| llm_execution_process_error(&format!("Json parse error: {content}")))
}

/// Port of `JsonParser._clean_markdown_blocks`.
///
/// Trims the text and, when it starts and ends with a fence, drops the opening
/// line (including a ` ```json ` info string) and the trailing ` ``` ` line.
fn clean_markdown_blocks(content: &str) -> String {
    let content = content.trim();
    if !(content.starts_with("```") && content.ends_with("```")) {
        return content.to_owned();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if lines.first().is_some_and(|line| line.starts_with("```")) {
        lines.remove(0);
    }
    if lines.last().is_some_and(|line| *line == "```") {
        lines.pop();
    }
    lines.join("\n").trim().to_owned()
}

/// Port of `OutputFormatter._extract_configured_fields` (generated-schema branch).
fn extract_configured_fields(parsed: &Map<String, Value>, config: &Value) -> Result<Value> {
    let Some(config_map) = config.as_object() else {
        return Ok(Value::Object(Map::new()));
    };
    let mut output = Map::new();
    let mut missing: Vec<String> = Vec::new();
    for (field_name, field_config) in config_map {
        let field = field_config.as_object();
        let Some(value) = parsed.get(field_name) else {
            if field_is_required(field) {
                missing.push(field_name.clone());
            }
            continue;
        };
        if let Some(expected) = field
            .and_then(|config| config.get("type"))
            .and_then(Value::as_str)
        {
            validate_field_type(field_name, value, expected)?;
        }
        output.insert(field_name.clone(), project_field_value(value, field));
    }
    if !missing.is_empty() {
        return Err(llm_execution_process_error(&format!(
            "missing required fields: {}",
            missing.join(", ")
        )));
    }
    Ok(Value::Object(output))
}

/// Schema-mode branch: project through `properties`, checking `required`.
fn project_schema_output(parsed: &Map<String, Value>, schema: &Value) -> Result<Value> {
    let Some(schema_map) = schema.as_object() else {
        return Ok(Value::Object(Map::new()));
    };
    let required: Vec<&str> = schema_map
        .get("required")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let missing: Vec<String> = required
        .iter()
        .filter(|name| !parsed.contains_key(**name))
        .map(|name| (*name).to_owned())
        .collect();
    if !missing.is_empty() {
        return Err(llm_execution_process_error(&format!(
            "missing required fields: {}",
            missing.join(", ")
        )));
    }
    let mut output = Map::new();
    if let Some(properties) = schema_map.get("properties").and_then(Value::as_object) {
        for (name, property_schema) in properties {
            let Some(value) = parsed.get(name) else {
                continue;
            };
            if let Some(expected) = property_schema.get("type").and_then(Value::as_str) {
                validate_field_type(name, value, expected)?;
            }
            output.insert(name.clone(), value.clone());
        }
    }
    Ok(Value::Object(output))
}

/// Whether a field's config marks it required: absent or explicit `true`.
fn field_is_required(field: Option<&Map<String, Value>>) -> bool {
    !matches!(
        field.and_then(|config| config.get("required")),
        Some(Value::Bool(false))
    )
}

/// Keeps only the sub-keys a field declares in `"properties"`.
///
/// A field without declared properties, or a non-object value, is returned as is.
fn project_field_value(value: &Value, field: Option<&Map<String, Value>>) -> Value {
    let Some(properties) = field
        .and_then(|config| config.get("properties"))
        .and_then(Value::as_object)
    else {
        return value.clone();
    };
    let Value::Object(object) = value else {
        return value.clone();
    };
    if object.is_empty() {
        return value.clone();
    }
    let mut filtered = Map::new();
    for (key, item) in object {
        if properties.contains_key(key) {
            filtered.insert(key.clone(), item.clone());
        }
    }
    Value::Object(filtered)
}

/// Port of `ValidationUtils.validate_type` for a configured field.
fn validate_field_type(field: &str, value: &Value, expected: &str) -> Result<()> {
    if value_matches_type(value, expected) {
        return Ok(());
    }
    Err(llm_execution_process_error(&format!(
        "expected type {expected} for field '{field}'"
    )))
}

/// Whether `value` satisfies the Python type name `expected`.
///
/// Booleans are neither integers nor numbers, matching Python's `isinstance`
/// checks; an unknown type name never matches.
fn value_matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "number" => value.is_number(),
        _ => false,
    }
}

/// Whether `output_config` already is a whole JSON Schema.
fn is_whole_schema(output_config: &Value) -> bool {
    output_config.get("type").and_then(Value::as_str) == Some("object")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn contents(messages: &[BaseMessage]) -> Vec<String> {
        messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect()
    }

    #[test]
    fn text_format_leaves_the_conversation_untouched() {
        let messages = vec![BaseMessage::system("s"), BaseMessage::user("u")];
        let formatted = format_prompt(
            messages.clone(),
            LlmResponseFormat::Text,
            None,
            None,
            Some(&json!({"answer": {}})),
        );
        assert_eq!(formatted, messages);
    }

    #[test]
    fn markdown_replaces_only_the_last_user_message() {
        let messages = vec![
            BaseMessage::system("sys"),
            BaseMessage::user("first"),
            BaseMessage::assistant("reply"),
            BaseMessage::user("second"),
        ];
        let formatted = format_prompt(messages, LlmResponseFormat::Markdown, None, None, None);

        assert_eq!(
            formatted.iter().map(|m| m.role.clone()).collect::<Vec<_>>(),
            vec![Role::System, Role::User, Role::Assistant, Role::User]
        );
        let texts = contents(&formatted);
        assert_eq!(texts[0], "sys");
        assert_eq!(texts[1], "first");
        assert_eq!(texts[2], "reply");
        assert!(texts[3].contains("Please return the answer in markdown format."));
        assert!(texts[3].ends_with("The question is: second."));
    }

    #[test]
    fn markdown_override_replaces_the_default_instruction() {
        let formatted = format_prompt(
            vec![BaseMessage::user("why")],
            LlmResponseFormat::Markdown,
            Some("Answer as markdown. Question: ${query}"),
            None,
            None,
        );
        assert_eq!(
            contents(&formatted),
            vec!["Answer as markdown. Question: why".to_owned()]
        );
    }

    #[test]
    fn json_embeds_the_compact_schema_and_the_query() {
        let output_config = json!({"answer": {"type": "string"}});
        let formatted = format_prompt(
            vec![BaseMessage::user("why")],
            LlmResponseFormat::Json,
            None,
            None,
            Some(&output_config),
        );
        let text = contents(&formatted).remove(0);
        let expected_schema = serde_json::to_string(&generate_json_schema(&output_config)).unwrap();
        assert!(text.contains("Here is the JSON schema:"));
        assert!(text.contains(&expected_schema));
        assert!(text.contains("\"answer\""));
        assert!(text.ends_with("The question is: why."));
    }

    #[test]
    fn json_replaces_the_schema_then_the_query() {
        // Python substitutes `${json_schema}` first and `${query}` second, so both
        // the instruction's placeholder and any literal `${query}` inside the
        // serialized schema are replaced by the query.
        let output_config = json!({"${query}": {"type": "string"}});
        let formatted = format_prompt(
            vec![BaseMessage::user("real question")],
            LlmResponseFormat::Json,
            None,
            None,
            Some(&output_config),
        );
        let text = contents(&formatted).remove(0);
        assert!(!text.contains("${query}"));
        assert!(text.contains("\"real question\""));
        assert!(text.ends_with("The question is: real question."));
    }

    #[test]
    fn no_user_message_is_left_untouched() {
        let messages = vec![BaseMessage::system("only")];
        let formatted = format_prompt(messages.clone(), LlmResponseFormat::Json, None, None, None);
        assert_eq!(formatted, messages);
    }

    #[test]
    fn generate_json_schema_from_a_field_config() {
        let schema = generate_json_schema(&json!({
            "a": {"type": "integer"},
            "b": {"required": false},
        }));
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "string"},
                },
                "required": ["a"],
            })
        );
    }

    #[test]
    fn generate_json_schema_passes_through_extras() {
        let schema = generate_json_schema(&json!({
            "list": {"type": "array", "items": {"type": "integer"}},
            "nested": {
                "type": "object",
                "description": "n",
                "properties": {"x": {"type": "string"}},
            },
        }));
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "list": {"type": "array", "items": {"type": "integer"}},
                    "nested": {
                        "type": "object",
                        "description": "n",
                        "properties": {"x": {"type": "string"}},
                    },
                },
                "required": ["list", "nested"],
            })
        );
    }

    #[test]
    fn generate_json_schema_returns_a_whole_schema_unchanged() {
        let whole = json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        });
        assert_eq!(generate_json_schema(&whole), whole);
    }

    #[test]
    fn output_config_none_defaults_to_result() {
        assert_eq!(output_config_object(None), json!({"result": {}}));
        assert_eq!(
            output_config_object(Some(&json!({}))),
            json!({"result": {}})
        );
    }

    #[test]
    fn shape_text_uses_the_single_key() {
        let output = shape_output(
            "hello",
            LlmResponseFormat::Text,
            Some(&json!({"answer": {"type": "string"}})),
        )
        .unwrap();
        assert_eq!(output, json!({"answer": "hello"}));
    }

    #[test]
    fn shape_text_defaults_to_the_result_key() {
        let output = shape_output("hello", LlmResponseFormat::Text, None).unwrap();
        assert_eq!(output, json!({"result": "hello"}));
    }

    #[test]
    fn shape_text_two_keys_is_a_config_error() {
        let error = shape_output(
            "hello",
            LlmResponseFormat::Markdown,
            Some(&json!({"a": {}, "b": {}})),
        )
        .unwrap_err();
        assert!(error.message().contains("[101002]"), "{error}");
        assert!(
            error.message().contains(
                "text/markdown response type, outputs_config must contain only one field"
            ),
            "{error}"
        );
    }

    #[test]
    fn shape_json_parses_a_fenced_reply_and_projects_configured_keys() {
        let output = shape_output(
            "```json\n{\"answer\": \"42\", \"extra\": 1}\n```",
            LlmResponseFormat::Json,
            Some(&json!({"answer": {"type": "string"}})),
        )
        .unwrap();
        assert_eq!(output, json!({"answer": "42"}));
    }

    #[test]
    fn shape_json_drops_undeclared_sub_properties() {
        let output = shape_output(
            "{\"answer\": {\"a\": 1, \"b\": 2}}",
            LlmResponseFormat::Json,
            Some(&json!({"answer": {"type": "object", "properties": {"a": {}}}})),
        )
        .unwrap();
        assert_eq!(output, json!({"answer": {"a": 1}}));
    }

    #[test]
    fn shape_json_invalid_json_is_a_process_error() {
        let error = shape_output(
            "not json",
            LlmResponseFormat::Json,
            Some(&json!({"answer": {}})),
        )
        .unwrap_err();
        assert!(error.message().contains("[101004]"), "{error}");
        assert!(error.message().contains("Json parse error"), "{error}");
    }

    #[test]
    fn shape_json_missing_required_field_names_it() {
        let error = shape_output(
            "{\"a\": \"x\"}",
            LlmResponseFormat::Json,
            Some(&json!({"a": {}, "b": {}})),
        )
        .unwrap_err();
        assert!(error.message().contains("[101004]"), "{error}");
        assert!(
            error.message().contains("missing required fields: b"),
            "{error}"
        );
    }

    #[test]
    fn shape_json_type_mismatch_names_the_field() {
        let error = shape_output(
            "{\"a\": \"text\"}",
            LlmResponseFormat::Json,
            Some(&json!({"a": {"type": "integer"}})),
        )
        .unwrap_err();
        assert!(error.message().contains("[101004]"), "{error}");
        assert!(error.message().contains("field 'a'"), "{error}");
    }

    #[test]
    fn shape_json_whole_schema_projects_through_properties() {
        let schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        });
        let output = shape_output(
            "{\"answer\": \"42\", \"extra\": 1}",
            LlmResponseFormat::Json,
            Some(&schema),
        )
        .unwrap();
        assert_eq!(output, json!({"answer": "42"}));
    }

    #[test]
    fn shape_json_whole_schema_checks_required() {
        let schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        });
        let error =
            shape_output("{\"extra\": 1}", LlmResponseFormat::Json, Some(&schema)).unwrap_err();
        assert!(error.message().contains("[101004]"), "{error}");
        assert!(
            error.message().contains("missing required fields: answer"),
            "{error}"
        );
    }

    #[test]
    fn clean_markdown_blocks_strips_info_string_and_trailing_fence() {
        assert_eq!(
            clean_markdown_blocks("  ```json\n{\"a\": 1}\n```\n"),
            "{\"a\": 1}"
        );
        assert_eq!(clean_markdown_blocks("```\n{}\n```"), "{}");
        assert_eq!(clean_markdown_blocks(" plain "), "plain");
    }
}
