//! Per-node-type `data` builders of the DL -> DSL transformer.
//!
//! Ports the `_convert_specific_config` bodies of Python's
//! `dl_transformer/converters/*.py`. Every builder returns the fields merged into
//! a platform node's `data` object after its `title`; edges are built separately
//! by [`NodeConversion::edges`].
//!
//! Each builder consumes the node's inputs before its outputs, so the shared
//! variable index counts the input leaves first (Python's questioner counts its
//! outputs first; the port uses one order everywhere).
//!
//! [`NodeConversion::edges`]: super::NodeConversion::edges

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use super::super::model::{DlCondition, DlNodeType, DlVariable};
use super::{
    convert_reference, index_value, llm_model_config, llm_param, object_with, template,
    NodeConversion, CONSTANT, DEFAULT_EXPRESSION, DRAFT_PLUGIN_VERSION, REF,
};

/// The description of the questioner's generated `user_response` output.
const USER_RESPONSE_DESCRIPTION: &str = "用户响应输出变量";
/// The description of the questioner's generated `output` output.
const OUTPUT_DESCRIPTION: &str = "输出变量";

/// Builds the `data` object of one node: its title plus its per-type fields.
///
/// # Errors
///
/// Returns [`Error::Validation`] when a required configuration entry is missing,
/// blank, or malformed, or when a branch condition cannot be converted.
pub(super) fn node_data(
    conversion: &mut NodeConversion<'_>,
    kind: DlNodeType,
) -> Result<Map<String, Value>> {
    let mut data = object_with([("title", Value::String(conversion.node.description.clone()))]);
    let fields = match kind {
        DlNodeType::Start => start(conversion),
        DlNodeType::End => end(conversion)?,
        DlNodeType::Llm => llm(conversion)?,
        DlNodeType::Branch => branch(conversion)?,
        DlNodeType::Questioner => questioner(conversion)?,
        DlNodeType::Code => code(conversion)?,
        DlNodeType::Plugin => plugin(conversion)?,
        DlNodeType::Output => output(conversion)?,
        DlNodeType::IntentDetection => intent_detection(conversion)?,
    };
    data.extend(fields);
    Ok(data)
}

/// The start node: its outputs, all of them required.
fn start(conversion: &mut NodeConversion<'_>) -> Map<String, Value> {
    let mut outputs = conversion.outputs();
    let required = top_level_keys(&outputs);
    if !required.is_empty() {
        outputs.insert("required".to_string(), serde_json::json!(required));
    }
    object_with([("outputs", Value::Object(outputs))])
}

/// The end node: its inputs only.
fn end(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let parameters = conversion.inputs()?;
    Ok(object_with([(
        "inputs",
        Value::Object(object_with([(
            "input_parameters",
            Value::Object(parameters),
        )])),
    )]))
}

/// The LLM node: prompt templates, the fixed model configuration, the
/// `response_format`, and the output schema.
fn llm(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let system_prompt = conversion.prompt("system_prompt")?;
    let user_prompt = conversion.prompt("user_prompt")?;
    let output_format = llm_output_format(conversion);
    let parameters = conversion.inputs()?;

    let mut parameters_with_llm = llm_param(&system_prompt, &user_prompt);
    parameters_with_llm.insert(
        "response_format".to_string(),
        serde_json::json!({"type": output_format}),
    );
    let inputs = object_with([
        ("input_parameters", Value::Object(parameters)),
        ("llm_param", Value::Object(parameters_with_llm)),
    ]);
    let outputs = conversion.outputs();

    Ok(object_with([
        ("output_format", Value::String(output_format)),
        ("inputs", Value::Object(inputs)),
        ("outputs", Value::Object(outputs)),
    ]))
}

/// The branch node: one converted branch entry per condition.
fn branch(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let node_id = conversion.id().to_string();
    let conditions = conversion.node.parameters.conditions.clone();
    let mut branches = Vec::with_capacity(conditions.len());
    for condition in &conditions {
        branches.push(convert_condition(&node_id, condition)?);
    }
    Ok(object_with([("branches", Value::Array(branches))]))
}

/// The questioner node: its prompt templates, the default `input` reference, and
/// the generated `user_response`/`output` outputs.
fn questioner(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let prompt = conversion.config("prompt")?;
    let mut parameters = conversion.inputs()?;
    if parameters.is_empty() {
        parameters.insert(
            "input".to_string(),
            object_with([
                ("type", Value::String(REF.to_string())),
                ("content", Value::String("node_start.query".to_string())),
                ("extra", index_value(0)),
            ])
            .into(),
        );
    }

    let inputs = object_with([
        ("input_parameters", Value::Object(parameters)),
        ("llm_param", Value::Object(llm_param(&prompt, ""))),
        ("system_prompt", template(&prompt)),
        ("history_enable", Value::Bool(false)),
        ("max_response", serde_json::json!(3)),
    ]);

    let mut outputs = conversion.outputs();
    let had_outputs = properties_are_non_empty(&outputs);
    let required = {
        let properties = properties_mut(&mut outputs, conversion.id())?;
        properties
            .entry("user_response".to_string())
            .or_insert_with(|| {
                object_with([
                    ("type", Value::String("string".to_string())),
                    (
                        "description",
                        Value::String(USER_RESPONSE_DESCRIPTION.to_string()),
                    ),
                ])
                .into()
            });
        if had_outputs {
            properties.entry("output".to_string()).or_insert_with(|| {
                object_with([
                    ("type", Value::String("string".to_string())),
                    ("description", Value::String(OUTPUT_DESCRIPTION.to_string())),
                ])
                .into()
            });
        }
        required_keys(properties, Some("output"))
    };
    outputs.insert("required".to_string(), serde_json::json!(required));

    Ok(object_with([
        ("inputs", Value::Object(inputs)),
        ("outputs", Value::Object(outputs)),
    ]))
}

/// The code node: its language/code inputs, its outputs, and the fixed
/// exception configuration.
fn code(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let source = conversion.config("code")?;
    let parameters = conversion.inputs()?;
    let mut outputs = conversion.outputs();
    let required = top_level_keys(&outputs);
    if !required.is_empty() {
        outputs.insert("required".to_string(), serde_json::json!(required));
    }

    let inputs = object_with([
        ("input_parameters", Value::Object(parameters)),
        ("language", Value::String("python".to_string())),
        ("code", Value::String(source)),
    ]);

    Ok(object_with([
        ("inputs", Value::Object(inputs)),
        ("outputs", Value::Object(outputs)),
        ("exception_config", exception_config()),
    ]))
}

/// The plugin node: the resolved plugin parameters and the plugin output
/// defaults.
fn plugin(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let tool_id = conversion.config("tool_id")?;
    let resource = conversion
        .resources
        .iter()
        .find(|resource| resource.tool_id == tool_id);
    let plugin_param = plugin_param(resource);

    let parameters = conversion.inputs()?;
    let mut outputs = conversion.outputs();
    {
        let properties = properties_mut(&mut outputs, conversion.id())?;
        for (name, value) in plugin_default_outputs() {
            properties.entry(name.to_string()).or_insert(value);
        }
    }
    outputs.insert(
        "required".to_string(),
        serde_json::json!(["error_code", "error_message", "data"]),
    );

    let inputs = object_with([
        ("input_parameters", Value::Object(parameters)),
        ("plugin_param", Value::Object(plugin_param)),
    ]);

    Ok(object_with([
        ("inputs", Value::Object(inputs)),
        ("outputs", Value::Object(outputs)),
    ]))
}

/// The output node: its inputs and the template that formats the answer.
fn output(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let content = conversion.config("template")?;
    let parameters = conversion.inputs()?;
    let inputs = object_with([
        ("input_parameters", Value::Object(parameters)),
        ("content", template(&content)),
    ]);
    Ok(object_with([("inputs", Value::Object(inputs))]))
}

/// The intent-detection node: the prompt, the intent list, and the
/// `classification_id` output.
fn intent_detection(conversion: &mut NodeConversion<'_>) -> Result<Map<String, Value>> {
    let prompt = conversion.config("prompt")?;
    let parameters = conversion.inputs()?;
    let query = parameters.into_values().next();
    let conditions = conversion.node.parameters.conditions.clone();
    if conditions.is_empty() {
        return Err(Error::validation(format!(
            "node {}: conditions are required",
            conversion.id()
        )));
    }

    let node_id = conversion.id().to_string();
    let mut intents = Vec::new();
    let mut ports = Vec::new();
    for (position, condition) in conditions.iter().enumerate() {
        let expression = intent_expression(&node_id, condition)?;
        if expression == DEFAULT_EXPRESSION {
            continue;
        }
        let name = intent_name(&node_id, condition, expression)?;
        let id = condition
            .intent_id
            .clone()
            .unwrap_or_else(|| format!("intent_{node_id}_{position}"));
        intents.push(object_with([
            ("name", Value::String(name)),
            ("id", Value::String(id.clone())),
        ]));
        ports.push((condition.branch.clone(), id));
    }
    conversion.intent_ports = ports;

    let input_parameters = match query {
        Some(value) => object_with([("query", value)]),
        None => Map::new(),
    };
    let parameters_with_llm = object_with([
        ("systemPrompt", template("")),
        ("prompt", template(&prompt)),
        ("model", llm_model_config()),
    ]);
    let inputs = object_with([
        ("input_parameters", Value::Object(input_parameters)),
        ("llm_param", Value::Object(parameters_with_llm)),
        (
            "intents",
            Value::Array(intents.into_iter().map(Value::Object).collect()),
        ),
    ]);

    let classification = DlVariable {
        name: "classification_id".to_string(),
        var_type: Some("integer".to_string()),
        ..DlVariable::default()
    };
    let mut outputs = conversion.outputs_for(&[classification]);
    outputs.insert(
        "required".to_string(),
        serde_json::json!(["classification_id"]),
    );

    Ok(object_with([
        ("inputs", Value::Object(inputs)),
        ("outputs", Value::Object(outputs)),
    ]))
}

/// Python's `convert_llm_param` output format rules.
fn llm_output_format(conversion: &NodeConversion<'_>) -> String {
    let outputs_count = conversion.node.parameters.outputs.len();
    let mut output_format = conversion
        .node
        .parameters
        .configs
        .get("output_format")
        .and_then(Value::as_str)
        .unwrap_or("text")
        .to_string();

    if outputs_count > 1 && matches!(output_format.as_str(), "text" | "markdown") {
        output_format = "json".to_string();
    }
    if outputs_count == 1 && !matches!(output_format.as_str(), "text" | "markdown" | "json") {
        output_format = "text".to_string();
    }

    output_format
}

/// Converts one branch condition into a platform branch entry.
fn convert_condition(node_id: &str, condition: &DlCondition) -> Result<Value> {
    let mut entry = Map::new();
    match (&condition.expression, &condition.expressions) {
        (Some(expression), None) => {
            let conditions = if expression == DEFAULT_EXPRESSION {
                Vec::new()
            } else {
                vec![convert_expression(node_id, expression)?]
            };
            entry.insert("conditions".to_string(), Value::Array(conditions));
        }
        (None, Some(expressions)) => {
            let logic = logic_value(node_id, condition)?;
            let mut conditions = Vec::with_capacity(expressions.len());
            for expression in expressions {
                conditions.push(convert_expression(node_id, expression)?);
            }
            entry.insert("conditions".to_string(), Value::Array(conditions));
            entry.insert("logic".to_string(), serde_json::json!(logic));
        }
        _ => {
            return Err(Error::validation(format!(
                "node {node_id}: condition {} must have exactly one of expression or expressions",
                condition.branch
            )))
        }
    }
    entry.insert(
        "branchId".to_string(),
        Value::String(condition.branch.clone()),
    );
    Ok(Value::Object(entry))
}

/// Maps a condition's logical operator to Python's `BRANCH_LOGIC_MAP` value.
fn logic_value(node_id: &str, condition: &DlCondition) -> Result<u8> {
    match condition.operator.as_deref().map(str::trim) {
        Some("or") => Ok(1),
        Some("and") => Ok(2),
        other => Err(Error::validation(format!(
            "node {node_id}: condition {} has unknown logic operator: {}",
            condition.branch,
            other.unwrap_or("<missing>")
        ))),
    }
}

/// Converts one branch expression into a platform comparison.
fn convert_expression(node_id: &str, expression: &str) -> Result<Value> {
    let (operator_token, left, right) = split_expression(node_id, expression)?;
    let operator = operator_name(operator_token).ok_or_else(|| {
        Error::validation(format!(
            "node {node_id}: unknown branch operator: {operator_token}"
        ))
    })?;

    let mut entry = Map::new();
    if let Some(left) = build_side(node_id, left)? {
        entry.insert("left".to_string(), left);
    }
    entry.insert("operator".to_string(), Value::String(operator.to_string()));
    if let Some(right) = build_side(node_id, right)? {
        entry.insert("right".to_string(), right);
    }
    Ok(Value::Object(entry))
}

/// Splits `expression` into its operator token and the raw left/right sides.
fn split_expression<'e>(node_id: &str, expression: &'e str) -> Result<(&'e str, &'e str, &'e str)> {
    let trimmed = expression.trim();
    let spans = token_spans(trimmed);
    let Some((start, end)) = spans.get(1).copied() else {
        return Err(Error::validation(format!(
            "node {node_id}: cannot parse branch expression: {expression}"
        )));
    };
    let operator = &trimmed[start..end];
    Ok((operator, trimmed[..start].trim(), trimmed[end..].trim()))
}

/// The byte spans of the whitespace-separated tokens of `text`.
fn token_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (index, character) in text.char_indices() {
        if character.is_whitespace() {
            if let Some(begin) = start.take() {
                spans.push((begin, index));
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(begin) = start {
        spans.push((begin, text.len()));
    }
    spans
}

/// Maps a Python DL branch operator to the platform operator.
fn operator_name(token: &str) -> Option<&'static str> {
    match token {
        "eq" => Some("eq"),
        "not_eq" => Some("neq"),
        "contain" => Some("contains"),
        "not_contain" => Some("not_contains"),
        "longer_than" => Some("gt"),
        "longer_than_or_eq" => Some("gte"),
        "short_than" => Some("lt"),
        "short_than_or_eq" => Some("lte"),
        "is_empty" => Some("is_empty"),
        "is_not_empty" => Some("is_not_empty"),
        _ => None,
    }
}

/// Builds one side of a comparison; an empty side yields no value.
fn build_side(node_id: &str, value: &str) -> Result<Option<Value>> {
    if value.is_empty() {
        return Ok(None);
    }
    if value.contains("${") {
        return convert_reference(node_id, value).map(Some);
    }
    Ok(Some(Value::Object(object_with([
        ("type", Value::String(CONSTANT.to_string())),
        ("content", Value::String(value.to_string())),
        (
            "schema",
            Value::Object(object_with([
                ("type", Value::String("string".to_string())),
                ("extra", serde_json::json!({"weak": true})),
            ])),
        ),
    ]))))
}

/// The single expression of an intent-detection condition.
fn intent_expression<'c>(node_id: &str, condition: &'c DlCondition) -> Result<&'c str> {
    condition.expression.as_deref().ok_or_else(|| {
        Error::validation(format!(
            "node {node_id}: condition {} has no expression",
            condition.branch
        ))
    })
}

/// The intent name carried by `<value> contain <name>`.
fn intent_name(node_id: &str, condition: &DlCondition, expression: &str) -> Result<String> {
    let parts: Vec<&str> = expression.split(" contain ").collect();
    match parts.get(1) {
        Some(name) => Ok((*name).to_string()),
        None => Err(Error::validation(format!(
            "node {node_id}: condition {} is not a \" contain \" expression: {expression}",
            condition.branch
        ))),
    }
}

/// The top-level property names of an output schema, in insertion order.
fn top_level_keys(outputs: &Map<String, Value>) -> Vec<String> {
    outputs
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default()
}

/// Whether an output schema already declares a property.
fn properties_are_non_empty(outputs: &Map<String, Value>) -> bool {
    outputs
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|properties| !properties.is_empty())
}

/// The property names of an output schema, optionally excluding one.
fn required_keys(properties: &Map<String, Value>, excluded: Option<&str>) -> Vec<String> {
    properties
        .keys()
        .filter(|key| Some(key.as_str()) != excluded)
        .cloned()
        .collect()
}

/// The mutable `properties` object of an output schema.
fn properties_mut<'m>(
    outputs: &'m mut Map<String, Value>,
    node_id: &str,
) -> Result<&'m mut Map<String, Value>> {
    outputs
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| Error::validation(format!("node {node_id}: outputs have no properties")))
}

/// Python's `CODE_EXCEPTION_CONFIG`.
fn exception_config() -> Value {
    serde_json::json!({
        "retryTimes": 3,
        "timeoutSeconds": 30,
        "processType": "break",
        "executeStep": {"defaultStep": "0", "errorStep": "1"},
    })
}

/// The plugin parameters of a resolved resource (or the empty defaults).
fn plugin_param(resource: Option<&super::PluginResource>) -> Map<String, Value> {
    let Some(resource) = resource else {
        return object_with([
            ("toolID", Value::String(String::new())),
            ("toolName", Value::String(String::new())),
            ("pluginID", Value::String(String::new())),
            ("pluginName", Value::String(String::new())),
            (
                "pluginVersion",
                Value::String(DRAFT_PLUGIN_VERSION.to_string()),
            ),
        ]);
    };

    let version = if resource.plugin_version.trim().is_empty() {
        DRAFT_PLUGIN_VERSION.to_string()
    } else {
        resource.plugin_version.clone()
    };
    object_with([
        ("toolID", Value::String(resource.tool_id.clone())),
        ("toolName", Value::String(resource.tool_name.clone())),
        ("pluginID", Value::String(resource.plugin_id.clone())),
        ("pluginName", Value::String(resource.plugin_name.clone())),
        ("pluginVersion", Value::String(version)),
    ])
}

/// Python's `PLUGIN_DEFAULT_OUTPUTS`, in insertion order.
fn plugin_default_outputs() -> Vec<(&'static str, Value)> {
    vec![
        (
            "error_code",
            object_with([
                ("type", Value::String("integer".to_string())),
                ("extra", index_value(1)),
            ])
            .into(),
        ),
        (
            "error_message",
            object_with([
                ("type", Value::String("string".to_string())),
                ("extra", index_value(2)),
            ])
            .into(),
        ),
        (
            "data",
            object_with([
                ("type", Value::String("object".to_string())),
                ("extra", index_value(3)),
                ("properties", Value::Object(Map::new())),
            ])
            .into(),
        ),
    ]
}
