//! Extraction prompts, placeholder rendering, and the continue-ask wording.
//!
//! Holds the verbatim Chinese and English extraction templates ported from
//! `questioner_comp.py`, the `{{key}}` renderer they share with the question
//! template, the `dialogue_history` renderer, and the wording used when required
//! fields are still missing.
//!
//! The `dialogue_history` keyword is one `role: content` line per message, joined with
//! a newline (Python's `"\n".join([f"{_.role}: {_.content}" …])`); the separator is
//! `": "` for `accept_language == "en"` and `"："` otherwise, exactly like Python.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use openjiuwen_foundation::BaseMessage;

use super::config::{FieldInfo, QuestionerConfig};
use super::python_str;

/// Chinese extraction system prompt (Python `QUESTIONER_SYSTEM_TEMPLATE_ZH`).
const QUESTIONER_SYSTEM_TEMPLATE_ZH: &str = r"你是一个信息收集助手，你需要根据指定的参数收集用户的信息，然后提交到系统。
请注意：不要使用任何工具、不用理会问题的具体含义，并保证你的输出仅有 JSON 格式的结果数据。
请严格遵循如下规则：
  1. 让我们一步一步思考。
  2. 用户输入中没有提及的参数提取为 null，并直接向询问用户没有明确提供的参数。
  3. 通过用户提供的对话历史以及当前输入中提取 {{required_name}}，不要追问任何其他信息。
  4. 参数收集完成后，将收集到的信息通过 JSON 的方式展示给用户。

## Specified Parameters
{{required_params_list}}

## Constraints
{{extra_info}}

## Examples
{{example}}
";

/// Chinese extraction user prompt (Python `QUESTIONER_USER_TEMPLATE_ZH`).
const QUESTIONER_USER_TEMPLATE_ZH: &str = r"对话历史
{{dialogue_history}}

请充分考虑以上对话历史及用户输入，正确提取最符合约束要求的 JSON 格式参数。
";

/// English extraction system prompt (Python `QUESTIONER_SYSTEM_TEMPLATE_EN`).
const QUESTIONER_SYSTEM_TEMPLATE_EN: &str = r"You are an information collection assistant. You need to collect user information based on the specified parameters and submit it to the system.
Please note: Do not use any tools, do not consider the specific meaning of the questions, and ensure your output contains only JSON-formatted result data.
Strictly follow these rules:
  1. Let's think step by step.
  2. Parameters not mentioned in user input should be extracted as null, and directly ask the user for parameters not explicitly provided.
  3. Extract {{required_name}} from the conversation history and current user input. Do not ask for any other information.
  4. After parameter collection is complete, display the collected information in JSON format.

## Specified Parameters
{{required_params_list}}

## Constraints
{{extra_info}}

## Examples
{{example}}
";

/// English extraction user prompt (Python `QUESTIONER_USER_TEMPLATE_EN`).
const QUESTIONER_USER_TEMPLATE_EN: &str = r"Conversation History
{{dialogue_history}}

Please fully consider the above conversation history and user input, and correctly extract the JSON-formatted parameters that best meet the constraints.
";

/// Builds the extraction conversation from the Python prompt templates.
///
/// `history` is the conversation window the caller already prepared: the turns to
/// show plus the current user query as its last entry (see
/// [`QuestionerComponent::chat_history_window`](super::QuestionerComponent)).
pub(super) fn extraction_messages(
    config: &QuestionerConfig,
    history: &[BaseMessage],
) -> Vec<BaseMessage> {
    let english = config.accept_language == "en";
    let mut params = Vec::new();
    let mut required = Vec::new();
    for field in &config.field_names {
        params.push(format!("{}: {}", field.field_name, field.description));
        if field.required {
            required.push(if field.cn_field_name.is_empty() {
                field.description.clone()
            } else {
                field.cn_field_name.clone()
            });
        }
    }
    let required_name = if english {
        format!(
            "{} ({} required field(s))",
            required.join(", "),
            required.len()
        )
    } else {
        format!("{}{}个必要信息", required.join("、"), required.len())
    };
    let dialogue_history = render_dialogue_history(english, history);
    let values = BTreeMap::from([
        ("required_name".to_owned(), required_name),
        ("required_params_list".to_owned(), params.join("\n")),
        (
            "extra_info".to_owned(),
            config.extra_prompt_for_fields_extraction.clone(),
        ),
        ("example".to_owned(), config.example_content.clone()),
        ("dialogue_history".to_owned(), dialogue_history),
    ]);
    let (system, user) = if english {
        (QUESTIONER_SYSTEM_TEMPLATE_EN, QUESTIONER_USER_TEMPLATE_EN)
    } else {
        (QUESTIONER_SYSTEM_TEMPLATE_ZH, QUESTIONER_USER_TEMPLATE_ZH)
    };
    vec![
        BaseMessage::system(render_prompt(system, &values)),
        BaseMessage::user(render_prompt(user, &values)),
    ]
}

/// Renders a conversation window as one `role<separator>content` line per message
/// (Python's `_create_prompt_template_keywords`).
///
/// Every message contributes its plain text ([`BaseMessage::text`]), so structured
/// content never panics; the lines are joined with `"\n"` and an empty window renders
/// as the empty string.
fn render_dialogue_history(english: bool, history: &[BaseMessage]) -> String {
    let separator = if english { ": " } else { "：" };
    history
        .iter()
        .map(|message| format!("{}{separator}{}", message.role.as_str(), message.text()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Formats the batch question template (Python `QuestionerUtils.format_template`).
///
/// Every `{{key}}` becomes `str(value)`; an unresolved placeholder becomes `None`.
pub(super) fn format_template(template: &str, fields: &Map<String, Value>) -> String {
    substitute(template, |key| {
        Some(python_str(fields.get(key).unwrap_or(&Value::Null)))
    })
}

/// Renders a named placeholder template, leaving unknown placeholders verbatim.
fn render_prompt(template: &str, values: &BTreeMap<String, String>) -> String {
    substitute(template, |key| values.get(key).cloned())
}

/// Replaces `{{key}}` placeholders with `lookup`'s answer.
fn substitute(template: &str, lookup: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            out.push_str(&rest[start..]);
            return out;
        };
        if let Some(value) = lookup(&after[..end]) {
            out.push_str(&value);
        } else {
            out.push_str("{{");
            out.push_str(&after[..end]);
            out.push_str("}}");
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    out
}

/// The continue-ask question for the still-missing fields
/// (Python `format_continue_ask_question`).
pub(super) fn format_continue_ask(fields: &[&FieldInfo], accept_language: &str) -> String {
    let names: Vec<&str> = fields
        .iter()
        .map(|field| {
            if field.cn_field_name.is_empty() {
                field.description.as_str()
            } else {
                field.cn_field_name.as_str()
            }
        })
        .collect();
    let names = names.join(", ");
    if accept_language == "en" {
        format!("Please provide information related to: {names}")
    } else {
        format!("请您提供{names}相关的信息")
    }
}
