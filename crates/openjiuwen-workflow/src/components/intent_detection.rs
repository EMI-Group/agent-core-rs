//! Built-in `intent_detection` workflow component.
//!
//! Ports `openjiuwen/core/workflow/components/llm/intent_detection_comp.py`: the model
//! is asked to classify the current input into one of the configured categories, and the
//! reply is post-processed into `{"classification_id", "category_name", "reason"}`.
//!
//! # Category model
//!
//! With `category_name_list = ["weather", "travel"]` the component knows three classes:
//! `分类0` (the default class, displayed as `默认意图`) followed by `分类1` (displayed as
//! `weather`) and `分类2` (`travel`). `accept_language` selects the label prefix and the
//! default display name: `"zh"` → `分类`/`默认意图`, `"en"` → `Category`/`Default intent`.
//!
//! # Reply post-processing
//!
//! The outermost `{…}` block of the reply is parsed as JSON and its `class` field is
//! sanitised (newlines, spaces and quotes removed) and matched against `Category\d+`
//! (`-i`) or `分类\d+`. Any failure — an unparsable reply, a non-object reply, a missing
//! `class`, or a class outside the configured list — yields the default class plus a
//! language-specific explanation in `reason`, byte-identical to Python's templates.
//!
//! # Deviations from Python
//!
//! - **Rule path added.** Python only calls a model. When no model is configured this
//!   component scores the configured [`IntentDetectionCompConfig::keywords`] against the
//!   query, so the component is usable (and testable) fully offline. With no keywords
//!   configured every category's display name is its single keyword.
//! - **Model injected.** Python builds a `Model` from `model_client_config`/`model_config`
//!   (or resolves `model_id` through the resource manager); here the caller passes an
//!   `Arc<dyn Model>` through [`IntentDetectionCompConfig::model`].
//! - **History comes from the session.** Python reads a `ModelContext`; here the previous
//!   turn is read from the session global lane `__messages__` (a JSON array of
//!   `{"role", "content"}` objects) and the completed turn is appended back to it.
//! - **Strict JSON.** Python rewrites `false`/`true`/`null` and uses `ast.literal_eval`;
//!   this port parses the extracted block with `serde_json`, so `{'class': 'x'}` is not
//!   accepted while plain JSON is.
//! - **Routing is external.** [`crate::Component::route`] is deliberately *not*
//!   implemented: the engine would then discard this component's output instead of
//!   writing it to the graph state, and the successor choice needs that output. Callers
//!   wire a [`IntentRouter`] with
//!   [`Workflow::add_conditional_connection`](crate::Workflow::add_conditional_connection)
//!   instead.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{
    BaseMessage, Model, ModelRequestConfig, PromptTemplate, TemplateContent,
};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;

/// Input parameters are unusable (Python `COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR`).
pub const COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR: u32 = 101_050;
/// The model could not be created (Python `COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED`).
///
/// Reserved: the model is injected, so this port never builds one.
pub const COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED: u32 = 101_051;
/// The model call failed (Python `COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED`).
pub const COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED: u32 = 101_052;
/// A category display name is blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID: u32 = 101_053;
/// A category display name repeats another one (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE: u32 = 101_054;
/// `accept_language` is neither `"zh"` nor `"en"` (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID: u32 = 101_055;
/// A `keywords` key is not a category display name (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN: u32 = 101_056;
/// A `keywords` entry is blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_KEYWORD_INVALID: u32 = 101_057;
/// An `example_content` entry is blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID: u32 = 101_058;
/// `user_prompt` is present but blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_PROMPT_INVALID: u32 = 101_059;
/// The configuration is otherwise unusable (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_CONFIG_ERROR: u32 = 101_060;
/// The history configuration is unusable (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR: u32 = 101_061;
/// An [`IntentRouter`] is unusable (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR: u32 = 101_062;

/// The rule path matched no keyword.
const NO_KEYWORD_MATCH_REASON: &str = "no keyword matched";
/// The session global-lane key holding the conversation history.
const MESSAGES_STATE_KEY: &str = "__messages__";
/// The `accept_language` value selecting the English prompts and labels.
const LANGUAGE_EN: &str = "en";
/// The `accept_language` value selecting the Chinese prompts and labels.
const LANGUAGE_ZH: &str = "zh";
/// The default display name in Chinese (Python `CLASSIFICATION_DEFAULT_NAME_ZH`).
const DEFAULT_NAME_ZH: &str = "默认意图";
/// The default display name in English (Python `CLASSIFICATION_DEFAULT_NAME_EN`).
const DEFAULT_NAME_EN: &str = "Default intent";
/// The Chinese system prompt (Python `DEFAULT_SYSTEM_PROMPT_ZH`).
const DEFAULT_SYSTEM_PROMPT_ZH: &str = "你是一个识别用户输入意图的AI助手。";
/// The English system prompt (Python `DEFAULT_SYSTEM_PROMPT_EN`).
const DEFAULT_SYSTEM_PROMPT_EN: &str = "You are an AI assistant that identifies user input intent.";
/// The Chinese user prompt (Python `DEFAULT_USER_PROMPT_ZH`).
const DEFAULT_USER_PROMPT_ZH: &str = "
{{user_prompt}}

当前可供选择的功能分类如下：
{{category_info}}

用户与助手的对话历史：
{{chat_history}}

当前输入：
{{input}}

请根据当前输入和对话历史分析并输出最适合的功能分类。输出格式为 JSON，包含以下两个字段：
class: 代表分类结果
reason: 说明为何选择该分类
例如: {\"class\": \"分类xx\", \"reason\": \"当前输入xxx\"}
请参考以下示例：
{{example_content}}
如果没有合适的分类，请输出 {{default_class}}。
";
/// The English user prompt (Python `DEFAULT_USER_PROMPT_EN`).
const DEFAULT_USER_PROMPT_EN: &str = "
{{user_prompt}}

Available function categories:
{{category_info}}

Conversation history between user and assistant:
{{chat_history}}

Current input:
{{input}}

Please analyze the current input and conversation history, and output the most suitable function category. Output format is JSON with the following two fields:
class: represents the classification result
reason: explains why this classification was chosen
Example: {\"class\": \"Category1\", \"reason\": \"Current input xxx\"}
Please refer to the following examples:
{{example_content}}
If no suitable category exists, output {{default_class}}.
";
/// The Chinese unparsable-reply reason (Python `JSON_PARSE_FAIL_REASON_ZH`).
const JSON_PARSE_FAIL_REASON_ZH: &str =
    "当前意图识别的输出:'{result}'格式不符合有效的JSON规范，导致解析失败，因此返回默认分类。";
/// The English unparsable-reply reason (Python `JSON_PARSE_FAIL_REASON_EN`).
const JSON_PARSE_FAIL_REASON_EN: &str = "The intent detection output '{result}' does not conform to valid JSON format, parsing failed, therefore default category is returned.";
/// The Chinese missing-`class` reason (Python `CLASS_KEY_MISSING_REASON_ZH`).
const CLASS_KEY_MISSING_REASON_ZH: &str =
    "当前意图识别的输出 '{result}' 缺少必要的输出'class'分类信息，因此返回默认分类。";
/// The English missing-`class` reason (Python `CLASS_KEY_MISSING_REASON_EN`).
const CLASS_KEY_MISSING_REASON_EN: &str = "The intent detection output '{result}' is missing the required 'class' classification field, therefore default category is returned.";
/// The Chinese unknown-class reason (Python `VALIDATION_FAIL_REASON_ZH`).
const VALIDATION_FAIL_REASON_ZH: &str = "当前意图识别的输出类别 '{intent_class}' 不在预定义的分类列表: '{category_list}'中，因此系统返回默认分类。";
/// The English unknown-class reason (Python `VALIDATION_FAIL_REASON_EN`).
const VALIDATION_FAIL_REASON_EN: &str = "The intent detection output class '{intent_class}' is not in the predefined category list: '{category_list}', therefore default category is returned.";

/// The input parameters are unusable (`101050`).
pub fn intent_detection_input_param_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR,
        format!("component intent_detection_input parameter error, reason: {reason}"),
    )
}

/// The model could not be created (`101051`).
///
/// Reserved: this port takes an already-built model, so nothing calls it.
pub fn intent_detection_llm_init_failed(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED,
        format!("component intent_detection_llm initialization failed, reason: {reason}"),
    )
}

/// The model call failed (`101052`).
pub fn intent_detection_invoke_call_failed(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED,
        format!("component intent_detection_invoke call failed, reason: {reason}"),
    )
}

/// A category display name is blank (`101053`).
pub fn intent_detection_category_name_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID,
        format!("component intent_detection_category_name parameter error, reason: {reason}"),
    )
}

/// A category display name repeats another one (`101054`).
pub fn intent_detection_category_name_duplicate(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE,
        format!("component intent_detection_category_name_list parameter error, reason: {reason}"),
    )
}

/// `accept_language` is unsupported (`101055`).
pub fn intent_detection_accept_language_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID,
        format!("component intent_detection_accept_language parameter error, reason: {reason}"),
    )
}

/// A `keywords` key is not a category display name (`101056`).
pub fn intent_detection_keyword_category_unknown(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN,
        format!("component intent_detection_keywords parameter error, reason: {reason}"),
    )
}

/// A `keywords` entry is blank (`101057`).
pub fn intent_detection_keyword_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_KEYWORD_INVALID,
        format!("component intent_detection_keywords entry parameter error, reason: {reason}"),
    )
}

/// An `example_content` entry is blank (`101058`).
pub fn intent_detection_example_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID,
        format!("component intent_detection_example_content parameter error, reason: {reason}"),
    )
}

/// `user_prompt` is present but blank (`101059`).
pub fn intent_detection_prompt_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_PROMPT_INVALID,
        format!("component intent_detection_user_prompt parameter error, reason: {reason}"),
    )
}

/// The configuration is otherwise unusable (`101060`).
pub fn intent_detection_config_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_CONFIG_ERROR,
        format!("component intent_detection config error, reason: {reason}"),
    )
}

/// The history configuration is unusable (`101061`).
pub fn intent_detection_history_config_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR,
        format!("component intent_detection_chat_history configuration error, reason: {reason}"),
    )
}

/// An [`IntentRouter`] is unusable (`101062`).
pub fn intent_detection_router_config_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR,
        format!("component intent_detection_router configuration error, reason: {reason}"),
    )
}

/// Renders one Python-shaped coded workflow failure.
fn coded(code: u32, message: impl Into<String>) -> Error {
    Error::workflow(format!("[{code}] {}", message.into()))
}

/// Configuration of an [`IntentDetectionComponent`].
///
/// Field names and defaults match Python's `IntentDetectionCompConfig`, except that the
/// model and the rule-path keywords are not serializable: they hold runtime objects, so
/// both are skipped during serialization and defaulted on deserialization.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IntentDetectionCompConfig {
    /// Logical model id carried for configuration parity.
    pub model_id: Option<String>,
    /// Display names of the non-default categories, in class-id order (class 1 first).
    pub category_name_list: Vec<String>,
    /// Extra instructions rendered as `{{user_prompt}}`.
    pub user_prompt: String,
    /// Few-shot examples rendered as `{{example_content}}` (joined with a blank line).
    pub example_content: Vec<String>,
    /// Whether the previous conversation is rendered and the completed turn recorded.
    pub enable_history: bool,
    /// How many trailing history messages are rendered (Python slices messages).
    pub chat_history_max_turn: u32,
    /// Prompt and label language, `"zh"` or `"en"`.
    pub accept_language: String,
    /// Rule-path triggers per category display name; skipped by serde.
    #[serde(skip)]
    pub keywords: BTreeMap<String, Vec<String>>,
    /// The model to call; when `None` the keyword rule path runs instead.
    #[serde(skip)]
    pub model: Option<Arc<dyn Model>>,
}

impl Default for IntentDetectionCompConfig {
    fn default() -> Self {
        Self {
            model_id: None,
            category_name_list: Vec::new(),
            user_prompt: String::new(),
            example_content: Vec::new(),
            enable_history: false,
            chat_history_max_turn: 3,
            accept_language: LANGUAGE_ZH.to_owned(),
            keywords: BTreeMap::new(),
            model: None,
        }
    }
}

impl fmt::Debug for IntentDetectionCompConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntentDetectionCompConfig")
            .field("model", &self.model.as_deref().map(Model::model_name))
            .field("model_id", &self.model_id)
            .field("category_name_list", &self.category_name_list)
            .field("user_prompt", &self.user_prompt)
            .field("example_content", &self.example_content)
            .field("enable_history", &self.enable_history)
            .field("chat_history_max_turn", &self.chat_history_max_turn)
            .field("accept_language", &self.accept_language)
            .field("keywords", &self.keywords)
            .finish()
    }
}

impl IntentDetectionCompConfig {
    /// Sets the model to call, switching the component to the model path.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the logical model id.
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Replaces the non-default category display names.
    #[must_use]
    pub fn with_category_names(mut self, categories: Vec<String>) -> Self {
        self.category_name_list = categories;
        self
    }

    /// Sets the extra instructions rendered as `{{user_prompt}}`.
    #[must_use]
    pub fn with_user_prompt(mut self, user_prompt: impl Into<String>) -> Self {
        self.user_prompt = user_prompt.into();
        self
    }

    /// Replaces the few-shot examples.
    #[must_use]
    pub fn with_example_content(mut self, examples: Vec<String>) -> Self {
        self.example_content = examples;
        self
    }

    /// Enables (or disables) history rendering and recording.
    #[must_use]
    pub fn with_enable_history(mut self, enable_history: bool) -> Self {
        self.enable_history = enable_history;
        self
    }

    /// Sets how many trailing history messages are rendered.
    #[must_use]
    pub fn with_chat_history_max_turn(mut self, max_turn: u32) -> Self {
        self.chat_history_max_turn = max_turn;
        self
    }

    /// Sets the prompt and label language (`"zh"` or `"en"`).
    #[must_use]
    pub fn with_accept_language(mut self, accept_language: impl Into<String>) -> Self {
        self.accept_language = accept_language.into();
        self
    }

    /// Replaces the rule-path trigger keywords.
    #[must_use]
    pub fn with_keywords(mut self, keywords: BTreeMap<String, Vec<String>>) -> Self {
        self.keywords = keywords;
        self
    }
}

/// The built-in `intent_detection` component.
///
/// `invoke` classifies its `query` input and returns
/// `{"classification_id": <int>, "category_name": "<display name>"}`, plus `"reason"` when
/// the classification carries one (Python's `exclude_defaults=True`).
pub struct IntentDetectionComponent {
    config: IntentDetectionCompConfig,
    default_class: String,
    category_list: Vec<String>,
    category_names: Vec<String>,
    category_pattern: Regex,
}

impl fmt::Debug for IntentDetectionComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntentDetectionComponent")
            .field("config", &self.config)
            .field("default_class", &self.default_class)
            .field("category_list", &self.category_list)
            .field("category_names", &self.category_names)
            .finish_non_exhaustive()
    }
}

impl IntentDetectionComponent {
    /// Builds a component from `config`.
    ///
    /// # Errors
    ///
    /// Returns an [`ErrorKind::Workflow`](openjiuwen_common::error::ErrorKind::Workflow)
    /// error carrying `101053`–`101061` when the configuration is unusable: an
    /// unsupported `accept_language`, a blank, duplicated, or default-colliding category
    /// name, an unknown keyword category, a blank keyword or example, a whitespace-only
    /// `user_prompt`, or `enable_history` without room for any message.
    pub fn new(config: IntentDetectionCompConfig) -> Result<Self> {
        validate_config(&config)?;
        let english = config.accept_language == LANGUAGE_EN;
        let prefix = if english { "Category" } else { "分类" };
        let default_name = default_display_name(&config.accept_language);
        let default_class = format!("{prefix}0");
        let mut category_list = vec![default_class.clone()];
        let mut category_names = vec![default_name.to_owned()];
        for (index, name) in config.category_name_list.iter().enumerate() {
            category_list.push(format!("{prefix}{}", index + 1));
            category_names.push(name.clone());
        }
        let pattern = if english {
            r"(?i)Category\d+"
        } else {
            r"(?i)分类\d+"
        };
        let category_pattern = Regex::new(pattern).map_err(|error| {
            intent_detection_config_error(&format!("invalid category pattern '{pattern}': {error}"))
        })?;
        Ok(Self {
            config,
            default_class,
            category_list,
            category_names,
            category_pattern,
        })
    }

    /// The configuration this component was built from.
    #[must_use]
    pub fn config(&self) -> &IntentDetectionCompConfig {
        &self.config
    }

    /// The display names of every class; index 0 is the default class.
    #[must_use]
    pub fn categories(&self) -> &[String] {
        &self.category_names
    }

    /// The class labels; index 0 is the default class (`分类0` / `Category0`).
    #[must_use]
    pub fn category_list(&self) -> &[String] {
        &self.category_list
    }

    /// The default class label (`分类0` / `Category0`).
    #[must_use]
    pub fn default_class(&self) -> &str {
        &self.default_class
    }

    /// The model this component calls, when one is configured.
    #[must_use]
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.config.model.as_ref()
    }

    /// Whether the English prompts, labels, and reasons are used.
    fn english(&self) -> bool {
        self.config.accept_language == LANGUAGE_EN
    }

    /// Renders the `category_info` placeholder: one `<label>: <display name>` line each.
    fn category_info(&self) -> String {
        self.category_list
            .iter()
            .zip(self.category_names.iter())
            .map(|(label, name)| format!("{label}: {name}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Renders the previous conversation as `Role: content` lines.
    ///
    /// Only the trailing `chat_history_max_turn` messages are used; messages without a
    /// known role are skipped.
    fn format_history(&self, history: &[Value]) -> String {
        let max_turn = usize::try_from(self.config.chat_history_max_turn).unwrap_or(usize::MAX);
        let start = history.len().saturating_sub(max_turn);
        let mut rendered = String::new();
        for message in &history[start..] {
            let Some(role) = message.get("role").and_then(Value::as_str) else {
                continue;
            };
            let Some(label) = self.role_label(role) else {
                continue;
            };
            let content = match message.get("content") {
                Some(Value::String(text)) => text.clone(),
                Some(Value::Null) | None => continue,
                Some(other) => other.to_string(),
            };
            rendered.push_str(label);
            rendered.push_str(": ");
            rendered.push_str(&content);
            rendered.push('\n');
        }
        rendered
    }

    /// The display label of `role` in the configured language.
    fn role_label(&self, role: &str) -> Option<&'static str> {
        match (self.english(), role) {
            (true, "user") => Some("User"),
            (true, "assistant") => Some("Assistant"),
            (true, "system") => Some("System"),
            (true, "tool") => Some("Tool"),
            (false, "user") => Some("用户"),
            (false, "assistant") => Some("助手"),
            (false, "system") => Some("系统"),
            (false, "tool") => Some("工具"),
            _ => None,
        }
    }

    /// Builds the `[system, user]` conversation sent to the model.
    fn build_messages(&self, query: &str, history: &[Value]) -> Vec<BaseMessage> {
        let (system, user) = if self.english() {
            (DEFAULT_SYSTEM_PROMPT_EN, DEFAULT_USER_PROMPT_EN)
        } else {
            (DEFAULT_SYSTEM_PROMPT_ZH, DEFAULT_USER_PROMPT_ZH)
        };
        let mut values = BTreeMap::new();
        values.insert(
            "user_prompt".to_owned(),
            Value::String(self.config.user_prompt.clone()),
        );
        values.insert(
            "category_info".to_owned(),
            Value::String(self.category_info()),
        );
        values.insert(
            "default_class".to_owned(),
            Value::String(self.default_class.clone()),
        );
        values.insert(
            "example_content".to_owned(),
            Value::String(self.config.example_content.join("\n\n")),
        );
        values.insert(
            "chat_history".to_owned(),
            Value::String(self.format_history(history)),
        );
        values.insert("input".to_owned(), Value::String(query.to_owned()));
        vec![
            BaseMessage::system(render_text(system, &values)),
            BaseMessage::user(render_text(user, &values)),
        ]
    }

    /// Reads the previous conversation from the session `__messages__` lane.
    ///
    /// A missing lane is an empty history; a non-array value is a history configuration
    /// error (`101061`).
    async fn read_history(&self, ctx: &ComponentContext) -> Result<Vec<Value>> {
        if !self.config.enable_history {
            return Ok(Vec::new());
        }
        match ctx.get_state(MESSAGES_STATE_KEY).await? {
            None => Ok(Vec::new()),
            Some(Value::Array(messages)) => Ok(messages),
            Some(other) => Err(intent_detection_history_config_error(&format!(
                "'{MESSAGES_STATE_KEY}' must be a JSON array, got {}",
                type_name(&other)
            ))),
        }
    }

    /// Appends the completed turn to the session `__messages__` lane.
    async fn append_history(&self, ctx: &ComponentContext, query: &str, reply: &str) -> Result<()> {
        if !self.config.enable_history {
            return Ok(());
        }
        let mut messages = self.read_history(ctx).await?;
        messages.push(serde_json::json!({"role": "user", "content": query}));
        messages.push(serde_json::json!({"role": "assistant", "content": reply}));
        let mut patch = Map::new();
        patch.insert(MESSAGES_STATE_KEY.to_owned(), Value::Array(messages));
        ctx.update_state(Value::Object(patch)).await
    }

    /// The `query` of the model path: required and a string (Python's input model).
    fn model_query(inputs: &Value) -> Result<String> {
        match inputs.get("query") {
            Some(Value::String(query)) => Ok(query.clone()),
            Some(other) => Err(intent_detection_input_param_error(&format!(
                "query must be a string, got {}",
                type_name(other)
            ))),
            None => Err(intent_detection_input_param_error(
                "query is required and must be a string",
            )),
        }
    }

    /// The query text of the rule path.
    ///
    /// `query` when it is a string and not `null`, otherwise `user_prompt`, otherwise the
    /// whole input rendered as text. A present, non-null, non-string `query` is an input
    /// parameter error.
    fn rule_query(&self, inputs: &Value) -> Result<String> {
        match inputs.get("query") {
            Some(Value::String(query)) => Ok(query.clone()),
            Some(Value::Null) | None => {
                if self.config.user_prompt.trim().is_empty() {
                    Ok(match inputs {
                        Value::String(text) => text.clone(),
                        Value::Null => String::new(),
                        other => other.to_string(),
                    })
                } else {
                    Ok(self.config.user_prompt.clone())
                }
            }
            Some(other) => Err(intent_detection_input_param_error(&format!(
                "query must be a string, got {}",
                type_name(other)
            ))),
        }
    }

    /// Calls the model and returns the raw reply text.
    async fn call_model(
        &self,
        model: &Arc<dyn Model>,
        query: &str,
        history: &[Value],
    ) -> Result<String> {
        let messages = self.build_messages(query, history);
        let request = ModelRequestConfig::new(model.model_name());
        let response = model.invoke(&messages, &request).await.map_err(|error| {
            intent_detection_invoke_call_failed(&format!(
                "failed to invoke llm and get result: {error}"
            ))
        })?;
        Ok(response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default())
    }

    /// Scores the configured keywords against `query` (`101052`-free, model-free path).
    ///
    /// Every category counts the distinct keywords (case-insensitively) appearing in the
    /// query; the highest count wins, ties go to the lowest class id. A category without
    /// configured keywords falls back to its display name.
    fn classify_by_keywords(&self, query: &str) -> (i64, String) {
        let haystack = query.to_lowercase();
        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (id, name) in self.category_names.iter().enumerate() {
            let keywords: Vec<&str> = match self.config.keywords.get(name) {
                Some(configured) if !configured.is_empty() => {
                    configured.iter().map(String::as_str).collect()
                }
                _ => vec![name.as_str()],
            };
            let mut seen = BTreeSet::new();
            let mut matched = Vec::new();
            for keyword in keywords {
                let needle = keyword.trim().to_lowercase();
                if needle.is_empty() || !seen.insert(needle.clone()) {
                    continue;
                }
                if haystack.contains(&needle) {
                    matched.push(keyword.trim().to_owned());
                }
            }
            let count = matched.len();
            if count == 0 {
                continue;
            }
            match &best {
                Some((_, best_count, _)) if *best_count >= count => {}
                _ => best = Some((id, count, matched)),
            }
        }
        match best {
            Some((id, _, matched)) => (
                i64::try_from(id).unwrap_or_default(),
                format!("keyword matched: {}", matched.join(", ")),
            ),
            None => (0, NO_KEYWORD_MATCH_REASON.to_owned()),
        }
    }

    /// Post-processes a raw model reply into `(class label, reason)`.
    fn post_process(&self, reply: &str) -> (String, String) {
        let extracted = extract_json_block(reply).unwrap_or_else(|| reply.to_owned());
        let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(&extracted) else {
            return (
                self.default_class.clone(),
                self.parse_fail_reason(&extracted),
            );
        };
        let Some(class_value) = parsed.get("class").filter(|value| !is_falsy(value)) else {
            return (
                self.default_class.clone(),
                self.class_missing_reason(&parsed),
            );
        };
        let raw_class = match class_value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        let mut label = sanitize_class(&raw_class);
        if let Some(matched) = self.category_pattern.find(&label) {
            let matched = matched.as_str();
            label = if self.english() {
                let digits: String = matched.chars().filter(char::is_ascii_digit).collect();
                if digits.is_empty() {
                    matched.to_owned()
                } else {
                    format!("Category{digits}")
                }
            } else {
                matched.to_owned()
            };
        }
        if !self.category_list.iter().any(|category| category == &label) {
            return (
                self.default_class.clone(),
                self.validation_fail_reason(&label),
            );
        }
        let reason = match parsed.get("reason") {
            Some(Value::String(text)) => text.clone(),
            _ => String::new(),
        };
        (label, reason)
    }

    /// The reason explaining an unparsable reply.
    fn parse_fail_reason(&self, result: &str) -> String {
        let template = if self.english() {
            JSON_PARSE_FAIL_REASON_EN
        } else {
            JSON_PARSE_FAIL_REASON_ZH
        };
        template.replace("{result}", result)
    }

    /// The reason explaining a reply without a `class` field.
    fn class_missing_reason(&self, parsed: &Map<String, Value>) -> String {
        let template = if self.english() {
            CLASS_KEY_MISSING_REASON_EN
        } else {
            CLASS_KEY_MISSING_REASON_ZH
        };
        template.replace("{result}", &Value::Object(parsed.clone()).to_string())
    }

    /// The reason explaining an unknown class label.
    fn validation_fail_reason(&self, label: &str) -> String {
        let template = if self.english() {
            VALIDATION_FAIL_REASON_EN
        } else {
            VALIDATION_FAIL_REASON_ZH
        };
        template
            .replace("{intent_class}", label)
            .replace("{category_list}", &python_list(&self.category_list))
    }

    /// The class id of `label`, or the default class id when it is unknown.
    fn class_id(&self, label: &str) -> i64 {
        self.category_list
            .iter()
            .position(|category| category == label)
            .and_then(|index| i64::try_from(index).ok())
            .unwrap_or_default()
    }

    /// The component output for `class_id`, with `reason` only when it is non-empty.
    fn output(&self, class_id: i64, reason: &str) -> Value {
        let name = usize::try_from(class_id)
            .ok()
            .and_then(|index| self.category_names.get(index))
            .cloned()
            .unwrap_or_default();
        let mut output = Map::new();
        output.insert("classification_id".to_owned(), Value::from(class_id));
        output.insert("category_name".to_owned(), Value::String(name));
        if !reason.is_empty() {
            output.insert("reason".to_owned(), Value::String(reason.to_owned()));
        }
        Value::Object(output)
    }
}

#[async_trait::async_trait]
impl Component for IntentDetectionComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("intent_detection", "intent_detection").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "intent_detection"
    }

    /// # Errors
    ///
    /// Returns `101050` for an unusable `query` input, `101052` when the model call fails,
    /// `101061` when the `__messages__` lane cannot be used as history, and propagates
    /// session and model failures.
    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        ctx.check_cancelled()?;
        if let Some(model) = &self.config.model {
            let query = Self::model_query(&inputs)?;
            let history = self.read_history(ctx).await?;
            let reply = self.call_model(model, &query, &history).await?;
            self.append_history(ctx, &query, &reply).await?;
            let (label, reason) = self.post_process(&reply);
            Ok(self.output(self.class_id(&label), &reason))
        } else {
            let query = self.rule_query(&inputs)?;
            let (class_id, reason) = self.classify_by_keywords(&query);
            Ok(self.output(class_id, &reason))
        }
    }
}

/// Routes to a set of successors selected by the detected classification id.
///
/// An [`openjiuwen_graph::Router`] reading `<component_id>.classification_id` from the
/// graph state. Because [`Component::route`] is deliberately not implemented by
/// [`IntentDetectionComponent`], the classification is written to the state as a normal
/// node output and the router is registered on the edges with
/// [`Workflow::add_conditional_connection`](crate::Workflow::add_conditional_connection):
///
/// ```
/// use openjiuwen_workflow::components::intent_detection::IntentRouter;
///
/// let router = IntentRouter::new(
///     "intent",
///     vec![(1, vec!["end_weather".to_owned()]), (2, vec!["end_travel".to_owned()])],
///     vec!["end_travel".to_owned()],
/// );
/// assert_eq!(router.component_id(), "intent");
/// assert_eq!(router.fallback(), ["end_travel"]);
/// ```
///
/// An unknown or absent classification id selects the fallback; an empty target list is
/// legitimate and ends the run without running a successor.
#[derive(Debug, Clone)]
pub struct IntentRouter {
    component_id: String,
    branches: Vec<(i64, Vec<String>)>,
    fallback: Vec<String>,
}

impl IntentRouter {
    /// Builds a router over `component_id` with one target list per classification id.
    ///
    /// Nothing is validated here; call [`IntentRouter::validate`] to reject a blank
    /// component id, a duplicate classification id, or a blank target.
    #[must_use]
    pub fn new(
        component_id: impl Into<String>,
        branches: Vec<(i64, Vec<String>)>,
        fallback: Vec<String>,
    ) -> Self {
        Self {
            component_id: component_id.into(),
            branches,
            fallback,
        }
    }

    /// The component id whose `classification_id` output is read.
    #[must_use]
    pub fn component_id(&self) -> &str {
        &self.component_id
    }

    /// The configured classification-id branches.
    #[must_use]
    pub fn branches(&self) -> &[(i64, Vec<String>)] {
        &self.branches
    }

    /// The targets selected for an absent or unknown classification id.
    #[must_use]
    pub fn fallback(&self) -> &[String] {
        &self.fallback
    }

    /// Checks that the router can be wired (Rust-only validation path, `101062`).
    ///
    /// # Errors
    ///
    /// Returns an [`ErrorKind::Workflow`](openjiuwen_common::error::ErrorKind::Workflow)
    /// error when the component id is blank, a classification id is configured twice, or
    /// a target id is blank.
    pub fn validate(&self) -> Result<()> {
        if self.component_id.trim().is_empty() {
            return Err(intent_detection_router_config_error(
                "component id must not be blank",
            ));
        }
        let mut seen = BTreeSet::new();
        for (class_id, targets) in &self.branches {
            if !seen.insert(*class_id) {
                return Err(intent_detection_router_config_error(&format!(
                    "classification id {class_id} is configured twice"
                )));
            }
            Self::validate_targets(targets)?;
        }
        Self::validate_targets(&self.fallback)
    }

    /// Rejects a blank target id.
    fn validate_targets(targets: &[String]) -> Result<()> {
        if targets.iter().any(|target| target.trim().is_empty()) {
            return Err(intent_detection_router_config_error(
                "target component ids must not be blank",
            ));
        }
        Ok(())
    }

    /// This router as the graph engine's router handle.
    #[must_use]
    pub fn router(self) -> Arc<dyn openjiuwen_graph::Router> {
        Arc::new(self)
    }
}

#[async_trait::async_trait]
impl openjiuwen_graph::Router for IntentRouter {
    fn name(&self) -> &'static str {
        "intent_detection"
    }

    async fn route(&self, ctx: &openjiuwen_graph::RouterContext) -> Result<Vec<String>> {
        let state = ctx.state().as_value();
        let class_id = state
            .get(&self.component_id)
            .and_then(|output| output.get("classification_id"))
            .and_then(Value::as_i64);
        match class_id {
            Some(id) => match self.branches.iter().find(|(branch, _)| *branch == id) {
                Some((_, targets)) => Ok(targets.clone()),
                None => Ok(self.fallback.clone()),
            },
            None => Ok(self.fallback.clone()),
        }
    }
}

/// Checks everything [`IntentDetectionComponent::new`] rejects.
fn validate_config(config: &IntentDetectionCompConfig) -> Result<()> {
    let language = config.accept_language.as_str();
    if language != LANGUAGE_ZH && language != LANGUAGE_EN {
        return Err(intent_detection_accept_language_invalid(&format!(
            "accept_language must be '{LANGUAGE_ZH}' or '{LANGUAGE_EN}', got '{language}'"
        )));
    }
    let default_name = default_display_name(language);
    let mut seen = BTreeSet::new();
    for (index, name) in config.category_name_list.iter().enumerate() {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(intent_detection_category_name_invalid(&format!(
                "category_name_list entry {index} is blank"
            )));
        }
        if trimmed == default_name {
            return Err(intent_detection_config_error(&format!(
                "category name '{trimmed}' collides with the default class display name"
            )));
        }
        if !seen.insert(trimmed.to_owned()) {
            return Err(intent_detection_category_name_duplicate(&format!(
                "duplicate category name '{trimmed}'"
            )));
        }
    }
    for (index, example) in config.example_content.iter().enumerate() {
        if example.trim().is_empty() {
            return Err(intent_detection_example_invalid(&format!(
                "example_content entry {index} is blank"
            )));
        }
    }
    if !config.user_prompt.is_empty() && config.user_prompt.trim().is_empty() {
        return Err(intent_detection_prompt_invalid(
            "user_prompt must not be whitespace-only",
        ));
    }
    if config.enable_history && config.chat_history_max_turn == 0 {
        return Err(intent_detection_history_config_error(
            "chat_history_max_turn must be greater than 0 when enable_history is set",
        ));
    }
    for (name, keywords) in &config.keywords {
        let known = name == default_name
            || config
                .category_name_list
                .iter()
                .any(|category| category == name);
        if !known {
            return Err(intent_detection_keyword_category_unknown(&format!(
                "keywords key '{name}' is not a category display name"
            )));
        }
        if keywords.iter().any(|keyword| keyword.trim().is_empty()) {
            return Err(intent_detection_keyword_invalid(&format!(
                "keywords entry of '{name}' must not be blank"
            )));
        }
    }
    Ok(())
}

/// The default class display name of `language`.
fn default_display_name(language: &str) -> &'static str {
    if language == LANGUAGE_EN {
        DEFAULT_NAME_EN
    } else {
        DEFAULT_NAME_ZH
    }
}

/// Renders `{{name}}` placeholders in one prompt string.
fn render_text(template: &str, values: &BTreeMap<String, Value>) -> String {
    match PromptTemplate::new(template).format(values).content {
        TemplateContent::Text(text) => text,
        // `PromptTemplate::new` always builds text content; join defensively so a
        // future change cannot panic here.
        TemplateContent::Messages(messages) => messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect(),
    }
}

/// The outermost `{…}` block of `reply` (Python's DOTALL `re.search(r'\{.*\}')`).
fn extract_json_block(reply: &str) -> Option<String> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end < start {
        return None;
    }
    Some(reply[start..=end].to_owned())
}

/// Whether `value` is falsy in Python's sense (used for the `class` presence check).
fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(flag) => !(*flag),
        Value::Number(number) => number.as_f64().is_some_and(|value| value == 0.0),
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
    }
}

/// Removes the characters Python strips from a `class` value.
fn sanitize_class(value: &str) -> String {
    value
        .chars()
        .filter(|character| !matches!(character, '\n' | ' ' | '"' | '\''))
        .collect()
}

/// Renders `items` the way a Python `list[str]` prints (`['a', 'b']`).
fn python_list(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|item| format!("'{item}'")).collect();
    format!("[{}]", quoted.join(", "))
}

/// The JSON type name of `value`, for a readable error.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;

    fn config() -> IntentDetectionCompConfig {
        IntentDetectionCompConfig::default()
            .with_category_names(vec!["weather".to_owned(), "travel".to_owned()])
    }

    fn component(config: IntentDetectionCompConfig) -> IntentDetectionComponent {
        IntentDetectionComponent::new(config).expect("the config is valid")
    }

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("intent-tests"));
        ComponentContext::new(session, "intent", "wf")
    }

    #[test]
    fn categories_are_prefixed_per_language() {
        let zh = component(config());
        assert_eq!(zh.default_class(), "分类0");
        assert_eq!(
            zh.category_list().to_vec(),
            vec!["分类0".to_owned(), "分类1".to_owned(), "分类2".to_owned()]
        );
        assert_eq!(
            zh.categories().to_vec(),
            vec![
                "默认意图".to_owned(),
                "weather".to_owned(),
                "travel".to_owned()
            ]
        );

        let en = component(config().with_accept_language("en"));
        assert_eq!(en.default_class(), "Category0");
        assert_eq!(
            en.category_list().to_vec(),
            vec![
                "Category0".to_owned(),
                "Category1".to_owned(),
                "Category2".to_owned()
            ]
        );
        assert_eq!(en.categories()[0], "Default intent");
    }

    #[test]
    fn no_categories_leaves_only_the_default_class() {
        let component = component(IntentDetectionCompConfig::default());
        assert_eq!(component.category_list(), ["分类0"]);
        assert_eq!(component.categories(), ["默认意图"]);
    }

    #[test]
    fn an_unsupported_language_is_code_101055() {
        let error = IntentDetectionComponent::new(config().with_accept_language("fr")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101055]"), "{error}");
        assert!(error.message().contains("'fr'"), "{error}");
    }

    #[test]
    fn a_blank_category_name_is_code_101053() {
        let error = IntentDetectionComponent::new(
            config().with_category_names(vec!["weather".to_owned(), "  ".to_owned()]),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101053]"), "{error}");
    }

    #[test]
    fn a_duplicate_category_name_is_code_101054() {
        let error = IntentDetectionComponent::new(
            config().with_category_names(vec!["weather".to_owned(), "weather".to_owned()]),
        )
        .unwrap_err();
        assert!(error.message().contains("[101054]"), "{error}");
        assert!(error.message().contains("weather"), "{error}");
    }

    #[test]
    fn a_category_colliding_with_the_default_name_is_code_101060() {
        let error = IntentDetectionComponent::new(
            config().with_category_names(vec!["默认意图".to_owned()]),
        )
        .unwrap_err();
        assert!(error.message().contains("[101060]"), "{error}");
    }

    #[test]
    fn an_unknown_keyword_category_is_code_101056() {
        let keywords = BTreeMap::from([("ghost".to_owned(), vec!["x".to_owned()])]);
        let error = IntentDetectionComponent::new(config().with_keywords(keywords)).unwrap_err();
        assert!(error.message().contains("[101056]"), "{error}");
        assert!(error.message().contains("ghost"), "{error}");
    }

    #[test]
    fn a_blank_keyword_is_code_101057() {
        let keywords = BTreeMap::from([("weather".to_owned(), vec![" ".to_owned()])]);
        let error = IntentDetectionComponent::new(config().with_keywords(keywords)).unwrap_err();
        assert!(error.message().contains("[101057]"), "{error}");
    }

    #[test]
    fn keyword_keys_may_use_the_default_display_name() {
        let keywords = BTreeMap::from([("默认意图".to_owned(), vec!["hello".to_owned()])]);
        let component = component(config().with_keywords(keywords));
        assert_eq!(
            component.classify_by_keywords("hello there"),
            (0, "keyword matched: hello".to_owned())
        );
    }

    #[test]
    fn a_blank_example_is_code_101058() {
        let error = IntentDetectionComponent::new(
            config().with_example_content(vec!["ok".to_owned(), String::new()]),
        )
        .unwrap_err();
        assert!(error.message().contains("[101058]"), "{error}");
    }

    #[test]
    fn a_whitespace_only_prompt_is_code_101059() {
        let error = IntentDetectionComponent::new(config().with_user_prompt("   ")).unwrap_err();
        assert!(error.message().contains("[101059]"), "{error}");
    }

    #[test]
    fn history_without_room_for_a_message_is_code_101061() {
        let error = IntentDetectionComponent::new(
            config()
                .with_enable_history(true)
                .with_chat_history_max_turn(0),
        )
        .unwrap_err();
        assert!(error.message().contains("[101061]"), "{error}");
    }

    #[test]
    fn an_english_reply_selects_the_configured_class() {
        let component = component(config().with_accept_language("en"));
        let (label, reason) =
            component.post_process(r#"{"class": "Category2", "reason": "travel"}"#);
        assert_eq!(label, "Category2");
        assert_eq!(reason, "travel");
        assert_eq!(component.class_id(&label), 2);
    }

    #[test]
    fn a_chinese_reply_selects_the_configured_class() {
        let component = component(config());
        let (label, reason) = component.post_process(r#"{"class": "分类1"}"#);
        assert_eq!(label, "分类1");
        assert_eq!(reason, "");
        assert_eq!(component.class_id(&label), 1);
    }

    #[test]
    fn class_matching_is_case_insensitive() {
        let component = component(config().with_accept_language("en"));
        assert_eq!(
            component
                .post_process(r#"{"class": "category1", "reason": "w"}"#)
                .0,
            "Category1"
        );
    }

    #[test]
    fn prose_around_the_json_is_ignored() {
        let component = component(config().with_accept_language("en"));
        let reply = r#"Sure! Here it is: {"class": "Category2"} — thanks."#;
        assert_eq!(component.post_process(reply).0, "Category2");
    }

    #[test]
    fn quoted_and_multiline_class_values_are_sanitised() {
        let component = component(config().with_accept_language("en"));
        let multiline = r#"{"class": "Category\n2"}"#;
        assert_eq!(component.post_process(multiline).0, "Category2");
        let quoted = r#"{"class": "\"category2\""}"#;
        assert_eq!(component.post_process(quoted).0, "Category2");
    }

    #[test]
    fn an_unknown_class_returns_the_default_with_the_validation_reason() {
        let component = component(config().with_accept_language("en"));
        let (label, reason) = component.post_process(r#"{"class": "Category9"}"#);
        assert_eq!(label, "Category0");
        assert_eq!(component.class_id(&label), 0);
        assert!(reason.contains("'Category9'"), "{reason}");
        assert!(
            reason.contains("['Category0', 'Category1', 'Category2']"),
            "{reason}"
        );
    }

    #[test]
    fn an_unparsable_reply_returns_the_default_with_the_parse_reason() {
        let english = component(config().with_accept_language("en"));
        let (label, reason) = english.post_process("no json here");
        assert_eq!(label, "Category0");
        assert!(reason.contains("no json here"), "{reason}");
        assert!(
            reason.contains("does not conform to valid JSON"),
            "{reason}"
        );

        let chinese = component(config());
        let (label, reason) = chinese.post_process("不是 JSON");
        assert_eq!(label, "分类0");
        assert!(reason.contains("不符合有效的JSON规范"), "{reason}");
    }

    #[test]
    fn a_missing_class_returns_the_default_with_the_missing_reason() {
        let component = component(config().with_accept_language("en"));
        let (label, reason) = component.post_process(r#"{"reason": "no class"}"#);
        assert_eq!(label, "Category0");
        assert!(reason.contains("missing the required 'class'"), "{reason}");
        assert!(reason.contains(r#"{"reason":"no class"}"#), "{reason}");
    }

    #[test]
    fn keyword_hit_selects_the_best_category() {
        let keywords = BTreeMap::from([
            (
                "weather".to_owned(),
                vec!["weather".to_owned(), "rain".to_owned()],
            ),
            ("travel".to_owned(), vec!["flight".to_owned()]),
        ]);
        let component = component(config().with_keywords(keywords));
        assert_eq!(
            component.classify_by_keywords("Rain or weather today?"),
            (1, "keyword matched: weather, rain".to_owned())
        );
        assert_eq!(
            component.classify_by_keywords("Book a FLIGHT"),
            (2, "keyword matched: flight".to_owned())
        );
    }

    #[test]
    fn a_keyword_tie_goes_to_the_lowest_class_id() {
        let keywords = BTreeMap::from([
            ("weather".to_owned(), vec!["rain".to_owned()]),
            ("travel".to_owned(), vec!["flight".to_owned()]),
        ]);
        let component = component(config().with_keywords(keywords));
        assert_eq!(component.classify_by_keywords("rain and flight").0, 1);
    }

    #[test]
    fn duplicate_keywords_count_once() {
        let keywords = BTreeMap::from([(
            "weather".to_owned(),
            vec!["rain".to_owned(), "rain".to_owned()],
        )]);
        let component = component(config().with_keywords(keywords));
        assert_eq!(
            component.classify_by_keywords("rain"),
            (1, "keyword matched: rain".to_owned())
        );
    }

    #[test]
    fn no_keyword_match_returns_the_default() {
        let keywords = BTreeMap::from([("travel".to_owned(), vec!["flight".to_owned()])]);
        let component = component(config().with_keywords(keywords));
        assert_eq!(
            component.classify_by_keywords("hello"),
            (0, NO_KEYWORD_MATCH_REASON.to_owned())
        );
    }

    #[test]
    fn display_names_are_the_fallback_keywords() {
        let component = component(config());
        assert_eq!(
            component.classify_by_keywords("any weather today?"),
            (1, "keyword matched: weather".to_owned())
        );
    }

    #[test]
    fn the_output_omits_an_empty_reason() {
        let component = component(config());
        assert_eq!(
            component.output(1, ""),
            json!({"classification_id": 1, "category_name": "weather"})
        );
        assert_eq!(
            component.output(0, "no keyword matched"),
            json!({
                "classification_id": 0,
                "category_name": "默认意图",
                "reason": "no keyword matched"
            })
        );
    }

    #[test]
    fn route_is_deliberately_not_implemented() {
        let component = component(config());
        assert!(component.route(&json!({"query": "hi"})).is_none());
    }

    #[test]
    fn identity_matches_the_python_component() {
        let component = component(config());
        assert_eq!(component.component_type(), "intent_detection");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        let card = component.card();
        assert_eq!(card.id, "intent_detection");
        assert_eq!(card.component_type, "intent_detection");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
    }

    #[tokio::test]
    async fn the_rule_path_answers_without_a_model() {
        let keywords = BTreeMap::from([("travel".to_owned(), vec!["flight".to_owned()])]);
        let component = component(config().with_keywords(keywords));
        let output = component
            .invoke(json!({"query": "book a flight"}), &context())
            .await
            .unwrap();
        assert_eq!(
            output,
            json!({
                "classification_id": 2,
                "category_name": "travel",
                "reason": "keyword matched: flight"
            })
        );
    }

    #[tokio::test]
    async fn a_non_string_query_is_code_101050() {
        let component = component(config());
        let error = component
            .invoke(json!({"query": 7}), &context())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101050]"), "{error}");
        assert!(error.message().contains("number"), "{error}");
    }

    #[tokio::test]
    async fn the_rule_path_falls_back_to_the_whole_input() {
        let component = component(config());
        let output = component
            .invoke(json!({"query": null, "text": "weather please"}), &context())
            .await
            .unwrap();
        assert_eq!(output["classification_id"], json!(1));
    }

    #[test]
    fn the_router_rejects_an_ambiguous_configuration() {
        let duplicate = IntentRouter::new(
            "intent",
            vec![(1, vec!["a".to_owned()]), (1, vec!["b".to_owned()])],
            Vec::new(),
        );
        let error = duplicate.validate().unwrap_err();
        assert!(error.message().contains("[101062]"), "{error}");

        let blank_id = IntentRouter::new(" ", Vec::new(), Vec::new());
        assert!(blank_id.validate().is_err());

        let blank_target = IntentRouter::new("intent", Vec::new(), vec![" ".to_owned()]);
        assert!(blank_target.validate().is_err());
    }

    #[test]
    fn the_router_exposes_its_configuration() {
        let router = IntentRouter::new(
            "intent",
            vec![(2, vec!["end_b".to_owned()])],
            vec!["end_a".to_owned()],
        );
        assert!(router.validate().is_ok());
        assert_eq!(router.component_id(), "intent");
        assert_eq!(router.branches(), [(2, vec!["end_b".to_owned()])]);
        assert_eq!(router.fallback(), ["end_a"]);
        let handle = router.router();
        assert_eq!(handle.name(), "intent_detection");
    }

    #[test]
    fn the_config_builders_are_chainable() {
        let config = IntentDetectionCompConfig::default()
            .with_model_id("m")
            .with_category_names(vec!["weather".to_owned()])
            .with_user_prompt("detect")
            .with_example_content(vec!["example".to_owned()])
            .with_enable_history(true)
            .with_chat_history_max_turn(2)
            .with_accept_language("en")
            .with_keywords(BTreeMap::from([(
                "weather".to_owned(),
                vec!["rain".to_owned()],
            )]));
        assert_eq!(config.model_id.as_deref(), Some("m"));
        assert_eq!(config.chat_history_max_turn, 2);
        assert_eq!(config.accept_language, "en");
        assert!(config.enable_history);
        // The runtime-only fields are not serialized.
        let wire = serde_json::to_value(&config).unwrap();
        assert!(wire.get("keywords").is_none(), "{wire}");
        assert!(wire.get("model").is_none(), "{wire}");
        let restored: IntentDetectionCompConfig = serde_json::from_value(wire).unwrap();
        assert!(restored.keywords.is_empty());
        assert!(restored.model.is_none());
    }
}
