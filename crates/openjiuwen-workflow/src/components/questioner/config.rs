//! Configuration of the `questioner` component.
//!
//! Ports Python's `QuestionerConfig` dataclass and its `FieldInfo` pydantic model;
//! the component, its state machine and the prompt templates live in the sibling
//! `prompts` module and the parent module.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_foundation::Model;

use super::RESPONSE_TYPE_REPLY_DIRECTLY;

/// One field the component collects from the conversation.
///
/// The wire shape matches Python's `FieldInfo` pydantic model: `field_type` is
/// serialized as `"type"`, `default_value` defaults to `""`, and `field_type` to
/// `"string"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FieldInfo {
    /// The key the extracted value is stored under.
    pub field_name: String,
    /// Human-readable description used by the extraction prompt.
    pub description: String,
    /// The JSON type the value is converted to (`string`, `integer`, `number`, `boolean`).
    #[serde(rename = "type")]
    pub field_type: String,
    /// Chinese name used by the continue-ask question; falls back to `description`.
    pub cn_field_name: String,
    /// Whether the round keeps asking until this field is extracted.
    pub required: bool,
    /// Value filled in when the model does not extract the field.
    pub default_value: Value,
}

impl Default for FieldInfo {
    fn default() -> Self {
        Self {
            field_name: String::new(),
            description: String::new(),
            field_type: "string".to_owned(),
            cn_field_name: String::new(),
            required: false,
            default_value: Value::String(String::new()),
        }
    }
}

/// Configuration of a [`QuestionerComponent`](crate::components::questioner::QuestionerComponent).
///
/// Field names and defaults match Python's `QuestionerConfig` dataclass. The struct is
/// [`Default`], `snake_case` on the wire, tolerant of unknown fields, and every setter
/// consumes and returns `self`. The model is injected at runtime and is therefore not
/// serialized.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct QuestionerConfig {
    /// The model used for field extraction; `None` leaves only `question_content` mode.
    #[serde(skip)]
    pub model: Option<Arc<dyn Model>>,
    /// Logical model id carried for configuration parity (Python's `Runner` lookup).
    pub model_id: Option<String>,
    /// The answer shape; only `"reply_directly"` is supported.
    pub response_type: String,
    /// Question template rendered from the node inputs (`{{name}}` placeholders).
    pub question_content: String,
    /// Whether key fields are extracted from the conversation with the model.
    pub extract_fields_from_response: bool,
    /// The fields the component collects.
    pub field_names: Vec<FieldInfo>,
    /// How many answers one round accepts before it fails.
    pub max_response: u32,
    /// Whether chat history is injected; deferred (see the module docs).
    pub with_chat_history: bool,
    /// How many history rounds Python would inject; deferred (see the module docs).
    pub chat_history_max_rounds: u32,
    /// Extra instructions rendered into the prompt's `## Constraints` section.
    pub extra_prompt_for_fields_extraction: String,
    /// Sample JSON rendered into the prompt's `## Examples` section.
    pub example_content: String,
    /// Prompt and question language: `"en"` selects English, anything else Chinese.
    pub accept_language: String,
}

impl Default for QuestionerConfig {
    fn default() -> Self {
        Self {
            model: None,
            model_id: None,
            response_type: RESPONSE_TYPE_REPLY_DIRECTLY.to_owned(),
            question_content: String::new(),
            extract_fields_from_response: true,
            field_names: Vec::new(),
            max_response: 3,
            with_chat_history: false,
            chat_history_max_rounds: 5,
            extra_prompt_for_fields_extraction: String::new(),
            example_content: String::new(),
            accept_language: "zh".to_owned(),
        }
    }
}

impl fmt::Debug for QuestionerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuestionerConfig")
            .field("model", &self.model.as_deref().map(Model::model_name))
            .field("model_id", &self.model_id)
            .field("response_type", &self.response_type)
            .field("question_content", &self.question_content)
            .field(
                "extract_fields_from_response",
                &self.extract_fields_from_response,
            )
            .field("field_names", &self.field_names)
            .field("max_response", &self.max_response)
            .field("with_chat_history", &self.with_chat_history)
            .field("chat_history_max_rounds", &self.chat_history_max_rounds)
            .field(
                "extra_prompt_for_fields_extraction",
                &self.extra_prompt_for_fields_extraction,
            )
            .field("example_content", &self.example_content)
            .field("accept_language", &self.accept_language)
            .finish()
    }
}

impl QuestionerConfig {
    /// Sets the extraction model.
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

    /// Sets the answer shape.
    #[must_use]
    pub fn with_response_type(mut self, response_type: impl Into<String>) -> Self {
        self.response_type = response_type.into();
        self
    }

    /// Sets the question template.
    #[must_use]
    pub fn with_question_content(mut self, question_content: impl Into<String>) -> Self {
        self.question_content = question_content.into();
        self
    }

    /// Enables (or disables) field extraction from the conversation.
    #[must_use]
    pub fn with_extract_fields_from_response(mut self, extract: bool) -> Self {
        self.extract_fields_from_response = extract;
        self
    }

    /// Replaces the collected fields.
    #[must_use]
    pub fn with_field_names(mut self, field_names: Vec<FieldInfo>) -> Self {
        self.field_names = field_names;
        self
    }

    /// Sets how many answers one round accepts.
    #[must_use]
    pub fn with_max_response(mut self, max_response: u32) -> Self {
        self.max_response = max_response;
        self
    }

    /// Enables (or disables) chat-history injection; deferred (see the module docs).
    #[must_use]
    pub fn with_chat_history(mut self, with_chat_history: bool) -> Self {
        self.with_chat_history = with_chat_history;
        self
    }

    /// Sets how many history rounds would be injected; deferred (see the module docs).
    #[must_use]
    pub fn with_chat_history_max_rounds(mut self, rounds: u32) -> Self {
        self.chat_history_max_rounds = rounds;
        self
    }

    /// Sets the extra extraction constraints.
    #[must_use]
    pub fn with_extra_prompt_for_fields_extraction(
        mut self,
        extra_prompt: impl Into<String>,
    ) -> Self {
        self.extra_prompt_for_fields_extraction = extra_prompt.into();
        self
    }

    /// Sets the extraction example.
    #[must_use]
    pub fn with_example_content(mut self, example_content: impl Into<String>) -> Self {
        self.example_content = example_content.into();
        self
    }

    /// Sets the prompt and question language (`"en"` or `"zh"`).
    #[must_use]
    pub fn with_accept_language(mut self, accept_language: impl Into<String>) -> Self {
        self.accept_language = accept_language.into();
        self
    }
}
