//! Meta-template prompt construction.
//!
//! Ports `openjiuwen.dev_tools.prompt_builder.builder.meta_template_builder`
//! (`MetaTemplateBuilder`). The builder turns a raw instruction into the
//! standardized prompt the SDK asks its model to write: one assembly step picks
//! the general, planning, or a caller-registered meta-template, fills the
//! `{{instruction}}` and `{{tools}}` slots, and exactly one model call produces
//! the answer.
//!
//! # Deviations from Python
//!
//! - Validation is strict: a blank prompt, a missing or blank custom template
//!   name, an unregistered template name, and a blank registry name are
//!   `Error::Validation` instead of a logged warning or a silent `general`
//!   fallback.
//! - Registry keys are the caller-supplied names verbatim. Python's hidden
//!   `META_TEMPLATE_` prefix is deliberately not reproduced, so `register`,
//!   `get`, and `pop` share one uniform key space.
//! - A model answer with no choice or with blank text is `Error::Llm`; Python
//!   returns `response.content`, which may be `None`.
//! - [`MetaTemplateBuilder::stream_build`] yields only non-empty text deltas;
//!   Python yields every chunk's `content`, `None` included.
//! - Registration takes [`IntoMetaTemplate`] rather than `Into<PromptTemplate>`
//!   because the foundation crate implements `From<&str>`/`From<String>` for
//!   `TemplateContent`, not for [`PromptTemplate`]. The three accepted forms
//!   (`&str`, `String`, [`PromptTemplate`]) are the ones Python accepts.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::{BaseMessage, PromptTemplate};
use openjiuwen_foundation::tool::ToolInfo;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::language::Language;
use super::templates::{bank, system_template, user_template};

/// Which meta-template family [`MetaTemplateBuilder::build`] assembles.
///
/// Mirrors Python's `Literal["general", "plan", "other"]` argument and
/// serializes with the same `snake_case` tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplateType {
    /// The no-tools markdown-skeleton template; the default.
    #[default]
    General,
    /// The tool-aware planning template, which fills `{{tools}}`.
    Plan,
    /// A caller-registered template, named by
    /// [`MetaBuildOptions::custom_template_name`].
    Other,
}

/// Options for one [`MetaTemplateBuilder::build`] call.
///
/// Groups the Python keyword arguments of `build`/`stream_build` into one
/// value, so adding an option never changes the method signature. Defaults
/// match Python's: no tools, [`TemplateType::General`], no custom template, and
/// [`Language::ZhCn`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MetaBuildOptions {
    /// Tools the standardized prompt may reference (used by
    /// [`TemplateType::Plan`] and by custom templates).
    pub tools: Vec<ToolInfo>,
    /// The template family to assemble.
    pub template_type: TemplateType,
    /// Registry key of the template used by [`TemplateType::Other`].
    pub custom_template_name: Option<String>,
    /// Language of the bundled template bank.
    pub language: Language,
}

/// Conversion into a registry [`PromptTemplate`].
///
/// Implemented for the three forms Python's `register_meta_template` accepts:
/// `&str` and `String` become a single-text template (one user message on
/// render), and a [`PromptTemplate`] is stored unchanged so its message roles
/// survive. The trait exists because [`PromptTemplate`] is a foreign type
/// without `From<&str>`/`From<String>` implementations, so the standard
/// `Into<PromptTemplate>` bound cannot express these conversions.
pub trait IntoMetaTemplate {
    /// Converts this value into the template stored in the registry.
    fn into_meta_template(self) -> PromptTemplate;
}

impl IntoMetaTemplate for PromptTemplate {
    fn into_meta_template(self) -> PromptTemplate {
        self
    }
}

impl IntoMetaTemplate for &str {
    fn into_meta_template(self) -> PromptTemplate {
        PromptTemplate::new(self)
    }
}

impl IntoMetaTemplate for String {
    fn into_meta_template(self) -> PromptTemplate {
        PromptTemplate::new(self)
    }
}

/// Assembles meta-template messages and asks a model for the standardized
/// prompt.
///
/// The registry is per-builder and starts empty: bundled templates come from
/// [`super::templates`], and only [`TemplateType::Other`] consults the
/// registry. Cloning the builder is not supported (the registry is mutable);
/// share it behind a lock if several tasks need the same registrations.
pub struct MetaTemplateBuilder {
    /// Model used for the single standardization call.
    model: Arc<dyn Model>,
    /// Custom templates, keyed by the caller-supplied name.
    registry: BTreeMap<String, PromptTemplate>,
}

impl MetaTemplateBuilder {
    /// Creates an empty builder over `model`.
    ///
    /// The model is shared, never rebuilt, so the caller keeps ownership of its
    /// configuration. No template is registered yet.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            registry: BTreeMap::new(),
        }
    }

    /// Assembles the messages for `prompt` and makes exactly one model call.
    ///
    /// Returns the first choice's text unchanged.
    ///
    /// # Errors
    ///
    /// - `Error::Validation` when the prompt is blank, the custom template name
    ///   is missing, blank, or unregistered, or the tool list cannot be
    ///   serialized.
    /// - `Error::Llm` when the response carries no choice or its text is blank.
    /// - Any error the model returns, unchanged.
    pub async fn build(&self, prompt: &str, options: &MetaBuildOptions) -> Result<String> {
        let messages = assemble(&self.registry, prompt, options)?;
        let request = ModelRequestConfig::new(self.model.model_name());
        let response = self.model.invoke(&messages, &request).await?;
        answer_text(&response, self.model.model_name())
    }

    /// Assembles the messages once and streams the model answer.
    ///
    /// The builder has no non-streamed pre-stage, so the whole model answer is
    /// streamed. The returned stream owns a clone of the [`Arc<dyn Model>`] and
    /// of the registry, so the builder is not borrowed and may be dropped.
    ///
    /// Only non-empty text deltas are forwarded as `Ok(String)`; usage, done,
    /// and empty deltas are skipped. An assembly error is yielded as the single
    /// `Err` item and ends the stream, and a model error is yielded as `Err`
    /// before the stream terminates.
    pub fn stream_build(
        &self,
        prompt: impl Into<String> + Send + 'static,
        options: MetaBuildOptions,
    ) -> BoxStream<'static, Result<String>> {
        let model = Arc::clone(&self.model);
        let registry = self.registry.clone();
        let prompt = prompt.into();

        Box::pin(async_stream::stream! {
            let messages = match assemble(&registry, &prompt, &options) {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let request = ModelRequestConfig::new(model.model_name());
            let mut stream = model.stream(messages, request);
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamChunk::Delta { content: Some(text), .. }) if !text.is_empty() => {
                        yield Ok(text);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        })
    }

    /// Registers `template` under `name`, replacing an existing entry.
    ///
    /// # Errors
    ///
    /// `Error::Validation` when `name` is empty or whitespace-only.
    pub fn register_meta_template(
        &mut self,
        name: impl Into<String>,
        template: impl IntoMetaTemplate,
    ) -> Result<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::validation(
                "meta-template name cannot be empty or blank",
            ));
        }
        self.registry.insert(name, template.into_meta_template());
        Ok(())
    }

    /// Looks up a registered template by its exact name.
    #[must_use]
    pub fn get_meta_template(&self, name: &str) -> Option<&PromptTemplate> {
        self.registry.get(name)
    }

    /// Removes and returns the template registered under `name`, if any.
    pub fn pop_meta_template(&mut self, name: &str) -> Option<PromptTemplate> {
        self.registry.remove(name)
    }

    /// Every registered key, sorted (the registry is a [`BTreeMap`]).
    #[must_use]
    pub fn registered_names(&self) -> Vec<String> {
        self.registry.keys().cloned().collect()
    }
}

/// Assembles the messages `build` and `stream_build` send.
///
/// Shared by both entry points so the streamed and non-streamed calls always
/// see the same conversation.
fn assemble(
    registry: &BTreeMap<String, PromptTemplate>,
    prompt: &str,
    options: &MetaBuildOptions,
) -> Result<Vec<BaseMessage>> {
    if prompt.trim().is_empty() {
        return Err(Error::validation("prompt cannot be empty"));
    }

    let prompt_bank = bank(options.language);
    match options.template_type {
        TemplateType::General => {
            let mut messages = system_template(prompt_bank.meta_general_system).to_messages();
            messages.extend(
                user_template(prompt_bank.meta_general_user)
                    .format(&instruction_values(prompt))
                    .to_messages(),
            );
            Ok(messages)
        }
        TemplateType::Plan => {
            let tools = render_tools(&options.tools)?;
            let mut messages = system_template(prompt_bank.meta_plan_system).to_messages();
            messages.extend(
                user_template(prompt_bank.meta_plan_user)
                    .format(&instruction_and_tools(prompt, &tools))
                    .to_messages(),
            );
            Ok(messages)
        }
        TemplateType::Other => custom_messages(registry, prompt, options),
    }
}

/// Builds the `instruction`-only value map used by the general template.
fn instruction_values(prompt: &str) -> BTreeMap<String, Value> {
    BTreeMap::from([("instruction".to_owned(), Value::String(prompt.to_owned()))])
}

/// Builds the `instruction` plus `tools` value map used by plan and custom
/// templates.
fn instruction_and_tools(prompt: &str, tools: &str) -> BTreeMap<String, Value> {
    let mut values = instruction_values(prompt);
    values.insert("tools".to_owned(), Value::String(tools.to_owned()));
    values
}

/// Renders the tool list for the `{{tools}}` slot.
///
/// An empty list renders Python's `str(None)`, the literal `"None"`; otherwise
/// the compact OpenAI-shaped JSON array is used.
fn render_tools(tools: &[ToolInfo]) -> Result<String> {
    if tools.is_empty() {
        return Ok("None".to_owned());
    }
    serde_json::to_string(tools)
        .map_err(|error| Error::validation(format!("failed to serialize tools: {error}")))
}

/// Resolves and formats a registered template for [`TemplateType::Other`].
fn custom_messages(
    registry: &BTreeMap<String, PromptTemplate>,
    prompt: &str,
    options: &MetaBuildOptions,
) -> Result<Vec<BaseMessage>> {
    let name = options.custom_template_name.as_deref().unwrap_or("").trim();
    if name.is_empty() {
        return Err(Error::validation(
            "template_type \"other\" requires a custom_template_name",
        ));
    }
    let Some(template) = registry.get(name) else {
        return Err(Error::validation(format!(
            "custom meta-template {name:?} is not registered"
        )));
    };
    let tools = render_tools(&options.tools)?;
    Ok(template
        .format(&instruction_and_tools(prompt, &tools))
        .to_messages())
}

/// Extracts the first choice's text, rejecting an unusable answer.
fn answer_text(response: &ChatResponse, model_name: &str) -> Result<String> {
    let Some(message) = response.message() else {
        return Err(Error::llm(format!(
            "model {model_name:?} returned no completion choice"
        )));
    };
    let text = message.text().into_owned();
    if text.trim().is_empty() {
        return Err(Error::llm(format!(
            "model {model_name:?} returned an empty answer"
        )));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use openjiuwen_common::error::{ErrorKind, Result};
    use openjiuwen_foundation::prompt::Role;
    use openjiuwen_test_support::{CallMode, ScriptedError, ScriptedModel};

    use super::*;

    /// Builds a recording model and the `Arc<dyn Model>` view handed to the
    /// builder; both share one script and call history.
    fn scripted(text: &str) -> (ScriptedModel, Arc<dyn Model>) {
        let recorder = ScriptedModel::builder()
            .model_name("scripted-test")
            .respond_text(text)
            .build();
        let model: Arc<dyn Model> = Arc::new(recorder.clone());
        (recorder, model)
    }

    /// A plan-mode tool with a stable name, used to assert JSON rendering.
    fn search_tool() -> ToolInfo {
        ToolInfo::new(
            "search",
            "Search the web",
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }

    /// The text a template renders to, for message assertions.
    fn rendered(template: &str, key: &str, value: &str) -> String {
        user_template(template).format_one(key, value).to_messages()[0]
            .text()
            .into_owned()
    }

    #[tokio::test]
    async fn general_mode_sends_one_system_and_one_user_message() {
        let (recorder, model) = scripted("standardized prompt");
        let builder = MetaTemplateBuilder::new(Arc::clone(&model));

        let answer = builder
            .build("Write a haiku", &MetaBuildOptions::default())
            .await
            .unwrap();

        assert_eq!(answer, "standardized prompt");
        assert_eq!(recorder.call_count(), 1);
        let call = recorder.last_call().unwrap();
        assert_eq!(call.messages.len(), 2);
        assert_eq!(call.messages[0].role, Role::System);
        assert_eq!(
            call.messages[0].text(),
            bank(Language::ZhCn).meta_general_system
        );
        assert_eq!(call.messages[1].role, Role::User);
        assert_eq!(
            call.messages[1].text(),
            rendered(
                bank(Language::ZhCn).meta_general_user,
                "instruction",
                "Write a haiku"
            )
        );
        assert!(call.messages[1].text().contains("Write a haiku"));
        assert!(!call.messages[1].text().contains("{{instruction}}"));
        assert_eq!(call.request.model_name, model.model_name());
    }

    #[tokio::test]
    async fn general_mode_does_not_interpolate_tools() {
        let (recorder, model) = scripted("answer");
        let builder = MetaTemplateBuilder::new(model);

        builder
            .build("Summarize", &MetaBuildOptions::default())
            .await
            .unwrap();
        let without_tools = recorder.last_call().unwrap().messages[1]
            .text()
            .into_owned();

        let options = MetaBuildOptions {
            tools: vec![search_tool()],
            ..MetaBuildOptions::default()
        };
        builder.build("Summarize", &options).await.unwrap();
        let with_tools = recorder.last_call().unwrap().messages[1]
            .text()
            .into_owned();

        assert_eq!(without_tools, with_tools);
        assert!(!with_tools.contains("search"));
        assert!(!with_tools.contains("{{tools}}"));
    }

    #[tokio::test]
    async fn plan_mode_interpolates_the_prompt_and_tool_json() {
        let (recorder, model) = scripted("plan");
        let builder = MetaTemplateBuilder::new(model);
        let options = MetaBuildOptions {
            tools: vec![search_tool()],
            template_type: TemplateType::Plan,
            ..MetaBuildOptions::default()
        };

        builder.build("Book a flight", &options).await.unwrap();

        let call = recorder.last_call().unwrap();
        assert_eq!(call.messages.len(), 2);
        assert_eq!(
            call.messages[0].text(),
            bank(Language::ZhCn).meta_plan_system
        );
        let user = call.messages[1].text();
        assert_eq!(call.messages[1].role, Role::User);
        assert!(user.contains("Book a flight"));
        assert!(user.contains("\"name\":\"search\""));
        assert!(user.contains("Search the web"));
        assert!(!user.contains("{{instruction}}"));
        assert!(!user.contains("{{tools}}"));
    }

    #[tokio::test]
    async fn plan_mode_renders_none_for_empty_tools() {
        let (recorder, model) = scripted("plan");
        let builder = MetaTemplateBuilder::new(model);
        let options = MetaBuildOptions {
            template_type: TemplateType::Plan,
            ..MetaBuildOptions::default()
        };

        builder.build("Plan the day", &options).await.unwrap();

        let user = recorder.last_call().unwrap().messages[1]
            .text()
            .into_owned();
        assert!(user.contains("None"));
        assert!(!user.contains("{{tools}}"));
        assert!(!user.contains("\"type\":\"function\""));
    }

    #[tokio::test]
    async fn english_language_selects_the_en_bank() {
        let (recorder, model) = scripted("answer");
        let builder = MetaTemplateBuilder::new(model);
        let options = MetaBuildOptions {
            language: Language::EnUs,
            ..MetaBuildOptions::default()
        };

        builder.build("Write a poem", &options).await.unwrap();

        let call = recorder.last_call().unwrap();
        let en_bank = crate::prompt_builder::templates::en_us::BANK;
        assert_eq!(call.messages[0].text(), en_bank.meta_general_system);
        assert_eq!(
            call.messages[1].text(),
            rendered(en_bank.meta_general_user, "instruction", "Write a poem")
        );
        assert_ne!(
            call.messages[0].text(),
            bank(Language::ZhCn).meta_general_system
        );
    }

    #[tokio::test]
    async fn blank_prompts_are_rejected_without_a_model_call() {
        let (recorder, model) = scripted("unused");
        let builder = MetaTemplateBuilder::new(model);

        for prompt in ["", "   ", "   \n\t"] {
            let error = builder
                .build(prompt, &MetaBuildOptions::default())
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{prompt:?}");
        }
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn other_mode_requires_a_custom_template_name() {
        let (recorder, model) = scripted("unused");
        let builder = MetaTemplateBuilder::new(model);

        for name in [None, Some(String::from("   "))] {
            let options = MetaBuildOptions {
                template_type: TemplateType::Other,
                custom_template_name: name.clone(),
                ..MetaBuildOptions::default()
            };
            let error = builder.build("Rewrite", &options).await.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{name:?}");
            assert!(
                error.to_string().contains("template name")
                    || error.to_string().contains("custom_template_name"),
                "{error}"
            );
        }
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn other_mode_rejects_an_unregistered_name() {
        let (recorder, model) = scripted("unused");
        let builder = MetaTemplateBuilder::new(model);
        let options = MetaBuildOptions {
            template_type: TemplateType::Other,
            custom_template_name: Some(String::from("missing")),
            ..MetaBuildOptions::default()
        };

        let error = builder.build("Rewrite", &options).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("missing"));
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn other_mode_formats_a_registered_text_template() {
        let (recorder, model) = scripted("rewritten");
        let mut builder = MetaTemplateBuilder::new(model);
        builder
            .register_meta_template("custom", "Rewrite: {{instruction}}")
            .unwrap();
        let options = MetaBuildOptions {
            template_type: TemplateType::Other,
            custom_template_name: Some(String::from("custom")),
            ..MetaBuildOptions::default()
        };

        builder.build("first draft", &options).await.unwrap();
        let first = recorder.last_call().unwrap().messages.clone();
        builder.build("second draft", &options).await.unwrap();
        let second = recorder.last_call().unwrap().messages.clone();

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].role, Role::User);
        assert!(first[0].text().contains("first draft"));
        assert!(!first[0].text().contains("{{instruction}}"));
        // Formatting happens at build time, not at registration time.
        assert!(second[0].text().contains("second draft"));
        assert!(!second[0].text().contains("first draft"));
    }

    #[tokio::test]
    async fn other_mode_preserves_a_registered_message_template_roles() {
        let (recorder, model) = scripted("rewritten");
        let mut builder = MetaTemplateBuilder::new(model);
        let template = PromptTemplate::from_messages(vec![BaseMessage::system(
            "You rewrite prompts: {{instruction}}",
        )]);
        builder
            .register_meta_template("system-custom", template)
            .unwrap();
        let options = MetaBuildOptions {
            template_type: TemplateType::Other,
            custom_template_name: Some(String::from("system-custom")),
            ..MetaBuildOptions::default()
        };

        builder.build("be concise", &options).await.unwrap();

        let messages = recorder.last_call().unwrap().messages;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::System);
        assert!(messages[0].text().contains("be concise"));
    }

    #[test]
    fn registry_round_trip_replaces_and_reports_sorted_names() {
        let (_recorder, model) = scripted("unused");
        let mut builder = MetaTemplateBuilder::new(model);
        assert_eq!(builder.registered_names(), Vec::<String>::new());
        assert!(builder.get_meta_template("b").is_none());

        builder
            .register_meta_template("b", "B: {{instruction}}")
            .unwrap();
        builder
            .register_meta_template("a", PromptTemplate::named("a", "A"))
            .unwrap();
        assert_eq!(
            builder.registered_names(),
            vec!["a".to_owned(), "b".to_owned()]
        );
        assert!(builder.get_meta_template("b").is_some());

        builder
            .register_meta_template("b", "B2: {{instruction}}")
            .unwrap();
        let replaced = builder.get_meta_template("b").unwrap();
        assert_eq!(replaced.to_messages()[0].text(), "B2: {{instruction}}");

        let popped = builder.pop_meta_template("b").unwrap();
        assert_eq!(popped.to_messages()[0].text(), "B2: {{instruction}}");
        assert!(builder.pop_meta_template("b").is_none());
        assert!(builder.get_meta_template("b").is_none());
        assert_eq!(builder.registered_names(), vec!["a".to_owned()]);
    }

    #[test]
    fn registering_a_blank_name_is_a_validation_error() {
        let (_recorder, model) = scripted("unused");
        let mut builder = MetaTemplateBuilder::new(model);

        for name in ["", " ", "\t\n"] {
            let error = builder.register_meta_template(name, "text").unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{name:?}");
        }
        assert_eq!(builder.registered_names(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn an_empty_model_answer_is_an_llm_error() {
        let recorder = ScriptedModel::builder().respond_text("").build();
        let model: Arc<dyn Model> = Arc::new(recorder.clone());
        let builder = MetaTemplateBuilder::new(model);

        let error = builder
            .build("Write something", &MetaBuildOptions::default())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert_eq!(recorder.call_count(), 1);
    }

    #[tokio::test]
    async fn stream_build_forwards_only_text_deltas() {
        let (recorder, model) = scripted("streamed answer");
        let builder = MetaTemplateBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build("Write a poem", MetaBuildOptions::default())
            .collect()
            .await;

        assert!(items.iter().all(std::result::Result::is_ok), "{items:?}");
        let text: String = items.into_iter().map(|item| item.unwrap()).collect();
        assert_eq!(text, "streamed answer");
        assert_eq!(recorder.call_count(), 1);
        let call = recorder.last_call().unwrap();
        assert_eq!(call.mode, CallMode::Stream);
        assert_eq!(call.messages.len(), 2);
    }

    #[tokio::test]
    async fn stream_build_yields_one_assembly_error_item() {
        let (recorder, model) = scripted("unused");
        let builder = MetaTemplateBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build("   ", MetaBuildOptions::default())
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Validation);
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn stream_build_yields_a_model_error_and_stops() {
        let recorder = ScriptedModel::builder()
            .model_name("scripted-test")
            .fail(ScriptedError::llm("boom"))
            .build();
        let model: Arc<dyn Model> = Arc::new(recorder.clone());
        let builder = MetaTemplateBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build("Write a poem", MetaBuildOptions::default())
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Llm);
    }

    #[test]
    fn options_serde_round_trips_with_python_tags() {
        let json = r#"{"tools":[],"template_type":"plan","custom_template_name":null,"language":"en-US","unknown":1}"#;
        let options: MetaBuildOptions = serde_json::from_str(json).unwrap();
        assert_eq!(options.tools, Vec::<ToolInfo>::new());
        assert_eq!(options.template_type, TemplateType::Plan);
        assert!(options.custom_template_name.is_none());
        assert_eq!(options.language, Language::EnUs);

        assert_eq!(
            serde_json::to_string(&TemplateType::General).unwrap(),
            "\"general\""
        );
        assert_eq!(
            serde_json::to_string(&TemplateType::Plan).unwrap(),
            "\"plan\""
        );
        assert_eq!(
            serde_json::to_string(&TemplateType::Other).unwrap(),
            "\"other\""
        );
    }

    #[test]
    fn default_options_are_general_zh_cn_without_tools() {
        let options = MetaBuildOptions::default();
        assert_eq!(options.tools, Vec::<ToolInfo>::new());
        assert_eq!(options.template_type, TemplateType::General);
        assert!(options.custom_template_name.is_none());
        assert_eq!(options.language, Language::ZhCn);
    }
}
