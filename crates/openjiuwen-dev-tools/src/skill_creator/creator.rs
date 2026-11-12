//! Skill specs, requests, and the [`SkillCreator`] entry points.
//!
//! Two creation paths exist, mirroring the Python package:
//!
//! - [`SkillCreator::create`] renders a [`SkillSpec`] deterministically: the
//!   instructions are a `{{variable}}` template, so no model is involved.
//! - [`SkillCreator::create_with_model`] drafts a skill from a natural-language
//!   [`SkillRequest`] with one model call that must answer with a fenced JSON
//!   object (`{"name", "description", "instructions"}`).
//!
//! Both paths validate the resulting [`SkillDefinition`] before returning it, and
//! [`SkillCreator::write`] / [`SkillCreator::read`] delegate to
//! [`SkillBundle`] so callers never touch the filesystem layout themselves.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::skill_creator::bundle::SkillBundle;
use crate::skill_creator::definition::{SkillDefinition, SkillResource};
use crate::skill_creator::template;
use crate::util::extract_json_fence;

/// The default request language, matching the Python default wording.
const DEFAULT_LANGUAGE: &str = "zh-CN";

/// The `zh-CN` system prompt: schema instructions for the drafting model.
const SYSTEM_PROMPT_ZH: &str = "\
你是一个技能设计助手。根据用户需求生成一个可移植的技能定义，并只输出一个 ```json 代码块。
JSON 字段：
- name：只包含 ASCII 字母、数字、点、下划线或连字符，且以字母或数字开头
- description：一句话说明技能用途
- instructions：markdown 正文，说明使用步骤
- version（可选）：版本字符串
- metadata（可选）：字符串到任意值的映射";

/// The `en-US` system prompt: schema instructions for the drafting model.
const SYSTEM_PROMPT_EN: &str = "\
You design portable agent skills. Answer with exactly one ```json code block whose object has:
- name: ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit
- description: one sentence describing what the skill does
- instructions: markdown body with the usage steps
- version (optional): a version string
- metadata (optional): a string-keyed object of extra values";

/// A deterministic skill specification.
///
/// The instructions may contain `{{variable}}` placeholders; [`SkillCreator::create`]
/// renders them strictly against [`variables`](Self::variables).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillSpec {
    /// Portable skill name.
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Markdown instruction body; may contain `{{variable}}` placeholders.
    pub instructions: String,
    /// Values substituted into the instruction template.
    pub variables: BTreeMap<String, String>,
    /// Text resources to ship beside the document.
    pub resources: Vec<SkillResource>,
    /// Optional version string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Free-form metadata written into the frontmatter.
    pub metadata: BTreeMap<String, Value>,
}

impl SkillSpec {
    /// Creates a spec from its identity and instruction template.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
            ..Self::default()
        }
    }

    /// Adds one template variable.
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(name.into(), value.into());
        self
    }

    /// Appends a resource.
    #[must_use]
    pub fn with_resource(mut self, resource: SkillResource) -> Self {
        self.resources.push(resource);
        self
    }

    /// Sets the version string.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
}

/// A natural-language request for a model-drafted skill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillRequest {
    /// What the skill should do, in prose.
    pub request: String,
    /// Relative resource paths to declare as (initially empty) files.
    pub resource_hints: Vec<String>,
    /// Prompt language: `zh-CN` (default) or `en-US`.
    #[serde(default = "default_language")]
    pub language: String,
}

impl SkillRequest {
    /// Creates a request in the default `zh-CN` wording.
    #[must_use]
    pub fn new(request: impl Into<String>) -> Self {
        Self {
            request: request.into(),
            resource_hints: Vec::new(),
            language: DEFAULT_LANGUAGE.to_owned(),
        }
    }

    /// Selects the prompt language (`zh-CN` or `en-US`).
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Declares resource files the drafted skill should ship.
    #[must_use]
    pub fn with_resource_hints(
        mut self,
        hints: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.resource_hints = hints.into_iter().map(Into::into).collect();
        self
    }
}

impl Default for SkillRequest {
    /// An empty `zh-CN` request.
    fn default() -> Self {
        Self {
            request: String::new(),
            resource_hints: Vec::new(),
            language: default_language(),
        }
    }
}

/// Creates, templates, and persists skill definitions.
///
/// The creator is cheap to clone and holds an optional model: a creator built
/// with [`SkillCreator::new`] handles the deterministic path only, while
/// [`SkillCreator::with_model`] enables [`SkillCreator::create_with_model`].
#[derive(Clone, Default)]
pub struct SkillCreator {
    /// Model used by [`SkillCreator::create_with_model`], when configured.
    model: Option<Arc<dyn Model>>,
}

impl std::fmt::Debug for SkillCreator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SkillCreator")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

impl SkillCreator {
    /// Creates a deterministic creator with no model attached.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Attaches the model used for model-assisted creation.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Whether a model is attached.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Builds a validated definition from `spec`, rendering its template.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when an instruction placeholder has no variable (the message names every
    /// missing one) or when the resulting definition does not validate.
    pub fn create(&self, spec: &SkillSpec) -> Result<SkillDefinition> {
        let instructions = template::render(&spec.instructions, &spec.variables)?;
        let mut definition =
            SkillDefinition::new(spec.name.clone(), spec.description.clone(), instructions);
        definition.version.clone_from(&spec.version);
        definition.metadata.clone_from(&spec.metadata);
        definition.resources.clone_from(&spec.resources);
        definition.validate()?;
        Ok(definition)
    }

    /// Drafts a skill from a natural-language request with one model call.
    ///
    /// The model must answer with a fenced `json` object holding `name`,
    /// `description`, and `instructions` (optionally `version` and `metadata`).
    /// Every [`SkillRequest::resource_hints`] entry becomes a resource with empty
    /// content, ready for the author to fill in.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when no model is attached, the language is unsupported, the answer holds no
    /// JSON fence, the payload is malformed, the hints are invalid paths, or the
    /// resulting definition does not validate; model failures propagate as
    /// [`Error::llm`](openjiuwen_common::error::Error::llm).
    pub async fn create_with_model(&self, request: &SkillRequest) -> Result<SkillDefinition> {
        let Some(model) = self.model.as_ref() else {
            return Err(Error::validation(
                "model-assisted skill creation requires a creator built with SkillCreator::with_model",
            ));
        };
        let system = system_prompt(&request.language)?;
        let user = user_prompt(request);

        let completion = model
            .invoke(
                &[BaseMessage::system(system), BaseMessage::user(user)],
                &ModelRequestConfig::new(model.model_name()),
            )
            .await?;

        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        let Some(payload) = extract_json_fence(&answer) else {
            return Err(Error::validation(
                "model answer contained no ```json fenced skill definition",
            ));
        };
        let payload: ModelSkillPayload = serde_json::from_value(payload).map_err(|error| {
            Error::validation(format!("invalid skill definition JSON from model: {error}"))
        })?;

        let mut definition = SkillDefinition::new(
            payload.required("name")?,
            payload.required("description")?,
            payload.required("instructions")?,
        );
        definition.version = payload.version.filter(|version| !version.trim().is_empty());
        definition.metadata = payload.metadata;
        for hint in &request.resource_hints {
            definition
                .resources
                .push(SkillResource::new(hint.clone(), String::new()));
        }
        definition.validate()?;
        Ok(definition)
    }

    /// Writes `definition` as a bundle under `root`; see [`SkillBundle::write`].
    ///
    /// # Errors
    ///
    /// See [`SkillBundle::write`].
    pub async fn write(&self, definition: &SkillDefinition, root: &Path) -> Result<PathBuf> {
        SkillBundle::write(definition, root).await
    }

    /// Reads the bundle in `directory`; see [`SkillBundle::read`].
    ///
    /// # Errors
    ///
    /// See [`SkillBundle::read`].
    pub async fn read(&self, directory: &Path) -> Result<SkillDefinition> {
        SkillBundle::read(directory).await
    }
}

/// The JSON payload a drafting model is expected to return.
#[derive(Debug, Default, Deserialize)]
struct ModelSkillPayload {
    /// Skill name; required once parsed.
    #[serde(default)]
    name: Option<String>,
    /// Skill description; required once parsed.
    #[serde(default)]
    description: Option<String>,
    /// Instruction body; required once parsed.
    #[serde(default)]
    instructions: Option<String>,
    /// Optional version string.
    #[serde(default)]
    version: Option<String>,
    /// Optional metadata object.
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
}

impl ModelSkillPayload {
    /// Reads a required, non-blank field, naming it in the failure.
    fn required(&self, field: &str) -> Result<String> {
        let value = match field {
            "name" => self.name.as_deref(),
            "description" => self.description.as_deref(),
            "instructions" => self.instructions.as_deref(),
            other => {
                return Err(Error::validation(format!(
                    "unknown skill payload field '{other}'"
                )));
            }
        };
        match value {
            Some(value) if !value.trim().is_empty() => Ok(value.to_owned()),
            _ => Err(Error::validation(format!(
                "skill definition JSON from model is missing a non-blank '{field}'"
            ))),
        }
    }
}

/// Selects the system prompt for `language`.
fn system_prompt(language: &str) -> Result<&'static str> {
    match language.trim() {
        "" | "zh" | "zh-CN" | "zh_CN" => Ok(SYSTEM_PROMPT_ZH),
        "en" | "en-US" | "en_US" => Ok(SYSTEM_PROMPT_EN),
        other => Err(Error::validation(format!(
            "unsupported skill request language {other:?}: expected 'zh-CN' or 'en-US'"
        ))),
    }
}

/// Builds the user message: the request prose plus any resource hints.
fn user_prompt(request: &SkillRequest) -> String {
    let mut prompt = request.request.trim().to_owned();
    if !request.resource_hints.is_empty() {
        prompt.push_str("\n\nResource files to declare:\n");
        for hint in &request.resource_hints {
            prompt.push_str("- ");
            prompt.push_str(hint);
            prompt.push('\n');
        }
    }
    prompt
}

/// The default language code for [`SkillRequest`].
fn default_language() -> String {
    DEFAULT_LANGUAGE.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
    use serde_json::json;

    fn spec() -> SkillSpec {
        SkillSpec::new(
            "summarize",
            "Summarize a document",
            "Summarize {{document}} for {{audience}}.",
        )
        .with_variable("document", "the README")
        .with_variable("audience", "beginners")
        .with_version("1.0.0")
    }

    fn fenced(payload: Value) -> String {
        format!(
            "Here you go:\n```json\n{}\n```\n",
            serde_json::to_string_pretty(&payload).expect("payload serializes")
        )
    }

    #[test]
    fn create_renders_variables() {
        let definition = SkillCreator::new().create(&spec()).expect("creates");
        assert_eq!(
            definition.instructions,
            "Summarize the README for beginners."
        );
        assert_eq!(definition.version.as_deref(), Some("1.0.0"));
        assert!(!SkillCreator::new().has_model());
    }

    #[test]
    fn create_reports_missing_variables() {
        let broken = SkillSpec::new("summarize", "Summarize", "Use {{missing}} here");
        let error = SkillCreator::new()
            .create(&broken)
            .expect_err("missing variable");
        assert!(error.to_string().contains("missing"), "{error}");
    }

    #[test]
    fn create_validates_the_definition() {
        let invalid = SkillSpec::new("bad name", "Description", "Body");
        let error = SkillCreator::new().create(&invalid).expect_err("bad name");
        assert!(error.to_string().starts_with("validation:"), "{error}");
    }

    #[tokio::test]
    async fn create_with_model_parses_a_json_fence() {
        let model = ScriptedModel::text(fenced(json!({
            "name": "release-notes",
            "description": "Draft release notes",
            "instructions": "# Steps\n\n1. Collect commits.",
            "version": "0.2.0",
            "metadata": {"team": "sdk"}
        })));
        let creator = SkillCreator::with_model(Arc::new(model.clone()));
        let request = SkillRequest::new("Draft release notes from git log")
            .with_resource_hints(["references/style.md"]);

        let definition = creator.create_with_model(&request).await.expect("creates");
        assert_eq!(definition.name, "release-notes");
        assert_eq!(definition.instructions, "# Steps\n\n1. Collect commits.");
        assert_eq!(definition.version.as_deref(), Some("0.2.0"));
        assert_eq!(definition.metadata.get("team"), Some(&json!("sdk")));
        assert_eq!(definition.resources.len(), 1);
        assert_eq!(definition.resources[0].path, "references/style.md");

        assert_eq!(model.call_count(), 1);
        let call = model.last_call().expect("recorded call");
        assert!(call
            .last_user_text()
            .is_some_and(|text| text.contains("Draft release notes from git log")));
        assert!(call
            .last_user_text()
            .is_some_and(|text| text.contains("references/style.md")));
        assert_eq!(call.request.model_name, "scripted-model");
    }

    #[tokio::test]
    async fn create_with_model_selects_the_prompt_language() {
        let payload = fenced(json!({
            "name": "zh-skill",
            "description": "演示",
            "instructions": "步骤"
        }));
        let model = ScriptedModel::builder()
            .respond_text(payload.clone())
            .respond_text(payload)
            .build();
        let creator = SkillCreator::with_model(Arc::new(model.clone()));

        creator
            .create_with_model(&SkillRequest::new("演示技能"))
            .await
            .expect("creates in zh-CN");
        let first = model.last_call().expect("call");
        assert!(first.messages[0].text().contains("技能设计助手"));

        creator
            .create_with_model(&SkillRequest::new("demo skill").with_language("en-US"))
            .await
            .expect("creates in en-US");
        let second = model.last_call().expect("call");
        assert!(second.messages[0]
            .text()
            .contains("You design portable agent skills"));
        assert_eq!(model.call_count(), 2);
    }

    #[tokio::test]
    async fn create_with_model_rejects_a_plain_text_answer() {
        let model = ScriptedModel::text("Sure! Here is a skill: do the thing.");
        let creator = SkillCreator::with_model(Arc::new(model.clone()));
        let error = creator
            .create_with_model(&SkillRequest::new("anything"))
            .await
            .expect_err("no fence");
        assert!(error.to_string().contains("json"), "{error}");
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn create_with_model_rejects_incomplete_payloads() {
        let model = ScriptedModel::text(fenced(json!({"name": "only-name"})));
        let creator = SkillCreator::with_model(Arc::new(model));
        let error = creator
            .create_with_model(&SkillRequest::new("anything"))
            .await
            .expect_err("missing fields");
        assert!(error.to_string().contains("description"), "{error}");
    }

    #[tokio::test]
    async fn create_with_model_requires_a_model() {
        let error = SkillCreator::new()
            .create_with_model(&SkillRequest::new("anything"))
            .await
            .expect_err("no model");
        assert!(error.to_string().contains("with_model"), "{error}");
    }

    #[tokio::test]
    async fn create_with_model_rejects_unknown_languages_without_calling_the_model() {
        let model = ScriptedModel::text("unused");
        let creator = SkillCreator::with_model(Arc::new(model.clone()));
        let error = creator
            .create_with_model(&SkillRequest::new("anything").with_language("fr-FR"))
            .await
            .expect_err("unsupported language");
        assert!(error.to_string().contains("zh-CN"), "{error}");
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn create_with_model_rejects_invalid_resource_hints() {
        let model = ScriptedModel::text(fenced(json!({
            "name": "demo",
            "description": "Demo",
            "instructions": "Body"
        })));
        let creator = SkillCreator::with_model(Arc::new(model));
        let request = SkillRequest::new("demo").with_resource_hints(["../escape.md"]);
        let error = creator
            .create_with_model(&request)
            .await
            .expect_err("invalid hint");
        assert!(error.to_string().contains("../escape.md"), "{error}");
    }

    #[tokio::test]
    async fn write_and_read_through_the_creator() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let creator = SkillCreator::new();
        let definition = creator
            .create(&spec().with_resource(SkillResource::new("references/guide.md", "guide")))
            .expect("creates");

        let directory = creator
            .write(&definition, &workspace.path("skills"))
            .await
            .expect("writes");
        assert!(directory.ends_with("summarize"));
        assert!(tokio::fs::try_exists(directory.join("SKILL.md"))
            .await
            .expect("probe"));

        let parsed = creator.read(&directory).await.expect("reads");
        assert_eq!(parsed.name, definition.name);
        assert_eq!(parsed.instructions, definition.instructions);
    }

    #[test]
    fn request_defaults_to_chinese_wording() {
        let request = SkillRequest::default();
        assert_eq!(request.language, "zh-CN");
        assert_eq!(request.resource_hints.len(), 0);

        let decoded: SkillRequest =
            serde_json::from_value(json!({"request": "x"})).expect("deserializes");
        assert_eq!(decoded.language, "zh-CN");
    }

    #[test]
    fn spec_json_round_trip() {
        let spec = spec().with_resource(SkillResource::new("notes.md", "n"));
        let decoded: SkillSpec =
            serde_json::from_value(serde_json::to_value(&spec).expect("serializes"))
                .expect("deserializes");
        assert_eq!(decoded, spec);
    }
}
