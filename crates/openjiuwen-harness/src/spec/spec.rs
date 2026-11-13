//! The declarative [`DeepAgentSpec`] template and its leaf specification types.
//!
//! Ports the serializable half of `openjiuwen/harness/schema/deep_agent_spec.py`
//! and the identity/persona subset of `schema/extension_spec.py` as pure serde
//! data. Live handles (`Model`, `Tool`, rails, `SysOperation`) are never part of
//! a template: model settings are plain hints and tools are model-facing names
//! resolved through [`BuiltinTool::from_name`](crate::tools::BuiltinTool::from_name).
//!
//! # Formats
//!
//! [`DeepAgentSpec::from_path`] dispatches on the file extension:
//!
//! - `.yaml` / `.yml` — a YAML mapping.
//! - `.json` — a JSON object.
//! - `.md` / `.markdown` — a `---`-fenced YAML front-matter mapping plus a body
//!   that becomes the agent's system prompt (the same naive fence split
//!   `SKILL.md` uses).
//!
//! A file that omits `name` takes its file stem, so
//! `.openjiuwen/agents/reviewer.yaml` can be an unnamed template called
//! `reviewer`. Unknown fields are ignored and every field has a documented
//! default.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::PromptMode;
use crate::skills::parse_skill_front_matter;
use crate::tools::BuiltinTool;
use crate::workspace::is_safe_relative_path;

/// Rail names an agent template may declare.
///
/// These are the rail seams the Rust harness implements; the `core.*` aliases
/// of the Python manifest are accepted by [`crate::spec::resolve`] where a
/// counterpart exists. Deferred Python rails (`core.lsp`, `core.verification`,
/// `core.agent_mode`, …) are deliberately absent and are rejected with a typed
/// validation error instead of silently mounting nothing.
pub const KNOWN_RAILS: &[&str] = &[
    "security",
    "memory",
    "skills",
    "subagent",
    "evolution",
    "harness_prompt",
    "permission_interrupt",
];

/// File extensions [`DeepAgentSpec::from_path`] and [`crate::spec::SpecRegistry`]
/// understand.
pub const SUPPORTED_SPEC_EXTENSIONS: &[&str] = &["yaml", "yml", "json", "md", "markdown"];

/// Wire format of one agent template file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecFormat {
    /// YAML mapping (`.yaml`, `.yml`).
    Yaml,
    /// JSON object (`.json`).
    Json,
    /// Markdown with a `---`-fenced YAML front matter (`.md`, `.markdown`).
    Markdown,
}

impl SpecFormat {
    /// Detects the format from a path extension, case-insensitively.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming the path and the supported
    /// extensions when the extension is absent or unknown.
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match extension.as_str() {
            "yaml" | "yml" => Ok(Self::Yaml),
            "json" => Ok(Self::Json),
            "md" | "markdown" => Ok(Self::Markdown),
            other => Err(Error::validation(format!(
                "unsupported agent spec extension '{other}' for {}; supported: {}",
                path.display(),
                SUPPORTED_SPEC_EXTENSIONS.join(", ")
            ))),
        }
    }

    /// Returns the canonical extension of this format.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Yaml => "yaml",
            Self::Json => "json",
            Self::Markdown => "md",
        }
    }
}

/// Plain model hints carried by a template.
///
/// The Rust [`DeepAgentConfig`](crate::schema::DeepAgentConfig) keeps the model
/// *data* while the live [`Model`](openjiuwen_foundation::llm::Model) is supplied
/// by the host builder, so a template can only describe the identity and the
/// sampling settings. An API key is intentionally not representable: secrets
/// never travel inside a template or a plugin manifest.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelSpec {
    /// Provider name (`openai`, `anthropic`, `scripted`, …).
    pub provider: Option<String>,
    /// Provider-facing model name.
    pub model: Option<String>,
    /// Base URL of the provider; `None` uses the provider default.
    pub api_base: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Upper bound on generated tokens.
    pub max_tokens: Option<u32>,
}

impl ModelSpec {
    /// Returns `true` when no hint is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// One named prompt fragment appended after the agent's system prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptFragment {
    /// Stable fragment name (for diagnostics and host tooling).
    pub name: String,
    /// Fragment body.
    pub content: String,
    /// Ordering priority; lower values are rendered first.
    pub priority: i32,
}

impl Default for PromptFragment {
    fn default() -> Self {
        Self {
            name: String::new(),
            content: String::new(),
            priority: 100,
        }
    }
}

/// One skill root declared by a template.
///
/// Relative roots are resolved against the template's own directory (or the
/// workspace root for built-in templates) by
/// [`crate::spec::resolve`]. A disabled root stays in the template for
/// documentation but is never mounted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillSpec {
    /// Skill directory or skill-root directory.
    pub root: String,
    /// Whether the root is mounted on the agent.
    pub enabled: bool,
}

impl Default for SkillSpec {
    fn default() -> Self {
        Self {
            root: String::new(),
            enabled: true,
        }
    }
}

/// One sub-agent declared by a template.
///
/// Either `preset` or `description` must be present: a preset starts from the
/// built-in [`SubAgentPreset`](crate::subagents::SubAgentPreset) configuration,
/// while a description defines a custom sub-agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubAgentSpec {
    /// Registry name of the sub-agent (its `AgentCard.name`).
    pub name: String,
    /// Human-readable description shown in the parent's prompt.
    pub description: String,
    /// Built-in preset name (`code`, `explore`, `plan`, `research`,
    /// `verification`, with or without the `_agent` suffix).
    pub preset: Option<String>,
    /// Model-facing tool names mounted on the sub-agent.
    pub tools: Vec<String>,
    /// Model hints for the sub-agent.
    pub model: Option<ModelSpec>,
    /// System prompt overriding the preset's prompt.
    pub system_prompt: Option<String>,
    /// Prompt language overriding the parent's language.
    pub language: Option<String>,
    /// Inner `ReAct` iteration cap overriding the preset's cap.
    pub max_iterations: Option<u32>,
    /// Free-form discovery metadata.
    pub metadata: BTreeMap<String, Value>,
}

/// Workspace layout hints of a template.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkspaceHints {
    /// Workspace root path; `"./"` when absent.
    pub root: Option<String>,
    /// Schema language (`cn` or `en`); inherits the template language when absent.
    pub language: Option<String>,
    /// Additional relative directories appended to the default schema.
    pub directories: Vec<String>,
}

/// Fully JSON/YAML-serializable template of one `DeepAgent`.
///
/// # Defaults
///
/// `name`, `id`, `description`, `tools`, `skills`, `subagents`, `rails` and
/// `metadata` default to empty; `task_loop` and `restrict_to_work_dir` default to
/// `false`/`true`; `prompt_sections` keeps the declaration order.
///
/// # Examples
///
/// ```
/// use openjiuwen_harness::spec::DeepAgentSpec;
///
/// let spec = DeepAgentSpec::parse(
///     openjiuwen_harness::spec::SpecFormat::Yaml,
///     "name: helper\ndescription: A helper\ntools: [read_file]\n",
/// )
/// .expect("valid template");
/// assert_eq!(spec.name, "helper");
/// assert_eq!(spec.tools, ["read_file"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DeepAgentSpec {
    /// Stable identifier; an absent id keeps the fresh id `AgentCard::new` generates.
    pub id: String,
    /// Registry name and `AgentCard.name`; required unless derived from the file stem.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Prompt language (`cn` or `en`); inherits the workspace language when absent.
    pub language: Option<String>,
    /// System prompt. For Markdown templates the body is stored here.
    pub system_prompt: Option<String>,
    /// Additional named prompt fragments.
    pub prompt_sections: Vec<PromptFragment>,
    /// Model hints.
    pub model: Option<ModelSpec>,
    /// Model-facing tool names, in mount order.
    pub tools: Vec<String>,
    /// Skill roots.
    pub skills: Vec<SkillSpec>,
    /// Sub-agent declarations.
    pub subagents: Vec<SubAgentSpec>,
    /// Declared rail names, validated against [`KNOWN_RAILS`].
    pub rails: Vec<String>,
    /// Workspace layout hints.
    pub workspace: WorkspaceHints,
    /// Prompt verbosity; `None` keeps the harness default (`full`).
    pub prompt_mode: Option<PromptMode>,
    /// Inner `ReAct` iteration cap.
    pub max_iterations: Option<u32>,
    /// Run the outer task loop; also accepted as `enable_task_loop`.
    #[serde(alias = "enable_task_loop")]
    pub task_loop: bool,
    /// Restrict file operations to the workspace; `None` keeps the default (`true`).
    pub restrict_to_work_dir: Option<bool>,
    /// Free-form discovery metadata.
    pub metadata: BTreeMap<String, Value>,
    /// Absolute path the template was loaded from; absent for built-in templates.
    #[serde(skip)]
    pub source_path: Option<PathBuf>,
}

impl DeepAgentSpec {
    /// Loads a template from `path`, dispatching on the file extension.
    ///
    /// YAML and JSON map directly onto the struct; Markdown uses the `---`
    /// fenced front matter as the struct and the body as the system prompt. A
    /// template without a `name` takes the file stem.
    ///
    /// # Errors
    ///
    /// - [`Error::sys_op`] when the file cannot be read;
    /// - [`Error::validation`] when the extension is unsupported, the document
    ///   is malformed, or a Markdown file has no front matter.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let format = SpecFormat::from_path(path)?;
        let text = std::fs::read_to_string(path).map_err(|error| {
            Error::sys_op(format!(
                "failed to read agent spec {}: {error}",
                path.display()
            ))
        })?;
        let mut spec = Self::parse_inner(format, &text, Some(path))?;
        if spec.name.is_empty() {
            if let Some(stem) = path.file_stem().and_then(std::ffi::OsStr::to_str) {
                spec.name = stem.to_string();
            }
        }
        spec.source_path = Some(path.to_path_buf());
        Ok(spec)
    }

    /// Parses a template of an explicit format from memory.
    ///
    /// Unlike [`Self::from_path`], no file-stem fallback applies: the caller is
    /// responsible for setting `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the document is malformed or a
    /// Markdown document has no front matter.
    pub fn parse(format: SpecFormat, text: &str) -> Result<Self> {
        Self::parse_inner(format, text, None)
    }

    /// Shared parser; `source` is only used to make error messages actionable.
    fn parse_inner(format: SpecFormat, text: &str, source: Option<&Path>) -> Result<Self> {
        let label = source.map_or_else(|| "<memory>".to_string(), |p| p.display().to_string());
        match format {
            SpecFormat::Yaml => {
                serde_yaml_ng::from_str(text).map_err(|error| parse_error(&label, "YAML", &error))
            }
            SpecFormat::Json => {
                serde_json::from_str(text).map_err(|error| parse_error(&label, "JSON", &error))
            }
            SpecFormat::Markdown => markdown_spec(text, &label),
        }
    }

    /// Validates names, references and model hints of the template.
    ///
    /// Tool names are resolved through
    /// [`BuiltinTool::from_name`](crate::tools::BuiltinTool::from_name), rail
    /// names through [`KNOWN_RAILS`], sub-agent presets through the built-in
    /// preset table, and workspace hints through
    /// [`is_safe_relative_path`]. Every message names the offending item and,
    /// when the template came from a file, that file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an empty name, an unknown tool, rail or
    /// preset, a duplicate sub-agent name, a non-positive `max_tokens`, a
    /// non-finite temperature, or an unsafe workspace hint.
    pub fn validate(&self) -> Result<()> {
        let label = self.label();
        if self.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "agent spec {label} is missing a non-empty 'name'"
            )));
        }
        let mut tools = Vec::new();
        for tool in &self.tools {
            if BuiltinTool::from_name(tool).is_none() {
                return Err(Error::validation(format!(
                    "unknown tool '{tool}' in agent spec '{}' ({label}); known tools: {}",
                    self.name,
                    crate::tools::extended_tool_names().join(", ")
                )));
            }
            if tools.contains(&tool.as_str()) {
                return Err(Error::validation(format!(
                    "duplicate tool '{tool}' in agent spec '{}' ({label})",
                    self.name
                )));
            }
            tools.push(tool.as_str());
        }
        for rail in &self.rails {
            normalize_rail(rail).map_err(|error| {
                Error::validation(format!(
                    "{} in agent spec '{}' ({label})",
                    error.message(),
                    self.name
                ))
            })?;
        }
        for (index, skill) in self.skills.iter().enumerate() {
            if skill.root.trim().is_empty() {
                return Err(Error::validation(format!(
                    "skill #{index} of agent spec '{}' ({label}) has an empty 'root'",
                    self.name
                )));
            }
        }
        let mut subagent_names = Vec::new();
        for subagent in &self.subagents {
            if subagent.name.trim().is_empty() {
                return Err(Error::validation(format!(
                    "a sub-agent of agent spec '{}' ({label}) is missing a non-empty 'name'",
                    self.name
                )));
            }
            if subagent_names.contains(&subagent.name.as_str()) {
                return Err(Error::validation(format!(
                    "duplicate sub-agent '{}' in agent spec '{}' ({label})",
                    subagent.name, self.name
                )));
            }
            subagent_names.push(subagent.name.as_str());
            if subagent.preset.is_none() && subagent.description.trim().is_empty() {
                return Err(Error::validation(format!(
                    "sub-agent '{}' of agent spec '{}' ({label}) must declare a 'preset' or a 'description'",
                    subagent.name, self.name
                )));
            }
            if let Some(preset) = &subagent.preset {
                crate::spec::resolve::parse_preset(preset).map_err(|error| {
                    Error::validation(format!(
                        "sub-agent '{}' of agent spec '{}' ({label}): {}",
                        subagent.name,
                        self.name,
                        error.message()
                    ))
                })?;
            }
            for tool in &subagent.tools {
                if BuiltinTool::from_name(tool).is_none() {
                    return Err(Error::validation(format!(
                        "unknown tool '{tool}' on sub-agent '{}' of agent spec '{}' ({label})",
                        subagent.name, self.name
                    )));
                }
            }
            if let Some(model) = &subagent.model {
                validate_model(model, &format!("sub-agent '{}'", subagent.name), &label)?;
            }
        }
        if let Some(model) = &self.model {
            validate_model(model, "agent", &label)?;
        }
        for directory in &self.workspace.directories {
            if !is_safe_relative_path(directory) {
                return Err(Error::validation(format!(
                    "unsafe workspace directory hint '{directory}' in agent spec '{}' ({label})",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// Returns a human-readable location used in validation messages.
    #[must_use]
    pub fn label(&self) -> String {
        self.source_path.as_ref().map_or_else(
            || format!("agent spec '{}' (built-in resources)", self.name),
            |path| format!("agent spec '{}' ({})", self.name, path.display()),
        )
    }

    /// Returns the template with `path` as its source location.
    #[must_use]
    pub fn with_source_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.source_path = Some(path.into());
        self
    }
}

/// Renders a Markdown template: fenced YAML front matter plus a prompt body.
fn markdown_spec(text: &str, label: &str) -> Result<DeepAgentSpec> {
    let parsed = parse_skill_front_matter(text).map_err(|error| {
        Error::validation(format!(
            "invalid front matter in Markdown {label}: {}",
            error.message()
        ))
    })?;
    let Some((metadata, body)) = parsed else {
        return Err(Error::validation(format!(
            "Markdown {label} must start with a '---' fenced YAML front matter"
        )));
    };
    let object: serde_json::Map<String, Value> = metadata.into_iter().collect();
    let mut spec: DeepAgentSpec = serde_json::from_value(Value::Object(object))
        .map_err(|error| parse_error(label, "Markdown front matter", &error))?;
    if !body.is_empty() {
        spec.system_prompt = Some(match spec.system_prompt.take() {
            Some(existing) => format!("{body}\n\n{existing}"),
            None => body,
        });
    }
    Ok(spec)
}

/// Builds a parse error carrying the file label and format.
fn parse_error(label: &str, format: &str, error: &dyn std::fmt::Display) -> Error {
    Error::validation(format!("failed to parse {format} {label}: {error}"))
}

/// Validates the model hints of one owner.
fn validate_model(model: &ModelSpec, owner: &str, label: &str) -> Result<()> {
    if let Some(temperature) = model.temperature {
        if !temperature.is_finite() {
            return Err(Error::validation(format!(
                "{owner} temperature must be finite in {label}"
            )));
        }
    }
    if let Some(top_p) = model.top_p {
        if !top_p.is_finite() {
            return Err(Error::validation(format!(
                "{owner} top_p must be finite in {label}"
            )));
        }
    }
    if model.max_tokens == Some(0) {
        return Err(Error::validation(format!(
            "{owner} max_tokens must be positive in {label}"
        )));
    }
    Ok(())
}

/// Normalizes a declared rail name to the canonical Rust rail name.
///
/// The `core.*` aliases of the Python manifest are accepted when the Rust
/// harness has a counterpart; every other name is rejected.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the unknown rail and the known set.
pub(crate) fn normalize_rail(name: &str) -> Result<&'static str> {
    let canonical = match name.trim() {
        "security" | "core.security" => "security",
        "memory" | "core.memory" => "memory",
        "skills" | "core.skill_use" | "core.skills" => "skills",
        "subagent" | "core.subagent" => "subagent",
        "evolution" | "core.evolution" => "evolution",
        "harness_prompt" | "prompt" | "core.prompt" => "harness_prompt",
        "permission_interrupt" | "core.confirm_interrupt" | "core.permission_interrupt" => {
            "permission_interrupt"
        }
        other => {
            return Err(Error::validation(format!(
                "unknown rail '{other}'; known rails: {}",
                KNOWN_RAILS.join(", ")
            )))
        }
    };
    Ok(canonical)
}
