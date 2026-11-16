//! The `AgentGroup` package format: cards and the JSON-only loader.
//!
//! This file is intentionally long: it holds the whole package format in one place — cards,
//! loader, manifest validation, persona assembly and path safety — because every rule is enforced
//! during a single load, and it carries the fixture-driven test suite that pins those rules.
//!
//! Python parity source: `openjiuwen/agent_teams/experts/agent_group.py`.
//!
//! An `AgentGroup` package bundles a `leader` and its member templates for one team:
//!
//! ```text
//! <group>/
//!   manifest.json              package_type = "agent_group", name, agents, instruction, skills, ...
//!   skills/<skill>/SKILL.md    shared skills (declared and/or auto-discovered)
//!   agents/<member>/
//!     manifest.json            package_type = "agent_template", persona.dir, skills, runtime, ...
//!     AGENT.md                 leader only, optional
//!     <persona.dir>/*.md        persona sections, concatenated in path order
//! ```
//!
//! The loader is deliberately narrow compared with Python: it reads JSON manifests (never YAML),
//! materializes display cards plus the persona text, and validates shared-skill existence. It does
//! not build an `AgentTemplateSpec`, does not interpret `tools`/`mcps`/`rails`/`model`/`subagents`
//! and does not mount skills; those are the member runtime's inputs and are read from the package
//! by the consumer.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::catalog::ExpertRef;
use crate::constants::DEFAULT_LANGUAGE;

/// Roster name every `AgentGroup` must contain; the leader is never spawnable as a teammate.
pub const LEADER_MEMBER: &str = "leader";

const GROUP_PACKAGE_TYPE: &str = "agent_group";
const TEMPLATE_PACKAGE_TYPE: &str = "agent_template";
const NATIVE_RUNTIME: &str = "native";
const EXTERNAL_RUNTIME: &str = "external";

/// A display text that is either language-independent or keyed by language.
///
/// Deserializes from `"text"` or from `{"cn": "...", "en": "..."}`. [`LocalizedText::resolve`]
/// resolves a requested language, aliasing `zh` to `cn`, accepting a `zh` map key as a synonym for
/// `cn` (published Python packages use `zh`), and falling back to `cn`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LocalizedText {
    /// One text used for every language.
    Plain(String),
    /// Per-language texts keyed by language code (`cn`, `en`, …).
    Localized(BTreeMap<String, String>),
}

impl Default for LocalizedText {
    fn default() -> Self {
        Self::Plain(String::new())
    }
}

impl From<&str> for LocalizedText {
    fn from(text: &str) -> Self {
        Self::Plain(text.to_owned())
    }
}

impl From<String> for LocalizedText {
    fn from(text: String) -> Self {
        Self::Plain(text)
    }
}

impl LocalizedText {
    /// Returns the text for `language`, falling back to `cn`.
    ///
    /// A `zh` request (and `zh-CN` / `zh_CN`) is treated as `cn`; a `zh` map key is accepted as a
    /// synonym for `cn`. A [`LocalizedText::Plain`] value ignores the language entirely.
    #[must_use]
    pub fn resolve(&self, language: &str) -> &str {
        match self {
            Self::Plain(text) => text,
            Self::Localized(texts) => {
                let requested = language.trim().to_ascii_lowercase();
                let key = if requested == "zh"
                    || requested.starts_with("zh-")
                    || requested.starts_with("zh_")
                {
                    "cn"
                } else {
                    requested.as_str()
                };
                lookup_language(texts, key)
                    .or_else(|| lookup_language(texts, DEFAULT_LANGUAGE))
                    .unwrap_or("")
            }
        }
    }

    /// [`LocalizedText::resolve`] for an optional language; `None` resolves `cn`.
    #[must_use]
    pub fn resolve_optional(&self, language: Option<&str>) -> &str {
        self.resolve(language.unwrap_or(DEFAULT_LANGUAGE))
    }

    /// Returns `true` when every text in this value is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Plain(text) => text.trim().is_empty(),
            Self::Localized(texts) => texts.values().all(|text| text.trim().is_empty()),
        }
    }
}

/// Looks up `key`, accepting `zh` as a synonym for `cn` and vice versa.
fn lookup_language<'a>(texts: &'a BTreeMap<String, String>, key: &str) -> Option<&'a str> {
    texts
        .get(key)
        .or_else(|| match key {
            "cn" => texts.get("zh"),
            "zh" => texts.get("cn"),
            _ => None,
        })
        .map(String::as_str)
}

/// The display card of one member template (`agents/<member>/manifest.json`).
///
/// Python parity: `AgentTemplateCard` (display subset). `runtime` is `native` unless the member
/// manifest declares an external runtime provider, in which case the provider name and its
/// `config` are carried so a consumer can launch the external CLI member.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentTemplateCard {
    /// User-facing name from the member manifest, defaulting to the directory (member) name.
    pub name: String,
    /// Human-readable description.
    pub description: LocalizedText,
    /// Optional localized display name.
    pub display_name: Option<LocalizedText>,
    /// Optional localized long display description.
    pub display_description: Option<LocalizedText>,
    /// Optional category, resolved to one string when the manifest localizes it.
    pub category: Option<String>,
    /// Tag identities, resolved to one string each in the template's default language.
    pub tags: Vec<String>,
    /// Optional template version.
    pub version: Option<String>,
    /// Optional avatar reference.
    pub avatar: Option<String>,
    /// `native` or `external`.
    pub runtime: String,
    /// External runtime provider name (`codex`, `claudecode`, …) for external members.
    pub runtime_provider: Option<String>,
    /// External runtime configuration (`runtime.config` of the member manifest).
    pub runtime_config: Option<Value>,
    /// Skill directory names available to the member, including the group's shared skills.
    pub skills: Vec<String>,
    /// Connector identities declared by the member manifest.
    pub connectors: Vec<String>,
}

impl Default for AgentTemplateCard {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: LocalizedText::default(),
            display_name: None,
            display_description: None,
            category: None,
            tags: Vec::new(),
            version: None,
            avatar: None,
            runtime: NATIVE_RUNTIME.to_owned(),
            runtime_provider: None,
            runtime_config: None,
            skills: Vec::new(),
            connectors: Vec::new(),
        }
    }
}

impl AgentTemplateCard {
    /// Builds a native-runtime card with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Returns `true` when the card declares an external CLI runtime.
    #[must_use]
    pub fn is_external(&self) -> bool {
        self.runtime == EXTERNAL_RUNTIME
    }

    /// Display label in `language`: localized display name, then the manifest name, then `fallback`.
    #[must_use]
    pub fn display_label<'a>(&'a self, language: &str, fallback: &'a str) -> &'a str {
        self.display_name
            .as_ref()
            .map(|text| text.resolve(language))
            .filter(|text| !text.trim().is_empty())
            .or_else(|| (!self.name.trim().is_empty()).then_some(self.name.as_str()))
            .unwrap_or(fallback)
    }
}

/// Display information of one `AgentGroup` package (manifests only, never persona or skill content).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentGroupCard {
    /// Package name; the loader requires it to equal the package directory name.
    pub name: String,
    /// Human-readable description.
    pub description: LocalizedText,
    /// Optional localized display name.
    pub display_name: Option<LocalizedText>,
    /// Optional category, resolved to one string when the manifest localizes it.
    pub category: Option<String>,
    /// Capability/tag identities: `capabilities` when declared, otherwise `tags`, deduplicated in
    /// order and resolved to one string each in the default language.
    pub tags: Vec<String>,
    /// Optional package version.
    pub version: Option<String>,
    /// Team-level instruction applied to every member by the assembly step, when declared.
    pub instruction: Option<String>,
}

impl AgentGroupCard {
    /// Display label in `language`, falling back to [`AgentGroupCard::name`].
    #[must_use]
    pub fn display_label(&self, language: &str) -> &str {
        self.display_name
            .as_ref()
            .map(|text| text.resolve(language))
            .filter(|text| !text.trim().is_empty())
            .unwrap_or(self.name.as_str())
    }
}

/// One roster entry of an `AgentGroup` card: the member's identity plus its template card.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentGroupMemberCard {
    /// Member name — the `agents/<name>/` directory, which is the team `member_name`.
    pub name: String,
    /// Owning `AgentGroup` name.
    pub group: String,
    /// The member manifest's card; its own `name` stays the manifest value.
    pub card: AgentTemplateCard,
}

impl AgentGroupMemberCard {
    /// Global identity of this member: `(group, name)`.
    #[must_use]
    pub fn expert_ref(&self) -> ExpertRef {
        ExpertRef {
            name: self.name.clone(),
            group: Some(self.group.clone()),
        }
    }

    /// Returns `true` when this entry is the group's leader.
    #[must_use]
    pub fn is_leader(&self) -> bool {
        self.name == LEADER_MEMBER
    }

    /// Display label in `language`, falling back to the member name.
    #[must_use]
    pub fn display_label(&self, language: &str) -> &str {
        self.card.display_label(language, &self.name)
    }
}

/// One validated `AgentGroup` load, reusable within a single call chain.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoadedAgentGroupPackage {
    /// Canonical package root directory.
    pub package_dir: PathBuf,
    /// Parsed group card.
    pub manifest: AgentGroupCard,
    /// Member templates keyed (and therefore ordered) by member name.
    pub templates: BTreeMap<String, AgentTemplateCard>,
}

impl LoadedAgentGroupPackage {
    /// The team-level instruction applied to members, or `""`.
    #[must_use]
    pub fn instruction(&self) -> &str {
        self.manifest.instruction.as_deref().unwrap_or("")
    }

    /// Capability/tag identities of the package.
    #[must_use]
    pub fn capabilities(&self) -> &[String] {
        &self.manifest.tags
    }

    /// One member template by name.
    #[must_use]
    pub fn template(&self, name: &str) -> Option<&AgentTemplateCard> {
        self.templates.get(name)
    }

    /// The roster as member cards, ordered by member name.
    #[must_use]
    pub fn member_cards(&self) -> Vec<AgentGroupMemberCard> {
        self.templates
            .iter()
            .map(|(name, card)| AgentGroupMemberCard {
                name: name.clone(),
                group: self.manifest.name.clone(),
                card: card.clone(),
            })
            .collect()
    }

    /// Reads and concatenates one member's persona from the package directory.
    ///
    /// Persona sections are the `.md` files under the member manifest's `persona.dir`, ordered by
    /// resolved path and joined with a blank line; front matter is preserved verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::NotFound`] when the member is not in this
    /// package, and a validation error when the package no longer parses (the directory was
    /// removed or edited after loading).
    pub fn load_member_persona(&self, member: &str) -> Result<String> {
        if !self.templates.contains_key(member) {
            return Err(Error::not_found(format!(
                "AgentGroup member not found: {member}"
            )));
        }
        let agent_dir = resolve_member_dir(&self.package_dir, member)?;
        let manifest_path = package_file(
            &agent_dir.join("manifest.json"),
            &agent_dir,
            &format!("AgentTemplate manifest for {member:?}"),
        )?;
        let manifest = read_json_object(
            &manifest_path,
            &format!("AgentTemplate manifest for {member:?}"),
        )?;
        persona_text(&agent_dir, &manifest, member)
    }
}

/// Loads and validates one `AgentGroup` package.
///
/// The loader canonicalizes `dir`, validates `<dir>/manifest.json`, every roster member's
/// `<dir>/agents/<member>/manifest.json`, each member's persona directory, the leader-only
/// `AGENT.md` rule, and the existence of every shared skill's `SKILL.md`. It never mutates or reads
/// anything outside the package directory.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] with the offending path or member
/// name when the directory is missing, a manifest is absent or malformed, the manifest name does
/// not match the directory, the roster is invalid, a member violates the `AGENT.md` rule, a persona
/// directory has no Markdown files, or a shared skill is missing.
pub fn load_agent_group(dir: impl AsRef<Path>) -> Result<LoadedAgentGroupPackage> {
    let package_dir = resolve_existing_dir(dir.as_ref(), "agent_group package directory")?;
    let dir_name = package_dir
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| {
            Error::validation(format!(
                "agent_group package has no directory name: {}",
                package_dir.display()
            ))
        })?
        .to_owned();

    let manifest_path = package_file(
        &package_dir.join("manifest.json"),
        &package_dir,
        "agent_group manifest",
    )?;
    let payload = read_json_object(&manifest_path, "agent_group manifest")?;

    if payload.get("package_type").and_then(Value::as_str) != Some(GROUP_PACKAGE_TYPE) {
        return Err(Error::validation(
            "agent_group manifest package_type must be 'agent_group'",
        ));
    }
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::validation("agent_group manifest name must be a string"))?;
    if name != dir_name {
        return Err(Error::validation(format!(
            "agent_group manifest name must match directory: {name:?} != {dir_name:?}"
        )));
    }

    let instruction = manifest_instruction(&payload)?;
    let shared = shared_skills(&package_dir, &payload)?;
    let roster = agent_names(&payload)?;

    let mut templates = BTreeMap::new();
    for agent_name in roster {
        let template = load_member_template(&package_dir, &agent_name, &shared)?;
        templates.insert(agent_name, template);
    }

    let manifest = AgentGroupCard {
        name: name.to_owned(),
        description: localized_field(&payload, "description")?.unwrap_or_default(),
        display_name: localized_field(&payload, "display_name")?,
        category: localized_string(&payload, "category")?,
        tags: group_tag_identities(&payload)?,
        version: localized_string(&payload, "version")?,
        instruction,
    };

    Ok(LoadedAgentGroupPackage {
        package_dir,
        manifest,
        templates,
    })
}

/// A validated shared skill of the group package.
#[derive(Debug, Clone)]
struct SharedSkill {
    name: String,
    path: PathBuf,
}

fn load_member_template(
    package_dir: &Path,
    agent_name: &str,
    shared: &[SharedSkill],
) -> Result<AgentTemplateCard> {
    let agent_dir = resolve_member_dir(package_dir, agent_name)?;
    let label = format!("AgentTemplate manifest for {agent_name:?}");
    let manifest_path = package_file(&agent_dir.join("manifest.json"), &agent_dir, &label)?;
    let manifest = read_json_object(&manifest_path, &label)?;

    if manifest.get("package_type").and_then(Value::as_str) != Some(TEMPLATE_PACKAGE_TYPE) {
        return Err(Error::validation(format!(
            "AgentTemplate {agent_name:?} must declare package_type='agent_template'"
        )));
    }

    // Validate the persona directory and its Markdown files even though the card itself only
    // carries metadata; the assembly step reads the persona through `load_member_persona`.
    let persona_dir = persona_dir(&agent_dir, &manifest, agent_name)?;
    persona_markdown(&persona_dir, &agent_dir, agent_name)?;

    check_agent_markdown(&agent_dir, agent_name)?;

    let (runtime, runtime_provider, runtime_config) = parse_runtime(&manifest, agent_name)?;
    let mut skills = skill_names(&manifest, "skills")?;
    for shared_skill in shared {
        dedup_push(&mut skills, shared_skill.name.clone());
    }

    let name = manifest
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map_or_else(|| agent_name.to_owned(), str::to_owned);

    Ok(AgentTemplateCard {
        name,
        description: localized_field(&manifest, "description")?.unwrap_or_default(),
        display_name: localized_field(&manifest, "display_name")?,
        display_description: localized_field(&manifest, "display_description")?,
        category: localized_string(&manifest, "category")?,
        tags: localized_tag_list(&manifest, "tags")?,
        version: localized_string(&manifest, "version")?,
        avatar: localized_string(&manifest, "avatar")?,
        runtime,
        runtime_provider,
        runtime_config,
        skills,
        connectors: string_list(&manifest, "connectors")?,
    })
}

/// Resolves `<package>/agents/<member>` and rejects a directory outside the package.
fn resolve_member_dir(package_dir: &Path, agent_name: &str) -> Result<PathBuf> {
    let agents_root = resolve_existing_dir(&package_dir.join("agents"), "agents directory")?;
    if !agents_root.starts_with(package_dir) {
        return Err(Error::validation(format!(
            "agents directory escapes AgentGroup package: {}",
            agents_root.display()
        )));
    }
    safe_component(agent_name, "agent name")?;
    let agent_dir = std::fs::canonicalize(agents_root.join(agent_name))
        .map_err(|_| Error::validation(format!("agent directory not found: {agent_name}")))?;
    if !agent_dir.starts_with(&agents_root) {
        return Err(Error::validation(format!(
            "agent directory escapes AgentGroup package: {agent_name}"
        )));
    }
    if !agent_dir.is_dir() {
        return Err(Error::validation(format!(
            "agent directory is not a directory: {}",
            agent_dir.display()
        )));
    }
    Ok(agent_dir)
}

/// Returns the roster in manifest order, validating every name and the mandatory leader.
fn agent_names(payload: &Map<String, Value>) -> Result<Vec<String>> {
    let raw = payload
        .get("agents")
        .ok_or_else(|| Error::validation("agent_group manifest agents must be a non-empty list"))?;
    let items = raw
        .as_array()
        .ok_or_else(|| Error::validation("agent_group manifest agents must be a non-empty list"))?;
    if items.is_empty() {
        return Err(Error::validation(
            "agent_group manifest agents must be a non-empty list",
        ));
    }
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let raw_name = item.as_str().ok_or_else(|| {
            Error::validation("invalid agent name in agent_group manifest: expected a string")
        })?;
        let name = safe_component(raw_name.trim(), "agent name")?;
        if !is_roster_name(&name) {
            return Err(Error::validation(format!(
                "invalid agent name in agent_group manifest: {name:?}; must start with a lowercase \
                 ASCII letter (a-z), followed by lowercase letters, digits (0-9), underscore (_) or \
                 hyphen (-)"
            )));
        }
        if names.contains(&name) {
            return Err(Error::validation(format!(
                "duplicate agent name in agent_group manifest: {name}"
            )));
        }
        names.push(name);
    }
    if !names.iter().any(|name| name == LEADER_MEMBER) {
        return Err(Error::validation(
            "agent_group manifest agents must contain 'leader'",
        ));
    }
    Ok(names)
}

/// Roster-name pattern `^[a-z][a-z0-9_-]*$` (hand-written, no regex, no global state).
fn is_roster_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
}

fn manifest_instruction(payload: &Map<String, Value>) -> Result<Option<String>> {
    match payload.get("instruction") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            Ok((!trimmed.is_empty()).then(|| trimmed.to_owned()))
        }
        Some(_) => Err(Error::validation(
            "agent_group manifest instruction must be a string",
        )),
    }
}

/// Group capability identities: `capabilities` when present, otherwise `tags`.
fn group_tag_identities(payload: &Map<String, Value>) -> Result<Vec<String>> {
    let key = if payload.contains_key("capabilities") {
        "capabilities"
    } else {
        "tags"
    };
    let Some(raw) = payload.get(key) else {
        return Ok(Vec::new());
    };
    let items = raw
        .as_array()
        .ok_or_else(|| Error::validation(format!("agent_group manifest {key} must be a list")))?;
    let mut tags = Vec::new();
    for item in items {
        let text = match item {
            Value::String(text) => text.clone(),
            Value::Object(map) => map
                .get("id")
                .or_else(|| map.get("zh"))
                .or_else(|| map.get("en"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            _ => {
                return Err(Error::validation(format!(
                    "agent_group manifest {key} entries must be strings or localized objects"
                )))
            }
        };
        dedup_push(&mut tags, text.trim().to_owned());
    }
    Ok(tags)
}

fn localized_field(payload: &Map<String, Value>, key: &str) -> Result<Option<LocalizedText>> {
    match payload.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(LocalizedText::Plain(text.clone()))),
        Some(Value::Object(map)) => {
            let mut texts = BTreeMap::new();
            for (language, value) in map {
                let text = value.as_str().ok_or_else(|| {
                    Error::validation(format!(
                        "manifest field {key:?} must map language codes to strings"
                    ))
                })?;
                texts.insert(language.clone(), text.to_owned());
            }
            Ok(Some(LocalizedText::Localized(texts)))
        }
        Some(_) => Err(Error::validation(format!(
            "manifest field {key:?} must be a string or a localized object"
        ))),
    }
}

fn localized_string(payload: &Map<String, Value>, key: &str) -> Result<Option<String>> {
    Ok(localized_field(payload, key)?.and_then(|text| {
        let resolved = text.resolve(DEFAULT_LANGUAGE).trim().to_owned();
        (!resolved.is_empty()).then_some(resolved)
    }))
}

fn localized_tag_list(payload: &Map<String, Value>, key: &str) -> Result<Vec<String>> {
    let Some(raw) = payload.get(key) else {
        return Ok(Vec::new());
    };
    let items = raw
        .as_array()
        .ok_or_else(|| Error::validation(format!("manifest field {key:?} must be a list")))?;
    let mut tags = Vec::new();
    for item in items {
        let text = match item {
            Value::String(text) => text.clone(),
            Value::Object(map) => {
                let mut texts = BTreeMap::new();
                for (language, value) in map {
                    let text = value.as_str().ok_or_else(|| {
                        Error::validation(format!(
                            "manifest field {key:?} entries must map language codes to strings"
                        ))
                    })?;
                    texts.insert(language.clone(), text.to_owned());
                }
                let localized = LocalizedText::Localized(texts);
                let resolved = localized.resolve(DEFAULT_LANGUAGE).trim().to_owned();
                if resolved.is_empty() {
                    map.get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim()
                        .to_owned()
                } else {
                    resolved
                }
            }
            _ => {
                return Err(Error::validation(format!(
                    "manifest field {key:?} entries must be strings or localized objects"
                )))
            }
        };
        dedup_push(&mut tags, text);
    }
    Ok(tags)
}

fn string_list(payload: &Map<String, Value>, key: &str) -> Result<Vec<String>> {
    let Some(raw) = payload.get(key) else {
        return Ok(Vec::new());
    };
    let items = raw
        .as_array()
        .ok_or_else(|| Error::validation(format!("manifest field {key:?} must be a list")))?;
    let mut values = Vec::new();
    for item in items {
        let text = match item {
            Value::String(text) => text.clone(),
            Value::Object(map) => map
                .get("name")
                .or_else(|| map.get("id"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            _ => {
                return Err(Error::validation(format!(
                    "manifest field {key:?} entries must be strings or objects with a name"
                )))
            }
        };
        dedup_push(&mut values, text.trim().to_owned());
    }
    Ok(values)
}

fn skill_names(payload: &Map<String, Value>, key: &str) -> Result<Vec<String>> {
    let Some(raw) = payload.get(key) else {
        return Ok(Vec::new());
    };
    let items = raw
        .as_array()
        .ok_or_else(|| Error::validation(format!("manifest field {key:?} must be a list")))?;
    let mut names = Vec::new();
    for item in items {
        let value = match item {
            Value::String(text) => text.as_str(),
            Value::Object(map) => map
                .get("dir")
                .or_else(|| map.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Error::validation(format!(
                        "manifest field {key:?} entries must be names or dir mappings"
                    ))
                })?,
            _ => {
                return Err(Error::validation(format!(
                    "manifest field {key:?} entries must be names or dir mappings"
                )))
            }
        };
        dedup_push(&mut names, skill_basename(value));
    }
    Ok(names)
}

/// Returns the last path component of a skill directory reference.
pub(super) fn skill_basename(value: &str) -> String {
    let trimmed = value.trim().trim_end_matches(['/', '\\']);
    trimmed
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(trimmed)
        .to_owned()
}

fn dedup_push(values: &mut Vec<String>, value: String) {
    if !value.is_empty() && !values.contains(&value) {
        values.push(value);
    }
}

fn parse_runtime(
    manifest: &Map<String, Value>,
    agent_name: &str,
) -> Result<(String, Option<String>, Option<Value>)> {
    match manifest.get("runtime") {
        None | Some(Value::Null) => Ok((NATIVE_RUNTIME.to_owned(), None, None)),
        Some(Value::String(kind)) => match kind.as_str() {
            NATIVE_RUNTIME => Ok((NATIVE_RUNTIME.to_owned(), None, None)),
            EXTERNAL_RUNTIME => {
                if agent_name == LEADER_MEMBER {
                    Err(Error::validation(
                        "AgentGroup leader does not support external runtime",
                    ))
                } else {
                    Ok((EXTERNAL_RUNTIME.to_owned(), None, None))
                }
            }
            other => Err(Error::validation(format!(
                "AgentTemplate {agent_name:?} runtime must be 'native' or 'external', got {other:?}"
            ))),
        },
        Some(Value::Object(map)) => {
            let provider = map
                .get("provider_name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty());
            let Some(provider) = provider else {
                return Ok((NATIVE_RUNTIME.to_owned(), None, None));
            };
            if agent_name == LEADER_MEMBER {
                return Err(Error::validation(
                    "AgentGroup leader does not support external runtime",
                ));
            }
            let config = map
                .get("config")
                .cloned()
                .or_else(|| Some(Value::Object(map.clone())));
            Ok((
                EXTERNAL_RUNTIME.to_owned(),
                Some(provider.to_owned()),
                config,
            ))
        }
        Some(_) => Err(Error::validation(format!(
            "AgentTemplate {agent_name:?} runtime must be an object or 'native'/'external'"
        ))),
    }
}

fn check_agent_markdown(agent_dir: &Path, agent_name: &str) -> Result<()> {
    let agent_md = agent_dir.join("AGENT.md");
    let exists = std::fs::symlink_metadata(&agent_md).is_ok();
    if agent_name == LEADER_MEMBER {
        if exists {
            package_file(&agent_md, agent_dir, "AgentGroup leader AGENT.md")?;
        }
        Ok(())
    } else if exists {
        Err(Error::validation(format!(
            "AgentGroup member {agent_name:?} must not contain AGENT.md; put its responsibilities \
             in persona instead"
        )))
    } else {
        Ok(())
    }
}

fn persona_dir(
    agent_dir: &Path,
    manifest: &Map<String, Value>,
    agent_name: &str,
) -> Result<PathBuf> {
    let raw = manifest
        .get("persona")
        .and_then(Value::as_object)
        .and_then(|persona| persona.get("dir"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::validation(format!(
                "AgentTemplate {agent_name:?} must declare persona.dir"
            ))
        })?;
    if Path::new(raw).is_absolute() || is_windows_absolute(raw) {
        return Err(Error::validation(format!(
            "AgentTemplate {agent_name:?} persona.dir must be package-relative"
        )));
    }
    let resolved = std::fs::canonicalize(agent_dir.join(raw)).map_err(|_| {
        Error::validation(format!(
            "AgentTemplate {agent_name:?} persona.dir does not exist: {raw}"
        ))
    })?;
    if !resolved.starts_with(agent_dir) || !resolved.is_dir() {
        return Err(Error::validation(format!(
            "AgentTemplate {agent_name:?} persona.dir escapes its package"
        )));
    }
    Ok(resolved)
}

fn persona_text(
    agent_dir: &Path,
    manifest: &Map<String, Value>,
    agent_name: &str,
) -> Result<String> {
    let dir = persona_dir(agent_dir, manifest, agent_name)?;
    let files = persona_markdown(&dir, agent_dir, agent_name)?;
    let mut sections = Vec::with_capacity(files.len());
    for file in files {
        let text = std::fs::read_to_string(&file).map_err(|error| {
            Error::validation(format!(
                "AgentTemplate {agent_name:?} persona Markdown is unreadable: {} ({error})",
                file.display()
            ))
        })?;
        sections.push(text);
    }
    Ok(sections.join("\n\n"))
}

fn persona_markdown(dir: &Path, agent_dir: &Path, agent_name: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = std::fs::read_dir(&current).map_err(|error| {
            Error::validation(format!(
                "AgentTemplate {agent_name:?} persona.dir is unreadable: {} ({error})",
                current.display()
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                Error::validation(format!(
                    "AgentTemplate {agent_name:?} persona.dir is unreadable: {} ({error})",
                    current.display()
                ))
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|error| {
                Error::validation(format!(
                    "AgentTemplate {agent_name:?} persona entry is unreadable: {} ({error})",
                    path.display()
                ))
            })?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if !path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
            {
                continue;
            }
            let resolved = std::fs::canonicalize(&path).map_err(|error| {
                Error::validation(format!(
                    "AgentTemplate {agent_name:?} persona Markdown is unreadable: {} ({error})",
                    path.display()
                ))
            })?;
            if !resolved.starts_with(agent_dir) || !resolved.is_file() {
                return Err(Error::validation(format!(
                    "AgentTemplate {agent_name:?} persona Markdown must be a file inside its \
                     package: {}",
                    path.display()
                )));
            }
            files.push(resolved);
        }
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(Error::validation(format!(
            "AgentTemplate {agent_name:?} persona.dir has no Markdown files"
        )));
    }
    Ok(files)
}

fn shared_skills(package_dir: &Path, payload: &Map<String, Value>) -> Result<Vec<SharedSkill>> {
    let raw = match payload.get("skills") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(items)) => items.as_slice(),
        Some(_) => {
            return Err(Error::validation(
                "agent_group manifest skills must be a list",
            ))
        }
    };
    let skills_root = package_dir.join("skills");
    let mut skills: Vec<SharedSkill> = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();

    for item in raw {
        let skill = declared_shared_skill(package_dir, &skills_root, item)?;
        if seen.contains(&skill.path) {
            return Err(Error::validation(format!(
                "duplicate shared skill in agent_group manifest: {}",
                skill.name
            )));
        }
        seen.push(skill.path.clone());
        skills.push(skill);
    }

    if std::fs::symlink_metadata(&skills_root).is_ok() {
        discover_shared_skills(package_dir, &skills_root, &mut seen, &mut skills)?;
    }

    Ok(skills)
}

fn declared_shared_skill(
    package_dir: &Path,
    skills_root: &Path,
    item: &Value,
) -> Result<SharedSkill> {
    let (label, path) = match item {
        Value::String(name) => {
            let name = safe_component(name.trim(), "skill name")?;
            (name.clone(), skills_root.join(&name))
        }
        Value::Object(map) => {
            let raw_dir = map.get("dir").and_then(Value::as_str).ok_or_else(|| {
                Error::validation(
                    "agent_group manifest skills entries must be names or dir mappings",
                )
            })?;
            if Path::new(raw_dir).is_absolute() || is_windows_absolute(raw_dir) {
                return Err(Error::validation(
                    "agent_group skill dir must be package-relative",
                ));
            }
            (skill_basename(raw_dir), package_dir.join(raw_dir))
        }
        _ => {
            return Err(Error::validation(
                "agent_group manifest skills entries must be names or dir mappings",
            ))
        }
    };
    let path = validate_shared_skill(package_dir, &path, &label)?;
    Ok(SharedSkill { name: label, path })
}

fn discover_shared_skills(
    package_dir: &Path,
    skills_root: &Path,
    seen: &mut Vec<PathBuf>,
    skills: &mut Vec<SharedSkill>,
) -> Result<()> {
    let root = resolve_existing_dir(skills_root, "skills directory")?;
    if !root.starts_with(package_dir) {
        return Err(Error::validation(format!(
            "skills directory escapes AgentGroup package: {}",
            root.display()
        )));
    }
    let mut children: Vec<_> = std::fs::read_dir(&root)
        .map_err(|error| {
            Error::validation(format!(
                "skills directory is unreadable: {} ({error})",
                root.display()
            ))
        })?
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|error| {
            Error::validation(format!(
                "skills directory is unreadable: {} ({error})",
                root.display()
            ))
        })?;
    children.sort_by_key(std::fs::DirEntry::file_name);
    for child in children {
        let path = child.path();
        let directory = child.file_type().is_ok_and(|file_type| file_type.is_dir());
        if !directory {
            continue;
        }
        let label = child.file_name().to_string_lossy().into_owned();
        if !std::fs::metadata(path.join("SKILL.md")).is_ok_and(|metadata| metadata.is_file()) {
            tracing::debug!(
                package = %package_dir.display(),
                skill = %label,
                "AgentGroup skill candidate has no SKILL.md; not mounting it"
            );
            continue;
        }
        let validated = validate_shared_skill(package_dir, &path, &label)?;
        if seen.contains(&validated) {
            continue;
        }
        seen.push(validated.clone());
        let name = validated
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
            .to_owned();
        skills.push(SharedSkill {
            name,
            path: validated,
        });
    }
    Ok(())
}

fn validate_shared_skill(package_dir: &Path, path: &Path, label: &str) -> Result<PathBuf> {
    let dir = std::fs::canonicalize(path)
        .map_err(|_| Error::validation(format!("shared skill not found: {label:?}")))?;
    if !dir.starts_with(package_dir) || !dir.is_dir() {
        return Err(Error::validation(format!(
            "shared skill escapes AgentGroup package: {label:?}"
        )));
    }
    let skill_md = dir.join("SKILL.md");
    if !std::fs::metadata(&skill_md).is_ok_and(|metadata| metadata.is_file()) {
        return Err(Error::validation(format!(
            "shared skill {label:?} is missing SKILL.md"
        )));
    }
    Ok(dir)
}

fn resolve_existing_dir(path: &Path, label: &str) -> Result<PathBuf> {
    let resolved = std::fs::canonicalize(path).map_err(|error| {
        Error::validation(format!("{label} not found: {} ({error})", path.display()))
    })?;
    if !resolved.is_dir() {
        return Err(Error::validation(format!(
            "{label} is not a directory: {}",
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn package_file(path: &Path, root: &Path, label: &str) -> Result<PathBuf> {
    let resolved = std::fs::canonicalize(path)
        .map_err(|_| Error::validation(format!("{label} missing: {}", path.display())))?;
    if !resolved.starts_with(root) || !resolved.is_file() {
        return Err(Error::validation(format!(
            "{label} must be a file inside {}: {}",
            root.display(),
            path.display()
        )));
    }
    Ok(resolved)
}

fn read_json_object(path: &Path, label: &str) -> Result<Map<String, Value>> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        Error::validation(format!("{label} missing: {} ({error})", path.display()))
    })?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|_| Error::validation(format!("{label} is not valid JSON: {}", path.display())))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(Error::validation(format!(
            "{label} must contain a JSON object: {}",
            path.display()
        ))),
    }
}

fn safe_component(value: &str, label: &str) -> Result<String> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || Path::new(value).is_absolute()
        || is_windows_absolute(value)
    {
        return Err(Error::validation(format!("invalid {label}: {value:?}")));
    }
    Ok(value.to_owned())
}

fn is_windows_absolute(value: &str) -> bool {
    let bytes = value.as_bytes();
    (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || value.starts_with("\\\\")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::experts::test_support::{
        minimal_group_dir as minimal_group, update_manifest, write_text,
    };
    use serde_json::json;

    #[test]
    fn loads_a_group_with_leader_and_members() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1", "member2"]);

        let package = load_agent_group(&group).unwrap();

        assert_eq!(package.package_dir, group.canonicalize().unwrap());
        assert_eq!(package.manifest.name, "group");
        assert_eq!(package.manifest.display_label("en"), "Alpha");
        assert_eq!(package.manifest.description.resolve("zh"), "测试团");
        assert_eq!(package.manifest.version.as_deref(), Some("2.0.0"));
        assert_eq!(package.instruction(), "work together");
        assert_eq!(package.capabilities(), Vec::<String>::new());
        assert_eq!(
            package.templates.keys().cloned().collect::<Vec<_>>(),
            vec!["leader", "member1", "member2"]
        );

        let member = package.template("member2").unwrap();
        assert_eq!(member.name, "member2 display");
        assert_eq!(member.description.resolve("en"), "member2 description");
        assert_eq!(member.runtime, "native");
        assert!(!member.is_external());

        let leader = package.load_member_persona("leader").unwrap();
        assert!(leader.contains("# leader"), "leader persona: {leader}");
    }

    #[test]
    fn localized_text_accepts_strings_objects_and_zh_alias() {
        let plain: LocalizedText = serde_json::from_value(json!("hello")).unwrap();
        assert_eq!(plain.resolve("en"), "hello");

        let localized: LocalizedText =
            serde_json::from_value(json!({"cn": "你好", "en": "Hello"})).unwrap();
        assert_eq!(localized.resolve("en"), "Hello");
        assert_eq!(localized.resolve("zh"), "你好");
        assert_eq!(localized.resolve("zh-CN"), "你好");
        assert_eq!(localized.resolve("fr"), "你好");
        assert_eq!(localized.resolve_optional(None), "你好");

        // `zh` is accepted as a map key synonym for `cn`.
        let legacy: LocalizedText = serde_json::from_value(json!({"zh": "旧"})).unwrap();
        assert_eq!(legacy.resolve("cn"), "旧");

        assert!(LocalizedText::default().is_empty());
        assert_eq!(
            serde_json::to_value(&localized).unwrap(),
            json!({"cn": "你好", "en": "Hello"})
        );
    }

    #[test]
    fn persona_files_are_concatenated_in_path_order() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        let persona = group.join("agents/member1/persona");
        write_text(&persona.join("10-second.md"), "SECOND");
        write_text(&persona.join("00-first.md"), "FIRST");
        write_text(&persona.join("nested/05-nested.md"), "NESTED");
        write_text(&persona.join("notes.txt"), "ignored");

        let text = load_agent_group(&group)
            .unwrap()
            .load_member_persona("member1")
            .unwrap();

        assert_eq!(text, "FIRST\n\nSECOND\n\n# member1\n\nNESTED");
        assert!(!text.contains("ignored"));
    }

    #[test]
    fn rejects_a_manifest_name_that_does_not_match_the_directory() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("name".to_owned(), json!("other"));
        });

        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(error.contains("must match directory"), "error: {error}");
    }

    #[test]
    fn rejects_missing_agents_and_missing_leader() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);

        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("agents".to_owned(), json!([]));
        });
        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(
            error.contains("agents must be a non-empty list"),
            "error: {error}"
        );

        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("agents".to_owned(), json!(["member1"]));
        });
        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(error.contains("must contain 'leader'"), "error: {error}");
    }

    #[test]
    fn rejects_non_leader_agent_markdown() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        write_text(&group.join("agents/member1/AGENT.md"), "# unexpected\n");

        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(
            error.contains("must not contain AGENT.md"),
            "error: {error}"
        );
    }

    #[test]
    fn rejects_invalid_roster_names_and_paths() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        for name in ["Member1", "_member", "1member", "../member1"] {
            update_manifest(&group.join("manifest.json"), |manifest| {
                manifest.insert("agents".to_owned(), json!(["leader", name]));
            });
            assert!(
                load_agent_group(&group).is_err(),
                "{name} should be rejected"
            );
        }
    }

    #[test]
    fn accepts_kebab_and_snake_case_member_names() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["backend-dev-1", "narrative_designer"]);
        let package = load_agent_group(&group).unwrap();
        assert_eq!(
            package.templates.keys().cloned().collect::<Vec<_>>(),
            vec!["backend-dev-1", "leader", "narrative_designer"]
        );
        assert!(is_roster_name("narrative_designer"));
        assert!(!is_roster_name("Narrative"));
        assert!(!is_roster_name("_narrative"));
    }

    #[test]
    fn merges_declared_and_discovered_shared_skills() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("skills".to_owned(), json!(["declared_skill"]));
        });
        for skill in ["declared_skill", "another_skill"] {
            write_text(
                &group.join("skills").join(skill).join("SKILL.md"),
                "# skill\n",
            );
        }

        let package = load_agent_group(&group).unwrap();
        for template in package.templates.values() {
            assert_eq!(template.skills, vec!["declared_skill", "another_skill"]);
        }

        // A declared skill without SKILL.md is a validation error.
        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("skills".to_owned(), json!(["absent_skill"]));
        });
        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(error.contains("shared skill not found"), "error: {error}");
    }

    #[test]
    fn external_runtime_is_recorded_and_rejected_for_the_leader() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        update_manifest(&group.join("agents/member1/manifest.json"), |manifest| {
            manifest.insert(
                "runtime".to_owned(),
                json!({"provider_name": "codex", "provider_version": "0.1.0", "config": {"mode": "all"}}),
            );
        });
        let package = load_agent_group(&group).unwrap();
        let member = package.template("member1").unwrap();
        assert!(member.is_external());
        assert_eq!(member.runtime_provider.as_deref(), Some("codex"));
        assert_eq!(member.runtime_config, Some(json!({"mode": "all"})));

        update_manifest(&group.join("agents/leader/manifest.json"), |manifest| {
            manifest.insert(
                "runtime".to_owned(),
                json!({"provider_name": "codex", "provider_version": "0.1.0", "config": {}}),
            );
        });
        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(error.contains("leader does not support"), "error: {error}");
    }

    #[test]
    fn rejects_a_package_with_the_wrong_type_or_a_missing_persona() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);

        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("package_type".to_owned(), json!("agent_template"));
        });
        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(
            error.contains("package_type must be 'agent_group'"),
            "error: {error}"
        );

        update_manifest(&group.join("manifest.json"), |manifest| {
            manifest.insert("package_type".to_owned(), json!("agent_group"));
        });
        update_manifest(&group.join("agents/member1/manifest.json"), |manifest| {
            manifest.remove("persona");
        });
        let error = load_agent_group(&group).unwrap_err().to_string();
        assert!(error.contains("must declare persona.dir"), "error: {error}");
    }

    #[test]
    fn member_cards_use_the_directory_as_identity() {
        let temp = tempfile::tempdir().unwrap();
        let group = minimal_group(temp.path(), &["member1"]);
        let package = load_agent_group(&group).unwrap();

        let cards = package.member_cards();
        let leader = cards.iter().find(|card| card.name == "leader").unwrap();
        assert!(leader.is_leader());
        assert_eq!(leader.expert_ref().label(), "group/leader");
        let member = cards.iter().find(|card| card.name == "member1").unwrap();
        assert_eq!(member.card.name, "member1 display");
        assert_eq!(member.display_label("en"), "member1 display");
        assert_eq!(member.group, "group");
    }
}
