//! Agent-facing evolution draft schema and its normalizers.
//!
//! Ports Python `openjiuwen.agent_evolving.experience.draft_schema`: a draft is
//! a typed envelope ([`EvolutionSubject`]) plus a list of
//! [`DraftExperience`] items ([`EvolveDraft`]) or library-maintenance actions
//! ([`SimplifyDraft`]). The vocabularies are the Python literals: subject kinds
//! `skill | swarm-skill` (the legacy alias `team-skill` normalizes to
//! `swarm-skill`), targets `description | body | script`, sections
//! `Instructions | Examples | Troubleshooting | Scripts | Collaboration | Roles |
//! Constraints | Workflow`, and simplify actions `DELETE | MERGE | REFINE | KEEP`.
//!
//! Normalization is strict: every malformed field is an
//! [`Error::validation`] naming the field and the rule it broke. Raw drafts are
//! read from [`serde_json::Value`] rather than a struct so unknown keys from a
//! newer producer are ignored exactly as Python ignores them.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The longest `summary` a draft experience may carry, in characters.
pub const SUMMARY_LIMIT: usize = 160;

/// The subject kinds an agent-facing experience draft may target.
pub const SUPPORTED_SUBJECT_KINDS: [EvolutionSubjectKind; 2] = [
    EvolutionSubjectKind::Skill,
    EvolutionSubjectKind::SwarmSkill,
];

/// Which artefact a draft targets.
///
/// The wire literals are the Python protocol values `skill` and `swarm-skill`;
/// the legacy alias `team-skill` is accepted on input and normalized to
/// [`EvolutionSubjectKind::SwarmSkill`].
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvolutionSubjectKind {
    /// A single skill.
    #[default]
    #[serde(rename = "skill")]
    Skill,
    /// A team/swarm skill; `team-skill` is its legacy alias.
    #[serde(rename = "swarm-skill", alias = "team-skill")]
    SwarmSkill,
}

impl EvolutionSubjectKind {
    /// The kind as its protocol wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Skill => "skill",
            Self::SwarmSkill => "swarm-skill",
        }
    }

    /// Parses a kind, accepting the legacy `team-skill` alias.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the trimmed value is not one of
    /// [`SUPPORTED_SUBJECT_KINDS`].
    pub fn parse(kind: &str) -> Result<Self> {
        normalize_subject_kind(kind)
    }
}

/// Normalizes an evolution subject kind, mapping the legacy `team-skill` alias.
///
/// # Errors
///
/// [`Error::validation`] when the trimmed value is empty or unsupported.
pub fn normalize_subject_kind(kind: &str) -> Result<EvolutionSubjectKind> {
    let normalized = kind.trim();
    match normalized {
        "skill" => Ok(EvolutionSubjectKind::Skill),
        "team-skill" | "swarm-skill" => Ok(EvolutionSubjectKind::SwarmSkill),
        _ => Err(Error::validation(
            "subject.kind must be one of: skill, swarm-skill",
        )),
    }
}

/// The normalized subject envelope of an agent-facing draft.
///
/// `scope` carries optional addressing metadata (for example a workspace id)
/// and is omitted from [`EvolutionSubject::payload`] when absent or empty.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolutionSubject {
    /// Which artefact the draft targets.
    pub kind: EvolutionSubjectKind,
    /// The skill name; never empty for a normalized subject.
    pub name: String,
    /// Optional addressing metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<BTreeMap<String, Value>>,
}

impl EvolutionSubject {
    /// Creates a subject with a validated, trimmed name.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `name` is blank after trimming.
    pub fn new(kind: EvolutionSubjectKind, name: impl Into<String>) -> Result<Self> {
        let name = name.into().trim().to_owned();
        if name.is_empty() {
            return Err(Error::validation("subject.name is required"));
        }
        Ok(Self {
            kind,
            name,
            scope: None,
        })
    }

    /// Creates a [`EvolutionSubjectKind::Skill`] subject.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `name` is blank after trimming.
    pub fn for_skill(name: impl Into<String>) -> Result<Self> {
        Self::new(EvolutionSubjectKind::Skill, name)
    }

    /// Returns the subject with `scope` installed.
    #[must_use]
    pub fn with_scope(mut self, scope: BTreeMap<String, Value>) -> Self {
        self.scope = Some(scope);
        self
    }

    /// The stable agent-facing payload: `kind`, `name` and, when present,
    /// `scope`.
    #[must_use]
    pub fn payload(&self) -> Value {
        let mut payload = Map::new();
        payload.insert(
            "kind".to_owned(),
            Value::String(self.kind.as_str().to_owned()),
        );
        payload.insert("name".to_owned(), Value::String(self.name.clone()));
        if let Some(scope) = &self.scope {
            if !scope.is_empty() {
                let values = scope
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                payload.insert("scope".to_owned(), Value::Object(values));
            }
        }
        Value::Object(payload)
    }
}

/// Normalizes a raw subject envelope into an [`EvolutionSubject`].
///
/// # Errors
///
/// [`Error::validation`] when `raw` is not an object, when `kind` is missing or
/// unsupported, when `name` is blank, or when `scope` is present but not an
/// object.
pub fn normalize_subject(raw: &Value) -> Result<EvolutionSubject> {
    let Some(object) = raw.as_object() else {
        return Err(Error::validation("subject must be an object"));
    };
    let kind = raw_text(object.get("kind"));
    if kind.is_empty() {
        return Err(Error::validation("subject.kind is required"));
    }
    let kind = EvolutionSubjectKind::parse(&kind)?;
    let name = raw_text(object.get("name")).trim().to_owned();
    if name.is_empty() {
        return Err(Error::validation("subject.name is required"));
    }
    let scope = match object.get("scope") {
        None | Some(Value::Null) => None,
        Some(Value::Object(values)) => Some(
            values
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<BTreeMap<_, _>>(),
        ),
        Some(_) => return Err(Error::validation("subject.scope must be an object")),
    };
    Ok(EvolutionSubject { kind, name, scope })
}

/// Where inside a skill document a draft experience belongs.
///
/// The Python protocol restricts this target to three values, unlike the wider
/// [`crate::signal::EvolutionTarget`] routing vocabulary.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionTarget {
    /// The skill description shown before the body.
    Description,
    /// The main instruction block of a skill.
    #[default]
    Body,
    /// A script asset belonging to the skill.
    Script,
}

impl EvolutionTarget {
    /// The target as its protocol wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Description => "description",
            Self::Body => "body",
            Self::Script => "script",
        }
    }

    /// Parses one of the three protocol target literals.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for any other value.
    pub fn parse(target: &str) -> Result<Self> {
        match target.trim() {
            "description" => Ok(Self::Description),
            "body" => Ok(Self::Body),
            "script" => Ok(Self::Script),
            other => Err(Error::validation(format!("invalid target: {other}"))),
        }
    }
}

/// The section heading a draft experience is written under.
///
/// The vocabulary is the Python `VALID_SECTIONS` set and keeps its
/// `PascalCase` wire spelling.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "PascalCase")]
pub enum DraftSection {
    /// Core instructions for the skill.
    Instructions,
    /// Worked examples.
    Examples,
    /// Failure recovery guidance.
    #[default]
    Troubleshooting,
    /// Script assets.
    Scripts,
    /// Collaboration rules for team skills.
    Collaboration,
    /// Role definitions for team skills.
    Roles,
    /// Hard limits and quality gates.
    Constraints,
    /// Step ordering for team skills.
    Workflow,
}

impl DraftSection {
    /// The section as its `PascalCase` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Instructions => "Instructions",
            Self::Examples => "Examples",
            Self::Troubleshooting => "Troubleshooting",
            Self::Scripts => "Scripts",
            Self::Collaboration => "Collaboration",
            Self::Roles => "Roles",
            Self::Constraints => "Constraints",
            Self::Workflow => "Workflow",
        }
    }

    /// Parses one of the eight protocol section literals.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for any other value.
    pub fn parse(section: &str) -> Result<Self> {
        match section.trim() {
            "Instructions" => Ok(Self::Instructions),
            "Examples" => Ok(Self::Examples),
            "Troubleshooting" => Ok(Self::Troubleshooting),
            "Scripts" => Ok(Self::Scripts),
            "Collaboration" => Ok(Self::Collaboration),
            "Roles" => Ok(Self::Roles),
            "Constraints" => Ok(Self::Constraints),
            "Workflow" => Ok(Self::Workflow),
            other => Err(Error::validation(format!("invalid section: {other}"))),
        }
    }

    /// The section a target defaults to when the draft names none.
    #[must_use]
    pub const fn default_for_target(target: EvolutionTarget) -> Self {
        match target {
            EvolutionTarget::Description => Self::Instructions,
            EvolutionTarget::Body | EvolutionTarget::Script => Self::Troubleshooting,
        }
    }
}

/// One normalized experience in an [`EvolveDraft`].
///
/// The optional script fields are only populated for
/// [`EvolutionTarget::Script`] items, and `source_refs` holds the run-scoped
/// evidence references the approval view shows but the persistence view drops.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DraftExperience {
    /// One-line lesson summary, whitespace-collapsed and at most
    /// [`SUMMARY_LIMIT`] characters.
    pub summary: String,
    /// The Markdown lesson body.
    pub content: String,
    /// Where the lesson belongs; defaults to [`EvolutionTarget::Body`].
    pub target: EvolutionTarget,
    /// The heading it is written under; forced to [`DraftSection::Scripts`] for
    /// script targets.
    pub section: DraftSection,
    /// Why the lesson was drafted; may be empty.
    pub reason: String,
    /// The script file name, when the experience ships a script.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script_filename: Option<String>,
    /// The script language identifier, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script_language: Option<String>,
    /// What the script is for, when it ships one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script_purpose: Option<String>,
    /// Run-scoped evidence references; omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_refs: Vec<String>,
}

impl DraftExperience {
    /// Normalizes one raw draft item.
    ///
    /// `target` defaults to `body`; `section` is forced to `Scripts` for script
    /// targets and otherwise defaults to `Instructions` (description) or
    /// `Troubleshooting` (body).
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `raw` is not an object, `content` or
    /// `summary` is missing/blank, the summary exceeds [`SUMMARY_LIMIT`], the
    /// target or section literal is invalid, `script_filename` is not a plain
    /// file name, or `source_refs` is not a list.
    pub fn normalize(raw: &Value) -> Result<Self> {
        let Some(object) = raw.as_object() else {
            return Err(Error::validation("experience must be an object"));
        };
        let content = raw_text(object.get("content")).trim().to_owned();
        if content.is_empty() {
            return Err(Error::validation("experience content is required"));
        }
        let target = match raw_text(object.get("target")).trim() {
            "" => EvolutionTarget::Body,
            other => EvolutionTarget::parse(other)?,
        };
        let section = if target == EvolutionTarget::Script {
            DraftSection::Scripts
        } else {
            match raw_text(object.get("section")).trim() {
                "" => DraftSection::default_for_target(target),
                other => DraftSection::parse(other)?,
            }
        };
        Ok(Self {
            summary: normalize_summary(object.get("summary"))?,
            content,
            target,
            section,
            reason: raw_text(object.get("reason")).trim().to_owned(),
            script_filename: normalize_script_filename(object.get("script_filename"))?,
            script_language: optional_text(object.get("script_language")),
            script_purpose: optional_text(object.get("script_purpose")),
            source_refs: normalize_source_refs(object.get("source_refs"))?,
        })
    }
}

/// The normalized evolve draft: a subject plus its experiences.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolveDraft {
    /// The artefact the draft targets.
    pub subject: EvolutionSubject,
    /// The normalized experiences, in submission order.
    pub experiences: Vec<DraftExperience>,
}

impl EvolveDraft {
    /// Creates a draft from its parts.
    #[must_use]
    pub fn new(subject: EvolutionSubject, experiences: Vec<DraftExperience>) -> Self {
        Self {
            subject,
            experiences,
        }
    }

    /// Whether the draft carries no experience.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }

    /// The approval-facing JSON view, including `source_refs`.
    #[must_use]
    pub fn approval_view(&self) -> Value {
        let experiences = serde_json::to_value(&self.experiences).unwrap_or_default();
        let mut view = Map::new();
        view.insert("experiences".to_owned(), experiences);
        Value::Object(view)
    }

    /// The persistence-facing JSON view, without run-scoped `source_refs`.
    #[must_use]
    pub fn persistence_view(&self) -> Value {
        let experiences: Vec<Value> = self
            .experiences
            .iter()
            .map(|experience| {
                let mut persisted = experience.clone();
                persisted.source_refs.clear();
                serde_json::to_value(&persisted).unwrap_or_default()
            })
            .collect();
        let mut view = Map::new();
        view.insert("experiences".to_owned(), Value::Array(experiences));
        Value::Object(view)
    }
}

/// Normalizes a non-empty list of raw evolve experiences.
///
/// # Errors
///
/// [`Error::validation`] when `experiences` is empty or any item fails
/// [`DraftExperience::normalize`].
pub fn normalize_evolve_draft(
    subject: EvolutionSubject,
    experiences: &[Value],
) -> Result<EvolveDraft> {
    if experiences.is_empty() {
        return Err(Error::validation("experiences must be a non-empty list"));
    }
    let mut normalized = Vec::with_capacity(experiences.len());
    for (index, raw) in experiences.iter().enumerate() {
        let experience = DraftExperience::normalize(raw)
            .map_err(|error| Error::validation(format!("experience at index {index}: {error}")))?;
        normalized.push(experience);
    }
    Ok(EvolveDraft::new(subject, normalized))
}

/// One library-maintenance action in a [`SimplifyDraft`].
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SimplifyActionKind {
    /// Drop the record.
    Delete,
    /// Fold the record into another one.
    Merge,
    /// Rewrite the record's content.
    Refine,
    /// Keep the record unchanged.
    #[default]
    Keep,
}

impl SimplifyActionKind {
    /// The action as its `SCREAMING_SNAKE_CASE` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Delete => "DELETE",
            Self::Merge => "MERGE",
            Self::Refine => "REFINE",
            Self::Keep => "KEEP",
        }
    }

    /// Parses one of the four action literals, case-insensitively.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for any other value.
    pub fn parse(action: &str) -> Result<Self> {
        match action.trim().to_ascii_uppercase().as_str() {
            "DELETE" => Ok(Self::Delete),
            "MERGE" => Ok(Self::Merge),
            "REFINE" => Ok(Self::Refine),
            "KEEP" => Ok(Self::Keep),
            other => Err(Error::validation(format!(
                "invalid simplify action: {other}"
            ))),
        }
    }
}

/// One normalized maintenance action over an existing experience record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SimplifyAction {
    /// What to do with the record.
    pub action: SimplifyActionKind,
    /// The record the action targets.
    pub record_id: String,
    /// Why the action was chosen; may be empty.
    pub reason: String,
    /// Replacement content, required for `REFINE` and `MERGE`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_content: Option<String>,
    /// Records to fold into the target, required for `MERGE`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub merge_remove_ids: Vec<String>,
}

impl SimplifyAction {
    /// Normalizes one raw maintenance action.
    ///
    /// `new_content` is only retained for `REFINE`/`MERGE`, matching Python's
    /// normalization, and `merge_remove_ids` rejects empty ids, the target's own
    /// id and duplicates.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] on a malformed action, a missing `record_id`, a
    /// `REFINE` without content, a `MERGE` without content or merge ids, or a
    /// malformed `merge_remove_ids` entry.
    pub fn normalize(raw: &Value) -> Result<Self> {
        let Some(object) = raw.as_object() else {
            return Err(Error::validation("action must be an object"));
        };
        let action = SimplifyActionKind::parse(&raw_text(object.get("action")))?;
        let record_id = raw_text(object.get("record_id")).trim().to_owned();
        if record_id.is_empty() {
            return Err(Error::validation("record_id is required"));
        }
        let new_content = raw_text(object.get("new_content")).trim().to_owned();
        let merge_remove_ids =
            normalize_merge_remove_ids(object.get("merge_remove_ids"), &record_id)?;
        if action == SimplifyActionKind::Refine && new_content.is_empty() {
            return Err(Error::validation("REFINE requires new_content"));
        }
        if action == SimplifyActionKind::Merge {
            if new_content.is_empty() {
                return Err(Error::validation("MERGE requires new_content"));
            }
            if merge_remove_ids.is_empty() {
                return Err(Error::validation("MERGE requires merge_remove_ids"));
            }
        }
        let keep_content = matches!(
            action,
            SimplifyActionKind::Refine | SimplifyActionKind::Merge
        );
        Ok(Self {
            action,
            record_id,
            reason: raw_text(object.get("reason")).trim().to_owned(),
            new_content: keep_content.then_some(new_content),
            merge_remove_ids,
        })
    }
}

/// The normalized simplify draft: a subject plus its actions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SimplifyDraft {
    /// The artefact the maintenance pass runs for.
    pub subject: EvolutionSubject,
    /// The normalized actions, in submission order.
    pub actions: Vec<SimplifyAction>,
}

impl SimplifyDraft {
    /// Creates a draft from its parts.
    #[must_use]
    pub fn new(subject: EvolutionSubject, actions: Vec<SimplifyAction>) -> Self {
        Self { subject, actions }
    }

    /// Whether the draft carries no action.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// The approval-facing JSON view.
    #[must_use]
    pub fn approval_view(&self) -> Value {
        self.actions_view()
    }

    /// The persistence-facing JSON view; identical to the approval view because
    /// simplify actions carry no run-scoped references.
    #[must_use]
    pub fn persistence_view(&self) -> Value {
        self.actions_view()
    }

    /// The shared `{"actions": [...]}` projection of both views.
    fn actions_view(&self) -> Value {
        let mut view = Map::new();
        view.insert(
            "actions".to_owned(),
            serde_json::to_value(&self.actions).unwrap_or_default(),
        );
        Value::Object(view)
    }
}

/// Normalizes a non-empty list of raw simplify actions.
///
/// # Errors
///
/// [`Error::validation`] when `actions` is empty or any item fails
/// [`SimplifyAction::normalize`].
pub fn normalize_simplify_draft(
    subject: EvolutionSubject,
    actions: &[Value],
) -> Result<SimplifyDraft> {
    if actions.is_empty() {
        return Err(Error::validation("actions must be a non-empty list"));
    }
    let mut normalized = Vec::with_capacity(actions.len());
    for (index, raw) in actions.iter().enumerate() {
        let action = SimplifyAction::normalize(raw)
            .map_err(|error| Error::validation(format!("action at index {index}: {error}")))?;
        normalized.push(action);
    }
    Ok(SimplifyDraft::new(subject, normalized))
}

/// Checks every simplify action against the ids that actually exist.
///
/// # Errors
///
/// [`Error::validation`] naming the first missing `record_id` or
/// `merge_remove_ids` entry.
pub fn validate_simplify_record_refs(
    draft: &SimplifyDraft,
    existing_record_ids: &BTreeSet<String>,
) -> Result<()> {
    for action in &draft.actions {
        if !existing_record_ids.contains(&action.record_id) {
            return Err(Error::validation(format!(
                "record not found: {}",
                action.record_id
            )));
        }
        for remove_id in &action.merge_remove_ids {
            if !existing_record_ids.contains(remove_id) {
                return Err(Error::validation(format!("record not found: {remove_id}")));
            }
        }
    }
    Ok(())
}

/// Collapses runs of whitespace and validates the [`SUMMARY_LIMIT`] bound.
fn normalize_summary(value: Option<&Value>) -> Result<String> {
    let summary = raw_text(value)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if summary.is_empty() {
        return Err(Error::validation("experience summary is required"));
    }
    if summary.chars().count() > SUMMARY_LIMIT {
        return Err(Error::validation(format!(
            "experience summary must be at most {SUMMARY_LIMIT} characters"
        )));
    }
    Ok(summary)
}

/// Validates a script file name: a bare name, never a path.
fn normalize_script_filename(value: Option<&Value>) -> Result<Option<String>> {
    let filename = raw_text(value).trim().to_owned();
    if filename.is_empty() {
        return Ok(None);
    }
    // A POSIX or Windows separator anywhere, a leading separator, a drive/colon
    // prefix, or the `.`/`..` directory names all mean the value is a path.
    let absolute = filename.starts_with('/') || filename.starts_with('\\');
    let has_separator = filename.contains('/') || filename.contains('\\');
    let has_windows_drive = filename.contains(':');
    if filename == "." || filename == ".." || absolute || has_separator || has_windows_drive {
        return Err(Error::validation("script_filename must be a file name"));
    }
    Ok(Some(filename))
}

/// Trims and drops the empty or dash-only placeholders a model emits for "none".
fn normalize_source_refs(value: Option<&Value>) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(items) = value.as_array() else {
        return Err(Error::validation("source_refs must be a list"));
    };
    let mut refs = Vec::with_capacity(items.len());
    for item in items {
        let normalized = raw_text(Some(item)).trim().to_owned();
        if normalized.is_empty() || is_dash_placeholder(&normalized) {
            continue;
        }
        refs.push(normalized);
    }
    Ok(refs)
}

/// Validates the merge-remove ids against the target id, emptiness and
/// duplicates.
fn normalize_merge_remove_ids(value: Option<&Value>, record_id: &str) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(items) = value.as_array() else {
        return Err(Error::validation("merge_remove_ids must be a list"));
    };
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(items.len());
    for item in items {
        let remove_id = raw_text(Some(item)).trim().to_owned();
        if remove_id.is_empty() {
            return Err(Error::validation(
                "merge_remove_ids must not contain empty values",
            ));
        }
        if remove_id == record_id {
            return Err(Error::validation(
                "merge_remove_ids must not contain record_id",
            ));
        }
        if !seen.insert(remove_id.clone()) {
            return Err(Error::validation(format!(
                "duplicate merge_remove_id: {remove_id}"
            )));
        }
        normalized.push(remove_id);
    }
    Ok(normalized)
}

/// Reads a raw JSON value as text, treating absent/null as empty.
///
/// Strings, numbers and booleans have their natural spelling; arrays and objects
/// have no textual form here and read as empty, so a wrongly typed field fails
/// the required-field check instead of being stringified.
fn raw_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Bool(boolean)) => boolean.to_string(),
        Some(Value::Null | Value::Array(_) | Value::Object(_)) | None => String::new(),
    }
}

/// Reads an optional textual field, dropping empty values.
fn optional_text(value: Option<&Value>) -> Option<String> {
    let text = raw_text(value).trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Whether `text` is only dash characters, the placeholder a model writes for
/// "no references".
fn is_dash_placeholder(text: &str) -> bool {
    text.chars()
        .all(|character| matches!(character, '-' | '–' | '—'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn subject() -> EvolutionSubject {
        EvolutionSubject::for_skill("pdf-report").expect("valid skill subject")
    }

    fn normalize_one(raw: serde_json::Value) -> Result<DraftExperience> {
        DraftExperience::normalize(&raw)
    }

    #[test]
    fn subject_kind_normalizes_the_team_alias() {
        assert_eq!(
            normalize_subject_kind(" skill ").expect("valid kind"),
            EvolutionSubjectKind::Skill
        );
        assert_eq!(
            normalize_subject_kind("team-skill").expect("alias"),
            EvolutionSubjectKind::SwarmSkill
        );
        assert_eq!(
            normalize_subject_kind("swarm-skill").expect("wire kind"),
            EvolutionSubjectKind::SwarmSkill
        );
        assert!(normalize_subject_kind("agent").is_err());
        assert!(normalize_subject_kind(" ").is_err());
    }

    #[test]
    fn subject_normalization_validates_every_field() {
        let normalized = normalize_subject(&json!({
            "kind": "team-skill",
            "name": "  squad  ",
            "scope": {"team": "alpha"},
            "unknown": true
        }))
        .expect("valid subject");
        assert_eq!(normalized.kind, EvolutionSubjectKind::SwarmSkill);
        assert_eq!(normalized.name, "squad");
        assert_eq!(
            normalized.payload(),
            json!({"kind": "swarm-skill", "name": "squad", "scope": {"team": "alpha"}})
        );

        assert!(normalize_subject(&json!([])).is_err());
        assert!(normalize_subject(&json!({"name": "x"})).is_err());
        assert!(normalize_subject(&json!({"kind": "skill"})).is_err());
        assert!(normalize_subject(&json!({"kind": "agent", "name": "x"})).is_err());
        assert!(normalize_subject(&json!({"kind": "skill", "name": "x", "scope": 3})).is_err());
        assert_eq!(
            EvolutionSubject::for_skill("  ").unwrap_err().to_string(),
            "validation: subject.name is required"
        );
    }

    #[test]
    fn target_and_section_vocabularies_round_trip() {
        assert_eq!(
            EvolutionTarget::parse("body").expect("body"),
            EvolutionTarget::Body
        );
        assert_eq!(
            EvolutionTarget::parse("script").expect("script").as_str(),
            "script"
        );
        assert!(EvolutionTarget::parse("tool").is_err());
        assert_eq!(
            DraftSection::parse("Collaboration")
                .expect("valid section")
                .as_str(),
            "Collaboration"
        );
        assert!(DraftSection::parse("troubleshooting").is_err());
        assert_eq!(
            DraftSection::default_for_target(EvolutionTarget::Description),
            DraftSection::Instructions
        );
        assert_eq!(
            DraftSection::default_for_target(EvolutionTarget::Body),
            DraftSection::Troubleshooting
        );
    }

    #[test]
    fn experience_normalizes_defaults_and_script_fields() {
        let experience = normalize_one(json!({
            "summary": "  raise   the read timeout ",
            "content": " ## Tuning \n",
            "script_filename": "fix.py",
            "script_language": " python ",
            "source_refs": [" run-1 ", "", "—", "run-2"]
        }))
        .expect("valid experience");
        assert_eq!(experience.summary, "raise the read timeout");
        assert_eq!(experience.content, "## Tuning");
        assert_eq!(experience.target, EvolutionTarget::Body);
        assert_eq!(experience.section, DraftSection::Troubleshooting);
        assert_eq!(experience.script_filename.as_deref(), Some("fix.py"));
        assert_eq!(experience.script_language.as_deref(), Some("python"));
        assert_eq!(experience.script_purpose, None);
        assert_eq!(experience.source_refs, vec!["run-1", "run-2"]);

        let description = normalize_one(json!({
            "summary": "s",
            "content": "c",
            "target": "description"
        }))
        .expect("description default section");
        assert_eq!(description.section, DraftSection::Instructions);

        let script = normalize_one(json!({
            "summary": "s",
            "content": "c",
            "target": "script",
            "section": "Instructions"
        }))
        .expect("script forces Scripts");
        assert_eq!(script.section, DraftSection::Scripts);
    }

    #[test]
    fn experience_rejects_malformed_items() {
        assert!(normalize_one(json!({"summary": "s"})).is_err());
        assert!(normalize_one(json!({"content": "c"})).is_err());
        assert!(normalize_one(json!({"summary": " ", "content": "c"})).is_err());
        assert!(normalize_one(json!({"summary": "s", "content": "c", "target": "tool"})).is_err());
        assert!(normalize_one(json!({"summary": "s", "content": "c", "section": "Nope"})).is_err());
        assert!(
            normalize_one(json!({"summary": "s", "content": "c", "source_refs": "x"})).is_err()
        );
        let long = "x".repeat(SUMMARY_LIMIT + 1);
        let error = normalize_one(json!({"summary": long, "content": "c"})).unwrap_err();
        assert!(error.to_string().contains("at most 160 characters"));
        let exact = "x".repeat(SUMMARY_LIMIT);
        assert!(normalize_one(json!({"summary": exact, "content": "c"})).is_ok());
    }

    #[test]
    fn evolve_draft_views_differ_only_in_source_refs() {
        let draft = normalize_evolve_draft(
            subject(),
            &[json!({
                "summary": "s",
                "content": "c",
                "source_refs": ["signal-1"]
            })],
        )
        .expect("valid draft");
        assert_eq!(draft.experiences.len(), 1);
        let approval = draft.approval_view();
        assert_eq!(
            approval["experiences"][0]["source_refs"],
            json!(["signal-1"])
        );
        let persisted = draft.persistence_view();
        assert_eq!(persisted["experiences"][0].get("source_refs"), None);
        assert!(normalize_evolve_draft(subject(), &[]).is_err());
        assert!(normalize_evolve_draft(subject(), &[json!("prose")]).is_err());
    }

    #[test]
    fn simplify_actions_enforce_merge_and_refine_rules() {
        let keep = SimplifyAction::normalize(&json!({"action": "keep", "record_id": "ev-1"}))
            .expect("keep");
        assert_eq!(keep.action, SimplifyActionKind::Keep);
        assert_eq!(keep.new_content, None);

        let refine = SimplifyAction::normalize(&json!({
            "action": "REFINE",
            "record_id": "ev-1",
            "new_content": " new "
        }))
        .expect("refine");
        assert_eq!(refine.new_content.as_deref(), Some("new"));

        let merge = SimplifyAction::normalize(&json!({
            "action": "MERGE",
            "record_id": "ev-1",
            "new_content": "merged",
            "merge_remove_ids": ["ev-2", "ev-3"]
        }))
        .expect("merge");
        assert_eq!(merge.merge_remove_ids, vec!["ev-2", "ev-3"]);

        assert!(
            SimplifyAction::normalize(&json!({"action": "ARCHIVE", "record_id": "ev-1"})).is_err()
        );
        assert!(SimplifyAction::normalize(&json!({"action": "KEEP"})).is_err());
        assert!(
            SimplifyAction::normalize(&json!({"action": "REFINE", "record_id": "ev-1"})).is_err()
        );
        assert!(SimplifyAction::normalize(
            &json!({"action": "MERGE", "record_id": "ev-1", "new_content": "x"})
        )
        .is_err());
        assert!(SimplifyAction::normalize(&json!({
            "action": "MERGE", "record_id": "ev-1", "new_content": "x", "merge_remove_ids": ["ev-1"]
        }))
        .is_err());
        assert!(SimplifyAction::normalize(&json!({
            "action": "MERGE", "record_id": "ev-1", "new_content": "x", "merge_remove_ids": ["ev-2", "ev-2"]
        }))
        .is_err());
        assert!(SimplifyAction::normalize(&json!({
            "action": "MERGE", "record_id": "ev-1", "new_content": "x", "merge_remove_ids": [""]
        }))
        .is_err());
        assert!(SimplifyAction::normalize(
            &json!({"action": "KEEP", "record_id": "ev-1", "merge_remove_ids": "x"})
        )
        .is_err());
    }
}
