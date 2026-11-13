//! Normalized memory records shared by every `context_evolver` pipeline.
//!
//! Ports the schema layer of Python's
//! `openjiuwen/extensions/context_evolver/schema/memory.py` — `TaskMemory`,
//! `ReasoningBankMemory`, and `PersonalMemory` — plus the
//! `get_score`/`to_vector_node` helpers that hang off those classes. Python
//! keeps one class per algorithm family; this port normalizes them into a
//! single [`MemoryRecord`] so summarize pipelines, parsers, retrieval, and the
//! persistence seam all speak one shape:
//!
//! * [`MemoryDraft`] is the payload a summarize parser (or an `add_memory`
//!   caller) produces before an id exists.
//! * [`MemoryRecord`] is the persisted record: a draft plus `id`, `user_id`,
//!   and the helpful/harmful feedback counters.
//!
//! Fields that only one Python class owns (`when_to_use`/`section` for
//! `TaskMemory`, `title`/`description`/`source_type` for
//! `ReasoningBankMemory`, `label` for the ACE playbook) are all optional on the
//! normalized record; pipelines that do not use a field leave it empty.
//!
//! # Deviations from Python
//!
//! * Python derives ids and content hashes with `hashlib.md5`; this port uses
//!   `sha2::Sha256` and keeps only the first 16 hex characters, so ids are
//!   deterministic and collision-resistant without carrying an MD5 dependency.
//! * [`MemoryRecord::identity`] and [`generate_memory_id`] share one identity
//!   builder (`title|content|when_to_use`, trimmed and lowercased), where
//!   Python hashes `content` alone (`TaskMemory`) or `title|content`
//!   (`ReasoningBankMemory`). Including `when_to_use` keeps two memories that
//!   share their content but not their applicability distinct.
//! * The vector-node metadata key `type` is the normalized `"memory"` instead
//!   of `"task_memory"`/`"reasoning_bank_memory"`, because the persisted record
//!   is family-agnostic; the family is recoverable from the id prefix.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/`
//! (schema/memory tests); see `#[cfg(test)] mod tests` below.
use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::memory_store::VectorNode;
use super::{AddMemoryRequest, Algo, AlgoFamily};

/// The normalized payload produced by summarize-pipeline parsers and by
/// `add_memory` callers, before an id is assigned.
///
/// Deserialization is tolerant: a missing field takes its default, a JSON
/// `null` for [`MemoryDraft::attributes`] or [`MemoryDraft::section`] behaves
/// like a missing field, and unknown keys are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryDraft {
    /// The memory text to store.
    pub content: String,
    /// Conditions under which the memory is useful (`TaskMemory`).
    pub when_to_use: Option<String>,
    /// Concise strategy identifier (`ReasoningBankMemory`).
    pub title: Option<String>,
    /// One-sentence summary (`ReasoningBankMemory`).
    pub description: Option<String>,
    /// Memory section/category; Python's default is `"general"`.
    #[serde(
        default = "super::default_section",
        deserialize_with = "deserialize_section"
    )]
    pub section: String,
    /// Optional label used for grouping (ACE playbook).
    pub label: Option<String>,
    /// Arbitrary extra attributes; JSON `null` and a missing field both mean an
    /// empty map.
    #[serde(default, deserialize_with = "super::deserialize_attributes")]
    pub attributes: BTreeMap<String, Value>,
    /// `ReasoningBank` provenance: `success`, `failure`, or `comparative`.
    pub source_type: Option<String>,
}

// The derive list in the module contract renders `section` as an empty string;
// this manual implementation keeps the Rust default identical to the serde
// default (`"general"`), so `MemoryDraft::default()` and `{}` agree.
impl Default for MemoryDraft {
    fn default() -> Self {
        Self {
            content: String::new(),
            when_to_use: None,
            title: None,
            description: None,
            section: super::default_section(),
            label: None,
            attributes: BTreeMap::new(),
            source_type: None,
        }
    }
}

/// Accepts a missing section, an explicit `null`, or a section string.
fn deserialize_section<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let section = Option::<String>::deserialize(deserializer)?;
    Ok(section
        .filter(|section| !section.trim().is_empty())
        .unwrap_or_else(super::default_section))
}

impl MemoryDraft {
    /// Builds a draft holding only `content`; every other field is defaulted.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Projects an [`AddMemoryRequest`] onto the normalized draft shape.
    ///
    /// [`AddMemoryRequest::query`] has no draft counterpart and is dropped;
    /// [`MemoryDraft::source_type`] stays empty because the request cannot
    /// express `ReasoningBank` provenance.
    #[must_use]
    pub fn from_request(request: &AddMemoryRequest) -> Self {
        Self {
            content: request.content.clone(),
            when_to_use: request.when_to_use.clone(),
            title: request.title.clone(),
            description: request.description.clone(),
            section: request.section.clone(),
            label: request.label.clone(),
            attributes: request.attributes.clone(),
            source_type: None,
        }
    }

    /// Reports whether the draft carries no content.
    ///
    /// A whitespace-only [`MemoryDraft::content`] counts as empty, mirroring
    /// [`MemoryRecord::validate`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }
}

/// The persisted, id-bearing memory record shared by every pipeline.
///
/// Serialization is tolerant exactly like [`MemoryDraft`]: a missing field
/// takes its default (`section` is `"general"`, the counters are `0`), a JSON
/// `null` for [`MemoryRecord::attributes`] or [`MemoryRecord::section`]
/// behaves like a missing field, and unknown keys are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryRecord {
    /// Stable identifier; empty until [`MemoryRecord::with_id`] or
    /// [`generate_memory_id`] assigns one.
    pub id: String,
    /// Identifier of the memory owner.
    pub user_id: String,
    /// The memory text.
    pub content: String,
    /// Conditions under which the memory is useful (`TaskMemory`).
    pub when_to_use: Option<String>,
    /// Concise strategy identifier (`ReasoningBankMemory`).
    pub title: Option<String>,
    /// One-sentence summary (`ReasoningBankMemory`).
    pub description: Option<String>,
    /// Memory section/category; Python's default is `"general"`.
    #[serde(
        default = "super::default_section",
        deserialize_with = "deserialize_section"
    )]
    pub section: String,
    /// Optional label used for grouping (ACE playbook).
    pub label: Option<String>,
    /// Arbitrary extra attributes; JSON `null` and a missing field both mean an
    /// empty map.
    #[serde(default, deserialize_with = "super::deserialize_attributes")]
    pub attributes: BTreeMap<String, Value>,
    /// Times the memory was marked helpful (`get_score` numerator).
    pub helpful: u32,
    /// Times the memory was marked harmful (`get_score` denominator).
    pub harmful: u32,
    /// `ReasoningBank` provenance: `success`, `failure`, or `comparative`.
    pub source_type: Option<String>,
}

impl Default for MemoryRecord {
    fn default() -> Self {
        Self {
            id: String::new(),
            user_id: String::new(),
            content: String::new(),
            when_to_use: None,
            title: None,
            description: None,
            section: super::default_section(),
            label: None,
            attributes: BTreeMap::new(),
            helpful: 0,
            harmful: 0,
            source_type: None,
        }
    }
}

impl MemoryRecord {
    /// Builds a record from a draft, leaving [`MemoryRecord::id`] empty.
    ///
    /// Callers either assign an id with [`MemoryRecord::with_id`] or derive a
    /// deterministic one with [`generate_memory_id`].
    #[must_use]
    pub fn new(user_id: impl Into<String>, draft: MemoryDraft) -> Self {
        Self {
            id: String::new(),
            user_id: user_id.into(),
            content: draft.content,
            when_to_use: draft.when_to_use,
            title: draft.title,
            description: draft.description,
            section: draft.section,
            label: draft.label,
            attributes: draft.attributes,
            helpful: 0,
            harmful: 0,
            source_type: draft.source_type,
        }
    }

    /// Assigns the record identifier, returning the updated record.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Builds a record from an [`AddMemoryRequest`], leaving the id empty.
    #[must_use]
    pub fn from_request(user_id: &str, request: &AddMemoryRequest) -> Self {
        Self::new(user_id, MemoryDraft::from_request(request))
    }

    /// Projects the record back onto the [`AddMemoryRequest`] card.
    ///
    /// [`AddMemoryRequest::query`] is always empty: the card's query is a
    /// retrieval-time input, not part of the stored memory.
    #[must_use]
    pub fn to_request(&self) -> AddMemoryRequest {
        AddMemoryRequest {
            content: self.content.clone(),
            query: None,
            when_to_use: self.when_to_use.clone(),
            title: self.title.clone(),
            description: self.description.clone(),
            section: self.section.clone(),
            label: self.label.clone(),
            attributes: self.attributes.clone(),
        }
    }

    /// Projects the persisted record onto its [`MemoryDraft`] shape, dropping
    /// `id`, `user_id`, and the feedback counters.
    #[must_use]
    pub fn to_draft(&self) -> MemoryDraft {
        MemoryDraft {
            content: self.content.clone(),
            when_to_use: self.when_to_use.clone(),
            title: self.title.clone(),
            description: self.description.clone(),
            section: self.section.clone(),
            label: self.label.clone(),
            attributes: self.attributes.clone(),
            source_type: self.source_type.clone(),
        }
    }

    /// The dedup identity: `title|content|when_to_use`, each trimmed and
    /// lowercased, with absent or blank fields skipped.
    ///
    /// Two records with the same identity are considered the same memory.
    #[must_use]
    pub fn identity(&self) -> String {
        memory_identity(
            self.title.as_deref(),
            &self.content,
            self.when_to_use.as_deref(),
        )
    }

    /// The first 16 hex characters of `sha2::Sha256` over
    /// [`MemoryRecord::identity`].
    ///
    /// Stable across runs and platforms, so it can key dedup maps or file
    /// names.
    #[must_use]
    pub fn dedup_hash(&self) -> String {
        sha256_hex_prefix(&self.identity(), 16)
    }

    /// The prompt-readable block for this record, Python-shaped.
    ///
    /// Lines are emitted only for non-blank fields, in the order `Title`,
    /// `Description`, `When to use`, `Section`, `Content` — the order Python's
    /// `ReasoningBankMemory.to_vector_node` composes its embedding text in.
    /// `label`, `attributes`, and the feedback counters are not rendered.
    #[must_use]
    pub fn render_block(&self) -> String {
        let mut lines = Vec::new();
        if let Some(title) = non_blank(self.title.as_deref()) {
            lines.push(format!("Title: {title}"));
        }
        if let Some(description) = non_blank(self.description.as_deref()) {
            lines.push(format!("Description: {description}"));
        }
        if let Some(when_to_use) = non_blank(self.when_to_use.as_deref()) {
            lines.push(format!("When to use: {when_to_use}"));
        }
        if let Some(section) = non_blank(Some(self.section.as_str())) {
            lines.push(format!("Section: {section}"));
        }
        if let Some(content) = non_blank(Some(self.content.as_str())) {
            lines.push(format!("Content: {content}"));
        }
        lines.join("\n")
    }

    /// Python's `get_score`: `(helpful + 1) / (harmful + 1)`.
    ///
    /// The `+1` smoothing keeps a brand-new memory at `1.0` and bounds the
    /// result below by `1/(harmful + 1)` > 0, so it never divides by zero.
    // The counters are counts, not measurements: `f32` precision is ample.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn feedback_score(&self) -> f32 {
        (self.helpful as f32 + 1.0) / (self.harmful as f32 + 1.0)
    }

    /// Validates the identity-bearing fields.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when [`MemoryRecord::user_id`],
    /// [`MemoryRecord::content`], or [`MemoryRecord::id`] is blank; the
    /// message names the record.
    pub fn validate(&self) -> Result<()> {
        let label = self.describe();
        if self.user_id.trim().is_empty() {
            return Err(Error::Validation(format!(
                "context_evolver memory record '{label}' has a blank user_id"
            )));
        }
        if self.content.trim().is_empty() {
            return Err(Error::Validation(format!(
                "context_evolver memory record '{label}' has blank content"
            )));
        }
        if self.id.trim().is_empty() {
            return Err(Error::Validation(format!(
                "context_evolver memory record '{label}' has a blank id"
            )));
        }
        Ok(())
    }

    /// Returns the record with the feedback counters increased.
    ///
    /// The increments saturate at [`u32::MAX`] instead of wrapping.
    #[must_use]
    pub fn with_feedback(mut self, helpful_delta: u32, harmful_delta: u32) -> Self {
        self.helpful = self.helpful.saturating_add(helpful_delta);
        self.harmful = self.harmful.saturating_add(harmful_delta);
        self
    }

    /// Builds the vector-store node for this record.
    ///
    /// Metadata always carries `type` (`"memory"`), `user_id`, `content`, and
    /// `section`, plus `when_to_use`/`title`/`description`/`label`/
    /// `source_type` when present and the `helpful`/`harmful` counters. Extra
    /// [`MemoryRecord::attributes`] are flattened into the metadata; an
    /// attribute that collides with a reserved key never overwrites it. The
    /// node id is [`MemoryRecord::id`].
    #[must_use]
    pub fn to_vector_node(&self, embedding: Vec<f32>) -> VectorNode {
        let mut metadata = BTreeMap::new();
        metadata.insert("type".to_string(), Value::from("memory"));
        metadata.insert("user_id".to_string(), Value::from(self.user_id.clone()));
        metadata.insert("content".to_string(), Value::from(self.content.clone()));
        metadata.insert("section".to_string(), Value::from(self.section.clone()));
        for (key, value) in [
            ("when_to_use", &self.when_to_use),
            ("title", &self.title),
            ("description", &self.description),
            ("label", &self.label),
            ("source_type", &self.source_type),
        ] {
            if let Some(value) = value {
                metadata.insert(key.to_string(), Value::from(value.clone()));
            }
        }
        metadata.insert("helpful".to_string(), Value::from(self.helpful));
        metadata.insert("harmful".to_string(), Value::from(self.harmful));
        for (key, value) in &self.attributes {
            metadata.entry(key.clone()).or_insert_with(|| value.clone());
        }
        VectorNode {
            id: self.id.clone(),
            embedding,
            metadata,
        }
    }

    /// The name used in validation messages: the id when present, otherwise
    /// the user, otherwise `<unassigned>`.
    fn describe(&self) -> String {
        if !self.id.trim().is_empty() {
            self.id.clone()
        } else if !self.user_id.trim().is_empty() {
            self.user_id.clone()
        } else {
            "<unassigned>".to_string()
        }
    }
}

/// Generates the deterministic id for a draft: `{prefix}_{user}_{hash16}`.
///
/// `prefix` comes from [`Algo::family`] (`ace`, `reasoning_bank`, `reme`,
/// `cognition`), `user` is the sanitized user id, and `hash16` is the first 16
/// hex characters of SHA-256 over the draft identity
/// (`title|content|when_to_use`, trimmed and lowercased, absent or blank fields
/// skipped). The same input always yields the same id, so re-running a
/// summarize pipeline upserts instead of duplicating.
///
/// # Examples
///
/// ```
/// use openjiuwen_extensions::context_evolver::{generate_memory_id, Algo, MemoryDraft};
///
/// let draft = MemoryDraft::new("Prefer `rg` over `grep`");
/// let id = generate_memory_id(Algo::ReasoningBank, "user 1", &draft);
/// assert!(id.starts_with("reasoning_bank_user_1_"));
/// assert_eq!(id, generate_memory_id(Algo::ReasoningBank, "user 1", &draft));
/// ```
#[must_use]
pub fn generate_memory_id(algo: Algo, user_id: &str, draft: &MemoryDraft) -> String {
    let identity = memory_identity(
        draft.title.as_deref(),
        &draft.content,
        draft.when_to_use.as_deref(),
    );
    format!(
        "{}_{}_{}",
        family_prefix(algo.family()),
        sanitize_user(user_id),
        sha256_hex_prefix(&identity, 16)
    )
}

/// The id prefix of every algorithm family.
const fn family_prefix(family: AlgoFamily) -> &'static str {
    match family {
        AlgoFamily::Ace => "ace",
        AlgoFamily::ReasoningBank => "reasoning_bank",
        AlgoFamily::Reme => "reme",
        AlgoFamily::Cognition => "cognition",
    }
}

/// Builds the shared dedup identity from the three identity-bearing fields.
///
/// Each field is trimmed and lowercased; absent or blank fields are skipped, so
/// `Some("")` and `None` behave alike. One builder backs both
/// [`MemoryRecord::identity`] and [`generate_memory_id`], which keeps the
/// persisted dedup hash and the generated id in lockstep.
#[must_use]
pub(crate) fn memory_identity(
    title: Option<&str>,
    content: &str,
    when_to_use: Option<&str>,
) -> String {
    let mut parts = Vec::new();
    for value in [title, Some(content), when_to_use].into_iter().flatten() {
        let normalized = value.trim().to_lowercase();
        if !normalized.is_empty() {
            parts.push(normalized);
        }
    }
    parts.join("|")
}

/// The first `count` hex characters of SHA-256 over `value`.
///
/// Upper-case hex is never produced; digests shorter than `count` characters
/// simply yield the whole digest.
#[must_use]
pub(crate) fn sha256_hex_prefix(value: &str, count: usize) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut hex = String::with_capacity(count);
    for byte in digest {
        for nibble in [byte >> 4, byte & 0x0f] {
            if hex.len() == count {
                return hex;
            }
            hex.push(char::from_digit(u32::from(nibble), 16).unwrap_or('0'));
        }
    }
    hex
}

/// Sanitizes a user id for use in ids and file names.
///
/// ASCII alphanumerics and `.`, `_`, `-` are kept; every other character
/// becomes `_`. An empty user id becomes `"default"`.
#[must_use]
pub(crate) fn sanitize_user(user_id: &str) -> String {
    if user_id.is_empty() {
        return "default".to_string();
    }
    user_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// Returns the trimmed value when it carries text.
fn non_blank(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    // Python counterparts: `tests/unit_tests/extensions/context_evolver/`
    // (schema/memory tests for `TaskMemory`, `ReasoningBankMemory`,
    // `PersonalMemory`, and their `get_score`/`to_vector_node` helpers).
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn draft() -> MemoryDraft {
        MemoryDraft {
            content: "Prefer `rg` over `grep`".into(),
            when_to_use: Some("searching a workspace".into()),
            title: Some("Use ripgrep".into()),
            description: Some("Fast recursive search".into()),
            section: "tooling".into(),
            label: Some("search".into()),
            attributes: BTreeMap::new(),
            source_type: Some("success".into()),
        }
    }

    fn record() -> MemoryRecord {
        MemoryRecord::new("user-1", draft()).with_id("ace_user-1_0123456789abcdef")
    }

    #[test]
    fn draft_tolerates_missing_null_and_unknown_fields() {
        let minimal: MemoryDraft = serde_json::from_value(json!({"content": "hello"})).unwrap();
        assert_eq!(minimal.content, "hello");
        assert_eq!(minimal.section, "general");
        assert!(minimal.attributes.is_empty());
        assert_eq!(minimal, MemoryDraft::new("hello"));

        let nulls: MemoryDraft = serde_json::from_value(json!({
            "content": "hello",
            "section": null,
            "attributes": null,
            "when_to_use": null
        }))
        .unwrap();
        assert_eq!(nulls.section, "general");
        assert!(nulls.attributes.is_empty());
        assert_eq!(nulls.when_to_use, None);

        let extras: MemoryDraft = serde_json::from_value(json!({
            "content": "hello",
            "unknown_field": 42,
            "attributes": {"target": "x"}
        }))
        .unwrap();
        assert_eq!(extras.attributes.get("target"), Some(&json!("x")));
    }

    #[test]
    fn record_tolerates_missing_null_and_unknown_fields() {
        let minimal: MemoryRecord = serde_json::from_value(json!({"content": "hello"})).unwrap();
        assert_eq!(minimal.section, "general");
        assert!(minimal.attributes.is_empty());
        assert_eq!(minimal.helpful, 0);
        assert_eq!(minimal.harmful, 0);
        assert_eq!(minimal.id, "");
        assert_eq!(
            minimal,
            MemoryRecord {
                content: "hello".into(),
                ..MemoryRecord::default()
            }
        );

        let nulls: MemoryRecord = serde_json::from_value(json!({
            "id": "id-1",
            "user_id": "user-1",
            "content": "hello",
            "attributes": null,
            "section": null,
            "source_type": null
        }))
        .unwrap();
        assert_eq!(nulls.section, "general");
        assert!(nulls.attributes.is_empty());
        assert_eq!(nulls.source_type, None);

        let extras: MemoryRecord = serde_json::from_value(json!({
            "id": "id-1",
            "content": "hello",
            "unknown_field": true,
            "attributes": {"target": "x"}
        }))
        .unwrap();
        assert_eq!(extras.attributes.get("target"), Some(&json!("x")));
    }

    #[test]
    fn request_round_trip_keeps_attributes() {
        let mut request = AddMemoryRequest {
            content: "remember this".into(),
            query: Some("what?".into()),
            when_to_use: Some("when testing".into()),
            title: Some("title".into()),
            description: Some("description".into()),
            section: "general".into(),
            label: Some("label".into()),
            attributes: BTreeMap::new(),
        };
        request
            .attributes
            .insert("target".into(), json!("workspace"));

        let record = MemoryRecord::from_request("user-1", &request);
        assert_eq!(record.user_id, "user-1");
        assert_eq!(record.id, "");
        assert_eq!(record.attributes, request.attributes);
        assert_eq!(record.source_type, None);

        let restored = record.to_request();
        assert_eq!(restored.content, request.content);
        assert_eq!(restored.query, None);
        assert_eq!(restored.when_to_use, request.when_to_use);
        assert_eq!(restored.title, request.title);
        assert_eq!(restored.description, request.description);
        assert_eq!(restored.section, request.section);
        assert_eq!(restored.label, request.label);
        assert_eq!(restored.attributes, request.attributes);

        let draft = record.to_draft();
        assert_eq!(draft.content, record.content);
        assert_eq!(draft.attributes, record.attributes);
        assert_eq!(MemoryDraft::from_request(&restored), draft);
    }

    #[test]
    fn memory_id_is_deterministic_and_prefixed_per_family() {
        let draft = draft();
        let cases = [
            (Algo::Ace, "ace"),
            (Algo::ReasoningBank, "reasoning_bank"),
            (Algo::Reme, "reme"),
            (Algo::Refcon, "reme"),
            (Algo::Divcon, "reme"),
            (Algo::Cognition, "cognition"),
        ];
        for (algo, prefix) in cases {
            let id = generate_memory_id(algo, "user-1", &draft);
            assert!(id.starts_with(&format!("{prefix}_user-1_")), "{id}");
            assert_eq!(id.len(), prefix.len() + 1 + "user-1".len() + 1 + 16);
            assert_eq!(id, generate_memory_id(algo, "user-1", &draft));
        }
        let hash = generate_memory_id(Algo::Ace, "user-1", &draft)
            .rsplit('_')
            .next()
            .unwrap()
            .to_string();
        assert_eq!(hash, record().dedup_hash());
    }

    #[test]
    fn memory_id_sanitizes_the_user_segment() {
        let draft = MemoryDraft::new("content");
        assert_eq!(
            generate_memory_id(Algo::Ace, "a b/c", &draft),
            format!("ace_a_b_c_{}", sha256_hex_prefix("content", 16))
        );
        assert!(generate_memory_id(Algo::Ace, "", &draft).starts_with("ace_default_"));
        assert!(generate_memory_id(Algo::Ace, "..ok._-9", &draft).contains("_..ok._-9_"));
        assert_eq!(sanitize_user("üser"), "_ser");
    }

    #[test]
    fn identity_and_dedup_hash_are_stable_and_case_insensitive() {
        let record = record();
        assert_eq!(
            record.identity(),
            "use ripgrep|prefer `rg` over `grep`|searching a workspace"
        );
        assert_eq!(record.dedup_hash().len(), 16);

        let mut other = record.clone();
        other.content = "  PREFER `RG` OVER `GREP`  ".into();
        other.title = Some(" use RIPGREP ".into());
        other.when_to_use = Some("Searching A Workspace".into());
        assert_eq!(other.identity(), record.identity());
        assert_eq!(other.dedup_hash(), record.dedup_hash());

        let mut blank = record;
        blank.title = Some("   ".into());
        blank.when_to_use = None;
        assert_eq!(blank.identity(), "prefer `rg` over `grep`");
        assert_ne!(blank.dedup_hash(), other.dedup_hash());

        let empty = MemoryRecord::default();
        assert_eq!(empty.identity(), "");
        assert_eq!(empty.dedup_hash(), sha256_hex_prefix("", 16));
    }

    #[test]
    fn render_block_selects_present_fields() {
        assert_eq!(
            record().render_block(),
            "Title: Use ripgrep\n\
             Description: Fast recursive search\n\
             When to use: searching a workspace\n\
             Section: tooling\n\
             Content: Prefer `rg` over `grep`"
        );

        let plain = MemoryRecord::new("user-1", MemoryDraft::new("only content"));
        assert_eq!(
            plain.render_block(),
            "Section: general\nContent: only content"
        );

        let sparse = MemoryRecord {
            content: "   ".into(),
            section: String::new(),
            ..MemoryRecord::default()
        };
        assert_eq!(sparse.render_block(), "");
    }

    #[test]
    fn feedback_score_matches_python_math() {
        let fresh = record();
        assert_eq!(fresh.feedback_score(), 1.0);

        let boosted = fresh.clone().with_feedback(3, 0);
        assert_eq!(boosted.helpful, 3);
        assert_eq!(boosted.feedback_score(), 4.0);

        let penalized = fresh.with_feedback(1, 3);
        assert_eq!(penalized.harmful, 3);
        assert_eq!(penalized.feedback_score(), 0.5);

        let saturated = penalized.with_feedback(u32::MAX, u32::MAX);
        assert_eq!(saturated.helpful, u32::MAX);
        assert_eq!(saturated.harmful, u32::MAX);
        assert_eq!(saturated.feedback_score(), 1.0);
    }

    #[test]
    fn validate_reports_validation_errors_naming_the_record() {
        let mut broken = record();
        broken.user_id = "  ".into();
        let error = broken.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("user_id"), "{}", error.message());
        assert!(error.message().contains(&broken.id), "{}", error.message());

        let mut broken = record();
        broken.content = "\n".into();
        let error = broken.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("content"), "{}", error.message());

        let mut broken = record();
        broken.id = String::new();
        let error = broken.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("blank id"), "{}", error.message());
        assert!(
            error.message().contains(&broken.user_id),
            "{}",
            error.message()
        );

        let error = MemoryRecord::default().validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("<unassigned>"),
            "{}",
            error.message()
        );

        record().validate().unwrap();
    }

    #[test]
    fn to_vector_node_carries_metadata() {
        let mut memory = record();
        memory
            .attributes
            .insert("content".into(), json!("must not overwrite"));
        memory.attributes.insert("target".into(), json!("search"));
        let node = memory.to_vector_node(vec![0.5, 0.25]);

        assert_eq!(node.id, memory.id);
        assert_eq!(node.embedding, vec![0.5, 0.25]);
        assert_eq!(node.metadata["type"], json!("memory"));
        assert_eq!(node.metadata["user_id"], json!("user-1"));
        assert_eq!(node.metadata["content"], json!(memory.content));
        assert_eq!(node.metadata["section"], json!("tooling"));
        assert_eq!(node.metadata["when_to_use"], json!("searching a workspace"));
        assert_eq!(node.metadata["title"], json!("Use ripgrep"));
        assert_eq!(node.metadata["description"], json!("Fast recursive search"));
        assert_eq!(node.metadata["label"], json!("search"));
        assert_eq!(node.metadata["source_type"], json!("success"));
        assert_eq!(node.metadata["helpful"], json!(0));
        assert_eq!(node.metadata["harmful"], json!(0));
        assert_eq!(node.metadata["target"], json!("search"));

        let plain = MemoryRecord::new("user-1", MemoryDraft::new("content"));
        let node = plain.to_vector_node(Vec::new());
        assert_eq!(node.metadata.len(), 6);
        assert!(!node.metadata.contains_key("title"));
        assert!(!node.metadata.contains_key("target"));
    }
}
