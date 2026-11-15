//! Item sources that feed the capability-tree index builder.
//!
//! Ports Python `openjiuwen.symphony.retrieval.build.scanners` (`BaseScanner`,
//! `ScannedItem`, `SkillScanner`, `PluginScanner`) and
//! `build/io/items_jsonl.py` (`parse_jsonl_scanned_items`, `load_items_jsonl_text`), and
//! adapts them to this crate's types: a source produces [`SourceItem`](crate::retrieval::build::scanner::SourceItem)s whose `item` is a
//! [`RetrieverItem`](crate::retrieval::types::RetrieverItem) keyed by a worker id, plus
//! the per-item fields the later build stages read (`content`, `skill_path`,
//! `classification`, `tags`, `metadata`).
//!
//! A source may be a set of capability descriptors, a scanned
//! [`SkillInventory`](crate::discovery::models::SkillInventory), a `SKILL.md` directory, or a JSONL item file. Every source uses
//! one flat, [`ItemSources::sort`](crate::retrieval::build::scanner::ItemSources::sort)ed item set and reports problems per item in
//! [`ItemSources::diagnostics`](crate::retrieval::build::scanner::ItemSources::diagnostics); one bad item never fails a whole scan. Only an
//! unusable root (a skill directory that cannot be listed) is an error.
//!
//! # Resolution rules
//!
//! - **Descriptors** — `worker_id` is the descriptor's `capability_id`, `content` its
//!   `semantic_content`, `content_hash` its cached content hash, `skill_path` the
//!   descriptor's `metadata["skill_file"]` (or `entrypoint`/`source_root`, which is what
//!   the discovery scanner stamps) when present, else `capability://<type>/<id>`, and
//!   `item.payload` the descriptor's JSON. A blank `capability_id` is diagnosed and
//!   skipped.
//! - **Inventories** — one item per [`SkillRecord`](crate::discovery::models::SkillRecord), `skill_path` from `skill_file`,
//!   metadata from `source`/`version`/`author`, `item.payload` the record's JSON.
//! - **Skill directories** — the [`SkillFolderScanner`](crate::discovery::scanner::SkillFolderScanner) discovery result; each
//!   [`ScannedSkill`](crate::discovery::scanner::ScannedSkill) contributes its
//!   descriptor's fields plus the markdown `body` as `content`, and a relative
//!   `skill_path` is joined onto the scan root. Skill diagnostics are converted one for
//!   one, so a broken skill folder is reported instead of aborting the scan.
//! - **JSONL** — one JSON object per line, tolerant of blank lines, unknown keys and the
//!   legacy `contentExtendParam.skillId` shape. A row without a usable id is diagnosed
//!   and skipped, as is a row that is not a JSON object; a repeated id keeps the first
//!   row. Reading a JSONL file delegates to [`ItemSources::from_jsonl`](crate::retrieval::build::scanner::ItemSources::from_jsonl) after the read.
//!
//! Everything is deterministic: items are ordered by `worker_id`, diagnostics by
//! `(subject, code)`, tags are sorted and de-duplicated, and no output depends on the
//! wall clock or on hash-map iteration order.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::discovery::models::{SkillInventory, SkillRecord};
use crate::discovery::scanner::{
    DiagnosticSeverity, SkillDiagnostic, SkillFolderScanner, SkillScanConfig,
};
use crate::error::{validation, Error, Result};
use crate::model::CapabilityDescriptor;
use crate::retrieval::types::{RetrieverItem, DEFAULT_ITEM_TYPE};

/// Diagnostic code: a JSONL row is not a JSON object, or a descriptor could not be
/// rendered as an item payload.
pub const DIAGNOSTIC_MALFORMED_ROW: &str = "malformed_row";
/// Diagnostic code: an item has no usable id, so it was skipped.
pub const DIAGNOSTIC_MISSING_ITEM_ID: &str = "missing_item_id";
/// Diagnostic code: two items share a worker id; the first one wins.
pub const DIAGNOSTIC_DUPLICATE_ITEM_ID: &str = "duplicate_item_id";
/// Diagnostic code: a source file could not be read.
///
/// This is the code the discovery scanner uses for unreadable skill files, so its
/// diagnostics keep their code when they are converted into [`SourceDiagnostic`]s.
pub const DIAGNOSTIC_UNREADABLE_FILE: &str = "unreadable_file";
/// Diagnostic code: an item has no description and no content of its own, so only its
/// id is indexable; the item is kept.
pub const DIAGNOSTIC_BLANK_ITEM: &str = "blank_item";

/// The item type of a skill source ([`DEFAULT_ITEM_TYPE`]).
pub const ITEM_TYPE_SKILL: &str = "skill";
/// The item type of a plugin source.
pub const ITEM_TYPE_PLUGIN: &str = "plugin";

/// The path prefix of a JSONL item's synthetic source path.
pub const JSONL_PATH_PREFIX: &str = "jsonl://skill/";

/// Descriptor metadata keys consulted, in order, for an item's source path.
const DESCRIPTOR_PATH_KEYS: [&str; 3] = ["skill_file", "entrypoint", "source_root"];
/// Descriptor metadata keys copied onto a [`SourceItem`] as scalar text.
const DESCRIPTOR_METADATA_KEYS: [&str; 9] = [
    "author",
    "dont_select_when",
    "github_url",
    "is_official",
    "official",
    "select_when",
    "source",
    "stars",
    "version",
];
/// JSONL metadata keys copied onto a [`SourceItem`]'s metadata as scalar text: the
/// canonical key and the camel-case spelling the legacy rows use.
const JSONL_METADATA_KEYS: [(&str, &str); 7] = [
    ("author", "author"),
    ("github_url", "githubUrl"),
    ("is_official", "isOfficial"),
    ("official", "official"),
    ("source", "source"),
    ("stars", "stars"),
    ("version", "version"),
];

/// One index-build input item, resolved from any source.
///
/// The identity trio is `worker_id` (the catalog's key), `item.item_id` (the same value)
/// and `item.payload` (the serialized source record when the source has one, else the
/// worker id). `content`, `skill_path`, `classification`, `tags`, and `metadata` are what
/// the grouping, preset and catalog stages read.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SourceItem {
    /// The stable id of the item; it is also the catalog record's `worker_id`.
    pub worker_id: String,
    /// The retrievable item handed to the tree, with `item_id = worker_id`.
    pub item: RetrieverItem,
    /// The display name; the source's own name, without a fallback.
    pub name: String,
    /// The routing description shown in a disclosure.
    pub description: String,
    /// The markdown body or semantic content; it feeds `retrieval_text`.
    pub content: String,
    /// The item's source path: a `SKILL.md` path, a `capability://` URI or a `jsonl://` URI.
    pub skill_path: String,
    /// The raw category hint (for example `documents/pdf`); may be empty.
    pub classification: String,
    /// The retrieval tags, sorted and de-duplicated.
    pub tags: Vec<String>,
    /// The item kind: [`ITEM_TYPE_SKILL`] or [`ITEM_TYPE_PLUGIN`].
    pub item_type: String,
    /// The SHA-256 hex digest of the source bytes when the source knows it, else empty.
    pub content_hash: String,
    /// Scalar provenance: `select_when`, `dont_select_when`, `source`, `author`,
    /// `version`, `official`, `github_url`, `stars`, ….
    pub metadata: BTreeMap<String, String>,
}

impl SourceItem {
    /// Whether the item carries nothing but its id: no name, no description, no content.
    fn is_blank(&self) -> bool {
        let named = !self.name.trim().is_empty() && self.name.trim() != self.worker_id;
        !named && self.description.trim().is_empty() && self.content.trim().is_empty()
    }
}

/// A per-item or per-row problem; the scan continues.
///
/// `severity` is the discovery scanner's [`DiagnosticSeverity`], so a converted skill
/// diagnostic keeps its meaning: [`DiagnosticSeverity::Error`] means the item was
/// skipped, [`DiagnosticSeverity::Warning`] means it was kept.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SourceDiagnostic {
    /// A stable machine-readable code (one of the `DIAGNOSTIC_*` constants).
    pub code: String,
    /// What the problem is about: a worker id, a source path, or a `line:<n>` subject.
    pub subject: String,
    /// A human-readable explanation.
    pub detail: String,
    /// Whether the item was skipped ([`DiagnosticSeverity::Error`]) or kept.
    pub severity: DiagnosticSeverity,
}

impl SourceDiagnostic {
    /// Builds a warning diagnostic, which leaves the item in place.
    #[must_use]
    pub fn warning(
        code: impl Into<String>,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            subject: subject.into(),
            detail: detail.into(),
            severity: DiagnosticSeverity::Warning,
        }
    }

    /// Builds an error diagnostic, which means the item was skipped.
    #[must_use]
    pub fn error(
        code: impl Into<String>,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            subject: subject.into(),
            detail: detail.into(),
            severity: DiagnosticSeverity::Error,
        }
    }

    /// Whether this diagnostic skipped its item.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.severity == DiagnosticSeverity::Error
    }
}

/// Everything one or more item sources produced.
///
/// `items` is sorted by `worker_id` and `diagnostics` by `(subject, code)` after every
/// constructor and after [`ItemSources::merge`], so two runs over the same sources
/// compare equal. `item_paths` is the source-path inventory the manifest records, in
/// first-seen order with duplicates removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ItemSources {
    /// The item kind shared by every item in this set.
    pub item_type: String,
    /// The resolved items, sorted by `worker_id`.
    pub items: Vec<SourceItem>,
    /// The problems found, sorted by `(subject, code)`.
    pub diagnostics: Vec<SourceDiagnostic>,
    /// The distinct source paths, in first-seen order.
    pub item_paths: Vec<String>,
}

impl Default for ItemSources {
    /// Returns an empty skill source set.
    fn default() -> Self {
        Self::new(DEFAULT_ITEM_TYPE)
    }
}

impl ItemSources {
    /// Builds an empty source set of `item_type`, trimmed and lowercased.
    ///
    /// An unrecognized `item_type` is kept as given (trimmed and lowercased) so a caller
    /// can assemble sources before validating it; [`normalize_item_type`] is the
    /// validating entry point.
    #[must_use]
    pub fn new(item_type: impl Into<String>) -> Self {
        Self {
            item_type: lenient_item_type(&item_type.into()),
            items: Vec::new(),
            diagnostics: Vec::new(),
            item_paths: Vec::new(),
        }
    }

    /// Resolves capability descriptors into items.
    ///
    /// See the module docs for the field mapping; a descriptor with a blank
    /// `capability_id` becomes a [`DIAGNOSTIC_MISSING_ITEM_ID`] error and is skipped.
    #[must_use]
    pub fn from_descriptors(items: impl IntoIterator<Item = CapabilityDescriptor>) -> Self {
        let mut sources = Self::new(DEFAULT_ITEM_TYPE);
        for descriptor in items {
            if let Some(item) = descriptor_item(&descriptor, &mut sources.diagnostics) {
                push_path(&mut sources.item_paths, &item.skill_path);
                sources.items.push(item);
            }
        }
        sources.finished()
    }

    /// Resolves a scanned [`SkillInventory`] into items, one per [`SkillRecord`].
    #[must_use]
    pub fn from_inventory(inventory: &SkillInventory) -> Self {
        let mut sources = Self::new(ITEM_TYPE_SKILL);
        for record in &inventory.items {
            if let Some(item) = record_item(record, &mut sources.diagnostics) {
                push_path(&mut sources.item_paths, &item.skill_path);
                sources.items.push(item);
            }
        }
        sources.finished()
    }

    /// Scans a `SKILL.md` root and resolves every discovered skill into an item.
    ///
    /// A relative `skill_path` is joined onto `root`, `content` is the markdown body
    /// after the front matter, and every [`SkillDiagnostic`] becomes a
    /// [`SourceDiagnostic`]. `item_paths` holds the absolute `root` as its single entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `root` when the root does not exist or is not
    /// a directory. A broken skill folder is *not* an error: it arrives as a diagnostic.
    pub async fn from_skill_directory(
        root: impl AsRef<Path>,
        config: &SkillScanConfig,
    ) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let discovery = SkillFolderScanner::with_config(root.clone(), config.clone())
            .discover()
            .await?;
        let mut sources = Self::new(ITEM_TYPE_SKILL);
        for diagnostic in &discovery.diagnostics {
            sources.diagnostics.push(converted_diagnostic(diagnostic));
        }
        for skill in &discovery.skills {
            if let Some(mut item) = descriptor_item(&skill.descriptor, &mut sources.diagnostics) {
                item.content = skill.body.trim().to_string();
                item.skill_path = rooted_skill_path(&root, &item.skill_path);
                sources.items.push(item);
            }
        }
        let absolute = tokio::fs::canonicalize(&root).await.unwrap_or(root);
        sources.item_paths = vec![path_text(&absolute)];
        Ok(sources.finished())
    }

    /// Reads JSONL item rows, tolerating anything that is not a usable item.
    ///
    /// Blank lines are skipped. Each remaining line must be a JSON object with a usable
    /// id (`id`, `worker_id`, or the legacy `contentExtendParam.skillId`); everything
    /// else is diagnosed and skipped. This never returns an error for content problems.
    ///
    /// ```
    /// use openjiuwen_symphony::retrieval::build::scanner::ItemSources;
    ///
    /// let sources = ItemSources::from_jsonl(r#"{"id":"pdf","name":"PDF","tags":["docs"]}"#);
    /// assert_eq!(sources.worker_ids(), vec!["pdf"]);
    /// assert_eq!(sources.items[0].tags, vec!["docs"]);
    /// ```
    #[must_use]
    pub fn from_jsonl(text: &str) -> Self {
        let mut sources = Self::new(ITEM_TYPE_SKILL);
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        for (index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let subject = format!("line:{}", index + 1);
            let value: Value = match serde_json::from_str(trimmed) {
                Ok(value) => value,
                Err(error) => {
                    sources.diagnostics.push(SourceDiagnostic::error(
                        DIAGNOSTIC_MALFORMED_ROW,
                        subject,
                        format!("row is not valid JSON: {error}"),
                    ));
                    continue;
                }
            };
            let Some(object) = value.as_object() else {
                sources.diagnostics.push(SourceDiagnostic::error(
                    DIAGNOSTIC_MALFORMED_ROW,
                    subject,
                    "row is not a JSON object",
                ));
                continue;
            };
            let Some(worker_id) = jsonl_id(object) else {
                sources.diagnostics.push(SourceDiagnostic::error(
                    DIAGNOSTIC_MISSING_ITEM_ID,
                    subject,
                    "row has no usable id (`id`, `worker_id` or `contentExtendParam.skillId`)",
                ));
                continue;
            };
            if !seen.insert(worker_id.clone()) {
                sources.diagnostics.push(SourceDiagnostic::warning(
                    DIAGNOSTIC_DUPLICATE_ITEM_ID,
                    worker_id.clone(),
                    format!("duplicate item id `{worker_id}`; keeping the first row"),
                ));
                continue;
            }
            let item = jsonl_item(object, &worker_id);
            push_path(&mut sources.item_paths, &item.skill_path);
            sources.items.push(item);
        }
        sources.finished()
    }

    /// Reads JSONL items from `path`, then delegates to [`ItemSources::from_jsonl`](crate::retrieval::build::scanner::ItemSources::from_jsonl).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the file does not exist, and [`Error::External`]
    /// for any other read failure.
    pub async fn from_jsonl_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match tokio::fs::read_to_string(path).await {
            Ok(text) => Ok(Self::from_jsonl(&text)),
            Err(error) if error.kind() == ErrorKind::NotFound => Err(Error::not_found(format!(
                "jsonl item file `{}` does not exist",
                path.display()
            ))),
            Err(error) => Err(Error::external(error)),
        }
    }

    /// Unions `other` into this set; on a shared worker id `self` wins.
    ///
    /// The shared id is reported as a [`DIAGNOSTIC_DUPLICATE_ITEM_ID`] warning,
    /// diagnostics concatenate, `item_paths` keeps its order with duplicates removed,
    /// and the result is sorted. `item_type` stays this set's kind.
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        let mut seen: BTreeSet<String> = self
            .items
            .iter()
            .map(|item| item.worker_id.clone())
            .collect();
        for item in other.items {
            if !seen.insert(item.worker_id.clone()) {
                self.diagnostics.push(SourceDiagnostic::warning(
                    DIAGNOSTIC_DUPLICATE_ITEM_ID,
                    item.worker_id.as_str(),
                    format!(
                        "worker id `{}` is already present; keeping the existing item",
                        item.worker_id
                    ),
                ));
                continue;
            }
            self.items.push(item);
        }
        self.diagnostics.extend(other.diagnostics);
        for path in other.item_paths {
            push_path(&mut self.item_paths, &path);
        }
        self.sort();
        self
    }

    /// The number of items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the set holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The worker ids of the items, sorted.
    #[must_use]
    pub fn worker_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self
            .items
            .iter()
            .map(|item| item.worker_id.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Replaces the item kind, normalized leniently like [`ItemSources::new`].
    #[must_use]
    pub fn with_item_type(mut self, item_type: impl Into<String>) -> Self {
        self.item_type = lenient_item_type(&item_type.into());
        self
    }

    /// The diagnostics whose subject is `subject`, in sorted order.
    #[must_use]
    pub fn diagnostics_for(&self, subject: &str) -> Vec<&SourceDiagnostic> {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.subject == subject)
            .collect()
    }

    /// Sorts the items by `worker_id` and the diagnostics by `(subject, code)`.
    pub fn sort(&mut self) {
        self.items.sort_by(|left, right| {
            left.worker_id
                .cmp(&right.worker_id)
                .then_with(|| left.skill_path.cmp(&right.skill_path))
        });
        self.diagnostics.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.code.cmp(&right.code))
        });
    }

    /// Diagnoses the blank items, then sorts.
    fn finished(mut self) -> Self {
        let blanks: Vec<SourceDiagnostic> = self
            .items
            .iter()
            .filter(|item| item.is_blank())
            .map(|item| {
                SourceDiagnostic::warning(
                    DIAGNOSTIC_BLANK_ITEM,
                    item.worker_id.clone(),
                    "item has no description and no content of its own; only its id is indexable",
                )
            })
            .collect();
        self.diagnostics.extend(blanks);
        self.sort();
        self
    }
}

/// Normalizes an item type to [`ITEM_TYPE_SKILL`] or [`ITEM_TYPE_PLUGIN`].
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `item_type` for any other value.
pub fn normalize_item_type(raw: &str) -> Result<String> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        ITEM_TYPE_SKILL | ITEM_TYPE_PLUGIN => Ok(normalized),
        _ => Err(validation(
            "item_type",
            format!(
                "expected `{ITEM_TYPE_SKILL}` or `{ITEM_TYPE_PLUGIN}`, got `{}`",
                raw.trim()
            ),
        )),
    }
}

/// Normalizes an item type without rejecting an unknown one.
fn lenient_item_type(raw: &str) -> String {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        DEFAULT_ITEM_TYPE.to_string()
    } else {
        normalized
    }
}

/// Resolves one capability descriptor into an item, diagnosing instead of failing.
fn descriptor_item(
    descriptor: &CapabilityDescriptor,
    diagnostics: &mut Vec<SourceDiagnostic>,
) -> Option<SourceItem> {
    let worker_id = descriptor.capability_id.trim().to_string();
    if worker_id.is_empty() {
        diagnostics.push(SourceDiagnostic::error(
            DIAGNOSTIC_MISSING_ITEM_ID,
            "<blank capability_id>",
            format!(
                "capability of type `{}` has no capability_id, so it cannot be indexed",
                descriptor.capability_type.trim()
            ),
        ));
        return None;
    }
    let payload = match serde_json::to_string(descriptor) {
        Ok(payload) => payload,
        Err(error) => {
            diagnostics.push(SourceDiagnostic::warning(
                DIAGNOSTIC_MALFORMED_ROW,
                worker_id.clone(),
                format!("cannot render the capability descriptor as an item payload: {error}"),
            ));
            worker_id.clone()
        }
    };
    let name = descriptor.name.trim().to_string();
    Some(SourceItem {
        item: RetrieverItem::new(worker_id.clone(), payload)
            .with_label(name.clone())
            .with_description(descriptor.description.trim()),
        worker_id,
        name,
        description: descriptor.description.trim().to_string(),
        content: descriptor.semantic_content.trim().to_string(),
        skill_path: descriptor_skill_path(descriptor),
        classification: descriptor.classification.trim().to_string(),
        tags: sorted_tags(descriptor.tags.iter().cloned()),
        item_type: DEFAULT_ITEM_TYPE.to_string(),
        content_hash: descriptor.content_hash.trim().to_string(),
        metadata: descriptor_metadata(descriptor),
    })
}

/// Resolves one inventory record into an item, diagnosing instead of failing.
fn record_item(
    record: &SkillRecord,
    diagnostics: &mut Vec<SourceDiagnostic>,
) -> Option<SourceItem> {
    let worker_id = record.worker_id.trim().to_string();
    if worker_id.is_empty() {
        diagnostics.push(SourceDiagnostic::error(
            DIAGNOSTIC_MISSING_ITEM_ID,
            "<blank worker_id>",
            "skill inventory record has no worker_id, so it cannot be indexed",
        ));
        return None;
    }
    let payload = match serde_json::to_string(record) {
        Ok(payload) => payload,
        Err(error) => {
            diagnostics.push(SourceDiagnostic::warning(
                DIAGNOSTIC_MALFORMED_ROW,
                worker_id.clone(),
                format!("cannot render the skill record as an item payload: {error}"),
            ));
            worker_id.clone()
        }
    };
    let name = record.name.trim().to_string();
    let description = record.description.trim().to_string();
    let mut metadata = BTreeMap::new();
    for (key, value) in [
        ("source", record.source.as_str()),
        ("version", record.version.as_str()),
        ("author", record.author.as_str()),
    ] {
        let value = value.trim();
        if !value.is_empty() {
            metadata.insert(key.to_string(), value.to_string());
        }
    }
    let source_root = record.source_root.trim();
    if !source_root.is_empty() {
        metadata.insert("source_root".to_string(), source_root.to_string());
    }
    Some(SourceItem {
        item: RetrieverItem::new(worker_id.clone(), payload)
            .with_label(name.clone())
            .with_description(description.as_str()),
        worker_id,
        name,
        description,
        content: String::new(),
        skill_path: record.skill_file.trim().to_string(),
        classification: String::new(),
        tags: Vec::new(),
        item_type: ITEM_TYPE_SKILL.to_string(),
        content_hash: record.content_hash.trim().to_string(),
        metadata,
    })
}

/// Converts a discovery diagnostic into a source diagnostic, keeping its code and path.
fn converted_diagnostic(diagnostic: &SkillDiagnostic) -> SourceDiagnostic {
    let severity = if diagnostic.is_error() {
        DiagnosticSeverity::Error
    } else {
        DiagnosticSeverity::Warning
    };
    SourceDiagnostic {
        code: diagnostic.code.clone(),
        subject: diagnostic.path.clone(),
        detail: diagnostic.message.clone(),
        severity,
    }
}

/// The item's source path: the descriptor's own path, else a `capability://` URI.
fn descriptor_skill_path(descriptor: &CapabilityDescriptor) -> String {
    for key in DESCRIPTOR_PATH_KEYS {
        if let Some(value) = descriptor.metadata.get(key).and_then(Value::as_str) {
            let value = value.trim();
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    let capability_type = descriptor.capability_type.trim();
    let kind = if capability_type.is_empty() {
        "capability"
    } else {
        capability_type
    };
    format!("capability://{kind}/{}", descriptor.capability_id.trim())
}

/// Joins a relative `raw` path onto `root`, leaving URIs and absolute paths untouched.
fn rooted_skill_path(root: &Path, raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.starts_with("capability://")
        || trimmed.starts_with(JSONL_PATH_PREFIX)
    {
        return trimmed.to_string();
    }
    let path = Path::new(trimmed);
    if path.is_absolute() {
        trimmed.to_string()
    } else {
        path_text(&root.join(path))
    }
}

/// Renders `path` as a lossy string.
fn path_text(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

/// The descriptor's scalar metadata, plus its `source` field when the map lacks one.
fn descriptor_metadata(descriptor: &CapabilityDescriptor) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    for key in DESCRIPTOR_METADATA_KEYS {
        if let Some(value) = descriptor.metadata.get(key).and_then(scalar_text) {
            let value = value.trim();
            if !value.is_empty() {
                metadata.insert(key.to_string(), value.to_string());
            }
        }
    }
    let source = descriptor.source.trim();
    if !source.is_empty() {
        metadata
            .entry("source".to_string())
            .or_insert_with(|| source.to_string());
    }
    metadata
}

/// Renders a JSON scalar as text; arrays, objects and null yield nothing.
fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Trims, sorts, and de-duplicates tags.
fn sorted_tags(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let tags: BTreeSet<String> = values
        .into_iter()
        .map(|tag| tag.trim().to_string())
        .filter(|tag| !tag.is_empty())
        .collect();
    tags.into_iter().collect()
}

/// Appends `path` when it is non-blank and not already present.
fn push_path(paths: &mut Vec<String>, path: &str) {
    let path = path.trim();
    if path.is_empty() || paths.iter().any(|existing| existing == path) {
        return;
    }
    paths.push(path.to_string());
}

/// The usable id of one JSONL row, in priority order.
fn jsonl_id(object: &Map<String, Value>) -> Option<String> {
    jsonl_text(object, &["id", "worker_id"])
        .or_else(|| jsonl_nested_text(object, &["skillId", "skill_id"]))
}

/// Builds one item from a JSONL row that already has a usable id.
fn jsonl_item(object: &Map<String, Value>, worker_id: &str) -> SourceItem {
    let name = jsonl_text(object, &["name", "skillName"])
        .or_else(|| jsonl_nested_text(object, &["skillName", "name"]))
        .unwrap_or_else(|| worker_id.to_string());
    let description = jsonl_text(object, &["description", "skillDesc"])
        .or_else(|| jsonl_nested_text(object, &["skillDesc", "description"]))
        .unwrap_or_else(|| name.clone());
    let skill_path = jsonl_text(object, &["skill_path", "skillPath", "path"])
        .or_else(|| jsonl_nested_text(object, &["skillPath", "skill_path", "path"]))
        .unwrap_or_else(|| format!("{JSONL_PATH_PREFIX}{worker_id}"));
    let content = jsonl_text(object, &["content", "skill_content", "skillContent"])
        .or_else(|| jsonl_nested_text(object, &["skillContent", "skill_content", "content"]))
        .unwrap_or_default();
    let content_hash = jsonl_text(object, &["content_hash", "contentHash"])
        .or_else(|| jsonl_nested_text(object, &["contentHash", "content_hash"]))
        .unwrap_or_default();
    let classification = jsonl_text(object, &["classification", "category"])
        .or_else(|| jsonl_nested_text(object, &["classification", "category"]))
        .unwrap_or_default();
    let mut tags = jsonl_strings(object.get("tags"));
    tags.extend(jsonl_strings(
        jsonl_nested(object).and_then(|nested| nested.get("tags")),
    ));
    let tags = sorted_tags(tags);
    let mut metadata = BTreeMap::new();
    for (key, alias) in JSONL_METADATA_KEYS {
        if let Some(value) =
            jsonl_text(object, &[key, alias]).or_else(|| jsonl_nested_text(object, &[alias, key]))
        {
            metadata.insert(key.to_string(), value);
        }
    }
    SourceItem {
        item: RetrieverItem::new(worker_id.to_string(), worker_id.to_string())
            .with_label(name.clone())
            .with_description(description.as_str()),
        worker_id: worker_id.to_string(),
        name,
        description,
        content,
        skill_path,
        classification,
        tags,
        item_type: ITEM_TYPE_SKILL.to_string(),
        content_hash,
        metadata,
    }
}

/// Reads the first non-blank scalar among `keys` on the row itself.
fn jsonl_text(object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(scalar_text))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Reads the first non-blank scalar among `keys` inside `contentExtendParam`.
fn jsonl_nested_text(object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    let nested = jsonl_nested(object)?;
    jsonl_text(nested, keys)
}

/// The legacy `contentExtendParam` object of a row, when it is an object.
fn jsonl_nested(object: &Map<String, Value>) -> Option<&Map<String, Value>> {
    object.get("contentExtendParam")?.as_object()
}

/// The string members of `value` when it is an array, trimmed and with blanks dropped.
fn jsonl_strings(value: Option<&Value>) -> Vec<String> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(Value::as_str)
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    impl SourceItem {
        /// Reads one metadata value, or `""` when the key is absent.
        fn metadata_of(&self, key: &str) -> &str {
            self.metadata.get(key).map_or("", String::as_str)
        }
    }

    /// Writes one skill folder holding `contents` as its `SKILL.md`.
    async fn write_skill(root: &Path, folder: &str, contents: &str) {
        let directory = root.join(folder);
        tokio::fs::create_dir_all(&directory).await.unwrap();
        tokio::fs::write(directory.join("SKILL.md"), contents)
            .await
            .unwrap();
    }

    /// One fully populated descriptor; `skill_file` points at `/skills/pdf/SKILL.md`.
    fn descriptor(id: &str) -> CapabilityDescriptor {
        CapabilityDescriptor::new("skill", id, format!("{id} skill"))
            .with_description(format!("{id} description"))
            .with_classification("documents/pdf")
            .with_tags(["pdf", "docs", "pdf"])
            .with_semantic_content("# Body\ncontent")
            .with_source("local")
            .with_metadata("skill_file", Value::String("/skills/pdf/SKILL.md".into()))
            .with_metadata("select_when", Value::String("reading pdfs".into()))
            .with_metadata("version", Value::String("1.2.0".into()))
            .with_metadata("stars", Value::from(7))
    }

    #[test]
    fn descriptors_resolve_into_items() {
        let sources = ItemSources::from_descriptors([descriptor("pdf")]);
        assert_eq!(sources.item_type, ITEM_TYPE_SKILL);
        assert_eq!(sources.worker_ids(), vec!["pdf"]);
        assert_eq!(sources.item_paths, vec!["/skills/pdf/SKILL.md"]);
        let item = &sources.items[0];
        assert_eq!(
            (item.name.as_str(), item.description.as_str()),
            ("pdf skill", "pdf description")
        );
        assert_eq!(item.content, "# Body\ncontent");
        assert_eq!(item.classification, "documents/pdf");
        assert_eq!(item.tags, vec!["docs", "pdf"]);
        assert_eq!(item.item_type, ITEM_TYPE_SKILL);
        assert_eq!(item.item.item_id, "pdf");
        assert_eq!(item.item.label, "pdf skill");
        assert_eq!(
            ["select_when", "version", "stars", "source"].map(|key| item.metadata_of(key)),
            ["reading pdfs", "1.2.0", "7", "local"]
        );
        let payload: Value = serde_json::from_str(&item.item.payload).unwrap();
        assert_eq!(payload["capability_id"], "pdf");
        assert_eq!(payload["capability_type"], "skill");
        assert_eq!(sources.diagnostics, Vec::<SourceDiagnostic>::new());
    }

    #[test]
    fn descriptors_fall_back_to_capability_uris_and_diagnose_bad_ones() {
        let sources = ItemSources::from_descriptors([
            CapabilityDescriptor::new("tool", "search", "Search"),
            CapabilityDescriptor::new("", "typed", ""),
        ]);
        assert_eq!(sources.items[0].skill_path, "capability://tool/search");
        assert_eq!(sources.items[1].skill_path, "capability://capability/typed");
        assert_eq!(sources.items[1].name, "");
        assert_eq!(sources.diagnostics.len(), 1);
        assert_eq!(sources.diagnostics[0].code, DIAGNOSTIC_BLANK_ITEM);
        assert_eq!(sources.diagnostics[0].subject, "typed");
        assert!(!sources.diagnostics[0].is_error());
        assert_eq!(sources.diagnostics_for("typed").len(), 1);

        let mut hashed = CapabilityDescriptor::new("skill", "empty", "");
        hashed.content_hash = "abc123".to_string();
        let sources = ItemSources::from_descriptors([hashed]);
        assert_eq!(sources.items[0].content_hash, "abc123");
        assert_eq!(sources.diagnostics[0].code, DIAGNOSTIC_BLANK_ITEM);

        let blank = CapabilityDescriptor::new("skill", "  ", "Nameless");
        let sources = ItemSources::from_descriptors([blank]);
        assert_eq!(sources.len(), 0);
        assert_eq!(sources.diagnostics.len(), 1);
        assert_eq!(sources.diagnostics[0].code, DIAGNOSTIC_MISSING_ITEM_ID);
        assert!(sources.diagnostics[0].is_error());
        assert_eq!(sources.diagnostics[0].subject, "<blank capability_id>");
        assert_eq!(
            sources.diagnostics_for("other"),
            Vec::<&SourceDiagnostic>::new()
        );
    }

    #[test]
    fn inventory_records_resolve_into_items() {
        let inventory = SkillInventory::new(vec![
            SkillRecord::new("beta", "Beta")
                .with_description("second")
                .with_skill_file("beta/SKILL.md")
                .with_version("2.0.0")
                .with_author("Ada")
                .with_content_hash("def456")
                .with_source_root("/skills"),
            SkillRecord::new("alpha", "Alpha").with_skill_file("alpha/SKILL.md"),
            SkillRecord::new("gamma", "").with_skill_file("gamma/SKILL.md"),
        ]);
        let sources = ItemSources::from_inventory(&inventory);
        assert_eq!(sources.worker_ids(), vec!["alpha", "beta", "gamma"]);
        assert_eq!(
            sources.item_paths,
            vec!["alpha/SKILL.md", "beta/SKILL.md", "gamma/SKILL.md"]
        );
        let beta = &sources.items[1];
        assert_eq!(beta.description, "second");
        assert_eq!(beta.content_hash, "def456");
        assert_eq!(beta.content, "");
        assert_eq!(
            ["version", "author", "source", "source_root"].map(|key| beta.metadata_of(key)),
            ["2.0.0", "Ada", "local", "/skills"]
        );
        let payload: Value = serde_json::from_str(&beta.item.payload).unwrap();
        assert_eq!(payload["worker_id"], "beta");
        // A record with no name, description or content is reported, not dropped; a name
        // alone is routable text, so `alpha` stays quiet.
        assert_eq!(sources.diagnostics.len(), 1);
        assert_eq!(sources.diagnostics[0].code, DIAGNOSTIC_BLANK_ITEM);
        assert_eq!(sources.diagnostics[0].subject, "gamma");
        assert!(!sources.diagnostics[0].is_error());
    }

    #[test]
    fn jsonl_rows_resolve_into_items_and_ignore_unknown_keys() {
        let text = concat!(
            "\n",
            "{\"id\":\"pdf\",\"name\":\"PDF\",\"description\":\"read pdfs\",",
            "\"skill_path\":\"/skills/pdf\",\"content\":\"# PDF\",",
            "\"content_hash\":\"abc\",\"classification\":\"documents\",",
            "\"tags\":[\"docs\",\"pdf\",\"docs\"],\"stars\":12,\"is_official\":true,",
            "\"author\":\"Ada\",\"github_url\":\"https://example.test/pdf\",",
            "\"future_field\":{\"nested\":1}}\n",
            "{\"worker_id\":\"solo\"}\n",
        );
        let sources = ItemSources::from_jsonl(text);
        assert_eq!(sources.worker_ids(), vec!["pdf", "solo"]);
        assert_eq!(
            sources.item_paths,
            vec!["/skills/pdf", "jsonl://skill/solo"]
        );
        let item = &sources.items[0];
        assert_eq!(
            (item.name.as_str(), item.description.as_str()),
            ("PDF", "read pdfs")
        );
        assert_eq!(item.item.payload, "pdf");
        assert_eq!(item.skill_path, "/skills/pdf");
        assert_eq!(item.content, "# PDF");
        assert_eq!(item.content_hash, "abc");
        assert_eq!(item.classification, "documents");
        assert_eq!(item.tags, vec!["docs", "pdf"]);
        assert_eq!(
            ["stars", "is_official", "author", "github_url"].map(|key| item.metadata_of(key)),
            ["12", "true", "Ada", "https://example.test/pdf"]
        );
        assert_eq!(sources.items[1].name, "solo");
        assert_eq!(sources.items[1].description, "solo");
        assert_eq!(sources.items[1].skill_path, "jsonl://skill/solo");
        assert_eq!(sources.diagnostics, Vec::<SourceDiagnostic>::new());
    }

    #[test]
    fn jsonl_accepts_the_legacy_content_extend_param_shape() {
        let text = concat!(
            "{\"contentExtendParam\":{\"skillId\":\"legacy\",\"skillName\":\"Legacy\",",
            "\"skillDesc\":\"old shape\",\"skillPath\":\"/skills/legacy\",",
            "\"skillContent\":\"body\",\"contentHash\":\"fff\",\"stars\":\"3\",",
            "\"isOfficial\":false,\"author\":\"Bob\"}}\n",
        );
        let sources = ItemSources::from_jsonl(text);
        assert_eq!(sources.worker_ids(), vec!["legacy"]);
        let item = &sources.items[0];
        assert_eq!(
            (item.name.as_str(), item.description.as_str()),
            ("Legacy", "old shape")
        );
        assert_eq!(item.skill_path, "/skills/legacy");
        assert_eq!(item.content, "body");
        assert_eq!(item.content_hash, "fff");
        assert_eq!(
            ["stars", "is_official", "author"].map(|key| item.metadata_of(key)),
            ["3", "false", "Bob"]
        );
    }

    #[test]
    fn jsonl_reports_missing_ids_and_malformed_rows() {
        let text = concat!(
            "{}\n",
            "{\"name\":\"no id\"}\n",
            "not json\n",
            "[1,2]\n",
            "{\"id\":\"good\"}\n",
        );
        let sources = ItemSources::from_jsonl(text);
        assert_eq!(sources.worker_ids(), vec!["good"]);
        let codes: Vec<&str> = sources
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect();
        assert_eq!(
            codes,
            vec![
                DIAGNOSTIC_MISSING_ITEM_ID,
                DIAGNOSTIC_MISSING_ITEM_ID,
                DIAGNOSTIC_MALFORMED_ROW,
                DIAGNOSTIC_MALFORMED_ROW,
            ]
        );
        assert!(sources.diagnostics.iter().all(SourceDiagnostic::is_error));
        assert!(sources
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.subject.starts_with("line:")));
    }

    #[test]
    fn jsonl_keeps_the_first_duplicate_and_orders_paths_by_first_seen() {
        let text = concat!(
            "{\"id\":\"b\",\"name\":\"First B\"}\n",
            "{\"id\":\"a\"}\n",
            "{\"id\":\"b\",\"name\":\"Second B\"}\n",
        );
        let sources = ItemSources::from_jsonl(text);
        assert_eq!(sources.worker_ids(), vec!["a", "b"]);
        assert_eq!(
            sources.item_paths,
            vec!["jsonl://skill/b", "jsonl://skill/a"]
        );
        let beta = sources
            .items
            .iter()
            .find(|item| item.worker_id == "b")
            .unwrap();
        assert_eq!(beta.name, "First B");
        assert_eq!(sources.diagnostics.len(), 1);
        assert_eq!(sources.diagnostics[0].code, DIAGNOSTIC_DUPLICATE_ITEM_ID);
        assert_eq!(sources.diagnostics[0].subject, "b");
        assert!(!sources.diagnostics[0].is_error());
    }

    #[test]
    fn normalize_item_type_accepts_the_two_kinds_and_rejects_the_rest() {
        assert_eq!(normalize_item_type(" skill ").unwrap(), ITEM_TYPE_SKILL);
        assert_eq!(normalize_item_type("Plugin").unwrap(), ITEM_TYPE_PLUGIN);
        let error = normalize_item_type("workflow").unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("item_type"));
        assert_eq!(
            normalize_item_type("").unwrap_err().kind(),
            crate::error::ErrorKind::Validation
        );
        assert_eq!(ItemSources::new(" PLUGIN ").item_type, ITEM_TYPE_PLUGIN);
        assert_eq!(ItemSources::new("").item_type, DEFAULT_ITEM_TYPE);
        assert_eq!(ItemSources::default().item_type, DEFAULT_ITEM_TYPE);
        let sources = ItemSources::from_jsonl("{\"id\":\"p\"}").with_item_type("plugin");
        assert_eq!(sources.item_type, ITEM_TYPE_PLUGIN);
    }

    #[test]
    fn merge_unions_by_worker_id_and_keeps_self() {
        let first = ItemSources::from_descriptors([descriptor("b")]);
        let second = ItemSources::from_descriptors([descriptor("a"), descriptor("b")]);
        let merged = first.merge(second);
        assert_eq!(merged.worker_ids(), vec!["a", "b"]);
        assert_eq!(merged.items[1].name, "b skill");
        assert_eq!(merged.item_paths, vec!["/skills/pdf/SKILL.md"]);
        assert_eq!(merged.item_type, ITEM_TYPE_SKILL);
        assert_eq!(merged.diagnostics.len(), 1);
        assert_eq!(merged.diagnostics[0].code, DIAGNOSTIC_DUPLICATE_ITEM_ID);
        assert_eq!(merged.diagnostics[0].subject, "b");
        assert!(!merged.diagnostics[0].is_error());

        let first = ItemSources::from_jsonl("{\"id\":\"a\",\"skill_path\":\"/one\"}");
        let second = ItemSources::from_jsonl("{\"id\":\"c\",\"skill_path\":\"/two\"}");
        let merged = first.merge(second.with_item_type("plugin"));
        assert_eq!(merged.item_paths, vec!["/one", "/two"]);
        assert_eq!(merged.worker_ids(), vec!["a", "c"]);
        assert_eq!(
            merged.item_type, ITEM_TYPE_SKILL,
            "the receiver keeps its kind"
        );
        assert_eq!(merged.diagnostics, Vec::<SourceDiagnostic>::new());
    }

    #[test]
    fn sources_round_trip_through_json() {
        let sources = ItemSources::from_descriptors([descriptor("pdf")]);
        let decoded: ItemSources =
            serde_json::from_str(&serde_json::to_string(&sources).unwrap()).unwrap();
        assert_eq!(decoded, sources);
        let tolerant: ItemSources =
            serde_json::from_str(r#"{"items":[{"worker_id":"a","future":1}]}"#).unwrap();
        assert_eq!(tolerant.item_type, DEFAULT_ITEM_TYPE);
        assert_eq!(tolerant.items[0].worker_id, "a");
        assert_eq!(tolerant.diagnostics, Vec::<SourceDiagnostic>::new());
        let item: SourceItem = serde_json::from_str(r#"{"worker_id":"a","unknown":true}"#).unwrap();
        assert_eq!(item.tags, Vec::<String>::new());
        let diagnostic = SourceDiagnostic::warning(DIAGNOSTIC_BLANK_ITEM, "a", "blank");
        let decoded: SourceDiagnostic =
            serde_json::from_str(&serde_json::to_string(&diagnostic).unwrap()).unwrap();
        assert_eq!(decoded, diagnostic);
    }

    #[tokio::test]
    async fn skill_directory_yields_items_and_diagnoses_broken_folders() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let alpha = "---\nname: Alpha\ndescription: First skill\n\
                     category: documents\n---\n# Alpha\nbody\n";
        write_skill(root, "alpha", alpha).await;
        write_skill(root, "beta", "# Beta\n\nNo front matter here.\n").await;
        write_skill(root, "broken", "---\nname: Broken\nnever closed\n").await;

        let sources = ItemSources::from_skill_directory(root, &SkillScanConfig::default())
            .await
            .unwrap();
        assert_eq!(sources.item_type, ITEM_TYPE_SKILL);
        assert_eq!(sources.worker_ids(), vec!["alpha", "beta"]);
        let item = &sources.items[0];
        assert_eq!(
            (
                item.name.as_str(),
                item.description.as_str(),
                item.classification.as_str()
            ),
            ("Alpha", "First skill", "documents")
        );
        assert_eq!(item.content, "# Alpha\nbody");
        assert!(item.skill_path.ends_with("alpha/SKILL.md"));
        assert!(item.skill_path.starts_with(&path_text(root)));
        assert_eq!(item.content_hash.len(), 64);
        assert_eq!(sources.items[1].content, "# Beta\n\nNo front matter here.");
        assert_eq!(
            sources.item_paths,
            vec![path_text(&tokio::fs::canonicalize(root).await.unwrap())]
        );
        let codes: Vec<&str> = sources
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect();
        assert!(codes.contains(&"missing_front_matter"));
        assert!(codes.contains(&"unterminated_front_matter"));
        let broken = sources
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == "unterminated_front_matter")
            .unwrap();
        assert!(broken.is_error());
        assert!(broken.subject.ends_with("broken/SKILL.md"));
        assert_eq!(sources.len(), 2, "the broken folder yields no item");

        let missing = root.join("absent");
        let error = ItemSources::from_skill_directory(&missing, &SkillScanConfig::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("root"));
    }

    #[tokio::test]
    async fn jsonl_file_reads_rows_and_reports_a_missing_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("items.jsonl");
        tokio::fs::write(&path, "{\"id\":\"a\"}\n{\"id\":\"b\"}\n")
            .await
            .unwrap();
        let sources = ItemSources::from_jsonl_file(&path).await.unwrap();
        assert_eq!(sources.worker_ids(), vec!["a", "b"]);
        let error = ItemSources::from_jsonl_file(directory.path().join("absent.jsonl"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
        assert!(error.message().contains("absent.jsonl"));
    }
}
