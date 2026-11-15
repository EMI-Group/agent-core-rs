//! The virtual category tree over one `SkillFs` catalog.
//!
//! Ports `DirectoryEntry` and `SkillDirectoryView` from Python
//! `openjiuwen.symphony.discovery.skillfs`, but not Python's tree model: Python projects
//! a `RetrieverNode` taxonomy (LLM-written category nodes) into paths, while the Rust
//! crate derives the same path space from each skill's
//! `CapabilityDescriptor::classification`, which the
//! scanner fills from the front-matter `category`.
//!
//! # The category tree
//!
//! A classification is split on `>` (or, when it has no `>`, on `/`), each segment is
//! trimmed, and blank segments are dropped; a blank classification puts the skill
//! directly under the root. `Data > Export` therefore creates the node `/Data`, the node
//! `/Data/Export`, and puts `Data > Export`-classified skills one level deeper at
//! `/Data/Export/<worker_id>`.
//!
//! Segment collisions inside one parent — the same segment spelling in two different
//! case forms, or a category label that equals a skill's worker id — are resolved in
//! deterministic `(segment.to_lowercase(), segment, kind, worker_id.to_lowercase(),
//! worker_id)` order: the first candidate keeps the segment and every later one gets a
//! `~2`, `~3`, … suffix. Python instead sanitizes segments with a safe-character regex
//! and suffixes with a short content hash; see the deviations below.
//!
//! # Paths
//!
//! Every path is absolute with `/` separators ([`SkillDirectoryView::normalize_path`]),
//! the root is `/` labelled [`ROOT_LABEL`] with description [`ROOT_DESCRIPTION`], a
//! category is a plain path (`/Data/Export`), and a skill's path is its category path
//! plus its segment (`/Data/Export/pdf-export`). A skill's metadata path is its path
//! plus [`METADATA_FILENAME`], which is what resolves a `META.md` path back to its skill.
//!
//! # Deviations from Python
//!
//! - The tree comes from `classification` chains, not from a retriever taxonomy, so
//!   category descriptions are always empty (Python's nodes carry LLM-written routing
//!   text) and the root is always labelled [`ROOT_LABEL`].
//! - Segments are the labels themselves: Python's `_segment_base` rewrites unsafe
//!   characters to `-` and its `_unique_segment` falls back to an 8-hex-digit content
//!   hash; here only the `~n` suffix is added.
//! - A chain deeper than [`MAX_CATEGORY_DEPTH`] is truncated (Python has no bound); this
//!   keeps tree building and walking recursive over data the caller controls.
//! - `casefold()` becomes `to_lowercase()`; see
//!   [`matcher`].
//!
//! ```
//! # use openjiuwen_symphony::discovery::SkillFs;
//! # use openjiuwen_symphony::discovery::directory_toolkit::SkillDirectoryView;
//! # async fn example() -> openjiuwen_symphony::Result<()> {
//! let catalog = SkillFs::open("skills").await?;
//! let view = SkillDirectoryView::from_catalog(&catalog);
//! assert_eq!(view.root_path(), "/");
//! assert!(view.skill_count("/")? <= catalog.records().len());
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::discovery::models::SkillRecord;
use crate::discovery::SkillFs;
use crate::error::{Error, Result};
use crate::model::CapabilityDescriptor;

use super::render::inline_text;

/// The root path of the virtual directory.
pub const ROOT_PATH: &str = "/";
/// The root's label.
pub const ROOT_LABEL: &str = "ROOT";
/// The root's description.
pub const ROOT_DESCRIPTION: &str = "Installed Skills";
/// The metadata file name that [`SkillDirectoryView::metadata_path`] appends.
pub const METADATA_FILENAME: &str = "META.md";
/// The separator between two category levels in a classification.
pub const CATEGORY_SEPARATOR: char = '>';
/// The separator used instead when a classification has no [`CATEGORY_SEPARATOR`].
pub const CATEGORY_PATH_SEPARATOR: char = '/';
/// The separator rendered between two category labels (`A > B`).
pub const CATEGORY_LABEL_SEPARATOR: &str = " > ";
/// The deepest category chain a record contributes; deeper segments are dropped.
pub const MAX_CATEGORY_DEPTH: usize = 32;
/// The suffix introducer used to disambiguate a colliding sibling segment.
pub const DISAMBIGUATION_MARKER: char = '~';

/// The kind of one directory entry.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// A virtual category.
    #[default]
    Dir,
    /// A Skill.
    Skill,
}

impl EntryKind {
    /// Every variant, in path-sort order.
    pub const ALL: [Self; 2] = [Self::Dir, Self::Skill];

    /// The wire name of this kind (`"dir"` / `"skill"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dir => "dir",
            Self::Skill => "skill",
        }
    }
}

impl std::fmt::Display for EntryKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One model-visible row of the directory: a category or a Skill.
///
/// `depth` is `0` for an entry as it sits in its parent's child list and the relative
/// depth below the requested scope for the entries [`SkillDirectoryView::entries`] and
/// [`SkillDirectoryView::tree_entries`] return.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DirectoryEntry {
    /// The absolute virtual path (`/`, `/Data`, `/Data/Export/pdf-export`).
    pub path: String,
    /// Whether this entry is a category or a Skill.
    pub kind: EntryKind,
    /// The display label: a category label or the Skill's display name.
    pub label: String,
    /// The description; always empty for a category here.
    pub description: String,
    /// The Skill's worker id; empty for a category.
    pub worker_id: String,
    /// The depth relative to the requested scope root.
    pub depth: usize,
}

impl DirectoryEntry {
    /// Builds a category entry at `path` with `label`.
    #[must_use]
    pub fn directory(path: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind: EntryKind::Dir,
            label: label.into(),
            ..Self::default()
        }
    }

    /// Builds a Skill entry at `path` for `worker_id`.
    #[must_use]
    pub fn skill(path: impl Into<String>, worker_id: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind: EntryKind::Skill,
            worker_id: worker_id.into(),
            ..Self::default()
        }
    }

    /// Whether this entry is a category.
    #[must_use]
    pub fn is_directory(&self) -> bool {
        self.kind == EntryKind::Dir
    }

    /// Returns the entry with `depth` replaced.
    #[must_use]
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

/// One category node of the virtual tree.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DirectoryNode {
    /// The absolute path (`/` for the root).
    pub path: String,
    /// The display label (the classification segment, or [`ROOT_LABEL`]).
    pub label: String,
    /// The description; empty for every category, [`ROOT_DESCRIPTION`] for the root.
    pub description: String,
}

impl DirectoryNode {
    /// Builds a node at `path` with `label` and `description`.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            label: label.into(),
            description: description.into(),
        }
    }
}

/// One node of the raw category tree built before segment disambiguation.
#[derive(Debug, Default)]
struct RawDirectory {
    /// Child categories keyed by their raw segment.
    directories: BTreeMap<String, Self>,
    /// Worker ids of the skills sitting directly in this category.
    skills: Vec<String>,
}

/// A candidate child segment before disambiguation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Segment {
    /// The raw segment (a category label or a worker id).
    base: String,
    /// Whether this candidate is a category or a Skill.
    kind: EntryKind,
    /// The raw child key: the raw segment for a category, the worker id for a Skill.
    key: String,
}

/// The path projection of the installed-Skill catalog.
///
/// The view owns the category nodes, each node's direct children, the per-node Skill
/// count (its own skills plus every descendant's), the records, their paths, and the
/// `META.md` index. It never copies markdown bodies: read those through
/// [`SkillFs::read_body`](crate::discovery::SkillFs::read_body).
#[derive(Debug, Clone, Default)]
pub struct SkillDirectoryView {
    /// The root label ([`ROOT_LABEL`]).
    root_label: String,
    /// The root description ([`ROOT_DESCRIPTION`]).
    root_description: String,
    /// Category nodes keyed by absolute path, the root included.
    nodes: BTreeMap<String, DirectoryNode>,
    /// Direct children keyed by the parent path, in tree order.
    children: BTreeMap<String, Vec<DirectoryEntry>>,
    /// Skill counts keyed by path: this category plus every descendant.
    skill_counts: BTreeMap<String, usize>,
    /// The records keyed by worker id.
    records: BTreeMap<String, SkillRecord>,
    /// Each skill's directory path keyed by worker id.
    record_paths: BTreeMap<String, String>,
    /// The worker id behind each `META.md` path.
    meta_index: BTreeMap<String, String>,
}

impl SkillDirectoryView {
    /// Builds the view over a catalog, taking each skill's category from its
    /// descriptor's `classification`.
    #[must_use]
    pub fn from_catalog(catalog: &SkillFs) -> Self {
        let categories: BTreeMap<String, String> = catalog
            .records()
            .iter()
            .map(|record| {
                let classification = catalog
                    .descriptor(&record.worker_id)
                    .map(|descriptor| descriptor.classification.clone())
                    .unwrap_or_default();
                (record.worker_id.clone(), classification)
            })
            .collect();
        Self::from_records(catalog.records(), &categories)
    }

    /// Builds the view from records plus their category chains.
    ///
    /// `categories` maps a worker id to its classification (`Data > Export`); a worker
    /// id it does not mention sits directly under the root. A record with a blank
    /// worker id is skipped, and a duplicate worker id keeps the last record.
    #[must_use]
    pub fn from_records(records: &[SkillRecord], categories: &BTreeMap<String, String>) -> Self {
        let mut records_by_id: BTreeMap<String, SkillRecord> = BTreeMap::new();
        let mut raw = RawDirectory::default();
        for record in records {
            if record.worker_id.trim().is_empty() {
                continue;
            }
            let segments = categories
                .get(&record.worker_id)
                .map(|classification| category_segments(classification))
                .unwrap_or_default();
            let mut cursor = &mut raw;
            for segment in &segments {
                cursor = cursor.directories.entry(segment.clone()).or_default();
            }
            cursor.skills.push(record.worker_id.clone());
            records_by_id.insert(record.worker_id.clone(), record.clone());
        }

        let mut view = Self {
            root_label: ROOT_LABEL.to_string(),
            root_description: ROOT_DESCRIPTION.to_string(),
            nodes: BTreeMap::new(),
            children: BTreeMap::new(),
            skill_counts: BTreeMap::new(),
            records: records_by_id,
            record_paths: BTreeMap::new(),
            meta_index: BTreeMap::new(),
        };
        view.nodes.insert(
            ROOT_PATH.to_string(),
            DirectoryNode::new(ROOT_PATH, ROOT_LABEL, ROOT_DESCRIPTION),
        );
        view.build(&raw, ROOT_PATH);
        view
    }

    /// Normalizes a virtual directory path.
    ///
    /// The result always starts with `/`, never ends with `/` (except the root itself),
    /// and collapses empty and `.` segments. `"/"`, `""`, and `"/a//b/"` normalize to
    /// `/`, `/`, and `/a/b`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the path contains a `..` segment: the view is
    /// a virtual namespace, so there is no parent traversal.
    pub fn normalize_path(value: &str) -> Result<String> {
        let trimmed = value.trim();
        let raw = if trimmed.is_empty() {
            ROOT_PATH
        } else {
            trimmed
        };
        let mut segments: Vec<&str> = Vec::new();
        for segment in raw.split('/') {
            if segment.is_empty() || segment == "." {
                continue;
            }
            if segment == ".." {
                return Err(Error::validation(
                    "Skill directory paths must not contain '..'",
                ));
            }
            segments.push(segment);
        }
        if segments.is_empty() {
            return Ok(ROOT_PATH.to_string());
        }
        Ok(format!("/{}", segments.join("/")))
    }

    /// The root path ([`ROOT_PATH`]).
    #[must_use]
    pub fn root_path(&self) -> &str {
        ROOT_PATH
    }

    /// The root label ([`ROOT_LABEL`], or the empty string for a view built from no
    /// records).
    #[must_use]
    pub fn root_label(&self) -> &str {
        &self.root_label
    }

    /// The root description ([`ROOT_DESCRIPTION`]).
    #[must_use]
    pub fn root_description(&self) -> &str {
        &self.root_description
    }

    /// The category node at `path`, when it exists.
    #[must_use]
    pub fn node(&self, path: &str) -> Option<&DirectoryNode> {
        self.nodes.get(path)
    }

    /// Every category path, the root included, sorted.
    #[must_use]
    pub fn node_paths(&self) -> Vec<&str> {
        self.nodes.keys().map(String::as_str).collect()
    }

    /// The direct children of the category at `path`, in tree order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] `No such Skill directory: <path>` when no category
    /// has that (normalized) path.
    pub fn children(&self, path: &str) -> Result<Vec<DirectoryEntry>> {
        let normalized = Self::normalize_path(path)?;
        if !self.nodes.contains_key(&normalized) {
            return Err(Error::validation(format!(
                "No such Skill directory: {normalized}"
            )));
        }
        Ok(self.children.get(&normalized).cloned().unwrap_or_default())
    }

    /// The number of Skills in the category at `path` plus every descendant.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] `No such Skill directory: <path>` for an unknown
    /// path.
    pub fn skill_count(&self, path: &str) -> Result<usize> {
        let normalized = Self::normalize_path(path)?;
        match self.skill_counts.get(&normalized) {
            Some(count) => Ok(*count),
            None => Err(Error::validation(format!(
                "No such Skill directory: {normalized}"
            ))),
        }
    }

    /// The number of Skills at `path`, or `0` for an unknown path.
    ///
    /// This is the total variant [`skill_count`](Self::skill_count) that rendering uses:
    /// a row must always be renderable, even for a hand-built entry.
    #[must_use]
    pub fn skill_count_or_zero(&self, path: &str) -> usize {
        Self::normalize_path(path)
            .ok()
            .and_then(|normalized| self.skill_counts.get(&normalized).copied())
            .unwrap_or(0)
    }

    /// The record of the skill with `worker_id`, when it is known.
    #[must_use]
    pub fn record(&self, worker_id: &str) -> Option<&SkillRecord> {
        self.records.get(worker_id)
    }

    /// Every worker id in the view, sorted.
    #[must_use]
    pub fn record_ids(&self) -> Vec<&str> {
        self.records.keys().map(String::as_str).collect()
    }

    /// The virtual path of the skill with `worker_id`, when it is known.
    #[must_use]
    pub fn record_path(&self, worker_id: &str) -> Option<&str> {
        self.record_paths.get(worker_id).map(String::as_str)
    }

    /// The entries in the `paths` scopes.
    ///
    /// Reproduces Python's `_scoped_entries` for the `entries` entry point: a scope
    /// contributes its own category row when `directory_entry` is set, then its children
    /// (recursed into only when `recursive`), with `max_depth` counted relative to the
    /// scope root and `directories_only` dropping Skill rows. Entries are de-duplicated
    /// by `(kind, path)` across scopes and keep scope order.
    ///
    /// `directory_entry` also passes Python's `recursive=None` (rather than `false`),
    /// and `_scoped_entries` only walks when `recursive is not None`. The result is the
    /// scope row by itself — `directory_entry` asks for the category, not for its
    /// listing, and the execution layer renders that one row.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown scope path (or a malformed one).
    pub fn entries(
        &self,
        paths: &[String],
        recursive: bool,
        max_depth: Option<usize>,
        directories_only: bool,
        directory_entry: bool,
    ) -> Result<Vec<DirectoryEntry>> {
        self.scoped(
            paths,
            max_depth,
            directory_entry,
            if directory_entry {
                None
            } else {
                Some(recursive)
            },
            directories_only,
            false,
        )
    }

    /// Each scope root followed by its descendants, in tree order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown scope path (or a malformed one).
    pub fn tree_entries(
        &self,
        paths: &[String],
        max_depth: Option<usize>,
        directories_only: bool,
    ) -> Result<Vec<DirectoryEntry>> {
        self.scoped(paths, max_depth, true, Some(true), directories_only, false)
    }

    /// Every category and Skill entry in the `paths` scopes.
    ///
    /// A `META.md` path resolves to its Skill entry instead of a category.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown scope path (or a malformed one).
    pub fn searchable_entries(
        &self,
        paths: &[String],
        max_depth: Option<usize>,
    ) -> Result<Vec<DirectoryEntry>> {
        self.scoped(paths, max_depth, true, Some(true), false, true)
    }

    /// The records in the `paths` scopes, de-duplicated in entry order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown scope path (or a malformed one).
    pub fn scoped_records(
        &self,
        paths: &[String],
        max_depth: Option<usize>,
    ) -> Result<Vec<SkillRecord>> {
        let entries = self.searchable_entries(paths, max_depth)?;
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut records = Vec::new();
        for entry in &entries {
            if entry.worker_id.is_empty() || !seen.insert(entry.worker_id.as_str()) {
                continue;
            }
            if let Some(record) = self.records.get(&entry.worker_id) {
                records.push(record.clone());
            }
        }
        Ok(records)
    }

    /// The `META.md` metadata path of the skill with `worker_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] `Unknown Skill ID: <worker_id>` for an unknown
    /// skill.
    pub fn metadata_path(&self, worker_id: &str) -> Result<String> {
        match self.record_paths.get(worker_id) {
            Some(path) => Ok(format!("{path}/{METADATA_FILENAME}")),
            None => Err(Error::validation(format!("Unknown Skill ID: {worker_id}"))),
        }
    }

    /// The record behind a `META.md` path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] `No such Skill metadata path: <path>` when the
    /// (normalized) path is not a known skill's metadata path.
    pub fn resolve_metadata_path(&self, path: &str) -> Result<&SkillRecord> {
        let normalized = Self::normalize_path(path)?;
        let worker_id = self.meta_index.get(&normalized).ok_or_else(|| {
            Error::validation(format!("No such Skill metadata path: {normalized}"))
        })?;
        self.records
            .get(worker_id)
            .ok_or_else(|| Error::validation(format!("No such Skill metadata path: {normalized}")))
    }

    /// The category chain of a path, joined with [`CATEGORY_LABEL_SEPARATOR`].
    ///
    /// The root (and any path the view does not know at all) renders as
    /// [`ROOT_LABEL`]; each label is escaped with
    /// [`inline_text`], so a
    /// label can never forge another row.
    #[must_use]
    pub fn category_from_path(&self, path: &str) -> String {
        let mut labels: Vec<String> = Vec::new();
        let mut current = Self::normalize_path(path).unwrap_or_else(|_| ROOT_PATH.to_string());
        while current != ROOT_PATH && !current.is_empty() && current != "." {
            let label = self.nodes.get(&current).map_or_else(
                || last_segment(&current).to_string(),
                |node| node.label.clone(),
            );
            labels.push(inline_text(&label));
            current = parent_path(&current);
        }
        if labels.is_empty() {
            return ROOT_LABEL.to_string();
        }
        labels.reverse();
        labels.join(CATEGORY_LABEL_SEPARATOR)
    }

    /// The category chain of one skill.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] `Unknown Skill ID: <worker_id>` for an unknown
    /// skill.
    pub fn skill_category(&self, worker_id: &str) -> Result<String> {
        let path = self
            .record_paths
            .get(worker_id)
            .ok_or_else(|| Error::validation(format!("Unknown Skill ID: {worker_id}")))?;
        Ok(self.category_from_path(&parent_path(path)))
    }

    /// The category chain of one skill, or the empty string for an unknown skill.
    ///
    /// This is the total variant [`skill_category`](Self::skill_category) that rendering
    /// uses.
    #[must_use]
    pub fn skill_category_or_empty(&self, worker_id: &str) -> String {
        self.skill_category(worker_id).unwrap_or_default()
    }

    /// The ordered category names and descriptions above one skill, one per line.
    ///
    /// Python emits the label followed by its searchable description for every level,
    /// outermost first, dropping empty parts. Category descriptions are always empty
    /// here, so a `Data > Export` skill yields `Data\nExport`. An unknown skill yields
    /// the empty string.
    #[must_use]
    pub fn category_text(&self, worker_id: &str) -> String {
        let Some(record_path) = self.record_paths.get(worker_id) else {
            return String::new();
        };
        let mut levels: Vec<(String, String)> = Vec::new();
        let mut current = parent_path(record_path);
        while current != ROOT_PATH && !current.is_empty() && current != "." {
            if let Some(node) = self.nodes.get(&current) {
                levels.push((
                    node.label.clone(),
                    searchable_description(&node.description),
                ));
            }
            current = parent_path(&current);
        }
        let mut parts: Vec<String> = Vec::new();
        for (label, description) in levels.iter().rev() {
            for part in [label, description] {
                if !part.is_empty() {
                    parts.push(part.clone());
                }
            }
        }
        parts.join("\n")
    }

    /// The path of the one category whose chain (or single label) matches `category`.
    ///
    /// `/`, `ROOT`, and `root` are the root. A `A > B` value matches the full chain
    /// case-insensitively; a single-label value also matches a node whose own label is
    /// that value, which is how `Export` finds `/Data/Export`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] `Unknown Skill category: <category>` when nothing
    /// matches, and `Ambiguous Skill category; use its full category chain: <category>`
    /// when more than one category matches.
    pub fn category_path(&self, category: &str) -> Result<String> {
        let value = category.trim();
        if value.is_empty() {
            return Err(Error::validation("category: must be a non-empty string"));
        }
        if value == ROOT_PATH || value.eq_ignore_ascii_case(ROOT_LABEL) {
            return Ok(ROOT_PATH.to_string());
        }
        let normalized = normalize_category_chain(value);
        let folded = normalized.to_lowercase();
        let mut matches: Vec<&String> = self
            .nodes
            .keys()
            .filter(|path| path.as_str() != ROOT_PATH)
            .filter(|path| self.category_from_path(path).to_lowercase() == folded)
            .collect();
        if matches.is_empty() && !normalized.contains(CATEGORY_SEPARATOR) {
            matches = self
                .nodes
                .iter()
                .filter(|(path, _)| path.as_str() != ROOT_PATH)
                .filter(|(_, node)| node.label.trim().to_lowercase() == folded)
                .map(|(path, _)| path)
                .collect();
        }
        match matches.len() {
            0 => Err(Error::validation(format!(
                "Unknown Skill category: {value}"
            ))),
            1 => Ok(matches.remove(0).clone()),
            _ => Err(Error::validation(format!(
                "Ambiguous Skill category; use its full category chain: {value}"
            ))),
        }
    }

    /// Every directory and Skill label in the view, sorted for diagnostics.
    ///
    /// The order is `(label.to_lowercase(), label)` with duplicates collapsed; the root
    /// label is included.
    #[must_use]
    pub fn entry_labels(&self) -> Vec<String> {
        let mut labels: BTreeSet<(String, String)> = BTreeSet::new();
        for node in self.nodes.values() {
            labels.insert((node.label.to_lowercase(), node.label.clone()));
        }
        for entries in self.children.values() {
            for entry in entries {
                labels.insert((entry.label.to_lowercase(), entry.label.clone()));
            }
        }
        labels.into_iter().map(|(_, label)| label).collect()
    }

    /// Builds the final nodes and child lists under `parent_path`.
    ///
    /// Returns the number of Skills in this subtree (its own plus every descendant's).
    fn build(&mut self, directory: &RawDirectory, parent_path: &str) -> usize {
        let mut candidates: Vec<Segment> = directory
            .directories
            .keys()
            .map(|segment| Segment {
                base: segment.clone(),
                kind: EntryKind::Dir,
                key: segment.clone(),
            })
            .collect();
        candidates.extend(directory.skills.iter().map(|worker_id| Segment {
            base: worker_id.clone(),
            kind: EntryKind::Skill,
            key: worker_id.clone(),
        }));
        candidates.sort_by(segment_order);

        let mut used: BTreeSet<String> = BTreeSet::new();
        let mut resolved: Vec<(String, Segment)> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let segment = Self::unique_segment(&candidate.base, &mut used);
            resolved.push((segment, candidate));
        }
        resolved.sort_by(|left, right| segment_order_of(&left.0, &left.1, &right.0, &right.1));

        let mut entries: Vec<DirectoryEntry> = Vec::with_capacity(resolved.len());
        let mut count = 0;
        for (segment, candidate) in resolved {
            let path = join_path(parent_path, &segment);
            if candidate.kind == EntryKind::Skill {
                let worker_id = candidate.key;
                let (label, description) = self.records.get(&worker_id).map_or_else(
                    || (worker_id.clone(), String::new()),
                    |record| {
                        let label = if record.name.trim().is_empty() {
                            worker_id.clone()
                        } else {
                            record.name.clone()
                        };
                        (label, record.description.clone())
                    },
                );
                self.record_paths.insert(worker_id.clone(), path.clone());
                self.meta_index
                    .insert(format!("{path}/{METADATA_FILENAME}"), worker_id.clone());
                entries.push(DirectoryEntry {
                    path,
                    kind: EntryKind::Skill,
                    label,
                    description,
                    worker_id,
                    depth: 0,
                });
                count += 1;
                continue;
            }
            let child = directory.directories.get(&candidate.key);
            let label = category_label(&candidate.base);
            self.nodes.insert(
                path.clone(),
                DirectoryNode::new(path.clone(), label.clone(), ""),
            );
            entries.push(DirectoryEntry {
                path: path.clone(),
                kind: EntryKind::Dir,
                label,
                description: String::new(),
                worker_id: String::new(),
                depth: 0,
            });
            if let Some(child) = child {
                count += self.build(child, &path);
            }
        }
        self.children.insert(parent_path.to_string(), entries);
        self.skill_counts.insert(parent_path.to_string(), count);
        count
    }

    /// Returns a segment that no sibling in `used` already owns.
    fn unique_segment(base: &str, used: &mut BTreeSet<String>) -> String {
        if used.insert(base.to_lowercase()) {
            return base.to_string();
        }
        let mut index = 2usize;
        loop {
            let candidate = format!("{base}{DISAMBIGUATION_MARKER}{index}");
            if used.insert(candidate.to_lowercase()) {
                return candidate;
            }
            index += 1;
        }
    }

    /// Implements Python's `_scoped_entries`.
    fn scoped(
        &self,
        paths: &[String],
        max_depth: Option<usize>,
        include_root: bool,
        recursive: Option<bool>,
        directories_only: bool,
        allow_metadata: bool,
    ) -> Result<Vec<DirectoryEntry>> {
        let mut rows: Vec<DirectoryEntry> = Vec::new();
        let mut seen: BTreeSet<(EntryKind, String)> = BTreeSet::new();
        for raw in paths {
            let path = Self::normalize_path(raw)?;
            let mut candidates: Vec<DirectoryEntry> = Vec::new();
            let worker_id = if allow_metadata {
                self.meta_index.get(&path)
            } else {
                None
            };
            if let Some(worker_id) = worker_id {
                let entry_path = self
                    .record_paths
                    .get(worker_id)
                    .cloned()
                    .unwrap_or_else(|| path.clone());
                let mut entry = DirectoryEntry::skill(entry_path, worker_id.clone());
                if let Some(record) = self.records.get(worker_id) {
                    entry.label = if record.name.trim().is_empty() {
                        worker_id.clone()
                    } else {
                        record.name.clone()
                    };
                    entry.description.clone_from(&record.description);
                }
                candidates.push(entry);
            } else {
                let Some(node) = self.nodes.get(&path) else {
                    return Err(Error::validation(format!(
                        "No such Skill directory: {path}"
                    )));
                };
                if include_root {
                    let description = if !node.description.is_empty() {
                        node.description.clone()
                    } else if recursive == Some(true) && path == ROOT_PATH {
                        self.root_description.clone()
                    } else {
                        String::new()
                    };
                    candidates.push(DirectoryEntry {
                        path: path.clone(),
                        kind: EntryKind::Dir,
                        label: node.label.clone(),
                        description,
                        worker_id: String::new(),
                        depth: 0,
                    });
                }
                if let Some(recursive) = recursive {
                    self.walk(&path, recursive, max_depth, &mut candidates);
                }
            }
            for entry in candidates {
                if directories_only && entry.kind != EntryKind::Dir {
                    continue;
                }
                if seen.insert((entry.kind, entry.path.clone())) {
                    rows.push(entry);
                }
            }
        }
        Ok(rows)
    }

    /// Visits the children of `path`, depth-first, tagging each with its relative depth.
    ///
    /// Python recurses (`_walk`); the port keeps the same pre-order — a child row is
    /// emitted before its subtree and siblings stay in tree order — with an explicit
    /// frame stack, so a deep category chain cannot exhaust the call stack.
    #[allow(clippy::while_let_loop)] // the frame stack is mutated inside the body
    fn walk(
        &self,
        path: &str,
        recursive: bool,
        max_depth: Option<usize>,
        rows: &mut Vec<DirectoryEntry>,
    ) {
        let mut frames: Vec<(Vec<DirectoryEntry>, usize, usize)> =
            vec![(self.children.get(path).cloned().unwrap_or_default(), 0, 0)];
        loop {
            let Some((entries, index, depth)) = frames.last_mut() else {
                break;
            };
            if *index >= entries.len() {
                frames.pop();
                continue;
            }
            let entry = entries[*index].clone();
            let relative = *depth + 1;
            *index += 1;
            if let Some(limit) = max_depth {
                if relative > limit {
                    continue;
                }
            }
            rows.push(entry.clone().with_depth(relative));
            if recursive && entry.kind == EntryKind::Dir {
                if let Some(children) = self.children.get(&entry.path) {
                    frames.push((children.clone(), 0, relative));
                }
            }
        }
    }
}

/// Splits a classification into its category segments.
///
/// The value is split on `>` (or on `/` when it has no `>`), each segment is trimmed,
/// blank segments are dropped, and the chain is truncated to [`MAX_CATEGORY_DEPTH`].
#[must_use]
pub fn category_segments(classification: &str) -> Vec<String> {
    let separator = if classification.contains(CATEGORY_SEPARATOR) {
        CATEGORY_SEPARATOR
    } else {
        CATEGORY_PATH_SEPARATOR
    };
    let mut segments: Vec<String> = classification
        .split(separator)
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(ToString::to_string)
        .collect();
    segments.truncate(MAX_CATEGORY_DEPTH);
    segments
}

/// Joins the non-blank `>` segments of `value` into a normalized category chain.
fn normalize_category_chain(value: &str) -> String {
    let parts: Vec<&str> = value
        .split(CATEGORY_SEPARATOR)
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        value.trim().to_string()
    } else {
        parts.join(CATEGORY_LABEL_SEPARATOR)
    }
}

/// Renders one classification segment as a node label.
fn category_label(segment: &str) -> String {
    segment.trim().to_string()
}

/// The last `/` segment of `path`.
fn last_segment(path: &str) -> &str {
    path.rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or("")
}

/// The parent path of `path` (`/a/b` → `/a`, `/a` → `/`, `/` → `/`).
fn parent_path(path: &str) -> String {
    match path.rfind('/') {
        Some(0) | None => ROOT_PATH.to_string(),
        Some(index) => path[..index].to_string(),
    }
}

/// Joins `parent` and `segment` into an absolute path.
fn join_path(parent: &str, segment: &str) -> String {
    if parent == ROOT_PATH {
        format!("/{segment}")
    } else {
        format!("{parent}/{segment}")
    }
}

/// The deterministic sibling order: `(segment.to_lowercase(), segment, kind, key…, key)`.
fn segment_order(left: &Segment, right: &Segment) -> std::cmp::Ordering {
    segment_order_of(&left.base, left, &right.base, right)
}

/// The deterministic order of two resolved candidates.
fn segment_order_of(
    left_segment: &str,
    left: &Segment,
    right_segment: &str,
    right: &Segment,
) -> std::cmp::Ordering {
    left_segment
        .to_lowercase()
        .cmp(&right_segment.to_lowercase())
        .then_with(|| left_segment.cmp(right_segment))
        .then_with(|| left.kind.cmp(&right.kind))
        .then_with(|| left.key.to_lowercase().cmp(&right.key.to_lowercase()))
        .then_with(|| left.key.cmp(&right.key))
}

/// Collapses a category description the way Python's `_searchable_category_description`
/// does: drop blank lines and index-construction statistics, join the rest with spaces.
fn searchable_description(value: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for raw_line in value.split('\n') {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let lowered = line.to_lowercase();
        if lowered.starts_with("covers ")
            || lowered.starts_with("representative ")
            || lowered.starts_with("don't select when:")
        {
            continue;
        }
        lines.push(line);
    }
    lines.join(" ")
}

/// Reports whether `descriptor`'s classification yields any category segment.
#[must_use]
pub fn has_category(descriptor: &CapabilityDescriptor) -> bool {
    !category_segments(&descriptor.classification).is_empty()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn record(worker_id: &str, name: &str) -> SkillRecord {
        SkillRecord::new(worker_id, name).with_description(format!("{name} description"))
    }

    fn view(pairs: &[(&str, &str)]) -> SkillDirectoryView {
        let records: Vec<SkillRecord> = pairs
            .iter()
            .map(|(worker_id, _)| record(worker_id, &worker_id.to_uppercase()))
            .collect();
        let categories: BTreeMap<String, String> = pairs
            .iter()
            .map(|(worker_id, category)| ((*worker_id).to_string(), (*category).to_string()))
            .collect();
        SkillDirectoryView::from_records(&records, &categories)
    }

    #[test]
    fn normalize_path_collapses_and_rejects() {
        assert_eq!(SkillDirectoryView::normalize_path("/").unwrap(), "/");
        assert_eq!(SkillDirectoryView::normalize_path("").unwrap(), "/");
        assert_eq!(
            SkillDirectoryView::normalize_path("//a//b/").unwrap(),
            "/a/b"
        );
        assert_eq!(
            SkillDirectoryView::normalize_path("/a/./b").unwrap(),
            "/a/b"
        );
        assert_eq!(SkillDirectoryView::normalize_path("ROOT").unwrap(), "/ROOT");
        assert_eq!(
            SkillDirectoryView::normalize_path("/a/../b")
                .unwrap_err()
                .kind(),
            crate::error::ErrorKind::Validation
        );
    }

    #[test]
    fn category_segments_split_trim_and_truncate() {
        use std::fmt::Write as _;

        assert_eq!(category_segments("Data > Export"), ["Data", "Export"]);
        assert_eq!(category_segments("Data/Export"), ["Data", "Export"]);
        assert_eq!(
            category_segments("  > Data >  > Export >  "),
            ["Data", "Export"]
        );
        assert!(category_segments("   ").is_empty());
        let mut deep = String::new();
        for index in 0..40 {
            write!(deep, "L{index} > ").unwrap();
        }
        assert_eq!(category_segments(&deep).len(), MAX_CATEGORY_DEPTH);
    }

    #[test]
    fn collisions_are_disambiguated_in_deterministic_order() {
        let view = view(&[
            ("notes", ""),
            ("note-taker", "notes"),
            ("docs-a", "Docs"),
            ("docs-b", "docs"),
        ]);
        let mut children: Vec<(String, EntryKind)> = view
            .children("/")
            .unwrap()
            .into_iter()
            .map(|entry| (entry.path, entry.kind))
            .collect();
        children.sort();
        assert!(children.contains(&("/notes".to_string(), EntryKind::Dir)));
        assert!(children.contains(&("/notes~2".to_string(), EntryKind::Skill)));
        assert!(children.contains(&("/Docs".to_string(), EntryKind::Dir)));
        assert!(children.contains(&("/docs~2".to_string(), EntryKind::Dir)));
    }

    #[test]
    fn skill_counts_include_descendants() {
        let view = view(&[
            ("a", "Data > Export"),
            ("b", "Data > Export"),
            ("c", "Data"),
            ("d", ""),
        ]);
        assert_eq!(view.skill_count("/").unwrap(), 4);
        assert_eq!(view.skill_count("/Data").unwrap(), 3);
        assert_eq!(view.skill_count("/Data/Export").unwrap(), 2);
        assert!(view.skill_count("/missing").is_err());
    }

    #[test]
    fn walking_stays_in_tree_order() {
        let view = view(&[("a", "Data > Export"), ("b", "Data"), ("c", "")]);
        let paths: Vec<String> = view
            .tree_entries(&["/".to_string()], None, false)
            .unwrap()
            .into_iter()
            .map(|entry| format!("{}:{}", entry.path, entry.depth))
            .collect();
        assert_eq!(
            paths,
            [
                "/:0",
                "/c:1",
                "/Data:1",
                "/Data/b:2",
                "/Data/Export:2",
                "/Data/Export/a:3"
            ]
        );
    }

    #[test]
    fn scoped_entries_resolve_metadata_paths() {
        let view = view(&[("a", "Data")]);
        let metadata = view.metadata_path("a").unwrap();
        assert_eq!(metadata, "/Data/a/META.md");
        let entries = view
            .searchable_entries(std::slice::from_ref(&metadata), None)
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, EntryKind::Skill);
        assert_eq!(
            view.resolve_metadata_path(&metadata).unwrap().worker_id,
            "a"
        );
        assert!(view.resolve_metadata_path("/nope/META.md").is_err());
        let records = view.scoped_records(&["/".to_string()], None).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn categories_render_from_labels() {
        let view = view(&[("a", "Data > Export")]);
        assert_eq!(view.category_from_path("/"), "ROOT");
        assert_eq!(view.category_from_path("/Data"), "Data");
        assert_eq!(view.category_from_path("/Data/Export"), "Data > Export");
        assert_eq!(view.skill_category("a").unwrap(), "Data > Export");
        assert_eq!(view.category_text("a"), "Data\nExport");
        assert_eq!(
            view.skill_category("missing").unwrap_err().kind(),
            crate::error::ErrorKind::Validation
        );
    }

    #[test]
    fn category_paths_accept_root_chain_and_label_lookups() {
        let view = view(&[("a", "Data > Export"), ("b", "Language")]);
        assert_eq!(view.category_path("/").unwrap(), "/");
        assert_eq!(view.category_path("ROOT").unwrap(), "/");
        assert_eq!(view.category_path("Data > Export").unwrap(), "/Data/Export");
        assert_eq!(view.category_path("Export").unwrap(), "/Data/Export");
        assert_eq!(view.category_path("language").unwrap(), "/Language");
        assert!(view.category_path("Missing").is_err());
    }
}
