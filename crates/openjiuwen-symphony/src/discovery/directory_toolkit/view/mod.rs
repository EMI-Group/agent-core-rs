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
//!   [`matcher`](crate::discovery::directory_toolkit::matcher).
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

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::discovery::models::SkillRecord;

pub mod category;
pub mod entries;
pub mod paths;
pub mod tree;

pub use category::{category_segments, has_category};

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
