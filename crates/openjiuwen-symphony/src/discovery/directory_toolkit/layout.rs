//! The derived `SkillFs` layout and the index-baseline state a `skill_index` result
//! reports.
//!
//! Python's discovery layer projects a prebuilt retriever index and can report whether that
//! index is `tree`- or `flat`-shaped, and whether it is missing, fresh, or stale. The Rust
//! port has no pinned index artifact, so both values are derived here instead of read from
//! one:
//!
//! - [`SkillFsLayout`] is a pure function of the catalog shape: it looks at how many
//!   directory components each `SkillRecord::skill_file` path has (see
//!   [`layout_for_records`]).
//! - [`IndexState`] compares a caller-supplied [`IndexBaseline`] with the catalog's current
//!   inventory fingerprint.
//!
//! # The state describes the baseline in force *before* the call
//!
//! A [`SkillIndexToolkit`](crate::discovery::directory_toolkit::toolkit::SkillIndexToolkit) records a baseline over its current catalog after every result it
//! produces, so the first result on a fresh toolkit reports [`INDEX_MISSING`] and a second,
//! identical call over the same catalog reports [`INDEX_FRESH`]: the reported state always
//! describes the baseline that was in force when the call started.
//!
//! One toolkit's [`SkillFs`] never mutates. A [`SkillIndexToolkit::reload`](crate::discovery::directory_toolkit::toolkit::SkillIndexToolkit::reload) replaces the
//! whole snapshot, so [`INDEX_STALE`] appears when a caller hands a baseline recorded from
//! one catalog snapshot to a toolkit over a different one — through
//! [`SkillIndexToolkit::with_index_baseline`](crate::discovery::directory_toolkit::toolkit::SkillIndexToolkit::with_index_baseline), [`SkillIndexToolkit::set_index_baseline`](crate::discovery::directory_toolkit::toolkit::SkillIndexToolkit::set_index_baseline), or
//! a baseline persisted from an earlier process.
//!
//! Everything here is offline and deterministic: no clock, no randomness, no hash-map
//! iteration order, and no filesystem access.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::discovery::models::SkillRecord;
use crate::discovery::SkillFs;

/// The layout name for a catalog whose skills are all immediate children of the scan root.
pub const LAYOUT_FLAT: &str = "flat";

/// The layout name for a catalog that nests at least one skill below a category directory.
pub const LAYOUT_TREE: &str = "tree";

/// The index-state name for a toolkit that holds no baseline yet.
pub const INDEX_MISSING: &str = "missing";

/// The index-state name for a baseline recorded over the catalog in hand.
pub const INDEX_FRESH: &str = "fresh";

/// The index-state name for a baseline recorded over a different catalog snapshot.
pub const INDEX_STALE: &str = "stale";

/// The shape of a discovered skill catalog, derived from its records' paths.
///
/// [`Flat`](Self::Flat) means every skill sits in an immediate child folder of the scan
/// root (the default scan mode); [`Tree`](Self::Tree) means at least one skill sits below a
/// nested category directory, as a bounded recursive scan produces. The default is
/// [`Flat`](Self::Flat).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkillFsLayout {
    /// Every skill's `SKILL.md` is one directory component below the scan root.
    #[default]
    Flat,
    /// At least one skill's `SKILL.md` is two or more directory components below the root.
    Tree,
}

impl SkillFsLayout {
    /// The wire name of this layout.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flat => LAYOUT_FLAT,
            Self::Tree => LAYOUT_TREE,
        }
    }

    /// Parses a wire name, ignoring surrounding whitespace and ASCII case.
    ///
    /// Returns `None` for a blank or unknown value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case(LAYOUT_FLAT) {
            Some(Self::Flat)
        } else if trimmed.eq_ignore_ascii_case(LAYOUT_TREE) {
            Some(Self::Tree)
        } else {
            None
        }
    }
}

impl std::fmt::Display for SkillFsLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How a toolkit's index baseline relates to the catalog it is asked about.
///
/// The default is [`Missing`](Self::Missing): a toolkit that was never handed a baseline
/// reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IndexState {
    /// No baseline is in force.
    #[default]
    Missing,
    /// The baseline was recorded over exactly this catalog.
    Fresh,
    /// The baseline was recorded over a different catalog snapshot.
    Stale,
}

impl IndexState {
    /// The wire name of this state.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Missing => INDEX_MISSING,
            Self::Fresh => INDEX_FRESH,
            Self::Stale => INDEX_STALE,
        }
    }

    /// Parses a wire name, ignoring surrounding whitespace and ASCII case.
    ///
    /// Returns `None` for a blank or unknown value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case(INDEX_MISSING) {
            Some(Self::Missing)
        } else if trimmed.eq_ignore_ascii_case(INDEX_FRESH) {
            Some(Self::Fresh)
        } else if trimmed.eq_ignore_ascii_case(INDEX_STALE) {
            Some(Self::Stale)
        } else {
            None
        }
    }
}

impl std::fmt::Display for IndexState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The SHA-256 hex digest of the canonical JSON
/// `{"inventory_fingerprint", "layout", "indexed_skills"}`.
///
/// The three fields are the whole content of an [`IndexBaseline`], so the digest is
/// identical for identical baselines, independent of how the caller built them.
#[must_use]
pub fn index_fingerprint(
    inventory_fingerprint: &str,
    layout: &str,
    indexed_skills: usize,
) -> String {
    let document = json!({
        "inventory_fingerprint": inventory_fingerprint,
        "layout": layout,
        "indexed_skills": indexed_skills,
    });
    let mut hasher = Sha256::new();
    hasher.update(document.to_string().as_bytes());
    format!("{:x}", hasher.finalize())
}

/// A recorded index baseline: the catalog snapshot an index was built over.
///
/// Only [`inventory_fingerprint`](Self::inventory_fingerprint) decides the reported
/// [`IndexState`]; [`layout`](Self::layout),
/// [`indexed_skills`](Self::indexed_skills), and
/// [`index_fingerprint`](Self::index_fingerprint) are informational. A baseline whose skill
/// content changed is therefore stale even when the skill count is unchanged, and a baseline
/// whose digest matches still reports fresh when only the count moved (the digest covers the
/// count, so that cannot happen for baselines built by [`Self::from_catalog`]).
///
/// The [`Default`] baseline is empty and therefore stale for every real catalog; use
/// [`Self::from_catalog`] or
/// [`SkillIndexToolkit::record_index_baseline`](crate::discovery::directory_toolkit::toolkit::SkillIndexToolkit::record_index_baseline)
/// to build one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IndexBaseline {
    /// The [`SkillInventory::fingerprint`](crate::discovery::models::SkillInventory::fingerprint)
    /// the baseline was recorded over.
    pub inventory_fingerprint: String,
    /// The layout name ([`SkillFsLayout::as_str`]) at record time.
    pub layout: String,
    /// How many [`SkillRecord`]s the baseline covers.
    pub indexed_skills: usize,
    /// [`index_fingerprint`] over the three fields above.
    pub index_fingerprint: String,
}

impl IndexBaseline {
    /// Records `catalog`'s current shape as a baseline.
    ///
    /// Deterministic: the inventory fingerprint comes from
    /// [`SkillInventory::fingerprint`](crate::discovery::models::SkillInventory::fingerprint),
    /// the layout from [`skillfs_layout`], and the count from [`SkillFs::records`].
    #[must_use]
    pub fn from_catalog(catalog: &SkillFs) -> Self {
        let inventory_fingerprint = catalog.inventory().fingerprint.clone();
        let layout = skillfs_layout(catalog).as_str().to_string();
        let indexed_skills = catalog.records().len();
        let fingerprint = index_fingerprint(&inventory_fingerprint, &layout, indexed_skills);
        Self {
            inventory_fingerprint,
            layout,
            indexed_skills,
            index_fingerprint: fingerprint,
        }
    }

    /// How this baseline relates to `catalog`; see [`index_state_for`].
    #[must_use]
    pub fn index_state(&self, catalog: &SkillFs) -> IndexState {
        index_state_for(catalog, Some(self))
    }
}

/// The layout derived from `records`.
///
/// Each record's [`skill_file`](SkillRecord::skill_file) is split on `/` and empty segments
/// are dropped; the last segment is the file name, so the directory-component count is one
/// less than the segment count (a blank `skill_file` counts as zero). The result is
/// [`SkillFsLayout::Flat`] when **every** record has at most one directory component — every
/// skill is an immediate child of the scan root, which is what the default scan mode
/// produces even though the folders have different names — and
/// [`SkillFsLayout::Tree`] as soon as **any** record has two or more (a nested category
/// directory such as `Data/pdf/SKILL.md`, as a bounded recursive scan produces). An empty
/// catalog reports [`SkillFsLayout::Flat`]; `\` is never a separator, because the scanner
/// emits `/`.
#[must_use]
pub fn layout_for_records(records: &[SkillRecord]) -> SkillFsLayout {
    let nested = records.iter().any(|record| {
        let segments = record
            .skill_file
            .split('/')
            .filter(|segment| !segment.is_empty())
            .count();
        segments > 2
    });
    if nested {
        SkillFsLayout::Tree
    } else {
        SkillFsLayout::Flat
    }
}

/// The layout of `catalog`; [`layout_for_records`] over [`SkillFs::records`].
#[must_use]
pub fn skillfs_layout(catalog: &SkillFs) -> SkillFsLayout {
    layout_for_records(catalog.records())
}

/// The index state of `catalog` against `baseline`.
///
/// `None` is [`IndexState::Missing`]; a baseline whose `inventory_fingerprint` equals the
/// catalog's current one is [`IndexState::Fresh`]; anything else is [`IndexState::Stale`].
/// Only the inventory fingerprint is compared, so a baseline recorded over a catalog whose
/// skill content changed is stale even when the skill count is unchanged.
#[must_use]
pub fn index_state_for(catalog: &SkillFs, baseline: Option<&IndexBaseline>) -> IndexState {
    match baseline {
        None => IndexState::Missing,
        Some(baseline) => {
            if baseline.inventory_fingerprint == catalog.inventory().fingerprint {
                IndexState::Fresh
            } else {
                IndexState::Stale
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::discovery::directory_toolkit::toolkit::SkillIndexResult;
    use crate::discovery::directory_toolkit::{SkillIndexConfig, SkillIndexToolkit};

    /// Writes a `SKILL.md` at `root/relative`, creating parent directories.
    fn write_skill(root: &Path, relative: &str, name: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let content = format!("---\nname: {name}\ndescription: {name} description\n---\n\nBody.\n");
        fs::write(path, content).unwrap();
    }

    /// A record whose only path information is its relative skill file.
    fn record(worker_id: &str, skill_file: &str) -> SkillRecord {
        SkillRecord::new(worker_id, worker_id).with_skill_file(skill_file)
    }

    // -- layout derivation ---------------------------------------------------

    #[test]
    fn sibling_skill_folders_are_flat() {
        let records = [
            record("alpha", "alpha/SKILL.md"),
            record("beta", "beta/SKILL.md"),
        ];
        assert_eq!(layout_for_records(&records), SkillFsLayout::Flat);
        assert_eq!(layout_for_records(&records).as_str(), LAYOUT_FLAT);
    }

    #[test]
    fn one_nested_skill_makes_the_catalog_a_tree() {
        let records = [
            record("alpha", "alpha/SKILL.md"),
            record("pdf-extract", "Data/pdf/SKILL.md"),
        ];
        assert_eq!(layout_for_records(&records), SkillFsLayout::Tree);
        assert_eq!(layout_for_records(&records).as_str(), LAYOUT_TREE);
    }

    #[test]
    fn an_empty_catalog_is_flat() {
        assert_eq!(layout_for_records(&[]), SkillFsLayout::Flat);
    }

    #[test]
    fn blank_skill_files_and_repeated_separators_do_not_nest() {
        let records = [
            record("blank", ""),
            record("trailing", "alpha/SKILL.md/"),
            record("doubled", "/alpha//SKILL.md"),
        ];
        assert_eq!(layout_for_records(&records), SkillFsLayout::Flat);
        let backslash = [record("windows", "alpha\\beta\\SKILL.md")];
        assert_eq!(layout_for_records(&backslash), SkillFsLayout::Flat);
    }

    // -- enum wire names -----------------------------------------------------

    #[test]
    fn layout_names_round_trip() {
        assert_eq!(SkillFsLayout::default(), SkillFsLayout::Flat);
        assert_eq!(SkillFsLayout::Flat.as_str(), "flat");
        assert_eq!(SkillFsLayout::Tree.as_str(), "tree");
        assert_eq!(SkillFsLayout::Flat.to_string(), "flat");
        assert_eq!(SkillFsLayout::Tree.to_string(), "tree");
        for layout in [SkillFsLayout::Flat, SkillFsLayout::Tree] {
            assert_eq!(SkillFsLayout::parse(layout.as_str()), Some(layout));
            assert_eq!(
                SkillFsLayout::parse(&format!(" {} ", layout.as_str().to_uppercase())),
                Some(layout)
            );
        }
        assert_eq!(SkillFsLayout::parse(""), None);
        assert_eq!(SkillFsLayout::parse("  "), None);
        assert_eq!(SkillFsLayout::parse("nested"), None);
    }

    #[test]
    fn index_state_names_round_trip() {
        assert_eq!(IndexState::default(), IndexState::Missing);
        assert_eq!(IndexState::Missing.as_str(), "missing");
        assert_eq!(IndexState::Fresh.as_str(), "fresh");
        assert_eq!(IndexState::Stale.as_str(), "stale");
        assert_eq!(IndexState::Fresh.to_string(), "fresh");
        assert_eq!(IndexState::Stale.to_string(), "stale");
        for state in [IndexState::Missing, IndexState::Fresh, IndexState::Stale] {
            assert_eq!(IndexState::parse(state.as_str()), Some(state));
            assert_eq!(IndexState::parse(state.as_str()), Some(state));
        }
        assert_eq!(IndexState::parse("FRESH "), Some(IndexState::Fresh));
        assert_eq!(IndexState::parse(""), None);
        assert_eq!(IndexState::parse("expired"), None);
    }

    // -- the index fingerprint ----------------------------------------------

    #[test]
    fn the_index_fingerprint_is_stable_and_content_sensitive() {
        let base = index_fingerprint("abc", LAYOUT_FLAT, 2);
        assert_eq!(base, index_fingerprint("abc", LAYOUT_FLAT, 2));
        assert_eq!(base.len(), 64);
        assert!(base.chars().all(|character| character.is_ascii_hexdigit()));
        assert_ne!(base, index_fingerprint("abd", LAYOUT_FLAT, 2));
        assert_ne!(base, index_fingerprint("abc", LAYOUT_TREE, 2));
        assert_ne!(base, index_fingerprint("abc", LAYOUT_FLAT, 3));
    }

    // -- baselines over a real catalog --------------------------------------

    #[tokio::test]
    async fn a_fresh_toolkit_reports_missing_then_fresh() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(directory.path(), "alpha/SKILL.md", "Alpha");
        write_skill(directory.path(), "beta/SKILL.md", "Beta");
        let catalog = SkillFs::open(directory.path()).await.unwrap();
        let fingerprint = catalog.inventory().fingerprint.clone();
        assert_eq!(skillfs_layout(&catalog), SkillFsLayout::Flat);
        let mut toolkit =
            SkillIndexToolkit::new(catalog, SkillIndexConfig::new("session-state")).unwrap();

        let first = toolkit
            .invoke(&serde_json::json!({"operation": "list"}))
            .unwrap();
        assert_eq!(first.skillfs_layout, LAYOUT_FLAT);
        assert_eq!(first.index_state, INDEX_MISSING);
        assert_eq!(first.inventory_fingerprint, fingerprint);
        assert_eq!(first.index_baseline_fingerprint, None);
        let recorded = toolkit.index_baseline().unwrap().clone();
        assert_eq!(recorded.inventory_fingerprint, fingerprint);
        assert_eq!(recorded.layout, LAYOUT_FLAT);
        assert_eq!(recorded.indexed_skills, 2);

        let second = toolkit
            .invoke(&serde_json::json!({"operation": "list"}))
            .unwrap();
        assert_eq!(second.index_state, INDEX_FRESH);
        assert_eq!(
            second.index_baseline_fingerprint,
            Some(recorded.index_fingerprint.clone())
        );
        assert_eq!(second.skillfs_layout, LAYOUT_FLAT);
        assert_eq!(second.inventory_fingerprint, fingerprint);
        assert_eq!(toolkit.index_state(), IndexState::Fresh);
    }

    #[tokio::test]
    async fn a_baseline_from_an_older_catalog_is_stale() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(directory.path(), "alpha/SKILL.md", "Alpha");
        let first = SkillFs::open(directory.path()).await.unwrap();
        let baseline = IndexBaseline::from_catalog(&first);
        assert_eq!(baseline.indexed_skills, 1);
        assert_eq!(
            first.inventory().fingerprint,
            baseline.inventory_fingerprint
        );

        write_skill(directory.path(), "beta/SKILL.md", "Beta");
        let second = SkillFs::open(directory.path()).await.unwrap();
        assert_ne!(
            second.inventory().fingerprint,
            baseline.inventory_fingerprint
        );

        let mut toolkit = SkillIndexToolkit::new(second, SkillIndexConfig::new("session-stale"))
            .unwrap()
            .with_index_baseline(baseline.clone());
        assert_eq!(toolkit.index_state(), IndexState::Stale);
        assert_eq!(baseline.index_state(toolkit.catalog()), IndexState::Stale);

        let result = toolkit
            .invoke(&serde_json::json!({"operation": "list"}))
            .unwrap();
        assert_eq!(result.index_state, INDEX_STALE);
        assert_eq!(
            result.index_baseline_fingerprint,
            Some(baseline.index_fingerprint.clone())
        );
        assert_eq!(
            result.inventory_fingerprint,
            toolkit.catalog().inventory().fingerprint
        );
        // The call recorded a fresh baseline over the catalog in hand.
        assert_eq!(toolkit.index_state(), IndexState::Fresh);

        toolkit.set_index_baseline(Some(baseline));
        assert_eq!(toolkit.index_state(), IndexState::Stale);
        toolkit.set_index_baseline(None);
        assert_eq!(toolkit.index_state(), IndexState::Missing);
        assert!(toolkit.index_baseline().is_none());
    }

    #[tokio::test]
    async fn recording_a_baseline_is_deterministic() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(directory.path(), "alpha/SKILL.md", "Alpha");
        let catalog = SkillFs::open_with(
            crate::discovery::SkillScanConfig::recursive(None),
            directory.path(),
        )
        .await
        .unwrap();
        let mut toolkit =
            SkillIndexToolkit::new(catalog, SkillIndexConfig::new("session-record")).unwrap();
        let first = toolkit.record_index_baseline();
        let second = toolkit.record_index_baseline();
        assert_eq!(first, second);
        assert_eq!(toolkit.index_baseline(), Some(&second));
        assert_eq!(
            first.index_fingerprint,
            index_fingerprint(
                &first.inventory_fingerprint,
                &first.layout,
                first.indexed_skills
            )
        );
        assert_eq!(IndexBaseline::from_catalog(toolkit.catalog()), first);

        write_skill(directory.path(), "nested/Data/pdf/SKILL.md", "Pdf");
        toolkit.reload().await.unwrap();
        let changed = toolkit.record_index_baseline();
        assert_ne!(changed, first);
        assert_eq!(changed.layout, LAYOUT_TREE);
        assert_eq!(changed.indexed_skills, 2);
        assert_eq!(skillfs_layout(toolkit.catalog()), SkillFsLayout::Tree);
    }

    #[tokio::test]
    async fn a_default_baseline_is_stale_and_none_is_missing() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(directory.path(), "alpha/SKILL.md", "Alpha");
        let catalog = SkillFs::open(directory.path()).await.unwrap();
        assert_eq!(index_state_for(&catalog, None), IndexState::Missing);
        assert_eq!(
            index_state_for(&catalog, Some(&IndexBaseline::default())),
            IndexState::Stale
        );
        let baseline = IndexBaseline::from_catalog(&catalog);
        assert_eq!(
            index_state_for(&catalog, Some(&baseline)),
            IndexState::Fresh
        );
        assert_eq!(baseline.index_state(&catalog), IndexState::Fresh);
    }

    // -- serde ---------------------------------------------------------------

    #[test]
    fn a_baseline_round_trips_through_json_and_defaults_from_an_empty_object() {
        let baseline = IndexBaseline {
            inventory_fingerprint: "abc".to_string(),
            layout: LAYOUT_TREE.to_string(),
            indexed_skills: 7,
            index_fingerprint: index_fingerprint("abc", LAYOUT_TREE, 7),
        };
        let rendered = serde_json::to_string(&baseline).unwrap();
        let parsed: IndexBaseline = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed, baseline);
        assert!(rendered.contains("\"indexed_skills\":7"), "{rendered}");
        assert!(rendered.contains("\"inventory_fingerprint\":\"abc\""));

        let empty: IndexBaseline = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, IndexBaseline::default());
        assert!(empty.inventory_fingerprint.is_empty());
        assert_eq!(empty.indexed_skills, 0);
    }

    #[tokio::test]
    async fn a_result_round_trips_through_json() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(directory.path(), "alpha/SKILL.md", "Alpha");
        let catalog = SkillFs::open(directory.path()).await.unwrap();
        let fingerprint = catalog.inventory().fingerprint.clone();
        let mut toolkit =
            SkillIndexToolkit::new(catalog, SkillIndexConfig::new("session-serde")).unwrap();
        toolkit
            .invoke(&serde_json::json!({"operation": "list"}))
            .unwrap();
        let result = toolkit
            .invoke(&serde_json::json!({"operation": "list"}))
            .unwrap();

        let rendered = serde_json::to_string(&result).unwrap();
        let parsed: SkillIndexResult = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed.skillfs_layout, result.skillfs_layout);
        assert_eq!(parsed.skillfs_layout, LAYOUT_FLAT);
        assert_eq!(parsed.index_state, INDEX_FRESH);
        assert_eq!(parsed.inventory_fingerprint, fingerprint);
        assert_eq!(
            parsed.index_baseline_fingerprint,
            result.index_baseline_fingerprint
        );
        assert!(parsed.index_baseline_fingerprint.is_some());
        assert_eq!(parsed, result);
    }
}
