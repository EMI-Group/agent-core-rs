//! Tree construction: the raw category chains, their deterministic disambiguation, and the
//! children/skill-count indexes.

use std::collections::{BTreeMap, BTreeSet};

use crate::discovery::models::SkillRecord;
use crate::discovery::SkillFs;

use super::category::category_segments;
use super::{
    DirectoryEntry, DirectoryNode, EntryKind, RawDirectory, Segment, SkillDirectoryView,
    DISAMBIGUATION_MARKER, METADATA_FILENAME, ROOT_DESCRIPTION, ROOT_LABEL, ROOT_PATH,
};

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
}

/// Renders one classification segment as a node label.
fn category_label(segment: &str) -> String {
    segment.trim().to_string()
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
