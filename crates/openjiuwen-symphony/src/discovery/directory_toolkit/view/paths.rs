//! Path lookups: normalization, nodes, children, skill counts, records and `META.md` paths.

use crate::discovery::models::SkillRecord;
use crate::error::{Error, Result};

use super::{DirectoryEntry, DirectoryNode, SkillDirectoryView, METADATA_FILENAME, ROOT_PATH};

impl SkillDirectoryView {
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
}
