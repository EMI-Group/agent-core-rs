//! Category names and paths: the chain of a path or a Skill, and the lookups a caller resolves.

use std::collections::BTreeSet;

use crate::discovery::directory_toolkit::render::inline_text;
use crate::error::{Error, Result};
use crate::model::CapabilityDescriptor;

use super::{
    SkillDirectoryView, CATEGORY_LABEL_SEPARATOR, CATEGORY_PATH_SEPARATOR, CATEGORY_SEPARATOR,
    MAX_CATEGORY_DEPTH, ROOT_LABEL, ROOT_PATH,
};

impl SkillDirectoryView {
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
