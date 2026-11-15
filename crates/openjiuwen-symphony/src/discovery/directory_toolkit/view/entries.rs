//! Entry views: the scoped entries a `list` renders and the records a `search` walks.

use std::collections::BTreeSet;

use crate::discovery::models::SkillRecord;
use crate::error::{Error, Result};

use super::{DirectoryEntry, EntryKind, SkillDirectoryView, ROOT_PATH};

impl SkillDirectoryView {
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
