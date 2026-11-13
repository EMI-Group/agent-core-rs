//! The `SkillFs` skill catalog: records, bodies, and lexical skill search.
//!
//! Ports Python `openjiuwen.symphony.discovery`'s host-side installed-skill catalog
//! (`SkillFS` plus its BM25 content search), rebuilt on the crate's own retrieval
//! pieces: every discovered skill becomes a [`CapabilityDescriptor`] whose
//! `semantic_content` is the skill's markdown body, that catalog is indexed by
//! [`LexicalIndex`], and hits are mapped back to [`SkillRecord`]s.
//!
//! Search is deterministic: hits rank by BM25 score and break ties on `worker_id`; an
//! empty (or all-punctuation) query is a filter-only request returning every record in
//! `worker_id` order with score `0.0` and an empty excerpt; `top_k == 0` means
//! unlimited.
//!
//! ```
//! # use openjiuwen_symphony::discovery::SkillFs;
//! # async fn example() -> openjiuwen_symphony::Result<()> {
//! let catalog = SkillFs::open("skills").await?;
//! let hits = catalog.search("pdf", 5)?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::discovery::models::{SkillInventory, SkillRecord};
use crate::discovery::scanner::{
    ScannedSkill, SkillDiagnostic, SkillFolderScanner, SkillScanConfig,
};
use crate::error::Result;
use crate::model::CapabilityDescriptor;
use crate::search::lexical::{tokenize, LexicalIndex};

/// The characters of context an excerpt shows.
pub const EXCERPT_WIDTH: usize = 160;
/// The characters of context kept before the matched term in an excerpt.
pub const EXCERPT_LEAD: usize = 60;

/// One lexical skill hit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillHit {
    /// The skill's worker id.
    pub worker_id: String,
    /// The skill's display name.
    pub name: String,
    /// The skill's description.
    pub description: String,
    /// The BM25 score; higher is more relevant, `0.0` for a filter-only request.
    pub score: f32,
    /// The distinct query terms the skill matched, sorted.
    pub matched_terms: Vec<String>,
    /// A whitespace-collapsed window around the first matched term, with `…` marking
    /// truncated sides; empty when no term matched.
    pub excerpt: String,
}

/// The compact view of one skill used to build a selection prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SelectionCard {
    /// The skill's worker id.
    pub worker_id: String,
    /// The skill's display name.
    pub name: String,
    /// The skill's description.
    pub description: String,
    /// The skill's tags.
    pub tags: Vec<String>,
    /// The skill's version; empty when unset.
    pub version: String,
}

/// The host-side installed-skill catalog.
///
/// Opening the catalog scans the root once; the resulting records, bodies, and BM25
/// index are immutable afterwards, so searches are deterministic and cheap.
#[derive(Debug, Clone)]
pub struct SkillFs {
    /// The scan root.
    root: PathBuf,
    /// The config the catalog was built with.
    config: SkillScanConfig,
    /// The records, sorted by worker id.
    records: Vec<SkillRecord>,
    /// The descriptors, in the same order as `records`.
    descriptors: Vec<CapabilityDescriptor>,
    /// The markdown bodies, keyed by worker id.
    bodies: BTreeMap<String, String>,
    /// The diagnostics the scan produced.
    diagnostics: Vec<SkillDiagnostic>,
    /// The BM25 index over the descriptors (body as retrieval content).
    index: LexicalIndex,
    /// The fingerprinted inventory.
    inventory: SkillInventory,
}

impl SkillFs {
    /// Opens the catalog at `root` with the default (immediate-child) scan config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) naming
    /// `root` when the root does not exist or is not a directory.
    pub async fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(SkillScanConfig::default(), root).await
    }

    /// Opens the catalog at `root` with an explicit scan config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) naming
    /// `root` when the root does not exist or is not a directory.
    pub async fn open_with(config: SkillScanConfig, root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let scanner = SkillFolderScanner::with_config(root.clone(), config.clone());
        let discovery = scanner.discover().await?;
        for diagnostic in &discovery.diagnostics {
            match diagnostic.severity {
                crate::discovery::scanner::DiagnosticSeverity::Warning => {
                    tracing::warn!(code = %diagnostic.code, path = %diagnostic.path, "{}", diagnostic.message);
                }
                crate::discovery::scanner::DiagnosticSeverity::Error => {
                    tracing::error!(code = %diagnostic.code, path = %diagnostic.path, "{}", diagnostic.message);
                }
            }
        }

        let source_root = root.to_string_lossy().to_string();
        let skills: Vec<ScannedSkill> = discovery.skills;
        let records: Vec<SkillRecord> = skills
            .iter()
            .map(|skill| record_for(&skill.descriptor, &source_root))
            .collect();
        let descriptors: Vec<CapabilityDescriptor> = skills
            .iter()
            .map(|skill| skill.descriptor.clone())
            .collect();
        let mut bodies: BTreeMap<String, String> = BTreeMap::new();
        for skill in &skills {
            bodies.insert(skill.descriptor.capability_id.clone(), skill.body.clone());
        }
        let index = LexicalIndex::build(skills.iter().map(|skill| {
            skill
                .descriptor
                .clone()
                .with_semantic_content(skill.body.clone())
        }));

        Ok(Self {
            root,
            config,
            inventory: SkillInventory::new(records.clone()),
            records,
            descriptors,
            bodies,
            diagnostics: discovery.diagnostics,
            index,
        })
    }

    /// The scan root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The scan config the catalog was built with.
    #[must_use]
    pub fn config(&self) -> &SkillScanConfig {
        &self.config
    }

    /// The fingerprinted inventory of every discovered skill.
    #[must_use]
    pub fn inventory(&self) -> &SkillInventory {
        &self.inventory
    }

    /// The skill records, sorted by worker id.
    #[must_use]
    pub fn records(&self) -> &[SkillRecord] {
        &self.records
    }

    /// The diagnostics the scan produced.
    #[must_use]
    pub fn diagnostics(&self) -> &[SkillDiagnostic] {
        &self.diagnostics
    }

    /// The markdown body of the skill with `worker_id`, when it is known.
    #[must_use]
    pub fn read_body(&self, worker_id: &str) -> Option<&str> {
        self.bodies.get(worker_id).map(String::as_str)
    }

    /// The compact selection cards, sorted by worker id.
    #[must_use]
    pub fn selection_cards(&self) -> Vec<SelectionCard> {
        self.records
            .iter()
            .zip(self.descriptors.iter())
            .map(|(record, descriptor)| SelectionCard {
                worker_id: record.worker_id.clone(),
                name: record.name.clone(),
                description: record.description.clone(),
                tags: descriptor.tags.clone(),
                version: record.version.clone(),
            })
            .collect()
    }

    /// Searches the catalog for `query`, returning at most `top_k` hits.
    ///
    /// An empty query (or one with no alphanumeric tokens) is a filter-only request:
    /// every record is returned in `worker_id` order with score `0.0`, no matched
    /// terms, and an empty excerpt. `top_k == 0` means unlimited.
    ///
    /// # Errors
    ///
    /// Reserved for future index validation; the lexical index has no failing path.
    #[allow(clippy::unnecessary_wraps)]
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SkillHit>> {
        if tokenize(query).is_empty() {
            let mut hits: Vec<SkillHit> = self
                .records
                .iter()
                .map(|record| SkillHit {
                    worker_id: record.worker_id.clone(),
                    name: record.name.clone(),
                    description: record.description.clone(),
                    score: 0.0,
                    matched_terms: Vec::new(),
                    excerpt: String::new(),
                })
                .collect();
            hits.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
            if top_k > 0 {
                hits.truncate(top_k);
            }
            return Ok(hits);
        }

        let mut hits: Vec<SkillHit> = self
            .index
            .search(query, 0)
            .into_iter()
            .filter_map(|hit| {
                let worker_id = hit
                    .qualified_id
                    .split_once(':')
                    .map_or(hit.qualified_id.as_str(), |(_, id)| id);
                let record = self
                    .records
                    .iter()
                    .find(|record| record.worker_id == worker_id)?;
                let excerpt = self.excerpt_for(record, &hit.matched_terms);
                Some(SkillHit {
                    worker_id: record.worker_id.clone(),
                    name: record.name.clone(),
                    description: record.description.clone(),
                    score: hit.score,
                    matched_terms: hit.matched_terms,
                    excerpt,
                })
            })
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.worker_id.cmp(&right.worker_id))
        });
        if top_k > 0 {
            hits.truncate(top_k);
        }
        Ok(hits)
    }

    /// Builds the excerpt for one hit: the best of the description and the body.
    fn excerpt_for(&self, record: &SkillRecord, matched_terms: &[String]) -> String {
        let description = collapse_whitespace(&record.description);
        let body = self
            .bodies
            .get(&record.worker_id)
            .map(|body| collapse_whitespace(body))
            .unwrap_or_default();
        let description_matches = matched_term_count(&description, matched_terms);
        let body_matches = matched_term_count(&body, matched_terms);
        let text = if body_matches > description_matches {
            &body
        } else {
            &description
        };
        excerpt_around(text, matched_terms)
    }
}

/// Builds the host record for one discovered skill descriptor.
fn record_for(descriptor: &CapabilityDescriptor, source_root: &str) -> SkillRecord {
    let metadata_string = |key: &str| {
        descriptor
            .metadata
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let aliases: Vec<String> = descriptor
        .metadata
        .get("aliases")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default();
    SkillRecord::new(descriptor.capability_id.clone(), descriptor.name.clone())
        .with_description(descriptor.description.clone())
        .with_skill_file(metadata_string("entrypoint"))
        .with_source(descriptor.source.clone())
        .with_version(metadata_string("version"))
        .with_author(metadata_string("author"))
        .with_content_hash(descriptor.content_hash.clone())
        .with_source_root(source_root)
        .with_aliases(aliases)
}

/// Collapses runs of whitespace into single spaces.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Counts how many of `terms` occur in `text`, ignoring case.
fn matched_term_count(text: &str, terms: &[String]) -> usize {
    if text.is_empty() {
        return 0;
    }
    let lowered = text.to_lowercase();
    terms
        .iter()
        .filter(|term| !term.is_empty() && lowered.contains(&term.to_lowercase()))
        .count()
}

/// Builds a `…`-delimited window around the earliest matched term of `text`.
fn excerpt_around(text: &str, terms: &[String]) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut earliest: Option<(usize, usize)> = None;
    for term in terms {
        if let Some(position) = find_ignoring_case(&characters, term) {
            if earliest.is_none_or(|(start, _)| position < start) {
                earliest = Some((position, term.chars().count()));
            }
        }
    }
    let Some((position, term_length)) = earliest else {
        return String::new();
    };

    let total = characters.len();
    let mut start = position.saturating_sub(EXCERPT_LEAD);
    if start + EXCERPT_WIDTH > total {
        start = total.saturating_sub(EXCERPT_WIDTH);
    }
    let end = (start + EXCERPT_WIDTH)
        .max(position + term_length)
        .min(total);
    let snippet: String = characters[start..end].iter().collect();
    let snippet = snippet.trim();

    let mut excerpt = String::new();
    if start > 0 {
        excerpt.push('…');
    }
    excerpt.push_str(snippet);
    if end < total {
        excerpt.push('…');
    }
    excerpt
}

/// Finds the first occurrence of `term` in `characters`, ignoring case.
fn find_ignoring_case(characters: &[char], term: &str) -> Option<usize> {
    let needle: Vec<char> = term.chars().collect();
    if needle.is_empty() || needle.len() > characters.len() {
        return None;
    }
    (0..=characters.len() - needle.len()).find(|&start| {
        needle.iter().enumerate().all(|(offset, expected)| {
            characters[start + offset]
                .to_lowercase()
                .eq(expected.to_lowercase())
        })
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn excerpts_window_around_the_first_term() {
        let text = format!("{} needle {}", "a ".repeat(200), "b ".repeat(200));
        let excerpt = excerpt_around(&text, &["needle".to_string()]);
        assert!(excerpt.contains("needle"));
        assert!(excerpt.starts_with('…'));
        assert!(excerpt.ends_with('…'));
        assert!(excerpt.chars().count() <= EXCERPT_WIDTH + 2);
    }

    #[test]
    fn excerpts_are_empty_without_a_match() {
        assert_eq!(excerpt_around("some text", &["absent".to_string()]), "");
        assert_eq!(excerpt_around("", &["absent".to_string()]), "");
        assert_eq!(excerpt_around("some text", &[]), "");
    }

    #[test]
    fn short_text_has_no_ellipses() {
        assert_eq!(
            excerpt_around("a short text", &["short".to_string()]),
            "a short text"
        );
    }

    #[test]
    fn matching_ignores_case_and_collapses_whitespace() {
        assert_eq!(collapse_whitespace("  a \n b\tc "), "a b c");
        assert_eq!(
            find_ignoring_case(&"PDF export".chars().collect::<Vec<_>>(), "pdf"),
            Some(0)
        );
        assert_eq!(
            matched_term_count("PDF export", &["pdf".to_string(), "absent".to_string()]),
            1
        );
    }
}
