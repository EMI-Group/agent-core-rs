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

use crate::discovery::models::{inventory_fingerprint, SkillInventory, SkillRecord};
use crate::discovery::notice::SkillNoticeState;
use crate::discovery::scanner::{
    ScannedSkill, SkillDiagnostic, SkillFolderScanner, SkillScanConfig,
};
use crate::discovery::settings::DEFAULT_MAX_OUTPUT_CHARS;
use crate::error::Result;
use crate::model::CapabilityDescriptor;
use crate::search::lexical::{tokenize, LexicalIndex};

/// The characters of context an excerpt shows.
pub const EXCERPT_WIDTH: usize = 160;
/// The characters of context kept before the matched term in an excerpt.
pub const EXCERPT_LEAD: usize = 60;

/// The default character budget of a prompt snapshot.
///
/// Mirrors Python's `DiscoverySettings.max_output_chars` default (`12_000` characters).
pub const DEFAULT_PROMPT_MAX_CHARS: usize = DEFAULT_MAX_OUTPUT_CHARS;

/// The prompt-snapshot header line, matching Python's markdown skill-section header.
pub const PROMPT_HEADER: &str = "# Skills";

/// The text a prompt snapshot renders when no skill is in scope.
///
/// This is the exact fallback Python's `# Skills` section uses when no skill is
/// available, so an empty catalog never renders a blank snapshot.
pub const PROMPT_EMPTY_TEXT: &str = "# Skills\n\nNo skills are available for this task.";

/// The category a skill without a front-matter `category` is grouped under.
pub const UNCATEGORIZED_CATEGORY: &str = "Uncategorized";

/// The marker appended when the character budget leaves part of the catalog out.
pub const PROMPT_TRUNCATION_MARKER: &str = "[truncated]";

/// The shortest partial skill line a truncated snapshot keeps.
const MIN_TRUNCATED_LINE_CHARS: usize = 8;

/// Options for [`SkillFs::prompt_snapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptSnapshotOptions {
    /// The maximum number of characters the rendered snapshot may use.
    ///
    /// `0` means no character budget for skill lines; the header and the truncation
    /// marker are still rendered so the snapshot is never blank.
    pub max_chars: usize,
    /// Whether skills are grouped under `## <category>` section headings.
    pub include_categories: bool,
    /// Restricts the snapshot to one worker id; `None` renders the whole catalog.
    pub worker_id: Option<String>,
}

impl Default for PromptSnapshotOptions {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_PROMPT_MAX_CHARS,
            include_categories: false,
            worker_id: None,
        }
    }
}

impl PromptSnapshotOptions {
    /// Builds options with `max_chars` and every other field defaulted.
    #[must_use]
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            ..Self::default()
        }
    }

    /// Enables or disables category section headings.
    #[must_use]
    pub fn with_categories(mut self, include_categories: bool) -> Self {
        self.include_categories = include_categories;
        self
    }

    /// Restricts the snapshot to one worker id.
    #[must_use]
    pub fn with_worker_id(mut self, worker_id: impl Into<String>) -> Self {
        self.worker_id = Some(worker_id.into());
        self
    }
}

/// The rendered model-facing view of the visible skill catalog.
///
/// `included` lists the worker ids whose lines appear in `rendered`, in render order;
/// `omitted` lists the in-scope worker ids the character budget left out, in
/// `(worker_id.casefold(), worker_id)` order. `fingerprint` is
/// [`inventory_fingerprint`] over the included records, so equal included content
/// yields the same digest regardless of render order or category layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptSnapshot {
    /// The rendered text, starting with [`PROMPT_HEADER`] and, when truncated, ending
    /// with [`PROMPT_TRUNCATION_MARKER`].
    pub rendered: String,
    /// The worker ids whose lines were rendered, in render order.
    pub included: Vec<String>,
    /// The in-scope worker ids omitted by the character budget, sorted by name.
    pub omitted: Vec<String>,
    /// Whether the character budget left at least one in-scope skill out.
    pub truncated: bool,
    /// The content-addressed fingerprint of the included skills.
    pub fingerprint: String,
}

impl Default for PromptSnapshot {
    fn default() -> Self {
        Self {
            rendered: PROMPT_EMPTY_TEXT.to_string(),
            included: Vec::new(),
            omitted: Vec::new(),
            truncated: false,
            fingerprint: inventory_fingerprint(&[]),
        }
    }
}

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

    /// The capability descriptors the scan derived, in the same order as [`Self::records`].
    ///
    /// The directory view reads each skill's front-matter `category` from
    /// [`CapabilityDescriptor::classification`](crate::model::CapabilityDescriptor::classification).
    #[must_use]
    pub fn descriptors(&self) -> &[CapabilityDescriptor] {
        &self.descriptors
    }

    /// The capability descriptor of the skill with `worker_id`, when it is known.
    #[must_use]
    pub fn descriptor(&self, worker_id: &str) -> Option<&CapabilityDescriptor> {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.capability_id == worker_id)
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

    /// The worker ids of the enabled, discovered skills.
    ///
    /// The order is Python's prompt order: `(worker_id.casefold(), worker_id)`, so it
    /// is independent of directory iteration order. Skills removed by
    /// [`SkillScanConfig::disabled_skills`] never reach the catalog, so they are
    /// absent here as well.
    #[must_use]
    pub fn visible_skill_names(&self) -> Vec<String> {
        self.prompt_order(None)
            .into_iter()
            .map(|index| self.records[index].worker_id.clone())
            .collect()
    }

    /// Renders the model-facing prompt snapshot of the visible catalog.
    ///
    /// The text always starts with [`PROMPT_HEADER`] and, when no skill is in scope,
    /// is exactly [`PROMPT_EMPTY_TEXT`]. Otherwise it renders one Python-format line
    /// per skill (`- <worker_id>: <whitespace-collapsed description>`), optionally
    /// grouped under `## <category>` headings, and appends
    /// [`PROMPT_TRUNCATION_MARKER`] when `options.max_chars` leaves skills out.
    ///
    /// Rendering never splits a code point: lines are appended whole until the budget
    /// is reached, and the final in-budget line may be shortened to its first
    /// characters plus `...`. `included` is the render order, `omitted` is the
    /// `(worker_id.casefold(), worker_id)` order of in-scope skills that did not fit.
    #[must_use]
    pub fn prompt_snapshot(&self, options: &PromptSnapshotOptions) -> PromptSnapshot {
        let order = self.prompt_order(options.worker_id.as_deref());
        if order.is_empty() {
            return PromptSnapshot::default();
        }

        let sections = self.prompt_sections(&order, options.include_categories);
        let body = render_prompt_body(&sections);
        let header_chars = PROMPT_HEADER.chars().count();
        let full_chars = header_chars + 2 + body.chars().count();
        if options.max_chars > 0 && full_chars <= options.max_chars {
            let included: Vec<String> = sections
                .iter()
                .flat_map(|(_, skills)| skills.iter())
                .map(|(index, _)| self.records[*index].worker_id.clone())
                .collect();
            return PromptSnapshot {
                rendered: format!("{PROMPT_HEADER}\n\n{body}"),
                included,
                omitted: Vec::new(),
                truncated: false,
                fingerprint: self.included_fingerprint(&order),
            };
        }

        let marker_chars = PROMPT_TRUNCATION_MARKER.chars().count();
        let content_budget = options
            .max_chars
            .saturating_sub(header_chars + 2 + 2 + marker_chars);
        let selection = select_prompt_lines(&sections, content_budget);
        let omitted: Vec<String> = order
            .iter()
            .filter(|index| !selection.included.contains(index))
            .map(|index| self.records[*index].worker_id.clone())
            .collect();
        let included_names: Vec<String> = selection
            .included
            .iter()
            .map(|index| self.records[*index].worker_id.clone())
            .collect();
        let fingerprint = self.included_fingerprint(&selection.included);
        let mut rendered = String::from(PROMPT_HEADER);
        if !selection.body.is_empty() {
            rendered.push_str("\n\n");
            rendered.push_str(&selection.body);
        }
        rendered.push_str("\n\n");
        rendered.push_str(PROMPT_TRUNCATION_MARKER);
        PromptSnapshot {
            rendered,
            included: included_names,
            omitted,
            truncated: true,
            fingerprint,
        }
    }

    /// A notification cursor seeded with this catalog as its baseline.
    ///
    /// Callers that need to persist the cursor can serialize the returned state;
    /// [`SkillNoticeState::consume`] then diffs a later inventory against it.
    #[must_use]
    pub fn notice_state(&self) -> SkillNoticeState {
        SkillNoticeState::new(&self.inventory)
    }

    /// The indices of in-scope records in Python's prompt order.
    fn prompt_order(&self, worker_id: Option<&str>) -> Vec<usize> {
        let mut order: Vec<usize> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, record)| worker_id.is_none_or(|id| record.worker_id == id))
            .map(|(index, _)| index)
            .collect();
        order.sort_by_cached_key(|index| {
            (
                self.records[*index].worker_id.to_lowercase(),
                self.records[*index].worker_id.clone(),
            )
        });
        order
    }

    /// Groups the prompt order into optional category sections.
    fn prompt_sections(&self, order: &[usize], include_categories: bool) -> Vec<PromptSection> {
        if !include_categories {
            return vec![(
                None,
                order
                    .iter()
                    .map(|index| (*index, self.prompt_line(*index)))
                    .collect(),
            )];
        }

        let mut grouped: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
        for index in order {
            grouped
                .entry(self.category_for(*index))
                .or_default()
                .push((*index, self.prompt_line(*index)));
        }
        let mut categories: Vec<String> = grouped.keys().cloned().collect();
        categories.sort_by_cached_key(|category| (category.to_lowercase(), category.clone()));
        categories
            .into_iter()
            .map(|category| {
                let heading = format!("## {category}");
                (Some(heading), grouped.remove(&category).unwrap_or_default())
            })
            .collect()
    }

    /// The category label of one record: its descriptor classification or
    /// [`UNCATEGORIZED_CATEGORY`].
    fn category_for(&self, index: usize) -> String {
        let classification = self
            .descriptors
            .get(index)
            .map(|descriptor| descriptor.classification.trim())
            .unwrap_or_default();
        if classification.is_empty() {
            UNCATEGORIZED_CATEGORY.to_string()
        } else {
            classification.to_string()
        }
    }

    /// The Python-format prompt line of one record.
    fn prompt_line(&self, index: usize) -> String {
        let record = &self.records[index];
        format!(
            "- {}: {}",
            record.worker_id,
            collapse_whitespace(&record.description)
        )
    }

    /// The content-addressed fingerprint of the records the snapshot includes.
    fn included_fingerprint(&self, indices: &[usize]) -> String {
        let records: Vec<SkillRecord> = indices
            .iter()
            .map(|index| self.records[*index].clone())
            .collect();
        inventory_fingerprint(&records)
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

/// One prompt section: an optional `## ...` heading and its `(record index, line)`s.
type PromptSection = (Option<String>, Vec<(usize, String)>);

/// Renders every section of a prompt snapshot without a character budget.
fn render_prompt_body(sections: &[PromptSection]) -> String {
    let mut body = String::new();
    for (heading, skills) in sections {
        if let Some(heading) = heading {
            if !body.is_empty() {
                body.push_str("\n\n");
            }
            body.push_str(heading);
        }
        for (position, (_, line)) in skills.iter().enumerate() {
            if !body.is_empty() {
                body.push('\n');
                if position == 0 {
                    body.push('\n');
                }
            }
            body.push_str(line);
        }
    }
    body
}

/// The result of fitting prompt sections into a character budget.
struct PromptSelection {
    /// The rendered body after [`PROMPT_HEADER`].
    body: String,
    /// The record indices whose lines are present, in render order.
    included: Vec<usize>,
}

/// Fits section headings and whole skill lines into `budget` characters.
///
/// A line that does not fit is shortened to `budget` minus its separator and ends in
/// `...` when at least [`MIN_TRUNCATED_LINE_CHARS`] characters remain; otherwise the
/// line and every later line are left out. Category headings are only emitted when at
/// least one of their skills fits, so a truncated snapshot never renders an empty
/// section.
fn select_prompt_lines(sections: &[PromptSection], budget: usize) -> PromptSelection {
    let mut body = String::new();
    let mut body_chars = 0usize;
    let mut included: Vec<usize> = Vec::new();
    'sections: for (heading, skills) in sections {
        for (position, (index, line)) in skills.iter().enumerate() {
            let line_chars = line.chars().count();
            let heading_chars = if position == 0 {
                heading.as_ref().map_or(0, |heading| {
                    let separator = if body.is_empty() { 0 } else { 2 };
                    separator + heading.chars().count()
                })
            } else {
                0
            };
            let projected = body_chars + heading_chars;
            let separator = if position == 0 {
                if projected == 0 {
                    0
                } else {
                    2
                }
            } else {
                1
            };
            if projected + separator + line_chars <= budget {
                append_prompt_line(&mut body, heading.as_deref(), position, line);
                body_chars = projected + separator + line_chars;
                included.push(*index);
                continue;
            }

            let remaining = budget.saturating_sub(projected + separator);
            if let Some(shortened) = shorten_prompt_line(line, remaining) {
                append_prompt_line(&mut body, heading.as_deref(), position, &shortened);
                included.push(*index);
            }
            break 'sections;
        }
    }
    PromptSelection { body, included }
}

/// Appends one selected prompt line, preceded by its heading and separators.
fn append_prompt_line(body: &mut String, heading: Option<&str>, position: usize, line: &str) {
    if position == 0 {
        if let Some(heading) = heading {
            if !body.is_empty() {
                body.push_str("\n\n");
            }
            body.push_str(heading);
        }
    }
    if !body.is_empty() {
        body.push('\n');
        if position == 0 {
            body.push('\n');
        }
    }
    body.push_str(line);
}

/// Shortens one prompt line to at most `remaining` characters, keeping code points
/// intact and ending in `...`; returns `None` when too little room remains.
fn shorten_prompt_line(line: &str, remaining: usize) -> Option<String> {
    if remaining < MIN_TRUNCATED_LINE_CHARS {
        return None;
    }
    let keep = remaining - 3;
    let prefix: String = line.chars().take(keep).collect();
    let trimmed = prefix.trim_end();
    Some(format!("{trimmed}..."))
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
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

    /// Writes one `SKILL.md` under `root/<folder>`.
    fn write_skill(root: &Path, folder: &str, front_matter: &str, body: &str) {
        let directory = root.join(folder);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("SKILL.md"),
            format!("---\n{front_matter}\n---\n\n{body}\n"),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn prompt_snapshot_pins_pythons_rendered_format() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(
            directory.path(),
            "alpha",
            "name: Alpha\ndescription: Adds numbers",
            "Adds numbers",
        );
        write_skill(
            directory.path(),
            "beta",
            "name: Beta\ndescription: Translates text",
            "Translates text",
        );
        let catalog = SkillFs::open(directory.path()).await.unwrap();
        let snapshot = catalog.prompt_snapshot(&PromptSnapshotOptions::default());
        assert_eq!(
            snapshot.rendered,
            "# Skills\n\n- alpha: Adds numbers\n- beta: Translates text"
        );
        assert_eq!(snapshot.included, ["alpha", "beta"]);
        assert!(snapshot.omitted.is_empty());
        assert!(!snapshot.truncated);
        assert_eq!(snapshot.fingerprint, catalog.inventory().fingerprint);
    }

    #[tokio::test]
    async fn prompt_snapshot_pins_category_sections() {
        let directory = tempfile::tempdir().unwrap();
        write_skill(
            directory.path(),
            "alpha",
            "name: Alpha\ndescription: Adds numbers\ncategory: documents",
            "Adds numbers",
        );
        write_skill(
            directory.path(),
            "beta",
            "name: Beta\ndescription: Translates text",
            "Translates text",
        );
        let catalog = SkillFs::open(directory.path()).await.unwrap();
        let options = PromptSnapshotOptions::default().with_categories(true);
        let snapshot = catalog.prompt_snapshot(&options);
        assert_eq!(
            snapshot.rendered,
            "# Skills\n\n\
             ## documents\n\n\
             - alpha: Adds numbers\n\n\
             ## Uncategorized\n\n\
             - beta: Translates text"
        );
    }

    #[tokio::test]
    async fn prompt_snapshot_renders_the_documented_empty_text() {
        let directory = tempfile::tempdir().unwrap();
        let catalog = SkillFs::open(directory.path()).await.unwrap();
        let snapshot = catalog.prompt_snapshot(&PromptSnapshotOptions::default());
        assert_eq!(
            snapshot.rendered,
            "# Skills\n\nNo skills are available for this task."
        );
        assert!(snapshot.included.is_empty());
        assert!(snapshot.omitted.is_empty());
        assert!(!snapshot.truncated);
        assert_eq!(snapshot.fingerprint, inventory_fingerprint(&[]));
    }
}
