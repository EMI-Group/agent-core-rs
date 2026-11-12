//! The Markdown source of truth of the lite coding memory.
//!
//! A [`MarkdownMemory`] is one human-editable document holding every coding fact of a
//! scope, mirroring the shape of Python's `MEMORY.md`: a `---` front-matter block, a
//! title, and one section per fact type. Each fact is an `### name` entry carrying a
//! bullet attribute block (`description`, `tags`, `path`, `updated_at`, plus any
//! unknown keys preserved verbatim) followed by the fact body.
//!
//! ```text
//! ---
//! schema: openjiuwen-memory-lite
//! version: 1
//! ---
//!
//! # Memory
//!
//! ## Project
//!
//! ### build
//!
//! - description: How to build the workspace
//! - tags: ["build"]
//! - updated_at: 2026-11-13T07:00:00Z
//!
//! cargo build
//! ```
//!
//! The renderer output is the only shape the parser is required to round-trip, and it
//! does so byte for byte ([`MarkdownMemory::render`] is idempotent under
//! [`MarkdownMemory::parse`]). Human edits outside that shape never fail the parse:
//! unknown front-matter keys, unknown sections, text inside a known section that is not
//! a fact entry, and non-`key: value` front-matter lines are all preserved verbatim,
//! and the order of fact sections is stable.
//!
//! # Normalization rules
//!
//! - Input is normalized to LF line endings; the renderer always emits LF and exactly
//!   one newline between blocks.
//! - Leading and trailing blank lines of a fact body are dropped, and blank lines
//!   between the `### name` heading, the attribute block, and the body are canonical.
//! - A body line that would otherwise look like a heading is escaped with a backslash
//!   (`\## note`) and unescaped on parse, so fact bodies can contain Markdown headings.
//! - A missing or unparsable `updated_at` attribute decodes to
//!   [`chrono::DateTime::default`] (the Unix epoch) rather than "now", which keeps a
//!   parse deterministic.
//! - `description` and `path` attributes are single-line by construction: embedded
//!   newlines are folded to spaces when rendering.
//! - `tags` are written as a JSON array (a comma-separated value is also accepted when
//!   parsing, so hand-written files stay convenient).
use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::error::Result;

use super::fact::{CodingFact, CodingFactType};
use super::index::type_label;
use crate::record::MemoryScope;

/// Front-matter key naming the document schema.
const FRONT_MATTER_SCHEMA: &str = "schema";
/// Schema marker written into a fresh document's front matter.
const SCHEMA_MARKER: &str = "openjiuwen-memory-lite";
/// Front-matter key holding the document format version.
const FRONT_MATTER_VERSION: &str = "version";
/// Format version written into a fresh document's front matter.
const FORMAT_VERSION: &str = "1";
/// Default document title (the text after `# `).
const DEFAULT_TITLE: &str = "Memory";
/// Prefix of an entry attribute line.
const ATTR_PREFIX: &str = "- ";
/// Entry attribute holding the one-line description.
const ATTR_DESCRIPTION: &str = "description";
/// Entry attribute holding the tag list.
const ATTR_TAGS: &str = "tags";
/// Entry attribute holding the source path.
const ATTR_PATH: &str = "path";
/// Entry attribute holding the last modification timestamp.
const ATTR_UPDATED_AT: &str = "updated_at";

/// The ordered `key: value` block at the top of a [`MarkdownMemory`].
///
/// The map preserves insertion (document) order and keeps lines that are not a
/// `key: value` pair verbatim in a raw tail, so an unrecognized or hand-written front
/// matter block round-trips unchanged. Keys and values are stored trimmed, and
/// [`FrontMatter::insert`] keeps the position of an existing key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrontMatter {
    /// Ordered key/value pairs.
    entries: Vec<(String, String)>,
    /// Lines of the block that are not `key: value` pairs.
    raw: Vec<String>,
}

impl FrontMatter {
    /// Builds an empty front matter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the value of `key` (exact, case-sensitive match).
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value.as_str())
    }

    /// Inserts or replaces `key`, keeping the position of an existing key.
    ///
    /// Returns the previous value when the key was already present.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> Option<String> {
        let key = key.into().trim().to_owned();
        let value = value.into().trim().to_owned();
        if let Some((_, existing)) = self
            .entries
            .iter_mut()
            .find(|(candidate, _)| *candidate == key)
        {
            return Some(std::mem::replace(existing, value));
        }
        self.entries.push((key, value));
        None
    }

    /// Removes `key`, returning its previous value.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        let position = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == key)?;
        Some(self.entries.remove(position).1)
    }

    /// Iterates the ordered key/value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }

    /// Number of `key: value` pairs (raw lines are not counted).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the block holds no `key: value` pair and no raw line.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.raw.is_empty()
    }

    /// Parses the lines between the two `---` fences.
    fn parse_lines(lines: &[&str]) -> Self {
        let mut front_matter = Self::default();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            front_matter.parse_line(line);
        }
        front_matter
    }

    /// Parses one `key: value` line, keeping anything else verbatim.
    fn parse_line(&mut self, line: &str) {
        match line.split_once(':') {
            Some((key, value)) if !key.trim().is_empty() => {
                let _previous = self.insert(key.trim(), value.trim());
            }
            _ => self.raw.push(line.trim_end().to_owned()),
        }
    }

    /// Renders the block with its `---` fences, or `None` when it is empty.
    fn render(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut lines = Vec::with_capacity(self.entries.len() + self.raw.len() + 2);
        lines.push("---".to_owned());
        for (key, value) in &self.entries {
            lines.push(format!("{key}: {value}"));
        }
        for raw in &self.raw {
            lines.push(raw.clone());
        }
        lines.push("---".to_owned());
        Some(lines.join("\n"))
    }
}

/// One fact together with the entry attributes this crate does not interpret.
#[derive(Debug, Clone, PartialEq)]
struct FactEntry {
    /// The fact itself (scope is synchronized with the owning document).
    fact: CodingFact,
    /// Ordered unknown attributes, preserved across edits.
    extra: Vec<(String, String)>,
}

/// One top-level block of the document.
#[derive(Debug, Clone, PartialEq)]
enum Block {
    /// A recognized fact-type section.
    Facts {
        /// The fact type the section holds.
        fact_type: CodingFactType,
        /// The entries in document order.
        entries: Vec<FactEntry>,
        /// Section content that is not a fact entry, kept verbatim.
        raw: Vec<String>,
    },
    /// A section (or loose text) this crate does not interpret, kept verbatim.
    Raw { text: String },
}

/// A parsed `MEMORY.md` document: front matter, title, and fact sections.
///
/// The document carries the [`MemoryScope`] its facts belong to (`parse` cannot know
/// the scope, so it defaults to [`MemoryScope::default`] and callers set it with
/// [`MarkdownMemory::with_scope`]). The scope is not written to the file: one file
/// holds one scope.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkdownMemory {
    /// The `---` front matter block.
    front_matter: FrontMatter,
    /// The document title (the text after `# `), when present.
    title: Option<String>,
    /// Top-level blocks in document order.
    blocks: Vec<Block>,
    /// The scope the facts belong to.
    scope: MemoryScope,
}

impl Default for MarkdownMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownMemory {
    /// Builds an empty document: schema front matter, the `# Memory` title, no facts.
    #[must_use]
    pub fn new() -> Self {
        let mut front_matter = FrontMatter::new();
        let _schema = front_matter.insert(FRONT_MATTER_SCHEMA, SCHEMA_MARKER);
        let _version = front_matter.insert(FRONT_MATTER_VERSION, FORMAT_VERSION);
        Self {
            front_matter,
            title: Some(DEFAULT_TITLE.to_owned()),
            blocks: Vec::new(),
            scope: MemoryScope::default(),
        }
    }

    /// Parses a document from Markdown text.
    ///
    /// The parse is total for free-form input: unknown or malformed content is kept as
    /// a raw block instead of failing. Only an input that cannot be a document at all
    /// (a front-matter fence that is opened and never closed) is reported as
    /// [`openjiuwen_common::error::Error::Validation`].
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an unterminated
    /// front-matter block.
    pub fn parse(text: &str) -> Result<Self> {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let lines: Vec<&str> = normalized.split('\n').collect();
        let mut index = 0;
        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }
        let mut front_matter = FrontMatter::new();
        if index < lines.len() && lines[index].trim_end() == "---" {
            let body_start = index + 1;
            let fence = (body_start..lines.len()).find(|line| lines[*line].trim() == "---");
            if let Some(fence) = fence {
                front_matter = FrontMatter::parse_lines(&lines[body_start..fence]);
                index = fence + 1;
            } else if lines[body_start..]
                .iter()
                .any(|line| attribute_of(line).is_some() || line.contains(':'))
            {
                return Err(openjiuwen_common::error::Error::validation(
                    "markdown memory front matter starts with `---` but is never closed",
                ));
            }
        }
        let mut document = Self {
            front_matter,
            title: None,
            blocks: Vec::new(),
            scope: MemoryScope::default(),
        };
        while index < lines.len() {
            let line = lines[index];
            if document.title.is_none() && document.blocks.is_empty() {
                if let Some(title) = title_of(line) {
                    document.title = Some(title.to_owned());
                    index += 1;
                    continue;
                }
            }
            if let Some(fact_type) = section_fact_type(line) {
                let (block, next) = Self::parse_section(&lines, index + 1, fact_type);
                document.blocks.push(block);
                index = next;
                continue;
            }
            let start = index;
            index += 1;
            while index < lines.len() && !is_section_boundary(lines[index]) {
                index += 1;
            }
            let raw = trim_blank_edges(&lines[start..index]);
            if !raw.is_empty() {
                document.blocks.push(Block::Raw { text: raw });
            }
        }
        if document.front_matter.is_empty()
            && document.title.is_none()
            && document.blocks.is_empty()
        {
            return Ok(Self::new());
        }
        Ok(document)
    }

    /// Parses the entries of one recognized section, starting after its heading.
    fn parse_section(lines: &[&str], start: usize, fact_type: CodingFactType) -> (Block, usize) {
        let mut entries = Vec::new();
        let mut raw = Vec::new();
        let mut index = start;
        while index < lines.len() && !is_section_boundary(lines[index]) {
            if let Some(name) = entry_name(lines[index]) {
                let (entry, next) = Self::parse_entry(lines, index + 1, name, fact_type);
                entries.push(entry);
                index = next;
                continue;
            }
            raw.push(lines[index].trim_end().to_owned());
            index += 1;
        }
        (
            Block::Facts {
                fact_type,
                entries,
                raw,
            },
            index,
        )
    }

    /// Parses one entry, starting after its `### name` heading.
    fn parse_entry(
        lines: &[&str],
        start: usize,
        name: &str,
        fact_type: CodingFactType,
    ) -> (FactEntry, usize) {
        let mut index = start;
        let mut attributes: Vec<(String, String)> = Vec::new();
        while index < lines.len()
            && !is_section_boundary(lines[index])
            && entry_name(lines[index]).is_none()
        {
            let trimmed = lines[index].trim();
            if trimmed.is_empty() {
                index += 1;
                continue;
            }
            let Some((key, value)) = attribute_of(lines[index]) else {
                break;
            };
            attributes.push((key, value));
            index += 1;
        }
        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }
        let body_start = index;
        while index < lines.len()
            && !is_section_boundary(lines[index])
            && entry_name(lines[index]).is_none()
        {
            index += 1;
        }
        let mut body: Vec<String> = lines[body_start..index]
            .iter()
            .map(|line| unescape_body_line(line))
            .collect();
        while body.last().is_some_and(|line| line.trim().is_empty()) {
            let _dropped = body.pop();
        }
        let mut fact =
            CodingFact::new(&MemoryScope::default(), name, body.join("\n")).with_type(fact_type);
        fact.updated_at = DateTime::<Utc>::default();
        let mut extra = Vec::new();
        for (key, value) in attributes {
            match key.as_str() {
                ATTR_DESCRIPTION => fact.description = value,
                ATTR_TAGS => fact.tags = parse_tags(&value),
                ATTR_PATH => {
                    if !value.is_empty() {
                        fact.path = Some(value);
                    }
                }
                ATTR_UPDATED_AT => {
                    if let Ok(timestamp) = DateTime::parse_from_rfc3339(&value) {
                        fact.updated_at = timestamp.with_timezone(&Utc);
                    }
                }
                _ => extra.push((key, value)),
            }
        }
        (FactEntry { fact, extra }, index)
    }

    /// Renders the document, ending with exactly one newline.
    #[must_use]
    pub fn render(&self) -> String {
        let mut chunks: Vec<String> = Vec::new();
        if let Some(front_matter) = self.front_matter.render() {
            chunks.push(front_matter);
        }
        if let Some(title) = &self.title {
            if !title.is_empty() {
                chunks.push(format!("# {title}"));
            }
        }
        for block in &self.blocks {
            match block {
                Block::Facts {
                    fact_type,
                    entries,
                    raw,
                } => {
                    let raw_text = trim_blank_edges(raw);
                    if entries.is_empty() && raw_text.is_empty() {
                        continue;
                    }
                    chunks.push(format!("## {}", type_label(*fact_type)));
                    for entry in entries {
                        chunks.push(render_entry(entry));
                    }
                    if !raw_text.is_empty() {
                        chunks.push(raw_text);
                    }
                }
                Block::Raw { text } => {
                    let text = trim_blank_edges(&[text.as_str()]);
                    if !text.is_empty() {
                        chunks.push(text);
                    }
                }
            }
        }
        let mut rendered = chunks.join("\n\n");
        rendered.push('\n');
        rendered
    }

    /// The scope the document's facts belong to.
    #[must_use]
    pub fn scope(&self) -> &MemoryScope {
        &self.scope
    }

    /// Returns the document bound to `scope`.
    #[must_use]
    pub fn with_scope(mut self, scope: MemoryScope) -> Self {
        self.set_scope(scope);
        self
    }

    /// Rebinds every fact of the document to `scope`.
    pub fn set_scope(&mut self, scope: MemoryScope) {
        for fact in self.facts_mut() {
            fact.scope = scope.clone();
        }
        self.scope = scope;
    }

    /// The document's front matter.
    #[must_use]
    pub fn front_matter(&self) -> &FrontMatter {
        &self.front_matter
    }

    /// Returns the document with `key` set in its front matter.
    #[must_use]
    pub fn with_front_matter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let _previous = self.front_matter.insert(key, value);
        self
    }

    /// The document title (the text after `# `), when present.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Inserts or replaces `fact` by name, returning `true` when it replaced an entry.
    ///
    /// The fact is stored in the section of its [`CodingFact::fact_type`], so changing
    /// a fact's type moves it; unknown entry attributes of the replaced fact are kept.
    pub fn upsert_fact(&mut self, fact: &CodingFact) -> bool {
        let mut fact = fact.clone();
        fact.scope = self.scope.clone();
        let previous = self.take_fact(&fact.name);
        let replaced = previous.is_some();
        let extra = previous.map(|entry| entry.extra).unwrap_or_default();
        let rank = type_rank(fact.fact_type);
        let block = self.section_slot(rank);
        if let Block::Facts { entries, .. } = block {
            entries.push(FactEntry { fact, extra });
        }
        replaced
    }

    /// Removes the fact named `name`, returning whether it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.take_fact(name).is_some()
    }

    /// Returns the fact named `name`.
    #[must_use]
    pub fn fact(&self, name: &str) -> Option<CodingFact> {
        self.entry(name).map(|entry| entry.fact.clone())
    }

    /// Returns every fact of the document, in document order.
    #[must_use]
    pub fn facts(&self) -> Vec<CodingFact> {
        self.blocks
            .iter()
            .filter_map(|block| match block {
                Block::Facts { entries, .. } => Some(entries.iter()),
                Block::Raw { .. } => None,
            })
            .flatten()
            .map(|entry| entry.fact.clone())
            .collect()
    }

    /// Whether the document holds no fact.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self
            .blocks
            .iter()
            .any(|block| matches!(block, Block::Facts { entries, .. } if !entries.is_empty()))
    }

    /// Returns the entry named `name`, if any.
    fn entry(&self, name: &str) -> Option<&FactEntry> {
        self.blocks
            .iter()
            .filter_map(|block| match block {
                Block::Facts { entries, .. } => Some(entries.iter()),
                Block::Raw { .. } => None,
            })
            .flatten()
            .find(|entry| entry.fact.name == name)
    }

    /// Removes (and returns) the entry named `name` from whichever section holds it.
    fn take_fact(&mut self, name: &str) -> Option<FactEntry> {
        for block in &mut self.blocks {
            if let Block::Facts { entries, .. } = block {
                if let Some(position) = entries.iter().position(|entry| entry.fact.name == name) {
                    return Some(entries.remove(position));
                }
            }
        }
        None
    }

    /// Mutable access to every fact, for scope rebinding.
    fn facts_mut(&mut self) -> impl Iterator<Item = &mut CodingFact> {
        self.blocks
            .iter_mut()
            .filter_map(|block| match block {
                Block::Facts { entries, .. } => Some(entries.iter_mut()),
                Block::Raw { .. } => None,
            })
            .flatten()
            .map(|entry| &mut entry.fact)
    }

    /// Returns the block of `rank`, creating it in canonical section order.
    fn section_slot(&mut self, rank: u8) -> &mut Block {
        let existing = self.blocks.iter().position(|block| match block {
            Block::Facts { fact_type, .. } => type_rank(*fact_type) == rank,
            Block::Raw { .. } => false,
        });
        if let Some(position) = existing {
            return &mut self.blocks[position];
        }
        let mut insert_at = self.blocks.len();
        for (position, block) in self.blocks.iter().enumerate() {
            if let Block::Facts { fact_type, .. } = block {
                if type_rank(*fact_type) > rank {
                    insert_at = position;
                    break;
                }
            }
        }
        self.blocks.insert(
            insert_at,
            Block::Facts {
                fact_type: rank_type(rank),
                entries: Vec::new(),
                raw: Vec::new(),
            },
        );
        &mut self.blocks[insert_at]
    }
}

/// Renders one fact entry.
fn render_entry(entry: &FactEntry) -> String {
    let mut lines = vec![format!("### {}", entry.fact.name), String::new()];
    lines.push(format!(
        "{ATTR_PREFIX}{ATTR_DESCRIPTION}: {}",
        one_line(&entry.fact.description)
    ));
    let tags = serde_json::to_string(&entry.fact.tags).unwrap_or_else(|_error| "[]".to_owned());
    lines.push(format!("{ATTR_PREFIX}{ATTR_TAGS}: {tags}"));
    if let Some(path) = &entry.fact.path {
        lines.push(format!("{ATTR_PREFIX}{ATTR_PATH}: {}", one_line(path)));
    }
    lines.push(format!(
        "{ATTR_PREFIX}{ATTR_UPDATED_AT}: {}",
        entry
            .fact
            .updated_at
            .to_rfc3339_opts(SecondsFormat::Secs, true)
    ));
    for (key, value) in &entry.extra {
        lines.push(format!("{ATTR_PREFIX}{key}: {value}"));
    }
    if !entry.fact.content.is_empty() {
        lines.push(String::new());
        for line in entry.fact.content.split('\n') {
            lines.push(escape_body_line(line));
        }
    }
    lines.join("\n")
}

/// Folds embedded newlines into spaces, so an attribute stays one line.
fn one_line(value: &str) -> String {
    value
        .split(['\n', '\r'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Parses a tag attribute: JSON array first, comma-separated fallback.
fn parse_tags(value: &str) -> Vec<String> {
    if let Ok(tags) = serde_json::from_str::<Vec<String>>(value) {
        return tags
            .into_iter()
            .map(|tag| tag.trim().to_owned())
            .filter(|tag| !tag.is_empty())
            .collect();
    }
    value
        .split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Escapes a body line that would otherwise parse as a heading.
fn escape_body_line(line: &str) -> String {
    if is_section_boundary(line) || line.starts_with("### ") || line == "###" {
        return format!("\\{line}");
    }
    line.to_owned()
}

/// Reverses [`escape_body_line`].
fn unescape_body_line(line: &str) -> String {
    match line.strip_prefix("\\#") {
        Some(rest) => format!("#{rest}"),
        None => line.to_owned(),
    }
}

/// The title text of a `# heading` line, if it is one.
fn title_of(line: &str) -> Option<&str> {
    let heading = line.strip_prefix("# ")?;
    let heading = heading.trim();
    if heading.is_empty() {
        None
    } else {
        Some(heading)
    }
}

/// The fact type of a `## heading` section, if the heading names one.
fn section_fact_type(line: &str) -> Option<CodingFactType> {
    let heading = line.strip_prefix("## ")?;
    let heading = heading.trim().to_ascii_lowercase();
    CodingFactType::ALL
        .into_iter()
        .find(|fact_type| fact_type.as_str() == heading)
}

/// The fact name of a `### heading` entry line, if it is one.
fn entry_name(line: &str) -> Option<&str> {
    let heading = line.strip_prefix("### ")?;
    let heading = heading.trim();
    if heading.is_empty() {
        None
    } else {
        Some(heading)
    }
}

/// A `- key: value` attribute line, if the line is one.
fn attribute_of(line: &str) -> Option<(String, String)> {
    let attribute = line.trim().strip_prefix(ATTR_PREFIX)?;
    let (key, value) = attribute.split_once(':')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_owned(), value.trim().to_owned()))
}

/// Whether `line` starts a new top-level or section-level heading block.
fn is_section_boundary(line: &str) -> bool {
    let trimmed = line.trim_end();
    trimmed == "#" || trimmed == "##" || trimmed.starts_with("# ") || trimmed.starts_with("## ")
}

/// Trims leading and trailing blank lines and joins the rest with `\n`.
fn trim_blank_edges<S: AsRef<str>>(lines: &[S]) -> String {
    let first = lines
        .iter()
        .position(|line| !line.as_ref().trim().is_empty());
    let Some(first) = first else {
        return String::new();
    };
    let last = lines
        .iter()
        .rposition(|line| !line.as_ref().trim().is_empty())
        .unwrap_or(first);
    lines[first..=last]
        .iter()
        .map(|line| line.as_ref())
        .collect::<Vec<&str>>()
        .join("\n")
}

/// The canonical section rank of a fact type.
const fn type_rank(fact_type: CodingFactType) -> u8 {
    match fact_type {
        CodingFactType::User => 0,
        CodingFactType::Feedback => 1,
        CodingFactType::Project => 2,
        CodingFactType::Reference => 3,
    }
}

/// The fact type at a canonical section rank.
const fn rank_type(rank: u8) -> CodingFactType {
    match rank {
        0 => CodingFactType::User,
        1 => CodingFactType::Feedback,
        3 => CodingFactType::Reference,
        _ => CodingFactType::Project,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn fixed_time() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 13, 7, 0, 0).unwrap()
    }

    fn fact(name: &str, content: &str) -> CodingFact {
        let mut fact = CodingFact::new(&scope(), name, content);
        fact.updated_at = fixed_time();
        fact
    }

    fn sample_document() -> MarkdownMemory {
        let mut document = MarkdownMemory::new().with_scope(scope());
        let _ = document.upsert_fact(
            &fact("build", "cargo build")
                .with_description("How to build")
                .with_tags(vec!["build".to_owned()])
                .with_path("docs/BUILD.md"),
        );
        let _ =
            document.upsert_fact(&fact("tone", "be concise").with_type(CodingFactType::Feedback));
        let _ = document.upsert_fact(&fact("alice", "the user").with_type(CodingFactType::User));
        document
    }

    #[test]
    fn a_fresh_document_holds_the_schema_and_title() {
        let document = MarkdownMemory::new();
        assert_eq!(document.title(), Some("Memory"));
        assert_eq!(document.front_matter().get("schema"), Some(SCHEMA_MARKER));
        assert_eq!(document.front_matter().get("version"), Some(FORMAT_VERSION));
        assert!(document.is_empty());
        assert!(document.facts().is_empty());
        assert_eq!(
            document.render(),
            "---\nschema: openjiuwen-memory-lite\nversion: 1\n---\n\n# Memory\n"
        );
    }

    #[test]
    fn render_then_parse_preserves_every_fact() {
        let document = sample_document();
        let rendered = document.render();
        let parsed = MarkdownMemory::parse(&rendered)
            .unwrap()
            .with_scope(scope());
        assert_eq!(parsed.facts(), document.facts());
        assert_eq!(parsed.render(), rendered);
        let build = parsed.fact("build").unwrap();
        assert_eq!(build.description, "How to build");
        assert_eq!(build.tags, vec!["build".to_owned()]);
        assert_eq!(build.path.as_deref(), Some("docs/BUILD.md"));
        assert_eq!(build.updated_at, fixed_time());
    }

    #[test]
    fn render_is_idempotent_under_parse() {
        let rendered = sample_document().render();
        let once = MarkdownMemory::parse(&rendered).unwrap().render();
        let twice = MarkdownMemory::parse(&once).unwrap().render();
        assert_eq!(rendered, once);
        assert_eq!(once, twice);
    }

    #[test]
    fn sections_follow_the_canonical_order() {
        let rendered = sample_document().render();
        let user = rendered.find("## User").unwrap();
        let feedback = rendered.find("## Feedback").unwrap();
        let project = rendered.find("## Project").unwrap();
        assert!(user < feedback, "{rendered}");
        assert!(feedback < project, "{rendered}");
    }

    #[test]
    fn a_blank_file_parses_to_an_empty_document() {
        for text in ["", "\n", "   \n\n\t\n"] {
            let document = MarkdownMemory::parse(text).unwrap();
            assert!(document.is_empty());
            assert!(document.facts().is_empty());
            // Rendering an empty document still emits the canonical header.
            assert!(document.render().ends_with("# Memory\n"));
        }
    }

    #[test]
    fn crlf_input_parses_and_renders_lf() {
        let text = "---\r\nschema: x\r\n---\r\n\r\n# Memory\r\n\r\n## Project\r\n\r\n\
                    ### build\r\n\r\n- description: d\r\n- tags: [\"t\"]\r\n\r\nbody line\r\n";
        let document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
        assert_eq!(document.front_matter().get("schema"), Some("x"));
        let build = document.fact("build").unwrap();
        assert_eq!(build.content, "body line");
        assert!(!document.render().contains('\r'));
    }

    #[test]
    fn unknown_front_matter_keys_and_raw_lines_are_preserved() {
        let text = "---\nschema: x\nnot a pair\ncustom: keep me\n---\n\n# Memory\n";
        let document = MarkdownMemory::parse(text).unwrap();
        assert_eq!(document.front_matter().get("custom"), Some("keep me"));
        let rendered = document.render();
        assert!(rendered.contains("not a pair"), "{rendered}");
        assert!(rendered.contains("custom: keep me"), "{rendered}");
        let again = MarkdownMemory::parse(&rendered).unwrap();
        assert_eq!(again.front_matter().get("custom"), Some("keep me"));
    }

    #[test]
    fn unknown_sections_survive_a_round_trip() {
        let text = concat!(
            "# Memory\n\n",
            "## Project\n\n### build\n\n- description: d\n\nbody\n\n",
            "## Notes\n\nfree-form text\nwith two lines\n\n",
            "### not-a-fact\n\n- loose: yes\n",
        );
        let document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
        assert_eq!(document.facts().len(), 1);
        let rendered = document.render();
        assert!(rendered.contains("## Notes"), "{rendered}");
        assert!(
            rendered.contains("free-form text\nwith two lines"),
            "{rendered}"
        );
        assert!(rendered.contains("### not-a-fact"), "{rendered}");
        assert_eq!(MarkdownMemory::parse(&rendered).unwrap().render(), rendered);
    }

    #[test]
    fn stray_text_inside_a_fact_section_is_kept() {
        let text =
            "# Memory\n\n## Project\n\nintro line\n\n### build\n\n- description: d\n\nbody\n";
        let document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
        assert_eq!(document.facts().len(), 1);
        let rendered = document.render();
        assert!(rendered.contains("intro line"), "{rendered}");
        assert_eq!(MarkdownMemory::parse(&rendered).unwrap().render(), rendered);
    }

    #[test]
    fn an_unterminated_front_matter_is_a_validation_error() {
        let error = MarkdownMemory::parse("---\nschema: x\n").unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error.to_string().contains("front matter"));
    }

    #[test]
    fn facts_upsert_by_name_and_move_between_sections() {
        let mut document = MarkdownMemory::new().with_scope(scope());
        assert!(!document.upsert_fact(&fact("build", "first")));
        assert!(document.upsert_fact(&fact("build", "second")));
        assert_eq!(document.facts().len(), 1);
        assert_eq!(document.fact("build").unwrap().content, "second");

        let promoted = fact("build", "second").with_type(CodingFactType::Feedback);
        assert!(document.upsert_fact(&promoted));
        let rendered = document.render();
        assert!(rendered.contains("## Feedback"), "{rendered}");
        assert!(!rendered.contains("## Project"), "{rendered}");
        assert_eq!(
            document.fact("build").unwrap().fact_type,
            CodingFactType::Feedback
        );
    }

    #[test]
    fn removing_a_fact_is_reported_and_drops_its_section() {
        let mut document = sample_document();
        assert!(document.remove("build"));
        assert!(!document.remove("build"));
        assert!(document.fact("build").is_none());
        assert_eq!(document.facts().len(), 2);
        assert!(!document.render().contains("### build"));
    }

    #[test]
    fn the_scope_is_applied_to_every_fact() {
        let document = sample_document();
        for fact in document.facts() {
            assert_eq!(fact.scope, scope());
        }
        let text = document.render();
        let parsed = MarkdownMemory::parse(&text).unwrap().with_scope(scope());
        assert_eq!(parsed.facts()[0].scope, scope());
        assert_eq!(parsed.scope(), &scope());
    }

    #[test]
    fn bodies_keep_heading_lookalikes_and_blank_lines() {
        let mut document = MarkdownMemory::new().with_scope(scope());
        let body = "intro\n\n## not a section\n\n### also not\n\ntail";
        let _ = document.upsert_fact(&fact("multi", body));
        let rendered = document.render();
        assert!(rendered.contains("\\## not a section"), "{rendered}");
        assert!(rendered.contains("\\### also not"), "{rendered}");
        let parsed = MarkdownMemory::parse(&rendered)
            .unwrap()
            .with_scope(scope());
        assert_eq!(parsed.fact("multi").unwrap().content, body);
        assert_eq!(parsed.facts().len(), 1);
    }

    #[test]
    fn attribute_values_are_single_line() {
        let mut document = MarkdownMemory::new().with_scope(scope());
        let mut review = fact("review", "body").with_description("line one\nline two");
        review.path = Some("docs/\nREF.md".to_owned());
        let _ = document.upsert_fact(&review);
        let parsed = MarkdownMemory::parse(&document.render()).unwrap();
        let parsed = parsed.with_scope(scope());
        let restored = parsed.fact("review").unwrap();
        assert_eq!(restored.description, "line one line two");
        assert_eq!(restored.path.as_deref(), Some("docs/ REF.md"));
    }

    #[test]
    fn tags_accept_json_and_comma_separated_forms() {
        let text = concat!(
            "# Memory\n\n## Project\n\n### a\n\n- description: d\n- tags: [\"one\", \"two\"]\n\nbody\n",
            "\n### b\n\n- description: d\n- tags: one, two\n\nbody\n",
            "\n### c\n\n- description: d\n- tags: not json, , three\n\nbody\n",
        );
        let document = MarkdownMemory::parse(text).unwrap();
        assert_eq!(document.fact("a").unwrap().tags, vec!["one", "two"]);
        assert_eq!(document.fact("b").unwrap().tags, vec!["one", "two"]);
        assert_eq!(document.fact("c").unwrap().tags, vec!["not json", "three"]);
    }

    #[test]
    fn a_missing_or_bad_timestamp_decodes_to_the_epoch() {
        let text = "# Memory\n\n## Project\n\n### a\n\n- description: d\n\nbody\n\n\
                    ### b\n\n- description: d\n- updated_at: yesterday\n\nbody\n";
        let document = MarkdownMemory::parse(text).unwrap();
        assert_eq!(
            document.fact("a").unwrap().updated_at,
            DateTime::<Utc>::default()
        );
        assert_eq!(
            document.fact("b").unwrap().updated_at,
            DateTime::<Utc>::default()
        );
    }

    #[test]
    fn unknown_entry_attributes_survive_an_upsert() {
        let text =
            "# Memory\n\n## Project\n\n### build\n\n- description: d\n- owner: alice\n\nbody\n";
        let mut document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
        let _ = document.upsert_fact(&fact("build", "new body").with_description("e"));
        let rendered = document.render();
        assert!(rendered.contains("- owner: alice"), "{rendered}");
        let parsed = MarkdownMemory::parse(&rendered).unwrap();
        assert_eq!(parsed.fact("build").unwrap().content, "new body");
    }

    #[test]
    fn front_matter_is_ordered_and_keeps_positions() {
        let mut front_matter = FrontMatter::new();
        assert_eq!(front_matter.insert("b", "1"), None);
        assert_eq!(front_matter.insert("a", "2"), None);
        assert_eq!(front_matter.insert("b", "3"), Some("1".to_owned()));
        let keys: Vec<&str> = front_matter.iter().map(|(key, _)| key).collect();
        assert_eq!(keys, vec!["b", "a"]);
        assert_eq!(front_matter.get("b"), Some("3"));
        assert_eq!(front_matter.len(), 2);
        assert_eq!(front_matter.remove("b"), Some("3".to_owned()));
        assert!(!front_matter.is_empty());
        assert_eq!(front_matter.remove("missing"), None);
        assert!(FrontMatter::new().is_empty());
    }

    #[test]
    fn with_front_matter_adds_a_key_without_reordering() {
        let document = MarkdownMemory::new()
            .with_scope(scope())
            .with_front_matter("owner", "alice");
        let rendered = document.render();
        let schema = rendered.find("schema:").unwrap();
        let owner = rendered.find("owner: alice").unwrap();
        assert!(schema < owner, "{rendered}");
        let parsed = MarkdownMemory::parse(&rendered).unwrap();
        assert_eq!(parsed.front_matter().get("owner"), Some("alice"));
    }

    #[test]
    fn a_document_without_a_title_renders_no_title_line() {
        let document = MarkdownMemory::parse("## Project\n\n### a\n\n- description: d\n\nbody\n")
            .unwrap()
            .with_scope(scope());
        assert!(document.title().is_none());
        let rendered = document.render();
        assert!(!rendered.contains("# Memory"), "{rendered}");
        assert!(rendered.starts_with("## Project"), "{rendered}");
        assert_eq!(document.facts().len(), 1);
    }

    #[test]
    fn empty_sections_render_nothing() {
        let mut document = MarkdownMemory::new().with_scope(scope());
        let _ = document.upsert_fact(&fact("build", "body"));
        let _ = document.remove("build");
        let rendered = document.render();
        assert!(!rendered.contains("## Project"), "{rendered}");
        assert!(rendered.ends_with("# Memory\n"), "{rendered}");
    }

    #[test]
    fn a_fact_with_an_empty_body_round_trips() {
        let mut document = MarkdownMemory::new().with_scope(scope());
        let _ = document.upsert_fact(&fact("empty", ""));
        let rendered = document.render();
        let parsed = MarkdownMemory::parse(&rendered)
            .unwrap()
            .with_scope(scope());
        assert_eq!(parsed.fact("empty").unwrap().content, "");
    }

    #[test]
    fn parsing_never_panics_on_hostile_input() {
        for text in [
            "---",
            "---\n---",
            "#",
            "##",
            "###",
            "### \n\n- : \n\n",
            "## Project\n### ###\n- description\n",
            "\u{feff}# Memory\n",
        ] {
            let document = MarkdownMemory::parse(text).unwrap_or_else(|error| {
                panic!("unexpected error for {text:?}: {error}");
            });
            let _rendered = document.render();
        }
    }
}
