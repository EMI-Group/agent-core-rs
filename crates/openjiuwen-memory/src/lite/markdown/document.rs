//! The `MEMORY.md` document model: title, fact sections, and the parse/render round trip.
//!
//! A [`MarkdownMemory`] is the parsed form of one lite coding-memory file: its
//! front matter ([`FrontMatter`](crate::lite::markdown::FrontMatter)), an optional
//! title line, and one block per fact-type section. Parsing is total for free-form
//! input — unknown sections, stray text inside a known section, unknown entry
//! attributes, and non-`key: value` front-matter lines are kept verbatim — so a
//! hand-edited file never fails to load. Rendering produces the single canonical shape
//! that parsing round-trips byte for byte (see the module documentation of
//! [`crate::lite::markdown`] for the format and its normalization rules).
//!
//! The document carries the [`MemoryScope`] its facts belong to. The scope is never
//! written to the file: one file holds one scope, and `parse` returns a document bound
//! to the default scope for the caller to rebind.
use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::error::Result;

use super::front_matter::{
    FrontMatter, FORMAT_VERSION, FRONT_MATTER_SCHEMA, FRONT_MATTER_VERSION, SCHEMA_MARKER,
};
use crate::lite::fact::{CodingFact, CodingFactType};
use crate::lite::index::type_label;
use crate::record::MemoryScope;

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

    /// Returns the document with its title line replaced (`# Memory` by default).
    ///
    /// Only the title changes: the front matter, the fact sections, and the scope are
    /// kept. An empty title renders no title line at all.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.set_title(title);
        self
    }

    /// Replaces the document's title line in place.
    ///
    /// Only the title changes: the front matter, the fact sections, and the scope are
    /// kept. An empty title renders no title line at all.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
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
        .map(AsRef::as_ref)
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
