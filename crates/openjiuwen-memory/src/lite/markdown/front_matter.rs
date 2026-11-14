//! The `---` front-matter block at the top of a lite `MEMORY.md` document.
//!
//! A [`FrontMatter`](crate::lite::markdown::FrontMatter) is the ordered `key: value` block that opens the document.
//! It preserves document order, keeps an existing key's position on insert, and keeps
//! every line that is not a `key: value` pair verbatim in a raw tail, so an
//! unrecognized or hand-written front-matter block round-trips unchanged.
//!
//! This module also owns the four schema constants (`FRONT_MATTER_SCHEMA`,
//! `SCHEMA_MARKER`, `FRONT_MATTER_VERSION`, `FORMAT_VERSION`) that describe a fresh
//! document's front matter: the document module writes them into a new document, and
//! the parser here keeps them (or any replacement) in document order.

/// Front-matter key naming the document schema.
pub(super) const FRONT_MATTER_SCHEMA: &str = "schema";
/// Schema marker written into a fresh document's front matter.
pub(super) const SCHEMA_MARKER: &str = "openjiuwen-memory-lite";
/// Front-matter key holding the document format version.
pub(super) const FRONT_MATTER_VERSION: &str = "version";
/// Format version written into a fresh document's front matter.
pub(super) const FORMAT_VERSION: &str = "1";

/// The ordered `key: value` block at the top of a [`MarkdownMemory`](crate::lite::markdown::MarkdownMemory).
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
    pub(super) fn parse_lines(lines: &[&str]) -> Self {
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
    pub(super) fn render(&self) -> Option<String> {
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
