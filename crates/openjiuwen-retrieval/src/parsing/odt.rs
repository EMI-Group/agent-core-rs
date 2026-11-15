//! `ODT` (`OpenDocument` Text) text extraction.
//!
//! An `.odt` file is a ZIP archive of `OpenDocument` XML parts: the document body
//! lives in `content.xml` and the document properties in `meta.xml`. This reader
//! is the sibling of [`crate::parsing::docx`] over the same ZIP and XML layers,
//! so no office-format crate is needed.
//!
//! Mapping, in document order inside `<office:body><office:text>`:
//!
//! - `<text:h text:outline-level="N">` becomes a heading line, its text prefixed
//!   with `#` × `N`; `N` is clamped to `1..=9` and an unparsable or missing level
//!   falls back to `1`. Empty headings are dropped;
//! - `<text:p>` becomes a paragraph line (trimmed, empty paragraphs dropped);
//! - `<table:table>` becomes a Markdown table rendered by [`TableData`], one row
//!   per `<table:table-row>` and one cell per `<table:table-cell>` *or*
//!   `<table:covered-table-cell>`; cell paragraphs are joined by newlines, rows
//!   are padded to the widest row, the first row is the header, and empty tables
//!   are dropped;
//! - the containers `<text:section>`, `<text:list>`, `<text:list-item>` and
//!   `<text:list-header>` are transparent, so nested lists flatten into
//!   consecutive lines instead of keeping their outline; every other element
//!   (styles, bookmarks, table-of-contents markers, …) contributes no text.
//!
//! Inline content of a paragraph or heading is collected recursively: `<text:s>`
//! contributes its `text:c` spaces (default one, capped at 1000), `<text:tab>` a
//! tab, `<text:line-break>` a newline, and `<text:span>`, `<text:a>` and
//! `<text:bookmark-ref>` are descended into.
//! `<text:note>` with its citation and body is skipped, as are `draw:*` elements
//! and nested tables, so footnotes and drawings never leak into the body text.
//!
//! Drawings (`draw:*`, including a nested `draw:image`), footnotes
//! (`text:note`), tables of contents (`text:table-of-content`, `text:index-*`),
//! tracked changes (`text:tracked-changes`), annotations (`office:annotation`)
//! and every unknown block-level element contribute no text, but they are not
//! lost: each distinct element is reported in
//! [`OdtDocument::unsupported_elements`] together with its occurrence count and
//! logged once at `warn`.
//!
//! The XML tree keeps an element's direct text in one string (independent of the
//! child elements), so a paragraph's own text is emitted before its child
//! elements: `<text:p>a <text:span>b</text:span></text:p>` renders `a b`, while
//! text that *trails* a span moves ahead of it. That is the only rendering
//! deviation, and it keeps the common leading-text case exact.
//!
//! Images and OCR are out of scope: drawings are skipped and reported, so an
//! image-only `.odt` parses to empty text (with the drawings listed in
//! [`OdtDocument::unsupported_elements`]) instead of failing.
//!
//! `meta.xml` supplies `dc:title` when it is present and readable; a missing or
//! malformed properties part leaves [`OdtDocument::title`] as `None` instead of
//! failing the parse.

use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::{debug, warn};
use serde::{Deserialize, Serialize};

use crate::multimodal::TableData;

use super::xml::{local_name, parse, XmlElement};
use super::ZipArchive;

/// Archive path of the main content part.
pub const CONTENT_PART: &str = "content.xml";

/// Archive path of the document properties part (title, creator, …).
pub const META_PART: &str = "meta.xml";

/// Highest `text:outline-level` rendered as a Markdown heading.
const MAX_OUTLINE_LEVEL: u8 = 9;

/// Largest number of spaces honoured for one `<text:s text:c="…"/>`.
const MAX_REPEAT_SPACES: usize = 1000;

/// A parsed `ODT` document.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OdtDocument {
    /// Rendered body text: non-empty blocks joined by `\n`.
    pub text: String,
    /// Number of paragraph and heading lines emitted.
    pub paragraph_count: usize,
    /// Number of table blocks emitted.
    pub table_count: usize,
    /// `dc:title` from the properties part, when present.
    pub title: Option<String>,
    /// Unsupported or ignored body elements met while walking the document,
    /// each rendered as `<prefixed local name> (×<count>)`.
    ///
    /// Entries are sorted by element name, so the list is deterministic; a
    /// document without such elements reports an empty vector, never `None`.
    #[serde(default)]
    pub unsupported_elements: Vec<String>,
}

impl OdtDocument {
    /// Returns `true` when the document produced no text.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Returns the rendered body text.
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Reads `ODT` documents and renders their body as text.
#[derive(Debug, Clone, Copy, Default)]
pub struct OdtParser;

impl OdtParser {
    /// Creates a parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a whole `.odt` file from its bytes.
    ///
    /// # Errors
    ///
    /// Propagates the [`ZipArchive`] errors (a corrupt container) and returns
    /// [`Error::Validation`] when the archive has no [`CONTENT_PART`] part.
    /// Malformed XML inside the part is [`Error::Validation`] as well, with the
    /// part path as context.
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<OdtDocument> {
        let archive = ZipArchive::from_bytes(bytes.to_vec())?;
        self.parse_archive(&archive)
    }

    /// Parses an already opened archive.
    ///
    /// # Errors
    ///
    /// Same as [`OdtParser::parse_bytes`].
    pub fn parse_archive(&self, archive: &ZipArchive) -> Result<OdtDocument> {
        if !archive.has_entry(CONTENT_PART) {
            return Err(Error::Validation(format!(
                "ODT archive has no {CONTENT_PART} part"
            )));
        }
        let markup = archive.read_text(CONTENT_PART)?;
        let document = parse(&markup).map_err(|error| error.with_context(CONTENT_PART))?;
        let mut blocks = Blocks::default();
        let mut unsupported = Unsupported::default();
        let body = document
            .root()
            .and_then(|root| root.find_first("body"))
            .and_then(|body| body.find_first("text"))
            .cloned()
            .unwrap_or_default();
        collect_blocks(&body, &mut blocks, &mut unsupported);
        for (name, count) in &unsupported.counts {
            warn!(
                element = %name,
                count = *count,
                "ODT element not rendered as text"
            );
        }
        let text = blocks
            .lines
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<String>>()
            .join("\n");
        Ok(OdtDocument {
            text,
            paragraph_count: blocks.paragraph_count,
            table_count: blocks.table_count,
            title: read_title(archive),
            unsupported_elements: unsupported.into_entries(),
        })
    }

    /// Parses the `.odt` file at `path` (read with `tokio::fs`).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a missing file, [`Error::Transport`] for any other
    /// I/O failure, plus every [`OdtParser::parse_bytes`] error.
    pub async fn parse_path(&self, path: impl AsRef<Path>) -> Result<OdtDocument> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => {
                    Error::NotFound(format!("File {} does not exist", path.display()))
                }
                _ => Error::Transport(format!("reading {}: {error}", path.display())),
            })?;
        self.parse_bytes(&bytes)
    }
}

/// Rendered body blocks with their per-kind counters.
#[derive(Debug, Default)]
struct Blocks {
    /// One entry per rendered block, in document order.
    lines: Vec<String>,
    /// Number of paragraph and heading lines pushed so far.
    paragraph_count: usize,
    /// Number of table blocks pushed so far.
    table_count: usize,
}

impl Blocks {
    /// Appends a paragraph or heading line.
    fn push_paragraph(&mut self, line: String) {
        self.lines.push(line);
        self.paragraph_count += 1;
    }

    /// Appends a rendered table block.
    fn push_table(&mut self, markdown: String) {
        self.lines.push(markdown);
        self.table_count += 1;
    }
}

/// Occurrence counts of the body elements that contributed no text.
///
/// The ordered map keeps [`OdtDocument::unsupported_elements`] sorted by
/// element name regardless of document order.
#[derive(Debug, Default)]
struct Unsupported {
    /// One entry per distinct element name (namespace prefix kept).
    counts: std::collections::BTreeMap<String, usize>,
}

impl Unsupported {
    /// Counts one occurrence of `name`.
    fn record(&mut self, name: &str) {
        *self.counts.entry(name.to_owned()).or_default() += 1;
    }

    /// Consumes the collector into its sorted `<name> (×<count>)` entries.
    fn into_entries(self) -> Vec<String> {
        self.counts
            .into_iter()
            .map(|(name, count)| format!("{name} (×{count})"))
            .collect()
    }
}

/// Reads `dc:title` from the properties part, ignoring an absent or broken part.
fn read_title(archive: &ZipArchive) -> Option<String> {
    if !archive.has_entry(META_PART) {
        return None;
    }
    let markup = match archive.read_text(META_PART) {
        Ok(markup) => markup,
        Err(error) => {
            debug!("ignoring unreadable {META_PART}: {error}");
            return None;
        }
    };
    let document = match parse(&markup) {
        Ok(document) => document,
        Err(error) => {
            debug!("ignoring unparsable {META_PART}: {error}");
            return None;
        }
    };
    document
        .root()
        .and_then(|root| root.find_first("title"))
        .map(XmlElement::text_content)
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

/// Walks the text container, appending one block per heading, paragraph, or table.
///
/// Sections and list containers are transparent, so list items appear as
/// consecutive lines; every other element (styles, bookmarks, indices,
/// tracked changes, annotations, drawings, …) contributes no text and is
/// recorded in `unsupported`, and table cells are rendered by their table,
/// never as body lines.
fn collect_blocks(element: &XmlElement, blocks: &mut Blocks, unsupported: &mut Unsupported) {
    for child in &element.children {
        match local_name(&child.name) {
            "h" => {
                if let Some(line) = heading_line(child, unsupported) {
                    blocks.push_paragraph(line);
                }
            }
            "p" => {
                let text = inline_text(child, unsupported).trim().to_owned();
                if !text.is_empty() {
                    blocks.push_paragraph(text);
                }
            }
            "table" => {
                if let Some(markdown) = table_markdown(child, unsupported) {
                    blocks.push_table(markdown);
                }
            }
            "section" | "list" | "list-item" | "list-header" => {
                collect_blocks(child, blocks, unsupported);
            }
            _ => {
                if child.name.starts_with("draw:") {
                    record_draw_subtree(unsupported, child);
                } else {
                    unsupported.record(&child.name);
                }
            }
        }
    }
}

/// Records `element` and every nested `draw:*` element inside it.
///
/// A `draw:frame` usually carries a `draw:image` (and often a text box), so
/// reporting only the outermost drawing would hide the rest; non-drawing
/// descendants are not reported individually because their parent already
/// stands for them.
fn record_draw_subtree(unsupported: &mut Unsupported, element: &XmlElement) {
    unsupported.record(&element.name);
    for child in &element.children {
        if child.name.starts_with("draw:") {
            record_draw_subtree(unsupported, child);
        }
    }
}

/// Builds the Markdown heading line for one `text:h`, or `None` when it is blank.
fn heading_line(heading: &XmlElement, unsupported: &mut Unsupported) -> Option<String> {
    let text = inline_text(heading, unsupported).trim().to_owned();
    if text.is_empty() {
        return None;
    }
    let level = usize::from(heading_level(heading));
    Some(format!("{} {text}", "#".repeat(level)))
}

/// Returns the heading level (`#` count) from `text:outline-level`.
///
/// A missing or unparsable value is level 1, and parsed levels are clamped to
/// `1..=9` so a hostile attribute cannot produce an enormous prefix.
fn heading_level(heading: &XmlElement) -> u8 {
    heading
        .attr("outline-level")
        .and_then(|value| value.trim().parse::<u8>().ok())
        .map_or(1, |level| level.clamp(1, MAX_OUTLINE_LEVEL))
}

/// Collects the inline text of a paragraph or heading in child order.
fn inline_text(element: &XmlElement, unsupported: &mut Unsupported) -> String {
    let mut text = String::new();
    collect_inline(element, &mut text, unsupported);
    text
}

/// Appends the text of `element` and its descendants.
///
/// Direct text of `element` is appended first (see the module documentation),
/// then its children are visited in order. `<text:s>`, `<text:tab>` and
/// `<text:line-break>` contribute their characters, drawings, notes and nested
/// tables contribute no text (they are recorded in `unsupported` instead), and
/// any other element is descended into.
fn collect_inline(element: &XmlElement, text: &mut String, unsupported: &mut Unsupported) {
    text.push_str(&element.text);
    for child in &element.children {
        if child.name.starts_with("draw:") {
            record_draw_subtree(unsupported, child);
            continue;
        }
        match local_name(&child.name) {
            "s" => text.push_str(&" ".repeat(space_count(child))),
            "tab" => text.push('\t'),
            "line-break" => text.push('\n'),
            // Footnotes and nested tables keep no inline text.
            "note" | "table" => unsupported.record(&child.name),
            "note-citation" => {}
            _ => collect_inline(child, text, unsupported),
        }
    }
}

/// Returns the number of spaces one `text:s` element stands for.
///
/// `text:c` defaults to 1 and is capped at [`MAX_REPEAT_SPACES`]; a value that is
/// not a number also falls back to 1.
fn space_count(element: &XmlElement) -> usize {
    element
        .attr("c")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map_or(1, |count| count.min(MAX_REPEAT_SPACES))
}

/// Renders one `table:table` as a Markdown table.
///
/// Rows are the direct `table:table-row` children, cells are their
/// `table:table-cell` and `table:covered-table-cell` children (a covered cell
/// keeps the column position of a merged cell and contributes an empty value),
/// and rows are padded to the widest row so the Markdown table stays
/// rectangular. The first row is the header. A table without any cell is
/// dropped.
fn table_markdown(table: &XmlElement, unsupported: &mut Unsupported) -> Option<String> {
    let mut rows: Vec<Vec<String>> = table
        .children_named("table-row")
        .iter()
        .map(|row| {
            row.children
                .iter()
                .filter(|cell| {
                    matches!(local_name(&cell.name), "table-cell" | "covered-table-cell")
                })
                .map(|cell| cell_text(cell, unsupported))
                .collect()
        })
        .collect();
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return None;
    }
    for row in &mut rows {
        row.resize(width, String::new());
    }
    let header = rows.remove(0);
    let markdown = TableData::new(header, rows).to_markdown();
    (!markdown.trim().is_empty()).then_some(markdown)
}

/// Collects the text of one table cell: its paragraphs joined by newlines.
fn cell_text(cell: &XmlElement, unsupported: &mut Unsupported) -> String {
    cell.children_named("p")
        .iter()
        .map(|paragraph| inline_text(paragraph, unsupported).trim().to_owned())
        .filter(|text| !text.is_empty())
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
