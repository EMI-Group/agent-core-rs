//! `DOCX` (`WordprocessingML`) text extraction.
//!
//! Port of the Python `WordParser`
//! (`indexing/processor/parser/word_parser.py`) without `python-docx`: the
//! package's `word/document.xml` part is parsed directly by [`crate::parsing::xml`]
//! and its body blocks are rendered to Markdown-flavoured text.
//!
//! Block rules, mirroring the reference implementation:
//!
//! - a paragraph becomes its plain text (`w:t` runs concatenated, `w:tab` → tab,
//!   `w:br`/`w:cr` → newline); empty paragraphs are dropped;
//! - a styled paragraph becomes a heading line: `Title` → `# `, `Subtitle` → `## `
//!   (the reference leaves `Subtitle` as plain text; the Rust port treats it as a
//!   second-level heading), `Heading1…Heading9` → `#`×(level+1) (`Heading1` → `## `);
//! - a table becomes a Markdown table rendered by
//!   [`TableData`], with every cell's paragraphs
//!   joined by newlines and pipes escaped.
//!
//! The document text joins the rendered blocks with `\n` and drops empty blocks.
//! The reference appends a blank line after each table; this port does not, which
//! is the only rendering difference.
//!
//! `docProps/core.xml` supplies `dc:title` when it is present and readable; a
//! missing or malformed core-properties part leaves [`DocxDocument::title`] as
//! `None` instead of failing the parse.

use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::debug;
use serde::{Deserialize, Serialize};

use crate::multimodal::TableData;

use super::xml::{local_name, parse, XmlElement};
use super::ZipArchive;

/// Archive path of the main document part.
pub const DOCUMENT_PART: &str = "word/document.xml";

/// Archive path of the core properties part (title, author, …).
pub const CORE_PROPERTIES_PART: &str = "docProps/core.xml";

/// Highest `HeadingN` style level mapped to a Markdown heading.
const MAX_HEADING_LEVEL: u8 = 9;

/// One rendered block of the document body, in document order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DocxBlock {
    /// A plain paragraph; `text` is trimmed but keeps inner tabs/newlines.
    Paragraph {
        /// Paragraph text.
        text: String,
    },
    /// A styled paragraph; `level` is the number of `#` characters (`Title` = 1,
    /// `Heading1` = 2, …).
    Heading {
        /// Markdown heading level (`#` count).
        level: u8,
        /// Heading text.
        text: String,
    },
    /// A table rendered as a Markdown table.
    Table {
        /// Markdown table, one line per row.
        markdown: String,
    },
}

impl DocxBlock {
    /// Renders the block as it appears in [`DocxDocument::text`].
    pub fn render(&self) -> String {
        match self {
            Self::Paragraph { text } => text.clone(),
            Self::Heading { level, text } => format!("{} {text}", "#".repeat(usize::from(*level))),
            Self::Table { markdown } => markdown.clone(),
        }
    }

    /// Returns the block text without heading markers or table markup.
    pub fn text(&self) -> &str {
        match self {
            Self::Paragraph { text } | Self::Heading { text, .. } => text,
            Self::Table { markdown } => markdown,
        }
    }
}

/// A parsed `DOCX` document.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DocxDocument {
    /// Rendered body text: non-empty blocks joined by `\n`.
    pub text: String,
    /// Body blocks in document order.
    pub blocks: Vec<DocxBlock>,
    /// `dc:title` from the core properties, when present.
    pub title: Option<String>,
    /// Number of paragraph blocks emitted (headings included).
    pub paragraph_count: usize,
    /// Number of table blocks emitted.
    pub table_count: usize,
}

impl DocxDocument {
    /// Returns `true` when the document produced no text.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Reads `DOCX` documents and renders their body as text.
#[derive(Debug, Clone, Copy, Default)]
pub struct DocxParser;

impl DocxParser {
    /// Creates a parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a whole `.docx` file from its bytes.
    ///
    /// # Errors
    ///
    /// Propagates the [`ZipArchive`] errors (a corrupt container) and returns
    /// [`Error::Retrieval`] when the archive has no [`DOCUMENT_PART`] part.
    /// Malformed XML inside the part is [`Error::Validation`].
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<DocxDocument> {
        let archive = ZipArchive::from_bytes(bytes.to_vec())?;
        self.parse_archive(&archive)
    }

    /// Parses an already opened archive.
    ///
    /// # Errors
    ///
    /// Same as [`DocxParser::parse_bytes`].
    pub fn parse_archive(&self, archive: &ZipArchive) -> Result<DocxDocument> {
        if !archive.has_entry(DOCUMENT_PART) {
            return Err(Error::Retrieval(format!(
                "DOCX archive has no {DOCUMENT_PART} part"
            )));
        }
        let markup = archive.read_text(DOCUMENT_PART)?;
        let document = parse(&markup).map_err(|error| error.with_context(DOCUMENT_PART))?;
        let mut docx = DocxDocument {
            title: read_title(archive),
            ..DocxDocument::default()
        };
        let body = document
            .root()
            .and_then(|root| root.find_first("body"))
            .cloned()
            .unwrap_or_default();
        collect_blocks(&body, &mut docx.blocks);
        docx.paragraph_count = docx
            .blocks
            .iter()
            .filter(|block| {
                matches!(
                    block,
                    DocxBlock::Paragraph { .. } | DocxBlock::Heading { .. }
                )
            })
            .count();
        docx.table_count = docx
            .blocks
            .iter()
            .filter(|block| matches!(block, DocxBlock::Table { .. }))
            .count();
        docx.text = docx
            .blocks
            .iter()
            .map(DocxBlock::render)
            .filter(|block| !block.trim().is_empty())
            .collect::<Vec<String>>()
            .join("\n");
        Ok(docx)
    }

    /// Parses the `.docx` file at `path` (read with `tokio::fs`).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a missing file, [`Error::Transport`] for any other
    /// I/O failure, plus every [`DocxParser::parse_bytes`] error.
    pub async fn parse_path(&self, path: impl AsRef<Path>) -> Result<DocxDocument> {
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

/// Reads `dc:title` from the core properties, ignoring an absent or broken part.
fn read_title(archive: &ZipArchive) -> Option<String> {
    if !archive.has_entry(CORE_PROPERTIES_PART) {
        return None;
    }
    let markup = match archive.read_text(CORE_PROPERTIES_PART) {
        Ok(markup) => markup,
        Err(error) => {
            debug!("ignoring unreadable {CORE_PROPERTIES_PART}: {error}");
            return None;
        }
    };
    let document = match parse(&markup) {
        Ok(document) => document,
        Err(error) => {
            debug!("ignoring unparsable {CORE_PROPERTIES_PART}: {error}");
            return None;
        }
    };
    let title = document
        .root()
        .and_then(|root| root.find_first("title"))
        .map(XmlElement::text_content)
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty());
    title
}

/// Walks the body, appending one block per paragraph or table.
///
/// Content-control and custom-XML wrappers (`w:sdt`, `w:sdtContent`,
/// `w:customXml`, `w:smartTag`) are transparent so wrapped paragraphs are not
/// lost; every other child (section properties, bookmarks, …) is ignored.
fn collect_blocks(element: &XmlElement, blocks: &mut Vec<DocxBlock>) {
    for child in &element.children {
        match local_name(&child.name) {
            "p" => {
                if let Some(block) = paragraph_block(child) {
                    blocks.push(block);
                }
            }
            "tbl" => {
                if let Some(markdown) = table_markdown(child) {
                    blocks.push(DocxBlock::Table { markdown });
                }
            }
            "sdt" | "sdtContent" | "customXml" | "smartTag" => collect_blocks(child, blocks),
            _ => {}
        }
    }
}

/// Builds the block for one `w:p`, or `None` when the paragraph is blank.
fn paragraph_block(paragraph: &XmlElement) -> Option<DocxBlock> {
    let text = paragraph_text(paragraph).trim().to_owned();
    if text.is_empty() {
        return None;
    }
    match paragraph_style(paragraph).and_then(|style| heading_level(&style)) {
        Some(level) => Some(DocxBlock::Heading { level, text }),
        None => Some(DocxBlock::Paragraph { text }),
    }
}

/// Returns the `w:pPr/w:pStyle/@w:val` style id of a paragraph.
fn paragraph_style(paragraph: &XmlElement) -> Option<String> {
    paragraph
        .find_first("pPr")
        .and_then(|properties| properties.find_first("pStyle"))
        .and_then(|style| style.attr("val"))
        .map(str::to_owned)
}

/// Maps a style id to its Markdown heading level (`#` count).
///
/// Style ids are compared case-insensitively and ignoring spaces, so both
/// `Heading1` (the id Word writes) and `Heading 1` (the display name) work.
/// `Title` is level 1 and `Subtitle` level 2; `HeadingN` maps to `N + 1`, which
/// is the reference implementation's `Title=#, H1=##, H2=###` convention.
fn heading_level(style: &str) -> Option<u8> {
    let normalized: String = style
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    if normalized == "title" {
        return Some(1);
    }
    if normalized == "subtitle" {
        return Some(2);
    }
    let level = normalized.strip_prefix("heading")?.parse::<u8>().ok()?;
    (1..=MAX_HEADING_LEVEL)
        .contains(&level)
        .then_some(level + 1)
}

/// Collects the visible text of a paragraph in run order.
fn paragraph_text(element: &XmlElement) -> String {
    let mut text = String::new();
    collect_run_text(element, &mut text);
    text
}

/// Appends `w:t`/`w:tab`/`w:br` content of `element` and its descendants.
fn collect_run_text(element: &XmlElement, text: &mut String) {
    for child in &element.children {
        match local_name(&child.name) {
            "t" => text.push_str(&child.text),
            "tab" | "ptab" => text.push('\t'),
            "br" | "cr" => text.push('\n'),
            // A nested table is a block of its own; never inline its cells here.
            "tbl" => {}
            _ => collect_run_text(child, text),
        }
    }
}

/// Renders one `w:tbl` as a Markdown table.
///
/// The first row is the header (as in the reference implementation). Cells are
/// their paragraphs joined by newlines, trimmed; rows are padded to the widest
/// row so the Markdown table stays rectangular.
fn table_markdown(table: &XmlElement) -> Option<String> {
    let mut rows: Vec<Vec<String>> = table
        .children_named("tr")
        .iter()
        .map(|row| {
            row.children_named("tc")
                .iter()
                .map(|cell| cell_text(cell))
                .collect()
        })
        .collect();
    if rows.is_empty() {
        return None;
    }
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

/// Collects the text of one `w:tc`: its paragraphs joined by newlines.
fn cell_text(cell: &XmlElement) -> String {
    let paragraphs: Vec<String> = cell
        .children_named("p")
        .iter()
        .map(|paragraph| paragraph_text(paragraph).trim().to_owned())
        .filter(|text| !text.is_empty())
        .collect();
    paragraphs.join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    const DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Rust rewrite</w:t></w:r></w:p>
    <w:p><w:r><w:t>Intro</w:t><w:tab/><w:t>after tab</w:t><w:br/><w:t>next line</w:t></w:r></w:p>
    <w:p/>
    <w:p><w:pPr><w:pStyle w:val="Heading 1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
    <w:sdt><w:sdtContent>
      <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Wrapped</w:t></w:r></w:p>
    </w:sdtContent></w:sdt>
    <w:tbl>
      <w:tblPr/>
      <w:tr><w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Notes</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>alice</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>a|b</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>bob</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>"#;

    #[test]
    fn renders_paragraphs_headings_and_tables() {
        let markup = parse(DOCUMENT).unwrap();
        let body = markup.root().unwrap().find_first("body").unwrap();
        let mut blocks = Vec::new();
        collect_blocks(body, &mut blocks);
        assert_eq!(
            blocks,
            vec![
                DocxBlock::Heading {
                    level: 1,
                    text: "Rust rewrite".to_owned()
                },
                DocxBlock::Paragraph {
                    text: "Intro\tafter tab\nnext line".to_owned()
                },
                DocxBlock::Heading {
                    level: 2,
                    text: "Overview".to_owned()
                },
                DocxBlock::Heading {
                    level: 3,
                    text: "Wrapped".to_owned()
                },
                DocxBlock::Table {
                    markdown: "| Name | Notes |\n| --- | --- |\n| alice | a\\|b |\n| bob |  |"
                        .to_owned()
                },
            ]
        );
        assert_eq!(blocks[0].render(), "# Rust rewrite");
        assert_eq!(blocks[1].text(), "Intro\tafter tab\nnext line");
        assert_eq!(blocks[2].render(), "## Overview");
        assert_eq!(blocks[3].render(), "### Wrapped");
    }

    #[test]
    fn style_ids_map_to_markdown_levels() {
        assert_eq!(heading_level("Title"), Some(1));
        assert_eq!(heading_level(" subtitle "), Some(2));
        assert_eq!(heading_level("Heading1"), Some(2));
        assert_eq!(heading_level("Heading 6"), Some(7));
        assert_eq!(heading_level("heading9"), Some(10));
        assert_eq!(heading_level("Heading10"), None);
        assert_eq!(heading_level("Heading"), None);
        assert_eq!(heading_level("Normal"), None);
        assert_eq!(heading_level("MyHeading1"), None);
    }

    #[test]
    fn cell_text_joins_paragraphs_and_empty_tables_are_skipped() {
        let markup = parse(
            r"<w:tbl><w:tr><w:tc><w:p><w:r><w:t> one </w:t></w:r></w:p>
               <w:p><w:r><w:t>two</w:t></w:r></w:p></w:tc></w:tr></w:tbl>",
        )
        .unwrap();
        let table = markup.root().unwrap();
        let cell = table.find_first("tc").unwrap();
        assert_eq!(cell_text(cell), "one\ntwo");
        assert_eq!(table_markdown(table).unwrap(), "| one<br>two |\n| --- |");
        let empty = parse("<w:tbl><w:tr/></w:tbl>").unwrap();
        assert_eq!(table_markdown(empty.root().unwrap()), None);
    }

    #[test]
    fn paragraphs_without_styles_or_text_are_plain_and_dropped() {
        let markup = parse("<w:p><w:r><w:t>plain</w:t></w:r></w:p><w:p><w:r/></w:p>").unwrap();
        assert_eq!(markup.roots.len(), 2);
        let plain = paragraph_block(&markup.roots[0]).unwrap();
        assert_eq!(plain.text(), "plain");
        assert!(matches!(plain, DocxBlock::Paragraph { .. }));
        assert_eq!(paragraph_block(&markup.roots[1]), None);
        // A styled but empty paragraph is dropped as well.
        let styled = parse(r#"<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr></w:p>"#).unwrap();
        assert_eq!(paragraph_block(styled.root().unwrap()), None);
    }

    #[test]
    fn missing_document_part_is_reported_as_retrieval() {
        // A legal but empty archive: just the end-of-central-directory record.
        let empty = vec![
            0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let archive = ZipArchive::from_bytes(empty).unwrap();
        let error = DocxParser::new().parse_archive(&archive).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.to_string().contains(DOCUMENT_PART), "{error}");
    }
}
