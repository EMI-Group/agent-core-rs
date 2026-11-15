//! `PPTX` (`PowerPoint` / `PresentationML`) text extraction.
//!
//! The presentation is a ZIP container of XML parts, so this reader sits on the
//! same [`crate::parsing::zip`] and [`crate::parsing::xml`] layers as the DOCX and
//! XLSX readers and adds no dependency.
//!
//! What is read:
//!
//! - slide order from `ppt/presentation.xml` (`<p:sldIdLst>`) through the
//!   presentation relationships, falling back to numeric `ppt/slides/slideN.xml`
//!   ordering only when the presentation part declares no slides at all (a
//!   missing presentation or relationships part, no `<p:sldIdLst>`, or an empty
//!   list);
//! - each slide's shapes (`<p:sp>`) and their paragraphs (`<a:p>`, with every
//!   `<a:t>` run concatenated and `<a:br>` turned into a newline);
//! - tables (`<a:tbl>`) inside graphic frames, rendered by
//!   [`TableData`] as Markdown;
//! - speaker notes from `ppt/notesSlides/notesSlideN.xml`;
//! - the deck title from `docProps/core.xml` (`dc:title`).
//!
//! Images, charts, `SmartArt`, and embedded media are out of scope: their text
//! payloads would need OCR or a rendering stack, so they contribute no text.
//! Group shapes (`<p:grpSp>`) are walked transparently.
//!
//! Slide text is rendered Markdown-style: a `## Slide {index}` heading (1-based),
//! the slide body lines, and, when notes exist, a `### Notes` heading followed by
//! the notes lines.
//!
//! An unresolved slide keeps its deck position instead of disappearing: when a
//! `<p:sldId>` has no `r:id`, its id is missing from the presentation
//! relationships, the target is not a slide part, or the target part is absent
//! from the archive, the slide becomes a placeholder
//! ([`PptxSlide::missing_part`]) whose body reads `! missing slide part:
//! {reference}`. Every unresolved entry is listed in [`PptxDocument::problems`]
//! in presentation order and logged once at `warn`, so a lossy deck is visible in
//! the parsed result rather than silently short of a slide.

use std::collections::HashMap;
use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::{debug, warn};
use serde::{Deserialize, Serialize};

use crate::multimodal::TableData;

use super::xml::{local_name, parse, XmlDocument, XmlElement};
use super::ZipArchive;

/// Archive path of the presentation part (slide order and size).
pub const PRESENTATION_PART: &str = "ppt/presentation.xml";

/// Archive path of the presentation relationships part (`rId` → slide part).
pub const PRESENTATION_RELS_PART: &str = "ppt/_rels/presentation.xml.rels";

/// Archive directory holding the slide parts.
pub const SLIDES_DIR: &str = "ppt/slides/";

/// Archive directory holding the speaker-notes parts.
pub const NOTES_SLIDES_DIR: &str = "ppt/notesSlides/";

/// Archive path of the core properties part (title, author, …).
pub const CORE_PROPERTIES_PART: &str = "docProps/core.xml";

/// Directory the presentation relationship targets are relative to.
const PRESENTATION_DIR: &str = "ppt";

/// Reference reported for a `<p:sldId>` that carries no `r:id` at all.
const NO_RELATIONSHIP_REFERENCE: &str = "<no r:id>";

/// Placeholder types whose shape text becomes the slide title.
const TITLE_PLACEHOLDERS: [&str; 2] = ["title", "ctrTitle"];

/// One ordered, rendered slide.
///
/// A slide whose part could not be resolved from the presentation relationships
/// is kept as a placeholder record instead of being dropped: see
/// [`missing_part`](PptxSlide::missing_part).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PptxSlide {
    /// 1-based position of the slide in presentation order.
    pub index: usize,
    /// Rendered slide body lines joined by `\n` (no `## Slide` heading, no notes).
    pub text: String,
    /// Title placeholder text, or the first non-empty body line.
    pub title: Option<String>,
    /// Speaker-notes text, when the slide has a non-empty notes part.
    pub notes: Option<String>,
    /// Whether a notes part exists for this slide, even when it is empty.
    pub has_notes: bool,
    /// The unresolved relationship reference (e.g. `rId7`) when this slide's part
    /// could not be resolved, `None` for a real slide.
    ///
    /// A placeholder slide keeps its 1-based deck position in
    /// [`index`](PptxSlide::index), sets [`text`](PptxSlide::text) to exactly
    /// `! missing slide part: {reference}`, and leaves
    /// [`title`](PptxSlide::title) as `None` with no notes
    /// ([`notes`](PptxSlide::notes) is `None`, [`has_notes`](PptxSlide::has_notes)
    /// is `false`). A `<p:sldId>` that carries no `r:id` at all reports the
    /// reference `<no r:id>`; every other unresolved slide reports its `r:id`.
    /// The same string appears in [`PptxDocument::problems`].
    #[serde(default)]
    pub missing_part: Option<String>,
}

/// A parsed `PPTX` presentation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PptxDocument {
    /// Rendered document text: `## Slide {index}` blocks joined by `\n`.
    pub text: String,
    /// Slides in presentation order, placeholder slides included.
    pub slides: Vec<PptxSlide>,
    /// Number of slides, equal to `slides.len()`; placeholder slides count,
    /// because they are slides in deck order.
    pub slide_count: usize,
    /// `dc:title` from the core properties, when present.
    pub title: Option<String>,
    /// The unresolved slide entries, in presentation order.
    ///
    /// Each entry names the 1-based slide position, the relationship id (or its
    /// absence), the target when known, and the reason, e.g.
    /// `slide 2: rId7 is not mapped by ppt/_rels/presentation.xml.rels`,
    /// `slide 3: rId8 target slides/slide3.xml is not a slide part`, or
    /// `slide 4: rId9 target slides/slide4.xml is missing from the archive`.
    /// An empty vector means the deck resolved completely.
    #[serde(default)]
    pub problems: Vec<String>,
}

impl PptxDocument {
    /// Returns `true` when the document produced no text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Returns the slide at the 1-based position `index`.
    ///
    /// `index == 0` and indexes past the last slide yield `None`.
    #[must_use]
    pub fn slide(&self, index: usize) -> Option<&PptxSlide> {
        index
            .checked_sub(1)
            .and_then(|zero_based| self.slides.get(zero_based))
    }
}

/// Reads `PPTX` presentations and renders their slides as text.
#[derive(Debug, Clone, Copy, Default)]
pub struct PptxParser;

impl PptxParser {
    /// Creates a parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a whole `.pptx` file from its bytes.
    ///
    /// # Errors
    ///
    /// Propagates the [`ZipArchive`] errors (non-ZIP or corrupt bytes) and
    /// returns [`Error::Validation`] when the archive holds no slide part.
    /// Malformed slide, notes, presentation, or relationship XML is
    /// [`Error::Validation`] with the part path as context.
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<PptxDocument> {
        let archive = ZipArchive::from_bytes(bytes.to_vec())?;
        self.parse_archive(&archive)
    }

    /// Parses an already opened archive.
    ///
    /// # Errors
    ///
    /// Same as [`PptxParser::parse_bytes`].
    pub fn parse_archive(&self, archive: &ZipArchive) -> Result<PptxDocument> {
        let slots = slide_parts(archive)?;
        if slots.is_empty() {
            return Err(Error::Validation(format!(
                "PPTX has no slides: no slide part is reachable through {PRESENTATION_PART} \
                 or {SLIDES_DIR}"
            )));
        }
        let mut document = PptxDocument {
            title: read_title(archive),
            ..PptxDocument::default()
        };
        for (position, slot) in slots.iter().enumerate() {
            let index = position + 1;
            let Some(part) = &slot.part else {
                // Unresolved slot: keep the deck position and surface the loss.
                document
                    .slides
                    .push(placeholder_slide(index, &slot.reference));
                let problem = slot
                    .problem
                    .clone()
                    .unwrap_or_else(|| format!("slide {index}: slide part unresolved"));
                warn!(problem = %problem, "PPTX slide part unresolved");
                document.problems.push(problem);
                continue;
            };
            document.slides.push(parse_slide(archive, part, index)?);
        }
        document.slide_count = document.slides.len();
        document.text = render_document(&document.slides);
        Ok(document)
    }

    /// Parses the `.pptx` file at `path` (read with `tokio::fs`).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a missing file, [`Error::Transport`] for any other
    /// I/O failure, plus every [`PptxParser::parse_bytes`] error.
    pub async fn parse_path(&self, path: impl AsRef<Path>) -> Result<PptxDocument> {
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

/// One slot in presentation order: a resolved slide part, or the reason the part
/// could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SlideSlot {
    /// Archive path of the resolved slide part.
    part: Option<String>,
    /// Relationship reference (`rId7`, or [`NO_RELATIONSHIP_REFERENCE`]) of an
    /// unresolved slot.
    reference: String,
    /// Human-readable reason an unresolved slot could not be turned into a part.
    problem: Option<String>,
}

impl SlideSlot {
    /// A slot resolving to `part`.
    fn resolved(part: String) -> Self {
        Self {
            part: Some(part),
            reference: String::new(),
            problem: None,
        }
    }

    /// A slot that keeps its deck position but has no part.
    fn unresolved(reference: impl Into<String>, problem: String) -> Self {
        Self {
            part: None,
            reference: reference.into(),
            problem: Some(problem),
        }
    }
}

/// Returns the slides in presentation order.
///
/// The presentation and relationships parts are authoritative when both exist
/// and declare at least one `<p:sldId>`, whether or not the entries resolve: an
/// unresolved entry becomes an unresolved [`SlideSlot`] so the caller can keep
/// it as a placeholder. Only a presentation part that declares no slides at all
/// (missing part, no `<p:sldIdLst>`, or an empty list) falls back to numeric
/// `ppt/slides/slideN.xml` ordering, which also covers hand-made decks without a
/// presentation part.
fn slide_parts(archive: &ZipArchive) -> Result<Vec<SlideSlot>> {
    if archive.has_entry(PRESENTATION_PART) && archive.has_entry(PRESENTATION_RELS_PART) {
        let ordered = presentation_slide_parts(archive)?;
        if !ordered.is_empty() {
            return Ok(ordered);
        }
    }
    Ok(numeric_slide_parts(archive)
        .into_iter()
        .map(SlideSlot::resolved)
        .collect())
}

/// Resolves `<p:sldIdLst>` through the presentation relationships.
///
/// Entries the relationship chain cannot resolve are returned as unresolved
/// slots, never dropped.
fn presentation_slide_parts(archive: &ZipArchive) -> Result<Vec<SlideSlot>> {
    let presentation = read_part(archive, PRESENTATION_PART)?;
    let Some(list) = presentation
        .root()
        .and_then(|root| root.find_first("sldIdLst"))
    else {
        return Ok(Vec::new());
    };
    let relationships = read_relationships(archive)?;
    let mut slots = Vec::new();
    for (position, entry) in list.find_all("sldId").iter().enumerate() {
        let index = position + 1;
        let Some(id) = entry
            .attr("r:id")
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            slots.push(SlideSlot::unresolved(
                NO_RELATIONSHIP_REFERENCE,
                format!("slide {index}: <p:sldId> has no r:id"),
            ));
            continue;
        };
        let Some(target) = relationships.get(id) else {
            slots.push(SlideSlot::unresolved(
                id,
                format!("slide {index}: {id} is not mapped by {PRESENTATION_RELS_PART}"),
            ));
            continue;
        };
        let part = resolve_part(PRESENTATION_DIR, target);
        if !is_slide_part(&part) {
            slots.push(SlideSlot::unresolved(
                id,
                format!("slide {index}: {id} target {target} is not a slide part"),
            ));
            continue;
        }
        if !archive.has_entry(&part) {
            slots.push(SlideSlot::unresolved(
                id,
                format!("slide {index}: {id} target {target} is missing from the archive"),
            ));
            continue;
        }
        slots.push(SlideSlot::resolved(part));
    }
    Ok(slots)
}

/// Collects `ppt/slides/slideN.xml` parts, ordered by `N` then name.
fn numeric_slide_parts(archive: &ZipArchive) -> Vec<String> {
    let mut numbered: Vec<(usize, String)> = archive
        .entry_names()
        .into_iter()
        .filter_map(|name| slide_number(name).map(|number| (number, name.to_owned())))
        .collect();
    numbered.sort_by(|(left_number, left_name), (right_number, right_name)| {
        left_number
            .cmp(right_number)
            .then_with(|| left_name.cmp(right_name))
    });
    numbered.into_iter().map(|(_, name)| name).collect()
}

/// Returns the `N` of a `ppt/slides/slideN.xml` entry name.
fn slide_number(name: &str) -> Option<usize> {
    let digits = name
        .strip_prefix(SLIDES_DIR)?
        .strip_suffix(".xml")?
        .strip_prefix("slide")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Returns the archive path of a slide's notes part.
///
/// The name-number convention is used directly (`slide3.xml` →
/// `notesSlide3.xml`) rather than following the slide's own relationships part.
fn notes_part(slide_part: &str) -> Option<String> {
    slide_number(slide_part).map(|number| format!("{NOTES_SLIDES_DIR}notesSlide{number}.xml"))
}

/// Reads one package part and parses it, naming the part on failure.
fn read_part(archive: &ZipArchive, part: &str) -> Result<XmlDocument> {
    let markup = archive.read_text(part)?;
    parse(&markup).map_err(|error| error.with_context(part))
}

/// Reads the presentation relationship map (`Id` → `Target`).
fn read_relationships(archive: &ZipArchive) -> Result<HashMap<String, String>> {
    let document = read_part(archive, PRESENTATION_RELS_PART)?;
    let mut relationships = HashMap::new();
    let Some(root) = document.root() else {
        return Ok(relationships);
    };
    for relationship in root.find_all("Relationship") {
        if let (Some(id), Some(target)) = (relationship.attr("Id"), relationship.attr("Target")) {
            relationships.insert(id.trim().to_owned(), target.trim().to_owned());
        }
    }
    Ok(relationships)
}

/// Resolves a relationship target against the presentation directory.
fn resolve_part(base_dir: &str, target: &str) -> String {
    match target.strip_prefix('/') {
        Some(absolute) => normalize_part(absolute),
        None => normalize_part(&format!("{base_dir}/{target}")),
    }
}

/// Collapses `.`/`..` segments and duplicate slashes in a package path.
fn normalize_part(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    segments.join("/")
}

/// Returns `true` for a resolved part inside [`SLIDES_DIR`] ending in `.xml`.
fn is_slide_part(part: &str) -> bool {
    part.strip_prefix(SLIDES_DIR).is_some_and(|rest| {
        !rest.is_empty()
            && !rest.contains('/')
            && Path::new(rest)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("xml"))
    })
}

/// Parses one slide part and its notes into a [`PptxSlide`].
fn parse_slide(archive: &ZipArchive, part: &str, index: usize) -> Result<PptxSlide> {
    let document = read_part(archive, part)?;
    let mut lines = Vec::new();
    let mut placeholder_title = None;
    if let Some(tree) = shape_tree(&document) {
        collect_shapes(tree, &mut lines, &mut placeholder_title);
    }
    let text = lines.join("\n");
    let title = if text.is_empty() {
        None
    } else {
        placeholder_title.or_else(|| lines.first().cloned())
    };
    let (notes, has_notes) = read_notes(archive, part)?;
    Ok(PptxSlide {
        index,
        text,
        title,
        notes,
        has_notes,
        missing_part: None,
    })
}

/// Builds the placeholder slide that keeps an unresolved slot's deck position.
fn placeholder_slide(index: usize, reference: &str) -> PptxSlide {
    PptxSlide {
        index,
        text: format!("! missing slide part: {reference}"),
        title: None,
        notes: None,
        has_notes: false,
        missing_part: Some(reference.to_owned()),
    }
}

/// Returns a part's `<p:cSld>/<p:spTree>`, when it has one.
fn shape_tree(document: &XmlDocument) -> Option<&XmlElement> {
    document
        .root()?
        .find_first("cSld")
        .and_then(|common| common.find_first("spTree"))
}

/// Walks shapes in document order, appending body lines and capturing the title.
fn collect_shapes(element: &XmlElement, lines: &mut Vec<String>, title: &mut Option<String>) {
    for child in &element.children {
        match local_name(&child.name) {
            "sp" => {
                let shape_lines = text_body_lines(child);
                if title.is_none() && is_title_placeholder(child) {
                    *title = shape_lines.first().cloned();
                }
                lines.extend(shape_lines);
            }
            "graphicFrame" => {
                if let Some(markdown) = table_markdown(child) {
                    lines.push(markdown);
                }
            }
            "grpSp" => collect_shapes(child, lines, title),
            _ => {}
        }
    }
}

/// Collects one shape's `<a:p>` lines, trimmed and without empty paragraphs.
fn text_body_lines(shape: &XmlElement) -> Vec<String> {
    let Some(body) = shape.find_first("txBody") else {
        return Vec::new();
    };
    body.children_named("p")
        .iter()
        .map(|paragraph| paragraph_text(paragraph).trim().to_owned())
        .filter(|line| !line.is_empty())
        .collect()
}

/// Returns the placeholder type declared by a shape's `p:nvSpPr/p:nvPr/p:ph`.
fn placeholder_type(shape: &XmlElement) -> Option<&str> {
    shape
        .find_first("nvSpPr")
        .and_then(|properties| properties.find_first("ph"))
        .and_then(|placeholder| placeholder.attr("type"))
        .map(str::trim)
}

/// Returns `true` when the shape is the slide's title placeholder.
fn is_title_placeholder(shape: &XmlElement) -> bool {
    placeholder_type(shape).is_some_and(|kind| TITLE_PLACEHOLDERS.contains(&kind))
}

/// Returns `true` when the shape is a slide-number placeholder.
fn is_slide_number_placeholder(shape: &XmlElement) -> bool {
    placeholder_type(shape) == Some("sldNum")
}

/// Concatenates a paragraph's `<a:t>` runs, mapping `<a:br>` to a newline.
fn paragraph_text(element: &XmlElement) -> String {
    let mut text = String::new();
    collect_paragraph_text(element, &mut text);
    text
}

/// Appends `a:t`/`a:br` content of `element` and its descendants.
fn collect_paragraph_text(element: &XmlElement, text: &mut String) {
    for child in &element.children {
        match local_name(&child.name) {
            "t" => text.push_str(&child.text),
            "br" => text.push('\n'),
            // A nested table is a block of its own; never inline its cells here.
            "tbl" => {}
            _ => collect_paragraph_text(child, text),
        }
    }
}

/// Renders the first `<a:tbl>` of a graphic frame as a Markdown table.
fn table_markdown(frame: &XmlElement) -> Option<String> {
    let table = frame.find_first("tbl")?;
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

/// Collects a table cell's `<a:p>` lines, joined by newlines and trimmed.
fn cell_text(cell: &XmlElement) -> String {
    cell.find_all("p")
        .iter()
        .map(|paragraph| paragraph_text(paragraph).trim().to_owned())
        .filter(|line| !line.is_empty())
        .collect::<Vec<String>>()
        .join("\n")
}

/// Reads a slide's speaker notes; `has_notes` reports part presence.
fn read_notes(archive: &ZipArchive, part: &str) -> Result<(Option<String>, bool)> {
    let Some(notes_part) = notes_part(part) else {
        return Ok((None, false));
    };
    if !archive.has_entry(&notes_part) {
        return Ok((None, false));
    }
    let document = read_part(archive, &notes_part)?;
    let mut lines = Vec::new();
    if let Some(tree) = shape_tree(&document) {
        collect_notes_shapes(tree, &mut lines);
    }
    let text = lines.join("\n");
    let notes = (!text.trim().is_empty()).then_some(text);
    Ok((notes, true))
}

/// Walks the notes body, skipping the slide-number placeholder.
fn collect_notes_shapes(element: &XmlElement, lines: &mut Vec<String>) {
    for child in &element.children {
        match local_name(&child.name) {
            "sp" => {
                if !is_slide_number_placeholder(child) {
                    lines.extend(text_body_lines(child));
                }
            }
            "grpSp" => collect_notes_shapes(child, lines),
            _ => {}
        }
    }
}

/// Joins the per-slide Markdown blocks into the document text.
fn render_document(slides: &[PptxSlide]) -> String {
    let mut sections = Vec::with_capacity(slides.len());
    for slide in slides {
        let mut blocks = vec![format!("## Slide {}", slide.index)];
        if !slide.text.trim().is_empty() {
            blocks.push(slide.text.clone());
        }
        if let Some(notes) = &slide.notes {
            blocks.push("### Notes".to_owned());
            blocks.push(notes.clone());
        }
        sections.push(blocks.join("\n"));
    }
    sections.join("\n")
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
    document
        .root()
        .and_then(|root| root.find_first("title"))
        .map(XmlElement::text_content)
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
