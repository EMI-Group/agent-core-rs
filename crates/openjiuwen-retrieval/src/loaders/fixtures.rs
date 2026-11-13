//! In-memory byte fixtures for the binary document loaders.
//!
//! These builders emit the smallest archives the [`crate::parsing`] readers
//! accept, so loader tests, integration tests, and examples need no office
//! suite, no compression library, and no temporary files: every archive stores
//! its entries uncompressed ([`ZipWriter::add_stored`]) and the `PDF` fixture
//! uses an uncompressed content stream.
//!
//! The builders are *fixtures*, not documents: they only cover the structures
//! the parsers read (a Word body, workbook parts, slides, body paragraphs) and
//! never re-implement parsing. Text is XML-escaped, and the only structural
//! convention callers must know is that `xlsx_workbook` rows and
//! `pptx_slides` paragraphs are plain strings without escaping.
//!
//! ```
//! # use openjiuwen_retrieval::loaders::fixtures::{build_zip, docx_document};
//! # fn example() -> openjiuwen_common::error::Result<()> {
//! let archive = build_zip(&[("word/document.xml", "<w:document/>")])?;
//! assert!(!archive.is_empty());
//! let docx = docx_document(&["Intro", "Body"])?;
//! assert!(!docx.is_empty());
//! # Ok(())
//! # }
//! ```

use std::fmt::Write as _;

use openjiuwen_common::error::Result;

use crate::parsing::ZipWriter;

/// `WordprocessingML` namespace every generated `DOCX` part declares.
const WORD_NAMESPACE: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
/// `SpreadsheetML` namespace every generated `XLSX` part declares.
const SHEET_NAMESPACE: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
/// Office relationship namespace shared by `XLSX` and `PPTX` parts.
const RELATIONSHIP_NAMESPACE: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
/// Package relationship namespace used by `_rels` parts.
const PACKAGE_RELATIONSHIP_NAMESPACE: &str =
    "http://schemas.openxmlformats.org/package/2006/relationships";
/// `PresentationML` namespace used by `PPTX` slide parts.
const PRESENTATION_NAMESPACE: &str = "http://schemas.openxmlformats.org/presentationml/2006/main";
/// `DrawingML` namespace used by `PPTX` shape text.
const DRAWING_NAMESPACE: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
/// `OpenDocument` office namespace used by `ODT` parts.
const OFFICE_NAMESPACE: &str = "urn:oasis:names:tc:opendocument:xmlns:office:1.0";
/// `OpenDocument` text namespace used by `ODT` body markup.
const TEXT_NAMESPACE: &str = "urn:oasis:names:tc:opendocument:xmlns:text:1.0";

/// Builds a `ZIP` archive from `(entry name, UTF-8 body)` pairs.
///
/// Entries are stored, never deflated, and keep the order they are given in.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] for an empty or
/// over-long entry name, more than `65_535` entries, or an archive that no
/// longer fits the 32-bit fields of a non-ZIP64 archive.
pub fn build_zip(entries: &[(&str, &str)]) -> Result<Vec<u8>> {
    let mut writer = ZipWriter::new();
    for (name, body) in entries {
        writer.add_stored(*name, body.as_bytes().to_vec())?;
    }
    writer.finish()
}

/// Wraps `body` in a minimal file-level `PDF` document.
///
/// `body` is spliced verbatim, so callers pass complete indirect objects, for
/// example `plain_stream("BT (Hello) Tj ET")`.
#[must_use]
pub fn pdf_document(body: &str) -> Vec<u8> {
    format!("%PDF-1.7\n{body}\n%%EOF\n").into_bytes()
}

/// Builds one uncompressed `PDF` content-stream object holding `content`.
#[must_use]
pub fn plain_stream(content: &str) -> String {
    format!(
        "1 0 obj\n<< /Length {} >>\nstream\n{content}\nendstream\nendobj",
        content.len()
    )
}

/// Builds a `DOCX` archive whose body holds one paragraph per entry.
///
/// The archive has no `docProps/core.xml` part, so the reader reports no title;
/// heading and table markup is out of scope for this fixture.
///
/// # Errors
///
/// Returns the [`build_zip`] errors.
pub fn docx_document(paragraphs: &[&str]) -> Result<Vec<u8>> {
    let mut body = String::new();
    for paragraph in paragraphs {
        let _ = write!(
            body,
            "<w:p><w:r><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>",
            escape_xml(paragraph)
        );
    }
    let document = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <w:document xmlns:w=\"{WORD_NAMESPACE}\"><w:body>{body}</w:body></w:document>"
    );
    build_zip(&[("word/document.xml", document.as_str())])
}

/// Builds an `XLSX` workbook with one worksheet per sheet.
///
/// Each sheet is `(sheet name, rows)`; a row is one string whose cells are
/// separated by commas and trimmed, so `"Name,City"` is a two-cell header row
/// and an empty cell between two commas leaves a gap. Cell text goes through
/// `xl/sharedStrings.xml`, and the first non-empty row is the header row — the
/// convention [`crate::parsing::XlsxParser`] reads. An empty `sheets` slice
/// yields a valid workbook with no sheets.
///
/// # Errors
///
/// Returns the [`build_zip`] errors.
pub fn xlsx_workbook(sheets: &[(&str, &[&str])]) -> Result<Vec<u8>> {
    let mut sheet_entries = String::new();
    let mut relationship_entries = String::new();
    let mut parts: Vec<(String, String)> = Vec::new();
    let mut shared_strings: Vec<String> = Vec::new();

    for (position, (name, rows)) in sheets.iter().enumerate() {
        let number = position + 1;
        let _ = write!(
            sheet_entries,
            "<sheet name=\"{}\" sheetId=\"{number}\" r:id=\"rId{number}\"/>",
            escape_attribute(name)
        );
        let _ = write!(
            relationship_entries,
            "<Relationship Id=\"rId{number}\" Type=\"worksheet\" \
             Target=\"worksheets/sheet{number}.xml\"/>"
        );
        let mut sheet_data = String::new();
        for (row_position, row) in rows.iter().enumerate() {
            let row_number = row_position + 1;
            let _ = write!(sheet_data, "<row r=\"{row_number}\">");
            for (column_position, cell) in row.split(',').map(str::trim).enumerate() {
                if cell.is_empty() {
                    continue;
                }
                let reference = format!("{}{row_number}", column_name(column_position));
                let index = shared_string_index(&mut shared_strings, cell);
                let _ = write!(
                    sheet_data,
                    "<c r=\"{reference}\" t=\"s\"><v>{index}</v></c>"
                );
            }
            sheet_data.push_str("</row>");
        }
        parts.push((
            format!("xl/worksheets/sheet{number}.xml"),
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                 <worksheet xmlns=\"{SHEET_NAMESPACE}\"><sheetData>{sheet_data}</sheetData>\
                 </worksheet>"
            ),
        ));
    }

    let workbook = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <workbook xmlns=\"{SHEET_NAMESPACE}\" xmlns:r=\"{RELATIONSHIP_NAMESPACE}\">\
         <sheets>{sheet_entries}</sheets></workbook>"
    );
    let relationships = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <Relationships xmlns=\"{PACKAGE_RELATIONSHIP_NAMESPACE}\">\
         {relationship_entries}</Relationships>"
    );
    let mut shared = String::new();
    for value in &shared_strings {
        let _ = write!(shared, "<si><t>{}</t></si>", escape_xml(value));
    }
    let shared_strings = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <sst xmlns=\"{SHEET_NAMESPACE}\" count=\"{count}\" uniqueCount=\"{count}\">{shared}</sst>",
        count = shared_strings.len()
    );

    let mut entries: Vec<(&str, &str)> = vec![
        ("xl/workbook.xml", workbook.as_str()),
        ("xl/_rels/workbook.xml.rels", relationships.as_str()),
        ("xl/sharedStrings.xml", shared_strings.as_str()),
    ];
    entries.extend(
        parts
            .iter()
            .map(|(name, body)| (name.as_str(), body.as_str())),
    );
    build_zip(&entries)
}

/// Builds a `PPTX` archive with one slide per entry.
///
/// Each slide is `(title, paragraphs)`: the title becomes a title-placeholder
/// shape and every paragraph becomes one body line. Slides are linked from
/// `ppt/presentation.xml` in the given order and carry no speaker notes, so a
/// reader reports `has_notes == false`.
///
/// # Errors
///
/// Returns the [`build_zip`] errors.
pub fn pptx_slides(slides: &[(&str, &[&str])]) -> Result<Vec<u8>> {
    let mut slide_ids = String::new();
    let mut relationship_entries = String::new();
    let mut parts: Vec<(String, String)> = Vec::new();

    for (position, (title, paragraphs)) in slides.iter().enumerate() {
        let number = position + 1;
        let _ = write!(
            slide_ids,
            "<p:sldId id=\"{}\" r:id=\"rId{number}\"/>",
            255 + number
        );
        let _ = write!(
            relationship_entries,
            "<Relationship Id=\"rId{number}\" Type=\"slide\" Target=\"slides/slide{number}.xml\"/>"
        );
        parts.push((
            format!("ppt/slides/slide{number}.xml"),
            slide_part(title, paragraphs),
        ));
    }

    let presentation = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <p:presentation xmlns:p=\"{PRESENTATION_NAMESPACE}\" \
         xmlns:r=\"{RELATIONSHIP_NAMESPACE}\"><p:sldIdLst>{slide_ids}</p:sldIdLst>\
         </p:presentation>"
    );
    let relationships = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <Relationships xmlns=\"{PACKAGE_RELATIONSHIP_NAMESPACE}\">\
         {relationship_entries}</Relationships>"
    );

    let mut entries: Vec<(&str, &str)> = vec![
        ("ppt/presentation.xml", presentation.as_str()),
        ("ppt/_rels/presentation.xml.rels", relationships.as_str()),
    ];
    entries.extend(
        parts
            .iter()
            .map(|(name, body)| (name.as_str(), body.as_str())),
    );
    build_zip(&entries)
}

/// Builds an `ODT` archive whose body holds one paragraph per entry.
///
/// Only `content.xml` is written: there is no `meta.xml`, so the reader reports
/// no title, and no headings or tables are emitted.
///
/// # Errors
///
/// Returns the [`build_zip`] errors.
pub fn odt_document(paragraphs: &[&str]) -> Result<Vec<u8>> {
    let mut body = String::new();
    for paragraph in paragraphs {
        let _ = write!(body, "<text:p>{}</text:p>", escape_xml(paragraph));
    }
    let content = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <office:document-content xmlns:office=\"{OFFICE_NAMESPACE}\" \
         xmlns:text=\"{TEXT_NAMESPACE}\"><office:body><office:text>{body}</office:text>\
         </office:body></office:document-content>"
    );
    build_zip(&[("content.xml", content.as_str())])
}

/// Renders one `PPTX` slide part with a title shape and one body shape.
fn slide_part(title: &str, paragraphs: &[&str]) -> String {
    let mut shapes = format!(
        "<p:sp><p:nvSpPr><p:nvPr><p:ph type=\"title\"/></p:nvPr></p:nvSpPr><p:txBody>\
         <a:bodyPr/><a:p><a:r><a:t>{}</a:t></a:r></a:p></p:txBody></p:sp>",
        escape_xml(title)
    );
    if !paragraphs.is_empty() {
        let mut body = String::new();
        for paragraph in paragraphs {
            let _ = write!(
                body,
                "<a:p><a:r><a:t>{}</a:t></a:r></a:p>",
                escape_xml(paragraph)
            );
        }
        let _ = write!(
            shapes,
            "<p:sp><p:nvSpPr><p:nvPr><p:ph idx=\"1\"/></p:nvPr></p:nvSpPr><p:txBody>\
             <a:bodyPr/>{body}</p:txBody></p:sp>"
        );
    }
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <p:sld xmlns:p=\"{PRESENTATION_NAMESPACE}\" xmlns:a=\"{DRAWING_NAMESPACE}\">\
         <p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>"
    )
}

/// Returns the shared-string index of `value`, appending it on first use.
fn shared_string_index(shared: &mut Vec<String>, value: &str) -> usize {
    if let Some(index) = shared.iter().position(|existing| existing == value) {
        index
    } else {
        shared.push(value.to_owned());
        shared.len() - 1
    }
}

/// Returns the `A1`-style column name of a zero-based column position.
///
/// Positions `0..=25` map to `A..=Z`, `26` to `AA`, and so on.
fn column_name(position: usize) -> String {
    let mut name = String::new();
    let mut position = position;
    loop {
        let letter = u8::try_from(position % 26).unwrap_or(0);
        name.insert(0, char::from(b'A' + letter));
        if position < 26 {
            return name;
        }
        position = position / 26 - 1;
    }
}

/// Escapes the XML metacharacters that can appear in element text.
fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escapes the XML metacharacters that can appear in a double-quoted attribute.
fn escape_attribute(text: &str) -> String {
    escape_xml(text).replace('"', "&quot;")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::parsing::{DocxParser, OdtParser, PdfParser, PptxParser, XlsxParser, ZipArchive};

    #[test]
    fn build_zip_writes_stored_entries_in_order() {
        let archive =
            ZipArchive::from_bytes(build_zip(&[("a.txt", "first"), ("b.txt", "second")]).unwrap())
                .unwrap();
        assert_eq!(archive.entry_names(), vec!["a.txt", "b.txt"]);
        assert_eq!(archive.read_text("a.txt").unwrap(), "first");
        // Stored entries, never deflated.
        assert_eq!(
            archive.entry("a.txt").unwrap().compression_method,
            crate::parsing::zip::COMPRESSION_METHOD_STORED
        );
        assert!(ZipArchive::from_bytes(build_zip(&[]).unwrap())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn pdf_fixture_parses_to_page_text() {
        let pdf = pdf_document(&plain_stream("BT (Fixture text) Tj ET"));
        let document = PdfParser::new().parse_bytes(&pdf).unwrap();
        assert_eq!(document.text, "Fixture text");
        assert!(document.has_text);
        assert_eq!(document.page_count, 1);
        assert!(plain_stream("x").starts_with("1 0 obj"));
        assert!(plain_stream("x").contains("/Length 1"));
    }

    #[test]
    fn docx_fixture_parses_into_paragraphs() {
        let document = DocxParser::new()
            .parse_bytes(&docx_document(&["Intro", "Body & <text>"]).unwrap())
            .unwrap();
        assert_eq!(document.text, "Intro\nBody & <text>");
        assert_eq!(document.paragraph_count, 2);
        assert_eq!(document.table_count, 0);
        assert_eq!(document.title, None);
    }

    #[test]
    fn xlsx_fixture_exposes_shared_strings_and_gapped_cells() {
        let workbook = XlsxParser::new()
            .parse_bytes(
                &xlsx_workbook(&[
                    ("People", &["Name,City", "alice,paris", "bob,"]),
                    ("Empty", &[]),
                ])
                .unwrap(),
            )
            .unwrap();
        assert_eq!(workbook.sheet_names(), vec!["People", "Empty"]);
        let sheet = workbook.sheet(0usize).unwrap();
        assert_eq!(sheet.table.headers, vec!["Name", "City"]);
        assert_eq!(
            sheet.table.rows,
            vec![vec!["alice", "paris"], vec!["bob", ""]]
        );
    }

    #[test]
    fn xlsx_fixture_supports_more_than_twenty_six_columns() {
        let mut header = String::new();
        for column in 0..28 {
            let _ = write!(header, "C{column},");
        }
        let header = header.trim_end_matches(',');
        let worksheet = XlsxParser::new()
            .parse_bytes(&xlsx_workbook(&[("Wide", &[header])]).unwrap())
            .unwrap();
        let sheet = worksheet.sheet(0usize).unwrap();
        assert_eq!(sheet.table.headers.len(), 28);
        assert_eq!(sheet.table.headers[26], "C26");
    }

    #[test]
    fn xlsx_fixture_accepts_a_workbook_without_sheets() {
        let workbook = XlsxParser::new()
            .parse_bytes(&xlsx_workbook(&[]).unwrap())
            .unwrap();
        assert!(workbook.is_empty());
    }

    #[test]
    fn pptx_fixture_parses_titles_paragraphs_and_order() {
        let presentation = PptxParser::new()
            .parse_bytes(
                &pptx_slides(&[
                    ("Rust rewrite", &["Agenda", "Offline"]),
                    ("Closing & <thanks>", &[]),
                ])
                .unwrap(),
            )
            .unwrap();
        assert_eq!(presentation.slide_count, 2);
        assert_eq!(
            presentation.text,
            "## Slide 1\nRust rewrite\nAgenda\nOffline\n## Slide 2\nClosing & <thanks>"
        );
        assert_eq!(
            presentation.slide(1).unwrap().title.as_deref(),
            Some("Rust rewrite")
        );
        assert_eq!(
            presentation.slide(1).unwrap().text,
            "Rust rewrite\nAgenda\nOffline"
        );
        assert!(!presentation.slide(1).unwrap().has_notes);
        assert_eq!(
            presentation.slide(2).unwrap().title.as_deref(),
            Some("Closing & <thanks>")
        );
        assert_eq!(presentation.title, None);
    }

    #[test]
    fn odt_fixture_parses_into_paragraphs() {
        let document = OdtParser::new()
            .parse_bytes(&odt_document(&["Intro", "Body & <text>"]).unwrap())
            .unwrap();
        assert_eq!(document.text(), "Intro\nBody & <text>");
        assert_eq!(document.paragraph_count, 2);
        assert_eq!(document.table_count, 0);
        assert_eq!(document.title, None);
        assert!(!document.is_empty());
    }

    #[test]
    fn fixture_writers_escape_xml_metacharacters() {
        assert_eq!(escape_xml("a & b <c>"), "a &amp; b &lt;c&gt;");
        assert_eq!(
            escape_attribute(r#"a "quoted" & <b>"#),
            "a &quot;quoted&quot; &amp; &lt;b&gt;"
        );
        assert_eq!(column_name(0), "A");
        assert_eq!(column_name(25), "Z");
        assert_eq!(column_name(26), "AA");
        assert_eq!(column_name(27), "AB");
        assert_eq!(column_name(51), "AZ");
        assert_eq!(column_name(52), "BA");
    }
}
