#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::cast_possible_truncation
)]
//! Integration tests for the DOCX, XLSX, and PDF parsers.
//!
//! Every fixture is built in this file: `ZIP` archives are assembled byte by byte
//! (`build_zip`, stored entries only — the deflated path is covered by
//! `tests/parsing_zip.rs`) and the `PDF` `FlateDecode` stream is produced by
//! `zlib_stored` (a `zlib` header, one stored `DEFLATE` block per 64 KiB, and the
//! `Adler-32` trailer). Nothing touches the network or the wall clock.

use std::collections::BTreeSet;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::parsing::{
    adler32, crc32, parse_xml, DocxBlock, DocxParser, PdfParser, XlsxParser, XmlDocument,
};

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Builds a `ZIP` archive with stored (uncompressed) entries.
#[derive(Default)]
struct ZipBuilder {
    /// Local headers plus payloads.
    data: Vec<u8>,
    /// Central-directory records.
    directory: Vec<u8>,
    /// Number of entries added.
    count: u16,
}

impl ZipBuilder {
    /// Appends one stored entry.
    fn add(&mut self, name: &str, payload: &[u8]) {
        let name = name.as_bytes();
        let offset = self.data.len() as u32;
        let crc = crc32(payload);
        let size = payload.len() as u32;
        self.data.extend_from_slice(&0x0403_4B50u32.to_le_bytes());
        self.data.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.data.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.data.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        self.data.extend_from_slice(&0u16.to_le_bytes()); // time
        self.data.extend_from_slice(&0u16.to_le_bytes()); // date
        self.data.extend_from_slice(&crc.to_le_bytes());
        self.data.extend_from_slice(&size.to_le_bytes());
        self.data.extend_from_slice(&size.to_le_bytes());
        self.data
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.data.extend_from_slice(&0u16.to_le_bytes()); // extra
        self.data.extend_from_slice(name);
        self.data.extend_from_slice(payload);

        self.directory
            .extend_from_slice(&0x0201_4B50u32.to_le_bytes());
        self.directory.extend_from_slice(&20u16.to_le_bytes()); // version made by
        self.directory.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // method
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // time
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // date
        self.directory.extend_from_slice(&crc.to_le_bytes());
        self.directory.extend_from_slice(&size.to_le_bytes());
        self.directory.extend_from_slice(&size.to_le_bytes());
        self.directory
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // extra
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // comment
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // disk
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        self.directory.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        self.directory.extend_from_slice(&offset.to_le_bytes());
        self.directory.extend_from_slice(name);
        self.count += 1;
    }

    /// Seals the archive with its central directory and end record.
    fn finish(self) -> Vec<u8> {
        let directory_offset = self.data.len() as u32;
        let directory_size = self.directory.len() as u32;
        let mut bytes = self.data;
        bytes.extend_from_slice(&self.directory);
        bytes.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes()); // this disk
        bytes.extend_from_slice(&0u16.to_le_bytes()); // directory disk
        bytes.extend_from_slice(&self.count.to_le_bytes());
        bytes.extend_from_slice(&self.count.to_le_bytes());
        bytes.extend_from_slice(&directory_size.to_le_bytes());
        bytes.extend_from_slice(&directory_offset.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes()); // comment length
        bytes
    }
}

/// Builds a `ZIP` archive from `(name, contents)` pairs.
fn build_zip(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut builder = ZipBuilder::default();
    for (name, contents) in entries {
        builder.add(name, contents.as_bytes());
    }
    builder.finish()
}

/// Wraps `data` in a `zlib` stream made of stored `DEFLATE` blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut stream = vec![0x78, 0x01];
    let mut chunks = data.chunks(65_535).peekable();
    if chunks.peek().is_none() {
        stream.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    } else {
        while let Some(chunk) = chunks.next() {
            let last = chunks.peek().is_none();
            stream.push(u8::from(last));
            let length = chunk.len() as u16;
            stream.extend_from_slice(&length.to_le_bytes());
            stream.extend_from_slice(&(!length).to_le_bytes());
            stream.extend_from_slice(chunk);
        }
    }
    stream.extend_from_slice(&adler32(data).to_be_bytes());
    stream
}

/// Concatenates PDF objects into one document body.
fn pdf_document(body: &[u8]) -> Vec<u8> {
    let mut document = b"%PDF-1.7\n".to_vec();
    document.extend_from_slice(body);
    document.extend_from_slice(b"\n%%EOF\n");
    document
}

/// Builds one uncompressed PDF content-stream object.
fn plain_stream(content: &str) -> Vec<u8> {
    format!(
        "1 0 obj\n<< /Length {} >>\nstream\n{content}\nendstream\nendobj",
        content.len()
    )
    .into_bytes()
}

/// Builds one `/FlateDecode` PDF content-stream object.
fn flate_stream(content: &str) -> Vec<u8> {
    let compressed = zlib_stored(content.as_bytes());
    let mut object = format!(
        "2 0 obj\n<< /Filter /FlateDecode /Length {} >>\nstream\n",
        compressed.len()
    )
    .into_bytes();
    object.extend_from_slice(&compressed);
    object.extend_from_slice(b"\nendstream\nendobj");
    object
}

// ---------------------------------------------------------------------------
// XML
// ---------------------------------------------------------------------------

#[test]
fn xml_is_exposed_through_the_parsing_module() {
    let document: XmlDocument = parse_xml(
        r#"<w:document xmlns:w="urn:w"><w:body><w:p><w:t>A &amp; B</w:t></w:p></w:body></w:document>"#,
    )
    .unwrap();
    let text = document
        .root()
        .unwrap()
        .find_first("body")
        .unwrap()
        .find_first("t")
        .unwrap();
    assert_eq!(text.text, "A & B");

    let error = parse_xml("<a><b></a>").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("mismatched end tag"), "{error}");
}

// ---------------------------------------------------------------------------
// DOCX
// ---------------------------------------------------------------------------

/// `word/document.xml` fixture: title, paragraph, headings, and a table.
const DOCX_DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Rust rewrite</w:t></w:r></w:p>
    <w:p><w:r><w:t>Intro</w:t><w:tab/><w:t>after tab</w:t><w:br/><w:t>next line</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Notes</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>alice</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>a|b</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>"#;

/// `docProps/core.xml` fixture carrying `dc:title`.
const DOCX_CORE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="urn:cp" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Design notes</dc:title>
</cp:coreProperties>"#;

#[test]
fn docx_renders_headings_paragraphs_tables_and_title() {
    let document = build_zip(&[
        ("word/document.xml", DOCX_DOCUMENT),
        ("docProps/core.xml", DOCX_CORE),
    ]);
    let parsed = DocxParser::new().parse_bytes(&document).unwrap();
    assert_eq!(parsed.title.as_deref(), Some("Design notes"));
    assert_eq!(parsed.paragraph_count, 3);
    assert_eq!(parsed.table_count, 1);
    assert_eq!(
        parsed.blocks,
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
            DocxBlock::Table {
                markdown: "| Name | Notes |\n| --- | --- |\n| alice | a\\|b |".to_owned()
            },
        ]
    );
    assert_eq!(
        parsed.text,
        "# Rust rewrite\nIntro\tafter tab\nnext line\n## Overview\n\
         | Name | Notes |\n| --- | --- |\n| alice | a\\|b |"
    );
    assert!(!parsed.is_empty());
}

#[test]
fn docx_without_core_properties_has_no_title() {
    let document = build_zip(&[("word/document.xml", "<w:document><w:body/></w:document>")]);
    let parsed = DocxParser::new().parse_bytes(&document).unwrap();
    assert_eq!(parsed.title, None);
    assert_eq!(parsed.text, "");
    assert_eq!(parsed.blocks, Vec::<DocxBlock>::new());
    assert_eq!(parsed.paragraph_count, 0);
    assert_eq!(parsed.table_count, 0);
    assert!(parsed.is_empty());
}

#[test]
fn docx_without_document_part_is_an_error() {
    let document = build_zip(&[("docProps/core.xml", DOCX_CORE)]);
    let error = DocxParser::new().parse_bytes(&document).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.to_string().contains("word/document.xml"), "{error}");

    // A corrupt container is reported by the ZIP layer, not swallowed.
    let error = DocxParser::new()
        .parse_bytes(b"PK\x03\x04not a zip")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn docx_parses_from_disk() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("notes.docx");
    std::fs::write(&path, build_zip(&[("word/document.xml", DOCX_DOCUMENT)])).unwrap();
    let parsed = DocxParser::new().parse_path(&path).await.unwrap();
    assert!(parsed.text.starts_with("# Rust rewrite"), "{}", parsed.text);

    let missing = directory.path().join("missing.docx");
    let error = DocxParser::new().parse_path(&missing).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

// ---------------------------------------------------------------------------
// XLSX
// ---------------------------------------------------------------------------

/// Workbook listing two sheets.
const XLSX_WORKBOOK: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="People" sheetId="1" r:id="rId1"/>
    <sheet name="Second" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"#;

/// Relationships: a relative target for the first sheet, an absolute one for the
/// second (both must resolve to `xl/worksheets/…`).
const XLSX_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="worksheet" Target="/xl/worksheets/sheet2.xml"/>
</Relationships>"#;

/// Shared strings with a plain entry and a rich-text entry.
const XLSX_SHARED_STRINGS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="5">
  <si><t>Name</t></si>
  <si><r><t>Ali</t></r><r><t>ce</t></r></si>
  <si><t>Score</t></si>
  <si><t>Notes</t></si>
  <si><t>Extra</t></si>
</sst>"#;

/// First worksheet: shared strings, inline strings, numbers, a boolean, a gap
/// row, and a value in a fourth column.
const XLSX_SHEET1: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="B1" t="s"><v>2</v></c>
      <c r="C1" t="s"><v>3</v></c>
      <c r="D1" t="s"><v>4</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>1</v></c>
      <c r="B2"><v>12.0</v></c>
      <c r="C2" t="inlineStr"><is><t>ok</t></is></c>
    </row>
    <row r="3">
      <c r="B3"><v>7.5</v></c>
      <c r="C3" t="b"><v>0</v></c>
      <c r="D3" t="b"><v>1</v></c>
    </row>
    <row r="5">
      <c r="D5" t="str"><v>wide</v></c>
    </row>
  </sheetData>
</worksheet>"#;

/// Second worksheet: one header row and one data row.
const XLSX_SHEET2: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>Key</t></is></c></row>
    <row r="2"><c r="A2" t="inlineStr"><is><t>v</t></is></c></row>
  </sheetData>
</worksheet>"#;

/// Builds the two-sheet workbook archive.
fn xlsx_archive() -> Vec<u8> {
    build_zip(&[
        ("xl/workbook.xml", XLSX_WORKBOOK),
        ("xl/_rels/workbook.xml.rels", XLSX_RELS),
        ("xl/sharedStrings.xml", XLSX_SHARED_STRINGS),
        ("xl/worksheets/sheet1.xml", XLSX_SHEET1),
        ("xl/worksheets/sheet2.xml", XLSX_SHEET2),
    ])
}

#[test]
fn xlsx_reads_sheets_cells_and_rich_strings() {
    let workbook = XlsxParser::new().parse_bytes(&xlsx_archive()).unwrap();
    assert_eq!(workbook.sheet_names(), vec!["People", "Second"]);
    assert_eq!(workbook.len(), 2);
    assert!(!workbook.is_empty());

    let people = workbook.sheet("People").unwrap();
    assert_eq!(people.index, 0);
    assert_eq!(
        people.table.headers,
        vec!["Name", "Score", "Notes", "Extra"]
    );
    // Rows are padded to the table width, including the gap row at r="4".
    assert_eq!(people.table.rows.len(), 4);
    assert_eq!(people.table.rows[0], vec!["Alice", "12", "ok", ""]);
    assert_eq!(people.table.rows[1], vec!["", "7.5", "false", "true"]);
    assert_eq!(people.table.rows[2], vec!["", "", "", ""]);
    assert_eq!(people.table.rows[3], vec!["", "", "", "wide"]);
    assert_eq!(
        people.table.row_text(0).unwrap(),
        "Name: Alice, Score: 12, Notes: ok, Extra: "
    );

    let second = workbook.sheet(1).unwrap();
    assert_eq!(second.name, "Second");
    assert_eq!(second.table.headers, vec!["Key"]);
    assert_eq!(second.table.rows, vec![vec!["v".to_owned()]]);
    assert!(workbook.sheet("Missing").is_none());
    assert!(workbook.sheet(9).is_none());
}

#[test]
fn xlsx_documents_follow_the_row_and_column_scheme_per_sheet() {
    let workbook = XlsxParser::new().parse_bytes(&xlsx_archive()).unwrap();
    let documents = workbook.to_documents("book");
    let ids: Vec<&str> = documents.iter().map(|doc| doc.id.as_str()).collect();
    assert_eq!(
        ids,
        vec![
            // People: rows (header is row 1, so data starts at 2), then columns.
            "book_s0_r2",
            "book_s0_r3",
            "book_s0_r4",
            "book_s0_r5",
            "book_s0_c0",
            "book_s0_c1",
            "book_s0_c2",
            "book_s0_c3",
            // Second: one row document and one column document.
            "book_s1_r2",
            "book_s1_c0",
        ]
    );

    let first = &documents[0];
    assert_eq!(
        first.metadata.get("source_type"),
        Some(&serde_json::Value::String("row".to_owned()))
    );
    assert_eq!(
        first.metadata.get("row_index"),
        Some(&serde_json::Value::from(2))
    );
    assert_eq!(
        first.metadata.get("sheet_name"),
        Some(&serde_json::Value::String("People".to_owned()))
    );

    let column = &documents[4];
    assert_eq!(
        column.metadata.get("source_type"),
        Some(&serde_json::Value::String("column".to_owned()))
    );
    assert_eq!(
        column.metadata.get("column_name"),
        Some(&serde_json::Value::String("Name".to_owned()))
    );
    assert_eq!(column.text, "Column name: Name. Values: Alice");

    assert_eq!(
        documents[8].metadata.get("sheet_name"),
        Some(&serde_json::Value::String("Second".to_owned()))
    );
    assert_eq!(documents[8].text, "Key: v");
}

#[test]
fn xlsx_falls_back_to_conventional_sheet_paths_without_relationships() {
    let archive = build_zip(&[
        (
            "xl/workbook.xml",
            r#"<workbook><sheets><sheet name="Only" r:id="rId7"/></sheets></workbook>"#,
        ),
        ("xl/worksheets/sheet1.xml", XLSX_SHEET2),
    ]);
    let workbook = XlsxParser::new().parse_bytes(&archive).unwrap();
    assert_eq!(workbook.sheet_names(), vec!["Only"]);
    assert_eq!(workbook.sheets[0].table.headers, vec!["Key"]);
}

#[test]
fn xlsx_missing_parts_are_reported_and_empty_workbooks_are_not() {
    let empty = build_zip(&[("dummy.txt", "x")]);
    let error = XlsxParser::new().parse_bytes(&empty).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.to_string().contains("xl/workbook.xml"), "{error}");

    // A sheet whose target relationship points at an absent part names it.
    let missing_sheet = build_zip(&[
        (
            "xl/workbook.xml",
            r#"<workbook><sheets><sheet name="Gone" r:id="rId1"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<Relationships><Relationship Id="rId1" Target="worksheets/gone.xml"/></Relationships>"#,
        ),
    ]);
    let error = XlsxParser::new().parse_bytes(&missing_sheet).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(
        error.to_string().contains("xl/worksheets/gone.xml"),
        "{error}"
    );

    // No `<sheet>` elements: an empty workbook, not an error.
    let no_sheets = build_zip(&[("xl/workbook.xml", "<workbook><sheets/></workbook>")]);
    let workbook = XlsxParser::new().parse_bytes(&no_sheets).unwrap();
    assert!(workbook.is_empty());
    assert_eq!(workbook.to_documents("book").len(), 0);
}

#[tokio::test]
async fn xlsx_parses_from_disk() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("book.xlsx");
    std::fs::write(&path, xlsx_archive()).unwrap();
    let workbook = XlsxParser::new().parse_path(&path).await.unwrap();
    assert_eq!(workbook.len(), 2);

    let missing = directory.path().join("missing.xlsx");
    let error = XlsxParser::new().parse_path(&missing).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

/// Sanity check that the builders produce archives the reader can enumerate, so a
/// parser failure is never a fixture bug.
#[test]
fn zip_fixtures_are_readable() {
    let archive = openjiuwen_retrieval::parsing::ZipArchive::from_bytes(xlsx_archive()).unwrap();
    let names: BTreeSet<&str> = archive.entry_names().into_iter().collect();
    assert_eq!(names.len(), 5);
    assert!(names.contains("xl/worksheets/sheet1.xml"));
    assert_eq!(archive.read_text("xl/workbook.xml").unwrap(), XLSX_WORKBOOK);
}

// ---------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------

#[test]
fn pdf_reads_uncompressed_text() {
    let document = pdf_document(&plain_stream("BT (Hello) Tj ET"));
    let parsed = PdfParser::new().parse_bytes(&document).unwrap();
    assert_eq!(parsed.text, "Hello");
    assert_eq!(parsed.pages, vec!["Hello"]);
    assert_eq!(parsed.page_count, 1);
    assert!(parsed.has_text);
    assert!(!parsed.incomplete);
    assert!(parsed.skipped_filters.is_empty());
    assert!(parsed.skipped_streams.is_empty());
}

#[test]
fn pdf_reads_tj_arrays_escapes_and_hex_strings() {
    let document = pdf_document(&plain_stream(
        r"BT [(Hel) -300 (lo \(x\) \101)] TJ T* <48656C6C6F> Tj ET",
    ));
    let parsed = PdfParser::new().parse_bytes(&document).unwrap();
    assert_eq!(parsed.pages, vec!["Hel lo (x) A\nHello"]);
}

#[test]
fn pdf_reads_flate_streams_and_keeps_stream_order() {
    let mut body = plain_stream("BT (first) Tj ET");
    body.extend_from_slice(b"\n");
    body.extend_from_slice(&flate_stream("BT (compressed) Tj ET"));
    let parsed = PdfParser::new().parse_bytes(&pdf_document(&body)).unwrap();
    assert_eq!(parsed.pages, vec!["first", "compressed"]);
    assert_eq!(parsed.text, "first\ncompressed");
    assert_eq!(parsed.page_count, 2);

    // The helper really is a zlib stream.
    let compressed = zlib_stored(b"BT (compressed) Tj ET");
    assert_eq!(
        openjiuwen_retrieval::parsing::zlib_decompress(&compressed, 1024).unwrap(),
        b"BT (compressed) Tj ET"
    );
}

#[test]
fn pdf_skips_streams_with_unsupported_filters() {
    let mut body = plain_stream("BT (kept) Tj ET");
    body.extend_from_slice(b"\n3 0 obj\n<< /Filter /DCTDecode /Length 8 >>\nstream\n\x01\x02\x03\x04\x05\x06\x07\x08\nendstream\nendobj\n");
    body.extend_from_slice(b"\n");
    body.extend_from_slice(&plain_stream("BT (also kept) Tj ET"));
    let parsed = PdfParser::new().parse_bytes(&pdf_document(&body)).unwrap();
    assert_eq!(parsed.pages, vec!["kept", "also kept"]);
    // The skipped image stream is reported instead of disappearing.
    assert!(parsed.incomplete);
    assert_eq!(parsed.skipped_filters, vec!["DCTDecode"]);
    assert_eq!(parsed.skipped_streams.len(), 1);
    assert!(
        parsed.skipped_streams[0].contains("stream 1: image payload filter DCTDecode"),
        "unexpected record: {}",
        parsed.skipped_streams[0]
    );
}

#[test]
fn pdf_encrypted_and_object_streams_are_unsupported() {
    let encrypted = pdf_document(b"trailer\n<< /Size 3 /Encrypt 2 0 R >>");
    let error = PdfParser::new().parse_bytes(&encrypted).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("encrypted"), "{error}");

    let object_stream = pdf_document(
        b"1 0 obj\n<< /Type /ObjStm /N 1 /First 4 >>\nstream\n2 0\n<<>>\nendstream\nendobj",
    );
    let error = PdfParser::new().parse_bytes(&object_stream).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ObjStm"), "{error}");
}

#[test]
fn pdf_without_text_is_not_an_error() {
    let scanned = pdf_document(
        b"4 0 obj\n<< /Filter /DCTDecode /Length 4 >>\nstream\n\xff\xd8\xff\xd9\nendstream\nendobj",
    );
    let parsed = PdfParser::new().parse_bytes(&scanned).unwrap();
    assert!(!parsed.has_text);
    assert_eq!(parsed.text, "");
    assert_eq!(parsed.pages, Vec::<String>::new());
    assert_eq!(parsed.page_count, 0);
    assert!(parsed.is_empty());
    assert!(parsed.incomplete);
    assert!(parsed.skipped_filters.contains(&"DCTDecode".to_owned()));
    assert_eq!(parsed.skipped_streams.len(), 1);
}

#[tokio::test]
async fn pdf_parses_from_disk() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("doc.pdf");
    std::fs::write(&path, pdf_document(&flate_stream("BT (from disk) Tj ET"))).unwrap();
    let parsed = PdfParser::new().parse_path(&path).await.unwrap();
    assert_eq!(parsed.text, "from disk");

    let missing = directory.path().join("missing.pdf");
    let error = PdfParser::new().parse_path(&missing).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}
