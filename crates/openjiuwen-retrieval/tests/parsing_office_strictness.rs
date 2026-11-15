#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Cross-format integration tests for the three office-reader silent-loss paths.
//!
//! Each of the three readers used to lose information without saying so; the
//! public API now reports it:
//!
//! - **`XLSX`** — a `t="s"` cell whose shared-string index cannot be resolved is
//!   a typed `Error::Retrieval` naming the worksheet part, the cell, the index
//!   and the size of the table, instead of decoding to an empty string.
//! - **`PPTX`** — a `<p:sldId>` the presentation relationships cannot resolve
//!   keeps its deck position as a placeholder slide, and every loss is listed in
//!   `PptxDocument::problems`.
//! - **`ODT`** — body elements that contribute no text are listed, sorted, with
//!   an occurrence count in `OdtDocument::unsupported_elements`.
//!
//! Containers are built in memory with the crate's own stored-entry `ZipWriter`
//! (or the public `loaders::fixtures` builders), so the suite needs no office
//! fixture files and no temporary files. The per-module unit tests cover the
//! internal helpers; this file is the public-API, cross-format statement of the
//! same three fixes, with deterministic assertions only.
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_retrieval::loaders::fixtures;
use openjiuwen_retrieval::parsing::{
    OdtDocument, OdtParser, PptxDocument, PptxParser, XlsxParser, ZipWriter, CONTENT_PART,
    PRESENTATION_PART, PRESENTATION_RELS_PART, SHARED_STRINGS_PART,
};

// ---------------------------------------------------------------------------
// Shared fixture plumbing
// ---------------------------------------------------------------------------

/// Builds an archive from `(part name, UTF-8 body)` pairs.
///
/// Entries are stored (never deflated) and keep the order they are given in, so
/// the fixtures hold exactly the parts named by each test.
fn archive(parts: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = ZipWriter::new();
    for (name, body) in parts {
        writer.add_stored(*name, body.as_bytes().to_vec()).unwrap();
    }
    writer.finish().unwrap()
}

/// Asserts that `error` has `kind` and that its rendered message names every
/// fragment in `fragments`.
///
/// Fragments are stable substrings (cell references, indices, entry counts,
/// part paths) rather than whole sentences, so the tests pin the diagnostic
/// content without freezing the wording around it.
fn assert_error(error: &Error, kind: ErrorKind, fragments: &[&str]) {
    assert_eq!(error.kind(), kind, "{error}");
    let message = error.to_string();
    for fragment in fragments {
        assert!(
            message.contains(fragment),
            "{fragment:?} missing from {message}"
        );
    }
}

// ---------------------------------------------------------------------------
// XLSX: an unresolvable shared-string reference is an error, not a blank cell
// ---------------------------------------------------------------------------

/// `xl/workbook.xml`: one sheet wired to `rId1`.
const XLSX_WORKBOOK: &str = concat!(
    r#"<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
    r#"<sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
);

/// `xl/_rels/workbook.xml.rels`: `rId1` is `xl/worksheets/sheet1.xml`.
const XLSX_RELS: &str = concat!(
    r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    r#"<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>"#,
    r#"</Relationships>"#,
);

/// Archive path of the single worksheet these fixtures point at.
const XLSX_SHEET_PART: &str = "xl/worksheets/sheet1.xml";

/// Builds a one-sheet workbook whose `sheetData` holds `sheet_data`.
///
/// `shared` is the body of `xl/sharedStrings.xml`; `None` omits the part
/// entirely, which is the legitimate inline-string-only workbook shape.
fn xlsx_archive(sheet_data: &str, shared: Option<&str>) -> Vec<u8> {
    let worksheet = format!("<worksheet><sheetData>{sheet_data}</sheetData></worksheet>");
    let mut parts: Vec<(&str, &str)> = vec![
        ("xl/workbook.xml", XLSX_WORKBOOK),
        ("xl/_rels/workbook.xml.rels", XLSX_RELS),
        (XLSX_SHEET_PART, worksheet.as_str()),
    ];
    if let Some(shared) = shared {
        parts.push((SHARED_STRINGS_PART, shared));
    }
    archive(&parts)
}

#[test]
fn xlsx_shared_string_lookup_miss_names_the_cell() {
    // The sheet has an inline-string header (so the workbook is otherwise
    // valid) and one `t="s"` cell, but no `xl/sharedStrings.xml` part at all.
    let bytes = xlsx_archive(
        concat!(
            r#"<row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c></row>"#,
            r#"<row r="2"><c r="B2" t="s"><v>0</v></c></row>"#,
        ),
        None,
    );
    let error = XlsxParser::new().parse_bytes(&bytes).unwrap_err();
    // Before the fix this cell decoded to "" and the sheet parsed as a
    // one-column table with a blank value.
    assert_error(
        &error,
        ErrorKind::Retrieval,
        &[XLSX_SHEET_PART, "cell B2", "shared string 0", "0 entries"],
    );
}

#[test]
fn xlsx_shared_string_index_past_the_table_names_the_size() {
    // Two entries in the table, one cell asking for index 5.
    let bytes = xlsx_archive(
        r#"<row r="2"><c r="B2" t="s"><v>5</v></c></row>"#,
        Some("<sst><si><t>Name</t></si><si><t>alice</t></si></sst>"),
    );
    let error = XlsxParser::new().parse_bytes(&bytes).unwrap_err();
    assert_error(
        &error,
        ErrorKind::Retrieval,
        &[XLSX_SHEET_PART, "cell B2", "shared string 5", "2 entries"],
    );
}

#[test]
fn xlsx_inline_string_workbooks_need_no_shared_string_table() {
    // No `xl/sharedStrings.xml` and no `t="s"` cell: a complete workbook that
    // must keep parsing exactly as before.
    let bytes = xlsx_archive(
        concat!(
            r#"<row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c>"#,
            r#"<c r="B1" t="inlineStr"><is><t>Price</t></is></c></row>"#,
            r#"<row r="2"><c r="A2" t="inlineStr"><is><t>alice</t></is></c>"#,
            r#"<c r="B2"><v>12.0</v></c></row>"#,
        ),
        None,
    );
    let workbook = XlsxParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(workbook.sheet_names(), vec!["Data"]);
    let table = &workbook.sheets[0].table;
    assert_eq!(table.headers, vec!["Name".to_owned(), "Price".to_owned()]);
    assert_eq!(table.rows, vec![vec!["alice".to_owned(), "12".to_owned()]]);
}

#[test]
fn xlsx_unparsable_shared_strings_are_a_typed_retrieval_error() {
    // The part exists but is truncated, so the table cannot be decoded at all.
    let bytes = xlsx_archive(
        r#"<row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c></row>"#,
        Some("<sst><si><t>oops"),
    );
    let error = XlsxParser::new().parse_bytes(&bytes).unwrap_err();
    assert_error(
        &error,
        ErrorKind::Retrieval,
        &["shared strings", SHARED_STRINGS_PART],
    );
}

#[test]
fn xlsx_shared_string_count_declarations_are_informational() {
    // `count`/`uniqueCount` claim 99 entries over a 2-entry table; the
    // declarations stay informational and every in-range lookup still works.
    let bytes = xlsx_archive(
        concat!(
            r#"<row r="1"><c r="A1" t="s"><v>0</v></c></row>"#,
            r#"<row r="2"><c r="A2" t="s"><v>1</v></c></row>"#,
        ),
        Some(r#"<sst count="99" uniqueCount="99"><si><t>Name</t></si><si><t>alice</t></si></sst>"#),
    );
    let workbook = XlsxParser::new().parse_bytes(&bytes).unwrap();
    let table = &workbook.sheets[0].table;
    assert_eq!(table.headers, vec!["Name"]);
    assert_eq!(table.rows, vec![vec!["alice".to_owned()]]);
}

// ---------------------------------------------------------------------------
// PPTX: an unresolved slide keeps its place in the deck
// ---------------------------------------------------------------------------

/// `ppt/presentation.xml`: three declared slides, the middle one unmapped.
const PPTX_PRESENTATION: &str = concat!(
    r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main""#,
    r#" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
    r#"<p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId7"/>"#,
    r#"<p:sldId id="258" r:id="rId8"/></p:sldIdLst></p:presentation>"#,
);

/// `ppt/_rels/presentation.xml.rels`: `rId1` resolves, `rId7` is unmapped, and
/// `rId8` targets `slides/slide3.xml`, a part the archive below omits.
const PPTX_RELS: &str = concat!(
    r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    r#"<Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/>"#,
    r#"<Relationship Id="rId8" Type="slide" Target="slides/slide3.xml"/></Relationships>"#,
);

/// A presentation part that declares an empty `<p:sldIdLst>`.
const PPTX_PRESENTATION_WITHOUT_SLIDES: &str = concat!(
    r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main""#,
    r#" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
    r#"<p:sldIdLst/></p:presentation>"#,
);

/// A relationships part with no entries.
const PPTX_EMPTY_RELS: &str =
    r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"#;

/// Renders one slide part holding a single body line.
fn pptx_slide(text: &str) -> String {
    format!(
        r#"<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:nvPr/></p:nvSpPr><p:txBody><a:bodyPr/><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#
    )
}

#[test]
fn pptx_unresolved_slides_keep_their_deck_position() {
    let slide = pptx_slide("only real slide");
    let bytes = archive(&[
        (PRESENTATION_PART, PPTX_PRESENTATION),
        (PRESENTATION_RELS_PART, PPTX_RELS),
        ("ppt/slides/slide1.xml", slide.as_str()),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    // Three declared slides, three parsed slides: nothing was dropped.
    assert_eq!(document.slide_count, 3);
    assert_eq!(document.slides.len(), 3);
    assert_eq!(
        document
            .slides
            .iter()
            .map(|slide| slide.index)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let resolved = document.slide(1).unwrap();
    assert_eq!(resolved.missing_part, None);
    assert_eq!(resolved.text, "only real slide");
    // The unmapped relationship and the absent part are both placeholders that
    // name their relationship id.
    for (position, reference) in [(2, "rId7"), (3, "rId8")] {
        let placeholder = document.slide(position).unwrap();
        assert_eq!(placeholder.missing_part.as_deref(), Some(reference));
        assert_eq!(
            placeholder.text,
            format!("! missing slide part: {reference}")
        );
    }
    assert_eq!(document.problems.len(), 2, "{:?}", document.problems);
    assert!(
        document.problems[0].contains("rId7"),
        "{:?}",
        document.problems
    );
    assert!(
        document.problems[1].contains("rId8"),
        "{:?}",
        document.problems
    );
    assert!(document.problems[1].contains("missing from the archive"));
    // Deck order survives: the three headings appear once each, in order, and
    // both placeholder lines are rendered underneath their own heading.
    let positions: Vec<usize> = ["## Slide 1", "## Slide 2", "## Slide 3"]
        .iter()
        .map(|heading| document.text.find(heading).unwrap())
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "deck order lost: {positions:?}"
    );
    assert!(
        document
            .text
            .contains("## Slide 2\n! missing slide part: rId7"),
        "{}",
        document.text
    );
    assert!(
        document
            .text
            .contains("## Slide 3\n! missing slide part: rId8"),
        "{}",
        document.text
    );
}

#[test]
fn pptx_complete_decks_report_no_problems() {
    // The public fixture builder writes a deck whose relationship chain is
    // complete; the clean path must stay clean.
    let bytes =
        fixtures::pptx_slides(&[("Intro", &["first body"]), ("Data", &["second body"])]).unwrap();
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(document.slide_count, 2);
    assert!(document.problems.is_empty(), "{:?}", document.problems);
    assert!(document
        .slides
        .iter()
        .all(|slide| slide.missing_part.is_none()));
    assert_eq!(
        document.text,
        "## Slide 1\nIntro\nfirst body\n## Slide 2\nData\nsecond body"
    );
}

#[test]
fn pptx_empty_slide_list_keeps_the_numeric_fallback() {
    // The numeric fallback exists for a presentation part that declares no
    // slides at all; an empty `<p:sldIdLst>` is that case, not a lossy deck.
    let first = pptx_slide("one");
    let second = pptx_slide("two");
    let bytes = archive(&[
        (PRESENTATION_PART, PPTX_PRESENTATION_WITHOUT_SLIDES),
        (PRESENTATION_RELS_PART, PPTX_EMPTY_RELS),
        ("ppt/slides/slide1.xml", first.as_str()),
        ("ppt/slides/slide2.xml", second.as_str()),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(document.slide_count, 2);
    assert!(document.problems.is_empty(), "{:?}", document.problems);
    assert_eq!(document.slide(1).unwrap().text, "one");
    assert_eq!(document.slide(2).unwrap().text, "two");
    assert!(document
        .slides
        .iter()
        .all(|slide| slide.missing_part.is_none()));
}

// ---------------------------------------------------------------------------
// ODT: elements that contribute no text are reported, not silently dropped
// ---------------------------------------------------------------------------

/// `content.xml` with a heading, two drawings (one per paragraph), a footnote,
/// a table of contents, and surrounding paragraphs that must still render.
const ODT_CONTENT_WITH_LOSSES: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8"?>"#,
    r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0""#,
    r#" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0""#,
    r#" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0">"#,
    r#"<office:body><office:text>"#,
    r#"<text:h text:outline-level="1">Report</text:h>"#,
    r#"<text:p>Before the drawing</text:p>"#,
    r#"<text:p><draw:frame><draw:image href="Pictures/a.png"/></draw:frame></text:p>"#,
    r#"<text:p><draw:frame><draw:image href="Pictures/b.png"/></draw:frame></text:p>"#,
    r#"<text:p><text:note><text:note-citation>1</text:note-citation>"#,
    r#"<text:note-body><text:p>footnote</text:p></text:note-body></text:note></text:p>"#,
    r#"<text:table-of-content text:name="toc"/>"#,
    r#"<text:p>After the drawing</text:p>"#,
    r#"</office:text></office:body></office:document-content>"#,
);

/// `content.xml` holding only a heading and one paragraph.
const ODT_CLEAN_CONTENT: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8"?>"#,
    r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0""#,
    r#" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">"#,
    r#"<office:body><office:text>"#,
    r#"<text:h text:outline-level="2">Details</text:h>"#,
    r#"<text:p>Plain body</text:p>"#,
    r#"</office:text></office:body></office:document-content>"#,
);

#[test]
fn odt_reports_elements_that_contribute_no_text() {
    let bytes = archive(&[(CONTENT_PART, ODT_CONTENT_WITH_LOSSES)]);
    let document = OdtParser::new().parse_bytes(&bytes).unwrap();
    // The text around the losses is still rendered, in document order.
    assert_eq!(
        document.text,
        "# Report\nBefore the drawing\nAfter the drawing"
    );
    assert_eq!(document.paragraph_count, 3);
    assert_eq!(document.table_count, 0);
    // One entry per distinct element, sorted by name, each with its count and
    // the Unicode multiplication sign.
    assert_eq!(
        document.unsupported_elements,
        vec![
            "draw:frame (×2)".to_owned(),
            "draw:image (×2)".to_owned(),
            "text:note (×1)".to_owned(),
            "text:table-of-content (×1)".to_owned(),
        ]
    );
    assert!(
        document
            .unsupported_elements
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "entries are sorted by element name: {:?}",
        document.unsupported_elements
    );
}

#[test]
fn odt_clean_documents_report_no_unsupported_elements() {
    let bytes = archive(&[(CONTENT_PART, ODT_CLEAN_CONTENT)]);
    let document = OdtParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(document.text, "## Details\nPlain body");
    assert_eq!(document.paragraph_count, 2);
    assert_eq!(document.table_count, 0);
    assert_eq!(document.unsupported_elements, Vec::<String>::new());
}

// ---------------------------------------------------------------------------
// Serde: the new fields are additive
// ---------------------------------------------------------------------------

#[test]
fn office_documents_tolerate_payloads_without_the_new_fields() {
    // A payload written by an older build has no `unsupported_elements` /
    // `problems` key; both default to an empty list instead of failing.
    let odt: OdtDocument = serde_json::from_str(r#"{"text":"x"}"#).unwrap();
    assert_eq!(odt.text, "x");
    assert_eq!(odt.unsupported_elements, Vec::<String>::new());
    let pptx: PptxDocument = serde_json::from_str(r#"{"text":"x"}"#).unwrap();
    assert_eq!(pptx.text, "x");
    assert_eq!(pptx.problems, Vec::<String>::new());
    // Both keys are always serialised, so a consumer never has to guess.
    let odt_json = serde_json::to_value(&odt).unwrap();
    assert_eq!(
        odt_json.get("unsupported_elements"),
        Some(&serde_json::json!([]))
    );
    let pptx_json = serde_json::to_value(&pptx).unwrap();
    assert_eq!(pptx_json.get("problems"), Some(&serde_json::json!([])));
}
