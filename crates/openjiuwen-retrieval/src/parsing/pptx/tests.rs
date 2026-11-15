//! Unit tests for the `PPTX` reader.
//!
//! Fixtures are built in memory with [`ZipWriter`](crate::parsing::zip::ZipWriter);
//! the XML builders below emit the smallest `PresentationML` snippets the parser
//! needs, so every test names exactly the part structure under test.

use std::fmt::Write as _;

use openjiuwen_common::error::ErrorKind;

use super::*;
use crate::parsing::zip::ZipWriter;

/// Builds a stored-entry archive from `(name, body)` pairs.
fn archive(parts: &[(&str, String)]) -> Vec<u8> {
    let mut writer = ZipWriter::new();
    for (name, body) in parts {
        writer.add_stored(*name, body.clone()).unwrap();
    }
    writer.finish().unwrap()
}

/// Wraps shape markup in a slide part.
fn slide_part(shapes: &str) -> String {
    format!(
        r#"<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>"#
    )
}

/// A plain shape holding `paragraphs`.
fn shape(paragraphs: &str) -> String {
    format!("<p:sp><p:nvSpPr><p:nvPr/></p:nvSpPr><p:txBody>{paragraphs}</p:txBody></p:sp>")
}

/// A shape with a placeholder of `kind`.
fn placeholder(kind: &str, paragraphs: &str) -> String {
    format!(
        "<p:sp><p:nvSpPr><p:nvPr><p:ph type=\"{kind}\"/></p:nvPr></p:nvSpPr><p:txBody>{paragraphs}</p:txBody></p:sp>"
    )
}

/// One `<a:p>` holding `runs`.
fn paragraph(runs: &str) -> String {
    format!("<a:p>{runs}</a:p>")
}

/// One `<a:t>` run.
fn run(text: &str) -> String {
    format!("<a:r><a:t>{text}</a:t></a:r>")
}

/// One table row: `cells` are already rendered `<a:tc>` elements.
fn table_row(cells: &[String]) -> String {
    format!("<a:tr>{}</a:tr>", cells.concat())
}

/// One table cell holding one paragraph.
fn table_cell(text: &str) -> String {
    format!(
        "<a:tc><a:txBody>{}</a:txBody></a:tc>",
        paragraph(&run(text))
    )
}

/// A graphic frame holding `rows`.
fn graphic_frame(rows: &[String]) -> String {
    format!(
        "<p:graphicFrame><a:graphic><a:graphicData><a:tbl>{}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>",
        rows.concat()
    )
}

/// A presentation part listing `ids` in order.
fn presentation(ids: &[&str]) -> String {
    let mut entries = String::new();
    for id in ids {
        write!(entries, r#"<p:sldId id="256" r:id="{id}"/>"#).unwrap();
    }
    format!(
        r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst>{entries}</p:sldIdLst></p:presentation>"#
    )
}

/// A presentation part wrapping already-rendered `<p:sldId>` entries.
fn presentation_raw(entries: &str) -> String {
    format!(
        r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst>{entries}</p:sldIdLst></p:presentation>"#
    )
}

/// A relationships part mapping `(id, target)` pairs.
fn relationships(entries: &[(&str, &str)]) -> String {
    let mut items = String::new();
    for (id, target) in entries {
        write!(
            items,
            r#"<Relationship Id="{id}" Type="slide" Target="{target}"/>"#
        )
        .unwrap();
    }
    format!(
        r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{items}</Relationships>"#
    )
}

/// A notes part holding one body shape plus a slide-number placeholder.
fn notes_part(text: &str) -> String {
    format!(
        r#"<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>{}{}</p:spTree></p:cSld></p:notes>"#,
        shape(&paragraph(&run(text))),
        placeholder("sldNum", &paragraph(&run("1")))
    )
}

#[test]
fn presentation_order_wins_over_numeric_order() {
    let bytes = archive(&[
        (
            "ppt/presentation.xml",
            presentation(&["rId3", "rId1", "rId2"]),
        ),
        (
            "ppt/_rels/presentation.xml.rels",
            relationships(&[
                ("rId1", "slides/slide1.xml"),
                ("rId2", "./slides/slide2.xml"),
                ("rId3", "/ppt/slides/slide3.xml"),
            ]),
        ),
        (
            "ppt/slides/slide1.xml",
            slide_part(&shape(&paragraph(&run("slide one")))),
        ),
        (
            "ppt/slides/slide2.xml",
            slide_part(&shape(&paragraph(&run("slide two")))),
        ),
        (
            "ppt/slides/slide3.xml",
            slide_part(&shape(&paragraph(&run("slide three")))),
        ),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(document.slide_count, 3);
    assert_eq!(
        document
            .slides
            .iter()
            .map(|slide| slide.index)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(document.slide(1).unwrap().text, "slide three");
    assert_eq!(document.slide(2).unwrap().text, "slide one");
    assert_eq!(document.slide(3).unwrap().text, "slide two");
    assert!(document.slide(0).is_none());
    assert!(document.slide(4).is_none());
    assert!(document.text.starts_with("## Slide 1\nslide three"));
    assert!(document.text.contains("## Slide 2\nslide one"));
    assert!(document.title.is_none());
}

#[test]
fn numeric_fallback_orders_slides_by_number() {
    let bytes = archive(&[
        (
            "ppt/slides/slide10.xml",
            slide_part(&shape(&paragraph(&run("ten")))),
        ),
        (
            "ppt/slides/slide2.xml",
            slide_part(&shape(&paragraph(&run("two")))),
        ),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(document.slide_count, 2);
    assert_eq!(document.slide(1).unwrap().text, "two");
    assert_eq!(document.slide(2).unwrap().text, "ten");
    assert_eq!(document.slide(1).unwrap().title.as_deref(), Some("two"));
}

#[test]
fn paragraph_runs_breaks_and_title_placeholder() {
    let body = shape(&format!(
        "{}{}{}",
        paragraph(&format!("{}{}", run("Hello"), run(" world"))),
        paragraph(&format!("{}<a:br/>{}", run("line1"), run("line2"))),
        paragraph("<a:r><a:t>   </a:t></a:r>"),
    ));
    let slide = slide_part(&format!(
        "{}{}",
        placeholder("title", &paragraph(&run("Deck Title"))),
        body
    ));
    let document = PptxParser::new()
        .parse_bytes(&archive(&[("ppt/slides/slide1.xml", slide)]))
        .unwrap();
    let parsed = document.slide(1).unwrap();
    assert_eq!(parsed.text, "Deck Title\nHello world\nline1\nline2");
    assert_eq!(parsed.title.as_deref(), Some("Deck Title"));
}

#[test]
fn tables_render_as_markdown() {
    let escaped_cell = format!(
        "<a:tc><a:txBody>{}</a:txBody><a:tcPr/></a:tc>",
        paragraph(&run("a|b"))
    );
    let frame = graphic_frame(&[
        table_row(&[table_cell("A"), table_cell("B")]),
        table_row(&[escaped_cell, table_cell("second")]),
    ]);
    let document = PptxParser::new()
        .parse_bytes(&archive(&[("ppt/slides/slide1.xml", slide_part(&frame))]))
        .unwrap();
    assert_eq!(
        document.slide(1).unwrap().text,
        "| A | B |\n| --- | --- |\n| a\\|b | second |"
    );
}

#[test]
fn notes_are_read_and_numbered_placeholders_skipped() {
    let bytes = archive(&[
        (
            "ppt/slides/slide1.xml",
            slide_part(&shape(&paragraph(&run("first")))),
        ),
        (
            "ppt/slides/slide2.xml",
            slide_part(&shape(&paragraph(&run("second")))),
        ),
        ("ppt/notesSlides/notesSlide1.xml", notes_part("note one")),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    let first = document.slide(1).unwrap();
    assert!(first.has_notes);
    assert_eq!(first.notes.as_deref(), Some("note one"));
    let second = document.slide(2).unwrap();
    assert!(!second.has_notes);
    assert_eq!(second.notes, None);
    assert!(document.text.contains("first\n### Notes\nnote one"));
    assert!(!document.text.contains("### Notes\n1"));
}

#[test]
fn core_properties_supply_the_title() {
    let core = r#"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Quarterly Deck</dc:title></cp:coreProperties>"#.to_owned();
    let bytes = archive(&[
        (
            "ppt/slides/slide1.xml",
            slide_part(&shape(&paragraph(&run("body")))),
        ),
        ("docProps/core.xml", core),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();
    assert_eq!(document.title.as_deref(), Some("Quarterly Deck"));
}

#[test]
fn archives_without_slides_are_rejected() {
    let presentation_only = archive(&[("ppt/presentation.xml", presentation(&[]))]);
    let error = PptxParser::new()
        .parse_bytes(&presentation_only)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("no slides"), "{error}");

    let empty = archive(&[]);
    let error = PptxParser::new().parse_bytes(&empty).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("no slides"), "{error}");
}

#[test]
fn malformed_parts_and_non_zip_bytes_are_validation_errors() {
    let broken_slide = archive(&[("ppt/slides/slide1.xml", "<p:sld><p:cSld>".to_owned())]);
    let error = PptxParser::new().parse_bytes(&broken_slide).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("ppt/slides/slide1.xml"),
        "{error}"
    );

    let non_zip = PptxParser::new()
        .parse_bytes(b"not a zip archive")
        .unwrap_err();
    assert_eq!(non_zip.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn parse_path_reads_files_and_reports_missing_ones() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("deck.pptx");
    let bytes = archive(&[(
        "ppt/slides/slide1.xml",
        slide_part(&shape(&paragraph(&run("from disk")))),
    )]);
    tokio::fs::write(&path, &bytes).await.unwrap();
    let document = PptxParser::new().parse_path(&path).await.unwrap();
    assert_eq!(document.slide_count, 1);
    assert_eq!(document.slide(1).unwrap().text, "from disk");

    let missing = PptxParser::new()
        .parse_path(directory.path().join("absent.pptx"))
        .await
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
}

#[test]
fn unresolved_slide_parts_keep_their_deck_position() {
    let bytes = archive(&[
        (
            "ppt/presentation.xml",
            presentation(&["rId1", "rId7", "rId8"]),
        ),
        (
            "ppt/_rels/presentation.xml.rels",
            relationships(&[("rId1", "slides/slide1.xml"), ("rId8", "slides/slide3.xml")]),
        ),
        (
            "ppt/slides/slide1.xml",
            slide_part(&shape(&paragraph(&run("real one")))),
        ),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();

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
    assert_eq!(resolved.text, "real one");
    assert_eq!(resolved.missing_part, None);

    let unmapped = document.slide(2).unwrap();
    assert_eq!(unmapped.missing_part.as_deref(), Some("rId7"));
    assert_eq!(unmapped.text, "! missing slide part: rId7");
    assert_eq!(unmapped.title, None);
    assert_eq!(unmapped.notes, None);
    assert!(!unmapped.has_notes);

    let absent = document.slide(3).unwrap();
    assert_eq!(absent.missing_part.as_deref(), Some("rId8"));
    assert_eq!(absent.text, "! missing slide part: rId8");
    assert_eq!(absent.title, None);

    assert_eq!(
        document.problems,
        vec![
            "slide 2: rId7 is not mapped by ppt/_rels/presentation.xml.rels".to_owned(),
            "slide 3: rId8 target slides/slide3.xml is missing from the archive".to_owned(),
        ]
    );
    assert!(
        document
            .text
            .starts_with("## Slide 1\nreal one\n## Slide 2\n"),
        "{}",
        document.text
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
fn sld_id_without_relationship_id_is_a_placeholder() {
    let bytes = archive(&[
        (
            "ppt/presentation.xml",
            presentation_raw(r#"<p:sldId id="256"/>"#),
        ),
        ("ppt/_rels/presentation.xml.rels", relationships(&[])),
        (
            "ppt/slides/slide1.xml",
            slide_part(&shape(&paragraph(&run("unreachable")))),
        ),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();

    // Declared entries exist, so the numeric fallback must not resurrect the
    // part the presentation part failed to reference.
    assert_eq!(document.slide_count, 1);
    let slide = document.slide(1).unwrap();
    assert_eq!(slide.missing_part.as_deref(), Some("<no r:id>"));
    assert_eq!(slide.text, "! missing slide part: <no r:id>");
    assert_eq!(
        document.problems,
        vec!["slide 1: <p:sldId> has no r:id".to_owned()]
    );
    assert_eq!(document.text, "## Slide 1\n! missing slide part: <no r:id>");
}

#[test]
fn relationship_target_that_is_not_a_slide_is_a_placeholder() {
    let bytes = archive(&[
        ("ppt/presentation.xml", presentation(&["rId2"])),
        (
            "ppt/_rels/presentation.xml.rels",
            relationships(&[("rId2", "media/image1.png")]),
        ),
        ("ppt/media/image1.png", "not a slide part".to_owned()),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();

    assert_eq!(document.slide_count, 1);
    let slide = document.slide(1).unwrap();
    assert_eq!(slide.missing_part.as_deref(), Some("rId2"));
    assert_eq!(
        document.problems,
        vec!["slide 1: rId2 target media/image1.png is not a slide part".to_owned()]
    );
    assert!(document.text.contains("! missing slide part: rId2"));
}

#[test]
fn complete_decks_report_no_problems() {
    let bytes = archive(&[
        ("ppt/presentation.xml", presentation(&["rId1", "rId2"])),
        (
            "ppt/_rels/presentation.xml.rels",
            relationships(&[("rId1", "slides/slide1.xml"), ("rId2", "slides/slide2.xml")]),
        ),
        (
            "ppt/slides/slide1.xml",
            slide_part(&shape(&paragraph(&run("one")))),
        ),
        (
            "ppt/slides/slide2.xml",
            slide_part(&shape(&paragraph(&run("two")))),
        ),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();

    assert_eq!(document.problems, Vec::<String>::new());
    assert!(document
        .slides
        .iter()
        .all(|slide| slide.missing_part.is_none()));
    assert_eq!(document.slide_count, 2);
    assert_eq!(document.text, "## Slide 1\none\n## Slide 2\ntwo");
}

#[test]
fn empty_slide_list_still_falls_back_to_numeric_order() {
    let bytes = archive(&[
        ("ppt/presentation.xml", presentation(&[])),
        ("ppt/_rels/presentation.xml.rels", relationships(&[])),
        (
            "ppt/slides/slide10.xml",
            slide_part(&shape(&paragraph(&run("ten")))),
        ),
        (
            "ppt/slides/slide2.xml",
            slide_part(&shape(&paragraph(&run("two")))),
        ),
    ]);
    let document = PptxParser::new().parse_bytes(&bytes).unwrap();

    assert_eq!(document.slide_count, 2);
    assert_eq!(document.slide(1).unwrap().text, "two");
    assert_eq!(document.slide(2).unwrap().text, "ten");
    assert_eq!(document.problems, Vec::<String>::new());
    assert!(document
        .slides
        .iter()
        .all(|slide| slide.missing_part.is_none()));
}

#[test]
fn placeholder_fields_serialise_and_older_payloads_still_load() {
    let placeholder = PptxSlide {
        index: 2,
        text: "! missing slide part: rId7".to_owned(),
        missing_part: Some("rId7".to_owned()),
        ..PptxSlide::default()
    };
    let json = serde_json::to_value(&placeholder).unwrap();
    assert_eq!(json["missing_part"], serde_json::json!("rId7"));
    assert_eq!(json["index"], serde_json::json!(2));

    let without_field = serde_json::json!({
        "index": 1,
        "text": "body",
        "title": null,
        "notes": null,
        "has_notes": false
    });
    let parsed: PptxSlide = serde_json::from_value(without_field).unwrap();
    assert_eq!(parsed.missing_part, None);

    let document_without_problems = serde_json::json!({
        "text": "## Slide 1\nbody",
        "slides": [],
        "slide_count": 0,
        "title": null
    });
    let parsed: PptxDocument = serde_json::from_value(document_without_problems).unwrap();
    assert_eq!(parsed.problems, Vec::<String>::new());
}
