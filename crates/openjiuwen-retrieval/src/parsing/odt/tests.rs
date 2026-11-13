//! Unit tests for the `ODT` reader: block rendering, inline content, tables,
//! lists, metadata, error paths, and file parsing.

use openjiuwen_common::error::ErrorKind;

use crate::parsing::zip::ZipWriter;

use super::*;

/// Wraps body markup into a complete `content.xml` document.
fn content(body: &str) -> String {
    format!(
        concat!(
            r#"<?xml version="1.0" encoding="UTF-8"?>"#,
            r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0""#,
            r#" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0""#,
            r#" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0""#,
            r#" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0">"#,
            r#"<office:body><office:text>{body}</office:text></office:body>"#,
            r#"</office:document-content>"#,
        ),
        body = body
    )
}

/// Builds a stored-entry archive from `(name, payload)` parts.
fn archive(parts: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = ZipWriter::new();
    for (name, payload) in parts {
        writer
            .add_stored(*name, payload.as_bytes().to_vec())
            .unwrap();
    }
    writer.finish().unwrap()
}

/// Parses body markup wrapped into a one-part archive.
fn parse_content(body: &str) -> OdtDocument {
    let markup = content(body);
    OdtParser::new()
        .parse_bytes(&archive(&[(CONTENT_PART, markup.as_str())]))
        .unwrap()
}

#[test]
fn renders_headings_and_paragraphs_in_order() {
    let document = parse_content(concat!(
        r#"<text:h text:outline-level="1">Rust rewrite</text:h>"#,
        r#"<text:p>Intro paragraph</text:p>"#,
        r#"<text:h text:outline-level="2">Overview</text:h>"#,
        r#"<text:p>  Spaced paragraph  </text:p>"#,
        r#"<text:p></text:p>"#,
        r#"<text:h text:outline-level="12">Deep</text:h>"#,
        r#"<text:h text:outline-level="oops">Broken level</text:h>"#,
        r#"<text:h/>"#,
    ));
    assert_eq!(
        document.text(),
        "# Rust rewrite\nIntro paragraph\n## Overview\nSpaced paragraph\n######### Deep\n# Broken level"
    );
    assert_eq!(document.paragraph_count, 6);
    assert_eq!(document.table_count, 0);
    assert_eq!(document.title, None);
    assert!(!document.is_empty());
}

#[test]
fn collects_inline_content_in_child_order() {
    let document = parse_content(concat!(
        r#"<text:p><text:span>alpha</text:span><text:s text:c="3"/>"#,
        r#"<text:span>beta</text:span><text:s/><text:tab/><text:line-break/>"#,
        r#"<text:span>nested <text:a href="https://example.com">link</text:a></text:span>"#,
        r#"<text:bookmark-ref>ref</text:bookmark-ref>"#,
        r#"<text:note><text:note-citation>1</text:note-citation>"#,
        r#"<text:note-body><text:p>footnote</text:p></text:note-body></text:note>"#,
        r#"<draw:frame><draw:text-box><text:p>drawn</text:p></draw:text-box></draw:frame>"#,
        r#"</text:p>"#,
    ));
    assert_eq!(document.text(), "alpha   beta \t\nnested linkref");
    assert_eq!(document.paragraph_count, 1);
}

#[test]
fn renders_tables_with_covered_cells_and_escaped_pipes() {
    let document = parse_content(concat!(
        r#"<table:table><table:table-row>"#,
        r#"<table:table-cell><text:p>Name</text:p></table:table-cell>"#,
        r#"<table:table-cell><text:p>Notes</text:p></table:table-cell>"#,
        r#"<table:table-cell><text:p>Extra</text:p></table:table-cell>"#,
        r#"</table:table-row><table:table-row>"#,
        r#"<table:table-cell><text:p>alice</text:p></table:table-cell>"#,
        r#"<table:covered-table-cell/><table:table-cell><text:p>a|b</text:p></table:table-cell>"#,
        r#"</table:table-row><table:table-row>"#,
        r#"<table:table-cell><text:p>bob</text:p></table:table-cell>"#,
        r#"</table:table-row></table:table>"#,
    ));
    assert_eq!(
        document.text(),
        "| Name | Notes | Extra |\n| --- | --- | --- |\n| alice |  | a\\|b |\n| bob |  |  |"
    );
    assert_eq!(document.table_count, 1);
    assert_eq!(document.paragraph_count, 0);
    // A table without any cell is dropped instead of rendering an empty grid.
    let empty = parse_content("<table:table><table:table-row/></table:table>");
    assert_eq!(empty.text(), "");
    assert_eq!(empty.table_count, 0);
}

#[test]
fn flattens_lists_into_consecutive_lines() {
    let document = parse_content(concat!(
        "<text:list><text:list-item><text:p>one</text:p>",
        "<text:list><text:list-item><text:p>nested</text:p></text:list-item></text:list>",
        "</text:list-item><text:list-item><text:p>two</text:p></text:list-item>",
        "<text:list-header><text:p>header</text:p></text:list-header></text:list>",
        "<text:section><text:p>after section</text:p></text:section>",
    ));
    assert_eq!(document.text(), "one\nnested\ntwo\nheader\nafter section");
    assert_eq!(document.paragraph_count, 5);
}

#[test]
fn empty_documents_have_no_text_and_no_counts() {
    let document = parse_content("");
    assert_eq!(document.text(), "");
    assert_eq!(document, OdtDocument::default());
    assert!(document.is_empty());
    assert_eq!(document.paragraph_count, 0);
    assert_eq!(document.table_count, 0);

    // A self-closing body container behaves the same.
    let markup = content("").replace("<office:text></office:text>", "<office:text/>");
    let self_closing = OdtParser::new()
        .parse_bytes(&archive(&[(CONTENT_PART, markup.as_str())]))
        .unwrap();
    assert!(self_closing.is_empty());
    assert_eq!(self_closing, OdtDocument::default());
    let _: OdtParser = OdtParser;
}

#[test]
fn reports_missing_and_malformed_content() {
    let missing = OdtParser::new()
        .parse_bytes(&archive(&[(META_PART, "<office:document-meta/>")]))
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::Validation);
    assert!(missing.to_string().contains("content.xml"), "{missing}");

    let not_a_zip = OdtParser::new().parse_bytes(b"not a zip").unwrap_err();
    assert_eq!(not_a_zip.kind(), ErrorKind::Validation);

    let malformed = OdtParser::new()
        .parse_bytes(&archive(&[(
            CONTENT_PART,
            "<office:document-content><office:body>",
        )]))
        .unwrap_err();
    assert_eq!(malformed.kind(), ErrorKind::Validation);
    assert!(malformed.to_string().contains(CONTENT_PART), "{malformed}");
}

#[test]
fn reads_the_title_from_the_properties_part() {
    let meta = concat!(
        r#"<?xml version="1.0" encoding="UTF-8"?>"#,
        r#"<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0""#,
        r#" xmlns:dc="http://purl.org/dc/elements/1.1/">"#,
        r#"<office:meta><dc:title> Quarterly report </dc:title>"#,
        r#"<dc:creator>alice</dc:creator></office:meta></office:document-meta>"#,
    );
    let markup = content("<text:p>body</text:p>");
    let document = OdtParser::new()
        .parse_bytes(&archive(&[
            (CONTENT_PART, markup.as_str()),
            (META_PART, meta),
        ]))
        .unwrap();
    assert_eq!(document.title.as_deref(), Some("Quarterly report"));
    assert_eq!(document.text(), "body");

    // A missing, empty, or broken properties part only drops the title.
    assert_eq!(parse_content("<text:p>body</text:p>").title, None);
    let parts = [
        (CONTENT_PART, markup.as_str()),
        (
            META_PART,
            "<office:document-meta><office:meta><dc:title>  </dc:title></office:meta></office:document-meta>",
        ),
    ];
    let empty_title = OdtParser::new().parse_bytes(&archive(&parts)).unwrap();
    assert_eq!(empty_title.title, None);
    let broken = OdtParser::new()
        .parse_bytes(&archive(&[
            (CONTENT_PART, markup.as_str()),
            (META_PART, "<office:document-meta"),
        ]))
        .unwrap();
    assert_eq!(broken.title, None);
}

#[tokio::test]
async fn parse_path_reads_a_file_and_reports_missing_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("document.odt");
    let markup =
        content(r#"<text:h text:outline-level="1">From disk</text:h><text:p>body</text:p>"#);
    tokio::fs::write(&path, archive(&[(CONTENT_PART, markup.as_str())]))
        .await
        .unwrap();
    let document = OdtParser::new().parse_path(&path).await.unwrap();
    assert_eq!(document.text(), "# From disk\nbody");
    assert_eq!(document.paragraph_count, 2);

    let missing = OdtParser::new()
        .parse_path(directory.path().join("absent.odt"))
        .await
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
}

#[test]
fn documents_round_trip_through_json() {
    let document = parse_content("<text:p>body</text:p>");
    let json = serde_json::to_string(&document).unwrap();
    let restored: OdtDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, document);
    assert_eq!(
        serde_json::from_str::<OdtDocument>("{}").unwrap(),
        OdtDocument::default()
    );
}
