//! Tests for the document route of the scrape stage.
//!
//! Every document is built in memory by
//! [`fixtures`](openjiuwen_core::retrieval::loaders::fixtures), so a test needs
//! no office suite, no temporary file, and no network. Assertions are
//! deterministic: nothing here reads the environment, a wall clock, or a
//! registry.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::retrieval::common::Document;
use openjiuwen_core::retrieval::loaders::fixtures;
use serde_json::Value;

use super::*;
use crate::skill_creator::omni::html::Heading;

/// A successful 200 response for `url` carrying `body`.
fn resource(url: &str, content_type: Option<&str>, body: Vec<u8>) -> FetchedResource {
    FetchedResource {
        url: url.to_owned(),
        final_url: url.to_owned(),
        status: 200,
        content_type: content_type.map(str::to_owned),
        body,
    }
}

/// Extracts the page of a document served at `url`.
async fn extract_page(url: &str, content_type: &str, body: Vec<u8>) -> ScrapedPage {
    document_page(&resource(url, Some(content_type), body), url)
        .await
        .expect("extracts the document")
        .expect("a document resource")
}

#[test]
fn document_source_dispatches_on_the_url_suffix() {
    for (url, name, extension) in [
        ("https://example.com/files/report.pdf", "report.pdf", ".pdf"),
        (
            "https://example.com/report.PDF?v=2#page=3",
            "report.PDF",
            ".pdf",
        ),
        ("https://example.com/a/b/deck.pptx/", "deck.pptx", ".pptx"),
        (
            "https://example.com/notes.docx?download=1",
            "notes.docx",
            ".docx",
        ),
        ("https://example.com/sheet.xlsx", "sheet.xlsx", ".xlsx"),
        ("https://example.com/memo.odt", "memo.odt", ".odt"),
        ("https://example.com/readme.md", "readme.md", ".md"),
        (
            "https://example.com/readme.markdown",
            "readme.markdown",
            ".markdown",
        ),
        ("https://example.com/notes.txt", "notes.txt", ".txt"),
        ("https://example.com/people.csv", "people.csv", ".csv"),
        ("https://example.com/people.tsv", "people.tsv", ".tsv"),
        ("https://example.com/data.json", "data.json", ".json"),
    ] {
        let source = document_source(url, None).unwrap_or_else(|| panic!("{url} is a document"));
        assert_eq!(source.name, name, "{url}");
        assert_eq!(source.extension, extension, "{url}");
        assert!(is_document(url, None), "{url}");
    }
}

#[test]
fn document_source_ignores_non_document_resources() {
    for url in [
        "https://example.com/page.html",
        "https://example.com/page.htm",
        "https://example.com/screenshot.png",
        "https://example.com/photo.jpeg",
        "https://example.com/archive.zip",
        "https://example.com/archive.tar.gz",
        "https://example.com/",
        "https://example.com",
        "https://example.com/download",
    ] {
        assert!(document_source(url, None).is_none(), "{url}");
        assert!(!is_document(url, None), "{url}");
    }
    for content_type in [
        "application/octet-stream",
        "text/plain",
        "image/png",
        "application/zip",
        "",
    ] {
        assert!(
            document_source("https://example.com/download", Some(content_type)).is_none(),
            "{content_type}"
        );
    }
}

#[test]
fn document_source_falls_back_to_the_content_type() {
    for (url, content_type, name, extension) in [
        (
            "https://example.com/download",
            "application/pdf",
            "download.pdf",
            ".pdf",
        ),
        (
            "https://example.com/download",
            "APPLICATION/PDF; charset=binary",
            "download.pdf",
            ".pdf",
        ),
        (
            "https://example.com/download",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "download.docx",
            ".docx",
        ),
        (
            "https://example.com/",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "document.xlsx",
            ".xlsx",
        ),
        (
            "https://example.com/download",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "download.pptx",
            ".pptx",
        ),
        (
            "https://example.com/download",
            "application/vnd.oasis.opendocument.text",
            "download.odt",
            ".odt",
        ),
        (
            "https://example.com/download",
            "text/markdown",
            "download.md",
            ".md",
        ),
        (
            "https://example.com/download",
            "text/x-markdown",
            "download.md",
            ".md",
        ),
        (
            "https://example.com/download",
            "text/csv",
            "download.csv",
            ".csv",
        ),
        (
            "https://example.com/download",
            "text/tab-separated-values",
            "download.tsv",
            ".tsv",
        ),
        (
            "https://example.com/download",
            "application/json",
            "download.json",
            ".json",
        ),
    ] {
        let source = document_source(url, Some(content_type))
            .unwrap_or_else(|| panic!("{content_type} is a document"));
        assert_eq!(source.name, name, "{content_type}");
        assert_eq!(source.extension, extension, "{content_type}");
    }
    // A URL suffix the loaders do not read falls through to the content type.
    let source = document_source("https://example.com/download.bin", Some("application/pdf"))
        .expect("the content type decides");
    assert_eq!(source.name, "download.bin.pdf");
    assert_eq!(source.extension, ".pdf");
    // A supported suffix wins over the header.
    let source = document_source("https://example.com/notes.md", Some("application/pdf"))
        .expect("the suffix decides");
    assert_eq!(source.name, "notes.md");
    assert_eq!(source.extension, ".md");
}

#[test]
fn page_from_documents_joins_text_and_reads_the_markdown() {
    let mut titled = Document::with_id("doc-1", "# Guide\n\nRun it.\n\n```sh\ncargo run\n```");
    titled
        .metadata
        .insert("title".to_owned(), Value::String("Guide Doc".to_owned()));
    let blank = Document::with_id("doc-2", "  \n\n");
    let page = page_from_documents("https://example.com/report.pdf", &[titled, blank]);
    assert_eq!(page.url, "https://example.com/report.pdf");
    assert_eq!(page.title, "Guide Doc");
    assert_eq!(page.text, "# Guide\n\nRun it.\n\n```sh\ncargo run\n```");
    assert_eq!(
        page.headings,
        vec![Heading {
            level: 1,
            text: "Guide".to_owned(),
        }]
    );
    assert_eq!(page.code_blocks, vec!["cargo run".to_owned()]);
    assert_eq!(page.description, "");
    assert_eq!(page.links, Vec::new());
    assert_eq!(page.images, Vec::new());

    // Without a metadata title the first level-1 heading becomes the title.
    let document = Document::with_id("doc-3", "no heading\n\n# Second\n\nbody");
    let page = page_from_documents("https://example.com/a.md", &[document]);
    assert_eq!(page.title, "Second");
    assert_eq!(page.headings.len(), 1);

    // A blank metadata title is ignored, and a heading-less text keeps it empty.
    let mut untitled = Document::with_id("doc-4", "plain text");
    untitled
        .metadata
        .insert("title".to_owned(), Value::String("   ".to_owned()));
    let page = page_from_documents("https://example.com/a.txt", &[untitled]);
    assert_eq!(page.title, "");
    assert_eq!(page.text, "plain text");

    let page = page_from_documents("https://example.com/a.pdf", &[]);
    assert!(page.is_blank());
    assert_eq!(page.title, "");
}

#[tokio::test]
async fn a_pdf_resource_becomes_a_page() {
    let url = "https://example.com/files/paper.pdf";
    let body = fixtures::pdf_document(&fixtures::plain_stream("BT (Hello PDF) Tj ET"));
    let page = extract_page(url, "application/pdf", body).await;
    assert!(page.text.contains("Hello PDF"), "{:?}", page.text);
    assert_eq!(page.title, "paper");
    assert_eq!(page.description, "");
    assert_eq!(page.headings, Vec::new());
    assert_eq!(page.code_blocks, Vec::<String>::new());
}

#[tokio::test]
async fn an_office_container_is_extracted_by_its_loader() {
    let docx = fixtures::docx_document(&["Intro", "Body"]).expect("builds a docx");
    let page = extract_page(
        "https://example.com/files/notes.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        docx,
    )
    .await;
    assert_eq!(page.text, "Intro\nBody");
    assert_eq!(page.title, "notes");

    let xlsx = fixtures::xlsx_workbook(&[("People", &["Name", "alice"])]).expect("builds an xlsx");
    let page = extract_page(
        "https://example.com/files/people.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        xlsx,
    )
    .await;
    assert!(page.text.contains("alice"), "{:?}", page.text);
    assert_eq!(page.title, "people");

    let pptx = fixtures::pptx_slides(&[("Deck", &["Agenda"])]).expect("builds a pptx");
    let page = extract_page(
        "https://example.com/files/talk.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        pptx,
    )
    .await;
    assert_eq!(page.text, "## Slide 1\nDeck\nAgenda");
    assert_eq!(page.title, "talk");
    assert_eq!(
        page.headings,
        vec![Heading {
            level: 2,
            text: "Slide 1".to_owned(),
        }]
    );

    let odt = fixtures::odt_document(&["Intro", "Body"]).expect("builds an odt");
    let page = extract_page(
        "https://example.com/files/memo.odt",
        "application/vnd.oasis.opendocument.text",
        odt,
    )
    .await;
    assert_eq!(page.text, "Intro\nBody");
    assert_eq!(page.title, "memo");
}

#[tokio::test]
async fn a_corrupt_document_falls_back_to_plain_text() {
    let url = "https://example.com/files/broken.pdf";
    let page = document_page(
        &resource(
            url,
            Some("application/pdf"),
            b"plain words, not a pdf".to_vec(),
        ),
        url,
    )
    .await
    .expect("a failed loader is never fatal")
    .expect("a document resource");
    assert_eq!(page.text, "plain words, not a pdf");
    assert_eq!(page.title, "");
    assert_eq!(page.headings, Vec::new());
    assert_eq!(page.code_blocks, Vec::<String>::new());
}

#[tokio::test]
async fn a_document_that_yields_nothing_names_the_url_and_format() {
    let url = "https://example.com/files/empty.pdf";
    let error = document_page(
        &resource(url, Some("application/pdf"), b" \n\t\n".to_vec()),
        url,
    )
    .await
    .expect_err("an unusable document");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("extracted no text"), "{message}");
    assert!(message.contains(url), "{message}");
    assert!(message.contains(".pdf"), "{message}");
}

#[tokio::test]
async fn a_non_document_resource_is_left_to_the_plain_text_route() {
    for (url, content_type) in [
        ("https://example.com/page.html", Some("text/html")),
        ("https://example.com/download", None),
        (
            "https://example.com/download",
            Some("application/octet-stream"),
        ),
    ] {
        let page = document_page(&resource(url, content_type, b"<html/>".to_vec()), url)
            .await
            .expect("no document is not an error")
            .map(|page| page.text);
        assert!(page.is_none(), "{url} {content_type:?}");
    }
}
