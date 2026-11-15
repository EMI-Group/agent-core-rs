#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]
//! End-to-end tests for the document loaders exposed by the crate root.
//!
//! Ports the behavior of the Python parser tests in
//! `tests/unit_tests/core/retrieval/indexing/processor/parser/` (text, JSON,
//! CSV/TSV, HTML, web page, and auto dispatch). File-backed loaders run against a
//! `tempfile` directory; the URL loader and the auto-dispatch URL branch run
//! against a loopback `wiremock` server. Everything is offline and deterministic:
//! generated ids are only checked for shape, never for a concrete value.
//! Binary `DOCX`/`XLSX`/`PDF`/`PPTX`/`ODT` bytes come from the public
//! `loaders::fixtures` builders, so the suite owns no `ZIP` writer of its own.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_retrieval::loaders::{fixtures, DocxLoader};
use openjiuwen_retrieval::{
    extract_title, html_to_text, AutoLoader, CsvLoader, Document, DocumentLoader, HtmlLoader,
    JsonLoader, LoaderRegistry, MarkdownLoader, TextLoader, UrlLoader,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Writes `contents` into `dir/name` and returns the path as a `String`.
fn write_file(dir: &tempfile::TempDir, name: &str, contents: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, contents).unwrap();
    path.to_str().unwrap().to_owned()
}

/// A loader that always returns one fixed document.
///
/// Used to prove that extension registration keeps the first loader and that
/// [`AutoLoader::register`] cannot shadow a default.
struct FixedLoader {
    /// Text the fixed document carries.
    text: &'static str,
}

#[async_trait]
impl DocumentLoader for FixedLoader {
    fn supports(&self, _source: &str) -> bool {
        true
    }

    async fn load(&self, _source: &str) -> Result<Vec<Document>> {
        Ok(vec![Document::with_id("fixed", self.text)])
    }
}

#[tokio::test]
async fn text_loader_reports_missing_empty_and_trims_content() {
    let dir = tempfile::tempdir().unwrap();
    let loader = TextLoader::new();

    let missing = dir.path().join("missing.txt");
    let error = loader.load(missing.to_str().unwrap()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("does not exist"),
        "unexpected message: {error}"
    );

    let empty = write_file(&dir, "empty.txt", " \n\t\n");
    let error = loader.load(&empty).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let path = write_file(&dir, "notes.txt", "  first line\nsecond line  \n");
    let documents = loader.load(&path).await.unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "first line\nsecond line");
    assert_eq!(documents[0].id.len(), 36);
    assert!(documents[0].metadata.is_empty());
}

#[tokio::test]
async fn markdown_loader_reads_markdown_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "readme.markdown", "# Heading\n\nbody\n");

    let documents = MarkdownLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "# Heading\n\nbody");
}

#[tokio::test]
async fn json_loader_pretty_prints_and_keeps_malformed_text() {
    let dir = tempfile::tempdir().unwrap();
    let loader = JsonLoader::new();

    let valid = write_file(&dir, "data.json", r#"{"a":1,"b":[true,null]}"#);
    let documents = loader.load(&valid).await.unwrap();
    assert!(documents[0].text.contains('\n'), "expected pretty output");
    let parsed: Value = serde_json::from_str(&documents[0].text).unwrap();
    assert_eq!(parsed, serde_json::json!({"a": 1, "b": [true, null]}));

    let malformed = write_file(&dir, "broken.json", "{not json}\n");
    let documents = loader.load(&malformed).await.unwrap();
    assert_eq!(documents[0].text, "{not json}\n");

    let empty = write_file(&dir, "empty.json", "\n");
    let error = loader.load(&empty).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn csv_loader_emits_row_documents_then_column_documents() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "people.csv", "name,city\nalice,paris\nbob,\n");

    let documents = CsvLoader::new().load(&path).await.unwrap();
    let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "name: alice, city: paris",
            "name: bob, city: ",
            "Column name: name. Values: alice, bob",
            "Column name: city. Values: paris",
        ]
    );

    // Row ids count the header row, so the first data row is row 2.
    assert_eq!(documents[0].id, format!("{path}_s0_r2"));
    assert_eq!(documents[1].id, format!("{path}_s0_r3"));
    assert_eq!(documents[2].id, format!("{path}_s0_c0"));
    assert_eq!(documents[3].id, format!("{path}_s0_c1"));

    let row = &documents[0].metadata;
    assert_eq!(row.get("sheet_name"), Some(&Value::from("people")));
    assert_eq!(row.get("row_index"), Some(&Value::from(2_u64)));
    assert_eq!(row.get("source_type"), Some(&Value::from("row")));

    let column = &documents[2].metadata;
    assert_eq!(column.get("sheet_name"), Some(&Value::from("people")));
    assert_eq!(column.get("column_name"), Some(&Value::from("name")));
    assert_eq!(column.get("source_type"), Some(&Value::from("column")));
    assert!(column.get("row_index").is_none());
}

#[tokio::test]
async fn csv_header_only_file_yields_column_documents_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "header.csv", "alpha,beta\n");

    let documents = CsvLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0].text, "Column name: alpha. Values: (empty)");
    assert_eq!(documents[1].text, "Column name: beta. Values: (empty)");
}

#[tokio::test]
async fn csv_include_header_false_keeps_values_and_positional_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "values.csv", "first,second\nx,\ny,z\n");

    let documents = CsvLoader::new()
        .with_include_header(false)
        .load(&path)
        .await
        .unwrap();
    let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "x",
            "y, z",
            "Column name: column_0. Values: x, y",
            "Column name: column_1. Values: z",
        ]
    );
    assert_eq!(
        documents[2].metadata.get("column_name"),
        Some(&Value::from("column_0"))
    );
}

#[tokio::test]
async fn csv_tsv_uses_tabs_and_decodes_quoted_escaped_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(
        &dir,
        "table.tsv",
        "id\tnote\r\n1\t\"hello, world\"\r\n2\t\"say \"\"hi\"\"\"\r\n",
    );

    let documents = CsvLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 4);
    assert_eq!(documents[0].text, "id: 1, note: hello, world");
    assert_eq!(documents[1].text, "id: 2, note: say \"hi\"");
    assert_eq!(documents[2].text, "Column name: id. Values: 1, 2");
    assert_eq!(
        documents[3].text,
        "Column name: note. Values: hello, world, say \"hi\""
    );
}

#[test]
fn html_helpers_strip_scripts_and_prefer_the_open_graph_title() {
    let html = "<html><head><title>Plain title</title>\
                <meta property=\"og:title\" content=\"Open &amp; Graph\">\
                <style>.hidden { color: red; }</style>\
                <script>var x = 1;</script></head>\
                <body><p>Hello&nbsp;world &#8212; &#x41;</p></body></html>";
    assert_eq!(extract_title(html), Some("Open & Graph".to_owned()));
    assert_eq!(html_to_text(html), "Plain title Hello world \u{2014} A");
}

#[tokio::test]
async fn html_loader_builds_one_web_page_document() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(
        &dir,
        "page.html",
        "<html><head><title>Docs</title><script>ignored()</script></head>\
         <body><h1>Hello</h1><p>world</p></body></html>",
    );

    let documents = HtmlLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "Docs Hello world");
    assert_eq!(
        documents[0].metadata.get("title"),
        Some(&Value::from("Docs"))
    );
    assert_eq!(
        documents[0].metadata.get("source_type"),
        Some(&Value::from("web_page"))
    );
}

#[tokio::test]
async fn html_loader_rejects_pages_without_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "empty.html", "<html><script>x</script></html>");

    let error = HtmlLoader::new().load(&path).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("No text content found in HTML"));
}

#[tokio::test]
async fn url_loader_reads_html_pages_with_title_and_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "<html><head><title>Article</title></head><body><p>Body &amp; text</p></body></html>",
            "text/html; charset=utf-8",
        ))
        .mount(&server)
        .await;

    let url = format!("{}/article", server.uri());
    let documents = UrlLoader::new()
        .with_timeout(Duration::from_secs(10))
        .load(&url)
        .await
        .unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "Article Body & text");
    assert_eq!(documents[0].id, url);
    assert_eq!(
        documents[0].metadata.get("title"),
        Some(&Value::from("Article"))
    );
    assert_eq!(
        documents[0].metadata.get("source_url"),
        Some(&Value::from(url.as_str()))
    );
    assert_eq!(
        documents[0].metadata.get("source_type"),
        Some(&Value::from("web_page"))
    );
}

#[tokio::test]
async fn url_loader_reads_plain_text_bodies() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/plain"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("  plain text body \n", "text/plain; charset=utf-8"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/plain", server.uri());
    let documents = UrlLoader::new()
        .with_timeout(Duration::from_secs(10))
        .load(&url)
        .await
        .unwrap();
    assert_eq!(documents[0].text, "plain text body");
    assert_eq!(documents[0].metadata.get("title"), Some(&Value::from("")));
}

#[tokio::test]
async fn url_loader_maps_error_statuses_and_bad_schemes() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let url = format!("{}/broken", server.uri());
    let error = UrlLoader::new().load(&url).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(
        error.to_string().contains("Web page request failed: 500"),
        "unexpected message: {error}"
    );

    let error = UrlLoader::new()
        .load("ftp://example.com/file")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[tokio::test]
async fn auto_loader_dispatches_by_extension_and_stamps_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let loader = AutoLoader::new();

    let cases = [
        ("notes.txt", "plain body\n", "plain body"),
        ("notes.md", "# Title\n", "# Title"),
        (
            "page.html",
            "<html><body><p>html body</p></body></html>",
            "html body",
        ),
    ];
    for (name, contents, expected) in cases {
        let path = write_file(&dir, name, contents);
        let documents = loader.load(&path).await.unwrap();
        assert_eq!(documents.len(), 1, "{name}");
        assert_eq!(documents[0].text, expected, "{name}");
        let metadata = &documents[0].metadata;
        assert_eq!(
            metadata.get("doc_id"),
            Some(&Value::from(documents[0].id.as_str()))
        );
        let stem = name.rsplit_once('.').unwrap().0;
        assert_eq!(metadata.get("title"), Some(&Value::from(stem)), "{name}");
        assert_eq!(metadata.get("file_path"), Some(&Value::from(path.as_str())));
        let extension = name.rsplit_once('.').unwrap().1;
        assert_eq!(
            metadata.get("file_ext"),
            Some(&Value::from(format!(".{extension}"))),
            "{name}"
        );
    }

    // CSV keeps its own ids and gets them stamped as `doc_id`.
    let csv = write_file(&dir, "rows.csv", "a\n1\n");
    let documents = loader.load(&csv).await.unwrap();
    assert_eq!(documents[0].id, format!("{csv}_s0_r2"));
    assert_eq!(
        documents[0].metadata.get("doc_id"),
        Some(&Value::from(documents[0].id.as_str()))
    );
    assert_eq!(
        documents[0].metadata.get("file_ext"),
        Some(&Value::from(".csv"))
    );
}

#[tokio::test]
async fn auto_loader_rejects_unknown_formats_and_missing_files() {
    let dir = tempfile::tempdir().unwrap();
    let loader = AutoLoader::new();

    for name in [
        "missing.txt",
        "missing.docx",
        "missing.xlsx",
        "missing.pdf",
        "missing.png",
    ] {
        let missing = dir.path().join(name);
        let error = loader.load(missing.to_str().unwrap()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound, "{name}");
        assert!(
            error.to_string().contains("does not exist"),
            "unexpected message: {error}"
        );
    }

    let unknown = write_file(&dir, "archive.xyz", "unknown bytes");
    let error = loader.load(&unknown).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("Unsupported format: .xyz"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn auto_loader_routes_urls_through_the_url_loader() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/auto"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("<html><title>Auto</title><p>page</p></html>", "text/html"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/auto", server.uri());
    let documents = AutoLoader::new().load(&url).await.unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].id, url);
    assert_eq!(documents[0].text, "Auto page");
    // URLs are returned as the URL loader built them: no file provenance.
    assert!(documents[0].metadata.get("file_path").is_none());
    assert_eq!(
        documents[0].metadata.get("source_type"),
        Some(&Value::from("web_page"))
    );
}

#[tokio::test]
async fn auto_loader_keeps_default_loaders_when_reregistering() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "notes.TXT", "from the file\n");

    let mut loader = AutoLoader::new();
    loader.register(&["TXT"], Arc::new(FixedLoader { text: "fixed" }));
    let documents = loader.load(&path).await.unwrap();
    assert_eq!(documents[0].text, "from the file");
    assert_eq!(
        documents[0].metadata.get("file_ext"),
        Some(&Value::from(".txt"))
    );

    // The custom loader is still reachable for its own extension.
    loader.register(&["fixed"], Arc::new(FixedLoader { text: "fixed" }));
    let documents = loader.load(&write_file(&dir, "x.fixed", "")).await.unwrap();
    assert_eq!(documents[0].text, "fixed");
}

#[tokio::test]
async fn loader_registry_is_case_insensitive_and_first_registration_wins() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "notes.TXT", "from the file\n");

    let mut defaults = LoaderRegistry::with_defaults();
    defaults.register(&["TXT"], Arc::new(FixedLoader { text: "fixed" }));
    let loader = defaults.get_for_path(&path).unwrap();
    assert_eq!(loader.load(&path).await.unwrap()[0].text, "from the file");

    let mut custom = LoaderRegistry::new();
    custom.register(&["TXT"], Arc::new(FixedLoader { text: "fixed" }));
    let loader = custom.get_for_path(&path).unwrap();
    assert_eq!(loader.load(&path).await.unwrap()[0].text, "fixed");

    assert!(defaults.get_for_path("archive.pdf").is_some());
    assert!(defaults.get_for_path("notes.docx").is_some());
    assert!(defaults.get_for_path("sheet.xlsx").is_some());
    assert!(defaults.get_for_path("photo.PNG").is_some());
    assert!(defaults.get_for_path("legacy.xls").is_none());
    assert!(defaults.get_for_path("archive.zip").is_none());
    // Text/Markdown/JSON/CSV/TSV/HTML, DOCX, ODT, XLSX, PDF, PPTX, and the 8
    // image extensions.
    assert!(defaults.get_for_path("memo.odt").is_some());
    assert!(defaults.get_for_path("deck.pptx").is_some());
    assert_eq!(defaults.extensions().len(), 19);
}

// ---------------------------------------------------------------------------
// Binary and image fixtures
// ---------------------------------------------------------------------------

/// Writes raw `bytes` into `dir/name` and returns the path as a `String`.
fn write_bytes(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_str().unwrap().to_owned()
}

/// `word/document.xml`: heading, paragraph, and a two-row table.
///
/// The shared `fixtures::docx_document` builder only emits plain paragraphs, so
/// this test keeps its own richer part and wraps it with
/// [`fixtures::build_zip`].
const DOCX_DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
    <w:p><w:r><w:t>Body text</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Notes</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>alice</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>ok</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>"#;

/// `docProps/core.xml` carrying `dc:title`.
const DOCX_CORE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="urn:cp" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Design notes</dc:title>
</cp:coreProperties>"#;

/// Builds the `DOCX` archive used by the loader tests with the shared writer.
fn docx_archive() -> Vec<u8> {
    fixtures::build_zip(&[
        ("word/document.xml", DOCX_DOCUMENT),
        ("docProps/core.xml", DOCX_CORE),
    ])
    .unwrap()
}

/// Builds the one-sheet `XLSX` archive used by the loader tests.
fn xlsx_archive() -> Vec<u8> {
    fixtures::xlsx_workbook(&[("People", &["Name,City", "alice,paris"])]).unwrap()
}

// ---------------------------------------------------------------------------
// Binary and image dispatch through AutoLoader
// ---------------------------------------------------------------------------

#[tokio::test]
async fn auto_loader_reads_docx_headings_and_tables() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "notes.docx", &docx_archive());

    let documents = AutoLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 1);
    let document = &documents[0];
    assert_eq!(
        document.text,
        "## Overview\nBody text\n| Name | Notes |\n| --- | --- |\n| alice | ok |"
    );
    let metadata = &document.metadata;
    assert_eq!(metadata.get("paragraph_count"), Some(&Value::from(2)));
    assert_eq!(metadata.get("table_count"), Some(&Value::from(1)));
    assert_eq!(metadata.get("source_type"), Some(&Value::from("document")));
    // AutoLoader stamps its provenance keys over the loader's own metadata.
    assert_eq!(
        metadata.get("doc_id"),
        Some(&Value::from(document.id.as_str()))
    );
    assert_eq!(metadata.get("title"), Some(&Value::from("notes")));
    assert_eq!(metadata.get("file_ext"), Some(&Value::from(".docx")));
    assert_eq!(metadata.get("file_path"), Some(&Value::from(path.as_str())));
}

#[tokio::test]
async fn docx_loader_keeps_the_core_properties_title() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "notes.docx", &docx_archive());

    let documents = DocxLoader::new().load(&path).await.unwrap();
    assert_eq!(
        documents[0].metadata.get("title"),
        Some(&Value::from("Design notes"))
    );
}

#[tokio::test]
async fn auto_loader_reads_xlsx_row_and_column_documents() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "people.xlsx", &xlsx_archive());

    let documents = AutoLoader::new().load(&path).await.unwrap();
    let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "Name: alice, City: paris",
            "Column name: Name. Values: alice",
            "Column name: City. Values: paris",
        ]
    );
    // Ids are keyed by the file stem, mirroring Python's `base_id = doc`.
    let ids: Vec<&str> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    assert_eq!(ids, vec!["people_s0_r2", "people_s0_c0", "people_s0_c1"]);

    let row = &documents[0].metadata;
    assert_eq!(row.get("sheet_name"), Some(&Value::from("People")));
    assert_eq!(row.get("row_index"), Some(&Value::from(2_u64)));
    assert_eq!(row.get("source_type"), Some(&Value::from("row")));
    assert_eq!(row.get("file_ext"), Some(&Value::from(".xlsx")));
    assert_eq!(row.get("doc_id"), Some(&Value::from("people_s0_r2")));

    let column = &documents[1].metadata;
    assert_eq!(column.get("column_name"), Some(&Value::from("Name")));
    assert_eq!(column.get("source_type"), Some(&Value::from("column")));
    assert_eq!(column.get("doc_id"), Some(&Value::from("people_s0_c0")));
}

#[tokio::test]
async fn auto_loader_uses_the_doc_id_as_the_base_prefix_for_xlsx() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "people.xlsx", &xlsx_archive());

    let documents = AutoLoader::new()
        .load_as(&path, Some("doc-base"), None)
        .await
        .unwrap();
    let ids: Vec<&str> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["doc-base_s0_r2", "doc-base_s0_c0", "doc-base_s0_c1"]
    );
    for document in &documents {
        assert_eq!(
            document.metadata.get("doc_id"),
            Some(&Value::from(document.id.as_str()))
        );
    }
}

#[tokio::test]
async fn auto_loader_reads_images_as_captioned_documents() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "cat.png", b"\x89PNG\r\n");

    let documents = AutoLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 1);
    let document = &documents[0];
    assert_eq!(document.text, "[image] cat.png");
    let metadata = &document.metadata;
    assert_eq!(
        metadata.get("image_path"),
        Some(&Value::from(path.as_str()))
    );
    assert_eq!(metadata.get("mime_type"), Some(&Value::from("image/png")));
    assert_eq!(metadata.get("modality"), Some(&Value::from("image")));
    assert_eq!(metadata.get("file_ext"), Some(&Value::from(".png")));
    assert_eq!(metadata.get("title"), Some(&Value::from("cat")));
    assert_eq!(
        metadata.get("doc_id"),
        Some(&Value::from(document.id.as_str()))
    );
}

#[tokio::test]
async fn auto_loader_reads_pdf_text_and_page_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(
        &dir,
        "paper.pdf",
        &fixtures::pdf_document(&fixtures::plain_stream("BT (Hello PDF) Tj ET")),
    );

    let documents = AutoLoader::new().load(&path).await.unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "Hello PDF");
    assert_eq!(
        documents[0].metadata.get("page_count"),
        Some(&Value::from(1))
    );
    assert_eq!(
        documents[0].metadata.get("has_text"),
        Some(&Value::from(true))
    );
    assert_eq!(
        documents[0].metadata.get("pdf_incomplete"),
        Some(&Value::from(false))
    );
    assert_eq!(
        documents[0].metadata.get("pdf_skipped_filters"),
        Some(&Value::Array(Vec::new()))
    );
}

#[tokio::test]
async fn auto_loader_rejects_scanned_pdfs() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(
        &dir,
        "scanned.pdf",
        &fixtures::pdf_document(
            "4 0 obj\n<< /Filter /DCTDecode /Length 4 >>\nstream\nscan\nendstream\nendobj",
        ),
    );

    let error = AutoLoader::new().load(&path).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("PDF has no extractable text"),
        "unexpected message: {error}"
    );
}
