#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the in-memory document-loading path.
//!
//! The `DocumentLoader::load_bytes` family is a Rust extension: Python's parsers
//! only accept paths. These tests prove the extension is behaviour-preserving —
//! every format loads from bytes with the same documents, text, and metadata as
//! the same bytes written to a temporary file — and that the shared
//! [`openjiuwen_retrieval::loaders::fixtures`] builders are usable from outside
//! the crate (they replace the per-test `ZIP` writers).
//!
//! Everything is offline and deterministic: fixtures come from
//! `loaders::fixtures` or inline literals, ids are only checked for shape or for
//! the documented prefix rules, and no test depends on wall-clock ordering.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_retrieval::loaders::{fixtures, DocumentBytes, DocxLoader, LoadOptions};
use openjiuwen_retrieval::parsing::{ZipArchive, ZipWriter};
use openjiuwen_retrieval::{AutoLoader, Document, DocumentLoader, Metadata, UrlLoader};
use serde_json::Value;

/// The provenance keys stamped by `AutoLoader`, which legitimately differ
/// between a path source and an in-memory source with the same file name.
const VOLATILE_KEYS: [&str; 2] = ["doc_id", "file_path"];

/// Returns the metadata without the path-dependent provenance keys.
fn comparable(metadata: &Metadata) -> Metadata {
    let mut comparable = metadata.clone();
    for key in VOLATILE_KEYS {
        comparable.remove(key);
    }
    comparable
}

/// Returns the file stem of `name` (`notes.txt` -> `notes`).
fn file_stem(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(stem, _)| stem)
}

/// Loads `bytes` as `name` through [`AutoLoader::load_bytes`] with no overrides.
async fn load_bytes(name: &str, bytes: &[u8]) -> Vec<Document> {
    AutoLoader::new()
        .load_bytes(name, bytes, &LoadOptions::new())
        .await
        .unwrap()
}

/// Asserts that the in-memory path reproduces the path loader for one source.
///
/// The documents must line up one for one with the same text; every metadata key
/// except `doc_id`/`file_path` must match too (ids are name- or path-derived for
/// the multi-document loaders, and `file_path` records the source that was used).
async fn assert_matches_path(name: &str, bytes: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();

    let loader = AutoLoader::new();
    let from_path = loader.load(path.to_str().unwrap()).await.unwrap();
    let from_bytes = load_bytes(name, bytes).await;

    assert_eq!(
        from_bytes.len(),
        from_path.len(),
        "{name}: document count from bytes vs path"
    );
    for (index, (byte_document, path_document)) in
        from_bytes.iter().zip(from_path.iter()).enumerate()
    {
        assert_eq!(
            byte_document.text, path_document.text,
            "{name}: text of document {index}"
        );
        assert_eq!(
            comparable(&byte_document.metadata),
            comparable(&path_document.metadata),
            "{name}: metadata of document {index}"
        );
        // Both paths through `AutoLoader` stamp the same name-derived title.
        assert_eq!(
            byte_document.metadata.get("title"),
            Some(&Value::from(file_stem(name))),
            "{name}: title of document {index}"
        );
        assert_eq!(
            byte_document.metadata.get("file_ext"),
            Some(&Value::from(format!(
                ".{}",
                name.rsplit_once('.').unwrap().1
            ))),
            "{name}: file_ext of document {index}"
        );
    }
}

#[tokio::test]
async fn text_formats_load_the_same_documents_from_bytes_and_paths() {
    let cases: [(&str, &[u8]); 6] = [
        ("notes.txt", b"  first line\nsecond line  \n"),
        ("readme.md", b"# Heading\n\nBody\n"),
        ("data.json", br#"{"name":"alice","tags":["a","b"]}"#),
        ("people.csv", b"name,city\nalice,paris\nbob,\n"),
        ("table.tsv", b"id\tnote\r\n1\t\"hello, world\"\r\n"),
        (
            "page.html",
            b"<html><head><title>Docs</title></head>\
              <body><p>Hello <b>there</b></p></body></html>",
        ),
    ];
    for (name, bytes) in cases {
        assert_matches_path(name, bytes).await;
    }

    // The delimited formats key their multi-document ids by the source name, so
    // the byte path uses the supplied name as the id base.
    let documents = load_bytes("people.csv", b"name,city\nalice,paris\nbob,\n").await;
    let ids: Vec<&str> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec![
            "people.csv_s0_r2",
            "people.csv_s0_r3",
            "people.csv_s0_c0",
            "people.csv_s0_c1"
        ]
    );
    assert_eq!(documents[0].text, "name: alice, city: paris");

    let documents = load_bytes("table.tsv", b"id\tnote\r\n1\t\"hello, world\"\r\n").await;
    assert_eq!(documents[0].id, "table.tsv_s0_r2");
    assert_eq!(documents[0].text, "id: 1, note: hello, world");

    let documents = load_bytes("notes.txt", b"  first line\nsecond line  \n").await;
    assert_eq!(documents[0].text, "first line\nsecond line");

    let documents = load_bytes(
        "page.html",
        b"<html><head><title>Docs</title></head><body><p>Hello <b>there</b></p></body></html>",
    )
    .await;
    assert_eq!(documents[0].text, "Docs Hello there");
    assert_eq!(
        documents[0].metadata.get("source_type"),
        Some(&Value::from("web_page"))
    );
}

#[tokio::test]
async fn binary_formats_load_the_same_documents_from_bytes_and_paths() {
    let docx = fixtures::docx_document(&["Intro", "Body & <text>"]).unwrap();
    let xlsx =
        fixtures::xlsx_workbook(&[("People", &["Name,City", "alice,paris", "bob,"])]).unwrap();
    let pdf = fixtures::pdf_document(&fixtures::plain_stream("BT (Hello PDF) Tj ET"));
    let pptx = fixtures::pptx_slides(&[("Rust rewrite", &["Agenda", "Offline"]), ("Closing", &[])])
        .unwrap();
    let odt = fixtures::odt_document(&["Intro", "Body & <text>"]).unwrap();

    for (name, bytes) in [
        ("notes.docx", docx.as_slice()),
        ("people.xlsx", xlsx.as_slice()),
        ("paper.pdf", pdf.as_slice()),
        ("talk.pptx", pptx.as_slice()),
        ("memo.odt", odt.as_slice()),
    ] {
        assert_matches_path(name, bytes).await;
    }

    // Pin the rendered text and the format-specific metadata each reader adds.
    let documents = load_bytes("notes.docx", &docx).await;
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "Intro\nBody & <text>");
    assert_eq!(
        documents[0].metadata.get("paragraph_count"),
        Some(&Value::from(2))
    );
    assert_eq!(
        documents[0].metadata.get("table_count"),
        Some(&Value::from(0))
    );
    assert_eq!(
        documents[0].metadata.get("source_type"),
        Some(&Value::from("document"))
    );

    let documents = load_bytes("people.xlsx", &xlsx).await;
    let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "Name: alice, City: paris",
            "Name: bob, City: ",
            "Column name: Name. Values: alice, bob",
            "Column name: City. Values: paris",
        ]
    );
    // The workbook loader keys ids by the file stem, which is the same for a
    // path and for the supplied name.
    let ids: Vec<&str> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec![
            "people_s0_r2",
            "people_s0_r3",
            "people_s0_c0",
            "people_s0_c1"
        ]
    );

    let documents = load_bytes("paper.pdf", &pdf).await;
    assert_eq!(documents[0].text, "Hello PDF");
    assert_eq!(
        documents[0].metadata.get("page_count"),
        Some(&Value::from(1))
    );
    assert_eq!(
        documents[0].metadata.get("has_text"),
        Some(&Value::from(true))
    );

    let documents = load_bytes("talk.pptx", &pptx).await;
    assert_eq!(
        documents[0].text,
        "## Slide 1\nRust rewrite\nAgenda\nOffline\n## Slide 2\nClosing"
    );
    assert_eq!(
        documents[0].metadata.get("slide_count"),
        Some(&Value::from(2))
    );
    assert_eq!(
        documents[0].metadata.get("has_notes"),
        Some(&Value::from(false))
    );

    let documents = load_bytes("memo.odt", &odt).await;
    assert_eq!(documents[0].text, "Intro\nBody & <text>");
    assert_eq!(
        documents[0].metadata.get("paragraph_count"),
        Some(&Value::from(2))
    );
    assert_eq!(
        documents[0].metadata.get("table_count"),
        Some(&Value::from(0))
    );
}

#[tokio::test]
async fn load_bytes_stamps_doc_id_title_and_provenance() {
    let loader = AutoLoader::new();

    // Single-document loaders take `doc_id` verbatim and overwrite the title.
    let documents = loader
        .load_bytes(
            "notes.txt",
            &b"body"[..],
            &LoadOptions::new().with_doc_id("doc-1").with_title("Custom"),
        )
        .await
        .unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].id, "doc-1");
    assert_eq!(
        documents[0].metadata.get("doc_id"),
        Some(&Value::from("doc-1"))
    );
    assert_eq!(
        documents[0].metadata.get("title"),
        Some(&Value::from("Custom"))
    );
    assert_eq!(
        documents[0].metadata.get("file_path"),
        Some(&Value::from("notes.txt"))
    );
    assert_eq!(
        documents[0].metadata.get("file_ext"),
        Some(&Value::from(".txt"))
    );

    // Without options the loader-generated id is stamped and the title defaults
    // to the file stem.
    let documents = load_bytes("notes.txt", b"body").await;
    assert_eq!(documents[0].id.len(), 36);
    assert_eq!(
        documents[0].metadata.get("doc_id"),
        Some(&Value::from(documents[0].id.as_str()))
    );
    assert_eq!(
        documents[0].metadata.get("title"),
        Some(&Value::from("notes"))
    );
    assert_eq!(
        documents[0].metadata.get("file_path"),
        Some(&Value::from("notes.txt"))
    );

    // The binary single-document loaders follow the same rule.
    let documents = loader
        .load_bytes(
            "memo.odt",
            fixtures::odt_document(&["Body"]).unwrap(),
            &LoadOptions::new().with_doc_id("doc-9"),
        )
        .await
        .unwrap();
    assert_eq!(documents[0].id, "doc-9");
    assert_eq!(
        documents[0].metadata.get("doc_id"),
        Some(&Value::from("doc-9"))
    );
}

#[tokio::test]
async fn load_bytes_uses_doc_id_as_the_base_prefix_for_multi_document_sources() {
    let loader = AutoLoader::new();

    // CSV/TSV are multi-document, so `doc_id` replaces the id base prefix while
    // every generated `_s{sheet}_r{row}`/`_s{sheet}_c{col}` suffix survives.
    let documents = loader
        .load_bytes(
            "people.csv",
            &b"Name,Age\nAlice,30\n"[..],
            &LoadOptions::new().with_doc_id("doc-base"),
        )
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

    // XLSX keys ids by the file stem, which `doc_id` also replaces.
    let documents = loader
        .load_bytes(
            "people.xlsx",
            fixtures::xlsx_workbook(&[("People", &["Name,City", "alice,paris"])]).unwrap(),
            &LoadOptions::new().with_doc_id("doc-base"),
        )
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

    // The title override applies to every document a multi-document source emits.
    let documents = loader
        .load_bytes(
            "people.csv",
            &b"Name,Age\nAlice,30\n"[..],
            &LoadOptions::new().with_title("Custom"),
        )
        .await
        .unwrap();
    for document in &documents {
        assert_eq!(document.metadata.get("title"), Some(&Value::from("Custom")));
        assert_eq!(
            document.metadata.get("file_path"),
            Some(&Value::from("people.csv"))
        );
        assert_eq!(
            document.metadata.get("file_ext"),
            Some(&Value::from(".csv"))
        );
    }
}

#[tokio::test]
async fn unsupported_byte_sources_are_typed_errors() {
    let loader = AutoLoader::new();

    let error = loader
        .load_bytes("archive.xyz", b"unknown".to_vec(), &LoadOptions::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("Unsupported format: .xyz"),
        "unexpected message: {error}"
    );

    // A name without an extension has nothing to dispatch on.
    let error = loader
        .load_bytes("README", b"body".to_vec(), &LoadOptions::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("Unsupported format: "),
        "unexpected message: {error}"
    );

    // A URL name has no extension, so it is not a fetch request either.
    let error = loader
        .load_bytes(
            "https://example.com/page",
            b"<p>hi</p>".to_vec(),
            &LoadOptions::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    // The URL loader is fetch-only: bytes cannot be turned into a web page.
    let source = DocumentBytes::new("page.html", b"<p>hi</p>".to_vec());
    let error = UrlLoader::new()
        .load_bytes(&source, &LoadOptions::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error
            .to_string()
            .contains("URL loader needs an HTTP(S) source; use load(): page.html"),
        "unexpected message: {error}"
    );

    // The same override is reached through the trait object (not the trait's
    // generic `Unsupported` default).
    let loader: Arc<dyn DocumentLoader> = Arc::new(UrlLoader::new());
    let error = loader
        .load_bytes(&source, &LoadOptions::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error
        .to_string()
        .contains("URL loader needs an HTTP(S) source"));
}

#[tokio::test]
async fn empty_bytes_are_typed_errors_not_panics() {
    let loader = AutoLoader::new();

    // ZIP-based office formats reject an empty container.
    for name in ["empty.docx", "empty.xlsx", "empty.pptx", "empty.odt"] {
        let error = loader
            .load_bytes(name, Vec::new(), &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{name}: {error}");
    }

    // An empty PDF parses to no text, which the loader reports rather than
    // returning a blank document (there is no OCR).
    let error = loader
        .load_bytes("empty.pdf", Vec::new(), &LoadOptions::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    assert!(
        error
            .to_string()
            .contains("PDF has no extractable text: empty.pdf"),
        "unexpected message: {error}"
    );

    // Text formats reject content with no non-whitespace text.
    for (name, bytes) in [
        ("empty.txt", &b" \n\t\n"[..]),
        ("empty.md", &b"\n"[..]),
        ("empty.html", &b"<html><script>x</script></html>"[..]),
    ] {
        let error = loader
            .load_bytes(name, bytes, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{name}: {error}");
    }

    let error = loader
        .load_bytes("empty.json", &b"\n"[..], &LoadOptions::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
}

#[tokio::test]
async fn document_bytes_and_options_builders_and_trait_object_dispatch() {
    let source = DocumentBytes::new("Report.DOCX", vec![1_u8, 2, 3]);
    assert_eq!(source.source(), "Report.DOCX");
    assert_eq!(source.name, "Report.DOCX");
    assert_eq!(source.extension(), ".docx");
    assert_eq!(source.len(), 3);
    assert!(!source.is_empty());

    let nameless = DocumentBytes::new("README", Vec::new());
    assert_eq!(nameless.extension(), "");
    assert!(nameless.is_empty());
    assert_eq!(nameless.len(), 0);

    let options = LoadOptions::new().with_doc_id("doc-2").with_title("Report");
    assert_eq!(options.doc_id.as_deref(), Some("doc-2"));
    assert_eq!(options.title.as_deref(), Some("Report"));
    assert_eq!(LoadOptions::default(), LoadOptions::new());

    // `AutoLoader` reached through the trait takes the byte path instead of the
    // trait's `Unsupported` default.
    let loader: Arc<dyn DocumentLoader> = Arc::new(AutoLoader::new());
    let documents = loader
        .load_bytes(
            &DocumentBytes::new("notes.txt", &b"body"[..]),
            &LoadOptions::new().with_doc_id("doc-3"),
        )
        .await
        .unwrap();
    assert_eq!(documents[0].id, "doc-3");
    assert_eq!(documents[0].text, "body");

    // A concrete format loader keeps the same contract through the trait object.
    let loader: Arc<dyn DocumentLoader> = Arc::new(DocxLoader::new());
    let documents = loader
        .load_bytes(
            &DocumentBytes::new("notes.docx", fixtures::docx_document(&["Only"]).unwrap()),
            &LoadOptions::new(),
        )
        .await
        .unwrap();
    assert_eq!(documents[0].text, "Only");
    assert_eq!(
        documents[0].metadata.get("doc_id"),
        Some(&Value::from(documents[0].id.as_str()))
    );
}

#[tokio::test]
async fn zip_writer_archives_round_trip_and_feed_the_loaders() {
    // `parsing::ZipWriter` and `parsing::ZipArchive` are the public halves of
    // the crate's ZIP support, so an external crate can build an archive.
    let mut writer = ZipWriter::new();
    writer
        .add_stored("a.txt", "first")
        .unwrap()
        .add_stored("b.txt", "second")
        .unwrap();
    let bytes = writer.finish().unwrap();

    let archive = ZipArchive::from_bytes(bytes.clone()).unwrap();
    assert_eq!(archive.entry_names(), vec!["a.txt", "b.txt"]);
    assert_eq!(archive.read_text("a.txt").unwrap(), "first");
    assert_eq!(archive.read_text("b.txt").unwrap(), "second");
    assert_eq!(archive.archive_size(), bytes.len());

    // Empty or over-long entry names are rejected, not silently skipped.
    let error = ZipWriter::new().add_stored("", "body").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    // `loaders::fixtures::build_zip` makes the same writer available as a
    // one-call fixture, so an external test can assemble a DOCX inline.
    let document_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>Inline body</w:t></w:r></w:p></w:body>
</w:document>"#;
    let docx = fixtures::build_zip(&[("word/document.xml", document_xml)]).unwrap();
    let documents = load_bytes("inline.docx", &docx).await;
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "Inline body");
    assert_eq!(
        documents[0].metadata.get("paragraph_count"),
        Some(&Value::from(1))
    );
    assert_eq!(
        documents[0].metadata.get("file_path"),
        Some(&Value::from("inline.docx"))
    );
}

/// Proves a trait-object loader still gets the typed default when it only
/// implements the path API, so a byte call never silently returns no documents.
struct PathOnlyLoader;

#[async_trait]
impl DocumentLoader for PathOnlyLoader {
    fn supports(&self, _source: &str) -> bool {
        false
    }

    async fn load(&self, _source: &str) -> Result<Vec<Document>> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn the_trait_default_rejects_bytes_for_path_only_loaders() {
    let error = PathOnlyLoader
        .load_bytes(
            &DocumentBytes::new("data.pdf", &b"bytes"[..]),
            &LoadOptions::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("PathOnlyLoader"),
        "unexpected message: {error}"
    );
    assert!(
        error
            .to_string()
            .contains("does not support in-memory bytes"),
        "unexpected message: {error}"
    );
}
