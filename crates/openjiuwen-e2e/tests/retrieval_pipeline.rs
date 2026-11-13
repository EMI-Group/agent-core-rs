//! Retrieval pipeline scenario: office fixtures → loaders → chunking → `BM25` → knowledge base.
//!
//! This is the cross-crate seam of the offline retrieval stack: hand-built
//! `DOCX`/`XLSX`/`PDF` fixtures are written into a temp directory, loaded by the
//! real `AutoLoader`, chunked by the real `CharChunker`, indexed in the real
//! `Bm25Index`, served through `SparseRetriever`, and ingested into a
//! `SimpleKnowledgeBase` configured with the `bm25` index type. No crate's own
//! unit tests are duplicated here; the scenario proves that the stages compose.
//!
//! Everything is offline and deterministic. The office fixtures are ordinary
//! `ZIP` containers (built below with stored, uncompressed entries so the test
//! needs no compression dependency) and the `PDF` is a plain uncompressed
//! content stream, so the pure-Rust parsers in `openjiuwen-retrieval` produce
//! byte-identical text on every run. Assertions are made only on deterministic
//! values — fixture text, the ids derived from the `XLSX` file stem, metadata
//! counts, scores and their ordering — never on the `uuid4` ids the loaders and
//! chunkers generate (those are only read back from the producer and compared
//! against each other).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]

mod support;

use std::sync::Arc;

use openjiuwen::core::common::error::ErrorKind;
use openjiuwen::core::retrieval::chunking::{CharChunker, Chunker};
use openjiuwen::core::retrieval::loaders::{AutoLoader, DocumentLoader, DocxLoader};
use openjiuwen::core::retrieval::retriever::{RetrieveMode, Retriever, SparseRetriever};
use openjiuwen::core::retrieval::sparse::{Bm25Config, Bm25Index, Bm25Query};
use openjiuwen::core::retrieval::{
    Document, HashEmbedding, InMemoryVectorStore, KnowledgeBaseConfig, SimpleKnowledgeBase,
    TextChunk, INDEX_TYPE_BM25,
};
use serde_json::Value;
use support::within;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// `ZIP` fixture builder (the library's own builder is crate-private)
// ---------------------------------------------------------------------------

/// Builds a `ZIP` archive with stored (uncompressed) entries.
///
/// `DOCX`/`XLSX` fixtures are ordinary `ZIP` containers; writing the archive by
/// hand keeps the scenario free of a compression dependency. The checksums come
/// from the crate's public `parsing::crc32`, so the archives are exactly what a
/// real office suite would accept.
fn build_zip(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut directory = Vec::new();
    let count = entries.len() as u16;
    for (name, contents) in entries {
        let name = name.as_bytes();
        let payload = contents.as_bytes();
        let offset = data.len() as u32;
        let crc = openjiuwen::core::retrieval::parsing::crc32(payload);
        let size = payload.len() as u32;
        data.extend_from_slice(&0x0403_4B50u32.to_le_bytes());
        data.extend_from_slice(&20u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&crc.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(payload);
        directory.extend_from_slice(&0x0201_4B50u32.to_le_bytes());
        directory.extend_from_slice(&20u16.to_le_bytes());
        directory.extend_from_slice(&20u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&crc.to_le_bytes());
        directory.extend_from_slice(&size.to_le_bytes());
        directory.extend_from_slice(&size.to_le_bytes());
        directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes());
        directory.extend_from_slice(&offset.to_le_bytes());
        directory.extend_from_slice(name);
    }
    let directory_offset = data.len() as u32;
    let directory_size = directory.len() as u32;
    data.extend_from_slice(&directory);
    data.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&count.to_le_bytes());
    data.extend_from_slice(&count.to_le_bytes());
    data.extend_from_slice(&directory_size.to_le_bytes());
    data.extend_from_slice(&directory_offset.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data
}

/// Writes raw `bytes` into `dir/name` and returns the path as a `String`.
fn write_bytes(dir: &TempDir, name: &str, bytes: &[u8]) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_str().unwrap().to_owned()
}

// ---------------------------------------------------------------------------
// Fixture parts
// ---------------------------------------------------------------------------

/// `word/document.xml`: heading, paragraph, and a two-row table.
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

/// The rendered `DOCX` body: heading, paragraph, and a Markdown table.
const DOCX_TEXT: &str = "## Overview\nBody text\n| Name | Notes |\n| --- | --- |\n| alice | ok |";

/// Builds the `DOCX` archive used by the scenario.
fn docx_archive() -> Vec<u8> {
    build_zip(&[
        ("word/document.xml", DOCX_DOCUMENT),
        ("docProps/core.xml", DOCX_CORE),
    ])
}

/// Workbook with one `People` sheet.
const XLSX_WORKBOOK: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="People" sheetId="1" r:id="rId1"/></sheets>
</workbook>"#;

/// Relationship pointing at the single worksheet part.
const XLSX_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#;

/// Worksheet with a `Name`/`City` header and one data row.
const XLSX_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Name</t></is></c>
      <c r="B1" t="inlineStr"><is><t>City</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>alice</t></is></c>
      <c r="B2" t="inlineStr"><is><t>paris</t></is></c>
    </row>
  </sheetData>
</worksheet>"#;

/// Text of the one data row, as the `XLSX` loader renders it.
const XLSX_ROW_TEXT: &str = "Name: alice, City: paris";

/// Deterministic id the `XLSX` loader derives from the file stem.
const XLSX_ROW_ID: &str = "people_s0_r2";

/// Builds the one-sheet `XLSX` archive used by the scenario.
fn xlsx_archive() -> Vec<u8> {
    build_zip(&[
        ("xl/workbook.xml", XLSX_WORKBOOK),
        ("xl/_rels/workbook.xml.rels", XLSX_RELS),
        ("xl/worksheets/sheet1.xml", XLSX_SHEET),
    ])
}

/// Builds one uncompressed `PDF` content-stream object.
fn plain_stream(content: &str) -> String {
    format!(
        "1 0 obj\n<< /Length {} >>\nstream\n{content}\nendstream\nendobj",
        content.len()
    )
}

/// Wraps `body` in a minimal `PDF` document.
fn pdf_document(body: &str) -> Vec<u8> {
    format!("%PDF-1.7\n{body}\n%%EOF\n").into_bytes()
}

// ---------------------------------------------------------------------------
// Scenario helpers
// ---------------------------------------------------------------------------

/// Writes the three fixtures into `dir` and loads them through one `AutoLoader`.
///
/// The corpus is document-major: one `DOCX` document, the three `XLSX`
/// row/column documents, and one `PDF` document.
async fn load_office_corpus(dir: &TempDir) -> Vec<Document> {
    let docx = write_bytes(dir, "notes.docx", &docx_archive());
    let xlsx = write_bytes(dir, "people.xlsx", &xlsx_archive());
    let pdf = write_bytes(
        dir,
        "paper.pdf",
        &pdf_document(&plain_stream("BT (Hello PDF) Tj ET")),
    );
    let loader = AutoLoader::new();
    let mut documents = Vec::new();
    for path in [docx, xlsx, pdf] {
        let loaded = within("loading a fixture", loader.load(&path))
            .await
            .unwrap();
        documents.extend(loaded);
    }
    documents
}

/// Splits the office corpus into chunks and indexes them, returning both.
///
/// The window is deliberately wider than every fixture text, so every document
/// contributes exactly one chunk and chunk order mirrors document order.
fn indexed_corpus(documents: &[Document]) -> (Vec<TextChunk>, Bm25Index) {
    let chunks = CharChunker::new(128, 16)
        .unwrap()
        .chunk_documents(documents)
        .unwrap();
    let mut index = Bm25Index::new(Bm25Config::default()).unwrap();
    assert_eq!(index.add_chunks(&chunks), chunks.len());
    (chunks, index)
}

// ---------------------------------------------------------------------------
// Loaders: one test per container format
// ---------------------------------------------------------------------------

#[tokio::test]
async fn docx_fixture_loads_paragraphs_and_tables() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "notes.docx", &docx_archive());
    let documents = within("loading notes.docx", AutoLoader::new().load(&path))
        .await
        .unwrap();
    assert_eq!(documents.len(), 1);
    let document = &documents[0];
    // The rendered body keeps the heading marker and the table as Markdown.
    assert_eq!(document.text, DOCX_TEXT);
    assert!(document.text.contains("| alice | ok |"));
    let metadata = &document.metadata;
    assert_eq!(metadata.get("source_type"), Some(&Value::from("document")));
    assert_eq!(metadata.get("paragraph_count"), Some(&Value::from(2)));
    assert_eq!(metadata.get("table_count"), Some(&Value::from(1)));
    assert_eq!(metadata.get("file_ext"), Some(&Value::from(".docx")));
    assert_eq!(metadata.get("file_path"), Some(&Value::from(path.as_str())));
    // The generated id is a uuid, so only the round trip is asserted, never the
    // value itself.
    assert_eq!(
        metadata.get("doc_id"),
        Some(&Value::from(document.id.as_str()))
    );
    // The auto loader stamps the file stem; the Word core properties are what
    // `DocxLoader` itself reports.
    assert_eq!(metadata.get("title"), Some(&Value::from("notes")));
    let direct = within("loading notes.docx directly", DocxLoader::new().load(&path))
        .await
        .unwrap();
    assert_eq!(
        direct[0].metadata.get("title"),
        Some(&Value::from("Design notes"))
    );
    assert_eq!(direct[0].text, DOCX_TEXT);
}

#[tokio::test]
async fn xlsx_fixture_loads_row_and_column_documents() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(&dir, "people.xlsx", &xlsx_archive());
    let documents = within("loading people.xlsx", AutoLoader::new().load(&path))
        .await
        .unwrap();
    // One row document plus one document per populated column.
    let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            XLSX_ROW_TEXT,
            "Column name: Name. Values: alice",
            "Column name: City. Values: paris",
        ]
    );
    let ids: Vec<&str> = documents.iter().map(|doc| doc.id.as_str()).collect();
    assert_eq!(ids, vec![XLSX_ROW_ID, "people_s0_c0", "people_s0_c1"]);
    let row = &documents[0].metadata;
    assert_eq!(row.get("sheet_name"), Some(&Value::from("People")));
    assert_eq!(row.get("row_index"), Some(&Value::from(2)));
    assert_eq!(row.get("source_type"), Some(&Value::from("row")));
    assert_eq!(row.get("doc_id"), Some(&Value::from(XLSX_ROW_ID)));
    assert_eq!(row.get("file_ext"), Some(&Value::from(".xlsx")));
    let column = &documents[1].metadata;
    assert_eq!(column.get("column_name"), Some(&Value::from("Name")));
    assert_eq!(column.get("source_type"), Some(&Value::from("column")));
    assert_eq!(column.get("sheet_name"), Some(&Value::from("People")));
    assert_eq!(column.get("doc_id"), Some(&Value::from("people_s0_c0")));
}

#[tokio::test]
async fn pdf_fixture_loads_extracted_text_and_page_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bytes(
        &dir,
        "paper.pdf",
        &pdf_document(&plain_stream("BT (Hello PDF) Tj ET")),
    );
    let documents = within("loading paper.pdf", AutoLoader::new().load(&path))
        .await
        .unwrap();
    assert_eq!(documents.len(), 1);
    let document = &documents[0];
    assert_eq!(document.text, "Hello PDF");
    let metadata = &document.metadata;
    assert_eq!(metadata.get("page_count"), Some(&Value::from(1)));
    assert_eq!(metadata.get("has_text"), Some(&Value::from(true)));
    assert_eq!(metadata.get("source_type"), Some(&Value::from("document")));
    assert_eq!(metadata.get("file_ext"), Some(&Value::from(".pdf")));
    assert_eq!(metadata.get("title"), Some(&Value::from("paper")));
    assert_eq!(
        metadata.get("doc_id"),
        Some(&Value::from(document.id.as_str()))
    );
}

// ---------------------------------------------------------------------------
// Chunking + BM25
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chunked_fixtures_are_ranked_by_the_bm25_index() {
    let dir = tempfile::tempdir().unwrap();
    let documents = load_office_corpus(&dir).await;
    assert_eq!(documents.len(), 5, "1 DOCX + 3 XLSX + 1 PDF document");
    let (chunks, index) = indexed_corpus(&documents);
    assert_eq!(
        chunks.len(),
        documents.len(),
        "128-char window keeps one chunk per document"
    );
    // Every chunk points back at its parent document, which is what turns a
    // lexical hit into a traceable retrieval result.
    for chunk in &chunks {
        assert!(documents.iter().any(|doc| doc.id == chunk.doc_id));
        assert_eq!(chunk.metadata.get("chunk_index"), Some(&Value::from(0)));
    }
    assert_eq!(index.len(), chunks.len());

    let results = index.search(&Bm25Query::new("alice").with_top_k(5));
    assert_eq!(results.len(), 3, "three chunks mention `alice`");
    // Only matching chunks are returned, and the shortest one wins on BM25
    // length normalisation: the spreadsheet row, which is also the only hit
    // whose id is deterministic.
    for result in &results {
        assert!(
            result.text.to_lowercase().contains("alice"),
            "non-matching hit: {}",
            result.text
        );
    }
    assert_eq!(results[0].text, XLSX_ROW_TEXT);
    assert_eq!(
        results[0].metadata.get("doc_id"),
        Some(&Value::from(XLSX_ROW_ID))
    );
    let row_chunk = chunks
        .iter()
        .find(|chunk| chunk.text == XLSX_ROW_TEXT)
        .expect("the spreadsheet row is one of the chunks");
    // The index echoes the producer's chunk id rather than inventing one.
    assert_eq!(results[0].id, row_chunk.id);
    assert_eq!(
        results[0].metadata.get("chunk_id"),
        Some(&Value::from(row_chunk.id.as_str()))
    );
    for pair in results.windows(2) {
        assert!(
            pair[0].score > pair[1].score,
            "scores must be strictly descending: {} then {}",
            pair[0].score,
            pair[1].score
        );
    }

    let stats = index.stats();
    assert_eq!(stats.document_count, chunks.len());
    assert!(stats.term_count > 0, "vocabulary must not be empty");
    assert!(stats.total_tokens > 0);
    assert!(stats.average_document_length > 0.0);
    assert!(stats.term_count <= stats.total_tokens);
    // `alice` is common in this corpus, so it is worth less than a term the
    // index has never seen.
    assert!(index.idf("alice") > 0.0);
    assert!(index.idf("unseenterm") > index.idf("alice"));
    assert!(index.contains(&row_chunk.id));
    assert!(!index.contains("missing-chunk"));
}

#[tokio::test]
async fn sparse_retriever_serves_bm25_and_rejects_other_modes() {
    let dir = tempfile::tempdir().unwrap();
    let documents = load_office_corpus(&dir).await;
    let (_chunks, index) = indexed_corpus(&documents);
    let retriever = SparseRetriever::new(Arc::new(index));
    assert_eq!(retriever.index().len(), 5);

    let results = within(
        "sparse retrieval",
        retriever.retrieve("alice", 5, None, RetrieveMode::Sparse, None),
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].text, XLSX_ROW_TEXT);
    assert_eq!(results[0].doc_id.as_deref(), Some(XLSX_ROW_ID));
    // `chunk_id` is the result id; the metadata copy keeps both links explicit.
    assert_eq!(
        results[0].chunk_id.as_deref(),
        Some(
            results[0]
                .metadata
                .get("chunk_id")
                .unwrap()
                .as_str()
                .unwrap()
        )
    );
    for pair in results.windows(2) {
        assert!(pair[0].score > pair[1].score);
    }

    let top_only = within(
        "top-1 sparse retrieval",
        retriever.retrieve("alice", 1, None, RetrieveMode::Sparse, None),
    )
    .await
    .unwrap();
    assert_eq!(top_only.len(), 1);
    assert_eq!(top_only[0].text, XLSX_ROW_TEXT);

    // The threshold is inclusive: the worst score still keeps its hit.
    let floor = results.last().unwrap().score;
    let at_floor = within(
        "thresholded sparse retrieval",
        retriever.retrieve("alice", 5, Some(floor), RetrieveMode::Sparse, None),
    )
    .await
    .unwrap();
    assert_eq!(at_floor.len(), results.len());
    let above_all = within(
        "over-threshold sparse retrieval",
        retriever.retrieve(
            "alice",
            5,
            Some(results[0].score + 1.0),
            RetrieveMode::Sparse,
            None,
        ),
    )
    .await
    .unwrap();
    assert!(above_all.is_empty());

    // A lexical retriever has no dense channel, so it rejects every other mode
    // instead of silently returning the sparse results.
    for mode in [RetrieveMode::Vector, RetrieveMode::Hybrid] {
        let error = within(
            "rejecting a non-sparse mode",
            retriever.retrieve("alice", 5, None, mode, None),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("only supports 'sparse' mode"),
            "unexpected message: {error}"
        );
        assert!(error.to_string().contains(mode.as_str()));
    }
    let error = within(
        "rejecting a non-sparse mode on the raw result path",
        retriever.retrieve_search_results("alice", 5, RetrieveMode::Hybrid, None),
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

// ---------------------------------------------------------------------------
// Knowledge base
// ---------------------------------------------------------------------------

#[tokio::test]
async fn knowledge_base_parses_files_and_retrieves_through_bm25() {
    let dir = tempfile::tempdir().unwrap();
    let docx = write_bytes(&dir, "notes.docx", &docx_archive());
    let xlsx = write_bytes(&dir, "people.xlsx", &xlsx_archive());

    let mut config = KnowledgeBaseConfig::new("kb-retrieval");
    config.index_type = INDEX_TYPE_BM25.to_owned();
    config.chunk_size = 128;
    config.chunk_overlap = 16;
    let chunker = Arc::new(CharChunker::new(128, 16).unwrap());
    let kb = SimpleKnowledgeBase::new(
        config,
        Arc::new(InMemoryVectorStore::new()),
        Arc::new(HashEmbedding::new(64)),
        chunker,
    );
    // The `bm25` index type is the only one that turns the sparse leg on by
    // itself.
    assert!(kb.has_sparse_index());
    assert_eq!(kb.id(), "kb-retrieval");

    let parsed = within(
        "parsing the fixtures",
        kb.parse_files(&[docx.clone(), xlsx.clone()]),
    )
    .await
    .unwrap();
    // Parsing is read-only: the knowledge base has not been fed yet.
    assert_eq!(parsed.len(), 4, "1 DOCX + 3 XLSX documents");
    assert_eq!(
        within("counting an empty base", kb.count()).await.unwrap(),
        0
    );
    assert_eq!(kb.sparse_index_len(), 0);

    let ids = within("ingesting the fixtures", kb.add_documents(parsed))
        .await
        .unwrap();
    assert_eq!(ids.len(), 4);
    for expected in [XLSX_ROW_ID, "people_s0_c0", "people_s0_c1"] {
        assert!(
            ids.iter().any(|id| id == expected),
            "missing deterministic id {expected} in {ids:?}"
        );
    }
    let count = within("counting the ingested base", kb.count())
        .await
        .unwrap();
    assert!(count > 0);
    // Every stored chunk is mirrored into the lexical index.
    assert_eq!(kb.sparse_index_len(), count);

    let hits = within("knowledge-base retrieval", kb.retrieve("alice"))
        .await
        .unwrap();
    assert!(!hits.is_empty());
    for hit in &hits {
        assert!(
            hit.text.to_lowercase().contains("alice"),
            "non-matching hit: {}",
            hit.text
        );
        assert!(hit.chunk_id.is_some());
        assert!(hit.metadata.contains_key("chunk_id"));
        assert!(hit.metadata.contains_key("doc_id"));
    }
    // The spreadsheet row is one of the parsed-and-ingested documents, so the
    // deterministic doc id must come back through the whole pipeline.
    assert!(
        hits.iter()
            .any(|hit| hit.doc_id.as_deref() == Some(XLSX_ROW_ID)),
        "expected a hit from {XLSX_ROW_ID}, got {:?}",
        hits.iter().map(|hit| &hit.doc_id).collect::<Vec<_>>()
    );
    assert_eq!(hits[0].text, XLSX_ROW_TEXT);

    let statistics = within("reading the statistics", kb.get_statistics())
        .await
        .unwrap();
    assert_eq!(statistics.kb_id, "kb-retrieval");
    assert_eq!(statistics.index_type, INDEX_TYPE_BM25);
    assert_eq!(statistics.chunk_count, count);
    assert_eq!(statistics.document_count, 4);
    assert_eq!(statistics.sparse_index_size, count);
    assert!(statistics.has_loader);
    assert!(!statistics.has_reranker);
    assert!(!statistics.has_graph);
}
