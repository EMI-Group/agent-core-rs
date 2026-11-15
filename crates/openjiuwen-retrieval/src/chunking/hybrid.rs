//! Hybrid chunker: keep table units whole, delegate everything else.
//!
//! Port of Python
//! `openjiuwen/core/retrieval/indexing/processor/chunker/hybrid_chunker.py`.
//! Documents matched by a no-split predicate are emitted as exactly one chunk;
//! every other document is handed to the inner chunker unchanged.
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde_json::Value;

use crate::common::{Document, TextChunk};

use super::{stamp_chunks, Chunker};

/// Predicate deciding that a document must stay a single chunk.
pub type NoSplitPredicate = Arc<dyn Fn(&Document) -> bool + Send + Sync>;

/// Wraps an inner chunker and keeps selected documents whole.
///
/// Python's `HybridChunker` is the recommended chunker for mixed Word/Excel
/// ingestion: spreadsheet row and column documents become one chunk each (their
/// `source_type` metadata is `"row"`/`"column"`, stamped by this crate's CSV and
/// XLSX loaders and by [`TableData`](crate::multimodal::TableData)), while every
/// other document is split by the inner chunker.
///
/// [`Chunker::chunk_text`] always delegates: the predicate only sees whole
/// documents, so it cannot influence a bare text split.
pub struct HybridChunker {
    /// Chunker used for documents the predicate does not match.
    inner: Arc<dyn Chunker>,
    /// Predicate that marks a document as a single chunk.
    no_split_when: NoSplitPredicate,
}

impl HybridChunker {
    /// Wraps `inner` with the default no-split predicate.
    ///
    /// The default predicate is [`default_no_split`]: `source_type` metadata is
    /// `"row"` or `"column"`.
    pub fn new(inner: Arc<dyn Chunker>) -> Self {
        Self {
            inner,
            no_split_when: Arc::new(default_no_split),
        }
    }

    /// Wraps `inner` with a custom no-split predicate.
    pub fn with_predicate(inner: Arc<dyn Chunker>, predicate: NoSplitPredicate) -> Self {
        Self {
            inner,
            no_split_when: predicate,
        }
    }

    /// Returns the wrapped chunker.
    pub fn inner(&self) -> &Arc<dyn Chunker> {
        &self.inner
    }

    /// Returns whether `document` must be emitted as a single chunk.
    ///
    /// A matched document with blank text is still delegated, so this predicate
    /// alone does not decide the output shape; see [`Chunker::chunk_documents`].
    pub fn should_keep_whole(&self, document: &Document) -> bool {
        (self.no_split_when)(document)
    }
}

impl Chunker for HybridChunker {
    fn chunk_text(&self, text: &str) -> Result<Vec<String>> {
        self.inner.chunk_text(text)
    }

    /// Splits documents, keeping predicate matches whole.
    ///
    /// A matched document whose text is non-blank after trimming becomes one
    /// chunk holding `text.trim()`, with `chunk_index = 0`, `total_chunks = 1`
    /// and `chunk_id` equal to the fresh chunk id. Every other document goes
    /// through `inner.chunk_documents` one document at a time, so the inner
    /// chunker's `chunk_index`/`total_chunks` stay document-relative. Output
    /// order is the input document order; no merging or de-duplication happens.
    fn chunk_documents(&self, documents: &[Document]) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        for document in documents {
            let trimmed = document.text.trim();
            if !trimmed.is_empty() && self.should_keep_whole(document) {
                chunks.extend(stamp_chunks(document, vec![trimmed.to_owned()]));
            } else {
                chunks.extend(self.inner.chunk_documents(std::slice::from_ref(document))?);
            }
        }
        Ok(chunks)
    }

    fn chunk_size(&self) -> usize {
        self.inner.chunk_size()
    }

    fn chunk_overlap(&self) -> usize {
        self.inner.chunk_overlap()
    }
}

impl fmt::Debug for HybridChunker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HybridChunker")
            .field("chunk_size", &self.inner.chunk_size())
            .field("chunk_overlap", &self.inner.chunk_overlap())
            .finish_non_exhaustive()
    }
}

/// Python's `_default_no_split`: table units (`source_type` row/column) stay whole.
pub fn default_no_split(document: &Document) -> bool {
    matches!(
        document.metadata.get("source_type"),
        Some(Value::String(kind)) if kind == "row" || kind == "column"
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    use super::super::CharChunker;

    fn hybrid(size: usize, overlap: usize) -> HybridChunker {
        HybridChunker::new(Arc::new(CharChunker::new(size, overlap).unwrap()))
    }

    // Port of tests/unit_tests/core/retrieval/indexing/processor/chunker/test_hybrid_chunker.py
    #[test]
    fn default_predicate_matches_row_and_column_units_only() {
        let row = Document::with_id("1", "a").metadata_entry("source_type", json!("row"));
        let column = Document::with_id("2", "a").metadata_entry("source_type", json!("column"));
        let paragraph = Document::with_id("3", "a").metadata_entry("source_type", json!("paragraph"));
        let titled = Document::with_id("4", "a").metadata_entry("title", json!("hello"));
        let bare = Document::with_id("5", "a");
        assert!(default_no_split(&row));
        assert!(default_no_split(&column));
        assert!(!default_no_split(&paragraph));
        assert!(!default_no_split(&titled));
        assert!(!default_no_split(&bare));
    }

    #[test]
    fn parameters_delegate_to_the_inner_chunker() {
        let chunker = hybrid(256, 30);
        assert_eq!(chunker.chunk_size(), 256);
        assert_eq!(chunker.chunk_overlap(), 30);
        assert_eq!(chunker.inner().chunk_size(), 256);
    }

    #[test]
    fn chunk_text_ignores_the_predicate() {
        let chunker = hybrid(10, 2);
        let text = "hello world, this is a test";
        assert_eq!(
            chunker.chunk_text(text).unwrap(),
            CharChunker::new(10, 2).unwrap().chunk_text(text).unwrap()
        );
    }

    #[test]
    fn row_and_column_documents_stay_whole() {
        let chunker = hybrid(10, 2);
        let row = Document::with_id("row1", "  姓名: 张三, 部门: 研发, 工号: 1001  ")
            .metadata_entry("source_type", json!("row"))
            .metadata_entry("sheet_name", json!("Sheet1"))
            .metadata_entry("row_index", json!(2));
        let column = Document::with_id("col1", "列名: 姓名。取值: 张三, 李四, 王五")
            .metadata_entry("source_type", json!("column"))
            .metadata_entry("column_name", json!("姓名"));

        let chunks = chunker.chunk_documents(&[row, column]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].doc_id, "row1");
        assert_eq!(chunks[0].text, "姓名: 张三, 部门: 研发, 工号: 1001");
        assert_eq!(chunks[1].doc_id, "col1");
        assert_eq!(chunks[1].text, "列名: 姓名。取值: 张三, 李四, 王五");
        for chunk in &chunks {
            assert_eq!(chunk.metadata.get("chunk_index"), Some(&json!(0)));
            assert_eq!(chunk.metadata.get("total_chunks"), Some(&json!(1)));
            assert_eq!(
                chunk.metadata.get("chunk_id"),
                Some(&json!(chunk.id.clone()))
            );
            assert!(chunk.embedding.is_none());
        }
        assert_eq!(chunks[0].metadata.get("sheet_name"), Some(&json!("Sheet1")));
        assert_eq!(chunks[0].metadata.get("row_index"), Some(&json!(2)));
        assert_eq!(chunks[1].metadata.get("column_name"), Some(&json!("姓名")));
        assert_ne!(chunks[0].id, chunks[1].id);
    }

    #[test]
    fn other_documents_delegate_with_document_relative_indices() {
        let chunker = hybrid(10, 2);
        let document = Document::with_id(
            "doc1",
            "This is a long document that should be split into multiple chunks by the inner chunker",
        )
        .metadata_entry("title", json!("test"));
        let chunks = chunker.chunk_documents(&[document.clone()]).unwrap();
        let inner_chunks = CharChunker::new(10, 2)
            .unwrap()
            .chunk_documents(&[document])
            .unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(chunks.len(), inner_chunks.len());
        assert!(chunks.iter().all(|chunk| chunk.doc_id == "doc1"));
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.metadata.get("chunk_index"),
                Some(&json!(u64::try_from(index).unwrap()))
            );
            assert_eq!(
                chunk.metadata.get("total_chunks"),
                Some(&json!(u64::try_from(chunks.len()).unwrap()))
            );
            assert_eq!(chunk.text, inner_chunks[index].text);
        }
    }

    #[test]
    fn mixed_documents_keep_input_order() {
        let chunker = hybrid(10, 2);
        let documents = [
            Document::with_id("row1", "姓名: 张三").metadata_entry("source_type", json!("row")),
            Document::with_id(
                "doc1",
                "This is a long document that should be split into multiple chunks",
            ),
            Document::with_id("col1", "列名: 部门").metadata_entry("source_type", json!("column")),
        ];
        let chunks = chunker.chunk_documents(&documents).unwrap();
        let row_chunks: Vec<&TextChunk> =
            chunks.iter().filter(|chunk| chunk.doc_id == "row1").collect();
        let column_chunks: Vec<&TextChunk> = chunks
            .iter()
            .filter(|chunk| chunk.doc_id == "col1")
            .collect();
        let document_chunks: Vec<&TextChunk> = chunks
            .iter()
            .filter(|chunk| chunk.doc_id == "doc1")
            .collect();
        assert_eq!(row_chunks.len(), 1);
        assert_eq!(column_chunks.len(), 1);
        assert!(document_chunks.len() > 1);
        assert_eq!(chunks.first().map(|chunk| chunk.doc_id.as_str()), Some("row1"));
        assert_eq!(chunks.last().map(|chunk| chunk.doc_id.as_str()), Some("col1"));
    }

    #[test]
    fn blank_text_matched_documents_delegate() {
        let chunker = hybrid(512, 50);
        let inner = CharChunker::new(512, 50).unwrap();
        for text in ["   ", ""] {
            let document =
                Document::with_id("row1", text).metadata_entry("source_type", json!("row"));
            let chunks = chunker.chunk_documents(std::slice::from_ref(&document)).unwrap();
            let inner_chunks = inner
                .chunk_documents(std::slice::from_ref(&document))
                .unwrap();
            // The hybrid chunker contributes nothing of its own: the inner
            // chunker decides, so a whitespace-only document keeps its
            // (untrimmed) window and an empty document yields no chunk.
            let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
            let inner_texts: Vec<&str> = inner_chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect();
            assert_eq!(texts, inner_texts);
        }
        let empty = Document::with_id("row1", "").metadata_entry("source_type", json!("row"));
        assert!(chunker
            .chunk_documents(std::slice::from_ref(&empty))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn custom_predicate_overrides_the_default() {
        let chunker = HybridChunker::with_predicate(
            Arc::new(CharChunker::new(10, 2).unwrap()),
            Arc::new(|document: &Document| {
                document.metadata.get("keep_whole") == Some(&json!(true))
            }),
        );
        let keep = Document::with_id("a", "short text").metadata_entry("keep_whole", json!(true));
        let split = Document::with_id(
            "b",
            "This is a long document that should be split by the inner chunker",
        )
        .metadata_entry("source_type", json!("row"));

        let chunks = chunker.chunk_documents(&[keep, split]).unwrap();
        assert_eq!(chunks.iter().filter(|chunk| chunk.doc_id == "a").count(), 1);
        assert!(chunks.iter().filter(|chunk| chunk.doc_id == "b").count() > 1);
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        let chunks = hybrid(512, 50).chunk_documents(&[]).unwrap();
        assert!(chunks.is_empty());
    }
}
