//! Document chunking strategies.
//!
//! Port of the Python `openjiuwen/core/retrieval/indexing/processor/chunker/`
//! package: the [`Chunker`] trait with its chunk-stamping contract, fixed-size
//! character windows ([`CharChunker`]), recursive character splitting
//! ([`RecursiveCharChunker`]), token approximation ([`TokenChunker`]), and a
//! name-based factory registry ([`ChunkerRegistry`]).
//!
//! Chunking is pure CPU work: every method is synchronous and deterministic
//! except for the generated chunk identifiers.
mod char;
mod recursive;
mod registry;
mod token;

use openjiuwen_common::error::{Error, Result};
use uuid::Uuid;

use crate::common::{Document, TextChunk};

pub use char::CharChunker;
pub use recursive::RecursiveCharChunker;
pub use registry::{ChunkerFactory, ChunkerRegistry};
pub use token::{TokenChunker, TokenCounter, WhitespaceTokenCounter};

pub use crate::common::{DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE};

/// Splits documents into chunks.
///
/// Implementations only need [`Chunker::chunk_text`], [`Chunker::chunk_size`],
/// and [`Chunker::chunk_overlap`]; the [`Chunker::chunk_documents`] default
/// implementation stamps chunk metadata the same way Python's base
/// `Chunker.chunk_documents` does.
pub trait Chunker: Send + Sync {
    /// Splits a single text into chunk strings.
    ///
    /// Chunk order matches the source text; an empty text yields no chunks.
    fn chunk_text(&self, text: &str) -> Result<Vec<String>>;

    /// Splits documents into [`TextChunk`]s, document by document.
    ///
    /// Iteration is document-major: all chunks of the first document come
    /// first. Every chunk gets a fresh `uuid4` id, copied document metadata,
    /// and the extra keys `chunk_index` (0-based within the document),
    /// `total_chunks` (that document's chunk count), and `chunk_id` (the same
    /// uuid as `id`).
    fn chunk_documents(&self, documents: &[Document]) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        for doc in documents {
            let texts = self.chunk_text(&doc.text)?;
            let total_chunks = texts.len();
            for (chunk_index, text) in texts.into_iter().enumerate() {
                let id = Uuid::new_v4().to_string();
                let mut metadata = doc.metadata.clone();
                metadata.insert(
                    "chunk_index".to_owned(),
                    serde_json::Value::from(chunk_index),
                );
                metadata.insert(
                    "total_chunks".to_owned(),
                    serde_json::Value::from(total_chunks),
                );
                metadata.insert("chunk_id".to_owned(), serde_json::Value::from(id.clone()));
                let mut chunk = TextChunk::from_document(doc, text, Some(id));
                chunk.metadata = metadata;
                chunks.push(chunk);
            }
        }
        Ok(chunks)
    }

    /// Configured chunk size, in characters (or tokens for [`TokenChunker`]).
    fn chunk_size(&self) -> usize;

    /// Configured overlap, in the same unit as [`Chunker::chunk_size`].
    fn chunk_overlap(&self) -> usize;
}

/// Validates chunk window parameters shared by every chunker.
///
/// `chunk_size` must be greater than zero and `chunk_overlap` must be smaller
/// than it; an overlap of zero is allowed.
pub fn validate_params(chunk_size: usize, chunk_overlap: usize) -> Result<()> {
    if chunk_size == 0 {
        return Err(Error::Validation(format!(
            "chunk_size must be greater than 0, current value: {chunk_size}"
        )));
    }
    if chunk_overlap >= chunk_size {
        return Err(Error::Validation(format!(
            "chunk_overlap must be less than chunk_size, current values: chunk_overlap={chunk_overlap}, chunk_size={chunk_size}"
        )));
    }
    Ok(())
}

/// Counts characters (Unicode scalar values), matching Python's `len(str)`.
pub(crate) fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Splits `text` into character windows of at most `size` characters.
///
/// Mirrors Python's `CharSplitter.split`: windows start at character index
/// `start`, end at `min(len, start + size)`, and advance by `size - overlap`.
/// Slicing happens on character boundaries, so multibyte input is safe.
///
/// The caller must guarantee `size > overlap` (see [`validate_params`]).
pub(crate) fn char_windows(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || size == 0 {
        return Vec::new();
    }
    let mut boundaries: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    boundaries.push(text.len());
    let total = boundaries.len() - 1;
    let step = size - overlap.min(size - 1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < total {
        let end = total.min(start + size);
        chunks.push(text[boundaries[start]..boundaries[end]].to_owned());
        start += step;
    }
    chunks
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashSet;

    fn document(id: &str, text: &str) -> Document {
        Document::with_id(id, text)
    }

    #[test]
    fn chunk_documents_stamps_metadata_and_ids() {
        let doc = Document::with_id("doc-1", "abcdefghij")
            .metadata_entry("source", json!("unit"))
            .metadata_entry("chunk_index", json!(99));
        let chunks = CharChunker::new(5, 0)
            .unwrap()
            .chunk_documents(&[doc])
            .unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "abcde");
        assert_eq!(chunks[1].text, "fghij");
        assert_eq!(chunks[0].doc_id, "doc-1");
        assert!(chunks.iter().all(|chunk| chunk.embedding.is_none()));
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.get("source"), Some(&json!("unit")));
            // Stamp keys win over pre-existing document metadata.
            assert_eq!(chunk.metadata.get("chunk_index"), Some(&json!(index)));
            assert_eq!(chunk.metadata.get("total_chunks"), Some(&json!(2)));
            assert_eq!(chunk.metadata.get("chunk_id"), Some(&json!(chunk.id)));
            assert!(Uuid::parse_str(&chunk.id).is_ok());
        }
    }

    #[test]
    fn chunk_documents_is_document_major_and_ids_are_unique() {
        let chunks = CharChunker::new(4, 0)
            .unwrap()
            .chunk_documents(&[
                document("doc-a", "12345678"),
                document("doc-b", ""),
                document("doc-c", "xyz"),
            ])
            .unwrap();
        let doc_ids: Vec<&str> = chunks.iter().map(|chunk| chunk.doc_id.as_str()).collect();
        assert_eq!(doc_ids, ["doc-a", "doc-a", "doc-c"]);
        assert_eq!(chunks[0].metadata.get("chunk_index"), Some(&json!(0)));
        assert_eq!(chunks[1].metadata.get("chunk_index"), Some(&json!(1)));
        assert_eq!(chunks[2].metadata.get("chunk_index"), Some(&json!(0)));
        assert_eq!(chunks[2].metadata.get("total_chunks"), Some(&json!(1)));
        let ids: HashSet<&String> = chunks.iter().map(|chunk| &chunk.id).collect();
        assert_eq!(ids.len(), chunks.len());
    }

    #[test]
    fn validate_params_rejects_invalid_windows() {
        assert!(validate_params(0, 0).is_err());
        assert!(validate_params(10, 10).is_err());
        assert!(validate_params(10, 11).is_err());
        assert!(validate_params(10, 0).is_ok());
        assert!(validate_params(1, 0).is_ok());
        let error = validate_params(0, 0).unwrap_err();
        assert!(error.to_string().contains("current value: 0"));
    }
}
