//! Preprocessing-aware chunker facade.
//!
//! Port of Python
//! `openjiuwen/core/retrieval/indexing/processor/chunker/chunking.py`: a fixed
//! size chunker that cleans each document with a [`PreprocessingPipeline`]
//! before it splits the text, selecting character or token windows through
//! [`ChunkUnit`].
use std::fmt;
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};

use crate::common::{
    Document, TextChunk, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
};

use super::preprocess::PreprocessingPipeline;
use super::{stamp_chunks, CharChunker, Chunker, TokenChunker};

/// Unit of the chunk window.
///
/// Python selects the unit with the string `chunk_unit` (`"char"` or `"token"`);
/// the string form is available through [`ChunkUnit::as_str`],
/// [`Display`](fmt::Display), and [`FromStr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkUnit {
    /// Window size and overlap counted in characters ([`CharChunker`]).
    #[default]
    Char,
    /// Window size and overlap counted in tokens ([`TokenChunker`]).
    Token,
}

impl ChunkUnit {
    /// Returns Python's `chunk_unit` string for this unit.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Token => "token",
        }
    }
}

impl fmt::Display for ChunkUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ChunkUnit {
    type Err = Error;

    /// Parses Python's `chunk_unit` strings (trimmed, case-insensitive).
    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "char" => Ok(Self::Char),
            "token" => Ok(Self::Token),
            other => Err(Error::Validation(format!(
                "chunk_unit must be 'char' or 'token', current value: {other}"
            ))),
        }
    }
}

/// Fixed size chunker that preprocesses text before splitting it.
///
/// Preprocessing happens here and nowhere else: [`CharChunker`],
/// [`TokenChunker`], and [`HybridChunker`] never touch the text they receive, so
/// a chunker used without this facade splits documents verbatim. An empty
/// pipeline (the default) is therefore a plain fixed-size chunker.
///
/// The metadata stamped on every chunk is identical to the [`Chunker`] default
/// implementation: `chunk_index` (0-based, per document), `total_chunks`, and
/// `chunk_id` (the chunk's own id).
pub struct TextChunker {
    /// Configured window size, in characters or tokens depending on `unit`.
    chunk_size: usize,
    /// Configured overlap, in the same unit.
    chunk_overlap: usize,
    /// Window unit and the chunker it selects.
    unit: ChunkUnit,
    /// Preprocessors applied before every split.
    pipeline: PreprocessingPipeline,
    /// Chunker built from the unit; never preprocesses on its own.
    inner: Box<dyn Chunker>,
}

impl TextChunker {
    /// Creates a chunker for `chunk_size`/`chunk_overlap` in `unit` windows.
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when the window parameters are invalid for the chosen unit.
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        unit: ChunkUnit,
        pipeline: PreprocessingPipeline,
    ) -> Result<Self> {
        let inner = build_inner(chunk_size, chunk_overlap, unit)?;
        Ok(Self {
            chunk_size,
            chunk_overlap,
            unit,
            pipeline,
            inner,
        })
    }

    /// Creates a chunker from Python's `preprocess_options` flags.
    ///
    /// `normalize_whitespace` installs a
    /// [`WhitespaceNormalizer`](super::WhitespaceNormalizer) and
    /// `remove_url_email` a [`URLEmailRemover`](super::URLEmailRemover), in that
    /// order, matching Python's option handling.
    pub fn from_options(
        chunk_size: usize,
        chunk_overlap: usize,
        unit: ChunkUnit,
        normalize_whitespace: bool,
        remove_url_email: bool,
    ) -> Result<Self> {
        Self::new(
            chunk_size,
            chunk_overlap,
            unit,
            PreprocessingPipeline::from_options(normalize_whitespace, remove_url_email),
        )
    }

    /// Replaces the preprocessing pipeline, keeping the window parameters.
    pub fn with_pipeline(mut self, pipeline: PreprocessingPipeline) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// Switches the window unit, rebuilding the inner chunker.
    pub fn with_unit(mut self, unit: ChunkUnit) -> Result<Self> {
        self.inner = build_inner(self.chunk_size, self.chunk_overlap, unit)?;
        self.unit = unit;
        Ok(self)
    }

    /// Returns the configured window size (characters or tokens).
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the configured overlap (characters or tokens).
    pub fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    /// Returns the window unit.
    pub fn chunk_unit(&self) -> ChunkUnit {
        self.unit
    }

    /// Returns the preprocessing pipeline.
    pub fn pipeline(&self) -> &PreprocessingPipeline {
        &self.pipeline
    }
}

impl Default for TextChunker {
    /// Python's defaults: 512 characters with 50 characters of overlap, no
    /// preprocessing.
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            unit: ChunkUnit::Char,
            pipeline: PreprocessingPipeline::default(),
            inner: Box::new(CharChunker::default()),
        }
    }
}

impl Chunker for TextChunker {
    /// Preprocesses `text`, then delegates the split to the inner chunker.
    fn chunk_text(&self, text: &str) -> Result<Vec<String>> {
        let processed = self.pipeline.process(text);
        self.inner.chunk_text(&processed)
    }

    /// Preprocesses every document, then splits and stamps it.
    ///
    /// Document order is preserved and stamping is document-relative, exactly as
    /// in the [`Chunker`] default implementation; only the text that reaches the
    /// inner chunker differs (it has been preprocessed).
    fn chunk_documents(&self, documents: &[Document]) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        for document in documents {
            let processed = self.pipeline.process(&document.text);
            let texts = self.inner.chunk_text(&processed)?;
            chunks.extend(stamp_chunks(document, texts));
        }
        Ok(chunks)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

impl fmt::Debug for TextChunker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TextChunker")
            .field("chunk_size", &self.chunk_size)
            .field("chunk_overlap", &self.chunk_overlap)
            .field("chunk_unit", &self.unit)
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

/// Builds the chunker selected by `unit`.
fn build_inner(
    chunk_size: usize,
    chunk_overlap: usize,
    unit: ChunkUnit,
) -> Result<Box<dyn Chunker>> {
    Ok(match unit {
        ChunkUnit::Char => Box::new(CharChunker::new(chunk_size, chunk_overlap)?),
        ChunkUnit::Token => Box::new(TokenChunker::new(chunk_size, chunk_overlap)?),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::super::preprocess::{SpecialCharacterNormalizer, URLEmailRemover};

    fn document(text: &str) -> Document {
        Document::with_id("doc-1", text).metadata_entry("source", json!("unit"))
    }

    // Port of tests/unit_tests/core/retrieval/indexing/processor/chunker/test_chunking.py
    #[test]
    fn chunk_unit_parses_python_strings() {
        assert_eq!("char".parse::<ChunkUnit>().unwrap(), ChunkUnit::Char);
        assert_eq!(" TOKEN ".parse::<ChunkUnit>().unwrap(), ChunkUnit::Token);
        assert_eq!(ChunkUnit::Char.as_str(), "char");
        assert_eq!(ChunkUnit::Token.as_str(), "token");
        assert_eq!(ChunkUnit::Token.to_string(), "token");
        assert_eq!(ChunkUnit::default(), ChunkUnit::Char);
        let error = "word".parse::<ChunkUnit>().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("current value: word"));
    }

    #[test]
    fn defaults_are_512_characters_without_preprocessing() {
        let chunker = TextChunker::default();
        assert_eq!(chunker.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(chunker.chunk_overlap(), DEFAULT_CHUNK_OVERLAP);
        assert_eq!(chunker.chunk_unit(), ChunkUnit::Char);
        assert!(chunker.pipeline().is_empty());
    }

    #[test]
    fn character_windows_delegate_to_the_char_chunker() {
        let chunker = TextChunker::new(4, 0, ChunkUnit::Char, PreprocessingPipeline::default())
            .unwrap();
        assert_eq!(
            chunker.chunk_text("abcdefgh").unwrap(),
            CharChunker::new(4, 0).unwrap().chunk_text("abcdefgh").unwrap()
        );
        assert_eq!(chunker.chunk_text("abcdefgh").unwrap(), ["abcd", "efgh"]);
    }

    #[test]
    fn token_windows_delegate_to_the_token_chunker() {
        let chunker =
            TextChunker::new(2, 0, ChunkUnit::Token, PreprocessingPipeline::default()).unwrap();
        assert_eq!(chunker.chunk_text("a b c").unwrap(), ["a b", "c"]);
        assert_eq!(chunker.chunk_unit(), ChunkUnit::Token);
        let with_overlap = TextChunker::new(3, 1, ChunkUnit::Token, PreprocessingPipeline::default())
            .unwrap();
        assert_eq!(with_overlap.chunk_size(), 3);
        assert_eq!(with_overlap.chunk_overlap(), 1);
    }

    #[test]
    fn preprocessing_is_visible_in_the_chunk_text() {
        let pipeline = PreprocessingPipeline::new(vec![Box::new(URLEmailRemover::new())]);
        let chunker = TextChunker::new(64, 0, ChunkUnit::Char, pipeline).unwrap();
        assert_eq!(
            chunker.chunk_text("Visit http://example.com now").unwrap(),
            ["Visit  now"]
        );
        let chunks = chunker
            .chunk_documents(&[document("Contact test@example.com today")])
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Contact  today");
    }

    #[test]
    fn preprocessing_applies_before_token_windowing() {
        let pipeline = PreprocessingPipeline::new(vec![Box::new(URLEmailRemover::new())]);
        let chunker = TextChunker::new(2, 0, ChunkUnit::Token, pipeline).unwrap();
        assert_eq!(
            chunker.chunk_text("one http://example.com two three").unwrap(),
            ["one two", "three"]
        );
    }

    #[test]
    fn chunk_documents_stamps_metadata_and_keeps_document_order() {
        let pipeline = PreprocessingPipeline::new(vec![Box::new(
            SpecialCharacterNormalizer::new().with_chars_to_remove("~~~"),
        )]);
        let chunker = TextChunker::new(4, 0, ChunkUnit::Char, pipeline).unwrap();
        let first = document("abcdefgh");
        let second = Document::with_id("doc-2", "uvwxyz").metadata_entry("source", json!("unit"));

        let chunks = chunker.chunk_documents(&[first, second]).unwrap();
        let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(texts, ["abcd", "efgh", "uvwx", "yz"]);
        let indexes: Vec<&serde_json::Value> = chunks
            .iter()
            .filter_map(|chunk| chunk.metadata.get("chunk_index"))
            .collect();
        assert_eq!(indexes, [&json!(0), &json!(1), &json!(0), &json!(1)]);
        for chunk in &chunks {
            assert_eq!(chunk.metadata.get("source"), Some(&json!("unit")));
            assert_eq!(
                chunk.metadata.get("chunk_id"),
                Some(&json!(chunk.id.clone()))
            );
            assert_eq!(
                chunk.metadata.get("total_chunks"),
                Some(&json!(u64::try_from(
                    chunks
                        .iter()
                        .filter(|other| other.doc_id == chunk.doc_id)
                        .count()
                )
                .unwrap()))
            );
        }
    }

    #[test]
    fn preprocessing_can_shrink_a_document_to_nothing() {
        let pipeline = PreprocessingPipeline::new(vec![Box::new(
            SpecialCharacterNormalizer::new().with_chars_to_remove("abc"),
        )]);
        let chunker = TextChunker::new(4, 0, ChunkUnit::Char, pipeline).unwrap();
        let chunks = chunker
            .chunk_documents(&[Document::with_id("doc-1", "abc")])
            .unwrap();
        assert!(chunks.is_empty());
        assert_eq!(chunker.chunk_text("abc").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn options_build_the_python_pipeline() {
        let chunker = TextChunker::from_options(32, 0, ChunkUnit::Char, true, true).unwrap();
        assert_eq!(chunker.pipeline().len(), 2);
        assert_eq!(
            chunker.chunk_text("Visit   http://example.com   now").unwrap(),
            ["Visit  now"]
        );
        let plain = TextChunker::from_options(32, 0, ChunkUnit::Char, false, false).unwrap();
        assert!(plain.pipeline().is_empty());
    }

    #[test]
    fn invalid_windows_are_rejected_for_both_units() {
        for unit in [ChunkUnit::Char, ChunkUnit::Token] {
            let error =
                TextChunker::new(0, 0, unit, PreprocessingPipeline::default()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            let error =
                TextChunker::new(4, 4, unit, PreprocessingPipeline::default()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
        }
    }

    #[test]
    fn unit_can_be_switched_and_pipeline_replaced() {
        let chunker = TextChunker::new(2, 0, ChunkUnit::Char, PreprocessingPipeline::default())
            .unwrap()
            .with_unit(ChunkUnit::Token)
            .unwrap();
        assert_eq!(chunker.chunk_unit(), ChunkUnit::Token);
        assert_eq!(chunker.chunk_text("a b c").unwrap(), ["a b", "c"]);

        let pipeline = PreprocessingPipeline::from_options(true, false);
        let chunker = TextChunker::default().with_pipeline(pipeline);
        assert_eq!(chunker.pipeline().len(), 1);
        assert_eq!(chunker.chunk_text("a   b").unwrap(), ["a b"]);
    }
}
