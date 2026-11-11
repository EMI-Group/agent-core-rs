//! Fixed-size character chunker.
//!
//! Port of Python `openjiuwen/core/retrieval/indexing/processor/chunker/char_chunker.py`
//! and the `CharSplitter` it delegates to: windows of at most `chunk_size`
//! characters advancing by `chunk_size - chunk_overlap`. Text is never trimmed
//! and whitespace is preserved.
use openjiuwen_common::error::Result;

use crate::common::{DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE};

use super::{char_windows, validate_params, Chunker};

/// Fixed-size chunker based on character length.
///
/// Chunk `k` covers characters `[k * step, k * step + chunk_size)`, clipped to
/// the text length, where `step = chunk_size - chunk_overlap`. The parameters
/// are validated at construction time, so chunks never exceed `chunk_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharChunker {
    /// Maximum number of characters per chunk.
    chunk_size: usize,
    /// Number of characters shared by consecutive chunks.
    chunk_overlap: usize,
}

impl CharChunker {
    /// Creates a chunker from a character window size and overlap.
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when
    /// `chunk_size` is zero or `chunk_overlap` is not smaller than it.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        validate_params(chunk_size, chunk_overlap)?;
        Ok(Self {
            chunk_size,
            chunk_overlap,
        })
    }
}

impl Default for CharChunker {
    /// The crate defaults: [`DEFAULT_CHUNK_SIZE`] with [`DEFAULT_CHUNK_OVERLAP`].
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
        }
    }
}

impl Chunker for CharChunker {
    fn chunk_text(&self, text: &str) -> Result<Vec<String>> {
        Ok(char_windows(text, self.chunk_size, self.chunk_overlap))
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    // port of tests/unit_tests/core/retrieval/indexing/chunker/test_char_chunker.py
    #[test]
    fn splits_on_exact_character_boundaries() {
        let chunker = CharChunker::new(4, 0).unwrap();
        assert_eq!(
            chunker.chunk_text("abcdefghij").unwrap(),
            ["abcd", "efgh", "ij"]
        );
    }

    #[test]
    fn applies_overlap_between_windows() {
        let chunker = CharChunker::new(4, 2).unwrap();
        assert_eq!(
            chunker.chunk_text("abcdefghij").unwrap(),
            ["abcd", "cdef", "efgh", "ghij", "ij"]
        );
        assert_eq!(chunker.chunk_size(), 4);
        assert_eq!(chunker.chunk_overlap(), 2);
    }

    #[test]
    fn chunk_count_is_len_over_step_ceiled() {
        let chunker = CharChunker::new(5, 2).unwrap();
        // step = 3, len = 11 -> ceil(11 / 3) = 4 chunks.
        let chunks = chunker.chunk_text("abcdefghijk").unwrap();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks, ["abcde", "defgh", "ghijk", "jk"]);
    }

    #[test]
    fn rejects_overlap_not_smaller_than_size() {
        assert_eq!(
            CharChunker::new(4, 4).unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            CharChunker::new(4, 9).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }

    #[test]
    fn rejects_zero_size() {
        assert_eq!(
            CharChunker::new(0, 0).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }

    #[test]
    fn allows_zero_overlap() {
        let chunker = CharChunker::new(3, 0).unwrap();
        assert_eq!(chunker.chunk_text("abcdef").unwrap(), ["abc", "def"]);
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        let chunker = CharChunker::default();
        assert_eq!(chunker.chunk_text("").unwrap(), Vec::<String>::new());
        assert_eq!(chunker.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(chunker.chunk_overlap(), DEFAULT_CHUNK_OVERLAP);
    }

    #[test]
    fn multibyte_text_does_not_split_characters() {
        let chunker = CharChunker::new(3, 1).unwrap();
        let chunks = chunker.chunk_text("你好世界abc").unwrap();
        assert_eq!(chunks, ["你好世", "世界a", "abc", "c"]);
        // Byte slicing would have panicked above; every chunk is valid UTF-8.
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 3));
    }

    #[test]
    fn whitespace_is_preserved() {
        let chunker = CharChunker::new(4, 0).unwrap();
        assert_eq!(
            chunker.chunk_text("  a b\n c ").unwrap(),
            ["  a ", "b\n c", " "]
        );
    }
}
