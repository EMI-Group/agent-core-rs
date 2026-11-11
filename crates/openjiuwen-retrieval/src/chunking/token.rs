//! Token-based chunker approximation.
//!
//! Python's token chunker uses `tiktoken` (`cl100k_base`) to encode text into
//! token ids and decode each window back into a string. Rust has no tokenizer
//! dependency here, so the unit stream is produced by splitting on whitespace
//! and windows are joined with a single space; the token budget is measured by
//! a pluggable [`TokenCounter`]. This deviation is documented in the crate
//! `CONTEXT.md`; chunk boundaries are deterministic for a given counter.
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;

use super::{validate_params, Chunker};

/// Counts tokens in a text.
///
/// Implementations must be cheap and deterministic; they are consulted once per
/// candidate token unit while building a window.
pub trait TokenCounter: Send + Sync {
    /// Returns the number of tokens in `text`.
    fn count(&self, text: &str) -> usize;
}

/// Token counter that treats every whitespace-separated word as one token.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

/// Chunker that slides a token window over whitespace-separated units.
///
/// Window `k` starts at unit `k * step`, where `step = window_len - overlap`
/// (`window_len == chunk_size_tokens` for the default counter), and is emitted
/// as its units joined by a single space.
pub struct TokenChunker {
    /// Maximum number of tokens per chunk.
    chunk_size_tokens: usize,
    /// Number of tokens shared by consecutive chunks.
    overlap_tokens: usize,
    /// Counter used to measure a candidate window.
    counter: Arc<dyn TokenCounter>,
}

impl TokenChunker {
    /// Creates a chunker using [`WhitespaceTokenCounter`].
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when
    /// `chunk_size_tokens` is zero or `overlap_tokens` is not smaller than it.
    pub fn new(chunk_size_tokens: usize, overlap_tokens: usize) -> Result<Self> {
        validate_params(chunk_size_tokens, overlap_tokens)?;
        Ok(Self {
            chunk_size_tokens,
            overlap_tokens,
            counter: Arc::new(WhitespaceTokenCounter),
        })
    }

    /// Replaces the default whitespace counter.
    #[must_use]
    pub fn with_token_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }

    /// Returns the window `[start, end)` and its joined text.
    ///
    /// At least one unit is always consumed so callers make progress; a unit
    /// whose own token count exceeds the budget is emitted on its own.
    fn window(&self, units: &[&str], start: usize) -> (usize, String) {
        let mut window = String::new();
        let mut end = start;
        while end < units.len() {
            let previous_len = window.len();
            if !window.is_empty() {
                window.push(' ');
            }
            window.push_str(units[end]);
            end += 1;
            if self.counter.count(&window) > self.chunk_size_tokens {
                if end - start == 1 {
                    break;
                }
                // Drop the unit that overflowed the budget; `previous_len` is a
                // character boundary because only whole units were appended.
                window.truncate(previous_len);
                end -= 1;
                break;
            }
        }
        (end, window)
    }
}

impl fmt::Debug for TokenChunker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenChunker")
            .field("chunk_size_tokens", &self.chunk_size_tokens)
            .field("overlap_tokens", &self.overlap_tokens)
            .finish_non_exhaustive()
    }
}

impl Chunker for TokenChunker {
    fn chunk_text(&self, text: &str) -> Result<Vec<String>> {
        let units: Vec<&str> = text.split_whitespace().collect();
        if units.is_empty() {
            return Ok(Vec::new());
        }
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < units.len() {
            let (end, window) = self.window(&units, start);
            chunks.push(window);
            if end >= units.len() {
                break;
            }
            let window_len = end - start;
            let step = window_len.saturating_sub(self.overlap_tokens).max(1);
            start += step;
        }
        Ok(chunks)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size_tokens
    }

    fn chunk_overlap(&self) -> usize {
        self.overlap_tokens
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn slides_a_token_window_with_overlap() {
        let chunker = TokenChunker::new(3, 1).unwrap();
        assert_eq!(
            chunker.chunk_text("a  b\tc\nd e").unwrap(),
            ["a b c", "c d e"]
        );
        assert_eq!(chunker.chunk_size(), 3);
        assert_eq!(chunker.chunk_overlap(), 1);
    }

    #[test]
    fn windows_are_rejoined_with_single_spaces() {
        let chunker = TokenChunker::new(10, 0).unwrap();
        assert_eq!(chunker.chunk_text("a\n\tb   c").unwrap(), ["a b c"]);
    }

    #[test]
    fn empty_and_whitespace_only_input_yield_no_chunks() {
        let chunker = TokenChunker::new(2, 0).unwrap();
        assert_eq!(chunker.chunk_text("").unwrap(), Vec::<String>::new());
        assert_eq!(chunker.chunk_text(" \n\t ").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn rejects_invalid_windows() {
        assert_eq!(
            TokenChunker::new(0, 0).unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            TokenChunker::new(3, 3).unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert!(TokenChunker::new(3, 0).is_ok());
    }

    #[test]
    fn custom_counter_drives_the_window_size() {
        struct CharCounter;

        impl TokenCounter for CharCounter {
            fn count(&self, text: &str) -> usize {
                text.chars().count()
            }
        }

        let chunker = TokenChunker::new(5, 0)
            .unwrap()
            .with_token_counter(Arc::new(CharCounter));
        assert_eq!(chunker.chunk_text("ab cd ef").unwrap(), ["ab cd", "ef"]);
    }

    #[test]
    fn oversized_single_unit_is_still_emitted() {
        struct CharCounter;

        impl TokenCounter for CharCounter {
            fn count(&self, text: &str) -> usize {
                text.chars().count()
            }
        }

        let chunker = TokenChunker::new(2, 1)
            .unwrap()
            .with_token_counter(Arc::new(CharCounter));
        assert_eq!(chunker.chunk_text("abcdef gh").unwrap(), ["abcdef", "gh"]);
    }
}
