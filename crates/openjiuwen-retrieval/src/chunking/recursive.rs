//! Recursive character chunker.
//!
//! Splits text on a separator hierarchy (`"\n\n"`, `"\n"`, `" "`, then a hard
//! character split), merges the resulting pieces back into chunks of at most
//! `chunk_size` characters, and carries a trailing overlap window across chunk
//! boundaries.
use openjiuwen_common::error::Result;

use crate::common::{DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE};

use super::{char_len, char_windows, validate_params, Chunker};

/// Default separator hierarchy, from coarsest to finest; the empty string is
/// the final "no separator left" marker that triggers a hard character split.
const DEFAULT_SEPARATORS: [&str; 4] = ["\n\n", "\n", " ", ""];

/// A leaf produced by the recursive split.
#[derive(Debug)]
enum Piece {
    /// A piece short enough to merge with its neighbours.
    Soft(String),
    /// Hard-split windows that already carry their own overlap and must not be
    /// merged, so no character is duplicated or dropped.
    Final(Vec<String>),
}

/// Recursive character chunker with a configurable separator hierarchy.
///
/// Text is split on the coarsest separator, oversized pieces are split again
/// with the finer separators, and neighbouring pieces are merged up to
/// `chunk_size`. `chunk_overlap` trailing characters of the previous chunk are
/// reused as the prefix of the next one whenever they fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveCharChunker {
    /// Maximum number of characters per chunk.
    chunk_size: usize,
    /// Number of characters shared by consecutive chunks.
    chunk_overlap: usize,
    /// Separators tried in order.
    separators: Vec<String>,
}

impl RecursiveCharChunker {
    /// Creates a chunker with the default separator hierarchy
    /// `["\n\n", "\n", " ", ""]`.
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when
    /// `chunk_size` is zero or `chunk_overlap` is not smaller than it.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        Self::with_separators(
            chunk_size,
            chunk_overlap,
            DEFAULT_SEPARATORS
                .iter()
                .map(|separator| (*separator).to_owned())
                .collect(),
        )
    }

    /// Creates a chunker with an explicit separator hierarchy.
    ///
    /// Separators are tried in order. An empty separator (or an exhausted
    /// list) means "split on raw character boundaries". Parameters are
    /// validated exactly as in [`RecursiveCharChunker::new`].
    pub fn with_separators(
        chunk_size: usize,
        chunk_overlap: usize,
        separators: Vec<String>,
    ) -> Result<Self> {
        validate_params(chunk_size, chunk_overlap)?;
        Ok(Self {
            chunk_size,
            chunk_overlap,
            separators,
        })
    }

    /// Recursively splits `text` into pieces no longer than `chunk_size`.
    fn split_pieces(&self, text: &str, separators: &[String], out: &mut Vec<Piece>) {
        if text.is_empty() {
            return;
        }
        if char_len(text) <= self.chunk_size {
            out.push(Piece::Soft(text.to_owned()));
            return;
        }
        let Some((separator, rest)) = separators.split_first() else {
            self.hard_split(text, out);
            return;
        };
        if separator.is_empty() {
            self.hard_split(text, out);
            return;
        }
        for piece in text.split_inclusive(separator.as_str()) {
            if !piece.is_empty() {
                self.split_pieces(piece, rest, out);
            }
        }
    }

    /// Falls back to fixed-size character windows once no separator is left.
    fn hard_split(&self, text: &str, out: &mut Vec<Piece>) {
        let windows = char_windows(text, self.chunk_size, self.chunk_overlap);
        if !windows.is_empty() {
            out.push(Piece::Final(windows));
        }
    }

    /// Merges adjacent pieces into chunks and carries overlap across boundaries.
    fn merge_pieces(&self, pieces: Vec<Piece>) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        // True while `current` holds only a carried overlap tail and no new text.
        let mut tail_only = false;
        for piece in pieces {
            match piece {
                Piece::Soft(text) => {
                    let text_len = char_len(&text);
                    if char_len(&current) + text_len <= self.chunk_size {
                        current.push_str(&text);
                        tail_only = false;
                        continue;
                    }
                    if current.is_empty() {
                        // Defensive: a piece longer than `chunk_size` should not
                        // reach this point, but split it rather than emit it.
                        chunks.extend(char_windows(&text, self.chunk_size, self.chunk_overlap));
                        continue;
                    }
                    let tail = tail_chars(&current, self.chunk_overlap).to_owned();
                    if tail_only {
                        // Never emit a chunk made only of carried overlap text.
                        current.clear();
                    } else {
                        chunks.push(std::mem::take(&mut current));
                    }
                    if char_len(&tail) + text_len <= self.chunk_size {
                        current.push_str(&tail);
                    }
                    current.push_str(&text);
                    tail_only = false;
                }
                Piece::Final(windows) => {
                    if !current.is_empty() && !tail_only {
                        chunks.push(std::mem::take(&mut current));
                    }
                    current.clear();
                    if let Some(last) = windows.last() {
                        let tail = tail_chars(last, self.chunk_overlap).to_owned();
                        chunks.extend(windows);
                        current.push_str(&tail);
                        tail_only = true;
                    }
                }
            }
        }
        if !current.is_empty() && !tail_only {
            chunks.push(current);
        }
        chunks
    }
}

impl Default for RecursiveCharChunker {
    /// The crate defaults: [`DEFAULT_CHUNK_SIZE`], [`DEFAULT_CHUNK_OVERLAP`], and
    /// the default separator hierarchy.
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            separators: DEFAULT_SEPARATORS
                .iter()
                .map(|separator| (*separator).to_owned())
                .collect(),
        }
    }
}

impl Chunker for RecursiveCharChunker {
    fn chunk_text(&self, text: &str) -> Result<Vec<String>> {
        let mut pieces = Vec::new();
        self.split_pieces(text, &self.separators, &mut pieces);
        Ok(self.merge_pieces(pieces))
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

/// Returns the last `count` characters of `text` (fewer when it is shorter).
fn tail_chars(text: &str, count: usize) -> &str {
    if count == 0 {
        return "";
    }
    let total = char_len(text);
    if count >= total {
        return text;
    }
    let skip = total - count;
    let offset = text
        .char_indices()
        .nth(skip)
        .map_or(text.len(), |(offset, _)| offset);
    &text[offset..]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn prefers_the_coarsest_separator() {
        let chunker = RecursiveCharChunker::new(20, 0).unwrap();
        // "one two three" is 13 characters and would fit with "four" only if
        // the splitter ignored the paragraph break.
        let chunks = chunker.chunk_text("one two three\n\nfour five").unwrap();
        assert_eq!(chunks, ["one two three\n\n", "four five"]);
    }

    #[test]
    fn merges_adjacent_pieces_up_to_chunk_size() {
        let chunker = RecursiveCharChunker::with_separators(7, 0, vec!["\n".to_owned()]).unwrap();
        let chunks = chunker.chunk_text("aa\nbb\ncc").unwrap();
        assert_eq!(chunks, ["aa\nbb\n", "cc"]);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 7));
    }

    #[test]
    fn carries_an_overlap_window_into_the_next_chunk() {
        let chunker = RecursiveCharChunker::with_separators(10, 3, vec!["\n".to_owned()]).unwrap();
        let chunks = chunker.chunk_text("aaaaa\nbbbbb\nccccc").unwrap();
        assert_eq!(chunks, ["aaaaa\n", "aa\nbbbbb\n", "bb\nccccc"]);
    }

    #[test]
    fn oversized_pieces_fall_back_to_character_windows() {
        let chunker = RecursiveCharChunker::with_separators(10, 2, vec![String::new()]).unwrap();
        let chunks = chunker.chunk_text("abcdefghijklmnopqrstuvwx").unwrap();
        assert_eq!(chunks, ["abcdefghij", "ijklmnopqr", "qrstuvwx"]);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 10));
    }

    #[test]
    fn hard_split_handles_multibyte_text() {
        let chunker = RecursiveCharChunker::new(4, 0).unwrap();
        let chunks = chunker.chunk_text("你好世界你好").unwrap();
        assert_eq!(chunks, ["你好世界", "你好"]);
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        let chunker = RecursiveCharChunker::default();
        assert_eq!(chunker.chunk_text("").unwrap(), Vec::<String>::new());
        assert_eq!(chunker.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(chunker.chunk_overlap(), DEFAULT_CHUNK_OVERLAP);
    }

    #[test]
    fn short_text_is_a_single_untouched_chunk() {
        let chunker = RecursiveCharChunker::new(100, 10).unwrap();
        let text = "  a\tb\n\nc  ";
        assert_eq!(chunker.chunk_text(text).unwrap(), [text]);
    }

    #[test]
    fn rejects_invalid_window_parameters() {
        assert_eq!(
            RecursiveCharChunker::new(0, 0).unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            RecursiveCharChunker::new(5, 5).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }

    #[test]
    fn a_carried_tail_is_never_emitted_as_its_own_chunk() {
        let chunker = RecursiveCharChunker::new(10, 5).unwrap();
        let chunks = chunker.chunk_text("abcdefghijkl\nmnopqrstuv").unwrap();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks.last().unwrap(), "mnopqrstuv");
        assert!(chunks
            .iter()
            .all(|chunk| !chunk.is_empty() && chunk.chars().count() <= 10));
    }

    #[test]
    fn never_exceeds_chunk_size_on_mixed_text() {
        let chunker = RecursiveCharChunker::new(12, 3).unwrap();
        let text = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta";
        let chunks = chunker.chunk_text(text).unwrap();
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 12));
        // The whole text is preserved across chunks (overlap duplicates aside).
        assert!(chunks.last().unwrap().ends_with("theta"));
        assert!(chunks[0].starts_with("alpha"));
    }
}
