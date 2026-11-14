//! Line-granular Markdown chunking with the reference token estimate.
//!
//! This is a faithful port of Python `core/memory/lite/internal.py`'s
//! `estimate_tokens` and `chunk_markdown`: content is split on `'\n'`, whole lines
//! are accumulated until the next line would push the chunk past
//! `ChunkingConfig::target_tokens`, and the tail of the flushed chunk is carried
//! into the next chunk as overlap.
//!
//! ```text
//! for each line (1-based number i):
//!     if current_tokens + line_tokens > target_tokens and current is not empty:
//!         emit (start_line..=i-1)
//!         overlap := the longest suffix of the flushed chunk whose tokens fit overlap
//!         current := overlap
//!         start_line := i - overlap.len()
//!     current.push(line)
//! emit (start_line..=lines.len()) when current is not empty
//! ```
//!
//! # Properties
//!
//! - A chunk boundary never falls inside a line, so a chunk always maps back to a
//!   line range of `MEMORY.md` and can be cited as `path#L<start>-L<end>`.
//! - A single line longer than `target_tokens` becomes its own chunk: the flush that
//!   pushes it out sees an empty overlap budget, so it is not carried forward.
//! - `start_line` is recomputed after a flush so the overlap lines are cited even
//!   though they were already emitted by the previous chunk (the ranges of adjacent
//!   chunks therefore overlap by the overlap lines).
//! - Empty content yields no chunks; every chunk is otherwise produced by joining
//!   whole source lines with `'\n'`, so no text is ever trimmed or rewritten.
//!
//! The module is pure string processing: no clock, filesystem, or model access, and
//! nothing here can panic on user input.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The divisor of the character-based token estimate (~4 characters per token).
const CHARS_PER_TOKEN: usize = 4;

/// The default target chunk size, in estimated tokens.
const DEFAULT_TARGET_TOKENS: usize = 256;

/// The default overlap between adjacent chunks, in estimated tokens.
const DEFAULT_OVERLAP: usize = 32;

/// Bounds of the chunk shape produced by [`chunk_markdown`].
///
/// `target_tokens` is a soft cap: a single line larger than the cap is never split, so
/// one chunk may exceed it. `overlap` is the estimated-token budget of the suffix of a
/// flushed chunk that is repeated at the start of the next chunk; whole lines are
/// carried, so the realized overlap can be smaller than the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ChunkingConfig {
    /// The target chunk size in estimated tokens (default `256`).
    pub target_tokens: usize,
    /// The overlap budget in estimated tokens (default `32`).
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_tokens: DEFAULT_TARGET_TOKENS,
            overlap: DEFAULT_OVERLAP,
        }
    }
}

impl ChunkingConfig {
    /// Checks that the configuration can produce a meaningful chunk shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `target_tokens` is `0` (every chunk would be
    /// empty) or when `overlap >= target_tokens` (the overlap would swallow the chunk
    /// it is meant to bound).
    pub fn validate(&self) -> Result<()> {
        if self.target_tokens == 0 {
            return Err(Error::validation(
                "chunking config: target_tokens must be greater than 0",
            ));
        }
        if self.overlap >= self.target_tokens {
            return Err(Error::validation(format!(
                "chunking config: overlap {} must be smaller than target_tokens {}",
                self.overlap, self.target_tokens
            )));
        }
        Ok(())
    }
}

/// One contiguous run of whole lines of a document.
///
/// `start_line` and `end_line` are 1-based and inclusive, so
/// `text == lines[start_line - 1..end_line].join("\n")` for the document that produced
/// the chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemChunk {
    /// The chunk's text: its source lines joined with `'\n'`.
    pub text: String,
    /// The 1-based number of the chunk's first source line.
    pub start_line: usize,
    /// The 1-based number of the chunk's last source line.
    pub end_line: usize,
}

impl MemChunk {
    /// The number of source lines the chunk covers (always at least one).
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// Estimates the token count of `text` as `chars / 4`.
///
/// This mirrors Python's `len(text) // 4`: the count is character-based (Unicode scalar
/// values, which is what Python's `len` counts for `str`), so it is deterministic and
/// needs no tokenizer artifact. It is an estimate, not a model-exact count.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / CHARS_PER_TOKEN
}

/// Splits `content` into whole-line chunks under `config`.
///
/// See the [module docs](self) for the exact algorithm and its properties. An empty
/// `content` yields no chunks; otherwise the returned ranges are ascending, non-empty,
/// and cover every line of the document.
#[must_use]
pub fn chunk_markdown(content: &str, config: &ChunkingConfig) -> Vec<MemChunk> {
    if content.is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = content.split('\n').collect();
    let mut chunks: Vec<MemChunk> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut current_tokens = 0_usize;
    let mut start_line = 1_usize;
    for (index, line) in lines.iter().enumerate() {
        let line_number = index + 1;
        let line_tokens = estimate_tokens(line);
        if current_tokens.saturating_add(line_tokens) > config.target_tokens && !current.is_empty()
        {
            chunks.push(MemChunk {
                text: current.join("\n"),
                start_line,
                end_line: line_number - 1,
            });
            let mut overlap_lines: Vec<&str> = Vec::new();
            let mut overlap_tokens = 0_usize;
            for candidate in current.iter().rev() {
                let tokens = estimate_tokens(candidate);
                if overlap_tokens.saturating_add(tokens) > config.overlap {
                    break;
                }
                overlap_lines.insert(0, candidate);
                overlap_tokens = overlap_tokens.saturating_add(tokens);
            }
            start_line = line_number.saturating_sub(overlap_lines.len());
            current = overlap_lines;
            current_tokens = overlap_tokens;
        }
        current.push(line);
        current_tokens = current_tokens.saturating_add(line_tokens);
    }
    if !current.is_empty() {
        chunks.push(MemChunk {
            text: current.join("\n"),
            start_line,
            end_line: lines.len(),
        });
    }
    chunks
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// A line of `chars` characters, unique per `label`, so line identity is readable
    /// in assertion failures.
    fn line(label: &str, chars: usize) -> String {
        let mut line = label.to_owned();
        while line.chars().count() < chars {
            line.push('x');
        }
        line
    }

    /// A document whose lines each estimate to exactly `tokens` tokens.
    fn document(lines: usize, tokens_per_line: usize) -> String {
        (1..=lines)
            .map(|number| line(&format!("L{number:02}"), tokens_per_line * CHARS_PER_TOKEN))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn ranges(chunks: &[MemChunk]) -> Vec<(usize, usize)> {
        chunks
            .iter()
            .map(|chunk| (chunk.start_line, chunk.end_line))
            .collect()
    }

    #[test]
    fn estimate_tokens_is_four_characters_per_token() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 1);
        assert_eq!(estimate_tokens(&"x".repeat(400)), 100);
    }

    #[test]
    fn estimate_tokens_counts_characters_not_bytes() {
        // Four CJK characters are four chars (twelve UTF-8 bytes).
        assert_eq!(estimate_tokens("你好世界"), 1);
        assert_eq!(estimate_tokens("🙂🙂🙂🙂"), 1);
    }

    #[test]
    fn defaults_are_256_and_32() {
        let config = ChunkingConfig::default();
        assert_eq!(config.target_tokens, 256);
        assert_eq!(config.overlap, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_round_trips_through_json_and_tolerates_unknown_fields() {
        let config = ChunkingConfig {
            target_tokens: 64,
            overlap: 8,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, r#"{"target_tokens":64,"overlap":8}"#);
        assert_eq!(
            serde_json::from_str::<ChunkingConfig>(&json).unwrap(),
            config
        );
        let partial: ChunkingConfig =
            serde_json::from_str(r#"{"overlap":4,"extra":true}"#).unwrap();
        assert_eq!(
            partial,
            ChunkingConfig {
                target_tokens: 256,
                overlap: 4
            }
        );
    }

    #[test]
    fn validate_rejects_a_zero_target_and_an_overlapping_overlap() {
        let zero = ChunkingConfig {
            target_tokens: 0,
            overlap: 0,
        };
        assert_eq!(zero.validate().unwrap_err().kind(), ErrorKind::Validation);
        let equal = ChunkingConfig {
            target_tokens: 32,
            overlap: 32,
        };
        assert_eq!(equal.validate().unwrap_err().kind(), ErrorKind::Validation);
        let larger = ChunkingConfig {
            target_tokens: 8,
            overlap: 32,
        };
        assert!(larger.validate().is_err());
        assert!(ChunkingConfig {
            target_tokens: 8,
            overlap: 7
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn empty_content_yields_no_chunks() {
        assert_eq!(
            chunk_markdown("", &ChunkingConfig::default()),
            Vec::<MemChunk>::new()
        );
    }

    #[test]
    fn a_single_line_document_is_one_chunk() {
        let config = ChunkingConfig::default();
        let chunks = chunk_markdown("only one line", &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "only one line");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 1));
    }

    #[test]
    fn a_trailing_newline_adds_a_final_empty_line_to_the_last_chunk() {
        let config = ChunkingConfig::default();
        let chunks = chunk_markdown("first\n\n", &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "first\n\n");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 3));
        assert_eq!(chunks[0].line_count(), 3);
    }

    #[test]
    fn a_small_document_never_splits() {
        let config = ChunkingConfig {
            target_tokens: 100,
            overlap: 10,
        };
        let content = document(4, 5);
        let chunks = chunk_markdown(&content, &config);
        assert_eq!(ranges(&chunks), vec![(1, 4)]);
        assert_eq!(chunks[0].text, content);
    }

    #[test]
    fn full_chunks_flush_without_overlap_when_the_budget_is_zero() {
        let config = ChunkingConfig {
            target_tokens: 12,
            overlap: 0,
        };
        // Lines of 10 tokens: two of them no longer fit, so each flush is one line.
        let content = document(4, 10);
        let chunks = chunk_markdown(&content, &config);
        assert_eq!(ranges(&chunks), vec![(1, 1), (2, 2), (3, 3), (4, 4)]);
        assert_eq!(chunks[1].text, line("L02", 40));
    }

    #[test]
    fn overlap_lines_carry_into_the_next_chunk_and_correct_start_line() {
        let config = ChunkingConfig {
            target_tokens: 25,
            overlap: 10,
        };
        // Twelve lines of 5 tokens. Five lines fill the target exactly; a sixth would
        // overflow, so the last two lines (10 tokens) are carried as overlap.
        let content = document(12, 5);
        let lines: Vec<&str> = content.split('\n').collect();
        let chunks = chunk_markdown(&content, &config);
        assert_eq!(
            ranges(&chunks),
            vec![(1, 5), (4, 8), (7, 11), (10, 12)],
            "each flushed chunk is followed by its two-line overlap"
        );
        assert_eq!(chunks[0].text, lines[0..5].join("\n"));
        assert_eq!(chunks[1].text, lines[3..8].join("\n"));
        assert_eq!(chunks[2].text, lines[6..11].join("\n"));
        assert_eq!(chunks[3].text, lines[9..12].join("\n"));
        // The overlap really repeats the previous chunk's tail.
        assert!(chunks[1].text.starts_with(&lines[3..5].join("\n")));
        assert!(chunks[0].text.ends_with(&lines[3..5].join("\n")));
    }

    #[test]
    fn no_chunk_splits_a_line_and_every_line_is_covered() {
        let config = ChunkingConfig {
            target_tokens: 20,
            overlap: 5,
        };
        let content = [
            "# Memory",
            "",
            "## Project",
            "",
            &"build steps ".repeat(12),
            "a short line",
            &"y".repeat(200),
            "",
            "final line",
        ]
        .join("\n");
        let lines: Vec<&str> = content.split('\n').collect();
        let chunks = chunk_markdown(&content, &config);
        assert!(chunks.len() > 1, "the document must flush at least twice");
        for chunk in &chunks {
            assert!(chunk.start_line >= 1);
            assert!(chunk.end_line >= chunk.start_line);
            assert!(chunk.end_line <= lines.len());
            assert_eq!(
                chunk.text,
                lines[chunk.start_line - 1..chunk.end_line].join("\n"),
                "chunk {:?} is not a whole-line slice",
                (chunk.start_line, chunk.end_line)
            );
            assert_eq!(chunk.line_count(), chunk.end_line - chunk.start_line + 1);
        }
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[chunks.len() - 1].end_line, lines.len());
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start_line <= pair[0].end_line + 1,
                "no line may be skipped between adjacent chunks"
            );
        }
    }

    #[test]
    fn an_oversized_line_becomes_its_own_chunk() {
        let config = ChunkingConfig {
            target_tokens: 10,
            overlap: 8,
        };
        let oversized = line("huge", 400);
        let content = [oversized.clone(), line("L02", 8), line("L03", 8)].join("\n");
        let chunks = chunk_markdown(&content, &config);
        assert_eq!(ranges(&chunks), vec![(1, 1), (2, 3)]);
        assert_eq!(chunks[0].text, oversized);
        assert_eq!(chunks[1].text, [line("L02", 8), line("L03", 8)].join("\n"));
    }

    #[test]
    fn an_oversized_line_is_never_carried_as_overlap() {
        let config = ChunkingConfig {
            target_tokens: 50,
            overlap: 40,
        };
        let oversized = line("huge", 400);
        let content = [line("L01", 8), oversized.clone(), line("L03", 8)].join("\n");
        let chunks = chunk_markdown(&content, &config);
        // The predecessor fits the overlap budget, so it is repeated at the start of
        // the flush that holds the oversized line; the oversized line itself exceeds
        // any overlap budget below the target and is therefore not repeated.
        assert_eq!(ranges(&chunks), vec![(1, 1), (1, 2), (3, 3)]);
        assert_eq!(chunks[0].text, line("L01", 8));
        assert_eq!(chunks[1].text, [line("L01", 8), oversized].join("\n"));
        assert_eq!(chunks[2].text, line("L03", 8));
    }

    #[test]
    fn zero_token_lines_stay_in_the_current_chunk() {
        let config = ChunkingConfig {
            target_tokens: 1,
            overlap: 0,
        };
        // "a" and "" estimate to zero tokens, so they never trigger a flush.
        let chunks = chunk_markdown("a\n\nbcdef\nghij", &config);
        assert_eq!(ranges(&chunks), vec![(1, 3), (4, 4)]);
        assert_eq!(chunks[0].text, "a\n\nbcdef");
    }
}
