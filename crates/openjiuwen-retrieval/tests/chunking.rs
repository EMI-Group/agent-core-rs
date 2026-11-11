#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end chunking tests over the public crate API.
//!
//! Ports the behavior of the Python chunker tests in
//! `tests/unit_tests/core/retrieval/indexing/chunker/` (`test_char_chunker.py`,
//! the recursive splitter, and the chunker registry) plus the chunk-stamping
//! contract of the base `Chunker.chunk_documents`.
//! Every assertion is offline and deterministic: generated ids are only checked
//! for shape and uniqueness, never for a concrete value.
//!
//! `CharChunker::new` rejects rather than clamps invalid windows, so the
//! parameter tests below assert the rejection behavior (`ErrorKind::Validation`).

use std::collections::HashSet;
use std::sync::Arc;

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_retrieval::{
    CharChunker, Chunker, ChunkerFactory, ChunkerRegistry, Document, RecursiveCharChunker,
    TextChunk, TokenChunker, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
};
use serde_json::{json, Value};
use uuid::Uuid;

/// Returns the first `count` characters of `text`.
fn head(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

/// Returns the last `count` characters of `text` (fewer when it is shorter).
fn tail(text: &str, count: usize) -> String {
    let total = text.chars().count();
    text.chars().skip(total.saturating_sub(count)).collect()
}

/// A `CharChunker` factory suitable for `ChunkerRegistry::register`.
fn create_char(size: usize, overlap: usize) -> Result<Arc<dyn Chunker>> {
    Ok(Arc::new(CharChunker::new(size, overlap)?))
}

/// Asserts every chunk holds at most `size` characters.
fn assert_all_within_char_budget(chunks: &[String], size: usize) {
    for chunk in chunks {
        assert!(
            chunk.chars().count() <= size,
            "chunk {chunk:?} exceeds the {size} character budget"
        );
    }
}

/// Asserts chunk metadata carries the documented stamping keys.
fn stamped_u64(chunk: &TextChunk, key: &str) -> Option<u64> {
    chunk.metadata.get(key).and_then(Value::as_u64)
}

#[test]
fn char_chunker_matches_exact_windows_and_repeats_the_overlap() {
    // Windows start at `k * (size - overlap)`, clipped to the text length.
    let chunker = CharChunker::new(10, 3).unwrap();
    let text = "0123456789abcdefghijklmno";
    assert_eq!(text.chars().count(), 25);
    let chunks = chunker.chunk_text(text).unwrap();

    assert_eq!(chunks, ["0123456789", "789abcdefg", "efghijklmn", "lmno"]);
    // ceil(len / step) == ceil(25 / 7) == 4 chunks.
    assert_eq!(chunks.len(), 25_usize.div_ceil(10 - 3));
    assert_eq!(chunks.first().unwrap(), "0123456789");
    assert_eq!(chunks.last().unwrap(), "lmno");
    assert_all_within_char_budget(&chunks, chunker.chunk_size());

    // Consecutive chunks share exactly the configured overlap region.
    for pair in chunks.windows(2) {
        assert!(
            pair[0].ends_with(&head(&pair[1], chunker.chunk_overlap())),
            "chunk {:?} must end with the first {} characters of {:?}",
            pair[0],
            chunker.chunk_overlap(),
            pair[1]
        );
    }
    assert_eq!(chunker.chunk_size(), 10);
    assert_eq!(chunker.chunk_overlap(), 3);
}

#[test]
fn char_chunker_empty_input_yields_no_chunks_and_defaults_are_wired() {
    let chunker = CharChunker::default();
    assert_eq!(chunker.chunk_size(), DEFAULT_CHUNK_SIZE);
    assert_eq!(chunker.chunk_overlap(), DEFAULT_CHUNK_OVERLAP);
    assert_eq!(chunker.chunk_text("").unwrap(), Vec::<String>::new());

    let small = CharChunker::new(4, 0).unwrap();
    assert_eq!(small.chunk_text("").unwrap(), Vec::<String>::new());
    // Shorter than the window: a single untouched chunk.
    assert_eq!(small.chunk_text("abc").unwrap(), ["abc"]);
}

#[test]
fn char_chunker_zero_overlap_splits_without_repeating_text() {
    let chunker = CharChunker::new(4, 0).unwrap();
    let text = "abcdefghij";
    let chunks = chunker.chunk_text(text).unwrap();

    assert_eq!(chunks, ["abcd", "efgh", "ij"]);
    assert_eq!(chunks.concat(), text);
    assert_all_within_char_budget(&chunks, 4);
    assert!(!chunks[0].ends_with(&head(&chunks[1], 1)));
    assert!(!chunks[1].ends_with(&head(&chunks[2], 1)));
    assert_eq!(chunker.chunk_overlap(), 0);
}

#[test]
fn chunkers_reject_invalid_window_parameters() {
    // `validate_params` rejects instead of clamping, matching the documented
    // contract; an overlap of zero stays legal.
    let overlap_equal = CharChunker::new(10, 10).unwrap_err();
    assert_eq!(overlap_equal.kind(), ErrorKind::Validation);
    assert!(overlap_equal
        .to_string()
        .contains("chunk_overlap must be less than chunk_size"));

    let overlap_beyond = CharChunker::new(4, 9).unwrap_err();
    assert_eq!(overlap_beyond.kind(), ErrorKind::Validation);

    let zero_size = CharChunker::new(0, 0).unwrap_err();
    assert_eq!(zero_size.kind(), ErrorKind::Validation);
    assert!(zero_size
        .to_string()
        .contains("chunk_size must be greater than 0"));

    assert_eq!(
        RecursiveCharChunker::new(0, 0).unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        RecursiveCharChunker::new(5, 5).unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        TokenChunker::new(0, 0).unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        TokenChunker::new(3, 3).unwrap_err().kind(),
        ErrorKind::Validation
    );

    assert!(CharChunker::new(1, 0).is_ok());
    assert!(RecursiveCharChunker::new(1, 0).is_ok());
    assert!(TokenChunker::new(1, 0).is_ok());
}

#[test]
fn char_chunker_splits_multibyte_text_on_char_boundaries() {
    // Slicing byte offsets would panic or corrupt these strings; every chunk
    // must still be valid UTF-8 of at most `chunk_size` characters.
    let chunker = CharChunker::new(2, 1).unwrap();
    let text = "aé中😀b";
    assert_eq!(text.chars().count(), 5);
    let chunks = chunker.chunk_text(text).unwrap();

    assert_eq!(chunks, ["aé", "é中", "中😀", "😀b", "b"]);
    assert_all_within_char_budget(&chunks, 2);
    for pair in chunks.windows(2) {
        assert!(pair[0].ends_with(&head(&pair[1], 1)));
    }

    // Without overlap the chunks concatenate back to the source text.
    let reassembled = CharChunker::new(2, 0)
        .unwrap()
        .chunk_text("😀é中b")
        .unwrap();
    assert_eq!(reassembled, ["😀é", "中b"]);
    assert_eq!(reassembled.concat(), "😀é中b");
}

#[test]
fn recursive_chunker_prefers_paragraph_separators() {
    let text = "aaa\n\nbbb\n\nccc";
    assert_eq!(text.chars().count(), 13);

    // The recursive splitter fills chunks up to the budget but breaks on the
    // paragraph separator instead of cutting through a paragraph.
    let recursive = RecursiveCharChunker::new(10, 0).unwrap();
    let chunks = recursive.chunk_text(text).unwrap();
    assert_eq!(chunks, ["aaa\n\nbbb\n\n", "ccc"]);
    assert_all_within_char_budget(&chunks, recursive.chunk_size());
    assert!(chunks[0].ends_with("\n\n"));
    assert_eq!(chunks.concat(), text);

    // A fixed character window ignores the paragraph boundary, which is the
    // behavior the recursive splitter improves on: with a tighter budget the
    // window cuts through "bbb" while the recursive splitter breaks after the
    // paragraph separator.
    let char_chunks = CharChunker::new(8, 0).unwrap().chunk_text(text).unwrap();
    assert_eq!(char_chunks, ["aaa\n\nbbb", "\n\nccc"]);
    let tighter = RecursiveCharChunker::new(8, 0).unwrap();
    let tighter_chunks = tighter.chunk_text(text).unwrap();
    assert_eq!(tighter_chunks, ["aaa\n\n", "bbb\n\nccc"]);
    assert_eq!(tighter_chunks.concat(), text);
    assert!(tighter_chunks[0].ends_with("\n\n"));
    assert!(!char_chunks[0].ends_with("\n\n"));
    assert_ne!(char_chunks, tighter_chunks);
}

#[test]
fn recursive_chunker_carries_overlap_across_chunks() {
    let chunker = RecursiveCharChunker::new(12, 4).unwrap();
    let chunks = chunker.chunk_text("aaa\n\nbbb\n\nccc").unwrap();

    assert_eq!(chunks, ["aaa\n\nbbb\n\n", "bb\n\nccc"]);
    assert_all_within_char_budget(&chunks, 12);
    // The next chunk opens with the trailing overlap window of the previous one.
    assert!(chunks[1].starts_with(&tail(&chunks[0], 4)));
    assert_eq!(tail(&chunks[0], 4), "bb\n\n");
    assert!(chunks[1].ends_with("ccc"));
    assert_eq!(chunker.chunk_size(), 12);
    assert_eq!(chunker.chunk_overlap(), 4);
}

#[test]
fn token_chunker_slides_a_window_with_overlap() {
    let chunker = TokenChunker::new(3, 1).unwrap();
    // Whitespace runs (spaces, tabs, newlines) collapse into single separators.
    let chunks = chunker.chunk_text("a  b\tc\nd e").unwrap();

    assert_eq!(chunks, ["a b c", "c d e"]);
    assert_eq!(chunker.chunk_size(), 3);
    assert_eq!(chunker.chunk_overlap(), 1);
    // The overlap token is repeated at the window boundary.
    assert!(chunks[0].ends_with('c'));
    assert!(chunks[1].starts_with('c'));
    assert!(chunks
        .iter()
        .all(|chunk| chunk.split_whitespace().count() <= 3));
}

#[test]
fn token_chunker_rejoins_windows_with_single_spaces() {
    let chunker = TokenChunker::new(2, 0).unwrap();
    let text = "one two three four five";
    let chunks = chunker.chunk_text(text).unwrap();

    assert_eq!(chunks, ["one two", "three four", "five"]);
    // No overlap: the windows rejoin to the whitespace-normalized source.
    assert_eq!(chunks.join(" "), text);
    assert!(chunks
        .iter()
        .all(|chunk| chunk.split_whitespace().count() <= 2));
}

#[test]
fn token_chunker_empty_and_whitespace_only_input_yield_no_chunks() {
    let chunker = TokenChunker::new(2, 0).unwrap();
    assert_eq!(chunker.chunk_text("").unwrap(), Vec::<String>::new());
    assert_eq!(chunker.chunk_text(" \n\t ").unwrap(), Vec::<String>::new());
}

#[test]
fn registry_defaults_create_every_builtin_chunker() {
    let registry = ChunkerRegistry::with_defaults();
    let sample = "alpha beta gamma";

    for name in ["char", "fixed", "recursive", "token"] {
        assert!(registry.contains(name), "{name} must be registered");
        let chunker = registry.create(name, 5, 0).unwrap();
        assert_eq!(chunker.chunk_size(), 5, "{name} size");
        assert_eq!(chunker.chunk_overlap(), 0, "{name} overlap");
        let chunks = chunker.chunk_text(sample).unwrap();
        assert!(!chunks.is_empty(), "{name} must produce chunks");
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
        if name == "token" {
            assert!(chunks
                .iter()
                .all(|chunk| chunk.split_whitespace().count() <= 5));
        } else {
            assert_all_within_char_budget(&chunks, 5);
        }
    }

    // `fixed` is the alias of the character chunker.
    let char_chunks = registry.create("char", 5, 0).unwrap();
    let fixed_chunks = registry.create("  FIXED ", 5, 0).unwrap();
    let expected = ["alpha", " beta", " gamm", "a"];
    assert_eq!(char_chunks.chunk_text(sample).unwrap(), expected);
    assert_eq!(fixed_chunks.chunk_text(sample).unwrap(), expected);
    assert_eq!(fixed_chunks.chunk_size(), 5);
}

#[test]
fn registry_rejects_unknown_names_and_duplicate_registrations() {
    let registry = ChunkerRegistry::with_defaults();
    // `create` returns `Error::Unsupported` for an unregistered name.
    let unknown = registry.create("hybrid", 8, 2).err().unwrap();
    assert_eq!(unknown.kind(), ErrorKind::Unsupported);
    assert!(unknown.to_string().contains("unknown chunker: hybrid"));
    assert!(!registry.contains("hybrid"));

    // Registering an existing name is a validation error.
    let mut registry = ChunkerRegistry::with_defaults();
    let factory: ChunkerFactory = Arc::new(create_char);
    let duplicate = registry.register("char", factory).unwrap_err();
    assert_eq!(duplicate.kind(), ErrorKind::Validation);
    assert!(duplicate.to_string().contains("already registered"));

    // Custom names can be added and resolve case-insensitively.
    let mut registry = ChunkerRegistry::new();
    registry.register("Char2", Arc::new(create_char)).unwrap();
    let chunker = registry.create(" char2 ", 4, 0).unwrap();
    assert_eq!(chunker.chunk_text("abcdefgh").unwrap(), ["abcd", "efgh"]);

    // Blank names are rejected.
    let blank = registry
        .register(" \t ", Arc::new(create_char))
        .unwrap_err();
    assert_eq!(blank.kind(), ErrorKind::Validation);
}

#[test]
fn chunk_documents_stamps_metadata_across_documents() {
    let doc_a = Document::with_id("doc-a", "12345678").metadata_entry("source", json!("unit"));
    let doc_b = Document::with_id("doc-b", "uvwxyz").metadata_entry("source", json!("unit"));
    let empty = Document::with_id("doc-empty", "");

    let chunks = CharChunker::new(4, 0)
        .unwrap()
        .chunk_documents(&[doc_a, doc_b, empty])
        .unwrap();

    // Document-major order; the empty document contributes no chunks.
    assert_eq!(chunks.len(), 4);
    let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    assert_eq!(texts, ["1234", "5678", "uvwx", "yz"]);
    let doc_ids: Vec<&str> = chunks.iter().map(|chunk| chunk.doc_id.as_str()).collect();
    assert_eq!(doc_ids, ["doc-a", "doc-a", "doc-b", "doc-b"]);
    assert!(chunks.iter().all(|chunk| chunk.doc_id != "doc-empty"));

    // `chunk_index` restarts for every document, `total_chunks` is per document.
    let indexes: Vec<Option<u64>> = chunks
        .iter()
        .map(|chunk| stamped_u64(chunk, "chunk_index"))
        .collect();
    assert_eq!(indexes, [Some(0), Some(1), Some(0), Some(1)]);
    for chunk in &chunks {
        let per_document = chunks
            .iter()
            .filter(|other| other.doc_id == chunk.doc_id)
            .count();
        assert_eq!(
            stamped_u64(chunk, "total_chunks"),
            Some(u64::try_from(per_document).unwrap())
        );
        // `chunk_id` mirrors the chunk id and document metadata is copied.
        assert_eq!(
            chunk.metadata.get("chunk_id").and_then(Value::as_str),
            Some(chunk.id.as_str())
        );
        assert_eq!(chunk.metadata.get("source"), Some(&json!("unit")));
        assert!(chunk.embedding.is_none());
        assert!(Uuid::parse_str(&chunk.id).is_ok());
    }

    // Chunk identifiers are fresh uuids, unique across all documents.
    let ids: HashSet<&String> = chunks.iter().map(|chunk| &chunk.id).collect();
    assert_eq!(ids.len(), chunks.len());
}
