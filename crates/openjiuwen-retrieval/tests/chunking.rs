#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end chunking tests over the public crate API.
//!
//! Ports the behavior of the Python chunker tests in
//! `tests/unit_tests/core/retrieval/indexing/chunker/` (`test_char_chunker.py`,
//! the recursive splitter, the chunker registry, `test_hybrid_chunker.py`,
//! `test_text_preprocessor.py`, and `test_chunking.py`) plus the chunk-stamping
//! contract of the base `Chunker.chunk_documents`.
//! Every assertion is offline and deterministic: generated ids are only checked
//! for shape and uniqueness, never for a concrete value.
//!
//! `CharChunker::new` rejects rather than clamps invalid windows, so the
//! parameter tests below assert the rejection behavior (`ErrorKind::Validation`).

use std::collections::HashSet;
use std::sync::Arc;

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_retrieval::chunking::{
    default_no_split, ChunkUnit, HybridChunker, PreprocessingPipeline, SpecialCharacterNormalizer,
    TextChunker, TextPreprocessor, URLEmailRemover, WhitespaceNormalizer,
};
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

/// A `HybridChunker` over character windows, using the default no-split rule.
fn hybrid(size: usize, overlap: usize) -> HybridChunker {
    HybridChunker::new(Arc::new(CharChunker::new(size, overlap).unwrap()))
}

/// Asserts the `chunker_documents` stamping contract of one chunk.
fn assert_stamped(chunk: &TextChunk, chunk_index: u64, total_chunks: u64) {
    assert_eq!(
        chunk.metadata.get("chunk_index"),
        Some(&json!(chunk_index)),
        "chunk_index of {:?}",
        chunk.text
    );
    assert_eq!(
        chunk.metadata.get("total_chunks"),
        Some(&json!(total_chunks)),
        "total_chunks of {:?}",
        chunk.text
    );
    assert_eq!(
        chunk.metadata.get("chunk_id"),
        Some(&json!(chunk.id.clone()))
    );
    assert!(
        Uuid::parse_str(&chunk.id).is_ok(),
        "{} is not a uuid",
        chunk.id
    );
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

    for name in ["char", "fixed", "hybrid", "recursive", "token"] {
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
    let unknown = registry.create("semantic", 8, 2).err().unwrap();
    assert_eq!(unknown.kind(), ErrorKind::Unsupported);
    assert!(unknown.to_string().contains("unknown chunker: semantic"));
    assert!(!registry.contains("semantic"));

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

// ---------------------------------------------------------------------------
// Hybrid chunker, text preprocessors, and the preprocessing `TextChunker`.
//
// Ports `test_hybrid_chunker.py`, `test_text_preprocessor.py`, and
// `test_chunking.py` through the public crate surface.
// ---------------------------------------------------------------------------

#[test]
fn hybrid_chunker_keeps_row_and_column_units_whole() {
    let chunker = hybrid(10, 2);
    let row = Document::with_id("row-1", "  姓名: 张三, 部门: 研发, 工号: 1001  ")
        .metadata_entry("source_type", json!("row"))
        .metadata_entry("sheet_name", json!("Sheet1"))
        .metadata_entry("row_index", json!(2));
    let column = Document::with_id("col-1", " 列名: 姓名。取值: 张三, 李四, 王五 ")
        .metadata_entry("source_type", json!("column"))
        .metadata_entry("column_name", json!("姓名"));

    // The default predicate is exactly the table-unit rule.
    assert!(chunker.should_keep_whole(&row));
    assert!(chunker.should_keep_whole(&column));
    assert!(default_no_split(&row));
    assert!(!default_no_split(&Document::with_id("doc-1", "text")));
    assert!(!default_no_split(
        &Document::with_id("doc-2", "text").metadata_entry("source_type", json!("paragraph"))
    ));

    let chunks = chunker.chunk_documents(&[row, column]).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].doc_id, "row-1");
    // The single chunk holds the trimmed document text, even though it is far
    // longer than the ten character window.
    assert_eq!(chunks[0].text, "姓名: 张三, 部门: 研发, 工号: 1001");
    assert_eq!(chunks[1].doc_id, "col-1");
    assert_eq!(chunks[1].text, "列名: 姓名。取值: 张三, 李四, 王五");
    for chunk in &chunks {
        assert_stamped(chunk, 0, 1);
        assert!(chunk.embedding.is_none());
    }

    // Document metadata survives, including the keys the loaders stamp.
    assert_eq!(chunks[0].metadata.get("source_type"), Some(&json!("row")));
    assert_eq!(chunks[0].metadata.get("sheet_name"), Some(&json!("Sheet1")));
    assert_eq!(chunks[0].metadata.get("row_index"), Some(&json!(2)));
    assert_eq!(
        chunks[1].metadata.get("source_type"),
        Some(&json!("column"))
    );
    assert_eq!(chunks[1].metadata.get("column_name"), Some(&json!("姓名")));
    assert_ne!(chunks[0].id, chunks[1].id);
}

#[test]
fn hybrid_chunker_delegates_other_documents_with_document_relative_indices() {
    let chunker = hybrid(10, 2);
    let document = Document::with_id(
        "doc-1",
        "This is a long paragraph that the inner chunker has to split",
    )
    .metadata_entry("source_type", json!("paragraph"))
    .metadata_entry("title", json!("Test"));
    assert!(!chunker.should_keep_whole(&document));
    assert_eq!(chunker.inner().chunk_size(), 10);
    assert_eq!(chunker.chunk_size(), 10);
    assert_eq!(chunker.chunk_overlap(), 2);

    let chunks = chunker
        .chunk_documents(std::slice::from_ref(&document))
        .unwrap();
    let inner_chunks = CharChunker::new(10, 2)
        .unwrap()
        .chunk_documents(&[document])
        .unwrap();
    assert!(chunks.len() > 1);
    assert_eq!(chunks.len(), inner_chunks.len());
    // Delegation is byte-identical to the inner chunker, and the indices stay
    // document-relative.
    assert_eq!(chunks.len(), inner_chunks.len());
    let total = u64::try_from(chunks.len()).unwrap();
    for (index, (chunk, inner)) in chunks.iter().zip(inner_chunks.iter()).enumerate() {
        assert_eq!(chunk.text, inner.text);
        assert_eq!(chunk.doc_id, "doc-1");
        assert_stamped(chunk, u64::try_from(index).unwrap(), total);
        assert_eq!(chunk.metadata.get("source_type"), Some(&json!("paragraph")));
        assert_eq!(chunk.metadata.get("title"), Some(&json!("Test")));
    }
}

#[test]
fn hybrid_chunker_text_split_ignores_the_predicate() {
    // The predicate only sees whole documents, so `chunk_text` is the inner split.
    let chunker = hybrid(10, 2);
    let text = "a text that is long enough for the inner chunker to split";
    assert_eq!(
        chunker.chunk_text(text).unwrap(),
        CharChunker::new(10, 2).unwrap().chunk_text(text).unwrap()
    );
    assert!(chunker.chunk_text(text).unwrap().len() > 1);
}

#[test]
fn hybrid_chunker_delegates_blank_matched_documents() {
    let chunker = hybrid(512, 50);
    let inner = CharChunker::new(512, 50).unwrap();
    let spaces = Document::with_id("row-blank", "    ").metadata_entry("source_type", json!("row"));
    assert!(chunker.should_keep_whole(&spaces));

    // Nothing is trimmed: the inner chunker keeps its untrimmed window.
    let chunks = chunker
        .chunk_documents(std::slice::from_ref(&spaces))
        .unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "    ");
    assert_eq!(
        chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<&str>>(),
        inner
            .chunk_documents(&[spaces])
            .unwrap()
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<&str>>()
    );

    // An empty matched document yields no chunk at all.
    let empty = Document::with_id("row-empty", "").metadata_entry("source_type", json!("row"));
    let empty_chunks = chunker
        .chunk_documents(std::slice::from_ref(&empty))
        .unwrap();
    assert_eq!(empty_chunks, Vec::<TextChunk>::new());
}

#[test]
fn hybrid_chunker_custom_predicate_overrides_the_default() {
    let chunker = HybridChunker::with_predicate(
        Arc::new(CharChunker::new(10, 2).unwrap()),
        Arc::new(|document: &Document| document.metadata.get("keep_whole") == Some(&json!(true))),
    );
    let keep = Document::with_id("keep", "  a short document kept whole  ")
        .metadata_entry("keep_whole", json!(true));
    let split = Document::with_id(
        "split",
        "A row document that is long enough to be split by the inner chunker",
    )
    .metadata_entry("source_type", json!("row"));

    assert!(chunker.should_keep_whole(&keep));
    // The default row/column rule no longer applies.
    assert!(!chunker.should_keep_whole(&split));

    let chunks = chunker.chunk_documents(&[keep, split]).unwrap();
    let kept: Vec<&TextChunk> = chunks
        .iter()
        .filter(|chunk| chunk.doc_id == "keep")
        .collect();
    let split_chunks: Vec<&TextChunk> = chunks
        .iter()
        .filter(|chunk| chunk.doc_id == "split")
        .collect();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].text, "a short document kept whole");
    assert_stamped(kept[0], 0, 1);
    assert!(split_chunks.len() > 1);
    let total = u64::try_from(split_chunks.len()).unwrap();
    for (index, chunk) in split_chunks.iter().enumerate() {
        assert_stamped(chunk, u64::try_from(index).unwrap(), total);
    }
    // Input order is preserved.
    assert_eq!(
        chunks.first().map(|chunk| chunk.doc_id.as_str()),
        Some("keep")
    );
    assert_eq!(
        chunks.last().map(|chunk| chunk.doc_id.as_str()),
        Some("split")
    );
}

#[test]
fn registry_builds_the_hybrid_chunker_from_defaults() {
    let registry = ChunkerRegistry::with_defaults();
    assert!(registry.contains("hybrid"));
    assert!(registry.contains(" HYBRID "));

    let chunker = registry.create("hybrid", 12, 3).unwrap();
    assert_eq!(chunker.chunk_size(), 12);
    assert_eq!(chunker.chunk_overlap(), 3);

    // The factory wraps a character window with the default no-split rule.
    let row = Document::with_id(
        "row-1",
        "  a row that is much longer than the twelve character window  ",
    )
    .metadata_entry("source_type", json!("row"));
    let chunks = chunker.chunk_documents(std::slice::from_ref(&row)).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].text,
        "a row that is much longer than the twelve character window"
    );
    assert_stamped(&chunks[0], 0, 1);

    // A non-table document goes through the character window.
    let text = "alpha beta gamma";
    let alias = registry.create(" HYBRID ", 5, 0).unwrap();
    assert_eq!(alias.chunk_size(), 5);
    assert_eq!(alias.chunk_overlap(), 0);
    assert_eq!(
        alias.chunk_text(text).unwrap(),
        ["alpha", " beta", " gamm", "a"]
    );
}

#[test]
fn whitespace_normalizer_collapses_runs_and_trims_the_ends() {
    let normalizer = WhitespaceNormalizer::new();
    assert_eq!(normalizer.process("This is a test"), "This is a test");
    assert_eq!(
        normalizer.process("  This \t is\n\n a   test  "),
        "This is a test"
    );
    assert_eq!(normalizer.process("\t\n "), "");
    assert_eq!(normalizer.process(""), "");
}

#[test]
fn url_email_remover_drops_urls_domains_and_mail_addresses() {
    let remover = URLEmailRemover::new();
    assert!(remover.remove_urls());
    assert!(remover.remove_emails());
    assert_eq!(remover.replacement(), "");
    assert_eq!(
        remover.process("Visit https://example.com/docs today"),
        "Visit  today"
    );
    assert_eq!(
        remover.process("See www.example.org for more"),
        "See  for more"
    );
    // A bare domain without a scheme is removed as well.
    assert_eq!(remover.process("See example.com for more"), "See  for more");
    assert_eq!(
        remover.process("Contact test@example.com today"),
        "Contact  today"
    );
    assert_eq!(remover.process(""), "");

    // The replacement text is inserted literally, never as an expansion template.
    let tagged = URLEmailRemover::new().with_replacement("[gone]");
    assert_eq!(tagged.replacement(), "[gone]");
    assert_eq!(
        tagged.process("Visit http://example.com now"),
        "Visit [gone] now"
    );
    let literal = URLEmailRemover::new().with_replacement("$$");
    assert_eq!(
        literal.process("Visit http://example.com now"),
        "Visit $$ now"
    );

    let emails_only = URLEmailRemover::new().with_remove_urls(false);
    assert!(!emails_only.remove_urls());
    assert!(emails_only.remove_emails());
    assert_eq!(
        emails_only.process("mail test@example.com about example.com"),
        "mail  about example.com"
    );
}

#[test]
fn special_character_normalizer_strips_control_and_redundant_symbols() {
    let normalizer = SpecialCharacterNormalizer::new();
    assert_eq!(normalizer.chars_to_remove(), "");
    assert_eq!(normalizer.chars_to_replace().len(), 0);

    // Control characters go first.
    assert_eq!(normalizer.process("a\u{0}b\u{1F}c\u{7F}d"), "abcd");
    // Runs of two or more redundant symbols are removed; a single one is kept.
    assert_eq!(normalizer.process("hello!!!world"), "helloworld");
    assert_eq!(normalizer.process("a~~~b"), "ab");
    assert_eq!(normalizer.process("hello!world"), "hello!world");
    assert_eq!(normalizer.process("a~b"), "a~b");

    // CJK punctuation at the start of a run is protected, its removable suffix
    // still goes away.
    assert_eq!(normalizer.process("你好！！世界"), "你好！！世界");
    assert_eq!(normalizer.process("你好！！！!!!"), "你好！！！");
    assert_eq!(normalizer.process("看，。！"), "看，。！");

    // Markdown markers are never redundant symbols.
    assert_eq!(normalizer.process("## Heading"), "## Heading");
    assert_eq!(
        normalizer.process("**bold** and _italic_"),
        "**bold** and _italic_"
    );
    assert_eq!(normalizer.process("a | b | c"), "a | b | c");
    assert_eq!(normalizer.process("---"), "---");

    // Configured replacements run in order, then configured removals.
    let configured = SpecialCharacterNormalizer::new()
        .with_chars_to_remove("~")
        .with_replacement("Mr.", "Mister");
    assert_eq!(configured.chars_to_remove(), "~");
    assert_eq!(
        configured.chars_to_replace(),
        [("Mr.".to_owned(), "Mister".to_owned())]
    );
    assert_eq!(configured.process("Mr. Smith~"), "Mister Smith");
    let mapped = SpecialCharacterNormalizer::new().with_chars_to_replace([
        ("a".to_owned(), "A".to_owned()),
        ("A".to_owned(), "b".to_owned()),
    ]);
    assert_eq!(mapped.process("aaa"), "bbb");
}

#[test]
fn pipeline_applies_preprocessors_in_order_and_from_options_counts_them() {
    let ordered = PreprocessingPipeline::new(vec![
        Box::new(SpecialCharacterNormalizer::new().with_replacement("a", "b")),
        Box::new(SpecialCharacterNormalizer::new().with_replacement("b", "c")),
    ]);
    assert_eq!(ordered.len(), 2);
    assert!(!ordered.is_empty());
    // The first stage's output feeds the second stage.
    assert_eq!(ordered.process("aaa"), "ccc");

    let reversed = PreprocessingPipeline::new(vec![
        Box::new(SpecialCharacterNormalizer::new().with_replacement("b", "c")),
        Box::new(SpecialCharacterNormalizer::new().with_replacement("a", "b")),
    ]);
    assert_eq!(reversed.process("aaa"), "bbb");

    // Whitespace collapse first, then URL removal, so the surrounding spaces stay.
    let mut pipeline = PreprocessingPipeline::new(vec![Box::new(WhitespaceNormalizer::new())]);
    pipeline.add_preprocessor(Box::new(URLEmailRemover::new()));
    assert_eq!(pipeline.len(), 2);
    assert_eq!(
        pipeline.process("Visit   http://example.com   now"),
        "Visit  now"
    );

    assert!(PreprocessingPipeline::from_options(false, false).is_empty());
    assert_eq!(PreprocessingPipeline::from_options(false, false).len(), 0);
    assert_eq!(PreprocessingPipeline::from_options(true, false).len(), 1);
    assert_eq!(PreprocessingPipeline::from_options(false, true).len(), 1);
    let both = PreprocessingPipeline::from_options(true, true);
    assert_eq!(both.len(), 2);
    assert_eq!(
        both.process("Visit   http://example.com   now"),
        "Visit  now"
    );

    // An empty pipeline is the identity.
    assert_eq!(
        PreprocessingPipeline::default().process("  untouched  "),
        "  untouched  "
    );
}

#[test]
fn text_chunker_preprocesses_before_char_windowing() {
    let chunker = TextChunker::new(
        4,
        0,
        ChunkUnit::Char,
        PreprocessingPipeline::from_options(true, false),
    )
    .unwrap();
    assert_eq!(chunker.chunk_unit(), ChunkUnit::Char);
    assert_eq!(chunker.chunk_size(), 4);
    assert_eq!(chunker.chunk_overlap(), 0);
    assert_eq!(chunker.pipeline().len(), 1);

    let chunks = chunker.chunk_text("abc   def\t\tghi").unwrap();
    assert_eq!(chunks, ["abc ", "def ", "ghi"]);
    // The windows are taken over the preprocessed text.
    assert_eq!(chunks.concat(), "abc def ghi");
}

#[test]
fn text_chunker_preprocesses_before_token_windowing() {
    let pipeline = PreprocessingPipeline::new(vec![Box::new(URLEmailRemover::new())]);
    let chunker = TextChunker::new(2, 0, ChunkUnit::Token, pipeline).unwrap();
    assert_eq!(chunker.chunk_unit(), ChunkUnit::Token);
    assert_eq!(
        chunker
            .chunk_text("one http://example.com two three")
            .unwrap(),
        ["one two", "three"]
    );

    // The same pipeline built through the option flags.
    let mail = TextChunker::from_options(3, 1, ChunkUnit::Token, false, true).unwrap();
    assert_eq!(
        mail.chunk_text("one test@example.com two three four")
            .unwrap(),
        ["one two three", "three four"]
    );
}

#[test]
fn text_chunker_stamps_chunks_per_document() {
    let chunker = TextChunker::new(
        4,
        0,
        ChunkUnit::Char,
        PreprocessingPipeline::from_options(true, false),
    )
    .unwrap();
    // The pipeline trims the padded document before it is split.
    let first = Document::with_id("doc-a", "  abcdefgh  ").metadata_entry("source", json!("int"));
    let second = Document::with_id("doc-b", "uvwxyz");

    let chunks = chunker.chunk_documents(&[first, second]).unwrap();
    let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    assert_eq!(texts, ["abcd", "efgh", "uvwx", "yz"]);
    let doc_ids: Vec<&str> = chunks.iter().map(|chunk| chunk.doc_id.as_str()).collect();
    assert_eq!(doc_ids, ["doc-a", "doc-a", "doc-b", "doc-b"]);
    let indexes: Vec<Option<u64>> = chunks
        .iter()
        .map(|chunk| stamped_u64(chunk, "chunk_index"))
        .collect();
    assert_eq!(indexes, [Some(0), Some(1), Some(0), Some(1)]);

    for (chunk, (index, total)) in chunks.iter().zip([(0, 2), (1, 2), (0, 2), (1, 2)]) {
        assert_stamped(chunk, index, total);
        assert!(chunk.embedding.is_none());
    }
    assert_eq!(chunks[0].metadata.get("source"), Some(&json!("int")));
    let ids: HashSet<&String> = chunks.iter().map(|chunk| &chunk.id).collect();
    assert_eq!(ids.len(), chunks.len());
}

#[test]
fn text_chunker_default_is_a_plain_char_window_and_pipelines_can_empty_a_document() {
    let plain = TextChunker::default();
    assert_eq!(plain.chunk_size(), DEFAULT_CHUNK_SIZE);
    assert_eq!(plain.chunk_overlap(), DEFAULT_CHUNK_OVERLAP);
    assert_eq!(plain.chunk_unit(), ChunkUnit::Char);
    assert!(plain.pipeline().is_empty());
    assert_eq!(plain.chunk_text("abcdefgh").unwrap(), ["abcdefgh"]);

    // A pipeline that removes the whole text leaves no chunk behind.
    let pipeline = PreprocessingPipeline::new(vec![Box::new(
        SpecialCharacterNormalizer::new().with_chars_to_remove("abc"),
    )]);
    let chunker = TextChunker::new(4, 0, ChunkUnit::Char, pipeline).unwrap();
    let removed = chunker
        .chunk_documents(&[Document::with_id("doc-1", "abc")])
        .unwrap();
    assert_eq!(removed, Vec::<TextChunk>::new());
    assert_eq!(chunker.chunk_text("abc").unwrap(), Vec::<String>::new());

    // So does whitespace-only input once the whitespace is normalized away.
    let trimming = TextChunker::new(
        4,
        0,
        ChunkUnit::Char,
        PreprocessingPipeline::from_options(true, false),
    )
    .unwrap();
    let blank = trimming
        .chunk_documents(&[Document::with_id("doc-2", " \n\t ")])
        .unwrap();
    assert_eq!(blank, Vec::<TextChunk>::new());
}
