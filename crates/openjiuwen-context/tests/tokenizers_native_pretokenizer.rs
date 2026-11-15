#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Regex `Split`, `Replace` and `CharDelimiterSplit` pre-tokenizers driving a real `tokenizer.json` pipeline.

use openjiuwen_context::tokenizers::{load_native_tokenizer, NativeTokenizer, TokenizerSource};
use openjiuwen_foundation::prompt::TokenCounter;
use serde_json::json;

/// Parses a `tokenizer.json` fixture.
fn parse(value: &serde_json::Value) -> NativeTokenizer {
    NativeTokenizer::from_json_str(&value.to_string()).expect("fixture loads")
}

#[test]
fn a_regex_split_pretokenizer_drives_the_bpe_pipeline() {
    let tokenizer = parse(&json!({
        "name": "regex-split-bpe",
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, " ": 5},
            "merges": ["a b", "ab c"]
        },
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
        "decoder": {"type": "Fuse"}
    }));
    assert!(
        tokenizer.warnings().is_empty(),
        "a compiled regex must not warn: {:?}",
        tokenizer.warnings()
    );
    // `Isolated` keeps the whitespace as its own piece, so it is encoded as id 5.
    assert_eq!(tokenizer.encode("abc abc"), vec![4, 5, 4]);
    assert_eq!(tokenizer.decode(&[4, 5, 4]), "abc abc");
}

#[test]
fn a_regex_split_follows_its_behavior() {
    let tokenizer = parse(&json!({
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2, "1": 3, "2": 4, "12": 5},
            "merges": ["a b", "1 2"]
        },
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\d+"}, "behavior": "Removed"}
    }));
    assert!(
        tokenizer.warnings().is_empty(),
        "unexpected warnings: {:?}",
        tokenizer.warnings()
    );
    // `Removed` drops the digits, so the two `ab` runs are encoded on their own.
    assert_eq!(tokenizer.encode("ab12ab"), vec![2, 2]);
}

#[test]
fn replace_and_split_compose_inside_a_sequence() {
    let tokenizer = parse(&json!({
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "ab": 4, "cd": 5},
            "merges": ["a b", "c d"]
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": "_"},
                {"type": "Split", "pattern": {"Regex": "_"}, "behavior": "Removed"}
            ]
        }
    }));
    assert!(
        tokenizer.warnings().is_empty(),
        "unexpected warnings: {:?}",
        tokenizer.warnings()
    );
    assert_eq!(tokenizer.encode("ab cd"), vec![4, 5]);
}

#[test]
fn a_char_delimiter_split_removes_the_delimiter() {
    let tokenizer = parse(&json!({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
        "pre_tokenizer": {"type": "CharDelimiterSplit", "delimiter": "|"}
    }));
    assert!(
        tokenizer.warnings().is_empty(),
        "unexpected warnings: {:?}",
        tokenizer.warnings()
    );
    assert_eq!(tokenizer.encode("ab|ab"), vec![2, 2]);
}

#[test]
fn an_uncompilable_regex_warns_and_still_counts() {
    let tokenizer = parse(&json!({
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "(unclosed"}, "behavior": "Isolated"}
    }));
    assert_eq!(tokenizer.warnings().len(), 1);
    assert!(
        tokenizer.warnings()[0].contains("(unclosed"),
        "the warning must name the pattern: {}",
        tokenizer.warnings()[0]
    );
    // The degraded whitespace split still yields one id per `a`.
    assert_eq!(tokenizer.encode("a a"), vec![0, 0]);
}

#[test]
fn a_regex_split_artifact_loads_from_a_file_without_warnings() {
    let fixture = json!({
        "name": "regex-split-file",
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, " ": 5},
            "merges": ["a b", "ab c"]
        },
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"}
    });
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, fixture.to_string()).expect("write fixture");

    let selection = load_native_tokenizer(&path).expect("loads");
    assert_eq!(selection.source(), TokenizerSource::NativeTokenizer);
    assert!(!selection.estimated());
    assert_eq!(selection.tokenizer(), Some("regex-split-file"));
    assert_eq!(selection.count_text("abc abc"), 3);
}
