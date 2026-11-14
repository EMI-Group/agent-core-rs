//! Unit tests for the `tiktoken` `.model` loader.
use std::fmt::Write as _;

use base64::prelude::{Engine as _, BASE64_STANDARD};
use openjiuwen_common::error::Error;

use super::tiktoken_model::{looks_like_tiktoken_model, TiktokenModel};

fn b64(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

fn tiny_model() -> TiktokenModel {
    let mut text = String::new();
    for (token, rank) in [("a", 0_u32), ("b", 1), ("c", 2), ("ab", 3), ("abc", 4)] {
        writeln!(text, "{} {rank}", b64(token.as_bytes())).expect("writing to a String");
    }
    TiktokenModel::from_text(&text).expect("tiny fixture")
}

#[test]
fn merges_apply_in_rank_order() {
    let model = tiny_model();
    assert_eq!(model.encode("abc"), vec![4]);
    assert_eq!(model.encode("ab"), vec![3]);
    assert_eq!(model.encode("aabc"), vec![0, 4]);
    assert_eq!(model.encode("abcabc"), vec![4, 4]);
}

#[test]
fn decode_round_trips_mergeable_tokens() {
    let model = tiny_model();
    assert_eq!(model.decode(&[4]), "abc");
    assert_eq!(model.decode(&model.encode("abca")), "abca");
}

#[test]
fn repeated_ranks_are_rejected_with_a_line_number() {
    let text = format!("{} 0\n{} 0\n", b64(b"a"), b64(b"b"));
    let error = TiktokenModel::from_text(&text).expect_err("duplicate rank");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("line 2"));
}

#[test]
fn a_document_without_entries_is_rejected() {
    let error = TiktokenModel::from_text("# only a comment\n\n").expect_err("empty");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("empty"));
}

#[test]
fn model_content_is_sniffed_from_the_first_entry() {
    assert!(looks_like_tiktoken_model("# comment\nYQ== 0\n"));
    assert!(!looks_like_tiktoken_model("{\"model\": {}}"));
    assert!(!looks_like_tiktoken_model(""));
}
