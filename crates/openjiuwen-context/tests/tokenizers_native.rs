#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Native `tokenizer.json` decoding: model pipelines, added tokens, decoders and the selection seam.

use openjiuwen_common::error::Error;
use openjiuwen_context::tokenizers::{
    byte_level_alphabet, load_native_tokenizer, NativeTokenizer, NativeTokenizerKind,
    TokenizerSource,
};
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};

/// A tiny BPE artifact: `a`, `b`, `c` merge into `abc`, nothing else is known.
const BPE_FIXTURE: &str = r#"{
    "name": "tiny-bpe",
    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": ["a b", "ab c"]},
    "pre_tokenizer": {"type": "Whitespace"}
}"#;

/// A byte-level BPE whose vocabulary is the 256-entry byte alphabet and which declares no merges.
fn byte_level_fixture() -> String {
    let mut vocab = serde_json::Map::new();
    for (byte, ch) in byte_level_alphabet().iter().enumerate() {
        vocab.insert(ch.to_string(), serde_json::Value::from(byte as u64));
    }
    serde_json::json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "decoder": {"type": "ByteLevel"}
    })
    .to_string()
}

fn parse(json: &str) -> NativeTokenizer {
    NativeTokenizer::from_json_str(json).expect("fixture loads")
}

#[test]
fn bpe_merges_produce_the_expected_ids() {
    let tokenizer = parse(BPE_FIXTURE);
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Bpe);
    assert_eq!(tokenizer.tokenizer_id(), Some("tiny-bpe"));
    assert_eq!(tokenizer.vocab_size(), 5);
    assert_eq!(tokenizer.warnings().len(), 0);

    assert_eq!(tokenizer.encode("abc"), vec![4]);
    // `a b` merges at the second position, then `ab c` completes the run.
    assert_eq!(tokenizer.encode("aabc"), vec![0, 4]);
    assert_eq!(tokenizer.encode("abc abc"), vec![4, 4]);
    assert_eq!(tokenizer.encode("").len(), 0);
}

#[test]
fn bpe_decode_round_trips_a_single_piece() {
    let tokenizer = parse(BPE_FIXTURE);
    assert_eq!(tokenizer.decode(&[4]), "abc");
    assert_eq!(tokenizer.decode(&tokenizer.encode("abc")), "abc");
    // Unknown ids are skipped instead of failing.
    assert_eq!(tokenizer.decode(&[99, 4]), "abc");
}

#[test]
fn byte_level_bytes_survive_encode_then_decode() {
    let tokenizer = parse(&byte_level_fixture());
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Bpe);
    for text in ["hello", "héllo", "😀", "中文 mixed", "a\tb\nc"] {
        let ids = tokenizer.encode(text);
        assert!(!ids.is_empty(), "{text:?} produced no ids");
        assert_eq!(tokenizer.decode(&ids), text, "round trip of {text:?}");
    }
    // One id per byte: the fixture declares no merges.
    assert_eq!(tokenizer.encode("😀").len(), 4);
}

#[test]
fn count_text_is_the_number_of_encoded_ids() {
    let tokenizer = parse(&byte_level_fixture());
    for text in ["", "a", "hello world", "😀"] {
        assert_eq!(tokenizer.count_text(text), tokenizer.encode(text).len());
    }
    assert_eq!(tokenizer.count_text("ab"), 2);
}

#[test]
fn added_and_special_tokens_encode_as_one_id() {
    let tokenizer = parse(
        r#"{
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "cat": 3, "category": 4, "<|endoftext|>": 5}, "unk_token": "[UNK]"},
            "pre_tokenizer": {"type": "Whitespace"},
            "added_tokens": [
                {"id": 5, "content": "<|endoftext|>", "special": true},
                {"id": 3, "content": "cat", "single_word": true}
            ]
        }"#,
    );
    assert_eq!(tokenizer.encode("hello <|endoftext|> world"), vec![1, 5, 2]);
    assert_eq!(tokenizer.special_tokens(), vec!["<|endoftext|>"]);
    // `single_word` keeps the added token out of the middle of `category`.
    assert_eq!(tokenizer.encode("cat category"), vec![3, 4]);
    assert_eq!(tokenizer.decode(&[1, 5, 2]), "hello <|endoftext|> world");
}

#[test]
fn added_tokens_can_strip_surrounding_whitespace() {
    let tokenizer = parse(
        r#"{
            "model": {"type": "WordLevel", "vocab": {"[MASK]": 0, "a": 1, "b": 2}, "unk_token": "[MASK]"},
            "added_tokens": [{"id": 0, "content": "[MASK]", "lstrip": true, "rsw": true}]
        }"#,
    );
    assert_eq!(tokenizer.encode("a  [MASK]  b"), vec![1, 0, 2]);
}

#[test]
fn wordpiece_respects_continuations_and_the_character_limit() {
    let tokenizer = parse(
        r###"{
            "model": {
                "type": "WordPiece",
                "vocab": {"[UNK]": 0, "hello": 1, "##ing": 2, "##s": 3, "supercalifragilistic": 4},
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 8
            },
            "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": true, "lowercase": true},
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "decoder": {"type": "WordPiece", "prefix": "##"}
        }"###,
    );
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::WordPiece);
    assert_eq!(tokenizer.encode("hello"), vec![1]);
    assert_eq!(tokenizer.encode("helloing"), vec![1, 2]);
    assert_eq!(tokenizer.encode("Hellos"), vec![1, 3]);
    // The normalizer lowercases and strips the accent before the lookup.
    assert_eq!(tokenizer.encode("HÉLLO"), vec![1]);
    // A word longer than `max_input_chars_per_word` becomes `[UNK]`, even when it is in the vocabulary.
    assert_eq!(tokenizer.encode("supercalifragilistic"), vec![0]);
    assert_eq!(tokenizer.decode(&[1, 2]), "helloing");
}

#[test]
fn unigram_picks_the_highest_scoring_segmentation() {
    let tokenizer = parse(
        r#"{
            "model": {
                "type": "Unigram",
                "vocab": [
                    ["<unk>", 0.0], ["a", -2.0], ["b", -2.0], ["c", -2.0], ["d", -2.0],
                    ["ab", -0.5], ["cd", -0.5], ["abc", -10.0], ["bcd", -10.0], ["abcd", -10.0]
                ],
                "unk_id": 0
            },
            "pre_tokenizer": {"type": "Whitespace"}
        }"#,
    );
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Unigram);
    // `abcd` (-10) and four single characters (8) both lose to `ab` + `cd` (-1).
    assert_eq!(tokenizer.encode("abcd"), vec![5, 6]);
    assert_eq!(tokenizer.encode("az"), vec![1, 0]);
    assert_eq!(tokenizer.decode(&[5, 6]), "ab cd");
}

#[test]
fn sequence_pretokenizers_and_lowercase_normalizers_compose() {
    let tokenizer = parse(
        r#"{
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "1": 2, "a1": 3}, "merges": ["a 1"]},
            "normalizer": {"type": "Lowercase"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Whitespace"}, {"type": "Digits"}]}
        }"#,
    );
    assert_eq!(tokenizer.warnings().len(), 0);
    // `A` lowercases to `a`; the digits pre-tokenizer stops `a1` from merging.
    assert_eq!(tokenizer.encode("A1 B"), vec![0, 2, 1]);
}

#[test]
fn unsupported_pipeline_nodes_warn_but_still_count() {
    let tokenizer = parse(
        r#"{
            "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
            "pre_tokenizer": {"type": "FancySplitter"}
        }"#,
    );
    assert_eq!(tokenizer.encode("a a"), vec![0, 0]);
    assert_eq!(tokenizer.warnings().len(), 1);
    assert!(tokenizer.warnings()[0].contains("FancySplitter"));
}

#[test]
fn regex_splits_are_applied_without_a_warning() {
    let tokenizer = parse(
        r#"{
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, " ": 2}, "merges": []},
            "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"}
        }"#,
    );
    assert!(
        tokenizer.warnings().is_empty(),
        "a regex split is supported: {:?}",
        tokenizer.warnings()
    );
    // `Isolated` keeps the whitespace as its own piece, so it is encoded as id 2.
    assert_eq!(tokenizer.encode("a a"), vec![0, 2, 0]);
}

#[test]
fn malformed_documents_are_validation_errors() {
    let error = NativeTokenizer::from_json_str("{ not json").expect_err("malformed JSON");
    assert!(matches!(error, Error::Validation(_)));

    let error = NativeTokenizer::from_json_str(r#"{"model": {"type": "BPE"}}"#)
        .expect_err("BPE without vocab");
    assert!(matches!(error, Error::Validation(_)));

    let error = NativeTokenizer::from_json_str(r#"{"model": {"type": "WordPiece"}}"#)
        .expect_err("WordPiece without vocab");
    assert!(matches!(error, Error::Validation(_)));
}

#[test]
fn unknown_model_types_are_unsupported() {
    for model in ["UnigramV2", "BPE-dropout", "Fancy"] {
        let json = format!(r#"{{"model": {{"type": "{model}"}}}}"#);
        let error = NativeTokenizer::from_json_str(&json).expect_err("unknown model type");
        assert!(
            matches!(error, Error::Unsupported(_)),
            "{model} should be Unsupported"
        );
        assert!(
            error.to_string().contains(model),
            "the message should name `{model}`: {error}"
        );
    }
}

#[test]
fn missing_artifacts_are_not_found() {
    let error = load_native_tokenizer("/nonexistent/tokenizer.json").expect_err("missing path");
    assert!(matches!(error, Error::NotFound(_)));
}

#[test]
fn tiktoken_model_files_load_as_a_native_selection() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tiktoken.model");
    // `YQ==` is the byte `a`, rank 0: a one-entry `.model` vocabulary.
    std::fs::write(&path, b"YQ== 0\n").expect("write");
    let selection = load_native_tokenizer(&path).expect(".model loads");
    assert_eq!(selection.source(), TokenizerSource::NativeTokenizer);
    assert!(!selection.estimated());
    assert_eq!(selection.fallback_reason(), None);
    // A `.model` artifact carries no name, so the path is used.
    let display = path.display().to_string();
    assert_eq!(selection.tokenizer(), Some(display.as_str()));
    assert_eq!(selection.count_text("a"), 1);
}

#[test]
fn loading_from_a_file_yields_a_native_selection() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, BPE_FIXTURE).expect("write fixture");

    let selection = load_native_tokenizer(&path).expect("fixture loads");
    assert_eq!(selection.source(), TokenizerSource::NativeTokenizer);
    assert!(!selection.estimated());
    assert_eq!(selection.fallback_reason(), None);
    // The artifact id wins over the path.
    assert_eq!(selection.tokenizer(), Some("tiny-bpe"));
    assert_eq!(selection.count_text("abc"), 1);

    let measurement = selection.measure("abc abc");
    assert_eq!(measurement.tokens, 2);
    assert_eq!(measurement.source, TokenizerSource::NativeTokenizer);
    assert!(!measurement.estimated);
    assert_eq!(measurement.tokenizer.as_deref(), Some("tiny-bpe"));

    let messages = vec![BaseMessage::user("abc"), BaseMessage::assistant("abc")];
    let message_measurement = selection.measure_messages(&messages);
    assert_eq!(
        message_measurement.tokens,
        selection.count_messages(&messages)
    );
    assert!(message_measurement.tokens > selection.count_text("abcabc"));
}

#[test]
fn the_counter_adds_chat_framing_per_message() {
    let tokenizer = parse(BPE_FIXTURE);
    let messages = vec![BaseMessage::user("abc"), BaseMessage::assistant("abc")];
    let raw = tokenizer.count_text("abc") + tokenizer.count_text("abc");
    assert_eq!(tokenizer.count_messages(&[]), 0);
    assert!(
        tokenizer.count_messages(&messages) > raw,
        "framing should add tokens on top of the raw text"
    );
    // The workspace convention is a fixed four-token overhead per message.
    assert_eq!(tokenizer.count_messages(&messages), raw + 8);
}
