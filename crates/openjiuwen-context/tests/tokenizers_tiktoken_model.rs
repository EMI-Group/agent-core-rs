#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! `tiktoken` `.model` loading: parsing, the merge loop, pre-tokenization, special tokens and the selection seam.
use std::fmt::Write as _;
use std::io::Write as _;

use base64::prelude::{Engine as _, BASE64_STANDARD};
use openjiuwen_common::error::Error;
use openjiuwen_context::tokenizers::{
    load_native_tokenizer, NativeTokenizer, NativeTokenizerKind, PretokenizePattern, TiktokenModel,
    TiktokenSpecials, TokenizerSource,
};
use openjiuwen_foundation::prompt::TokenCounter;

/// Base64-encodes one vocabulary token.
fn b64(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

/// A `.model` document holding the 256 single-byte tokens.
fn byte_vocabulary() -> String {
    let mut text = String::new();
    for value in 0_u16..=255 {
        let byte = u8::try_from(value).expect("0..=255 fits a byte");
        writeln!(text, "{} {byte}", b64(&[byte])).expect("writing to a String");
    }
    text
}

/// A `.model` document with the byte vocabulary plus extra merged tokens.
fn model_with_merges(merges: &[(&str, u32)]) -> String {
    let mut text = byte_vocabulary();
    for (token, rank) in merges {
        writeln!(text, "{} {rank}", b64(token.as_bytes())).expect("writing to a String");
    }
    text
}

/// A hand-written vocabulary: `a`, `b`, `c` merge through `ab` into `abc`.
fn chained_model_text() -> String {
    let mut text = String::new();
    for (token, rank) in [("a", 0_u32), ("b", 1), ("c", 2), ("ab", 3), ("abc", 4)] {
        writeln!(text, "{} {rank}", b64(token.as_bytes())).expect("writing to a String");
    }
    text
}

/// Parses a model or panics with a useful message.
fn parse_model(text: &str) -> TiktokenModel {
    TiktokenModel::from_text(text).expect("fixture parses")
}

/// Parses the byte vocabulary and splits `text` with its (default Kimi) rule.
fn pretokenize(text: &str) -> Vec<String> {
    parse_model(&byte_vocabulary()).pretokenize(text)
}

#[test]
fn hand_written_model_merges_in_rank_order() {
    let model = parse_model(&chained_model_text());
    assert_eq!(model.mergeable_count(), 5);
    assert_eq!(model.vocab_size(), 5);
    assert_eq!(model.max_token_bytes(), 3);
    // The fixture declares only five tokens, so the byte-coverage warning is expected.
    assert!(
        model
            .warnings()
            .join("\n")
            .contains("single-byte tokens are missing"),
        "{:?}",
        model.warnings()
    );

    // `a b` is the lowest-ranked pair, then `ab c`.
    assert_eq!(model.encode("abc"), vec![4]);
    assert_eq!(model.encode("ab"), vec![3]);
    assert_eq!(model.encode("a"), vec![0]);
    assert_eq!(model.encode("aabc"), vec![0, 4]);
    assert_eq!(model.encode("abcabc"), vec![4, 4]);
    assert_eq!(model.encode(""), Vec::<u32>::new());
}

#[test]
fn three_way_merge_chain_runs_through_every_rung() {
    let model = parse_model(&chained_model_text());
    // No piece of length three exists before the two merges; the ids prove the
    // chain ran instead of emitting one id per byte.
    assert_eq!(model.encode("abc"), vec![4]);
    assert_ne!(model.encode("abc"), vec![0, 1, 2]);
    assert_eq!(model.decode(&[4]), "abc");
    assert_eq!(model.decode(&model.encode("abca")), "abca");
    // Unknown ids are skipped, like every other native decoder.
    assert_eq!(model.decode(&[99, 4]), "abc");
}

#[test]
fn a_lower_rank_merge_beats_a_longer_vocabulary_piece() {
    // `ab` (rank 5) is the longer piece, but `bc` (rank 3) merges first.
    let mut text = String::new();
    for (token, rank) in [("a", 0_u32), ("b", 1), ("c", 2), ("bc", 3), ("ab", 5)] {
        writeln!(text, "{} {rank}", b64(token.as_bytes())).expect("writing to a String");
    }
    let model = parse_model(&text);
    assert_eq!(model.encode("abc"), vec![0, 3], "b+c merges before a+b");
    assert_eq!(model.encode("ab"), vec![5]);
}

#[test]
fn byte_level_model_round_trips_utf8() {
    let model = parse_model(&byte_vocabulary());
    for text in ["hello", "héllo", "😀", "中文 mixed", "a\tb\nc"] {
        let ids = model.encode(text);
        assert!(!ids.is_empty(), "{text:?} produced no ids");
        assert_eq!(model.decode(&ids), text, "round trip of {text:?}");
    }
    // One id per byte: the fixture declares no merges.
    assert_eq!(model.encode("😀").len(), 4);
}

#[test]
fn count_text_is_the_number_of_encoded_ids() {
    let text = model_with_merges(&[("ab", 256_u32), ("abc", 257)]);
    let tokenizer = NativeTokenizer::from_tiktoken_model(&text).expect("fixture loads");
    for sample in ["", "a", "abc", "hello world", "😀"] {
        assert_eq!(tokenizer.count_text(sample), tokenizer.encode(sample).len());
    }
    assert_eq!(tokenizer.count_text("abc"), 1);
}

#[test]
fn native_tokenizer_surface_for_a_model_document() {
    let text = model_with_merges(&[("ab", 256_u32), ("abc", 257)]);
    let tokenizer = NativeTokenizer::from_tiktoken_model(&text).expect("loads");
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Tiktoken);
    // 258 mergeable entries plus the 23 built-in K2 tokens and 256 reserved ids.
    assert_eq!(tokenizer.vocab_size(), 537);
    assert_eq!(tokenizer.tokenizer_id(), None);
    assert!(
        tokenizer.warnings().is_empty(),
        "{:?}",
        tokenizer.warnings()
    );
    let specials = tokenizer.special_tokens();
    assert!(specials.contains(&"<|im_end|>"));
    assert!(specials.contains(&"<|reserved_token_258|>"));
    assert_eq!(tokenizer.encode("abc"), vec![257]);
    // A document held in memory has no model name, so the K2 fallback is used.
    assert_eq!(tokenizer.encode("<|im_end|>"), vec![163_586]);
    assert_eq!(tokenizer.decode(&tokenizer.encode("abcabc")), "abcabc");
    assert_eq!(tokenizer.count_text(""), 0);
}

#[test]
fn a_incomplete_byte_vocabulary_is_reported_not_hidden() {
    let model = parse_model(&chained_model_text());
    let warnings = model.warnings().join("\n");
    assert!(
        warnings.contains("single-byte tokens are missing"),
        "{warnings}"
    );
}

#[test]
fn kimi_pattern_keeps_a_leading_space_with_the_word() {
    assert_eq!(pretokenize("hello"), vec!["hello"]);
    assert_eq!(pretokenize(" hello"), vec![" hello"]);
    assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
    // Two spaces: the lookahead leaves the first behind and the second leads the word.
    assert_eq!(pretokenize("  hello"), vec![" ", " hello"]);
}

#[test]
fn kimi_pattern_attaches_contractions_to_the_previous_word() {
    assert_eq!(pretokenize("don't"), vec!["don't"]);
    assert_eq!(pretokenize("it's"), vec!["it's"]);
    assert_eq!(pretokenize("I'll"), vec!["I'll"]);
    assert_eq!(pretokenize("we're"), vec!["we're"]);
    assert_eq!(pretokenize("they've"), vec!["they've"]);
    assert_eq!(pretokenize("I'm"), vec!["I'm"]);
}

#[test]
fn kimi_pattern_limits_digit_runs_to_three() {
    assert_eq!(pretokenize("123"), vec!["123"]);
    assert_eq!(pretokenize("1234"), vec!["123", "4"]);
    assert_eq!(pretokenize("1234567"), vec!["123", "456", "7"]);
    assert_eq!(pretokenize("a1"), vec!["a", "1"]);
}

#[test]
fn kimi_pattern_groups_symbol_runs() {
    assert_eq!(pretokenize("!?!\n"), vec!["!?!\n"]);
    assert_eq!(pretokenize("@@"), vec!["@@"]);
    // The Kimi rule lets one symbol lead the next word.
    assert_eq!(pretokenize("a-b"), vec!["a", "-b"]);
}

#[test]
fn kimi_pattern_splits_han_runs() {
    assert_eq!(pretokenize("你好世界"), vec!["你好世界"]);
    assert_eq!(pretokenize("中文abc"), vec!["中文", "abc"]);
    assert_eq!(pretokenize("abc中文"), vec!["abc", "中文"]);
}

#[test]
fn kimi_pattern_applies_the_trailing_whitespace_rule() {
    assert_eq!(pretokenize("hello   "), vec!["hello", "   "]);
    // `\s+(?!\S)` backtracks one character, so two of the three spaces stay alone.
    assert_eq!(pretokenize("hello   world"), vec!["hello", "  ", " world"]);
    assert_eq!(pretokenize("a\n  b"), vec!["a", "\n", " ", " b"]);
    // The pieces always cover the whole input.
    for text in ["hello   ", "hello   world", "a\n  b", " \n\t "] {
        assert_eq!(pretokenize(text).concat(), text);
    }
}

#[test]
fn kimi_is_the_default_pattern_and_gpt2_is_opt_in() {
    let byte_model = parse_model(&byte_vocabulary());
    assert_eq!(byte_model.pattern(), PretokenizePattern::Kimi);

    let gpt2 = byte_model.with_pattern(PretokenizePattern::Gpt2);
    assert_eq!(gpt2.pattern(), PretokenizePattern::Gpt2);
    // GPT-2 cuts the contraction off; Kimi attaches it.
    assert_eq!(gpt2.pretokenize("don't"), vec!["don", "'t"]);
    assert_eq!(gpt2.pretokenize(" hello"), vec![" hello"]);
    // GPT-2 keeps a whole digit run.
    assert_eq!(gpt2.pretokenize("12345"), vec!["12345"]);
    assert_eq!(
        gpt2.pretokenize("hello   world"),
        vec!["hello", "  ", " world"]
    );
}

#[test]
fn special_tokens_come_from_a_tokenizer_config_document() {
    let config = r#"{
        "added_tokens_decoder": {
            "5": {"content": "<|end|>", "special": true},
            "6": {"content": "<|sep|>", "special": true}
        }
    }"#;
    let specials = TiktokenSpecials::from_config_json(config, 5)
        .expect("valid config")
        .expect("declares tokens");
    assert_eq!(specials.token_id("<|end|>"), Some(5));
    assert_eq!(specials.token_id("<|sep|>"), Some(6));
    assert_eq!(specials.text(5), Some("<|end|>"));
    // Reserved ids start at the base size and skip declared ones.
    assert_eq!(specials.token_id("<|reserved_token_5|>"), None);
    assert_eq!(specials.token_id("<|reserved_token_7|>"), Some(7));
    assert_eq!(specials.text(7), Some("<|reserved_token_7|>"));
}

#[test]
fn a_config_without_added_tokens_returns_none() {
    assert!(TiktokenSpecials::from_config_json("{}", 5)
        .expect("valid JSON")
        .is_none());
    assert!(TiktokenSpecials::from_config_json(
        r#"{"added_tokens_decoder": {"0": {"content": ""}}}"#,
        5
    )
    .expect("valid JSON")
    .is_none());
}

#[test]
fn a_malformed_tokenizer_config_is_a_validation_error() {
    let error = TiktokenSpecials::from_config_json("not json", 5).expect_err("malformed");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("tokenizer_config.json"));
}

#[test]
fn built_in_kimi_fallbacks_are_selected_by_model_name() {
    let k2 = TiktokenSpecials::for_model("kimi-k2", 163_584);
    assert_eq!(k2.token_id("<|im_end|>"), Some(163_586));
    assert_eq!(k2.token_id("<|tool_call_begin|>"), Some(163_597));

    let k3 = TiktokenSpecials::for_model("moonshot/Kimi-K3", 163_584);
    assert_eq!(k3.token_id("<|end_of_msg|>"), Some(163_586));
    assert_eq!(k3.token_id("<osagent_mode>"), Some(163_649));
    assert_eq!(k3.token_id("<|im_end|>"), None);

    // Reserved ids pad from the base size and skip used ones.
    let padded = TiktokenSpecials::for_model("kimi-k2", 100);
    assert_eq!(padded.token_id("<|reserved_token_100|>"), Some(100));
    assert_eq!(padded.token_id("<|reserved_token_355|>"), Some(355));
    assert_eq!(padded.token_id("<|reserved_token_356|>"), None);
}

#[test]
fn a_special_token_encodes_to_exactly_one_id() {
    let config = r#"{"added_tokens_decoder": {"300": {"content": "<|end|>", "special": true}}}"#;
    let specials = TiktokenSpecials::from_config_json(config, 256)
        .expect("valid config")
        .expect("declares a token");
    let model = parse_model(&byte_vocabulary()).with_special_tokens(specials);

    assert_eq!(model.special_token_id("<|end|>"), Some(300));
    assert_eq!(model.encode("<|end|>"), vec![300]);
    assert_eq!(model.encode("a<|end|>b"), vec![97, 300, 98]);
    assert_eq!(model.decode(&[300]), "<|end|>");
    assert_eq!(model.decode(&model.encode("a<|end|>b")), "a<|end|>b");
    let texts = model.special_tokens().texts();
    assert!(texts.contains(&"<|end|>"));
    // 255 reserved ids (256..512 minus the declared 300) plus the declared one.
    assert_eq!(texts.len(), 256);
}

#[test]
fn native_tokenizer_carries_special_tokens_from_a_wrapped_vocabulary() {
    let specials = TiktokenSpecials::from_config_json(
        r#"{"added_tokens_decoder": {"256": {"content": "<|end|>", "special": true}}}"#,
        256,
    )
    .expect("valid config")
    .expect("declares a token");
    let model = parse_model(&byte_vocabulary()).with_special_tokens(specials);
    let tokenizer = NativeTokenizer::from_tiktoken(model);

    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Tiktoken);
    assert!(tokenizer.special_tokens().contains(&"<|end|>"));
    assert_eq!(tokenizer.encode("<|end|>"), vec![256]);
    assert_eq!(tokenizer.decode(&tokenizer.encode("a<|end|>")), "a<|end|>");
}

#[test]
fn malformed_lines_name_their_line_number() {
    let text = format!("{} 0\nnot-base64 1\n", b64(b"a"));
    let error = TiktokenModel::from_text(&text).expect_err("bad base64");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("line 2"), "{error}");

    let text = format!("{} zero\n", b64(b"a"));
    let error = TiktokenModel::from_text(&text).expect_err("bad rank");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("line 1"), "{error}");
    assert!(error.to_string().contains("not an integer"), "{error}");
}

#[test]
fn an_empty_file_is_rejected() {
    for text in ["", "\n\n", "# only a comment\n"] {
        let error = TiktokenModel::from_text(text).expect_err("no entries");
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("empty"), "{error}");
    }
}

#[test]
fn a_duplicate_rank_is_rejected_with_a_line_number() {
    let text = format!("{} 0\n{} 0\n", b64(b"a"), b64(b"b"));
    let error = TiktokenModel::from_text(&text).expect_err("duplicate rank");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("line 2"), "{error}");
    assert!(error.to_string().contains("duplicate rank 0"), "{error}");
}

#[test]
fn a_duplicate_token_is_rejected() {
    let text = format!("{} 0\n{} 1\n", b64(b"a"), b64(b"a"));
    let error = TiktokenModel::from_text(&text).expect_err("duplicate token");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("duplicate token"), "{error}");
}

#[test]
fn comments_and_blank_lines_are_tolerated() {
    let text = format!("# header\n\n{} 0\n  \n{} 1\n", b64(b"a"), b64(b"b"));
    let model = parse_model(&text);
    assert_eq!(model.mergeable_count(), 2);
}

#[test]
fn a_model_file_loads_as_an_exact_native_selection() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("tiny.model");
    std::fs::write(&path, model_with_merges(&[("ab", 256_u32), ("abc", 257)]))
        .expect("write fixture");

    let selected = load_native_tokenizer(&path).expect("model loads");
    let display = path.display().to_string();
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert!(!selected.estimated());
    assert_eq!(selected.fallback_reason(), None);
    assert_eq!(selected.tokenizer(), Some(display.as_str()));
    assert_eq!(selected.count_text("abc"), 1);
    assert_eq!(selected.count_text(""), 0);
    assert_eq!(selected.count_text("hello"), 5);
}

#[test]
fn a_model_file_reads_the_sibling_tokenizer_config() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("kimi-k3.model");
    std::fs::write(&path, model_with_merges(&[])).expect("write vocabulary");
    let config = r#"{"added_tokens_decoder": {"400": {"content": "<|end|>", "special": true}}}"#;
    std::fs::write(directory.path().join("tokenizer_config.json"), config).expect("write config");

    let selected = load_native_tokenizer(&path).expect("model loads");
    assert_eq!(selected.count_text("<|end|>"), 1);
    assert_eq!(selected.count_text("a<|end|>"), 2);
}

#[test]
fn a_model_file_without_config_uses_the_built_in_kimi_fallback() {
    let directory = tempfile::tempdir().expect("temp dir");
    let k3_path = directory.path().join("Kimi-K3.model");
    std::fs::write(&k3_path, model_with_merges(&[])).expect("write vocabulary");
    let k3 = NativeTokenizer::from_tiktoken_model_file(&k3_path).expect("loads");
    assert!(k3.special_tokens().contains(&"<|end_of_msg|>"));
    assert_eq!(k3.encode("<|end_of_msg|>"), vec![163_586]);

    let k2_path = directory.path().join("kimi-k2.model");
    std::fs::write(&k2_path, model_with_merges(&[])).expect("write vocabulary");
    let k2 = NativeTokenizer::from_tiktoken_model_file(&k2_path).expect("loads");
    assert!(k2.special_tokens().contains(&"<|im_end|>"));
    assert_eq!(k2.encode("<|im_end|>"), vec![163_586]);

    // An explicit model name overrides the file stem for the K2/K3 choice.
    let explicit =
        NativeTokenizer::from_tiktoken_model_file_with_model(&k2_path, "moonshotai/Kimi-K3")
            .expect("loads");
    assert!(explicit.special_tokens().contains(&"<|end_of_msg|>"));
    assert_eq!(explicit.encode("<|end_of_msg|>"), vec![163_586]);

    let vocabulary =
        TiktokenModel::from_file_with_model(&k2_path, "moonshotai/Kimi-K3").expect("loads");
    assert_eq!(vocabulary.special_token_id("<|end_of_msg|>"), Some(163_586));
    assert_eq!(vocabulary.special_token_id("<|im_end|>"), None);
}

#[test]
fn a_tokenizer_json_file_still_loads() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("tokenizer.json");
    let document = r#"{
        "name": "tiny-bpe",
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": ["a b", "ab c"]},
        "pre_tokenizer": {"type": "Whitespace"}
    }"#;
    let mut file = std::fs::File::create(&path).expect("create");
    file.write_all(document.as_bytes()).expect("write");

    let selected = load_native_tokenizer(&path).expect("tokenizer.json loads");
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert!(!selected.estimated());
    assert_eq!(selected.count_text("abc"), 1);
}

#[test]
fn an_unrecognised_extension_is_sniffed_as_a_model_file() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("vocabulary.txt");
    std::fs::write(&path, model_with_merges(&[("ab", 256_u32), ("abc", 257)]))
        .expect("write fixture");

    let selected = load_native_tokenizer(&path).expect("sniffed model loads");
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert_eq!(selected.count_text("abc"), 1);
}

#[test]
fn an_undecodable_file_reports_both_attempts() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("vocabulary.bin");
    std::fs::write(&path, "this is not a vocabulary\n").expect("write fixture");

    let error = load_native_tokenizer(&path).expect_err("nothing decodes");
    assert!(matches!(error, Error::Validation(_)));
    let message = error.to_string();
    assert!(
        message.contains("neither a decodable tokenizer.json"),
        "{message}"
    );
    assert!(message.contains("`.model` vocabulary"), "{message}");
}

#[test]
fn a_malformed_model_file_is_a_validation_error() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("bad.model");
    std::fs::write(&path, "not a vocabulary\n").expect("write fixture");

    let error = load_native_tokenizer(&path).expect_err("nothing decodes");
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("line 1"), "{error}");
}

#[test]
fn a_missing_artifact_is_not_found() {
    let directory = tempfile::tempdir().expect("temp dir");
    let error =
        load_native_tokenizer(directory.path().join("missing.model")).expect_err("missing file");
    assert!(matches!(error, Error::NotFound(_)));
}
