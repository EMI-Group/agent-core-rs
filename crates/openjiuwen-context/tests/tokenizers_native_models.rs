#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Model-level decoding for native artifacts: BPE and Unigram round-trips and single-sequence post-processing.
//!
//! Every fixture is built with `serde_json::json!`; the fixtures that declare no decoder node are decoded with the explicit `NativeTokenizer::decode_model_level` opt-in, which is what reassembles byte-level spellings, while `decode` itself keeps `HuggingFace`'s single-space join. The assertions also cover the framing a declared post-processor adds on encode.

use openjiuwen_context::tokenizers::{NativeTokenizer, NativeTokenizerKind};
use openjiuwen_foundation::prompt::TokenCounter;

fn parse(document: &serde_json::Value) -> NativeTokenizer {
    NativeTokenizer::from_json_str(&document.to_string()).expect("fixture loads")
}

/// The 256 `<0xXX>` spellings a byte-fallback vocabulary stores, ranked by byte value.
fn byte_vocab() -> serde_json::Map<String, serde_json::Value> {
    let mut vocab = serde_json::Map::new();
    for byte in 0_u16..=0xFF {
        vocab.insert(format!("<0x{byte:02X}>"), serde_json::Value::from(byte));
    }
    vocab
}

#[test]
fn bpe_continuation_prefix_round_trips_without_a_decoder() {
    let tokenizer = parse(&serde_json::json!({
        "name": "continuation-bpe",
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "##b": 3, "##c": 4},
            "merges": [],
            "continuing_subword_prefix": "##"
        },
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Bpe);
    assert_eq!(tokenizer.warnings().len(), 0, "{:?}", tokenizer.warnings());
    assert_eq!(tokenizer.encode("abc"), vec![0, 3, 4]);
    // The continuation prefix is stripped again, so the count round-trips through text.
    assert_eq!(
        tokenizer.decode_model_level(&tokenizer.encode("abc")),
        "abc"
    );
    // Without the opt-in the spellings are joined the way `HuggingFace` joins them.
    assert_eq!(tokenizer.decode(&tokenizer.encode("abc")), "a ##b ##c");
    assert_eq!(tokenizer.count_text("abc"), 3);
}

#[test]
fn bpe_end_of_word_suffix_round_trips_without_a_decoder() {
    let tokenizer = parse(&serde_json::json!({
        "model": {
            "type": "BPE",
            "vocab": {"hel": 0, "lo</w>": 1, "world</w>": 2},
            "merges": [],
            "end_of_word_suffix": "</w>"
        },
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    assert_eq!(tokenizer.decode_model_level(&[0, 1]), "hello");
    assert_eq!(tokenizer.decode_model_level(&[2]), "world");
    // `decode` keeps the space join, so the suffix stays visible.
    assert_eq!(tokenizer.decode(&[0, 1]), "hel lo</w>");
}

#[test]
fn bpe_byte_fallback_round_trips_multibyte_text() {
    let tokenizer = parse(&serde_json::json!({
        "name": "byte-fallback-bpe",
        "model": {
            "type": "BPE",
            "vocab": byte_vocab(),
            "merges": [],
            "byte_fallback": true
        },
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    // `é` is `0xC3 0xA9`; the ids are the byte values because the fixture ranks them that way.
    assert_eq!(tokenizer.encode("é"), vec![0xC3, 0xA9]);
    assert_eq!(
        tokenizer.encode("héllo"),
        vec![0x68, 0xC3, 0xA9, 0x6C, 0x6C, 0x6F]
    );
    // A space can only survive with a decoder node that puts it back, so the round trips stay single-piece.
    for text in ["é", "héllo", "中文", "😀"] {
        let ids = tokenizer.encode(text);
        assert_eq!(
            tokenizer.decode_model_level(&ids),
            text,
            "round trip of {text:?}"
        );
    }
    assert_eq!(tokenizer.count_text("中文"), 6);
}

#[test]
fn bpe_byte_fallback_round_trips_a_character_split_across_tokens() {
    // `中` is three bytes, so the decode has to assemble a whole character from three tokens.
    let tokenizer = parse(&serde_json::json!({
        "model": {
            "type": "BPE",
            "vocab": byte_vocab(),
            "merges": [],
            "byte_fallback": true
        },
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    let ids = tokenizer.encode("中");
    assert_eq!(ids.len(), 3);
    assert_eq!(tokenizer.decode_model_level(&ids), "中");
    // A truncated run degrades to the replacement character instead of failing.
    assert_eq!(tokenizer.decode_model_level(&ids[..2]), "\u{FFFD}");
}

#[test]
fn unigram_encode_and_decode_agree_through_the_model() {
    let tokenizer = parse(&serde_json::json!({
        "name": "tiny-unigram",
        "model": {
            "type": "Unigram",
            "vocab": [
                ["<unk>", 0.0],
                ["a", -2.0],
                ["b", -2.0],
                ["c", -2.0],
                ["d", -2.0],
                ["ab", -0.5],
                ["cd", -0.5],
                ["abcd", -10.0]
            ],
            "unk_id": 0
        },
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    assert_eq!(tokenizer.kind(), NativeTokenizerKind::Unigram);
    assert_eq!(tokenizer.encode("abcd"), vec![5, 6]);
    assert_eq!(
        tokenizer.decode_model_level(&tokenizer.encode("abcd")),
        "abcd"
    );
    // An uncovered character becomes the unknown id, and the model decode keeps its spelling.
    assert_eq!(tokenizer.encode("az"), vec![1, 0]);
    assert_eq!(tokenizer.decode_model_level(&[1, 0]), "a<unk>");
    // The declared-decoder path has none, so it space-joins the same ids.
    assert_eq!(tokenizer.decode(&[1, 0]), "a <unk>");
}

#[test]
fn unigram_byte_fallback_round_trips_multibyte_text() {
    let mut vocab = vec![
        serde_json::json!(["<unk>", 0.0]),
        serde_json::json!(["a", -1.0]),
    ];
    for byte in 0_u16..=0xFF {
        vocab.push(serde_json::json!([format!("<0x{byte:02X}>"), -20.0]));
    }
    let tokenizer = parse(&serde_json::json!({
        "model": {"type": "Unigram", "vocab": vocab, "unk_id": 0, "byte_fallback": true},
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    for text in ["é", "中文"] {
        let ids = tokenizer.encode(text);
        assert_eq!(
            tokenizer.decode_model_level(&ids),
            text,
            "round trip of {text:?}"
        );
    }
    // `中` needs three fallback tokens, `é` two.
    assert_eq!(tokenizer.count_text("中"), 3);
    assert_eq!(tokenizer.count_text("é"), 2);
}

#[test]
fn a_template_post_processor_frames_encode_and_count() {
    let tokenizer = parse(&serde_json::json!({
        "model": {
            "type": "WordLevel",
            "vocab": {"[CLS]": 0, "[SEP]": 1, "hello": 2, "world": 3},
            "unk_token": "[CLS]"
        },
        "pre_tokenizer": {"type": "Whitespace"},
        "decoder": {"type": "Fuse"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [0], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [1], "tokens": ["[SEP]"]}
            }
        }
    }));
    assert_eq!(tokenizer.encode("hello world"), vec![0, 2, 3, 1]);
    assert_eq!(tokenizer.count_text("hello world"), 4);
    // One word: the framing is the only thing added.
    assert_eq!(tokenizer.count_text("hello"), 3);
    // The pair template is not applied to a single sequence.
    assert_eq!(tokenizer.encode("hello"), vec![0, 2, 1]);
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("hello")),
        "[CLS]hello[SEP]"
    );
}

#[test]
fn a_bert_post_processor_frames_encode_and_count() {
    let tokenizer = parse(&serde_json::json!({
        "model": {
            "type": "WordPiece",
            "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "##ing": 4},
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##"
        },
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "decoder": {"type": "WordPiece", "prefix": "##"},
        "post_processor": {"type": "BertProcessing", "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]}
    }));
    assert_eq!(tokenizer.encode("helloing"), vec![1, 3, 4, 2]);
    assert_eq!(tokenizer.count_text("helloing"), 4);
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("HELLOING")),
        "[CLS] helloing [SEP]"
    );
}

#[test]
fn no_post_processor_means_no_framing() {
    let tokenizer = parse(&serde_json::json!({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
        "pre_tokenizer": {"type": "Whitespace"}
    }));
    assert_eq!(tokenizer.encode("a b"), vec![0, 1]);
    assert_eq!(tokenizer.count_text("a b"), 2);
    assert_eq!(tokenizer.decode(&[0, 1]), "a b");
    assert_eq!(tokenizer.decode_model_level(&[0, 1]), "ab");
}
