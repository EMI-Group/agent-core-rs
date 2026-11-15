#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Native `tokenizer.json` normalization: the `NFC` family, regex `Replace` nodes and their decoders.

use openjiuwen_context::tokenizers::NativeTokenizer;

/// A tokenizer whose normalizer folds compatibility forms, collapses whitespace and lowercases,
/// split on whitespace, with a BPE vocabulary and a decoder that normalizes and rewrites after joining.
fn fixture() -> String {
    serde_json::json!({
        "name": "fold-fix",
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "NFKC"},
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
                {"type": "Lowercase"}
            ]
        },
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "model": {
            "type": "BPE",
            "vocab": {
                "a": 0, "b": 1, "c": 2, "f": 3, "i": 4, "s": 5, "h": 6,
                "ab": 7, "abc": 8, "fi": 9
            },
            "merges": ["a b", "ab c", "f i"]
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "NFC"},
                {"type": "Replace", "pattern": {"Regex": "(abc)"}, "content": "<$1>"}
            ]
        }
    })
    .to_string()
}

fn parse(json: &str) -> NativeTokenizer {
    NativeTokenizer::from_json_str(json).expect("fixture loads")
}

#[test]
fn a_sequence_normalizer_feeds_the_model_and_loads_without_warnings() {
    let tokenizer = parse(&fixture());
    assert_eq!(tokenizer.warnings(), Vec::<String>::new());

    // `NFKC` folds the full-width letters, the regex collapses the tab, and `Lowercase` folds the case.
    assert_eq!(tokenizer.encode("ＡＢＣ\tabc"), vec![8, 8]);
    assert_eq!(tokenizer.encode("ABC  abc"), vec![8, 8]);
    assert_eq!(tokenizer.encode("abc"), vec![8]);
}

#[test]
fn the_decoder_sequence_normalizes_and_rewrites_the_joined_text() {
    let tokenizer = parse(&fixture());
    // The last decoder step replaces the capture group on the space-joined text.
    assert_eq!(tokenizer.decode(&[8, 8]), "<abc> <abc>");
    assert_eq!(tokenizer.decode(&[8]), "<abc>");
    // A decode of an encoded document returns the normalized (folded) text.
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("ＡＢＣ\tabc")),
        "<abc> <abc>"
    );
}

#[test]
fn a_compatibility_fold_makes_the_ligature_encode_like_ascii() {
    let tokenizer = parse(&fixture());
    // `ﬁ` (U+FB01) is folded by `NFKC`, so both spellings reach the vocabulary the same way.
    assert_eq!(tokenizer.encode("\u{fb01}sh"), tokenizer.encode("fish"));
    assert_eq!(tokenizer.encode("\u{fb01}sh"), vec![9, 5, 6]);
    assert_eq!(tokenizer.decode(&[9, 5, 6]), "fi s h");
}

#[test]
fn a_regex_that_does_not_compile_warns_and_still_loads() {
    let json = serde_json::json!({
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Replace", "pattern": {"Regex": "(unclosed"}, "content": "x"},
                {"type": "Lowercase"}
            ]
        },
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
        "decoder": {"type": "Replace", "pattern": {"Regex": "[["}, "content": "_"}
    })
    .to_string();

    let tokenizer = parse(&json);
    let warnings = tokenizer.warnings();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(warnings[0].contains("(unclosed"), "{warnings:?}");
    assert!(warnings[0].contains("does not compile"), "{warnings:?}");
    assert!(warnings[1].contains("[["), "{warnings:?}");

    // The broken normalizer step is skipped, so the rest of the pipeline still runs.
    assert_eq!(tokenizer.encode("A"), vec![0]);
    // The broken decoder step is skipped, so the plain join applies.
    assert_eq!(tokenizer.decode(&[0, 1]), "a b");
}

#[test]
fn the_nfc_family_is_no_longer_a_warning() {
    for kind in ["NFC", "NFD", "NFKC", "NFKD"] {
        let json = serde_json::json!({
            "normalizer": {"type": kind},
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        })
        .to_string();
        let tokenizer = parse(&json);
        assert_eq!(tokenizer.warnings(), Vec::<String>::new(), "{kind}");
        assert_eq!(tokenizer.encode("a b"), vec![0, 1], "{kind}");
    }
}

#[test]
fn a_strip_decoder_runs_before_the_final_join() {
    let json = serde_json::json!({
        "model": {
            "type": "WordLevel",
            "vocab": {"\u{2581}hel": 0, "\u{2581}lo": 1},
            "unk_token": "\u{2581}hel"
        },
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Strip", "content": "\u{2581}", "start": 1, "stop": 0},
            {"type": "Fuse"}
        ]}
    })
    .to_string();
    let tokenizer = parse(&json);
    assert_eq!(tokenizer.warnings(), Vec::<String>::new());
    assert_eq!(tokenizer.decode(&[0, 1]), "hello");
}
