//! Legacy CJK decoding end to end: name handling, explicit decoding, `auto` detection,
//! replacement of invalid sequences, and the streaming decoder.
//!
//! Python parity: `openjiuwen.core.sys_operation.local._text_encoding` (detection) and the
//! `errors="replace"` decoding of every local read. Every byte fixture here was produced
//! with (and verified against) Python's `gbk`/`gb18030`, `big5` and `shift_jis` codecs; the
//! expectation strings are hardcoded so the suite stays offline and needs no `python3`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_sysop::encoding::cjk::{decode, decode_strict, CjkDecoder, CjkEncoding};
use openjiuwen_sysop::encoding::{
    decode_text, detect_text_encoding, detected_encoding_name, encode_text, is_binary, TextEncoding,
};
use openjiuwen_sysop::BinaryPolicy;

/// `中文` in GBK/GB18030.
const ZHONG_WEN_GBK: [u8; 4] = [0xD6, 0xD0, 0xCE, 0xC4];

/// `中文` in Big5.
const ZHONG_WEN_BIG5: [u8; 4] = [0xA4, 0xA4, 0xA4, 0xE5];

/// `龍門` in Big5 (traditional-only characters).
const LONG_MEN_BIG5: [u8; 4] = [0xC0, 0x73, 0xAA, 0xF9];

/// `日本語` in Shift-JIS.
const NIHONGO_SJIS: [u8; 6] = [0x93, 0xFA, 0x96, 0x7B, 0x8C, 0xEA];

#[test]
fn gb18030_aliases_decode_simplified_chinese() {
    for name in ["gbk", "GBK", "gb2312", "cp936", "gb18030"] {
        assert_eq!(
            decode_text(&ZHONG_WEN_GBK, name, BinaryPolicy::Reject).unwrap(),
            "中文",
            "encoding name {name}"
        );
    }
    // A `gbk` request reports the canonical covering encoding.
    assert_eq!(TextEncoding::parse("gbk"), Some(TextEncoding::Gb18030));
}

#[test]
fn gb18030_decodes_mixed_ascii_and_cjk() {
    let bytes = b"name: \xD6\xD0\xCE\xC4 (GBK)\n";
    assert_eq!(
        decode_text(bytes, "gbk", BinaryPolicy::Lossy).unwrap(),
        "name: 中文 (GBK)\n"
    );
}

#[test]
fn gb18030_decodes_four_byte_sequences() {
    // 0x81 0x30 0x81 0x30 is the first four-byte index (U+0080); 0x90 0x30 0x81 0x30 the
    // first supplementary code point, U+10000; 0x95 0x32 0x82 0x36 is U+20000.
    assert_eq!(
        decode(&[0x81, 0x30, 0x81, 0x30], CjkEncoding::Gb18030),
        "\u{80}"
    );
    assert_eq!(
        decode(&[0x90, 0x30, 0x81, 0x30], CjkEncoding::Gb18030),
        "\u{10000}"
    );
    assert_eq!(
        decode(&[0x95, 0x32, 0x82, 0x36], CjkEncoding::Gb18030),
        "\u{20000}"
    );
    // The last supplementary mapping, U+10FFFF.
    assert_eq!(
        decode(&[0xE3, 0x32, 0x9A, 0x35], CjkEncoding::Gb18030),
        "\u{10FFFF}"
    );
    assert_eq!(
        decode_text(&[0x90, 0x30, 0x81, 0x30], "gb18030", BinaryPolicy::Lossy).unwrap(),
        "\u{10000}"
    );
}

#[test]
fn big5_aliases_decode_traditional_chinese() {
    for name in ["big5", "Big5", "big5-hkscs", "cp950"] {
        assert_eq!(
            decode_text(&ZHONG_WEN_BIG5, name, BinaryPolicy::Reject).unwrap(),
            "中文",
            "encoding name {name}"
        );
    }
    assert_eq!(
        decode_text(&LONG_MEN_BIG5, "big5", BinaryPolicy::Lossy).unwrap(),
        "龍門"
    );
}

#[test]
fn shift_jis_aliases_decode_japanese() {
    for name in ["shift-jis", "shift_jis", "sjis", "cp932", "Shift_JIS"] {
        assert_eq!(
            decode_text(&NIHONGO_SJIS, name, BinaryPolicy::Reject).unwrap(),
            "日本語",
            "encoding name {name}"
        );
    }
    // Halfwidth katakana is single byte: 0xA1..0xDF -> U+FF61..
    assert_eq!(
        decode_text(&[0xA1, 0xB1], "shift-jis", BinaryPolicy::Lossy).unwrap(),
        "\u{FF61}\u{FF71}"
    );
}

#[test]
fn invalid_and_truncated_sequences_become_replacement_characters() {
    // Truncated pair at the end of the input: one replacement for the lead byte.
    assert_eq!(
        decode_text(b"ascii then \xD6", "gbk", BinaryPolicy::Lossy).unwrap(),
        "ascii then \u{FFFD}"
    );
    // Invalid trail: the lead is replaced and the ASCII byte survives.
    assert_eq!(
        decode_text(&[0xD6, b' '], "gbk", BinaryPolicy::Lossy).unwrap(),
        "\u{FFFD} "
    );
    // Invalid trail in the middle resynchronises on the next valid pair.
    assert_eq!(
        decode_text(b"\xD6 x\xD6\xD0", "gbk", BinaryPolicy::Lossy).unwrap(),
        "\u{FFFD} x中"
    );
    // Big5 has 263 unmapped code positions; one lead between valid pairs.
    assert_eq!(
        decode_text(
            &[0xA4, 0xA4, 0xA1, 0x7F, 0xA4, 0xE5],
            "big5",
            BinaryPolicy::Lossy
        )
        .unwrap(),
        "中\u{FFFD}\u{7F}文"
    );
    // Truncated four-byte GB18030 sequence at the end: a single replacement.
    assert_eq!(
        decode_text(&[0x81, 0x30, 0x81], "gb18030", BinaryPolicy::Lossy).unwrap(),
        "\u{FFFD}"
    );
}

#[test]
fn auto_detection_reports_gb18030_for_simplified_chinese() {
    assert_eq!(detect_text_encoding(&ZHONG_WEN_GBK), TextEncoding::Gb18030);
    assert_eq!(detected_encoding_name(&ZHONG_WEN_GBK), "gb18030");
    assert_eq!(
        decode_text(&ZHONG_WEN_GBK, "auto", BinaryPolicy::Lossy).unwrap(),
        "中文"
    );
    // Ambiguous bytes decode under both codecs; GB18030 wins the tie (Python's order).
    assert!(decode_strict(&ZHONG_WEN_GBK, CjkEncoding::Big5).is_some());
    assert_eq!(
        decode_text(&ZHONG_WEN_GBK, "auto", BinaryPolicy::Lossy).unwrap(),
        decode_text(&ZHONG_WEN_GBK, "gb18030", BinaryPolicy::Lossy).unwrap()
    );
}

#[test]
fn auto_detection_reports_big5_for_big5_only_text() {
    // These bytes decode to private-use code points under GB18030, so Big5 wins.
    assert_eq!(detect_text_encoding(&LONG_MEN_BIG5), TextEncoding::Big5);
    assert_eq!(detected_encoding_name(&LONG_MEN_BIG5), "big5");
    assert_eq!(
        decode_text(&LONG_MEN_BIG5, "auto", BinaryPolicy::Lossy).unwrap(),
        "龍門"
    );
    // Even `中文` (which GB18030 decodes as kana) is reported as Big5.
    assert_eq!(detect_text_encoding(&ZHONG_WEN_BIG5), TextEncoding::Big5);
}

#[test]
fn auto_detection_does_not_guess_shift_jis() {
    // Shift-JIS bytes are valid GB18030, and Python's detector does not consider
    // Shift-JIS at all; the explicit name is the supported path.
    assert_eq!(detect_text_encoding(&NIHONGO_SJIS), TextEncoding::Gb18030);
    assert_ne!(detected_encoding_name(&NIHONGO_SJIS), "shift-jis");
    assert_eq!(
        decode_text(&NIHONGO_SJIS, "shift-jis", BinaryPolicy::Lossy).unwrap(),
        "日本語"
    );
}

#[test]
fn ascii_and_utf8_paths_are_unchanged() {
    assert_eq!(detect_text_encoding(b"plain ascii"), TextEncoding::Utf8);
    assert_eq!(detect_text_encoding("héllo".as_bytes()), TextEncoding::Utf8);
    assert_eq!(detect_text_encoding("中文".as_bytes()), TextEncoding::Utf8);
    assert_eq!(
        decode_text("中文".as_bytes(), "auto", BinaryPolicy::Lossy).unwrap(),
        "中文"
    );
    assert_eq!(
        decode_text("中文".as_bytes(), "utf-8", BinaryPolicy::Reject).unwrap(),
        "中文"
    );
    // A BOM still wins over CJK detection.
    let mut with_bom = vec![0xEF, 0xBB, 0xBF];
    with_bom.extend_from_slice("中文".as_bytes());
    assert_eq!(detect_text_encoding(&with_bom), TextEncoding::Utf8);
    assert_eq!(detected_encoding_name(&with_bom), "utf-8");
}

#[test]
fn legacy_cjk_text_is_not_binary() {
    assert!(!is_binary(&ZHONG_WEN_GBK));
    assert!(!is_binary(&ZHONG_WEN_BIG5));
    assert!(!is_binary(&NIHONGO_SJIS));
    assert!(is_binary(&[0x00, 0x01, 0x02]));
    // The reject policy accepts CJK content and keeps rejecting NUL-bearing input.
    assert!(decode_text(&ZHONG_WEN_BIG5, "big5", BinaryPolicy::Reject).is_ok());
    assert_eq!(
        decode_text(&[0x00, 0x01], "gb18030", BinaryPolicy::Reject)
            .unwrap_err()
            .kind()
            .clone(),
        ErrorKind::Validation
    );
}

#[test]
fn writing_legacy_encodings_is_unsupported() {
    for name in ["gbk", "gb18030", "big5", "shift-jis", "sjis"] {
        let error = encode_text("中文", name).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported, "encoding {name}");
        assert!(error.message().contains("not supported for writing"));
    }
    assert_eq!(encode_text("ascii", "utf-8").unwrap(), b"ascii");
}

#[test]
fn unknown_encoding_names_are_unsupported() {
    let error = decode_text(b"x", "koi8-r", BinaryPolicy::Lossy).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("koi8-r"));
    assert!(error.message().contains("gb18030"));
}

#[test]
fn canonical_names_round_trip_through_parse() {
    for encoding in [
        TextEncoding::Utf8,
        TextEncoding::Utf16Le,
        TextEncoding::Utf16Be,
        TextEncoding::Latin1,
        TextEncoding::Gb18030,
        TextEncoding::Big5,
        TextEncoding::ShiftJis,
    ] {
        assert_eq!(
            TextEncoding::parse(encoding.as_str()),
            Some(encoding),
            "canonical name {} must parse back",
            encoding.as_str()
        );
    }
    assert_eq!(TextEncoding::Gb18030.to_string(), "gb18030");
    assert_eq!(TextEncoding::Big5.to_string(), "big5");
    assert_eq!(TextEncoding::ShiftJis.to_string(), "shift-jis");
    assert_eq!(CjkEncoding::Gb18030.as_str(), "gb18030");
    assert_eq!(CjkEncoding::from_text_encoding(TextEncoding::Utf8), None);
}

#[test]
fn streaming_decoder_matches_a_single_pass() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GBK:");
    bytes.extend_from_slice(&ZHONG_WEN_GBK);
    bytes.extend_from_slice(b"!");
    bytes.extend_from_slice(&[0xD6]); // dangling lead byte at the very end

    let expected = decode(&bytes, CjkEncoding::Gb18030);
    assert!(expected.ends_with('\u{FFFD}'));
    for split in 0..=bytes.len() {
        let mut decoder = CjkDecoder::new(CjkEncoding::Gb18030);
        let mut text = decoder.push(&bytes[..split]);
        text.push_str(&decoder.push(&bytes[split..]));
        text.push_str(&decoder.finish());
        assert_eq!(text, expected, "chunk split at byte {split}");
    }

    let mut decoder = CjkDecoder::new(CjkEncoding::Gb18030);
    assert_eq!(decoder.push(b"GBK:"), "GBK:");
    assert_eq!(decoder.push(&ZHONG_WEN_GBK[..1]), "");
    assert_eq!(decoder.push(&ZHONG_WEN_GBK[1..3]), "中");
    assert_eq!(decoder.push(&ZHONG_WEN_GBK[3..]), "文");
    assert!(!decoder.invalid_input());
    assert_eq!(decoder.push(b"!"), "!");
    assert_eq!(decoder.finish(), "");
    assert!(!decoder.invalid_input());
    assert_eq!(decoder.encoding(), CjkEncoding::Gb18030);
}

#[test]
fn decoding_is_linear_for_repetitive_input() {
    let mut bytes = Vec::new();
    for _ in 0..20_000 {
        bytes.extend_from_slice(&ZHONG_WEN_GBK);
    }
    let text = decode(&bytes, CjkEncoding::Gb18030);
    assert_eq!(text.chars().count(), 40_000);
    assert!(text.starts_with("中文中文"));
    assert_eq!(detect_text_encoding(&bytes), TextEncoding::Gb18030);
}
