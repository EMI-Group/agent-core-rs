//! Declared write encodings end to end: every supported encoding encodes to real codec
//! bytes and decodes back to the same text.
//!
//! Python parity: `write_file` encodes text content with `content.encode(params.encoding)`
//! and `read_file` decodes it with the same name, so a declared encoding must round-trip.
//! Every byte expectation here was produced with (and verified against) Python's `gb18030`,
//! `big5`, `shift_jis` and `latin-1` codecs; the expectations are hardcoded so the suite
//! stays offline and needs no `python3`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_sysop::encoding::{decode_text, encode_text, TextEncoding};
use openjiuwen_sysop::BinaryPolicy;

/// `中文` in GBK/GB18030.
const ZHONG_WEN_GBK: [u8; 4] = [0xD6, 0xD0, 0xCE, 0xC4];
/// `中文` in Big5.
const ZHONG_WEN_BIG5: [u8; 4] = [0xA4, 0xA4, 0xA4, 0xE5];
/// `日本語` in Shift-JIS.
const NIHONGO_SJIS: [u8; 6] = [0x93, 0xFA, 0x96, 0x7B, 0x8C, 0xEA];

/// Encodes `text` with `encoding` and decodes the bytes back with the same name.
fn round_trip(text: &str, encoding: &str) {
    let bytes = encode_text(text, encoding)
        .unwrap_or_else(|error| panic!("{encoding} could not encode {text:?}: {error}"));
    let decoded = decode_text(&bytes, encoding, BinaryPolicy::Lossy)
        .unwrap_or_else(|error| panic!("{encoding} could not decode its own bytes: {error}"));
    assert_eq!(decoded, text, "encoding {encoding}");
}

/// Text that must survive a write/read cycle in every encoding.
fn universal_corpus() -> Vec<&'static str> {
    vec![
        "",
        "plain ascii text",
        "line one\nline two\n",
        "tabs\tand\r\ncrlf\n",
        "trailing newline\n",
        "punctuation: !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    ]
}

#[test]
fn every_supported_encoding_writes_and_reads_back() {
    for encoding in [
        "utf-8",
        "utf-16le",
        "utf-16be",
        "latin-1",
        "gb18030",
        "gbk",
        "big5",
        "shift-jis",
    ] {
        for text in universal_corpus() {
            round_trip(text, encoding);
        }
    }
}

#[test]
fn utf_encodings_keep_unicode_and_newlines() {
    for encoding in ["utf-8", "utf-16le", "utf-16be"] {
        for text in [
            "café ünïcode",
            "中文 日本語 한국어",
            "mixed ascii 中文\nsecond line\n",
            "\u{1F600} emoji and \u{20000} supplementary",
        ] {
            round_trip(text, encoding);
        }
    }
    assert_eq!(encode_text("é", "utf-8").unwrap(), vec![0xC3, 0xA9]);
    assert_eq!(encode_text("é", "utf-16le").unwrap(), vec![0xE9, 0x00]);
    assert_eq!(encode_text("é", "utf-16be").unwrap(), vec![0x00, 0xE9]);
}

#[test]
fn latin1_writes_one_byte_per_character() {
    assert_eq!(encode_text("é", "latin-1").unwrap(), vec![0xE9]);
    assert_eq!(encode_text("café", "latin-1").unwrap(), b"caf\xE9".to_vec());
    assert_eq!(encode_text("\u{FF}", "latin-1").unwrap(), vec![0xFF]);
    assert_eq!(encode_text("\u{0}", "latin-1").unwrap(), vec![0x00]);
    for text in [
        "café",
        "Über",
        "plain ascii",
        "mixed ascii café\n",
        "élève\n",
    ] {
        round_trip(text, "latin-1");
    }
    // Anything above U+00FF is a typed failure, not a UTF-8 fallback.
    let error = encode_text("中文", "latin-1").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("latin-1"), "{}", error.message());
    assert!(error.message().contains("U+4E2D"), "{}", error.message());
    let error = encode_text("\u{20AC}", "latin-1").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("U+20AC"), "{}", error.message());
}

#[test]
fn gb18030_writes_gbk_bytes_for_the_common_area() {
    for name in ["gb18030", "gbk", "gb2312", "cp936", "GB18030"] {
        assert_eq!(
            encode_text("中文", name).unwrap(),
            ZHONG_WEN_GBK.to_vec(),
            "encoding {name}"
        );
    }
    for text in [
        "中文",
        "简体中文文本",
        "GBK: 中文 and ascii",
        "第一行\n第二行\n",
        "全角标点：，。！？",
    ] {
        round_trip(text, "gb18030");
    }
    // Characters outside the two-byte area use the four-byte form, including the
    // supplementary planes, and still round-trip.
    assert_eq!(
        encode_text("\u{FFFD}", "gb18030").unwrap(),
        vec![0x84, 0x31, 0xA4, 0x37]
    );
    assert_eq!(
        encode_text("\u{20000}", "gb18030").unwrap(),
        vec![0x95, 0x32, 0x82, 0x36]
    );
    assert_eq!(
        encode_text("\u{1F600}", "gb18030").unwrap(),
        vec![0x94, 0x39, 0xFC, 0x36]
    );
    for text in ["𠀀", "\u{1F600}", "emoji 😀 and 中文\n", "\u{10FFFF}"] {
        round_trip(text, "gb18030");
    }
}

#[test]
fn big5_writes_traditional_bytes() {
    for name in ["big5", "big5-hkscs", "cp950"] {
        assert_eq!(
            encode_text("中文", name).unwrap(),
            ZHONG_WEN_BIG5.to_vec(),
            "encoding {name}"
        );
    }
    assert_eq!(
        encode_text("龍門", "big5").unwrap(),
        vec![0xC0, 0x73, 0xAA, 0xF9]
    );
    for text in [
        "中文",
        "繁體中文",
        "Big5: 中文 and ascii",
        "第一行\n第二行\n",
    ] {
        round_trip(text, "big5");
    }
    let error = encode_text("\u{1F600}", "big5").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("big5"), "{}", error.message());
    assert!(error.message().contains("U+1F600"), "{}", error.message());
}

#[test]
fn shift_jis_writes_japanese_bytes_and_halfwidth_katakana() {
    for name in ["shift-jis", "shift_jis", "sjis", "cp932"] {
        assert_eq!(
            encode_text("日本語", name).unwrap(),
            NIHONGO_SJIS.to_vec(),
            "encoding {name}"
        );
    }
    for text in [
        "日本語",
        "ひらがなカタカナ",
        "mixed ascii 日本語\n",
        "\u{FF61}\u{FF71}\u{FF9F}",
    ] {
        round_trip(text, "shift-jis");
    }
    // Halfwidth katakana is positional: U+FF61 -> 0xA1 .. U+FF9F -> 0xDF.
    assert_eq!(
        encode_text("\u{FF61}\u{FF71}", "shift-jis").unwrap(),
        vec![0xA1, 0xB1]
    );
    let error = encode_text("\u{1F600}", "shift-jis").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("shift-jis"), "{}", error.message());
}

#[test]
fn encoding_produces_the_declared_codec_not_utf8() {
    // GB18030 `中文` is four bytes, while UTF-8 would be six.
    assert_eq!(encode_text("中文", "gb18030").unwrap().len(), 4);
    assert_eq!(encode_text("中文", "utf-8").unwrap().len(), 6);
    assert_ne!(
        encode_text("中文", "big5").unwrap(),
        encode_text("中文", "gb18030").unwrap()
    );
    assert_ne!(
        encode_text("é", "latin-1").unwrap(),
        encode_text("é", "utf-8").unwrap()
    );
}

#[test]
fn unknown_and_auto_encodings_are_configuration_errors() {
    let error = encode_text("x", "koi8-r").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config, "{}", error.message());
    assert!(error.message().contains("koi8-r"), "{}", error.message());
    assert!(
        error.message().contains("supported encodings are"),
        "{}",
        error.message()
    );
    for name in [
        "utf-8",
        "utf-16le",
        "utf-16be",
        "latin-1",
        "gb18030",
        "big5",
        "shift-jis",
    ] {
        assert!(
            error.message().contains(name),
            "the message must list {name}: {}",
            error.message()
        );
    }
    for name in ["auto", "AUTO", " Auto "] {
        let error = encode_text("x", name).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config, "{name}");
        assert!(
            error.message().contains("cannot be used for writing"),
            "{name}: {}",
            error.message()
        );
    }
    // The failure kind is never `Unsupported`: every parsed encoding really writes.
    for encoding in [
        TextEncoding::Utf8,
        TextEncoding::Utf16Le,
        TextEncoding::Utf16Be,
        TextEncoding::Latin1,
        TextEncoding::Gb18030,
        TextEncoding::Big5,
        TextEncoding::ShiftJis,
    ] {
        assert!(
            encode_text("plain ascii", encoding.as_str()).is_ok(),
            "{encoding} must write"
        );
    }
}

#[test]
fn write_encodings_match_their_read_encodings() {
    // Text written as one encoding is not readable as another codec's text.
    let gb = encode_text("中文", "gb18030").unwrap();
    assert_eq!(
        decode_text(&gb, "gb18030", BinaryPolicy::Lossy).unwrap(),
        "中文"
    );
    assert_ne!(
        decode_text(&gb, "utf-8", BinaryPolicy::Lossy).unwrap(),
        "中文"
    );
    let sjis = encode_text("日本語", "shift-jis").unwrap();
    assert_eq!(
        decode_text(&sjis, "shift-jis", BinaryPolicy::Lossy).unwrap(),
        "日本語"
    );
}
