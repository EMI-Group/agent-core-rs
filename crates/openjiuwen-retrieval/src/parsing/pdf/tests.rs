//! Unit tests for the PDF content-stream reader.
//!
//! Kept beside the module (like the `inflate` tests) so `pdf.rs` stays within the
//! file-size budget; everything here is offline and deterministic.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;

use super::*;

#[test]
fn reads_simple_tj_operators() {
    assert_eq!(extract_text(b"BT (Hello) Tj ET"), "Hello");
    // Consecutive `Tj` calls stay on the same line: only a positioning
    // operator starts a new one.
    assert_eq!(
        extract_text(b"BT /F1 12 Tf 72 720 Td (Two) Tj (lines) Tj ET"),
        "Twolines"
    );
    assert_eq!(
        extract_text(b"BT /F1 12 Tf 72 720 Td (Two) Tj 0 -14 Td (lines) Tj ET"),
        "Two\nlines"
    );
}

#[test]
fn reads_tj_arrays_and_kerning_spaces() {
    assert_eq!(extract_text(b"BT [(Hel) (lo)] TJ ET"), "Hello");
    assert_eq!(extract_text(b"BT [(Hel) -250 (lo)] TJ ET"), "Hel lo");
    // Small adjustments do not add spaces.
    assert_eq!(extract_text(b"BT [(Hel) -20 (lo)] TJ ET"), "Hello");
    assert_eq!(extract_text(b"BT [] TJ (after) Tj ET"), "after");
}

#[test]
fn quote_operators_break_the_line() {
    assert_eq!(extract_text(b"BT (one) Tj (two) ' ET"), "one\ntwo");
    assert_eq!(extract_text(b"BT (one) Tj -2 3 (two) \" ET"), "one\ntwo");
    assert_eq!(extract_text(b"BT (one) Tj T* (two) Tj ET"), "one\ntwo");
}

#[test]
fn literal_strings_decode_escapes_and_nesting() {
    assert_eq!(literal_text("line\\nnext"), "line\nnext");
    assert_eq!(literal_text("a\\(b\\)c"), "a(b)c");
    assert_eq!(literal_text("back\\\\slash"), "back\\slash");
    assert_eq!(literal_text("\\101\\102"), "AB");
    assert_eq!(literal_text("\\1011"), "A1");
    assert_eq!(literal_text("\\q"), "q");
    assert_eq!(literal_text("paren (nested) end"), "paren (nested) end");
    // A line continuation joins the halves without a newline.
    assert_eq!(literal_text("joined\\\nnext"), "joinednext");
    assert_eq!(literal_text("joined\\\r\nnext"), "joinednext");
    assert_eq!(literal_text("bell\\bform\\f"), "bell\x08form\x0c");
    assert_eq!(
        extract_text(b"BT (paren (nested) end) Tj ET"),
        "paren (nested) end"
    );
}

#[test]
fn hex_strings_decode_latin1_and_utf16() {
    assert_eq!(decode_string(&read_hex_string("48656C6C6F")), "Hello");
    assert_eq!(decode_string(&read_hex_string("48 65 6C6C 6F")), "Hello");
    // An odd digit count completes the last byte with 0.
    assert_eq!(decode_string(&read_hex_string("41424")), "AB@");
    // A byte-order mark switches to UTF-16BE.
    assert_eq!(decode_string(&read_hex_string("FEFF00480069")), "Hi");
    assert_eq!(extract_text(b"BT <48656C6C6F> Tj ET"), "Hello");
    // Above ASCII: a bare UTF-8 stream keeps its characters, Latin-1 bytes map
    // one-to-one.
    assert_eq!(decode_string("é".as_bytes()), "é");
    assert_eq!(decode_string(&[0xE9]), "é");
}

#[test]
fn pages_are_ordered_and_text_free_documents_are_not_errors() {
    let document = b"%PDF-1.4\n\
        << /Length 12 >>\nstream\nBT (one) Tj ET\nendstream\n\
        << /Length 12 >>\nstream\nBT (two) Tj ET\nendstream\n\
        << /Length 0 >>\nstream\n\nendstream\n%%EOF";
    let parsed = PdfParser::new().parse_bytes(document).unwrap();
    assert_eq!(parsed.pages, vec!["one", "two", ""]);
    assert_eq!(parsed.page_count, 3);
    assert_eq!(parsed.text, "one\ntwo");
    assert!(parsed.has_text);
    assert!(!parsed.is_empty());

    let empty = PdfParser::new().parse_bytes(b"%PDF-1.4\n%%EOF\n").unwrap();
    assert_eq!(empty.text, "");
    assert_eq!(empty.pages, Vec::<String>::new());
    assert_eq!(empty.page_count, 0);
    assert!(!empty.has_text);
    assert!(empty.is_empty());
}

#[test]
fn encrypted_documents_and_object_streams_are_unsupported() {
    let error = PdfParser::new()
        .parse_bytes(b"%PDF-1.7\ntrailer << /Encrypt 5 0 R >>\n%%EOF")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("encrypted"), "{error}");

    let error = PdfParser::new()
        .parse_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /ObjStm /N 2 >>\nstream\nendstream\nendobj")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ObjStm"), "{error}");
}

#[test]
fn filters_are_classified_from_the_stream_dictionary() {
    assert!(matches!(
        classify_stream(b"<< /Length 5 >>"),
        StreamEncoding::Plain
    ));
    assert!(matches!(
        classify_stream(b"<< /Filter /FlateDecode >>"),
        StreamEncoding::Flate
    ));
    assert!(matches!(
        classify_stream(b"<< /Filter /Fl >>"),
        StreamEncoding::Flate
    ));
    assert!(matches!(
        classify_stream(b"<< /Filter [/FlateDecode] >>"),
        StreamEncoding::Flate
    ));
    assert!(matches!(
        classify_stream(b"<< /Filter /DCTDecode >>"),
        StreamEncoding::Unsupported("DCTDecode")
    ));
    assert!(matches!(
        classify_stream(b"<< /Filter /ASCIIHexDecode >>"),
        StreamEncoding::Unsupported("ASCIIHexDecode")
    ));
    assert!(matches!(
        classify_stream(b"<< /Filter [/FlateDecode /LZWDecode] >>"),
        StreamEncoding::Unsupported("LZWDecode")
    ));
    // `/Filter` alone is not a filter name.
    assert!(matches!(
        classify_stream(b"<< /Filter >>"),
        StreamEncoding::Plain
    ));
    assert!(contains_token(b"/Fl ", b"/Fl"));
    assert!(!contains_token(b"/FlateDecode", b"/Fl"));
}

#[test]
fn stream_keywords_must_be_delimited_by_an_end_of_line() {
    assert_eq!(stream_data_start(b"stream\nabc", 0), Some(7));
    assert_eq!(stream_data_start(b"stream\r\nabc", 0), Some(8));
    assert_eq!(stream_data_start(b"stream\rabc", 0), Some(7));
    assert_eq!(stream_data_start(b"stream abc", 0), None);
    // "endstream" must not be read as a stream keyword.
    assert_eq!(stream_data_start(b"xendstream\nabc", 1), None);
    assert_eq!(trim_stream_end(b"abc\r\n", 0, 5), 3);
}

/// Reads a hex string body from its digits (test convenience).
fn read_hex_string(digits: &str) -> Vec<u8> {
    let input = format!("<{digits}>");
    let mut lexer = Lexer::new(input.as_bytes());
    lexer.pos = 1;
    lexer.read_hex()
}

/// Decodes one literal string body, escapes and nesting included.
fn literal_text(body: &str) -> String {
    let input = format!("({body})");
    let mut lexer = Lexer::new(input.as_bytes());
    lexer.pos = 1;
    decode_string(&lexer.read_literal())
}
