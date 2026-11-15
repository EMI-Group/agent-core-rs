//! Unit tests for the PDF content-stream reader.
//!
//! Kept beside the module (like the `inflate` tests) so `pdf.rs` stays within the
//! file-size budget; everything here is offline and deterministic.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

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
    assert!(!parsed.incomplete);
    assert!(parsed.skipped_filters.is_empty());

    let empty = PdfParser::new().parse_bytes(b"%PDF-1.4\n%%EOF\n").unwrap();
    assert_eq!(empty.text, "");
    assert_eq!(empty.pages, Vec::<String>::new());
    assert_eq!(empty.page_count, 0);
    assert!(!empty.has_text);
    assert!(empty.is_empty());
    assert!(!empty.incomplete);
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
fn stream_dictionaries_expose_their_filter_chain() {
    // No `/Filter` key at all.
    assert!(dictionary::parse_filter_chain(b"<< /Length 5 >>").is_empty());
    // A single long or abbreviated name.
    assert_eq!(
        dictionary::parse_filter_chain(b"<< /Filter /FlateDecode >>"),
        vec![filters::Filter::Flate]
    );
    assert_eq!(
        dictionary::parse_filter_chain(b"<< /Filter /Fl >>"),
        vec![filters::Filter::Flate]
    );
    // An array keeps the dictionary's decode order.
    assert_eq!(
        dictionary::parse_filter_chain(b"<< /Filter [/ASCII85Decode /ASCIIHexDecode] >>"),
        vec![filters::Filter::Ascii85, filters::Filter::AsciiHex]
    );
    // `/Filter` alone is not a filter name.
    assert!(dictionary::parse_filter_chain(b"<< /Filter >>").is_empty());
    // `/Filters` is a different key.
    assert!(dictionary::parse_filter_chain(b"<< /Filters 2 >>").is_empty());
    // Image payload and aborting filters keep their classification.
    let dct = dictionary::parse_filter_chain(b"<< /Filter /DCTDecode >>");
    assert_eq!(dct, vec![filters::Filter::Dct]);
    assert!(dct[0].is_image_payload());
    let lzw = dictionary::parse_filter_chain(b"<< /Filter /LZWDecode >>");
    assert_eq!(lzw, vec![filters::Filter::Lzw]);
    assert!(lzw[0].aborts_extraction());
    assert!(contains_token(b"/Fl ", b"/Fl"));
    assert!(!contains_token(b"/FlateDecode", b"/Fl"));
}

#[test]
fn decode_parms_are_read_from_the_stream_dictionary() {
    assert_eq!(
        dictionary::parse_decode_parms(b"<< /DecodeParms << /Predictor 12 /Columns 8 >> >>"),
        vec![Some(predictor::PredictorParams {
            predictor: 12,
            columns: 8,
            colors: 1,
            bits_per_component: 8,
        })]
    );
    assert_eq!(
        dictionary::parse_decode_parms(b"<< /DP [null << /Predictor 11 >>] >>"),
        vec![
            None,
            Some(predictor::PredictorParams {
                predictor: 11,
                ..predictor::PredictorParams::default()
            }),
        ]
    );
    assert!(dictionary::parse_decode_parms(b"<< /Length 5 >>").is_empty());
}

#[test]
fn encoded_streams_are_decoded_before_the_text_operators_are_read() {
    let hex = parse_pdf(&stream_object(
        "/Filter /ASCIIHexDecode",
        &ascii_hex_encode(b"BT (Hex decoded) Tj ET"),
    ));
    assert_eq!(hex.pages, vec!["Hex decoded"]);
    assert!(!hex.incomplete);
    assert!(hex.skipped_filters.is_empty());

    let ascii85 = parse_pdf(&stream_object(
        "/Filter /A85",
        &ascii85_encode(b"BT (A85 decoded) Tj ET"),
    ));
    assert_eq!(ascii85.pages, vec!["A85 decoded"]);
    assert!(!ascii85.incomplete);

    let run_length = parse_pdf(&stream_object(
        "/Filter /RunLengthDecode",
        &run_length_encode(b"BT (Run length decoded) Tj ET"),
    ));
    assert_eq!(run_length.pages, vec!["Run length decoded"]);
    assert!(!run_length.incomplete);
}

#[test]
fn filter_chains_are_decoded_in_dictionary_order() {
    // ASCII85 wraps the hex payload, so the array names ASCII85 first.
    let payload = ascii85_encode(&ascii_hex_encode(b"BT (chained) Tj ET"));
    let parsed = parse_pdf(&stream_object(
        "/Filter [/ASCII85Decode /ASCIIHexDecode]",
        &payload,
    ));
    assert_eq!(parsed.pages, vec!["chained"]);
    assert!(!parsed.incomplete);

    let payload = ascii85_encode(&run_length_encode(b"BT (run then a85) Tj ET"));
    let parsed = parse_pdf(&stream_object("/Filter [/A85 /RL]", &payload));
    assert_eq!(parsed.pages, vec!["run then a85"]);
    assert!(!parsed.incomplete);
}

#[test]
fn predictors_are_undone_after_their_filter() {
    // Four-byte PNG `Up` rows over the padded content stream.
    let rows = png_up_encode(b"BT (pred) Tj ET ", 4);
    let payload = ascii_hex_encode(&rows);
    let parsed = parse_pdf(&stream_object(
        "/Filter /ASCIIHexDecode /DecodeParms << /Predictor 12 /Columns 4 >>",
        &payload,
    ));
    assert_eq!(parsed.pages, vec!["pred"]);
    assert!(!parsed.incomplete);

    // A `null` entry before the predictor dictionary keeps its default.
    let parsed = parse_pdf(&stream_object(
        "/Filter [/A85 /ASCIIHexDecode] /DecodeParms [null << /Predictor 11 /Columns 4 >>]",
        &ascii85_encode(&ascii_hex_encode(&png_up_encode(
            b"BT (null safe) Tj ET",
            4,
        ))),
    ));
    assert_eq!(parsed.pages, vec!["null safe"]);
    assert!(!parsed.incomplete);
}

#[test]
fn skipped_streams_are_recorded_instead_of_disappearing() {
    let mut body = stream_object(
        "/Filter /ASCIIHexDecode",
        &ascii_hex_encode(b"BT (kept) Tj ET"),
    );
    body.extend_from_slice(&stream_object("/Filter /DCTDecode", b"\xff\xd8\xff\xd9"));
    body.extend_from_slice(&stream_object("/Filter /FlateDecode", b"not a zlib stream"));
    body.extend_from_slice(&stream_object("/Filter /FooDecode", b"unknown payload"));
    let parsed = parse_pdf(&body);

    assert_eq!(parsed.pages, vec!["kept"]);
    assert!(parsed.incomplete);
    assert_eq!(
        parsed.skipped_filters,
        vec![
            "DCTDecode".to_owned(),
            "FlateDecode".to_owned(),
            "FooDecode".to_owned(),
        ]
    );
    assert_eq!(parsed.skipped_streams.len(), 3);
    assert!(parsed.skipped_streams[0].starts_with("stream 1: image payload filter DCTDecode"));
    assert!(parsed.skipped_streams[1].contains("FlateDecode decode failed"));
    assert!(parsed.skipped_streams[2].contains("FooDecode"));
}

#[test]
fn document_markers_default_to_a_complete_document() {
    let parsed = parse_pdf(&stream_object("", b"BT (plain) Tj ET"));
    assert_eq!(parsed.pages, vec!["plain"]);
    assert!(!parsed.incomplete);
    assert!(parsed.skipped_filters.is_empty());
    assert!(parsed.skipped_streams.is_empty());
    let default = PdfDocument::default();
    assert!(!default.incomplete);
    assert!(default.skipped_filters.is_empty());
    assert!(default.skipped_streams.is_empty());
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

/// Wraps `body` in a minimal PDF file.
fn pdf_document(body: &[u8]) -> Vec<u8> {
    let mut document = b"%PDF-1.7\n".to_vec();
    document.extend_from_slice(body);
    document.extend_from_slice(b"\n%%EOF\n");
    document
}

/// Builds one stream object from extra dictionary entries and a payload.
fn stream_object(dictionary_extra: &str, payload: &[u8]) -> Vec<u8> {
    let mut object = format!(
        "1 0 obj\n<< {dictionary_extra} /Length {} >>\nstream\n",
        payload.len()
    )
    .into_bytes();
    object.extend_from_slice(payload);
    object.extend_from_slice(b"\nendstream\nendobj\n");
    object
}

/// Parses a PDF built from `body`.
fn parse_pdf(body: &[u8]) -> PdfDocument {
    PdfParser::new().parse_bytes(&pdf_document(body)).unwrap()
}

/// Hex-encodes `data` with the `/ASCIIHexDecode` terminator.
fn ascii_hex_encode(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = data
        .iter()
        .flat_map(|byte| format!("{byte:02X}").into_bytes())
        .collect();
    out.push(b'>');
    out
}

/// Base-85 encodes `data` the way `/ASCII85Decode` expects.
fn ascii85_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in data.chunks(4) {
        let mut group = [0u8; 4];
        group[..chunk.len()].copy_from_slice(chunk);
        let value = u32::from_be_bytes(group);
        if chunk.len() == 4 && value == 0 {
            out.push(b'z');
            continue;
        }
        let mut digits = [0u8; 5];
        let mut remaining = value;
        for slot in digits.iter_mut().rev() {
            *slot = (remaining % 85) as u8;
            remaining /= 85;
        }
        let count = if chunk.len() == 4 { 5 } else { chunk.len() + 1 };
        out.extend(digits.iter().take(count).map(|digit| b'!' + digit));
    }
    out.extend_from_slice(b"~>");
    out
}

/// Run-length encodes `data` as literal runs plus one repeat run.
fn run_length_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in data.chunks(128) {
        out.push((chunk.len() - 1) as u8);
        out.extend_from_slice(chunk);
    }
    // A repeat run of two spaces, then the end-of-data marker.
    out.push(255);
    out.push(b' ');
    out.push(128);
    out
}

/// PNG `Up` encodes `data` as rows of `columns` bytes with a filter-type byte.
fn png_up_encode(data: &[u8], columns: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut previous = vec![0u8; columns];
    for row in data.chunks(columns) {
        let mut raw = previous.clone();
        raw[..row.len()].copy_from_slice(row);
        out.push(2);
        for index in 0..row.len() {
            out.push(raw[index].wrapping_sub(previous[index]));
        }
        previous = raw;
    }
    out
}
