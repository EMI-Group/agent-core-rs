//! Integration tests for the `PDF` stream filter pipeline.
//!
//! Every fixture is built in this file: the document body, each stream object, the
//! `zlib` wrapper (`zlib_stored`: a `zlib` header, stored `DEFLATE` blocks, and
//! the `Adler-32` trailer), and the `ASCIIHex`/`ASCII85`/`RunLength` encoders. No
//! compression crate, no fixture files, no network.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::cast_possible_truncation
)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::parsing::{adler32, PdfDocument, PdfParser};

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Wraps `body` in a minimal `PDF` file.
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

/// Wraps `data` in a `zlib` stream made of stored `DEFLATE` blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut stream = vec![0x78, 0x01];
    let mut chunks = data.chunks(65_535).peekable();
    if chunks.peek().is_none() {
        stream.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    } else {
        while let Some(chunk) = chunks.next() {
            let last = chunks.peek().is_none();
            stream.push(u8::from(last));
            let length = chunk.len() as u16;
            stream.extend_from_slice(&length.to_le_bytes());
            stream.extend_from_slice(&(!length).to_le_bytes());
            stream.extend_from_slice(chunk);
        }
    }
    stream.extend_from_slice(&adler32(data).to_be_bytes());
    stream
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
///
/// Full zero groups become `z`, and a final partial group keeps `len + 1`
/// characters, exactly as the specification allows.
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

/// Run-length encodes `data` as literal runs plus two repeated spaces.
fn run_length_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in data.chunks(128) {
        out.push((chunk.len() - 1) as u8);
        out.extend_from_slice(chunk);
    }
    // A repeat run (`257 - 255 == 2`) followed by the end-of-data marker.
    out.push(255);
    out.push(b' ');
    out.push(128);
    out
}

/// PNG `Up` encodes `data` as rows of `columns` bytes, each behind filter type 2.
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

/// Parses a document built from `body`.
fn parse(body: &[u8]) -> PdfDocument {
    PdfParser::new().parse_bytes(&pdf_document(body)).unwrap()
}

// ---------------------------------------------------------------------------
// Decodable filters
// ---------------------------------------------------------------------------

#[test]
fn ascii_hex_streams_are_decoded() {
    let parsed = parse(&stream_object(
        "/Filter /ASCIIHexDecode",
        &ascii_hex_encode(b"BT (Hex text) Tj ET"),
    ));
    assert_eq!(parsed.pages, vec!["Hex text"]);
    assert_eq!(parsed.text, "Hex text");
    assert!(parsed.has_text);
    assert!(!parsed.incomplete);
    assert!(parsed.skipped_filters.is_empty());
    assert!(parsed.skipped_streams.is_empty());
}

#[test]
fn ascii85_streams_are_decoded() {
    // Pad the content so the encoder emits a `z` group and a partial final group.
    let mut content = b"BT (A85 zero) Tj ET".to_vec();
    while content.len() % 4 != 0 {
        content.push(b' ');
    }
    content.extend_from_slice(&[0, 0, 0, 0]);
    content.extend_from_slice(b"  ");
    let payload = ascii85_encode(&content);
    assert!(
        payload.contains(&b'z'),
        "the fixture must exercise the `z` shortcut"
    );
    assert!(payload.ends_with(b"~>"));

    let parsed = parse(&stream_object("/Filter /ASCII85Decode", &payload));
    assert_eq!(parsed.pages, vec!["A85 zero"]);
    assert!(!parsed.incomplete);
}

#[test]
fn run_length_streams_are_decoded() {
    let payload = run_length_encode(b"BT [(Run) -250 (Length)] TJ ET");
    let parsed = parse(&stream_object("/Filter /RunLengthDecode", &payload));
    assert_eq!(parsed.pages, vec!["Run Length"]);
    assert!(!parsed.incomplete);
}

#[test]
fn filter_chains_are_decoded_in_dictionary_order() {
    // `ASCII85Decode` runs first, so it wraps the hex payload.
    let payload = ascii85_encode(&ascii_hex_encode(b"BT (Chained) Tj ET"));
    let parsed = parse(&stream_object(
        "/Filter [/ASCII85Decode /ASCIIHexDecode]",
        &payload,
    ));
    assert_eq!(parsed.pages, vec!["Chained"]);
    assert!(!parsed.incomplete);

    let payload = ascii85_encode(&run_length_encode(b"BT (A85 then run) Tj ET"));
    let parsed = parse(&stream_object("/Filter [/A85 /RL]", &payload));
    assert_eq!(parsed.pages, vec!["A85 then run"]);
    assert!(!parsed.incomplete);
}

// ---------------------------------------------------------------------------
// Predictors
// ---------------------------------------------------------------------------

#[test]
fn flate_with_a_png_predictor_is_decoded() {
    let mut content = b"BT (Predictor) Tj ET".to_vec();
    while content.len() % 8 != 0 {
        content.push(b' ');
    }
    assert_eq!(content.len(), 24);
    let payload = zlib_stored(&png_up_encode(&content, 8));

    let parsed = parse(&stream_object(
        "/Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 8 >>",
        &payload,
    ));
    assert_eq!(parsed.pages, vec!["Predictor"]);
    assert_eq!(parsed.text, "Predictor");
    assert!(!parsed.incomplete);
}

#[test]
fn decode_parms_arrays_align_with_the_filter_chain() {
    let mut content = b"BT (Parm array) Tj ET".to_vec();
    while content.len() % 8 != 0 {
        content.push(b' ');
    }
    let encoded = png_up_encode(&content, 8);
    let payload = ascii_hex_encode(&zlib_stored(&encoded));
    let parsed = parse(&stream_object(
        "/Filter [/ASCIIHexDecode /FlateDecode] \
         /DecodeParms [null << /Predictor 11 /Columns 8 >>]",
        &payload,
    ));
    assert_eq!(parsed.pages, vec!["Parm array"]);
    assert!(!parsed.incomplete);

    // A predictor this crate cannot apply is a document-level error.
    let error = PdfParser::new()
        .parse_bytes(&pdf_document(&stream_object(
            "/Filter /ASCIIHexDecode /DecodeParms << /Predictor 3 >>",
            &ascii_hex_encode(b"BT (nope) Tj ET"),
        )))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("predictor 3"), "{error}");
}

// ---------------------------------------------------------------------------
// Undecodable and skipped streams
// ---------------------------------------------------------------------------

#[test]
fn lzw_streams_abort_extraction() {
    for name in ["/LZWDecode", "/LZW"] {
        let error = PdfParser::new()
            .parse_bytes(&pdf_document(&stream_object(
                &format!("/Filter {name}"),
                b"payload",
            )))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("LZWDecode"), "{error}");
    }
}

#[test]
fn image_payload_streams_are_recorded() {
    let mut body = stream_object("", b"BT (Readable) Tj ET");
    body.extend_from_slice(&stream_object("/Filter /DCTDecode", b"\xff\xd8\xff\xd9"));
    let parsed = parse(&body);

    assert_eq!(parsed.pages, vec!["Readable"]);
    assert_eq!(parsed.text, "Readable");
    assert!(parsed.incomplete);
    assert_eq!(parsed.skipped_filters, vec!["DCTDecode"]);
    assert_eq!(parsed.skipped_streams.len(), 1);
    assert!(
        parsed.skipped_streams[0].contains("stream 1: image payload filter DCTDecode"),
        "unexpected record: {}",
        parsed.skipped_streams[0]
    );
}

#[test]
fn undecodable_flate_streams_are_recorded() {
    let mut body = stream_object("", b"BT (Still readable) Tj ET");
    body.extend_from_slice(&stream_object("/Filter /FlateDecode", b"garbage bytes"));
    let parsed = parse(&body);

    assert_eq!(parsed.pages, vec!["Still readable"]);
    assert!(parsed.incomplete);
    assert_eq!(parsed.skipped_filters, vec!["FlateDecode"]);
    assert_eq!(parsed.skipped_streams.len(), 1);
    assert!(
        parsed.skipped_streams[0].contains("FlateDecode decode failed"),
        "unexpected record: {}",
        parsed.skipped_streams[0]
    );
}

#[test]
fn unknown_filters_are_recorded() {
    let mut body = stream_object("", b"BT (Kept) Tj ET");
    body.extend_from_slice(&stream_object("/Filter /FooDecode", b"opaque"));
    let parsed = parse(&body);

    assert_eq!(parsed.pages, vec!["Kept"]);
    assert!(parsed.incomplete);
    assert_eq!(parsed.skipped_filters, vec!["FooDecode"]);
    assert!(
        parsed.skipped_streams[0].contains("FooDecode"),
        "unexpected record: {}",
        parsed.skipped_streams[0]
    );
}

#[test]
fn skipped_streams_are_recorded_once_per_filter() {
    let mut body = stream_object("/Filter /DCTDecode", b"\xff\xd8\xff\xd9");
    body.extend_from_slice(&stream_object("/Filter /DCTDecode", b"\xff\xd8\xff\xd9"));
    let parsed = parse(&body);

    assert!(parsed.pages.is_empty());
    assert!(parsed.incomplete);
    assert_eq!(parsed.skipped_filters, vec!["DCTDecode"]);
    assert_eq!(parsed.skipped_streams.len(), 2);
}

// ---------------------------------------------------------------------------
// Document markers
// ---------------------------------------------------------------------------

#[test]
fn completeness_markers_serialise() {
    let mut body = stream_object("", b"BT (Text) Tj ET");
    body.extend_from_slice(&stream_object("/Filter /DCTDecode", b"\xff\xd8\xff\xd9"));
    let parsed = parse(&body);

    let value = serde_json::to_value(&parsed).unwrap();
    assert_eq!(value["incomplete"], serde_json::Value::Bool(true));
    assert_eq!(value["skipped_filters"], serde_json::json!(["DCTDecode"]));
    assert_eq!(value["skipped_streams"].as_array().unwrap().len(), 1);
    assert_eq!(value["pages"], serde_json::json!(["Text"]));

    // A complete document round-trips with empty records.
    let complete = parse(&stream_object("", b"BT (Text) Tj ET"));
    let value = serde_json::to_value(&complete).unwrap();
    assert_eq!(value["incomplete"], serde_json::Value::Bool(false));
    assert_eq!(value["skipped_filters"], serde_json::json!([]));
    assert_eq!(value["skipped_streams"], serde_json::json!([]));
}

#[test]
fn documents_without_skipped_streams_stay_complete() {
    let parsed = parse(
        b"1 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n\
          2 0 obj\n<< /Type /Catalog >>\nendobj",
    );
    assert_eq!(parsed.pages, vec![""]);
    assert!(!parsed.has_text);
    assert!(parsed.is_empty());
    assert!(!parsed.incomplete);
    assert!(parsed.skipped_filters.is_empty());
    assert!(parsed.skipped_streams.is_empty());

    let empty = parse(b"2 0 obj\n<< /Type /Catalog >>\nendobj");
    assert!(empty.pages.is_empty());
    assert!(!empty.incomplete);
}
