//! Hand-written byte fixtures for the `Content-Length` frame encoder and decoder.
//!
//! These tests deliberately split headers and bodies across `feed` calls and cover malformed
//! frames, oversized bodies, and multi-byte UTF-8 bodies whose length is measured in bytes.
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

use crate::lsp::protocol::{encode_message, FrameDecoder, MessageReader, MAX_CONTENT_LENGTH};

/// Builds one raw frame for `body`.
fn frame(body: &str) -> Vec<u8> {
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

#[test]
fn encode_message_writes_byte_length_header_and_body() {
    let message = json!({"jsonrpc": "2.0", "id": 1, "method": "ping"});
    let bytes = encode_message(&message).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    let (header, body) = text.split_once("\r\n\r\n").unwrap();
    assert_eq!(body, serde_json::to_string(&message).unwrap());
    assert_eq!(header, format!("Content-Length: {}", body.len()));
    assert_eq!(serde_json::from_str::<Value>(body).unwrap(), message);
}

#[test]
fn encode_message_counts_utf8_bytes_not_characters() {
    let message = json!({"text": "héllo 世界"});
    let bytes = encode_message(&message).unwrap();
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap();
    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
    let declared: usize = header
        .strip_prefix("Content-Length: ")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(bytes.len() - header_end - 4, declared);
}

#[test]
fn decoder_accepts_a_frame_split_across_every_boundary() {
    let raw = frame(r#"{"jsonrpc":"2.0","id":7,"result":null}"#);
    let mut decoder = FrameDecoder::new();
    for byte in &raw {
        decoder.feed(std::slice::from_ref(byte));
        let decoded = decoder.try_next().unwrap();
        if decoder.is_empty() {
            assert_eq!(decoded.unwrap()["id"], 7);
        } else {
            assert!(decoded.is_none());
        }
    }
    assert!(decoder.finish().unwrap().is_none());
}

#[test]
fn decoder_tolerates_extra_headers_and_case_insensitive_names() {
    let body = r#"{"jsonrpc":"2.0","method":"note"}"#;
    let raw = format!(
        "content-length: {}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n{body}",
        body.len()
    );
    let mut decoder = FrameDecoder::new();
    decoder.feed(raw.as_bytes());
    let decoded = decoder.try_next().unwrap().unwrap();
    assert_eq!(decoded["method"], "note");
}

#[test]
fn decoder_returns_all_frames_from_one_feed() {
    let mut raw = frame(r#"{"jsonrpc":"2.0","id":1,"result":1}"#);
    raw.extend(frame(r#"{"jsonrpc":"2.0","id":2,"result":2}"#));
    let mut decoder = FrameDecoder::new();
    decoder.feed(&raw);
    assert_eq!(decoder.try_next().unwrap().unwrap()["id"], 1);
    assert_eq!(decoder.try_next().unwrap().unwrap()["id"], 2);
    assert!(decoder.try_next().unwrap().is_none());
}

#[test]
fn decoder_waits_for_a_partial_header() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(b"Content-Length: 12\r\n");
    assert!(decoder.try_next().unwrap().is_none());
    decoder.feed(b"\r\n{\"jsonrpc\":");
    assert!(decoder.try_next().unwrap().is_none());
}

#[test]
fn decoder_waits_for_a_partial_body() {
    let body = r#"{"jsonrpc":"2.0","id":3,"result":"hello"}"#;
    let raw = frame(body);
    let split = raw.len() - 3;
    let mut decoder = FrameDecoder::new();
    decoder.feed(&raw[..split]);
    assert!(decoder.try_next().unwrap().is_none());
    decoder.feed(&raw[split..]);
    assert_eq!(decoder.try_next().unwrap().unwrap()["id"], 3);
}

#[test]
fn decoder_rejects_a_missing_content_length() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(b"Content-Type: application/json\r\n\r\n{}");
    let error = decoder.try_next().unwrap_err();
    assert!(error.to_string().contains("Content-Length"), "{error}");
}

#[test]
fn decoder_rejects_a_non_numeric_content_length() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(b"Content-Length: twelve\r\n\r\n{}");
    let error = decoder.try_next().unwrap_err();
    assert!(
        error.to_string().contains("invalid LSP Content-Length"),
        "{error}"
    );
}

#[test]
fn decoder_rejects_an_oversized_body_before_buffering_it() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(format!("Content-Length: {}\r\n\r\n", MAX_CONTENT_LENGTH + 1).as_bytes());
    let error = decoder.try_next().unwrap_err();
    assert!(error.to_string().contains("exceeds"), "{error}");
}

#[test]
fn decoder_rejects_invalid_json_and_non_object_bodies() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(&frame("{not json"));
    let error = decoder.try_next().unwrap_err();
    assert!(
        error.to_string().contains("invalid LSP JSON body"),
        "{error}"
    );

    let mut decoder = FrameDecoder::new();
    decoder.feed(&frame("[1,2,3]"));
    let error = decoder.try_next().unwrap_err();
    assert!(error.to_string().contains("JSON object"), "{error}");
}

#[test]
fn decoder_rejects_a_malformed_header_line() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(b"not a header\r\n\r\n{}");
    let error = decoder.try_next().unwrap_err();
    assert!(
        error.to_string().contains("malformed LSP header line"),
        "{error}"
    );
}

#[test]
fn decoder_rejects_an_endless_header_block() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(&vec![b'x'; 9000]);
    let error = decoder.try_next().unwrap_err();
    assert!(error.to_string().contains("header block"), "{error}");
}

#[test]
fn decoder_finish_rejects_a_truncated_frame_and_accepts_a_clean_end() {
    let mut decoder = FrameDecoder::new();
    decoder.feed(b"Content-Length: 5\r\n\r\n{");
    let error = decoder.finish().unwrap_err();
    assert!(error.to_string().contains("middle of a frame"), "{error}");

    let decoder = FrameDecoder::new();
    assert!(decoder.finish().unwrap().is_none());
}

#[tokio::test]
async fn message_reader_decodes_frames_written_in_split_chunks() {
    let (mut writer, reader) = tokio::io::duplex(64);
    let mut messages = MessageReader::new(reader);
    let raw = frame(r#"{"jsonrpc":"2.0","id":9,"result":{"ok":true}}"#);
    let handle = tokio::spawn(async move {
        writer.write_all(&raw[..5]).await.unwrap();
        tokio::task::yield_now().await;
        writer.write_all(&raw[5..]).await.unwrap();
        writer.shutdown().await.unwrap();
    });
    let message = messages.next().await.unwrap().unwrap();
    assert_eq!(message["id"], 9);
    assert!(messages.next().await.unwrap().is_none());
    handle.await.unwrap();
}

#[tokio::test]
async fn message_reader_reports_a_frame_truncated_by_eof() {
    let (mut writer, reader) = tokio::io::duplex(64);
    let mut messages = MessageReader::new(reader);
    writer
        .write_all(b"Content-Length: 20\r\n\r\n{\"id\"")
        .await
        .unwrap();
    writer.shutdown().await.unwrap();
    let error = messages.next().await.unwrap_err();
    assert!(error.to_string().contains("middle of a frame"), "{error}");
}
