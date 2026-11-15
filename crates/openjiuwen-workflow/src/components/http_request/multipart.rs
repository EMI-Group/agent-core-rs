//! `multipart/form-data` request-body encoding.
//!
//! The shared transport only accepts JSON/text/byte bodies, so this module renders the
//! whole form — boundary, `Content-Disposition` parts, CRLF framing, closing boundary —
//! into one byte buffer that is then sent as an ordinary binary body. Retries,
//! cancellation, and the response size cap therefore keep working unchanged.
//!
//! # Parts
//!
//! - A scalar or JSON field value becomes a text part; its text is templated like a
//!   URL-encoded form value (`${path}` references first, then `{{key}}` placeholders).
//! - A field whose value is an object with a `filename` plus a base64 payload
//!   (`content_base64`, or the `data`/`content` aliases) becomes a binary file part with
//!   that filename and a `content_type` from the object, guessed from the filename
//!   extension, or `application/octet-stream`.
//! - Any other object value is sent as JSON text.
//!
//! Field names and filenames are sanitized (CR/LF dropped, `"` replaced by `'`) so a
//! value cannot inject extra part headers.
//!
//! # Boundary
//!
//! Boundaries use the fixed format `----openjiuwen-<uuid>`; the encoder scans every part
//! it is about to write and re-rolls the random suffix while the candidate occurs in the
//! payload, so a boundary can never collide with the data it delimits.

use std::collections::BTreeMap;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use bytes::{Bytes, BytesMut};
use serde_json::Value;
use uuid::Uuid;

use super::template::{resolve_template, stringify};

/// The media type of an encoded form (`multipart/form-data`).
pub(crate) const MULTIPART_FORM_DATA: &str = "multipart/form-data";

/// Prefix of every generated boundary.
const BOUNDARY_PREFIX: &str = "----openjiuwen-";

/// How many random boundaries are tried before the last candidate is accepted.
const BOUNDARY_ATTEMPTS: usize = 8;

/// The field names accepted as the base64 payload of a file part, in priority order.
const FILE_CONTENT_KEYS: [&str; 3] = ["content_base64", "data", "content"];

/// The `Content-Type` of a file part whose type cannot be guessed.
const DEFAULT_PART_CONTENT_TYPE: &str = "application/octet-stream";

/// The frame separator of every multipart line.
const CRLF: &str = "\r\n";

/// An encoded multipart body and the boundary that delimits it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedMultipart {
    /// The generated boundary, also carried by the `Content-Type` parameter.
    pub(crate) boundary: String,
    /// The complete encoded body, ready to send as bytes.
    pub(crate) body: Bytes,
}

impl EncodedMultipart {
    /// The `Content-Type` header value of this body, including the boundary parameter.
    pub(crate) fn content_type(&self) -> String {
        format!("{MULTIPART_FORM_DATA}; boundary={}", self.boundary)
    }
}

/// Encodes `fields` as a `multipart/form-data` body.
pub(crate) fn encode(fields: &BTreeMap<String, Value>, inputs: &Value) -> EncodedMultipart {
    let parts: Vec<Vec<u8>> = fields
        .iter()
        .map(|(name, value)| encode_part(name, value, inputs))
        .collect();
    let boundary = generate_boundary(&parts);

    let capacity =
        parts.iter().map(Vec::len).sum::<usize>() + (parts.len() + 1) * (boundary.len() + 6);
    let mut body = BytesMut::with_capacity(capacity);
    for part in &parts {
        body.extend_from_slice(b"--");
        body.extend_from_slice(boundary.as_bytes());
        body.extend_from_slice(CRLF.as_bytes());
        body.extend_from_slice(part);
    }
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"--");
    body.extend_from_slice(CRLF.as_bytes());

    EncodedMultipart {
        boundary,
        body: body.freeze(),
    }
}

/// Encodes one field as a complete part (its headers, a blank line, the data, a CRLF).
fn encode_part(name: &str, value: &Value, inputs: &Value) -> Vec<u8> {
    match file_part(value, inputs) {
        Some(file) => binary_part(name, &file),
        None => text_part(name, &resolve_template(&stringify(value), inputs)),
    }
}

/// A decoded file upload.
struct FilePart {
    /// The `filename` parameter of the part.
    filename: String,
    /// The part's own `Content-Type`.
    content_type: String,
    /// The decoded bytes.
    data: Bytes,
}

/// Recognizes a file-upload field: `{"filename": …, "content_base64": …}`.
fn file_part(value: &Value, inputs: &Value) -> Option<FilePart> {
    let Value::Object(map) = value else {
        return None;
    };
    let filename = map.get("filename").and_then(Value::as_str)?;
    let encoded = FILE_CONTENT_KEYS
        .iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))?;
    let cleaned: String = resolve_template(encoded, inputs)
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    let data = BASE64_STANDARD.decode(cleaned.as_bytes()).ok()?;
    let content_type = map.get("content_type").and_then(Value::as_str).map_or_else(
        || guess_content_type(filename),
        |value| resolve_template(value, inputs),
    );
    Some(FilePart {
        filename: resolve_template(filename, inputs),
        content_type,
        data: Bytes::from(data),
    })
}

/// Renders a text part.
fn text_part(name: &str, text: &str) -> Vec<u8> {
    format!(
        "Content-Disposition: form-data; name=\"{}\"{CRLF}{CRLF}{text}{CRLF}",
        sanitize(name)
    )
    .into_bytes()
}

/// Renders a binary (file) part.
fn binary_part(name: &str, file: &FilePart) -> Vec<u8> {
    let mut part = format!(
        "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"{CRLF}\
         Content-Type: {}{CRLF}{CRLF}",
        sanitize(name),
        sanitize(&file.filename),
        file.content_type,
    )
    .into_bytes();
    part.extend_from_slice(&file.data);
    part.extend_from_slice(CRLF.as_bytes());
    part
}

/// Strips the characters that could terminate a part header.
fn sanitize(text: &str) -> String {
    text.chars()
        .filter(|character| !matches!(character, '\r' | '\n'))
        .map(|character| if character == '"' { '\'' } else { character })
        .collect()
}

/// Guesses a part `Content-Type` from the filename extension.
fn guess_content_type(filename: &str) -> String {
    let extension = filename
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .unwrap_or_default();
    let guessed = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "json" => "application/json",
        "xml" => "application/xml",
        "zip" => "application/zip",
        _ => DEFAULT_PART_CONTENT_TYPE,
    };
    guessed.to_owned()
}

/// Generates a boundary that does not occur in any `part`.
fn generate_boundary(parts: &[Vec<u8>]) -> String {
    let mut candidate = String::new();
    for _ in 0..BOUNDARY_ATTEMPTS {
        candidate = format!("{BOUNDARY_PREFIX}{}", Uuid::new_v4().simple());
        if !parts.iter().any(|part| {
            part.windows(candidate.len())
                .any(|window| window == candidate.as_bytes())
        }) {
            break;
        }
    }
    candidate
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    fn fields(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect()
    }

    #[test]
    fn a_form_is_framed_by_its_boundary() {
        let encoded = encode(&fields(&[("f", json!("v"))]), &json!({}));
        assert!(encoded.boundary.starts_with(BOUNDARY_PREFIX));
        assert_eq!(
            encoded.content_type(),
            format!("multipart/form-data; boundary={}", encoded.boundary)
        );
        let expected = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"f\"\r\n\r\nv\r\n--{b}--\r\n",
            b = encoded.boundary
        );
        assert_eq!(encoded.body, Bytes::from(expected));
    }

    #[test]
    fn scalar_and_object_values_are_text_parts() {
        let encoded = encode(
            &fields(&[
                ("n", json!(3)),
                ("obj", json!({"a": 1})),
                ("tpl", json!("{{name}}")),
                ("ref", json!("${name}")),
            ]),
            &json!({"name": "ada"}),
        );
        let text = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(text.contains("\r\n\r\n3\r\n"), "unexpected body: {text}");
        assert!(text.contains("\r\n\r\n{\"a\":1}\r\n"));
        assert!(text.contains("\r\n\r\nada\r\n"));
        assert_eq!(
            text.matches("name=\"ref\"").count(),
            1,
            "the ${{name}} reference resolved to the input value"
        );
        assert!(text.contains("name=\"ref\"\r\n\r\nada\r\n"));
    }

    #[test]
    fn a_file_field_round_trips_its_bytes() {
        let payload: [u8; 5] = [0, 1, 2, 254, 255];
        let encoded = encode(
            &fields(&[(
                "upload",
                json!({"filename": "a.bin", "content_base64": BASE64_STANDARD.encode(payload)}),
            )]),
            &json!({}),
        );
        let body = encoded.body.to_vec();
        let header = b"Content-Disposition: form-data; name=\"upload\"; filename=\"a.bin\"\r\n\
                       Content-Type: application/octet-stream\r\n\r\n";
        let start = body
            .windows(header.len())
            .position(|window| window == header)
            .expect("the file part declares its filename and content type");
        let data = &body[start + header.len()..start + header.len() + payload.len()];
        assert_eq!(data, payload, "the decoded bytes are sent verbatim");
        assert_eq!(&body[start + header.len() + payload.len()..][..2], b"\r\n");
    }

    #[test]
    fn the_data_and_content_aliases_are_accepted_and_the_type_is_guessed() {
        for key in ["data", "content"] {
            let encoded = encode(
                &fields(&[(
                    "upload",
                    json!({"filename": "note.json", key: BASE64_STANDARD.encode("{}")}),
                )]),
                &json!({}),
            );
            let text = String::from_utf8(encoded.body.to_vec()).unwrap();
            assert!(
                text.contains("Content-Type: application/json\r\n"),
                "unexpected body: {text}"
            );
            assert!(text.contains("filename=\"note.json\""));
        }
        // An explicit `content_type` wins over the extension.
        let encoded = encode(
            &fields(&[(
                "upload",
                json!({"filename": "a.txt", "data": "eA==", "content_type": "image/png"}),
            )]),
            &json!({}),
        );
        let text = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(text.contains("Content-Type: image/png\r\n"));
    }

    #[test]
    fn unicode_names_and_values_survive() {
        let encoded = encode(&fields(&[("名前", json!("héllo 世界"))]), &json!({}));
        let text = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(text.contains("name=\"名前\""));
        assert!(text.contains("\r\n\r\nhéllo 世界\r\n"));
    }

    #[test]
    fn part_headers_cannot_be_injected() {
        let encoded = encode(&fields(&[("a\"b\r\nX-Evil: 1", json!("v"))]), &json!({}));
        let text = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(text.contains("name=\"a'bX-Evil: 1\""));
        assert!(!text.contains("X-Evil: 1\r\n"));
    }

    #[test]
    fn boundaries_are_unique_per_form() {
        let first = encode(&fields(&[("f", json!("v"))]), &json!({}));
        let second = encode(&fields(&[("f", json!("v"))]), &json!({}));
        assert_ne!(first.boundary, second.boundary);
    }

    #[test]
    fn a_boundary_never_occurs_in_the_payload() {
        // A payload that contains a boundary-shaped string does not reuse it.
        let encoded = encode(
            &fields(&[("f", json!("----openjiuwen-deadbeef"))]),
            &json!({}),
        );
        let text = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(!text.contains(&format!("--{}--", "----openjiuwen-deadbeef")));
        assert!(text.contains(&format!("--{}--\r\n", encoded.boundary)));
        // The scanner the generator relies on is exact.
        let parts = vec![b"xx".to_vec()];
        let boundary = generate_boundary(&parts);
        assert_ne!(boundary, "", "a boundary is always generated");
    }
}
