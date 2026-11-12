//! [`MultimodalDocument`]: a [`Document`] that also carries typed content parts.
//!
//! Port of the Python `MultimodalDocument` in
//! `openjiuwen/core/retrieval/common/document.py`. Python keeps the parts in the
//! private `_data` list of `(kind, data, data_id)` tuples and exposes a cached
//! `content` property in the OpenAI-compatible chat format; Rust stores them as a public
//! `parts: Vec<ContentPart>` (serializable) and computes [`MultimodalDocument::content`]
//! on demand.
//!
//! Two deliberate deviations, both documented in the module [`super`] docs:
//! `strip()` returns `None` only when the document has neither text nor parts
//! (Python only looks at `_data`), and a content part may be built without a
//! `data_id` (Python always generates one for non-text fields).

use std::path::Path;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};

use crate::common::{Document, Metadata};

use super::modality::{guess_mime_type, Modality};

/// One typed piece of multimodal content.
///
/// `data` is a URL, a `data:<mime>;base64,<payload>` URI, or plain text, and
/// `data_id` is the optional 32-character cache key Python attaches as `uuid`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContentPart {
    /// Modality of this part; defaults to [`Modality::Text`].
    pub modality: Modality,
    /// The payload: text, a URL, or a base64 data URI.
    pub data: String,
    /// Optional 32-character cache key emitted as `uuid` by
    /// [`MultimodalDocument::content`].
    pub data_id: String,
}

/// A document whose content may mix text, images, audio, and video.
///
/// The `text` field stays the text-only fallback (what a non-multimodal embedder
/// would index); the typed parts live in `parts`. Deserialization defaults every
/// field and tolerates unknown keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MultimodalDocument {
    /// Document identifier, serialized as `id_` like [`Document::id`].
    #[serde(rename = "id_", alias = "id")]
    pub id: String,
    /// Text-only fallback content.
    pub text: String,
    /// Document metadata.
    pub metadata: Metadata,
    /// Typed content parts, in insertion order.
    pub parts: Vec<ContentPart>,
}

impl MultimodalDocument {
    /// Creates an empty-parts document with a generated `uuid4` id.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let document = Document::new(text);
        Self {
            id: document.id,
            text: document.text,
            metadata: Metadata::new(),
            parts: Vec::new(),
        }
    }

    /// Creates a document with an explicit id and no parts.
    #[must_use]
    pub fn with_id(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: Metadata::new(),
            parts: Vec::new(),
        }
    }

    /// Appends a text part (`add_field("text", ...)` in Python).
    ///
    /// The fallback `text` field is left untouched, matching Python, where
    /// `add_field` only appends to the private parts list.
    pub fn add_text(&mut self, text: impl Into<String>) {
        self.parts.push(ContentPart {
            modality: Modality::Text,
            data: text.into(),
            data_id: String::new(),
        });
    }

    /// Appends a part verbatim, without validating its payload.
    ///
    /// Use [`MultimodalDocument::add_field`] to validate a data URI or URL.
    pub fn add_part(
        &mut self,
        modality: Modality,
        data: impl Into<String>,
        data_id: impl Into<String>,
    ) {
        self.parts.push(ContentPart {
            modality,
            data: data.into(),
            data_id: data_id.into(),
        });
    }

    /// Appends a validated text, image, audio, or video field.
    ///
    /// Text accepts any payload; image and video accept a `data:image/` or
    /// `data:video/` base64 URI or an `http(s)://` URL; audio accepts only a
    /// `data:audio/` base64 URI (Python parity). Non-text fields get a generated
    /// 32-character hex `data_id`, text fields an empty one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the payload does not match the
    /// modality, with Python's message.
    pub fn add_field(&mut self, modality: Modality, data: &str) -> Result<()> {
        if !accepts_field_data(modality, data) {
            return Err(Error::Validation(format!(
                "MultimodalDocument.add_field received invalid \"data\", this can be url or start with \"data:{modality}/\""
            )));
        }
        let data_id = if modality == Modality::Text {
            String::new()
        } else {
            new_data_id()
        };
        self.add_part(modality, data, data_id);
        Ok(())
    }

    /// Reads `path` and appends it as a validated part.
    ///
    /// The MIME type comes from the file extension, the payload is base64-encoded
    /// as `data:<mime>;base64,<payload>` (text files are read as UTF-8 instead),
    /// and the part gets a generated 32-character hex `data_id`. Text parts keep
    /// Python's empty `data_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for a missing file, [`Error::Transport`] for
    /// other I/O failures, and [`Error::Validation`] when the extension does not
    /// determine a MIME type.
    pub async fn add_file(&mut self, modality: Modality, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = match tokio::fs::read(path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotFound(format!(
                    "Unable to open {modality} file at {}",
                    path.display()
                )));
            }
            Err(error) => {
                return Err(Error::Transport(format!(
                    "Unable to read {modality} file at {}: {error}",
                    path.display()
                )));
            }
        };
        let mime = guess_mime_type(&path.to_string_lossy()).ok_or_else(|| {
            Error::Validation(format!(
                "Unable to determine mimetype for {modality} file: {}",
                path.display()
            ))
        })?;
        if modality == Modality::Text {
            self.add_part(modality, String::from_utf8_lossy(&bytes), String::new());
        } else {
            self.add_part(modality, to_data_uri(modality, mime, &bytes), new_data_id());
        }
        Ok(())
    }

    /// Returns the parts in the OpenAI-compatible chat content format.
    ///
    /// Text parts become `{"type": "text", "text": ...}`, images and videos become
    /// `{"type": "image_url" | "video_url", ...}`, and audio becomes
    /// `{"type": "input_audio", "input_audio": {"data": ..., "format": ...}}`.
    /// Parts carrying a `data_id` gain a `"uuid"` key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when an audio part is not a
    /// `data:audio/<format>;base64,` URI (Python raises the same error).
    pub fn content(&self) -> Result<Vec<Value>> {
        let mut content = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            let mut item = match part.modality {
                Modality::Text => json!({"type": "text", "text": part.data.clone()}),
                Modality::Image => json!({
                    "type": "image_url",
                    "image_url": {"url": part.data.clone()},
                }),
                Modality::Video => json!({
                    "type": "video_url",
                    "video_url": {"url": part.data.clone()},
                }),
                Modality::Audio => {
                    let format = audio_format(&part.data).ok_or_else(|| {
                        Error::Validation(
                            "Audio data must be a base64 data URI starting with \"data:audio/\""
                                .to_owned(),
                        )
                    })?;
                    json!({
                        "type": "input_audio",
                        "input_audio": {"data": part.data.clone(), "format": format},
                    })
                }
            };
            if !part.data_id.is_empty() {
                item["uuid"] = Value::String(part.data_id.clone());
            }
            content.push(item);
        }
        Ok(content)
    }

    /// Returns `None` when the document has neither text nor parts.
    ///
    /// Python returns `self if self._data else None`, which treats a text-only
    /// document as empty; the Rust port keeps a non-empty `text` alive.
    #[must_use]
    pub fn strip(self) -> Option<Self> {
        if self.text.is_empty() && self.parts.is_empty() {
            None
        } else {
            Some(self)
        }
    }

    /// Projects the text fallback and identity into a plain [`Document`].
    ///
    /// The typed parts are dropped, matching Python's `text`-only services path.
    #[must_use]
    pub fn to_document(&self) -> Document {
        Document {
            id: self.id.clone(),
            text: self.text.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Builds a part-less multimodal document from a plain [`Document`].
    #[must_use]
    pub fn from_document(document: &Document) -> Self {
        Self {
            id: document.id.clone(),
            text: document.text.clone(),
            metadata: document.metadata.clone(),
            parts: Vec::new(),
        }
    }

    /// Counts the parts per modality name, sorted by name.
    #[must_use]
    pub fn modality_summary(&self) -> std::collections::BTreeMap<String, usize> {
        let mut summary = std::collections::BTreeMap::new();
        for part in &self.parts {
            *summary
                .entry(part.modality.as_str().to_owned())
                .or_insert(0_usize) += 1;
        }
        summary
    }
}

/// Encodes `bytes` as a base64 data URI with the modality's media type.
///
/// `mime` is repaired when it does not belong to `modality` (Python rewrites
/// `image/bmp` to `audio/bmp` for audio fields); a bare subtype such as `"png"`
/// gains the modality prefix.
#[must_use]
pub fn to_data_uri(modality: Modality, mime: &str, bytes: &[u8]) -> String {
    format!(
        "data:{};base64,{}",
        align_mime(modality, mime),
        STANDARD.encode(bytes)
    )
}

/// Splits a base64 data URI into its media type and decoded bytes.
///
/// Returns `None` for anything that is not `data:<mime>;base64,<payload>` or
/// whose payload is not valid (padded) base64; percent-encoded data URIs are not
/// decoded.
#[must_use]
pub fn parse_data_uri(uri: &str) -> Option<(String, Vec<u8>)> {
    let rest = uri.strip_prefix("data:")?;
    let (header, payload) = rest.split_once(',')?;
    let mime = header.strip_suffix(";base64")?;
    if mime.is_empty() || !mime.contains('/') {
        return None;
    }
    let bytes = STANDARD.decode(payload.trim()).ok()?;
    Some((mime.to_owned(), bytes))
}

/// Rewrites `mime` so that it starts with the modality prefix.
fn align_mime(modality: Modality, mime: &str) -> String {
    if mime.starts_with(modality.mime_prefix()) {
        return mime.to_owned();
    }
    let subtype = mime.split_once('/').map_or(mime, |(_, subtype)| subtype);
    format!("{}{subtype}", modality.mime_prefix())
}

/// Returns a fresh 32-character hex identifier (Python's `uuid.uuid4().hex`).
fn new_data_id() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Returns `true` when `data` is an acceptable payload for `modality`.
fn accepts_field_data(modality: Modality, data: &str) -> bool {
    if modality == Modality::Text {
        return true;
    }
    if has_data_uri_prefix(modality, data) {
        return true;
    }
    modality != Modality::Audio && is_url(data)
}

/// Returns `true` for `data:<modality>/<subtype>;base64,<payload>`.
fn has_data_uri_prefix(modality: Modality, data: &str) -> bool {
    let Some(rest) = data.strip_prefix("data:") else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(modality.mime_prefix()) else {
        return false;
    };
    rest.find(";base64,").is_some_and(|index| index > 0)
}

/// Matches Python's `data.startswith("http") and "://" in data`.
fn is_url(data: &str) -> bool {
    data.starts_with("http") && data.contains("://")
}

/// Extracts the `<format>` of a `data:audio/<format>;base64,` URI.
fn audio_format(data: &str) -> Option<&str> {
    let rest = data.strip_prefix("data:audio/")?;
    let index = rest.find(";base64,")?;
    (index > 0).then(|| &rest[..index])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn data_uri_round_trips_through_the_helpers() {
        let uri = to_data_uri(Modality::Image, "image/png", b"\x89PNG");
        assert_eq!(uri, "data:image/png;base64,iVBORw==");
        let (mime, bytes) = parse_data_uri(&uri).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, b"\x89PNG");

        // An empty payload is a valid, empty data URI.
        let empty = to_data_uri(Modality::Audio, "audio/wav", b"");
        assert_eq!(empty, "data:audio/wav;base64,");
        assert_eq!(
            parse_data_uri(&empty),
            Some(("audio/wav".to_owned(), Vec::new()))
        );
    }

    #[test]
    fn data_uri_helpers_reject_malformed_input() {
        assert!(parse_data_uri("https://example.com/a.png").is_none());
        assert!(parse_data_uri("data:image/png,abc").is_none());
        assert!(parse_data_uri("data:;base64,AAAA").is_none());
        assert!(parse_data_uri("data:png;base64,AAAA").is_none());
        assert!(parse_data_uri("data:image/png;base64,***").is_none());
    }

    #[test]
    fn to_data_uri_aligns_the_media_type_with_the_modality() {
        assert_eq!(
            to_data_uri(Modality::Image, "png", b"x"),
            "data:image/png;base64,eA=="
        );
        assert_eq!(
            to_data_uri(Modality::Audio, "audio/wav", b"x"),
            "data:audio/wav;base64,eA=="
        );
        // Python rewrites a foreign top-level type instead of rejecting it.
        assert_eq!(
            to_data_uri(Modality::Audio, "image/bmp", b"x"),
            "data:audio/bmp;base64,eA=="
        );
    }

    #[test]
    fn content_shapes_follow_the_openai_chat_format() {
        let mut document = MultimodalDocument::with_id("doc-1", "fallback");
        document.add_part(Modality::Text, "describe", "");
        document.add_part(Modality::Image, "https://example.com/a.png", "");
        document.add_part(Modality::Video, "https://example.com/a.mp4", "vid");
        document.add_part(Modality::Audio, "data:audio/wav;base64,eA==", "aud");

        assert_eq!(
            document.content().unwrap(),
            vec![
                json!({"type": "text", "text": "describe"}),
                json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
                json!({
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/a.mp4"},
                    "uuid": "vid",
                }),
                json!({
                    "type": "input_audio",
                    "input_audio": {"data": "data:audio/wav;base64,eA==", "format": "wav"},
                    "uuid": "aud",
                }),
            ]
        );
    }

    #[test]
    fn content_rejects_audio_without_a_base64_data_uri() {
        let mut document = MultimodalDocument::new("");
        document.add_part(Modality::Audio, "https://example.com/a.wav", "");
        let error = document.content().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("data:audio/"));
    }

    #[test]
    fn add_field_validates_the_payload_per_modality() {
        let mut document = MultimodalDocument::new("body");
        document.add_field(Modality::Text, "anything goes").unwrap();
        document
            .add_field(Modality::Image, "data:image/png;base64,eA==")
            .unwrap();
        document
            .add_field(Modality::Image, "https://example.com/a.png")
            .unwrap();
        document
            .add_field(Modality::Video, "data:video/mp4;base64,eA==")
            .unwrap();
        document
            .add_field(Modality::Audio, "data:audio/wav;base64,eA==")
            .unwrap();

        assert_eq!(document.parts.len(), 5);
        assert_eq!(document.parts[0].data_id, "");
        for part in &document.parts[1..] {
            assert_eq!(part.data_id.len(), 32);
            assert!(part.data_id.chars().all(|c| c.is_ascii_hexdigit()));
        }

        let error = document.add_field(Modality::Image, "a.png").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("this can be url or start with \"data:image/\""));
        // Python keeps audio strictly base64-only.
        assert!(document
            .add_field(Modality::Audio, "https://example.com/a.wav")
            .is_err());
        // A wrong media type prefix is rejected too.
        assert!(document
            .add_field(Modality::Image, "data:audio/wav;base64,eA==")
            .is_err());
        assert!(document
            .add_field(Modality::Video, "data:image/png;base64,eA==")
            .is_err());
    }

    #[tokio::test]
    async fn add_file_encodes_images_and_reads_text_files() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("pixel.PNG");
        std::fs::write(&image, b"\x89PNG").unwrap();
        let text = dir.path().join("notes.txt");
        std::fs::write(&text, "hello").unwrap();

        let mut document = MultimodalDocument::new("");
        document.add_file(Modality::Image, &image).await.unwrap();
        document.add_file(Modality::Text, &text).await.unwrap();

        assert_eq!(document.parts.len(), 2);
        assert_eq!(document.parts[0].modality, Modality::Image);
        assert_eq!(document.parts[0].data, "data:image/png;base64,iVBORw==");
        assert_eq!(document.parts[0].data_id.len(), 32);
        assert_eq!(document.parts[1].modality, Modality::Text);
        assert_eq!(document.parts[1].data, "hello");
        assert_eq!(document.parts[1].data_id, "");
    }

    #[tokio::test]
    async fn add_file_reports_missing_and_unknown_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut document = MultimodalDocument::new("");
        let missing = dir.path().join("absent.png");
        let error = document
            .add_file(Modality::Image, &missing)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("Unable to open image file at"));

        let unknown = dir.path().join("data.bin");
        std::fs::write(&unknown, b"x").unwrap();
        let error = document
            .add_file(Modality::Image, &unknown)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("Unable to determine mimetype"));
    }

    #[test]
    fn strip_keeps_documents_with_text_or_parts() {
        assert!(MultimodalDocument::new("").strip().is_none());
        assert!(MultimodalDocument::new("body").strip().is_some());
        let mut with_part = MultimodalDocument::new("");
        with_part.add_text("hi");
        assert!(with_part.strip().is_some());
    }

    #[test]
    fn documents_convert_both_ways_and_summarize_modalities() {
        let mut document = MultimodalDocument::with_id("doc-9", "caption");
        document.metadata.insert("k".to_owned(), json!("v"));
        document.add_text("a");
        document.add_text("b");
        document.add_part(Modality::Image, "https://example.com/a.png", "");

        let summary = document.modality_summary();
        assert_eq!(summary.get("text"), Some(&2));
        assert_eq!(summary.get("image"), Some(&1));
        assert_eq!(summary.len(), 2);

        let plain = document.to_document();
        assert_eq!(plain.id, "doc-9");
        assert_eq!(plain.text, "caption");
        assert_eq!(plain.metadata, document.metadata);

        let restored = MultimodalDocument::from_document(&plain);
        assert_eq!(restored.id, "doc-9");
        assert_eq!(restored.text, "caption");
        assert_eq!(restored.parts.len(), 0);
    }

    #[test]
    fn serde_defaults_and_unknown_fields_are_tolerated() {
        let document: MultimodalDocument =
            serde_json::from_str(r#"{"text": "body", "extra": 1}"#).unwrap();
        assert_eq!(document.text, "body");
        assert_eq!(document.id, "");
        assert_eq!(document.parts.len(), 0);

        let part: ContentPart = serde_json::from_str("{}").unwrap();
        assert_eq!(part.modality, Modality::Text);
        assert_eq!(part.data, "");

        let value = serde_json::to_value(ContentPart {
            modality: Modality::Image,
            data: "data:image/png;base64,eA==".to_owned(),
            data_id: "abc".to_owned(),
        })
        .unwrap();
        assert_eq!(
            value,
            json!({"modality": "image", "data": "data:image/png;base64,eA==", "data_id": "abc"})
        );
    }

    #[test]
    fn generated_documents_serialize_with_the_python_id_name() {
        let document = MultimodalDocument::with_id("doc-1", "body");
        let value = serde_json::to_value(&document).unwrap();
        assert_eq!(
            value,
            json!({"id_": "doc-1", "text": "body", "metadata": {}, "parts": []})
        );
        let from_alias: MultimodalDocument =
            serde_json::from_value(json!({"id": "doc-2", "text": "x"})).unwrap();
        assert_eq!(from_alias.id, "doc-2");

        assert_eq!(MultimodalDocument::new("x").id.len(), 36);
    }
}
