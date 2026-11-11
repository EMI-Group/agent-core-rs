//! Message content: plain text or a list of multimodal content parts.
//!
//! Ports the `content` field of `openjiuwen/core/foundation/llm/schema/message.py`,
//! where Python types it as `Union[str, List[Union[str, dict]]]`.
//! This module keeps the same tolerance on the wire — a JSON string, a JSON
//! array, or `null` all deserialize — while offering typed access to the four
//! part shapes the OpenAI-compatible and Anthropic transports share.
//! Any other part is preserved verbatim in [`ContentPart::Other`], so
//! provider-specific blocks (video, cache markers, citations, ...) round-trip
//! without being lost or rewritten.

use serde::de::{self, Deserializer};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

/// The content of a chat message.
///
/// Serializes as a JSON string for [`MessageContent::Text`] and as a JSON array
/// for [`MessageContent::Parts`]. Deserialization accepts either shape plus
/// `null` (which becomes empty text), matching the Python union type.
#[derive(Debug, Clone, PartialEq)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multimodal content: an ordered list of [`ContentPart`]s.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// The text content, or `None` when the content is a list of parts.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Parts(_) => None,
        }
    }

    /// Flattens the content to text: the text itself, or the [`ContentPart::text`]
    /// parts joined with `\n` for a part list (non-text parts are skipped).
    #[must_use]
    pub fn to_plain_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(ContentPart::text)
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// The content parts; an empty slice for [`MessageContent::Text`].
    #[must_use]
    pub fn parts(&self) -> &[ContentPart] {
        match self {
            Self::Parts(parts) => parts,
            Self::Text(_) => &[],
        }
    }

    /// Whether the content carries nothing: empty text, or no parts at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }

    /// Number of content parts: `0` for empty text and `1` for non-empty text,
    /// the vector length for [`MessageContent::Parts`].
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Text(text) => usize::from(!text.is_empty()),
            Self::Parts(parts) => parts.len(),
        }
    }
}

impl Default for MessageContent {
    /// Empty text content.
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<&str> for MessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for MessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<ContentPart>> for MessageContent {
    fn from(value: Vec<ContentPart>) -> Self {
        Self::Parts(value)
    }
}

impl Serialize for MessageContent {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text(text) => serializer.serialize_str(text),
            Self::Parts(parts) => parts.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Null => Ok(Self::Text(String::new())),
            Value::String(text) => Ok(Self::Text(text)),
            Value::Array(items) => items
                .into_iter()
                .map(part_from_value)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map(Self::Parts)
                .map_err(de::Error::custom),
            other => Err(de::Error::custom(format!(
                "message content must be a string, an array of content parts, or null, found {}",
                json_kind(&other)
            ))),
        }
    }
}

/// One element of a multimodal message.
///
/// Serializes as a tagged object: `{"type":"text","text":...}`,
/// `{"type":"image_url","image_url":{...}}`, `{"type":"audio","audio":{...}}`,
/// or `{"type":"file","file":{...}}`. [`ContentPart::Other`] serializes its
/// payload verbatim, and a part with an unknown or missing `type` deserializes
/// into it so vendor blocks are never rewritten.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    /// A text block.
    Text {
        /// The text itself.
        text: String,
    },
    /// An image referenced by URL (or a `data:` URI).
    ImageUrl {
        /// The image payload.
        image_url: ImageUrl,
    },
    /// An audio block.
    Audio(AudioPart),
    /// A file block.
    File(FilePart),
    /// A part this crate does not model, preserved as its raw JSON object.
    Other(Value),
}

impl ContentPart {
    /// The text of a [`ContentPart::Text`], or `None` for any other part.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }

    /// The image of a [`ContentPart::ImageUrl`], or `None` for any other part.
    #[must_use]
    pub fn image_url(&self) -> Option<&ImageUrl> {
        match self {
            Self::ImageUrl { image_url } => Some(image_url),
            _ => None,
        }
    }

    /// The wire `type` tag: `"text"`, `"image_url"`, `"audio"`, `"file"`, or —
    /// for [`ContentPart::Other`] — the raw `type` field, falling back to
    /// `"other"` when the object has none.
    #[must_use]
    pub fn part_type(&self) -> &str {
        match self {
            Self::Text { .. } => "text",
            Self::ImageUrl { .. } => "image_url",
            Self::Audio(_) => "audio",
            Self::File(_) => "file",
            Self::Other(value) => value.get("type").and_then(Value::as_str).unwrap_or("other"),
        }
    }
}

impl Serialize for ContentPart {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text { text } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            Self::ImageUrl { image_url } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "image_url")?;
                map.serialize_entry("image_url", image_url)?;
                map.end()
            }
            Self::Audio(part) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "audio")?;
                map.serialize_entry("audio", part)?;
                map.end()
            }
            Self::File(part) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "file")?;
                map.serialize_entry("file", part)?;
                map.end()
            }
            Self::Other(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        part_from_value(value).map_err(de::Error::custom)
    }
}

/// An image reference: `{"url": ..., "detail": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The image URL, possibly a `data:` URI.
    pub url: String,
    /// Optional provider detail hint (`"auto"`, `"low"`, `"high"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// An audio payload: `{"data": ..., "format": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioPart {
    /// Base64-encoded audio bytes, or a URL when the provider allows one.
    pub data: String,
    /// Optional audio container format (for example `"wav"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// A file payload: `{"url": ..., "data": ..., "mime_type": ..., "filename": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePart {
    /// URL of the file, when it is hosted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Inline file bytes (typically base64), when the file is attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Optional media type, for example `"application/pdf"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional original file name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Short JSON kind name used in deserialization error messages.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Converts one raw array element into a [`ContentPart`].
///
/// A bare string becomes [`ContentPart::Text`] for Python-union compatibility;
/// an object is interpreted by its `type` tag; anything else is a validation
/// error routed to `serde::de::Error::custom` by the callers.
fn part_from_value(value: Value) -> std::result::Result<ContentPart, String> {
    match value {
        Value::String(text) => Ok(ContentPart::Text { text }),
        Value::Object(map) => Ok(object_part(map)),
        other => Err(format!(
            "content part must be a string or an object, found {}",
            json_kind(&other)
        )),
    }
}

/// Interprets a tagged content-part object, preserving it when the tag or its
/// payload is not one this crate models.
fn object_part(map: Map<String, Value>) -> ContentPart {
    let tag = map.get("type").and_then(Value::as_str).map(str::to_owned);
    let parsed = match tag.as_deref() {
        Some("text") => map
            .get("text")
            .and_then(Value::as_str)
            .map(|text| ContentPart::Text {
                text: text.to_owned(),
            }),
        Some("image_url") => map
            .get("image_url")
            .and_then(|payload| serde_json::from_value::<ImageUrl>(payload.clone()).ok())
            .map(|image_url| ContentPart::ImageUrl { image_url }),
        Some("audio") => map
            .get("audio")
            .and_then(|payload| serde_json::from_value::<AudioPart>(payload.clone()).ok())
            .map(ContentPart::Audio),
        Some("file") => map
            .get("file")
            .and_then(|payload| serde_json::from_value::<FilePart>(payload.clone()).ok())
            .map(ContentPart::File),
        _ => None,
    };
    parsed.unwrap_or_else(|| ContentPart::Other(Value::Object(map)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(raw: &str) -> MessageContent {
        serde_json::from_str(raw).unwrap()
    }

    #[test]
    fn text_content_round_trips_as_a_json_string() {
        let content = MessageContent::Text("hello".to_owned());
        assert_eq!(serde_json::to_value(&content).unwrap(), json!("hello"));
        assert_eq!(parse("\"hello\""), content);
    }

    #[test]
    fn parts_content_round_trips_as_a_json_array() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/cat.png".to_owned(),
                    detail: Some("high".to_owned()),
                },
            },
        ]);
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(
            value,
            json!([
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://example.test/cat.png", "detail": "high"}},
            ])
        );
        assert_eq!(
            serde_json::from_value::<MessageContent>(value).unwrap(),
            content
        );
    }

    #[test]
    fn null_content_becomes_empty_text() {
        let content = parse("null");
        assert_eq!(content, MessageContent::Text(String::new()));
        assert!(content.is_empty());
        assert_eq!(serde_json::to_value(&content).unwrap(), json!(""));
    }

    #[test]
    fn bare_string_array_elements_become_text_parts() {
        let content = parse(r#"["a", {"type": "text", "text": "b"}]"#);
        assert_eq!(
            content,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "a".to_owned()
                },
                ContentPart::Text {
                    text: "b".to_owned()
                },
            ])
        );
    }

    #[test]
    fn empty_array_is_an_empty_part_list() {
        let content = parse("[]");
        assert_eq!(content, MessageContent::Parts(Vec::new()));
        assert!(content.is_empty());
        assert_eq!(content.len(), 0);
        assert_eq!(serde_json::to_value(&content).unwrap(), json!([]));
    }

    #[test]
    fn unknown_parts_are_preserved_verbatim() {
        // ports the Python tolerance for arbitrary dict content parts.
        let raw = json!({
            "type": "video",
            "source": {"url": "https://example.test/clip.mp4", "media_type": "video/mp4"},
            "duration": 3,
        });
        let part: ContentPart = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(part, ContentPart::Other(raw.clone()));
        assert_eq!(part.part_type(), "video");
        assert_eq!(serde_json::to_value(&part).unwrap(), raw);
    }

    #[test]
    fn unknown_part_without_type_keeps_the_raw_object() {
        let raw = json!({"image": "data:image/png;base64,AAAA"});
        let part: ContentPart = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(part, ContentPart::Other(raw));
        assert_eq!(part.part_type(), "other");
    }

    #[test]
    fn image_url_detail_is_optional() {
        let without: ContentPart =
            serde_json::from_value(json!({"type": "image_url", "image_url": {"url": "https://x"}}))
                .unwrap();
        assert_eq!(
            without,
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://x".to_owned(),
                    detail: None,
                }
            }
        );
        assert_eq!(
            serde_json::to_value(&without).unwrap(),
            json!({"type": "image_url", "image_url": {"url": "https://x"}}),
            "a missing detail must not be serialized as null"
        );

        let with: ContentPart = serde_json::from_value(
            json!({"type": "image_url", "image_url": {"url": "https://x", "detail": "low"}}),
        )
        .unwrap();
        assert_eq!(with.image_url().unwrap().detail.as_deref(), Some("low"));
    }

    #[test]
    fn unknown_fields_inside_a_known_part_are_ignored() {
        let part: ContentPart = serde_json::from_value(json!({
            "type": "text",
            "text": "hi",
            "cache_control": {"type": "ephemeral"},
        }))
        .unwrap();
        assert_eq!(
            part,
            ContentPart::Text {
                text: "hi".to_owned()
            }
        );
    }

    #[test]
    fn audio_and_file_parts_round_trip_without_none_fields() {
        let audio = ContentPart::Audio(AudioPart {
            data: "AAAA".to_owned(),
            format: Some("wav".to_owned()),
        });
        assert_eq!(
            serde_json::to_value(&audio).unwrap(),
            json!({"type": "audio", "audio": {"data": "AAAA", "format": "wav"}})
        );

        let file = ContentPart::File(FilePart {
            url: Some("https://example.test/a.pdf".to_owned()),
            data: None,
            mime_type: Some("application/pdf".to_owned()),
            filename: None,
        });
        assert_eq!(
            serde_json::to_value(&file).unwrap(),
            json!({
                "type": "file",
                "file": {"url": "https://example.test/a.pdf", "mime_type": "application/pdf"},
            })
        );
        assert_eq!(
            serde_json::from_value::<ContentPart>(serde_json::to_value(&file).unwrap()).unwrap(),
            file
        );
    }

    #[test]
    fn malformed_known_part_falls_back_to_the_raw_object() {
        let raw = json!({"type": "image_url", "image_url": "https://x"});
        let part: ContentPart = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(part, ContentPart::Other(raw));
    }

    #[test]
    fn to_plain_text_joins_text_parts_and_skips_others() {
        assert_eq!(
            MessageContent::Text("plain".to_owned()).to_plain_text(),
            "plain"
        );
        let parts = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "first".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://x".to_owned(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "second".to_owned(),
            },
        ]);
        assert_eq!(parts.to_plain_text(), "first\nsecond");
    }

    #[test]
    fn as_text_parts_is_empty_and_len() {
        let text = MessageContent::Text("hi".to_owned());
        assert_eq!(text.as_text(), Some("hi"));
        assert_eq!(text.len(), 1);
        assert_eq!(text.parts().len(), 0, "text content has no parts");
        assert!(!text.is_empty());

        let empty = MessageContent::Text(String::new());
        assert_eq!(empty.as_text(), Some(""));
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let parts = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "a".to_owned(),
            },
            ContentPart::Text {
                text: "b".to_owned(),
            },
        ]);
        assert_eq!(parts.as_text(), None);
        assert_eq!(parts.len(), 2);
        assert!(!parts.is_empty());
        assert_eq!(parts.parts().len(), 2);
    }

    #[test]
    fn part_type_reports_the_wire_tag() {
        let image = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://x".to_owned(),
                detail: None,
            },
        };
        let audio = ContentPart::Audio(AudioPart {
            data: String::new(),
            format: None,
        });
        let file = ContentPart::File(FilePart {
            url: None,
            data: None,
            mime_type: None,
            filename: None,
        });
        assert_eq!(
            ContentPart::Text {
                text: String::new()
            }
            .part_type(),
            "text"
        );
        assert_eq!(image.part_type(), "image_url");
        assert_eq!(audio.part_type(), "audio");
        assert_eq!(file.part_type(), "file");
        assert_eq!(ContentPart::Other(json!({"type": 5})).part_type(), "other");
    }

    #[test]
    fn text_and_image_accessors_only_match_their_own_variant() {
        let text = ContentPart::Text {
            text: "hi".to_owned(),
        };
        assert_eq!(text.text(), Some("hi"));
        assert!(text.image_url().is_none());

        let image = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://x".to_owned(),
                detail: None,
            },
        };
        assert!(image.text().is_none());
        assert_eq!(image.image_url().unwrap().url, "https://x");
    }

    #[test]
    fn non_string_non_array_content_is_an_error() {
        for raw in ["42", "{}", "true", "1.5"] {
            assert!(
                serde_json::from_str::<MessageContent>(raw).is_err(),
                "`{raw}` must not deserialize into message content"
            );
        }
    }

    #[test]
    fn non_object_array_elements_are_errors() {
        for raw in ["[42]", "[null]", "[true]"] {
            assert!(
                serde_json::from_str::<MessageContent>(raw).is_err(),
                "`{raw}` must not deserialize into message content"
            );
        }
    }

    #[test]
    fn default_and_from_conversions() {
        assert_eq!(
            MessageContent::default(),
            MessageContent::Text(String::new())
        );
        assert!(MessageContent::default().is_empty());
        assert_eq!(
            MessageContent::from("hi"),
            MessageContent::Text("hi".to_owned())
        );
        assert_eq!(
            MessageContent::from("hi".to_owned()),
            MessageContent::Text("hi".to_owned())
        );
        let parts = vec![ContentPart::Text {
            text: "hi".to_owned(),
        }];
        assert_eq!(
            MessageContent::from(parts.clone()),
            MessageContent::Parts(parts)
        );
    }
}
