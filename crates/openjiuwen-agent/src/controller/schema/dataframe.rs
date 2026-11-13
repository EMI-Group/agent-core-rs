//! Controller payload frames.
//!
//! Ports `openjiuwen/core/controller/schema/dataframe.py`: text, file, and
//! JSON frames are the basic unit of data exchange between an event, a task,
//! and the controller's output. Python's explicit `type` discriminator lives
//! in the [`DataFrame`] tag, so a text frame serializes as
//! `{"type":"text","text":"…"}`.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::BTreeMap;
/// Text payload frame.
///
/// Carries plain text such as user input or a task description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TextDataFrame {
    /// Text content.
    pub text: String,
}
impl TextDataFrame {
    /// Builds a text frame from any string-like value.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}
/// File payload frame.
///
/// Carries a file either inline (`bytes`) or by reference (`uri`); exactly one
/// of the two sources must be populated. `mime_type` keeps Python's
/// `mimeType` wire name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FileDataFrame {
    /// File name.
    pub name: String,
    /// MIME type, such as `"image/png"` (wire name `mimeType`).
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// Inline file content; mutually exclusive with [`uri`](Self::uri).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// File URI; mutually exclusive with [`bytes`](Self::bytes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}
impl FileDataFrame {
    /// Builds a file frame that references its content by URI.
    #[must_use]
    pub fn from_uri(
        name: impl Into<String>,
        mime_type: impl Into<String>,
        uri: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            mime_type: mime_type.into(),
            bytes: None,
            uri: Some(uri.into()),
        }
    }
    /// Builds a file frame that carries its content inline.
    #[must_use]
    pub fn from_bytes(
        name: impl Into<String>,
        mime_type: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            name: name.into(),
            mime_type: mime_type.into(),
            bytes: Some(bytes),
            uri: None,
        }
    }
    /// Checks that exactly one of `bytes` and `uri` is populated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when both or neither source is set,
    /// matching Python's `_exactly_one_source` model validator.
    pub fn validate(&self) -> Result<()> {
        if self.bytes.is_none() == self.uri.is_none() {
            return Err(Error::validation(
                "FileDataFrame requires exactly one of bytes or uri",
            ));
        }
        Ok(())
    }
}
/// JSON payload frame.
///
/// Carries a structured object; non-object values are wrapped under the
/// `"value"` key so the frame stays a map.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JsonDataFrame {
    /// Structured payload, kept as an object.
    pub data: BTreeMap<String, Value>,
}
impl JsonDataFrame {
    /// Builds a JSON frame from an object map.
    #[must_use]
    pub fn new(data: BTreeMap<String, Value>) -> Self {
        Self { data }
    }
    /// Builds a JSON frame from an arbitrary value.
    ///
    /// Objects are stored as-is; every other value is wrapped under
    /// `"value"`.
    #[must_use]
    pub fn from_value(value: Value) -> Self {
        match value {
            Value::Object(map) => Self {
                data: map.into_iter().collect(),
            },
            other => Self {
                data: BTreeMap::from([("value".to_owned(), other)]),
            },
        }
    }
}
/// A controller payload frame.
///
/// The wire shape puts the variant discriminator in the `type` field:
/// `{"type":"text", …}`, `{"type":"file", …}`, or `{"type":"json", …}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataFrame {
    /// Plain text payload.
    Text(TextDataFrame),
    /// File payload.
    File(FileDataFrame),
    /// JSON object payload.
    Json(JsonDataFrame),
}
impl DataFrame {
    /// Builds a text frame.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextDataFrame::new(text))
    }
    /// Builds a JSON frame from an arbitrary value.
    #[must_use]
    pub fn json(value: Value) -> Self {
        Self::Json(JsonDataFrame::from_value(value))
    }
    /// Returns the wire discriminator: `"text"`, `"file"`, or `"json"`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::File(_) => "file",
            Self::Json(_) => "json",
        }
    }
    /// Renders the frame as text.
    ///
    /// Text frames borrow their content, JSON frames serialize their object,
    /// and file frames have no textual form.
    #[must_use]
    pub fn as_text(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::Text(frame) => Some(Cow::Borrowed(frame.text.as_str())),
            Self::Json(frame) => Some(Cow::Owned(
                serde_json::to_string(&frame.data).unwrap_or_default(),
            )),
            Self::File(_) => None,
        }
    }
    /// Validates the frame.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for a file frame without exactly one
    /// content source; text and JSON frames are always valid.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::File(frame) => frame.validate(),
            Self::Text(_) | Self::Json(_) => Ok(()),
        }
    }
}
/// Concatenates the text and JSON frames of `frames` into one string.
///
/// Parts are separated by `"\n"` and file frames are ignored; returns `None`
/// when no frame contributes text.
#[must_use]
pub fn frames_text(frames: &[DataFrame]) -> Option<String> {
    let parts: Vec<String> = frames
        .iter()
        .filter_map(|frame| frame.as_text().map(Cow::into_owned))
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn text_frame_wire_shape() {
        let value = serde_json::to_value(DataFrame::text("hi")).unwrap();
        assert_eq!(value, json!({"type": "text", "text": "hi"}));
        assert_eq!(
            serde_json::to_string(&DataFrame::text("hi")).unwrap(),
            r#"{"type":"text","text":"hi"}"#
        );
    }
    #[test]
    fn json_frame_round_trips() {
        let frame = JsonDataFrame::from_value(json!({"a": 1, "b": [true]}));
        let wire = serde_json::to_value(DataFrame::Json(frame.clone())).unwrap();
        assert_eq!(wire, json!({"type": "json", "data": {"a": 1, "b": [true]}}));
        let parsed: DataFrame = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, DataFrame::Json(frame));
    }
    #[test]
    fn json_frame_wraps_non_objects() {
        let frame = JsonDataFrame::from_value(json!("plain"));
        assert_eq!(frame.data.get("value"), Some(&json!("plain")));
        assert_eq!(DataFrame::json(json!([1, 2])).kind(), "json");
    }
    #[test]
    fn file_frame_requires_exactly_one_source() {
        assert!(
            FileDataFrame::from_uri("a.png", "image/png", "file:///a.png")
                .validate()
                .is_ok()
        );
        assert!(FileDataFrame::from_bytes("a.png", "image/png", vec![1, 2])
            .validate()
            .is_ok());
        assert!(FileDataFrame::default().validate().is_err());
        let both = FileDataFrame {
            bytes: Some(vec![1]),
            uri: Some("file:///a.png".to_owned()),
            ..FileDataFrame::default()
        };
        let error = both.validate().unwrap_err();
        assert!(error.to_string().contains("exactly one of bytes or uri"));
        assert!(DataFrame::File(both).validate().is_err());
    }
    #[test]
    fn frames_text_joins_text_and_json_ignoring_files() {
        let frames = vec![
            DataFrame::text("first"),
            DataFrame::File(FileDataFrame::from_uri("a", "text/plain", "u")),
            DataFrame::json(json!({"k": 1})),
        ];
        assert_eq!(frames_text(&frames), Some("first\n{\"k\":1}".to_owned()));
        assert_eq!(frames_text(&frames[1..2]), None);
        assert_eq!(frames_text(&[]), None);
    }
    #[test]
    fn file_frame_wire_names() {
        let wire = serde_json::to_value(DataFrame::File(FileDataFrame::from_uri(
            "a",
            "image/png",
            "u",
        )))
        .unwrap();
        assert_eq!(
            wire,
            json!({"type": "file", "name": "a", "mimeType": "image/png", "uri": "u"})
        );
    }
}
