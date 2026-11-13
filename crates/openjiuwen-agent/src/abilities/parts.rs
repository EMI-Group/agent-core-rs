//! Multimodal content parts of a tool result.
//!
//! A tool may answer with text, images, audio, or files. [`extract_parts`]
//! recognises the JSON shapes a tool uses and normalises them into the
//! foundation's [`ContentPart`] model, so the loop can attach them to the
//! observation and to the tool message without a second content model.
//! [`part_summary`] and [`render_with_parts`] give a provider that cannot
//! receive parts a short, readable placeholder instead of dropping the payload.
use openjiuwen_foundation::{ContentPart, FilePart};
use serde_json::{json, Map, Value};

/// Extracts the multimodal content parts of a tool result.
///
/// Recognised shapes:
///
/// - `{"content_parts": [...]}`;
/// - `{"content": [...]}`, and `{"content": {"type": "parts", "parts": [...]}}`;
/// - a bare top-level array of part objects (or bare strings);
/// - a foundation-content-shaped object: `{"type": "parts", "parts": [...]}` or
///   `{"type": "text", "text": "..."}`.
///
/// Each element is normalised and converted through [`ContentPart`]'s
/// deserializer: a JSON string becomes [`ContentPart::Text`], an object tagged
/// `text`/`image_url`/`audio`/`file` keeps its typed shape, and an object tagged
/// `image` is rewritten to the `image_url` spelling first. Scalars, `null`, plain
/// strings, unsupported tags, and parts that do not deserialize are skipped
/// rather than reported, and the input order is preserved.
#[must_use]
pub fn extract_parts(value: &Value) -> Vec<ContentPart> {
    match value {
        Value::Array(items) => parts_of(items),
        Value::Object(map) => {
            if let Some(items) = map.get("content_parts").and_then(Value::as_array) {
                return parts_of(items);
            }
            if let Some(content) = map.get("content") {
                match content {
                    Value::Array(items) => return parts_of(items),
                    Value::Object(inner) => return object_parts(inner),
                    _ => {}
                }
            }
            object_parts(map)
        }
        _ => Vec::new(),
    }
}

/// A short, model-readable placeholder for one content part.
///
/// Text keeps its own text; an image reports its media type for a `data:` URI
/// or else its file name; audio reports its container format; a file reports its
/// name, media type, or remote file name; any other part reports its type tag.
#[must_use]
pub fn part_summary(part: &ContentPart) -> String {
    match part {
        ContentPart::Text { text } => text.clone(),
        ContentPart::ImageUrl { image_url } => image_summary(&image_url.url),
        ContentPart::Audio(audio) => audio_summary(audio.format.as_deref()),
        ContentPart::File(file) => file_summary(file),
        ContentPart::Other(_) => format!("[{}]", part.part_type()),
    }
}

/// Renders `text` followed by one placeholder line per non-text part.
///
/// This is the fallback for a provider that cannot receive content parts: the
/// text parts are skipped because their text is already in `text`, empty
/// summaries are skipped, and the result carries no trailing newline.
#[must_use]
pub fn render_with_parts(text: &str, parts: &[ContentPart]) -> String {
    let mut lines = Vec::with_capacity(parts.len() + 1);
    if !text.is_empty() {
        lines.push(text.to_owned());
    }
    for part in parts {
        if part.text().is_some() {
            continue;
        }
        let summary = part_summary(part);
        if summary.is_empty() {
            continue;
        }
        lines.push(summary);
    }
    lines.join("\n")
}

/// Converts every normalisable element of `items`, skipping the rest.
fn parts_of(items: &[Value]) -> Vec<ContentPart> {
    items.iter().filter_map(normalise).collect()
}

/// Interprets an object that carries the parts of a message content.
///
/// `{"type": "parts", "parts": [...]}` holds a list; any other object is one
/// content part of its own.
fn object_parts(map: &Map<String, Value>) -> Vec<ContentPart> {
    if map.get("type").and_then(Value::as_str) == Some("parts") {
        return map
            .get("parts")
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |items| parts_of(items));
    }
    normalise(&Value::Object(map.clone())).into_iter().collect()
}

/// Normalises one element into a supported [`ContentPart`].
///
/// Returns `None` for anything that is not a string or object, and for a part
/// that stays unmodelled (an unknown or payload-less tag).
fn normalise(value: &Value) -> Option<ContentPart> {
    let candidate = match value {
        Value::String(_) => value.clone(),
        Value::Object(map) => match map.get("type").and_then(Value::as_str) {
            Some("image") => image_part(map)?,
            _ => value.clone(),
        },
        _ => return None,
    };
    match serde_json::from_value::<ContentPart>(candidate).ok()? {
        ContentPart::Other(_) => None,
        part => Some(part),
    }
}

/// Rewrites an OpenAI-style `{"type": "image", ...}` object to `image_url`.
///
/// The payload is the object's own `image_url` field (an object or a URL
/// string), else its `url` string; an object carrying neither is skipped.
fn image_part(map: &Map<String, Value>) -> Option<Value> {
    let payload = match map.get("image_url") {
        Some(Value::Object(_)) => map.get("image_url")?.clone(),
        Some(Value::String(url)) => json!({"url": url}),
        _ => match map.get("url") {
            Some(Value::String(url)) => json!({"url": url}),
            _ => return None,
        },
    };
    Some(json!({"type": "image_url", "image_url": payload}))
}

/// The media type of a `data:<mime>;base64,` URI.
fn data_uri_mime(url: &str) -> Option<&str> {
    let (mime, _) = url.strip_prefix("data:")?.split_once(";base64,")?;
    (!mime.is_empty()).then_some(mime)
}

/// The last non-empty `/`-separated segment of `url`.
fn last_path_segment(url: &str) -> Option<&str> {
    url.split('/').rev().find(|segment| !segment.is_empty())
}

/// Placeholder for an image reference.
fn image_summary(url: &str) -> String {
    if let Some(mime) = data_uri_mime(url) {
        return format!("[image: {mime}]");
    }
    last_path_segment(url).map_or_else(|| "[image]".to_owned(), |name| format!("[image: {name}]"))
}

/// Placeholder for an audio block.
fn audio_summary(format: Option<&str>) -> String {
    match format.filter(|format| !format.is_empty()) {
        Some(format) if format.contains('/') => format!("[audio: {format}]"),
        Some(format) => format!("[audio: audio/{format}]"),
        None => "[audio]".to_owned(),
    }
}

/// Placeholder for a file block.
fn file_summary(file: &FilePart) -> String {
    if let Some(name) = file.filename.as_deref().filter(|name| !name.is_empty()) {
        return format!("[file: {name}]");
    }
    if let Some(mime) = file.mime_type.as_deref().filter(|mime| !mime.is_empty()) {
        return format!("[file: {mime}]");
    }
    if let Some(name) = file.url.as_deref().and_then(last_path_segment) {
        return format!("[file: {name}]");
    }
    "[file]".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::{AudioPart, ImageUrl};

    fn image(url: &str) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.to_owned(),
                detail: None,
            },
        }
    }

    fn text(value: &str) -> ContentPart {
        ContentPart::Text {
            text: value.to_owned(),
        }
    }

    #[test]
    fn content_parts_key_is_read() {
        let value = json!({
            "content_parts": [
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ],
        });
        assert_eq!(
            extract_parts(&value),
            vec![text("ok"), image("data:image/png;base64,AAA")]
        );
    }

    #[test]
    fn content_array_is_read() {
        let value = json!({
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}},
            ],
        });
        assert_eq!(
            extract_parts(&value),
            vec![text("ok"), image("https://example.test/cat.png")]
        );
    }

    #[test]
    fn bare_array_of_strings_and_objects_is_read() {
        let value = json!([
            "plain",
            {"type": "text", "text": "tagged"},
            {"type": "audio", "audio": {"data": "AAA", "format": "wav"}},
        ]);
        let parts = extract_parts(&value);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], text("plain"));
        assert_eq!(parts[1], text("tagged"));
        assert_eq!(
            parts[2],
            ContentPart::Audio(AudioPart {
                data: "AAA".to_owned(),
                format: Some("wav".to_owned()),
            })
        );
    }

    #[test]
    fn foundation_content_objects_are_unwrapped() {
        let parts = json!([{"type": "text", "text": "inner"}]);
        assert_eq!(
            extract_parts(&json!({"type": "parts", "parts": parts})),
            vec![text("inner")]
        );
        assert_eq!(
            extract_parts(&json!({"content": {"type": "parts", "parts": parts}})),
            vec![text("inner")]
        );
        assert_eq!(
            extract_parts(&json!({"type": "text", "text": "hi"})),
            vec![text("hi")]
        );
        assert_eq!(
            extract_parts(&json!({"content": {"type": "text", "text": "hi"}})),
            vec![text("hi")]
        );
    }

    #[test]
    fn image_tag_is_normalised_to_image_url() {
        let absolute = json!([{"type": "image", "image_url": {"url": "https://x.test/a.png"}}]);
        assert_eq!(
            extract_parts(&absolute),
            vec![image("https://x.test/a.png")]
        );

        let nested_string = json!([{"type": "image", "image_url": "https://x.test/b.png"}]);
        assert_eq!(
            extract_parts(&nested_string),
            vec![image("https://x.test/b.png")]
        );

        let flat_url = json!([{"type": "image", "url": "data:image/png;base64,AAA"}]);
        assert_eq!(
            extract_parts(&flat_url),
            vec![image("data:image/png;base64,AAA")]
        );
    }

    #[test]
    fn malformed_entries_are_skipped() {
        let value = json!([
            42,
            null,
            true,
            {"type": "image_url"},
            {"type": "image"},
            {"type": "video", "video": {"url": "https://x.test/v.mp4"}},
            {"type": "text", "text": "kept"},
        ]);
        assert_eq!(extract_parts(&value), vec![text("kept")]);
    }

    #[test]
    fn scalars_strings_and_plain_text_content_have_no_parts() {
        for value in [
            json!(null),
            json!(7),
            json!(true),
            json!("plain text"),
            json!({"content": "plain text"}),
            json!({}),
        ] {
            assert!(
                extract_parts(&value).is_empty(),
                "`{value}` must not produce content parts"
            );
        }
    }

    #[test]
    fn part_summary_describes_every_part_kind() {
        assert_eq!(part_summary(&text("hello")), "hello");
        assert_eq!(
            part_summary(&image("data:image/png;base64,AAA")),
            "[image: image/png]"
        );
        assert_eq!(
            part_summary(&image("https://example.test/shots/cat.png")),
            "[image: cat.png]"
        );
        assert_eq!(part_summary(&image("")), "[image]");
        assert_eq!(
            part_summary(&ContentPart::Audio(AudioPart {
                data: String::new(),
                format: Some("wav".to_owned()),
            })),
            "[audio: audio/wav]"
        );
        assert_eq!(
            part_summary(&ContentPart::Audio(AudioPart {
                data: String::new(),
                format: Some("audio/mpeg".to_owned()),
            })),
            "[audio: audio/mpeg]"
        );
        assert_eq!(
            part_summary(&ContentPart::Audio(AudioPart {
                data: String::new(),
                format: None,
            })),
            "[audio]"
        );
    }

    #[test]
    fn part_summary_describes_files_by_name_mime_or_url() {
        let named = ContentPart::File(FilePart {
            url: None,
            data: None,
            mime_type: Some("application/pdf".to_owned()),
            filename: Some("report.pdf".to_owned()),
        });
        assert_eq!(part_summary(&named), "[file: report.pdf]");
        let typed = ContentPart::File(FilePart {
            url: None,
            data: None,
            mime_type: Some("application/pdf".to_owned()),
            filename: None,
        });
        assert_eq!(part_summary(&typed), "[file: application/pdf]");
        let hosted = ContentPart::File(FilePart {
            url: Some("https://example.test/files/report.csv".to_owned()),
            data: None,
            mime_type: None,
            filename: None,
        });
        assert_eq!(part_summary(&hosted), "[file: report.csv]");
        let bare = ContentPart::File(FilePart {
            url: None,
            data: None,
            mime_type: None,
            filename: None,
        });
        assert_eq!(part_summary(&bare), "[file]");
    }

    #[test]
    fn part_summary_uses_the_type_tag_of_other_parts() {
        let video = ContentPart::Other(json!({"type": "video", "video": {"url": "x"}}));
        assert_eq!(part_summary(&video), "[video]");
        let untagged = ContentPart::Other(json!({"url": "x"}));
        assert_eq!(part_summary(&untagged), "[other]");
    }

    #[test]
    fn render_with_parts_appends_non_text_summaries() {
        let parts = vec![
            text("ok"),
            image("data:image/png;base64,AAA"),
            ContentPart::File(FilePart {
                url: None,
                data: None,
                mime_type: None,
                filename: Some("report.pdf".to_owned()),
            }),
        ];
        assert_eq!(
            render_with_parts("ok", &parts),
            "ok\n[image: image/png]\n[file: report.pdf]"
        );
        assert_eq!(
            render_with_parts("", &parts),
            "[image: image/png]\n[file: report.pdf]"
        );
        assert_eq!(render_with_parts("only text", &[]), "only text");
        assert_eq!(render_with_parts("", &[]), "");
    }
}
