//! Shared helpers for parsing model output.
//!
//! Language models rarely answer with bare JSON or bare values: they wrap
//! payloads in fenced code blocks and mark sections with XML-like tags. The
//! helpers here extract those payloads deterministically, without any
//! dependency beyond [`serde_json`] and [`tracing`].
//!
//! The behavior is ported from the Python
//! `openjiuwen/dev_tools/tune/utils.py` (`parse_json_from_llm_response`,
//! `parse_list_from_llm_response`) and the prompt-builder tag parsers.

use serde_json::Value;

/// A fenced code block: its info string and the trimmed body.
struct Fence<'a> {
    /// Text between the opening backticks and the end of that line.
    info: &'a str,
    /// The block body, trimmed of surrounding whitespace.
    body: &'a str,
}

/// Scan `text` for fenced code blocks delimited by at least three backticks.
///
/// An unclosed fence terminates the scan; its body is not returned.
fn fences(text: &str) -> Vec<Fence<'_>> {
    let mut blocks = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let line_end = text[pos..].find('\n').map_or(text.len(), |n| pos + n);
        let line = &text[pos..line_end];
        let trimmed = line.trim_start();
        let ticks = trimmed.chars().take_while(|c| *c == '`').count();
        if ticks < 3 {
            if line_end >= text.len() {
                break;
            }
            pos = line_end + 1;
            continue;
        }

        let info = trimmed[ticks..].trim();
        if line_end >= text.len() {
            break;
        }
        let body_start = line_end + 1;

        let mut scan = body_start;
        let mut close = None;
        while scan <= text.len() {
            let end = text[scan..].find('\n').map_or(text.len(), |n| scan + n);
            if text[scan..end].trim_start().starts_with("```") {
                close = Some((scan, end));
                break;
            }
            if end >= text.len() {
                break;
            }
            scan = end + 1;
        }

        let Some((close_start, close_end)) = close else {
            break;
        };
        blocks.push(Fence {
            info,
            body: text[body_start..close_start].trim(),
        });
        pos = if close_end < text.len() {
            close_end + 1
        } else {
            text.len()
        };
    }
    blocks
}

/// Extract the first fenced JSON payload from `text`.
///
/// The first fence whose info string starts with `json` (case-insensitive)
/// wins; its trimmed body is parsed with [`serde_json`]. Returns `None` when no
/// such fence exists or when its body is not valid JSON. Arrays are accepted —
/// the value is returned as-is.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::util::extract_json_fence;
/// let text = "Answer:\n```json\n{\"ok\": true}\n```";
/// assert_eq!(extract_json_fence(text).unwrap()["ok"], true);
/// ```
#[must_use]
pub fn extract_json_fence(text: &str) -> Option<Value> {
    for fence in fences(text) {
        if fence.info.to_ascii_lowercase().starts_with("json") {
            return match serde_json::from_str::<Value>(fence.body) {
                Ok(value) => Some(value),
                Err(error) => {
                    tracing::debug!(%error, "failed to parse JSON code fence");
                    None
                }
            };
        }
    }
    None
}

/// Extract the first fenced JSON list payload from `text`.
///
/// A fence qualifies when its info string starts with `list` or `json`
/// (case-insensitive). The body must parse as a JSON array; an object or any
/// other value yields `None`. Returns `None` when no qualifying fence exists.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::util::extract_list_fence;
/// let text = "```json\n[1, 2]\n```";
/// assert_eq!(extract_list_fence(text).unwrap().len(), 2);
/// ```
#[must_use]
pub fn extract_list_fence(text: &str) -> Option<Vec<Value>> {
    for fence in fences(text) {
        let info = fence.info.to_ascii_lowercase();
        if !(info.starts_with("list") || info.starts_with("json")) {
            continue;
        }
        return match serde_json::from_str::<Value>(fence.body) {
            Ok(Value::Array(items)) => Some(items),
            Ok(_) => {
                tracing::debug!("JSON code fence did not contain an array");
                None
            }
            Err(error) => {
                tracing::debug!(%error, "failed to parse list code fence");
                None
            }
        };
    }
    None
}

/// Extract the content of the last `<tag>...</tag>` element in `text`.
///
/// The tag name is matched case-sensitively and content may span multiple
/// lines; the returned string is trimmed. If the last opening tag is not
/// followed by a matching closing tag, `None` is returned. An empty `tag`
/// argument also yields `None`.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::util::extract_tag;
/// assert_eq!(extract_tag("<a> one </a>", "a").as_deref(), Some("one"));
/// ```
#[must_use]
pub fn extract_tag(text: &str, tag: &str) -> Option<String> {
    if tag.is_empty() {
        return None;
    }
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");

    let mut last_open = None;
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(&open) {
        let at = from + rel;
        last_open = Some(at);
        from = at + open.len();
    }

    let start = last_open? + open.len();
    let end = start + text[start..].find(&close)?;
    Some(text[start..end].trim().to_string())
}

/// Extract a boolean flag from the last `<tag>...</tag>` element in `text`.
///
/// The trimmed content is compared case-insensitively with `true` and `false`;
/// any other content, a missing tag, or an unclosed tag yields `None`.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::util::extract_bool_tag;
/// assert_eq!(extract_bool_tag("<intent> TRUE </intent>", "intent"), Some(true));
/// ```
#[must_use]
pub fn extract_bool_tag(text: &str, tag: &str) -> Option<bool> {
    let value = extract_tag(text, tag)?;
    match value.to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Render `value` as compact JSON and truncate it to at most `max_len` chars.
///
/// The result is measured in characters, not bytes, so multi-byte Unicode is
/// never split. When truncation happens the result ends with `"..."` and still
/// fits within `max_len`. A limit below four characters (the minimum needed for
/// the ellipsis) yields an empty string, as does a serialization failure.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::util::truncate_json;
/// let value = serde_json::json!({"key": "value"});
/// assert_eq!(truncate_json(&value, 100), "{\"key\":\"value\"}");
/// assert_eq!(truncate_json(&value, 7), "{\"ke...");
/// ```
#[must_use]
pub fn truncate_json(value: &Value, max_len: usize) -> String {
    if max_len < 4 {
        return String::new();
    }
    match serde_json::to_string(value) {
        Ok(serialized) => {
            if serialized.chars().count() <= max_len {
                return serialized;
            }
            let mut truncated: String = serialized.chars().take(max_len - 3).collect();
            truncated.push_str("...");
            truncated
        }
        Err(error) => {
            tracing::debug!(%error, "failed to serialize JSON value");
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_fence_with_language() {
        let text = "Here is the answer:\n```json\n{\"name\": \"demo\", \"n\": 2}\n```\nDone.";
        assert_eq!(
            extract_json_fence(text),
            Some(json!({"name": "demo", "n": 2}))
        );
    }

    #[test]
    fn json_fence_uppercase_info() {
        let text = "```JSON\n[1, 2, 3]\n```";
        assert_eq!(extract_json_fence(text), Some(json!([1, 2, 3])));
    }

    #[test]
    fn json_fence_first_of_two_wins() {
        let text = "```json\n{\"first\": true}\n```\ntext\n```json\n{\"second\": true}\n```";
        assert_eq!(extract_json_fence(text), Some(json!({"first": true})));
    }

    #[test]
    fn json_fence_requires_json_info() {
        let text = "```rust\nfn main() {}\n```\n```list\n[]\n```";
        assert_eq!(extract_json_fence(text), None);
    }

    #[test]
    fn json_fence_absent() {
        assert_eq!(extract_json_fence("no code here"), None);
        assert_eq!(extract_json_fence(""), None);
    }

    #[test]
    fn json_fence_malformed() {
        let text = "```json\n{not: valid}\n```";
        assert_eq!(extract_json_fence(text), None);
    }

    #[test]
    fn list_fence_from_json_array() {
        let text = "```json\n[{\"id\": 1}, {\"id\": 2}]\n```";
        let items = extract_list_fence(text).expect("array should parse");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], json!({"id": 1}));
    }

    #[test]
    fn list_fence_accepts_list_info() {
        let text = "```list\n[\"a\", \"b\"]\n```";
        assert_eq!(extract_list_fence(text), Some(vec![json!("a"), json!("b")]));
    }

    #[test]
    fn list_fence_rejects_object() {
        let text = "```json\n{\"not\": \"an array\"}\n```";
        assert_eq!(extract_list_fence(text), None);
    }

    #[test]
    fn tag_multiline() {
        let text = "prefix\n<intent>\n  create a file\n  and run it\n</intent>\nsuffix";
        assert_eq!(
            extract_tag(text, "intent").as_deref(),
            Some("create a file\n  and run it")
        );
    }

    #[test]
    fn tag_last_occurrence_wins() {
        let text = "<plan>first</plan> middle <plan>second</plan>";
        assert_eq!(extract_tag(text, "plan").as_deref(), Some("second"));
    }

    #[test]
    fn tag_missing_or_unclosed() {
        assert_eq!(extract_tag("no tags", "plan"), None);
        assert_eq!(extract_tag("<plan>unclosed", "plan"), None);
        assert_eq!(extract_tag("", "plan"), None);
    }

    #[test]
    fn tag_empty_name_rejected() {
        assert_eq!(extract_tag("<x>value</x>", ""), None);
    }

    #[test]
    fn bool_tag_parses_case_insensitively() {
        let text = "<intent> TRUE </intent>\n<intent> False </intent>";
        assert_eq!(extract_bool_tag(text, "intent"), Some(false));
        assert_eq!(extract_bool_tag("<flag> true </flag>", "flag"), Some(true));
        assert_eq!(extract_bool_tag("<flag> yes </flag>", "flag"), None);
        assert_eq!(extract_bool_tag("no tag", "flag"), None);
    }

    #[test]
    fn truncate_json_short_value_is_unchanged() {
        let value = json!({"key": "value"});
        assert_eq!(truncate_json(&value, 100), "{\"key\":\"value\"}");
        assert_eq!(truncate_json(&value, 15), "{\"key\":\"value\"}");
    }

    #[test]
    fn truncate_json_honors_length_bound() {
        let value = json!({"key": "value"});
        let truncated = truncate_json(&value, 7);
        assert_eq!(truncated, "{\"ke...");
        assert!(truncated.chars().count() <= 7);
        assert_eq!(truncate_json(&value, 4), "{...");
    }

    #[test]
    fn truncate_json_is_unicode_safe() {
        let value = json!({"text": "汉字汉字汉字"});
        // The serialized form is `{"text":"汉字汉字汉字"}`; a limit of 13 keeps
        // the first Chinese character whole and never slices mid-codepoint.
        let truncated = truncate_json(&value, 13);
        assert!(truncated.chars().count() <= 13);
        assert!(truncated.ends_with("..."));
        assert_eq!(truncated, "{\"text\":\"汉...");
    }

    #[test]
    fn truncate_json_tiny_limit_is_empty() {
        let value = json!({"key": "value"});
        assert_eq!(truncate_json(&value, 3), "");
        assert_eq!(truncate_json(&value, 0), "");
    }
}
