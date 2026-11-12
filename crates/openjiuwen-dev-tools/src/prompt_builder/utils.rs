//! Deterministic helpers for the prompt builders.
//!
//! Ports the pure parts of `openjiuwen/dev_tools/prompt_builder/builder/`:
//!
//! - `utils.py::get_string_prompt` — flatten a [`PromptTemplate`] to plain text.
//! - `feedback_prompt_builder.py` — the char-indexed insert/select splicing of
//!   [`INSERT_STR`] and the feedback-intent parser
//!   (`_extract_intent_from_responses`).
//! - `badcase_prompt_builder.py::_build_bad_case_string` — render evaluated
//!   cases through the bank's `format_bad_case` text, and
//!   `_parse_feedback_summary` — read the `<intent>`/`<summary>` tags.
//!
//! Nothing here calls a model: every function is a pure string transform, so
//! the model-driven builders keep only the call sequence.
//!
//! # Deviations from Python
//!
//! - Positions and ranges are **character** indices, matching Python's `str`
//!   slicing. Rust string slicing is byte-indexed, so the helpers convert char
//!   indices to byte offsets and never split a multi-byte codepoint.
//! - Bad cases render their maps as compact JSON (with sorted keys) instead of
//!   Python's `str(dict)` repr; this is deterministic and re-parseable.
//! - Invalid input (an out-of-range insert position, an inverted range, a
//!   non-text message content) is an [`Error::Validation`] instead of a silent
//!   fallback or a Python `TypeError`.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{PromptTemplate, TemplateContent};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::templates::PromptBank;
use super::{INSERT_STR, JSON_STRING_MAX_LENGTH};
use crate::tune::EvaluatedCase;
use crate::util::{extract_bool_tag, extract_json_fence, extract_tag, truncate_json};

/// The byte offset of the character at `char_index`, or the text length.
///
/// Index `0` is the first character and `char_len` (one past the last character)
/// is the string length, mirroring Python's slice semantics.
fn byte_offset(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map_or(text.len(), |(byte, _)| byte)
}

/// Flattens a prompt template to plain text.
///
/// Ports `get_string_prompt`: a text template is returned verbatim (its
/// `{{placeholders}}` are *not* rendered or trimmed), and a message-list
/// template joins the text content of its messages with `"\n"` in order. An
/// empty message list yields `""`.
///
/// Python falls back to `"\n".join(item.values())` when the content is neither a
/// string nor a list of messages; here such content is rejected instead.
///
/// # Errors
///
/// [`Error::Validation`] when a message carries
/// [`MessageContent::Parts`](openjiuwen_foundation::prompt::MessageContent::Parts)
/// instead of text; the message names the offending index.
pub fn get_string_prompt(template: &PromptTemplate) -> Result<String> {
    match &template.content {
        TemplateContent::Text(text) => Ok(text.clone()),
        TemplateContent::Messages(messages) => {
            let mut texts = Vec::with_capacity(messages.len());
            for (index, message) in messages.iter().enumerate() {
                let Some(text) = message.content.as_text() else {
                    return Err(Error::validation(format!(
                        "prompt message at index {index} has non-text content; only text \
                         messages can be flattened to a plain string"
                    )));
                };
                texts.push(text.to_owned());
            }
            Ok(texts.join("\n"))
        }
    }
}

/// Splices [`INSERT_STR`] into `prompt` at character index `position`.
///
/// Ports `_insert_sting`: `0` prefixes the marker, `position == char_len`
/// appends it, and any position in between splits the prompt there. The index is
/// a character index, so a multi-byte prompt is never corrupted.
///
/// # Errors
///
/// [`Error::Validation`] when `position` is past the end of the prompt; the
/// message reports both the position and the character length.
pub fn insert_marker(prompt: &str, position: usize) -> Result<String> {
    let char_len = prompt.chars().count();
    if position > char_len {
        return Err(Error::validation(format!(
            "insert position {position} is out of bounds for a prompt of {char_len} characters; \
             requires 0 <= position <= {char_len}"
        )));
    }

    let byte = byte_offset(prompt, position);
    let mut spliced = String::with_capacity(prompt.len() + INSERT_STR.len());
    spliced.push_str(&prompt[..byte]);
    spliced.push_str(INSERT_STR);
    spliced.push_str(&prompt[byte..]);
    Ok(spliced)
}

/// Returns the characters in `start..end` of `prompt`.
///
/// Ports the `prompt[start_pos:end_pos]` slice used by select-mode feedback.
/// Both bounds are character indices and the range is half-open.
///
/// # Errors
///
/// [`Error::Validation`] unless `0 <= start < end <= len`; the message states
/// the rule. Equal, inverted, and past-the-end ranges are all rejected.
pub fn select_range(prompt: &str, start: usize, end: usize) -> Result<String> {
    let char_len = prompt.chars().count();
    if start >= end || end > char_len {
        return Err(Error::validation(format!(
            "select range {start}..{end} is invalid for a prompt of {char_len} characters; \
             requires 0 <= start < end <= {char_len}"
        )));
    }

    let start_byte = byte_offset(prompt, start);
    let end_byte = byte_offset(prompt, end);
    Ok(prompt[start_byte..end_byte].to_owned())
}

/// Builds a JSON object value from a borrowed map, cloning its entries.
///
/// Python renders the whole `dict` (`str(case.case.inputs)`); JSON is the
/// documented deviation and keeps a deterministic key order.
fn object_value(map: &BTreeMap<String, Value>) -> Value {
    let object: Map<String, Value> = map
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Value::Object(object)
}

/// Renders one evaluated case's fields as the compact, truncated JSON string.
fn json_field(map: &BTreeMap<String, Value>) -> Value {
    Value::String(truncate_json(&object_value(map), JSON_STRING_MAX_LENGTH))
}

/// Renders `cases` through the bank's `format_bad_case` text, joined by `"\n"`.
///
/// Ports `_build_bad_case_string`. Each case fills the four placeholders:
///
/// - `question` — compact JSON of [`Case::inputs`](crate::tune::Case::inputs);
/// - `label` — compact JSON of [`Case::label`](crate::tune::Case::label);
/// - `answer` — compact JSON of [`EvaluatedCase::answer`];
/// - `reason` — [`EvaluatedCase::reason`] verbatim (Python does not truncate it).
///
/// JSON values are truncated to [`JSON_STRING_MAX_LENGTH`] characters. An empty
/// slice renders `""`; the builder validates the case count separately.
///
/// # Errors
///
/// [`Error::Validation`] if a rendered template cannot be flattened (the bank's
/// text is a plain string template, so this never happens in practice).
pub fn render_bad_cases(cases: &[EvaluatedCase], bank: &PromptBank) -> Result<String> {
    let template = PromptTemplate::new(bank.format_bad_case);
    let mut rendered = Vec::with_capacity(cases.len());
    for case in cases {
        let values: BTreeMap<String, Value> = BTreeMap::from([
            ("question".to_owned(), json_field(&case.case.inputs)),
            ("label".to_owned(), json_field(&case.case.label)),
            ("answer".to_owned(), json_field(&case.answer)),
            ("reason".to_owned(), Value::String(case.reason.clone())),
        ]);
        rendered.push(get_string_prompt(&template.format(&values))?);
    }
    Ok(rendered.join("\n"))
}

/// The parsed intent block of a feedback-recognition answer.
///
/// Ports the `(intent, optimized_feedback)` tuple of
/// `_extract_intent_from_responses`, extended with the optional
/// `optimization_directions` list the feedback prompt may return.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FeedbackIntent {
    /// `"true"` when the feedback is a valid optimization request, `"false"`
    /// when the model explicitly rejects it, `None` when the key is absent or
    /// holds another value; see [`parse_feedback_intent`].
    pub intent: Option<String>,
    /// The normalized feedback with blanks trimmed; `None` when absent or not a
    /// string.
    pub optimized_feedback: Option<String>,
    /// Optional optimization hints; always a list, empty when absent.
    pub optimization_directions: Vec<String>,
}

/// Normalizes the `intent` JSON value to `"true"`/`"false"`.
///
/// Ports Python's `parsed_json.get("intent", False) in ("true", True, "True")`
/// membership test, which also admits a `false` flag (normalized to `"false"`
/// here so the caller can tell "rejected" from "absent").
fn normalize_intent(value: Option<&Value>) -> Option<String> {
    let value = value?;
    match value {
        Value::Bool(flag) => Some(flag.to_string()),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" => Some("true".to_owned()),
            "false" => Some("false".to_owned()),
            _ => None,
        },
        _ => None,
    }
}

/// Normalizes `optimization_directions` to a list of strings.
///
/// An array contributes its string entries (non-strings are skipped); a bare
/// string is accepted as a one-element list; anything absent or of another type
/// yields an empty list.
fn normalize_directions(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        Some(Value::String(text)) => vec![text.clone()],
        _ => Vec::new(),
    }
}

/// Parses a feedback-recognition answer into a [`FeedbackIntent`].
///
/// Ports `_extract_intent_from_responses`: the first ` ```json ` fenced payload
/// is parsed, then `intent`, `optimized_feedback` (trimmed), and
/// `optimization_directions` are read from the object. Returns `None` when there
/// is no JSON fence or when its payload is not a JSON object.
#[must_use]
pub fn parse_feedback_intent(text: &str) -> Option<FeedbackIntent> {
    let value = extract_json_fence(text)?;
    let Value::Object(object) = value else {
        return None;
    };

    Some(FeedbackIntent {
        intent: normalize_intent(object.get("intent")),
        optimized_feedback: object
            .get("optimized_feedback")
            .and_then(Value::as_str)
            .map(|feedback| feedback.trim().to_owned()),
        optimization_directions: normalize_directions(object.get("optimization_directions")),
    })
}

/// Extracts the intent flag and feedback summary from an analysis answer.
///
/// Ports `_parse_feedback_summary`. The flag comes from the last
/// `<intent>true|false</intent>` tag and defaults to `false`. The summary is the
/// content of the **last** `<summary>...</summary>` tag, trimmed; when no such
/// tag exists the whole text, trimmed, is returned.
#[must_use]
pub fn parse_bad_case_analysis(text: &str) -> (bool, String) {
    let intent = extract_bool_tag(text, "intent").unwrap_or(false);
    let summary = extract_tag(text, "summary").unwrap_or_else(|| text.trim().to_owned());
    (intent, summary)
}

#[cfg(test)]
mod tests {
    use openjiuwen_foundation::prompt::{BaseMessage, ContentPart, MessageContent};

    use super::*;
    use crate::prompt_builder::language::Language;
    use crate::prompt_builder::templates::bank;
    use crate::tune::Case;

    /// A map from `(key, value)` pairs.
    fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    /// An evaluated case with a fixed id and a neutral score.
    fn evaluated(
        inputs: BTreeMap<String, Value>,
        label: BTreeMap<String, Value>,
        answer: BTreeMap<String, Value>,
        reason: &str,
    ) -> EvaluatedCase {
        let case = Case::new(inputs, label)
            .expect("valid case")
            .with_case_id("case-1");
        EvaluatedCase::new(case, answer, 0.5, reason).expect("valid evaluated case")
    }

    /// A small, fully specified evaluated case.
    fn sample() -> EvaluatedCase {
        evaluated(
            map(&[("question", Value::String("hi".to_owned()))]),
            map(&[("answer", Value::String("hello".to_owned()))]),
            map(&[("result", Value::String("nope".to_owned()))]),
            "wrong answer",
        )
    }

    #[test]
    fn flattens_text_template_verbatim() {
        let template = PromptTemplate::new("\n{{name}}\n");
        assert_eq!(
            get_string_prompt(&template).expect("text template"),
            "\n{{name}}\n"
        );
    }

    #[test]
    fn flattens_messages_joined_by_newline() {
        let template = PromptTemplate::new(vec![
            BaseMessage::system("first\nline"),
            BaseMessage::user("second"),
        ]);
        assert_eq!(
            get_string_prompt(&template).expect("message template"),
            "first\nline\nsecond"
        );
    }

    #[test]
    fn flattens_empty_message_list_to_empty_string() {
        let template = PromptTemplate::new(Vec::<BaseMessage>::new());
        assert_eq!(get_string_prompt(&template).expect("empty list"), "");
    }

    #[test]
    fn rejects_non_text_message_content() {
        let parts = MessageContent::Parts(vec![ContentPart::Text {
            text: "not a plain string".to_owned(),
        }]);
        let template =
            PromptTemplate::new(vec![BaseMessage::user("fine"), BaseMessage::user(parts)]);
        let error = get_string_prompt(&template).expect_err("parts content is unsupported");
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("index 1")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn inserts_at_the_start() {
        assert_eq!(
            insert_marker("hello", 0).expect("valid position"),
            format!("{INSERT_STR}hello")
        );
    }

    #[test]
    fn inserts_in_the_middle() {
        assert_eq!(
            insert_marker("hello", 2).expect("valid position"),
            format!("he{INSERT_STR}llo")
        );
    }

    #[test]
    fn inserts_at_the_end() {
        assert_eq!(
            insert_marker("hello", 5).expect("valid position"),
            format!("hello{INSERT_STR}")
        );
    }

    #[test]
    fn rejects_a_position_beyond_the_end() {
        let error = insert_marker("hello", 6).expect_err("position past the end");
        assert!(
            matches!(&error, Error::Validation(message) if message.contains('6') && message.contains('5')),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn inserts_into_an_empty_prompt() {
        assert_eq!(
            insert_marker("", 0).expect("valid position"),
            INSERT_STR.to_owned()
        );
    }

    #[test]
    fn inserts_by_character_index_in_multibyte_text() {
        // `你好世界` has 4 chars / 12 bytes: char index 2 is byte offset 6, so a
        // byte-indexed splice at 2 would panic or corrupt the text.
        assert_eq!(
            insert_marker("你好世界", 2).expect("valid position"),
            "你好[用户要插入的位置]世界"
        );
        // Char index 1 is byte offset 3 — not a byte boundary of the whole
        // string either; the helper must use char indices throughout.
        assert_eq!(
            insert_marker("你好世界", 1).expect("valid position"),
            "你[用户要插入的位置]好世界"
        );
    }

    #[test]
    fn selects_a_substring() {
        assert_eq!(
            select_range("hello world", 0, 5).expect("valid range"),
            "hello"
        );
        assert_eq!(
            select_range("hello world", 6, 11).expect("valid range"),
            "world"
        );
    }

    #[test]
    fn selects_the_full_range() {
        assert_eq!(select_range("hello", 0, 5).expect("valid range"), "hello");
        assert!(
            select_range("", 0, 0).is_err(),
            "an empty range is always invalid"
        );
    }

    #[test]
    fn selects_by_character_index_in_multibyte_text() {
        assert_eq!(select_range("你好世界", 1, 3).expect("valid range"), "好世");
        assert_eq!(
            select_range("你好世界", 0, 4).expect("valid range"),
            "你好世界"
        );
    }

    #[test]
    fn rejects_invalid_ranges() {
        for (start, end) in [(1, 1), (3, 1), (0, 6), (5, 6)] {
            let error = select_range("hello", start, end).expect_err("invalid range");
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("0 <= start < end")),
                "unexpected error for {start}..{end}: {error}"
            );
        }
        assert!(
            select_range("你好世界", 1, 5).is_err(),
            "end past the char length"
        );
    }

    #[test]
    fn renders_two_cases_joined_by_one_newline() {
        let bank = bank(Language::ZhCn);
        let second = evaluated(
            map(&[("question", Value::String("bye".to_owned()))]),
            map(&[("answer", Value::String("goodbye".to_owned()))]),
            map(&[("result", Value::String("later".to_owned()))]),
            "off by one",
        );
        let first_text = render_bad_cases(&[sample()], bank).expect("rendered");
        let second_text = render_bad_cases(std::slice::from_ref(&second), bank).expect("rendered");
        let joined = render_bad_cases(&[sample(), second], bank).expect("rendered");

        assert_eq!(joined, format!("{first_text}\n{second_text}"));
        assert_eq!(joined.matches("[question]").count(), 2);
        assert!(
            joined.contains("=== \n\n\n[question]"),
            "cases are separated by exactly one newline: {joined}"
        );
    }

    #[test]
    fn renders_every_placeholder() {
        let bank = bank(Language::ZhCn);
        let rendered = render_bad_cases(&[sample()], bank).expect("rendered");
        assert_eq!(
            rendered,
            "\n[question]: {\"question\":\"hi\"}\n[expected answer]: {\"answer\":\"hello\"}\n\
             [assistant answer]: {\"result\":\"nope\"}\n[reason]: wrong answer\n=== \n"
        );
        assert!(!rendered.contains("{{"), "no placeholder may survive");
    }

    #[test]
    fn truncates_long_json_values() {
        let bank = bank(Language::ZhCn);
        let long = "x".repeat(JSON_STRING_MAX_LENGTH + 100);
        let case = evaluated(
            map(&[("question", Value::String(long))]),
            map(&[("answer", Value::String("ok".to_owned()))]),
            BTreeMap::new(),
            "",
        );
        let rendered = render_bad_cases(&[case], bank).expect("rendered");

        let question_line = rendered
            .lines()
            .find(|line| line.starts_with("[question]: "))
            .expect("question line");
        let question = question_line.trim_start_matches("[question]: ");
        assert!(question.ends_with("..."), "truncation is marked");
        assert!(
            question.chars().count() <= JSON_STRING_MAX_LENGTH,
            "truncated to at most {JSON_STRING_MAX_LENGTH} chars, got {}",
            question.chars().count()
        );
    }

    #[test]
    fn renders_no_cases_as_an_empty_string() {
        assert_eq!(
            render_bad_cases(&[], bank(Language::EnUs)).expect("rendered"),
            ""
        );
    }

    #[test]
    fn parses_a_fenced_feedback_intent() {
        let text = "Here:\n```json\n{\n  \"intent\": \"true\",\n  \"optimized_feedback\": \
                    \"  be concise  \",\n  \"optimization_directions\": [\"shorten\", 3, \
                    \"clarify\"]\n}\n```\nDone.";
        let intent = parse_feedback_intent(text).expect("parsed");
        assert_eq!(
            intent,
            FeedbackIntent {
                intent: Some("true".to_owned()),
                optimized_feedback: Some("be concise".to_owned()),
                optimization_directions: vec!["shorten".to_owned(), "clarify".to_owned()],
            }
        );
    }

    #[test]
    fn feedback_intent_requires_a_json_fence() {
        assert_eq!(parse_feedback_intent("no code fence"), None);
        assert_eq!(parse_feedback_intent(""), None);
        assert_eq!(
            parse_feedback_intent("```text\n{\"intent\": \"true\"}\n```"),
            None
        );
    }

    #[test]
    fn feedback_intent_rejects_malformed_json() {
        assert_eq!(
            parse_feedback_intent("```json\n{not: valid}\n```"),
            None,
            "malformed payload"
        );
    }

    #[test]
    fn feedback_intent_rejects_non_object_payloads() {
        assert_eq!(parse_feedback_intent("```json\n[true, false]\n```"), None);
        assert_eq!(parse_feedback_intent("```json\n\"intent\"\n```"), None);
    }

    #[test]
    fn feedback_intent_normalizes_bool_and_string_flags() {
        let bool_true = parse_feedback_intent("```json\n{\"intent\": true}\n```").expect("parsed");
        assert_eq!(bool_true.intent.as_deref(), Some("true"));

        let capitalized =
            parse_feedback_intent("```json\n{\"intent\": \"True\"}\n```").expect("parsed");
        assert_eq!(capitalized.intent.as_deref(), Some("true"));

        let rejected =
            parse_feedback_intent("```json\n{\"intent\": \"false\"}\n```").expect("parsed");
        assert_eq!(rejected.intent.as_deref(), Some("false"));

        let unknown =
            parse_feedback_intent("```json\n{\"intent\": \"maybe\"}\n```").expect("parsed");
        assert_eq!(unknown.intent, None);
    }

    #[test]
    fn feedback_intent_defaults_missing_keys() {
        let intent = parse_feedback_intent("```json\n{\"other\": 1}\n```").expect("parsed");
        assert_eq!(intent, FeedbackIntent::default());

        // A bare string is one direction; a numeric feedback is not a string.
        let partial = parse_feedback_intent(
            "```json\n{\"optimized_feedback\": 7, \"optimization_directions\": \"be brief\"}\n```",
        )
        .expect("parsed");
        assert_eq!(partial.optimized_feedback, None);
        assert_eq!(partial.optimization_directions, vec!["be brief".to_owned()]);
    }

    #[test]
    fn parses_the_last_summary_and_the_intent_flag() {
        let text = "<intent>true</intent>\n<summary>first draft</summary>\n\
                    <summary>final answer</summary>";
        assert_eq!(
            parse_bad_case_analysis(text),
            (true, "final answer".to_owned())
        );
    }

    #[test]
    fn parses_an_intent_tag_without_a_summary() {
        let text = "  <intent>true</intent> leftover text  ";
        assert_eq!(
            parse_bad_case_analysis(text),
            (true, "<intent>true</intent> leftover text".to_owned())
        );
    }

    #[test]
    fn analysis_without_tags_falls_back_to_the_trimmed_text() {
        let text = "\n  plain analysis  \n";
        assert_eq!(
            parse_bad_case_analysis(text),
            (false, "plain analysis".to_owned())
        );
        assert_eq!(parse_bad_case_analysis(""), (false, String::new()));
    }

    #[test]
    fn feedback_intent_round_trips_through_json() {
        let intent = FeedbackIntent {
            intent: Some("true".to_owned()),
            optimized_feedback: Some("be concise".to_owned()),
            optimization_directions: vec!["shorten".to_owned()],
        };
        let value = serde_json::to_value(&intent).expect("serialized");
        let object = value.as_object().expect("object");
        for key in ["intent", "optimized_feedback", "optimization_directions"] {
            assert!(object.contains_key(key), "missing field {key}");
        }

        let restored: FeedbackIntent = serde_json::from_value(value).expect("deserialized");
        assert_eq!(restored, intent);

        let empty: FeedbackIntent = serde_json::from_str("{}").expect("defaults");
        assert_eq!(empty, FeedbackIntent::default());
    }
}
