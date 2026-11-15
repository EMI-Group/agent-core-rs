//! Tolerant, hand-rolled JSON repair.
//!
//! The Python reference delegates to `json_repair`; this module reimplements the
//! part of that grammar a tool result actually carries, one small named step per
//! rule, and reports what it could not repair instead of guessing:
//!
//! 1. [`strip_code_fence`] drops a leading ```` ```json ```` line and its closing
//!    fence,
//! 2. [`extract_json_span`] drops the prose before the first `{`/`[` and the
//!    prose after the bracket that closes it,
//! 3. [`normalize_smart_quotes`] folds `“ ” ‘ ’` (and `„ ‟`) into `"`/`'`,
//! 4. [`normalize_python_literals`] rewrites `True`/`False`/`None` as
//!    `true`/`false`/`null`,
//! 5. [`normalize_single_quotes`] rewrites `'…'` strings (escapes included) as
//!    `"…"`,
//! 6. [`quote_unquoted_keys`] quotes `{key: …}` member names,
//! 7. [`strip_trailing_commas`] removes a `,` that precedes `}` or `]`,
//! 8. [`close_truncated`] closes an unterminated string and the open brackets,
//! 9. [`drop_incomplete_tail`] drops a half-written final element,
//! 10. [`wrap_value_sequence`] and [`wrap_bare_object_members`] add the outer
//!     brackets a body lost (JSON Lines, comma-separated values, bare members).
//!
//! Every step is pure and deterministic, and every step is only applied when the
//! earlier one left something to fix.
//! A body that no candidate repairs is reported as a typed [`JsonRepairFailure`]
//! and one `tracing::debug!` line names the failure kind; nothing is ever
//! silently truncated to make it parse.
use serde_json::{Map, Value};
use std::fmt;

/// How a repair attempt failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonRepairFailure {
    /// The text holds no `{`/`[` to anchor on and no bare members or values.
    NoJsonBody,
    /// The text opens a bracket it never closes and no cut-back repaired it.
    UnbalancedBrackets,
    /// Every repaired candidate still failed to parse.
    Unparsable,
    /// The text repaired into a value that is not a JSON array.
    NotAnArray,
}
impl JsonRepairFailure {
    /// The short label this failure is reported under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoJsonBody => "no_json_body",
            Self::UnbalancedBrackets => "unbalanced_brackets",
            Self::Unparsable => "unparsable",
            Self::NotAnArray => "not_an_array",
        }
    }
}
impl fmt::Display for JsonRepairFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl std::error::Error for JsonRepairFailure {}

/// Repairs `text` into a JSON value, or reports why it could not.
///
/// The repair is a fixed ladder (see the module docs); a text that already parses
/// is returned unchanged.
pub fn repair_json(text: &str) -> Result<Value, JsonRepairFailure> {
    let repaired = repair_document(text)?;
    serde_json::from_str(&repaired).map_err(|_| JsonRepairFailure::Unparsable)
}

/// Repairs `text` into a JSON array, or reports why it could not.
pub fn repair_json_array_checked(text: &str) -> Result<Value, JsonRepairFailure> {
    let value = repair_json(text)?;
    if value.is_array() {
        Ok(value)
    } else {
        Err(JsonRepairFailure::NotAnArray)
    }
}

/// Repairs `text` into a JSON array, reporting a failure through `tracing`.
///
/// This is the entry point the router and the JSON-array compressor share: a body
/// that cannot be repaired yields `None` plus one `tracing::debug!` line naming
/// the failure kind, so a misrouted tool result stays diagnosable.
#[must_use]
pub fn repair_json_array(text: &str) -> Option<Value> {
    match repair_json_array_checked(text) {
        Ok(value) => Some(value),
        Err(failure) => {
            tracing::debug!(
                failure = failure.as_str(),
                bytes = text.len(),
                "rule compression: JSON repair failed"
            );
            None
        }
    }
}

/// Repairs `text` into the text of a JSON value that parses.
fn repair_document(text: &str) -> Result<String, JsonRepairFailure> {
    let unfenced = strip_code_fence(text);
    if unfenced.trim().is_empty() {
        return Err(JsonRepairFailure::NoJsonBody);
    }
    let mut single: Option<String> = None;
    let mut single_failure = JsonRepairFailure::NoJsonBody;
    // A single value: strip the prose around it, normalize it, then close it.
    if let Some((start, end)) = extract_json_span(&unfenced) {
        match repair_span(&unfenced[start..end]) {
            Ok(repaired) => {
                // A span that ends before more JSON is one element of a value
                // sequence; wrapping keeps every element instead of the first.
                if !unfenced[end..].contains(['{', '[']) {
                    return Ok(repaired);
                }
                single = Some(repaired);
            }
            Err(failure) => single_failure = failure,
        }
    }
    // No braces, or several values: the body lost its outer brackets.
    if let Some(wrapped) = wrap_value_sequence(&unfenced) {
        let body = normalize_body(&wrapped);
        if let Some(repaired) = first_parsable(&body) {
            return Ok(repaired);
        }
    }
    if let Some(wrapped) = wrap_bare_object_members(&unfenced) {
        let body = normalize_body(&wrapped);
        if let Some(repaired) = first_parsable(&body) {
            return Ok(repaired);
        }
    }
    match single {
        Some(repaired) => Ok(repaired),
        None => Err(single_failure),
    }
}

/// Repairs one extracted `{…}`/`[…]` span, or reports why it could not.
fn repair_span(span: &str) -> Result<String, JsonRepairFailure> {
    let body = normalize_body(span);
    if let Some(repaired) = first_parsable(&body) {
        return Ok(repaired);
    }
    // A body that still fails is reported, never cut into something else.
    if scan_state(&body).open_brackets.is_empty() {
        Err(JsonRepairFailure::Unparsable)
    } else {
        Err(JsonRepairFailure::UnbalancedBrackets)
    }
}

/// Applies steps 3–7 of the ladder to one extracted body.
///
/// Single quotes are rewritten before Python literals so a `'True'` string keeps
/// its spelling: once it is a `"…"` string, the literal pass leaves it alone.
fn normalize_body(body: &str) -> String {
    let body = normalize_smart_quotes(body);
    let body = normalize_single_quotes(&body);
    let body = normalize_python_literals(&body);
    let body = quote_unquoted_keys(&body);
    strip_trailing_commas(&body)
}

/// The first candidate derived from `body` that parses to a non-empty value.
///
/// The candidates are `body` itself, then its bracket-closed form, then the form
/// with the half-written final element dropped.
fn first_parsable(body: &str) -> Option<String> {
    let mut candidates = vec![body.to_owned()];
    let closed = close_truncated(body);
    if closed != body {
        candidates.push(closed);
    }
    if let Some(tail) = drop_incomplete_tail(body) {
        let tail = close_truncated(&tail);
        if !candidates.contains(&tail) {
            candidates.push(tail);
        }
    }
    for (index, candidate) in candidates.iter().enumerate() {
        let Ok(value) = serde_json::from_str::<Value>(candidate) else {
            continue;
        };
        // Closing or cutting must not turn a one-element body into an empty one.
        if index > 0 && is_empty_container(&value) {
            continue;
        }
        return Some(candidate.clone());
    }
    None
}

/// Whether `value` is an empty object or array.
fn is_empty_container(value: &Value) -> bool {
    value.as_array().is_some_and(Vec::is_empty) || value.as_object().is_some_and(Map::is_empty)
}

/// Drops a leading ```` ``` ```` fence line and a trailing fence.
pub(crate) fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed.to_owned();
    };
    let body = match rest.split_once('\n') {
        Some((_, body)) => body,
        None => rest,
    };
    let body = body.trim_end();
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim().to_owned()
}

/// The byte range of the JSON value in `text`, with the prose around it dropped.
fn extract_json_span(text: &str) -> Option<(usize, usize)> {
    let start = text
        .char_indices()
        .find(|(_, value)| matches!(value, '{' | '['))
        .map(|(index, _)| index)?;
    let span = truncate_at_json_end(&text[start..]);
    Some((start, start + span.len()))
}

/// Cuts `text` right after the bracket that closes its first `{`/`[`.
fn truncate_at_json_end(text: &str) -> &str {
    let mut depth = 0_i64;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (index, value) in text.char_indices() {
        if let Some(active) = quote {
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == active {
                quote = None;
            }
            continue;
        }
        match value {
            '"' | '\'' => quote = Some(value),
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return &text[..=index];
                }
            }
            _ => {}
        }
    }
    text
}

/// Folds typographic quotes outside a `"…"` string into their ASCII forms.
fn normalize_smart_quotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    for value in text.chars() {
        if in_string {
            out.push(value);
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
            }
            continue;
        }
        match value {
            '"' => {
                in_string = true;
                out.push('"');
            }
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => out.push('"'),
            '\u{2018}' | '\u{2019}' => out.push('\''),
            _ => out.push(value),
        }
    }
    out
}

/// Rewrites the Python literals `True`/`False`/`None` outside strings.
fn normalize_python_literals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < chars.len() {
        let value = chars[index];
        if in_string {
            out.push(value);
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if value == '"' {
            in_string = true;
            out.push(value);
            index += 1;
            continue;
        }
        if is_identifier_start(value) {
            let start = index;
            let mut end = index + 1;
            while end < chars.len() && is_identifier_char(chars[end]) {
                end += 1;
            }
            let word: String = chars[start..end].iter().collect();
            match word.as_str() {
                "True" => out.push_str("true"),
                "False" => out.push_str("false"),
                "None" => out.push_str("null"),
                _ => out.push_str(&word),
            }
            index = end;
            continue;
        }
        out.push(value);
        index += 1;
    }
    out
}

/// Rewrites `'…'` strings as `"…"`, escaping inner double quotes.
fn normalize_single_quotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(value) = chars.next() {
        if in_string {
            out.push(value);
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
            }
            continue;
        }
        match value {
            '"' => {
                in_string = true;
                out.push(value);
            }
            '\'' => {
                out.push('"');
                while let Some(inner) = chars.next() {
                    match inner {
                        '\'' => break,
                        '"' => out.push_str("\\\""),
                        '\\' => match chars.peek().copied() {
                            Some('\'') => {
                                chars.next();
                                out.push('\'');
                            }
                            Some('"') => {
                                chars.next();
                                out.push_str("\\\"");
                            }
                            _ => out.push('\\'),
                        },
                        other => out.push(other),
                    }
                }
                // A string cut by the end of the body is closed here, so the
                // caller sees a complete literal rather than a silent drop.
                out.push('"');
            }
            other => out.push(other),
        }
    }
    out
}

/// Quotes `{key: …}` member names that lost their quotes.
fn quote_unquoted_keys(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut last_significant: Option<char> = None;
    let mut index = 0;
    while index < chars.len() {
        let value = chars[index];
        if in_string {
            out.push(value);
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
                last_significant = Some(value);
            }
            index += 1;
            continue;
        }
        match value {
            '"' => {
                in_string = true;
                out.push(value);
                index += 1;
            }
            '{' | '[' => {
                stack.push(value);
                out.push(value);
                last_significant = Some(value);
                index += 1;
            }
            '}' | ']' => {
                stack.pop();
                out.push(value);
                last_significant = Some(value);
                index += 1;
            }
            _ if is_identifier_start(value)
                && stack.last() == Some(&'{')
                && matches!(last_significant, Some('{' | ',')) =>
            {
                let start = index;
                let mut end = index + 1;
                while end < chars.len() && is_identifier_char(chars[end]) {
                    end += 1;
                }
                let mut after = end;
                while after < chars.len() && chars[after].is_whitespace() {
                    after += 1;
                }
                if chars.get(after) == Some(&':') {
                    out.push('"');
                    out.extend(&chars[start..end]);
                    out.push('"');
                    last_significant = Some('"');
                    index = end;
                    continue;
                }
                out.push(value);
                last_significant = Some(value);
                index += 1;
            }
            other => {
                out.push(other);
                if !other.is_whitespace() {
                    last_significant = Some(other);
                }
                index += 1;
            }
        }
    }
    out
}

/// Removes a `,` that directly precedes `}` or `]`, outside strings.
fn strip_trailing_commas(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    for (index, value) in chars.iter().copied().enumerate() {
        if in_string {
            out.push(value);
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
            }
            continue;
        }
        match value {
            '"' => {
                in_string = true;
                out.push(value);
            }
            ',' => {
                let next = chars[index + 1..]
                    .iter()
                    .find(|candidate| !candidate.is_whitespace());
                if matches!(next, Some(']' | '}')) {
                    continue;
                }
                out.push(value);
            }
            other => out.push(other),
        }
    }
    out
}

/// What one bracket scan saw.
struct ScanState {
    /// The bracket kinds still open, outermost first.
    open_brackets: Vec<char>,
    /// The quote character of an unterminated string, when there is one.
    open_quote: Option<char>,
}

/// Scans `text` for open brackets and an unterminated string.
fn scan_state(text: &str) -> ScanState {
    let mut open_brackets = Vec::new();
    let mut open_quote: Option<char> = None;
    let mut escaped = false;
    for value in text.chars() {
        if let Some(active) = open_quote {
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == active {
                open_quote = None;
            }
            continue;
        }
        match value {
            '"' => open_quote = Some('"'),
            '{' | '[' => open_brackets.push(value),
            '}' | ']' => {
                open_brackets.pop();
            }
            _ => {}
        }
    }
    ScanState {
        open_brackets,
        open_quote,
    }
}

/// Closes an unterminated string and every open bracket.
fn close_truncated(text: &str) -> String {
    let state = scan_state(text);
    let mut out = text.trim_end().to_owned();
    if let Some(quote) = state.open_quote {
        out.push(quote);
    } else if out.ends_with(',') {
        // A comma left dangling by the truncation cannot precede a closer.
        out.pop();
        out = out.trim_end().to_owned();
    }
    for opener in state.open_brackets.iter().rev() {
        out.push(if *opener == '{' { '}' } else { ']' });
    }
    out
}

/// Drops a half-written final element by cutting at the last `,` outside strings.
fn drop_incomplete_tail(text: &str) -> Option<String> {
    let mut in_string = false;
    let mut escaped = false;
    let mut last_comma = None;
    for (index, value) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
            }
            continue;
        }
        match value {
            '"' => in_string = true,
            ',' => last_comma = Some(index),
            _ => {}
        }
    }
    let cut = last_comma?;
    let prefix = text.get(..cut)?.trim_end();
    if prefix.is_empty() {
        return None;
    }
    Some(prefix.to_owned())
}

/// Wraps a body of several values (JSON Lines, comma-separated) in `[ … ]`.
fn wrap_value_sequence(text: &str) -> Option<String> {
    let trimmed = trim_trailing_prose(text);
    if trimmed.is_empty() || serde_json::from_str::<Value>(trimmed).is_ok() {
        return None;
    }
    let wrapped = format!(
        "[{}]",
        strip_trailing_commas(&separate_top_level_values(trimmed))
    );
    let normalized = normalize_body(&wrapped);
    first_parsable(&normalized)
        .filter(|candidate| {
            serde_json::from_str::<Value>(candidate).is_ok_and(|value| value.is_array())
        })
        .map(|_| wrapped)
}

/// Inserts the comma two adjacent top-level values need to share one array.
///
/// A JSON Lines body writes one value per line and never separates them; the
/// comma has to be added before the body can be wrapped.
fn separate_top_level_values(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut depth = 0_i64;
    let mut in_string = false;
    let mut escaped = false;
    let mut value_ended = false;
    for value in text.chars() {
        if in_string {
            out.push(value);
            if escaped {
                escaped = false;
            } else if value == '\\' {
                escaped = true;
            } else if value == '"' {
                in_string = false;
            }
            continue;
        }
        match value {
            '"' | '{' | '[' if depth == 0 && value_ended => {
                push_comma_before_whitespace(&mut out);
                value_ended = false;
                if value == '"' {
                    in_string = true;
                } else {
                    depth += 1;
                }
                out.push(value);
            }
            '"' => {
                in_string = true;
                out.push(value);
            }
            '{' | '[' => {
                depth += 1;
                out.push(value);
            }
            '}' | ']' => {
                depth -= 1;
                out.push(value);
                if depth == 0 {
                    value_ended = true;
                }
            }
            ',' if depth == 0 => {
                value_ended = false;
                out.push(value);
            }
            other => {
                if depth == 0 && value_ended && !other.is_whitespace() {
                    out.push(',');
                    value_ended = false;
                }
                out.push(other);
            }
        }
    }
    out
}

/// Appends `,` before the whitespace that already ends `out`.
fn push_comma_before_whitespace(out: &mut String) {
    let mut whitespace = String::new();
    while out.chars().next_back().is_some_and(char::is_whitespace) {
        if let Some(value) = out.pop() {
            whitespace.push(value);
        }
    }
    out.push(',');
    out.extend(whitespace.chars().rev());
}

/// Wraps bare `key: value` members in `{ … }`.
fn wrap_bare_object_members(text: &str) -> Option<String> {
    let trimmed = trim_trailing_prose(text);
    if trimmed.is_empty() || !trimmed.contains(':') {
        return None;
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return None;
    }
    Some(format!("{{{trimmed}}}"))
}

/// Drops prose that follows the last `}` or `]` in `text`.
fn trim_trailing_prose(text: &str) -> &str {
    let end = text
        .char_indices()
        .filter(|(_, value)| matches!(value, '}' | ']'))
        .map(|(index, value)| index + value.len_utf8())
        .next_back();
    match end {
        Some(end) => text[..end].trim(),
        None => text.trim(),
    }
}

/// Whether `value` can start an unquoted identifier.
fn is_identifier_start(value: char) -> bool {
    value.is_alphabetic() || value == '_'
}

/// Whether `value` can continue an unquoted identifier.
fn is_identifier_char(value: char) -> bool {
    value.is_alphanumeric() || value == '_' || value == '-'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_fences_are_dropped() {
        assert_eq!(
            strip_code_fence("```json\n[1, 2]\n```"),
            "[1, 2]".to_owned()
        );
        assert_eq!(strip_code_fence("```[1, 2]```"), "[1, 2]".to_owned());
        assert_eq!(strip_code_fence("  [1, 2]  "), "[1, 2]".to_owned());
    }

    #[test]
    fn leading_and_trailing_prose_are_dropped() {
        let value = repair_json_array("Here are the rows: [{\"a\": 1}] — done").expect("repaired");
        assert_eq!(value.to_string(), r#"[{"a":1}]"#);
        assert_eq!(
            truncate_at_json_end("[1] and more"),
            "[1]".to_owned(),
            "the closing bracket ends the span"
        );
    }

    #[test]
    fn trailing_commas_are_removed() {
        let value = repair_json_array("[{\"a\": 1,}, {\"b\": 2,},]").expect("repaired");
        assert_eq!(value.as_array().map(Vec::len), Some(2));
        assert_eq!(
            strip_trailing_commas("{\"a\": [1, 2,],}"),
            "{\"a\": [1, 2]}"
        );
        // A comma inside a string is data, not syntax.
        assert_eq!(strip_trailing_commas(r#"["a,]"]"#), r#"["a,]"]"#);
    }

    #[test]
    fn single_quoted_strings_are_rewritten() {
        assert_eq!(normalize_single_quotes("[{'a': 'b'}]"), "[{\"a\": \"b\"}]");
        assert_eq!(
            normalize_single_quotes(r#"['don\'t', 'say \"hi\"']"#),
            r#"["don't", "say \"hi\""]"#
        );
        // A legitimate double-quoted string keeps its apostrophes untouched.
        assert_eq!(
            normalize_single_quotes(r#"["it's fine"]"#),
            r#"["it's fine"]"#
        );
    }

    #[test]
    fn unquoted_keys_are_quoted() {
        assert_eq!(
            quote_unquoted_keys("{a: 1, b_c: 2}"),
            "{\"a\": 1, \"b_c\": 2}"
        );
        // Values are left alone, and an already-quoted key is not doubled.
        assert_eq!(quote_unquoted_keys("{\"a\": wide}"), "{\"a\": wide}");
        assert_eq!(quote_unquoted_keys("[{x: 1}]"), "[{\"x\": 1}]");
    }

    #[test]
    fn python_literals_are_rewritten() {
        assert_eq!(
            normalize_python_literals("[True, False, None]"),
            "[true, false, null]"
        );
        // A literal-looking word inside a string is data.
        assert_eq!(
            normalize_python_literals(r#"["True story"]"#),
            r#"["True story"]"#
        );
        assert_eq!(
            normalize_python_literals("[Trueish, Conversely]"),
            "[Trueish, Conversely]"
        );
    }

    #[test]
    fn smart_quotes_are_folded() {
        assert_eq!(
            normalize_smart_quotes("{\u{201c}a\u{201d}: \u{2018}b\u{2019}}"),
            "{\"a\": 'b'}"
        );
        // Smart quotes inside a real JSON string stay as written.
        assert_eq!(
            normalize_smart_quotes("[\"he said \u{201c}hi\u{201d}\"]"),
            "[\"he said \u{201c}hi\u{201d}\"]"
        );
    }

    #[test]
    fn a_truncated_final_element_is_dropped() {
        assert_eq!(close_truncated("[1, 2"), "[1, 2]".to_owned());
        assert_eq!(close_truncated("[{\"a\": 1"), "[{\"a\": 1}]".to_owned());
        assert_eq!(
            close_truncated("[{\"a\": 1}, {\"b\": 2"),
            "[{\"a\": 1}, {\"b\": 2}]"
        );
        assert_eq!(
            drop_incomplete_tail("[{\"a\": 1}, {\"b\":"),
            Some("[{\"a\": 1}".to_owned())
        );
        let repaired = repair_json_array("[{\"a\": 1}, {\"b\":").expect("repaired");
        assert_eq!(repaired.to_string(), r#"[{"a":1}]"#);
    }

    #[test]
    fn truncated_arrays_and_strings_are_closed() {
        assert_eq!(close_truncated("[\"abc"), "[\"abc\"]".to_owned());
        let value = repair_json_array("[\"abc").expect("repaired");
        assert_eq!(value.to_string(), "[\"abc\"]");
        // A body that never held a complete element is reported, not emptied.
        let body = "[\"abc";
        assert_eq!(close_truncated(body), "[\"abc\"]");
        // `[{` is closed rather than emptied: the array keeps the object it opened.
        assert_eq!(
            repair_json_array("[{").map(|value| value.to_string()),
            Some("[{}]".to_owned())
        );
    }

    #[test]
    fn a_bare_value_sequence_is_wrapped() {
        let value = repair_json_array("{\"a\": 1}\n{\"b\": 2}").expect("repaired");
        assert_eq!(value.to_string(), r#"[{"a":1},{"b":2}]"#);
        let comma_separated = repair_json_array("{\"a\": 1}, {\"b\": 2}").expect("repaired");
        assert_eq!(comma_separated.as_array().map(Vec::len), Some(2));
    }
    #[test]
    fn top_level_values_are_comma_separated() {
        assert_eq!(
            separate_top_level_values("{\"a\": 1}\n{\"b\": 2}"),
            "{\"a\": 1},\n{\"b\": 2}"
        );
        assert_eq!(separate_top_level_values("[1]\n[2]"), "[1],\n[2]");
        // A body that already separates its values is left alone.
        assert_eq!(separate_top_level_values("[1], [2]"), "[1], [2]");
        // A comma inside a value is data, not a separator.
        assert_eq!(separate_top_level_values("[\"a,b\"]"), "[\"a,b\"]");
    }
    #[test]
    fn bare_object_members_are_wrapped() {
        let value = repair_json("\"a\": 1, \"b\": 2").expect("repaired");
        assert_eq!(value.to_string(), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn a_single_quoted_literal_stays_a_string() {
        let value = repair_json_array("[{'flag': 'True'}]").expect("repaired");
        assert_eq!(value.to_string(), r#"[{"flag":"True"}]"#);
    }

    #[test]
    fn a_json_array_round_trips_through_the_ladder() {
        let value = repair_json_array("```json\n[{'a': True,},]\n```").expect("repaired");
        assert_eq!(value.to_string(), r#"[{"a":true}]"#);
    }

    #[test]
    fn an_unrepairable_body_is_reported() {
        assert_eq!(
            repair_json_array_checked("[{\"a\": oops}]"),
            Err(JsonRepairFailure::Unparsable)
        );
        assert_eq!(
            repair_json_array_checked("just prose about cats"),
            Err(JsonRepairFailure::NoJsonBody)
        );
        assert_eq!(
            repair_json_array_checked("{\"a\": 1}"),
            Err(JsonRepairFailure::NotAnArray)
        );
        assert_eq!(repair_json_array("[{\"a\": oops}]"), None);
        assert_eq!(JsonRepairFailure::Unparsable.as_str(), "unparsable");
        assert_eq!(JsonRepairFailure::NotAnArray.to_string(), "not_an_array");
    }
}
