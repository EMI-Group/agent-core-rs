//! The ordered redaction scanner.
//!
//! Every pass mirrors one Python regex from
//! `openjiuwen.symphony.models._redaction` or
//! `openjiuwen.symphony.flow.privacy`, reimplemented by hand because Python's
//! lookbehind assertions have no `regex`-crate equivalent. Scanning is
//! left-to-right and non-overlapping, exactly like `re.sub`, and every offset
//! used for slicing sits on a UTF-8 character boundary.

use super::names::is_sensitive_name;
use super::{DISTILLED_TEXT_LIMIT, REDACTED, REDACTED_CREDENTIAL, REDACTED_REQUEST};

/// Redacts recognized credentials from free text.
///
/// The passes run in this order and each replaces only its matched span, so the
/// surrounding text and line structure survive: private-key blocks, bearer
/// tokens, JWTs, OpenAI-style keys, AWS access key ids, then sensitive
/// `key=value` / `"key": "value"` assignments.
///
/// Redaction is idempotent — applying it twice equals applying it once — never
/// panics on non-ASCII input, and never rewrites text it does not recognize.
///
/// ```
/// use openjiuwen_symphony::redaction::redact_sensitive_text;
///
/// let text = "Authorization: Bearer abcdefgh12345678\nnote: keep me";
/// assert_eq!(
///     redact_sensitive_text(text),
///     "Authorization=<redacted>\nnote: keep me"
/// );
/// ```
#[must_use]
pub fn redact_sensitive_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let bare = redact_common_credentials(text);
    redact_assignments(&bare)
}

/// Redacts bare credentials that carry no field name.
///
/// Mirrors Python's `flow.privacy.redact_common_credentials`: private-key
/// blocks, bearer tokens, JWTs, OpenAI-style keys, and AWS access key ids. A
/// `key=value` assignment whose key names the secret is left to
/// [`redact_sensitive_text`].
#[must_use]
pub fn redact_common_credentials(text: &str) -> String {
    let output = redact_private_key_blocks(text);
    let output = redact_bearer_tokens(&output);
    let output = redact_jwts(&output);
    let output = redact_openai_keys(&output);
    redact_aws_access_keys(&output)
}

/// Redacts credentials and a verbatim source request from model-derived prose.
///
/// This is the flow pipeline's single text entry point: credentials are
/// redacted first, then `verbatim_query` (when given) is replaced by
/// [`REDACTED_REQUEST`], and the result is truncated to
/// [`DISTILLED_TEXT_LIMIT`] characters on a character boundary. Leading and
/// trailing whitespace is trimmed, like Python's `sanitize_distilled_text`.
///
/// ```
/// use openjiuwen_symphony::redaction::sanitize_distilled_text;
///
/// let text = sanitize_distilled_text(
///     "  summarize the release notes  ",
///     Some("summarize the release notes"),
/// );
/// assert_eq!(text, "<redacted-request>");
/// ```
#[must_use]
pub fn sanitize_distilled_text(text: &str, verbatim_query: Option<&str>) -> String {
    let queries: Vec<String> = verbatim_query
        .filter(|query| !query.trim().is_empty())
        .map(|query| vec![query.to_string()])
        .unwrap_or_default();
    sanitize_with_queries(text, &queries)
}

/// Redacts prose against several verbatim source requests in one pass.
///
/// Python's `source_queries` argument accepts an iterable, so the flow engine
/// uses this shared entry point instead of sanitizing per query (which would
/// redact the same text repeatedly).
pub(crate) fn sanitize_with_queries(text: &str, queries: &[String]) -> String {
    let mut output = redact_sensitive_text(text.trim());
    if !queries.is_empty() {
        let mut candidates: Vec<String> = Vec::new();
        for query in queries {
            let normalized = query.trim();
            if normalized.is_empty() {
                continue;
            }
            push_candidate(&mut candidates, normalized.to_string());
            let redacted = redact_sensitive_text(normalized);
            if !redacted.is_empty() {
                push_candidate(&mut candidates, redacted);
            }
        }
        // Longest first, then lexicographic, so overlapping requests resolve
        // deterministically instead of by hash-map iteration order.
        candidates
            .sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
        for candidate in candidates {
            if output.contains(&candidate) {
                output = output.replace(&candidate, REDACTED_REQUEST);
            }
        }
    }
    truncate_on_char_boundary(&output, DISTILLED_TEXT_LIMIT)
}

/// Adds `candidate` once, preserving the caller's relative order.
fn push_candidate(candidates: &mut Vec<String>, candidate: String) {
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

/// Truncates `text` to `limit` characters without splitting a code point.
fn truncate_on_char_boundary(text: &str, limit: usize) -> String {
    match text.char_indices().nth(limit) {
        Some((index, _)) => text[..index].to_string(),
        None => text.to_string(),
    }
}

/// One matched `key=value` / `"key": "value"` assignment.
struct Assignment<'a> {
    /// The raw key, as written.
    key: &'a str,
    /// The quote around the key, or `'\0'` when the key is unquoted.
    quote: char,
    /// The byte index one past the matched assignment.
    end: usize,
    /// Whether the matched value is already the redaction marker.
    already_redacted: bool,
}

/// Rewrites the value of every recognized sensitive assignment.
fn redact_assignments(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut copied = 0;
    while cursor < text.len() {
        let Some(found) = match_assignment(text, cursor) else {
            cursor += char_len_at(text, cursor);
            continue;
        };
        if is_sensitive_name(found.key) && !found.already_redacted {
            output.push_str(&text[copied..cursor]);
            if found.quote != '\0' {
                output.push(found.quote);
            }
            output.push_str(found.key);
            if found.quote != '\0' {
                output.push(found.quote);
            }
            output.push('=');
            output.push_str(REDACTED);
            copied = found.end;
        }
        cursor = found.end;
    }
    output.push_str(&text[copied..]);
    output
}

/// Matches one assignment starting at `start` (a character boundary).
fn match_assignment(text: &str, start: usize) -> Option<Assignment<'_>> {
    let rest = text.get(start..)?;
    let bytes = rest.as_bytes();
    let mut cursor = 0;
    let quote = match bytes.first().copied() {
        Some(byte) if byte == b'"' || byte == b'\'' => {
            cursor = 1;
            char::from(byte)
        }
        _ => '\0',
    };
    if !bytes.get(cursor)?.is_ascii_alphabetic() {
        return None;
    }
    let key_start = cursor;
    cursor += 1;
    let mut consumed = 1;
    while consumed < 65 && bytes.get(cursor).is_some_and(|byte| is_key_byte(*byte)) {
        cursor += 1;
        consumed += 1;
    }
    if consumed < 2 || bytes.get(cursor).is_some_and(|byte| is_key_byte(*byte)) {
        return None;
    }
    let key = rest.get(key_start..cursor)?;
    if quote != '\0' {
        if bytes.get(cursor) != Some(&(quote as u8)) {
            return None;
        }
        cursor += 1;
    }
    cursor = skip_whitespace(rest, cursor);
    if !matches!(bytes.get(cursor), Some(b':' | b'=')) {
        return None;
    }
    cursor += 1;
    cursor = skip_whitespace(rest, cursor);
    let value_start = cursor;
    let end = start + match_value(rest, value_start)?;
    let already_redacted = text
        .get(start + value_start..end)
        .is_some_and(|value| value.trim() == REDACTED);
    Some(Assignment {
        key,
        quote,
        end,
        already_redacted,
    })
}

/// Returns one past the end of the value that starts at `start` in `text`.
///
/// The value is either a quoted string with backslash escapes or a run without
/// `,`, `;`, `{}`, `[]`, or line breaks — Python's `_ASSIGNMENT` alternatives.
fn match_value(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let first = *bytes.get(start)?;
    if first == b'"' || first == b'\'' {
        let mut cursor = start + 1;
        while let Some(byte) = bytes.get(cursor) {
            if *byte == b'\\' {
                cursor += 2;
                continue;
            }
            if *byte == first {
                return Some(cursor + 1);
            }
            cursor += 1;
        }
        // An unterminated quoted value falls through to the bare-value arm,
        // exactly like the regex alternation.
    }
    let mut cursor = start;
    while let Some(byte) = bytes.get(cursor) {
        if matches!(
            *byte,
            b',' | b';' | b'\r' | b'\n' | b'{' | b'}' | b'[' | b']'
        ) {
            break;
        }
        cursor += 1;
    }
    (cursor > start).then_some(cursor)
}

/// Replaces every non-overlapping match reported by `find`.
fn replace_matches(
    text: &str,
    replacement: &str,
    mut find: impl FnMut(&str, usize) -> Option<(usize, usize)>,
) -> String {
    let mut output = String::with_capacity(text.len());
    let mut copied = 0;
    let mut cursor = 0;
    while cursor < text.len() {
        let Some((start, end)) = find(text, cursor) else {
            break;
        };
        if start < cursor || end <= start || end > text.len() {
            break;
        }
        output.push_str(&text[copied..start]);
        output.push_str(replacement);
        copied = end;
        cursor = end;
    }
    output.push_str(&text[copied..]);
    output
}

/// Redacts `-----BEGIN … PRIVATE KEY-----` blocks.
fn redact_private_key_blocks(text: &str) -> String {
    replace_matches(text, REDACTED_CREDENTIAL, find_private_key_block)
}

/// Finds one PEM private-key block at or after `from`.
fn find_private_key_block(text: &str, from: usize) -> Option<(usize, usize)> {
    const BEGIN: &str = "-----BEGIN ";
    const END: &str = "-----END ";
    let mut cursor = from;
    while let Some(offset) = text.get(cursor..)?.find(BEGIN) {
        let begin = cursor + offset;
        let after_begin = begin + BEGIN.len();
        if let Some(body_start) = find_armoured_tail(text, after_begin) {
            let mut search = body_start;
            while let Some(end_offset) = text.get(search..).and_then(|rest| rest.find(END)) {
                let after_end = search + end_offset + END.len();
                if let Some(block_end) = find_armoured_tail(text, after_end) {
                    return Some((begin, block_end));
                }
                search = search + end_offset + 1;
            }
        }
        cursor = after_begin;
    }
    None
}

/// Returns one past `PRIVATE KEY-----` when only `[A-Z ]` precedes it.
fn find_armoured_tail(text: &str, from: usize) -> Option<usize> {
    const TAIL: &str = "PRIVATE KEY-----";
    let mut search = from;
    loop {
        let offset = text.get(search..)?.find(TAIL)?;
        let index = search + offset;
        let armoured = text
            .get(from..index)?
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b' ');
        if armoured {
            return Some(index + TAIL.len());
        }
        search = index + 1;
    }
}

/// Redacts `Bearer <token>` credentials.
fn redact_bearer_tokens(text: &str) -> String {
    replace_matches(text, REDACTED_CREDENTIAL, find_bearer_token)
}

/// Finds one bearer token at or after `from`.
fn find_bearer_token(text: &str, from: usize) -> Option<(usize, usize)> {
    const KEYWORD: &[u8] = b"bearer";
    let bytes = text.as_bytes();
    let mut cursor = from;
    while cursor + KEYWORD.len() <= text.len() {
        let boundary = cursor == 0 || !is_word_byte(bytes[cursor - 1]);
        if boundary && starts_with_ignore_ascii_case(bytes, cursor, KEYWORD) {
            let after_keyword = cursor + KEYWORD.len();
            let token_start = skip_whitespace(text, after_keyword);
            if token_start > after_keyword {
                let mut token_end = token_start;
                while bytes
                    .get(token_end)
                    .is_some_and(|byte| is_bearer_byte(*byte))
                {
                    token_end += 1;
                }
                if token_end - token_start >= 8 {
                    return Some((cursor, token_end));
                }
            }
        }
        cursor += char_len_at(text, cursor);
    }
    None
}

/// Redacts JWT-shaped credentials.
fn redact_jwts(text: &str) -> String {
    replace_matches(text, REDACTED_CREDENTIAL, find_jwt)
}

/// Finds one JWT at or after `from`.
fn find_jwt(text: &str, from: usize) -> Option<(usize, usize)> {
    const HEAD: &[u8] = b"eyJ";
    let bytes = text.as_bytes();
    let mut cursor = from;
    while cursor + HEAD.len() <= text.len() {
        let boundary = cursor == 0 || !is_word_byte(bytes[cursor - 1]);
        if boundary && starts_with_bytes(bytes, cursor, HEAD) {
            if let Some(end) = jwt_end(text, cursor) {
                return Some((cursor, end));
            }
        }
        cursor += char_len_at(text, cursor);
    }
    None
}

/// Returns one past the signature segment of the JWT starting at `start`.
fn jwt_end(text: &str, start: usize) -> Option<usize> {
    const HEAD: &[u8] = b"eyJ";
    let bytes = text.as_bytes();
    let mut cursor = run_end(text, start + HEAD.len(), is_base64url_byte);
    if cursor == start + HEAD.len() || bytes.get(cursor) != Some(&b'.') {
        return None;
    }
    cursor += 1;
    if !starts_with_bytes(bytes, cursor, HEAD) {
        return None;
    }
    let payload_head_end = cursor + HEAD.len();
    cursor = run_end(text, payload_head_end, is_base64url_byte);
    if cursor == payload_head_end || bytes.get(cursor) != Some(&b'.') {
        return None;
    }
    cursor += 1;
    let signature_start = cursor;
    let mut end = run_end(text, signature_start, is_base64url_byte);
    if end == signature_start {
        return None;
    }
    while end > signature_start && !word_boundary_at(text, end) {
        end -= 1;
    }
    (end > signature_start).then_some(end)
}

/// Redacts OpenAI-style `sk-…` keys.
fn redact_openai_keys(text: &str) -> String {
    replace_matches(text, REDACTED_CREDENTIAL, find_openai_key)
}

/// Finds one OpenAI-style key at or after `from`.
fn find_openai_key(text: &str, from: usize) -> Option<(usize, usize)> {
    const PREFIX: &[u8] = b"sk-";
    let bytes = text.as_bytes();
    let mut cursor = from;
    while cursor + PREFIX.len() <= text.len() {
        let boundary = cursor == 0 || !is_word_byte(bytes[cursor - 1]);
        if boundary && starts_with_bytes(bytes, cursor, PREFIX) {
            let token_start = cursor + PREFIX.len();
            let mut token_end = run_end(text, token_start, is_base64url_byte);
            while token_end > token_start && !word_boundary_at(text, token_end) {
                token_end -= 1;
            }
            if token_end - token_start >= 8 {
                return Some((cursor, token_end));
            }
        }
        cursor += char_len_at(text, cursor);
    }
    None
}

/// Redacts AWS access key ids.
fn redact_aws_access_keys(text: &str) -> String {
    replace_matches(text, REDACTED_CREDENTIAL, find_aws_access_key)
}

/// Finds one AWS access key id at or after `from`.
fn find_aws_access_key(text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut cursor = from;
    while cursor + 4 <= text.len() {
        let boundary = cursor == 0 || !is_word_byte(bytes[cursor - 1]);
        let prefixed =
            starts_with_bytes(bytes, cursor, b"AKIA") || starts_with_bytes(bytes, cursor, b"ASIA");
        if boundary && prefixed {
            let id_start = cursor + 4;
            let id_end = id_start + 16;
            let well_formed = text.get(id_start..id_end).is_some_and(|id| {
                id.bytes()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
            });
            if well_formed && word_boundary_at(text, id_end) {
                return Some((cursor, id_end));
            }
        }
        cursor += char_len_at(text, cursor);
    }
    None
}

/// Returns the end of the maximal run of `allowed` bytes starting at `start`.
fn run_end(text: &str, start: usize, allowed: fn(u8) -> bool) -> usize {
    let bytes = text.as_bytes();
    let mut end = start;
    while bytes.get(end).is_some_and(|byte| allowed(*byte)) {
        end += 1;
    }
    end
}

/// Reports a regex `\b` boundary at `index` over ASCII word bytes.
fn word_boundary_at(text: &str, index: usize) -> bool {
    let bytes = text.as_bytes();
    let before = index
        .checked_sub(1)
        .and_then(|previous| bytes.get(previous))
        .is_some_and(|byte| is_word_byte(*byte));
    let after = bytes.get(index).is_some_and(|byte| is_word_byte(*byte));
    before != after
}

/// Reports whether `bytes[index..]` starts with `needle`.
fn starts_with_bytes(bytes: &[u8], index: usize, needle: &[u8]) -> bool {
    bytes.get(index..index + needle.len()) == Some(needle)
}

/// Reports whether `bytes[index..]` starts with `needle`, ignoring ASCII case.
fn starts_with_ignore_ascii_case(bytes: &[u8], index: usize, needle: &[u8]) -> bool {
    bytes
        .get(index..index + needle.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
}

/// Returns the length in bytes of the character starting at `index`.
fn char_len_at(text: &str, index: usize) -> usize {
    text.get(index..)
        .and_then(|rest| rest.chars().next())
        .map_or(1, char::len_utf8)
}

/// Skips Unicode whitespace from `cursor`.
fn skip_whitespace(text: &str, cursor: usize) -> usize {
    let mut index = cursor;
    while let Some(current) = text.get(index..).and_then(|rest| rest.chars().next()) {
        if !current.is_whitespace() {
            break;
        }
        index += current.len_utf8();
    }
    index
}

/// Reports a byte that can continue an assignment key.
fn is_key_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

/// Reports an ASCII regex word byte (`\w` without Unicode).
fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Reports a byte that can continue a base64url/`sk-` token.
fn is_base64url_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

/// Reports a byte that can continue a bearer token.
fn is_bearer_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'+' | b'/' | b'=' | b'-')
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const PEM: &str =
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----";

    #[test]
    fn assignments_are_rewritten_for_sensitive_keys() {
        assert_eq!(
            redact_sensitive_text("api_key: sk-live-0123456789abcdef"),
            "api_key=<redacted>"
        );
        assert_eq!(
            redact_sensitive_text("\"client_secret\": \"s3cr3t\""),
            "\"client_secret\"=<redacted>"
        );
        assert_eq!(
            redact_sensitive_text("password = hunter2 ; note"),
            "password=<redacted>; note"
        );
        assert_eq!(redact_sensitive_text("token: 'abc'"), "token=<redacted>");
        assert_eq!(
            redact_sensitive_text("Authorization: Bearer abcdefgh12345678"),
            "Authorization=<redacted>"
        );
    }

    #[test]
    fn non_sensitive_assignments_survive() {
        let text = "query: user input; timeout=30; token_count: 4";
        assert_eq!(redact_sensitive_text(text), text);
        let prose = "用法：先检索，再总结。";
        assert_eq!(redact_sensitive_text(prose), prose);
    }

    #[test]
    fn bare_credentials_are_redacted() {
        assert_eq!(
            redact_common_credentials("key is AKIAIOSFODNN7EXAMPLE ok"),
            format!("key is {REDACTED_CREDENTIAL} ok")
        );
        assert_eq!(
            redact_common_credentials("see eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl done"),
            format!("see {REDACTED_CREDENTIAL} done")
        );
        assert_eq!(
            redact_common_credentials("token sk-proj-0123456789abcdef"),
            format!("token {REDACTED_CREDENTIAL}")
        );
        assert_eq!(
            redact_common_credentials("header bearer abcdefgh12345678"),
            format!("header {REDACTED_CREDENTIAL}")
        );
        assert_eq!(
            redact_common_credentials(&format!("before\n{PEM}\nafter")),
            format!("before\n{REDACTED_CREDENTIAL}\nafter")
        );
    }

    #[test]
    fn short_or_malformed_credentials_survive() {
        for text in [
            "sk-abc",
            "AKIA1234",
            "bearer short",
            "eyJhbGciOiJIUzI1NiJ9",
            "-----BEGIN RSA PRIVATE KEY-----\nno end",
        ] {
            assert_eq!(redact_common_credentials(text), text, "{text}");
        }
    }

    #[test]
    fn redaction_is_idempotent() {
        let samples = [
            "api_key: sk-live-0123456789abcdef, password = hunter2",
            "Authorization: Bearer abcdefgh12345678",
            "note: token: \"abc\" and email a@b.c",
            "AWS AKIAIOSFODNN7EXAMPLE and eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",
            &format!("pem {PEM} tail"),
            "secret = \"quoted value\" ; trailing",
        ];
        for sample in samples {
            let once = redact_sensitive_text(sample);
            let twice = redact_sensitive_text(&once);
            assert_eq!(once, twice, "not idempotent for {sample}");
        }
    }

    #[test]
    fn arbitrary_non_ascii_input_never_panics() {
        let long_accented = "é".repeat(300);
        for text in [
            "密码=秘密",
            "🔐 key=🔑",
            long_accented.as_str(),
            "token: 密码\u{0301}",
            "-----BEGIN 秘密 PRIVATE KEY-----x",
            "\u{10FFFF}=v",
        ] {
            let redacted = redact_sensitive_text(text);
            assert_eq!(redacted, redact_sensitive_text(&redacted));
        }
    }

    #[test]
    fn sanitize_trims_redacts_queries_and_truncates() {
        assert_eq!(
            sanitize_distilled_text(
                "  summarize the release notes  ",
                Some("summarize the release notes")
            ),
            REDACTED_REQUEST
        );
        assert_eq!(
            sanitize_distilled_text("api_key: sk-live-0123456789abcdef", None),
            "api_key=<redacted>"
        );
        let long = "密".repeat(DISTILLED_TEXT_LIMIT + 10);
        let truncated = sanitize_distilled_text(&long, None);
        assert_eq!(truncated.chars().count(), DISTILLED_TEXT_LIMIT);
        assert_eq!(truncated, "密".repeat(DISTILLED_TEXT_LIMIT));
        assert_eq!(sanitize_distilled_text("", None), "");
    }

    #[test]
    fn sanitize_prefers_the_longest_matching_query() {
        let queries = vec!["run alpha".to_string(), "run alpha beta".to_string()];
        assert_eq!(
            sanitize_with_queries("run alpha beta", &queries),
            REDACTED_REQUEST
        );
        assert_eq!(
            sanitize_with_queries("run alpha and run alpha beta", &queries),
            format!("{REDACTED_REQUEST} and {REDACTED_REQUEST}")
        );
    }

    #[test]
    fn assignment_forms_cover_quoted_and_escaped_values() {
        assert_eq!(
            redact_sensitive_text("password: \"a\\\"b\""),
            "password=<redacted>"
        );
        assert_eq!(
            redact_sensitive_text("password: \"unterminated"),
            "password=<redacted>"
        );
        assert_eq!(
            redact_sensitive_text("{\"token\": \"abc\", \"note\": \"x\"}"),
            "{\"token\"=<redacted>, \"note\": \"x\"}"
        );
    }
}
