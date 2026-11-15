//! Credential and error-reason redaction for observability attributes.
//!
//! Port of the error/credential half of `openjiuwen/extensions/observability/redaction.py`
//! (v0.1.20). The prompt/completion half of that file — policy-driven `sha256:` hashing and
//! attribute truncation — already lives in [`crate::otel::redaction`] and is reused here rather
//! than duplicated: [`hash_value`] is re-exported and every length cap goes through that module's
//! `truncate`, so the truncation marker stays identical across the two redaction layers.
//!
//! # What is masked
//!
//! - [`mask_secrets`] masks the *value* of a credential-shaped assignment (`password = …`,
//!   `api-key=…`, `client_secret: "…"`, `Authorization: Bearer …`) while keeping the key and the
//!   separator, then masks any standalone `Bearer <token>` credential.
//! - [`mask_bearer_tokens`] is that second pass on its own.
//! - [`redact_error_summary`] masks, collapses all whitespace runs to single spaces, and caps the
//!   result — the text shown next to a Failed trajectory status.
//! - [`redact_error_stacktrace`] masks and caps, but keeps the line structure.
//!
//! Error reasons are deliberately not hashed: the reason has to stay readable, so only
//! credential-shaped fragments are masked in place, and the cap ([`ERROR_SUMMARY_MAX_LENGTH`]) sits
//! well below the general attribute cap so the trajectory status line stays compact.
//!
//! # Best effort, on purpose
//!
//! Python compiles two regular expressions for this; the `regex` crate is not a dependency of this
//! crate, so both patterns are re-implemented as hand-rolled character scanners. They walk
//! character positions, never split a multi-byte character, and never panic on non-ASCII input. The
//! masking is documented as *pattern-based*: it recognises credential-shaped text, it is not a
//! config/JSON/env parser, so a credential split across lines, built at runtime, or spelled
//! outside the key vocabulary below passes through untouched. Untouched text is preserved exactly.
//!
//! # Python counterparts and intentional differences
//!
//! - `_mask_sensitive_text` → [`mask_secrets`] (same pass order: assignments first, bearer second).
//! - `redact_error_summary` / `redact_error_stacktrace` → same names here, minus the
//!   `ObservabilityConfig` argument: the cap is always [`ERROR_SUMMARY_MAX_LENGTH`], where Python
//!   uses `min(config.attribute_value_max_length, 4096)` when a config is supplied. A caller that
//!   must honour a tighter cap applies its own cap afterwards.
//! - Python's `object` parameter is coerced with `str()` and `None` becomes `""`; these helpers take
//!   `&str`, so that coercion belongs to the caller.
//! - `_hash` → [`hash_value`] in [`crate::otel::redaction`], reused verbatim (`sha256:` plus the
//!   first 16 hex digits), so identical inputs correlate without the content leaking.
//!
//! # No secret is ever logged
//!
//! These helpers only build the masked text a caller stores on a span attribute. Raw prompts,
//! completions, credentials and error payloads are never logged by this crate: `tracing` events stay
//! free of payload contents, and masked text is what reaches an exporter.

use crate::otel::redaction::truncate;

pub use crate::otel::redaction::hash_value;

/// Replacement text for a masked secret, Python's `_SECRET_MASK`.
pub const MASK: &str = "***";

/// Replacement text for a standalone bearer credential.
///
/// Python writes it as the literal `"Bearer ***"` (capital `B`), whatever the source casing.
pub const BEARER_MASK: &str = "Bearer ***";

/// Hard cap for an error summary or stack trace, Python's `_ERROR_SUMMARY_MAX_LENGTH`.
pub const ERROR_SUMMARY_MAX_LENGTH: usize = 4096;

/// Prefix identifying a hashed value, Python's `_REDACTED_PREFIX`.
pub const REDACTED_HASH_PREFIX: &str = "sha256:";

/// Minimum length of a bearer credential before it is masked.
///
/// Python's `_BEARER_TOKEN_RE` requires `{8,}` characters from the token alphabet, which keeps short
/// words after the word "bearer" readable.
const BEARER_TOKEN_MIN_LENGTH: usize = 8;

/// Sensitive assignment keys, in Python's alternation order.
///
/// Each entry is one or more words joined by an optional `_` or `-`, mirroring the Python pattern
/// `api[_-]?key`: `api_key`, `api-key` and `apikey` all match. Order matters only where two keys
/// share a prefix (`password` before `passwd`), exactly as in the Python alternation.
const SENSITIVE_KEYS: [&[&str]; 12] = [
    &["authorization"],
    &["api", "key"],
    &["access", "token"],
    &["refresh", "token"],
    &["session", "token"],
    &["id", "token"],
    &["cookie"],
    &["password"],
    &["passwd"],
    &["secret"],
    &["client", "secret"],
    &["private", "key"],
];

/// Masks every credential-shaped assignment in `value`, then any bearer credential.
///
/// The key and the separator are kept (with their original casing and spacing) so the text stays
/// readable, and only the value becomes [`MASK`]. Passes run in Python's order: labelled
/// assignments first, standalone bearer credentials second, which also makes the result stable under
/// repeated application — an already-masked value has nothing left to match.
///
/// ```
/// use openjiuwen_extensions::observability::redaction::mask_secrets;
///
/// assert_eq!(mask_secrets(r#"password = "hunter2""#), "password = ***");
/// assert_eq!(mask_secrets("Authorization: Bearer abcdefgh12345"), "Authorization: ***");
/// assert_eq!(mask_secrets("api-key=xyz"), "api-key=***");
/// assert_eq!(mask_secrets("call bearer abcdefgh12345 now"), "call Bearer *** now");
/// ```
#[must_use]
pub fn mask_secrets(value: &str) -> String {
    mask_bearer_tokens(&mask_sensitive_assignments(value))
}

/// Masks a standalone bearer credential: `bearer` plus at least
/// [`BEARER_TOKEN_MIN_LENGTH`] token characters becomes [`BEARER_MASK`].
///
/// The word boundary, the case-insensitive `bearer` keyword and the token alphabet
/// (`A-Z`, `a-z`, `0-9`, `.`, `_`, `~`, `+`, `/`, `=`, `-`) match Python's
/// `(?i)\bbearer\s+([a-z0-9._~+/=-]{8,})`.
///
/// ```
/// use openjiuwen_extensions::observability::redaction::mask_bearer_tokens;
///
/// assert_eq!(mask_bearer_tokens("bearer abcdefgh12345"), "Bearer ***");
/// // Too short to be a credential: short words after "bearer" stay readable.
/// assert_eq!(mask_bearer_tokens("bearer abc"), "bearer abc");
/// ```
#[must_use]
pub fn mask_bearer_tokens(value: &str) -> String {
    replace_matches(value, match_bearer_token, BEARER_MASK)
}

/// Masks credentials in an error reason, collapses whitespace, and caps the length.
///
/// This is the text the trajectory UI shows next to a Failed status: masked in place, never hashed,
/// folded onto one line (`" ".join(text.split())` in Python), then capped at
/// [`ERROR_SUMMARY_MAX_LENGTH`].
///
/// ```
/// use openjiuwen_extensions::observability::redaction::redact_error_summary;
///
/// assert_eq!(
///     redact_error_summary("  boom\n\tAuthorization: Bearer abcdefgh12345  at x  "),
///     "boom Authorization: *** at x"
/// );
/// ```
#[must_use]
pub fn redact_error_summary(value: &str) -> String {
    let masked = mask_secrets(value);
    let collapsed = collapse_whitespace(&masked);
    truncate(&collapsed, ERROR_SUMMARY_MAX_LENGTH)
}

/// Masks credentials in a stack trace and caps the length, keeping the line structure.
///
/// Python's `redact_error_stacktrace` does not collapse whitespace, and neither does this: indented
/// frames and line breaks carry the call structure. The cap is [`ERROR_SUMMARY_MAX_LENGTH`].
///
/// ```
/// use openjiuwen_extensions::observability::redaction::redact_error_stacktrace;
///
/// let trace = "Traceback:\n  File \"x.py\", line 1\n    password=hunter2";
/// assert_eq!(
///     redact_error_stacktrace(trace),
///     "Traceback:\n  File \"x.py\", line 1\n    password=***"
/// );
/// ```
#[must_use]
pub fn redact_error_stacktrace(value: &str) -> String {
    truncate(&mask_secrets(value), ERROR_SUMMARY_MAX_LENGTH)
}

// ---------------------------------------------------------------------------
// Scanner internals
// ---------------------------------------------------------------------------

/// Replaces every non-overlapping match found by `matcher` with `replacement`.
///
/// The scan is left-to-right and consumes matched spans whole, like `re.sub`; unmatched text is
/// copied verbatim, so non-ASCII input is preserved byte for byte.
fn replace_matches(
    value: &str,
    matcher: fn(&[char], usize) -> Option<usize>,
    replacement: &str,
) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len());
    let mut emitted = 0;
    let mut position = 0;
    while position < chars.len() {
        let at_boundary = position == 0 || !is_word_char(chars[position - 1]);
        if at_boundary {
            if let Some(end) = matcher(&chars, position) {
                out.extend(chars[emitted..position].iter());
                out.push_str(replacement);
                emitted = end;
                position = end;
                continue;
            }
        }
        position += 1;
    }
    out.extend(chars[emitted..].iter());
    out
}

/// `_SENSITIVE_ASSIGNMENT_RE.sub(...)`: masks the value of each labelled assignment.
fn mask_sensitive_assignments(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len());
    let mut emitted = 0;
    let mut position = 0;
    while position < chars.len() {
        let at_boundary = position == 0 || !is_word_char(chars[position - 1]);
        if at_boundary {
            if let Some(key_end) = match_sensitive_key(&chars, position) {
                if let Some(separator_end) = match_assignment_separator(&chars, key_end) {
                    if let Some(value_end) = match_assignment_value(&chars, separator_end) {
                        // Python keeps group 1 and group 2 verbatim: key, separator, mask.
                        out.extend(chars[emitted..separator_end].iter());
                        out.push_str(MASK);
                        emitted = value_end;
                        position = value_end;
                        continue;
                    }
                }
            }
        }
        position += 1;
    }
    out.extend(chars[emitted..].iter());
    out
}

/// Matches the first sensitive key that fits at `at`, returning the end of the key.
fn match_sensitive_key(chars: &[char], at: usize) -> Option<usize> {
    SENSITIVE_KEYS
        .iter()
        .find_map(|words| match_key_words(chars, at, words))
}

/// Matches one key's words, allowing one optional `_` or `-` between them.
fn match_key_words(chars: &[char], at: usize, words: &[&str]) -> Option<usize> {
    let mut position = at;
    for (index, word) in words.iter().enumerate() {
        if index > 0 && matches!(chars.get(position).copied(), Some('_' | '-')) {
            position += 1;
        }
        position = match_ascii_case_insensitive(chars, position, word)?;
    }
    Some(position)
}

/// Matches `\s*[=:]\s*` at `at`, returning the end of the separator run.
fn match_assignment_separator(chars: &[char], at: usize) -> Option<usize> {
    let mut position = at;
    while chars.get(position).is_some_and(|c| c.is_whitespace()) {
        position += 1;
    }
    if !matches!(chars.get(position), Some('=' | ':')) {
        return None;
    }
    position += 1;
    while chars.get(position).is_some_and(|c| c.is_whitespace()) {
        position += 1;
    }
    Some(position)
}

/// Matches Python's value group at `at`: a quoted string, or a bare run with an optional
/// `bearer` scheme in front of it.
fn match_assignment_value(chars: &[char], at: usize) -> Option<usize> {
    match chars.get(at) {
        Some('"') => match_quoted(chars, at, '"'),
        Some('\'') => match_quoted(chars, at, '\''),
        Some(_) => match_bare_value(chars, at),
        None => None,
    }
}

/// Matches `"…"` or `'…'` starting at `at`, returns the position after the closing quote.
fn match_quoted(chars: &[char], at: usize, quote: char) -> Option<usize> {
    let mut position = at + 1;
    while let Some(&current) = chars.get(position) {
        position += 1;
        if current == quote {
            return Some(position);
        }
    }
    None
}

/// Matches `(?:bearer\s+)?[^\s,;&"]+` at `at`.
fn match_bare_value(chars: &[char], at: usize) -> Option<usize> {
    if let Some(after_scheme) = match_ascii_case_insensitive(chars, at, "bearer") {
        let mut position = after_scheme;
        while chars.get(position).is_some_and(|c| c.is_whitespace()) {
            position += 1;
        }
        // Python backtracks the optional scheme group when no bare run follows it.
        if position > after_scheme {
            if let Some(end) = match_bare_run(chars, position) {
                return Some(end);
            }
        }
    }
    match_bare_run(chars, at)
}

/// Matches `[^\s,;&"]+` at `at`, returning the end of the run.
fn match_bare_run(chars: &[char], at: usize) -> Option<usize> {
    let mut position = at;
    while let Some(&current) = chars.get(position) {
        if current.is_whitespace() || matches!(current, ',' | ';' | '&' | '"') {
            break;
        }
        position += 1;
    }
    (position > at).then_some(position)
}

/// Matches `\bbearer\s+[A-Za-z0-9._~+/=-]{8,}` at `at`, returning the end of the credential.
fn match_bearer_token(chars: &[char], at: usize) -> Option<usize> {
    if !chars.get(at).is_some_and(|c| c.eq_ignore_ascii_case(&'b')) {
        return None;
    }
    let mut position = match_ascii_case_insensitive(chars, at, "bearer")?;
    let whitespace_start = position;
    while chars.get(position).is_some_and(|c| c.is_whitespace()) {
        position += 1;
    }
    if position == whitespace_start {
        return None;
    }
    let token_start = position;
    while chars
        .get(position)
        .is_some_and(|c| is_bearer_token_char(*c))
    {
        position += 1;
    }
    (position - token_start >= BEARER_TOKEN_MIN_LENGTH).then_some(position)
}

/// Matches an ASCII literal case-insensitively at `at`, returning the position after it.
fn match_ascii_case_insensitive(chars: &[char], at: usize, literal: &str) -> Option<usize> {
    let mut position = at;
    for expected in literal.chars() {
        let current = *chars.get(position)?;
        if !current.eq_ignore_ascii_case(&expected) {
            return None;
        }
        position += 1;
    }
    Some(position)
}

/// Characters of a bearer credential: `[A-Za-z0-9._~+/=-]`.
fn is_bearer_token_char(current: char) -> bool {
    current.is_ascii_alphanumeric() || matches!(current, '.' | '_' | '~' | '+' | '/' | '=' | '-')
}

/// Python's `\w` approximation for the word-boundary check, Unicode-aware like `\b` on `str`.
fn is_word_char(current: char) -> bool {
    current.is_alphanumeric() || current == '_'
}

/// Python's `" ".join(text.split())`: runs of whitespace become one space, edges are trimmed.
fn collapse_whitespace(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut pending_space = false;
    for current in value.chars() {
        if current.is_whitespace() {
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(current);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_a_password_assignment() {
        assert_eq!(mask_secrets(r#"password = "hunter2""#), "password = ***");
        assert_eq!(mask_secrets("password=hunter2"), "password=***");
        assert_eq!(mask_secrets("PASSWORD=hunter2"), "PASSWORD=***");
    }

    #[test]
    fn masks_an_authorization_header_as_one_unit() {
        // Python's value group swallows the `bearer ` scheme, so the whole credential goes.
        assert_eq!(
            mask_secrets("Authorization: Bearer abcdefgh12345"),
            "Authorization: ***"
        );
        assert_eq!(
            mask_secrets("authorization: BEARER abcdefgh12345"),
            "authorization: ***"
        );
        assert_eq!(
            mask_secrets("authorization=Bearer abcdefgh12345"),
            "authorization=***"
        );
    }

    #[test]
    fn masks_hyphenated_and_underscored_api_keys() {
        assert_eq!(mask_secrets("api-key=xyz"), "api-key=***");
        assert_eq!(mask_secrets("api_key=xyz"), "api_key=***");
        assert_eq!(mask_secrets("apikey=xyz"), "apikey=***");
        assert_eq!(mask_secrets("curl -H api-key: xyz"), "curl -H api-key: ***");
    }

    #[test]
    fn masks_a_quoted_client_secret() {
        assert_eq!(
            mask_secrets(r#"client_secret="s3cr3t value""#),
            "client_secret=***"
        );
        // Python parity: the separator must follow the key directly, so a quoted JSON key
        // (`"client-secret": …`) is not an assignment shape and stays readable.
        assert_eq!(
            mask_secrets(r#"{"client-secret": "s3cr3t"}"#),
            r#"{"client-secret": "s3cr3t"}"#
        );
        assert_eq!(
            mask_secrets(r#"client_secret='s3cr3t'"#),
            "client_secret=***"
        );
    }

    #[test]
    fn masks_the_rest_of_the_key_vocabulary() {
        for (input, expected) in [
            ("access_token=abc123", "access_token=***"),
            ("refresh-token: abc123", "refresh-token: ***"),
            ("session_token=abc123", "session_token=***"),
            ("id_token=abc123", "id_token=***"),
            ("cookie=abc123", "cookie=***"),
            ("passwd=abc123", "passwd=***"),
            ("secret=abc123", "secret=***"),
            ("private_key=abc123", "private_key=***"),
        ] {
            assert_eq!(mask_secrets(input), expected, "input was {input:?}");
        }
    }

    #[test]
    fn an_already_masked_value_is_stable() {
        for input in [
            "password=***",
            "Authorization: ***",
            "Bearer ***",
            "client_secret=***",
        ] {
            assert_eq!(mask_secrets(input), input, "input was {input:?}");
            assert_eq!(mask_secrets(&mask_secrets(input)), input);
        }
    }

    #[test]
    fn non_ascii_input_neither_panics_nor_breaks_boundaries() {
        let input = "密码=秘密 word=ünïcode";
        assert_eq!(mask_secrets(input), input);
        assert_eq!(
            mask_secrets("密码=秘密 password=hunter2 ünïcode"),
            "密码=秘密 password=*** ünïcode"
        );
        // `\b` before `password` fails after a word character, so Python leaves this alone too.
        assert_eq!(mask_secrets("épassword=hunter2"), "épassword=hunter2");
        // Multi-byte characters survive masking around them unchanged.
        assert_eq!(mask_secrets("→ secret=hunter2 ←"), "→ secret=*** ←");
    }

    #[test]
    fn an_unlabelled_credential_is_not_masked() {
        // Python masks credential-shaped *assignments*; prose stays readable.
        assert_eq!(
            mask_secrets("the password was rejected"),
            "the password was rejected"
        );
        // A key that is only a prefix of a sensitive word is not an assignment key.
        assert_eq!(mask_secrets("secret_key=abc"), "secret_key=abc");
        assert_eq!(mask_secrets("passwords=abc"), "passwords=abc");
    }

    #[test]
    fn masks_standalone_bearer_tokens() {
        assert_eq!(
            mask_bearer_tokens("token=bearer abcdefgh12345 next"),
            "token=Bearer *** next"
        );
        assert_eq!(mask_bearer_tokens("Bearer abcdefgh12345"), "Bearer ***");
        assert_eq!(
            mask_bearer_tokens("bearer eyJhbGciOi.J9+/==, done"),
            "Bearer ***, done"
        );
        // Fewer than eight token characters is not a credential.
        assert_eq!(mask_bearer_tokens("bearer abc"), "bearer abc");
        assert_eq!(
            mask_bearer_tokens("bearers abcdefgh12345"),
            "bearers abcdefgh12345"
        );
    }

    #[test]
    fn summary_collapses_whitespace_while_stacktrace_keeps_it() {
        let raw = "  boom\n\tAuthorization: Bearer abcdefgh12345  at x  ";
        assert_eq!(redact_error_summary(raw), "boom Authorization: *** at x");
        assert_eq!(
            redact_error_stacktrace(raw),
            "  boom\n\tAuthorization: ***  at x  "
        );
    }

    #[test]
    fn summaries_and_stacktraces_are_capped_without_splitting_characters() {
        let long = "é".repeat(ERROR_SUMMARY_MAX_LENGTH + 904);
        for output in [redact_error_summary(&long), redact_error_stacktrace(&long)] {
            assert_eq!(
                output,
                format!(
                    "{}...<OTel attribute truncated: 904 chars omitted>",
                    "é".repeat(ERROR_SUMMARY_MAX_LENGTH)
                )
            );
        }
        // A value at the cap is untouched, cap counting characters rather than bytes.
        let at_cap = "é".repeat(ERROR_SUMMARY_MAX_LENGTH);
        assert_eq!(redact_error_summary(&at_cap), at_cap);
        assert_eq!(redact_error_stacktrace(&at_cap), at_cap);
    }

    #[test]
    fn truncation_happens_after_masking() {
        let raw = format!("{} password=hunter2", "a".repeat(ERROR_SUMMARY_MAX_LENGTH));
        let summary = redact_error_summary(&raw);
        assert!(!summary.contains("hunter2"));
        assert!(summary.ends_with("...<OTel attribute truncated: 13 chars omitted>"));
    }

    #[test]
    fn hash_value_is_reused_from_the_tracer_redaction_layer() {
        let hashed = hash_value("secret prompt");
        assert!(hashed.starts_with(REDACTED_HASH_PREFIX));
        assert_eq!(hashed.len(), REDACTED_HASH_PREFIX.len() + 16);
        assert_eq!(hashed, crate::otel::redaction::hash_value("secret prompt"));
        // The error helpers never hash: they keep the reason readable.
        assert_eq!(redact_error_summary("boom"), "boom");
    }

    #[test]
    fn mask_constant_matches_python() {
        assert_eq!(MASK, "***");
        assert_eq!(BEARER_MASK, "Bearer ***");
        assert_eq!(ERROR_SUMMARY_MAX_LENGTH, 4096);
        assert!(BEARER_MASK.ends_with(MASK));
    }
}
