//! Sensitive field-name normalization and recognition.
//!
//! Ports `normalize_sensitive_name` and `is_sensitive_name` from
//! `openjiuwen.symphony.models._redaction`. Names are folded to a canonical
//! snake-case key and then matched against Python's exact-name set, its
//! sensitive-token set, its key-qualifier set, and its non-secret suffix list.

/// Field names that conventionally carry credential material verbatim.
///
/// The first 22 entries are Python's `_EXACT_SENSITIVE_NAMES`; `passwd`, `pwd`,
/// and `sas` are a documented Rust extension covering the common short spellings
/// Python reaches only through its `password` token.
const EXACT_SENSITIVE_NAMES: [&str; 25] = [
    "access_key",
    "access_key_id",
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "client_secret",
    "cookie",
    "credential",
    "credentials",
    "header",
    "headers",
    "password",
    "passwd",
    "private_key",
    "pwd",
    "refresh_token",
    "sas",
    "secret",
    "secret_access_key",
    "session_id",
    "token",
];

/// Name tokens that always imply credential material.
const SENSITIVE_TOKENS: [&str; 6] = [
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "password",
    "secret",
];

/// Tokens that turn a `key` token into a credential name.
const SENSITIVE_KEY_QUALIFIERS: [&str; 6] = [
    "api",
    "access",
    "encryption",
    "private",
    "secret",
    "signing",
];

/// Name suffixes that describe a secret rather than carrying one.
const NON_SECRET_SUFFIXES: [&str; 8] = [
    "_count",
    "_enabled",
    "_length",
    "_name",
    "_policy",
    "_present",
    "_required",
    "_type",
];

/// Normalizes snake, kebab, spaced, and camelCase identifiers.
///
/// `AWSAccessKeyId`, `aws-access-key-id`, and `aws_access_key_id` all normalize
/// to `aws_access_key_id`. Characters outside `[a-z0-9]` after case folding
/// collapse into a single `_`, and leading/trailing separators are dropped.
///
/// ```
/// use openjiuwen_symphony::redaction::normalize_sensitive_name;
///
/// assert_eq!(normalize_sensitive_name("AWSAccessKeyId"), "aws_access_key_id");
/// assert_eq!(normalize_sensitive_name(" API-Key "), "api_key");
/// ```
#[must_use]
pub fn normalize_sensitive_name(name: &str) -> String {
    let characters: Vec<char> = name.chars().collect();
    let mut split = String::with_capacity(name.len() + 8);
    for (index, current) in characters.iter().enumerate() {
        if index > 0 {
            let previous = characters[index - 1];
            let next = characters.get(index + 1).copied();
            let acronym_boundary = previous.is_ascii_uppercase()
                && current.is_ascii_uppercase()
                && next.is_some_and(|value| value.is_ascii_lowercase());
            let camel_boundary = (previous.is_ascii_lowercase() || previous.is_ascii_digit())
                && current.is_ascii_uppercase();
            if acronym_boundary || camel_boundary {
                split.push('_');
            }
        }
        split.push(*current);
    }
    let mut normalized = String::with_capacity(split.len());
    let mut pending_separator = false;
    for lowered in split.to_lowercase().chars() {
        if lowered.is_ascii_lowercase() || lowered.is_ascii_digit() {
            if pending_separator && !normalized.is_empty() {
                normalized.push('_');
            }
            pending_separator = false;
            normalized.push(lowered);
        } else {
            pending_separator = true;
        }
    }
    normalized
}

/// Reports whether a field name conventionally carries credential material.
///
/// Names ending in a non-secret suffix (`token_count`, `secret_name`) are never
/// sensitive; otherwise a name is sensitive when it is one of Python's exact
/// credential names, contains a sensitive token, ends in `_token`, or pairs a
/// `key` token with a qualifier such as `api` or `secret`.
///
/// ```
/// use openjiuwen_symphony::redaction::is_sensitive_name;
///
/// assert!(is_sensitive_name("client_secret"));
/// assert!(is_sensitive_name("X-Api-Key"));
/// assert!(!is_sensitive_name("token_count"));
/// assert!(!is_sensitive_name("query"));
/// ```
#[must_use]
pub fn is_sensitive_name(name: &str) -> bool {
    let normalized = normalize_sensitive_name(name);
    if NON_SECRET_SUFFIXES
        .iter()
        .any(|suffix| normalized.ends_with(suffix))
    {
        return false;
    }
    if EXACT_SENSITIVE_NAMES.contains(&normalized.as_str()) {
        return true;
    }
    let mut has_key = false;
    let mut has_qualifier = false;
    let mut has_sensitive_token = false;
    for token in normalized.split('_').filter(|token| !token.is_empty()) {
        has_key |= token == "key";
        has_qualifier |= SENSITIVE_KEY_QUALIFIERS.contains(&token);
        has_sensitive_token |= SENSITIVE_TOKENS.contains(&token);
    }
    if has_sensitive_token || normalized.ends_with("_token") {
        return true;
    }
    has_key && has_qualifier
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn names_fold_camel_kebab_and_spaces() {
        assert_eq!(normalize_sensitive_name("apiKey"), "api_key");
        assert_eq!(normalize_sensitive_name("API_KEY"), "api_key");
        assert_eq!(normalize_sensitive_name("APIKey"), "api_key");
        assert_eq!(
            normalize_sensitive_name("aws-access-key-id"),
            "aws_access_key_id"
        );
        assert_eq!(normalize_sensitive_name("  Secret  "), "secret");
        assert_eq!(
            normalize_sensitive_name("HTTPServerCert"),
            "http_server_cert"
        );
        assert_eq!(normalize_sensitive_name(""), "");
        assert_eq!(normalize_sensitive_name("___"), "");
        // Non-ASCII folds into separators instead of panicking.
        assert_eq!(normalize_sensitive_name("密码 token"), "token");
    }

    #[test]
    fn exact_names_tokens_and_qualifiers_are_sensitive() {
        for name in [
            "api_key",
            "apikey",
            "authorization",
            "cookie",
            "credentials",
            "headers",
            "private_key",
            "session_id",
            "refresh_token",
            "client_secret",
            "aws_secret_access_key",
            "secret_key",
            "signing_key",
            "encryptionKey",
            "access_token",
            "passwd",
            "pwd",
            "sas",
        ] {
            assert!(is_sensitive_name(name), "{name} should be sensitive");
        }
    }

    #[test]
    fn non_secret_names_are_not_sensitive() {
        for name in [
            "token_count",
            "token_enabled",
            "secret_name",
            "api_key_required",
            "password_policy",
            "query",
            "description",
            "key",
            "keyboard",
            "session",
            "author",
            "",
        ] {
            assert!(!is_sensitive_name(name), "{name} should not be sensitive");
        }
    }
}
