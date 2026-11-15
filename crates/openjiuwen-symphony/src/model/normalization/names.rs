//! I/O-name normalization and conservative plural aliasing.
//!
//! Ports `normalize_io_name` and `_conservative_io_name_alias` from Python
//! `openjiuwen.symphony.shared.fingerprint.normalization`. A raw I/O label (a card's
//! parameter name, or an LLM-written name in an extracted spec) is reduced to one
//! canonical `snake_case` token so two spellings of the same concept compare equal.
//!
//! # Deviations from Python
//!
//! - **Compatibility folding is approximated.** Python calls
//!   `unicodedata.normalize("NFKC", ...)`, which needs Unicode tables this crate does not
//!   depend on. [`normalize_io_name`] folds the full-width ASCII block
//!   (`U+FF01 ..= U+FF5E`) to ASCII and the Unicode space separators (ideographic space,
//!   no-break space, the `U+2000`-block spaces) to `' '`, which covers the cases an I/O
//!   label realistically carries. Compatibility ligatures, super/subscripts, Roman
//!   numerals, and circled forms are *not* decomposed.
//! - **Case folding is lowercase.** Python case-folds (`ß` → `ss`); Rust's
//!   [`char::to_lowercase`] leaves `ß` alone.
//! - **Every non-alphanumeric run collapses to one `_`, including underscores.** Python's
//!   `[^\w]+` treats `_` as a word character, so `read__file` keeps both underscores
//!   there; here `read__file` and `read-file` and `read file` all become `read_file`,
//!   matching [`normalize_name_key`](crate::model::normalize_name_key) and making the
//!   token idempotent under re-normalization.
//! - **A length bound exists.** Python has no cap; see [`MAX_IO_NAME_LEN`].

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::redaction::redact_sensitive_text;

/// The maximum length, in characters, of a normalized I/O name.
///
/// Python applies no bound; the cap is a Rust-only guard so a hostile multi-megabyte
/// label cannot become an unbounded metadata key. A longer token is cut at this many
/// characters and any trailing `_` left by the cut is removed, which keeps
/// [`normalize_io_name`] idempotent.
pub const MAX_IO_NAME_LEN: usize = 120;

/// The recommended resolution method: the token is a canonical vocabulary term.
pub const METHOD_VOCAB_EXACT: &str = "vocab_exact";
/// The recommended resolution method: the token is a recorded alias of a term.
pub const METHOD_VOCAB_ALIAS: &str = "vocab_alias";
/// The resolution method for a token the vocabulary has never seen.
pub const METHOD_VOCAB_UNSEEN: &str = "unseen";

/// Converts an I/O label to a deterministic `snake_case` semantic name.
///
/// The value is first passed through
/// [`redact_sensitive_text`] so a credential
/// smuggled into a name never reaches the output. It is then compatibility-folded,
/// trimmed, lowercased, and reduced to alphanumeric runs joined by single underscores;
/// leading and trailing separators are dropped and the result is bounded by
/// [`MAX_IO_NAME_LEN`]. When nothing is left, `fallback` is normalized with the same
/// rules and returned instead.
///
/// ```
/// use openjiuwen_symphony::model::normalize_io_name;
///
/// assert_eq!(normalize_io_name("  Read File (v2)! ", ""), "read_file_v2");
/// assert_eq!(normalize_io_name("ＦＵＬＬ－ＷＩＤＴＨ", ""), "full_width");
/// assert_eq!(normalize_io_name("!!!", "Input"), "input");
/// ```
#[must_use]
pub fn normalize_io_name(raw: &str, fallback: &str) -> String {
    let primary = normalize_token(raw);
    if primary.is_empty() {
        normalize_token(fallback)
    } else {
        primary
    }
}

/// Normalizes one label to its canonical token, without a fallback.
fn normalize_token(raw: &str) -> String {
    let safe = redact_sensitive_text(raw);
    let folded = fold_compatibility(&safe);
    let mut token = String::with_capacity(folded.len());
    let mut pending_separator = false;
    for character in folded.trim().chars() {
        if character.is_alphanumeric() {
            if pending_separator && !token.is_empty() {
                token.push('_');
            }
            pending_separator = false;
            token.extend(character.to_lowercase());
        } else {
            pending_separator = true;
        }
    }
    bound(token)
}

/// Approximates `unicodedata.normalize("NFKC", ...)` for the cases an I/O label needs.
fn fold_compatibility(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            // Unicode space separators, all of which NFKC maps to a plain space.
            '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => {
                folded.push(' ');
            }
            // The full-width ASCII block, which NFKC decomposes to ASCII.
            '\u{FF01}'..='\u{FF5E}' => {
                let ascii = u32::from(character) - 0xFEE0;
                folded.push(char::from_u32(ascii).unwrap_or(character));
            }
            _ => folded.push(character),
        }
    }
    folded
}

/// Cuts a token to [`MAX_IO_NAME_LEN`] characters and drops a trailing separator.
fn bound(token: String) -> String {
    if token.chars().count() <= MAX_IO_NAME_LEN {
        return token;
    }
    let truncated: String = token.chars().take(MAX_IO_NAME_LEN).collect();
    truncated.trim_end_matches('_').to_string()
}

/// Resolves only an unambiguous spelling variant to an existing canonical term.
///
/// Candidates are the Python plural folds (`…ies` → `…y`, trailing `s` → nothing, the
/// latter only for tokens longer than three characters that do not end in `ss`) plus,
/// as a documented Rust extension, the final `_`-separated segment of a multi-segment
/// token when that segment is at least three characters long (`user_name` → `name`).
/// Only candidates that are canonical names count, and the alias is recorded **only**
/// when exactly one distinct candidate matches — a tie returns `None`, so an ambiguous
/// token always stays unaliased.
///
/// ```
/// use openjiuwen_symphony::model::normalization::conservative_io_name_alias;
/// use std::collections::BTreeSet;
///
/// let canonical: BTreeSet<String> = ["prompt", "body", "bodie"]
///     .iter()
///     .map(|name| (*name).to_string())
///     .collect();
/// assert_eq!(conservative_io_name_alias("prompts", &canonical), Some("prompt".to_string()));
/// // `bodies` folds to both `body` and `bodie`, so the tie leaves it unaliased.
/// assert_eq!(conservative_io_name_alias("bodies", &canonical), None);
/// assert_eq!(conservative_io_name_alias("query", &canonical), None);
/// ```
#[must_use]
pub fn conservative_io_name_alias(
    token: &str,
    canonical_names: &BTreeSet<String>,
) -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if token.chars().count() > 4 {
        if let Some(stem) = token.strip_suffix("ies") {
            candidates.push(format!("{stem}y"));
        }
    }
    if token.chars().count() > 3 && token.ends_with('s') && !token.ends_with("ss") {
        candidates.push(token[..token.len() - 1].to_string());
    }
    if let Some((_, last)) = token.rsplit_once('_') {
        if last.chars().count() >= 3 {
            candidates.push(last.to_string());
        }
    }
    let mut matches: Vec<String> = candidates
        .into_iter()
        .filter(|candidate| candidate != token && canonical_names.contains(candidate))
        .collect();
    matches.sort();
    matches.dedup();
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

/// The resolution of one label against an observed-corpus I/O vocabulary.
///
/// Mirrors Python's `IONameResolution` dataclass: `raw` is the trimmed input, `token`
/// its syntax-normalized form, `normalized` the vocabulary term (or alias target) it
/// resolves to, `known` whether that exists, and `method` the
/// [`METHOD_VOCAB_EXACT`]/[`METHOD_VOCAB_ALIAS`]/[`METHOD_VOCAB_UNSEEN`] discriminant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IONameResolution {
    /// The trimmed input value.
    pub raw: String,
    /// The canonical token derived from `raw` by [`normalize_io_name`].
    pub token: String,
    /// The vocabulary term the token resolves to, when the vocabulary knows it.
    pub normalized: Option<String>,
    /// Whether `normalized` holds a vocabulary term.
    pub known: bool,
    /// How the resolution was reached; one of the `METHOD_VOCAB_*` constants.
    pub method: String,
}

impl IONameResolution {
    /// Builds the resolution for `raw` given its `token` and optional vocabulary term.
    #[must_use]
    pub fn new(
        raw: impl Into<String>,
        token: impl Into<String>,
        normalized: Option<String>,
    ) -> Self {
        let raw = raw.into();
        let token = token.into();
        let known = normalized.is_some();
        let method = if known {
            if normalized.as_deref() == Some(token.as_str()) {
                METHOD_VOCAB_EXACT
            } else {
                METHOD_VOCAB_ALIAS
            }
        } else {
            METHOD_VOCAB_UNSEEN
        };
        Self {
            raw,
            token,
            normalized,
            known,
            method: method.to_string(),
        }
    }

    /// Reports whether the vocabulary recognized the label.
    #[must_use]
    pub fn is_known(&self) -> bool {
        self.known
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn names_fold_case_separators_and_ascii_forms() {
        assert_eq!(normalize_io_name("Read File", ""), "read_file");
        assert_eq!(normalize_io_name("read__file", ""), "read_file");
        assert_eq!(normalize_io_name("--read-file--", ""), "read_file");
        assert_eq!(normalize_io_name("ｒｅａｄ　ｆｉｌｅ", ""), "read_file");
        assert_eq!(normalize_io_name("\u{00A0}query\u{00A0}", ""), "query");
        assert_eq!(normalize_io_name("Prompt", ""), "prompt");
    }

    #[test]
    fn names_fall_back_and_bound_length() {
        assert_eq!(normalize_io_name("", ""), "");
        assert_eq!(normalize_io_name("!!!", "Result"), "result");
        let long = "a".repeat(MAX_IO_NAME_LEN + 40);
        let bounded = normalize_io_name(&long, "");
        assert_eq!(bounded.chars().count(), MAX_IO_NAME_LEN);
        assert_eq!(normalize_io_name(&bounded, ""), bounded);
    }

    #[test]
    fn names_are_idempotent_and_redacted() {
        let name = normalize_io_name("API Key = sk-live-0123456789abcdef", "");
        assert!(!name.contains("sk-live"));
        assert_eq!(normalize_io_name(&name, ""), name);
        assert_eq!(normalize_io_name("Über Ärger", ""), "über_ärger");
    }

    #[test]
    fn aliases_require_one_unambiguous_candidate() {
        let canonical: BTreeSet<String> = ["prompt", "body", "bodie", "name"]
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert_eq!(
            conservative_io_name_alias("prompts", &canonical),
            Some("prompt".to_string())
        );
        // `bodies` folds to both `body` and `bodie`, so it stays unaliased.
        assert_eq!(conservative_io_name_alias("bodies", &canonical), None);
        assert_eq!(
            conservative_io_name_alias("user_name", &canonical),
            Some("name".to_string())
        );
        assert_eq!(conservative_io_name_alias("query", &canonical), None);
    }

    #[test]
    fn resolution_reports_its_method() {
        let exact = IONameResolution::new("Prompt", "prompt", Some("prompt".to_string()));
        assert!(exact.is_known());
        assert_eq!(exact.method, METHOD_VOCAB_EXACT);
        let aliased = IONameResolution::new("prompts", "prompts", Some("prompt".to_string()));
        assert_eq!(aliased.method, METHOD_VOCAB_ALIAS);
        assert!(aliased.is_known());
        let unseen = IONameResolution::new("thing", "thing", None);
        assert_eq!(unseen.method, METHOD_VOCAB_UNSEEN);
        assert!(!unseen.is_known());
    }
}
