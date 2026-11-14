//! Persisted source-link book and publication-time target resolution.
//!
//! The portable subset of Python's `personal_context/source_link_book.py`: the
//! locator key ([`locator_key`]), the registration-time target resolver
//! ([`resolve_target`]), the identified link book ([`SourceLinkBook`]) with its
//! atomic `JSON` persistence, and the label-only preview renderer
//! ([`SourceLinkBook::preview`]).
//!
//! **Deferred from Python.** `normalize_source_links` (Markdown-token rewriting
//! during registration) and `resolve_source_links` (rewriting every published
//! page during publication) are not ported: this crate has no Markdown parser,
//! and the Python versions rewrite whole prose pages. This module owns the book
//! and target resolution plus label-only preview rendering of already-registered
//! tokens; a caller that needs registration or page rewriting renders the
//! `[label](pcs-source-link:<identity>)` token itself.
//!
//! **Wire shape.** A book is persisted as
//! `{"version": 1, "source_links": {<identity>: <record>}}`, where a record keeps
//! the Python field names (`source_id`, `origin`, `original_target`,
//! `resolved_target`, `label`) and the identity is the first
//! [`IDENTITY_HEX_CHARS`] lowercase hex characters of the `SHA-256` digest of the
//! record's canonical `JSON` (`json.dumps(record, sort_keys=True)`), so books
//! written by the Python implementation stay readable.
//!
//! **Deviations.** Python returns an empty resolved target for an unsupported
//! scheme and swallows `ValueError` from `urlsplit`; this port returns a typed
//! `Validation` error instead. The relative-URL resolver covers the practical
//! reference forms (absolute, scheme-relative, root-relative, empty and
//! dot-segment relative) rather than the full `RFC 3986` algorithm, and `HTML`
//! unescaping covers the five entities that extracted anchors contain.
// Size note: ~915 lines of implementation plus an inline test module; the parity
// tests stay beside the code they verify.
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::rails::sha256::sha256_prefix;

use super::atomic::{read_json_opt, write_json_atomic};
use super::source::{normalize_source_locator, source_id_for_locator};

/// Prefix of a source-link token in rendered Markdown.
pub const LINK_TOKEN_PREFIX: &str = "pcs-source-link:";

/// Number of hex characters kept from the `SHA-256` digest of a record.
pub const IDENTITY_HEX_CHARS: usize = 32;

/// Version of the persisted link-book document.
pub const LINK_BOOK_VERSION: u32 = 1;

/// The document label used in every persistence error message.
const BOOK_LABEL: &str = "source link book";

/// The message for a record whose identity or source id does not match.
const IDENTITY_ERROR: &str = "source link record identity is invalid";

/// The message for a blank origin handed to [`SourceLinkBook::register`].
const ORIGIN_ERROR: &str = "source link origin is invalid";

/// The five `HTML` entities replaced before a target is resolved.
///
/// Python's `html.unescape` knows every named and numeric entity; the sources
/// this port reads only produce these five.
const HTML_ENTITIES: [(&str, char); 5] = [
    ("&amp;", '&'),
    ("&lt;", '<'),
    ("&gt;", '>'),
    ("&quot;", '"'),
    ("&#39;", '\''),
];

/// One registered source link: a target as written, its resolved locator and the
/// label shown to the reader; the field set and names match Python's record
/// exactly, and unknown fields in a stored document are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SourceLink {
    /// Stable id of the source the link was written on, `src_` plus 32 hex.
    pub source_id: String,
    /// Locator of the page the link was written on, as registered.
    pub origin: String,
    /// Target exactly as it appeared in the registered document.
    pub original_target: String,
    /// Locator the target resolves to, see [`resolve_target`].
    pub resolved_target: String,
    /// Reader-facing text of the link.
    pub label: String,
}

impl SourceLink {
    /// Builds a record from already-derived parts.
    ///
    /// [`SourceLinkBook::register`] and [`SourceLinkBook::insert`] derive the
    /// identity and validate the source id; prefer them over storing a hand-built
    /// record.
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        origin: impl Into<String>,
        original_target: impl Into<String>,
        resolved_target: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            origin: origin.into(),
            original_target: original_target.into(),
            resolved_target: resolved_target.into(),
            label: label.into(),
        }
    }

    /// Returns the book identity of this record: the first
    /// [`IDENTITY_HEX_CHARS`] characters of the `SHA-256` digest of the canonical
    /// `JSON` with alphabetically sorted keys, matching Python's
    /// `json.dumps(record, sort_keys=True)`.
    ///
    /// # Errors
    ///
    /// The canonical form is written directly and cannot fail today; the `Result`
    /// keeps the identity surface uniform with the book operations and leaves room
    /// for a fallible encoder.
    #[must_use = "the identity is the key the record is stored under"]
    #[allow(clippy::unnecessary_wraps)]
    pub fn identity(&self) -> Result<String> {
        let record: BTreeMap<&str, &str> = BTreeMap::from([
            ("label", self.label.as_str()),
            ("origin", self.origin.as_str()),
            ("original_target", self.original_target.as_str()),
            ("resolved_target", self.resolved_target.as_str()),
            ("source_id", self.source_id.as_str()),
        ]);
        Ok(sha256_prefix(
            canonical_json(&record).as_bytes(),
            IDENTITY_HEX_CHARS,
        ))
    }

    /// Returns whether the record is internally consistent: its identity is
    /// derivable and its `source_id` is the one [`source_id_for_locator`] derives
    /// from `origin`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.identity().is_ok() && record_identity_matches(self)
    }
}

/// Returns whether `link`'s `source_id` is the one its `origin` derives.
fn record_identity_matches(link: &SourceLink) -> bool {
    match source_id_for_locator(&link.origin) {
        Ok(expected) => expected == link.source_id,
        Err(_) => false,
    }
}

/// Renders `record` as Python's `json.dumps(record, sort_keys=True)` would: a
/// [`BTreeMap`] iterates alphabetically, separators carry Python's default single
/// spaces, and strings use Python's `ensure_ascii` escaping.
fn canonical_json(record: &BTreeMap<&str, &str>) -> String {
    let mut rendered = String::from("{");
    for (index, (key, value)) in record.iter().enumerate() {
        if index > 0 {
            rendered.push_str(", ");
        }
        push_json_string(&mut rendered, key);
        rendered.push_str(": ");
        push_json_string(&mut rendered, value);
    }
    rendered.push('}');
    rendered
}

/// Appends `value` as a `JSON` string literal with Python's `ensure_ascii`
/// escaping: quote, backslash and the five short control escapes stay short,
/// every other control character and every non-ASCII character becomes a
/// lowercase `\uXXXX` escape (a surrogate pair above the basic plane).
fn push_json_string(rendered: &mut String, value: &str) {
    rendered.push('"');
    for character in value.chars() {
        match character {
            '"' => rendered.push_str("\\\""),
            '\\' => rendered.push_str("\\\\"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            '\u{8}' => rendered.push_str("\\b"),
            '\u{c}' => rendered.push_str("\\f"),
            ' '..='~' => rendered.push(character),
            escaped => {
                let mut units = [0_u16; 2];
                for unit in escaped.encode_utf16(&mut units) {
                    push_unicode_escape(rendered, *unit);
                }
            }
        }
    }
    rendered.push('"');
}

/// Appends one lowercase `\uXXXX` escape for a `UTF-16` code unit.
fn push_unicode_escape(rendered: &mut String, code_unit: u16) {
    const HEX: [u8; 16] = *b"0123456789abcdef";
    rendered.push_str("\\u");
    for shift in [12_u16, 8, 4, 0] {
        let digit = (code_unit >> shift) & 0xF;
        rendered.push(char::from(HEX[usize::from(digit)]));
    }
}

/// Returns the compiled preview pattern, or `None` if it failed to compile.
fn link_pattern() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| {
            Regex::new(&format!(
                r"\[([^\]\n]*)\]\({}([0-9a-f]{{32}})\)",
                regex::escape(LINK_TOKEN_PREFIX)
            ))
            .ok()
        })
        .as_ref()
}

/// Resolves `target`, as written on the page at `origin`, into a comparable
/// locator.
///
/// The port of Python's `_resolve_target`: the target is `HTML`-unescaped and
/// freed of backslash escapes before ASCII punctuation; a Windows drive or `UNC`
/// target becomes its [`locator_key`]; `file`, `http` and `https` targets become
/// their [`locator_key`]; and a scheme-less target is resolved against `origin` —
/// through the practical `urljoin` forms when the origin is `http(s)`, or by
/// joining the origin's directory when it is a `POSIX` or Windows path. An empty
/// target path keeps the origin's key.
///
/// # Errors
///
/// `Validation` for a scheme other than `file`, `http` or `https`, and for a
/// target that cannot be parsed (an unbalanced bracketed authority).
pub fn resolve_target(origin: &str, target: &str) -> Result<String> {
    let target = strip_backslash_escapes(&html_unescape(target));
    if is_windows_path(&target) {
        return Ok(locator_key(&target));
    }
    let parsed = split_url(&target);
    if !parsed.brackets_balanced {
        return Err(Error::validation(format!(
            "source link target is not resolvable: {target}"
        )));
    }
    if !parsed.scheme.is_empty() {
        if !matches!(parsed.scheme.as_str(), "file" | "http" | "https") {
            return Err(Error::validation(format!(
                "unsupported source link scheme {:?}",
                parsed.scheme
            )));
        }
        return Ok(locator_key(&target));
    }
    if is_http_locator(origin) {
        return Ok(locator_key(&url_join(origin, &target)));
    }
    let path = percent_decode(parsed.path);
    let base = locator_key(origin);
    if path.is_empty() {
        return Ok(base);
    }
    let joined = if is_windows_path(&base) {
        nt_join(&nt_dirname(&base), &path)
    } else {
        posix_join(&posix_dirname(&base), &path)
    };
    Ok(locator_key(&joined))
}

/// Reduces a locator to the key two spellings of the same target share.
///
/// A Windows drive or `UNC` locator is normalised with its separators (`\`) and
/// lowercased; an `http(s)` locator goes through [`normalize_source_locator`]
/// (falling back to the input when that rejects it); a `file:` `URL` contributes
/// its percent-decoded path and host, normalised recursively; anything else is
/// normalised like `posixpath.normpath`.
#[must_use]
pub fn locator_key(locator: &str) -> String {
    if is_windows_path(locator) {
        return nt_normcase(&nt_normpath(locator));
    }
    let parsed = split_url(locator);
    match parsed.scheme.as_str() {
        "http" | "https" => {
            normalize_source_locator(locator).unwrap_or_else(|_| locator.to_string())
        }
        "file" => {
            let mut path = percent_decode(parsed.path);
            if !parsed.authority.is_empty() {
                path = posix_join(
                    &posix_join("//", parsed.authority),
                    path.trim_start_matches('/'),
                );
            }
            // `/C:/dir` is a drive path: drop the leading slash before recursing.
            let bytes = path.as_bytes();
            if bytes.len() >= 4
                && bytes[0] == b'/'
                && bytes[1].is_ascii_alphabetic()
                && bytes[2] == b':'
                && bytes[3] == b'/'
            {
                path = path[1..].to_string();
            }
            locator_key(&path)
        }
        _ => posix_normpath(locator),
    }
}

/// Returns whether `value` starts with a drive letter followed by a separator or
/// with a `UNC` prefix.
fn is_windows_path(value: &str) -> bool {
    if value.starts_with("\\\\") {
        return true;
    }
    let bytes = value.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Returns whether `value` is an `http` or `https` locator.
fn is_http_locator(value: &str) -> bool {
    matches!(split_url(value).scheme.as_str(), "http" | "https")
}

/// The parts of a locator that [`locator_key`] and [`resolve_target`] need.
struct UrlParts<'a> {
    scheme: String,
    authority: &'a str,
    path: &'a str,
    query: &'a str,
    fragment: &'a str,
    /// Whether the locator carried an explicit `//` authority marker.
    has_authority: bool,
    /// Whether a bracketed host, if any, is balanced.
    brackets_balanced: bool,
}

/// Splits a locator the way Python's `urlsplit` does for the practical cases: the
/// scheme must be alphabetic and made of scheme characters, the authority ends at
/// the first `/`, `?` or `#`, and the fragment splits before the query.
fn split_url(value: &str) -> UrlParts<'_> {
    let (scheme, remainder) = match value.find(':') {
        Some(index) if index > 0 => {
            let candidate = &value[..index];
            let valid = candidate
                .chars()
                .next()
                .is_some_and(|first| first.is_ascii_alphabetic())
                && candidate.chars().all(is_scheme_char);
            if valid {
                (candidate.to_ascii_lowercase(), &value[index + 1..])
            } else {
                (String::new(), value)
            }
        }
        _ => (String::new(), value),
    };
    let (authority, remainder, has_authority) = match remainder.strip_prefix("//") {
        Some(after) => {
            let end = after.find(['/', '?', '#']).unwrap_or(after.len());
            (&after[..end], &after[end..], true)
        }
        None => ("", remainder, false),
    };
    let brackets_balanced = authority.contains('[') == authority.contains(']');
    let (remainder, fragment) = match remainder.split_once('#') {
        Some((before, after)) => (before, after),
        None => (remainder, ""),
    };
    let (path, query) = match remainder.split_once('?') {
        Some((before, after)) => (before, after),
        None => (remainder, ""),
    };
    UrlParts {
        scheme,
        authority,
        path,
        query,
        fragment,
        has_authority,
        brackets_balanced,
    }
}

/// Returns whether `character` may continue a `URL` scheme.
fn is_scheme_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
}

/// Resolves `reference` against the absolute `origin`: an absolute reference
/// keeps its scheme, a scheme-relative one inherits it, a root-relative one
/// replaces the path, a dot-segment-relative one merges with the origin's
/// directory, and an empty one keeps the origin. Query and fragment are carried
/// through (they are dropped later by [`locator_key`] for `http(s)`).
fn url_join(origin: &str, reference: &str) -> String {
    let base = split_url(origin);
    let target = split_url(reference);
    let authority = if target.has_authority {
        target.authority
    } else {
        base.authority
    };
    let path = if target.has_authority {
        remove_dot_segments(target.path)
    } else if target.path.is_empty() {
        base.path.to_string()
    } else if target.path.starts_with('/') {
        remove_dot_segments(target.path)
    } else {
        remove_dot_segments(&merge_paths(&base, target.path))
    };
    let query = if !target.query.is_empty() {
        target.query
    } else if target.path.is_empty() && !target.has_authority {
        base.query
    } else {
        ""
    };
    let scheme = if target.scheme.is_empty() {
        base.scheme.as_str()
    } else {
        target.scheme.as_str()
    };

    let mut joined = String::new();
    if !scheme.is_empty() {
        joined.push_str(scheme);
        joined.push(':');
    }
    if target.has_authority || !authority.is_empty() {
        joined.push_str("//");
        joined.push_str(authority);
    }
    joined.push_str(&path);
    if !query.is_empty() {
        joined.push('?');
        joined.push_str(query);
    }
    if !target.fragment.is_empty() {
        joined.push('#');
        joined.push_str(target.fragment);
    }
    joined
}

/// Merges a relative path with the base path, per `RFC 3986` §5.2.3.
fn merge_paths(base: &UrlParts<'_>, reference_path: &str) -> String {
    if base.path.is_empty() && base.has_authority {
        return format!("/{reference_path}");
    }
    match base.path.rfind('/') {
        Some(index) => format!("{}{reference_path}", &base.path[..=index]),
        None => reference_path.to_string(),
    }
}

/// Removes `.` and `..` segments, keeping a leading `/` and a trailing `/` that
/// came from those segments.
fn remove_dot_segments(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    let trailing = path.ends_with('/') || path.ends_with("/.") || path.ends_with("/..");
    let mut normalized = String::new();
    if absolute {
        normalized.push('/');
    }
    normalized.push_str(&segments.join("/"));
    if trailing && !segments.is_empty() {
        normalized.push('/');
    }
    normalized
}

/// Normalises a `POSIX` path like `posixpath.normpath`.
fn posix_normpath(path: &str) -> String {
    if path.is_empty() {
        return String::from(".");
    }
    let initial_slashes = if path.starts_with("//") && !path.starts_with("///") {
        2
    } else {
        usize::from(path.starts_with('/'))
    };
    let mut components: Vec<&str> = Vec::new();
    for component in path.split('/') {
        if component.is_empty() || component == "." {
            continue;
        }
        if component != ".."
            || (initial_slashes == 0 && components.is_empty())
            || components.last() == Some(&"..")
        {
            components.push(component);
        } else if !components.is_empty() {
            components.pop();
        }
    }
    let joined = components.join("/");
    let normalized = match initial_slashes {
        0 => joined,
        1 => format!("/{joined}"),
        _ => format!("//{joined}"),
    };
    if normalized.is_empty() {
        String::from(".")
    } else {
        normalized
    }
}

/// Joins a `POSIX` base path with a path, like `posixpath.join`.
fn posix_join(base: &str, path: &str) -> String {
    if path.starts_with('/') {
        return path.to_string();
    }
    if base.is_empty() || base.ends_with('/') {
        return format!("{base}{path}");
    }
    format!("{base}/{path}")
}

/// Returns the directory of a `POSIX` path, like `posixpath.dirname`.
fn posix_dirname(path: &str) -> String {
    let Some(index) = path.rfind('/') else {
        return String::new();
    };
    let mut head = &path[..=index];
    if head.chars().any(|character| character != '/') {
        head = head.trim_end_matches('/');
    }
    head.to_string()
}

/// Normalises a Windows path like `ntpath.normpath`.
fn nt_normpath(path: &str) -> String {
    let normalized = path.replace('/', "\\");
    let (drive, root, tail) = nt_split_root(&normalized);
    let mut prefix = String::from(drive);
    prefix.push_str(root);
    let mut components: Vec<&str> = Vec::new();
    for component in tail.split('\\') {
        if component.is_empty() || component == "." {
            continue;
        }
        if component == ".." {
            match components.last() {
                Some(last) if *last != ".." => {
                    components.pop();
                }
                None if root.is_empty() => components.push(component),
                None => {}
                Some(_) => components.push(component),
            }
        } else {
            components.push(component);
        }
    }
    if prefix.is_empty() && components.is_empty() {
        return String::from(".");
    }
    prefix.push_str(&components.join("\\"));
    prefix
}

/// Lowercases a Windows path and folds `/` to `\`, like `ntpath.normcase`.
fn nt_normcase(path: &str) -> String {
    path.replace('/', "\\").to_lowercase()
}

/// Joins a Windows base path with a path, like `ntpath.join` for the forms this
/// module reaches (a relative path, a root-relative path or an empty path).
fn nt_join(base: &str, path: &str) -> String {
    let base = base.replace('/', "\\");
    let path = path.replace('/', "\\");
    let (drive, root, tail) = nt_split_root(&base);
    let (path_drive, path_root, path_tail) = nt_split_root(&path);
    let drive = if path_drive.is_empty() {
        drive
    } else {
        path_drive
    };
    if !path_root.is_empty() {
        return format!("{drive}{path_root}{path_tail}");
    }
    let mut combined = String::from(tail);
    if !combined.is_empty() && !combined.ends_with('\\') {
        combined.push('\\');
    }
    combined.push_str(path_tail);
    format!("{drive}{root}{combined}")
}

/// Returns the directory of a Windows path, like `ntpath.dirname`.
fn nt_dirname(path: &str) -> String {
    let normalized = path.replace('/', "\\");
    let (drive, root, tail) = nt_split_root(&normalized);
    let head = match tail.rfind('\\') {
        Some(index) => &tail[..=index],
        None => "",
    };
    format!("{drive}{root}{}", head.trim_end_matches('\\'))
}

/// Splits a Windows path into its drive/`UNC` prefix, its root separator and its
/// tail, like `ntpath.splitroot` after separators have been folded to `\`.
fn nt_split_root(path: &str) -> (&str, &str, &str) {
    let bytes = path.as_bytes();
    if bytes.first() == Some(&b'\\') {
        if bytes.get(1) == Some(&b'\\') {
            let start = if bytes.len() >= 8 && bytes[..8].eq_ignore_ascii_case(b"\\\\?\\UNC\\") {
                8
            } else {
                2
            };
            let Some(index) = path[start..].find('\\').map(|offset| start + offset) else {
                return (path, "", "");
            };
            let Some(second) = path[index + 1..]
                .find('\\')
                .map(|offset| index + 1 + offset)
            else {
                return (path, "", "");
            };
            (&path[..second], &path[second..=second], &path[second + 1..])
        } else {
            ("", &path[..1], &path[1..])
        }
    } else if bytes.get(1) == Some(&b':') {
        if bytes.get(2) == Some(&b'\\') {
            (&path[..2], &path[2..3], &path[3..])
        } else {
            (&path[..2], "", &path[2..])
        }
    } else {
        ("", "", path)
    }
}

/// Replaces the five `HTML` entities of [`HTML_ENTITIES`] in a single pass, so
/// `&amp;lt;` becomes `&lt;` and not `<`.
fn html_unescape(value: &str) -> String {
    let mut unescaped = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(index) = rest.find('&') {
        unescaped.push_str(&rest[..index]);
        rest = &rest[index..];
        let mut matched = false;
        for (entity, replacement) in HTML_ENTITIES {
            if let Some(after) = rest.strip_prefix(entity) {
                unescaped.push(replacement);
                rest = after;
                matched = true;
                break;
            }
        }
        if !matched {
            unescaped.push('&');
            rest = &rest[1..];
        }
    }
    unescaped.push_str(rest);
    unescaped
}

/// Drops a backslash that escapes an ASCII punctuation character, keeping every
/// other backslash, like Python's `\\([punct])` rewrite.
fn strip_backslash_escapes(value: &str) -> String {
    let mut unescaped = String::with_capacity(value.len());
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '\\' {
            unescaped.push(character);
            continue;
        }
        match characters.peek() {
            Some(next) if next.is_ascii_punctuation() => {
                unescaped.push(*next);
                characters.next();
            }
            _ => unescaped.push('\\'),
        }
    }
    unescaped
}

/// Percent-decodes a `URL` path, leaving malformed escapes untouched and
/// replacing an invalid byte sequence with `U+FFFD`, like `urllib.parse.unquote`.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let (Some(high), Some(low)) =
                (hex_digit(bytes[index + 1]), hex_digit(bytes[index + 2]))
            {
                decoded.push((high << 4) | low);
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// Returns the value of one hexadecimal digit.
fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// The persisted link-book document.
#[derive(Serialize)]
struct BookDocument<'a> {
    version: u32,
    source_links: &'a BTreeMap<String, SourceLink>,
}

/// Registered source links, keyed by their identity.
///
/// The book deduplicates identical links (one identity per record) and keeps
/// identity order through its [`BTreeMap`], so loaded and freshly built books
/// serialize identically.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceLinkBook {
    entries: BTreeMap<String, SourceLink>,
}

impl SourceLinkBook {
    /// Returns an empty book.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of registered links.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the book holds no link.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the link registered under `identity`.
    #[must_use]
    pub fn get(&self, identity: &str) -> Option<&SourceLink> {
        self.entries.get(identity)
    }

    /// Iterates over every identity and its link in identity order.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &SourceLink)> {
        self.entries.iter()
    }

    /// Returns every identity in identity order.
    #[must_use]
    pub fn identities(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Registers the link `label` → `original_target` written on `origin` and
    /// returns its identity.
    ///
    /// The source id and the resolved target are derived here, so callers never
    /// supply an identity. Registering the same origin, target and label twice
    /// keeps one entry; a different label is a different link.
    ///
    /// # Errors
    ///
    /// `Validation` for a blank origin, for a record [`SourceLinkBook::insert`]
    /// rejects, and for a target [`resolve_target`] cannot resolve.
    pub fn register(&mut self, origin: &str, original_target: &str, label: &str) -> Result<String> {
        if origin.trim().is_empty() {
            return Err(Error::validation(ORIGIN_ERROR));
        }
        let source_id = source_id_for_locator(origin)?;
        let resolved_target = resolve_target(origin, original_target)?;
        self.insert(SourceLink::new(
            source_id,
            origin,
            original_target,
            resolved_target,
            label,
        ))
    }

    /// Stores `link` under its recomputed identity and returns that identity.
    ///
    /// Storing an existing identity replaces it; because the identity covers
    /// every field, that can only happen for an identical record.
    ///
    /// # Errors
    ///
    /// `Validation` when `link`'s `source_id` is not the one its `origin`
    /// derives.
    pub fn insert(&mut self, link: SourceLink) -> Result<String> {
        let identity = link.identity()?;
        if !record_identity_matches(&link) {
            return Err(Error::validation(IDENTITY_ERROR));
        }
        self.entries.insert(identity.clone(), link);
        Ok(identity)
    }

    /// Merges `other` into this book; on a shared identity `other` wins.
    ///
    /// Identical identities name identical records, so the merge is effectively a
    /// union in practice.
    pub fn merge(&mut self, other: &Self) {
        for (identity, link) in &other.entries {
            self.entries.insert(identity.clone(), link.clone());
        }
    }

    /// Renders `markdown` with every `[label](pcs-source-link:<identity>)` token
    /// reduced to `label`.
    ///
    /// Every other byte of `markdown` is preserved: ordinary links, prose,
    /// program tokens and malformed source-link tokens stay untouched. Bounded
    /// summaries use this so a token never consumes their budget.
    #[must_use]
    pub fn preview(markdown: &str) -> String {
        match link_pattern() {
            Some(pattern) => pattern.replace_all(markdown, "$1").into_owned(),
            None => markdown.to_string(),
        }
    }

    /// Writes the book to `path` atomically as
    /// `{"version": 1, "source_links": {<identity>: <record>}}`.
    ///
    /// # Errors
    ///
    /// `Config` for a path that cannot be created, encoded or replaced.
    pub fn save(&self, path: &Path) -> Result<()> {
        let document = BookDocument {
            version: LINK_BOOK_VERSION,
            source_links: &self.entries,
        };
        write_json_atomic(path, &document, BOOK_LABEL)
    }

    /// Reads a book from `path`, returning an empty book when the file is absent.
    ///
    /// Unknown fields (in the document and in every record) and a missing
    /// `version` are tolerated, so a book written by a later schema stays
    /// readable. Each stored identity must equal the recomputed identity of its
    /// record, and each record's `source_id` must be the one its `origin`
    /// derives.
    ///
    /// # Errors
    ///
    /// `Validation` for malformed `JSON`, a document that is not an object, a
    /// `source_links` member that is not an object, a record that does not decode,
    /// or a record whose identity or source id does not match; `Config` for a file
    /// that cannot be read.
    pub fn load(path: &Path) -> Result<Self> {
        let Some(document) = read_json_opt::<Value>(path, BOOK_LABEL)? else {
            return Ok(Self::new());
        };
        let Value::Object(mut document) = document else {
            return Err(Error::validation("source link book is invalid"));
        };
        let records = document
            .remove("source_links")
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        let Value::Object(records) = records else {
            return Err(Error::validation("source link book is invalid"));
        };
        let mut book = Self::new();
        for (identity, value) in records {
            let link: SourceLink = serde_json::from_value(value)
                .map_err(|_| Error::validation("source link record is invalid"))?;
            if link.identity()? != identity || !record_identity_matches(&link) {
                return Err(Error::validation(IDENTITY_ERROR));
            }
            book.entries.insert(identity, link);
        }
        Ok(book)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use crate::rails::sha256::sha256_hex;

    use super::*;

    fn sample_link() -> SourceLink {
        SourceLink::new(
            "src_0123456789abcdef0123456789abcdef",
            "https://user:pw@example.com/notes/readme.md",
            "../img/a.png",
            "https://example.com/img/a.png",
            "image",
        )
    }

    fn sample_book() -> SourceLinkBook {
        let mut book = SourceLinkBook::new();
        book.register("/notes/readme.md", "../img/a.png", "A")
            .unwrap();
        book.register("/notes/readme.md", "b.png", "B").unwrap();
        book
    }

    #[test]
    fn identity_is_stable_and_field_sensitive() {
        let link = sample_link();
        let identity = link.identity().unwrap();
        assert_eq!(identity.len(), IDENTITY_HEX_CHARS);
        assert!(identity
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
        assert_eq!(link.identity().unwrap(), identity);

        let record = BTreeMap::from([
            ("label", link.label.as_str()),
            ("origin", link.origin.as_str()),
            ("original_target", link.original_target.as_str()),
            ("resolved_target", link.resolved_target.as_str()),
            ("source_id", link.source_id.as_str()),
        ]);
        assert_eq!(
            identity,
            sha256_hex(canonical_json(&record))[..IDENTITY_HEX_CHARS]
        );

        for mutation in [
            SourceLink {
                source_id: String::from("src_ffffffffffffffffffffffffffffffff"),
                ..link.clone()
            },
            SourceLink {
                origin: String::from("https://example.com/other.md"),
                ..link.clone()
            },
            SourceLink {
                original_target: String::from("a.png"),
                ..link.clone()
            },
            SourceLink {
                resolved_target: String::from("https://example.com/other.png"),
                ..link.clone()
            },
            SourceLink {
                label: String::from("Image"),
                ..link.clone()
            },
        ] {
            assert_ne!(mutation.identity().unwrap(), identity);
        }
    }

    /// Reference value produced by Python's
    /// `hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:32]`
    /// for this record, whose label exercises `ensure_ascii` escaping.
    #[test]
    fn identity_matches_python_canonical_json() {
        let link = SourceLink::new(
            "src_0123456789abcdef0123456789abcdef",
            "https://user:pw@example.com/notes/readme.md",
            "../img/a.png",
            "https://example.com/img/a.png",
            "图 A",
        );
        assert_eq!(link.identity().unwrap(), "e71ca92f0517d46ffaeb9c6fb3b24d09");
    }

    #[test]
    fn register_derives_source_id_and_resolved_target() {
        let mut book = SourceLinkBook::new();
        let origin = "https://user:pw@Example.com/notes/readme.md";
        let identity = book.register(origin, "../img/a.png", "image").unwrap();
        let link = book.get(&identity).unwrap();

        assert_eq!(
            link.source_id,
            source_id_for_locator("https://example.com/notes/readme.md").unwrap()
        );
        assert_eq!(link.resolved_target, "https://example.com/img/a.png");
        assert_eq!(link.original_target, "../img/a.png");
        assert_eq!(link.label, "image");
        assert_eq!(link.identity().unwrap(), identity);
        assert!(link.is_valid());

        assert_eq!(
            book.register(origin, "../img/a.png", "image").unwrap(),
            identity
        );
        assert_eq!(book.len(), 1);
        let other_label = book.register(origin, "../img/a.png", "Image").unwrap();
        assert_ne!(other_label, identity);
        assert_eq!(book.len(), 2);
    }

    #[test]
    fn register_rejects_a_blank_origin() {
        let error = SourceLinkBook::new()
            .register("   ", "a.png", "a")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains(ORIGIN_ERROR));
    }

    #[test]
    fn insert_rejects_a_mismatched_source_id() {
        let mut book = SourceLinkBook::new();
        let error = book
            .insert(SourceLink::new(
                "src_ffffffffffffffffffffffffffffffff",
                "/notes/readme.md",
                "a.png",
                "/notes/a.png",
                "a",
            ))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains(IDENTITY_ERROR));
        assert!(book.is_empty());
    }

    #[test]
    fn resolve_target_handles_posix_origins() {
        assert_eq!(
            resolve_target("/notes/readme.md", "../img/a.png").unwrap(),
            "/img/a.png"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", "/assets/x.png").unwrap(),
            "/assets/x.png"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", "a.png").unwrap(),
            "/notes/a.png"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", "").unwrap(),
            "/notes/readme.md"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", "./sub/../a.png").unwrap(),
            "/notes/a.png"
        );
    }

    #[test]
    fn resolve_target_handles_http_origins() {
        assert_eq!(
            resolve_target("https://example.com/a/b.html", "c.png").unwrap(),
            "https://example.com/a/c.png"
        );
        assert_eq!(
            resolve_target("https://example.com/a/b.html", "../img/a.png").unwrap(),
            "https://example.com/img/a.png"
        );
        assert_eq!(
            resolve_target("https://example.com", "a.png").unwrap(),
            "https://example.com/a.png"
        );
        assert_eq!(
            resolve_target("https://example.com/a/b.html", "//other.test/x").unwrap(),
            "https://other.test/x"
        );
        assert_eq!(
            resolve_target("https://example.com/a/b.html", "https://Other.Test/x?y#z").unwrap(),
            "https://other.test/x"
        );
        assert_eq!(
            resolve_target("https://user:pw@example.com/a", "file:///tmp/a%20b.txt").unwrap(),
            "/tmp/a b.txt"
        );
    }

    #[test]
    fn resolve_target_handles_windows_and_escapes() {
        assert_eq!(
            resolve_target("/notes/readme.md", "C:/Temp/File.txt").unwrap(),
            r"c:\temp\file.txt"
        );
        // The escape stripping runs first, so a two-backslash target is no longer
        // a UNC path (Python behaves the same way).
        assert_eq!(
            resolve_target("/notes/readme.md", r"\\server\share\a.txt").unwrap(),
            "/notes/\\server\\share\\a.txt"
        );
        assert_eq!(
            resolve_target(r"C:\Notes\readme.md", "../img/a.png").unwrap(),
            r"c:\img\a.png"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", "a%26b&amp;c.png").unwrap(),
            "/notes/a&b&c.png"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", r"a\_b\*c.png").unwrap(),
            "/notes/a_b*c.png"
        );
        // A lone backslash is kept: a space is not ASCII punctuation.
        assert_eq!(
            resolve_target("/notes/readme.md", r"a\ b.png").unwrap(),
            "/notes/a\\ b.png"
        );
        assert_eq!(
            resolve_target("/notes/readme.md", "file:///tmp/a%20b.txt").unwrap(),
            "/tmp/a b.txt"
        );
    }

    #[test]
    fn resolve_target_reports_unsupported_schemes_and_bad_authorities() {
        let error = resolve_target("/notes/readme.md", "mailto:someone@example.com").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unsupported source link scheme"));
        assert!(error.to_string().contains("mailto"));

        let error = resolve_target("/notes/readme.md", "//[::1/x").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("not resolvable"));
    }

    #[test]
    fn locator_key_normalises_comparably() {
        assert_eq!(locator_key("/notes/./readme.md"), "/notes/readme.md");
        assert_eq!(locator_key("a/../../b"), "../b");
        assert_eq!(locator_key("a//b/"), "a/b");
        assert_eq!(locator_key("//a//b"), "//a/b");
        assert_eq!(locator_key("///a"), "/a");
        assert_eq!(locator_key(""), ".");
        assert_eq!(locator_key("C:/Temp/File.txt"), r"c:\temp\file.txt");
        assert_eq!(locator_key(r"C:\Temp\..\File.txt"), r"c:\file.txt");
        assert_eq!(
            locator_key(r"\\Server\Share\A.txt"),
            r"\\server\share\a.txt"
        );
        assert_eq!(
            locator_key("https://user:pw@Example.com/a?q=1#f"),
            "https://example.com/a"
        );
        assert_eq!(
            locator_key("HTTPS://Example.com:443/a"),
            "https://example.com:443/a"
        );
        assert_eq!(locator_key("file:///C:/Temp/x.TXT"), r"c:\temp\x.txt");
        assert_eq!(
            locator_key("file://server/share/a.txt"),
            "//server/share/a.txt"
        );
    }

    #[test]
    fn preview_keeps_labels_only() {
        let token = format!("[图 A]({LINK_TOKEN_PREFIX}0123456789abcdef0123456789abcdef)");
        let markdown = format!(
            "See {token} and [site](https://example.com) and [bad]({LINK_TOKEN_PREFIX}XYZ)."
        );
        assert_eq!(
            SourceLinkBook::preview(&markdown),
            "See 图 A and [site](https://example.com) and [bad](pcs-source-link:XYZ)."
        );
        assert_eq!(SourceLinkBook::preview("plain prose"), "plain prose");
        // A token split across lines is not a token.
        let broken = "[a\nb](pcs-source-link:0123456789abcdef0123456789abcdef)";
        assert_eq!(SourceLinkBook::preview(broken), broken);
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("source_links.json");
        let book = sample_book();

        book.save(&path).unwrap();
        let document: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(document["version"], serde_json::json!(LINK_BOOK_VERSION));
        assert_eq!(
            document["source_links"].as_object().unwrap().len(),
            book.len()
        );
        assert!(!path.with_extension("json.tmp").exists());

        let loaded = SourceLinkBook::load(&path).unwrap();
        assert_eq!(loaded, book);
        assert_eq!(loaded.identities(), book.identities());
        assert_eq!(
            loaded.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            book.iter().map(|(id, _)| id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn load_of_a_missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let book = SourceLinkBook::load(&dir.path().join("absent.json")).unwrap();
        assert!(book.is_empty());
    }

    #[test]
    fn load_tolerates_unknown_fields_and_a_missing_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source_links.json");
        let book = sample_book();
        let identity = book.identities().remove(0);
        let record = book.get(&identity).unwrap();
        let mut value = serde_json::to_value(record).unwrap();
        value["future_field"] = serde_json::json!("ignored");
        let document = serde_json::json!({"source_links": {identity.clone(): value}});
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();

        let loaded = SourceLinkBook::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get(&identity), Some(record));
    }

    #[test]
    fn load_rejects_malformed_documents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source_links.json");

        let documents: [&[u8]; 3] = [b"{", b"[]", br#"{"source_links": []}"#];
        for document in documents {
            std::fs::write(&path, document).unwrap();
            let error = SourceLinkBook::load(&path).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
        }

        std::fs::write(&path, br#"{"source_links": {"abc": "not a record"}}"#).unwrap();
        let error = SourceLinkBook::load(&path).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("source link record is invalid"));
    }

    #[test]
    fn load_rejects_a_tampered_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source_links.json");
        sample_book().save(&path).unwrap();

        let mut document: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let records = document["source_links"].as_object_mut().unwrap();
        let (identity, record) = records.iter_mut().next().unwrap();
        let mut tampered = record.clone();
        tampered["label"] = serde_json::json!("tampered");
        let identity = identity.clone();
        records.clear();
        records.insert(identity, tampered);
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();

        let error = SourceLinkBook::load(&path).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains(IDENTITY_ERROR));
    }

    #[test]
    fn merge_is_a_union_with_the_later_book_winning() {
        let mut book = sample_book();
        let mut other = SourceLinkBook::new();
        other.register("/notes/other.md", "c.png", "C").unwrap();
        // The same link registered twice keeps one entry.
        other
            .register("/notes/readme.md", "../img/a.png", "A")
            .unwrap();

        book.merge(&other);
        assert_eq!(book.len(), 3);
        assert!(book
            .iter()
            .any(|(_, link)| link.resolved_target == "/notes/c.png"));
    }
}
