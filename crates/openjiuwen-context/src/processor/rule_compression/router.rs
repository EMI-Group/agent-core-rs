//! Content-type detection and dispatch.
//!
//! Ports `rule_compression/router.py`: [`RuleContentRouter::detect`] types a tool
//! result, [`RuleContentRouter::compress`] routes it to the one matching
//! compressor. Detection runs in Python's precedence order:
//!
//! 1. `SOURCE_FILE` when the tool call named a source/markup/config/prose file —
//!    the content sniffers cannot tell JSX from HTML,
//! 2. `JSON_ARRAY` when serde parses an array, or the body starts with `[` (once
//!    a code fence is dropped) and the lenient repair ([`repair_json_array`])
//!    produces one,
//! 3. `HTML` on a doctype/root-tag prefix or two structural tags,
//! 4. `LOG` when any line matches the build/test/logger line patterns,
//! 5. `GIT_DIFF` on `diff --git ` or `@@ … @@` lines,
//! 6. `SEARCH_RESULTS` when at least 30 % of the non-empty lines look like
//!    `path:line:content` and are not ISO timestamps,
//! 7. `PLAIN_TEXT` otherwise.
//!
//! Display line-number prefixes (`  12\t`, `   12 | `) are stripped before
//! sniffing; a git diff keeps its body whitespace, and plain text and source
//! files are routed verbatim.
//!
//! Two passes outrank that ladder:
//!
//! - [`sniff_content`] classifies the payload itself ([`SniffKind`]): a UTF-16
//!   body is re-encoded to UTF-8 first, and a binary or replacement-heavy body is
//!   passed through untouched instead of compressed.
//! - [`is_timestamp_line`] folds full-width and Arabic-Indic decimal digits (and
//!   `NFKC` compatibility forms) before the ISO shape check, so a localized log
//!   line is still a timestamp and not a `path:line:` search hit.
//!
//! [`ContentType`] is the router's single routing label and has no binary
//! variant, so a pass-through reports `PLAIN_TEXT`; the [`SniffKind`] names the
//! reason in the result's `details` and in one `tracing::debug!` line.
use self::repair::strip_code_fence;
use super::common::{
    find_word, is_source_file_path, strip_display_line_prefixes,
    strip_display_line_prefixes_preserving_body_whitespace,
};
use super::compressors::diff::DiffCompressor;
use super::compressors::html::HtmlCompressor;
use super::compressors::json_array::JsonArrayCompressor;
use super::compressors::log::LogCompressor;
use super::compressors::plain_text::PlainTextCompressor;
use super::compressors::search_results::SearchResultsCompressor;
use super::compressors::source_file::SourceFileCompressor;
use super::compressors::RuleCompressor;
use super::types::{ContentType, RuleCompressionResult, RuleContext};
use serde_json::{Map, Value};
use unicode_normalization::UnicodeNormalization;

mod repair;
mod sniff;

pub use repair::{repair_json, repair_json_array, repair_json_array_checked, JsonRepairFailure};
pub use sniff::{
    sniff_bytes, sniff_content, SniffKind, SniffedContent, BINARY_SNIFF_BYTES,
    MAX_CONTROL_RATIO_PERCENT, REPLACEMENT_CHAR_MIN, REPLACEMENT_CHAR_RATIO_PERCENT,
};
/// Structural tag names the HTML sniffer counts (Python's `_HTML_STRUCTURAL_TAG_RE`).
const HTML_STRUCTURAL_TAGS: &[&str] = &[
    "html", "head", "body", "article", "main", "section", "div", "script", "style", "nav",
    "footer", "aside", "table", "thead", "tbody", "tr", "th", "td", "ul", "ol", "li", "p", "pre",
    "code", "strong", "span", "h1", "h2", "h3", "h4", "h5", "h6",
];
/// Lower-cased prefixes that alone identify an HTML document.
const HTML_ROOT_PREFIXES: &[&str] = &["<!doctype html", "<html", "<head", "<body"];
/// Fraction of lines that must look like search hits.
const SEARCH_RESULTS_MIN_RATIO: f64 = 0.3;
/// How much of the content the HTML sniffer inspects.
const HTML_SNIFF_CHARS: usize = 20_000;
/// Log levels matched at the start of a line.
const LOG_LEVELS: &[&str] = &[
    "error", "warn", "warning", "info", "debug", "trace", "fatal", "critical",
];
/// Test/build counters after a leading count (`^\d+\s+(failed|passed|…)`).
const LOG_COUNTER_WORDS: &[&str] = &[
    "failed", "passed", "skipped", "error", "errors", "warning", "warnings",
];
/// Detects a tool result's content type and dispatches to its compressor.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuleContentRouter;
impl RuleContentRouter {
    /// A router. It is stateless; the compressors are created per call.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
    /// Types `content` for the tool call described by `ctx`.
    ///
    /// The payload is sniffed first ([`sniff_content`]): a UTF-16 body is typed
    /// from its UTF-8 re-encoding, and a binary or replacement-heavy body is
    /// typed as `PLAIN_TEXT` because no compressor may touch it.
    #[must_use]
    pub fn detect(&self, content: &str, ctx: &RuleContext) -> ContentType {
        if ctx
            .file_path
            .as_deref()
            .is_some_and(|path| !path.is_empty() && is_source_file_path(path))
        {
            return ContentType::SourceFile;
        }
        let sniffed = sniff_content(content);
        if sniffed.is_passthrough() {
            return ContentType::PlainText;
        }
        Self::detect_text(sniffed.text())
    }

    /// Types an already-sniffed UTF-8 body.
    fn detect_text(content: &str) -> ContentType {
        let text = content.trim();
        if text.is_empty() {
            return ContentType::PlainText;
        }
        let display_text = strip_display_line_prefixes(text);
        let json_text = if display_text == text {
            text.to_owned()
        } else {
            display_text.clone()
        };
        if is_json_array(&json_text) {
            return ContentType::JsonArray;
        }
        let html_text = if display_text == text {
            text.to_owned()
        } else {
            display_text.clone()
        };
        if looks_like_html(&html_text) {
            return ContentType::Html;
        }
        if display_text
            .lines()
            .any(|line| is_log_line(line.trim_end()))
        {
            return ContentType::Log;
        }
        if display_text.lines().any(|line| {
            let line = line.trim_end();
            line.starts_with("diff --git ") || is_hunk_header(line)
        }) {
            return ContentType::GitDiff;
        }
        let lines: Vec<&str> = display_text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        if !lines.is_empty() {
            let matches = lines
                .iter()
                .filter(|line| is_search_line(line) && !is_timestamp_line(line))
                .count();
            if matches as f64 / lines.len() as f64 >= SEARCH_RESULTS_MIN_RATIO {
                return ContentType::SearchResults;
            }
        }
        ContentType::PlainText
    }
    /// Detects the content type and compresses accordingly.
    ///
    /// A binary or replacement-heavy payload is passed through untouched: the
    /// input is returned with `modified = false` and a `details` payload naming
    /// the [`SniffKind`], plus one `tracing::debug!` line. Every other body is
    /// compressed from its sniffed UTF-8 text, and a re-encoded `UTF-16` body
    /// carries its conversion in `details`.
    #[must_use]
    pub fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        let sniffed = sniff_content(content);
        if sniffed.is_passthrough() {
            tracing::debug!(
                kind = sniffed.kind.as_str(),
                bytes = content.len(),
                nul_bytes = sniffed.nul_bytes,
                control_ratio = sniffed.control_ratio,
                replacement_characters = sniffed.replacement_characters,
                "rule compression: payload passed through untouched"
            );
            return RuleCompressionResult::declined(
                content,
                ContentType::PlainText,
                sniffed.details(),
            );
        }
        let text = sniffed.text();
        let content_type = Self::detect_text(text);
        let routed = match content_type {
            ContentType::GitDiff => strip_display_line_prefixes_preserving_body_whitespace(text),
            ContentType::Html
            | ContentType::JsonArray
            | ContentType::SearchResults
            | ContentType::Log => strip_display_line_prefixes(text),
            ContentType::PlainText | ContentType::SourceFile => text.to_owned(),
        };
        let result = match content_type {
            ContentType::JsonArray => JsonArrayCompressor.compress(&routed, ctx),
            ContentType::GitDiff => DiffCompressor.compress(&routed, ctx),
            ContentType::Html => HtmlCompressor::default().compress(&routed, ctx),
            ContentType::SearchResults => SearchResultsCompressor.compress(&routed, ctx),
            ContentType::Log => LogCompressor.compress(&routed, ctx),
            ContentType::SourceFile => SourceFileCompressor.compress(&routed, ctx),
            ContentType::PlainText => PlainTextCompressor.compress(&routed, ctx),
        };
        with_sniff_details(result, &sniffed)
    }
}

/// Merges the sniff facts of a re-encoded body into a compressor's `details`.
///
/// A body that was not re-encoded (and is not a pass-through) keeps its
/// compressor's `details` byte for byte, so sniffing changes nothing for the
/// ordinary UTF-8 path.
fn with_sniff_details(
    result: RuleCompressionResult,
    sniffed: &SniffedContent,
) -> RuleCompressionResult {
    let Some(extra) = sniffed.details() else {
        return result;
    };
    let RuleCompressionResult {
        content,
        content_type,
        modified,
        lossy,
        details,
    } = result;
    let mut merged = match details {
        Some(Value::Object(map)) => map,
        Some(other) => {
            let mut map = Map::new();
            map.insert("compressor_details".to_owned(), other);
            map
        }
        None => Map::new(),
    };
    if let Value::Object(sniff) = extra {
        for (key, value) in sniff {
            merged.entry(key).or_insert(value);
        }
    }
    RuleCompressionResult {
        content,
        content_type,
        modified,
        lossy,
        details: Some(Value::Object(merged)),
    }
}
/// Python's alias for [`RuleContentRouter`].
pub type ContentRouter = RuleContentRouter;
/// Whether `text` parses (or leniently repairs) into a JSON array.
///
/// The gate keeps Python's precedence: a body must start with `[` once a code
/// fence is dropped, so prose that merely mentions an array stays plain text
/// while a fenced or truncated array is still recognised.
fn is_json_array(text: &str) -> bool {
    if serde_json::from_str::<Value>(text).is_ok_and(|value| value.is_array()) {
        return true;
    }
    if !strip_code_fence(text).starts_with('[') {
        return false;
    }
    repair_json_array(text).is_some_and(|value| value.is_array())
}
/// Whether `text` looks like an HTML document.
fn looks_like_html(text: &str) -> bool {
    let lowered: String = text
        .chars()
        .take(HTML_SNIFF_CHARS)
        .collect::<String>()
        .to_lowercase();
    if HTML_ROOT_PREFIXES
        .iter()
        .any(|prefix| lowered.starts_with(prefix))
    {
        return true;
    }
    let mut hits = 0_usize;
    let mut rest = lowered.as_str();
    while let Some(offset) = rest.find('<') {
        rest = &rest[offset + 1..];
        let candidate = rest.strip_prefix('/').unwrap_or(rest);
        let name_len = candidate
            .bytes()
            .position(|value| !value.is_ascii_alphanumeric())
            .unwrap_or(candidate.len());
        let name = &candidate[..name_len];
        if HTML_STRUCTURAL_TAGS.contains(&name) {
            hits += 1;
            if hits >= 2 {
                return true;
            }
        }
    }
    false
}
/// Whether `line` matches one of Python's `_LOG_RE` alternatives.
pub(crate) fn is_log_line(line: &str) -> bool {
    if line.is_empty() {
        return false;
    }
    let lowered = line.to_lowercase();
    let trimmed = lowered.trim_start();
    is_pytest_banner(trimmed)
        || (trimmed.starts_with("collected ") && followed_by_count(trimmed, "collected "))
        || trimmed.starts_with("planning next action for user request:")
        || trimmed.starts_with("tool call requested:")
        || trimmed.starts_with("tool call completed:")
        || trimmed.starts_with("tool result received:")
        || is_failed_test_line(trimmed)
        || is_count_then_counter(trimmed)
        || contains_counter_pair(&lowered)
        || trimmed.starts_with("npm err!")
        || trimmed.starts_with("npm warn")
        || trimmed.starts_with("npm info")
        || is_pass_fail_line(trimmed)
        || trimmed.starts_with("test suites:")
        || is_build_progress_line(trimmed)
        || is_compiler_diagnostic(trimmed)
        || is_bracket_level_line(trimmed)
        || is_timestamped_level_line(trimmed)
        || is_file_line_col_diagnostic(trimmed)
        || trimmed.starts_with("make:")
        || trimmed_starts_make_subshell(trimmed)
        || trimmed.starts_with("traceback (most recent call last)")
        || is_python_frame(trimmed)
        || is_javascript_frame(trimmed)
        || is_arrow_location(trimmed)
}
/// `^=+ (test session starts|failures|errors|short test summary info) =+$`
fn is_pytest_banner(line: &str) -> bool {
    let Some(rest) = line.strip_prefix('=') else {
        return false;
    };
    let rest = rest.trim_start_matches('=');
    for banner in [
        "test session starts",
        "failures",
        "errors",
        "short test summary info",
    ] {
        let Some(tail) = rest.strip_prefix(banner) else {
            continue;
        };
        let tail = tail.trim_start();
        if tail.is_empty() {
            return true;
        }
        if tail.chars().all(|value| value == '=') {
            return true;
        }
    }
    false
}
/// Whether `rest` (already past `prefix`) starts with a run of digits then a space.
fn followed_by_count(line: &str, prefix: &str) -> bool {
    let rest = &line[prefix.len()..];
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    digits > 0
        && rest[digits..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
}
/// `^(FAILED|ERROR) .+::.+(- …|$)` — a failing test id.
fn is_failed_test_line(line: &str) -> bool {
    let rest = line
        .strip_prefix("failed ")
        .or_else(|| line.strip_prefix("error "));
    rest.is_some_and(|rest| rest.contains("::"))
}
/// `^\d+\s+(failed|passed|skipped|errors?|warnings?)`
fn is_count_then_counter(line: &str) -> bool {
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let rest = line[digits..].trim_start();
    LOG_COUNTER_WORDS
        .iter()
        .any(|word| rest.strip_prefix(word).is_some_and(is_word_boundary))
}
/// `\b\d+\s+failed,\s+\d+\s+passed\b` anywhere in the line.
fn contains_counter_pair(line: &str) -> bool {
    let Some(failed) = find_word(line, "failed") else {
        return false;
    };
    let after = &line[failed..];
    after.contains(',') && find_word(after, "passed").is_some()
}
/// `^(PASS|FAIL) \S+`
fn is_pass_fail_line(line: &str) -> bool {
    let rest = line
        .strip_prefix("pass ")
        .or_else(|| line.strip_prefix("fail "));
    rest.is_some_and(starts_nonspace)
}
/// `^(Compiling|Finished|Running) \S+`
fn is_build_progress_line(line: &str) -> bool {
    ["compiling ", "finished ", "running "]
        .iter()
        .any(|prefix| line.strip_prefix(prefix).is_some_and(starts_nonspace))
}
/// `^(error|warning)(\[[A-Z]\d+\])?:`
fn is_compiler_diagnostic(line: &str) -> bool {
    for prefix in ["error", "warning"] {
        let Some(rest) = line.strip_prefix(prefix) else {
            continue;
        };
        if let Some(rest) = rest.strip_prefix('[') {
            let code_len = rest
                .bytes()
                .position(|value| value == b']')
                .unwrap_or(rest.len());
            let code = &rest[..code_len];
            let valid = code.len() >= 2
                && code.as_bytes()[0].is_ascii_alphabetic()
                && code[1..].bytes().all(|value| value.is_ascii_digit());
            if valid && rest[code_len..].trim_start().starts_with("]:") {
                return true;
            }
            continue;
        }
        if rest.starts_with(':') {
            return true;
        }
    }
    false
}
/// `^(\[[A-Z]+\]\s*)?(ERROR|WARN|WARNING|INFO|DEBUG|TRACE|FATAL|CRITICAL)\b`
fn is_bracket_level_line(line: &str) -> bool {
    let mut rest = line;
    if let Some(tail) = rest.strip_prefix('[') {
        let close = tail
            .bytes()
            .position(|value| value == b']')
            .unwrap_or(tail.len());
        let label = &tail[..close];
        if !label.is_empty() && label.bytes().all(|value| value.is_ascii_uppercase()) {
            rest = tail[close + 1..].trim_start();
        }
    }
    starts_with_level(rest)
}
/// Whether `line` starts with one of [`LOG_LEVELS`] as a whole word.
fn starts_with_level(line: &str) -> bool {
    LOG_LEVELS
        .iter()
        .any(|level| line.strip_prefix(level).is_some_and(is_word_boundary))
}
/// `^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}.*\b(ERROR|WARN|…)\b`
fn is_timestamped_level_line(line: &str) -> bool {
    if !is_timestamp_line(line) {
        return false;
    }
    LOG_LEVELS
        .iter()
        .any(|level| find_word(line, level).is_some())
}
/// `^.+?:\d+(:\d+)?:\s*(error|warning):`
fn is_file_line_col_diagnostic(line: &str) -> bool {
    let mut rest = line;
    while let Some(colon) = rest.find(':') {
        let digits = rest[colon + 1..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        if digits > 0 {
            let after = &rest[colon + 1 + digits..];
            let after = after.strip_prefix(':').map_or(after, |tail| {
                let more = tail.bytes().take_while(u8::is_ascii_digit).count();
                &tail[more..]
            });
            let after = after.trim_start();
            if after.starts_with("error:") || after.starts_with("warning:") {
                return true;
            }
        }
        rest = &rest[colon + 1..];
    }
    false
}
/// `^make(\[\d+\])?:`
fn trimmed_starts_make_subshell(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("make[") else {
        return false;
    };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    digits > 0 && rest[digits..].starts_with("]:")
}
/// `^\s*File ".+", line \d+`
fn is_python_frame(line: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix("file \"") else {
        return false;
    };
    rest.contains("\", line ")
}
/// `^\s*at .+\(.+:\d+:\d+\)`
fn is_javascript_frame(line: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix("at ") else {
        return false;
    };
    let Some(open) = rest.rfind('(') else {
        return false;
    };
    let tail = &rest[open..];
    tail.ends_with(')') && tail.matches(':').count() >= 2
}
/// `^-->\s+.+:\d+:\d+`
fn is_arrow_location(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("-->") else {
        return false;
    };
    let rest = rest.trim_start();
    rest.matches(':').count() >= 2
        && rest.rsplit(':').next().is_some_and(|tail| {
            !tail.is_empty() && tail.bytes().all(|value| value.is_ascii_digit())
        })
}
/// `^@@ … @@`
fn is_hunk_header(line: &str) -> bool {
    line.starts_with("@@ ") && line[3..].contains(" @@")
}
/// `^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b`, with any decimal digits.
///
/// The line is folded twice before the ASCII shape check: `NFKC` maps the
/// full-width forms (`２０２４`, `：`, `－`, the ideographic space) onto their ASCII
/// equivalents, and `digit_value` maps the Arabic-Indic blocks (`٠١٢…` and
/// `۰۱۲…`) onto `0`–`9`.
/// The 19-character shape is counted in characters, not bytes, so a localized
/// line whose digits take three bytes each is still recognised — and, because it
/// is a timestamp, is no longer miscounted as a `path:line:` search hit.
#[must_use]
pub fn is_timestamp_line(line: &str) -> bool {
    let folded: Vec<char> = line
        .nfkc()
        .map(|value| match digit_value(value) {
            Some(digit) => char::from_digit(digit, 10).unwrap_or(value),
            None => value,
        })
        .collect();
    if folded.len() < 19 {
        return false;
    }
    let digits_at = |positions: &[usize]| {
        positions
            .iter()
            .all(|position| folded.get(*position).is_some_and(char::is_ascii_digit))
    };
    if !digits_at(&[0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]) {
        return false;
    }
    folded[4] == '-'
        && folded[7] == '-'
        && matches!(folded[10], ' ' | 'T')
        && folded[13] == ':'
        && folded[16] == ':'
}
/// The decimal value of `value`, for the scripts `NFKC` leaves alone.
///
/// `char::to_digit` only knows ASCII, and `NFKC` folds the full-width forms but
/// not the Arabic-Indic digits, so those two blocks are mapped here.
fn digit_value(value: char) -> Option<u32> {
    if let Some(digit) = value.to_digit(10) {
        return Some(digit);
    }
    match value {
        '\u{0660}'..='\u{0669}' => Some(u32::from(value) - 0x0660),
        '\u{06f0}'..='\u{06f9}' => Some(u32::from(value) - 0x06f0),
        _ => None,
    }
}
/// `^.+?:\d+[:\-].+$` — a `path:line:` hit.
fn is_search_line(line: &str) -> bool {
    match super::compressors::search_results::parse_search_match(line, 0) {
        Some(matched) => !matched.content.trim().is_empty(),
        None => false,
    }
}
/// Whether `text` starts with a non-whitespace character.
fn starts_nonspace(text: &str) -> bool {
    text.chars()
        .next()
        .is_some_and(|value| !value.is_whitespace())
}
/// Whether `rest` begins with a non-word character (Python's `\b` after a word).
fn is_word_boundary(rest: &str) -> bool {
    rest.chars()
        .next()
        .is_none_or(|value| !(value.is_alphanumeric() || value == '_'))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detection_prefers_the_named_source_file() {
        let mut ctx = RuleContext::new(1_000);
        ctx.file_path = Some("src/app.tsx".to_owned());
        assert_eq!(
            RuleContentRouter.detect("<div class=\"x\"><p>hi</p></div>", &ctx),
            ContentType::SourceFile
        );
    }
    #[test]
    fn a_truncated_or_fenced_array_is_still_a_json_array() {
        let router = RuleContentRouter;
        let ctx = RuleContext::new(1_000);
        let truncated = "[{\"a\": 1}, {\"b\": 2";
        assert_eq!(router.detect(truncated, &ctx), ContentType::JsonArray);
        assert_eq!(
            router.detect("```json\n[1, 2, 3]\n```", &ctx),
            ContentType::JsonArray
        );
        // Prose that merely mentions an array keeps Python's plain-text routing.
        assert_eq!(
            router.detect("the rows are [1, 2, 3] as shown", &ctx),
            ContentType::PlainText
        );
    }

    #[test]
    fn localized_timestamps_are_recognised() {
        assert!(is_timestamp_line("2024-01-01T00:00:00 ERROR boom"));
        // Full-width digits, a full-width colon and an ideographic space.
        assert!(is_timestamp_line(
            "２０２４-０１-０１　００：００：００ ERROR boom"
        ));
        // Arabic-Indic digits, which `NFKC` leaves alone.
        assert!(is_timestamp_line("٢٠٢٤-٠١-٠١T٠٠:٠٠:٠٠ ERROR boom"));
        assert!(!is_timestamp_line("2024/01/01 00:00:00"));
        assert!(!is_timestamp_line("short 2024-01-01"));
    }

    #[test]
    fn binary_payloads_are_passed_through() {
        let router = RuleContentRouter;
        let ctx = RuleContext::new(10);
        let binary = "payload\u{0}with\u{0}nuls";
        assert_eq!(router.detect(binary, &ctx), ContentType::PlainText);
        let result = router.compress(binary, &ctx);
        assert!(!result.modified);
        assert_eq!(result.content, binary);
        assert_eq!(
            result
                .details
                .as_ref()
                .and_then(|value| value["content_kind"].as_str()),
            Some("binary")
        );
    }
    #[test]
    fn logs_diffs_and_search_hits_are_recognised() {
        let router = RuleContentRouter;
        let ctx = RuleContext::new(1_000);
        let log = "collected 12 items\nsrc/test_a.py ..F\nshort test summary info\nFAILED src/test_a.py::test_x - assert 1";
        assert_eq!(router.detect(log, &ctx), ContentType::Log);
        let diff = "diff --git a/a.rs b/a.rs\n@@ -1,3 +1,3 @@\n-a\n+b";
        assert_eq!(router.detect(diff, &ctx), ContentType::GitDiff);
        let search = "src/a.rs:12:fn one()\nsrc/b.rs:7:fn two()\nsrc/c.rs:2:fn three()";
        assert_eq!(router.detect(search, &ctx), ContentType::SearchResults);
        assert_eq!(
            router.detect("just prose about cats", &ctx),
            ContentType::PlainText
        );
    }
    #[test]
    fn numbered_read_file_output_is_stripped_before_sniffing() {
        let content = "     1\t{\"name\": \"value\"}\n     2\t";
        // The numbered `read_file` rendering hides the array behind prefixes, but
        // the router strips them before sniffing.
        assert!(strip_display_line_prefixes(content).starts_with('{'));
    }
}
