//! Helpers shared by the router, the compressors and the pipeline.
//!
//! Ports `rule_compression/common.py`: token accounting (`count_tokens`,
//! `meets_savings_ratio`, `fits_budget_and_saves`), the tool-call path lookup
//! (`extract_file_path_argument`, `is_source_file_path`), the tool display
//! prefix strippers and the `ERROR_RE` word matcher.
//!
//! Python uses `re`; rule compression compiles no regular expression of its own —
//! the crate depends on `regex` only for its tokenizer nodes, never here — so
//! every matcher in this module is hand-rolled over bytes. Matching is ASCII-only
//! (Python's `\d`/`\s` also accept non-ASCII digits and spaces), which keeps the
//! compressors deterministic and dependency-free; the difference only shows up on
//! non-ASCII numerals.
use super::types::RuleContext;
use serde_json::Value;
/// Argument keys whose string value names the single file a tool call reads.
const FILE_PATH_ARGUMENT_KEYS: [&str; 5] = ["file_path", "filepath", "file", "filename", "path"];
/// File types whose content the model needs verbatim.
///
/// Program source, markup and templates, configuration and prose documents:
/// content sniffing misroutes these (JSX tags look like HTML, a line starting
/// with `Error:` looks like a log). Data files the compressors are designed for
/// (`.json`, `.jsonl`, `.log`, `.txt`, `.csv`, `.diff`, `.patch`) are
/// deliberately absent so they keep content-based routing.
const SOURCE_FILE_EXTENSIONS: &[&str] = &[
    // Program source.
    ".py",
    ".pyi",
    ".pyx",
    ".ipynb",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".groovy",
    ".gradle",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
    ".cs",
    ".fs",
    ".vb",
    ".swift",
    ".dart",
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".jl",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".clj",
    ".elm",
    ".zig",
    ".nim",
    ".v",
    ".sol",
    ".proto",
    ".thrift",
    ".graphql",
    ".gql",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".psm1",
    ".bat",
    ".cmd",
    // Markup, styles and templates.
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".xsd",
    ".xsl",
    ".svg",
    ".vue",
    ".svelte",
    ".astro",
    ".jinja",
    ".j2",
    ".hbs",
    ".ejs",
    ".erb",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".styl",
    // Configuration and build definitions.
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".env",
    ".tf",
    ".hcl",
    ".nix",
    ".cmake",
    ".mk",
    ".bzl",
    // Prose documents.
    ".md",
    ".mdx",
    ".markdown",
    ".rst",
    ".adoc",
    ".tex",
];
/// Extensionless file names that are source by convention.
const SOURCE_FILE_NAMES: &[&str] = &[
    "makefile",
    "gnumakefile",
    "dockerfile",
    "containerfile",
    "jenkinsfile",
    "vagrantfile",
    "gemfile",
    "rakefile",
    "podfile",
    "cmakelists.txt",
    "justfile",
];
/// Words that make plain text count as an error signal (`ERROR_RE`).
const ERROR_WORDS: &[&str] = &[
    "error",
    "failed",
    "failure",
    "traceback",
    "exception",
    "warn",
    "warning",
];
/// Measures `text` the way `ctx` prescribes, with a floor of one token.
#[must_use]
pub fn count_tokens(text: &str, ctx: &RuleContext) -> usize {
    ctx.token_count(text).max(1)
}
/// Whether `candidate` saves at least `ctx.min_savings_ratio` of `original`.
#[must_use]
pub fn meets_savings_ratio(original: &str, candidate: &str, ctx: &RuleContext) -> bool {
    let original_tokens = count_tokens(original, ctx);
    if original_tokens == 0 {
        return false;
    }
    let candidate_tokens = count_tokens(candidate, ctx);
    1.0 - candidate_tokens as f64 / original_tokens as f64 >= ctx.min_savings_ratio
}
/// Whether `candidate` both fits `ctx.max_tokens` and saves enough.
#[must_use]
pub fn fits_budget_and_saves(original: &str, candidate: &str, ctx: &RuleContext) -> bool {
    if count_tokens(candidate, ctx) > ctx.max_tokens {
        return false;
    }
    meets_savings_ratio(original, candidate, ctx)
}
/// Whether `text` carries an error signal word (`ERROR_RE`).
#[must_use]
pub fn is_error_text(text: &str) -> bool {
    contains_word(text, ERROR_WORDS)
}
/// Whether `text` contains any of `words` as a whole word, case-insensitively.
#[must_use]
pub fn contains_word(text: &str, words: &[&str]) -> bool {
    let lowered = text.to_lowercase();
    words
        .iter()
        .any(|word| find_word(&lowered, &word.to_lowercase()).is_some())
}
/// The byte offset of `needle` in `haystack` when it sits on word boundaries.
///
/// `haystack` must already be lower-cased; `needle` must be ASCII.
#[must_use]
pub fn find_word(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let mut from = 0;
    while let Some(offset) = haystack[from..].find(needle) {
        let start = from + offset;
        let end = start + needle.len();
        let before_ok = haystack[..start]
            .chars()
            .next_back()
            .is_none_or(|value| !is_word_char(value));
        let after_ok = haystack[end..]
            .chars()
            .next()
            .is_none_or(|value| !is_word_char(value));
        if before_ok && after_ok {
            return Some(start);
        }
        from = start + needle.len();
        if from >= haystack.len() {
            break;
        }
    }
    None
}
/// Python's `\w`: an alphanumeric character or an underscore.
fn is_word_char(value: char) -> bool {
    value.is_alphanumeric() || value == '_'
}
/// The file path a tool call names in its arguments.
///
/// `arguments` is either the raw JSON string or `None`; the first non-empty
/// string under a known file-path key wins. Anything else (no arguments, invalid
/// JSON, another shape) names no single file and yields `None`.
#[must_use]
pub fn extract_file_path_argument(arguments: Option<&str>) -> Option<String> {
    let raw = arguments?;
    let parsed: Value = serde_json::from_str(raw).ok()?;
    let object = parsed.as_object()?;
    for key in FILE_PATH_ARGUMENT_KEYS {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    None
}
/// The lower-cased base name of a POSIX or Windows path.
#[must_use]
pub fn file_name(file_path: &str) -> &str {
    file_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(file_path)
        .trim()
}
/// Whether `file_path` names a source, markup, config or document file.
///
/// Those files must keep their structure, so they route to
/// [`ContentType::SourceFile`](super::types::ContentType::SourceFile) before any
/// content sniffing.
#[must_use]
pub fn is_source_file_path(file_path: &str) -> bool {
    let name = file_name(file_path).to_lowercase();
    if name.is_empty() {
        return false;
    }
    if SOURCE_FILE_NAMES.contains(&name.as_str()) {
        return true;
    }
    let extension = match name.rfind('.') {
        // `.env`-style dotfiles have no extension; `.env` itself is listed.
        Some(0) | None => &name[..],
        Some(index) => &name[index..],
    };
    SOURCE_FILE_EXTENSIONS.contains(&extension)
}
/// Removes the line-display prefixes tools such as `read_file` or `grep` add.
#[must_use]
pub fn strip_display_line_prefixes(content: &str) -> String {
    strip_prefixes(content, false)
}
/// Removes display prefixes while preserving the body's leading whitespace.
///
/// The diff compressor uses this variant so indentation inside a hunk survives.
#[must_use]
pub fn strip_display_line_prefixes_preserving_body_whitespace(content: &str) -> String {
    strip_prefixes(content, true)
}
/// Applies a display-prefix strip when enough lines carry a prefix.
///
/// Python gates on `matches / non_empty_lines >= 0.3` so ordinary prose that
/// happens to start with a number is left alone; the same gate is applied here
/// with integer arithmetic.
fn strip_prefixes(content: &str, preserve_body_whitespace: bool) -> String {
    if content.is_empty() {
        return String::new();
    }
    let non_empty = content
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .count();
    if non_empty == 0 {
        return content.to_owned();
    }
    let mut matched = 0_usize;
    let mut stripped = String::with_capacity(content.len());
    for line in content.split_inclusive('\n') {
        let (body, terminator) = split_line_terminator(line);
        match match_display_prefix(body, preserve_body_whitespace) {
            Some(prefix_len) => {
                matched += 1;
                stripped.push_str(body.get(prefix_len..).unwrap_or(""));
                stripped.push_str(terminator);
            }
            None => stripped.push_str(line),
        }
    }
    if matched * 10 < non_empty * 3 {
        return content.to_owned();
    }
    stripped
}
/// Splits a line into its body and its `\n`/`\r\n` terminator.
fn split_line_terminator(line: &str) -> (&str, &str) {
    if let Some(body) = line.strip_suffix("\r\n") {
        return (body, "\r\n");
    }
    if let Some(body) = line.strip_suffix('\n') {
        return (body, "\n");
    }
    (line, "")
}
/// The byte length of the display prefix on `line`, when it has one.
fn match_display_prefix(line: &str, preserve_body_whitespace: bool) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if index >= bytes.len() {
        return None;
    }
    [
        numbered_prefix_end(line, index, preserve_body_whitespace),
        decorated_prefix_end(line, index),
        word_prefix_end(line, index),
    ]
    .into_iter()
    .flatten()
    .find(|candidate| body_is_present(line, *candidate, preserve_body_whitespace))
}
/// Whether a body follows the prefix at `start` (`\S.*` or `.*\S.*`).
fn body_is_present(line: &str, start: usize, preserve_body_whitespace: bool) -> bool {
    let body = line.get(start..).unwrap_or("");
    if preserve_body_whitespace {
        body.chars().any(|value| !value.is_whitespace())
    } else {
        body.chars()
            .next()
            .is_some_and(|value| !value.is_whitespace())
    }
}
/// `\d+[\t ]+` (or `\d+\t` / `\d+ +` when body whitespace is preserved).
fn numbered_prefix_end(line: &str, start: usize, preserve_body_whitespace: bool) -> Option<usize> {
    let digits_end = digits_end(line, start);
    if digits_end == start {
        return None;
    }
    let tail = line.as_bytes().get(digits_end).copied();
    if preserve_body_whitespace {
        return match tail {
            // A tab ends the prefix; the body's own indentation survives.
            Some(b'\t') => Some(digits_end + 1),
            // Space padding is part of the prefix, so it is consumed whole.
            Some(b' ') => Some(skip_blanks(line, digits_end)),
            _ => None,
        };
    }
    match tail {
        Some(b'\t' | b' ') => Some(skip_blanks(line, digits_end)),
        _ => None,
    }
}
/// `[|>:#-]\s*\d+\s*[|:.)\]-]?\s*`.
fn decorated_prefix_end(line: &str, start: usize) -> Option<usize> {
    let marker = *line.as_bytes().get(start)?;
    if !matches!(marker, b'|' | b'>' | b':' | b'#' | b'-') {
        return None;
    }
    let digits_start = skip_blanks(line, start + 1);
    let digits_end = digits_end(line, digits_start);
    if digits_end == digits_start {
        return None;
    }
    let after_digits = skip_blanks(line, digits_end);
    let after_punctuation = match line.as_bytes().get(after_digits) {
        Some(b'|' | b':' | b'.' | b')' | b']' | b'-') => after_digits + 1,
        _ => after_digits,
    };
    Some(skip_blanks(line, after_punctuation))
}
/// `(?:line|row)\s+\d+\s*[:|.)\]-]\s*`.
fn word_prefix_end(line: &str, start: usize) -> Option<usize> {
    let rest = line.get(start..)?;
    let rest_bytes = rest.as_bytes();
    let head_len = if starts_with_ignore_case(rest_bytes, b"line") {
        4
    } else if starts_with_ignore_case(rest_bytes, b"row") {
        3
    } else {
        return None;
    };
    let mut index = start + head_len;
    let blanks_start = index;
    while line
        .as_bytes()
        .get(index)
        .is_some_and(u8::is_ascii_whitespace)
    {
        index += 1;
    }
    if index == blanks_start {
        return None;
    }
    let digits_end = digits_end(line, index);
    if digits_end == index {
        return None;
    }
    let after_digits = skip_blanks(line, digits_end);
    match line.as_bytes().get(after_digits) {
        Some(b':' | b'|' | b'.' | b')' | b']' | b'-') => Some(skip_blanks(line, after_digits + 1)),
        _ => None,
    }
}
/// Whether the bytes at `start` spell `needle`, ignoring ASCII case.
fn starts_with_ignore_case(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.len() >= needle.len()
        && bytes[..needle.len()]
            .iter()
            .zip(needle)
            .all(|(left, right)| left.eq_ignore_ascii_case(right))
}
/// The end of the digit run starting at `start`.
fn digits_end(line: &str, start: usize) -> usize {
    let bytes = line.as_bytes();
    let mut index = start;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    index
}
/// The first index at or after `start` that is not a space or tab.
fn skip_blanks(line: &str, start: usize) -> usize {
    let bytes = line.as_bytes();
    let mut index = start;
    while index < bytes.len() && matches!(bytes[index], b' ' | b'\t') {
        index += 1;
    }
    index
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_counting_has_a_floor_of_one() {
        let ctx = RuleContext::new(10);
        assert_eq!(count_tokens("", &ctx), 1);
        assert_eq!(count_tokens(&"a".repeat(30), &ctx), 10);
        let mut fine = RuleContext::new(10);
        fine.count_tokens = super::super::types::TokenCountPolicy::CharactersPerToken(1);
        assert_eq!(count_tokens(&"a".repeat(30), &fine), 30);
    }
    #[test]
    fn savings_ratio_is_relative() {
        let ctx = RuleContext::new(100);
        assert!(meets_savings_ratio(&"a".repeat(100), &"a".repeat(50), &ctx));
        assert!(!meets_savings_ratio(
            &"a".repeat(100),
            &"a".repeat(95),
            &ctx
        ));
        let tight = RuleContext::new(10);
        assert!(fits_budget_and_saves(
            &"a".repeat(300),
            &"a".repeat(30),
            &tight
        ));
        assert!(!fits_budget_and_saves(
            &"a".repeat(300),
            &"a".repeat(60),
            &tight
        ));
    }
    #[test]
    fn file_path_arguments_use_the_first_known_key() {
        let args = r#"{"path": "src/lib.rs", "file_path": "ignored"}"#;
        assert_eq!(
            extract_file_path_argument(Some(args)).as_deref(),
            Some("ignored")
        );
        assert_eq!(
            extract_file_path_argument(Some(r#"{"file": " docs/a.md "}"#)).as_deref(),
            Some("docs/a.md")
        );
        assert_eq!(extract_file_path_argument(Some("{not json")), None);
        assert_eq!(extract_file_path_argument(None), None);
        assert_eq!(extract_file_path_argument(Some(r#"{"path": "  "}"#)), None);
    }
    #[test]
    fn source_file_detection_covers_names_and_dotfiles() {
        assert!(is_source_file_path("src/main.rs"));
        assert!(is_source_file_path("C:\\work\\README.md"));
        assert!(is_source_file_path("/tmp/Makefile"));
        assert!(is_source_file_path("/tmp/.env"));
        assert!(!is_source_file_path("logs/build.log"));
        assert!(!is_source_file_path("data/report.json"));
        assert!(!is_source_file_path(""));
    }
    #[test]
    fn display_prefixes_are_stripped_above_the_gate() {
        let numbered = "1\tfn main() {\n2\t    body();\n3\t}\n";
        // The non-preserving strip eats the padding, so the body's indentation goes.
        assert_eq!(
            strip_display_line_prefixes(numbered),
            "fn main() {\nbody();\n}\n"
        );
        let preserving = "    12\tfn main() {\n    13\t    body();\n";
        assert_eq!(
            strip_display_line_prefixes_preserving_body_whitespace(preserving),
            "fn main() {\n    body();\n"
        );
        // One prefixed line out of five stays untouched.
        let sparse = "1\tone\ntwo\nthree\nfour\nfive\n";
        assert_eq!(strip_display_line_prefixes(sparse), sparse);
    }
    #[test]
    fn error_text_matches_whole_words_only() {
        assert!(is_error_text("an ERROR happened"));
        assert!(is_error_text("a warning"));
        // `warnings` is not `warning` on a word boundary.
        assert!(!is_error_text("warnings: 3"));
        assert!(!is_error_text("terror"));
        assert!(!is_error_text("informal"));
    }
}
