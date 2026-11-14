//! Log compression: keep errors, warnings, traces and summaries.
//!
//! Ports `rule_compression/compressors/log_compressor.py`. Each line is
//! classified into a [`LogLevel`] and flagged when it sits inside a Python, JS,
//! Java, Rust or Go stack trace. Selection keeps:
//!
//! - the first, last and highest-scoring error and failure lines,
//! - deduplicated warnings (address, number and path runs normalised),
//! - the first traces up to `log_max_stack_traces`,
//! - every summary line (test totals, `=== FAILURES`, build results),
//!
//! widened by `log_error_context_lines` neighbours and capped at
//! `log_max_total_lines`. The candidate starts with a
//! `[LOG compressed: format=..., kept=X/Y, omitted=N]` header and ends with a
//! `[LOG omitted: ...]` footer whenever anything was dropped.
use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::{contains_word, meets_savings_ratio};
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use serde_json::json;
use std::collections::BTreeSet;
/// The severity a log line was classified as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// `error`, `fatal`, `critical`, `exception`.
    Error,
    /// `fail`, `failed`, `failure`.
    Fail,
    /// `warn`, `warning`.
    Warn,
    /// `info`.
    Info,
    /// `debug`.
    Debug,
    /// `trace`.
    Trace,
    /// Nothing recognised.
    Unknown,
}
impl LogLevel {
    /// The level words, in classification order.
    fn words(self) -> &'static [&'static str] {
        match self {
            Self::Error => &["error", "fatal", "critical", "exception"],
            Self::Fail => &["fail", "failed", "failure"],
            Self::Warn => &["warn", "warning"],
            Self::Info => &["info"],
            Self::Debug => &["debug"],
            Self::Trace => &["trace"],
            Self::Unknown => &[],
        }
    }
    /// The score the line contributes to the selection.
    fn base_score(self) -> u32 {
        match self {
            Self::Error | Self::Fail => 90,
            Self::Warn => 45,
            Self::Info => 12,
            Self::Debug => 4,
            Self::Trace => 2,
            Self::Unknown => 15,
        }
    }
}
/// One classified log line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogLine {
    /// Zero-based position in the input.
    pub position: usize,
    /// The raw line.
    pub content: String,
    /// The classified severity.
    pub level: LogLevel,
    /// Whether the line is part of a stack trace.
    pub stack_trace: bool,
    /// Whether the line looks like a summary.
    pub summary: bool,
    /// Selection score, capped at 100.
    pub score: u32,
}
/// The format markers each known tool leaves behind.
const FORMAT_MARKERS: [(&str, &[&str]); 5] = [
    (
        "pytest",
        &[
            "=== FAILURES",
            "=== ERRORS",
            "test session",
            "collected ",
            " passed",
            " failed",
        ],
    ),
    ("npm", &["npm ERR!", "npm WARN", "npm info", "npm http"]),
    (
        "cargo",
        &[
            "Compiling ",
            "Finished ",
            "Running ",
            "warning: ",
            "error[E",
        ],
    ),
    ("jest", &["PASS ", "FAIL ", "Test Suites:"]),
    ("make", &["make[", "make:", "gcc ", "g++", "clang "]),
];
/// Compresses build and test logs.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogCompressor;
impl RuleCompressor for LogCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        let raw_lines: Vec<&str> = content.lines().collect();
        if raw_lines.len() < ctx.log_min_lines {
            return RuleCompressionResult::unchanged(content, ContentType::Log);
        }
        let (lines, traces_seen) = classify_lines(&raw_lines, ctx.log_stack_trace_max_lines);
        let (mut selected, warnings_deduplicated) = select(&lines, ctx);
        if selected.is_empty() {
            selected = head_tail(&lines, ctx.log_max_total_lines.min(20));
        }
        selected = add_context(&lines, selected, ctx.log_error_context_lines);
        selected = apply_cap(&lines, selected, ctx.log_max_total_lines);
        let omitted = lines.len().saturating_sub(selected.len());
        let counts = level_counts(&lines);
        let format_detected = detect_format(&raw_lines);
        let mut output: Vec<String> = vec![format!(
            "[LOG compressed: format={format_detected}, kept={}/{}, omitted={omitted}]",
            selected.len(),
            lines.len()
        )];
        for position in &selected {
            if let Some(line) = lines.get(*position) {
                output.push(line.content.clone());
            }
        }
        if omitted > 0 {
            let labels = non_zero_level_labels(&counts);
            let label_text = if labels.is_empty() {
                String::new()
            } else {
                format!(": {}", labels.join(", "))
            };
            output.push(format!(
                "[LOG omitted: {omitted} lines omitted{label_text}]"
            ));
        }
        let candidate = output.join("\n");
        let stack_traces_retained = count_trace_groups(&lines, &selected);
        let details = json!({
            "format_detected": format_detected,
            "total_lines": lines.len(),
            "selected_lines": selected.len(),
            "omitted_lines": omitted,
            "errors": counts.get(&LogLevel::Error).copied().unwrap_or(0),
            "fails": counts.get(&LogLevel::Fail).copied().unwrap_or(0),
            "warnings": counts.get(&LogLevel::Warn).copied().unwrap_or(0),
            "info": counts.get(&LogLevel::Info).copied().unwrap_or(0),
            "stack_traces_seen": traces_seen,
            "stack_traces_retained": stack_traces_retained,
            "warnings_deduplicated": warnings_deduplicated,
            "selected_line_numbers": selected.iter().map(|position| position + 1).collect::<Vec<_>>(),
        });
        if omitted > 0 && meets_savings_ratio(content, &candidate, ctx) {
            return RuleCompressionResult::applied(
                candidate,
                ContentType::Log,
                true,
                Some(details),
            );
        }
        RuleCompressionResult::declined(content, ContentType::Log, Some(details))
    }
}
/// Classifies every line and counts the traces that opened.
fn classify_lines(lines: &[&str], max_trace_lines: usize) -> (Vec<LogLine>, usize) {
    let mut classified: Vec<LogLine> = Vec::with_capacity(lines.len());
    let mut trace_flavor: Option<&'static str> = None;
    let mut trace_lines = 0_usize;
    let mut traces_seen = 0_usize;
    for (position, content) in lines.iter().enumerate() {
        let opened = trace_flavor_of(content);
        if opened.is_some() && trace_flavor.is_none() {
            trace_flavor = opened;
            trace_lines = 0;
            traces_seen += 1;
        } else if let Some(flavor) = trace_flavor {
            if terminates_trace(flavor, content) {
                trace_flavor = None;
                trace_lines = 0;
                if let Some(reopened) = trace_flavor_of(content) {
                    trace_flavor = Some(reopened);
                    traces_seen += 1;
                }
            }
        }
        let in_trace = trace_flavor.is_some();
        if in_trace {
            trace_lines += 1;
            if trace_lines >= max_trace_lines {
                trace_flavor = None;
            }
        }
        let level = classify_level(content);
        let summary = is_summary_line(content);
        let mut score = level.base_score();
        if in_trace {
            score += 35;
        }
        if summary {
            score += 35;
        }
        classified.push(LogLine {
            position,
            content: (*content).to_owned(),
            level,
            stack_trace: in_trace,
            summary,
            score: score.min(100),
        });
    }
    (classified, traces_seen)
}
/// The highest-priority classification of a line.
fn classify_level(line: &str) -> LogLevel {
    for level in [
        LogLevel::Error,
        LogLevel::Fail,
        LogLevel::Warn,
        LogLevel::Info,
        LogLevel::Debug,
        LogLevel::Trace,
    ] {
        if contains_word(line, level.words()) {
            return level;
        }
    }
    LogLevel::Unknown
}
/// Python's `_SUMMARY_RE`, anchored to the start of the line and case-sensitive.
fn is_summary_line(line: &str) -> bool {
    if line.starts_with("===") || line.starts_with("---") {
        return true;
    }
    if starts_with_count_word(line) {
        return true;
    }
    if starts_with_test_count(line) {
        return true;
    }
    for word in ["TOTAL", "Total", "Summary"] {
        if let Some(rest) = line.strip_prefix(word) {
            if rest
                .chars()
                .next()
                .is_none_or(|value| !(value.is_alphanumeric() || value == '_'))
            {
                return true;
            }
        }
    }
    starts_with_build_result(line)
}
/// `\d+\s+(?:passed|failed|skipped|errors?|warnings?)\b`.
fn starts_with_count_word(line: &str) -> bool {
    let digits_end = line
        .bytes()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(line.len());
    if digits_end == 0 {
        return false;
    }
    let rest = &line[digits_end..];
    let Some(body) = rest.strip_prefix(|value: char| value.is_whitespace()) else {
        return false;
    };
    let body = body.trim_start();
    [
        "passed", "failed", "skipped", "error", "errors", "warning", "warnings",
    ]
    .iter()
    .any(|word| body.starts_with(word))
}
/// `(?:Tests?|Suites?):?\s+\d+`.
fn starts_with_test_count(line: &str) -> bool {
    for word in ["Test", "Tests", "Suite", "Suites"] {
        if let Some(rest) = line.strip_prefix(word) {
            let rest = rest.strip_prefix(':').unwrap_or(rest);
            let rest = rest.trim_start();
            return rest
                .chars()
                .next()
                .is_some_and(|value| value.is_ascii_digit());
        }
    }
    false
}
/// `(?:Build|Compile|Test).*(?:succeeded|failed|complete)`.
fn starts_with_build_result(line: &str) -> bool {
    ["Build", "Compile", "Test"].iter().any(|word| {
        line.strip_prefix(word).is_some_and(|rest| {
            ["succeeded", "failed", "complete"]
                .iter()
                .any(|result| rest.contains(result))
        })
    })
}
/// The trace flavour a line opens, if any.
fn trace_flavor_of(line: &str) -> Option<&'static str> {
    let stripped = line.trim_start();
    if stripped.starts_with("Traceback (most recent call last)") || stripped.starts_with("File \"")
    {
        return Some("python");
    }
    if let Some(rest) = stripped.strip_prefix("at ") {
        if has_call_site(stripped) {
            return Some("js");
        }
        let identifier_end = rest
            .find('(')
            .filter(|index| *index > 0 && rest[..*index].chars().all(is_java_identifier));
        if identifier_end.is_some() {
            return Some("java");
        }
    }
    if stripped.starts_with("--> ") && has_line_col(stripped) {
        return Some("rust");
    }
    if starts_with_go_address(stripped) {
        return Some("go");
    }
    None
}
/// Whether `line` carries a `(…:\d+:\d+)` call site (JS frames).
fn has_call_site(line: &str) -> bool {
    line.find('(').is_some_and(|start| {
        let rest = &line[start + 1..];
        let Some(close) = rest.rfind(')') else {
            return false;
        };
        let inner = &rest[..close];
        let Some(last_colon) = inner.rfind(':') else {
            return false;
        };
        let Some(before_col) = inner[..last_colon].rfind(':') else {
            return false;
        };
        let line_part = &inner[before_col + 1..last_colon];
        let column = &inner[last_colon + 1..];
        !line_part.is_empty()
            && line_part.bytes().all(|byte| byte.is_ascii_digit())
            && !column.is_empty()
            && column.bytes().all(|byte| byte.is_ascii_digit())
    })
}
/// Whether `value` is a Java identifier character (`\w`, `.`, `$`).
fn is_java_identifier(value: char) -> bool {
    value.is_alphanumeric() || matches!(value, '_' | '.' | '$')
}
/// Whether `line` contains a `:\d+:\d+` sequence (Rust diagnostics).
fn has_line_col(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b':' {
            let line_start = index + 1;
            let line_end = line_start
                + bytes[line_start..]
                    .iter()
                    .position(|byte| !byte.is_ascii_digit())
                    .unwrap_or(bytes.len() - line_start);
            if line_end > line_start && bytes.get(line_end) == Some(&b':') {
                let column_start = line_end + 1;
                let column_end = column_start
                    + bytes[column_start..]
                        .iter()
                        .position(|byte| !byte.is_ascii_digit())
                        .unwrap_or(bytes.len() - column_start);
                if column_end > column_start {
                    return true;
                }
            }
            index = line_end;
            continue;
        }
        index += 1;
    }
    false
}
/// Whether `line` starts with a Go `\d+: 0x…` frame.
fn starts_with_go_address(line: &str) -> bool {
    let digits = line
        .bytes()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(line.len());
    if digits == 0 {
        return false;
    }
    let Some(rest) = line[digits..].strip_prefix(':') else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rest) = rest
        .get(..2)
        .filter(|prefix| prefix.eq_ignore_ascii_case("0x"))
        .map(|_| &rest[2..])
    else {
        return false;
    };
    rest.bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_hexdigit())
}
/// Whether `line` ends the trace `flavor` left open.
fn terminates_trace(flavor: &str, line: &str) -> bool {
    let stripped = line.trim_start();
    match flavor {
        "python" => {
            if line.is_empty() || line.starts_with(char::is_whitespace) {
                return false;
            }
            if stripped.starts_with("Traceback")
                || stripped.starts_with("During handling")
                || stripped.starts_with("The above exception")
            {
                return false;
            }
            !starts_with_exception_footer(stripped)
        }
        "js" | "java" => !line.is_empty() && !stripped.starts_with("at "),
        "rust" => !line.is_empty() && !stripped.starts_with("--> "),
        _ => !line.is_empty() && !starts_with_go_address(stripped),
    }
}
/// `[A-Za-z_][\w.]*?(?:Error|Exception)?:` at the start of `line`.
fn starts_with_exception_footer(line: &str) -> bool {
    match line.chars().next() {
        Some(value) if value.is_ascii_alphabetic() || value == '_' => {}
        _ => return false,
    }
    for value in line.chars().skip(1) {
        if value == ':' {
            return true;
        }
        if !(value.is_alphanumeric() || matches!(value, '_' | '.')) {
            return false;
        }
    }
    false
}
/// Selects the lines to keep and the number of deduplicated warnings.
fn select(lines: &[LogLine], ctx: &RuleContext) -> (Vec<usize>, usize) {
    let errors = members(lines, LogLevel::Error);
    let fails = members(lines, LogLevel::Fail);
    let warnings = members(lines, LogLevel::Warn);
    let (deduped_warnings, deduplicated) = dedupe_warnings(lines, &warnings);
    let traces = group_traces(lines, &all_positions(lines));
    let summaries: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.summary)
        .map(|(index, _)| index)
        .collect();

    let mut selected: Vec<usize> = Vec::new();
    selected.extend(first_last_top(lines, &errors, ctx.log_max_errors));
    selected.extend(first_last_top(lines, &fails, ctx.log_max_errors));
    selected.extend(deduped_warnings.iter().take(ctx.log_max_warnings).copied());
    for trace in traces.iter().take(ctx.log_max_stack_traces) {
        selected.extend(trace.iter().take(ctx.log_stack_trace_max_lines).copied());
    }
    selected.extend(summaries);
    let unique: BTreeSet<usize> = selected.into_iter().collect();
    (unique.into_iter().collect(), deduplicated)
}
/// Indices of the lines classified as `level`.
fn members(lines: &[LogLine], level: LogLevel) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.level == level)
        .map(|(index, _)| index)
        .collect()
}
/// Every line index, in order.
fn all_positions(lines: &[LogLine]) -> Vec<usize> {
    (0..lines.len()).collect()
}
/// First, last and highest-scoring members, limited to `limit`.
fn first_last_top(lines: &[LogLine], members: &[usize], limit: usize) -> Vec<usize> {
    if members.len() <= limit {
        return members.to_vec();
    }
    let mut selected: BTreeSet<usize> = BTreeSet::new();
    if let Some(first) = members.first() {
        selected.insert(*first);
    }
    if let Some(last) = members.last() {
        selected.insert(*last);
    }
    let mut ranked = members.to_vec();
    ranked.sort_unstable_by(|left, right| {
        score_of(lines, *right)
            .cmp(&score_of(lines, *left))
            .then(left.cmp(right))
    });
    for position in ranked {
        if selected.len() >= limit {
            break;
        }
        selected.insert(position);
    }
    selected.into_iter().collect()
}
/// The score of the line at `position`.
fn score_of(lines: &[LogLine], position: usize) -> u32 {
    lines.get(position).map_or(0, |line| line.score)
}
/// Deduplicates warnings on address/number/path-normalised text.
fn dedupe_warnings(lines: &[LogLine], warnings: &[usize]) -> (Vec<usize>, usize) {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<usize> = Vec::new();
    for position in warnings {
        let Some(line) = lines.get(*position) else {
            continue;
        };
        if seen.insert(warning_key(&line.content)) {
            kept.push(*position);
        }
    }
    let deduplicated = warnings.len().saturating_sub(kept.len());
    (kept, deduplicated)
}
/// The de-duplication key of a warning line.
fn warning_key(content: &str) -> String {
    let (prefix, separator, suffix) = match content.split_once(':') {
        Some((prefix, suffix)) => (prefix, ":", suffix),
        None => match content.split_once('=') {
            Some((prefix, suffix)) => (prefix, "=", suffix),
            None => (content, "", ""),
        },
    };
    let normalized = replace_paths(&replace_digits(&replace_hex_addresses(suffix)));
    format!("{prefix}{separator}{normalized}")
}
/// Replaces `0x…` runs with `ADDR`.
fn replace_hex_addresses(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < bytes.len() {
        let is_hex_prefix = index + 1 < bytes.len()
            && bytes[index].eq_ignore_ascii_case(&b'0')
            && bytes[index + 1].eq_ignore_ascii_case(&b'x');
        if is_hex_prefix {
            let mut end = index + 2;
            while end < bytes.len() && bytes[end].is_ascii_hexdigit() {
                end += 1;
            }
            if end > index + 2 {
                out.push_str("ADDR");
                index = end;
                continue;
            }
        }
        let Some(value) = text[index..].chars().next() else {
            break;
        };
        out.push(value);
        index += value.len_utf8();
    }
    out
}
/// Replaces digit runs with `N`.
fn replace_digits(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_digits = false;
    for value in text.chars() {
        if value.is_ascii_digit() {
            if !in_digits {
                out.push('N');
                in_digits = true;
            }
        } else {
            in_digits = false;
            out.push(value);
        }
    }
    out
}
/// Replaces `[/\\][\w./\\-]+` runs with `/PATH/`.
fn replace_paths(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((_, value)) = chars.next() {
        if !matches!(value, '/' | '\\') {
            out.push(value);
            continue;
        }
        let mut body = String::new();
        while let Some((_, next)) = chars.peek() {
            if next.is_alphanumeric() || matches!(next, '_' | '.' | '/' | '\\' | '-') {
                body.push(*next);
                chars.next();
            } else {
                break;
            }
        }
        if body.is_empty() {
            out.push(value);
        } else {
            out.push_str("/PATH/");
        }
    }
    out
}
/// Groups consecutive stack-trace lines from `positions` in `lines`.
fn group_traces(lines: &[LogLine], positions: &[usize]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for position in positions {
        let is_trace = lines.get(*position).is_some_and(|line| line.stack_trace);
        if is_trace {
            current.push(*position);
        } else if !current.is_empty() {
            groups.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}
/// How many trace groups survive the selection.
fn count_trace_groups(lines: &[LogLine], selected: &[usize]) -> usize {
    group_traces(lines, selected).len()
}
/// Widens `selected` by `radius` neighbours on both sides.
fn add_context(lines: &[LogLine], selected: Vec<usize>, radius: usize) -> Vec<usize> {
    if lines.is_empty() {
        return Vec::new();
    }
    let mut positions: BTreeSet<usize> = selected.into_iter().collect();
    let seeds: Vec<usize> = positions.iter().copied().collect();
    for position in seeds {
        let start = position.saturating_sub(radius);
        let end = (position + radius).min(lines.len().saturating_sub(1));
        for neighbour in start..=end {
            positions.insert(neighbour);
        }
    }
    positions.into_iter().collect()
}
/// Caps the selection at `limit` lines by score.
fn apply_cap(lines: &[LogLine], selected: Vec<usize>, limit: usize) -> Vec<usize> {
    if selected.len() <= limit {
        return selected;
    }
    let mut ranked = selected;
    ranked.sort_unstable_by(|left, right| {
        score_of(lines, *right)
            .cmp(&score_of(lines, *left))
            .then(left.cmp(right))
    });
    ranked.truncate(limit);
    ranked.sort_unstable();
    ranked
}
/// The head and tail of the history when nothing scored.
fn head_tail(lines: &[LogLine], limit: usize) -> Vec<usize> {
    if limit == 0 {
        return Vec::new();
    }
    if lines.len() <= limit {
        return all_positions(lines);
    }
    let head_count = (limit / 2).max(1);
    let tail_count = limit - head_count;
    let mut selected: BTreeSet<usize> = (0..head_count).collect();
    selected.extend(lines.len().saturating_sub(tail_count)..lines.len());
    selected.into_iter().collect()
}
/// Per-level line counts.
fn level_counts(lines: &[LogLine]) -> std::collections::BTreeMap<LogLevel, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for line in lines {
        *counts.entry(line.level).or_insert(0) += 1;
    }
    counts
}
/// The `N ERROR, N FAIL, ...` labels of the omitted footer.
fn non_zero_level_labels(counts: &std::collections::BTreeMap<LogLevel, usize>) -> Vec<String> {
    let mut labels = Vec::new();
    for (level, label) in [
        (LogLevel::Error, "ERROR"),
        (LogLevel::Fail, "FAIL"),
        (LogLevel::Warn, "WARN"),
        (LogLevel::Info, "INFO"),
    ] {
        if let Some(count) = counts.get(&level).filter(|count| **count > 0) {
            labels.push(format!("{count} {label}"));
        }
    }
    labels
}
/// The detected log format.
fn detect_format(lines: &[&str]) -> String {
    let sample: Vec<&str> = lines.iter().take(100).copied().collect();
    let mut best = ("generic", 0_usize);
    for (name, markers) in FORMAT_MARKERS {
        let score = sample
            .iter()
            .filter(|line| markers.iter().any(|marker| line.contains(marker)))
            .count();
        if score > best.1 {
            best = (name, score);
        }
    }
    best.0.to_owned()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn pytest_log() -> String {
        let mut lines = vec!["= test session starts =".to_owned()];
        for index in 0..60 {
            lines.push(format!("tests/test_mod.py::{index} PASSED"));
        }
        lines.push("FAILED tests/test_mod.py::test_one - AssertionError: boom".to_owned());
        lines.push("E   AssertionError: expected 1 got 2".to_owned());
        lines.push("= 1 failed, 60 passed in 0.42s =".to_owned());
        lines.join("\n")
    }
    #[test]
    fn errors_and_summaries_survive_a_pytest_log() {
        let content = pytest_log();
        let ctx = RuleContext::new(10_000);
        let result = LogCompressor.compress(&content, &ctx);
        assert!(result.modified);
        assert!(result.content.starts_with("[LOG compressed: format=pytest"));
        assert!(result
            .content
            .contains("FAILED tests/test_mod.py::test_one"));
        assert!(result.content.contains("1 failed, 60 passed"));
        assert!(result.content.contains("[LOG omitted:"));
        let details = result.details.expect("details");
        assert_eq!(details["format_detected"], "pytest");
        assert_eq!(details["total_lines"], 64);
    }
    #[test]
    fn short_logs_are_declined() {
        let content = "line one\nline two\n";
        let result = LogCompressor.compress(content, &RuleContext::new(100));
        assert!(!result.modified);
        assert!(result.details.is_none());
    }
    #[test]
    fn python_tracebacks_are_flagged() {
        let (lines, seen) = classify_lines(
            &[
                "Traceback (most recent call last):",
                "  File \"a.py\", line 3, in <module>",
                "ValueError: boom",
                "done",
            ],
            20,
        );
        assert_eq!(seen, 1);
        assert!(lines[0].stack_trace);
        assert!(lines[1].stack_trace);
        assert!(lines[2].stack_trace);
        assert!(!lines[3].stack_trace);
    }
}
