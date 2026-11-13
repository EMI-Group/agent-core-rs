//! Pure parsers for pytest reports.
//!
//! Two formats, both produced by the pytest run in [`super::python`]:
//!
//! - `pytest-json-report`'s JSON document (`--json-report`), whose `summary`
//!   counters and `tests[].outcome` carry everything a gate needs;
//! - `JUnit` XML (`--junitxml`), which needs no plugin and is what a bare pytest
//!   installation can always write.
//!
//! No I/O and no XML crate: the `JUnit` scanner walks tags by hand, tolerating
//! attribute order, self-closing elements, comments, CDATA, and the five common
//! XML entities. Malformed input is an [`Error::validation`], never a panic and
//! never a silently empty report.
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::python::ReportFormat;

/// Maximum number of characters kept from a failure message.
const MESSAGE_LIMIT: usize = 500;

/// One failed or errored test.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TestFailure {
    /// Test name: the `nodeid` for JSON reports, `classname::name` for `JUnit`.
    pub name: String,
    /// First line of the failure text, truncated to 500 characters.
    pub message: String,
}

/// Counters and failures from one pytest run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PytestReport {
    /// Total number of tests.
    pub total: u32,
    /// Tests that passed.
    pub passed: u32,
    /// Tests that failed.
    pub failed: u32,
    /// Tests that errored (setup/teardown failures).
    pub errors: u32,
    /// Tests that were skipped.
    pub skipped: u32,
    /// One entry per failed or errored test, in report order.
    pub failures: Vec<TestFailure>,
}

/// Parses a `pytest-json-report` document.
///
/// Missing `summary` counters are derived from the `tests` list, and a missing
/// `passed` counter is `total − failed − errors − skipped` (saturating). Failure
/// names come from `nodeid`; messages come from `call`/`setup`/`teardown`
/// `longrepr` or `crash.message`.
///
/// # Errors
///
/// Returns [`Error::validation`] when the text is not JSON or its root is not an
/// object.
pub fn parse_json_report(text: &str) -> Result<PytestReport> {
    let root: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| Error::validation(format!("invalid pytest JSON report: {error}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| Error::validation("pytest JSON report root is not an object"))?;
    let summary = object.get("summary").and_then(serde_json::Value::as_object);
    let mut derived = PytestReport::default();
    if let Some(tests) = object.get("tests").and_then(serde_json::Value::as_array) {
        for test in tests {
            derived.total = derived.total.saturating_add(1);
            match test
                .get("outcome")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
            {
                "failed" => {
                    derived.failed = derived.failed.saturating_add(1);
                    derived.failures.push(json_failure(test));
                }
                "error" => {
                    derived.errors = derived.errors.saturating_add(1);
                    derived.failures.push(json_failure(test));
                }
                "skipped" | "xfailed" => {
                    derived.skipped = derived.skipped.saturating_add(1);
                }
                _ => {}
            }
        }
    }
    let total = counter(summary, &["total"]).unwrap_or(derived.total);
    let failed = counter(summary, &["failed"]).unwrap_or(derived.failed);
    let errors = counter(summary, &["error", "errors"]).unwrap_or(derived.errors);
    let skipped = counter(summary, &["skipped"]).unwrap_or(derived.skipped);
    let passed = counter(summary, &["passed"]).unwrap_or_else(|| {
        total
            .saturating_sub(failed)
            .saturating_sub(errors)
            .saturating_sub(skipped)
    });
    Ok(PytestReport {
        total,
        passed,
        failed,
        errors,
        skipped,
        failures: derived.failures,
    })
}

/// Reads one numeric counter, trying each key in order.
fn counter(
    summary: Option<&serde_json::Map<String, serde_json::Value>>,
    keys: &[&str],
) -> Option<u32> {
    let summary = summary?;
    keys.iter().find_map(|key| {
        summary
            .get(*key)
            .and_then(serde_json::Value::as_u64)
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
    })
}

/// Builds a [`TestFailure`] from one JSON test entry.
fn json_failure(test: &serde_json::Value) -> TestFailure {
    let name = test
        .get("nodeid")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<unknown>")
        .to_owned();
    let message = ["call", "setup", "teardown"]
        .iter()
        .find_map(|phase| {
            let phase = test.get(*phase)?;
            phase
                .get("longrepr")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    phase
                        .get("crash")
                        .and_then(|crash| crash.get("message"))
                        .and_then(serde_json::Value::as_str)
                })
        })
        .or_else(|| test.get("longrepr").and_then(serde_json::Value::as_str))
        .map_or_else(String::new, first_line);
    TestFailure { name, message }
}

/// Takes the first non-empty line of `text`, trimmed and length-capped.
fn first_line(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    truncate(line, MESSAGE_LIMIT)
}

/// Truncates `text` to at most `limit` characters.
fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    text.chars().take(limit).collect()
}

/// Parses a `JUnit` XML report.
///
/// `<testsuite>` attributes win where present; anything missing is derived from
/// the `<testcase>` elements. A self-closing `<testcase/>` counts as a pass, and
/// a nested `<failure>` or `<error>` produces one [`TestFailure`] named
/// `classname::name`.
///
/// # Errors
///
/// Returns [`Error::validation`] when the text has no elements or its root is
/// not `<testsuite>`/`<testsuites>`.
pub fn parse_junit_xml(text: &str) -> Result<PytestReport> {
    let tags = scan_tags(text);
    let root = tags
        .first()
        .ok_or_else(|| Error::validation("JUnit report contains no XML elements"))?;
    if root.closing {
        return Err(Error::validation("JUnit report starts with a closing tag"));
    }
    if root.name != "testsuite" && root.name != "testsuites" {
        return Err(Error::validation(format!(
            "JUnit report root is <{}>, expected <testsuite> or <testsuites>",
            root.name
        )));
    }

    let mut suite_tests = Counter::default();
    let mut suite_failures = Counter::default();
    let mut suite_errors = Counter::default();
    let mut suite_skipped = Counter::default();
    for tag in tags
        .iter()
        .filter(|tag| tag.name == "testsuite" && !tag.closing)
    {
        suite_tests.add(tag, "tests");
        suite_failures.add(tag, "failures");
        suite_errors.add(tag, "errors");
        suite_skipped.add(tag, "skipped");
    }

    let mut testcase_count = 0u32;
    let mut failure_count = 0u32;
    let mut error_count = 0u32;
    let mut skipped_count = 0u32;
    let mut failures = Vec::new();
    let mut current: Option<CurrentCase> = None;
    for tag in &tags {
        if tag.closing {
            if tag.name == "testcase" {
                current = None;
            }
            continue;
        }
        match tag.name.as_str() {
            "testcase" => {
                testcase_count = testcase_count.saturating_add(1);
                if !tag.self_closing {
                    current = Some(CurrentCase {
                        classname: tag.attribute("classname").unwrap_or_default(),
                        name: tag.attribute("name").unwrap_or_default(),
                        reported: false,
                    });
                }
            }
            "failure" => {
                failure_count = failure_count.saturating_add(1);
                record_failure(text, tag, &mut current, &mut failures);
            }
            "error" => {
                error_count = error_count.saturating_add(1);
                record_failure(text, tag, &mut current, &mut failures);
            }
            "skipped" => skipped_count = skipped_count.saturating_add(1),
            _ => {}
        }
    }

    let total = suite_tests.value.unwrap_or(testcase_count);
    let failed = suite_failures.value.unwrap_or(failure_count);
    let errors = suite_errors.value.unwrap_or(error_count);
    let skipped = suite_skipped.value.unwrap_or(skipped_count);
    let passed = total
        .saturating_sub(failed)
        .saturating_sub(errors)
        .saturating_sub(skipped);
    Ok(PytestReport {
        total,
        passed,
        failed,
        errors,
        skipped,
        failures,
    })
}

/// Parses a report in the format the run requested.
///
/// # Errors
///
/// Propagates the JSON and `JUnit` parse errors.
pub fn parse_report(text: &str, format: ReportFormat) -> Result<PytestReport> {
    match format {
        ReportFormat::Json => parse_json_report(text),
        ReportFormat::JUnit => parse_junit_xml(text),
    }
}

/// A sum over `<testsuite>` attributes, remembering whether any were present.
#[derive(Debug, Default)]
struct Counter {
    value: Option<u32>,
}

impl Counter {
    /// Adds the numeric attribute `name` of `tag`, when present and readable.
    fn add(&mut self, tag: &Tag, name: &str) {
        let Some(text) = tag.attribute(name) else {
            return;
        };
        let Ok(parsed) = text.trim().parse::<u32>() else {
            return;
        };
        let current = self.value.unwrap_or(0);
        self.value = Some(current.saturating_add(parsed));
    }
}

/// The testcase a failure or error tag belongs to.
#[derive(Debug)]
struct CurrentCase {
    classname: String,
    name: String,
    reported: bool,
}

impl CurrentCase {
    /// `classname::name`, or just `name` when no class is known.
    fn display_name(&self) -> String {
        if self.classname.is_empty() {
            self.name.clone()
        } else if self.name.is_empty() {
            self.classname.clone()
        } else {
            format!("{}::{}", self.classname, self.name)
        }
    }
}

/// Records a failure element against the current testcase, once.
fn record_failure(
    xml: &str,
    tag: &Tag,
    current: &mut Option<CurrentCase>,
    failures: &mut Vec<TestFailure>,
) {
    let Some(case) = current.as_mut() else {
        return;
    };
    if case.reported {
        return;
    }
    case.reported = true;
    let message = tag
        .attribute("message")
        .filter(|message| !message.trim().is_empty())
        .or_else(|| {
            if tag.self_closing {
                None
            } else {
                Some(unescape(inner_text(xml, tag).trim()))
            }
        })
        .map_or_else(String::new, |message| first_line(&message));
    failures.push(TestFailure {
        name: case.display_name(),
        message,
    });
}

/// One XML tag found by [`scan_tags`].
#[derive(Debug, Clone)]
struct Tag {
    name: String,
    attributes: Vec<(String, String)>,
    closing: bool,
    self_closing: bool,
    /// Byte offset just past the closing `>`.
    end: usize,
}

impl Tag {
    /// Returns the attribute value named `key`.
    fn attribute(&self, key: &str) -> Option<String> {
        self.attributes
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    }
}

/// Scans every element tag in `xml`, skipping comments, declarations, and CDATA.
fn scan_tags(xml: &str) -> Vec<Tag> {
    let mut tags = Vec::new();
    let mut offset = 0usize;
    while let Some(relative) = xml[offset..].find('<') {
        let tag_start = offset + relative;
        let after = &xml[tag_start + 1..];
        if after.starts_with("!--") {
            let Some(end) = after.find("-->") else { break };
            offset = tag_start + 1 + end + 3;
            continue;
        }
        if after.starts_with("![CDATA[") {
            let Some(end) = after.find("]]>") else { break };
            offset = tag_start + 1 + end + 3;
            continue;
        }
        if after.starts_with('?') || after.starts_with('!') {
            let Some(end) = after.find('>') else { break };
            offset = tag_start + 1 + end + 1;
            continue;
        }
        let Some(end) = tag_end(after) else { break };
        let next = tag_start + 1 + end + 1;
        if let Some(mut tag) = parse_tag(&after[..end]) {
            tag.end = next;
            tags.push(tag);
        }
        offset = next;
    }
    tags
}

/// Finds the `>` that closes a tag, ignoring quoted attribute values.
fn tag_end(text: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (index, ch) in text.char_indices() {
        match quote {
            Some(active) => {
                if ch == active {
                    quote = None;
                }
            }
            None => match ch {
                '"' | '\'' => quote = Some(ch),
                '>' => return Some(index),
                _ => {}
            },
        }
    }
    None
}

/// Parses one tag body (without `<` and `>`).
fn parse_tag(inner: &str) -> Option<Tag> {
    let inner = inner.trim();
    let (closing, body) = match inner.strip_prefix('/') {
        Some(body) => (true, body.trim()),
        None => (false, inner),
    };
    let (self_closing, body) = match body.strip_suffix('/') {
        Some(body) => (true, body.trim_end()),
        None => (false, body),
    };
    let name_end = body.find(char::is_whitespace).unwrap_or(body.len());
    let name = body[..name_end].to_owned();
    if name.is_empty() {
        return None;
    }
    Some(Tag {
        name,
        attributes: parse_attributes(&body[name_end..]),
        closing,
        self_closing,
        end: 0,
    })
}

/// Parses `key="value"` pairs in any order.
fn parse_attributes(text: &str) -> Vec<(String, String)> {
    let mut attributes = Vec::new();
    let mut rest = text;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let key_end = rest
            .find(|ch: char| ch.is_whitespace() || ch == '=')
            .unwrap_or(rest.len());
        let key = rest[..key_end].to_owned();
        let tail = rest[key_end..].trim_start();
        let Some(after_eq) = tail.strip_prefix('=') else {
            if key.is_empty() {
                break;
            }
            attributes.push((key, String::new()));
            rest = tail;
            continue;
        };
        let after_eq = after_eq.trim_start();
        let quote = after_eq
            .chars()
            .next()
            .filter(|ch| *ch == '"' || *ch == '\'');
        let Some(quote) = quote else {
            let end = after_eq.find(char::is_whitespace).unwrap_or(after_eq.len());
            attributes.push((key, unescape(&after_eq[..end])));
            rest = &after_eq[end..];
            continue;
        };
        let value_rest = &after_eq[quote.len_utf8()..];
        let Some(end) = value_rest.find(quote) else {
            break;
        };
        attributes.push((key, unescape(&value_rest[..end])));
        rest = &value_rest[end + quote.len_utf8()..];
    }
    attributes
}

/// Returns the text between `tag` and its closing `</name>`.
fn inner_text<'a>(xml: &'a str, tag: &Tag) -> &'a str {
    let tail = &xml[tag.end..];
    let close = format!("</{}>", tag.name);
    tail.find(&close).map_or("", |end| &tail[..end])
}

/// Unescapes the five common XML entities plus numeric character references.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find('&') {
        out.push_str(&rest[..index]);
        let tail = &rest[index..];
        let Some(end) = tail.find(';').filter(|end| *end <= 10) else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let entity = &tail[1..end];
        let decoded = match entity {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => entity
                .strip_prefix("#x")
                .map(|code| u32::from_str_radix(code, 16).ok())
                .or_else(|| {
                    entity
                        .strip_prefix('#')
                        .map(|code| code.parse::<u32>().ok())
                })
                .flatten()
                .and_then(char::from_u32),
        };
        if let Some(ch) = decoded {
            out.push(ch);
            rest = &tail[end + 1..];
        } else {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescape_handles_named_and_numeric_entities() {
        assert_eq!(
            unescape("a &lt; b &amp;&amp; c &gt; d &quot;q&quot; &apos;s&apos;"),
            "a < b && c > d \"q\" 's'"
        );
        assert_eq!(unescape("&#65;&#x42;"), "AB");
        assert_eq!(unescape("a & b"), "a & b");
        assert_eq!(unescape("&#xZZ;"), "&#xZZ;");
    }

    #[test]
    fn tag_scanner_skips_comments_and_cdata() {
        let tags = scan_tags("<a><!-- <b/> --><c data=\"x>y\">text</c></a>");
        let names = tags.iter().map(|tag| tag.name.as_str()).collect::<Vec<_>>();
        assert_eq!(names, vec!["a", "c", "c", "a"]);
        assert_eq!(tags[1].attribute("data").as_deref(), Some("x>y"));
        assert!(tags[2].closing);
    }

    #[test]
    fn attributes_parse_in_any_order() {
        let attributes = parse_attributes(r#" name='x' classname="y" tests=3"#);
        assert_eq!(
            attributes,
            vec![
                ("name".to_owned(), "x".to_owned()),
                ("classname".to_owned(), "y".to_owned()),
                ("tests".to_owned(), "3".to_owned()),
            ]
        );
    }
}
