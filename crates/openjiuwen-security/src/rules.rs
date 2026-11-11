//! Shared regex rule engine and redaction primitives.
//!
//! Every built-in detector compiles its patterns into [`CompiledRule`]s, finds
//! non-overlapping [`TextMatch`]es, and redacts byte ranges with
//! [`redact_ranges`]. A malformed user-supplied pattern is a construction error,
//! never a panic.

use std::cmp::Reverse;

use regex::RegexBuilder;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use crate::RiskLevel;

/// One compiled detection pattern.
#[derive(Debug, Clone)]
pub struct CompiledRule {
    /// Stable rule name, surfaced in `GuardrailResult::details`.
    pub name: String,
    /// The pattern source, stored verbatim for reporting.
    pub pattern: String,
    /// Severity this rule reports when it matches, if the detector sets one.
    pub risk_level: Option<RiskLevel>,
    /// The compiled regular expression; private so callers go through
    /// [`CompiledRule::find`] and [`CompiledRule::is_match`].
    regex: regex::Regex,
    /// Whether matching is case-sensitive (mirror of the builder flag).
    case_sensitive: bool,
}

impl CompiledRule {
    /// Compiles `pattern` into a rule.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// pattern is not a valid regular expression; the message starts with
    /// `invalid pattern '<name>'`.
    pub fn compile(name: impl Into<String>, pattern: &str, case_sensitive: bool) -> Result<Self> {
        let name = name.into();
        let regex = RegexBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|error| Error::validation(format!("invalid pattern '{name}': {error}")))?;
        Ok(Self {
            name,
            pattern: pattern.to_owned(),
            risk_level: None,
            regex,
            case_sensitive,
        })
    }

    /// Attaches the severity this rule reports.
    #[must_use]
    pub fn with_risk_level(mut self, level: RiskLevel) -> Self {
        self.risk_level = Some(level);
        self
    }

    /// The byte range of the first match in `text`.
    #[must_use]
    pub fn find(&self, text: &str) -> Option<(usize, usize)> {
        self.regex
            .find(text)
            .map(|found| (found.start(), found.end()))
    }

    /// Whether the rule matches anywhere in `text`.
    #[must_use]
    pub fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }

    /// Whether matching is case-sensitive.
    #[must_use]
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }
}

/// One non-overlapping match of a rule in a text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextMatch {
    /// Name of the rule that matched.
    pub rule: String,
    /// Start byte offset (inclusive).
    pub start: usize,
    /// End byte offset (exclusive).
    pub end: usize,
}

/// Finds the first non-overlapping match of every rule in `text`.
///
/// Rules are evaluated in order and, within a rule, in match order. A match is
/// skipped when its byte range overlaps a range already claimed by an earlier
/// rule or by an earlier match of the same rule, so the result never contains
/// overlapping ranges.
#[must_use]
pub fn find_all(rules: &[CompiledRule], text: &str) -> Vec<TextMatch> {
    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut matches = Vec::new();
    for rule in rules {
        for found in rule.regex.find_iter(text) {
            let range = (found.start(), found.end());
            if overlaps_any(&claimed, range) {
                continue;
            }
            claimed.push(range);
            matches.push(TextMatch {
                rule: rule.name.clone(),
                start: range.0,
                end: range.1,
            });
        }
    }
    matches
}

/// Replaces every non-overlapping range in `text` with `marker`.
///
/// Ranges are applied from the end of the string backwards, so byte offsets stay
/// valid while replacing. Invalid ranges are skipped instead of panicking: an
/// empty or inverted range, a range beyond the end of `text`, a range whose
/// bounds are not UTF-8 character boundaries, and a range overlapping one that
/// was already replaced.
#[must_use]
pub fn redact_ranges(text: &str, ranges: &[(usize, usize)], marker: &str) -> String {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable_by_key(|&(start, _)| Reverse(start));
    let mut result = text.to_owned();
    let mut replaced: Vec<(usize, usize)> = Vec::new();
    for (start, end) in sorted {
        if start >= end
            || end > text.len()
            || !text.is_char_boundary(start)
            || !text.is_char_boundary(end)
        {
            continue;
        }
        if overlaps_any(&replaced, (start, end)) {
            continue;
        }
        result.replace_range(start..end, marker);
        replaced.push((start, end));
    }
    result
}

/// Whether `range` shares at least one byte with a claimed range.
fn overlaps_any(claimed: &[(usize, usize)], range: (usize, usize)) -> bool {
    claimed
        .iter()
        .any(|&(start, end)| range.0 < end && start < range.1)
}

/// Validates a number with the Luhn checksum, ignoring non-digit characters.
///
/// Empty, single-digit, and non-numeric input is invalid, as is a digit string
/// whose Luhn sum is not a multiple of ten.
#[must_use]
pub fn luhn_valid(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut count = 0u32;
    let mut double = false;
    for byte in digits.bytes().rev() {
        if !byte.is_ascii_digit() {
            continue;
        }
        let mut digit = u32::from(byte - b'0');
        if double {
            digit *= 2;
            if digit > 9 {
                digit -= 9;
            }
        }
        sum += digit;
        count += 1;
        double = !double;
    }
    count >= 2 && sum % 10 == 0
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn rule(pattern: &str) -> CompiledRule {
        CompiledRule::compile("test", pattern, false).unwrap()
    }

    #[test]
    fn case_insensitive_by_default() {
        let insensitive = CompiledRule::compile("ignore", r"ignore previous", false).unwrap();
        assert!(insensitive.is_match("Please IGNORE PREVIOUS instructions"));
        assert_eq!(insensitive.find("xx ignore previous yy"), Some((3, 18)));
        assert!(!insensitive.is_case_sensitive());

        let sensitive = CompiledRule::compile("ignore", r"ignore previous", true).unwrap();
        assert!(sensitive.is_match("ignore previous"));
        assert!(!sensitive.is_match("IGNORE PREVIOUS"));
        assert!(sensitive.is_case_sensitive());
    }

    #[test]
    fn invalid_regex_is_a_validation_error() {
        let error = CompiledRule::compile("broken", "([a-z", false).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid pattern 'broken'"),
            "{error}"
        );
    }

    #[test]
    fn with_risk_level_sets_severity() {
        assert_eq!(rule("x").risk_level, None);
        assert_eq!(
            rule("x").with_risk_level(RiskLevel::High).risk_level,
            Some(RiskLevel::High)
        );
    }

    #[test]
    fn find_all_skips_overlapping_ranges_across_rules() {
        let rules = vec![
            CompiledRule::compile("first", r"abc", false).unwrap(),
            CompiledRule::compile("second", r"cde", false).unwrap(),
            CompiledRule::compile("third", r"xyz", false).unwrap(),
        ];
        let matches = find_all(&rules, "abcde xyz");
        assert_eq!(
            matches,
            vec![
                TextMatch {
                    rule: "first".to_owned(),
                    start: 0,
                    end: 3
                },
                TextMatch {
                    rule: "third".to_owned(),
                    start: 6,
                    end: 9
                },
            ]
        );
    }

    #[test]
    fn find_all_returns_rule_order_then_match_order() {
        let rules = vec![
            CompiledRule::compile("word", r"[a-z]+", false).unwrap(),
            CompiledRule::compile("digit", r"\d+", false).unwrap(),
        ];
        let matches = find_all(&rules, "ab1 cd2");
        let described: Vec<(&str, usize)> = matches
            .iter()
            .map(|found| (found.rule.as_str(), found.start))
            .collect();
        assert_eq!(
            described,
            vec![("word", 0), ("word", 4), ("digit", 2), ("digit", 6)]
        );
    }

    #[test]
    fn luhn_accepts_and_rejects_card_numbers() {
        assert!(luhn_valid("4111111111111111"));
        assert!(luhn_valid("4111 1111 1111 1111"));
        assert!(!luhn_valid("4111111111111112"));
        assert!(!luhn_valid(""));
        assert!(!luhn_valid("7"));
        assert!(!luhn_valid("abcd"));
    }

    #[test]
    fn redact_ranges_replaces_and_skips_invalid_ranges() {
        assert_eq!(
            redact_ranges("call 4111 1111 1111 1111 now", &[(5, 24)], "<REDACTED>"),
            "call <REDACTED> now"
        );
        let text = "héllo wörld";
        let start = text.find('é').unwrap();
        let end = start + 'é'.len_utf8();
        assert_eq!(redact_ranges(text, &[(start, end)], "*"), "h*llo wörld");
    }

    #[test]
    fn redact_ranges_tolerates_bad_input() {
        let text = "héllo";
        let inside = text.find('é').unwrap() + 1;
        assert_eq!(redact_ranges(text, &[], "*"), text);
        assert_eq!(redact_ranges(text, &[(2, 2)], "*"), text);
        assert_eq!(redact_ranges(text, &[(3, 1)], "*"), text);
        assert_eq!(redact_ranges(text, &[(0, 999)], "*"), text);
        assert_eq!(redact_ranges(text, &[(inside, inside + 1)], "*"), text);
    }

    #[test]
    fn redact_ranges_skips_overlapping_ranges() {
        assert_eq!(redact_ranges("abcdef", &[(0, 4), (2, 6)], "*"), "ab*");
    }
}
