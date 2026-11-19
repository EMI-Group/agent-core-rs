//! Shared regex rule engine and redaction primitives.
//!
//! Every built-in detector compiles its patterns into [`CompiledRule`]s, finds
//! non-overlapping [`TextMatch`]es, and redacts byte ranges with
//! [`redact_ranges`]. A malformed user-supplied pattern is a construction error,
//! never a panic.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

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
///
/// Claims are kept in an ordered `RangeClaims` map, so the overlap test is
/// `O(log m)` instead of a scan: `m` matches cost `O(m log m)` rather than
/// `O(m²)`, which matters for a tool result holding thousands of `PII` hits.
#[must_use]
pub fn find_all(rules: &[CompiledRule], text: &str) -> Vec<TextMatch> {
    let mut claimed = RangeClaims::new();
    let mut matches = Vec::new();
    for rule in rules {
        for found in rule.regex.find_iter(text) {
            let range = (found.start(), found.end());
            if !claimed.try_claim(range) {
                continue;
            }
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
/// Ranges are considered from the end of the string backwards, so when two
/// ranges overlap the **rightmost** one wins; ranges with the same start are
/// considered in input order, so the first of them wins (the previous
/// end-first implementation sorted them unstably, leaving that tie
/// unspecified). Invalid ranges are skipped instead of panicking: an empty or
/// inverted range, a range beyond the end of `text`, and a range whose bounds
/// are not UTF-8 character boundaries.
///
/// The accepted ranges are selected in one descending pass and the result is
/// rebuilt in one ascending pass with a single pre-sized allocation, so `m`
/// ranges cost `O(m log m)` time (the sort) and no per-range string memmove.
#[must_use]
pub fn redact_ranges(text: &str, ranges: &[(usize, usize)], marker: &str) -> String {
    let mut candidates: Vec<(usize, usize)> = ranges
        .iter()
        .copied()
        .filter(|&(start, end)| {
            start < end
                && end <= text.len()
                && text.is_char_boundary(start)
                && text.is_char_boundary(end)
        })
        .collect();
    // Descending start order, input order as the tiebreak.
    candidates.sort_by_key(|&(start, _)| Reverse(start));

    // While walking right to left the accepted ranges stay pairwise disjoint,
    // so only the leftmost accepted range can still touch a candidate: if it
    // starts before the candidate ends, the two overlap. That makes selection
    // a single linear pass.
    let mut accepted: Vec<(usize, usize)> = Vec::with_capacity(candidates.len());
    let mut leftmost_start: Option<usize> = None;
    for (start, end) in candidates {
        if leftmost_start.is_some_and(|leftmost| end > leftmost) {
            continue;
        }
        leftmost_start = Some(start);
        accepted.push((start, end));
    }

    let mut result = String::with_capacity(text.len() + accepted.len() * marker.len());
    let mut cursor = 0;
    for &(start, end) in accepted.iter().rev() {
        result.push_str(&text[cursor..start]);
        result.push_str(marker);
        cursor = end;
    }
    result.push_str(&text[cursor..]);
    result
}

/// The byte ranges one scan has already claimed, indexed by offset.
///
/// A candidate is claimable when it shares no byte with an accepted range.
/// Accepted ranges are pairwise disjoint, so keeping them ordered by start
/// makes that test two [`BTreeMap`] probes instead of a scan over every claim:
/// claiming `m` ranges costs `O(m log m)` rather than `O(m²)`.
///
/// An empty range (`start == end`, which an empty-matching regex can produce)
/// blocks nothing by itself but does block a candidate that strictly contains
/// its offset, exactly like the byte-wise overlap test it replaces; those
/// offsets live in a separate set so they cannot shadow a real claim.
#[derive(Debug, Default)]
pub(crate) struct RangeClaims {
    /// Non-empty claims, start → end; disjoint, so starts are unique.
    starts: BTreeMap<usize, usize>,
    /// Offsets of accepted empty claims.
    points: BTreeSet<usize>,
}

impl RangeClaims {
    /// An empty claim set.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Claims `range`, reporting whether it was free.
    ///
    /// An inverted range is treated as unclaimable input: like
    /// [`redact_ranges`], the caller's invalid range neither blocks nor is
    /// recorded.
    pub(crate) fn try_claim(&mut self, (start, end): (usize, usize)) -> bool {
        if start > end {
            return true;
        }
        if self.overlaps(start, end) {
            return false;
        }
        if start == end {
            self.points.insert(start);
        } else {
            // Two non-empty claims can never share a start (the second would
            // overlap the first), so a start-keyed map is lossless here.
            self.starts.insert(start, end);
        }
        true
    }

    /// The claimed ranges, ascending by start, empty claims included (as
    /// `(point, point)`), so a caller sees exactly what it claimed.
    pub(crate) fn ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = self
            .starts
            .iter()
            .map(|(&start, &end)| (start, end))
            .chain(self.points.iter().map(|&point| (point, point)))
            .collect();
        ranges.sort_unstable();
        ranges
    }

    /// Whether `start..end` shares at least one byte with a claimed range.
    fn overlaps(&self, start: usize, end: usize) -> bool {
        // An empty claim blocks only when its offset sits strictly inside the
        // candidate: `start < point && point < end`.
        if start < end && self.points.range(start + 1..end).next().is_some() {
            return true;
        }
        // The nearest claim starting before `start` overlaps when it ends after
        // `start`.
        if self
            .starts
            .range(..start)
            .next_back()
            .is_some_and(|(_, claimed_end)| *claimed_end > start)
        {
            return true;
        }
        // The nearest claim starting at or after `start` overlaps when it
        // starts before `end` and ends after `start` (the latter only fails for
        // a claim that shares the candidate's start and is empty).
        self.starts
            .range(start..)
            .next()
            .is_some_and(|(claimed_start, claimed_end)| {
                *claimed_start < end && *claimed_end > start
            })
    }
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

    /// The replaced end-first implementation, kept verbatim as the differential
    /// oracle for [`redact_ranges`].
    fn redact_ranges_reference(text: &str, ranges: &[(usize, usize)], marker: &str) -> String {
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
            if replaced
                .iter()
                .any(|&(claimed_start, claimed_end)| start < claimed_end && claimed_start < end)
            {
                continue;
            }
            result.replace_range(start..end, marker);
            replaced.push((start, end));
        }
        result
    }

    /// The replaced rule-order scan, kept verbatim as the differential oracle
    /// for [`find_all`].
    fn find_all_reference(rules: &[CompiledRule], text: &str) -> Vec<TextMatch> {
        let mut claimed: Vec<(usize, usize)> = Vec::new();
        let mut matches = Vec::new();
        for rule in rules {
            for found in rule.regex.find_iter(text) {
                let (start, end) = (found.start(), found.end());
                if claimed
                    .iter()
                    .any(|&(claimed_start, claimed_end)| start < claimed_end && claimed_start < end)
                {
                    continue;
                }
                claimed.push((start, end));
                matches.push(TextMatch {
                    rule: rule.name.clone(),
                    start,
                    end,
                });
            }
        }
        matches
    }

    #[test]
    fn redact_ranges_is_byte_identical_to_the_end_first_algorithm() {
        let unicode = "héllo wörld ünïcode";
        let é = unicode.find('é').unwrap();
        let ö = unicode.find('ö').unwrap();
        let cases: Vec<(&str, Vec<(usize, usize)>)> = vec![
            ("", vec![]),
            ("", vec![(0, 0), (0, 1)]),
            ("abcdef", vec![]),
            ("abcdef", vec![(0, 6)]),
            // Rightmost wins.
            ("abcdef", vec![(2, 6), (0, 4)]),
            ("abcdef", vec![(0, 4), (2, 6)]),
            // Nested, distinct starts.
            ("abcdefgh", vec![(0, 8), (2, 5), (1, 7)]),
            ("abcdefgh", vec![(2, 5), (0, 8), (1, 7)]),
            // Adjacent ranges all apply.
            ("abcdef", vec![(0, 2), (2, 4), (4, 6)]),
            ("abcdef", vec![(4, 6), (0, 2), (2, 4)]),
            // Identical ranges: one replacement.
            ("abcdef", vec![(2, 5), (2, 5)]),
            ("abcdef", vec![(2, 5), (2, 5), (2, 5)]),
            // Empty, inverted, and out-of-bounds ranges are skipped.
            (
                "abcdef",
                vec![(2, 2), (3, 1), (0, 999), (5, 5), (999, 1000)],
            ),
            // Multibyte ranges.
            (unicode, vec![(é, é + 2), (ö, ö + 2)]),
            (unicode, vec![(é, ö), (ö, ö + 2), (0, é)]),
            // A boundary inside a multibyte character is skipped.
            (unicode, vec![(é + 1, é + 3)]),
            // Out-of-order with a duplicate.
            ("abcdef", vec![(4, 6), (0, 2), (4, 6), (1, 3)]),
        ];
        for (text, ranges) in cases {
            assert_eq!(
                redact_ranges(text, &ranges, "<R>"),
                redact_ranges_reference(text, &ranges, "<R>"),
                "text {text:?} ranges {ranges:?}"
            );
        }
    }

    #[test]
    fn redact_ranges_matches_the_end_first_algorithm_on_many_overlaps() {
        let text: String = (0..400u32)
            .map(|index| {
                if index % 7 == 0 {
                    '\n'
                } else {
                    char::from(b'a' + (index % 26) as u8)
                }
            })
            .collect();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut ranges: Vec<(usize, usize)> = (0..200)
            .map(|index| {
                // Strictly increasing starts keep the tie-break out of the
                // oracle comparison; the lengths make the ranges overlap.
                let start = index * 2;
                (start, (start + (next() % 9) as usize).min(text.len()))
            })
            .collect();
        ranges.rotate_left(37);
        assert_eq!(
            redact_ranges(&text, &ranges, "<R>"),
            redact_ranges_reference(&text, &ranges, "<R>")
        );
    }

    #[test]
    fn redact_ranges_keeps_the_rightmost_range_and_the_first_input_tiebreak() {
        // The rightmost range wins an overlap, so the left replacement is lost.
        assert_eq!(redact_ranges("abcdef", &[(0, 4), (2, 6)], "*"), "ab*");
        assert_eq!(redact_ranges("abcdef", &[(2, 6), (0, 4)], "*"), "ab*");
        // Ranges with the same start: the first one in input order wins (the
        // replaced implementation sorted them unstably, so this was unspecified).
        assert_eq!(redact_ranges("abcdef", &[(2, 6), (2, 4)], "*"), "ab*");
        assert_eq!(redact_ranges("abcdef", &[(2, 4), (2, 6)], "*"), "ab*ef");
        // The inner range wins and the enclosing one is dropped, because the
        // inner one starts further right.
        assert_eq!(redact_ranges("abcdef", &[(0, 6), (2, 3)], "*"), "ab*def");
        assert_eq!(redact_ranges("abcdef", &[(2, 3), (0, 6)], "*"), "ab*def");
    }

    #[test]
    fn redact_ranges_handles_thousands_of_matches() {
        let mut text = String::new();
        for index in 0..5_000 {
            text.push_str("user");
            text.push_str(&index.to_string());
            text.push_str("@example.com ");
        }
        let ranges: Vec<(usize, usize)> = text
            .match_indices('@')
            .map(|(at, _)| {
                let start = text[..at].rfind(' ').map_or(0, |space| space + 1);
                let end = text[at..].find(' ').map_or(text.len(), |space| at + space);
                (start, end)
            })
            .collect();
        assert_eq!(ranges.len(), 5_000);
        let redacted = redact_ranges(&text, &ranges, "<REDACTED>");
        assert_eq!(redacted.matches("<REDACTED>").count(), 5_000);
        assert!(!redacted.contains('@'));
    }

    #[test]
    fn redact_ranges_handles_thousands_of_overlapping_candidates() {
        // Every one of the 5_000 ranges overlaps its neighbours, so the old
        // quadratic overlap scan did ~12.5M comparisons; the result must still
        // be the same rightmost-wins selection.
        let text: String = "x".repeat(20_000);
        let ranges: Vec<(usize, usize)> = (0..5_000)
            .map(|index| {
                let start = index * 2;
                (start, start + 8)
            })
            .collect();
        let redacted = redact_ranges(&text, &ranges, "*");
        assert_eq!(redacted, redact_ranges_reference(&text, &ranges, "*"));
        // The leftmost 8 bytes of every accepted block are claimed first, and
        // the 2-byte stride means only every fourth candidate survives: 1_250
        // disjoint 8-byte blocks replace 10_000 bytes.
        assert_eq!(redacted.matches('*').count(), 1_250);
        assert_eq!(redacted.len(), 20_000 - 1_250 * 8 + 1_250);
    }

    #[test]
    fn find_all_is_identical_to_the_rule_order_scan() {
        let rules = vec![
            CompiledRule::compile("first", r"abc", false).unwrap(),
            CompiledRule::compile("second", r"cde", false).unwrap(),
            CompiledRule::compile("third", r"xyz", false).unwrap(),
            CompiledRule::compile("outer", r"bcd", false).unwrap(),
            CompiledRule::compile("inner", r"c", false).unwrap(),
        ];
        for text in ["abcde xyz", "xyzabc", "", "ccc", "abcdefghij"] {
            assert_eq!(
                find_all(&rules, text),
                find_all_reference(&rules, text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn find_all_keeps_empty_matches_from_claiming_a_later_range() {
        // `x*` matches the empty string at every offset; the old byte-wise
        // overlap test let an empty claim at a strictly interior offset block a
        // later range, and the map-backed claim set must reproduce that.
        let empty = CompiledRule::compile("empty", r"x*", false).unwrap();
        let word = CompiledRule::compile("word", r"ab", false).unwrap();
        let rules = vec![empty.clone(), word.clone()];
        let matches = find_all(&rules, "ab");
        assert_eq!(
            matches
                .iter()
                .map(|found| (found.rule.as_str(), found.start, found.end))
                .collect::<Vec<_>>(),
            vec![("empty", 0, 0), ("empty", 1, 1), ("empty", 2, 2)]
        );
        assert_eq!(matches, find_all_reference(&rules, "ab"));
        // With no empty claim in between, the word range is accepted.
        assert_eq!(
            find_all(std::slice::from_ref(&word), "ab"),
            vec![TextMatch {
                rule: "word".to_owned(),
                start: 0,
                end: 2
            }]
        );
        // Reversing the rule order claims the word first, so only the empty
        // matches outside its bytes survive.
        let reversed = vec![word, empty];
        let matches = find_all(&reversed, "ab");
        assert_eq!(
            matches
                .iter()
                .map(|found| (found.rule.as_str(), found.start, found.end))
                .collect::<Vec<_>>(),
            vec![("word", 0, 2), ("empty", 0, 0), ("empty", 2, 2)]
        );
        assert_eq!(matches, find_all_reference(&reversed, "ab"));
    }

    #[test]
    fn find_all_claims_thousands_of_ranges_without_quadratic_scans() {
        let mut text = String::new();
        for index in 0..2_000 {
            text.push('a');
            text.push_str(&index.to_string());
            text.push(' ');
        }
        let word = CompiledRule::compile("word", r"a\w*", false).unwrap();
        let digit = CompiledRule::compile("digit", r"\d+", false).unwrap();
        let matches = find_all(&[word, digit], &text);
        assert_eq!(
            matches.len(),
            2_000,
            "every inner digit match overlaps a word"
        );
        assert!(matches.iter().all(|found| found.rule == "word"));
    }
}
