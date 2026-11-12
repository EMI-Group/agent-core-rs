//! Expectation assertions and their outcomes.
//!
//! Python expressed expectations as free-form strings checked by ad-hoc code.
//! Here an expectation is parsed once into an [`Assertion`], which is both
//! serializable and executable, and every check yields an [`AssertionOutcome`]
//! carrying the evidence that justified it.
//!
//! Five kinds exist, each written as `kind:argument`:
//!
//! | Expression | Meaning |
//! |---|---|
//! | `contains:text` | the output contains `text` |
//! | `not_contains:text` | the output does not contain `text` |
//! | `equals:text` | the trimmed output equals the trimmed `text` |
//! | `starts_with:text` | the left-trimmed output starts with `text` |
//! | `regex:pattern` | the pattern matches anywhere in the output |
//!
//! Prefixes are matched case-insensitively; a textual expectation with no
//! recognized prefix is a [`Assertion::Contains`].

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

/// The longest evidence excerpt kept in an [`AssertionOutcome`], in characters.
const EVIDENCE_LIMIT: usize = 200;

/// One checkable expectation on a skill's answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Assertion {
    /// The output contains the argument.
    Contains(String),
    /// The output does not contain the argument.
    NotContains(String),
    /// The trimmed output equals the trimmed argument.
    Equals(String),
    /// The left-trimmed output starts with the argument.
    StartsWith(String),
    /// The argument compiles as a regular expression matching the output.
    Regex(String),
}

impl Assertion {
    /// Parses an expectation expression.
    ///
    /// The input is trimmed first. A recognized `kind:` prefix (case-insensitive)
    /// selects the assertion kind and the remainder of the input, trimmed, is its
    /// argument; otherwise the whole trimmed input is a
    /// [`Contains`](Self::Contains) argument.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for a blank expression, or when a `regex:` argument does not compile.
    pub fn parse(text: &str) -> Result<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(Error::validation("assertion must not be blank"));
        }
        if let Some(argument) = strip_prefix_ignore_ascii_case(trimmed, "not_contains:") {
            return Ok(Self::NotContains(argument.trim().to_string()));
        }
        if let Some(argument) = strip_prefix_ignore_ascii_case(trimmed, "contains:") {
            return Ok(Self::Contains(argument.trim().to_string()));
        }
        if let Some(argument) = strip_prefix_ignore_ascii_case(trimmed, "equals:") {
            return Ok(Self::Equals(argument.trim().to_string()));
        }
        if let Some(argument) = strip_prefix_ignore_ascii_case(trimmed, "starts_with:") {
            return Ok(Self::StartsWith(argument.trim().to_string()));
        }
        if let Some(argument) = strip_prefix_ignore_ascii_case(trimmed, "regex:") {
            let pattern = argument.trim();
            Regex::new(pattern).map_err(|error| {
                Error::validation(format!("invalid assertion regex {pattern:?}: {error}"))
            })?;
            return Ok(Self::Regex(pattern.to_string()));
        }
        Ok(Self::Contains(trimmed.to_string()))
    }

    /// Renders the assertion in canonical `kind:argument` form.
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::Contains(argument) => format!("contains:{argument}"),
            Self::NotContains(argument) => format!("not_contains:{argument}"),
            Self::Equals(argument) => format!("equals:{argument}"),
            Self::StartsWith(argument) => format!("starts_with:{argument}"),
            Self::Regex(argument) => format!("regex:{argument}"),
        }
    }

    /// Checks this assertion against an output.
    ///
    /// The returned outcome records whether the assertion held and a prefix
    /// excerpt of the output as evidence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when a [`Regex`](Self::Regex) argument no longer compiles.
    pub fn check(&self, output: &str) -> Result<AssertionOutcome> {
        let passed = match self {
            Self::Contains(argument) => output.contains(argument.as_str()),
            Self::NotContains(argument) => !output.contains(argument.as_str()),
            Self::Equals(argument) => output.trim() == argument.trim(),
            Self::StartsWith(argument) => output.trim_start().starts_with(argument.as_str()),
            Self::Regex(argument) => Regex::new(argument)
                .map_err(|error| {
                    Error::validation(format!("invalid assertion regex {argument:?}: {error}"))
                })?
                .is_match(output),
        };
        Ok(AssertionOutcome {
            text: self.text(),
            passed,
            evidence: excerpt(output),
        })
    }
}

/// The result of checking one [`Assertion`] against an output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AssertionOutcome {
    /// The assertion in canonical `kind:argument` form.
    pub text: String,
    /// Whether the assertion held.
    pub passed: bool,
    /// A prefix excerpt of the checked output (at most 200 characters).
    pub evidence: String,
}

/// Returns the first 200 characters of `output` as evidence.
///
/// The excerpt is a plain prefix, so it is deterministic and cheap: it always
/// starts at the beginning of the output rather than around the matched text.
/// Truncation happens on a character boundary and is marked with a trailing `…`,
/// which still keeps the result at 200 characters. An empty output yields an
/// empty excerpt.
fn excerpt(output: &str) -> String {
    if output.chars().count() <= EVIDENCE_LIMIT {
        return output.to_string();
    }
    let mut kept: String = output.chars().take(EVIDENCE_LIMIT - 1).collect();
    kept.push('…');
    kept
}

/// Strips an ASCII-case-insensitive prefix, returning the remaining text.
fn strip_prefix_ignore_ascii_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &text[prefix.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reads_every_kind() {
        assert_eq!(
            Assertion::parse("contains:hello").expect("contains"),
            Assertion::Contains("hello".to_string())
        );
        assert_eq!(
            Assertion::parse("not_contains:secret").expect("not_contains"),
            Assertion::NotContains("secret".to_string())
        );
        assert_eq!(
            Assertion::parse("equals:  done  ").expect("equals"),
            Assertion::Equals("done".to_string())
        );
        assert_eq!(
            Assertion::parse("starts_with:Hello").expect("starts_with"),
            Assertion::StartsWith("Hello".to_string())
        );
        assert_eq!(
            Assertion::parse("regex:^a+$").expect("regex"),
            Assertion::Regex("^a+$".to_string())
        );
    }

    #[test]
    fn parse_accepts_mixed_case_prefixes() {
        assert_eq!(
            Assertion::parse("Contains:x").expect("mixed case"),
            Assertion::Contains("x".to_string())
        );
        assert_eq!(
            Assertion::parse("REGEX:^a+$").expect("upper case"),
            Assertion::Regex("^a+$".to_string())
        );
        assert_eq!(
            Assertion::parse("Not_Contains:y").expect("upper not_contains"),
            Assertion::NotContains("y".to_string())
        );
    }

    #[test]
    fn parse_distinguishes_not_contains_from_contains() {
        let assertion = Assertion::parse("not_contains:foo").expect("parse");
        assert_eq!(assertion, Assertion::NotContains("foo".to_string()));
        assert!(!assertion.check("foo bar").expect("check").passed);
    }

    #[test]
    fn parse_treats_bare_text_as_contains() {
        assert_eq!(
            Assertion::parse("  plain text  ").expect("bare"),
            Assertion::Contains("plain text".to_string())
        );
    }

    #[test]
    fn parse_rejects_blank_input() {
        let error = Assertion::parse("   ").expect_err("blank");
        assert!(error.to_string().contains("blank"));
        assert!(Assertion::parse("").is_err());
    }

    #[test]
    fn parse_rejects_invalid_regex() {
        let error = Assertion::parse("regex:[").expect_err("invalid regex");
        assert!(error.to_string().contains("invalid assertion regex"));
    }

    #[test]
    fn text_round_trips_every_kind() {
        let expressions = [
            "contains:hello",
            "not_contains:secret",
            "equals:done",
            "starts_with:Hello",
            "regex:^a+$",
        ];
        for expression in expressions {
            let assertion = Assertion::parse(expression).expect("parse");
            assert_eq!(assertion.text(), expression);
            assert_eq!(
                Assertion::parse(&assertion.text()).expect("re-parse"),
                assertion
            );
        }
    }

    #[test]
    fn check_contains_passes_and_fails() {
        let assertion = Assertion::Contains("needle".to_string());
        assert!(assertion.check("a needle here").expect("check").passed);
        assert!(!assertion.check("nothing").expect("check").passed);
    }

    #[test]
    fn check_not_contains_passes_and_fails() {
        let assertion = Assertion::NotContains("secret".to_string());
        assert!(assertion.check("clean").expect("check").passed);
        assert!(!assertion.check("a secret").expect("check").passed);
    }

    #[test]
    fn check_equals_trims_both_sides() {
        let assertion = Assertion::Equals("done".to_string());
        assert!(assertion.check("  done\n").expect("check").passed);
        assert!(!assertion.check("done more").expect("check").passed);
    }

    #[test]
    fn check_starts_with_trims_leading_whitespace() {
        let assertion = Assertion::StartsWith("Hello".to_string());
        assert!(assertion.check("  Hello there").expect("check").passed);
        assert!(!assertion.check("Bye Hello").expect("check").passed);
    }

    #[test]
    fn check_regex_matches_anywhere() {
        let assertion = Assertion::Regex(r"\d{3}".to_string());
        assert!(assertion.check("id 123").expect("check").passed);
        assert!(!assertion.check("id 12").expect("check").passed);
        assert_eq!(assertion.check("").expect("check").evidence, "");
    }

    #[test]
    fn check_evidence_is_empty_for_empty_output() {
        let outcome = Assertion::Contains("x".to_string())
            .check("")
            .expect("check");
        assert_eq!(outcome.evidence, "");
        assert!(!outcome.passed);
        assert_eq!(outcome.text, "contains:x");
    }

    #[test]
    fn check_evidence_is_capped_at_200_chars() {
        let output = "a".repeat(1000);
        let outcome = Assertion::Contains("a".to_string())
            .check(&output)
            .expect("check");
        assert_eq!(outcome.evidence.chars().count(), EVIDENCE_LIMIT);
        assert!(outcome.evidence.ends_with('…'));

        let short = "b".repeat(200);
        let outcome = Assertion::Contains("b".to_string())
            .check(&short)
            .expect("check");
        assert_eq!(outcome.evidence, short);
    }

    #[test]
    fn check_evidence_truncates_on_char_boundaries() {
        let output = "汉字".repeat(300);
        let outcome = Assertion::Contains("汉字".to_string())
            .check(&output)
            .expect("check");
        assert_eq!(outcome.evidence.chars().count(), EVIDENCE_LIMIT);
    }

    #[test]
    fn outcome_round_trips_through_json() {
        let outcome = AssertionOutcome {
            text: "contains:x".to_string(),
            passed: true,
            evidence: "x marks the spot".to_string(),
        };
        let json = serde_json::to_string(&outcome).expect("serialize");
        let back: AssertionOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, outcome);
    }

    #[test]
    fn assertion_round_trips_through_json() {
        let assertion = Assertion::StartsWith("abc".to_string());
        let json = serde_json::to_string(&assertion).expect("serialize");
        let back: Assertion = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, assertion);
    }
}
