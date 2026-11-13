//! Deterministic keyword attribution: text vocabulary -> failure target.
//!
//! Ports the determinism of Python `openjiuwen.agent_evolving.attribution.keyword_detect`
//! — the rule pass never reads meaning into words, it only matches a fixed
//! vocabulary — and keeps the match data (`target`, `keyword`, `weight`, byte
//! `position`) so a host can explain why a run was blamed on a target.
//!
//! Python's Jev confidence mode and the trajectory-shaped signal construction are
//! deferred: they need a decision model and the trajectory projection.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The default [`KeywordRule::weight`]: every default rule counts the same.
fn default_weight() -> f64 {
    1.0
}

/// The default [`KeywordRule::case_sensitive`]: vocabulary matching ignores case.
fn default_case_sensitive() -> bool {
    false
}

/// One keyword rule: any of `keywords` in the text blames `target`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct KeywordRule {
    /// The keywords to look for; each must be non-empty.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// The target the blame is assigned to (`"tool"`, `"memory"`, ...).
    #[serde(default)]
    pub target: String,
    /// How strongly a hit counts; must be finite and positive.
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Whether matching is case-sensitive.
    #[serde(default = "default_case_sensitive")]
    pub case_sensitive: bool,
}

impl KeywordRule {
    /// Creates a case-insensitive rule with weight `1.0`.
    #[must_use]
    pub fn new(keywords: Vec<String>, target: impl Into<String>) -> Self {
        Self {
            keywords,
            target: target.into(),
            weight: default_weight(),
            case_sensitive: default_case_sensitive(),
        }
    }

    /// Sets the hit weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Sets whether matching is case-sensitive.
    #[must_use]
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Checks that the rule can be applied.
    ///
    /// Rejects an empty keyword list or target, a blank keyword, and a weight that
    /// is not finite and positive.
    pub fn validate(&self) -> Result<()> {
        if self.target.trim().is_empty() {
            return Err(Error::validation("keyword rule target must not be blank"));
        }
        if self.keywords.is_empty() {
            return Err(Error::validation(format!(
                "keyword rule for target `{}` must declare at least one keyword",
                self.target
            )));
        }
        if self
            .keywords
            .iter()
            .any(|keyword| keyword.trim().is_empty())
        {
            return Err(Error::validation(format!(
                "keyword rule for target `{}` carries a blank keyword",
                self.target
            )));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(Error::validation(format!(
                "keyword rule for target `{}` must carry a finite, positive weight, got {}",
                self.target, self.weight
            )));
        }
        Ok(())
    }
}

/// One keyword hit, with the byte offset of the match in the original text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct KeywordMatch {
    /// The rule target blamed by the hit.
    #[serde(default)]
    pub target: String,
    /// The keyword that matched.
    #[serde(default)]
    pub keyword: String,
    /// The rule weight.
    #[serde(default)]
    pub weight: f64,
    /// The match start as a byte offset into the original text.
    #[serde(default)]
    pub position: usize,
}

/// An ordered set of [`KeywordRule`]s.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::attribution::{KeywordAttributor, KeywordRule};
///
/// let attributor = KeywordAttributor::new(vec![
///     KeywordRule::new(vec!["timeout".to_owned()], "tool").with_weight(2.0),
/// ]);
/// let matches = attributor.attribute("request timeout");
/// assert_eq!(matches.len(), 1);
/// assert_eq!(matches[0].target, "tool");
/// assert_eq!(matches[0].position, 8);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct KeywordAttributor {
    /// The rules, applied in order.
    #[serde(default)]
    pub rules: Vec<KeywordRule>,
}

impl KeywordAttributor {
    /// Creates an attributor over `rules`.
    #[must_use]
    pub fn new(rules: Vec<KeywordRule>) -> Self {
        Self { rules }
    }

    /// Appends one rule.
    #[must_use]
    pub fn with_rule(mut self, rule: KeywordRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// The rules, in application order.
    #[must_use]
    pub fn rules(&self) -> &[KeywordRule] {
        &self.rules
    }

    /// The built-in rule set: tool failure, memory, instruction, script, and
    /// environment vocabulary, each at weight `1.0` and case-insensitive.
    ///
    /// The English and Chinese keywords mirror the failure vocabulary of the Python
    /// rule pass (failed tool rows, the user's follow-up corrections) without
    /// needing a trajectory.
    #[must_use]
    pub fn default_rules() -> Self {
        let rule = |target: &str, keywords: &[&str]| {
            KeywordRule::new(
                keywords
                    .iter()
                    .map(|keyword| (*keyword).to_owned())
                    .collect(),
                target,
            )
        };
        Self {
            rules: vec![
                rule(
                    "tool",
                    &[
                        "error",
                        "exception",
                        "traceback",
                        "failed",
                        "failure",
                        "timeout",
                        "timed out",
                        "denied",
                        "not found",
                        "报错",
                        "失败",
                    ],
                ),
                rule(
                    "memory",
                    &["memory", "remember", "forgot", "forgotten", "内存", "记忆"],
                ),
                rule(
                    "instruction",
                    &[
                        "instruction",
                        "requirement",
                        "as asked",
                        "you should",
                        "did not follow",
                        "指令",
                        "要求",
                    ],
                ),
                rule(
                    "script",
                    &[
                        "script", "python", "bash", "command", "shell", "脚本", "命令",
                    ],
                ),
                rule(
                    "environment",
                    &[
                        "environment",
                        "dependency",
                        "install",
                        "version",
                        "package",
                        "环境",
                        "依赖",
                    ],
                ),
            ],
        }
    }

    /// Every keyword hit in `text`, including hits from overlapping rules and
    /// repeated occurrences of one keyword.
    ///
    /// Results are ordered by byte position ascending, then weight descending, then
    /// target ascending. Blank keywords are skipped (they would match everywhere);
    /// rule weights are used as stored, so call
    /// [`KeywordRule::validate`] before relying on them.
    #[must_use]
    pub fn attribute(&self, text: &str) -> Vec<KeywordMatch> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut matches = Vec::new();
        for rule in &self.rules {
            for keyword in &rule.keywords {
                let keyword_chars: Vec<char> = keyword.chars().collect();
                if keyword_chars.is_empty() || keyword_chars.len() > chars.len() {
                    continue;
                }
                let last_start = chars.len() - keyword_chars.len();
                for start in 0..=last_start {
                    let window = &chars[start..start + keyword_chars.len()];
                    if window_matches(window, &keyword_chars, rule.case_sensitive) {
                        matches.push(KeywordMatch {
                            target: rule.target.clone(),
                            keyword: keyword.clone(),
                            weight: rule.weight,
                            position: chars[start].0,
                        });
                    }
                }
            }
        }
        matches.sort_by(|left, right| {
            left.position
                .cmp(&right.position)
                .then_with(|| right.weight.total_cmp(&left.weight))
                .then_with(|| left.target.cmp(&right.target))
        });
        matches
    }

    /// The strongest hit in `text`: highest weight, then earliest position, then
    /// target ascending.
    #[must_use]
    pub fn best_match(&self, text: &str) -> Option<KeywordMatch> {
        let mut matches = self.attribute(text);
        matches.sort_by(|left, right| {
            right
                .weight
                .total_cmp(&left.weight)
                .then_with(|| left.position.cmp(&right.position))
                .then_with(|| left.target.cmp(&right.target))
        });
        matches.into_iter().next()
    }
}

/// Whether a char window equals the keyword under the requested case folding.
///
/// Lowercasing is compared per `char` (a `char` may lowercase to several chars, and
/// comparing the iterators handles that), so the scan never needs a lowered copy of
/// the text and never risks slicing a `str` at a non-boundary: the window comes from
/// [`str::char_indices`], so `position` is always a valid byte offset of the
/// original text.
fn window_matches(window: &[(usize, char)], keyword: &[char], case_sensitive: bool) -> bool {
    window.iter().zip(keyword).all(|(entry, expected)| {
        let actual = entry.1;
        if case_sensitive {
            actual == *expected
        } else {
            actual.to_lowercase().eq(expected.to_lowercase())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rules_cover_every_target() {
        let attributor = KeywordAttributor::default_rules();
        let targets: Vec<&str> = attributor
            .rules()
            .iter()
            .map(|rule| rule.target.as_str())
            .collect();
        assert_eq!(
            targets,
            ["tool", "memory", "instruction", "script", "environment"]
        );
        for rule in attributor.rules() {
            let target = rule.target.as_str();
            assert!(rule.validate().is_ok(), "rule {target} is invalid");
            assert!(!rule.case_sensitive);
            assert!((rule.weight - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn windows_compare_chars_not_bytes() {
        let chars: Vec<(usize, char)> = "Échec".char_indices().collect();
        let keyword: Vec<char> = "échec".chars().collect();
        assert!(window_matches(&chars, &keyword, false));
        assert!(!window_matches(&chars, &keyword, true));
    }

    #[test]
    fn validation_rejects_bad_rules() {
        assert!(KeywordRule::new(Vec::new(), "tool").validate().is_err());
        assert!(KeywordRule::new(vec![String::new()], "tool")
            .validate()
            .is_err());
        assert!(KeywordRule::new(vec!["x".to_owned()], "  ")
            .validate()
            .is_err());
        assert!(KeywordRule::new(vec!["x".to_owned()], "tool")
            .with_weight(0.0)
            .validate()
            .is_err());
        assert!(KeywordRule::new(vec!["x".to_owned()], "tool")
            .with_weight(f64::INFINITY)
            .validate()
            .is_err());
    }
}
