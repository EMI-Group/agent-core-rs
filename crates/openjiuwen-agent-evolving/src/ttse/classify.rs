//! Keyword-based tip classification.
//!
//! Ports `ttse/classify.py`: an ordered keyword table votes on the category of a
//! newly induced rule. Python can also ask a model to classify; that path is an
//! optional seam, while this default is deterministic, offline and unit tested.
//!
//! Scoring is deliberately simple: one point per matched keyword times the rule
//! weight. The winning rule decides the category and the confidence is the share
//! of that rule's keywords that matched.

use serde::{Deserialize, Serialize};

use crate::ttse::models::TipCategory;

/// One classification rule: the keywords it votes with and how loudly.
///
/// Unknown JSON fields are ignored; missing fields fall back to
/// [`ClassificationRule::default`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassificationRule {
    /// Substrings that vote for [`Self::category`]; matched case-insensitively.
    pub keywords: Vec<String>,
    /// The category this rule votes for.
    pub category: TipCategory,
    /// How much one matched keyword is worth.
    pub weight: f64,
}

impl Default for ClassificationRule {
    fn default() -> Self {
        Self {
            keywords: Vec::new(),
            category: TipCategory::Other,
            weight: 1.0,
        }
    }
}

/// The default classification rule table, ranked by how specific each category is.
///
/// [`TipClassifier::new`] sorts the table, so the order written here only needs
/// to stay readable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TipClassifier {
    /// The rules, sorted by `(weight desc, category order asc, keywords asc)`.
    pub rules: Vec<ClassificationRule>,
}

impl Default for TipClassifier {
    fn default() -> Self {
        Self::new(vec![
            ClassificationRule {
                keywords: keywords(&[
                    "tool", "tools", "call", "command", "cli", "shell", "api", "script",
                ]),
                category: TipCategory::Tool,
                weight: 1.0,
            },
            ClassificationRule {
                keywords: keywords(&[
                    "workflow", "order", "step", "steps", "sequence", "first", "then", "before",
                    "after", "pipeline",
                ]),
                category: TipCategory::Workflow,
                weight: 1.0,
            },
            ClassificationRule {
                keywords: keywords(&[
                    "error",
                    "fail",
                    "failed",
                    "failure",
                    "avoid",
                    "never",
                    "wrong",
                    "mistake",
                    "timeout",
                    "exception",
                ]),
                category: TipCategory::Pitfall,
                weight: 1.2,
            },
            ClassificationRule {
                keywords: keywords(&[
                    "environment",
                    "env",
                    "path",
                    "directory",
                    "config",
                    "install",
                    "permission",
                    "workspace",
                ]),
                category: TipCategory::Environment,
                weight: 1.0,
            },
            ClassificationRule {
                keywords: keywords(&[
                    "explain",
                    "ask",
                    "confirm",
                    "clarify",
                    "report",
                    "summarize",
                    "communicate",
                    "tell",
                ]),
                category: TipCategory::Communication,
                weight: 0.9,
            },
        ])
    }
}

/// Builds a keyword vector from string literals.
fn keywords(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

impl TipClassifier {
    /// Builds a classifier, sorting the rules deterministically by
    /// `(weight desc, category order asc, keywords asc)`.
    #[must_use]
    pub fn new(mut rules: Vec<ClassificationRule>) -> Self {
        rules.sort_by(|left, right| {
            right
                .weight
                .total_cmp(&left.weight)
                .then_with(|| left.category.order().cmp(&right.category.order()))
                .then_with(|| left.keywords.join(" ").cmp(&right.keywords.join(" ")))
        });
        Self { rules }
    }

    /// The category of `text`, or [`TipCategory::Other`] when nothing matches.
    #[must_use]
    pub fn classify(&self, text: &str) -> TipCategory {
        self.classify_scored(text).0
    }

    /// The category of `text` and the confidence of the winning rule.
    ///
    /// Every rule scores `matched keywords * weight`; the highest score wins and
    /// ties go to the earlier (already sorted) rule. The confidence is the share
    /// of the winning rule's keywords that matched, clamped to `0.0..=1.0`. Text
    /// with no match yields `(TipCategory::Other, 0.0)`.
    #[must_use]
    pub fn classify_scored(&self, text: &str) -> (TipCategory, f64) {
        let haystack = text.to_lowercase();
        let mut best: Option<(TipCategory, f64, f64)> = None;
        for rule in &self.rules {
            if rule.keywords.is_empty() {
                continue;
            }
            let matched = rule
                .keywords
                .iter()
                .filter(|keyword| haystack.contains(&keyword.to_lowercase()))
                .count();
            if matched == 0 {
                continue;
            }
            let score = matched as f64 * rule.weight;
            let better = match &best {
                None => true,
                Some((_, best_score, _)) => score > *best_score,
            };
            if better {
                let confidence = (matched as f64 / rule.keywords.len() as f64).clamp(0.0, 1.0);
                best = Some((rule.category, score, confidence));
            }
        }
        match best {
            Some((category, _, confidence)) => (category, confidence),
            None => (TipCategory::Other, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn classifier() -> TipClassifier {
        TipClassifier::default()
    }

    #[test]
    fn tool_and_pitfall_sentences_classify() {
        let classifier = classifier();
        assert_eq!(
            classifier.classify("call the shell command with a script"),
            TipCategory::Tool
        );
        assert_eq!(
            classifier.classify("avoid the timeout error, never retry blindly"),
            TipCategory::Pitfall
        );
        assert_eq!(
            classifier.classify("an exception means the call failed"),
            TipCategory::Pitfall
        );
    }

    #[test]
    fn workflow_and_communication_sentences_classify() {
        let classifier = classifier();
        assert_eq!(
            classifier.classify("first run the tests, then the formatter"),
            TipCategory::Workflow
        );
        assert_eq!(
            classifier.classify("ask the user to confirm and report the result"),
            TipCategory::Communication
        );
        assert_eq!(
            classifier.classify("the workspace directory path comes from config"),
            TipCategory::Environment
        );
    }

    #[test]
    fn cjk_text_is_classified_not_other() {
        // The default table is ASCII (as specified); CJK keywords are matched as
        // plain substrings, so a localized table classifies CJK text directly.
        let classifier = TipClassifier::new(vec![
            ClassificationRule {
                keywords: vec!["命令".to_owned(), "工具".to_owned()],
                category: TipCategory::Tool,
                weight: 1.0,
            },
            ClassificationRule {
                keywords: vec!["环境".to_owned(), "路径".to_owned()],
                category: TipCategory::Environment,
                weight: 1.0,
            },
        ]);
        let (category, confidence) = classifier.classify_scored("运行命令时先检查环境");
        assert_ne!(category, TipCategory::Other);
        assert_eq!(category, TipCategory::Tool);
        assert!(confidence > 0.0 && confidence <= 1.0);
        assert_eq!(
            classifier.classify("运行命令时先检查环境"),
            TipCategory::Tool
        );
    }

    #[test]
    fn empty_or_unmatched_text_falls_back_to_other() {
        let classifier = classifier();
        assert_eq!(classifier.classify_scored(""), (TipCategory::Other, 0.0));
        assert_eq!(
            classifier.classify_scored("zzz qqq"),
            (TipCategory::Other, 0.0)
        );
        assert_eq!(
            TipClassifier::new(Vec::new()).classify("anything"),
            TipCategory::Other
        );
    }

    #[test]
    fn new_sorts_rules_deterministically() {
        let forward = TipClassifier::default();
        let mut reversed = forward.rules.clone();
        reversed.reverse();
        assert_eq!(TipClassifier::new(reversed), forward);
        let weights: Vec<f64> = forward.rules.iter().map(|rule| rule.weight).collect();
        assert!(
            (weights[0] - 1.2).abs() < f64::EPSILON,
            "pitfall ranks first"
        );
        assert_eq!(
            forward.classify("explain and confirm"),
            TipCategory::Communication
        );
    }

    #[test]
    fn serde_round_trip_and_defaults() {
        let rule = ClassificationRule::default();
        assert!((rule.weight - 1.0).abs() < f64::EPSILON);
        assert_eq!(rule.category, TipCategory::Other);
        assert_eq!(rule.keywords.len(), 0);

        let classifier = classifier();
        let json = serde_json::to_string(&classifier).expect("serializable");
        let restored: TipClassifier = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(restored, classifier);

        let partial: TipClassifier =
            serde_json::from_value(json!({ "unknown": true })).expect("partial");
        assert_eq!(partial, TipClassifier::default());
    }
}
