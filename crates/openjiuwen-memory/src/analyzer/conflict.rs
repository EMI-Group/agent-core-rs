//! Conflict classification: is an incoming memory redundant, conflicting, or new?
//!
//! Python's `MemUpdateChecker` asks an LLM to classify recalled memories and merges or
//! deletes accordingly. This module keeps that seam ([`ConflictClassifier`]) and ships a
//! deterministic implementation ([`RuleClassifier`]) so the write path never depends on
//! a healthy model:
//!
//! 1. **redundant** — [`crate::recall::dedupe::find_duplicate`] matches the existing
//!    record (same id, equal normalized text, or a cosine/Jaccard duplicate); the merged
//!    text comes from [`crate::recall::dedupe::merge_records`];
//! 2. **conflicting** — one of three deterministic signals fires:
//!    (i) a correction marker (`actually`, `no longer`, `not ... anymore`, `instead`,
//!    `updated:`, `changed to`, `switched to`, `correction`, `rather than`) **and** a
//!    shared salient token;
//!    (ii) both texts state `X is/are/was/were/am Y` with the same normalized subject
//!    and a different value;
//!    (iii) exactly one side is negated and the negator-stripped token sets reach
//!    [`DedupeConfig::jaccard_threshold`];
//! 3. **unrelated** — everything else.
//!
//! New information never deletes anything here: a verdict only describes the relation
//! and (for redundant/conflicting) the text that should be stored.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::rule::{compile, matches};
use crate::config::DedupeConfig;
use crate::recall::keyword::{jaccard, tokenize};
use crate::recall::{find_duplicate, merge_records, Candidate};
use crate::record::{normalize_text, MemoryRecord, MemoryScope, MemoryType, NewMemory};
use async_trait::async_trait;
use chrono::Utc;
use openjiuwen_common::error::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::LazyLock;

/// Matches a correction marker that signals the incoming statement replaces the old one.
static CORRECTION: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)(?:\b(?:actually|no\s+longer|instead|changed\s+to|switched\s+to|correction|rather\s+than)\b|updated\s*:|\bnot\b[^.;。！？]{0,60}\banymore\b)",
    )
});
/// Captures the subject and value of an `X is/are/was/were/am Y` statement.
static SUBJECT_VALUE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(r"(?i)^(?:(?:the|a|an)\s+)?([\w][\w '\-]{0,48}?)\s+(?:is|are|was|were|am)\s+(.+)$")
});

/// Tokens ignored when looking for a "salient" shared token.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but", "if",
    "then", "so", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "this", "that",
    "these", "those", "it", "its", "i", "we", "you", "they", "he", "she", "my", "our", "your",
    "their", "his", "her", "me", "us", "them", "not", "no", "never", "do", "does", "did", "has",
    "have", "had", "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "there", "here", "now",
];

/// Tokens that mark a negated statement (including apostrophe-contraction stems).
const NEGATORS: &[&str] = &[
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "without",
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "wouldn", "shouldn", "couldn",
    "hasn", "haven", "hadn", "ain", "dont", "doesnt",
];

/// How an incoming memory relates to one existing record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    /// Both memories state the same thing.
    Redundant,
    /// Both memories cannot hold at the same time.
    Conflicting,
    /// Neither redundant nor conflicting; the memory can be added.
    #[default]
    Unrelated,
}

impl ConflictKind {
    /// Returns `true` when the verdict points at an existing record.
    #[must_use]
    pub const fn targets_existing(self) -> bool {
        matches!(self, Self::Redundant | Self::Conflicting)
    }
}

/// The classifier's judgement about an incoming memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConflictVerdict {
    /// The relation to the existing record.
    pub kind: ConflictKind,
    /// Id of the existing record the verdict points at.
    pub target_id: Option<String>,
    /// Why the classifier reached this kind.
    pub reason: Option<String>,
    /// The text that should replace/merge into the target.
    pub merged_content: Option<String>,
    /// Set when a deterministic fallback produced this verdict.
    pub degraded: Option<String>,
}

/// Classifies an incoming memory against existing records.
///
/// Implementations are `Send + Sync` and return one verdict per supplied record, in the
/// order they were supplied.
#[async_trait]
pub trait ConflictClassifier: Send + Sync {
    /// The classifier name used in diagnostics (`rule`/`llm`).
    fn name(&self) -> &str;

    /// Classifies `incoming` against `existing`, using `config` for the thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when `incoming` is not a
    /// valid write input, and the model provider's error for a model-backed classifier
    /// that cannot recover.
    async fn classify(
        &self,
        incoming: &NewMemory,
        existing: &[MemoryRecord],
        config: &DedupeConfig,
    ) -> Result<Vec<ConflictVerdict>>;
}

/// Deterministic, model-free conflict classification.
pub struct RuleClassifier {
    /// The analyzer settings (importance tiers, candidate caps).
    config: AnalyzerConfig,
}

impl RuleClassifier {
    /// Builds a rule classifier with `config`.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }
}

/// Classifies `incoming` against one existing record.
fn verdict_for(
    incoming: &MemoryRecord,
    existing: &MemoryRecord,
    dedupe: &DedupeConfig,
) -> ConflictVerdict {
    let candidate = Candidate::new(existing.clone(), None);
    if let Some(matched) = find_duplicate(incoming, std::slice::from_ref(&candidate), dedupe) {
        let merged = merge_records(existing, incoming, Utc::now());
        return ConflictVerdict {
            kind: ConflictKind::Redundant,
            target_id: Some(existing.id.clone()),
            reason: Some(format!("duplicate detected by `{}`", matched.kind.as_str())),
            merged_content: Some(merged.content),
            degraded: None,
        };
    }
    if let Some(reason) = conflicting_reason(&incoming.content, &existing.content, dedupe) {
        return ConflictVerdict {
            kind: ConflictKind::Conflicting,
            target_id: Some(existing.id.clone()),
            reason: Some(reason.to_owned()),
            merged_content: Some(incoming.content.trim().to_owned()),
            degraded: None,
        };
    }
    ConflictVerdict::default()
}

#[async_trait]
impl ConflictClassifier for RuleClassifier {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "rule"
    }

    async fn classify(
        &self,
        incoming: &NewMemory,
        existing: &[MemoryRecord],
        config: &DedupeConfig,
    ) -> Result<Vec<ConflictVerdict>> {
        incoming.validate()?;
        let incoming_record = record_from_new(incoming);
        Ok(existing
            .iter()
            .map(|record| verdict_for(&incoming_record, record, config))
            .collect())
    }
}

/// Builds a throw-away record for a write input so the dedupe helpers can be reused.
fn record_from_new(memory: &NewMemory) -> MemoryRecord {
    let kind = if memory.kind.is_searchable() {
        memory.kind
    } else {
        MemoryType::SemanticMemory
    };
    let mut record = MemoryRecord::new(MemoryScope::default(), memory.content.clone(), kind);
    if let Some(importance) = memory.importance.filter(|value| value.is_finite()) {
        record.importance = importance.clamp(0.0, 1.0);
    }
    record.metadata = memory.metadata.clone();
    record.source.clone_from(&memory.source);
    record
}

/// Returns the reason the two texts conflict, or `None` when no signal fires.
fn conflicting_reason(
    incoming: &str,
    existing: &str,
    config: &DedupeConfig,
) -> Option<&'static str> {
    if matches(CORRECTION.as_ref(), incoming) && shares_salient_token(incoming, existing) {
        return Some("correction marker");
    }
    if let (Some((incoming_subject, incoming_value)), Some((existing_subject, existing_value))) =
        (subject_value(incoming), subject_value(existing))
    {
        if incoming_subject == existing_subject && incoming_value != existing_value {
            return Some("same subject with a different value");
        }
    }
    if negation_flip(incoming, existing, config.jaccard_threshold) {
        return Some("negation flip");
    }
    None
}

/// Returns whether the two texts share a non-stopword token of three or more characters.
fn shares_salient_token(left: &str, right: &str) -> bool {
    !salient_tokens(left).is_disjoint(&salient_tokens(right))
}

/// Collects the salient tokens of one text.
fn salient_tokens(text: &str) -> HashSet<String> {
    tokenize(text)
        .into_iter()
        .filter(|token| token.chars().count() >= 3 && !STOPWORDS.contains(&token.as_str()))
        .collect()
}

/// Splits an `X is/are/was/were/am Y` statement into its normalized subject and value.
fn subject_value(text: &str) -> Option<(String, String)> {
    let captures = SUBJECT_VALUE.as_ref()?.captures(text)?;
    let subject = normalize_text(captures.get(1)?.as_str());
    let value = normalize_text(captures.get(2)?.as_str());
    if subject.is_empty() || value.is_empty() {
        return None;
    }
    Some((subject, value))
}

/// Returns whether exactly one text is negated and the negator-stripped texts match.
fn negation_flip(incoming: &str, existing: &str, threshold: f32) -> bool {
    let incoming_tokens = tokenize(incoming);
    let existing_tokens = tokenize(existing);
    let incoming_negated = has_negator(&incoming_tokens);
    if incoming_negated == has_negator(&existing_tokens) {
        return false;
    }
    let stripped_incoming = strip_negators(&incoming_tokens);
    let stripped_existing = strip_negators(&existing_tokens);
    if stripped_incoming.is_empty() || stripped_existing.is_empty() {
        return false;
    }
    jaccard(&stripped_incoming, &stripped_existing) >= threshold
}

/// Returns whether any token is a negator.
fn has_negator(tokens: &[String]) -> bool {
    tokens
        .iter()
        .any(|token| NEGATORS.contains(&token.as_str()))
}

/// Joins the non-negator tokens of `tokens` into one string.
fn strip_negators(tokens: &[String]) -> String {
    tokens
        .iter()
        .filter(|token| !NEGATORS.contains(&token.as_str()))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::SemanticMemory,
        )
        .with_id(id)
    }

    async fn classify(incoming: &str, existing: &[MemoryRecord]) -> Vec<ConflictVerdict> {
        RuleClassifier::new(AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic(incoming),
                existing,
                &DedupeConfig::default(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn detects_redundant_memories() {
        let existing = vec![record("id-1", "Alice prefers dark mode")];
        let verdicts = classify("alice prefers DARK mode.", &existing).await;
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
        assert_eq!(verdicts[0].target_id.as_deref(), Some("id-1"));
        assert!(verdicts[0]
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("exact")));
        // The incoming text is longer, so the merge keeps it.
        assert_eq!(
            verdicts[0].merged_content.as_deref(),
            Some("alice prefers DARK mode.")
        );
        assert!(verdicts[0].degraded.is_none());
        assert!(ConflictKind::Redundant.targets_existing());
    }

    #[tokio::test]
    async fn detects_the_correction_marker_signal() {
        let existing = vec![record("id-1", "I prefer coffee")];
        let verdicts = classify("Actually, I prefer tea now", &existing).await;
        assert_eq!(verdicts[0].kind, ConflictKind::Conflicting);
        assert_eq!(verdicts[0].target_id.as_deref(), Some("id-1"));
        assert_eq!(verdicts[0].reason.as_deref(), Some("correction marker"));

        // A marker without a shared salient token is not a conflict.
        let verdicts = classify("Actually, the weather is nice", &existing).await;
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
    }

    #[tokio::test]
    async fn detects_the_same_subject_different_value_signal() {
        let existing = vec![record("id-1", "Alice is a developer")];
        let verdicts = classify("Alice is a manager", &existing).await;
        assert_eq!(verdicts[0].kind, ConflictKind::Conflicting);
        assert_eq!(
            verdicts[0].reason.as_deref(),
            Some("same subject with a different value")
        );
        assert_eq!(
            verdicts[0].merged_content.as_deref(),
            Some("Alice is a manager")
        );

        // Same value: still redundant-by-text only when the text matches, else unrelated.
        let verdicts = classify("Alice is a manager", &[record("id-1", "Bob is a manager")]).await;
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
    }

    #[tokio::test]
    async fn detects_the_negation_flip_signal() {
        let existing = vec![record("id-1", "backups run nightly and weekly")];
        let verdicts = classify("backups never run nightly and weekly either", &existing).await;
        assert_eq!(verdicts[0].kind, ConflictKind::Conflicting);
        assert_eq!(verdicts[0].reason.as_deref(), Some("negation flip"));

        // Both sides negated: no flip.
        let verdicts = classify(
            "backups never run weekly",
            &[record("id-1", "backups never run nightly")],
        )
        .await;
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
    }

    #[tokio::test]
    async fn a_textual_duplicate_wins_over_the_conflict_signals() {
        // `find_duplicate` runs first, so a negation that keeps the token sets close
        // enough is reported as redundant rather than conflicting.
        let existing = vec![record("id-1", "the cache is warmed nightly")];
        let verdicts = classify("the cache is not warmed nightly", &existing).await;
        assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
    }

    #[tokio::test]
    async fn keeps_input_order_and_reports_unrelated_records() {
        let existing = vec![
            record("id-1", "The service uses postgres"),
            record("id-2", "Alice prefers dark mode"),
            record("id-3", "The build pipeline has three stages"),
        ];
        let verdicts = classify("Alice prefers dark mode", &existing).await;
        assert_eq!(verdicts.len(), 3);
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
        assert_eq!(verdicts[1].kind, ConflictKind::Redundant);
        assert_eq!(verdicts[1].target_id.as_deref(), Some("id-2"));
        assert_eq!(verdicts[2].kind, ConflictKind::Unrelated);
        assert!(verdicts[0].target_id.is_none());
        assert!(!ConflictKind::Unrelated.targets_existing());
    }

    #[tokio::test]
    async fn empty_existing_list_yields_no_verdicts() {
        let verdicts = classify("Alice prefers dark mode", &[]).await;
        assert_eq!(verdicts.len(), 0);
    }

    #[tokio::test]
    async fn reject_invalid_incoming_input() {
        let classifier = RuleClassifier::new(AnalyzerConfig::default());
        let error = classifier
            .classify(
                &NewMemory::semantic("   "),
                &[record("id-1", "text")],
                &DedupeConfig::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));

        let error = classifier
            .classify(
                &NewMemory::new("x", MemoryType::Variable),
                &[],
                &DedupeConfig::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(ConflictClassifier::name(&classifier), "rule");
    }

    #[tokio::test]
    async fn respects_the_dedupe_policy_thresholds() {
        let config = DedupeConfig {
            similarity_threshold: 1.0,
            jaccard_threshold: 1.0,
            ..Default::default()
        };
        let existing = vec![record("id-1", "the build pipeline runs nightly")];
        let verdicts = RuleClassifier::new(AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic("the build pipeline runs nightly now"),
                &existing,
                &config,
            )
            .await
            .unwrap();
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
    }

    #[test]
    fn helpers_are_total_on_adversarial_input() {
        for text in [
            "",
            "   ",
            "is",
            "the is x",
            "日本語 テスト",
            "not",
            "not anymore",
        ] {
            let _ = subject_value(text);
            let _ = salient_tokens(text);
            let _ = has_negator(&tokenize(text));
            let _ = strip_negators(&tokenize(text));
            let _ = negation_flip(text, "other text", 0.75);
            let _ = shares_salient_token(text, "other text");
        }
        assert!(!negation_flip("", "", 0.75));
    }

    #[test]
    fn verdict_defaults_and_serde_round_trip() {
        let verdict = ConflictVerdict::default();
        assert_eq!(verdict.kind, ConflictKind::Unrelated);
        assert!(verdict.target_id.is_none());
        assert!(verdict.reason.is_none());
        assert!(verdict.merged_content.is_none());
        assert!(verdict.degraded.is_none());

        let verdict = ConflictVerdict {
            kind: ConflictKind::Conflicting,
            target_id: Some("id-1".to_owned()),
            reason: Some("negation flip".to_owned()),
            merged_content: Some("the api does not support pagination".to_owned()),
            degraded: None,
        };
        let json = serde_json::to_value(&verdict).unwrap();
        for key in ["kind", "target_id", "reason", "merged_content", "degraded"] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["kind"], serde_json::json!("conflicting"));
        let restored: ConflictVerdict = serde_json::from_value(json).unwrap();
        assert_eq!(restored, verdict);
        let tolerant: ConflictVerdict = serde_json::from_value(serde_json::json!({
            "kind": "conflicting",
            "unknown": 1,
        }))
        .unwrap();
        assert_eq!(tolerant.kind, ConflictKind::Conflicting);
        assert!(tolerant.target_id.is_none());
        for (kind, name) in [
            (ConflictKind::Redundant, "redundant"),
            (ConflictKind::Conflicting, "conflicting"),
            (ConflictKind::Unrelated, "unrelated"),
        ] {
            assert_eq!(serde_json::to_value(kind).unwrap(), serde_json::json!(name));
        }
    }
}
