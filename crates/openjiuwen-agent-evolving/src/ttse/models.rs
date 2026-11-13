//! Tip records and the small result types the TTSE pipeline passes around.
//!
//! Ports the schema part of `ttse/tip_parse.py`, `ttse/success.py` and
//! `ttse/dream.py`: a tip carries one actionable rule, its category, matching
//! keywords and observation counters.
//!
//! Tip identity is content-addressed ([`tip_id_for`]) rather than random, so
//! identical inputs always produce identical records — tests, checkpoints and
//! bank files stay reproducible without clocks or UUIDs.

use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::trajectory::TrajectoryOutcome;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The kind of advice one [`TipRecord`] carries.
///
/// Python classifies tips into these buckets so consultation can render them in
/// a stable order and the dreamer can merge within a bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TipCategory {
    /// How to call tools, commands, CLIs or APIs.
    Tool,
    /// The order in which workflow steps must happen.
    Workflow,
    /// A pitfall to avoid, typically learned from a failure.
    Pitfall,
    /// Environment, path, config or permission facts.
    Environment,
    /// How to talk to the user or to another agent.
    Communication,
    /// Any other advice; also the classifier's fallback.
    #[default]
    Other,
}

impl TipCategory {
    /// The stable `snake_case` label used by JSON and by the rendered prompt.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Workflow => "workflow",
            Self::Pitfall => "pitfall",
            Self::Environment => "environment",
            Self::Communication => "communication",
            Self::Other => "other",
        }
    }

    /// The stable render order: tool, workflow, pitfall, environment,
    /// communication, other.
    #[must_use]
    pub const fn order(self) -> u8 {
        match self {
            Self::Tool => 0,
            Self::Workflow => 1,
            Self::Pitfall => 2,
            Self::Environment => 3,
            Self::Communication => 4,
            Self::Other => 5,
        }
    }

    /// Parses a category label case-insensitively.
    ///
    /// `"env"` is accepted for [`TipCategory::Environment`] and `"comm"` for
    /// [`TipCategory::Communication`], matching the shorthand Python tips use.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_lowercase().as_str() {
            "tool" => Some(Self::Tool),
            "workflow" => Some(Self::Workflow),
            "pitfall" => Some(Self::Pitfall),
            "environment" | "env" => Some(Self::Environment),
            "communication" | "comm" => Some(Self::Communication),
            "other" => Some(Self::Other),
            _ => None,
        }
    }
}

impl fmt::Display for TipCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One acquired tip: a rule, how to match it, and how well it has worked.
///
/// Unknown JSON fields are ignored and missing fields fall back to
/// [`TipRecord::default`], so bank files written by a newer producer stay
/// readable. [`TipRecord::new`] is the only constructor that checks the rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TipRecord {
    /// The content-addressed identity; see [`tip_id_for`].
    pub tip_id: String,
    /// The actionable rule, one sentence.
    pub rule: String,
    /// What kind of advice this is.
    pub category: TipCategory,
    /// Lowercase matching keywords, sorted and deduped for determinism.
    pub keywords: Vec<String>,
    /// The ranking base weight; observations never change it, the dreamer may.
    pub weight: f64,
    /// How often the tip was consulted.
    pub hits: u64,
    /// How often a consulted tip was followed by a successful run.
    pub successes: u64,
    /// How often a consulted tip was followed by a failed run.
    pub failures: u64,
    /// Provenance label, for example `induction` or `dream`.
    pub source: String,
    /// The bank tick the tip was first stored at.
    pub created_tick: u64,
    /// The bank tick the tip was last touched at.
    pub updated_tick: u64,
}

impl Default for TipRecord {
    fn default() -> Self {
        Self {
            tip_id: String::new(),
            rule: String::new(),
            category: TipCategory::Other,
            keywords: Vec::new(),
            weight: 1.0,
            hits: 0,
            successes: 0,
            failures: 0,
            source: String::new(),
            created_tick: 0,
            updated_tick: 0,
        }
    }
}

impl TipRecord {
    /// Builds a tip from a rule, rejecting a blank rule.
    ///
    /// The id is derived from the category and the normalized rule, so the same
    /// input always yields the same record.
    pub fn new(rule: impl Into<String>, category: TipCategory) -> Result<Self> {
        let rule = rule.into();
        let rule = rule.trim();
        if rule.is_empty() {
            return Err(Error::validation("ttse tip rule must not be empty"));
        }
        Ok(Self {
            tip_id: tip_id_for(category, rule),
            rule: rule.to_owned(),
            category,
            weight: 1.0,
            ..Self::default()
        })
    }

    /// Overrides the derived id, for example when restoring a bank file.
    #[must_use]
    pub fn with_id(mut self, tip_id: impl Into<String>) -> Self {
        self.tip_id = tip_id.into();
        self
    }

    /// Sets the matching keywords, normalized to lowercase, sorted and deduped.
    #[must_use]
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = normalize_keywords(keywords);
        self
    }

    /// Sets the ranking weight; it must be finite and positive.
    pub fn with_weight(mut self, weight: f64) -> Result<Self> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(Error::validation(format!(
                "ttse tip `weight` must be finite and greater than 0.0, got {weight}"
            )));
        }
        self.weight = weight;
        Ok(self)
    }

    /// Sets the provenance label.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Sets the creation and last-update ticks.
    #[must_use]
    pub fn with_ticks(mut self, created_tick: u64, updated_tick: u64) -> Self {
        self.created_tick = created_tick;
        self.updated_tick = updated_tick;
        self
    }

    /// Sets the observation counters.
    #[must_use]
    pub fn with_observation(mut self, hits: u64, successes: u64, failures: u64) -> Self {
        self.hits = hits;
        self.successes = successes;
        self.failures = failures;
        self
    }

    /// The number of decided outcomes, `successes + failures`.
    #[must_use]
    pub fn total_observations(&self) -> u64 {
        self.successes.saturating_add(self.failures)
    }

    /// The share of decided outcomes that succeeded; `0.0` before any observation.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_observations();
        if total == 0 {
            return 0.0;
        }
        self.successes as f64 / total as f64
    }

    /// The effective ranking score, `weight * (0.5 + success_rate())`.
    ///
    /// An unobserved tip scores `0.5 * weight`, a perfect tip `1.5 * weight`, so
    /// a well-observed tip outranks an equally weighted unknown one.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.weight * (0.5 + self.success_rate())
    }

    /// Checks the invariants a bank relies on.
    pub fn validate(&self) -> Result<()> {
        if self.tip_id.trim().is_empty() {
            return Err(Error::validation("ttse tip `tip_id` must not be empty"));
        }
        if self.rule.trim().is_empty() {
            return Err(Error::validation("ttse tip `rule` must not be empty"));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(Error::validation(format!(
                "ttse tip `weight` must be finite and greater than 0.0, got {}",
                self.weight
            )));
        }
        Ok(())
    }
}

/// Normalizes keywords: trimmed, lowercased, non-empty, sorted and deduped.
pub(crate) fn normalize_keywords<I>(keywords: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut normalized: Vec<String> = keywords
        .into_iter()
        .map(|keyword| keyword.trim().to_lowercase())
        .filter(|keyword| !keyword.is_empty())
        .collect();
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

/// Collapses whitespace and lowercases a rule for identity purposes.
fn normalize_rule(rule: &str) -> String {
    rule.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// FNV-1a over the UTF-8 bytes of `text`.
fn fnv1a(text: &str) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Derives the deterministic tip id for a category and rule.
///
/// The id is the 16-lowercase-hex FNV-1a hash of `"{category}:{normalized rule}"`
/// where the normalized rule is trimmed, whitespace-collapsed and lowercased.
/// No clock and no random source is involved, so identical inputs always produce
/// identical ids.
#[must_use]
pub fn tip_id_for(category: TipCategory, rule: &str) -> String {
    let payload = format!("{}:{}", category.as_str(), normalize_rule(rule));
    format!("{:016x}", fnv1a(&payload))
}

/// The outcome of judging whether a trajectory run succeeded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SuccessAssessment {
    /// Whether the run is considered successful.
    pub successful: bool,
    /// The success confidence in `0.0..=1.0`.
    pub score: f64,
    /// The human-readable justification, and the tip evidence once induced.
    pub evidence: String,
}

impl SuccessAssessment {
    /// A successful assessment with a full score.
    #[must_use]
    pub fn success(evidence: impl Into<String>) -> Self {
        Self {
            successful: true,
            score: 1.0,
            evidence: evidence.into(),
        }
    }

    /// A failed assessment with a zero score.
    #[must_use]
    pub fn failure(evidence: impl Into<String>) -> Self {
        Self {
            successful: false,
            score: 0.0,
            evidence: evidence.into(),
        }
    }
}

/// The trajectory evidence one tip was induced from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TipEvidence {
    /// The trajectory the tip came from.
    pub trajectory_id: String,
    /// A bounded excerpt that justifies the tip.
    pub excerpt: String,
    /// The recorded outcome of that trajectory.
    pub outcome: TrajectoryOutcome,
    /// The recorded score of that trajectory, when one was set.
    pub score: Option<f64>,
}

/// What one dream pass changed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DreamReport {
    /// How many tips were merged into a representative.
    pub merged: usize,
    /// How many tips were pruned.
    pub pruned: usize,
    /// How many tips were promoted (weight or category raised).
    pub promoted: usize,
    /// How many tips the pass examined.
    pub examined: usize,
}

/// The ranked tips one consultation produced.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsultResult {
    /// The selected tips, already ranked.
    pub tips: Vec<TipRecord>,
    /// The rendered prompt fragment.
    pub rendered: String,
    /// Whether the rendering hit [`crate::ttse::config::TtseConfig::consult_max_chars`].
    pub truncated: bool,
}

/// Truncates `text` to at most `max_chars` characters, UTF-8 safely.
///
/// When cut, the last kept character is replaced by `…`, so the result is
/// exactly `max_chars` characters. `max_chars == 0` yields the empty string and
/// text that already fits is returned unchanged; a code point is never split.
pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let keep = max_chars - 1;
    let mut truncated: String = text.chars().take(keep).collect();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    #[test]
    fn score_and_success_rate_math() {
        let tip = TipRecord::new("use rg instead of grep", TipCategory::Tool).expect("valid rule");
        assert_eq!(tip.total_observations(), 0);
        assert!((tip.success_rate() - 0.0).abs() < f64::EPSILON);
        assert!((tip.score() - 0.5).abs() < f64::EPSILON);

        let perfect = tip.clone().with_observation(4, 4, 0);
        assert!((perfect.success_rate() - 1.0).abs() < f64::EPSILON);
        assert!((perfect.score() - 1.5).abs() < f64::EPSILON);

        let mixed = tip
            .with_observation(4, 1, 3)
            .with_weight(2.0)
            .expect("valid weight");
        assert!((mixed.success_rate() - 0.25).abs() < f64::EPSILON);
        assert!((mixed.score() - 1.5).abs() < f64::EPSILON);
        assert_eq!(mixed.total_observations(), 4);
    }

    #[test]
    fn tip_id_is_deterministic_and_normalized() {
        let first = tip_id_for(TipCategory::Tool, "  Use   rg  ");
        let second = tip_id_for(TipCategory::Tool, "use rg");
        assert_eq!(first, second);
        assert_eq!(
            TipRecord::new("use rg", TipCategory::Tool).unwrap().tip_id,
            first
        );
        assert_ne!(tip_id_for(TipCategory::Pitfall, "use rg"), first);
        assert_eq!(first.len(), 16);
        assert!(first
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
    }

    #[test]
    fn new_rejects_blank_rule_and_weight_must_be_positive() {
        assert!(TipRecord::new("   ", TipCategory::Tool).is_err());
        let tip = TipRecord::new("check env", TipCategory::Environment).expect("valid rule");
        assert!(tip.clone().with_weight(0.0).is_err());
        assert!(tip.clone().with_weight(f64::NAN).is_err());
        assert!(tip.clone().with_weight(-1.0).is_err());
        assert!((tip.with_weight(2.5).expect("valid weight").weight - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn keywords_are_normalized_sorted_and_deduped() {
        let tip = TipRecord::new("rule", TipCategory::Tool)
            .expect("valid rule")
            .with_keywords(vec![
                " Zed ".to_owned(),
                "alpha".to_owned(),
                String::new(),
                "ALPHA".to_owned(),
            ]);
        assert_eq!(tip.keywords, vec!["alpha".to_owned(), "zed".to_owned()]);
        assert!(tip.validate().is_ok());
    }

    #[test]
    fn truncate_chars_is_utf8_safe_and_exact() {
        assert_eq!(truncate_chars("", 4), "");
        assert_eq!(truncate_chars("日本語テキスト", 3), "日本…");
        assert_eq!(truncate_chars("日本語テキスト", 5).chars().count(), 5);
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abc", 0), "");
        assert_eq!(truncate_chars("😀😀", 1), "…");
    }

    #[test]
    fn serde_round_trip_and_unknown_field_tolerance() {
        let tip = TipRecord::new("confirm before deleting", TipCategory::Pitfall)
            .expect("valid rule")
            .with_keywords(vec!["delete".to_owned()])
            .with_source("induction")
            .with_ticks(3, 7)
            .with_observation(2, 1, 1);
        let json = serde_json::to_string(&tip).expect("serializable");
        let restored: TipRecord = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(restored, tip);

        let partial: Value = json!({ "rule": "hi", "unknown": true });
        let parsed: TipRecord = serde_json::from_value(partial).expect("partial record loads");
        assert_eq!(parsed.rule, "hi");
        assert_eq!(parsed.category, TipCategory::Other);
        assert!((parsed.weight - 1.0).abs() < f64::EPSILON);

        let category: TipCategory = serde_json::from_value(json!("environment")).expect("known");
        assert_eq!(category, TipCategory::Environment);
        assert_eq!(TipCategory::default(), TipCategory::Other);
    }

    #[test]
    fn category_labels_order_and_parse() {
        assert_eq!(TipCategory::Tool.as_str(), "tool");
        assert_eq!(TipCategory::Workflow.to_string(), "workflow");
        assert_eq!(TipCategory::Pitfall.order(), 2);
        assert_eq!(TipCategory::Other.order(), 5);
        let ordered: Vec<u8> = [
            TipCategory::Tool,
            TipCategory::Workflow,
            TipCategory::Pitfall,
            TipCategory::Environment,
            TipCategory::Communication,
            TipCategory::Other,
        ]
        .iter()
        .map(|category| category.order())
        .collect();
        assert_eq!(ordered, vec![0, 1, 2, 3, 4, 5]);

        assert_eq!(TipCategory::parse("  TOOL "), Some(TipCategory::Tool));
        assert_eq!(TipCategory::parse("env"), Some(TipCategory::Environment));
        assert_eq!(TipCategory::parse("COMM"), Some(TipCategory::Communication));
        assert_eq!(TipCategory::parse("nope"), None);
        for category in [
            TipCategory::Tool,
            TipCategory::Workflow,
            TipCategory::Pitfall,
            TipCategory::Environment,
            TipCategory::Communication,
            TipCategory::Other,
        ] {
            assert_eq!(TipCategory::parse(category.as_str()), Some(category));
        }
    }

    #[test]
    fn assessment_evidence_and_report_defaults() {
        let success = SuccessAssessment::success("tests pass");
        assert!(success.successful);
        assert!((success.score - 1.0).abs() < f64::EPSILON);
        let failure = SuccessAssessment::failure("tests fail");
        assert!(!failure.successful);
        assert!((failure.score - 0.0).abs() < f64::EPSILON);
        assert_eq!(SuccessAssessment::default().evidence, "");

        let evidence = TipEvidence {
            trajectory_id: "t-1".to_owned(),
            excerpt: "step".to_owned(),
            outcome: TrajectoryOutcome::Success,
            score: Some(1.0),
        };
        let restored: TipEvidence =
            serde_json::from_str(&serde_json::to_string(&evidence).expect("serializable"))
                .expect("deserializable");
        assert_eq!(restored, evidence);

        assert_eq!(DreamReport::default().examined, 0);
        let consult = ConsultResult::default();
        assert!(consult.tips.is_empty() && consult.rendered.is_empty() && !consult.truncated);
    }
}
