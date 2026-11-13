//! Diagnostic-signal records and signal tiers.
//!
//! Ports the pure-data part of Python `openjiuwen.agent_evolving.attribution.model`:
//! [`SignalSource`] (provenance), [`DiagnosticSignal`] (one diagnosis output fed to
//! attribution) and the tier helpers.
//!
//! Two Python fields are deliberately not carried over:
//!
//! - `ts_ns` — wall-clock nanoseconds make a signal non-deterministic, and nothing
//!   in this crate orders signals by time; callers that need a timestamp keep it in
//!   [`DiagnosticSignal::metadata`].
//! - the sha256 digest — a local FNV-1a hash keeps the crate dependency-free; the
//!   id is still content-stable, just a different digest from Python's.

use std::collections::BTreeMap;
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The tiers a [`DiagnosticSignal`] can carry, weakest-disclosure last.
///
/// Python: `SIGNAL_TIERS`. A `perfect` signal needs no attribution work (the
/// deliverable is fine); `fair` is room to improve; `weak` is a defect;
/// `uncertain` means the diagnosis could not rate the evidence.
pub const SIGNAL_TIERS: [&str; 4] = ["perfect", "fair", "weak", "uncertain"];

/// The tier a rule-derived signal starts with when diagnosis does not rate it.
const DEFAULT_TIER: &str = "uncertain";

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Maps a normalised reward score in `0.0..=1.0` to a diagnostic tier.
///
/// Bands are closed at the lower edge: `>= 0.9` is `"perfect"`, `>= 0.75` is
/// `"fair"`, anything else is `"weak"`. Python's bands run over a `0..=100`
/// reward sub-dimension score (90 / 75); this crate's reward totals are
/// normalised to `0.0..=1.0`, so the same bands are expressed on that scale.
#[must_use]
pub fn reward_tier(score: f64) -> &'static str {
    if score >= 0.9 {
        "perfect"
    } else if score >= 0.75 {
        "fair"
    } else {
        "weak"
    }
}

/// Where a [`DiagnosticSignal`] came from.
///
/// Python: `SignalSource`. This crate widens the two keyword sources into
/// [`SignalSource::UserFeedback`] / [`SignalSource::ReviewFeedback`] and
/// [`SignalSource::Conversation`] so the host can say what it actually observed;
/// [`SignalSource::Other`] is the tolerant catch-all.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SignalSource {
    /// The end user corrected, complained or asked for more.
    UserFeedback,
    /// A reviewer (human or process) flagged the work in a review round.
    ReviewFeedback,
    /// A programmatic evaluator scored the work.
    Evaluator,
    /// The reward stage scored the work.
    Reward,
    /// A conversation row the host judged worth diagnosing.
    Conversation,
    /// A runtime-recorded failure (a failed tool call, an exception).
    System,
    /// Provenance the caller did not classify.
    #[default]
    Other,
}

impl SignalSource {
    /// The wire spelling of this source, matching the serde `snake_case` value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserFeedback => "user_feedback",
            Self::ReviewFeedback => "review_feedback",
            Self::Evaluator => "evaluator",
            Self::Reward => "reward",
            Self::Conversation => "conversation",
            Self::System => "system",
            Self::Other => "other",
        }
    }

    /// The FNV-1a digest input for `text`: the source spelling and the text,
    /// separated by a NUL byte (Python's `make_signal_id` separates them too).
    fn digest(self, text: &str) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for byte in self.as_str().bytes().chain([0u8]).chain(text.bytes()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

impl fmt::Display for SignalSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One diagnosis output fed to attribution.
///
/// Python: `DiagnosticSignal`. `node_ids` are *lead anchors* — where the problem
/// surfaced — not the located cause; attribution walks the dependency graph from
/// them. The Python `ts_ns` field is intentionally absent: signal ids are
/// content-stable here, so the same text from the same source always yields the
/// same id, independent of when it was observed.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::attribution::{DiagnosticSignal, SignalSource};
///
/// let signal = DiagnosticSignal::new(SignalSource::Reward, "score dropped")
///     .expect("non-blank text")
///     .with_node_ids(vec!["step-1".to_owned()])
///     .with_score(0.25);
/// signal.validate().expect("valid signal");
/// assert_eq!(signal.tier, "uncertain");
/// assert_eq!(signal.node_ids, ["step-1".to_owned()]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DiagnosticSignal {
    /// The stable signal id (`diag-<16hex>`), derived from source and text.
    #[serde(default)]
    pub signal_id: String,
    /// Where the signal came from.
    #[serde(default)]
    pub source: SignalSource,
    /// The diagnosis text, copied verbatim from the evidence.
    #[serde(default)]
    pub text: String,
    /// The diagnostic tier; see [`SIGNAL_TIERS`].
    #[serde(default)]
    pub tier: String,
    /// The trajectory nodes the signal points at (lead anchors, not causes).
    #[serde(default)]
    pub node_ids: Vec<String>,
    /// The score the producing stage assigned, when it assigned one.
    #[serde(default)]
    pub score: Option<f64>,
    /// How much the producer trusts this signal, in `0.0..=1.0`.
    #[serde(default)]
    pub confidence: Option<f64>,
    /// The message the signal was taken from, when there is one.
    #[serde(default)]
    pub message_id: Option<String>,
    /// Extra producer-specific payload; unknown host fields survive here.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl DiagnosticSignal {
    /// Creates a signal with a content-stable id and the `"uncertain"` tier.
    ///
    /// The id is `diag-{16 hex}` of an FNV-1a 64-bit hash over the source
    /// spelling and the text. Python hashes `source\0ts_ns\0text` with sha256 and
    /// keeps 24 hex digits; this crate drops the timestamp (see the module docs)
    /// and uses a shorter, locally computed digest, so ids are stable but not
    /// byte-identical to Python's.
    pub fn new(source: SignalSource, text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(Error::validation(
                "diagnostic signal text must not be blank",
            ));
        }
        let signal_id = format!("diag-{:016x}", source.digest(&text));
        Ok(Self {
            signal_id,
            source,
            text,
            tier: DEFAULT_TIER.to_owned(),
            node_ids: Vec::new(),
            score: None,
            confidence: None,
            message_id: None,
            metadata: BTreeMap::new(),
        })
    }

    /// Overrides the derived signal id.
    #[must_use]
    pub fn with_signal_id(mut self, signal_id: impl Into<String>) -> Self {
        self.signal_id = signal_id.into();
        self
    }

    /// Sets the diagnostic tier.
    #[must_use]
    pub fn with_tier(mut self, tier: impl Into<String>) -> Self {
        self.tier = tier.into();
        self
    }

    /// Sets the lead-anchor node ids.
    #[must_use]
    pub fn with_node_ids(mut self, node_ids: Vec<String>) -> Self {
        self.node_ids = node_ids;
        self
    }

    /// Sets the score the producing stage assigned.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }

    /// Sets the producer's confidence in the signal.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Sets the originating message id.
    #[must_use]
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }

    /// Replaces the metadata payload.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Checks the constraints attribution relies on.
    ///
    /// Rejects a blank `signal_id` or `text`, a non-finite `score`, a
    /// `confidence` that is not finite or outside `0.0..=1.0`, and blank node
    /// ids. The tier is not restricted to [`SIGNAL_TIERS`], so a host can carry
    /// its own vocabulary through.
    pub fn validate(&self) -> Result<()> {
        if self.signal_id.trim().is_empty() {
            return Err(Error::validation("diagnostic signal id must not be blank"));
        }
        if self.text.trim().is_empty() {
            return Err(Error::validation(
                "diagnostic signal text must not be blank",
            ));
        }
        if self.score.is_some_and(|score| !score.is_finite()) {
            return Err(Error::validation(format!(
                "diagnostic signal `{}` carries a non-finite score",
                self.signal_id
            )));
        }
        if let Some(confidence) = self.confidence {
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(Error::validation(format!(
                    "diagnostic signal `{}` confidence must be finite and within 0.0..=1.0, got {confidence}",
                    self.signal_id
                )));
            }
        }
        if self
            .node_ids
            .iter()
            .any(|node_id| node_id.trim().is_empty())
        {
            return Err(Error::validation(format!(
                "diagnostic signal `{}` carries a blank node id",
                self.signal_id
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_ids_are_content_stable() {
        let first = DiagnosticSignal::new(SignalSource::Reward, "score dropped").unwrap();
        let second = DiagnosticSignal::new(SignalSource::Reward, "score dropped").unwrap();
        assert_eq!(first.signal_id, second.signal_id);
        assert!(first.signal_id.starts_with("diag-"));
        assert_eq!(first.signal_id.len(), "diag-".len() + 16);
        let other_source = DiagnosticSignal::new(SignalSource::System, "score dropped").unwrap();
        assert_ne!(first.signal_id, other_source.signal_id);
        let other_text = DiagnosticSignal::new(SignalSource::Reward, "score rose").unwrap();
        assert_ne!(first.signal_id, other_text.signal_id);
        assert_eq!(first.tier, DEFAULT_TIER);
    }

    #[test]
    fn signal_validation_rejects_each_bad_field() {
        assert!(DiagnosticSignal::new(SignalSource::Reward, "   ").is_err());
        let mut signal = DiagnosticSignal::new(SignalSource::Reward, "x").unwrap();
        assert!(signal.validate().is_ok());
        signal.signal_id.clear();
        assert!(signal.validate().is_err());
        signal.signal_id = "diag-0".to_owned();
        signal.score = Some(f64::NAN);
        assert!(signal.validate().is_err());
        signal.score = Some(1.0);
        signal.confidence = Some(1.5);
        assert!(signal.validate().is_err());
        signal.confidence = Some(f64::INFINITY);
        assert!(signal.validate().is_err());
        signal.confidence = Some(1.0);
        signal.node_ids = vec![String::new()];
        assert!(signal.validate().is_err());
    }

    #[test]
    fn reward_tier_bands() {
        assert_eq!(reward_tier(1.0), "perfect");
        assert_eq!(reward_tier(0.9), "perfect");
        assert_eq!(reward_tier(0.89), "fair");
        assert_eq!(reward_tier(0.75), "fair");
        assert_eq!(reward_tier(0.74), "weak");
        assert_eq!(reward_tier(0.0), "weak");
        assert_eq!(reward_tier(f64::NAN), "weak");
        for tier in SIGNAL_TIERS {
            assert_ne!(tier.len(), 0);
        }
        assert_eq!(SIGNAL_TIERS.len(), 4);
    }

    #[test]
    fn source_display_and_serde_spelling_agree() {
        let cases = [
            (SignalSource::UserFeedback, "user_feedback"),
            (SignalSource::ReviewFeedback, "review_feedback"),
            (SignalSource::Evaluator, "evaluator"),
            (SignalSource::Reward, "reward"),
            (SignalSource::Conversation, "conversation"),
            (SignalSource::System, "system"),
            (SignalSource::Other, "other"),
        ];
        for (source, spelling) in cases {
            assert_eq!(source.to_string(), spelling);
            assert_eq!(source.as_str(), spelling);
            let value = serde_json::to_value(source).unwrap();
            assert_eq!(value, Value::String(spelling.to_owned()));
            let restored: SignalSource = serde_json::from_value(value).unwrap();
            assert_eq!(restored, source);
        }
        assert_eq!(SignalSource::default(), SignalSource::Other);
    }

    #[test]
    fn signal_serde_round_trip_and_defaults() {
        let signal = DiagnosticSignal::new(SignalSource::Reward, "score dropped")
            .unwrap()
            .with_node_ids(vec!["step-1".to_owned()])
            .with_score(0.25)
            .with_confidence(0.5)
            .with_message_id("msg-1")
            .with_metadata(BTreeMap::from([("k".to_owned(), Value::from(1))]));
        let value = serde_json::to_value(&signal).unwrap();
        assert_eq!(value["source"], "reward");
        assert_eq!(value["node_ids"][0], "step-1");
        let restored: DiagnosticSignal = serde_json::from_value(value).unwrap();
        assert_eq!(restored, signal);

        // Every field defaults, and unknown fields are tolerated.
        let sparse: DiagnosticSignal =
            serde_json::from_str(r#"{"text": "t", "signal_id": "diag-1", "bogus": 1}"#).unwrap();
        assert_eq!(sparse.source, SignalSource::Other);
        assert_eq!(sparse.tier, "");
        assert_eq!(sparse.node_ids.len(), 0);
        assert_eq!(sparse.metadata.len(), 0);
    }
}
