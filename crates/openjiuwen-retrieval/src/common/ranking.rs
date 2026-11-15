//! Result-ranking configurations and the fixed channel layout they index.
//!
//! Port of the Python `openjiuwen/core/retrieval/common/result_ranking.py`
//! re-export shim, whose real definitions live in
//! `openjiuwen/core/foundation/store/graph/result_ranking.py`:
//!
//! - [`WeightedRankConfig`] is Milvus' weighted ranker: one weight per channel
//!   in `[0, 1]`, defaulting to `0.15 / 0.6 / 0.25`.
//! - [`RrfRankConfig`] is Milvus' reciprocal rank fusion ranker: a smoothing
//!   constant `k` (default `40`) plus one on/off flag per channel.
//! - [`RankConfig`] selects one of them and can apply it to a set of channel
//!   result lists through [`RankConfig::apply`].
//!
//! Both Python configs index the same fixed channel order, exposed here as
//! [`NAME_DENSE`], [`CONTENT_DENSE`], and [`CONTENT_SPARSE`]. This crate's
//! retrievers have no name-embedding channel, so they feed [`CONTENT_DENSE`] and
//! [`CONTENT_SPARSE`] and leave [`NAME_DENSE`] empty; an empty channel
//! contributes nothing to either fusion formula.
//!
//! The Python `register_result_ranker_cls` helper keys ranker callables by
//! backend name (`"milvus"` → pymilvus stubs); Rust has a single built-in
//! implementation, so no registry is needed. The weighted and RRF *formulas*
//! themselves are executed server-side by Milvus, so they are implemented here
//! (see `crate::retriever::fusion`).

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::result::SearchResult;
use crate::retriever::fusion::{rrf_fusion_channels, weighted_fusion_impl};

/// Number of ranking channels (Milvus' fixed order).
pub const RANK_CHANNEL_COUNT: usize = 3;

/// Index of the name-embedding channel (unused by this crate's retrievers).
pub const NAME_DENSE: usize = 0;

/// Index of the content-embedding (dense vector) channel.
pub const CONTENT_DENSE: usize = 1;

/// Index of the content-sparse (lexical/BM25) channel.
pub const CONTENT_SPARSE: usize = 2;

/// Python's `RRFRankConfig.k` default (`40`).
///
/// This deliberately differs from [`crate::retriever::fusion::DEFAULT_RRF_K`]
/// (`60`), which is used by the hybrid, graph, and agentic retrievers: the
/// Python ranking config keeps Milvus' `k = 40`, while `utils/fusion.py` uses
/// `k = 60`. The two constants are intentionally not unified.
pub const DEFAULT_RRF_RANK_K: usize = 40;

/// Default weight for the name-embedding channel.
pub const DEFAULT_NAME_DENSE_WEIGHT: f32 = 0.15;

/// Default weight for the content-embedding channel.
pub const DEFAULT_CONTENT_DENSE_WEIGHT: f32 = 0.6;

/// Default weight for the content-sparse channel.
pub const DEFAULT_CONTENT_SPARSE_WEIGHT: f32 = 0.25;

/// The ranking method a [`RankConfig`] selects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankMethod {
    /// Weighted sum of per-channel min-max normalised scores.
    Weighted,
    /// Reciprocal rank fusion over the active channels.
    #[default]
    Rrf,
}

impl RankMethod {
    /// The Python wire name (`"weighted"` or `"rrf"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::Rrf => "rrf",
        }
    }
}

/// Milvus-style weighted ranker configuration.
///
/// The three weights are positional, matching [`NAME_DENSE`], [`CONTENT_DENSE`],
/// and [`CONTENT_SPARSE`]; each must be finite and within `[0, 1]`.
/// [`Self::normalized_weights`] drops zero weights and renormalises the rest to
/// sum to `1`, which is what Python's `args` property returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WeightedRankConfig {
    /// Weight of the name-embedding channel.
    pub name_dense: f32,
    /// Weight of the content-embedding channel.
    pub content_dense: f32,
    /// Weight of the content-sparse channel.
    pub content_sparse: f32,
}

impl Default for WeightedRankConfig {
    fn default() -> Self {
        Self {
            name_dense: DEFAULT_NAME_DENSE_WEIGHT,
            content_dense: DEFAULT_CONTENT_DENSE_WEIGHT,
            content_sparse: DEFAULT_CONTENT_SPARSE_WEIGHT,
        }
    }
}

impl WeightedRankConfig {
    /// Builds a validated weighted configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending field and value when a
    /// weight is not finite or falls outside `[0, 1]`.
    pub fn new(name_dense: f32, content_dense: f32, content_sparse: f32) -> Result<Self> {
        let config = Self {
            name_dense,
            content_dense,
            content_sparse,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates every weight against `0.0..=1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending field and value.
    pub fn validate(&self) -> Result<()> {
        validate_weight("name_dense", self.name_dense)?;
        validate_weight("content_dense", self.content_dense)?;
        validate_weight("content_sparse", self.content_sparse)?;
        Ok(())
    }

    /// The positional weights as `[name_dense, content_dense, content_sparse]`.
    pub fn weights(&self) -> [f32; RANK_CHANNEL_COUNT] {
        [self.name_dense, self.content_dense, self.content_sparse]
    }

    /// Channel weights renormalised so the positive weights sum to `1`.
    ///
    /// Zero weights are retained at `0.0` so the result stays aligned with the
    /// channel indices; use [`Self::normalized_weights`] for Python's compacted
    /// `args` shape.
    pub fn channel_weights(&self) -> [f32; RANK_CHANNEL_COUNT] {
        let raw = self.weights();
        let total: f32 = raw.iter().copied().filter(|weight| *weight > 0.0).sum();
        let mut normalized = [0.0_f32; RANK_CHANNEL_COUNT];
        if total <= 0.0 {
            return normalized;
        }
        for (slot, weight) in normalized.iter_mut().zip(raw) {
            if weight > 0.0 {
                *slot = weight / total;
            }
        }
        normalized
    }

    /// The positive weights renormalised to sum to `1`, dropping zero weights.
    ///
    /// This mirrors Python's `WeightedRankConfig.args`: an all-zero configuration
    /// yields an empty vector.
    pub fn normalized_weights(&self) -> Vec<f32> {
        let raw = self.weights();
        let active: Vec<f32> = raw.iter().copied().filter(|weight| *weight > 0.0).collect();
        let total: f32 = active.iter().sum();
        if total <= 0.0 {
            return Vec::new();
        }
        active.into_iter().map(|weight| weight / total).collect()
    }

    /// Whether each channel carries a positive weight (`[name, dense, sparse]`).
    pub fn is_active(&self) -> [bool; RANK_CHANNEL_COUNT] {
        let weights = self.weights();
        [weights[0] > 0.0, weights[1] > 0.0, weights[2] > 0.0]
    }

    /// The Python method name (`"weighted"`).
    pub fn method_name(&self) -> &'static str {
        "weighted"
    }

    /// Weighted ranking is distance-based in Milvus, so higher is not better there.
    pub fn higher_is_better(&self) -> bool {
        false
    }

    /// The ranker arguments Python forwards to the backend (compacted weights).
    pub fn args(&self) -> Vec<f32> {
        self.normalized_weights()
    }
}

/// Milvus-style reciprocal rank fusion ranker configuration.
///
/// Unlike [`WeightedRankConfig`], RRF carries no per-channel weights: each
/// channel is on or off, governed by [`Self::is_active`], and only the smoothing
/// constant `k` matters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RrfRankConfig {
    /// RRF smoothing constant (Python default `40`).
    pub k: usize,
    /// Whether the name-embedding channel participates.
    pub name_dense: bool,
    /// Whether the content-embedding channel participates.
    pub content_dense: bool,
    /// Whether the content-sparse channel participates.
    pub content_sparse: bool,
}

impl Default for RrfRankConfig {
    fn default() -> Self {
        Self {
            k: DEFAULT_RRF_RANK_K,
            name_dense: true,
            content_dense: true,
            content_sparse: true,
        }
    }
}

impl RrfRankConfig {
    /// Builds an RRF configuration with `k` and every channel enabled.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Returns this configuration with a different smoothing constant.
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Validates the configuration.
    ///
    /// Python requires `k >= 0`; [`usize`] cannot be negative, so this only
    /// exists for symmetry with [`WeightedRankConfig::validate`].
    ///
    /// # Errors
    ///
    /// Never fails today.
    pub fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Whether each channel participates (`[name, dense, sparse]`).
    pub fn is_active(&self) -> [bool; RANK_CHANNEL_COUNT] {
        [self.name_dense, self.content_dense, self.content_sparse]
    }

    /// The Python method name (`"rrf"`).
    pub fn method_name(&self) -> &'static str {
        "rrf"
    }

    /// Reciprocal rank fusion orders by higher-is-better fused score.
    pub fn higher_is_better(&self) -> bool {
        true
    }

    /// The ranker arguments Python forwards to the backend (`[k]`).
    pub fn args(&self) -> Vec<usize> {
        vec![self.k]
    }
}

/// The backend arguments a [`RankConfig`] exposes through
/// [`RankConfig::ranker_args`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RankerArgs {
    /// Compacted, renormalised channel weights (weighted ranker).
    Weights(Vec<f32>),
    /// The RRF smoothing constant (RRF ranker).
    Rrf(usize),
}

/// A selectable result-ranking configuration.
///
/// Serialises as an internally tagged object: `{"method": "rrf", "k": 40, ...}`
/// or `{"method": "weighted", "name_dense": 0.15, ...}`. The serde default is
/// [`RankConfig::Rrf`] with [`RrfRankConfig::default`] (`k = 40`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RankConfig {
    /// Milvus-style weighted ranker.
    Weighted(WeightedRankConfig),
    /// Reciprocal rank fusion ranker.
    Rrf(RrfRankConfig),
}

impl Default for RankConfig {
    fn default() -> Self {
        Self::Rrf(RrfRankConfig::default())
    }
}

impl From<WeightedRankConfig> for RankConfig {
    fn from(config: WeightedRankConfig) -> Self {
        Self::Weighted(config)
    }
}

impl From<RrfRankConfig> for RankConfig {
    fn from(config: RrfRankConfig) -> Self {
        Self::Rrf(config)
    }
}

impl RankConfig {
    /// The selected ranking method.
    pub fn method(&self) -> RankMethod {
        match self {
            Self::Weighted(_) => RankMethod::Weighted,
            Self::Rrf(_) => RankMethod::Rrf,
        }
    }

    /// The Python method name (`"weighted"` or `"rrf"`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Weighted(config) => config.method_name(),
            Self::Rrf(config) => config.method_name(),
        }
    }

    /// Whether the fused score is higher-is-better (RRF) or not (weighted).
    pub fn higher_is_better(&self) -> bool {
        match self {
            Self::Weighted(config) => config.higher_is_better(),
            Self::Rrf(config) => config.higher_is_better(),
        }
    }

    /// Whether each channel participates (`[name, dense, sparse]`).
    pub fn is_active(&self) -> [bool; RANK_CHANNEL_COUNT] {
        match self {
            Self::Weighted(config) => config.is_active(),
            Self::Rrf(config) => config.is_active(),
        }
    }

    /// The backend arguments this configuration exposes.
    pub fn ranker_args(&self) -> RankerArgs {
        match self {
            Self::Weighted(config) => RankerArgs::Weights(config.args()),
            Self::Rrf(config) => RankerArgs::Rrf(config.k),
        }
    }

    /// The RRF smoothing constant, when this is an RRF configuration.
    pub fn rrf_k(&self) -> Option<usize> {
        match self {
            Self::Rrf(config) => Some(config.k),
            Self::Weighted(_) => None,
        }
    }

    /// Returns an RRF configuration with `k`, keeping the existing RRF flags.
    ///
    /// A weighted configuration is replaced by the default RRF configuration.
    #[must_use]
    pub fn with_rrf_k(self, k: usize) -> Self {
        match self {
            Self::Rrf(mut config) => {
                config.k = k;
                Self::Rrf(config)
            }
            Self::Weighted(_) => Self::Rrf(RrfRankConfig::new(k)),
        }
    }

    /// Validates the selected configuration.
    ///
    /// # Errors
    ///
    /// Propagates [`WeightedRankConfig::validate`].
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Weighted(config) => config.validate(),
            Self::Rrf(config) => config.validate(),
        }
    }

    /// Fuses channel result lists with this configuration.
    ///
    /// `channels` holds one result list per ranking channel, in the fixed order
    /// [`NAME_DENSE`], [`CONTENT_DENSE`], [`CONTENT_SPARSE`]; this crate's
    /// retrievers pass an empty first list. Inactive channels and channels with a
    /// zero weight contribute nothing. The returned list is sorted by descending
    /// fused score with stable ties and keyed by result text (the first-seen
    /// result object supplies the metadata), exactly like
    /// [`crate::retriever::fusion::rrf_fusion`].
    pub fn apply(&self, channels: Vec<Vec<SearchResult>>) -> Vec<SearchResult> {
        match self {
            Self::Weighted(config) => {
                let positional = config.channel_weights();
                let weights: Vec<f32> = (0..channels.len())
                    .map(|index| positional.get(index).copied().unwrap_or(0.0))
                    .collect();
                weighted_fusion_impl(channels, &weights)
            }
            Self::Rrf(config) => rrf_fusion_channels(channels, config),
        }
    }
}

fn validate_weight(field: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::validation(format!(
            "weighted rank config `{field}` must be a finite number between 0.0 and 1.0, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use crate::common::Metadata;
    use serde_json::json;

    const TOLERANCE: f32 = 1e-6;

    fn hit(text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: text.to_owned(),
            text: text.to_owned(),
            score,
            metadata: Metadata::new(),
        }
    }

    fn assert_weights(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < TOLERANCE,
                "weight {actual} != {expected}"
            );
        }
    }

    #[test]
    fn weighted_defaults_match_python() {
        let config = WeightedRankConfig::default();
        assert_weights(&config.weights(), &[0.15, 0.6, 0.25]);
        assert_eq!(config.method_name(), "weighted");
        assert!(!config.higher_is_better());
        assert_eq!(config.is_active(), [true, true, true]);
        assert_eq!(config.args().len(), 3);
    }

    #[test]
    fn weighted_validation_names_the_field_and_value() {
        let error = WeightedRankConfig::new(1.5, 0.6, 0.25).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let message = error.to_string();
        assert!(message.contains("name_dense"), "{message}");
        assert!(message.contains("1.5"), "{message}");

        assert!(WeightedRankConfig::new(-0.1, 0.6, 0.25).is_err());
        assert!(WeightedRankConfig::new(0.0, 0.6, 0.25).is_ok());
        assert!(WeightedRankConfig::new(f32::NAN, 0.6, 0.25).is_err());
    }

    #[test]
    fn weighted_normalisation_drops_zero_and_renormalises() {
        let defaults = WeightedRankConfig::default();
        assert_weights(&defaults.normalized_weights(), &[0.15, 0.6, 0.25]);
        let sum: f32 = defaults.normalized_weights().iter().sum();
        assert!((sum - 1.0).abs() < TOLERANCE);

        let no_name = WeightedRankConfig::new(0.0, 0.6, 0.25).unwrap();
        assert_weights(&no_name.normalized_weights(), &[0.6 / 0.85, 0.25 / 0.85]);
        assert_eq!(no_name.is_active(), [false, true, true]);
        assert_weights(&no_name.channel_weights(), &[0.0, 0.6 / 0.85, 0.25 / 0.85]);

        let all_zero = WeightedRankConfig::new(0.0, 0.0, 0.0).unwrap();
        assert!(all_zero.normalized_weights().is_empty());
        assert_eq!(all_zero.is_active(), [false, false, false]);
        assert_weights(&all_zero.channel_weights(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn rrf_defaults_and_arguments() {
        let config = RrfRankConfig::default();
        assert_eq!(config.k, DEFAULT_RRF_RANK_K);
        assert_eq!(config.k, 40);
        assert!(config.higher_is_better());
        assert_eq!(config.method_name(), "rrf");
        assert_eq!(config.is_active(), [true, true, true]);
        assert_eq!(config.args(), vec![40]);
        assert!(config.validate().is_ok());

        let masked = RrfRankConfig {
            k: 12,
            name_dense: false,
            content_dense: true,
            content_sparse: false,
        };
        assert_eq!(masked.is_active(), [false, true, false]);
        assert_eq!(RrfRankConfig::new(7).k, 7);
    }

    #[test]
    fn rank_config_defaults_to_rrf_with_python_k() {
        let config = RankConfig::default();
        assert_eq!(config.method(), RankMethod::Rrf);
        assert_eq!(config.name(), "rrf");
        assert_eq!(config.rrf_k(), Some(40));
        assert_eq!(config.ranker_args(), RankerArgs::Rrf(40));
        assert_ne!(DEFAULT_RRF_RANK_K, crate::retriever::fusion::DEFAULT_RRF_K);

        let weighted = RankConfig::Weighted(WeightedRankConfig::default());
        assert_eq!(weighted.method(), RankMethod::Weighted);
        assert_eq!(weighted.name(), "weighted");
        assert!(!weighted.higher_is_better());
        assert_eq!(weighted.rrf_k(), None);
        match weighted.ranker_args() {
            RankerArgs::Weights(weights) => assert_weights(&weights, &[0.15, 0.6, 0.25]),
            RankerArgs::Rrf(_) => panic!("expected weights"),
        }
        assert_eq!(weighted.with_rrf_k(9).rrf_k(), Some(9));
    }

    #[test]
    fn configs_serde_round_trip_and_tolerate_unknown_fields() {
        let weighted = WeightedRankConfig::default();
        let value = serde_json::to_value(&weighted).unwrap();
        assert_eq!(value["name_dense"], json!(0.15_f32));
        assert_eq!(
            serde_json::from_value::<WeightedRankConfig>(value).unwrap(),
            weighted
        );

        let rank = RankConfig::Weighted(weighted.clone());
        let value = serde_json::to_value(&rank).unwrap();
        assert_eq!(value["method"], json!("weighted"));
        assert_eq!(serde_json::from_value::<RankConfig>(value).unwrap(), rank);

        let rank = RankConfig::default();
        let value = serde_json::to_value(&rank).unwrap();
        assert_eq!(value["method"], json!("rrf"));
        assert_eq!(value["k"], json!(40));
        assert_eq!(serde_json::from_value::<RankConfig>(value).unwrap(), rank);

        let parsed: RrfRankConfig = serde_json::from_value(json!({"k": 5, "extra": true})).unwrap();
        assert_eq!(parsed.k, 5);
        assert_eq!(
            serde_json::from_value::<RrfRankConfig>(json!({})).unwrap(),
            RrfRankConfig::default()
        );
        assert_eq!(
            serde_json::from_value::<WeightedRankConfig>(json!({})).unwrap(),
            WeightedRankConfig::default()
        );
    }

    #[test]
    fn apply_fuses_in_fixed_channel_order() {
        let channels = vec![
            Vec::new(),
            vec![hit("alpha", 0.9), hit("beta", 0.5), hit("gamma", 0.1)],
            vec![hit("gamma", 10.0), hit("beta", 9.0), hit("alpha", 8.0)],
        ];

        let rrf = RankConfig::default().apply(channels.clone());
        let rrf_texts: Vec<&str> = rrf.iter().map(|result| result.text.as_str()).collect();
        // alpha and gamma both score 1/61 + 1/63; alpha is seen first.
        assert_eq!(rrf_texts, vec!["alpha", "gamma", "beta"]);

        let weighted = RankConfig::Weighted(WeightedRankConfig::default()).apply(channels);
        let weighted_texts: Vec<&str> =
            weighted.iter().map(|result| result.text.as_str()).collect();
        assert_eq!(weighted_texts, vec!["alpha", "beta", "gamma"]);
        assert!((weighted[0].score - 0.6).abs() < TOLERANCE);
        assert!((weighted[1].score - 0.425).abs() < TOLERANCE);
        assert!((weighted[2].score - 0.25).abs() < TOLERANCE);
    }
}
