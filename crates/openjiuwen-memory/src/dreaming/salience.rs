//! Deterministic salience scoring: which memories are worth consolidating.
//!
//! A consolidation sweep needs to decide *what* to compact first, so every stored
//! [`MemoryRecord`] is scored on four independent, observable components — importance,
//! recency, access frequency, and kind — and the weighted average ranks the candidates.
//! The scoring is fully deterministic and offline: the clock is an argument ([`now`] is
//! always injected), the components need no model, and the same record always produces
//! the same [`SalienceScore`].
//!
//! The formula (implemented by [`salience_score`]) is fixed:
//!
//! ```text
//! importance = record.importance                          (clamped into 0.0..=1.0; non-finite -> 0.0)
//! age_secs   = max(0, (now - record.timestamp).num_seconds())
//! recency    = 0.5 ^ (age_secs / half_life_secs)
//! access     = access_count / (access_count + access_saturation)   (0.0 when access_count == 0)
//! kind       = kind_weights.weight(record.kind)
//! total      = (wi * importance + wr * recency + wa * access + wk * kind)
//!              / (wi + wr + wa + wk)
//! ```
//!
//! `wi`/`wr`/`wa`/`wk` are [`SalienceWeights`] and the sum in the denominator is the
//! renormalising factor, so disabling a term by setting its weight to `0.0` is safe; when
//! every weight is `0.0` the total is `0.0` rather than `NaN`. A timestamp in the future
//! has its age clamped to `0` (recency `1.0`), and the access term saturates towards `1.0`
//! as the access count grows past [`SalienceConfig::access_saturation`].
//!
//! Python's `core.memory.dreaming` package has no salience step at all; this module
//! replaces the previously deferred ML salience model with a rule-based scorer.
//! [`SalienceBackend::Ml`] is kept as an explicit, typed seam: it returns
//! [`Error::Unsupported`] naming ML-based salience as deferred and never silently falls
//! back to the heuristic scorer.
use crate::record::{MemoryRecord, MemoryType};
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Default recency half-life in seconds (30 days), matching the recall blend.
const DEFAULT_HALF_LIFE_SECS: u64 = 2_592_000;

/// Default access count at which the access term reaches `0.5`.
const DEFAULT_ACCESS_SATURATION: f32 = 10.0;

/// Which salience implementation a [`SalienceConfig`] selects.
///
/// Only [`SalienceBackend::Heuristic`] is implemented; the ML variant exists so a caller
/// that asks for a model-backed scorer gets a typed [`Error::Unsupported`] instead of a
/// silent downgrade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SalienceBackend {
    /// The deterministic, model-free scorer documented in the module header.
    #[default]
    Heuristic,
    /// A model-backed salience model. Deferred: every scoring call returns
    /// [`Error::Unsupported`].
    Ml,
}

/// The four term weights of the salience blend.
///
/// Each weight is finite and non-negative; the terms are renormalised by their sum, so
/// only the ratios between the weights matter. The defaults weight every term equally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SalienceWeights {
    /// Weight of the record's own `importance` term; default `1.0`.
    pub importance: f32,
    /// Weight of the exponential recency term; default `1.0`.
    pub recency: f32,
    /// Weight of the access-frequency term; default `1.0`.
    pub access: f32,
    /// Weight of the memory-kind term; default `1.0`.
    pub kind: f32,
}

impl Default for SalienceWeights {
    /// The documented defaults: every term weighs `1.0`.
    fn default() -> Self {
        Self {
            importance: 1.0,
            recency: 1.0,
            access: 1.0,
            kind: 1.0,
        }
    }
}

impl SalienceWeights {
    /// Returns the weights with the `importance` weight replaced.
    #[must_use]
    pub const fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    /// Returns the weights with the `recency` weight replaced.
    #[must_use]
    pub const fn with_recency(mut self, recency: f32) -> Self {
        self.recency = recency;
        self
    }

    /// Returns the weights with the `access` weight replaced.
    #[must_use]
    pub const fn with_access(mut self, access: f32) -> Self {
        self.access = access;
        self
    }

    /// Returns the weights with the `kind` weight replaced.
    #[must_use]
    pub const fn with_kind(mut self, kind: f32) -> Self {
        self.kind = kind;
        self
    }

    /// Validates the weights, returning `Error::Validation` naming the offending field.
    ///
    /// Every weight must be finite and greater than or equal to `0.0` (NaN, infinity, and
    /// negative weights are rejected rather than clamped).
    pub fn validate(&self) -> Result<()> {
        validate_weight("salience weights `importance`", self.importance)?;
        validate_weight("salience weights `recency`", self.recency)?;
        validate_weight("salience weights `access`", self.access)?;
        validate_weight("salience weights `kind`", self.kind)?;
        Ok(())
    }
}

/// The per-kind multipliers of the salience kind term.
///
/// Durable user facts rank above derived facts and raw events, so the defaults descend
/// from [`MemoryType::UserProfile`] (`1.0`) to [`MemoryType::Unknown`] (`0.5`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KindWeights {
    /// Weight of [`MemoryType::UserProfile`]; default `1.0`.
    pub user_profile: f32,
    /// Weight of [`MemoryType::SemanticMemory`]; default `0.9`.
    pub semantic_memory: f32,
    /// Weight of [`MemoryType::EpisodicMemory`]; default `0.7`.
    pub episodic_memory: f32,
    /// Weight of [`MemoryType::Summary`]; default `0.8`.
    pub summary: f32,
    /// Weight of [`MemoryType::Coding`]; default `0.85`.
    pub coding: f32,
    /// Weight of [`MemoryType::Variable`]; default `0.6`.
    pub variable: f32,
    /// Weight of [`MemoryType::Unknown`]; default `0.5`.
    pub unknown: f32,
}

impl Default for KindWeights {
    /// The documented defaults: `1.0`, `0.9`, `0.7`, `0.8`, `0.85`, `0.6`, `0.5`.
    fn default() -> Self {
        Self {
            user_profile: 1.0,
            semantic_memory: 0.9,
            episodic_memory: 0.7,
            summary: 0.8,
            coding: 0.85,
            variable: 0.6,
            unknown: 0.5,
        }
    }
}

impl KindWeights {
    /// Returns the weight of one memory kind.
    ///
    /// The value is the raw configured multiplier, not clamped: the defaults all sit in
    /// `0.0..=1.0`, and a caller that configures more than `1.0` gets that larger term.
    #[must_use]
    pub const fn weight(&self, kind: MemoryType) -> f32 {
        match kind {
            MemoryType::UserProfile => self.user_profile,
            MemoryType::SemanticMemory => self.semantic_memory,
            MemoryType::EpisodicMemory => self.episodic_memory,
            MemoryType::Summary => self.summary,
            MemoryType::Coding => self.coding,
            MemoryType::Variable => self.variable,
            MemoryType::Unknown => self.unknown,
        }
    }

    /// Validates the weights, returning `Error::Validation` naming the offending field.
    ///
    /// Every kind weight must be finite and greater than or equal to `0.0`.
    pub fn validate(&self) -> Result<()> {
        validate_weight("kind weights `user_profile`", self.user_profile)?;
        validate_weight("kind weights `semantic_memory`", self.semantic_memory)?;
        validate_weight("kind weights `episodic_memory`", self.episodic_memory)?;
        validate_weight("kind weights `summary`", self.summary)?;
        validate_weight("kind weights `coding`", self.coding)?;
        validate_weight("kind weights `variable`", self.variable)?;
        validate_weight("kind weights `unknown`", self.unknown)?;
        Ok(())
    }
}

/// The salience configuration: backend, recency half-life, access saturation, and weights.
///
/// [`SalienceConfig::validate`] must pass before scoring; it rejects a zero half-life and
/// a non-positive or non-finite access saturation, and delegates to
/// [`SalienceWeights::validate`] and [`KindWeights::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SalienceConfig {
    /// The scorer to use; default [`SalienceBackend::Heuristic`].
    pub backend: SalienceBackend,
    /// Recency half-life in seconds; default `2_592_000` (30 days).
    ///
    /// An age equal to this value scores `0.5`; two half-lives score `0.25`. Matches the
    /// recall blend's half-life so consolidation and recall agree on "stale".
    pub half_life_secs: u64,
    /// Access count at which the access term reaches `0.5`; default `10.0`.
    ///
    /// Larger values make the access term rise more slowly toward its `1.0` asymptote.
    pub access_saturation: f32,
    /// The four blend weights; default [`SalienceWeights::default`].
    pub weights: SalienceWeights,
    /// The per-kind multipliers; default [`KindWeights::default`].
    pub kind_weights: KindWeights,
}

impl Default for SalienceConfig {
    /// The documented defaults: heuristic backend, a 30-day half-life, an access
    /// saturation of `10.0`, and the default weights.
    fn default() -> Self {
        Self {
            backend: SalienceBackend::Heuristic,
            half_life_secs: DEFAULT_HALF_LIFE_SECS,
            access_saturation: DEFAULT_ACCESS_SATURATION,
            weights: SalienceWeights::default(),
            kind_weights: KindWeights::default(),
        }
    }
}

impl SalienceConfig {
    /// Returns the configuration with the scoring backend replaced.
    #[must_use]
    pub const fn with_backend(mut self, backend: SalienceBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the configuration with the recency half-life replaced (in seconds).
    #[must_use]
    pub const fn with_half_life_secs(mut self, half_life_secs: u64) -> Self {
        self.half_life_secs = half_life_secs;
        self
    }

    /// Returns the configuration with the access saturation replaced.
    #[must_use]
    pub const fn with_access_saturation(mut self, access_saturation: f32) -> Self {
        self.access_saturation = access_saturation;
        self
    }

    /// Returns the configuration with the blend weights replaced.
    #[must_use]
    pub fn with_weights(mut self, weights: SalienceWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Returns the configuration with the per-kind multipliers replaced.
    #[must_use]
    pub fn with_kind_weights(mut self, kind_weights: KindWeights) -> Self {
        self.kind_weights = kind_weights;
        self
    }

    /// Validates the configuration, returning `Error::Validation` naming the field.
    ///
    /// Requirements: a non-zero `half_life_secs`, a finite `access_saturation` greater
    /// than `0.0`, valid `weights`, and valid `kind_weights`.
    pub fn validate(&self) -> Result<()> {
        if self.half_life_secs == 0 {
            return Err(Error::validation(
                "salience config `half_life_secs` must be greater than 0",
            ));
        }
        if !self.access_saturation.is_finite() || self.access_saturation <= 0.0 {
            return Err(Error::validation(format!(
                "salience config `access_saturation` must be finite and greater than 0 (got {})",
                self.access_saturation
            )));
        }
        self.weights.validate()?;
        self.kind_weights.validate()
    }
}

/// The salience of one record: the total plus the four components it was blended from.
///
/// The formula documented on [`salience_score`] fixes every field:
/// - `importance` is the record's own importance clamped into `0.0..=1.0` (a non-finite
///   value scores `0.0`);
/// - `recency` is `0.5 ^ (age_secs / half_life_secs)` with the age clamped at `0`, so it
///   lies in `0.0..=1.0` and is `1.0` for a fresh or future-dated record;
/// - `access` is `access_count / (access_count + access_saturation)`, exactly `0.0` for an
///   untouched record and rising monotonically toward `1.0`;
/// - `kind` is [`KindWeights::weight`] for the record's kind;
/// - `total` is the weighted average of the four components, or `0.0` when every weight is
///   `0.0`.
///
/// The components are kept so a consolidation plan can explain itself; only `total` is
/// used for ordering by [`rank_by_salience`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct SalienceScore {
    /// The blend of every component; the only field used for ordering.
    pub total: f32,
    /// The record's clamped `importance`.
    pub importance: f32,
    /// The exponential recency, `1.0` at age `0` and halving per half-life.
    pub recency: f32,
    /// The saturating access-frequency term, `0.0` when never recalled.
    pub access: f32,
    /// The per-kind multiplier from [`SalienceConfig::kind_weights`].
    pub kind: f32,
}

/// Scores one record at `now`.
///
/// The configuration is validated first ([`SalienceConfig::validate`]), then the backend
/// is checked: [`SalienceBackend::Ml`] returns [`Error::Unsupported`] naming ML-based
/// salience as deferred rather than falling back to the heuristic scorer.
///
/// With a valid configuration the components and the total are computed exactly as
/// documented in the module header:
///
/// ```text
/// importance = record.importance                          (clamped into 0.0..=1.0; non-finite -> 0.0)
/// age_secs   = max(0, (now - record.timestamp).num_seconds())
/// recency    = 0.5 ^ (age_secs / half_life_secs)
/// access     = access_count / (access_count + access_saturation)   (0.0 when access_count == 0)
/// kind       = kind_weights.weight(record.kind)
/// total      = (wi * importance + wr * recency + wa * access + wk * kind)
///              / (wi + wr + wa + wk)
/// ```
///
/// A weight sum of `0.0` yields a total of `0.0` (the components are still reported); the
/// total is never `NaN` because the weights are validated as finite and non-negative.
pub fn salience_score(
    record: &MemoryRecord,
    now: DateTime<Utc>,
    config: &SalienceConfig,
) -> Result<SalienceScore> {
    config.validate()?;
    ensure_heuristic_backend(config.backend)?;
    Ok(heuristic_score(record, now, config))
}

/// Scores every record at `now` and returns `(record id, score)` pairs sorted by
/// descending total, with ascending record id as the tie-break.
///
/// The configuration is validated once and the backend checked once, so an invalid or
/// [`SalienceBackend::Ml`] configuration yields the same error as [`salience_score`] and
/// no record is scored. Sorting uses [`f32::total_cmp`], so equal totals are ordered
/// deterministically by id instead of depending on input order.
pub fn rank_by_salience(
    records: &[MemoryRecord],
    now: DateTime<Utc>,
    config: &SalienceConfig,
) -> Result<Vec<(String, SalienceScore)>> {
    config.validate()?;
    ensure_heuristic_backend(config.backend)?;
    let mut scored: Vec<(String, SalienceScore)> = records
        .iter()
        .map(|record| (record.id.clone(), heuristic_score(record, now, config)))
        .collect();
    scored.sort_by(|(left_id, left), (right_id, right)| {
        right
            .total
            .total_cmp(&left.total)
            .then_with(|| left_id.cmp(right_id))
    });
    Ok(scored)
}

/// Rejects the deferred ML backend, naming it explicitly.
fn ensure_heuristic_backend(backend: SalienceBackend) -> Result<()> {
    match backend {
        SalienceBackend::Heuristic => Ok(()),
        SalienceBackend::Ml => Err(Error::unsupported(
            "salience backend `ml` is deferred: ML-based salience scoring is not implemented; \
             use `SalienceBackend::Heuristic`",
        )),
    }
}

/// Computes the score for an already validated heuristic configuration.
fn heuristic_score(
    record: &MemoryRecord,
    now: DateTime<Utc>,
    config: &SalienceConfig,
) -> SalienceScore {
    let importance = clamp_unit(record.importance);
    let recency = crate::recall::recency_score(record.timestamp, now, config.half_life_secs);
    let access = access_component(record.access_count, config.access_saturation);
    let kind = config.kind_weights.weight(record.kind);
    let weights = &config.weights;
    let weight_sum = weights.importance + weights.recency + weights.access + weights.kind;
    let total = if weight_sum > 0.0 {
        (weights.importance * importance
            + weights.recency * recency
            + weights.access * access
            + weights.kind * kind)
            / weight_sum
    } else {
        0.0
    };
    SalienceScore {
        total,
        importance,
        recency,
        access,
        kind,
    }
}

/// The access-frequency term: `access_count / (access_count + saturation)`.
///
/// The fraction is computed in `f64` so large access counts keep their precision, then
/// narrowed to `f32`. An untouched record scores exactly `0.0`.
fn access_component(access_count: u64, saturation: f32) -> f32 {
    if access_count == 0 {
        return 0.0;
    }
    let count = access_count as f64;
    let saturation = f64::from(saturation);
    (count / (count + saturation)) as f32
}

/// Clamps a component into `0.0..=1.0`, mapping a non-finite value to `0.0`.
fn clamp_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Rejects a weight that is non-finite or negative, naming the field.
fn validate_weight(field: &str, weight: f32) -> Result<()> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(Error::validation(format!(
            "{field} must be finite and >= 0 (got {weight})"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::record::{MemoryRecord, MemoryScope};
    use chrono::Duration;
    use openjiuwen_common::error::Error;

    /// One microsecond of slack: the components are `f32` arithmetic.
    const EPSILON: f32 = 1e-6;

    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    fn approx(left: f32, right: f32) -> bool {
        (left - right).abs() <= EPSILON
    }

    /// A record stamped at [`fixed_time`] with default importance and zero accesses.
    fn record(content: &str, kind: MemoryType) -> MemoryRecord {
        let now = fixed_time();
        let mut record = MemoryRecord::new(MemoryScope::new("alice", "ws"), content, kind);
        record.timestamp = now;
        record.updated_at = now;
        record.access_count = 0;
        record
    }

    #[test]
    fn defaults_are_documented_values() {
        let config = SalienceConfig::default();
        assert_eq!(config.backend, SalienceBackend::Heuristic);
        assert_eq!(config.backend, SalienceBackend::default());
        assert_eq!(config.half_life_secs, 2_592_000);
        assert!(approx(config.access_saturation, 10.0));
        let weights = &config.weights;
        for weight in [
            weights.importance,
            weights.recency,
            weights.access,
            weights.kind,
        ] {
            assert!(approx(weight, 1.0));
        }
        // In `MemoryType::ALL` order: user, semantic, episodic, variable, summary, coding, unknown.
        let expected = [1.0, 0.9, 0.7, 0.6, 0.8, 0.85, 0.5];
        for (kind, expected) in MemoryType::ALL.into_iter().zip(expected) {
            assert!(
                approx(config.kind_weights.weight(kind), expected),
                "kind {kind}"
            );
        }
        let score = SalienceScore::default();
        for value in [
            score.total,
            score.importance,
            score.recency,
            score.access,
            score.kind,
        ] {
            assert!(approx(value, 0.0));
        }
        assert!(config.validate().is_ok());
    }

    #[test]
    fn builders_are_chainable() {
        let weights = SalienceWeights::default()
            .with_importance(0.5)
            .with_recency(0.4)
            .with_access(0.3)
            .with_kind(0.2);
        assert!(approx(weights.importance, 0.5) && approx(weights.recency, 0.4));
        assert!(approx(weights.access, 0.3) && approx(weights.kind, 0.2));
        let kinds = KindWeights {
            episodic_memory: 0.1,
            ..KindWeights::default()
        };
        assert!(approx(kinds.episodic_memory, 0.1));
        let config = SalienceConfig::default()
            .with_backend(SalienceBackend::Heuristic)
            .with_half_life_secs(86_400)
            .with_access_saturation(4.0)
            .with_weights(weights.clone())
            .with_kind_weights(kinds.clone());
        assert_eq!(config.half_life_secs, 86_400);
        assert!(approx(config.access_saturation, 4.0));
        assert_eq!(config.weights, weights);
        assert_eq!(config.kind_weights, kinds);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn formula_matches_at_fixed_now() {
        let now = fixed_time();
        let mut record = record("Alice prefers dark mode", MemoryType::UserProfile);
        record.importance = 0.8;
        record.timestamp = now - Duration::seconds(2_592_000);
        record.access_count = 10;
        let score = salience_score(&record, now, &SalienceConfig::default()).unwrap();
        assert!(approx(score.importance, 0.8));
        assert!(approx(score.recency, 0.5));
        assert!(approx(score.access, 0.5));
        assert!(approx(score.kind, 1.0));
        // (0.8 + 0.5 + 0.5 + 1.0) / 4.0
        assert!(approx(score.total, (0.8 + 0.5 + 0.5 + 1.0) / 4.0));
        assert!(approx(score.total, 0.7));
    }

    #[test]
    fn formula_honours_custom_weights_and_saturation() {
        let now = fixed_time();
        let mut record = record("fact", MemoryType::SemanticMemory);
        record.importance = 1.0;
        record.timestamp = now - Duration::seconds(86_400);
        record.access_count = 3;
        let config = SalienceConfig::default()
            .with_half_life_secs(86_400)
            .with_access_saturation(1.0)
            .with_weights(
                SalienceWeights::default()
                    .with_importance(2.0)
                    .with_recency(1.0)
                    .with_access(1.0)
                    .with_kind(0.0),
            );
        let score = salience_score(&record, now, &config).unwrap();
        // recency = 0.5 ^ (86_400 / 86_400) = 0.5; access = 3 / (3 + 1) = 0.75.
        assert!(approx(score.recency, 0.5));
        assert!(approx(score.access, 0.75));
        assert!(approx(score.kind, 0.9));
        // kind weight is 0.0, so the kind term is dropped from the blend.
        let expected = (2.0 * 1.0 + 1.0 * 0.5 + 1.0 * 0.75) / 4.0;
        assert!(approx(score.total, expected));
        assert!(approx(score.total, 0.8125));
    }

    #[test]
    fn default_components_for_a_fresh_record() {
        let now = fixed_time();
        let score = salience_score(
            &record("fresh", MemoryType::SemanticMemory),
            now,
            &SalienceConfig::default(),
        )
        .unwrap();
        assert!(approx(score.importance, 0.5));
        assert!(approx(score.recency, 1.0));
        assert!(approx(score.access, 0.0));
        assert!(approx(score.kind, 0.9));
        assert!(approx(score.total, (0.5 + 1.0 + 0.0 + 0.9) / 4.0));
    }

    #[test]
    fn future_timestamps_clamp_age_to_zero() {
        let now = fixed_time();
        let mut future = record("future", MemoryType::UserProfile);
        future.timestamp = now + Duration::days(3_650);
        let score = salience_score(&future, now, &SalienceConfig::default()).unwrap();
        assert!(approx(score.recency, 1.0));
        assert!(approx(score.total, (0.5 + 1.0 + 0.0 + 1.0) / 4.0));
    }

    #[test]
    fn recency_halves_per_half_life() {
        let now = fixed_time();
        let config = SalienceConfig::default();
        let mut record = record("decaying", MemoryType::SemanticMemory);
        let mut previous = 1.0_f32;
        for half_lives in 1..=4_i64 {
            record.timestamp = now - Duration::seconds(2_592_000 * half_lives);
            let score = salience_score(&record, now, &config).unwrap();
            assert!(score.recency < previous);
            assert!(approx(
                score.recency,
                0.5_f32.powi(i32::try_from(half_lives).unwrap())
            ));
            previous = score.recency;
        }
    }

    #[test]
    fn importance_is_clamped_and_non_finite_scores_zero() {
        let now = fixed_time();
        let config = SalienceConfig::default();
        for (importance, expected) in [
            (1.5, 1.0),
            (-0.25, 0.0),
            (f32::NAN, 0.0),
            (f32::INFINITY, 0.0),
            (f32::NEG_INFINITY, 0.0),
            (0.25, 0.25),
        ] {
            let record = record("importance", MemoryType::UserProfile).with_importance(importance);
            let score = salience_score(&record, now, &config).unwrap();
            assert!(
                approx(score.importance, expected),
                "importance {importance}"
            );
        }
    }

    #[test]
    fn access_is_monotonic_and_saturating() {
        let now = fixed_time();
        let config = SalienceConfig::default();
        let component = |access_count: u64| {
            let mut record = record("accessed", MemoryType::SemanticMemory);
            record.access_count = access_count;
            salience_score(&record, now, &config).unwrap().access
        };
        assert!(approx(component(0), 0.0));
        let values: Vec<f32> = (0..=4).map(component).collect();
        for pair in values.windows(2) {
            assert!(pair[1] > pair[0], "access must increase: {values:?}");
        }
        // access = n / (n + 10): 1/11, 2/12, 3/13, 4/14.
        assert!(approx(values[0], 0.0));
        assert!(approx(values[1], 1.0 / 11.0));
        assert!(approx(values[2], 2.0 / 12.0));
        // Successive gains shrink as the term saturates.
        let gains: Vec<f32> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(gains[1] < gains[0]);
        assert!(gains[2] < gains[1]);
        assert!(gains[3] < gains[2]);
        // Even a huge access count stays below 1.0.
        assert!(component(10_000) < 1.0);
        assert!(approx(component(10), 0.5));
    }

    #[test]
    fn kind_weights_cover_every_kind() {
        let kinds = KindWeights::default();
        assert!(approx(kinds.weight(MemoryType::UserProfile), 1.0));
        assert!(approx(kinds.weight(MemoryType::SemanticMemory), 0.9));
        assert!(approx(kinds.weight(MemoryType::EpisodicMemory), 0.7));
        assert!(approx(kinds.weight(MemoryType::Summary), 0.8));
        assert!(approx(kinds.weight(MemoryType::Coding), 0.85));
        assert!(approx(kinds.weight(MemoryType::Variable), 0.6));
        assert!(approx(kinds.weight(MemoryType::Unknown), 0.5));
        assert!(kinds.validate().is_ok());
    }

    #[test]
    fn zero_weight_sum_yields_zero_total() {
        let now = fixed_time();
        let mut record = record("weightless", MemoryType::UserProfile);
        record.importance = 1.0;
        record.access_count = 42;
        let config = SalienceConfig::default().with_weights(
            SalienceWeights::default()
                .with_importance(0.0)
                .with_recency(0.0)
                .with_access(0.0)
                .with_kind(0.0),
        );
        let score = salience_score(&record, now, &config).unwrap();
        assert!(approx(score.total, 0.0));
        assert!(score.total.is_finite());
        // The components are still reported.
        assert!(approx(score.importance, 1.0));
        assert!(approx(score.recency, 1.0));
        assert!(score.access > 0.0);
        assert!(approx(score.kind, 1.0));
    }

    #[test]
    fn weights_are_renormalized() {
        let now = fixed_time();
        let mut record = record("scaled", MemoryType::Coding);
        record.importance = 0.7;
        record.timestamp = now - Duration::seconds(1_296_000);
        record.access_count = 4;
        let unit = salience_score(&record, now, &SalienceConfig::default()).unwrap();
        let scaled = SalienceWeights::default()
            .with_importance(2.0)
            .with_recency(2.0)
            .with_access(2.0)
            .with_kind(2.0);
        let doubled = salience_score(
            &record,
            now,
            &SalienceConfig::default().with_weights(scaled),
        )
        .unwrap();
        assert!(approx(unit.total, doubled.total));
    }

    #[test]
    fn negative_or_non_finite_weights_are_rejected() {
        let cases = [
            (
                SalienceWeights::default().with_importance(-0.1),
                "importance",
            ),
            (SalienceWeights::default().with_recency(f32::NAN), "recency"),
            (
                SalienceWeights::default().with_access(f32::INFINITY),
                "access",
            ),
            (SalienceWeights::default().with_kind(-1.0), "kind"),
        ];
        for (weights, field) in cases {
            let error = weights.validate().unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert!(
                error.to_string().contains(field),
                "message `{error}` must name `{field}`"
            );
            let config = SalienceConfig::default().with_weights(weights);
            assert!(config.validate().is_err());
            let record = record("invalid weights", MemoryType::UserProfile);
            assert!(salience_score(&record, fixed_time(), &config).is_err());
        }
    }

    #[test]
    fn invalid_kind_weights_are_rejected() {
        let defaults = KindWeights::default();
        let cases = [
            (
                KindWeights {
                    user_profile: -0.5,
                    ..defaults.clone()
                },
                "user_profile",
            ),
            (
                KindWeights {
                    semantic_memory: f32::NAN,
                    ..defaults.clone()
                },
                "semantic_memory",
            ),
            (
                KindWeights {
                    coding: f32::INFINITY,
                    ..defaults.clone()
                },
                "coding",
            ),
            (
                KindWeights {
                    unknown: -0.01,
                    ..defaults
                },
                "unknown",
            ),
        ];
        for (kind_weights, field) in cases {
            let error = kind_weights.validate().unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert!(
                error.to_string().contains(field),
                "message `{error}` must name `{field}`"
            );
            let config = SalienceConfig::default().with_kind_weights(kind_weights);
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn half_life_and_access_saturation_are_validated() {
        let error = SalienceConfig::default()
            .with_half_life_secs(0)
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("half_life_secs"));
        for saturation in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let error = SalienceConfig::default()
                .with_access_saturation(saturation)
                .validate()
                .unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert!(
                error.to_string().contains("access_saturation"),
                "saturation {saturation} must be named in `{error}`"
            );
        }
        assert!(SalienceConfig::default()
            .with_half_life_secs(1)
            .with_access_saturation(0.5)
            .validate()
            .is_ok());
    }

    #[test]
    fn ml_backend_is_unsupported() {
        let config = SalienceConfig::default().with_backend(SalienceBackend::Ml);
        assert!(config.validate().is_ok());
        let record = record("model backed", MemoryType::UserProfile);
        let error = salience_score(&record, fixed_time(), &config).unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
        let message = error.to_string();
        assert!(message.contains("ML"), "message was `{message}`");
        assert!(message.contains("deferred"), "message was `{message}`");
        let ranked = rank_by_salience(&[record], fixed_time(), &config).unwrap_err();
        assert!(matches!(ranked, Error::Unsupported(_)));
    }

    #[test]
    fn rank_orders_by_total_and_each_component() {
        let now = fixed_time();
        let config = SalienceConfig::default();
        let low = record("low", MemoryType::Unknown).with_importance(0.1);
        let high = record("high", MemoryType::UserProfile).with_importance(0.9);
        let middle = record("middle", MemoryType::Summary).with_importance(0.5);
        let ranked = rank_by_salience(&[low, high, middle], now, &config).unwrap();
        assert_eq!(ranked.len(), 3);
        assert!(approx(ranked[0].1.kind, 1.0));
        assert!(approx(ranked[2].1.kind, 0.5));
        for pair in ranked.windows(2) {
            assert!(pair[0].1.total >= pair[1].1.total);
        }

        // Importance: everything else equal, the more important record wins.
        let important = record("important", MemoryType::SemanticMemory).with_importance(0.9);
        let unimportant = record("unimportant", MemoryType::SemanticMemory).with_importance(0.1);
        let ranked = rank_by_salience(&[unimportant, important], now, &config).unwrap();
        assert!(approx(ranked[0].1.importance, 0.9));
        assert!(approx(ranked[1].1.importance, 0.1));

        // Recency: the fresher record wins.
        let fresh = record("fresh", MemoryType::SemanticMemory);
        let mut stale = record("stale", MemoryType::SemanticMemory);
        stale.timestamp = now - Duration::days(90);
        let ranked = rank_by_salience(&[stale, fresh], now, &config).unwrap();
        assert!(approx(ranked[0].1.recency, 1.0));
        assert!(ranked[1].1.recency < ranked[0].1.recency);

        // Access: the hotter record wins.
        let mut hot = record("hot", MemoryType::SemanticMemory);
        hot.access_count = 100;
        let mut cold = record("cold", MemoryType::SemanticMemory);
        cold.access_count = 1;
        let ranked = rank_by_salience(&[cold, hot], now, &config).unwrap();
        assert!(ranked[0].1.access > ranked[1].1.access);
        assert!(approx(ranked[1].1.access, 1.0 / 11.0));

        // Kind: a durable user profile outranks an unknown kind.
        let profile = record("profile", MemoryType::UserProfile);
        let unknown = record("unknown", MemoryType::Unknown);
        let ranked = rank_by_salience(&[unknown, profile], now, &config).unwrap();
        assert!(approx(ranked[0].1.kind, 1.0));
        assert!(approx(ranked[1].1.kind, 0.5));

        // No records: an empty ranking, not an error.
        let empty: Vec<(String, SalienceScore)> = Vec::new();
        assert_eq!(rank_by_salience(&[], now, &config).unwrap(), empty);
    }

    #[test]
    fn rank_breaks_ties_by_id() {
        let now = fixed_time();
        let config = SalienceConfig::default();
        let build = |id: &str| {
            let mut record = record("identical", MemoryType::SemanticMemory);
            record.id = id.to_owned();
            record
        };
        let ranked = rank_by_salience(&[build("c"), build("a"), build("b")], now, &config).unwrap();
        let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert!(approx(ranked[0].1.total, ranked[2].1.total));
    }

    #[test]
    fn serde_round_trip_and_defaults() {
        let config = SalienceConfig::default();
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["backend"], serde_json::json!("heuristic"));
        assert_eq!(json["half_life_secs"], serde_json::json!(2_592_000));
        assert_eq!(json["access_saturation"], serde_json::json!(10.0_f32));
        assert_eq!(json["weights"]["importance"], serde_json::json!(1.0_f32));
        assert_eq!(json["kind_weights"]["coding"], serde_json::json!(0.85_f32));
        let restored: SalienceConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
        // Scores round-trip too, and tolerate missing/unknown fields.
        let mut record = record("scored", MemoryType::Coding);
        record.access_count = 2;
        let score = salience_score(&record, fixed_time(), &config).unwrap();
        let json = serde_json::to_value(&score).unwrap();
        assert_eq!(json["kind"], serde_json::json!(0.85_f32));
        assert_eq!(
            serde_json::from_value::<SalienceScore>(json).unwrap(),
            score
        );
        let from_empty: SalienceScore = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(from_empty, SalienceScore::default());
        let partial: SalienceScore =
            serde_json::from_value(serde_json::json!({"total": 0.5, "ignored": 9})).unwrap();
        assert!(approx(partial.total, 0.5));
        assert!(approx(partial.recency, 0.0));
    }

    #[test]
    fn serde_tolerates_missing_and_unknown_fields() {
        let from_empty: SalienceConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(from_empty, SalienceConfig::default());
        let tolerant: SalienceConfig = serde_json::from_value(serde_json::json!({
            "backend": "ml",
            "half_life_secs": 3_600,
            "unknown_field": {"nested": true},
        }))
        .unwrap();
        assert_eq!(tolerant.backend, SalienceBackend::Ml);
        assert_eq!(tolerant.half_life_secs, 3_600);
        assert_eq!(tolerant.weights, SalienceWeights::default());
        let weights: SalienceWeights =
            serde_json::from_value(serde_json::json!({"access": 0.25})).unwrap();
        assert_eq!(weights, SalienceWeights::default().with_access(0.25));
        let kinds: KindWeights =
            serde_json::from_value(serde_json::json!({"unknown": 0.1, "extra": 1})).unwrap();
        assert_eq!(
            kinds,
            KindWeights {
                unknown: 0.1,
                ..KindWeights::default()
            }
        );
        assert_eq!(
            serde_json::to_value(SalienceBackend::Ml).unwrap(),
            serde_json::json!("ml")
        );
        let backend: SalienceBackend =
            serde_json::from_value(serde_json::json!("heuristic")).unwrap();
        assert_eq!(backend, SalienceBackend::Heuristic);
    }
}
