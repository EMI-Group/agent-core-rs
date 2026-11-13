//! Reward interfaces, built-in components, and the composite reward model.
//!
//! Ports `reward.py` from the Python prompt-search package. A
//! [`RewardComponent`] maps one [`Execution`] to a scalar; [`CompositeReward`]
//! combines components into a per-candidate [`RewardBreakdown`].
//!
//! Anti-reward-hacking guards applied by [`CompositeReward`]:
//!
//! - **min-correctness gate** — a candidate below the correctness floor cannot
//!   win by trading quality for latency or tokens; its reward is capped at its
//!   correctness.
//! - **drift penalty** — semantic deviation from the objective is subtracted.
//! - **hidden-set consistency** — when visible correctness far exceeds held-out
//!   correctness the candidate is overfitting and is penalized.
//!
//! Deviation from Python: `RewardComponent::score` is synchronous. Nothing in
//! this module performs I/O — correctness is read from the per-case scores the
//! caller's evaluator supplied, so no LLM judge lives inside `tune` — and a
//! synchronous trait keeps the batch evaluation deterministic and free of an
//! ambient runtime.

// `RewardComponent::name` must stay `&str` so `CustomReward` can carry a runtime
// name; clippy's `&'static str` suggestion does not fit that implementation.
#![allow(clippy::unnecessary_literal_bound)]

use std::collections::BTreeMap;

use serde_json::Value;

use super::models::{CaseResult, Execution, PromptTaskSpec, RewardBreakdown};

/// A caller-supplied scoring function, shared by [`CustomReward`].
type RewardFn = Box<dyn Fn(&Execution, &PromptTaskSpec) -> f64 + Send + Sync>;

/// How far visible correctness may exceed held-out correctness before a
/// candidate counts as overfitting.
pub const OVERFIT_MARGIN: f64 = 0.25;

/// The fraction of the overfitting gap subtracted from the reward.
pub const OVERFIT_PENALTY: f64 = 0.5;

/// Clamps `value` into `0.0..=1.0`.
///
/// A non-finite input (`NaN`, `±inf`) becomes `0.0` rather than propagating
/// through a reward, matching Python's `clamp01` for out-of-domain values while
/// staying total.
#[must_use]
pub fn clamp01(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

/// Scales `values` into `0.0..=1.0` by min-max normalization.
///
/// An empty input yields an empty vector, constant input maps to all-`1.0` (no
/// signal), and a spread below `1e-9` counts as constant. A non-finite input
/// value is treated as `0.0` before the range is computed, so the result is
/// always finite.
#[must_use]
pub fn min_max_normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let finite: Vec<f64> = values
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect();
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for value in &finite {
        low = low.min(*value);
        high = high.max(*value);
    }
    let spread = high - low;
    if spread <= 1e-9 {
        return vec![1.0; values.len()];
    }
    finite.iter().map(|value| (value - low) / spread).collect()
}

/// One reward metric, returning a scalar (ideally already in `0.0..=1.0`).
pub trait RewardComponent: Send + Sync {
    /// The component's stable name, used as its weight and report key.
    fn name(&self) -> &str;

    /// Whether this component measures correctness.
    ///
    /// The composite reward uses this to decide whether the min-correctness gate
    /// and the hidden-set overfitting check apply; at most one component should
    /// report `true`.
    fn is_correctness(&self) -> bool {
        false
    }

    /// The raw score of `execution` for `task` (higher is better).
    fn score(&self, execution: &Execution, task: &PromptTaskSpec) -> f64;
}

/// The mean grade of `results`, ignoring failed ones.
fn mean_grade(results: &[CaseResult]) -> f64 {
    let mut total = 0.0;
    let mut graded = 0usize;
    for result in results {
        if result.error.is_none() {
            total += clamp01(result.score);
            graded += 1;
        }
    }
    if graded == 0 {
        return 0.0;
    }
    total / graded as f64
}

/// Averages the per-case scores of the visible results.
///
/// Python grades with a `BaseEvaluator` (an LLM judge by default). Here the
/// grade is the `score` the caller's evaluator already put on each
/// [`CaseResult`], so correctness is shared with the rest of `tune` and needs no
/// second judge call.
#[derive(Debug, Clone, Copy, Default)]
pub struct CorrectnessReward;

impl CorrectnessReward {
    /// Creates the component.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The mean grade of `results`, ignoring results that carried an error.
    ///
    /// `0.0` when every result failed or the list is empty. Scores are clamped
    /// into `0.0..=1.0` so a caller-supplied outlier cannot escape the reward
    /// range.
    #[must_use]
    pub fn correctness_on(&self, results: &[CaseResult]) -> f64 {
        mean_grade(results)
    }
}

impl RewardComponent for CorrectnessReward {
    fn name(&self) -> &str {
        "correctness"
    }

    fn is_correctness(&self) -> bool {
        true
    }

    fn score(&self, execution: &Execution, _task: &PromptTaskSpec) -> f64 {
        let results: Vec<CaseResult> = execution.visible_results().into_iter().cloned().collect();
        self.correctness_on(&results)
    }
}

/// The fraction of visible cases that produced non-trivial output.
#[derive(Debug, Clone, Copy)]
pub struct CompletenessReward {
    /// Minimum trimmed output length that counts as a complete answer.
    min_chars: usize,
}

impl CompletenessReward {
    /// Creates the component with the Python default of one character.
    #[must_use]
    pub fn new() -> Self {
        Self { min_chars: 1 }
    }

    /// Sets the minimum trimmed output length; clamped to at least `1`.
    #[must_use]
    pub fn with_min_chars(mut self, min_chars: usize) -> Self {
        self.min_chars = min_chars.max(1);
        self
    }

    /// The configured minimum trimmed output length.
    #[must_use]
    pub fn min_chars(&self) -> usize {
        self.min_chars
    }
}

impl Default for CompletenessReward {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardComponent for CompletenessReward {
    fn name(&self) -> &str {
        "completeness"
    }

    fn score(&self, execution: &Execution, _task: &PromptTaskSpec) -> f64 {
        let results = execution.visible_results();
        if results.is_empty() {
            return 0.0;
        }
        let ok = results
            .iter()
            .filter(|result| {
                result.error.is_none() && result.output.trim().chars().count() >= self.min_chars
            })
            .count();
        ok as f64 / results.len() as f64
    }
}

/// Rewards lower latency; `half_life_s` is where the score reaches `0.5`.
#[derive(Debug, Clone, Copy)]
pub struct LatencyReward {
    /// The latency at which the score is `0.5`, in seconds.
    half_life_s: f64,
}

impl LatencyReward {
    /// Creates the component with `half_life_s` clamped to at least `1e-3`.
    #[must_use]
    pub fn new(half_life_s: f64) -> Self {
        Self {
            half_life_s: half_life_s.max(1e-3),
        }
    }

    /// The configured half-life in seconds.
    #[must_use]
    pub fn half_life_s(&self) -> f64 {
        self.half_life_s
    }
}

impl Default for LatencyReward {
    fn default() -> Self {
        Self::new(5.0)
    }
}

impl RewardComponent for LatencyReward {
    fn name(&self) -> &str {
        "latency"
    }

    fn score(&self, execution: &Execution, _task: &PromptTaskSpec) -> f64 {
        1.0 / (1.0 + execution.latency_s() / self.half_life_s)
    }
}

/// Rewards fewer tokens; `half_life_tokens` is where the score reaches `0.5`.
#[derive(Debug, Clone, Copy)]
pub struct TokenUsageReward {
    /// The token count at which the score is `0.5`.
    half_life_tokens: f64,
}

impl TokenUsageReward {
    /// Creates the component with `half_life_tokens` clamped to at least `1.0`.
    #[must_use]
    pub fn new(half_life_tokens: f64) -> Self {
        Self {
            half_life_tokens: half_life_tokens.max(1.0),
        }
    }

    /// The configured half-life in tokens.
    #[must_use]
    pub fn half_life_tokens(&self) -> f64 {
        self.half_life_tokens
    }
}

impl Default for TokenUsageReward {
    fn default() -> Self {
        Self::new(2000.0)
    }
}

impl RewardComponent for TokenUsageReward {
    fn name(&self) -> &str {
        "token_usage"
    }

    fn score(&self, execution: &Execution, _task: &PromptTaskSpec) -> f64 {
        1.0 / (1.0 + execution.total_tokens as f64 / self.half_life_tokens)
    }
}

/// Rewards lower monetary cost, priced per million tokens.
#[derive(Debug, Clone, Copy)]
pub struct CostReward {
    /// Currency per one million tokens.
    price_per_mtok: f64,
    /// The cost at which the score is `0.5`.
    half_life_cost: f64,
}

impl CostReward {
    /// Creates the component, clamping `price_per_mtok` to at least `0.0` and
    /// `half_life_cost` to at least `1e-9`.
    #[must_use]
    pub fn new(price_per_mtok: f64, half_life_cost: f64) -> Self {
        Self {
            price_per_mtok: price_per_mtok.max(0.0),
            half_life_cost: half_life_cost.max(1e-9),
        }
    }

    /// The configured price per million tokens.
    #[must_use]
    pub fn price_per_mtok(&self) -> f64 {
        self.price_per_mtok
    }

    /// The configured cost half-life.
    #[must_use]
    pub fn half_life_cost(&self) -> f64 {
        self.half_life_cost
    }
}

impl Default for CostReward {
    fn default() -> Self {
        Self::new(0.0, 0.01)
    }
}

impl RewardComponent for CostReward {
    fn name(&self) -> &str {
        "cost"
    }

    fn score(&self, execution: &Execution, _task: &PromptTaskSpec) -> f64 {
        let cost = execution.total_tokens as f64 / 1_000_000.0 * self.price_per_mtok;
        1.0 / (1.0 + cost / self.half_life_cost)
    }
}

/// The fraction of visible outputs that pass a caller-supplied validator.
///
/// Deviation from Python: a panicking validator is not caught — a panic in a
/// `Fn` closure is a bug in the caller's code, not a case outcome, so it
/// propagates instead of being silently scored as a failure.
pub struct StructuredValidationReward {
    /// Decides whether one output is structurally acceptable.
    validator: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl std::fmt::Debug for StructuredValidationReward {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StructuredValidationReward")
            .finish_non_exhaustive()
    }
}

impl StructuredValidationReward {
    /// Creates the component from a validator.
    #[must_use]
    pub fn new(validator: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        Self {
            validator: Box::new(validator),
        }
    }

    /// The preset that accepts any output parsing as JSON.
    #[must_use]
    pub fn json() -> Self {
        Self::new(|output| serde_json::from_str::<Value>(output).is_ok())
    }
}

impl RewardComponent for StructuredValidationReward {
    fn name(&self) -> &str {
        "structured_validation"
    }

    fn score(&self, execution: &Execution, _task: &PromptTaskSpec) -> f64 {
        let results = execution.visible_results();
        if results.is_empty() {
            return 0.0;
        }
        let ok = results
            .iter()
            .filter(|result| (self.validator)(result.output.as_str()))
            .count();
        ok as f64 / results.len() as f64
    }
}

/// A reward component wrapping an arbitrary caller function.
pub struct CustomReward {
    /// The component's name.
    name: String,
    /// The caller's scoring function.
    function: RewardFn,
}

impl std::fmt::Debug for CustomReward {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CustomReward")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CustomReward {
    /// Creates a component named `name` whose score is `function`, clamped into
    /// `0.0..=1.0`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        function: impl Fn(&Execution, &PromptTaskSpec) -> f64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            function: Box::new(function),
        }
    }
}

impl RewardComponent for CustomReward {
    fn name(&self) -> &str {
        &self.name
    }

    fn score(&self, execution: &Execution, task: &PromptTaskSpec) -> f64 {
        clamp01((self.function)(execution, task))
    }
}

/// The Python default component weights.
///
/// `cost` carries weight `0.0`, so a [`CompositeReward`] built from these weights
/// drops the cost component (Python parity: a zero-weight component is never
/// scored). Set a positive cost weight to make cost contribute.
#[must_use]
pub fn default_weights() -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("correctness".to_owned(), 1.0),
        ("completeness".to_owned(), 0.3),
        ("latency".to_owned(), 0.1),
        ("token_usage".to_owned(), 0.1),
        ("cost".to_owned(), 0.0),
    ])
}

/// A weighted combination of [`RewardComponent`]s.
///
/// Only components with a positive weight are kept; each execution's score is
/// the weighted mean of the kept components, then gated, penalized for
/// overfitting and drift, and clamped into `0.0..=1.0`.
pub struct CompositeReward {
    /// The kept components, in caller order.
    components: Vec<Box<dyn RewardComponent>>,
    /// The weight of every component by name.
    weights: BTreeMap<String, f64>,
    /// The correctness floor below which the reward is capped.
    min_correctness: f64,
    /// The multiplier of the drift penalty.
    drift_penalty: f64,
    /// Whether components are min-max normalized across the batch.
    normalize: bool,
    /// The component name whose value is used as `correctness`.
    correctness_name: String,
}

impl std::fmt::Debug for CompositeReward {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompositeReward")
            .field("components", &self.component_names())
            .field("weights", &self.weights)
            .field("min_correctness", &self.min_correctness)
            .field("drift_penalty", &self.drift_penalty)
            .field("normalize", &self.normalize)
            .field("correctness_name", &self.correctness_name)
            .finish()
    }
}

impl CompositeReward {
    /// Creates a composite reward, dropping every component whose weight is not
    /// strictly positive.
    #[must_use]
    pub fn new(components: Vec<Box<dyn RewardComponent>>, weights: BTreeMap<String, f64>) -> Self {
        let components: Vec<Box<dyn RewardComponent>> = components
            .into_iter()
            .filter(|component| weights.get(component.name()).copied().unwrap_or(0.0) > 0.0)
            .collect();
        Self {
            components,
            weights,
            min_correctness: 0.5,
            drift_penalty: 0.5,
            normalize: false,
            correctness_name: "correctness".to_owned(),
        }
    }

    /// The Python default components: correctness, completeness, latency, token
    /// usage, and cost.
    #[must_use]
    pub fn default_components() -> Vec<Box<dyn RewardComponent>> {
        vec![
            Box::new(CorrectnessReward::new()),
            Box::new(CompletenessReward::new()),
            Box::new(LatencyReward::default()),
            Box::new(TokenUsageReward::default()),
            Box::new(CostReward::default()),
        ]
    }

    /// Sets the correctness floor below which the reward is capped.
    #[must_use]
    pub fn with_min_correctness(mut self, min_correctness: f64) -> Self {
        self.min_correctness = clamp01(min_correctness);
        self
    }

    /// Sets the drift-penalty multiplier (clamped to at least `0.0`).
    #[must_use]
    pub fn with_drift_penalty(mut self, drift_penalty: f64) -> Self {
        self.drift_penalty = if drift_penalty.is_finite() {
            drift_penalty.max(0.0)
        } else {
            0.0
        };
        self
    }

    /// Enables or disables cross-candidate min-max normalization.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets the component name whose value is reported as `correctness`.
    #[must_use]
    pub fn with_correctness_name(mut self, correctness_name: impl Into<String>) -> Self {
        self.correctness_name = correctness_name.into();
        self
    }

    /// The kept component names, in order.
    #[must_use]
    pub fn component_names(&self) -> Vec<&str> {
        self.components
            .iter()
            .map(|component| component.name())
            .collect()
    }

    /// The configured weights.
    #[must_use]
    pub fn weights(&self) -> &BTreeMap<String, f64> {
        &self.weights
    }

    /// The configured correctness floor.
    #[must_use]
    pub fn min_correctness(&self) -> f64 {
        self.min_correctness
    }

    /// The configured drift-penalty multiplier.
    #[must_use]
    pub fn drift_penalty(&self) -> f64 {
        self.drift_penalty
    }

    /// Whether cross-candidate normalization is enabled.
    #[must_use]
    pub fn normalize(&self) -> bool {
        self.normalize
    }

    /// The component name reported as `correctness`.
    #[must_use]
    pub fn correctness_name(&self) -> &str {
        &self.correctness_name
    }

    /// Scores every execution of one batch.
    ///
    /// The order of operations is fixed: raw per-component scores, optional
    /// min-max normalization across the batch, the weighted mean, the
    /// min-correctness gate, the overfitting penalty, the drift penalty, the
    /// execution-error override, and a final clamp. `drift_scores` maps
    /// candidate id to semantic deviation. An empty batch yields an empty
    /// vector.
    #[must_use]
    pub fn evaluate(
        &self,
        executions: &[Execution],
        task: &PromptTaskSpec,
        drift_scores: &BTreeMap<String, f64>,
    ) -> Vec<RewardBreakdown> {
        if executions.is_empty() {
            return Vec::new();
        }

        let mut raw: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for component in &self.components {
            let values = executions
                .iter()
                .map(|execution| component.score(execution, task))
                .collect();
            raw.insert(component.name().to_owned(), values);
        }
        let normalized: BTreeMap<String, Vec<f64>> = raw
            .iter()
            .map(|(name, values)| {
                let scaled = if self.normalize {
                    min_max_normalize(values)
                } else {
                    values.clone()
                };
                (name.clone(), scaled)
            })
            .collect();

        let mut total_weight = 0.0;
        for component in &self.components {
            total_weight += self.weights.get(component.name()).copied().unwrap_or(0.0);
        }
        if total_weight <= 0.0 {
            total_weight = 1.0;
        }

        let hidden_correctness = self.hidden_correctness(executions, task);
        let has_correctness = self
            .components
            .iter()
            .any(|component| component.is_correctness());

        let mut breakdowns = Vec::with_capacity(executions.len());
        for (index, execution) in executions.iter().enumerate() {
            breakdowns.push(self.breakdown_for(
                index,
                execution,
                &raw,
                &normalized,
                total_weight,
                &hidden_correctness,
                drift_scores,
                has_correctness,
            ));
        }
        breakdowns
    }

    /// Scores one execution of the batch, applying the guards in order.
    fn breakdown_for(
        &self,
        index: usize,
        execution: &Execution,
        raw: &BTreeMap<String, Vec<f64>>,
        normalized: &BTreeMap<String, Vec<f64>>,
        total_weight: f64,
        hidden_correctness: &BTreeMap<String, f64>,
        drift_scores: &BTreeMap<String, f64>,
        has_correctness: bool,
    ) -> RewardBreakdown {
        let mut components = BTreeMap::new();
        let mut raw_components = BTreeMap::new();
        for component in &self.components {
            let name = component.name();
            components.insert(
                name.to_owned(),
                normalized.get(name).map_or(0.0, |values| values[index]),
            );
            raw_components.insert(
                name.to_owned(),
                raw.get(name).map_or(0.0, |values| values[index]),
            );
        }

        let mut weighted = 0.0;
        for component in &self.components {
            let name = component.name();
            weighted += self.weights.get(name).copied().unwrap_or(0.0)
                * components.get(name).copied().unwrap_or(0.0);
        }
        let mut score = weighted / total_weight;

        let correctness = raw_components
            .get(&self.correctness_name)
            .copied()
            .unwrap_or(1.0);
        let mut notes = Vec::new();
        let mut gated = false;

        if has_correctness && correctness < self.min_correctness {
            gated = true;
            score = score.min(correctness);
            notes.push(format!(
                "correctness {correctness:.2} < gate {:.2}: reward capped",
                self.min_correctness
            ));
        }

        if let Some(hidden) = hidden_correctness
            .get(&execution.candidate.candidate_id)
            .copied()
        {
            if correctness - hidden > OVERFIT_MARGIN {
                let gap = correctness - hidden;
                let penalty = OVERFIT_PENALTY * gap;
                score = (score - penalty).max(0.0);
                notes.push(format!(
                    "overfitting: visible {correctness:.2} vs hidden {hidden:.2} (-{penalty:.2})"
                ));
            }
        }

        let drift = clamp01(
            drift_scores
                .get(&execution.candidate.candidate_id)
                .copied()
                .unwrap_or(0.0),
        );
        let drift_applied = self.drift_penalty * drift;
        if drift_applied > 0.0 {
            score = (score - drift_applied).max(0.0);
            notes.push(format!("drift {drift:.2} penalty -{drift_applied:.2}"));
        }

        if let Some(error) = &execution.error {
            score = 0.0;
            notes.push(format!("execution error: {error}"));
        }

        RewardBreakdown {
            score: clamp01(score),
            components,
            raw_components,
            correctness,
            drift,
            drift_penalty_applied: drift_applied,
            gated,
            notes,
        }
    }

    /// The held-out correctness of every execution, keyed by candidate id.
    ///
    /// Empty when there is no correctness component or the task declares no
    /// hidden case (Python parity).
    fn hidden_correctness(
        &self,
        executions: &[Execution],
        task: &PromptTaskSpec,
    ) -> BTreeMap<String, f64> {
        if !self
            .components
            .iter()
            .any(|component| component.is_correctness())
            || task.hidden_cases().is_empty()
        {
            return BTreeMap::new();
        }
        executions
            .iter()
            .map(|execution| {
                let results: Vec<CaseResult> =
                    execution.hidden_results().into_iter().cloned().collect();
                (
                    execution.candidate.candidate_id.clone(),
                    mean_grade(&results),
                )
            })
            .collect()
    }
}

impl Default for CompositeReward {
    fn default() -> Self {
        Self::new(Self::default_components(), default_weights())
    }
}

#[cfg(test)]
#[path = "reward_tests.rs"]
mod tests;
