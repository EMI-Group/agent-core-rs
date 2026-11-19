//! Updaters: the conversion point from evolution evidence to update maps.
//!
//! Ports Python `openjiuwen.agent_evolving.updater`. An updater owns no
//! optimization algorithm and no persistence: it feeds trajectories and signals
//! to a [`TextualOptimizer`] and returns the [`Updates`] a caller applies through
//! [`crate::updater::execute_updates`].
//!
//! - [`Updater`] — the routing protocol (`bind`, `requires_forward_data`,
//!   `process`, `update`, `get_state`, `load_state`).
//! - [`SingleDimUpdater`] — one optimizer for the whole agent: write the
//!   trajectories, run one `backward` over the batch, return `step()`.
//! - [`MultiDimUpdater`] — one optimizer per [`UpdateDomain`], with deterministic
//!   ordering: domains are applied in [`UpdateDomain::ALL`] order (the
//!   `BTreeMap` key order of its domain map), each domain sees the trajectories in
//!   input order, and operators and targets keep the order the optimizer wrote
//!   them in. A domain with no attributed signal is never fed or stepped, and
//!   signals for a domain without an optimizer are an error rather than a silent
//!   drop.
//!
//! [`update`](Updater::update) is the offline adapter over
//! [`process`](Updater::process): it converts evaluated cases to signals with
//! [`signals_from_cases`] and delegates. `bind` returning `0` is the Python
//! soft-exit signal for a caller ("nothing optimizable, return the agent
//! unchanged"), not an error.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};

use crate::dataset::EvaluatedCase;
use crate::signal::{signals_from_evaluated_cases, EvolutionSignal, EvolutionTarget};
use crate::trajectory::Trajectory;

use super::apply::OperatorMap;
use super::types::{UpdateDomain, Updates};

/// The configuration a `process`/`update` call carries (Python's `config` dict).
///
/// Python passes a free-form dict; the typed fields below are the ones this crate
/// reads, and [`Self::extra`] keeps optimizer-specific keys without inventing
/// generic ones.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpdateConfig {
    /// Cases scoring at or above this threshold are skipped by
    /// [`Updater::update`]; `None` keeps every case.
    pub score_threshold: Option<f64>,
    /// Optimizer-specific configuration, keys sorted.
    pub extra: BTreeMap<String, Value>,
}

impl UpdateConfig {
    /// Creates an empty configuration (no threshold, no extra keys).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the offline score threshold.
    #[must_use]
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    /// Adds one optimizer-specific key.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Reads one optimizer-specific key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.extra.get(key)
    }
}

/// Converts evaluated cases into evolution signals.
///
/// Python: `openjiuwen.agent_evolving.signal.from_eval.from_evaluated_cases`.
/// Cases scoring at or above `score_threshold` are skipped (the comparison is
/// inclusive), `None` keeps every case, a zero score is reported as `low_score`
/// and every other score as `evaluated`, and the input order is preserved. This
/// reuses [`signals_from_evaluated_cases`], so the result is also capped at its
/// `MAX_SIGNALS_PER_INPUT` bound.
///
/// # Errors
///
/// [`Error::validation`] when `score_threshold` is not finite.
pub fn signals_from_cases(
    cases: &[EvaluatedCase],
    score_threshold: Option<f64>,
) -> Result<Vec<EvolutionSignal>> {
    let threshold = match score_threshold {
        Some(threshold) if !threshold.is_finite() => {
            return Err(Error::validation(format!(
                "UpdateConfig.score_threshold must be finite, got {threshold}"
            )))
        }
        Some(threshold) => threshold,
        None => f64::INFINITY,
    };
    Ok(signals_from_evaluated_cases(cases, threshold))
}

/// The conversion protocol from trajectories and signals to an update map.
///
/// Python: `openjiuwen.agent_evolving.updater.protocol.Updater`. Rust needs no
/// duck typing, so every method is explicit; [`Self::update`] has the same
/// evaluated-case adapter body Python gives both of its updaters.
#[async_trait]
pub trait Updater: Send + Sync {
    /// Binds the optimizable operators and returns how many were bound.
    ///
    /// Returning `0` is the documented soft exit: the caller logs and returns the
    /// agent untrained. `targets` narrows the bound operator targets when given.
    fn bind(&mut self, operators: &OperatorMap, targets: Option<&[String]>) -> usize;

    /// Whether the caller must run a forward pass over the training cases first.
    ///
    /// Defaults to `false` for black-box optimizers that generate, execute and
    /// evaluate internally.
    fn requires_forward_data(&self) -> bool {
        false
    }

    /// Signal-first entry: consume trajectories and signals, return the updates.
    async fn process(
        &mut self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
        config: &UpdateConfig,
    ) -> Result<Updates>;

    /// Offline adapter: converts `evaluated_cases` to signals, then [`Self::process`].
    async fn update(
        &mut self,
        trajectories: &[Trajectory],
        evaluated_cases: &[EvaluatedCase],
        config: &UpdateConfig,
    ) -> Result<Updates> {
        let signals = signals_from_cases(evaluated_cases, config.score_threshold)?;
        self.process(trajectories, &signals, config).await
    }

    /// The checkpointable state of the updater.
    fn get_state(&self) -> Value;

    /// Restores state produced by [`Self::get_state`].
    fn load_state(&mut self, state: &Value) -> Result<()>;
}

/// The optimizer contract an [`Updater`] drives (Python `BaseOptimizer`).
///
/// The four methods with no default are the load-bearing ones: an implementation
/// must not turn them into no-ops. In particular [`Self::backward`] on an
/// optimizer that was never bound is an [`Error::validation`], not a silent
/// success.
#[async_trait]
pub trait TextualOptimizer: Send + Sync {
    /// Binds the optimizable operators and returns how many were bound.
    ///
    /// `0` is the Python soft-exit value and is not an error.
    fn bind(&mut self, operators: &OperatorMap, targets: Option<&[String]>) -> usize;

    /// Records one trajectory for the next [`Self::backward`].
    fn add_trajectory(&mut self, trajectory: &Trajectory);

    /// Accumulates textual gradients from `signals`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when called before [`Self::bind`].
    async fn backward(&mut self, signals: &[EvolutionSignal]) -> Result<()>;

    /// Projects the accumulated gradients into the update map to apply.
    fn step(&mut self) -> Updates;

    /// Whether the caller must run a forward pass over the training cases first.
    fn requires_forward_data(&self) -> bool {
        false
    }

    /// The checkpointable state of the optimizer.
    fn state(&self) -> Value;

    /// Restores state produced by [`Self::state`].
    fn load_state(&mut self, state: &Value) -> Result<()>;
}

#[async_trait]
impl<T: TextualOptimizer + ?Sized> TextualOptimizer for Box<T> {
    fn bind(&mut self, operators: &OperatorMap, targets: Option<&[String]>) -> usize {
        (**self).bind(operators, targets)
    }

    fn add_trajectory(&mut self, trajectory: &Trajectory) {
        (**self).add_trajectory(trajectory);
    }

    async fn backward(&mut self, signals: &[EvolutionSignal]) -> Result<()> {
        (**self).backward(signals).await
    }

    fn step(&mut self) -> Updates {
        (**self).step()
    }

    fn requires_forward_data(&self) -> bool {
        (**self).requires_forward_data()
    }

    fn state(&self) -> Value {
        (**self).state()
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        (**self).load_state(state)
    }
}

/// A single-dimension updater: one optimizer for the whole agent.
///
/// Python: `openjiuwen.agent_evolving.updater.single_dim.SingleDimUpdater`.
/// [`Updater::process`] feeds every trajectory in input order, runs exactly one
/// `backward` for the whole batch, and returns one `step()`.
pub struct SingleDimUpdater<O: TextualOptimizer> {
    optimizer: O,
}

impl<O: TextualOptimizer> SingleDimUpdater<O> {
    /// Creates an updater over `optimizer`.
    pub fn new(optimizer: O) -> Self {
        Self { optimizer }
    }

    /// The wrapped optimizer.
    #[must_use]
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// The wrapped optimizer, mutably.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: TextualOptimizer> std::fmt::Debug for SingleDimUpdater<O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SingleDimUpdater")
            .field("optimizer_state", &self.optimizer.state())
            .finish()
    }
}

#[async_trait]
impl<O: TextualOptimizer> Updater for SingleDimUpdater<O> {
    fn bind(&mut self, operators: &OperatorMap, targets: Option<&[String]>) -> usize {
        self.optimizer.bind(operators, targets)
    }

    fn requires_forward_data(&self) -> bool {
        self.optimizer.requires_forward_data()
    }

    async fn process(
        &mut self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
        _config: &UpdateConfig,
    ) -> Result<Updates> {
        for trajectory in trajectories {
            self.optimizer.add_trajectory(trajectory);
        }
        self.optimizer.backward(signals).await?;
        Ok(self.optimizer.step())
    }

    fn get_state(&self) -> Value {
        json!({ "optimizer": self.optimizer.state() })
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        let Some(object) = state.as_object() else {
            return Err(Error::validation(format!(
                "SingleDimUpdater.load_state expects a JSON object, got {state}"
            )));
        };
        let Some(optimizer) = object.get("optimizer") else {
            return Err(Error::validation(
                "SingleDimUpdater.load_state: missing \"optimizer\" entry",
            ));
        };
        self.optimizer.load_state(optimizer)
    }
}

/// A multi-dimension updater: one optimizer per [`UpdateDomain`].
///
/// Python: `openjiuwen.agent_evolving.updater.multi_dim.MultiDimUpdater`, whose
/// `bind`/`process` are abstract there and concrete here. Operators are routed to
/// a domain with [`crate::updater::UpdateTarget::domain`], and signals with their
/// [`EvolutionTarget`]: `tool` signals go to the [`UpdateDomain::Tool`] optimizer,
/// `memory` signals to [`UpdateDomain::Memory`], and everything else to
/// [`UpdateDomain::Llm`].
///
/// Domains are processed in [`UpdateDomain::ALL`] order (equivalently: the sorted
/// `BTreeMap` key order of the domain map), so a merged update map is reproducible.
#[derive(Default)]
pub struct MultiDimUpdater {
    domain_optimizers: BTreeMap<UpdateDomain, Box<dyn TextualOptimizer>>,
}

impl MultiDimUpdater {
    /// Creates an updater with no domain optimizer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers (or replaces) the optimizer of `domain`.
    #[must_use]
    pub fn with_optimizer(
        mut self,
        domain: UpdateDomain,
        optimizer: Box<dyn TextualOptimizer>,
    ) -> Self {
        self.domain_optimizers.insert(domain, optimizer);
        self
    }

    /// The registered domains, in application order.
    #[must_use]
    pub fn domains(&self) -> Vec<UpdateDomain> {
        self.domain_optimizers.keys().copied().collect()
    }

    /// The optimizer of `domain`, when one is registered.
    #[must_use]
    pub fn optimizer(&self, domain: UpdateDomain) -> Option<&dyn TextualOptimizer> {
        self.domain_optimizers
            .get(&domain)
            .map(std::convert::AsRef::as_ref)
    }

    /// The optimizer of `domain`, mutably, when one is registered.
    pub fn optimizer_mut(
        &mut self,
        domain: UpdateDomain,
    ) -> Option<&mut (dyn TextualOptimizer + '_)> {
        match self.domain_optimizers.get_mut(&domain) {
            Some(optimizer) => Some(optimizer.as_mut()),
            None => None,
        }
    }
}

/// The domain a signal is attributed to.
///
/// The rule is the signal's [`EvolutionTarget`]: `tool` and `memory` have their
/// own domain, every other target (instruction, description, body, script, skill,
/// context, other) belongs to the LLM domain.
#[must_use]
pub const fn domain_for_signal(signal: &EvolutionSignal) -> UpdateDomain {
    match signal.target() {
        EvolutionTarget::Tool => UpdateDomain::Tool,
        EvolutionTarget::Memory => UpdateDomain::Memory,
        _ => UpdateDomain::Llm,
    }
}

impl std::fmt::Debug for MultiDimUpdater {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MultiDimUpdater")
            .field("domains", &self.domains())
            .finish()
    }
}

#[async_trait]
impl Updater for MultiDimUpdater {
    /// Binds each domain optimizer to the operators that report its domain.
    ///
    /// Returns the sum over the domains, so `0` means "no operator was
    /// optimizable in any registered domain". `targets` is forwarded unchanged.
    fn bind(&mut self, operators: &OperatorMap, targets: Option<&[String]>) -> usize {
        let mut bound = 0;
        for (domain, optimizer) in &mut self.domain_optimizers {
            let mut subset = OperatorMap::new();
            for (operator_id, target) in operators {
                if target.domain(operator_id) == *domain {
                    subset.insert(operator_id.clone(), std::sync::Arc::clone(target));
                }
            }
            bound += optimizer.bind(&subset, targets);
        }
        bound
    }

    fn requires_forward_data(&self) -> bool {
        self.domain_optimizers
            .values()
            .any(TextualOptimizer::requires_forward_data)
    }

    async fn process(
        &mut self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
        _config: &UpdateConfig,
    ) -> Result<Updates> {
        let unattributed: Vec<&str> = signals
            .iter()
            .map(domain_for_signal)
            .filter(|domain| !self.domain_optimizers.contains_key(domain))
            .map(UpdateDomain::as_str)
            .collect();
        if let Some(domain) = unattributed.first() {
            return Err(Error::validation(format!(
                "MultiDimUpdater has no {domain:?} optimizer for {} signal(s)",
                unattributed.len()
            )));
        }
        let mut merged = Updates::new();
        for (domain, optimizer) in &mut self.domain_optimizers {
            let domain_signals: Vec<EvolutionSignal> = signals
                .iter()
                .filter(|signal| domain_for_signal(signal) == *domain)
                .cloned()
                .collect();
            if domain_signals.is_empty() {
                continue;
            }
            for trajectory in trajectories {
                optimizer.add_trajectory(trajectory);
            }
            optimizer.backward(&domain_signals).await?;
            merged.merge(optimizer.step())?;
        }
        Ok(merged)
    }

    fn get_state(&self) -> Value {
        let domains: serde_json::Map<String, Value> = self
            .domain_optimizers
            .iter()
            .map(|(domain, optimizer)| (domain.as_str().to_owned(), optimizer.state()))
            .collect();
        json!({ "domains": Value::Object(domains) })
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        let Some(object) = state.as_object() else {
            return Err(Error::validation(format!(
                "MultiDimUpdater.load_state expects a JSON object, got {state}"
            )));
        };
        let Some(Value::Object(domains)) = object.get("domains") else {
            return Err(Error::validation(
                "MultiDimUpdater.load_state: missing \"domains\" object",
            ));
        };
        for (name, domain_state) in domains {
            let domain = UpdateDomain::parse(name).ok_or_else(|| {
                Error::validation(format!(
                    "MultiDimUpdater.load_state: unknown domain {name:?}"
                ))
            })?;
            let Some(optimizer) = self.domain_optimizers.get_mut(&domain) else {
                return Err(Error::validation(format!(
                    "MultiDimUpdater.load_state: no {name:?} optimizer is registered"
                )));
            };
            optimizer.load_state(domain_state)?;
        }
        Ok(())
    }
}

/// A deterministic [`TextualOptimizer`] that records every call.
///
/// The default [`Self::step`] emits one `replace` update per bound operator for
/// the `description` target, carrying the recorded trajectory count, signal count
/// and step index, so a test can assert exactly what the updater fed in without a
/// model. [`Self::with_updates`] scripts the returned maps instead, and
/// [`Self::with_requires_forward_data`] exercises the caller-facing flag.
#[derive(Debug, Default)]
pub struct RecordingOptimizer {
    requires_forward_data: bool,
    bound: Vec<String>,
    targets: Option<Vec<String>>,
    trajectories: Vec<String>,
    backward_calls: Vec<Vec<EvolutionSignal>>,
    steps: usize,
    scripted: Vec<Updates>,
}

impl RecordingOptimizer {
    /// Creates an optimizer that has not been bound yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reports `requires_forward_data` instead of the default `false`.
    #[must_use]
    pub fn with_requires_forward_data(mut self, requires: bool) -> Self {
        self.requires_forward_data = requires;
        self
    }

    /// Scripts the maps [`Self::step`] returns, one per call.
    ///
    /// Once the scripted maps are exhausted, `step` returns an empty map.
    #[must_use]
    pub fn with_updates(mut self, updates: Vec<Updates>) -> Self {
        self.scripted = updates;
        self
    }

    /// The bound operator ids, in bind order.
    #[must_use]
    pub fn bound(&self) -> &[String] {
        &self.bound
    }

    /// The `targets` filter the last [`TextualOptimizer::bind`] received.
    #[must_use]
    pub fn bound_targets(&self) -> Option<&[String]> {
        self.targets.as_deref()
    }

    /// The recorded trajectory ids, in call order (duplicates kept).
    #[must_use]
    pub fn trajectory_ids(&self) -> &[String] {
        &self.trajectories
    }

    /// How many [`TextualOptimizer::backward`] calls succeeded.
    #[must_use]
    pub fn backward_calls(&self) -> usize {
        self.backward_calls.len()
    }

    /// The signals each [`TextualOptimizer::backward`] call received, in call order.
    #[must_use]
    pub fn signals(&self) -> &[Vec<EvolutionSignal>] {
        &self.backward_calls
    }

    /// How many [`TextualOptimizer::step`] calls happened.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }
}

#[async_trait]
impl TextualOptimizer for RecordingOptimizer {
    fn bind(&mut self, operators: &OperatorMap, targets: Option<&[String]>) -> usize {
        self.bound = operators.keys().cloned().collect();
        self.targets = targets.map(<[String]>::to_vec);
        self.bound.len()
    }

    fn add_trajectory(&mut self, trajectory: &Trajectory) {
        self.trajectories.push(trajectory.trajectory_id.clone());
    }

    async fn backward(&mut self, signals: &[EvolutionSignal]) -> Result<()> {
        if self.bound.is_empty() {
            return Err(Error::validation(
                "RecordingOptimizer::backward requires bind first",
            ));
        }
        self.backward_calls.push(signals.to_vec());
        Ok(())
    }

    fn step(&mut self) -> Updates {
        let index = self.steps;
        self.steps += 1;
        if !self.scripted.is_empty() {
            return self.scripted.get(index).cloned().unwrap_or_default();
        }
        let mut updates = Updates::new();
        let signals = self.backward_calls.last().map_or(0, Vec::len);
        for operator_id in &self.bound {
            updates.insert_update(
                operator_id.clone(),
                "description",
                json!({
                    "trajectories": self.trajectories.len(),
                    "signals": signals,
                    "step": index,
                }),
            );
        }
        updates
    }

    fn requires_forward_data(&self) -> bool {
        self.requires_forward_data
    }

    fn state(&self) -> Value {
        json!({
            "bound": self.bound,
            "targets": self.targets,
            "trajectories": self.trajectories.len(),
            "backward_calls": self.backward_calls.len(),
            "steps": self.steps,
        })
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        let Some(object) = state.as_object() else {
            return Err(Error::validation(format!(
                "RecordingOptimizer.load_state expects a JSON object, got {state}"
            )));
        };
        self.bound = string_list(object.get("bound"), "bound")?;
        self.targets = match object.get("targets") {
            None | Some(Value::Null) => None,
            Some(value) => Some(string_list(Some(value), "targets")?),
        };
        self.steps = match object.get("steps") {
            None => 0,
            Some(value) => value.as_u64().ok_or_else(|| {
                Error::validation(format!(
                    "RecordingOptimizer.load_state: steps must be an unsigned integer, got {value}"
                ))
            })? as usize,
        };
        self.trajectories.clear();
        self.backward_calls.clear();
        Ok(())
    }
}

/// Reads a JSON array of strings, naming `field` in the error.
fn string_list(value: Option<&Value>, field: &str) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        return Err(Error::validation(format!(
            "RecordingOptimizer.load_state: {field} must be an array of strings, got {value}"
        )));
    };
    items
        .iter()
        .map(|item| {
            item.as_str().map(str::to_owned).ok_or_else(|| {
                Error::validation(format!(
                    "RecordingOptimizer.load_state: {field} must be an array of strings, got {item}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::updater::apply::{RecordingTarget, UpdateTarget};
    use crate::updater::types::{ApplyResult, Update, UpdateMode};
    use std::sync::Arc;

    fn trajectory(id: &str) -> Trajectory {
        let mut trajectory = Trajectory::new(id).expect("trajectory");
        trajectory.outcome = crate::trajectory::TrajectoryOutcome::Success;
        trajectory
    }

    fn signal(target: EvolutionTarget) -> EvolutionSignal {
        EvolutionSignal::new("evaluated", target, "score=0.50").expect("signal")
    }

    fn operators(target: Arc<RecordingTarget>) -> OperatorMap {
        let mut operators = OperatorMap::new();
        operators.insert("op".to_owned(), target);
        operators
    }

    #[tokio::test]
    async fn single_dim_process_feeds_every_trajectory_then_steps_once() {
        let mut updater = SingleDimUpdater::new(RecordingOptimizer::new());
        assert!(!updater.requires_forward_data());
        assert_eq!(
            updater.bind(&operators(Arc::new(RecordingTarget::new())), None),
            1
        );
        let trajectories = [trajectory("t-1"), trajectory("t-2")];
        let signals = [signal(EvolutionTarget::Instruction)];
        let processed = updater
            .process(&trajectories, &signals, &UpdateConfig::new())
            .await
            .expect("process");
        assert_eq!(updater.optimizer().trajectory_ids(), ["t-1", "t-2"]);
        assert_eq!(updater.optimizer().backward_calls(), 1);
        assert_eq!(updater.optimizer().signals()[0].len(), 1);
        assert_eq!(updater.optimizer().steps(), 1);
        assert_eq!(
            processed
                .get("op", "description")
                .and_then(|update| update)
                .map(Update::payload),
            Some(&json!({"trajectories": 2, "signals": 1, "step": 0}))
        );
    }

    #[tokio::test]
    async fn single_dim_backward_before_bind_is_a_validation_error() {
        let mut updater = SingleDimUpdater::new(RecordingOptimizer::new());
        let error = updater
            .process(
                &[trajectory("t-1")],
                &[signal(EvolutionTarget::Body)],
                &UpdateConfig::new(),
            )
            .await
            .expect_err("unbound optimizer");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("requires bind first"));
    }

    #[tokio::test]
    async fn update_converts_cases_and_honours_the_score_threshold() {
        let case = crate::dataset::EvaluatedCase::new(
            crate::dataset::Case::new(
                [("question".to_owned(), json!("q"))].into_iter().collect(),
                [("answer".to_owned(), json!("a"))].into_iter().collect(),
            )
            .expect("case"),
            None,
            0.0,
            "wrong",
        )
        .expect("evaluated case");
        let better = crate::dataset::EvaluatedCase::new(case.case.clone(), None, 0.9, "right")
            .expect("evaluated case");
        let mut updater = SingleDimUpdater::new(RecordingOptimizer::new());
        updater.bind(&operators(Arc::new(RecordingTarget::new())), None);
        let config = UpdateConfig::new().with_score_threshold(0.5);
        updater
            .update(&[trajectory("t-1")], &[case.clone(), better], &config)
            .await
            .expect("update");
        let seen = &updater.optimizer().signals()[0];
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].signal_type(), "low_score");
    }

    #[tokio::test]
    async fn update_rejects_a_non_finite_threshold() {
        let mut updater = SingleDimUpdater::new(RecordingOptimizer::new());
        let error = updater
            .update(
                &[],
                &[],
                &UpdateConfig::new().with_score_threshold(f64::NAN),
            )
            .await
            .expect_err("nan threshold");
        assert!(error.to_string().contains("must be finite"));
    }

    #[test]
    fn single_dim_delegates_forward_data_flag_and_state() {
        let mut updater =
            SingleDimUpdater::new(RecordingOptimizer::new().with_requires_forward_data(true));
        assert!(updater.requires_forward_data());
        updater.bind(&operators(Arc::new(RecordingTarget::new())), None);
        let state = updater.get_state();
        assert_eq!(state["optimizer"]["bound"], json!(["op"]));
        let mut restored = SingleDimUpdater::new(RecordingOptimizer::new());
        restored.load_state(&state).expect("load state");
        assert_eq!(restored.optimizer().bound(), ["op"]);
        assert!(restored.load_state(&json!([])).is_err());
        assert!(restored.load_state(&json!({})).is_err());
    }

    #[tokio::test]
    async fn multi_dim_routes_operators_and_signals_per_domain() {
        let llm = Arc::new(RecordingTarget::new());
        let tool = Arc::new(RecordingTarget::with_domain(UpdateDomain::Tool));
        let mut operators = OperatorMap::new();
        let llm_target: Arc<dyn UpdateTarget> = Arc::<RecordingTarget>::clone(&llm);
        let tool_target: Arc<dyn UpdateTarget> = Arc::<RecordingTarget>::clone(&tool);
        operators.insert("agent/llm_1".to_owned(), llm_target);
        operators.insert("agent/tool_1".to_owned(), tool_target);
        let mut updater = MultiDimUpdater::new()
            .with_optimizer(UpdateDomain::Tool, Box::new(RecordingOptimizer::new()))
            .with_optimizer(UpdateDomain::Llm, Box::new(RecordingOptimizer::new()));
        assert_eq!(updater.domains(), [UpdateDomain::Llm, UpdateDomain::Tool]);
        assert_eq!(updater.bind(&operators, None), 2);
        let processed = updater
            .process(
                &[trajectory("t-1")],
                &[
                    signal(EvolutionTarget::Instruction),
                    signal(EvolutionTarget::Tool),
                ],
                &UpdateConfig::new(),
            )
            .await
            .expect("process");
        let keys: Vec<(&str, &str)> = processed
            .iter()
            .map(|(operator_id, target, _)| (operator_id, target))
            .collect();
        assert_eq!(
            keys,
            [
                ("agent/llm_1", "description"),
                ("agent/tool_1", "description")
            ]
        );
        let llm = updater.optimizer(UpdateDomain::Llm).expect("llm");
        assert_eq!(llm.state()["bound"], json!(["agent/llm_1"]));
    }

    #[tokio::test]
    async fn multi_dim_rejects_signals_for_an_unregistered_domain() {
        let mut operators = OperatorMap::new();
        let erased: Arc<dyn UpdateTarget> = Arc::new(RecordingTarget::new());
        operators.insert("agent/llm_1".to_owned(), erased);
        let mut updater = MultiDimUpdater::new()
            .with_optimizer(UpdateDomain::Llm, Box::new(RecordingOptimizer::new()));
        updater.bind(&operators, None);
        let error = updater
            .process(&[], &[signal(EvolutionTarget::Tool)], &UpdateConfig::new())
            .await
            .expect_err("no tool optimizer");
        assert!(error.to_string().contains("no \"tool\" optimizer"));
    }

    #[test]
    fn multi_dim_state_round_trips_domains() {
        let mut updater = MultiDimUpdater::new()
            .with_optimizer(UpdateDomain::Llm, Box::new(RecordingOptimizer::new()))
            .with_optimizer(UpdateDomain::Tool, Box::new(RecordingOptimizer::new()));
        let state = updater.get_state();
        assert_eq!(state["domains"]["llm"]["bound"], json!([]));
        assert!(updater.load_state(&state).is_ok());
        assert!(updater
            .load_state(&json!({"domains": {"audio": {}}}))
            .is_err());
        assert!(updater.load_state(&json!({})).is_err());
    }

    #[test]
    fn recording_optimizer_scripts_its_updates() {
        let mut scripted = Updates::new();
        scripted.insert_update("op", "body", json!("scripted"));
        let mut optimizer = RecordingOptimizer::new().with_updates(vec![scripted.clone()]);
        optimizer.bind(
            &operators(Arc::new(RecordingTarget::new())),
            Some(&["body".to_owned()]),
        );
        assert_eq!(
            optimizer.bound_targets(),
            Some(["body".to_owned()].as_slice())
        );
        assert_eq!(optimizer.step(), scripted);
        assert!(optimizer.step().is_empty());
        assert_eq!(optimizer.steps(), 2);
    }

    #[test]
    fn recording_optimizer_state_rejects_malformed_payloads() {
        let mut optimizer = RecordingOptimizer::new();
        assert!(optimizer.load_state(&json!({"bound": "op"})).is_err());
        assert!(optimizer.load_state(&json!({"steps": "many"})).is_err());
        optimizer
            .load_state(&json!({"bound": ["op"], "targets": null, "steps": 4}))
            .expect("load state");
        assert_eq!(optimizer.bound(), ["op"]);
        assert_eq!(optimizer.steps(), 4);
    }

    #[test]
    fn apply_mode_defaults_are_replace_state() {
        let result = ApplyResult::default();
        assert_eq!(result.mode, UpdateMode::Replace);
        assert_eq!(result.effect, crate::updater::types::UpdateEffect::State);
    }
}
