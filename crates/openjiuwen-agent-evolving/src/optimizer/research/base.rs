//! The deterministic textual-gradient search engine shared by all research
//! optimizers.
//!
//! Ports the `backward` / `step` shape of Python
//! `openjiuwen.agent_evolving.optimizer.base.BaseOptimizer` onto the crate's
//! [`Optimizer`](crate::optimizer::Optimizer) seam:
//!
//! - [`TextualStrategy`] — the per-domain policy: which signals to consume, which
//!   parameter they update, the deterministic rewrite, and the model prompt.
//! - [`TextualGradientOptimizer`] — the engine: [`TextualGradientOptimizer::backward`]
//!   accumulates feedback as textual gradients, [`Optimizer::propose`] turns the
//!   accumulated state into a candidate, and [`Optimizer::observe`] promotes the
//!   best candidate back into the live state.
//! - [`ProposalRecord`] — one proposal and the provenance of how it was produced.
//!
//! The engine is deterministic by default: without an injected
//! `openjiuwen_core::foundation::llm::Model` every proposal is the strategy's
//! deterministic [`TextualStrategy::rewrite`]. With a model the strategy's prompt
//! is sent once per pending parameter, and any model error, timeout or
//! unparseable answer falls back to the deterministic rewrite with the reason
//! recorded in the optimizer's provenance map — never a silent no-op.
//!
//! # Model answer convention
//!
//! A model answer must wrap the improved value in tags, by default
//! `<improved>…</improved>`; [`TextualStrategy::parse`] extracts the body and
//! returns `None` when the tag is absent or empty. The convention keeps prose
//! explanations out of the stored parameter value.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tracing::debug;

use crate::optimizer::{BestSoFar, Candidate, LlmInvokePolicy, Optimizer};
use crate::signal::{normalize_excerpt, truncate_excerpt, EvolutionSignal, SignalTracker};

use super::params::{OperatorKind, OperatorState};

/// The tag a model answer uses to delimit the improved value.
pub const IMPROVED_TAG: &str = "improved";

/// Separator between an operator id and a parameter name in a qualified
/// [`TextualStrategy::parameter_name`] key, for example `tool.search::description`.
pub const PARAMETER_SEPARATOR: &str = "::";

/// The `parameters` key reserved for candidate metadata; it is never restored
/// into an operator state.
pub const CANDIDATE_META_KEY: &str = "candidate_meta";

/// Default cap on proposals produced by one [`TextualGradientOptimizer`].
///
/// The cap is deliberately small: textual search is expensive to evaluate, so
/// the optimizer reports exhaustion instead of proposing forever.
pub const DEFAULT_MAX_PROPOSALS: usize = 8;

/// System prompt sent alongside every strategy prompt.
const SYSTEM_PROMPT: &str = "You are an evolutionary optimiser for agent instructions. \
Answer with exactly one <improved>…</improved> block containing only the improved text.";

/// The per-domain policy of a textual-gradient optimizer.
///
/// `rewrite` receives only the signals relevant to the parameter being rewritten
/// (the engine resolves [`TextualStrategy::parameter_name`] against the operator
/// and parameter it is updating), so a domain strategy never has to filter by
/// operator itself.
pub trait TextualStrategy: Send + Sync {
    /// Short strategy name used in candidate ids.
    fn name(&self) -> &str;

    /// Selects the signals this strategy consumes from the incoming set.
    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal>;

    /// The parameter `signal` updates.
    ///
    /// Return a plain parameter name (`"instruction"`) when the signal applies to
    /// every operator of [`TextualStrategy::operator_kind`], or a qualified
    /// `"{operator_id}{PARAMETER_SEPARATOR}{parameter_name}"` key when it targets
    /// one operator, as the tool optimizer does for `tool.{name}`.
    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String>;

    /// Deterministically rewrites `current` from the accumulated `gradient` and
    /// the relevant `signals`.
    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String;

    /// Builds the model prompt for `current`, `gradient` and `signals`.
    fn prompt(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String;

    /// Extracts the improved value from a model answer.
    ///
    /// The default accepts the [`IMPROVED_TAG`] convention and returns `None`
    /// when the tag is missing or its body is empty.
    fn parse(&self, response: &str) -> Option<String> {
        parse_tagged(response, IMPROVED_TAG)
    }

    /// The operator layer a plain parameter name applies to.
    ///
    /// Overridden by every strategy; the default targets the instruction layer.
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Instruction
    }
}

/// Extracts the body of `<tag>…</tag>` from `response`.
///
/// The first complete pair wins; the body is trimmed, and `None` is returned
/// when the pair is absent or the body is empty.
#[must_use]
pub fn parse_tagged(response: &str, tag: &str) -> Option<String> {
    let opening = format!("<{tag}>");
    let closing = format!("</{tag}>");
    let start = response.find(&opening)? + opening.len();
    let remainder = response.get(start..)?;
    let end = remainder.find(&closing)?;
    let body = remainder.get(..end)?.trim();
    if body.is_empty() {
        None
    } else {
        Some(body.to_owned())
    }
}

/// One proposal produced by [`TextualGradientOptimizer::propose`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProposalRecord {
    /// The proposed candidate id.
    pub candidate_id: String,
    /// The optimizer generation that produced the candidate.
    pub generation: u64,
    /// Observed score, `None` until [`Optimizer::observe`] is called.
    #[serde(default)]
    pub score: Option<f64>,
    /// How the candidate was produced (`deterministic`, `model`, or a
    /// `fallback: …` reason).
    pub provenance: String,
}

/// The deterministic textual-gradient search engine.
///
/// See the module documentation for the model-answer convention and fallback
/// behaviour. Two optimizers configured with the same strategy, operators and
/// signals propose byte-identical candidates.
#[derive(Clone)]
pub struct TextualGradientOptimizer<S: TextualStrategy> {
    strategy: S,
    operators: BTreeMap<String, OperatorState>,
    tracker: SignalTracker,
    selected: Vec<EvolutionSignal>,
    best: BestSoFar,
    best_candidate: Option<Candidate>,
    generation: u64,
    max_proposals: usize,
    model: Option<Arc<dyn Model>>,
    request: ModelRequestConfig,
    policy: LlmInvokePolicy,
    history: Vec<ProposalRecord>,
    provenance: BTreeMap<String, String>,
}

impl<S: TextualStrategy> std::fmt::Debug for TextualGradientOptimizer<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextualGradientOptimizer")
            .field("strategy", &self.strategy.name())
            .field("operators", &self.operators)
            .field("tracker", &self.tracker)
            .field("selected", &self.selected)
            .field("best", &self.best)
            .field("best_candidate", &self.best_candidate)
            .field("generation", &self.generation)
            .field("max_proposals", &self.max_proposals)
            .field("model", &self.model.is_some())
            .field("request", &self.request)
            .field("policy", &self.policy)
            .field("history", &self.history)
            .field("provenance", &self.provenance)
            .finish()
    }
}

impl<S: TextualStrategy> TextualGradientOptimizer<S> {
    /// Creates an optimizer with the default eight-proposal budget and no model.
    #[must_use]
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            operators: BTreeMap::new(),
            tracker: SignalTracker::new(),
            selected: Vec::new(),
            best: BestSoFar::new(),
            best_candidate: None,
            generation: 0,
            max_proposals: DEFAULT_MAX_PROPOSALS,
            model: None,
            request: ModelRequestConfig::default(),
            policy: LlmInvokePolicy::default(),
            history: Vec::new(),
            provenance: BTreeMap::new(),
        }
    }

    /// Registers (or replaces) one operator.
    #[must_use]
    pub fn with_operator(mut self, operator: OperatorState) -> Self {
        self.operators
            .insert(operator.operator_id.clone(), operator);
        self
    }

    /// Injects the model used for proposals.
    ///
    /// When the request configuration has no model name yet, the model's own name
    /// is adopted so the default request is valid.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        if self.request.model_name.trim().is_empty() {
            model.model_name().clone_into(&mut self.request.model_name);
        }
        self.model = Some(model);
        self
    }

    /// Replaces the proposal budget.
    ///
    /// A budget of `0` makes every `propose` call return an empty vector.
    #[must_use]
    pub fn with_max_proposals(mut self, max_proposals: usize) -> Self {
        self.max_proposals = max_proposals;
        self
    }

    /// Replaces the model request configuration.
    #[must_use]
    pub fn with_request(mut self, request: ModelRequestConfig) -> Self {
        self.request = request;
        self
    }

    /// Replaces the model retry/timeout policy.
    ///
    /// Only [`LlmInvokePolicy::attempt_timeout_secs`] is applied: proposals make
    /// a single attempt and fall back deterministically on any failure.
    #[must_use]
    pub fn with_policy(mut self, policy: LlmInvokePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The domain strategy.
    #[must_use]
    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    /// The registered operators, ordered by id.
    #[must_use]
    pub fn operators(&self) -> &BTreeMap<String, OperatorState> {
        &self.operators
    }

    /// The proposal history, in proposal order.
    #[must_use]
    pub fn history(&self) -> &[ProposalRecord] {
        &self.history
    }

    /// The signal accumulator fed by [`TextualGradientOptimizer::backward`].
    #[must_use]
    pub fn signal_tracker(&self) -> &SignalTracker {
        &self.tracker
    }

    /// How the candidate with `candidate_id` was produced, if it was proposed by
    /// this optimizer.
    #[must_use]
    pub fn provenance(&self, candidate_id: &str) -> Option<&str> {
        self.provenance.get(candidate_id).map(String::as_str)
    }

    /// The generation counter the next proposal will use.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The proposal budget.
    #[must_use]
    pub fn max_proposals(&self) -> usize {
        self.max_proposals
    }

    /// Records textual gradients for `signals` and returns how many parameters
    /// were updated.
    ///
    /// This is Python's `backward`: the strategy selects the consumable signals,
    /// the tracker deduplicates and weighs them, and every signal that names a
    /// parameter appends its excerpt to that parameter's gradient. Signals that
    /// name no parameter (or name one the optimizer does not hold) update nothing.
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` keeps parity with the Python
    /// `backward` signature so a future strategy can fail without a breaking
    /// change.
    pub fn backward(&mut self, signals: &[EvolutionSignal]) -> Result<usize> {
        let selected = self.strategy.select_signals(signals);
        if selected.is_empty() {
            return Ok(0);
        }
        self.tracker.record_all(selected.iter().cloned());
        self.selected.clone_from(&selected);
        let plain_kind = self.strategy.operator_kind();
        let mut updated = 0;
        for (operator_id, operator) in &mut self.operators {
            let names: Vec<String> = operator.parameters.keys().cloned().collect();
            for name in names {
                let mut applied = false;
                for signal in &selected {
                    let Some(key) = self.strategy.parameter_name(signal) else {
                        continue;
                    };
                    let matches = match key.split_once(PARAMETER_SEPARATOR) {
                        Some((signal_operator, signal_parameter)) => {
                            signal_operator == operator_id.as_str() && signal_parameter == name
                        }
                        None => operator.kind == plain_kind && key == name,
                    };
                    if !matches {
                        continue;
                    }
                    if let Some(parameter) = operator.parameters.get_mut(&name) {
                        parameter.apply_gradient(signal.excerpt.clone());
                        applied = true;
                    }
                }
                if applied {
                    updated += 1;
                }
            }
        }
        Ok(updated)
    }

    /// Projects the operator states to the JSON object stored in
    /// `EvolveCheckpoint.operators_state`.
    #[must_use]
    pub fn operators_state(&self) -> Value {
        let states: Map<String, Value> = self
            .operators
            .iter()
            .map(|(operator_id, operator)| (operator_id.clone(), operator.state()))
            .collect();
        Value::Object(states)
    }

    /// Restores operator states produced by
    /// [`TextualGradientOptimizer::operators_state`].
    ///
    /// Unknown operator ids are created with the strategy's
    /// [`TextualStrategy::operator_kind`], so a fresh optimizer can resume from a
    /// checkpoint without re-declaring every operator. An object wrapped in
    /// `{"operators": {...}}` is unwrapped, and the reserved
    /// [`CANDIDATE_META_KEY`] entry is skipped.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `state` (or its `operators` member) is not a
    /// JSON object, or when a per-operator state is malformed.
    pub fn restore_state(&mut self, state: &Value) -> Result<()> {
        let Some(object) = state.as_object() else {
            return Err(Error::validation(
                "TextualGradientOptimizer.restore_state expects a JSON object",
            ));
        };
        let operators = match object.get("operators") {
            Some(Value::Object(inner)) => inner,
            Some(_) => {
                return Err(Error::validation(
                    "TextualGradientOptimizer.restore_state: operators must be a JSON object",
                ));
            }
            None => object,
        };
        let kind = self.strategy.operator_kind();
        for (operator_id, operator_state) in operators {
            if operator_id == CANDIDATE_META_KEY {
                continue;
            }
            let operator = self
                .operators
                .entry(operator_id.clone())
                .or_insert_with(|| OperatorState::new(operator_id.clone(), kind));
            operator.restore(operator_state)?;
        }
        Ok(())
    }

    /// Restores the operator states embedded in one candidate's `parameters`.
    fn restore_operators(&mut self, parameters: &BTreeMap<String, Value>) -> Result<()> {
        let kind = self.strategy.operator_kind();
        for (operator_id, state) in parameters {
            if operator_id == CANDIDATE_META_KEY {
                continue;
            }
            let operator = self
                .operators
                .entry(operator_id.clone())
                .or_insert_with(|| OperatorState::new(operator_id.clone(), kind));
            operator.restore(state)?;
        }
        Ok(())
    }

    /// Proposes the next value for one operator parameter and records how it was
    /// produced.
    async fn propose_parameter(
        &self,
        operator_id: &str,
        name: &str,
        operator: &mut OperatorState,
        selected: &[EvolutionSignal],
        notes: &mut Vec<String>,
    ) {
        let relevant: Vec<EvolutionSignal> = selected
            .iter()
            .filter(|signal| {
                self.strategy
                    .parameter_name(signal)
                    .is_some_and(|key| signal_matches(&key, operator_id, name))
            })
            .cloned()
            .collect();
        let (current, gradient) = operator
            .parameters
            .get(name)
            .map(|parameter| {
                (
                    parameter.value().to_owned(),
                    parameter.gradient.clone().unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        let deterministic = self.strategy.rewrite(&current, &gradient, &relevant);
        let has_feedback = !gradient.trim().is_empty();
        let value = if let (Some(model), true) = (&self.model, has_feedback) {
            match self
                .invoke_model(model, &current, &gradient, &relevant)
                .await
            {
                Ok(Some(improved)) => {
                    notes.push(format!("{operator_id}.{name}: model"));
                    improved
                }
                Ok(None) => {
                    let reason =
                        format!("fallback: unparseable model response for {operator_id}.{name}");
                    debug!(reason = %reason, "textual optimizer fell back");
                    notes.push(reason);
                    deterministic
                }
                Err(error) => {
                    let reason = format!("fallback: model error for {operator_id}.{name}: {error}");
                    debug!(reason = %reason, "textual optimizer fell back");
                    notes.push(reason);
                    deterministic
                }
            }
        } else {
            if has_feedback {
                notes.push(format!("{operator_id}.{name}: deterministic"));
            }
            deterministic
        };
        if let Some(parameter) = operator.parameters.get_mut(name) {
            parameter.set_value(value);
        }
    }

    /// The timeout applied to one model attempt.
    fn attempt_timeout(&self) -> Option<Duration> {
        let seconds = self.policy.attempt_timeout_secs;
        if seconds.is_finite() && seconds > 0.0 {
            Some(Duration::from_secs_f64(seconds))
        } else {
            None
        }
    }

    /// Requests one model completion and parses the strategy's improvement tag.
    async fn invoke_model(
        &self,
        model: &Arc<dyn Model>,
        current: &str,
        gradient: &str,
        signals: &[EvolutionSignal],
    ) -> Result<Option<String>> {
        let prompt = self.strategy.prompt(current, gradient, signals);
        let messages = [
            BaseMessage::system(SYSTEM_PROMPT),
            BaseMessage::user(prompt),
        ];
        let mut request = self.request.clone();
        if request.model_name.trim().is_empty() {
            request.model_name = model.model_name().to_owned();
        }
        let attempt = model.invoke(&messages, &request);
        let response = match self.attempt_timeout() {
            Some(timeout) => match tokio::time::timeout(timeout, attempt).await {
                Ok(result) => result?,
                Err(_) => {
                    return Err(Error::timeout(format!(
                        "model invocation exceeded {timeout:?}"
                    )));
                }
            },
            None => attempt.await?,
        };
        let text = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        Ok(self.strategy.parse(&text))
    }
}

#[async_trait]
impl<S: TextualStrategy + 'static> Optimizer for TextualGradientOptimizer<S> {
    fn name(&self) -> &str {
        self.strategy.name()
    }

    async fn propose(&mut self, limit: usize) -> Result<Vec<Candidate>> {
        if limit == 0 || self.operators.is_empty() || self.history.len() >= self.max_proposals {
            return Ok(Vec::new());
        }
        let generation = self.generation;
        let candidate_id = format!("{}-{generation}-0", self.strategy.name());
        // Feedback keeps the order it arrived in (a stable, newest-last order for
        // the list-valued strategies); the tracker supplies dedupe and weights
        // but its weight-descending order is not used here.
        let selected: Vec<EvolutionSignal> = if self.selected.is_empty() {
            self.tracker.signals().into_iter().cloned().collect()
        } else {
            self.selected.clone()
        };
        let mut proposed = self.operators.clone();
        let mut notes: Vec<String> = Vec::new();

        for (operator_id, operator) in &mut proposed {
            let names: Vec<String> = operator.parameters.keys().cloned().collect();
            for name in names {
                self.propose_parameter(operator_id, &name, operator, &selected, &mut notes)
                    .await;
            }
        }

        let mut parameters: BTreeMap<String, Value> = proposed
            .into_iter()
            .map(|(operator_id, operator)| (operator_id, operator.state()))
            .collect();
        parameters.insert(
            CANDIDATE_META_KEY.to_owned(),
            json!({"generation": generation, "strategy": self.strategy.name()}),
        );
        let provenance = if notes.is_empty() {
            "deterministic".to_owned()
        } else {
            notes.join("; ")
        };
        let candidate = Candidate {
            id: candidate_id.clone(),
            parameters,
            score: None,
            generation,
        };
        self.provenance
            .insert(candidate_id.clone(), provenance.clone());
        self.history.push(ProposalRecord {
            candidate_id,
            generation,
            score: None,
            provenance,
        });
        self.generation = self.generation.saturating_add(1);
        Ok(vec![candidate])
    }

    fn observe(&mut self, candidate: &Candidate, score: f64) {
        if !score.is_finite() {
            return;
        }
        if let Some(record) = self
            .history
            .iter_mut()
            .find(|record| record.candidate_id == candidate.id)
        {
            record.score = Some(score);
        } else {
            let provenance = self
                .provenance
                .get(&candidate.id)
                .cloned()
                .unwrap_or_default();
            self.history.push(ProposalRecord {
                candidate_id: candidate.id.clone(),
                generation: candidate.generation,
                score: Some(score),
                provenance,
            });
        }
        if !self.best.is_improved(score) {
            // A non-best candidate leaves the live state untouched: the next
            // proposal keeps searching from the incumbent parameters.
            return;
        }
        self.best.consider(candidate.id.clone(), score);
        let mut scored = candidate.clone();
        scored.score = Some(score);
        self.best_candidate = Some(scored);
        if let Err(error) = self.restore_operators(&candidate.parameters) {
            debug!(%error, candidate = %candidate.id, "could not restore best parameters");
        }
    }

    fn best(&self) -> Option<&Candidate> {
        self.best_candidate.as_ref()
    }
}

/// Whether a strategy key addresses `parameter_name` of `operator_id`.
fn signal_matches(key: &str, operator_id: &str, parameter_name: &str) -> bool {
    match key.split_once(PARAMETER_SEPARATOR) {
        Some((key_operator, key_parameter)) => {
            key_operator == operator_id && key_parameter == parameter_name
        }
        None => key == parameter_name,
    }
}

/// The feedback texts a rewrite should consider: signal excerpts in signal order,
/// or the non-empty gradient lines when no signal was selected.
pub(super) fn feedback_texts(signals: &[EvolutionSignal], gradient: &str) -> Vec<(String, f64)> {
    if signals.is_empty() {
        return gradient
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| (line.to_owned(), 1.0))
            .collect();
    }
    signals
        .iter()
        .map(|signal| (signal.excerpt.clone(), signal.weight))
        .collect()
}

/// Renders the signal list for a model prompt, one bullet per signal.
pub(super) fn render_signals(signals: &[EvolutionSignal]) -> String {
    if signals.is_empty() {
        return "(no new feedback)".to_owned();
    }
    signals
        .iter()
        .map(|signal| {
            format!(
                "- [{} {}] {}",
                signal.target.as_str(),
                signal.category.as_str(),
                signal.excerpt
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses a parameter value holding a JSON array of strings.
///
/// Tolerant parsing keeps the optimizers usable on hand-written values: a JSON
/// array is used as-is, other JSON text is ignored, and plain text is split into
/// non-empty lines with any leading `- ` bullet marker removed.
pub(super) fn parse_text_list(value: &str) -> Vec<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(trimmed) {
        return items
            .into_iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(text.trim().to_owned()),
                Value::Object(fields) => fields
                    .get("text")
                    .and_then(Value::as_str)
                    .map(|text| text.trim().to_owned()),
                Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => None,
            })
            .filter(|item| !item.is_empty())
            .collect();
    }
    trimmed
        .lines()
        .map(|line| strip_bullet(line.trim()).to_owned())
        .filter(|line| !line.is_empty())
        .collect()
}

/// Encodes a string list as the compact JSON array stored in a parameter value.
pub(super) fn encode_text_list(items: &[String]) -> String {
    serde_json::to_string(items).unwrap_or_else(|error| {
        debug!(%error, "could not encode textual list; storing an empty list");
        "[]".to_owned()
    })
}

/// Removes one leading `- ` bullet marker, if present.
pub(super) fn strip_bullet(text: &str) -> &str {
    text.strip_prefix("- ")
        .or_else(|| text.strip_prefix('*'))
        .map_or(text, str::trim)
}

/// Normalises text to a sorted token set for similarity comparisons.
pub(super) fn normalized_tokens(text: &str) -> BTreeSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity of two token sets, in `0.0..=1.0`.
///
/// Two empty sets are treated as identical (`1.0`) so blank feedback never
/// creates a spurious cluster.
pub(super) fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let union = left.union(right).count();
    if union == 0 {
        return 1.0;
    }
    let intersection = left.intersection(right).count();
    intersection as f64 / union as f64
}

/// Whether two texts are similar enough to merge.
pub(super) fn near_duplicate(left: &str, right: &str, threshold: f64) -> bool {
    if normalize_excerpt(left) == normalize_excerpt(right) {
        return true;
    }
    jaccard(&normalized_tokens(left), &normalized_tokens(right)) >= threshold
}

/// Truncates `text` to `max_chars` characters, appending an ellipsis on a cut.
pub(super) fn shorten(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let cut = truncate_excerpt(text, max_chars.saturating_sub(1));
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::EvolutionTarget;

    #[test]
    fn tagged_parsing_accepts_first_complete_pair() {
        assert_eq!(
            parse_tagged("intro <improved>  better  </improved> tail", IMPROVED_TAG),
            Some("better".to_owned())
        );
        assert_eq!(parse_tagged("<improved></improved>", IMPROVED_TAG), None);
        assert_eq!(parse_tagged("no tags", IMPROVED_TAG), None);
        assert_eq!(parse_tagged("<improved>open", IMPROVED_TAG), None);
    }

    #[test]
    fn signal_matching_supports_qualified_and_plain_keys() {
        assert!(signal_matches("instruction", "planner", "instruction"));
        assert!(!signal_matches("instruction", "planner", "style"));
        assert!(signal_matches(
            "tool.search::description",
            "tool.search",
            "description"
        ));
        assert!(!signal_matches(
            "tool.fetch::description",
            "tool.search",
            "description"
        ));
    }

    #[test]
    fn text_list_parsing_is_tolerant() {
        assert_eq!(parse_text_list(""), Vec::<String>::new());
        assert_eq!(
            parse_text_list(r#"["one", "two"]"#),
            vec!["one".to_owned(), "two".to_owned()]
        );
        assert_eq!(
            parse_text_list("- first\n- second"),
            vec!["first".to_owned(), "second".to_owned()]
        );
        assert_eq!(parse_text_list("[1, 2]"), Vec::<String>::new());
    }

    #[test]
    fn similarity_helpers_are_deterministic() {
        assert!(near_duplicate("Retry the step", "retry the step", 0.5));
        assert!(near_duplicate(
            "retry the failed step twice",
            "retry the failed step",
            0.5
        ));
        assert!(!near_duplicate(
            "completely different",
            "another thing",
            0.5
        ));
        assert_eq!(
            jaccard(&normalized_tokens("a b"), &normalized_tokens("b c")),
            1.0 / 3.0
        );
        assert_eq!(jaccard(&BTreeSet::new(), &BTreeSet::new()), 1.0);
    }

    #[test]
    fn feedback_falls_back_to_gradient_lines() {
        let feedback = feedback_texts(&[], "first\n\nsecond");
        assert_eq!(
            feedback,
            vec![("first".to_owned(), 1.0), ("second".to_owned(), 1.0)]
        );
        let signal = EvolutionSignal::new("low_score", EvolutionTarget::Instruction, "excerpt")
            .expect("signal");
        assert_eq!(
            feedback_texts(&[signal], "ignored"),
            vec![("excerpt".to_owned(), 1.0)]
        );
    }
}
