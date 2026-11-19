//! [`ExampleOptimizer`]: greedy few-shot example selection.
//!
//! Ports `openjiuwen/dev_tools/tune/optimizer/example_optimizer.py`. Python first
//! samples a few cases (preferring the ones that scored zero), optionally asks a
//! model which of them are most representative, and appends the rendered examples
//! to the prompt.
//!
//! The port keeps the two artifacts — the example block format and the model's
//! fenced `list` index answer — and replaces the sampling with a measurable
//! greedy search:
//!
//! 1. Score the seed prompt.
//! 2. When a model is attached, ask it for a fenced `list` block of case indices.
//!    The list is validated (integers, in range, unique, at most the target
//!    count); an invalid or unparsable list logs and falls back to an empty
//!    proposal, so greedy selection starts from scratch.
//! 3. Greedily append the example that maximises the evaluator score, one at a
//!    time, until `num_examples` examples (or every case) are in the prompt, and
//!    stop after [`OptimizerConfig::iterations`] rounds at the latest, like the
//!    instruction and joint optimizers. A round proposes one candidate per
//!    still-unselected case, so ranking them all on the whole case set would cost
//!    a quadratic number of evaluations; each round ranks its candidates on a
//!    deterministic, evenly spaced subsample of at most
//!    [`MAX_SCORED_CASES_PER_ROUND`] cases, and the winning prompt is re-scored on
//!    the whole set so the reported score stays the full-case mean. Ties go to
//!    the smaller `case_id`.
//! 4. Return the best-scoring prompt seen, which is the seed when no addition
//!    helped.
//!
//! The rendered block matches Python's `convert_cases_to_examples` exactly:
//!
//! ```text
//! example 1:
//! [question]: question:2 + 2?
//! [expected answer]: answer:4
//! ```
//!
//! `num_examples == 0` is a documented no-op that returns the prompt unchanged
//! with its measured score; a count above [`MAX_EXAMPLE_NUM`] is a validation
//! error.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;

use crate::tune::base::Case;
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, dict_line, ensure_placeholders, is_better, score_candidates,
    scores_equal, validate_inputs,
};
use crate::tune::optimizer::{OptimizationResult, MAX_EXAMPLE_NUM, MIN_EXAMPLE_NUM};
use crate::util::extract_list_fence;

/// The system prompt of the example-selection request.
const SELECTION_SYSTEM_PROMPT: &str = "\
You are a prompt-optimisation expert. Choose the most representative examples from the given \
set. Answer with exactly one ```list code block holding a JSON array of zero-based indices, for \
example ```list\n[0, 2]\n```. Choose the requested number of indices, each at most once.";

/// Renders cases as Python's few-shot example block.
///
/// One `example N:` block per case, in order, with the inputs on the
/// `[question]:` line and the label on the `[expected answer]:` line, both as
/// `key:value | …` pairs.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use openjiuwen_dev_tools::tune::{render_examples, Case};
/// # use serde_json::Value;
/// let mut inputs = BTreeMap::new();
/// inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
/// let mut label = BTreeMap::new();
/// label.insert("answer".to_owned(), Value::String("4".to_owned()));
/// let case = Case::new(inputs, label)?;
/// assert_eq!(
///     render_examples(&[case]),
///     "example 1:\n[question]: question:2 + 2?\n[expected answer]: answer:4",
/// );
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
#[must_use]
pub fn render_examples(cases: &[Case]) -> String {
    cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            format!(
                "example {}:\n[question]: {}\n[expected answer]: {}",
                index + 1,
                dict_line(&case.inputs),
                dict_line(&case.label)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// How many cases one greedy round scores every candidate on.
///
/// A round proposes one candidate per still-unselected case, so ranking the
/// candidates on the whole case set costs a quadratic number of evaluations
/// (500 cases is about 5·10⁶ evaluator calls over three rounds). The round ranks
/// its candidates on a deterministic, evenly spaced subsample of at most this
/// many cases instead, and re-scores only the winner on the whole set, which
/// keeps a page-sized case set linear. A case set that already fits is scored
/// whole, so a small configuration behaves exactly as before.
const MAX_SCORED_CASES_PER_ROUND: usize = 16;

/// A deterministic, evenly spaced subsample of `cases` holding at most `limit` cases.
///
/// A set that fits the limit is returned whole. Otherwise every `cases.len() /
/// limit`-th case is taken, starting at the first, so the sample is spread across
/// the whole set instead of being biased toward its front.
fn score_sample(cases: &[Case], limit: usize) -> Vec<Case> {
    if cases.len() <= limit || limit == 0 {
        return cases.to_vec();
    }
    (0..limit)
        .filter_map(|position| cases.get(position * cases.len() / limit).cloned())
        .collect()
}

/// Appends a rendered example block to a prompt.
///
/// An empty block leaves the prompt untouched; otherwise the block is appended
/// on its own line after the prompt's trailing whitespace is trimmed.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::tune::format_prompt;
/// assert_eq!(format_prompt("Answer.", ""), "Answer.");
/// assert_eq!(format_prompt("Answer.\n", "example 1:"), "Answer.\nexample 1:");
/// ```
#[must_use]
pub fn format_prompt(prompt: &str, examples: &str) -> String {
    if examples.trim().is_empty() {
        prompt.to_owned()
    } else {
        format!("{}\n{}", prompt.trim_end(), examples)
    }
}

/// Selects few-shot examples for a prompt out of a case set.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.ExampleOptimizer`.
pub struct ExampleOptimizer {
    /// The search budget.
    config: OptimizerConfig,
    /// The model consulted for an index proposal, when one is attached.
    model: Option<Arc<dyn Model>>,
    /// The evaluator, or the deterministic default when none was injected.
    evaluator: Option<Arc<dyn CaseEvaluator>>,
}

impl Default for ExampleOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ExampleOptimizer {
    /// Creates an optimizer with the default configuration, no model, and the
    /// deterministic evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            model: None,
            evaluator: None,
        }
    }

    /// Replaces the search budget.
    #[must_use]
    pub fn with_config(mut self, config: OptimizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Attaches a model that proposes an example selection.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the evaluator.
    #[must_use]
    pub fn with_evaluator(mut self, evaluator: Arc<dyn CaseEvaluator>) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// The search budget.
    #[must_use]
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Whether a model is attached.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// The evaluator in use for `prompt`.
    fn evaluator_for(&self, prompt: &str) -> Arc<dyn CaseEvaluator> {
        self.evaluator
            .clone()
            .unwrap_or_else(|| default_evaluator(prompt))
    }

    /// Selects up to `num_examples` examples for `prompt` from `cases`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the prompt is blank, the case list is empty,
    /// `num_examples` is outside `MIN_EXAMPLE_NUM..=MAX_EXAMPLE_NUM`, or the
    /// configuration is out of range; otherwise the evaluator's own error.
    pub async fn optimize(
        &self,
        prompt: &str,
        cases: &[Case],
        num_examples: usize,
    ) -> Result<OptimizationResult> {
        validate_inputs(prompt, cases)?;
        self.config.validate()?;
        if num_examples > MAX_EXAMPLE_NUM {
            return Err(Error::validation(format!(
                "num_examples should be between {MIN_EXAMPLE_NUM} and {MAX_EXAMPLE_NUM}, \
                 got {num_examples}"
            )));
        }

        let evaluator = self.evaluator_for(prompt);
        let seed = ensure_placeholders(prompt, prompt.trim());
        let (seed_score, _) = evaluator.score_cases(&seed, cases).await?;
        if num_examples == 0 {
            return Ok(OptimizationResult {
                best_prompt: seed,
                score: seed_score,
                iterations: 0,
                ..OptimizationResult::default()
            });
        }

        let target = num_examples.min(cases.len());
        let (mut best_prompt, mut best_score) = (seed.clone(), seed_score);
        let mut selected = self.model_proposal(&seed, cases, target).await;
        if !selected.is_empty() {
            let candidate = render_selection(&seed, cases, &selected);
            let (score, _) = evaluator.score_cases(&candidate, cases).await?;
            if is_better(score, &candidate, best_score, &best_prompt) {
                best_score = score;
                best_prompt = candidate;
            }
        }

        let mut iterations = 0;
        while selected.len() < target && iterations < self.config.iterations {
            if best_score + 1e-12 >= self.config.early_stop_score {
                break;
            }
            let mut candidates = Vec::new();
            let mut indices = Vec::new();
            for (index, case) in cases.iter().enumerate() {
                if selected.contains(&index) {
                    continue;
                }
                let mut proposal = selected.clone();
                proposal.push(index);
                let _ = case;
                candidates.push(render_selection(&seed, cases, &proposal));
                indices.push(index);
            }
            if candidates.is_empty() {
                break;
            }
            let sample = score_sample(cases, MAX_SCORED_CASES_PER_ROUND);
            let scores =
                score_candidates(&*evaluator, &candidates, &sample, self.config.num_parallel)
                    .await?;
            iterations += 1;

            let mut chosen: Option<usize> = None;
            for position in 0..candidates.len() {
                let better = match chosen {
                    None => true,
                    Some(current) => {
                        scores[position] > scores[current]
                            || (scores_equal(scores[position], scores[current])
                                && cases[indices[position]].case_id
                                    < cases[indices[current]].case_id)
                    }
                };
                if better {
                    chosen = Some(position);
                }
            }
            let Some(position) = chosen else {
                break;
            };
            selected.push(indices[position]);

            // Re-score the winning prompt on every case: the reported score stays
            // the full-case mean (comparable with the seed) and the per-case
            // results cover the whole set, not just the ranking subsample.
            let chosen_prompt = &candidates[position];
            let (score, _) = evaluator.score_cases(chosen_prompt, cases).await?;
            if is_better(score, chosen_prompt, best_score, &best_prompt) {
                best_score = score;
                best_prompt.clone_from(chosen_prompt);
            }
        }

        Ok(OptimizationResult {
            best_prompt,
            score: best_score,
            iterations,
            ..OptimizationResult::default()
        })
    }

    /// Asks the model for an index list; an invalid answer yields no proposal.
    async fn model_proposal(&self, seed: &str, cases: &[Case], target: usize) -> Vec<usize> {
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        let messages = selection_messages(seed, cases, target);
        let request = ModelRequestConfig::new(model.model_name());
        let answer = match model.invoke(&messages, &request).await {
            Ok(completion) => completion
                .message()
                .map(|message| message.text().into_owned())
                .unwrap_or_default(),
            Err(error) => {
                tracing::warn!(%error, "example selection model failed; using greedy selection");
                return Vec::new();
            }
        };
        let Some(items) = extract_list_fence(&answer) else {
            tracing::warn!(
                "example selection answer carried no ```list block; using greedy selection"
            );
            return Vec::new();
        };
        let mut indices = Vec::with_capacity(items.len());
        for item in items {
            let Some(index) = item.as_u64().and_then(|value| usize::try_from(value).ok()) else {
                tracing::warn!("example selection answer held a non-integer index; ignoring it");
                return Vec::new();
            };
            if index >= cases.len() || indices.contains(&index) {
                tracing::warn!(
                    index,
                    case_count = cases.len(),
                    "example selection answer held an out-of-range or repeated index; ignoring it"
                );
                return Vec::new();
            }
            indices.push(index);
        }
        indices.truncate(target);
        indices
    }
}

impl fmt::Debug for ExampleOptimizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExampleOptimizer")
            .field("config", &self.config)
            .field("model", &describe_model(self.model.as_ref()))
            .field("custom_evaluator", &self.evaluator.is_some())
            .finish()
    }
}

/// Renders the prompt with the examples at `selected` appended.
fn render_selection(seed: &str, cases: &[Case], selected: &[usize]) -> String {
    let examples: Vec<Case> = selected
        .iter()
        .filter_map(|index| cases.get(*index).cloned())
        .collect();
    format_prompt(seed, &render_examples(&examples))
}

/// Renders the example-selection conversation.
fn selection_messages(seed: &str, cases: &[Case], target: usize) -> Vec<BaseMessage> {
    let listing = cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            format!(
                "index: {index}\nquestion: {}\nassistant answer: {}",
                dict_line(&case.inputs),
                dict_line(&case.label)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let user = format!(
        "[task description]\n{seed}\n\nChoose the {target} most representative examples from \
         the set below.\n\n{listing}\n\n[choose the examples]"
    );
    vec![
        BaseMessage::system(SELECTION_SYSTEM_PROMPT.to_owned()),
        BaseMessage::user(user),
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::Value;

    use super::*;
    use crate::tune::base::EvaluatedCase;
    use crate::tune::evaluator::DeterministicEvaluator;

    fn case(id: &str, question: &str, answer: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(question.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(answer.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    fn cases() -> Vec<Case> {
        vec![
            case("c1", "capital of France", "Paris"),
            case("c2", "2 + 2", "four"),
            case("c3", "tallest mountain", "Everest"),
        ]
    }

    const SEED: &str = "Answer the question.";

    #[test]
    fn renders_pythons_example_format() {
        let cases = cases();
        let rendered = render_examples(&cases[..1]);
        assert_eq!(
            rendered,
            "example 1:\n[question]: question:capital of France\n[expected answer]: answer:Paris"
        );
        let two = render_examples(&cases[..2]);
        assert!(
            two.contains("example 2:\n[question]: question:2 + 2"),
            "{two}"
        );
        assert_eq!(render_examples(&[]).len(), 0);
    }

    #[test]
    fn format_prompt_appends_and_trims() {
        assert_eq!(
            format_prompt("Answer.\n", "example 1:"),
            "Answer.\nexample 1:"
        );
        assert_eq!(format_prompt("Answer.", "  "), "Answer.");
        assert_eq!(format_prompt("Answer.", ""), "Answer.");
    }

    #[tokio::test]
    async fn appends_exactly_the_requested_examples() {
        let cases = cases();
        let result = ExampleOptimizer::new()
            .optimize(SEED, &cases, 1)
            .await
            .unwrap();
        assert_eq!(result.iterations, 1);
        assert!(
            result.best_prompt.starts_with(SEED),
            "{}",
            result.best_prompt
        );
        assert!(
            result.best_prompt.contains("example 1:"),
            "{}",
            result.best_prompt
        );
        assert!(
            !result.best_prompt.contains("example 2:"),
            "one example only: {}",
            result.best_prompt
        );

        let two = ExampleOptimizer::new()
            .optimize(SEED, &cases, 2)
            .await
            .unwrap();
        assert!(
            two.best_prompt.contains("example 2:"),
            "{}",
            two.best_prompt
        );
        assert_eq!(two.iterations, 2);
    }

    #[tokio::test]
    async fn zero_examples_is_a_documented_no_op() {
        let result = ExampleOptimizer::new()
            .optimize(SEED, &cases(), 0)
            .await
            .unwrap();
        assert_eq!(result.best_prompt, SEED);
        assert_eq!(result.iterations, 0);
        let expected = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &cases())
            .await
            .unwrap()
            .0;
        assert!((result.score - expected).abs() < 1e-9);
    }

    #[tokio::test]
    async fn counts_are_capped_by_the_case_set() {
        let cases = cases();
        let result = ExampleOptimizer::new()
            .optimize(SEED, &cases, MAX_EXAMPLE_NUM)
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("example 3:"),
            "{}",
            result.best_prompt
        );
        assert!(
            !result.best_prompt.contains("example 4:"),
            "cannot exceed the case count: {}",
            result.best_prompt
        );
        assert_eq!(result.iterations, cases.len());
    }

    #[tokio::test]
    async fn selection_improves_or_keeps_the_seed_score() {
        let cases = cases();
        let baseline = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &cases)
            .await
            .unwrap()
            .0;
        let result = ExampleOptimizer::new()
            .optimize(SEED, &cases, 2)
            .await
            .unwrap();
        assert!(
            result.score + 1e-12 >= baseline,
            "{} >= {baseline}",
            result.score
        );
    }

    #[tokio::test]
    async fn repeat_runs_are_identical() {
        let optimizer = ExampleOptimizer::new();
        let first = optimizer.optimize(SEED, &cases(), 2).await.unwrap();
        let second = optimizer.optimize(SEED, &cases(), 2).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn rejects_invalid_inputs() {
        let error = ExampleOptimizer::new()
            .optimize("", &cases(), 1)
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let error = ExampleOptimizer::new()
            .optimize(SEED, &[], 1)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cases"));

        let error = ExampleOptimizer::new()
            .optimize(SEED, &cases(), MAX_EXAMPLE_NUM + 1)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("num_examples"), "{error}");

        let error = ExampleOptimizer::new()
            .with_config(OptimizerConfig::default().with_iterations(0))
            .optimize(SEED, &cases(), 1)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("iterations"));
    }

    #[tokio::test]
    async fn early_stop_before_the_first_selection() {
        let result = ExampleOptimizer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
            .optimize(SEED, &cases(), 2)
            .await
            .unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.best_prompt, SEED);
    }

    #[tokio::test]
    async fn uses_a_model_index_list() {
        let model = ScriptedModel::text("```list\n[1, 0]\n```");
        let optimizer = ExampleOptimizer::new().with_model(Arc::new(model.clone()));
        let result = optimizer.optimize(SEED, &cases(), 2).await.unwrap();
        assert_eq!(model.call_count(), 1);
        // The model's order is respected: case 1's question comes first.
        let first = result.best_prompt.find("2 + 2").expect("case 1 rendered");
        let second = result
            .best_prompt
            .find("capital of France")
            .expect("case 0 rendered");
        assert!(first < second, "{}", result.best_prompt);

        let call = model.last_call().expect("recorded call");
        assert!(call.messages[0].text().contains("```list"));
        assert!(call.messages[1].text().contains("index: 2"));
    }

    #[tokio::test]
    async fn falls_back_to_greedy_on_bad_model_lists() {
        for answer in [
            "no list here",
            "```list\n[9]\n```",
            "```list\n[0, 0]\n```",
            "```list\n[\"zero\"]\n```",
        ] {
            let model = ScriptedModel::text(answer);
            let result = ExampleOptimizer::new()
                .with_model(Arc::new(model.clone()))
                .optimize(SEED, &cases(), 2)
                .await
                .unwrap();
            assert_eq!(model.call_count(), 1, "{answer}");
            assert!(
                result.best_prompt.contains("example 2:"),
                "{answer}: {}",
                result.best_prompt
            );
        }
    }

    #[tokio::test]
    async fn model_failure_falls_back_to_greedy() {
        let model = ScriptedModel::new([openjiuwen_test_support::ScriptedTurn::fail("down")]);
        let result = ExampleOptimizer::new()
            .with_model(Arc::new(model))
            .optimize(SEED, &cases(), 1)
            .await
            .unwrap();
        assert!(result.best_prompt.contains("example 1:"));
    }

    #[tokio::test]
    async fn a_shorter_model_list_is_completed_greedily() {
        let model = ScriptedModel::text("```list\n[2]\n```");
        let result = ExampleOptimizer::new()
            .with_model(Arc::new(model))
            .optimize(SEED, &cases(), 3)
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("example 3:"),
            "{}",
            result.best_prompt
        );
        assert!(
            result.best_prompt.contains("Everest"),
            "the model proposal is kept"
        );
    }

    #[tokio::test]
    async fn injected_evaluator_drives_the_selection() {
        /// Rewards prompts that mention `Everest`.
        struct Everest;

        #[async_trait]
        impl CaseEvaluator for Everest {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = if prompt.contains("Everest") { 1.0 } else { 0.0 };
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "everest")
            }
        }

        let result = ExampleOptimizer::new()
            .with_evaluator(Arc::new(Everest))
            .optimize(SEED, &cases(), 1)
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("Everest"),
            "{}",
            result.best_prompt
        );
        assert!((result.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn debug_reports_the_model_choice() {
        assert!(format!("{:?}", ExampleOptimizer::new()).contains("model: \"none\""));
    }

    /// A case set large enough that full-case ranking would be quadratic.
    fn many_cases(count: usize) -> Vec<Case> {
        (0..count)
            .map(|index| case(&format!("c{index:03}"), &format!("q{index}"), "a"))
            .collect()
    }

    /// Counts every per-case evaluation and rewards a prompt that carries the
    /// case's question.
    struct Counting {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl CaseEvaluator for Counting {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let question = case
                .inputs
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let score = if prompt.contains(question) { 1.0 } else { 0.5 };
            EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "counted")
        }
    }

    #[tokio::test]
    async fn the_round_budget_bounds_the_evaluation_count() {
        let cases = many_cases(200);
        let evaluator = Arc::new(Counting {
            calls: AtomicUsize::new(0),
        });
        let result = ExampleOptimizer::new()
            .with_config(OptimizerConfig::default().with_iterations(2))
            .with_evaluator(evaluator.clone())
            .optimize(SEED, &cases, 20)
            .await
            .unwrap();

        // Post-conditions: the round budget bound the selection even though the
        // target was 20 examples, and the seed is still the prefix.
        assert_eq!(result.iterations, 2, "the round budget stops the search");
        assert!(
            result.best_prompt.starts_with(SEED),
            "{}",
            result.best_prompt
        );
        assert!(
            result.best_prompt.contains("example 2:"),
            "{}",
            result.best_prompt
        );
        assert!(
            !result.best_prompt.contains("example 3:"),
            "{}",
            result.best_prompt
        );

        // seed scoring (200) + two rounds of `<= 200 candidates x <= 16 sampled
        // cases` plus one full-set re-score of each winner (200): 6 984 calls.
        // Ranking every candidate on every case, as before, cost about 10^6.
        let calls = evaluator.calls.load(Ordering::Relaxed);
        assert!(
            calls <= 8_000,
            "{calls} evaluations for 200 cases, 2 rounds"
        );

        // The reported score is the winner's mean over the whole case set, not a
        // subsample estimate.
        let (expected, evaluated) = evaluator
            .score_cases(&result.best_prompt, &cases)
            .await
            .unwrap();
        assert_eq!(evaluated.len(), cases.len());
        assert!(
            (result.score - expected).abs() < 1e-12,
            "{} != {expected}",
            result.score
        );
    }

    #[test]
    fn the_round_sample_is_bounded_and_evenly_spaced() {
        let cases = many_cases(100);
        let small = score_sample(&cases[..10], MAX_SCORED_CASES_PER_ROUND);
        assert_eq!(small, cases[..10].to_vec(), "a small set is scored whole");

        let sampled = score_sample(&cases, MAX_SCORED_CASES_PER_ROUND);
        assert_eq!(sampled.len(), MAX_SCORED_CASES_PER_ROUND);
        assert_eq!(sampled[0].case_id, "c000", "the first case is sampled");
        assert!(sampled
            .windows(2)
            .all(|pair| pair[0].case_id < pair[1].case_id));
    }
}
