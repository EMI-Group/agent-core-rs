//! Cross-module integration suite for the tuning workflow.
//!
//! Exercises the public `tune` surface end to end: Python-shaped [`Case`]s are
//! loaded and split, the deterministic searches run twice to prove
//! reproducibility, the example selector respects its bound, the trainer
//! reports the score its evaluator computes, and the model seam is asserted both
//! for a usable rewrite and for an unparsable one.
//!
//! Everything runs offline and deterministically: the default evaluator is
//! model-free, the one model path uses [`ScriptedModel`], and case ids are
//! explicit so tie-breaks never depend on generated ids.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::tune::{
    optimize_prompt, Case, CaseEvaluator, CaseLoader, CaseScorer, DeterministicEvaluator,
    ExampleOptimizer, InstructionOptimizer, LabelScorer, OptimizationResult, OptimizerConfig,
    Trainer, MAX_ITERATION_NUM,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

/// The instruction template every search starts from.
const SEED: &str = "Answer the question {{question}}.";

/// A model rewrite that keeps the placeholder, covers every case term, and
/// carries all four instruction-cue categories, so it wins the search outright.
const MODEL_PROMPT: &str = "Answer the question {{question}} about alpha, beta, and gamma. \
                             First think about it, then respond with one, two, or three. \
                             You must only return the expected answer.";

/// Coerces a scripted model into the trait object the optimizers take.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// Builds a Python-field case with an explicit id.
fn case(id: &str, question: &str, answer: &str) -> Case {
    Case::new(
        BTreeMap::from([("question".to_string(), json!(question))]),
        BTreeMap::from([("answer".to_string(), json!(answer))]),
    )
    .expect("the case fields are non-empty")
    .with_case_id(id)
}

/// Four cases with single-token questions and answers.
fn cases() -> Vec<Case> {
    vec![
        case("case-1", "alpha", "one"),
        case("case-2", "beta", "two"),
        case("case-3", "gamma", "three"),
        case("case-4", "delta", "four"),
    ]
}

/// The result of the default instruction search over the first three cases.
async fn search() -> OptimizationResult {
    InstructionOptimizer::new()
        .optimize(SEED, &cases()[..3])
        .await
        .expect("the default search succeeds")
}

/// Asserts an [`OptimizationResult`] is well formed and returned a prompt.
fn assert_well_formed(result: &OptimizationResult, invariant: &str) {
    assert!(!result.best_prompt.trim().is_empty(), "{invariant}: prompt");
    assert!(result.score.is_finite(), "{invariant}: finite score");
    assert!(
        (0.0..=1.0).contains(&result.score),
        "{invariant}: score {} is out of range",
        result.score
    );
}

#[tokio::test]
async fn loader_split_and_instruction_search_are_deterministic() {
    let loader = CaseLoader::new(cases());
    assert_eq!(loader.size(), 4);
    let (train, val) = loader.split(0.75).expect("0.75 is a valid ratio");
    assert_eq!(train.size(), 3);
    assert_eq!(val.size(), 1);
    assert_eq!(
        train.get(0).map(|case| case.case_id.as_str()),
        Some("case-1")
    );

    let first = InstructionOptimizer::new()
        .optimize(SEED, train.cases())
        .await
        .expect("the first search succeeds");
    let second = InstructionOptimizer::new()
        .optimize(SEED, train.cases())
        .await
        .expect("the second search succeeds");

    assert_eq!(first, second, "a repeated search is byte-identical");
    assert_well_formed(&first, "instruction search");
    assert!(
        first.best_prompt.contains("{{question}}"),
        "the seed placeholder must survive, got {:?}",
        first.best_prompt
    );
}

#[tokio::test]
async fn example_selection_bounds_examples_and_is_a_no_op_at_zero() {
    let cases = cases();

    let selected = ExampleOptimizer::new()
        .optimize(SEED, &cases, 2)
        .await
        .expect("two examples is inside the bound");
    assert_well_formed(&selected, "example selection");
    assert!(
        selected.best_prompt.contains(SEED.trim()),
        "the seed text must lead the prompt, got {:?}",
        selected.best_prompt
    );
    let answers = selected.best_prompt.matches("[expected answer]").count();
    assert!(answers <= 2, "at most two example blocks, got {answers}");
    assert!(answers >= 1, "the search should add at least one example");

    let none = ExampleOptimizer::new()
        .optimize(SEED, &cases, 0)
        .await
        .expect("zero examples is a documented no-op");
    assert_eq!(none.best_prompt, SEED.trim());
    assert_eq!(none.iterations, 0);
}

#[tokio::test]
async fn the_trainer_reports_its_evaluator_score_and_early_stops() {
    let loader = CaseLoader::new(cases());
    let (train, val) = loader.split(0.75).expect("0.75 is a valid ratio");
    let evaluator: Arc<dyn CaseEvaluator> = Arc::new(DeterministicEvaluator::new(SEED));
    let trainer = Trainer::new().with_evaluator(Arc::clone(&evaluator));

    let result = trainer
        .train(SEED, train.cases(), val.cases())
        .await
        .expect("the trainer succeeds");
    assert_well_formed(&result, "trainer");
    assert!(result.iterations <= 3, "the default budget is three rounds");

    let (reported, evaluated) = trainer
        .evaluate(&result.best_prompt, val.cases())
        .await
        .expect("the same evaluator scores the validation split");
    assert_eq!(evaluated.len(), val.size());
    assert!(
        (reported - result.score).abs() < 1e-9,
        "trainer score {} differs from the evaluated score {reported}",
        result.score
    );
    let direct = DeterministicEvaluator::new(SEED)
        .score_cases(&result.best_prompt, val.cases())
        .await
        .expect("a fresh deterministic evaluator agrees")
        .0;
    assert!((direct - result.score).abs() < 1e-9);

    let stopped = Trainer::new()
        .with_evaluator(evaluator)
        .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
        .train(SEED, train.cases(), val.cases())
        .await
        .expect("a satisfied threshold stops on the first check");
    assert_eq!(stopped.iterations, 0);
    assert_eq!(stopped.best_prompt, SEED.trim());
    assert!(stopped.score >= 0.0);
}

#[tokio::test]
async fn the_instruction_search_accepts_a_model_rewrite() {
    let model = Arc::new(ScriptedModel::new([ScriptedTurn::text(format!(
        "<PROMPT_OPTIMIZED>\n{MODEL_PROMPT}\n</PROMPT_OPTIMIZED>"
    ))]));
    let result = InstructionOptimizer::new()
        .with_model(dynamic(&model))
        .with_config(OptimizerConfig::default().with_iterations(1))
        .optimize(SEED, &cases()[..3])
        .await
        .expect("the model-assisted search succeeds");

    assert_eq!(model.call_count(), 1, "one rewrite request");
    assert_well_formed(&result, "model-assisted search");
    assert_eq!(result.best_prompt, MODEL_PROMPT);
    assert_eq!(result.iterations, 1);
}

#[tokio::test]
async fn an_unparsable_model_answer_still_completes_the_search() {
    let model = Arc::new(ScriptedModel::new([ScriptedTurn::text(
        "I could not improve this prompt.",
    )]));
    let result = InstructionOptimizer::new()
        .with_model(dynamic(&model))
        .with_config(OptimizerConfig::default().with_iterations(1))
        .optimize(SEED, &cases()[..3])
        .await
        .expect("a missing rewrite tag is not fatal");

    assert_eq!(model.call_count(), 1, "the model was consulted once");
    assert_well_formed(&result, "fallback search");
    assert!(result.best_prompt.contains("{{question}}"));
    assert!(
        (1..=3).contains(&result.iterations),
        "the deterministic bank still drives the search, got {} rounds",
        result.iterations
    );
}

#[tokio::test]
async fn the_convenience_helper_matches_the_default_optimizer() {
    let cases = cases();
    let convenience = optimize_prompt(SEED, &cases[..3])
        .await
        .expect("the helper succeeds");
    let explicit = search().await;
    assert_eq!(convenience, explicit);
}

#[tokio::test]
async fn invalid_inputs_are_validation_errors() {
    let cases = cases();

    let blank = InstructionOptimizer::new()
        .optimize("   ", &cases)
        .await
        .expect_err("a blank prompt is rejected");
    assert_eq!(blank.kind(), ErrorKind::Validation);

    let empty = InstructionOptimizer::new()
        .optimize(SEED, &[])
        .await
        .expect_err("an empty case list is rejected");
    assert_eq!(empty.kind(), ErrorKind::Validation);

    let trainer = Trainer::new()
        .train(SEED, &[], &cases)
        .await
        .expect_err("an empty train split is rejected");
    assert_eq!(trainer.kind(), ErrorKind::Validation);

    let too_many = ExampleOptimizer::new()
        .optimize(SEED, &cases, 21)
        .await
        .expect_err("the example cap is 20");
    assert_eq!(too_many.kind(), ErrorKind::Validation);

    assert!(OptimizerConfig::default().validate().is_ok());
    let iterations = OptimizerConfig::default()
        .with_iterations(MAX_ITERATION_NUM + 1)
        .validate()
        .expect_err("out-of-range iterations are rejected");
    assert_eq!(iterations.kind(), ErrorKind::Validation);
    let threshold = OptimizerConfig::default()
        .with_early_stop_score(1.5)
        .validate()
        .expect_err("an out-of-range threshold is rejected");
    assert_eq!(threshold.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn the_public_scorer_and_evaluator_expose_their_bands() {
    let case = case("case-1", "capital of France?", "Paris");
    let scorer = LabelScorer::new();
    let score = |answer: &str| {
        scorer
            .score(
                &case,
                &BTreeMap::from([("answer".to_string(), json!(answer))]),
            )
            .0
    };

    assert!((score("Paris") - 1.0).abs() < 1e-9, "exact match");
    assert!(
        (score("paris") - 0.9).abs() < 1e-9,
        "case-insensitive match"
    );
    assert!(
        (score("Paris, the capital") - 0.75).abs() < 1e-9,
        "substring match"
    );
    assert!(score("Lyon") < 1e-9, "no shared tokens scores zero");

    let evaluator = DeterministicEvaluator::new(SEED);
    let (mean, evaluated) = evaluator
        .score_cases(SEED, std::slice::from_ref(&case))
        .await
        .expect("the deterministic evaluator scores");
    assert_eq!(evaluated.len(), 1);
    assert!((0.0..=1.0).contains(&mean));

    let blank = evaluator.score_parts("   ", &case);
    assert_eq!(blank.total, 0.0);
    let blank_case = CaseEvaluator::evaluate(&evaluator, "   ", &case)
        .await
        .expect("a blank candidate is scored, not rejected");
    assert_eq!(blank_case.score, 0.0);
}
