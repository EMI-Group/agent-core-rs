//! Cross-module integration suite for the tuning workflow.
//!
//! Exercises the public `tune` surface end to end: Python-shaped [`Case`]s are
//! loaded and split, the deterministic searches run twice to prove
//! reproducibility, the example selector respects its bound, the trainer
//! reports the score its evaluator computes, and the model seam is asserted both
//! for a usable rewrite and for an unparsable one.
//!
//! The dataset layer is covered through its public seams as well: the three
//! samplers subsample a tagged [`SplitDataset`], [`BatchScheduler`] plans epochs
//! and batches (kept and dropped tails, a seeded shuffle), and
//! [`CaseLoader::split_tagged`] tags both halves so
//! [`Trainer::train_split`] reports an in-sample and a held-out score. A
//! recording prompt runner proves the held-out split is only ever scored whole,
//! never inside an instruction round.
//!
//! The run log is exercised from the search to the file: a [`JsonlRunLog`] under
//! a [`TempWorkspace`] receives one record per round, [`read_run_log`] returns
//! exactly those rounds, and the JSONL wire shape is asserted line by line
//! against [`InMemoryRunLog`] and `read_run_log_limited`.
//!
//! Everything runs offline and deterministically: the default evaluator is
//! model-free, the one model path uses [`ScriptedModel`], the run log is stamped
//! with a fixed timestamp, and case ids are explicit so tie-breaks never depend
//! on generated ids.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::tune::{
    optimize_prompt, read_run_log, read_run_log_limited, AdoptPolicy, AdoptReason, Batch,
    BatchScheduler, Case, CaseEvaluator, CaseLoader, CaseScorer, CaseSplit, DeterministicEvaluator,
    Epoch, ExampleOptimizer, InMemoryRunLog, InstructionOptimizer, JsonlRunLog, LabelScorer,
    OptimizationResult, OptimizerConfig, PromptRunner, PromptSearchOptimizer, RandomSampler,
    RunLog, RunLogRecord, RunnerEvaluator, ScriptedPromptRunner, SequentialSampler, SplitDataset,
    StratifiedSampler, Trainer, MAX_ITERATION_NUM,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn, TempWorkspace};
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

// -------------------------------------- dataset sampling and batch plans ----

/// Nine cases whose answer label cycles through three strata of three cases.
fn stratified_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for stratum in ["alpha", "beta", "gamma"] {
        for index in 1..=3 {
            cases.push(case(
                &format!("{stratum}-{index}"),
                &format!("question {stratum} {index}"),
                stratum,
            ));
        }
    }
    cases
}

/// Five cases with distinct questions and answers.
fn five_cases() -> Vec<Case> {
    (1..=5)
        .map(|index| {
            case(
                &format!("c{index}"),
                &format!("question {index}"),
                &format!("answer {index}"),
            )
        })
        .collect()
}

/// The case ids of `entries`, in order.
fn case_ids(entries: &[Case]) -> Vec<String> {
    entries.iter().map(|entry| entry.case_id.clone()).collect()
}

/// The positions a plan visits, all epochs flattened.
fn planned_positions(plan: &[Epoch]) -> Vec<Vec<usize>> {
    plan.iter()
        .map(|epoch| {
            epoch
                .batches
                .iter()
                .flat_map(|batch| batch.indices.clone())
                .collect()
        })
        .collect()
}

#[test]
fn sequential_and_random_samplers_keep_their_seeded_contract() {
    let dataset = SplitDataset::new(CaseSplit::Train, stratified_cases());
    assert_eq!(dataset.len(), 9);
    assert_eq!(dataset.split, CaseSplit::Train);

    // Sequential: the input prefix, an oversized count clamped, zero empty.
    let sequential = SequentialSampler::new(2);
    assert_eq!(sequential.count(), 2);
    assert_eq!(
        case_ids(
            &dataset
                .sample(&sequential)
                .expect("the sampler never fails")
        ),
        ["alpha-1", "alpha-2"]
    );
    assert_eq!(
        dataset
            .sample(&SequentialSampler::new(0))
            .expect("zero is a valid count"),
        Vec::new(),
        "a zero count samples nothing"
    );
    let everything = dataset
        .sample(&SequentialSampler::new(99))
        .expect("an oversized count clamps");
    assert_eq!(case_ids(&everything), case_ids(dataset.cases()));
    assert_eq!(SequentialSampler::new(2).with_count(4).count(), 4);

    // Random: a seeded permutation, so one seed repeats and another differs.
    let seeded = RandomSampler::new(5).with_count(4);
    assert_eq!((seeded.seed(), seeded.count()), (5, Some(4)));
    let first = dataset.sample(&seeded).expect("the sampler never fails");
    assert_eq!(first.len(), 4);
    assert_eq!(
        first,
        dataset.sample(&seeded).expect("the sampler never fails"),
        "one seed samples the same subset twice"
    );
    let other = dataset
        .sample(&RandomSampler::new(9).with_count(4))
        .expect("the sampler never fails");
    assert_ne!(first, other, "another seed samples another subset");
    assert_eq!(RandomSampler::new(5).count(), None);
    let full = dataset
        .sample(&RandomSampler::new(5))
        .expect("a fresh sampler keeps every case");
    let mut permuted = case_ids(&full);
    permuted.sort();
    let mut input = case_ids(dataset.cases());
    input.sort();
    assert_eq!(permuted, input, "a full sample is a permutation");
}

#[test]
fn stratified_sampler_allocates_over_labelled_strata() {
    let dataset = SplitDataset::new(CaseSplit::Train, stratified_cases());

    // The label is the default stratum key, and the count is allocated
    // proportionally over the sorted strata.
    let stratified = StratifiedSampler::new(3).with_count(6);
    let strata = stratified.strata(dataset.cases());
    assert_eq!(strata.len(), 3, "one stratum per label value");
    assert!(strata.keys().any(|key| key.contains("alpha")), "{strata:?}");
    let sample = dataset
        .sample(&stratified)
        .expect("the sampler never fails");
    assert_eq!(sample.len(), 6);
    assert_eq!(
        sample,
        dataset
            .sample(&stratified)
            .expect("the sampler never fails"),
        "one seed samples the same strata twice"
    );
    let mut per_stratum: BTreeMap<String, usize> = BTreeMap::new();
    for entry in &sample {
        let answered = entry
            .label
            .get("answer")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_owned();
        *per_stratum.entry(answered).or_default() += 1;
    }
    assert_eq!(
        per_stratum,
        BTreeMap::from([
            ("alpha".to_owned(), 2),
            ("beta".to_owned(), 2),
            ("gamma".to_owned(), 2),
        ]),
        "two of every three-case stratum"
    );

    // A caller-supplied key replaces the label grouping: grouped by question
    // length, the longer `gamma` questions form their own stratum.
    let by_length = StratifiedSampler::new(3)
        .with_stratum_key(|entry| {
            let question = entry
                .inputs
                .get("question")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            format!("len-{}", question.len())
        })
        .with_count(3);
    let regrouped = by_length.strata(dataset.cases());
    assert_eq!(regrouped.len(), 2, "{regrouped:?}");
    assert!(regrouped.keys().all(|key| key.starts_with("len-")));
}

#[test]
fn batch_scheduler_plans_epochs_and_the_kept_tail() {
    let dataset = SplitDataset::new(CaseSplit::Train, five_cases());

    // A zero batch size is rejected at construction; zero epochs at planning.
    let zero_batch = BatchScheduler::new(0).expect_err("a zero batch size is rejected");
    assert_eq!(zero_batch.kind(), ErrorKind::Validation);
    let zero_epochs = BatchScheduler::new(2)
        .expect("a valid batch size")
        .with_epochs(0);
    let unplannable = zero_epochs
        .plan(dataset.cases())
        .expect_err("zero epochs cannot be planned");
    assert_eq!(unplannable.kind(), ErrorKind::Validation);

    // Two epochs of batches of two: the five-case tail is kept by default.
    let scheduler = BatchScheduler::new(2)
        .expect("a valid batch size")
        .with_epochs(2);
    assert_eq!(scheduler.batch_size(), 2);
    assert_eq!(scheduler.epochs(), 2);
    assert_eq!(scheduler.shuffle_seed(), None);
    assert!(!scheduler.drop_last());
    assert_eq!(scheduler.batches_per_epoch(dataset.len()), 3);
    let plan = dataset.batches(&scheduler).expect("a plannable dataset");
    assert_eq!(plan.len(), 2, "one entry per epoch");
    assert_eq!(
        plan.iter().map(Epoch::len).collect::<Vec<usize>>(),
        [3, 3],
        "two full batches and the kept tail"
    );
    assert_eq!(
        plan[0]
            .batches
            .iter()
            .map(Batch::len)
            .collect::<Vec<usize>>(),
        [2, 2, 1]
    );
    let steps: Vec<usize> = plan
        .iter()
        .flat_map(|epoch| epoch.batches.iter().map(|batch| batch.step))
        .collect();
    assert_eq!(steps, [1, 2, 3, 4, 5, 6], "steps count across the plan");
    assert_eq!(
        planned_positions(&plan),
        [vec![0, 1, 2, 3, 4], vec![0, 1, 2, 3, 4]],
        "an unshuffled plan keeps the input order"
    );

    // `with_drop_last` discards the short tail; `without_drop_last` restores it.
    let dropping = BatchScheduler::new(2)
        .expect("a valid batch size")
        .with_epochs(2)
        .with_drop_last();
    assert!(dropping.drop_last());
    assert_eq!(dropping.batches_per_epoch(dataset.len()), 2);
    let dropped = dataset.batches(&dropping).expect("a plannable dataset");
    assert_eq!(
        dropped.iter().map(Epoch::len).collect::<Vec<usize>>(),
        [2, 2]
    );
    let dropped_steps: Vec<usize> = dropped
        .iter()
        .flat_map(|epoch| epoch.batches.iter().map(|batch| batch.step))
        .collect();
    assert_eq!(
        dropped_steps,
        [1, 2, 3, 4],
        "the dropped tail skips its step"
    );
    let restored = dropping.without_drop_last();
    assert!(!restored.drop_last());
    assert_eq!(
        dataset.batches(&restored).expect("a plannable dataset"),
        plan,
        "restoring the tail rebuilds the kept plan"
    );
}

#[test]
fn batch_scheduler_shuffles_every_epoch_from_its_seed() {
    let dataset = SplitDataset::new(CaseSplit::Train, five_cases());

    // A seeded shuffle is reproducible across two plans and reorders each epoch.
    let shuffled = BatchScheduler::new(2)
        .expect("a valid batch size")
        .with_epochs(2)
        .with_shuffle(7);
    let first = dataset.batches(&shuffled).expect("a plannable dataset");
    assert_eq!(
        first,
        dataset.batches(&shuffled).expect("a plannable dataset"),
        "one seed plans the same order twice"
    );
    let reordered = dataset
        .batches(
            &BatchScheduler::new(2)
                .expect("a valid batch size")
                .with_epochs(2)
                .with_shuffle(8),
        )
        .expect("a plannable dataset");
    assert_ne!(first, reordered, "another seed plans another order");
    for epoch in &first {
        let mut seen: Vec<usize> = epoch
            .batches
            .iter()
            .flat_map(|batch| batch.indices.clone())
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3, 4], "every epoch is a permutation");
    }
    assert_ne!(
        planned_positions(&first)[0],
        planned_positions(&first)[1],
        "each epoch derives its own order"
    );
}

#[test]
fn split_tagged_tags_both_halves_and_validates_the_ratio() {
    let loader = CaseLoader::new(cases());
    let (train, val) = loader.split_tagged(0.5).expect("0.5 is a valid ratio");

    assert_eq!(train.split, CaseSplit::Train);
    assert_eq!(val.split, CaseSplit::Val);
    assert_eq!((train.len(), val.len()), (2, 2));
    assert_eq!(case_ids(train.cases()), ["case-1", "case-2"]);
    assert_eq!(case_ids(val.cases()), ["case-3", "case-4"]);
    // The tag is the Python wire spelling, both as a string and as `Display`.
    assert_eq!(train.split.as_str(), "train");
    assert_eq!(val.split.as_str(), "val");
    // The halves concatenate back to the input order.
    let rejoined: Vec<String> = train
        .iter()
        .chain(val.iter())
        .map(|entry| entry.case_id.clone())
        .collect();
    assert_eq!(rejoined, case_ids(loader.cases()));

    // The cut is the untagged split's `floor(ratio * size)`, tag aside.
    let (plain_train, plain_val) = loader.split(0.5).expect("0.5 is a valid ratio");
    assert_eq!(case_ids(train.cases()), case_ids(plain_train.cases()));
    assert_eq!(case_ids(val.cases()), case_ids(plain_val.cases()));

    // Out-of-range, non-finite, and NaN ratios are validation errors.
    for ratio in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        let error = loader
            .split_tagged(ratio)
            .expect_err("an out-of-range ratio is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "ratio {ratio}");
    }

    // A one-case loader cuts at `floor(0.5 * 1) = 0`, so the only case lands on
    // the val side; at ratio 1.0 the val side is the empty one.
    let tiny = CaseLoader::new([case("only", "alpha", "one")]);
    let (tiny_train, tiny_val) = tiny.split_tagged(0.5).expect("0.5 is a valid ratio");
    assert_eq!(tiny_train.split, CaseSplit::Train);
    assert!(tiny_train.is_empty());
    assert_eq!(tiny_val.split, CaseSplit::Val);
    assert_eq!(case_ids(tiny_val.cases()), ["only"]);
    let (full_train, empty_val) = tiny.split_tagged(1.0).expect("1.0 is a valid ratio");
    assert_eq!(case_ids(full_train.cases()), ["only"]);
    assert_eq!(empty_val.split, CaseSplit::Val);
    assert!(empty_val.is_empty());
}

#[tokio::test]
async fn train_split_reports_both_scores_and_the_selection_split() {
    let loader = CaseLoader::new(cases());
    let (train, val) = loader.split_tagged(0.75).expect("0.75 is a valid ratio");
    assert_eq!((train.len(), val.len()), (3, 1));

    let result = Trainer::new()
        .train_split(SEED, &train, &val)
        .await
        .expect("the tagged trainer succeeds");
    assert_well_formed(&result, "tagged trainer");
    let train_score = result
        .train_score
        .expect("a tagged run reports its in-sample score");
    assert!(
        (0.0..=1.0).contains(&train_score),
        "in-sample {train_score}"
    );
    assert_eq!(result.val_score, Some(result.score));
    assert_eq!(result.selected_on, Some(CaseSplit::Val));

    // The in-sample score is the winner measured on the train split.
    let (in_sample, _) = DeterministicEvaluator::new(SEED)
        .score_cases(&result.best_prompt, train.cases())
        .await
        .expect("the deterministic evaluator scores");
    assert!((in_sample - train_score).abs() < 1e-12);
    let repeated = Trainer::new()
        .train_split(SEED, &train, &val)
        .await
        .expect("a repeated tagged run succeeds");
    assert_eq!(result, repeated, "two identical tagged runs agree");

    // `evaluate_split` keeps whatever tag the dataset carries.
    let trainer = Trainer::new();
    let on_val = trainer
        .evaluate_split(SEED, &val)
        .await
        .expect("a valid evaluation");
    assert_eq!(on_val.split, CaseSplit::Val);
    assert_eq!(on_val.results.len(), val.len());
    let on_train = trainer
        .evaluate_split(SEED, &train)
        .await
        .expect("a valid evaluation");
    assert_eq!(on_train.split, CaseSplit::Train);
    assert_eq!(on_train.results.len(), train.len());
    // An untagged evaluation reports no partition at all.
    let untagged = trainer
        .evaluate_traced(SEED, val.cases())
        .await
        .expect("a valid evaluation");
    assert_eq!(untagged.split, CaseSplit::Unspecified);

    // The untagged trainer leaves the provenance fields at their documented
    // `None` values, and reports the held-out score it selected with.
    let plain = trainer
        .train(SEED, train.cases(), val.cases())
        .await
        .expect("the untagged trainer succeeds");
    assert!(
        plain.train_score.is_none(),
        "an untagged run reports no in-sample score"
    );
    assert!(
        plain.selected_on.is_none(),
        "an untagged run names no selection split"
    );
    assert_eq!(plain.val_score, Some(plain.score));
}

// ---------------------------------------- held-out isolation (instrumented) ----

/// A runner that answers every case with the same mismatching answer, so every
/// candidate and every split scores alike and the trainer uses its whole budget
/// instead of early-stopping.
fn mismatching_runner() -> Arc<ScriptedPromptRunner> {
    Arc::new(
        ScriptedPromptRunner::new()
            .with_fallback(BTreeMap::from([("answer".to_owned(), json!("no match"))])),
    )
}

#[tokio::test]
async fn the_held_out_split_is_only_ever_scored_whole() {
    let train = SplitDataset::new(
        CaseSplit::Train,
        [
            case("t1", "alpha", "one"),
            case("t2", "beta", "two"),
            case("t3", "gamma", "three"),
        ],
    );
    let val = SplitDataset::new(
        CaseSplit::Val,
        [case("v1", "delta", "four"), case("v2", "epsilon", "five")],
    );

    // The public `PromptRunner` + `RunnerEvaluator` seam is the instrument: every
    // evaluation is a recorded `PromptCall` naming the case and the prompt.
    let runner = mismatching_runner();
    let recorded: Arc<dyn PromptRunner> = runner.clone();
    let evaluator: Arc<dyn CaseEvaluator> = Arc::new(RunnerEvaluator::new(recorded));
    let result = Trainer::new()
        .with_evaluator(evaluator)
        .train_split(SEED, &train, &val)
        .await
        .expect("the tagged trainer succeeds");
    assert!(result.iterations >= 1, "the budget leaves room to iterate");

    let calls = runner.calls();
    let train_ids = case_ids(train.cases());
    let val_ids = case_ids(val.cases());
    // A held-out case may only be evaluated as part of a *whole-val pass*: the
    // trainer scores the split once for the baseline and once per epoch's
    // selection check, and never hands a val case to the instruction optimizer.
    // A pass is exactly `val.len()` consecutive calls covering the val ids in
    // order, so a val case scored inside an instruction round (which would have
    // to be followed by the rest of the split) fails here.
    let mut passes = 0usize;
    let mut index = 0usize;
    while index < calls.len() {
        let call = &calls[index];
        if val_ids.contains(&call.case_id) {
            let end = index + val.len();
            assert!(
                end <= calls.len(),
                "a held-out evaluation must be a whole val pass, got {}",
                calls.len() - index
            );
            let block: Vec<String> = calls[index..end]
                .iter()
                .map(|entry| entry.case_id.clone())
                .collect();
            assert_eq!(block, val_ids, "a held-out evaluation spans the split");
            passes += 1;
            index = end;
        } else {
            assert!(
                train_ids.contains(&call.case_id),
                "only split cases are evaluated, got {}",
                call.case_id
            );
            index += 1;
        }
    }
    assert_eq!(
        passes,
        result.iterations + 1,
        "one baseline pass plus one selection pass per epoch"
    );
    let val_evaluations = calls
        .iter()
        .filter(|call| val_ids.contains(&call.case_id))
        .count();
    assert_eq!(
        val_evaluations,
        val.len() * passes,
        "no held-out case is ever scored outside a whole pass"
    );

    // The baseline scores the seed prompt on the held-out split, and the run ends
    // with the winner's in-sample measurement on the train split.
    assert_eq!(calls[0].prompt, SEED.trim());
    let last = calls.last().expect("at least one evaluation");
    assert!(
        train_ids.contains(&last.case_id),
        "the run ends on train cases"
    );
    assert_eq!(last.prompt, result.best_prompt);
}

// ------------------------------------------------------- the JSONL run log ----

/// The timestamp every logged run here stamps its records with (no clock read).
const RUN_LOG_NOW: f64 = 1_700_000_000.5;

/// Runs the same bounded prompt search with `log` attached and an adoption
/// policy in place, so every record carries a verdict.
async fn logged_search(log: Arc<Mutex<dyn RunLog>>) -> OptimizationResult {
    PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(DeterministicEvaluator::new(SEED)))
        .with_config(OptimizerConfig::default().with_iterations(2))
        .with_adopt_policy(AdoptPolicy::new())
        .with_timestamp(RUN_LOG_NOW)
        .with_run_log(log)
        .optimize(SEED, &cases())
        .await
        .expect("the bounded prompt search succeeds")
}

/// A shared file run log plus the handle the optimizer takes.
fn file_log(path: &std::path::Path) -> (Arc<Mutex<JsonlRunLog>>, Arc<Mutex<dyn RunLog>>) {
    let concrete = Arc::new(Mutex::new(
        JsonlRunLog::create(path).expect("a writable run log"),
    ));
    let handle: Arc<Mutex<dyn RunLog>> = concrete.clone();
    (concrete, handle)
}

#[tokio::test]
async fn a_prompt_search_logs_one_record_per_round_that_mirrors_its_result() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let path = workspace.path("runs/run.jsonl");
    let (concrete, handle) = file_log(&path);
    assert_eq!(
        concrete.lock().expect("an unpoisoned log").path(),
        path.as_path(),
        "the log reports the file it writes"
    );

    let result = logged_search(handle).await;
    let records = read_run_log(&path).expect("a readable run log");

    assert!(!records.is_empty(), "a search runs at least one round");
    assert_eq!(records.len(), result.iterations, "one record per round");
    assert_eq!(
        records
            .iter()
            .map(RunLogRecord::iteration)
            .collect::<Vec<usize>>(),
        (1..=records.len()).collect::<Vec<usize>>(),
        "round indices ascend from one"
    );
    // Every record mirrors its round: the winner, every candidate's score, and
    // the run's provenance.
    for (record, round) in records.iter().zip(&result.records) {
        assert_ne!(
            record.candidate.prompt().trim(),
            "",
            "a real candidate prompt"
        );
        assert!(
            round.candidates.contains(record.candidate()),
            "the logged winner is one of the round's candidates"
        );
        assert_eq!(record.scores.len(), round.candidates.len());
        assert!(record.scores.contains_key(record.candidate.candidate_id()));
        let own_score = record.scores[record.candidate.candidate_id()];
        assert!(
            (record.reward - own_score).abs() < 1e-12,
            "the reward is the winner's score"
        );
        assert_eq!(record.run_id, result.run_id);
        assert!((record.timestamp - RUN_LOG_NOW).abs() < 1e-12);
        // Adoption is reported per round and never changes the result.
        let decision = record.adopt.expect("a verdict per round");
        assert_eq!(record.adopted(), decision.adopted());
    }
    let first = records[0].adopt.expect("a verdict in round one");
    assert!(first.adopted(), "the first candidate is the baseline");
    assert_eq!(first.reason(), &AdoptReason::FirstCandidate);

    // The stopping verdict and the run's final best reach the last record.
    let last = records.last().expect("at least one round");
    assert!((last.best_score - result.score).abs() < 1e-12);
    assert_eq!(last.converged, result.converged);
    assert_eq!(last.convergence_reason, result.convergence_reason);
    // `best_score` never decreases, and `selected` marks exactly the rounds that
    // moved the elite: round one establishes it, later rounds only on a strict
    // gain of the run's best.
    for pair in records.windows(2) {
        assert!(
            pair[1].best_score >= pair[0].best_score - 1e-12,
            "the best score never decreases: {} -> {}",
            pair[0].best_score,
            pair[1].best_score
        );
        assert_eq!(
            pair[1].selected,
            pair[1].best_score > pair[0].best_score,
            "`selected` mirrors a strict improvement"
        );
    }
}

#[tokio::test]
async fn the_run_log_file_is_jsonl_and_the_readers_agree() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let path = workspace.path("runs/run.jsonl");
    let (_, handle) = file_log(&path);
    let filed = logged_search(handle).await;

    let memory = Arc::new(Mutex::new(InMemoryRunLog::new()));
    let memory_handle: Arc<Mutex<dyn RunLog>> = memory.clone();
    let remembered = logged_search(memory_handle).await;
    assert_eq!(filed, remembered, "a run log never changes the result");

    let records = read_run_log(&path).expect("a readable run log");
    let text = std::fs::read_to_string(&path).expect("the log file is readable");
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), records.len(), "one JSON object per line");
    for (line, record) in lines.iter().zip(&records) {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("each line is one JSON value");
        assert!(value.is_object(), "each line is a JSON object: {line}");
        // The wire keys are snake_case, matching the Python record names.
        for key in [
            "iteration",
            "candidate",
            "candidate_prompts",
            "scores",
            "best_score",
            "converged",
            "selected",
            "adopt",
            "run_id",
        ] {
            assert!(value.get(key).is_some(), "missing wire key `{key}`: {line}");
        }
        let candidate = value
            .get("candidate")
            .and_then(serde_json::Value::as_object)
            .expect("a candidate object");
        assert!(candidate.contains_key("candidate_id"), "{candidate:?}");
        // Every line round-trips through the public codec.
        assert_eq!(
            RunLogRecord::from_json_line(line).as_ref(),
            Some(record),
            "a line round-trips to its record"
        );
    }

    // The limited reader keeps the tail, and `0` means no limit.
    let last = records.last().expect("at least one record").clone();
    assert_eq!(
        read_run_log_limited(&path, 1).expect("a readable run log"),
        vec![last]
    );
    assert_eq!(
        read_run_log_limited(&path, 0).expect("a readable run log"),
        records
    );
    // A missing log is empty rather than an error.
    assert_eq!(
        read_run_log(workspace.path("runs/missing.jsonl")).expect("a missing log reads as empty"),
        Vec::new()
    );

    // The in-memory sink saw the same rounds as the file.
    let in_memory = memory.lock().expect("an unpoisoned log").records().to_vec();
    assert_eq!(in_memory, records, "both sinks record the same rounds");
}
