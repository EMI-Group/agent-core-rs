use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_foundation::tool::ToolInfo;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::{json, Value};

use super::*;
use crate::tune::evaluator::{
    DeterministicEvaluator, EvaluatedRun, ModelPromptRunner, RunnerEvaluator,
};
use crate::tune::tool::{ToolCallRecord, ToolStub, ToolStubs, ToolTrace};

fn case(id: &str, question: &str, answer: &str) -> Case {
    let mut inputs = BTreeMap::new();
    inputs.insert("question".to_owned(), Value::String(question.to_owned()));
    let mut label = BTreeMap::new();
    label.insert("answer".to_owned(), Value::String(answer.to_owned()));
    Case::new(inputs, label).unwrap().with_case_id(id)
}

fn train() -> Vec<Case> {
    vec![
        case("t1", "capital of France", "Paris"),
        case("t2", "2 + 2", "four"),
    ]
}

fn val() -> Vec<Case> {
    vec![case("v1", "capital of Italy", "Rome")]
}

const SEED: &str = "Answer the question.";

/// A tool-aware trainer: the model calls `lookup`, then answers `7`.
fn tool_trainer() -> Trainer {
    let model = ScriptedModel::builder()
        .model_name("test-model")
        .completion(MockCompletion::tool_call("lookup", json!({"q": "7"})))
        .respond_text("7")
        .build();
    let stubs = ToolStubs::new().with_stub(ToolStub::constant("lookup", json!("7")).unwrap());
    let runner = Arc::new(ModelPromptRunner::new(Arc::new(model)).with_stubs(stubs));
    Trainer::new().with_evaluator(Arc::new(RunnerEvaluator::new(runner)))
}

/// The one case declaring the tool [`tool_trainer`] calls.
fn tool_case() -> Case {
    Case::from_text("what is 7?", "7")
        .unwrap()
        .with_tools(vec![ToolInfo::new(
            "lookup",
            "Look something up",
            json!({"type": "object"}),
        )])
}

#[tokio::test]
async fn baseline_early_stop_skips_optimization() {
    let result = Trainer::new()
        .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
        .train(SEED, &train(), &val())
        .await
        .unwrap();
    assert_eq!(result.iterations, 0);
    assert_eq!(result.best_prompt, SEED);

    let (baseline, _) = Trainer::new().evaluate(SEED, &val()).await.unwrap();
    assert!((result.score - baseline).abs() < 1e-9);
}

#[tokio::test]
async fn improves_or_keeps_the_validation_score() {
    let (baseline, _) = Trainer::new().evaluate(SEED, &val()).await.unwrap();
    let result = Trainer::new().train(SEED, &train(), &val()).await.unwrap();
    assert!(
        result.score + 1e-12 >= baseline,
        "{} >= {baseline}",
        result.score
    );
    assert!(result.iterations >= 1);
}

#[tokio::test]
async fn epochs_are_bounded_by_the_configuration() {
    let result = Trainer::new()
        .with_config(
            OptimizerConfig::default()
                .with_iterations(2)
                .with_early_stop_score(1.0),
        )
        .train(SEED, &train(), &val())
        .await
        .unwrap();
    assert_eq!(result.iterations, 2);
}

#[tokio::test]
async fn an_empty_validation_split_falls_back_to_train() {
    let result = Trainer::new().train(SEED, &train(), &[]).await.unwrap();
    assert!(result.iterations >= 1);
    assert!(result.score > 0.0);
    // The score is now measured on the train split, so nothing is held out:
    // no held-out score and no selection tag are reported.
    assert!(result.val_score.is_none());
    assert!(result.selected_on.is_none());
    assert!(result.train_score.is_none());
    let (train_score, _) = Trainer::new()
        .evaluate(&result.best_prompt, &train())
        .await
        .unwrap();
    assert!((result.score - train_score).abs() < 1e-9);
}

#[tokio::test]
async fn an_empty_train_split_is_a_validation_error() {
    let error = Trainer::new().train(SEED, &[], &val()).await.unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("train"));

    let error = Trainer::new()
        .train("  ", &train(), &val())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("prompt"));
}

#[tokio::test]
async fn rejects_an_invalid_configuration() {
    let error = Trainer::new()
        .with_config(OptimizerConfig::default().with_iterations(0))
        .train(SEED, &train(), &val())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("iterations"));
}

#[tokio::test]
async fn evaluate_reports_the_mean_and_every_case() {
    let (mean, cases) = Trainer::new().evaluate(SEED, &train()).await.unwrap();
    assert_eq!(cases.len(), 2);
    let recomputed: f64 = cases.iter().map(|case| case.score).sum::<f64>() / 2.0;
    assert!((mean - recomputed).abs() < 1e-9);

    let (empty_mean, empty) = Trainer::new().evaluate(SEED, &[]).await.unwrap();
    assert!(empty_mean.abs() < 1e-9);
    assert_eq!(empty.len(), 0);
}

#[tokio::test]
async fn repeat_runs_are_identical() {
    let trainer = Trainer::new();
    let first = trainer.train(SEED, &train(), &val()).await.unwrap();
    let second = trainer.train(SEED, &train(), &val()).await.unwrap();
    assert_eq!(first, second);
}

#[tokio::test]
async fn keeps_the_seed_placeholders() {
    let seed = "Answer the question about {{topic}}.";
    let result = Trainer::new()
        .with_config(OptimizerConfig::default().with_early_stop_score(1.0))
        .train(seed, &train(), &val())
        .await
        .unwrap();
    assert!(
        result.best_prompt.contains("{{topic}}"),
        "{}",
        result.best_prompt
    );
}

#[tokio::test]
async fn model_rewrites_are_used_during_training() {
    let model = ScriptedModel::text(
        "<PROMPT_OPTIMIZED>Answer the question step by step and verify the result.</PROMPT_OPTIMIZED>",
    );
    let result = Trainer::new()
        .with_model(Arc::new(model.clone()))
        .train(SEED, &train(), &val())
        .await
        .unwrap();
    assert!(model.call_count() > 0);
    assert!(result.score > 0.0);
}

#[tokio::test]
async fn an_injected_evaluator_drives_the_selection() {
    /// Rewards prompts mentioning `Rome`, the validation answer.
    struct Rome;

    #[async_trait]
    impl CaseEvaluator for Rome {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            let score = if prompt.contains("Rome") { 1.0 } else { 0.0 };
            EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "rome")
        }
    }

    let result = Trainer::new()
        .with_evaluator(Arc::new(Rome))
        .train(SEED, &train(), &val())
        .await
        .unwrap();
    // Every candidate scores 0.0 on validation, so the seed stays.
    assert_eq!(result.best_prompt, SEED);
    assert!(result.score.abs() < 1e-9);
    assert_eq!(result.iterations, 3);
}

#[test]
fn debug_reports_the_model_choice() {
    assert!(format!("{:?}", Trainer::new()).contains("model: \"none\""));
}

#[tokio::test]
async fn the_default_evaluator_matches_the_deterministic_one() {
    let (score, _) = Trainer::new().evaluate(SEED, &train()).await.unwrap();
    let expected = DeterministicEvaluator::new(SEED)
        .score_cases(SEED, &train())
        .await
        .unwrap()
        .0;
    assert!((score - expected).abs() < 1e-9);
}

#[tokio::test]
async fn evaluate_traced_reports_tool_usage() {
    let summary = tool_trainer()
        .evaluate_traced("Answer with the tool.", &[tool_case()])
        .await
        .unwrap();
    assert!(summary.mean_score > 0.0);
    assert_eq!(summary.results.len(), 1);
    assert_eq!(
        summary.tool_usage,
        BTreeMap::from([("lookup".to_owned(), 1)])
    );
    assert_eq!(summary.tool_names(), vec!["lookup"]);
    assert!(!summary.is_empty());
}

#[tokio::test]
async fn evaluate_traced_matches_evaluate() {
    let cases = [tool_case()];
    let (mean, results) = tool_trainer()
        .evaluate("Answer with the tool.", &cases)
        .await
        .unwrap();
    let summary = tool_trainer()
        .evaluate_traced("Answer with the tool.", &cases)
        .await
        .unwrap();
    assert!((summary.mean_score - mean).abs() < 1e-12);
    assert_eq!(summary.results, results);
}

#[tokio::test]
async fn evaluate_traced_on_no_cases_is_empty() {
    let summary = tool_trainer()
        .evaluate_traced("Answer with the tool.", &[])
        .await
        .unwrap();
    assert!(summary.is_empty());
    assert!(summary.mean_score.abs() < 1e-12);
    assert!(summary.tool_usage.is_empty());
    assert_eq!(summary.tool_names(), Vec::<&str>::new());
    assert!(summary.telemetry.is_empty());
}

#[tokio::test]
async fn evaluate_traced_without_tools_reports_no_usage() {
    let summary = Trainer::new()
        .evaluate_traced(SEED, &train())
        .await
        .unwrap();
    assert!(summary.tool_usage.is_empty());
    assert_eq!(summary.tool_names(), Vec::<&str>::new());
    assert_eq!(summary.results.len(), 2);
    assert!(summary.mean_score > 0.0);
}

#[tokio::test]
async fn evaluate_traced_honors_an_injected_evaluator() {
    /// Scores every case `0.25` and reports one `search` call.
    struct Searcher;

    #[async_trait]
    impl CaseEvaluator for Searcher {
        async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            EvaluatedCase::new(case.clone(), BTreeMap::new(), 0.25, "searched")
        }

        async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
            let evaluated = self.evaluate(prompt, case).await?;
            let trace = ToolTrace::new([ToolCallRecord::success("search", json!({}), json!("ok"))]);
            Ok(EvaluatedRun::new(evaluated, trace, RunTelemetry::default()))
        }
    }

    let summary = Trainer::new()
        .with_evaluator(Arc::new(Searcher))
        .evaluate_traced(SEED, &[case("s1", "capital of Spain", "Madrid")])
        .await
        .unwrap();
    assert!((summary.mean_score - 0.25).abs() < 1e-12);
    assert_eq!(summary.tool_usage.get("search"), Some(&1));
    assert_eq!(summary.tool_names(), vec!["search"]);
    assert!(summary.telemetry.is_empty());
}

/// Scores every case `0.4`, calls `search` once, and reports 10 ms and 3+4
/// tokens per case.
struct Measured;

#[async_trait]
impl CaseEvaluator for Measured {
    async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        EvaluatedCase::new(case.clone(), BTreeMap::new(), 0.4, "measured")
    }

    async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
        let evaluated = self.evaluate(prompt, case).await?;
        let trace = ToolTrace::new([ToolCallRecord::success("search", json!({}), json!("ok"))]);
        let telemetry = RunTelemetry::default()
            .with_latency_ms(10.0)
            .with_prompt_tokens(3)
            .with_completion_tokens(4);
        Ok(EvaluatedRun::new(evaluated, trace, telemetry))
    }
}

#[tokio::test]
async fn evaluate_traced_aggregates_the_run_telemetry() {
    let cases = train();
    let trainer = Trainer::new().with_evaluator(Arc::new(Measured));
    let (mean, results) = trainer.evaluate(SEED, &cases).await.unwrap();
    let summary = trainer.evaluate_traced(SEED, &cases).await.unwrap();

    // Score, results, and tool usage are exactly `evaluate`'s.
    assert!((summary.mean_score - mean).abs() < 1e-12);
    assert_eq!(summary.results, results);
    assert_eq!(
        summary.tool_usage,
        BTreeMap::from([("search".to_owned(), 2)])
    );
    // Telemetry is summed in case order.
    assert_eq!(summary.telemetry.latency_ms, Some(20.0));
    assert_eq!(summary.telemetry.prompt_tokens, Some(6));
    assert_eq!(summary.telemetry.completion_tokens, Some(8));
    assert_eq!(summary.telemetry.total_tokens(), Some(14));
}

#[tokio::test]
async fn evaluation_summary_without_measurements_has_absent_telemetry() {
    let summary = Trainer::new()
        .evaluate_traced(SEED, &train())
        .await
        .unwrap();
    assert!(summary.telemetry.is_empty());
    assert!(!summary.is_empty());
}

#[test]
fn evaluation_summary_round_trips_through_json() {
    let evaluated = EvaluatedCase::new(case("c1", "q", "a"), BTreeMap::new(), 0.5, "half").unwrap();
    let summary = EvaluationSummary {
        mean_score: 0.5,
        results: vec![evaluated],
        tool_usage: BTreeMap::from([("search".to_owned(), 2)]),
        telemetry: RunTelemetry::default()
            .with_latency_ms(1.5)
            .with_prompt_tokens(2)
            .with_completion_tokens(3),
        split: CaseSplit::Val,
    };

    let value = serde_json::to_value(&summary).unwrap();
    let object = value.as_object().unwrap();
    for key in ["mean_score", "results", "tool_usage", "telemetry", "split"] {
        assert!(object.contains_key(key), "missing field {key}");
    }
    assert_eq!(value["telemetry"]["latency_ms"], json!(1.5));
    assert_eq!(value["telemetry"]["prompt_tokens"], json!(2));
    assert_eq!(value["split"], json!("val"));
    let restored: EvaluationSummary = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored, summary);

    let mut unknown = value;
    unknown["extra"] = json!(true);
    let restored: EvaluationSummary = serde_json::from_value(unknown).unwrap();
    assert_eq!(restored, summary);

    let partial: EvaluationSummary =
        serde_json::from_str(r#"{"mean_score": 0.0, "results": []}"#).unwrap();
    assert!(partial.is_empty());
    assert!(partial.tool_usage.is_empty());
    // An older payload without the key deserializes with absent telemetry and
    // no partition tag.
    assert!(partial.telemetry.is_empty());
    assert_eq!(partial.split, CaseSplit::Unspecified);
    let defaulted: EvaluationSummary = serde_json::from_str("{}").unwrap();
    assert_eq!(defaulted, EvaluationSummary::default());
    assert_eq!(defaulted.split, CaseSplit::Unspecified);
}

/// The train split of [`train`] tagged for a trainer.
fn train_split() -> SplitDataset {
    SplitDataset::new(CaseSplit::Train, train())
}

/// The validation split of [`val`] tagged for a trainer.
fn val_split() -> SplitDataset {
    SplitDataset::new(CaseSplit::Val, val())
}

#[tokio::test]
async fn evaluate_split_tags_the_summary_and_matches_evaluate_traced() {
    let dataset = val_split();
    let summary = Trainer::new().evaluate_split(SEED, &dataset).await.unwrap();
    let untagged = Trainer::new()
        .evaluate_traced(SEED, dataset.cases())
        .await
        .unwrap();

    assert_eq!(summary.split, CaseSplit::Val);
    assert_eq!(summary.mean_score, untagged.mean_score);
    assert_eq!(summary.results, untagged.results);
    assert_eq!(summary.tool_usage, untagged.tool_usage);
    assert_eq!(summary.telemetry, untagged.telemetry);
    // The untagged sibling keeps reporting no partition.
    assert_eq!(untagged.split, CaseSplit::Unspecified);
}

#[tokio::test]
async fn evaluate_split_keeps_the_train_tag_and_the_traced_telemetry() {
    let dataset = SplitDataset::new(CaseSplit::Train, train());
    let summary = Trainer::new()
        .with_evaluator(Arc::new(Measured))
        .evaluate_split(SEED, &dataset)
        .await
        .unwrap();
    assert_eq!(summary.split, CaseSplit::Train);
    assert_eq!(summary.results.len(), 2);
    assert_eq!(
        summary.tool_usage,
        BTreeMap::from([("search".to_owned(), 2)])
    );
    assert_eq!(summary.telemetry.latency_ms, Some(20.0));
    assert_eq!(summary.telemetry.total_tokens(), Some(14));

    let empty = Trainer::new()
        .evaluate_split(SEED, &SplitDataset::new(CaseSplit::Val, Vec::new()))
        .await
        .unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.split, CaseSplit::Val);
}

#[tokio::test]
async fn evaluate_split_traced_is_the_same_tagged_contract() {
    let dataset = val_split();
    let explicit = Trainer::new()
        .evaluate_split_traced(SEED, &dataset)
        .await
        .unwrap();
    let short = Trainer::new().evaluate_split(SEED, &dataset).await.unwrap();
    assert_eq!(explicit, short);
    assert_eq!(explicit.split, CaseSplit::Val);
    assert!(!explicit.is_empty());
}

#[tokio::test]
async fn train_split_reports_both_scores_and_the_selection_split() {
    let train = train_split();
    let val = val_split();
    let result = Trainer::new()
        .train_split(SEED, &train, &val)
        .await
        .unwrap();

    assert_eq!(result.selected_on, Some(CaseSplit::Val));
    assert_eq!(result.val_score, Some(result.score));
    let train_score = result
        .train_score
        .expect("the in-sample score must be reported");
    assert!((0.0..=1.0).contains(&train_score), "{train_score}");

    // The in-sample score is the winner measured over the train cases only.
    let (expected, _) = Trainer::new()
        .evaluate(&result.best_prompt, train.cases())
        .await
        .unwrap();
    assert!((train_score - expected).abs() < 1e-9);

    // The selection discipline is exactly `train`'s.
    let untagged = Trainer::new()
        .train(SEED, train.cases(), val.cases())
        .await
        .unwrap();
    assert_eq!(result.best_prompt, untagged.best_prompt);
    assert_eq!(result.score, untagged.score);
    assert_eq!(result.iterations, untagged.iterations);
    assert!(result.iterations >= 1);
}

#[tokio::test]
async fn train_split_early_stop_skips_optimization() {
    let train = train_split();
    let val = val_split();
    let result = Trainer::new()
        .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
        .train_split(SEED, &train, &val)
        .await
        .unwrap();
    assert_eq!(result.iterations, 0);
    assert_eq!(result.best_prompt, SEED);
    assert_eq!(result.selected_on, Some(CaseSplit::Val));
    assert_eq!(result.val_score, Some(result.score));
    // The seed is still measured in-sample, once.
    assert!(result.train_score.is_some());
}

#[tokio::test]
async fn train_split_epochs_are_bounded_by_the_configuration() {
    let result = Trainer::new()
        .with_config(
            OptimizerConfig::default()
                .with_iterations(2)
                .with_early_stop_score(1.0),
        )
        .train_split(SEED, &train_split(), &val_split())
        .await
        .unwrap();
    assert_eq!(result.iterations, 2);
    assert_eq!(result.selected_on, Some(CaseSplit::Val));
}

#[tokio::test]
async fn train_split_rejects_mis_tagged_splits() {
    for (train, val, expected) in [
        (
            SplitDataset::new(CaseSplit::Train, train()),
            SplitDataset::new(CaseSplit::Train, val()),
            "val",
        ),
        (
            SplitDataset::new(CaseSplit::Unspecified, train()),
            SplitDataset::new(CaseSplit::Val, val()),
            "train",
        ),
        (
            SplitDataset::new(CaseSplit::Val, train()),
            SplitDataset::new(CaseSplit::Train, val()),
            "train",
        ),
        (
            SplitDataset::new(CaseSplit::Train, train()),
            SplitDataset::new(CaseSplit::Unspecified, val()),
            "val",
        ),
    ] {
        let error = Trainer::new()
            .train_split(SEED, &train, &val)
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let message = error.to_string();
        assert!(message.contains("tagged"), "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[tokio::test]
async fn train_split_rejects_empty_splits() {
    let empty_train = SplitDataset::new(CaseSplit::Train, Vec::new());
    let error = Trainer::new()
        .train_split(SEED, &empty_train, &val_split())
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("train"));

    // A tagged run promises a held-out score, so an empty validation split is
    // an error rather than the untagged fallback to train.
    let empty_val = SplitDataset::new(CaseSplit::Val, Vec::new());
    let error = Trainer::new()
        .train_split(SEED, &train_split(), &empty_val)
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("val"));

    let error = Trainer::new()
        .train_split("  ", &train_split(), &val_split())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("prompt"));

    let error = Trainer::new()
        .with_config(OptimizerConfig::default().with_iterations(0))
        .train_split(SEED, &train_split(), &val_split())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("iterations"));
}

#[tokio::test]
async fn train_reports_the_val_score_without_a_selection_tag() {
    let result = Trainer::new().train(SEED, &train(), &val()).await.unwrap();
    assert_eq!(result.val_score, Some(result.score));
    assert!(result.selected_on.is_none());
    // The untagged caller gets no in-sample score, because the only split it
    // supplied is the held-out one.
    assert!(result.train_score.is_none());
}

/// Records the case ids of every `score_cases` call, then delegates.
struct Recorder {
    /// The evaluator whose scores the trainer actually uses.
    inner: Arc<dyn CaseEvaluator>,
    /// The case-id list of each `score_cases` call, in call order.
    calls: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
}

#[async_trait]
impl CaseEvaluator for Recorder {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        self.inner.evaluate(prompt, case).await
    }

    async fn score_cases(&self, prompt: &str, cases: &[Case]) -> Result<(f64, Vec<EvaluatedCase>)> {
        let ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
        self.calls
            .lock()
            .expect("the recording lock is not poisoned")
            .push(ids);
        self.inner.score_cases(prompt, cases).await
    }
}

#[tokio::test]
async fn candidate_search_never_sees_validation_cases() {
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let trainer = Trainer::new().with_evaluator(Arc::new(Recorder {
        inner: Arc::new(DeterministicEvaluator::new(SEED)),
        calls: Arc::clone(&calls),
    }));
    let train = train_split();
    let val = val_split();
    trainer.train_split(SEED, &train, &val).await.unwrap();

    let calls = calls
        .lock()
        .expect("the recording lock is not poisoned")
        .clone();
    assert!(!calls.is_empty(), "the trainer must score something");
    let train_ids = ["t1", "t2"];
    let val_ids = ["v1"];
    for ids in &calls {
        let all_train = ids.iter().all(|id| train_ids.contains(&id.as_str()));
        let all_val = ids.iter().all(|id| val_ids.contains(&id.as_str()));
        assert!(
            all_train || all_val,
            "a scored case set mixed the splits: {ids:?}"
        );
    }
    assert!(
        calls
            .iter()
            .any(|ids| !ids.is_empty() && ids.iter().all(|id| train_ids.contains(&id.as_str()))),
        "at least one call must score only the train split: {calls:?}"
    );
    assert!(
        calls
            .iter()
            .any(|ids| !ids.is_empty() && ids.iter().all(|id| val_ids.contains(&id.as_str()))),
        "at least one call must score only the val split: {calls:?}"
    );
}

/// A validation case whose inputs and label carry a unique marker token.
fn poisoned_val() -> Vec<Case> {
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "question".to_owned(),
        Value::String("does VALONLYTOKEN appear?".to_owned()),
    );
    let mut label = BTreeMap::new();
    label.insert(
        "answer".to_owned(),
        Value::String("VALONLYTOKEN".to_owned()),
    );
    vec![Case::new(inputs, label).unwrap().with_case_id("v1")]
}

#[tokio::test]
async fn train_split_never_learns_a_validation_token() {
    const TOKEN: &str = "VALONLYTOKEN";
    let train = train_split();
    let val = SplitDataset::new(CaseSplit::Val, poisoned_val());
    let model = ScriptedModel::text("<PROMPT_OPTIMIZED>Answer step by step.</PROMPT_OPTIMIZED>");
    let result = Trainer::new()
        .with_model(Arc::new(model.clone()))
        .train_split(SEED, &train, &val)
        .await
        .unwrap();

    assert!(result.iterations >= 1);
    assert!(
        !result.best_prompt.contains(TOKEN),
        "the winning prompt learned a validation token: {}",
        result.best_prompt
    );
    for record in &result.records {
        for candidate in &record.candidates {
            assert!(
                !candidate.prompt().contains(TOKEN),
                "a candidate prompt learned a validation token: {}",
                candidate.prompt()
            );
        }
    }
    // Nothing the model was asked to rewrite carried the validation case.
    assert!(model.call_count() >= 1);
    for call in model.calls() {
        for message in &call.messages {
            assert!(
                !message.text().contains(TOKEN),
                "a rewrite request carried the validation case: {}",
                message.text()
            );
        }
    }
}
