use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::tool::ToolInfo;
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
use serde_json::Value;

use super::*;
use crate::tune::base::EvaluatedCase;
use crate::tune::prompt_search::{JsonlPromptMemory, NullDriftJudge};
use crate::tune::{DeterministicEvaluator, PromptMemory};

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
    ]
}

const SEED: &str = "Answer the question.";

/// An evaluator returning a fixed score and a complete answer.
struct Fixed {
    score: f64,
}

#[async_trait]
impl CaseEvaluator for Fixed {
    async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        let mut answer = BTreeMap::new();
        answer.insert(
            "output".to_owned(),
            Value::String("complete answer".to_owned()),
        );
        EvaluatedCase::new(case.clone(), answer, self.score, "fixed")
    }
}

#[tokio::test]
async fn runs_multiple_rounds_and_records_each_one() {
    let result = PromptSearchOptimizer::new()
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert_eq!(result.records.len(), result.iterations);
    assert!(
        (1..=3).contains(&result.iterations),
        "{}",
        result.iterations
    );
    assert!(result.records.iter().all(|record| record.index >= 1));
    assert!(result.candidate_prompts().contains(&result.best_prompt));
    assert!(result.score > 0.0);
}

#[tokio::test]
async fn the_run_is_deterministic() {
    let optimizer = PromptSearchOptimizer::new();
    let first = optimizer.optimize(SEED, &cases()).await.unwrap();
    let second = optimizer.optimize(SEED, &cases()).await.unwrap();
    assert_eq!(first, second);
    assert_eq!(first.run_id, second.run_id);
    assert_ne!(first.run_id.len(), 0);
}

#[tokio::test]
async fn keeps_placeholders_and_the_seed_contract() {
    let seed = "Answer about {{topic}}.";
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(DeterministicEvaluator::new(seed)))
        .optimize(seed, &cases())
        .await
        .unwrap();
    assert!(
        result.best_prompt.contains("{{topic}}"),
        "{}",
        result.best_prompt
    );
    assert!(result.best_prompt.starts_with("Answer about {{topic}}."));
}

#[tokio::test]
async fn rejects_invalid_inputs() {
    let error = PromptSearchOptimizer::new()
        .optimize("  ", &cases())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = PromptSearchOptimizer::new()
        .optimize(SEED, &[])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cases"));

    let error = PromptSearchOptimizer::new()
        .with_config(OptimizerConfig::default().with_beam_width(0))
        .optimize(SEED, &cases())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("beam_width"));

    let error = PromptSearchOptimizer::new()
        .with_search_config(PromptSearchConfig::default().with_candidate_prompts(0))
        .optimize(SEED, &cases())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("candidate_prompts"));
}

#[tokio::test]
async fn a_model_rewrite_can_win_a_round() {
    /// Rewards the marker only the model answer carries.
    struct Marker;

    #[async_trait]
    impl CaseEvaluator for Marker {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            let score = if prompt.contains("MAGIC") { 1.0 } else { 0.0 };
            EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "marker")
        }
    }

    let model = ScriptedModel::text("<PROMPT_OPTIMIZED>MAGIC answer.</PROMPT_OPTIMIZED>");
    let result = PromptSearchOptimizer::new()
        .with_search_config(
            PromptSearchConfig::default()
                .with_drift_penalty(0.0)
                .with_candidate_prompts(10),
        )
        .with_model(Arc::new(model.clone()))
        .with_evaluator(Arc::new(Marker))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(model.call_count() >= 1);
    assert_eq!(result.best_prompt, "MAGIC answer.");
    assert!(result.score > 0.0, "{}", result.score);
}

#[tokio::test]
async fn tool_declarations_reach_the_rewrite_prompt() {
    let model = ScriptedModel::text("<PROMPT_OPTIMIZED>MAGIC answer.</PROMPT_OPTIMIZED>");
    let cases = vec![
        case("c1", "capital of France", "Paris").with_tools(vec![ToolInfo::new(
            "lookup",
            "Look something up",
            serde_json::json!({"type": "object", "properties": {}}),
        )]),
        case("c2", "2 + 2", "four"),
    ];
    PromptSearchOptimizer::new()
        .with_model(Arc::new(model.clone()))
        .optimize(SEED, &cases)
        .await
        .unwrap();

    let call = model.last_call().expect("one model rewrite");
    let user = call.messages[1].text();
    assert!(user.contains("<tools_description>"), "{user}");
    assert!(user.contains("lookup"), "{user}");
    assert!(call.request.tools.is_none(), "{user}");
}

#[tokio::test]
async fn the_policy_temperature_and_lessons_reach_the_rewrite_prompt() {
    let model = ScriptedModel::text("plain text, no tags");
    PromptSearchOptimizer::new()
        .with_search_config(
            PromptSearchConfig::default()
                .with_policy_temperature(1.5)
                .with_max_iterations(2),
        )
        .with_config(OptimizerConfig::default().with_iterations(2))
        .with_model(Arc::new(model.clone()))
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let calls = model.calls();
    assert_eq!(calls.len(), 2, "one rewrite per round");
    assert_eq!(calls[0].request.temperature, Some(1.5));
    // Round 1 has no lessons yet; round 2 carries the compressed history.
    assert!(!calls[0].messages[1].text().contains("<policy_context>"));
    let second = calls[1].messages[1].text();
    assert!(second.contains("<policy_context>"), "{second}");
    assert!(second.contains("Distilled lessons"), "{second}");
}

#[tokio::test]
async fn an_unusable_model_answer_keeps_the_bank() {
    let model = ScriptedModel::text("plain text, no tags");
    let result = PromptSearchOptimizer::new()
        .with_model(Arc::new(model.clone()))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(model.call_count() >= 1);
    assert_ne!(result.best_prompt, SEED);
    assert!(result.score > 0.0);
}

#[tokio::test]
async fn the_seed_wins_when_nothing_improves() {
    /// Scores the seed highest, so no rewrite may win.
    struct SeedOnly;

    #[async_trait]
    impl CaseEvaluator for SeedOnly {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            let score = if prompt == SEED { 1.0 } else { 0.2 };
            let mut answer = BTreeMap::new();
            answer.insert("output".to_owned(), Value::String("answer".to_owned()));
            EvaluatedCase::new(case.clone(), answer, score, "seed")
        }
    }

    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(SeedOnly))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert_eq!(result.best_prompt, SEED);
    assert!((result.score - 1.0).abs() < 1e-9, "{}", result.score);
}

#[tokio::test]
async fn converges_on_a_target_reward() {
    let result = PromptSearchOptimizer::new()
        .with_search_config(PromptSearchConfig::default().with_target_score(Some(0.1)))
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(result.converged);
    let reason = result.convergence_reason.clone().unwrap();
    assert!(reason.contains("target reward"), "{reason}");
    assert_eq!(result.iterations, 1);
}

#[tokio::test]
async fn converges_on_a_flat_plateau() {
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(result.converged);
    let reason = result.convergence_reason.clone().unwrap();
    assert!(reason.contains("variance"), "{reason}");
    assert_eq!(result.iterations, 3);
}

#[tokio::test]
async fn stops_on_the_early_stop_score() {
    let result = PromptSearchOptimizer::new()
        .with_config(OptimizerConfig::default().with_early_stop_score(0.5))
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(result.converged);
    let reason = result.convergence_reason.clone().unwrap();
    assert!(reason.contains("early stop score"), "{reason}");
    assert_eq!(result.iterations, 1);
}

#[tokio::test]
async fn a_budget_bound_run_reports_no_convergence() {
    let result = PromptSearchOptimizer::new()
        .with_config(OptimizerConfig::default().with_iterations(2))
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(!result.converged);
    assert!(result.convergence_reason.is_none());
    assert_eq!(result.iterations, 2);
    assert!(result.token_usage.is_empty());
}

#[tokio::test]
async fn the_candidate_budget_bounds_every_round() {
    let result = PromptSearchOptimizer::new()
        .with_search_config(PromptSearchConfig::default().with_candidate_prompts(2))
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    for record in &result.records {
        assert!(record.candidates.len() <= 2, "{}", record.candidates.len());
        assert_eq!(record.candidates.len(), record.scores.len());
    }
}

#[tokio::test]
async fn hidden_cases_punish_an_overfitted_candidate() {
    /// Rewards the marker on the visible cases only.
    struct Overfit;

    #[async_trait]
    impl CaseEvaluator for Overfit {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            let mut answer = BTreeMap::new();
            answer.insert("output".to_owned(), Value::String("answer".to_owned()));
            let score = if prompt.contains("MAGIC") {
                if case
                    .inputs
                    .values()
                    .any(|value| value.as_str() == Some("held out"))
                {
                    0.0
                } else {
                    1.0
                }
            } else {
                0.6
            };
            EvaluatedCase::new(case.clone(), answer, score, "overfit")
        }
    }

    async fn run(cases: Vec<PromptTaskCase>) -> OptimizationResult {
        let model = ScriptedModel::text("<PROMPT_OPTIMIZED>MAGIC answer.</PROMPT_OPTIMIZED>");
        let task = PromptTaskSpec::new(SEED)
            .with_base_prompt(SEED)
            .with_cases(cases);
        PromptSearchOptimizer::new()
            .with_search_config(
                PromptSearchConfig::default()
                    .with_drift_penalty(0.0)
                    .with_candidate_prompts(10),
            )
            .with_model(Arc::new(model))
            .with_evaluator(Arc::new(Overfit))
            .optimize_task(&task)
            .await
            .unwrap()
    }

    let visible_case = PromptTaskCase::from_text("capital of France", "Paris").unwrap();
    let held_out_case = PromptTaskCase::from_text("held out", "secret").unwrap();

    // Without a held-out case the marker is visible-perfect and wins.
    let visible = run(vec![visible_case.clone()]).await;
    assert_eq!(visible.best_prompt, "MAGIC answer.");
    assert!((visible.score - 1.0).abs() < 1e-9, "{}", visible.score);

    // Held out, the same candidate's visible-perfect score is penalised.
    let overfitted = run(vec![visible_case, held_out_case.with_hidden(true)]).await;
    assert_ne!(overfitted.best_prompt, "MAGIC answer.");
    assert!(overfitted.score < visible.score);
    let record = &overfitted.records[0];
    let marker = record
        .candidates
        .iter()
        .position(|candidate| candidate.prompt() == "MAGIC answer.")
        .expect("the overfitted candidate was tried");
    assert!(
        record.scores[marker] < record.scores[0],
        "the hidden set put it behind the seed: {:?}",
        record.scores
    );
}

#[tokio::test]
async fn warm_starts_and_persists_through_the_memory() {
    let workspace = TempWorkspace::new().unwrap();
    let memory = Arc::new(JsonlPromptMemory::new(workspace.path("memory")).unwrap());
    let prior = PromptRecord::new("prior", "Answer with a full sentence.", 0.25, SEED)
        .with_characteristics(SEED);
    memory.store(prior.clone()).unwrap();

    let result = PromptSearchOptimizer::new()
        .with_memory(memory.clone())
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let first_round = &result.records[0];
    assert!(
        first_round
            .candidates
            .iter()
            .any(|candidate| candidate.prompt() == prior.prompt
                && candidate.rationale() == "warm start from prompt memory"),
        "{:?}",
        first_round.candidates
    );

    let records = memory.records();
    assert_eq!(records.len(), 2);
    let stored = records
        .iter()
        .find(|record| record.record_id.starts_with(&result.run_id))
        .expect("the run persisted its best prompt");
    assert_eq!(stored.prompt, result.best_prompt);
    assert_eq!(stored.baseline_reward, Some(prior.reward));
    assert_eq!(
        stored.metadata["iterations"],
        Value::from(result.iterations)
    );
}

#[tokio::test]
async fn an_injected_drift_judge_penalises_a_marker_prompt() {
    /// Deviation 1.0 for the marker, 0.0 otherwise.
    struct MarkerDrift;

    #[async_trait]
    impl DriftJudge for MarkerDrift {
        async fn deviation(&self, _objective: &str, candidate: &PromptCandidate) -> f64 {
            if candidate.prompt().contains("MAGIC") {
                1.0
            } else {
                0.0
            }
        }
    }

    /// The marker scores best, everything else lags behind it.
    struct Marker;

    #[async_trait]
    impl CaseEvaluator for Marker {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            let mut answer = BTreeMap::new();
            answer.insert("output".to_owned(), Value::String("answer".to_owned()));
            let score = if prompt.contains("MAGIC") { 1.0 } else { 0.7 };
            EvaluatedCase::new(case.clone(), answer, score, "marker")
        }
    }

    let optimizer = |judge: Option<Arc<dyn DriftJudge>>| {
        // A fresh single-turn model per run: the first rewrite carries the
        // marker, later rounds reuse the default fallback answer.
        let model = ScriptedModel::text("<PROMPT_OPTIMIZED>MAGIC answer.</PROMPT_OPTIMIZED>");
        let optimizer = PromptSearchOptimizer::new()
            .with_search_config(PromptSearchConfig::default().with_candidate_prompts(10))
            .with_model(Arc::new(model))
            .with_evaluator(Arc::new(Marker));
        match judge {
            Some(judge) => optimizer.with_drift_judge(judge),
            None => optimizer,
        }
    };

    let penalised = optimizer(Some(Arc::new(MarkerDrift)))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert_ne!(penalised.best_prompt, "MAGIC answer.");

    let unpenalised = optimizer(Some(Arc::new(NullDriftJudge::new())))
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert_eq!(unpenalised.best_prompt, "MAGIC answer.");
    assert!(penalised.score < unpenalised.score);
}

#[tokio::test]
async fn optimize_task_keeps_constraints_and_metadata() {
    let mut metadata = BTreeMap::new();
    metadata.insert("suite".to_owned(), Value::String("demo".to_owned()));
    let task = PromptTaskSpec::new(SEED)
        .with_base_prompt("Answer in one sentence.")
        .with_constraints(vec!["no lists".to_owned()])
        .with_cases(vec![
            PromptTaskCase::from_text("capital of France", "Paris").unwrap(),
            PromptTaskCase::from_text("2 + 2", "four").unwrap(),
        ])
        .with_metadata(metadata);
    let result = PromptSearchOptimizer::new()
        .optimize_task(&task)
        .await
        .unwrap();
    assert_eq!(result.run_id, PromptSearchOptimizer::run_id(0, &task));
    assert_eq!(result.iterations, result.records.len());
    assert!(result.score > 0.0);
    assert_ne!(result.best_prompt.len(), 0);

    let seeded = PromptSearchOptimizer::new()
        .with_run_id_seed(7)
        .optimize_task(&task)
        .await
        .unwrap();
    assert_eq!(seeded.run_id, PromptSearchOptimizer::run_id(7, &task));
    assert_ne!(seeded.run_id, result.run_id);
}

#[tokio::test]
async fn optimize_task_validates_its_inputs() {
    let blank = PromptSearchOptimizer::new()
        .optimize_task(&PromptTaskSpec::new("  "))
        .await
        .unwrap_err();
    assert_eq!(blank.kind(), ErrorKind::Validation);

    let empty = PromptSearchOptimizer::new()
        .optimize_task(&PromptTaskSpec::new(SEED))
        .await
        .unwrap_err();
    assert!(empty.to_string().contains("cases"));

    let blank_case = PromptSearchOptimizer::new()
        .optimize_task(
            &PromptTaskSpec::new(SEED)
                .with_cases(vec![PromptTaskCase::from_text("q", "a").unwrap()]),
        )
        .await;
    assert!(blank_case.is_ok());

    let invalid_case = PromptSearchOptimizer::new()
        .optimize_task(&PromptTaskSpec::new(SEED).with_cases(vec![PromptTaskCase::default()]))
        .await
        .unwrap_err();
    assert_eq!(invalid_case.kind(), ErrorKind::Validation);
    assert!(invalid_case.to_string().contains("cases[0]"));
}

#[tokio::test]
async fn a_blank_base_prompt_falls_back_to_the_objective() {
    let task = PromptTaskSpec::new(SEED)
        .with_cases(vec![PromptTaskCase::from_text("2 + 2", "four").unwrap()]);
    let result = PromptSearchOptimizer::new()
        .optimize_task(&task)
        .await
        .unwrap();
    assert!(
        result.best_prompt.starts_with(SEED),
        "{}",
        result.best_prompt
    );
}

#[test]
fn debug_reports_the_model_choice() {
    let debug = format!("{:?}", PromptSearchOptimizer::new());
    assert!(debug.contains("model: \"none\""));
    assert!(debug.contains("search_config"));
    assert!(debug.contains("custom_reward"));
    assert!(debug.contains("custom_drift_judge"));
    assert!(debug.contains("has_memory"));
}

#[test]
fn run_id_is_deterministic_and_task_sensitive() {
    let task = PromptTaskSpec::new("objective").with_base_prompt("base");
    let more_cases = task
        .clone()
        .with_cases(vec![PromptTaskCase::from_text("q", "a").unwrap()]);
    assert_eq!(
        PromptSearchOptimizer::run_id(0, &task),
        PromptSearchOptimizer::run_id(0, &task)
    );
    assert_ne!(
        PromptSearchOptimizer::run_id(0, &task),
        PromptSearchOptimizer::run_id(1, &task)
    );
    assert_ne!(
        PromptSearchOptimizer::run_id(0, &task),
        PromptSearchOptimizer::run_id(0, &more_cases)
    );
    assert_eq!(PromptSearchOptimizer::run_id(0, &task).len(), 16 + 1 + 8);
}
