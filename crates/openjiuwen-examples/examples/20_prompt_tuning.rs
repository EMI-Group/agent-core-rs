//! `20_prompt_tuning` — prompt tuning through `openjiuwen::dev_tools::tune`.
//!
//! Python counterpart: `openjiuwen/dev_tools/tune/` — the `Case`/`CaseLoader`
//! data layer, the `evaluator/` seam, `optimizer/`, and `trainer/`. Four parts:
//!
//! 1. **case data** — four `Case`s (one carrying tools), the
//!    `{"cases": [...]}` JSON round trip, a seeded shuffle, and a split that a
//!    second identical loader reproduces.
//! 2. **scoring** — `RunnerEvaluator` + `LabelScorer` over a
//!    `ScriptedPromptRunner` reproduces the documented bands (exact `1.0`,
//!    case-insensitive `0.9`, containment `0.75`); then the offline
//!    `DeterministicEvaluator` scores a weak seed prompt per case and on average.
//! 3. **optimisation** — `InstructionOptimizer::optimize` returns an
//!    `OptimizationResult { best_prompt, score, iterations }` above the seed,
//!    `Trainer::train` runs the train/validation loop (and early-stops on a
//!    satisfied threshold), and `placeholders` reports a prompt's `{{slot}}`s.
//! 4. **the live switch** — `RunMode::Live` grades candidates with a
//!    `ModelEvaluator`, so the same optimiser runs against a real model; the
//!    offline branch says so in one note line.
//!
//! Offline and deterministic throughout: canned scripted answers, the
//! deterministic evaluator (a pure function of seed, candidate, and case),
//! explicit case ids, and scores printed to three decimals.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use openjiuwen::core::foundation::tool::ToolInfo;
use openjiuwen::dev_tools::tune::{
    placeholders, Case, CaseEvaluator, CaseLoader, DeterministicEvaluator, EvaluatedCase,
    InstructionOptimizer, ModelEvaluator, OptimizerConfig, PromptRunner, RunnerEvaluator,
    ScriptedPromptRunner, Trainer, DEFAULT_ITERATION_NUM,
};
use serde_json::{json, Value};

/// The prompt every search starts from: one imperative sentence, with no steps,
/// no output format and no constraint, so the structure dimension can grow.
const SEED: &str = "Answer the question.";

/// A prompt with `{{...}}` slots, for the placeholder walkthrough.
const SLOT_PROMPT: &str = "Summarise {{document}} for {{audience}} in the requested output format.";

/// The fixed shuffle seed: the whole state of the permutation.
const SHUFFLE_SEED: u64 = 7;

/// The train/validation split ratio: two cases each out of four.
const SPLIT_RATIO: f64 = 0.5;

/// The case that also carries a tool declaration.
const TOOL_CASE_ID: &str = "case-4";

/// The three score bands of `LabelScorer`, as (case id, label, answer).
const BANDS: [(&str, &str, &str); 3] = [
    ("exact", "Paris", "Paris"),
    ("case-insensitive", "Paris", "paris"),
    ("containment", "Paris", "Paris, the capital of France"),
];

/// The four tuning cases, as (id, question, expected answer).
const FIXTURES: [(&str, &str, &str); 4] = [
    (
        "case-1",
        "How do I reset my password?",
        "Use the account settings page.",
    ),
    (
        "case-2",
        "Where do I find my invoice?",
        "The invoice is in the billing section.",
    ),
    (
        "case-3",
        "How long does the trial last?",
        "The trial lasts 30 days.",
    ),
    (
        TOOL_CASE_ID,
        "Can I export my data?",
        "Export the data from the report view.",
    ),
];

/// Builds one [`Case`] with an explicit id, so no generated UUID reaches stdout.
fn case(id: &str, question: &str, answer: &str) -> Result<Case> {
    let inputs = BTreeMap::from([("question".to_owned(), json!(question))]);
    let label = BTreeMap::from([("answer".to_owned(), json!(answer))]);
    Ok(Case::new(inputs, label)
        .with_context(|| format!("case `{id}` needs inputs and a label"))?
        .with_case_id(id))
}

/// Builds the four cases; the last one also declares a tool.
fn cases() -> Result<Vec<Case>> {
    let mut built = Vec::with_capacity(FIXTURES.len());
    for (id, question, answer) in FIXTURES {
        let case = case(id, question, answer)?;
        built.push(if id == TOOL_CASE_ID {
            case.with_tools(vec![ToolInfo::new(
                "lookup_account",
                "look up an account by its e-mail address",
                json!({"type": "object", "properties": {"email": {"type": "string"}}}),
            )])
        } else {
            case
        });
    }
    Ok(built)
}

/// The case ids of `loader`, in order.
fn ids(loader: &CaseLoader) -> Vec<String> {
    loader.iter().map(|case| case.case_id.clone()).collect()
}

/// The case ids of `loader`, sorted, for a permutation check.
fn sorted_ids(loader: &CaseLoader) -> Vec<String> {
    let mut sorted = ids(loader);
    sorted.sort();
    sorted
}

/// Prints `id: score — reason` for every evaluated case.
fn note_evaluated(evaluated: &[EvaluatedCase]) {
    for case in evaluated {
        common::banner::note(&format!(
            "`{}`: {:.3} — {}",
            case.case_id(),
            case.score,
            case.reason
        ));
    }
}

/// Part 1: the case data layer; returns the loader plus its two splits.
fn case_data() -> Result<(CaseLoader, CaseLoader, CaseLoader)> {
    common::banner::step("part 1 — case data: build, JSON round trip, seeded shuffle, split");

    let fixtures = cases()?;
    let loader = CaseLoader::new(fixtures.clone());
    let tool_case = fixtures.last().context("the fixture set is not empty")?;
    let tools = tool_case
        .tools
        .as_deref()
        .context("the tool case exposes its declarations")?;
    let names: Vec<&str> = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
    common::banner::note(&format!(
        "{} case(s) [{}]; `{}` also declares tool(s) {}",
        loader.size(),
        ids(&loader).join(", "),
        tool_case.case_id,
        names.join(", ")
    ));

    // The Python field names survive the JSON form unchanged, so a fixture file
    // written by one process is a loader in the next.
    let payload = serde_json::to_value(&loader).context("the loader serializes")?;
    let entries = payload
        .get("cases")
        .and_then(Value::as_array)
        .context("the JSON form is a `cases` array")?
        .len();
    let restored: CaseLoader = serde_json::from_value(payload).context("the JSON parses back")?;
    ensure!(
        restored == loader,
        "the loader must survive its own JSON form"
    );
    common::banner::note(&format!(
        "json round trip: {{\"cases\": [...]}} with {entries} entries parses back equal, tools included"
    ));

    // A seeded Fisher-Yates shuffle: the same seed is the whole state.
    let shuffled = loader.shuffle(SHUFFLE_SEED);
    ensure!(
        ids(&shuffled) != ids(&loader),
        "the fixed seed must actually reorder the cases"
    );
    ensure!(
        sorted_ids(&shuffled) == sorted_ids(&loader),
        "a shuffle keeps every case exactly once"
    );
    common::banner::note(&format!(
        "shuffle(seed {SHUFFLE_SEED}) reorders the ids: {}",
        ids(&shuffled).join(", ")
    ));

    let (train, val) = shuffled
        .split(SPLIT_RATIO)
        .with_context(|| format!("ratio {SPLIT_RATIO} is a valid split"))?;
    common::banner::note(&format!(
        "split({SPLIT_RATIO}) → train [{}], validation [{}]",
        ids(&train).join(", "),
        ids(&val).join(", ")
    ));

    let replay = CaseLoader::new(fixtures).shuffle(SHUFFLE_SEED);
    let (replay_train, replay_val) = replay
        .split(SPLIT_RATIO)
        .context("replaying the split at the same ratio")?;
    ensure!(
        ids(&replay_train) == ids(&train) && ids(&replay_val) == ids(&val),
        "the same seed and ratio must reproduce the split on a second loader"
    );
    common::banner::note("a second identical loader reproduces both splits exactly");

    Ok((loader, train, val))
}

/// Part 2: the scoring seam, first through a scripted runner, then offline.
async fn scoring(loader: &CaseLoader) -> Result<()> {
    common::banner::step(
        "part 2 — scoring: `RunnerEvaluator` + `LabelScorer` over a `ScriptedPromptRunner`, then the \
         offline `DeterministicEvaluator`",
    );

    // The three documented bands, produced by the scorer rather than hard-coded.
    let mut probes = Vec::with_capacity(BANDS.len());
    for (id, label, _) in BANDS {
        probes.push(case(id, "capital of France?", label)?);
    }
    let scripted = Arc::new(
        ScriptedPromptRunner::new()
            .with_text_answer("exact", "Paris")
            .with_text_answer("case-insensitive", "paris")
            .with_text_answer("containment", "Paris, the capital of France"),
    );
    let runner: Arc<dyn PromptRunner> = Arc::clone(&scripted) as Arc<dyn PromptRunner>;
    let (mean, evaluated) = RunnerEvaluator::new(runner)
        .score_cases("You answer trivia questions.", &probes)
        .await
        .context("scoring the scripted answers")?;
    common::banner::note(
        "the scripted runner replays one canned answer per case id; `LabelScorer` grades it",
    );
    note_evaluated(&evaluated);
    for (case, band) in evaluated.iter().zip([1.0, 0.9, 0.75]) {
        ensure!(
            (case.score - band).abs() < 1e-9,
            "`{}` should score {band:.2}, got {:.3}",
            case.case_id(),
            case.score
        );
    }
    common::banner::note(&format!(
        "mean over {} probe(s): {:.3}; the runner recorded {} call(s)",
        evaluated.len(),
        mean,
        scripted.call_count()
    ));

    // The offline evaluator: no model, no answers, a pure function.
    let evaluator = DeterministicEvaluator::new(SEED);
    common::banner::note(&format!(
        "`DeterministicEvaluator::new({SEED:?})`: 0.35·placeholder + 0.20·structure + \
         0.30·coverage + 0.15·hygiene, with no model call"
    ));
    let (seed_mean, seed_cases) = evaluator
        .score_cases(SEED, loader.cases())
        .await
        .context("scoring the seed prompt offline")?;
    note_evaluated(&seed_cases);
    common::banner::note(&format!(
        "weak seed prompt mean over {} case(s): {:.3}; only coverage varies by case",
        seed_cases.len(),
        seed_mean
    ));

    Ok(())
}

/// Part 3a: the instruction search; returns the seed and tuned mean scores.
async fn search(loader: &CaseLoader) -> Result<(f64, f64)> {
    common::banner::step(
        "part 3 — optimisation: `InstructionOptimizer` beam search, then `Trainer` on the \
         train/validation split",
    );

    let evaluator: Arc<dyn CaseEvaluator> = Arc::new(DeterministicEvaluator::new(SEED));
    let (seed_score, _) = evaluator
        .score_cases(SEED, loader.cases())
        .await
        .context("scoring the seed prompt")?;
    common::banner::note(&format!(
        "seed prompt {SEED:?} scores {seed_score:.3} over {} case(s)",
        loader.size()
    ));

    let result = InstructionOptimizer::new()
        .with_evaluator(Arc::clone(&evaluator))
        .optimize(SEED, loader.cases())
        .await
        .context("the instruction search runs")?;
    ensure!(
        result.score > seed_score,
        "the tuned score {:.3} must beat the seed score {seed_score:.3}",
        result.score
    );
    for line in result
        .best_prompt
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        common::banner::note(&format!("tuned | {line}"));
    }
    common::banner::note(&format!(
        "tuned score {:.3} after {}/{} round(s): ties go to the seed, so a rewrite has to earn its place",
        result.score, result.iterations, DEFAULT_ITERATION_NUM
    ));

    Ok((seed_score, result.score))
}

/// Part 3b: the train/validation loop and its early-stop outcome.
async fn training(train: &CaseLoader, val: &CaseLoader) -> Result<()> {
    common::banner::step(
        "the trainer: one instruction round per epoch on the train split, selected on the validation \
         split",
    );

    let evaluator: Arc<dyn CaseEvaluator> = Arc::new(DeterministicEvaluator::new(SEED));
    let trainer = Trainer::new().with_evaluator(Arc::clone(&evaluator));
    let (baseline, _) = trainer
        .evaluate(SEED, val.cases())
        .await
        .context("the baseline validation score")?;
    let outcome = trainer
        .train(SEED, train.cases(), val.cases())
        .await
        .context("the training loop runs")?;
    ensure!(
        outcome.score + 1e-12 >= baseline,
        "training must never lower the validation score ({:.3} < {baseline:.3})",
        outcome.score
    );
    common::banner::note(&format!(
        "baseline validation {baseline:.3} over {} case(s) → reported {:.3} after {}/{} epoch(s); the \
         score is always measured on the validation split",
        val.size(),
        outcome.score,
        outcome.iterations,
        DEFAULT_ITERATION_NUM
    ));

    // A threshold the baseline already meets stops before any epoch runs.
    let stopped = Trainer::new()
        .with_evaluator(Arc::clone(&evaluator))
        .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
        .train(SEED, train.cases(), val.cases())
        .await
        .context("the early-stop run")?;
    ensure!(
        stopped.iterations == 0 && stopped.best_prompt == SEED.trim(),
        "a satisfied threshold runs no epoch and keeps the seed verbatim"
    );
    common::banner::note(&format!(
        "early stop: the threshold 0.00 is met by the baseline, so 0 epoch(s) ran and the seed \
         ({:.3}) stays",
        stopped.score
    ));
    Ok(())
}

/// Part 3c: the `{{name}}` slots of a prompt, and the repair a rewrite gets.
async fn slots(loader: &CaseLoader) -> Result<()> {
    common::banner::step(
        "placeholders: the `{{name}}` slots of a prompt, and the repair a rewrite gets",
    );

    let declared = placeholders(SLOT_PROMPT);
    let names: Vec<&str> = declared.iter().map(String::as_str).collect();
    common::banner::note(&format!(
        "{SLOT_PROMPT:?} declares {} slot(s): {}",
        declared.len(),
        names.join(", ")
    ));
    ensure!(declared.len() == 2, "the slot prompt declares two names");

    let kept = InstructionOptimizer::new()
        .with_config(OptimizerConfig::default().with_iterations(1))
        .optimize(SLOT_PROMPT, loader.cases())
        .await
        .context("the slot-preserving search runs")?;
    ensure!(
        placeholders(&kept.best_prompt) == declared,
        "every seed slot must survive the rewrite"
    );
    common::banner::note(&format!(
        "one round on the slot prompt keeps both slots and scores {:.3}",
        kept.score
    ));
    Ok(())
}

/// Part 4: the live switch, or the note explaining why it is off.
async fn live(mode: &common::RunMode, loader: &CaseLoader) -> Result<()> {
    common::banner::step("part 4 — the live switch: the same optimiser with a `ModelEvaluator`");
    let common::RunMode::Live { provider, model } = mode else {
        common::banner::note(
            "offline: every score above came from the deterministic evaluator, with no model call; \
             set OPENJIUWEN_LIVE=1 plus MODEL_NAME and a credential for a real model",
        );
        return Ok(());
    };

    let client = common::live_model()
        .await
        .context("building the live model from the environment")?;
    common::banner::note(&format!(
        "live: {provider}/{model} scores each candidate through `ModelEvaluator`, one model call per \
         case; the algorithm is unchanged"
    ));
    let evaluator: Arc<dyn CaseEvaluator> = Arc::new(ModelEvaluator::new(client));
    let result = InstructionOptimizer::new()
        .with_config(OptimizerConfig::default().with_iterations(1))
        .with_evaluator(evaluator)
        .optimize(SEED, loader.cases())
        .await
        .context("the model-assisted search runs")?;
    common::banner::note(&format!(
        "model-scored result {:.3} after {} round(s) over {} case(s)",
        result.score,
        result.iterations,
        loader.size()
    ));
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    common::banner::title("20 prompt tuning");

    let (loader, train, val) = case_data()?;
    scoring(&loader).await?;
    let (seed_score, tuned_score) = search(&loader).await?;
    training(&train, &val).await?;
    slots(&loader).await?;
    live(&mode, &loader).await?;

    common::banner::step("done");
    common::banner::answer(&format!(
        "mean score {seed_score:.3} (seed) → {tuned_score:.3} (tuned) over {} case(s)",
        loader.size()
    ));
    Ok(())
}
