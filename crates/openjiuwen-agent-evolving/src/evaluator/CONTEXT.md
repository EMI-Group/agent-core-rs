# evaluator

## Intent

Scoring pipeline: agent-execution seam, deterministic metrics, evaluator trait, and evaluation reports.
Ports Python `openjiuwen.agent_evolving.evaluator` (`BaseEvaluator`, `MetricEvaluator`, `DefaultEvaluator`, metrics) and replaces the Python model-judge paths with documented stubs.

## API Surface

- `Evaluator` — async trait: `name`, `evaluate`, default `batch_evaluate` (length-mismatch and `1..=20` parallel validation, `futures::buffered`).
- `CaseRunner` — async trait producing a prediction `Value` for a case; `AgentCaseRunner` (wraps `Arc<dyn Agent>`), `MapCaseRunner` (deterministic map).
- `Metric`, `MetricValue` (`Scalar`, `Named`), `ExactMatchMetric`, `LlmAsJudgeMetric` (stub), `MetricEvaluator`, `Aggregate`.
- `DefaultEvaluator` — API-shaped stub returning `Error::Unsupported`.
- `EvaluationReport` — `run_id`, `num_cases`, `mean_score`, `min_score`, `max_score`, `per_metric`, `cases`.
- `EvaluationConfig`, `evaluate_cases(run_id, ...)` — bounded-parallel evaluation over a runner and evaluator; the caller-supplied `run_id` keeps reports deterministic.

## Constraints

- All scoring is deterministic and offline; no model calls.
- `Metric::compute` returns `openjiuwen_common::error::Result`; numerical coercion in `MetricEvaluator` falls back to `0.0` instead of erroring.
- `LlmAsJudgeMetric`/`DefaultEvaluator` must return `Error::Unsupported` naming the deferred live-model dependency.

## Design Decisions

- Prediction production and scoring are separate traits so `evaluate_cases` works with fake agents and the trainer can reuse the same report shape.
- `ExactMatchMetric` accepts `answer`-keyed labels, single-key label maps, and full-object labels, so dataset files and agent outputs need no adapter.

## Routing Table

- `./mod.rs` → `Evaluator`, `CaseRunner` re-export, `EvaluationReport`, `EvaluationConfig`, `evaluate_cases`
- `./metrics.rs` → `Metric`, `MetricValue`, `ExactMatchMetric`, `LlmAsJudgeMetric`, `MetricEvaluator`, `Aggregate`
- `./runner.rs` → `CaseRunner`, `AgentCaseRunner`, `MapCaseRunner`
