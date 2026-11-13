# evaluator

## Intent
The scoring pipeline: agent-execution seam, deterministic metrics, the model-backed LLM-as-judge, evaluator traits, and evaluation reports.
Ports Python `openjiuwen.agent_evolving.evaluator` (`BaseEvaluator`, `MetricEvaluator`, `DefaultEvaluator`, `metrics/`, `templates.LLM_METRIC_TEMPLATE`) onto this crate's own seams instead of transliterating the Python prompt-string plumbing.

## API Surface
### Seams and driver
- `Evaluator` (async trait) — `name`, `evaluate(case, prediction)`, provided `batch_evaluate` with length-mismatch and `1..=20` parallel validation via `futures::buffered`.
- `CaseRunner` (async trait) — produces a prediction `Value` for a case; `AgentCaseRunner` (wraps `Arc<dyn Agent>`, serializes the full `AgentOutput`), `MapCaseRunner` (deterministic `case_id -> Value` map).
- `EvaluationReport` — `run_id`, `num_cases`, `mean_score`, `min_score`, `max_score`, `per_metric`, `cases`; `from_evaluated`/`from_cases`, empty input yields a zeroed report.
- `EvaluationConfig { num_parallel, max_cases }`, `evaluate_cases(run_id, runner, evaluator, cases, cancel, config)` — bounded-parallel run/score/report; the caller-supplied `run_id` keeps reports deterministic.

### Metrics
- `Metric` trait — `name`, `higher_is_better` (default `true`), `compute(prediction, label) -> MetricValue`; `MetricValue` is `Scalar(f64)` or `Named(BTreeMap<String, f64>)`.
- `ExactMatchMetric { normalize }` — compares the answer projection (label key `answer`, single-key maps, else the full JSON object); normalization trims, lowercases and collapses whitespace.
- `LlmAsJudgeMetric` — the offline counterpart of the judge: parses a prediction that already is a verdict with `JudgeVerdict::parse`, and falls back to `ExactMatchMetric` when the prediction is not verdict-shaped.
- `MetricEvaluator { metrics, aggregate }` — per-metric values merged into `per_metric`; `Aggregate { Mean, First }`; numeric coercion never raises (bad values become `0.0`).

### Judge subsystem (`./judge/`)
- `JudgeRubric` / `JudgeCriterion` — criteria (name, description, weight) plus `max_score` and `passing_score`, as pure data; `default_rubric`, `single`, `new`, accessors, `validate`.
- `build_judge_messages(case, prediction, rubric, max_prediction_chars)` — the deterministic `[system, user]` prompt: rubric rendering, sorted JSON projections, and an explicit `... [truncated N chars]` marker when the prediction is cut.
- `JudgeVerdict` / `JudgeVerdict::parse` — the tolerant response reader: balanced `{...}` objects (including fenced and prose-embedded), score aliases `score`/`total_score`/`rating`, numeric strings (`"8"`, `"8/10"`, `"8 out of 10"`, `"80%"`), per-criterion aliases `per_criterion`/`criteria`/`scores`, reasoning aliases, rubric-weighted fallback, boolean `result`/`passed` shape, then a free-text `score: X` scan; everything is normalised to `0.0..=1.0` while `raw_score` keeps the model's scale.
- `LlmJudge` — the live path: `new` validates the rubric, `with_attempts` (clamped to `1..=3`, retries only on unparseable responses), `with_max_prediction_chars` (`0` disables truncation); `judge(case, prediction)` calls the model and parses the reply.
- `LlmJudgeEvaluator` — scores with the judge and falls back to a deterministic `Metric`; every fallback states whether no judge was configured or which error occurred, and records the fallback metric in `per_metric`.
- `DefaultEvaluator` — the Python entry point: `new(metric)`, `exact_match`, `with_judge`, `with_model` (wrapped per evaluation), `with_rubric`, `with_fallback`, `with_request`; judge failure is never swallowed.

## Constraints
- Deterministic metrics never touch the network; the only I/O is a model call from `LlmJudge`/`DefaultEvaluator::with_model`, and unit tests substitute a scripted model.
- `Metric::compute` returns `Result`; `MetricEvaluator` coerces unusable values to `0.0` rather than erroring.
- `JudgeRubric`/`JudgeCriterion` validation failures use `Error::validation`, and `LlmJudge::new` validates once instead of per case.
- `EvaluationReport` carries a caller-supplied `run_id`: no UUID source, no clock.

## Design Decisions
- Prediction production (`CaseRunner`) and scoring (`Evaluator`) are separate traits so the driver works with fake agents and the trainer reuses the same report shape.
- `ExactMatchMetric` accepts the three label projections Python callers use in practice, so dataset files and agent outputs need no adapter.
- The rubric is data, not prose inside a Python template string, so a caller can reweight or extend the judge without editing prompts.
- `LlmAsJudgeMetric` scores already-produced verdicts, which keeps recorded model responses replayable offline; the async live path lives in the judge module.
- The judge prompt is a pure function of its arguments (sorted keys, explicit truncation marker), so prompt regressions are unit-testable without a model.
- Fallbacks are visible: the reason names the failure and `per_metric` names the fallback metric, rather than silently degrading.
- `AgentCaseRunner` serializes the full `AgentOutput`; callers pairing it with `ExactMatchMetric` should project `prediction["output"]`, since the metric only auto-projects `answer`-keyed or single-key predictions.

## Routing Table
- `./mod.rs` → `Evaluator`, `EvaluationReport`, `EvaluationConfig`, `evaluate_cases`, re-exports
- `./metrics.rs` → `Metric`, `MetricValue`, `ExactMatchMetric`, `LlmAsJudgeMetric`, `MetricEvaluator`, `Aggregate`, `DefaultEvaluator`
- `./runner.rs` → `CaseRunner`, `AgentCaseRunner`, `MapCaseRunner`
- `./judge/` (`mod.rs`, `rubric.rs`, `prompt.rs`, `verdict.rs`, `live.rs`) → rubric, prompt, verdict parsing, `LlmJudge`, `LlmJudgeEvaluator`
