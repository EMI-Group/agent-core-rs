# openjiuwen-dev-tools / tune / evaluator

## Intent

The evaluator seam of `tune`: turn a candidate prompt plus a case into a scored `EvaluatedCase`, offline or through a model, and keep the tool calls and the cost behind the answer observable.
Splits Python `openjiuwen/dev_tools/tune/evaluator/evaluator.py` into three small seams (runner, scorer, evaluator) so any pipeline can be driven with a scripted runner and a deterministic scorer.
Everything here is offline and deterministic: no environment reads, no network beyond the injected `Model`; the only clock read is the model runner's own latency measurement.

## API Surface

- `PromptRunner` — `async fn run(&self, prompt, case) -> Result<BTreeMap<String, Value>>` plus the provided `async fn run_traced(&self, prompt, case) -> Result<RunOutcome>` whose default reports an empty trace and absent telemetry.
- `render_case_inputs(case)` — `[key]: value` per input, strings bare and other JSON compact; `render_tool_result(&Value)` — string results bare, other JSON compact, `null` when unserializable.
- `ModelPromptRunner` — `new(Arc<dyn Model>)` (no stubs, `DEFAULT_MAX_TOOL_ROUNDS`), `with_stubs(ToolStubs)`, `with_max_tool_rounds(usize)`, accessors `model()`/`stubs()`/`max_tool_rounds()`.
  `run_traced` declares `Case::tools()` in the request, runs the model's tool calls against the stubs in order, records every call (failures included) in a `ToolTrace`, feeds each outcome back as a `Role::Tool` message, and accepts the model's text once it stops calling tools or the round budget is exhausted.
  `run_traced` also measures `RunTelemetry`: the wall-clock latency of the whole run and the prompt/completion tokens summed over every model call; cost stays `None`.
- `DEFAULT_MAX_TOOL_ROUNDS = 8` — the default tool-round budget.
- `ScriptedPromptRunner` — canned answers keyed by case id, optional fallback, recorded `PromptCall`s, `Error::not_found` for an unscripted case; runs no tools and reports absent telemetry.
- `CaseScorer` — `score(case, answer)` plus the provided `score_with_trace(case, answer, trace)` whose default ignores the trace.
- `LabelScorer` — the deterministic default (bands documented in the parent module); `score_pair`/`tokens` helpers.
- `ToolUsageScorer` — multiplies an inner score (`LabelScorer` by default) by `ToolTrace::coverage` of the required names: `new(Vec<String>)`, `from_case(&Case)`, `with_inner(Box<dyn CaseScorer>)`, `required()`, `inner()`, `missing(&ToolTrace)`, inherent `score`/`score_with_trace` plus the `CaseScorer` impl, `Debug` showing names only.
- `CaseEvaluator` — `evaluate(prompt, case)`, the provided `evaluate_run(prompt, case) -> Result<EvaluatedRun>`, the provided `evaluate_traced(prompt, case) -> Result<(EvaluatedCase, ToolTrace)>`, the provided `score_cases`, the provided `score_cases_traced` returning the mean plus per-case `(EvaluatedCase, ToolTrace)` pairs, and the provided `score_runs` returning the mean plus per-case `EvaluatedRun`s (`(0.0, vec![])` for no cases).
- `EvaluatedRun` — `evaluated`, `trace`, `telemetry` (public, serde `snake_case` with per-field defaults), `new(evaluated, trace, telemetry)`, accessors `score()`, `answer()`, `used_tools()`, `is_empty_telemetry()`, and a `Default` that is an empty run over a placeholder case.
- `tool_usage(&[(EvaluatedCase, ToolTrace)]) -> BTreeMap<String, usize>` — summed per-tool call counts.
- `RunnerEvaluator` — any `PromptRunner` plus any `CaseScorer`; `evaluate_run` is the single run (one runner call), `evaluate` returns its `evaluated`, and the inherited `evaluate_traced`/`score_cases_traced` drop the telemetry.
- `ModelEvaluator` — `ModelPromptRunner` + `LabelScorer` presets: `new(model)`, `with_stubs(model, stubs)`, `with_scorer`, `inner()`, forwarding `evaluate` and `evaluate_run` to the inner evaluator.
- `DeterministicEvaluator`/`DeterministicScore` — the offline scorer of placeholder preservation, structure, coverage, and hygiene; it runs no model and reports absent telemetry.

## Constraints

- Public items carry `///` docs, one sentence per line; unreachable or private items are linked as plain code spans.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for a zero round budget, a blank required tool name, or an undeclarable tool; `Error::llm` for an answerless completion; `Error::not_found` for an unscripted case or an unregistered tool.
- Only model/transport errors propagate out of `run_traced`; a failing stub, an unknown tool, and malformed call arguments are recorded in the trace and fed back to the model instead.
- The round budget is checked before the first model call, so `max_tool_rounds == 0` never reaches the provider.
- `EvaluatedCase` carries no trace or telemetry field: they travel through `evaluate_run`/`score_runs` (as `EvaluatedRun`) and, telemetry-free, through `run_traced`/`evaluate_traced`/`score_cases_traced`.
- A run measures latency and input/output tokens only: cost is never inferred from tokens, and absent values mean "not reported" rather than zero.
- Absent telemetry keeps downstream rewards constant: the prompt-search latency, token-usage, and cost components read a caller-supplied `Execution`, so a scripted or deterministic evaluator leaves those components at their constant values.
- Every new provided method delegates in one direction only — `evaluate_traced` goes through `evaluate_run`, so an implementor that overrides only `evaluate_run` is honoured and no evaluation makes two runner calls.
- `ScriptedPromptRunner` and custom runners keep compiling because every traced and telemetry-carrying method is provided.
- Deterministic behaviour for a `LabelScorer` over a scripted runner is unchanged from the untraced seam.

## Design Decisions

- The trace and the telemetry are return values rather than fields, so scoring stays a pure `(case, answer)` function and existing reports keep their JSON shape.
- `score_with_trace` is a provided trait method, so a scorer adds trace awareness without breaking the trait's implementors.
- `ToolUsageScorer` scales rather than vetoes the inner score, and an empty requirement list is a documented no-op (it scores the inner result unchanged).
- Tool failures are data, not control flow: the model is told the error text and may recover within the budget, which keeps a mis-declared tool visible in the trace instead of aborting a tuning run.
- `ToolStubs` is injected into the runner and never constructed from configuration, so tests pass deterministic fixtures and no environment is read.
- `EvaluatedRun` is the one carrier of "score + trace + cost", so `EvaluatedCase` keeps Python's field names and no report gains a telemetry column implicitly.
- Latency is measured with `std::time::Instant` around the whole run (both the answered and the truncated return), and tokens are summed over every model call, because a multi-round tool loop costs more than its last completion reports.
- Cost stays absent in the model runner: pricing depends on the caller's provider contract, so only a caller that knows the price sets `RunTelemetry::with_cost`.

## Test Strategy

- `tests.rs` holds the suite in `#[cfg(test)]`: fixed case ids, `ScriptedModel`/`MockCompletion`/`MockToolCall` for model paths, `ToolStub`/`ToolStubs` fixtures, no network, no environment reads, no wall-clock reads.
- Tool paths assert both the outcome and the request: the recorded `ModelCall` messages, tool declarations, call ids, and the trace records.
- Coverage: declarations reaching the request, tool-less cases, single and parallel calls, unknown tools, failing stubs, the zero-budget rejection, the looping-model bound, `ToolUsageScorer` bands/reasons/nesting, `RunnerEvaluator` trace parity, `ModelPromptRunner` execution through `RunnerEvaluator`, and `ModelEvaluator::with_stubs`.
- `telemetry_tests.rs` (declared from `tests.rs` via `#[path]`) covers the cost path in `#[cfg(test)]`: the model runner's measured latency (only asserted finite and non-negative) and exact token sums from scripted `Usage`s across a two-call tool loop and a truncated loop, the zero-usage response, absent telemetry for `ScriptedPromptRunner`/the provided `run_traced` default/`DeterministicEvaluator`, `evaluate_run`'s default, an `evaluate_run`-only override being honoured through `evaluate_traced`, `RunnerEvaluator`/`ModelEvaluator` forwarding the telemetry with one model call per evaluation, `score_runs` (mean, empty slice, parity with `score_cases`/`score_cases_traced`), and `EvaluatedRun`'s serde round-trip with per-field defaulting, unknown-field tolerance, and accessors.

## Routing Table

- `./mod.rs` → module docs, private submodules, public re-exports
- `./runner.rs` → `PromptRunner`, `ModelPromptRunner` tool loop and telemetry measurement, `render_case_inputs`, `ScriptedPromptRunner`, `DEFAULT_MAX_TOOL_ROUNDS`
- `./scorer.rs` → `CaseScorer`, `LabelScorer`, `ToolUsageScorer`, band helpers
- `./seam.rs` → `CaseEvaluator` (`evaluate`, `evaluate_run`, `evaluate_traced`, `score_cases`, `score_cases_traced`, `score_runs`), `EvaluatedRun`, `RunnerEvaluator`, `ModelEvaluator`, `tool_usage`, score clamping
- `./deterministic.rs` → `DeterministicEvaluator`, `DeterministicScore`
- `./tests.rs` → the evaluator test suite, and the declaration of the telemetry suite
- `./telemetry_tests.rs` → the run-telemetry and `EvaluatedRun` suite
- `../tool.rs` → `ToolStub`/`ToolStubs`, `ToolTrace`/`ToolCallRecord`, `RunTelemetry`, `RunOutcome` (outside this node — read-only, escalate writes to the crate root)
- `../CONTEXT.md` → the parent `tune` node (read-only here)
