# openjiuwen-dev-tools / tune / evaluator

## Intent

The evaluator seam of `tune`: turn a candidate prompt plus a case into a scored `EvaluatedCase`, offline or through a model, and keep the tool calls behind the answer observable.
Splits Python `openjiuwen/dev_tools/tune/evaluator/evaluator.py` into three small seams (runner, scorer, evaluator) so any pipeline can be driven with a scripted runner and a deterministic scorer.
Everything here is offline and deterministic: no environment reads, no wall-clock dependence, no network beyond the injected `Model`.

## API Surface

- `PromptRunner` — `async fn run(&self, prompt, case) -> Result<BTreeMap<String, Value>>` plus the provided `async fn run_traced(&self, prompt, case) -> Result<RunOutcome>` whose default reports an empty trace.
- `render_case_inputs(case)` — `[key]: value` per input, strings bare and other JSON compact; `render_tool_result(&Value)` — string results bare, other JSON compact, `null` when unserializable.
- `ModelPromptRunner` — `new(Arc<dyn Model>)` (no stubs, `DEFAULT_MAX_TOOL_ROUNDS`), `with_stubs(ToolStubs)`, `with_max_tool_rounds(usize)`, accessors `model()`/`stubs()`/`max_tool_rounds()`.
  `run_traced` declares `Case::tools()` in the request, runs the model's tool calls against the stubs in order, records every call (failures included) in a `ToolTrace`, feeds each outcome back as a `Role::Tool` message, and accepts the model's text once it stops calling tools or the round budget is exhausted.
- `DEFAULT_MAX_TOOL_ROUNDS = 8` — the default tool-round budget.
- `ScriptedPromptRunner` — canned answers keyed by case id, optional fallback, recorded `PromptCall`s, `Error::not_found` for an unscripted case; runs no tools.
- `CaseScorer` — `score(case, answer)` plus the provided `score_with_trace(case, answer, trace)` whose default ignores the trace.
- `LabelScorer` — the deterministic default (bands documented in the parent module); `score_pair`/`tokens` helpers.
- `ToolUsageScorer` — multiplies an inner score (`LabelScorer` by default) by `ToolTrace::coverage` of the required names: `new(Vec<String>)`, `from_case(&Case)`, `with_inner(Box<dyn CaseScorer>)`, `required()`, `inner()`, `missing(&ToolTrace)`, inherent `score`/`score_with_trace` plus the `CaseScorer` impl, `Debug` showing names only.
- `CaseEvaluator` — `evaluate(prompt, case)`, the provided `evaluate_traced(prompt, case) -> Result<(EvaluatedCase, ToolTrace)>`, the provided `score_cases`, and the provided `score_cases_traced` returning the mean plus per-case `(EvaluatedCase, ToolTrace)` pairs (`(0.0, vec![])` for no cases).
- `tool_usage(&[(EvaluatedCase, ToolTrace)]) -> BTreeMap<String, usize>` — summed per-tool call counts.
- `RunnerEvaluator` — any `PromptRunner` plus any `CaseScorer`; `evaluate` delegates to `evaluate_traced`, which scores with `score_with_trace` and clamps the score.
- `ModelEvaluator` — `ModelPromptRunner` + `LabelScorer` presets: `new(model)`, `with_stubs(model, stubs)`, `with_scorer`, `inner()`, and a traced override that delegates to the inner runner.
- `DeterministicEvaluator`/`DeterministicScore` — the offline scorer of placeholder preservation, structure, coverage, and hygiene.

## Constraints

- Public items carry `///` docs, one sentence per line; unreachable or private items are linked as plain code spans.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for a zero round budget, a blank required tool name, or an undeclarable tool; `Error::llm` for an answerless completion; `Error::not_found` for an unscripted case or an unregistered tool.
- Only model/transport errors propagate out of `run_traced`; a failing stub, an unknown tool, and malformed call arguments are recorded in the trace and fed back to the model instead.
- The round budget is checked before the first model call, so `max_tool_rounds == 0` never reaches the provider.
- `EvaluatedCase` carries no trace field: the trace travels only through `run_traced`/`evaluate_traced`/`score_cases_traced`.
- `ScriptedPromptRunner` and custom runners keep compiling because every traced method is provided.
- Deterministic behaviour for a `LabelScorer` over a scripted runner is unchanged from the untraced seam.

## Design Decisions

- The trace is a return value rather than a field, so scoring stays a pure `(case, answer)` function and existing reports keep their JSON shape.
- `score_with_trace` is a provided trait method, so a scorer adds trace awareness without breaking the trait's implementors.
- `ToolUsageScorer` scales rather than vetoes the inner score, and an empty requirement list is a documented no-op (it scores the inner result unchanged).
- Tool failures are data, not control flow: the model is told the error text and may recover within the budget, which keeps a mis-declared tool visible in the trace instead of aborting a tuning run.
- `ToolStubs` is injected into the runner and never constructed from configuration, so tests pass deterministic fixtures and no environment is read.

## Test Strategy

- `tests.rs` holds the whole suite in `#[cfg(test)]`: fixed case ids, `ScriptedModel`/`MockCompletion`/`MockToolCall` for model paths, `ToolStub`/`ToolStubs` fixtures, no network, no environment reads, no wall-clock reads.
- Tool paths assert both the outcome and the request: the recorded `ModelCall` messages, tool declarations, call ids, and the trace records.
- Coverage: declarations reaching the request, tool-less cases, single and parallel calls, unknown tools, failing stubs, the zero-budget rejection, the looping-model bound, `ToolUsageScorer` bands/reasons/nesting, `RunnerEvaluator` trace parity, `ModelPromptRunner` execution through `RunnerEvaluator`, and `ModelEvaluator::with_stubs`.

## Routing Table

- `./mod.rs` → module docs, private submodules, public re-exports
- `./runner.rs` → `PromptRunner`, `ModelPromptRunner` tool loop, `render_case_inputs`, `ScriptedPromptRunner`, `DEFAULT_MAX_TOOL_ROUNDS`
- `./scorer.rs` → `CaseScorer`, `LabelScorer`, `ToolUsageScorer`, band helpers
- `./seam.rs` → `CaseEvaluator`, `RunnerEvaluator`, `ModelEvaluator`, `tool_usage`, score clamping
- `./deterministic.rs` → `DeterministicEvaluator`, `DeterministicScore`
- `./tests.rs` → the evaluator test suite
- `../tool.rs` → `ToolStub`/`ToolStubs`, `ToolTrace`/`ToolCallRecord`, `RunOutcome` (outside this node — read-only, escalate writes to the crate root)
- `../CONTEXT.md` → the parent `tune` node (read-only here)
