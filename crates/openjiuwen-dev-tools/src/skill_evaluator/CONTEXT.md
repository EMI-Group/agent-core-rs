# skill_evaluator

## Intent

Evaluation of skills — design review, functional cases, safety gate, and weighted final score — plus an optional rubric-driven LLM-judge path and a runnable evaluation harness.
Ports the artifacts of the Python `openjiuwen.dev_tools.skill_evaluator` (`EvalCase`, `DesignReport`, `FunctionalReport`, `SafetyReport`, `FinalScore`) and the `skills/skill_tester/scripts/run_eval_query.py` single-query runner.
Python graded design/safety by prompting a model with prose rubrics and ran cases through a shell harness; here the default is deterministic structure/assertion scoring, and a model is consulted only when the caller supplies one.
The node owns the data models, the assertion engine, the scoring functions, the judge rubrics and fallbacks, the scripted/model agents and runners, the evaluator, and the harness that loads cases and writes run artifacts.

## API Surface

Deterministic data and scoring (re-exported at `openjiuwen_dev_tools::skill_evaluator`):

- `EvalCase` — `id`, `name`, `case_category`, `prompt`, `expected_output`, `files`, `expectations`; builders `new/with_expectation/with_expectations/with_expected_output/with_category/with_files`; `validate()` requires non-blank id/name/prompt and at least one expectation.
- `CaseStatus` — `Passed` (default) / `Partial` / `Failed`, serialized as `"passed"`/`"partial"`/`"failed"`.
- `Assertion` — `Contains`/`NotContains`/`Equals`/`StartsWith`/`Regex`; `parse(&str)` accepts case-insensitive `kind:` prefixes and treats bare text as `Contains`; `text()` renders canonical `kind:argument`; `check(&str) -> Result<AssertionOutcome>`.
- `AssertionOutcome` — `text`, `passed`, `evidence` (prefix excerpt of at most 200 chars, `…` marked when truncated).
- `ScoreDimension`, `DesignReport`, `CaseSummary`, `CaseResult`, `FunctionalReport`, `SafetyFinding`, `SafetyReport`, `EvaluatorScore`, `EvaluatorScores`, `FinalScore`, `EvaluationReport`.
- `JudgeRecord` — `design_source`/`safety_source` (`crate::judge::VerdictSource`); carried as `EvaluationReport::judge: Option<JudgeRecord>`, serialized only on the judged path.
- `grade_for(f64) -> String` — A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else `F`.
- `score_design(&SkillDefinition) -> DesignReport`, `scan_safety(&SkillDefinition, &[SafetyRule], f64) -> Result<SafetyReport>`, `aggregate(&DesignReport, &FunctionalReport, &SafetyReport, f64, f64) -> FinalScore`, `SafetyRule` (`pattern`, `dimension`, `severity`, `message`).
- `CaseRunner` (`async fn run(&self, &EvalCase, &SkillDefinition) -> Result<String>`), `ScriptedCaseRunner` (`new`, `respond`, `from_expected_outputs`, `responses`), `ModelCaseRunner` (`new(Arc<dyn Model>)`).
- `EvaluatorConfig` — `design_weight` 0.40, `functional_weight` 0.60, `safety_gate_threshold` 0.80, `safety_rules`, `partial_threshold` 0.50.
- `SkillEvaluator` — `new(runner)`, `with_config`, `scripted`, `runner`, `config`, `evaluate`, `evaluate_at`.

LLM judge (`skill_evaluator::judge_scoring`):

- `design_rubric() -> Rubric` — the seven `score_design` dimensions with weight equal to each point maximum (10/25/15/10/15/15/10), `scale = 100.0`.
- `safety_rubric() -> Rubric` — `d1_harmful_instructions` 30, `d2_unsafe_code_execution` 25, `d3_secret_leakage` 25, `d4_prompt_injection` 20, `scale = 100.0`.
- `DesignJudgement { verdict, report, used_fallback }`, `SafetyJudgement { verdict, report, used_fallback }`.
- `SkillJudge` — `new(Arc<dyn Model>)`, `with_system_prompt`, `judge() -> &ModelJudge`, `async design(&SkillDefinition)`, `async safety(&SkillDefinition, &[SafetyRule])`.
- `design_report_from_verdict(&str, &Verdict) -> DesignReport`.
- `SkillEvaluator` additions: `with_judge(Arc<dyn Model>)`, `with_skill_judge(SkillJudge)`, `judge() -> Option<&Arc<SkillJudge>>`, `evaluate_with_judge`, `evaluate_with_judge_at`.

Evaluation harness (`skill_evaluator::eval`):

- `load_cases_json(&str) -> Result<Vec<EvalCase>>`, `async load_cases_file(path) -> Result<Vec<EvalCase>>`.
- `EvalAgent` (`async fn run(&self, prompt: &str) -> Result<String>`), `ModelAgent` (`new`, `with_system_prompt`), `CaseRunnerAgent` (`new(Arc<dyn CaseRunner>, SkillDefinition)`).
- `EvalHarness` — `new(agent)`, `with_skill`, `agent`, `skill`, `async run(&[EvalCase]) -> Result<EvalRun>`, `async run_query(&QueryRequest) -> Result<QueryOutcome>`.
- `EvalRun { cases, summary }` — `to_json`, `to_json_string`, `to_jsonl`, `async write_json`, `async write_jsonl`.
- `QueryRequest { prompt, output_path, files_base_dir, timestamp, max_iterations }`, `QueryOutcome { output_file, report_file, answer }`, `resolve_output_file(&Path, timestamp) -> PathBuf`.

## Evaluation Pipeline

`SkillEvaluator::evaluate` stamps the report with the current UNIX epoch milliseconds (a clock failure falls back to `0`); `evaluate_at` takes an explicit timestamp for deterministic runs.
The skill is validated first, then every case, so the first invalid input fails before any case runs.
Cases run sequentially in input order through the `CaseRunner`; a runner error aborts immediately.
Each expectation is parsed with `Assertion::parse` and checked with `check`; an assertion-parse error is a validation error.
Per-case status is all/none based via `report::status_from`: all assertions passing is `Passed`, none passing is `Failed`, otherwise `Partial`.
The pass rate is `(passed + 0.5 * partial) / total` (partial cases earn half a pass — a deviation from Python's strict counts); an empty case list is `0.0`; `report::summarize` is the single implementation shared by the evaluator and the harness.
`skill_path` is `skills/<name>`, `timestamp` is the supplied milliseconds, and `recommendations` lists the names of failed and partial cases in input order.
Final assembly calls `score_design`, `scan_safety` with the configured rules and gate, and `aggregate` with the configured weights.
`evaluate`/`evaluate_at` stay purely deterministic and always leave `judge: None`.
`evaluate_with_judge` (wall clock) and `evaluate_with_judge_at` (explicit timestamp) require a configured judge and reuse the same functional helper, then judge design and safety and call `aggregate`; they record `JudgeRecord { design_source, safety_source }`.
`partial_threshold` is exposed for parity and future use; per-case status does not consult it.

## Design Rubric

Seven dimensions, each awarded `0`, half the maximum, or the full maximum: `d1_knowledge_ratio` (10), `d2_expert_knowledge_craft` (25), `d3_spec_compliance` (15), `d4_progressive_disclosure` (10), `d5_pattern_freedom_fit` (15), `d6_practical_usability` (15), `d7_output_specification` (10).
The maxima sum to exactly 100, so `design_score` is the raw sum of the awarded points: all-full reads 100.0, half-full 50.0.
`design_score_pct` equals `design_score`; `knowledge_delta = design_score / 100`; `ready_for_functional_testing` is `design_score_pct >= 50.0`.
Safety starts at 100 and loses 25 points per matching rule, clamped at zero; the gate is closed when `score_pct / 100 < gate_threshold` or any finding's severity is exactly `critical`.
`aggregate` weights design at `design_weight` (0.40) and the functional pass rate at `functional_weight` (0.60); gate order is safety, then design readiness, then completion, and `gates_passed` is `["e1_llm_judge", "e2_functional", "e3_safety"]` on completion.

## LLM Judge Path

Both rubrics score every dimension on `0..=100` and weight it by its point maximum, so weights sum to 100 and `Verdict::score * 100.0` is the design score in points.
`SkillJudge::design` asks the model for a structured verdict with the design rubric and falls back to `score_design` on an invalid rubric, a provider error, an unparsable answer, or a verdict failing validation.
The fallback verdict reuses the deterministic points (`design_score / 100`) and carries each dimension's awarded points rescaled to `0..=100`, so provenance is the only visible difference.
`design_report_from_verdict` rebuilds the `DesignReport`: dimension points are `clamp(raw, 0, 100) / 100 * max` summed when the verdict carries known dimensions, otherwise `verdict.score * 100` points are distributed proportionally to the maxima; grade, `knowledge_delta`, and the readiness gate follow the deterministic formulas.
`SkillJudge::safety` always runs `scan_safety` with the fixed 0.80 gate, then asks the model with the safety rubric; the returned report keeps every scan finding, takes `min(model_score_pct, scan_score_pct)`, and closes the gate when the scan was blocked or the effective score is below 0.80.
Hard rule: a safety rule hit can never pass the gate, even when the model returns a perfect verdict; a `critical` finding blocks unconditionally.
`used_fallback` mirrors `verdict.source == VerdictSource::Deterministic`; `evaluate_with_judge*` copies both sources into `EvaluationReport::judge`.
The model only supplies a verdict; the report types, the aggregation formula, and the gates are unchanged from the deterministic path.

## Evaluation Harness

`load_cases_json` accepts a JSON array, `{"cases": [...]}`, or a single case object, deserializes with the tolerant `EvalCase` serde shape, and validates every case; an empty array is a valid empty run.
`load_cases_file` reads asynchronously, parses `.jsonl` as one case per non-blank line (errors name the 1-based line number), sniffs content for a `.json` file that is really JSONL, and reports a missing or blank file as a validation error naming the path.
`EvalAgent` is the one-method seam; `ModelAgent` makes one system + user call per prompt and treats an empty answer as `Error::llm`; `CaseRunnerAgent` adapts a `CaseRunner` plus a `SkillDefinition` by synthesizing a `query` case.
`EvalHarness::run` validates id/name/prompt itself rather than `EvalCase::validate`, so an expected-output-only case is runnable; expectations parse into assertions, and without expectations a normalized (trim + collapse whitespace) expected-output comparison is used; neither source is a validation error.
Cases run sequentially in input order, a runner error aborts the run, and per-case statuses plus `CaseSummary` come from `report::status_from`/`summarize`.
`EvalRun::to_json` emits `{"cases": [...], "summary": {...}}`; each case object uses the wire keys `case_id`, `name`, `status`, `passed`, `reason` (failed assertion texts joined by `"; "`), and `assertions`.
`to_jsonl` emits one compact case object per line with a trailing newline; `write_json`/`write_jsonl` create parent directories with `tokio::fs` and surface I/O failures as `Error::external` naming the path.
`resolve_output_file` mirrors `run_eval_query.py`: an existing directory, a path with no extension, or a trailing `/`/`\` becomes `<dir>/skill_test_result_<timestamp>.txt`; anything else is used as the explicit file.
`EvalHarness::run_query` creates the directories, defaults `files_base_dir` to the output directory (never the environment or cwd), prefixes the prompt with the script's context block (file base dir, output dir, optional max-iterations line), runs the agent once, writes the answer to the output file, and writes a JSON report beside it (extension replaced by `json`) with `prompt`, `output_file`, `report_file`, `files_base_dir`, `timestamp`, `max_iterations`, `answer`.
Timestamps are caller-supplied, no clock or environment reads happen, and relative paths are handed to the OS as given.

## Constraints

- Only files under `skill_evaluator/` are edited here; `src/lib.rs`, `Cargo.toml`, `src/util.rs`, `src/judge/`, `tests/`, `docs/`, and sibling modules belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for blank/invalid input, malformed assertions, malformed case documents, missing scripted responses, missing judge configuration, and invalid rule patterns; `Error::llm` for missing/blank model answers; `Error::external` for serialization and file I/O failures; `skill.to_markdown()`/`validate()` errors propagate unchanged.
- Public data types derive `Debug, Clone, Default, PartialEq, Serialize, Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields; `Assertion` and `CaseStatus` are enums and omit the serde `default` attribute, which serde rejects on enums.
- `ScriptedCaseRunner` and `EvalRun` derive `Debug, Clone, Default, PartialEq`; `ModelCaseRunner`, `ModelAgent`, `CaseRunnerAgent`, and `SkillEvaluator` hold trait objects and derive nothing; `SkillJudge` has a manual `Debug` naming the judge model.
- Async-first: judge calls, case runs, file loading, and report writes are `async`; no `block_on` bridging and no blocking calls on async tasks.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing; no network, environment reads, or wall-clock dependence except the documented `evaluate`/`evaluate_with_judge` timestamps.
- Files ≤ 1000 lines with one responsibility each; `eval.rs` keeps the code and `eval/tests.rs` the harness tests.
- Rustdoc on every public item (workspace `missing_docs`, clippy pedantic + `-D warnings`).

## Design Decisions

- Evidence is a deterministic prefix excerpt of the output rather than a window around the match, so identical inputs produce identical reports.
- `d2` counts distinct `##`/`###` heading texts plus numbered-step lines, so duplicated headings cannot inflate the score; `d4` credits a resource reference by path or last path component; `d6` treats any list item or fenced code block as full and a backtick pair as half; `d7` looks for output/result/format wording before falling back to return/response wording.
- `e1_llm_judge` is always populated because the design rubric, not a model, plays the judge role; the model path replaces its score without changing the report type.
- `SafetyReport::dimensions` stays `None` and serializes as `null`: both the deterministic scan and the model judgement are a penalty/gate tally, not a published dimension rubric; per-dimension model scores live in the `Verdict`.
- `ScriptedCaseRunner` is the strict offline default: an unregistered case id is a validation error, never an empty answer.
- `ModelCaseRunner` and `ModelAgent` make exactly one non-streaming call per case/prompt (no retries), so a scripted run is reproducible.
- The judge rubrics mirror `score_design`/`scan_safety` exactly so a model verdict and the deterministic fallback are interchangeable on the same 0–100 scale.
- Safety combination is deliberately one-directional: the model can lower the score or add no findings, but it cannot remove a rule finding or reopen the gate.
- `JudgeRecord` is optional and omitted from JSON so the deterministic report stays byte-compatible with the Python artifact.
- Serialized shapes keep the Python key names but use percentage points (`81.0`, contributions `32.0`) where the Python skill artifacts use fractions (`0.81`, `0.296`); `DesignReport::knowledge_delta` is an `f64` (Python emits a prose string), `FunctionalReport::timestamp` is epoch milliseconds `i64` (Python emits ISO-8601), and `CaseResult::id` is a `String` (Python uses an integer).
- No per-artifact file writer exists: `final_score.json`/`llm_judge_score.json`/`skill_safety_score.json`/`skill_tester_report.json` are the `final_score`/`design`/`safety`/`functional` members of one `EvaluationReport` emitted by `EvaluationReport::to_json`/`to_json_string`; the only library file writers are `EvalRun::write_json`/`write_jsonl` and `EvalHarness::run_query`.
- `aggregate` always sets `evaluators.e1_llm_judge` to `Some`; the `Option` (serialized `null`) exists only for a future evaluator set.
- The harness reuses `status_from`/`summarize` and the `Assertion` engine instead of re-implementing scoring, so the evaluator and the harness cannot disagree about pass rates.
- `EvalRun`'s wire name `case_id` follows the run-artifact convention while the in-memory `CaseResult` keeps `id`; the mapping is explicit in `EvalRun::to_json`/`to_jsonl`.

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`): assertion, case, report, scoring, runner, judge, evaluator, judge_scoring, and `eval/tests.rs` for the harness (456 crate unit tests today).
- Assertion tests cover all five kinds, mixed-case prefixes, canonical round-trip, pass/fail checks, empty-output and 200-char truncation evidence, char-boundary safety, and serde round-trips.
- Report tests cover the grade matrix, top-level JSON keys, `dimensions: null`, unknown-field tolerance, `status_from`/`summarize`, and the optional `judge` record.
- Scoring tests cover the rich/minimal skill rubric, the seven maxima summing to 100, invalid names, the safety matrix (no rules, soft match, critical block, invalid pattern, unrenderable skill), and all aggregation branches.
- Judge tests (`judge_scoring.rs`) cover rubric names/weights/scale, model success, malformed and failed answers, fallback dimension scores on the 0–100 scale, and the hard safety rule overriding a passing model verdict.
- Evaluator tests cover the deterministic end-to-end run, partial credit, validation/runner error propagation, the critical safety block, judged provenance (model/model, mixed, deterministic/deterministic), the `BLOCKED_AT_SAFETY` override, the no-judge validation error, and judged JSON shape/round-trip.
- Harness tests cover the three JSON shapes, JSONL with line-numbered errors and content sniffing, pass/fail/partial scoring with summary parity, normalized expected-output matching, `ModelAgent` message shape and empty-answer failure, JSON/JSONL byte shape, file writes, and `run_query` path resolution for directory/extension-less/explicit-file cases with written contents.
- Runner tests cover scripted lookup, missing-id validation, `from_expected_outputs`, empty `Default`, and the model runner's single call, message shape, returned text, and empty/whitespace failure.
- Model-driven paths use `ScriptedModel` with queued responses; filesystem paths use `TempWorkspace`; no network, environment reads, or wall-clock dependence; float comparisons use an absolute tolerance (`< 1e-9`).

## Deferred

- The Python `SkillEvaluator`/`create_subagent` LLM orchestration (a `ReActAgent` with sysop/code/shell tools and a subagent tool that shells out to skill directories) — not ported; the Rust harness replaces it with the `EvalAgent` seam and caller-supplied agents/tools.
- Tool-using or shell-executing agents inside `run_query` — the harness runs one prompt through an `EvalAgent` and writes the answer; sandboxed tool execution belongs to the sys-op/harness crates.
- LLM-as-judge scoring of functional case answers — functional scoring stays assertion-based; the judge path covers design and safety only.
- `run_eval_query.py`'s environment-driven model construction (`API_BASE`, `API_KEY`, `MODEL_NAME`, `MODEL_PROVIDER`, `LLM_SSL_VERIFY`, `FILES_BASE_DIR`, `MAX_ITERATIONS`) — configuration stays with the caller, and `files_base_dir`/`max_iterations` are explicit `QueryRequest` fields.

## Routing Table

- `./case.rs` → `EvalCase`, `CaseStatus`, case validation
- `./assertion.rs` → `Assertion` parsing/rendering/checking, `AssertionOutcome`, evidence excerpt
- `./report.rs` → all report/score models, `JudgeRecord`, `EvaluationReport::to_json*`, `grade_for`, shared `status_from`/`summarize`
- `./scoring.rs` → `score_design` rubric, `SafetyRule`/`scan_safety`, `aggregate`, `round2`
- `./runner.rs` → `CaseRunner`, `ScriptedCaseRunner`, `ModelCaseRunner`
- `./judge_scoring.rs` → design/safety rubrics, `SkillJudge`, judgement types, deterministic fallback and the hard safety rule
- `./evaluator.rs` → `EvaluatorConfig`, `SkillEvaluator`, deterministic and judge-driven pipelines
- `./eval.rs` → case loading, `EvalAgent`/`ModelAgent`/`CaseRunnerAgent`, `EvalHarness`, `EvalRun` reports, `run_query`
- `./eval/tests.rs` → harness test matrix (loading, scoring, reports, query path resolution)
- `./mod.rs` → module tree and crate-level re-exports
- `../skill_creator/definition.rs` → `SkillDefinition`, `SkillResource`, `NAME_PATTERN` (sibling — read-only, escalate writes to parent)
- `../../judge/` → `ModelJudge`, `JudgeRequest`, `Rubric`, `Verdict`, `VerdictSource` (sibling module — read-only, escalate writes to parent)
