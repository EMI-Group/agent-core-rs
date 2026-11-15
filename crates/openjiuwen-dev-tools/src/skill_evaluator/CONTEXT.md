# skill_evaluator

## Intent

Evaluation of skills — design review, functional cases, safety gate, weighted final score — plus an optional rubric-driven LLM-judge path, a runnable evaluation harness, and the four Python pipeline artifacts.
Ports the artifacts of the Python `openjiuwen.dev_tools.skill_evaluator` (`EvalCase`, `DesignReport`, `FunctionalReport`, `SafetyReport`, `FinalScore`) and the `skills/skill_tester/scripts/run_eval_query.py` single-query runner.
The Python skill prompts (`skill_eval_pipeline`, `skill_judge`, `skill_safety_judge`, `skill_tester`) document the JSON key sets; no Python code writes those files, so the typed structs here *are* the contract and match those key sets verbatim.
Python graded design/safety by prompting a model with prose rubrics and ran cases through a shell harness; here the default is deterministic structure/assertion scoring, and a model is consulted only when the caller supplies one.
The node owns the data models, the unit convention, the assertion engine, the scoring functions and gates, the judge rubrics/fallbacks, the scripted/model agents and runners, the evaluator, the artifact writers, and the harness that loads cases and writes run artifacts.

## API Surface

Deterministic data and scoring (re-exported at `openjiuwen_dev_tools::skill_evaluator`):

- `EvalCase` — `id`, `name`, `case_category`, `prompt`, `expected_output`, `files`, `expectations`; builders `new/with_expectation/with_expectations/with_expected_output/with_category/with_files`; `validate()` requires non-blank id/name/prompt and at least one expectation.
- `CaseStatus` — `Passed` (default) / `Partial` / `Failed`, serialized as `"passed"`/`"partial"`/`"failed"`.
- `Assertion` — `Contains`/`NotContains`/`Equals`/`StartsWith`/`Regex`; `parse(&str)` accepts case-insensitive `kind:` prefixes and treats bare text as `Contains`; `text()` renders canonical `kind:argument`; `check(&str) -> Result<AssertionOutcome>`.
- `AssertionOutcome` — `text`, `passed`, `evidence` (prefix excerpt of at most 200 chars, `…` marked when truncated).
- `ScoreDimension` — `score`/`max` in **points** (0–100).
- `DesignReport` — `skill_name`, `design_score` (points 0–100), `design_score_max` (100.0), `design_score_pct` (**fraction**), `grade`, `knowledge_delta` (numeric fraction), `ready_for_functional_testing`, `dimensions` (seven ids, maxima 10/25/15/10/15/15/10).
- `CaseSummary` — `total`, `passed`, `partial`, `failed`, `pass_rate` (**fraction**).
- `CaseResult`, `FunctionalReport` (`skill_name`, `skill_path`, `timestamp`, `summary`, `results`, `recommendations`), `SafetyFinding` (`dimension`, `severity`, `message`), `SafetyReport`, `EvaluatorScore`, `EvaluatorScores`, `FinalScore`, `EvaluationReport`, `JudgeRecord`.
- `SafetyReport` — `skill_name`, `score_pct` (**fraction**), `gate_threshold` (fraction, default 0.80), `blocked`, `auto_fail: Option<AutoFailCode>`, `dimensions: Option<BTreeMap<String, ScoreDimension>>` (serialized `null` only on auto-fail), `findings`.
- `AutoFailCode` — `BypassSafety` (AF1), `HardcodedCredentials` (AF2), `UndeclaredExfiltration` (AF3), `TargetingRealIndividuals` (AF4), `InstructingDeception` (AF5); `code()`, `description()`, `parse(&str)` (bare code or `"AF<n>: …"`), `Display` = `"AF<n>: <description>"`; serde is that Display string.
- `FinalScore` — `skill_name`, `pipeline_status`, `final_score_pct` (**fraction**), `grade`, `evaluators`, `gates_passed`, `recommendation`.
- Constants — `PIPELINE_COMPLETED` (`"COMPLETED"`), `PIPELINE_BLOCKED_AT_E1` (`"BLOCKED_AT_E1"`), `PIPELINE_BLOCKED_AT_E3` (`"BLOCKED_AT_E3"`), `GATE_E1` (`"e1_llm_judge"`), `GATE_E2` (`"e2_functional"`), `GATE_E3` (`"e3_safety"`).
- Unit-named accessors — `DesignReport::score_points()`/`score_fraction()`/`knowledge_delta_label()`, `FinalScore::score_fraction()`/`is_completed()`, `SafetyReport::score_fraction()`/`passes_gate()`, `EvaluatorScore::weighted(score_fraction, weight)`.
- `grade_for(f64)` — A ≥ 0.90, B ≥ 0.80, C ≥ 0.70, D ≥ 0.60, else F; the argument is the **fraction**.
- `score_design(&SkillDefinition) -> DesignReport`, `scan_safety(&SkillDefinition, &[SafetyRule], f64) -> Result<SafetyReport>`, `aggregate(&DesignReport, &FunctionalReport, &SafetyReport, f64, f64) -> FinalScore`.
- `SafetyRule` — `pattern`, `dimension`, `severity`, `message`, `auto_fail: Option<AutoFailCode>`; `SafetyRule::new(pattern, dimension, severity, message)` and `with_auto_fail(code)`.
- `CaseRunner` (`async fn run(&self, &EvalCase, &SkillDefinition) -> Result<String>`), `ScriptedCaseRunner`, `ModelCaseRunner`.
- `EvaluatorConfig` — `design_weight` 0.40, `functional_weight` 0.60, `safety_gate_threshold` 0.80, `safety_rules`, `partial_threshold` 0.50 (parity only; status derivation does not consult it).
- `safety_gate_threshold` reaches only the deterministic `scan_safety` path; `evaluate_with_judge*` pins `judge_scoring::SAFETY_GATE_THRESHOLD` (0.80), so a different configured gate has no effect on a judged run and the report's `gate_threshold` shows the pinned value.
- `SkillEvaluator` — `new(runner)`, `with_config`, `scripted`, `runner`, `config`, `evaluate`, `evaluate_at`, plus the judged path below.

LLM judge (`skill_evaluator::judge_scoring`):

- `design_rubric()` — the seven `score_design` dimensions weighted by their point maxima (10/25/15/10/15/15/10), `scale = 100.0`.
- `safety_rubric()` — the five canonical ids/maxima: `d1_harmful_instruction` 30, `d2_scope_containment` 25, `d3_data_handling` 20, `d4_injection_resilience` 15, `d5_guardrails` 10; `scale = 100.0`.
- `DesignJudgement { verdict, report, used_fallback }`, `SafetyJudgement { verdict, report, used_fallback }`.
- `SkillJudge` — `new(Arc<dyn Model>)`, `with_system_prompt`, `judge() -> &ModelJudge`, `async design(&SkillDefinition)`, `async safety(&SkillDefinition, &[SafetyRule])`.
- `design_report_from_verdict(&str, &Verdict) -> DesignReport`.
- `SkillEvaluator` additions: `with_judge`, `with_skill_judge`, `judge()`, `evaluate_with_judge`, `evaluate_with_judge_at`.

Evaluation harness (`skill_evaluator::eval`):

- `load_cases_json`, `async load_cases_file`, `EvalAgent`, `ModelAgent`, `CaseRunnerAgent`, `EvalHarness::{new, with_skill, agent, skill, run, run_query}`.
- `EvalRun { cases, summary }` — `to_json`, `to_json_string`, `to_jsonl`, `async write_json`, `async write_jsonl` (run-artifact wire keys `case_id`/`name`/`status`/`passed`/`reason`/`assertions`), plus `report(skill_name, skill_path, timestamp) -> Result<serde_json::Value>` and `async write_report(...)` for the Python `skill_tester_report.json` key set.
- `QueryRequest { prompt, output_path, files_base_dir, timestamp, max_iterations }`, `QueryOutcome { output_file, report_file, answer }`, `resolve_output_file(&Path, timestamp) -> PathBuf`.

Artifacts (`skill_evaluator::artifacts`):

- Constants `FINAL_SCORE_ARTIFACT` (`final_score.json`), `DESIGN_ARTIFACT` (`llm_judge_score.json`), `SAFETY_ARTIFACT` (`skill_safety_score.json`), `FUNCTIONAL_ARTIFACT` (`skill_tester_report.json`).
- `ArtifactPaths { dir, design, safety, functional, final_score: Option<PathBuf> }`.
- `EvaluationReport::{write_design_report, write_safety_report, write_functional_report, write_final_score, write_artifacts}`; each is async, creates the directory, rejects a path that exists but is not a directory with `Error::validation`, and writes pretty JSON from the existing serializers (`write_final_score` returns `None` and writes nothing unless the pipeline completed).

## Units

Points (0–100) live only in `design_score`, `design_score_max`, `ScoreDimension::score`, and `ScoreDimension::max`.
Every `*_pct` field (`design_score_pct`, `final_score_pct`, `safety.score_pct`, `EvaluatorScore::score_pct`), every `contribution`, every `gate_threshold`, `safety_gate_threshold`, `pass_rate`, and every judge `Verdict::score` is a fraction of `1.0`.
`grade_for` takes the fraction and applies the Python bands (0.90/0.80/0.70/0.60).
`design_score_pct = round4(design_score / 100.0)` and contributions are `round4(weight · score_fraction)`, so `0.4 · 0.74` reads exactly `0.296`.
The helpers `round2`, `round4`, `fraction_from_points`, `points_from_fraction`, and `DESIGN_GATE_THRESHOLD` (0.60) live in `scoring.rs`; points use two decimals, fractions four.

## Evaluation Pipeline

`SkillEvaluator::evaluate` stamps the report with the current UNIX epoch milliseconds; `evaluate_at` takes an explicit timestamp for deterministic runs.
The skill is validated first, then every case, so the first invalid input fails before any case runs.
Cases run sequentially in input order through the `CaseRunner`; a runner error aborts immediately.
Each expectation is parsed with `Assertion::parse` and checked with `check`; an assertion-parse error is a validation error.
Per-case status comes from `report::status_from`: all assertions passing is `Passed`, none passing is `Failed`, otherwise `Partial`.
The pass rate is **`passed / total`** (partial cases are not counted; empty list → `0.0`), matching the Python worked example 3 passed / 1 partial / 0 failed → `0.75`; `report::summarize` is the single implementation shared by the evaluator and the harness.
`skill_path` is `skills/<name>` and `timestamp` is the supplied epoch milliseconds.
`recommendations` lists the names of failed and partial cases in input order.
Final assembly calls `score_design`, `scan_safety` with the configured rules and gate, and `aggregate` with the configured weights.
`evaluate`/`evaluate_at` stay purely deterministic and always leave `judge: None`.
`evaluate_with_judge` (wall clock) and `evaluate_with_judge_at` (explicit timestamp) require a configured judge and reuse the same functional helper, then judge design and safety and call `aggregate`; they record `JudgeRecord { design_source, safety_source }`.

## Final Score and Gates

Gates run in Python's order: **E1 → E3 → aggregate**.
E1 is the design gate: `design_score_pct < 0.60` stops the pipeline with `pipeline_status = BLOCKED_AT_E1`.
E3 is the safety gate (the only gate-role evaluator in the registry): a `blocked` safety report stops the pipeline with `BLOCKED_AT_E3`.
`ready_for_functional_testing` uses the same 0.60 fraction, so the flag and the E1 gate cannot disagree.
A blocked run reports `final_score_pct = 0.0`, `grade = "F"`, an empty `gates_passed`, and a prose `recommendation` naming the blocking gate; both evaluator entries stay populated so the author sees the scores that led to the block.
A completed run reports `final_score_pct = round4(0.40 · design_fraction + 0.60 · pass_rate)` and `gates_passed = ["e3_safety"]`, the one gate-role evaluator that passed.
The completion recommendation is prose keyed on the fraction: ≥ 0.80 `ready for release`, ≥ 0.60 `usable, refine failing cases`, otherwise `not ready for release`.

## Design Rubric

Seven dimensions, each awarded `0`, half the maximum, or the full maximum: `d1_knowledge_ratio` (10), `d2_expert_knowledge_craft` (25), `d3_spec_compliance` (15), `d4_progressive_disclosure` (10), `d5_pattern_freedom_fit` (15), `d6_practical_usability` (15), `d7_output_specification` (10).
The maxima sum to exactly 100, so `design_score` is the raw sum of the awarded points: all-full reads 100 points (`design_score_pct = 1.0`), half-full 50 points (`0.5`).
`knowledge_delta` is the numeric `round2(design_score / 100.0)`; `knowledge_delta_label()` additionally renders Python's D1 band label from the `d1_knowledge_ratio` points (≥ 9 `Excellent`, ≥ 6 `Good`, ≥ 3 `Mediocre`, else `Poor`).
Safety starts at 100 points and loses 25 points per matching rule, clamped at zero; the gate is closed when the score fraction is below `gate_threshold` or any finding's severity is exactly `critical`.

## Safety Contract

The canonical safety dimensions live once, in `report.rs`: `d1_harmful_instruction`/30, `d2_scope_containment`/25, `d3_data_handling`/20, `d4_injection_resilience`/15, `d5_guardrails`/10 (Python's singular `d1_harmful_instruction`).
`scan_safety` renders the document, matches every rule (duplicates are not collapsed), and takes the first matching rule that carries an `auto_fail` code as the run's auto-fail.
An auto-fail forces `score_pct = 0.0`, `blocked = true`, `dimensions = None`, and keeps the collected findings — the auto-fail short-circuit reports the finding and stops scoring.
Without an auto-fail the score is `fraction_from_points((100 - 25 · findings).max(0))`, `blocked` when the fraction is below the gate or a `critical` finding exists, and `dimensions` is filled proportionally to the maxima (`round2(score_fraction · max)`) because the regex scan is not dimension-resolved.
`auto_fail` is `None` for an ordinary critical block: the gate is closed, but no AF code applies.
`SkillJudge::safety` always runs `scan_safety` first and short-circuits on an auto-fail without consulting the model.
Otherwise the returned report keeps every scan finding, takes `min(model_fraction, scan_fraction)`, and populates `dimensions` from the verdict's dimension scores when the model verdict is used, else from the deterministic scan.
Hard rule: a safety rule hit can never pass the gate, even when the model returns a perfect verdict; a `critical` finding blocks unconditionally.
`used_fallback` mirrors `verdict.source == VerdictSource::Deterministic`; `evaluate_with_judge*` copies both sources into `EvaluationReport::judge`.

## Artifacts

`EvaluationReport::write_artifacts(dir)` writes the four Python pipeline files and returns their `ArtifactPaths`.
Each writer pretty-prints one report member with the existing serializer, so the files always agree with `EvaluationReport::to_json` and no key list is duplicated in code.
`llm_judge_score.json` = `design` (`skill_name`, `design_score`, `design_score_max`, `design_score_pct`, `grade`, `knowledge_delta`, `dimensions`, `ready_for_functional_testing`).
`skill_safety_score.json` = `safety` (`skill_name`, `score_pct`, `gate_threshold`, `blocked`, `auto_fail`, `dimensions`, `findings`).
`skill_tester_report.json` = `functional` (`skill_name`, `skill_path`, `timestamp`, `summary`, `results`, `recommendations`).
`final_score.json` = `final_score` (`skill_name`, `pipeline_status`, `final_score_pct`, `grade`, `evaluators`, `gates_passed`, `recommendation`), written only for a completed pipeline because a blocked pipeline has no valid final score.
A writer creates missing directories, reports a non-directory target as `Error::validation`, and reports I/O or serialization failures as `Error::external` naming the path.
`EvalRun::report(skill_name, skill_path, timestamp)` produces the same `skill_tester_report.json` key set for a harness-only run; `write_report` persists it.
`EvalRun::to_json`/`to_jsonl` keep the run-artifact wire names (`case_id`, `passed`, `reason`) and are unchanged by the report helper.

## Design Decisions

- The unit convention is the Python one: points only where Python uses points (`design_score`, `design_score_max`, rubric dimensions), fractions everywhere else, with unit-named accessors so a caller cannot mix them.
- Fractions round to four decimals and points to two, so `0.296`/`0.522` reproduce the Python example contributions without float noise.
- Gate order is E1 then E3, matching the pipeline diagram; a run that fails both reports `BLOCKED_AT_E1`.
- `gates_passed` names the gate-role evaluators that passed, i.e. `["e3_safety"]` on completion and `[]` on any block; E1/E2 are weighted evaluators whose scores live in `evaluators`.
- The E1 threshold constant (`DESIGN_GATE_THRESHOLD = 0.60`) is shared by `aggregate`, the readiness flag, and the judge's `design_report_from_verdict`, so the three cannot drift.
- Partial functional cases are reported as `partial` but do not earn fractional pass credit: Python's artifact defines `pass_rate = passed / total`.
- Auto-fail is typed (`AutoFailCode`) rather than a boolean mirror of `blocked`, so "which hard rule fired" survives serialization; a rule-driven AF code is the only deterministic source because the regex scan has no AF detectors of its own.
- Deterministic safety dimensions are proportional to the maxima: honest about the scan's lack of dimension resolution while still emitting the Python shape.
- Safety findings stay typed objects (`dimension`, `severity`, `message`) where Python's artifact shows strings; the richer object is the documented deviation.
- `timestamp` stays epoch milliseconds (`i64`) instead of Python's `"YYYY-MM-DDTHH:MM:SSZ"` string, so the library never formats dates or reads a clock; callers supply it.
- `skill_path` is `skills/<name>` and `CaseResult::id` is a `String` (Python: a filesystem path and an integer).
- `EvaluationReport::to_json`/`to_json_string` remain the in-memory serialization entry points (top-level `design`/`functional`/`safety`/`final_score`, plus `judge` on the judged path); the per-file writers are the artifact layer on top.
- The markdown reports (`llm_judge_report.md`, `pipeline_report.md`, `final_report.md`, `skill_safety_report.md`) are not generated; only the four machine-readable JSON artifacts are.
- An auto-fail safety scan returns a deterministic verdict without a model call, keeping scripted-model call accounting predictable while the hard gate stays closed.
- `ScriptedCaseRunner` is the strict offline default: an unregistered case id is a validation error, never an empty answer.
- The judge rubrics mirror `score_design`/`scan_safety` exactly so a model verdict and the deterministic fallback are interchangeable on the same 0–100 rubric scale.
- `eval.rs` keeps the harness code and `eval/tests.rs` its tests; `scoring.rs` keeps the scoring code and `scoring/tests.rs` its tests.

## Constraints

- Only files under `skill_evaluator/` are edited here; `src/lib.rs`, `Cargo.toml`, `src/util.rs`, `src/judge/`, `tests/`, `docs/`, and sibling modules belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for blank/invalid input, malformed assertions, malformed case documents, missing scripted responses, missing judge configuration, invalid rule patterns, and non-directory artifact paths; `Error::llm` for missing/blank model answers; `Error::external` for serialization and file I/O failures; `skill.to_markdown()`/`validate()` errors propagate unchanged.
- Public data types derive `Debug, Clone, Default, PartialEq, Serialize, Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields; `Assertion`, `CaseStatus`, and `AutoFailCode` are enums and omit the serde `default` attribute, which serde rejects on enums.
- `ScriptedCaseRunner` and `EvalRun` derive `Debug, Clone, Default, PartialEq`; `ArtifactPaths` also derives `Eq`; `ModelCaseRunner`, `ModelAgent`, `CaseRunnerAgent`, and `SkillEvaluator` hold trait objects and derive nothing; `SkillJudge` has a manual `Debug` naming the judge model.
- Async-first: judge calls, case runs, file loading, and every artifact write are `async`; no `block_on` bridging and no blocking calls on async tasks.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing; no network, environment reads, or wall-clock dependence except the documented `evaluate`/`evaluate_with_judge` timestamps.
- Files ≤ 1000 lines with one responsibility each.
- Rustdoc on every public item (workspace `missing_docs`, clippy pedantic + `-D warnings`).

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`); `scoring/tests.rs` holds the scoring matrix and `eval/tests.rs` the harness tests, and `artifacts.rs` carries its own writer tests.
- Unit-conversion tests pin the fraction/point boundary conversions and the grade bands at exactly 0.90/0.80/0.70/0.60 and just below, with NaN and negative scores mapping to `F`.
- Gate tests pin E1 at 0.60 and E3 at 0.80 on both sides of each boundary, the `BLOCKED_AT_E1`/`BLOCKED_AT_E3`/`COMPLETED` tokens, the gate order, and `gates_passed`.
- Pass-rate tests pin `passed/total`, including the 3-passed/1-partial/0-failed → 0.75 example and the empty run.
- Auto-fail tests cover all five codes' Display/parse/serde round-trip, a coded rule forcing score 0.0 / blocked / `dimensions: null`, and a passing model verdict failing to reopen the gate.
- Safety-dimension tests assert the five Python ids and maxima on the deterministic path and their override from a judged verdict.
- Artifact tests assert each file's exact key set, re-read the written files as JSON, skip `final_score.json` when blocked, reject a non-directory target, and create nested directories — always through `openjiuwen_test_support::TempWorkspace`.
- `EvalRun::report` tests pin the `skill_tester_report.json` key set and the failed/partial recommendation list.
- Model-driven paths use `ScriptedModel` with queued responses and assert both the result and the recorded request; no network, environment reads, or wall-clock dependence; float comparisons use an absolute tolerance (`< 1e-9`).

## Deferred

- The Python `SkillEvaluator`/`create_subagent` LLM orchestration (a `ReActAgent` with sysop/code/shell tools and a subagent tool that shells out to skill directories) — not ported; the Rust harness replaces it with the `EvalAgent` seam and caller-supplied agents/tools.
- Tool-using or shell-executing agents inside `run_query` — the harness runs one prompt through an `EvalAgent` and writes the answer; sandboxed tool execution belongs to the sys-op/harness crates.
- LLM-as-judge scoring of functional case answers — functional scoring stays assertion-based; the judge path covers design and safety only.
- Python's deterministic AF1–AF5 detectors — auto-fail codes come from caller-supplied `SafetyRule::with_auto_fail` rules or the judge, because the regex scan has no hardcoded-credential/exfiltration detectors.
- The human-readable reports (`llm_judge_report.md`, `final_report.md`, `pipeline_report.md`, `skill_safety_report.md`) and the `skill_tester_cases.json`/`run-N/` test scaffolding — only the four machine-readable JSON artifacts are written.
- `run_eval_query.py`'s environment-driven model construction (`API_BASE`, `API_KEY`, `MODEL_NAME`, `MODEL_PROVIDER`, `LLM_SSL_VERIFY`, `FILES_BASE_DIR`, `MAX_ITERATIONS`) — configuration stays with the caller, and `files_base_dir`/`max_iterations` are explicit `QueryRequest` fields.
- `QueryRequest.max_iterations` is announced in the query prompt only; no iteration budget is enforced, and the prompt's "You may use tools when necessary" line has no tool channel behind it.
- `EvalCase::files` is carried in the model but never consumed: no evaluator, runner, or harness path materializes or passes case input files, so a case declaring files runs without them and without an error.
- `EvalRun::report` accepts a blank timestamp string (unlike `QueryRequest`, which rejects one), so a harness-only report can carry an empty `timestamp`.
- ISO-8601 timestamp rendering — `FunctionalReport::timestamp` stays epoch milliseconds.

## Routing Table

- `./case.rs` → `EvalCase`, `CaseStatus`, case validation
- `./assertion.rs` → `Assertion` parsing/rendering/checking, `AssertionOutcome`, evidence excerpt
- `./report.rs` → all report/score models, the unit convention and accessors, `AutoFailCode`, pipeline/gate constants, `JudgeRecord`, `EvaluationReport::to_json*`, `grade_for`, shared `status_from`/`summarize`
- `./scoring.rs` → `score_design` rubric, `SafetyRule`/`scan_safety`, `aggregate`, gate order, `round2`/`round4`/`fraction_from_points`/`points_from_fraction`
- `./scoring/tests.rs` → scoring test matrix (design rubric, safety scan, auto-fail, aggregation gates)
- `./runner.rs` → `CaseRunner`, `ScriptedCaseRunner`, `ModelCaseRunner`
- `./judge_scoring.rs` → design/safety rubrics, `SkillJudge`, judgement types, deterministic fallback, safety-dimension population, the hard safety rule
- `./evaluator.rs` → `EvaluatorConfig`, `SkillEvaluator`, deterministic and judge-driven pipelines
- `./eval.rs` → case loading, `EvalAgent`/`ModelAgent`/`CaseRunnerAgent`, `EvalHarness`, `EvalRun` reports, `run_query`
- `./eval/tests.rs` → harness test matrix (loading, scoring, reports, query path resolution)
- `./artifacts.rs` → artifact-name constants, `ArtifactPaths`, the four per-file writers and `write_artifacts`
- `./mod.rs` → module tree and crate-level re-exports
- `../skill_creator/definition.rs` → `SkillDefinition`, `SkillResource`, `NAME_PATTERN` (sibling — read-only, escalate writes to parent)
- `../../judge/` → `ModelJudge`, `JudgeRequest`, `Rubric`, `Verdict`, `VerdictSource` (sibling module — read-only, escalate writes to parent)
- `../../../../tests/skill_lifecycle.rs`, `../../../../tests/judge_and_pipeline.rs` → crate-root integration suites asserting report values (sibling — read-only here, escalate writes to parent)
