# skill_evaluator

## Intent

Evaluation of skills — design review, functional cases (assertion-based, or through a bounded ReAct tool loop), safety gate, weighted final score — plus an optional rubric-driven LLM-judge path, a runnable evaluation harness, and the JSON + Markdown pipeline artifacts.
Ports the artifacts of the Python `openjiuwen.dev_tools.skill_evaluator` (`EvalCase`, `DesignReport`, `FunctionalReport`, `SafetyReport`, `FinalScore`) and the `skills/skill_tester/scripts/run_eval_query.py` single-query runner.
The Python skill prompts (`skill_eval_pipeline`, `skill_judge`, `skill_safety_judge`, `skill_tester`) document the JSON key sets and the prose report headings; no Python code writes those files, so the typed structs here *are* the contract and match those key sets verbatim.
Python graded design/safety by prompting a model with prose rubrics and ran cases through a shell harness; here the default is deterministic structure/assertion scoring, and a model is consulted only when the caller supplies one.
The node owns the data models, the unit convention, the assertion engine, the scoring functions and gates, the judge rubrics/fallbacks with their configurable gate, the scripted/model/ReAct runners, the evaluator, the eval harness, the ISO-8601 timestamp rendering, and the JSON + Markdown artifact layers.

## API Surface

Deterministic data and scoring (re-exported at `openjiuwen_dev_tools::skill_evaluator`):

- `EvalCase` — `id`, `name`, `case_category`, `prompt`, `expected_output`, `files`, `expectations`; builders `new/with_expectation/with_expectations/with_expected_output/with_category/with_files`; `validate()` requires non-blank id/name/prompt and at least one expectation.
- `CaseStatus` — `Passed` (default) / `Partial` / `Failed`, serialized as `"passed"`/`"partial"`/`"failed"`.
- `Assertion` — `Contains`/`NotContains`/`Equals`/`StartsWith`/`Regex`; `parse(&str)` accepts case-insensitive `kind:` prefixes and treats bare text as `Contains`; `text()` renders canonical `kind:argument`; `check(&str) -> Result<AssertionOutcome>`.
- `AssertionOutcome` — `text`, `passed`, `evidence` (prefix excerpt of at most 200 chars, `…` marked when truncated).
- `ScoreDimension` — `score`/`max` in **points** (0–100).
- `DesignReport` — `skill_name`, `design_score` (points 0–100), `design_score_max` (100.0), `design_score_pct` (**fraction**), `grade`, `knowledge_delta` (numeric fraction), `ready_for_functional_testing`, `dimensions` (seven ids, maxima 10/25/15/10/15/15/10).
- `CaseSummary` — `total`, `passed`, `partial`, `failed`, `pass_rate` (**fraction**).
- `CaseResult`, `FunctionalReport` (`skill_name`, `skill_path`, `timestamp` in epoch milliseconds, `summary`, `results`, `recommendations`, plus `timestamp_iso()`), `SafetyFinding` (`dimension`, `severity`, `message`), `SafetyReport`, `EvaluatorScore`, `EvaluatorScores`, `FinalScore`, `EvaluationReport`, `JudgeRecord`.
- `FunctionalReport::timestamp_iso()` renders the caller-supplied epoch milliseconds as Python's `YYYY-MM-DDTHH:MM:SSZ`, so the struct keeps its integer contract while the prose reports show ISO-8601.
- `SafetyReport` — `skill_name`, `score_pct` (**fraction**), `gate_threshold` (fraction, default 0.80), `blocked`, `auto_fail: Option<AutoFailCode>`, `dimensions: Option<BTreeMap<String, ScoreDimension>>` (serialized `null` only on auto-fail), `findings`.
- `AutoFailCode` — `BypassSafety` (AF1), `HardcodedCredentials` (AF2), `UndeclaredExfiltration` (AF3), `TargetingRealIndividuals` (AF4), `InstructingDeception` (AF5); `code()`, `description()`, `parse(&str)` (bare code or `"AF<n>: …"`), `Display` = `"AF<n>: <description>"`; serde is that Display string.
- `FinalScore` — `skill_name`, `pipeline_status`, `final_score_pct` (**fraction**), `grade`, `evaluators`, `gates_passed`, `recommendation`.
- Constants — `PIPELINE_COMPLETED` (`"COMPLETED"`), `PIPELINE_BLOCKED_AT_E1` (`"BLOCKED_AT_E1"`), `PIPELINE_BLOCKED_AT_E3` (`"BLOCKED_AT_E3"`), `GATE_E1` (`"e1_llm_judge"`), `GATE_E2` (`"e2_functional"`), `GATE_E3` (`"e3_safety"`).
- Unit-named accessors — `DesignReport::score_points()`/`score_fraction()`/`knowledge_delta_label()`, `FinalScore::score_fraction()`/`is_completed()`, `SafetyReport::score_fraction()`/`passes_gate()`, `EvaluatorScore::weighted(score_fraction, weight)`.
- `grade_for(f64)` — A ≥ 0.90, B ≥ 0.80, C ≥ 0.70, D ≥ 0.60, else F; the argument is the **fraction**.
- `score_design(&SkillDefinition) -> DesignReport`, `scan_safety(&SkillDefinition, &[SafetyRule], f64) -> Result<SafetyReport>`, `aggregate(&DesignReport, &FunctionalReport, &SafetyReport, f64, f64) -> FinalScore`.
- `SafetyRule` — `pattern`, `dimension`, `severity`, `message`, `auto_fail: Option<AutoFailCode>`; `SafetyRule::new(pattern, dimension, severity, message)` and `with_auto_fail(code)`.
- `CaseRunner` (`async fn run(&self, &EvalCase, &SkillDefinition) -> Result<String>`), `ScriptedCaseRunner`, `ModelCaseRunner`, and the ReAct runner below.
- `EvaluatorConfig` — `design_weight` 0.40, `functional_weight` 0.60, `safety_gate_threshold` 0.80, `safety_rules`, `partial_threshold` 0.50 (parity only; status derivation does not consult it).
- `safety_gate_threshold` is consumed by both paths: `evaluate_at` hands it to `scan_safety` and `evaluate_with_judge_at` hands it to `SkillJudge::safety_with_threshold`, so a configured gate always gates and the value applied is the one `SafetyReport::gate_threshold` reports.
- `SkillEvaluator` — `new(runner)`, `with_config`, `scripted`, `runner`, `config`, `evaluate`, `evaluate_at`, plus the judged path below.

ReAct tool loop (`skill_evaluator::react`):

- `ReactCaseRunner` — a model-backed `CaseRunner` that runs one case through a bounded tool loop: `new(Arc<dyn Model>)`, `with_stubs(ToolStubs)`, `with_tool_declarations(Vec<serde_json::Value>)`, `with_max_rounds(usize)`, `with_system_prompt(impl Into<String>)`, accessors `model()`/`stubs()`/`max_rounds()`, `async run_traced(&EvalCase, &SkillDefinition) -> Result<ReactOutcome>`, and `CaseRunner::run` returning the answer text alone.
- Conversation shape: system message = the configured system prompt or the skill document (`SkillDefinition::to_markdown`), user message = the case prompt plus an `Expected output: …` hint when the case declares one.
- Tool declarations default to one `{"type":"function","function":{…}}` descriptor per registered `ToolStubs` entry (`ToolInfo` serialization, empty parameter object); `with_tool_declarations` overrides them.
- `ReactOutcome { answer, trace, rounds, truncated }` with `used_tools()`/`used(name)`; `truncated` records a budget stop so a capped run is distinguishable from a model stop.
- `DEFAULT_MAX_REACT_ROUNDS` = 8.
- The loop reuses `crate::tune::tool::{ToolStub, ToolStubs, ToolCallRecord, ToolTrace}` for fixtures and tracing; it does not define a second registry, trace, or stub type.

Timestamps (`skill_evaluator::timestamp`):

- `iso8601_utc(epoch_ms: i64) -> String` — UTC `YYYY-MM-DDTHH:MM:SSZ` at second precision, sub-second milliseconds truncated toward negative infinity (`-1` reads `1969-12-31T23:59:59Z`), in-crate Howard-Hinnant civil-date arithmetic with floor-division arithmetic only (no `as` casts, no `chrono`/`time`), exact for every Gregorian leap year and for pre-1970 epochs.

LLM judge (`skill_evaluator::judge_scoring`):

- `design_rubric()` — the seven `score_design` dimensions weighted by their point maxima (10/25/15/10/15/15/10), `scale = 100.0`.
- `safety_rubric()` — the five canonical ids/maxima: `d1_harmful_instruction` 30, `d2_scope_containment` 25, `d3_data_handling` 20, `d4_injection_resilience` 15, `d5_guardrails` 10; `scale = 100.0`.
- `DesignJudgement { verdict, report, used_fallback }`, `SafetyJudgement { verdict, report, used_fallback }`.
- `SkillJudge` — `new(Arc<dyn Model>)`, `with_system_prompt`, `with_gate_threshold(f64)`, `gate_threshold()`, `judge() -> &ModelJudge`, `async design(&SkillDefinition)`, `async safety(&SkillDefinition, &[SafetyRule])` (uses the judge's own threshold, default 0.80), `async safety_with_threshold(&SkillDefinition, &[SafetyRule], f64)` (applies the supplied one).
- A gate threshold that is non-finite or outside `0.0..=1.0` is `Error::validation` before any model call.
- `design_report_from_verdict(&str, &Verdict) -> DesignReport`.
- `SkillEvaluator` additions: `with_judge`, `with_skill_judge`, `judge()`, `evaluate_with_judge`, `evaluate_with_judge_at`.

Evaluation harness (`skill_evaluator::eval`):

- `load_cases_json`, `async load_cases_file`, `EvalAgent`, `ModelAgent`, `CaseRunnerAgent`, `EvalHarness::{new, with_skill, agent, skill, run, run_query}`.
- `EvalRun { cases, summary }` — `to_json`, `to_json_string`, `to_jsonl`, `async write_json`, `async write_jsonl` (run-artifact wire keys `case_id`/`name`/`status`/`passed`/`reason`/`assertions`), plus `report(skill_name, skill_path, timestamp) -> Result<serde_json::Value>` and `async write_report(...)` for the Python `skill_tester_report.json` key set.
- `QueryRequest { prompt, output_path, files_base_dir, timestamp, max_iterations }`, `QueryOutcome { output_file, report_file, answer }`, `resolve_output_file(&Path, timestamp) -> PathBuf`.

Artifacts (`skill_evaluator::artifacts`):

- JSON constants `FINAL_SCORE_ARTIFACT` (`final_score.json`), `DESIGN_ARTIFACT` (`llm_judge_score.json`), `SAFETY_ARTIFACT` (`skill_safety_score.json`), `FUNCTIONAL_ARTIFACT` (`skill_tester_report.json`); `ArtifactPaths { dir, design, safety, functional, final_score: Option<PathBuf> }`.
- Markdown constants `LLM_JUDGE_REPORT_ARTIFACT` (`llm_judge_report.md`), `SAFETY_REPORT_ARTIFACT` (`skill_safety_report.md`), `FINAL_REPORT_ARTIFACT` (`final_report.md`), `PIPELINE_REPORT_ARTIFACT` (`pipeline_report.md`); `MarkdownArtifactPaths { dir, llm_judge, safety, final_report: Option<PathBuf>, pipeline_report: Option<PathBuf> }`.
- `EvaluationReport::{write_design_report, write_safety_report, write_functional_report, write_final_score, write_artifacts, write_llm_judge_report, write_safety_report_md, write_final_report, write_pipeline_report, write_markdown_artifacts}`; every writer is async, creates the directory, rejects a path that exists but is not a directory with `Error::validation`, and reports I/O or serialization failures as `Error::external` naming the path.
- `write_final_score`/`write_final_report` return `None` and write nothing unless the pipeline completed; `write_pipeline_report` returns `None` and writes nothing unless the pipeline is blocked (E1 or E3).
- Markdown renderers (`artifacts::markdown`, re-exported as `artifacts::{llm_judge_report, skill_safety_report, final_report, pipeline_report}`) — pure `String` renderers over `DesignReport`, `SafetyReport`, and `EvaluationReport`.

## Units

Points (0–100) live only in `design_score`, `design_score_max`, `ScoreDimension::score`, and `ScoreDimension::max`.
Every `*_pct` field (`design_score_pct`, `final_score_pct`, `safety.score_pct`, `EvaluatorScore::score_pct`), every `contribution`, every `gate_threshold`, `safety_gate_threshold`, `pass_rate`, and every judge `Verdict::score` is a fraction of `1.0`.
`grade_for` takes the fraction and applies the Python bands (0.90/0.80/0.70/0.60).
`design_score_pct = round4(design_score / 100.0)` and contributions are `round4(weight · score_fraction)`, so `0.4 · 0.74` reads exactly `0.296`.
The Markdown layer renders a fraction as a trimmed percentage (`0.875` → `87.5%`) and a weighted pipeline score/contribution as a decimal (`0.818`, `0.296`); points stay points.
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
The judged path applies `EvaluatorConfig::safety_gate_threshold` through `SkillJudge::safety_with_threshold`, so the deterministic and judged paths cannot disagree on the gate.

## Functional ReAct Tool Loop

`ReactCaseRunner` is the Python `create_subagent` ReAct orchestration reduced to its seam: a bounded model-turn → tool-call → tool-result loop over caller-supplied tool fixtures.
Each turn is one non-streaming `Model::invoke`; a completion with no tool calls ends the run with its text as the answer.
Otherwise every call runs in model order against `ToolStubs`, is recorded in a `ToolTrace` (success or failure), and its rendered result is appended as a `Role::Tool` message carrying the call id (`call_<trace len>` when the model omitted one).
An unknown tool, a failing stub, and unparseable call arguments are data, not control flow: each is recorded as a failed `ToolCallRecord` and its error text is fed back so the conversation continues.
The loop stops after `max_rounds` tool rounds and accepts the model's current text with `ReactOutcome::truncated = true`; `max_rounds == 0` is `Error::validation` before any model call.
Model and transport failures propagate as `Error::llm`; a completion without choices is `Error::llm`.
A `ReactCaseRunner` plugs straight into `SkillEvaluator::new(Arc::new(runner))`, so tool-using cases score through the unchanged assertion path with a `ScriptedModel` and `ToolStubs`, fully offline.

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
`SkillJudge::safety`/`safety_with_threshold` always run `scan_safety` first with the applied threshold and short-circuit on an auto-fail without consulting the model.
Otherwise the returned report keeps every scan finding, takes `min(model_fraction, scan_fraction)`, applies the same threshold to the effective score, and populates `dimensions` from the verdict's dimension scores when the model verdict is used, else from the deterministic scan.
Hard rule: a safety rule hit can never pass the gate, even when the model returns a perfect verdict; a `critical` finding blocks unconditionally.
`used_fallback` mirrors `verdict.source == VerdictSource::Deterministic`; `evaluate_with_judge*` copies both sources into `EvaluationReport::judge`.

## Artifacts

`EvaluationReport::write_artifacts(dir)` writes the four Python pipeline JSON files and returns their `ArtifactPaths`.
Each JSON writer pretty-prints one report member with the existing serializer, so the files always agree with `EvaluationReport::to_json` and no key list is duplicated in code.
`llm_judge_score.json` = `design` (`skill_name`, `design_score`, `design_score_max`, `design_score_pct`, `grade`, `knowledge_delta`, `dimensions`, `ready_for_functional_testing`).
`skill_safety_score.json` = `safety` (`skill_name`, `score_pct`, `gate_threshold`, `blocked`, `auto_fail`, `dimensions`, `findings`).
`skill_tester_report.json` = `functional` (`skill_name`, `skill_path`, `timestamp`, `summary`, `results`, `recommendations`).
`final_score.json` = `final_score` (`skill_name`, `pipeline_status`, `final_score_pct`, `grade`, `evaluators`, `gates_passed`, `recommendation`), written only for a completed pipeline because a blocked pipeline has no valid final score.
A writer creates missing directories, reports a non-directory target as `Error::validation`, and reports I/O or serialization failures as `Error::external` naming the path.
`EvalRun::report(skill_name, skill_path, timestamp)` produces the same `skill_tester_report.json` key set for a harness-only run; `write_report` persists it.
`EvalRun::to_json`/`to_jsonl` keep the run-artifact wire names (`case_id`, `passed`, `reason`) and are unchanged by the report helper.

## Markdown Reports

`write_markdown_artifacts(dir)` writes `llm_judge_report.md` and `skill_safety_report.md` for every run, plus exactly one of `final_report.md` (completed) or `pipeline_report.md` (blocked at E1 or E3) — the mirror of the `final_score.json` skip rule.
Each document is a pure rendering of the report models: no I/O in the renderers, no clock, no environment reads, and the only date is `FunctionalReport::timestamp_iso()` under a `**Run timestamp**` line.
`llm_judge_report.md` — `# Skill Evaluation Report: <skill>`; `## Summary` (`**Total Score**`, `**Grade**`, `**Knowledge Delta**`, `**Ready for functional testing**`); `## Dimension Scores` table over the seven humanised design dimensions with the canonical maxima; `## Critical Issues` (a closed design gate plus every zero-credit dimension); `## Detailed Analysis` (every dimension below 80% of its maximum).
`skill_safety_report.md` — `# Safety Evaluation Report: <skill>`; `## Verdict` (`**Score**` as points plus percentage, `**Gate**: PASS|FAIL`, `**Blocked**`, `**Auto-fail triggered**`: `None` or the `AF<n>: <description>` Display form); `## Dimension Scores` table over the five canonical safety dimensions; `## Findings` (dimension, severity, message); `## Cleared Dimensions` (the dimensions that kept their full maximum).
`final_report.md` — `# Eval Pipeline Report: <skill>`; `## Final Verdict` (`**Score**` as fraction plus grade, `**Status**`, `**Ship-ready**` at the 0.80 fraction); `## Evaluator Breakdown` table (`E1 Design (skill_llm_judge)`, `E2 Functional (skill_tester)`, and the `E3 Safety (skill_safety_judge)` gate row); `## Key Issues to Fix` (functional recommendations, sub-80% design dimensions, safety findings); `## Full Reports` naming the four JSON artifacts.
`pipeline_report.md` — the same four pipeline sections for a blocked run: `**Status**` is the blocking token, `**Blocked by**` names the gate, and the evidence in `## Key Issues to Fix` is what closed the gate.
Python's `**Pattern**` summary line is omitted because `DesignReport` has no pattern field, and a dimension absent from a report renders `-` in its `Score` cell with the canonical maximum rather than an invented score.
Headings, bullet labels, table headers, and dimension labels are the Python contract and are reproduced verbatim; each section is separated by exactly one blank line, no line carries trailing whitespace, and every document ends with a single newline.

## Design Decisions

- The unit convention is the Python one: points only where Python uses points (`design_score`, `design_score_max`, rubric dimensions), fractions everywhere else, with unit-named accessors so a caller cannot mix them.
- Fractions round to four decimals and points to two, so `0.296`/`0.522` reproduce the Python example contributions without float noise.
- Gate order is E1 then E3, matching the pipeline diagram; a run that fails both reports `BLOCKED_AT_E1`.
- `gates_passed` names the gate-role evaluators that passed, i.e. `["e3_safety"]` on completion and `[]` on any block; E1/E2 are weighted evaluators whose scores live in `evaluators`.
- The E1 threshold constant (`DESIGN_GATE_THRESHOLD = 0.60`) is shared by `aggregate`, the readiness flag, and the judge's `design_report_from_verdict`, so the three cannot drift.
- The safety gate threshold is data, not a constant: `SkillJudge` carries a configurable threshold (0.80 default) that `evaluate_with_judge_at` seeds from `EvaluatorConfig`, the applied value is the one stored in `SafetyReport::gate_threshold`, and an out-of-range threshold is a validation error rather than a silent fallback.
- Partial functional cases are reported as `partial` but do not earn fractional pass credit: Python's artifact defines `pass_rate = passed / total`.
- Auto-fail is typed (`AutoFailCode`) rather than a boolean mirror of `blocked`, so "which hard rule fired" survives serialization; a rule-driven AF code is the only deterministic source because the regex scan has no AF detectors of its own.
- Deterministic safety dimensions are proportional to the maxima: honest about the scan's lack of dimension resolution while still emitting the Python shape.
- Safety findings stay typed objects (`dimension`, `severity`, `message`) where Python's artifact shows strings; the richer object is the documented deviation.
- `FunctionalReport::timestamp` keeps the caller-supplied epoch milliseconds as its input contract and serde shape; `timestamp_iso()` renders Python's ISO-8601 UTC form through in-crate civil-date arithmetic, so the library still never reads a clock and needs no date crate.
- ISO rendering truncates sub-second milliseconds toward negative infinity, so a pre-1970 epoch never renders as the following second.
- `skill_path` is `skills/<name>` and `CaseResult::id` is a `String` (Python: a filesystem path and an integer).
- `EvaluationReport::to_json`/`to_json_string` remain the in-memory serialization entry points (top-level `design`/`functional`/`safety`/`final_score`, plus `judge` on the judged path); the per-file writers are the artifact layer on top.
- The Markdown renderers are pure functions over the report models and the Markdown writers only resolve paths and write text, so no report key list or heading set is duplicated between layers.
- `final_report.md` and `pipeline_report.md` are mutually exclusive, mirroring the `final_score.json` skip rule: a blocked run has no valid final score and a completed run has no blocking evidence.
- The Markdown tables humanise the canonical dimension ids in a fixed order (the Python labels), so a dimension absent from a report still occupies its row.
- The ReAct runner reuses `crate::tune::tool`'s `ToolStubs`/`ToolTrace`/`ToolCallRecord` instead of a second tool registry; only the tool-result renderer is restated (tune's is `pub(super)`), with a pointer comment back to it.
- `ReactOutcome::truncated` is an explicit budget flag, unlike the tuning runner that cannot distinguish a capped run from a model stop.
- `ScriptedCaseRunner` is the strict offline default: an unregistered case id is a validation error, never an empty answer.
- The judge rubrics mirror `score_design`/`scan_safety` exactly so a model verdict and the deterministic fallback are interchangeable on the same 0–100 rubric scale.
- `judge_scoring/`, `artifacts/`, `artifacts/markdown/`, `scoring/`, and `eval/` keep their larger suites in sibling `tests.rs` files so each document stays under the size baseline.

## Constraints

- Only files under `skill_evaluator/` and the crate's `tests/skill_lifecycle.rs` are edited here; `src/lib.rs`, `Cargo.toml`, `src/util.rs`, `src/judge/`, other `tests/*.rs` files, `docs/`, and sibling modules belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for blank/invalid input, malformed assertions, malformed case documents, missing scripted responses, missing judge configuration, invalid rule patterns, a non-finite or out-of-range safety gate threshold, a zero ReAct round budget, and non-directory artifact paths; `Error::llm` for missing/blank model answers and ReAct completions without choices; `Error::external` for serialization and file I/O failures; `skill.to_markdown()`/`validate()` errors propagate unchanged.
- Public data types derive `Debug, Clone, Default, PartialEq, Serialize, Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields; `Assertion`, `CaseStatus`, and `AutoFailCode` are enums and omit the serde `default` attribute, which serde rejects on enums.
- `ScriptedCaseRunner`, `EvalRun`, and `ReactOutcome` derive `Debug, Clone, Default, PartialEq`; `ArtifactPaths` and `MarkdownArtifactPaths` also derive `Eq`; `ModelCaseRunner`, `ModelAgent`, `CaseRunnerAgent`, `SkillEvaluator`, and `ReactCaseRunner` hold trait objects and derive nothing (`ReactCaseRunner` and `SkillJudge` have manual `Debug` impls naming the model and the configured gate).
- Async-first: judge calls, case runs, ReAct turns, file loading, and every artifact write are `async`; the Markdown renderers stay synchronous pure functions; no `block_on` bridging and no blocking calls on async tasks.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing; no network, environment reads, or wall-clock dependence except the documented `evaluate`/`evaluate_with_judge` timestamps.
- No new dependencies: the timestamp conversion is in-crate arithmetic, and the tool loop runs on `openjiuwen_foundation`'s `Model`/message types plus `crate::tune::tool`.
- Files ≤ 1000 lines with one responsibility each.

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`); `scoring/tests.rs`, `eval/tests.rs`, `judge_scoring/tests.rs`, and `artifacts/tests.rs` hold the larger matrices, `artifacts/markdown/tests.rs` the report-format matrix, and `timestamp.rs`/`react.rs`/`report.rs` carry inline suites.
- Unit-conversion tests pin the fraction/point boundary conversions and the grade bands at exactly 0.90/0.80/0.70/0.60 and just below, with NaN and negative scores mapping to `F`.
- Gate tests pin E1 at 0.60 and E3 at 0.80 on both sides of each boundary, the `BLOCKED_AT_E1`/`BLOCKED_AT_E3`/`COMPLETED` tokens, the gate order, and `gates_passed`.
- Gate-threshold tests pin both directions on the judged path: a 0.875 verdict is blocked at 0.90 and open at 0.70 and at the 0.80 default, with `SafetyReport::gate_threshold` carrying the applied value each time; a single non-critical rule hit (0.75) is blocked at 0.80/0.90 and open at 0.70; `SkillJudge::with_gate_threshold` moves the plain `safety()` call; 1.5/-0.1/NaN are validation errors with no model call; an auto-fail still short-circuits a raised gate; and `evaluate_at` and `evaluate_with_judge_at` agree at 0.70 and 0.90 on the same skill.
- Pass-rate tests pin `passed/total`, including the 3-passed/1-partial/0-failed → 0.75 example and the empty run.
- Auto-fail tests cover all five codes' Display/parse/serde round-trip, a coded rule forcing score 0.0 / blocked / `dimensions: null`, and a passing model verdict failing to reopen the gate.
- Safety-dimension tests assert the five Python ids and maxima on the deterministic path and their override from a judged verdict.
- JSON-artifact tests assert each file's exact key set, re-read the written files as JSON, skip `final_score.json` when blocked, reject a non-directory target, and create nested directories — always through `openjiuwen_test_support::TempWorkspace`.
- Markdown tests pin whole documents and each Python heading/bullet/table row, the seven design and five safety dimension rows, the auto-fail verdict (`AF2: …`, `FAIL`, `Yes`, `-` table), the `None.` empty states, `**Ready for functional testing**: No` below 0.60, the completed/blocked exclusivity of `final_report.md` vs `pipeline_report.md` (including the `None` returns that write nothing), nested-directory creation, the non-directory validation error, and a trailing-whitespace/blank-line hygiene check on every document.
- Timestamp tests pin a known-value table (`0`, `1_700_000_000_000`, `1_700_000_000_999`, `-1`, `-1_000`, 2000/2024 leap days, the 2100 non-leap boundary, a year rollover, year 3000, `i64::MAX`/`i64::MIN`) against an independent day-count verifier written in the test module, plus `FunctionalReport::timestamp_iso()` and the integer serde shape.
- ReAct tests script `ScriptedModel`/`MockCompletion`/`MockToolCall` turns to cover one tool round then an answer (with the `Role::Tool` reply carrying the call id), a tool-free turn, several calls in one turn, an unknown tool, a failing stub, unparseable arguments, the round-budget truncation flag, the zero-budget validation error, and an end-to-end `SkillEvaluator::evaluate_at` over a tool-derived answer that still scores `passed` through the unchanged assertion path.
- `EvalRun::report` tests pin the `skill_tester_report.json` key set and the failed/partial recommendation list.
- Model-driven paths use `ScriptedModel` with queued responses and assert both the result and the recorded request; no network, environment reads, or wall-clock dependence; float comparisons use an absolute tolerance (`< 1e-9`).

## Deferred

- The Python `SkillEvaluator`/`create_subagent` orchestration around a real `ReActAgent` — the bounded tool loop is `ReactCaseRunner`, but it runs against caller-supplied `ToolStubs`; the sys-op/code/shell tool bundle, the `Runner.resource_mgr` sys-op registration, the per-call subagent config copy, and skill-directory loading (`register_skill`) are not wired.
- Tool-using execution inside `EvalHarness::run_query` — the harness still runs one prompt through an `EvalAgent` and writes the answer; a caller wanting tools uses `ReactCaseRunner` as the evaluator's `CaseRunner`.
- LLM-as-judge scoring of functional case answers — functional scoring stays assertion-based; the judge path covers design and safety only.
- Python's deterministic AF1–AF5 detectors — auto-fail codes come from caller-supplied `SafetyRule::with_auto_fail` rules or the judge, because the regex scan has no hardcoded-credential/exfiltration detectors.
- The `skill_tester_cases.json`/`run-N/` test scaffolding and the design `**Pattern**` summary line — the four JSON artifacts and four Markdown reports are written, nothing else.
- `run_eval_query.py`'s environment-driven model construction (`API_BASE`, `API_KEY`, `MODEL_NAME`, `MODEL_PROVIDER`, `LLM_SSL_VERIFY`, `FILES_BASE_DIR`, `MAX_ITERATIONS`) — configuration stays with the caller, and `files_base_dir`/`max_iterations` are explicit `QueryRequest` fields.
- `QueryRequest.max_iterations` is announced in the query prompt only; no iteration budget is enforced, and the prompt's "You may use tools when necessary" line has no tool channel behind it.
- `EvalCase::files` is carried in the model but never consumed: no evaluator, runner, or harness path materializes or passes case input files, so a case declaring files runs without them and without an error.
- `EvalRun::report` accepts a blank timestamp string (unlike `QueryRequest`, which rejects one), so a harness-only report can carry an empty `timestamp`.

## Routing Table

- `./case.rs` → `EvalCase`, `CaseStatus`, case validation
- `./assertion.rs` → `Assertion` parsing/rendering/checking, `AssertionOutcome`, evidence excerpt
- `./report.rs` → all report/score models, the unit convention and accessors, `timestamp_iso`, `AutoFailCode`, pipeline/gate constants, `JudgeRecord`, `EvaluationReport::to_json*`, `grade_for`, shared `status_from`/`summarize`
- `./scoring.rs` → `score_design` rubric, `SafetyRule`/`scan_safety`, `aggregate`, gate order, `round2`/`round4`/`fraction_from_points`/`points_from_fraction`
- `./scoring/tests.rs` → scoring test matrix (design rubric, safety scan, auto-fail, aggregation gates)
- `./runner.rs` → `CaseRunner`, `ScriptedCaseRunner`, `ModelCaseRunner`
- `./react.rs` → the bounded ReAct tool loop (`ReactCaseRunner`, `ReactOutcome`, `DEFAULT_MAX_REACT_ROUNDS`), tool declarations, tool-failure records
- `./timestamp.rs` → `iso8601_utc` and the in-crate civil-date arithmetic
- `./judge_scoring/` → design/safety rubrics, `SkillJudge` with its configurable gate threshold, judgement types, deterministic fallback, safety-dimension population, the hard safety rule (`mod.rs` + `tests.rs`)
- `./evaluator.rs` → `EvaluatorConfig`, `SkillEvaluator`, deterministic and judge-driven pipelines
- `./eval.rs` → case loading, `EvalAgent`/`ModelAgent`/`CaseRunnerAgent`, `EvalHarness`, `EvalRun` reports, `run_query`
- `./eval/tests.rs` → harness test matrix (loading, scoring, reports, query path resolution)
- `./artifacts/` → JSON and Markdown artifact-name constants, `ArtifactPaths`/`MarkdownArtifactPaths`, the per-file writers and `write_artifacts`/`write_markdown_artifacts`, path resolution (`mod.rs` + `tests.rs`)
- `./artifacts/markdown/` → the four deterministic Markdown renderers over the report models (`mod.rs` + `tests.rs`)
- `./mod.rs` → module tree and crate-level re-exports
- `../skill_creator/definition.rs` → `SkillDefinition`, `SkillResource`, `NAME_PATTERN` (sibling — read-only, escalate writes to parent)
- `../tune/tool.rs` → `ToolStub`/`ToolStubs`, `ToolCallRecord`/`ToolTrace` reused by the ReAct runner (sibling module — read-only, escalate writes to parent)
- `../../judge/` → `ModelJudge`, `JudgeRequest`, `Rubric`, `Verdict`, `VerdictSource` (sibling module — read-only, escalate writes to parent)
- `../../../../tests/skill_lifecycle.rs`, `../../../../tests/judge_and_pipeline.rs` → crate-root integration suites asserting report values and the Markdown artifacts (sibling — read-only here, escalate writes to parent)
