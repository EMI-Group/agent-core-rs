# skill_evaluator

## Intent

Deterministic, offline evaluation of skills: a structural design rubric, expectation-checked functional cases, a regex safety gate, and the weighted final score.
Ports the report artifacts of the Python `openjiuwen.dev_tools.skill_evaluator` (`EvalCase`, `DesignReport`, `FunctionalReport`, `SafetyReport`, `FinalScore`) onto typed Rust models.
Python graded design/safety by prompting a model with prose rubrics and ran cases through a shell harness; here scoring is pure structure/assertion logic, so results are reproducible without a model.
Case execution (the runner) and the end-to-end evaluator pipeline are added by a later executor; this node owns the data models, the assertion engine, and the scoring functions they call.

## API Surface

- `EvalCase` — `id`, `name`, `case_category`, `prompt`, `expected_output`, `files`, `expectations`; builders `new/with_expectation/with_expectations/with_expected_output/with_category/with_files`; `validate()` requires non-blank id/name/prompt and at least one expectation.
- `CaseStatus` — `Passed` (default) / `Partial` / `Failed`, serialized as `"passed"`/`"partial"`/`"failed"`.
- `Assertion` — `Contains`/`NotContains`/`Equals`/`StartsWith`/`Regex`; `parse(&str)` accepts case-insensitive `kind:` prefixes and treats bare text as `Contains`; `text()` renders canonical `kind:argument`; `check(&str) -> Result<AssertionOutcome>`.
- `AssertionOutcome` — `text`, `passed`, `evidence` (prefix excerpt of at most 200 chars, `…` marked when truncated).
- `ScoreDimension` — `score`, `max`.
- `DesignReport` — `skill_name`, `design_score`, `design_score_max` (100.0), `design_score_pct`, `grade`, `knowledge_delta`, `ready_for_functional_testing`, `dimensions: BTreeMap<String, ScoreDimension>`.
- `CaseSummary`, `CaseResult`, `FunctionalReport` — counts/pass rate, per-case assertion outcomes, and the run artifact (`skill_name`, `skill_path`, `timestamp`, `summary`, `results`, `recommendations`).
- `SafetyFinding` and `SafetyReport` — `score_pct`, `gate_threshold`, `blocked`, `auto_fail`, `dimensions: Option<…>`, `findings`.
- `EvaluatorScore`, `EvaluatorScores`, `FinalScore` — weighted contributions (`e1_llm_judge`, `e2_functional`), `pipeline_status`, `final_score_pct`, `grade`, `gates_passed`, `recommendation`.
- `EvaluationReport` — the four reports plus `to_json()`/`to_json_string()`.
- `grade_for(f64) -> String` — A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else `F` (NaN and negatives are `F`).
- `score_design(&SkillDefinition) -> DesignReport`, `scan_safety(&SkillDefinition, &[SafetyRule], f64) -> Result<SafetyReport>`, `aggregate(&DesignReport, &FunctionalReport, &SafetyReport, f64, f64) -> FinalScore`.
- `SafetyRule` — `pattern`, `dimension`, `severity`, `message`; `new(...)`.
- Everything is re-exported at `openjiuwen_dev_tools::skill_evaluator`.

## Design Rubric

Seven dimensions, each awarded `0`, half the maximum, or the full maximum: `d1_knowledge_ratio` (10), `d2_expert_knowledge_craft` (25), `d3_spec_compliance` (15), `d4_progressive_disclosure` (10), `d5_pattern_freedom_fit` (10), `d6_practical_usability` (15), `d7_output_specification` (10).
The maxima sum to 95; `design_score` normalizes the awarded points onto the reported 100-point scale, so an all-full rubric reads 100.0 and a half-full one 50.0.
`design_score_pct` equals `design_score` (the same 0–100 scale); `knowledge_delta = design_score / 100`; `ready_for_functional_testing` is `design_score_pct >= 50.0`.
Safety starts at 100 and loses 25 points per matching rule, clamped at zero; the gate is closed when `score_pct / 100 < gate_threshold` or any finding's severity is exactly `critical`.
`aggregate` weights design at `design_weight` (0.40 in the Python formula) and the functional pass rate at `functional_weight` (0.60); gate order is safety, then design readiness, then completion, and `gates_passed` is `["e1_llm_judge", "e2_functional", "e3_safety"]` on completion.

## Constraints

- Only files under `skill_evaluator/` are edited here; `src/lib.rs`, `Cargo.toml`, `src/util.rs`, `docs/`, and sibling modules belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for blank/invalid input, malformed assertions, and invalid rule patterns; `Error::external` for serialization failures; `skill.to_markdown()` errors propagate unchanged.
- Public data types derive `Debug, Clone, Default, PartialEq, Serialize, Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields; `Assertion` and `CaseStatus` are enums and omit the serde `default` attribute, which serde rejects on enums.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing; no network, environment reads, or wall-clock dependence.
- Files ≤ 1000 lines with one responsibility each; rustdoc on every public item (workspace `missing_docs`, clippy pedantic + `-D warnings`).

## Design Decisions

- Evidence is a deterministic prefix excerpt of the output rather than a window around the match: identical inputs produce identical reports.
- `d2` counts distinct `##`/`###` heading texts (a `####` line never counts) plus numbered-step lines, so duplicated headings cannot inflate the score.
- `d4` credits a resource reference by path or last path component; with no resources, a resource-like filename in the body (`references/guide.md`, `install.sh`) still earns the half score as a signposted but unshipped disclosure.
- `d6` treats any list item or fenced code block as full, and a backtick pair as the half level; `d7` looks for output/result/format wording (`输出`/`结果` included) before falling back to return/response wording.
- The rubric maxima sum to 95 while the contract reports a 100-point design score; points are normalized instead of rescaling the per-dimension maxima, keeping the documented maxima and making an all-full rubric score exactly 100.0.
- `e1_llm_judge` is always populated by `aggregate` because the design rubric, not a model, plays the judge role; a future model judge can overwrite it without changing the type.
- `SafetyReport::dimensions` stays `None` and serializes as `null`: the safety scan is a penalty tally, not a dimension rubric.

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`): 17 assertion, 9 case, 6 report, and 20 scoring tests.
- Assertion tests cover all five kinds, mixed-case prefixes, canonical round-trip, pass/fail checks, empty-output and 200-char truncation evidence, char-boundary safety, and serde round-trips.
- Case tests cover builder coverage, every validation error, the snake_case status strings, unknown-field tolerance, and JSON round-trip.
- Report tests cover the grade boundary matrix (including negative and NaN), top-level JSON keys, `dimensions: null`, and unknown-field tolerance.
- Scoring tests build a rich skill (all dimensions full → 100.0/A/ready) and a minimal one (< 50, not ready), check an invalid name lowers `d3` without panicking, and cover the safety matrix (no rules, soft match at a 0.5 gate, critical block, two findings, non-match, invalid pattern, unrenderable skill) plus all three aggregation branches and recommendation boundaries.
- Float comparisons use an absolute tolerance (`< 1e-9`), never `==`.

## Routing Table

- `./case.rs` → `EvalCase`, `CaseStatus`, case validation
- `./assertion.rs` → `Assertion` parsing/rendering/checking, `AssertionOutcome`, evidence excerpt
- `./report.rs` → all report/score models, `EvaluationReport::to_json*`, `grade_for`
- `./scoring.rs` → `score_design` rubric, `SafetyRule`/`scan_safety`, `aggregate`, `round2`
- `./mod.rs` → module tree and crate-level re-exports (runner/evaluator are appended by a later executor)
- `../skill_creator/definition.rs` → `SkillDefinition`, `SkillResource`, `NAME_PATTERN` (sibling — read-only, escalate writes to parent)
