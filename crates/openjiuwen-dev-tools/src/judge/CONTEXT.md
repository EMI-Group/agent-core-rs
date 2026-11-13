# The judge module

## Intent

Rubric-driven LLM judging over the `openjiuwen_foundation::llm::Model` seam, with tolerant structured-verdict parsing.
This is the shared scoring foundation for `../skill_evaluator/` and `../tune/`: those modules consume `Judge`, `Verdict`, and `Rubric` instead of calling a model themselves.

## API Surface

- `mod.rs` — module docs and the re-exports: `Judge`, `JudgeRequest`, `ModelJudge`, `Rubric`, `RubricDimension`, `DEFAULT_PASS_THRESHOLD`, `Verdict`, `VerdictSource`.
- `rubric.rs` — `Rubric`/`RubricDimension` data, `validate`, `normalize` (weighted, clamped to `0.0..=1.0`), `is_passing`, `uniform`, and the `0.6` pass threshold.
- `verdict.rs` — `Verdict` (normalized score, pass flag, reasoning, raw dimension scores, source), `VerdictSource`, `deterministic`/`model` constructors, `validate`, and the tolerant `Verdict::parse`.
- `prompt.rs` — the constant `system_prompt`, `render_rubric`, and `rubric_messages` (exactly one system plus one user message).
- `model_judge.rs` — `JudgeRequest`, the `Judge` trait, `ModelJudge` (one `Model::invoke` per judgement), and `judge_with_fallback`.

## Constraints

- Pure prompt rendering and parsing: no environment or clock reads, no panics on model output.
- `Verdict::parse` accepts a fenced JSON object, tag sections, or a bare JSON object; percent strings mean percent of the rubric scale; a raw score of at most `1.0` on a scale above `1.0` is treated as already normalized.
- Errors are `openjiuwen_common::error::Error::Validation` (invalid rubric, no parsable verdict) or `Error::Llm` (no choices); `judge_with_fallback` logs with `tracing::warn!` and marks the fallback `VerdictSource::Deterministic`.
- Tests are offline: `openjiuwen_test_support::ScriptedModel` supplies every model answer.

## Routing Table

- `./rubric.rs` → rubric data, validation, weighted normalization, pass threshold
- `./verdict.rs` → verdict data and the tolerant answer parser
- `./prompt.rs` → system prompt, rubric rendering, judge message assembly
- `./model_judge.rs` → the `Judge` seam, `ModelJudge`, and the fallback path
