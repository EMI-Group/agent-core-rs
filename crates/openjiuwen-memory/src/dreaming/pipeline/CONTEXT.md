# src/dreaming/pipeline

## Intent

The model-backed consolidation pipeline: it turns one claimed `DreamJob` into a bounded candidate list, asks a `Model` which records to merge/keep/summarize/forget, parses the answer, and applies the resulting plan through an injected `ConsolidationSink`.
It is the implementation of the `DreamConsolidator` seam the planner was built around; `NoopConsolidator` remains the unwired application's default.
Everything is offline-testable: a `None` model, a disabled config, a provider error, or an unusable answer degrades to the deterministic rule plan and records why.
The module is split by responsibility: `pipeline.rs` owns I/O, prompt bounding, the model loop, and the sink; `plan.rs` owns the pure decision layer; `tests.rs` is the offline unit suite.

## API Surface

`pipeline.rs`:
- `DreamCandidate` (id, `content` as JSON `"text"`, `kind` as JSON `"type"`, importance, timestamp, access_count) with `new`, `from_record`, `with_*` builders, `validate`, and a `Default` whose timestamp is the Unix epoch.
- `DreamCandidateSource` trait with `StaticCandidateSource` and `PayloadCandidateSource` (reads `PAYLOAD_CANDIDATES_KEY = "candidates"`; an absent key is empty, a malformed value is `Error::Validation` naming the job).
- `ConsolidationSink` trait and `RecordOnlySink` (the default: applies nothing, records the plan metrics).
- `LlmConsolidatorConfig` (enabled, model_retries, max_candidates, max_content_chars, fallback_similarity) with `with_*` builders and `validate`.
- `LlmConsolidator` with `new`, `payload_driven`, `with_sink`, `config`, `plan`, and `impl DreamConsolidator::consolidate`.
- Private: `render_request`, `bounded_text`, `DEFAULT_MAX_CONTENT_CHARS`, `DEFAULT_FALLBACK_SIMILARITY`, `unavailable_reason`, `rules_plan`, `build_messages`.

`plan.rs` (the pure, clock-free, model-free decision layer; re-exported from `pipeline` so the public path is `openjiuwen_memory::dreaming::*`):
- `DreamConsolidationAction` (`Merge`/`Keep`/`Summarize`/`Forget`) with `action_name`, `affected_ids`, `reason`, `note`.
- `DreamConsolidationOrigin` (`Rules`/`Llm`), `DreamConsolidationPlan` (job_id, origin, actions, degraded, metrics) with `new`, `is_degraded`, `is_empty`, `notes`.
- `parse_consolidation_response`, `deterministic_plan` (both public), plus private `plan_metrics`, `chars_of`, `RawPlan`/`RawAction`, `normalize_action`, `longest_content`.

`tests.rs` (`#[cfg(test)] mod tests`): the offline unit suite, see Test Strategy below.

## Behavioral Contract

The model is attempted `model_retries + 1` times; a provider error, an empty answer, unparseable JSON, or an answer whose every action names invented ids counts as a failed attempt, and the first failure becomes the `degraded` reason.
Cancellation is the one non-degrading failure: `plan` checks `ctx` before candidate selection, before the first call, and between attempts, and returns `Error::Cancelled`.
The answer is parsed from the first `{` to the last `}` (prose and code fences are tolerated), action names and field aliases are normalised, and an action naming no known candidate is dropped; an answer whose actions are all dropped is unusable, while an explicit `{"actions":[]}` is a valid empty plan.
A merge names its kept record with `target_id` and the absorbed ones with `source_ids`; both join the affected id set, so the documented shape parses without repeating the target in `source_ids`.
`consolidate` applies the plan through the sink, returns `Skipped` with the note `no consolidation actions` when the plan is empty, and appends the plan notes to the outcome.
A sink error propagates (the planner retries the attempt); a cancelled sink call propagates `Error::Cancelled`.
The deterministic fallback merges exact/near-duplicate texts (token Jaccard `>= fallback_similarity`, clamped, non-finite uses `0.75`) into the first group member with the longest text, and keeps every other candidate.
Metrics are derived from the actions: `records_scanned` from candidates, `records_merged` from merge sources, `records_created` from summaries, `records_deleted` from forgotten ids, `tokens_saved` as `reclaimed_chars / 4`.

## Constraints

- `pipeline.rs` and `plan.rs` stay under ~800 lines each; `tests.rs` is the only large file and holds no production code.
- No `unwrap`/`expect`/`panic` outside `#[cfg(test)]`; `tracing` only, targets `openjiuwen_memory::dreaming::pipeline`.
- Errors are `openjiuwen_common::error::{Error, ErrorKind, Result}`: `Validation` for bad config/candidates/payloads, `Cancelled` propagated, `External` for a request that cannot be encoded.
- Port and text bounds are applied before the model sees anything: `max_candidates` truncates in source order, `max_content_chars` truncates each text with a trailing `…`.
- The module performs no memory writes of its own: every mutation belongs to the injected `ConsolidationSink`, which is also the only owner of real store access.
- No clock of its own: the pipeline reads `job` fields only, which keeps the tests reproducible.

## Test Strategy

`tests.rs` covers, all offline with `openjiuwen_test_support::ScriptedModel` and a fixed timestamp (`DateTime::from_timestamp(1_700_000_000, 0)`):
the LLM happy path (origin `Llm`, two actions, `records_merged == 1`, `records_scanned == 3`, one call whose user text carries the candidate ids and bounded text);
retry-then-fallback for prose answers and provider errors (degraded reason names the failure, exactly `model_retries + 1` calls);
no model, disabled config, and no candidates (each degradation reason, zero calls);
invented ids, empty-action answers (`Skipped` with `no consolidation actions`);
`consolidate` through `RecordOnlySink` and through a failing sink;
pre-cancelled `CallOptions` returning `Error::Cancelled` with zero calls;
config validation branches, `max_candidates` truncation, and `max_content_chars` truncation (`…`);
`PayloadCandidateSource` (absent, valid, malformed), `StaticCandidateSource` order;
`DreamCandidate` serde aliases/defaults, validation, and `from_record` projection;
and the pure layer: action helpers, plan metrics/notes, tolerant parsing with aliases and fences, unusable answers, and `deterministic_plan` threshold handling.
Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory --lib dreaming::pipeline` with and without `--all-features`, `cargo clippy -p openjiuwen-memory --all-targets --all-features -- -D warnings`, and `cargo doc -p openjiuwen-memory --no-deps` warning-free.
