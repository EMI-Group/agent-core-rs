# dataset

## Intent

Case models and dataset loading for evaluation and training.
Ports Python `openjiuwen.agent_evolving.dataset` (`case.py`, loader helpers) and adds file loading (the Python package leaves dataset formats to the host).

## API Surface

- `Case` — Python fields `inputs`, `label`, `tools`, `case_id`; validating `new`, `with_tools`, `with_case_id`, `validate`.
- `EvaluatedCase` — Python fields `case`, `answer`, `score`, `reason`, `per_metric`; constructor clamps finite scores to `0.0..=1.0`, rejects non-finite scores; accessors forward to `case`; `From<Case>` yields score `0.0` and no answer.
- `CaseLoader` — `new`, `len`/`is_empty`/`size`, `cases`, `into_cases`, `get`, `iter`, `shuffle(seed)`, `split(ratio)`; serde as `{"cases": [...]}`.
- `shuffle_cases`, `split_cases` — module-level Python helpers (positional split, no shuffle).
- `CaseFormat`, `parse_cases_json`, `parse_cases_jsonl`, `parse_cases_str`, `load_cases_from_path` — JSON accepts an array, `{"cases": [...]}`, or one case object; JSONL skips blank lines and `#` comments; all parsed cases are validated.

## Constraints

- Python field names and serde defaults; unknown fields tolerated; `schemars::JsonSchema` on `Case`/`EvaluatedCase`.
- Validation errors use `Error::validation`; no panics on user input.
- Determinism: seeded xorshift64* shuffle (seed `0` remapped) so the crate needs no `rand` dependency.

## Design Decisions

- Serialization stays permissive (malformed-but-parseable JSON round-trips); `new`/`validate` enforce Python's pydantic constraints.
- `answer` and `per_metric` serialize as `null` when absent, matching Python `model_dump`.
- `EvaluatedCase::new` clamps instead of rejecting out-of-range finite scores because Python's `field_validator` clamps; only non-finite values are rejected.

## Routing Table

- `./mod.rs` → `Case`, `EvaluatedCase`, module re-exports
- `./loader.rs` → `CaseLoader`, shuffle/split helpers, JSON/JSONL parsing and async file loading
