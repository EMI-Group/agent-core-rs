# signal

## Intent
The feedback vocabulary every evolution subsystem shares: typed signal records, the extractors that derive them, and the in-process tracker that dedupes and orders them.
Ports the portable part of Python `openjiuwen.agent_evolving.signal` and the signal records of `optimizer/base.py`.
Signals are pure serde data: no model calls, no clocks, no environment reads.

## API Surface
- `EvolutionCategory` — closed vocabulary `description`, `body`, `script`, `tool`, `memory`, `skill`, `workflow`, `context`, `other`; `as_str` gives the wire spelling.
- `EvolutionTarget` — `instruction` plus the same layers, defaulting to `other`; `EvolutionTarget::category` projects a target onto a category.
- `EvolutionSignal` — Python fields `signal_type`, `category`, `target`, `section`, `excerpt`, `skill_name`, `tool_name`, `source`, `confidence`, `weight`, `context`; `new` trims and validates `signal_type`/`excerpt`, derives the category from the target, and starts confidence/weight at `1.0`.
- `EvolutionSignal::dedupe_key` — the tracker's stable identity, `"{category}:{target}:{normalized excerpt}"`, so the same complaint from two producers collapses into one entry.
- `MAX_EXCERPT_CHARS` (240), `normalize_excerpt`, `truncate_excerpt` — the excerpt bounding the constructors and extractors share.
- Extractors: `signals_from_evaluated_cases`, `signals_from_conversation`, `signals_from_review_feedback`, `signals_from_team_feedback`, `signals_from_failure`, plus `MAX_SIGNALS_PER_INPUT` (64) and `NEGATIVE_MARKERS`.
- `SignalTracker` / `TrackedSignal` — bounded dedupe map with `record`, `record_all`, `decay_all`, `signals`, `top`, `group_by_target`, `drain`, `clear`; `capacity` (`0` stores nothing), `decay`, and a monotonic `seq`.

## Constraints
- Every extractor is a pure function of its inputs: output order follows input order and each call stops at `MAX_SIGNALS_PER_INPUT`.
- Excerpts are cut to `MAX_EXCERPT_CHARS`; evaluation excerpts truncate each rendered map to 80 characters first.
- Ordering and eviction use the monotonic sequence counter, never wall-clock time, so identical inputs give byte-identical state.
- `SignalTracker::with_decay` rejects non-finite or out-of-range factors with `Error::validation`.

## Design Decisions
- Signal identity is a data-derived `dedupe_key` rather than an assigned id, so two producers emitting the same feedback collapse into one tracked entry.
- `TrackedSignal` keeps the first recorded signal verbatim and only accumulates `hits`, `weight` and `last_seq`; the highest confidence wins on merge.
- Eviction is a pure function of `(weight, last_seq, key)` so a bounded tracker replays identically.
- The extractors are deliberately rule-based (markers, score bands, failure keywords) because Python's model-driven variants need a live model and are non-deterministic.

## Routing Table
- `./mod.rs` → module documentation and the re-export surface
- `./types.rs` → `EvolutionSignal`, `EvolutionCategory`, `EvolutionTarget`, excerpt helpers
- `./extract.rs` → evaluation/conversation/review/team/failure extractors
- `./tracker.rs` → `SignalTracker`, `TrackedSignal`, dedupe/decay/eviction
