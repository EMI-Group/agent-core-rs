# experience

## Intent
Reusable lessons derived from signals and trajectories: the record schema, the deterministic archive (dedupe/merge/rank/query), extraction, and the lifecycle maintenance pass.
Ports the offline part of Python `openjiuwen.agent_evolving.experience`: `types.py`, `archive.py`, `query.py`, `scorer.py`, `lifecycle.py` and `rebuild.py`.
The LLM draft pipeline and the online orchestrator are not ported; extraction and ranking here are deterministic.

## API Surface
- `ExperienceRecord` — `record_id`, `skill_name`, `section`, `content`, `keywords`, `source`, `confidence`, `weight`, `hits`, `successes`, `failures`, `shared`, `created_seq`, `updated_seq`, `metadata`; builders `with_keywords`, `record_success`, `record_failure`, `with_confidence`, and `score`.
- `ExperienceSection` — `description`, `body`, `script`, `tool`, `memory`, `other` (Python's lesson-section vocabulary).
- `ExperienceArchive` — `add` (insert or merge into a near-duplicate), `get`/`update`/`remove`, `query`, `top`, `by_skill`, `by_section`, `shared`, `decay`, `prune_below`, `merge_from`, `stats`; `DEFAULT_CAPACITY` 512, `DEFAULT_DEDUPE_THRESHOLD` 0.85; `AddOutcome` (`Inserted`/`Merged`), `MergeSummary`, `ExperienceStats`.
- `ExperienceMatch` / `ExperienceQuery` — filters plus the returning match with its `score` and `matched_keywords`.
- `ExperienceExtractor` — `from_signals`, `from_trajectories`, `from_evaluated_cases`, with `DEFAULT_MIN_CONFIDENCE` 0.5, `DEFAULT_MAX_RECORDS` 64, `DEFAULT_KEYWORD_LIMIT` 8, `DEFAULT_SKILL_NAME`.
- `ExperienceLifecycle` / `LifecycleConfig` / `LifecycleReport` — `apply(archive)` promotes proven records, decays and prunes stale ones; `rebuild(records)` replays a record stream through the archive's dedupe rule; `LifecycleConfig` defaults to `min_hits` 3, `promote_success_rate` 0.7, `stale_seq_gap` 100, `min_weight` 0.1.
- Text primitives shared with `sharing`: `tokens`, `jaccard`, `stopwords`.

## Constraints
- Clock-free: recency is measured in archive sequence numbers (`created_seq`, `updated_seq`) and staleness in sequence steps, never wall-clock time.
- Ranking is a fixed linear blend of keyword overlap, text overlap and observed success rate; `ExperienceRecord::score` is `weight * confidence * (0.5 + success_rate)`.
- Dedupe is content-token Jaccard at or above the archive threshold; a full archive evicts its lowest-scoring record.
- Constructors validate: blank record ids, skill names and content are rejected with `Error::validation`.
- Record ids are content-derived FNV-1a 64-bit hashes (`sig-`/`traj-`/`case-` plus 16 hex digits); `created_seq`/`updated_seq` are stamped only by `ExperienceArchive::add`.
- Serde is asymmetric: `ExperienceRecord`, `ExperienceQuery`, `ExperienceMatch`, `ExperienceSection`, `AddOutcome`, `MergeSummary`, `ExperienceStats` and `ExperienceArchive` (private fields) serialize, but `ExperienceExtractor`, `LifecycleConfig`, `LifecycleReport` and `ExperienceLifecycle` do not derive `Serialize`/`Deserialize`.
- Deserializing an `ExperienceArchive` restores its records verbatim without re-validating them; callers must run `ExperienceRecord::validate` if the payload is untrusted.

## Known Issues
- `ExperienceArchive::prune_below` documents that a non-finite threshold removes nothing, but a `NaN` threshold makes the `weight >= NaN` retain predicate false for every record and prunes the whole archive.

## Design Decisions
- Sequence counters replace Python's wall-clock timestamp and free-form patch payload, so archives replay identically from a fixed input order.
- The archive is in-memory and JSON-file-free, unlike Python's per-skill JSON files; persistence is the caller's concern and the archive itself is `Serialize`.
- A never-used record still scores half of an equally weighted perfect record, so a fresh lesson is neither invisible nor dominant.
- Extraction is rule-based (confident signals, the first error step, successful multi-tool runs, sub-threshold cases) because the Python equivalent asks a model to draft the lesson.
- `LifecycleConfig` is a plain `Copy` threshold record with `Default`, so a lifecycle pass is a pure function of the archive and its config.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./models.rs` → `ExperienceRecord`, `ExperienceSection`, `ExperienceQuery`, `ExperienceMatch`
- `./archive.rs` → `ExperienceArchive`, dedupe/merge/rank/query, text primitives
- `./extract.rs` → `ExperienceExtractor`
- `./lifecycle.rs` → `ExperienceLifecycle`, `LifecycleConfig`, `LifecycleReport`
