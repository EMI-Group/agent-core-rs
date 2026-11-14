# experience

## Intent
Reusable lessons derived from signals and trajectories, plus the agent-facing draft pipeline that turns them into staged or persisted records.
Ports Python `openjiuwen.agent_evolving.experience` (`types.py`, `archive.py`, `query.py`, `scorer.py`, `lifecycle.py`, `rebuild.py`, `draft_schema.py`, `online_orchestrator.py`) and the experience half of `optimizer/skill_call` (`experience_optimizer.py`, `experience_draft_parser.py`).
The archive, extraction, ranking, simplification and orchestration are deterministic; an injected model only enriches a draft, and every model failure falls back to the deterministic path.

## API Surface
- `ExperienceRecord` — `record_id`, `skill_name`, `section`, `content`, `keywords`, `source`, `confidence`, `weight`, `hits`, `successes`, `failures`, `shared`, `created_seq`, `updated_seq`, `metadata`; builders `with_keywords`, `record_success`, `record_failure`, `with_confidence`, `with_source`, `with_skill`, `with_section`, and `score`.
- `ExperienceSection` — `description`, `body`, `script`, `tool`, `memory`, `other` (Python's lesson-section vocabulary).
- `ExperienceArchive` — `add` (insert or merge into a near-duplicate), `get`/`update`/`remove`, `query`, `top`, `by_skill`, `by_section`, `shared`, `decay`, `prune_below`, `merge_from`, `stats`; `DEFAULT_CAPACITY` 512, `DEFAULT_DEDUPE_THRESHOLD` 0.85; `AddOutcome` (`Inserted`/`Merged`), `MergeSummary`, `ExperienceStats`.
- `ExperienceMatch` / `ExperienceQuery` — filters plus the returning match with its `score` and `matched_keywords`.
- `ExperienceExtractor` — `from_signals`, `from_trajectories`, `from_evaluated_cases`, with `DEFAULT_MIN_CONFIDENCE` 0.5, `DEFAULT_MAX_RECORDS` 64, `DEFAULT_KEYWORD_LIMIT` 8, `DEFAULT_SKILL_NAME`.
- `ExperienceLifecycle` / `LifecycleConfig` / `LifecycleReport` — `apply(archive)` promotes proven records, decays and prunes stale ones; `rebuild(records)` replays a record stream through the archive's dedupe rule; `LifecycleConfig` defaults to `min_hits` 3, `promote_success_rate` 0.7, `stale_seq_gap` 100, `min_weight` 0.1.
- `draft` — `EvolutionSubject` / `EvolutionSubjectKind` with `payload()` and `normalize_subject`, `normalize_subject_kind`, `EvolutionTarget` (`description | body | script`), `DraftSection` with `parse` / `as_str`, `DraftExperience`, `EvolveDraft` (`approval_view` keeps `source_refs`, `persistence_view` drops them), `SimplifyActionKind`, `SimplifyAction`, `SimplifyDraft`, `normalize_evolve_draft`, `normalize_simplify_draft`, `validate_simplify_record_refs`, `SUMMARY_LIMIT` 160, `SUPPORTED_SUBJECT_KINDS`.
- `drafter` — `ExperienceDrafter` (`new`, `with_model`, `has_model`, `with_max_items`, `with_max_summary_chars`, `with_max_skill_chars`), `DraftInput` with `with_trajectories` / `with_user_query` / `with_existing`, `draft`, `draft_offline`, `simplify`, `simplify_offline`, `parse_draft_text`, `extract_json_value`, `summarize_skill_content`, `existing_summaries`, `draft_keywords`, `build_draft_prompt`, `build_repair_prompt`, `build_simplify_prompt`, `DraftParse`, `DRAFT_PROMPT`, `FORMAT_REPAIR_PROMPT`, `SIMPLIFY_PROMPT`, `DEFAULT_MAX_ITEMS` 3, `DEFAULT_MAX_SUMMARY_CHARS` 160, `DEFAULT_MAX_SKILL_CHARS` 6000, `SIMPLIFY_KEEP_SCORE` 0.4.
- `orchestrator` — `OnlineEvolutionOrchestrator` (`new`, `with_drafter`, `with_archive`, `drafter`, `archive`, `archive_mut`, `evolve`, `complete`), `EvolutionRequest` with lookup flags and approval switch, `EvolutionStatus`, `OnlineEvolutionResult` (`status`, `subject`, `staged`, `committed`, `reason`), `initial_confidence`, `INITIAL_SCORE_BY_SIGNAL`, `DEFAULT_INITIAL_SCORE` 0.6, `DEFAULT_SOURCE`.
- Text primitives shared with `sharing`: `tokens`, `jaccard`, `stopwords`.

## Literals
- Subject kinds are `skill` and `swarm-skill`; the legacy alias `team-skill` normalizes to `swarm-skill`.
- Targets are `description`, `body`, `script`; sections are `Instructions`, `Examples`, `Troubleshooting`, `Scripts`, `Collaboration`, `Roles`, `Constraints`, `Workflow`.
- Simplify actions are `DELETE`, `MERGE`, `REFINE`, `KEEP`.
- Online statuses are `staged`, `auto_approved`, `no_evolution_no_records`, `skipped_no_input`, `skipped_skill_not_found`, `skipped_skill_definition_not_found`, `persistence_failed`, `generation_failed`.
- Initial record confidence by source is `execution_failure` 0.65, `user_intent` 0.70, `script_artifact` 0.60, `conversation_review` 0.50, default 0.6, applied through `ExperienceRecord::confidence`.
- A draft summary is whitespace-collapsed, required, and at most 160 characters.

## Constraints
- Clock-free: recency is measured in archive sequence numbers (`created_seq`, `updated_seq`) and staleness in sequence steps, never wall-clock time; draft records use content-derived FNV-1a ids.
- Ranking is a fixed linear blend of keyword overlap, text overlap and observed success rate; `ExperienceRecord::score` is `weight * confidence * (0.5 + success_rate)`.
- Dedupe is content-token Jaccard at or above the archive threshold; a full archive evicts its lowest-scoring record.
- Constructors validate: blank record ids, skill names and content are rejected with `Error::validation`; draft normalization rejects a bad target, section, summary, script filename, source-ref type or merge id with a message naming the field and rule.
- Record ids are content-derived FNV-1a 64-bit hashes (`sig-`/`traj-`/`case-` plus 16 hex digits); orchestrator records use `ev-` plus 16 hex digits over the skill, target, section and content; `created_seq`/`updated_seq` are stamped only by `ExperienceArchive::add`.
- Serde is asymmetric: `ExperienceRecord`, `ExperienceQuery`, `ExperienceMatch`, `ExperienceSection`, `AddOutcome`, `MergeSummary`, `ExperienceStats`, `ExperienceArchive` (private fields), the draft types and the orchestrator result serialize, but `ExperienceExtractor`, `LifecycleConfig`, `LifecycleReport`, `ExperienceLifecycle`, `ExperienceDrafter`, `DraftInput`, `EvolutionRequest` and `OnlineEvolutionOrchestrator` do not.
- Deserializing an `ExperienceArchive` restores its records verbatim without re-validating them; callers must run `ExperienceRecord::validate` if the payload is untrusted.
- No new dependency: tolerant JSON extraction is hand-written (one outer fence, `//` comments outside `scheme://`, trailing commas, outermost `[...]` then `{...}`), with no `regex` and no `serde_json` repair crate.
- Drafting never errors on model trouble: only a blank skill name is an `Error::validation`; model errors, unparseable answers and unknown simplify ids fall back to the deterministic path.

## Design Decisions
- Sequence counters replace Python's wall-clock timestamp and free-form patch payload, so archives replay identically from a fixed input order.
- The archive is in-memory and JSON-file-free, unlike Python's per-skill JSON files; persistence is the caller's concern and the archive itself is `Serialize`.
- A never-used record still scores half of an equally weighted perfect record, so a fresh lesson is neither invisible nor dominant.
- Extraction is rule-based (confident signals, the first error step, successful multi-tool runs, sub-threshold cases) because the Python equivalent asks a model to draft the lesson.
- `LifecycleConfig` is a plain `Copy` threshold record with `Default`, so a lifecycle pass is a pure function of the archive and its config.
- The deterministic fallback is the spine of the drafter: `ExperienceExtractor::from_signals` then `from_trajectories` feed one `body` experience per record, so a model outage degrades drafting into extraction instead of failing it.
- The fallback marks every experience `target = body` because Python's optimizer owns placement decisions; the original record section still selects the draft section, and `reason` names the evidence source and confidence.
- An explicit all-`skip` model answer is honoured as an empty draft; only an answer with neither appended nor skipped items triggers the fallback, so the fallback cannot override a deliberate rejection of the evidence.
- One repair retry (format-repair prompt) precedes the fallback, matching Python's two-attempt LLM policy without its timeout/attempt policy machinery.
- The model answer is parsed into the draft schema before it reaches the archive, so a badly shaped experience fails normalization (and therefore falls back) instead of corrupting the archive.
- Script file names are rejected when they contain a separator, a drive/colon prefix, or are `.`/`..`, matching Python's `PurePosixPath`/`PureWindowsPath` name check without those libraries.
- `source_refs` drops blank and dash-only placeholders; the approval view keeps them and the persistence view drops them, so run-scoped evidence never enters a stored record.
- The simplify prompt path validates every action against the records the caller supplied; an unknown id falls back to the deterministic pass, and the deterministic rule deletes only records scored below 0.4 that were never used (Python's first maintenance rule).
- The orchestrator owns its archive so persistence and the archive's dedupe/merge rules stay in one place; the skill-store lookups are caller-supplied booleans instead of an async store trait.
- `complete` is public because generation and persistence are separable: `evolve` calls it with the drafter result, and a caller replaying a stored draft gets the same staging, persistence and status mapping.
- Draft metadata (`summary`, `reason`, script fields) rides in `ExperienceRecord::metadata`, the only free-form slot the existing record schema has.
- `prompts` is a private module re-exported through `drafter`, so the prompt constants keep their documented placeholders while keeping each file under the size guideline.
- The pipeline suite lives in `src/experience/pipeline_tests.rs` behind `#[cfg(test)]` rather than `tests/`, because the worker's write scope is the `src/experience` node; it drives `ScriptedModel` offline exactly as an integration binary would.

## Known Issues
- `ExperienceArchive::prune_below` documents that a non-finite threshold removes nothing, but a `NaN` threshold makes the `weight >= NaN` retain predicate false for every record and prunes the whole archive.
- The deterministic drafter ignores `DraftInput::user_query` and `existing` (they shape only the prompt), and it attributes every signal-derived lesson to the request's skill rather than the signal's own `skill_name`.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./models.rs` → `ExperienceRecord`, `ExperienceSection`, `ExperienceQuery`, `ExperienceMatch`
- `./archive.rs` → `ExperienceArchive`, dedupe/merge/rank/query, text primitives
- `./extract.rs` → `ExperienceExtractor`
- `./lifecycle.rs` → `ExperienceLifecycle`, `LifecycleConfig`, `LifecycleReport`
- `./draft.rs` → subject/target/section/action vocabulary, draft schema, normalizers, simplify-reference validation
- `./drafter.rs` → `ExperienceDrafter`, deterministic fallback, tolerant JSON parsing, prompt builders
- `./prompts.rs` → `DRAFT_PROMPT`, `FORMAT_REPAIR_PROMPT`, `SIMPLIFY_PROMPT`, `render_template`
- `./orchestrator.rs` → `OnlineEvolutionOrchestrator`, request/status/result, initial-score map, persistence
- `./pipeline_tests.rs` → test-only pipeline suite (model happy path, tolerance, repair, fallback, status matrix, persistence)
