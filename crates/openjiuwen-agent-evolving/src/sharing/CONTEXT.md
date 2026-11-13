# sharing

## Intent
Packaging experience records into repeatable skill bundles: the wire types, deterministic keyword extraction, the approval-gated stager, and the pluggable backend seam.
Ports the offline part of Python `openjiuwen.agent_evolving.sharing`: `types.py`, `keyword_extractor.py`, `share_stager.py` and `backends/local_file.py`.
The remote hub client is not ported; the backend seam is real and testable offline.

## API Surface
- `SharingMeta`, `SharedExperience`, `SharedSkillBundle`, `SkillPackageMeta`, `SkillSearchResult`, `QueryKeywords` — pure serde wire data.
- `SharedSkillBundle::new` derives `bundle_id` as `bundle-{16 hex}` of an FNV-1a hash over the skill id and the sorted distinct record ids; `SharedExperience` carries the record payload plus its provenance.
- `KeywordExtractor` — deterministic keyword and summary extraction (`extract`), with the shared `experience::stopwords` vocabulary.
- `ShareStager` — `min_confidence`, `require_approval`, `max_experiences`; `build_bundle(records, meta)` filters, dedupes, wraps and keywords; `publish`/`share` enforce the approval gate; `StagedBundle` is the pending result.
- `SharingMeta::default()` uses `upload_trigger = "user_approval"` and `confidence = 0.7`; `SharedSkillBundle::new` requires a non-blank `skill_id` and derives `bundle_id` itself.
- `SharingBackend` (async trait) — `publish`, `list`, `get`, `remove`, `search`; implementations `LocalFileSharingBackend` (one JSON file per bundle id) and `MemorySharingBackend`.
- Both backends share one ranking rule, `score = 0.7 * keyword_ratio + 0.3 * intent_ratio`.

## Constraints
- Deterministic: bundle ids are derived from content, never from `uuid4` or a clock, so re-publishing the same records yields the same id.
- The stager never mutates or deletes input records; a rejected record stays in the local archive untouched.
- `build_bundle` drops records that fail `ExperienceRecord::validate` or fall below `min_confidence`, and rejects a bundle with no surviving record; `max_experiences` `0` means unlimited.
- Backends are async and return `Error::Io`; the in-memory backend guards its map with a `Mutex` that is never held across an `.await`.

## Design Decisions
- A content-derived `bundle_id` replaces Python's random one, which keeps tests, re-publishes and file names reproducible.
- The offline port collapses Python's stager-plus-upload-queue into one `build_bundle` step and leaves approval explicit in `publish`/`share`.
- The trait takes string ids rather than paths so a future remote hub or database slots in beside the two offline backends.
- Keyword extraction is local tokenising and ranking instead of a model call, so bundles carry usable retrieval metadata with no network.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./types.rs` → wire types, `SharedSkillBundle`, bundle-id derivation
- `./keywords.rs` → `KeywordExtractor`
- `./stager.rs` → `ShareStager`, `StagedBundle`, approval and quality gates
- `./backend.rs` → `SharingBackend`, `LocalFileSharingBackend`, `MemorySharingBackend` and their private `score_bundle` ranking
