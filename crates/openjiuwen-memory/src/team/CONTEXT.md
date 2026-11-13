# openjiuwen-memory / src / team

## Intent

Team-scoped memory: typed records shared by an agent-team runtime, with per-member privacy, tags, expiry, deterministic keyword recall, and optional durable persistence.
Ports the memory *data* layer of the Python `openjiuwen.agent_teams.memory` package (the `TEAM_MEMORY.md` shared summary plus the per-member private store) into a store contract instead of a Markdown file.
The Python package owns no storage engine: it wraps `core.memory.lite` for member memory and a single Markdown file for the team summary.
This module gives that model an explicit, queryable backend: `TeamMemoryRecord` rows behind the `TeamMemoryStore` trait, an in-memory reference backend, an optional SQLite backend, and a projection into the crate's unified `MemoryRecord`.
This node depends only on the model types, `openjiuwen_common`, `openjiuwen_foundation` (through `MemoryRecord`), and the crate's own `recall`/`config` helpers; it never touches `src/store/` internals or `openjiuwen-context`.

Mapping to the Python runtime:
- `TeamMemoryScope::team_id` is the Python team scope id (the team name that also prefixes each member's `agent_id`).
- `TeamMemoryScope::member_id` is a member's user id (`f"{team_name}.{member_name}"` in Python); `None` means a team-wide record or the leader's team-wide view.
- `TeamMemoryScope::session_id` is the coordination session that wrote the memory.
- `TeamMemoryKind::extractor_tag` maps kinds onto the four `TEAM_MEMORY.md` section tags (`decision`, `lesson`, `member`, `context`) the Python extractor writes.

## API Surface

Crate path: `openjiuwen_memory::team` (also re-exported at the crate root as `openjiuwen_memory::TeamMemoryRecord` etc.).

Model (`team::model`, plain `serde` data, no I/O):
- `TeamMemoryScope { team_id, member_id: Option<String>, session_id: Option<String> }` with `new`, `member`, `with_member`, `with_session`, `validate`, and `allows(&record)` (the read rule).
- `TeamMemoryKind` — `SharedFact` (default), `MemberProfile`, `Decision`, `Task`, `Artifact`, `HandoffNote`, `Custom(String)`; `as_str`, `BUILTIN`, `is_builtin`, `extractor_tag`, `parse`, `FromStr`, `Display`; serde is a plain string and unknown strings become `Custom`.
- `TeamVisibility` — `Team` (default), `Members(Vec<String>)`, `Private`; `members(iter)` normalizes the list, `member_list`, `validate`.
- `TeamMemoryRecord` — `id`, `team_id`, `member_id`, `session_id`, `kind`, `content`, `title`, `importance`, `tags`, `metadata: serde_json::Value`, `created_at`, `updated_at`, `source`, `visibility`, `expires_at`; `new(scope, content, kind)`, builders, `is_expired(now)`, `metadata_object`, `metadata_str`, `validate`.
- `NewTeamMemory` — constructors `shared_fact`, `member_profile`, `decision`, `task`, `artifact`, `handoff_note`, builders, `validate`, `to_record(now, default_importance)`.
- `TeamMemoryUpdate` — builders, `is_empty`, `validate`, `apply(&mut record, now)`.
- `TeamMemoryFilter` — `for_team` (mandatory team id), `kind`, `member`, `with_limit`, `with_offset`, `validate`, `matches(record, now)`.
- `TeamMemoryStats { total, by_kind, by_member, expired, oldest, newest }`; team-wide records count under `TEAM_WIDE_MEMBER_KEY = "__team__"`.
- `TeamSearchHit { #[serde(flatten)] record, score, keyword, importance, recency }`.
- `normalize_tags`, `new_team_memory_id`, and the limit constants (`MAX_SCOPE_ID_CHARS` 128, `MAX_RECORD_ID_CHARS` 128, `MAX_CONTENT_CHARS` 16384, `MAX_TITLE_CHARS` 256, `MAX_TAGS` 32, `MAX_TAG_CHARS` 64, `MAX_VISIBILITY_MEMBERS` 256, `DEFAULT_IMPORTANCE` 0.5).

Store (`team::store`):
- `#[async_trait] TeamMemoryStore: Send + Sync` — `upsert`, `get(team_id, id)`, `list(filter)`, `search(scope, query, options)`, `delete`, `clear`, `stats`, `set_expiry`, `prune_expired(now)`.
- `TeamSearchOptions { top_k, threshold, kinds, tags, include_expired, call }` (`CallOptions`, so it is not serde) with builders and `validate`.
- `InMemoryTeamMemoryStore` — `new`, `len`, `is_empty`, `Default`.
- `TeamMemory` facade — `new(store)`, `in_memory()`, `sqlite(path)` (feature `sqlite`), `with_default_importance`, `default_importance`, `store`, `remember`, `remember_many`, `get`, `update`, `list`, `search`, `forget`, `clear`, `stats`, `set_expiry`, `prune_expired`, `to_jsonl`, `from_jsonl`.
- Constants `DEFAULT_SEARCH_TOP_K = 5`, `DEFAULT_SEARCH_THRESHOLD = 0.3`, `SEARCH_RECENCY_HALF_LIFE_SECS = 2_592_000`.
- `rank_candidates(records, query, options, now)` is the crate-internal ranking shared by every backend.

SQLite (`team::sqlite`, feature `sqlite`):
- `SqliteTeamMemoryStore` — `open(path)`, `connect(url)`, `from_pool(pool)`, `pool()`, full `TeamMemoryStore` impl.
- `TeamMemory::sqlite(path)` is the facade constructor.

Projection (`team::projection`):
- `TeamMemoryRecord::to_memory_record()` and `TeamMemoryRecord::from_memory_record(&MemoryRecord)`.
- `TEAM_MEMORY_METADATA_KEY = "team_memory"` is the reserved metadata marker.

## Record JSON Shape

`snake_case`, tolerant of unknown fields, `skip_serializing_if` on absent optionals:
```json
{
  "id": "018f1d2e3f4a5b6c7d8e9f01",
  "team_id": "review-team",
  "member_id": "alice",
  "session_id": "s-1",
  "kind": "decision",
  "content": "Use SQLite for the shared store.",
  "title": "Storage choice",
  "importance": 0.8,
  "tags": ["storage", "sqlite"],
  "metadata": {"issue": 42},
  "created_at": "2026-01-02T03:04:05Z",
  "updated_at": "2026-01-02T03:04:05Z",
  "source": "leader-extraction",
  "visibility": {"members": ["alice", "bob"]},
  "expires_at": null
}
```
`visibility` is externally tagged: `"team"`, `"private"`, or `{"members": ["alice", "bob"]}`.
`kind` is a plain string, so an unknown kind round-trips as `TeamMemoryKind::Custom`.
`metadata` must be a JSON object; `importance` is in `0.0..=1.0`; timestamps are RFC 3339 UTC.

## Read Rule and Validation

`TeamMemoryScope::allows(record)`:
- the team must match;
- a scope `session_id` admits records of that session and session-less records;
- a scope *without* `member_id` is the team-wide (leader/admin) view and admits every visibility;
- a scope *with* `member_id` admits `Team` records, admits `Private` records only when the scope member authored them, and admits `Members(list)` records when the member authored them or is listed.

Validation rules (`Error::Validation`, message names the field and offending value):
- ids (team/member/session/record) are non-empty, NUL-free, free of `/` and `\`, control-character-free, and at most 128 characters;
- content is non-whitespace and at most 16384 characters; a title is non-whitespace and at most 256 characters;
- importance is finite and inside `0.0..=1.0`; `NewTeamMemory::to_record` and `TeamMemoryUpdate::apply` clamp finite out-of-range values, non-finite input is rejected;
- tags are trimmed, lowercased, deduplicated, non-empty, at most 64 characters, and at most 32 per record; a *stored* record rejects case-insensitive duplicates;
- `TeamVisibility::Members` names 1..=256 members;
- a filter requires a non-empty `team_id` and a `limit` greater than zero when set.

## Store Contract

- A `TeamMemoryStore` is the single source of truth; the facade caches nothing and the projection is derived from stored rows.
- `upsert` validates the record, overwrites the row with that `id`, and keeps the existing `created_at` (a new row uses the record's `created_at`); `updated_at` always comes from the input.
- `get` is a direct lookup: it returns expired records too and `None` for an unknown id or another team.
- `list` validates the filter, matches at `Utc::now()`, applies `offset`/`limit`, and orders by `updated_at` descending then `id` ascending.
- `search` validates scope and options, keeps records the scope `allows`, drops expired records unless `include_expired`, drops candidates with zero keyword overlap, ranks, and truncates to `top_k`.
- `delete` returns `false` when absent; `clear` requires a filter (always team-scoped) and returns the deleted count; `set_expiry` returns `false` when absent or owned by another team and never touches `updated_at`; `prune_expired(now)` deletes rows with `expires_at <= now` across teams and returns the count.
- `stats` aggregates the filter-matched set with per-kind counts, per-member counts, the expired count, and the oldest/newest `created_at`.
- The facade adds cancellation: `CallOptions::check` runs before and after each store call, and a triggered token yields `Error::Cancelled` without a partial write beyond records already committed by `remember_many`/`from_jsonl`.
- `update` returns `Error::NotFound` for an unknown id; `remember_many` stops at the first error keeping earlier writes; `to_jsonl` emits one compact JSON object per line with a trailing newline, and `from_jsonl` skips blank lines but rejects a malformed line with `Error::Validation` naming the 1-based line number.

## Search Ranking

```
document = title + " " + content + " " + tags.join(" ")
keyword  = normalized_bm25_scores(query, documents)      // 0 when no term overlaps
recency  = 0.5 ^ (age_secs(updated_at) / 30 days)
score    = blend(0.0, keyword, importance, recency, RankingConfig { similarity_weight: 0.0, ..default })
```
Because `similarity_weight` is zero, the semantic group renormalizes to the keyword term, giving `score = 0.85 * keyword + 0.10 * importance + 0.05 * recency`.
Candidates with `keyword == 0.0` are dropped before ranking, so a recent or important but unrelated record is never returned.
Ordering: `score` descending, then `updated_at` descending, then `id` ascending (deterministic for identical scores).

## SQLite Schema and Behaviour

```sql
CREATE TABLE IF NOT EXISTS team_memories (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    member_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL,
    visibility_json TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    importance REAL NOT NULL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);
CREATE INDEX IF NOT EXISTS team_memories_team_updated_idx ON team_memories (team_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS team_memories_team_kind_idx ON team_memories (team_id, kind);
CREATE INDEX IF NOT EXISTS team_memories_team_member_idx ON team_memories (team_id, member_id);
```
- Timestamps are RFC 3339 UTC strings with millisecond precision (`2026-01-02T03:04:05.000Z`), so lexical ordering equals time ordering; `expires_at` is `NULL` when absent.
- Every statement is parameter-bound; dynamic predicates are built with a private `Where`/`Param` helper using numbered `?N` placeholders.
- `list`/`clear`/`stats` express the whole filter in SQL (team/member/session, `kind IN (...)`, exact `visibility_json`, tags via `json_each`, `text_contains` via escaped `LIKE`, importance, time bounds, expiry) and `list` applies `LIMIT`/`OFFSET` in SQL.
- `search` selects the team's rows and reuses the shared `rank_candidates`, so ranking cannot drift between backends.
- Degradation (documented, never fails a query): malformed `visibility_json` → `TeamVisibility::Private` (fail closed); malformed `tags_json` → `[]`; malformed/non-object `metadata_json` → `{}`; non-finite or out-of-range `importance` → clamped or `DEFAULT_IMPORTANCE`; an unparseable `created_at`/`updated_at` skips the row; an unparseable `expires_at` means no expiry. Every degradation logs `tracing::warn!(target: "openjiuwen_memory::team", ...)`.
- Deliberate backend differences: SQLite `lower`/`LIKE` fold ASCII only (the in-memory backend folds full Unicode), and SQLite stores millisecond-precision timestamps.

## Projection Rules

`TeamMemoryRecord::to_memory_record()` builds a `MemoryRecord` view for unified recall:
- `scope.scope_id = team_id`; `scope.user_id = member_id` (falling back to `team_id` for team-wide records); `scope.session_id` copied; `agent_id = None`;
- `kind`: `MemberProfile` → `MemoryType::UserProfile`, everything else (including `Custom`) → `MemoryType::SemanticMemory`;
- `content`, `id`, `importance`, `source` copied; `timestamp = created_at`; `updated_at` copied; `last_access = None`, `access_count = 0`, `embedding = None`;
- the record's own metadata entries are kept (minus the reserved key) and a reserved `team_memory` object is added with `team_id`, `member_id`, `session_id`, `kind`, `title`, `tags`, `visibility`, and `expires_at`.

`TeamMemoryRecord::from_memory_record(&MemoryRecord)` requires the reserved `team_memory` object (a plain `MemoryRecord` is rejected with `Error::Validation`), rebuilds every team field from it, takes `content`/`importance`/`source`/timestamps from the record, restores the non-reserved metadata entries, and validates the result.
Round-tripping is lossless for every kind (including `Custom`) and every visibility, including title, tags, metadata, source, expiry, importance, id, and both timestamps.
The projection is a view: it is not a second source of truth, and writing it back into the long-term memory engine does not update the team store.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for bad input, `NotFound` for an update of an unknown id, `Cancelled` for a triggered token; storage errors propagate with their original kind.
- Async-first on tokio; no blocking I/O on async tasks; `parking_lot::RwLock`/`tokio` locks are never held across `.await`; the SQLite backend uses the shared `sqlx` pool.
- Every public data type is `serde`-serializable, `snake_case`, and tolerant of unknown/missing fields (container `default` plus manual `Default` impls).
- No `unsafe`; `tracing` only with target `openjiuwen_memory::team`; no `println!`/`eprintln!`; no `unwrap`/`expect`/panic outside `#[cfg(test)]`.
- Files stay around ≤1000 lines; the module is split into `model/`, `store/`, `sqlite/`, and `projection.rs` by responsibility.
- Default features compile and test offline (in-memory backend); the `sqlite` feature adds a bundled-SQLite backend and tests over temp directories.
- Time is injectable: `list`/`stats`/`search`/`is_expired` take `now`; the facade and filters use `Utc::now()` at the boundary, so tests use fixed timestamps.

## Notes for Agents

- The crate root re-exports this module (`pub use team::*;`), so team types are reachable at `openjiuwen_memory::…` as well as `openjiuwen_memory::team::…`.
- `TeamMemoryKind::Custom` preserves the original string verbatim; only the known aliases are normalized, so a custom kind's case is significant.
- `TeamMemoryFilter` is always team-scoped: there is no cross-team listing by design (`prune_expired` is the only cross-team operation).
- `TeamMemory::in_memory()` is the offline reference backend; `TeamMemory::sqlite(path)` is the durable one and shares the trait.
- The JSONL format is the record JSON itself (one per line), so an export can be inspected, edited, and re-imported with `from_jsonl`.
- Adding a backend means implementing `TeamMemoryStore` and (for search) calling `rank_candidates`; do not reimplement the ranking formula.

## Deferrals

- Team-memory sync/bus integration (the Python `TeamMemoryManager` lifecycle hooks, `MemberMemoryToolkit` tools, and `register_tools` wiring) is not part of this data layer; `openjiuwen-agent-teams` owns the runtime.
- The Python `TEAM_MEMORY.md` / `SharedMemoryManager` file format and its 200-line read cap are not reproduced; records plus `kind.extractor_tag()` carry the same information, and Markdown rendering is not implemented here.
- The Python extractor agent (`extract_team_memories`) and its LLM prompt are not ported; callers write records through the facade.
- Cross-process/distributed team store beyond a SQLite file (network databases, Redis, replication, locks) is deferred; `sqlite` is the only durable backend.
- A retention daemon, background consolidation/dreaming for team memory, OTel metrics/tracing of team recall, and an org/tenant layer above `team_id` are deferred.
- Embedding-based team recall is deferred: the schema has no embedding column, so search is keyword + importance + recency.

## Test Strategy

- Unit tests live beside every module (`#[cfg(test)] mod tests`): validation/normalization, scope visibility matrix, kind/visibility serde, record/input/update semantics, filter clauses, ranking formula, in-memory store behaviour, facade cancellation/JSONL, projection round-trip matrix, and SQLite CRUD/filters/search/stats/expiry/decode.
- Counts: the model layer contributes 60 tests, store and projection bring the `team` filter to 103 without `sqlite`, and the SQLite backend brings it to 139 with `--features sqlite`.
- Deterministic offline data: fixed `DateTime::from_timestamp` values, `tempfile::tempdir()` for SQLite files, no network, no wall-clock ordering.
- Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo check`/`cargo clippy --all-targets -- -D warnings` with and without `--features sqlite`, `cargo test -p openjiuwen-memory --lib team` with and without `sqlite`, and `cargo doc -p openjiuwen-memory --no-deps` warning-free.

## Routing Table

- `./model/` → `TeamMemoryScope`, `TeamMemoryKind`, `TeamVisibility`, `TeamMemoryRecord`, `NewTeamMemory`, `TeamMemoryUpdate`, `TeamMemoryFilter`, stats/hit types, validation helpers (`model/mod.rs`)
- `./store.rs` → `TeamMemoryStore` trait, `TeamSearchOptions`, shared ranking (`store.rs`)
- `./store/memory.rs` → `InMemoryTeamMemoryStore`
- `./store/facade.rs` → `TeamMemory` facade, JSONL import/export, `TeamMemory::sqlite`
- `./sqlite.rs` → `SqliteTeamMemoryStore` and its `schema`/`query`/`decode` helpers plus the feature-gated test suite (`sqlite/`)
- `./projection.rs` → `TeamMemoryRecord` ⇄ `MemoryRecord` projection and `TEAM_MEMORY_METADATA_KEY`
