# openjiuwen-memory / src / team

## Intent

Team-scoped memory: typed records shared by an agent-team runtime, with per-member privacy, tags, expiry, deterministic keyword recall, and optional durable persistence.
Ports the Python `openjiuwen.agent_teams.memory` package (`TeamMemoryManager`, `MemberMemoryToolkit`, the `extract_team_memories` extractor, and the `TEAM_MEMORY.md` shared summary) onto an explicit store contract plus caller-driven runtime adapters.
The Python package owns no storage engine: it wraps `core.memory.lite` for member memory and a single Markdown file for the team summary.
This node gives that model an explicit, queryable backend — `TeamMemoryRecord` rows behind the `TeamMemoryStore` trait (in-memory + optional SQLite), a projection into the crate's unified `MemoryRecord` — and keeps the Python runtime seams as separately testable adapters over the same facade:

- `document` — the `TEAM_MEMORY.md` document (render + lossless parse + facade import/export), compatible with the Python entry syntax.
- `extractor`/`prompts` — a `TeamMemoryExtractor` seam with a deterministic rule default and a model-backed implementation that falls back to the rules.
- `hooks` — `TeamMemoryHooks`: explicit, caller-driven lifecycle hooks (session start, member join/leave, turn buffering, session-end extraction).
- `toolkit` — `MemberMemoryToolkit`: the model-facing tool schemas plus a JSON dispatcher over the facade.
- `observe` — a `TeamMemoryObserver` seam with no-op and `tracing` implementations (the OTel exporter stays in `openjiuwen-extensions`).

This node depends only on the model types, `openjiuwen_common`, `openjiuwen_foundation` (through `MemoryRecord` and `Model`), and the crate's own `recall`/`config` helpers; it never touches `src/store/` internals or `openjiuwen-context`.

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

Document (`team::document`):
- `TEAM_MEMORY_FILE = "TEAM_MEMORY.md"`, `TEAM_MEMORY_MAX_READ_LINES = 200`, `TEAM_MEMORY_FORMAT = "openjiuwen-team-memory/v1"`.
- `TeamMemoryDocument { team_id, generated_at, entries }` with `from_records(team_id, entries, generated_at)`, `render()`, `parse(content, fallback_team_id)`, `len`, `is_empty`.
- `TeamMemory::export_markdown(&TeamMemoryFilter) -> String` and `TeamMemory::import_markdown(&str, &CallOptions) -> Vec<TeamMemoryRecord>`.
- `document_team_id(content) -> Option<String>` and `truncate_for_prompt(content) -> String` (the 200-line injection cap).

Extractor (`team::extractor` + `team::prompts`):
- `TeamTurn { from_member, to_member, content, at }`, `TeamTranscript { team_id, session_id, turns, task_notes }`, `TeamExtraction { records, degraded }`.
- `TeamMemoryExtractor` (async trait: `name`, `extract(&TeamTranscript, &CallOptions)`), `TeamExtractConfig`, `RuleTeamExtractor`, `LlmTeamExtractor`, `parse_team_extraction(raw, team_id, config)`.
- Constants: `RULE_SOURCE`, `LLM_SOURCE`, per-kind importances (`DECISION_IMPORTANCE` 0.7 … `CONTEXT_IMPORTANCE` 0.4), `TITLE_MAX_CHARS` 80, and the `DEFAULT_*` config values (max records 8, turn chars 1000, transcript chars 8000, min content 12, model retries 1).
- Prompts: `EXTRACTION_SYSTEM_PROMPT`, `EXTRACTION_RESPONSE_SCHEMA`, `TEAM_MEMORY_TAGS`, `render_transcript(&TeamTranscript, max_chars)`, `TASK_CONTENT_PREVIEW_MAX` 2000, `MESSAGE_CONTENT_PREVIEW_MAX` 1000, `TRUNCATION_MARKER`.
- `EXTRACTION_SYSTEM_PROMPT` stays path-qualified (`team::prompts::…`): `openjiuwen-memory`'s crate root already binds `graph::EXTRACTION_SYSTEM_PROMPT`, so `team/mod.rs` re-exports the other prompt items explicitly instead of globbing `prompts`.

Hooks (`team::hooks`):
- `TeamMemoryHooks::new(team_id, Arc<TeamMemory>, Arc<dyn TeamMemoryExtractor>, TeamMemoryHookConfig)` and `with_rule_extractor(team_id, memory, config)`.
- Accessors: `team_id`, `memory`, `extractor_name`, `config`, `session_id`, `pending_turns`, `members`.
- Hooks: `on_session_start(session_id, call)`, `on_member_join(member_id, call)`, `on_turn(turn, call)` (+ `sync_turn` alias), `on_session_end(call) -> TeamMemoryFlush`, `on_member_leave(member_id, call)`.
- `TeamMemoryHookConfig { enabled, member_profile_on_join, handoff_on_leave, flush_on_session_end, member_profile_importance, handoff_importance, max_pending_turns }` with `Default` + `validate`.
- `TeamMemoryFlush { records, degraded, turns }`; constants `HOOK_SOURCE = "team-memory-hooks"`, `DEFAULT_MEMBER_PROFILE_IMPORTANCE` 0.4, `DEFAULT_HANDOFF_IMPORTANCE` 0.6, `DEFAULT_MAX_PENDING_TURNS` 64.

Toolkit (`team::toolkit`):
- `MemberMemoryToolkit::new(Arc<TeamMemory>, MemberMemoryContext)` with `memory`, `context`, `tool_names`, `get_tool_schemas() -> Vec<Value>`, `handle_tool_call(name, args) -> Result<Value>`, `error_envelope(name, &Error) -> Value`.
- `MemberMemoryContext { team_id, member_id, session_id, read_only }` with `new`/`with_member`/`with_session`/`read_only`.
- Tool names: `TOOL_REMEMBER` (`team_memory_remember`), `TOOL_SEARCH` (`team_memory_search`), `TOOL_FORGET` (`team_memory_forget`), `TOOL_STATS` (`team_memory_stats`).

Observability (`team::observe`):
- `TeamMemoryOperation` (`search`, `write`, `forget`, `clear`, `prune`, `extract`), `TeamMemoryOutcome` (`ok`, `failed`).
- `TeamMemoryEvent { operation, team_id, member_id, kind, count, outcome, error }` with `new`, `with_member`, `with_kind`, `with_count`, `failed`, `is_failure`.
- `TeamMemoryObserver` (sync `fn observe(&TeamMemoryEvent)`), `NoopObserver`, `TracingObserver` (target `openjiuwen_memory::team`).
- Facade wiring: `TeamMemory::with_observer(Arc<dyn TeamMemoryObserver>)`, `observer()`, `emit(&TeamMemoryEvent)`.

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

## TEAM_MEMORY.md Format

Python's format is a flat list of `### [decision] <title>` + body entries with no front matter, ids or timestamps, injected only for its first 200 lines (`shared_memory.py`, `extractor.py`).
The Rust document keeps that entry syntax and makes it lossless:

```text
---
team_memory: openjiuwen-team-memory/v1
team_id: review-team
generated_at: 2026-01-02T03:04:05Z
entries: 2
---

# Team memory — review-team

## decision

### [decision] Storage choice
<!-- team-memory: {"id":"…","member_id":"alice","kind":"decision","title":"Storage choice","importance":0.8,"tags":["storage"],"metadata":{},"source":"leader","visibility":"team","created_at":"…","updated_at":"…","expires_at":null} -->
*id: … · member: alice · importance: 0.80 · tags: storage · updated: 2026-01-02*

Use SQLite for the shared store.
```

- The HTML-comment payload is the parse-time source of truth (so multi-line content, CJK, and content lines starting with `#` round-trip exactly); the heading/body/metadata line are the human view.
- `team_id` lives in the front matter and `content` in the body, so neither is repeated in the payload.
- Sections are the four `extractor_tag` values in the fixed order `decision`, `member`, `lesson`, `context`, then `custom`; empty sections are omitted.
- Entry order is kind rank, then `updated_at` descending, then `id` ascending; re-rendering a parsed document is byte-identical.
- Body lines beginning with `#` or `\` gain one leading `\` on export and lose it on parse (the only non-verbatim body byte path).
- A payload-less entry (a Python-authored file) parses from the heading tag (falling back to the enclosing section), with a deterministic `md-<16 hex FNV-1a>` id over team/kind/title/content, `DEFAULT_IMPORTANCE`, `source = "TEAM_MEMORY.md"`, `created_at = updated_at = generated_at` (the Unix epoch without front matter) and `Team` visibility — so re-importing the same file adds nothing.
- Unknown sections, stray headings and empty-body entries are skipped with a `tracing::warn!`; CRLF and unknown front-matter keys are tolerated; a bad `team_id` line or an unparseable payload is `Error::Validation` naming the 1-based line.
- An HTML comment cannot contain `--`, so a title or metadata value containing the literal `-->` cannot be represented; such a record renders a payload the parser rejects rather than dropping fields silently.
- `import_markdown` upserts in document order through the store (ids preserved, `created_at` preserved by `upsert`), so import is idempotent; `truncate_for_prompt` applies the Python 200-line injection cap.

## Lifecycle Hooks

`TeamMemoryHooks` is caller-driven: nothing registers in a global registry, starts a timer, or spawns a task.
One hooks object is bound to one `team_id`; state is a `parking_lot::Mutex<HookState>` (`session_id`, buffered `turns`, `task_notes`, `members`) never held across an `.await`.

| Hook | Python counterpart | Behaviour |
| --- | --- | --- |
| `on_session_start` | `bind_session_id` | validates and installs the session id, clears the previous session's buffered turns |
| `on_member_join` | manager construction (`init_toolkit`/`register_tools`) | tracks the roster and (unless disabled) writes one `MemberProfile` presence record |
| `on_turn` / `sync_turn` | — (deliberate addition: Python reads a team database at pause) | validates and buffers one turn, dropping the oldest past `max_pending_turns`; no store write |
| `on_session_end` | `extract_after_round` | extract → validate → deterministic ids → one batch write → clear the buffer |
| `on_member_leave` | `close` | removes from the roster and (unless disabled) writes one `HandoffNote` record |

`on_session_end` ordering is the contract: (1) snapshot the buffer under the lock and release it; (2) an empty buffer returns an empty flush, and a disabled flush keeps the buffer for a later manual call; (3) an extractor error (including `Error::Cancelled`) is returned unchanged with the buffer kept; (4) every record is validated *before* any write, so one invalid record means zero writes; (5) each id is overwritten with `hook-<16 hex FNV-1a>` over team/session/kind/content, so a retry upserts the same rows instead of duplicating them; (6) one `remember_many` call, then the buffer is cleared and `TeamMemoryFlush` returns the stored records, the extractor's `degraded` reason and the turn count.
Cancellation is checked at every hook entry (and between extractor attempts); a cancelled hook writes nothing and leaves the buffer unchanged.
Python swallows every hook error with a warning log; here every failure is a typed error returned to the caller plus one `Failed` observer event.

## Toolkit Tools

`MemberMemoryToolkit` is transport-neutral: `get_tool_schemas()` returns Python-shaped cards (`id` = `team_memory.<team>.<member|team>.<name>`, `name`, `description`, `input_schema`), and `handle_tool_call(name, args)` dispatches:

| Tool | Required | Optional | Result |
| --- | --- | --- | --- |
| `team_memory_remember` | `content` | `title`, `kind`, `tags`, `importance`, `member_id`, `visibility`, `expires_at` | the stored record |
| `team_memory_search` | `query` | `max_results`, `min_score`, `kinds`, `tags`, `include_expired` | `{count, results}` (ranked hits) |
| `team_memory_forget` | `id` | — | `{deleted, id}` |
| `team_memory_stats` | — | — | the visible-set statistics |

- Success is `{"ok":true,"tool":<name>,"result":<value>}`; every failure is a typed `Err` (`Validation` naming the field for unknown tools/bad args, `Unsupported` for a read-only context, `NotFound` for a delete the caller may not perform), and `error_envelope(name, &Error)` renders the same failure as `{"ok":false,…,"error":{"kind","message"}}` for a protocol layer.
- Reads use `TeamMemoryScope::allows`; deletes additionally require team visibility or own authorship, so another member's private record is `NotFound` (existence is not leaked) and an unknown id is `deleted: false`.
- Unknown argument names are rejected; `importance`/`min_score` must be finite, `max_results` in `1..=100`; writes carry the context session and default the member to the caller.
- Deliberately not ported: Python's file tools (`memory_get`/`read_memory`/`write_memory`/`edit_memory`), the coding-scenario tool set, the `MEMORY_ENABLED` env switch, the process-global tool registry and `MemoryIndexManager` lifecycle.

## Extraction

`TeamMemoryExtractor` turns a `TeamTranscript` into `NewTeamMemory` records; extraction never touches the store (the hooks own the write).
`RuleTeamExtractor` (default, offline) scans each turn for documented markers, first match wins: `decision` ("we decided", "chose to", "instead of", 决定/选择) → `lesson` ("learned", "should have", "root cause", 教训/经验) → `member` ("is good at", "owns", 擅长/负责) → `task` ("action item", "todo", 待办) → `context` ("constraint", "deadline", 约束/背景).
ASCII markers require word boundaries; CJK markers match as substrings; content shorter than `min_content_chars` is skipped; records are deduped by normalized content, truncated to `max_turn_chars`, tagged with the matched category plus the kind's `extractor_tag`, `source = "rule-extractor"`, `visibility = Team`, and capped at `max_records` (task notes become `Task` records after the turn records).
`LlmTeamExtractor` renders the transcript with `render_transcript` (Python's context shape: `# Team collaboration record for <team>`, `## Tasks`, `## Conversation`, per-item caps 2000/1000 chars, `… [truncated]` marker), sends `EXTRACTION_SYSTEM_PROMPT` and the transcript, and parses the answer with `parse_team_extraction` (fenced or bare JSON; `{"memories":[…]}` or a bare array; `kind`/`content`/`title`/`tags`/`importance`/`member_id`/`session_id`/`visibility`).
Provider errors, an empty answer and an unparseable answer each consume one of `model_retries + 1` attempts; after the budget the rule extractor runs and the result carries `degraded = Some("llm team memory extraction fell back to rules: <first failure>")`. `ErrorKind::Cancelled` propagates without falling back. `enabled = false` yields an empty extraction in both implementations.
The answer shape (`{"memories":[{"kind":"decision","content":"…","title":"…","tags":["…"],"importance":0.7,"member_id":"…","visibility":"team"}]}`) and its JSON Schema live in `prompts.rs` beside the prompt.
Python instead runs a DeepAgent whose only output is file writes (no parsing, no dedupe, no thresholds); the Rust seam returns typed records so validation, caps and the fallback are explicit.

## Observability

`TeamMemory::with_observer` attaches a `TeamMemoryObserver`; the default is `None`, so an unobserved facade behaves and costs exactly as before.
Emitted events: `remember`/`update` (`Write`, one record + kind), `remember_many` (one `Write` with the stored count), `search` (`Search`, hit count), `forget` (`Forget`, 1/0), `clear` (`Clear`), `prune_expired` (`Prune`), and `Extract` from the hooks' session-end flush.
`get`/`list`/`stats`/`set_expiry`/`to_jsonl`/`from_jsonl` emit nothing: they are inspection/export paths, and observing them would let a monitoring observer generate its own traffic.
A failure emits one `Failed` event carrying the error's `Display` text and still returns the same typed error unchanged.
`TracingObserver` logs at target `openjiuwen_memory::team` (`debug!` for `ok`, `warn!` for `failed`) with operation/team/member/kind/count/outcome/error fields; `NoopObserver` is the explicit no-op.
This crate deliberately ships no exporter: the OTel/OTLP implementation lives in `openjiuwen-extensions`, and a bridge is a thin `TeamMemoryObserver` over that SDK.
Python has only `memory_logger` warning lines, so the event seam is a Rust addition with no wire-format consequence.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for bad input, `NotFound` for an update of an unknown id, `Cancelled` for a triggered token; storage errors propagate with their original kind.
- Async-first on tokio; no blocking I/O on async tasks; `parking_lot::RwLock`/`tokio` locks are never held across `.await`; the SQLite backend uses the shared `sqlx` pool.
- Every public data type is `serde`-serializable, `snake_case`, and tolerant of unknown/missing fields (container `default` plus manual `Default` impls).
- No `unsafe`; `tracing` only with target `openjiuwen_memory::team`; no `println!`/`eprintln!`; no `unwrap`/`expect`/panic outside `#[cfg(test)]`.
- Files stay around ≤1000 lines; the module is split into `model/`, `store/`, `sqlite/`, `projection.rs`, `document.rs`, `extractor.rs`, `prompts.rs`, `hooks.rs`, `toolkit.rs` and `observe.rs` by responsibility.
- `extractor.rs`, `document.rs`, `toolkit.rs` and `hooks.rs` exceed that baseline because each carries an inline `#[cfg(test)] mod tests` of 500–650 lines covering one scenario group per behaviour; their implementation bodies are ~650–1120 lines, so the size is test bulk rather than mixed responsibilities.
- No global bus, registry, timer or background task: hooks, observers and extractors are injected into the object that uses them, and every lifecycle step is an explicit call.
- The observer is `None` by default and never changes behaviour; the hooks and toolkit never emit their own store writes outside the facade.
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

- Python's `register_tools`/`strip_rails_by_type`/`load_and_inject` wiring and the per-member `MemoryIndexManager` lite stack: the agent-team runtime (`openjiuwen-agent-teams`) owns tool registration, rail stripping and prompt-section injection; this node exposes the hooks, the toolkit and the document, not the harness plumbing.
- The `MemoryProvider` lifecycle surface of `core/memory/external` (`prefetch`, `sync_turn`, `system_prompt_block`, `handle_tool_call`, `on_session_end`) is a different subsystem (external memory providers) and stays out of this node; `TeamMemoryHooks` mirrors the team manager's hooks, not that ABC.
- Python's extractor DeepAgent (its `read_memory_file`/`write_memory_file`/`list_memory_files` tools, the iteration loop and the model-rewrites-the-file strategy) is not reproduced: the Rust extractor returns typed records, the hooks write them through the facade, and the prompt asks for JSON instead of file edits.
- The Python `MEMORY_ENABLED` environment switch, the `scenario: "coding"` tool set and the coding-memory tools are not ported; a read-only `MemberMemoryContext` is the only capability switch.
- The team summary is exported/imported on demand, not rewritten by the extractor on every round, so Python's "keep the file ≤200 lines by merging/deleting" prompt instruction has no enforcement here; `truncate_for_prompt` applies the 200-line cap only when injecting.
- Cross-process/distributed team store beyond a SQLite file (network databases, Redis, replication, distributed locks) and a cross-process memory bus are deferred; `sqlite` is the only durable backend, and hooks never coordinate between processes.
- A retention daemon, timer-driven pruning, background consolidation/dreaming for team memory, and monitor/UI reporting are deferred; the caller owns `prune_expired` and `on_session_end`.
- The OTel exporter itself (spans, metrics, attribute names) is deferred to `openjiuwen-extensions`; this node ships the `TeamMemoryObserver` seam, the no-op and the `tracing` implementation only.
- An org/tenant layer above `team_id` (membership services, tenant isolation, cross-team sharing) is not modelled.
- Embedding-based team recall is deferred: the schema has no embedding column, so search is keyword + importance + recency.

## Test Strategy

- Unit tests live beside every module (`#[cfg(test)] mod tests`): validation/normalization, scope visibility matrix, kind/visibility serde, record/input/update semantics, filter clauses, ranking formula, in-memory store behaviour, facade cancellation/JSONL, projection round-trip matrix, and SQLite CRUD/filters/search/stats/expiry/decode.
- New suites: `document` (lossless export→parse round trip across every kind/visibility/CJK/multi-line/`#`-prefixed content, render determinism, Python-authored import, idempotent re-import, SQLite export/import across a reopen), `extractor`/`prompts` (rule marker inventory, dedupe/caps, parse variants, `ScriptedModel` happy path, fallback with a `degraded` reason, cancellation), `hooks` (full sequence, retry after extractor failure, invalid-output rollback, cancellation, buffer bounds, config switches, observer events, SQLite persistence across a reopen), `toolkit` (schema shape, happy paths, argument validation, privacy/deletes, read-only contexts, envelope), `observe` (event builders/serde, emit matrix, failure events).
- Counts: the `team` filter holds 185 tests without `sqlite` and 225 with `--features sqlite`; the crate as a whole is 972 tests with default features and 1058 with `--all-features`.
- Deterministic offline data: fixed `DateTime::from_timestamp` values, `tempfile::tempdir()` for SQLite files, `ScriptedModel` for the model-backed extractor, no network, no wall-clock ordering.
- Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo clippy -p openjiuwen-memory --all-targets --all-features -- -D warnings`, `cargo test -p openjiuwen-memory` with and without `--all-features`, and `cargo doc -p openjiuwen-memory --no-deps --all-features` warning-free.

## Routing Table

- `./model/` → `TeamMemoryScope`, `TeamMemoryKind`, `TeamVisibility`, `TeamMemoryRecord`, `NewTeamMemory`, `TeamMemoryUpdate`, `TeamMemoryFilter`, stats/hit types, validation helpers (`model/mod.rs`)
- `./store.rs` → `TeamMemoryStore` trait, `TeamSearchOptions`, shared ranking (`store.rs`)
- `./store/memory.rs` → `InMemoryTeamMemoryStore`
- `./store/facade.rs` → `TeamMemory` facade, JSONL import/export, `TeamMemory::sqlite`
- `./sqlite.rs` → `SqliteTeamMemoryStore` and its `schema`/`query`/`decode` helpers plus the feature-gated test suite (`sqlite/`)
- `./projection.rs` → `TeamMemoryRecord` ⇄ `MemoryRecord` projection and `TEAM_MEMORY_METADATA_KEY`
- `./document.rs` → `TEAM_MEMORY.md` format: front matter, per-kind sections, payload codec, `TeamMemoryDocument`, `TeamMemory::export_markdown`/`import_markdown`, `truncate_for_prompt`
- `./extractor.rs` → `TeamTurn`/`TeamTranscript`/`TeamExtraction`, `TeamMemoryExtractor` seam, `RuleTeamExtractor`, `LlmTeamExtractor`, `parse_team_extraction`, `TeamExtractConfig`
- `./prompts.rs` → `EXTRACTION_SYSTEM_PROMPT`, `EXTRACTION_RESPONSE_SCHEMA`, `TEAM_MEMORY_TAGS`, `render_transcript`, preview caps and truncation marker
- `./hooks.rs` → `TeamMemoryHooks` lifecycle seam, `TeamMemoryHookConfig`, `TeamMemoryFlush`, `on_session_end` ordering and idempotent ids
- `./toolkit.rs` → `MemberMemoryToolkit`, `MemberMemoryContext`, the four tool schemas and the JSON dispatcher/`error_envelope`
- `./observe.rs` → `TeamMemoryObserver` seam, `TeamMemoryEvent`/operations, `NoopObserver`, `TracingObserver`
