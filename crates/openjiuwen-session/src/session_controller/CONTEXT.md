# session_controller

## Intent

Disk-backed product sessions: scope-isolated session storage (`sessions.json` + `state.data`), the per-agent controller cache in front of it, and the persisted session link graph.
Ports the storage subset of `openjiuwen/core/session/session_controller/` (v0.1.20): `scope.py`, `schema.py`, the storage half of the controller, and the downlink relationships Python keeps as `.link` files beside a data container.
The stored `data` document is exactly `SessionSnapshot` JSON, so a record loads straight into a concrete session through `Session::restore`.
The link graph is metadata (`links.json` per agent); state propagation belongs to the crates that own a run loop, so Python's `ChainSession` is deliberately not ported.

## API Surface

### `scope` — isolation keys (`src/session_controller/scope.rs`)

- `Scope { Main /* "main" */, Custom(String) }` — `name()`, `Display`, `FromStr` (empty or colon-containing names are `Error::Validation`), serde through the string form.
- `Subject { Direct { user_id } /* direct:<user> */, Group { group_id } /* group:<group> */, GroupUser { group_id, user_id } /* group:<group>:user:<user> */ }`; a `group-user-v2:` form with percent-escaped ids carries group/user ids that themselves contain colons, and legacy percent sequences stay literal.
- `SessionScope { scope, subject: Option<Subject> }` — constructors `main()`, `direct(user_id)`, `group(group_id)`, `group_user(group_id, user_id)`, `custom(scope_name)`; string forms `"main"`, `"main:direct:u1"`, `"main:group:g1"`, `"main:group:g1:user:u1"`, or a custom scope name.
- `SessionScopeKey { agent_id, scope }` — the `sessions.json` key, rendered `agent:<agent_id>:<scope>`.
- All four types are `Display` + `FromStr` + serde-through-string; malformed values are `Error::Validation` (a missing `agent:` prefix, an empty agent id, an unknown subject form).

### `schema` — persisted documents (`src/session_controller/schema.rs`)

- `SessionMeta { session_id, created_at, updated_at /* UTC epoch seconds, f64 */, version, is_active, data_container_type, title, metadata }` with `new` (active, version 1) and `touch` (strictly monotonic `updated_at` via `TIMESTAMP_EPSILON`, version bump); `DEFAULT_DATA_CONTAINER_TYPE = "agent"`.
- `ScopeSessionsMeta { session_scope_key, active_session, sessions }` — `new`, `get_session`, `upsert` (an active meta deactivates every sibling and becomes `active_session`), `remove` (clears `active_session` when it matched), `active`, `sort_sessions` (newest `updated_at` first, ties by session id).
- `SessionsIndex { agent_id, updated_at, scopes }` — the `sessions/sessions.json` document, `scopes` keyed by `SessionScopeKey` string.
- `SessionRecord { meta, snapshot }` — serialized exactly as `{"meta": …, "data": <SessionSnapshot>}` (`snapshot` carries `#[serde(rename = "data")]`), so a `data` payload restores with no adapter.
- Every document is `snake_case`, tolerant of unknown fields, with `#[serde(default)]`; `now_seconds()` is the crate-internal clock.

### `store` — disk layout and controller (`src/session_controller/store.rs`)

- Layout under a base directory: `<base>/<agent_id>/sessions/sessions.json` (index), `<base>/<agent_id>/sessions/<session_id>/state.data` (record), `<base>/<agent_id>/sessions/<session_id>/history/` (version history, owned by `vcs::SessionHistory`), `<base>/<agent_id>/links.json` (link graph).
- `DiskSessionStore::new(base)` plus path builders `base`, `agent_dir`, `sessions_dir`, `sessions_index_path`, `session_dir`, `state_path`, `history_dir`, `links_path`, and `history() -> SessionHistory`.
- `DiskSessionStore` methods: `save(agent_id, scope, record) -> SessionMeta`, `save_snapshot`, `save_from_session`, `load`, `load_snapshot`, `load_into_session` (one `Session::restore`), `exists`, `delete`, `list(agent_id, Option<&SessionScope>)`, `list_all`, `list_agents`, `read_index`, `write_index`.
- `save` semantics: a new session starts at version 1, becomes the scope's active session and deactivates its siblings; an existing record keeps `created_at`, takes the new `title`/`metadata`/`data_container_type`, and bumps `version`; `updated_at` is strictly monotonic per agent, so newest-first listings are deterministic; an empty snapshot `session_id` is filled from the metadata.
- `delete` removes the session directory, its index entry (dropping a scope that becomes empty) and its link entry, returning `false` when the directory is absent.
- Write semantics: every document is pretty JSON written to a sibling `.<name>.<uuid>.tmp` then renamed, so readers never observe a partial file and a failed write leaves no temp file; corrupt documents are `Error::Session`; agent and session ids go through `checkpoint::validate_id` (`Error::Validation`).
- Clones share the base path, the history commit lock and the link-graph write lock; two independently constructed stores over the same base do **not** share them, so writers must use store clones.
- `SessionController::new(store, agent_id)` — the per-agent cache in front of the store (`Arc<parking_lot::Mutex<BTreeMap<String, SessionRecord>>>`): `create(scope, session_id)` (empty `SessionType::Agent` snapshot), `upsert`/`flush` (persist in the scope the index already records), `upsert_in`/`save_in` (explicit scope, needed for a session the index does not know; `save` forwards to `save_in`), `get` (cache first, populated on a store read), `delete`, `list` (a scope-less listing merges cache-only records), plus `agent_id` and `store`.
- Scope resolution is strict: `scope_of` returns `Ok(None)` for a session the index does not know but an unparseable scope key is `Error::Session` naming the key (never a warn-and-skip), and `require_scope` turns the unknown case into `Error::Validation` ("save it with an explicit scope first") for `upsert`/`flush`.
- Link-graph wrappers: `read_links`/`write_links` (a missing document is an empty graph; writes are atomic and serialized by the store's link lock), `link_parent(agent_id, child, parent)`, `link_fork(agent_id, child, origin)`, `parent_of`, `children_of`, `ancestors`, `descendants`, `chain` (each takes `agent_id` first).

### `links` — the session link graph (`src/session_controller/links.rs`)

- The file name is the crate-private constant `LINKS_FILE = "links.json"`; the persisted document is `LinksIndex { agent_id, updated_at /* epoch seconds */, links }`; `SessionLinks` is serde-`transparent` over `BTreeMap<String, SessionLink>`.
- `LinkKind { Parent /* "parent" */, Fork /* "fork" */ }` with `as_str`/`Display`; `SessionLink { session_id, parent, forked_from, children, created_at /* RFC 3339 */ }` with `new(session_id, kind, origin)`, `origin()` (prefers `parent`) and `kind()` (`None` for a root or placeholder).
- `SessionLinks` API: `new`, `from_map`, `as_map`, `into_map`, `iter`, `len`, `is_empty`, `get`, `contains`, `link`, `record_parent`, `record_fork`, `parent_of`, `children_of`, `forks_of`, `remove`, `roots`, `leaves`, `chain`, `ancestors`, `descendants`, `depth`, `check_acyclic`.
- A node carries at most one lineage pointer; children are recorded on the parent entry in insertion order and relinking moves the child between parents; forks are discovered by scanning `forked_from`, so they are downstream of and never children of their origin.
- Traversal is deterministic and cycle-safe: `chain` returns the inclusive root→session path, `ancestors` drops the session, `descendants` is breadth-first (children before forks, forks included), `depth` is 0 for a root and for an unknown session, `roots`/`leaves` are session-id ordered, and an unknown session walks to empty results.
- Errors: a self-link or a link that would make a session its own ancestor is `Error::Validation`; a cycle already stored on disk is `Error::Session` naming the node; malformed ids are `Error::Validation`.
- `remove` drops one entry: the parent stops listing it as a child and its children/forks lose the pointer (they become roots); every other entry survives.
- Deferred from Python's controller: data-container and permission policies, the process-global `GlobalSessionController` singleton, and the `ChainSession` fan-out (see below).

## Constraints

- Offline and deterministic: `tokio::fs` only, atomic temp-file-plus-rename writes, no blocking I/O on async tasks, no network.
- Errors: `Error::Validation` for bad ids/scopes/user input, `Error::Session` for I/O, corrupt documents, and a corrupt index key; never panic on user input.
- Every public type is serde-serializable, `snake_case`, tolerant of unknown fields, with `#[serde(default)]` unless noted.
- `Session` semantics (state merge, component nesting, checkpoint snapshot/restore) come from `../CONTEXT.md` and are not re-derived here; this module only persists `SessionSnapshot`.
- Files ≤ ~1000 lines (`store.rs` is at that boundary; split by concern before adding another document type); `tracing` only, no stdout logging, no `unsafe`.

## Design Decisions

- **The `data` document is the crate's `SessionSnapshot`.** Python's data container is opaque; persisting the snapshot keeps `state.data` loadable through `Session::restore`, while the `{"meta": …, "data": …}` envelope preserves Python's metadata and index shape.
- **One index document per agent, one record file per session.** `sessions.json` carries scope collections and the active-session marker, so listings and newest-first ordering need no directory scan; `list_agents` scans for `sessions/sessions.json` instead.
- **Scope resolution never guesses.** A session whose scope the index does not record is `Error::Validation` (the caller uses `upsert_in`/`save_in`), and an unparseable index key is `Error::Session`, so a corrupt index can never silently move a session into `main()`.
- **The link graph is metadata, not a `ChainSession`.** Python's `ChainSession` pushes writes to every downstream session; here a session is an `Arc<dyn Session>` the caller owns, state writes go through it, and run loops decide what a fork inherits. Duplicating that fan-out would need a second shadow store, so the port keeps the graph as JSON and leaves propagation to the callers of `SessionLinks::chain`.
- **Lineage is one pointer per node.** `parent` vs `forked_from` keeps the graph a forest, so every traversal can be a bounded walk and a cycle is detectable (`check_acyclic`) rather than a possible hang.

## Routing Table

- `./scope.rs` → `Scope`, `Subject`, `SessionScope`, `SessionScopeKey` and their string forms
- `./schema.rs` → `SessionMeta`, `ScopeSessionsMeta`, `SessionsIndex`, `SessionRecord`, `DEFAULT_DATA_CONTAINER_TYPE`, the monotonic clock helper
- `./store.rs` → `DiskSessionStore` (layout, atomic writes, index/record operations, link wrappers) and `SessionController` (cache, strict scope resolution)
- `./links.rs` → `LinkKind`, `SessionLink`, `SessionLinks`, `LinksIndex`, the `links.json` document helpers
- `./mod.rs` → module wiring and re-exports (`session_controller::{…}`)
