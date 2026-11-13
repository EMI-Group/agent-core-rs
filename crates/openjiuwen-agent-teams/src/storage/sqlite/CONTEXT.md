# openjiuwen-agent-teams/src/storage/sqlite

## Intent

The persistent `TeamStore` backend (Cargo feature `sqlite`): the same rows, ordering, compare-and-set transitions, stamps and mailbox semantics as `InMemoryTeamStore`, stored in a SQLite database through `sqlx`.
Python parity source: `openjiuwen/agent_teams/tools/database/` (the `SQLAlchemy` DAOs `team_dao`, `member_dao`, `task_dao`, `message_dao` and the SQLite `TeamDatabase` they run on).
The backend is a state seam only: dependency unblocking, review gating, dispatch and message routing stay in `../../taskboard/`, `../../member/` and `../../runtime/`.
`../../memory.rs` is the behavioural spec; a change here that alters an observable answer is a parity bug.

## API Surface

```rust
pub const SQLITE_STORAGE_KIND: &str = "sqlite";

pub struct SqliteTeamStore { /* private Target(Memory|Path(PathBuf)|Url(String)) + OnceCell<SqlitePool> */ }

impl SqliteTeamStore {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self>;   // file; creates the parent directory + schema
    pub async fn in_memory() -> Result<Self>;                    // single-connection private `:memory:` database
    pub(crate) fn lazy_from_params(params: &BTreeMap<String, Value>) -> Result<Self>;  // the `open_store` seam
}

impl TeamStore for SqliteTeamStore { /* the whole trait; `name()` is `SQLITE_STORAGE_KIND` */ }
```

`StorageSpec::params` keys, in precedence order: `path`, `url`, `database`.
A non-string value is `Error::validation("storage param '<key>' must be a string")`; no key at all is `Error::config("storage backend 'sqlite' requires a 'path' or 'url' parameter")`; an empty value is `Error::validation("sqlite storage requires a non-empty 'path' or 'url'")`.
Accepted forms: `data/team.db`, `sqlite://data/team.db` (relative), `sqlite:///abs/team.db`, `:memory:` (any value containing it).

## Constraints

- Only this directory is edited; `../mod.rs`, `../records.rs`, `../memory.rs`, `../stamps.rs` and `../../schema/` belong to sibling nodes.
- The `sqlite` feature gates the whole module: `--no-default-features` must still build, and nothing here may be referenced without `#[cfg(feature = "sqlite")]`.
- `openjiuwen_common::error::{Error, Result}` only; every `sqlx::Error` goes through `rows::sqlite_error`; no `unwrap`/`expect` outside tests.
- No blocking calls: all I/O is `sqlx` async plus `tokio::fs::create_dir_all`; no lock or transaction is held across an unrelated `.await`.
- Enum columns are written and read through `serde` (`rows::encode_enum` / `rows::decode_enum`), never a second hand-written wire table.
- `tracing` only; one `debug!` when a pool opens. No `unwrap` on user input; conversion helpers saturate instead of panicking.
- Every public item carries a doc comment and a `# Errors` section where it is fallible.

## Behaviour

- Connect: file pools use `create_if_missing`, WAL, `synchronous=NORMAL`, a 5 s busy timeout, `foreign_keys=ON` and at most 4 connections; the in-memory pool is pinned to one connection (`:memory:` is private to its connection) with no idle/lifetime timeout.
- A file database therefore materializes `team.db` plus the WAL sidecars `team.db-wal`/`team.db-shm` in its directory; a tempdir test must expect those, while an in-memory store creates no file at all.
- Lazy: `open_store` is synchronous, so the pool lives in a `tokio::sync::OnceCell` and every trait method starts with `self.pool().await?`; `open`/`in_memory` connect eagerly so construction errors surface immediately.
- Schema: seven tables (`team_info`, `team_member`, `team_task`, `team_task_dependency`, `team_review_vote`, `team_message`, `team_read_watermark`) plus seven indexes, all `CREATE ... IF NOT EXISTS`; `PRAGMA user_version` is `1` (`0` is stamped, `> 1` is `Error::unsupported`).
- No foreign keys are declared on purpose: the trait allows a member, task, dependency or message row before its team row.
- Write paths mirror `InMemoryTeamStore` exactly: `upsert_team` keeps the stored `created` and stamps through `stamps::stamp`; `insert_member`/`insert_task` stamp against the previous `updated_at`; `update_member`/`update_task` apply the patch in Rust and rewrite the whole row; `transition_*` is a single conditional `UPDATE ... AND status = ?` with `updated_at = MAX(?, updated_at + 1)`; `insert_message`/`insert_vote` are `ON CONFLICT` upserts.
- `update_task` delegates the patch to `rows::apply_task_patch`, which owns the revision bump and the `r{n}: …` change note.
- `list_tasks` builds one statement with optional `status`/`assignee`/`parent_task_id` filters and `LIMIT`, ordered `updated_at DESC, task_id ASC`.
- Mailbox: direct rows use `is_read`; broadcasts use the per-member watermark (bound twice in `(? IS NULL OR timestamp > ?)`), and a sender never sees its own broadcast as unread. `mark_read` flags direct rows only; `set_read_watermark` never regresses (`MAX(...)`).

## Routing Table

- `./mod.rs` → `SQLITE_STORAGE_KIND`, `SqliteTeamStore`, connect/pool, the `TeamStore` impl
- `./statements.rs` → the column lists and named SQL statements the impl runs
- `./schema.rs` → DDL/index statements, `SCHEMA_VERSION`, `ensure_schema` migration step
- `./rows.rs` → column getters, record decoders, enum/JSON codecs, `to_i64`, `sqlite_error`, `apply_task_patch`
- `./tests/` → the SQLite suite (`#[cfg(test)]`): `mod.rs` (module map), `support.rs` (row/spec fixtures), `contract.rs` (backend-independent sequence), `parity.rs` (contract over both backends), `persistence.rs` (reopen), `races.rs` (concurrent CAS), `migration.rs` (schema guard), `open_store.rs` (`StorageSpec` dispatch)

## Test Strategy

The suite (`cargo test -p openjiuwen-agent-teams --features sqlite --lib`) is 18 tests, offline and deterministic: file databases live in `tempfile::tempdir()`, nothing sleeps, nothing is `#[ignore]`d, and stamps are caller-supplied or only asserted to have strictly grown.

- Parity: `contract.rs` runs the same representative sequence as the in-memory suite over the `dyn TeamStore` seam (team rows and stamps, roster patches and CAS, task filters/ordering/revisions/transitions, dependency edges, per-round votes, mailbox and watermark) and `parity.rs` runs it twice — once over `InMemoryTeamStore`, once over `SqliteTeamStore::in_memory()` — so any observable divergence fails one of the two.
- Reopen: `persistence.rs` seeds a team, two members, three tasks with a dependency edge, votes in two rounds, two messages and a watermark into `dir/team.db`, drops the store, reopens with `SqliteTeamStore::open` and asserts every row, the `updated_at` stamps, the listing order, the filters, the watermark and that the reopened database still accepts a transition.
- Races: `races.rs` opens one file database into an `Arc`, spawns two writers on a multi-threaded runtime and asserts that `transition_task(pending → in_progress)` and `transition_member(unstarted → starting)` each produce exactly one `true`, one `false`, the expected stored status and a `false` replay.
- Migration: `migration.rs` opens the same file twice (idempotent DDL), bumps `PRAGMA user_version` to `99` through a raw `sqlx` connection and asserts `SqliteTeamStore::open` rejects it with `ErrorKind::Unsupported` and a message containing "newer than this build", and verifies that `open` creates a nested missing parent directory.
- Dispatch: `open_store.rs` covers `memory` vs `sqlite`, the `path`/`url`/`database` keys and their precedence, `:memory:` forms, the malformed-parameter errors (`Config` / `Validation`), the deferred `postgresql`/`mysql`/unknown kinds, laziness (no file exists until the first async call) and independence of two `in_memory()` stores.

## Notes for Agents

- Keep `../../memory.rs` and this backend in lockstep: read a semantics change there, apply it here (and vice versa), including sorting, stamping and "silent no-op" rules; `./tests/contract.rs` is the shared proof.
- SQLite has no unsigned integers: bind through `rows::to_i64` and read back with the saturating getters.
- `desc` is a SQL keyword, so the column is `description` while the record field stays `desc`.
- `open_store` is synchronous, so the pool is lazy in a `tokio::sync::OnceCell` and a connect error surfaces on the first trait call; `open`/`in_memory` connect eagerly for callers that want it at construction.
