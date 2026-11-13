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

## Notes for Agents

- Keep `../../memory.rs` and this backend in lockstep: read a semantics change there, apply it here (and vice versa), including sorting, stamping and "silent no-op" rules.
- SQLite has no unsigned integers: bind through `rows::to_i64` and read back with the saturating getters.
- `desc` is a SQL keyword, so the column is `description` while the record field stays `desc`.
- A missing sqlite test suite is a known gap of this phase, not a licence to change behaviour untested: assert against `InMemoryTeamStore` when adding one.
