//! Versioned schema migrations for the memory SQLite backend.
//!
//! Ports the *version bookkeeping* of the Python package
//! `openjiuwen.core.memory.migration` to a small, forward-only migration runner:
//! Python keeps a KV marker (`MEMORY_MIGRATION_KV_SCHEMA_VERSION`, a numeric string)
//! for the KV store and one `memory_meta` row per SQL table, whereas this module
//! stores one row per applied migration in a single version table, which is the same
//! information in one place. The Python operation registries (`sql_registry`,
//! `kv_registry`, …) and their `AddColumn`/`RenameField`/`UpdateKV` operation objects
//! are not ported: this backend only creates tables, so a migration is a list of DDL
//! statements.
//!
//! - [`registry`] — the versioned steps, their validation, and the builtin plan.
//! - [`runner`] — the forward-only `SQLite` runner, its report/status types, and
//!   `migrate_sqlite` (feature `sqlite`).
//! - [`plan`] — the migration context, steps, serialisable plan, the `Migrator` trait, and
//!   the planner that chains migrators in registration order.
//! - [`operation`] — the executable operations, their registry, the operation journal that
//!   makes a run resumable, and the run/report types.
//!
//! # Semantics
//!
//! - **Forward-only.** There are no down migrations: a schema version only ever grows.
//!   A database that records a version above the registry's latest is left alone (for
//!   example after a downgrade of the library); the runner's status logs a warning and
//!   reports no pending work instead of failing.
//! - **Version table.** Infrastructure, not a migration. It is created idempotently
//!   before every status/run:
//!
//!   ```sql
//!   CREATE TABLE IF NOT EXISTS memory_schema_version (
//!       version    INTEGER PRIMARY KEY,
//!       name       TEXT NOT NULL,
//!       applied_at TEXT NOT NULL   -- RFC 3339
//!   )
//!   ```
//!
//! - **Atomic per migration.** Each migration runs inside its own transaction, so its
//!   DDL and its version row commit together; a failing statement rolls the whole
//!   migration back and records nothing. The one documented exception: a `PRAGMA`
//!   (for example `journal_mode`) cannot run inside a transaction and takes effect
//!   immediately, so connection pragmas belong in connection setup
//!   (the `SqliteConnectOptions`), never in a migration step.
//! - **Contiguous versions.** Versions start at `1` and are contiguous and strictly
//!   ascending, so a version number is also a "reservation": the next free number is
//!   simply the latest + 1. The builtin registry ships v1 (`kv_store`) and v2
//!   (`documents`); modules that add real tables later (for example dreaming and team
//!   memory) append v3, v4, … through `MigrationRegistry::register` /
//!   `MigrationRegistry::with`. Placeholder or no-op steps are never shipped, because a
//!   no-op still consumes a version number forever.
//! - **Registration order is execution order.** The registry is validated on every
//!   insertion, so it is always sorted ascending and gap-free.
//!
//! The registry half compiles without the `sqlite` feature; only the runner
//! (`MigrationRunner`, `MigrationStatus`, `migrate_sqlite`) needs `sqlx`.

pub mod index_version;
pub mod message;
pub mod operation;
pub mod plan;
pub mod registry;
pub mod runner;
pub mod sql;
pub mod standard;
pub mod vector;

pub use operation::*;
pub use plan::*;
pub use registry::*;
#[cfg(feature = "sqlite")]
pub use runner::*;
