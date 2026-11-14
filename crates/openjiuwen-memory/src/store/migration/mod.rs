//! Versioned memory migrations: the SQLite schema runner and the data-migrator suite.
//!
//! Ports `openjiuwen.core.memory.migration` to two cooperating layers.
//!
//! # Layer 1 — the SQLite schema runner ([`registry`], [`runner`])
//!
//! Python keeps a KV marker (`MEMORY_MIGRATION_KV_SCHEMA_VERSION`, a numeric string) for
//! the KV store and one `memory_meta` row per SQL table; here one row per applied
//! migration lives in a single version table, which is the same information in one place.
//!
//! - [`registry`] — the versioned steps, their validation, and the builtin plan.
//! - [`runner`] — the forward-only `SQLite` runner, its report/status types, and
//!   `migrate_sqlite` (feature `sqlite`).
//!
//! Forward-only: there are no down migrations, a schema version only ever grows, and a
//! database recording a version above the registry's latest is left alone (the runner's
//! status logs a warning and reports no pending work instead of failing).
//!
//! The version table is infrastructure, not a migration; it is created idempotently before
//! every status/run:
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS memory_schema_version (
//!     version    INTEGER PRIMARY KEY,
//!     name       TEXT NOT NULL,
//!     applied_at TEXT NOT NULL   -- RFC 3339
//! )
//! ```
//!
//! Each migration runs in its own transaction, so its DDL and its version row commit
//! together and a failing statement records nothing. The one documented exception is a
//! `PRAGMA`, which SQLite cannot roll back: connection pragmas belong in connection setup
//! (the `SqliteConnectOptions`), never in a migration step. Versions start at `1`, are
//! contiguous and strictly ascending, so a version number is also a reservation for the
//! next one; the builtin registry ships v1 (`kv_store`) and v2 (`documents`), and modules
//! that add real tables append v3, v4, … through `MigrationRegistry::register` /
//! `MigrationRegistry::with`. Placeholder or no-op steps are never shipped, because a
//! no-op still consumes a version number forever.
//!
//! The registry half compiles without the `sqlite` feature; only the runner needs `sqlx`.
//!
//! # Layer 2 — the data-migrator suite ([`plan`], [`operation`], the families, [`standard`])
//!
//! Python's per-backend migrator objects (its `sql_registry`, `kv_registry`, … operation
//! objects) are replaced by four migrator families that convert *data* between layouts:
//!
//! - [`plan`] — the migration context, steps, serialisable plan, the `Migrator` trait, and
//!   the planner that chains migrators in registration order.
//! - [`operation`] — the executable operations, their registry, the operation journal that
//!   makes a run resumable, and the run/report types.
//! - [`sql`] — `SqlMigrator`: legacy variable rows into the current prefixes plus the
//!   `scope_user_mapping` backfill.
//! - [`message`] — `MessageMigrator`: legacy message rows into the `messages` document
//!   collection.
//! - [`vector`] — `VectorMigrator`: legacy embedding collections recollected into the
//!   canonical `uid_{user}_gid_{scope}_mtype_{kind}` names.
//! - [`index_version`] — `IndexVersionMigrator`: per-collection index stamps (and the
//!   opt-in purge of stale embeddings).
//! - [`standard`] — the standard set in execution order, with planner, registry, plan, and
//!   run entry points.
//!
//! Planning is read-only and produces a durable [`MigrationPlan`] (JSON); execution
//! resolves each step id through an [`OperationRegistry`] and journals every attempt, so an
//! interrupted run resumes from the same [`MigrationJournal`] and a dry run records
//! `Skipped` entries without invoking a handler. A failing operation is recorded and then
//! propagated; deferred capabilities are typed
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) rather than silent
//! no-ops.
//!
//! The families are exported explicitly rather than through globs: each module defines its
//! own `MIGRATOR_ID`/`MIGRATOR_VERSION` and operation ids, and a glob would make those
//! names ambiguous.
//!
//! The two layers are independent: the schema runner creates the tables the backends open,
//! and the migrator suite converts rows inside them. A crate consumer normally runs
//! `migrate_sqlite` first (feature `sqlite`) and then
//! [`run_migrations`] over the opened stores.

pub mod index_version;
pub mod message;
pub mod operation;
pub mod plan;
pub mod registry;
pub mod runner;
pub mod sql;
pub mod standard;
pub mod vector;

pub use index_version::{
    IndexStamp, IndexVersionMigrator, IndexVersionOptions, CURRENT_INDEX_VERSION,
    INDEX_VERSION_PREFIX,
};
pub use message::{LegacyMessageRow, MessageMigrator, LEGACY_MESSAGE_PREFIXES};
pub use operation::*;
pub use plan::*;
pub use registry::*;
#[cfg(feature = "sqlite")]
pub use runner::*;
pub use sql::SqlMigrator;
pub use standard::{
    plan_migrations, run_migrations, run_migrations_with, standard_migrators, standard_planner,
    standard_registry,
};
pub use vector::{RecollectionPair, VectorMigrator};
