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
//! # Semantics
//!
//! - **Forward-only.** There are no down migrations: a schema version only ever grows.
//!   A database that records a version above the registry's latest is left alone (for
//!   example after a downgrade of the library); `MigrationRunner::status` logs a
//!   warning and reports no pending work instead of failing.
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
//!   memory) append v3, v4, … through [`MigrationRegistry::register`] /
//!   [`MigrationRegistry::with`]. Placeholder or no-op steps are never shipped,
//!   because a no-op still consumes a version number forever.
//! - **Registration order is execution order.** The registry is validated on every
//!   insertion, so it is always sorted ascending and gap-free.
//!
//! The registry types compile without the `sqlite` feature; only the runner
//! (`MigrationRunner`, `MigrationStatus`, `migrate_sqlite`) needs `sqlx`.

use openjiuwen_common::error::{Error, Result};

#[cfg(feature = "sqlite")]
use std::path::Path;
#[cfg(feature = "sqlite")]
use std::str::FromStr;

#[cfg(feature = "sqlite")]
use chrono::Utc;
#[cfg(feature = "sqlite")]
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
#[cfg(feature = "sqlite")]
use sqlx::{Row, SqlitePool};

/// One versioned schema step: a name, a description, and the DDL it executes.
///
/// The SQL statements are static strings so a migration can be described in a `const`
/// and shipped inside the binary; they must each be a single statement, because they
/// are handed to `sqlx` one at a time, and they should be idempotent
/// (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`) so a database that
/// already contains the objects (for example one created by
/// `openjiuwen_foundation::store::SqliteKvStore`) migrates cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Schema version this step brings the database to; starts at `1`.
    pub version: u32,
    /// Stable short identifier, unique within a registry (for example `kv_store`).
    pub name: &'static str,
    /// Human-readable summary of the step, recorded nowhere but documentation/logs.
    pub description: &'static str,
    /// The DDL statements of the step, executed in order inside one transaction.
    pub up_sql: &'static [&'static str],
}

impl Migration {
    /// Builds a migration step; validation happens in [`MigrationRegistry`].
    #[must_use]
    pub const fn new(
        version: u32,
        name: &'static str,
        description: &'static str,
        up_sql: &'static [&'static str],
    ) -> Self {
        Self {
            version,
            name,
            description,
            up_sql,
        }
    }
}

/// An ordered, validated set of [`Migration`]s.
///
/// Every insertion revalidates the whole list, so the invariants below hold for any
/// value that exists: versions start at `1`, are contiguous and strictly ascending,
/// names are non-empty and unique, and no step is empty (no empty name, description,
/// or statement list). Rejections are `Error::Validation` naming the offending
/// version and name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MigrationRegistry {
    /// Validated migrations in execution order.
    migrations: Vec<Migration>,
}

impl MigrationRegistry {
    /// Builds a registry from `migrations`, validating them in the given order.
    ///
    /// An empty list is valid (an application with nothing to migrate); otherwise the
    /// first version must be `1`.
    pub fn new(migrations: Vec<Migration>) -> Result<Self> {
        validate(&migrations)?;
        Ok(Self { migrations })
    }

    /// Appends `migration`, which must be the registry's next version.
    ///
    /// The registry is left untouched when the migration is rejected, so a failed
    /// registration never corrupts an existing plan.
    pub fn register(&mut self, migration: Migration) -> Result<()> {
        self.migrations.push(migration);
        if let Err(error) = validate(&self.migrations) {
            self.migrations.pop();
            return Err(error);
        }
        Ok(())
    }

    /// Builder form of [`Self::register`]: appends and returns the registry.
    pub fn with(mut self, migration: Migration) -> Result<Self> {
        self.register(migration)?;
        Ok(self)
    }

    /// The migrations in execution order.
    #[must_use]
    pub fn migrations(&self) -> &[Migration] {
        &self.migrations
    }

    /// Number of registered migrations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.migrations.len()
    }

    /// Whether the registry holds no migration at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.migrations.is_empty()
    }

    /// The highest registered version, or `None` for an empty registry.
    #[must_use]
    pub fn latest_version(&self) -> Option<u32> {
        self.migrations.last().map(|migration| migration.version)
    }

    /// Looks up a migration by version.
    #[must_use]
    pub fn get(&self, version: u32) -> Option<&Migration> {
        // Versions are contiguous from 1, so the index is `version - 1`.
        version
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.migrations.get(index))
            .filter(|migration| migration.version == version)
    }
}

/// The builtin registry: v1 `kv_store` and v2 `documents`.
///
/// It is built directly from [`builtin_migrations`] instead of going through
/// [`MigrationRegistry::new`] so callers (notably `migrate_sqlite`) need no `Result`;
/// the list is validated by a unit test, and [`MigrationRegistry::register`] remains
/// the validated path for anything appended at runtime.
#[must_use]
pub fn default_registry() -> MigrationRegistry {
    MigrationRegistry {
        migrations: builtin_migrations(),
    }
}

/// The builtin migration steps, in version order.
///
/// The DDL mirrors `openjiuwen_foundation::store::{SqliteKvStore, SqliteDocumentStore}`
/// exactly (same columns, same indexes) but is `IF NOT EXISTS`, so running the runner
/// against a database that already has the tables — every database the foundation
/// created — is a no-op that only records the version rows.
#[must_use]
pub fn builtin_migrations() -> Vec<Migration> {
    vec![
        Migration::new(
            1,
            "kv_store",
            "key-value table for memory records, variables, and scope config",
            KV_STORE_UP,
        ),
        Migration::new(
            2,
            "documents",
            "collection registry and document table for message history",
            DOCUMENTS_UP,
        ),
    ]
}

/// DDL of the foundation `kv_store` table.
const KV_STORE_TABLE: &str = "
CREATE TABLE IF NOT EXISTS kv_store (
    k TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    kind TEXT NOT NULL,
    expiry_ms INTEGER
)";

/// DDL of the foundation `kv_store` TTL index.
const KV_STORE_EXPIRY_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS kv_store_expiry_idx ON kv_store (expiry_ms)";

/// The v1 statements, in execution order.
const KV_STORE_UP: &[&str] = &[KV_STORE_TABLE, KV_STORE_EXPIRY_INDEX];

/// DDL of the foundation `document_collections` table.
const DOCUMENT_COLLECTIONS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS document_collections (
    name TEXT PRIMARY KEY
)";

/// DDL of the foundation `documents` table.
const DOCUMENTS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS documents (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    PRIMARY KEY (collection, id)
)";

/// DDL of the foundation `documents` collection index.
const DOCUMENTS_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection)";

/// The v2 statements, in execution order.
const DOCUMENTS_UP: &[&str] = &[DOCUMENT_COLLECTIONS_TABLE, DOCUMENTS_TABLE, DOCUMENTS_INDEX];

/// Validates a candidate migration list against the registry invariants.
fn validate(migrations: &[Migration]) -> Result<()> {
    let mut previous: Option<&Migration> = None;
    for migration in migrations {
        if migration.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "memory migration v{} has an empty name",
                migration.version
            )));
        }
        if migration.description.trim().is_empty() {
            return Err(Error::validation(format!(
                "memory migration v{} (`{}`) has an empty description",
                migration.version, migration.name
            )));
        }
        if migration.up_sql.is_empty() {
            return Err(Error::validation(format!(
                "memory migration v{} (`{}`) has no statements",
                migration.version, migration.name
            )));
        }
        let expected = previous.map_or(1, |last| last.version.saturating_add(1));
        if migration.version != expected {
            let detail = if previous.is_some_and(|last| last.version == migration.version) {
                format!("duplicate version v{}", migration.version)
            } else {
                format!(
                    "expected v{expected} after v{}, found v{}",
                    previous.map_or(0, |last| last.version),
                    migration.version
                )
            };
            return Err(Error::validation(format!(
                "memory migration v{} (`{}`) breaks the contiguous ascending sequence: {detail}",
                migration.version, migration.name
            )));
        }
        if migrations
            .iter()
            .take_while(|earlier| earlier.version < migration.version)
            .any(|earlier| earlier.name == migration.name)
        {
            return Err(Error::validation(format!(
                "memory migration v{} (`{}`) reuses a migration name",
                migration.version, migration.name
            )));
        }
        previous = Some(migration);
    }
    Ok(())
}

/// Name of the table that records applied migrations.
///
/// One row per applied step: `version` (primary key), `name`, and `applied_at`
/// (RFC 3339). It is infrastructure: the runner creates it before doing anything else.
#[cfg(feature = "sqlite")]
pub const SCHEMA_VERSION_TABLE: &str = "memory_schema_version";

/// DDL of [`SCHEMA_VERSION_TABLE`], applied before any status/run.
#[cfg(feature = "sqlite")]
const SCHEMA_VERSION_DDL: &str = "
CREATE TABLE IF NOT EXISTS memory_schema_version (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
)";

/// Statement recording one applied migration.
#[cfg(feature = "sqlite")]
const RECORD_APPLIED: &str =
    "INSERT INTO memory_schema_version (version, name, applied_at) VALUES (?1, ?2, ?3)";

/// Statement listing applied migrations oldest first.
#[cfg(feature = "sqlite")]
const SELECT_APPLIED: &str =
    "SELECT version, name, applied_at FROM memory_schema_version ORDER BY version";

/// One row of the version table: a migration that this database has applied.
#[cfg(feature = "sqlite")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AppliedMigration {
    /// Schema version the database reached.
    #[serde(default)]
    pub version: u32,
    /// Name of the applied migration.
    #[serde(default)]
    pub name: String,
    /// RFC 3339 timestamp of the commit.
    #[serde(default)]
    pub applied_at: String,
}

/// The outcome of one [`MigrationRunner::run`].
#[cfg(feature = "sqlite")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MigrationReport {
    /// Version recorded before the run (`None` for a database with no rows).
    #[serde(default)]
    pub from_version: Option<u32>,
    /// Version recorded after the run; equals `from_version` when nothing was applied.
    #[serde(default)]
    pub to_version: Option<u32>,
    /// The migrations this run applied, in execution order.
    #[serde(default)]
    pub applied: Vec<AppliedMigration>,
}

#[cfg(feature = "sqlite")]
impl MigrationReport {
    /// Number of migrations this run applied.
    #[must_use]
    pub fn applied_count(&self) -> usize {
        self.applied.len()
    }

    /// Whether this run applied nothing (the database was already up to date).
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.applied.is_empty()
    }

    /// The versions this run applied, in execution order.
    pub fn applied_versions(&self) -> impl Iterator<Item = u32> + '_ {
        self.applied.iter().map(|migration| migration.version)
    }
}

/// The migration state of one database: where it is and what is pending.
#[cfg(feature = "sqlite")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MigrationStatus {
    /// Highest recorded version, or `None` for a database with no migration rows.
    #[serde(default)]
    pub current: Option<u32>,
    /// Registry versions above `current`, ascending.
    #[serde(default)]
    pub pending: Vec<u32>,
    /// Number of recorded version rows.
    #[serde(default)]
    pub applied: usize,
}

/// Applies a [`MigrationRegistry`] to one SQLite database, forward only.
///
/// The runner is stateless: it reads the version table on every call, so concurrent
/// or repeated calls observe the committed state rather than a cached one.
#[cfg(feature = "sqlite")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationRunner {
    /// The plan to apply.
    registry: MigrationRegistry,
}

#[cfg(feature = "sqlite")]
impl MigrationRunner {
    /// Builds a runner for `registry`.
    #[must_use]
    pub fn new(registry: MigrationRegistry) -> Self {
        Self { registry }
    }

    /// The plan this runner applies.
    #[must_use]
    pub fn registry(&self) -> &MigrationRegistry {
        &self.registry
    }

    /// Reports where the database is relative to this plan, without changing it.
    ///
    /// Creates the version table when absent (so a fresh database reports
    /// `current: None`). A recorded version above the plan's latest is a warning, not
    /// an error: downgrades never delete data or schema, and nothing is pending.
    pub async fn status(&self, pool: &SqlitePool) -> Result<MigrationStatus> {
        bootstrap(pool).await?;
        let applied = fetch_applied(pool).await?;
        warn_about_newer_database(&self.registry, &applied);
        let current = applied.iter().map(|migration| migration.version).max();
        let pending = self
            .registry
            .migrations()
            .iter()
            .filter(|migration| current.is_none_or(|current| migration.version > current))
            .map(|migration| migration.version)
            .collect();
        Ok(MigrationStatus {
            current,
            pending,
            applied: applied.len(),
        })
    }

    /// Applies every pending migration, oldest first.
    ///
    /// Each migration commits its DDL and its version row in one transaction, so the
    /// database is never observed at a half-applied version. The first failure stops
    /// the run: earlier migrations stay applied, the failing one leaves nothing behind,
    /// and the storage error is returned with the failing version in its context.
    pub async fn run(&self, pool: &SqlitePool) -> Result<MigrationReport> {
        bootstrap(pool).await?;
        let applied = fetch_applied(pool).await?;
        warn_about_newer_database(&self.registry, &applied);
        let from_version = applied.iter().map(|migration| migration.version).max();

        let mut just_applied = Vec::new();
        for migration in self.registry.migrations() {
            if from_version.is_some_and(|current| migration.version <= current) {
                continue;
            }
            apply(pool, migration).await?;
            tracing::info!(
                target: "openjiuwen_memory::store",
                version = migration.version,
                name = migration.name,
                "applied memory schema migration"
            );
            just_applied.push(AppliedMigration {
                version: migration.version,
                name: migration.name.to_string(),
                applied_at: Utc::now().to_rfc3339(),
            });
        }

        let to_version = just_applied
            .last()
            .map(|migration| migration.version)
            .or(from_version);
        Ok(MigrationReport {
            from_version,
            to_version,
            applied: just_applied,
        })
    }

    /// Whether the database records every migration of this plan.
    pub async fn is_up_to_date(&self, pool: &SqlitePool) -> Result<bool> {
        Ok(self.status(pool).await?.pending.is_empty())
    }
}

/// Migrates the SQLite database at `path` with the [`default_registry`].
///
/// This is the entry point for a caller that owns the database path (a CLI, a setup
/// step) but not a pool. It creates the parent directory when needed, opens its own
/// pool with `create_if_missing`, migrates, and closes the pool again.
///
/// It exists because [`crate::store::MemoryStores`] bundles `Arc<dyn KvStore>` and
/// `Arc<dyn DocumentStore>` without a downcast seam, so the SQLite URL cannot be
/// recovered from a bundle; a `run_on_stores(&MemoryStores)` variant would need a
/// downcast hook in `openjiuwen-foundation`, which is deferred.
#[cfg(feature = "sqlite")]
pub async fn migrate_sqlite(path: impl AsRef<Path>) -> Result<MigrationReport> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::external(error).with_context(format!(
                    "creating the memory SQLite directory `{}`",
                    parent.display()
                ))
            })?;
        }
    }
    let url = format!("sqlite:{}", path.display());
    let options = SqliteConnectOptions::from_str(&url)
        .map_err(Error::external)?
        .create_if_missing(true);
    let pool = SqlitePoolOptions::new()
        .connect_with(options)
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!("opening the memory database `{url}`"))
        })?;
    let report = MigrationRunner::new(default_registry()).run(&pool).await;
    pool.close().await;
    report
}

/// Creates the version table when absent.
#[cfg(feature = "sqlite")]
async fn bootstrap(pool: &SqlitePool) -> Result<()> {
    sqlx::query(SCHEMA_VERSION_DDL)
        .execute(pool)
        .await
        .map_err(Error::external)
        .map_err(|error| error.with_context(format!("bootstrapping `{SCHEMA_VERSION_TABLE}`")))?;
    Ok(())
}

/// Reads every recorded migration, oldest first.
#[cfg(feature = "sqlite")]
async fn fetch_applied(pool: &SqlitePool) -> Result<Vec<AppliedMigration>> {
    let rows = sqlx::query(SELECT_APPLIED)
        .fetch_all(pool)
        .await
        .map_err(Error::external)?;
    rows.iter().map(decode_applied).collect()
}

/// Decodes one version-table row.
#[cfg(feature = "sqlite")]
fn decode_applied(row: &sqlx::sqlite::SqliteRow) -> Result<AppliedMigration> {
    let version: i64 = row.try_get("version").map_err(Error::external)?;
    let version = u32::try_from(version).map_err(|_| {
        Error::validation(format!(
            "`{SCHEMA_VERSION_TABLE}` holds an invalid version `{version}`"
        ))
    })?;
    Ok(AppliedMigration {
        version,
        name: row.try_get("name").map_err(Error::external)?,
        applied_at: row.try_get("applied_at").map_err(Error::external)?,
    })
}

/// Logs when the database is ahead of the plan instead of downgrading it.
#[cfg(feature = "sqlite")]
fn warn_about_newer_database(registry: &MigrationRegistry, applied: &[AppliedMigration]) {
    let Some(current) = applied.iter().map(|migration| migration.version).max() else {
        return;
    };
    let Some(latest) = registry.latest_version() else {
        tracing::warn!(
            target: "openjiuwen_memory::store",
            current,
            "the memory database is versioned but this build ships no migrations; \
             leaving it untouched"
        );
        return;
    };
    if current > latest {
        tracing::warn!(
            target: "openjiuwen_memory::store",
            current,
            latest,
            "the memory database schema is newer than this build; \
             no downgrade is attempted and nothing is pending"
        );
    }
}

/// Applies one migration and records it, atomically.
#[cfg(feature = "sqlite")]
async fn apply(pool: &SqlitePool, migration: &Migration) -> Result<()> {
    let context = format!(
        "applying memory migration v{} (`{}`)",
        migration.version, migration.name
    );
    let mut transaction = pool
        .begin()
        .await
        .map_err(|error| Error::external(error).with_context(&context))?;
    for statement in migration.up_sql {
        sqlx::query(statement)
            .execute(&mut *transaction)
            .await
            .map_err(|error| Error::external(error).with_context(&context))?;
    }
    sqlx::query(RECORD_APPLIED)
        .bind(i64::from(migration.version))
        .bind(migration.name)
        .bind(Utc::now().to_rfc3339())
        .execute(&mut *transaction)
        .await
        .map_err(|error| Error::external(error).with_context(&context))?;
    transaction
        .commit()
        .await
        .map_err(|error| Error::external(error).with_context(&context))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid step with the given version and name.
    fn step(version: u32, name: &'static str) -> Migration {
        Migration::new(version, name, "test step", &["SELECT 1"])
    }

    /// The validation error message, asserting the error kind first.
    fn rejection(result: Result<MigrationRegistry>) -> String {
        let error = result.expect_err("registry must be rejected");
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        error.to_string()
    }

    #[test]
    fn duplicate_version_is_rejected() {
        let message = rejection(MigrationRegistry::new(vec![step(1, "one"), step(1, "two")]));
        assert!(message.contains("duplicate version v1"), "{message}");
        assert!(message.contains("`two`"), "{message}");
    }

    #[test]
    fn gap_is_rejected() {
        let message = rejection(MigrationRegistry::new(vec![
            step(1, "one"),
            step(3, "three"),
        ]));
        assert!(message.contains("found v3"), "{message}");
        assert!(message.contains("expected v2"), "{message}");
    }

    #[test]
    fn sequence_must_start_at_one() {
        let message = rejection(MigrationRegistry::new(vec![step(2, "two")]));
        assert!(message.contains("expected v1"), "{message}");
    }

    #[test]
    fn out_of_order_is_rejected() {
        // A version that goes backwards is neither a duplicate of the previous step
        // nor the expected next version, so the sequence check rejects it.
        let message = rejection(MigrationRegistry::new(vec![
            step(1, "one"),
            step(2, "two"),
            step(1, "again"),
        ]));
        assert!(message.contains("found v1"), "{message}");
        assert!(message.contains("expected v3"), "{message}");
    }

    #[test]
    fn empty_name_description_and_sql_are_rejected() {
        let empty_name = rejection(MigrationRegistry::new(vec![Migration::new(
            1,
            "  ",
            "desc",
            &["SELECT 1"],
        )]));
        assert!(empty_name.contains("empty name"), "{empty_name}");

        let empty_description = rejection(MigrationRegistry::new(vec![Migration::new(
            1,
            "one",
            "",
            &["SELECT 1"],
        )]));
        assert!(
            empty_description.contains("empty description"),
            "{empty_description}"
        );

        let empty_sql = rejection(MigrationRegistry::new(vec![Migration::new(
            1,
            "one",
            "desc",
            &[],
        )]));
        assert!(empty_sql.contains("no statements"), "{empty_sql}");
    }

    #[test]
    fn duplicate_name_is_rejected() {
        let message = rejection(MigrationRegistry::new(vec![
            step(1, "same"),
            step(2, "same"),
        ]));
        assert!(message.contains("reuses a migration name"), "{message}");
    }

    #[test]
    fn empty_registry_is_valid() {
        let registry = MigrationRegistry::new(Vec::new()).expect("empty registry");
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.latest_version(), None);
        assert_eq!(registry.get(1), None);
    }

    #[test]
    fn with_appends_the_next_version_and_rejects_a_duplicate() {
        let registry = MigrationRegistry::new(vec![step(1, "one")])
            .expect("first")
            .with(step(2, "two"))
            .expect("second");
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.latest_version(), Some(2));
        assert_eq!(registry.get(2).map(|migration| migration.name), Some("two"));
        assert_eq!(registry.get(3), None);

        let error = registry
            .clone()
            .with(step(2, "again"))
            .expect_err("duplicate rejected");
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert_eq!(registry.len(), 2, "a rejected append never mutates");
    }

    #[test]
    fn register_rejects_and_leaves_the_registry_unchanged() {
        let mut registry = MigrationRegistry::new(vec![step(1, "one")]).expect("first");
        assert!(registry.register(step(3, "three")).is_err());
        assert_eq!(registry.migrations().len(), 1);
        assert_eq!(registry.latest_version(), Some(1));
        registry
            .register(step(2, "two"))
            .expect("contiguous append");
        assert_eq!(registry.latest_version(), Some(2));
    }

    #[test]
    fn default_registry_ships_kv_and_documents() {
        let registry = default_registry();
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.latest_version(), Some(2));
        let names: Vec<&str> = registry
            .migrations()
            .iter()
            .map(|migration| migration.name)
            .collect();
        assert_eq!(names, vec!["kv_store", "documents"]);
        assert_eq!(registry.migrations()[0].version, 1);
        assert_eq!(registry.migrations()[1].version, 2);
        // The builtin list is what `new` accepts, so the unchecked constructor is safe.
        MigrationRegistry::new(builtin_migrations()).expect("builtin migrations are valid");
        for migration in registry.migrations() {
            assert_ne!(migration.description, "");
            assert_ne!(migration.up_sql.len(), 0);
            for statement in migration.up_sql {
                assert_ne!(statement.trim(), "");
            }
        }
        assert!(registry.migrations()[0]
            .up_sql
            .iter()
            .any(|statement| statement.contains("kv_store")));
        assert!(registry.migrations()[1]
            .up_sql
            .iter()
            .any(|statement| statement.contains("documents")));
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod sqlite_tests {
    use super::*;
    use openjiuwen_foundation::store::{KvStore, KvValue, SqliteDocumentStore, SqliteKvStore};

    /// Opens a pool over `path`, creating the file when absent.
    async fn pool_at(path: &Path) -> SqlitePool {
        let url = format!("sqlite:{}", path.display());
        let options = SqliteConnectOptions::from_str(&url)
            .expect("valid url")
            .create_if_missing(true);
        SqlitePoolOptions::new()
            .connect_with(options)
            .await
            .expect("pool")
    }

    /// A pool over `<dir>/memory.db`.
    async fn pool(dir: &tempfile::TempDir) -> SqlitePool {
        pool_at(&dir.path().join("memory.db")).await
    }

    /// The `sqlite_master` count of `table`.
    async fn table_exists(pool: &SqlitePool, table: &str) -> bool {
        let count: i64 = sqlx::query(
            "SELECT COUNT(*) AS n FROM sqlite_master WHERE type = 'table' AND name = ?1",
        )
        .bind(table)
        .fetch_one(pool)
        .await
        .expect("sqlite_master")
        .try_get("n")
        .expect("n");
        count > 0
    }

    #[tokio::test]
    async fn fresh_database_migrates_to_latest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pool = pool(&dir).await;
        let runner = MigrationRunner::new(default_registry());

        let status = runner.status(&pool).await.expect("status");
        assert_eq!(status.current, None);
        assert_eq!(status.pending, vec![1, 2]);
        assert_eq!(status.applied, 0);

        let report = runner.run(&pool).await.expect("run");
        assert_eq!(report.applied_count(), 2);
        assert_eq!(report.applied_versions().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(report.from_version, None);
        assert_eq!(report.to_version, Some(2));
        assert!(!report.is_noop());

        let status = runner.status(&pool).await.expect("status");
        assert_eq!(status.current, Some(2));
        assert_eq!(status.pending, Vec::<u32>::new());
        assert_eq!(status.applied, 2);
        assert!(runner.is_up_to_date(&pool).await.expect("up to date"));

        let rows = fetch_applied(&pool).await.expect("rows");
        assert_eq!(
            rows.iter().map(|row| row.version).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(rows[0].name, "kv_store");
        assert_eq!(rows[1].name, "documents");
        for row in &rows {
            let parsed = chrono::DateTime::parse_from_rfc3339(&row.applied_at)
                .expect("applied_at is RFC 3339");
            assert!(parsed.timestamp() > 0);
        }
        assert!(table_exists(&pool, "kv_store").await);
        assert!(table_exists(&pool, "documents").await);
        assert!(table_exists(&pool, "document_collections").await);
        assert!(table_exists(&pool, SCHEMA_VERSION_TABLE).await);
        pool.close().await;
    }

    #[tokio::test]
    async fn second_run_applies_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pool = pool(&dir).await;
        let runner = MigrationRunner::new(default_registry());
        runner.run(&pool).await.expect("first run");

        let report = runner.run(&pool).await.expect("second run");
        assert_eq!(report.applied_count(), 0);
        assert!(report.is_noop());
        assert!(report.applied_versions().next().is_none());
        assert_eq!(report.from_version, Some(2));
        assert_eq!(report.to_version, Some(2));

        let count: i64 = sqlx::query("SELECT COUNT(*) AS n FROM memory_schema_version")
            .fetch_one(&pool)
            .await
            .expect("count")
            .try_get("n")
            .expect("n");
        assert_eq!(count, 2);
        pool.close().await;
    }

    #[tokio::test]
    async fn a_failing_migration_rolls_back_completely() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pool = pool(&dir).await;
        let kv = builtin_migrations().remove(0);
        let broken = Migration::new(
            2,
            "broken",
            "a step whose second statement is invalid",
            &["CREATE TABLE rolled_back (id INTEGER)", "THIS IS NOT SQL"],
        );
        let runner = MigrationRunner::new(MigrationRegistry::new(vec![kv, broken]).expect("plan"));

        let error = runner.run(&pool).await.expect_err("run must fail");
        let message = error.to_string();
        assert!(message.contains("v2"), "{message}");
        assert!(message.contains("`broken`"), "{message}");

        let rows = fetch_applied(&pool).await.expect("rows");
        assert_eq!(
            rows.iter().map(|row| row.version).collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(runner.status(&pool).await.expect("status").current, Some(1));
        assert!(!table_exists(&pool, "rolled_back").await);
        pool.close().await;
    }

    #[tokio::test]
    async fn pre_existing_populated_database_keeps_its_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("memory.db");
        let url = format!("sqlite:{}", path.display());
        let kv = SqliteKvStore::connect(&url).await.expect("kv store");
        let documents = SqliteDocumentStore::connect(&url)
            .await
            .expect("document store");
        kv.set("legacy", KvValue::text("survives"))
            .await
            .expect("write");

        let pool = pool_at(&path).await;
        let report = MigrationRunner::new(default_registry())
            .run(&pool)
            .await
            .expect("run");
        assert_eq!(report.applied_count(), 2);
        assert_eq!(report.to_version, Some(2));

        assert_eq!(
            kv.get("legacy").await.expect("read"),
            Some(KvValue::text("survives"))
        );
        assert!(table_exists(&pool, "documents").await);
        let _ = documents.pool();
        pool.close().await;
    }

    #[tokio::test]
    async fn migrate_sqlite_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("memory.db");

        let first = migrate_sqlite(&path).await.expect("first migration");
        assert_eq!(first.applied_count(), 2);
        assert_eq!(first.from_version, None);
        assert_eq!(first.to_version, Some(2));

        let second = migrate_sqlite(&path).await.expect("second migration");
        assert_eq!(second.applied_count(), 0);
        assert!(second.is_noop());
        assert_eq!(second.from_version, Some(2));

        let pool = pool_at(&path).await;
        assert!(MigrationRunner::new(default_registry())
            .is_up_to_date(&pool)
            .await
            .expect("up to date"));
        pool.close().await;
    }

    #[tokio::test]
    async fn a_newer_database_is_left_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pool = pool(&dir).await;
        let runner = MigrationRunner::new(default_registry());
        runner.run(&pool).await.expect("baseline run");
        sqlx::query(
            "INSERT INTO memory_schema_version (version, name, applied_at) VALUES (?1, ?2, ?3)",
        )
        .bind(99_i64)
        .bind("future")
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .expect("future row");

        let status = runner.status(&pool).await.expect("status");
        assert_eq!(status.current, Some(99));
        assert_eq!(status.pending, Vec::<u32>::new());
        assert_eq!(status.applied, 3);

        let report = runner.run(&pool).await.expect("run");
        assert_eq!(report.applied_count(), 0);
        assert_eq!(report.from_version, Some(99));
        assert_eq!(report.to_version, Some(99));
        assert!(runner.is_up_to_date(&pool).await.expect("up to date"));
        pool.close().await;
    }

    #[tokio::test]
    async fn reports_round_trip_through_json() {
        let report = MigrationReport {
            from_version: None,
            to_version: Some(2),
            applied: vec![AppliedMigration {
                version: 1,
                name: "kv_store".to_string(),
                applied_at: "2026-01-01T00:00:00+00:00".to_string(),
            }],
        };
        let encoded = serde_json::to_string(&report).expect("encode");
        assert_eq!(
            serde_json::from_str::<MigrationReport>(&encoded).expect("decode"),
            report
        );
        assert_eq!(report.applied_count(), 1);
        assert_eq!(report.applied_versions().collect::<Vec<_>>(), vec![1]);

        let status: MigrationStatus =
            serde_json::from_str("{}").expect("missing fields are tolerated");
        assert_eq!(status.current, None);
        assert_eq!(status.pending, Vec::<u32>::new());
        assert_eq!(status.applied, 0);
    }
}
