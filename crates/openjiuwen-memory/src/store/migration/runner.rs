//! The forward-only SQLite migration runner (feature `sqlite`).
//!
//! Every item in this module is gated on the `sqlite` feature, because the runner is
//! the only part of the migration stack that touches a database: it creates the version
//! table (`SCHEMA_VERSION_TABLE`) idempotently, reads the recorded rows, and applies
//! whatever the registry has above them.
//!
//! - **Version table.** Infrastructure, not a migration: one row per applied step
//!   (`version` primary key, `name`, RFC 3339 `applied_at`), created idempotently
//!   before every status/run.
//! - **Atomic per migration.** Each migration's DDL and its version row commit in one
//!   transaction, so a failure records nothing and rolls the DDL back. The one
//!   documented exception is `PRAGMA`, which SQLite cannot roll back: connection
//!   pragmas belong in pool setup, never in a migration step.
//! - **Forward-only.** `status` reports pending versions relative to the recorded
//!   maximum; a database ahead of the plan (for example after a library downgrade) only
//!   logs a warning, and nothing is pending or downgraded.

#[cfg(feature = "sqlite")]
use super::registry::{default_registry, Migration, MigrationRegistry};
#[cfg(feature = "sqlite")]
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

#[cfg(all(test, feature = "sqlite"))]
mod sqlite_tests {
    use super::super::registry::builtin_migrations;
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
