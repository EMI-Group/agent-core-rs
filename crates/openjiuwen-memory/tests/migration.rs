// `missing_docs` is allowed before the `cfg` gate: when the `sqlite` feature is off the
// whole crate (including `//!` docs) is stripped, and a later `allow` would be stripped
// with it, leaving the denied `missing_docs` lint to fire on the empty crate root.
#![allow(missing_docs)]
#![cfg(feature = "sqlite")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration coverage of the SQLite schema-migration runner.
//!
//! [`openjiuwen_memory::store::migration`] versions the memory SQLite schema: a version
//! table records one row per applied step, the builtin plan ships v1 (`kv_store`) and v2
//! (`documents`), and every step is idempotent. This suite drives the runner through its
//! public entry points only — [`migrate_sqlite`], [`MemoryStores::sqlite_migrated`], and
//! [`LongTermMemory::sqlite_migrated`] — and pins what is observable from outside: the
//! versions a fresh database applies, the no-op of a second run, the pre- and post-run
//! [`MigrationRunner::status`] of the builtin and of a custom plan, and records written
//! through a migrated database surviving a drop and reopen.
//!
//! [`MigrationRunner::status`] and [`MigrationRunner::run`] take a `sqlx` pool. The
//! memory store bundle holds `Arc<dyn KvStore>` (no downcast seam), so the pool comes
//! from the foundation's public SQLite key-value store over the same path — the suite
//! needs no direct `sqlx` dependency.
//!
//! Every test owns its [`tempfile::TempDir`] and derives the database path from it, and
//! record timestamps are pinned, so nothing depends on the wall clock.

use std::path::Path;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_foundation::store::SqliteKvStore;
use openjiuwen_memory::config::MemoryEngineConfig;
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{AddOptions, DedupePolicy, MemoryScope, NewMemory, SearchOptions};
use openjiuwen_memory::store::migration::{
    builtin_migrations, default_registry, migrate_sqlite, Migration, MigrationRegistry,
    MigrationRunner,
};
use openjiuwen_memory::store::MemoryStores;

/// The content of the record written by the round-trip tests.
const RECORD_TEXT: &str = "Alice prefers dark mode for every editor";

/// The engine configuration shared by every test.
fn config() -> MemoryEngineConfig {
    MemoryEngineConfig::default()
}

/// A fixed creation time so every assertion is wall-clock independent.
fn fixed_timestamp(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed timestamp")
}

/// The SQLite URL of `path`, the form the foundation stores take.
fn url(path: &Path) -> String {
    format!("sqlite:{}", path.display())
}

/// The versions of `registry`, in execution order.
fn versions(registry: &MigrationRegistry) -> Vec<u32> {
    registry
        .migrations()
        .iter()
        .map(|migration| migration.version)
        .collect()
}

/// Opens an independent pool over `path` for the migration runner.
///
/// The foundation store creates the file when it does not exist yet, so the helper also
/// serves a database the runner has never seen.
async fn pool_store(path: &Path) -> SqliteKvStore {
    SqliteKvStore::connect(&url(path))
        .await
        .expect("open the pool used by the migration runner")
}

/// A minimal valid step with the given version and name.
fn step(version: u32, name: &'static str) -> Migration {
    Migration::new(version, name, "test step", &["SELECT 1"])
}

/// Search options that keep every candidate, whatever its score.
fn permissive(top_k: usize) -> SearchOptions {
    SearchOptions::default()
        .with_threshold(0.0)
        .with_top_k(top_k)
}

/// Adds one record with dedupe disabled and returns the engine-minted id.
async fn add_record(ltm: &LongTermMemory, scope: &MemoryScope, memory: NewMemory) -> String {
    let options = AddOptions::default().with_dedupe(DedupePolicy::Add);
    let outcome = ltm.add(scope, memory, &options).await.expect("add record");
    assert_eq!(
        outcome.added.len(),
        1,
        "the write must add exactly one record: {outcome:?}"
    );
    outcome.added[0].id.clone()
}

#[tokio::test]
async fn migrate_sqlite_applies_the_builtin_versions_in_order() {
    let directory = tempfile::tempdir().expect("tempdir");
    // The parent directory does not exist yet: `migrate_sqlite` creates it.
    let path = directory.path().join("nested").join("memory.db");
    let expected = versions(&default_registry());

    let report = migrate_sqlite(&path).await.expect("migrate fresh database");

    assert!(report.applied_count() > 0, "{report:?}");
    assert!(!report.is_noop(), "a fresh database is not a no-op");
    assert_eq!(report.applied_versions().collect::<Vec<u32>>(), expected);
    assert_eq!(report.applied_count(), expected.len());
    assert_eq!(report.from_version, None, "nothing was recorded before");
    assert_eq!(report.to_version, expected.last().copied());
    // Ascending: the report lists exactly the registry, in execution order.
    let applied = report.applied_versions().collect::<Vec<u32>>();
    assert!(
        applied.windows(2).all(|pair| pair[0] < pair[1]),
        "versions must ascend: {applied:?}"
    );
    // The builtin plan is what `builtin_migrations()` describes.
    let names: Vec<&str> = report
        .applied
        .iter()
        .map(|migration| migration.name.as_str())
        .collect();
    let builtin: Vec<&str> = builtin_migrations()
        .iter()
        .map(|migration| migration.name)
        .collect();
    assert_eq!(names, builtin);
    assert!(report
        .applied
        .iter()
        .all(|migration| !migration.applied_at.is_empty()));
}

#[tokio::test]
async fn a_second_migration_of_the_same_path_applies_nothing() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");

    let first = migrate_sqlite(&path).await.expect("first migration");
    assert!(
        !first.is_noop(),
        "the first run applies the plan: {first:?}"
    );

    let second = migrate_sqlite(&path).await.expect("second migration");
    assert!(second.is_noop(), "the second run is a no-op: {second:?}");
    assert_eq!(second.applied_count(), 0);
    assert_eq!(
        second.applied_versions().count(),
        0,
        "an up-to-date database records no new version"
    );
    assert_eq!(
        second.from_version, first.to_version,
        "the second run starts where the first ended"
    );
    assert_eq!(
        second.to_version, first.to_version,
        "a no-op run never moves the version"
    );
}

#[tokio::test]
async fn status_reports_pending_before_migrating_and_current_after() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let runner = MigrationRunner::new(default_registry());
    let expected = versions(&default_registry());

    {
        // A database the runner has never touched: nothing recorded, everything pending.
        let fresh = pool_store(&path).await;
        let status = runner.status(fresh.pool()).await.expect("fresh status");
        assert_eq!(status.current, None);
        assert_eq!(status.applied, 0);
        assert_eq!(status.pending, expected, "every builtin version is pending");
        assert!(
            !runner
                .is_up_to_date(fresh.pool())
                .await
                .expect("fresh up to date"),
            "an unmigrated database is not up to date"
        );
    }

    let report = migrate_sqlite(&path).await.expect("migrate");

    let migrated = pool_store(&path).await;
    let status = runner.status(migrated.pool()).await.expect("status");
    assert_eq!(status.current, runner.registry().latest_version());
    assert_eq!(status.current, report.to_version);
    assert!(status.pending.is_empty(), "{status:?}");
    assert_eq!(status.applied, report.applied_count());
    assert!(
        runner
            .is_up_to_date(migrated.pool())
            .await
            .expect("up to date"),
        "a migrated database records the whole plan"
    );
}

#[tokio::test]
async fn records_written_through_migrated_stores_survive_a_reopen() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let scope = MemoryScope::new("alice", "ws");

    let id = {
        let stores = MemoryStores::sqlite_migrated(&path)
            .await
            .expect("migrated stores");
        let ltm = LongTermMemory::with_stores(config(), stores)
            .await
            .expect("facade over the migrated stores");
        let id = add_record(
            &ltm,
            &scope,
            NewMemory::semantic(RECORD_TEXT).with_timestamp(fixed_timestamp(0)),
        )
        .await;
        let read_back = ltm
            .get(&scope, &id)
            .await
            .expect("get record")
            .expect("the record is readable immediately");
        assert_eq!(read_back.content, RECORD_TEXT);
        assert_eq!(read_back.timestamp, fixed_timestamp(0));
        id
    };

    // Reopening an already migrated path is a no-op migration, so anything observed here
    // was committed by the write above.
    let reopened = MemoryStores::sqlite_migrated(&path)
        .await
        .expect("reopen migrated stores");
    let ltm = LongTermMemory::with_stores(config(), reopened)
        .await
        .expect("facade over the reopened stores");
    let record = ltm
        .get(&scope, &id)
        .await
        .expect("get record")
        .expect("the record must survive the reopen");
    assert_eq!(record.id, id);
    assert_eq!(record.content, RECORD_TEXT);
    assert_eq!(record.timestamp, fixed_timestamp(0));
    assert_eq!(record.scope, scope);
}

#[tokio::test]
async fn sqlite_migrated_facade_round_trips_a_scoped_record() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    // A session- and agent-scoped record: the migrated database must preserve the scope,
    // so recall is still narrowed to the caller's session and agent.
    let scope = MemoryScope::new("alice", "ws").session("s1").agent("a1");

    let ltm = LongTermMemory::sqlite_migrated(&path, config())
        .await
        .expect("migrated facade");
    let id = add_record(
        &ltm,
        &scope,
        NewMemory::semantic(RECORD_TEXT)
            .with_timestamp(fixed_timestamp(1))
            .with_metadata("theme", serde_json::json!("dark")),
    )
    .await;

    let stored = ltm
        .get(&scope, &id)
        .await
        .expect("get record")
        .expect("the record is present");
    assert_eq!(stored.content, RECORD_TEXT);
    assert_eq!(stored.scope, scope);

    let listed = ltm.list(&scope, &[]).await.expect("list records");
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0].id, id);

    let hits = ltm
        .search(&scope, "dark mode editor", &permissive(5))
        .await
        .expect("search");
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].record.id, id);
    assert_eq!(
        hits[0].record.metadata.get("theme").cloned(),
        Some(serde_json::json!("dark"))
    );

    // The scope is part of the storage key, so another session sees nothing.
    let other_session = MemoryScope::new("alice", "ws").session("s2").agent("a1");
    assert!(
        ltm.list(&other_session, &[])
            .await
            .expect("list other session")
            .is_empty(),
        "a different session must not see the record"
    );
    drop(ltm);

    // Migrating again after the facade wrote data is still a no-op, and the database
    // records the whole plan.
    let report = migrate_sqlite(&path).await.expect("re-migrate");
    assert!(report.is_noop(), "{report:?}");
    let pool = pool_store(&path).await;
    assert!(
        MigrationRunner::new(default_registry())
            .is_up_to_date(pool.pool())
            .await
            .expect("up to date"),
        "the migrated database must record every builtin version"
    );
}

#[tokio::test]
async fn a_custom_registry_runs_to_completion_and_is_current() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let registry = default_registry()
        .with(Migration::new(
            3,
            "custom_notes",
            "downstream table appended by an application module",
            &["CREATE TABLE IF NOT EXISTS custom_notes (id TEXT PRIMARY KEY, body TEXT NOT NULL)"],
        ))
        .expect("append v3");
    let runner = MigrationRunner::new(registry);
    assert_eq!(runner.registry().len(), 3);
    assert_eq!(runner.registry().latest_version(), Some(3));
    assert_eq!(
        versions(runner.registry()),
        [1, 2, 3],
        "the custom plan extends the builtin one"
    );

    let pool = pool_store(&path).await;
    let status = runner.status(pool.pool()).await.expect("fresh status");
    assert_eq!(status.current, None);
    assert_eq!(status.pending, vec![1, 2, 3]);

    let report = runner.run(pool.pool()).await.expect("run the custom plan");
    assert_eq!(report.applied_versions().collect::<Vec<u32>>(), [1, 2, 3]);
    assert_eq!(report.from_version, None);
    assert_eq!(report.to_version, Some(3));

    let status = runner.status(pool.pool()).await.expect("status");
    assert_eq!(status.current, Some(3));
    assert_eq!(status.applied, 3);
    assert!(status.pending.is_empty(), "{status:?}");
    assert!(
        runner.is_up_to_date(pool.pool()).await.expect("up to date"),
        "the custom plan is fully applied"
    );
    // The builtin plan is a prefix of the custom one, so the same database is up to date
    // for the shipped registry too: a newer version is never a pending step.
    assert!(MigrationRunner::new(default_registry())
        .is_up_to_date(pool.pool())
        .await
        .expect("builtin plan"));

    // Validation is part of the registry, and the documented rejections are errors, not
    // panics: versions must be unique and must start at one.
    let duplicate = MigrationRegistry::new(vec![step(1, "one"), step(1, "two")])
        .expect_err("duplicate versions are rejected");
    assert!(matches!(duplicate, Error::Validation(_)), "{duplicate}");
    assert!(
        duplicate.to_string().contains("duplicate version v1"),
        "{duplicate}"
    );

    let starts_above_one =
        MigrationRegistry::new(vec![step(2, "two")]).expect_err("a plan must start at v1");
    assert!(
        matches!(starts_above_one, Error::Validation(_)),
        "{starts_above_one}"
    );
    assert!(
        starts_above_one.to_string().contains("expected v1"),
        "{starts_above_one}"
    );
}
