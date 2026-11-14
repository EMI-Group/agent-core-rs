//! The migration registry: versioned steps, their validation, and the builtin plan.
//!
//! A [`Migration`] is a version, a name, a description, and the DDL it executes; a
//! [`MigrationRegistry`] is an ordered, validated list of them. Every insertion
//! revalidates the whole list, so every value that exists satisfies the crate's
//! invariants: versions start at `1`, are contiguous and strictly ascending, names are
//! non-empty and unique, and no step is empty.
//!
//! This half compiles without the `sqlite` feature, so a caller can describe a plan on
//! every build; only the runner (`MigrationRunner`, `MigrationStatus`, `migrate_sqlite`,
//! in `super::runner`) needs `sqlx`. The builtin plan ships v1 (`kv_store`) and v2
//! (`documents`), whose DDL mirrors the foundation's SQLite stores exactly and is
//! `IF NOT EXISTS`, so migrating an already-populated foundation database only records
//! the version rows.

use openjiuwen_common::error::{Error, Result};

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
