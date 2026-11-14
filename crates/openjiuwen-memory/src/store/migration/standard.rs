//! The standard migrator set: every family in one planner, registry, and run entry point.
//!
//! [`standard_migrators`] assembles the four migrator families the crate ships, in the
//! only order that is safe:
//!
//! 1. [`SqlMigrator`] — legacy variable rows into the current prefixes, then the
//!    `scope_user_mapping` backfill. It runs first because every later family derives its
//!    target from the same [`KvPrefixRegistry`], and the mapping is derived from the
//!    record keyspace, so the layout must be settled before anything reads it.
//! 2. [`MessageMigrator`] — legacy message rows into the `messages` document collection,
//!    whose name also comes from the prefix layout the first step has just normalized.
//! 3. [`VectorMigrator`] — legacy embedding collections into the canonical
//!    `uid_{user}_gid_{scope}_mtype_{kind}` names. Collection names encode user, scope and
//!    kind, so recollection must happen after the record/variable layout is final.
//! 4. [`IndexVersionMigrator`] — index stamps last, because it reads the *resulting*
//!    collection set: stamping before recollection would stamp a collection that is about
//!    to be renamed away and would miss the collection it became.
//!
//! The set is a plain value: there is no process-global registry, so a caller can append
//! its own migrator to [`standard_planner`], run a subset, or hand a shared
//! [`MigrationJournal`] across processes. Deferred capabilities (for example a remote lock
//! broker coordinating two nodes on one keyspace) surface as
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) rather than a
//! silent no-op, so an unsupported run never looks like a completed one.
//!
//! [`KvPrefixRegistry`]: crate::store::prefix::KvPrefixRegistry

use std::sync::Arc;

use openjiuwen_common::error::Result;

use super::index_version::IndexVersionMigrator;
use super::message::MessageMigrator;
use super::operation::{MigrationJournal, MigrationRun, OperationRegistry};
use super::plan::{MigrationContext, MigrationPlan, MigrationPlanner, Migrator};
use super::sql::SqlMigrator;
use super::vector::VectorMigrator;

/// The standard migrator set, in execution order: variable layout, messages, embedding
/// collections, then index stamps.
#[must_use]
pub fn standard_migrators() -> Vec<Arc<dyn Migrator>> {
    vec![
        Arc::new(SqlMigrator::new()),
        Arc::new(MessageMigrator::new()),
        Arc::new(VectorMigrator::new()),
        Arc::new(IndexVersionMigrator::new()),
    ]
}

/// A planner over [`standard_migrators`].
#[must_use]
pub fn standard_planner() -> MigrationPlanner {
    MigrationPlanner::new(standard_migrators())
}

/// A registry holding every standard migrator's operations (duplicate or invalid ids are
/// [`Error::Validation`](openjiuwen_common::error::Error::Validation)).
///
/// The registry is what makes a persisted [`MigrationPlan`] executable: every id a
/// standard plan can contain is registered here, and a duplicated or malformed id fails
/// eagerly instead of shadowing a handler.
pub fn standard_registry() -> Result<OperationRegistry> {
    standard_planner().registry()
}

/// Plans the standard set against `context` (read-only; `context.dry_run` mirrored on the
/// plan).
///
/// Planning scans but never writes, so the result is safe to inspect, persist, or hand to
/// an operator before [`run_migrations`] applies it.
pub async fn plan_migrations(context: &MigrationContext) -> Result<MigrationPlan> {
    standard_planner().plan(context).await
}

/// Plans and runs the standard set with a fresh journal.
pub async fn run_migrations(context: &MigrationContext) -> Result<MigrationRun> {
    run_migrations_with(context, &mut MigrationJournal::new()).await
}

/// Plans and runs the standard set, resuming from (and updating) a caller-held journal.
///
/// One journal is shared by every migrator of the set, so an operation that already
/// succeeded in `journal` is reported `already completed` instead of running again; pass
/// the journal of an interrupted run to continue it.
pub async fn run_migrations_with(
    context: &MigrationContext,
    journal: &mut MigrationJournal,
) -> Result<MigrationRun> {
    standard_planner().run(context, journal).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;

    use chrono::{DateTime, Utc};
    use serde_json::{Map, Value};

    use openjiuwen_foundation::store::{
        CollectionSchema, FieldSchema, KvValue, Query, VectorDataType,
    };

    use crate::record::{MemoryScope, MemoryType};
    use crate::store::migration::index_version::{
        CURRENT_INDEX_VERSION, PURGE_STALE_EMBEDDINGS, STAMP_COLLECTIONS,
    };
    use crate::store::migration::message::{IMPORT_LEGACY_ROWS, LEGACY_MESSAGE_PREFIXES};
    use crate::store::migration::operation::{OperationOutcome, OperationReport};
    use crate::store::migration::sql::{COPY_SESSION_VARIABLES, COPY_USER_VARIABLES};
    use crate::store::migration::vector::RECOLLECT;
    use crate::store::migration::{IndexStamp, MigrationJournal};
    use crate::store::prefix::{KvFamily, KvPrefixRegistry};
    use crate::store::{keys, MemoryStores, StoredMessage};

    /// The scope every seeded row belongs to.
    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    /// An in-memory store bundle.
    fn stores() -> MemoryStores {
        MemoryStores::in_memory()
    }

    /// Writes one raw KV row.
    async fn seed(stores: &MemoryStores, key: &str, value: &str) {
        stores
            .kv
            .set(key, KvValue::text(value))
            .await
            .expect("seed key");
    }

    /// One raw KV row's text, when it exists and is text.
    async fn kv_text(stores: &MemoryStores, key: &str) -> Option<String> {
        stores
            .kv
            .get(key)
            .await
            .expect("read key")
            .and_then(|value| value.as_str().map(str::to_owned))
    }

    /// Every key under `prefix`, sorted.
    async fn kv_keys(stores: &MemoryStores, prefix: &str) -> Vec<String> {
        stores
            .kv
            .get_by_prefix(prefix)
            .await
            .expect("scan prefix")
            .into_keys()
            .collect()
    }

    /// Seeds one legacy message row under `user_message/{key}`.
    async fn seed_legacy_message(stores: &MemoryStores, key: &str, row: &str) {
        seed(
            stores,
            &format!("{}/{key}", LEGACY_MESSAGE_PREFIXES[0]),
            row,
        )
        .await;
    }

    /// Every document of the `messages` collection.
    async fn message_documents(stores: &MemoryStores) -> Vec<StoredMessage> {
        stores
            .document
            .query(keys::MESSAGES_COLLECTION, &Query::default())
            .await
            .expect("query messages")
            .into_iter()
            .map(|document| {
                serde_json::from_value(document.to_value()).expect("stored message decodes")
            })
            .collect()
    }

    /// A three-dimensional collection schema with a string primary key.
    fn vector_schema() -> CollectionSchema {
        CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_owned(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_owned(),
                dtype: VectorDataType::FloatVector,
                dim: Some(3),
                ..FieldSchema::default()
            },
        ])
    }

    /// One document of [`vector_schema`].
    fn vector_document(id: &str, vector: [f32; 3]) -> Map<String, Value> {
        let mut document = Map::new();
        document.insert("id".to_owned(), Value::from(id));
        document.insert(
            "embedding".to_owned(),
            Value::Array(vector.iter().map(|value| Value::from(*value)).collect()),
        );
        document
    }

    /// Creates `collection` and inserts `documents` into it.
    async fn seed_collection(
        stores: &MemoryStores,
        collection: &str,
        documents: Vec<Map<String, Value>>,
    ) {
        stores
            .vector
            .create_collection(collection, vector_schema())
            .await
            .expect("collection created");
        if !documents.is_empty() {
            stores
                .vector
                .add_docs(collection, documents)
                .await
                .expect("documents added");
        }
    }

    /// The ids of every document in `collection`, sorted.
    async fn collection_ids(stores: &MemoryStores, collection: &str) -> Vec<String> {
        let mut ids: Vec<String> =
            super::super::plan::scan_collection(stores.vector.as_ref(), collection)
                .await
                .expect("collection scan")
                .iter()
                .filter_map(|document| {
                    document
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect();
        ids.sort();
        ids
    }

    /// Every collection the vector store holds, sorted.
    async fn collection_names(stores: &MemoryStores) -> Vec<String> {
        let mut names = stores
            .vector
            .list_collection_names()
            .await
            .expect("collection names");
        names.sort();
        names
    }

    /// Seeds one row of every family the standard set migrates.
    async fn seed_every_family(stores: &MemoryStores) -> (String, String, String) {
        let scope = scope();
        seed(stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        seed(
            stores,
            &keys::legacy_session_variable_key(&scope, "s1", "step"),
            "2",
        )
        .await;
        seed_legacy_message(
            stores,
            "a",
            r#"{"message_id":"m1","user_id":"alice","scope_id":"ws","content":"hello","role":"user","timestamp":1700000000000}"#,
        )
        .await;
        seed_legacy_message(
            stores,
            "b",
            r#"{"id":"m2","user_id":"alice","scope_id":"ws","session_id":"s1","content":"hi","role":"assistant","timestamp":"2026-01-02T03:04:05Z"}"#,
        )
        .await;
        let legacy = "uid_alice_gid_ws_mtype_semantic";
        let canonical = keys::vector_collection(&scope, MemoryType::SemanticMemory);
        seed_collection(
            stores,
            legacy,
            vec![
                vector_document("v1", [1.0, 0.0, 0.0]),
                vector_document("v2", [0.0, 1.0, 0.0]),
            ],
        )
        .await;
        // A canonical collection that no migrator has ever stamped.
        let unstamped = "uid_bob_gid_ws_mtype_coding";
        seed_collection(
            stores,
            unstamped,
            vec![vector_document("c1", [0.0, 0.0, 1.0])],
        )
        .await;
        (legacy.to_owned(), canonical, unstamped.to_owned())
    }

    /// Asserts the variables of [`scope`] live in the current layout only.
    async fn assert_variables_migrated(stores: &MemoryStores) {
        let scope = scope();
        assert_eq!(
            kv_text(stores, &keys::variable_key(&scope, "theme"))
                .await
                .as_deref(),
            Some("dark")
        );
        assert_eq!(
            kv_text(stores, &keys::session_variable_key(&scope, "s1", "step"))
                .await
                .as_deref(),
            Some("2")
        );
        assert_eq!(
            kv_keys(stores, keys::LEGACY_VARIABLE_PREFIX).await,
            Vec::<String>::new(),
            "no legacy user-variable key survives"
        );
        assert_eq!(
            kv_keys(stores, keys::LEGACY_SESSION_VARIABLE_PREFIX).await,
            Vec::<String>::new(),
            "no legacy session-variable key survives"
        );
    }

    /// Asserts the legacy message rows became `expected` documents and left no key behind.
    async fn assert_messages_migrated(stores: &MemoryStores, expected: &[&str]) {
        let mut message_ids: Vec<String> = message_documents(stores)
            .await
            .into_iter()
            .map(|message| message.id)
            .collect();
        message_ids.sort();
        assert_eq!(message_ids, expected);
        assert_eq!(
            kv_keys(stores, LEGACY_MESSAGE_PREFIXES[0]).await,
            Vec::<String>::new(),
            "no legacy message row survives"
        );
    }

    /// Asserts the recollected, stamped state of the vector store.
    async fn assert_vectors_migrated(
        stores: &MemoryStores,
        legacy: &str,
        canonical: &str,
        unstamped: &str,
        now: DateTime<Utc>,
    ) {
        assert_eq!(collection_ids(stores, canonical).await, vec!["v1", "v2"]);
        assert!(
            !stores
                .vector
                .collection_exists(legacy)
                .await
                .expect("existence"),
            "the legacy collection is recollected and removed"
        );
        assert_eq!(
            collection_ids(stores, unstamped).await,
            vec!["c1"],
            "unrelated memory collections are left alone"
        );
        assert_eq!(
            collection_names(stores).await,
            vec![canonical.to_owned(), unstamped.to_owned()]
        );
        for collection in [canonical, unstamped] {
            let stamp = IndexStamp::load(stores.kv.as_ref(), collection)
                .await
                .expect("load stamp")
                .expect("stamp written");
            assert_eq!(stamp.version, CURRENT_INDEX_VERSION);
            assert!(stamp.needs_rebuild);
            assert_eq!(stamp.dimensions, Some(3));
            assert_eq!(stamp.updated_at, Some(now));
        }
        assert!(IndexStamp::load(stores.kv.as_ref(), legacy)
            .await
            .expect("load stamp")
            .is_none());
    }

    #[test]
    fn the_standard_set_is_ordered_and_every_operation_is_registered() {
        let migrators = standard_migrators();
        assert_eq!(
            migrators
                .iter()
                .map(|migrator| migrator.id())
                .collect::<Vec<_>>(),
            vec!["sql", "message", "vector", "index_version"],
            "the families run in dependency order"
        );
        assert_eq!(standard_planner().migrators().len(), migrators.len());
        let registry = standard_registry().expect("standard registry");
        assert_eq!(registry.len(), 7);
        assert_eq!(
            registry
                .operation_ids()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                COPY_USER_VARIABLES,
                COPY_SESSION_VARIABLES,
                "sql.backfill_scope_user_mapping",
                IMPORT_LEGACY_ROWS,
                RECOLLECT,
                STAMP_COLLECTIONS,
                PURGE_STALE_EMBEDDINGS,
            ])
        );
        assert!(!registry.contains("index_version.unknown"));
        assert_eq!(registry.operation_ids(), {
            let mut ids = registry.operation_ids();
            ids.sort_unstable();
            ids
        });
    }

    #[tokio::test]
    async fn the_standard_set_migrates_every_family_and_is_idempotent() {
        let stores = stores();
        let (legacy_collection, canonical, unstamped) = seed_every_family(&stores).await;
        assert_eq!(
            canonical, "uid_alice_gid_ws_mtype_semantic_memory",
            "the canonical name is the one the store derives"
        );
        let context = MigrationContext::new(stores.clone());
        let registry = standard_registry().expect("standard registry");
        let run = run_migrations(&context).await.expect("standard run");
        assert!(run.is_success());
        assert_eq!(
            run.plan.operation_ids(),
            vec![
                COPY_USER_VARIABLES,
                COPY_SESSION_VARIABLES,
                IMPORT_LEGACY_ROWS,
                RECOLLECT,
                STAMP_COLLECTIONS,
            ]
        );
        assert!(run
            .plan
            .operation_ids()
            .iter()
            .all(|operation_id| registry.contains(operation_id)));
        assert_eq!(run.executed(), 5);

        assert_variables_migrated(&stores).await;
        assert_messages_migrated(&stores, &["m1", "m2"]).await;
        assert_vectors_migrated(
            &stores,
            &legacy_collection,
            &canonical,
            &unstamped,
            context.now,
        )
        .await;

        // A second run finds no work and changes nothing.
        let snapshot = (
            kv_keys(&stores, "").await,
            message_documents(&stores).await,
            collection_names(&stores).await,
        );
        let second = run_migrations(&context).await.expect("second run");
        assert!(second.is_success());
        assert_eq!(second.plan.operation_ids(), Vec::<&str>::new());
        assert_eq!(second.executed(), 0);
        assert_eq!(
            (
                kv_keys(&stores, "").await,
                message_documents(&stores).await,
                collection_names(&stores).await
            ),
            snapshot
        );
    }

    #[tokio::test]
    async fn a_dry_run_plans_the_work_and_writes_nothing() {
        let stores = stores();
        let (legacy_collection, canonical, unstamped) = seed_every_family(&stores).await;
        let context = MigrationContext::new(stores.clone()).with_dry_run(true);

        let plan = plan_migrations(&context).await.expect("dry plan");
        assert!(!plan.is_empty());
        assert!(plan.dry_run);
        assert_eq!(plan.len(), 5);
        let run = run_migrations(&context).await.expect("dry run");
        assert!(run.is_success());
        assert_eq!(run.executed(), 0);
        assert_eq!(run.reports.len(), 5);
        for report in &run.reports {
            assert_eq!(report.outcome, OperationOutcome::Skipped);
            assert_eq!(report.note.as_deref(), Some("dry run"));
        }
        assert!(run
            .journal
            .records()
            .all(|record| record.outcome == OperationOutcome::Skipped));

        // Every seeded row is still exactly where it was.
        let scope = scope();
        assert_eq!(
            kv_text(&stores, &keys::legacy_variable_key(&scope, "theme"))
                .await
                .as_deref(),
            Some("dark")
        );
        assert_eq!(
            kv_text(
                &stores,
                &keys::legacy_session_variable_key(&scope, "s1", "step")
            )
            .await
            .as_deref(),
            Some("2")
        );
        assert!(kv_text(&stores, &keys::variable_key(&scope, "theme"))
            .await
            .is_none());
        assert_eq!(
            kv_keys(&stores, LEGACY_MESSAGE_PREFIXES[0]).await.len(),
            2,
            "legacy message rows are untouched"
        );
        assert_eq!(
            message_documents(&stores).await,
            Vec::<StoredMessage>::new(),
            "no message document was written"
        );
        assert_eq!(
            collection_ids(&stores, &legacy_collection).await,
            vec!["v1", "v2"],
            "the legacy collection is untouched"
        );
        assert!(!stores
            .vector
            .collection_exists(&canonical)
            .await
            .expect("existence"));
        for collection in [&canonical, &unstamped] {
            assert!(IndexStamp::load(stores.kv.as_ref(), collection)
                .await
                .expect("load stamp")
                .is_none());
        }
        assert_eq!(
            collection_names(&stores).await,
            {
                let mut names = vec![legacy_collection, unstamped];
                names.sort();
                names
            },
            "no collection was created"
        );
    }

    #[tokio::test]
    async fn a_shared_journal_reports_completed_operations_instead_of_rerunning_them() {
        let stores = stores();
        let scope = scope();
        seed_every_family(&stores).await;
        let context = MigrationContext::new(stores.clone());
        let mut journal = MigrationJournal::new();
        let first = run_migrations_with(&context, &mut journal)
            .await
            .expect("first run");
        assert!(first.is_success());
        assert_eq!(first.executed(), 5);
        assert_eq!(journal.len(), 5);
        let completed = journal.completed();

        // The same journal over unchanged stores plans nothing and runs nothing.
        let repeated = run_migrations_with(&context, &mut journal)
            .await
            .expect("repeated run");
        assert!(repeated.is_success());
        assert!(repeated.plan.is_empty());
        assert_eq!(repeated.reports, Vec::<OperationReport>::new());
        assert_eq!(journal.completed(), completed);

        // Work that shows up after the first pass is planned again, but the journal keeps
        // the completed operations completed: every planned step is reported as already
        // completed and no handler runs.
        seed(&stores, &keys::legacy_variable_key(&scope, "lang"), "rust").await;
        seed_legacy_message(
            &stores,
            "late",
            r#"{"message_id":"m9","user_id":"alice","scope_id":"ws","content":"later","role":"user","timestamp":1700000000002}"#,
        )
        .await;
        let resumed = run_migrations_with(&context, &mut journal)
            .await
            .expect("resumed run");
        assert!(resumed.is_success());
        assert_eq!(
            resumed.plan.operation_ids(),
            vec![COPY_USER_VARIABLES, IMPORT_LEGACY_ROWS],
            "the new rows are planned"
        );
        assert_eq!(resumed.reports.len(), 2);
        for report in &resumed.reports {
            assert_eq!(report.outcome, OperationOutcome::Skipped);
            assert_eq!(report.note.as_deref(), Some("already completed"));
        }
        // `MigrationRun::executed` counts the journal, not this run, so the per-run check
        // is that no report of the resumed run succeeded.
        assert_eq!(
            resumed
                .reports
                .iter()
                .filter(|report| report.outcome == OperationOutcome::Succeeded)
                .count(),
            0
        );
        assert_eq!(journal.len(), 5, "no new attempt was recorded");
        assert_eq!(
            kv_text(&stores, &keys::legacy_variable_key(&scope, "lang"))
                .await
                .as_deref(),
            Some("rust"),
            "a completed operation is not rerun"
        );
        assert_eq!(message_documents(&stores).await.len(), 2);

        // A fresh journal picks the new work up.
        let fresh = run_migrations(&context).await.expect("fresh journal");
        assert!(fresh.is_success());
        assert_eq!(
            kv_text(&stores, &keys::variable_key(&scope, "lang"))
                .await
                .as_deref(),
            Some("rust")
        );
        let mut message_ids: Vec<String> = message_documents(&stores)
            .await
            .into_iter()
            .map(|message| message.id)
            .collect();
        message_ids.sort();
        assert_eq!(
            message_ids,
            vec!["m1".to_owned(), "m2".to_owned(), "m9".to_owned()]
        );
    }

    #[tokio::test]
    async fn the_standard_set_runs_on_local_file_stores() {
        let directory = tempfile::tempdir().expect("temp dir");
        let root = directory.path().join("memory");
        let stores = MemoryStores::local_file(&root).await.expect("local stores");
        let scope = scope();
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        let context = MigrationContext::new(stores);
        let run = run_migrations(&context).await.expect("standard run");
        assert!(run.is_success());

        let reopened = MemoryStores::local_file(&root)
            .await
            .expect("reopened stores");
        assert_eq!(
            kv_text(&reopened, &keys::variable_key(&scope, "theme"))
                .await
                .as_deref(),
            Some("dark"),
            "the copied row is durable"
        );
        assert!(
            kv_text(&reopened, &keys::legacy_variable_key(&scope, "theme"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_renamed_user_variables_prefix_flows_through_the_standard_set() {
        let stores = stores();
        let scope = scope();
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        let prefixes = KvPrefixRegistry::default()
            .with_current(KvFamily::UserVariables, "vars_v2")
            .expect("renamed family");
        let context = MigrationContext::new(stores.clone()).with_prefixes(prefixes);
        let run = run_migrations(&context).await.expect("standard run");
        assert!(run.is_success());
        assert_eq!(
            kv_text(&stores, "vars_v2/alice/ws/theme").await.as_deref(),
            Some("dark"),
            "the row lands under the renamed prefix"
        );
        assert!(
            kv_text(&stores, &keys::legacy_variable_key(&scope, "theme"))
                .await
                .is_none()
        );
        assert!(
            kv_text(&stores, &keys::variable_key(&scope, "theme"))
                .await
                .is_none(),
            "the demoted prefix stays a read-only alias"
        );
    }
}
