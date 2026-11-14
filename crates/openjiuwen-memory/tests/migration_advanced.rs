//! Integration coverage of the memory *data*-migration suite and its operational seams.
//!
//! The SQLite schema runner has its own suite (`tests/migration.rs`); this file drives the
//! data layer that converts rows inside an already-open database:
//!
//! - the standard planner (`standard_planner` plus `run_migrations`) over a keyspace seeded
//!   with historical prefixes, and the idempotency of a second run at the same reference time,
//! - journal-keyed resume after a handler failure,
//! - dry runs (a plan reports the work, nothing is created or deleted),
//! - key-value backup/restore parity (the `backup` module's `dump` and `restore`) on
//!   in-memory and local-file stores, plus verification and JSON round trips,
//! - prefix-registry renames surviving a backup round trip, and prefix validation,
//! - the distributed KV lock seam (`InProcessKvLock`, `FileKvLock`, `UnsupportedKvLock`),
//! - the persisted `scope_user_mapping` store.
//!
//! Every test is offline and deterministic: `tempfile` roots, fixed
//! `DateTime::from_timestamp` values (including `MigrationContext::now`), and bounded lock
//! waits of at most 250 ms. The `sqlite`-gated test at the end proves the same data
//! migrations and a backup round trip on the SQLite backend.
//!
//! Record keys are deliberately *not* rewritten by the suite: the mapping backfill
//! (`sql.backfill_scope_user_mapping`) derives its pairs from the record keyspace and leaves
//! the legacy record rows readable through the store's documented legacy read-through.
//! The assertions below pin exactly that behavior.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::store::{
    CollectionSchema, FieldSchema, KvValue, VectorDataType, VectorQuery,
};
use openjiuwen_memory::config::{MemoryEngineConfig, MemoryScopeConfig};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{AddOptions, MemoryRecord, MemoryScope, MemoryType, NewMemory};
use openjiuwen_memory::store::backup::{self, MemoryBackup, BACKUP_FORMAT_VERSION};
use openjiuwen_memory::store::keys;
use openjiuwen_memory::store::kv_lock::{
    DistributedKvLock, FileKvLock, InProcessKvLock, LockOptions, UnsupportedKvLock,
};
use openjiuwen_memory::store::mapping::ScopeUserMappingStore;
use openjiuwen_memory::store::migration::index_version::{
    IndexStamp, CURRENT_INDEX_VERSION, PURGE_STALE_EMBEDDINGS, STAMP_COLLECTIONS,
};
use openjiuwen_memory::store::migration::message::{IMPORT_LEGACY_ROWS, LEGACY_MESSAGE_PREFIXES};
use openjiuwen_memory::store::migration::operation::{
    MigrationJournal, MigrationOperationHandler, OperationOutcome, OperationRegistry,
    OperationReport,
};
use openjiuwen_memory::store::migration::sql::{
    BACKFILL_SCOPE_USER_MAPPING, COPY_SESSION_VARIABLES, COPY_USER_VARIABLES,
};
use openjiuwen_memory::store::migration::vector::RECOLLECT;
use openjiuwen_memory::store::migration::{
    plan_migrations, run_migrations, standard_registry, MigrationContext, MigrationPlan,
    MigrationStep,
};
use openjiuwen_memory::store::prefix::{KvFamily, KvPrefixRegistry};
use openjiuwen_memory::store::MemoryStores;

/// Id of the record seeded under the legacy record prefix.
const RECORD_ID: &str = "legacy-record-1";

/// The content of [`RECORD_ID`].
const RECORD_TEXT: &str = "Alice prefers dark mode for every editor";

/// The two record contents the backup fixtures write through the facade.
const CONTENTS: [&str; 2] = [
    "Alice prefers dark mode for every editor",
    "Alice ships on Friday afternoons",
];

/// The scope every record and user variable belongs to.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// A scope with a session, so `set_variable` writes a session variable.
fn session_scope() -> MemoryScope {
    MemoryScope::new("alice", "ws").session("s1")
}

/// The engine configuration shared by every facade instance.
fn config() -> MemoryEngineConfig {
    MemoryEngineConfig::default()
}

/// The scope configuration the backup fixtures write.
fn scope_config() -> MemoryScopeConfig {
    MemoryScopeConfig {
        user_profile_definition: Some("stable identity facts".to_owned()),
        semantic_memory_definition: Some("durable project facts".to_owned()),
        episodic_memory_definition: Some("dated events".to_owned()),
    }
}

/// A fixed reference time so no assertion depends on the wall clock.
fn fixed(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed timestamp")
}

/// A context pinned to [`fixed`] so stamps and issue reports are reproducible.
fn context(stores: &MemoryStores) -> MigrationContext {
    let mut context = MigrationContext::new(stores.clone());
    context.now = fixed(0);
    context
}

/// Writes one raw key-value row.
async fn seed(stores: &MemoryStores, key: &str, value: &str) {
    stores
        .kv
        .set(key, KvValue::text(value))
        .await
        .expect("seed key");
}

/// One raw row's text, when the key exists and holds text.
async fn kv_text(stores: &MemoryStores, key: &str) -> Option<String> {
    stores
        .kv
        .get(key)
        .await
        .expect("read key")
        .and_then(|value| value.as_str().map(str::to_owned))
}

/// Every key under `prefix`, sorted; `prefix` is one key segment, so the scan uses `prefix/`.
async fn kv_keys(stores: &MemoryStores, prefix: &str) -> Vec<String> {
    let mut keys: Vec<String> = stores
        .kv
        .get_by_prefix(&format!("{prefix}/"))
        .await
        .expect("scan prefix")
        .into_keys()
        .collect();
    keys.sort();
    keys
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
fn vector_document(id: &str, vector: [f32; 3]) -> serde_json::Map<String, serde_json::Value> {
    let mut document = serde_json::Map::new();
    document.insert("id".to_owned(), serde_json::Value::from(id));
    document.insert(
        "embedding".to_owned(),
        serde_json::Value::Array(
            vector
                .iter()
                .map(|value| serde_json::Value::from(*value))
                .collect(),
        ),
    );
    document
}

/// Creates `collection` and inserts `documents` into it.
async fn seed_collection(
    stores: &MemoryStores,
    collection: &str,
    documents: Vec<serde_json::Map<String, serde_json::Value>>,
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
///
/// The vector store has no listing API, so this is the documented snapshot scan: a zero
/// vector with a generous `top_k` and no filter.
async fn collection_ids(stores: &MemoryStores, collection: &str) -> Vec<String> {
    let query = VectorQuery::new(vec![0.0, 0.0, 0.0]).with_top_k(64);
    let mut ids: Vec<String> = stores
        .vector
        .search(collection, &query)
        .await
        .expect("vector search")
        .into_iter()
        .filter_map(|hit| {
            hit.fields
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .collect();
    ids.sort();
    ids
}

/// Seeds one row of every family the standard set migrates.
///
/// Returns the canonical embedding collection the recollection targets.
async fn seed_legacy_world(stores: &MemoryStores) -> String {
    let scope = scope();
    seed(stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
    seed(
        stores,
        &keys::legacy_session_variable_key(&scope, "s1", "step"),
        "2",
    )
    .await;
    let record = MemoryRecord::new(scope.clone(), RECORD_TEXT, MemoryType::SemanticMemory)
        .with_id(RECORD_ID)
        .with_timestamp(fixed(0));
    seed(
        stores,
        &keys::legacy_record_key(&scope, RECORD_ID),
        &serde_json::to_string(&record).expect("record JSON"),
    )
    .await;
    seed(
        stores,
        &format!("{}/row-1", LEGACY_MESSAGE_PREFIXES[0]),
        r#"{"message_id":"m1","user_id":"alice","scope_id":"ws","session_id":"s1","content":"hello","role":"user","timestamp":1700000000000}"#,
    )
    .await;
    seed_collection(
        stores,
        "uid_alice_gid_ws_mtype_semantic",
        vec![
            vector_document("v1", [1.0, 0.0, 0.0]),
            vector_document("v2", [0.0, 1.0, 0.0]),
        ],
    )
    .await;
    // A canonical collection that no migrator has ever stamped.
    seed_collection(
        stores,
        "uid_bob_gid_ws_mtype_coding",
        vec![vector_document("c1", [0.0, 0.0, 1.0])],
    )
    .await;
    keys::vector_collection(&scope, MemoryType::SemanticMemory)
}

// ---------------------------------------------------------------------------
// Scenario 1 + 2: the standard planner, and the idempotency of a second run.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_standard_planner_migrates_a_seeded_legacy_keyspace() {
    let stores = MemoryStores::in_memory();
    let canonical = seed_legacy_world(&stores).await;

    let run = run_migrations(&context(&stores))
        .await
        .expect("standard run");
    assert!(run.is_success(), "{:?}", run.reports);

    // The registry holds exactly the seven documented operations, and a standard run plans
    // the six of them that apply to this keyspace: `index_version.purge_stale_embeddings`
    // is registered but only planned by an `IndexVersionMigrator` built with
    // `IndexVersionOptions { purge_stale: true }`, which `standard_migrators` never uses.
    let registry = standard_registry().expect("registry");
    let mut registered = registry.operation_ids();
    registered.sort_unstable();
    let mut documented = vec![
        COPY_USER_VARIABLES,
        COPY_SESSION_VARIABLES,
        BACKFILL_SCOPE_USER_MAPPING,
        IMPORT_LEGACY_ROWS,
        RECOLLECT,
        STAMP_COLLECTIONS,
        PURGE_STALE_EMBEDDINGS,
    ];
    documented.sort_unstable();
    assert_eq!(
        registered, documented,
        "the standard registry ships the seven documented operation ids"
    );
    let mut reported: Vec<&str> = run
        .reports
        .iter()
        .map(|report| report.operation_id.as_str())
        .collect();
    reported.sort_unstable();
    let mut applicable = vec![
        COPY_USER_VARIABLES,
        COPY_SESSION_VARIABLES,
        BACKFILL_SCOPE_USER_MAPPING,
        IMPORT_LEGACY_ROWS,
        RECOLLECT,
        STAMP_COLLECTIONS,
    ];
    applicable.sort_unstable();
    assert_eq!(
        reported, applicable,
        "every applicable operation is reported; the purge is opt-in"
    );
    assert_eq!(
        run.plan.operation_ids(),
        vec![
            COPY_USER_VARIABLES,
            COPY_SESSION_VARIABLES,
            BACKFILL_SCOPE_USER_MAPPING,
            IMPORT_LEGACY_ROWS,
            RECOLLECT,
            STAMP_COLLECTIONS,
        ],
        "the plan keeps the dependency order of the four migrator families"
    );

    // Legacy variable rows moved into the current prefixes and left nothing behind.
    let scope = scope();
    assert_eq!(
        kv_text(&stores, &keys::variable_key(&scope, "theme"))
            .await
            .as_deref(),
        Some("dark")
    );
    assert_eq!(
        kv_text(&stores, &keys::session_variable_key(&scope, "s1", "step"))
            .await
            .as_deref(),
        Some("2")
    );
    assert_eq!(
        kv_keys(&stores, keys::LEGACY_VARIABLE_PREFIX).await,
        Vec::<String>::new(),
        "no legacy user-variable key survives"
    );
    assert_eq!(
        kv_keys(&stores, keys::LEGACY_SESSION_VARIABLE_PREFIX).await,
        Vec::<String>::new(),
        "no legacy session-variable key survives"
    );

    // The legacy record key is read through, not rewritten, but the backfill derived the
    // scope-user pair from it.
    assert!(
        kv_text(&stores, &keys::legacy_record_key(&scope, RECORD_ID))
            .await
            .is_some(),
        "record keys are a read-only legacy layout, not a copied family"
    );
    let facade = LongTermMemory::with_stores(config(), stores.clone())
        .await
        .expect("facade over the migrated stores");
    let restored = facade
        .get(&scope, RECORD_ID)
        .await
        .expect("read the legacy record")
        .expect("the legacy record stays readable through the facade");
    assert_eq!(restored.content, RECORD_TEXT);
    let mapping = ScopeUserMappingStore::from_stores(&stores);
    assert!(mapping
        .contains("alice", "ws")
        .await
        .expect("mapping lookup"));
    assert_eq!(
        mapping.users_for_scope("ws").await.expect("users"),
        vec!["alice".to_owned()],
        "the record keyspace backfilled exactly one pair"
    );

    // The legacy message row became a document and its key is gone.
    let document = stores
        .document
        .get(keys::MESSAGES_COLLECTION, "m1")
        .await
        .expect("read message")
        .expect("the legacy row was imported");
    let value = document.to_value();
    assert_eq!(value["content"], serde_json::json!("hello"));
    assert_eq!(value["role"], serde_json::json!("user"));
    assert_eq!(
        kv_keys(&stores, LEGACY_MESSAGE_PREFIXES[0]).await,
        Vec::<String>::new(),
        "no legacy message row survives"
    );

    // Embeddings were recollected into the canonical collection and the short spelling is
    // gone; the unrelated memory collection was left alone.
    assert!(
        !stores
            .vector
            .collection_exists("uid_alice_gid_ws_mtype_semantic")
            .await
            .expect("collection existence"),
        "the legacy collection is recollected and removed"
    );
    assert_eq!(
        collection_ids(&stores, &canonical).await,
        vec!["v1".to_owned(), "v2".to_owned()]
    );
    assert_eq!(
        collection_ids(&stores, "uid_bob_gid_ws_mtype_coding").await,
        vec!["c1".to_owned()]
    );

    // Both surviving collections carry a rebuild-requesting stamp at the pinned time.
    for collection in [canonical.as_str(), "uid_bob_gid_ws_mtype_coding"] {
        let stamp = IndexStamp::load(stores.kv.as_ref(), collection)
            .await
            .expect("load stamp")
            .expect("collection stamped");
        assert_eq!(stamp.version, CURRENT_INDEX_VERSION);
        assert!(stamp.needs_rebuild);
        assert_eq!(stamp.dimensions, Some(3));
        assert_eq!(stamp.updated_at, Some(fixed(0)));
    }
    assert!(
        IndexStamp::load(stores.kv.as_ref(), "uid_alice_gid_ws_mtype_semantic")
            .await
            .expect("load stamp")
            .is_none()
    );
}

#[tokio::test]
async fn a_second_run_at_the_same_reference_time_is_a_no_op() {
    let stores = MemoryStores::in_memory();
    let scope = scope();
    seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
    seed(
        &stores,
        &keys::legacy_session_variable_key(&scope, "s1", "step"),
        "2",
    )
    .await;
    let record = MemoryRecord::new(scope.clone(), RECORD_TEXT, MemoryType::SemanticMemory)
        .with_id(RECORD_ID)
        .with_timestamp(fixed(0));
    seed(
        &stores,
        &keys::legacy_record_key(&scope, RECORD_ID),
        &serde_json::to_string(&record).expect("record JSON"),
    )
    .await;

    let first = run_migrations(&context(&stores)).await.expect("first run");
    assert!(first.is_success());
    assert_eq!(first.plan.operation_ids().len(), 3);

    let second_context = context(&stores);
    let plan = plan_migrations(&second_context).await.expect("second plan");
    assert!(
        plan.is_empty(),
        "nothing is left to plan, got {:?}",
        plan.operation_ids()
    );
    let second = run_migrations(&second_context).await.expect("second run");
    assert!(second.is_success());
    assert_eq!(second.reports, Vec::new());
    assert!(second.plan.is_empty());
    assert!(second.journal.is_empty());
    assert_eq!(second.executed(), 0);
    assert_eq!(second.counts().affected, 0);

    // Nothing was moved or deleted by the rerun, and the mapping did not grow.
    assert_eq!(
        kv_text(&stores, &keys::variable_key(&scope, "theme"))
            .await
            .as_deref(),
        Some("dark")
    );
    assert_eq!(
        kv_text(&stores, &keys::session_variable_key(&scope, "s1", "step"))
            .await
            .as_deref(),
        Some("2")
    );
    assert_eq!(
        kv_keys(&stores, keys::LEGACY_VARIABLE_PREFIX).await,
        Vec::<String>::new()
    );
    assert_eq!(
        kv_text(&stores, &keys::legacy_record_key(&scope, RECORD_ID)).await,
        Some(serde_json::to_string(&record).expect("record JSON")),
        "the legacy record row is untouched by a rerun"
    );
    let mapping = ScopeUserMappingStore::from_stores(&stores);
    assert_eq!(mapping.len().await.expect("mapping rows"), 1);
}

// ---------------------------------------------------------------------------
// Scenario 3: journal-keyed resume after a handler failure.
// ---------------------------------------------------------------------------

/// A handler that counts its executions and records them in the key-value store.
struct CountingHandler {
    /// The id this handler answers to.
    operation_id: &'static str,
    /// Executions, shared with the test so it can assert the handler did not rerun.
    runs: Arc<AtomicUsize>,
}

#[async_trait]
impl MigrationOperationHandler for CountingHandler {
    fn operation_id(&self) -> &'static str {
        self.operation_id
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let run = self.runs.fetch_add(1, Ordering::SeqCst) + 1;
        context
            .kv()
            .set(
                &format!("custom/{}/runs", self.operation_id),
                KvValue::text(run.to_string()),
            )
            .await?;
        Ok(OperationReport::succeeded(self.operation_id, 1, 0, 0))
    }
}

/// A handler that fails on its first attempt and succeeds on every retry.
struct FailOnceHandler {
    /// The id this handler answers to.
    operation_id: &'static str,
    /// Attempts, shared with the test so it can assert the retry happened.
    attempts: Arc<AtomicUsize>,
}

#[async_trait]
impl MigrationOperationHandler for FailOnceHandler {
    fn operation_id(&self) -> &'static str {
        self.operation_id
    }

    async fn execute(
        &self,
        _context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt == 1 {
            return Err(Error::memory("simulated migration failure"));
        }
        Ok(OperationReport::succeeded(self.operation_id, 1, 0, 0))
    }
}

#[tokio::test]
async fn a_journal_resumes_after_a_failed_operation() {
    let stores = MemoryStores::in_memory();
    let context = context(&stores);
    let completed_runs = Arc::new(AtomicUsize::new(0));
    let failed_attempts = Arc::new(AtomicUsize::new(0));
    let first: Arc<dyn MigrationOperationHandler> = Arc::new(CountingHandler {
        operation_id: "custom.first",
        runs: Arc::clone(&completed_runs),
    });
    let second: Arc<dyn MigrationOperationHandler> = Arc::new(FailOnceHandler {
        operation_id: "custom.second",
        attempts: Arc::clone(&failed_attempts),
    });
    let registry = OperationRegistry::new()
        .with_handler(first)
        .expect("register the first handler")
        .with_handler(second)
        .expect("register the second handler");
    assert_eq!(
        registry.operation_ids(),
        vec!["custom.first", "custom.second"]
    );
    let plan = MigrationPlan::new(vec![
        MigrationStep::new("custom.first", "custom", 1, "record the first operation"),
        MigrationStep::new("custom.second", "custom", 1, "fail once, then succeed"),
    ]);

    let mut journal = MigrationJournal::new();
    let error = registry
        .execute(&context, &plan, &mut journal)
        .await
        .expect_err("the first run stops on the failing handler");
    assert_eq!(error.kind(), ErrorKind::Memory, "{error}");
    assert!(error.message().contains("custom.second"), "{error}");

    let completed = journal.get("custom.first").expect("first journalled");
    assert_eq!(completed.outcome, OperationOutcome::Succeeded);
    assert_eq!(completed.attempts, 1);
    assert!(completed.succeeded());
    let failed = journal.get("custom.second").expect("second journalled");
    assert_eq!(failed.outcome, OperationOutcome::Failed);
    assert_eq!(failed.attempts, 1);
    assert_eq!(failed.failed, 1);
    assert!(
        failed
            .error
            .as_deref()
            .is_some_and(|error| error.contains("simulated migration failure")),
        "{failed:?}"
    );
    assert!(failed.retryable());
    assert_eq!(journal.completed(), vec!["custom.first".to_owned()]);
    assert_eq!(journal.failed().len(), 1);
    assert_eq!(completed_runs.load(Ordering::SeqCst), 1);

    // The same journal continues: the completed operation is skipped untouched and only the
    // failed one is retried.
    let resumed = registry
        .execute(&context, &plan, &mut journal)
        .await
        .expect("the resumed run finishes");
    assert!(resumed.is_success(), "{:?}", resumed.reports);
    assert_eq!(resumed.reports.len(), 2);
    assert_eq!(resumed.reports[0].outcome, OperationOutcome::Skipped);
    assert_eq!(
        resumed.reports[0].note.as_deref(),
        Some("already completed")
    );
    assert_eq!(resumed.reports[0].operation_id, "custom.first");
    assert_eq!(resumed.reports[1].outcome, OperationOutcome::Succeeded);
    assert_eq!(resumed.reports[1].operation_id, "custom.second");
    assert_eq!(
        completed_runs.load(Ordering::SeqCst),
        1,
        "an operation the journal records as succeeded is never rerun"
    );
    assert_eq!(failed_attempts.load(Ordering::SeqCst), 2);
    let retried = journal.get("custom.second").expect("second journalled");
    assert_eq!(retried.outcome, OperationOutcome::Succeeded);
    assert_eq!(retried.attempts, 2, "the attempt counter grows on retry");
    assert!(retried.error.is_none());
    assert_eq!(
        kv_text(&stores, "custom/custom.first/runs")
            .await
            .as_deref(),
        Some("1"),
        "the completed handler ran exactly once"
    );
}

// ---------------------------------------------------------------------------
// Scenario 4: dry runs create and delete nothing while reporting the work.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_dry_run_reports_the_work_without_creating_or_deleting_anything() {
    let stores = MemoryStores::in_memory();
    let scope = scope();
    seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
    seed(
        &stores,
        &keys::legacy_session_variable_key(&scope, "s1", "step"),
        "2",
    )
    .await;
    let context = context(&stores).with_dry_run(true);

    let plan = plan_migrations(&context).await.expect("dry plan");
    assert!(
        plan.dry_run,
        "the context's dry run is mirrored on the plan"
    );
    assert_eq!(
        plan.operation_ids(),
        vec![COPY_USER_VARIABLES, COPY_SESSION_VARIABLES]
    );
    assert_eq!(
        plan.counts().affected,
        2,
        "the plan reports the rows a real run would copy"
    );

    let run = run_migrations(&context).await.expect("dry run");
    assert!(run.is_success());
    assert_eq!(run.reports.len(), 2);
    for report in &run.reports {
        assert_eq!(report.outcome, OperationOutcome::Skipped);
        assert_eq!(report.note.as_deref(), Some("dry run"));
        assert_eq!(report.affected, 0, "no handler ran, so nothing was counted");
    }
    assert_eq!(run.journal.len(), 2);
    assert!(run
        .journal
        .records()
        .all(|record| record.outcome == OperationOutcome::Skipped));
    assert_eq!(
        run.journal.completed(),
        Vec::<String>::new(),
        "a dry run completes no operation"
    );

    // Nothing was created and nothing was deleted.
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
    assert!(
        kv_text(&stores, &keys::session_variable_key(&scope, "s1", "step"))
            .await
            .is_none()
    );

    // A real run over the same stores then converts exactly what the dry plan advertised.
    let real = run_migrations(&context.with_dry_run(false))
        .await
        .expect("real run");
    assert!(real.is_success());
    assert!(real
        .reports
        .iter()
        .all(|report| report.outcome == OperationOutcome::Succeeded));
    assert_eq!(real.counts().affected, plan.counts().affected);
    assert_eq!(
        kv_text(&stores, &keys::variable_key(&scope, "theme"))
            .await
            .as_deref(),
        Some("dark")
    );
    assert_eq!(
        kv_keys(&stores, keys::LEGACY_VARIABLE_PREFIX).await,
        Vec::<String>::new()
    );
}

// ---------------------------------------------------------------------------
// Scenario 5: backup and restore parity.
// ---------------------------------------------------------------------------

/// Writes the backup fixture through the facade and returns the record ids it created.
async fn seed_fixture(stores: &MemoryStores) -> Vec<String> {
    let facade = LongTermMemory::with_stores(config(), stores.clone())
        .await
        .expect("facade");
    let mut ids = Vec::new();
    for (content, offset) in CONTENTS.iter().zip([0_i64, 1]) {
        let memory = NewMemory::semantic(*content).with_timestamp(fixed(offset));
        let outcome = facade
            .add(&scope(), memory, &AddOptions::default())
            .await
            .expect("add record");
        assert_eq!(outcome.added.len(), 1);
        ids.push(outcome.added[0].id.clone());
    }
    facade
        .set_variable(&scope(), "theme", "dark")
        .await
        .expect("user variable");
    facade
        .set_variable(&session_scope(), "step", "2")
        .await
        .expect("session variable");
    facade
        .set_scope_config("ws", scope_config())
        .await
        .expect("scope config");
    ids
}

/// Asserts records, variables, and the scope config are readable through a facade.
async fn assert_fixture(stores: &MemoryStores, ids: &[String]) {
    let facade = LongTermMemory::with_stores(config(), stores.clone())
        .await
        .expect("facade");
    for (id, content) in ids.iter().zip(CONTENTS) {
        let record = facade
            .get(&scope(), id)
            .await
            .expect("read record")
            .expect("restored record exists");
        assert_eq!(record.content, content);
        assert_eq!(record.id, *id);
    }
    assert_eq!(
        facade
            .get_variables(&scope(), None)
            .await
            .expect("variables"),
        BTreeMap::from([("theme".to_owned(), "dark".to_owned())])
    );
    assert_eq!(
        facade
            .get_variables(&session_scope(), None)
            .await
            .expect("session variables"),
        BTreeMap::from([("step".to_owned(), "2".to_owned())])
    );
    assert_eq!(
        facade.get_scope_config("ws").await.expect("scope config"),
        Some(scope_config())
    );
}

#[tokio::test]
async fn in_memory_backup_and_restore_preserve_records_variables_and_configs() {
    let stores = MemoryStores::in_memory();
    let ids = seed_fixture(&stores).await;

    let backup = backup::dump(&stores).await.expect("dump");
    assert_eq!(backup.len(), 5, "{:?}", backup.entries.keys());
    assert_eq!(
        backup.counts(),
        backup::BackupCounts {
            records: 2,
            user_variables: 1,
            session_variables: 1,
            scope_configs: 1,
            total: 5,
        }
    );

    // Mutate the source after the dump: a restore must bring the dumped values back.
    let facade = LongTermMemory::with_stores(config(), stores.clone())
        .await
        .expect("facade");
    assert!(facade
        .delete(&scope(), &ids[0])
        .await
        .expect("delete record"));
    facade
        .set_variable(&scope(), "theme", "light")
        .await
        .expect("overwrite the variable");
    assert!(facade
        .delete_scope_config("ws")
        .await
        .expect("delete the scope config"));

    let fresh = MemoryStores::in_memory();
    let report = backup::restore(&fresh, &backup).await.expect("restore");
    assert!(!report.dry_run);
    assert_eq!(report.total, backup.len());
    assert_eq!(report.written, backup.len());
    assert_eq!(report.skipped, 0);
    assert_eq!(report.processed(), report.total);
    assert_eq!(report.counts, backup.counts());
    assert_fixture(&fresh, &ids).await;

    // Restoring the same backup again is additive: every entry already matches.
    let again = backup::restore(&fresh, &backup)
        .await
        .expect("second restore");
    assert_eq!(again.written, 0);
    assert_eq!(again.skipped, backup.len());
    assert_eq!(again.counts.total, 0);

    // A dry run reports the same work and writes nothing.
    let dry_target = MemoryStores::in_memory();
    let dry = backup::restore_with(&dry_target, &backup, &KvPrefixRegistry::default(), true)
        .await
        .expect("dry restore");
    assert!(dry.dry_run);
    assert_eq!(dry.written, backup.len());
    assert_eq!(dry.skipped, 0);
    assert_eq!(dry.counts, backup.counts());
    let untouched = backup::dump(&dry_target)
        .await
        .expect("dump the dry target");
    assert!(
        untouched.is_empty(),
        "a dry-run restore writes nothing, got {:?}",
        untouched.entries.keys()
    );
}

#[tokio::test]
async fn local_file_backup_restore_into_a_fresh_store_survives_a_reopen() {
    let source_dir = tempfile::tempdir().expect("source dir");
    let target_dir = tempfile::tempdir().expect("target dir");
    let ids = {
        let stores = MemoryStores::local_file(source_dir.path())
            .await
            .expect("source stores");
        let ids = seed_fixture(&stores).await;
        assert_fixture(&stores, &ids).await;
        let backup = backup::dump(&stores).await.expect("dump");
        assert_eq!(backup.len(), 5);

        // Delete one record and overwrite the variable before restoring.
        let facade = LongTermMemory::with_stores(config(), stores.clone())
            .await
            .expect("facade");
        assert!(facade.delete(&scope(), &ids[1]).await.expect("delete"));
        facade
            .set_variable(&scope(), "theme", "light")
            .await
            .expect("overwrite");

        let target = MemoryStores::local_file(target_dir.path())
            .await
            .expect("target stores");
        let report = backup::restore(&target, &backup).await.expect("restore");
        assert_eq!(report.written, 5);
        assert_eq!(report.skipped, 0);
        assert_fixture(&target, &ids).await;
        ids
    };

    // The restored rows are durable: reopening the target directory sees them.
    let reopened = MemoryStores::local_file(target_dir.path())
        .await
        .expect("reopen the target");
    assert_fixture(&reopened, &ids).await;
}

#[tokio::test]
async fn backup_verification_and_json_round_trips() {
    let stores = MemoryStores::in_memory();
    let ids = seed_fixture(&stores).await;
    let backup = backup::dump(&stores).await.expect("dump");

    let text = backup.to_json().expect("serialize the backup");
    let decoded = MemoryBackup::from_json(&text).expect("decode the backup");
    assert_eq!(decoded, backup, "to_json/from_json round-trips exactly");
    assert_eq!(decoded.counts(), backup.counts());
    backup.verify().expect("a fresh dump verifies");

    // A foreign format marker is rejected and named.
    let mut foreign = backup.clone();
    foreign.format = "another.subsystem.backup".to_owned();
    let error = foreign.verify().expect_err("a foreign format is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(
        error.message().contains("another.subsystem.backup"),
        "{error}"
    );
    let foreign_json = serde_json::to_string(&foreign).expect("serialize the foreign document");
    let error = MemoryBackup::from_json(&foreign_json).expect_err("from_json verifies");
    assert!(matches!(error, Error::Validation(_)), "{error}");

    // Declared counts that disagree with the entries are rejected.
    let mut mismatch = backup.clone();
    mismatch.counts.records += 1;
    mismatch.counts.total += 1;
    let error = mismatch.verify().expect_err("a count mismatch is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(error.message().contains("count mismatch"), "{error}");

    // A newer format version is a typed `Unsupported`, not a silent downgrade.
    let mut newer = backup.clone();
    newer.version = BACKUP_FORMAT_VERSION + 1;
    let error = newer.verify().expect_err("a newer format is unsupported");
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    assert!(error.message().contains("format version"), "{error}");

    // The dump covered every fixture row, so a restore is a full round trip.
    assert_eq!(backup.counts().records, ids.len());
}

// ---------------------------------------------------------------------------
// Scenario 6: prefix registry switching.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_family_rename_survives_a_backup_round_trip() {
    let stores = MemoryStores::in_memory();
    let ids = seed_fixture(&stores).await;

    // `with_current`/`register_current` demote the old prefix to an alias.
    let renamed = KvPrefixRegistry::default()
        .with_current(KvFamily::UserVariables, "vars_v2")
        .expect("rename the user-variables family");
    assert_eq!(renamed.current(KvFamily::UserVariables), "vars_v2");
    assert_eq!(
        renamed.aliases(KvFamily::UserVariables),
        vec!["user_var".to_owned(), "user_variable".to_owned()],
        "the old prefix stays readable as an alias"
    );
    assert!(renamed.contains_prefix(KvFamily::UserVariables, "user_var"));
    assert!(renamed.contains_prefix(KvFamily::UserVariables, "user_variable"));

    // The dump rewrites the alias rows onto the renamed current prefix.
    let dump = backup::dump_with(&stores, &renamed)
        .await
        .expect("renamed dump");
    assert_eq!(dump.len(), 5);
    assert_eq!(
        dump.prefixes.get("user_variables").map(String::as_str),
        Some("vars_v2")
    );
    assert_eq!(dump.counts().user_variables, 1);
    assert!(
        dump.entries.contains_key("vars_v2/alice/ws/theme"),
        "{:?}",
        dump.entries.keys().collect::<Vec<_>>()
    );
    assert!(!dump.entries.contains_key("user_var/alice/ws/theme"));

    // Restoring into a default-layout store brings the rows back under the default prefix.
    let fresh = MemoryStores::in_memory();
    let report = backup::restore(&fresh, &dump).await.expect("restore");
    assert_eq!(report.written, 5);
    let facade = LongTermMemory::with_stores(config(), fresh.clone())
        .await
        .expect("facade");
    assert_eq!(
        facade
            .get_variables(&scope(), None)
            .await
            .expect("variables"),
        BTreeMap::from([("theme".to_owned(), "dark".to_owned())]),
        "a family rename survives the round trip"
    );
    assert_eq!(
        kv_text(&fresh, &keys::variable_key(&scope(), "theme"))
            .await
            .as_deref(),
        Some("dark")
    );
    assert!(kv_text(&fresh, "vars_v2/alice/ws/theme").await.is_none());

    // Restoring under the renamed layout keeps the renamed prefix, and the other families
    // are untouched by it.
    let renamed_target = MemoryStores::in_memory();
    let renamed_report = backup::restore_with(&renamed_target, &dump, &renamed, false)
        .await
        .expect("renamed restore");
    assert_eq!(renamed_report.written, 5);
    assert_eq!(
        kv_text(&renamed_target, "vars_v2/alice/ws/theme")
            .await
            .as_deref(),
        Some("dark")
    );
    assert!(
        kv_text(&renamed_target, &keys::variable_key(&scope(), "theme"))
            .await
            .is_none()
    );
    let renamed_facade = LongTermMemory::with_stores(config(), renamed_target.clone())
        .await
        .expect("facade over the renamed target");
    for (id, content) in ids.iter().zip(CONTENTS) {
        let record = renamed_facade
            .get(&scope(), id)
            .await
            .expect("read record")
            .expect("record restored");
        assert_eq!(record.content, content);
    }
    assert_eq!(
        renamed_facade.get_scope_config("ws").await.expect("config"),
        Some(scope_config())
    );
}

#[test]
fn prefix_registration_keeps_demoted_prefixes_readable_and_rejects_bad_segments() {
    let mut registry = KvPrefixRegistry::default();
    assert_eq!(registry.current(KvFamily::Records), keys::RECORD_PREFIX);
    assert_eq!(
        registry.aliases(KvFamily::Records),
        vec![keys::LEGACY_RECORD_PREFIX.to_owned()]
    );

    registry
        .register_current(KvFamily::Records, "records_v2")
        .expect("rename the records family");
    assert_eq!(registry.current(KvFamily::Records), "records_v2");
    assert_eq!(
        registry.aliases(KvFamily::Records),
        vec![
            keys::RECORD_PREFIX.to_owned(),
            keys::LEGACY_RECORD_PREFIX.to_owned()
        ],
        "the demoted prefix sits first among the aliases"
    );
    assert!(registry.contains_prefix(KvFamily::Records, keys::RECORD_PREFIX));

    // Switching back to the documented prefix drops it from the alias list again.
    registry
        .register_current(KvFamily::Records, keys::RECORD_PREFIX)
        .expect("restore the documented prefix");
    assert_eq!(registry.current(KvFamily::Records), keys::RECORD_PREFIX);
    assert_eq!(
        registry.aliases(KvFamily::Records),
        vec![
            "records_v2".to_owned(),
            keys::LEGACY_RECORD_PREFIX.to_owned()
        ]
    );

    // `register_legacy` appends an alias once and ignores the current prefix.
    registry
        .register_legacy(KvFamily::Records, "records_v2")
        .expect("an existing alias is a no-op");
    assert_eq!(
        registry.aliases(KvFamily::Records),
        vec![
            "records_v2".to_owned(),
            keys::LEGACY_RECORD_PREFIX.to_owned()
        ]
    );
    registry
        .register_legacy(KvFamily::Records, keys::RECORD_PREFIX)
        .expect("the current prefix can never be an alias");
    assert!(!registry
        .aliases(KvFamily::Records)
        .iter()
        .any(|alias| alias == keys::RECORD_PREFIX));

    // Every invalid segment is a validation error naming the family and the value.
    let long = "p".repeat(129);
    for invalid in [
        "",
        "   ",
        "a/b",
        "back\\slash",
        "nul\0segment",
        long.as_str(),
    ] {
        let error = KvPrefixRegistry::default()
            .with_current(KvFamily::UserVariables, invalid)
            .expect_err("an invalid current prefix is rejected");
        assert!(
            matches!(error, Error::Validation(_)),
            "{invalid:?}: {error}"
        );
        assert!(
            error.message().contains("user_variables"),
            "{invalid:?}: {error}"
        );

        let mut registry = KvPrefixRegistry::default();
        let error = registry
            .register_legacy(KvFamily::ScopeConfig, invalid)
            .expect_err("an invalid alias is rejected");
        assert!(
            matches!(error, Error::Validation(_)),
            "{invalid:?}: {error}"
        );
        assert!(
            error.message().contains("scope_config"),
            "{invalid:?}: {error}"
        );
        assert_eq!(
            registry.aliases(KvFamily::ScopeConfig),
            vec![keys::LEGACY_SCOPE_CONFIG_PREFIX.to_owned()],
            "a rejected registration leaves the registry unchanged"
        );
    }
}

// ---------------------------------------------------------------------------
// Scenario 7: the distributed KV lock seam.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_process_locks_exclude_two_handles_and_time_out() {
    let locks = InProcessKvLock::new();
    let handle_a = locks.clone();
    let handle_b = locks.clone();
    let guard = handle_a
        .acquire("migrate", &LockOptions::new("run-a"))
        .await
        .expect("handle A acquires");
    assert!(guard.is_held());
    assert_eq!(guard.backend(), "in-process");
    assert_eq!(guard.name(), "migrate");
    assert_eq!(handle_b.held_names(), vec!["migrate".to_owned()]);

    let error = handle_b
        .acquire(
            "migrate",
            &LockOptions::new("run-b")
                .with_timeout(Duration::from_millis(50))
                .with_poll_interval(Duration::from_millis(5)),
        )
        .await
        .expect_err("handle B is excluded while A holds the name");
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert!(error.message().contains("migrate"), "{error}");
    assert!(error.message().contains("in-process"), "{error}");

    // Dropping the guard releases the name synchronously.
    drop(guard);
    assert_eq!(handle_b.held_names(), Vec::<String>::new());
    let second = handle_b
        .acquire(
            "migrate",
            &LockOptions::new("run-b").with_timeout(Duration::from_millis(250)),
        )
        .await
        .expect("handle B acquires once A released");
    assert!(second.is_held());
    assert_eq!(second.owner(), "run-b");
    second.release();
    assert_eq!(handle_b.held_names(), Vec::<String>::new());

    // Unusable names and owners are validation errors, not panics.
    let error = handle_a
        .acquire("migrate/run", &LockOptions::new("run-a"))
        .await
        .expect_err("a name with a path separator is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    let error = handle_a
        .acquire("migrate", &LockOptions::new("   "))
        .await
        .expect_err("a blank owner is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    let error = handle_a
        .acquire(
            "migrate",
            &LockOptions::new("run-a").with_timeout(Duration::ZERO),
        )
        .await
        .expect_err("a zero timeout is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(
        handle_a.held_names().is_empty(),
        "no failed acquire holds a name"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn file_locks_exclude_two_handles_on_one_root() {
    let directory = tempfile::tempdir().expect("lock root");
    let root = directory.path().join("locks");
    let handle_a = FileKvLock::new(&root).expect("handle A");
    assert!(handle_a.is_enabled(), "unix takes a real advisory flock");
    assert_eq!(handle_a.root(), root.as_path());
    assert_eq!(handle_a.lock_path("migrate"), root.join("migrate.lock"));
    let handle_b = FileKvLock::new(&root)
        .expect("handle B")
        .with_timeout(Some(Duration::from_millis(50)));

    let guard = handle_a
        .acquire("migrate", &LockOptions::new("run-a"))
        .await
        .expect("handle A acquires");
    assert_eq!(guard.backend(), "file");
    assert!(
        handle_a.sidecar_path("migrate").is_file(),
        "the advisory sidecar exists while the lock is held"
    );

    let error = handle_b
        .acquire("migrate", &LockOptions::new("run-b"))
        .await
        .expect_err("handle B is excluded by the flock");
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert!(error.message().contains("migrate"), "{error}");

    drop(guard);
    let second = handle_b
        .acquire("migrate", &LockOptions::new("run-b"))
        .await
        .expect("handle B acquires once the flock was released");
    assert!(second.is_held());

    // A different root never contends.
    let other_directory = tempfile::tempdir().expect("second lock root");
    let elsewhere = FileKvLock::new(other_directory.path().join("locks"))
        .expect("third handle")
        .with_timeout(Some(Duration::from_millis(50)));
    let third = elsewhere
        .acquire("migrate", &LockOptions::new("run-c"))
        .await
        .expect("a different root does not contend");
    assert!(third.is_held());

    // Validation applies to the file backend too, and never touches the filesystem first.
    let error = elsewhere
        .acquire("migrate", &LockOptions::new(""))
        .await
        .expect_err("a blank owner is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    let error = elsewhere
        .acquire("migrate/run", &LockOptions::new("run-c"))
        .await
        .expect_err("a name with a path separator is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
}

#[tokio::test]
async fn remote_lock_brokers_are_typed_unsupported() {
    let lock = UnsupportedKvLock::redis();
    assert_eq!(lock.backend(), "redis");
    let error = lock
        .acquire("migrate", &LockOptions::new("run-a"))
        .await
        .expect_err("the redis broker is deferred");
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    assert!(error.message().contains("redis"), "{error}");
    assert!(error.message().contains("multi-node"), "{error}");
    assert!(
        error.message().contains("InProcessKvLock"),
        "the error points at the local alternatives: {error}"
    );
}

// ---------------------------------------------------------------------------
// Scenario 8: the scope-user mapping store.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scope_user_mapping_crud_ordering_and_idempotent_add() {
    let stores = MemoryStores::in_memory();
    let mapping = ScopeUserMappingStore::from_stores(&stores);
    assert!(mapping.is_empty().await.expect("empty"));
    assert_eq!(mapping.len().await.expect("len"), 0);

    // `add` reports whether the row was inserted, so the second call is `false`.
    assert!(mapping.add("alice", "ws").await.expect("add alice/ws"));
    assert!(!mapping
        .add("alice", "ws")
        .await
        .expect("add alice/ws twice"));
    assert!(mapping.add("bob", "ws").await.expect("add bob/ws"));
    assert!(mapping.add("alice", "ws2").await.expect("add alice/ws2"));
    assert_eq!(mapping.len().await.expect("len"), 3);
    assert!(!mapping.is_empty().await.expect("not empty"));

    assert!(mapping.contains("alice", "ws").await.expect("contains"));
    assert!(!mapping.contains("carol", "ws").await.expect("absent user"));
    let row = mapping
        .get("alice", "ws2")
        .await
        .expect("get")
        .expect("row exists");
    assert_eq!(row.user_id, "alice");
    assert_eq!(row.scope_id, "ws2");
    assert_eq!(mapping.get("carol", "ws").await.expect("absent"), None);

    assert_eq!(
        mapping.users_for_scope("ws").await.expect("users"),
        vec!["alice".to_owned(), "bob".to_owned()]
    );
    assert_eq!(
        mapping.users_for_scope("ws2").await.expect("users"),
        vec!["alice".to_owned()]
    );
    assert_eq!(
        mapping
            .users_for_scope("other")
            .await
            .expect("unknown scope"),
        Vec::<String>::new()
    );
    assert_eq!(
        mapping.scopes_for_user("alice").await.expect("scopes"),
        vec!["ws".to_owned(), "ws2".to_owned()]
    );
    let listed = mapping.list().await.expect("list");
    assert_eq!(
        listed
            .iter()
            .map(|row| (row.scope_id.as_str(), row.user_id.as_str()))
            .collect::<Vec<_>>(),
        vec![("ws", "alice"), ("ws", "bob"), ("ws2", "alice")],
        "the listing is sorted by (scope_id, user_id)"
    );

    // The row is the documented JSON shape under `scope_user_mapping/{scope}/{user}`.
    assert_eq!(
        kv_text(&stores, "scope_user_mapping/ws/alice")
            .await
            .as_deref(),
        Some(r#"{"user_id":"alice","scope_id":"ws"}"#)
    );

    assert!(mapping.delete("bob", "ws").await.expect("delete"));
    assert!(!mapping.delete("bob", "ws").await.expect("delete twice"));
    assert_eq!(
        mapping.users_for_scope("ws").await.expect("users"),
        vec!["alice".to_owned()]
    );
    assert_eq!(mapping.delete_scope("ws2").await.expect("delete scope"), 1);
    assert_eq!(mapping.delete_scope("ws2").await.expect("delete twice"), 0);
    assert_eq!(
        mapping.scopes_for_user("alice").await.expect("scopes"),
        vec!["ws".to_owned()]
    );
    assert_eq!(mapping.delete_user("alice").await.expect("delete user"), 1);
    assert_eq!(mapping.delete_user("alice").await.expect("delete twice"), 0);
    assert!(mapping.is_empty().await.expect("empty again"));
}

#[tokio::test]
async fn scope_user_mapping_persists_on_local_file() {
    let directory = tempfile::tempdir().expect("mapping root");
    {
        let stores = MemoryStores::local_file(directory.path())
            .await
            .expect("open local-file stores");
        let mapping = ScopeUserMappingStore::from_stores(&stores);
        assert!(mapping.add("alice", "ws").await.expect("add"));
        assert!(mapping.add("bob", "ws2").await.expect("add"));
    }
    let reopened = MemoryStores::local_file(directory.path())
        .await
        .expect("reopen local-file stores");
    let mapping = ScopeUserMappingStore::from_stores(&reopened);
    assert_eq!(mapping.len().await.expect("len"), 2);
    assert_eq!(
        mapping.users_for_scope("ws").await.expect("users"),
        vec!["alice".to_owned()]
    );
    assert_eq!(
        mapping.scopes_for_user("bob").await.expect("scopes"),
        vec!["ws2".to_owned()]
    );
    assert!(
        !mapping.add("alice", "ws").await.expect("add again"),
        "an existing row stays idempotent after a reopen"
    );
    assert_eq!(mapping.len().await.expect("len"), 2);
}

#[tokio::test]
async fn scope_user_mapping_rejects_invalid_ids_without_writing() {
    let stores = MemoryStores::in_memory();
    let mapping = ScopeUserMappingStore::from_stores(&stores);
    let long = "x".repeat(129);
    let invalid = [
        "",
        "   ",
        "a/b",
        "back\\slash",
        "nul\0id",
        " padded",
        long.as_str(),
    ];

    for id in invalid {
        let error = mapping
            .add(id, "ws")
            .await
            .expect_err("an invalid user id is rejected");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        assert!(error.message().contains("user_id"), "{id:?}: {error}");

        let error = mapping
            .add("alice", id)
            .await
            .expect_err("an invalid scope id is rejected");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        assert!(error.message().contains("scope_id"), "{id:?}: {error}");
    }
    assert!(
        mapping.is_empty().await.expect("empty"),
        "a rejected add writes nothing"
    );

    // The reads and deletes validate the ids too, so nothing is silently matched.
    for id in invalid {
        let error = mapping
            .contains(id, "ws")
            .await
            .expect_err("contains validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        let error = mapping.get("alice", id).await.expect_err("get validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        let error = mapping
            .delete("alice", id)
            .await
            .expect_err("delete validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        let error = mapping
            .users_for_scope(id)
            .await
            .expect_err("users_for_scope validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        let error = mapping
            .scopes_for_user(id)
            .await
            .expect_err("scopes_for_user validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        let error = mapping
            .delete_scope(id)
            .await
            .expect_err("delete_scope validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
        let error = mapping
            .delete_user(id)
            .await
            .expect_err("delete_user validates");
        assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
    }
    assert_eq!(mapping.len().await.expect("len"), 0);
}

// ---------------------------------------------------------------------------
// Scenario 5 + 1 on the SQLite backend (feature `sqlite`).
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn the_sqlite_backend_migrates_and_restores_a_backup() {
    let directory = tempfile::tempdir().expect("sqlite dir");
    let source_path = directory.path().join("source.db");
    let target_path = directory.path().join("target.db");
    let scope = scope();
    let ids = {
        let stores = MemoryStores::sqlite(&source_path)
            .await
            .expect("sqlite source stores");
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        seed(
            &stores,
            &keys::legacy_session_variable_key(&scope, "s1", "step"),
            "2",
        )
        .await;
        let ids = seed_fixture(&stores).await;
        let run = run_migrations(&context(&stores))
            .await
            .expect("standard run");
        assert!(run.is_success(), "{:?}", run.reports);

        // The variable rows moved into the current prefixes and the facade sees them.
        assert_eq!(
            kv_keys(&stores, keys::LEGACY_VARIABLE_PREFIX).await,
            Vec::<String>::new(),
            "no legacy user-variable key survives"
        );
        let facade = LongTermMemory::with_stores(config(), stores.clone())
            .await
            .expect("facade");
        assert_eq!(
            facade.get_variables(&scope, None).await.expect("variables"),
            BTreeMap::from([("theme".to_owned(), "dark".to_owned())])
        );
        assert_eq!(
            facade
                .get_variables(&session_scope(), None)
                .await
                .expect("session variables"),
            BTreeMap::from([("step".to_owned(), "2".to_owned())])
        );

        let dump = backup::dump(&stores).await.expect("dump");
        assert_eq!(dump.len(), 5, "{:?}", dump.entries.keys());

        let target = MemoryStores::sqlite(&target_path)
            .await
            .expect("sqlite target stores");
        let report = backup::restore(&target, &dump).await.expect("restore");
        assert_eq!(report.written, 5);
        assert_eq!(report.skipped, 0);
        assert_fixture(&target, &ids).await;
        ids
    };

    // The restored rows are committed: reopening the database sees them.
    let reopened = MemoryStores::sqlite(&target_path)
        .await
        .expect("reopen the sqlite target");
    assert_fixture(&reopened, &ids).await;
}
