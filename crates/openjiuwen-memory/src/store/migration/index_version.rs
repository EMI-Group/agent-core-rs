//! Index-version stamps for embedding collections, so a stale index requests a rebuild.
//!
//! Python's `index_version_migrator.py` records a per-index version and rebuilds an index
//! whose layout changed. This crate has no index process to rebuild, so the migrator ports
//! the *bookkeeping*: one [`IndexStamp`] per memory embedding collection, stored in the
//! [`INDEX_VERSION_PREFIX`] KV namespace. A collection is **stale** when it has no stamp
//! or its stamp records a version below [`CURRENT_INDEX_VERSION`].
//!
//! - [`STAMP_COLLECTIONS`] writes the stamps: it records the collection's dimensions (from
//!   the vector schema), the current version, `needs_rebuild = true`, and the run's
//!   reference timestamp, so a caller knows a rebuild is owed.
//! - [`PURGE_STALE_EMBEDDINGS`] is the opt-in second step ([`IndexVersionOptions::purge_stale`]):
//!   it deletes the collections whose stamp still asks for a rebuild, counts the documents
//!   it dropped, and keeps the stamp so the rebuild stays visible.
//!
//! Only `uid_*` collections (the memory vector namespace) are considered; any other
//! collection is left completely alone. Stamps are metadata, not a
//! [`KvFamily`](super::super::prefix::KvFamily), so renaming a family never moves them.
//!
//! Stamping is idempotent: once every collection carries the current version, planning
//! finds nothing stale and produces no steps. Purging is idempotent too, because a deleted
//! collection no longer appears in the vector store's collection list.
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{KvStore, KvValue};

use super::operation::{MigrationOperationHandler, OperationReport};
use super::plan::{
    scan_collection, vector_field, MigrationContext, MigrationCounts, MigrationStep, Migrator,
};

/// KV prefix of the per-collection index stamps (metadata, not a `KvFamily`).
pub const INDEX_VERSION_PREFIX: &str = "memory_index_version";

/// The index layout this binary writes.
pub const CURRENT_INDEX_VERSION: u32 = 1;

/// Stable identifier of this migration family.
pub const MIGRATOR_ID: &str = "index_version";

/// The data schema version this migrator brings the stores to.
pub const MIGRATOR_VERSION: u32 = 1;

/// Operation id: stamp stale collections as needing a rebuild.
pub const STAMP_COLLECTIONS: &str = "index_version.stamp_collections";

/// Operation id: purge the stale embedding collections (opt-in).
pub const PURGE_STALE_EMBEDDINGS: &str = "index_version.purge_stale_embeddings";

/// The vector-collection namespace holding memory embeddings (`uid_{user}_gid_{scope}_...`).
const MEMORY_COLLECTION_PREFIX: &str = "uid_";

/// One collection's persisted index stamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexStamp {
    /// The embedding collection this stamp describes.
    pub collection: String,
    /// The index layout version the collection was written with.
    #[serde(default)]
    pub version: u32,
    /// The collection's vector dimension, when the schema declares it.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Whether the collection's vectors must be rebuilt.
    #[serde(default)]
    pub needs_rebuild: bool,
    /// When the stamp was last written.
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,
    /// Why the collection needs a rebuild, when it does.
    #[serde(default)]
    pub reason: Option<String>,
}

impl IndexStamp {
    /// Builds a stamp for `collection` without requesting a rebuild.
    #[must_use]
    pub fn new(collection: impl Into<String>, version: u32, dimensions: Option<usize>) -> Self {
        Self {
            collection: collection.into(),
            version,
            dimensions,
            needs_rebuild: false,
            updated_at: None,
            reason: None,
        }
    }

    /// The KV key of one collection's stamp: `memory_index_version/{collection}`.
    #[must_use]
    pub fn key(collection: &str) -> String {
        format!("{INDEX_VERSION_PREFIX}/{collection}")
    }

    /// Whether the stamp records a version older than `target`.
    #[must_use]
    pub fn is_stale(&self, target: u32) -> bool {
        self.version < target
    }

    /// Reads a collection's stamp, or `None` when it is absent or malformed.
    ///
    /// A malformed stamp is a warning and an absent stamp, never an error: a corrupt
    /// bookkeeping row must not block the migration, and an absent stamp makes the
    /// collection stale, which is exactly the rebuild request a corrupt row deserves.
    ///
    /// # Errors
    ///
    /// Propagates store errors unchanged.
    pub async fn load(kv: &dyn KvStore, collection: &str) -> Result<Option<Self>> {
        let Some(value) = kv.get(&Self::key(collection)).await? else {
            return Ok(None);
        };
        let text = match value {
            KvValue::Text(text) => text,
            KvValue::Bytes(bytes) => match String::from_utf8(bytes) {
                Ok(text) => text,
                Err(error) => {
                    tracing::warn!(
                        target: "openjiuwen_memory::store",
                        %error,
                        collection,
                        "ignoring an undecodable index stamp"
                    );
                    return Ok(None);
                }
            },
        };
        match serde_json::from_str::<Self>(&text) {
            Ok(mut stamp) => {
                if stamp.collection.is_empty() {
                    collection.clone_into(&mut stamp.collection);
                }
                Ok(Some(stamp))
            }
            Err(error) => {
                tracing::warn!(
                    target: "openjiuwen_memory::store",
                    %error,
                    collection,
                    "ignoring a malformed index stamp"
                );
                Ok(None)
            }
        }
    }

    /// Persists the stamp under [`IndexStamp::key`].
    ///
    /// # Errors
    ///
    /// Propagates store and serialization errors.
    pub async fn store(&self, kv: &dyn KvStore) -> Result<()> {
        let text = serde_json::to_string(self).map_err(|error| {
            Error::external(error).with_context(format!(
                "serializing the index stamp of collection {:?}",
                self.collection
            ))
        })?;
        kv.set(&Self::key(&self.collection), KvValue::text(text))
            .await
    }

    /// Whether the collection's stamp asks for a rebuild.
    ///
    /// An absent or malformed stamp reads as `false`; staleness (which those cases *do*
    /// imply) is the migrator's decision, decided against [`Self::is_stale`].
    ///
    /// # Errors
    ///
    /// Propagates store errors unchanged.
    pub async fn needs_rebuild(kv: &dyn KvStore, collection: &str) -> Result<bool> {
        Ok(Self::load(kv, collection)
            .await?
            .is_some_and(|stamp| stamp.needs_rebuild))
    }
}

/// How the migrator should behave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexVersionOptions {
    /// The index version every memory collection should record.
    pub target_version: u32,
    /// Whether the stale embedding collections are deleted after stamping.
    pub purge_stale: bool,
}

impl Default for IndexVersionOptions {
    fn default() -> Self {
        Self {
            target_version: CURRENT_INDEX_VERSION,
            purge_stale: false,
        }
    }
}

/// Stamps memory embedding collections with the current index version and, opt-in, purges
/// the collections whose vectors are stale.
#[derive(Debug, Clone, Default)]
pub struct IndexVersionMigrator {
    /// The migrator's behaviour.
    options: IndexVersionOptions,
}

impl IndexVersionMigrator {
    /// Builds the migrator with [`IndexVersionOptions::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the migrator with explicit options.
    #[must_use]
    pub fn with_options(options: IndexVersionOptions) -> Self {
        Self { options }
    }

    /// The options this migrator runs with.
    #[must_use]
    pub fn options(&self) -> &IndexVersionOptions {
        &self.options
    }

    /// The state of one memory collection's index at plan time.
    async fn inspect(
        &self,
        context: &MigrationContext,
        collection: &str,
    ) -> Result<(Option<usize>, bool, u32, Option<String>)> {
        let stamp = IndexStamp::load(context.kv().as_ref(), collection).await?;
        let current_version = stamp.as_ref().map_or(0, |stamp| stamp.version);
        let stale = stamp
            .as_ref()
            .is_none_or(|stamp| stamp.is_stale(self.options.target_version));
        let (dimensions, issue) = collection_dimensions(context, collection).await?;
        Ok((dimensions, stale, current_version, issue))
    }
}

#[async_trait]
impl Migrator for IndexVersionMigrator {
    fn id(&self) -> &'static str {
        MIGRATOR_ID
    }

    fn version(&self) -> u32 {
        MIGRATOR_VERSION
    }

    fn description(&self) -> &'static str {
        "Stamp memory embedding collections with the current index version"
    }

    async fn is_applicable(&self, context: &MigrationContext) -> Result<bool> {
        Ok(!self.plan(context).await?.is_empty())
    }

    async fn plan(&self, context: &MigrationContext) -> Result<Vec<MigrationStep>> {
        let mut counts = MigrationCounts::default();
        let mut collections = Vec::new();
        let mut issues = Vec::new();
        for collection in memory_collections(context).await? {
            counts.scanned += 1;
            let (dimensions, stale, current_version, issue) =
                self.inspect(context, &collection).await?;
            if !stale {
                counts.skipped += 1;
                continue;
            }
            counts.affected += 1;
            collections.push(json!({
                "collection": collection,
                "current_version": current_version,
                "dimensions": dimensions,
            }));
            if let Some(reason) = issue {
                issues.push((collection, reason));
            }
        }
        if collections.is_empty() {
            return Ok(Vec::new());
        }

        let mut steps = Vec::new();
        let mut stamp = MigrationStep::new(
            STAMP_COLLECTIONS,
            MIGRATOR_ID,
            MIGRATOR_VERSION,
            "stamp memory embedding collections with the current index version",
        )
        .with_prefixes(
            Some(INDEX_VERSION_PREFIX.to_owned()),
            Some(INDEX_VERSION_PREFIX.to_owned()),
        )
        .with_params(json!({ "collections": collections.clone() }))
        .with_counts(counts);
        for (key, reason) in issues {
            stamp.push_issue(key, reason, context.max_issues);
        }
        steps.push(stamp);

        if self.options.purge_stale {
            steps.push(
                MigrationStep::new(
                    PURGE_STALE_EMBEDDINGS,
                    MIGRATOR_ID,
                    MIGRATOR_VERSION,
                    "purge the embedding collections of stale indexes",
                )
                .with_prefixes(
                    Some(INDEX_VERSION_PREFIX.to_owned()),
                    Some(INDEX_VERSION_PREFIX.to_owned()),
                )
                .with_params(json!({ "collections": collections }))
                .with_counts(MigrationCounts {
                    scanned: counts.affected,
                    affected: counts.affected,
                    skipped: 0,
                    failed: 0,
                }),
            );
        }
        Ok(steps)
    }

    fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>> {
        vec![
            Arc::new(StampCollections {
                options: self.options.clone(),
            }),
            Arc::new(PurgeStaleEmbeddings {
                options: self.options.clone(),
            }),
        ]
    }
}

/// The [`STAMP_COLLECTIONS`] operation: write a rebuild-requesting stamp per stale collection.
struct StampCollections {
    /// The migrator's behaviour.
    options: IndexVersionOptions,
}

#[async_trait]
impl MigrationOperationHandler for StampCollections {
    fn operation_id(&self) -> &'static str {
        STAMP_COLLECTIONS
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let target = self.options.target_version;
        let mut affected = 0;
        let mut skipped = 0;
        for collection in memory_collections(context).await? {
            let current = IndexStamp::load(context.kv().as_ref(), &collection)
                .await?
                .map_or(0, |stamp| stamp.version);
            if current >= target {
                skipped += 1;
                continue;
            }
            let (dimensions, _) = collection_dimensions(context, &collection).await?;
            let stamp = IndexStamp {
                collection: collection.clone(),
                version: target,
                dimensions,
                needs_rebuild: true,
                updated_at: Some(context.now),
                reason: Some(format!("index version {current} is older than {target}")),
            };
            if !context.is_dry_run() {
                stamp.store(context.kv().as_ref()).await?;
            }
            affected += 1;
        }
        tracing::info!(
            target: "openjiuwen_memory::store",
            affected,
            skipped,
            "stamped memory embedding collections"
        );
        Ok(OperationReport::succeeded(
            STAMP_COLLECTIONS,
            affected,
            skipped,
            0,
        ))
    }
}

/// The [`PURGE_STALE_EMBEDDINGS`] operation: delete the collections whose stamp still asks
/// for a rebuild, counting the documents they held.
struct PurgeStaleEmbeddings {
    /// The migrator's behaviour.
    options: IndexVersionOptions,
}

#[async_trait]
impl MigrationOperationHandler for PurgeStaleEmbeddings {
    fn operation_id(&self) -> &'static str {
        PURGE_STALE_EMBEDDINGS
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let target = self.options.target_version;
        let mut purged = 0;
        let mut documents = 0;
        let mut skipped = 0;
        for collection in memory_collections(context).await? {
            let stamp = IndexStamp::load(context.kv().as_ref(), &collection).await?;
            let Some(stamp) = stamp else {
                skipped += 1;
                continue;
            };
            if !stamp.needs_rebuild || stamp.version < target {
                skipped += 1;
                continue;
            }
            match scan_collection(context.vector().as_ref(), &collection).await {
                Ok(docs) => documents += docs.len(),
                Err(error) => tracing::warn!(
                    target: "openjiuwen_memory::store",
                    %error,
                    collection,
                    "could not count the documents of a stale collection; purging it anyway"
                ),
            }
            if !context.is_dry_run() {
                context.vector().delete_collection(&collection).await?;
            }
            purged += 1;
        }
        tracing::info!(
            target: "openjiuwen_memory::store",
            purged,
            documents,
            skipped,
            "purged stale memory embedding collections"
        );
        Ok(OperationReport::succeeded(
            PURGE_STALE_EMBEDDINGS,
            purged + documents,
            skipped,
            0,
        ))
    }
}

/// The sorted `uid_*` collections of the injected vector store.
async fn memory_collections(context: &MigrationContext) -> Result<Vec<String>> {
    let mut names = context.vector().list_collection_names().await?;
    names.retain(|name| name.starts_with(MEMORY_COLLECTION_PREFIX));
    names.sort();
    Ok(names)
}

/// The collection's vector dimensions and, when they cannot be read, the reason.
///
/// A collection whose schema declares no vector field cannot hold embeddings, so its
/// dimensions are unknown; that is reported as an issue but never a panic, and the
/// collection is never purged on the strength of a schema it does not have.
async fn collection_dimensions(
    context: &MigrationContext,
    collection: &str,
) -> Result<(Option<usize>, Option<String>)> {
    match context.vector().get_schema(collection).await? {
        Some(schema) => match vector_field(&schema) {
            Some((_, dimensions)) => Ok((Some(dimensions), None)),
            None => Ok((
                None,
                Some(format!(
                    "vector collection {collection:?} declares no FLOAT_VECTOR field; its dimensions are unknown"
                )),
            )),
        },
        None => Ok((
            None,
            Some(format!("vector collection {collection:?} has no schema")),
        )),
    }
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)] // `assert!(x.is_empty())` reads better in these tests
mod tests {
    use super::super::operation::{MigrationJournal, OperationOutcome};
    use super::super::plan::MigrationPlan;
    use super::*;
    use crate::store::MemoryStores;
    use openjiuwen_foundation::store::{
        CollectionSchema, FieldSchema, VectorDataType, VectorQuery, VectorSearchResult, VectorStore,
    };
    use serde_json::{Map, Value};

    const FIRST: &str = "uid_alice_gid_ws_mtype_semantic";
    const SECOND: &str = "uid_bob_gid_ws_mtype_coding";

    /// A schema with a `dim`-dimensional embedding field and a string primary key.
    fn schema(dim: usize) -> CollectionSchema {
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
                dim: Some(dim),
                ..FieldSchema::default()
            },
        ])
    }

    /// One document of [`schema`].
    fn embedding_doc(id: &str, dim: usize) -> Map<String, Value> {
        let mut document = Map::new();
        document.insert("id".to_owned(), Value::from(id));
        document.insert(
            "embedding".to_owned(),
            Value::Array((0..dim).map(|index| Value::from(index as f64)).collect()),
        );
        document
    }

    /// Creates `collection` with a schema of `dim` dimensions.
    async fn create(context: &MigrationContext, collection: &str, dim: usize) {
        context
            .vector()
            .create_collection(collection, schema(dim))
            .await
            .expect("collection created");
    }

    /// A real-run context over fresh in-memory stores.
    fn context() -> MigrationContext {
        MigrationContext::new(MemoryStores::in_memory())
    }

    /// Plans `migrator` into a [`MigrationPlan`].
    async fn planned(migrator: &IndexVersionMigrator, context: &MigrationContext) -> MigrationPlan {
        MigrationPlan::new(migrator.plan(context).await.expect("plan"))
    }

    #[tokio::test]
    async fn unstamped_collections_are_stamped_and_a_rerun_plans_nothing() {
        let context = context();
        create(&context, FIRST, 3).await;
        create(&context, SECOND, 4).await;
        let migrator = IndexVersionMigrator::new();

        let plan = planned(&migrator, &context).await;
        assert_eq!(plan.len(), 1);
        let step = &plan.steps[0];
        assert_eq!(step.operation_id, STAMP_COLLECTIONS);
        assert_eq!(step.migrator, MIGRATOR_ID);
        assert_eq!(step.version, MIGRATOR_VERSION);
        assert_eq!(step.counts.scanned, 2);
        assert_eq!(step.counts.affected, 2);
        assert_eq!(step.counts.skipped, 0);
        assert!(plan
            .operation_ids()
            .iter()
            .all(|id| *id != PURGE_STALE_EMBEDDINGS));

        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports.len(), 1);
        assert_eq!(run.reports[0].affected, 2);

        let first = IndexStamp::load(context.kv().as_ref(), FIRST)
            .await
            .expect("load")
            .expect("stamp");
        assert_eq!(first.collection, FIRST);
        assert_eq!(first.version, CURRENT_INDEX_VERSION);
        assert_eq!(first.dimensions, Some(3));
        assert!(first.needs_rebuild);
        assert_eq!(first.updated_at, Some(context.now));
        assert!(first
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("older than")));
        let second = IndexStamp::load(context.kv().as_ref(), SECOND)
            .await
            .expect("load")
            .expect("stamp");
        assert_eq!(second.dimensions, Some(4));

        assert!(planned(&migrator, &context).await.is_empty());
        assert!(!migrator.is_applicable(&context).await.expect("applicable"));
    }

    #[tokio::test]
    async fn current_collections_are_skipped() {
        let context = context();
        create(&context, FIRST, 3).await;
        create(&context, SECOND, 4).await;
        IndexStamp::new(FIRST, CURRENT_INDEX_VERSION, Some(3))
            .store(context.kv().as_ref())
            .await
            .expect("stamp");

        let plan = planned(&IndexVersionMigrator::new(), &context).await;
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.steps[0].counts.scanned, 2);
        assert_eq!(plan.steps[0].counts.affected, 1);
        assert_eq!(plan.steps[0].counts.skipped, 1);
    }

    #[tokio::test]
    async fn malformed_stamps_read_as_absent_and_make_the_collection_stale() {
        let context = context();
        create(&context, FIRST, 3).await;
        context
            .kv()
            .set(&IndexStamp::key(FIRST), KvValue::text("not json"))
            .await
            .expect("seed malformed");
        assert_eq!(
            IndexStamp::load(context.kv().as_ref(), FIRST)
                .await
                .expect("load"),
            None
        );
        assert!(!IndexStamp::needs_rebuild(context.kv().as_ref(), FIRST)
            .await
            .expect("needs rebuild"));

        let plan = planned(&IndexVersionMigrator::new(), &context).await;
        assert_eq!(plan.steps[0].counts.affected, 1);

        let mut flagged = IndexStamp::new(FIRST, CURRENT_INDEX_VERSION, Some(3));
        flagged.needs_rebuild = true;
        flagged.store(context.kv().as_ref()).await.expect("store");
        assert!(IndexStamp::needs_rebuild(context.kv().as_ref(), FIRST)
            .await
            .expect("needs rebuild"));

        let mut clean = IndexStamp::new(FIRST, CURRENT_INDEX_VERSION, Some(3));
        clean.needs_rebuild = false;
        clean.store(context.kv().as_ref()).await.expect("store");
        assert!(!IndexStamp::needs_rebuild(context.kv().as_ref(), FIRST)
            .await
            .expect("needs rebuild"));
    }

    #[tokio::test]
    async fn purging_deletes_stale_collections_and_keeps_the_rebuild_request() {
        let context = context();
        create(&context, FIRST, 3).await;
        create(&context, SECOND, 3).await;
        context
            .vector()
            .add_docs(FIRST, vec![embedding_doc("a", 3), embedding_doc("b", 3)])
            .await
            .expect("docs");
        context
            .vector()
            .add_docs(SECOND, vec![embedding_doc("c", 3)])
            .await
            .expect("docs");

        let migrator = IndexVersionMigrator::with_options(IndexVersionOptions {
            target_version: CURRENT_INDEX_VERSION,
            purge_stale: true,
        });
        let plan = planned(&migrator, &context).await;
        assert_eq!(plan.len(), 2);
        assert_eq!(
            plan.operation_ids(),
            vec![STAMP_COLLECTIONS, PURGE_STALE_EMBEDDINGS]
        );

        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports.len(), 2);
        assert_eq!(run.reports[0].affected, 2);
        assert_eq!(
            run.reports[1].affected, 5,
            "two collections plus three documents"
        );

        for collection in [FIRST, SECOND] {
            assert!(!context
                .vector()
                .collection_exists(collection)
                .await
                .expect("exists"));
            let stamp = IndexStamp::load(context.kv().as_ref(), collection)
                .await
                .expect("load")
                .expect("stamp kept");
            assert!(stamp.needs_rebuild, "the rebuild request is kept");
            assert_eq!(stamp.version, CURRENT_INDEX_VERSION);
        }
        assert!(planned(&migrator, &context).await.is_empty());
    }

    #[tokio::test]
    async fn a_dry_run_purge_deletes_nothing() {
        let context = context();
        create(&context, FIRST, 3).await;
        context
            .vector()
            .add_docs(FIRST, vec![embedding_doc("a", 3)])
            .await
            .expect("docs");
        let mut stamp = IndexStamp::new(FIRST, CURRENT_INDEX_VERSION, Some(3));
        stamp.needs_rebuild = true;
        stamp.store(context.kv().as_ref()).await.expect("stamp");

        let dry = context.clone().with_dry_run(true);
        let handler = PurgeStaleEmbeddings {
            options: IndexVersionOptions {
                target_version: CURRENT_INDEX_VERSION,
                purge_stale: true,
            },
        };
        let step = MigrationStep::new(
            PURGE_STALE_EMBEDDINGS,
            MIGRATOR_ID,
            MIGRATOR_VERSION,
            "purge",
        );
        let report = handler.execute(&dry, &step).await.expect("handler");
        assert_eq!(report.outcome, OperationOutcome::Succeeded);
        assert_eq!(report.affected, 2, "one collection plus one document");
        assert!(context
            .vector()
            .collection_exists(FIRST)
            .await
            .expect("exists"));
    }

    #[tokio::test]
    async fn a_dry_run_plan_writes_nothing() {
        let context = context();
        create(&context, FIRST, 3).await;
        let dry = context.clone().with_dry_run(true);
        let migrator = IndexVersionMigrator::with_options(IndexVersionOptions {
            target_version: CURRENT_INDEX_VERSION,
            purge_stale: true,
        });
        let plan = planned(&migrator, &dry).await;
        assert_eq!(plan.len(), 2);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&dry, &plan, &mut journal)
            .await
            .expect("dry apply");
        assert!(run
            .reports
            .iter()
            .all(|report| report.outcome == OperationOutcome::Skipped));
        assert_eq!(
            IndexStamp::load(context.kv().as_ref(), FIRST)
                .await
                .expect("load"),
            None
        );
        assert!(context
            .vector()
            .collection_exists(FIRST)
            .await
            .expect("exists"));
    }

    #[tokio::test]
    async fn non_memory_collections_are_ignored() {
        let context = context();
        create(&context, "other_collection", 3).await;
        create(&context, FIRST, 3).await;
        let plan = planned(&IndexVersionMigrator::new(), &context).await;
        assert_eq!(plan.steps[0].counts.scanned, 1);
        assert_eq!(plan.steps[0].params["collections"][0]["collection"], FIRST);
    }

    #[tokio::test]
    async fn a_collection_without_a_vector_field_is_an_issue_not_a_panic() {
        let mut stores = MemoryStores::in_memory();
        stores.vector = Arc::new(SchemaOnlyStore {
            schema: CollectionSchema::new(vec![FieldSchema {
                name: "id".to_owned(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            }]),
        });
        let context = MigrationContext::new(stores);
        let migrator = IndexVersionMigrator::new();
        let plan = planned(&migrator, &context).await;
        assert_eq!(plan.len(), 1);
        let step = &plan.steps[0];
        assert_eq!(step.counts.affected, 1);
        assert_eq!(step.issues.len(), 1);
        assert_eq!(step.issues[0].key, "uid_odd");
        assert!(step.issues[0].reason.contains("no FLOAT_VECTOR field"));

        let mut journal = MigrationJournal::new();
        migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(context
            .vector()
            .collection_exists("uid_odd")
            .await
            .expect("exists"));
    }

    #[test]
    fn stamp_keys_and_json_round_trip_tolerantly() {
        assert_eq!(IndexStamp::key("x"), "memory_index_version/x");
        let minimal: IndexStamp =
            serde_json::from_str(r#"{"collection":"x"}"#).expect("tolerant decode");
        assert_eq!(minimal.version, 0);
        assert_eq!(minimal.dimensions, None);
        assert!(!minimal.needs_rebuild);
        assert_eq!(minimal.updated_at, None);
        assert_eq!(minimal.reason, None);
        assert!(minimal.is_stale(CURRENT_INDEX_VERSION));

        let full = IndexStamp::new("x", CURRENT_INDEX_VERSION, Some(8));
        let text = serde_json::to_string(&full).expect("serialize");
        assert_eq!(
            serde_json::from_str::<IndexStamp>(&text).expect("decode"),
            full
        );
    }

    /// A vector store that reports one `uid_*` collection whose schema has no vector field.
    struct SchemaOnlyStore {
        schema: CollectionSchema,
    }

    #[async_trait]
    impl VectorStore for SchemaOnlyStore {
        async fn create_collection(
            &self,
            _collection: &str,
            _schema: CollectionSchema,
        ) -> Result<()> {
            Err(Error::unsupported("unused in this test"))
        }
        async fn delete_collection(&self, _collection: &str) -> Result<()> {
            Err(Error::unsupported("unused in this test"))
        }
        async fn collection_exists(&self, _collection: &str) -> Result<bool> {
            Ok(true)
        }
        async fn get_schema(&self, _collection: &str) -> Result<Option<CollectionSchema>> {
            Ok(Some(self.schema.clone()))
        }
        async fn add_docs(
            &self,
            _collection: &str,
            _docs: Vec<Map<String, Value>>,
        ) -> Result<usize> {
            Err(Error::unsupported("unused in this test"))
        }
        async fn search(
            &self,
            _collection: &str,
            _query: &VectorQuery,
        ) -> Result<Vec<VectorSearchResult>> {
            Err(Error::unsupported("unused in this test"))
        }
        async fn delete_docs_by_ids(&self, _collection: &str, _ids: &[String]) -> Result<usize> {
            Err(Error::unsupported("unused in this test"))
        }
        async fn delete_docs_by_filters(
            &self,
            _collection: &str,
            _filters: &Value,
        ) -> Result<usize> {
            Err(Error::unsupported("unused in this test"))
        }
        async fn list_collection_names(&self) -> Result<Vec<String>> {
            Ok(vec!["uid_odd".to_owned()])
        }
    }
}
