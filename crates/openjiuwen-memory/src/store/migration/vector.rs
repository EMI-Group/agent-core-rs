//! Vector-family migrator: legacy embedding collections into the canonical per-`(user, scope, kind)` names.
//!
//! Python's `openjiuwen.core.memory.migration.migrator.vector_migrator.VectorMigrator` tracks a
//! metadata version per vector collection and re-creates a collection when its layout changed.
//! The rename it exists for is the one this port must preserve: an embedding collection is named
//! `uid_{user}_gid_{scope}_mtype_{kind}` (see [`crate::store::keys::vector_collection`]), so a
//! spelling that does not match that grammar — a short kind name such as `mtype_semantic`, a
//! historic `SemanticMemory` spelling, an unsanitized segment — points at a collection name the
//! current code never looks up. Renaming it without copying would silently lose every vector,
//! so this migrator **recollects** the data instead.
//!
//! # Detection
//!
//! [`VectorMigrator`] lists the store's collections and keeps the names starting with `uid_`
//! that parse as `user`, `scope` and `kind` around the `_gid_` and `_mtype_` separators. The kind
//! segment is decoded into a [`MemoryType`] with its Python-compatible serde name, and the
//! canonical target is [`crate::store::keys::vector_collection`] of the parsed scope:
//!
//! - a name that is already canonical (`target == source`) is left alone and not even reported,
//! - a `uid_…` name that does not parse, has an empty segment, or names no known kind becomes a
//!   [`MigrationIssue`] and is **never** touched — a stray collection is reported, not deleted,
//! - a collection that is not a memory collection at all (it does not start with `uid_`) is
//!   ignored, because it belongs to another subsystem.
//!
//! # Plan and apply
//!
//! Planning produces at most one [`RECOLLECT`] step whose `params` carry the planned
//! [`RecollectionPair`]s (informational only: the handler re-detects the work from the store) and
//! whose counters are `scanned` = source documents, `affected` = documents the target is missing,
//! `skipped` = documents the target already holds, `failed` = collections that could not be read.
//!
//! Applying one pair copies the source schema to a missing target, inserts only the documents the
//! target does not already have (in batches of `COPY_BATCH`, because the store rejects a batch
//! containing an existing primary key), **verifies** that every source document is present in the
//! target, and only then deletes the source collection. A pair that cannot be verified keeps its
//! source and is reported `failed`, so the next run copies the remainder instead of losing it.
//!
//! Idempotency: once a pair succeeded its source is gone and its target is canonical, so a rerun
//! detects nothing and plans no step. A dry run creates no collection, inserts nothing, deletes
//! nothing, and reports the counts the real run would produce.
//!
//! [`MigrationIssue`]: super::plan::MigrationIssue

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{CollectionSchema, VectorStore};

use super::operation::{MigrationOperationHandler, OperationOutcome, OperationReport};
use super::plan::{
    scan_collection, MigrationContext, MigrationCounts, MigrationIssue, MigrationStep, Migrator,
};
use crate::record::{MemoryScope, MemoryType};
use crate::store::keys;

/// Migrator id.
pub const MIGRATOR_ID: &str = "vector";

/// Migrator schema version.
pub const MIGRATOR_VERSION: u32 = 1;

/// The single operation that recollects every legacy collection.
pub const RECOLLECT: &str = "vector.recollection";

/// Prefix every embedding collection name starts with.
const COLLECTION_PREFIX: &str = "uid_";

/// Separator that introduces the scope segment of a collection name.
const SCOPE_MARKER: &str = "_gid_";

/// Separator that introduces the kind segment of a collection name.
const KIND_MARKER: &str = "_mtype_";

/// Maximum number of documents inserted per `add_docs` call.
const COPY_BATCH: usize = 100;

/// One planned source→target collection pair.
///
/// The pair is the human-readable record of what the migrator decided; the handler re-derives
/// the work from the store, so a stale plan can never move data into a wrong collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecollectionPair {
    /// Collection the documents are read from.
    pub source: String,
    /// Canonical collection the documents are written to.
    pub target: String,
    /// User segment of the collection name.
    pub user_id: String,
    /// Scope segment of the collection name.
    pub scope_id: String,
    /// Memory kind, as its [`MemoryType`] serde name.
    pub kind: String,
    /// Number of documents the source held when the pair was planned.
    #[serde(default)]
    pub documents: usize,
}

/// The vector-family migrator.
///
/// It is stateless: the store and the run options arrive through the [`MigrationContext`], so one
/// instance is reused for every run and is cheap to share.
#[derive(Debug, Clone, Default)]
pub struct VectorMigrator;

impl VectorMigrator {
    /// Builds the migrator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Migrator for VectorMigrator {
    fn id(&self) -> &'static str {
        MIGRATOR_ID
    }

    fn version(&self) -> u32 {
        MIGRATOR_VERSION
    }

    fn description(&self) -> &'static str {
        "recollect legacy embedding collections into the canonical per-(user, scope, kind) names"
    }

    /// Whether the store holds a legacy collection or one that cannot be read.
    ///
    /// This is a read-only double scan: it plans the work and reports whether the plan is empty,
    /// which keeps "applicable" and "has steps" from drifting apart.
    async fn is_applicable(&self, context: &MigrationContext) -> Result<bool> {
        Ok(!self.plan(context).await?.is_empty())
    }

    async fn plan(&self, context: &MigrationContext) -> Result<Vec<MigrationStep>> {
        let inspection = inspect(context).await?;
        if inspection.candidates.is_empty() && inspection.failed == 0 {
            return Ok(Vec::new());
        }
        let counts = inspection.counts();
        let pairs: Vec<RecollectionPair> = inspection
            .candidates
            .iter()
            .map(|candidate| candidate.pair.clone())
            .collect();
        tracing::debug!(
            target: "openjiuwen_memory::store",
            pairs = pairs.len(),
            failed = inspection.failed,
            "planned the embedding-collection recollection"
        );
        let mut step = MigrationStep::new(
            RECOLLECT,
            MIGRATOR_ID,
            MIGRATOR_VERSION,
            "recollect legacy embedding collections into their canonical names",
        )
        .with_params(serde_json::json!({ "pairs": pairs }))
        .with_counts(counts);
        for issue in inspection.issues {
            step.push_issue(issue.key, issue.reason, context.max_issues);
        }
        Ok(vec![step])
    }

    fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>> {
        vec![Arc::new(Recollect)]
    }
}

/// What a collection name parses into.
enum NameParse {
    /// Not a memory embedding collection (it does not start with `uid_`).
    NotMemory,
    /// A `uid_{user}_gid_{scope}_mtype_{kind}` name.
    Legacy(LegacyName),
    /// A `uid_…` name that cannot be parsed, with the reason.
    Broken(String),
}

/// The segments parsed out of a `uid_{user}_gid_{scope}_mtype_{kind}` collection name.
struct LegacyName {
    /// User segment.
    user_id: String,
    /// Scope segment.
    scope_id: String,
    /// Memory kind named by the trailing segment.
    kind: MemoryType,
}

/// Parses a collection name into its three segments.
fn parse_collection_name(name: &str) -> NameParse {
    let Some(rest) = name.strip_prefix(COLLECTION_PREFIX) else {
        return NameParse::NotMemory;
    };
    let Some((user_id, rest)) = rest.split_once(SCOPE_MARKER) else {
        return NameParse::Broken(format!(
            "a memory collection name must contain {SCOPE_MARKER:?}"
        ));
    };
    let Some((scope_id, raw_kind)) = rest.split_once(KIND_MARKER) else {
        return NameParse::Broken(format!(
            "a memory collection name must contain {KIND_MARKER:?}"
        ));
    };
    if user_id.is_empty() || scope_id.is_empty() || raw_kind.is_empty() {
        return NameParse::Broken("a memory collection name has an empty segment".to_owned());
    }
    let Ok(kind) = serde_json::from_value::<MemoryType>(Value::String(raw_kind.to_owned())) else {
        return NameParse::Broken(format!(
            "the kind segment {raw_kind:?} is not a memory kind"
        ));
    };
    if kind == MemoryType::Unknown {
        return NameParse::Broken(format!(
            "the kind segment {raw_kind:?} is not a known memory kind"
        ));
    }
    NameParse::Legacy(LegacyName {
        user_id: user_id.to_owned(),
        scope_id: scope_id.to_owned(),
        kind,
    })
}

/// One legacy collection that needs recollecting, with everything a copy needs.
struct Candidate {
    /// The planned pair, including its documents count.
    pair: RecollectionPair,
    /// The source schema, copied to a missing target.
    source_schema: CollectionSchema,
    /// Primary-key field name, shared by the source and (when it exists) the target.
    primary_key: String,
    /// Every source document, with its vector restored.
    source_docs: Vec<Map<String, Value>>,
    /// Source primary keys.
    source_ids: BTreeSet<String>,
    /// Primary keys the target already holds.
    target_ids: BTreeSet<String>,
    /// Whether the target collection exists.
    target_exists: bool,
    /// Documents the target is missing.
    missing: usize,
    /// Documents the target already holds.
    present: usize,
}

/// Everything the migrator learned from the vector store.
#[derive(Default)]
struct Inspection {
    /// Legacy collections that can be recollected.
    candidates: Vec<Candidate>,
    /// Collections that could not be read.
    issues: Vec<MigrationIssue>,
    /// Number of collections reported in `issues`.
    failed: usize,
    /// Documents found across the source collections.
    scanned: usize,
}

impl Inspection {
    /// The counters of the whole recollection.
    fn counts(&self) -> MigrationCounts {
        let mut counts = MigrationCounts {
            scanned: self.scanned,
            failed: self.failed,
            ..MigrationCounts::default()
        };
        for candidate in &self.candidates {
            counts.affected += candidate.missing;
            counts.skipped += candidate.present;
        }
        counts
    }
}

/// What inspecting one collection found.
enum CollectionCheck {
    /// A legacy collection with its live contents.
    Candidate(Box<Candidate>),
    /// A collection that must be reported and left alone.
    Issue(MigrationIssue),
}

/// Builds a [`CollectionCheck::Issue`] for `key`.
fn issue(key: &str, reason: impl Into<String>) -> CollectionCheck {
    CollectionCheck::Issue(MigrationIssue {
        key: key.to_owned(),
        reason: reason.into(),
    })
}

/// The comparable form of a document's primary key.
fn document_key(doc: &Map<String, Value>, primary_key: &str) -> Option<String> {
    doc.get(primary_key).map(|value| match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    })
}

/// Scans the store for legacy collections and reads what a copy would need.
async fn inspect(context: &MigrationContext) -> Result<Inspection> {
    let store = context.vector().as_ref();
    let mut inspection = Inspection::default();
    for name in store.list_collection_names().await? {
        let parsed = match parse_collection_name(&name) {
            NameParse::NotMemory => continue,
            NameParse::Broken(reason) => {
                inspection.failed += 1;
                inspection.issues.push(MigrationIssue { key: name, reason });
                continue;
            }
            NameParse::Legacy(parsed) => parsed,
        };
        let target = keys::vector_collection(
            &MemoryScope::new(&parsed.user_id, &parsed.scope_id),
            parsed.kind,
        );
        if target == name {
            // Already canonical: the current code reads it, so it needs no recollection.
            continue;
        }
        match inspect_collection(store, name, parsed, target).await? {
            CollectionCheck::Candidate(candidate) => {
                inspection.scanned += candidate.pair.documents;
                inspection.candidates.push(*candidate);
            }
            CollectionCheck::Issue(problem) => {
                inspection.failed += 1;
                inspection.issues.push(problem);
            }
        }
    }
    Ok(inspection)
}

/// The reason a scan reported, when it is one this migrator reports instead of raising.
///
/// [`scan_collection`] answers `Validation` for a schema no search can be defined on (no
/// `FLOAT_VECTOR` field, or no positive dimension) and `Unsupported` for a collection the
/// store cannot describe at all; both name exactly one collection, so the recollector reports
/// them as a [`CollectionCheck::Issue`] and keeps going. Every other error — a store that
/// failed, a validation error the store raised on its own query — is raised unchanged.
fn scan_issue_reason(error: &Error) -> Option<String> {
    match error {
        Error::Validation(reason) | Error::Unsupported(reason) => Some(reason.clone()),
        _ => None,
    }
}

/// Reads one legacy collection and its canonical counterpart.
///
/// A collection that cannot be read (its schema has no searchable `FLOAT_VECTOR` field, a
/// document has no primary key, the target uses another primary key) is a
/// [`CollectionCheck::Issue`]; a store error is propagated.
async fn inspect_collection(
    store: &dyn VectorStore,
    source: String,
    parsed: LegacyName,
    target: String,
) -> Result<CollectionCheck> {
    let Some(source_schema) = store.get_schema(&source).await? else {
        return Ok(issue(
            &source,
            "the collection disappeared while it was inspected",
        ));
    };
    let Some(primary_key) = source_schema.primary_key().map(|field| field.name.clone()) else {
        return Ok(issue(
            &source,
            "the collection schema declares no primary key",
        ));
    };
    let source_docs = match scan_collection(store, &source).await {
        Ok(docs) => docs,
        Err(error) => match scan_issue_reason(&error) {
            Some(reason) => return Ok(issue(&source, reason)),
            None => return Err(error),
        },
    };
    let mut source_ids = BTreeSet::new();
    for doc in &source_docs {
        let Some(id) = document_key(doc, &primary_key) else {
            return Ok(issue(
                &source,
                format!("a document has no {primary_key:?} primary key"),
            ));
        };
        source_ids.insert(id);
    }

    let target_schema = store.get_schema(&target).await?;
    let mut target_ids = BTreeSet::new();
    if let Some(schema) = &target_schema {
        if schema.primary_key().map(|field| field.name.as_str()) != Some(primary_key.as_str()) {
            return Ok(issue(
                &source,
                format!("the target {target:?} uses a different primary key"),
            ));
        }
        target_ids = match scan_collection(store, &target).await {
            Ok(docs) => docs
                .iter()
                .filter_map(|doc| document_key(doc, &primary_key))
                .collect(),
            Err(error) => match scan_issue_reason(&error) {
                Some(reason) => return Ok(issue(&target, reason)),
                None => return Err(error),
            },
        };
    }

    let documents = source_ids.len();
    let present = source_ids
        .iter()
        .filter(|id| target_ids.contains(*id))
        .count();
    Ok(CollectionCheck::Candidate(Box::new(Candidate {
        pair: RecollectionPair {
            source,
            target,
            user_id: parsed.user_id,
            scope_id: parsed.scope_id,
            kind: parsed.kind.as_str().to_owned(),
            documents,
        },
        source_schema,
        primary_key,
        source_docs,
        source_ids,
        target_ids,
        target_exists: target_schema.is_some(),
        missing: documents - present,
        present,
    })))
}

/// What copying one pair did.
#[derive(Default)]
struct PairOutcome {
    /// Documents copied into the target.
    affected: usize,
    /// Documents the target already held.
    skipped: usize,
    /// `1` when the pair could not be completed, else `0`.
    failed: usize,
    /// Why the pair could not be completed.
    note: Option<String>,
}

impl PairOutcome {
    /// An outcome for a pair that could not be copied, with everything it did before.
    fn incomplete(affected: usize, skipped: usize, note: impl Into<String>) -> Self {
        Self {
            affected,
            skipped,
            failed: 1,
            note: Some(note.into()),
        }
    }
}

/// Copies one legacy collection into its canonical name, then removes the source.
///
/// The source is deleted only after every source document was confirmed present in the target;
/// anything else keeps it and reports the pair as failed, so the next run can finish the copy.
async fn copy_candidate(context: &MigrationContext, candidate: &Candidate) -> Result<PairOutcome> {
    let store = context.vector().as_ref();
    let dry_run = context.is_dry_run();
    let target = candidate.pair.target.as_str();
    let source = candidate.pair.source.as_str();

    if !candidate.target_exists && !dry_run {
        if let Err(error) = store
            .create_collection(target, candidate.source_schema.clone())
            .await
        {
            return Ok(PairOutcome::incomplete(
                0,
                candidate.present,
                format!("cannot create {target:?}: {error}"),
            ));
        }
    }

    let pending: Vec<Map<String, Value>> = candidate
        .source_docs
        .iter()
        .filter(|doc| {
            document_key(doc, &candidate.primary_key)
                .is_some_and(|id| !candidate.target_ids.contains(&id))
        })
        .cloned()
        .collect();
    let mut copied = 0;
    if dry_run {
        copied = pending.len();
    } else {
        for batch in pending.chunks(COPY_BATCH) {
            match store.add_docs(target, batch.to_vec()).await {
                Ok(_) => copied += batch.len(),
                Err(error) => {
                    return Ok(PairOutcome::incomplete(
                        copied,
                        candidate.present,
                        format!("cannot copy documents into {target:?}: {error}"),
                    ))
                }
            }
        }
    }

    if !dry_run {
        match scan_collection(store, target).await {
            Ok(docs) => {
                let stored: BTreeSet<String> = docs
                    .iter()
                    .filter_map(|doc| document_key(doc, &candidate.primary_key))
                    .collect();
                if !candidate.source_ids.iter().all(|id| stored.contains(id)) {
                    return Ok(PairOutcome::incomplete(
                        copied,
                        candidate.present,
                        format!("{target:?} is missing documents; {source:?} is kept"),
                    ));
                }
            }
            Err(error) => match scan_issue_reason(&error) {
                Some(reason) => {
                    return Ok(PairOutcome::incomplete(
                        copied,
                        candidate.present,
                        format!("cannot verify {target:?}: {reason}"),
                    ))
                }
                None => return Err(error),
            },
        }
        if let Err(error) = store.delete_collection(source).await {
            return Ok(PairOutcome::incomplete(
                copied,
                candidate.present,
                format!("cannot delete {source:?}: {error}"),
            ));
        }
    }

    tracing::debug!(
        target: "openjiuwen_memory::store",
        source,
        target = %target,
        copied,
        skipped = candidate.present,
        dry_run,
        "recollected an embedding collection"
    );
    Ok(PairOutcome {
        affected: copied,
        skipped: candidate.present,
        failed: 0,
        note: None,
    })
}

/// Recollects every legacy collection into its canonical name.
///
/// The handler re-detects the work from the store instead of trusting the planned pairs, so a
/// plan that outlived its data cannot move documents into a block that has been deleted since.
/// Per-pair failures are counted and named in the note; the outcome is `Failed` then, so a
/// journal-driven rerun retries the operation.
struct Recollect;

#[async_trait]
impl MigrationOperationHandler for Recollect {
    fn operation_id(&self) -> &'static str {
        RECOLLECT
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let inspection = inspect(context).await?;
        if inspection.candidates.is_empty() && inspection.failed == 0 {
            return Ok(OperationReport::skipped(
                RECOLLECT,
                "no legacy embedding collection to recollect",
            ));
        }
        let mut counts = MigrationCounts {
            scanned: inspection.scanned,
            failed: inspection.failed,
            ..MigrationCounts::default()
        };
        let mut note = inspection
            .issues
            .first()
            .map(|problem| format!("{}: {}", problem.key, problem.reason.trim_end_matches('.')));
        for candidate in &inspection.candidates {
            let outcome = copy_candidate(context, candidate).await?;
            counts.affected += outcome.affected;
            counts.skipped += outcome.skipped;
            counts.failed += outcome.failed;
            note = note.or(outcome.note);
        }
        let outcome = if counts.failed > 0 {
            OperationOutcome::Failed
        } else {
            OperationOutcome::Succeeded
        };
        tracing::info!(
            target: "openjiuwen_memory::store",
            affected = counts.affected,
            skipped = counts.skipped,
            failed = counts.failed,
            dry_run = context.is_dry_run(),
            "recollected legacy embedding collections"
        );
        Ok(OperationReport {
            operation_id: RECOLLECT.to_owned(),
            outcome,
            affected: counts.affected,
            skipped: counts.skipped,
            failed: counts.failed,
            note,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use openjiuwen_foundation::store::{FieldSchema, VectorDataType};
    use serde_json::json;

    use crate::store::migration::{MigrationJournal, MigrationPlan};
    use crate::store::MemoryStores;

    /// A three-dimensional collection schema with a string primary key.
    fn schema() -> CollectionSchema {
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

    /// One document of [`schema`].
    fn document(id: &str, vector: [f32; 3]) -> Map<String, Value> {
        let mut doc = Map::new();
        doc.insert("id".to_owned(), Value::from(id));
        doc.insert(
            "embedding".to_owned(),
            Value::Array(vector.iter().map(|value| Value::from(*value)).collect()),
        );
        doc
    }

    async fn seed(stores: &MemoryStores, collection: &str, docs: Vec<Map<String, Value>>) {
        stores
            .vector
            .create_collection(collection, schema())
            .await
            .expect("collection created");
        if !docs.is_empty() {
            stores
                .vector
                .add_docs(collection, docs)
                .await
                .expect("documents added");
        }
    }

    async fn documents_of(stores: &MemoryStores, collection: &str) -> Vec<Map<String, Value>> {
        scan_collection(stores.vector.as_ref(), collection)
            .await
            .expect("collection scan")
    }

    async fn ids_of(stores: &MemoryStores, collection: &str) -> Vec<String> {
        let mut ids: Vec<String> = documents_of(stores, collection)
            .await
            .iter()
            .filter_map(|doc| document_key(doc, "id"))
            .collect();
        ids.sort();
        ids
    }

    fn context(stores: &MemoryStores) -> MigrationContext {
        MigrationContext::new(stores.clone())
    }

    fn counts(scanned: usize, affected: usize, skipped: usize, failed: usize) -> MigrationCounts {
        MigrationCounts {
            scanned,
            affected,
            skipped,
            failed,
        }
    }

    #[test]
    fn collection_names_parse_into_user_scope_and_kind() {
        let NameParse::Legacy(parsed) =
            parse_collection_name("uid_alice_gid_ws_mtype_semantic_memory")
        else {
            panic!("canonical name must parse");
        };
        assert_eq!(parsed.user_id, "alice");
        assert_eq!(parsed.scope_id, "ws");
        assert_eq!(parsed.kind, MemoryType::SemanticMemory);

        assert!(matches!(
            parse_collection_name("messages"),
            NameParse::NotMemory
        ));
        assert!(matches!(
            parse_collection_name("uid_"),
            NameParse::Broken(_)
        ));
        assert!(matches!(
            parse_collection_name("uid_a_gid_b"),
            NameParse::Broken(_)
        ));
        assert!(matches!(
            parse_collection_name("uid__gid_b_mtype_coding"),
            NameParse::Broken(_)
        ));
        assert!(matches!(
            parse_collection_name("uid_a_gid__mtype_coding"),
            NameParse::Broken(_)
        ));
        assert!(matches!(
            parse_collection_name("uid_a_gid_b_mtype_"),
            NameParse::Broken(_)
        ));
        let NameParse::Broken(reason) = parse_collection_name("uid_a_gid_b_mtype_not_a_kind")
        else {
            panic!("an unknown kind must be reported");
        };
        assert!(reason.contains("not_a_kind"), "{reason}");
    }

    #[test]
    fn migrator_identity_is_stable() {
        let migrator = VectorMigrator::new();
        assert_eq!(migrator.id(), MIGRATOR_ID);
        assert_eq!(migrator.version(), MIGRATOR_VERSION);
        assert!(migrator.description().contains("canonical"));
        let operation_ids: Vec<&str> = migrator
            .operations()
            .iter()
            .map(|handler| handler.operation_id())
            .collect();
        assert_eq!(operation_ids, vec![RECOLLECT]);
    }

    #[tokio::test]
    async fn an_empty_store_plans_nothing() {
        let stores = MemoryStores::in_memory();
        let context = context(&stores);
        let migrator = VectorMigrator::new();
        assert_eq!(migrator.plan(&context).await.expect("plan"), Vec::new());
        assert!(!migrator
            .is_applicable(&context)
            .await
            .expect("applicability"));
    }

    #[tokio::test]
    async fn legacy_collections_are_recollected_into_the_canonical_name() {
        let stores = MemoryStores::in_memory();
        let source = "uid_alice_gid_ws_mtype_semantic";
        let target = "uid_alice_gid_ws_mtype_semantic_memory";
        seed(
            &stores,
            source,
            vec![
                document("a", [1.0, 0.0, 0.0]),
                document("b", [0.0, 1.0, 0.0]),
            ],
        )
        .await;

        let context = context(&stores);
        let migrator = VectorMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.operation_id, RECOLLECT);
        assert_eq!(step.migrator, MIGRATOR_ID);
        assert_eq!(step.version, MIGRATOR_VERSION);
        assert_eq!(step.counts, counts(2, 2, 0, 0));
        assert_eq!(step.issues, Vec::new());
        let pairs: Vec<RecollectionPair> =
            serde_json::from_value(step.params["pairs"].clone()).expect("planned pairs");
        assert_eq!(
            pairs,
            vec![RecollectionPair {
                source: source.to_owned(),
                target: target.to_owned(),
                user_id: "alice".to_owned(),
                scope_id: "ws".to_owned(),
                kind: "semantic_memory".to_owned(),
                documents: 2,
            }]
        );

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(run.is_success());
        assert_eq!(run.reports[0].affected, 2);
        assert_eq!(run.reports[0].skipped, 0);
        assert_eq!(
            ids_of(&stores, target).await,
            vec!["a".to_owned(), "b".to_owned()]
        );
        let copied = documents_of(&stores, target).await;
        let first = copied.iter().find(|doc| doc["id"] == "a").expect("a");
        assert_eq!(first["embedding"], json!([1.0, 0.0, 0.0]));
        let second = copied.iter().find(|doc| doc["id"] == "b").expect("b");
        assert_eq!(second["embedding"], json!([0.0, 1.0, 0.0]));
        assert!(!stores
            .vector
            .collection_exists(source)
            .await
            .expect("source exists"));

        // A rerun detects no legacy collection and plans no step.
        assert_eq!(migrator.plan(&context).await.expect("replan"), Vec::new());
        assert!(!migrator
            .is_applicable(&context)
            .await
            .expect("applicability"));
    }

    #[tokio::test]
    async fn canonical_and_foreign_collections_are_left_alone() {
        let stores = MemoryStores::in_memory();
        let canonical = "uid_alice_gid_ws_mtype_semantic_memory";
        seed(&stores, canonical, vec![document("a", [1.0, 0.0, 0.0])]).await;
        seed(&stores, "scratch", vec![document("z", [0.0, 0.0, 1.0])]).await;

        let context = context(&stores);
        let migrator = VectorMigrator::new();
        assert_eq!(migrator.plan(&context).await.expect("plan"), Vec::new());
        assert!(!migrator
            .is_applicable(&context)
            .await
            .expect("applicability"));

        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &MigrationPlan::new(Vec::new()), &mut journal)
            .await
            .expect("apply of an empty plan");
        assert_eq!(run.reports, Vec::new());
        assert_eq!(ids_of(&stores, canonical).await, vec!["a".to_owned()]);
        assert_eq!(ids_of(&stores, "scratch").await, vec!["z".to_owned()]);
    }

    #[tokio::test]
    async fn unusable_collection_names_are_reported_and_never_deleted() {
        let stores = MemoryStores::in_memory();
        let broken = ["uid_", "uid_a_gid_b", "uid_a_gid_b_mtype_not_a_kind"];
        for name in broken {
            seed(&stores, name, vec![document("a", [1.0, 0.0, 0.0])]).await;
        }

        let context = context(&stores);
        let migrator = VectorMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.operation_id, RECOLLECT);
        assert_eq!(step.counts, counts(0, 0, 0, 3));
        assert_eq!(step.issues.len(), 3);
        assert_eq!(step.params, json!({"pairs": []}));
        assert!(migrator
            .is_applicable(&context)
            .await
            .expect("applicability"));

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports[0].failed, 3);
        assert_eq!(run.reports[0].outcome, OperationOutcome::Failed);
        assert!(!run.is_success());
        for name in broken {
            assert!(
                stores.vector.collection_exists(name).await.expect("exists"),
                "{name} must be kept"
            );
        }
    }

    #[tokio::test]
    async fn a_dry_run_creates_copies_and_deletes_nothing() {
        let stores = MemoryStores::in_memory();
        let source = "uid_alice_gid_ws_mtype_semantic";
        let target = "uid_alice_gid_ws_mtype_semantic_memory";
        seed(
            &stores,
            source,
            vec![
                document("a", [1.0, 0.0, 0.0]),
                document("b", [0.0, 1.0, 0.0]),
            ],
        )
        .await;

        let real = context(&stores);
        let dry = real.clone().with_dry_run(true);
        let migrator = VectorMigrator::new();
        let planned = migrator.plan(&real).await.expect("plan");
        let planned_dry = migrator.plan(&dry).await.expect("dry plan");
        assert_eq!(planned[0].counts, planned_dry[0].counts);
        assert_eq!(planned_dry[0].counts, counts(2, 2, 0, 0));

        let plan = MigrationPlan::new(planned_dry.clone()).with_dry_run(true);
        let mut journal = MigrationJournal::new();
        migrator
            .apply(&dry, &plan, &mut journal)
            .await
            .expect("dry apply");
        assert!(!stores
            .vector
            .collection_exists(target)
            .await
            .expect("target exists"));
        assert_eq!(
            ids_of(&stores, source).await,
            vec!["a".to_owned(), "b".to_owned()]
        );

        // The handler honours the flag itself when it is called directly.
        let handler = migrator.operations().into_iter().next().expect("handler");
        let report = handler
            .execute(&dry, &planned_dry[0])
            .await
            .expect("dry execute");
        assert_eq!(report.outcome, OperationOutcome::Succeeded);
        assert_eq!(report.affected, 2);
        assert_eq!(report.skipped, 0);
        assert!(!stores
            .vector
            .collection_exists(target)
            .await
            .expect("target exists"));
        assert_eq!(
            ids_of(&stores, source).await,
            vec!["a".to_owned(), "b".to_owned()]
        );
    }

    #[tokio::test]
    async fn documents_the_canonical_collection_already_holds_are_skipped() {
        let stores = MemoryStores::in_memory();
        let source = "uid_alice_gid_ws_mtype_semantic";
        let target = "uid_alice_gid_ws_mtype_semantic_memory";
        seed(
            &stores,
            source,
            vec![
                document("a", [1.0, 0.0, 0.0]),
                document("b", [0.0, 1.0, 0.0]),
            ],
        )
        .await;
        seed(&stores, target, vec![document("a", [0.0, 0.0, 1.0])]).await;

        let context = context(&stores);
        let migrator = VectorMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps[0].counts, counts(2, 1, 1, 0));

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(run.is_success());
        assert_eq!(run.reports[0].affected, 1);
        assert_eq!(run.reports[0].skipped, 1);
        assert_eq!(
            ids_of(&stores, target).await,
            vec!["a".to_owned(), "b".to_owned()]
        );
        let target_docs = documents_of(&stores, target).await;
        let pre_existing = target_docs.iter().find(|doc| doc["id"] == "a").expect("a");
        assert_eq!(
            pre_existing["embedding"],
            json!([0.0, 0.0, 1.0]),
            "the pre-existing document is never rewritten"
        );
        assert!(!stores
            .vector
            .collection_exists(source)
            .await
            .expect("source exists"));
    }

    #[tokio::test]
    async fn an_empty_legacy_collection_is_still_migrated() {
        let stores = MemoryStores::in_memory();
        let source = "uid_alice_gid_ws_mtype_code";
        let target = "uid_alice_gid_ws_mtype_coding";
        seed(&stores, source, Vec::new()).await;

        let context = context(&stores);
        let migrator = VectorMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].counts, counts(0, 0, 0, 0));
        let pairs: Vec<RecollectionPair> =
            serde_json::from_value(steps[0].params["pairs"].clone()).expect("planned pairs");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].documents, 0);
        assert_eq!(pairs[0].kind, "coding");

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(run.is_success());
        assert!(stores
            .vector
            .collection_exists(target)
            .await
            .expect("target exists"));
        assert!(!stores
            .vector
            .collection_exists(source)
            .await
            .expect("source exists"));
    }

    #[tokio::test]
    async fn the_step_carries_at_most_one_recollection() {
        let stores = MemoryStores::in_memory();
        seed(&stores, "uid_alice_gid_ws_mtype_semantic", Vec::new()).await;
        seed(&stores, "uid_bob_gid_ws_mtype_SemanticMemory", Vec::new()).await;
        seed(&stores, "uid_a_gid_b_mtype_not_a_kind", Vec::new()).await;

        let context = context(&stores);
        let migrator = VectorMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        let pairs: Vec<RecollectionPair> =
            serde_json::from_value(steps[0].params["pairs"].clone()).expect("planned pairs");
        let targets: Vec<&str> = pairs.iter().map(|pair| pair.target.as_str()).collect();
        assert_eq!(
            targets,
            vec![
                "uid_alice_gid_ws_mtype_semantic_memory",
                "uid_bob_gid_ws_mtype_semantic_memory"
            ]
        );
        assert_eq!(steps[0].counts, counts(0, 0, 0, 1));

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        for target in &targets {
            assert!(
                stores
                    .vector
                    .collection_exists(target)
                    .await
                    .expect("exists"),
                "{target}"
            );
        }
    }
}
