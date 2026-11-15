//! Migration context, steps, plans, and the `Migrator` trait.
//!
//! A *migrator* is one migration family (SQL tables, vector collections, message
//! documents, index versions). It declares an identity, a schema version, whether it is
//! applicable to the injected stores, the [`MigrationStep`]s it would run, and the named
//! operations that execute those steps. A [`MigrationPlanner`] chains several migrators
//! in registration order: it filters by [`Migrator::is_applicable`], collects their steps
//! into one serializable [`MigrationPlan`], and then executes each migrator's sub-plan
//! against one shared [`MigrationJournal`], so an interrupted run resumes where it
//! stopped.
//!
//! Every step carries its counters and its (bounded) list of [`MigrationIssue`]s, so a
//! plan is both the work order and the report; the JSON forms of [`MigrationPlan`] and
//! [`MigrationJournal`] are the durable artifacts of a migration.
//!
//! ```text
//! MigrationPlanner ──plan()──▶ MigrationPlan { steps: [step, …] }
//!        │                            │
//!        └──run()──▶ Migrator::apply ──▶ OperationRegistry::execute
//!                                              │
//!                                              ▼
//!                                    MigrationRun { plan, journal, reports }
//! ```
//!
//! Planning never touches a data store except through the read-only helpers below
//! (`vector_field`, `scan_vector_field`, `scan_collection`); a dry run
//! ([`MigrationContext::with_dry_run`]
//! or [`MigrationPlan::with_dry_run`]) records `Skipped` journal entries without invoking
//! any handler, so it always reports the work a real run would do.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{
    CollectionSchema, DocumentStore, KvStore, VectorQuery, VectorStore,
};

use crate::store::prefix::KvPrefixRegistry;
use crate::store::MemoryStores;

use super::operation::{
    MigrationJournal, MigrationOperationHandler, MigrationRun, OperationRegistry,
};

/// Format version stamped on every newly produced [`MigrationPlan`].
///
/// It versions the *plan document*, not a data schema: [`MigrationPlan::from_json`]
/// accepts older documents as long as their fields still decode, and a reader can use
/// this field to reject a document written by an incompatible planner.
pub const PLAN_FORMAT_VERSION: u32 = 1;

/// Default bound on the [`MigrationIssue`]s a single step reports.
const DEFAULT_MAX_ISSUES: usize = 100;

/// Everything a migrator needs: the stores, the prefix registry, and run options.
///
/// The context is built once per run and passed by reference, so every migrator in a
/// plan observes the same stores, the same key layout, and the same `now`. It is
/// deliberately not `Debug` (it holds trait objects); tests assert its fields instead.
#[derive(Clone)]
pub struct MigrationContext {
    /// The memory stores the migrators read and write.
    pub stores: MemoryStores,
    /// The key-prefix layout every derived key comes from.
    pub prefixes: KvPrefixRegistry,
    /// Whether handlers must report what they would do instead of writing.
    pub dry_run: bool,
    /// Reference timestamp handed to handlers, so their output is reproducible.
    pub now: DateTime<Utc>,
    /// Upper bound on the [`MigrationIssue`]s collected per step.
    pub max_issues: usize,
}

impl MigrationContext {
    /// Builds a context over `stores` with the default prefix layout and a real run.
    ///
    /// `dry_run` is `false`, `max_issues` is `100`, and `now` is the current UTC time; a
    /// test or a reproducible caller overrides either with the builder methods.
    #[must_use]
    pub fn new(stores: MemoryStores) -> Self {
        Self {
            stores,
            prefixes: KvPrefixRegistry::new(),
            dry_run: false,
            now: Utc::now(),
            max_issues: DEFAULT_MAX_ISSUES,
        }
    }

    /// Replaces the key-prefix layout.
    #[must_use]
    pub fn with_prefixes(mut self, prefixes: KvPrefixRegistry) -> Self {
        self.prefixes = prefixes;
        self
    }

    /// Sets whether the run only reports what it would do.
    #[must_use]
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Whether this run is a dry run.
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    /// The key-value store handlers read and write.
    #[must_use]
    pub fn kv(&self) -> &Arc<dyn KvStore> {
        &self.stores.kv
    }

    /// The document store handlers read and write.
    #[must_use]
    pub fn document(&self) -> &Arc<dyn DocumentStore> {
        &self.stores.document
    }

    /// The vector store handlers read and write.
    #[must_use]
    pub fn vector(&self) -> &Arc<dyn VectorStore> {
        &self.stores.vector
    }
}

/// One malformed item a migrator refused to convert.
///
/// `key` names the offending item in the migrator's own terms (a KV key, a vector
/// collection plus primary key, a document id, …) and `reason` is a human-readable
/// explanation. Issues never fail a run on their own: they bound what a step reports and
/// are counted as `skipped` in its [`MigrationCounts`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationIssue {
    /// Identifier of the item that could not be converted.
    pub key: String,
    /// Why it was left alone.
    pub reason: String,
}

/// Per-step or merged counters of a migration.
///
/// `scanned` counts the items a step examined, `affected` those it converted, `skipped`
/// those it deliberately left alone (including reported [`MigrationIssue`]s), and
/// `failed` those whose conversion raised an error. Summing per-step counts is the
/// meaning of [`MigrationPlan::counts`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationCounts {
    /// Items examined by the step.
    pub scanned: usize,
    /// Items converted by the step.
    pub affected: usize,
    /// Items skipped on purpose.
    pub skipped: usize,
    /// Items whose conversion failed.
    pub failed: usize,
}

impl MigrationCounts {
    /// Adds `other` into `self`, field by field.
    pub fn merge(&mut self, other: &Self) {
        self.scanned = self.scanned.saturating_add(other.scanned);
        self.affected = self.affected.saturating_add(other.affected);
        self.skipped = self.skipped.saturating_add(other.skipped);
        self.failed = self.failed.saturating_add(other.failed);
    }

    /// Returns the field-by-field sum of `self` and `other`, leaving both unchanged.
    #[must_use]
    pub fn merged(mut self, other: &Self) -> Self {
        self.merge(other);
        self
    }

    /// The number of items examined, i.e. [`Self::scanned`].
    #[must_use]
    pub fn total(&self) -> usize {
        self.scanned
    }
}

/// One planned unit of work, serialisable and executable by id.
///
/// A step is the contract between a planner and the [`OperationRegistry`]: the planner
/// fills in what to do (prefixes, parameters, description) and the operation registered
/// under [`MigrationStep::operation_id`] performs it. `counts` and `issues` are filled in
/// by whoever planned the step and are refreshed by the executing handler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Operation that executes the step, globally unique across a plan.
    pub operation_id: String,
    /// Migrator that planned the step (`sql`, `vector`, `message`, `index_version`).
    pub migrator: String,
    /// The migrator's schema version this step belongs to.
    pub version: u32,
    /// Human-readable summary of the step.
    pub description: String,
    /// Namespace the step reads from, when it converts between two layouts.
    #[serde(default)]
    pub source_prefix: Option<String>,
    /// Namespace the step writes to, when it converts between two layouts.
    #[serde(default)]
    pub target_prefix: Option<String>,
    /// Opaque parameters interpreted by the step's handler.
    #[serde(default)]
    pub params: Value,
    /// Counters known at planning time.
    #[serde(default)]
    pub counts: MigrationCounts,
    /// The bounded list of malformed items found while planning.
    #[serde(default)]
    pub issues: Vec<MigrationIssue>,
}

impl MigrationStep {
    /// Builds a step without prefixes, parameters, counters, or issues.
    #[must_use]
    pub fn new(
        operation_id: impl Into<String>,
        migrator: impl Into<String>,
        version: u32,
        description: impl Into<String>,
    ) -> Self {
        Self {
            operation_id: operation_id.into(),
            migrator: migrator.into(),
            version,
            description: description.into(),
            source_prefix: None,
            target_prefix: None,
            params: Value::Null,
            counts: MigrationCounts::default(),
            issues: Vec::new(),
        }
    }

    /// Sets the source and target namespaces the step converts between.
    #[must_use]
    pub fn with_prefixes(mut self, source: Option<String>, target: Option<String>) -> Self {
        self.source_prefix = source;
        self.target_prefix = target;
        self
    }

    /// Sets the handler-specific parameters.
    #[must_use]
    pub fn with_params(mut self, params: Value) -> Self {
        self.params = params;
        self
    }

    /// Sets the counters known at planning time.
    #[must_use]
    pub fn with_counts(mut self, counts: MigrationCounts) -> Self {
        self.counts = counts;
        self
    }

    /// Records one malformed item, ignoring it once `max_issues` issues are collected.
    ///
    /// The bound keeps a pathological input from turning a plan document into a
    /// multi-megabyte report; the counters and the `skipped` tally still account for
    /// every item, so nothing is hidden.
    pub fn push_issue(
        &mut self,
        key: impl Into<String>,
        reason: impl Into<String>,
        max_issues: usize,
    ) {
        if self.issues.len() >= max_issues {
            return;
        }
        self.issues.push(MigrationIssue {
            key: key.into(),
            reason: reason.into(),
        });
    }
}

/// An ordered, serialisable plan produced by a planner over the applicable migrators.
///
/// The plan is the durable work order of a migration: [`MigrationPlan::to_json`] writes
/// it out, [`MigrationPlan::from_json`] reads it back (tolerantly, so a document that
/// omits optional fields still decodes), and [`MigrationPlan::steps_for`] narrows it to
/// one migrator's share. Execution happens through
/// [`OperationRegistry::execute`](super::operation::OperationRegistry::execute).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Format version of this document; see [`PLAN_FORMAT_VERSION`].
    #[serde(default = "default_plan_format_version")]
    pub format_version: u32,
    /// Whether executing the plan only reports what it would do.
    #[serde(default)]
    pub dry_run: bool,
    /// The steps, in execution order.
    #[serde(default)]
    pub steps: Vec<MigrationStep>,
}

impl MigrationPlan {
    /// Builds a real (non-dry) plan from `steps`.
    #[must_use]
    pub fn new(steps: Vec<MigrationStep>) -> Self {
        Self {
            format_version: PLAN_FORMAT_VERSION,
            dry_run: false,
            steps,
        }
    }

    /// Marks the plan as a dry run (or as a real run).
    #[must_use]
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Appends a step.
    pub fn push(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Number of steps in the plan.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the plan holds no step.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The step ids, in execution order.
    #[must_use]
    pub fn operation_ids(&self) -> Vec<&str> {
        self.steps
            .iter()
            .map(|step| step.operation_id.as_str())
            .collect()
    }

    /// All counters merged over the steps.
    #[must_use]
    pub fn counts(&self) -> MigrationCounts {
        let mut total = MigrationCounts::default();
        for step in &self.steps {
            total.merge(&step.counts);
        }
        total
    }

    /// Every issue of every step, in plan order.
    pub fn issues(&self) -> impl Iterator<Item = &MigrationIssue> + '_ {
        self.steps.iter().flat_map(|step| step.issues.iter())
    }

    /// The sub-plan holding only the steps planned by `migrator`, order preserved.
    ///
    /// The result keeps this plan's `dry_run` flag, so a dry run stays a dry run after
    /// being narrowed to one migrator's share.
    #[must_use]
    pub fn steps_for(&self, migrator: &str) -> MigrationPlan {
        Self {
            format_version: self.format_version,
            dry_run: self.dry_run,
            steps: self
                .steps
                .iter()
                .filter(|step| step.migrator == migrator)
                .cloned()
                .collect(),
        }
    }

    /// Serializes the plan to JSON.
    ///
    /// # Errors
    ///
    /// Returns the `serde_json` error (kind [`openjiuwen_common::error::ErrorKind::External`])
    /// annotated with the serialization context.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            Error::external(error).with_context("serializing the memory migration plan")
        })
    }

    /// Decodes a plan produced by [`MigrationPlan::to_json`], tolerating omitted fields.
    ///
    /// A missing `format_version` decodes to [`PLAN_FORMAT_VERSION`], a missing `dry_run`
    /// to `false`, and a missing `steps` to an empty list; a malformed document is
    /// `Error::Validation`.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("invalid memory migration plan JSON: {error}"))
        })
    }
}

/// Serde default of [`MigrationPlan::format_version`].
fn default_plan_format_version() -> u32 {
    PLAN_FORMAT_VERSION
}

/// One migration family: declares its identity, applicability, plan and how to apply it.
///
/// Implementations live beside the family they migrate (`sql`, `vector`, `message`,
/// `index_version`). A migrator must be cheap to share: the planner holds it behind an
/// `Arc` and calls it concurrently with the others in a plan.
#[async_trait]
pub trait Migrator: Send + Sync {
    /// Stable identifier, also the `migrator` of every step this migrator plans.
    fn id(&self) -> &'static str;

    /// The data schema version this migrator brings the stores to.
    fn version(&self) -> u32;

    /// Human-readable summary of what the migrator does.
    fn description(&self) -> &'static str;

    /// Whether the injected stores hold data this migrator needs to convert.
    ///
    /// A migrator that is not applicable contributes no steps and is not planned.
    async fn is_applicable(&self, context: &MigrationContext) -> Result<bool>;

    /// Plans this migrator's steps without writing anything.
    ///
    /// Planning may scan stores (through `scan_collection` and the KV/document
    /// traits) to fill in counts and issues, but must never mutate data.
    async fn plan(&self, context: &MigrationContext) -> Result<Vec<MigrationStep>>;

    /// The operations that execute this migrator's steps.
    ///
    /// Ids must be unique across the whole planner; the default [`Self::apply`]
    /// registers them and executes through the resulting
    /// [`OperationRegistry`].
    fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>>;

    /// Applies `plan`, i.e. the steps whose `migrator` equals [`Self::id`].
    ///
    /// The default implementation narrows `plan` with
    /// [`MigrationPlan::steps_for`](MigrationPlan::steps_for), registers
    /// [`Self::operations`], and executes the result into `journal`. Override it only
    /// when a migrator needs to coordinate across its steps (for example to reopen a
    /// store between them).
    ///
    /// # Errors
    ///
    /// Returns a validation error for a duplicated or malformed operation id and the
    /// handler's error (with the operation id in its context) for a failing step; the
    /// journal records the attempt either way.
    async fn apply(
        &self,
        context: &MigrationContext,
        plan: &MigrationPlan,
        journal: &mut MigrationJournal,
    ) -> Result<MigrationRun> {
        let mut registry = OperationRegistry::new();
        registry.register_all(self.operations())?;
        let steps = plan.steps_for(self.id());
        registry.execute(context, &steps, journal).await
    }
}

/// Plans and runs a set of migrators in registration order.
///
/// Registration order is both planning order and execution order, so a planner built as
/// `[sql, vector, message]` plans and applies the SQL steps first. The planner is a
/// plain value (no global registry), cheap to clone, and empty by default.
#[derive(Clone, Default)]
pub struct MigrationPlanner {
    /// Migrators in registration order.
    migrators: Vec<Arc<dyn Migrator>>,
}

impl MigrationPlanner {
    /// Builds a planner over `migrators`, which keep the given order.
    #[must_use]
    pub fn new(migrators: Vec<Arc<dyn Migrator>>) -> Self {
        Self { migrators }
    }

    /// The registered migrators, in order.
    #[must_use]
    pub fn migrators(&self) -> &[Arc<dyn Migrator>] {
        &self.migrators
    }

    /// Appends `migrator` after the ones already registered.
    pub fn add(&mut self, migrator: Arc<dyn Migrator>) {
        self.migrators.push(migrator);
    }

    /// Whether no migrator is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.migrators.is_empty()
    }

    /// Registers every migrator's operations into one registry.
    ///
    /// # Errors
    ///
    /// Returns a validation error naming the operation when two migrators claim the same
    /// operation id, or when an id is malformed; no registry is returned then.
    pub fn registry(&self) -> Result<OperationRegistry> {
        let mut registry = OperationRegistry::new();
        for migrator in &self.migrators {
            registry.register_all(migrator.operations())?;
        }
        Ok(registry)
    }

    /// Plans every applicable migrator, in registration order.
    ///
    /// A migrator whose [`Migrator::is_applicable`] answers `false` contributes no steps
    /// and is not asked to plan. The result's `dry_run` mirrors `context.dry_run`, so a
    /// dry-run context produces a dry-run plan.
    ///
    /// # Errors
    ///
    /// Propagates the first applicability check or planning error unchanged.
    pub async fn plan(&self, context: &MigrationContext) -> Result<MigrationPlan> {
        let mut steps: Vec<MigrationStep> = Vec::new();
        for migrator in &self.migrators {
            if !migrator.is_applicable(context).await? {
                tracing::debug!(
                    target: "openjiuwen_memory::store",
                    migrator = migrator.id(),
                    "memory migration migrator is not applicable; skipping it"
                );
                continue;
            }
            steps.extend(migrator.plan(context).await?);
        }
        Ok(MigrationPlan::new(steps).with_dry_run(context.is_dry_run()))
    }

    /// Plans, then applies each migrator's sub-plan in registration order.
    ///
    /// One journal is shared by every migrator, so an operation that already succeeded is
    /// not repeated and a failed one is retried. The returned run carries the full plan,
    /// the final journal, and every report in execution order.
    ///
    /// # Errors
    ///
    /// Returns the first failing operation's error after the journal has recorded the
    /// attempt, so the caller resumes by passing the same journal again.
    pub async fn run(
        &self,
        context: &MigrationContext,
        journal: &mut MigrationJournal,
    ) -> Result<MigrationRun> {
        let plan = self.plan(context).await?;
        let mut reports = Vec::new();
        for migrator in &self.migrators {
            let steps = plan.steps_for(migrator.id());
            if steps.is_empty() {
                continue;
            }
            let run = migrator.apply(context, &steps, journal).await?;
            reports.extend(run.reports);
        }
        Ok(MigrationRun {
            plan,
            journal: journal.clone(),
            reports,
        })
    }
}

/// The collection's vector field: (field name, dimensions).
///
/// The first `FLOAT_VECTOR` field with a positive dimension wins, mirroring the store's own
/// default-vector-field rule, so a schema declaring `[embedding: None, dense: Some(4)]`
/// scans through `dense`. `None` means no field carries the dimension a scan has to size its
/// query vector with; [`scan_vector_field`] turns that into the typed error naming the cause.
pub(crate) fn vector_field(schema: &CollectionSchema) -> Option<(&str, usize)> {
    schema.vector_fields().into_iter().find_map(|field| {
        field
            .dim
            .filter(|dim| *dim > 0)
            .map(|dim| (field.name.as_str(), dim))
    })
}

/// Why a schema that [`vector_field`] answers `None` for cannot be searched, in one sentence.
///
/// Two shapes are unsearchable, and `CollectionSchema::validate` rejects both: a schema with
/// no `FLOAT_VECTOR` field at all, and one whose `FLOAT_VECTOR` fields declare no positive
/// dimension. Neither can be worked around locally: a search sizes its query vector from the
/// dimension, and the vector store exposes no document listing that could stand in for one,
/// so the reason names the collection, the offending fields, and what the caller can do.
pub(crate) fn unsearchable_reason(collection: &str, schema: &CollectionSchema) -> String {
    let vectors = schema.vector_fields();
    if vectors.is_empty() {
        return format!(
            "vector collection {collection:?} declares no FLOAT_VECTOR field, so it holds nothing to search; \
             a valid collection schema requires one (recreate the collection)"
        );
    }
    let names: Vec<&str> = vectors.iter().map(|field| field.name.as_str()).collect();
    format!(
        "vector collection {collection:?} declares its FLOAT_VECTOR field(s) {names:?} without a positive \
         dimension, and a search sizes its query vector from that dimension; recreate the collection with \
         a dimensional schema"
    )
}

/// The vector field a scan of `collection` ranks against, or the reason it cannot be scanned.
///
/// # Errors
///
/// `Error::Validation` carrying [`unsearchable_reason`] when the schema declares no
/// `FLOAT_VECTOR` field with a positive dimension, because such a schema cannot be searched.
pub(crate) fn scan_vector_field<'a>(
    collection: &str,
    schema: &'a CollectionSchema,
) -> Result<(&'a str, usize)> {
    let Some((field, dim)) = vector_field(schema) else {
        return Err(Error::validation(unsearchable_reason(collection, schema)));
    };
    Ok((field, dim))
}

/// Upper bound passed to the snapshot scan; the store truncates results to `top_k`.
pub(crate) const SCAN_TOP_K: usize = usize::MAX;

/// Every document of `collection`, with vector fields restored as JSON arrays.
///
/// The vector store has no document-listing API, so a scan is a similarity search: the
/// query is a zero vector of the collection's dimension with no filters and
/// [`SCAN_TOP_K`], which returns a snapshot of every document (the scores are
/// meaningless and discarded). Each map includes the primary key.
///
/// The query length is the one number a scan needs from the schema, so
/// [`scan_vector_field`] supplies it: the first `FLOAT_VECTOR` field with a positive `dim`,
/// which is also the field the store searches itself when a query names none.
///
/// # Errors
///
/// A missing collection scans empty. Everything else this build cannot decide by itself is a
/// typed error naming the collection, never a guess:
///
/// - `Error::Validation` when the schema the store returns is not a searchable collection
///   schema: no `FLOAT_VECTOR` field at all, or none with a positive dimension.
///   [`CollectionSchema::validate`] rejects both, so no conforming store can create such a
///   collection, and no store can search it either — the collection has to be recreated, not
///   scanned again. A dimensionless vector field cannot stand in for the missing number
///   either: the store sizes the query against that same `dim`, so an empty query would only
///   suit a backend that reads a missing dimension as zero, which the [`VectorStore`] contract
///   does not promise.
/// - `Error::Unsupported` when the store reports the collection exists but returns no schema:
///   without a dimension there is no query length, and [`VectorStore`] offers no document
///   listing or get-by-id call to read the documents without one. This is the one case the
///   store seam genuinely blocks, and it means the store cannot describe what it holds.
/// - A store error from `search` propagates unchanged, so a backend that cannot answer an
///   unfiltered top-k keeps its own kind and message.
pub(crate) async fn scan_collection(
    store: &dyn VectorStore,
    collection: &str,
) -> Result<Vec<Map<String, Value>>> {
    let Some(schema) = store.get_schema(collection).await? else {
        if store.collection_exists(collection).await? {
            return Err(Error::unsupported(format!(
                "cannot scan vector collection {collection:?}: it exists but the store returns no schema, so \
                 there is no dimension to size the scan query with and the vector store exposes no document listing"
            )));
        }
        return Ok(Vec::new());
    };
    let (field, dim) = scan_vector_field(collection, &schema)?;
    let query = VectorQuery::new(vec![0.0; dim])
        .with_vector_field(field)
        .with_top_k(SCAN_TOP_K)
        .with_filters(Value::Null);
    let hits = store.search(collection, &query).await?;
    Ok(hits.into_iter().map(|hit| hit.fields).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use openjiuwen_foundation::store::{
        FieldSchema, InMemoryVectorStore, VectorDataType, VectorSearchResult,
    };

    use crate::store::migration::operation::{OperationOutcome, OperationRecord, OperationReport};

    /// A minimal step with no prefixes, parameters, or counters.
    fn step(operation_id: &str, migrator: &str) -> MigrationStep {
        MigrationStep::new(operation_id, migrator, 1, "test step")
    }

    /// A handler that counts its calls and can fail its first `fail_times` attempts.
    struct CountingHandler {
        id: &'static str,
        calls: Arc<AtomicUsize>,
        fail_times: usize,
    }

    impl CountingHandler {
        fn new(id: &'static str, calls: Arc<AtomicUsize>, fail_times: usize) -> Self {
            Self {
                id,
                calls,
                fail_times,
            }
        }
    }

    #[async_trait]
    impl MigrationOperationHandler for CountingHandler {
        fn operation_id(&self) -> &'static str {
            self.id
        }

        async fn execute(
            &self,
            _context: &MigrationContext,
            step: &MigrationStep,
        ) -> Result<OperationReport> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            if attempt < self.fail_times {
                return Err(Error::memory(format!("boom in {}", step.operation_id)));
            }
            Ok(OperationReport::succeeded(
                step.operation_id.clone(),
                1,
                0,
                0,
            ))
        }
    }

    /// A fake migrator with a fixed identity, applicability, steps, and operations.
    struct FakeMigrator {
        id: &'static str,
        version: u32,
        applicable: bool,
        steps: Vec<MigrationStep>,
        handlers: Vec<Arc<dyn MigrationOperationHandler>>,
        plan_calls: Arc<AtomicUsize>,
    }

    impl FakeMigrator {
        fn new(id: &'static str, applicable: bool) -> Self {
            Self {
                id,
                version: 1,
                applicable,
                steps: Vec::new(),
                handlers: Vec::new(),
                plan_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_step(mut self, step: MigrationStep) -> Self {
            self.steps.push(step);
            self
        }

        fn with_handler(mut self, handler: Arc<dyn MigrationOperationHandler>) -> Self {
            self.handlers.push(handler);
            self
        }
    }

    #[async_trait]
    impl Migrator for FakeMigrator {
        fn id(&self) -> &'static str {
            self.id
        }

        fn version(&self) -> u32 {
            self.version
        }

        fn description(&self) -> &'static str {
            "fake migrator"
        }

        async fn is_applicable(&self, _context: &MigrationContext) -> Result<bool> {
            Ok(self.applicable)
        }

        async fn plan(&self, _context: &MigrationContext) -> Result<Vec<MigrationStep>> {
            self.plan_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.steps.clone())
        }

        fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>> {
            self.handlers.clone()
        }
    }

    fn context() -> MigrationContext {
        MigrationContext::new(MemoryStores::in_memory())
    }

    // --- context -----------------------------------------------------------

    #[test]
    fn context_defaults_and_builders() {
        let context = context();
        assert!(!context.is_dry_run());
        assert_eq!(context.max_issues, 100);
        assert_eq!(context.prefixes, KvPrefixRegistry::new());
        let renamed = context.with_dry_run(true).with_prefixes(
            KvPrefixRegistry::new()
                .with_current(crate::store::prefix::KvFamily::Records, "NEW")
                .expect("valid prefix"),
        );
        assert!(renamed.is_dry_run());
        assert_eq!(
            renamed
                .prefixes
                .current(crate::store::prefix::KvFamily::Records),
            "NEW"
        );
        // The store accessors expose the same Arcs as the bundle.
        assert!(Arc::ptr_eq(renamed.kv(), &renamed.stores.kv));
        assert!(Arc::ptr_eq(renamed.document(), &renamed.stores.document));
        assert!(Arc::ptr_eq(renamed.vector(), &renamed.stores.vector));
    }

    // --- plan --------------------------------------------------------------

    #[test]
    fn plan_json_round_trip_and_tolerant_decode() {
        let mut first = step("sql.copy_user_variables", "sql")
            .with_prefixes(
                Some("user_variable".to_owned()),
                Some("user_var".to_owned()),
            )
            .with_params(serde_json::json!({"limit": 10}))
            .with_counts(MigrationCounts {
                scanned: 3,
                affected: 2,
                skipped: 1,
                failed: 0,
            });
        first.push_issue("user_variable/x", "not a JSON object", 10);
        let mut plan = MigrationPlan::new(vec![first, step("vector.reindex", "vector")]);
        plan.push(step("message.move", "message"));
        plan = plan.with_dry_run(true);

        let text = plan.to_json().expect("serializable");
        let decoded = MigrationPlan::from_json(&text).expect("decodable");
        assert_eq!(decoded, plan);
        assert_eq!(decoded.format_version, PLAN_FORMAT_VERSION);
        assert!(decoded.dry_run);
        assert_eq!(decoded.operation_ids().len(), 3);
    }

    #[test]
    fn plan_decodes_an_empty_document_and_minimal_steps() {
        let empty = MigrationPlan::from_json("{}").expect("tolerant decode");
        assert_eq!(empty.format_version, PLAN_FORMAT_VERSION);
        assert!(!empty.dry_run);
        assert!(empty.is_empty());

        let minimal = MigrationPlan::from_json(
            r#"{"steps":[{"operation_id":"sql.one","migrator":"sql","version":2,"description":"d"}]}"#,
        )
        .expect("tolerant step decode");
        assert_eq!(minimal.len(), 1);
        let step = &minimal.steps[0];
        assert_eq!(step.params, Value::Null);
        assert_eq!(step.counts, MigrationCounts::default());
        assert_eq!(step.issues.len(), 0);
        assert_eq!(step.source_prefix, None);
        assert_eq!(step.target_prefix, None);
        assert_eq!(step.version, 2);
    }

    #[test]
    fn plan_rejects_malformed_json() {
        let error = MigrationPlan::from_json("not a plan").expect_err("invalid");
        assert!(matches!(error, Error::Validation(_)), "{error}");
    }

    #[test]
    fn plan_counts_merge_over_steps() {
        let plan = MigrationPlan::new(vec![
            step("sql.one", "sql").with_counts(MigrationCounts {
                scanned: 4,
                affected: 3,
                skipped: 1,
                failed: 0,
            }),
            step("sql.two", "sql").with_counts(MigrationCounts {
                scanned: 2,
                affected: 0,
                skipped: 1,
                failed: 1,
            }),
        ]);
        let counts = plan.counts();
        assert_eq!(counts.scanned, 6);
        assert_eq!(counts.affected, 3);
        assert_eq!(counts.skipped, 2);
        assert_eq!(counts.failed, 1);
        assert_eq!(counts.total(), 6);
        assert_eq!(
            MigrationCounts::default().merged(&counts),
            counts,
            "merging into the default is the identity"
        );
    }

    #[test]
    fn plan_steps_for_filters_and_keeps_dry_run() {
        let plan = MigrationPlan::new(vec![
            step("sql.one", "sql"),
            step("vector.one", "vector"),
            step("sql.two", "sql"),
        ])
        .with_dry_run(true);

        let sql = plan.steps_for("sql");
        assert_eq!(sql.operation_ids(), vec!["sql.one", "sql.two"]);
        assert!(sql.dry_run);
        assert_eq!(sql.format_version, PLAN_FORMAT_VERSION);
        assert_eq!(plan.steps_for("index_version").len(), 0);
        assert_eq!(
            plan.operation_ids(),
            vec!["sql.one", "vector.one", "sql.two"]
        );
    }

    #[test]
    fn step_issues_are_bounded_and_plan_issues_are_flattened() {
        let mut first = step("sql.one", "sql");
        for index in 0..5 {
            first.push_issue(format!("k{index}"), "bad", 2);
        }
        assert_eq!(first.issues.len(), 2);
        assert_eq!(first.issues[0].key, "k0");
        assert_eq!(first.issues[1].reason, "bad");

        let mut second = step("sql.two", "sql");
        second.push_issue("k9", "also bad", 2);
        let plan = MigrationPlan::new(vec![first, second]);
        let keys: Vec<&str> = plan.issues().map(|issue| issue.key.as_str()).collect();
        assert_eq!(keys, vec!["k0", "k1", "k9"]);

        let mut none = step("sql.three", "sql");
        none.push_issue("k", "r", 0);
        assert!(none.issues.is_empty(), "a zero bound collects nothing");
    }

    // --- planner -----------------------------------------------------------

    #[tokio::test]
    async fn planner_skips_inapplicable_migrators_and_preserves_order() {
        let first = Arc::new(
            FakeMigrator::new("sql", true)
                .with_step(MigrationStep::new("sql.one", "sql", 2, "one")),
        );
        let skipped = Arc::new(FakeMigrator::new("vector", false));
        let last = Arc::new(
            FakeMigrator::new("message", true).with_step(MigrationStep::new(
                "message.one",
                "message",
                1,
                "one",
            )),
        );
        let planner = MigrationPlanner::new(vec![first.clone(), skipped.clone(), last.clone()]);
        assert_eq!(planner.migrators().len(), 3);
        assert!(!planner.is_empty());

        let plan = planner
            .plan(&context().with_dry_run(true))
            .await
            .expect("plan");
        assert_eq!(plan.operation_ids(), vec!["sql.one", "message.one"]);
        assert!(plan.dry_run);
        assert_eq!(first.plan_calls.load(Ordering::SeqCst), 1);
        assert_eq!(last.plan_calls.load(Ordering::SeqCst), 1);
        assert_eq!(skipped.plan_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn empty_planner_plans_and_runs_nothing() {
        let planner = MigrationPlanner::default();
        assert!(planner.is_empty());
        let mut journal = MigrationJournal::new();
        let run = planner.run(&context(), &mut journal).await.expect("run");
        assert!(run.plan.is_empty());
        assert!(run.journal.is_empty());
        assert_eq!(run.reports.len(), 0);
        assert!(run.is_success());
    }

    #[tokio::test]
    async fn planner_registry_rejects_duplicate_ids_across_migrators() {
        let calls = Arc::new(AtomicUsize::new(0));
        let first = Arc::new(
            FakeMigrator::new("sql", true).with_handler(Arc::new(CountingHandler::new(
                "sql.one",
                calls.clone(),
                0,
            ))),
        );
        let second = Arc::new(
            FakeMigrator::new("vector", true)
                .with_handler(Arc::new(CountingHandler::new("sql.one", calls, 0))),
        );
        let planner = MigrationPlanner::new(vec![first, second]);
        let error = planner.registry().err().expect("duplicate id");
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("sql.one"), "{error}");
    }

    #[tokio::test]
    async fn default_apply_executes_through_the_registry() {
        let calls = Arc::new(AtomicUsize::new(0));
        let migrator = FakeMigrator::new("sql", true).with_handler(Arc::new(CountingHandler::new(
            "sql.one",
            calls.clone(),
            0,
        )));
        let plan = MigrationPlan::new(vec![step("sql.one", "sql"), step("vector.one", "vector")]);
        let mut journal = MigrationJournal::new();

        let run = migrator
            .apply(&context(), &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(run.plan.operation_ids(), vec!["sql.one"]);
        assert_eq!(run.reports.len(), 1);
        assert!(run.is_success());
        assert!(journal.succeeded("sql.one"));
        assert_eq!(journal.len(), 1);
    }

    #[tokio::test]
    async fn planner_run_records_the_journal_before_failing_and_resumes() {
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let migrator = FakeMigrator::new("sql", true)
            .with_step(MigrationStep::new("sql.one", "sql", 1, "one"))
            .with_step(MigrationStep::new("sql.two", "sql", 1, "two"))
            .with_handler(Arc::new(CountingHandler::new(
                "sql.one",
                first_calls.clone(),
                0,
            )))
            .with_handler(Arc::new(CountingHandler::new(
                "sql.two",
                second_calls.clone(),
                1,
            )));
        let planner = MigrationPlanner::new(vec![Arc::new(migrator)]);
        let mut journal = MigrationJournal::new();

        let error = planner
            .run(&context(), &mut journal)
            .await
            .expect_err("second operation fails");
        assert!(error.to_string().contains("sql.two"), "{error}");
        assert!(journal.succeeded("sql.one"));
        let failed = journal.get("sql.two").expect("failure recorded");
        assert_eq!(failed.outcome, OperationOutcome::Failed);
        assert_eq!(failed.attempts, 1);
        assert!(failed
            .error
            .as_deref()
            .is_some_and(|text| text.contains("boom")));

        let run = planner
            .run(&context(), &mut journal)
            .await
            .expect("resumed");
        assert_eq!(run.plan.len(), 2);
        assert_eq!(run.reports.len(), 2);
        assert!(run.is_success());
        assert_eq!(
            first_calls.load(Ordering::SeqCst),
            1,
            "completed step is not repeated"
        );
        assert_eq!(second_calls.load(Ordering::SeqCst), 2);
        assert_eq!(journal.get("sql.two").expect("record").attempts, 2);
    }

    // --- scan_collection ---------------------------------------------------

    /// A store with one fixed schema that records the query every search receives.
    ///
    /// It exists because the in-memory store rejects the schemas a *foreign* or hand-edited
    /// backend can hand back (`CollectionSchema::validate` forbids a dimension-less vector
    /// field), so the scan's own schema classification is only reachable through a fake.
    struct ScriptedVectorStore {
        /// What `collection_exists` answers.
        exists: bool,
        /// What `get_schema` answers.
        schema: Option<CollectionSchema>,
        /// The fields every search hit carries.
        hits: Vec<Map<String, Value>>,
        /// The queries `search` was called with, in order.
        queries: Mutex<Vec<VectorQuery>>,
    }

    impl ScriptedVectorStore {
        /// A store that answers `schema` and returns one hit per entry in `hits`.
        fn new(schema: CollectionSchema, hits: Vec<Map<String, Value>>) -> Self {
            Self {
                exists: true,
                schema: Some(schema),
                hits,
                queries: Mutex::new(Vec::new()),
            }
        }

        /// A store whose collection exists but has no schema at all.
        fn hidden() -> Self {
            Self {
                exists: true,
                schema: None,
                hits: Vec::new(),
                queries: Mutex::new(Vec::new()),
            }
        }

        /// The last query `search` received, when it was called at all.
        fn last_query(&self) -> Option<VectorQuery> {
            self.queries.lock().expect("queries").last().cloned()
        }
    }

    #[async_trait]
    impl VectorStore for ScriptedVectorStore {
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
            Ok(self.exists)
        }

        async fn get_schema(&self, _collection: &str) -> Result<Option<CollectionSchema>> {
            Ok(self.schema.clone())
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
            query: &VectorQuery,
        ) -> Result<Vec<VectorSearchResult>> {
            self.queries.lock().expect("queries").push(query.clone());
            Ok(self
                .hits
                .iter()
                .map(|fields| VectorSearchResult::new(0.0, fields.clone()))
                .collect())
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
            Ok(vec!["odd".to_owned()])
        }
    }

    /// A three-dimensional schema with a string primary key.
    fn embedding_schema() -> CollectionSchema {
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

    /// One document of [`embedding_schema`].
    fn embedding_doc(id: &str, vector: [f32; 3]) -> Map<String, Value> {
        let mut doc = Map::new();
        doc.insert("id".to_owned(), Value::from(id));
        doc.insert(
            "embedding".to_owned(),
            Value::Array(vector.iter().map(|value| Value::from(*value)).collect()),
        );
        doc
    }

    #[tokio::test]
    async fn scan_collection_returns_every_document_with_its_vector() {
        let store = InMemoryVectorStore::new();
        store
            .create_collection("uid_alice_gid_ws_mtype_coding", embedding_schema())
            .await
            .expect("collection created");
        store
            .add_docs(
                "uid_alice_gid_ws_mtype_coding",
                vec![
                    embedding_doc("a", [1.0, 0.0, 0.0]),
                    embedding_doc("b", [0.0, 1.0, 0.0]),
                ],
            )
            .await
            .expect("documents added");

        let docs = scan_collection(&store, "uid_alice_gid_ws_mtype_coding")
            .await
            .expect("scan");
        assert_eq!(docs.len(), 2);
        let ids: BTreeSet<String> = docs
            .iter()
            .filter_map(|doc| doc.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("a") && ids.contains("b"));
        for doc in &docs {
            let embedding = doc.get("embedding").expect("vector restored");
            let components = embedding.as_array().expect("JSON array");
            assert_eq!(components.len(), 3);
            assert!(components.iter().all(Value::is_number));
        }
    }

    #[tokio::test]
    async fn scan_collection_of_a_missing_collection_is_empty() {
        let store = InMemoryVectorStore::new();
        let docs = scan_collection(&store, "absent").await.expect("scan");
        assert_eq!(docs.len(), 0);
    }

    #[tokio::test]
    async fn scan_collection_of_an_existing_collection_without_a_schema_is_unsupported() {
        let store = ScriptedVectorStore::hidden();
        let error = scan_collection(&store, "ghost")
            .await
            .expect_err("unsupported");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        assert!(error.to_string().contains("ghost"), "{error}");
        assert!(error.to_string().contains("no schema"), "{error}");
        assert!(
            store.last_query().is_none(),
            "a schema-less collection is never searched: there is no dimension to size the query"
        );
    }

    #[tokio::test]
    async fn scan_collection_without_a_vector_field_is_validation() {
        let store = ScriptedVectorStore::new(
            CollectionSchema::new(vec![FieldSchema {
                name: "id".to_owned(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            }]),
            Vec::new(),
        );
        let error = scan_collection(&store, "odd")
            .await
            .expect_err("validation");
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("odd"), "{error}");
        assert!(
            error.to_string().contains("no FLOAT_VECTOR field"),
            "{error}"
        );
        assert!(
            store.last_query().is_none(),
            "an unsearchable collection is reported, never searched"
        );
        assert_eq!(vector_field(store.schema.as_ref().expect("schema")), None);
    }

    #[tokio::test]
    async fn scan_collection_of_a_dimensionless_vector_field_is_validation() {
        for dim in [None, Some(0)] {
            let store = ScriptedVectorStore::new(
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
                        dim,
                        ..FieldSchema::default()
                    },
                ]),
                Vec::new(),
            );
            let error = scan_collection(&store, "uid_a_gid_b_mtype_coding")
                .await
                .expect_err("validation");
            assert!(matches!(error, Error::Validation(_)), "{error}");
            let message = error.to_string();
            assert!(message.contains("uid_a_gid_b_mtype_coding"), "{message}");
            assert!(message.contains(r#"["embedding"]"#), "{message}");
            assert!(message.contains("positive"), "{message}");
            assert!(
                store.last_query().is_none(),
                "a dimensionless field cannot size a query, so nothing is searched"
            );
        }
    }

    #[tokio::test]
    async fn scan_collection_scans_through_the_first_dimensional_field() {
        let store = ScriptedVectorStore::new(
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
                    dim: None,
                    ..FieldSchema::default()
                },
                FieldSchema {
                    name: "dense".to_owned(),
                    dtype: VectorDataType::FloatVector,
                    dim: Some(2),
                    ..FieldSchema::default()
                },
            ]),
            vec![embedding_doc("a", [1.0, 0.0, 0.0])],
        );
        let docs = scan_collection(&store, "uid_a_gid_b_mtype_coding")
            .await
            .expect("scan");
        assert_eq!(docs.len(), 1);
        let query = store.last_query().expect("one query");
        assert_eq!(query.vector_field.as_deref(), Some("dense"));
        assert_eq!(query.query_vector, vec![0.0, 0.0]);
        assert_eq!(query.top_k, SCAN_TOP_K);
        assert_eq!(query.filters, Value::Null);
    }

    #[test]
    fn vector_field_picks_the_first_dimensional_vector() {
        assert_eq!(vector_field(&embedding_schema()), Some(("embedding", 3)));
        let dimensionless = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_owned(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_owned(),
                dtype: VectorDataType::FloatVector,
                dim: None,
                ..FieldSchema::default()
            },
        ]);
        assert_eq!(vector_field(&dimensionless), None);

        let zero_dimension = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_owned(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_owned(),
                dtype: VectorDataType::FloatVector,
                dim: Some(0),
                ..FieldSchema::default()
            },
        ]);
        assert_eq!(
            vector_field(&zero_dimension),
            None,
            "a zero dimension sizes a zero-length query and is rejected"
        );

        let mixed = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_owned(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_owned(),
                dtype: VectorDataType::FloatVector,
                dim: None,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "dense".to_owned(),
                dtype: VectorDataType::FloatVector,
                dim: Some(2),
                ..FieldSchema::default()
            },
        ]);
        assert_eq!(vector_field(&mixed), Some(("dense", 2)));
        assert_eq!(
            scan_vector_field("mixed", &mixed).expect("searchable"),
            ("dense", 2)
        );
        let reason = unsearchable_reason("mixed", &dimensionless);
        assert!(reason.contains("mixed"), "{reason}");
        assert!(reason.contains("recreate the collection"), "{reason}");
        assert!(matches!(
            scan_vector_field("mixed", &dimensionless),
            Err(Error::Validation(_))
        ));
    }

    // Keep the journal types exercised from this module too, so a change to the
    // re-export surface breaks a test beside the planner rather than only elsewhere.
    #[test]
    fn journal_and_run_types_are_reachable_from_the_plan_module() {
        let mut journal = MigrationJournal::new();
        journal.record(OperationRecord {
            operation_id: "sql.one".to_owned(),
            attempts: 1,
            outcome: OperationOutcome::Succeeded,
            error: None,
            affected: 2,
            skipped: 0,
            failed: 0,
        });
        let run = MigrationRun {
            plan: MigrationPlan::new(vec![step("sql.one", "sql")]),
            journal,
            reports: vec![OperationReport::succeeded("sql.one", 2, 0, 0)],
        };
        assert_eq!(run.executed(), 1);
        assert_eq!(run.counts().affected, 2);
    }
}
