//! SQL-family migrator: legacy variable rows into the current layout, plus the `scope_user_mapping` backfill.
//!
//! Python's `openjiuwen.core.memory.migration.migrator.sql_migrator.SQLMigrator` owns the
//! relational half of a memory migration: it runs Alembic-style column operations
//! (add/rename/retype) on the per-table message and session-variable stores and registers
//! them in its operation registry.
//!
//! This port has no per-table relational schema to alter, so the family is defined by the
//! work that is actually left:
//!
//! - **The schema half lives in [`super::runner`].** Tables are created by the versioned DDL
//!   of [`super::registry`] before the backends open the database, so this migrator ships no
//!   column DDL at all.
//! - **The per-table stores became key-value families.** Python's session-variable table and
//!   message table map to [`KvFamily::SessionVariables`], [`KvFamily::UserVariables`] and the
//!   `messages` document collection (see [`crate::store::keys`]), so the relational work that
//!   remains is moving rows from a historical prefix into the current one.
//! - **`scope_user_mapping/{scope_id}/{user_id}` is a KV family**, so Python's scope→user
//!   table backfill is a data migration like any other; it is derived from the record
//!   keyspace, exactly as the Python migrator derives it from the message table.
//!
//! [`SqlMigrator`] plans at most three steps, always in this order:
//!
//! | operation | converts |
//! | --- | --- |
//! | [`COPY_USER_VARIABLES`] | `user_variable/…` rows into the current `user_var/…` prefix |
//! | [`COPY_SESSION_VARIABLES`] | `session_variable/…` rows into the current `session_var/…` prefix |
//! | [`BACKFILL_SCOPE_USER_MAPPING`] | `{user}/{scope}/{id}` record keys into `scope_user_mapping` rows |
//!
//! # Planning
//!
//! Planning is a read-only scan. A copy step is planned only when its family has at least one
//! historical alias (see [`KvPrefixRegistry`]) **and** that alias namespace holds at least one
//! key; the backfill step is planned only when it would insert a row or report a key it could
//! not parse. [`Migrator::is_applicable`] is therefore literally "the plan is not empty", which
//! costs a second read-only scan and is documented as such.
//!
//! Per step the counters mean:
//!
//! - `scanned` — legacy keys found (copy steps) or record keys found (backfill step),
//! - `affected` — rows to copy, or mapping pairs to insert,
//! - `skipped` — rows the current layout already holds (the current value wins), keys that
//!   cannot be rewritten, or pairs that are already mapped,
//! - `failed` — record keys that are not `{user}/{scope}/{id}`, or ids the mapping store
//!   rejects. They never abort the plan: each one becomes a [`MigrationIssue`] bounded by
//!   [`MigrationContext::max_issues`].
//!
//! # Applying
//!
//! Every handler recomputes its work from [`MigrationContext::prefixes`] and the live store and
//! never trusts [`MigrationStep::params`] for key derivation — the parameters are informational
//! (`{"sources": [...], "target": "..."}`, or `{"records_prefixes": [...],
//! "mapping_prefix": "..."}`) so a persisted plan stays readable. The handlers work on dry runs
//! too: the registry short-circuits a dry run before calling them, but a direct call must not
//! write anything, so each one honours [`MigrationContext::is_dry_run`] itself.
//!
//! Idempotency: a copy step deletes every legacy key it converted, so once it succeeded its
//! alias namespaces are empty and a rerun plans no copy step; the backfill finds every pair
//! already mapped, so it plans no step either.
//!
//! [`MigrationIssue`]: super::plan::MigrationIssue
//! [`KvPrefixRegistry`]: crate::store::prefix::KvPrefixRegistry

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::KvValue;

use super::operation::{MigrationOperationHandler, OperationOutcome, OperationReport};
use super::plan::{MigrationContext, MigrationCounts, MigrationIssue, MigrationStep, Migrator};
use crate::store::keys;
use crate::store::mapping::ScopeUserMappingStore;
use crate::store::prefix::KvFamily;

/// Migrator id.
pub const MIGRATOR_ID: &str = "sql";

/// Migrator schema version.
pub const MIGRATOR_VERSION: u32 = 1;

/// Operation id: legacy user-variable rows into the current prefix.
pub const COPY_USER_VARIABLES: &str = "sql.copy_user_variables";

/// Operation id: legacy session-variable rows into the current prefix.
pub const COPY_SESSION_VARIABLES: &str = "sql.copy_session_variables";

/// Operation id: backfill `scope_user_mapping` from the record keyspace.
pub const BACKFILL_SCOPE_USER_MAPPING: &str = "sql.backfill_scope_user_mapping";

/// The SQL-family migrator.
///
/// It is stateless: the stores, the prefix layout and the run options all arrive through the
/// [`MigrationContext`], so one instance is reused for every run and is cheap to share.
#[derive(Debug, Clone, Default)]
pub struct SqlMigrator;

impl SqlMigrator {
    /// Builds the migrator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Migrator for SqlMigrator {
    fn id(&self) -> &'static str {
        MIGRATOR_ID
    }

    fn version(&self) -> u32 {
        MIGRATOR_VERSION
    }

    fn description(&self) -> &'static str {
        "copy legacy variable rows into the current prefixes and backfill the scope-user mapping"
    }

    /// Whether the injected stores hold a legacy row or an unmapped record.
    ///
    /// This is a read-only double scan: it plans the work and reports whether the plan is
    /// empty, which keeps "applicable" and "has steps" from drifting apart.
    async fn is_applicable(&self, context: &MigrationContext) -> Result<bool> {
        Ok(!self.plan(context).await?.is_empty())
    }

    async fn plan(&self, context: &MigrationContext) -> Result<Vec<MigrationStep>> {
        let mut steps = Vec::new();
        for (operation_id, family) in [
            (COPY_USER_VARIABLES, KvFamily::UserVariables),
            (COPY_SESSION_VARIABLES, KvFamily::SessionVariables),
        ] {
            if let Some(step) = plan_variables(context, operation_id, family).await? {
                steps.push(step);
            }
        }
        if let Some(step) = plan_mapping(context).await? {
            steps.push(step);
        }
        Ok(steps)
    }

    fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>> {
        vec![
            Arc::new(CopyVariables::new(
                COPY_USER_VARIABLES,
                KvFamily::UserVariables,
            )),
            Arc::new(CopyVariables::new(
                COPY_SESSION_VARIABLES,
                KvFamily::SessionVariables,
            )),
            Arc::new(BackfillScopeUserMapping),
        ]
    }
}

/// One legacy row that is to be rewritten into the current layout.
struct LegacyRow {
    /// The key under the historical prefix.
    key: String,
    /// The key the same row has in the current layout.
    current_key: String,
    /// The stored value, moved to `current_key` when it is copied.
    value: KvValue,
}

/// What a scan of one variable family found.
#[derive(Default)]
struct RowScan {
    /// Rows that can be rewritten, in alias then key order.
    rows: Vec<LegacyRow>,
    /// Legacy keys that cannot be rewritten.
    issues: Vec<MigrationIssue>,
    /// Number of legacy keys seen, rewritable or not.
    scanned: usize,
}

/// Every legacy row of `family`, rewritten to its current-layout key.
///
/// Each alias prefix is scanned as `{alias}/` (the trailing separator keeps `user_var` from
/// matching the neighbouring `user_variable` namespace); a key without a suffix after its alias
/// is reported as an issue instead of being rewritten.
async fn legacy_rows(context: &MigrationContext, family: KvFamily) -> Result<RowScan> {
    let target = context.prefixes.current(family).to_owned();
    let mut scan = RowScan::default();
    for alias in context.prefixes.aliases(family) {
        let prefix = format!("{alias}/");
        for (key, value) in context.kv().get_by_prefix(&prefix).await? {
            scan.scanned += 1;
            let rewritten = key
                .strip_prefix(prefix.as_str())
                .filter(|suffix| !suffix.is_empty())
                .map(|suffix| format!("{target}/{suffix}"));
            match rewritten {
                Some(current_key) => scan.rows.push(LegacyRow {
                    key,
                    current_key,
                    value,
                }),
                None => scan.issues.push(MigrationIssue {
                    key,
                    reason: format!("has no suffix to rewrite under the current prefix {target:?}"),
                }),
            }
        }
    }
    Ok(scan)
}

/// Whether the current layout already holds data for `current_key`.
///
/// `written` carries the keys an earlier legacy row would create in the same pass, so a dry run
/// classifies rows exactly like a real run (where the earlier copy is already stored).
async fn already_present(
    context: &MigrationContext,
    current_key: &str,
    written: &BTreeSet<String>,
) -> Result<bool> {
    if written.contains(current_key) {
        return Ok(true);
    }
    Ok(context.kv().get(current_key).await?.is_some())
}

/// Plans the copy step of one variable family, or `None` when there is nothing to copy.
async fn plan_variables(
    context: &MigrationContext,
    operation_id: &'static str,
    family: KvFamily,
) -> Result<Option<MigrationStep>> {
    let aliases = context.prefixes.aliases(family);
    if aliases.is_empty() {
        return Ok(None);
    }
    let scan = legacy_rows(context, family).await?;
    if scan.scanned == 0 {
        return Ok(None);
    }
    let target = context.prefixes.current(family).to_owned();
    let mut counts = MigrationCounts {
        scanned: scan.scanned,
        skipped: scan.issues.len(),
        ..MigrationCounts::default()
    };
    let mut written: BTreeSet<String> = BTreeSet::new();
    for row in &scan.rows {
        if already_present(context, &row.current_key, &written).await? {
            counts.skipped += 1;
        } else {
            written.insert(row.current_key.clone());
            counts.affected += 1;
        }
    }
    let mut step = MigrationStep::new(
        operation_id,
        MIGRATOR_ID,
        MIGRATOR_VERSION,
        format!(
            "copy legacy `{}` rows into the current `{target}` prefix",
            family.label()
        ),
    )
    .with_prefixes(aliases.first().cloned(), Some(target.clone()))
    .with_params(json!({"sources": aliases, "target": target}))
    .with_counts(counts);
    for issue in scan.issues {
        step.push_issue(issue.key, issue.reason, context.max_issues);
    }
    Ok(Some(step))
}

/// Copies every legacy key of `family` into the current prefix, then removes it.
///
/// The current layout wins: a row whose current key already exists is reported `skipped` and
/// only its legacy duplicate is deleted. A dry run counts the same rows and writes nothing.
struct CopyVariables {
    /// The id this handler answers to.
    operation_id: &'static str,
    /// The family whose rows it moves.
    family: KvFamily,
}

impl CopyVariables {
    /// Builds the handler for `operation_id` over `family`.
    fn new(operation_id: &'static str, family: KvFamily) -> Self {
        Self {
            operation_id,
            family,
        }
    }
}

#[async_trait]
impl MigrationOperationHandler for CopyVariables {
    fn operation_id(&self) -> &'static str {
        self.operation_id
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let scan = legacy_rows(context, self.family).await?;
        if scan.scanned == 0 {
            return Ok(OperationReport::skipped(
                self.operation_id,
                "no legacy rows to migrate",
            ));
        }
        let dry_run = context.is_dry_run();
        let unrewritable = scan.issues.len();
        let mut affected = 0;
        let mut skipped = unrewritable;
        let mut written: BTreeSet<String> = BTreeSet::new();
        for row in scan.rows {
            if already_present(context, &row.current_key, &written).await? {
                skipped += 1;
                if !dry_run {
                    context.kv().delete(&row.key).await?;
                }
                continue;
            }
            affected += 1;
            if dry_run {
                written.insert(row.current_key);
                continue;
            }
            context.kv().set(&row.current_key, row.value).await?;
            context.kv().delete(&row.key).await?;
        }
        tracing::info!(
            target: "openjiuwen_memory::store",
            operation = self.operation_id,
            family = self.family.label(),
            affected,
            skipped,
            dry_run,
            "migrated legacy variable rows into the current layout"
        );
        Ok(OperationReport {
            operation_id: self.operation_id.to_owned(),
            outcome: OperationOutcome::Succeeded,
            affected,
            skipped,
            failed: 0,
            note: issue_note(unrewritable),
        })
    }
}

/// A note naming the legacy keys a handler could not rewrite, when there were any.
fn issue_note(unrewritable: usize) -> Option<String> {
    (unrewritable > 0).then(|| format!("{unrewritable} legacy key(s) could not be rewritten"))
}

/// What one scan of the record keyspace found.
#[derive(Default)]
struct RecordScan {
    /// Distinct `(user, scope)` pairs, in key order.
    pairs: BTreeSet<(String, String)>,
    /// Record keys that are not `{user}/{scope}/{id}`.
    issues: Vec<MigrationIssue>,
    /// Number of record keys seen.
    scanned: usize,
}

/// The record namespaces to scan: the current one, then every alias in registration order.
fn record_scan_prefixes(context: &MigrationContext) -> Vec<String> {
    let current = keys::record_prefix_for_scope_in(&context.prefixes, "");
    let mut prefixes = vec![current.clone()];
    prefixes.extend(keys::legacy_alternatives_in(&context.prefixes, &current));
    prefixes
}

/// Reads every record key of the current layout and of each historical alias.
async fn scan_records(context: &MigrationContext) -> Result<RecordScan> {
    let mut scan = RecordScan::default();
    for prefix in record_scan_prefixes(context) {
        for key in context.kv().get_by_prefix(&prefix).await?.into_keys() {
            scan.scanned += 1;
            match record_pair(&prefix, &key) {
                Some((user, scope)) => {
                    scan.pairs.insert((user.to_owned(), scope.to_owned()));
                }
                None => scan.issues.push(MigrationIssue {
                    key,
                    reason: format!("is not `{{user}}/{{scope}}/{{id}}` after {prefix:?}"),
                }),
            }
        }
    }
    Ok(scan)
}

/// The `(user, scope)` pair of one record key, or `None` when the key is malformed.
///
/// The id is whatever follows the scope, so an id containing `/` still parses; the user and
/// scope must both be present and non-empty.
fn record_pair<'a>(prefix: &str, key: &'a str) -> Option<(&'a str, &'a str)> {
    let rest = key.strip_prefix(prefix)?;
    let mut segments = rest.splitn(3, '/');
    let user = segments.next()?;
    let scope = segments.next()?;
    let id = segments.next()?;
    (!user.is_empty() && !scope.is_empty() && !id.is_empty()).then_some((user, scope))
}

/// A mapping store over the context's key-value backend and prefix layout.
fn mapping_store(context: &MigrationContext) -> ScopeUserMappingStore {
    ScopeUserMappingStore::from_stores(&context.stores).with_prefixes(context.prefixes.clone())
}

/// Plans the `scope_user_mapping` backfill, or `None` when every pair is already mapped.
async fn plan_mapping(context: &MigrationContext) -> Result<Option<MigrationStep>> {
    let scan = scan_records(context).await?;
    let store = mapping_store(context);
    let mut counts = MigrationCounts {
        scanned: scan.scanned,
        failed: scan.issues.len(),
        ..MigrationCounts::default()
    };
    let mut issues = scan.issues;
    for (user, scope) in scan.pairs {
        match store.contains(&user, &scope).await {
            Ok(true) => counts.skipped += 1,
            Ok(false) => counts.affected += 1,
            Err(Error::Validation(reason)) => {
                counts.failed += 1;
                issues.push(MigrationIssue {
                    key: format!("{user}/{scope}"),
                    reason,
                });
            }
            Err(error) => return Err(error),
        }
    }
    if counts.affected == 0 && counts.failed == 0 {
        return Ok(None);
    }
    let mapping_prefix = context
        .prefixes
        .current(KvFamily::ScopeUserMapping)
        .to_owned();
    let prefixes = record_scan_prefixes(context);
    let source_prefix = prefixes.first().cloned();
    let mut step = MigrationStep::new(
        BACKFILL_SCOPE_USER_MAPPING,
        MIGRATOR_ID,
        MIGRATOR_VERSION,
        "backfill the scope-user mapping from the record keyspace",
    )
    .with_prefixes(source_prefix, Some(mapping_prefix.clone()))
    .with_params(json!({"records_prefixes": prefixes, "mapping_prefix": mapping_prefix}))
    .with_counts(counts);
    for issue in issues {
        step.push_issue(issue.key, issue.reason, context.max_issues);
    }
    Ok(Some(step))
}

/// Inserts the `scope_user_mapping` row of every distinct `(user, scope)` pair of the records.
///
/// The write is the mapping store's conditional `add`, so an already mapped pair is `skipped`
/// and never overwritten. An id the store rejects is an issue, never a fatal error; a dry run
/// asks the store instead of writing and reports the same counts.
struct BackfillScopeUserMapping;

#[async_trait]
impl MigrationOperationHandler for BackfillScopeUserMapping {
    fn operation_id(&self) -> &'static str {
        BACKFILL_SCOPE_USER_MAPPING
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let scan = scan_records(context).await?;
        if scan.scanned == 0 {
            return Ok(OperationReport::skipped(
                BACKFILL_SCOPE_USER_MAPPING,
                "no record keys to map",
            ));
        }
        let store = mapping_store(context);
        let dry_run = context.is_dry_run();
        let mut issues = scan.issues;
        let mut affected = 0_usize;
        let mut skipped = 0_usize;
        let mut failed = issues.len();
        for (user, scope) in scan.pairs {
            let outcome = if dry_run {
                store.contains(&user, &scope).await.map(|exists| !exists)
            } else {
                store.add(&user, &scope).await
            };
            match outcome {
                Ok(true) => affected += 1,
                Ok(false) => skipped += 1,
                Err(Error::Validation(reason)) => {
                    failed += 1;
                    issues.push(MigrationIssue {
                        key: format!("{user}/{scope}"),
                        reason,
                    });
                }
                Err(error) => return Err(error),
            }
        }
        tracing::info!(
            target: "openjiuwen_memory::store",
            operation = BACKFILL_SCOPE_USER_MAPPING,
            affected,
            skipped,
            failed,
            dry_run,
            "backfilled the scope-user mapping from the record keyspace"
        );
        Ok(OperationReport {
            operation_id: BACKFILL_SCOPE_USER_MAPPING.to_owned(),
            outcome: OperationOutcome::Succeeded,
            affected,
            skipped,
            failed,
            note: (!issues.is_empty())
                .then(|| format!("{} record key(s) could not be mapped", issues.len())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::record::MemoryScope;
    use crate::store::migration::{MigrationJournal, MigrationPlan, OperationRegistry};
    use crate::store::prefix::KvPrefixRegistry;
    use crate::store::MemoryStores;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn stores() -> MemoryStores {
        MemoryStores::in_memory()
    }

    async fn seed(stores: &MemoryStores, key: &str, value: &str) {
        stores
            .kv
            .set(key, KvValue::text(value))
            .await
            .expect("seed key");
    }

    async fn read(stores: &MemoryStores, key: &str) -> Option<String> {
        stores
            .kv
            .get(key)
            .await
            .expect("read key")
            .and_then(|value| value.as_str().map(str::to_owned))
    }

    async fn keys_under(stores: &MemoryStores, prefix: &str) -> Vec<String> {
        stores
            .kv
            .get_by_prefix(prefix)
            .await
            .expect("scan prefix")
            .into_keys()
            .collect()
    }

    fn step_of<'a>(steps: &'a [MigrationStep], operation_id: &str) -> &'a MigrationStep {
        steps
            .iter()
            .find(|step| step.operation_id == operation_id)
            .expect("planned step")
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
    fn migrator_identity_is_stable() {
        let migrator = SqlMigrator::new();
        assert_eq!(migrator.id(), MIGRATOR_ID);
        assert_eq!(migrator.version(), MIGRATOR_VERSION);
        assert!(migrator.description().contains("scope-user mapping"));
    }

    #[tokio::test]
    async fn an_empty_store_plans_nothing() {
        let context = MigrationContext::new(stores());
        let migrator = SqlMigrator::new();
        assert_eq!(migrator.plan(&context).await.expect("plan"), Vec::new());
        assert!(!migrator
            .is_applicable(&context)
            .await
            .expect("applicability"));
    }

    #[tokio::test]
    async fn legacy_variable_rows_are_copied_and_the_current_layout_wins() {
        let stores = stores();
        let scope = scope();
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        seed(&stores, &keys::legacy_variable_key(&scope, "lang"), "rust").await;
        seed(&stores, &keys::variable_key(&scope, "theme"), "light").await;
        seed(
            &stores,
            &keys::legacy_session_variable_key(&scope, "s1", "step"),
            "2",
        )
        .await;

        let context = MigrationContext::new(stores.clone());
        let migrator = SqlMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(
            steps
                .iter()
                .map(|step| step.operation_id.as_str())
                .collect::<Vec<_>>(),
            vec![COPY_USER_VARIABLES, COPY_SESSION_VARIABLES]
        );

        let user_step = step_of(&steps, COPY_USER_VARIABLES);
        assert_eq!(user_step.counts, counts(2, 1, 1, 0));
        assert_eq!(user_step.migrator, MIGRATOR_ID);
        assert_eq!(user_step.version, MIGRATOR_VERSION);
        assert_eq!(user_step.source_prefix.as_deref(), Some("user_variable"));
        assert_eq!(user_step.target_prefix.as_deref(), Some("user_var"));
        assert_eq!(
            user_step.params,
            json!({"sources": ["user_variable"], "target": "user_var"})
        );
        assert_eq!(user_step.issues, Vec::new());

        let session_step = step_of(&steps, COPY_SESSION_VARIABLES);
        assert_eq!(session_step.counts, counts(1, 1, 0, 0));
        assert_eq!(
            session_step.source_prefix.as_deref(),
            Some("session_variable")
        );
        assert_eq!(session_step.target_prefix.as_deref(), Some("session_var"));

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(run.is_success());
        assert_eq!(run.executed(), 2);
        assert_eq!(run.counts().affected, 2);
        assert_eq!(run.counts().skipped, 1);

        // The migrated rows are in the current layout, the current value wins, and every
        // legacy key is gone.
        assert_eq!(
            read(&stores, "user_var/alice/ws/lang").await.as_deref(),
            Some("rust")
        );
        assert_eq!(
            read(&stores, "user_var/alice/ws/theme").await.as_deref(),
            Some("light")
        );
        assert_eq!(
            read(&stores, "session_var/alice/ws/s1/step")
                .await
                .as_deref(),
            Some("2")
        );
        assert_eq!(
            keys_under(&stores, "user_variable/").await,
            Vec::<String>::new()
        );
        assert_eq!(
            keys_under(&stores, "session_variable/").await,
            Vec::<String>::new()
        );

        // A rerun finds no legacy row and plans no step.
        assert_eq!(migrator.plan(&context).await.expect("replan"), Vec::new());
        assert!(!migrator
            .is_applicable(&context)
            .await
            .expect("applicability"));
    }

    #[tokio::test]
    async fn a_dry_run_reports_the_same_counts_and_writes_nothing() {
        let stores = stores();
        let scope = scope();
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        seed(&stores, &keys::legacy_variable_key(&scope, "lang"), "rust").await;
        seed(&stores, &keys::variable_key(&scope, "theme"), "light").await;

        let context = MigrationContext::new(stores.clone());
        let dry = context.clone().with_dry_run(true);
        let migrator = SqlMigrator::new();
        let planned = migrator.plan(&context).await.expect("plan");
        let planned_dry = migrator.plan(&dry).await.expect("dry plan");
        assert_eq!(
            planned.iter().map(|step| step.counts).collect::<Vec<_>>(),
            planned_dry
                .iter()
                .map(|step| step.counts)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            step_of(&planned_dry, COPY_USER_VARIABLES).counts,
            counts(2, 1, 1, 0)
        );

        // Applying a dry-run plan through the registry invokes no handler.
        let plan = MigrationPlan::new(planned_dry.clone()).with_dry_run(true);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&dry, &plan, &mut journal)
            .await
            .expect("dry apply");
        assert_eq!(run.skipped(), 1);
        assert_eq!(keys_under(&stores, "user_variable/").await.len(), 2);
        assert_eq!(keys_under(&stores, "user_var/").await.len(), 1);

        // A handler called directly on a dry context reports the same work and writes nothing.
        let handler = migrator
            .operations()
            .into_iter()
            .find(|handler| handler.operation_id() == COPY_USER_VARIABLES)
            .expect("copy handler");
        let report = handler
            .execute(&dry, step_of(&planned_dry, COPY_USER_VARIABLES))
            .await
            .expect("dry execute");
        assert_eq!(report.outcome, OperationOutcome::Succeeded);
        assert_eq!(report.affected, 1);
        assert_eq!(report.skipped, 1);
        assert_eq!(keys_under(&stores, "user_variable/").await.len(), 2);
        assert_eq!(keys_under(&stores, "user_var/").await.len(), 1);
        assert_eq!(
            read(&stores, "user_var/alice/ws/theme").await.as_deref(),
            Some("light")
        );
    }

    #[tokio::test]
    async fn the_mapping_backfill_collects_every_distinct_pair() {
        let stores = stores();
        seed(&stores, "UMD/alice/ws/r1", "{}").await;
        seed(&stores, "UMD/alice/ws/r2", "{}").await;
        seed(&stores, "UMD/bob/ws/r3", "{}").await;
        seed(
            &stores,
            &keys::legacy_record_key(&MemoryScope::new("carol", "ws"), "r4"),
            "{}",
        )
        .await;

        let context = MigrationContext::new(stores.clone());
        let migrator = SqlMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.operation_id, BACKFILL_SCOPE_USER_MAPPING);
        assert_eq!(step.counts, counts(4, 3, 0, 0));
        assert_eq!(step.source_prefix.as_deref(), Some("UMD/"));
        assert_eq!(step.target_prefix.as_deref(), Some("scope_user_mapping"));
        assert_eq!(
            step.params,
            json!({
                "records_prefixes": ["UMD/", "user_mem/"],
                "mapping_prefix": "scope_user_mapping",
            })
        );
        assert_eq!(step.issues, Vec::new());

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(run.is_success());
        assert_eq!(run.reports[0].affected, 3);

        let mapping = ScopeUserMappingStore::from_stores(&stores);
        assert_eq!(
            mapping.users_for_scope("ws").await.expect("users"),
            vec!["alice".to_owned(), "bob".to_owned(), "carol".to_owned()]
        );

        // Every pair is mapped now, so a rerun plans no step.
        assert_eq!(migrator.plan(&context).await.expect("replan"), Vec::new());
    }

    #[tokio::test]
    async fn a_malformed_record_key_is_an_issue_and_never_stops_the_plan() {
        let stores = stores();
        seed(&stores, "UMD/alice/r1", "{}").await;
        seed(&stores, "UMD/alice/ws/", "{}").await;
        seed(&stores, "UMD/dana/ws/r2", "{}").await;

        let context = MigrationContext::new(stores.clone());
        let migrator = SqlMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.operation_id, BACKFILL_SCOPE_USER_MAPPING);
        assert_eq!(step.counts, counts(3, 1, 0, 2));
        assert_eq!(step.issues.len(), 2);
        assert_eq!(step.issues[0].key, "UMD/alice/r1");
        assert!(step.issues[0].reason.contains("UMD/"));

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports[0].failed, 2);
        assert!(run.reports[0]
            .note
            .as_deref()
            .is_some_and(|note| note.contains("2 record key(s)")));

        let mapping = ScopeUserMappingStore::from_stores(&stores);
        assert_eq!(
            mapping.users_for_scope("ws").await.expect("users"),
            vec!["dana".to_owned()]
        );
    }

    #[tokio::test]
    async fn a_renamed_family_migrates_every_alias_into_the_new_prefix() {
        let stores = stores();
        let scope = scope();
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        seed(&stores, &keys::variable_key(&scope, "lang"), "rust").await;

        let prefixes = KvPrefixRegistry::new()
            .with_current(KvFamily::UserVariables, "vars_v2")
            .expect("valid prefix");
        assert_eq!(
            prefixes.aliases(KvFamily::UserVariables),
            ["user_var", "user_variable"]
        );
        let context = MigrationContext::new(stores.clone()).with_prefixes(prefixes);

        let migrator = SqlMigrator::new();
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.counts, counts(2, 2, 0, 0));
        assert_eq!(step.source_prefix.as_deref(), Some("user_var"));
        assert_eq!(step.target_prefix.as_deref(), Some("vars_v2"));
        assert_eq!(
            step.params,
            json!({"sources": ["user_var", "user_variable"], "target": "vars_v2"})
        );

        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert!(run.is_success());
        assert_eq!(
            read(&stores, "vars_v2/alice/ws/theme").await.as_deref(),
            Some("dark")
        );
        assert_eq!(
            read(&stores, "vars_v2/alice/ws/lang").await.as_deref(),
            Some("rust")
        );
        assert_eq!(keys_under(&stores, "user_var/").await, Vec::<String>::new());
        assert_eq!(
            keys_under(&stores, "user_variable/").await,
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn operations_are_unique_and_execute_end_to_end() {
        let migrator = SqlMigrator::new();
        let operation_ids: Vec<&str> = migrator
            .operations()
            .iter()
            .map(|handler| handler.operation_id())
            .collect();
        assert_eq!(
            operation_ids,
            vec![
                COPY_USER_VARIABLES,
                COPY_SESSION_VARIABLES,
                BACKFILL_SCOPE_USER_MAPPING
            ]
        );

        let mut registry = OperationRegistry::new();
        registry
            .register_all(migrator.operations())
            .expect("unique operation ids");
        assert_eq!(registry.len(), 3);
        for operation_id in &operation_ids {
            assert!(registry.contains(operation_id));
        }

        let stores = stores();
        let scope = scope();
        seed(
            &stores,
            &keys::legacy_session_variable_key(&scope, "s1", "step"),
            "2",
        )
        .await;
        seed(&stores, "UMD/alice/ws/r1", "{}").await;

        let context = MigrationContext::new(stores.clone());
        let steps = migrator.plan(&context).await.expect("plan");
        assert_eq!(steps.len(), 2);
        let plan = MigrationPlan::new(steps);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.executed(), 2);
        assert_eq!(run.plan.len(), 2);
        assert_eq!(
            read(&stores, "session_var/alice/ws/s1/step")
                .await
                .as_deref(),
            Some("2")
        );
        let mapping = ScopeUserMappingStore::from_stores(&stores);
        assert_eq!(
            mapping.users_for_scope("ws").await.expect("users"),
            vec!["alice".to_owned()]
        );

        // The journal is the resume token: a second apply executes nothing.
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("resume");
        assert_eq!(run.reports.len(), 2);
        assert!(run
            .reports
            .iter()
            .all(|report| report.outcome == OperationOutcome::Skipped));
    }

    #[tokio::test]
    async fn a_local_file_store_round_trips_the_migrated_layout() {
        let dir = tempfile::tempdir().expect("temp dir");
        let stores = MemoryStores::local_file(dir.path()).await.expect("stores");
        let scope = scope();
        seed(&stores, &keys::legacy_variable_key(&scope, "theme"), "dark").await;
        seed(&stores, "UMD/alice/ws/r1", "{}").await;

        let context = MigrationContext::new(stores.clone());
        let migrator = SqlMigrator::new();
        let plan = MigrationPlan::new(migrator.plan(&context).await.expect("plan"));
        let mut journal = MigrationJournal::new();
        migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");

        let reopened = MemoryStores::local_file(dir.path())
            .await
            .expect("reopen stores");
        assert_eq!(
            read(&reopened, "user_var/alice/ws/theme").await.as_deref(),
            Some("dark")
        );
        let mapping = ScopeUserMappingStore::from_stores(&reopened);
        assert_eq!(
            mapping.users_for_scope("ws").await.expect("users"),
            vec!["alice".to_owned()]
        );
    }
}
