//! Named migration operations, the `OperationRegistry`, and the execution journal.
//!
//! A [`MigrationOperationHandler`] is the executable half of a migration step: it is
//! registered under a stable id (for example `sql.copy_user_variables`), it receives the
//! [`MigrationContext`] and the concrete [`MigrationStep`] it was planned from, and it
//! reports exactly one [`OperationReport`].
//!
//! The [`OperationRegistry`] owns the executable plan:
//!
//! - **Validate first.** Every step id must be unique in the plan and registered, so a
//!   typo fails the whole run before any handler executes.
//! - **Journal-driven resume.** A [`MigrationJournal`] keyed by operation id is the
//!   durable state of a run. A step whose record already `Succeeded` is not executed
//!   again (reported `Skipped` with the note `already completed`), while a `Failed` or
//!   `Skipped` record is retried and its attempt counter grows. The journal therefore is
//!   the resume token: pass the same journal to continue an interrupted migration.
//! - **Dry runs write nothing.** When either the plan or the context is a dry run, every
//!   step is journalled `Skipped` with the note `dry run` and no handler is invoked.
//! - **Failures are recorded, then propagated.** A handler error records a `Failed`
//!   journal entry with the error message before the error is returned with the
//!   operation id in its context, so a caller can inspect — or retry from — the journal.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::plan::{MigrationContext, MigrationCounts, MigrationPlan, MigrationStep};

/// Maximum accepted length of an operation id, in characters.
const MAX_OPERATION_ID_LENGTH: usize = 128;

/// How one operation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationOutcome {
    /// The operation converted everything it was asked to.
    Succeeded,
    /// The operation raised an error; see [`OperationRecord::error`].
    Failed,
    /// The operation deliberately did nothing (dry run, already completed, or a
    /// reported [`OperationReport::skipped`]).
    Skipped,
}

/// The journal entry of one operation: attempts and outcome.
///
/// A record is keyed by [`Self::operation_id`] in a [`MigrationJournal`] and holds the
/// content of [`OperationReport`] plus the attempt counter, so a resumed run knows what
/// happened before it started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationRecord {
    /// Id of the operation that was attempted.
    pub operation_id: String,
    /// How many times the operation was attempted, including the recorded attempt.
    #[serde(default)]
    pub attempts: u32,
    /// How the attempt ended.
    pub outcome: OperationOutcome,
    /// Message of the error that failed the attempt, when it failed.
    #[serde(default)]
    pub error: Option<String>,
    /// Items converted by the last attempt.
    #[serde(default)]
    pub affected: usize,
    /// Items deliberately skipped by the last attempt.
    #[serde(default)]
    pub skipped: usize,
    /// Items the last attempt failed on.
    #[serde(default)]
    pub failed: usize,
}

impl OperationRecord {
    /// Whether the recorded outcome is [`OperationOutcome::Succeeded`].
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.outcome == OperationOutcome::Succeeded
    }

    /// Whether a rerun should attempt the operation again.
    ///
    /// `Failed` and `Skipped` are both retryable: a failure may be transient, and a
    /// skipped operation (for example one left alone by a dry run) has still to be done.
    #[must_use]
    pub fn retryable(&self) -> bool {
        matches!(
            self.outcome,
            OperationOutcome::Failed | OperationOutcome::Skipped
        )
    }
}

/// Resume state of a migration run, keyed by operation id.
///
/// The journal is a plain value the caller owns and persists with
/// [`MigrationJournal::to_json`]; passing the same journal to a second
/// [`OperationRegistry::execute`] resumes the run instead of repeating work. Its
/// `BTreeMap` storage keeps iteration, [`MigrationJournal::completed`] and
/// [`MigrationJournal::failed`] sorted by id, so serialized output is stable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationJournal {
    /// One entry per attempted operation, by id.
    #[serde(default)]
    entries: BTreeMap<String, OperationRecord>,
}

impl MigrationJournal {
    /// Builds an empty journal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The record of `operation_id`, when it was attempted.
    #[must_use]
    pub fn get(&self, operation_id: &str) -> Option<&OperationRecord> {
        self.entries.get(operation_id)
    }

    /// Every record, sorted by operation id.
    pub fn records(&self) -> impl Iterator<Item = &OperationRecord> + '_ {
        self.entries.values()
    }

    /// Number of recorded operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no operation was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The ids of the operations that succeeded, sorted.
    #[must_use]
    pub fn completed(&self) -> Vec<String> {
        self.entries
            .values()
            .filter(|record| record.succeeded())
            .map(|record| record.operation_id.clone())
            .collect()
    }

    /// The records of the operations that failed, sorted by id.
    #[must_use]
    pub fn failed(&self) -> Vec<&OperationRecord> {
        self.entries
            .values()
            .filter(|record| record.outcome == OperationOutcome::Failed)
            .collect()
    }

    /// Whether `operation_id` is recorded as succeeded.
    #[must_use]
    pub fn succeeded(&self, operation_id: &str) -> bool {
        self.entries
            .get(operation_id)
            .is_some_and(OperationRecord::succeeded)
    }

    /// Records `record`, replacing any previous record of the same operation.
    pub fn record(&mut self, record: OperationRecord) {
        self.entries.insert(record.operation_id.clone(), record);
    }

    /// Drops every record, so the journal no longer resumes anything.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Serializes the journal to JSON.
    ///
    /// # Errors
    ///
    /// Returns the `serde_json` error (kind [`openjiuwen_common::error::ErrorKind::External`])
    /// annotated with the serialization context.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            Error::external(error).with_context("serializing the memory migration journal")
        })
    }

    /// Decodes a journal produced by [`MigrationJournal::to_json`].
    ///
    /// A malformed document is `Error::Validation`; omitted per-record counters decode to
    /// zero, and a missing `entries` map decodes to an empty journal.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("invalid memory migration journal JSON: {error}"))
        })
    }
}

/// What a handler did for one step.
///
/// A report is the handler's own account of the step, which the registry copies into the
/// journal. `affected`/`skipped`/`failed` mirror [`MigrationCounts`] and `note` carries a
/// human-readable reason for a skip or a failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationReport {
    /// Id of the executed operation.
    pub operation_id: String,
    /// How the operation ended.
    pub outcome: OperationOutcome,
    /// Items converted.
    #[serde(default)]
    pub affected: usize,
    /// Items deliberately skipped.
    #[serde(default)]
    pub skipped: usize,
    /// Items the operation failed on.
    #[serde(default)]
    pub failed: usize,
    /// Reason for a skip or a failure, when there is one.
    #[serde(default)]
    pub note: Option<String>,
}

impl OperationReport {
    /// Builds a successful report.
    #[must_use]
    pub fn succeeded(
        operation_id: impl Into<String>,
        affected: usize,
        skipped: usize,
        failed: usize,
    ) -> Self {
        Self {
            operation_id: operation_id.into(),
            outcome: OperationOutcome::Succeeded,
            affected,
            skipped,
            failed,
            note: None,
        }
    }

    /// Builds a report for a step that deliberately did nothing.
    #[must_use]
    pub fn skipped(operation_id: impl Into<String>, note: impl Into<String>) -> Self {
        Self {
            operation_id: operation_id.into(),
            outcome: OperationOutcome::Skipped,
            affected: 0,
            skipped: 0,
            failed: 0,
            note: Some(note.into()),
        }
    }

    /// Builds a report for a step that failed without raising an error.
    #[must_use]
    pub fn failed(operation_id: impl Into<String>, note: impl Into<String>) -> Self {
        Self {
            operation_id: operation_id.into(),
            outcome: OperationOutcome::Failed,
            affected: 0,
            skipped: 0,
            failed: 1,
            note: Some(note.into()),
        }
    }
}

/// The result of executing a plan: what was planned, what the journal says, and the
/// handler reports in execution order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationRun {
    /// The plan that was executed.
    pub plan: MigrationPlan,
    /// The journal after the run; pass it to the next run to resume.
    pub journal: MigrationJournal,
    /// One report per step that was not skipped as already completed, in order.
    #[serde(default)]
    pub reports: Vec<OperationReport>,
}

impl MigrationRun {
    /// Whether every recorded operation succeeded or was skipped.
    ///
    /// An empty journal is a success (nothing failed), which is what a fully completed
    /// rerun reports.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.journal
            .records()
            .all(|record| record.outcome != OperationOutcome::Failed)
    }

    /// Number of operations recorded as succeeded.
    #[must_use]
    pub fn executed(&self) -> usize {
        self.journal
            .records()
            .filter(|record| record.succeeded())
            .count()
    }

    /// Number of operations recorded as skipped.
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.journal
            .records()
            .filter(|record| record.outcome == OperationOutcome::Skipped)
            .count()
    }

    /// All journal counters merged.
    ///
    /// Every journalled attempt examined `affected + skipped + failed` items, so that sum
    /// is reported as `scanned`; the other fields are the per-record sums.
    #[must_use]
    pub fn counts(&self) -> MigrationCounts {
        let mut total = MigrationCounts::default();
        for record in self.journal.records() {
            total.scanned = total.scanned.saturating_add(
                record
                    .affected
                    .saturating_add(record.skipped)
                    .saturating_add(record.failed),
            );
            total.affected = total.affected.saturating_add(record.affected);
            total.skipped = total.skipped.saturating_add(record.skipped);
            total.failed = total.failed.saturating_add(record.failed);
        }
        total
    }

    /// Serializes the run to JSON.
    ///
    /// # Errors
    ///
    /// Returns the `serde_json` error (kind [`openjiuwen_common::error::ErrorKind::External`])
    /// annotated with the serialization context.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            Error::external(error).with_context("serializing the memory migration run")
        })
    }

    /// Decodes a run produced by [`MigrationRun::to_json`].
    ///
    /// A malformed document is `Error::Validation`; a missing `reports` list decodes to an
    /// empty one.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("invalid memory migration run JSON: {error}"))
        })
    }
}

/// One executable migration operation.
///
/// Handlers are stateless apart from their counters and their injected stores: everything
/// variable arrives through the [`MigrationContext`] and the [`MigrationStep`]. Returning
/// `Err` marks the step `Failed` and stops the run; returning
/// [`OperationReport::failed`] records a failure and lets the run continue.
#[async_trait]
pub trait MigrationOperationHandler: Send + Sync {
    /// Stable id, matching the `operation_id` of the steps this handler executes.
    ///
    /// It is `namespace.name`: non-empty, containing a `.`, with no whitespace or NUL,
    /// and at most 128 characters.
    fn operation_id(&self) -> &'static str;

    /// Executes `step`.
    ///
    /// # Errors
    ///
    /// Any store or conversion error; the registry records it in the journal and
    /// propagates it with the operation id in its context.
    async fn execute(
        &self,
        context: &MigrationContext,
        step: &MigrationStep,
    ) -> Result<OperationReport>;
}

/// Named operations, executed against a plan.
///
/// The registry is the only place step ids are resolved, so a plan is data and the
/// registry is the code that can execute it. Registration validates ids eagerly and
/// duplicate registration is rejected, so a registry never silently shadows a handler.
#[derive(Clone, Default)]
pub struct OperationRegistry {
    /// Handlers by operation id.
    handlers: BTreeMap<String, Arc<dyn MigrationOperationHandler>>,
}

impl OperationRegistry {
    /// Builds an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `handler` under its operation id.
    ///
    /// # Errors
    ///
    /// Returns a validation error naming the id when it is empty, longer than 128
    /// characters, contains whitespace or NUL, lacks the `.` namespace separator, or is
    /// already registered. The registry is left unchanged on error.
    pub fn register(&mut self, handler: Arc<dyn MigrationOperationHandler>) -> Result<()> {
        let operation_id = handler.operation_id();
        validate_operation_id(operation_id)?;
        if self.handlers.contains_key(operation_id) {
            return Err(Error::validation(format!(
                "memory migration operation {operation_id:?} is registered twice"
            )));
        }
        self.handlers.insert(operation_id.to_owned(), handler);
        Ok(())
    }

    /// Registers every handler of `handlers`, all or nothing.
    ///
    /// # Errors
    ///
    /// Returns the validation error of the first rejected handler; because the
    /// registrations are staged, the registry then holds exactly what it held before.
    pub fn register_all(
        &mut self,
        handlers: Vec<Arc<dyn MigrationOperationHandler>>,
    ) -> Result<()> {
        let mut staged = self.handlers.clone();
        for handler in handlers {
            let operation_id = handler.operation_id();
            validate_operation_id(operation_id)?;
            if staged.contains_key(operation_id) {
                return Err(Error::validation(format!(
                    "memory migration operation {operation_id:?} is registered twice"
                )));
            }
            staged.insert(operation_id.to_owned(), handler);
        }
        self.handlers = staged;
        Ok(())
    }

    /// Builder form of [`Self::register`].
    ///
    /// # Errors
    ///
    /// Returns the same validation error as [`Self::register`].
    pub fn with_handler(mut self, handler: Arc<dyn MigrationOperationHandler>) -> Result<Self> {
        self.register(handler)?;
        Ok(self)
    }

    /// The registered operation ids, sorted.
    #[must_use]
    pub fn operation_ids(&self) -> Vec<&str> {
        self.handlers.keys().map(String::as_str).collect()
    }

    /// Whether `operation_id` has a handler.
    #[must_use]
    pub fn contains(&self, operation_id: &str) -> bool {
        self.handlers.contains_key(operation_id)
    }

    /// Number of registered handlers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether no handler is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Executes every step of `plan` in order, journalling each attempt.
    ///
    /// Validation happens first: a duplicate step id or an unknown operation id is a
    /// validation error, and then nothing runs. After that, each step is
    ///
    /// - skipped without invoking its handler when the journal already records it as
    ///   succeeded (report note `already completed`, journal unchanged),
    /// - recorded as `Skipped` with the note `dry run` when `plan.dry_run` or
    ///   `context.dry_run` is set (no handler runs, nothing is written), and
    /// - executed otherwise; a `Failed` or `Skipped` journal record is retried, so its
    ///   attempt counter grows by one.
    ///
    /// The journal is updated even when a handler fails: the `Failed` record with the
    /// error message is in place before the error (annotated with the operation id)
    /// returns, so the caller can resume by passing the same journal.
    ///
    /// # Errors
    ///
    /// Returns the plan validation error, the handler error with the operation id in its
    /// context, or a store error from the handler unchanged.
    pub async fn execute(
        &self,
        context: &MigrationContext,
        plan: &MigrationPlan,
        journal: &mut MigrationJournal,
    ) -> Result<MigrationRun> {
        self.validate_plan(plan)?;
        let dry_run = plan.dry_run || context.dry_run;
        let mut reports = Vec::with_capacity(plan.steps.len());

        for step in &plan.steps {
            let operation_id = step.operation_id.as_str();
            let previous = journal.get(operation_id);
            let attempts = previous.map_or(0, |record| record.attempts);

            if previous.is_some_and(OperationRecord::succeeded) {
                reports.push(OperationReport::skipped(operation_id, "already completed"));
                continue;
            }

            if dry_run {
                journal.record(OperationRecord {
                    operation_id: operation_id.to_owned(),
                    attempts,
                    outcome: OperationOutcome::Skipped,
                    error: None,
                    affected: 0,
                    skipped: 0,
                    failed: 0,
                });
                reports.push(OperationReport::skipped(operation_id, "dry run"));
                continue;
            }

            let handler = self.handlers.get(operation_id).ok_or_else(|| {
                Error::validation(format!(
                    "memory migration plan names unknown operation {operation_id:?}"
                ))
            })?;

            match handler.execute(context, step).await {
                Ok(report) => {
                    let outcome = report.outcome;
                    let error = if outcome == OperationOutcome::Failed {
                        report.note.clone()
                    } else {
                        None
                    };
                    journal.record(OperationRecord {
                        operation_id: operation_id.to_owned(),
                        attempts: attempts.saturating_add(1),
                        outcome,
                        error,
                        affected: report.affected,
                        skipped: report.skipped,
                        failed: report.failed,
                    });
                    reports.push(report);
                }
                Err(error) => {
                    journal.record(OperationRecord {
                        operation_id: operation_id.to_owned(),
                        attempts: attempts.saturating_add(1),
                        outcome: OperationOutcome::Failed,
                        error: Some(error.to_string()),
                        affected: 0,
                        skipped: 0,
                        failed: 1,
                    });
                    return Err(error.with_context(format!(
                        "running memory migration operation `{operation_id}`"
                    )));
                }
            }
        }

        Ok(MigrationRun {
            plan: plan.clone(),
            journal: journal.clone(),
            reports,
        })
    }

    /// Rejects a plan with a repeated step id or an unregistered operation id.
    fn validate_plan(&self, plan: &MigrationPlan) -> Result<()> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for step in &plan.steps {
            if !seen.insert(step.operation_id.as_str()) {
                return Err(Error::validation(format!(
                    "memory migration plan repeats operation {:?}",
                    step.operation_id
                )));
            }
            if !self.contains(&step.operation_id) {
                return Err(Error::validation(format!(
                    "memory migration plan names unknown operation {:?}",
                    step.operation_id
                )));
            }
        }
        Ok(())
    }
}

/// Validates one operation id: `namespace.name`, no whitespace or NUL, at most 128 chars.
fn validate_operation_id(operation_id: &str) -> Result<()> {
    if operation_id.is_empty() {
        return Err(Error::validation(
            "memory migration operation id must not be empty",
        ));
    }
    if operation_id
        .chars()
        .any(|character| character.is_whitespace() || character == '\0')
    {
        return Err(Error::validation(format!(
            "memory migration operation id {operation_id:?} must not contain whitespace or NUL"
        )));
    }
    if !operation_id.contains('.') {
        return Err(Error::validation(format!(
            "memory migration operation id {operation_id:?} must contain a `.` namespace separator"
        )));
    }
    if operation_id.chars().count() > MAX_OPERATION_ID_LENGTH {
        return Err(Error::validation(format!(
            "memory migration operation id {operation_id:?} is longer than {MAX_OPERATION_ID_LENGTH} characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::store::MemoryStores;

    /// A handler with no behaviour beyond counting its calls.
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

        fn never(id: &'static str, calls: Arc<AtomicUsize>) -> Self {
            Self::new(id, calls, 0)
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

    fn context() -> MigrationContext {
        MigrationContext::new(MemoryStores::in_memory())
    }

    fn step(operation_id: &str) -> MigrationStep {
        MigrationStep::new(operation_id, "sql", 1, "test step")
    }

    fn registry_with(handlers: Vec<Arc<dyn MigrationOperationHandler>>) -> OperationRegistry {
        let mut registry = OperationRegistry::new();
        registry.register_all(handlers).expect("valid handlers");
        registry
    }

    fn journal_with(record: OperationRecord) -> MigrationJournal {
        let mut journal = MigrationJournal::new();
        journal.record(record);
        journal
    }

    fn succeeded_record(operation_id: &str, attempts: u32) -> OperationRecord {
        OperationRecord {
            operation_id: operation_id.to_owned(),
            attempts,
            outcome: OperationOutcome::Succeeded,
            error: None,
            affected: 1,
            skipped: 0,
            failed: 0,
        }
    }

    // --- journal -----------------------------------------------------------

    #[test]
    fn journal_tracks_completed_and_failed_records() {
        let mut journal = MigrationJournal::new();
        assert!(journal.is_empty());
        assert_eq!(journal.len(), 0);

        journal.record(succeeded_record("sql.two", 1));
        journal.record(OperationRecord {
            operation_id: "sql.one".to_owned(),
            attempts: 2,
            outcome: OperationOutcome::Failed,
            error: Some("boom".to_owned()),
            affected: 0,
            skipped: 0,
            failed: 1,
        });
        journal.record(OperationRecord {
            operation_id: "sql.three".to_owned(),
            attempts: 1,
            outcome: OperationOutcome::Skipped,
            error: None,
            affected: 0,
            skipped: 0,
            failed: 0,
        });

        assert_eq!(journal.len(), 3);
        assert_eq!(journal.completed(), vec!["sql.two".to_owned()]);
        assert!(journal.succeeded("sql.two"));
        assert!(!journal.succeeded("sql.one"));
        assert!(!journal.succeeded("absent"));
        let failed = journal.failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].operation_id, "sql.one");
        assert!(failed[0].retryable());
        assert!(journal.get("sql.three").expect("record").retryable());
        assert!(journal.get("sql.two").expect("record").succeeded());
        assert!(!journal.get("sql.two").expect("record").retryable());

        let records: Vec<&str> = journal
            .records()
            .map(|record| record.operation_id.as_str())
            .collect();
        assert_eq!(records, vec!["sql.one", "sql.three", "sql.two"]);

        // Recording the same id again replaces the record.
        journal.record(succeeded_record("sql.one", 3));
        assert_eq!(journal.len(), 3);
        assert!(journal.succeeded("sql.one"));
        assert_eq!(journal.failed().len(), 0);
        assert_eq!(journal.completed(), vec!["sql.one", "sql.two"]);

        journal.clear();
        assert!(journal.is_empty());
    }

    #[test]
    fn journal_json_round_trip_is_tolerant() {
        let mut journal = MigrationJournal::new();
        journal.record(succeeded_record("sql.one", 2));
        let text = journal.to_json().expect("serializable");
        assert_eq!(
            MigrationJournal::from_json(&text).expect("decodable"),
            journal
        );

        let minimal = MigrationJournal::from_json(
            r#"{"entries":{"sql.one":{"operation_id":"sql.one","outcome":"failed"}}}"#,
        )
        .expect("tolerant decode");
        let record = minimal.get("sql.one").expect("record");
        assert_eq!(record.attempts, 0);
        assert_eq!(record.error, None);
        assert_eq!(record.affected, 0);

        assert!(MigrationJournal::from_json("{}")
            .expect("empty journal")
            .is_empty());
        let error = MigrationJournal::from_json("not a journal").expect_err("malformed document");
        assert!(matches!(error, Error::Validation(_)), "{error}");
    }

    #[test]
    fn run_counts_and_success_derive_from_the_journal() {
        let plan = MigrationPlan::new(vec![step("sql.one"), step("sql.two")]);
        let mut journal = MigrationJournal::new();
        journal.record(OperationRecord {
            operation_id: "sql.one".to_owned(),
            attempts: 1,
            outcome: OperationOutcome::Succeeded,
            error: None,
            affected: 3,
            skipped: 1,
            failed: 0,
        });
        journal.record(OperationRecord {
            operation_id: "sql.two".to_owned(),
            attempts: 1,
            outcome: OperationOutcome::Skipped,
            error: None,
            affected: 0,
            skipped: 0,
            failed: 0,
        });
        let run = MigrationRun {
            plan,
            journal,
            reports: vec![OperationReport::succeeded("sql.one", 3, 1, 0)],
        };
        assert!(run.is_success());
        assert_eq!(run.executed(), 1);
        assert_eq!(run.skipped(), 1);
        let counts = run.counts();
        assert_eq!(counts.scanned, 4);
        assert_eq!(counts.affected, 3);
        assert_eq!(counts.skipped, 1);
        assert_eq!(counts.failed, 0);

        let text = run.to_json().expect("serializable");
        assert_eq!(MigrationRun::from_json(&text).expect("decodable"), run);

        let failing = OperationRecord {
            operation_id: "sql.three".to_owned(),
            attempts: 1,
            outcome: OperationOutcome::Failed,
            error: Some("boom".to_owned()),
            affected: 0,
            skipped: 0,
            failed: 2,
        };
        let run = MigrationRun {
            plan: MigrationPlan::new(vec![step("sql.three")]),
            journal: journal_with(failing),
            reports: Vec::new(),
        };
        assert!(!run.is_success());
        assert_eq!(run.executed(), 0);
        assert_eq!(run.skipped(), 0);
        assert_eq!(run.counts().scanned, 2);
        assert_eq!(run.counts().failed, 2);
    }

    // --- registration ------------------------------------------------------

    #[test]
    fn registration_validates_ids() {
        let calls = Arc::new(AtomicUsize::new(0));
        let long: &'static str = Box::leak(format!("sql.{}", "a".repeat(125)).into_boxed_str());
        assert_eq!(long.chars().count(), 129);

        for id in ["", "sqla", "sql. two", "sql.\0two", long] {
            let mut registry = OperationRegistry::new();
            let error = registry
                .register(Arc::new(CountingHandler::never(id, calls.clone())))
                .expect_err("invalid id");
            assert!(matches!(error, Error::Validation(_)), "{id:?}: {error}");
            assert!(registry.is_empty(), "{id:?} left the registry untouched");
        }

        let mut registry = OperationRegistry::new();
        registry
            .register(Arc::new(CountingHandler::never("sql.one", calls.clone())))
            .expect("valid id");
        let error = registry
            .register(Arc::new(CountingHandler::never("sql.one", calls)))
            .expect_err("duplicate id");
        assert!(error.to_string().contains("sql.one"), "{error}");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn registration_is_sorted_and_register_all_is_atomic() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = registry_with(vec![
            Arc::new(CountingHandler::never("sql.b", calls.clone())),
            Arc::new(CountingHandler::never("message.a", calls.clone())),
        ]);
        assert_eq!(registry.operation_ids(), vec!["message.a", "sql.b"]);
        assert!(registry.contains("sql.b"));
        assert!(!registry.contains("sql.c"));
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());

        let error = registry
            .register_all(vec![
                Arc::new(CountingHandler::never("vector.a", calls.clone())),
                Arc::new(CountingHandler::never("sql.b", calls.clone())),
            ])
            .expect_err("duplicate inside the batch");
        assert!(error.to_string().contains("sql.b"), "{error}");
        assert!(
            !registry.contains("vector.a"),
            "a rejected batch registers nothing"
        );
        assert_eq!(registry.len(), 2);

        registry
            .register_all(vec![Arc::new(CountingHandler::never("vector.a", calls))])
            .expect("distinct batch");
        assert_eq!(
            registry.operation_ids(),
            vec!["message.a", "sql.b", "vector.a"]
        );

        let built = OperationRegistry::new()
            .with_handler(Arc::new(CountingHandler::never(
                "sql.one",
                Arc::new(AtomicUsize::new(0)),
            )))
            .expect("builder");
        assert_eq!(built.operation_ids(), vec!["sql.one"]);
    }

    // --- execution ---------------------------------------------------------

    #[tokio::test]
    async fn execute_records_the_failure_and_resumes_from_the_journal() {
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let third = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![
            Arc::new(CountingHandler::never("sql.one", first.clone())),
            Arc::new(CountingHandler::new("sql.two", second.clone(), 1)),
            Arc::new(CountingHandler::never("sql.three", third.clone())),
        ]);
        let plan = MigrationPlan::new(vec![step("sql.one"), step("sql.two"), step("sql.three")]);
        let mut journal = MigrationJournal::new();

        let error = registry
            .execute(&context(), &plan, &mut journal)
            .await
            .expect_err("second step fails");
        assert!(error.to_string().contains("sql.two"), "{error}");
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(second.load(Ordering::SeqCst), 1);
        assert_eq!(
            third.load(Ordering::SeqCst),
            0,
            "the run stops at the failure"
        );

        let one = journal.get("sql.one").expect("first recorded");
        assert_eq!(one.outcome, OperationOutcome::Succeeded);
        assert_eq!(one.attempts, 1);
        let two = journal.get("sql.two").expect("second recorded");
        assert_eq!(two.outcome, OperationOutcome::Failed);
        assert_eq!(two.attempts, 1);
        assert_eq!(two.failed, 1);
        assert!(two
            .error
            .as_deref()
            .is_some_and(|text| text.contains("boom")));
        assert!(journal.get("sql.three").is_none());

        let run = registry
            .execute(&context(), &plan, &mut journal)
            .await
            .expect("resumed run");
        assert!(run.is_success());
        assert_eq!(
            first.load(Ordering::SeqCst),
            1,
            "a completed step is skipped"
        );
        assert_eq!(journal.get("sql.one").expect("record").attempts, 1);
        assert_eq!(second.load(Ordering::SeqCst), 2, "a failed step is retried");
        assert_eq!(journal.get("sql.two").expect("record").attempts, 2);
        assert_eq!(third.load(Ordering::SeqCst), 1);
        assert_eq!(journal.get("sql.three").expect("record").attempts, 1);
        assert_eq!(journal.completed().len(), 3);
        assert_eq!(run.reports.len(), 3);
        assert_eq!(run.reports[0].outcome, OperationOutcome::Skipped);
        assert_eq!(run.reports[0].note.as_deref(), Some("already completed"));
        assert_eq!(run.reports[1].outcome, OperationOutcome::Succeeded);
    }

    #[tokio::test]
    async fn validation_failures_run_nothing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![Arc::new(CountingHandler::never(
            "sql.one",
            calls.clone(),
        ))]);

        let unknown = MigrationPlan::new(vec![step("sql.one"), step("sql.unknown")]);
        let mut journal = MigrationJournal::new();
        let error = registry
            .execute(&context(), &unknown, &mut journal)
            .await
            .expect_err("unknown operation");
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("sql.unknown"), "{error}");

        let duplicated = MigrationPlan::new(vec![step("sql.one"), step("sql.one")]);
        let error = registry
            .execute(&context(), &duplicated, &mut journal)
            .await
            .expect_err("duplicate step");
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("sql.one"), "{error}");

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(journal.is_empty());
    }

    #[tokio::test]
    async fn dry_runs_invoke_no_handler() {
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![Arc::new(CountingHandler::never(
            "sql.one",
            calls.clone(),
        ))]);
        let plan = MigrationPlan::new(vec![step("sql.one")]);

        let mut context_journal = MigrationJournal::new();
        let run = registry
            .execute(&context().with_dry_run(true), &plan, &mut context_journal)
            .await
            .expect("dry run through the context");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(run.is_success());
        assert_eq!(run.skipped(), 1);
        let record = context_journal.get("sql.one").expect("recorded");
        assert_eq!(record.outcome, OperationOutcome::Skipped);
        assert_eq!(record.attempts, 0);
        assert_eq!(run.reports[0].note.as_deref(), Some("dry run"));

        let mut plan_journal = MigrationJournal::new();
        let run = registry
            .execute(
                &context(),
                &plan.clone().with_dry_run(true),
                &mut plan_journal,
            )
            .await
            .expect("dry run through the plan");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(run.skipped(), 1);
        assert!(!plan_journal.succeeded("sql.one"));
        assert_eq!(
            plan_journal.get("sql.one").expect("recorded").outcome,
            OperationOutcome::Skipped
        );
    }

    #[tokio::test]
    async fn a_completed_step_is_skipped_without_touching_its_record() {
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![Arc::new(CountingHandler::never(
            "sql.one",
            calls.clone(),
        ))]);
        let plan = MigrationPlan::new(vec![step("sql.one")]);
        let mut journal = journal_with(succeeded_record("sql.one", 4));

        let run = registry
            .execute(&context(), &plan, &mut journal)
            .await
            .expect("nothing to do");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(run.is_success());
        assert_eq!(run.executed(), 1);
        assert_eq!(run.skipped(), 0);
        assert_eq!(run.reports.len(), 1);
        assert_eq!(run.reports[0].note.as_deref(), Some("already completed"));
        let record = journal.get("sql.one").expect("record kept");
        assert_eq!(record.outcome, OperationOutcome::Succeeded);
        assert_eq!(record.attempts, 4, "the completed record is untouched");
    }

    #[tokio::test]
    async fn skipped_and_failed_records_are_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![Arc::new(CountingHandler::never(
            "sql.one",
            calls.clone(),
        ))]);
        let plan = MigrationPlan::new(vec![step("sql.one")]);
        let mut journal = journal_with(OperationRecord {
            operation_id: "sql.one".to_owned(),
            attempts: 3,
            outcome: OperationOutcome::Skipped,
            error: None,
            affected: 0,
            skipped: 0,
            failed: 0,
        });

        let run = registry
            .execute(&context(), &plan, &mut journal)
            .await
            .expect("retried");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(run.is_success());
        let record = journal.get("sql.one").expect("record");
        assert_eq!(record.outcome, OperationOutcome::Succeeded);
        assert_eq!(record.attempts, 4);
        assert_eq!(record.affected, 1);
    }

    #[tokio::test]
    async fn a_reported_failure_is_journalled_and_the_run_continues() {
        struct FailingReport;

        #[async_trait]
        impl MigrationOperationHandler for FailingReport {
            fn operation_id(&self) -> &'static str {
                "sql.fail"
            }

            async fn execute(
                &self,
                _context: &MigrationContext,
                _step: &MigrationStep,
            ) -> Result<OperationReport> {
                Ok(OperationReport::failed("sql.fail", "nothing convertible"))
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![
            Arc::new(FailingReport),
            Arc::new(CountingHandler::never("sql.after", calls.clone())),
        ]);
        let plan = MigrationPlan::new(vec![step("sql.fail"), step("sql.after")]);
        let mut journal = MigrationJournal::new();

        let run = registry
            .execute(&context(), &plan, &mut journal)
            .await
            .expect("a reported failure is not an error");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!run.is_success());
        assert_eq!(run.executed(), 1);
        let record = journal.get("sql.fail").expect("record");
        assert_eq!(record.outcome, OperationOutcome::Failed);
        assert_eq!(record.failed, 1);
        assert_eq!(record.error.as_deref(), Some("nothing convertible"));
        assert!(record.retryable());
        assert_eq!(run.reports.len(), 2);
        assert_eq!(run.reports[0].outcome, OperationOutcome::Failed);
    }

    #[tokio::test]
    async fn a_plan_narrowed_to_another_migrator_executes_nothing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(vec![Arc::new(CountingHandler::never(
            "sql.one",
            calls.clone(),
        ))]);
        let plan = MigrationPlan::new(vec![step("sql.one")]).steps_for("vector");
        let mut journal = MigrationJournal::new();
        let run = registry
            .execute(&context(), &plan, &mut journal)
            .await
            .expect("empty plan");
        assert_eq!(run.reports.len(), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(journal.is_empty());
    }
}
