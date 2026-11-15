//! Discovery of the sessions a user can list, show or resume.
//!
//! Three independent stores hold state a conversation can be continued from,
//! and this module presents them as one catalogue:
//!
//! 1. **cli** — the CLI session records under `<home>/sessions/<id>.json`
//!    ([`SessionStore`]). These are what `run`/`chat` write, and they hold the
//!    plain transcript.
//! 2. **execution** — the persistent-agent execution journals under the
//!    workspace's `persistent_agent/logs` ([`ExecutionCatalog`]). One journal is
//!    one execution; its entries reassemble into a model window.
//! 3. **state** — the persistent-agent session records under
//!    `persistent_agent/sessions` ([`FileSessionStore`]). A record carries the
//!    model window, an optional `deepagent` state payload and the cursor naming
//!    the execution it can be resumed from.
//!
//! [`Catalog::list`] merges the three into [`Row`]s, newest first.
//! [`Catalog::resolve`] turns an id (or "the most recent one") into a
//! [`Resolved`] turn: the session id to continue, the history to seed, the
//! optional `deepagent` state, and the [`ResumePlan`] when a journal was traced.
//! [`Catalog::prune`] deletes rows, optionally as a dry run.
//!
//! Everything here is read-only apart from [`Catalog::prune`], so listing works
//! on a home and a workspace that were never materialised.
use std::cmp::Ordering;
use std::path::Path;

use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::prompt::BaseMessage;
use openjiuwen_harness::cli::{
    session_id, CliOptions, MessageRecord, SessionRecord, SessionStore, SessionSummary,
};
use openjiuwen_harness::{
    resume, Capacity, Continuation, ExecutionCatalog, ExecutionSummary, FileLinkStore,
    FileSessionStore, LogCapacity, PersistentAgentLayout, ResumePlan, RootLogSource,
    SessionStateStore, SessionSummary as StateSummary, WindowAssembler, Workspace,
};

/// Kind tag of a CLI session record.
pub(crate) const KIND_CLI: &str = "cli";
/// Kind tag of a persistent-agent execution journal.
pub(crate) const KIND_EXECUTION: &str = "execution";
/// Kind tag of a persistent-agent session record.
pub(crate) const KIND_STATE: &str = "state";
/// How many executions a resume may trace when it rebuilds the window.
const CHAIN_LIMIT: usize = 4;

/// The store entry a [`Row`] was built from, kept for the prune path.
///
/// It is deliberately not serialized: the row already carries the identifying
/// fields, and the entry holds paths and internal flags a JSON consumer has no
/// use for.
#[derive(Debug, Clone)]
pub(crate) enum Detail {
    /// A CLI session record.
    Cli(SessionSummary),
    /// A persistent-agent execution journal.
    Execution(ExecutionSummary),
    /// A persistent-agent session record.
    State(StateSummary),
}

/// One persisted candidate as the catalogue reports it.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Row {
    /// Session (or execution) identifier.
    pub(crate) id: String,
    /// Which store the row came from: `cli`, `execution` or `state`.
    pub(crate) kind: &'static str,
    /// RFC 3339 creation timestamp, when the store records one.
    pub(crate) created_at: Option<String>,
    /// One-line description of what the row holds.
    pub(crate) detail: String,
    /// The store entry behind the row.
    #[serde(skip)]
    pub(crate) data: Detail,
}

/// What the user asked to resume.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ResumeRequest<'a> {
    /// Resume one identified session or execution.
    Id(&'a str),
    /// Resume the most recent resumable candidate.
    Continue,
}

/// A resolved resume: everything a resumed turn needs, plus how it was found.
#[derive(Debug, Clone)]
pub(crate) struct Resolved {
    /// Session id the new turn continues.
    pub(crate) id: String,
    /// Which store the resume came from (`cli`, `execution` or `state`).
    pub(crate) kind: &'static str,
    /// One-line note for stderr: where the history came from.
    pub(crate) summary: String,
    /// Prior message window, oldest first.
    pub(crate) history: Vec<BaseMessage>,
    /// Persisted `deepagent` state payload, when the record carried an object.
    pub(crate) state: Option<serde_json::Value>,
    /// The traced resume plan, when a journal could be reopened.
    pub(crate) plan: Option<ResumePlan>,
    /// The CLI session record behind the resume, when it came from one.
    pub(crate) record: Option<SessionRecord>,
}

/// The persisted session state of one home and one workspace.
#[derive(Debug, Clone)]
pub(crate) struct Catalog {
    home: OpenJiuwenHome,
    layout: PersistentAgentLayout,
}

impl Catalog {
    /// Builds the catalogue `options` and an optional `--data-dir` describe.
    ///
    /// Without a data directory the persistent-agent layout is the workspace's
    /// own (`<workspace>/persistent_agent`), the one the harness writes.
    #[must_use]
    pub(crate) fn for_options(options: &CliOptions, data_dir: Option<&Path>) -> Self {
        let layout = match data_dir {
            Some(dir) => PersistentAgentLayout::new(dir),
            None => {
                PersistentAgentLayout::for_workspace(&Workspace::new(options.workspace.clone()))
            }
        };
        Self {
            home: options.home.clone(),
            layout,
        }
    }

    /// Returns the CLI session records of the current home.
    fn cli_store(&self) -> SessionStore {
        SessionStore::new(&self.home)
    }

    /// Returns the persistent-agent session records of the layout.
    fn state_store(&self) -> FileSessionStore {
        FileSessionStore::new(self.layout.sessions_root())
    }

    /// Returns the persistent-agent execution catalogue of the layout.
    fn execution_catalog(&self) -> ExecutionCatalog {
        ExecutionCatalog::new(self.layout.logs_root(), LogCapacity::default())
    }

    /// Lists every persisted candidate, newest first.
    ///
    /// # Errors
    ///
    /// Returns whatever the three stores report when a root exists but cannot be
    /// read at all; a single damaged record is skipped by the store itself.
    pub(crate) async fn list(&self) -> Result<Vec<Row>> {
        let mut rows = Vec::new();
        for summary in self.cli_store().list()? {
            rows.push(Row {
                id: summary.session_id.clone(),
                kind: KIND_CLI,
                created_at: non_empty(&summary.created_at),
                detail: format!("{} messages, model {}", summary.messages, summary.model),
                data: Detail::Cli(summary),
            });
        }
        for summary in self.execution_catalog().list().await? {
            rows.push(Row {
                id: summary.execution_id.clone(),
                kind: KIND_EXECUTION,
                created_at: non_empty(&summary.created_at),
                detail: format!(
                    "{} entries, session {}, usable {}",
                    summary.entries, summary.session_id, summary.usable
                ),
                data: Detail::Execution(summary),
            });
        }
        for summary in self.state_store().list().await? {
            rows.push(Row {
                id: summary.binding.session_id.clone(),
                kind: KIND_STATE,
                created_at: state_timestamp(&summary).await.map(format_timestamp),
                detail: format!(
                    "agent {}, waiting {}, ending {}",
                    summary.binding.agent_id, summary.waiting, summary.has_ending
                ),
                data: Detail::State(summary),
            });
        }
        sort_rows(&mut rows);
        Ok(rows)
    }

    /// Resolves `request` into the turn a resumed run continues.
    ///
    /// [`ResumeRequest::Id`] looks for a persistent session record, then a CLI
    /// session record, then an execution journal, all by their own identifier.
    /// [`ResumeRequest::Continue`] picks the newest usable execution, else the
    /// newest persistent session record, else the newest CLI session record, and
    /// then resolves it the same way.
    ///
    /// # Errors
    ///
    /// `Error::not_found` when nothing matches, and whatever the stores, the
    /// journal reader or [`resume`] report.
    pub(crate) async fn resolve(&self, request: ResumeRequest<'_>) -> Result<Resolved> {
        match request {
            ResumeRequest::Id(id) => self.resolve_id(id).await,
            ResumeRequest::Continue => self.resolve_most_recent().await,
        }
    }

    /// Resolves one id through the documented precedence.
    async fn resolve_id(&self, id: &str) -> Result<Resolved> {
        let state = self
            .state_store()
            .list()
            .await?
            .into_iter()
            .find(|summary| summary.binding.session_id == id);
        if let Some(summary) = state {
            if let Some(resolved) = self.resolve_state(&summary, false).await? {
                return Ok(resolved);
            }
        }
        let cli = self.cli_store();
        let cli_summary = cli
            .list()?
            .into_iter()
            .find(|summary| summary.session_id == id);
        if let Some(summary) = cli_summary {
            if let Some(resolved) = Self::resolve_cli(&cli, &summary, false)? {
                return Ok(resolved);
            }
        }
        let catalog = self.execution_catalog();
        let execution = catalog
            .list()
            .await?
            .into_iter()
            .find(|summary| summary.execution_id == id);
        if let Some(summary) = execution {
            return self.resolve_execution(&catalog, &summary, false).await;
        }
        Err(Error::not_found(format!("unknown session: {id}")))
    }

    /// Resolves the newest resumable candidate.
    async fn resolve_most_recent(&self) -> Result<Resolved> {
        let catalog = self.execution_catalog();
        if let Some(summary) = newest_execution(catalog.list().await?) {
            return self.resolve_execution(&catalog, &summary, true).await;
        }
        if let Some(summary) = self.newest_state().await? {
            if let Some(resolved) = self.resolve_state(&summary, true).await? {
                return Ok(resolved);
            }
        }
        let cli = self.cli_store();
        if let Some(summary) = newest_cli(cli.list()?) {
            if let Some(resolved) = Self::resolve_cli(&cli, &summary, true)? {
                return Ok(resolved);
            }
        }
        Err(Error::not_found("no resumable session found"))
    }

    /// Returns the persistent session record with the newest `state.json`.
    async fn newest_state(&self) -> Result<Option<StateSummary>> {
        let mut newest: Option<(StateSummary, DateTime<Utc>)> = None;
        for summary in self.state_store().list().await? {
            let Some(at) = state_timestamp(&summary).await else {
                continue;
            };
            let better = match &newest {
                None => true,
                Some((current, current_at)) => {
                    at > *current_at
                        || (at == *current_at
                            && summary.binding.session_id < current.binding.session_id)
                }
            };
            if better {
                newest = Some((summary, at));
            }
        }
        Ok(newest.map(|(summary, _)| summary))
    }

    /// Loads one persistent session record and rebuilds its window.
    ///
    /// `Ok(None)` means the record vanished between the listing and the load, so
    /// the caller can fall through to the next store.
    async fn resolve_state(
        &self,
        summary: &StateSummary,
        most_recent: bool,
    ) -> Result<Option<Resolved>> {
        let binding = summary.binding.clone();
        let store = self.state_store();
        let Some(record) = store.load(&binding).await? else {
            return Ok(None);
        };
        let prefix = history_prefix(&binding.session_id, most_recent);
        let recovery = record
            .ending
            .as_ref()
            .map(|ending| ending.execution_id.clone())
            .filter(|id| !id.is_empty())
            .or_else(|| {
                record
                    .checkpoint
                    .as_ref()
                    .map(|checkpoint| checkpoint.execution_id.clone())
                    .filter(|id| !id.is_empty())
            })
            .or_else(|| {
                record
                    .continuation
                    .as_ref()
                    .and_then(|cursor| cursor.recovery_ref.clone())
                    .filter(|id| !id.is_empty())
            });
        if let Some(execution_id) = recovery {
            let continuation = Continuation::new(execution_id.clone(), None, CHAIN_LIMIT)?;
            let plan = resume(
                &store,
                &RootLogSource::new(self.layout.logs_root(), LogCapacity::default()),
                &FileLinkStore::new(self.layout.links_root()),
                &binding,
                &session_id(),
                &continuation,
                &WindowAssembler::new(Capacity::default()),
            )
            .await?;
            let history = plan.seed_history();
            let state = plan
                .record
                .deepagent_state
                .clone()
                .filter(serde_json::Value::is_object);
            let note = format!(
                "{prefix} from the persistent session record via execution {execution_id} \
                 ({} messages restored)",
                history.len()
            );
            return Ok(Some(Resolved {
                id: binding.session_id,
                kind: KIND_STATE,
                summary: note,
                history,
                state,
                plan: Some(plan),
                record: None,
            }));
        }
        let history = record.messages.clone();
        let note = format!(
            "{prefix} from the persistent session record ({} messages restored)",
            history.len()
        );
        Ok(Some(Resolved {
            id: binding.session_id,
            kind: KIND_STATE,
            summary: note,
            history,
            state: record.deepagent_state.filter(serde_json::Value::is_object),
            plan: None,
            record: None,
        }))
    }

    /// Loads one CLI session record and maps its transcript.
    ///
    /// `Ok(None)` means the record vanished between the listing and the load.
    fn resolve_cli(
        store: &SessionStore,
        summary: &SessionSummary,
        most_recent: bool,
    ) -> Result<Option<Resolved>> {
        let Some(record) = store.load(&summary.session_id)? else {
            return Ok(None);
        };
        // The record's own id wins; an empty one continues under the id the file
        // was found with, so a hand-written record still resumes under that name.
        let id = if record.session_id.is_empty() {
            summary.session_id.clone()
        } else {
            record.session_id.clone()
        };
        let history: Vec<BaseMessage> = record
            .messages
            .iter()
            .filter_map(message_from_record)
            .collect();
        let note = format!(
            "{} from the CLI session record ({} messages)",
            history_prefix(&id, most_recent),
            history.len()
        );
        Ok(Some(Resolved {
            id,
            kind: KIND_CLI,
            summary: note,
            history,
            state: None,
            plan: None,
            record: Some(record),
        }))
    }

    /// Reopens one execution journal and reassembles its window.
    async fn resolve_execution(
        &self,
        catalog: &ExecutionCatalog,
        summary: &ExecutionSummary,
        most_recent: bool,
    ) -> Result<Resolved> {
        let log = catalog.open(&summary.execution_id).await?;
        let snapshot = log.read().await?;
        let history = WindowAssembler::new(Capacity::default())
            .assemble(&snapshot.entries)
            .messages;
        let id = if summary.session_id.is_empty() {
            summary.execution_id.clone()
        } else {
            summary.session_id.clone()
        };
        let summary_text = format!(
            "{} from execution {} ({} messages restored)",
            history_prefix(&id, most_recent),
            summary.execution_id,
            history.len()
        );
        Ok(Resolved {
            id,
            kind: KIND_EXECUTION,
            summary: summary_text,
            history,
            state: None,
            plan: None,
            record: None,
        })
    }

    /// Returns the row and the resolution behind one id.
    ///
    /// # Errors
    ///
    /// Everything [`Catalog::resolve`] reports.
    pub(crate) async fn show(&self, id: &str) -> Result<(Row, Resolved)> {
        let resolved = self.resolve(ResumeRequest::Id(id)).await?;
        let rows = self.list().await?;
        let row = rows
            .into_iter()
            .find(|row| row.id == resolved.id && row.kind == resolved.kind)
            // A record that disappeared between the two scans has no row left;
            // the resolution itself already carries everything `show` prints.
            .unwrap_or_else(|| Row {
                id: resolved.id.clone(),
                kind: resolved.kind,
                created_at: None,
                detail: resolved.summary.clone(),
                data: Detail::Cli(SessionSummary::default()),
            });
        Ok((row, resolved))
    }

    /// Selects (and, unless `dry_run`, deletes) persisted candidates.
    ///
    /// `all` selects every row; otherwise `older_than_days` selects the rows
    /// whose creation timestamp is older than that many days. Rows without a
    /// timestamp are only selected by `all`, never by an age filter.
    ///
    /// # Errors
    ///
    /// `Error::validation` when neither selector is given, plus the listing and
    /// deletion errors of the stores.
    pub(crate) async fn prune(
        &self,
        older_than_days: Option<f64>,
        all: bool,
        dry_run: bool,
    ) -> Result<Vec<Row>> {
        if !all && older_than_days.is_none() {
            return Err(Error::validation(
                "prune needs --older-than <DAYS> or --all",
            ));
        }
        let cutoff = older_than_days.map(cutoff);
        let selected: Vec<Row> = self
            .list()
            .await?
            .into_iter()
            .filter(|row| match cutoff {
                None => all,
                Some(cutoff) => {
                    all || matches!(parse_timestamp(row.created_at.as_deref()), Some(at) if at < cutoff)
                }
            })
            .collect();
        if dry_run {
            return Ok(selected);
        }
        for row in &selected {
            self.remove(row).await?;
        }
        Ok(selected)
    }

    /// Deletes the store entry behind one row.
    async fn remove(&self, row: &Row) -> Result<()> {
        match &row.data {
            Detail::Cli(summary) => {
                remove_file(&self.cli_store().path_for(&summary.session_id)).await
            }
            Detail::Execution(summary) => {
                remove_dir(&self.execution_catalog().path_for(&summary.execution_id)).await
            }
            Detail::State(summary) => match summary.state_path.parent() {
                Some(dir) => remove_dir(dir).await,
                None => Ok(()),
            },
        }
    }
}

/// Returns `value` as an owned timestamp, or `None` when it is blank.
fn non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_owned())
}

/// Formats `at` as an RFC 3339 UTC timestamp with second precision.
fn format_timestamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// Parses an RFC 3339 timestamp, tolerating a missing one.
fn parse_timestamp(value: Option<&str>) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value?)
        .ok()
        .map(|at| at.with_timezone(&Utc))
}

/// Returns the modification time of a session record's `state.json`.
async fn state_timestamp(summary: &StateSummary) -> Option<DateTime<Utc>> {
    let modified = tokio::fs::metadata(&summary.state_path)
        .await
        .ok()?
        .modified()
        .ok()?;
    Some(DateTime::<Utc>::from(modified))
}

/// Returns `now - days`, the boundary an age filter compares against.
///
/// Fractional days are truncated to whole milliseconds; `--older-than` is a
/// convenience threshold, not a precise clock.
#[allow(clippy::cast_possible_truncation)] // whole milliseconds of a finite day count
fn cutoff(days: f64) -> DateTime<Utc> {
    let millis = (days * 86_400_000.0) as i64;
    Utc::now() - chrono::Duration::milliseconds(millis)
}

/// Orders rows newest first, timestamp-less rows last, ties by id ascending.
fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|left, right| {
        match (
            parse_timestamp(left.created_at.as_deref()),
            parse_timestamp(right.created_at.as_deref()),
        ) {
            (Some(left_at), Some(right_at)) => {
                right_at.cmp(&left_at).then_with(|| left.id.cmp(&right.id))
            }
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left.id.cmp(&right.id),
        }
    });
}

/// Returns the newest usable execution, breaking ties by id.
fn newest_execution(executions: Vec<ExecutionSummary>) -> Option<ExecutionSummary> {
    executions
        .into_iter()
        .filter(|summary| summary.usable)
        .max_by(|left, right| {
            parse_timestamp(Some(&left.created_at))
                .cmp(&parse_timestamp(Some(&right.created_at)))
                .then_with(|| right.execution_id.cmp(&left.execution_id))
        })
}

/// Returns the newest CLI session record, breaking ties by id.
fn newest_cli(records: Vec<SessionSummary>) -> Option<SessionSummary> {
    records.into_iter().max_by(|left, right| {
        parse_timestamp(Some(&left.created_at))
            .cmp(&parse_timestamp(Some(&right.created_at)))
            .then_with(|| right.session_id.cmp(&left.session_id))
    })
}

/// Maps one persisted CLI message onto the kernel message type.
///
/// Roles the kernel has no constructor for (a hand-edited record, a role this
/// version does not know) are skipped: a resumed transcript only carries what
/// the model can consume.
fn message_from_record(record: &MessageRecord) -> Option<BaseMessage> {
    match record.role.as_str() {
        "user" => Some(BaseMessage::user(record.content.clone())),
        "assistant" => Some(BaseMessage::assistant(record.content.clone())),
        "system" => Some(BaseMessage::system(record.content.clone())),
        _ => None,
    }
}

/// Returns the `resuming …` prefix every resume note starts with.
fn history_prefix(id: &str, most_recent: bool) -> String {
    if most_recent {
        format!("resuming the most recent session {id}")
    } else {
        format!("resuming session {id}")
    }
}

/// Removes one file, treating an already-missing file as success.
async fn remove_file(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Removes one directory tree, treating an already-missing one as success.
async fn remove_dir(path: &Path) -> Result<()> {
    match tokio::fs::remove_dir_all(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_harness::cli::MapEnv;
    use openjiuwen_harness::{PersistentSessionRecord, RecordEnding, SessionBinding};
    use openjiuwen_test_support::TempWorkspace;

    use super::*;
    use clap::Parser;

    /// Resolves the harness options of an isolated temp home and workspace.
    fn options(fixtures: &TempWorkspace) -> CliOptions {
        let env = MapEnv::from_pairs([("OPENJIUWEN_HOME", fixtures.home().display().to_string())]);
        let cli = openjiuwen_harness::cli::Cli::parse_from(["openjiuwen"]);
        CliOptions::resolve_with(&cli, &env, None, &fixtures.workspace()).expect("test options")
    }

    fn row(id: &str, created_at: Option<&str>) -> Row {
        Row {
            id: id.to_owned(),
            kind: KIND_CLI,
            created_at: created_at.map(str::to_owned),
            detail: String::new(),
            data: Detail::Cli(SessionSummary::default()),
        }
    }

    #[test]
    fn rows_sort_newest_first_with_timestampless_rows_last() {
        let mut rows = vec![
            row("b", None),
            row("a", Some("2024-01-01T00:00:00+00:00")),
            row("d", Some("2024-03-01T00:00:00Z")),
            row("c", Some("2024-01-01T00:00:00Z")),
            row("e", None),
        ];
        sort_rows(&mut rows);
        let ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(ids, vec!["d", "a", "c", "b", "e"]);
    }

    #[test]
    fn an_unparsable_timestamp_sorts_without_one() {
        let mut rows = vec![
            row("a", Some("not a timestamp")),
            row("b", Some("2024-01-01T00:00:00Z")),
        ];
        sort_rows(&mut rows);
        assert_eq!(rows[0].id, "b");
        assert_eq!(rows[1].id, "a");
    }

    #[test]
    fn cli_messages_map_onto_kernel_roles() {
        let message = |role: &str| MessageRecord {
            role: role.to_owned(),
            content: "hi".to_owned(),
            timestamp: String::new(),
        };
        assert_eq!(
            message_from_record(&message("user")).map(|message| message.role.as_str().to_owned()),
            Some("user".to_owned())
        );
        assert_eq!(
            message_from_record(&message("assistant"))
                .map(|message| message.role.as_str().to_owned()),
            Some("assistant".to_owned())
        );
        assert_eq!(
            message_from_record(&message("system")).map(|message| message.role.as_str().to_owned()),
            Some("system".to_owned())
        );
        let tool = MessageRecord {
            role: "tool".to_owned(),
            content: "result".to_owned(),
            timestamp: String::new(),
        };
        assert!(message_from_record(&tool).is_none());
    }

    #[test]
    fn resume_notes_use_the_documented_prefix() {
        assert_eq!(history_prefix("cli-1", false), "resuming session cli-1");
        assert_eq!(
            history_prefix("exec-1", true),
            "resuming the most recent session exec-1"
        );
    }

    #[test]
    fn blank_timestamps_are_reported_as_missing() {
        assert_eq!(non_empty("  "), None);
        assert_eq!(
            non_empty("2024-01-01T00:00:00Z").as_deref(),
            Some("2024-01-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn a_persistent_state_record_lists_resolves_and_prunes() {
        let fixtures = TempWorkspace::new().expect("temp workspace");
        let options = options(&fixtures);
        let layout =
            PersistentAgentLayout::for_workspace(&Workspace::new(options.workspace.clone()));
        let store = FileSessionStore::new(layout.sessions_root());
        let binding = SessionBinding {
            agent_id: "agent-one".to_owned(),
            session_id: "sess-1".to_owned(),
        };
        let mut record = PersistentSessionRecord::new(&binding).expect("draft record");
        record.messages = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
        let record = record
            .with_ending(RecordEnding {
                outcome: "completed".to_owned(),
                request_id: "req-1".to_owned(),
                ..RecordEnding::default()
            })
            .expect("ended record");
        store.save(&record).await.expect("save record");

        let catalog = Catalog::for_options(&options, None);
        let rows = catalog.list().await.expect("list");
        let row = rows
            .iter()
            .find(|row| row.id == "sess-1")
            .expect("state row");
        assert_eq!(row.kind, KIND_STATE);
        assert!(row.created_at.is_some());
        assert_eq!(row.detail, "agent agent-one, waiting false, ending true");

        let resolved = catalog
            .resolve(ResumeRequest::Id("sess-1"))
            .await
            .expect("resolve id");
        assert_eq!(resolved.kind, KIND_STATE);
        assert_eq!(resolved.history.len(), 2);
        assert!(resolved.plan.is_none());
        assert_eq!(
            resolved.summary,
            "resuming session sess-1 from the persistent session record (2 messages restored)"
        );

        let continued = catalog
            .resolve(ResumeRequest::Continue)
            .await
            .expect("resolve latest");
        assert_eq!(continued.id, "sess-1");
        assert!(continued
            .summary
            .starts_with("resuming the most recent session sess-1"));

        let pruned = catalog.prune(None, true, false).await.expect("prune");
        assert_eq!(pruned.len(), 1);
        assert!(catalog.list().await.expect("list after prune").is_empty());
    }

    #[tokio::test]
    async fn a_cli_record_resolves_with_its_transcript() {
        let fixtures = TempWorkspace::new().expect("temp workspace");
        let options = options(&fixtures);
        let store = SessionStore::new(&options.home);
        store
            .new_session("cli-1", "mock-model")
            .expect("new session");
        store
            .append_message("cli-1", "user", "hi")
            .expect("append user");
        store
            .append_message("cli-1", "assistant", "hello")
            .expect("append assistant");

        let catalog = Catalog::for_options(&options, None);
        let resolved = catalog
            .resolve(ResumeRequest::Id("cli-1"))
            .await
            .expect("resolve id");
        assert_eq!(resolved.kind, KIND_CLI);
        assert_eq!(resolved.history.len(), 2);
        assert_eq!(resolved.history[0].role.as_str(), "user");
        assert_eq!(
            resolved.summary,
            "resuming session cli-1 from the CLI session record (2 messages)"
        );
        assert_eq!(
            resolved.record.expect("cli record").model,
            "mock-model".to_owned()
        );

        let (row, shown) = catalog.show("cli-1").await.expect("show");
        assert_eq!(row.kind, KIND_CLI);
        assert_eq!(shown.id, "cli-1");
    }

    #[tokio::test]
    async fn an_unknown_id_and_an_empty_store_are_typed_not_found() {
        let fixtures = TempWorkspace::new().expect("temp workspace");
        let options = options(&fixtures);
        let catalog = Catalog::for_options(&options, None);
        let missing = catalog
            .resolve(ResumeRequest::Id("nope"))
            .await
            .expect_err("unknown id");
        assert!(missing.to_string().contains("unknown session: nope"));
        let empty = catalog
            .resolve(ResumeRequest::Continue)
            .await
            .expect_err("nothing to continue");
        assert!(empty.to_string().contains("no resumable session found"));
        let invalid = catalog
            .prune(None, false, false)
            .await
            .expect_err("no selector");
        assert!(invalid.to_string().contains("--older-than"));
    }
}
