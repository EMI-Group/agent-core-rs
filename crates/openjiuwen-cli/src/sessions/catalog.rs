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
//! One CLI run writes to all three stores, so [`Catalog::list`] folds the entries
//! that belong to the same `(agent id, session id)` into **one [`Row`] per
//! session**, richest source first (`state` > `cli` > `execution`), newest-first
//! overall. The other members of a group are kept in [`Row::sources`], so
//! `--kind` still finds the stores a session used and `prune` deletes all of
//! them.
//!
//! [`Catalog::resolve`] turns an id (or "the most recent one") into a
//! [`Resolved`] turn: the session id to continue, the history to seed, the
//! optional `deepagent` state, and the [`ResumePlan`] when a journal was traced.
//! [`ResumeRequest::Continue`] takes the newest resumable row and then resolves
//! it through the documented precedence (`state` > `cli` > `execution`).
//! [`Catalog::prune`] deletes every source of the rows it selects, optionally as
//! a dry run.
//!
//! Everything here is read-only apart from [`Catalog::prune`], so listing works
//! on a home and a workspace that were never materialised.
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::prompt::BaseMessage;
use openjiuwen_harness::cli::{
    session_id, CliOptions, MessageRecord, SessionRecord, SessionStore, SessionSummary,
};
use openjiuwen_harness::{
    resume, Capacity, Continuation, ExecutionCatalog, ExecutionSummary, FileLinkStore,
    FileSessionStore, LogCapacity, PersistentAgentLayout, PersistentSessionRecord, ResumePlan,
    RootLogSource, SessionBinding, SessionStateStore, SessionSummary as StateSummary,
    WindowAssembler, Workspace,
};

/// Kind tag of a CLI session record.
pub(crate) const KIND_CLI: &str = "cli";
/// Kind tag of a persistent-agent execution journal.
pub(crate) const KIND_EXECUTION: &str = "execution";
/// Kind tag of a persistent-agent session record.
pub(crate) const KIND_STATE: &str = "state";
/// Agent id a CLI run binds its persistent-agent state to.
///
/// A CLI session record carries no agent id, so the catalogue keys those records
/// under this id to fold them together with the `state` and `execution` entries
/// the same run leaves behind.
///
/// It mirrors the pinned card id `crates/openjiuwen-harness/src/cli/run.rs`
/// journals CLI runs with (`CLI_AGENT_ID` there, private and not re-exported by
/// `openjiuwen_harness::cli`); only the CLI writes CLI session records, so the
/// two spellings have to agree.
pub(crate) const CLI_AGENT_ID: &str = "openjiuwen-cli";
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

impl Detail {
    /// Returns the kind tag of the store this entry came from.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Cli(_) => KIND_CLI,
            Self::Execution(_) => KIND_EXECUTION,
            Self::State(_) => KIND_STATE,
        }
    }

    /// Returns the `(agent id, session id)` pair the entry belongs to.
    ///
    /// A CLI record is keyed under [`CLI_AGENT_ID`] because it carries no agent
    /// id; an execution with no session id keeps its own identity, so unrelated
    /// session-less executions never collapse into one row.
    fn group_key(&self) -> (String, String) {
        match self {
            Self::Cli(summary) => (CLI_AGENT_ID.to_owned(), summary.session_id.clone()),
            Self::Execution(summary) => {
                let session = if summary.session_id.is_empty() {
                    summary.execution_id.clone()
                } else {
                    summary.session_id.clone()
                };
                (summary.agent_id.clone(), session)
            }
            Self::State(summary) => (
                summary.binding.agent_id.clone(),
                summary.binding.session_id.clone(),
            ),
        }
    }

    /// Returns how much a resume restores from this entry: `state` 3, `cli` 2,
    /// `execution` 1.
    fn richness(&self) -> u8 {
        match self {
            Self::State(_) => 3,
            Self::Cli(_) => 2,
            Self::Execution(_) => 1,
        }
    }
}

/// One persisted session as the catalogue reports it.
///
/// After the merge there is exactly one row per session, described by its
/// richest source; the JSON shape is the row's own fields, never the group.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Row {
    /// Session (or execution) identifier of the richest source.
    pub(crate) id: String,
    /// Which store the row is reported from: `cli`, `execution` or `state`.
    pub(crate) kind: &'static str,
    /// RFC 3339 creation timestamp, when any source of the session records one.
    pub(crate) created_at: Option<String>,
    /// One-line description of what the row holds.
    pub(crate) detail: String,
    /// The store entry behind the row.
    #[serde(skip)]
    pub(crate) data: Detail,
    /// The other stores holding the same session, richest first then by id.
    #[serde(skip)]
    pub(crate) sources: Vec<Detail>,
}

impl Row {
    /// Returns whether any store behind the row is of `kind`.
    ///
    /// A CLI run leaves state in all three stores but is listed once, from its
    /// richest one (`state`); `--kind cli` still finds it, because the CLI record
    /// is one of the row's sources.
    pub(crate) fn has_kind(&self, kind: &str) -> bool {
        self.data.kind() == kind || self.sources.iter().any(|detail| detail.kind() == kind)
    }

    /// Returns whether a resume could restore anything from the row.
    ///
    /// Only a group made entirely of unusable execution journals is
    /// unresumable: every other source carries a window or a transcript.
    pub(crate) fn is_resumable(&self) -> bool {
        std::iter::once(&self.data)
            .chain(self.sources.iter())
            .any(|detail| !matches!(detail, Detail::Execution(summary) if !summary.usable))
    }
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

/// What one prune did: the rows it selected and the deletions that failed.
#[derive(Debug, Clone)]
pub(crate) struct PruneReport {
    /// Rows the selectors matched, in listing order.
    pub(crate) pruned: Vec<Row>,
    /// One line per store entry that could not be deleted, naming the row.
    pub(crate) failures: Vec<String>,
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

    /// Lists every persisted session, newest first.
    ///
    /// The three stores are read into rows and then folded by `(agent id,
    /// session id)`, so a conversation that left state everywhere is listed once
    /// — from its richest source — with the rest kept in [`Row::sources`].
    ///
    /// This is the only command that reads every store: `show`, `run --resume`
    /// and `--continue` resolve a single id directly (see [`Catalog::resolve_detail`]).
    ///
    /// # Errors
    ///
    /// Returns whatever the three stores report when a root exists but cannot be
    /// read at all; a single damaged record is skipped by the store itself.
    pub(crate) async fn list(&self) -> Result<Vec<Row>> {
        let mut rows = self.cli_rows().await?;
        rows.extend(self.execution_rows().await?);
        rows.extend(self.state_rows().await?);
        let mut rows = merge_rows(rows);
        sort_rows(&mut rows);
        Ok(rows)
    }

    /// Builds one row per CLI session record.
    async fn cli_rows(&self) -> Result<Vec<Row>> {
        Ok(self
            .list_cli()
            .await?
            .into_iter()
            .map(Self::cli_row)
            .collect())
    }

    /// Builds one row per persistent-agent execution journal.
    async fn execution_rows(&self) -> Result<Vec<Row>> {
        Ok(self
            .execution_catalog()
            .list()
            .await?
            .into_iter()
            .map(Self::execution_row)
            .collect())
    }

    /// Builds one row per persistent-agent session record.
    ///
    /// A record has no `created_at`, so the row is stamped with the `state.json`
    /// modification time.
    async fn state_rows(&self) -> Result<Vec<Row>> {
        let mut rows = Vec::new();
        for summary in self.state_store().list().await? {
            rows.push(Self::state_row(summary).await);
        }
        Ok(rows)
    }

    /// Builds the row of one CLI session record.
    fn cli_row(summary: SessionSummary) -> Row {
        Row {
            id: summary.session_id.clone(),
            kind: KIND_CLI,
            created_at: non_empty(&summary.created_at),
            detail: format!("{} messages, model {}", summary.messages, summary.model),
            data: Detail::Cli(summary),
            sources: Vec::new(),
        }
    }

    /// Builds the row of one execution journal.
    fn execution_row(summary: ExecutionSummary) -> Row {
        Row {
            id: summary.execution_id.clone(),
            kind: KIND_EXECUTION,
            created_at: non_empty(&summary.created_at),
            detail: format!(
                "{} entries, session {}, usable {}",
                summary.entries, summary.session_id, summary.usable
            ),
            data: Detail::Execution(summary),
            sources: Vec::new(),
        }
    }

    /// Builds the row of one persistent session record, stamped with its `state.json` mtime.
    async fn state_row(summary: StateSummary) -> Row {
        Row {
            id: summary.binding.session_id.clone(),
            kind: KIND_STATE,
            created_at: state_timestamp(&summary).await.map(format_timestamp),
            detail: format!(
                "agent {}, waiting {}, ending {}",
                summary.binding.agent_id, summary.waiting, summary.has_ending
            ),
            data: Detail::State(summary),
            sources: Vec::new(),
        }
    }

    /// Runs one blocking session-record operation on the blocking pool.
    ///
    /// `SessionStore::load`/`list` are synchronous `std::fs` reads of whole
    /// transcripts, so running them on a worker would stall every other task on
    /// that thread for the length of the read.
    async fn off_reactor<T, F>(label: &'static str, work: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        tokio::task::spawn_blocking(work)
            .await
            .map_err(|error| Error::session(format!("{label} failed to run: {error}")))?
    }

    /// Lists the CLI session records off the reactor.
    async fn list_cli(&self) -> Result<Vec<SessionSummary>> {
        let store = self.cli_store();
        Self::off_reactor("listing the CLI session records", move || store.list()).await
    }

    /// Loads one CLI session record off the reactor.
    async fn load_cli(&self, id: &str) -> Result<Option<SessionRecord>> {
        let store = self.cli_store();
        let id = id.to_owned();
        Self::off_reactor("loading a CLI session record", move || store.load(&id)).await
    }

    /// Summarises a loaded CLI record the way `SessionStore::list` reports it.
    fn cli_summary(id: &str, record: &SessionRecord) -> SessionSummary {
        SessionSummary {
            session_id: if record.session_id.is_empty() {
                id.to_owned()
            } else {
                record.session_id.clone()
            },
            model: record.model.clone(),
            created_at: record.created_at.clone(),
            messages: record.messages.len(),
        }
    }

    /// Summarises a loaded persistent record the way `FileSessionStore::list` reports it.
    fn state_summary(
        binding: &SessionBinding,
        record: &PersistentSessionRecord,
        state_path: PathBuf,
    ) -> StateSummary {
        StateSummary {
            binding: binding.clone(),
            waiting: record.is_waiting(),
            current_window_id: record.current_window_id.clone(),
            window_seq: record.window_seq,
            has_checkpoint: record.checkpoint.is_some(),
            has_ending: record.ending.is_some(),
            state_path,
        }
    }

    /// Resolves `request` into the turn a resumed run continues.
    ///
    /// Both forms resolve through the same precedence — a persistent session
    /// record, then a CLI session record, then an execution journal, all by their
    /// own identifier. They differ in how the id is chosen:
    /// [`ResumeRequest::Id`] takes the caller's id, while
    /// [`ResumeRequest::Continue`] takes the newest resumable row of
    /// [`Catalog::list`] and then resolves that row's richest source.
    ///
    /// # Errors
    ///
    /// `Error::not_found` when nothing matches, and whatever the stores, the
    /// journal reader or [`resume`] report.
    pub(crate) async fn resolve(&self, request: ResumeRequest<'_>) -> Result<Resolved> {
        match request {
            ResumeRequest::Id(id) => self.resolve_id(id, false).await,
            ResumeRequest::Continue => self.resolve_most_recent().await,
        }
    }

    /// Resolves one id through the documented precedence.
    ///
    /// `most_recent` only changes the note: a `--continue` run says it continued
    /// the most recent session, not the one the user named.
    async fn resolve_id(&self, id: &str, most_recent: bool) -> Result<Resolved> {
        let (resolved, _) = self.resolve_detail(id, most_recent).await?;
        Ok(resolved)
    }

    /// Resolves one id directly, without listing the stores, and reports its entry.
    ///
    /// Each store is probed by the id it owns — `FileSessionStore::load` for a
    /// persistent record, `SessionStore::load` for a CLI record,
    /// `ExecutionCatalog::open` for a journal — so a resolution reads the records
    /// behind the id instead of every transcript in the home. The returned
    /// [`Detail`] is what a listing would have reported for that id, so
    /// [`Catalog::show`] never lists either.
    async fn resolve_detail(&self, id: &str, most_recent: bool) -> Result<(Resolved, Detail)> {
        if let Some(found) = self.resolve_state_by_id(id, most_recent).await? {
            return Ok(found);
        }
        if let Some(found) = self.resolve_cli_by_id(id, most_recent).await? {
            return Ok(found);
        }
        if let Some(found) = self.resolve_execution_by_id(id, most_recent).await? {
            return Ok(found);
        }
        Err(Error::not_found(format!("unknown session: {id}")))
    }

    /// Resolves a persistent session record by session id.
    ///
    /// The CLI binds its state under [`CLI_AGENT_ID`], so that binding is loaded
    /// first and the store is only scanned (record heads, no transcripts) when it
    /// misses, which is the case for a session a persistent agent named itself.
    async fn resolve_state_by_id(
        &self,
        id: &str,
        most_recent: bool,
    ) -> Result<Option<(Resolved, Detail)>> {
        if let Ok(binding) = SessionBinding::new(CLI_AGENT_ID, id) {
            if let Some(found) = self.resolve_state_binding(&binding, most_recent).await? {
                return Ok(Some(found));
            }
        }
        let store = self.state_store();
        let Some(summary) = store
            .list()
            .await?
            .into_iter()
            .find(|summary| summary.binding.session_id == id)
        else {
            return Ok(None);
        };
        self.resolve_state_binding(&summary.binding, most_recent)
            .await
    }

    /// Loads one persistent record by binding and resolves it.
    ///
    /// `Ok(None)` means no record exists for the binding, so the caller falls
    /// through to the next store.
    async fn resolve_state_binding(
        &self,
        binding: &SessionBinding,
        most_recent: bool,
    ) -> Result<Option<(Resolved, Detail)>> {
        let store = self.state_store();
        let Some(record) = store.load(binding).await? else {
            return Ok(None);
        };
        let summary = Self::state_summary(binding, &record, store.path_for(binding));
        let resolved = self.resolve_state(&summary, &record, most_recent).await?;
        Ok(Some((resolved, Detail::State(summary))))
    }

    /// Rebuilds the window and the resume plan of one loaded persistent record.
    async fn resolve_state(
        &self,
        summary: &StateSummary,
        record: &PersistentSessionRecord,
        most_recent: bool,
    ) -> Result<Resolved> {
        let binding = summary.binding.clone();
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
                &self.state_store(),
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
            return Ok(Resolved {
                id: binding.session_id,
                kind: KIND_STATE,
                summary: note,
                history,
                state,
                plan: Some(plan),
                record: None,
            });
        }
        let history = record.messages.clone();
        let note = format!(
            "{prefix} from the persistent session record ({} messages restored)",
            history.len()
        );
        Ok(Resolved {
            id: binding.session_id,
            kind: KIND_STATE,
            summary: note,
            history,
            state: record
                .deepagent_state
                .clone()
                .filter(serde_json::Value::is_object),
            plan: None,
            record: None,
        })
    }

    /// Resolves one CLI record by session id.
    async fn resolve_cli_by_id(
        &self,
        id: &str,
        most_recent: bool,
    ) -> Result<Option<(Resolved, Detail)>> {
        let Some(record) = self.load_cli(id).await? else {
            return Ok(None);
        };
        let summary = Self::cli_summary(id, &record);
        let resolved = Self::resolve_cli(&summary, &record, most_recent);
        Ok(Some((resolved, Detail::Cli(summary))))
    }

    /// Maps one loaded CLI record onto a resumable turn.
    fn resolve_cli(
        summary: &SessionSummary,
        record: &SessionRecord,
        most_recent: bool,
    ) -> Resolved {
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
        Resolved {
            id,
            kind: KIND_CLI,
            summary: note,
            history,
            state: None,
            plan: None,
            record: Some(record.clone()),
        }
    }

    /// Resolves one execution journal by its own id.
    ///
    /// `ExecutionCatalog::open` is the direct lookup: it reads the journal's own
    /// metadata and segments instead of scanning and re-reading every journal.
    async fn resolve_execution_by_id(
        &self,
        id: &str,
        most_recent: bool,
    ) -> Result<Option<(Resolved, Detail)>> {
        let catalog = self.execution_catalog();
        let log = match catalog.open(id).await {
            Ok(log) => log,
            Err(Error::NotFound(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let snapshot = log.read().await?;
        let identity = log.identity();
        // The journal reader exposes no creation timestamp without a catalogue
        // scan, so `show` reports `-` for it; `list` still has the real value.
        let summary = ExecutionSummary {
            execution_id: identity.execution_id.clone(),
            namespace_id: identity.namespace_id.clone(),
            agent_id: identity.agent_id.clone(),
            session_id: identity.session_id.clone(),
            request_id: identity.request_id.clone(),
            source_ref: identity.source_ref.clone(),
            created_at: String::new(),
            generation: log.cursor().await.generation,
            entries: snapshot.entries.len(),
            last_seq: snapshot.entries.last().map(|entry| entry.seq),
            usable: snapshot.is_usable(),
        };
        let resolved = Self::resolve_execution(&summary, &snapshot, most_recent);
        Ok(Some((resolved, Detail::Execution(summary))))
    }

    /// Reassembles the window of one open journal.
    fn resolve_execution(
        summary: &ExecutionSummary,
        snapshot: &openjiuwen_harness::LogSnapshot,
        most_recent: bool,
    ) -> Resolved {
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
        Resolved {
            id,
            kind: KIND_EXECUTION,
            summary: summary_text,
            history,
            state: None,
            plan: None,
            record: None,
        }
    }

    /// Resolves the newest resumable session.
    ///
    /// The merged listing is newest-first, so the first row a resume could
    /// restore is the session to continue; ties break by id. That row is then
    /// resolved from its richest source directly, so `--continue` pays for one
    /// listing plus one record read, never a second scan.
    async fn resolve_most_recent(&self) -> Result<Resolved> {
        for row in self.list().await?.iter().filter(|row| row.is_resumable()) {
            if let Some(resolved) = self.resolve_row(row, true).await? {
                return Ok(resolved);
            }
        }
        Err(Error::not_found("no resumable session found"))
    }

    /// Resolves one listed row from its richest source, without listing again.
    ///
    /// A source that disappeared between the listing and the load falls through
    /// to the next one, and `Ok(None)` means every source of the row is gone.
    async fn resolve_row(&self, row: &Row, most_recent: bool) -> Result<Option<Resolved>> {
        for detail in std::iter::once(&row.data).chain(row.sources.iter()) {
            if let Some(resolved) = self.resolve_detail_entry(detail, most_recent).await? {
                return Ok(Some(resolved));
            }
        }
        Ok(None)
    }

    /// Resolves one already-listed store entry by reading only its own record.
    async fn resolve_detail_entry(
        &self,
        detail: &Detail,
        most_recent: bool,
    ) -> Result<Option<Resolved>> {
        match detail {
            Detail::State(summary) => Ok(self
                .resolve_state_binding(&summary.binding, most_recent)
                .await?
                .map(|(resolved, _)| resolved)),
            Detail::Cli(summary) => {
                let Some(record) = self.load_cli(&summary.session_id).await? else {
                    return Ok(None);
                };
                Ok(Some(Self::resolve_cli(summary, &record, most_recent)))
            }
            Detail::Execution(summary) => {
                let catalog = self.execution_catalog();
                let log = match catalog.open(&summary.execution_id).await {
                    Ok(log) => log,
                    Err(Error::NotFound(_)) => return Ok(None),
                    Err(error) => return Err(error),
                };
                let snapshot = log.read().await?;
                Ok(Some(Self::resolve_execution(
                    summary,
                    &snapshot,
                    most_recent,
                )))
            }
        }
    }

    /// Returns the row and the resolution behind one id.
    ///
    /// The row is built from the entry the resolution read — its kind, its
    /// creation timestamp and its detail line — so `show` costs one record read
    /// instead of a full listing. A journal is reported without a creation
    /// timestamp, because the journal reader exposes none without a scan.
    ///
    /// # Errors
    ///
    /// Everything [`Catalog::resolve`] reports.
    pub(crate) async fn show(&self, id: &str) -> Result<(Row, Resolved)> {
        let (resolved, detail) = self.resolve_detail(id, false).await?;
        let mut row = match &detail {
            Detail::Cli(summary) => Self::cli_row(summary.clone()),
            Detail::Execution(summary) => Self::execution_row(summary.clone()),
            Detail::State(summary) => Self::state_row(summary.clone()).await,
        };
        // `show` reports the id the resume continues, which is the entry's own id
        // except for a journal reached through its execution id.
        row.id.clone_from(&resolved.id);
        row.kind = resolved.kind;
        Ok((row, resolved))
    }

    /// Selects (and, unless `dry_run`, deletes) persisted candidates.
    ///
    /// `all` selects every row; otherwise `older_than_days` selects the rows
    /// whose creation timestamp is older than that many days. Rows without a
    /// timestamp are only selected by `all`, never by an age filter.
    ///
    /// Deletion is best effort: one unremovable store entry never stops the
    /// others, and every failure is reported back in the [`PruneReport`], so a
    /// prune that could not finish says so instead of stopping silently.
    ///
    /// # Errors
    ///
    /// `Error::validation` when neither selector is given, plus the listing errors
    /// of the stores. Deletion failures are reported, not returned.
    pub(crate) async fn prune(
        &self,
        older_than_days: Option<f64>,
        all: bool,
        dry_run: bool,
    ) -> Result<PruneReport> {
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
            return Ok(PruneReport {
                pruned: selected,
                failures: Vec::new(),
            });
        }
        let mut failures = Vec::new();
        for row in &selected {
            failures.extend(self.remove(row).await);
        }
        Ok(PruneReport {
            pruned: selected,
            failures,
        })
    }

    /// Deletes the store entries behind one row, returning the deletions that failed.
    ///
    /// A session that left state in several stores is deleted from **every** one
    /// of them: deleting only the richest source would let the session reappear
    /// from a leftover record on the next listing. A failure on one entry does not
    /// stop the next one, so a half-deleted session stays prunable.
    async fn remove(&self, row: &Row) -> Vec<String> {
        let mut failures = Vec::new();
        for detail in std::iter::once(&row.data).chain(row.sources.iter()) {
            if let Err(error) = self.remove_detail(detail).await {
                failures.push(format!("{} {}: {error}", detail.kind(), row.id));
            }
        }
        failures
    }

    /// Deletes the store entry behind one [`Detail`].
    async fn remove_detail(&self, detail: &Detail) -> Result<()> {
        match detail {
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
        compare_timestamps(left.created_at.as_deref(), right.created_at.as_deref())
            .then_with(|| left.id.cmp(&right.id))
    });
}

/// Compares two timestamps newest first, treating a missing one as oldest.
fn compare_timestamps(left: Option<&str>, right: Option<&str>) -> Ordering {
    match (parse_timestamp(left), parse_timestamp(right)) {
        (Some(left), Some(right)) => right.cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Folds rows that describe the same conversation into one row per session.
///
/// The grouping key is the `(agent id, session id)` pair of the entry behind the
/// row (see [`Detail::group_key`]). Groups keep the order of their first member,
/// and [`fold_group`] is deterministic, so the merge itself never reorders the
/// catalogue: [`sort_rows`] does that afterwards.
fn merge_rows(rows: Vec<Row>) -> Vec<Row> {
    let mut groups: Vec<((String, String), Vec<Row>)> = Vec::new();
    for row in rows {
        let key = row.data.group_key();
        match groups.iter_mut().find(|(existing, _)| *existing == key) {
            Some((_, members)) => members.push(row),
            None => groups.push((key, vec![row])),
        }
    }
    groups
        .into_iter()
        .filter_map(|(_, members)| fold_group(members))
        .collect()
}

/// Folds the members of one group into the single row the catalogue reports.
///
/// The kept member is the richest one (`state` > `cli` > `execution`), then the
/// newest, then the one with the smallest id; its own timestamp is kept, and when
/// it has none the row inherits the newest timestamp of the group, so ordering
/// and `--older-than` still work. Every other member becomes a source, richest
/// first then by id, so `--kind` and `prune` still reach it.
fn fold_group(mut members: Vec<Row>) -> Option<Row> {
    members.sort_by(by_richness);
    let mut remaining = members.into_iter();
    let mut kept = remaining.next()?;
    let rest: Vec<Row> = remaining.collect();
    if kept.created_at.is_none() {
        kept.created_at = newest_timestamp(rest.iter().filter_map(|row| row.created_at.as_deref()));
    }
    let mut rest = rest;
    rest.sort_by(by_kind_then_id);
    kept.sources = rest.into_iter().map(|row| row.data).collect();
    Some(kept)
}

/// Orders group members by richness: richest first, then newest, then by id.
fn by_richness(left: &Row, right: &Row) -> Ordering {
    right
        .data
        .richness()
        .cmp(&left.data.richness())
        .then_with(|| compare_timestamps(left.created_at.as_deref(), right.created_at.as_deref()))
        .then_with(|| left.id.cmp(&right.id))
}

/// Orders source entries richest first, then by the id they are reported under.
fn by_kind_then_id(left: &Row, right: &Row) -> Ordering {
    right
        .data
        .richness()
        .cmp(&left.data.richness())
        .then_with(|| left.id.cmp(&right.id))
}

/// Returns the newest parseable timestamp among `values`.
fn newest_timestamp<'a>(values: impl Iterator<Item = &'a str>) -> Option<String> {
    values
        .filter_map(|value| parse_timestamp(Some(value)).map(|at| (at, value)))
        .max_by_key(|(at, _)| *at)
        .map(|(_, value)| value.to_owned())
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
mod tests;
