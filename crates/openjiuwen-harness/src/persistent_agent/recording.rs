//! Recording a live turn as a persistent, resumable execution.
//!
//! The rest of this module tree is a set of stores and pure functions: a journal
//! ([`ExecutionLog`]), a session record store ([`FileSessionStore`]), a window assembler and the
//! continuation planner. Nothing in it binds the three together for a running agent, because the
//! module deliberately owns no agent loop. This file is that binding, kept as thin as the stores
//! allow:
//!
//! 1. [`ExecutionRecorder::begin`] opens the journal of a fresh execution under
//!    [`PersistentAgentLayout::logs_root`] and appends the [`EntryPayload::ExecutionStarted`] goal,
//!    which is the entry [`crate::persistent_agent::ResumePlan::seed_history`] rebuilds the
//!    question from;
//! 2. [`ActiveExecution::record_model_turn`] / [`ActiveExecution::finish`] append the model turn
//!    and the terminal outcome, so the assembled window carries both the question and the answer;
//! 3. [`ActiveExecution::save_session`] writes the persisted session record the continuation
//!    planner loads, so [`crate::persistent_agent::resume`] has a record to resume from.
//!
//! The recorder never calls a model, never renders anything and never decides a turn's outcome —
//! the caller does, and hands the result in. Every id a caller does not supply is minted here
//! (a `uuid` v4 in the `req_`/`exec_` shape), so a plain
//! [`PersistentRunConfig::new`] is enough to journal a turn.
use std::sync::atomic::{AtomicUsize, Ordering};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::message::BaseMessage;
use serde_json::Value;

use crate::workspace::Workspace;

use super::entry::{EntryPayload, FinalStatus, ModelUsage, DETAIL_LIMIT};
use super::identity::{ExecutionIdentity, SessionBinding};
use super::layout::PersistentAgentLayout;
use super::log::{ExecutionLog, LogCapacity};
use super::rails::deterministic_model_call_id;
use super::session::{FileSessionStore, PersistentSessionRecord, RecordEnding, SessionStateStore};
use super::util::truncate_chars;

/// Namespace an execution is recorded under when the caller names none.
pub const DEFAULT_NAMESPACE_ID: &str = "deepagent";

/// Source reference an execution is recorded with when the caller names none.
pub const DEFAULT_SOURCE_REF: &str = "deepagent";

/// Outcome a persisted session record is closed with by [`ActiveExecution::save_session`].
///
/// The record is written after the turn finished, so `completed` is the honest marker even when the
/// turn itself failed: the session ended, and the failure is the journal's `execution_final` entry.
pub const SESSION_ENDING_OUTCOME: &str = "completed";

/// Why a model turn is recorded without token counts.
///
/// The harness records what it can see: the answer text. Per-call token accounting lives in the
/// kernel's usage telemetry, so claiming zero counts here would be a lie and the journal marks the
/// counts unknown instead.
pub const USAGE_UNKNOWN_REASON: &str =
    "the harness records the answer text only; per-call token usage lives in the kernel telemetry";

/// What one recorded execution is called.
///
/// `agent_id` and `session_id` are required and must be identifiers; the remaining fields are
/// optional overrides whose defaults keep a caller's `new` call sufficient.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentRunConfig {
    /// Agent that runs the execution.
    pub agent_id: String,
    /// Session the execution belongs to.
    pub session_id: String,
    /// Request that started the execution; a fresh `req_…` id when absent.
    pub request_id: Option<String>,
    /// Host namespace that owns the execution; [`DEFAULT_NAMESPACE_ID`] when absent.
    pub namespace_id: Option<String>,
    /// Host correlation reference; [`DEFAULT_SOURCE_REF`] when absent.
    pub source_ref: Option<String>,
    /// Retention limits of the journal.
    pub capacity: LogCapacity,
}

impl PersistentRunConfig {
    /// Creates a configuration with the default capacity and no optional ids.
    #[must_use]
    pub fn new(agent_id: impl Into<String>, session_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            session_id: session_id.into(),
            request_id: None,
            namespace_id: None,
            source_ref: None,
            capacity: LogCapacity::default(),
        }
    }

    /// Overrides the request id.
    #[must_use]
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Overrides the namespace id.
    #[must_use]
    pub fn with_namespace_id(mut self, namespace_id: impl Into<String>) -> Self {
        self.namespace_id = Some(namespace_id.into());
        self
    }

    /// Overrides the source reference.
    #[must_use]
    pub fn with_source_ref(mut self, source_ref: impl Into<String>) -> Self {
        self.source_ref = Some(source_ref.into());
        self
    }

    /// Overrides the journal capacity.
    #[must_use]
    pub fn with_capacity(mut self, capacity: LogCapacity) -> Self {
        self.capacity = capacity;
        self
    }

    /// Resolves the execution identity, minting the ids the caller left out.
    ///
    /// # Errors
    /// `Error::Validation` when the agent or session id is not an identifier, or when a supplied
    /// optional id is not one.
    pub fn identity(&self) -> Result<ExecutionIdentity> {
        ExecutionIdentity::new(
            self.namespace_id
                .clone()
                .unwrap_or_else(|| DEFAULT_NAMESPACE_ID.to_string()),
            self.agent_id.clone(),
            self.session_id.clone(),
            self.request_id.clone().unwrap_or_else(|| fresh_id("req_")),
            fresh_id("exec_"),
            self.source_ref
                .clone()
                .unwrap_or_else(|| DEFAULT_SOURCE_REF.to_string()),
        )
    }
}

/// Returns a fresh, identifier-safe id: `prefix` plus a UUID v4 in simple hex form.
#[must_use]
pub fn fresh_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

/// Starts and finalises executions in the workspace's persistent-agent layout.
///
/// The recorder holds no open handle: [`Self::begin`] opens a
/// [`ExecutionLog`] per execution, so one recorder can journal any number of turns of any number
/// of sessions, and a recorder that is dropped leaves nothing to flush.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRecorder {
    /// Where the journals and the session records live.
    layout: PersistentAgentLayout,
    /// Retention limits every journal is opened with.
    capacity: LogCapacity,
}

impl ExecutionRecorder {
    /// Builds a recorder over `layout` with the default log capacity.
    #[must_use]
    pub fn new(layout: PersistentAgentLayout) -> Self {
        Self {
            layout,
            capacity: LogCapacity::default(),
        }
    }

    /// Builds a recorder over the layout of `workspace`.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Self {
        Self::new(PersistentAgentLayout::for_workspace(workspace))
    }

    /// Overrides the journal capacity.
    #[must_use]
    pub fn with_capacity(mut self, capacity: LogCapacity) -> Self {
        self.capacity = capacity;
        self
    }

    /// Returns the layout the recorder writes into.
    #[must_use]
    pub fn layout(&self) -> &PersistentAgentLayout {
        &self.layout
    }

    /// Returns the capacity journals are opened with.
    #[must_use]
    pub fn capacity(&self) -> LogCapacity {
        self.capacity
    }

    /// Opens the journal of a new execution and records its goal.
    ///
    /// The journal lives at `<logs root>/<percent-encoded execution id>` and its first entry is
    /// [`EntryPayload::execution_started`] carrying `query`, which is the entry the resumed window
    /// rebuilds the question from.
    ///
    /// # Errors
    /// [`PersistentRunConfig::identity`] errors, and every error [`ExecutionLog::open`] or
    /// [`ExecutionLog::append`] reports.
    pub async fn begin(
        &self,
        config: &PersistentRunConfig,
        query: &str,
    ) -> Result<ActiveExecution> {
        let identity = config.identity()?;
        let dir = self.layout.logs_root().join(identity.directory_name());
        let log = ExecutionLog::open(dir, identity.clone(), self.capacity).await?;
        log.append(EntryPayload::execution_started(query)).await?;
        Ok(ActiveExecution {
            log,
            identity,
            store: FileSessionStore::new(self.layout.sessions_root()),
            model_turns: AtomicUsize::new(0),
        })
    }
}

/// One execution whose journal and session record are open for appending.
#[derive(Debug)]
pub struct ActiveExecution {
    /// Journal of the execution.
    log: ExecutionLog,
    /// Identity of the execution.
    identity: ExecutionIdentity,
    /// Store of the persisted session record.
    store: FileSessionStore,
    /// Number of model turns recorded so far, used to derive their deterministic ids.
    model_turns: AtomicUsize,
}

impl ActiveExecution {
    /// Returns the journal.
    #[must_use]
    pub fn log(&self) -> &ExecutionLog {
        &self.log
    }

    /// Returns the identity of the execution.
    #[must_use]
    pub fn identity(&self) -> &ExecutionIdentity {
        &self.identity
    }

    /// Returns the execution id.
    #[must_use]
    pub fn execution_id(&self) -> &str {
        &self.identity.execution_id
    }

    /// Returns the `(agent, session)` binding of the execution's session record.
    #[must_use]
    pub fn binding(&self) -> SessionBinding {
        self.identity.session_binding()
    }

    /// Returns the store of the persisted session record.
    #[must_use]
    pub fn store(&self) -> &FileSessionStore {
        &self.store
    }

    /// Appends one completed model turn.
    ///
    /// The model-call id is derived from the identity and the turn's position in this execution
    /// (see [`deterministic_model_call_id`]), so replaying the same execution produces the same
    /// ids. Token usage is recorded as unknown — see [`USAGE_UNKNOWN_REASON`].
    ///
    /// # Errors
    /// Every error [`ExecutionLog::append`] reports.
    pub async fn record_model_turn(&self, text: &str) -> Result<()> {
        let iteration = self.model_turns.fetch_add(1, Ordering::Relaxed);
        let call_id = deterministic_model_call_id(&self.identity, iteration);
        self.log
            .append(EntryPayload::model_turn(
                call_id,
                text,
                ModelUsage::unknown(USAGE_UNKNOWN_REASON),
            ))
            .await
            .map(|_| ())
    }

    /// Records the turn's answer and the terminal outcome of the execution.
    ///
    /// A non-blank `output` becomes one [`EntryPayload::model_turn`] entry before the
    /// [`EntryPayload::execution_final`] entry, so a resumed window carries the answer as well as
    /// the question. The output is also passed as the final entry's reason, bounded to
    /// [`DETAIL_LIMIT`] characters — the failure text of a failed turn is exactly what a host wants
    /// to see there, and the answer is already recorded in full by the model turn.
    ///
    /// # Errors
    /// Every error [`ExecutionLog::append`] reports; entries appended before a failure stay
    /// persisted.
    pub async fn finish(&self, status: FinalStatus, output: Option<&str>) -> Result<()> {
        if let Some(text) = output {
            if !text.trim().is_empty() {
                self.record_model_turn(text).await?;
            }
        }
        let reason = output.map(|text| truncate_chars(text, DETAIL_LIMIT));
        self.log
            .append(EntryPayload::execution_final(status, reason))
            .await
            .map(|_| ())
    }

    /// Writes the persisted session record of this execution's session.
    ///
    /// The record carries `messages` as its window, `state` as the opaque `DeepAgent` state
    /// payload and a [`SESSION_ENDING_OUTCOME`] ending naming this execution, which is exactly the
    /// shape [`crate::persistent_agent::resume`] loads.
    ///
    /// # Errors
    /// `Error::Validation` when the record cannot be built (an invalid id or an invalid ending),
    /// and every error [`SessionStateStore::save`] reports.
    pub async fn save_session(
        &self,
        messages: Vec<BaseMessage>,
        state: Option<Value>,
    ) -> Result<()> {
        let mut record = PersistentSessionRecord::new(&self.binding())?;
        record.messages = messages;
        record.deepagent_state = state;
        let record = record.with_ending(RecordEnding {
            outcome: SESSION_ENDING_OUTCOME.to_string(),
            request_id: self.identity.request_id.clone(),
            execution_id: self.identity.execution_id.clone(),
            ..RecordEnding::default()
        })?;
        self.store.save(&record).await
    }

    /// Stores a caller-built session record unchanged.
    ///
    /// Use this instead of [`Self::save_session`] when the record needs fields that method does not
    /// set (a checkpoint, an interrupted window, remembered confirmations).
    ///
    /// # Errors
    /// Every error [`SessionStateStore::save`] reports.
    pub async fn persist_session(&self, record: &PersistentSessionRecord) -> Result<()> {
        self.store.save(record).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::Ordering;

    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::prompt::message::BaseMessage;
    use serde_json::json;

    use super::{
        fresh_id, ActiveExecution, ExecutionRecorder, PersistentRunConfig, DEFAULT_NAMESPACE_ID,
        DEFAULT_SOURCE_REF, SESSION_ENDING_OUTCOME,
    };
    use crate::persistent_agent::{
        EntryKind, EntryPayload, ExecutionLog, FinalStatus, LogCapacity, PersistentAgentLayout,
        PersistentSessionRecord, SessionBinding, SessionStateStore, UsageStatus,
    };
    use crate::workspace::Workspace;

    /// A recorder over a temp workspace, with the layout it writes into.
    fn recorder() -> (tempfile::TempDir, ExecutionRecorder) {
        let dir = tempfile::TempDir::new().unwrap();
        let workspace = Workspace::for_language(dir.path(), "en");
        let recorder = ExecutionRecorder::for_workspace(&workspace);
        (dir, recorder)
    }

    #[test]
    fn config_defaults_fill_the_optional_identity_fields() {
        let config = PersistentRunConfig::new("agent", "session");
        assert_eq!(config.capacity, LogCapacity::default());
        let identity = config.identity().unwrap();
        assert_eq!(identity.namespace_id, DEFAULT_NAMESPACE_ID);
        assert_eq!(identity.source_ref, DEFAULT_SOURCE_REF);
        assert_eq!(identity.agent_id, "agent");
        assert_eq!(identity.session_id, "session");
        assert!(identity.request_id.starts_with("req_"));
        assert!(identity.execution_id.starts_with("exec_"));
        // Validation is part of `new`, so a second resolution mints a different execution.
        assert_ne!(
            config.identity().unwrap().execution_id,
            identity.execution_id
        );
    }

    #[test]
    fn config_overrides_are_used_verbatim() {
        let config = PersistentRunConfig::new("agent", "session")
            .with_namespace_id("ns")
            .with_request_id("req-1")
            .with_source_ref("cli")
            .with_capacity(LogCapacity::default());
        let identity = config.identity().unwrap();
        assert_eq!(identity.namespace_id, "ns");
        assert_eq!(identity.request_id, "req-1");
        assert_eq!(identity.source_ref, "cli");
    }

    #[test]
    fn invalid_ids_are_rejected() {
        let error = PersistentRunConfig::new("agent name", "session")
            .identity()
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("agent_id"));
        assert!(PersistentRunConfig::new("agent", "")
            .with_request_id("bad id")
            .identity()
            .is_err());
    }

    #[test]
    fn fresh_ids_are_identifier_safe_and_unique() {
        let first = fresh_id("req_");
        assert!(first.starts_with("req_"));
        assert_eq!(first.len(), 4 + 32);
        assert!(first[4..].chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_ne!(first, fresh_id("req_"));
    }

    #[tokio::test]
    async fn begin_journals_the_goal_and_exposes_the_identity() {
        let (dir, recorder) = recorder();
        assert_eq!(recorder.capacity(), LogCapacity::default());
        assert_eq!(
            recorder.layout(),
            &PersistentAgentLayout::for_workspace(&Workspace::for_language(dir.path(), "en"))
        );

        let config = PersistentRunConfig::new("agent", "session").with_request_id("req-1");
        let active = recorder.begin(&config, "hello question").await.unwrap();
        assert_eq!(active.identity().request_id, "req-1");
        assert_eq!(active.binding().agent_id, "agent");
        assert_eq!(active.binding().session_id, "session");
        assert_eq!(
            active.log().dir(),
            recorder
                .layout()
                .logs_root()
                .join(active.identity().directory_name())
                .as_path()
        );

        // The goal is on disk as the first entry, which is what a resume rebuilds.
        let snapshot = active.log().read().await.unwrap();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].kind(), EntryKind::ExecutionStarted);
        assert_eq!(
            snapshot.entries[0].payload,
            EntryPayload::execution_started("hello question")
        );
    }

    #[tokio::test]
    async fn model_turns_get_deterministic_ids_and_unknown_usage() {
        let (_dir, recorder) = recorder();
        let active = recorder
            .begin(&PersistentRunConfig::new("agent", "session"), "q")
            .await
            .unwrap();
        active.record_model_turn("partial").await.unwrap();
        active.record_model_turn("final").await.unwrap();

        let entries = active.log().read().await.unwrap().entries;
        assert_eq!(entries.len(), 3);
        let first = match &entries[1].payload {
            EntryPayload::ModelTurn {
                model_call_id,
                response,
                usage,
            } => (model_call_id.clone(), response.clone(), usage.clone()),
            other => panic!("expected a model turn, got {other:?}"),
        };
        assert_eq!(first.1, "partial");
        assert_eq!(first.2.status, UsageStatus::Unknown);
        assert!(first.2.reason.is_some());
        assert_eq!(
            first.0,
            crate::persistent_agent::deterministic_model_call_id(active.identity(), 0)
        );
        // The second turn uses the next iteration, so the ids differ.
        match (&entries[2].payload, &entries[1].payload) {
            (
                EntryPayload::ModelTurn {
                    model_call_id: second,
                    ..
                },
                EntryPayload::ModelTurn {
                    model_call_id: first,
                    ..
                },
            ) => assert_ne!(first, second),
            other => panic!("expected two model turns, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn finish_records_the_answer_and_the_terminal_status() {
        let (_dir, recorder) = recorder();
        let active = recorder
            .begin(&PersistentRunConfig::new("agent", "session"), "q")
            .await
            .unwrap();
        active
            .finish(FinalStatus::Completed, Some("the answer"))
            .await
            .unwrap();

        let entries = active.log().read().await.unwrap().entries;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1].kind(), EntryKind::ModelTurn);
        assert_eq!(entries[2].kind(), EntryKind::ExecutionFinal);
        assert_eq!(
            entries[2].payload,
            EntryPayload::execution_final(FinalStatus::Completed, Some("the answer".to_string()))
        );

        // A blank output records no model turn at all.
        let silent = recorder
            .begin(&PersistentRunConfig::new("agent", "silent"), "q")
            .await
            .unwrap();
        silent
            .finish(FinalStatus::Failed, Some("   "))
            .await
            .unwrap();
        let entries = silent.log().read().await.unwrap().entries;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].kind(), EntryKind::ExecutionFinal);
    }

    #[tokio::test]
    async fn a_journal_reopens_and_keeps_every_entry() {
        let (_dir, recorder) = recorder();
        let active = recorder
            .begin(&PersistentRunConfig::new("agent", "session"), "q")
            .await
            .unwrap();
        active.record_model_turn("answer").await.unwrap();
        active
            .finish(FinalStatus::Completed, Some("answer"))
            .await
            .unwrap();
        let identity = active.identity().clone();
        let dir = active.log().dir().to_path_buf();
        let entries = active.log().read().await.unwrap().entries.len();
        drop(active);

        // A fresh handle reads what reached disk: the append is durable on its own.
        let reopened = ExecutionLog::open(&dir, identity, LogCapacity::default())
            .await
            .unwrap();
        assert_eq!(reopened.read().await.unwrap().entries.len(), entries);
        assert_eq!(
            reopened.next_seq().await,
            u64::try_from(entries).unwrap() + 1
        );
    }

    #[tokio::test]
    async fn save_session_writes_the_record_a_resume_loads() {
        let (_dir, recorder) = recorder();
        let active = recorder
            .begin(&PersistentRunConfig::new("agent", "session"), "q")
            .await
            .unwrap();
        let messages = vec![BaseMessage::user("q"), BaseMessage::assistant("a")];
        active
            .save_session(messages.clone(), Some(json!({"iteration": 1})))
            .await
            .unwrap();

        let binding = SessionBinding::new("agent", "session").unwrap();
        let record = active.store().load(&binding).await.unwrap().unwrap();
        let ending = record.ending.as_ref().unwrap();
        assert_eq!(ending.outcome, SESSION_ENDING_OUTCOME);
        assert_eq!(ending.execution_id, active.execution_id());
        assert_eq!(ending.request_id, active.identity().request_id);
        assert_eq!(record.messages, messages);
        assert_eq!(record.deepagent_state, Some(json!({"iteration": 1})));

        let listed = active.store().list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].binding, binding);
        assert!(listed[0].has_ending);
    }

    #[tokio::test]
    async fn persist_session_stores_a_caller_record_unchanged() {
        let (_dir, recorder) = recorder();
        let active = recorder
            .begin(&PersistentRunConfig::new("agent", "session"), "q")
            .await
            .unwrap();
        let binding = SessionBinding::new("agent", "session").unwrap();
        let record: PersistentSessionRecord = PersistentSessionRecord::new(&binding).unwrap();
        active.persist_session(&record).await.unwrap();
        let loaded = active.store().load(&binding).await.unwrap().unwrap();
        assert_eq!(loaded, record);
        // No ending, no checkpoint: the record is a valid draft.
        assert!(loaded.ending.is_none() && loaded.checkpoint.is_none());
    }

    #[tokio::test]
    async fn the_turn_counter_starts_at_zero_per_execution() {
        let (_dir, recorder) = recorder();
        let first = recorder
            .begin(&PersistentRunConfig::new("agent", "one"), "q")
            .await
            .unwrap();
        let second: ActiveExecution = recorder
            .begin(&PersistentRunConfig::new("agent", "two"), "q")
            .await
            .unwrap();
        assert_eq!(first.model_turns.load(Ordering::Relaxed), 0);
        assert_eq!(second.model_turns.load(Ordering::Relaxed), 0);
    }
}
