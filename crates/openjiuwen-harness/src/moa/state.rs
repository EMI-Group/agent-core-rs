//! Invocation-local consultation state, immutable diagnostic snapshots, consultation
//! outcomes and the bounded history.
//!
//! Ports `openjiuwen/harness/moa/state.py`. Python keeps one `MoAState` per outer agent
//! invocation in a `ContextVar`; Rust stores no process-global state, so the owner of the
//! consultation (the rail or the host) holds the [`MoaState`] and passes it explicitly.
//! The context-variable half of `accepts_result` (the work binding) is not ported; the
//! epoch and the closed flags are.
//!
//! [`MoaState`] holds the baseline conversation, so it is not `serde`; the diagnostics
//! ([`MoaSnapshot`]), the outcome ([`ConsultationOutcome`]) and the history
//! ([`ConsultationHistory`]) round-trip through JSON with the keys Python persists.
// The public names (`MoaState`, `MoaSnapshot`, `ConsultationOutcome`, ...) mirror the
// Python contract and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;
use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::BaseMessage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::config::DEFAULT_HISTORY_LIMIT;

/// The logical decision a set of proposals belongs to.
///
/// Python models this as a `(turn, iteration)` tuple; Rust keeps both halves named so a
/// persisted or logged key is self-describing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
pub struct TurnKey {
    /// Outer task-loop turn number.
    pub turn: u64,
    /// Iteration inside that turn.
    pub iteration: u64,
}

impl TurnKey {
    /// Builds the key of `iteration` inside `turn`.
    #[must_use]
    pub const fn new(turn: u64, iteration: u64) -> Self {
        Self { turn, iteration }
    }
}

/// One reference's original response, in configured order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProposalSource {
    /// One-based position of the reference model in the configuration.
    pub index: usize,
    /// The retained proposal text (already trimmed to the configured budgets).
    pub text: String,
}

impl ProposalSource {
    /// Builds a proposal for the reference at `index`.
    #[must_use]
    pub fn new(index: usize, text: impl Into<String>) -> Self {
        Self {
            index,
            text: text.into(),
        }
    }
}

/// One reference model that failed to produce a proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProposerFailure {
    /// One-based position of the reference model in the configuration.
    pub index: usize,
    /// Why the model produced no usable proposal.
    pub error: String,
}

impl ProposerFailure {
    /// Builds a failure record for the reference at `index`.
    #[must_use]
    pub fn new(index: usize, error: impl Into<String>) -> Self {
        Self {
            index,
            error: error.into(),
        }
    }
}

/// Request-local diagnostic of how one batch of advice reached the model request.
///
/// Never decides delivery eligibility; it only explains the injection decision.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct ProposalAttempt {
    /// Decision the advice was prepared for; `None` while the target is unresolved.
    pub key: Option<TurnKey>,
    /// Whether the advice actually reached a model request.
    pub injected: bool,
    /// Why the advice was or was not injected.
    pub reason: String,
    /// Number of advice characters retained for the request.
    pub kept_chars: usize,
    /// Whether the text came from a fallback rather than a fresh consultation.
    pub fallback: bool,
}

impl ProposalAttempt {
    /// Builds an attempt diagnostic that was not injected.
    #[must_use]
    pub fn new(key: Option<TurnKey>, reason: impl Into<String>) -> Self {
        Self {
            key,
            injected: false,
            reason: reason.into(),
            kept_chars: 0,
            fallback: false,
        }
    }
}

/// One batch of proposals produced by a single consultation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProposalBatch {
    /// The proposals that survived budget trimming, in configured order.
    pub sources: Vec<ProposalSource>,
    /// Decision the batch was prepared for.
    pub target: TurnKey,
    /// Injection diagnostics recorded for this batch.
    pub attempts: Vec<ProposalAttempt>,
    /// Caller-assigned identity of the batch.
    pub batch_id: u64,
    /// Reason the batch was produced with no usable advice, if any.
    pub disabled_reason: Option<String>,
    /// Decision that was active when the batch was prepared.
    pub origin: Option<TurnKey>,
}

impl ProposalBatch {
    /// Builds an empty attempt log for `sources` targeting `target`.
    ///
    /// `batch_id` is caller-assigned: Python mints ids from a module-level counter, while
    /// this port keeps the identity local to the owner of the state.
    #[must_use]
    pub fn new(batch_id: u64, sources: Vec<ProposalSource>, target: TurnKey) -> Self {
        Self {
            sources,
            target,
            attempts: Vec::new(),
            batch_id,
            disabled_reason: None,
            origin: None,
        }
    }

    /// Returns this batch attributed to the decision that was active when it was prepared.
    #[must_use]
    pub fn with_origin(mut self, origin: TurnKey) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Returns this batch carrying the reason it has no usable advice.
    #[must_use]
    pub fn with_disabled_reason(mut self, disabled_reason: impl Into<String>) -> Self {
        self.disabled_reason = Some(disabled_reason.into());
        self
    }

    /// Records one injection diagnostic.
    pub fn push_attempt(&mut self, attempt: ProposalAttempt) {
        self.attempts.push(attempt);
    }
}

/// Candidate storage owned by one logical user task.
///
/// `ready` holds advice waiting for the next decision boundary; `activate` promotes it once
/// that boundary is a *different* decision. `epoch` fences asynchronous work: a worker that
/// started before an [`ProposalStore::invalidate`] sees a stale epoch and must discard its
/// result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ProposalStore {
    /// Advice committed to the current decision.
    pub active: Option<ProposalBatch>,
    /// Advice waiting for the next decision boundary.
    pub ready: Option<ProposalBatch>,
    /// Fencing epoch, bumped by every invalidation.
    pub epoch: u64,
    /// Number of batches issued by this store.
    pub issued_ticket: u64,
    /// Highest ticket whose batch was accepted into `active` or `ready`.
    pub accepted_ticket: u64,
    /// Whether the owning task retired.
    pub closed: bool,
}

impl ProposalStore {
    /// Promotes ready advice only at a different logical decision.
    ///
    /// Does nothing when the store is closed, when no advice is ready, or when the ready
    /// advice was prepared for `key` itself (it is already at its decision).
    pub fn activate(&mut self, key: TurnKey) {
        if self.closed {
            return;
        }
        let promote = self
            .ready
            .as_ref()
            .is_some_and(|ready| ready.origin != Some(key));
        if promote {
            self.active = self.ready.take();
        }
    }

    /// Fences asynchronous work, optionally retaining committed advice.
    ///
    /// Every call bumps [`Self::epoch`], so a worker holding the previous epoch must
    /// discard its result. `clear` drops both batches (used when ownership ends rather
    /// than when a new decision starts).
    pub fn invalidate(&mut self, reason: &str, clear: bool) {
        self.epoch = self.epoch.saturating_add(1);
        if clear {
            self.active = None;
            self.ready = None;
        }
        tracing::debug!(reason, clear, "MoA proposal store invalidated");
    }

    /// Retires the task without allowing late publication.
    pub fn close(&mut self, reason: &str) {
        self.closed = true;
        self.invalidate(reason, true);
    }
}

/// Ephemeral consultation and scoring state for one outer invocation.
///
/// Created by the owner of the consultation (Python uses a `ContextVar`), dropped with the
/// invocation, and never shared between invocations.
#[derive(Debug)]
pub struct MoaState {
    /// The problem statement handed to the references.
    pub problem: String,
    /// Conversation snapshot the proposals were prepared against.
    pub baseline: Vec<BaseMessage>,
    /// Candidate storage.
    pub proposals: ProposalStore,
    /// Decision the acting agent is currently working on.
    pub active_key: Option<TurnKey>,
    /// Completed outer task-loop turns.
    pub completed_turns: usize,
    /// Progress steps observed by the verifier.
    pub steps: Vec<String>,
    /// Score of the previous verification checkpoint.
    pub previous_score: Option<f32>,
    /// Number of consultation batches spent.
    pub reference_calls: usize,
    /// Number of verifier checkpoints spent.
    pub verifier_calls: usize,
    /// Reason the last consultation was refused, if it was.
    pub consultation_rejection: Option<String>,
    /// Whether the invocation retired.
    pub closed: bool,
    /// Number of router checkpoints spent in [`super::config::MoaMode::JevRouted`].
    pub jev_calls: usize,
    /// Last router decision, when the router answered.
    pub last_jev_decision: Option<bool>,
    /// Reason reported with the last router decision.
    pub last_jev_reason: Option<String>,
    /// Identity of this invocation, unique across the process lifetime.
    pub invoke_id: String,
}

impl MoaState {
    /// Starts a fresh invocation state for `problem`.
    #[must_use]
    pub fn new(problem: impl Into<String>) -> Self {
        Self {
            problem: problem.into(),
            baseline: Vec::new(),
            proposals: ProposalStore::default(),
            active_key: None,
            completed_turns: 0,
            steps: Vec::new(),
            previous_score: None,
            reference_calls: 0,
            verifier_calls: 0,
            consultation_rejection: None,
            closed: false,
            jev_calls: 0,
            last_jev_decision: None,
            last_jev_reason: None,
            invoke_id: Uuid::new_v4().simple().to_string(),
        }
    }

    /// Whether work that captured `epoch` may still publish its result.
    ///
    /// Fails after the invocation closed, after the proposal store closed, and after any
    /// invalidation bumped the store epoch.
    #[must_use]
    pub fn accepts_result(&self, epoch: u64) -> bool {
        !self.closed && !self.proposals.closed && self.proposals.epoch == epoch
    }

    /// Returns the read-only routing diagnostics of this invocation.
    ///
    /// The snapshot deliberately carries no reference text: only routing metadata leaves
    /// the state.
    #[must_use]
    pub fn snapshot(&self) -> MoaSnapshot {
        let pending = self
            .proposals
            .ready
            .as_ref()
            .or(self.proposals.active.as_ref());
        let latest_attempt = self
            .proposals
            .active
            .as_ref()
            .and_then(|batch| batch.attempts.last())
            .cloned();
        MoaSnapshot {
            active_key: self.active_key,
            completed_turns: self.completed_turns,
            reference_calls: self.reference_calls,
            verifier_calls: self.verifier_calls,
            previous_score: self.previous_score,
            pending_target: pending.map(|batch| batch.target),
            latest_attempt,
            jev_calls: self.jev_calls,
            last_jev_decision: self.last_jev_decision,
            last_jev_reason: self.last_jev_reason.clone(),
        }
    }
}

/// Read-only routing metadata of one invocation; excludes reference text and mutable state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MoaSnapshot {
    /// Decision the acting agent is currently working on.
    pub active_key: Option<TurnKey>,
    /// Completed outer task-loop turns.
    pub completed_turns: usize,
    /// Number of consultation batches spent.
    pub reference_calls: usize,
    /// Number of verifier checkpoints spent.
    pub verifier_calls: usize,
    /// Score of the previous verification checkpoint.
    pub previous_score: Option<f32>,
    /// Decision the pending (ready or active) advice belongs to.
    pub pending_target: Option<TurnKey>,
    /// Latest injection diagnostic of the active batch.
    pub latest_attempt: Option<ProposalAttempt>,
    /// Number of router checkpoints spent.
    pub jev_calls: usize,
    /// Last router decision, when the router answered.
    pub last_jev_decision: Option<bool>,
    /// Reason reported with the last router decision.
    pub last_jev_reason: Option<String>,
}

/// One proposal ranked by the verifier.
///
/// Shared by [`ConsultationOutcome`] and the verifier module, which produces the ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RankedProposal {
    /// [`ProposalSource::index`] this ranking refers to.
    pub index: usize,
    /// Score in `0.0..=1.0`; higher is better.
    pub score: f32,
    /// Why the proposal received this score.
    pub reasons: Vec<String>,
}

/// Everything one consultation produced, kept for diagnostics and history.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ConsultationOutcome {
    /// Proposals that survived budget trimming, in configured order.
    pub proposals: Vec<ProposalSource>,
    /// Reference models that answered with a usable proposal.
    pub success_count: usize,
    /// Reference models that failed.
    pub failed_count: usize,
    /// One record per failed reference model, in configured order.
    pub failures: Vec<ProposerFailure>,
    /// Verifier ranking of [`Self::proposals`]; empty when no verifier ran.
    pub ranking: Vec<RankedProposal>,
    /// Aggregated answer, when an aggregator ran.
    pub aggregate: Option<String>,
    /// Progress score of the verification checkpoint, when one ran.
    pub verifier_score: Option<f32>,
}

impl ConsultationOutcome {
    /// Whether the consultation produced no usable proposal.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.proposals.is_empty()
    }

    /// The proposal the ranking prefers, or the first proposal without a ranking.
    #[must_use]
    pub fn best(&self) -> Option<&ProposalSource> {
        let top = self
            .ranking
            .iter()
            .max_by(|left, right| left.score.total_cmp(&right.score));
        if let Some(top) = top {
            if let Some(source) = self
                .proposals
                .iter()
                .find(|source| source.index == top.index)
            {
                return Some(source);
            }
        }
        self.proposals.first()
    }
}

/// One problem and the outcome of consulting the references about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConsultationRecord {
    /// The problem statement the references answered.
    pub problem: String,
    /// What the consultation produced.
    pub outcome: ConsultationOutcome,
}

impl ConsultationRecord {
    /// Builds a record pairing `problem` with `outcome`.
    #[must_use]
    pub fn new(problem: impl Into<String>, outcome: ConsultationOutcome) -> Self {
        Self {
            problem: problem.into(),
            outcome,
        }
    }
}

/// Bounded, oldest-first history of consultations.
///
/// The retained count never exceeds [`Self::limit`], and pushing past it evicts the oldest
/// record through the front of the deque, so a full history accepts a new record in constant
/// time. `limit == 0` disables the history: nothing is retained and [`Self::push`] discards
/// the record instead of trimming a buffer it would never report. Persisting is an explicit
/// [`Self::save`] call against a caller-supplied path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ConsultationHistory {
    /// Maximum number of retained records; `0` disables the history.
    pub limit: usize,
    /// Retained records, oldest first.
    pub records: VecDeque<ConsultationRecord>,
}

impl Default for ConsultationHistory {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_LIMIT)
    }
}

impl ConsultationHistory {
    /// Builds an empty history retaining at most `limit` records.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            records: VecDeque::new(),
        }
    }

    /// Appends `record`, evicting the oldest records beyond [`Self::limit`].
    ///
    /// Eviction pops the oldest record, so a full history always stays at [`Self::len`] equal
    /// to its limit. A disabled history (`limit == 0`) retains nothing: the record is dropped
    /// and anything already retained is emptied, so the state never exceeds the declared limit.
    pub fn push(&mut self, record: ConsultationRecord) {
        if self.limit == 0 {
            self.records.clear();
            return;
        }
        while self.records.len() >= self.limit {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Replaces the retained records with `records`, keeping the newest [`Self::limit`].
    ///
    /// Used to adopt a persisted history: the receiver's limit wins, so a smaller or disabled
    /// limit is honoured instead of inheriting the limit stored in the file.
    pub fn replace(&mut self, records: impl IntoIterator<Item = ConsultationRecord>) {
        self.records.clear();
        for record in records {
            self.push(record);
        }
    }

    /// The retained records, oldest first.
    #[must_use]
    pub fn records(&self) -> &VecDeque<ConsultationRecord> {
        &self.records
    }

    /// Number of retained records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no record is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Drops every retained record, keeping [`Self::limit`].
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Writes the history to `path` as pretty JSON, replacing any existing file.
    ///
    /// # Errors
    ///
    /// Returns an error when the parent directory is missing or unwritable, or when the
    /// history cannot be serialized.
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let rendered = serde_json::to_vec_pretty(self)?;
        tokio::fs::write(path, rendered).await?;
        Ok(())
    }

    /// Reads a history written by [`Self::save`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `path` does not exist, an [`Error::External`] I/O
    /// error when it cannot be read, and [`Error::Validation`] when the file is not a
    /// consultation history.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = match tokio::fs::read(path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::config(format!(
                    "consultation history not found at {}",
                    path.display()
                )));
            }
            Err(error) => return Err(Error::external(error)),
        };
        serde_json::from_slice(&bytes).map_err(|error| {
            Error::validation(format!(
                "invalid consultation history at {}: {error}",
                path.display()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn source(index: usize, text: &str) -> ProposalSource {
        ProposalSource::new(index, text)
    }

    fn batch(batch_id: u64, origin: TurnKey) -> ProposalBatch {
        ProposalBatch::new(batch_id, vec![source(1, "a")], TurnKey::new(1, 0)).with_origin(origin)
    }

    #[test]
    fn turn_keys_round_trip() {
        let key = TurnKey::new(2, 3);
        assert_eq!(key.turn, 2);
        assert_eq!(key.iteration, 3);
        let json = serde_json::to_string(&key).unwrap();
        assert_eq!(json, "{\"turn\":2,\"iteration\":3}");
        assert_eq!(serde_json::from_str::<TurnKey>(&json).unwrap(), key);
    }

    #[test]
    fn store_promotes_ready_advice_only_at_a_new_decision() {
        let origin = TurnKey::new(1, 0);
        let mut store = ProposalStore {
            ready: Some(batch(1, origin)),
            ..ProposalStore::default()
        };

        store.activate(origin);
        assert!(store.active.is_none());
        assert!(store.ready.is_some());

        store.activate(TurnKey::new(1, 1));
        assert!(store.active.is_some());
        assert!(store.ready.is_none());

        store.ready = Some(batch(2, TurnKey::new(1, 1)));
        store.close("done");
        store.activate(TurnKey::new(1, 2));
        assert!(store.closed);
        assert!(store.active.is_none());
        assert!(store.ready.is_none());
    }

    #[test]
    fn store_invalidate_fences_and_optionally_keeps_advice() {
        let mut store = ProposalStore {
            active: Some(batch(1, TurnKey::new(0, 0))),
            ready: Some(batch(2, TurnKey::new(0, 0))),
            ..ProposalStore::default()
        };

        store.invalidate("retain", false);
        assert_eq!(store.epoch, 1);
        assert!(store.active.is_some());
        assert!(store.ready.is_some());

        store.invalidate("drop", true);
        assert_eq!(store.epoch, 2);
        assert!(store.active.is_none());
        assert!(store.ready.is_none());
    }

    #[test]
    fn state_snapshot_reports_routing_metadata() {
        let mut state = MoaState::new("fix the build");
        assert_eq!(state.problem, "fix the build");
        assert_ne!(state.invoke_id, "");
        assert!(state.accepts_result(state.proposals.epoch));

        let target = TurnKey::new(2, 1);
        let mut active = ProposalBatch::new(
            7,
            vec![source(1, "reference text that must not leak")],
            target,
        )
        .with_origin(target);
        let mut attempt = ProposalAttempt::new(Some(target), "budget available");
        attempt.injected = true;
        attempt.kept_chars = 42;
        active.push_attempt(attempt);
        state.proposals.active = Some(active);
        state.active_key = Some(target);
        state.completed_turns = 2;
        state.reference_calls = 1;
        state.verifier_calls = 2;
        state.previous_score = Some(0.5);
        state.jev_calls = 3;
        state.last_jev_decision = Some(true);
        state.last_jev_reason = Some("stalled".to_owned());

        let snapshot = state.snapshot();
        assert_eq!(snapshot.active_key, Some(target));
        assert_eq!(snapshot.pending_target, Some(target));
        assert_eq!(snapshot.completed_turns, 2);
        assert_eq!(snapshot.reference_calls, 1);
        assert_eq!(snapshot.verifier_calls, 2);
        assert_eq!(snapshot.previous_score, Some(0.5));
        assert_eq!(snapshot.jev_calls, 3);
        assert_eq!(snapshot.last_jev_decision, Some(true));
        assert_eq!(snapshot.last_jev_reason.as_deref(), Some("stalled"));
        let latest = snapshot.latest_attempt.as_ref().unwrap();
        assert!(latest.injected);
        assert_eq!(latest.kept_chars, 42);
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(
            !json.contains("reference text that must not leak"),
            "snapshot must not carry reference text"
        );
    }

    #[test]
    fn state_accepts_result_only_before_invalidation_and_close() {
        let mut state = MoaState::new("problem");
        let epoch = state.proposals.epoch;
        assert!(state.accepts_result(epoch));

        state.proposals.invalidate("stale", false);
        assert!(!state.accepts_result(epoch));
        assert!(state.accepts_result(state.proposals.epoch));

        let current = state.proposals.epoch;
        state.closed = true;
        assert!(!state.accepts_result(current));
    }

    #[test]
    fn outcome_reports_emptiness_and_best_proposal() {
        let mut outcome = ConsultationOutcome::default();
        assert!(outcome.is_empty());
        assert!(outcome.best().is_none());

        outcome.proposals = vec![source(1, "short"), source(2, "longer proposal")];
        outcome.success_count = 2;
        assert!(!outcome.is_empty());
        assert_eq!(outcome.best().unwrap().text, "short");

        outcome.ranking = vec![
            RankedProposal {
                index: 1,
                score: 0.2,
                reasons: vec!["thin".to_owned()],
            },
            RankedProposal {
                index: 2,
                score: 0.9,
                reasons: vec!["complete".to_owned()],
            },
        ];
        assert_eq!(outcome.best().unwrap().text, "longer proposal");

        outcome.ranking = vec![RankedProposal {
            index: 99,
            score: 1.0,
            reasons: Vec::new(),
        }];
        assert_eq!(outcome.best().unwrap().text, "short");
    }

    #[test]
    fn history_evicts_the_oldest_records() {
        let mut history = ConsultationHistory::new(2);
        assert!(history.is_empty());
        for index in 0..3 {
            history.push(ConsultationRecord::new(
                format!("problem {index}"),
                ConsultationOutcome::default(),
            ));
        }
        assert_eq!(history.len(), 2);
        assert_eq!(history.records()[0].problem, "problem 1");
        assert_eq!(history.records()[1].problem, "problem 2");

        history.clear();
        assert!(history.is_empty());

        let mut empty = ConsultationHistory::new(0);
        empty.push(ConsultationRecord::new("x", ConsultationOutcome::default()));
        assert!(empty.is_empty());
    }

    #[test]
    fn a_full_history_keeps_only_the_newest_records() {
        let mut history = ConsultationHistory::new(3);
        for index in 0..1_000 {
            history.push(ConsultationRecord::new(
                format!("problem {index}"),
                ConsultationOutcome::default(),
            ));
            assert!(history.len() <= 3);
        }
        assert_eq!(history.len(), 3);
        let problems: Vec<&str> = history
            .records()
            .iter()
            .map(|record| record.problem.as_str())
            .collect();
        assert_eq!(problems, ["problem 997", "problem 998", "problem 999"]);
    }

    #[test]
    fn a_zero_limit_history_is_disabled() {
        let mut disabled = ConsultationHistory::new(0);
        assert_eq!(disabled.limit, 0);
        disabled.push(ConsultationRecord::new("a", ConsultationOutcome::default()));
        disabled.push(ConsultationRecord::new("b", ConsultationOutcome::default()));
        assert_eq!(disabled.len(), 0);
        assert!(disabled.is_empty());
        assert!(disabled.records().is_empty());

        // Lowering the limit on a populated history drops what it had retained instead of
        // leaving records behind a limit that says nothing is kept.
        let mut lowered = ConsultationHistory::new(2);
        for problem in ["a", "b"] {
            lowered.push(ConsultationRecord::new(
                problem,
                ConsultationOutcome::default(),
            ));
        }
        assert_eq!(lowered.len(), 2);
        lowered.limit = 0;
        lowered.push(ConsultationRecord::new("c", ConsultationOutcome::default()));
        assert!(lowered.is_empty());
    }

    #[test]
    fn replace_adopts_the_newest_records_within_the_local_limit() {
        let mut history = ConsultationHistory::new(2);
        history.push(ConsultationRecord::new(
            "stale",
            ConsultationOutcome::default(),
        ));

        history.replace([
            ConsultationRecord::new("first", ConsultationOutcome::default()),
            ConsultationRecord::new("second", ConsultationOutcome::default()),
            ConsultationRecord::new("third", ConsultationOutcome::default()),
        ]);
        let problems: Vec<&str> = history
            .records()
            .iter()
            .map(|record| record.problem.as_str())
            .collect();
        assert_eq!(problems, ["second", "third"]);

        // The receiving instance's limit wins, so a disabled history adopts nothing.
        let mut disabled = ConsultationHistory::new(0);
        disabled.replace([ConsultationRecord::new(
            "ignored",
            ConsultationOutcome::default(),
        )]);
        assert!(disabled.is_empty());
    }

    #[test]
    fn history_json_shape_is_a_limit_plus_a_record_array() {
        let mut history = ConsultationHistory::new(2);
        history.push(ConsultationRecord::new(
            "problem",
            ConsultationOutcome::default(),
        ));

        let value = serde_json::to_value(&history).unwrap();
        let object = value.as_object().unwrap();
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ["limit", "records"]);
        assert_eq!(object["limit"], serde_json::json!(2));
        assert_eq!(object["records"].as_array().unwrap().len(), 1);
        assert_eq!(
            serde_json::from_value::<ConsultationHistory>(value).unwrap(),
            history
        );
    }

    #[tokio::test]
    async fn history_round_trips_through_json() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("history.json");

        let mut history = ConsultationHistory::new(4);
        let mut outcome = ConsultationOutcome {
            proposals: vec![source(1, "proposal")],
            success_count: 1,
            failed_count: 1,
            failures: vec![ProposerFailure::new(2, "timeout")],
            aggregate: Some("answer".to_owned()),
            verifier_score: Some(0.75),
            ..ConsultationOutcome::default()
        };
        outcome.ranking = vec![RankedProposal {
            index: 1,
            score: 0.75,
            reasons: vec!["covers the task".to_owned()],
        }];
        history.push(ConsultationRecord::new("problem", outcome));

        history.save(&path).await.unwrap();
        let loaded = ConsultationHistory::load(&path).await.unwrap();
        assert_eq!(loaded, history);
    }

    #[tokio::test]
    async fn history_load_reports_missing_and_malformed_files() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("absent.json");
        let error = ConsultationHistory::load(&missing).await.unwrap_err();
        assert!(matches!(error, Error::Config(_)));
        assert!(error.to_string().contains("consultation history not found"));

        let malformed = temp.path().join("broken.json");
        tokio::fs::write(&malformed, b"{oops").await.unwrap();
        let error = ConsultationHistory::load(&malformed).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("invalid consultation history"));
    }

    #[test]
    fn batches_keep_origin_attempts_and_disable_reason() {
        let target = TurnKey::new(3, 1);
        let mut batch = ProposalBatch::new(5, vec![source(1, "a")], target)
            .with_origin(TurnKey::new(3, 0))
            .with_disabled_reason("no budget");
        batch.push_attempt(ProposalAttempt::new(Some(target), "budget exhausted"));
        assert_eq!(batch.batch_id, 5);
        assert_eq!(batch.target, target);
        assert_eq!(batch.origin, Some(TurnKey::new(3, 0)));
        assert_eq!(batch.disabled_reason.as_deref(), Some("no budget"));
        assert_eq!(batch.attempts.len(), 1);
        assert!(!batch.attempts[0].injected);
    }
}
