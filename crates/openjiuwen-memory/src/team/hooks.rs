//! `TeamMemoryHooks`: explicit, caller-driven team memory lifecycle hooks (member join,
//! session start, turn sync, session end, member leave).
//!
//! The Python `openjiuwen.agent_teams.memory.manager.TeamMemoryManager` drives team memory
//! from inside the agent runtime: it binds a session id, registers tools, injects the team
//! summary into a prompt, and extracts memories at `kernel.pause()`/`stop()`.
//! This module keeps those lifecycle *events* and drops the machinery: a [`TeamMemoryHooks`]
//! is a plain value the caller invokes at each point, so nothing is registered in a
//! process-global registry, no timer or background task is spawned, and there is no bus to
//! subscribe to.
//! Invitation is therefore explicit — a hook runs when, and only when, its caller calls it.
//! One instance belongs to exactly one team: the team id is supplied at construction
//! ([`TeamMemoryHooks::new`]) and never changes, so build one hooks object per team.
//!
//! # Hooks
//!
//! | Hook | Call it when | Store effect |
//! | --- | --- | --- |
//! | [`TeamMemoryHooks::on_session_start`] | a coordination session begins | binds the session id and discards the previous session's buffered turns |
//! | [`TeamMemoryHooks::on_member_join`] | a member joins the team | one `MemberProfile` record |
//! | [`TeamMemoryHooks::on_turn`] (alias [`TeamMemoryHooks::sync_turn`]) | after a team turn | none: the turn is buffered for the session-end flush |
//! | [`TeamMemoryHooks::on_session_end`] | the session pauses or stops | extracts and writes the buffered turns as one batch |
//! | [`TeamMemoryHooks::on_member_leave`] | a member leaves the team | one `HandoffNote` record |
//!
//! `on_turn` buffers even when no session was started; the transcript then carries no
//! session id, and the flushed records are session-less rather than rejected.
//! The turn buffer is bounded by [`TeamMemoryHookConfig::max_pending_turns`]: the oldest
//! turns are dropped (with a `tracing::warn!`) so a long session cannot grow without limit.
//!
//! # `on_session_end` ordering contract
//!
//! The flush is ordered, and every step either completes or leaves the buffer intact:
//!
//! 1. check cancellation, validate the team id, then snapshot (clone) the buffered turns,
//!    task notes, and session id under the state lock, releasing the lock before any
//!    `.await`;
//! 2. return an empty [`TeamMemoryFlush`] without clearing anything when the buffer is
//!    empty, when [`TeamMemoryHookConfig::enabled`] is `false`, or when
//!    [`TeamMemoryHookConfig::flush_on_session_end`] is `false` (a disabled flush keeps the
//!    buffer so a later manual flush can still use it);
//! 3. run the extractor over a [`TeamTranscript`]; a failing extractor returns its error
//!    unchanged and the buffer is kept, so the caller can retry;
//! 4. validate every returned [`NewTeamMemory`] *before* any write; one invalid record fails
//!    the whole flush with [`Error::Validation`] naming the field, nothing is written, and
//!    the buffer is kept;
//! 5. overwrite each id with a deterministic id derived from the team, the session, the
//!    kind, and the content, and stamp the session id when the extractor left it out;
//! 6. write the batch with a single `TeamMemory::remember_many` call, then clear the buffer
//!    and return the stored records with the turn count and the extractor's degradation
//!    reason.
//!
//! Retrying a flush is safe and idempotent: the ids are a pure function of the extracted
//! content, so a retry after a partial write upserts the same rows instead of duplicating
//! them, and a retry after a failed write starts from the intact buffer.
//! Two hooks objects for the same team and session converge on the same rows for the same
//! turns.
//!
//! # Cancellation
//!
//! Every hook checks the [`CallOptions`] token before it does anything, so a triggered token
//! returns [`Error::Cancelled`] with the state untouched: `on_turn` does not buffer,
//! `on_member_join`/`on_member_leave` neither record the membership nor write, and
//! `on_session_end` keeps its buffer.
//! The same token is handed to the extractor and to the batch write, which check it around
//! their own awaits.
//!
//! # Errors
//!
//! Failures are never swallowed: every degraded path is either a typed error returned to the
//! caller or an explicit `degraded` reason on [`TeamMemoryFlush`] plus a `tracing::warn!`.
//! Invalid ids and an invalid [`TeamMemoryHookConfig`] are [`Error::Validation`], and a write
//! error leaves the buffer intact so the caller can retry.
//!
//! # Observability
//!
//! The hooks emit through `TeamMemory::emit`, so an observer attached to the facade sees the
//! hooks' writes as the facade's own `Write` events plus one `Extract` event per flush
//! carrying the stored record count — or `TeamMemoryOutcome::Failed` with the error text.
//!
//! # Python mapping
//!
//! | Python `TeamMemoryManager` | Rust |
//! | --- | --- |
//! | `bind_session_id(session_id)` | [`TeamMemoryHooks::on_session_start`] |
//! | `extract_after_round()` | [`TeamMemoryHooks::on_session_end`] |
//! | construction (`init_toolkit`, `register_tools`) | [`TeamMemoryHooks::on_member_join`] |
//! | `close()` | [`TeamMemoryHooks::on_member_leave`] |
//! | `load_and_inject()` / `system_prompt_block` | not here: the crate exposes prompt rendering, the caller injects |
//!
//! Two differences are deliberate.
//! First, Python has no per-turn hook and no team-database-independent input: its extractor
//! reads the team db at pause. This port buffers turns through `on_turn`/`sync_turn`, so
//! extraction works for a caller that keeps no team database at all.
//! Second, Python logs and swallows every hook error (`logger.warning`), while these hooks
//! return typed errors and guarantee that the store is never left half-written.
//!
//! The intended caller is the `openjiuwen-agent-teams` runtime, which owns the session and
//! membership events; `openjiuwen-memory` itself never starts a hook on its own.
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::record::CallOptions;
use crate::team::extractor::{
    RuleTeamExtractor, TeamExtractConfig, TeamMemoryExtractor, TeamTranscript, TeamTurn,
};
use crate::team::model::{
    NewTeamMemory, TeamMemoryKind, TeamMemoryRecord, TeamVisibility, MAX_SCOPE_ID_CHARS,
};
use crate::team::observe::{TeamMemoryEvent, TeamMemoryOperation};
use crate::team::store::TeamMemory;

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// The `source` stamped on every record a hook writes itself.
pub const HOOK_SOURCE: &str = "team-memory-hooks";
/// Prefix of every deterministic id this module derives.
const HOOK_ID_PREFIX: &str = "hook-";
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Byte hashed between the parts of a derived id, so part boundaries matter.
const PART_SEPARATOR: u8 = 0x1f;
/// Default importance of a member-join profile record.
pub const DEFAULT_MEMBER_PROFILE_IMPORTANCE: f32 = 0.4;
/// Default importance of a member-leave handoff record.
pub const DEFAULT_HANDOFF_IMPORTANCE: f32 = 0.6;
/// Default number of turns buffered before the oldest is dropped.
pub const DEFAULT_MAX_PENDING_TURNS: usize = 64;

/// The switches and importances of one [`TeamMemoryHooks`] instance.
///
/// The defaults enable every hook, which is what a normal team runtime wants; a caller that
/// only wants session-end extraction switches the presence records off.
/// [`TeamMemoryHookConfig::validate`] is applied by every hook that writes, so an invalid
/// importance or a zero turn bound is reported instead of silently degrading.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryHookConfig {
    /// Master switch: when `false`, no hook writes anything (turns are still buffered).
    pub enabled: bool,
    /// Write a `MemberProfile` record on [`TeamMemoryHooks::on_member_join`].
    pub member_profile_on_join: bool,
    /// Write a `HandoffNote` record on [`TeamMemoryHooks::on_member_leave`].
    pub handoff_on_leave: bool,
    /// Extract and write the buffered turns on [`TeamMemoryHooks::on_session_end`].
    pub flush_on_session_end: bool,
    /// Importance of the member-join profile record, finite inside `0.0..=1.0`.
    pub member_profile_importance: f32,
    /// Importance of the member-leave handoff record, finite inside `0.0..=1.0`.
    pub handoff_importance: f32,
    /// Maximum buffered turns; the oldest are dropped past this bound.
    pub max_pending_turns: usize,
}

impl Default for TeamMemoryHookConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            member_profile_on_join: true,
            handoff_on_leave: true,
            flush_on_session_end: true,
            member_profile_importance: DEFAULT_MEMBER_PROFILE_IMPORTANCE,
            handoff_importance: DEFAULT_HANDOFF_IMPORTANCE,
            max_pending_turns: DEFAULT_MAX_PENDING_TURNS,
        }
    }
}

impl TeamMemoryHookConfig {
    /// Validates the two importances and the turn bound.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when an importance is not finite or outside `0.0..=1.0`, or
    /// when `max_pending_turns` is zero.
    pub fn validate(&self) -> Result<()> {
        validate_importance("member_profile_importance", self.member_profile_importance)?;
        validate_importance("handoff_importance", self.handoff_importance)?;
        if self.max_pending_turns == 0 {
            return Err(Error::validation(
                "team memory hook `max_pending_turns` must be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }
}

/// What one [`TeamMemoryHooks::on_session_end`] call did.
///
/// `records` are the stored rows (already upserted, so a retry returns the same ids),
/// `degraded` is the extractor's degradation reason when it fell back to rules, and `turns`
/// is how many buffered turns the pass consumed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryFlush {
    /// The records this flush stored, in extraction order.
    pub records: Vec<TeamMemoryRecord>,
    /// Why a degraded extraction path produced the records, when one did.
    pub degraded: Option<String>,
    /// How many buffered turns the pass consumed.
    pub turns: usize,
}

impl TeamMemoryFlush {
    /// An empty flush: nothing to extract, or flushing is switched off.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns `true` when no record was stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the number of stored records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }
}

/// The mutable lifecycle state of one [`TeamMemoryHooks`] instance.
#[derive(Debug, Default)]
struct HookState {
    /// The session installed by [`TeamMemoryHooks::on_session_start`].
    session_id: Option<String>,
    /// The buffered turns, oldest first.
    turns: Vec<TeamTurn>,
    /// The buffered task notes (the transcript's `task_notes`).
    task_notes: Vec<String>,
    /// The members that joined and have not left, in join order.
    members: Vec<String>,
}

/// Explicit team memory lifecycle hooks for one team.
///
/// The hooks own all of their state (the session id, the turn buffer, and the member list)
/// behind one `parking_lot` mutex, so they need no global registry, no task, and no timer.
/// The lock is never held across an `.await`: each hook clones what it needs, releases the
/// guard, and only then writes.
/// Callers are expected to drive the hooks in lifecycle order for one session at a time; a
/// turn buffered concurrently with an in-flight flush survives only if it lands after the
/// flush clears the buffer, so serializing the hooks is the caller's responsibility.
pub struct TeamMemoryHooks {
    /// The team this instance is bound to.
    team_id: String,
    /// The facade the hooks read and write.
    memory: Arc<TeamMemory>,
    /// The extractor the session-end flush runs.
    extractor: Arc<dyn TeamMemoryExtractor>,
    /// The switches and importances in use.
    config: TeamMemoryHookConfig,
    /// The session id, turn buffer, and member list.
    state: Mutex<HookState>,
}

impl fmt::Debug for TeamMemoryHooks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TeamMemoryHooks")
            .field("team_id", &self.team_id)
            .field("extractor", &self.extractor.name())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl TeamMemoryHooks {
    /// Builds the hooks of `team_id` over `memory`, extracting with `extractor`.
    ///
    /// The team id is validated by every hook (never here), so an unusable id surfaces as
    /// [`Error::Validation`] from the first hook call instead of a panic at construction.
    #[must_use]
    pub fn new(
        team_id: impl Into<String>,
        memory: Arc<TeamMemory>,
        extractor: Arc<dyn TeamMemoryExtractor>,
        config: TeamMemoryHookConfig,
    ) -> Self {
        Self {
            team_id: team_id.into(),
            memory,
            extractor,
            config,
            state: Mutex::new(HookState::default()),
        }
    }

    /// Builds the hooks of `team_id` over `memory` with the deterministic rule extractor.
    #[must_use]
    pub fn with_rule_extractor(
        team_id: impl Into<String>,
        memory: Arc<TeamMemory>,
        config: TeamMemoryHookConfig,
    ) -> Self {
        Self::new(
            team_id,
            memory,
            Arc::new(RuleTeamExtractor::new(TeamExtractConfig::default())),
            config,
        )
    }

    /// The team memory facade these hooks read and write.
    #[must_use]
    pub fn memory(&self) -> &Arc<TeamMemory> {
        &self.memory
    }

    /// The extractor's name (`"rule"`, `"llm"`, or a caller's own).
    #[must_use]
    pub fn extractor_name(&self) -> &str {
        self.extractor.name()
    }

    /// The switches and importances in use.
    #[must_use]
    pub fn config(&self) -> &TeamMemoryHookConfig {
        &self.config
    }

    /// The team these hooks are bound to.
    #[must_use]
    pub fn team_id(&self) -> &str {
        &self.team_id
    }

    /// The bound session id, when [`Self::on_session_start`] has run.
    #[must_use]
    pub fn session_id(&self) -> Option<String> {
        self.state.lock().session_id.clone()
    }

    /// How many turns are buffered for the next flush.
    #[must_use]
    pub fn pending_turns(&self) -> usize {
        self.state.lock().turns.len()
    }

    /// The members that joined and have not left, in join order.
    #[must_use]
    pub fn members(&self) -> Vec<String> {
        self.state.lock().members.clone()
    }

    /// Binds `session_id` and discards the previous session's buffered turns.
    ///
    /// Idempotent for the same id: re-binding the current session keeps the buffer.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] when `call`'s token was triggered, and [`Error::Validation`]
    /// when the team or session id is unusable (empty, NUL, a path separator, a control
    /// character, or longer than 128 characters).
    #[allow(clippy::unused_async_trait_impl)] // async keeps the hook uniform with the others
    pub async fn on_session_start(&self, session_id: &str, call: &CallOptions) -> Result<()> {
        call.check()?;
        self.validate_team()?;
        validate_hook_id("session_id", session_id)?;
        {
            let mut state = self.state.lock();
            if state.session_id.as_deref() != Some(session_id) {
                let dropped = state.turns.len() + state.task_notes.len();
                state.turns.clear();
                state.task_notes.clear();
                if dropped > 0 {
                    warn!(
                        target: LOG_TARGET,
                        team_id = %self.team_id,
                        dropped,
                        "discarding the buffered turns of the previous team session"
                    );
                }
            }
            state.session_id = Some(session_id.to_owned());
        }
        debug!(
            target: LOG_TARGET,
            team_id = %self.team_id,
            session_id = %session_id,
            "team memory session started"
        );
        Ok(())
    }

    /// Records that `member_id` joined, writing one `MemberProfile` record when enabled.
    ///
    /// A join never touches the turn buffer, and a repeated join of the same member upserts
    /// the same deterministic row instead of adding a second one.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] when `call`'s token was triggered, [`Error::Validation`] for an
    /// unusable team or member id or an invalid config, and the store's own error.
    pub async fn on_member_join(
        &self,
        member_id: &str,
        call: &CallOptions,
    ) -> Result<Option<TeamMemoryRecord>> {
        call.check()?;
        self.validate_team()?;
        validate_hook_id("member_id", member_id)?;
        let should_write = self.config.enabled && self.config.member_profile_on_join;
        if should_write {
            self.config.validate()?;
        }
        let session = {
            let mut state = self.state.lock();
            if !state.members.iter().any(|known| known == member_id) {
                state.members.push(member_id.to_owned());
            }
            state.session_id.clone()
        };
        if !should_write {
            debug!(
                target: LOG_TARGET,
                team_id = %self.team_id,
                member_id = %member_id,
                "team member joined without a profile record"
            );
            return Ok(None);
        }
        let mut input = NewTeamMemory::member_profile(
            self.team_id.as_str(),
            format!("Member {member_id} joined team {}.", self.team_id),
        )
        .with_id(self.presence_id(
            session.as_deref(),
            &TeamMemoryKind::MemberProfile,
            member_id,
        ))
        .with_member(member_id)
        .with_title(format!("Member {member_id} joined"))
        .with_importance(self.config.member_profile_importance)
        .with_source(HOOK_SOURCE)
        .with_visibility(TeamVisibility::Team)
        .with_tag("member")
        .with_tag("presence");
        if let Some(session_id) = session.as_deref() {
            input = input.with_session(session_id);
        }
        let stored = self.memory.remember(input, call).await?;
        Ok(Some(stored))
    }

    /// Buffers one team turn for the session-end flush, without writing anything.
    ///
    /// A turn is buffered even when no session was started; the transcript then has no
    /// session id. Past [`TeamMemoryHookConfig::max_pending_turns`] the oldest turns are
    /// dropped with a `tracing::warn!`.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] when `call`'s token was triggered (nothing is buffered), and
    /// [`Error::Validation`] for an unusable team id or an invalid turn.
    #[allow(clippy::unused_async_trait_impl)] // async keeps the hook uniform with the others
    pub async fn on_turn(&self, turn: TeamTurn, call: &CallOptions) -> Result<()> {
        call.check()?;
        self.validate_team()?;
        turn.validate()?;
        let limit = self.config.max_pending_turns.max(1);
        let mut dropped = 0;
        {
            let mut state = self.state.lock();
            state.turns.push(turn);
            if state.turns.len() > limit {
                dropped = state.turns.len() - limit;
                state.turns.drain(..dropped);
            }
        }
        if dropped > 0 {
            warn!(
                target: LOG_TARGET,
                team_id = %self.team_id,
                dropped,
                limit,
                "dropping the oldest buffered team turns"
            );
        }
        Ok(())
    }

    /// The documented alias of [`Self::on_turn`], spelled for a turn-synchronisation caller.
    ///
    /// # Errors
    ///
    /// The same errors as [`Self::on_turn`].
    pub async fn sync_turn(&self, turn: TeamTurn, call: &CallOptions) -> Result<()> {
        self.on_turn(turn, call).await
    }

    /// Extracts the buffered turns and writes them as one batch.
    ///
    /// See the module docs for the ordering contract: snapshot, extract, validate,
    /// deterministic ids, one batch write, then clear the buffer.
    /// A failure at any step returns the error with the buffer intact, so the call can be
    /// retried; the deterministic ids make a retry converge instead of duplicating rows.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] when `call`'s token was triggered (the buffer is kept),
    /// [`Error::Validation`] for an unusable team id, an invalid config, or an extractor
    /// record that fails validation (nothing is written), the extractor's own error, and the
    /// store's own error.
    pub async fn on_session_end(&self, call: &CallOptions) -> Result<TeamMemoryFlush> {
        call.check()?;
        self.validate_team()?;
        let (session, turns, task_notes) = {
            let state = self.state.lock();
            (
                state.session_id.clone(),
                state.turns.clone(),
                state.task_notes.clone(),
            )
        };
        if turns.is_empty() && task_notes.is_empty() {
            debug!(
                target: LOG_TARGET,
                team_id = %self.team_id,
                "team memory flush had nothing buffered"
            );
            return Ok(TeamMemoryFlush::empty());
        }
        if !self.config.enabled || !self.config.flush_on_session_end {
            debug!(
                target: LOG_TARGET,
                team_id = %self.team_id,
                "team memory flush is disabled; keeping the buffered turns"
            );
            return Ok(TeamMemoryFlush::empty());
        }
        self.config.validate()?;
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Extract, self.team_id.as_str());
        match self.flush(session, turns, task_notes, call).await {
            Ok(flush) => {
                self.memory.emit(&event.with_count(flush.len()));
                Ok(flush)
            }
            Err(error) => {
                self.memory.emit(&event.failed(&error));
                Err(error)
            }
        }
    }

    /// The single extraction-and-write pass of [`Self::on_session_end`].
    async fn flush(
        &self,
        session: Option<String>,
        turns: Vec<TeamTurn>,
        task_notes: Vec<String>,
        call: &CallOptions,
    ) -> Result<TeamMemoryFlush> {
        let used_turns = turns.len();
        let mut transcript = TeamTranscript::new(self.team_id.as_str());
        if let Some(session_id) = session.as_deref() {
            transcript = transcript.with_session(session_id);
        }
        transcript.turns = turns;
        transcript.task_notes = task_notes;
        let extraction = self.extractor.extract(&transcript, call).await?;
        let mut prepared = Vec::with_capacity(extraction.records.len());
        for mut input in extraction.records {
            input.validate()?;
            input.id = Some(self.extract_id(session.as_deref(), &input.kind, &input.content));
            if input.session_id.is_none() {
                session.clone_into(&mut input.session_id);
            }
            prepared.push(input);
        }
        if let Some(reason) = extraction.degraded.as_deref() {
            warn!(
                target: LOG_TARGET,
                team_id = %self.team_id,
                reason,
                "team memory extraction ran on a degraded path"
            );
        }
        let records = if prepared.is_empty() {
            Vec::new()
        } else {
            self.memory.remember_many(prepared, call).await?
        };
        {
            let mut state = self.state.lock();
            state.turns.clear();
            state.task_notes.clear();
        }
        debug!(
            target: LOG_TARGET,
            team_id = %self.team_id,
            turns = used_turns,
            records = records.len(),
            degraded = extraction.degraded.is_some(),
            "flushed team memory"
        );
        Ok(TeamMemoryFlush {
            records,
            degraded: extraction.degraded,
            turns: used_turns,
        })
    }

    /// Records that `member_id` left, writing one `HandoffNote` record when enabled.
    ///
    /// The member is removed from [`Self::members`] whether or not the record is written.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] when `call`'s token was triggered, [`Error::Validation`] for an
    /// unusable team or member id or an invalid config, and the store's own error.
    pub async fn on_member_leave(
        &self,
        member_id: &str,
        call: &CallOptions,
    ) -> Result<Option<TeamMemoryRecord>> {
        call.check()?;
        self.validate_team()?;
        validate_hook_id("member_id", member_id)?;
        let should_write = self.config.enabled && self.config.handoff_on_leave;
        if should_write {
            self.config.validate()?;
        }
        let session = {
            let mut state = self.state.lock();
            state.members.retain(|known| known != member_id);
            state.session_id.clone()
        };
        if !should_write {
            debug!(
                target: LOG_TARGET,
                team_id = %self.team_id,
                member_id = %member_id,
                "team member left without a handoff record"
            );
            return Ok(None);
        }
        let mut input = NewTeamMemory::handoff_note(
            self.team_id.as_str(),
            format!("Member {member_id} left team {}.", self.team_id),
        )
        .with_id(self.presence_id(session.as_deref(), &TeamMemoryKind::HandoffNote, member_id))
        .with_member(member_id)
        .with_title(format!("Member {member_id} left"))
        .with_importance(self.config.handoff_importance)
        .with_source(HOOK_SOURCE)
        .with_visibility(TeamVisibility::Team)
        .with_tag("handoff")
        .with_tag("member");
        if let Some(session_id) = session.as_deref() {
            input = input.with_session(session_id);
        }
        let stored = self.memory.remember(input, call).await?;
        Ok(Some(stored))
    }

    /// Validates the team id this instance is bound to.
    fn validate_team(&self) -> Result<()> {
        validate_hook_id("team_id", &self.team_id)
    }

    /// The deterministic id of a presence record (join or leave).
    fn presence_id(&self, session: Option<&str>, kind: &TeamMemoryKind, member_id: &str) -> String {
        stable_id(&[
            self.team_id.as_str(),
            session.unwrap_or_default(),
            kind.as_str(),
            member_id,
        ])
    }

    /// The deterministic id of one extracted record.
    fn extract_id(&self, session: Option<&str>, kind: &TeamMemoryKind, content: &str) -> String {
        stable_id(&[
            self.team_id.as_str(),
            session.unwrap_or_default(),
            kind.as_str(),
            content.trim(),
        ])
    }
}

/// Validates an importance knob.
fn validate_importance(field: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::validation(format!(
            "team memory hook `{field}` must be finite and inside 0.0..=1.0 (got {value})"
        )));
    }
    Ok(())
}

/// Validates a hook id with the component rules of [`crate::team::model`].
///
/// Ids are storage-key segments, so they must be non-blank, NUL-free, free of `/` and `\`,
/// control-character-free, and at most [`MAX_SCOPE_ID_CHARS`] characters long.
fn validate_hook_id(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::validation(format!(
            "team memory hook `{field}` must not be empty"
        )));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!(
            "team memory hook `{field}` must not contain NUL (got {value:?})"
        )));
    }
    if value.contains('/') || value.contains('\\') {
        return Err(Error::validation(format!(
            "team memory hook `{field}` must not contain a path separator (got {value:?})"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::validation(format!(
            "team memory hook `{field}` must not contain control characters (got {value:?})"
        )));
    }
    let length = value.chars().count();
    if length > MAX_SCOPE_ID_CHARS {
        return Err(Error::validation(format!(
            "team memory hook `{field}` must be at most {MAX_SCOPE_ID_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}

/// A deterministic record id (FNV-1a 64-bit over `parts`, 16 lowercase hex characters).
///
/// The id depends only on the parts, so running the same hook twice upserts the same row
/// instead of duplicating it.
fn stable_id(parts: &[&str]) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(PART_SEPARATOR);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{HOOK_ID_PREFIX}{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::team::extractor::{LlmTeamExtractor, TeamExtraction};
    use crate::team::model::TeamMemoryFilter;
    use crate::team::observe::{TeamMemoryObserver, TeamMemoryOutcome};
    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use openjiuwen_test_support::ScriptedModel;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio_util::sync::CancellationToken;

    const TEAM: &str = "review-team";
    const DECISION: &str = "We decided to use SQLite for the shared store";
    const LESSON: &str = "Next time we should run the migration first";

    /// A fixed timestamp: the hooks never assert on wall-clock time.
    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).expect("a valid timestamp")
    }

    fn turn(from: &str, content: &str) -> TeamTurn {
        TeamTurn::new(from, content, at(1_767_000_000))
    }

    fn decision_turn() -> TeamTurn {
        turn("alice", DECISION)
    }

    fn lesson_turn() -> TeamTurn {
        turn("bob", LESSON)
    }

    fn rule_hooks(config: TeamMemoryHookConfig) -> (Arc<TeamMemory>, TeamMemoryHooks) {
        let memory = Arc::new(TeamMemory::in_memory());
        let hooks = TeamMemoryHooks::with_rule_extractor(TEAM, Arc::clone(&memory), config);
        (memory, hooks)
    }

    async fn stored(memory: &TeamMemory) -> Vec<TeamMemoryRecord> {
        memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("the team lists")
    }

    /// An extractor that counts its calls, optionally failing the first one.
    struct TestExtractor {
        calls: AtomicUsize,
        fail_first: bool,
        records: Vec<NewTeamMemory>,
    }

    impl TestExtractor {
        fn new(records: Vec<NewTeamMemory>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                fail_first: false,
                records,
            }
        }

        fn failing_once(records: Vec<NewTeamMemory>) -> Self {
            Self {
                fail_first: true,
                ..Self::new(records)
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl TeamMemoryExtractor for TestExtractor {
        #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
        fn name(&self) -> &str {
            "test"
        }

        async fn extract(
            &self,
            _transcript: &TeamTranscript,
            options: &CallOptions,
        ) -> Result<TeamExtraction> {
            options.check()?;
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 && self.fail_first {
                return Err(Error::memory("the extractor is unavailable"));
            }
            Ok(TeamExtraction::new(self.records.clone(), None))
        }
    }

    /// Collects every event an attached facade emits.
    #[derive(Clone, Default)]
    struct RecordingObserver {
        events: Arc<Mutex<Vec<TeamMemoryEvent>>>,
    }

    impl RecordingObserver {
        fn take(&self) -> Vec<TeamMemoryEvent> {
            std::mem::take(&mut *self.events.lock())
        }
    }

    impl TeamMemoryObserver for RecordingObserver {
        fn observe(&self, event: &TeamMemoryEvent) {
            self.events.lock().push(event.clone());
        }
    }

    #[tokio::test]
    async fn full_lifecycle_sequence_writes_profiles_then_extractions() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
        let call = CallOptions::new();
        assert_eq!(hooks.team_id(), TEAM);
        assert_eq!(hooks.extractor_name(), "rule");
        hooks.on_session_start("s-1", &call).await.unwrap();
        assert_eq!(hooks.session_id().as_deref(), Some("s-1"));
        let profile = hooks
            .on_member_join("alice", &call)
            .await
            .unwrap()
            .expect("a profile record");
        assert_eq!(profile.kind, TeamMemoryKind::MemberProfile);
        assert_eq!(profile.member_id.as_deref(), Some("alice"));
        assert_eq!(profile.session_id.as_deref(), Some("s-1"));
        assert_eq!(profile.source.as_deref(), Some(HOOK_SOURCE));
        assert_eq!(profile.title.as_deref(), Some("Member alice joined"));
        assert!((profile.importance - 0.4).abs() < f32::EPSILON);
        assert_eq!(
            profile.tags,
            vec!["member".to_owned(), "presence".to_owned()]
        );
        assert_eq!(profile.visibility, TeamVisibility::Team);
        assert_eq!(stored(&memory).await.len(), 1);
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.sync_turn(lesson_turn(), &call).await.unwrap();
        assert_eq!(hooks.pending_turns(), 2);
        assert_eq!(
            stored(&memory).await.len(),
            1,
            "turns are buffered, not written"
        );
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert_eq!(flush.turns, 2);
        assert_eq!(flush.len(), 2);
        assert!(flush.degraded.is_none());
        assert_eq!(hooks.pending_turns(), 0);
        assert_eq!(hooks.session_id().as_deref(), Some("s-1"));
        assert_eq!(hooks.members(), vec!["alice".to_owned()]);
        let kinds: Vec<TeamMemoryKind> = flush
            .records
            .iter()
            .map(|record| record.kind.clone())
            .collect();
        assert_eq!(
            kinds,
            vec![TeamMemoryKind::Decision, TeamMemoryKind::HandoffNote]
        );
        assert_eq!(flush.records[0].content, DECISION);
        assert_eq!(flush.records[0].member_id.as_deref(), Some("alice"));
        assert_eq!(flush.records[1].member_id.as_deref(), Some("bob"));
        assert!(flush
            .records
            .iter()
            .all(|record| record.session_id.as_deref() == Some("s-1")));
        assert_eq!(flush.records[0].source.as_deref(), Some("rule-extractor"));
        assert_eq!(stored(&memory).await.len(), 3);
        let again = hooks.on_session_end(&call).await.unwrap();
        assert!(again.is_empty());
        assert_eq!(again.turns, 0);
    }

    #[tokio::test]
    async fn a_failed_extraction_keeps_the_buffer_and_a_retry_converges() {
        let memory = Arc::new(TeamMemory::in_memory());
        let extractor = Arc::new(TestExtractor::failing_once(vec![
            NewTeamMemory::decision(TEAM, DECISION).with_member("alice"),
            NewTeamMemory::handoff_note(TEAM, LESSON).with_member("bob"),
        ]));
        let hooks = TeamMemoryHooks::new(
            TEAM,
            Arc::clone(&memory),
            extractor.clone() as Arc<dyn TeamMemoryExtractor>,
            TeamMemoryHookConfig::default(),
        );
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_turn(lesson_turn(), &call).await.unwrap();
        let error = hooks.on_session_end(&call).await.unwrap_err();
        assert!(matches!(error, Error::Memory(_)), "{error}");
        assert_eq!(hooks.pending_turns(), 2, "the buffer survives a failure");
        assert_eq!(stored(&memory).await.len(), 0);
        assert_eq!(extractor.calls(), 1);
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert_eq!(flush.len(), 2);
        assert_eq!(hooks.pending_turns(), 0);
        assert_eq!(stored(&memory).await.len(), 2);
        assert_eq!(extractor.calls(), 2);
        // Re-buffering the same turns under the same session upserts the same rows.
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_turn(lesson_turn(), &call).await.unwrap();
        let again = hooks.on_session_end(&call).await.unwrap();
        assert_eq!(again.len(), 2);
        assert_eq!(
            stored(&memory).await.len(),
            2,
            "deterministic ids are idempotent"
        );
        assert_eq!(again.records[0].id, flush.records[0].id);
        assert_eq!(again.records[1].id, flush.records[1].id);
    }

    #[tokio::test]
    async fn an_invalid_extracted_record_writes_nothing_and_keeps_the_buffer() {
        let memory = Arc::new(TeamMemory::in_memory());
        let extractor = Arc::new(TestExtractor::new(vec![
            NewTeamMemory::decision(TEAM, DECISION),
            NewTeamMemory::decision(TEAM, "   "),
        ]));
        let hooks = TeamMemoryHooks::new(
            TEAM,
            Arc::clone(&memory),
            extractor,
            TeamMemoryHookConfig::default(),
        );
        let call = CallOptions::new();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        let error = hooks.on_session_end(&call).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("content"), "{error}");
        assert_eq!(hooks.pending_turns(), 1);
        assert_eq!(stored(&memory).await.len(), 0, "no record was written");
    }

    #[tokio::test]
    async fn a_cancelled_token_leaves_the_state_untouched() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let cancelled = CallOptions::new().with_cancellation(token);
        let error = hooks.on_turn(lesson_turn(), &cancelled).await.unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(hooks.pending_turns(), 1);
        let error = hooks.on_member_join("alice", &cancelled).await.unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(hooks.members().len(), 0);
        let error = hooks.on_session_end(&cancelled).await.unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(hooks.pending_turns(), 1);
        let error = hooks.on_session_start("s-2", &cancelled).await.unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(hooks.session_id().as_deref(), Some("s-1"));
        assert_eq!(stored(&memory).await.len(), 0);
    }

    #[tokio::test]
    async fn the_turn_buffer_is_bounded_and_keeps_the_newest_turns() {
        let config = TeamMemoryHookConfig {
            max_pending_turns: 2,
            ..Default::default()
        };
        let (memory, hooks) = rule_hooks(config);
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        let contents = [
            DECISION,
            "We decided to ship on friday",
            "We decided to use Postgres instead of SQLite",
            "We decided to defer the migration to next sprint",
        ];
        for content in contents {
            hooks.on_turn(turn("alice", content), &call).await.unwrap();
        }
        assert_eq!(hooks.pending_turns(), 2);
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert_eq!(flush.turns, 2);
        assert_eq!(flush.records[0].content, contents[2]);
        assert_eq!(flush.records[1].content, contents[3]);
        assert_eq!(stored(&memory).await.len(), 2);
    }

    #[tokio::test]
    async fn a_degraded_llm_extraction_flushes_rule_records() {
        let memory = Arc::new(TeamMemory::in_memory());
        let model = ScriptedModel::builder().respond_text("not json").build();
        let hooks = TeamMemoryHooks::new(
            TEAM,
            Arc::clone(&memory),
            Arc::new(LlmTeamExtractor::new(
                Arc::new(model),
                TeamExtractConfig::default(),
            )),
            TeamMemoryHookConfig::default(),
        );
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_turn(lesson_turn(), &call).await.unwrap();
        let flush = hooks.on_session_end(&call).await.unwrap();
        let degraded = flush.degraded.as_deref().expect("a degradation reason");
        assert!(degraded.contains("fell back to rules"), "{degraded}");
        assert_eq!(flush.len(), 2);
        assert_eq!(flush.records[0].kind, TeamMemoryKind::Decision);
        assert_eq!(stored(&memory).await.len(), 2);
    }

    #[tokio::test]
    async fn disabled_hooks_write_nothing_and_keep_the_buffer() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig {
            enabled: false,
            ..Default::default()
        });
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        assert!(hooks
            .on_member_join("alice", &call)
            .await
            .unwrap()
            .is_none());
        assert_eq!(hooks.members(), vec!["alice".to_owned()]);
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        assert!(hooks
            .on_member_leave("alice", &call)
            .await
            .unwrap()
            .is_none());
        assert_eq!(hooks.members().len(), 0);
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert!(flush.is_empty());
        assert_eq!(flush.turns, 0);
        assert_eq!(
            hooks.pending_turns(),
            1,
            "a disabled flush keeps the buffer"
        );
        assert_eq!(stored(&memory).await.len(), 0);
    }

    #[tokio::test]
    async fn switches_suppress_individual_writes() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig {
            member_profile_on_join: false,
            handoff_on_leave: false,
            flush_on_session_end: false,
            ..Default::default()
        });
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        assert!(hooks
            .on_member_join("alice", &call)
            .await
            .unwrap()
            .is_none());
        assert_eq!(hooks.members(), vec!["alice".to_owned()]);
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_turn(lesson_turn(), &call).await.unwrap();
        assert!(hooks
            .on_member_leave("alice", &call)
            .await
            .unwrap()
            .is_none());
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert!(flush.is_empty());
        assert_eq!(hooks.pending_turns(), 2, "the buffer is preserved");
        assert_eq!(stored(&memory).await.len(), 0);
    }

    #[tokio::test]
    async fn observers_see_the_writes_and_the_flush() {
        let observer = RecordingObserver::default();
        let memory = Arc::new(TeamMemory::in_memory().with_observer(Arc::new(observer.clone())));
        let extractor = Arc::new(TestExtractor::new(vec![NewTeamMemory::decision(
            TEAM, DECISION,
        )]));
        let hooks = TeamMemoryHooks::new(
            TEAM,
            Arc::clone(&memory),
            extractor,
            TeamMemoryHookConfig::default(),
        );
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        hooks.on_member_join("alice", &call).await.unwrap().unwrap();
        hooks
            .on_member_leave("alice", &call)
            .await
            .unwrap()
            .unwrap();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_session_end(&call).await.unwrap();
        let events = observer.take();
        let writes = events
            .iter()
            .filter(|event| event.operation == TeamMemoryOperation::Write)
            .count();
        assert_eq!(writes, 3, "join, leave, and the extraction batch");
        let extract = events.last().expect("an extract event");
        assert_eq!(extract.operation, TeamMemoryOperation::Extract);
        assert_eq!(extract.team_id, TEAM);
        assert_eq!(extract.count, 1);
        assert_eq!(extract.outcome, TeamMemoryOutcome::Ok);
        assert!(extract.error.is_none());
        // A failing flush reports a failed extract event.
        let observer = RecordingObserver::default();
        let memory = Arc::new(TeamMemory::in_memory().with_observer(Arc::new(observer.clone())));
        let extractor = Arc::new(TestExtractor::failing_once(vec![NewTeamMemory::decision(
            TEAM, DECISION,
        )]));
        let hooks = TeamMemoryHooks::new(
            TEAM,
            Arc::clone(&memory),
            extractor,
            TeamMemoryHookConfig::default(),
        );
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        let error = hooks.on_session_end(&call).await.unwrap_err();
        assert!(matches!(error, Error::Memory(_)), "{error}");
        let events = observer.take();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].operation, TeamMemoryOperation::Extract);
        assert_eq!(events[0].outcome, TeamMemoryOutcome::Failed);
        assert_eq!(events[0].count, 0);
        assert!(events[0]
            .error
            .as_deref()
            .is_some_and(|text| text.contains("unavailable")));
    }

    #[tokio::test]
    async fn starting_a_new_session_discards_the_previous_turns() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_turn(lesson_turn(), &call).await.unwrap();
        assert_eq!(hooks.pending_turns(), 2);
        // The same session keeps its buffer.
        hooks.on_session_start("s-1", &call).await.unwrap();
        assert_eq!(hooks.pending_turns(), 2);
        hooks.on_session_start("s-2", &call).await.unwrap();
        assert_eq!(hooks.pending_turns(), 0);
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert_eq!(flush.turns, 1);
        assert_eq!(flush.records[0].session_id.as_deref(), Some("s-2"));
        assert_eq!(stored(&memory).await.len(), 1);
    }

    #[tokio::test]
    async fn unusable_ids_are_validation_errors() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
        let call = CallOptions::new();
        for session in ["", "   ", "a/b", "a\\b", "a\u{0}b", "a\nb"] {
            let error = hooks.on_session_start(session, &call).await.unwrap_err();
            assert!(matches!(error, Error::Validation(_)), "{error}");
            assert!(error.to_string().contains("session_id"), "{error}");
        }
        let long = "x".repeat(MAX_SCOPE_ID_CHARS + 1);
        let error = hooks.on_session_start(&long, &call).await.unwrap_err();
        assert!(error.to_string().contains("at most"), "{error}");
        assert!(hooks.session_id().is_none());
        for member in ["", "alice/bob"] {
            let error = hooks.on_member_join(member, &call).await.unwrap_err();
            assert!(error.to_string().contains("member_id"), "{error}");
            let error = hooks.on_member_leave(member, &call).await.unwrap_err();
            assert!(error.to_string().contains("member_id"), "{error}");
        }
        assert_eq!(hooks.members().len(), 0);
        assert_eq!(stored(&memory).await.len(), 0);
    }

    #[tokio::test]
    async fn an_invalid_config_is_rejected_before_writing() {
        let (memory, hooks) = rule_hooks(TeamMemoryHookConfig {
            handoff_importance: 1.5,
            ..Default::default()
        });
        let call = CallOptions::new();
        let error = hooks.on_member_join("alice", &call).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("handoff_importance"), "{error}");
        assert_eq!(stored(&memory).await.len(), 0);
        let error = TeamMemoryHookConfig {
            max_pending_turns: 0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.to_string().contains("max_pending_turns"), "{error}");
        assert!(TeamMemoryHookConfig::default().validate().is_ok());
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_hooks_persist_across_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("team-memory.db");
        let memory = Arc::new(TeamMemory::sqlite(&path).await.unwrap());
        let hooks = TeamMemoryHooks::with_rule_extractor(
            TEAM,
            Arc::clone(&memory),
            TeamMemoryHookConfig::default(),
        );
        let call = CallOptions::new();
        hooks.on_session_start("s-1", &call).await.unwrap();
        hooks.on_member_join("alice", &call).await.unwrap().unwrap();
        hooks.on_turn(decision_turn(), &call).await.unwrap();
        hooks.on_turn(lesson_turn(), &call).await.unwrap();
        let flush = hooks.on_session_end(&call).await.unwrap();
        assert_eq!(flush.len(), 2);
        drop(hooks);
        drop(memory);
        let reopened = TeamMemory::sqlite(&path).await.unwrap();
        let records = reopened
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .unwrap();
        assert_eq!(records.len(), 3);
        let profile = records
            .iter()
            .find(|record| record.kind == TeamMemoryKind::MemberProfile)
            .expect("the joined member persisted");
        assert_eq!(profile.member_id.as_deref(), Some("alice"));
        assert_eq!(profile.session_id.as_deref(), Some("s-1"));
        let extracted: Vec<&TeamMemoryRecord> = records
            .iter()
            .filter(|record| record.kind != TeamMemoryKind::MemberProfile)
            .collect();
        assert_eq!(extracted.len(), 2);
        assert!(extracted
            .iter()
            .all(|record| record.session_id.as_deref() == Some("s-1")));
    }
}
