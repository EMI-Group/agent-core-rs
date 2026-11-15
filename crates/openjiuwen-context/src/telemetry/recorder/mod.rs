//! The per-processor state recorder.
//!
//! A processor hands a [`ContextProcessorStateInput`] to a
//! [`ContextProcessorStateRecorder`] before and after its pass; the recorder
//! derives a [`ContextCompressionState`], keeps the last [`DEFAULT_HISTORY_LIMIT`]
//! records, and fans them out to an optional [`CompressionStateSink`] and an
//! optional session stream frame.
//! Telemetry never fails a processor pass: sink and stream failures are logged
//! and swallowed, and [`ContextProcessorStateRecorder::emit`] always returns the
//! record it built.

mod metrics;

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::SystemTime;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};
use openjiuwen_session::{Session, StreamItem};

use crate::offload::OffloadMarker;
use crate::session_memory::CONTEXT_MESSAGE_ID_KEY;
use crate::stats::{compute_stats, ContextStats};
use crate::telemetry::state::{
    CompressionStatePhase, CompressionStateStatus, ContextCompressionMetric,
    ContextCompressionModifiedMessage, ContextCompressionState, ContextCompressionUsage,
    CONTEXT_COMPRESSION_STATE_TYPE,
};
use crate::tokenizers::{SelectedTokenizer, TokenizerSource};
use crate::tokens::count_messages;
use crate::usage::context_tokens::STRING_LENGTH_NOTE;
use crate::usage::source_name;

pub use self::metrics::format_timestamp;

use self::metrics::{build_summary, context_percent, duration_ms, saved_delta};

/// How many records a recorder keeps in memory by default.
pub const DEFAULT_HISTORY_LIMIT: usize = 100;

/// The index a compression-state frame is published at, matching Python's
/// `OutputSchema(type=..., index=0, payload=...)`.
const STREAM_INDEX: i64 = 0;

/// Characters per token of the string-length fallback, Python's `len(text) // 3`.
const ESTIMATE_CHARS_PER_TOKEN: usize = 3;

/// What a recorder knows about how its token numbers are produced.
///
/// A record's `tokens` value is only useful if the reader knows whether it is a
/// count or an estimate, so every measurement carries this alongside it and the
/// record states both facts (`estimated` and `tokens_source`).
/// The vocabulary is shared with the crate's other honesty carriers:
/// [`TokenProvenance::source`] is [`source_name`] of a [`TokenizerSource`] (or the
/// recorded `string_length_estimate` fallback), and
/// [`TokenProvenance::note`] is that source's [`TokenizerSource::estimation_note`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenProvenance {
    /// Whether the count is an estimate rather than an exact count.
    estimated: bool,
    /// The counting strategy's wire name, when one can be named.
    source: Option<String>,
    /// Why the count is an estimate, when it is one.
    note: Option<String>,
}

impl TokenProvenance {
    /// A count produced by a counter that counts exactly and can name itself.
    #[must_use]
    pub fn counted(tokenizer: impl Into<String>) -> Self {
        Self {
            estimated: false,
            source: Some(tokenizer.into()),
            note: None,
        }
    }

    /// An estimate, naming the strategy and why it estimates.
    #[must_use]
    pub fn estimated(source: impl Into<String>, note: impl Into<String>) -> Self {
        Self {
            estimated: true,
            source: Some(source.into()),
            note: Some(note.into()),
        }
    }

    /// The provenance of a tokenizer source.
    ///
    /// Only [`TokenizerSource::NativeTokenizer`] is exact; every other source
    /// estimates, and the source's own note explains why.
    #[must_use]
    pub fn from_source(source: TokenizerSource) -> Self {
        Self {
            estimated: !source.is_native(),
            source: Some(source_name(source).to_owned()),
            note: source.estimation_note().map(str::to_owned),
        }
    }

    /// The provenance of an already-resolved tokenizer selection.
    ///
    /// The selection owns the artefact and model identities, while the record's
    /// `tokens_source` names the strategy, so this is
    /// [`TokenProvenance::from_source`] of the strategy the selector chose.
    #[must_use]
    pub fn of_selected(selected: &SelectedTokenizer) -> Self {
        Self::from_source(selected.source())
    }

    /// The provenance of the recorder's `ceil(characters / 3)` fallback.
    #[must_use]
    pub fn string_length_estimate() -> Self {
        Self::estimated(
            STRING_LENGTH_NOTE,
            "estimated from the character count at three characters per token",
        )
    }

    /// A count from a counter whose provenance cannot be inspected.
    ///
    /// A boxed [`TokenCounter`] cannot describe itself, so an undeclared counter
    /// is reported as an unnamed estimate rather than guessed exact.
    #[must_use]
    pub fn unverified(note: impl Into<String>) -> Self {
        Self {
            estimated: true,
            source: None,
            note: Some(note.into()),
        }
    }

    /// Whether the count is an estimate.
    #[must_use]
    pub fn is_estimated(&self) -> bool {
        self.estimated
    }

    /// The counting strategy's name, when one can be named.
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Why the count is an estimate, when it is one.
    #[must_use]
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }
}

impl Default for TokenProvenance {
    /// The safe default: an unnamed estimate, because nothing is known yet.
    fn default() -> Self {
        Self::unverified("the recorder's token counter does not declare how it counts")
    }
}

/// One measured token count together with how it was produced.
#[derive(Debug, Clone)]
struct MeasuredTokens {
    /// The measured or estimated total.
    tokens: usize,
    /// How `tokens` was produced.
    provenance: TokenProvenance,
}

/// The counter a recorder measures with, plus what it knows about it.
#[derive(Clone, Default)]
struct CounterBinding {
    /// The configured counter; `None` means the string-length fallback runs.
    counter: Option<Arc<dyn TokenCounter + Send + Sync>>,
    /// How the counter produces its numbers.
    provenance: TokenProvenance,
    /// Whether a caller set the counter, in which case the engine must not
    /// replace it.
    caller_set: bool,
}

// The counter has no `Debug` representation; the presence flag reports it.
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for CounterBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CounterBinding")
            .field("has_counter", &self.counter.is_some())
            .field("provenance", &self.provenance)
            .field("caller_set", &self.caller_set)
            .finish()
    }
}

/// The `ceil(characters / 3)` counter of the fallback path.
fn estimate_counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(ESTIMATE_CHARS_PER_TOKEN)
}

/// Where a recorder hands its finished records.
///
/// Implementations are shared as `Arc` and called once per
/// [`ContextProcessorStateRecorder::emit`]; an error is logged, never propagated.
#[async_trait]
pub trait CompressionStateSink: Send + Sync {
    /// Delivers one record.
    ///
    /// # Errors
    ///
    /// Any error is logged by the recorder and swallowed; returning one only
    /// marks the delivery as failed.
    async fn on_state(&self, state: &ContextCompressionState) -> Result<()>;
}

/// An in-memory [`CompressionStateSink`] that keeps every record it receives.
///
/// Tests read it back through [`RecordingSink::states`];
/// [`RecordingSink::failing`] rejects every record, which is how the suite proves
/// that delivery failures do not fail an `emit`.
#[derive(Debug, Default)]
pub struct RecordingSink {
    states: Mutex<Vec<ContextCompressionState>>,
    fail: bool,
}

impl RecordingSink {
    /// Creates a sink that accepts and keeps every record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a sink whose every delivery fails.
    #[must_use]
    pub fn failing() -> Self {
        Self {
            states: Mutex::new(Vec::new()),
            fail: true,
        }
    }

    /// Every record received so far, in delivery order.
    #[must_use]
    pub fn states(&self) -> Vec<ContextCompressionState> {
        self.states
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The statuses of the records received so far, in delivery order.
    #[must_use]
    pub fn statuses(&self) -> Vec<CompressionStateStatus> {
        self.states()
            .into_iter()
            .map(|state| state.status)
            .collect()
    }

    /// The most recent record, if any.
    #[must_use]
    pub fn last(&self) -> Option<ContextCompressionState> {
        self.states
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .last()
            .cloned()
    }

    /// How many records were received.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether no record was received yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl CompressionStateSink for RecordingSink {
    async fn on_state(&self, state: &ContextCompressionState) -> Result<()> {
        if self.fail {
            return Err(Error::validation("recording sink is configured to fail"));
        }
        self.states
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(state.clone());
        Ok(())
    }
}

/// Everything a recorder needs to describe one processor pass.
///
/// `before` is the history the processor received and `after` is what it
/// produced (`None` for a `started` record); the optional statistics let a caller
/// supply per-role counts it already computed.
#[derive(Debug, Clone, Default)]
pub struct ContextProcessorStateInput<'a> {
    /// Id correlating the pass with the model calls it makes.
    pub operation_id: String,
    /// Where in the lifecycle this record sits.
    pub status: CompressionStateStatus,
    /// Which context-engine entry point produced the record.
    pub phase: CompressionStatePhase,
    /// Name of the processor that ran.
    pub processor: String,
    /// Model the processor was configured with, when it has one.
    pub model: String,
    /// The context window the percentages are measured against.
    pub context_max: Option<u32>,
    /// The history the processor received.
    pub before: &'a [BaseMessage],
    /// The history the processor produced, when the pass is over.
    pub after: Option<&'a [BaseMessage]>,
    /// Per-role counts of `before`, when the caller already has them.
    pub stats_before: Option<ContextStats>,
    /// Per-role counts of `after`, when the caller already has them.
    pub stats_after: Option<ContextStats>,
    /// Model tokens the pass consumed.
    pub usage: Option<ContextCompressionUsage>,
    /// Error text of a failed pass.
    pub error: Option<String>,
    /// Why this state was reached.
    pub reason: String,
    /// Overrides the generated summary.
    pub summary: Option<String>,
    /// Short summary the processor produced.
    pub compact_summary: String,
    /// When the pass started.
    pub started_at: Option<SystemTime>,
    /// When the pass ended.
    pub ended_at: Option<SystemTime>,
}

impl<'a> ContextProcessorStateInput<'a> {
    /// Creates an input for `operation_id`/`status`/`phase` of `processor`.
    #[must_use]
    pub fn new(
        operation_id: impl Into<String>,
        status: CompressionStateStatus,
        phase: CompressionStatePhase,
        processor: impl Into<String>,
        before: &'a [BaseMessage],
    ) -> Self {
        Self {
            operation_id: operation_id.into(),
            status,
            phase,
            processor: processor.into(),
            before,
            ..Self::default()
        }
    }

    /// Sets the history the pass produced.
    #[must_use]
    pub fn with_after(mut self, after: &'a [BaseMessage]) -> Self {
        self.after = Some(after);
        self
    }

    /// Overrides [`ContextProcessorStateInput::model`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the window the percentages refer to.
    #[must_use]
    pub fn with_context_max(mut self, context_max: u32) -> Self {
        self.context_max = Some(context_max);
        self
    }

    /// Supplies the per-role counts of the before and after histories.
    #[must_use]
    pub fn with_stats(mut self, before: ContextStats, after: Option<ContextStats>) -> Self {
        self.stats_before = Some(before);
        self.stats_after = after;
        self
    }

    /// Sets the model usage of the pass.
    #[must_use]
    pub fn with_usage(mut self, usage: ContextCompressionUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Sets when the pass started and ended.
    #[must_use]
    pub fn with_timing(mut self, started_at: SystemTime, ended_at: Option<SystemTime>) -> Self {
        self.started_at = Some(started_at);
        self.ended_at = ended_at;
        self
    }

    /// Overrides the generated summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Sets the short summary a processor produced.
    #[must_use]
    pub fn with_compact_summary(mut self, compact_summary: impl Into<String>) -> Self {
        self.compact_summary = compact_summary.into();
        self
    }

    /// Sets why the record's state was reached.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Sets the error of a failed pass.
    #[must_use]
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }
}

/// Builds, keeps, and fans out compression-state records for one context.
///
/// The recorder is `Send + Sync` and shared as `Arc`; the sink, session, and
/// token counter can be replaced at any time, and no lock is held across an
/// `await`.
///
/// A record's `before`/`after` token counts are measured here, so they are only
/// as good as the recorder's counter: with one installed they carry the
/// counter's provenance, and without one every count is the
/// `ceil(characters / 3)` string-length estimate, published with
/// `estimated = true` and `tokens_source = "string_length_estimate"`. The first
/// such measurement of an instance logs one warning naming the missing counter.
pub struct ContextProcessorStateRecorder {
    session_id: String,
    context_id: String,
    history_limit: usize,
    history: RwLock<Vec<ContextCompressionState>>,
    sink: RwLock<Option<Arc<dyn CompressionStateSink>>>,
    session: RwLock<Option<Arc<dyn Session>>>,
    counter: RwLock<CounterBinding>,
    estimate_warned: AtomicBool,
}

// The three trait-object fields have no `Debug` representation; the presence
// flags below report them instead.
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for ContextProcessorStateRecorder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ContextProcessorStateRecorder")
            .field("session_id", &self.session_id)
            .field("context_id", &self.context_id)
            .field("history_limit", &self.history_limit)
            .field("history_len", &self.history_len())
            .field("has_sink", &self.sink().is_some())
            .field("has_session", &self.session().is_some())
            .field("has_token_counter", &self.token_counter().is_some())
            .field("token_provenance", &self.token_provenance())
            .finish()
    }
}

impl ContextProcessorStateRecorder {
    /// Creates a recorder for one context, with no sink, session, or counter.
    #[must_use]
    pub fn new(session_id: impl Into<String>, context_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            context_id: context_id.into(),
            history_limit: DEFAULT_HISTORY_LIMIT,
            history: RwLock::new(Vec::new()),
            sink: RwLock::new(None),
            session: RwLock::new(None),
            counter: RwLock::new(CounterBinding::default()),
            estimate_warned: AtomicBool::new(false),
        }
    }

    /// Overrides [`DEFAULT_HISTORY_LIMIT`]; `0` keeps no history at all.
    #[must_use]
    pub fn with_history_limit(mut self, history_limit: usize) -> Self {
        self.history_limit = history_limit;
        self
    }

    /// Attaches the record sink.
    #[must_use]
    pub fn with_sink(self, sink: Arc<dyn CompressionStateSink>) -> Self {
        self.set_sink(Some(sink));
        self
    }

    /// Attaches the session a record is streamed to.
    #[must_use]
    pub fn with_session(self, session: Arc<dyn Session>) -> Self {
        self.set_session(Some(session));
        self
    }

    /// Attaches the token counter used to measure messages and build statistics.
    ///
    /// A counter attached this way reports an unnamed estimate
    /// ([`TokenProvenance::default`]) until its provenance is declared with
    /// [`ContextProcessorStateRecorder::with_token_provenance`], because a boxed
    /// [`TokenCounter`] cannot describe itself.
    #[must_use]
    pub fn with_token_counter(self, counter: Arc<dyn TokenCounter + Send + Sync>) -> Self {
        self.set_token_counter(Some(counter));
        self
    }

    /// Attaches the token counter together with how it produces its numbers.
    #[must_use]
    pub fn with_token_counter_provenance(
        self,
        counter: Arc<dyn TokenCounter + Send + Sync>,
        provenance: TokenProvenance,
    ) -> Self {
        self.set_token_counter_with_provenance(counter, provenance);
        self
    }

    /// Declares how the configured counter produces its numbers.
    ///
    /// This is the declaration the record's `estimated`/`tokens_source` pair is
    /// published from; [`TokenProvenance::from_source`] and
    /// [`TokenProvenance::of_selected`] derive it from the crate's tokenizer
    /// selection, so the record names the same strategy the counter uses.
    #[must_use]
    pub fn with_token_provenance(self, provenance: TokenProvenance) -> Self {
        self.set_token_provenance(provenance);
        self
    }

    /// Returns the owning session id.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the owning context id.
    #[must_use]
    pub fn context_id(&self) -> &str {
        &self.context_id
    }

    /// Returns the in-memory history cap.
    #[must_use]
    pub fn history_limit(&self) -> usize {
        self.history_limit
    }

    /// Replaces the sink; `None` disables delivery.
    pub fn set_sink(&self, sink: Option<Arc<dyn CompressionStateSink>>) {
        *self.sink.write().unwrap_or_else(PoisonError::into_inner) = sink;
    }

    /// Replaces the session a record is streamed to; `None` disables streaming.
    pub fn set_session(&self, session: Option<Arc<dyn Session>>) {
        *self.session.write().unwrap_or_else(PoisonError::into_inner) = session;
    }

    /// Replaces the token counter used for measurement and statistics.
    ///
    /// A counter set this way is caller-owned: the engine never replaces it, and
    /// its provenance resets to [`TokenProvenance::default`] (an unnamed
    /// estimate) unless [`ContextProcessorStateRecorder::set_token_provenance`]
    /// declares it. `None` returns to the string-length fallback and lets a later
    /// engine installation install its own counter.
    pub fn set_token_counter(&self, counter: Option<Arc<dyn TokenCounter + Send + Sync>>) {
        let mut binding = self.counter.write().unwrap_or_else(PoisonError::into_inner);
        binding.caller_set = counter.is_some();
        binding.provenance = TokenProvenance::default();
        binding.counter = counter;
    }

    /// Replaces the counter and its provenance together.
    pub fn set_token_counter_with_provenance(
        &self,
        counter: Arc<dyn TokenCounter + Send + Sync>,
        provenance: TokenProvenance,
    ) {
        let mut binding = self.counter.write().unwrap_or_else(PoisonError::into_inner);
        binding.counter = Some(counter);
        binding.provenance = provenance;
        binding.caller_set = true;
    }

    /// Declares how the configured counter produces its numbers.
    pub fn set_token_provenance(&self, provenance: TokenProvenance) {
        self.counter
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .provenance = provenance;
    }

    /// How the configured counter produces its numbers.
    #[must_use]
    pub fn token_provenance(&self) -> TokenProvenance {
        self.counter_binding().provenance
    }

    /// Installs the engine's counter unless a caller already set one.
    ///
    /// The engine measures its contexts with the counter it hands to the
    /// pipeline, and the pipeline computes the record's per-role statistic with
    /// that same counter; installing it here makes the record's `before`/`after`
    /// counts equal to the statistic by construction.
    /// Precedence: a counter the caller set ([`ContextProcessorStateRecorder::with_token_counter`],
    /// [`ContextProcessorStateRecorder::set_token_counter`], or
    /// [`ContextProcessorStateRecorder::set_token_counter_with_provenance`]) wins
    /// and is never replaced.
    pub(crate) fn install_token_counter(
        &self,
        counter: Arc<dyn TokenCounter + Send + Sync>,
        provenance: TokenProvenance,
    ) {
        let mut binding = self.counter.write().unwrap_or_else(PoisonError::into_inner);
        if binding.caller_set {
            return;
        }
        binding.counter = Some(counter);
        binding.provenance = provenance;
    }

    /// Every record kept so far, oldest first.
    #[must_use]
    pub fn history(&self) -> Vec<ContextCompressionState> {
        self.history
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The most recent record, if any.
    #[must_use]
    pub fn last(&self) -> Option<ContextCompressionState> {
        self.history
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .last()
            .cloned()
    }

    /// Drops every kept record.
    pub fn clear_history(&self) {
        self.history
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Seeds the history from persisted records, keeping the newest ones.
    pub fn load_history(&self, history: Vec<ContextCompressionState>) {
        let mut incoming = history;
        if incoming.len() > self.history_limit {
            let excess = incoming.len() - self.history_limit;
            incoming.drain(..excess);
        }
        *self.history.write().unwrap_or_else(PoisonError::into_inner) = incoming;
    }

    /// Builds the record for `input` without recording or delivering it.
    #[must_use]
    pub fn build_state(&self, input: &ContextProcessorStateInput<'_>) -> ContextCompressionState {
        let before = self.metric(input.before, input.context_max, input.started_at);
        let after = input
            .after
            .map(|after| self.metric(after, input.context_max, input.ended_at));
        let saved = saved_delta(&before, after.as_ref());
        let modified = input
            .after
            .map_or_else(Vec::new, |after| modified_messages(input.before, after));
        let summary = input.summary.clone().unwrap_or_else(|| {
            build_summary(
                input.status,
                &before,
                after.as_ref(),
                saved.as_ref(),
                &input.reason,
                modified.len(),
            )
        });
        ContextCompressionState {
            state_type: CONTEXT_COMPRESSION_STATE_TYPE.to_owned(),
            operation_id: input.operation_id.clone(),
            status: input.status,
            phase: input.phase,
            processor: input.processor.clone(),
            model: input.model.clone(),
            before,
            after,
            statistic: self.statistic(input),
            saved,
            modified_messages: modified,
            compression_usage: input.usage.clone(),
            duration_ms: duration_ms(input.started_at, input.ended_at),
            context_max: input.context_max,
            summary,
            compact_summary: input.compact_summary.clone(),
            error: input.error.clone(),
            reason: input.reason.clone(),
        }
    }

    /// Builds the record for `input`, keeps it, and fans it out.
    ///
    /// The record is appended to the bounded history, handed to the sink, and
    /// written to the session stream as a [`CONTEXT_COMPRESSION_STATE_TYPE`]
    /// frame; failures are logged and swallowed.
    pub async fn emit(&self, input: ContextProcessorStateInput<'_>) -> ContextCompressionState {
        let state = self.build_state(&input);
        self.record(&state);
        tracing::info!(
            status = state.status.as_str(),
            phase = state.phase.as_str(),
            reason = state.reason.as_str(),
            processor = state.processor.as_str(),
            session_id = %self.session_id,
            context_id = %self.context_id,
            operation_id = %state.operation_id,
            before_messages = state.before.messages,
            before_tokens = state.before.tokens,
            before_tokens_estimated = state.before.estimated,
            before_tokens_source = state.before.tokens_source.as_deref(),
            after_messages = state.after.as_ref().map(|metric| metric.messages),
            after_tokens = state.after.as_ref().map(|metric| metric.tokens),
            "context compression state"
        );
        self.deliver_to_sink(&state).await;
        self.deliver_to_session(&state).await;
        state
    }

    async fn deliver_to_sink(&self, state: &ContextCompressionState) {
        let Some(sink) = self.sink() else {
            return;
        };
        if let Err(error) = sink.on_state(state).await {
            tracing::warn!(
                %error,
                session_id = %self.session_id,
                context_id = %self.context_id,
                operation_id = %state.operation_id,
                status = state.status.as_str(),
                "failed to deliver context compression state to the sink"
            );
        }
    }

    async fn deliver_to_session(&self, state: &ContextCompressionState) {
        let Some(session) = self.session() else {
            tracing::debug!(
                session_id = %self.session_id,
                context_id = %self.context_id,
                operation_id = %state.operation_id,
                "context compression state stream skipped: no session writer"
            );
            return;
        };
        let payload = match serde_json::to_value(state) {
            Ok(payload) => payload,
            Err(error) => {
                tracing::warn!(
                    %error,
                    session_id = %self.session_id,
                    context_id = %self.context_id,
                    operation_id = %state.operation_id,
                    "failed to serialize context compression state"
                );
                return;
            }
        };
        let item = StreamItem::output(CONTEXT_COMPRESSION_STATE_TYPE, Some(STREAM_INDEX), payload);
        if let Err(error) = session.write_stream(item).await {
            tracing::warn!(
                %error,
                session_id = %self.session_id,
                context_id = %self.context_id,
                operation_id = %state.operation_id,
                status = state.status.as_str(),
                "failed to emit context compression state to the session stream"
            );
        } else {
            tracing::debug!(
                session_id = %self.session_id,
                context_id = %self.context_id,
                operation_id = %state.operation_id,
                status = state.status.as_str(),
                "context compression state stream emitted"
            );
        }
    }

    fn sink(&self) -> Option<Arc<dyn CompressionStateSink>> {
        self.sink
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn session(&self) -> Option<Arc<dyn Session>> {
        self.session
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn token_counter(&self) -> Option<Arc<dyn TokenCounter + Send + Sync>> {
        self.counter_binding().counter
    }

    fn counter_binding(&self) -> CounterBinding {
        self.counter
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn history_len(&self) -> usize {
        self.history
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    fn record(&self, state: &ContextCompressionState) {
        let mut history = self.history.write().unwrap_or_else(PoisonError::into_inner);
        if self.history_limit == 0 {
            return;
        }
        history.push(state.clone());
        if history.len() > self.history_limit {
            let excess = history.len() - self.history_limit;
            history.drain(..excess);
        }
    }

    fn metric(
        &self,
        messages: &[BaseMessage],
        context_max: Option<u32>,
        observed_at: Option<SystemTime>,
    ) -> ContextCompressionMetric {
        let measured = self.measure(messages);
        ContextCompressionMetric {
            time: observed_at.map(format_timestamp),
            messages: messages.len(),
            tokens: measured.tokens,
            context_percent: context_percent(measured.tokens, context_max),
            estimated: measured.provenance.is_estimated(),
            tokens_source: measured.provenance.source().map(str::to_owned),
        }
    }

    /// Measures with the configured counter, or with the string-length estimate.
    fn measure(&self, messages: &[BaseMessage]) -> MeasuredTokens {
        let (counter, provenance) = self.measurement_source();
        let tokens = counter.map_or_else(
            || count_messages(&estimate_counter(), messages),
            |counter| counter.count_messages(messages),
        );
        MeasuredTokens { tokens, provenance }
    }

    /// The counter the next measurement uses, plus how it produces its numbers.
    ///
    /// Without a counter this is the `ceil(characters / 3)` fallback, which is
    /// announced once per recorder instance rather than once per pass.
    fn measurement_source(&self) -> (Option<Arc<dyn TokenCounter + Send + Sync>>, TokenProvenance) {
        let binding = self.counter_binding();
        if let Some(counter) = binding.counter {
            (Some(counter), binding.provenance)
        } else {
            self.warn_about_estimated_tokens();
            (None, TokenProvenance::string_length_estimate())
        }
    }

    /// Warns once per recorder instance that its counts are estimates.
    fn warn_about_estimated_tokens(&self) {
        if self.estimate_warned.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::warn!(
            session_id = %self.session_id,
            context_id = %self.context_id,
            tokens_source = STRING_LENGTH_NOTE,
            "publishing estimated context compression state: this recorder has no token counter, \
             so every `tokens` value is the ceil(characters / 3) string-length estimate and is \
             flagged `estimated`; install a counter with `with_token_counter` (and declare how it \
             counts with `with_token_provenance`) to make the numbers exact and equal to the \
             per-role statistic"
        );
    }

    fn statistic(&self, input: &ContextProcessorStateInput<'_>) -> ContextStats {
        // A caller that already computed the per-role counts owns them, including
        // their provenance: the processor pipeline measures with the context's own
        // counter, which this recorder cannot inspect.
        if let Some(stats) = input
            .stats_after
            .clone()
            .or_else(|| input.stats_before.clone())
        {
            return stats;
        }
        let messages = input.after.unwrap_or(input.before);
        let (counter, provenance) = self.measurement_source();
        let stats = match counter {
            Some(counter) => compute_stats(messages, counter.as_ref()),
            None => compute_stats(messages, &estimate_counter()),
        };
        stats.with_estimated(provenance.is_estimated())
    }
}

/// Reports the messages present in both lists whose content or offload state
/// changed, in output order.
///
/// Indices cannot name a rewrite (compressors index the input list, offloaders
/// the output list), but a message keeps its `context_message_id` across a
/// rewrite, so matching ids names exactly the rewritten ones. Messages without
/// an id, and removed or newly created messages, are not rewrites.
#[must_use]
pub fn modified_messages(
    before: &[BaseMessage],
    after: &[BaseMessage],
) -> Vec<ContextCompressionModifiedMessage> {
    let mut before_by_id: HashMap<&str, &BaseMessage> = HashMap::new();
    for message in before {
        if let Some(message_id) = context_message_id(message) {
            before_by_id.entry(message_id).or_insert(message);
        }
    }

    let mut modified = Vec::new();
    for message in after {
        let Some(message_id) = context_message_id(message) else {
            continue;
        };
        let Some(original) = before_by_id.get(message_id) else {
            continue;
        };
        let new_marker = OffloadMarker::parse(&message.content.to_plain_text());
        let old_handle = OffloadMarker::parse(&original.content.to_plain_text()).map(|m| m.handle);
        let newly_offloaded = new_marker
            .as_ref()
            .is_some_and(|marker| Some(&marker.handle) != old_handle.as_ref());
        if !newly_offloaded && original.content == message.content {
            continue;
        }
        let (offload_handle, offload_type) = if newly_offloaded {
            new_marker.as_ref().map_or((None, None), |marker| {
                (
                    Some(marker.handle.clone()),
                    Some(marker.kind.as_marker_str().to_owned()),
                )
            })
        } else {
            (None, None)
        };
        modified.push(ContextCompressionModifiedMessage {
            message_id: message_id.to_owned(),
            role: message.role.as_str().to_owned(),
            tool_call_id: message.tool_call_id.clone(),
            offload_handle,
            offload_type,
        });
    }
    modified
}

/// The `context_message_id` of `message`, when it carries one.
fn context_message_id(message: &BaseMessage) -> Option<&str> {
    message
        .metadata
        .get(CONTEXT_MESSAGE_ID_KEY)
        .and_then(|value| value.as_str())
        .filter(|id| !id.is_empty())
}

#[cfg(test)]
mod tests;
