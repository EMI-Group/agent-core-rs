//! Pipeline execution: the ordered run of processors, its telemetry and its
//! checkpoints.
//!
//! [`ContextPipeline`] owns the processor list and runs it, either plainly or
//! instrumented: with a [`ContextProcessorStateRecorder`] attached every pass is
//! bracketed by `context.compression_state` records and wrapped in a compaction
//! operation scope, so a model call a processor makes carries the pass's
//! operation id. [`PipelineDeps`] is what a pipeline build needs beyond the
//! serialized [`ProcessorConfig`]s — the stores and seams that cannot be
//! serialized. Execution never mutates stored history by itself: it runs on the
//! caller's working copy.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::SystemTime;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, Usage};

use super::compressor::{
    CurrentRoundCompressor, DialogueCompressor, FullCompactProcessor, FullCompactProcessorConfig,
    MicroCompactProcessor, RoundLevelCompressor,
};
use super::config::{ProcessorConfig, ProcessorContext};
use super::offloader::{MessageOffloader, ToolResultBudgetProcessor};
use super::summarize::SummarizationProcessor;
use super::summary_offloader::MessageSummaryOffloader;
use super::tool_result::ToolResultProcessor;
use super::tool_result_window::ToolResultWindowProcessor;
use super::truncate::TruncationProcessor;
use super::ContextProcessor;
use crate::events::ContextEvent;
use crate::offload::OffloadStore;
use crate::processor::compressor::reinjection::SnapshotRegistry;
use crate::recall::RecallArchiveStore;
use crate::session_memory::SessionMemoryStore;
use crate::session_memory_agent::SessionMemoryAgent;
use crate::stats::compute_stats;
use crate::telemetry::{
    with_operation, CompressionStatePhase, CompressionStateStatus, ContextCompressionUsage,
    ContextProcessorStateInput, ContextProcessorStateRecorder,
};
use crate::window::drop_orphan_tool_results;

/// The stores and seams a pipeline hands to the processors it builds.
///
/// Building a processor needs more than the [`ProcessorConfig`]: offloaded
/// content, the summarisation model, recall archives, and session-memory notes
/// all live outside the serialized configuration. [`PipelineDeps`] carries them
/// so [`ContextPipeline::from_configs_with`] can wire every processor that uses
/// one.
#[derive(Clone)]
pub struct PipelineDeps {
    /// Store offloaded content is written to; required.
    pub offload: Arc<dyn OffloadStore>,
    /// Model used by processors that call one (summarisation, compaction).
    pub model: Option<Arc<dyn Model>>,
    /// Archive store compressors write replaced history to.
    pub recall: Option<Arc<dyn RecallArchiveStore>>,
    /// Session-memory notes full compaction reinjects.
    pub session_memory: Option<Arc<dyn SessionMemoryStore>>,
    /// Agent full compaction runs to extract notes the session-memory store lacks.
    ///
    /// `None` lets [`ContextPipeline::from_configs_with`] build the default
    /// agent whenever a session-memory store is mounted for an enabled
    /// session-memory strategy; an explicitly supplied agent always wins.
    pub session_memory_agent: Option<Arc<SessionMemoryAgent>>,
    /// Host-rendered state sections full compaction reinjects after a pass.
    pub reinjection: Option<Arc<SnapshotRegistry>>,
    /// Recorder every processor pass of the built pipeline reports to.
    ///
    /// `None` leaves the pipeline uninstrumented: no history is cloned and no
    /// telemetry is produced.
    pub recorder: Option<Arc<ContextProcessorStateRecorder>>,
}

impl std::fmt::Debug for PipelineDeps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineDeps")
            .field("offload", &self.offload.kind())
            .field("has_model", &self.model.is_some())
            .field("has_recall", &self.recall.is_some())
            .field("has_session_memory", &self.session_memory.is_some())
            .field(
                "has_session_memory_agent",
                &self.session_memory_agent.is_some(),
            )
            .field("has_reinjection", &self.reinjection.is_some())
            .field("has_recorder", &self.recorder.is_some())
            .finish()
    }
}

impl PipelineDeps {
    /// The minimum a pipeline needs: an offload store and nothing else.
    #[must_use]
    pub fn new(offload: Arc<dyn OffloadStore>) -> Self {
        Self {
            offload,
            model: None,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
            reinjection: None,
            recorder: None,
        }
    }

    /// Sets the summarisation model.
    #[must_use]
    pub fn with_model(mut self, model: Option<Arc<dyn Model>>) -> Self {
        self.model = model;
        self
    }

    /// Sets the recall archive store.
    #[must_use]
    pub fn with_recall(mut self, recall: Option<Arc<dyn RecallArchiveStore>>) -> Self {
        self.recall = recall;
        self
    }

    /// Sets the session-memory store.
    #[must_use]
    pub fn with_session_memory(
        mut self,
        session_memory: Option<Arc<dyn SessionMemoryStore>>,
    ) -> Self {
        self.session_memory = session_memory;
        self
    }

    /// Sets the registry of state sections full compaction may reinject.
    #[must_use]
    pub fn with_reinjection(mut self, reinjection: Option<Arc<SnapshotRegistry>>) -> Self {
        self.reinjection = reinjection;
        self
    }

    /// Sets the agent full compaction runs to extract missing session-memory notes.
    ///
    /// [`ContextPipeline::from_configs_with`] falls back to
    /// [`SessionMemoryAgent::default`] when a store is mounted and no agent is
    /// supplied, so an explicit agent is the only way to change the extraction
    /// configuration of a pipeline.
    #[must_use]
    pub fn with_session_memory_agent(mut self, agent: Option<Arc<SessionMemoryAgent>>) -> Self {
        self.session_memory_agent = agent;
        self
    }

    /// Sets the recorder the built pipeline reports every processor pass to.
    #[must_use]
    pub fn with_recorder(mut self, recorder: Option<Arc<ContextProcessorStateRecorder>>) -> Self {
        self.recorder = recorder;
        self
    }
}

// Why a recorded pass was reached, in Python's `ContextProcessorStateInput`
// reason spellings. They are private: a record carries the string, never a flag.
/// The processor pass began.
const REASON_TRIGGERED: &str = "processor_triggered";
/// The processor reported a change, or changed the history without one.
const REASON_COMPLETED: &str = "processor_completed";
/// The processor ran and changed nothing.
const REASON_NOOP: &str = "processor_noop";
/// The processor had an empty history to work on.
const REASON_SKIPPED: &str = "processor_skipped";
/// The processor raised an error.
const REASON_ERROR: &str = "processor_error";
/// The processor stopped because the run was cancelled.
const REASON_CANCELLED: &str = "user_cancel";

/// Accumulates the model usage of the processor pass that is currently running.
///
/// The pipeline binds one collector per instrumented pass; the shared
/// compaction summariser
/// ([`summarize`](crate::processor::summarize)) reports what each model call
/// cost, and the pass's terminal record carries the total as
/// `compression_usage`. Outside an instrumented pass the task-local is unbound,
/// so a summariser call records nothing and the record leaves
/// `compression_usage` as `None` — usage is never fabricated.
#[derive(Debug, Default)]
struct PassUsage {
    usage: std::sync::Mutex<Option<ContextCompressionUsage>>,
}

impl PassUsage {
    /// Adds one model call's usage; an empty usage is not a measurement and is
    /// ignored.
    fn record(&self, usage: &Usage, model_name: &str) {
        if usage.is_empty() {
            return;
        }
        let mut total = self
            .usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match total.as_mut() {
            Some(total) => total.record(usage),
            None => *total = Some(ContextCompressionUsage::from_usage(usage, model_name)),
        }
    }

    /// The total recorded so far, `None` when no call reported a usage.
    fn snapshot(&self) -> Option<ContextCompressionUsage> {
        self.usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

tokio::task_local! {
    /// Collector of the instrumented pass the current task runs in.
    static PASS_USAGE: Arc<PassUsage>;
}

/// Reports the usage of one processor model call to the running pass.
///
/// Called by the shared compaction summariser. The call is a no-op when no
/// instrumented pass is in scope, so an uninstrumented pipeline pays only a
/// failed task-local lookup.
pub(crate) fn record_model_usage(usage: &Usage, model_name: &str) {
    let _ = PASS_USAGE.try_with(|pass| pass.record(usage, model_name));
}

/// The state as a pass that returned successfully leaves it.
///
/// Mirrors Python's `status = "completed" if event is not None else "noop"` and
/// additionally treats an undeclared history change as a completion, so a
/// processor that rewrites without reporting cannot hide the change. An empty
/// history a processor left alone means it had nothing to do, which is the one
/// case recorded as [`CompressionStateStatus::Skipped`].
fn finished_status(
    event: Option<&ContextEvent>,
    before: &[BaseMessage],
    after: &[BaseMessage],
) -> CompressionStateStatus {
    if event.is_some() || before != after {
        return CompressionStateStatus::Completed;
    }
    if before.is_empty() {
        return CompressionStateStatus::Skipped;
    }
    CompressionStateStatus::Noop
}

/// Why a finished pass reached `status`, in Python's reason spellings.
const fn finished_reason(status: CompressionStateStatus) -> &'static str {
    match status {
        CompressionStateStatus::Completed => REASON_COMPLETED,
        CompressionStateStatus::Skipped => REASON_SKIPPED,
        CompressionStateStatus::Failed => REASON_ERROR,
        CompressionStateStatus::Cancelled => REASON_CANCELLED,
        CompressionStateStatus::Started
        | CompressionStateStatus::Noop
        | CompressionStateStatus::Paused => REASON_NOOP,
    }
}

/// An ordered list of processors.
#[derive(Clone, Default)]
pub struct ContextPipeline {
    processors: Vec<Arc<dyn ContextProcessor>>,
    recorder: Option<Arc<ContextProcessorStateRecorder>>,
}

impl std::fmt::Debug for ContextPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self
            .processors
            .iter()
            .map(|processor| processor.name())
            .collect();
        f.debug_struct("ContextPipeline")
            .field("processors", &names)
            .field("recording", &self.recorder.is_some())
            .finish()
    }
}

impl ContextPipeline {
    /// Creates a pipeline over `processors`, in execution order.
    ///
    /// The pipeline records nothing; attach a recorder with
    /// [`ContextPipeline::with_recorder`].
    #[must_use]
    pub fn new(processors: Vec<Arc<dyn ContextProcessor>>) -> Self {
        Self {
            processors,
            recorder: None,
        }
    }

    /// Builds the pipeline described by `configs`.
    ///
    /// Delegates to [`ContextPipeline::from_configs_with`] with only the model
    /// and the offload store set, so recall archives and session-memory notes
    /// stay unconfigured.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a summarisation processor is configured
    /// without a model, and when an offloader config fails its own
    /// [`validate`](crate::processor::MessageOffloaderConfig::validate) checks. The summary
    /// offloader never fails for a missing model: it falls back to the
    /// deterministic offline summary. Truncation and tool-result processors
    /// never fail.
    pub fn from_configs(
        configs: &[ProcessorConfig],
        model: Option<Arc<dyn Model>>,
        offload: Arc<dyn OffloadStore>,
    ) -> Result<Self> {
        Self::from_configs_with(configs, &PipelineDeps::new(offload).with_model(model))
    }

    /// Builds the pipeline described by `configs` with the full set of stores.
    ///
    /// This is [`ContextPipeline::from_configs`] plus the seams that are not
    /// part of a [`ProcessorConfig`]: [`FullCompactProcessor`] receives
    /// [`PipelineDeps::session_memory`], [`PipelineDeps::session_memory_agent`],
    /// [`PipelineDeps::reinjection`] and [`PipelineDeps::model`], and
    /// [`MessageSummaryOffloader`] receives [`PipelineDeps::model`]. The recall
    /// store is not held by a processor; it reaches them through
    /// [`ProcessorContext::recall`] when the pipeline runs.
    ///
    /// The session-memory agent follows one precedence: the
    /// [`PipelineDeps::session_memory_agent`] the caller supplied, otherwise a
    /// default [`SessionMemoryAgent`] whenever a store is mounted and
    /// [`FullCompactProcessorConfig::session_memory_enabled`] is on, otherwise
    /// none — and the compaction then reports why its session-memory branch
    /// could not run.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a summarisation processor is configured
    /// without a model, and when an offloader config fails its own
    /// [`validate`](crate::processor::MessageOffloaderConfig::validate) checks. The summary
    /// offloader and full compaction never fail for a missing model: they fall
    /// back to a deterministic offline summary.
    pub fn from_configs_with(configs: &[ProcessorConfig], deps: &PipelineDeps) -> Result<Self> {
        let mut processors: Vec<Arc<dyn ContextProcessor>> = Vec::with_capacity(configs.len());
        for config in configs {
            match config {
                ProcessorConfig::Truncation(config) => {
                    processors.push(Arc::new(TruncationProcessor {
                        config: config.clone(),
                    }));
                }
                ProcessorConfig::ToolResult(config) => {
                    processors.push(Arc::new(ToolResultProcessor {
                        config: config.clone(),
                        store: Arc::clone(&deps.offload),
                    }));
                }
                ProcessorConfig::Summarization(config) => {
                    let model = deps.model.clone().ok_or_else(|| {
                        Error::Config(
                            "the summarization processor requires a model, but none is configured"
                                .to_owned(),
                        )
                    })?;
                    processors.push(Arc::new(SummarizationProcessor {
                        config: config.clone(),
                        model,
                    }));
                }
                ProcessorConfig::DialogueCompressor(config) => {
                    config.validate()?;
                    processors.push(Arc::new(DialogueCompressor::new(
                        config.clone(),
                        deps.model.clone(),
                    )));
                }
                ProcessorConfig::CurrentRoundCompressor(config) => {
                    config.validate()?;
                    processors.push(Arc::new(CurrentRoundCompressor::new(
                        config.clone(),
                        deps.model.clone(),
                    )));
                }
                ProcessorConfig::RoundLevel(config) => {
                    config.validate()?;
                    processors.push(Arc::new(RoundLevelCompressor::new(
                        config.clone(),
                        deps.model.clone(),
                    )));
                }
                ProcessorConfig::FullCompact(config) => {
                    // The session-memory notes are preferred over a model
                    // summary; without notes the model or offline path runs.
                    processors.push(full_compact_processor(config, deps));
                }
                ProcessorConfig::MicroCompact(config) => {
                    config.validate()?;
                    processors.push(Arc::new(MicroCompactProcessor::new(config.clone())));
                }
                ProcessorConfig::MessageOffloader(config) => {
                    config.validate()?;
                    processors.push(Arc::new(MessageOffloader::new(
                        config.clone(),
                        Arc::clone(&deps.offload),
                    )));
                }
                ProcessorConfig::ToolResultBudget(config) => {
                    config.validate()?;
                    processors.push(Arc::new(ToolResultBudgetProcessor::new(
                        config.clone(),
                        Arc::clone(&deps.offload),
                    )));
                }
                ProcessorConfig::ToolResultWindow(config) => {
                    config.validate()?;
                    processors.push(Arc::new(ToolResultWindowProcessor::new(
                        config.clone(),
                        Arc::clone(&deps.offload),
                    )));
                }
                ProcessorConfig::MessageSummaryOffloader(config) => {
                    // A missing model is not an error: the processor falls back
                    // to the deterministic offline summary.
                    processors.push(Arc::new(MessageSummaryOffloader::new(
                        config.clone(),
                        deps.model.clone(),
                        Arc::clone(&deps.offload),
                    )));
                }
            }
        }
        Ok(Self {
            processors,
            recorder: deps.recorder.clone(),
        })
    }

    /// Attaches the recorder every processor pass of this pipeline reports to.
    ///
    /// `None` (the default) disables recording: the pipeline then runs exactly
    /// as it did before telemetry existed — no history snapshot, no statistics,
    /// and no compaction-operation scope around a processor.
    #[must_use]
    pub fn with_recorder(mut self, recorder: Option<Arc<ContextProcessorStateRecorder>>) -> Self {
        self.recorder = recorder;
        self
    }

    /// The recorder attached to this pipeline, if any.
    #[must_use]
    pub fn recorder(&self) -> Option<&Arc<ContextProcessorStateRecorder>> {
        self.recorder.as_ref()
    }

    /// The processors, in execution order.
    #[must_use]
    pub fn processors(&self) -> &[Arc<dyn ContextProcessor>] {
        &self.processors
    }

    /// Serializes the checkpoint state of every processor that keeps any.
    ///
    /// Entries are keyed by [`ContextProcessor::name`] in pipeline order; a
    /// second processor with the same name becomes `"<name>#2"`, a third
    /// `"<name>#3"`. Processors that report [`serde_json::Value::Null`] are
    /// omitted, so the map only holds real checkpoints.
    #[must_use]
    pub fn save_state(&self) -> BTreeMap<String, serde_json::Value> {
        let mut state = BTreeMap::new();
        for (processor, key) in self.state_keys() {
            let value = processor.save_state();
            if !value.is_null() {
                state.insert(key, value);
            }
        }
        state
    }

    /// Restores the processor checkpoints a [`ContextPipeline::save_state`] map
    /// carries.
    ///
    /// Every entry is handed to the processor its key names, using the same key
    /// scheme [`ContextPipeline::save_state`] used. A key that matches no
    /// processor is ignored and logged at debug level, so a checkpoint written
    /// by a different pipeline configuration still loads.
    pub fn load_state(&self, state: &BTreeMap<String, serde_json::Value>) {
        let keys = self.state_keys();
        for (processor, key) in &keys {
            if let Some(value) = state.get(key) {
                processor.load_state(value);
            }
        }
        for key in state.keys() {
            if !keys.iter().any(|(_, known)| known == key) {
                tracing::debug!(
                    processor = %key,
                    "ignored checkpoint state for an unknown processor"
                );
            }
        }
    }

    /// `(processor, checkpoint key)` pairs in pipeline order.
    ///
    /// The counter is positional, so save and load always agree on a key even
    /// when a processor in between keeps no state.
    fn state_keys(&self) -> Vec<(&Arc<dyn ContextProcessor>, String)> {
        let mut seen: HashMap<&'static str, usize> = HashMap::new();
        self.processors
            .iter()
            .map(|processor| {
                let name = processor.name();
                let count = seen.entry(name).or_insert(0);
                *count += 1;
                let key = if *count == 1 {
                    name.to_owned()
                } else {
                    format!("{name}#{count}")
                };
                (processor, key)
            })
            .collect()
    }

    /// Runs every processor in order and returns the events they produced.
    ///
    /// Cancellation is checked before each processor. A processor error is
    /// propagated as-is; changes made by earlier processors stay applied, and
    /// every processor that mutates builds its replacement first so a failure
    /// never leaves a half-written history. Afterwards orphan tool results are
    /// dropped (a processor may have removed the assistant message that owned
    /// one) and the repair is logged.
    ///
    /// The pass is recorded in phase [`CompressionStatePhase::ActiveCompress`],
    /// because this pipeline is the compaction seam (Python's
    /// `active_compress`). A caller that runs the same pipeline for another
    /// context-engine entry point passes the phase explicitly through
    /// [`ContextPipeline::run_in_phase`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the cancellation token is set, and the
    /// failing processor's error otherwise.
    pub async fn run(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Vec<ContextEvent>> {
        self.run_in_phase(messages, cx, CompressionStatePhase::ActiveCompress)
            .await
    }

    /// Runs every processor in order, recording each pass in `phase`.
    ///
    /// This is [`ContextPipeline::run`] with the phase Python derives from the
    /// engine entry point (`add_messages`, `get_context_window`,
    /// `active_compress`).
    ///
    /// When a recorder is attached every processor pass is bracketed by two
    /// records — `started` before the processor runs, then `completed` / `noop` /
    /// `skipped` / `failed` / `cancelled` — and the processor runs inside a
    /// [`with_operation`] scope, so a model call it makes carries the pass's
    /// operation id. Each record carries the processor name, the phase, the
    /// operation id, the configured model name, the context window the
    /// percentages are measured against, the before/after metrics and
    /// [`ContextStats`](crate::stats::ContextStats), the pass duration, the
    /// model usage the pass reported, and the event summary when the processor
    /// produced one. The compression trigger threshold is not part of the
    /// record's wire schema, so it is not reported. Recording never fails a
    /// pass: a sink or stream error is logged and swallowed by the recorder.
    ///
    /// Status mapping, following Python's
    /// `status = "completed" if event is not None else "noop"`: a processor that
    /// returns an event, or changes the history without one, is `completed`; one
    /// that leaves a non-empty history unchanged is `noop`; one that is handed an
    /// empty history and changes nothing is `skipped` (it had nothing to work
    /// on); an error is `failed`, and [`Error::Cancelled`] is `cancelled`. A
    /// cancellation noticed before a pass starts records nothing and returns
    /// immediately, matching Python's loop guard.
    ///
    /// Without a recorder the pipeline is byte-identical to the uninstrumented
    /// one: the history is never cloned and no task-local scope is entered.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the cancellation token is set, and the
    /// failing processor's error otherwise (after recording it).
    pub async fn run_in_phase(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
        phase: CompressionStatePhase,
    ) -> Result<Vec<ContextEvent>> {
        let mut events = Vec::new();
        match self.recorder.as_deref() {
            Some(recorder) => {
                self.run_recorded(recorder, messages, cx, phase, &mut events)
                    .await?;
            }
            None => {
                for processor in &self.processors {
                    if cx.cancellation.is_cancelled() {
                        return Err(Error::Cancelled);
                    }
                    if let Some(event) = processor.process(messages, cx).await? {
                        events.push(event);
                    }
                }
            }
        }
        let dropped = drop_orphan_tool_results(messages);
        if dropped > 0 {
            tracing::warn!(
                dropped,
                "dropped orphan tool results after the pipeline ran"
            );
        }
        Ok(events)
    }

    /// The instrumented loop of [`ContextPipeline::run_in_phase`].
    ///
    /// Every pass clones the working history once (the recorder needs the before
    /// image to measure and diff), emits a `started` record, runs the processor
    /// inside the operation scope with a usage collector bound, and emits the
    /// terminal record before returning or propagating the error.
    async fn run_recorded(
        &self,
        recorder: &ContextProcessorStateRecorder,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
        phase: CompressionStatePhase,
        events: &mut Vec<ContextEvent>,
    ) -> Result<()> {
        let context_max = context_max(cx);
        let model = cx.model.map(|model| model.model_name().to_owned());
        for processor in &self.processors {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let before = messages.clone();
            let stats_before = compute_stats(&before, cx.counter);
            let operation_id = uuid::Uuid::new_v4().simple().to_string();
            let started_at = SystemTime::now();
            let started = ContextProcessorStateInput::new(
                operation_id.clone(),
                CompressionStateStatus::Started,
                phase,
                processor.name(),
                &before,
            )
            .with_stats(stats_before.clone(), None)
            .with_timing(started_at, None)
            .with_reason(REASON_TRIGGERED);
            recorder
                .emit(with_pass_context(started, model.as_deref(), context_max))
                .await;

            let usage = Arc::new(PassUsage::default());
            let outcome = with_operation(
                operation_id.clone(),
                PASS_USAGE.scope(Arc::clone(&usage), processor.process(messages, cx)),
            )
            .await;
            let ended_at = SystemTime::now();
            let stats_after = compute_stats(messages, cx.counter);
            let status = match &outcome {
                Ok(event) => finished_status(event.as_ref(), &before, messages),
                Err(Error::Cancelled) => CompressionStateStatus::Cancelled,
                Err(_) => CompressionStateStatus::Failed,
            };
            let terminal = ContextProcessorStateInput::new(
                operation_id,
                status,
                phase,
                processor.name(),
                &before,
            )
            .with_after(messages.as_slice())
            .with_stats(stats_before, Some(stats_after))
            .with_timing(started_at, Some(ended_at))
            .with_reason(finished_reason(status));
            let mut terminal = with_pass_context(terminal, model.as_deref(), context_max);
            if let Some(usage) = usage.snapshot() {
                terminal = terminal.with_usage(usage);
            }
            match outcome {
                Ok(event) => {
                    if let Some(summary) = event.as_ref().and_then(|event| event.summary.as_deref())
                    {
                        terminal = terminal.with_compact_summary(summary);
                    }
                    recorder.emit(terminal).await;
                    if let Some(event) = event {
                        events.push(event);
                    }
                }
                Err(error) => {
                    if status == CompressionStateStatus::Failed {
                        terminal = terminal.with_error(error.to_string());
                    }
                    recorder.emit(terminal).await;
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

/// The context window the record percentages are measured against.
///
/// The configured window is preferred, falling back to the pass budget when the
/// configuration declares none; a window that does not fit in the record's
/// `u32` field is reported as unknown rather than truncated.
fn context_max(cx: &ProcessorContext<'_>) -> Option<u32> {
    let tokens = cx.config.max_context_tokens.unwrap_or(cx.budget);
    u32::try_from(tokens).ok()
}

/// Applies the optional model name and context window to `input`.
fn with_pass_context<'a>(
    mut input: ContextProcessorStateInput<'a>,
    model: Option<&str>,
    context_max: Option<u32>,
) -> ContextProcessorStateInput<'a> {
    if let Some(model) = model {
        input = input.with_model(model);
    }
    if let Some(context_max) = context_max {
        input = input.with_context_max(context_max);
    }
    input
}

/// Builds the full-compaction processor, including its session-memory agent.
///
/// The agent precedence is an explicit [`PipelineDeps::session_memory_agent`] →
/// a default [`SessionMemoryAgent`] whenever a store is mounted for an enabled
/// session-memory strategy → none, in which case the compaction reports why its
/// session-memory branch could not run.
fn full_compact_processor(
    config: &FullCompactProcessorConfig,
    deps: &PipelineDeps,
) -> Arc<dyn ContextProcessor> {
    let mut processor = FullCompactProcessor::new(
        config.clone(),
        deps.model.clone(),
        deps.session_memory.clone(),
    );
    let agent = deps.session_memory_agent.clone().or_else(|| {
        (config.session_memory_enabled && deps.session_memory.is_some())
            .then(|| Arc::new(SessionMemoryAgent::default()))
    });
    if let Some(agent) = agent {
        processor = processor.with_session_memory_agent(agent);
    }
    if let Some(registry) = deps.reinjection.as_ref() {
        processor = processor.with_reinjection(Arc::clone(registry));
    }
    Arc::new(processor)
}
