//! Model-backed memory consolidation with a deterministic fallback.
//!
//! [`LlmConsolidator`] implements [`DreamConsolidator`] by turning one claimed dream job
//! into a bounded list of candidate memories, asking an [`openjiuwen_foundation::llm::Model`]
//! which records to merge, keep, summarize, or forget, and applying the answer through an
//! injected [`ConsolidationSink`]. It is the pipeline the planner's
//! [`DreamConsolidator`] seam was designed for.
//!
//! # Request shape
//!
//! The model receives two messages (see [`super::prompts`]): a system message with the
//! rules plus the response schema, and a user message whose body is JSON:
//!
//! ```json
//! {
//!   "job_id": "…",
//!   "kind": "consolidation",
//!   "scope": {"user_id": "…", "scope_id": "…"},
//!   "max_actions": 12,
//!   "candidates": [
//!     {"id": "…", "text": "…", "kind": "semantic_memory", "importance": 0.5,
//!      "timestamp": "2025-01-01T00:00:00Z", "access_count": 0}
//!   ]
//! }
//! ```
//!
//! The answer is a JSON object with an `actions` array. Candidate count and per-candidate
//! text are both bounded by [`LlmConsolidatorConfig`], and every id in the answer must
//! name a supplied candidate: invented ids are dropped before a plan is built.
//!
//! # Degradation
//!
//! Every failure mode — no model configured, a disabled configuration, a provider error,
//! a timeout, or an answer with no usable actions — produces a deterministic plan from
//! [`deterministic_plan`] (rule-based merge of near-duplicates + keep everything else).
//! The plan's [`DreamConsolidationPlan::degraded`] field names the reason, so a caller (or a
//! [`DreamOutcome`] note) never mistakes a fallback for model output. Cancellation is the
//! one exception: it is returned as [`Error::Cancelled`] instead of degrading, because a
//! cancelled job must stop, not consume the remaining retries.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};

use crate::record::{CallOptions, MemoryRecord, MemoryScope, MemoryType};

use super::model::{DreamJob, DreamMetrics};
use super::outcome::DreamOutcome;
use super::planner::DreamConsolidator;
use super::prompts::{CONSOLIDATION_RESPONSE_SCHEMA, CONSOLIDATION_SYSTEM_PROMPT};

mod plan;
pub use plan::*;

#[cfg(test)]
mod tests;

/// The default per-candidate character budget sent to the model.
const DEFAULT_MAX_CONTENT_CHARS: usize = 500;

/// The payload key [`PayloadCandidateSource`] reads.
pub const PAYLOAD_CANDIDATES_KEY: &str = "candidates";

/// The deterministic fallback similarity threshold (token Jaccard).
const DEFAULT_FALLBACK_SIMILARITY: f32 = 0.75;

/// One memory offered to the consolidation pipeline.
///
/// The shape is intentionally close to [`MemoryRecord`] (and accepts its `text`/`type`
/// JSON aliases) so a caller can project stored records without a second schema. Only the
/// fields consolidation decisions need are carried; embeddings and metadata stay out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamCandidate {
    /// Record id; must be non-empty and NUL-free.
    pub id: String,
    /// Memory text; JSON field `"text"` (accepts `"content"` and `"mem"`).
    #[serde(rename = "text", alias = "content", alias = "mem")]
    pub content: String,
    /// Memory kind; JSON field `"type"` (accepts `"memory_type"`).
    #[serde(rename = "type", alias = "memory_type")]
    pub kind: MemoryType,
    /// Relative importance in `0.0..=1.0`.
    pub importance: f32,
    /// Creation time; the Unix epoch when absent from JSON.
    pub timestamp: DateTime<Utc>,
    /// Number of recalls that returned the record.
    pub access_count: u64,
}

/// Returns the Unix epoch, the default [`DreamCandidate::timestamp`].
///
/// A missing timestamp is deliberately the epoch rather than "now": it makes a
/// deserialized candidate reproducible and keeps prompt text stable.
fn candidate_epoch() -> DateTime<Utc> {
    DateTime::from_timestamp(0, 0).unwrap_or(DateTime::<Utc>::MIN_UTC)
}

impl Default for DreamCandidate {
    fn default() -> Self {
        Self {
            id: String::new(),
            content: String::new(),
            kind: MemoryType::Unknown,
            importance: 0.5,
            timestamp: candidate_epoch(),
            access_count: 0,
        }
    }
}

impl DreamCandidate {
    /// Builds a candidate with `id`, `content`, unknown kind, epoch timestamp, and the
    /// default importance `0.5`.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            ..Self::default()
        }
    }

    /// Projects a stored [`MemoryRecord`] into a candidate, copying the decision fields.
    #[must_use]
    pub fn from_record(record: &MemoryRecord) -> Self {
        Self {
            id: record.id.clone(),
            content: record.content.clone(),
            kind: record.kind,
            importance: record.importance,
            timestamp: record.timestamp,
            access_count: record.access_count,
        }
    }

    /// Returns the candidate with `kind`.
    #[must_use]
    pub const fn with_kind(mut self, kind: MemoryType) -> Self {
        self.kind = kind;
        self
    }

    /// Returns the candidate with `importance`.
    #[must_use]
    pub const fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    /// Returns the candidate created at `timestamp`.
    #[must_use]
    pub const fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Returns the candidate with `access_count`.
    #[must_use]
    pub const fn with_access_count(mut self, access_count: u64) -> Self {
        self.access_count = access_count;
        self
    }

    /// Checks that the id is usable and the importance is a finite fraction.
    pub fn validate(&self) -> Result<()> {
        super::model::validate_non_empty("dream candidate `id`", &self.id)?;
        if !self.importance.is_finite() || !(0.0..=1.0).contains(&self.importance) {
            return Err(Error::validation(format!(
                "dream candidate `{}` importance must be finite and in 0.0..=1.0 (got {})",
                self.id, self.importance
            )));
        }
        Ok(())
    }
}

/// Supplies the memories one dream job should consolidate.
///
/// The source owns candidate selection (recall, scope listing, job payload, …); the
/// pipeline only bounds and serialises what it returns. Implementations must respect
/// `ctx` around expensive awaits and never mutate memory themselves.
#[async_trait]
pub trait DreamCandidateSource: Send + Sync {
    /// Returns the candidates for `job`, in the order they should be considered.
    async fn candidates(&self, job: &DreamJob, ctx: &CallOptions) -> Result<Vec<DreamCandidate>>;
}

/// A source that always returns the same candidates.
///
/// This is the test/embedding shape: a caller that already knows the candidate set (or
/// wants a fixed replay) configures it once.
#[derive(Debug, Clone, Default)]
pub struct StaticCandidateSource {
    /// Candidates returned in order.
    candidates: Vec<DreamCandidate>,
}

impl StaticCandidateSource {
    /// Builds a source over `candidates`.
    #[must_use]
    pub fn new(candidates: Vec<DreamCandidate>) -> Self {
        Self { candidates }
    }

    /// Builds a source that returns nothing.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
}

impl From<Vec<DreamCandidate>> for StaticCandidateSource {
    fn from(candidates: Vec<DreamCandidate>) -> Self {
        Self::new(candidates)
    }
}

#[async_trait]
impl DreamCandidateSource for StaticCandidateSource {
    async fn candidates(&self, _job: &DreamJob, ctx: &CallOptions) -> Result<Vec<DreamCandidate>> {
        ctx.check()?;
        Ok(self.candidates.clone())
    }
}

/// A source that reads candidates from the job payload under
/// [`PAYLOAD_CANDIDATES_KEY`].
///
/// An absent key yields no candidates (a job may be enqueued without a snapshot and
/// consolidate nothing). A present but malformed value is [`Error::Validation`] naming the
/// job, because silently consolidating nothing would hide a producer bug.
#[derive(Debug, Clone, Copy, Default)]
pub struct PayloadCandidateSource;

#[async_trait]
impl DreamCandidateSource for PayloadCandidateSource {
    async fn candidates(&self, job: &DreamJob, ctx: &CallOptions) -> Result<Vec<DreamCandidate>> {
        ctx.check()?;
        let Some(value) = job.payload.get(PAYLOAD_CANDIDATES_KEY) else {
            return Ok(Vec::new());
        };
        serde_json::from_value::<Vec<DreamCandidate>>(value.clone()).map_err(|error| {
            Error::validation(format!(
                "dream job `{}` payload `{PAYLOAD_CANDIDATES_KEY}` is not a candidate list: {error}",
                job.id
            ))
        })
    }
}

/// Applies one consolidation plan to the memory store.
///
/// The default [`RecordOnlySink`] mutates nothing and returns the plan's own counters —
/// that is the pure "decide, don't write" mode useful for dry runs and tests. A real
/// application injects a sink backed by its memory facade; a sink error fails the
/// attempt (the planner retries it), and a sink must never turn a cancellation into a
/// success.
#[async_trait]
pub trait ConsolidationSink: Send + Sync {
    /// Applies `plan` and returns the counters to record on the completed job.
    async fn apply(
        &self,
        job: &DreamJob,
        plan: &DreamConsolidationPlan,
        ctx: &CallOptions,
    ) -> Result<DreamMetrics>;
}

/// The sink used when none is injected: it records the plan's counters and mutates
/// nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecordOnlySink;

#[async_trait]
impl ConsolidationSink for RecordOnlySink {
    async fn apply(
        &self,
        _job: &DreamJob,
        plan: &DreamConsolidationPlan,
        ctx: &CallOptions,
    ) -> Result<DreamMetrics> {
        ctx.check()?;
        Ok(plan.metrics)
    }
}

/// Knobs for [`LlmConsolidator`].
///
/// JSON is `snake_case` with defaults for every field. `max_candidates` bounds how many
/// candidates enter the prompt (the first ones in source order), `max_content_chars`
/// truncates each candidate's text, `model_retries` is the number of *additional* model
/// attempts after the first, and `fallback_similarity` is the token-Jaccard threshold the
/// deterministic fallback uses to call two records duplicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LlmConsolidatorConfig {
    /// Whether the model may be called at all; default `true`.
    ///
    /// A disabled pipeline always uses the deterministic fallback (with a degraded
    /// reason), which is the offline-testing and cost-control switch.
    pub enabled: bool,
    /// Additional model attempts after the first failure; default `1`.
    pub model_retries: u32,
    /// Maximum candidates sent to the model; default `50`.
    pub max_candidates: usize,
    /// Maximum characters of each candidate's text; default `500`.
    pub max_content_chars: usize,
    /// Token-Jaccard threshold for the deterministic merge rule; default `0.75`.
    pub fallback_similarity: f32,
}

impl Default for LlmConsolidatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_retries: 1,
            max_candidates: 50,
            max_content_chars: DEFAULT_MAX_CONTENT_CHARS,
            fallback_similarity: DEFAULT_FALLBACK_SIMILARITY,
        }
    }
}

impl LlmConsolidatorConfig {
    /// Returns the configuration with the model switch set.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Returns the configuration with `model_retries` extra attempts.
    #[must_use]
    pub const fn with_model_retries(mut self, model_retries: u32) -> Self {
        self.model_retries = model_retries;
        self
    }

    /// Returns the configuration with a candidate cap.
    #[must_use]
    pub const fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.max_candidates = max_candidates;
        self
    }

    /// Returns the configuration with a per-candidate text cap.
    #[must_use]
    pub const fn with_max_content_chars(mut self, max_content_chars: usize) -> Self {
        self.max_content_chars = max_content_chars;
        self
    }

    /// Returns the configuration with a fallback similarity threshold.
    #[must_use]
    pub const fn with_fallback_similarity(mut self, fallback_similarity: f32) -> Self {
        self.fallback_similarity = fallback_similarity;
        self
    }

    /// Validates the bounds: at least one candidate and one character, at most ten retries,
    /// and a finite similarity in `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.max_candidates == 0 {
            return Err(Error::validation(
                "llm consolidator `max_candidates` must be greater than 0",
            ));
        }
        if self.max_content_chars == 0 {
            return Err(Error::validation(
                "llm consolidator `max_content_chars` must be greater than 0",
            ));
        }
        if self.model_retries > 10 {
            return Err(Error::validation(format!(
                "llm consolidator `model_retries` must be at most 10 (got {})",
                self.model_retries
            )));
        }
        if !self.fallback_similarity.is_finite() || !(0.0..=1.0).contains(&self.fallback_similarity)
        {
            return Err(Error::validation(format!(
                "llm consolidator `fallback_similarity` must be finite and in 0.0..=1.0 (got {})",
                self.fallback_similarity
            )));
        }
        Ok(())
    }
}

/// Model-backed consolidation with a deterministic fallback.
///
/// Build it with [`LlmConsolidator::new`] (explicit model + source) or
/// [`LlmConsolidator::payload_driven`] (reads candidates from the job payload), inject a
/// store writer with [`LlmConsolidator::with_sink`], and hand it to a
/// [`DreamPlanner`](crate::dreaming::DreamPlanner) as the `DreamConsolidator`.
pub struct LlmConsolidator {
    /// The model, or `None` for the deterministic-only configuration.
    model: Option<Arc<dyn Model>>,
    /// Candidate selection.
    source: Arc<dyn DreamCandidateSource>,
    /// Prompt bounds, retries, and fallback threshold.
    config: LlmConsolidatorConfig,
    /// Plan application.
    sink: Arc<dyn ConsolidationSink>,
}

impl fmt::Debug for LlmConsolidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmConsolidator")
            .field("model", &self.model.as_deref().map(Model::model_name))
            .field("source", &"Arc<dyn DreamCandidateSource>")
            .field("config", &self.config)
            .field("sink", &"Arc<dyn ConsolidationSink>")
            .finish_non_exhaustive()
    }
}

impl LlmConsolidator {
    /// Builds a consolidator around `model` and `source`, validating `config`.
    ///
    /// `None` is a valid model: the pipeline then always uses the deterministic fallback
    /// and records `"no consolidation model is configured"` as the degradation reason.
    pub fn new(
        model: Option<Arc<dyn Model>>,
        source: Arc<dyn DreamCandidateSource>,
        config: LlmConsolidatorConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            model,
            source,
            config,
            sink: Arc::new(RecordOnlySink),
        })
    }

    /// Builds a consolidator whose candidates come from the job payload.
    pub fn payload_driven(
        model: Option<Arc<dyn Model>>,
        config: LlmConsolidatorConfig,
    ) -> Result<Self> {
        Self::new(model, Arc::new(PayloadCandidateSource), config)
    }

    /// Returns the consolidator with `sink` as its plan applier.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn ConsolidationSink>) -> Self {
        self.sink = sink;
        self
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &LlmConsolidatorConfig {
        &self.config
    }

    /// Produces a plan for `job` without applying it.
    ///
    /// The model is attempted `model_retries + 1` times; each unusable answer (provider
    /// error, empty response, unparseable JSON, or only invented ids) counts as a failed
    /// attempt. Once the budget is spent, the deterministic plan is returned with the
    /// first failure as [`DreamConsolidationPlan::degraded`]. Cancellation is checked before
    /// the first call and between attempts and is propagated, never degraded.
    pub async fn plan(&self, job: &DreamJob, ctx: &CallOptions) -> Result<DreamConsolidationPlan> {
        ctx.check()?;
        self.config.validate()?;
        let mut candidates = self.source.candidates(job, ctx).await?;
        for candidate in &candidates {
            candidate.validate()?;
        }
        let total = candidates.len();
        candidates.truncate(self.config.max_candidates);
        if candidates.is_empty() {
            return Ok(self.rules_plan(job, &candidates, "no candidate memories to consolidate"));
        }
        if let Some(reason) = self.unavailable_reason() {
            return Ok(self.rules_plan(job, &candidates, &reason));
        }
        let Some(model) = self.model.as_ref() else {
            return Ok(self.rules_plan(job, &candidates, "no consolidation model is configured"));
        };
        if total > candidates.len() {
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::pipeline",
                job_id = %job.id,
                total,
                kept = candidates.len(),
                "bounded the consolidation candidate list"
            );
        }
        let messages = self.build_messages(job, &candidates)?;
        let request = ModelRequestConfig::new(model.model_name().to_owned());
        let attempts = self.config.model_retries.saturating_add(1);
        let mut failure: Option<String> = None;
        for attempt in 1..=attempts {
            match ctx.run(model.invoke(&messages, &request)).await {
                Ok(Ok(response)) => {
                    let raw = response
                        .message()
                        .map(|message| message.text().into_owned())
                        .unwrap_or_default();
                    if let Some(plan) = parse_consolidation_response(&raw, &job.id, &candidates) {
                        return Ok(plan);
                    }
                    failure.get_or_insert_with(|| {
                        "the model answer had no usable consolidation actions".to_owned()
                    });
                    tracing::warn!(
                        target: "openjiuwen_memory::dreaming::pipeline",
                        job_id = %job.id,
                        attempt,
                        attempts,
                        "the consolidation answer was unusable; retrying"
                    );
                }
                Ok(Err(error)) | Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    failure.get_or_insert_with(|| format!("the model call failed: {error}"));
                    tracing::warn!(
                        target: "openjiuwen_memory::dreaming::pipeline",
                        job_id = %job.id,
                        attempt,
                        attempts,
                        %error,
                        "the consolidation model call failed; retrying"
                    );
                }
            }
            ctx.check()?;
        }
        let failure = failure.unwrap_or_else(|| "no attempt was made".to_owned());
        tracing::warn!(
            target: "openjiuwen_memory::dreaming::pipeline",
            job_id = %job.id,
            attempts,
            %failure,
            "llm consolidation exhausted its retries; falling back to rules"
        );
        Ok(self.rules_plan(
            job,
            &candidates,
            &format!("llm consolidation fell back to rules: {failure}"),
        ))
    }

    /// Returns why the model cannot be used, if it cannot.
    fn unavailable_reason(&self) -> Option<String> {
        if self.config.enabled {
            None
        } else {
            Some("the consolidation model is disabled in the configuration".to_owned())
        }
    }

    /// Builds the deterministic plan and stamps `reason` as its degradation.
    fn rules_plan(
        &self,
        job: &DreamJob,
        candidates: &[DreamCandidate],
        reason: &str,
    ) -> DreamConsolidationPlan {
        let mut plan = deterministic_plan(&job.id, candidates, self.config.fallback_similarity);
        plan.degraded = Some(reason.to_owned());
        plan
    }

    /// Builds the system and user messages for one model call.
    fn build_messages(
        &self,
        job: &DreamJob,
        candidates: &[DreamCandidate],
    ) -> Result<Vec<BaseMessage>> {
        let body = render_request(job, candidates, self.config.max_content_chars)?;
        Ok(vec![
            BaseMessage::system(format!(
                "{CONSOLIDATION_SYSTEM_PROMPT}\n{CONSOLIDATION_RESPONSE_SCHEMA}"
            )),
            BaseMessage::user(body),
        ])
    }
}

#[async_trait]
impl DreamConsolidator for LlmConsolidator {
    async fn consolidate(&self, job: &DreamJob, ctx: &CallOptions) -> Result<DreamOutcome> {
        let plan = self.plan(job, ctx).await?;
        let metrics = ctx.run(self.sink.apply(job, &plan, ctx)).await??;
        let mut outcome = if plan.is_empty() {
            DreamOutcome::skipped(job.id.clone(), "no consolidation actions")
        } else {
            DreamOutcome::succeeded(job.id.clone(), metrics)
        };
        outcome.notes.extend(plan.notes());
        Ok(outcome)
    }
}

/// Serialises the model input: job identity, scope, and the bounded candidate list.
fn render_request(
    job: &DreamJob,
    candidates: &[DreamCandidate],
    max_content_chars: usize,
) -> Result<String> {
    #[derive(Serialize)]
    struct Body<'a> {
        job_id: &'a str,
        kind: &'a str,
        scope: &'a MemoryScope,
        max_actions: usize,
        candidates: Vec<BodyCandidate<'a>>,
    }
    #[derive(Serialize)]
    struct BodyCandidate<'a> {
        id: &'a str,
        text: String,
        kind: MemoryType,
        importance: f32,
        timestamp: DateTime<Utc>,
        access_count: u64,
    }
    let body = Body {
        job_id: &job.id,
        kind: job.kind.as_str(),
        scope: &job.scope,
        max_actions: candidates.len(),
        candidates: candidates
            .iter()
            .map(|candidate| BodyCandidate {
                id: &candidate.id,
                text: bounded_text(&candidate.content, max_content_chars),
                kind: candidate.kind,
                importance: candidate.importance,
                timestamp: candidate.timestamp,
                access_count: candidate.access_count,
            })
            .collect(),
    };
    serde_json::to_string(&body).map_err(|error| {
        Error::external(std::io::Error::other(format!(
            "could not encode the consolidation request: {error}"
        )))
    })
}

/// Truncates `text` to at most `max_chars` characters, marking a cut with `…`.
fn bounded_text(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let kept: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{kept}…")
    } else {
        kept
    }
}
