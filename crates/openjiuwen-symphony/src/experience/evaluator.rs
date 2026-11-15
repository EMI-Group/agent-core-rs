//! Trace grading: deterministic quality signals plus an optional model-judged pass.
//!
//! Python's `experience/evaluator.py` asks one question — "were the skills this trace used
//! chosen correctly?" — and answers it with a model prompt, or conservatively marks the
//! trace `unevaluated` when no model is available. The Rust port keeps the question but
//! splits the answer in two:
//!
//! 1. A **deterministic rubric** that grades the evidence the trace itself carries
//!    (completion, tool failures, step count, repetition). It always runs, needs no model,
//!    and is fully reproducible offline.
//! 2. An **optional judged pass** over the crate's existing
//!    [`LlmJudgeMetric`] seam, which blends a model verdict
//!    into the deterministic score and records exactly where that verdict came from.
//!
//! # The deterministic rubric
//!
//! Every trace starts at `1.0` and loses documented penalties; the final score is clamped
//! to `[0, 1]`:
//!
//! | Signal | Condition | Penalty |
//! | --- | --- | --- |
//! | [`QualitySignal::Failed`](crate::experience::evaluator::QualitySignal) | the trace recorded a failure | `-0.50` |
//! | [`QualitySignal::EmptyOutput`](crate::experience::evaluator::QualitySignal) | the final output is blank | `-0.35` |
//! | [`QualitySignal::Completed`](crate::experience::evaluator::QualitySignal) | the final output is non-blank | `0.00` |
//! | [`QualitySignal::ToolErrors`](crate::experience::evaluator::QualitySignal) | at least one recorded error | `-0.30 × rate`, `rate = errors / max(calls, 1)` |
//! | [`QualitySignal::ExcessiveSteps`](crate::experience::evaluator::QualitySignal) | more tool calls than `max_steps` | `-0.15` |
//! | [`QualitySignal::RepeatedToolCalls`](crate::experience::evaluator::QualitySignal) | one tool name called `≥ 3` times | `-0.10` each, capped at `-0.20` |
//! | [`QualitySignal::RepeatedQueryTokens`](crate::experience::evaluator::QualitySignal) | one query token repeated `≥ 3` times | `-0.05` |
//!
//! `errors` counts the trace's recorded error messages plus the tool calls whose recorded
//! result is empty or carries an error marker (`error`, `failed`, `exception`, `traceback`,
//! `panic`); a tool call with no recorded result at all is *absence of evidence* and is not
//! counted. The most repeated query token is reported, ties broken by the lexicographically
//! smallest token, so the signal list never depends on hash order.
//!
//! A grade's `success` is the pass verdict: the trace recorded success **and** the final
//! score reached [`PASS_THRESHOLD`]. The grade's `score` is the quality measure, so a trace
//! that technically succeeded but produced an empty answer is unsuccessful here.
//!
//! # The judged pass
//!
//! The pass is opt-in ([`EvaluationConfig::model_judgement`]) and only runs when a
//! [`LlmJudgeMetric`] was attached with
//! [`TraceEvaluator::with_judge`](crate::experience::evaluator::TraceEvaluator::with_judge).
//! For each in-window trace it builds a synthetic
//! [`CapabilityDescriptor`] describing the query, the
//! final-output excerpt, the skills, and a tool-call summary, wraps it in an
//! [`EvaluationInput`] with a one-case
//! [`EvaluationTrace`], and calls the metric.
//!
//! The blend rule is explicit:
//!
//! - a verdict that actually came from the model (`JudgementProvenance::Model`) moves the
//!   score to `(1 - w) × deterministic + w × model`, clamped to `[0, 1]`, where `w` is
//!   [`EvaluationConfig::model_weight`];
//! - a judge with no model, or a judgement that fell back to the metric's static rubric,
//!   leaves the deterministic score untouched and only records the provenance — a fallback
//!   verdict is not evidence about the trace, so it must not move the grade.
//!
//! With [`FallbackMode::Error`](crate::llm::FallbackMode) a failed judgement propagates as
//! the metric's typed [`crate::error::Error`] instead of degrading.
//!
//! # Bounds
//!
//! [`EvaluationConfig::max_traces`] windows the input (sorted by trace id, so the window is
//! reproducible; `0` means unlimited) and [`EvaluationConfig::output_excerpt_chars`] bounds
//! every excerpt on a character boundary. Nothing here reads a clock, spawns a task, or
//! panics on user input.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::evaluation::{EvaluationCase, EvaluationInput, EvaluationTrace, Metric, MetricScore};
use crate::llm::LlmJudgeMetric;
use crate::model::{CapabilityDescriptor, CapabilityIo};
use crate::search::tokenize;

use super::{ClusteredQuery, TraceRecord, TraceToolCall};

/// The default number of tool calls a trace may make before the step limit penalty.
pub const DEFAULT_MAX_STEPS: usize = 20;

/// The default number of traces one grading pass inspects (`0` = unlimited).
pub const DEFAULT_MAX_TRACES: usize = 100;

/// The default number of characters kept in an output excerpt.
pub const DEFAULT_OUTPUT_EXCERPT_CHARS: usize = 400;

/// The score a graded trace needs to count as successful.
pub const PASS_THRESHOLD: f64 = 0.6;

/// The default weight of a model verdict in the blended score.
pub const DEFAULT_MODEL_WEIGHT: f64 = 0.5;

/// How often a tool name or query token must repeat to raise a signal.
pub const REPEAT_THRESHOLD: usize = 3;

/// The penalty for a trace that recorded a failure.
pub const FAILED_PENALTY: f64 = 0.5;

/// The penalty for an empty final output.
pub const EMPTY_OUTPUT_PENALTY: f64 = 0.35;

/// The maximum penalty for recorded tool errors (scaled by the error rate).
pub const TOOL_ERROR_PENALTY: f64 = 0.3;

/// The penalty for exceeding the configured step limit.
pub const EXCESSIVE_STEPS_PENALTY: f64 = 0.15;

/// The penalty for each tool name repeated at least [`REPEAT_THRESHOLD`] times.
pub const REPEATED_TOOL_PENALTY: f64 = 0.1;

/// The cap on the total repeated-tool-call penalty.
pub const REPEATED_TOOL_PENALTY_CAP: f64 = 0.2;

/// The penalty for a query token repeated at least [`REPEAT_THRESHOLD`] times.
pub const REPEATED_QUERY_PENALTY: f64 = 0.05;

/// The substrings that mark a recorded tool result as an error.
pub const ERROR_MARKERS: &[&str] = &["error", "failed", "exception", "traceback", "panic"];

/// The detail prefix [`LlmJudgeMetric`](crate::llm::LlmJudgeMetric) uses for its static
/// rubric fallback.
const FALLBACK_DETAIL_PREFIX: &str = "deterministic_fallback:";

/// The grading configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationConfig {
    /// The number of tool calls a trace may make before
    /// [`QualitySignal::ExcessiveSteps`](crate::experience::evaluator::QualitySignal) fires.
    pub max_steps: usize,
    /// The number of traces one grading pass inspects, sorted by trace id; `0` = unlimited.
    pub max_traces: usize,
    /// The number of characters kept in an output or query excerpt.
    pub output_excerpt_chars: usize,
    /// Whether the attached judge's model verdict participates in the score.
    pub model_judgement: bool,
    /// The weight of a model verdict in the blended score, clamped to `[0, 1]`.
    pub model_weight: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_traces: DEFAULT_MAX_TRACES,
            output_excerpt_chars: DEFAULT_OUTPUT_EXCERPT_CHARS,
            model_judgement: false,
            model_weight: DEFAULT_MODEL_WEIGHT,
        }
    }
}

impl EvaluationConfig {
    /// The default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the step limit.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets the trace window (`0` = unlimited).
    #[must_use]
    pub fn with_max_traces(mut self, max_traces: usize) -> Self {
        self.max_traces = max_traces;
        self
    }

    /// Sets the excerpt length in characters.
    #[must_use]
    pub fn with_output_excerpt_chars(mut self, output_excerpt_chars: usize) -> Self {
        self.output_excerpt_chars = output_excerpt_chars;
        self
    }

    /// Turns the judged pass on or off.
    #[must_use]
    pub fn with_model_judgement(mut self, model_judgement: bool) -> Self {
        self.model_judgement = model_judgement;
        self
    }

    /// Sets the model weight, clamped to `[0, 1]`.
    #[must_use]
    pub fn with_model_weight(mut self, model_weight: f64) -> Self {
        self.model_weight = clamp_unit(model_weight);
        self
    }

    /// The model weight, clamped to `[0, 1]` even after a caller-built struct literal.
    #[must_use]
    pub fn effective_model_weight(&self) -> f64 {
        clamp_unit(self.model_weight)
    }
}

/// One deterministic observation about a trace.
///
/// The variants are the rubric's vocabulary: every penalty the grade applies is named by a
/// signal, and a clean trace reports [`Self::Completed`] alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "signal", rename_all = "snake_case")]
pub enum QualitySignal {
    /// The trace recorded a failure, with the recorded error messages.
    Failed {
        /// The trace's recorded error messages, in order.
        errors: Vec<String>,
    },
    /// The trace produced a non-blank final output.
    Completed,
    /// The trace produced no final output.
    EmptyOutput,
    /// The trace recorded errors (messages plus failing tool calls).
    ToolErrors {
        /// The number of error signals: recorded messages plus failing tool calls.
        errors: usize,
        /// The number of tool calls, floored at `1` for the rate computation.
        calls: usize,
    },
    /// The trace made more tool calls than the configured step limit.
    ExcessiveSteps {
        /// The number of tool calls the trace made.
        steps: usize,
        /// The configured limit that `steps` exceeded.
        limit: usize,
    },
    /// One tool name was called at least [`REPEAT_THRESHOLD`] times.
    RepeatedToolCalls {
        /// The repeated tool name.
        name: String,
        /// How often it was called.
        count: usize,
    },
    /// One query token repeats at least [`REPEAT_THRESHOLD`] times.
    RepeatedQueryTokens {
        /// The repeated token, lowercased.
        token: String,
        /// How often it occurs.
        count: usize,
    },
}

impl QualitySignal {
    /// The stable `snake_case` label used in
    /// [`EvaluationSummary::signals`](crate::experience::evaluator::EvaluationSummary).
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Failed { .. } => "failed",
            Self::Completed => "completed",
            Self::EmptyOutput => "empty_output",
            Self::ToolErrors { .. } => "tool_errors",
            Self::ExcessiveSteps { .. } => "excessive_steps",
            Self::RepeatedToolCalls { .. } => "repeated_tool_calls",
            Self::RepeatedQueryTokens { .. } => "repeated_query_tokens",
        }
    }

    /// The score penalty this signal contributes to the deterministic rubric.
    // The ratio is a small count (errors over calls), so the `f64` mantissa is ample.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn penalty(&self) -> f64 {
        match self {
            Self::Failed { .. } => FAILED_PENALTY,
            Self::Completed => 0.0,
            Self::EmptyOutput => EMPTY_OUTPUT_PENALTY,
            Self::ToolErrors { errors, calls } => {
                let rate = *errors as f64 / (*calls).max(1) as f64;
                TOOL_ERROR_PENALTY * clamp_unit(rate)
            }
            Self::ExcessiveSteps { .. } => EXCESSIVE_STEPS_PENALTY,
            Self::RepeatedToolCalls { .. } => REPEATED_TOOL_PENALTY,
            Self::RepeatedQueryTokens { .. } => REPEATED_QUERY_PENALTY,
        }
    }
}

/// Where a grade's final score came from.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "judged_by", rename_all = "snake_case")]
pub enum JudgementProvenance {
    /// The grade is the deterministic rubric's; no model verdict was used.
    #[default]
    Deterministic,
    /// The grade blended a verdict the model produced.
    Model,
    /// The model was asked but fell back to the static rubric; the score stayed
    /// deterministic and `reason` records why.
    Fallback {
        /// Why the model verdict was not usable.
        reason: String,
    },
}

impl JudgementProvenance {
    /// The stable `snake_case` label: `deterministic`, `model`, or `fallback`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Model => "model",
            Self::Fallback { .. } => "fallback",
        }
    }

    /// Whether a model verdict participated in the score.
    #[must_use]
    pub fn is_model(&self) -> bool {
        matches!(self, Self::Model)
    }

    /// The fallback reason, when this provenance is a fallback.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Fallback { reason } => Some(reason),
            Self::Deterministic | Self::Model => None,
        }
    }
}

/// The grade of one trace.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraceGrade {
    /// The graded trace's id.
    pub trace_id: String,
    /// The pass verdict: the trace recorded success and the score reached
    /// [`PASS_THRESHOLD`].
    pub success: bool,
    /// The quality score in `[0, 1]`.
    pub score: f64,
    /// The deterministic signals, in rubric order.
    pub signals: Vec<QualitySignal>,
    /// Where the final score came from.
    pub judged: JudgementProvenance,
}

impl TraceGrade {
    /// The labels of the grade's signals, in order.
    #[must_use]
    pub fn signal_labels(&self) -> Vec<&'static str> {
        self.signals.iter().map(QualitySignal::label).collect()
    }

    /// Whether the grade carries `label`.
    #[must_use]
    pub fn has_signal(&self, label: &str) -> bool {
        self.signals.iter().any(|signal| signal.label() == label)
    }
}

/// The aggregate of one grading pass.
///
/// `signals` counts how many grades carried each signal label (a [`BTreeMap`], so the
/// report ordering never depends on hash order), and `provenance` summarises the judged
/// pass: `Deterministic` when no model verdict was used, `Model` when every trace was
/// judged by the model, and `Fallback` otherwise — naming the first fallback's reason or
/// how many traces the model judged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationSummary {
    /// The number of graded traces.
    pub traces: usize,
    /// The mean score; `0.0` when no trace was graded.
    pub mean_score: f64,
    /// The share of grades that passed; `0.0` when no trace was graded.
    pub success_rate: f64,
    /// The number of grades that did not pass.
    pub failed: usize,
    /// How many grades carried each signal label.
    pub signals: BTreeMap<String, usize>,
    /// How many grades blended a model verdict.
    pub judged_by_model: usize,
    /// The pass-wide judgement provenance.
    pub provenance: JudgementProvenance,
}

impl EvaluationSummary {
    /// Whether the pass graded no trace.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.traces == 0
    }
}

/// Grades recorded traces with the deterministic rubric and an optional judged pass.
#[derive(Debug, Clone, Default)]
pub struct TraceEvaluator {
    /// The grading configuration.
    config: EvaluationConfig,
    /// The optional judge metric.
    judge: Option<LlmJudgeMetric>,
}

impl TraceEvaluator {
    /// An evaluator with the default configuration and no judge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An evaluator with `config` and no judge.
    #[must_use]
    pub fn with_config(config: EvaluationConfig) -> Self {
        Self {
            config,
            judge: None,
        }
    }

    /// Attaches the judge metric used by the opt-in judged pass.
    #[must_use]
    pub fn with_judge(mut self, judge: LlmJudgeMetric) -> Self {
        self.judge = Some(judge);
        self
    }

    /// The grading configuration.
    #[must_use]
    pub fn config(&self) -> &EvaluationConfig {
        &self.config
    }

    /// The attached judge metric.
    #[must_use]
    pub fn judge(&self) -> Option<&LlmJudgeMetric> {
        self.judge.as_ref()
    }

    /// Whether a judge is attached.
    #[must_use]
    pub fn has_judge(&self) -> bool {
        self.judge.is_some()
    }

    /// Whether the judged pass runs: it needs both the config toggle and a judge.
    #[must_use]
    pub fn judges(&self) -> bool {
        self.config.model_judgement && self.judge.is_some()
    }

    /// Truncates `text` to the configured excerpt length on a character boundary.
    ///
    /// A truncated excerpt ends in `...`, matching Python's `_truncate`, and the total
    /// length stays at or below the configured limit. A limit of `0` yields an empty
    /// string.
    #[must_use]
    pub fn excerpt(&self, text: &str) -> String {
        let limit = self.config.output_excerpt_chars;
        let trimmed = text.trim();
        if limit == 0 {
            return String::new();
        }
        if trimmed.chars().count() <= limit {
            return trimmed.to_owned();
        }
        let mut excerpt: String = trimmed.chars().take(limit.saturating_sub(3)).collect();
        excerpt.push_str("...");
        excerpt
    }

    /// Grades one trace with the deterministic rubric; no model is consulted.
    #[must_use]
    pub fn grade_record(&self, record: &TraceRecord) -> TraceGrade {
        let mut signals = Vec::new();
        let mut score = 1.0_f64;
        if !record.success {
            signals.push(QualitySignal::Failed {
                errors: record.errors.clone(),
            });
            score -= FAILED_PENALTY;
        }
        if record.final_output.trim().is_empty() {
            signals.push(QualitySignal::EmptyOutput);
            score -= EMPTY_OUTPUT_PENALTY;
        } else {
            signals.push(QualitySignal::Completed);
        }
        let calls = record.tool_calls.len();
        let failed_calls = record
            .tool_calls
            .iter()
            .filter(|call| tool_call_failed(call))
            .count();
        let errors = record.errors.len() + failed_calls;
        if errors > 0 {
            let signal = QualitySignal::ToolErrors {
                errors,
                calls: calls.max(1),
            };
            score -= signal.penalty();
            signals.push(signal);
        }
        if calls > self.config.max_steps {
            let signal = QualitySignal::ExcessiveSteps {
                steps: calls,
                limit: self.config.max_steps,
            };
            score -= signal.penalty();
            signals.push(signal);
        }
        let mut repeated = 0.0_f64;
        let mut tool_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for call in &record.tool_calls {
            if !call.name.is_empty() {
                *tool_counts.entry(call.name.as_str()).or_insert(0) += 1;
            }
        }
        for (name, count) in tool_counts {
            if count < REPEAT_THRESHOLD {
                continue;
            }
            let signal = QualitySignal::RepeatedToolCalls {
                name: name.to_owned(),
                count,
            };
            repeated += signal.penalty();
            signals.push(signal);
        }
        score -= repeated.min(REPEATED_TOOL_PENALTY_CAP);
        if let Some((token, count)) = most_repeated_query_token(&record.query) {
            let signal = QualitySignal::RepeatedQueryTokens { token, count };
            score -= signal.penalty();
            signals.push(signal);
        }
        let score = clamp_unit(score);
        TraceGrade {
            trace_id: record.trace_id.clone(),
            success: record.success && score >= PASS_THRESHOLD,
            score,
            signals,
            judged: JudgementProvenance::Deterministic,
        }
    }

    /// Grades the traces in the configured window, running the judged pass when enabled.
    ///
    /// The window is the first [`EvaluationConfig::max_traces`] records after sorting by
    /// trace id (`0` = all of them), so the same input always yields the same grades in the
    /// same order.
    ///
    /// # Errors
    ///
    /// Propagates the judge metric's error; with
    /// [`FallbackMode::Error`](crate::llm::FallbackMode) a failed judgement surfaces here
    /// as a typed [`crate::error::Error`].
    pub async fn grade_traces(&self, records: &[TraceRecord]) -> Result<Vec<TraceGrade>> {
        let mut grades = Vec::with_capacity(records.len());
        for record in self.windowed(records) {
            let mut grade = self.grade_record(record);
            if self.judges() {
                self.apply_judgement(record, &mut grade).await?;
            }
            grades.push(grade);
        }
        Ok(grades)
    }

    /// Aggregates `grades` into a summary.
    ///
    /// Scores are summed in the given order and every ratio guards division by zero, so an
    /// empty pass reports zeros instead of `NaN`.
    // The ratios divide small trace counts, so the `f64` mantissa is ample.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn summarise_grades(&self, grades: &[TraceGrade]) -> EvaluationSummary {
        let traces = grades.len();
        let mut total = 0.0_f64;
        let mut passed = 0_usize;
        let mut signals: BTreeMap<String, usize> = BTreeMap::new();
        let mut judged_by_model = 0_usize;
        let mut fallback_reason: Option<String> = None;
        for grade in grades {
            total += grade.score;
            if grade.success {
                passed += 1;
            }
            for signal in &grade.signals {
                *signals.entry(signal.label().to_owned()).or_insert(0) += 1;
            }
            match &grade.judged {
                JudgementProvenance::Model => judged_by_model += 1,
                JudgementProvenance::Fallback { reason } => {
                    if fallback_reason.is_none() {
                        fallback_reason = Some(reason.clone());
                    }
                }
                JudgementProvenance::Deterministic => {}
            }
        }
        let provenance = if let Some(reason) = fallback_reason {
            JudgementProvenance::Fallback { reason }
        } else if judged_by_model > 0 && judged_by_model == traces {
            JudgementProvenance::Model
        } else if judged_by_model > 0 {
            JudgementProvenance::Fallback {
                reason: format!("{judged_by_model} of {traces} graded traces used a model verdict"),
            }
        } else {
            JudgementProvenance::Deterministic
        };
        EvaluationSummary {
            traces,
            mean_score: clamp_unit(if traces == 0 {
                0.0
            } else {
                total / traces as f64
            }),
            success_rate: clamp_unit(if traces == 0 {
                0.0
            } else {
                passed as f64 / traces as f64
            }),
            failed: traces - passed,
            signals,
            judged_by_model,
            provenance,
        }
    }

    /// Grades the in-window traces and summarises them.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Self::grade_traces`].
    pub async fn summarise(&self, records: &[TraceRecord]) -> Result<EvaluationSummary> {
        let grades = self.grade_traces(records).await?;
        Ok(self.summarise_grades(&grades))
    }

    /// Grades distilled clusters by synthesising one trace per cluster.
    ///
    /// A cluster carries only positive evidence — a pattern, its member trace ids, the
    /// skills, and a success count — because the collector distils successful runs. The
    /// synthetic record therefore takes the pattern as both query and completed output and
    /// reports success when [`ClusteredQuery::success_count`] is above zero. Grade the
    /// underlying [`TraceRecord`]s when failure evidence matters.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Self::grade_traces`].
    pub async fn grade_clusters(&self, clusters: &[ClusteredQuery]) -> Result<Vec<TraceGrade>> {
        let records: Vec<TraceRecord> = clusters
            .iter()
            .enumerate()
            .map(|(index, cluster)| cluster_record(cluster, index))
            .collect();
        self.grade_traces(&records).await
    }

    /// The in-window records, sorted by trace id.
    fn windowed<'a>(&self, records: &'a [TraceRecord]) -> Vec<&'a TraceRecord> {
        let mut windowed: Vec<&TraceRecord> = records.iter().collect();
        windowed.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));
        if self.config.max_traces > 0 {
            windowed.truncate(self.config.max_traces);
        }
        windowed
    }

    /// Runs the judged pass for `record` and folds a model verdict into `grade`.
    async fn apply_judgement(&self, record: &TraceRecord, grade: &mut TraceGrade) -> Result<()> {
        let Some(judge) = &self.judge else {
            return Ok(());
        };
        let verdict = judge.evaluate(&self.judgement_input(record)).await?;
        let provenance = provenance_of(judge, &verdict);
        if let (JudgementProvenance::Model, Some(model_score)) =
            (&provenance, verdict.effective_score())
        {
            let weight = self.config.effective_model_weight();
            grade.score =
                clamp_unit((1.0 - weight) * grade.score + weight * clamp_unit(model_score));
            grade.success = record.success && grade.score >= PASS_THRESHOLD;
        }
        grade.judged = provenance;
        Ok(())
    }

    /// The synthetic evaluation input a judge sees for `record`.
    fn judgement_input(&self, record: &TraceRecord) -> EvaluationInput {
        let descriptor_id = if record.trace_id.trim().is_empty() {
            "trace".to_owned()
        } else {
            record.trace_id.clone()
        };
        let capability = CapabilityDescriptor::new(
            "trace",
            descriptor_id,
            format!("Experience trace {}", self.excerpt(&record.trace_id)),
        )
        .with_description(self.describe(record))
        .with_inputs(vec![CapabilityIo::new("query", "string")
            .with_description("the user query the trace answered")])
        .with_outputs(vec![CapabilityIo::new("output", "string")
            .with_description("the final assistant text of the trace, excerpted")]);
        let case = EvaluationCase::new(record.trace_id.clone(), record.query.clone())
            .with_actual_answer(self.excerpt(&record.final_output));
        EvaluationInput::new(capability)
            .with_target("experience_trace")
            .with_trace(EvaluationTrace::from_cases(vec![case]))
    }

    /// The one-line trace description a judge reads, bounded by the excerpt length.
    fn describe(&self, record: &TraceRecord) -> String {
        let query = self.excerpt(&record.query);
        let output = self.excerpt(&record.final_output);
        let skills = if record.skills.is_empty() {
            "(none)".to_owned()
        } else {
            self.excerpt(&record.skills.join(", "))
        };
        let mut tool_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for call in &record.tool_calls {
            let name = if call.name.is_empty() {
                "(unnamed)"
            } else {
                call.name.as_str()
            };
            *tool_counts.entry(name).or_insert(0) += 1;
        }
        let tools = if tool_counts.is_empty() {
            "(none)".to_owned()
        } else {
            let rendered: Vec<String> = tool_counts
                .into_iter()
                .map(|(name, count)| format!("{name} x{count}"))
                .collect();
            self.excerpt(&rendered.join(", "))
        };
        format!(
            "Trace `{}` answered the query `{query}` using the skills {skills} and the tool \
             calls {tools}; the final output was: {output}",
            self.excerpt(&record.trace_id)
        )
    }
}

/// Derives the provenance of one judge verdict.
fn provenance_of(judge: &LlmJudgeMetric, verdict: &MetricScore) -> JudgementProvenance {
    if !judge.has_model() {
        return JudgementProvenance::Deterministic;
    }
    match verdict.detail.as_deref() {
        Some(detail) if detail.starts_with(FALLBACK_DETAIL_PREFIX) => {
            JudgementProvenance::Fallback {
                reason: detail
                    .trim_start_matches(FALLBACK_DETAIL_PREFIX)
                    .trim()
                    .to_owned(),
            }
        }
        _ => JudgementProvenance::Model,
    }
}

/// Synthesises the trace a [`ClusteredQuery`] stands for.
fn cluster_record(cluster: &ClusteredQuery, index: usize) -> TraceRecord {
    let trace_id = cluster
        .members
        .first()
        .cloned()
        .unwrap_or_else(|| format!("cluster:{index}"));
    TraceRecord {
        trace_id,
        query: cluster.pattern.clone(),
        final_output: cluster.pattern.clone(),
        skills: cluster.skill_ids.clone(),
        success: cluster.success_count > 0,
        ..TraceRecord::default()
    }
}

/// Whether a tool call's recorded result names a failure.
///
/// A call with no recorded result is absence of evidence, not a failure: the parser only
/// attaches a result when a matching `tool_result` event arrived.
fn tool_call_failed(call: &TraceToolCall) -> bool {
    let Some(result) = call.result.as_deref() else {
        return false;
    };
    let trimmed = result.trim();
    trimmed.is_empty() || contains_error_marker(trimmed)
}

/// Whether `text` contains one of [`ERROR_MARKERS`], case-insensitively.
fn contains_error_marker(text: &str) -> bool {
    let lowered = text.to_lowercase();
    ERROR_MARKERS.iter().any(|marker| lowered.contains(marker))
}

/// The most repeated query token at or above [`REPEAT_THRESHOLD`], ties broken on the
/// lexicographically smallest token.
fn most_repeated_query_token(query: &str) -> Option<(String, usize)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for token in tokenize(query) {
        *counts.entry(token).or_insert(0) += 1;
    }
    let mut best: Option<(String, usize)> = None;
    for (token, count) in counts {
        if count < REPEAT_THRESHOLD {
            continue;
        }
        let replace = match &best {
            None => true,
            Some((_, best_count)) => count > *best_count,
        };
        if replace {
            best = Some((token, count));
        }
    }
    best
}

/// Clamps a score to `[0, 1]`, mapping a non-finite value to `0.0`.
fn clamp_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::llm::{FallbackMode, StaticCompletionModel};
    use std::sync::Arc;

    /// A trace record with `trace_id`, a query, and a final output.
    fn record(trace_id: &str, query: &str, output: &str) -> TraceRecord {
        TraceRecord {
            trace_id: trace_id.to_owned(),
            session_id: "session".to_owned(),
            request_id: trace_id.to_owned(),
            query: query.to_owned(),
            final_output: output.to_owned(),
            skills: vec!["summarizer".to_owned()],
            success: true,
            ..TraceRecord::default()
        }
    }

    /// A tool call named `name` with an optional recorded result.
    fn call(name: &str, result: Option<&str>) -> TraceToolCall {
        TraceToolCall {
            name: name.to_owned(),
            arguments: serde_json::Value::Null,
            result: result.map(str::to_owned),
        }
    }

    /// The lone grade of `record` under the default evaluator.
    fn graded(record: &TraceRecord) -> TraceGrade {
        TraceEvaluator::new().grade_record(record)
    }

    #[test]
    fn a_clean_trace_scores_one_and_completes() {
        let grade = graded(&record("s/1", "summarize", "the summary"));
        assert_eq!(grade.score, 1.0);
        assert!(grade.success);
        assert_eq!(grade.signals, vec![QualitySignal::Completed]);
        assert_eq!(grade.signal_labels(), vec!["completed"]);
        assert!(grade.has_signal("completed"));
        assert!(!grade.has_signal("failed"));
        assert_eq!(grade.trace_id, "s/1");
        assert_eq!(grade.judged, JudgementProvenance::Deterministic);
    }

    #[test]
    fn an_empty_output_is_penalised_and_fails() {
        let grade = graded(&record("s/1", "summarize", "   "));
        assert_eq!(grade.score, 0.65);
        assert!(grade.success, "0.65 still clears the pass threshold");
        assert_eq!(grade.signals, vec![QualitySignal::EmptyOutput]);
        assert_eq!(grade.signals[0].penalty(), EMPTY_OUTPUT_PENALTY);
    }

    #[test]
    fn a_failed_trace_loses_the_failure_penalty() {
        let mut record = record("s/1", "summarize", "partial");
        record.success = false;
        record.errors = vec!["tool exploded".to_owned()];
        let grade = graded(&record);
        // 1.0 - 0.5 (Failed) - 0.3 (one error over max(0, 1) calls) = 0.2
        assert_eq!(grade.score, 0.2);
        assert!(!grade.success);
        assert_eq!(
            grade.signals[0],
            QualitySignal::Failed {
                errors: vec!["tool exploded".to_owned()]
            }
        );
        assert_eq!(grade.signals[0].penalty(), FAILED_PENALTY);
        assert!(grade.has_signal("tool_errors"));
    }

    #[test]
    fn tool_errors_scale_with_the_error_rate() {
        let mut record = record("s/1", "summarize", "done");
        record.tool_calls = vec![
            call("search", Some("Error: the index is unavailable")),
            call("read", Some("ok")),
        ];
        record.errors = vec!["retry exhausted".to_owned()];
        let grade = graded(&record);
        // Errors: one message + one failing call = 2; calls = 2 → rate 1.0 → -0.3.
        assert_eq!(grade.score, 0.7);
        let tool_errors = grade
            .signals
            .iter()
            .find_map(|signal| match signal {
                QualitySignal::ToolErrors { errors, calls } => Some((*errors, *calls)),
                _ => None,
            })
            .expect("tool error signal");
        assert_eq!(tool_errors, (2, 2));

        let mut quiet = record.clone();
        quiet.tool_calls = vec![call("search", None), call("read", Some("   "))];
        quiet.errors.clear();
        let grade = graded(&quiet);
        // One empty result over two calls → rate 0.5 → -0.15.
        assert!((grade.score - 0.85).abs() < 1e-9);
    }

    #[test]
    fn excessive_steps_are_bounded_by_the_configuration() {
        let evaluator = TraceEvaluator::with_config(EvaluationConfig::new().with_max_steps(2));
        let mut record = record("s/1", "summarize", "done");
        record.tool_calls = vec![
            call("a", Some("ok")),
            call("b", Some("ok")),
            call("c", Some("ok")),
        ];
        let grade = evaluator.grade_record(&record);
        assert!((grade.score - 0.85).abs() < 1e-9);
        assert!(grade.has_signal("excessive_steps"));
        let excessive = grade
            .signals
            .iter()
            .find_map(|signal| match signal {
                QualitySignal::ExcessiveSteps { steps, limit } => Some((*steps, *limit)),
                _ => None,
            })
            .expect("step signal");
        assert_eq!(excessive, (3, 2));
    }

    #[test]
    fn repeated_tool_calls_are_capped() {
        let mut record = record("s/1", "summarize", "done");
        record.tool_calls = vec![
            call("search", Some("ok")),
            call("search", Some("ok")),
            call("search", Some("ok")),
            call("read", Some("ok")),
            call("read", Some("ok")),
            call("read", Some("ok")),
        ];
        let grade = graded(&record);
        // Two repeated names would be -0.2, exactly the cap.
        assert!((grade.score - 0.8).abs() < 1e-9);
        let repeated: Vec<&str> = grade
            .signals
            .iter()
            .filter_map(|signal| match signal {
                QualitySignal::RepeatedToolCalls { name, count } => {
                    assert_eq!(*count, 3);
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(repeated, vec!["read", "search"]);
    }

    #[test]
    fn repeated_query_tokens_are_reported_once() {
        let mut record = record("s/1", "summarize summarize summarize the report", "done");
        record.tool_calls = Vec::new();
        let grade = graded(&record);
        assert!((grade.score - 0.95).abs() < 1e-9);
        assert_eq!(
            grade.signals.last(),
            Some(&QualitySignal::RepeatedQueryTokens {
                token: "summarize".to_owned(),
                count: 3
            })
        );
        // Ties break on the smallest token.
        let mut tied = record.clone();
        tied.query = "beta beta beta alpha alpha alpha".to_owned();
        let grade = graded(&tied);
        assert_eq!(
            grade.signals.last(),
            Some(&QualitySignal::RepeatedQueryTokens {
                token: "alpha".to_owned(),
                count: 3
            })
        );
    }

    #[test]
    fn the_worst_trace_clamps_to_zero() {
        let mut record = record("s/1", "a a a", "");
        record.success = false;
        record.errors = vec!["boom".to_owned()];
        let evaluator = TraceEvaluator::with_config(EvaluationConfig::new().with_max_steps(0));
        record.tool_calls = vec![call("x", Some("error"))];
        let grade = evaluator.grade_record(&record);
        assert_eq!(grade.score, 0.0);
        assert!(!grade.success);
        for signal in &grade.signals {
            assert!(signal.penalty() <= 1.0);
        }
    }

    #[test]
    fn summary_arithmetic_and_ordering_are_deterministic() {
        let evaluator = TraceEvaluator::new();
        let grades = vec![
            TraceGrade {
                trace_id: "s/1".to_owned(),
                success: true,
                score: 1.0,
                signals: vec![QualitySignal::Completed],
                judged: JudgementProvenance::Deterministic,
            },
            TraceGrade {
                trace_id: "s/2".to_owned(),
                success: false,
                score: 0.5,
                signals: vec![QualitySignal::EmptyOutput],
                judged: JudgementProvenance::Deterministic,
            },
            TraceGrade {
                trace_id: "s/3".to_owned(),
                success: false,
                score: 0.0,
                signals: vec![
                    QualitySignal::EmptyOutput,
                    QualitySignal::Failed { errors: vec![] },
                ],
                judged: JudgementProvenance::Deterministic,
            },
        ];
        let summary = evaluator.summarise_grades(&grades);
        assert_eq!(summary.traces, 3);
        assert!((summary.mean_score - 0.5).abs() < 1e-9);
        assert!((summary.success_rate - (1.0 / 3.0)).abs() < 1e-9);
        assert_eq!(summary.failed, 2);
        assert_eq!(
            summary.signals.keys().cloned().collect::<Vec<String>>(),
            vec!["completed", "empty_output", "failed"]
        );
        assert_eq!(summary.signals["empty_output"], 2);
        assert_eq!(summary.signals["completed"], 1);
        assert_eq!(summary.judged_by_model, 0);
        assert_eq!(summary.provenance, JudgementProvenance::Deterministic);
        assert!(!summary.is_empty());

        let empty = evaluator.summarise_grades(&[]);
        assert_eq!(empty.traces, 0);
        assert_eq!(empty.mean_score, 0.0);
        assert_eq!(empty.success_rate, 0.0);
        assert_eq!(empty.failed, 0);
        assert!(empty.is_empty());
        assert_eq!(empty.provenance, JudgementProvenance::Deterministic);
    }

    #[tokio::test]
    async fn grading_windows_the_input_by_trace_id() {
        let evaluator = TraceEvaluator::with_config(EvaluationConfig::new().with_max_traces(2));
        let records = vec![
            record("s/3", "third", "done"),
            record("s/1", "first", "done"),
            record("s/2", "second", "done"),
        ];
        let grades = evaluator.grade_traces(&records).await.expect("grade");
        let ids: Vec<&str> = grades.iter().map(|grade| grade.trace_id.as_str()).collect();
        assert_eq!(ids, vec!["s/1", "s/2"]);

        let unlimited = TraceEvaluator::with_config(EvaluationConfig::new().with_max_traces(0));
        assert_eq!(
            unlimited.grade_traces(&records).await.expect("grade").len(),
            3
        );
    }

    #[tokio::test]
    async fn clusters_grade_through_synthesised_traces() {
        let evaluator = TraceEvaluator::new();
        let clusters = vec![
            ClusteredQuery {
                pattern: "summarize the report".to_owned(),
                members: vec!["s/1".to_owned(), "s/2".to_owned()],
                query_examples: vec!["summarize the report".to_owned()],
                skill_ids: vec!["summarizer".to_owned()],
                success_count: 2,
            },
            ClusteredQuery {
                pattern: "translate text".to_owned(),
                members: Vec::new(),
                query_examples: Vec::new(),
                skill_ids: vec!["translator".to_owned()],
                success_count: 0,
            },
        ];
        let grades = evaluator.grade_clusters(&clusters).await.expect("grade");
        assert_eq!(grades.len(), 2);
        // `cluster:1` sorts before `s/1`, so the unnamed cluster comes first.
        assert_eq!(grades[0].trace_id, "cluster:1");
        assert!(!grades[0].success);
        assert_eq!(grades[0].score, 0.5);
        assert!(grades[0].has_signal("failed"));
        assert_eq!(grades[1].trace_id, "s/1");
        assert_eq!(grades[1].score, 1.0);
        assert!(grades[1].success);

        let summary = evaluator.summarise_grades(&grades);
        assert_eq!(summary.traces, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.signals.get("failed").copied(), Some(1));
        assert_eq!(summary.signals.get("completed").copied(), Some(2));
    }

    #[tokio::test]
    async fn the_judged_pass_blends_a_model_verdict() {
        let judge = LlmJudgeMetric::new("skill_correctness", "grade the skill selection")
            .with_model(Arc::new(StaticCompletionModel::single(
                r#"{"score":1,"reason":"the summarizer was the right skill"}"#,
            )));
        let evaluator = TraceEvaluator::with_config(
            EvaluationConfig::new()
                .with_model_judgement(true)
                .with_model_weight(0.5),
        )
        .with_judge(judge);
        assert!(evaluator.judges());
        assert!(evaluator.has_judge());
        assert_eq!(evaluator.judge().map(LlmJudgeMetric::has_model), Some(true));

        let mut record = record("s/1", "summarize", "the summary");
        record.tool_calls = vec![call("search", Some("ok"))];
        let grades = evaluator.grade_traces(&[record]).await.expect("grade");
        assert_eq!(grades.len(), 1);
        assert_eq!(grades[0].judged, JudgementProvenance::Model);
        assert_eq!(grades[0].judged.as_str(), "model");
        assert_eq!(grades[0].score, 1.0);
        let summary = evaluator.summarise_grades(&grades);
        assert_eq!(summary.judged_by_model, 1);
        assert_eq!(summary.provenance, JudgementProvenance::Model);
    }

    #[tokio::test]
    async fn the_judged_pass_respects_the_toggle() {
        let judge = LlmJudgeMetric::new("judge", "rubric").with_model(Arc::new(
            StaticCompletionModel::single(r#"{"score":0,"reason":"wrong skill"}"#),
        ));
        // The toggle is off, so the model is never consulted and the score is unchanged.
        let evaluator = TraceEvaluator::new().with_judge(judge);
        assert!(!evaluator.judges());
        let grades = evaluator
            .grade_traces(&[record("s/1", "summarize", "done")])
            .await
            .expect("grade");
        assert_eq!(grades[0].score, 1.0);
        assert_eq!(grades[0].judged, JudgementProvenance::Deterministic);
    }

    #[tokio::test]
    async fn a_judge_without_a_model_stays_deterministic() {
        let judge = LlmJudgeMetric::new("judge", "rubric");
        let evaluator =
            TraceEvaluator::with_config(EvaluationConfig::new().with_model_judgement(true))
                .with_judge(judge);
        let grades = evaluator
            .grade_traces(&[record("s/1", "summarize", "done")])
            .await
            .expect("grade");
        assert_eq!(grades[0].judged, JudgementProvenance::Deterministic);
        assert_eq!(grades[0].score, 1.0);
    }

    #[tokio::test]
    async fn an_unusable_reply_falls_back_without_moving_the_score() {
        let judge = LlmJudgeMetric::new("judge", "rubric").with_model(Arc::new(
            StaticCompletionModel::new(vec!["not json".to_owned(), "still not json".to_owned()]),
        ));
        let evaluator =
            TraceEvaluator::with_config(EvaluationConfig::new().with_model_judgement(true))
                .with_judge(judge);
        let grades = evaluator
            .grade_traces(&[record("s/1", "summarize", "done")])
            .await
            .expect("grade");
        assert_eq!(grades[0].score, 1.0);
        match &grades[0].judged {
            JudgementProvenance::Fallback { reason } => {
                assert!(reason.contains("rejected the model reply twice"));
                assert!(!reason.starts_with(FALLBACK_DETAIL_PREFIX));
            }
            other => panic!("expected a fallback provenance, got {other:?}"),
        }
        assert_eq!(grades[0].judged.as_str(), "fallback");
        assert!(grades[0].judged.reason().is_some());
        let summary = evaluator.summarise_grades(&grades);
        assert_eq!(summary.judged_by_model, 0);
        assert!(matches!(
            summary.provenance,
            JudgementProvenance::Fallback { .. }
        ));
    }

    #[tokio::test]
    async fn the_error_fallback_mode_surfaces_the_judge_failure() {
        let judge = LlmJudgeMetric::new("judge", "rubric")
            .with_model(Arc::new(StaticCompletionModel::new(vec![
                "not json".to_owned(),
                "still not json".to_owned(),
            ])))
            .with_fallback(FallbackMode::Error);
        let evaluator =
            TraceEvaluator::with_config(EvaluationConfig::new().with_model_judgement(true))
                .with_judge(judge);
        let error = evaluator
            .grade_traces(&[record("s/1", "summarize", "done")])
            .await
            .expect_err("judge failure must surface");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("rejected the model reply twice"));
    }

    #[test]
    fn excerpts_truncate_on_character_boundaries() {
        let evaluator =
            TraceEvaluator::with_config(EvaluationConfig::new().with_output_excerpt_chars(10));
        assert_eq!(evaluator.excerpt("short"), "short");
        assert_eq!(evaluator.excerpt("  padded  "), "padded");
        // Multi-byte characters are never split.
        let excerpt = evaluator.excerpt("日本語のテキストです、もっと長い文章");
        assert_eq!(excerpt.chars().count(), 10);
        assert!(excerpt.ends_with("..."));
        let empty =
            TraceEvaluator::with_config(EvaluationConfig::new().with_output_excerpt_chars(0));
        assert!(empty.excerpt("anything").is_empty());
    }

    #[test]
    fn the_judged_input_describes_the_trace() {
        let evaluator = TraceEvaluator::new();
        let mut record = record("s/1", "summarize the report", "the summary");
        record.tool_calls = vec![call("search", Some("ok")), call("search", None)];
        let input = evaluator.judgement_input(&record);
        assert_eq!(input.capability.capability_type, "trace");
        assert_eq!(input.capability.capability_id, "s/1");
        assert!(input
            .capability
            .description
            .contains("summarize the report"));
        assert!(input.capability.description.contains("summarizer"));
        assert!(input.capability.description.contains("search x2"));
        assert_eq!(input.capability.inputs.len(), 1);
        assert_eq!(input.capability.outputs.len(), 1);
        assert_eq!(input.target.as_deref(), Some("experience_trace"));
        let trace = input.trace.as_ref().expect("trace");
        assert_eq!(trace.cases.len(), 1);
        assert_eq!(trace.cases[0].case_id, "s/1");
        assert_eq!(trace.cases[0].actual_answer.as_deref(), Some("the summary"));

        let blank = TraceRecord::default();
        let input = evaluator.judgement_input(&blank);
        assert_eq!(input.capability.capability_id, "trace");
        assert!(input.capability.description.contains("(none)"));
    }

    #[test]
    fn configuration_clamps_and_round_trips() {
        let config = EvaluationConfig::new()
            .with_model_weight(5.0)
            .with_max_steps(3)
            .with_max_traces(0)
            .with_output_excerpt_chars(12)
            .with_model_judgement(true);
        assert_eq!(config.effective_model_weight(), 1.0);
        assert_eq!(config.max_steps, 3);
        assert_eq!(config.max_traces, 0);
        assert_eq!(config.output_excerpt_chars, 12);
        assert!(config.model_judgement);

        let literal = EvaluationConfig {
            model_weight: -2.0,
            ..EvaluationConfig::default()
        };
        assert_eq!(literal.effective_model_weight(), 0.0);

        let parsed: EvaluationConfig = serde_json::from_str("{}").expect("defaults");
        assert_eq!(parsed, EvaluationConfig::default());
        let json = serde_json::to_value(&config).expect("serialize");
        let parsed: EvaluationConfig = serde_json::from_value(json).expect("round trip");
        assert_eq!(parsed, config);

        let signal = QualitySignal::ToolErrors {
            errors: 3,
            calls: 0,
        };
        assert!((signal.penalty() - TOOL_ERROR_PENALTY).abs() < f64::EPSILON);
        let grade = TraceGrade::default();
        assert!(grade.signal_labels().is_empty());
        assert!(!grade.has_signal("completed"));
        assert_eq!(
            JudgementProvenance::default(),
            JudgementProvenance::Deterministic
        );
        assert!(JudgementProvenance::Model.reason().is_none());
        assert!(JudgementProvenance::Model.is_model());
    }

    #[test]
    fn error_markers_are_case_insensitive() {
        assert!(contains_error_marker("Command FAILED with exit 1"));
        assert!(contains_error_marker("Traceback (most recent call last)"));
        assert!(!contains_error_marker("all good"));
    }
}
