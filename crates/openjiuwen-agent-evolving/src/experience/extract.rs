//! Deriving experience records from signals, trajectories and failing cases.
//!
//! Python's experience optimizer asks a model to draft a lesson from an
//! `EvolutionContext`; that pipeline is deferred. This module keeps the offline
//! half — turning evidence the crate already models into candidate records — so
//! the archive, lifecycle and sharing paths are exercised without a model:
//!
//! - [`ExperienceExtractor::from_signals`] — one record per sufficiently
//!   confident evolution signal, section projected from the signal target.
//! - [`ExperienceExtractor::from_trajectories`] — a failure lesson from the
//!   first error step, and a tool-ordering lesson from a successful run with at
//!   least two tool calls.
//! - [`ExperienceExtractor::from_evaluated_cases`] — a lesson from each case
//!   that scored below the caller's threshold.
//! - [`ExperienceExtractor::merge`] — dedupe within one batch through the
//!   archive's own similarity rule.
//!
//! Record ids are content-derived (FNV-1a over the source and normalised
//! content), so repeated extraction over the same evidence yields the same ids
//! and the archive merges instead of duplicating. Sequence counters stay `0`
//! here; the archive stamps them on insertion.

use std::collections::BTreeSet;

use crate::dataset::EvaluatedCase;
use crate::experience::archive::{tokens, ExperienceArchive};
use crate::experience::models::{ExperienceRecord, ExperienceSection};
use crate::signal::{truncate_excerpt, EvolutionSignal};
use crate::trajectory::{first_error, render_digest, StepKind, Trajectory, TrajectoryOutcome};

/// The skill name used when the evidence carries none.
///
/// A record must name a skill, and inventing a per-source label would fragment
/// the archive, so unattributed evidence lands under this shared bucket.
pub const DEFAULT_SKILL_NAME: &str = "unknown";

/// The default [`ExperienceExtractor::min_confidence`].
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.5;

/// The default [`ExperienceExtractor::max_records`].
pub const DEFAULT_MAX_RECORDS: usize = 64;

/// The default number of keywords lifted from a record's content.
pub const DEFAULT_KEYWORD_LIMIT: usize = 8;

/// The excerpt length used when a record embeds evidence text.
const CONTENT_EXCERPT_CHARS: usize = 240;

/// Confidence assigned to a lesson derived from a recorded trajectory.
const TRAJECTORY_CONFIDENCE: f64 = 0.9;

/// The minimum number of tool calls a successful trajectory needs before it
/// yields a tool-ordering lesson.
const MIN_TOOLS_FOR_ORDERING: usize = 2;

/// Turns evolution evidence into candidate [`ExperienceRecord`]s.
///
/// The extractor is a pure configuration object: every method is deterministic
/// and free of clocks, randomness and model calls.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperienceExtractor {
    /// Signals below this confidence are ignored, in `0.0..=1.0`.
    pub min_confidence: f64,
    /// The maximum number of records one call returns (`0` means unlimited).
    pub max_records: usize,
    /// The maximum number of keywords lifted from each record's content.
    pub keyword_limit: usize,
}

impl Default for ExperienceExtractor {
    fn default() -> Self {
        Self {
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            max_records: DEFAULT_MAX_RECORDS,
            keyword_limit: DEFAULT_KEYWORD_LIMIT,
        }
    }
}

impl ExperienceExtractor {
    /// Creates an extractor with the default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the minimum signal confidence.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Sets the per-call record limit; `0` means unlimited.
    #[must_use]
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        self
    }

    /// Sets the per-record keyword limit.
    #[must_use]
    pub fn with_keyword_limit(mut self, keyword_limit: usize) -> Self {
        self.keyword_limit = keyword_limit;
        self
    }

    /// Builds one record per signal whose confidence reaches
    /// [`min_confidence`](Self::min_confidence).
    ///
    /// The content is the signal excerpt with runs of whitespace collapsed
    /// (case preserved, so the lesson stays readable) and bounded to 240
    /// characters; the section comes from the signal target; keywords are the
    /// leading content tokens; the skill name falls back to
    /// [`DEFAULT_SKILL_NAME`] and the source to `"signal"`. Records are
    /// deduplicated by id, so the same evidence reported twice yields one
    /// record.
    #[must_use]
    pub fn from_signals(&self, signals: &[EvolutionSignal]) -> Vec<ExperienceRecord> {
        let mut records = Vec::new();
        let mut seen = BTreeSet::new();
        for signal in signals {
            if signal.confidence < self.min_confidence {
                continue;
            }
            let content = normalise_content(signal.excerpt());
            if content.is_empty() {
                continue;
            }
            let record_id = derived_id("sig", &signal.dedupe_key());
            if !seen.insert(record_id.clone()) {
                continue;
            }
            let skill_name = signal
                .skill_name
                .clone()
                .unwrap_or_else(|| DEFAULT_SKILL_NAME.to_owned());
            let section = ExperienceSection::from(signal.target());
            let keywords = self.keywords_for(&content);
            let source = signal.source.clone().unwrap_or_else(|| "signal".to_owned());
            let record = ExperienceRecord {
                record_id,
                skill_name,
                section,
                content,
                keywords,
                source,
                confidence: signal.confidence,
                weight: signal.weight,
                ..ExperienceRecord::default()
            };
            records.push(record);
            if self.at_limit(records.len()) {
                break;
            }
        }
        records
    }

    /// Builds lessons from recorded trajectories.
    ///
    /// A [`TrajectoryOutcome::Failure`] run yields one "avoid" record built
    /// from its first error step (or, without an error step, its digest) and
    /// counts as one failure. A [`TrajectoryOutcome::Success`] run with at least
    /// two tool calls yields one record naming the first request and the tool
    /// order that worked, and counts as one success. Runs with no steps or an
    /// undecided outcome yield nothing.
    #[must_use]
    pub fn from_trajectories(&self, trajectories: &[Trajectory]) -> Vec<ExperienceRecord> {
        let mut records = Vec::new();
        let mut seen = BTreeSet::new();
        for trajectory in trajectories {
            if trajectory.is_empty() {
                continue;
            }
            let extracted = match trajectory.outcome {
                TrajectoryOutcome::Failure => self.failure_record(trajectory),
                TrajectoryOutcome::Success => self.success_record(trajectory),
                TrajectoryOutcome::Unknown => None,
            };
            let Some(record) = extracted else {
                continue;
            };
            if !seen.insert(record.record_id.clone()) {
                continue;
            }
            records.push(record);
            if self.at_limit(records.len()) {
                break;
            }
        }
        records
    }

    /// Builds a lesson from every evaluated case scoring below `threshold`.
    ///
    /// The content names the case, its score, its inputs and the expected
    /// label, bounded to 240 characters; confidence is `1 - score` (so a case
    /// that scored nothing yields a fully trusted lesson) and the record counts
    /// as one failure. Cases scoring at or above `threshold` are skipped.
    #[must_use]
    pub fn from_evaluated_cases(
        &self,
        cases: &[EvaluatedCase],
        threshold: f64,
    ) -> Vec<ExperienceRecord> {
        let mut records = Vec::new();
        let mut seen = BTreeSet::new();
        for case in cases {
            if case.score >= threshold {
                continue;
            }
            let inputs = serde_json::to_string(case.inputs()).unwrap_or_default();
            let label = serde_json::to_string(case.label()).unwrap_or_default();
            let raw = format!(
                "case {} scored {:.2}: inputs {} expected {}",
                case.case_id(),
                case.score,
                inputs,
                label
            );
            let content = truncate_excerpt(&normalise_content(&raw), CONTENT_EXCERPT_CHARS);
            let record_id = derived_id(
                "case",
                &format!("{}:{}", case.case_id(), normalise_content(&raw)),
            );
            if !seen.insert(record_id.clone()) {
                continue;
            }
            let keywords = self.keywords_for(&content);
            let confidence = (1.0 - case.score).clamp(0.0, 1.0);
            let record = ExperienceRecord {
                record_id,
                skill_name: DEFAULT_SKILL_NAME.to_owned(),
                section: ExperienceSection::Other,
                content,
                keywords,
                source: "evaluated_case".to_owned(),
                confidence,
                ..ExperienceRecord::default()
            }
            .record_failure();
            records.push(record);
            if self.at_limit(records.len()) {
                break;
            }
        }
        records
    }

    /// Deduplicates `candidates` through the archive's similarity rule.
    ///
    /// Candidates below [`min_confidence`](Self::min_confidence) are dropped,
    /// near-duplicates merge exactly as [`ExperienceArchive::add`] would, and
    /// the result is ordered by `record_id` so the output is independent of
    /// input order. The batch is capped at [`max_records`](Self::max_records).
    #[must_use]
    pub fn merge(&self, candidates: Vec<ExperienceRecord>) -> Vec<ExperienceRecord> {
        let mut archive = ExperienceArchive::new().with_capacity(0);
        for candidate in candidates {
            if candidate.confidence < self.min_confidence {
                continue;
            }
            // Invalid candidates cannot be stored; skipping them here is a
            // deliberate filter, not an error path (the archive would return
            // the same validation error for every caller).
            if candidate.validate().is_err() {
                continue;
            }
            // `add` fails only on validation, rejected above.
            if let Err(error) = archive.add(candidate) {
                tracing::debug!(%error, "skipping an unstorable experience candidate");
            }
        }
        let mut merged: Vec<ExperienceRecord> = archive.records().to_vec();
        merged.sort_by(|left, right| left.record_id.cmp(&right.record_id));
        if self.max_records > 0 {
            merged.truncate(self.max_records);
        }
        merged
    }

    /// Whether `count` reached the configured limit.
    fn at_limit(&self, count: usize) -> bool {
        self.max_records > 0 && count >= self.max_records
    }

    /// The leading tokens of `content`, capped at the keyword limit.
    fn keywords_for(&self, content: &str) -> Vec<String> {
        let mut keywords: Vec<String> = tokens(content).into_iter().collect();
        if self.keyword_limit > 0 {
            keywords.truncate(self.keyword_limit);
        }
        keywords
    }

    /// Builds the "avoid" lesson for a failed trajectory.
    fn failure_record(&self, trajectory: &Trajectory) -> Option<ExperienceRecord> {
        let (excerpt, section) = match first_error(trajectory) {
            Some(step) => {
                let section = if step.tool_name.is_some() {
                    ExperienceSection::Tool
                } else {
                    ExperienceSection::Other
                };
                (step.excerpt(CONTENT_EXCERPT_CHARS), section)
            }
            None => (
                truncate_excerpt(
                    &render_digest(trajectory, CONTENT_EXCERPT_CHARS),
                    CONTENT_EXCERPT_CHARS,
                ),
                ExperienceSection::Other,
            ),
        };
        let excerpt = normalise_content(&excerpt);
        if excerpt.is_empty() {
            return None;
        }
        let content = format!("avoid: {excerpt}");
        let record_id = derived_id(
            "traj",
            &format!("failure:{}:{}", trajectory.trajectory_id, content),
        );
        let keywords = self.keywords_for(&content);
        Some(
            ExperienceRecord {
                record_id,
                skill_name: Self::skill_name(trajectory),
                section,
                content,
                keywords,
                source: "trajectory".to_owned(),
                confidence: TRAJECTORY_CONFIDENCE,
                ..ExperienceRecord::default()
            }
            .record_failure(),
        )
    }

    /// Builds the tool-ordering lesson for a successful trajectory.
    fn success_record(&self, trajectory: &Trajectory) -> Option<ExperienceRecord> {
        let order = tool_order(trajectory);
        if order.len() < MIN_TOOLS_FOR_ORDERING {
            return None;
        }
        let subject = trajectory
            .steps()
            .iter()
            .find(|step| matches!(step.kind, StepKind::Message | StepKind::System))
            .map_or_else(|| trajectory.trajectory_id.clone(), |step| step.excerpt(80));
        let subject = normalise_content(&subject);
        let content = truncate_excerpt(
            &format!(
                "when doing {subject}, call tools in order: {}",
                order.join(" -> ")
            ),
            CONTENT_EXCERPT_CHARS,
        );
        let record_id = derived_id(
            "traj",
            &format!("success:{}:{}", trajectory.trajectory_id, content),
        );
        let keywords = self.keywords_for(&content);
        Some(
            ExperienceRecord {
                record_id,
                skill_name: Self::skill_name(trajectory),
                section: ExperienceSection::Tool,
                content,
                keywords,
                source: "trajectory".to_owned(),
                confidence: TRAJECTORY_CONFIDENCE,
                ..ExperienceRecord::default()
            }
            .record_success(),
        )
    }

    /// The skill recorded in the trajectory metadata, or the default bucket.
    fn skill_name(trajectory: &Trajectory) -> String {
        trajectory
            .metadata
            .get("skill_name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map_or_else(|| DEFAULT_SKILL_NAME.to_owned(), str::to_owned)
    }
}

/// The distinct tool names of `trajectory` in first-use order.
fn tool_order(trajectory: &Trajectory) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    for step in trajectory.steps() {
        let Some(name) = step.tool_name.as_deref().map(str::trim) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        if seen.insert(name.to_owned()) {
            order.push(name.to_owned());
        }
    }
    order
}

/// Collapses runs of whitespace and trims, preserving case.
fn normalise_content(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Derives a stable record id from `prefix` and `key` using FNV-1a.
fn derived_id(prefix: &str, key: &str) -> String {
    format!("{prefix}-{:016x}", fnv1a64(key.as_bytes()))
}

/// The FNV-1a 64-bit hash of `bytes`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
