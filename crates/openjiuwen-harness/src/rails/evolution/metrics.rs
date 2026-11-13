//! Per-iteration evolution metrics and the bounded recorder that keeps them.
//!
//! One [`EvolutionMetrics`] row summarises what happened between two outer
//! task-loop iterations: how long it took, how many model and tool calls it
//! made, which tools it used, how many of those calls failed, the token usage
//! the provider reported, and why the model stopped. [`IterationRecorder`]
//! keeps the last `capacity` rows of one run (unbounded at `capacity == 0`) and
//! folds them into a totals row for the run snapshot.
//!
//! Time is taken through an injectable [`MetricsClock`] so tests never depend
//! on the wall clock; the default is monotonic milliseconds since the first
//! call in the process.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Source of the millisecond timestamps recorded in [`EvolutionMetrics`].
///
/// The default ([`default_metrics_clock`]) is monotonic; a test injects a
/// counter through [`IterationRecorder::with_clock`] instead.
pub type MetricsClock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Anchor of the default monotonic clock: the first call is the zero point.
static ANCHOR: OnceLock<Instant> = OnceLock::new();

/// The default [`MetricsClock`]: monotonic milliseconds since the process
/// started reading metrics.
#[must_use]
pub fn default_metrics_clock() -> MetricsClock {
    Arc::new(|| {
        let anchor = *ANCHOR.get_or_init(Instant::now);
        u64::try_from(anchor.elapsed().as_millis()).unwrap_or(u64::MAX)
    })
}

/// What one outer task-loop iteration cost.
///
/// The token counters stay `None` unless a provider reported a non-zero usage
/// sum (see [`EvolutionMetrics::with_usage`]), so an offline run never claims
/// "0 tokens" for work whose cost is simply unknown.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct EvolutionMetrics {
    /// Iteration index inside its round, `0`-based.
    pub iteration: usize,
    /// Milliseconds since the previous iteration closed.
    pub elapsed_ms: u64,
    /// Model calls made during the iteration.
    pub model_calls: u64,
    /// Tool calls executed during the iteration.
    pub tool_calls: u64,
    /// Tool calls that returned a failure.
    pub failed_tool_calls: u64,
    /// Tool calls per tool name.
    pub tool_calls_by_name: BTreeMap<String, u64>,
    /// Prompt tokens reported for the iteration, when any were.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Completion tokens reported for the iteration, when any were.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens reported for the iteration, when any were.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Provider stop/finish reason of the last model call, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

impl EvolutionMetrics {
    /// Records one tool call, failed or not.
    pub fn record_tool_call(&mut self, name: &str, failed: bool) {
        self.tool_calls = self.tool_calls.saturating_add(1);
        if failed {
            self.failed_tool_calls = self.failed_tool_calls.saturating_add(1);
        }
        if !name.is_empty() {
            let count = self.tool_calls_by_name.entry(name.to_owned()).or_insert(0);
            *count = count.saturating_add(1);
        }
    }

    /// Sets the three token counters, or leaves them `None` when nothing was
    /// reported (the sum of the arguments is `0`).
    #[must_use]
    pub fn with_usage(mut self, input_tokens: u64, output_tokens: u64, total_tokens: u64) -> Self {
        if input_tokens
            .saturating_add(output_tokens)
            .saturating_add(total_tokens)
            == 0
        {
            return self;
        }
        self.input_tokens = Some(input_tokens);
        self.output_tokens = Some(output_tokens);
        self.total_tokens = Some(total_tokens);
        self
    }
}

/// Adds `next` to `current`, keeping `None` when neither side reported a value.
pub(super) fn sum_tokens(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (None, None) => None,
        (current, next) => Some(
            current
                .unwrap_or_default()
                .saturating_add(next.unwrap_or_default()),
        ),
    }
}

/// The bounded ring buffer of one run's [`EvolutionMetrics`] rows.
///
/// `capacity == 0` means unbounded; otherwise the oldest row is dropped when a
/// new one would overflow. Cheap to share: every accessor takes `&self`.
pub struct IterationRecorder {
    /// Maximum number of kept rows, `0` for unbounded.
    capacity: usize,
    /// Source of the row timestamps.
    clock: MetricsClock,
    /// Oldest first; guarded because every method takes `&self`.
    entries: Mutex<Vec<EvolutionMetrics>>,
}

impl IterationRecorder {
    /// Creates a recorder with the default monotonic clock.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::with_clock(capacity, default_metrics_clock())
    }

    /// Creates a recorder reading time from `clock`.
    #[must_use]
    pub fn with_clock(capacity: usize, clock: MetricsClock) -> Self {
        Self {
            capacity,
            clock,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Maximum number of kept rows, `0` when unbounded.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of kept rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Whether no row is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The current clock reading, in milliseconds.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        (self.clock)()
    }

    /// Appends one row, evicting the oldest when the buffer is full.
    pub fn push(&self, metrics: EvolutionMetrics) {
        let mut entries = self.entries.lock();
        if self.capacity > 0 && entries.len() >= self.capacity {
            let len = entries.len();
            let excess = len.saturating_add(1).saturating_sub(self.capacity);
            entries.drain(..excess.min(len));
        }
        entries.push(metrics);
    }

    /// The kept rows, oldest first.
    #[must_use]
    pub fn list(&self) -> Vec<EvolutionMetrics> {
        self.entries.lock().clone()
    }

    /// Drops every row; the capacity and clock are kept.
    pub fn clear(&self) {
        self.entries.lock().clear();
    }

    /// Folds the kept rows into one aggregate row.
    ///
    /// `iteration` is the number of folded rows, per-tool counts are merged,
    /// token counters are summed (and stay `None` when no row reported any),
    /// and `stop_reason` is the last non-`None` reason.
    #[must_use]
    pub fn totals(&self) -> EvolutionMetrics {
        let entries = self.entries.lock();
        let mut totals = EvolutionMetrics {
            iteration: entries.len(),
            ..EvolutionMetrics::default()
        };
        for metrics in entries.iter() {
            totals.elapsed_ms = totals.elapsed_ms.saturating_add(metrics.elapsed_ms);
            totals.model_calls = totals.model_calls.saturating_add(metrics.model_calls);
            totals.tool_calls = totals.tool_calls.saturating_add(metrics.tool_calls);
            totals.failed_tool_calls = totals
                .failed_tool_calls
                .saturating_add(metrics.failed_tool_calls);
            for (name, count) in &metrics.tool_calls_by_name {
                let slot = totals.tool_calls_by_name.entry(name.clone()).or_insert(0);
                *slot = slot.saturating_add(*count);
            }
            totals.input_tokens = sum_tokens(totals.input_tokens, metrics.input_tokens);
            totals.output_tokens = sum_tokens(totals.output_tokens, metrics.output_tokens);
            totals.total_tokens = sum_tokens(totals.total_tokens, metrics.total_tokens);
            if metrics.stop_reason.is_some() {
                totals.stop_reason.clone_from(&metrics.stop_reason);
            }
        }
        totals
    }
}

impl fmt::Debug for IterationRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IterationRecorder")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// A clock that only moves when the test advances it.
    struct TestClock(Arc<AtomicU64>);

    impl TestClock {
        fn new(start_ms: u64) -> Self {
            Self(Arc::new(AtomicU64::new(start_ms)))
        }

        fn advance(&self, delta_ms: u64) -> u64 {
            self.0.fetch_add(delta_ms, Ordering::SeqCst) + delta_ms
        }

        fn clock(&self) -> MetricsClock {
            let state = Arc::clone(&self.0);
            Arc::new(move || state.load(Ordering::SeqCst))
        }
    }

    fn metrics(iteration: usize) -> EvolutionMetrics {
        EvolutionMetrics {
            iteration,
            ..EvolutionMetrics::default()
        }
    }

    #[test]
    fn usage_is_omitted_unless_something_was_reported() {
        let empty = EvolutionMetrics::default().with_usage(0, 0, 0);
        assert_eq!(empty.input_tokens, None);
        assert_eq!(empty.output_tokens, None);
        assert_eq!(empty.total_tokens, None);

        let reported = EvolutionMetrics::default().with_usage(10, 0, 10);
        assert_eq!(reported.input_tokens, Some(10));
        assert_eq!(reported.output_tokens, Some(0));
        assert_eq!(reported.total_tokens, Some(10));
    }

    #[test]
    fn recording_a_tool_call_counts_names_and_failures() {
        let mut row = EvolutionMetrics::default();
        row.record_tool_call("bash", false);
        row.record_tool_call("bash", true);
        row.record_tool_call("read_file", false);

        assert_eq!(row.tool_calls, 3);
        assert_eq!(row.failed_tool_calls, 1);
        assert_eq!(
            row.tool_calls_by_name.get("bash").copied(),
            Some(2),
            "per-tool counts"
        );
        assert_eq!(row.tool_calls_by_name.get("read_file").copied(), Some(1));
    }

    #[test]
    fn bounded_recorder_evicts_the_oldest_rows() {
        let recorder = IterationRecorder::new(2);
        for iteration in 0..4 {
            recorder.push(metrics(iteration));
        }

        let kept: Vec<usize> = recorder.list().iter().map(|row| row.iteration).collect();
        assert_eq!(kept, [2, 3]);
        assert_eq!(recorder.len(), 2);
        assert_eq!(recorder.capacity(), 2);
    }

    #[test]
    fn zero_capacity_keeps_every_row() {
        let recorder = IterationRecorder::new(0);
        assert!(recorder.is_empty());
        for iteration in 0..5 {
            recorder.push(metrics(iteration));
        }

        assert_eq!(recorder.len(), 5);
        assert_eq!(recorder.capacity(), 0);
        assert!(!recorder.is_empty());
    }

    #[test]
    fn clear_drops_rows_but_keeps_the_capacity() {
        let recorder = IterationRecorder::new(4);
        recorder.push(metrics(0));
        recorder.clear();

        assert!(recorder.is_empty());
        assert_eq!(recorder.capacity(), 4);
    }

    #[test]
    fn clock_is_injectable() {
        let clock = TestClock::new(1_000);
        let recorder = IterationRecorder::with_clock(0, clock.clock());
        assert_eq!(recorder.now_ms(), 1_000);

        clock.advance(250);
        assert_eq!(recorder.now_ms(), 1_250);
    }

    #[test]
    fn totals_sum_counters_and_merge_tool_names() {
        let recorder = IterationRecorder::new(0);
        let mut first = EvolutionMetrics {
            iteration: 0,
            elapsed_ms: 10,
            model_calls: 1,
            stop_reason: Some("tool_calls".to_owned()),
            ..EvolutionMetrics::default()
        };
        first.record_tool_call("bash", false);
        first.input_tokens = Some(7);

        let mut second = EvolutionMetrics {
            iteration: 1,
            elapsed_ms: 20,
            model_calls: 2,
            ..EvolutionMetrics::default()
        };
        second.record_tool_call("bash", true);
        second.record_tool_call("grep", false);
        second.output_tokens = Some(3);
        second.stop_reason = Some("stop".to_owned());

        recorder.push(first);
        recorder.push(second);

        let totals = recorder.totals();
        assert_eq!(totals.iteration, 2, "totals report the folded row count");
        assert_eq!(totals.elapsed_ms, 30);
        assert_eq!(totals.model_calls, 3);
        assert_eq!(totals.tool_calls, 3);
        assert_eq!(totals.failed_tool_calls, 1);
        assert_eq!(totals.tool_calls_by_name.get("bash").copied(), Some(2));
        assert_eq!(totals.tool_calls_by_name.get("grep").copied(), Some(1));
        assert_eq!(totals.input_tokens, Some(7));
        assert_eq!(totals.output_tokens, Some(3));
        assert_eq!(totals.total_tokens, None, "no row reported a total");
        assert_eq!(totals.stop_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn totals_of_an_empty_recorder_are_default() {
        let recorder = IterationRecorder::new(8);
        assert_eq!(recorder.totals(), EvolutionMetrics::default());
    }

    #[test]
    fn debug_prints_capacity_and_len() {
        let recorder = IterationRecorder::new(3);
        recorder.push(metrics(0));
        let rendered = format!("{recorder:?}");
        assert!(rendered.contains("capacity: 3"), "{rendered}");
        assert!(rendered.contains("len: 1"), "{rendered}");
    }
}
