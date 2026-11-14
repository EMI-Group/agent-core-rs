//! Caller-supplied evaluation cases, the event-time window, and latency statistics.
//!
//! Ports the trace-facing half of Python `openjiuwen.symphony.models.evaluation`
//! (`EvaluationCase`, `CapabilityCall`) plus `EvaluationWindow` and
//! `latency_statistics` from `openjiuwen.symphony.evaluation.suite`/`evaluators`.
//!
//! # Contract
//!
//! Nothing here executes or inspects a real capability: a case is evidence the caller
//! supplies, and every derived value is a pure function of that evidence. There is no
//! clock read, no filesystem access, and no hash-map iteration order in any output.
//!
//! Times are RFC 3339 strings with an explicit offset (`Z` or `±HH:MM`), because the
//! window filter needs a total order across offsets. A case with no `event_time` is
//! dropped as soon as the window sets either boundary, matching Python's
//! `EvaluationSuite._select_cases`; a case whose `event_time` is not RFC 3339 is a
//! validation error naming the field.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{validation, Result};

/// Whether a trace case ended in success or failure.
///
/// Derived from the observed [`CapabilityCall::success`] flags by
/// [`EvaluationCase::outcome`]: Python carries an explicit `case.success`, while a Rust
/// case records the outcome per call so one case can describe a multi-capability run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaseOutcome {
    /// Every observed call succeeded.
    #[default]
    Success,
    /// At least one observed call failed.
    Failure,
}

/// One capability invocation observed in an evaluation trace.
///
/// Python's `CapabilityCall` also carries ids, typed inputs/outputs, and start/end
/// timestamps; this Rust shape keeps the fields the built-in trace metrics read — the
/// capability name, its success flag, its duration, its own event time, and the raw
/// call arguments.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityCall {
    /// The capability id or display name that was invoked.
    pub name: String,
    /// Whether the call succeeded; `None` means the trace did not record an outcome.
    pub success: Option<bool>,
    /// End-to-end duration in milliseconds; negative values are ignored by the metrics.
    pub duration_ms: Option<f64>,
    /// When the call started, as an RFC 3339 timestamp.
    pub event_time: Option<String>,
    /// The raw call arguments, kept verbatim for evidence.
    pub arguments: Value,
}

impl CapabilityCall {
    /// Builds a call for `name` with no outcome, duration, time, or arguments.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the recorded success flag.
    #[must_use]
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }

    /// Sets the observed duration in milliseconds.
    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: f64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Sets the call's own RFC 3339 event time.
    #[must_use]
    pub fn with_event_time(mut self, event_time: impl Into<String>) -> Self {
        self.event_time = Some(event_time.into());
        self
    }

    /// Sets the raw call arguments.
    #[must_use]
    pub fn with_arguments(mut self, arguments: Value) -> Self {
        self.arguments = arguments;
        self
    }
}

/// One caller-supplied piece of evidence for a capability evaluation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationCase {
    /// Stable case identifier, echoed in aggregated diagnostics.
    pub case_id: String,
    /// The caller's request, if the trace recorded one.
    pub query: String,
    /// When the case happened, as an RFC 3339 timestamp; required to survive a window.
    pub event_time: Option<String>,
    /// Capability invocations observed for this case, in execution order.
    pub calls: Vec<CapabilityCall>,
    /// The answer the caller expected, used by the accuracy exact-match short-circuit.
    pub expected_answer: Option<String>,
    /// The answer that was actually produced, used by the accuracy metric.
    pub actual_answer: Option<String>,
    /// Free-form caller metadata, never interpreted by the built-in metrics.
    pub metadata: Map<String, Value>,
}

impl EvaluationCase {
    /// Builds a case with `case_id` and `query` and nothing else.
    #[must_use]
    pub fn new(case_id: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            case_id: case_id.into(),
            query: query.into(),
            ..Self::default()
        }
    }

    /// Sets the case event time (RFC 3339).
    #[must_use]
    pub fn with_event_time(mut self, event_time: impl Into<String>) -> Self {
        self.event_time = Some(event_time.into());
        self
    }

    /// Sets the observed capability calls.
    #[must_use]
    pub fn with_calls(mut self, calls: Vec<CapabilityCall>) -> Self {
        self.calls = calls;
        self
    }

    /// Sets the expected and actual answers.
    #[must_use]
    pub fn with_answers(mut self, expected: impl Into<String>, actual: impl Into<String>) -> Self {
        self.expected_answer = Some(expected.into());
        self.actual_answer = Some(actual.into());
        self
    }

    /// Sets the actual answer only.
    #[must_use]
    pub fn with_actual_answer(mut self, actual: impl Into<String>) -> Self {
        self.actual_answer = Some(actual.into());
        self
    }

    /// Sets the caller metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Map<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Derives the case outcome from the observed success flags.
    ///
    /// `None` means no call recorded an outcome, which the success-rate metric reports
    /// as `not_applicable` (Python's `missing_success_outcome`). Otherwise the outcome is
    /// [`CaseOutcome::Failure`] as soon as one recorded call failed.
    #[must_use]
    pub fn outcome(&self) -> Option<CaseOutcome> {
        let mut recorded = false;
        let mut failed = false;
        for call in &self.calls {
            match call.success {
                Some(true) => recorded = true,
                Some(false) => {
                    recorded = true;
                    failed = true;
                }
                None => {}
            }
        }
        if !recorded {
            return None;
        }
        Some(if failed {
            CaseOutcome::Failure
        } else {
            CaseOutcome::Success
        })
    }
}

/// A caller-supplied collection of [`EvaluationCase`]s.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationTrace {
    /// The cases, in caller order; metrics never reorder them.
    pub cases: Vec<EvaluationCase>,
}

impl EvaluationTrace {
    /// Builds an empty trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a trace from `cases`.
    #[must_use]
    pub fn from_cases(cases: Vec<EvaluationCase>) -> Self {
        Self { cases }
    }

    /// The number of cases.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether the trace has no cases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

impl FromIterator<EvaluationCase> for EvaluationTrace {
    fn from_iter<I: IntoIterator<Item = EvaluationCase>>(iter: I) -> Self {
        Self {
            cases: iter.into_iter().collect(),
        }
    }
}

/// An optional caller-defined event-time window used for trace aggregation.
///
/// `start` is inclusive and `end` is exclusive, matching Python. Both bounds are RFC
/// 3339 strings; a window with neither bound selects every case.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationWindow {
    /// Inclusive lower bound (RFC 3339).
    pub start: Option<String>,
    /// Exclusive upper bound (RFC 3339).
    pub end: Option<String>,
}

impl EvaluationWindow {
    /// Builds an unbounded window.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a window from optional RFC 3339 bounds.
    #[must_use]
    pub fn between(start: Option<String>, end: Option<String>) -> Self {
        Self { start, end }
    }

    /// Whether either bound is set.
    ///
    /// An unbounded window selects every case, including cases without an `event_time`.
    #[must_use]
    pub fn is_bounded(&self) -> bool {
        self.start.is_some() || self.end.is_some()
    }

    /// Validates both bounds, returning [`crate::error::Error::Validation`] when a bound
    /// is not RFC 3339 or `end` precedes `start`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] naming `window.start`, `window.end`,
    /// or `window.end`/`window.start` ordering.
    pub fn validate(&self) -> Result<()> {
        let start = self.lower_bound()?;
        let end = self.upper_bound()?;
        if let (Some(start), Some(end)) = (start, end) {
            if end < start {
                return Err(validation("window.end", "must not precede window.start"));
            }
        }
        Ok(())
    }

    /// Parses the inclusive lower bound.
    fn lower_bound(&self) -> Result<Option<Instant>> {
        self.start
            .as_deref()
            .map(|raw| {
                parse_timestamp(raw)
                    .ok_or_else(|| validation("window.start", "must be an RFC 3339 timestamp"))
            })
            .transpose()
    }

    /// Parses the exclusive upper bound.
    fn upper_bound(&self) -> Result<Option<Instant>> {
        self.end
            .as_deref()
            .map(|raw| {
                parse_timestamp(raw)
                    .ok_or_else(|| validation("window.end", "must be an RFC 3339 timestamp"))
            })
            .transpose()
    }
}

/// Selects the cases a metric may inspect, applying `window` when it is bounded.
///
/// The rules mirror Python's `EvaluationSuite._select_cases`:
///
/// - no window, or a window with neither bound, keeps every case;
/// - otherwise a case without `event_time` is dropped and a case whose `event_time` is
///   unparseable is a validation error;
/// - `start` is inclusive and `end` is exclusive.
///
/// # Errors
///
/// Returns [`crate::error::Error::Validation`] for an invalid window bound, an
/// inverted window, or an unparseable `event_time` (`event_time` names the field).
pub fn select_cases<'a>(
    cases: &'a [EvaluationCase],
    window: Option<&EvaluationWindow>,
) -> Result<Vec<&'a EvaluationCase>> {
    let Some(window) = window.filter(|window| window.is_bounded()) else {
        return Ok(cases.iter().collect());
    };
    let start = window.lower_bound()?;
    let end = window.upper_bound()?;
    if let (Some(start), Some(end)) = (start, end) {
        if end < start {
            return Err(validation("window.end", "must not precede window.start"));
        }
    }
    let mut selected = Vec::with_capacity(cases.len());
    for case in cases {
        let Some(raw) = case.event_time.as_deref() else {
            continue;
        };
        let event_time = parse_timestamp(raw)
            .ok_or_else(|| validation("event_time", "must be an RFC 3339 timestamp"))?;
        if start.is_some_and(|start| event_time < start) {
            continue;
        }
        if end.is_some_and(|end| event_time >= end) {
            continue;
        }
        selected.push(case);
    }
    Ok(selected)
}

/// Raw latency distribution statistics; no threshold or target is applied.
///
/// Ports Python's `latency_statistics` dictionary: the keys are `avg_ms`, `p50_ms`,
/// `p95_ms`, `max_ms`, `count`, and `samples_ms` (the retained samples, in input order).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LatencyStatistics {
    /// Arithmetic mean of the retained samples.
    pub avg_ms: f64,
    /// Interpolated 50th percentile.
    pub p50_ms: f64,
    /// Interpolated 95th percentile.
    pub p95_ms: f64,
    /// Largest retained sample.
    pub max_ms: f64,
    /// Number of retained samples.
    pub count: usize,
    /// The retained samples, in input order.
    pub samples_ms: Vec<f64>,
}

impl LatencyStatistics {
    /// Computes the statistics from finite, non-negative `samples`.
    ///
    /// Returns `None` when no sample survives the filter, mirroring Python's empty
    /// dictionary.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // sample counts are far below 2^53
    pub fn from_samples(samples: impl IntoIterator<Item = f64>) -> Option<Self> {
        let samples: Vec<f64> = samples
            .into_iter()
            .filter(|sample| sample.is_finite() && *sample >= 0.0)
            .collect();
        if samples.is_empty() {
            return None;
        }
        let total: f64 = samples.iter().sum();
        let avg_ms = total / samples.len() as f64;
        Some(Self {
            avg_ms,
            p50_ms: percentile(&samples, 0.50)?,
            p95_ms: percentile(&samples, 0.95)?,
            max_ms: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            count: samples.len(),
            samples_ms: samples,
        })
    }

    /// Renders the statistics for a metric `detail` string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "observed latency without a threshold: avg_ms={:.3}, p50_ms={:.3}, p95_ms={:.3}, \
             max_ms={:.3}, count={}",
            self.avg_ms, self.p50_ms, self.p95_ms, self.max_ms, self.count
        )
    }
}

/// Computes Python's interpolated percentile over `samples`.
///
/// `position = (n - 1) * fraction`, interpolating between the two neighbouring sorted
/// samples. `fraction` is clamped to `[0.0, 1.0]` (a non-finite fraction becomes `0.0`),
/// and an empty input returns `None`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn percentile(samples: &[f64], fraction: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let fraction = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut ordered: Vec<f64> = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    if ordered.len() == 1 {
        return ordered.first().copied();
    }
    let position = (ordered.len() - 1) as f64 * fraction;
    let lower = position as usize;
    let upper = (lower + 1).min(ordered.len() - 1);
    let weight = position - lower as f64;
    let start = *ordered.get(lower)?;
    let end = *ordered.get(upper)?;
    Some(start + (end - start) * weight)
}

/// Computes [`LatencyStatistics`] from `values`, or `None` when no value is usable.
#[must_use]
pub fn latency_statistics(values: impl IntoIterator<Item = f64>) -> Option<LatencyStatistics> {
    LatencyStatistics::from_samples(values)
}

/// A timezone-normalized instant used only to order window bounds and case times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Instant {
    /// Seconds since the Unix epoch.
    seconds: i64,
    /// Nanoseconds within the second.
    nanos: u32,
}

/// Parses an RFC 3339 timestamp with an explicit offset into an [`Instant`].
///
/// Accepted shape: `YYYY-MM-DD[Tt ]HH:MM:SS[.fraction](Z|z|±HH:MM|±HHMM)`. A timestamp
/// without an offset is rejected, because two naive timestamps cannot be ordered
/// reliably across a window.
fn parse_timestamp(raw: &str) -> Option<Instant> {
    let text = raw.trim();
    let bytes = text.as_bytes();
    if bytes.len() < 20 {
        return None;
    }
    let year = parse_digits(bytes, 0, 4)?;
    if bytes.get(4) != Some(&b'-') {
        return None;
    }
    let month = parse_digits(bytes, 5, 2)?;
    if bytes.get(7) != Some(&b'-') {
        return None;
    }
    let day = parse_digits(bytes, 8, 2)?;
    if !matches!(bytes.get(10), Some(b'T' | b't' | b' ')) {
        return None;
    }
    let hour = parse_digits(bytes, 11, 2)?;
    if bytes.get(13) != Some(&b':') {
        return None;
    }
    let minute = parse_digits(bytes, 14, 2)?;
    if bytes.get(16) != Some(&b':') {
        return None;
    }
    let second = parse_digits(bytes, 17, 2)?;
    let mut index = 19;
    let mut nanos: u32 = 0;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            let digit = u32::from(bytes[index] - b'0');
            if index - start < 9 {
                nanos = nanos * 10 + digit;
            }
            index += 1;
        }
        let digits = index - start;
        if digits == 0 {
            return None;
        }
        for _ in digits..9 {
            nanos *= 10;
        }
    }
    let offset_seconds = match *bytes.get(index)? {
        b'Z' | b'z' => {
            index += 1;
            0_i64
        }
        sign @ (b'+' | b'-') => {
            index += 1;
            let offset_hour = parse_digits(bytes, index, 2)?;
            index += 2;
            if bytes.get(index) == Some(&b':') {
                index += 1;
            }
            let offset_minute = parse_digits(bytes, index, 2)?;
            index += 2;
            let magnitude = offset_hour * 3_600 + offset_minute * 60;
            if sign == b'-' {
                -magnitude
            } else {
                magnitude
            }
        }
        _ => return None,
    };
    if index != bytes.len() {
        return None;
    }
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=59).contains(&second)
    {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let seconds = days * 86_400 + hour * 3_600 + minute * 60 + second - offset_seconds;
    Some(Instant { seconds, nanos })
}

/// Parses exactly `count` ASCII digits starting at `start`.
fn parse_digits(bytes: &[u8], start: usize, count: usize) -> Option<i64> {
    if start + count > bytes.len() {
        return None;
    }
    let mut value: i64 = 0;
    for offset in start..start + count {
        let byte = *bytes.get(offset)?;
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + i64::from(byte - b'0');
    }
    Some(value)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    /// Compares two interpolated percentiles, which are not bit-exact for 0.95.
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{actual} is not within 1e-9 of {expected}"
        );
    }

    #[test]
    fn percentile_interpolates_between_neighbours() {
        assert_close(percentile(&[7.0], 0.5).unwrap(), 7.0);
        assert_close(percentile(&[1.0, 2.0], 0.5).unwrap(), 1.5);
        assert_close(percentile(&[1.0, 2.0], 0.95).unwrap(), 1.95);
        assert_close(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5).unwrap(), 2.5);
        assert_close(percentile(&[1.0, 2.0, 3.0, 4.0], 0.95).unwrap(), 3.85);
        assert_close(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5).unwrap(), 3.0);
        assert_close(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95).unwrap(), 4.8);
        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_close(percentile(&ten, 0.5).unwrap(), 5.5);
        assert_close(percentile(&ten, 0.95).unwrap(), 9.55);
        assert_eq!(percentile(&[], 0.5), None);
    }

    #[test]
    fn percentile_does_not_require_sorted_input() {
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5).unwrap(), 2.5);
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.0).unwrap(), 1.0);
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 1.0).unwrap(), 4.0);
    }

    #[test]
    fn latency_statistics_drop_unusable_samples() {
        let stats = latency_statistics([5.0, -1.0, f64::NAN, 1.0, f64::INFINITY, 3.0]).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.samples_ms, vec![5.0, 1.0, 3.0]);
        assert_eq!(stats.avg_ms, 3.0);
        assert_eq!(stats.p50_ms, 3.0);
        assert_eq!(stats.max_ms, 5.0);
        assert!(latency_statistics(Vec::<f64>::new()).is_none());
        assert!(latency_statistics([-1.0]).is_none());
    }

    #[test]
    fn timestamps_order_across_offsets() {
        let utc = parse_timestamp("2026-01-01T00:00:00Z").unwrap();
        let shanghai = parse_timestamp("2026-01-01T08:00:00+08:00").unwrap();
        assert_eq!(utc, shanghai);
        let later = parse_timestamp("2026-01-01T00:00:01Z").unwrap();
        assert!(later > utc);
        assert_eq!(
            parse_timestamp("2026-01-01T00:00:00.500Z").unwrap().nanos,
            500_000_000
        );
        assert_eq!(
            parse_timestamp("2026-01-01T00:00:00.123456789Z")
                .unwrap()
                .nanos,
            123_456_789
        );
        assert_eq!(
            parse_timestamp("2026-01-01 00:00:00+0800"),
            parse_timestamp("2025-12-31T16:00:00Z")
        );
        assert!(parse_timestamp("2026-01-01T00:00:00").is_none());
        assert!(parse_timestamp("2026-13-01T00:00:00Z").is_none());
        assert!(parse_timestamp("not a timestamp").is_none());
    }

    #[test]
    fn window_selection_is_start_inclusive_and_end_exclusive() {
        let cases = vec![
            EvaluationCase::new("early", "q").with_event_time("2026-01-01T00:00:00Z"),
            EvaluationCase::new("start", "q").with_event_time("2026-01-02T00:00:00Z"),
            EvaluationCase::new("end", "q").with_event_time("2026-01-03T00:00:00Z"),
            EvaluationCase::new("undated", "q"),
        ];
        let window = EvaluationWindow::between(
            Some("2026-01-02T00:00:00Z".to_string()),
            Some("2026-01-03T00:00:00Z".to_string()),
        );
        let selected = select_cases(&cases, Some(&window)).unwrap();
        let ids: Vec<&str> = selected.iter().map(|case| case.case_id.as_str()).collect();
        assert_eq!(ids, vec!["start"]);
        let all = select_cases(&cases, Some(&EvaluationWindow::new())).unwrap();
        assert_eq!(all.len(), 4);
        assert_eq!(select_cases(&cases, None).unwrap().len(), 4);
    }

    #[test]
    fn invalid_windows_and_event_times_are_validation_errors() {
        let cases = vec![EvaluationCase::new("bad", "q").with_event_time("yesterday")];
        let inverted = EvaluationWindow::between(
            Some("2026-01-02T00:00:00Z".to_string()),
            Some("2026-01-01T00:00:00Z".to_string()),
        );
        assert_eq!(
            inverted.validate().unwrap_err().kind(),
            crate::error::ErrorKind::Validation
        );
        let error = select_cases(&cases, Some(&inverted)).unwrap_err();
        assert!(error.message().starts_with("window.end:"));
        let missing = EvaluationWindow {
            start: Some("yesterday".to_string()),
            end: None,
        };
        let error = select_cases(&cases, Some(&missing)).unwrap_err();
        assert!(error.message().starts_with("window.start:"));
        let valid = EvaluationWindow::between(Some("2026-01-01T00:00:00Z".to_string()), None);
        let error = select_cases(&cases, Some(&valid)).unwrap_err();
        assert!(error.message().starts_with("event_time:"));
    }

    #[test]
    fn case_outcome_derives_from_recorded_success_flags() {
        assert_eq!(EvaluationCase::new("c", "q").outcome(), None);
        assert_eq!(
            EvaluationCase::new("c", "q")
                .with_calls(vec![CapabilityCall::new("a").with_success(true)])
                .outcome(),
            Some(CaseOutcome::Success)
        );
        assert_eq!(
            EvaluationCase::new("c", "q")
                .with_calls(vec![
                    CapabilityCall::new("a").with_success(true),
                    CapabilityCall::new("b").with_success(false),
                ])
                .outcome(),
            Some(CaseOutcome::Failure)
        );
        assert_eq!(
            EvaluationCase::new("c", "q")
                .with_calls(vec![CapabilityCall::new("a")])
                .outcome(),
            None
        );
    }
}
