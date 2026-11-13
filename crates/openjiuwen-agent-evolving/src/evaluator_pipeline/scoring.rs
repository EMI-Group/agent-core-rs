//! Turning a test command's result into a score.
//!
//! Python's benchmark adapters each parsed their own output (pytest JSON, a
//! custom verifier, an exit code). The Rust port keeps that flexibility as data:
//! [`ScoreSpec`] selects a parser and [`parse_score`] is a pure function from
//! an [`ExecResult`] to a [`ScoredRun`].
//!
//! - [`ScoreSpec::ExitCode`] — the default: exit code `0` is a full pass.
//! - [`ScoreSpec::Json`] — read the last JSON object line, optionally at a
//!   dotted `path`, expecting `score` (a `0..=1` fraction or a `0..=100`
//!   percentage) and an optional `passed` flag.
//! - [`ScoreSpec::TestSummary`] — count `N passed`/`N failed` words or TAP
//!   `ok`/`not ok` lines and score `passed / total`.
//! - [`ScoreSpec::Auto`] — try JSON, then the test summary, then the exit code.
//!
//! A timed-out command always scores `0.0` and never passes, whatever the spec:
//! a killed verifier produced no trustworthy verdict.
//!
//! Everything is deterministic and textual — no model judge, no clock, no
//! environment read. [`parse_last_json`] and [`parse_test_counts`] are the pure
//! helpers the parsers share and are public so callers can reuse or test them.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::models::ExecResult;

/// The highest percentage that is still interpreted as a `0..=100` score.
const MAX_PERCENTAGE: f64 = 100.0;

/// How a test command's result becomes a score.
///
/// The serde representation is `snake_case` (`exit_code`, `json`,
/// `test_summary`, `auto`), matching the crate's other config enums.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreSpec {
    /// `returncode == 0` is `1.0`/passed, anything else is `0.0`/failed.
    #[default]
    ExitCode,
    /// Read the last JSON object line of standard output.
    Json {
        /// An optional dotted path inside that object, for example
        /// `result.score`; `None` reads the object itself.
        #[serde(default)]
        path: Option<String>,
    },
    /// Count `passed`/`failed` tests in pytest-ish or TAP-ish output.
    TestSummary,
    /// Try [`ScoreSpec::Json`], then [`ScoreSpec::TestSummary`], then
    /// [`ScoreSpec::ExitCode`].
    Auto,
}

/// The score one run produced.
///
/// `score` is always inside `0.0..=1.0`; `detail` explains how the parser
/// arrived at it and is what the pipeline stores as a task's failure reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScoredRun {
    /// The score in `0.0..=1.0`.
    pub score: f64,
    /// Whether the parser considered the run a pass.
    pub passed: bool,
    /// A human-readable explanation of the verdict.
    pub detail: String,
}

impl ScoredRun {
    /// Builds a scored run.
    #[must_use]
    pub fn new(score: f64, passed: bool, detail: impl Into<String>) -> Self {
        Self {
            score,
            passed,
            detail: detail.into(),
        }
    }
}

/// The `passed`/`failed` counts one output block reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TestCounts {
    /// The number of passing tests.
    pub passed: u64,
    /// The number of failing tests.
    pub failed: u64,
}

impl TestCounts {
    /// Builds a count pair.
    #[must_use]
    pub const fn new(passed: u64, failed: u64) -> Self {
        Self { passed, failed }
    }

    /// The total number of tests counted.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.passed + self.failed
    }

    /// `true` when at least one test ran and none failed.
    #[must_use]
    pub const fn all_passed(&self) -> bool {
        self.failed == 0 && self.passed > 0
    }
}

/// Parses a command's result with `spec`.
///
/// Returns [`Error::validation`] when the selected parser cannot find the data
/// it needs (no JSON line, a missing/invalid `score`, no recognisable test
/// summary); the pipeline reports that as the task's failure.
pub fn parse_score(result: &ExecResult, spec: &ScoreSpec) -> Result<ScoredRun> {
    if result.timed_out {
        return Ok(ScoredRun::new(
            0.0,
            false,
            format!("command timed out: {}", result.stderr.trim()),
        ));
    }
    match spec {
        ScoreSpec::ExitCode => Ok(exit_code_score(result)),
        ScoreSpec::Json { path } => json_score(result, path.as_deref()),
        ScoreSpec::TestSummary => summary_score(result),
        ScoreSpec::Auto => Ok(auto_score(result)),
    }
}

/// The exit-code score: `0` is a full pass, anything else fails.
fn exit_code_score(result: &ExecResult) -> ScoredRun {
    if result.returncode == 0 {
        ScoredRun::new(1.0, true, "exit code 0")
    } else {
        ScoredRun::new(0.0, false, format!("exit code {}", result.returncode))
    }
}

/// The JSON score, failing when no JSON object line carries a score.
fn json_score(result: &ExecResult, path: Option<&str>) -> Result<ScoredRun> {
    let value = parse_last_json(&result.stdout)
        .ok_or_else(|| Error::validation("standard output contains no JSON object line"))?;
    let value = match path {
        Some(path) => resolve_path(&value, path)?,
        None => &value,
    };
    score_from_json(value)
}

/// The test-summary score, failing when no summary can be found.
fn summary_score(result: &ExecResult) -> Result<ScoredRun> {
    let counts = parse_test_counts(&result.stdout)
        .or_else(|| parse_test_counts(&result.stderr))
        .filter(|counts| counts.total() > 0);
    counts.map_or_else(
        || {
            Err(Error::validation(
                "no recognisable test summary in the command output",
            ))
        },
        |counts| Ok(scored_from_counts(counts)),
    )
}

/// The `Auto` policy: JSON, then a test summary, then the exit code.
fn auto_score(result: &ExecResult) -> ScoredRun {
    if let Some(value) = parse_last_json(&result.stdout) {
        if let Ok(scored) = score_from_json(&value) {
            return scored;
        }
    }
    let counts = parse_test_counts(&result.stdout)
        .or_else(|| parse_test_counts(&result.stderr))
        .filter(|counts| counts.total() > 0);
    if let Some(counts) = counts {
        return scored_from_counts(counts);
    }
    exit_code_score(result)
}

/// Projects the counts onto a [`ScoredRun`].
fn scored_from_counts(counts: TestCounts) -> ScoredRun {
    let total = counts.total();
    #[allow(clippy::cast_precision_loss)]
    let score = counts.passed as f64 / total as f64;
    ScoredRun::new(
        score,
        counts.all_passed(),
        format!("{}/{total} tests passed", counts.passed),
    )
}

/// Converts a JSON `score`/`passed` pair into a [`ScoredRun`].
///
/// A bare number is the score; an object needs a numeric `score` and may carry
/// `passed`. Values above `1.0` up to `100.0` are read as percentages, and
/// `passed` defaults to "the score is a full pass".
fn score_from_json(value: &Value) -> Result<ScoredRun> {
    let (raw, passed) = match value {
        Value::Number(number) => (
            number
                .as_f64()
                .ok_or_else(|| Error::validation("json score is not representable as f64"))?,
            None,
        ),
        Value::Object(map) => {
            let raw = map
                .get("score")
                .ok_or_else(|| Error::validation("json score object has no `score` field"))?
                .as_f64()
                .ok_or_else(|| Error::validation("json `score` field is not a number"))?;
            let passed = map.get("passed").and_then(Value::as_bool);
            (raw, passed)
        }
        other => {
            return Err(Error::validation(format!(
                "json score must be a number or an object, got {other}"
            )));
        }
    };
    let score = normalise_score(raw)?;
    let passed = passed.unwrap_or(score >= 1.0);
    Ok(ScoredRun::new(score, passed, format!("json score {score}")))
}

/// Normalises a raw JSON score into `0.0..=1.0`.
fn normalise_score(raw: f64) -> Result<f64> {
    if !raw.is_finite() {
        return Err(Error::validation(format!("Score {raw} is not finite")));
    }
    let score = if raw > 1.0 && raw <= MAX_PERCENTAGE {
        raw / MAX_PERCENTAGE
    } else {
        raw
    };
    if !(0.0..=1.0).contains(&score) {
        return Err(Error::validation(format!(
            "Score {raw} is outside the accepted 0..=1 or 0..=100 range"
        )));
    }
    Ok(score)
}

/// Resolves a dotted path (`a.b.0`) inside `value`.
fn resolve_path<'a>(value: &'a Value, path: &str) -> Result<&'a Value> {
    let mut current = value;
    for segment in path.split('.').filter(|segment| !segment.is_empty()) {
        current = match current {
            Value::Object(map) => map.get(segment).ok_or_else(|| {
                Error::validation(format!(
                    "json score path `{path}` has no `{segment}` segment"
                ))
            })?,
            Value::Array(items) => {
                let index: usize = segment.parse().map_err(|_| {
                    Error::validation(format!(
                        "json score path `{path}` uses `{segment}` on an array"
                    ))
                })?;
                items.get(index).ok_or_else(|| {
                    Error::validation(format!(
                        "json score path `{path}` index {index} is out of bounds"
                    ))
                })?
            }
            other => {
                return Err(Error::validation(format!(
                    "json score path `{path}` cannot descend into {other}"
                )));
            }
        };
    }
    Ok(current)
}

/// The last non-empty line of `text` that parses as a JSON object.
///
/// Returning the last line makes the parser indifferent to progress output the
/// test command printed before its final verdict. Lines that are not objects
/// (numbers, strings, arrays, truncated output) are skipped, so a torn trailing
/// line does not hide an earlier valid verdict.
#[must_use]
pub fn parse_last_json(text: &str) -> Option<Value> {
    text.lines()
        .rev()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(|line| match serde_json::from_str::<Value>(line) {
            Ok(value @ Value::Object(_)) => Some(value),
            _ => None,
        })
}

/// Counts the `passed`/`failed` tests one output block reports.
///
/// Recognised forms, checked in this order:
///
/// 1. TAP: lines starting with `ok` or `not ok` (and `ok`/`not ok` alone);
/// 2. a number followed by a `passed`/`passing` or
///    `failed`/`failing`/`failure(s)` word, which covers pytest's
///    `= 3 failed, 12 passed in 0.2s =` and `Tests: 3 failed, 5 passed`;
/// 3. a standalone `PASS`/`PASSED` or `FAIL`/`FAILED` line.
///
/// Returns `None` when nothing recognisable is present, so the caller can pick
/// another policy. Skipped tests are ignored: only passes and failures count.
#[must_use]
pub fn parse_test_counts(text: &str) -> Option<TestCounts> {
    let mut tap = TestCounts::default();
    let mut tap_seen = false;
    for line in text.lines() {
        let lowered = line.trim().to_ascii_lowercase();
        if lowered == "ok" || lowered.starts_with("ok ") || lowered.starts_with("ok\t") {
            tap_seen = true;
            tap.passed += 1;
        } else if lowered == "not ok"
            || lowered.starts_with("not ok ")
            || lowered.starts_with("not ok\t")
        {
            tap_seen = true;
            tap.failed += 1;
        }
    }
    if tap_seen {
        return Some(tap);
    }

    let tokens: Vec<String> = text
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                .to_ascii_lowercase()
        })
        .collect();
    let mut counts = TestCounts::default();
    let mut counted = false;
    for window in tokens.windows(2) {
        let Ok(amount) = window[0].parse::<u64>() else {
            continue;
        };
        let word = window[1].as_str();
        if word.starts_with("passed") || word.starts_with("passing") {
            counts.passed += amount;
            counted = true;
        } else if word.starts_with("failed")
            || word.starts_with("failing")
            || word.starts_with("failure")
        {
            counts.failed += amount;
            counted = true;
        }
    }
    if counted {
        return Some(counts);
    }

    let mut standalone = TestCounts::default();
    for line in text.lines() {
        match line.trim().to_ascii_lowercase().as_str() {
            "pass" | "passed" => standalone.passed = 1,
            "fail" | "failed" => standalone.failed = 1,
            _ => {}
        }
    }
    (standalone.total() > 0).then_some(standalone)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn output(stdout: &str) -> ExecResult {
        ExecResult::ok(stdout)
    }

    #[test]
    fn test_counts_read_pytest_tap_and_standalone_summaries() {
        assert_eq!(
            parse_test_counts("collected 15 items\n== 3 failed, 12 passed in 0.2s =="),
            Some(TestCounts::new(12, 3))
        );
        assert_eq!(
            parse_test_counts("Tests: 1 failed, 2 passed"),
            Some(TestCounts::new(2, 1))
        );
        assert_eq!(
            parse_test_counts("1..3\nok 1 - a\nok 2 - b\nnot ok 3 - c\n"),
            Some(TestCounts::new(2, 1))
        );
        assert_eq!(
            parse_test_counts("everything fine\nPASS"),
            Some(TestCounts::new(1, 0))
        );
        assert_eq!(parse_test_counts("FAILED"), Some(TestCounts::new(0, 1)));
        assert_eq!(parse_test_counts("nothing to see here"), None);
        assert_eq!(parse_test_counts(""), None);
    }

    #[test]
    fn last_json_ignores_noise_and_torn_lines() {
        let text =
            "building\n{\"score\": 0.5}\nfinally\n{\"score\": 1.0, \"passed\": true}\n{\"torn";
        let value = parse_last_json(text).unwrap();
        assert_eq!(value["score"].as_f64(), Some(1.0));
        assert!(parse_last_json("no json\n[1, 2]\n\"text\"").is_none());
        assert!(parse_last_json("{\"score\": 1} trailing").is_none());
    }

    #[test]
    fn exit_code_scoring_maps_zero_to_a_full_pass() {
        let pass = parse_score(&ExecResult::ok(""), &ScoreSpec::ExitCode).unwrap();
        assert_eq!((pass.score, pass.passed), (1.0, true));
        let fail = parse_score(&ExecResult::failure("nope", 2), &ScoreSpec::ExitCode).unwrap();
        assert_eq!((fail.score, fail.passed), (0.0, false));
        assert_eq!(fail.detail, "exit code 2");
    }

    #[test]
    fn json_scoring_accepts_fractions_percentages_and_paths() {
        let fraction = parse_score(
            &output("noise\n{\"score\": 0.75, \"passed\": true}\n"),
            &ScoreSpec::Json { path: None },
        )
        .unwrap();
        assert!((fraction.score - 0.75).abs() < f64::EPSILON);
        assert!(fraction.passed);

        let percentage =
            parse_score(&output("{\"score\": 80}"), &ScoreSpec::Json { path: None }).unwrap();
        assert!((percentage.score - 0.8).abs() < f64::EPSILON);
        assert!(!percentage.passed);

        let nested = parse_score(
            &output("{\"result\": {\"score\": 1}}"),
            &ScoreSpec::Json {
                path: Some("result.score".to_owned()),
            },
        )
        .unwrap();
        assert!(nested.passed);

        assert!(parse_score(&output("no json here"), &ScoreSpec::Json { path: None }).is_err());
        assert!(parse_score(&output("{\"other\": 1}"), &ScoreSpec::Json { path: None }).is_err());
        assert!(parse_score(&output("{\"score\": 250}"), &ScoreSpec::Json { path: None }).is_err());
        assert!(parse_score(
            &output("{\"score\": 1}"),
            &ScoreSpec::Json {
                path: Some("missing".to_owned())
            }
        )
        .is_err());
    }

    #[test]
    fn summary_scoring_divides_passes_by_the_total() {
        let scored = parse_score(
            &output("== 1 failed, 3 passed in 0.1s =="),
            &ScoreSpec::TestSummary,
        )
        .unwrap();
        assert!((scored.score - 0.75).abs() < f64::EPSILON);
        assert!(!scored.passed);

        let clean = parse_score(&output("2 passed"), &ScoreSpec::TestSummary).unwrap();
        assert_eq!((clean.score, clean.passed), (1.0, true));

        assert!(parse_score(&output("silence"), &ScoreSpec::TestSummary).is_err());
    }

    #[test]
    fn auto_prefers_json_over_a_summary_and_falls_back_to_the_exit_code() {
        let both = parse_score(&output("{\"score\": 0.25}\n2 passed"), &ScoreSpec::Auto).unwrap();
        assert!((both.score - 0.25).abs() < f64::EPSILON);

        let summary = parse_score(&output("2 passed"), &ScoreSpec::Auto).unwrap();
        assert_eq!(summary.score, 1.0);

        let exit = parse_score(&ExecResult::failure("", 1), &ScoreSpec::Auto).unwrap();
        assert_eq!(exit.score, 0.0);

        let malformed_json = parse_score(
            &output("{\"other\": 1}\n1 passed, 1 failed"),
            &ScoreSpec::Auto,
        )
        .unwrap();
        assert!((malformed_json.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn a_timed_out_command_never_passes() {
        let result = ExecResult::timeout("killed after 30s");
        let scored = parse_score(&result, &ScoreSpec::ExitCode).unwrap();
        assert_eq!((scored.score, scored.passed), (0.0, false));
        assert!(scored.detail.contains("timed out"));
    }
}
