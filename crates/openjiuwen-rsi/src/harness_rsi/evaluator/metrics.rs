//! Metrics summarisation over case traces and evaluation reports.
//!
//! Ports `harness_rsi/evaluator/metrics_collector.py`: aggregate the case
//! results of one evaluation run into `summary.json`
//! (`total_cases`, `passed_cases`, `failed_cases`, `average_score`,
//! `evaluation_method`, `case_results`).
//!
//! Aggregation rules kept from Python:
//!
//! - a case counts as passed when its status is `passed`;
//! - the average is taken over **graded** scores only, so an ungraded case
//!   (a failed judge, a missing report) never drags the average down;
//! - no graded scores means `average_score: null`, never `0.0`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::evolution::evaluation::{CaseResult, EvaluationReport};

use super::runner::CaseTrace;

/// One case row in a summary.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseResultRecord {
    /// The case id.
    pub case_id: String,
    /// The case status (`passed`, `failed`, `error`).
    pub status: String,
    /// The graded score; `None` when the case was not graded.
    pub score: Option<f64>,
    /// The failure text, when the case errored.
    pub error: Option<String>,
    /// Caller annotations (the trace's artifacts).
    pub metadata: BTreeMap<String, Value>,
}

/// The `summary.json` document.
// The Python field names share the `_cases` suffix; keeping them is the point.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MetricsSummary {
    /// The number of cases in the run.
    pub total_cases: usize,
    /// How many cases reported `passed`.
    pub passed_cases: usize,
    /// How many cases did not.
    pub failed_cases: usize,
    /// The mean of the graded scores, or `None` when nothing was graded.
    pub average_score: Option<f64>,
    /// The grading method the run used.
    pub evaluation_method: String,
    /// One row per case, in input order.
    pub case_results: Vec<CaseResultRecord>,
}

impl MetricsSummary {
    /// Returns whether every case reported `passed`.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.total_cases > 0 && self.passed_cases == self.total_cases
    }
}

/// Aggregates case traces into a summary.
///
/// `passed` follows [`CaseTrace::status`]; the average considers only graded
/// scores. Each record's `metadata` is the trace's artifacts.
#[must_use]
pub fn collect_metrics(traces: &[CaseTrace], evaluation_method: &str) -> MetricsSummary {
    let records: Vec<CaseResultRecord> = traces
        .iter()
        .map(|trace| CaseResultRecord {
            case_id: trace.case_id.clone(),
            status: trace.status.clone(),
            score: trace.result.score,
            error: trace.result.error.clone(),
            metadata: trace.artifacts.clone(),
        })
        .collect();
    finish_records(records, evaluation_method)
}

/// Aggregates evolution case results into a summary.
///
/// The status is `passed`/`failed` from [`CaseResult::passed`]; the metadata is
/// empty because a [`CaseResult`] carries no artifact map.
#[must_use]
pub fn summary_from_results(results: &[CaseResult], evaluation_method: &str) -> MetricsSummary {
    let records: Vec<CaseResultRecord> = results
        .iter()
        .map(|result| CaseResultRecord {
            case_id: result.case_id.clone(),
            status: if result.passed { "passed" } else { "failed" }.to_owned(),
            score: result.score,
            error: result.error.clone(),
            metadata: BTreeMap::new(),
        })
        .collect();
    finish_records(records, evaluation_method)
}

/// Summarises several evaluation reports as one run.
///
/// Every report result becomes a row; the method is the first non-empty
/// `evaluation_method` the caller supplies (Python's
/// `_aggregate_evaluation_method` takes the first case's method).
#[must_use]
pub fn summarize_reports(reports: &[EvaluationReport], evaluation_method: &str) -> MetricsSummary {
    let results: Vec<CaseResult> = reports
        .iter()
        .flat_map(|report| report.results.iter().cloned())
        .collect();
    summary_from_results(&results, evaluation_method)
}

/// Writes a summary as pretty JSON with a trailing newline.
///
/// # Errors
///
/// Returns [`Error::Session`] when the parent directory cannot be created, when
/// the JSON cannot be rendered, or when the file cannot be written.
pub async fn write_summary(path: &Path, summary: &MetricsSummary) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::session(format!(
                    "failed to create the metrics directory `{}`: {error}",
                    parent.display()
                ))
            })?;
        }
    }
    let mut encoded = serde_json::to_string_pretty(summary).map_err(|error| {
        Error::session(format!("failed to encode the metrics summary: {error}"))
    })?;
    encoded.push('\n');
    tokio::fs::write(path, encoded).await.map_err(|error| {
        Error::session(format!(
            "failed to write the metrics summary `{}`: {error}",
            path.display()
        ))
    })
}

/// Counts the summary fields and returns the finished document.
#[allow(clippy::cast_precision_loss)]
fn finish_records(records: Vec<CaseResultRecord>, evaluation_method: &str) -> MetricsSummary {
    let total_cases = records.len();
    let passed_cases = records
        .iter()
        .filter(|record| record.status == "passed")
        .count();
    let mut sum = 0.0;
    let mut graded = 0_usize;
    for score in records.iter().filter_map(|record| record.score) {
        sum += score;
        graded += 1;
    }
    let average_score = if graded == 0 {
        None
    } else {
        Some(sum / graded as f64)
    };
    MetricsSummary {
        total_cases,
        passed_cases,
        failed_cases: total_cases - passed_cases,
        average_score,
        evaluation_method: evaluation_method.to_owned(),
        case_results: records,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(case_id: &str, status: &str, score: Option<f64>) -> CaseTrace {
        CaseTrace {
            case_id: case_id.to_owned(),
            status: status.to_owned(),
            result: CaseResult {
                case_id: case_id.to_owned(),
                score,
                ..CaseResult::default()
            },
            ..CaseTrace::default()
        }
    }

    #[test]
    fn averages_only_graded_scores() {
        let summary = collect_metrics(
            &[
                trace("a", "passed", Some(1.0)),
                trace("b", "failed", Some(0.0)),
                trace("c", "error", None),
            ],
            "llm_as_judge",
        );
        assert_eq!(summary.total_cases, 3);
        assert_eq!(summary.passed_cases, 1);
        assert_eq!(summary.failed_cases, 2);
        assert_eq!(summary.average_score, Some(0.5));
        assert_eq!(summary.evaluation_method, "llm_as_judge");
        assert!(!summary.all_passed());
    }

    #[test]
    fn nothing_graded_leaves_the_average_empty() {
        let summary = collect_metrics(&[trace("a", "error", None)], "judge");
        assert_eq!(summary.average_score, None);
        assert!(collect_metrics(&[], "judge").average_score.is_none());
    }

    #[test]
    fn reports_flatten_into_one_summary() {
        let reports = vec![
            EvaluationReport::from_results(vec![CaseResult {
                case_id: "a".to_owned(),
                passed: true,
                score: Some(1.0),
                ..CaseResult::default()
            }]),
            EvaluationReport::from_results(vec![CaseResult {
                case_id: "b".to_owned(),
                passed: false,
                score: Some(0.0),
                ..CaseResult::default()
            }]),
        ];
        let summary = summarize_reports(&reports, "judge");
        assert_eq!(summary.total_cases, 2);
        assert_eq!(summary.passed_cases, 1);
        assert_eq!(summary.average_score, Some(0.5));
    }
}
