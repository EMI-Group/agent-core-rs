//! Read-only view of the auto-harness experience store.
//!
//! Python's `ExperienceStore` (JSONL under `<workspace>/auto_harness/experience/
//! experiences.jsonl`) backs `auto-harness experience search`, `experience list`
//! and `history`. The Rust port keeps the read path — load, keyword search,
//! recency ordering, type filter — and drops the write path, because nothing in
//! the Rust port produces experience records yet.
//!
//! Parity details reproduced deliberately:
//!
//! - search tokenises on whitespace, ignores words shorter than two characters,
//!   and counts a keyword once per matched field (`topic`, `summary`, `details`
//!   in one lowercased blob — `outcome` is never searched);
//! - the score is `hits + recency`, where recency is `1` for a fresh record and
//!   decays linearly to `0` at thirty days;
//! - `list` sorts every record by timestamp, truncates to `limit`, and only then
//!   filters by `--type` (exact, case-sensitive string equality);
//! - a line without a `type` is skipped, as are blank and malformed lines.
//!
//! Intentional differences from Python: unknown JSON keys are ignored instead of
//! making the line malformed (the workspace serde convention), and the
//! user-facing messages are English rather than Chinese.
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// File name of the JSONL archive inside the experience directory.
const EXPERIENCES_FILE: &str = "experiences.jsonl";

/// Default number of search hits, matching Python's `top_k=10` in the CLI.
const SEARCH_TOP_K: usize = 10;

/// Length of the recency window in seconds (thirty days).
const RECENCY_WINDOW_SECS: f64 = 2_592_000.0;

/// One experience record, shaped like Python's `Experience` dataclass.
///
/// `type` is required (a line without it is not an experience and is skipped);
/// every other field falls back to the Python default. Unknown keys are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExperienceRow {
    /// Record kind: `optimization`, `failure` or `insight` in Python.
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    topic: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    outcome: String,
    #[serde(default)]
    details: String,
    #[serde(default)]
    pr_url: String,
    #[serde(default)]
    signal: String,
    #[serde(default)]
    strategy: String,
    #[serde(default)]
    causal_chain: String,
    #[serde(default)]
    files_changed: Vec<String>,
    #[serde(default)]
    signal_frequency: i64,
    #[serde(default)]
    id: String,
    #[serde(default)]
    timestamp: f64,
}

impl ExperienceRow {
    /// Renders the one-line text form both `search` and `list` use.
    fn display_line(&self) -> String {
        let body = if self.summary.is_empty() {
            self.outcome.as_str()
        } else {
            self.summary.as_str()
        };
        format!("[{}] {}: {body}", self.kind, self.topic)
    }
}

/// Prints the search hits for `query`.
///
/// # Errors
///
/// Returns [`Error::Config`] when the store cannot be read or created and
/// [`Error::Config`] again when the JSON form cannot be serialised.
pub(crate) async fn search(workspace: &Path, query: &str, json: bool) -> Result<()> {
    let rows = load(workspace).await?;
    let matched = search_rows(&rows, query, SEARCH_TOP_K, now_secs());
    emit(&matched, json, "no matching experiences")
}

/// Prints the most recent records, optionally filtered by type.
///
/// # Errors
///
/// See [`search`].
pub(crate) async fn list(
    workspace: &Path,
    mem_type: Option<&str>,
    limit: usize,
    json: bool,
) -> Result<()> {
    let rows = load(workspace).await?;
    let selected = list_rows(&rows, limit, mem_type);
    emit(&selected, json, "no experience records")
}

/// Reads every record of the store, creating the directory first.
///
/// Python's store constructor calls `mkdir(parents=True, exist_ok=True)`, so a
/// first read on a fresh workspace creates the empty directory; a missing
/// `experiences.jsonl` is simply an empty store.
async fn load(workspace: &Path) -> Result<Vec<ExperienceRow>> {
    let dir = workspace.join("auto_harness").join("experience");
    tokio::fs::create_dir_all(&dir).await.map_err(|err| {
        Error::config(format!(
            "cannot create experience directory {}: {err}",
            dir.display()
        ))
    })?;
    let path = dir.join(EXPERIENCES_FILE);
    let text = match tokio::fs::read_to_string(&path).await {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(Error::config(format!(
                "cannot read experience store {}: {err}",
                path.display()
            )));
        }
    };
    Ok(parse_rows(&text))
}

/// Parses the JSONL text, skipping blank and malformed lines.
fn parse_rows(text: &str) -> Vec<ExperienceRow> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<ExperienceRow>(line) {
            Ok(row) => rows.push(row),
            Err(err) => tracing::warn!(
                line = index + 1,
                %err,
                "skipping malformed experience line"
            ),
        }
    }
    rows
}

/// Keyword search: hits plus recency, best first, `top_k` records.
///
/// `now` is injected so the ordering is testable without a clock.
fn search_rows(rows: &[ExperienceRow], query: &str, top_k: usize, now: f64) -> Vec<ExperienceRow> {
    let keywords: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|word| word.chars().count() >= 2)
        .map(str::to_owned)
        .collect();
    if keywords.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(f64, ExperienceRow)> = rows
        .iter()
        .filter_map(|row| {
            let hits = count_hits(&keywords, row);
            if hits == 0 {
                return None;
            }
            // A hit count is bounded by the number of query keywords, so the
            // conversion to the f64 score is lossless in practice.
            #[allow(clippy::cast_precision_loss)]
            let score = hits as f64 + recency(row.timestamp, now);
            Some((score, row.clone()))
        })
        .collect();
    // Stable sort, so records with the same score keep file order (Python's
    // `list.sort(key=..., reverse=True)` is stable in the same way).
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    scored.into_iter().take(top_k).map(|(_, row)| row).collect()
}

/// Counts keyword hits in the searchable fields of one record.
fn count_hits(keywords: &[String], row: &ExperienceRow) -> usize {
    let blob = format!("{} {} {}", row.topic, row.summary, row.details).to_lowercase();
    keywords
        .iter()
        .filter(|keyword| blob.contains(*keyword))
        .count()
}

/// Recency bonus in `[0, 1]`, decaying linearly over thirty days.
fn recency(timestamp: f64, now: f64) -> f64 {
    let age = (now - timestamp).max(0.0);
    if age >= RECENCY_WINDOW_SECS {
        0.0
    } else {
        1.0 - age / RECENCY_WINDOW_SECS
    }
}

/// Recent records: newest first, truncated to `limit`, then filtered by type.
fn list_rows(rows: &[ExperienceRow], limit: usize, mem_type: Option<&str>) -> Vec<ExperienceRow> {
    let mut sorted = rows.to_vec();
    // Stable: equal timestamps keep file order.
    sorted.sort_by(|left, right| right.timestamp.total_cmp(&left.timestamp));
    sorted.truncate(limit);
    if let Some(kind) = mem_type {
        sorted.retain(|row| row.kind == kind);
    }
    sorted
}

/// Writes the selected records as text lines or as a pretty JSON array.
fn emit(rows: &[ExperienceRow], json: bool, empty: &str) -> Result<()> {
    if json {
        let text = serde_json::to_string_pretty(rows)
            .map_err(|err| Error::config(format!("cannot serialise experiences: {err}")))?;
        println!("{text}");
        return Ok(());
    }
    if rows.is_empty() {
        println!("{empty}");
        return Ok(());
    }
    for row in rows {
        println!("{}", row.display_line());
    }
    Ok(())
}

/// Wall-clock seconds since the Unix epoch, or `0.0` before it.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

#[cfg(test)]
mod tests {
    //! Store tests: JSONL parsing, search scoring, ordering and the type filter.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_test_support::TempWorkspace;

    use super::{list_rows, parse_rows, recency, search_rows, ExperienceRow, RECENCY_WINDOW_SECS};

    /// A fixed clock for deterministic recency.
    const NOW: f64 = 1_700_000_000.0;

    /// Builds one record with the fields the tests care about.
    fn row(kind: &str, topic: &str, summary: &str, details: &str, timestamp: f64) -> ExperienceRow {
        ExperienceRow {
            kind: kind.to_owned(),
            topic: topic.to_owned(),
            summary: summary.to_owned(),
            outcome: String::new(),
            details: details.to_owned(),
            pr_url: String::new(),
            signal: String::new(),
            strategy: String::new(),
            causal_chain: String::new(),
            files_changed: Vec::new(),
            signal_frequency: 0,
            id: String::new(),
            timestamp,
        }
    }

    #[test]
    fn parsing_skips_blank_and_malformed_lines_but_keeps_valid_ones() {
        let text = concat!(
            "\n",
            "{\"type\": \"insight\", \"topic\": \"one\", \"timestamp\": 1.0}\n",
            "{not json}\n",
            "{\"topic\": \"no type\"}\n",
            "  \n",
            "{\"type\": \"failure\", \"topic\": \"two\", \"unknown_key\": 7}\n",
        );
        let rows = parse_rows(text);
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0].kind, "insight");
        assert_eq!(rows[0].topic, "one");
        assert_eq!(rows[1].kind, "failure");
        assert_eq!(rows[1].topic, "two");
        assert!(rows[1].files_changed.is_empty(), "default files_changed");
    }

    #[test]
    fn recency_is_full_for_fresh_records_and_zero_after_thirty_days() {
        assert!((recency(NOW, NOW) - 1.0).abs() < f64::EPSILON);
        assert!(recency(NOW - 86_400.0, NOW) < 1.0);
        assert!(recency(NOW - 86_400.0, NOW) > 0.9);
        assert!(recency(NOW - RECENCY_WINDOW_SECS, NOW).abs() < f64::EPSILON);
        assert!(recency(NOW - 2.0 * RECENCY_WINDOW_SECS, NOW).abs() < f64::EPSILON);
        // A record stamped in the future is not penalised.
        assert!((recency(NOW + 100.0, NOW) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn search_needs_a_keyword_of_at_least_two_characters() {
        let rows = vec![row("insight", "compile speed", "", "", NOW)];
        assert!(
            search_rows(&rows, "", 10, NOW).is_empty(),
            "an empty query has no keywords"
        );
        assert!(
            search_rows(&rows, "a i", 10, NOW).is_empty(),
            "single-character words are dropped"
        );
        assert_eq!(search_rows(&rows, "compile", 10, NOW).len(), 1);
    }

    #[test]
    fn search_counts_each_keyword_occurrence_and_ignores_outcome() {
        let mut with_outcome = row("insight", "cache", "", "", NOW);
        with_outcome.outcome = "compile".to_owned();
        assert!(
            search_rows(&[with_outcome.clone()], "compile", 10, NOW).is_empty(),
            "`outcome` must not be searched"
        );

        // Python counts one hit per keyword entry, so repeating a keyword in the
        // query doubles the hit count for a record that contains it.
        let repeated = vec!["cache".to_owned(), "cache".to_owned()];
        assert_eq!(super::count_hits(&repeated, &with_outcome), 2);
        assert_eq!(super::count_hits(&repeated[..1], &with_outcome), 1);
        // Two hits plus full recency is the score the search sort uses.
        assert_eq!(
            search_rows(&[with_outcome], "cache cache", 10, NOW).len(),
            1
        );
    }

    #[test]
    fn search_keeps_file_order_for_equal_scores() {
        let mut first = row("insight", "cache", "", "", NOW);
        first.id = "first".to_owned();
        let mut second = row("insight", "cache", "", "", NOW);
        second.id = "second".to_owned();
        let ranked = search_rows(&[first, second], "cache cache", 10, NOW);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].id, "first");
        assert_eq!(ranked[1].id, "second");
    }

    #[test]
    fn search_orders_by_hits_then_recency_and_honours_top_k() {
        let rows = vec![
            row("insight", "alpha", "", "", NOW - RECENCY_WINDOW_SECS),
            row("insight", "alpha beta", "alpha", "", NOW),
            row("failure", "unrelated", "", "", NOW),
        ];
        let ranked = search_rows(&rows, "alpha", 10, NOW);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].topic, "alpha beta");
        assert_eq!(ranked[1].topic, "alpha");
        assert_eq!(search_rows(&rows, "alpha", 1, NOW).len(), 1);
    }

    #[test]
    fn list_sorts_newest_first_and_applies_the_filter_after_the_limit() {
        let rows = vec![
            row("insight", "old", "", "", 10.0),
            row("failure", "newest", "", "", 30.0),
            row("insight", "middle", "", "", 20.0),
        ];
        let recent = list_rows(&rows, 2, None);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].topic, "newest");
        assert_eq!(recent[1].topic, "middle");
        // Python truncates first and filters second: a one-record window holding
        // only the failure comes back empty even though `middle` is an insight.
        assert!(
            list_rows(&rows, 1, Some("insight")).is_empty(),
            "the filter runs after the limit"
        );
        assert_eq!(list_rows(&rows, 3, Some("insight")).len(), 2);
        // The filter is exact and case-sensitive.
        assert!(
            list_rows(&rows, 3, Some("Insight")).is_empty(),
            "the type filter is case-sensitive"
        );
    }

    #[test]
    fn text_line_falls_back_to_outcome_only_when_summary_is_empty() {
        let mut entry = row("insight", "topic", "", "", NOW);
        entry.outcome = "fallback".to_owned();
        assert_eq!(entry.display_line(), "[insight] topic: fallback");
        entry.summary = "summary".to_owned();
        assert_eq!(entry.display_line(), "[insight] topic: summary");
        entry.summary.clear();
        entry.outcome.clear();
        assert_eq!(entry.display_line(), "[insight] topic: ");
    }

    #[test]
    fn reading_a_fresh_workspace_creates_the_directory_and_returns_nothing() {
        let workspace = TempWorkspace::new().unwrap();
        let rows = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(super::load(&workspace.workspace()))
            .unwrap();
        assert!(rows.is_empty(), "a fresh workspace has no records");
        assert!(workspace
            .workspace()
            .join("auto_harness")
            .join("experience")
            .is_dir());
    }

    #[test]
    fn reading_a_store_keeps_file_order() {
        let workspace = TempWorkspace::new().unwrap();
        let dir = workspace
            .workspace()
            .join("auto_harness")
            .join("experience");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("experiences.jsonl"),
            "{\"type\": \"insight\", \"topic\": \"a\", \"timestamp\": 1.0}\n\
             {\"type\": \"failure\", \"topic\": \"b\", \"timestamp\": 2.0}\n",
        )
        .unwrap();
        let rows = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(super::load(&workspace.workspace()))
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].topic, "a");
        assert_eq!(rows[1].topic, "b");
    }
}
