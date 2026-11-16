//! Report-shaped HTML detection and row pruning.
//!
//! A generated test report (`pytest-html`, `results-table`, or two tables plus two
//! report markers) is rendered like any other page, then pruned: once the
//! document holds a failure row, the "passed"/"ok" rows are dropped so the failure
//! stays in front of the model.
//! This mirrors the Python `BeautifulSoup` fallback and is a post-render pass, so
//! it never changes the parsing rules.
use crate::processor::rule_compression::common::contains_word;

/// Table rows that keep a report row.
const REPORT_ROW_KEEP_WORDS: &[&str] = &[
    "fail",
    "failed",
    "failure",
    "error",
    "broken",
    "skipped",
    "xfailed",
    "xpassed",
    "rerun",
    "unexpected",
    "assertionerror",
    "traceback",
];
/// Table rows a report drops once something failed.
const REPORT_ROW_DROP_WORDS: &[&str] = &["pass", "passed", "ok"];

/// Whether `html` looks like a generated test report.
pub(super) fn looks_like_report_html(html: &str) -> bool {
    let lowered: String = html.chars().take(20_000).collect::<String>().to_lowercase();
    if lowered.contains("pytest-html") || lowered.contains("results-table") {
        return true;
    }
    let tables = lowered.matches("<table").count();
    let markers = [
        "pytest-html",
        "results-table",
        "environment",
        "test results",
        "tests took",
        "report generated",
    ]
    .iter()
    .filter(|marker| lowered.contains(*marker))
    .count();
    tables >= 2 && markers >= 2
}

/// Drops "passed" report rows once the table also holds a failure row.
pub(super) fn prune_report_rows(blocks: &mut [String]) -> usize {
    let has_keep = blocks.iter().any(|block| {
        block
            .lines()
            .any(|line| contains_word(&line.to_lowercase(), REPORT_ROW_KEEP_WORDS))
    });
    if !has_keep {
        return 0;
    }
    let mut removed = 0_usize;
    for block in blocks.iter_mut() {
        if !block.contains('|') {
            continue;
        }
        let kept: Vec<&str> = block
            .lines()
            .filter(|line| {
                let lowered = line.to_lowercase();
                if contains_word(&lowered, REPORT_ROW_KEEP_WORDS) {
                    return true;
                }
                let droppable = contains_word(&lowered, REPORT_ROW_DROP_WORDS)
                    || lowered.contains("::test")
                    || lowered.starts_with("tests/");
                if droppable {
                    removed += 1;
                    return false;
                }
                true
            })
            .collect();
        *block = kept.join("\n");
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_markers_are_recognised() {
        assert!(looks_like_report_html(
            "<html><body class=\"pytest-html\"><table></table></body></html>"
        ));
        assert!(!looks_like_report_html(
            "<html><body><p>hi</p></body></html>"
        ));
    }

    #[test]
    fn passed_rows_are_dropped_once_a_failure_exists() {
        let mut blocks = vec![
            "| test | result |\n| --- | --- |\n| tests/a.py::test_x | failed |\n\
             | tests/b.py::test_y | passed |"
                .to_owned(),
        ];
        let removed = prune_report_rows(&mut blocks);
        assert_eq!(removed, 1);
        assert!(blocks[0].contains("test_x"));
        assert!(!blocks[0].contains("test_y"));
    }
}
