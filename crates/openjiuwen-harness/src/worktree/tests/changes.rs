//! Change-summary scenarios: porcelain parsing, diffstat parsing, the documented
//! git call order, bounding, degradation rules, and deterministic rendering.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use openjiuwen_common::error::ErrorKind;

use super::ScriptedGitRunner;
use crate::worktree::{
    change_summary, parse_diff_stat, render_change_summary, status_paths, WorktreeChangeSummary,
    SUMMARY_MAX_CHARS, SUMMARY_MAX_FILES, SUMMARY_MAX_LINES,
};

/// The worktree path every summary test uses.
fn worktree() -> PathBuf {
    PathBuf::from("/repo/.worktrees/feature")
}

const STATUS: &str = " M src/lib.rs\n?? new.txt\nR  old.txt -> new.txt\n";

#[tokio::test]
async fn change_summary_issues_the_documented_calls_in_order() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok(STATUS);
    runner.push_ok(" src/lib.rs | 2 +-\n 1 file changed, 2 insertions(+)\n");
    runner.push_ok("3\n");
    runner.push_ok("deadbeef\n");

    let summary = change_summary(&runner, &worktree(), Some("HEAD~2"))
        .await
        .expect("summary");

    assert_eq!(summary.changed_files, 3);
    assert_eq!(summary.files, ["src/lib.rs", "new.txt", "new.txt"]);
    assert_eq!(summary.commits, 3);
    assert_eq!(summary.unpushed, Some(true));
    assert_eq!(
        summary.diffstat,
        "src/lib.rs | 2 +-\n1 file changed, 2 insertions(+)"
    );
    assert!(!summary.truncated);
    assert!(summary.is_dirty());

    assert_eq!(
        runner.commands(),
        [
            "status --porcelain",
            "diff --stat HEAD~2..HEAD",
            "rev-list --count HEAD~2..HEAD",
            "rev-list --max-count=1 HEAD --not --remotes",
        ]
    );
    for call in runner.calls() {
        assert_eq!(call.cwd, worktree());
    }
    assert_eq!(runner.remaining(), 0);
}

#[tokio::test]
async fn without_a_base_only_status_and_the_unpushed_probe_run() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok(" M a.rs\n");
    runner.push_ok("");

    let summary = change_summary(&runner, &worktree(), None)
        .await
        .expect("summary");

    assert_eq!(summary.changed_files, 1);
    assert_eq!(summary.files, ["a.rs"]);
    assert_eq!(summary.commits, 0);
    assert_eq!(summary.unpushed, Some(false));
    assert_eq!(summary.diffstat.len(), 0);
    assert_eq!(
        runner.commands(),
        [
            "status --porcelain",
            "rev-list --max-count=1 HEAD --not --remotes",
        ]
    );
}

#[tokio::test]
async fn a_failing_unpushed_probe_degrades_to_none() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok(" M a.rs\n");
    runner.push_fail("fatal: no remotes configured");

    let summary = change_summary(&runner, &worktree(), None)
        .await
        .expect("summary");

    assert_eq!(summary.unpushed, None);
    assert_eq!(summary.changed_files, 1);
    assert_eq!(summary.commits, 0);
}

#[tokio::test]
async fn a_failing_commit_count_degrades_to_zero_but_the_probe_still_runs() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok(" M a.rs\n");
    runner.push_ok(" a.rs | 1 +\n 1 file changed, 1 insertion(+)\n");
    runner.push_fail("fatal: bad revision");
    runner.push_ok("feedface\n");

    let summary = change_summary(&runner, &worktree(), Some("base"))
        .await
        .expect("summary");

    assert_eq!(summary.commits, 0);
    assert_eq!(summary.unpushed, Some(true));
    assert_eq!(summary.changed_files, 1);
    assert_eq!(
        runner.commands(),
        [
            "status --porcelain",
            "diff --stat base..HEAD",
            "rev-list --count base..HEAD",
            "rev-list --max-count=1 HEAD --not --remotes",
        ]
    );
}

#[tokio::test]
async fn an_unparsable_commit_count_degrades_to_zero() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok("");
    runner.push_ok("");
    runner.push_ok("not-a-number");
    runner.push_ok("");

    let summary = change_summary(&runner, &worktree(), Some("base"))
        .await
        .expect("summary");

    assert_eq!(summary.commits, 0);
    assert_eq!(summary.unpushed, Some(false));
    assert!(summary.is_clean());
}

#[tokio::test]
async fn a_failing_status_is_a_typed_error() {
    let runner = ScriptedGitRunner::new();
    runner.push_fail("fatal: not a git repository");

    let error = change_summary(&runner, &worktree(), None)
        .await
        .expect_err("status failure");

    assert_eq!(error.kind(), ErrorKind::Tool);
    assert_eq!(runner.commands(), ["status --porcelain"]);
}

#[tokio::test]
async fn a_failing_diff_is_a_typed_error() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok(" M a.rs\n");
    runner.push_fail("fatal: bad revision");

    let error = change_summary(&runner, &worktree(), Some("missing"))
        .await
        .expect_err("diff failure");

    assert_eq!(error.kind(), ErrorKind::Tool);
    assert_eq!(
        runner.commands(),
        ["status --porcelain", "diff --stat missing..HEAD"]
    );
}

#[tokio::test]
async fn files_and_diffstat_are_bounded_and_mark_truncation() {
    let runner = ScriptedGitRunner::new();
    let status: String = (0..=SUMMARY_MAX_FILES)
        .map(|index| format!(" M file{index}.rs"))
        .collect::<Vec<_>>()
        .join("\n");
    let diffstat: String = (0..SUMMARY_MAX_LINES + 10)
        .map(|index| format!(" file{index}.rs | {index} +"))
        .collect::<Vec<_>>()
        .join("\n");
    runner.push_ok(&status);
    runner.push_ok(&diffstat);
    runner.push_ok("1\n");
    runner.push_ok("");

    let summary = change_summary(&runner, &worktree(), Some("base"))
        .await
        .expect("summary");

    assert_eq!(summary.changed_files, SUMMARY_MAX_FILES + 1);
    assert_eq!(summary.files.len(), SUMMARY_MAX_FILES);
    assert_eq!(summary.files[0], "file0.rs");
    assert_eq!(summary.files[SUMMARY_MAX_FILES - 1], "file19.rs");
    assert_eq!(
        summary.diffstat.lines().count(),
        SUMMARY_MAX_LINES,
        "diffstat is clipped to the line bound"
    );
    assert!(summary.truncated);
}

#[tokio::test]
async fn a_single_huge_diffstat_line_is_char_bounded() {
    let runner = ScriptedGitRunner::new();
    runner.push_ok("");
    runner.push_ok(&"z".repeat(SUMMARY_MAX_CHARS + 50));
    runner.push_ok("0\n");
    runner.push_ok("");

    let summary = change_summary(&runner, &worktree(), Some("base"))
        .await
        .expect("summary");

    assert!(summary.truncated);
    assert_eq!(summary.diffstat.chars().count(), SUMMARY_MAX_CHARS);
}

#[test]
fn status_paths_handles_every_porcelain_line_shape() {
    let text = " M src/lib.rs\n?? untracked.txt\nR  old.txt -> new.txt\nA  added.rs\n\n";
    assert_eq!(
        status_paths(text),
        ["src/lib.rs", "untracked.txt", "new.txt", "added.rs"]
    );

    // Short and malformed lines keep their trimmed remainder.
    assert_eq!(status_paths("M"), ["M"]);
    assert_eq!(status_paths(" M"), ["M"]);
    assert_eq!(status_paths("  \n\t\n"), Vec::<String>::new());
}

#[test]
fn diff_stat_parsing_drops_blank_lines() {
    let text = " src/a.rs | 2 +-\n\n 1 file changed, 1 insertion(+)\n";
    assert_eq!(
        parse_diff_stat(text),
        ["src/a.rs | 2 +-", "1 file changed, 1 insertion(+)"]
    );
    assert_eq!(parse_diff_stat("   \n\n").len(), 0);
}

#[test]
fn rendering_is_bilingual_deterministic_and_marks_unknown_state() {
    let summary = WorktreeChangeSummary {
        changed_files: 2,
        files: vec!["a.rs".to_owned(), "b.rs".to_owned()],
        commits: 3,
        unpushed: Some(true),
        diffstat: "1 file changed, 2 insertions(+)".to_owned(),
        truncated: false,
    };
    assert_eq!(
        render_change_summary(&summary, "en"),
        "Changed files: 2\n  - a.rs\n  - b.rs\nCommits since base: 3\nUnpushed commits: yes\nDiffstat:\n  1 file changed, 2 insertions(+)"
    );
    let chinese = render_change_summary(&summary, "zh");
    assert!(chinese.contains("变更文件: 2"));
    assert!(chinese.contains("未推送提交: 是"));
    // "en" is exact; any other tag renders Chinese.
    assert_eq!(render_change_summary(&summary, "en-US"), chinese);

    let unknown = WorktreeChangeSummary {
        unpushed: None,
        ..WorktreeChangeSummary::default()
    };
    assert!(render_change_summary(&unknown, "en").contains("Unpushed commits: unknown"));
    assert!(render_change_summary(&unknown, "zh").contains("未推送提交: 未知"));
    assert!(!render_change_summary(&unknown, "en").contains("Diffstat:"));
}
