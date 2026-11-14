//! Change summaries for a worktree: uncommitted files, commits since the base,
//! unpushed commits, and a bounded diffstat.
//!
//! Port of the change-detection half of
//! `openjiuwen/harness/tools/worktree/{git,manager,models}.py`: Python's
//! `count_changes` pairs `git status --porcelain` with
//! `git rev-list --count <base>..HEAD`, and `cleanup.py` refuses to remove a
//! worktree when `has_unpushed_commits` is `None` or `true` (fail-closed). This
//! port additionally renders a bounded, model-facing summary — the Python
//! `DiffSummaryRail` logs a stat-level diff, while here
//! [`render_change_summary`] produces deterministic text for a tool payload.
//!
//! # Call order
//!
//! [`change_summary`] issues exactly these git commands, in this order:
//!
//! 1. `git status --porcelain` in the worktree — always;
//! 2. `git diff --stat <base>..HEAD` — only when a base is supplied;
//! 3. `git rev-list --count <base>..HEAD` — only when a base is supplied;
//! 4. `git rev-list --max-count=1 HEAD --not --remotes` — always.
//!
//! A failing `status` or `diff` propagates the typed git error, because the
//! caller cannot claim a clean worktree it failed to inspect. A failing count or
//! unpushed probe degrades instead: `commits` stays `0`, `unpushed` stays `None`
//! — and `None` is the fail-closed signal that cleanup must not treat the
//! worktree as clean.
//!
//! Everything is bounded so the summary stays cheap to put into a prompt:
//! [`SUMMARY_MAX_FILES`] file names, [`SUMMARY_MAX_LINES`] diffstat lines and
//! [`SUMMARY_MAX_CHARS`] diffstat characters. Hitting any bound sets
//! [`WorktreeChangeSummary::truncated`].
use std::path::Path;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use super::git::{self, GitRunner};

/// Maximum number of file names kept in a summary (`changed_files` still counts
/// all of them).
pub const SUMMARY_MAX_FILES: usize = 20;

/// Maximum number of diffstat lines kept in a summary.
pub const SUMMARY_MAX_LINES: usize = 40;

/// Maximum number of diffstat characters kept in a summary.
pub const SUMMARY_MAX_CHARS: usize = 2_000;

/// Summary of the uncommitted and unpushed work in one worktree.
///
/// `changed_files` counts every porcelain entry while `files` holds at most
/// [`SUMMARY_MAX_FILES`] cleaned paths; `is_dirty` therefore works even when the
/// list was truncated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeChangeSummary {
    /// Number of uncommitted changed paths (untouched by the file bound).
    pub changed_files: usize,
    /// Cleaned changed paths, bounded to [`SUMMARY_MAX_FILES`].
    pub files: Vec<String>,
    /// Commits between the base reference and `HEAD`; `0` when unknown.
    pub commits: u64,
    /// Whether commits exist that no remote has; `None` when the probe failed.
    pub unpushed: Option<bool>,
    /// Bounded `git diff --stat` output, empty when no base was supplied.
    pub diffstat: String,
    /// Whether a file, line or character bound was hit.
    pub truncated: bool,
}

impl WorktreeChangeSummary {
    /// Whether the worktree holds uncommitted changes or new commits.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.changed_files > 0 || self.commits > 0
    }

    /// Whether the worktree has neither uncommitted changes nor new commits.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        !self.is_dirty()
    }
}

/// Parses `git status --porcelain` output into cleaned changed paths.
///
/// The two status columns and their separating space are stripped, so ` M
/// src/lib.rs` yields `src/lib.rs` and `?? new.txt` yields `new.txt`. A rename
/// or copy line (`R  old -> new`) yields the destination path. Blank lines are
/// skipped, and short or malformed lines keep their trimmed remainder instead of
/// being dropped.
#[must_use]
pub fn status_paths(text: &str) -> Vec<String> {
    text.lines().filter_map(porcelain_path).collect()
}

/// The path component of one porcelain line, if the line is not blank.
fn porcelain_path(line: &str) -> Option<String> {
    if line.trim().is_empty() {
        return None;
    }
    let rest = match third_char_boundary(line) {
        Some(index) => &line[index..],
        None => line.trim_start(),
    };
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    let path = rest.rsplit(" -> ").next().unwrap_or(rest).trim();
    (!path.is_empty()).then(|| path.to_owned())
}

/// Byte index just past the second status column plus one separator space.
fn third_char_boundary(line: &str) -> Option<usize> {
    let mut chars = line.char_indices();
    let _first = chars.next()?;
    let _second = chars.next()?;
    let (index, third) = chars.next()?;
    Some(if third == ' ' { index + 1 } else { index })
}

/// Parses `git diff --stat` output into its non-blank, trimmed lines.
#[must_use]
pub fn parse_diff_stat(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Collects the change summary of the worktree at `worktree`.
///
/// The git call order is documented on the module. `base` is a reference the
/// worktree branch started from (a SHA or `HEAD~n` form); without it there is no
/// commit range to diff or count, so `commits` is `0` and `diffstat` is empty.
///
/// # Errors
///
/// Returns a typed `Error::tool` when `git status` or `git diff` fails, or when
/// the runner itself cannot be invoked. A failing `rev-list` probe degrades to
/// `commits == 0` / `unpushed == None`.
pub async fn change_summary(
    runner: &dyn GitRunner,
    worktree: &Path,
    base: Option<&str>,
) -> Result<WorktreeChangeSummary> {
    let status = runner.run(&["status", "--porcelain"], worktree).await?;
    let paths = status_paths(&status.stdout);
    let changed_files = paths.len();
    let mut truncated = changed_files > SUMMARY_MAX_FILES;
    let files: Vec<String> = paths.into_iter().take(SUMMARY_MAX_FILES).collect();

    let mut commits = 0;
    let mut diffstat = String::new();
    if let Some(base) = base {
        diffstat = git::diff_stat(runner, worktree, Some(base)).await?;
        commits = git::rev_list_count(runner, worktree, &format!("{base}..HEAD"))
            .await?
            .unwrap_or(0);
    }
    let unpushed = git::has_unpushed_commits(runner, worktree).await?;

    let (diffstat, diffstat_truncated) = bound_diffstat(&diffstat);
    truncated |= diffstat_truncated;
    Ok(WorktreeChangeSummary {
        changed_files,
        files,
        commits,
        unpushed,
        diffstat,
        truncated,
    })
}

/// Bounds diffstat text to [`SUMMARY_MAX_LINES`] lines and
/// [`SUMMARY_MAX_CHARS`] characters, reporting whether anything was dropped.
fn bound_diffstat(text: &str) -> (String, bool) {
    let lines = parse_diff_stat(text);
    let mut truncated = lines.len() > SUMMARY_MAX_LINES;
    let mut bounded = lines
        .into_iter()
        .take(SUMMARY_MAX_LINES)
        .collect::<Vec<_>>()
        .join("\n");
    if bounded.chars().count() > SUMMARY_MAX_CHARS {
        truncated = true;
        bounded = bounded.chars().take(SUMMARY_MAX_CHARS).collect();
    }
    (bounded, truncated)
}

/// Renders a summary as deterministic text for a model or a log line.
///
/// `language == "en"` selects the English labels; every other value selects the
/// Chinese ones (the harness prompt language contract). The output has one
/// line per fact: the changed-file count, up to [`SUMMARY_MAX_FILES`] paths, the
/// commit count, the unpushed state (`yes`/`no`/`unknown`), and the diffstat
/// when non-empty. A truncated summary ends with a `...` marker, and the whole
/// rendering is capped at [`SUMMARY_MAX_CHARS`]. No clock or environment is
/// read.
#[must_use]
pub fn render_change_summary(summary: &WorktreeChangeSummary, language: &str) -> String {
    let english = language == "en";
    let mut lines = Vec::new();
    lines.push(if english {
        format!("Changed files: {}", summary.changed_files)
    } else {
        format!("变更文件: {}", summary.changed_files)
    });
    for file in summary.files.iter().take(SUMMARY_MAX_FILES) {
        lines.push(format!("  - {file}"));
    }
    lines.push(if english {
        format!("Commits since base: {}", summary.commits)
    } else {
        format!("自基准以来的提交: {}", summary.commits)
    });
    let unpushed = match (summary.unpushed, english) {
        (Some(true), true) => "yes",
        (Some(false), true) => "no",
        (None, true) => "unknown",
        (Some(true), false) => "是",
        (Some(false), false) => "否",
        (None, false) => "未知",
    };
    lines.push(if english {
        format!("Unpushed commits: {unpushed}")
    } else {
        format!("未推送提交: {unpushed}")
    });
    if !summary.diffstat.trim().is_empty() {
        lines.push(if english {
            "Diffstat:".to_owned()
        } else {
            "差异统计:".to_owned()
        });
        for line in summary.diffstat.lines() {
            lines.push(format!("  {line}"));
        }
    }
    if summary.truncated {
        lines.push("...".to_owned());
    }
    bound_chars(&lines.join("\n"), SUMMARY_MAX_CHARS)
}

/// Caps `text` at `max_chars` characters, keeping a `...` marker when it clips.
fn bound_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let keep = max_chars.saturating_sub(3);
    let mut bounded: String = text.chars().take(keep).collect();
    bounded.push_str("...");
    bounded
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn status_paths_strips_columns_and_resolves_renames() {
        let text = " M src/lib.rs\n?? new.txt\nR  old.txt -> new.txt\nA  added.rs\n\n";
        assert_eq!(
            status_paths(text),
            ["src/lib.rs", "new.txt", "new.txt", "added.rs"]
        );
    }

    #[test]
    fn status_paths_tolerates_short_and_malformed_lines() {
        assert_eq!(status_paths("M"), ["M"]);
        assert_eq!(status_paths(" M"), ["M"]);
        assert_eq!(status_paths("  \n\t\n"), Vec::<String>::new());
        assert_eq!(status_paths(" M  spaced name"), ["spaced name"]);
    }

    #[test]
    fn diff_stat_lines_are_trimmed_and_blank_free() {
        let text = " src/a.rs | 2 +-\n\n 1 file changed, 1 insertion(+), 1 deletion(-)\n";
        assert_eq!(
            parse_diff_stat(text),
            [
                "src/a.rs | 2 +-",
                "1 file changed, 1 insertion(+), 1 deletion(-)"
            ]
        );
        assert_eq!(parse_diff_stat("").len(), 0);
    }

    #[test]
    fn bounds_clip_diffstat_by_lines_then_chars() {
        let many = (0..=SUMMARY_MAX_LINES)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (bounded, truncated) = bound_diffstat(&many);
        assert!(truncated);
        assert_eq!(bounded.lines().count(), SUMMARY_MAX_LINES);

        let long = "x".repeat(SUMMARY_MAX_CHARS + 10);
        let (bounded, truncated) = bound_diffstat(&long);
        assert!(truncated);
        assert_eq!(bounded.chars().count(), SUMMARY_MAX_CHARS);
    }

    #[test]
    fn rendering_is_bilingual_and_deterministic() {
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
        assert!(render_change_summary(&summary, "zh").contains("未推送提交: 是"));

        let unknown = WorktreeChangeSummary {
            unpushed: None,
            ..WorktreeChangeSummary::default()
        };
        assert!(render_change_summary(&unknown, "en").contains("Unpushed commits: unknown"));
        assert!(render_change_summary(&unknown, "zh").contains("未推送提交: 未知"));

        let none = WorktreeChangeSummary {
            unpushed: Some(false),
            ..WorktreeChangeSummary::default()
        };
        assert!(render_change_summary(&none, "en").contains("Unpushed commits: no"));
        assert!(render_change_summary(&none, "zh").contains("未推送提交: 否"));
    }

    #[test]
    fn rendering_marks_truncation_and_stays_bounded() {
        let summary = WorktreeChangeSummary {
            changed_files: 1,
            files: vec!["a.rs".to_owned()],
            truncated: true,
            ..WorktreeChangeSummary::default()
        };
        let rendered = render_change_summary(&summary, "en");
        assert!(rendered.ends_with("..."));
        assert!(render_change_summary(&summary, "zh").ends_with("..."));

        let huge = WorktreeChangeSummary {
            changed_files: 1,
            files: vec!["a.rs".to_owned()],
            diffstat: "y".repeat(SUMMARY_MAX_CHARS * 2),
            ..WorktreeChangeSummary::default()
        };
        let rendered = render_change_summary(&huge, "en");
        assert!(rendered.chars().count() <= SUMMARY_MAX_CHARS);
        assert!(rendered.ends_with("..."));
    }

    #[test]
    fn dirty_and_clean_follow_files_and_commits() {
        assert!(WorktreeChangeSummary::default().is_clean());
        assert!(!WorktreeChangeSummary {
            changed_files: 1,
            ..WorktreeChangeSummary::default()
        }
        .is_clean());
        assert!(WorktreeChangeSummary {
            commits: 2,
            ..WorktreeChangeSummary::default()
        }
        .is_dirty());
    }
}
