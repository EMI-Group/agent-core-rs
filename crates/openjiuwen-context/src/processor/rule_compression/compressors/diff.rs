//! Diff compression: keep the interesting files, hunks and lines.
//!
//! Ports `rule_compression/compressors/diff_compressor.py`. A unified diff is
//! parsed into a preamble, files (`diff --git` / `diff --cc` / `diff --combined`
//! headers) and hunks (`@@` / `@@@` headers); combined diffs work because only the
//! header prefixes are matched.
//!
//! - Files: all are kept when there are at most `diff_max_files`; otherwise the
//!   ones with the most changes win.
//! - Hunks: all are kept when there are at most `diff_max_hunks_per_file`;
//!   otherwise the first, the last and the highest-scoring ones.
//! - Changed lines: at most `diff_max_changed_lines_per_hunk` per hunk and
//!   `diff_max_changed_lines_total` overall, priority words (error, fail, todo,
//!   fixme, security, ...) first.
//! - Context lines: only within `diff_max_context_lines` of a retained change.
//!
//! Omitted lines become `[N unchanged/context diff lines omitted]` and
//! `[N changed diff lines omitted]`; a trailing
//! `[F files changed, +A -D lines, H hunks omitted, X files omitted]` always
//! closes the candidate.
//!
//! `details` are computed for every parsed diff, including candidates the
//! savings gate rejected, and `should_offload_original` is
//! `compressed_line_count < 0.8 * original_line_count`. The two early exits (a
//! diff shorter than `diff_min_lines`, and unparsable output) report no
//! `details` at all, matching Python.
use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::{contains_word, meets_savings_ratio};
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use serde_json::json;
use std::collections::BTreeSet;
/// Words that make a changed line worth keeping first.
const PRIORITY_WORDS: &[&str] = &[
    "error",
    "exception",
    "fail",
    "failed",
    "failure",
    "fatal",
    "critical",
    "crash",
    "panic",
    "important",
    "note",
    "todo",
    "fixme",
    "hack",
    "xxx",
    "bug",
    "fix",
    "security",
    "auth",
    "password",
    "secret",
    "token",
];
/// The ratio below which the original is worth offloading instead.
const CCR_RATIO_THRESHOLD: f64 = 0.8;
/// One hunk of a file diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffHunk {
    /// The `@@ … @@` header line.
    pub header: String,
    /// The hunk body, without its header.
    pub lines: Vec<String>,
    /// Zero-based position within its file.
    pub position: usize,
}
impl DiffHunk {
    /// Lines added by this hunk.
    #[must_use]
    pub fn additions(&self) -> usize {
        self.lines
            .iter()
            .filter(|line| is_addition_line(line))
            .count()
    }
    /// Lines removed by this hunk.
    #[must_use]
    pub fn deletions(&self) -> usize {
        self.lines
            .iter()
            .filter(|line| is_deletion_line(line))
            .count()
    }
    /// Additions plus deletions.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.additions() + self.deletions()
    }
}
/// One file of a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffFile {
    /// The file header and any preamble lines that follow it.
    pub header: Vec<String>,
    /// The file's hunks, in order.
    pub hunks: Vec<DiffHunk>,
    /// Zero-based position within the diff.
    pub position: usize,
}
impl DiffFile {
    /// The first header line, or a positional fallback label.
    #[must_use]
    pub fn label(&self) -> String {
        self.header
            .first()
            .cloned()
            .unwrap_or_else(|| format!("diff-file-{}", self.position))
    }
    /// Additions across every hunk.
    #[must_use]
    pub fn additions(&self) -> usize {
        self.hunks.iter().map(DiffHunk::additions).sum()
    }
    /// Deletions across every hunk.
    #[must_use]
    pub fn deletions(&self) -> usize {
        self.hunks.iter().map(DiffHunk::deletions).sum()
    }
    /// Changes across every hunk.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.hunks.iter().map(DiffHunk::change_count).sum()
    }
}
/// Compresses unified diffs.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiffCompressor;
impl RuleCompressor for DiffCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        let original_line_count = content.lines().count();
        if original_line_count < ctx.diff_min_lines {
            return RuleCompressionResult::unchanged(content, ContentType::GitDiff);
        }
        let (preamble, files) = parse_diff(content);
        if files.is_empty() {
            return RuleCompressionResult::unchanged(content, ContentType::GitDiff);
        }

        let selected_files = select_files(&files, ctx);
        let selected_file_count = selected_files.len();
        let mut output: Vec<String> = preamble;
        let mut hunks_retained = 0_usize;
        let mut hunks_omitted = 0_usize;
        let mut context_lines_retained = 0_usize;
        let mut context_lines_omitted = 0_usize;
        let mut changed_lines_retained = 0_usize;
        let mut changed_lines_omitted = 0_usize;
        let mut changed_lines_remaining = ctx.diff_max_changed_lines_total;
        let additions: usize = files.iter().map(DiffFile::additions).sum();
        let deletions: usize = files.iter().map(DiffFile::deletions).sum();

        let mut ordered_files = selected_files;
        ordered_files.sort_by_key(|file| file.position);
        for file in &ordered_files {
            output.extend(file.header.iter().cloned());
            let selected_hunks = select_hunks(&file.hunks, ctx);
            let selected_positions: BTreeSet<usize> =
                selected_hunks.iter().map(|hunk| hunk.position).collect();
            let omitted_for_file = file.hunks.len() - selected_hunks.len();
            for hunk in &file.hunks {
                if !selected_positions.contains(&hunk.position) {
                    continue;
                }
                let changed_limit = ctx
                    .diff_max_changed_lines_per_hunk
                    .min(changed_lines_remaining);
                let reduced =
                    reduce_hunk_lines(&hunk.lines, ctx.diff_max_context_lines, changed_limit);
                changed_lines_remaining =
                    changed_lines_remaining.saturating_sub(reduced.changed_retained);
                output.push(hunk.header.clone());
                output.extend(reduced.lines);
                if reduced.context_omitted > 0 {
                    output.push(format!(
                        "[{} unchanged/context diff lines omitted]",
                        reduced.context_omitted
                    ));
                }
                if reduced.changed_omitted > 0 {
                    output.push(format!(
                        "[{} changed diff lines omitted]",
                        reduced.changed_omitted
                    ));
                }
                hunks_retained += 1;
                context_lines_retained += reduced.context_retained;
                context_lines_omitted += reduced.context_omitted;
                changed_lines_retained += reduced.changed_retained;
                changed_lines_omitted += reduced.changed_omitted;
            }
            hunks_omitted += omitted_for_file;
        }

        let files_omitted = files.len() - selected_file_count;
        output.push(format!(
            "[{} files changed, +{additions} -{deletions} lines, {hunks_omitted} hunks omitted, \
             {files_omitted} files omitted]",
            files.len()
        ));
        let candidate = output.join("\n");
        let compressed_line_count = candidate.lines().count();
        let lossy = files_omitted > 0
            || hunks_omitted > 0
            || context_lines_omitted > 0
            || changed_lines_omitted > 0;
        let details = json!({
            "files_affected": files.len(),
            "files_retained": selected_file_count,
            "files_omitted": files_omitted,
            "hunks_affected": files.iter().map(|file| file.hunks.len()).sum::<usize>(),
            "hunks_retained": hunks_retained,
            "hunks_omitted": hunks_omitted,
            "additions": additions,
            "deletions": deletions,
            "context_lines_retained": context_lines_retained,
            "context_lines_omitted": context_lines_omitted,
            "changed_lines_retained": changed_lines_retained,
            "changed_lines_omitted": changed_lines_omitted,
            "original_line_count": original_line_count,
            "compressed_line_count": compressed_line_count,
            "should_offload_original":
                (compressed_line_count as f64) < original_line_count as f64 * CCR_RATIO_THRESHOLD,
        });
        if candidate != content && lossy && meets_savings_ratio(content, &candidate, ctx) {
            return RuleCompressionResult::applied(
                candidate,
                ContentType::GitDiff,
                true,
                Some(details),
            );
        }
        RuleCompressionResult::declined(content, ContentType::GitDiff, Some(details))
    }
}
/// The files a compressed diff keeps.
fn select_files(files: &[DiffFile], ctx: &RuleContext) -> Vec<DiffFile> {
    if files.len() <= ctx.diff_max_files {
        return files.to_vec();
    }
    let mut ranked = files.to_vec();
    ranked.sort_by(|left, right| {
        right
            .change_count()
            .cmp(&left.change_count())
            .then(left.position.cmp(&right.position))
    });
    ranked.truncate(ctx.diff_max_files);
    ranked
}
/// The hunks a compressed file keeps.
fn select_hunks(hunks: &[DiffHunk], ctx: &RuleContext) -> Vec<DiffHunk> {
    if hunks.len() <= ctx.diff_max_hunks_per_file {
        return hunks.to_vec();
    }
    let mut selected: BTreeSet<usize> = BTreeSet::new();
    for hunk in [hunks.first(), hunks.last()].into_iter().flatten() {
        if selected.len() < ctx.diff_max_hunks_per_file {
            selected.insert(hunk.position);
        }
    }
    let mut ranked = hunks.to_vec();
    ranked.sort_by(|left, right| {
        score_text(right, ctx)
            .cmp(&score_text(left, ctx))
            .then(left.position.cmp(&right.position))
    });
    for hunk in ranked {
        if selected.len() >= ctx.diff_max_hunks_per_file {
            break;
        }
        selected.insert(hunk.position);
    }
    hunks
        .iter()
        .filter(|hunk| selected.contains(&hunk.position))
        .cloned()
        .collect()
}
/// The result of trimming one hunk.
struct ReducedHunk {
    lines: Vec<String>,
    context_retained: usize,
    context_omitted: usize,
    changed_retained: usize,
    changed_omitted: usize,
}
/// Keeps the priority changes plus the first `max_changed`, and nearby context.
fn reduce_hunk_lines(lines: &[String], max_context: usize, max_changed: usize) -> ReducedHunk {
    let change_positions: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| is_change_line(line))
        .map(|(index, _)| index)
        .collect();
    let priority: Vec<usize> = change_positions
        .iter()
        .copied()
        .filter(|index| {
            lines
                .get(*index)
                .is_some_and(|line| contains_word(&line.to_lowercase(), PRIORITY_WORDS))
        })
        .collect();
    let mut selected: BTreeSet<usize> = priority.into_iter().take(max_changed).collect();
    for index in change_positions {
        if selected.len() >= max_changed {
            break;
        }
        selected.insert(index);
    }

    let mut kept: Vec<String> = Vec::new();
    let mut reduced = ReducedHunk {
        lines: Vec::new(),
        context_retained: 0,
        context_omitted: 0,
        changed_retained: 0,
        changed_omitted: 0,
    };
    for (index, line) in lines.iter().enumerate() {
        if is_context_line(line) {
            let near = selected
                .iter()
                .any(|change| index.abs_diff(*change) <= max_context);
            if near {
                kept.push(line.clone());
                reduced.context_retained += 1;
            } else {
                reduced.context_omitted += 1;
            }
        } else if is_change_line(line) {
            if selected.contains(&index) {
                kept.push(line.clone());
                reduced.changed_retained += 1;
            } else {
                reduced.changed_omitted += 1;
            }
        } else {
            kept.push(line.clone());
        }
    }
    reduced.lines = kept;
    reduced
}
/// The ranking score of a hunk, in thousandths.
fn score_text(hunk: &DiffHunk, ctx: &RuleContext) -> i64 {
    let text = std::iter::once(hunk.header.as_str())
        .chain(hunk.lines.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();
    let mut score = (hunk.change_count() as f64 * 0.03).min(0.3);
    let query_hits = ctx
        .query_terms
        .iter()
        .filter(|term| text.contains(&term.to_lowercase()))
        .count();
    score += 0.2 * query_hits as f64;
    if contains_word(&text, PRIORITY_WORDS) {
        score += 0.3;
    }
    (score.min(1.0_f64) * 1000.0) as i64
}
/// Parses a diff into its preamble and files.
fn parse_diff(content: &str) -> (Vec<String>, Vec<DiffFile>) {
    let mut preamble: Vec<String> = Vec::new();
    let mut files: Vec<DiffFile> = Vec::new();
    let mut file_header: Option<Vec<String>> = None;
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut hunk_header: Option<String> = None;
    let mut hunk_lines: Vec<String> = Vec::new();

    for line in content.lines() {
        if is_file_header(line) {
            finish_file(
                &mut files,
                &mut file_header,
                &mut hunks,
                &mut hunk_header,
                &mut hunk_lines,
            );
            file_header = Some(vec![line.to_owned()]);
        } else if file_header.is_none() {
            preamble.push(line.to_owned());
        } else if is_hunk_header(line) {
            finish_hunk(&mut hunks, &mut hunk_header, &mut hunk_lines);
            hunk_header = Some(line.to_owned());
        } else if hunk_header.is_none() {
            if let Some(header) = file_header.as_mut() {
                header.push(line.to_owned());
            }
        } else {
            hunk_lines.push(line.to_owned());
        }
    }
    finish_file(
        &mut files,
        &mut file_header,
        &mut hunks,
        &mut hunk_header,
        &mut hunk_lines,
    );
    (preamble, files)
}
/// Closes the pending hunk, if any.
fn finish_hunk(
    hunks: &mut Vec<DiffHunk>,
    hunk_header: &mut Option<String>,
    hunk_lines: &mut Vec<String>,
) {
    if let Some(header) = hunk_header.take() {
        hunks.push(DiffHunk {
            header,
            lines: std::mem::take(hunk_lines),
            position: hunks.len(),
        });
    }
    hunk_lines.clear();
}
/// Closes the pending file, if any.
fn finish_file(
    files: &mut Vec<DiffFile>,
    file_header: &mut Option<Vec<String>>,
    hunks: &mut Vec<DiffHunk>,
    hunk_header: &mut Option<String>,
    hunk_lines: &mut Vec<String>,
) {
    finish_hunk(hunks, hunk_header, hunk_lines);
    if let Some(header) = file_header.take() {
        files.push(DiffFile {
            header,
            hunks: std::mem::take(hunks),
            position: files.len(),
        });
    }
    hunks.clear();
}
/// Whether `line` opens a file section.
fn is_file_header(line: &str) -> bool {
    ["diff --git ", "diff --cc ", "diff --combined "]
        .iter()
        .any(|prefix| line.starts_with(prefix))
}
/// Whether `line` opens a hunk section.
fn is_hunk_header(line: &str) -> bool {
    line.starts_with("@@ ") || line.starts_with("@@@ ")
}
/// Whether `line` adds or removes something.
fn is_change_line(line: &str) -> bool {
    is_addition_line(line) || is_deletion_line(line)
}
/// Whether `line` is an added line (and not the `+++` file header).
fn is_addition_line(line: &str) -> bool {
    line.starts_with('+') && !line.starts_with("+++ ")
}
/// Whether `line` is a removed line (and not the `---` file header).
fn is_deletion_line(line: &str) -> bool {
    line.starts_with('-') && !line.starts_with("--- ")
}
/// Whether `line` is unchanged context.
fn is_context_line(line: &str) -> bool {
    (line.starts_with(' ') || line.is_empty()) && !is_change_line(line)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    fn diff_with(files: usize, hunks: usize, lines: usize) -> String {
        let mut out =
            String::from("diff --git a/one b/one\nindex 111..222 100644\n--- a/one\n+++ b/one\n");
        for file in 0..files {
            if file > 0 {
                let _ = writeln!(
                    out,
                    "diff --git a/f{file} b/f{file}\n--- a/f{file}\n+++ b/f{file}"
                );
            }
            for hunk in 0..hunks {
                let _ = writeln!(out, "@@ -{hunk},0 +{hunk},0 @@");
                for line in 0..lines {
                    let _ = writeln!(out, "+added {file}-{hunk}-{line}");
                }
            }
        }
        out
    }
    #[test]
    fn short_diffs_are_declined_without_details() {
        let content = "diff --git a/x b/x\n+one\n";
        let result = DiffCompressor.compress(content, &RuleContext::new(100));
        assert!(!result.modified);
        assert!(result.details.is_none());
    }
    #[test]
    fn large_diffs_keep_priority_changes_and_report_details() {
        let content = diff_with(1, 1, 200);
        let ctx = RuleContext::new(10_000);
        let result = DiffCompressor.compress(&content, &ctx);
        assert!(result.modified);
        assert!(result.content.contains("[192 changed diff lines omitted]"));
        assert!(
            result.content.contains("200 files changed")
                || result.content.contains("1 files changed")
        );
        let details = result.details.expect("details");
        assert_eq!(details["changed_lines_retained"], 8);
        assert_eq!(details["changed_lines_omitted"], 192);
        assert_eq!(details["should_offload_original"], true);
    }
    #[test]
    fn an_untrimmed_diff_is_never_substituted() {
        let content = diff_with(1, 1, 3);
        let mut ctx = RuleContext::new(10_000);
        ctx.diff_min_lines = 5;
        let result = DiffCompressor.compress(&content, &ctx);
        assert!(!result.modified);
        let details = result.details.expect("details");
        assert_eq!(details["should_offload_original"], false);
    }
}
