//! `daily_memory/YYYY-MM-DD.md` rollups with a stable id and deterministic content
//! ordering.
//!
//! Ports the deferred `daily_memory` rollups of Python `core/memory/lite/`: one Markdown
//! document per calendar day, aggregating that day's coding facts and session lines into a
//! compact, prompt-facing summary.
//!
//! # Layout
//!
//! ```text
//! <root>/daily_memory/YYYY-MM-DD.md
//! ```
//!
//! The file name is the day in UTC `YYYY-MM-DD` form, so the layout is self-describing and
//! sortable. The document carries front matter with its days and a stable id — the id is
//! derived from the day, not from the content, so it survives re-aggregation — followed by
//! one section per source (facts, sessions) and one entry per contributing item.
//!
//! # Document shape
//!
//! [`render_rollup`](crate::lite::daily::render_rollup) is a pure string builder and emits exactly:
//!
//! ```text
//! ---
//! id: daily:2026-11-14
//! date: 2026-11-14
//! facts: 1
//! sessions: 1
//! ---
//!
//! # Daily Memory 2026-11-14
//!
//! ## Facts
//!
//! - [Project] build — how to build
//!
//! ## Sessions
//!
//! - sessions/a.jsonl:3 — ran the build
//! ```
//!
//! The front matter is a [`FrontMatter`](crate::lite::markdown::FrontMatter) block holding `id`, `date`, the `facts` count, and
//! the `sessions` count, in that order. One blank line follows the closing fence, then the
//! title `# Daily Memory <YYYY-MM-DD>`, then the two sections. A section with no entries
//! renders `_None._`. The document ends with exactly one newline; there is no other
//! blank-line variation.
//!
//! # Deterministic ordering
//!
//! Entries are ordered by a fixed rule rather than by discovery order or filesystem
//! timestamps: facts by the fixed type order (user, feedback, project, reference) then name
//! ascending, session lines by path then line number ascending, and each bullet is rendered
//! from the same fields every time. Aggregating the same inputs twice therefore produces
//! byte-identical documents, which makes the rollup diffable and safe to commit. A fact
//! contributes only when its `updated_at` date is the rollup day; a session line
//! contributes only when its own date is that day.
//!
//! # Idempotence
//!
//! [`write_rollup`](crate::lite::daily::write_rollup) rewrites nothing when the existing document already has the same bytes
//! and reports `changed = false`; running over a day whose inputs changed rewrites the
//! document in place, preserving its id. A document is written through a temporary file
//! plus rename, so a reader never sees a partially written day. Days are independent, so a
//! retried or partially completed sweep cannot corrupt an already written day.
//!
//! # Constraints
//!
//! Rollups are derived state: they may be deleted and regenerated from the facts and session
//! logs at any time. Nothing here starts a timer or a background task — a sweep is an explicit
//! caller request — and a missing day of inputs produces a document with `_None._` sections
//! rather than an error.
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use openjiuwen_common::error::{Error, Result};

use super::fact::{CodingFact, CodingFactType};
use super::index::type_label;
use super::markdown::FrontMatter;

/// Directory (relative to the root) holding the daily rollup documents.
pub const DAILY_DIR: &str = "daily_memory";

/// Maximum number of session-text characters kept before the trailing ellipsis.
const MAX_SESSION_CHARS: usize = 300;

/// The ellipsis appended to truncated session text.
const ELLIPSIS: char = '…';

/// The stable rollup id of `date`, `daily:YYYY-MM-DD`.
///
/// The id depends only on the day, never on the content, so re-aggregating a day keeps the
/// same identity.
#[must_use]
pub fn rollup_id(date: NaiveDate) -> String {
    format!("daily:{date}")
}

/// The file name of `date`'s rollup, `YYYY-MM-DD.md`.
#[must_use]
pub fn file_name(date: NaiveDate) -> String {
    format!("{date}.md")
}

/// The path of `date`'s rollup, `<root>/daily_memory/YYYY-MM-DD.md`.
#[must_use]
pub fn path_for(root: &Path, date: NaiveDate) -> PathBuf {
    root.join(DAILY_DIR).join(file_name(date))
}

/// One session line considered for a rollup.
///
/// This is deliberately decoupled from the session-log module: a caller maps its own log
/// records onto this shape, so the rollup renderer needs no session reader. `date` is the
/// line's own day (`None` when it cannot be dated); only lines whose date equals the rollup
/// day are rendered. `line` is the one-based line number used in the bullet and in the
/// dedicated ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DailySessionLine {
    /// Path of the session log the line came from (rendered verbatim).
    pub path: String,
    /// One-based line number inside that log.
    pub line: usize,
    /// The line's calendar day, when it is known.
    pub date: Option<NaiveDate>,
    /// The record text; line breaks are folded and the result is truncated when rendered.
    pub text: String,
}

/// One day's rollup: its identity, the contributing entries in rendered order, and the
/// rendered document.
///
/// [`facts`](Self::facts) and [`sessions`](Self::sessions) are the rendered order, not the
/// input order, so a caller can report exactly what the document contains. `written` is set
/// by the owner of the write and stays `false` on a freshly built value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DailyRollup {
    /// The stable id, [`rollup_id`] of [`date`](Self::date).
    pub id: String,
    /// The calendar day this rollup covers.
    pub date: NaiveDate,
    /// The names of the contributing facts, in rendered order.
    pub facts: Vec<String>,
    /// The `<path>:<line>` entries of the contributing session lines, in rendered order.
    pub sessions: Vec<String>,
    /// The rendered document, as [`render_rollup`] produced it.
    pub content: String,
    /// Whether [`write_rollup`] changed the file.
    pub written: bool,
}

/// Renders the rollup document of `date` from `facts` and `session_lines`.
///
/// Only facts whose `updated_at` date is `date` and session lines whose recorded date is
/// `date` contribute. Facts are ordered by the fixed type order (user, feedback, project,
/// reference) then name ascending; session lines by `(path, line)` ascending. Each fact is a
/// `- [<Type>] <name> — <description>` bullet (the em dash and description are omitted when
/// the description is blank); each session line is a `- <path>:<line> — <text>` bullet whose
/// text has every line break folded to a single space and is truncated to at most 300
/// characters followed by `…` when longer. An empty section renders `_None._`.
///
/// The output is byte-deterministic for the same input set and ends with exactly one
/// newline.
#[must_use]
pub fn render_rollup(
    date: NaiveDate,
    facts: &[CodingFact],
    session_lines: &[DailySessionLine],
) -> String {
    let mut day_facts: Vec<&CodingFact> = facts
        .iter()
        .filter(|fact| fact.updated_at.date_naive() == date)
        .collect();
    day_facts.sort_by(|left, right| {
        type_rank(left.fact_type)
            .cmp(&type_rank(right.fact_type))
            .then_with(|| left.name.cmp(&right.name))
    });

    let mut day_lines: Vec<&DailySessionLine> = session_lines
        .iter()
        .filter(|line| line.date == Some(date))
        .collect();
    day_lines.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| left.line.cmp(&right.line))
    });

    let mut front_matter = FrontMatter::new();
    let _previous = front_matter.insert("id", rollup_id(date));
    let _previous = front_matter.insert("date", date.to_string());
    let _previous = front_matter.insert("facts", day_facts.len().to_string());
    let _previous = front_matter.insert("sessions", day_lines.len().to_string());

    let mut lines: Vec<String> = Vec::with_capacity(day_facts.len() + day_lines.len() + 14);
    lines.push("---".to_owned());
    for (key, value) in front_matter.iter() {
        lines.push(format!("{key}: {value}"));
    }
    lines.push("---".to_owned());
    lines.push(String::new());
    lines.push(format!("# Daily Memory {date}"));
    lines.push(String::new());
    lines.push("## Facts".to_owned());
    lines.push(String::new());
    if day_facts.is_empty() {
        lines.push("_None._".to_owned());
    } else {
        for fact in &day_facts {
            lines.push(fact_bullet(fact));
        }
    }
    lines.push(String::new());
    lines.push("## Sessions".to_owned());
    lines.push(String::new());
    if day_lines.is_empty() {
        lines.push("_None._".to_owned());
    } else {
        for line in &day_lines {
            lines.push(session_bullet(line));
        }
    }
    let mut rendered = lines.join("\n");
    rendered.push('\n');
    rendered
}

/// Writes `content` to `date`'s rollup under `root`, atomically and idempotently.
///
/// The `daily_memory/` directory is created when missing, and the document is written to a
/// temporary file in the same directory and then renamed into place, so a concurrent reader
/// sees either the old document or the new one. When the existing file already has exactly
/// `content`'s bytes this is a no-op that leaves the file untouched. Returns the rollup's
/// path and whether the file changed.
///
/// # Errors
///
/// Returns [`Error::External`] with the failing path as context when the directory cannot be
/// created, the existing document cannot be read, or the temporary write/rename fails.
pub async fn write_rollup(root: &Path, date: NaiveDate, content: &str) -> Result<(PathBuf, bool)> {
    let directory = root.join(DAILY_DIR);
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!(
                "creating the daily memory directory {}",
                directory.display()
            ))
        })?;

    let path = path_for(root, date);
    match tokio::fs::read(&path).await {
        Ok(existing) if existing.as_slice() == content.as_bytes() => return Ok((path, false)),
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(Error::external(error)
                .with_context(format!("reading the daily rollup {}", path.display())));
        }
    }

    let temporary = path.with_extension("md.tmp");
    tokio::fs::write(&temporary, content.as_bytes())
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!(
                "writing the temporary daily rollup {}",
                temporary.display()
            ))
        })?;
    tokio::fs::rename(&temporary, &path)
        .await
        .map_err(|error| {
            Error::external(error)
                .with_context(format!("replacing the daily rollup {}", path.display()))
        })?;
    Ok((path, true))
}

/// The sort rank of a fact type: the declaration order of [`CodingFactType::ALL`].
fn type_rank(fact_type: CodingFactType) -> usize {
    CodingFactType::ALL
        .iter()
        .position(|candidate| *candidate == fact_type)
        .unwrap_or(CodingFactType::ALL.len())
}

/// Renders one fact as a rollup bullet.
fn fact_bullet(fact: &CodingFact) -> String {
    let label = type_label(fact.fact_type);
    if fact.description.trim().is_empty() {
        format!("- [{label}] {}", fact.name)
    } else {
        format!("- [{label}] {} — {}", fact.name, fact.description)
    }
}

/// Renders one session line as a rollup bullet.
fn session_bullet(line: &DailySessionLine) -> String {
    let text = truncate_text(&fold_line_breaks(&line.text), MAX_SESSION_CHARS);
    format!("- {}:{} — {text}", line.path, line.line)
}

/// Folds every line break (`\r\n`, `\r`, `\n`) into a single space.
fn fold_line_breaks(text: &str) -> String {
    text.replace("\r\n", "\n").replace(['\r', '\n'], " ")
}

/// Keeps at most `max_chars` characters, appending `…` when the text was longer.
///
/// The cut is made on a character boundary, so multi-byte text is never split.
fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut truncated: String = text.chars().take(max_chars).collect();
    truncated.push(ELLIPSIS);
    truncated
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::{DateTime, Utc};

    use super::*;

    /// A November 2026 day.
    fn date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, day).expect("valid November day")
    }

    /// 09:00 UTC on the given November day.
    fn at(day: u32) -> DateTime<Utc> {
        date(day)
            .and_hms_opt(9, 0, 0)
            .expect("valid time")
            .and_utc()
    }

    /// A fact updated on `day`.
    fn fact(name: &str, fact_type: CodingFactType, description: &str, day: u32) -> CodingFact {
        CodingFact {
            name: name.to_owned(),
            description: description.to_owned(),
            fact_type,
            content: format!("body of {name}"),
            updated_at: at(day),
            ..CodingFact::default()
        }
    }

    /// A session line recorded on `day`.
    fn line(path: &str, number: usize, day: u32, text: &str) -> DailySessionLine {
        DailySessionLine {
            path: path.to_owned(),
            line: number,
            date: Some(date(day)),
            text: text.to_owned(),
        }
    }

    #[test]
    fn ids_and_paths_are_derived_from_the_day() {
        assert_eq!(rollup_id(date(14)), "daily:2026-11-14");
        assert_eq!(file_name(date(14)), "2026-11-14.md");
        assert_eq!(
            path_for(Path::new("/tmp/root"), date(14)),
            PathBuf::from("/tmp/root/daily_memory/2026-11-14.md")
        );
    }

    #[test]
    fn the_document_has_the_documented_front_matter_and_sections() {
        let content = render_rollup(
            date(14),
            &[fact("build", CodingFactType::Project, "how to build", 14)],
            &[line("sessions/a.jsonl", 3, 14, "ran the build")],
        );
        let expected = concat!(
            "---\n",
            "id: daily:2026-11-14\n",
            "date: 2026-11-14\n",
            "facts: 1\n",
            "sessions: 1\n",
            "---\n",
            "\n",
            "# Daily Memory 2026-11-14\n",
            "\n",
            "## Facts\n",
            "\n",
            "- [Project] build — how to build\n",
            "\n",
            "## Sessions\n",
            "\n",
            "- sessions/a.jsonl:3 — ran the build\n",
        );
        assert_eq!(content, expected);
    }

    #[test]
    fn an_empty_day_renders_both_none_sections() {
        let content = render_rollup(date(14), &[], &[]);
        let expected = concat!(
            "---\n",
            "id: daily:2026-11-14\n",
            "date: 2026-11-14\n",
            "facts: 0\n",
            "sessions: 0\n",
            "---\n",
            "\n",
            "# Daily Memory 2026-11-14\n",
            "\n",
            "## Facts\n",
            "\n",
            "_None._\n",
            "\n",
            "## Sessions\n",
            "\n",
            "_None._\n",
        );
        assert_eq!(content, expected);
    }

    #[test]
    fn facts_render_in_type_then_name_order_regardless_of_input_order() {
        let facts = vec![
            fact("zeta", CodingFactType::Project, "last project", 14),
            fact("alpha", CodingFactType::Reference, "a reference", 14),
            fact("beta", CodingFactType::Project, "a project", 14),
            fact("gamma", CodingFactType::User, "the user", 14),
            fact("delta", CodingFactType::Feedback, "some feedback", 14),
        ];
        let mut shuffled = facts.clone();
        shuffled.reverse();
        shuffled.swap(0, 3);
        let expected = concat!(
            "- [User] gamma — the user\n",
            "- [Feedback] delta — some feedback\n",
            "- [Project] beta — a project\n",
            "- [Project] zeta — last project\n",
            "- [Reference] alpha — a reference\n",
        );
        assert_eq!(
            render_rollup(date(14), &facts, &[]),
            render_rollup(date(14), &shuffled, &[])
        );
        assert!(render_rollup(date(14), &facts, &[]).contains(expected));
    }

    #[test]
    fn session_lines_render_in_path_then_line_order_regardless_of_input_order() {
        let lines = vec![
            line("sessions/b.jsonl", 2, 14, "second file"),
            line("sessions/a.jsonl", 9, 14, "later line"),
            line("sessions/a.jsonl", 1, 14, "first line"),
        ];
        let mut shuffled = lines.clone();
        shuffled.reverse();
        let expected = concat!(
            "## Sessions\n",
            "\n",
            "- sessions/a.jsonl:1 — first line\n",
            "- sessions/a.jsonl:9 — later line\n",
            "- sessions/b.jsonl:2 — second file\n",
        );
        assert_eq!(
            render_rollup(date(14), &[], &lines),
            render_rollup(date(14), &[], &shuffled)
        );
        assert!(render_rollup(date(14), &[], &lines).contains(expected));
    }

    #[test]
    fn facts_from_another_day_are_excluded() {
        let content = render_rollup(
            date(14),
            &[
                fact("today", CodingFactType::Project, "today's", 14),
                fact("yesterday", CodingFactType::Project, "yesterday's", 13),
            ],
            &[],
        );
        assert!(content.contains("today"));
        assert!(!content.contains("yesterday"));
        assert!(content.contains("facts: 1\n"));
    }

    #[test]
    fn session_lines_from_another_day_or_without_a_date_are_excluded() {
        let mut undated = line("sessions/c.jsonl", 1, 14, "no date");
        undated.date = None;
        let content = render_rollup(
            date(14),
            &[],
            &[
                line("sessions/a.jsonl", 1, 14, "today"),
                line("sessions/b.jsonl", 1, 13, "yesterday"),
                undated,
            ],
        );
        assert!(content.contains("today"));
        assert!(!content.contains("yesterday"));
        assert!(!content.contains("no date"));
        assert!(content.contains("sessions: 1\n"));
    }

    #[test]
    fn an_empty_description_drops_the_em_dash() {
        let content = render_rollup(
            date(14),
            &[fact("bare", CodingFactType::Project, "", 14)],
            &[],
        );
        assert!(content.contains("\n- [Project] bare\n"));
        assert!(!content.contains("bare —"));
    }

    #[test]
    fn session_text_folds_line_breaks_to_single_spaces() {
        let content = render_rollup(
            date(14),
            &[],
            &[line("sessions/a.jsonl", 1, 14, "one\ntwo\r\nthree\rfour")],
        );
        assert!(content.contains("- sessions/a.jsonl:1 — one two three four\n"));
    }

    #[test]
    fn long_session_text_is_truncated_on_a_character_boundary() {
        let long = "é".repeat(400);
        let content = render_rollup(date(14), &[], &[line("sessions/a.jsonl", 1, 14, &long)]);
        let bullet = content
            .lines()
            .find(|line| line.starts_with("- sessions/a.jsonl:1 — "))
            .expect("the session bullet");
        let text = bullet.split_once(" — ").expect("the bullet separator").1;
        assert!(text.ends_with('…'));
        let kept: String = text.chars().filter(|character| *character != '…').collect();
        assert_eq!(kept.chars().count(), MAX_SESSION_CHARS);
        assert!(kept.chars().all(|character| character == 'é'));
    }

    #[test]
    fn text_at_the_limit_is_not_truncated() {
        let exact = "a".repeat(MAX_SESSION_CHARS);
        let content = render_rollup(date(14), &[], &[line("sessions/a.jsonl", 1, 14, &exact)]);
        assert!(content.contains(&format!("- sessions/a.jsonl:1 — {exact}\n")));
        assert!(!content.contains('…'));
    }

    #[tokio::test]
    async fn write_rollup_creates_the_file_and_an_identical_rerun_is_unchanged() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let content = render_rollup(
            date(14),
            &[fact("build", CodingFactType::Project, "how to build", 14)],
            &[],
        );
        let (path, changed) = write_rollup(directory.path(), date(14), &content)
            .await
            .expect("first write");
        assert!(changed);
        assert_eq!(
            path,
            directory.path().join("daily_memory").join("2026-11-14.md")
        );
        assert_eq!(
            tokio::fs::read_to_string(&path).await.expect("read back"),
            content
        );
        let before = tokio::fs::metadata(&path)
            .await
            .expect("metadata")
            .modified()
            .ok();

        let (same_path, changed) = write_rollup(directory.path(), date(14), &content)
            .await
            .expect("second write");
        assert_eq!(same_path, path);
        assert!(!changed);
        assert_eq!(
            tokio::fs::read_to_string(&path).await.expect("read back"),
            content
        );
        let after = tokio::fs::metadata(&path)
            .await
            .expect("metadata")
            .modified()
            .ok();
        assert_eq!(before, after);
    }

    #[tokio::test]
    async fn a_changed_document_is_rewritten_and_survives_a_reopen() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let first = render_rollup(date(14), &[], &[]);
        let (path, changed) = write_rollup(directory.path(), date(14), &first)
            .await
            .expect("first write");
        assert!(changed);

        let second = render_rollup(
            date(14),
            &[fact("build", CodingFactType::Project, "how to build", 14)],
            &[],
        );
        let (second_path, changed) = write_rollup(directory.path(), date(14), &second)
            .await
            .expect("second write");
        assert_eq!(second_path, path);
        assert!(changed);
        assert_eq!(
            tokio::fs::read_to_string(&path).await.expect("reopen"),
            second
        );
        assert_ne!(first, second);

        // A fresh handle on the same root sees the committed document and reports no change.
        let (_, changed) = write_rollup(directory.path(), date(14), &second)
            .await
            .expect("idempotent rerun after reopen");
        assert!(!changed);
    }

    #[tokio::test]
    async fn rollups_of_different_days_are_independent() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let (first, _) = write_rollup(directory.path(), date(13), "day 13\n")
            .await
            .expect("day 13");
        let (second, _) = write_rollup(directory.path(), date(14), "day 14\n")
            .await
            .expect("day 14");
        assert_ne!(first, second);
        assert_eq!(
            tokio::fs::read_to_string(&first).await.expect("read 13"),
            "day 13\n"
        );
        assert_eq!(
            tokio::fs::read_to_string(&second).await.expect("read 14"),
            "day 14\n"
        );
    }
}
