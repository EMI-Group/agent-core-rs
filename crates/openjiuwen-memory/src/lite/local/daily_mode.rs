//! `daily_memory/` rollups for the local coding-memory store: assembling a day's facts and
//! session lines, rendering the document, and writing it idempotently.
//!
//! Ports the `daily_memory` sweep of Python `core/memory/lite/` onto the local store: one
//! Markdown document per calendar day under
//! `<root>/daily_memory/YYYY-MM-DD.md`, aggregating the coding facts that were updated that
//! day and the session log lines recorded that day. The document format, the stable
//! `daily:<date>` id, the deterministic ordering, and the atomic idempotent write all belong
//! to [`crate::lite::daily`]; this module only supplies the inputs and owns the day's file.
//!
//! # Inputs
//!
//! - Facts come from [`LocalCodingMemory::list`], so a rollup always describes the document
//!   on disk (an external edit is picked up by the same staleness check every read applies).
//! - Session lines come from [`LocalCodingMemory::session_records`], with each record's day
//!   taken from its own timestamp (`None` when the line records none, which never
//!   contributes to a day).
//!
//! A fact contributes only when its `updated_at` date is the rollup day, and a session line
//! only when its own timestamp date is that day, so a sweep is independent of when it runs.
//!
//! # Rendering versus writing
//!
//! [`LocalCodingMemory::rollup_day`] renders only: it reports the [`DailyRollup`] with
//! `written = false` and touches no file. [`LocalCodingMemory::write_daily_rollup`] renders
//! the same document and then writes it through
//! [`write_rollup`](crate::lite::daily::write_rollup), reporting `written = changed`: an
//! unchanged document leaves the file byte-identical (and its modification time untouched),
//! and a day whose inputs changed is rewritten in place, preserving its id.
//!
//! Both entry points report the contributing inputs in rendered order — the fact names and
//! the `<path>:<line>` session entries — taken from
//! [`select_rollup_inputs`](crate::lite::daily::select_rollup_inputs), so the reported lists
//! are exactly what the rendered document contains rather than a re-derivation of the filter.
//!
//! # Constraints
//!
//! A rollup is derived state: it may be deleted and regenerated at any time, and nothing
//! here starts a timer, a background task, or a watcher — a sweep is an explicit caller
//! request. A missing day of inputs produces a document with `_None._` sections rather than
//! an error, and filesystem failures carry the failing path.
use std::path::PathBuf;

use chrono::NaiveDate;

use openjiuwen_common::error::Result;

use crate::lite::daily::{self, DailyRollup, DailySessionLine, DAILY_DIR};
use crate::lite::session::{SessionChunk, SessionStore};

use super::state::LocalCodingMemory;

impl LocalCodingMemory {
    /// The directory holding the daily rollup documents (`<root>/daily_memory`).
    ///
    /// The path is reported whether or not the directory exists yet;
    /// [`LocalCodingMemory::write_daily_rollup`] creates it on demand.
    #[must_use]
    pub fn daily_dir(&self) -> PathBuf {
        self.inner.root.join(DAILY_DIR)
    }

    /// Renders `date`'s rollup without writing anything.
    ///
    /// The returned [`DailyRollup`] carries the stable `daily:<date>` id, the contributing
    /// fact names and `<path>:<line>` session entries in rendered order, the rendered
    /// document, and `written = false`. Nothing is created on disk, so a dry run is free.
    ///
    /// # Errors
    ///
    /// Returns the [`LocalCodingMemory::list`] error when the Markdown document cannot be
    /// reloaded and the [`LocalCodingMemory::session_records`] error when a session log
    /// cannot be read.
    pub async fn rollup_day(&self, date: NaiveDate) -> Result<DailyRollup> {
        self.render_day(date).await
    }

    /// Renders `date`'s rollup and writes it to `<root>/daily_memory/YYYY-MM-DD.md`.
    ///
    /// The write is atomic (temporary file plus rename) and idempotent: when the existing
    /// document already has the rendered bytes nothing is rewritten and `written = false`,
    /// otherwise the document is replaced and `written = true`. The returned rollup is the
    /// same one [`LocalCodingMemory::rollup_day`] reports, with `written` filled in.
    ///
    /// # Errors
    ///
    /// Returns the [`LocalCodingMemory::rollup_day`] errors plus the filesystem error (with
    /// the failing path as context) when the directory cannot be created or the document
    /// cannot be written.
    pub async fn write_daily_rollup(&self, date: NaiveDate) -> Result<DailyRollup> {
        let mut rollup = self.render_day(date).await?;
        let (_path, changed) = daily::write_rollup(&self.inner.root, date, &rollup.content).await?;
        rollup.written = changed;
        Ok(rollup)
    }

    /// Assembles, renders, and reports one day, writing nothing.
    async fn render_day(&self, date: NaiveDate) -> Result<DailyRollup> {
        let facts = self.list().await?;
        let chunks = SessionStore::new(&self.inner.root).chunks().await?;
        let session_lines = daily_session_lines(&chunks);
        let content = daily::render_rollup(date, &facts, &session_lines);
        let (fact_names, sessions) = daily::select_rollup_inputs(date, &facts, &session_lines);
        Ok(DailyRollup {
            id: daily::rollup_id(date),
            date,
            facts: fact_names,
            sessions,
            content,
            written: false,
        })
    }
}

/// Maps decoded session chunks onto the rollup's session-line shape.
///
/// The day of a line is its own recorded timestamp, never the directory it lives in, so a
/// log whose writes span midnight still contributes each line to the right day.
fn daily_session_lines(chunks: &[SessionChunk]) -> Vec<DailySessionLine> {
    chunks
        .iter()
        .map(|chunk| DailySessionLine {
            path: chunk.path.clone(),
            line: chunk.line,
            date: chunk
                .record
                .timestamp
                .map(|timestamp| timestamp.date_naive()),
            text: chunk.record.text(),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::Utc;
    use tempfile::TempDir;

    use crate::lite::fact::CodingFact;
    use crate::lite::markdown::MarkdownMemory;
    use crate::lite::session::SessionRecord;

    use super::super::embedding::test_support::{open_keyword, scope};
    use super::*;

    /// The day the rollup tests aggregate (the wall clock when the facts are written).
    fn today() -> NaiveDate {
        Utc::now().date_naive()
    }

    /// The day before [`today`].
    fn yesterday() -> NaiveDate {
        today()
            .pred_opt()
            .expect("a day before the first representable day")
    }

    /// 09:00 UTC on `date`.
    fn at(date: NaiveDate) -> chrono::DateTime<Utc> {
        date.and_hms_opt(9, 0, 0)
            .expect("a valid clock time")
            .and_utc()
    }

    /// Writes `fact` into `store`'s Markdown document with its own timestamp, then reindexes.
    async fn seed(store: &LocalCodingMemory, fact: CodingFact) {
        let text = std::fs::read_to_string(store.memory_path()).expect("read MEMORY.md");
        let mut document = MarkdownMemory::parse(&text)
            .expect("parse MEMORY.md")
            .with_scope(scope());
        let _replaced = document.upsert_fact(&fact);
        std::fs::write(store.memory_path(), document.render()).expect("write MEMORY.md");
        let _indexed = store.reindex().await.expect("reindex");
    }

    #[tokio::test]
    async fn rollup_day_keeps_only_the_days_inputs_and_reports_what_it_rendered() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let day = today();
        store
            .write(CodingFact::new(&scope(), "build", "cargo build"))
            .await
            .unwrap();
        let mut older = CodingFact::new(&scope(), "old", "an older body");
        older.updated_at = at(yesterday());
        seed(&store, older).await;
        store
            .append_session(
                "today",
                day,
                &[SessionRecord::new("ran the build").with_timestamp(at(day))],
            )
            .await
            .unwrap();
        store
            .append_session(
                "old",
                yesterday(),
                &[SessionRecord::new("yesterday's turn").with_timestamp(at(yesterday()))],
            )
            .await
            .unwrap();
        assert_eq!(store.daily_dir(), temp.path().join("daily_memory"));

        let rollup = store.rollup_day(day).await.unwrap();
        assert_eq!(rollup.id, format!("daily:{day}"));
        assert_eq!(rollup.date, day);
        assert_eq!(rollup.facts, vec!["build".to_owned()]);
        assert_eq!(
            rollup.sessions,
            vec![format!("sessions/{day}/today.jsonl:1")]
        );
        assert!(!rollup.written);
        assert!(rollup.content.contains(&format!("id: daily:{day}")));
        assert!(rollup.content.contains(&format!("# Daily Memory {day}")));
        assert!(
            rollup.content.contains("- [Project] build"),
            "{}",
            rollup.content
        );
        for name in &rollup.facts {
            assert!(
                rollup.content.contains(&format!("- [Project] {name}")),
                "the reported fact `{name}` is missing from {}",
                rollup.content
            );
        }
        for entry in &rollup.sessions {
            assert!(
                rollup.content.contains(&format!("- {entry} — ")),
                "the reported session `{entry}` is missing from {}",
                rollup.content
            );
        }
        assert!(!rollup.content.contains("yesterday's turn"));
        assert!(
            std::fs::metadata(store.daily_dir()).is_err(),
            "a dry run writes nothing"
        );

        let older_day = store.rollup_day(yesterday()).await.unwrap();
        assert_eq!(older_day.facts, vec!["old".to_owned()]);
        assert_eq!(
            older_day.sessions,
            vec![format!("sessions/{}/old.jsonl:1", yesterday())]
        );
        assert!(older_day.content.contains("yesterday's turn"));
        assert!(!older_day.content.contains("- [Project] build"));
    }

    #[tokio::test]
    async fn write_daily_rollup_is_idempotent_and_rewrites_a_changed_day() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let day = today();
        store
            .write(CodingFact::new(&scope(), "build", "cargo build"))
            .await
            .unwrap();

        let first = store.write_daily_rollup(day).await.unwrap();
        assert!(first.written);
        let path = temp.path().join("daily_memory").join(format!("{day}.md"));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            first.content,
            "the written document is the rendered one"
        );

        let second = store.write_daily_rollup(day).await.unwrap();
        assert!(!second.written, "an unchanged day is not rewritten");
        assert_eq!(second.content, first.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), first.content);

        store
            .write(CodingFact::new(&scope(), "release", "ship it"))
            .await
            .unwrap();
        let third = store.write_daily_rollup(day).await.unwrap();
        assert!(third.written, "a changed day is rewritten");
        assert_eq!(third.facts, vec!["build".to_owned(), "release".to_owned()]);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), third.content);
        assert_ne!(third.content, first.content);
    }

    #[tokio::test]
    async fn an_empty_day_renders_both_none_sections() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let rollup = store.rollup_day(today()).await.unwrap();
        assert_eq!(rollup.facts, Vec::<String>::new());
        assert_eq!(rollup.sessions, Vec::<String>::new());
        assert_eq!(rollup.content.matches("_None._").count(), 2);
        assert!(!rollup.written);
    }
}
