//! `SQLite` FTS5 full-text index over lite coding facts (feature `sqlite`).
//!
//! Ports the search role of the Python lite-memory stack (`core/memory/lite/`, its
//! FTS5 + trigram index) to one contentful FTS5 table: facts are copied into
//! `coding_facts_fts` and recalled with a BM25-ranked `MATCH` query and snippets.
//! The index is derived state — the engine stays the source of truth — so
//! [`FtsIndex::rebuild`](crate::lite::fts::FtsIndex::rebuild) can always reconstruct the whole table from
//! [`CodingMemory::list`](super::CodingMemory::list).
//!
//! # Schema
//!
//! ```sql
//! CREATE VIRTUAL TABLE IF NOT EXISTS coding_facts_fts USING fts5(
//!     name UNINDEXED, content, description, tags, tokenize='unicode61'
//! )
//! ```
//!
//! Every constructor ([`FtsIndex::open`](crate::lite::fts::FtsIndex::open),
//! [`FtsIndex::open_path`](crate::lite::fts::FtsIndex::open_path),
//! [`FtsIndex::from_pool`](crate::lite::fts::FtsIndex::from_pool),
//! [`FtsIndex::in_memory`](crate::lite::fts::FtsIndex::in_memory)) applies this DDL idempotently.
//!
//! - `name` is `UNINDEXED`: it is the row key (upsert target, removal key, and the
//!   identifier reported on every hit) and, like every unindexed column, it is *not*
//!   matched by `MATCH`. Text search therefore uses the three indexed columns, and a
//!   hit reports the name of the fact it was stored under.
//! - `content` is the fact body and `description` its one-line summary.
//! - `tags` is the fact's tag list space-joined (`["build", "ci"]` → `build ci`); the
//!   column is tokenized, so each tag is an individual FTS term.
//!
//! # Query safety
//!
//! A user query is never interpolated into SQL and never reaches FTS5 as an
//! expression: [`FtsIndex::search`](crate::lite::fts::FtsIndex::search) tokenizes it with `\w+`, keeps at most ten tokens
//! that contain at least one alphanumeric character, and binds
//! `"tok1" OR "tok2"` as a statement parameter (embedded `"` doubled). FTS5 syntax
//! (`NEAR`, `AND`, `#`, `*`, `^`, parentheses) is therefore inert, and a query with
//! no usable token (`""`, whitespace, `"!@#$%^&*()"`, `"___"`) returns
//! `Ok(Vec::new())` without running a query at all.
//!
//! # Scoring
//!
//! Rows are ordered by `bm25()`; `SQLite` reports a stronger match as a more negative
//! value, so ascending `bm25` is best-first. Each hit exposes the raw `bm25` and a
//! normalized `score`: `1 / (1 + bm25)` when `bm25 >= 0`, otherwise
//! `1 / (1 - bm25)` — the Python reference's normalization, i.e. `1 / (1 + |bm25|)`
//! in `0..=1`. Note the consequence for callers that re-rank: a stronger match has a
//! larger `|bm25|` and therefore a *smaller* `score`, so rank by `bm25` ascending (or
//! keep the returned order), not by `score` descending.
use std::path::Path;
use std::str::FromStr;
use std::sync::OnceLock;

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqliteConnection, SqlitePool};

use super::CodingFact;

/// The tracing target of full-text-index diagnostics.
const TARGET: &str = "openjiuwen_memory::lite::fts";

/// The single contentful FTS5 table, applied idempotently by every constructor.
const SCHEMA: &str = "\
CREATE VIRTUAL TABLE IF NOT EXISTS coding_facts_fts USING fts5(
    name UNINDEXED, content, description, tags, tokenize='unicode61'
)";

/// Deletes the row stored under one fact name (the first half of an upsert).
const DELETE_BY_NAME_SQL: &str = "DELETE FROM coding_facts_fts WHERE name = ?";

/// Inserts one fact row (the second half of an upsert, and the whole of a rebuild
/// insert).
const INSERT_SQL: &str = "\
INSERT INTO coding_facts_fts (name, content, description, tags) VALUES (?, ?, ?, ?)";

/// The ranked `MATCH` query; `?1` is the match expression and `?2` the limit.
const SEARCH_SQL: &str = "\
SELECT name,
       bm25(coding_facts_fts) AS rank,
       snippet(coding_facts_fts, -1, '[', ']', '...', 12) AS snip
FROM coding_facts_fts
WHERE coding_facts_fts MATCH ?1
ORDER BY rank
LIMIT ?2";

/// Counts the indexed rows.
const COUNT_SQL: &str = "SELECT COUNT(*) AS total FROM coding_facts_fts";

/// Empties the index.
const CLEAR_SQL: &str = "DELETE FROM coding_facts_fts";

/// The query token pattern, identical to Python's `re.findall(r"\w+", query)`.
const TOKEN_PATTERN: &str = r"\w+";

/// The maximum number of query tokens turned into `OR` terms.
const MAX_QUERY_TOKENS: usize = 10;

/// Pool size for file-backed databases.
const DEFAULT_MAX_CONNECTIONS: u32 = 4;

/// The compiled token pattern, memoized (including a compilation failure, so the
/// error is reported instead of panicking).
static TOKEN_REGEX: OnceLock<std::result::Result<Regex, String>> = OnceLock::new();

/// One full-text hit.
///
/// `bm25` is the raw `SQLite` rank (negative, more negative is better) and `score`
/// its normalized `0..=1` form; see the [module docs](self) for the formula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FtsHit {
    /// The fact name stored in the `name` (unindexed) column.
    pub name: String,
    /// Raw `SQLite` `bm25()` value (negative; more negative is a better match).
    pub bm25: f64,
    /// Normalized `0..=1` score derived from `bm25`.
    pub score: f32,
    /// Excerpt from `snippet()`; empty when the row has no excerpt, never a panic.
    pub snippet: String,
}

/// A full-text index over coding facts, backed by one `SQLite` connection pool.
///
/// Clones share the pool. Use [`FtsIndex::open_path`] for a database file,
/// [`FtsIndex::in_memory`] for tests and short-lived runs, or
/// [`FtsIndex::from_pool`] to adopt a pool the caller already owns.
#[derive(Debug, Clone)]
pub struct FtsIndex {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
}

impl FtsIndex {
    /// Opens the database named by the `sqlite:` `url` and applies the schema.
    ///
    /// A missing database file is created. In-memory URLs (`sqlite::memory:` or a
    /// `mode=memory` parameter) are pinned to a single pooled connection, because an
    /// in-memory `SQLite` database is private to the connection that created it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `url` is not a `sqlite:` URL or cannot be
    /// parsed, and an [`Error::External`] carrying the `sqlx` failure when the
    /// database cannot be opened or the schema cannot be applied.
    pub async fn open(url: &str) -> Result<Self> {
        let url = url.trim();
        if !url.starts_with("sqlite:") {
            return Err(Error::validation(format!(
                "sqlite fts index: expected a `sqlite:` database URL, got \"{url}\""
            )));
        }
        let options = SqliteConnectOptions::from_str(url)
            .map_err(|error| {
                Error::validation(format!(
                    "sqlite fts index: invalid database URL \"{url}\": {error}"
                ))
            })?
            .create_if_missing(true);
        let max_connections = if is_memory_url(url) {
            1
        } else {
            DEFAULT_MAX_CONNECTIONS
        };
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await
            .map_err(db_error("open the database"))?;
        Self::from_pool(pool).await
    }

    /// Opens (creating if missing) the database file at `path` and applies the schema.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the file
    /// cannot be opened or the schema cannot be applied.
    pub async fn open_path(path: &Path) -> Result<Self> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(DEFAULT_MAX_CONNECTIONS)
            .connect_with(options)
            .await
            .map_err(db_error("open the database file"))?;
        Self::from_pool(pool).await
    }

    /// Adopts `pool` and applies the schema to it.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the schema
    /// cannot be applied (for example when the bundled `SQLite` lacks FTS5).
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        sqlx::query(SCHEMA)
            .execute(&pool)
            .await
            .map_err(db_error("create the full-text table"))?;
        Ok(Self { pool })
    }

    /// Creates a fresh in-memory index.
    ///
    /// # Errors
    ///
    /// Returns the [`FtsIndex::open`] errors.
    pub async fn in_memory() -> Result<Self> {
        Self::open("sqlite::memory:").await
    }

    /// The connection pool behind this index.
    #[must_use]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Inserts `fact`, replacing the row whose `name` matches.
    ///
    /// The delete and the insert run in one transaction, so a concurrent reader never
    /// observes a fact missing from the index.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the write fails;
    /// nothing is committed in that case.
    pub async fn upsert(&self, fact: &CodingFact) -> Result<()> {
        let mut transaction = self.pool.begin().await.map_err(db_error("begin a write"))?;
        write_row(&mut transaction, fact).await?;
        transaction
            .commit()
            .await
            .map_err(db_error("commit a write"))?;
        Ok(())
    }

    /// Removes the row stored under `name`, returning whether one was deleted.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the delete
    /// fails.
    pub async fn remove(&self, name: &str) -> Result<bool> {
        let outcome = sqlx::query(DELETE_BY_NAME_SQL)
            .bind(name)
            .execute(&self.pool)
            .await
            .map_err(db_error("delete a fact"))?;
        Ok(outcome.rows_affected() > 0)
    }

    /// Searches the index for `query`, best match first, at most `limit` hits.
    ///
    /// `limit` is raised to `1` when it is `0`. A query without a usable token (empty,
    /// whitespace-only, or punctuation-only) returns no hits without querying the
    /// database; see the [module docs](self) for the normalization and scoring.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `limit` cannot be represented as a `SQLite`
    /// integer and an [`Error::External`] carrying the `sqlx` failure when the query
    /// fails. No user-provided text can reach FTS5 as syntax, so a query is never
    /// rejected by FTS5 itself.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<FtsHit>> {
        let Some(expression) = match_expression(query)? else {
            tracing::debug!(
                target: TARGET,
                "query has no searchable token; skipping the full-text lookup"
            );
            return Ok(Vec::new());
        };
        let limit = limit.max(1);
        let limit = i64::try_from(limit).map_err(|_| {
            Error::validation(format!(
                "sqlite fts index: limit {limit} cannot be represented as a sqlite integer"
            ))
        })?;
        let rows = sqlx::query(SEARCH_SQL)
            .bind(expression)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
            .map_err(db_error("search"))?;
        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let name: String = row.try_get("name").map_err(db_error("read a hit name"))?;
            let bm25: f64 = row.try_get("rank").map_err(db_error("read a hit rank"))?;
            let snippet: Option<String> = row
                .try_get("snip")
                .map_err(db_error("read a hit snippet"))?;
            hits.push(FtsHit {
                name,
                bm25,
                score: normalize_score(bm25),
                snippet: snippet.unwrap_or_default(),
            });
        }
        Ok(hits)
    }

    /// Replaces the whole index with `facts`, returning how many rows were indexed.
    ///
    /// The clear and the inserts run in one transaction, so the index is never
    /// observed empty or half rebuilt. Duplicate names inside `facts` keep the last
    /// occurrence, exactly like repeated [`FtsIndex::upsert`] calls.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the rebuild
    /// fails; the previous index is left untouched in that case.
    pub async fn rebuild<'a>(
        &self,
        facts: impl IntoIterator<Item = &'a CodingFact>,
    ) -> Result<usize> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(db_error("begin a rebuild"))?;
        sqlx::query(CLEAR_SQL)
            .execute(&mut *transaction)
            .await
            .map_err(db_error("clear the index for a rebuild"))?;
        let mut indexed = 0;
        for fact in facts {
            write_row(&mut transaction, fact).await?;
            indexed += 1;
        }
        transaction
            .commit()
            .await
            .map_err(db_error("commit a rebuild"))?;
        tracing::debug!(
            target: TARGET,
            indexed,
            "rebuilt the coding-fact full-text index"
        );
        Ok(indexed)
    }

    /// The number of rows in the index.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the count
    /// cannot be read.
    pub async fn count(&self) -> Result<usize> {
        let row = sqlx::query(COUNT_SQL)
            .fetch_one(&self.pool)
            .await
            .map_err(db_error("count the index"))?;
        let total: i64 = row.try_get("total").map_err(db_error("read the count"))?;
        usize::try_from(total).map_err(Error::external)
    }

    /// Empties the index.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::External`] carrying the `sqlx` failure when the delete
    /// fails.
    pub async fn clear(&self) -> Result<()> {
        sqlx::query(CLEAR_SQL)
            .execute(&self.pool)
            .await
            .map_err(db_error("clear the index"))?;
        tracing::debug!(target: TARGET, "cleared the coding-fact full-text index");
        Ok(())
    }
}

/// Deletes the fact's previous row and inserts the new one on `connection`.
///
/// Shared by [`FtsIndex::upsert`] and [`FtsIndex::rebuild`] so both write paths keep
/// the same name-keyed, tag-joined encoding.
async fn write_row(connection: &mut SqliteConnection, fact: &CodingFact) -> Result<()> {
    sqlx::query(DELETE_BY_NAME_SQL)
        .bind(fact.name.as_str())
        .execute(&mut *connection)
        .await
        .map_err(db_error("delete the previous row"))?;
    sqlx::query(INSERT_SQL)
        .bind(fact.name.as_str())
        .bind(fact.content.as_str())
        .bind(fact.description.as_str())
        .bind(fact.tags.join(" "))
        .execute(&mut *connection)
        .await
        .map_err(db_error("insert a fact row"))?;
    Ok(())
}

/// Builds the `MATCH` operand for `query`, or `None` when no token is searchable.
///
/// Tokens come from [`TOKEN_PATTERN`], at most [`MAX_QUERY_TOKENS`] of them, and only
/// those holding at least one alphanumeric character are kept: a token made of `_`
/// alone would tokenize to nothing under `unicode61` and turn an otherwise harmless
/// query into an FTS5 phrase error. Each kept token is quoted, so every FTS5 operator
/// is read as text.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the compiled token pattern is unavailable (a
/// build-level defect, reported instead of panicking).
fn match_expression(query: &str) -> Result<Option<String>> {
    let regex = token_regex()?;
    let terms: Vec<String> = regex
        .find_iter(query)
        .map(|found| found.as_str())
        .filter(|token| token.chars().any(char::is_alphanumeric))
        .take(MAX_QUERY_TOKENS)
        .map(quote_token)
        .collect();
    if terms.is_empty() {
        return Ok(None);
    }
    Ok(Some(terms.join(" OR ")))
}

/// Wraps `token` as an FTS5 phrase string, doubling any embedded `"`.
fn quote_token(token: &str) -> String {
    format!("\"{}\"", token.replace('"', "\"\""))
}

/// The compiled [`TOKEN_PATTERN`], initialized on first use.
fn token_regex() -> Result<&'static Regex> {
    let cached =
        TOKEN_REGEX.get_or_init(|| Regex::new(TOKEN_PATTERN).map_err(|error| error.to_string()));
    cached.as_ref().map_err(|message| {
        Error::validation(format!(
            "sqlite fts index: invalid token pattern: {message}"
        ))
    })
}

/// Normalizes a raw `bm25` value into `0..=1`.
///
/// `1 / (1 + bm25)` for `bm25 >= 0` and `1 / (1 - bm25)` otherwise, which is
/// `1 / (1 + |bm25|)`; a non-finite value has no meaningful magnitude and maps to `0`.
fn normalize_score(bm25: f64) -> f32 {
    let normalizer = if bm25 >= 0.0 { 1.0 + bm25 } else { 1.0 - bm25 };
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return 0.0;
    }
    (1.0 / normalizer).clamp(0.0, 1.0) as f32
}

/// Whether `url` selects an in-memory database private to its connection.
fn is_memory_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.contains(":memory:") || lower.contains("mode=memory")
}

/// Wraps a `sqlx` failure with the index operation that produced it.
fn db_error(operation: &'static str) -> impl FnOnce(sqlx::Error) -> Error {
    move |error| Error::external(error).with_context(format!("sqlite fts index: {operation}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::MemoryScope;
    use openjiuwen_common::error::ErrorKind;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "workspace")
    }

    fn fact(name: &str, content: &str) -> CodingFact {
        CodingFact::new(&scope(), name, content)
    }

    #[tokio::test]
    async fn finds_facts_by_name_content_description_and_tags() {
        let index = FtsIndex::in_memory().await.unwrap();
        index
            .upsert(&fact("build", "how to compile the crate"))
            .await
            .unwrap();
        index
            .upsert(
                &fact("release", "how to publish the package")
                    .with_tags(vec!["shipping".to_owned(), "cargo".to_owned()]),
            )
            .await
            .unwrap();
        index
            .upsert(&fact("docs", "the handbook").with_description("style guide"))
            .await
            .unwrap();

        // The hit is identified by the indexed name key.
        let hits = index.search("compile", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "build");
        assert!(hits[0].snippet.contains("compile"), "{}", hits[0].snippet);
        assert!(hits[0].bm25 < 0.0);
        assert!((0.0..=1.0).contains(&hits[0].score));

        let hits = index.search("handbook", 10).await.unwrap();
        assert_eq!(hits[0].name, "docs");

        let hits = index.search("guide", 10).await.unwrap();
        assert_eq!(hits[0].name, "docs", "the description column is indexed");

        let hits = index.search("shipping", 10).await.unwrap();
        assert_eq!(hits[0].name, "release", "the tags column is indexed");
        let hits = index.search("cargo", 10).await.unwrap();
        assert_eq!(hits[0].name, "release");
    }

    #[tokio::test]
    async fn the_better_match_ranks_first() {
        let index = FtsIndex::in_memory().await.unwrap();
        index
            .upsert(
                &fact("dense", "compile compile compile the workspace")
                    .with_description("compile notes"),
            )
            .await
            .unwrap();
        index
            .upsert(&fact("sparse", "compile the workspace"))
            .await
            .unwrap();

        let hits = index.search("compile", 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].name, "dense", "the denser match ranks first");
        assert_eq!(hits[1].name, "sparse");
        assert!(hits[0].bm25 < hits[1].bm25);
    }

    #[tokio::test]
    async fn remove_count_and_clear_are_observable() {
        let index = FtsIndex::in_memory().await.unwrap();
        assert_eq!(index.count().await.unwrap(), 0);

        index.upsert(&fact("alpha", "first body")).await.unwrap();
        index.upsert(&fact("beta", "second body")).await.unwrap();
        assert_eq!(index.count().await.unwrap(), 2);

        assert!(index.remove("alpha").await.unwrap());
        assert!(!index.remove("alpha").await.unwrap());
        assert_eq!(index.count().await.unwrap(), 1);
        assert_eq!(index.search("first", 10).await.unwrap().len(), 0);

        index.clear().await.unwrap();
        assert_eq!(index.count().await.unwrap(), 0);
        assert_eq!(index.search("second", 10).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn upsert_with_the_same_name_replaces_the_row() {
        let index = FtsIndex::in_memory().await.unwrap();
        index
            .upsert(&fact("dup", "the first version"))
            .await
            .unwrap();
        index
            .upsert(&fact("dup", "the second version"))
            .await
            .unwrap();

        assert_eq!(index.count().await.unwrap(), 1);
        assert_eq!(index.search("first", 10).await.unwrap().len(), 0);
        let hits = index.search("second", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "dup");
    }

    #[tokio::test]
    async fn rebuild_replaces_the_whole_index() {
        let index = FtsIndex::in_memory().await.unwrap();
        index.upsert(&fact("stale", "old words")).await.unwrap();

        let facts = vec![
            fact("one", "the first body"),
            fact("two", "the second body"),
        ];
        let indexed = index.rebuild(&facts).await.unwrap();

        assert_eq!(indexed, 2);
        assert_eq!(index.count().await.unwrap(), 2);
        assert_eq!(
            index.search("old", 10).await.unwrap().len(),
            0,
            "the stale row is gone"
        );
        let hits = index.search("second", 10).await.unwrap();
        assert_eq!(hits[0].name, "two");
    }

    #[tokio::test]
    async fn rebuild_on_an_empty_input_empties_the_index() {
        let index = FtsIndex::in_memory().await.unwrap();
        index.upsert(&fact("stale", "old words")).await.unwrap();
        assert_eq!(index.rebuild(Vec::new()).await.unwrap(), 0);
        assert_eq!(index.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn unicode_content_round_trips() {
        let index = FtsIndex::in_memory().await.unwrap();
        index
            .upsert(
                &fact("构建流程", "项目使用 cargo 构建并发布")
                    .with_description("构建说明")
                    .with_tags(vec!["构建".to_owned()]),
            )
            .await
            .unwrap();

        let hits = index.search("构建", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "构建流程");
        assert_eq!(index.search("cargo", 10).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn open_path_persists_across_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("coding_facts_fts.db");
        {
            let index = FtsIndex::open_path(&path).await.unwrap();
            index
                .upsert(&fact("persisted", "durable full text"))
                .await
                .unwrap();
            assert_eq!(index.count().await.unwrap(), 1);
            index.pool().close().await;
        }

        let reopened = FtsIndex::open_path(&path).await.unwrap();
        assert_eq!(reopened.count().await.unwrap(), 1);
        let hits = reopened.search("durable", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "persisted");
    }

    #[tokio::test]
    async fn open_accepts_a_url_and_rejects_foreign_schemes() {
        let index = FtsIndex::open("sqlite::memory:").await.unwrap();
        index.upsert(&fact("url", "opened by url")).await.unwrap();
        assert_eq!(index.count().await.unwrap(), 1);

        let error = FtsIndex::open("postgres://localhost/facts")
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("sqlite"), "{error}");
    }

    #[tokio::test]
    async fn from_pool_applies_the_schema() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let index = FtsIndex::from_pool(pool).await.unwrap();
        index
            .upsert(&fact("pooled", "from a caller pool"))
            .await
            .unwrap();
        assert_eq!(index.count().await.unwrap(), 1);
        assert_eq!(index.pool().size(), 1);
    }

    #[tokio::test]
    async fn empty_and_punctuation_queries_are_safe() {
        let index = FtsIndex::in_memory().await.unwrap();
        index.upsert(&fact("a", "compile the crate")).await.unwrap();

        for query in [
            "",
            " ",
            "   \n\t ",
            "!@#$%^&*()",
            "___",
            "-",
            "\"",
            "'",
            "NEAR(",
            "a\" OR b",
            "OR AND NOT",
            "coding_facts_fts MATCH",
            "*",
            "^",
            "( )",
        ] {
            let hits = index.search(query, 5).await.unwrap_or_else(|error| {
                panic!("query {query:?} must not fail: {error}");
            });
            assert!(
                hits.iter().all(|hit| hit.name == "a"),
                "query {query:?} produced a foreign hit"
            );
        }

        for query in ["", "   ", "!@#$%^&*()", "___", "NEAR("] {
            let hits = index.search(query, 5).await.unwrap();
            assert_eq!(hits.len(), 0, "query {query:?} must not match");
        }

        // A pathological single token is still just a quoted phrase.
        let long = "a".repeat(500);
        assert_eq!(index.search(&long, 5).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn limit_zero_is_raised_to_one() {
        let index = FtsIndex::in_memory().await.unwrap();
        index.upsert(&fact("a", "compile the crate")).await.unwrap();
        index
            .upsert(&fact("b", "compile the binary"))
            .await
            .unwrap();
        assert_eq!(index.search("compile", 0).await.unwrap().len(), 1);
        assert_eq!(index.search("compile", 10).await.unwrap().len(), 2);
    }

    #[test]
    fn match_expression_quotes_and_caps_the_tokens() {
        assert_eq!(
            match_expression("build the crate").unwrap().as_deref(),
            Some("\"build\" OR \"the\" OR \"crate\"")
        );
        assert_eq!(match_expression("").unwrap(), None);
        assert_eq!(match_expression("   \n\t").unwrap(), None);
        assert_eq!(match_expression("!@#$%^&*()").unwrap(), None);
        assert_eq!(match_expression("___").unwrap(), None);

        // FTS5 syntax characters are separators, so they never reach the expression.
        assert_eq!(
            match_expression("\"quoted\" (parens) ^star*")
                .unwrap()
                .as_deref(),
            Some("\"quoted\" OR \"parens\" OR \"star\"")
        );

        let many = (0..12)
            .map(|index| format!("t{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let expression = match_expression(&many).unwrap().unwrap();
        assert_eq!(expression.matches(" OR ").count(), MAX_QUERY_TOKENS - 1);
        assert!(expression.starts_with("\"t0\" OR \"t1\""));
        assert!(!expression.contains("\"t10\""));

        // The `\w+` tokenizer never yields a quote, so this only guards the escaping
        // rule itself.
        assert_eq!(quote_token("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn score_is_normalized_from_bm25() {
        assert!((normalize_score(-3.0) - 0.25).abs() < f32::EPSILON);
        assert!((normalize_score(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((normalize_score(1.0) - 0.5).abs() < f32::EPSILON);
        for bm25 in [-100.0, -1.5, -0.01, 0.0, 0.5, 12.0] {
            let score = normalize_score(bm25);
            assert!((0.0..=1.0).contains(&score), "bm25 {bm25} -> {score}");
        }
        assert!((normalize_score(f64::NAN)).abs() < f32::EPSILON);
        assert!((normalize_score(f64::NEG_INFINITY)).abs() < f32::EPSILON);
    }

    #[test]
    fn memory_urls_are_detected() {
        assert!(is_memory_url("sqlite::memory:"));
        assert!(is_memory_url("sqlite://:memory:"));
        assert!(is_memory_url("sqlite://facts.db?mode=memory"));
        assert!(!is_memory_url("sqlite://facts.db"));
        assert!(!is_memory_url("sqlite://facts.db?mode=rw"));
    }
}
