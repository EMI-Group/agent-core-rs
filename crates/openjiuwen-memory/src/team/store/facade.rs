//! [`TeamMemory`]: the ergonomic entry point over a [`TeamMemoryStore`].
//!
//! The facade is deliberately thin: it owns the store handle and the default importance,
//! validates and materializes inputs, translates a missing record into
//! [`Error::NotFound`], and carries cancellation into the store calls. It never caches, so
//! two facades over the same store observe the same rows.
//!
//! # Cancellation
//!
//! The methods that take [`CallOptions`] (`remember`, `remember_many`, `update`,
//! `from_jsonl`) check the token *before* doing any work and again after the last store
//! call, and every store call is raced against the token with [`CallOptions::run`], so a
//! cancelled call returns [`Error::Cancelled`] instead of a half-finished batch.
//! `remember_many`/`from_jsonl` stop at the first failure; records already written stay
//! written (there is no cross-record transaction in the team memory path).
use super::memory::InMemoryTeamMemoryStore;
use super::{TeamMemoryStore, TeamSearchOptions};
use crate::record::CallOptions;
use crate::team::model::{
    NewTeamMemory, TeamMemoryFilter, TeamMemoryRecord, TeamMemoryScope, TeamMemoryStats,
    TeamMemoryUpdate, TeamSearchHit, DEFAULT_IMPORTANCE,
};
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use std::sync::Arc;
use tracing::debug;
/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// A team memory facade bound to one store.
///
/// Construct it with [`TeamMemory::in_memory`] for tests and single-process teams, or with
/// [`TeamMemory::new`] to inject a persistent [`TeamMemoryStore`].
pub struct TeamMemory {
    store: Arc<dyn TeamMemoryStore>,
    default_importance: f32,
}
impl TeamMemory {
    /// Builds a facade over `store`.
    #[must_use]
    pub fn new(store: Arc<dyn TeamMemoryStore>) -> Self {
        Self {
            store,
            default_importance: DEFAULT_IMPORTANCE,
        }
    }
    /// Builds a facade over a fresh [`InMemoryTeamMemoryStore`].
    #[must_use]
    pub fn in_memory() -> Self {
        Self::new(Arc::new(InMemoryTeamMemoryStore::new()))
    }
    /// Opens a SQLite-backed team memory at `path` (feature `sqlite`).
    ///
    /// The parent directory is created when needed and the schema is applied, so the path
    /// may point at a directory that does not exist yet; reopening the same path keeps the
    /// committed records.
    ///
    /// # Errors
    ///
    /// Propagates the store's construction error: [`Error::Memory`] when the parent
    /// directory cannot be created and [`Error::External`] when the database cannot be
    /// opened or the schema cannot be applied.
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(Arc::new(
            crate::team::sqlite::SqliteTeamMemoryStore::open(path).await?,
        )))
    }
    /// Sets the importance used when an input does not carry one.
    ///
    /// The value is clamped into `0.0..=1.0`; a non-finite value keeps
    /// [`DEFAULT_IMPORTANCE`] instead of poisoning every later write.
    #[must_use]
    pub fn with_default_importance(mut self, importance: f32) -> Self {
        self.default_importance = if importance.is_finite() {
            importance.clamp(0.0, 1.0)
        } else {
            DEFAULT_IMPORTANCE
        };
        self
    }
    /// The importance applied to inputs without one.
    #[must_use]
    pub fn default_importance(&self) -> f32 {
        self.default_importance
    }
    /// The underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn TeamMemoryStore> {
        &self.store
    }
    /// Materializes and stores one new team memory.
    ///
    /// The input is validated and normalized by [`NewTeamMemory::to_record`] (a missing
    /// id is generated, tags and visibility are normalized, and the facade's
    /// [`TeamMemory::default_importance`] fills an absent importance), then upserted.
    pub async fn remember(
        &self,
        input: NewTeamMemory,
        call: &CallOptions,
    ) -> Result<TeamMemoryRecord> {
        call.check()?;
        let record = input.to_record(Utc::now(), self.default_importance)?;
        let stored = call.run(self.store.upsert(&record)).await??;
        call.check()?;
        debug!(target: LOG_TARGET, team_id = %stored.team_id, id = %stored.id, "remembered team memory");
        Ok(stored)
    }
    /// Materializes and stores several team memories in order.
    ///
    /// Processing is sequential and stops at the first error; the records written before
    /// the failure stay in the store.
    pub async fn remember_many(
        &self,
        inputs: Vec<NewTeamMemory>,
        call: &CallOptions,
    ) -> Result<Vec<TeamMemoryRecord>> {
        call.check()?;
        let mut stored = Vec::with_capacity(inputs.len());
        for input in inputs {
            let record = input.to_record(Utc::now(), self.default_importance)?;
            let saved = call.run(self.store.upsert(&record)).await??;
            call.check()?;
            stored.push(saved);
        }
        Ok(stored)
    }
    /// Fetches one record by team and id.
    ///
    /// Returns `None` for an unknown id or another team's record; expired records are
    /// returned (see [`TeamMemoryStore::get`]).
    pub async fn get(&self, team_id: &str, id: &str) -> Result<Option<TeamMemoryRecord>> {
        self.store.get(team_id, id).await
    }
    /// Applies a partial update to an existing record.
    ///
    /// Returns [`Error::NotFound`] when the id is unknown or belongs to another team; the
    /// update is validated and applied by [`TeamMemoryUpdate::apply`] (which stamps
    /// `updated_at`) before the record is upserted again.
    pub async fn update(
        &self,
        team_id: &str,
        id: &str,
        update: &TeamMemoryUpdate,
        call: &CallOptions,
    ) -> Result<TeamMemoryRecord> {
        call.check()?;
        update.validate()?;
        let mut record = call
            .run(self.store.get(team_id, id))
            .await??
            .ok_or_else(|| {
                Error::not_found(format!("team memory `{id}` not found in team `{team_id}`"))
            })?;
        update.apply(&mut record, Utc::now())?;
        let stored = call.run(self.store.upsert(&record)).await??;
        call.check()?;
        debug!(target: LOG_TARGET, team_id = %team_id, id = %id, "updated team memory");
        Ok(stored)
    }
    /// Lists the records matching `filter`.
    ///
    /// See [`TeamMemoryStore::list`] for the ordering and pagination contract.
    pub async fn list(&self, filter: &TeamMemoryFilter) -> Result<Vec<TeamMemoryRecord>> {
        self.store.list(filter).await
    }
    /// Searches the records `scope` may read.
    ///
    /// The ranking formula is documented on [`super`]; the cancellation token carried by
    /// `options` is honored before and after the store call.
    pub async fn search(
        &self,
        scope: &TeamMemoryScope,
        query: &str,
        options: &TeamSearchOptions,
    ) -> Result<Vec<TeamSearchHit>> {
        options.call.check()?;
        let hits = options
            .call
            .run(self.store.search(scope, query, options))
            .await??;
        options.call.check()?;
        Ok(hits)
    }
    /// Deletes one record, returning `false` when it was already absent.
    pub async fn forget(&self, team_id: &str, id: &str) -> Result<bool> {
        self.store.delete(team_id, id).await
    }
    /// Deletes every record matching `filter` and returns how many were removed.
    pub async fn clear(&self, filter: &TeamMemoryFilter) -> Result<usize> {
        self.store.clear(filter).await
    }
    /// Aggregates the records matching `filter`.
    pub async fn stats(&self, filter: &TeamMemoryFilter) -> Result<TeamMemoryStats> {
        self.store.stats(filter).await
    }
    /// Sets or clears the expiry of one record, returning `false` when it was not found.
    pub async fn set_expiry(
        &self,
        team_id: &str,
        id: &str,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<bool> {
        self.store.set_expiry(team_id, id, expires_at).await
    }
    /// Prunes every record that has expired as of now, returning how many were removed.
    pub async fn prune_expired(&self) -> Result<usize> {
        self.store.prune_expired(Utc::now()).await
    }
    /// Serializes the records matching `filter` as JSON Lines.
    ///
    /// One compact JSON object per line (the [`TeamMemoryRecord`] serde shape, so the
    /// output round-trips through [`TeamMemory::from_jsonl`]), each terminated by `\n`.
    /// An empty match yields an empty string.
    pub async fn to_jsonl(&self, filter: &TeamMemoryFilter) -> Result<String> {
        let records = self.store.list(filter).await?;
        let mut output = String::new();
        for record in records {
            let line = serde_json::to_string(&record).map_err(Error::external)?;
            output.push_str(&line);
            output.push('\n');
        }
        Ok(output)
    }
    /// Imports JSON Lines produced by [`TeamMemory::to_jsonl`].
    ///
    /// Blank lines are skipped; every other line must parse into a valid
    /// [`TeamMemoryRecord`], and a malformed line is [`Error::Validation`] naming its
    /// 1-based line number (never a silent skip). Each valid record is upserted in file
    /// order and the stored records are returned. Importing into the same store is
    /// idempotent for unchanged records (`upsert` preserves `created_at`).
    pub async fn from_jsonl(
        &self,
        jsonl: &str,
        call: &CallOptions,
    ) -> Result<Vec<TeamMemoryRecord>> {
        call.check()?;
        let mut stored = Vec::new();
        for (index, line) in jsonl.lines().enumerate() {
            let number = index + 1;
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str::<TeamMemoryRecord>(line).map_err(|error| {
                Error::validation(format!(
                    "team memory jsonl line {number} is not a team memory record: {error}"
                ))
            })?;
            record
                .validate()
                .map_err(|error| with_line_number(number, error))?;
            let saved = call.run(self.store.upsert(&record)).await??;
            call.check()?;
            stored.push(saved);
        }
        debug!(target: LOG_TARGET, imported = stored.len(), "imported team memory jsonl");
        Ok(stored)
    }
}
/// Prefixes a validation error with the 1-based JSONL line number.
fn with_line_number(number: usize, error: Error) -> Error {
    match error {
        Error::Validation(message) => {
            Error::validation(format!("team memory jsonl line {number}: {message}"))
        }
        other => other,
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::team::model::{TeamMemoryKind, TeamVisibility};
    use openjiuwen_common::error::Error;
    use tokio_util::sync::CancellationToken;
    fn cancelled_call() -> CallOptions {
        let token = CancellationToken::new();
        token.cancel();
        CallOptions::new().with_cancellation(token)
    }
    fn decision(team: &str, content: &str) -> NewTeamMemory {
        NewTeamMemory::decision(team, content).with_member("alice")
    }
    #[tokio::test]
    async fn in_memory_facade_smoke() {
        let memory = TeamMemory::in_memory();
        assert!((memory.default_importance() - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
        let record = memory
            .remember(decision("review-team", "use sqlite"), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(record.team_id, "review-team");
        assert_eq!(record.member_id.as_deref(), Some("alice"));
        assert_eq!(record.id.len(), 24);
        let stored = memory.get("review-team", &record.id).await.unwrap();
        assert_eq!(stored.as_ref(), Some(&record));
        assert!(memory
            .get("other-team", &record.id)
            .await
            .unwrap()
            .is_none());
    }
    #[tokio::test]
    async fn remember_update_forget_and_clear() {
        let memory = TeamMemory::in_memory();
        let call = CallOptions::new();
        let first = memory
            .remember(decision("review-team", "use sqlite"), &call)
            .await
            .unwrap();
        let second = memory
            .remember(decision("review-team", "ship on friday"), &call)
            .await
            .unwrap();
        let update = TeamMemoryUpdate::default().content("use sqlite for the shared store");
        let updated = memory
            .update("review-team", &first.id, &update, &call)
            .await
            .unwrap();
        assert_eq!(updated.content, "use sqlite for the shared store");
        assert_eq!(updated.created_at, first.created_at);
        assert!(updated.updated_at >= first.updated_at);
        let listed = memory
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert_eq!(listed.len(), 2);
        assert!(memory.forget("review-team", &second.id).await.unwrap());
        assert!(!memory.forget("review-team", &second.id).await.unwrap());
        assert_eq!(
            memory
                .clear(&TeamMemoryFilter::for_team("review-team"))
                .await
                .unwrap(),
            1
        );
        assert!(memory
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap()
            .is_empty());
    }
    #[tokio::test]
    async fn update_reports_not_found() {
        let memory = TeamMemory::in_memory();
        let update = TeamMemoryUpdate::default().content("anything");
        let error = memory
            .update("review-team", "missing", &update, &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotFound(_)));
        assert!(error
            .to_string()
            .contains("team memory `missing` not found in team `review-team`"));
    }
    #[tokio::test]
    async fn default_importance_is_applied_and_clamped() {
        let memory = TeamMemory::in_memory().with_default_importance(0.9);
        assert!((memory.default_importance() - 0.9).abs() < f32::EPSILON);
        let record = memory
            .remember(decision("review-team", "content"), &CallOptions::new())
            .await
            .unwrap();
        assert!((record.importance - 0.9).abs() < f32::EPSILON);
        let memory = TeamMemory::in_memory().with_default_importance(7.0);
        assert!((memory.default_importance() - 1.0).abs() < f32::EPSILON);
        let memory = TeamMemory::in_memory().with_default_importance(f32::NAN);
        assert!((memory.default_importance() - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
        let memory = TeamMemory::in_memory().with_default_importance(0.2);
        let record = memory
            .remember(
                decision("review-team", "content").with_importance(0.75),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!((record.importance - 0.75).abs() < f32::EPSILON);
    }
    #[tokio::test]
    async fn cancelled_calls_return_cancelled() {
        let memory = TeamMemory::in_memory();
        let error = memory
            .remember(decision("review-team", "content"), &cancelled_call())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert!(memory
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap()
            .is_empty());
        let error = memory
            .from_jsonl("{\"id\":\"x\"}\n", &cancelled_call())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        let error = memory
            .remember_many(vec![decision("review-team", "content")], &cancelled_call())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        let options = TeamSearchOptions::default().with_cancellation({
            let token = CancellationToken::new();
            token.cancel();
            token
        });
        let error = memory
            .search(&TeamMemoryScope::new("review-team"), "content", &options)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
    }
    #[tokio::test]
    async fn jsonl_round_trip_through_a_second_facade() {
        let source = TeamMemory::in_memory();
        let call = CallOptions::new();
        let mut tagged = decision("review-team", "sqlite for the shared store")
            .with_title("Storage choice")
            .with_tag("Storage")
            .with_metadata("issue", serde_json::json!(42))
            .with_source("leader-extraction")
            .with_visibility(TeamVisibility::Private);
        tagged.kind = TeamMemoryKind::Custom("release_check".to_owned());
        let record = source.remember(tagged, &call).await.unwrap();
        source
            .remember(decision("review-team", "ship on friday"), &call)
            .await
            .unwrap();
        let jsonl = source
            .to_jsonl(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert!(jsonl.ends_with('\n'));
        assert_eq!(jsonl.lines().count(), 2);
        let target = TeamMemory::in_memory();
        let imported = target.from_jsonl(&jsonl, &call).await.unwrap();
        assert_eq!(imported.len(), 2);
        let restored = target
            .get("review-team", &record.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(restored, record);
        assert_eq!(
            target
                .list(&TeamMemoryFilter::for_team("review-team"))
                .await
                .unwrap()
                .len(),
            2
        );
    }
    #[tokio::test]
    async fn to_jsonl_of_an_empty_match_is_empty() {
        let memory = TeamMemory::in_memory();
        let jsonl = memory
            .to_jsonl(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert!(jsonl.is_empty());
    }
    #[tokio::test]
    async fn malformed_jsonl_names_the_line_number() {
        let memory = TeamMemory::in_memory();
        let call = CallOptions::new();
        let record = memory
            .remember(decision("review-team", "content"), &call)
            .await
            .unwrap();
        let good = serde_json::to_string(&record).unwrap();
        let jsonl = format!("{good}\n\n{{\"id\": 7}}\n");
        let error = memory.from_jsonl(&jsonl, &call).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        let message = error.to_string();
        assert!(message.contains("line 3"), "{message}");
        let invalid = format!("{good}\n{{\"id\":\"\",\"team_id\":\"\"}}\n");
        let error = memory.from_jsonl(&invalid, &call).await.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("line 2"), "{message}");
        let error = memory
            .from_jsonl("only-one-line-has-no-record", &call)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("line 1"));
    }
    #[tokio::test]
    async fn remember_many_stops_at_the_first_error() {
        let memory = TeamMemory::in_memory();
        let mut bad = decision("review-team", "content");
        bad.content = String::new();
        let error = memory
            .remember_many(
                vec![
                    decision("review-team", "first"),
                    bad,
                    decision("review-team", "third"),
                ],
                &CallOptions::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        let listed = memory
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].content, "first");
    }
    #[tokio::test]
    async fn expire_and_prune_through_the_facade() {
        let memory = TeamMemory::in_memory();
        let call = CallOptions::new();
        let record = memory
            .remember(decision("review-team", "content"), &call)
            .await
            .unwrap();
        let past = Utc::now() - chrono::Duration::seconds(1);
        assert!(memory
            .set_expiry("review-team", &record.id, Some(past))
            .await
            .unwrap());
        assert!(memory
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap()
            .is_empty());
        let stats = memory
            .stats(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert_eq!(stats.total, 0);
        assert_eq!(memory.prune_expired().await.unwrap(), 1);
        assert!(memory
            .get("review-team", &record.id)
            .await
            .unwrap()
            .is_none());
        assert_eq!(memory.prune_expired().await.unwrap(), 0);
    }
    #[tokio::test]
    async fn search_and_stats_through_the_facade() {
        let memory = TeamMemory::in_memory();
        let call = CallOptions::new();
        memory
            .remember(
                decision("review-team", "sqlite for the shared store")
                    .with_tag("storage")
                    .with_importance(0.9),
                &call,
            )
            .await
            .unwrap();
        memory
            .remember(decision("review-team", "ship on friday"), &call)
            .await
            .unwrap();
        let hits = memory
            .search(
                &TeamMemoryScope::member("review-team", "alice"),
                "sqlite storage",
                &TeamSearchOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.importance, 0.9);
        let stats = memory
            .stats(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.by_member.get("alice"), Some(&2));
    }
}
