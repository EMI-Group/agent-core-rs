//! Test-only seams around the team-memory wiring: a store that fails on demand.
//!
//! The engine's session-end flush writes its whole extraction batch through
//! [`TeamMemoryStore::upsert`](openjiuwen_memory::team::TeamMemoryStore::upsert) and keeps the
//! buffered turns when that write fails, so a suite that wants to exercise the failure path needs
//! a store that can be made to refuse. [`FailingTeamMemoryStore`] is that store: a working
//! [`InMemoryTeamMemoryStore`] that answers a typed [`Error::Memory`] from `upsert` while its flag
//! is set, and behaves like the plain store again once the flag is cleared.
//!
//! The intended use is [`TeamMemoryWiring::with_store_for_tests`]: open a wiring over the failing
//! store, buffer one turn, flush to observe the error and the retained buffer, clear the flag, and
//! flush again to observe the retained batch converge.
//!
//! The whole module is `#[cfg(test)]`; nothing here is part of the public surface.
//!
//! [`TeamMemoryWiring::with_store_for_tests`]: crate::memory::TeamMemoryWiring

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::team::{
    InMemoryTeamMemoryStore, TeamMemoryFilter, TeamMemoryRecord, TeamMemoryScope, TeamMemoryStats,
    TeamMemoryStore, TeamSearchHit, TeamSearchOptions,
};

/// The message of the error [`FailingTeamMemoryStore`] answers while it is failing.
pub(crate) const SCRIPTED_FAILURE: &str = "scripted store failure";

/// An in-memory team-memory store whose `upsert` can be made to fail.
///
/// Every method delegates to [`InMemoryTeamMemoryStore`], except `upsert`, which answers
/// [`Error::Memory`] with [`SCRIPTED_FAILURE`] while [`Self::is_failing`] is true. Reads therefore
/// keep working during a failure, so a suite can assert exactly which rows were stored.
#[derive(Debug, Default)]
pub(crate) struct FailingTeamMemoryStore {
    /// The store that holds the rows.
    inner: InMemoryTeamMemoryStore,
    /// Whether `upsert` refuses.
    failing: AtomicBool,
}

impl FailingTeamMemoryStore {
    /// Builds a store that already fails.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            inner: InMemoryTeamMemoryStore::new(),
            failing: AtomicBool::new(true),
        }
    }

    /// Builds the store behind a shared handle, ready to hand to a wiring.
    #[must_use]
    pub(crate) fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Whether `upsert` currently refuses.
    pub(crate) fn is_failing(&self) -> bool {
        self.failing.load(Ordering::SeqCst)
    }

    /// Makes `upsert` refuse, or stop refusing.
    pub(crate) fn set_failing(&self, failing: bool) {
        self.failing.store(failing, Ordering::SeqCst);
    }

    /// Makes `upsert` succeed again, keeping every row already stored.
    pub(crate) fn stop_failing(&self) {
        self.set_failing(false);
    }
}

#[async_trait]
impl TeamMemoryStore for FailingTeamMemoryStore {
    async fn upsert(&self, record: &TeamMemoryRecord) -> Result<TeamMemoryRecord> {
        if self.is_failing() {
            return Err(Error::memory(SCRIPTED_FAILURE));
        }
        self.inner.upsert(record).await
    }

    async fn get(&self, team_id: &str, id: &str) -> Result<Option<TeamMemoryRecord>> {
        self.inner.get(team_id, id).await
    }

    async fn list(&self, filter: &TeamMemoryFilter) -> Result<Vec<TeamMemoryRecord>> {
        self.inner.list(filter).await
    }

    async fn search(
        &self,
        scope: &TeamMemoryScope,
        query: &str,
        options: &TeamSearchOptions,
    ) -> Result<Vec<TeamSearchHit>> {
        self.inner.search(scope, query, options).await
    }

    async fn delete(&self, team_id: &str, id: &str) -> Result<bool> {
        self.inner.delete(team_id, id).await
    }

    async fn clear(&self, filter: &TeamMemoryFilter) -> Result<usize> {
        self.inner.clear(filter).await
    }

    async fn stats(&self, filter: &TeamMemoryFilter) -> Result<TeamMemoryStats> {
        self.inner.stats(filter).await
    }

    async fn set_expiry(
        &self,
        team_id: &str,
        id: &str,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<bool> {
        self.inner.set_expiry(team_id, id, expires_at).await
    }

    async fn prune_expired(&self, now: DateTime<Utc>) -> Result<usize> {
        self.inner.prune_expired(now).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::TimeZone as _;

    use super::*;
    use crate::memory::{MemoryConfig, TeamMemoryWiring};

    /// A fixed timestamp, so nothing in these tests depends on the wall clock.
    fn at(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).unwrap()
    }

    /// Opens a wiring over `store` in `dir`.
    async fn opened(
        dir: &tempfile::TempDir,
        store: &Arc<FailingTeamMemoryStore>,
    ) -> Arc<TeamMemoryWiring> {
        TeamMemoryWiring::with_store_for_tests(
            "team-1",
            MemoryConfig::default(),
            dir.path(),
            Arc::clone(store) as Arc<dyn TeamMemoryStore>,
        )
        .await
        .unwrap()
    }

    /// Lists everything the wiring's store holds for the team.
    async fn stored(wiring: &TeamMemoryWiring) -> Vec<TeamMemoryRecord> {
        wiring
            .memory()
            .list(&TeamMemoryFilter::for_team("team-1"))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn a_failing_store_surfaces_its_own_error_and_keeps_the_turn_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let store = FailingTeamMemoryStore::shared();
        assert!(store.is_failing());
        let wiring = opened(&dir, &store).await;
        wiring.on_session_start("s-1").await.unwrap();
        wiring
            .on_turn_message(
                "dev-1",
                None,
                "We decided to split the release train in two.",
                at(1_700_000_000),
            )
            .await
            .unwrap();
        assert_eq!(wiring.hooks().pending_turns(), 1);
        assert!(matches!(stored(&wiring).await.as_slice(), []));

        let error = wiring.flush().await.unwrap_err();
        assert!(
            matches!(&error, Error::Memory(message) if message.contains(SCRIPTED_FAILURE)),
            "{error}"
        );
        assert_eq!(
            wiring.hooks().pending_turns(),
            1,
            "a failed write keeps the buffer for a retry"
        );
        assert!(
            matches!(stored(&wiring).await.as_slice(), []),
            "the failure stored nothing"
        );
    }

    #[tokio::test]
    async fn a_retry_after_the_store_recovers_writes_the_retained_batch_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = FailingTeamMemoryStore::shared();
        let wiring = opened(&dir, &store).await;
        wiring.on_session_start("s-1").await.unwrap();
        wiring
            .on_turn_message(
                "dev-1",
                Some("dev-2"),
                "We decided to split the release train in two.",
                at(1_700_000_000),
            )
            .await
            .unwrap();
        assert!(wiring.flush().await.is_err());
        assert_eq!(wiring.hooks().pending_turns(), 1);

        store.stop_failing();
        assert!(!store.is_failing());
        let flush = wiring.flush().await.unwrap();
        assert_eq!(flush.turns, 1, "the retained turn is what the retry wrote");
        assert!(
            matches!(flush.records.as_slice(), [_, ..]),
            "the rule extractor wrote the buffered turn"
        );
        assert!(flush.degraded.is_none());
        assert_eq!(wiring.hooks().pending_turns(), 0);

        assert!(
            wiring.flush().await.unwrap().is_empty(),
            "the retry cleared the buffer"
        );
        let records = stored(&wiring).await;
        assert_eq!(
            records.len(),
            flush.records.len(),
            "the retry converged instead of duplicating rows"
        );
        assert!(records
            .iter()
            .all(|record| record.session_id.as_deref() == Some("s-1")));
    }
}
