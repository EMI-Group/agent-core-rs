//! [`InMemoryTeamMemoryStore`]: the reference [`TeamMemoryStore`] implementation.
//!
//! One process-local [`parking_lot::RwLock`]`<BTreeMap<id, record>>` behind the trait.
//! Every operation copies what it needs out of the lock and never holds the guard across
//! an `.await` (there is none inside a guard's scope), so the store is `Send + Sync` and
//! safe to share through `Arc<dyn TeamMemoryStore>`.
//!
//! The store keeps expired records: expiry is a read-view concern
//! ([`TeamMemoryFilter::include_expired`], [`TeamMemoryScope`] searches, and
//! [`TeamMemoryStore::prune_expired`]), so a record that expires stays inspectable until
//! something prunes it. `upsert` preserves the `created_at` of an existing row, so an
//! update never rewrites history.
use super::{rank_candidates, TeamMemoryStore, TeamSearchOptions};
use crate::team::model::{
    TeamMemoryFilter, TeamMemoryRecord, TeamMemoryScope, TeamMemoryStats, TeamSearchHit,
    TEAM_WIDE_MEMBER_KEY,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use tracing::debug;
/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// A process-local, `parking_lot`-backed team memory store.
///
/// Rows are keyed by record id; `team_id` is checked on every read and delete, so a
/// caller holding an id from another team cannot reach it. Cheap to clone the results
/// but not the store itself — share it as `Arc<InMemoryTeamMemoryStore>`.
#[derive(Debug, Default)]
pub struct InMemoryTeamMemoryStore {
    rows: RwLock<BTreeMap<String, TeamMemoryRecord>>,
}
impl InMemoryTeamMemoryStore {
    /// Builds an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Number of stored records, including expired ones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.read().len()
    }
    /// Returns `true` when the store holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.read().is_empty()
    }
    /// Clones every stored record out of the lock.
    fn snapshot(&self) -> Vec<TeamMemoryRecord> {
        self.rows.read().values().cloned().collect()
    }
    /// Removes every record matching `filter` and returns how many were removed.
    fn remove_matching(&self, filter: &TeamMemoryFilter, now: DateTime<Utc>) -> usize {
        let mut rows = self.rows.write();
        let doomed: Vec<String> = rows
            .values()
            .filter(|record| filter.matches(record, now))
            .map(|record| record.id.clone())
            .collect();
        for id in &doomed {
            rows.remove(id);
        }
        doomed.len()
    }
}
/// Sorts by `updated_at` descending then `id` ascending, and applies `offset`/`limit`.
fn paginate(
    mut records: Vec<TeamMemoryRecord>,
    filter: &TeamMemoryFilter,
) -> Vec<TeamMemoryRecord> {
    records.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    let offset = filter.offset.unwrap_or(0);
    let limit = filter.limit.unwrap_or(usize::MAX);
    records.into_iter().skip(offset).take(limit).collect()
}
#[allow(clippy::unused_async_trait_impl)] // every method is synchronous; the trait is async
#[async_trait]
impl TeamMemoryStore for InMemoryTeamMemoryStore {
    async fn upsert(&self, record: &TeamMemoryRecord) -> Result<TeamMemoryRecord> {
        record.validate()?;
        let mut stored = record.clone();
        {
            let mut rows = self.rows.write();
            if let Some(existing) = rows.get(&stored.id) {
                stored.created_at = existing.created_at;
            }
            rows.insert(stored.id.clone(), stored.clone());
        }
        debug!(
            target: LOG_TARGET,
            team_id = %stored.team_id,
            id = %stored.id,
            "stored team memory record"
        );
        Ok(stored)
    }
    async fn get(&self, team_id: &str, id: &str) -> Result<Option<TeamMemoryRecord>> {
        let rows = self.rows.read();
        Ok(rows
            .get(id)
            .filter(|record| record.team_id == team_id)
            .cloned())
    }
    async fn list(&self, filter: &TeamMemoryFilter) -> Result<Vec<TeamMemoryRecord>> {
        filter.validate()?;
        let now = Utc::now();
        let matched: Vec<TeamMemoryRecord> = self
            .snapshot()
            .into_iter()
            .filter(|record| filter.matches(record, now))
            .collect();
        Ok(paginate(matched, filter))
    }
    async fn search(
        &self,
        scope: &TeamMemoryScope,
        query: &str,
        options: &TeamSearchOptions,
    ) -> Result<Vec<TeamSearchHit>> {
        scope.validate()?;
        options.validate()?;
        let candidates: Vec<TeamMemoryRecord> = self
            .snapshot()
            .into_iter()
            .filter(|record| scope.allows(record))
            .collect();
        Ok(rank_candidates(candidates, query, options, Utc::now()))
    }
    async fn delete(&self, team_id: &str, id: &str) -> Result<bool> {
        let mut rows = self.rows.write();
        if rows.get(id).is_some_and(|record| record.team_id == team_id) {
            rows.remove(id);
            drop(rows);
            debug!(target: LOG_TARGET, team_id = %team_id, id = %id, "deleted team memory record");
            Ok(true)
        } else {
            Ok(false)
        }
    }
    async fn clear(&self, filter: &TeamMemoryFilter) -> Result<usize> {
        filter.validate()?;
        let removed = self.remove_matching(filter, Utc::now());
        debug!(target: LOG_TARGET, team_id = %filter.team_id, removed, "cleared team memory records");
        Ok(removed)
    }
    async fn stats(&self, filter: &TeamMemoryFilter) -> Result<TeamMemoryStats> {
        filter.validate()?;
        let now = Utc::now();
        let mut stats = TeamMemoryStats::default();
        for record in self
            .snapshot()
            .into_iter()
            .filter(|record| filter.matches(record, now))
        {
            stats.total += 1;
            *stats
                .by_kind
                .entry(record.kind.as_str().to_owned())
                .or_insert(0) += 1;
            let member = record
                .member_id
                .clone()
                .unwrap_or_else(|| TEAM_WIDE_MEMBER_KEY.to_owned());
            *stats.by_member.entry(member).or_insert(0) += 1;
            if record.is_expired(now) {
                stats.expired += 1;
            }
            stats.oldest = Some(
                stats
                    .oldest
                    .map_or(record.created_at, |oldest| oldest.min(record.created_at)),
            );
            stats.newest = Some(
                stats
                    .newest
                    .map_or(record.created_at, |newest| newest.max(record.created_at)),
            );
        }
        Ok(stats)
    }
    async fn set_expiry(
        &self,
        team_id: &str,
        id: &str,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<bool> {
        let mut rows = self.rows.write();
        match rows.get_mut(id) {
            Some(record) if record.team_id == team_id => {
                record.expires_at = expires_at;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    async fn prune_expired(&self, now: DateTime<Utc>) -> Result<usize> {
        let mut rows = self.rows.write();
        let doomed: Vec<String> = rows
            .values()
            .filter(|record| record.is_expired(now))
            .map(|record| record.id.clone())
            .collect();
        for id in &doomed {
            rows.remove(id);
        }
        let removed = doomed.len();
        drop(rows);
        debug!(target: LOG_TARGET, removed, "pruned expired team memory records");
        Ok(removed)
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::team::model::{TeamMemoryKind, TeamVisibility};
    fn record(
        team: &str,
        member: Option<&str>,
        id: &str,
        content: &str,
        kind: TeamMemoryKind,
    ) -> TeamMemoryRecord {
        let mut record = TeamMemoryRecord::new(
            &member.map_or_else(
                || TeamMemoryScope::new(team),
                |member| TeamMemoryScope::member(team, member),
            ),
            content,
            kind,
        )
        .with_id(id);
        record.member_id = member.map(str::to_owned);
        record
    }
    fn store() -> InMemoryTeamMemoryStore {
        InMemoryTeamMemoryStore::new()
    }
    #[test]
    fn new_store_is_empty() {
        let store = store();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }
    #[tokio::test]
    async fn crud_round_trip() {
        let store = store();
        let record = record(
            "review-team",
            Some("alice"),
            "id-a",
            "use sqlite",
            TeamMemoryKind::Decision,
        );
        let stored = store.upsert(&record).await.unwrap();
        assert_eq!(stored, record);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(
            store.get("review-team", "id-a").await.unwrap(),
            Some(record.clone())
        );
        assert!(store.get("other-team", "id-a").await.unwrap().is_none());
        assert!(store.get("review-team", "missing").await.unwrap().is_none());
        assert!(store.delete("review-team", "id-a").await.unwrap());
        assert!(!store.delete("review-team", "id-a").await.unwrap());
        assert!(store.is_empty());
    }
    #[tokio::test]
    async fn upsert_keeps_created_at_and_takes_new_content() {
        let store = store();
        let mut record = record(
            "review-team",
            Some("alice"),
            "id-a",
            "first",
            TeamMemoryKind::Decision,
        );
        record.created_at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        store.upsert(&record).await.unwrap();
        let mut updated = record.clone();
        updated.content = "second".to_owned();
        updated.updated_at = DateTime::from_timestamp(1_700_000_500, 0).unwrap();
        let stored = store.upsert(&updated).await.unwrap();
        assert_eq!(stored.content, "second");
        assert_eq!(stored.created_at, record.created_at);
        assert_eq!(stored.updated_at, updated.updated_at);
        assert_eq!(store.len(), 1);
    }
    #[tokio::test]
    async fn upsert_validates_the_record() {
        let store = store();
        let mut record = record("review-team", None, "id-a", "  ", TeamMemoryKind::Task);
        record.content = "  ".to_owned();
        let err = store.upsert(&record).await.unwrap_err();
        assert!(matches!(
            err,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(store.is_empty());
    }
    #[tokio::test]
    async fn delete_rejects_another_teams_id() {
        let store = store();
        let record = record("review-team", None, "id-a", "content", TeamMemoryKind::Task);
        store.upsert(&record).await.unwrap();
        assert!(!store.delete("other-team", "id-a").await.unwrap());
        assert_eq!(store.len(), 1);
    }
    #[tokio::test]
    async fn list_orders_by_updated_at_desc_then_id_and_paginates() {
        let store = store();
        for (id, seconds) in [
            ("id-b", 1_700_000_000),
            ("id-a", 1_700_000_000),
            ("id-c", 1_700_000_100),
        ] {
            let mut record = record("review-team", None, id, "content", TeamMemoryKind::Task);
            record.created_at = DateTime::from_timestamp(seconds, 0).unwrap();
            record.updated_at = record.created_at;
            store.upsert(&record).await.unwrap();
        }
        store
            .upsert(&record(
                "other-team",
                None,
                "id-z",
                "content",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        let filter = TeamMemoryFilter::for_team("review-team");
        let listed: Vec<String> = store
            .list(&filter)
            .await
            .unwrap()
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert_eq!(listed, vec!["id-c", "id-a", "id-b"]);
        let page = store.list(&filter.clone().with_offset(1)).await.unwrap();
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].id, "id-a");
        let page = store.list(&filter.with_limit(1)).await.unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].id, "id-c");
    }
    #[tokio::test]
    async fn list_honours_filter_combinations() {
        let store = store();
        let mut tagged = record(
            "review-team",
            Some("alice"),
            "id-a",
            "use sqlite",
            TeamMemoryKind::Decision,
        );
        tagged.tags = vec!["storage".to_owned()];
        tagged.importance = 0.9;
        store.upsert(&tagged).await.unwrap();
        store
            .upsert(&record(
                "review-team",
                Some("bob"),
                "id-b",
                "ship it",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        let filter = TeamMemoryFilter::for_team("review-team").member("bob");
        let listed = store.list(&filter).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "id-b");
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.tags = vec!["storage".to_owned()];
        assert_eq!(store.list(&filter).await.unwrap().len(), 1);
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.min_importance = Some(0.85);
        assert_eq!(store.list(&filter).await.unwrap().len(), 1);
        let filter = TeamMemoryFilter::for_team("review-team").kind(TeamMemoryKind::Task);
        assert_eq!(store.list(&filter).await.unwrap().len(), 1);
        let err = store.list(&TeamMemoryFilter::default()).await.unwrap_err();
        assert!(matches!(
            err,
            openjiuwen_common::error::Error::Validation(_)
        ));
    }
    #[tokio::test]
    async fn search_visibility_hides_private_records_from_other_members() {
        let store = store();
        store
            .upsert(
                &record(
                    "review-team",
                    Some("alice"),
                    "id-private",
                    "alice sqlite preference",
                    TeamMemoryKind::MemberProfile,
                )
                .with_visibility(TeamVisibility::Private),
            )
            .await
            .unwrap();
        store
            .upsert(&record(
                "review-team",
                Some("alice"),
                "id-shared",
                "sqlite storage decision",
                TeamMemoryKind::Decision,
            ))
            .await
            .unwrap();
        let alice = TeamMemoryScope::member("review-team", "alice");
        let bob = TeamMemoryScope::member("review-team", "bob");
        let team = TeamMemoryScope::new("review-team");
        let ids = |hits: Vec<TeamSearchHit>| -> Vec<String> {
            hits.into_iter().map(|hit| hit.record.id).collect()
        };
        let alice_hits = ids(store
            .search(&alice, "sqlite", &TeamSearchOptions::default())
            .await
            .unwrap());
        assert!(alice_hits.contains(&"id-private".to_owned()));
        let bob_hits = ids(store
            .search(&bob, "sqlite", &TeamSearchOptions::default())
            .await
            .unwrap());
        assert!(!bob_hits.contains(&"id-private".to_owned()));
        assert!(bob_hits.contains(&"id-shared".to_owned()));
        let team_hits = ids(store
            .search(&team, "sqlite", &TeamSearchOptions::default())
            .await
            .unwrap());
        assert_eq!(team_hits.len(), 2);
    }
    #[tokio::test]
    async fn search_filters_kinds_tags_and_expiry() {
        let store = store();
        let mut tagged = record(
            "review-team",
            None,
            "id-tagged",
            "sqlite storage decision",
            TeamMemoryKind::Decision,
        );
        tagged.tags = vec!["storage".to_owned()];
        store.upsert(&tagged).await.unwrap();
        store
            .upsert(&record(
                "review-team",
                None,
                "id-task",
                "sqlite storage cleanup",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        let expired = record(
            "review-team",
            None,
            "id-expired",
            "sqlite storage archaeology",
            TeamMemoryKind::Decision,
        );
        store.upsert(&expired).await.unwrap();
        let past = Utc::now() - chrono::Duration::seconds(1);
        assert!(store
            .set_expiry("review-team", "id-expired", Some(past))
            .await
            .unwrap());
        let scope = TeamMemoryScope::new("review-team");
        let default_hits = store
            .search(&scope, "sqlite storage", &TeamSearchOptions::default())
            .await
            .unwrap();
        assert!(!default_hits.iter().any(|hit| hit.record.id == "id-expired"));
        let all_hits = store
            .search(
                &scope,
                "sqlite storage",
                &TeamSearchOptions::default()
                    .include_expired()
                    .with_top_k(10)
                    .with_threshold(0.0),
            )
            .await
            .unwrap();
        assert_eq!(all_hits.len(), 3);
        let tagged_hits = store
            .search(
                &scope,
                "sqlite storage",
                &TeamSearchOptions::default().with_tags(["STORAGE"]),
            )
            .await
            .unwrap();
        assert_eq!(tagged_hits.len(), 1);
        assert_eq!(tagged_hits[0].record.id, "id-tagged");
        let task_hits = store
            .search(
                &scope,
                "sqlite storage",
                &TeamSearchOptions::default().with_kinds([TeamMemoryKind::Task]),
            )
            .await
            .unwrap();
        assert_eq!(task_hits.len(), 1);
        assert_eq!(task_hits[0].record.id, "id-task");
    }
    #[tokio::test]
    async fn search_validates_scope_and_options() {
        let store = store();
        let err = store
            .search(
                &TeamMemoryScope::default(),
                "sqlite",
                &TeamSearchOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            openjiuwen_common::error::Error::Validation(_)
        ));
        let err = store
            .search(
                &TeamMemoryScope::new("review-team"),
                "sqlite",
                &TeamSearchOptions::default().with_top_k(0),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("top_k"));
        let err = store
            .search(
                &TeamMemoryScope::new("review-team"),
                "sqlite",
                &TeamSearchOptions::default().with_threshold(f32::NAN),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("threshold"));
    }
    #[tokio::test]
    async fn clear_removes_only_matching_records() {
        let store = store();
        store
            .upsert(&record(
                "review-team",
                Some("alice"),
                "id-a",
                "content",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        store
            .upsert(&record(
                "review-team",
                Some("bob"),
                "id-b",
                "content",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        store
            .upsert(&record(
                "other-team",
                None,
                "id-c",
                "content",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        let removed = store
            .clear(&TeamMemoryFilter::for_team("review-team").member("alice"))
            .await
            .unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.len(), 2);
        assert!(store.get("review-team", "id-b").await.unwrap().is_some());
        let err = store.clear(&TeamMemoryFilter::default()).await.unwrap_err();
        assert!(matches!(
            err,
            openjiuwen_common::error::Error::Validation(_)
        ));
    }
    #[tokio::test]
    async fn stats_aggregate_the_filtered_set() {
        let store = store();
        store
            .upsert(&record(
                "review-team",
                Some("alice"),
                "id-a",
                "content",
                TeamMemoryKind::Decision,
            ))
            .await
            .unwrap();
        store
            .upsert(&record(
                "review-team",
                Some("alice"),
                "id-b",
                "content",
                TeamMemoryKind::Task,
            ))
            .await
            .unwrap();
        store
            .upsert(&record(
                "review-team",
                None,
                "id-c",
                "content",
                TeamMemoryKind::Decision,
            ))
            .await
            .unwrap();
        let stats = store
            .stats(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.by_kind.get("decision"), Some(&2));
        assert_eq!(stats.by_kind.get("task"), Some(&1));
        assert_eq!(stats.by_member.get("alice"), Some(&2));
        assert_eq!(stats.by_member.get(TEAM_WIDE_MEMBER_KEY), Some(&1));
        assert_eq!(stats.expired, 0);
        assert!(stats.oldest.is_some());
        assert!(stats.newest.is_some());
        assert!(stats.oldest <= stats.newest);
        let past = Utc::now() - chrono::Duration::seconds(1);
        store
            .set_expiry("review-team", "id-c", Some(past))
            .await
            .unwrap();
        let stats = store
            .stats(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.expired, 0);
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.include_expired = true;
        let stats = store.stats(&filter).await.unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.expired, 1);
        let empty = store
            .stats(&TeamMemoryFilter::for_team("unknown-team"))
            .await
            .unwrap();
        assert_eq!(empty, TeamMemoryStats::default());
    }
    #[tokio::test]
    async fn set_expiry_sets_and_clears_without_touching_updated_at() {
        let store = store();
        let mut record = record("review-team", None, "id-a", "content", TeamMemoryKind::Task);
        record.updated_at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        store.upsert(&record).await.unwrap();
        let expiry = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        assert!(store
            .set_expiry("review-team", "id-a", Some(expiry))
            .await
            .unwrap());
        let stored = store.get("review-team", "id-a").await.unwrap().unwrap();
        assert_eq!(stored.expires_at, Some(expiry));
        assert_eq!(stored.updated_at, record.updated_at);
        assert!(!store
            .set_expiry("other-team", "id-a", Some(expiry))
            .await
            .unwrap());
        assert!(!store
            .set_expiry("review-team", "missing", None)
            .await
            .unwrap());
        assert!(store.set_expiry("review-team", "id-a", None).await.unwrap());
        let stored = store.get("review-team", "id-a").await.unwrap().unwrap();
        assert_eq!(stored.expires_at, None);
    }
    #[tokio::test]
    async fn prune_expired_removes_only_expired_records() {
        let store = store();
        let now = Utc::now();
        let expired = record(
            "review-team",
            None,
            "id-expired",
            "content",
            TeamMemoryKind::Task,
        )
        .with_expires_at(now - chrono::Duration::seconds(10));
        let boundary = record(
            "other-team",
            None,
            "id-boundary",
            "content",
            TeamMemoryKind::Task,
        )
        .with_expires_at(now);
        let live = record(
            "review-team",
            None,
            "id-live",
            "content",
            TeamMemoryKind::Task,
        );
        for record in [expired, boundary, live] {
            store.upsert(&record).await.unwrap();
        }
        let removed = store.prune_expired(now).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.len(), 1);
        assert!(store.get("review-team", "id-live").await.unwrap().is_some());
        assert_eq!(store.prune_expired(now).await.unwrap(), 0);
    }
}
