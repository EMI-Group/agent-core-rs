//! The in-process pool of active team runtimes.
//!
//! Python parity source: `openjiuwen/agent_teams/runtime/pool.py` (`TeamRuntimePool`,
//! `ActiveTeam`).
//!
//! One pool belongs to one [`TeamRuntimeManager`](crate::runtime::manager::TeamRuntimeManager) and
//! answers the single question the run entry points ask: *which team, if any, is live right now and
//! in which session?* The Python pool is keyed by `team_name` only and stores an `ActiveTeam` record
//! whose `current_session_id` is a field; this port keys the pool by session first and team second
//! (`session_id -> team_name -> handle`), so the session binding is the key and can never disagree
//! with the stored record. That also makes the two questions the manager needs — "every team of a
//! session" and "the team under this name, with its session" — one map lookup each.
//!
//! ```
//! use openjiuwen_agent_teams::runtime::pool::TeamRuntimePool;
//!
//! let pool = TeamRuntimePool::new();
//! assert!(pool.is_empty());
//! assert_eq!(pool.len(), 0);
//! assert!(pool.snapshot().is_empty());
//! ```
//!
//! The pool is a pure registry: it starts, stops and inspects nothing. Locking is limited to one
//! map operation per call ([`parking_lot::Mutex`]) and no lock is ever held across an `.await`,
//! because every method here is synchronous — the manager awaits the team it took out of the pool.
use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::team_agent::TeamAgent;

/// The set of live team runtimes, grouped by the session that activated them.
///
/// Both maps are [`BTreeMap`]s so every listing (`snapshot`, `teams_of_session`, `find`) is
/// deterministic: sessions sort by session id and teams sort by team name, matching the ordering
/// rule the crate pins for snapshots.
#[derive(Debug, Default)]
pub struct TeamRuntimePool {
    /// `session_id -> team_name -> live handle`.
    teams: Mutex<BTreeMap<String, BTreeMap<String, Arc<TeamAgent>>>>,
}

impl TeamRuntimePool {
    /// Builds an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `team` under `session_id`, keyed by its own team name.
    ///
    /// Returns the handle previously registered under the same `(session_id, team_name)` pair, so a
    /// caller can see whether it replaced a live team (the manager does not: it removes the stale
    /// entry through [`Self::remove`] first).
    pub fn insert(&self, session_id: &str, team: Arc<TeamAgent>) -> Option<Arc<TeamAgent>> {
        let team_name = team.team_name().to_owned();
        self.teams
            .lock()
            .entry(session_id.to_owned())
            .or_default()
            .insert(team_name, team)
    }

    /// Returns the live team registered as `team_name` inside `session_id`.
    #[must_use]
    pub fn get(&self, session_id: &str, team_name: &str) -> Option<Arc<TeamAgent>> {
        self.teams
            .lock()
            .get(session_id)
            .and_then(|session| session.get(team_name))
            .cloned()
    }

    /// Returns a copy of the `team_name -> handle` map of `session_id`.
    ///
    /// Copies the handles, not the teams: the returned map is a detached view, so a caller may
    /// iterate it while other tasks add or remove teams.
    #[must_use]
    pub fn get_for_session(&self, session_id: &str) -> BTreeMap<String, Arc<TeamAgent>> {
        self.teams
            .lock()
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Removes the entry of `(session_id, team_name)`, pruning the session when it becomes empty.
    ///
    /// Returns the removed handle, or `None` when the pair was not registered.
    pub fn remove(&self, session_id: &str, team_name: &str) -> Option<Arc<TeamAgent>> {
        let mut teams = self.teams.lock();
        let session = teams.get_mut(session_id)?;
        let removed = session.remove(team_name);
        if session.is_empty() {
            teams.remove(session_id);
        }
        removed
    }

    /// Removes every team of `session_id` and returns them in team-name order.
    ///
    /// The session key is dropped whether or not it held anything, so an emptied pool never keeps a
    /// stale session behind.
    pub fn remove_session(&self, session_id: &str) -> Vec<Arc<TeamAgent>> {
        self.teams
            .lock()
            .remove(session_id)
            .map(|session| session.into_values().collect())
            .unwrap_or_default()
    }

    /// Returns the session id `team_name` is live in plus its handle.
    ///
    /// Sessions are visited in session-id order, so a name registered under several sessions always
    /// reports the same one. A name is normally live in exactly one session — the manager keys
    /// every team by the session that activated it and only a persistent team may change sessions,
    /// which re-registers nothing.
    #[must_use]
    pub fn find(&self, team_name: &str) -> Option<(String, Arc<TeamAgent>)> {
        let teams = self.teams.lock();
        teams.iter().find_map(|(session_id, session)| {
            session
                .get(team_name)
                .map(|team| (session_id.clone(), Arc::clone(team)))
        })
    }

    /// Returns the number of live teams across every session.
    #[must_use]
    pub fn len(&self) -> usize {
        self.teams.lock().values().map(BTreeMap::len).sum()
    }

    /// Returns `true` when no team is live in any session.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.teams.lock().values().all(BTreeMap::is_empty)
    }

    /// Returns `session_id -> sorted team names` for every session holding at least one team.
    #[must_use]
    pub fn snapshot(&self) -> BTreeMap<String, Vec<String>> {
        self.teams
            .lock()
            .iter()
            .filter(|(_, session)| !session.is_empty())
            .map(|(session_id, session)| {
                (
                    session_id.clone(),
                    session.keys().cloned().collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    /// Returns the names of the teams live in `session_id`, sorted.
    #[must_use]
    pub fn teams_of_session(&self, session_id: &str) -> Vec<String> {
        self.teams
            .lock()
            .get(session_id)
            .map(|session| session.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::runtime::team_agent::test_support::{
        blueprint, built, RecordingWorkspaceProvider, ScriptedFactory,
    };
    use crate::schema::TeamAgentSpec;

    /// Builds an unstarted leader handle for `team`; the pool never touches the runtime.
    async fn agent(team: &str) -> Arc<TeamAgent> {
        let spec: TeamAgentSpec = blueprint(team, "boss", &["dev-1"]);
        let (agent, _factory, _workspace): (
            Arc<TeamAgent>,
            Arc<ScriptedFactory>,
            Arc<RecordingWorkspaceProvider>,
        ) = built(&spec).await;
        agent
    }

    #[tokio::test]
    async fn insert_and_get_address_a_team_by_session() {
        let pool = TeamRuntimePool::new();
        let alpha = agent("alpha").await;

        assert!(pool.insert("session-a", Arc::clone(&alpha)).is_none());
        assert_eq!(pool.len(), 1);
        let found = pool.get("session-a", "alpha").unwrap();
        assert!(Arc::ptr_eq(&found, &alpha));
        assert!(pool.get("session-a", "beta").is_none());
        assert!(pool.get("session-b", "alpha").is_none());
    }

    #[tokio::test]
    async fn insert_replaces_the_previous_handle() {
        let pool = TeamRuntimePool::new();
        let first = agent("alpha").await;
        let second = agent("alpha").await;

        assert!(pool.insert("session-a", Arc::clone(&first)).is_none());
        let replaced = pool.insert("session-a", Arc::clone(&second)).unwrap();
        assert!(Arc::ptr_eq(&replaced, &first));
        assert!(Arc::ptr_eq(
            &pool.get("session-a", "alpha").unwrap(),
            &second
        ));
        assert_eq!(pool.len(), 1);
    }

    #[tokio::test]
    async fn sessions_are_isolated() {
        let pool = TeamRuntimePool::new();
        let alpha = agent("alpha").await;
        let beta = agent("beta").await;

        pool.insert("session-a", Arc::clone(&alpha));
        pool.insert("session-b", Arc::clone(&beta));

        assert_eq!(pool.teams_of_session("session-a"), vec!["alpha".to_owned()]);
        assert_eq!(pool.teams_of_session("session-b"), vec!["beta".to_owned()]);
        assert_eq!(pool.teams_of_session("session-c"), Vec::<String>::new());

        let session_a = pool.get_for_session("session-a");
        assert_eq!(session_a.keys().cloned().collect::<Vec<_>>(), vec!["alpha"]);
        assert!(Arc::ptr_eq(session_a.get("alpha").unwrap(), &alpha));
        assert!(pool.get_for_session("session-c").is_empty());
    }

    #[tokio::test]
    async fn remove_prunes_the_emptied_session() {
        let pool = TeamRuntimePool::new();
        let alpha = agent("alpha").await;
        pool.insert("session-a", Arc::clone(&alpha));

        let removed = pool.remove("session-a", "alpha").unwrap();
        assert!(Arc::ptr_eq(&removed, &alpha));
        assert!(pool.is_empty());
        assert!(pool.snapshot().is_empty(), "the session key is pruned");
        assert!(pool.remove("session-a", "alpha").is_none());
        assert!(pool.remove("session-b", "alpha").is_none());
    }

    #[tokio::test]
    async fn remove_keeps_the_session_while_other_teams_remain() {
        let pool = TeamRuntimePool::new();
        let alpha = agent("alpha").await;
        let beta = agent("beta").await;
        pool.insert("session-a", Arc::clone(&alpha));
        pool.insert("session-a", Arc::clone(&beta));

        assert!(pool.remove("session-a", "alpha").is_some());
        assert_eq!(
            pool.snapshot().get("session-a").map(Vec::as_slice),
            Some(["beta".to_owned()].as_slice())
        );
        assert_eq!(pool.len(), 1);
    }

    #[tokio::test]
    async fn remove_session_returns_every_team_in_name_order() {
        let pool = TeamRuntimePool::new();
        for name in ["gamma", "alpha", "beta"] {
            pool.insert("session-a", agent(name).await);
        }
        pool.insert("session-b", agent("delta").await);

        let removed = pool.remove_session("session-a");
        let names: Vec<&str> = removed
            .iter()
            .map(|team| team.team_name())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.teams_of_session("session-a"), Vec::<String>::new());
        assert_eq!(pool.teams_of_session("session-b"), vec!["delta".to_owned()]);
        assert!(pool.remove_session("session-a").is_empty());
        assert!(pool.remove_session("session-unknown").is_empty());
    }

    #[tokio::test]
    async fn find_reports_the_session_of_a_team() {
        let pool = TeamRuntimePool::new();
        let alpha = agent("alpha").await;
        pool.insert("session-a", Arc::clone(&alpha));

        let (session_id, found) = pool.find("alpha").unwrap();
        assert_eq!(session_id, "session-a");
        assert!(Arc::ptr_eq(&found, &alpha));
        assert!(pool.find("beta").is_none());
    }

    #[tokio::test]
    async fn find_visits_sessions_in_id_order() {
        let pool = TeamRuntimePool::new();
        let first = agent("alpha").await;
        let second = agent("alpha").await;
        // Same name, two sessions: the lowest session id wins, deterministically.
        pool.insert("session-b", second);
        pool.insert("session-a", Arc::clone(&first));

        let (session_id, found) = pool.find("alpha").unwrap();
        assert_eq!(session_id, "session-a");
        assert!(Arc::ptr_eq(&found, &first));
    }

    #[tokio::test]
    async fn len_counts_teams_across_sessions() {
        let pool = TeamRuntimePool::new();
        assert!(pool.is_empty());

        pool.insert("session-a", agent("alpha").await);
        pool.insert("session-a", agent("beta").await);
        pool.insert("session-b", agent("gamma").await);
        assert_eq!(pool.len(), 3);
        assert!(!pool.is_empty());

        pool.remove_session("session-a");
        assert_eq!(pool.len(), 1);
    }

    #[tokio::test]
    async fn snapshot_lists_sessions_in_order_with_sorted_team_names() {
        let pool = TeamRuntimePool::new();
        for name in ["gamma", "alpha"] {
            pool.insert("session-b", agent(name).await);
        }
        pool.insert("session-a", agent("beta").await);

        let snapshot = pool.snapshot();
        let sessions: Vec<&str> = snapshot.keys().map(String::as_str).collect();
        assert_eq!(sessions, vec!["session-a", "session-b"]);
        assert_eq!(
            snapshot.get("session-b").unwrap(),
            &vec!["alpha".to_owned(), "gamma".to_owned()]
        );
        assert_eq!(
            pool.teams_of_session("session-b"),
            vec!["alpha".to_owned(), "gamma".to_owned()]
        );
    }
}
