//! Topic-pattern subscriptions for the team message bus.
//!
//! Mirrors Python's in-memory subscription manager: a bidirectional index between subscription
//! patterns (`"team__p2p__"`, `"*__p2p__"`) and agent ids. Lookups are lock-protected but never
//! held across an `.await`, since none of the operations here are asynchronous.

use std::collections::{BTreeMap, BTreeSet};

use parking_lot::RwLock;

/// Bidirectional index: pattern -> agents and agent -> patterns.
#[derive(Debug, Default)]
struct Inner {
    by_topic: BTreeMap<String, BTreeSet<String>>,
    by_agent: BTreeMap<String, BTreeSet<String>>,
}

/// In-memory registry of topic-pattern subscriptions.
///
/// A subscription is a `(agent id, pattern)` pair. Patterns may use `*` and `?` wildcards, so one
/// pattern can match many topics; [`subscribers`](SubscriptionManager::subscribers) resolves a
/// concrete topic id against every stored pattern.
#[derive(Debug, Default)]
pub struct SubscriptionManager {
    inner: RwLock<Inner>,
}

impl SubscriptionManager {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner::default()),
        }
    }

    /// Subscribes `agent_id` to `topic`.
    ///
    /// Idempotent: subscribing twice keeps exactly one pair. An empty agent id or topic is
    /// ignored.
    pub fn subscribe(&self, agent_id: &str, topic: &str) {
        if agent_id.is_empty() || topic.is_empty() {
            return;
        }
        let mut inner = self.inner.write();
        inner
            .by_topic
            .entry(topic.to_owned())
            .or_default()
            .insert(agent_id.to_owned());
        inner
            .by_agent
            .entry(agent_id.to_owned())
            .or_default()
            .insert(topic.to_owned());
    }

    /// Removes the `(agent_id, topic)` subscription, pruning empty index entries.
    ///
    /// Idempotent: unsubscribing an unknown pair is a no-op. An empty agent id or topic is
    /// ignored.
    pub fn unsubscribe(&self, agent_id: &str, topic: &str) {
        if agent_id.is_empty() || topic.is_empty() {
            return;
        }
        let mut inner = self.inner.write();
        let topic_empty = inner.by_topic.get_mut(topic).is_some_and(|agents| {
            agents.remove(agent_id);
            agents.is_empty()
        });
        if topic_empty {
            inner.by_topic.remove(topic);
        }
        let agent_empty = inner.by_agent.get_mut(agent_id).is_some_and(|topics| {
            topics.remove(topic);
            topics.is_empty()
        });
        if agent_empty {
            inner.by_agent.remove(agent_id);
        }
    }

    /// Returns the sorted, de-duplicated ids of every agent whose pattern matches `topic_id`.
    ///
    /// An exact pattern such as `"team__p2p__"` matches only itself, while `"*__p2p__"` matches
    /// every topic ending in `"__p2p__"`.
    #[must_use]
    pub fn subscribers(&self, topic_id: &str) -> Vec<String> {
        let inner = self.inner.read();
        let mut agents = BTreeSet::new();
        for (pattern, subscribed) in &inner.by_topic {
            if topic_matches(pattern, topic_id) {
                agents.extend(subscribed.iter().cloned());
            }
        }
        agents.into_iter().collect()
    }

    /// Returns the sorted patterns `agent_id` is subscribed to.
    #[must_use]
    pub fn topics_of(&self, agent_id: &str) -> Vec<String> {
        let inner = self.inner.read();
        inner
            .by_agent
            .get(agent_id)
            .map(|topics| topics.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns the sorted ids of every agent holding at least one subscription.
    #[must_use]
    pub fn agents(&self) -> Vec<String> {
        let inner = self.inner.read();
        inner.by_agent.keys().cloned().collect()
    }

    /// Returns the total number of `(agent, pattern)` pairs.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        let inner = self.inner.read();
        inner.by_agent.values().map(BTreeSet::len).sum()
    }

    /// Drops every subscription held by `agent_id`.
    pub fn remove_agent(&self, agent_id: &str) {
        let _ = self.unsubscribe_all(agent_id);
    }

    /// Drops every subscription held by `agent_id`, returning how many were removed.
    ///
    /// Python parity: `SubscriptionManager.unsubscribe_all`, which the message
    /// bus calls from `remove_all_subscriptions`. An empty or unknown agent id
    /// removes nothing and returns zero.
    pub fn unsubscribe_all(&self, agent_id: &str) -> usize {
        if agent_id.is_empty() {
            return 0;
        }
        let mut inner = self.inner.write();
        let Some(topics) = inner.by_agent.remove(agent_id) else {
            return 0;
        };
        let removed = topics.len();
        for topic in topics {
            let topic_empty = inner.by_topic.get_mut(&topic).is_some_and(|agents| {
                agents.remove(agent_id);
                agents.is_empty()
            });
            if topic_empty {
                inner.by_topic.remove(&topic);
            }
        }
        removed
    }

    /// Returns the whole registry as `pattern -> sorted agent ids`.
    ///
    /// Python parity: `SubscriptionManager.list_subscriptions()` without an
    /// agent filter. Python returns unordered sets; the agent ids are sorted
    /// here, so the shape is deterministic.
    #[must_use]
    pub fn list(&self) -> BTreeMap<String, Vec<String>> {
        let inner = self.inner.read();
        inner
            .by_topic
            .iter()
            .map(|(pattern, agents)| (pattern.clone(), agents.iter().cloned().collect()))
            .collect()
    }

    /// Returns the sorted distinct subscription patterns currently stored.
    #[must_use]
    pub fn patterns(&self) -> Vec<String> {
        let inner = self.inner.read();
        inner.by_topic.keys().cloned().collect()
    }

    /// Returns the sorted agents subscribed to exactly `pattern` (no wildcard resolution).
    #[must_use]
    pub fn agents_for_exact(&self, pattern: &str) -> Vec<String> {
        let inner = self.inner.read();
        inner
            .by_topic
            .get(pattern)
            .map(|agents| agents.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// Returns `true` when `topic` matches `pattern`.
///
/// A pattern without `*` or `?` is compared literally. Otherwise Python `fnmatch` semantics are
/// applied for those two wildcards only: `*` matches any sequence of characters (including the
/// empty one) and `?` matches exactly one character. Every other character, including `[`, `]`,
/// `.` and `-`, is literal.
///
/// The matcher is a linear two-pointer scan with a single star/backtrack mark, so it does not
/// allocate and stays fast on long topics.
#[must_use]
pub fn topic_matches(pattern: &str, topic: &str) -> bool {
    if !pattern.contains(['*', '?']) {
        return pattern == topic;
    }
    let pattern: Vec<char> = pattern.chars().collect();
    let topic: Vec<char> = topic.chars().collect();

    let mut pat_idx = 0usize;
    let mut topic_idx = 0usize;
    // Index of the most recent `*` in the pattern and the topic position it currently covers.
    let mut star: Option<usize> = None;
    let mut resume = 0usize;

    while topic_idx < topic.len() {
        match (pattern.get(pat_idx), topic.get(topic_idx)) {
            (Some(&pat_ch), Some(&topic_ch)) if pat_ch == '?' || pat_ch == topic_ch => {
                pat_idx += 1;
                topic_idx += 1;
            }
            (Some(&'*'), _) => {
                star = Some(pat_idx);
                resume = topic_idx;
                pat_idx += 1;
            }
            _ => {
                let Some(star_idx) = star else {
                    return false;
                };
                // Let the last star swallow one more character and retry.
                resume += 1;
                pat_idx = star_idx + 1;
                topic_idx = resume;
            }
        }
    }

    while pattern.get(pat_idx) == Some(&'*') {
        pat_idx += 1;
    }
    pat_idx == pattern.len()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_is_idempotent() {
        let manager = SubscriptionManager::new();
        manager.subscribe("alice", "team__p2p__");
        manager.subscribe("alice", "team__p2p__");
        assert_eq!(manager.subscription_count(), 1);
        assert_eq!(manager.patterns(), vec!["team__p2p__"]);
        assert_eq!(manager.agents_for_exact("team__p2p__"), vec!["alice"]);
    }

    #[test]
    fn unsubscribe_prunes_and_is_idempotent() {
        let manager = SubscriptionManager::new();
        manager.subscribe("alice", "team__p2p__");
        manager.subscribe("alice", "team__pubsub__");
        manager.unsubscribe("alice", "team__p2p__");
        manager.unsubscribe("alice", "team__p2p__");
        manager.unsubscribe("bob", "team__pubsub__");
        assert_eq!(manager.subscription_count(), 1);
        assert_eq!(manager.patterns(), vec!["team__pubsub__"]);
        assert_eq!(manager.topics_of("alice"), vec!["team__pubsub__"]);

        manager.unsubscribe("alice", "team__pubsub__");
        assert_eq!(manager.subscription_count(), 0);
        assert_eq!(manager.patterns(), Vec::<String>::new());
        assert_eq!(manager.agents(), Vec::<String>::new());
        assert_eq!(manager.topics_of("alice"), Vec::<String>::new());
    }

    #[test]
    fn reverse_index_is_sorted_and_deduplicated() {
        let manager = SubscriptionManager::new();
        manager.subscribe("bob", "z-topic");
        manager.subscribe("alice", "b-topic");
        manager.subscribe("alice", "a-topic");
        manager.subscribe("alice", "a-topic");

        assert_eq!(manager.agents(), vec!["alice", "bob"]);
        assert_eq!(manager.topics_of("alice"), vec!["a-topic", "b-topic"]);
        assert_eq!(manager.patterns(), vec!["a-topic", "b-topic", "z-topic"]);
        assert_eq!(manager.subscription_count(), 3);
        assert_eq!(manager.topics_of("carol"), Vec::<String>::new());
        assert_eq!(manager.agents_for_exact("missing"), Vec::<String>::new());
    }

    #[test]
    fn subscribers_matches_exact_and_wildcard_patterns_sorted() {
        let manager = SubscriptionManager::new();
        manager.subscribe("zoe", "team__p2p__");
        manager.subscribe("alice", "team__p2p__");
        manager.subscribe("bob", "*__p2p__");
        manager.subscribe("cara", "other__p2p__");
        manager.subscribe("dave", "*__pubsub__");

        assert_eq!(
            manager.subscribers("team__p2p__"),
            vec!["alice", "bob", "zoe"]
        );
        assert_eq!(manager.subscribers("other__p2p__"), vec!["bob", "cara"]);
        assert_eq!(manager.subscribers("team__pubsub__"), vec!["dave"]);
        assert_eq!(manager.subscribers("unmatched"), Vec::<String>::new());
    }

    #[test]
    fn subscribers_deduplicates_across_patterns() {
        let manager = SubscriptionManager::new();
        manager.subscribe("alice", "team__p2p__");
        manager.subscribe("alice", "*__p2p__");
        manager.subscribe("alice", "team*");
        assert_eq!(manager.subscribers("team__p2p__"), vec!["alice"]);
    }

    #[test]
    fn remove_agent_drops_all_patterns() {
        let manager = SubscriptionManager::new();
        manager.subscribe("alice", "a");
        manager.subscribe("alice", "b");
        manager.subscribe("bob", "b");
        manager.remove_agent("alice");
        manager.remove_agent("alice");
        manager.remove_agent("");

        assert_eq!(manager.agents(), vec!["bob"]);
        assert_eq!(manager.topics_of("alice"), Vec::<String>::new());
        assert_eq!(manager.patterns(), vec!["b"]);
        assert_eq!(manager.subscribers("b"), vec!["bob"]);
        assert_eq!(manager.subscription_count(), 1);
    }

    #[test]
    fn empty_arguments_are_ignored() {
        let manager = SubscriptionManager::new();
        manager.subscribe("", "topic");
        manager.subscribe("alice", "");
        manager.subscribe("", "");
        assert_eq!(manager.subscription_count(), 0);
        assert_eq!(manager.agents(), Vec::<String>::new());
        assert_eq!(manager.patterns(), Vec::<String>::new());

        manager.subscribe("alice", "topic");
        manager.unsubscribe("", "topic");
        manager.unsubscribe("alice", "");
        assert_eq!(manager.subscription_count(), 1);
    }

    #[test]
    fn unsubscribe_all_counts_and_prunes() {
        let manager = SubscriptionManager::new();
        manager.subscribe("alice", "a");
        manager.subscribe("alice", "b");
        manager.subscribe("bob", "b");
        manager.subscribe("bob", "c");

        assert_eq!(manager.unsubscribe_all("alice"), 2);
        assert_eq!(manager.unsubscribe_all("alice"), 0);
        assert_eq!(manager.unsubscribe_all(""), 0);
        assert_eq!(manager.subscription_count(), 2);
        assert_eq!(manager.patterns(), vec!["b", "c"]);
        assert_eq!(manager.subscribers("b"), vec!["bob"]);
        assert_eq!(manager.topics_of("alice"), Vec::<String>::new());
    }

    #[test]
    fn list_returns_sorted_agents_per_pattern() {
        let manager = SubscriptionManager::new();
        assert_eq!(manager.list(), BTreeMap::new());
        manager.subscribe("zoe", "news");
        manager.subscribe("alice", "news");
        manager.subscribe("alice", "alerts");

        assert_eq!(
            manager.list(),
            BTreeMap::from([
                ("alerts".to_owned(), vec!["alice".to_owned()]),
                (
                    "news".to_owned(),
                    vec!["alice".to_owned(), "zoe".to_owned()]
                ),
            ])
        );
        assert_eq!(
            manager.subscription_count(),
            manager.list().values().map(Vec::len).sum::<usize>()
        );
    }

    #[test]
    fn wildcard_matching_table() {
        // Exact patterns do not wildcard-match.
        assert!(topic_matches("a.b", "a.b"));
        assert!(!topic_matches("a.b", "axb"));
        assert!(topic_matches("a", "a"));
        assert!(!topic_matches("a", "ab"));
        assert!(!topic_matches("[ab]", "a"));

        // `*` matches the empty sequence and any sequence.
        assert!(topic_matches("*", ""));
        assert!(topic_matches("*", "anything"));
        assert!(topic_matches("a*", "a"));
        assert!(topic_matches("a*", "abc"));
        assert!(!topic_matches("a*", "ba"));
        assert!(topic_matches("*b", "b"));
        assert!(topic_matches("*b", "aab"));
        assert!(!topic_matches("*b", "ba"));
        assert!(topic_matches("***", ""));
        assert!(topic_matches("***", "xyz"));

        // `?` matches exactly one character and never the empty string.
        assert!(topic_matches("a?c", "abc"));
        assert!(!topic_matches("a?c", "ac"));
        assert!(!topic_matches("a?c", "abbc"));
        assert!(topic_matches("?", "x"));
        assert!(!topic_matches("?", ""));
        assert!(topic_matches("a?c", "a.c"));
        assert!(topic_matches("a?c", "a[c"));
        assert!(topic_matches("[ab]", "[ab]"));

        // Combined wildcards with backtracking.
        assert!(topic_matches("a*b?c", "axxxbbc"));
        assert!(topic_matches("a*b?c", "abxc"));
        assert!(!topic_matches("a*b?c", "abc"));
        assert!(topic_matches("*__p2p__", "team__p2p__"));
        assert!(!topic_matches("*__p2p__", "team__pubsub__"));
        assert!(topic_matches("team_?__p2p__", "team_1__p2p__"));
        assert!(topic_matches("*a*b", "xxaxxb"));
        assert!(!topic_matches("*a*b", "xxaxxc"));
        assert!(topic_matches("a*b*c*", "abc"));
        assert!(topic_matches("a*b*c*", "azbyc"));
        assert!(topic_matches("a*b*c*", "abc-tail"));
    }

    #[test]
    fn wildcard_matching_handles_long_topics() {
        let long = "a".repeat(20_000);
        assert!(topic_matches("*", &long));
        assert!(topic_matches("a*", &long));
        assert!(topic_matches("*a", &long));
        assert!(!topic_matches("*b", &long));
        assert!(topic_matches(&format!("{long}*"), &format!("{long}T")));

        let alternating = "ab".repeat(5_000);
        assert!(topic_matches("a*b", &alternating));
        assert!(!topic_matches("a*c", &alternating));
    }

    #[test]
    fn unicode_topics_match_by_character() {
        assert!(topic_matches("é?", "éà"));
        assert!(topic_matches("*é", "café"));
        assert!(!topic_matches("é?", "é"));
    }
}
