//! Card-driven auto-subscription for the team message bus.
//!
//! Python parity: `EventDrivenTeamCard.subscriptions` is documented as a
//! "mapping of `agent_id` to list of subscribed topics" but the Python runtime
//! leaves the wiring to the caller. This module derives the subscription plan
//! from a [`TeamCard`] plus that optional override map, and applies it to the
//! subscription registry of a [`TeamRuntime`]'s transport (the in-process
//! [`crate::MessageBus`] by default).
//!
//! # Capability carrier
//!
//! Python's `AgentCard` carries free-form tags/capabilities; the Rust
//! [`AgentCard`](openjiuwen_agent::AgentCard) does not, so capability topics are
//! carried by [`TeamCard::tags`] and by the explicit
//! [`EventDrivenTeamCard::subscriptions`] map. A tag is only consulted when it
//! starts with [`SUBSCRIBE_TAG_PREFIX`] (`subscribe:`); every other tag is
//! ignored, so existing discovery tags keep working.
//!
//! # Derivation rules
//!
//! 1. The declared members are [`TeamCard::agent_cards`], identified by the
//!    `AgentCard::id` field. With no members the plan stays empty unless
//!    overrides name agents explicitly.
//! 2. Broadcast: when [`TeamCard::topic`] is non-empty, every member subscribes
//!    to `topic` and to `topic.*`.
//! 3. Directed: when [`TeamCard::topic`] and the member's `AgentCard::name` are
//!    non-empty, that member also subscribes to `topic.{name}`.
//! 4. `subscribe:<pattern>` subscribes every member to `<pattern>`;
//!    `subscribe:<agent_id>:<pattern>` subscribes only that agent, and only when
//!    it is a declared member. The split happens at the first `:` after the
//!    prefix, so a pattern may itself contain `:`. Malformed directives
//!    (`subscribe:`, `subscribe::x`, `subscribe:a:`) are ignored, never an
//!    error.
//! 5. Overrides ([`EventDrivenTeamCard::subscriptions`] or
//!    [`derive_with_overrides`]) replace the derived set of that agent id
//!    verbatim (sorted and de-duplicated, but an empty list is kept so
//!    [`AutoSubscriptionMode::Replace`] can clear an agent). Override ids that
//!    are not members are kept.
//!
//! Derived entries with no patterns are omitted, so [`AutoSubscriptionPlan::is_empty`]
//! means "no subscriptions", and every stored pattern list is sorted with
//! `sort_unstable` + `dedup` so iteration order is deterministic.
//!
//! # Applying a plan
//!
//! [`apply`] and [`apply_agent`] are synchronous because the transport
//! subscription registry is synchronous; each call takes short registry locks
//! and never awaits, so there is no deadlock path. [`AutoSubscriptionMode::Merge`]
//! counts only the patterns that were not subscribed before the call,
//! [`AutoSubscriptionMode::Replace`] first removes the agent's stale patterns
//! and then counts added plus removed patterns, and
//! [`AutoSubscriptionMode::Disabled`] never touches the registry and returns
//! `0`. Empty patterns are dropped on application, matching the bus, which
//! ignores empty agent ids and topics. A write a transport rejects is logged and
//! skipped ([`TeamRuntime::subscribe`] is the fallible entry point).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::card::{EventDrivenTeamCard, TeamCard};
use crate::runtime::TeamRuntime;
use crate::transport::TeamTransport;

/// Tag prefix that turns a [`TeamCard`] tag into a subscription directive.
pub const SUBSCRIBE_TAG_PREFIX: &str = "subscribe:";

/// How [`apply`] and [`apply_agent`] reconcile a plan with the bus registry.
///
/// Serialized as `"merge"`, `"replace"` or `"disabled"`; an unknown variant is
/// a deserialization error (an unknown mode must never silently change routing).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AutoSubscriptionMode {
    /// Add the planned patterns and keep everything the agents already hold.
    #[default]
    Merge,
    /// Make every planned agent's pattern set exactly its planned set.
    Replace,
    /// Leave the bus untouched.
    Disabled,
}

/// The subscription plan derived from a team card.
///
/// Every pattern list is sorted and de-duplicated, so iteration, application and
/// serialization are deterministic. The JSON shape is
/// `{"patterns": {"<agent id>": ["<pattern>", ...]}}`; an empty object
/// deserializes to an empty plan and unknown fields are tolerated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct AutoSubscriptionPlan {
    /// Subscription patterns keyed by agent id.
    patterns: BTreeMap<String, Vec<String>>,
}

impl AutoSubscriptionPlan {
    /// Creates an empty plan.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            patterns: BTreeMap::new(),
        }
    }

    /// Returns `true` when no agent has a pattern.
    ///
    /// An agent kept with an empty pattern list (an explicit override) still
    /// makes the plan non-empty, because it tells [`apply`] to clear that agent.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Returns the number of agents named by the plan, empty lists included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Returns the sorted patterns planned for `agent_id`, if it is in the plan.
    #[must_use]
    pub fn get(&self, agent_id: &str) -> Option<&[String]> {
        self.patterns.get(agent_id).map(Vec::as_slice)
    }

    /// Sets the patterns of `agent_id`, sorted and de-duplicated.
    ///
    /// An empty iterator keeps an entry with no patterns, which
    /// [`AutoSubscriptionMode::Replace`] reads as "unsubscribe everything".
    pub fn insert(
        &mut self,
        agent_id: impl Into<String>,
        patterns: impl IntoIterator<Item = String>,
    ) {
        self.patterns.insert(agent_id.into(), normalize(patterns));
    }

    /// Returns a copy of the patterns already planned for `agent_id`.
    ///
    /// Empty when the agent is not in the plan.
    #[must_use]
    pub fn entry(&mut self, agent_id: &str) -> Vec<String> {
        self.patterns.get(agent_id).cloned().unwrap_or_default()
    }

    /// Returns the planned agent ids, sorted.
    #[must_use]
    pub fn agents(&self) -> Vec<String> {
        self.patterns.keys().cloned().collect()
    }

    /// Returns the whole plan as a map from agent id to sorted patterns.
    #[must_use]
    pub fn patterns(&self) -> &BTreeMap<String, Vec<String>> {
        &self.patterns
    }

    /// Union-merges `other` into this plan, keeping both pattern sets.
    pub fn merge(&mut self, other: &AutoSubscriptionPlan) {
        for (agent_id, patterns) in &other.patterns {
            let slot = self.patterns.entry(agent_id.clone()).or_default();
            slot.extend(patterns.iter().cloned());
            let merged = normalize(slot.iter().cloned());
            *slot = merged;
        }
    }

    /// Returns the total number of `(agent, pattern)` pairs.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        self.patterns.values().map(Vec::len).sum()
    }
}

/// Derives the subscription plan of `card` without overrides.
#[must_use]
pub fn derive(card: &TeamCard) -> AutoSubscriptionPlan {
    derive_with_overrides(card, &BTreeMap::new())
}

/// Derives the subscription plan of `card`, then replaces the derived patterns
/// of every agent named in `overrides`.
///
/// The override values are used verbatim (only sorted and de-duplicated), and
/// their agent ids are kept even when they are not members of `card`.
#[must_use]
pub fn derive_with_overrides(
    card: &TeamCard,
    overrides: &BTreeMap<String, Vec<String>>,
) -> AutoSubscriptionPlan {
    let mut broadcast: Vec<String> = Vec::new();
    if !card.topic.is_empty() {
        broadcast.push(card.topic.clone());
        broadcast.push(format!("{}.{}", card.topic, "*"));
    }

    let mut directed: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for tag in &card.tags {
        let Some(directive) = tag.strip_prefix(SUBSCRIBE_TAG_PREFIX) else {
            continue;
        };
        if directive.is_empty() {
            continue;
        }
        match directive.split_once(':') {
            None => broadcast.push(directive.to_owned()),
            Some((agent_id, pattern)) => {
                if agent_id.is_empty() || pattern.is_empty() {
                    continue;
                }
                directed
                    .entry(agent_id.to_owned())
                    .or_default()
                    .push(pattern.to_owned());
            }
        }
    }
    let broadcast = normalize(broadcast);

    let mut plan = AutoSubscriptionPlan::new();
    for member in &card.agent_cards {
        let mut patterns = broadcast.clone();
        if !card.topic.is_empty() && !member.name.is_empty() {
            patterns.push(format!("{}.{}", card.topic, member.name));
        }
        if let Some(extra) = directed.get(&member.id) {
            patterns.extend(extra.iter().cloned());
        }
        let patterns = normalize(patterns);
        if patterns.is_empty() {
            continue;
        }
        let slot = plan.patterns.entry(member.id.clone()).or_default();
        slot.extend(patterns);
        let merged = normalize(slot.iter().cloned());
        *slot = merged;
    }

    for (agent_id, patterns) in overrides {
        plan.patterns
            .insert(agent_id.clone(), normalize(patterns.iter().cloned()));
    }
    plan
}

/// Derives the plan of an [`EventDrivenTeamCard`], equal to
/// [`derive_with_overrides`] with the card's `subscriptions` as overrides.
#[must_use]
pub fn derive_from_event_card(card: &EventDrivenTeamCard) -> AutoSubscriptionPlan {
    derive_with_overrides(&card.card, &card.subscriptions)
}

/// Applies `plan` to the transport registry of `runtime` and returns the number
/// of changes.
///
/// Only the agents named in the plan are touched: with
/// [`AutoSubscriptionMode::Replace`] the stale patterns of those agents are
/// removed and their planned ones added, while agents absent from the plan keep
/// their current subscriptions. See the module docs for the exact counting rule.
#[must_use]
pub fn apply(
    runtime: &Arc<TeamRuntime>,
    plan: &AutoSubscriptionPlan,
    mode: AutoSubscriptionMode,
) -> usize {
    if mode == AutoSubscriptionMode::Disabled || plan.is_empty() {
        return 0;
    }
    let transport = runtime.transport();
    let mut changes = 0usize;
    for (agent_id, patterns) in plan.patterns() {
        changes += reconcile(transport.as_ref(), agent_id, patterns, mode);
    }
    changes
}

/// Applies the part of `plan` that belongs to `agent_id` and returns the number
/// of changes.
///
/// With [`AutoSubscriptionMode::Replace`], an agent absent from the plan is
/// cleared (its planned set is empty). An empty `agent_id` is a no-op.
#[must_use]
pub fn apply_agent(
    runtime: &Arc<TeamRuntime>,
    plan: &AutoSubscriptionPlan,
    agent_id: &str,
    mode: AutoSubscriptionMode,
) -> usize {
    if agent_id.is_empty() || mode == AutoSubscriptionMode::Disabled {
        return 0;
    }
    let desired = plan.get(agent_id).unwrap_or_default();
    reconcile(runtime.transport().as_ref(), agent_id, desired, mode)
}

/// Returns the patterns derivation would give `agent_id`, ignoring overrides.
///
/// Empty when the agent would derive nothing (it is not a member, or nothing
/// applies to it). The result is sorted and de-duplicated.
#[must_use]
pub fn derived_for_member(card: &TeamCard, agent_id: &str) -> Vec<String> {
    derive(card)
        .patterns()
        .get(agent_id)
        .cloned()
        .unwrap_or_default()
}

/// Sorts and de-duplicates `patterns`.
fn normalize(patterns: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut patterns: Vec<String> = patterns.into_iter().collect();
    patterns.sort_unstable();
    patterns.dedup();
    patterns
}

/// Reconciles one agent's transport subscriptions with `desired`, returning the
/// number of added plus removed patterns.
///
/// A transport that rejects a write (one without a subscription registry) is
/// logged and skipped, so applying a plan stays best-effort for a backend that
/// cannot honour it; [`TeamRuntime::subscribe`] surfaces the typed error.
///
/// [`TeamRuntime::subscribe`]: crate::TeamRuntime::subscribe
fn reconcile(
    transport: &dyn TeamTransport,
    agent_id: &str,
    desired: &[String],
    mode: AutoSubscriptionMode,
) -> usize {
    if agent_id.is_empty() || mode == AutoSubscriptionMode::Disabled {
        return 0;
    }
    let desired: BTreeSet<&str> = desired
        .iter()
        .map(String::as_str)
        .filter(|pattern| !pattern.is_empty())
        .collect();
    let existing: BTreeSet<String> = transport
        .agent_subscriptions(agent_id)
        .into_iter()
        .collect();

    let mut changes = 0usize;
    if mode == AutoSubscriptionMode::Replace {
        for stale in &existing {
            if !desired.contains(stale.as_str()) {
                if let Err(error) = transport.remove_subscription(agent_id, stale) {
                    tracing::warn!(agent_id, pattern = %stale, %error,
                        "auto-subscription could not drop a stale pattern");
                }
                changes += 1;
            }
        }
    }
    for pattern in desired {
        if !existing.contains(pattern) {
            changes += 1;
        }
        if let Err(error) = transport.add_subscription(agent_id, pattern) {
            tracing::warn!(agent_id, pattern, %error,
                "auto-subscription could not add a planned pattern");
        }
    }
    changes
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_agent::AgentCard;
    use serde_json::json;

    fn card_with(topic: &str, tags: &[&str], members: &[&str]) -> TeamCard {
        TeamCard::new("team")
            .with_id("team")
            .with_topic(topic)
            .with_tags(tags.iter().map(|tag| (*tag).to_owned()).collect())
            .with_agent_cards(
                members
                    .iter()
                    .map(|name| AgentCard::new(*name).with_id(*name))
                    .collect(),
            )
    }

    #[test]
    fn default_mode_is_merge_and_serde_is_snake_case() {
        assert_eq!(AutoSubscriptionMode::default(), AutoSubscriptionMode::Merge);
        assert_eq!(
            serde_json::to_value(AutoSubscriptionMode::Merge).unwrap(),
            json!("merge")
        );
        assert_eq!(
            serde_json::to_value(AutoSubscriptionMode::Replace).unwrap(),
            json!("replace")
        );
        assert_eq!(
            serde_json::to_value(AutoSubscriptionMode::Disabled).unwrap(),
            json!("disabled")
        );
        assert_eq!(
            serde_json::from_value::<AutoSubscriptionMode>(json!("replace")).unwrap(),
            AutoSubscriptionMode::Replace
        );
        assert!(serde_json::from_value::<AutoSubscriptionMode>(json!("sometimes")).is_err());
    }

    #[test]
    fn plan_insert_sorts_and_dedups() {
        let mut plan = AutoSubscriptionPlan::new();
        plan.insert("b", vec!["z".to_owned(), "a".to_owned(), "z".to_owned()]);
        plan.insert("a", Vec::new());
        assert_eq!(plan.get("b"), Some(&["a".to_owned(), "z".to_owned()][..]));
        assert_eq!(plan.get("a"), Some(&[][..]));
        assert_eq!(plan.get("missing"), None);
        assert!(!plan.is_empty());
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.agents(), vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(plan.subscription_count(), 2);
        assert_eq!(plan.entry("b"), vec!["a".to_owned(), "z".to_owned()]);
        assert_eq!(plan.entry("missing"), Vec::<String>::new());
        assert!(AutoSubscriptionPlan::new().is_empty());
    }

    #[test]
    fn plan_merge_is_a_deduplicated_union() {
        let mut left = AutoSubscriptionPlan::new();
        left.insert("a", vec!["x".to_owned()]);
        let mut right = AutoSubscriptionPlan::new();
        right.insert("a", vec!["x".to_owned(), "y".to_owned()]);
        right.insert("b", vec!["z".to_owned()]);
        left.merge(&right);
        assert_eq!(left.get("a"), Some(&["x".to_owned(), "y".to_owned()][..]));
        assert_eq!(left.get("b"), Some(&["z".to_owned()][..]));
        assert_eq!(left.subscription_count(), 3);
    }

    #[test]
    fn plan_serde_round_trip_and_unknown_fields() {
        let mut plan = AutoSubscriptionPlan::new();
        plan.insert("worker", vec!["team".to_owned(), "team.*".to_owned()]);
        let value = serde_json::to_value(&plan).unwrap();
        assert_eq!(value["patterns"]["worker"], json!(["team", "team.*"]));
        let restored: AutoSubscriptionPlan = serde_json::from_value(value).unwrap();
        assert_eq!(restored, plan);

        let empty: AutoSubscriptionPlan = serde_json::from_value(json!({})).unwrap();
        assert!(empty.is_empty());
        let tolerant: AutoSubscriptionPlan =
            serde_json::from_value(json!({"patterns": {"a": ["t"]}, "extra": 1})).unwrap();
        assert_eq!(tolerant.get("a"), Some(&["t".to_owned()][..]));
    }

    #[test]
    fn empty_card_derives_nothing() {
        assert!(derive(&TeamCard::default()).is_empty());
        assert!(derive(&card_with("", &[], &[])).is_empty());
    }

    #[test]
    fn members_without_topic_or_tags_derive_nothing() {
        assert!(derive(&card_with("", &[], &["a", "b"])).is_empty());
    }

    #[test]
    fn team_topic_broadcasts_to_every_member() {
        let plan = derive(&card_with("team", &[], &["a", "b"]));
        let expected = ["a".to_owned(), "b".to_owned()];
        assert_eq!(
            plan.get("a"),
            Some(&["team".to_owned(), "team.*".to_owned(), "team.a".to_owned()][..])
        );
        assert_eq!(plan.agents(), expected);
        assert_eq!(plan.subscription_count(), 6);
    }

    #[test]
    fn directed_topic_uses_the_member_name() {
        let card = TeamCard::new("team").with_topic("t").with_agent_cards(vec![
            AgentCard::new("alice").with_id("a1"),
            AgentCard::new("").with_id("a2"),
        ]);
        let plan = derive(&card);
        assert_eq!(
            plan.get("a1"),
            Some(&["t".to_owned(), "t.*".to_owned(), "t.alice".to_owned()][..])
        );
        // No name means no directed topic, broadcast patterns only.
        assert_eq!(
            plan.get("a2"),
            Some(&["t".to_owned(), "t.*".to_owned()][..])
        );
    }

    #[test]
    fn tags_are_the_capability_carrier() {
        let card = card_with(
            "",
            &[
                "discovery",
                "subscribe:alerts",
                "subscribe:a:alerts.a",
                "subscribe:b:alerts.b",
            ],
            &["a", "b", "c"],
        );
        let plan = derive(&card);
        assert_eq!(
            plan.get("a"),
            Some(&["alerts".to_owned(), "alerts.a".to_owned()][..])
        );
        assert_eq!(
            plan.get("b"),
            Some(&["alerts".to_owned(), "alerts.b".to_owned()][..])
        );
        assert_eq!(plan.get("c"), Some(&["alerts".to_owned()][..]));
    }

    #[test]
    fn malformed_subscribe_tags_are_ignored() {
        let card = card_with(
            "",
            &[
                "subscribe:",
                "subscribe::x",
                "subscribe:a:",
                "subscribe::",
                "other",
            ],
            &["a"],
        );
        assert!(derive(&card).is_empty());
    }

    #[test]
    fn directed_tag_for_a_non_member_is_ignored() {
        let card = card_with("", &["subscribe:ghost:alerts"], &["a"]);
        assert!(derive(&card).is_empty());
        assert_eq!(derived_for_member(&card, "ghost"), Vec::<String>::new());
    }

    #[test]
    fn directed_tag_pattern_may_contain_colons() {
        let card = card_with("", &["subscribe:a:ns:alerts"], &["a"]);
        assert_eq!(derive(&card).get("a"), Some(&["ns:alerts".to_owned()][..]));
    }

    #[test]
    fn overrides_replace_the_derived_set_verbatim() {
        let card = card_with("team", &["subscribe:alerts"], &["a"]);
        let overrides = BTreeMap::from([
            ("a".to_owned(), vec!["extra".to_owned(), "extra".to_owned()]),
            ("ghost".to_owned(), vec!["phantom".to_owned()]),
        ]);
        let plan = derive_with_overrides(&card, &overrides);
        assert_eq!(plan.get("a"), Some(&["extra".to_owned()][..]));
        assert_eq!(plan.get("ghost"), Some(&["phantom".to_owned()][..]));
        assert_eq!(plan.agents(), vec!["a".to_owned(), "ghost".to_owned()]);
    }

    #[test]
    fn override_with_an_empty_list_is_kept() {
        let card = card_with("team", &[], &["a"]);
        let overrides = BTreeMap::from([("a".to_owned(), Vec::new())]);
        let plan = derive_with_overrides(&card, &overrides);
        assert!(!plan.is_empty());
        assert_eq!(plan.get("a"), Some(&[][..]));
        assert_eq!(plan.subscription_count(), 0);
    }

    #[test]
    fn event_card_uses_the_subscription_map() {
        let card = EventDrivenTeamCard {
            card: card_with("team", &[], &["a"]),
            subscriptions: BTreeMap::from([("a".to_owned(), vec!["only".to_owned()])]),
        };
        let plan = derive_from_event_card(&card);
        assert_eq!(plan.get("a"), Some(&["only".to_owned()][..]));
    }

    #[test]
    fn derivation_is_deterministic_and_deduplicated() {
        let card = card_with(
            "team",
            &["subscribe:team", "subscribe:team.*", "subscribe:team"],
            &["b", "a", "a"],
        );
        let first = derive(&card);
        let second = derive(&card);
        assert_eq!(first, second);
        assert_eq!(
            first.get("a"),
            Some(&["team".to_owned(), "team.*".to_owned(), "team.a".to_owned()][..])
        );
        assert_eq!(
            first.get("b"),
            Some(&["team".to_owned(), "team.*".to_owned(), "team.b".to_owned()][..])
        );
        assert_eq!(first.subscription_count(), 6);
    }

    #[test]
    fn derived_for_member_matches_derive() {
        let card = card_with("team", &["subscribe:alerts"], &["a"]);
        assert_eq!(
            derived_for_member(&card, "a"),
            vec![
                "alerts".to_owned(),
                "team".to_owned(),
                "team.*".to_owned(),
                "team.a".to_owned()
            ]
        );
        assert_eq!(derived_for_member(&card, "nobody"), Vec::<String>::new());
    }

    #[test]
    fn schema_exposes_the_patterns_map() {
        let schema = serde_json::to_value(schemars::schema_for!(AutoSubscriptionPlan)).unwrap();
        let text = schema.to_string();
        assert!(text.contains("patterns"));
        let mode_schema =
            serde_json::to_value(schemars::schema_for!(AutoSubscriptionMode)).unwrap();
        let text = mode_schema.to_string();
        assert!(text.contains("\"merge\""));
        assert!(text.contains("\"disabled\""));
    }
}
