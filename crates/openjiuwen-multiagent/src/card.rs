//! Team cards: the registry-safe identity of a team and its member agents.
//!
//! Python parity: `TeamCard` and `EventDrivenTeamCard` in
//! `openjiuwen/core/multi_agent/schema/card.py`, carrying the common card fields
//! (`id`, `name`, `description`) plus the member `agent_cards` list and the
//! pub/sub `topic`/`tags` metadata.

use openjiuwen_agent::AgentCard;
use openjiuwen_common::card::Card;
use openjiuwen_common::utils::id::uuid_hex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Serialized identity and metadata of a team.
///
/// The default id is a fresh 32-character lowercase hex uuid, matching Python's
/// `uuid.uuid4().hex`; `version` defaults to `"1.0.0"` and `agent_cards` to the
/// empty member list. Unknown JSON fields are tolerated and dropped on
/// deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamCard {
    /// Unique identifier; a fresh uuid hex by default.
    pub id: String,
    /// Human-readable name, unique inside the team registry.
    pub name: String,
    /// Free-form description used for discovery and routing.
    pub description: String,
    /// Cards of the agents that are members of this team.
    pub agent_cards: Vec<AgentCard>,
    /// Pub/sub topic associated with the team, when declared.
    pub topic: String,
    /// Card schema version.
    pub version: String,
    /// Free-form tags used for discovery.
    pub tags: Vec<String>,
}

impl Default for TeamCard {
    fn default() -> Self {
        Self {
            id: uuid_hex(),
            name: String::new(),
            description: String::new(),
            agent_cards: Vec::new(),
            topic: String::new(),
            version: "1.0.0".to_owned(),
            tags: Vec::new(),
        }
    }
}

impl TeamCard {
    /// Creates a card named `name` with a freshly generated id.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the identifier.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets the name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the member agent cards.
    #[must_use]
    pub fn with_agent_cards(mut self, agent_cards: Vec<AgentCard>) -> Self {
        self.agent_cards = agent_cards;
        self
    }

    /// Sets the pub/sub topic.
    #[must_use]
    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = topic.into();
        self
    }

    /// Sets the card version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Sets the discovery tags.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

impl Card for TeamCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        if self.description.is_empty() {
            None
        } else {
            Some(&self.description)
        }
    }
}

/// A [`TeamCard`] plus the topic subscriptions of an event-driven team.
///
/// Python parity: `EventDrivenTeamCard` in
/// `openjiuwen/core/multi_agent/schema/card.py`. The team fields are flattened
/// into the same JSON object, so the wire shape is a `TeamCard` with an extra
/// `subscriptions` map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct EventDrivenTeamCard {
    /// The underlying team card, flattened on the wire.
    #[serde(flatten)]
    pub card: TeamCard,
    /// Topic patterns to subscribe, keyed by subscriber agent id.
    pub subscriptions: BTreeMap<String, Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_card_has_uuid_hex_id_and_version() {
        let card = TeamCard::default();
        assert_eq!(card.id.len(), 32);
        assert!(card
            .id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_eq!(card.name, "");
        assert_eq!(card.description, "");
        assert_eq!(card.agent_cards, Vec::<AgentCard>::new());
        assert_eq!(card.topic, "");
        assert_eq!(card.version, "1.0.0");
        assert_eq!(card.tags, Vec::<String>::new());
    }

    #[test]
    fn json_uses_snake_case_field_names() {
        let value = serde_json::to_value(TeamCard::new("team")).unwrap();
        let object = value.as_object().unwrap();
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "agent_cards",
                "description",
                "id",
                "name",
                "tags",
                "topic",
                "version"
            ]
        );
        assert_eq!(object["name"], "team");
        assert_eq!(object["version"], "1.0.0");
    }

    #[test]
    fn empty_object_deserializes_to_defaults() {
        let card: TeamCard = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(card.id.len(), 32);
        assert_eq!(card.version, "1.0.0");
        assert_eq!(card.agent_cards, Vec::<AgentCard>::new());
    }

    #[test]
    fn unknown_keys_are_tolerated() {
        let card: TeamCard = serde_json::from_value(serde_json::json!({
            "name": "t",
            "surprise": {"nested": true}
        }))
        .unwrap();
        assert_eq!(card.name, "t");
    }

    #[test]
    fn round_trips_with_members() {
        let card = TeamCard::new("team")
            .with_id("0123456789abcdef0123456789abcdef")
            .with_description("desc")
            .with_agent_cards(vec![AgentCard::new("worker").with_id("worker")])
            .with_topic("team.topic")
            .with_version("2.0.0")
            .with_tags(vec!["tag".to_owned()]);
        let value = serde_json::to_value(&card).unwrap();
        let restored: TeamCard = serde_json::from_value(value).unwrap();
        assert_eq!(restored, card);
    }

    #[test]
    fn event_driven_card_flattens_the_team_fields() {
        let card = EventDrivenTeamCard {
            card: TeamCard::new("team"),
            subscriptions: BTreeMap::from([("worker".to_owned(), vec!["topic.*".to_owned()])]),
        };
        let value = serde_json::to_value(&card).unwrap();
        assert_eq!(value["name"], "team");
        assert_eq!(value["subscriptions"]["worker"][0], "topic.*");
        assert!(value.get("card").is_none());

        let restored: EventDrivenTeamCard = serde_json::from_value(value).unwrap();
        assert_eq!(restored, card);
    }

    #[test]
    fn card_trait_description_is_none_when_empty() {
        assert_eq!(Card::description(&TeamCard::new("t")), None);
        assert_eq!(
            Card::description(&TeamCard::new("t").with_description("d")),
            Some("d")
        );
    }
}
