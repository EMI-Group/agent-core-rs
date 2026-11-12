#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Wire-shape tests for `TeamCard`, `EventDrivenTeamCard`, and `TeamConfig`.
//!
//! These pin the Python-compatible JSON surface of the crate's data types: the
//! seven `TeamCard` fields (`agent_cards`, `description`, `id`, `name`, `tags`,
//! `topic`, `version`), the `"1.0.0"` default version, the flattened
//! `EventDrivenTeamCard` with its `subscriptions` map, and the `TeamConfig`
//! defaults `10` / `100` / `30.0`. Everything is pure serialization work, so
//! the suite is offline and deterministic.

mod common;

use openjiuwen_common::card::Card;
use openjiuwen_multiagent::{EventDrivenTeamCard, TeamCard, TeamConfig};
use serde_json::json;

/// The exact Python field set of a `TeamCard`, sorted.
const CARD_FIELDS: [&str; 7] = [
    "agent_cards",
    "description",
    "id",
    "name",
    "tags",
    "topic",
    "version",
];

#[test]
fn default_card_wire_shape() {
    let value = serde_json::to_value(TeamCard::default()).unwrap();
    let object = value.as_object().unwrap();
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, CARD_FIELDS);
    assert_eq!(object["version"], "1.0.0");
    assert_eq!(object["agent_cards"], json!([]));
    assert_eq!(object["tags"], json!([]));
    assert_eq!(object["name"], "");
    assert_eq!(object["description"], "");
    assert_eq!(object["topic"], "");
}

#[test]
fn card_id_is_uuid_hex_and_version_defaults() {
    let card: TeamCard = serde_json::from_value(json!({"name": "t"})).unwrap();
    assert_eq!(card.name, "t");
    assert_eq!(card.version, "1.0.0");
    assert_eq!(card.id.len(), 32);
    assert!(card
        .id
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(Card::name(&card), "t");
    assert_eq!(Card::description(&card), None);
}

#[test]
fn agent_cards_round_trip() {
    let card = TeamCard::new("team").with_agent_cards(vec![common::agent_card("worker")]);
    let value = serde_json::to_value(&card).unwrap();
    assert_eq!(value["agent_cards"][0]["id"], "worker");
    assert_eq!(value["agent_cards"][0]["name"], "worker");
    let restored: TeamCard = serde_json::from_value(value).unwrap();
    assert_eq!(restored, card);
}

#[test]
fn team_config_wire_shape() {
    let value = serde_json::to_value(TeamConfig::new()).unwrap();
    assert_eq!(
        value,
        json!({
            "max_agents": 10,
            "max_concurrent_messages": 100,
            "message_timeout": 30.0
        })
    );
}

#[test]
fn event_driven_card_flattens_and_serializes_subscriptions() {
    let card = EventDrivenTeamCard {
        card: TeamCard::new("team"),
        subscriptions: [("worker".to_owned(), vec!["topic.*".to_owned()])]
            .into_iter()
            .collect(),
    };
    let value = serde_json::to_value(&card).unwrap();
    assert_eq!(value["name"], "team");
    assert_eq!(value["subscriptions"]["worker"][0], "topic.*");
    assert_eq!(value["version"], "1.0.0");
    assert!(value.get("card").is_none());

    let restored: EventDrivenTeamCard = serde_json::from_value(value).unwrap();
    assert_eq!(restored, card);
}
