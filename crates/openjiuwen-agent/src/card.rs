//! The [`AgentCard`]: registry-safe identity of an agent.
//!
//! Python parity: `AgentCard` in `openjiuwen/core/single_agent/schema/card.py`,
//! with the common `BaseCard` fields (`id`, `name`, `description`) plus the
//! optional input/output parameter schemas and interface URL.
use openjiuwen_common::card::Card;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
/// Serialized identity and metadata of an agent.
///
/// The default id is a fresh 32-character lowercase hex uuid, matching Python's
/// `uuid.uuid4().hex`. Unknown JSON fields are tolerated and dropped on
/// deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentCard {
    /// Unique identifier; a fresh uuid hex by default.
    pub id: String,
    /// Human-readable name, unique inside the agent registry.
    pub name: String,
    /// Free-form description used for discovery and intent matching.
    pub description: String,
    /// JSON schema of the agent's accepted inputs, when declared.
    pub input_params: Option<Value>,
    /// JSON schema of the agent's produced outputs, when declared.
    pub output_params: Option<Value>,
    /// Remote interface URL for network-hosted agents.
    pub interface_url: Option<String>,
}
impl Default for AgentCard {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().simple().to_string(),
            name: String::new(),
            description: String::new(),
            input_params: None,
            output_params: None,
            interface_url: None,
        }
    }
}
impl AgentCard {
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
    /// Sets the input parameter schema.
    #[must_use]
    pub fn with_input_params(mut self, params: Value) -> Self {
        self.input_params = Some(params);
        self
    }
    /// Sets the output parameter schema.
    #[must_use]
    pub fn with_output_params(mut self, params: Value) -> Self {
        self.output_params = Some(params);
        self
    }
    /// Sets the remote interface URL.
    #[must_use]
    pub fn with_interface_url(mut self, url: impl Into<String>) -> Self {
        self.interface_url = Some(url.into());
        self
    }
    /// Returns the identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}
impl Card for AgentCard {
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_id_is_32_lowercase_hex() {
        let card = AgentCard::default();
        assert_eq!(card.id.len(), 32);
        assert!(card
            .id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_eq!(card.name, "");
        assert_eq!(card.description, "");
        assert!(card.input_params.is_none());
        assert!(card.output_params.is_none());
        assert!(card.interface_url.is_none());
    }
    #[test]
    fn json_round_trip() {
        let card = AgentCard::new("assistant")
            .with_id("0123456789abcdef0123456789abcdef")
            .with_description("helps")
            .with_input_params(serde_json::json!({"type": "object"}))
            .with_output_params(serde_json::json!({"type": "string"}))
            .with_interface_url("http://localhost/agent");
        let value = serde_json::to_value(&card).unwrap();
        let restored: AgentCard = serde_json::from_value(value).unwrap();
        assert_eq!(restored, card);
    }
    #[test]
    fn unknown_keys_are_tolerated() {
        let card: AgentCard = serde_json::from_value(serde_json::json!({
            "id": "abc",
            "name": "n",
            "description": "d",
            "surprise": {"nested": true}
        }))
        .unwrap();
        assert_eq!(card.id, "abc");
        assert_eq!(card.name, "n");
        assert_eq!(card.description, "d");
    }
    #[test]
    fn optional_fields_serialize_as_null() {
        let value = serde_json::to_value(AgentCard::new("n")).unwrap();
        assert!(value.get("input_params").unwrap().is_null());
        assert!(value.get("output_params").unwrap().is_null());
        assert!(value.get("interface_url").unwrap().is_null());
    }
    #[test]
    fn builders_set_fields() {
        let card = AgentCard::new("first")
            .with_id("id-1")
            .with_name("second")
            .with_description("desc")
            .with_interface_url("http://example.test");
        assert_eq!(card.id(), "id-1");
        assert_eq!(card.name(), "second");
        assert_eq!(card.description(), "desc");
        assert_eq!(card.interface_url.as_deref(), Some("http://example.test"));
    }
    #[test]
    fn card_trait_description_is_none_when_empty() {
        assert_eq!(Card::description(&AgentCard::new("n")), None);
        assert_eq!(
            Card::description(&AgentCard::new("n").with_description("d")),
            Some("d")
        );
    }
}
