//! [`Entity`] and [`Relation`]: the graph-memory data model and its node/edge mapping.
//!
//! Graph memory stores *entities* as [`GraphNode`]s labelled [`ENTITY_LABEL`] and
//! *relations* as [`GraphEdge`]s between those node ids. Because
//! [`GraphStore`](openjiuwen_foundation::store::GraphStore) has no edge query, every
//! relation is additionally persisted as a mirror [`GraphNode`] labelled
//! [`RELATION_LABEL`], so a scope's relations can be enumerated with
//! [`GraphStore::query_nodes`](openjiuwen_foundation::store::GraphStore::query_nodes).
//! Both representations carry the same relation id and are written together by
//! [`super::store::upsert_relation`].
//!
//! Node and edge ids are derived, not random ([`Entity::new`], [`Relation::new`]):
//! re-adding the same entity in the same scope therefore updates the existing node
//! instead of creating a second one. The derivation lives in [`super::store`].
use std::collections::BTreeMap;

use openjiuwen_foundation::store::{GraphEdge, GraphNode};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::store::{entity_node_id, relation_id};
use crate::record::MemoryScope;

/// Label carried by every entity node.
pub const ENTITY_LABEL: &str = "Entity";
/// Label carried by every relation mirror node.
pub const RELATION_LABEL: &str = "Relation";

/// Property key holding an entity's free-form importance in `[0, 1]`.
const IMPORTANCE_KEY: &str = "importance";
/// Property key holding an entity's embedding vector.
const EMBEDDING_KEY: &str = "embedding";
/// Entity type used when neither a property nor a label names one.
const DEFAULT_ENTITY_TYPE: &str = "Concept";

/// Node properties that describe the entity rather than carry user data.
///
/// [`Entity::from_node`] moves every other property back into
/// [`Entity::properties`], so a round-trip neither duplicates nor loses fields.
const ENTITY_RESERVED_KEYS: [&str; 6] = [
    "name",
    "entity_type",
    "user_id",
    "scope_id",
    "session_id",
    "agent_id",
];
/// Relation node/edge properties that describe the relation or its scope.
const RELATION_RESERVED_KEYS: [&str; 9] = [
    "from",
    "to",
    "relation_type",
    "from_node",
    "to_node",
    "user_id",
    "scope_id",
    "session_id",
    "agent_id",
];

/// A named, typed graph-memory entity owned by one [`MemoryScope`].
///
/// `id` is derived from the scope and the name ([`Entity::new`]) so the same entity
/// always lands on the same node. Free-form data lives in `properties`; the
/// `importance` and `embedding` conventions are exposed through
/// [`Entity::importance`] and [`Entity::embedding`].
///
/// JSON is flat: `entity_type` serializes as `"type"` (and accepts `"entity_type"` on
/// input), and the scope fields are flattened at the top level, matching the Python
/// memory documents.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Entity {
    /// Derived node id, `entity:{user}:{scope}:{name}`.
    pub id: String,
    /// Human-readable name the node id is derived from.
    pub name: String,
    /// Entity kind (`Person`, `Project`, `Concept`, ...); JSON key `type`.
    #[serde(rename = "type", alias = "entity_type")]
    pub entity_type: String,
    /// Free-form entity attributes; ordered so serialization is stable.
    pub properties: BTreeMap<String, Value>,
    /// Owning user/scope (and optional session/agent), flattened into the JSON object.
    #[serde(flatten)]
    pub scope: MemoryScope,
}

impl Entity {
    /// Builds an entity in `scope`, deriving its [`Entity::id`] from the name.
    ///
    /// `properties` start empty, so callers add `importance`, `content`, `embedding`,
    /// ... explicitly.
    #[must_use]
    pub fn new(
        scope: &MemoryScope,
        name: impl Into<String>,
        entity_type: impl Into<String>,
    ) -> Self {
        let name = name.into();
        Self {
            id: entity_node_id(scope, &name),
            name,
            entity_type: entity_type.into(),
            properties: BTreeMap::new(),
            scope: scope.clone(),
        }
    }

    /// Rebuilds an entity from a stored node, or `None` when the node is not one.
    ///
    /// A node qualifies when it carries the [`ENTITY_LABEL`] label and string
    /// properties `name`, `user_id`, and `scope_id`; anything else (for example a
    /// relation mirror node) yields `None`. The type comes from the `entity_type`
    /// property, else from the first non-[`ENTITY_LABEL`] label, else
    /// `"Concept"`. Every property outside the reserved identity/scope keys is
    /// restored into [`Entity::properties`].
    #[must_use]
    pub fn from_node(node: &GraphNode) -> Option<Self> {
        if !node.labels.iter().any(|label| label == ENTITY_LABEL) {
            return None;
        }
        let name = node_property(node, "name")?;
        let user_id = node_property(node, "user_id")?;
        let scope_id = node_property(node, "scope_id")?;
        let entity_type = node_property(node, "entity_type")
            .filter(|value| !value.is_empty())
            .or_else(|| {
                node.labels
                    .iter()
                    .find(|label| label.as_str() != ENTITY_LABEL)
                    .cloned()
            })
            .unwrap_or_else(|| DEFAULT_ENTITY_TYPE.to_owned());
        let scope = MemoryScope {
            user_id,
            scope_id,
            session_id: node_property(node, "session_id"),
            agent_id: node_property(node, "agent_id"),
        };
        let id = if node.id.is_empty() {
            entity_node_id(&scope, &name)
        } else {
            node.id.clone()
        };
        Some(Self {
            id,
            properties: user_properties(&node.properties, &ENTITY_RESERVED_KEYS),
            name,
            entity_type,
            scope,
        })
    }

    /// Projects the entity onto its [`GraphNode`].
    ///
    /// Labels are `["Entity", entity_type]` (the type label is omitted when the type is
    /// empty); properties are [`Entity::properties`] plus `name`, `entity_type`,
    /// `user_id`, `scope_id`, and `session_id`/`agent_id` when present. The id is used
    /// verbatim — build entities through [`Entity::new`] so it is derived from the
    /// scope and name.
    #[must_use]
    pub fn to_node(&self) -> GraphNode {
        let mut properties = value_map(&self.properties);
        properties.insert("name".to_owned(), Value::String(self.name.clone()));
        properties.insert(
            "entity_type".to_owned(),
            Value::String(self.entity_type.clone()),
        );
        insert_scope(&mut properties, &self.scope);
        GraphNode {
            id: self.id.clone(),
            labels: primary_and_type_labels(ENTITY_LABEL, &self.entity_type),
            properties,
        }
    }

    /// Returns the numeric `importance` property as `f32`, when it is finite.
    #[must_use]
    pub fn importance(&self) -> Option<f32> {
        self.properties
            .get(IMPORTANCE_KEY)
            .and_then(Value::as_f64)
            .and_then(narrow_f32)
    }

    /// Returns the `embedding` property as a finite vector.
    ///
    /// `None` when the property is absent, not an array, empty, or holds a non-numeric
    /// or non-finite entry.
    #[must_use]
    pub fn embedding(&self) -> Option<Vec<f32>> {
        let values = self.properties.get(EMBEDDING_KEY)?.as_array()?;
        if values.is_empty() {
            return None;
        }
        let mut vector = Vec::with_capacity(values.len());
        for value in values {
            vector.push(narrow_f32(value.as_f64()?)?);
        }
        Some(vector)
    }

    /// Returns a new entity merging `incoming` into `self`.
    ///
    /// Deterministic and side-effect free:
    /// - `id` and `scope` stay `self`'s, so a re-added entity cannot change owners;
    /// - `name` stays `self`'s (the id is derived from it);
    /// - `entity_type` becomes `incoming`'s when it is non-empty, else `self`'s;
    /// - `properties` are `self`'s overlaid with `incoming`'s (incoming wins per key);
    /// - `importance` becomes the maximum when both sides are numeric, otherwise
    ///   whichever side has one;
    /// - `embedding` becomes `incoming`'s when it is a valid vector, else `self`'s is
    ///   restored.
    #[must_use]
    pub fn merge(&self, incoming: &Entity) -> Entity {
        let mut properties = self.properties.clone();
        for (key, value) in &incoming.properties {
            properties.insert(key.clone(), value.clone());
        }
        let importance = match (self.importance(), incoming.importance()) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        };
        if let Some(value) = importance {
            properties.insert(IMPORTANCE_KEY.to_owned(), json_number(value));
        }
        // An incoming vector wins; otherwise the stored one is restored when the
        // incoming property is absent or unusable. Neither valid keeps the overlay.
        if let Some(vector) = incoming.embedding().or_else(|| self.embedding()) {
            properties.insert(EMBEDDING_KEY.to_owned(), json_vector(&vector));
        }
        let entity_type = if incoming.entity_type.is_empty() {
            self.entity_type.clone()
        } else {
            incoming.entity_type.clone()
        };
        Self {
            id: self.id.clone(),
            name: self.name.clone(),
            entity_type,
            properties,
            scope: self.scope.clone(),
        }
    }
}

/// A typed, directed relation between two entities, identified by their *names*.
///
/// `from`/`to` are entity names, not node ids: the node ids are derived on projection
/// ([`Relation::to_edge`], [`Relation::to_node`]), so a relation survives a sanitized
/// or normalized rename without dangling ids.
///
/// JSON matches [`Entity`]: `relation_type` serializes as `"type"` (accepting
/// `"relation_type"` on input) and the scope is flattened.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Relation {
    /// Derived id shared by the mirror node and the edge.
    pub id: String,
    /// Name of the source entity.
    pub from: String,
    /// Name of the target entity.
    pub to: String,
    /// Relation kind (`works_at`, `uses`, ...); JSON key `type`.
    #[serde(rename = "type", alias = "relation_type")]
    pub relation_type: String,
    /// Free-form relation attributes; ordered so serialization is stable.
    pub properties: BTreeMap<String, Value>,
    /// Owning user/scope (and optional session/agent), flattened into the JSON object.
    #[serde(flatten)]
    pub scope: MemoryScope,
}

impl Relation {
    /// Builds a relation in `scope`, deriving its [`Relation::id`] from the endpoints.
    #[must_use]
    pub fn new(
        scope: &MemoryScope,
        from: impl Into<String>,
        to: impl Into<String>,
        relation_type: impl Into<String>,
    ) -> Self {
        let from = from.into();
        let to = to.into();
        let relation_type = relation_type.into();
        Self {
            id: relation_id(scope, &from, &relation_type, &to),
            from,
            to,
            relation_type,
            properties: BTreeMap::new(),
            scope: scope.clone(),
        }
    }

    /// Projects the relation onto its [`GraphEdge`].
    ///
    /// `from`/`to` become the entity *node* ids, `label` becomes the relation type, and
    /// the properties carry the entity names plus `relation_type`, `user_id`,
    /// `scope_id`, and `session_id`/`agent_id` when present.
    #[must_use]
    pub fn to_edge(&self) -> GraphEdge {
        let mut properties = value_map(&self.properties);
        properties.insert("from".to_owned(), Value::String(self.from.clone()));
        properties.insert("to".to_owned(), Value::String(self.to.clone()));
        properties.insert(
            "relation_type".to_owned(),
            Value::String(self.relation_type.clone()),
        );
        insert_scope(&mut properties, &self.scope);
        GraphEdge {
            id: self.id.clone(),
            from: entity_node_id(&self.scope, &self.from),
            to: entity_node_id(&self.scope, &self.to),
            label: self.relation_type.clone(),
            properties,
        }
    }

    /// Projects the relation onto its mirror [`GraphNode`].
    ///
    /// Labels are `["Relation", relation_type]`; properties carry `from`/`to` names,
    /// `relation_type`, the derived `from_node`/`to_node` entity ids, and the scope.
    /// The mirror node exists because a [`GraphStore`](openjiuwen_foundation::store::GraphStore)
    /// cannot enumerate edges; each relation is persisted twice (node plus edge) under
    /// the same id.
    #[must_use]
    pub fn to_node(&self) -> GraphNode {
        let mut properties = value_map(&self.properties);
        properties.insert("from".to_owned(), Value::String(self.from.clone()));
        properties.insert("to".to_owned(), Value::String(self.to.clone()));
        properties.insert(
            "relation_type".to_owned(),
            Value::String(self.relation_type.clone()),
        );
        properties.insert(
            "from_node".to_owned(),
            Value::String(entity_node_id(&self.scope, &self.from)),
        );
        properties.insert(
            "to_node".to_owned(),
            Value::String(entity_node_id(&self.scope, &self.to)),
        );
        insert_scope(&mut properties, &self.scope);
        GraphNode {
            id: self.id.clone(),
            labels: primary_and_type_labels(RELATION_LABEL, &self.relation_type),
            properties,
        }
    }

    /// Rebuilds a relation from its mirror node, or `None` when the node is not one.
    ///
    /// Requires the [`RELATION_LABEL`] label and string properties `from`, `to`,
    /// `relation_type`, `user_id`, and `scope_id`; the identity/scope keys are dropped
    /// from [`Relation::properties`].
    #[must_use]
    pub fn from_node(node: &GraphNode) -> Option<Self> {
        if !node.labels.iter().any(|label| label == RELATION_LABEL) {
            return None;
        }
        let from = node_property(node, "from")?;
        let to = node_property(node, "to")?;
        let relation_type = node_property(node, "relation_type")?;
        let scope = MemoryScope {
            user_id: node_property(node, "user_id")?,
            scope_id: node_property(node, "scope_id")?,
            session_id: node_property(node, "session_id"),
            agent_id: node_property(node, "agent_id"),
        };
        let id = if node.id.is_empty() {
            relation_id(&scope, &from, &relation_type, &to)
        } else {
            node.id.clone()
        };
        Some(Self {
            id,
            from,
            to,
            relation_type,
            properties: user_properties(&node.properties, &RELATION_RESERVED_KEYS),
            scope,
        })
    }
}

/// Returns the string property `key` of `node`.
fn node_property(node: &GraphNode, key: &str) -> Option<String> {
    node.properties
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Copies the non-reserved node properties into an ordered property map.
fn user_properties(properties: &Map<String, Value>, reserved: &[&str]) -> BTreeMap<String, Value> {
    properties
        .iter()
        .filter(|(key, _)| !reserved.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Converts an ordered property map into the node/edge JSON object.
fn value_map(properties: &BTreeMap<String, Value>) -> Map<String, Value> {
    properties
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Writes the scope fields into `properties` (session/agent only when present).
fn insert_scope(properties: &mut Map<String, Value>, scope: &MemoryScope) {
    properties.insert("user_id".to_owned(), Value::String(scope.user_id.clone()));
    properties.insert("scope_id".to_owned(), Value::String(scope.scope_id.clone()));
    if let Some(session_id) = &scope.session_id {
        properties.insert("session_id".to_owned(), Value::String(session_id.clone()));
    }
    if let Some(agent_id) = &scope.agent_id {
        properties.insert("agent_id".to_owned(), Value::String(agent_id.clone()));
    }
}

/// Builds `[primary]`, or `[primary, secondary]` when `secondary` is non-empty.
fn primary_and_type_labels(primary: &str, secondary: &str) -> Vec<String> {
    let mut labels = vec![primary.to_owned()];
    if !secondary.is_empty() {
        labels.push(secondary.to_owned());
    }
    labels
}

/// Narrows a finite `f64` to a finite `f32`.
fn narrow_f32(value: f64) -> Option<f32> {
    let narrowed = value as f32;
    narrowed.is_finite().then_some(narrowed)
}

/// Encodes an `f32` as a JSON number, promoting to `f64` so no precision warning fires.
fn json_number(value: f32) -> Value {
    Value::from(f64::from(value))
}

/// Encodes an embedding vector as a JSON array of numbers.
fn json_vector(vector: &[f32]) -> Value {
    Value::Array(vector.iter().copied().map(json_number).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn entity() -> Entity {
        let mut entity = Entity::new(&scope(), "Alice Smith", "Person");
        entity
            .properties
            .insert("importance".to_owned(), json!(0.4));
        entity
            .properties
            .insert("content".to_owned(), json!("knows the API"));
        entity
    }

    #[test]
    fn entity_round_trips_through_graph_node() {
        let entity = entity();
        let node = entity.to_node();
        assert_eq!(node.id, entity.id);
        assert_eq!(node.labels, vec!["Entity".to_owned(), "Person".to_owned()]);
        assert_eq!(node.properties["name"], json!("Alice Smith"));
        assert_eq!(node.properties["entity_type"], json!("Person"));
        assert_eq!(node.properties["user_id"], json!("alice"));
        assert_eq!(node.properties["scope_id"], json!("ws"));
        assert!(!node.properties.contains_key("session_id"));
        assert_eq!(Entity::from_node(&node), Some(entity));
    }

    #[test]
    fn entity_round_trip_keeps_session_and_agent() {
        let scope = MemoryScope::new("alice", "ws").session("s1").agent("a1");
        let entity = Entity::new(&scope, "API", "Concept");
        let node = entity.to_node();
        assert_eq!(node.properties["session_id"], json!("s1"));
        assert_eq!(node.properties["agent_id"], json!("a1"));
        assert_eq!(Entity::from_node(&node), Some(entity));
    }

    #[test]
    fn entity_from_node_requires_label_and_identity() {
        let node = entity().to_node();
        let mut wrong_label = node.clone();
        wrong_label.labels = vec!["Person".to_owned()];
        assert_eq!(Entity::from_node(&wrong_label), None);

        let mut missing_name = node.clone();
        missing_name.properties.remove("name");
        assert_eq!(Entity::from_node(&missing_name), None);

        let mut missing_user = node;
        missing_user.properties.remove("user_id");
        assert_eq!(Entity::from_node(&missing_user), None);

        let mut bare = GraphNode {
            id: String::new(),
            labels: vec!["Entity".to_owned()],
            properties: Map::new(),
        };
        bare.properties.insert("name".to_owned(), json!("Solo"));
        bare.properties.insert("user_id".to_owned(), json!("alice"));
        bare.properties.insert("scope_id".to_owned(), json!("ws"));
        let restored = Entity::from_node(&bare).expect("minimal node converts");
        assert_eq!(restored.entity_type, "Concept");
        assert_eq!(restored.id, entity_node_id(&scope(), "Solo"));
        assert!(restored.properties.is_empty());
    }

    #[test]
    fn entity_from_node_prefers_the_type_property_over_labels() {
        let mut node = entity().to_node();
        node.labels = vec!["Entity".to_owned(), "Document".to_owned()];
        node.properties
            .insert("entity_type".to_owned(), json!("Project"));
        assert_eq!(
            Entity::from_node(&node).expect("converts").entity_type,
            "Project"
        );
        node.properties.remove("entity_type");
        assert_eq!(
            Entity::from_node(&node).expect("converts").entity_type,
            "Document"
        );
        node.labels = vec!["Entity".to_owned()];
        assert_eq!(
            Entity::from_node(&node).expect("converts").entity_type,
            "Concept"
        );
    }

    #[test]
    fn relation_round_trips_through_edge_and_node() {
        let relation = Relation::new(&scope(), "Alice Smith", "Acme", "works_at");
        let edge = relation.to_edge();
        assert_eq!(edge.id, relation.id);
        assert_eq!(edge.from, entity_node_id(&scope(), "Alice Smith"));
        assert_eq!(edge.to, entity_node_id(&scope(), "Acme"));
        assert_eq!(edge.label, "works_at");
        assert_eq!(edge.properties["from"], json!("Alice Smith"));
        assert_eq!(edge.properties["to"], json!("Acme"));
        assert_eq!(edge.properties["relation_type"], json!("works_at"));
        assert_eq!(edge.properties["user_id"], json!("alice"));
        assert_eq!(edge.properties["scope_id"], json!("ws"));

        let node = relation.to_node();
        assert_eq!(node.id, relation.id);
        assert_eq!(
            node.labels,
            vec!["Relation".to_owned(), "works_at".to_owned()]
        );
        assert_eq!(node.properties["from_node"], json!(edge.from));
        assert_eq!(node.properties["to_node"], json!(edge.to));
        assert_eq!(Relation::from_node(&node), Some(relation));
    }

    #[test]
    fn relation_from_node_rejects_entities_and_incomplete_relations() {
        let entity_node = entity().to_node();
        assert_eq!(Relation::from_node(&entity_node), None);

        let node = Relation::new(&scope(), "a", "b", "uses").to_node();
        for key in ["from", "to", "relation_type", "user_id", "scope_id"] {
            let mut incomplete = node.clone();
            incomplete.properties.remove(key);
            assert_eq!(Relation::from_node(&incomplete), None, "missing {key}");
        }
    }

    #[test]
    fn ids_are_derived_from_scope_and_names() {
        let entity = Entity::new(&scope(), "Alice  Smith", "Person");
        assert_eq!(entity.id, "entity:alice:ws:alice-smith");

        let relation = Relation::new(&scope(), "Alice Smith", "Acme Corp", "Works At");
        assert_eq!(
            relation.id,
            "rel:alice:ws:entity:alice:ws:alice-smith:works_at:entity:alice:ws:acme-corp"
        );
    }

    #[test]
    fn entity_json_uses_type_and_flattens_scope() {
        let entity = Entity::new(&scope(), "Alice Smith", "Person");
        let value = serde_json::to_value(&entity).expect("serializes");
        assert_eq!(value["type"], json!("Person"));
        assert_eq!(value["name"], json!("Alice Smith"));
        assert_eq!(value["user_id"], json!("alice"));
        assert_eq!(value["scope_id"], json!("ws"));
        assert!(value.get("entity_type").is_none());
        assert!(value.get("scope").is_none());
        assert!(value.get("session_id").is_none());
        assert!(value.get("agent_id").is_none());

        let restored: Entity = serde_json::from_value(value).expect("deserializes");
        assert_eq!(restored, entity);
    }

    #[test]
    fn entity_json_accepts_the_long_field_names_and_unknown_fields() {
        let restored: Entity = serde_json::from_value(json!({
            "id": "entity:alice:ws:api",
            "name": "API",
            "entity_type": "Concept",
            "user_id": "alice",
            "scope_id": "ws",
            "session_id": "s1",
            "future_field": {"nested": true},
        }))
        .expect("deserializes");
        assert_eq!(restored.entity_type, "Concept");
        assert_eq!(restored.scope.session_id.as_deref(), Some("s1"));
        assert!(restored.properties.is_empty());
    }

    #[test]
    fn relation_json_uses_type_field() {
        let relation = Relation::new(&scope(), "Alice", "Acme", "works_at");
        let value = serde_json::to_value(&relation).expect("serializes");
        assert_eq!(value["type"], json!("works_at"));
        assert_eq!(value["from"], json!("Alice"));
        assert_eq!(value["to"], json!("Acme"));
        assert_eq!(value["user_id"], json!("alice"));
        assert!(value.get("relation_type").is_none());

        let restored: Relation = serde_json::from_value(json!({
            "id": relation_id(&scope(), "Alice", "uses", "Acme"),
            "from": "Alice",
            "to": "Acme",
            "relation_type": "uses",
            "user_id": "alice",
            "scope_id": "ws",
            "ignored": 1,
        }))
        .expect("deserializes");
        assert_eq!(restored.relation_type, "uses");
        assert_eq!(restored, Relation::new(&scope(), "Alice", "Acme", "uses"));
    }

    #[test]
    fn defaults_round_trip_and_stay_flat() {
        let value = serde_json::to_value(Entity::default()).expect("serializes");
        assert_eq!(value["type"], json!(""));
        assert_eq!(value["user_id"], json!("__default__"));
        let entity: Entity = serde_json::from_value(value).expect("deserializes");
        assert_eq!(entity, Entity::default());
        assert_eq!(entity.scope, MemoryScope::default());

        let relation: Relation =
            serde_json::from_value(serde_json::to_value(Relation::default()).expect("serializes"))
                .expect("deserializes");
        assert_eq!(relation, Relation::default());
    }

    #[test]
    fn merge_overlays_properties_and_keeps_identity() {
        let stored = entity();
        let mut incoming = Entity::new(&MemoryScope::new("bob", "other"), "Alice Smith", "Actor");
        incoming
            .properties
            .insert("content".to_owned(), json!("updated"));
        incoming
            .properties
            .insert("role".to_owned(), json!("owner"));

        let merged = stored.merge(&incoming);
        assert_eq!(merged.id, stored.id);
        assert_eq!(merged.name, stored.name);
        assert_eq!(merged.scope, stored.scope);
        assert_eq!(merged.entity_type, "Actor");
        assert_eq!(merged.properties["content"], json!("updated"));
        assert_eq!(merged.properties["role"], json!("owner"));
        assert_eq!(merged.importance(), Some(0.4));

        let mut type_less = incoming.clone();
        type_less.entity_type = String::new();
        assert_eq!(stored.merge(&type_less).entity_type, "Person");
    }

    #[test]
    fn merge_takes_max_importance_and_replaces_embedding() {
        let mut stored = entity();
        stored
            .properties
            .insert("importance".to_owned(), json!(0.8));
        stored
            .properties
            .insert("embedding".to_owned(), json!([1.0, 0.0]));

        let mut incoming = Entity::new(&scope(), "Alice Smith", "Person");
        incoming
            .properties
            .insert("importance".to_owned(), json!(0.3));

        let merged = stored.merge(&incoming);
        assert_eq!(merged.importance(), Some(0.8_f32));
        assert_eq!(merged.embedding(), Some(vec![1.0, 0.0]));

        incoming
            .properties
            .insert("embedding".to_owned(), json!([0.0, 1.0]));
        let merged = stored.merge(&incoming);
        assert_eq!(merged.embedding(), Some(vec![0.0, 1.0]));
        assert_eq!(merged.importance(), Some(0.8_f32));

        let mut only_incoming = Entity::new(&scope(), "Alice Smith", "Person");
        only_incoming
            .properties
            .insert("importance".to_owned(), json!(0.25));
        assert_eq!(
            Entity::new(&scope(), "Bob", "Person")
                .merge(&only_incoming)
                .importance(),
            Some(0.25)
        );
    }

    #[test]
    fn importance_and_embedding_ignore_invalid_values() {
        let mut entity = Entity::new(&scope(), "Alice", "Person");
        assert_eq!(entity.importance(), None);
        assert_eq!(entity.embedding(), None);

        entity
            .properties
            .insert("importance".to_owned(), json!("high"));
        assert_eq!(entity.importance(), None);

        entity
            .properties
            .insert("importance".to_owned(), json!(0.75));
        assert_eq!(entity.importance(), Some(0.75));

        for invalid in [
            json!([]),
            json!("vec"),
            json!([1.0, "x"]),
            json!([1.0, null]),
        ] {
            entity.properties.insert("embedding".to_owned(), invalid);
            assert_eq!(entity.embedding(), None);
        }
        entity
            .properties
            .insert("embedding".to_owned(), json!([0.5, 0.25]));
        assert_eq!(entity.embedding(), Some(vec![0.5, 0.25]));
    }
}
