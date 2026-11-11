//! Scoped persistence helpers for entity/relation memory on a
//! [`GraphStore`].
//!
//! Every helper here is a free function over `store: &dyn GraphStore` rather than a
//! wrapper type, so graph memory stays a thin layer over the foundation store and the
//! future `GraphMemory` facade can reuse the same seams (including its own stats and
//! delete paths).
//!
//! # Partitioning
//!
//! Graph nodes and edges carry `user_id`/`scope_id` properties and every read is
//! filtered by them ([`scope_query`]). The partition is *user plus scope*, not
//! session or agent: every session and agent of one workspace shares its graph, which
//! matches the Python `GraphMemory` scope model and lets recall expand neighbors
//! across sessions.
//!
//! # Identity
//!
//! Ids are derived ([`entity_node_id`], [`relation_id`]) from sanitized scope ids and
//! the normalized entity name, so re-adding the same entity updates one node instead
//! of accumulating duplicates. [`normalize_name`] folds case and whitespace;
//! [`sanitize_segment`] keeps ids free of separators and path escapes.

use openjiuwen_common::error::Result;
use openjiuwen_foundation::store::{GraphQuery, GraphStore};
use serde_json::json;

use super::entity::{Entity, Relation, ENTITY_LABEL, RELATION_LABEL};
use crate::record::MemoryScope;

/// The 32-bit FNV-1a offset basis.
const FNV1A_32_OFFSET_BASIS: u32 = 0x811c_9dc5;
/// The 32-bit FNV-1a prime.
const FNV1A_32_PRIME: u32 = 0x0100_0193;

/// Makes `value` safe as one id segment.
///
/// A value that is non-empty, not `.`/`..`, and made only of `[A-Za-z0-9._-]` is
/// returned unchanged. Otherwise it is filtered to that alphabet (an empty result
/// becomes `_`) and suffixed with `_` plus the 8 lowercase hex digits of the 32-bit
/// FNV-1a hash of the original bytes, so two unsafe values that filter to the same
/// text stay distinct. The result is never `.` or `..`.
///
/// Mirrors `store::keys::sanitize_segment`; the logic is duplicated here because the
/// graph module must not depend on the record store module.
#[must_use]
pub fn sanitize_segment(value: &str) -> String {
    if is_safe_segment(value) {
        return value.to_owned();
    }
    let filtered: String = value.chars().filter(|c| is_segment_char(*c)).collect();
    let base = if filtered.is_empty() {
        "_".to_owned()
    } else {
        filtered
    };
    let mut sanitized = format!("{base}_{:08x}", fnv1a_32(value.as_bytes()));
    if sanitized == "." || sanitized == ".." {
        sanitized.insert(0, '_');
    }
    sanitized
}

/// Normalizes an entity or relation name for identity purposes.
///
/// Lowercases (full Unicode lowercase), trims, and collapses every run of internal
/// whitespace — ASCII or Unicode, including CJK ideographic spaces — to one ASCII
/// space. An empty name stays empty.
#[must_use]
pub fn normalize_name(name: &str) -> String {
    name.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Derives the node id of the entity `name` in `scope`.
///
/// `entity:{user}:{scope}:{name}` where the scope ids are sanitized and the name is
/// normalized and has its spaces replaced with `-` (`"Alice Smith"` becomes
/// `alice-smith`); unsafe characters gain the [`sanitize_segment`] hash suffix.
#[must_use]
pub fn entity_node_id(scope: &MemoryScope, name: &str) -> String {
    format!(
        "entity:{}:{}:{}",
        sanitize_segment(&scope.user_id),
        sanitize_segment(&scope.scope_id),
        entity_name_segment(name)
    )
}

/// Derives the id of the relation between two entities of `scope`.
///
/// `rel:{user}:{scope}:{from_node}:{type}:{to_node}`: the endpoints are the entity
/// node ids from [`entity_node_id`] and the type is normalized with `_` between its
/// words (`"Works At"` becomes `works_at`). The mirror node and the edge share it.
#[must_use]
pub fn relation_id(scope: &MemoryScope, from: &str, relation_type: &str, to: &str) -> String {
    format!(
        "rel:{}:{}:{}:{}:{}",
        sanitize_segment(&scope.user_id),
        sanitize_segment(&scope.scope_id),
        entity_node_id(scope, from),
        sanitize_segment(&normalize_name(relation_type).replace(' ', "_")),
        entity_node_id(scope, to)
    )
}

/// Builds the node query that selects one scope's nodes carrying any of `labels`.
///
/// The filter is exactly `{"user_id": ..., "scope_id": ...}`: a session or agent id on
/// `scope` deliberately does **not** narrow the query, because every session and agent
/// of a user/scope partition shares one graph. `limit` applies after the store sorts
/// by id.
#[must_use]
pub fn scope_query(scope: &MemoryScope, labels: &[&str], limit: Option<usize>) -> GraphQuery {
    GraphQuery {
        labels: labels.iter().map(|label| (*label).to_owned()).collect(),
        filter: json!({
            "user_id": scope.user_id,
            "scope_id": scope.scope_id,
        }),
        limit,
    }
}

/// Returns the scope's entities, sorted by node id, at most `limit` of them.
///
/// Nodes that do not convert to an [`Entity`] (for example a relation mirror node that
/// shares the label list, or a foreign node in the same scope) are skipped rather than
/// treated as errors, so a label collision can never make recall panic.
pub async fn query_entities(
    store: &dyn GraphStore,
    scope: &MemoryScope,
    limit: Option<usize>,
) -> Result<Vec<Entity>> {
    let nodes = store
        .query_nodes(&scope_query(scope, &[ENTITY_LABEL], limit))
        .await?;
    Ok(nodes.iter().filter_map(Entity::from_node).collect())
}

/// Returns every relation of `scope`, sorted by mirror node id.
///
/// Non-convertible nodes are skipped, mirroring [`query_entities`].
pub async fn query_relations(store: &dyn GraphStore, scope: &MemoryScope) -> Result<Vec<Relation>> {
    let nodes = store
        .query_nodes(&scope_query(scope, &[RELATION_LABEL], None))
        .await?;
    Ok(nodes.iter().filter_map(Relation::from_node).collect())
}

/// Looks up the entity named `name` in `scope`.
///
/// `Ok(None)` when no node exists under the derived id or when that node is not an
/// entity.
pub async fn get_entity(
    store: &dyn GraphStore,
    scope: &MemoryScope,
    name: &str,
) -> Result<Option<Entity>> {
    let id = entity_node_id(scope, name);
    Ok(store
        .get_node(&id)
        .await?
        .and_then(|node| Entity::from_node(&node)))
}

/// Inserts or updates `entity` and returns what was stored.
///
/// When a node already exists under `entity.id` and converts to an [`Entity`], the
/// stored entity is merged with the incoming one ([`Entity::merge`]: incoming
/// properties win, maximum importance, incoming embedding), so a repeated add keeps the
/// id and accumulates data instead of replacing it. Otherwise the incoming entity is
/// stored as-is.
pub async fn upsert_entity(store: &dyn GraphStore, entity: Entity) -> Result<Entity> {
    let existing = store
        .get_node(&entity.id)
        .await?
        .and_then(|node| Entity::from_node(&node));
    let stored = match existing {
        Some(current) => current.merge(&entity),
        None => entity,
    };
    store.upsert_nodes(vec![stored.to_node()]).await?;
    Ok(stored)
}

/// Persists `relation` twice: its mirror node and its edge.
///
/// The mirror node makes relations enumerable ([`query_relations`]) because a
/// [`GraphStore`] cannot query edges; the edge makes them traversable for neighbor
/// expansion. The node is written first, so a failed edge write leaves a recoverable
/// relation rather than an untraversable edge.
pub async fn upsert_relation(store: &dyn GraphStore, relation: Relation) -> Result<Relation> {
    let node = relation.to_node();
    let edge = relation.to_edge();
    store.upsert_nodes(vec![node]).await?;
    store.upsert_edges(vec![edge]).await?;
    Ok(relation)
}

/// Deletes the entity node `id` and every relation that touches it.
///
/// Returns `false` without mutating anything when no node exists under `id`. Deleting
/// the node already removes its incident edges; the scope's relations are then scanned
/// so the matching mirror nodes and their edge ids are removed explicitly.
pub async fn delete_entity(store: &dyn GraphStore, scope: &MemoryScope, id: &str) -> Result<bool> {
    if store.get_node(id).await?.is_none() {
        return Ok(false);
    }
    store.delete_nodes(&[id.to_owned()]).await?;
    let mut relation_ids = Vec::new();
    for relation in query_relations(store, scope).await? {
        let touches = entity_node_id(scope, &relation.from) == id
            || entity_node_id(scope, &relation.to) == id;
        if touches {
            relation_ids.push(relation.id);
        }
    }
    if !relation_ids.is_empty() {
        store.delete_edges(&relation_ids).await?;
        store.delete_nodes(&relation_ids).await?;
    }
    Ok(true)
}

/// Deletes every entity and relation of `scope`, returning how many entity nodes went.
///
/// Relation edges are deleted first (then their mirror nodes), then the entity nodes,
/// so no dangling edge survives the clear. Other scopes are untouched.
pub async fn clear_entities(store: &dyn GraphStore, scope: &MemoryScope) -> Result<usize> {
    let entities = query_entities(store, scope, None).await?;
    let relation_ids: Vec<String> = query_relations(store, scope)
        .await?
        .into_iter()
        .map(|relation| relation.id)
        .collect();
    if !relation_ids.is_empty() {
        store.delete_edges(&relation_ids).await?;
    }
    let mut node_ids: Vec<String> = entities.iter().map(|entity| entity.id.clone()).collect();
    node_ids.extend(relation_ids);
    if !node_ids.is_empty() {
        store.delete_nodes(&node_ids).await?;
    }
    Ok(entities.len())
}

/// Counts the entities stored in `scope`.
pub async fn count_entities(store: &dyn GraphStore, scope: &MemoryScope) -> Result<usize> {
    Ok(query_entities(store, scope, None).await?.len())
}

/// Counts the relations stored in `scope`.
pub async fn count_relations(store: &dyn GraphStore, scope: &MemoryScope) -> Result<usize> {
    Ok(query_relations(store, scope).await?.len())
}

/// Returns the id segment of an entity name: normalized, spaces to `-`, sanitized.
fn entity_name_segment(name: &str) -> String {
    sanitize_segment(&normalize_name(name).replace(' ', "-"))
}

/// Whether `value` is a non-empty safe segment that needs no rewriting.
fn is_safe_segment(value: &str) -> bool {
    !value.is_empty() && value != "." && value != ".." && value.chars().all(is_segment_char)
}

/// Whether `c` belongs to the accepted segment alphabet `[A-Za-z0-9._-]`.
fn is_segment_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')
}

/// Computes the 32-bit FNV-1a hash of `bytes`.
fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash = FNV1A_32_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV1A_32_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::store::InMemoryGraphStore;
    use serde_json::json;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn entity(name: &str, entity_type: &str) -> Entity {
        Entity::new(&scope(), name, entity_type)
    }

    #[test]
    fn sanitize_segment_passes_safe_values_through() {
        for value in ["alice", "a.b-c_1", "0", "__default__", "Alice"] {
            assert_eq!(sanitize_segment(value), value);
        }
    }

    #[test]
    fn sanitize_segment_suffixes_unsafe_values_deterministically() {
        let once = sanitize_segment("alice/bob");
        assert_eq!(once, sanitize_segment("alice/bob"));
        assert_eq!(once, "alicebob_d30866c3");
        assert_ne!(sanitize_segment("alice bob"), once);
        assert_eq!(sanitize_segment("a b"), "ab_10a3f9f2");
        assert_eq!(sanitize_segment(""), "__811c9dc5");
    }

    #[test]
    fn sanitize_segment_never_returns_dot_segments() {
        for value in [".", "..", "./", "../", "", "???", " ", "a\0b"] {
            let sanitized = sanitize_segment(value);
            assert_ne!(sanitized, ".");
            assert_ne!(sanitized, "..");
            assert_ne!(sanitized, "");
            assert!(!sanitized.contains('/'));
        }
        assert!(sanitize_segment(".").starts_with('.'));
    }

    #[test]
    fn normalize_name_folds_case_and_whitespace() {
        assert_eq!(normalize_name("  Alice   Smith  "), "alice smith");
        assert_eq!(normalize_name("Alice\tSmith\nJr"), "alice smith jr");
        assert_eq!(normalize_name("Alice\u{3000}Smith"), "alice smith");
        assert_eq!(normalize_name(""), "");
        assert_eq!(normalize_name("   "), "");
        assert_eq!(normalize_name("ÜBER"), "über");
    }

    #[test]
    fn entity_node_ids_use_the_normalized_name() {
        assert_eq!(
            entity_node_id(&scope(), "Alice Smith"),
            "entity:alice:ws:alice-smith"
        );
        assert_eq!(
            entity_node_id(&scope(), "  alice   smith "),
            "entity:alice:ws:alice-smith"
        );
        assert_eq!(
            entity_node_id(&MemoryScope::new("alice/bob", "ws name"), "ACME"),
            format!(
                "entity:{}:{}:acme",
                sanitize_segment("alice/bob"),
                sanitize_segment("ws name")
            )
        );
        let unsafe_name = entity_node_id(&scope(), "O'Brien");
        assert!(unsafe_name.starts_with("entity:alice:ws:obrien_"));
        assert_eq!(unsafe_name, entity_node_id(&scope(), "o'brien"));
    }

    #[test]
    fn relation_ids_include_both_endpoints_and_the_type() {
        assert_eq!(
            relation_id(&scope(), "Alice Smith", "Works At", "Acme Corp"),
            format!(
                "rel:alice:ws:{}:works_at:{}",
                entity_node_id(&scope(), "Alice Smith"),
                entity_node_id(&scope(), "Acme Corp")
            )
        );
        assert_eq!(
            Relation::new(&scope(), "a", "b", "uses").id,
            relation_id(&scope(), "a", "uses", "b")
        );
    }

    #[test]
    fn scope_query_filters_on_user_and_scope_only() {
        let bound = MemoryScope::new("alice", "ws").session("s1").agent("a1");
        let query = scope_query(&bound, &["Entity"], Some(3));
        assert_eq!(query.labels, vec!["Entity".to_owned()]);
        assert_eq!(query.limit, Some(3));
        assert_eq!(query.filter, json!({"user_id": "alice", "scope_id": "ws"}));
    }

    #[tokio::test]
    async fn upsert_entity_merges_repeat_writes_and_keeps_one_node() {
        let store = InMemoryGraphStore::new();
        let mut first = entity("Alice Smith", "Person");
        first
            .properties
            .insert("content".to_owned(), json!("first"));
        first.properties.insert("importance".to_owned(), json!(0.2));
        let stored = upsert_entity(&store, first).await.expect("upsert");
        assert_eq!(stored.id, "entity:alice:ws:alice-smith");

        let mut second = entity("ALICE  SMITH", "Person");
        second
            .properties
            .insert("content".to_owned(), json!("second"));
        second.properties.insert("role".to_owned(), json!("owner"));
        second
            .properties
            .insert("importance".to_owned(), json!(0.9));
        let merged = upsert_entity(&store, second).await.expect("upsert");

        assert_eq!(merged.id, stored.id);
        assert_eq!(merged.name, "Alice Smith");
        assert_eq!(merged.properties["content"], json!("second"));
        assert_eq!(merged.properties["role"], json!("owner"));
        assert_eq!(merged.importance(), Some(0.9));
        assert_eq!(count_entities(&store, &scope()).await.expect("count"), 1);
        assert_eq!(store.len().await.expect("len").0, 1);
    }

    #[tokio::test]
    async fn upsert_relation_stores_a_mirror_node_and_an_edge() {
        let store = InMemoryGraphStore::new();
        upsert_entity(&store, entity("Alice Smith", "Person"))
            .await
            .expect("entity");
        upsert_entity(&store, entity("Acme", "Project"))
            .await
            .expect("entity");
        let relation = upsert_relation(
            &store,
            Relation::new(&scope(), "Alice Smith", "Acme", "works_at"),
        )
        .await
        .expect("relation");

        assert_eq!(store.len().await.expect("len"), (3, 1));
        assert_eq!(count_relations(&store, &scope()).await.expect("count"), 1);
        let relations = query_relations(&store, &scope()).await.expect("relations");
        assert_eq!(relations, vec![relation.clone()]);
        let edge = store
            .get_edge(&relation.id)
            .await
            .expect("edge")
            .expect("edge exists");
        assert_eq!(edge.label, "works_at");
        assert_eq!(edge.from, "entity:alice:ws:alice-smith");
        assert_eq!(edge.to, "entity:alice:ws:acme");
        let node = store
            .get_node(&relation.id)
            .await
            .expect("node")
            .expect("mirror node exists");
        assert_eq!(
            node.labels,
            vec!["Relation".to_owned(), "works_at".to_owned()]
        );
    }

    #[tokio::test]
    async fn queries_are_isolated_by_scope() {
        let store = InMemoryGraphStore::new();
        upsert_entity(&store, entity("Alice", "Person"))
            .await
            .expect("alice");
        upsert_relation(&store, Relation::new(&scope(), "Alice", "Acme", "works_at"))
            .await
            .expect("relation");

        for other in [
            MemoryScope::new("alice", "other-ws"),
            MemoryScope::new("bob", "ws"),
        ] {
            let entities = query_entities(&store, &other, None)
                .await
                .expect("entities");
            assert_eq!(entities, Vec::new());
            let relations = query_relations(&store, &other).await.expect("relations");
            assert_eq!(relations, Vec::new());
            assert_eq!(count_entities(&store, &other).await.expect("count"), 0);
            assert_eq!(count_relations(&store, &other).await.expect("count"), 0);
        }
        assert_eq!(count_entities(&store, &scope()).await.expect("count"), 1);
    }

    #[tokio::test]
    async fn get_entity_finds_the_node_by_name() {
        let store = InMemoryGraphStore::new();
        assert_eq!(
            get_entity(&store, &scope(), "Alice").await.expect("lookup"),
            None
        );
        let stored = upsert_entity(&store, entity("Alice Smith", "Person"))
            .await
            .expect("upsert");
        assert_eq!(
            get_entity(&store, &scope(), "Alice Smith")
                .await
                .expect("lookup"),
            Some(stored.clone())
        );
        assert_eq!(
            get_entity(&store, &scope(), "  ALICE   smith ")
                .await
                .expect("lookup"),
            Some(stored)
        );
        assert_eq!(
            get_entity(&store, &MemoryScope::new("bob", "ws"), "Alice Smith")
                .await
                .expect("lookup"),
            None
        );
    }

    #[tokio::test]
    async fn delete_entity_removes_touching_relations() {
        let store = InMemoryGraphStore::new();
        let alice = upsert_entity(&store, entity("Alice", "Person"))
            .await
            .expect("alice");
        upsert_entity(&store, entity("Acme", "Project"))
            .await
            .expect("acme");
        upsert_entity(&store, entity("Bob", "Person"))
            .await
            .expect("bob");
        upsert_relation(&store, Relation::new(&scope(), "Alice", "Acme", "works_at"))
            .await
            .expect("relation");
        upsert_relation(&store, Relation::new(&scope(), "Bob", "Acme", "works_at"))
            .await
            .expect("relation");
        assert_eq!(count_relations(&store, &scope()).await.expect("count"), 2);

        assert!(delete_entity(&store, &scope(), &alice.id)
            .await
            .expect("delete"));

        let remaining = query_relations(&store, &scope()).await.expect("relations");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].from, "Bob");
        assert_eq!(count_entities(&store, &scope()).await.expect("count"), 2);
        assert_eq!(
            get_entity(&store, &scope(), "Alice").await.expect("lookup"),
            None
        );
        // Two entities plus one mirror node, one edge.
        assert_eq!(store.len().await.expect("len"), (3, 1));

        assert!(!delete_entity(&store, &scope(), "entity:alice:ws:missing")
            .await
            .expect("delete missing"));
        assert_eq!(store.len().await.expect("len"), (3, 1));
    }

    #[tokio::test]
    async fn clear_entities_only_clears_the_scope() {
        let store = InMemoryGraphStore::new();
        upsert_entity(&store, entity("Alice", "Person"))
            .await
            .expect("alice");
        upsert_entity(&store, entity("Acme", "Project"))
            .await
            .expect("acme");
        upsert_relation(&store, Relation::new(&scope(), "Alice", "Acme", "works_at"))
            .await
            .expect("relation");

        let other = MemoryScope::new("alice", "other-ws");
        upsert_entity(&store, Entity::new(&other, "Alice", "Person"))
            .await
            .expect("other entity");
        upsert_relation(&store, Relation::new(&other, "Alice", "Acme", "works_at"))
            .await
            .expect("other relation");

        assert_eq!(clear_entities(&store, &scope()).await.expect("clear"), 2);
        assert_eq!(count_entities(&store, &scope()).await.expect("count"), 0);
        assert_eq!(count_relations(&store, &scope()).await.expect("count"), 0);
        assert_eq!(store.len().await.expect("len"), (2, 1));

        assert_eq!(count_entities(&store, &other).await.expect("count"), 1);
        assert_eq!(count_relations(&store, &other).await.expect("count"), 1);
        assert_eq!(clear_entities(&store, &scope()).await.expect("clear"), 0);
    }
}
