//! Graph recall: seed scoring, BFS neighbor expansion, and hit ordering.
//!
//! Recall is deliberately simple and deterministic so offline tests can assert exact
//! scores:
//!
//! 1. Seeds are scored against the query by [`seed_score`]:
//!    `score = 0.7 * embedding_similarity + 0.3 * keyword_coverage`.
//! 2. The seeds are sorted by [`sort_hits`] (score descending, then depth, then id).
//! 3. Neighbors are expanded breadth-first for `max_depth` rounds with
//!    [`expand`]; a neighbor found in round `d` inherits
//!    `parent_score * 0.5^d`, so distance attenuates confidence geometrically.
//! 4. The merged hit list is ordered by [`sort_hits`] again, so a close, strongly
//!    scored seed always outranks the entities it pulled in.

use std::collections::HashSet;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::store::{Direction, GraphStore};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::entity::Entity;
use crate::recall::{keyword_score, normalized_similarity};

/// Weight of the embedding similarity in [`seed_score`].
const SIMILARITY_WEIGHT: f32 = 0.7;
/// Weight of the keyword coverage in [`seed_score`].
const KEYWORD_WEIGHT: f32 = 0.3;
/// Score factor applied per expansion hop (`0.5^depth`).
const HOP_ATTENUATION: f32 = 0.5;

/// One recalled entity with its score and graph distance from the closest seed.
///
/// `depth` is `0` for a seed (scored directly against the query) and the BFS round
/// number for an expanded neighbor. `score` is the [`seed_score`] of the entity for a
/// seed and the attenuated parent score for a neighbor. The entity's fields are
/// flattened into the JSON object, so a hit reads like the entity it wraps plus
/// `score`/`depth`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRecallHit {
    /// The recalled entity.
    #[serde(flatten)]
    pub entity: Entity,
    /// Recall score in `[0, 1]` for seeds; smaller for expanded neighbors.
    pub score: f32,
    /// Hops from the closest seed (`0` for the seed itself).
    pub depth: usize,
}

/// Renders the text a query is matched against: name, `content`, and `description`.
///
/// Only string properties take part; the pieces are single-space separated and
/// trimmed, so an entity without content or description matches on its name alone.
#[must_use]
pub fn entity_text(entity: &Entity) -> String {
    let mut parts: Vec<&str> = vec![entity.name.as_str()];
    for key in ["content", "description"] {
        if let Some(value) = entity.properties.get(key).and_then(Value::as_str) {
            parts.push(value);
        }
    }
    parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned()
}

/// Returns the normalized cosine similarity of `query_embedding` and the entity's
/// embedding, or `0.0` when either vector is absent.
#[must_use]
pub fn entity_similarity(entity: &Entity, query_embedding: Option<&[f32]>) -> f32 {
    match (query_embedding, entity.embedding()) {
        (Some(query), Some(vector)) => normalized_similarity(query, &vector),
        _ => 0.0,
    }
}

/// Scores a seed entity against `query`.
///
/// `0.7 * similarity + 0.3 * keyword`, where similarity is [`entity_similarity`] and
/// keyword is the [`keyword_score`] coverage of [`entity_text`]. The result is clamped
/// into `[0, 1]` and is always finite (a non-finite intermediate becomes `0.0`), so
/// callers may compare or sort scores without guards.
#[must_use]
pub fn seed_score(query: &str, entity: &Entity, query_embedding: Option<&[f32]>) -> f32 {
    let similarity = entity_similarity(entity, query_embedding);
    let keyword = keyword_score(query, &entity_text(entity));
    let score = SIMILARITY_WEIGHT * similarity + KEYWORD_WEIGHT * keyword;
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Orders hits by `score` descending, then `depth` ascending, then `entity.id`
/// ascending.
///
/// The comparison uses [`f32::total_cmp`], so `NaN` scores still produce a total order
/// instead of panicking, and the trailing id tie-break makes the order unique.
#[allow(clippy::ptr_arg)] // the `&mut Vec` receiver is the documented recall API
pub fn sort_hits(hits: &mut Vec<GraphRecallHit>) {
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.depth.cmp(&right.depth))
            .then_with(|| left.entity.id.cmp(&right.entity.id))
    });
}

/// Expands `seeds` with their graph neighbors, breadth-first.
///
/// Every round `d` in `1..=max_depth` visits each hit of the previous frontier (sorted
/// by [`sort_hits`]), follows all of its edges
/// ([`GraphStore::neighbors`] with [`Direction::Both`], so an entity can be reached
/// through an incoming or an outgoing relation), and keeps the first `max_neighbors`
/// not-yet-visited neighbors per parent. A neighbor's score is
/// `parent.score * 0.5^d` and its depth is `d`; nodes that do not convert to an
/// [`Entity`] (relation mirror nodes, foreign nodes) are skipped silently.
///
/// The seeds themselves are always part of the result, every id appears at most once
/// (a hit is only added the first time it is reached, at the shallowest depth), and the
/// returned vector is sorted by [`sort_hits`]. `max_depth == 0` or `max_neighbors == 0`
/// therefore returns exactly the sorted seeds.
pub async fn expand(
    store: &dyn GraphStore,
    seeds: Vec<GraphRecallHit>,
    max_depth: usize,
    max_neighbors: usize,
) -> Result<Vec<GraphRecallHit>> {
    let mut visited: HashSet<String> = seeds
        .iter()
        .map(|hit| hit.entity.id.clone())
        .collect::<HashSet<_>>();
    let mut result = seeds;
    sort_hits(&mut result);
    if max_depth == 0 || max_neighbors == 0 {
        return Ok(result);
    }
    let mut frontier = result.clone();
    let mut depth = 0;
    while depth < max_depth && !frontier.is_empty() {
        depth += 1;
        // `0.5^depth`, without a lossy `usize as i32` cast.
        let attenuation = HOP_ATTENUATION.powi(i32::try_from(depth).unwrap_or(i32::MAX));
        sort_hits(&mut frontier);
        let mut next_frontier = Vec::new();
        for parent in &frontier {
            let neighbors = store
                .neighbors(&parent.entity.id, Direction::Both, None)
                .await?;
            let mut taken = 0;
            for node in neighbors {
                if taken >= max_neighbors {
                    break;
                }
                let Some(entity) = Entity::from_node(&node) else {
                    continue;
                };
                if !visited.insert(entity.id.clone()) {
                    continue;
                }
                taken += 1;
                let hit = GraphRecallHit {
                    entity,
                    score: parent.score * attenuation,
                    depth,
                };
                result.push(hit.clone());
                next_frontier.push(hit);
            }
        }
        frontier = next_frontier;
    }
    sort_hits(&mut result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::store::InMemoryGraphStore;
    use serde_json::json;

    use crate::graph::entity::Relation;
    use crate::graph::store::{upsert_entity, upsert_relation};
    use crate::record::MemoryScope;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn hit(entity: Entity, score: f32, depth: usize) -> GraphRecallHit {
        GraphRecallHit {
            entity,
            score,
            depth,
        }
    }

    fn entity(name: &str) -> Entity {
        Entity::new(&scope(), name, "Concept")
    }

    /// Alice — Acme — Bob, all in scope `alice/ws`.
    async fn connected_store() -> InMemoryGraphStore {
        let store = InMemoryGraphStore::new();
        for entity in [entity("Alice"), entity("Acme"), entity("Bob")] {
            upsert_entity(&store, entity).await.expect("entity");
        }
        upsert_relation(&store, Relation::new(&scope(), "Alice", "Acme", "works_at"))
            .await
            .expect("relation");
        upsert_relation(&store, Relation::new(&scope(), "Acme", "Bob", "employs"))
            .await
            .expect("relation");
        store
    }

    #[test]
    fn entity_text_composes_name_content_and_description() {
        let mut entity = Entity::new(&scope(), "Alice", "Person");
        assert_eq!(entity_text(&entity), "Alice");

        entity
            .properties
            .insert("content".to_owned(), json!("knows the API"));
        assert_eq!(entity_text(&entity), "Alice knows the API");

        entity
            .properties
            .insert("description".to_owned(), json!("lead engineer"));
        assert_eq!(entity_text(&entity), "Alice knows the API lead engineer");

        entity.properties.insert("content".to_owned(), json!("  "));
        assert_eq!(entity_text(&entity), "Alice lead engineer");

        entity
            .properties
            .insert("description".to_owned(), json!(42));
        assert_eq!(entity_text(&entity), "Alice");

        assert_eq!(entity_text(&Entity::default()), "");
    }

    #[test]
    fn seed_score_blends_similarity_and_keyword() {
        let mut entity = Entity::new(&scope(), "Alice", "Person");
        entity
            .properties
            .insert("content".to_owned(), json!("widget expert"));
        entity
            .properties
            .insert("embedding".to_owned(), json!([1.0, 0.0]));

        let keyword_only = seed_score("alice unknown", &entity, None);
        assert!((keyword_only - 0.3 * 0.5).abs() < 1e-6);

        let with_embedding = seed_score("alice unknown", &entity, Some(&[1.0, 0.0]));
        assert!((with_embedding - (0.7 + 0.3 * 0.5)).abs() < 1e-6);

        let orthogonal = seed_score("alice unknown", &entity, Some(&[0.0, 1.0]));
        assert!((orthogonal - (0.7 * 0.5 + 0.3 * 0.5)).abs() < 1e-6);

        assert!(seed_score("", &entity, None).is_finite());
        assert_eq!(entity_similarity(&entity, Some(&[1.0, 0.0])), 1.0);
        assert_eq!(entity_similarity(&entity, Some(&[1.0])), 0.0);

        let without_embedding = Entity::new(&scope(), "Alice", "Person");
        assert_eq!(
            entity_similarity(&without_embedding, Some(&[1.0, 0.0])),
            0.0
        );
    }

    #[test]
    fn keyword_only_scores_rank_the_matching_entity_first() {
        let alice = entity("Alice");
        let unrelated = entity("Postgres database");
        let alice_score = seed_score("Alice", &alice, None);
        let other_score = seed_score("Alice", &unrelated, None);
        assert!(alice_score > other_score);
        assert!((alice_score - 0.3).abs() < 1e-6);
        assert!(other_score.abs() < 1e-6);
        for score in [alice_score, other_score] {
            assert!(score.is_finite());
            assert!((0.0..=1.0).contains(&score));
        }
    }

    #[test]
    fn sort_hits_breaks_ties_by_depth_then_id() {
        let mut hits = vec![
            hit(entity("b"), 0.5, 1),
            hit(entity("a"), 0.5, 1),
            hit(entity("c"), 0.5, 0),
            hit(entity("d"), 0.9, 3),
        ];
        sort_hits(&mut hits);
        let order: Vec<&str> = hits.iter().map(|hit| hit.entity.name.as_str()).collect();
        assert_eq!(order, vec!["d", "c", "a", "b"]);
    }

    #[tokio::test]
    async fn expand_orders_the_seed_before_its_neighbors() {
        let store = connected_store().await;
        let seed = hit(entity("Alice"), 0.8, 0);
        let hits = expand(&store, vec![seed], 2, 4).await.expect("expand");

        let order: Vec<&str> = hits.iter().map(|hit| hit.entity.name.as_str()).collect();
        assert_eq!(order, vec!["Alice", "Acme", "Bob"]);
        assert_eq!(hits[0].depth, 0);
        assert_eq!(hits[1].depth, 1);
        assert_eq!(hits[2].depth, 2);
    }

    #[tokio::test]
    async fn expand_attenuates_the_score_by_half_per_depth() {
        let store = connected_store().await;
        let hits = expand(&store, vec![hit(entity("Alice"), 0.8, 0)], 2, 4)
            .await
            .expect("expand");
        let score_of = |name: &str| {
            hits.iter()
                .find(|hit| hit.entity.name == name)
                .expect("hit")
                .score
        };
        // Depth 1: 0.8 * 0.5^1, depth 2: 0.4 * 0.5^2 (the attenuated parent score).
        assert!((score_of("Alice") - 0.8).abs() < 1e-6);
        assert!((score_of("Acme") - 0.4).abs() < 1e-6);
        assert!((score_of("Bob") - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn expand_reaches_both_edge_directions() {
        let store = connected_store().await;
        let hits = expand(&store, vec![hit(entity("Bob"), 0.6, 0)], 1, 4)
            .await
            .expect("expand");
        let names: Vec<&str> = hits.iter().map(|hit| hit.entity.name.as_str()).collect();
        assert_eq!(names, vec!["Bob", "Acme"]);
    }

    #[tokio::test]
    async fn expand_with_zero_depth_returns_sorted_seeds() {
        let store = connected_store().await;
        let seeds = vec![hit(entity("Bob"), 0.2, 0), hit(entity("Alice"), 0.9, 0)];
        let hits = expand(&store, seeds, 0, 4).await.expect("expand");
        let names: Vec<&str> = hits.iter().map(|hit| hit.entity.name.as_str()).collect();
        assert_eq!(names, vec!["Alice", "Bob"]);
    }

    #[tokio::test]
    async fn expand_with_zero_neighbors_returns_the_seeds_only() {
        let store = connected_store().await;
        let hits = expand(&store, vec![hit(entity("Alice"), 0.8, 0)], 3, 0)
            .await
            .expect("expand");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entity.name, "Alice");
    }

    #[tokio::test]
    async fn expand_never_duplicates_a_hit() {
        let store = connected_store().await;
        let seeds = vec![hit(entity("Alice"), 0.8, 0), hit(entity("Acme"), 0.7, 0)];
        let hits = expand(&store, seeds, 2, 4).await.expect("expand");
        let ids: HashSet<&str> = hits.iter().map(|hit| hit.entity.id.as_str()).collect();
        assert_eq!(ids.len(), hits.len());
        assert_eq!(hits.len(), 3);
        // Acme stays a seed (it was already visited), so Alice's expansion skips it.
        let acme = hits
            .iter()
            .find(|hit| hit.entity.name == "Acme")
            .expect("acme");
        assert_eq!(acme.depth, 0);
        assert!((acme.score - 0.7).abs() < 1e-6);
        let bob = hits
            .iter()
            .find(|hit| hit.entity.name == "Bob")
            .expect("bob");
        assert_eq!(bob.depth, 1);
        assert!((bob.score - 0.35).abs() < 1e-6);
    }

    #[tokio::test]
    async fn expand_caps_neighbors_per_parent() {
        let store = connected_store().await;
        // "Acme" neighbours are Alice and Bob, sorted by id: acme -> alice, bob.
        let hits = expand(&store, vec![hit(entity("Acme"), 0.5, 0)], 1, 1)
            .await
            .expect("expand");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].entity.name, "Alice");
    }

    #[tokio::test]
    async fn expand_stays_inside_the_seed_scope() {
        let store = connected_store().await;
        let other = MemoryScope::new("alice", "other-ws");
        for entity in [
            Entity::new(&other, "Alice", "Concept"),
            Entity::new(&other, "Zed", "Concept"),
        ] {
            upsert_entity(&store, entity).await.expect("entity");
        }
        upsert_relation(&store, Relation::new(&other, "Alice", "Zed", "knows"))
            .await
            .expect("relation");

        let hits = expand(&store, vec![hit(entity("Alice"), 0.8, 0)], 2, 8)
            .await
            .expect("expand");
        assert!(hits.iter().all(|hit| hit.entity.scope.scope_id == "ws"));
        assert!(hits.iter().all(|hit| hit.entity.name != "Zed"));
        let ids: HashSet<&str> = hits.iter().map(|hit| hit.entity.id.as_str()).collect();
        assert_eq!(ids.len(), hits.len());
    }
}
