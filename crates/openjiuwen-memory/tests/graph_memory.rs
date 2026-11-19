#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration coverage for the [`GraphMemory`] facade.
//!
//! The graph submodules have their own unit tests for id derivation, merge rules, BFS
//! scoring, and JSON parsing; this suite stays on the facade: extraction wiring,
//! persistence, seed scoring, neighbor expansion, scope isolation, deletion, and
//! cancellation.
//!
//! Extractions are built by hand where a heuristic quirk would obscure the assertion
//! and go through [`GraphMemory::add`] only to prove the extractor wiring.
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_foundation::store::{GraphStore, InMemoryGraphStore};
use openjiuwen_memory::config::GraphMemoryConfig;
use openjiuwen_memory::graph::recall::{entity_similarity, entity_text, seed_score};
use openjiuwen_memory::graph::store::upsert_entity;
use openjiuwen_memory::graph::{
    Entity, ExtractedEntity, ExtractedRelation, Extraction, GraphAddOptions, GraphMemory,
    GraphMemoryStats, GraphSearchOptions,
};
use openjiuwen_memory::recall::keyword_score;
use openjiuwen_memory::record::{CallOptions, MemoryScope};
use openjiuwen_retrieval::embedding::HashEmbedding;
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The scope shared by most tests.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// Builds an in-memory graph without any attachable capabilities.
fn memory(config: GraphMemoryConfig) -> GraphMemory {
    memory_with_store(config).0
}

/// Builds an in-memory graph plus a handle on its store, for tests that seed nodes
/// through the store helpers because [`Extraction`] carries no free-form properties.
fn memory_with_store(config: GraphMemoryConfig) -> (GraphMemory, Arc<dyn GraphStore>) {
    let store: Arc<dyn GraphStore> = Arc::new(InMemoryGraphStore::new());
    (GraphMemory::new(store.clone(), config), store)
}

/// A search that keeps every non-zero score and expands neighbors.
fn search_options() -> GraphSearchOptions {
    GraphSearchOptions {
        threshold: 0.01,
        ..Default::default()
    }
}

/// One extracted entity.
fn entity(name: &str, entity_type: &str, importance: Option<f32>) -> ExtractedEntity {
    ExtractedEntity {
        name: name.to_owned(),
        entity_type: entity_type.to_owned(),
        importance,
    }
}

/// One extracted relation.
fn relation(from: &str, to: &str, relation_type: &str) -> ExtractedRelation {
    ExtractedRelation {
        from: from.to_owned(),
        to: to.to_owned(),
        relation_type: relation_type.to_owned(),
    }
}

/// An extraction with no relations.
fn entities(names: &[&str]) -> Extraction {
    Extraction {
        entities: names
            .iter()
            .map(|name| entity(name, "Concept", None))
            .collect(),
        relations: Vec::new(),
        degraded: None,
    }
}

/// Persists `Alice -works_at-> Acme -uses-> Rust`.
async fn store_chain(memory: &GraphMemory, scope: &MemoryScope) {
    let extraction = Extraction {
        entities: vec![
            entity("Alice", "Person", None),
            entity("Acme", "Project", None),
            entity("Rust", "Concept", None),
        ],
        relations: vec![
            relation("Alice", "Acme", "works_at"),
            relation("Acme", "Rust", "uses"),
        ],
        degraded: None,
    };
    let (entities, relations) = memory
        .add_extracted(scope, &extraction)
        .await
        .expect("chain");
    assert_eq!((entities.len(), relations.len()), (3, 2));
}

#[tokio::test]
async fn add_extracted_persists_entities_relations_and_stats() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    let extraction = Extraction {
        entities: vec![
            entity("Alice", "Person", None),
            entity("Acme", "Project", None),
            entity("Rust", "Concept", None),
        ],
        relations: vec![
            relation("Alice", "Acme", "works_at"),
            relation("Acme", "Rust", "uses"),
        ],
        degraded: None,
    };
    let (stored_entities, stored_relations) = memory
        .add_extracted(&scope, &extraction)
        .await
        .expect("add");
    assert_eq!(stored_entities.len(), 3);
    assert_eq!(stored_relations.len(), 2);

    let entities = memory.entities(&scope, 0).await.expect("entities");
    let mut names: Vec<&str> = entities.iter().map(|item| item.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["Acme", "Alice", "Rust"]);

    let alice = memory
        .entity_named(&scope, "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert_eq!(alice.entity_type, "Person");
    assert_eq!(alice.scope, scope);

    let relations = memory.relations(&scope).await.expect("relations");
    let mut triplets: Vec<(&str, &str, &str)> = relations
        .iter()
        .map(|item| {
            (
                item.from.as_str(),
                item.relation_type.as_str(),
                item.to.as_str(),
            )
        })
        .collect();
    triplets.sort_unstable();
    assert_eq!(
        triplets,
        vec![("Acme", "uses", "Rust"), ("Alice", "works_at", "Acme")]
    );

    assert_eq!(
        memory.stats(&scope).await.expect("stats"),
        GraphMemoryStats {
            entities: 3,
            relations: 2,
        }
    );
}

#[tokio::test]
async fn repeat_add_merges_properties_and_keeps_the_id() {
    let (memory, store) = memory_with_store(GraphMemoryConfig::default());
    let scope = scope();
    let first = memory
        .add_extracted(
            &scope,
            &Extraction {
                entities: vec![entity("Alice", "Person", Some(0.2))],
                relations: Vec::new(),
                degraded: None,
            },
        )
        .await
        .expect("first add");
    let id = first.0[0].id.clone();

    // `Extraction` carries no free-form content, so the content property is written
    // through the store helper the facade uses.
    let mut rich = Entity::new(&scope, "Alice", "Person");
    rich.properties
        .insert("content".to_owned(), json!("Alice prefers dark mode"));
    upsert_entity(store.as_ref(), rich)
        .await
        .expect("dark content");

    let second = memory
        .add_extracted(
            &scope,
            &Extraction {
                entities: vec![entity("Alice", "Person", Some(0.6))],
                relations: Vec::new(),
                degraded: None,
            },
        )
        .await
        .expect("second add");
    assert_eq!(second.0[0].id, id, "the derived node id must stay stable");

    let stored = memory
        .entity_named(&scope, "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert_eq!(stored.id, id);
    assert_eq!(
        stored.properties.get("content"),
        Some(&json!("Alice prefers dark mode")),
        "properties from both writes survive"
    );
    assert!((stored.importance().expect("importance") - 0.6).abs() < 1e-6);
    assert_eq!(memory.stats(&scope).await.expect("stats").entities, 1);

    let hits = memory
        .search(&scope, "Alice", &search_options())
        .await
        .expect("search");
    assert_eq!(hits.len(), 1, "one node, one hit");
    assert_eq!(hits[0].entity.id, id);

    let mut light = Entity::new(&scope, "Alice", "Person");
    light
        .properties
        .insert("content".to_owned(), json!("Alice prefers light mode"));
    let merged = upsert_entity(store.as_ref(), light)
        .await
        .expect("light content");
    assert_eq!(
        merged.properties.get("content"),
        Some(&json!("Alice prefers light mode")),
        "the later content wins"
    );
}

#[tokio::test]
async fn search_scores_seeds_and_expands_neighbors() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    store_chain(&memory, &scope).await;

    let options = GraphSearchOptions {
        top_k: 1,
        threshold: 0.01,
        max_depth: 2,
        max_neighbors: 8,
        include_relations: true,
        call: CallOptions::default(),
    };
    let hits = memory
        .search(&scope, "Alice", &options)
        .await
        .expect("search");
    let names: Vec<&str> = hits.iter().map(|hit| hit.entity.name.as_str()).collect();
    assert_eq!(names, vec!["Alice", "Acme", "Rust"]);
    assert_eq!(
        hits.iter().map(|hit| hit.depth).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    let alice = hits[0].score;
    assert!(alice > hits[1].score);
    assert!(hits[1].score > hits[2].score);
    assert!((hits[1].score - alice * 0.5).abs() < 1e-6);
    assert!((hits[2].score - alice * 0.5 * 0.25).abs() < 1e-6);
}

#[tokio::test]
async fn search_without_an_embedding_is_keyword_only() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    store_chain(&memory, &scope).await;

    // Seeds only: expansion would pull in Alice's neighbors.
    let seeds_only = GraphSearchOptions {
        threshold: 0.01,
        include_relations: false,
        ..Default::default()
    };
    let unrelated = memory
        .search(&scope, "quantum chromodynamics", &seeds_only)
        .await
        .expect("search");
    assert_eq!(unrelated.len(), 0, "no entity matches unrelated terms");

    let hits = memory
        .search(&scope, "Alice", &seeds_only)
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!(hit.entity.name, "Alice");
    assert_eq!(hit.depth, 0);
    assert!(hit.score > 0.0);
    // Without an embedder the similarity half of the seed score is always zero, so the
    // score is exactly the 0.3-weighted keyword coverage.
    let keyword_only = 0.3 * keyword_score("Alice", &entity_text(&hit.entity));
    assert!((hit.score - keyword_only).abs() < 1e-6);
    assert!(hit.entity.embedding().is_none());
    assert!(entity_similarity(&hit.entity, Some(&[0.25; 8])).abs() < f32::EPSILON);
}

#[tokio::test]
async fn search_with_hash_embedding_uses_similarity() {
    let (memory, _store) = memory_with_store(GraphMemoryConfig::default());
    let memory = memory.with_embedding(Arc::new(HashEmbedding::new(32)));
    let scope = scope();
    // The entity text (its name) contains every query term, so the keyword half is 1.0.
    let query = "Rust ownership model";
    memory
        .add_extracted(&scope, &entities(&[query]))
        .await
        .expect("add");

    let stored = memory
        .entity_named(&scope, query)
        .await
        .expect("lookup")
        .expect("entity");
    let vector = stored.embedding().expect("stored embedding");
    assert_eq!(vector.len(), 32);

    let hits = memory
        .search(&scope, query, &search_options())
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!(hit.entity.name, query);
    let keyword_only = seed_score(query, &stored, None);
    assert!(
        hit.score > keyword_only,
        "embedding similarity ({}) must lift the score above the keyword-only {keyword_only}",
        hit.score
    );
}

#[tokio::test]
async fn heuristic_add_extracts_and_persists() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    let text = "I met Bob in New York; Bob uses Rust.";
    let extraction = memory
        .add(&scope, text, &GraphAddOptions::default())
        .await
        .expect("add");
    assert_ne!(
        extraction.entities.len(),
        0,
        "heuristics must find entities"
    );
    assert_eq!(
        memory.stats(&scope).await.expect("stats").entities,
        extraction.entities.len()
    );
    assert!(memory
        .entity_named(&scope, "Rust")
        .await
        .expect("lookup")
        .is_some());
    let hits = memory
        .search(&scope, "Rust", &search_options())
        .await
        .expect("search");
    assert!(hits.iter().any(|hit| hit.entity.name == "Rust"));
}

#[tokio::test]
async fn llm_add_uses_the_scripted_model() {
    let model = ScriptedModel::builder()
        .respond_text(
            r#"{"entities":[{"name":"Alice","type":"Person","importance":0.9}],"relations":[]}"#,
        )
        .build();
    let memory = memory(GraphMemoryConfig::default()).with_model(Arc::new(model));
    let scope = scope();
    let extraction = memory
        .add(&scope, "Alice", &GraphAddOptions::default())
        .await
        .expect("add");
    assert_eq!(extraction.entities.len(), 1);
    assert_eq!(extraction.entities[0].name, "Alice");
    assert_eq!(
        memory.stats(&scope).await.expect("stats"),
        GraphMemoryStats {
            entities: 1,
            relations: 0,
        }
    );
    let stored = memory
        .entity_named(&scope, "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert_eq!(stored.entity_type, "Person");
    assert!((stored.importance().expect("importance") - 0.9).abs() < 1e-6);
}

#[tokio::test]
async fn scope_isolation() {
    let memory = memory(GraphMemoryConfig::default());
    let ws = MemoryScope::new("alice", "ws");
    let other_scope = MemoryScope::new("alice", "other-ws");
    let other_user = MemoryScope::new("bob", "ws");
    for scope in [&ws, &other_scope, &other_user] {
        memory
            .add_extracted(scope, &entities(&["Alice"]))
            .await
            .expect("alice");
    }
    memory
        .add_extracted(
            &ws,
            &Extraction {
                // Both endpoints must come from the same extraction: relations name
                // entities, and unknown names are skipped as dangling.
                entities: vec![
                    entity("Alice", "Person", None),
                    entity("Acme", "Project", None),
                ],
                relations: vec![relation("Alice", "Acme", "works_at")],
                degraded: None,
            },
        )
        .await
        .expect("acme");

    assert_eq!(memory.entities(&ws, 0).await.expect("entities").len(), 2);
    assert_eq!(
        memory
            .entities(&other_scope, 0)
            .await
            .expect("entities")
            .len(),
        1
    );
    assert_eq!(
        memory
            .entities(&other_user, 0)
            .await
            .expect("entities")
            .len(),
        1
    );
    assert_eq!(memory.relations(&ws).await.expect("relations").len(), 1);
    assert_eq!(
        memory
            .relations(&other_scope)
            .await
            .expect("relations")
            .len(),
        0
    );
    assert_eq!(
        memory
            .relations(&other_user)
            .await
            .expect("relations")
            .len(),
        0
    );
    assert_eq!(
        memory.stats(&ws).await.expect("stats"),
        GraphMemoryStats {
            entities: 2,
            relations: 1,
        }
    );
    assert_eq!(
        memory.stats(&other_scope).await.expect("stats"),
        GraphMemoryStats {
            entities: 1,
            relations: 0,
        }
    );

    let hits = memory
        .search(&ws, "Alice", &search_options())
        .await
        .expect("search");
    let names: Vec<&str> = hits.iter().map(|hit| hit.entity.name.as_str()).collect();
    assert_eq!(names, vec!["Alice", "Acme"]);

    let other_hits = memory
        .search(&other_scope, "Alice", &search_options())
        .await
        .expect("search");
    assert_eq!(other_hits.len(), 1);
    assert_eq!(other_hits[0].entity.scope.scope_id, "other-ws");
}

#[tokio::test]
async fn delete_entity_removes_the_relations_that_touch_it() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    store_chain(&memory, &scope).await;
    let acme = memory
        .entities(&scope, 0)
        .await
        .expect("entities")
        .into_iter()
        .find(|item| item.name == "Acme")
        .expect("acme");

    assert!(memory
        .delete_entity(&scope, &acme.id)
        .await
        .expect("delete"));
    let stats = memory.stats(&scope).await.expect("stats");
    assert_eq!(stats.entities, 2);
    assert_eq!(stats.relations, 0);
    assert!(memory
        .entity_named(&scope, "Acme")
        .await
        .expect("lookup")
        .is_none());

    for name in ["Alice", "Rust"] {
        let hits = memory
            .search(&scope, name, &search_options())
            .await
            .expect("search");
        assert!(
            hits.iter().any(|hit| hit.entity.name == name),
            "remaining entity {name} must stay searchable"
        );
    }

    assert!(!memory
        .delete_entity(&scope, &acme.id)
        .await
        .expect("delete again"));
}

#[tokio::test]
async fn clear_removes_only_one_scope() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    let other = MemoryScope::new("alice", "other-ws");
    store_chain(&memory, &scope).await;
    memory
        .add_extracted(&other, &entities(&["Alice"]))
        .await
        .expect("other alice");

    assert_eq!(memory.clear(&scope).await.expect("clear"), 3);
    assert_eq!(
        memory.stats(&scope).await.expect("stats"),
        GraphMemoryStats::default()
    );
    assert_eq!(memory.entities(&scope, 0).await.expect("entities").len(), 0);
    assert_eq!(memory.relations(&scope).await.expect("relations").len(), 0);

    assert_eq!(
        memory.stats(&other).await.expect("stats"),
        GraphMemoryStats {
            entities: 1,
            relations: 0,
        }
    );
    let hits = memory
        .search(&other, "Alice", &search_options())
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(memory.clear(&scope).await.expect("clear empty"), 0);
}

#[tokio::test]
async fn validation_and_cancellation() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    let invalid = MemoryScope::new("alice", "bad/scope");

    for error in [
        memory
            .add(&scope, "   ", &GraphAddOptions::default())
            .await
            .expect_err("blank text"),
        memory
            .add(&invalid, "Alice", &GraphAddOptions::default())
            .await
            .expect_err("invalid scope"),
        memory
            .search(&scope, "   ", &search_options())
            .await
            .expect_err("blank query"),
        memory
            .search(&invalid, "Alice", &search_options())
            .await
            .expect_err("invalid search scope"),
        memory
            .entities(&invalid, 0)
            .await
            .expect_err("invalid entities scope"),
    ] {
        assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    }

    let token = CancellationToken::new();
    token.cancel();
    let add_options = GraphAddOptions {
        call: CallOptions::default().with_cancellation(token.clone()),
        ..GraphAddOptions::default()
    };
    let error = memory
        .add(&scope, "Alice", &add_options)
        .await
        .expect_err("cancelled add");
    assert!(matches!(error, Error::Cancelled), "got {error:?}");

    let search_options = GraphSearchOptions {
        call: CallOptions::default().with_cancellation(token),
        ..GraphSearchOptions::default()
    };
    let error = memory
        .search(&scope, "Alice", &search_options)
        .await
        .expect_err("cancelled search");
    assert!(matches!(error, Error::Cancelled), "got {error:?}");
}

#[tokio::test]
async fn embeddings_are_only_computed_when_configured() {
    let plain = memory(GraphMemoryConfig::default());
    let scope = scope();
    plain
        .add_extracted(&scope, &entities(&["Alice"]))
        .await
        .expect("add");
    let stored = plain
        .entity_named(&scope, "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert_eq!(stored.embedding(), None);

    let embedded =
        memory(GraphMemoryConfig::default()).with_embedding(Arc::new(HashEmbedding::new(16)));
    embedded
        .add_extracted(&scope, &entities(&["Acme"]))
        .await
        .expect("add");
    let stored = embedded
        .entity_named(&scope, "Acme")
        .await
        .expect("lookup")
        .expect("acme");
    assert_eq!(stored.embedding().expect("embedding").len(), 16);
}

/// A chain `Root -> Hop1 -> ... -> Hop<hops>` in one scope.
async fn store_hops(memory: &GraphMemory, scope: &MemoryScope, hops: usize) {
    let mut entities = vec![entity("Root", "Concept", None)];
    let mut relations = Vec::new();
    let mut previous = "Root".to_owned();
    for hop in 1..=hops {
        let name = format!("Hop{hop}");
        entities.push(entity(&name, "Concept", None));
        relations.push(relation(&previous, &name, "next"));
        previous = name;
    }
    memory
        .add_extracted(
            scope,
            &Extraction {
                entities,
                relations,
                degraded: None,
            },
        )
        .await
        .expect("chain");
}

/// A hub plus `leaves` one-hop neighbors.
async fn store_hub(memory: &GraphMemory, scope: &MemoryScope, leaves: usize) {
    let mut entities = vec![entity("Hub", "Concept", None)];
    let mut relations = Vec::new();
    for leaf in 0..leaves {
        let name = format!("Leaf{leaf:03}");
        entities.push(entity(&name, "Concept", None));
        relations.push(relation("Hub", &name, "owns"));
    }
    memory
        .add_extracted(
            scope,
            &Extraction {
                entities,
                relations,
                degraded: None,
            },
        )
        .await
        .expect("hub");
}

#[tokio::test]
async fn per_call_depth_is_clamped_to_the_validated_bound() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    store_hops(&memory, &scope, 8).await;
    let options = GraphSearchOptions {
        top_k: 0,
        threshold: 0.01,
        // Both are far past the validated bounds; the search must clamp them.
        max_depth: 20,
        max_neighbors: 10_000,
        include_relations: true,
        call: CallOptions::default(),
    };
    let hits = memory
        .search(&scope, "Root", &options)
        .await
        .expect("clamped search");
    let depths: Vec<usize> = hits.iter().map(|hit| hit.depth).collect();
    assert!(
        depths.iter().all(|depth| *depth <= 4),
        "depth 20 is clamped to 4: {depths:?}"
    );
    assert_eq!(
        depths.iter().max(),
        Some(&4),
        "the clamp still expands four hops"
    );
    assert_eq!(hits.len(), 5, "Root plus four hops, not the whole chain");
}

#[tokio::test]
async fn per_call_neighbors_are_clamped_to_the_validated_bound() {
    let memory = memory(GraphMemoryConfig::default());
    let scope = scope();
    store_hub(&memory, &scope, 300).await;
    let options = GraphSearchOptions {
        top_k: 0,
        threshold: 0.01,
        max_depth: 1,
        max_neighbors: 10_000,
        include_relations: true,
        call: CallOptions::default(),
    };
    let hits = memory
        .search(&scope, "Hub", &options)
        .await
        .expect("clamped search");
    assert_eq!(
        hits.len(),
        1 + 256,
        "10_000 neighbors per parent is clamped to the documented 256"
    );
    assert!(hits.iter().all(|hit| hit.depth <= 1));
}
