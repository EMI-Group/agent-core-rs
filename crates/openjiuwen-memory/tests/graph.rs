#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration coverage for graph memory **through the [`LongTermMemory`] facade**.
//!
//! `tests/graph_memory.rs` exercises [`openjiuwen_memory::graph::GraphMemory`] directly;
//! this suite targets the facade wiring instead: `ltm.graph()` (built only when the
//! config enables graph memory and the stores carry a graph backend), `search_all`, and
//! the facade-level extraction/persistence/recall behavior callers actually see.
//!
//! Every test is deterministic and offline: heuristic extraction plus
//! [`ScriptedModel`](openjiuwen_test_support::ScriptedModel) stand in for the LLM, and
//! all stores are ephemeral.
//! Text is written with a leading clause because the heuristic extractor deliberately
//! ignores sentence-initial capitalized runs (they carry no signal), so
//! `"I saw Alice works at Acme, and RustLang uses Cargo."` yields the four entities and
//! the two relations the Python heuristic path would surface.
use std::cmp::Ordering;
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_foundation::store::{InMemoryDocumentStore, InMemoryKvStore, InMemoryVectorStore};
use openjiuwen_memory::config::MemoryEngineConfig;
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::graph::{
    EntityExtractor, ExtractedEntity, Extraction, GraphAddOptions, GraphMemoryStats,
    GraphSearchOptions, HeuristicExtractor,
};
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryScope, NewMemory, SearchMode, SearchOptions,
};
use openjiuwen_memory::store::MemoryStores;
use openjiuwen_test_support::ScriptedModel;

/// A text the heuristic extractor turns into four entities and two relations.
const RELATION_TEXT: &str = "I saw Alice works at Acme, and RustLang uses Cargo.";

/// The scope shared by most tests.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// The default config with graph memory enabled.
fn graph_config() -> MemoryEngineConfig {
    let mut config = MemoryEngineConfig::default();
    config.graph.enabled = true;
    config
}

/// Builds a facade over ephemeral stores, which include an in-memory graph store.
async fn facade(config: MemoryEngineConfig) -> Arc<LongTermMemory> {
    LongTermMemory::with_stores(config, MemoryStores::in_memory())
        .await
        .expect("facade")
}

/// Builds a graph-enabled facade with `model` attached as the extraction LLM.
fn facade_with_model(config: MemoryEngineConfig, model: ScriptedModel) -> Arc<LongTermMemory> {
    LongTermMemory::builder(config)
        .stores(MemoryStores::in_memory())
        .model(Arc::new(model))
        .build()
        .expect("facade")
}

/// Runs the heuristic extractor the facade would use with `config`, for comparison.
async fn heuristic_extraction(config: &MemoryEngineConfig) -> Extraction {
    HeuristicExtractor::new(&config.graph)
        .extract(RELATION_TEXT, &scope(), &CallOptions::new())
        .await
        .expect("heuristic extraction")
}

/// Persists the shared relation text through the facade's graph memory.
async fn seed_graph(ltm: &Arc<LongTermMemory>) {
    ltm.graph()
        .expect("graph enabled")
        .add(&scope(), RELATION_TEXT, &GraphAddOptions::default())
        .await
        .expect("graph add");
}

#[tokio::test]
async fn heuristic_extraction_runs_through_the_facade() {
    let ltm = facade(graph_config()).await;
    let graph = ltm.graph().expect("graph enabled");

    let extraction = graph
        .add(&scope(), RELATION_TEXT, &GraphAddOptions::default())
        .await
        .expect("add");
    assert_eq!(
        extraction.entities.len(),
        4,
        "heuristics must find the four named entities"
    );
    assert_eq!(extraction.relations.len(), 2, "and the two verb relations");

    // Without a model attached, `add` must use exactly the heuristic extractor.
    let expected = heuristic_extraction(ltm.config()).await;
    assert_eq!(extraction, expected, "the default extractor is heuristic");
    assert_ne!(expected.entities.len(), 0);

    let entities = graph.entities(&scope(), 0).await.expect("entities");
    let mut names: Vec<&str> = entities.iter().map(|item| item.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["Acme", "Alice", "Cargo", "RustLang"]);

    let alice = graph
        .entity_named(&scope(), "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert_eq!(alice.entity_type, "Person");
    assert_eq!(alice.scope, scope());

    assert!(graph
        .entity_named(&scope(), "  ALICE ")
        .await
        .expect("normalized lookup")
        .is_some());

    assert_eq!(
        graph.stats(&scope()).await.expect("stats"),
        GraphMemoryStats {
            entities: 4,
            relations: 2,
        }
    );
}

#[tokio::test]
async fn graph_search_returns_the_seed_and_expands_neighbors() {
    let ltm = facade(graph_config()).await;
    let graph = ltm.graph().expect("graph enabled");
    seed_graph(&ltm).await;

    let options = GraphSearchOptions {
        top_k: 5,
        max_depth: 2,
        max_neighbors: 8,
        ..Default::default()
    };
    let hits = graph
        .search(&scope(), "Alice", &options)
        .await
        .expect("search");
    assert_eq!(hits.len(), 2, "the seed plus its direct neighbor");
    assert_eq!(hits[0].entity.name, "Alice");
    assert_eq!(hits[0].depth, 0, "a seed is scored directly");
    assert!(hits[0].score > 0.0);
    assert!(
        hits.iter().skip(1).any(|hit| hit.depth > 0),
        "at least one expanded neighbor must have depth > 0"
    );
    assert_eq!(hits[1].entity.name, "Acme");
    assert_eq!(hits[1].depth, 1);

    // Ordering invariant: score descending, then depth ascending.
    for pair in hits.windows(2) {
        let (left, right) = (&pair[0], &pair[1]);
        let by_score = right.score.total_cmp(&left.score);
        assert!(
            by_score == Ordering::Less
                || (by_score == Ordering::Equal && left.depth <= right.depth),
            "hits must be ordered by (score desc, depth asc): {left:?} vs {right:?}"
        );
    }
}

#[tokio::test]
async fn scripted_model_extraction_persists_entities_and_relations() {
    let model = ScriptedModel::builder()
        .respond_text(
            r#"{"entities":[{"name":"Alice","type":"Person"},{"name":"RustLang","type":"Concept"}],"relations":[{"from":"Alice","to":"RustLang","type":"uses"}]}"#,
        )
        .build();
    let handle = model.clone();
    let ltm = facade_with_model(graph_config(), model);
    let graph = ltm.graph().expect("graph enabled");

    let extraction = graph
        .add(&scope(), "Alice uses RustLang", &GraphAddOptions::default())
        .await
        .expect("add");
    assert_eq!(handle.call_count(), 1, "one model call, no retries needed");
    assert_eq!(extraction.entities.len(), 2);
    assert_eq!(extraction.entities[0].name, "Alice");

    let alice = graph
        .entity_named(&scope(), "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert_eq!(
        alice.entity_type, "Person",
        "the LLM-supplied type must be persisted, not a heuristic guess"
    );

    let relations = graph.relations(&scope()).await.expect("relations");
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].from, "Alice");
    assert_eq!(relations[0].to, "RustLang");
    assert_eq!(relations[0].relation_type, "uses");

    assert_eq!(
        graph.stats(&scope()).await.expect("stats"),
        GraphMemoryStats {
            entities: 2,
            relations: 1,
        }
    );
}

#[tokio::test]
async fn malformed_model_output_falls_back_to_heuristics() {
    let model = ScriptedModel::builder()
        .respond_text("not json at all")
        .build();
    let handle = model.clone();
    let ltm = facade_with_model(graph_config(), model);
    let graph = ltm.graph().expect("graph enabled");

    // The extractor retries, exhausts its budget, and degrades to heuristics; a parse
    // failure must never propagate out of `add`.
    let extraction = graph
        .add(&scope(), RELATION_TEXT, &GraphAddOptions::default())
        .await
        .expect("add must succeed despite the unusable model output");
    assert!(handle.call_count() > 1, "the extractor retried the model");

    let expected = heuristic_extraction(ltm.config()).await;
    assert!(!expected.is_degraded(), "a plain heuristic pass is clean");
    assert_eq!(
        extraction.entities, expected.entities,
        "the failed LLM pass must yield the heuristic entities"
    );
    assert_eq!(
        extraction.relations, expected.relations,
        "the failed LLM pass must yield the heuristic relations"
    );
    assert!(
        extraction.is_degraded(),
        "the fallback must be visible on the returned extraction"
    );
    assert!(
        extraction
            .degraded_reason()
            .is_some_and(|reason| reason.contains("attempt(s)")),
        "the reason must name the exhausted attempt count: {extraction:?}"
    );
    assert_ne!(extraction.entities.len(), 0);
    assert_ne!(extraction.relations.len(), 0);
    assert_eq!(
        graph.stats(&scope()).await.expect("stats").entities,
        extraction.entities.len()
    );
}

#[tokio::test]
async fn relations_are_listed_and_point_at_existing_entities() {
    let ltm = facade(graph_config()).await;
    let graph = ltm.graph().expect("graph enabled");
    seed_graph(&ltm).await;

    let relations = graph.relations(&scope()).await.expect("relations");
    assert_eq!(relations.len(), 2);

    let works_at = relations
        .iter()
        .find(|relation| relation.relation_type == "works_at")
        .expect("Alice works_at Acme");
    assert_eq!(works_at.from, "Alice");
    assert_eq!(works_at.to, "Acme");
    assert!(
        graph
            .entity_named(&scope(), &works_at.from)
            .await
            .expect("from lookup")
            .is_some(),
        "the relation source must be a stored entity"
    );
    assert!(
        graph
            .entity_named(&scope(), &works_at.to)
            .await
            .expect("to lookup")
            .is_some(),
        "the relation target must be a stored entity"
    );

    let uses = relations
        .iter()
        .find(|relation| relation.relation_type == "uses")
        .expect("RustLang uses Cargo");
    assert_eq!(
        (uses.from.as_str(), uses.to.as_str()),
        ("RustLang", "Cargo")
    );
}

#[tokio::test]
async fn scopes_never_see_each_others_graph() {
    let ltm = facade(graph_config()).await;
    let graph = ltm.graph().expect("graph enabled");
    let alice = MemoryScope::new("alice", "ws");
    let bob = MemoryScope::new("bob", "ws");

    graph
        .add(&alice, RELATION_TEXT, &GraphAddOptions::default())
        .await
        .expect("alice add");
    graph
        .add(
            &bob,
            "I saw Carol works at Globex.",
            &GraphAddOptions::default(),
        )
        .await
        .expect("bob add");

    let alice_names: Vec<String> = graph
        .entities(&alice, 0)
        .await
        .expect("alice entities")
        .into_iter()
        .map(|entity| entity.name)
        .collect();
    let bob_names: Vec<String> = graph
        .entities(&bob, 0)
        .await
        .expect("bob entities")
        .into_iter()
        .map(|entity| entity.name)
        .collect();
    assert_eq!(alice_names.len(), 4);
    assert_eq!(bob_names.len(), 2);
    assert!(!alice_names.iter().any(|name| name == "Carol"));
    assert!(!bob_names.iter().any(|name| name == "Alice"));

    assert_eq!(
        graph.stats(&alice).await.expect("alice stats"),
        GraphMemoryStats {
            entities: 4,
            relations: 2,
        }
    );
    assert_eq!(
        graph.stats(&bob).await.expect("bob stats"),
        GraphMemoryStats {
            entities: 2,
            relations: 1,
        }
    );

    let alice_carol = graph
        .search(&alice, "Carol", &GraphSearchOptions::default())
        .await
        .expect("alice search");
    assert!(alice_carol.is_empty(), "Carol belongs to Bob's graph");

    let bob_carol = graph
        .search(&bob, "Carol", &GraphSearchOptions::default())
        .await
        .expect("bob search");
    assert_ne!(bob_carol.len(), 0);
    assert_eq!(bob_carol[0].entity.name, "Carol");
    assert_eq!(bob_carol[0].entity.scope.user_id, "bob");
}

#[tokio::test]
async fn delete_clear_and_stats_go_through_the_facade() {
    let ltm = facade(graph_config()).await;
    let graph = ltm.graph().expect("graph enabled");
    let scope = scope();
    graph
        .add(
            &scope,
            "I saw Alice works at Acme.",
            &GraphAddOptions::default(),
        )
        .await
        .expect("add");

    let alice = graph
        .entity_named(&scope, "Alice")
        .await
        .expect("lookup")
        .expect("alice");
    assert!(graph
        .delete_entity(&scope, &alice.id)
        .await
        .expect("first delete"));
    assert!(
        !graph
            .delete_entity(&scope, &alice.id)
            .await
            .expect("second delete"),
        "deleting a missing node returns false"
    );
    assert!(graph
        .entity_named(&scope, "Alice")
        .await
        .expect("lookup")
        .is_none());
    assert!(!graph
        .entities(&scope, 0)
        .await
        .expect("entities")
        .iter()
        .any(|entity| entity.name == "Alice"));
    assert!(
        graph
            .entity_named(&scope, "Acme")
            .await
            .expect("lookup")
            .is_some(),
        "the other entity survives"
    );
    assert_eq!(
        graph.stats(&scope).await.expect("stats"),
        GraphMemoryStats {
            entities: 1,
            relations: 0,
        }
    );

    assert_eq!(graph.clear(&scope).await.expect("clear"), 1);
    assert_eq!(
        graph.stats(&scope).await.expect("stats"),
        GraphMemoryStats::default()
    );
    assert_eq!(graph.entities(&scope, 0).await.expect("entities").len(), 0);
    assert_eq!(graph.relations(&scope).await.expect("relations").len(), 0);

    let invalid = MemoryScope::new("alice", "bad/scope");
    let error = graph.stats(&invalid).await.expect_err("invalid scope");
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
}

#[tokio::test]
async fn search_all_merges_memory_hits_and_graph_entities() {
    let ltm = facade(graph_config()).await;
    let scope = scope();
    ltm.add(
        &scope,
        NewMemory::semantic("alice prefers dark mode"),
        &AddOptions::default(),
    )
    .await
    .expect("memory add");

    // `ExtractedEntity` carries the exact name the query matches; the heuristic
    // extractor would type a lowercase multi-word name differently.
    let extraction = Extraction {
        entities: vec![ExtractedEntity {
            name: "dark mode preferences".to_owned(),
            entity_type: "Concept".to_owned(),
            importance: None,
        }],
        relations: Vec::new(),
        degraded: None,
    };
    ltm.graph()
        .expect("graph enabled")
        .add_extracted(&scope, &extraction)
        .await
        .expect("graph add");

    let combined = ltm
        .search_all(&scope, "dark mode", &SearchOptions::default())
        .await
        .expect("search all");
    assert_eq!(combined.memories.len(), 1);
    assert_eq!(
        combined.memories[0].record.content,
        "alice prefers dark mode"
    );
    assert_eq!(combined.entities.len(), 1);
    assert_eq!(combined.entities[0].entity.name, "dark mode preferences");
    assert_eq!(combined.entities[0].depth, 0);
    assert!(combined.entities[0].score > 0.0);
}

#[tokio::test]
async fn search_all_degrades_without_failing() {
    // (a) Graph enabled but the store bundle has no graph backend: `graph()` is `None`
    // and `search_all` returns memory hits only.
    let stores = MemoryStores::from_parts(
        Arc::new(InMemoryKvStore::new()),
        Arc::new(InMemoryVectorStore::new()),
        Arc::new(InMemoryDocumentStore::new()),
    );
    let ltm = LongTermMemory::builder(graph_config())
        .stores(stores)
        .build()
        .expect("facade");
    assert!(ltm.graph().is_none(), "no graph store, no graph memory");
    ltm.add(
        &scope(),
        NewMemory::semantic("alice prefers dark mode"),
        &AddOptions::default(),
    )
    .await
    .expect("memory add");
    let combined = ltm
        .search_all(&scope(), "dark mode", &SearchOptions::default())
        .await
        .expect("search all");
    assert_eq!(combined.memories.len(), 1);
    assert_eq!(combined.entities.len(), 0);
    assert_eq!(
        combined.graph_error, None,
        "no graph store is not a graph failure"
    );
    assert!(!combined.is_partial());

    // (b) Graph wired and populated, but a narrow threshold excludes the graph entity
    // while memory recall still succeeds: graph recall returns no entities and reports
    // no failure, so "no match" stays distinguishable from "could not ask".
    let ltm = facade(graph_config()).await;
    ltm.add(
        &scope(),
        NewMemory::semantic("alice prefers dark mode").with_importance(1.0),
        &AddOptions::default(),
    )
    .await
    .expect("memory add");
    let extraction = Extraction {
        entities: vec![ExtractedEntity {
            name: "dark mode preferences".to_owned(),
            entity_type: "Concept".to_owned(),
            importance: None,
        }],
        relations: Vec::new(),
        degraded: None,
    };
    ltm.graph()
        .expect("graph enabled")
        .add_extracted(&scope(), &extraction)
        .await
        .expect("graph add");

    // The graph entity scores exactly 0.3 (keyword coverage only), the memory record
    // scores above 0.35 with importance 1.0 and an immediate timestamp.
    let narrow = SearchOptions::default().with_threshold(0.34);
    let combined = ltm
        .search_all(&scope(), "dark mode", &narrow)
        .await
        .expect("memory recall still succeeds");
    assert_eq!(combined.memories.len(), 1);
    assert!(
        combined.entities.is_empty(),
        "graph recall must degrade to empty, not fail"
    );
    assert_eq!(
        combined.graph_error, None,
        "a threshold miss is not a graph failure"
    );

    // Defaults propagate as documented: graph knobs defer to the config and expand
    // neighbors by default; search options keep the engine defaults.
    let graph_defaults = GraphSearchOptions::default();
    assert_eq!(graph_defaults.top_k, 0);
    assert_eq!(graph_defaults.max_depth, 0);
    assert_eq!(graph_defaults.max_neighbors, 0);
    assert!(graph_defaults.threshold.abs() < f32::EPSILON);
    assert!(
        graph_defaults.include_relations,
        "default graph search expands neighbors"
    );
    assert!(graph_defaults.call.timeout.is_none());

    let search_defaults = SearchOptions::default();
    assert_eq!(search_defaults.top_k, 5);
    assert!((search_defaults.threshold - 0.3).abs() < f32::EPSILON);
    assert_eq!(search_defaults.mode, SearchMode::Hybrid);
    assert_eq!(search_defaults.kinds.len(), 0);
    assert!(search_defaults.token_budget.is_none());
}

#[tokio::test]
async fn search_all_reports_a_graph_failure_instead_of_an_empty_entity_list() {
    /// Indexes documents normally but cannot embed a query, so graph writes succeed and
    /// only graph recall fails.
    struct QuerylessEmbedding;

    #[async_trait::async_trait]
    impl openjiuwen_retrieval::embedding::Embedding for QuerylessEmbedding {
        fn model_name(&self) -> &'static str {
            "queryless"
        }

        async fn embed_query(&self, _text: &str) -> openjiuwen_common::error::Result<Vec<f32>> {
            Err(openjiuwen_common::error::Error::retrieval(
                "query embedding backend unavailable",
            ))
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> openjiuwen_common::error::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.5_f32, 0.5]).collect())
        }
    }

    let ltm = LongTermMemory::builder(graph_config())
        .stores(MemoryStores::in_memory())
        .embedding(Arc::new(QuerylessEmbedding))
        .build()
        .expect("facade");
    ltm.add(
        &scope(),
        NewMemory::semantic("alice prefers dark mode"),
        &AddOptions::default().skip_embedding(),
    )
    .await
    .expect("memory add");
    // Graph writes embed documents (which this provider serves), so the graph is
    // populated and only recall fails.
    ltm.graph()
        .expect("graph enabled")
        .add(&scope(), RELATION_TEXT, &GraphAddOptions::default())
        .await
        .expect("graph add");

    // Memory recall is keyword-only here, so the graph's query embedding is the only
    // failing step.
    let options = SearchOptions::default()
        .with_threshold(0.0)
        .with_mode(SearchMode::Keyword);
    let combined = ltm
        .search_all(&scope(), "dark mode", &options)
        .await
        .expect("a graph failure must not fail the recall");
    assert_eq!(
        combined.memories.len(),
        1,
        "memory hits survive a graph failure"
    );
    assert_eq!(combined.entities.len(), 0);
    let reason = combined
        .graph_error
        .as_deref()
        .expect("the graph failure must be reported, not hidden behind an empty list");
    assert!(
        reason.contains("query embedding backend unavailable"),
        "the reason must carry the provider error: {reason}"
    );
    assert!(combined.is_partial());
}
