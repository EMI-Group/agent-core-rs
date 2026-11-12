#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for triple extraction and the in-memory graph store.
//!
//! Python's only extractor is LLM-based, so the rule extractor and the
//! [`GraphStore`] seam are Rust additions; these tests assert their documented,
//! deterministic behavior instead of porting a Python test file. Everything runs
//! offline: no model, no network, no wall-clock input.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::graph::{
    Entity, GraphStore, InMemoryGraphStore, RuleExtractionConfig, RuleTripleExtractor, Triple,
    TripleBeam, TripleExtractor, TripleMemory, TripleQuery,
};
use openjiuwen_retrieval::{Document, Metadata, MetadataFilter, SearchResult, TextChunk};
use serde_json::{json, Value};

/// Builds a chunk with an explicit document id and chunk id.
fn chunk(doc_id: &str, chunk_id: &str, text: &str) -> TextChunk {
    TextChunk::from_document(
        &Document::with_id(doc_id, text),
        text,
        Some(chunk_id.to_owned()),
    )
}

/// Renders triples as `(subject, predicate, object)` tuples for exact assertions.
fn tuples(triples: &[Triple]) -> Vec<(&str, &str, &str)> {
    triples
        .iter()
        .map(|triple| {
            (
                triple.subject.as_str(),
                triple.predicate.as_str(),
                triple.object.as_str(),
            )
        })
        .collect()
}

/// Reads a numeric metadata entry.
fn confidence(triple: &Triple) -> f64 {
    triple
        .metadata
        .get("confidence")
        .and_then(Value::as_f64)
        .expect("confidence metadata")
}

/// Reads a string metadata entry.
fn metadata_str<'a>(triple: &'a Triple, key: &str) -> &'a str {
    triple
        .metadata
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing metadata field {key}"))
}

/// A triple with empty metadata, for store tests.
fn triple(subject: &str, predicate: &str, object: &str) -> Triple {
    Triple::new(subject, predicate, object)
}

#[test]
fn extracts_copula_definitions() {
    let extractor = RuleTripleExtractor::new();
    let texts = "Acme Corporation is a technology company. Rust is a programming language.";
    let triples = extractor.extract_chunks(&[chunk("doc-1", "chunk-1", texts)]);
    assert_eq!(
        tuples(&triples),
        vec![
            ("Acme Corporation", "is_a", "technology company"),
            ("Rust", "is_a", "programming language"),
        ]
    );
}

#[test]
fn extracts_possession_and_containment() {
    let extractor = RuleTripleExtractor::new();
    let texts = "Acme has 500 employees. The toolbox contains a hammer.";
    let triples = extractor.extract_chunks(&[chunk("doc-1", "chunk-1", texts)]);
    assert_eq!(
        tuples(&triples),
        vec![
            ("Acme", "has", "500 employees"),
            ("toolbox", "contains", "hammer"),
        ]
    );
}

#[test]
fn extracts_inclusion_lists() {
    let extractor = RuleTripleExtractor::new();
    let texts = "Programming languages include Rust, Go and Python.\n\
                 Frameworks such as React and Vue are popular.";
    let triples = extractor.extract_chunks(&[chunk("doc-1", "chunk-1", texts)]);
    assert_eq!(
        tuples(&triples),
        vec![
            ("Programming languages", "includes", "Rust"),
            ("Programming languages", "includes", "Go"),
            ("Programming languages", "includes", "Python"),
            ("Frameworks", "includes", "React"),
            ("Frameworks", "includes", "Vue"),
        ]
    );
}

#[test]
fn extracts_creation_organization_and_usage() {
    let extractor = RuleTripleExtractor::new();
    let texts = "Acme developed the Rust SDK. Acme acquired Beta Corp. \
                 Gamma was founded by Delta. The service uses PostgreSQL. Acme with Rust.";
    let triples = extractor.extract_chunks(&[chunk("doc-1", "chunk-1", texts)]);
    assert_eq!(
        tuples(&triples),
        vec![
            ("Acme", "created", "Rust SDK"),
            ("Acme", "acquired", "Beta Corp"),
            ("Gamma", "founded_by", "Delta"),
            ("service", "uses", "PostgreSQL"),
            ("Acme", "with", "Rust"),
        ]
    );
}

#[test]
fn extracts_glossary_form() {
    let extractor = RuleTripleExtractor::new();
    let triples = extractor.extract_chunks(&[chunk(
        "doc-1",
        "chunk-1",
        "Hypertext: text with links to other documents.",
    )]);
    assert_eq!(
        tuples(&triples),
        vec![(
            "Hypertext",
            "describes",
            "text with links to other documents"
        )]
    );

    // Sentence-introducing labels are not glossary terms.
    let triples = extractor.extract_chunks(&[chunk("doc-1", "chunk-1", "Note: Acme uses Rust.")]);
    assert_eq!(tuples(&triples), vec![("Acme", "uses", "Rust")]);
}

#[test]
fn triple_metadata_records_provenance_and_confidence() {
    let extractor = RuleTripleExtractor::new();
    let triples = extractor.extract_chunks(&[chunk(
        "doc-42",
        "chunk-7",
        "Acme is a company. Beta is a Company.",
    )]);
    assert_eq!(triples.len(), 2);
    for triple in &triples {
        assert_eq!(metadata_str(triple, "doc_id"), "doc-42");
        assert_eq!(metadata_str(triple, "chunk_id"), "chunk-7");
        let confidence = confidence(triple);
        assert!((0.0..=1.0).contains(&confidence), "confidence {confidence}");
    }
    // A capitalised object is stronger evidence than a lowercase one.
    assert!(confidence(&triples[1]) > confidence(&triples[0]));
}

#[test]
fn deduplicates_and_caps_triples_per_chunk() {
    let extractor =
        RuleTripleExtractor::with_config(RuleExtractionConfig::new().with_max_triples_per_chunk(2))
            .unwrap();
    let texts = "Acme is a company. Acme is a Company. Acme has Rust. Acme uses Go.";
    let triples = extractor.extract_chunks(&[chunk("doc-1", "chunk-1", texts)]);
    // The duplicate is dropped and the cap stops the third and fourth sentences.
    assert_eq!(
        tuples(&triples),
        vec![("Acme", "is_a", "company"), ("Acme", "has", "Rust")]
    );

    let unlimited = RuleTripleExtractor::new();
    let texts = "Acme is a company. acme is a Company. Acme has Rust.";
    let triples = unlimited.extract_chunks(&[chunk("doc-1", "chunk-1", texts)]);
    assert_eq!(
        tuples(&triples),
        vec![("Acme", "is_a", "company"), ("Acme", "has", "Rust")]
    );
}

#[test]
fn handles_cjk_entities() {
    let extractor = RuleTripleExtractor::new();
    let triples =
        extractor.extract_chunks(&[chunk("doc-1", "chunk-1", "北京大学 is a university.")]);
    assert_eq!(tuples(&triples), vec![("北京大学", "is_a", "university")]);

    let entities = extractor.entities_in_text("杭州，西湖。");
    assert_eq!(
        entities
            .iter()
            .map(|entity| (entity.name.as_str(), entity.kind.as_deref()))
            .collect::<Vec<_>>(),
        vec![("杭州", Some("cjk")), ("西湖", Some("cjk"))]
    );

    // A CJK run of a single character is below the entity length floor.
    assert_eq!(extractor.entities_in_text("京"), Vec::<Entity>::new());

    // Bare CJK mentions become provenance anchors only when opted in.
    let anchored = RuleTripleExtractor::with_config(
        RuleExtractionConfig::new().with_include_single_entity(true),
    )
    .unwrap();
    let triples = anchored.extract_chunks(&[chunk("doc-1", "chunk-1", "杭州")]);
    assert_eq!(tuples(&triples), vec![("杭州", "appears_in", "doc-1")]);
}

#[tokio::test]
async fn empty_and_whitespace_chunks_yield_no_triples() {
    let extractor = RuleTripleExtractor::new();
    assert_eq!(extractor.extract(&[]).await.unwrap(), Vec::<Triple>::new());
    assert_eq!(
        extractor
            .extract(&[chunk("doc-1", "chunk-1", "   \n\t  ")])
            .await
            .unwrap(),
        Vec::<Triple>::new()
    );
    assert_eq!(extractor.entities_in_text(""), Vec::<Entity>::new());
}

#[test]
fn extraction_is_deterministic_across_chunks() {
    let extractor = RuleTripleExtractor::new();
    let chunks = [
        chunk("doc-1", "chunk-1", "Acme is a company."),
        chunk("doc-2", "chunk-2", "Beta is a company."),
        chunk("doc-3", "chunk-3", "Acme is a company."),
    ];
    let first = extractor.extract_chunks(&chunks);
    let second = extractor.extract_chunks(&chunks);
    assert_eq!(first, second);
    // Chunk-major order: the same triple in two documents is kept twice.
    assert_eq!(
        tuples(&first),
        vec![
            ("Acme", "is_a", "company"),
            ("Beta", "is_a", "company"),
            ("Acme", "is_a", "company"),
        ]
    );
    assert_eq!(metadata_str(&first[2], "doc_id"), "doc-3");
}

#[test]
fn triple_memory_formats_lowercase_and_dedupes() {
    let mut memory = TripleMemory::new();
    assert!(memory.extend(Triple::new("Acme", "is_a", "Company")));
    assert!(!memory.extend(Triple::new("ACME", "IS_A", "company")));
    assert_eq!(
        memory.batch_extend(vec![Triple::new("Acme", "has", "Rust")]),
        1
    );
    assert_eq!(memory.triples_str(), "(acme is_a company)\n(acme has rust)");
    assert!(memory.contains(&Triple::new("acme", "has", "RUST")));
    assert_eq!(memory.len(), 2);
}

#[test]
fn triple_beam_wraps_search_results() {
    let nodes = vec![
        SearchResult {
            id: "t1".to_owned(),
            text: "acme is_a company".to_owned(),
            score: 0.8,
            metadata: Metadata::new(),
        },
        SearchResult {
            id: "t2".to_owned(),
            text: "acme has rust".to_owned(),
            score: 0.7,
            metadata: Metadata::new(),
        },
    ];
    let beam = TripleBeam::new(nodes.clone(), 0.9);
    assert_eq!(beam.triples().len(), 2);
    assert_eq!(beam.score, 0.9);
    assert!(beam.contains_text("acme has rust"));
    assert_eq!(beam.texts().count(), 2);

    let from_pair: TripleBeam = (nodes, 0.5).into();
    assert_eq!(from_pair.len(), 2);
    let single: TripleBeam = SearchResult {
        id: "t3".to_owned(),
        text: "beta is_a company".to_owned(),
        score: 0.4,
        metadata: Metadata::new(),
    }
    .into();
    assert_eq!(single.score, 0.4);
    assert_eq!(TripleBeam::default(), TripleBeam::new(Vec::new(), 0.0));
}

#[test]
fn triple_round_trips_with_python_field_names() {
    let triple = Triple::new("Acme", "is_a", "company").metadata_entry("confidence", json!(0.85));
    let value = serde_json::to_value(&triple).unwrap();
    assert_eq!(
        value,
        json!({
            "subject": "Acme",
            "predicate": "is_a",
            "object": "company",
            "metadata": {"confidence": 0.85}
        })
    );
    let restored: Triple = serde_json::from_value(value).unwrap();
    assert_eq!(restored, triple);

    // Unknown fields are tolerated and missing ones default.
    let parsed: Triple = serde_json::from_value(json!({"subject": "X", "unknown": 1})).unwrap();
    assert_eq!(parsed.object, "");
    assert!(parsed.metadata.is_empty());
    assert!(parsed.has_blank_component());
    assert_eq!(parsed.text(), "X  ");
    assert_eq!(parsed.normalized_key(), "x||");
}

#[tokio::test]
async fn store_traverses_neighbours_by_hop() {
    let store = InMemoryGraphStore::new();
    store
        .add_triples(vec![
            triple("Acme", "is_a", "Company"),
            triple("Company", "has", "Employees"),
            triple("Employees", "uses", "Laptops"),
            triple("Other", "is_a", "Company"),
        ])
        .await
        .unwrap();

    let hop0 = store.neighbors("Acme", 0).await.unwrap();
    assert_eq!(tuples(&hop0), vec![("Acme", "is_a", "Company")]);

    let hop1 = store.neighbors("Acme", 1).await.unwrap();
    assert_eq!(
        tuples(&hop1),
        vec![
            ("Acme", "is_a", "Company"),
            ("Company", "has", "Employees"),
            ("Other", "is_a", "Company"),
        ]
    );

    let hop2 = store.neighbors("Acme", 2).await.unwrap();
    assert_eq!(
        tuples(&hop2),
        vec![
            ("Acme", "is_a", "Company"),
            ("Company", "has", "Employees"),
            ("Other", "is_a", "Company"),
            ("Employees", "uses", "Laptops"),
        ]
    );

    // The traversal is bounded: more hops repeat nothing.
    assert_eq!(store.neighbors("Acme", 99).await.unwrap().len(), 4);
    assert_eq!(
        store.neighbors("Acme", 0).await.unwrap(),
        store.triples_for_entity("Acme").await.unwrap()
    );
}

#[tokio::test]
async fn store_matches_entities_case_insensitively() {
    let store = InMemoryGraphStore::new();
    store
        .add_triples(vec![
            triple("Acme", "is_a", "Company"),
            triple("Company", "has", "Employees"),
        ])
        .await
        .unwrap();

    let hits = store.triples_for_entity("  acme ").await.unwrap();
    assert_eq!(tuples(&hits), vec![("Acme", "is_a", "Company")]);

    let hits = store.triples_for_entity("COMPANY").await.unwrap();
    assert_eq!(
        tuples(&hits),
        vec![("Acme", "is_a", "Company"), ("Company", "has", "Employees")]
    );

    assert_eq!(
        store.triples_for_entity("missing").await.unwrap(),
        Vec::<Triple>::new()
    );
    assert_eq!(
        store.triples_for_entity("  ").await.unwrap(),
        Vec::<Triple>::new()
    );
}

#[tokio::test]
async fn store_queries_filter_by_component_entity_metadata_and_limit() {
    let store = InMemoryGraphStore::new();
    store
        .add_triples(vec![
            triple("Acme", "is_a", "Company").metadata_entry("doc_id", json!("d1")),
            triple("Beta", "is_a", "Company").metadata_entry("doc_id", json!("d2")),
            triple("Acme", "has", "Rust"),
        ])
        .await
        .unwrap();

    let hits = store
        .search(&TripleQuery::new().with_predicate("IS_A"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);

    let hits = store
        .search(
            &TripleQuery::new()
                .with_subject("acme")
                .with_object("company"),
        )
        .await
        .unwrap();
    assert_eq!(tuples(&hits), vec![("Acme", "is_a", "Company")]);

    let hits = store
        .search(&TripleQuery::new().with_entity("ACME"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);

    let hits = store
        .search(&TripleQuery::new().with_metadata_filter(MetadataFilter::eq("doc_id", "d2")))
        .await
        .unwrap();
    assert_eq!(tuples(&hits), vec![("Beta", "is_a", "Company")]);

    let hits = store
        .search(&TripleQuery::new().with_limit(1))
        .await
        .unwrap();
    assert_eq!(tuples(&hits), vec![("Acme", "is_a", "Company")]);

    // A blank component is treated as absent, and `limit: 0` returns nothing.
    let hits = store
        .search(&TripleQuery::new().with_subject("  ").with_limit(0))
        .await
        .unwrap();
    assert_eq!(hits, Vec::<Triple>::new());

    // Serialized queries round-trip (Python's dictionary-shaped request).
    let query = TripleQuery::new()
        .with_subject("Acme")
        .with_limit(2)
        .with_metadata_filter(MetadataFilter::eq("doc_id", "d1"));
    let restored: TripleQuery =
        serde_json::from_value(serde_json::to_value(&query).unwrap()).unwrap();
    assert_eq!(restored, query);
}

#[tokio::test]
async fn store_tracks_statistics_and_clears() {
    let store = InMemoryGraphStore::new();
    assert!(store.is_empty().await.unwrap());
    let added = store
        .add_triples(vec![
            triple(" Acme ", "is_a", "Company "),
            triple("acme", "IS_A", "company"),
            triple("Acme", "has", "Rust"),
        ])
        .await
        .unwrap();
    assert_eq!(added, 2);
    assert_eq!(store.len().await.unwrap(), 2);

    let statistics = store.statistics().await.unwrap();
    assert_eq!(statistics.triple_count, 2);
    assert_eq!(statistics.entity_count, 3);
    assert_eq!(statistics.predicate_count, 2);

    let stored = store.all_triples().await.unwrap();
    assert_eq!(
        tuples(&stored),
        vec![("Acme", "is_a", "Company"), ("Acme", "has", "Rust")]
    );

    store.clear().await.unwrap();
    assert!(store.is_empty().await.unwrap());
    assert_eq!(
        store.triples_for_entity("Acme").await.unwrap(),
        Vec::<Triple>::new()
    );
}

#[tokio::test]
async fn store_rejects_blank_components_without_partial_writes() {
    let store = InMemoryGraphStore::new();
    let error = store
        .add_triples(vec![
            triple("Acme", "is_a", "Company"),
            triple(" ", "has", "Rust"),
        ])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(store.is_empty().await.unwrap());

    assert!(InMemoryGraphStore::with_triples(vec![triple("Acme", "", "Company")]).is_err());
}

#[test]
fn extractor_configuration_guards_extraction() {
    let only_copula =
        RuleTripleExtractor::with_config(RuleExtractionConfig::new().with_predicates(["is_a"]))
            .unwrap();
    let triples = only_copula.extract_chunks(&[chunk(
        "doc-1",
        "chunk-1",
        "Acme is a company. Acme has Rust.",
    )]);
    assert_eq!(tuples(&triples), vec![("Acme", "is_a", "company")]);

    assert!(RuleTripleExtractor::with_config(
        RuleExtractionConfig::new().with_max_triples_per_chunk(0)
    )
    .is_err());
}
