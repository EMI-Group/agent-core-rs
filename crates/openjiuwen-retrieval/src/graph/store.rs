//! The [`GraphStore`] seam and its in-memory, adjacency-indexed implementation.
//!
//! Python has no graph store: it embeds triples as text into a vector index and
//! walks them with beam search. Rust adds a real store so triples can be indexed
//! by entity and traversed by hops; this is a documented new capability. The
//! in-memory implementation is the offline default, and the trait is the seam for
//! external graph databases.

use std::collections::{BTreeMap, HashSet};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::triple::{normalize_component, Triple};
use crate::common::MetadataFilter;

/// A triple lookup request.
///
/// `subject`, `predicate`, and `object` are exact case-insensitive matches on the
/// trimmed component; `entity` matches either the subject or the object; a blank
/// string is treated like `None`. `metadata_filter` is evaluated against the
/// triple metadata, and `limit` truncates the result after filtering, preserving
/// store order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TripleQuery {
    /// Exact subject match.
    #[serde(default)]
    pub subject: Option<String>,
    /// Exact predicate match.
    #[serde(default)]
    pub predicate: Option<String>,
    /// Exact object match.
    #[serde(default)]
    pub object: Option<String>,
    /// Match when either the subject or the object equals this entity.
    #[serde(default)]
    pub entity: Option<String>,
    /// Metadata predicate.
    #[serde(default)]
    pub metadata_filter: Option<MetadataFilter>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl TripleQuery {
    /// Creates a query that matches every triple.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the exact subject.
    #[must_use]
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    /// Sets the exact predicate.
    #[must_use]
    pub fn with_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate = Some(predicate.into());
        self
    }

    /// Sets the exact object.
    #[must_use]
    pub fn with_object(mut self, object: impl Into<String>) -> Self {
        self.object = Some(object.into());
        self
    }

    /// Sets the entity that must match the subject or the object.
    #[must_use]
    pub fn with_entity(mut self, entity: impl Into<String>) -> Self {
        self.entity = Some(entity.into());
        self
    }

    /// Sets the metadata predicate.
    #[must_use]
    pub fn with_metadata_filter(mut self, filter: MetadataFilter) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Sets the maximum number of results.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Whether `triple` satisfies every populated constraint.
    pub fn matches(&self, triple: &Triple) -> bool {
        if !component_matches(self.subject.as_deref(), &triple.subject)
            || !component_matches(self.predicate.as_deref(), &triple.predicate)
            || !component_matches(self.object.as_deref(), &triple.object)
        {
            return false;
        }
        let entity_matches = match populated(self.entity.as_deref()) {
            None => true,
            Some(entity) => {
                let entity = normalize_component(entity);
                normalize_component(&triple.subject) == entity
                    || normalize_component(&triple.object) == entity
            }
        };
        if !entity_matches {
            return false;
        }
        self.metadata_filter
            .as_ref()
            .is_none_or(|filter| filter.matches(&triple.metadata))
    }
}

/// Counts describing a [`GraphStore`]'s contents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Number of distinct stored triples.
    #[serde(default)]
    pub triple_count: usize,
    /// Number of distinct entity names across subjects and objects.
    #[serde(default)]
    pub entity_count: usize,
    /// Number of distinct predicates.
    #[serde(default)]
    pub predicate_count: usize,
}

/// Backend-agnostic triple store with entity traversal.
///
/// Implementations must not panic on unknown entities, empty inputs, or blank
/// components; they report typed errors instead.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Stores triples and returns how many were new.
    ///
    /// Duplicates (by [`Triple::normalized_key`]) are stored once and not
    /// counted again. Components are trimmed before they are stored.
    async fn add_triples(&self, triples: Vec<Triple>) -> Result<usize>;

    /// Triples whose subject or object equals `entity` (case-insensitive).
    async fn triples_for_entity(&self, entity: &str) -> Result<Vec<Triple>>;

    /// Triples within `hops` expansion steps of `entity`.
    ///
    /// `hops == 0` returns the direct triples ([`Self::triples_for_entity`]);
    /// each further hop adds the triples of every entity discovered so far.
    /// The walk is a breadth-first search in store order, deduplicated by
    /// normalized key.
    async fn neighbors(&self, entity: &str, hops: usize) -> Result<Vec<Triple>>;

    /// Triples matching `query`, in store order.
    async fn search(&self, query: &TripleQuery) -> Result<Vec<Triple>>;

    /// Every stored triple, in insertion order.
    async fn all_triples(&self) -> Result<Vec<Triple>>;

    /// Number of stored triples.
    async fn len(&self) -> Result<usize>;

    /// Whether the store holds no triples.
    async fn is_empty(&self) -> Result<bool>;

    /// Removes every triple.
    async fn clear(&self) -> Result<()>;

    /// Triple, entity, and predicate counts.
    async fn statistics(&self) -> Result<GraphStatistics>;
}

/// Normalizes an optional filter value; blank strings count as absent.
fn populated(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Exact case-insensitive match of `expected` against `actual`.
fn component_matches(expected: Option<&str>, actual: &str) -> bool {
    populated(expected)
        .is_none_or(|expected| normalize_component(expected) == normalize_component(actual))
}

/// The in-memory index behind [`InMemoryGraphStore`].
#[derive(Debug, Default)]
struct GraphIndex {
    /// Triples in insertion order, keyed by normalized key.
    triples: IndexMap<String, Triple>,
    /// Entity name (normalized) to the keys of its triples, in insertion order.
    entities: BTreeMap<String, Vec<String>>,
    /// Predicate (normalized) to the number of triples using it.
    predicates: BTreeMap<String, usize>,
}

impl GraphIndex {
    /// Stores one triple; returns whether it was new.
    fn insert(&mut self, triple: Triple) -> bool {
        let key = triple.normalized_key();
        if self.triples.contains_key(&key) {
            return false;
        }
        let subject = normalize_component(&triple.subject);
        let object = normalize_component(&triple.object);
        self.entities
            .entry(subject.clone())
            .or_default()
            .push(key.clone());
        if object != subject {
            self.entities.entry(object).or_default().push(key.clone());
        }
        *self
            .predicates
            .entry(normalize_component(&triple.predicate))
            .or_insert(0) += 1;
        self.triples.insert(key, triple);
        true
    }

    /// Keys of the triples incident to `entity`.
    fn entity_keys(&self, entity: &str) -> &[String] {
        match self.entities.get(entity) {
            Some(keys) => keys.as_slice(),
            None => &[],
        }
    }

    /// Triples incident to `entity`, in insertion order.
    fn incident(&self, entity: &str) -> Vec<Triple> {
        self.entity_keys(entity)
            .iter()
            .filter_map(|key| self.triples.get(key).cloned())
            .collect()
    }
}

/// Adjacency-indexed in-memory [`GraphStore`].
///
/// Triples keep their insertion order, which defines the order of every read.
/// Duplicate triples are stored once, and each triple is reachable from both its
/// subject and its object.
#[derive(Debug, Default)]
pub struct InMemoryGraphStore {
    index: RwLock<GraphIndex>,
}

impl InMemoryGraphStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a store pre-populated with `triples`.
    ///
    /// # Errors
    ///
    /// Same as [`GraphStore::add_triples`].
    pub fn with_triples(triples: Vec<Triple>) -> Result<Self> {
        let store = Self::new();
        store.insert_validated(triples)?;
        Ok(store)
    }

    /// Trims components, validates them, and inserts the batch.
    fn insert_validated(&self, triples: Vec<Triple>) -> Result<usize> {
        let triples = triples.into_iter().map(trimmed).collect::<Vec<_>>();
        for triple in &triples {
            if triple.has_blank_component() {
                return Err(Error::validation(
                    "triples require non-blank subject, predicate, and object",
                ));
            }
        }
        let mut index = self.index.write();
        let mut added = 0;
        for triple in triples {
            if index.insert(triple) {
                added += 1;
            }
        }
        Ok(added)
    }
}

/// Trims the three components of a triple, keeping its metadata.
fn trimmed(mut triple: Triple) -> Triple {
    triple.subject = triple.subject.trim().to_owned();
    triple.predicate = triple.predicate.trim().to_owned();
    triple.object = triple.object.trim().to_owned();
    triple
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn add_triples(&self, triples: Vec<Triple>) -> Result<usize> {
        self.insert_validated(triples)
    }

    async fn triples_for_entity(&self, entity: &str) -> Result<Vec<Triple>> {
        let entity = normalize_component(entity);
        if entity.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self.index.read().incident(&entity))
    }

    async fn neighbors(&self, entity: &str, hops: usize) -> Result<Vec<Triple>> {
        let start = normalize_component(entity);
        if start.is_empty() {
            return Ok(Vec::new());
        }
        let index = self.index.read();
        let mut triples: Vec<Triple> = Vec::new();
        let mut seen_keys: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::from([start.clone()]);
        let mut frontier = vec![start];
        let mut level = 0;
        while !frontier.is_empty() && level <= hops {
            let mut next: Vec<String> = Vec::new();
            for name in &frontier {
                for key in index.entity_keys(name) {
                    if !seen_keys.insert(key.clone()) {
                        continue;
                    }
                    let Some(triple) = index.triples.get(key) else {
                        continue;
                    };
                    for candidate in triple.entities() {
                        let candidate = normalize_component(candidate);
                        if visited.insert(candidate.clone()) {
                            next.push(candidate);
                        }
                    }
                    triples.push(triple.clone());
                }
            }
            frontier = next;
            level += 1;
        }
        Ok(triples)
    }

    async fn search(&self, query: &TripleQuery) -> Result<Vec<Triple>> {
        let limit = query.limit.unwrap_or(usize::MAX);
        let index = self.index.read();
        Ok(index
            .triples
            .values()
            .filter(|triple| query.matches(triple))
            .take(limit)
            .cloned()
            .collect())
    }

    async fn all_triples(&self) -> Result<Vec<Triple>> {
        Ok(self.index.read().triples.values().cloned().collect())
    }

    async fn len(&self) -> Result<usize> {
        Ok(self.index.read().triples.len())
    }

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.index.read().triples.is_empty())
    }

    async fn clear(&self) -> Result<()> {
        *self.index.write() = GraphIndex::default();
        Ok(())
    }

    async fn statistics(&self) -> Result<GraphStatistics> {
        let index = self.index.read();
        Ok(GraphStatistics {
            triple_count: index.triples.len(),
            entity_count: index.entities.len(),
            predicate_count: index.predicates.len(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn triple(subject: &str, predicate: &str, object: &str) -> Triple {
        Triple::new(subject, predicate, object)
    }

    async fn chain_store() -> InMemoryGraphStore {
        let store = InMemoryGraphStore::new();
        let added = store
            .add_triples(vec![
                triple("Acme", "is_a", "Company"),
                triple("Company", "has", "Employees"),
                triple("Employees", "uses", "Laptops"),
            ])
            .await
            .unwrap();
        assert_eq!(added, 3);
        store
    }

    #[tokio::test]
    async fn inserts_trim_dedupe_and_count() {
        let store = InMemoryGraphStore::new();
        assert!(store.is_empty().await.unwrap());
        assert_eq!(store.len().await.unwrap(), 0);

        let added = store
            .add_triples(vec![
                triple(" Acme ", "is_a", "Company "),
                triple("acme", "IS_A", "company"),
            ])
            .await
            .unwrap();
        assert_eq!(added, 1);
        let stored = store.all_triples().await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0], triple("Acme", "is_a", "Company"));
    }

    #[tokio::test]
    async fn blank_components_are_rejected_without_partial_writes() {
        let store = InMemoryGraphStore::new();
        let error = store
            .add_triples(vec![
                triple("Acme", "is_a", "Company"),
                triple(" ", "has", "X"),
            ])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(store.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn entity_lookup_is_case_insensitive_for_subjects_and_objects() {
        let store = chain_store().await;
        let hits = store.triples_for_entity("acme").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "Acme");

        let hits = store.triples_for_entity("  COMPANY  ").await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].object, "Company");
        assert_eq!(hits[1].subject, "Company");

        assert_eq!(
            store.triples_for_entity("Unknown").await.unwrap(),
            Vec::new()
        );
        assert_eq!(store.triples_for_entity("   ").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn neighbors_expand_hop_by_hop() {
        let store = chain_store().await;

        let hop0 = store.neighbors("Acme", 0).await.unwrap();
        assert_eq!(hop0.len(), 1);
        assert_eq!(hop0[0].predicate, "is_a");

        let hop1 = store.neighbors("Acme", 1).await.unwrap();
        assert_eq!(hop1.len(), 2);
        assert_eq!(hop1[1].predicate, "has");

        let hop2 = store.neighbors("Acme", 2).await.unwrap();
        assert_eq!(hop2.len(), 3);
        assert_eq!(hop2[2].object, "Laptops");

        // Extra hops never duplicate and never run past the graph.
        assert_eq!(store.neighbors("Acme", 10).await.unwrap().len(), 3);
        assert_eq!(store.neighbors("Unknown", 2).await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn queries_filter_by_component_entity_metadata_and_limit() {
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
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "Acme");

        let hits = store
            .search(&TripleQuery::new().with_entity("acme"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);

        let hits = store
            .search(&TripleQuery::new().with_metadata_filter(MetadataFilter::eq("doc_id", "d2")))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "Beta");

        let hits = store
            .search(&TripleQuery::new().with_limit(1))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "Acme");

        let hits = store
            .search(&TripleQuery::new().with_object("Rust").with_limit(5))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);

        // Blank strings behave like absent filters.
        let hits = store
            .search(&TripleQuery::new().with_subject("  ").with_limit(0))
            .await
            .unwrap();
        assert_eq!(hits, Vec::new());
    }

    #[tokio::test]
    async fn statistics_and_clear_track_the_index() {
        let store = chain_store().await;
        let statistics = store.statistics().await.unwrap();
        assert_eq!(statistics.triple_count, 3);
        assert_eq!(statistics.entity_count, 4);
        assert_eq!(statistics.predicate_count, 3);

        store.clear().await.unwrap();
        assert!(store.is_empty().await.unwrap());
        assert_eq!(store.statistics().await.unwrap().triple_count, 0);
        assert_eq!(store.triples_for_entity("Acme").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn with_triples_validates_up_front() {
        let store = InMemoryGraphStore::with_triples(vec![triple("A", "b", "C")]).unwrap();
        assert_eq!(store.len().await.unwrap(), 1);
        assert!(InMemoryGraphStore::with_triples(vec![triple("A", "", "C")]).is_err());
    }
}
