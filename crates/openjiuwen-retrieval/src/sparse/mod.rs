//! Sparse (lexical) retrieval: the additive [`SparseVector`] representation and
//! a self-contained BM25 lexical index.
//!
//! Python's sparse retriever delegates scoring to Milvus, Chroma, or `PGVector`
//! and therefore ships no lexical index of its own; [`Bm25Index`] is a new Rust
//! capability that keeps sparse retrieval working offline. The retriever and
//! result surface stays Python-shaped: it produces
//! [`SearchResult`](crate::common::SearchResult) values exactly like the vector
//! stores do.

pub mod bm25;

pub use bm25::{tokenize, Bm25Config, Bm25Index, Bm25Query, Bm25Statistics, DEFAULT_STOPWORDS};

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An additive sparse vector: unique terms mapped to non-zero weights.
///
/// Entries are kept sorted by term so iteration, equality, and serialization are
/// deterministic regardless of insertion order. The JSON shape is a plain object
/// `{"term": weight, ...}`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseVector {
    /// Sorted `(term, weight)` entries with unique terms.
    entries: Vec<(String, f32)>,
}

impl SparseVector {
    /// Creates an empty sparse vector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a sparse vector from `(term, weight)` pairs.
    ///
    /// Duplicate terms are summed (the representation is additive) and terms
    /// whose accumulated weight is zero are dropped, so equivalent inputs always
    /// produce the same entries.
    pub fn from_weights<I, S>(weights: I) -> Self
    where
        I: IntoIterator<Item = (S, f32)>,
        S: Into<String>,
    {
        let mut merged: BTreeMap<String, f32> = BTreeMap::new();
        for (term, weight) in weights {
            *merged.entry(term.into()).or_insert(0.0) += weight;
        }
        let entries = merged
            .into_iter()
            .filter(|(_, weight)| weight.abs() > 0.0)
            .collect();
        Self { entries }
    }

    /// The sorted `(term, weight)` entries.
    pub fn entries(&self) -> &[(String, f32)] {
        &self.entries
    }

    /// The weight of `term`, or `None` when the term is absent.
    pub fn get(&self, term: &str) -> Option<f32> {
        self.entries
            .binary_search_by(|(candidate, _)| candidate.as_str().cmp(term))
            .ok()
            .map(|index| self.entries[index].1)
    }

    /// Number of stored terms.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vector holds no terms.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dot product with `other`; terms present in only one vector contribute `0`.
    pub fn dot(&self, other: &Self) -> f32 {
        let mut total = 0.0f32;
        let (mut left, mut right) = (0usize, 0usize);
        while left < self.entries.len() && right < other.entries.len() {
            match self.entries[left].0.cmp(&other.entries[right].0) {
                Ordering::Less => left += 1,
                Ordering::Greater => right += 1,
                Ordering::Equal => {
                    total += self.entries[left].1 * other.entries[right].1;
                    left += 1;
                    right += 1;
                }
            }
        }
        total
    }

    /// Euclidean norm of the weights.
    pub fn l2_norm(&self) -> f32 {
        self.entries
            .iter()
            .map(|(_, weight)| weight * weight)
            .sum::<f32>()
            .sqrt()
    }

    /// The `n` highest-weight terms, descending by weight and then by term.
    ///
    /// `n == 0` yields an empty vector.
    pub fn top_terms(&self, n: usize) -> Vec<(String, f32)> {
        let mut sorted = self.entries.clone();
        sorted.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        sorted.truncate(n);
        sorted
    }
}

impl Serialize for SparseVector {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (term, weight) in &self.entries {
            map.serialize_entry(term, weight)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for SparseVector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let weights = BTreeMap::<String, f32>::deserialize(deserializer)?;
        Ok(Self::from_weights(weights))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_vector_is_empty() {
        let vector = SparseVector::new();
        assert!(vector.is_empty());
        assert!(vector.entries().is_empty());
        assert_eq!(vector.l2_norm(), 0.0);
    }

    #[test]
    fn from_weights_sorts_merges_and_drops_zero_weights() {
        let vector = SparseVector::from_weights([
            ("rust", 2.0f32),
            ("search", 1.0),
            ("rust", 0.5),
            ("dropped", 1.0),
            ("dropped", -1.0),
        ]);
        assert_eq!(
            vector.entries(),
            &[("rust".to_owned(), 2.5), ("search".to_owned(), 1.0)]
        );
        assert_eq!(vector.get("rust"), Some(2.5));
        assert_eq!(vector.get("search"), Some(1.0));
        assert_eq!(vector.get("missing"), None);
    }

    #[test]
    fn dot_and_norm_follow_the_additive_representation() {
        let left = SparseVector::from_weights([("a", 2.0f32), ("b", 1.0)]);
        let right = SparseVector::from_weights([("b", 3.0f32), ("c", 4.0)]);
        assert_eq!(left.dot(&right), 3.0);
        assert_eq!(left.dot(&left), 5.0);
        assert_eq!(right.l2_norm(), 5.0);
        assert_eq!(left.dot(&SparseVector::new()), 0.0);
    }

    #[test]
    fn top_terms_orders_by_weight_then_term() {
        let vector =
            SparseVector::from_weights([("b", 1.0f32), ("a", 1.0), ("c", 2.5), ("d", -3.0)]);
        assert_eq!(
            vector.top_terms(2),
            vec![("c".to_owned(), 2.5), ("a".to_owned(), 1.0)]
        );
        assert_eq!(vector.top_terms(4).len(), 4);
        assert!(vector.top_terms(0).is_empty());
    }

    #[test]
    fn serde_uses_a_sorted_term_map() {
        let vector = SparseVector::from_weights([("b", 2.0f32), ("a", 1.0)]);
        let value = serde_json::to_value(&vector).unwrap();
        assert_eq!(value, json!({"a": 1.0, "b": 2.0}));
        let restored: SparseVector = serde_json::from_value(value).unwrap();
        assert_eq!(restored, vector);
        assert_eq!(restored.entries().first().unwrap().0, "a");
    }
}
