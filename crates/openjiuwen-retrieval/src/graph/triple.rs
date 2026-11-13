//! Triple and entity data models plus the `TripleMemory`/`TripleBeam` helpers.
//!
//! Port of the Python `openjiuwen/core/retrieval/common/triple.py`,
//! `triple_memory.py`, and `triple_beam.py` modules. A [`Triple`] keeps Python's
//! field names (`subject`, `predicate`, `object`, `metadata`); Python stores plain
//! tuples in `TripleMemory` and `RetrievalResult`s in `TripleBeam`, so the Rust
//! versions hold [`Triple`] and [`SearchResult`] values with the same
//! deduplication and formatting rules.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::common::{Metadata, SearchResult};

/// Normalizes one triple component: trims surrounding whitespace and lowercases.
///
/// Used for [`Triple::normalized_key`], triple-memory deduplication, and the
/// graph store's case-insensitive entity index.
pub(crate) fn normalize_component(value: &str) -> String {
    value.trim().to_lowercase()
}

/// A subject–predicate–object assertion with provenance metadata.
///
/// Python's pydantic model requires the three strings and defaults `metadata` to
/// `{}`; the Rust model defaults every missing field so partially populated
/// payloads still deserialize, and tolerates unknown fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    /// Subject of the assertion.
    #[serde(default)]
    pub subject: String,
    /// Relationship between subject and object.
    #[serde(default)]
    pub predicate: String,
    /// Object of the assertion.
    #[serde(default)]
    pub object: String,
    /// Provenance metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
}

impl Triple {
    /// Builds a triple with empty metadata.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            metadata: Metadata::new(),
        }
    }

    /// Replaces the metadata wholesale.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Inserts a single metadata entry, keeping any existing keys.
    #[must_use]
    pub fn metadata_entry(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// The triple rendered as `"subject predicate object"`.
    ///
    /// This is the text Python's beam search embeds and scores.
    pub fn text(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.object)
    }

    /// Case-insensitive deduplication key: trimmed, lowercased `subject|predicate|object`.
    ///
    /// The `|` separator cannot appear in a component that survived extraction
    /// trimming, so distinct triples map to distinct keys.
    pub fn normalized_key(&self) -> String {
        format!(
            "{}|{}|{}",
            normalize_component(&self.subject),
            normalize_component(&self.predicate),
            normalize_component(&self.object)
        )
    }

    /// The subject and object, in that order.
    pub fn entities(&self) -> [&str; 2] {
        [self.subject.as_str(), self.object.as_str()]
    }

    /// Whether any of the three components is empty or whitespace only.
    pub fn has_blank_component(&self) -> bool {
        [&self.subject, &self.predicate, &self.object]
            .iter()
            .any(|component| component.trim().is_empty())
    }
}

/// A named node extracted alongside triples.
///
/// Python has no entity model (the LLM extractor only emits triples); this type
/// exists so rule-based extraction and graph traversal can describe the nodes
/// they discover.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Entity surface form.
    #[serde(default)]
    pub name: String,
    /// Optional entity kind (for example `"capitalised"` or `"cjk"`).
    #[serde(default)]
    pub kind: Option<String>,
    /// Entity metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
}

impl Entity {
    /// Builds an entity without a kind or metadata.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: None,
            metadata: Metadata::new(),
        }
    }

    /// Sets the entity kind.
    #[must_use]
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// Replaces the metadata wholesale.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Case-insensitive entity name used for matching and deduplication.
    pub fn normalized_name(&self) -> String {
        normalize_component(&self.name)
    }
}

impl From<&Triple> for Triple {
    fn from(triple: &Triple) -> Self {
        triple.clone()
    }
}

/// Insertion-ordered, deduplicated collection of triples.
///
/// Mirrors Python's `TripleMemory`: a triple is remembered once, keyed by its
/// lowercased, trimmed component tuple.
#[derive(Debug, Clone, Default)]
pub struct TripleMemory {
    included: BTreeSet<String>,
    memory: Vec<Triple>,
}

impl TripleMemory {
    /// Creates an empty memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remembers one triple; returns whether it was new.
    ///
    /// Accepts owned values and references so callers can reuse their triples.
    pub fn extend(&mut self, triple: impl Into<Triple>) -> bool {
        let triple = triple.into();
        let key = triple.normalized_key();
        if !self.included.insert(key) {
            return false;
        }
        self.memory.push(triple);
        true
    }

    /// Remembers every triple in `triples`; returns how many were new.
    pub fn batch_extend<I, T>(&mut self, triples: I) -> usize
    where
        I: IntoIterator<Item = T>,
        T: Into<Triple>,
    {
        let mut added = 0;
        for triple in triples {
            if self.extend(triple) {
                added += 1;
            }
        }
        added
    }

    /// The remembered triples in insertion order.
    pub fn triples(&self) -> &[Triple] {
        &self.memory
    }

    /// The triples rendered as newline-joined, lowercased `"(s p o)"` lines.
    ///
    /// This is Python's `triples_str`, lowercased so prompt text is stable
    /// regardless of the source casing.
    pub fn triples_str(&self) -> String {
        self.memory
            .iter()
            .map(|triple| {
                format!(
                    "({} {} {})",
                    normalize_component(&triple.subject),
                    normalize_component(&triple.predicate),
                    normalize_component(&triple.object)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Whether the triple (by normalized key) is already remembered.
    pub fn contains(&self, triple: &Triple) -> bool {
        self.included.contains(&triple.normalized_key())
    }

    /// Number of remembered triples.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Whether no triple is remembered.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Forgets every triple.
    pub fn clear(&mut self) {
        self.included.clear();
        self.memory.clear();
    }
}

/// One beam of [`SearchResult`] triples plus its path score.
///
/// Mirrors Python's `TripleBeam`, which wraps `List[RetrievalResult]` and the
/// accumulated score. The Rust beam holds [`SearchResult`]s, the result type
/// exposed by this crate's vector stores.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TripleBeam {
    /// Triples in this beam, in path order.
    pub nodes: Vec<SearchResult>,
    /// Accumulated beam score.
    pub score: f32,
}

impl TripleBeam {
    /// Builds a beam from its nodes and score.
    pub fn new(nodes: Vec<SearchResult>, score: f32) -> Self {
        Self { nodes, score }
    }

    /// The beam's triples.
    pub fn triples(&self) -> &[SearchResult] {
        &self.nodes
    }

    /// The beam's triples as texts, in path order.
    pub fn texts(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|node| node.text.as_str())
    }

    /// Whether the beam already holds a triple with `text` (Python's `in`).
    pub fn contains_text(&self, text: &str) -> bool {
        self.nodes.iter().any(|node| node.text == text)
    }

    /// Number of triples in the beam.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the beam holds no triples.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl From<(Vec<SearchResult>, f32)> for TripleBeam {
    fn from((nodes, score): (Vec<SearchResult>, f32)) -> Self {
        Self::new(nodes, score)
    }
}

impl From<SearchResult> for TripleBeam {
    fn from(node: SearchResult) -> Self {
        let score = node.score;
        Self::new(vec![node], score)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn result(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: text.to_owned(),
            score,
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn triple_serializes_python_field_names() {
        let triple = Triple::new("Acme", "is_a", "company")
            .metadata_entry("doc_id", json!("doc-1"))
            .metadata_entry("confidence", json!(0.85));
        let value = serde_json::to_value(&triple).unwrap();
        assert_eq!(
            value,
            json!({
                "subject": "Acme",
                "predicate": "is_a",
                "object": "company",
                "metadata": {"doc_id": "doc-1", "confidence": 0.85}
            })
        );
        let restored: Triple = serde_json::from_value(value).unwrap();
        assert_eq!(restored, triple);
    }

    #[test]
    fn triple_defaults_missing_fields_and_ignores_unknown_ones() {
        let parsed: Triple =
            serde_json::from_value(json!({"subject": "Acme", "extra": [1, 2]})).unwrap();
        assert_eq!(parsed.subject, "Acme");
        assert_eq!(parsed.predicate, "");
        assert_eq!(parsed.object, "");
        assert!(parsed.metadata.is_empty());
        assert!(parsed.has_blank_component());

        let bare: Triple = serde_json::from_value(json!({})).unwrap();
        assert_eq!(bare, Triple::default());
    }

    #[test]
    fn triple_text_and_normalized_key_ignore_case_and_spacing() {
        let triple = Triple::new("  Acme ", " Is_A ", "Company");
        assert_eq!(triple.text(), "  Acme   Is_A  Company");
        assert_eq!(triple.normalized_key(), "acme|is_a|company");
        assert_eq!(triple.entities(), ["  Acme ", "Company"]);
        assert!(!triple.has_blank_component());
    }

    #[test]
    fn entity_normalizes_and_keeps_metadata() {
        let entity = Entity::new("Acme Corp")
            .with_kind("capitalised")
            .with_metadata(Metadata::from_iter([("source".to_owned(), json!("chunk"))]));
        assert_eq!(entity.normalized_name(), "acme corp");
        assert_eq!(entity.kind.as_deref(), Some("capitalised"));
        let value = serde_json::to_value(&entity).unwrap();
        assert_eq!(
            value,
            json!({
                "name": "Acme Corp",
                "kind": "capitalised",
                "metadata": {"source": "chunk"}
            })
        );
    }

    #[test]
    fn memory_dedupes_by_normalized_key_and_formats_lowercase() {
        let mut memory = TripleMemory::new();
        assert!(memory.is_empty());
        assert!(memory.extend(Triple::new("Acme", "is_a", "Company")));
        assert!(!memory.extend(Triple::new(" acme ", "IS_A", "company")));
        assert_eq!(memory.len(), 1);
        assert!(memory.contains(&Triple::new("ACME", "is_a", "company")));
        assert_eq!(memory.triples_str(), "(acme is_a company)");

        assert_eq!(
            memory.batch_extend([
                Triple::new("Acme", "has", "500 employees"),
                Triple::new("acme", "is_a", "company"),
            ]),
            1
        );
        assert_eq!(
            memory.triples_str(),
            "(acme is_a company)\n(acme has 500 employees)"
        );

        let borrowed = Triple::new("Acme", "uses", "Rust");
        assert!(memory.extend(&borrowed));
        assert_eq!(memory.len(), 3);
        memory.clear();
        assert!(memory.is_empty());
        assert_eq!(memory.triples_str(), "");
    }

    #[test]
    fn memory_accepts_borrowed_triples() {
        let triple = Triple::new("A", "b", "C");
        let mut memory = TripleMemory::new();
        assert_eq!(memory.batch_extend([&triple, &triple]), 1);
        assert_eq!(memory.triples(), &[triple]);
    }

    #[test]
    fn beam_exposes_nodes_score_and_containment() {
        let beam = TripleBeam::default();
        assert!(beam.is_empty());
        assert_eq!(beam.score, 0.0);
        assert_eq!(beam.triples(), &[]);

        let beam = TripleBeam::from((
            vec![result("t1", "a b c", 0.5), result("t2", "d e f", 0.25)],
            0.75,
        ));
        assert_eq!(beam.len(), 2);
        assert_eq!(beam.score, 0.75);
        assert_eq!(beam.texts().collect::<Vec<_>>(), vec!["a b c", "d e f"]);
        assert!(beam.contains_text("a b c"));
        assert!(!beam.contains_text("a b d"));

        let single = TripleBeam::from(result("t3", "g h i", 0.9));
        assert_eq!(single.len(), 1);
        assert_eq!(single.score, 0.9);
        assert_eq!(single.triples()[0].id, "t3");
    }
}
