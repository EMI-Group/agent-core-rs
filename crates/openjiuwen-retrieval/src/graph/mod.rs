//! Triple extraction and graph storage over retrieval chunks.
//!
//! Port of the Python `openjiuwen/core/retrieval/common/triple*.py` models and
//! the `indexing/processor/extractor/` interface, plus a graph-store seam that
//! Python does not have (it stores triples in a vector index instead).
//!
//! Three layers:
//! - [`triple`] — the [`Triple`], [`Entity`], [`TripleMemory`], and [`TripleBeam`]
//!   data models;
//! - [`extract`] — the [`TripleExtractor`] trait and the deterministic,
//!   model-free [`RuleTripleExtractor`];
//! - [`store`] — the [`GraphStore`] trait and the adjacency-indexed
//!   [`InMemoryGraphStore`].

mod text;

pub mod extract;
pub mod store;
pub mod triple;

pub use extract::{
    RuleExtractionConfig, RuleTripleExtractor, TripleExtractor, BUILT_IN_PREDICATES,
    DEFAULT_MAX_ENTITY_CHARS, DEFAULT_MAX_TRIPLES_PER_CHUNK, DEFAULT_MIN_ENTITY_CHARS,
};
pub use store::{GraphStatistics, GraphStore, InMemoryGraphStore, TripleQuery};
pub use triple::{Entity, Triple, TripleBeam, TripleMemory};
