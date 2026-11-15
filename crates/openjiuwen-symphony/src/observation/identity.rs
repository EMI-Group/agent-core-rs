//! Point-edge identity and the static graph index an observation is measured against.
//!
//! Ports Python `openjiuwen.symphony.observation.identity` (v0.1.20), the hashing and
//! validation layer the evidence store and the observation service are built on.
//!
//! # Point-edge identity
//!
//! Python's first static edge identity was a *port mapping*: an edge was the pair of
//! `(source_output, target_input)` port names the matcher had picked. That identity cannot
//! carry evidence, because a capability whose body changes while its ports stay the same
//! keeps the same id — earlier success counts would silently leak into the new body. Python
//! therefore switched to a *point edge*: the two endpoint ids, the relation type, **and**
//! the content hash of both endpoints at one static revision
//! ([`EdgeIdentity`]). Editing either endpoint
//! changes the identity, so the rebinding step marks the prior evidence `invalidated`;
//! deleting an endpoint leaves the identity unmatched (`orphaned`).
//!
//! # The static index
//!
//! [`StaticGraphIndex`] is the read-only
//! view one static revision is qualified against: the set of known capability ids, the
//! per-capability graph identity and content hashes, and the derived edge identities with
//! the static edge payload they belong to.
//!
//! Python builds that index from a graph artifact payload carrying `capabilities`,
//! `graph_identity_hashes` and `capability_hashes` maps; this port builds it from the
//! in-process [`CapabilityGraph`] instead, because the Rust
//! artifact's graph document *is* the graph. Two constructors cover the two ways a caller
//! can know a capability's content:
//!
//! - [`StaticGraphIndex::from_graph`](crate::observation::identity::StaticGraphIndex::from_graph)
//!   uses the node metadata's `content_hash` entry when the static artifact published one,
//!   and falls back to the node identity hash (the hash of
//!   `{capability_type, capability_id, label, node_type, metadata}`) when it did not.
//! - [`StaticGraphIndex::from_graph_with_descriptors`](crate::observation::identity::StaticGraphIndex::from_graph_with_descriptors)
//!   additionally hashes the live descriptors (`CapabilityDescriptor::compute_content_hash`),
//!   which wins over the node metadata for every capability it covers.
//!
//! Edges are indexed whether or not both endpoints are known: an edge out of a deleted node
//! still yields an identity (with an empty endpoint hash), and the rebinding step decides
//! whether it is `orphaned`. Unknown endpoints are never added to the capability id set.
//!
//! # Determinism
//!
//! No output depends on hash-map iteration order: ids live in `BTreeSet`/`BTreeMap`s and
//! every hashed document is assembled as a `BTreeMap` before it is rendered, so the same
//! content always produces the same 64-character lowercase hex digest. Nothing here reads
//! the clock, panics, or allocates through a non-deterministic container.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::graph::{CapabilityGraph, CapabilityNode};
use crate::llm::canonical_json;
use crate::model::CapabilityDescriptor;
use crate::observation::contracts::EvolutionGraphEdge;

/// The identity schema stamped into every serialized point-edge identity.
pub const POINT_EDGE_IDENTITY_SCHEMA: &str = "symphony.point-edge.v1";

/// The `identity_schema` JSON key of a serialized point-edge identity.
const IDENTITY_SCHEMA_FIELD: &str = "identity_schema";

/// The capability type assumed when neither the node metadata nor the node type names one.
const DEFAULT_CAPABILITY_TYPE: &str = "skill";

/// The prefix `normalize_capability_id` strips first.
const SKILL_PREFIX: &str = "skill:";

/// The prefix `normalize_capability_id` strips second, after [`SKILL_PREFIX`].
const CAPABILITY_PREFIX: &str = "capability:";

/// The node metadata key holding a capability's published content hash.
const CONTENT_HASH_FIELD: &str = "content_hash";

/// The node metadata key holding a capability's type.
const CAPABILITY_TYPE_FIELD: &str = "capability_type";

/// Returns the SHA-256 hex digest of the canonical JSON of `value`.
///
/// Canonical JSON sorts object keys recursively and emits no whitespace, so two values that
/// differ only in key order hash identically. This is Python's `stable_hash`
/// (`json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))`), rendered
/// by the crate's single canonical-JSON renderer
/// ([`canonical_json`]). Arrays keep their order, scalars keep
/// their lexical form, and containers deeper than the renderer's depth cap render as `null`
/// rather than aborting.
#[must_use]
pub fn stable_hash(value: &Value) -> String {
    sha256_hex(canonical_json(value).as_bytes())
}

/// Returns the SHA-256 hex digest of the UTF-8 bytes of `value`.
///
/// Python's evidence payloads store identifiers (trajectory, session, request ids) only as
/// digests; this is that digest, taken over the raw text rather than over a JSON rendering.
#[must_use]
pub fn private_hash(value: &str) -> String {
    sha256_hex(value.as_bytes())
}

/// Normalizes one capability id: trims, then strips one leading `skill:` and one leading
/// `capability:` prefix.
///
/// Both prefixes occur in practice — a JGF node id is `"{capability_type}:{capability_id}"`,
/// a rail may qualify a skill id with `skill:`, and Python's own helpers strip both. A
/// dangling prefix with nothing after it normalizes to an empty string, and blank input
/// stays blank.
#[must_use]
pub fn normalize_capability_id(raw: &str) -> String {
    let trimmed = raw.trim();
    let without_skill = trimmed.strip_prefix(SKILL_PREFIX).unwrap_or(trimmed);
    let without_capability = without_skill
        .strip_prefix(CAPABILITY_PREFIX)
        .unwrap_or(without_skill);
    without_capability.to_string()
}

/// Builds the lookup key of a capability: the lowercased, trimmed type plus `:` plus the
/// normalized capability id.
///
/// This is the key Python's artifact payloads index their `graph_identity_hashes` and
/// `capability_hashes` maps by, e.g. `"skill:web.search"`.
#[must_use]
pub fn capability_key(capability_type: &str, capability_id: &str) -> String {
    format!(
        "{}:{}",
        capability_type.trim().to_lowercase(),
        normalize_capability_id(capability_id)
    )
}

/// One observed transition bound to both endpoints' content at one static revision.
///
/// The identity is a value object, not a graph edge: it is the key evidence is counted
/// against, so its [`identity_hash`](crate::observation::identity::EdgeIdentity::identity_hash)
/// must change when *anything* that determines the transition's meaning changes — the
/// endpoints, the relation type, or either endpoint's content.
///
/// Deserialization is tolerant (every field defaults, unknown fields are ignored); use
/// [`from_value`](crate::observation::identity::EdgeIdentity::from_value) when the
/// `identity_schema` stamp has to be verified.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EdgeIdentity {
    /// The normalized id of the node the transition leaves.
    pub source_id: String,
    /// The normalized id of the node the transition enters.
    pub target_id: String,
    /// The relation type of the transition, `can_feed` for a data-flow edge.
    pub relation_type: String,
    /// The source's content hash at the pinned static revision.
    pub source_content_hash: String,
    /// The target's content hash at the pinned static revision.
    pub target_content_hash: String,
}

impl EdgeIdentity {
    /// Builds an identity from its five parts.
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        relation_type: impl Into<String>,
        source_content_hash: impl Into<String>,
        target_content_hash: impl Into<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            relation_type: relation_type.into(),
            source_content_hash: source_content_hash.into(),
            target_content_hash: target_content_hash.into(),
        }
    }

    /// The SHA-256 hex digest of [`Self::to_value`], i.e. the key evidence is counted
    /// against.
    #[must_use]
    pub fn identity_hash(&self) -> String {
        stable_hash(&self.to_value())
    }

    /// Renders the identity as the stamped JSON document it is hashed through.
    ///
    /// Keys are `identity_schema`, `source_id`, `target_id`, `relation_type`,
    /// `source_content_hash`, `target_content_hash`, assembled in a `BTreeMap` so the
    /// rendering never depends on map order.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut document = BTreeMap::new();
        document.insert(
            IDENTITY_SCHEMA_FIELD.to_string(),
            Value::String(POINT_EDGE_IDENTITY_SCHEMA.to_string()),
        );
        document.insert(
            "source_id".to_string(),
            Value::String(self.source_id.clone()),
        );
        document.insert(
            "target_id".to_string(),
            Value::String(self.target_id.clone()),
        );
        document.insert(
            "relation_type".to_string(),
            Value::String(self.relation_type.clone()),
        );
        document.insert(
            "source_content_hash".to_string(),
            Value::String(self.source_content_hash.clone()),
        );
        document.insert(
            "target_content_hash".to_string(),
            Value::String(self.target_content_hash.clone()),
        );
        Value::Object(document.into_iter().collect())
    }

    /// Rebuilds an identity from [`Self::to_value`] output.
    ///
    /// Returns [`None`] when the document carries no `identity_schema` string or names a
    /// different schema, so an identity minted under another schema version is never
    /// silently coerced. Absent or non-string identity fields default to empty strings.
    #[must_use]
    pub fn from_value(value: &Value) -> Option<Self> {
        let document = value.as_object()?;
        let schema = document.get(IDENTITY_SCHEMA_FIELD)?.as_str()?;
        if schema != POINT_EDGE_IDENTITY_SCHEMA {
            return None;
        }
        Some(Self {
            source_id: text_field(document, "source_id"),
            target_id: text_field(document, "target_id"),
            relation_type: text_field(document, "relation_type"),
            source_content_hash: text_field(document, "source_content_hash"),
            target_content_hash: text_field(document, "target_content_hash"),
        })
    }
}

/// The identity and schema indexes derived from one static capability graph.
///
/// Built through [`from_graph`](crate::observation::identity::StaticGraphIndex::from_graph)
/// or
/// [`from_graph_with_descriptors`](crate::observation::identity::StaticGraphIndex::from_graph_with_descriptors)
/// and then read-only: the observation service keeps one per static revision, qualifies
/// evidence against it, and rebinds every observed edge to it. Every map is ordered, so two
/// indexes built from the same graph compare and serialize deterministically.
#[derive(Debug, Clone)]
pub struct StaticGraphIndex {
    /// The static revision the index was built for.
    pub revision: String,
    /// The normalized ids of every known capability, i.e. the node ids of the graph.
    pub capability_ids: BTreeSet<String>,
    /// Each capability's graph identity hash, keyed by normalized capability id.
    pub graph_hash_by_id: BTreeMap<String, String>,
    /// Each capability's content hash, keyed by normalized capability id.
    pub content_hash_by_id: BTreeMap<String, String>,
    /// Each static edge's JSON, keyed by point-edge identity hash.
    pub edges_by_identity: BTreeMap<String, Value>,
    /// Each edge's point-edge identity, keyed by its own identity hash.
    pub edge_identities: BTreeMap<String, EdgeIdentity>,
}

impl StaticGraphIndex {
    /// Builds the index from `graph`, taking content hashes from node metadata.
    ///
    /// For every node the normalized id, the capability type (metadata
    /// `capability_type`, else the node type, else `skill`), the graph identity hash, and
    /// the content hash are recorded; for every edge a point-edge identity is derived and
    /// both it and the edge JSON are keyed by its identity hash.
    #[must_use]
    pub fn from_graph(revision: impl Into<String>, graph: &CapabilityGraph) -> Self {
        Self::build(revision.into(), graph, &[])
    }

    /// Builds the index from `graph`, taking content hashes from `descriptors`.
    ///
    /// A descriptor is matched by its normalized qualified id and by its normalized
    /// capability id, and its
    /// [`compute_content_hash`](crate::model::CapabilityDescriptor::compute_content_hash)
    /// wins over the node metadata's `content_hash` entry. Graph identity hashes are
    /// unaffected, so only content-derived edge identities move.
    #[must_use]
    pub fn from_graph_with_descriptors(
        revision: impl Into<String>,
        graph: &CapabilityGraph,
        descriptors: &[CapabilityDescriptor],
    ) -> Self {
        Self::build(revision.into(), graph, descriptors)
    }

    /// Reports whether every id is a known capability of this static revision.
    ///
    /// Each id is normalized first, so `"skill:a"` and `"a"` are the same capability, and
    /// an empty id set is trivially valid.
    #[must_use]
    pub fn validates_nodes<'a>(&self, node_ids: impl IntoIterator<Item = &'a str>) -> bool {
        node_ids
            .into_iter()
            .all(|id| self.capability_ids.contains(&normalize_capability_id(id)))
    }

    /// Builds the index; `descriptors` is empty for the metadata-only constructor.
    fn build(
        revision: String,
        graph: &CapabilityGraph,
        descriptors: &[CapabilityDescriptor],
    ) -> Self {
        let descriptor_hashes = descriptor_content_hashes(descriptors);
        let mut graph_hash_by_id = BTreeMap::new();
        let mut content_hash_by_id = BTreeMap::new();

        for node in graph.nodes.values() {
            let capability_id = normalize_capability_id(&node.id);
            let capability_type = capability_type_of(node);
            let graph_hash = stable_hash(&node_identity_value(node, &capability_type));
            let content_hash = descriptor_hashes
                .get(&capability_id)
                .cloned()
                .or_else(|| metadata_content_hash(node))
                .unwrap_or_else(|| graph_hash.clone());
            graph_hash_by_id.insert(capability_id.clone(), graph_hash);
            content_hash_by_id.insert(capability_id, content_hash);
        }

        let mut edges_by_identity = BTreeMap::new();
        let mut edge_identities = BTreeMap::new();
        for edge in &graph.edges {
            let source_id = normalize_capability_id(&edge.source);
            let target_id = normalize_capability_id(&edge.target);
            // An unknown endpoint keeps an empty hash instead of failing: the rebinding
            // step is what classifies such an edge as `orphaned`.
            let identity = EdgeIdentity::new(
                source_id.clone(),
                target_id.clone(),
                edge.relation.clone(),
                content_hash_by_id
                    .get(&source_id)
                    .cloned()
                    .unwrap_or_default(),
                content_hash_by_id
                    .get(&target_id)
                    .cloned()
                    .unwrap_or_default(),
            );
            let identity_hash = identity.identity_hash();
            // A `CapabilityEdge` only fails to serialize when its weight is non-finite,
            // which `CapabilityGraph::add_edge` already rejects; store `null` if a
            // hand-built edge sneaks through rather than dropping the identity.
            let edge_value = serde_json::to_value(edge).unwrap_or(Value::Null);
            edges_by_identity.insert(identity_hash.clone(), edge_value);
            edge_identities.insert(identity_hash, identity);
        }

        Self {
            revision,
            capability_ids: graph_hash_by_id.keys().cloned().collect(),
            graph_hash_by_id,
            content_hash_by_id,
            edges_by_identity,
            edge_identities,
        }
    }
}

/// Derives the point-edge identity of an observed transition against a static index.
///
/// Both endpoint ids are normalized and looked up in the index's content hashes; a missing
/// or empty hash on either side yields [`None`], because an identity cannot be pinned to
/// content the static revision does not know. This is the only place a
/// [`EvolutionGraphEdge`] is converted
/// into an identity, so the observation service and the overlay always agree on the key.
#[must_use]
pub fn edge_identity_from_observation(
    edge: &EvolutionGraphEdge,
    static_index: &StaticGraphIndex,
) -> Option<EdgeIdentity> {
    let source_id = normalize_capability_id(&edge.source_id);
    let target_id = normalize_capability_id(&edge.target_id);
    let source_content_hash = static_index.content_hash_by_id.get(&source_id)?;
    let target_content_hash = static_index.content_hash_by_id.get(&target_id)?;
    if source_content_hash.is_empty() || target_content_hash.is_empty() {
        return None;
    }
    Some(EdgeIdentity::new(
        source_id,
        target_id,
        edge.relation_type.clone(),
        source_content_hash.clone(),
        target_content_hash.clone(),
    ))
}

/// Returns the lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Reads one string field, defaulting to empty for absent values.
fn text_field(document: &Map<String, Value>, key: &str) -> String {
    document
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Resolves a node's capability type: metadata `capability_type`, else the node type, else
/// the documented `skill` fallback.
fn capability_type_of(node: &CapabilityNode) -> String {
    let from_metadata = node
        .metadata
        .get(CAPABILITY_TYPE_FIELD)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let from_node_type = Some(node.node_type.trim()).filter(|value| !value.is_empty());
    from_metadata
        .or(from_node_type)
        .unwrap_or(DEFAULT_CAPABILITY_TYPE)
        .to_string()
}

/// Reads a node's published content hash from its metadata, when it carries a usable one.
fn metadata_content_hash(node: &CapabilityNode) -> Option<String> {
    node.metadata
        .get(CONTENT_HASH_FIELD)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Renders the document a capability's graph identity hash is taken over.
///
/// The document is `{capability_type, capability_id, label, node_type, metadata}`; the keys
/// are assembled in a `BTreeMap` so [`stable_hash`] hashes the same rendering regardless of
/// how the node was built.
fn node_identity_value(node: &CapabilityNode, capability_type: &str) -> Value {
    let mut document = BTreeMap::new();
    document.insert(
        CAPABILITY_TYPE_FIELD.to_string(),
        Value::String(capability_type.to_string()),
    );
    document.insert(
        "capability_id".to_string(),
        Value::String(normalize_capability_id(&node.id)),
    );
    document.insert("label".to_string(), Value::String(node.label.clone()));
    document.insert(
        "node_type".to_string(),
        Value::String(node.node_type.clone()),
    );
    document.insert(
        "metadata".to_string(),
        Value::Object(node.metadata.clone().into_iter().collect()),
    );
    Value::Object(document.into_iter().collect())
}

/// Maps every descriptor's normalized ids to its content hash.
///
/// Both the qualified id (`tool:search`) and the bare capability id (`search`) are keyed, so
/// a node id that carries or omits the type prefix both resolve. Later descriptors win on a
/// collision, matching the registry's last-write-wins upsert.
fn descriptor_content_hashes(descriptors: &[CapabilityDescriptor]) -> BTreeMap<String, String> {
    let mut hashes = BTreeMap::new();
    for descriptor in descriptors {
        let content_hash = descriptor.compute_content_hash();
        hashes.insert(
            normalize_capability_id(&descriptor.qualified_id()),
            content_hash.clone(),
        );
        hashes.insert(
            normalize_capability_id(&descriptor.capability_id),
            content_hash,
        );
    }
    hashes
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).expect("valid JSON")
    }

    /// A two-node, one-edge graph: `skill:a` feeds `skill:b`.
    fn sample_graph() -> CapabilityGraph {
        let mut graph = CapabilityGraph::new("graph-1");
        graph.upsert_node(CapabilityNode::new("skill:a", "Skill A", "skill"));
        graph.upsert_node(CapabilityNode::new("skill:b", "Skill B", "skill"));
        graph
            .add_edge(crate::graph::CapabilityEdge::can_feed("skill:a", "skill:b"))
            .unwrap();
        graph
    }

    #[test]
    fn stable_hash_ignores_object_key_order() {
        let first = parse(r#"{"b": 1, "a": {"y": 2, "x": [3, {"d": 4, "c": 5}]}}"#);
        let second = parse(r#"{"a": {"x": [3, {"c": 5, "d": 4}], "y": 2}, "b": 1}"#);
        assert_eq!(stable_hash(&first), stable_hash(&second));
        assert_eq!(stable_hash(&first).len(), 64);

        let changed = parse(r#"{"b": 1, "a": {"y": 2, "x": [3, {"d": 4, "c": 6}]}}"#);
        assert_ne!(stable_hash(&first), stable_hash(&changed));

        let null = stable_hash(&Value::Null);
        assert_eq!(null, stable_hash(&Value::Null));
        assert_eq!(null.len(), 64);
        assert!(null.chars().all(|character| character.is_ascii_hexdigit()));
        assert!(null
            .chars()
            .all(|character| !character.is_ascii_uppercase()));
    }

    #[test]
    fn private_hash_is_utf8_sha256() {
        let digest = private_hash("s/1");
        assert_eq!(digest.len(), 64);
        assert_eq!(digest, private_hash("s/1"));
        assert_ne!(digest, private_hash("s/2"));
        // The empty string hashes like any other input instead of panicking.
        assert_eq!(private_hash("").len(), 64);
    }

    #[test]
    fn normalize_capability_id_strips_one_of_each_prefix() {
        assert_eq!(
            normalize_capability_id("  skill:web.search  "),
            "web.search"
        );
        assert_eq!(normalize_capability_id("capability:x"), "x");
        assert_eq!(normalize_capability_id("skill:capability:x"), "x");
        assert_eq!(normalize_capability_id(""), "");
        assert_eq!(normalize_capability_id("   "), "");
        assert_eq!(normalize_capability_id("plain.id"), "plain.id");
    }

    #[test]
    fn capability_key_lowercases_type_and_normalizes_id() {
        assert_eq!(capability_key("Skill", " skill:a "), "skill:a");
        assert_eq!(capability_key(" TOOL ", "capability:b"), "tool:b");
        assert_eq!(capability_key("", "c"), ":c");
    }

    #[test]
    fn edge_identity_round_trips_through_json() {
        let identity = EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b");
        let value = identity.to_value();
        assert_eq!(
            value["identity_schema"],
            Value::String(POINT_EDGE_IDENTITY_SCHEMA.to_string())
        );
        assert_eq!(value["source_id"], Value::String("a".to_string()));
        assert_eq!(EdgeIdentity::from_value(&value), Some(identity.clone()));

        // The derived serde form ignores the schema stamp and still round-trips.
        let rendered = serde_json::to_value(&identity).unwrap();
        assert_eq!(
            rendered["target_content_hash"],
            Value::String("hash-b".to_string())
        );
        assert_eq!(
            serde_json::from_value::<EdgeIdentity>(rendered).unwrap(),
            identity
        );
        assert_eq!(EdgeIdentity::default().source_id, "");
    }

    #[test]
    fn edge_identity_from_value_checks_the_schema_stamp() {
        let identity = EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b");
        let mut other = identity.to_value();
        other[IDENTITY_SCHEMA_FIELD] = Value::String("symphony.point-edge.v0".to_string());
        assert_eq!(EdgeIdentity::from_value(&other), None);

        let mut missing = identity.to_value();
        missing
            .as_object_mut()
            .unwrap()
            .remove(IDENTITY_SCHEMA_FIELD);
        assert_eq!(EdgeIdentity::from_value(&missing), None);
        assert_eq!(EdgeIdentity::from_value(&Value::Null), None);
        assert_eq!(
            EdgeIdentity::from_value(&parse(r#"{"identity_schema": 1}"#)),
            None
        );
    }

    #[test]
    fn identity_hash_moves_with_every_identity_part() {
        let base = EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b");
        assert_eq!(base.identity_hash(), base.identity_hash());
        assert_eq!(base.identity_hash().len(), 64);

        let variants = [
            EdgeIdentity::new("a", "c", "can_feed", "hash-a", "hash-b"),
            EdgeIdentity::new("a", "b", "feeds", "hash-a", "hash-b"),
            EdgeIdentity::new("a", "b", "can_feed", "hash-a2", "hash-b"),
            EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b2"),
        ];
        for variant in variants {
            assert_ne!(base.identity_hash(), variant.identity_hash());
        }
    }

    #[test]
    fn from_graph_indexes_nodes_and_edges() {
        let graph = sample_graph();
        let index = StaticGraphIndex::from_graph("v1", &graph);

        assert_eq!(index.revision, "v1");
        assert_eq!(
            index.capability_ids,
            BTreeSet::from(["a".to_string(), "b".to_string()])
        );
        assert_eq!(index.graph_hash_by_id.len(), 2);
        // No metadata content hash and no descriptors: the identity hash is the fallback.
        assert_eq!(index.content_hash_by_id["a"], index.graph_hash_by_id["a"]);
        assert_eq!(index.content_hash_by_id["b"], index.graph_hash_by_id["b"]);
        assert_ne!(index.graph_hash_by_id["a"], index.graph_hash_by_id["b"]);

        assert_eq!(index.edges_by_identity.len(), 1);
        assert_eq!(index.edge_identities.len(), 1);
        let (identity_hash, identity) = index.edge_identities.iter().next().unwrap();
        assert_eq!(identity.source_id, "a");
        assert_eq!(identity.target_id, "b");
        assert_eq!(identity.relation_type, "can_feed");
        assert_eq!(identity.source_content_hash, index.content_hash_by_id["a"]);
        assert_eq!(identity.target_content_hash, index.content_hash_by_id["b"]);
        assert_eq!(identity_hash, &identity.identity_hash());

        let edge_value = &index.edges_by_identity[identity_hash];
        assert_eq!(edge_value["source"], Value::String("skill:a".to_string()));
        assert_eq!(
            edge_value["relation"],
            Value::String("can_feed".to_string())
        );

        let observation = EvolutionGraphEdge {
            source_id: "skill:a".to_string(),
            target_id: " b ".to_string(),
            relation_type: "can_feed".to_string(),
            ..EvolutionGraphEdge::default()
        };
        let derived = edge_identity_from_observation(&observation, &index);
        assert_eq!(derived.as_ref(), Some(identity));
    }

    #[test]
    fn from_graph_keeps_edges_with_unknown_endpoints() {
        let mut graph = sample_graph();
        graph.nodes.remove("skill:b");
        let index = StaticGraphIndex::from_graph("v1", &graph);

        assert_eq!(index.capability_ids, BTreeSet::from(["a".to_string()]));
        assert_eq!(index.edges_by_identity.len(), 1);
        let identity = index.edge_identities.values().next().unwrap();
        assert_eq!(identity.target_id, "b");
        assert_eq!(identity.target_content_hash, "");

        let observation = EvolutionGraphEdge {
            source_id: "skill:a".to_string(),
            target_id: "skill:b".to_string(),
            relation_type: "can_feed".to_string(),
            ..EvolutionGraphEdge::default()
        };
        assert_eq!(edge_identity_from_observation(&observation, &index), None);
        assert!(!index.validates_nodes(["skill:a", "skill:b"]));
    }

    #[test]
    fn edge_identity_from_observation_rejects_unknown_and_blank_endpoints() {
        let mut graph = sample_graph();
        graph.nodes.get_mut("skill:b").unwrap().metadata.insert(
            CONTENT_HASH_FIELD.to_string(),
            Value::String("   ".to_string()),
        );
        let index = StaticGraphIndex::from_graph("v1", &graph);
        // A blank published hash falls back to the identity hash, so the edge still binds.
        assert_eq!(index.content_hash_by_id["b"], index.graph_hash_by_id["b"]);

        let unknown = EvolutionGraphEdge {
            source_id: "z".to_string(),
            target_id: "a".to_string(),
            relation_type: "can_feed".to_string(),
            ..EvolutionGraphEdge::default()
        };
        assert_eq!(edge_identity_from_observation(&unknown, &index), None);
    }

    #[test]
    fn validates_nodes_normalizes_before_lookup() {
        let index = StaticGraphIndex::from_graph("v1", &sample_graph());
        assert!(index.validates_nodes(["skill:a"]));
        assert!(index.validates_nodes(["a", " skill:b ", "capability:a"]));
        assert!(index.validates_nodes(Vec::<&str>::new()));
        assert!(!index.validates_nodes(["missing"]));
        assert!(!index.validates_nodes(["a", "missing"]));
    }

    #[test]
    fn descriptors_override_the_node_content_hash() {
        let graph = sample_graph();
        let plain = StaticGraphIndex::from_graph("v1", &graph);
        let descriptor = CapabilityDescriptor::new("skill", "a", "Skill A");
        let indexed = StaticGraphIndex::from_graph_with_descriptors(
            "v1",
            &graph,
            std::slice::from_ref(&descriptor),
        );

        let expected = descriptor.compute_content_hash();
        assert_eq!(indexed.content_hash_by_id["a"], expected);
        assert_ne!(
            indexed.content_hash_by_id["a"],
            plain.content_hash_by_id["a"]
        );
        // Only the covered node moves; graph identity hashes never depend on content.
        assert_eq!(
            indexed.content_hash_by_id["b"],
            plain.content_hash_by_id["b"]
        );
        assert_eq!(indexed.graph_hash_by_id, plain.graph_hash_by_id);

        let plain_identity = plain.edge_identities.values().next().unwrap();
        let indexed_identity = indexed.edge_identities.values().next().unwrap();
        assert_ne!(
            plain_identity.identity_hash(),
            indexed_identity.identity_hash()
        );
        assert_eq!(indexed_identity.source_content_hash, expected);
        assert_eq!(
            indexed_identity.target_content_hash,
            plain.content_hash_by_id["b"]
        );
        assert_eq!(indexed.capability_ids, plain.capability_ids);
    }

    #[test]
    fn descriptors_match_by_qualified_and_bare_id() {
        let mut graph = CapabilityGraph::new("graph-1");
        graph.upsert_node(CapabilityNode::new("a", "Bare A", "skill"));
        let bare = CapabilityDescriptor::new("skill", "a", "Bare A");
        let indexed = StaticGraphIndex::from_graph_with_descriptors(
            "v1",
            &graph,
            std::slice::from_ref(&bare),
        );
        assert_eq!(indexed.content_hash_by_id["a"], bare.compute_content_hash());

        let mut graph = CapabilityGraph::new("graph-1");
        graph.upsert_node(CapabilityNode::new("skill:a", "Qualified A", "skill"));
        let qualified = CapabilityDescriptor::new("skill", "a", "Qualified A");
        let indexed = StaticGraphIndex::from_graph_with_descriptors(
            "v1",
            &graph,
            std::slice::from_ref(&qualified),
        );
        assert_eq!(
            indexed.content_hash_by_id["a"],
            qualified.compute_content_hash()
        );
    }

    #[test]
    fn node_metadata_content_hash_wins_over_the_identity_hash() {
        let mut graph = CapabilityGraph::new("graph-1");
        graph.upsert_node(
            CapabilityNode::new("skill:a", "Skill A", "skill").with_metadata(
                CONTENT_HASH_FIELD,
                Value::String("artifact-hash-a".to_string()),
            ),
        );
        graph.upsert_node(CapabilityNode::new("skill:b", "Skill B", "skill"));
        graph
            .add_edge(crate::graph::CapabilityEdge::can_feed("skill:a", "skill:b"))
            .unwrap();
        let index = StaticGraphIndex::from_graph("v1", &graph);

        assert_eq!(index.content_hash_by_id["a"], "artifact-hash-a");
        assert_ne!(index.content_hash_by_id["a"], index.graph_hash_by_id["a"]);
        assert_eq!(index.content_hash_by_id["b"], index.graph_hash_by_id["b"]);

        // A non-string metadata entry is ignored instead of failing the scan.
        let mut graph = CapabilityGraph::new("graph-1");
        graph.upsert_node(
            CapabilityNode::new("skill:a", "Skill A", "skill")
                .with_metadata(CONTENT_HASH_FIELD, Value::from(7)),
        );
        let index = StaticGraphIndex::from_graph("v1", &graph);
        assert_eq!(index.content_hash_by_id["a"], index.graph_hash_by_id["a"]);
    }

    #[test]
    fn capability_type_falls_back_from_metadata_to_node_type_to_skill() {
        let mut graph = CapabilityGraph::new("graph-1");
        graph.upsert_node(
            CapabilityNode::new("a", "A", "workflow")
                .with_metadata(CAPABILITY_TYPE_FIELD, Value::String(" tool ".to_string())),
        );
        graph.upsert_node(CapabilityNode::new("b", "B", "workflow"));
        graph.upsert_node(CapabilityNode::new("c", "C", "  "));
        graph.upsert_node(
            CapabilityNode::new("d", "D", "tool")
                .with_metadata(CAPABILITY_TYPE_FIELD, Value::String("  ".to_string())),
        );
        let index = StaticGraphIndex::from_graph("v1", &graph);

        assert_eq!(
            capability_type_of(graph.nodes.get("a").unwrap()),
            "tool".to_string()
        );
        assert_eq!(
            capability_type_of(graph.nodes.get("b").unwrap()),
            "workflow".to_string()
        );
        assert_eq!(
            capability_type_of(graph.nodes.get("c").unwrap()),
            DEFAULT_CAPABILITY_TYPE.to_string()
        );
        assert_eq!(
            capability_type_of(graph.nodes.get("d").unwrap()),
            "tool".to_string()
        );
        assert_eq!(index.capability_ids.len(), 4);
    }

    #[test]
    fn graph_identity_hash_covers_metadata_and_type() {
        let mut plain = CapabilityGraph::new("graph-1");
        plain.upsert_node(CapabilityNode::new("a", "A", "skill"));
        let mut with_metadata = CapabilityGraph::new("graph-1");
        with_metadata.upsert_node(
            CapabilityNode::new("a", "A", "skill")
                .with_metadata("description", Value::String("described".to_string())),
        );
        let mut retyped = CapabilityGraph::new("graph-1");
        retyped.upsert_node(CapabilityNode::new("a", "A", "tool"));

        let plain_hash = StaticGraphIndex::from_graph("v1", &plain).graph_hash_by_id["a"].clone();
        assert_ne!(
            plain_hash,
            StaticGraphIndex::from_graph("v1", &with_metadata).graph_hash_by_id["a"]
        );
        assert_ne!(
            plain_hash,
            StaticGraphIndex::from_graph("v1", &retyped).graph_hash_by_id["a"]
        );
        assert_eq!(
            plain_hash,
            StaticGraphIndex::from_graph("v2", &plain).graph_hash_by_id["a"]
        );
    }
}
