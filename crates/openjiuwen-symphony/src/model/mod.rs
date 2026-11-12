//! Capability cards, fingerprints, data-type vocabulary, and naming helpers.
//!
//! This is the model layer of the capability-asset engine: the wire types a provider
//! hands over ([`CapabilityDescriptor`], [`CapabilityIo`], [`CapabilityKey`]), their
//! versioned semantic view ([`CapabilityFingerprint`], [`SemanticProfile`],
//! [`FingerprintArtifact`]), and the two deterministic comparison helpers used by the
//! graph and planner ([`DataTypeVocabulary`], [`normalize_name_key`]).
//!
//! Every type here is `serde`-serializable with `snake_case` names, every field
//! defaults, and unknown JSON fields are ignored. Derived values (schemas, content
//! hashes, fingerprints) are deterministic: no clock, no hash-map iteration order.

/// The data-type compatibility vocabulary behind `can_feed` decisions.
pub mod datatype;
/// Capability cards: identity, typed inputs/outputs, schemas, and content hashing.
pub mod descriptor;
/// Versioned capability fingerprints and the artifact envelope.
pub mod fingerprint;
/// Deterministic name normalization and fuzzy name comparison.
pub mod naming;

pub use datatype::{normalize_type, DataTypeVocabulary};
pub use descriptor::{CapabilityDescriptor, CapabilityIo, CapabilityKey, DEFAULT_IO_TYPE};
pub use fingerprint::{
    CapabilityFingerprint, FingerprintArtifact, SemanticProfile, ARTIFACT_SCHEMA_VERSION,
    FINGERPRINT_VERSION,
};
pub use naming::{fuzzy_name_distance, fuzzy_name_similarity, normalize_name_key};
