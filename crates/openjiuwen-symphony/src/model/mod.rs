//! Capability cards, fingerprints, data-type vocabulary, and naming helpers.
//!
//! This is the model layer of the capability-asset engine: the wire types a provider hands
//! over ([`CapabilityDescriptor`](crate::model::CapabilityDescriptor),
//! [`CapabilityIo`](crate::model::CapabilityIo),
//! [`CapabilityKey`](crate::model::CapabilityKey)), their versioned semantic view
//! ([`CapabilityFingerprint`](crate::model::CapabilityFingerprint),
//! [`SemanticProfile`](crate::model::SemanticProfile),
//! [`FingerprintArtifact`](crate::model::FingerprintArtifact)), and the two deterministic
//! comparison helpers used by the graph and planner
//! ([`DataTypeVocabulary`](crate::model::DataTypeVocabulary),
//! [`normalize_name_key`](crate::model::normalize_name_key)).
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
/// I/O-name and I/O-spec normalization with a full audit trail.
pub mod normalization;

pub use datatype::{normalize_type, DataTypeVocabulary};
pub use descriptor::{CapabilityDescriptor, CapabilityIo, CapabilityKey, DEFAULT_IO_TYPE};
pub use fingerprint::{
    CapabilityFingerprint, FingerprintArtifact, SemanticProfile, ARTIFACT_SCHEMA_VERSION,
    FINGERPRINT_VERSION,
};
pub use naming::{fuzzy_name_distance, fuzzy_name_similarity, normalize_name_key};
pub use normalization::{
    builtin_io_name_vocabulary, normalize_io_name, normalize_io_specs,
    normalize_io_specs_for_direction, normalize_io_specs_with_audit,
    normalize_io_specs_with_issues, resolve_data_type, sanitize_io_specs, DataTypeResolution,
    DataTypeSource, IONameResolution, IONameTerm, IONameVocabulary, IONormalizationResult,
    IoDirection, NormalizationDecision, NormalizationIssue, IO_NAME_VOCAB_VERSION, MAX_IO_NAME_LEN,
    MAX_TERM_VALUES, MAX_VOCABULARY_ALIASES, MAX_VOCABULARY_TERMS,
};
