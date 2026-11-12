//! Capability fingerprints: versioned semantic views and their artifact envelope.
//!
//! Ports the offline subset of Python `openjiuwen.symphony.models.fingerprint`. The
//! Python `CapabilityFingerprint` subclasses `CapabilityDescriptor`; here the
//! descriptor is flattened into the fingerprint, so the JSON is the descriptor's keys
//! plus the fingerprint's own keys — the same wire shape as Python's `model_dump`.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::descriptor::CapabilityDescriptor;

/// The current fingerprint format version.
pub const FINGERPRINT_VERSION: &str = "1.0.0";

/// The current artifact envelope schema version.
pub const ARTIFACT_SCHEMA_VERSION: &str = "1.0";

/// Retrieval-oriented, human-readable description of a capability.
///
/// Python derives these values with an LLM; [`CapabilityDescriptor::fingerprint`]
/// derives them deterministically from the card instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SemanticProfile {
    /// One-paragraph summary of what the capability does.
    pub summary: String,
    /// Identities (usually qualified ids) the fingerprint can satisfy.
    pub capabilities: Vec<String>,
    /// Situations the capability is meant for, usually its tags.
    pub use_cases: Vec<String>,
    /// Known limitations, e.g. an unavailable capability.
    pub limitations: Vec<String>,
    /// Sorted, deduplicated retrieval keywords.
    pub keywords: Vec<String>,
}

/// The versioned semantic and quality view of one capability.
///
/// Serialization is flat: the [`CapabilityDescriptor`] is `#[serde(flatten)]`ed, so a
/// fingerprint's JSON has exactly the descriptor's keys plus `version`, `static_data`,
/// `semantic_profile`, `failures`, `evidence`, and `suggestions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityFingerprint {
    /// The capability card this fingerprint describes, serialized inline.
    #[serde(flatten)]
    pub descriptor: CapabilityDescriptor,
    /// Fingerprint format version; [`FINGERPRINT_VERSION`] for the current format.
    pub version: String,
    /// Deterministic evidence about the card: content hash, source, availability.
    pub static_data: BTreeMap<String, Value>,
    /// The derived semantic profile.
    pub semantic_profile: SemanticProfile,
    /// Non-fatal problems observed while fingerprinting.
    pub failures: Vec<String>,
    /// Evidence references backing the fingerprint.
    pub evidence: Vec<Value>,
    /// Suggested improvements, as free-form strings.
    pub suggestions: Vec<String>,
}

impl Default for CapabilityFingerprint {
    fn default() -> Self {
        Self {
            descriptor: CapabilityDescriptor::default(),
            version: FINGERPRINT_VERSION.to_string(),
            static_data: BTreeMap::new(),
            semantic_profile: SemanticProfile::default(),
            failures: Vec::new(),
            evidence: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

/// A batch of fingerprints plus the provenance of the inventory they describe.
///
/// This is the on-disk artifact envelope: `schema_version` versions the envelope
/// itself, `generated_at` is metadata only (never used for ordering), and
/// `source_snapshot` records where the cards came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct FingerprintArtifact {
    /// Envelope schema version; [`ARTIFACT_SCHEMA_VERSION`] for the current format.
    pub schema_version: String,
    /// RFC 3339 creation timestamp (metadata only).
    pub generated_at: String,
    /// Provenance of the inventory the fingerprints were derived from.
    pub source_snapshot: BTreeMap<String, Value>,
    /// The fingerprints, in registration order.
    pub fingerprints: Vec<CapabilityFingerprint>,
}

impl Default for FingerprintArtifact {
    fn default() -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
            generated_at: String::new(),
            source_snapshot: BTreeMap::new(),
            fingerprints: Vec::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_carry_the_current_versions() {
        assert_eq!(
            CapabilityFingerprint::default().version,
            FINGERPRINT_VERSION
        );
        assert_eq!(
            FingerprintArtifact::default().schema_version,
            ARTIFACT_SCHEMA_VERSION
        );
        assert_eq!(SemanticProfile::default().keywords, Vec::<String>::new());
    }

    #[test]
    fn fingerprint_serializes_the_descriptor_inline() {
        let descriptor = CapabilityDescriptor::new("tool", "sum", "Sum")
            .with_description("adds numbers")
            .with_available(true);
        let fingerprint = descriptor.fingerprint();
        let value = serde_json::to_value(&fingerprint).expect("fingerprint serializes");
        assert_eq!(value["capability_id"], "sum");
        assert_eq!(value["version"], FINGERPRINT_VERSION);
        assert!(value.get("descriptor").is_none(), "flatten must not nest");
        assert_eq!(value["static_data"]["available"], Value::Bool(true));
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let artifact: FingerprintArtifact = serde_json::from_value(serde_json::json!({
            "schema_version": "1.0",
            "fingerprints": [],
            "future_field": {"ignored": true},
        }))
        .expect("artifact deserializes");
        assert_eq!(artifact.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(artifact.generated_at, String::new());
    }
}
