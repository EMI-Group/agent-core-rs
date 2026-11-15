# openjiuwen-symphony — `src/model`

## Intent

Capability cards and their deterministic derived views: identity and typed inputs/outputs, versioned fingerprints, the data-type compatibility vocabulary, name normalization, and the I/O-name/I/O-spec normalization layer ported from Python `shared/fingerprint/normalization.py`.
Everything here is pure data plus pure functions: no clock, no RNG, no hash-map ordering, no network, and no model calls.

## API Surface

```rust
pub struct CapabilityKey { pub capability_type: String, pub capability_id: String }   // descriptor.rs
pub struct CapabilityIo { pub name: String, #[serde(rename = "type")] pub io_type: String,
                          pub description: String, pub required: Option<bool>,
                          pub default: Option<Value>, pub metadata: BTreeMap<String, Value> }
pub struct CapabilityDescriptor { capability_id, capability_type, name, description, source,
                                 available, inputs, outputs, classification, tags, content_hash,
                                 semantic_content, metadata }
pub const DEFAULT_IO_TYPE: &str;                    // "any"

pub struct SemanticProfile;  pub struct CapabilityFingerprint;  pub struct FingerprintArtifact;
pub const ARTIFACT_SCHEMA_VERSION: &str;

pub struct DataTypeVocabulary;                      // normalize / canonical_types / can_feed / is_subtype
pub fn normalize_type(raw: &str) -> String;

pub fn normalize_name_key(name: &str) -> String;    // naming.rs
pub fn fuzzy_name_distance(a: &str, b: &str) -> usize;
pub fn fuzzy_name_similarity(a: &str, b: &str) -> f64;

// normalization/ — re-exported from this module
pub fn normalize_io_name(raw: &str, fallback: &str) -> String;
pub struct IONameResolution { pub raw, pub token, pub normalized: Option<String>, pub known, pub method }
pub struct IONameTerm;  pub struct IONameVocabulary;  pub enum IoDirection { Input, Output }
pub fn builtin_io_name_vocabulary() -> IONameVocabulary;
pub fn normalize_io_specs(raw: &Value, vocabulary: &IONameVocabulary, default_required: bool) -> Vec<CapabilityIo>;
pub fn normalize_io_specs_with_issues(..) -> (Vec<CapabilityIo>, Vec<NormalizationIssue>);
pub fn normalize_io_specs_with_audit(..) -> (Vec<CapabilityIo>, Vec<NormalizationIssue>, Vec<NormalizationDecision>);
pub fn normalize_io_specs_for_direction(raw: &Value, vocabulary: &IONameVocabulary,
                                        direction: IoDirection, default_required: Option<bool>) -> IONormalizationResult;
pub fn resolve_data_type(raw: &str) -> DataTypeResolution;
pub fn sanitize_io_specs(raw: &Value) -> Value;
pub struct NormalizationIssue;  pub struct NormalizationDecision;  pub struct IONormalizationResult;
pub const IO_NAME_VOCAB_VERSION, MAX_VOCABULARY_TERMS, MAX_TERM_VALUES, MAX_VOCABULARY_ALIASES,
         MAX_IO_NAME_LEN, MAX_JSON_DEPTH, DATA_TYPE_VOCABULARY_VERSION;
pub const ISSUE_CODE_*, METHOD_*, FIELD_*, VOCABULARY_*, IO_SPEC_KEYS;
```

## Normalization Rules

Names fold in this order: credential redaction, full-width ASCII and Unicode space separators to ASCII, lowercase, every non-alphanumeric run to one `_`, trim separators, bound to `MAX_IO_NAME_LEN`, else the normalized `fallback`.
No NFKC table is available offline, so ligatures, subscripts and circled forms are not decomposed; the fold covers the cases the Python tests exercise.
`conservative_io_name_alias` aliases a token to a canonical name only when exactly one plural/singular or `_`-suffix match exists; ties stay unaliased.
`IONameVocabulary::build` collects observed input/output names with counts, directions and normalized data types; `content_hash` is SHA-256 over canonical JSON of the sorted terms plus `version`, so input order never moves it.
`from_value` validates `version` and `content_hash` and bounds the term/alias counts; `builtin_io_name_vocabulary()` is the small documented offline corpus (Python ships none).
`normalize_io_specs*` accepts a list of strings/mappings, a single mapping, or a name-keyed mapping, and records a decision for every rename, coercion, default and redaction and an issue for every drop, unknown type, duplicate or invalid flag.
`required`/`optional` accept `true`/`false`/`1`/`0`/`"yes"`/`"no"`/`"required"`/`"optional"`; an unparseable value becomes an `invalid_required` issue plus a `defaulted` decision.
Sensitive names, sensitive metadata keys and sensitive defaults are replaced with `REDACTED` and audited; metadata values pass through `redact_sensitive_json`.
`sanitize_io_specs` returns a depth-capped (64) JSON-safe replay state, keeping only accepted spec keys and stringifying opaque leaves deterministically.

## Constraints

- No `unsafe`, no panics on malformed input, errors only through `crate::error`; nothing here returns `Error` except the vocabulary validators.
- Every public item is documented and `serde`-serializable with `snake_case` names and defaulted collections; `schemars::JsonSchema` is derived where sibling types derive it.
- Python parity is behavioural, not literal: the Rust layer documents its deviations (name cap, `_` collapsing, redacted sensitive names, Rust-only issue codes) in `normalization`'s module docs.

## Routing Table

- `./descriptor.rs` → `CapabilityIo`, `CapabilityDescriptor`, JSON Schema derivation, content hash
- `./fingerprint.rs` → `CapabilityFingerprint`, `SemanticProfile`, `FingerprintArtifact`
- `./datatype.rs` → `DataTypeVocabulary`, `can_feed`, `is_subtype`, `normalize_type`
- `./naming.rs` → `normalize_name_key`, fuzzy name distance/similarity
- `./normalization/` → I/O-name folding, the observed vocabulary, I/O-spec normalization and its audit trail
  - `./normalization/names.rs` → `normalize_io_name`, `conservative_io_name_alias`, `IONameResolution`
  - `./normalization/vocabulary.rs` → `IONameVocabulary`, `IONameTerm`, `IoDirection`, `builtin_io_name_vocabulary`
  - `./normalization/specs.rs` → issue/method constants, `NormalizationIssue`, `NormalizationDecision`, `resolve_data_type`, `normalize_io_specs*`, `sanitize_io_specs`, collection coercion
  - `./normalization/specs/entry.rs` → per-declaration normalization (name, metadata, type, flag, default) and its audit records
