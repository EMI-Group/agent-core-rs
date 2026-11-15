//! I/O-spec normalization: raw declarations to typed [`CapabilityIo`](crate::model::CapabilityIo)
//! values plus audit.
//!
//! Ports `_normalize_io_specs`, `_normalize_io_entry`, `_coerce_io_entries`,
//! `_normalize_required`, `_to_bool`, `_first_present`, `_normalize_metadata`,
//! `_to_json_value`, `_normalize_type_token`, `DataTypeResolution`, `NormalizationIssue`,
//! `NormalizationDecision`, `IONormalizationResult` and `sanitize_io_specs` from Python
//! `openjiuwen.symphony.shared.fingerprint.normalization`.
//!
//! A raw spec is whatever an extractor produced for one side of a capability: a list of
//! mappings, one mapping, a mapping of name → spec, or a list of plain strings. Every
//! accepted alias key is honoured, names are resolved through the corpus vocabulary,
//! types through [`DataTypeVocabulary`](crate::model::DataTypeVocabulary), `required`/`optional`
//! are coerced, and the result
//! is a vector of [`CapabilityIo`](crate::model::CapabilityIo) values whose every cleanup is
//! recorded — a dropped,
//! renamed, coerced, defaulted, redacted, or unrecognized field always leaves either a
//! [`NormalizationIssue`] or a [`NormalizationDecision`] behind.
//!
//! # Rust-only decisions
//!
//! - **Names never come from a model-free ambiguity.** The direction is explicit in
//!   [`normalize_io_specs_for_direction`]; the direction-less entry points speak for the
//!   input side (fallback name `input`, Python's `required: true` default when the caller
//!   does not say otherwise).
//! - **Sensitive names are redacted.** Python keeps an I/O name such as `api_key` and
//!   redacts only its default; this port additionally replaces the emitted name with
//!   [`REDACTED`](crate::redaction::REDACTED) and records a decision plus a
//!   `sensitive_io_name_redacted` issue, so a
//!   credential-shaped label never reaches a fingerprint verbatim.
//! - **Metadata is redacted during normalization.** Python redacts metadata only in the
//!   sanitized replay state; here [`redact_sensitive_json`](crate::redaction::redact_sensitive_json)
//!   runs on the way in, and every
//!   top-level key it recognized is audited.
//! - **Unknown types keep their spelling.** Python collapses an unrecognized type to the
//!   shared `unknown` sentinel from `data_types.yaml`;
//!   [`DataTypeVocabulary`](crate::model::DataTypeVocabulary) has no such
//!   sentinel, so the normalized token of the raw spelling is kept (`image/png` →
//!   `image_png`) and the decision's `source` is [`DataTypeSource::Unknown`]. Blank types
//!   become [`DEFAULT_IO_TYPE`](crate::model::DEFAULT_IO_TYPE) (`any`).
//! - **`invalid_required` and the two sensitive-redaction codes are Rust-only issues.**
//!   Python silently falls back to the direction default for an unparseable `required`.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::datatype::DataTypeVocabulary;
use crate::model::descriptor::{CapabilityIo, DEFAULT_IO_TYPE};
use crate::model::normalization::names::normalize_io_name;
use crate::model::normalization::vocabulary::{
    IONameVocabulary, IoDirection, IO_NAME_VOCAB_VERSION,
};
use crate::redaction::{is_sensitive_name, redact_sensitive_json, redact_sensitive_text, REDACTED};

mod entry;

pub(crate) use entry::{entry_payload, first_present, type_text, value_text, vocabulary_version};

/// The issue code for a raw value that is not a list, mapping, or string.
pub const ISSUE_CODE_INVALID_IO_COLLECTION: &str = "invalid_io_collection";
/// The issue code for one entry that is neither a string nor a mapping.
pub const ISSUE_CODE_INVALID_IO_ENTRY: &str = "invalid_io_entry";
/// The issue code for a declaration whose name was blank and fell back.
pub const ISSUE_CODE_MISSING_IO_NAME: &str = "missing_io_name";
/// The issue code for a name the corpus vocabulary has never observed.
pub const ISSUE_CODE_IO_NAME_NOT_IN_VOCABULARY: &str = "io_name_not_in_vocabulary";
/// The issue code for a name that collided and was suffixed with its occurrence number.
pub const ISSUE_CODE_DUPLICATE_IO_NAME: &str = "duplicate_io_name";
/// The issue code for a data type the vocabulary does not recognize.
pub const ISSUE_CODE_UNKNOWN_DATA_TYPE: &str = "unknown_data_type";
/// The issue code for a default that was dropped because the name is sensitive.
pub const ISSUE_CODE_SENSITIVE_DEFAULT_REDACTED: &str = "sensitive_default_redacted";
/// The issue code for a name that is itself a credential and was redacted.
pub const ISSUE_CODE_SENSITIVE_IO_NAME_REDACTED: &str = "sensitive_io_name_redacted";
/// The issue code for a metadata key that is a credential and was redacted.
pub const ISSUE_CODE_SENSITIVE_METADATA_REDACTED: &str = "sensitive_metadata_redacted";
/// The issue code for an unparseable `required`/`optional` value.
pub const ISSUE_CODE_INVALID_REQUIRED: &str = "invalid_required";

/// The decision method for a name derived by syntax alone (no corpus vocabulary).
pub const METHOD_SYNTAX: &str = "syntax";
/// The decision method for a name that fell back because the corpus did not know it.
pub const METHOD_VOCAB_FALLBACK: &str = "vocab_fallback";
/// The decision method for a name that was suffixed to make it unique.
pub const METHOD_DEDUPLICATE: &str = "deduplicate";
/// The decision method for a value that was coerced to another type.
pub const METHOD_COERCED: &str = "coerced";
/// The decision method for a value that was filled in from a default.
pub const METHOD_DEFAULTED: &str = "defaulted";
/// The decision method for a value that was redacted.
pub const METHOD_REDACTED: &str = "redacted";

/// The decision field name for the resolved I/O name.
pub const FIELD_NAME: &str = "name";
/// The decision field name for the resolved data type.
pub const FIELD_TYPE: &str = "type";
/// The decision field name for the description.
pub const FIELD_DESCRIPTION: &str = "description";
/// The decision field name for the `required` flag.
pub const FIELD_REQUIRED: &str = "required";
/// The decision field name for the default value.
pub const FIELD_DEFAULT: &str = "default";
/// The decision field name for the metadata map.
pub const FIELD_METADATA: &str = "metadata";

/// The vocabulary label used by name decisions.
pub const VOCABULARY_IO_NAME: &str = "io_name_vocab";
/// The vocabulary label used by type decisions.
pub const VOCABULARY_DATA_TYPE: &str = "data_type_vocab";
/// The vocabulary label used by metadata-redaction decisions.
pub const VOCABULARY_REDACTION: &str = "redaction";

/// The version of the embedded data-type vocabulary.
///
/// [`DataTypeVocabulary`] carries no version field of its own; this constant is the version
/// string of the Python `data_types.yaml` resource the crate's table mirrors.
pub const DATA_TYPE_VOCABULARY_VERSION: &str = "data-type-v1";

/// The maximum container depth the sanitizer and metadata copier descend into.
///
/// Deeper containers become [`Value::Null`], matching
/// [`MAX_METADATA_DEPTH`](crate::llm::identity::MAX_METADATA_DEPTH) and
/// `redaction`'s depth cap, so a pathological document cannot exhaust the stack.
pub const MAX_JSON_DEPTH: usize = 64;

/// The alias keys an I/O entry may use (Python's `_IO_SPEC_KEYS`).
pub const IO_SPEC_KEYS: [&str; 13] = [
    "name",
    "id",
    "type",
    "data_type",
    "datatype",
    "format",
    "mime_type",
    "description",
    "help",
    "required",
    "optional",
    "default",
    "metadata",
];

/// The alias keys a raw data type is read from, in precedence order.
pub(crate) const TYPE_KEYS: [&str; 5] = ["type", "data_type", "datatype", "format", "mime_type"];

/// The alias keys a JSON Schema type object exposes, in precedence order.
const TYPE_OBJECT_KEYS: [&str; 3] = ["format", "contentMediaType", "type"];

/// Where a normalized data type came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DataTypeSource {
    /// No type was declared; [`DEFAULT_IO_TYPE`] was used.
    #[default]
    Empty,
    /// The declared type is a canonical spelling.
    Exact,
    /// The declared type is an alias of a canonical spelling.
    Alias,
    /// The declared type is unknown; its normalized token was kept.
    Unknown,
}

impl DataTypeSource {
    /// The lowercase label Python records as the resolution `method`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Exact => "exact",
            Self::Alias => "alias",
            Self::Unknown => "unknown",
        }
    }
}

/// The result of resolving a raw data type against [`DataTypeVocabulary`].
///
/// `raw` is the trimmed declaration, `normalized` the canonical (or preserved) spelling,
/// and `source` how it was reached. [`DataTypeResolution::known`] is true only for
/// [`DataTypeSource::Exact`] and [`DataTypeSource::Alias`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DataTypeResolution {
    /// The trimmed raw declaration.
    pub raw: String,
    /// The canonical data type (or the preserved token when the type is unknown).
    pub normalized: String,
    /// How `normalized` was reached.
    pub source: DataTypeSource,
}

impl DataTypeResolution {
    /// Reports whether the declared type is a known canonical type or alias.
    #[must_use]
    pub fn known(&self) -> bool {
        matches!(self.source, DataTypeSource::Exact | DataTypeSource::Alias)
    }
}

/// One non-fatal cleanup or fallback observed during normalization.
///
/// Mirrors Python's `NormalizationIssue`: `code` is one of the `ISSUE_CODE_*` constants,
/// `message` explains it, and `direction`/`index` locate the declaration it came from.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct NormalizationIssue {
    /// The stable issue code; one of the `ISSUE_CODE_*` constants.
    pub code: String,
    /// A human-readable explanation.
    pub message: String,
    /// The direction the issue belongs to (`input`/`output`), filled in by the caller.
    pub direction: String,
    /// The zero-based position of the declaration, when the issue is per-entry.
    pub index: Option<usize>,
    /// Additional machine-readable detail, sorted by key.
    pub details: BTreeMap<String, Value>,
}

impl NormalizationIssue {
    /// Builds an issue with `code` and `message` and nothing else filled in.
    #[must_use]
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            ..Self::default()
        }
    }

    /// Sets the direction label.
    #[must_use]
    pub fn with_direction(mut self, direction: IoDirection) -> Self {
        self.direction = direction.as_str().to_string();
        self
    }

    /// Sets the declaration index.
    #[must_use]
    pub fn with_index(mut self, index: usize) -> Self {
        self.index = Some(index);
        self
    }

    /// Inserts one detail entry.
    #[must_use]
    pub fn with_detail(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.details.insert(key.to_string(), value.into());
        self
    }
}

/// One deterministic field-level normalization choice.
///
/// Mirrors Python's `NormalizationDecision`: it records the raw value, the normalized
/// token, the value emitted, the method, and the vocabulary (plus version) that produced
/// it. `raw_value` and `token` are redacted when they carry credential material.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct NormalizationDecision {
    /// The direction the decision belongs to (`input`/`output`).
    pub direction: String,
    /// The zero-based position of the declaration.
    pub index: Option<usize>,
    /// The field the decision is about; one of the `FIELD_*` constants.
    pub field: String,
    /// The raw value, redacted.
    pub raw_value: String,
    /// The normalized token, redacted when it is itself sensitive.
    pub token: String,
    /// The value emitted into the normalized declaration.
    pub normalized_value: String,
    /// How the value was reached; one of the `METHOD_*` constants.
    pub method: String,
    /// Which vocabulary produced the value; one of the `VOCABULARY_*` constants.
    pub vocabulary: String,
    /// The version of that vocabulary.
    pub vocabulary_version: String,
    /// Additional machine-readable detail, sorted by key.
    pub details: BTreeMap<String, Value>,
}

impl NormalizationDecision {
    /// Builds a decision for `field` (or the redaction of one) at `index`.
    #[must_use]
    pub fn new(direction: IoDirection, index: usize, field: &str, method: &str) -> Self {
        Self {
            direction: direction.as_str().to_string(),
            index: Some(index),
            field: field.to_string(),
            method: method.to_string(),
            ..Self::default()
        }
    }

    /// Sets the redacted raw value.
    #[must_use]
    pub fn with_raw_value(mut self, raw_value: impl Into<String>) -> Self {
        self.raw_value = raw_value.into();
        self
    }

    /// Sets the normalized token.
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = token.into();
        self
    }

    /// Sets the value emitted into the declaration.
    #[must_use]
    pub fn with_normalized_value(mut self, normalized_value: impl Into<String>) -> Self {
        self.normalized_value = normalized_value.into();
        self
    }

    /// Sets the vocabulary label and its version.
    #[must_use]
    pub fn with_vocabulary(mut self, vocabulary: &str, version: impl Into<String>) -> Self {
        self.vocabulary = vocabulary.to_string();
        self.vocabulary_version = version.into();
        self
    }

    /// Inserts one detail entry.
    #[must_use]
    pub fn with_detail(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.details.insert(key.to_string(), value.into());
        self
    }
}

/// Normalized I/O plus the complete deterministic audit trail.
///
/// Mirrors Python's `IONormalizationResult`; the direction-less entry points return its
/// pieces as tuples instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IONormalizationResult {
    /// The normalized declarations, in input order.
    pub items: Vec<CapabilityIo>,
    /// Every non-fatal cleanup, in discovery order.
    pub issues: Vec<NormalizationIssue>,
    /// Every field-level choice, in discovery order.
    pub decisions: Vec<NormalizationDecision>,
}

/// Resolves a raw data type against the standard [`DataTypeVocabulary`].
///
/// The declaration is trimmed, then resolved exact-first (a canonical spelling is
/// [`DataTypeSource::Exact`]), then as an alias ([`DataTypeSource::Alias`], so `str` →
/// `string` and `float` → `number`), and finally as an unknown token
/// ([`DataTypeSource::Unknown`], keeping the ASCII-folded spelling so distinct unknown
/// types stay distinguishable). A blank declaration becomes [`DEFAULT_IO_TYPE`] with
/// [`DataTypeSource::Empty`].
///
/// ```
/// use openjiuwen_symphony::model::normalization::{resolve_data_type, DataTypeSource};
///
/// assert_eq!(resolve_data_type("  STR ").normalized, "string");
/// assert_eq!(resolve_data_type("float").source, DataTypeSource::Alias);
/// assert_eq!(resolve_data_type("").source, DataTypeSource::Empty);
/// assert_eq!(resolve_data_type("image/png").normalized, "image_png");
/// assert!(!resolve_data_type("image/png").known());
/// ```
#[must_use]
pub fn resolve_data_type(raw: &str) -> DataTypeResolution {
    let raw = raw.trim().to_string();
    if raw.is_empty() {
        return DataTypeResolution {
            raw: String::new(),
            normalized: DEFAULT_IO_TYPE.to_string(),
            source: DataTypeSource::Empty,
        };
    }
    let vocabulary = DataTypeVocabulary::standard();
    let lowered = raw.to_lowercase();
    if is_canonical(&lowered, &vocabulary) {
        return DataTypeResolution {
            raw,
            normalized: lowered,
            source: DataTypeSource::Exact,
        };
    }
    let canonical = vocabulary.normalize(&lowered);
    if canonical != lowered && is_canonical(&canonical, &vocabulary) {
        return DataTypeResolution {
            raw,
            normalized: canonical,
            source: DataTypeSource::Alias,
        };
    }
    let token = type_token(&lowered);
    if token != lowered {
        let token_canonical = DataTypeVocabulary::standard().normalize(&token);
        if token_canonical != token || is_canonical(&token, &vocabulary) {
            return DataTypeResolution {
                raw,
                normalized: token_canonical,
                source: DataTypeSource::Alias,
            };
        }
    }
    let normalized = if token.is_empty() { lowered } else { token };
    DataTypeResolution {
        raw,
        normalized,
        source: DataTypeSource::Unknown,
    }
}

/// Normalizes raw I/O declarations, discarding the audit trail.
///
/// The direction-less convenience path: it speaks for the input side, so a blank name
/// falls back to `input` and `default_required` is the caller's decision (Python requires
/// every input). Use [`normalize_io_specs_for_direction`] to normalize outputs or to keep
/// the full audit.
///
/// ```
/// use openjiuwen_symphony::model::{builtin_io_name_vocabulary, normalize_io_specs};
/// use serde_json::json;
///
/// let items = normalize_io_specs(
///     &json!([{"name": "Query", "type": "str"}, {"name": "limit", "type": "int", "default": 10}]),
///     &builtin_io_name_vocabulary(),
///     true,
/// );
/// assert_eq!(items[0].name, "query");
/// assert_eq!(items[0].io_type, "string");
/// assert_eq!(items[1].required, Some(false));
/// ```
#[must_use]
pub fn normalize_io_specs(
    raw: &Value,
    vocabulary: &IONameVocabulary,
    default_required: bool,
) -> Vec<CapabilityIo> {
    normalize_io_specs_for_direction(raw, vocabulary, IoDirection::Input, Some(default_required))
        .items
}

/// Normalizes raw I/O declarations and keeps the cleanup diagnostics.
///
/// The direction-less convenience path; see [`normalize_io_specs`].
#[must_use]
pub fn normalize_io_specs_with_issues(
    raw: &Value,
    vocabulary: &IONameVocabulary,
    default_required: bool,
) -> (Vec<CapabilityIo>, Vec<NormalizationIssue>) {
    let result = normalize_io_specs_for_direction(
        raw,
        vocabulary,
        IoDirection::Input,
        Some(default_required),
    );
    (result.items, result.issues)
}

/// Normalizes raw I/O declarations and keeps decisions *and* issues.
///
/// The direction-less convenience path; see [`normalize_io_specs`].
#[must_use]
pub fn normalize_io_specs_with_audit(
    raw: &Value,
    vocabulary: &IONameVocabulary,
    default_required: bool,
) -> (
    Vec<CapabilityIo>,
    Vec<NormalizationIssue>,
    Vec<NormalizationDecision>,
) {
    let result = normalize_io_specs_for_direction(
        raw,
        vocabulary,
        IoDirection::Input,
        Some(default_required),
    );
    (result.items, result.issues, result.decisions)
}

/// Normalizes raw I/O declarations for one direction and returns the full audit trail.
///
/// `raw` accepts a list of strings or mappings, a single mapping, or a mapping of
/// name → spec. `default_required` is the `required` flag used when the declaration says
/// nothing (`direction.default_required()` for Python's own rule). Nothing is dropped or
/// changed silently: each rename, coercion, default, redaction and duplicate suffix
/// appears in the result's `issues` or `decisions`.
#[must_use]
pub fn normalize_io_specs_for_direction(
    raw: &Value,
    vocabulary: &IONameVocabulary,
    direction: IoDirection,
    default_required: Option<bool>,
) -> IONormalizationResult {
    let (entries, mut issues) = coerce_io_entries(raw);
    let mut items: Vec<CapabilityIo> = Vec::with_capacity(entries.len());
    let mut decisions: Vec<NormalizationDecision> = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let version = vocabulary_version(vocabulary);
    for (index, entry) in entries.iter().enumerate() {
        let (item, entry_decisions, entry_issues) = entry::normalize_entry(
            entry,
            index,
            direction,
            &version,
            vocabulary,
            default_required,
        );
        decisions.extend(entry_decisions);
        issues.extend(entry_issues);
        let Some(mut item) = item else {
            continue;
        };
        let occurrence = seen.entry(item.name.clone()).or_insert(0);
        *occurrence += 1;
        if *occurrence > 1 {
            let unique = format!("{}_{occurrence}", item.name);
            issues.push(
                NormalizationIssue::new(
                    ISSUE_CODE_DUPLICATE_IO_NAME,
                    format!("duplicate I/O name normalized to '{unique}'"),
                )
                .with_direction(direction)
                .with_index(index),
            );
            decisions.push(
                NormalizationDecision::new(direction, index, FIELD_NAME, METHOD_DEDUPLICATE)
                    .with_raw_value(&item.name)
                    .with_token(&item.name)
                    .with_normalized_value(&unique)
                    .with_vocabulary(VOCABULARY_IO_NAME, &version),
            );
            item.name = unique;
        }
        items.push(item);
    }
    for issue in &mut issues {
        if issue.direction.is_empty() {
            issue.direction = direction.as_str().to_string();
        }
    }
    IONormalizationResult {
        items,
        issues,
        decisions,
    }
}

/// Retains only the JSON state needed to repeat I/O normalization.
///
/// Each accepted entry is reduced to the recognized [`IO_SPEC_KEYS`] (unknown keys are
/// dropped), scanned by [`redact_sensitive_json`], and deep-copied past a depth cap of
/// [`MAX_JSON_DEPTH`], so a re-normalization replays the same rejected entries in the same
/// order without ever carrying a credential. A value that is not a collection yields
/// `0`, and a non-string, non-mapping entry is kept as `0` — Python's own sentinels for
/// "this was here and was invalid".
///
/// ```
/// use openjiuwen_symphony::model::sanitize_io_specs;
/// use serde_json::json;
///
/// let safe =
///     sanitize_io_specs(&json!([{"name": "q", "type": "text", "junk": 1, "metadata": {"api_key": "abc"}}]));
/// assert_eq!(
///     safe,
///     json!([{"name": "q", "type": "text", "metadata": {"api_key": "<redacted>"}}])
/// );
/// assert_eq!(sanitize_io_specs(&json!(3)), json!(0));
/// ```
#[must_use]
pub fn sanitize_io_specs(raw: &Value) -> Value {
    let (entries, issues) = coerce_io_entries(raw);
    if issues
        .iter()
        .any(|issue| issue.code == ISSUE_CODE_INVALID_IO_COLLECTION)
    {
        return Value::from(0);
    }
    Value::Array(entries.iter().map(sanitize_entry).collect())
}

/// Reduces one raw entry to its safe replay state.
fn sanitize_entry(entry: &Value) -> Value {
    match entry {
        Value::String(text) => Value::String(redact_sensitive_text(text)),
        Value::Object(map) => {
            let retained: Map<String, Value> = map
                .iter()
                .filter(|(key, _)| is_io_spec_key(key))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            json_safe(&redact_sensitive_json(&Value::Object(retained)), 0)
        }
        _ => Value::from(0),
    }
}

/// Coerces a raw declaration value into the list of entries to normalize.
///
/// A `null` yields nothing, a string one entry, a mapping either one entry (when it holds
/// an [`IO_SPEC_KEYS`] member) or one entry per `name → spec` pair (in sorted name order,
/// with `name` defaulted from the key), a list its items, and anything else nothing plus
/// an [`ISSUE_CODE_INVALID_IO_COLLECTION`] issue.
pub(crate) fn coerce_io_entries(raw: &Value) -> (Vec<Value>, Vec<NormalizationIssue>) {
    match raw {
        Value::Null => (Vec::new(), Vec::new()),
        Value::String(text) => (vec![Value::String(text.clone())], Vec::new()),
        Value::Object(map) => {
            if map.keys().any(|key| is_io_spec_key(key)) {
                return (vec![raw.clone()], Vec::new());
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut entries = Vec::with_capacity(keys.len());
            for key in keys {
                let Some(spec) = map.get(key) else {
                    continue;
                };
                entries.push(match spec {
                    Value::Object(spec) => {
                        let mut spec = spec.clone();
                        spec.entry("name".to_string())
                            .or_insert_with(|| Value::String(key.clone()));
                        Value::Object(spec)
                    }
                    other => {
                        let mut spec = Map::new();
                        spec.insert("name".to_string(), Value::String(key.clone()));
                        spec.insert("type".to_string(), other.clone());
                        Value::Object(spec)
                    }
                });
            }
            (entries, Vec::new())
        }
        Value::Array(items) => (items.clone(), Vec::new()),
        _ => (
            Vec::new(),
            vec![NormalizationIssue::new(
                ISSUE_CODE_INVALID_IO_COLLECTION,
                "I/O declarations must be a list, mapping, or string",
            )],
        ),
    }
}

/// Reports whether `raw` is a canonical data-type spelling in `vocabulary`.
fn is_canonical(raw: &str, vocabulary: &DataTypeVocabulary) -> bool {
    raw == "bytes" || vocabulary.canonical_types().contains(&raw)
}

/// Folds a type declaration to an ASCII token, following Python's `_normalize_type_token`.
fn type_token(raw: &str) -> String {
    let mut token = String::with_capacity(raw.len());
    let mut pending_separator = false;
    for character in raw.chars() {
        if character.is_ascii_alphanumeric() {
            if pending_separator && !token.is_empty() {
                token.push('_');
            }
            pending_separator = false;
            token.push(character.to_ascii_lowercase());
        } else {
            pending_separator = true;
        }
    }
    token
}

/// Reports whether `key` is one of the accepted I/O spec keys.
fn is_io_spec_key(key: &str) -> bool {
    IO_SPEC_KEYS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(key))
}

/// Deep-copies a JSON value, replacing containers deeper than [`MAX_JSON_DEPTH`] with null.
fn json_safe(value: &Value, depth: usize) -> Value {
    match value {
        Value::Object(map) => {
            if depth >= MAX_JSON_DEPTH {
                return Value::Null;
            }
            Value::Object(
                map.iter()
                    .map(|(key, item)| (key.clone(), json_safe(item, depth + 1)))
                    .collect(),
            )
        }
        Value::Array(items) => {
            if depth >= MAX_JSON_DEPTH {
                return Value::Null;
            }
            Value::Array(
                items
                    .iter()
                    .map(|item| json_safe(item, depth + 1))
                    .collect(),
            )
        }
        other => other.clone(),
    }
}

/// Coerces a raw flag to a boolean, following Python's `_to_bool`.
fn to_bool(value: &Value, default: Option<bool>) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => match text.trim().to_lowercase().as_str() {
            "true" | "yes" | "1" | "required" => Some(true),
            "false" | "no" | "0" | "optional" => Some(false),
            _ => default,
        },
        Value::Number(number) => match number.as_i64() {
            Some(1) => Some(true),
            Some(0) => Some(false),
            _ => default,
        },
        _ => default,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn data_types_resolve_exact_alias_empty_and_unknown() {
        assert_eq!(resolve_data_type(" text ").source, DataTypeSource::Exact);
        assert_eq!(resolve_data_type("BYTES").normalized, "bytes");
        assert_eq!(resolve_data_type("str").normalized, "string");
        assert_eq!(resolve_data_type("str").source, DataTypeSource::Alias);
        assert_eq!(resolve_data_type("integer").source, DataTypeSource::Exact);
        assert_eq!(resolve_data_type("").normalized, "any");
        assert_eq!(resolve_data_type("").raw, "");
        let unknown = resolve_data_type("image/png");
        assert_eq!(unknown.normalized, "image_png");
        assert_eq!(unknown.source, DataTypeSource::Unknown);
        assert!(!unknown.known());
    }

    #[test]
    fn json_safe_caps_depth_and_never_panics() {
        let mut value = json!({"leaf": 1});
        for _ in 0..(MAX_JSON_DEPTH + 5) {
            value = json!({"nested": value});
        }
        let safe = json_safe(&value, 0);
        assert!(safe.is_object());
        assert_eq!(json_safe(&json!(3), 0), json!(3));
    }

    #[test]
    fn sanitize_keeps_spec_keys_and_drops_unknown_keys() {
        let safe = sanitize_io_specs(&json!([
            "Quantity",
            {"name": "q", "type": "text", "junk": {"a": 1}, "metadata": {"api_key": "abc"}},
            7
        ]));
        assert_eq!(
            safe,
            json!([
                "Quantity",
                {"name": "q", "type": "text", "metadata": {"api_key": "<redacted>"}},
                0
            ])
        );
        assert_eq!(sanitize_io_specs(&json!("Quantity")), json!(["Quantity"]));
        assert_eq!(sanitize_io_specs(&json!(null)), json!([]));
        assert_eq!(sanitize_io_specs(&json!(3)), json!(0));
    }

    #[test]
    fn coercion_accepts_maps_and_name_keyed_objects() {
        let (entries, issues) = coerce_io_entries(&json!({"name": "q", "type": "text"}));
        assert_eq!(entries.len(), 1);
        assert_eq!(issues.len(), 0);
        let (entries, _) = coerce_io_entries(&json!({"alpha": "text", "beta": {"type": "int"}}));
        assert_eq!(
            entries,
            vec![
                json!({"name": "alpha", "type": "text"}),
                json!({"name": "beta", "type": "int"})
            ]
        );
        let (entries, issues) = coerce_io_entries(&json!(true));
        assert_eq!(entries.len(), 0);
        assert_eq!(issues[0].code, ISSUE_CODE_INVALID_IO_COLLECTION);
    }

    #[test]
    fn required_coercion_follows_python() {
        assert_eq!(to_bool(&json!("Yes"), None), Some(true));
        assert_eq!(to_bool(&json!("optional"), None), Some(false));
        assert_eq!(to_bool(&json!(1), None), Some(true));
        assert_eq!(to_bool(&json!(0), None), Some(false));
        assert_eq!(to_bool(&json!(2), Some(true)), Some(true));
        assert_eq!(to_bool(&json!(1.0), None), None);
        assert_eq!(to_bool(&json!("maybe"), None), None);
        assert_eq!(type_token("Image File"), "image_file");
        assert_eq!(type_token("日本語"), "");
    }
}
