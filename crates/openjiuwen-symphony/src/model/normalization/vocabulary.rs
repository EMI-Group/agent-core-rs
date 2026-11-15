//! The observed-corpus I/O-name vocabulary.
//!
//! Ports Python's `IONameTerm` / `IONameVocabulary` from
//! `openjiuwen.symphony.shared.fingerprint.normalization`. Python's I/O-name vocabulary is
//! **corpus-derived, not static**: it learns canonical `snake_case` terms from the
//! descriptors of one build, then (through `with_observed_specs`) from the raw I/O an LLM
//! extraction produced, so a build's names stay internally consistent.
//!
//! # Determinism
//!
//! - Terms, directions, data types and aliases are sorted and deduplicated, so two
//!   corpora containing the same I/O names in a different order produce the same
//!   [`IONameVocabulary::content_hash`].
//! - Counts merge with `max` rather than a sum, so re-merging the same evidence twice does
//!   not inflate them.
//! - Aliases that are themselves canonical terms are dropped, so an alias can never
//!   shadow a term.
//!
//! The hash is the SHA-256 of the canonical (sorted-key, whitespace-free) JSON of
//! `{"version", "terms"}`. Python renders the same document with `ensure_ascii=True`, so a
//! name containing non-ASCII characters hashes differently across the two
//! implementations; hashes are only ever compared inside one implementation.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{validation, Result};
use crate::model::datatype::DataTypeVocabulary;
use crate::model::descriptor::CapabilityDescriptor;
use crate::model::normalization::names::{
    conservative_io_name_alias, normalize_io_name, IONameResolution,
};
use crate::model::normalization::specs::{
    coerce_io_entries, entry_payload, first_present, type_text, value_text, TYPE_KEYS,
};
use crate::redaction::redact_sensitive_text;

/// The vocabulary schema version; every vocabulary this crate produces carries it.
pub const IO_NAME_VOCAB_VERSION: &str = "io-name-vocab-v1";

/// The maximum number of terms [`IONameVocabulary::from_value`] accepts.
///
/// Mirrors Python's `10_000` bound on a restored private-cache vocabulary.
pub const MAX_VOCABULARY_TERMS: usize = 10_000;

/// The maximum number of entries in one term's `directions`, `data_types` or `aliases`.
///
/// Mirrors Python's `1_000` bound on a restored term collection.
pub const MAX_TERM_VALUES: usize = 1_000;

/// The maximum number of aliases one term may carry.
///
/// A Rust-only bound: Python caps the restored collection at
/// [`MAX_TERM_VALUES`], which is far beyond what a deterministic plural fold can produce.
pub const MAX_VOCABULARY_ALIASES: usize = 32;

/// The direction an I/O declaration sits in.
///
/// Python passes the string `"input"`/`"output"` around; the enum keeps the default name
/// and the default `required` flag of each direction in one place.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IoDirection {
    /// A capability input.
    #[default]
    Input,
    /// A capability output.
    Output,
}

impl IoDirection {
    /// The label Python uses in audits and term evidence (`"input"`/`"output"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }

    /// The name a blank I/O declaration falls back to (`"input"`/`"result"`).
    #[must_use]
    pub fn default_name(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "result",
        }
    }

    /// The `required` flag a declaration of this direction gets by default.
    ///
    /// Python requires every input and leaves outputs unspecified.
    #[must_use]
    pub fn default_required(&self) -> Option<bool> {
        match self {
            Self::Input => Some(true),
            Self::Output => None,
        }
    }

    /// Maps a direction label to the enum; anything but `input` is an output.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        if label.trim().eq_ignore_ascii_case("input") {
            Self::Input
        } else {
            Self::Output
        }
    }
}

/// Auditable corpus evidence for one fine-grained semantic I/O name.
///
/// `count` is how many declarations carried the name, `directions` which sides of a card
/// they appeared on, `data_types` the normalized types observed for them, and `aliases`
/// the raw spellings that were conservatively folded into this term.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IONameTerm {
    /// The canonical `snake_case` term.
    pub name: String,
    /// How many I/O declarations carried this name.
    pub count: u64,
    /// The directions the name was observed in, sorted (`input`/`output`).
    pub directions: Vec<String>,
    /// The normalized data types observed for the name, sorted.
    pub data_types: Vec<String>,
    /// The spellings folded into this term, sorted.
    pub aliases: Vec<String>,
}

impl IONameTerm {
    /// Builds a term with `name` and `count` and no directions, types, or aliases.
    #[must_use]
    pub fn new(name: impl Into<String>, count: u64) -> Self {
        Self {
            name: name.into(),
            count,
            directions: Vec::new(),
            data_types: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// Sets the observed directions (sorted and deduplicated on construction).
    #[must_use]
    pub fn with_directions(
        mut self,
        directions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.directions = sorted(directions.into_iter().map(Into::into));
        self
    }

    /// Sets the observed data types (sorted and deduplicated on construction).
    #[must_use]
    pub fn with_data_types(
        mut self,
        data_types: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.data_types = sorted(
            data_types
                .into_iter()
                .map(Into::into)
                .map(|value| DataTypeVocabulary::standard().normalize(&value)),
        );
        self
    }
}

/// A deterministic I/O-name vocabulary derived from a descriptor corpus.
///
/// Build one with [`IONameVocabulary::build`] (from capability cards),
/// [`IONameVocabulary::with_observed_specs`] (from raw extracted I/O), or
/// [`builtin_io_name_vocabulary`] (the documented built-in default corpus); restore a
/// persisted one with [`IONameVocabulary::from_value`], which validates both `version` and
/// `content_hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IONameVocabulary {
    /// The schema version; always [`IO_NAME_VOCAB_VERSION`] for a vocabulary built here.
    pub version: String,
    /// The SHA-256 hex digest over the canonical JSON of the version and the terms.
    pub content_hash: String,
    /// The observed terms, sorted by name.
    pub terms: Vec<IONameTerm>,
}

impl Default for IONameVocabulary {
    /// Returns the empty vocabulary (no terms) with a valid version and hash.
    fn default() -> Self {
        Self::from_terms(Vec::new())
    }
}

impl IONameVocabulary {
    /// The empty vocabulary: no terms, so every label resolves as unseen.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Builds the vocabulary of a descriptor corpus.
    ///
    /// Every input and output name is normalized and counted, the direction and the
    /// normalized data type are recorded, and a spelling that is an unambiguous variant of
    /// another observed name is folded into it (see
    /// [`conservative_io_name_alias`]). Because folding is only recorded against
    /// canonical names and canonical names are never aliases, the resulting `aliases`
    /// lists stay empty for a self-consistent corpus — exactly Python's behaviour, where
    /// `accessories` can only alias to a term learned elsewhere.
    #[must_use]
    pub fn build(descriptors: &[CapabilityDescriptor]) -> Self {
        let vocabulary = DataTypeVocabulary::standard();
        let mut observed: BTreeMap<String, Observed> = BTreeMap::new();
        for descriptor in descriptors {
            for (direction, items) in [
                (IoDirection::Input, &descriptor.inputs),
                (IoDirection::Output, &descriptor.outputs),
            ] {
                for item in items {
                    let name = normalize_io_name(&item.name, "");
                    if name.is_empty() {
                        continue;
                    }
                    let evidence = observed.entry(name).or_default();
                    evidence.count += 1;
                    evidence.directions.insert(direction.as_str().to_string());
                    evidence
                        .data_types
                        .insert(vocabulary.normalize(&item.io_type));
                }
            }
        }
        Self::from_observations(observed)
    }

    /// Learns canonical terms and conservative aliases from raw extracted I/O.
    ///
    /// `inputs` and `outputs` accept everything [`coerce_io_entries`] accepts (a list of
    /// strings or mappings, or one mapping). A token already known to this vocabulary — as
    /// a term or as an alias — keeps its target; otherwise an unambiguous variant of a
    /// known term is folded into it; otherwise the token becomes a new term. Matching
    /// Python, the evidence is merged rather than appended, so counts never inflate.
    #[must_use]
    pub fn with_observed_specs(&self, inputs: &Value, outputs: &Value) -> Self {
        let vocabulary = DataTypeVocabulary::standard();
        let mut observed: BTreeMap<String, Observed> = BTreeMap::new();
        for (direction, raw) in [(IoDirection::Input, inputs), (IoDirection::Output, outputs)] {
            let (entries, _issues) = coerce_io_entries(raw);
            for entry in &entries {
                let Some(payload) = entry_payload(entry) else {
                    continue;
                };
                let raw_name = payload.get("name").or_else(|| payload.get("id"));
                let safe_name =
                    redact_sensitive_text(&raw_name.map_or_else(String::new, value_text));
                let token = normalize_io_name(&safe_name, "");
                if token.is_empty() {
                    continue;
                }
                let raw_type = first_present(&payload, &TYPE_KEYS);
                let data_type = vocabulary
                    .normalize(&raw_type.map_or_else(|| token.clone(), |value| type_text(value)));
                let evidence = observed.entry(token).or_default();
                evidence.count += 1;
                evidence.directions.insert(direction.as_str().to_string());
                evidence.data_types.insert(data_type);
            }
        }

        let canonical_names = self.canonical_names();
        let mut additions: Vec<IONameTerm> = Vec::with_capacity(observed.len());
        for (token, evidence) in observed {
            let target = self
                .lookup(&token)
                .or_else(|| conservative_io_name_alias(&token, &canonical_names))
                .unwrap_or_else(|| token.clone());
            let aliases = if token == target {
                Vec::new()
            } else {
                vec![token]
            };
            additions.push(IONameTerm {
                name: target,
                count: evidence.count,
                directions: sorted(evidence.directions),
                data_types: sorted(evidence.data_types),
                aliases,
            });
        }
        self.merge(&[Self::from_terms(additions)])
    }

    /// Merges this vocabulary with `others` without build-to-build count inflation.
    #[must_use]
    pub fn merge(&self, others: &[Self]) -> Self {
        let mut terms = self.terms.clone();
        for other in others {
            terms.extend(other.terms.iter().cloned());
        }
        Self::from_terms(terms)
    }

    /// The canonical term names, sorted.
    #[must_use]
    pub fn term_names(&self) -> Vec<&str> {
        self.terms.iter().map(|term| term.name.as_str()).collect()
    }

    /// The alias → term mapping of every term, sorted by alias.
    #[must_use]
    pub fn aliases(&self) -> BTreeMap<&str, &str> {
        let mut aliases = BTreeMap::new();
        for term in &self.terms {
            for alias in &term.aliases {
                aliases.insert(alias.as_str(), term.name.as_str());
            }
        }
        aliases
    }

    /// Resolves `raw` to a vocabulary term, or `None` when it is unseen.
    #[must_use]
    pub fn lookup(&self, raw: &str) -> Option<String> {
        self.resolve(raw).normalized
    }

    /// Resolves `raw`, reporting the token, the term and the method used.
    #[must_use]
    pub fn resolve(&self, raw: &str) -> IONameResolution {
        let raw = raw.trim().to_string();
        let token = normalize_io_name(&raw, "");
        let exact = self
            .terms
            .iter()
            .find(|term| term.name == token)
            .map(|term| term.name.clone());
        let normalized = exact.or_else(|| {
            self.terms
                .iter()
                .find(|term| term.aliases.iter().any(|alias| *alias == token))
                .map(|term| term.name.clone())
        });
        IONameResolution::new(raw, token, normalized)
    }

    /// Serializes the vocabulary to its canonical JSON document.
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Restores a persisted vocabulary after validating its version and hash.
    ///
    /// Terms are re-canonicalized (names normalized, evidence merged, aliases sorted and
    /// dropped when they collide with a canonical term) and the hash is recomputed, so a
    /// payload whose `content_hash` disagrees with its terms is rejected rather than
    /// silently trusted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `version` for
    /// an unsupported version, `terms` for a missing/oversized sequence, `terms[i]…` for a
    /// malformed term, and `content_hash` for a digest that does not match the terms.
    pub fn from_value(value: &Value) -> Result<Self> {
        let Value::Object(document) = value else {
            return Err(validation(
                "io_name_vocabulary",
                "expected a JSON object with `version`, `content_hash` and `terms`",
            ));
        };
        let version = document
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if version != IO_NAME_VOCAB_VERSION {
            return Err(validation(
                "version",
                format!(
                    "unsupported I/O-name vocabulary version `{version}`; expected \
                     `{IO_NAME_VOCAB_VERSION}`"
                ),
            ));
        }
        let Some(Value::Array(raw_terms)) = document.get("terms") else {
            return Err(validation("terms", "must be a JSON array"));
        };
        if raw_terms.len() > MAX_VOCABULARY_TERMS {
            return Err(validation(
                "terms",
                format!("holds more than {MAX_VOCABULARY_TERMS} terms"),
            ));
        }
        let mut terms = Vec::with_capacity(raw_terms.len());
        for (index, raw_term) in raw_terms.iter().enumerate() {
            terms.push(parse_term(raw_term, index)?);
        }
        let restored = Self::from_terms(terms);
        let expected = document
            .get("content_hash")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if expected != restored.content_hash {
            return Err(validation(
                "content_hash",
                "does not match the vocabulary terms",
            ));
        }
        Ok(restored)
    }

    /// The canonical term names as a set.
    fn canonical_names(&self) -> BTreeSet<String> {
        self.terms.iter().map(|term| term.name.clone()).collect()
    }

    /// Builds the canonical vocabulary from observed evidence.
    fn from_observations(observed: BTreeMap<String, Observed>) -> Self {
        let canonical_names: BTreeSet<String> = observed.keys().cloned().collect();
        let mut terms = Vec::with_capacity(observed.len());
        for (token, evidence) in observed {
            let target = conservative_io_name_alias(&token, &canonical_names)
                .unwrap_or_else(|| token.clone());
            let aliases = if target == token {
                Vec::new()
            } else {
                vec![token]
            };
            terms.push(IONameTerm {
                name: target,
                count: evidence.count,
                directions: sorted(evidence.directions),
                data_types: sorted(evidence.data_types),
                aliases,
            });
        }
        Self::from_terms(terms)
    }

    /// Canonicalizes `terms` and computes the content hash over them.
    fn from_terms(terms: Vec<IONameTerm>) -> Self {
        let terms = canonical_terms(terms);
        let mut document = Map::new();
        document.insert(
            "version".to_string(),
            Value::String(IO_NAME_VOCAB_VERSION.to_string()),
        );
        document.insert(
            "terms".to_string(),
            Value::Array(terms.iter().map(term_value).collect()),
        );
        let mut hasher = Sha256::new();
        hasher.update(canonical_json(&Value::Object(document)).as_bytes());
        Self {
            version: IO_NAME_VOCAB_VERSION.to_string(),
            content_hash: format!("{:x}", hasher.finalize()),
            terms,
        }
    }
}

/// The evidence gathered for one token while building a vocabulary.
#[derive(Debug, Clone, Default)]
struct Observed {
    /// How many declarations carried the token.
    count: u64,
    /// The directions it was observed in.
    directions: BTreeSet<String>,
    /// The normalized data types observed for it.
    data_types: BTreeSet<String>,
}

/// Merges duplicate terms by name and canonicalizes every collection.
fn canonical_terms(terms: Vec<IONameTerm>) -> Vec<IONameTerm> {
    let mut merged: BTreeMap<String, Observed> = BTreeMap::new();
    let mut aliases: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let vocabulary = DataTypeVocabulary::standard();
    for term in terms {
        let name = normalize_io_name(&term.name, "");
        if name.is_empty() {
            continue;
        }
        let evidence = merged.entry(name.clone()).or_default();
        evidence.count = evidence.count.max(term.count);
        for direction in term.directions {
            if direction == IoDirection::Input.as_str() || direction == IoDirection::Output.as_str()
            {
                evidence.directions.insert(direction);
            }
        }
        for data_type in term.data_types {
            if !data_type.trim().is_empty() {
                evidence.data_types.insert(vocabulary.normalize(&data_type));
            }
        }
        let term_aliases = aliases.entry(name.clone()).or_default();
        for alias in term.aliases {
            let token = normalize_io_name(&alias, "");
            if !token.is_empty() && token != name {
                term_aliases.insert(token);
            }
        }
    }

    let canonical_names: BTreeSet<String> = merged.keys().cloned().collect();
    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    for name in merged.keys() {
        if let Some(term_aliases) = aliases.get(name) {
            for alias in term_aliases {
                if !canonical_names.contains(alias) {
                    owners.entry(alias.clone()).or_insert_with(|| name.clone());
                }
            }
        }
    }

    merged
        .into_iter()
        .map(|(name, evidence)| {
            let term_aliases: Vec<String> = owners
                .iter()
                .filter(|(_, owner)| **owner == name)
                .map(|(alias, _)| alias.clone())
                .collect();
            IONameTerm {
                name,
                count: evidence.count,
                directions: sorted(evidence.directions),
                data_types: sorted(evidence.data_types),
                aliases: term_aliases,
            }
        })
        .collect()
}

/// Parses one serialized term at `index`.
fn parse_term(raw: &Value, index: usize) -> Result<IONameTerm> {
    let Value::Object(document) = raw else {
        return Err(validation(
            &format!("terms[{index}]"),
            "must be a JSON object",
        ));
    };
    let name = normalize_io_name(
        document
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        "",
    );
    if name.is_empty() {
        return Err(validation(
            &format!("terms[{index}].name"),
            "must not be blank after normalization",
        ));
    }
    let count = match document.get("count") {
        None | Some(Value::Null) => 0,
        Some(value) => value.as_u64().ok_or_else(|| {
            validation(
                &format!("terms[{index}].count"),
                "must be a non-negative integer",
            )
        })?,
    };
    let directions = read_strings(document.get("directions"), index, "directions")?;
    let data_types = read_strings(document.get("data_types"), index, "data_types")?;
    let raw_aliases = read_strings(document.get("aliases"), index, "aliases")?;
    if raw_aliases.len() > MAX_VOCABULARY_ALIASES {
        return Err(validation(
            &format!("terms[{index}].aliases"),
            format!("holds more than {MAX_VOCABULARY_ALIASES} aliases"),
        ));
    }
    let aliases = raw_aliases
        .iter()
        .filter_map(|alias| {
            let token = normalize_io_name(alias, "");
            (!token.is_empty() && token != name).then_some(token)
        })
        .collect();
    Ok(IONameTerm {
        name,
        count,
        directions,
        data_types,
        aliases,
    })
}

/// Reads one term collection, requiring a bounded array of strings.
fn read_strings(raw: Option<&Value>, index: usize, field: &str) -> Result<Vec<String>> {
    let Some(value) = raw else {
        return Ok(Vec::new());
    };
    let Value::Array(items) = value else {
        return Err(validation(
            &format!("terms[{index}].{field}"),
            "must be a JSON array of strings",
        ));
    };
    if items.len() > MAX_TERM_VALUES {
        return Err(validation(
            &format!("terms[{index}].{field}"),
            format!("holds more than {MAX_TERM_VALUES} entries"),
        ));
    }
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(validation(
                &format!("terms[{index}].{field}"),
                "must be a JSON array of strings",
            ));
        };
        values.push(text.to_string());
    }
    Ok(values)
}

/// Renders one term as the JSON object its hash is taken over.
fn term_value(term: &IONameTerm) -> Value {
    let mut document = Map::new();
    document.insert("name".to_string(), Value::String(term.name.clone()));
    document.insert("count".to_string(), Value::from(term.count));
    document.insert(
        "directions".to_string(),
        Value::Array(term.directions.iter().cloned().map(Value::String).collect()),
    );
    document.insert(
        "data_types".to_string(),
        Value::Array(term.data_types.iter().cloned().map(Value::String).collect()),
    );
    document.insert(
        "aliases".to_string(),
        Value::Array(term.aliases.iter().cloned().map(Value::String).collect()),
    );
    Value::Object(document)
}

/// Collects `values` into a sorted, deduplicated vector.
fn sorted(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let set: BTreeSet<String> = values.into_iter().collect();
    set.into_iter().collect()
}

/// Renders a JSON value as canonical JSON: object keys sorted, no whitespace.
fn canonical_json(value: &Value) -> String {
    let mut rendered = String::new();
    write_canonical(value, &mut rendered);
    rendered
}

/// Appends the canonical rendering of `value` to `output`.
fn write_canonical(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(flag) => output.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::String(text) => {
            let encoded = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string());
            output.push_str(&encoded);
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical(item, output);
            }
            output.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            output.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
                output.push_str(&encoded);
                output.push(':');
                if let Some(item) = map.get(key) {
                    write_canonical(item, output);
                }
            }
            output.push('}');
        }
    }
}

/// The documented built-in I/O-name corpus: `(name, count, directions, data types)`.
///
/// Python ships no such table — its vocabulary is always corpus-derived — so this is a
/// Rust-only default (see [`builtin_io_name_vocabulary`]).
const BUILTIN_IO_NAMES: &[(&str, u64, &[&str], &[&str])] = &[
    ("content", 3, &["input", "output"], &["text"]),
    ("count", 2, &["output"], &["integer"]),
    ("data", 3, &["output"], &["object"]),
    ("error", 2, &["output"], &["string"]),
    ("file", 2, &["input"], &["file"]),
    ("id", 2, &["input", "output"], &["string"]),
    ("input", 6, &["input"], &["text", "string"]),
    ("items", 2, &["output"], &["array"]),
    ("message", 3, &["input"], &["text"]),
    ("name", 2, &["input", "output"], &["string"]),
    ("output", 4, &["output"], &["any"]),
    ("path", 2, &["input"], &["string"]),
    ("prompt", 4, &["input"], &["text"]),
    ("query", 4, &["input"], &["string", "text"]),
    ("request", 2, &["input"], &["object"]),
    ("response", 3, &["output"], &["object"]),
    ("result", 6, &["output"], &["any"]),
    ("score", 2, &["output"], &["number"]),
    ("success", 2, &["output"], &["boolean"]),
    ("text", 3, &["input", "output"], &["text"]),
    ("url", 2, &["input"], &["string"]),
];

/// The deterministic built-in I/O-name vocabulary.
///
/// Python's `normalization.py` derives every vocabulary from a descriptor corpus and ships
/// no static I/O-name table, so this corpus is a documented Rust-only default: a
/// normalizer can run before any corpus exists, and the generic names it holds (Python's
/// own direction defaults `input` and `result`, plus the generic labels the crate's
/// examples use) resolve as known instead of being reported as unseen. Every count,
/// direction and data type is a constant of this crate, so the resulting
/// [`IONameVocabulary::content_hash`] is stable across builds and platforms.
///
/// ```
/// use openjiuwen_symphony::model::builtin_io_name_vocabulary;
///
/// let vocabulary = builtin_io_name_vocabulary();
/// assert_eq!(vocabulary.resolve("Query").normalized.as_deref(), Some("query"));
/// assert!(!vocabulary.terms.is_empty());
/// ```
#[must_use]
pub fn builtin_io_name_vocabulary() -> IONameVocabulary {
    let terms = BUILTIN_IO_NAMES
        .iter()
        .map(|(name, count, directions, data_types)| IONameTerm {
            name: (*name).to_string(),
            count: *count,
            directions: directions
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            data_types: data_types
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            aliases: Vec::new(),
        })
        .collect();
    IONameVocabulary::from_terms(terms)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::model::descriptor::CapabilityIo;
    use serde_json::json;

    fn descriptor(
        id: &str,
        inputs: &[(&str, &str)],
        outputs: &[(&str, &str)],
    ) -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", id, id)
            .with_inputs(
                inputs
                    .iter()
                    .map(|(name, io_type)| CapabilityIo::new(*name, *io_type))
                    .collect(),
            )
            .with_outputs(
                outputs
                    .iter()
                    .map(|(name, io_type)| CapabilityIo::new(*name, *io_type))
                    .collect(),
            )
    }

    #[test]
    fn build_counts_directions_and_types() {
        let vocabulary = IONameVocabulary::build(&[descriptor(
            "a",
            &[("Prompt", "str"), ("prompt", "text")],
            &[("Summary", "str")],
        )]);
        assert_eq!(vocabulary.term_names(), vec!["prompt", "summary"]);
        let prompt = &vocabulary.terms[0];
        assert_eq!(prompt.count, 2);
        assert_eq!(prompt.directions, vec!["input"]);
        assert_eq!(prompt.data_types, vec!["string", "text"]);
        assert!(prompt.aliases.is_empty());
        assert_eq!(vocabulary.terms[1].directions, vec!["output"]);
    }

    #[test]
    fn build_is_order_independent_and_hashing_is_stable() {
        let forward = IONameVocabulary::build(&[
            descriptor("a", &[("prompt", "text")], &[("summary", "str")]),
            descriptor("b", &[("query", "str")], &[("result", "text")]),
        ]);
        let reverse = IONameVocabulary::build(&[
            descriptor("b", &[("query", "str")], &[("result", "text")]),
            descriptor("a", &[("prompt", "text")], &[("summary", "str")]),
        ]);
        assert_eq!(forward.content_hash, reverse.content_hash);
        assert_eq!(forward, reverse);
        assert_eq!(forward.version, IO_NAME_VOCAB_VERSION);
        // The empty vocabulary has its own hash and still carries the version.
        assert_ne!(IONameVocabulary::empty().content_hash, forward.content_hash);
        assert!(IONameVocabulary::empty()
            .resolve("anything")
            .normalized
            .is_none());
    }

    #[test]
    fn merged_evidence_uses_max_counts() {
        let vocabulary = IONameVocabulary::build(&[descriptor("a", &[("prompt", "text")], &[])]);
        let merged = vocabulary.merge(&[vocabulary.clone()]);
        assert_eq!(merged.terms.len(), 1);
        assert_eq!(merged.terms[0].count, 1);
        assert_eq!(merged.content_hash, vocabulary.content_hash);
    }

    #[test]
    fn observed_specs_learn_aliases_and_types() {
        let vocabulary = IONameVocabulary::build(&[descriptor("a", &[("prompt", "text")], &[])]);
        let learned = vocabulary.with_observed_specs(
            &json!(["prompts", {"name": "Queries", "data_type": "str"}, "prompt"]),
            &json!({"summaries": "text"}),
        );
        assert_eq!(learned.term_names(), vec!["prompt", "queries", "summaries"]);
        let prompt = learned
            .terms
            .iter()
            .find(|term| term.name == "prompt")
            .unwrap();
        // Evidence merges with `max`, so observing the term twice does not inflate it.
        assert_eq!(prompt.count, 1);
        assert_eq!(
            learned.resolve("prompts").normalized.as_deref(),
            Some("prompt")
        );
        let queries = learned
            .terms
            .iter()
            .find(|term| term.name == "queries")
            .unwrap();
        assert_eq!(queries.data_types, vec!["string"]);
        assert_eq!(queries.directions, vec!["input"]);
    }

    #[test]
    fn serialized_vocabularies_round_trip_and_are_validated() {
        let vocabulary = IONameVocabulary::build(&[descriptor("a", &[("prompt", "text")], &[])]);
        let payload = vocabulary.to_value();
        assert_eq!(IONameVocabulary::from_value(&payload).unwrap(), vocabulary);

        let mut tampered = payload.clone();
        tampered["version"] = json!("io-name-vocab-v2");
        let error = IONameVocabulary::from_value(&tampered).unwrap_err();
        assert!(error.to_string().contains("version"), "{error}");

        let mut tampered = payload.clone();
        tampered["content_hash"] = json!("0".repeat(64));
        let error = IONameVocabulary::from_value(&tampered).unwrap_err();
        assert!(error.to_string().contains("content_hash"), "{error}");

        let mut tampered = payload.clone();
        tampered["terms"] = json!("prompt");
        let error = IONameVocabulary::from_value(&tampered).unwrap_err();
        assert!(error.to_string().contains("terms"), "{error}");

        let mut tampered = payload;
        tampered["terms"] = json!([{"name": "   ", "count": 1}]);
        let error = IONameVocabulary::from_value(&tampered).unwrap_err();
        assert!(error.to_string().contains("terms[0].name"), "{error}");
    }

    #[test]
    fn builtin_corpus_is_deterministic() {
        let first = builtin_io_name_vocabulary();
        let second = builtin_io_name_vocabulary();
        assert_eq!(first, second);
        assert_eq!(first.version, IO_NAME_VOCAB_VERSION);
        assert_eq!(
            first.resolve("Result").normalized.as_deref(),
            Some("result")
        );
        assert_eq!(first.resolve("nope").normalized, None);
        assert_eq!(
            IONameVocabulary::from_value(&first.to_value()).unwrap(),
            first
        );
    }

    #[test]
    fn directions_map_labels_to_defaults() {
        assert_eq!(IoDirection::from_label("INPUT"), IoDirection::Input);
        assert_eq!(IoDirection::from_label("output"), IoDirection::Output);
        assert_eq!(IoDirection::from_label(""), IoDirection::Output);
        assert_eq!(IoDirection::Input.default_name(), "input");
        assert_eq!(IoDirection::Output.default_name(), "result");
        assert_eq!(IoDirection::Input.default_required(), Some(true));
        assert_eq!(IoDirection::Output.default_required(), None);
    }
}
