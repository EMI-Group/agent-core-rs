//! Per-declaration normalization: name, metadata, data type, flag and default.
//!
//! Everything here is reached through `normalize_entry`, which one declaration of a raw
//! I/O collection calls; the collector accumulates the audit trail. Each helper appends to
//! the decision and issue vectors it is given, so a declaration produces the same records
//! regardless of the collection shape it arrived in.

use serde_json::{Map, Value};

use super::{
    is_sensitive_name, json_safe, normalize_io_name, redact_sensitive_json, redact_sensitive_text,
    resolve_data_type, to_bool, type_token, BTreeMap, CapabilityIo, DataTypeResolution,
    IoDirection, NormalizationDecision, NormalizationIssue, DATA_TYPE_VOCABULARY_VERSION,
    FIELD_DEFAULT, FIELD_METADATA, FIELD_NAME, FIELD_REQUIRED, FIELD_TYPE, IO_NAME_VOCAB_VERSION,
    ISSUE_CODE_INVALID_IO_ENTRY, ISSUE_CODE_INVALID_REQUIRED, ISSUE_CODE_IO_NAME_NOT_IN_VOCABULARY,
    ISSUE_CODE_MISSING_IO_NAME, ISSUE_CODE_SENSITIVE_DEFAULT_REDACTED,
    ISSUE_CODE_SENSITIVE_IO_NAME_REDACTED, ISSUE_CODE_SENSITIVE_METADATA_REDACTED,
    ISSUE_CODE_UNKNOWN_DATA_TYPE, METHOD_COERCED, METHOD_DEFAULTED, METHOD_REDACTED, METHOD_SYNTAX,
    METHOD_VOCAB_FALLBACK, REDACTED, TYPE_KEYS, TYPE_OBJECT_KEYS, VOCABULARY_DATA_TYPE,
    VOCABULARY_IO_NAME, VOCABULARY_REDACTION,
};
use crate::model::normalization::vocabulary::IONameVocabulary;

/// Returns the payload mapping of one entry, or `None` when the entry is not spec-shaped.
///
/// A plain string becomes `{"name": <string>, "type": <string>}`, mirroring Python.
pub(crate) fn entry_payload(entry: &Value) -> Option<Map<String, Value>> {
    match entry {
        Value::String(text) => {
            let mut map = Map::new();
            map.insert("name".to_string(), Value::String(text.clone()));
            map.insert("type".to_string(), Value::String(text.clone()));
            Some(map)
        }
        Value::Object(map) => Some(map.clone()),
        _ => None,
    }
}

/// Returns the first non-blank value among `keys`.
pub(crate) fn first_present<'a>(
    payload: &'a Map<String, Value>,
    keys: &[&str],
) -> Option<&'a Value> {
    for key in keys {
        if let Some(value) = payload.get(*key) {
            if !value_text(value).trim().is_empty() {
                return Some(value);
            }
        }
    }
    None
}

/// Renders a JSON leaf as the text Python's `str(...)` would produce for it.
///
/// `null` becomes the empty string, scalars their usual rendering, and containers their
/// canonical JSON (Python would show a `dict`/`list` repr, including object addresses for
/// opaque values; JSON keeps this deterministic).
pub(crate) fn value_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Renders a raw data type declaration, following Python's `_raw_data_type`.
///
/// A mapping is read through its `format`, `contentMediaType` and `type` members (in that
/// order); anything else is rendered as text. The result is redacted and trimmed.
pub(crate) fn type_text(value: &Value) -> String {
    let text = match value {
        Value::Object(map) => {
            first_present(map, &TYPE_OBJECT_KEYS).map_or_else(String::new, value_text)
        }
        other => value_text(other),
    };
    redact_sensitive_text(text.trim())
}

/// The vocabulary version to record in a decision.
pub(crate) fn vocabulary_version(vocabulary: &IONameVocabulary) -> String {
    if vocabulary.version.is_empty() {
        IO_NAME_VOCAB_VERSION.to_string()
    } else {
        vocabulary.version.clone()
    }
}

/// Where one declaration sits, for the audit trail.
struct EntrySite<'a> {
    /// The zero-based position of the declaration.
    index: usize,
    /// The direction being normalized.
    direction: IoDirection,
    /// The I/O-name vocabulary version the decisions record.
    version: &'a str,
}

impl EntrySite<'_> {
    /// Builds an issue for this declaration.
    fn issue(&self, code: &str, message: impl Into<String>) -> NormalizationIssue {
        NormalizationIssue::new(code, message)
            .with_direction(self.direction)
            .with_index(self.index)
    }

    /// Builds a decision about `field` reached by `method`.
    fn decision(&self, field: &str, method: &str) -> NormalizationDecision {
        NormalizationDecision::new(self.direction, self.index, field, method)
    }
}

/// The resolved identity of one declaration's I/O name.
struct EntryName {
    /// The name emitted into the declaration (redacted when it is sensitive).
    name: String,
    /// The name before redaction, used as the data-type fallback.
    resolved: String,
    /// Whether the name is itself credential material.
    sensitive: bool,
}

/// Normalizes one payload into a [`CapabilityIo`] plus its audit trail.
pub(crate) fn normalize_entry(
    entry: &Value,
    index: usize,
    direction: IoDirection,
    version: &str,
    vocabulary: &IONameVocabulary,
    default_required: Option<bool>,
) -> (
    Option<CapabilityIo>,
    Vec<NormalizationDecision>,
    Vec<NormalizationIssue>,
) {
    let mut decisions: Vec<NormalizationDecision> = Vec::new();
    let mut issues: Vec<NormalizationIssue> = Vec::new();
    let site = EntrySite {
        index,
        direction,
        version,
    };
    let Some(payload) = entry_payload(entry) else {
        issues.push(site.issue(
            ISSUE_CODE_INVALID_IO_ENTRY,
            "I/O entry must be a string or mapping",
        ));
        return (None, decisions, issues);
    };

    let resolved = resolve_entry_name(&payload, &site, vocabulary, &mut decisions, &mut issues);
    let mut metadata = normalize_entry_metadata(&payload, &site, &mut decisions, &mut issues);
    let resolution = resolve_entry_type(
        &payload,
        &site,
        &resolved.resolved,
        &metadata,
        &mut decisions,
    );
    let description = payload
        .get("description")
        .or_else(|| payload.get("help"))
        .map_or_else(String::new, value_text)
        .trim()
        .to_string();
    let required = normalize_required(
        &payload,
        &site,
        default_required,
        &mut decisions,
        &mut issues,
    );
    let default = normalize_default(
        &payload,
        &site,
        &mut metadata,
        resolved.sensitive,
        &mut decisions,
        &mut issues,
    );
    if !resolution.known() && !resolution.raw.is_empty() {
        metadata.insert(
            "raw_type".to_string(),
            Value::String(resolution.raw.clone()),
        );
        issues.push(site.issue(
            ISSUE_CODE_UNKNOWN_DATA_TYPE,
            format!(
                "unsupported data type '{}' normalized to '{}'",
                resolution.raw, resolution.normalized
            ),
        ));
    }

    let item = CapabilityIo {
        name: resolved.name,
        io_type: resolution.normalized,
        description,
        required,
        default,
        metadata,
    };
    (Some(item), decisions, issues)
}

/// Resolves one declaration's name, its method, and its audit records.
fn resolve_entry_name(
    payload: &Map<String, Value>,
    site: &EntrySite<'_>,
    vocabulary: &IONameVocabulary,
    decisions: &mut Vec<NormalizationDecision>,
    issues: &mut Vec<NormalizationIssue>,
) -> EntryName {
    let raw_name_text = payload
        .get("name")
        .or_else(|| payload.get("id"))
        .map_or_else(String::new, value_text);
    let raw_name_text = redact_sensitive_text(raw_name_text.trim());
    let token = normalize_io_name(&raw_name_text, "");
    let resolution = vocabulary.resolve(&raw_name_text);
    let corpus = !vocabulary.terms.is_empty();
    let resolved = if let Some(found) = resolution.normalized.clone() {
        found
    } else {
        let fallback = if token.is_empty() {
            site.direction.default_name().to_string()
        } else {
            token.clone()
        };
        if token.is_empty() {
            issues.push(site.issue(
                ISSUE_CODE_MISSING_IO_NAME,
                format!("missing I/O name normalized to '{fallback}'"),
            ));
        } else if corpus {
            issues.push(site.issue(
                ISSUE_CODE_IO_NAME_NOT_IN_VOCABULARY,
                format!("I/O name '{token}' was not observed in the corpus vocabulary"),
            ));
        }
        fallback
    };
    let method = if resolution.known {
        resolution.method.clone()
    } else if corpus {
        METHOD_VOCAB_FALLBACK.to_string()
    } else {
        METHOD_SYNTAX.to_string()
    };
    let sensitive = is_sensitive_name(&resolved);
    let name = if sensitive {
        REDACTED.to_string()
    } else {
        resolved.clone()
    };
    let token_is_sensitive = !token.is_empty() && is_sensitive_name(&token);
    let mut decision = site
        .decision(FIELD_NAME, &method)
        .with_raw_value(&raw_name_text)
        .with_token(if sensitive || token_is_sensitive {
            REDACTED
        } else {
            token.as_str()
        })
        .with_normalized_value(&name)
        .with_vocabulary(VOCABULARY_IO_NAME, site.version);
    if corpus {
        decision = decision
            .with_detail("vocabulary_hash", vocabulary.content_hash.clone())
            .with_detail("observed", resolution.known);
    }
    decisions.push(decision);
    if sensitive {
        decisions.push(
            site.decision(FIELD_NAME, METHOD_REDACTED)
                .with_token(REDACTED)
                .with_normalized_value(REDACTED)
                .with_vocabulary(VOCABULARY_IO_NAME, site.version)
                .with_detail("sensitive", true),
        );
        issues.push(site.issue(
            ISSUE_CODE_SENSITIVE_IO_NAME_REDACTED,
            "sensitive I/O name was redacted",
        ));
    }
    EntryName {
        name,
        resolved,
        sensitive,
    }
}

/// Normalizes, redacts and audits the metadata member of one declaration.
fn normalize_entry_metadata(
    payload: &Map<String, Value>,
    site: &EntrySite<'_>,
    decisions: &mut Vec<NormalizationDecision>,
    issues: &mut Vec<NormalizationIssue>,
) -> BTreeMap<String, Value> {
    let metadata = normalize_metadata(payload.get("metadata"));
    let Some(Value::Object(raw_metadata)) = payload.get("metadata") else {
        return metadata;
    };
    let mut keys: Vec<&String> = raw_metadata.keys().collect();
    keys.sort();
    for key in keys {
        if !is_sensitive_name(key) {
            continue;
        }
        decisions.push(
            site.decision(FIELD_METADATA, METHOD_REDACTED)
                .with_token(REDACTED)
                .with_normalized_value(REDACTED)
                .with_vocabulary(VOCABULARY_REDACTION, site.version)
                .with_detail("key", key.clone()),
        );
        issues.push(
            site.issue(
                ISSUE_CODE_SENSITIVE_METADATA_REDACTED,
                format!("sensitive metadata key '{key}' was redacted"),
            )
            .with_detail("key", key.clone()),
        );
    }
    metadata
}

/// Resolves and audits the data type of one declaration.
fn resolve_entry_type(
    payload: &Map<String, Value>,
    site: &EntrySite<'_>,
    resolved_name: &str,
    metadata: &BTreeMap<String, Value>,
    decisions: &mut Vec<NormalizationDecision>,
) -> DataTypeResolution {
    let mut resolution = match first_present(payload, &TYPE_KEYS) {
        Some(value) => resolve_data_type(&type_text(value)),
        None => resolve_data_type(resolved_name),
    };
    // Python replays a preserved raw type when the declaration resolves to its unknown
    // sentinel; this port generalizes the rule to any unresolved type.
    if !resolution.known() && !resolution.raw.is_empty() {
        if let Some(preserved) = metadata
            .get("raw_type")
            .filter(|value| !value_text(value).trim().is_empty())
        {
            let preserved = value_text(preserved);
            resolution = resolve_data_type(&preserved);
        }
    }
    decisions.push(
        site.decision(FIELD_TYPE, resolution.source.as_str())
            .with_raw_value(&resolution.raw)
            .with_token(type_token(&resolution.raw))
            .with_normalized_value(&resolution.normalized)
            .with_vocabulary(VOCABULARY_DATA_TYPE, DATA_TYPE_VOCABULARY_VERSION)
            .with_detail("known", resolution.known()),
    );
    resolution
}

/// Normalizes and audits the default value of one declaration.
fn normalize_default(
    payload: &Map<String, Value>,
    site: &EntrySite<'_>,
    metadata: &mut BTreeMap<String, Value>,
    sensitive_name: bool,
    decisions: &mut Vec<NormalizationDecision>,
    issues: &mut Vec<NormalizationIssue>,
) -> Option<Value> {
    let already_redacted = metadata
        .get("default_redacted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let issue_previous = |issues: &mut Vec<NormalizationIssue>| {
        issues.push(
            site.issue(
                ISSUE_CODE_SENSITIVE_DEFAULT_REDACTED,
                "sensitive I/O default was not retained",
            )
            .with_detail("reason", "previous_pass"),
        );
    };
    let Some(raw_default) = payload.get("default").filter(|value| !value.is_null()) else {
        if already_redacted {
            issue_previous(issues);
        }
        return None;
    };
    if sensitive_name {
        metadata.insert("default_redacted".to_string(), Value::Bool(true));
        issues.push(
            site.issue(
                ISSUE_CODE_SENSITIVE_DEFAULT_REDACTED,
                "sensitive I/O default was not retained",
            )
            .with_detail("reason", "sensitive_io_name"),
        );
        decisions.push(
            site.decision(FIELD_DEFAULT, METHOD_REDACTED)
                .with_vocabulary(VOCABULARY_REDACTION, site.version)
                .with_detail("reason", "sensitive_io_name"),
        );
        return None;
    }
    let plain = json_safe(raw_default, 0);
    let sanitized = redact_sensitive_json(&plain);
    if sanitized != plain {
        decisions.push(
            site.decision(FIELD_DEFAULT, METHOD_REDACTED)
                .with_vocabulary(VOCABULARY_REDACTION, site.version)
                .with_detail("reason", "credential_value"),
        );
    }
    if already_redacted {
        issue_previous(issues);
    }
    Some(sanitized)
}

/// Normalizes the `required`/`optional`/`default` trio into one optional flag.
fn normalize_required(
    payload: &Map<String, Value>,
    site: &EntrySite<'_>,
    default: Option<bool>,
    decisions: &mut Vec<NormalizationDecision>,
    issues: &mut Vec<NormalizationIssue>,
) -> Option<bool> {
    if let Some(raw) = payload.get("required") {
        if raw.is_null() {
            return None;
        }
        // Parse without a fallback first, so an unparseable value is reported instead of
        // silently inheriting the direction default.
        let coerced = to_bool(raw, None);
        if coerced.is_none() {
            issues.push(
                site.issue(
                    ISSUE_CODE_INVALID_REQUIRED,
                    "unparseable `required` value, fell back to the direction default",
                )
                .with_detail("raw_value", value_text(raw)),
            );
            decisions.push(
                site.decision(FIELD_REQUIRED, METHOD_DEFAULTED)
                    .with_raw_value(value_text(raw))
                    .with_normalized_value(flag_text(default))
                    .with_detail("reason", "invalid_value"),
            );
            return default;
        }
        if !raw.is_boolean() {
            decisions.push(
                site.decision(FIELD_REQUIRED, METHOD_COERCED)
                    .with_raw_value(value_text(raw))
                    .with_normalized_value(flag_text(coerced)),
            );
        }
        return coerced;
    }
    if let Some(raw) = payload.get("optional") {
        if raw.is_null() {
            return None;
        }
        let optional = to_bool(raw, None);
        if optional.is_none() {
            issues.push(
                site.issue(
                    ISSUE_CODE_INVALID_REQUIRED,
                    "unparseable `optional` value, left unspecified",
                )
                .with_detail("raw_value", value_text(raw)),
            );
            return None;
        }
        let required = optional.map(|optional| !optional);
        if !raw.is_boolean() {
            decisions.push(
                site.decision(FIELD_REQUIRED, METHOD_COERCED)
                    .with_raw_value(value_text(raw))
                    .with_normalized_value(flag_text(required))
                    .with_detail("source", "optional"),
            );
        }
        return required;
    }
    if payload.contains_key("default") {
        decisions.push(
            site.decision(FIELD_REQUIRED, METHOD_DEFAULTED)
                .with_normalized_value("false")
                .with_detail("reason", "default_present"),
        );
        return Some(false);
    }
    decisions.push(
        site.decision(FIELD_REQUIRED, METHOD_DEFAULTED)
            .with_normalized_value(flag_text(default))
            .with_detail("reason", "direction_default"),
    );
    default
}

/// Renders an optional flag for the audit trail.
fn flag_text(flag: Option<bool>) -> String {
    match flag {
        Some(true) => "true".to_string(),
        Some(false) => "false".to_string(),
        None => "unspecified".to_string(),
    }
}

/// Normalizes the metadata mapping of one declaration into sorted, redacted JSON values.
fn normalize_metadata(raw: Option<&Value>) -> BTreeMap<String, Value> {
    let Some(Value::Object(map)) = raw else {
        return BTreeMap::new();
    };
    match redact_sensitive_json(&json_safe(&Value::Object(map.clone()), 0)) {
        Value::Object(redacted) => redacted.into_iter().collect(),
        _ => BTreeMap::new(),
    }
}
