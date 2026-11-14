//! Stable model and metadata identity fingerprints for the LLM stages.
//!
//! Python's `symphony` keys its persistent relation-match cache on a *matcher
//! identity*: the configured model, the sanitized request/client configuration, and
//! every matcher knob that can change an answer. This module ports the identity half
//! of that machinery:
//!
//! - [`ModelIdentity`] — the model name, the backend, the sampling settings, and two
//!   SHA-256 digests over the request and client configuration;
//!   see [`ModelIdentity::of`].
//! - [`stable_metadata_sha256`] — the canonical (sorted-key, compact) JSON digest
//!   Python's `stable_metadata_sha256` produces.
//! - [`endpoint_sha256`] — a normalized digest of an endpoint, so a base URL can be
//!   part of a cache key without the URL (or its query string) ever being stored.
//! - [`sanitize_metadata`] — Python's recursive metadata sanitizer: credentials are
//!   dropped, endpoints are replaced by their digest, and everything else is kept in
//!   its original key order.
//! - [`graph_identity_sha256`] — the per-capability graph identity hash used for the
//!   source/target halves of a relation-match cache key.
//! - [`canonical_json`] — the sorted-key, whitespace-free rendering every digest here
//!   is taken over.
//!
//! Every function is total: malformed input produces a deterministic digest, never a
//! panic, and container recursion stops at [`MAX_METADATA_DEPTH`].
//!
//! # Determinism
//!
//! Object keys are always sorted before rendering, so a digest never depends on JSON
//! map order (the workspace enables `serde_json`'s `preserve_order`, so this is real
//! work, not a no-op). Non-finite floats — which `serde_json` cannot represent and maps
//! to `null` anyway — render as `null` instead of panicking.
use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::model::CapabilityDescriptor;
use crate::redaction::{is_sensitive_name, normalize_sensitive_name};

use super::{CompletionModel, ModelConfig};

/// The version tag mixed into every [`ModelIdentity`] digest.
///
/// Bumping it invalidates every cache key derived from an identity, which is the
/// intended escape hatch when the digest inputs themselves change shape.
pub const IDENTITY_VERSION: &str = "Symphony-model-identity-v1";

/// The maximum container depth [`sanitize_metadata`] and [`canonical_json`] descend
/// into.
///
/// Deeper containers are replaced by `null` (with a `tracing::warn!`), so a
/// pathological document cannot exhaust the stack.
pub const MAX_METADATA_DEPTH: usize = 64;

/// The backend recorded when no model is configured.
const NO_MODEL_BACKEND: &str = "none";

/// Key names that look like `key` fields but never carry credential material
/// (Python's `_SAFE_KEY_FIELDS`); they are kept verbatim.
const SAFE_KEY_FIELDS: [&str; 5] = [
    "cache_key",
    "key_type",
    "partition_key",
    "public_key",
    "routing_key",
];

/// Key names whose value is an endpoint (Python's `_ENDPOINT_FIELDS`).
const ENDPOINT_FIELDS: [&str; 5] = [
    "api_base",
    "api_url",
    "base_url",
    "endpoint",
    "endpoint_url",
];

/// Normalized key suffixes that name credential material.
const CREDENTIAL_SUFFIXES: [&str; 4] = ["_key", "secret", "token", "password"];

/// The stable identity of the model behind an LLM stage.
///
/// The two digests are SHA-256 hex strings over [`canonical_json`]:
///
/// - `request_config_sha256` covers [`IDENTITY_VERSION`], `temperature`, `top_p`,
///   `max_tokens`, and `stop`;
/// - `client_config_sha256` covers [`IDENTITY_VERSION`], `backend`, and `model`.
///
/// Field order is fixed by this declaration, so the serialized identity is itself
/// byte-stable and safe to embed in a cache key. `model` and `backend` fall back to
/// `"none"` when the configuration carries no model or an explicitly blank backend.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelIdentity {
    /// The completion-model name, or `"none"` when no model is configured.
    pub model: String,
    /// The provider/backend name, or `"none"` when it is blank.
    pub backend: String,
    /// The sampling temperature, when one is configured.
    pub temperature: Option<f64>,
    /// The nucleus-sampling probability, when one is configured.
    pub top_p: Option<f64>,
    /// The generated-token cap, when one is configured.
    pub max_tokens: Option<u64>,
    /// The stop sequences sent to the model.
    pub stop: Vec<String>,
    /// The SHA-256 of the canonical request configuration.
    pub request_config_sha256: String,
    /// The SHA-256 of the canonical client configuration.
    pub client_config_sha256: String,
}

impl ModelIdentity {
    /// Derives the identity of `config`, whose backend is named `backend`.
    ///
    /// [`ModelConfig`] carries only a temperature and a max-token hint; `top_p` and
    /// `stop` therefore start as `None`/empty and are filled in by
    /// [`ModelIdentity::with_top_p`] and [`ModelIdentity::with_stop`]. A non-finite
    /// sampling value is treated as absent, so it renders as `null` and the digest
    /// stays stable.
    #[must_use]
    pub fn of(config: &ModelConfig, backend: &str) -> Self {
        let model = config
            .model
            .as_deref()
            .map_or(NO_MODEL_BACKEND, CompletionModel::name);
        let backend = backend.trim();
        let mut identity = Self {
            model: model.to_string(),
            backend: if backend.is_empty() {
                NO_MODEL_BACKEND.to_string()
            } else {
                backend.to_string()
            },
            temperature: finite(config.temperature.map(f64::from)),
            top_p: None,
            max_tokens: config
                .max_tokens
                .and_then(|value| u64::try_from(value).ok()),
            stop: Vec::new(),
            request_config_sha256: String::new(),
            client_config_sha256: String::new(),
        };
        identity.refresh_request_digest();
        identity.client_config_sha256 = stable_metadata_sha256(&identity.client_document());
        identity
    }

    /// Sets the nucleus-sampling probability and refreshes the request digest.
    ///
    /// A non-finite value is dropped (the digest then covers `null`).
    #[must_use]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = finite(Some(top_p));
        self.refresh_request_digest();
        self
    }

    /// Replaces the stop sequences and refreshes the request digest.
    #[must_use]
    pub fn with_stop(mut self, stop: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.stop = stop.into_iter().map(Into::into).collect();
        self.refresh_request_digest();
        self
    }

    /// Recomputes `request_config_sha256` from the current sampling fields.
    fn refresh_request_digest(&mut self) {
        self.request_config_sha256 = stable_metadata_sha256(&self.request_document());
    }

    /// The canonical request configuration the request digest covers.
    fn request_document(&self) -> Value {
        let mut document = Map::new();
        document.insert(
            "identity_version".to_string(),
            Value::String(IDENTITY_VERSION.to_string()),
        );
        document.insert("temperature".to_string(), optional_number(self.temperature));
        document.insert("top_p".to_string(), optional_number(self.top_p));
        document.insert(
            "max_tokens".to_string(),
            self.max_tokens.map_or(Value::Null, Value::from),
        );
        document.insert(
            "stop".to_string(),
            Value::Array(self.stop.iter().cloned().map(Value::String).collect()),
        );
        Value::Object(document)
    }

    /// The canonical client configuration the client digest covers.
    fn client_document(&self) -> Value {
        let mut document = Map::new();
        document.insert(
            "identity_version".to_string(),
            Value::String(IDENTITY_VERSION.to_string()),
        );
        document.insert("backend".to_string(), Value::String(self.backend.clone()));
        document.insert("model".to_string(), Value::String(self.model.clone()));
        Value::Object(document)
    }
}

/// Keeps only finite sampling values.
fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|item| item.is_finite())
}

/// Renders an optional number as a JSON number or `null`.
fn optional_number(value: Option<f64>) -> Value {
    value.map_or(Value::Null, Value::from)
}

/// Renders `value` as canonical JSON: object keys sorted recursively, no whitespace.
///
/// Arrays keep their order and scalars keep their lexical form, so two values that
/// differ only in object-key order render byte-identically. Containers nested deeper
/// than [`MAX_METADATA_DEPTH`] render as `null`, and a non-finite number (unreachable
/// through `serde_json`, but possible through a hand-built `Number`) also renders as
/// `null`; neither case panics.
#[must_use]
pub fn canonical_json(value: &Value) -> String {
    let mut rendered = String::new();
    render_canonical(value, &mut rendered, 0);
    rendered
}

/// Appends the canonical rendering of `value` to `rendered`.
fn render_canonical(value: &Value, rendered: &mut String, depth: usize) {
    if depth > MAX_METADATA_DEPTH {
        rendered.push_str("null");
        return;
    }
    match value {
        Value::Null => rendered.push_str("null"),
        Value::Bool(flag) => rendered.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => {
            if number.as_f64().is_some_and(|item| !item.is_finite()) {
                rendered.push_str("null");
            } else {
                rendered.push_str(&number.to_string());
            }
        }
        Value::String(text) => match serde_json::to_string(text) {
            Ok(escaped) => rendered.push_str(&escaped),
            Err(_) => rendered.push_str("\"\""),
        },
        Value::Array(items) => {
            rendered.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                }
                render_canonical(item, rendered, depth + 1);
            }
            rendered.push(']');
        }
        Value::Object(map) => {
            let sorted: BTreeMap<&str, &Value> =
                map.iter().map(|(key, item)| (key.as_str(), item)).collect();
            rendered.push('{');
            for (index, (key, item)) in sorted.iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                }
                match serde_json::to_string(key) {
                    Ok(escaped) => rendered.push_str(&escaped),
                    Err(_) => rendered.push_str("\"\""),
                }
                rendered.push(':');
                render_canonical(item, rendered, depth + 1);
            }
            rendered.push('}');
        }
    }
}

/// Returns the SHA-256 hex digest of the canonical JSON of `value`.
///
/// This is Python's `stable_metadata_sha256`: sorted keys, compact separators, and a
/// `null` rendering for anything unrepresentable. Malformed or deeply nested input
/// still yields a deterministic digest.
#[must_use]
pub fn stable_metadata_sha256(value: &Value) -> String {
    sha256_hex(&canonical_json(value))
}

/// Returns the SHA-256 hex digest of the normalized `endpoint`.
///
/// Normalization mirrors Python's `_normalize_endpoint` closely enough for a cache
/// key: the text is trimmed, the scheme and host are lowercased, userinfo is dropped,
/// a default port (`https:443`, `http:80`) is dropped, the path's trailing slashes are
/// stripped, and the fragment is ignored. The **query string is excluded** — unlike
/// Python's sorted-query normalization — so a signed URL and its bare endpoint hash
/// identically and no query parameter ever influences identity. Text without a
/// `scheme://` prefix is only trimmed and stripped of trailing slashes.
///
/// An input whose authority cannot be parsed (for example an unbracketed IPv6 host or
/// an out-of-range port) falls back to the trimmed, slash-stripped text, exactly like
/// Python's `ValueError` branch.
#[must_use]
pub fn endpoint_sha256(endpoint: &str) -> String {
    sha256_hex(&normalize_endpoint(endpoint))
}

/// Normalizes one endpoint string; see [`endpoint_sha256`].
fn normalize_endpoint(endpoint: &str) -> String {
    let text = endpoint.trim();
    let stripped = text.trim_end_matches('/');
    let Some((scheme, rest)) = text.split_once("://") else {
        return stripped.to_string();
    };
    let scheme = scheme.trim().to_ascii_lowercase();
    let valid_scheme = !scheme.is_empty()
        && scheme
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "+-.".contains(character));
    if !valid_scheme {
        return stripped.to_string();
    }
    // The fragment and the query string are both excluded.
    let rest = rest.split(['#', '?']).next().unwrap_or("");
    let (authority, path) = match rest.find('/') {
        Some(index) => rest.split_at(index),
        None => (rest, ""),
    };
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    let Ok((host, port)) = split_host_port(host_port) else {
        return stripped.to_string();
    };
    let host = host.trim().to_ascii_lowercase();
    if host.is_empty() {
        return stripped.to_string();
    }
    let default_port = matches!(
        (scheme.as_str(), port),
        ("https", Some(443)) | ("http", Some(80))
    );
    let authority = match port {
        Some(port) if !default_port => format!("{host}:{port}"),
        _ => host,
    };
    format!("{scheme}://{authority}{}", path.trim_end_matches('/'))
}

/// Splits an authority into host (brackets kept) and an optional port.
///
/// A bracketed IPv6 host must be closed with `]`; a non-numeric or out-of-range port is
/// an error, so the caller can fall back to the raw text.
fn split_host_port(authority: &str) -> Result<(&str, Option<u16>), ()> {
    if let Some(rest) = authority.strip_prefix('[') {
        let Some((inner, tail)) = rest.split_once(']') else {
            return Err(());
        };
        let host = &authority[..inner.len() + 2];
        let port = match tail.strip_prefix(':') {
            Some(text) if !text.is_empty() => Some(text.parse::<u16>().map_err(|_| ())?),
            _ => None,
        };
        return Ok((host, port));
    }
    match authority.rsplit_once(':') {
        Some((host, text)) if !text.is_empty() => {
            Ok((host, Some(text.parse::<u16>().map_err(|_| ())?)))
        }
        Some((host, _)) => Ok((host, None)),
        None => Ok((authority, None)),
    }
}

/// Returns the graph identity digest of `descriptor`.
///
/// Python's `graph_identity_dict` covers `type`, `id`, `name`, `description`, `version`,
/// `inputs`, and `outputs`; the Rust [`CapabilityDescriptor`] has no `version` field, so
/// the digest covers `capability_id`, `capability_type`, `name`, `description`, `inputs`,
/// and `outputs`. That is exactly the information a relation match can depend on: a
/// changed IO list, name, or description invalidates the cache entry, while metadata and
/// availability do not.
#[must_use]
pub fn graph_identity_sha256(descriptor: &CapabilityDescriptor) -> String {
    let value = serde_json::json!({
        "capability_id": descriptor.capability_id,
        "capability_type": descriptor.capability_type,
        "name": descriptor.name,
        "description": descriptor.description,
        "inputs": descriptor.inputs,
        "outputs": descriptor.outputs,
    });
    stable_metadata_sha256(&value)
}

/// Recursively sanitizes metadata for storage or hashing.
///
/// Ports `openjiuwen.symphony.shared.identity.sanitize_metadata`:
///
/// - an endpoint key (`api_base`, `api_url`, `base_url`, `endpoint`, `endpoint_url`) is
///   replaced by `<key>_sha256` holding
///   [`endpoint_sha256`] of its string value (a non-string value is hashed canonically,
///   and a `null` value is dropped);
/// - a key [`is_sensitive_name`] recognizes — or one whose normalized form ends in
///   `_key`, `secret`, `token`, or `password` — is dropped entirely;
/// - the five safe key fields (`cache_key`, `key_type`, `partition_key`, `public_key`,
///   `routing_key`) are exceptions that never count as credentials;
/// - every other key is kept in its original position with its value sanitized
///   recursively (objects and arrays included).
///
/// Containers nested deeper than [`MAX_METADATA_DEPTH`] become `null` with a
/// `tracing::warn!`; a non-object top level is sanitized as a value. Key order is
/// preserved.
#[must_use]
pub fn sanitize_metadata(value: &Value) -> Value {
    sanitize_at(value, 0)
}

/// Sanitizes one JSON node at `depth`.
fn sanitize_at(value: &Value, depth: usize) -> Value {
    match value {
        Value::Object(map) => {
            if depth >= MAX_METADATA_DEPTH {
                tracing::warn!(
                    depth,
                    "metadata sanitizer: depth cap reached; the container became null"
                );
                return Value::Null;
            }
            let mut sanitized = Map::with_capacity(map.len());
            for (key, item) in map {
                if let Some((key, item)) = sanitize_entry(key, item, depth) {
                    sanitized.insert(key, item);
                }
            }
            Value::Object(sanitized)
        }
        Value::Array(items) => {
            if depth >= MAX_METADATA_DEPTH {
                tracing::warn!(
                    depth,
                    "metadata sanitizer: depth cap reached; the container became null"
                );
                return Value::Null;
            }
            Value::Array(
                items
                    .iter()
                    .map(|item| sanitize_at(item, depth + 1))
                    .collect(),
            )
        }
        other => other.clone(),
    }
}

/// Sanitizes one key/value pair, answering `None` when the key must be dropped.
fn sanitize_entry(key: &str, value: &Value, depth: usize) -> Option<(String, Value)> {
    let normalized = normalize_sensitive_name(key);
    if ENDPOINT_FIELDS.contains(&normalized.as_str()) {
        if value.is_null() {
            return None;
        }
        let digest = match value {
            Value::String(text) => endpoint_sha256(text),
            other => stable_metadata_sha256(other),
        };
        return Some((format!("{key}_sha256"), Value::String(digest)));
    }
    if !SAFE_KEY_FIELDS.contains(&normalized.as_str()) && is_credential_name(&normalized) {
        return None;
    }
    Some((key.to_string(), sanitize_at(value, depth + 1)))
}

/// Reports whether a *normalized* key names credential material.
fn is_credential_name(normalized: &str) -> bool {
    is_sensitive_name(normalized)
        || CREDENTIAL_SUFFIXES
            .iter()
            .any(|suffix| normalized.ends_with(suffix))
}

/// Returns the lowercase hex SHA-256 of `text`.
fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::llm::{FallbackMode, StaticCompletionModel};
    use std::sync::Arc;

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).expect("valid JSON")
    }

    #[test]
    fn canonical_json_sorts_keys_and_drops_whitespace() {
        let value = parse(r#"{"b": 1, "a": {"y": 2, "x": [3, {"d": 4, "c": 5}]}}"#);
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"x":[3,{"c":5,"d":4}],"y":2},"b":1}"#
        );
    }

    #[test]
    fn canonical_json_keeps_array_order_and_escapes_strings() {
        let value = parse(r#"{"text": "line\nbreak", "items": [2, 1]}"#);
        assert_eq!(
            canonical_json(&value),
            r#"{"items":[2,1],"text":"line\nbreak"}"#
        );
    }

    #[test]
    fn canonical_json_stops_at_the_depth_cap_without_panicking() {
        let mut value = Value::String("leaf".to_string());
        for _ in 0..(MAX_METADATA_DEPTH + 8) {
            value = Value::Array(vec![value]);
        }
        let rendered = canonical_json(&value);
        // The over-deep node became `null`, and the document stayed well formed.
        assert!(rendered.contains("null"));
        let reparsed: Value = serde_json::from_str(&rendered).expect("well-formed JSON");
        assert_eq!(canonical_json(&reparsed), rendered);
    }

    #[test]
    fn stable_metadata_sha256_is_key_order_independent() {
        let first = parse(r#"{"a": 1, "b": {"x": 2, "y": 3}}"#);
        let second = parse(r#"{"b": {"y": 3, "x": 2}, "a": 1}"#);
        assert_eq!(
            stable_metadata_sha256(&first),
            stable_metadata_sha256(&second)
        );
        assert_eq!(stable_metadata_sha256(&first).len(), 64);
    }

    #[test]
    fn stable_metadata_sha256_detects_value_changes() {
        let first = parse(r#"{"a": 1}"#);
        let second = parse(r#"{"a": 2}"#);
        assert_ne!(
            stable_metadata_sha256(&first),
            stable_metadata_sha256(&second)
        );
        // Numbers keep their lexical form, so 1 and 1.0 are different digests.
        assert_ne!(
            stable_metadata_sha256(&parse(r#"{"a": 1}"#)),
            stable_metadata_sha256(&parse(r#"{"a": 1.0}"#))
        );
    }

    #[test]
    fn endpoint_sha256_normalizes_scheme_host_port_and_slashes() {
        let canonical = endpoint_sha256("https://api.example.com/v1");
        assert_eq!(
            endpoint_sha256("  HTTPS://API.Example.COM:443/v1///  "),
            canonical
        );
        assert_eq!(endpoint_sha256("https://api.example.com/v1/"), canonical);
        assert_eq!(endpoint_sha256("https://api.example.com:8443/v1"), {
            endpoint_sha256("https://api.example.com:8443/v1/")
        });
        assert_ne!(
            endpoint_sha256("https://api.example.com:8443/v1"),
            canonical
        );
        assert_eq!(endpoint_sha256("http://api.example.com:80"), {
            endpoint_sha256("http://API.example.com/")
        });
    }

    #[test]
    fn endpoint_sha256_excludes_query_and_fragment() {
        let canonical = endpoint_sha256("https://api.example.com/v1");
        assert_eq!(
            endpoint_sha256("https://api.example.com/v1?token=secret&x=1"),
            canonical
        );
        assert_eq!(
            endpoint_sha256("https://api.example.com/v1#section"),
            canonical
        );
    }

    #[test]
    fn endpoint_sha256_drops_userinfo_and_keeps_ipv6_brackets() {
        assert_eq!(
            endpoint_sha256("https://user:pass@api.example.com/v1"),
            endpoint_sha256("https://api.example.com/v1")
        );
        assert_eq!(
            endpoint_sha256("http://[::1]:9000/a/"),
            endpoint_sha256("HTTP://[::1]:9000/a")
        );
    }

    #[test]
    fn endpoint_sha256_falls_back_for_scheme_less_and_unparsable_input() {
        assert_eq!(
            endpoint_sha256(" api.example.com/v1/ "),
            endpoint_sha256("api.example.com/v1")
        );
        // An out-of-range port falls back to the trimmed text instead of panicking.
        assert_eq!(
            endpoint_sha256("https://api.example.com:99999/v1"),
            sha256_hex("https://api.example.com:99999/v1")
        );
        // An unterminated IPv6 bracket is equally tolerant.
        assert_eq!(
            endpoint_sha256("http://[::1/v1"),
            sha256_hex("http://[::1/v1")
        );
    }

    #[test]
    fn sanitize_metadata_drops_credentials_at_any_depth() {
        let value = parse(
            r#"{
                "api_key": "sk-live",
                "client_secret": {"inner": 1},
                "access_token": "abc",
                "signing_key": "x",
                "password": "hunter2",
                "nested": {"password": "y", "refresh_token": "z", "note": "ok"},
                "items": [{"token": "t", "keep": 1}, "plain"],
                "count": 3,
                "flag": true
            }"#,
        );
        let sanitized = sanitize_metadata(&value);
        assert_eq!(
            sanitized,
            parse(
                r#"{"nested": {"note": "ok"}, "items": [{"keep": 1}, "plain"], "count": 3, "flag": true}"#
            )
        );
    }

    #[test]
    fn sanitize_metadata_hashes_endpoints_and_keeps_safe_key_fields() {
        let value = parse(
            r#"{
                "api_base": " HTTPS://API.Example.COM:443/v1/?signature=x ",
                "endpoint_url": null,
                "cache_key": "abc",
                "partition_key": 7,
                "key_type": "primary"
            }"#,
        );
        let sanitized = sanitize_metadata(&value);
        let object = sanitized.as_object().expect("object");
        assert_eq!(
            object.get("api_base_sha256").and_then(Value::as_str),
            Some(endpoint_sha256("https://api.example.com/v1").as_str())
        );
        assert!(!object.contains_key("api_base"));
        assert!(!object.contains_key("endpoint_url"));
        assert_eq!(object.get("cache_key"), Some(&Value::from("abc")));
        assert_eq!(object.get("partition_key"), Some(&Value::from(7)));
        assert_eq!(object.get("key_type"), Some(&Value::from("primary")));
    }

    #[test]
    fn sanitize_metadata_preserves_key_order() {
        let value = parse(r#"{"z": 1, "api_key": "x", "a": 2, "m": 3}"#);
        let sanitized = sanitize_metadata(&value);
        let keys: Vec<&str> = sanitized
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["z", "a", "m"]);
    }

    #[test]
    fn sanitize_metadata_caps_depth_and_accepts_non_objects() {
        let mut value = parse(r#"{"secret": "x"}"#);
        for _ in 0..(MAX_METADATA_DEPTH + 4) {
            value = Value::Object(Map::from_iter([("level".to_string(), value)]));
        }
        let sanitized = sanitize_metadata(&value);
        assert!(sanitized.to_string().contains("null"));
        assert_eq!(sanitize_metadata(&Value::from(5)), Value::from(5));
        assert_eq!(
            sanitize_metadata(&parse(r#"[{"token": "x"}, 1]"#)),
            parse(r"[{}, 1]")
        );
    }

    fn descriptor() -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", "a", "Extract")
            .with_description("extracts text")
            .with_outputs(vec![crate::model::CapabilityIo::new("text", "string")])
    }

    #[test]
    fn graph_identity_is_stable_and_sensitive_to_matching_fields() {
        let base = graph_identity_sha256(&descriptor());
        assert_eq!(base.len(), 64);
        assert_eq!(graph_identity_sha256(&descriptor()), base);
        // Availability and metadata do not participate in matching.
        let widened = descriptor()
            .with_available(false)
            .with_metadata("x", Value::from(1));
        assert_eq!(graph_identity_sha256(&widened), base);
        // A changed description or IO list does.
        let described = descriptor().with_description("extracts text from PDFs");
        assert_ne!(graph_identity_sha256(&described), base);
        let reio = descriptor().with_outputs(vec![crate::model::CapabilityIo::new("body", "text")]);
        assert_ne!(graph_identity_sha256(&reio), base);
    }

    #[test]
    fn model_identity_digests_cover_the_sampling_and_client_fields() {
        let config = ModelConfig::new()
            .with_model(Arc::new(StaticCompletionModel::single("{}")))
            .with_temperature(0.2)
            .with_max_tokens(64);
        let identity = ModelIdentity::of(&config, "static-completion-model");
        assert_eq!(identity.model, "static-completion-model");
        assert_eq!(identity.backend, "static-completion-model");
        assert_eq!(identity.temperature, Some(f64::from(0.2_f32)));
        assert_eq!(identity.max_tokens, Some(64));
        assert_eq!(identity.top_p, None);
        assert_eq!(identity.stop, [] as [String; 0]);
        assert_eq!(identity.request_config_sha256.len(), 64);
        assert_eq!(identity.client_config_sha256.len(), 64);

        // Deterministic: the same configuration derives the same digests.
        assert_eq!(
            ModelIdentity::of(&config, "static-completion-model"),
            identity
        );

        // A sampling change moves the request digest but not the client digest.
        let warmer = ModelConfig::new()
            .with_model(Arc::new(StaticCompletionModel::single("{}")))
            .with_temperature(0.9)
            .with_max_tokens(64);
        let warmer_identity = ModelIdentity::of(&warmer, "static-completion-model");
        assert_ne!(
            warmer_identity.request_config_sha256,
            identity.request_config_sha256
        );
        assert_eq!(
            warmer_identity.client_config_sha256,
            identity.client_config_sha256
        );

        // A backend change moves only the client digest.
        let other_backend = ModelIdentity::of(&config, "foundation-model");
        assert_ne!(
            other_backend.client_config_sha256,
            identity.client_config_sha256
        );
        assert_eq!(
            other_backend.request_config_sha256,
            identity.request_config_sha256
        );
    }

    #[test]
    fn model_identity_handles_absent_models_and_top_p_stop() {
        let bare = ModelIdentity::of(&ModelConfig::new(), "  ");
        assert_eq!(bare.model, "none");
        assert_eq!(bare.backend, "none");
        assert_eq!(bare.temperature, None);
        assert_eq!(bare.max_tokens, None);

        let tuned = ModelIdentity::of(&ModelConfig::new(), "backend")
            .with_top_p(0.9)
            .with_stop(["STOP", "END"]);
        assert_eq!(tuned.top_p, Some(0.9));
        assert_eq!(tuned.stop, vec!["STOP".to_string(), "END".to_string()]);
        assert_ne!(tuned.request_config_sha256, bare.request_config_sha256);
        // A non-finite sampling value is dropped rather than poisoning the digest.
        let infinite = tuned.clone().with_top_p(f64::INFINITY);
        assert_eq!(infinite.top_p, None);
        let without_top_p =
            ModelIdentity::of(&ModelConfig::new(), "backend").with_stop(["STOP", "END"]);
        assert_eq!(
            infinite.request_config_sha256,
            without_top_p.request_config_sha256
        );
    }

    #[test]
    fn model_identity_serializes_with_a_fixed_field_order() {
        let identity = ModelIdentity::of(
            &ModelConfig::new().with_fallback(FallbackMode::Error),
            "backend",
        );
        let value = serde_json::to_value(&identity).expect("serializable");
        let keys: Vec<&str> = value
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "model",
                "backend",
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "request_config_sha256",
                "client_config_sha256",
            ]
        );
        let round_trip: ModelIdentity = serde_json::from_value(value).expect("round-trips");
        assert_eq!(round_trip, identity);
        // Both schemas are derived, so the identity is usable in card tooling.
        let schema =
            serde_json::to_value(schemars::schema_for!(ModelIdentity)).expect("schema serializes");
        assert_eq!(schema["title"], "ModelIdentity");
    }
}
