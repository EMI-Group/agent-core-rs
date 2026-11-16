//! Request, configuration and cursor validation plus the structural index fingerprint.
//!
//! The model-facing entry point accepts a loose [`serde_json::Value`]; this module turns
//! it into a [`RetrievalRequest`] with strict key and type checks, then turns that request
//! into a [`ValidatedRequest`] with the cid resolved, the query trimmed and the page and
//! search bounds applied. It also owns the page cursor: the continuation state, its
//! deterministic token, the LRU store, and the resume checks.
//!
//! Every rejection is [`Error::Validation`](crate::error::Error::Validation) naming the
//! offending field, except an unknown *scope* or *item* cid, which is
//! [`Error::NotFound`](crate::error::Error::NotFound) and is raised by the operation
//! layer.
use std::str::FromStr;

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{
    RetrievalOperation, RetrievalRequest, RetrievalToolkitConfig, CURSOR_MAX_CHARS,
    CURSOR_STATE_VERSION, CURSOR_TOKEN_BYTES, DEFAULT_TOP_K, MAX_CURSORS, MAX_OUTPUT_CHARS,
    MAX_PAGE_SIZE, MAX_TOP_K, MIN_OUTPUT_CHARS,
};
use crate::error::{validation, Result};
use crate::retrieval::types::{LoadedIndex, ROOT_NODE_ID};

/// The argument keys the model-facing tool accepts.
const ACCEPTED_KEYS: [&str; 6] = ["operation", "cid", "query", "cursor", "page_size", "top_k"];

/// A request after validation: resolved cid, trimmed query, effective bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedRequest {
    /// The operation to run.
    pub operation: RetrievalOperation,
    /// The resolved scope cid (`ROOT` when none was given).
    pub cid: String,
    /// Whether the caller named a cid explicitly.
    pub cid_given: bool,
    /// The trimmed query; empty for the non-search operations.
    pub query: String,
    /// Whether the caller named a query explicitly.
    pub query_given: bool,
    /// The trimmed cursor token, when the request resumes a page.
    pub cursor: Option<String>,
    /// The effective page size of a fresh request.
    pub page_size: usize,
    /// The effective page size when the caller named one (explicitly or via `0`).
    pub page_size_given: Option<usize>,
    /// The effective `top_k` of a fresh search.
    pub top_k: usize,
    /// The effective `top_k` when the caller named one (explicitly or via `0`).
    pub top_k_given: Option<usize>,
}

/// One page cursor's continuation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CursorState {
    /// The operation the page belongs to.
    pub operation: RetrievalOperation,
    /// The resolved scope cid the page was minted for.
    pub cid: String,
    /// The search query; empty for the other operations.
    pub query: String,
    /// The offset the next page starts at.
    pub offset: usize,
    /// The page size the original request used.
    pub page_size: usize,
    /// The `top_k` the original search used.
    pub top_k: usize,
    /// The index fingerprint the page was minted against.
    pub fingerprint: String,
}

impl CursorState {
    /// The deterministic token for this state.
    ///
    /// The URL-safe unpadded base64 of the first [`CURSOR_TOKEN_BYTES`] of SHA-256 over
    /// the canonical JSON of `{v, operation, cid, query, offset, page_size, top_k,
    /// fingerprint}`. The field order is fixed and the document holds no maps, so the
    /// token is identical for identical states.
    pub(crate) fn token(&self) -> String {
        let document = json!({
            "v": CURSOR_STATE_VERSION,
            "operation": self.operation.as_str(),
            "cid": self.cid,
            "query": self.query,
            "offset": self.offset,
            "page_size": self.page_size,
            "top_k": self.top_k,
            "fingerprint": self.fingerprint,
        });
        let digest = Sha256::digest(document.to_string().as_bytes());
        let head = digest
            .get(..CURSOR_TOKEN_BYTES)
            .unwrap_or(digest.as_slice());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(head)
    }
}

/// The page cursors, ordered least recently used first.
#[derive(Debug, Default, Clone)]
pub(crate) struct CursorStore {
    /// `(token, state)` pairs, the most recently used last.
    entries: Vec<(String, CursorState)>,
}

impl CursorStore {
    /// The state behind `token`, when it is still held.
    pub(crate) fn get(&self, token: &str) -> Option<&CursorState> {
        self.entries
            .iter()
            .find(|(held, _)| held == token)
            .map(|(_, state)| state)
    }

    /// Stores `state` under `token`, evicting the least recently used past
    /// [`MAX_CURSORS`].
    pub(crate) fn push(&mut self, token: String, state: CursorState) {
        self.entries.retain(|(held, _)| held != &token);
        self.entries.push((token, state));
        while self.entries.len() > MAX_CURSORS {
            self.entries.remove(0);
        }
    }

    /// Drops `token` from the store.
    pub(crate) fn remove(&mut self, token: &str) {
        self.entries.retain(|(held, _)| held != token);
    }

    /// How many cursors the store holds.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drops every cursor.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Validates one toolkit configuration.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error::Validation) naming `session_scope`,
/// `max_output_chars`, `max_page_size`, `max_top_k` or `default_page_size`.
pub(crate) fn validate_config(config: &RetrievalToolkitConfig) -> Result<()> {
    if config.session_scope.trim().is_empty() {
        return Err(validation("session_scope", "must be non-empty"));
    }
    if !(MIN_OUTPUT_CHARS..=MAX_OUTPUT_CHARS).contains(&config.max_output_chars) {
        return Err(validation(
            "max_output_chars",
            format!("must be between {MIN_OUTPUT_CHARS} and {MAX_OUTPUT_CHARS}"),
        ));
    }
    if !(1..=MAX_PAGE_SIZE).contains(&config.max_page_size) {
        return Err(validation(
            "max_page_size",
            format!("must be between 1 and {MAX_PAGE_SIZE}"),
        ));
    }
    if !(1..=MAX_TOP_K).contains(&config.max_top_k) {
        return Err(validation(
            "max_top_k",
            format!("must be between 1 and {MAX_TOP_K}"),
        ));
    }
    if config.default_page_size == 0 || config.default_page_size > config.max_page_size {
        return Err(validation(
            "default_page_size",
            format!(
                "must be between 1 and max_page_size ({})",
                config.max_page_size
            ),
        ));
    }
    Ok(())
}

/// Parses the model-facing argument object into a typed request.
///
/// Exactly `operation`, `cid`, `query`, `cursor`, `page_size` and `top_k` are accepted;
/// `null` means "absent", and a blank `cid`/`query`/`cursor` string is kept so the
/// semantic validation can name the field.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error::Validation) naming `arguments`,
/// `operation`, `cid`, `query`, `cursor`, `page_size` or `top_k`.
pub(crate) fn parse_invoke_arguments(arguments: &Value) -> Result<RetrievalRequest> {
    let Value::Object(map) = arguments else {
        return Err(validation(
            "arguments",
            "must be a JSON object with operation, cid, query, cursor, page_size or top_k",
        ));
    };
    let mut unexpected: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|key| !ACCEPTED_KEYS.contains(key))
        .collect();
    if !unexpected.is_empty() {
        unexpected.sort_unstable();
        return Err(validation(
            "arguments",
            format!(
                "unknown retrieval toolkit argument(s): {}. Only operation, cid, query, cursor, \
page_size and top_k are accepted.",
                unexpected.join(", ")
            ),
        ));
    }
    let operation = match map.get("operation") {
        None | Some(Value::Null) => {
            return Err(validation(
                "operation",
                "is required; use one of: list, browse, search, read",
            ))
        }
        Some(Value::String(value)) => RetrievalOperation::from_str(value)?,
        Some(_) => return Err(validation("operation", "must be a string")),
    };
    let cid = string_field(map.get("cid"), "cid")?;
    let query = string_field(map.get("query"), "query")?;
    let cursor = string_field(map.get("cursor"), "cursor")?;
    let page_size = usize_field(map.get("page_size"), "page_size")?;
    let top_k = usize_field(map.get("top_k"), "top_k")?;
    Ok(RetrievalRequest {
        operation,
        cid,
        query,
        cursor,
        page_size,
        top_k,
    })
}

/// Reads one optional string argument.
fn string_field(value: Option<&Value>, field: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(validation(field, "must be a string")),
    }
}

/// Reads one optional non-negative integer argument.
fn usize_field(value: Option<&Value>, field: &str) -> Result<Option<usize>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| validation(field, "must be a non-negative integer")),
        Some(_) => Err(validation(field, "must be a non-negative integer")),
    }
}

/// Semantically validates one typed request.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error::Validation) naming `cid`, `query`,
/// `cursor`, `page_size` or `top_k`.
pub(crate) fn validate_request(
    request: &RetrievalRequest,
    config: &RetrievalToolkitConfig,
) -> Result<ValidatedRequest> {
    let operation = request.operation;
    if request.cursor.is_some() && operation == RetrievalOperation::Read {
        return Err(validation("cursor", "is not valid for operation=read"));
    }
    let (cid, cid_given) = if let Some(value) = request.cid.as_deref() {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(validation("cid", "must be non-empty when provided"));
        }
        if operation == RetrievalOperation::Search {
            return Err(validation("cid", "is valid only for list, browse and read"));
        }
        (trimmed.to_string(), true)
    } else {
        if operation == RetrievalOperation::Read {
            return Err(validation("cid", "is required for operation=read"));
        }
        (ROOT_NODE_ID.to_string(), false)
    };
    let (query, query_given) = match request.query.as_deref() {
        Some(value) => {
            if operation != RetrievalOperation::Search {
                return Err(validation("query", "is valid only for operation=search"));
            }
            (value.trim().to_string(), true)
        }
        None => (String::new(), false),
    };
    if request.page_size.is_some() && operation == RetrievalOperation::Read {
        return Err(validation(
            "page_size",
            "is valid only for list, browse and search",
        ));
    }
    if request.top_k.is_some() && operation != RetrievalOperation::Search {
        return Err(validation("top_k", "is valid only for operation=search"));
    }
    let page_size = resolve_page_size(request.page_size, config);
    let top_k = resolve_top_k(request.top_k, config);
    let cursor = match request.cursor.as_deref() {
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Err(validation("cursor", "must be a non-empty string"));
            }
            if trimmed.chars().count() > CURSOR_MAX_CHARS {
                return Err(validation(
                    "cursor",
                    format!("must not exceed {CURSOR_MAX_CHARS} characters"),
                ));
            }
            Some(trimmed.to_string())
        }
        None => None,
    };
    Ok(ValidatedRequest {
        operation,
        cid,
        cid_given,
        query,
        query_given,
        cursor,
        page_size: page_size.unwrap_or(config.default_page_size),
        page_size_given: page_size,
        top_k: top_k.unwrap_or_else(|| default_top_k(config)),
        top_k_given: top_k,
    })
}

/// Resolves a requested page size: absent stays absent, `0` is the default, and a value
/// above the configured maximum is clamped.
fn resolve_page_size(requested: Option<usize>, config: &RetrievalToolkitConfig) -> Option<usize> {
    match requested {
        None => None,
        Some(0) => Some(config.default_page_size),
        Some(value) => Some(value.min(config.max_page_size).max(1)),
    }
}

/// Resolves a requested `top_k`: absent stays absent, `0` is the default, and a value
/// above the configured maximum is clamped.
fn resolve_top_k(requested: Option<usize>, config: &RetrievalToolkitConfig) -> Option<usize> {
    match requested {
        None => None,
        Some(0) => Some(default_top_k(config)),
        Some(value) => Some(value.min(config.max_top_k).max(1)),
    }
}

/// The default `top_k` of one search: [`DEFAULT_TOP_K`] capped by `max_top_k`.
fn default_top_k(config: &RetrievalToolkitConfig) -> usize {
    DEFAULT_TOP_K.min(config.max_top_k).max(1)
}

/// Checks a cursor state against the request that resumes it.
///
/// The operation must match; an explicitly repeated cid, query, page size or `top_k` must
/// equal the original; and the index fingerprint must still be current.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error::Validation) naming `cursor`.
pub(crate) fn validate_resume(
    state: &CursorState,
    request: &ValidatedRequest,
    fingerprint: &str,
) -> Result<()> {
    if state.operation != request.operation {
        return Err(validation(
            "cursor",
            format!("belongs to operation={}", state.operation.as_str()),
        ));
    }
    if request.cid_given && state.cid != request.cid {
        return Err(validation(
            "cursor",
            "cid differs from the original request; omit cursor to change the scope",
        ));
    }
    if request.query_given && state.query != request.query {
        return Err(validation(
            "cursor",
            "query differs from the original request; omit cursor to start a new search",
        ));
    }
    if let Some(page_size) = request.page_size_given {
        if state.page_size != page_size {
            return Err(validation(
                "cursor",
                "page_size differs from the original request; omit cursor to change the page size",
            ));
        }
    }
    if let Some(top_k) = request.top_k_given {
        if state.top_k != top_k {
            return Err(validation(
                "cursor",
                "top_k differs from the original request; omit cursor to change the search bound",
            ));
        }
    }
    if state.fingerprint != fingerprint {
        return Err(validation(
            "cursor",
            "the index changed since the page was minted; start a new request",
        ));
    }
    Ok(())
}

/// The structural fingerprint of one loaded index.
///
/// SHA-256 over the canonical JSON of the tree (pre-order, children and items sorted) and
/// the catalog records (sorted by `cid`, then `worker_id`). Labels, descriptions, item
/// payloads and record text participate, so any content change invalidates a page cursor;
/// manifest bookkeeping that a search does not read (timestamps, paths) does not.
pub(crate) fn index_fingerprint(index: &LoadedIndex) -> String {
    let tree: Vec<Value> = index
        .root
        .walk()
        .into_iter()
        .map(|node| {
            let items: Vec<Value> = node
                .items()
                .into_iter()
                .map(|item| json!([item.item_id, item.payload, item.label, item.description]))
                .collect();
            json!({
                "node_id": node.node_id,
                "label": node.label,
                "description": node.description,
                "items": items,
            })
        })
        .collect();
    let mut records = index.records.clone();
    records.sort_by(|left, right| {
        left.cid
            .cmp(&right.cid)
            .then_with(|| left.worker_id.cmp(&right.worker_id))
    });
    let records: Vec<Value> = records
        .iter()
        .map(|record| {
            json!({
                "worker_id": record.worker_id,
                "cid": record.cid,
                "name": record.name,
                "description": record.description,
                "skill_path": record.skill_path,
                "retrieval_text": record.retrieval_text,
                "metadata": record.metadata,
            })
        })
        .collect();
    let document = json!({
        "v": CURSOR_STATE_VERSION,
        "tree": tree,
        "records": records,
    });
    format!("{:x}", Sha256::digest(document.to_string().as_bytes()))
}

/// Rejects a cursor token that cannot be resumed before the store is consulted; kept
/// here so both entry points share one bound.
pub(crate) fn validate_cursor_token(token: &str) -> Result<()> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Err(validation("cursor", "must be a non-empty string"));
    }
    if trimmed.chars().count() > CURSOR_MAX_CHARS {
        return Err(validation(
            "cursor",
            format!("must not exceed {CURSOR_MAX_CHARS} characters"),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::types::{
        CatalogRecord, IndexManifest, LoadedIndex, RetrieverItem, RetrieverNode,
    };
    use std::collections::BTreeMap;

    fn config() -> RetrievalToolkitConfig {
        RetrievalToolkitConfig::new("session-a")
    }

    #[test]
    fn config_rejects_every_bounded_field() {
        let mut value = config();
        value.session_scope = "  ".to_string();
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("session_scope"));

        let mut value = config();
        value.max_output_chars = MIN_OUTPUT_CHARS - 1;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("max_output_chars"));
        value.max_output_chars = MAX_OUTPUT_CHARS + 1;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("max_output_chars"));

        let mut value = config();
        value.max_page_size = 0;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("max_page_size"));
        value.max_page_size = MAX_PAGE_SIZE + 1;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("max_page_size"));

        let mut value = config();
        value.max_top_k = 0;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("max_top_k"));
        value.max_top_k = MAX_TOP_K + 1;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("max_top_k"));

        let mut value = config();
        value.default_page_size = 0;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("default_page_size"));
        value.default_page_size = value.max_page_size + 1;
        assert!(validate_config(&value)
            .unwrap_err()
            .message()
            .contains("default_page_size"));

        assert!(validate_config(&config()).is_ok());
    }

    #[test]
    fn invoke_arguments_accept_the_six_keys_and_reject_anything_else() {
        let request = parse_invoke_arguments(&json!({
            "operation": "search",
            "cid": "Skills.Data",
            "query": " csv ",
            "cursor": null,
            "page_size": 0,
            "top_k": 5,
        }))
        .unwrap();
        assert_eq!(request.operation, RetrievalOperation::Search);
        assert_eq!(request.cid.as_deref(), Some("Skills.Data"));
        assert_eq!(request.query.as_deref(), Some(" csv "));
        assert_eq!(request.page_size, Some(0));
        assert_eq!(request.top_k, Some(5));

        let error = parse_invoke_arguments(&json!({"operation": "list", "limit": 3})).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("limit"));
        assert!(error.message().contains("arguments"));

        let error = parse_invoke_arguments(&json!({"cid": "Skills"})).unwrap_err();
        assert!(error.message().contains("operation"));

        let error = parse_invoke_arguments(&json!([])).unwrap_err();
        assert!(error.message().contains("arguments"));
    }

    #[test]
    fn invoke_argument_types_are_checked_per_field() {
        for arguments in [
            json!({"operation": 3}),
            json!({"operation": "list", "cid": 3}),
            json!({"operation": "search", "query": ["a"]}),
            json!({"operation": "list", "cursor": 1}),
            json!({"operation": "list", "page_size": "2"}),
            json!({"operation": "list", "page_size": -1}),
            json!({"operation": "list", "page_size": 1.5}),
            json!({"operation": "search", "query": "q", "top_k": true}),
        ] {
            let error = parse_invoke_arguments(&arguments).unwrap_err();
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        }
        let error =
            parse_invoke_arguments(&json!({"operation": "list", "page_size": -1})).unwrap_err();
        assert!(error.message().contains("page_size"));
    }

    #[test]
    fn request_validation_names_every_misused_field() {
        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Read,
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("cid"));
        assert!(error.message().contains("required"));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::List,
                query: Some("q".to_string()),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("query"));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Browse,
                cid: Some("  ".to_string()),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("cid"));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Browse,
                top_k: Some(3),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("top_k"));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Read,
                cid: Some("Skills.Data.alpha".to_string()),
                page_size: Some(3),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("page_size"));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Read,
                cid: Some("Skills.Data.alpha".to_string()),
                cursor: Some("abc".to_string()),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("cursor"));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Search,
                query: Some("q".to_string()),
                cid: Some("Skills.Data".to_string()),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("cid"));
    }

    #[test]
    fn request_validation_resolves_bounds_and_sentinels() {
        let validated = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::Search,
                query: Some("  csv  ".to_string()),
                page_size: Some(0),
                top_k: Some(0),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap();
        assert_eq!(validated.cid, ROOT_NODE_ID);
        assert!(!validated.cid_given);
        assert_eq!(validated.query, "csv");
        assert!(validated.query_given);
        assert_eq!(validated.page_size, config().default_page_size);
        assert_eq!(validated.page_size_given, Some(config().default_page_size));
        assert_eq!(validated.top_k, DEFAULT_TOP_K.min(config().max_top_k));
        assert_eq!(
            validated.top_k_given,
            Some(DEFAULT_TOP_K.min(config().max_top_k))
        );

        let validated = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::List,
                page_size: Some(10_000),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap();
        assert_eq!(validated.page_size, config().max_page_size);
        assert_eq!(validated.page_size_given, Some(config().max_page_size));

        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::List,
                cursor: Some(" ".to_string()),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("cursor"));

        let too_long = "x".repeat(CURSOR_MAX_CHARS + 1);
        let error = validate_request(
            &RetrievalRequest {
                operation: RetrievalOperation::List,
                cursor: Some(too_long),
                ..RetrievalRequest::default()
            },
            &config(),
        )
        .unwrap_err();
        assert!(error.message().contains("cursor"));
    }

    #[test]
    fn resume_validation_covers_every_mismatch() {
        let state = CursorState {
            operation: RetrievalOperation::List,
            cid: "Skills.Data".to_string(),
            query: String::new(),
            offset: 3,
            page_size: 3,
            top_k: 10,
            fingerprint: "fp".to_string(),
        };
        let request = ValidatedRequest {
            operation: RetrievalOperation::List,
            cid: "Skills.Data".to_string(),
            cid_given: true,
            query: String::new(),
            query_given: false,
            cursor: Some("token".to_string()),
            page_size: 3,
            page_size_given: Some(3),
            top_k: 10,
            top_k_given: None,
        };
        assert!(validate_resume(&state, &request, "fp").is_ok());

        let mut other = request.clone();
        other.operation = RetrievalOperation::Browse;
        assert!(validate_resume(&state, &other, "fp")
            .unwrap_err()
            .message()
            .contains("operation=list"));

        let mut other = request.clone();
        other.cid = "Skills.Writing".to_string();
        assert!(validate_resume(&state, &other, "fp")
            .unwrap_err()
            .message()
            .contains("cid differs"));

        let mut other = request.clone();
        other.query_given = true;
        other.query = "csv".to_string();
        assert!(validate_resume(&state, &other, "fp")
            .unwrap_err()
            .message()
            .contains("query differs"));

        let mut other = request.clone();
        other.page_size_given = Some(5);
        other.page_size = 5;
        assert!(validate_resume(&state, &other, "fp")
            .unwrap_err()
            .message()
            .contains("page_size differs"));

        let mut other = request.clone();
        other.top_k_given = Some(20);
        other.top_k = 20;
        other.query_given = false;
        assert!(validate_resume(&state, &other, "fp")
            .unwrap_err()
            .message()
            .contains("top_k differs"));

        assert!(validate_resume(&state, &request, "changed")
            .unwrap_err()
            .message()
            .contains("index changed"));

        let mut other = request.clone();
        other.cid_given = false;
        other.page_size_given = None;
        assert!(validate_resume(&state, &other, "fp").is_ok());
    }

    #[test]
    fn cursor_tokens_are_deterministic_and_state_specific() {
        let state = CursorState {
            operation: RetrievalOperation::Search,
            cid: ROOT_NODE_ID.to_string(),
            query: "csv".to_string(),
            offset: 5,
            page_size: 5,
            top_k: 10,
            fingerprint: "fp".to_string(),
        };
        let token = state.token();
        assert_eq!(token, state.token());
        assert_eq!(token.len(), 22);
        assert!(!token.contains('='));

        let mut other = state.clone();
        other.offset += 1;
        assert_ne!(token, other.token());
        let mut other = state.clone();
        other.query = "other".to_string();
        assert_ne!(token, other.token());
        let mut other = state.clone();
        other.fingerprint = "fp2".to_string();
        assert_ne!(token, other.token());
    }

    #[test]
    fn cursor_store_is_lru_capped() {
        let mut store = CursorStore::default();
        let state = CursorState {
            operation: RetrievalOperation::List,
            cid: ROOT_NODE_ID.to_string(),
            query: String::new(),
            offset: 0,
            page_size: 1,
            top_k: 1,
            fingerprint: "fp".to_string(),
        };
        for offset in 0..(MAX_CURSORS + 2) {
            let mut current = state.clone();
            current.offset = offset;
            store.push(current.token(), current);
        }
        assert_eq!(store.len(), MAX_CURSORS);
        let first = {
            let mut current = state.clone();
            current.offset = 0;
            current.token()
        };
        assert!(store.get(&first).is_none(), "the oldest cursor was evicted");
        let latest = {
            let mut current = state.clone();
            current.offset = MAX_CURSORS + 1;
            current.token()
        };
        assert!(store.get(&latest).is_some());
        store.remove(&latest);
        assert!(store.get(&latest).is_none());
        store.clear();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn cursor_token_bounds_are_enforced() {
        assert!(validate_cursor_token("abc").is_ok());
        assert!(validate_cursor_token("  ")
            .unwrap_err()
            .message()
            .contains("cursor"));
        assert!(validate_cursor_token(&"x".repeat(CURSOR_MAX_CHARS + 1))
            .unwrap_err()
            .message()
            .contains("cursor"));
    }

    #[test]
    fn the_index_fingerprint_changes_with_tree_and_catalog_content() {
        let index = LoadedIndex {
            root: RetrieverNode::new(ROOT_NODE_ID, "ROOT").with_children(vec![RetrieverNode::new(
                "Skills.Data",
                "Data",
            )
            .with_items(vec![
                RetrieverItem::new("alpha", "Skills.Data.alpha").with_label("Alpha")
            ])]),
            records: vec![CatalogRecord::new("alpha", "Skills.Data.alpha").with_name("Alpha")],
            manifest: IndexManifest::default(),
            choices: Vec::new(),
            item_paths: BTreeMap::new(),
        };
        let baseline = index_fingerprint(&index);
        assert_eq!(baseline, index_fingerprint(&index.clone()));
        assert_eq!(baseline.len(), 64);
        assert!(baseline.chars().all(|value| value.is_ascii_hexdigit()));

        let mut renamed = index.clone();
        renamed.records[0].name = "Renamed".to_string();
        assert_ne!(baseline, index_fingerprint(&renamed));

        let mut relabelled = index.clone();
        relabelled.root.children[0].items[0].label = "Renamed".to_string();
        assert_ne!(baseline, index_fingerprint(&relabelled));

        let mut reordered = index.clone();
        reordered
            .records
            .push(CatalogRecord::new("beta", "Skills.Data.beta").with_name("Beta"));
        reordered.root.children[0]
            .items
            .push(RetrieverItem::new("beta", "Skills.Data.beta"));
        let first = index_fingerprint(&reordered);
        reordered.records.reverse();
        reordered.root.children[0].items.reverse();
        assert_eq!(first, index_fingerprint(&reordered), "order-independent");
    }
}
