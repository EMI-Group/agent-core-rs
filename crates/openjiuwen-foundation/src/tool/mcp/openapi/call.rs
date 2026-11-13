//! `OpenAPI` request building and response mapping.
//!
//! Turns one [`OpenApiOperation`] plus the caller's arguments into an HTTP
//! request (URL, query, headers, JSON body) and maps the response onto
//! [`McpToolResult`]. The builders are pure so they can be unit-tested without a
//! server; only [`execute`] performs I/O.

use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest, HttpResponse, Method};
use serde_json::{Map, Value};

use super::super::client::McpServerConfig;
use super::super::result::McpToolResult;
use super::spec::OpenApiOperation;

/// Maximum number of body characters kept in a failed result.
pub(crate) const MAX_ERROR_BODY_CHARS: usize = 512;

/// Calls one `OpenAPI` operation and maps the response onto [`McpToolResult`].
///
/// # Errors
///
/// Returns [`Error::Validation`] when a path parameter is missing or the method
/// cannot be represented, [`Error::Tool`] when a 2xx JSON body cannot be parsed,
/// and the [`HttpClient::execute`] errors ([`Error::Timeout`],
/// [`Error::Cancelled`], [`Error::Transport`]) unchanged.
pub(crate) async fn execute(
    http: &HttpClient,
    operation: &OpenApiOperation,
    base_url: &str,
    args: &Map<String, Value>,
    config: &McpServerConfig,
    timeout: Option<Duration>,
) -> Result<McpToolResult> {
    let url = build_url(base_url, &operation.path, args, operation)?;
    let method = Method::from_bytes(operation.method.as_bytes()).map_err(|error| {
        Error::validation(format!(
            "invalid HTTP method `{}` for openapi tool `{}`: {error}",
            operation.method, operation.name
        ))
    })?;
    let mut request = HttpRequest::new(method.clone(), url.clone())
        .queries(query_pairs(operation, args, config))
        .with_headers(header_pairs(operation, args, config));
    if let Some(body) = json_body(operation, args) {
        request = request.json(&body)?;
    }
    if let Some(limit) = timeout {
        request = request.timeout(limit);
    }

    let response = http.execute(request).await?;
    map_response(&response, operation, &method, &url)
}

/// Renders `base_url` + `path`, substituting every `{name}` path parameter.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the tool and the parameter when a
/// `{name}` placeholder has no argument or the argument is JSON null.
pub(crate) fn build_url(
    base: &str,
    path: &str,
    args: &Map<String, Value>,
    operation: &OpenApiOperation,
) -> Result<String> {
    let mut url = String::with_capacity(base.len() + path.len() + 16);
    url.push_str(base.trim_end_matches('/'));
    if !path.starts_with('/') {
        url.push('/');
    }
    let mut rest = path;
    while let Some(start) = rest.find('{') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            break;
        };
        let name = &after[..end];
        url.push_str(&rest[..start]);
        let value = args
            .get(name)
            .ok_or_else(|| missing_path_parameter(operation, name))?;
        let text = value_text(value).ok_or_else(|| {
            Error::validation(format!(
                "openapi tool `{}` requires a non-null value for path parameter `{name}`",
                operation.name
            ))
        })?;
        url.push_str(&percent_encode(&text));
        rest = &after[end + 1..];
    }
    url.push_str(rest);
    Ok(url)
}

/// Builds the missing-path-parameter validation error.
fn missing_path_parameter(operation: &OpenApiOperation, name: &str) -> Error {
    Error::validation(format!(
        "openapi tool `{}` requires path parameter `{name}`",
        operation.name
    ))
}

/// Collects declared query parameters plus the configured auth query parameters.
pub(crate) fn query_pairs(
    operation: &OpenApiOperation,
    args: &Map<String, Value>,
    config: &McpServerConfig,
) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for parameter in &operation.parameters {
        if parameter.location != "query" {
            continue;
        }
        if let Some(text) = args.get(&parameter.name).and_then(value_text) {
            pairs.push((parameter.name.clone(), text));
        }
    }
    pairs.extend(
        config
            .auth_query_params
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    pairs
}

/// Collects declared header parameters plus the configured auth headers.
///
/// When the call carries a JSON body, `Content-Type` and `Accept` are set to
/// `application/json` unless a parameter or auth header already names them.
pub(crate) fn header_pairs(
    operation: &OpenApiOperation,
    args: &Map<String, Value>,
    config: &McpServerConfig,
) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for parameter in &operation.parameters {
        if parameter.location != "header" {
            continue;
        }
        if let Some(text) = args.get(&parameter.name).and_then(value_text) {
            pairs.push((parameter.name.clone(), text));
        }
    }
    pairs.extend(
        config
            .auth_headers
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    if json_body(operation, args).is_some() {
        for (name, value) in [
            ("Content-Type", "application/json"),
            ("Accept", "application/json"),
        ] {
            let declared = pairs
                .iter()
                .any(|(existing, _)| existing.eq_ignore_ascii_case(name));
            if !declared {
                pairs.push((name.to_owned(), value.to_owned()));
            }
        }
    }
    pairs
}

/// Builds the JSON request body of `operation` from `args`.
///
/// Returns `None` when the operation declares no JSON body, `Some({})` when it
/// declares one but no argument matches a body property, the declared body
/// properties otherwise, and every argument that is not a path/query/header
/// parameter when the body schema has no `properties`.
pub(crate) fn json_body(operation: &OpenApiOperation, args: &Map<String, Value>) -> Option<Value> {
    let body_schema = operation.body_schema.as_ref()?;
    let mut body = Map::new();
    match body_schema.get("properties").and_then(Value::as_object) {
        Some(properties) => {
            for name in properties.keys() {
                if let Some(value) = args.get(name) {
                    body.insert(name.clone(), value.clone());
                }
            }
        }
        None => {
            for (name, value) in args {
                let consumed = operation
                    .parameters
                    .iter()
                    .any(|parameter| parameter.name == *name);
                if !consumed {
                    body.insert(name.clone(), value.clone());
                }
            }
        }
    }
    Some(Value::Object(body))
}

/// Renders one argument value as text (query/header/path form).
///
/// Strings stay verbatim, numbers and booleans use their JSON spelling, arrays
/// are comma-joined, objects become compact JSON, and null has no rendering.
fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Bool(_) | Value::Number(_) => Some(value.to_string()),
        Value::Array(items) => Some(
            items
                .iter()
                .filter_map(value_text)
                .collect::<Vec<_>>()
                .join(","),
        ),
        Value::Object(_) => serde_json::to_string(value).ok(),
    }
}

/// Percent-encodes everything outside the RFC 3986 unreserved set.
fn percent_encode(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(hex_digit(byte >> 4));
                encoded.push(hex_digit(byte & 0x0F));
            }
        }
    }
    encoded
}

/// Renders one upper-case hexadecimal digit.
fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        _ => char::from(b'A' + (value - 10)),
    }
}

/// Maps an HTTP response onto a tool result.
fn map_response(
    response: &HttpResponse,
    operation: &OpenApiOperation,
    method: &Method,
    url: &str,
) -> Result<McpToolResult> {
    if !response.is_success() {
        let body = truncate(&response.text());
        let mut data = Map::new();
        data.insert("status".to_owned(), Value::from(response.status));
        data.insert("body".to_owned(), Value::String(body.clone()));
        return Ok(McpToolResult {
            success: false,
            data,
            error: Some(format!(
                "HTTP {} from {method} {url}: {body}",
                response.status
            )),
        });
    }

    let content_type = response
        .headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if is_json_content_type(content_type) && !response.body.is_empty() {
        return match serde_json::from_slice::<Value>(&response.body) {
            Ok(Value::Object(map)) => Ok(McpToolResult::success(map)),
            Ok(other) => {
                let mut data = Map::new();
                data.insert("result".to_owned(), other);
                Ok(McpToolResult::success(data))
            }
            Err(error) => Err(Error::tool(format!(
                "openapi tool `{}` returned invalid JSON from {method} {url}: {error}",
                operation.name
            ))),
        };
    }

    let mut data = Map::new();
    data.insert("content".to_owned(), Value::String(response.text()));
    Ok(McpToolResult::success(data))
}

/// Reports whether `content_type` denotes a JSON payload.
fn is_json_content_type(content_type: &str) -> bool {
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    matches!(
        essence.as_str(),
        "application/json" | "text/json" | "application/x-json"
    ) || essence.ends_with("+json")
}

/// Truncates `text` to [`MAX_ERROR_BODY_CHARS`], marking the cut with `…`.
fn truncate(text: &str) -> String {
    let mut excerpt: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
    if text.chars().count() > MAX_ERROR_BODY_CHARS {
        excerpt.push('…');
    }
    excerpt
}
