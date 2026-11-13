//! `OpenAPI` document loading, `$ref` resolution, and operation collection.
//!
//! Ports the spec-facing half of
//! `openjiuwen/core/foundation/tool/mcp/client/openapi_client.py` (`load_conf`,
//! the `servers[0].url` handling, and the `parse_openapi_to_http_routes`
//! operation walk) without the `fastmcp`/`jsonschema-path` dependencies:
//! operations are collected directly from the `paths` object and their tool
//! schemas are flattened here.
//!
//! `$ref` resolution is deliberately **shallow**: a top-level `$ref` on a
//! schema (or a parameter / request body object) is resolved, while nested
//! references inside `properties` are left untouched so the emitted JSON Schema
//! keeps its `components` pointers.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

/// Maximum number of hops followed through a chain of local `$ref`s.
const MAX_REF_DEPTH: usize = 32;

/// Maximum length of a generated tool name, in characters.
const MAX_TOOL_NAME_CHARS: usize = 64;

/// The operation keys of a path item, in the order they are documented.
const OPERATION_METHODS: [&str; 7] = ["get", "put", "post", "delete", "patch", "head", "options"];

/// Deadline applied when the spec is fetched from a URL.
const SPEC_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// One path/query/header parameter of an `OpenAPI` operation.
///
/// Cookie parameters are dropped while collecting operations: they cannot be
/// expressed through the HTTP request seam this crate uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OpenApiParameter {
    /// Parameter name, exactly as declared in the document.
    pub name: String,
    /// Where the value travels: `path`, `query`, or `header` (lower-case).
    pub location: String,
    /// Whether the caller must supply it.
    pub required: bool,
    /// Resolved JSON Schema of the value (`{}` when the document omits it).
    pub schema: Value,
    /// Optional documentation, also merged into [`OpenApiParameter::schema`].
    pub description: Option<String>,
}

impl Default for OpenApiParameter {
    fn default() -> Self {
        Self {
            name: String::new(),
            location: String::new(),
            required: false,
            schema: Value::Object(Map::new()),
            description: None,
        }
    }
}

/// One HTTP operation of an `OpenAPI` document, projected as an MCP tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OpenApiOperation {
    /// Tool name (de-duplicated inside the spec, at most 64 characters).
    pub name: String,
    /// Tool description, including one line per documented parameter and body.
    pub description: String,
    /// Upper-case HTTP method (`GET`, `POST`, …).
    pub method: String,
    /// Path template, e.g. `/pets/{petId}`.
    pub path: String,
    /// Merged path-level and operation-level parameters (operation entries win).
    pub parameters: Vec<OpenApiParameter>,
    /// Resolved `requestBody.content["application/json"].schema`, when declared.
    pub body_schema: Option<Value>,
    /// Whether the document marks the request body as required.
    pub body_required: bool,
    /// Flattened tool input schema: one property per parameter plus the JSON body
    /// properties (`{"type":"object","properties":{…},"required":[…]}`).
    pub input_schema: Value,
}

impl Default for OpenApiOperation {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            method: String::new(),
            path: String::new(),
            parameters: Vec::new(),
            body_schema: None,
            body_required: false,
            input_schema: Value::Object(Map::new()),
        }
    }
}

/// A parsed `OpenAPI` document together with its operations and base URL.
///
/// Loading is side-effect free apart from one `GET` when the source is an
/// `http://`/`https://` URL; a filesystem source is read once and never written.
#[derive(Debug, Clone)]
pub struct OpenApiSpec {
    document: Value,
    base_url: String,
    operations: Vec<OpenApiOperation>,
    source_url: Option<String>,
}

/// The parser to use for a document, chosen from its extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatHint {
    /// Strict `serde_json` parsing (`.json`).
    Json,
    /// Strict `serde_yaml_ng` parsing (`.yaml`/`.yml`).
    Yaml,
    /// JSON first, YAML fallback (every other extension).
    Auto,
}

impl OpenApiSpec {
    /// Loads a spec from a URL or a filesystem path with the default HTTP client.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OpenApiSpec::load_with`], plus
    /// [`Error::Config`] when the default client cannot be built.
    pub async fn load(source: &str) -> Result<Self> {
        let http = HttpClient::with_defaults()?;
        Self::load_with(&http, source).await
    }

    /// Loads a spec using `http` for URL sources.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty source, a missing/non-file/
    /// symbolic-link path, an unparsable document, a document that is not an
    /// object, or a spec without a usable server URL; [`Error::Transport`] when a
    /// URL source answers with a non-2xx status, plus the transport errors of
    /// [`HttpClient::execute`].
    pub async fn load_with(http: &HttpClient, source: &str) -> Result<Self> {
        Self::load_with_base(http, source, None).await
    }

    /// Loads a spec, letting `base_override` (`params.base_url`) win over
    /// `servers[0].url`.
    ///
    /// # Errors
    ///
    /// Same as [`OpenApiSpec::load_with`].
    pub(crate) async fn load_with_base(
        http: &HttpClient,
        source: &str,
        base_override: Option<&str>,
    ) -> Result<Self> {
        let source = source.trim();
        if source.is_empty() {
            return Err(Error::validation("openapi spec source is empty"));
        }
        let hint = format_hint(source);
        if source.starts_with("http://") || source.starts_with("https://") {
            let request = HttpRequest::get(source).timeout(SPEC_FETCH_TIMEOUT);
            let response = http.execute(request).await?;
            if !response.is_success() {
                return Err(Error::transport(format!(
                    "failed to fetch OpenAPI spec from {source}: HTTP {}",
                    response.status
                )));
            }
            return Self::parse_text(&response.text(), hint, Some(source), base_override);
        }

        let metadata = std::fs::symlink_metadata(source).map_err(|error| {
            Error::validation(format!("openapi spec `{source}` is not readable: {error}"))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(Error::validation(format!(
                "openapi spec `{source}` must not be a symbolic link"
            )));
        }
        if !metadata.is_file() {
            return Err(Error::validation(format!(
                "openapi spec `{source}` is not a file"
            )));
        }
        let text = tokio::fs::read_to_string(source).await.map_err(|error| {
            Error::validation(format!("reading openapi spec `{source}`: {error}"))
        })?;
        Self::parse_text(&text, hint, None, base_override)
    }

    /// Parses an in-memory document, trying JSON first and YAML second.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is neither valid JSON nor
    /// valid YAML, or when the resulting document is not an object or has no
    /// usable server URL.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self> {
        Self::parse_text(text, FormatHint::Auto, None, None)
    }

    /// Builds a spec from an already-parsed document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `document` is not an object, has no
    /// usable server URL, or cannot be interpreted as an `OpenAPI` document.
    pub fn from_value(document: Value) -> Result<Self> {
        Self::assemble(document, None, None)
    }

    /// Returns the base URL every request is addressed against (no trailing `/`).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the collected operations, in document order.
    #[must_use]
    pub fn operations(&self) -> &[OpenApiOperation] {
        &self.operations
    }

    /// Returns the raw document, for callers that need the untouched spec.
    #[must_use]
    pub fn document(&self) -> &Value {
        &self.document
    }

    /// Returns the URL the spec was fetched from, when it came over HTTP.
    #[must_use]
    pub fn source_url(&self) -> Option<&str> {
        self.source_url.as_deref()
    }

    /// Parses `text` with `hint` and assembles the spec.
    fn parse_text(
        text: &str,
        hint: FormatHint,
        source_url: Option<&str>,
        base_override: Option<&str>,
    ) -> Result<Self> {
        let document = match hint {
            FormatHint::Json => serde_json::from_str::<Value>(text).map_err(|error| {
                Error::validation(format!("openapi spec is not valid JSON: {error}"))
            })?,
            FormatHint::Yaml => serde_yaml_ng::from_str::<Value>(text).map_err(|error| {
                Error::validation(format!("openapi spec is not valid YAML: {error}"))
            })?,
            FormatHint::Auto => parse_auto(text)?,
        };
        Self::assemble(document, source_url, base_override)
    }

    /// Resolves the base URL and collects the operations of `document`.
    fn assemble(
        document: Value,
        source_url: Option<&str>,
        base_override: Option<&str>,
    ) -> Result<Self> {
        if !document.is_object() {
            return Err(Error::validation(
                "openapi spec must be a JSON/YAML object at the top level",
            ));
        }
        let base_url = resolve_base_url(&document, source_url, base_override)?;
        let operations = collect_operations(&document)?;
        tracing::debug!(
            base_url = %base_url,
            operations = operations.len(),
            "parsed OpenAPI spec"
        );
        Ok(Self {
            document,
            base_url,
            operations,
            source_url: source_url.map(ToOwned::to_owned),
        })
    }
}

/// Parses `text` as JSON, falling back to YAML.
fn parse_auto(text: &str) -> Result<Value> {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => Ok(value),
        Err(json_error) => serde_yaml_ng::from_str::<Value>(text).map_err(|yaml_error| {
            Error::validation(format!(
                "openapi spec is neither valid JSON ({json_error}) nor valid YAML ({yaml_error})"
            ))
        }),
    }
}

/// Chooses the parser for `source` from its extension.
fn format_hint(source: &str) -> FormatHint {
    let path = source.split(['?', '#']).next().unwrap_or(source);
    let suffix = Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match suffix.as_str() {
        "json" => FormatHint::Json,
        "yaml" | "yml" => FormatHint::Yaml,
        _ => FormatHint::Auto,
    }
}

/// Resolves the base URL from `params.base_url`, `servers[0].url`, and the source.
fn resolve_base_url(
    document: &Value,
    source_url: Option<&str>,
    base_override: Option<&str>,
) -> Result<String> {
    let declared = document
        .get("servers")
        .and_then(Value::as_array)
        .and_then(|servers| servers.first())
        .and_then(|server| server.get("url"))
        .and_then(Value::as_str);
    let raw = base_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| declared.map(str::trim).filter(|value| !value.is_empty()));
    let Some(raw) = raw else {
        return Err(Error::validation(
            "openapi spec declares no server URL; set `params.base_url` or `servers[0].url`",
        ));
    };
    let absolute = raw.starts_with("http://") || raw.starts_with("https://");
    let resolved = if absolute {
        raw.to_owned()
    } else if let Some(source) = source_url {
        let base = Url::parse(source).map_err(|error| {
            Error::validation(format!("invalid openapi spec URL `{source}`: {error}"))
        })?;
        base.join(raw)
            .map_err(|error| {
                Error::validation(format!("invalid openapi server URL `{raw}`: {error}"))
            })?
            .to_string()
    } else {
        return Err(Error::validation(format!(
            "openapi server URL `{raw}` is relative and the spec was not loaded from a URL"
        )));
    };
    let resolved = resolved.trim_end_matches('/');
    if resolved.is_empty() {
        return Err(Error::validation("openapi server URL is empty"));
    }
    Ok(resolved.to_owned())
}

/// Collects every operation of `document`, de-duplicating names.
fn collect_operations(document: &Value) -> Result<Vec<OpenApiOperation>> {
    let Some(paths) = document.get("paths").and_then(Value::as_object) else {
        tracing::debug!("openapi spec declares no `paths` object");
        return Ok(Vec::new());
    };
    let mut used_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut operations = Vec::new();
    for (path, path_item) in paths {
        let Some(path_item) = path_item.as_object() else {
            continue;
        };
        let path_parameters = path_item
            .get("parameters")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (key, operation) in path_item {
            let Some(lower_method) = method_for(key) else {
                continue;
            };
            if !operation.is_object() {
                continue;
            }
            let method = lower_method.to_ascii_uppercase();
            let parameters = merged_parameters(document, &path_parameters, operation)?;
            let (body_schema, body_required) = request_body(document, operation)?;
            let generated = generate_name(&method, path, operation);
            let name = unique_name(&mut used_names, generated);
            let description = describe(&method, path, operation, &parameters, body_schema.as_ref());
            let input_schema = flat_schema(&parameters, body_schema.as_ref());
            operations.push(OpenApiOperation {
                name,
                description,
                method,
                path: path.clone(),
                parameters,
                body_schema,
                body_required,
                input_schema,
            });
        }
    }
    Ok(operations)
}

/// Maps a path-item key onto a lower-case HTTP method.
fn method_for(key: &str) -> Option<&'static str> {
    let lower = key.to_ascii_lowercase();
    let lower = lower.as_str();
    OPERATION_METHODS
        .iter()
        .find(|method| **method == lower)
        .copied()
}

/// Builds the merged parameter list, letting operation entries win on `(in, name)`.
fn merged_parameters(
    document: &Value,
    path_parameters: &[Value],
    operation: &Value,
) -> Result<Vec<OpenApiParameter>> {
    let mut raw: Vec<&Value> = path_parameters.iter().collect();
    if let Some(parameters) = operation.get("parameters").and_then(Value::as_array) {
        raw.extend(parameters.iter());
    }
    let mut merged: Vec<OpenApiParameter> = Vec::new();
    for entry in raw {
        let Some(parameter) = parse_parameter(document, entry)? else {
            continue;
        };
        match merged.iter_mut().find(|existing| {
            existing.location == parameter.location && existing.name == parameter.name
        }) {
            Some(existing) => *existing = parameter,
            None => merged.push(parameter),
        }
    }
    Ok(merged)
}

/// Parses one parameter object; `None` for cookie parameters and nameless entries.
fn parse_parameter(document: &Value, entry: &Value) -> Result<Option<OpenApiParameter>> {
    let resolved = maybe_ref(document, entry)?;
    let Some(object) = resolved.as_object() else {
        return Ok(None);
    };
    let Some(name) = object
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
    else {
        tracing::debug!("skipping openapi parameter without a name");
        return Ok(None);
    };
    let location = object
        .get("in")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if location == "cookie" {
        tracing::debug!(parameter = name, "skipping cookie parameter");
        return Ok(None);
    }
    if !matches!(location.as_str(), "path" | "query" | "header") {
        tracing::debug!(
            parameter = name,
            location = %location,
            "skipping unsupported parameter location"
        );
        return Ok(None);
    }
    let schema = match object.get("schema") {
        Some(schema) => resolve_schema(document, schema)?,
        None => Value::Object(Map::new()),
    };
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
        .map(ToOwned::to_owned);
    Ok(Some(OpenApiParameter {
        name: name.to_owned(),
        location,
        required: object
            .get("required")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        schema,
        description,
    }))
}

/// Resolves the JSON request-body schema and its required flag.
fn request_body(document: &Value, operation: &Value) -> Result<(Option<Value>, bool)> {
    let Some(raw) = operation.get("requestBody") else {
        return Ok((None, false));
    };
    let body = maybe_ref(document, raw)?;
    let required = body
        .get("required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let schema = body
        .get("content")
        .and_then(Value::as_object)
        .and_then(|content| content.get("application/json"))
        .and_then(|media| media.get("schema"))
        .map(|schema| resolve_schema(document, schema))
        .transpose()?;
    Ok((schema, required))
}

/// Generates the un-de-duplicated tool name of one operation.
fn generate_name(method: &str, path: &str, operation: &Value) -> String {
    let from_id = operation
        .get("operationId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(|id| id.split("__").next().unwrap_or(id).trim())
        .filter(|id| !id.is_empty());
    let from_summary = operation
        .get("summary")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|summary| !summary.is_empty());
    let candidate = from_id
        .or(from_summary)
        .map_or_else(|| format!("{method}_{path}"), ToOwned::to_owned);
    candidate.chars().take(MAX_TOOL_NAME_CHARS).collect()
}

/// De-duplicates `name` with a `_2`, `_3`, … suffix, tracking counts per name.
fn unique_name(used: &mut BTreeMap<String, usize>, name: String) -> String {
    let count = used.entry(name.clone()).or_insert(0);
    *count = count.saturating_add(1);
    if *count == 1 {
        name
    } else {
        let generated = format!("{name}_{count}");
        tracing::debug!(
            name = %name,
            generated = %generated,
            "openapi tool name collision"
        );
        generated
    }
}

/// Builds the tool description, appending parameter and body documentation.
fn describe(
    method: &str,
    path: &str,
    operation: &Value,
    parameters: &[OpenApiParameter],
    body_schema: Option<&Value>,
) -> String {
    let mut description = operation
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
        .or_else(|| {
            operation
                .get("summary")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|summary| !summary.is_empty())
        })
        .map_or_else(|| format!("Executes {method} {path}"), ToOwned::to_owned);
    for parameter in parameters {
        if let Some(text) = parameter.description.as_deref() {
            description.push_str("\n\n");
            description.push_str(&parameter.name);
            description.push_str(": ");
            description.push_str(text);
        }
    }
    if let Some(text) = body_schema
        .and_then(|schema| schema.get("description"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        description.push_str("\n\n");
        description.push_str(text);
    }
    description
}

/// Flattens parameters and JSON-body properties into one tool input schema.
fn flat_schema(parameters: &[OpenApiParameter], body_schema: Option<&Value>) -> Value {
    let mut properties = Map::new();
    let mut required: Vec<String> = Vec::new();
    for parameter in parameters {
        let mut schema = parameter.schema.clone();
        if let (Some(description), Value::Object(map)) =
            (parameter.description.as_deref(), &mut schema)
        {
            let documented = map
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.trim().is_empty());
            if !documented {
                map.insert(
                    "description".to_owned(),
                    Value::String(description.to_owned()),
                );
            }
        }
        properties.insert(parameter.name.clone(), schema);
        if parameter.required && !required.contains(&parameter.name) {
            required.push(parameter.name.clone());
        }
    }
    if let Some(body) = body_schema {
        if let Some(body_properties) = body.get("properties").and_then(Value::as_object) {
            for (name, schema) in body_properties {
                // A declared parameter wins a name collision with a body property.
                properties
                    .entry(name.clone())
                    .or_insert_with(|| schema.clone());
            }
        }
        if let Some(body_required) = body.get("required").and_then(Value::as_array) {
            for name in body_required.iter().filter_map(Value::as_str) {
                if !required.iter().any(|existing| existing == name) {
                    required.push(name.to_owned());
                }
            }
        }
    }
    let mut schema = Map::new();
    schema.insert("type".to_owned(), Value::String("object".to_owned()));
    schema.insert("properties".to_owned(), Value::Object(properties));
    schema.insert(
        "required".to_owned(),
        Value::Array(required.into_iter().map(Value::String).collect()),
    );
    Value::Object(schema)
}

/// Resolves `value` when it is a `$ref` object, otherwise clones it.
fn maybe_ref(document: &Value, value: &Value) -> Result<Value> {
    match value.get("$ref").and_then(Value::as_str) {
        Some(reference) => resolve_ref(document, reference),
        None => Ok(value.clone()),
    }
}

/// Resolves a local `#/...` JSON-pointer reference, following `$ref` chains.
///
/// References that do not start with `#` (remote/file references) are rejected,
/// and a chain longer than 32 hops or a reference that reaches itself is a
/// validation error.
pub(crate) fn resolve_ref(document: &Value, reference: &str) -> Result<Value> {
    let mut visited: Vec<&str> = Vec::new();
    let mut current = reference.trim();
    for _ in 0..MAX_REF_DEPTH {
        if visited.contains(&current) {
            return Err(Error::validation(format!(
                "cyclic $ref `{current}` in the OpenAPI spec"
            )));
        }
        visited.push(current);
        let target = pointer(document, current)?;
        match target.get("$ref").and_then(Value::as_str) {
            Some(next) => current = next,
            None => return Ok(target.clone()),
        }
    }
    Err(Error::validation(format!(
        "$ref `{reference}` exceeds the maximum chain depth of {MAX_REF_DEPTH}"
    )))
}

/// Resolves a schema whose top-level `$ref` must be inlined; sibling keys win.
///
/// Nested references inside `properties` are intentionally left in place.
pub(crate) fn resolve_schema(document: &Value, schema: &Value) -> Result<Value> {
    let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
        return Ok(schema.clone());
    };
    let resolved = resolve_ref(document, reference)?;
    let Some(siblings) = schema.as_object() else {
        return Ok(resolved);
    };
    let Value::Object(mut merged) = resolved else {
        return Ok(resolved);
    };
    for (key, value) in siblings {
        if key != "$ref" {
            merged.insert(key.clone(), value.clone());
        }
    }
    Ok(Value::Object(merged))
}

/// Walks one `#/...` JSON pointer.
fn pointer<'a>(document: &'a Value, reference: &str) -> Result<&'a Value> {
    let Some(pointer) = reference.strip_prefix('#') else {
        return Err(Error::validation(format!(
            "unsupported $ref `{reference}`: only local `#/...` pointers are supported"
        )));
    };
    if pointer.is_empty() {
        return Ok(document);
    }
    let Some(pointer) = pointer.strip_prefix('/') else {
        return Err(Error::validation(format!(
            "malformed $ref `{reference}`: expected `#/...`"
        )));
    };
    let mut current = document;
    for token in pointer.split('/') {
        let token = token.replace("~1", "/").replace("~0", "~");
        current = match current {
            Value::Object(map) => map.get(&token),
            Value::Array(items) => token
                .parse::<usize>()
                .ok()
                .and_then(|index| items.get(index)),
            _ => None,
        }
        .ok_or_else(|| {
            Error::validation(format!(
                "$ref `{reference}` does not resolve: `{token}` is missing"
            ))
        })?;
    }
    Ok(current)
}
