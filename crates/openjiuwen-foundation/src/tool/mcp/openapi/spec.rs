//! `OpenAPI` document loading, `$ref` resolution, and operation collection.
//!
//! Ports the spec-facing half of
//! `openjiuwen/core/foundation/tool/mcp/client/openapi_client.py` (`load_conf`,
//! the `servers[0].url` handling, and the `parse_openapi_to_http_routes`
//! operation walk) without the `fastmcp`/`jsonschema-path` dependencies:
//! operations are collected directly from the `paths` object and their tool
//! schemas are flattened here.
//!
//! `$ref` handling lives in [`refs`](super::refs): local `#/...` pointers, local
//! files (relative to the declaring document) and `http(s)` URLs all resolve,
//! while the *inline* constructors ([`OpenApiSpec::from_str`],
//! [`OpenApiSpec::from_value`]) have no file or URL base and can therefore only
//! follow local pointers. Resolution is deliberately **shallow**: the top-level
//! `$ref` of a schema (or a parameter / request body / response object) is
//! resolved, while nested references inside `properties` are left untouched so
//! the emitted JSON Schema keeps its `components` pointers.
//!
//! Only an `application/json` request body is mapped onto tool arguments: an
//! operation whose body is declared as `multipart/form-data` or
//! `application/x-www-form-urlencoded` keeps `body_schema = None` and logs a
//! `tracing::warn!` naming the ignored media types, so the omission is visible
//! rather than silent.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use super::refs::{DocumentBase, RefContext};

/// Maximum length of a generated tool name, in characters.
const MAX_TOOL_NAME_CHARS: usize = 64;

/// The operation keys of a path item, in the order they are documented.
const OPERATION_METHODS: [&str; 7] = ["get", "put", "post", "delete", "patch", "head", "options"];

/// Deadline applied when the spec is fetched from a URL.
const SPEC_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// One path/query/header/cookie parameter of an `OpenAPI` operation.
///
/// Every declared parameter is collected, including `in: cookie` ones: cookie
/// values travel in one merged `Cookie` request header (see `call::cookie_header`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OpenApiParameter {
    /// Parameter name, exactly as declared in the document.
    pub name: String,
    /// Where the value travels: `path`, `query`, `header`, or `cookie`
    /// (lower-case).
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
/// Loading is side-effect free apart from the `GET`s a URL source and the
/// documents its `$ref`s reach need; a filesystem source is read and never
/// written. Every referenced document is loaded eagerly, so a spec with a broken
/// reference fails here instead of on its first call.
#[derive(Debug, Clone)]
pub struct OpenApiSpec {
    refs: RefContext,
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
    /// Every document the spec references is loaded here (over `http` for URL
    /// sources), so the returned spec resolves `$ref`s without further I/O.
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
        if source.starts_with("http://") || source.starts_with("https://") {
            let request = HttpRequest::get(source).timeout(SPEC_FETCH_TIMEOUT);
            let response = http.execute(request).await?;
            if !response.is_success() {
                return Err(Error::transport(format!(
                    "failed to fetch OpenAPI spec from {source}: HTTP {}",
                    response.status
                )));
            }
            let document = parse_document(&response.text(), source)?;
            let refs =
                RefContext::load(document, DocumentBase::Url(source.to_owned()), http).await?;
            return Self::assemble(refs, Some(source), base_override);
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
        let document = parse_document(&text, source)?;
        let refs =
            RefContext::load(document, DocumentBase::File(Path::new(source).into()), http).await?;
        Self::assemble(refs, None, base_override)
    }

    /// Parses an in-memory document, trying JSON first and YAML second.
    ///
    /// The document has no file or URL base, so only local `#/...` pointers
    /// resolve; a `$ref` into another document is a validation error. Load the
    /// spec with [`OpenApiSpec::load`] to resolve external references.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is neither valid JSON nor
    /// valid YAML, or when the resulting document is not an object or has no
    /// usable server URL.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self> {
        Self::assemble(RefContext::inline(parse_auto(text)?), None, None)
    }

    /// Builds a spec from an already-parsed document.
    ///
    /// As with [`OpenApiSpec::from_str`], only local `#/...` pointers resolve.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `document` is not an object, has no
    /// usable server URL, or cannot be interpreted as an `OpenAPI` document.
    pub fn from_value(document: Value) -> Result<Self> {
        Self::assemble(RefContext::inline(document), None, None)
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
        self.refs.root_document()
    }

    /// Returns the URL the spec was fetched from, when it came over HTTP.
    #[must_use]
    pub fn source_url(&self) -> Option<&str> {
        self.source_url.as_deref()
    }

    /// Resolves `reference` against this spec, following `$ref` chains.
    ///
    /// Local `#/...` pointers resolve inside [`OpenApiSpec::document`];
    /// `common.yaml#/...` and `https://…#/...` references resolve inside the
    /// documents the spec pulled in at load time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed reference, a document or
    /// pointer that does not exist, a chain longer than the documented depth, or
    /// a cycle; the `refs` module documents the resolution rules.
    pub fn resolve(&self, reference: &str) -> Result<Value> {
        self.refs.resolve(reference)
    }

    /// Resolves the base URL and collects the operations of the root document.
    fn assemble(
        refs: RefContext,
        source_url: Option<&str>,
        base_override: Option<&str>,
    ) -> Result<Self> {
        let document = refs.root_document();
        if !document.is_object() {
            return Err(Error::validation(
                "openapi spec must be a JSON/YAML object at the top level",
            ));
        }
        let base_url = resolve_base_url(document, source_url, base_override)?;
        let operations = collect_operations(&refs)?;
        tracing::debug!(
            base_url = %base_url,
            operations = operations.len(),
            "parsed OpenAPI spec"
        );
        Ok(Self {
            refs,
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

/// Parses one `OpenAPI` document, choosing the parser from `source`'s extension.
///
/// `.json` is parsed strictly, `.yaml`/`.yml` strictly as YAML, and every other
/// source tries JSON first and YAML second.
pub(super) fn parse_document(text: &str, source: &str) -> Result<Value> {
    match format_hint(source) {
        FormatHint::Json => serde_json::from_str::<Value>(text).map_err(|error| {
            Error::validation(format!(
                "openapi document `{source}` is not valid JSON: {error}"
            ))
        }),
        FormatHint::Yaml => serde_yaml_ng::from_str::<Value>(text).map_err(|error| {
            Error::validation(format!(
                "openapi document `{source}` is not valid YAML: {error}"
            ))
        }),
        FormatHint::Auto => parse_auto(text),
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

/// Collects every operation of the root document, de-duplicating names.
fn collect_operations(refs: &RefContext) -> Result<Vec<OpenApiOperation>> {
    let Some(paths) = refs.root_document().get("paths").and_then(Value::as_object) else {
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
            let parameters = merged_parameters(refs, &path_parameters, operation)?;
            let (body_schema, body_required) = request_body(refs, operation)?;
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
    refs: &RefContext,
    path_parameters: &[Value],
    operation: &Value,
) -> Result<Vec<OpenApiParameter>> {
    let mut raw: Vec<&Value> = path_parameters.iter().collect();
    if let Some(parameters) = operation.get("parameters").and_then(Value::as_array) {
        raw.extend(parameters.iter());
    }
    let mut merged: Vec<OpenApiParameter> = Vec::new();
    for entry in raw {
        let Some(parameter) = parse_parameter(refs, entry)? else {
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

/// Parses one parameter object; `None` for nameless entries and unknown locations.
fn parse_parameter(refs: &RefContext, entry: &Value) -> Result<Option<OpenApiParameter>> {
    let resolved = refs.maybe_ref(entry)?;
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
    if !matches!(location.as_str(), "path" | "query" | "header" | "cookie") {
        tracing::debug!(
            parameter = name,
            location = %location,
            "skipping unsupported parameter location"
        );
        return Ok(None);
    }
    let schema = match object.get("schema") {
        Some(schema) => refs.resolve_schema(schema)?,
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
///
/// Only `application/json` bodies are mapped onto tool arguments; any other
/// declared media type (`multipart/form-data`, `application/x-www-form-urlencoded`,
/// …) leaves `body_schema` at `None` and is reported once per operation with a
/// `tracing::warn!`, so the omission is never silent.
fn request_body(refs: &RefContext, operation: &Value) -> Result<(Option<Value>, bool)> {
    let Some(raw) = operation.get("requestBody") else {
        return Ok((None, false));
    };
    let body = refs.maybe_ref(raw)?;
    let required = body
        .get("required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let content = body.get("content").and_then(Value::as_object);
    let schema = content
        .and_then(|content| content.get("application/json"))
        .and_then(|media| media.get("schema"))
        .map(|schema| refs.resolve_schema(schema))
        .transpose()?;
    if schema.is_none() {
        if let Some(content) = content.filter(|content| !content.is_empty()) {
            let media_types = content.keys().cloned().collect::<Vec<_>>().join(", ");
            tracing::warn!(
                media_types = %media_types,
                "ignoring the OpenAPI request body: only `application/json` is mapped onto \
                 tool arguments"
            );
        }
    }
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
