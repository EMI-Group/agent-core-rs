//! REST tools: call an HTTP endpoint described by [`RestfulApiConfig`].
//!
//! Ports `openjiuwen/core/foundation/tool/service_api/restful_api.py`
//! (`RestfulApiCard`, `RestfulApi`), `api_param_mapper.py` (`APIParamMapper`,
//! `APIParamLocation`) and the JSON/text half of `response_parser.py`.
//!
//! The card keeps the model-facing shape, so there is no `RestfulApiCard` type
//! here: a [`ToolCard`] plus a [`RestfulApiConfig`] carries the same data, and
//! [`RestfulApiTool`] joins them with a parameter mapper and the shared
//! [`HttpClient`].
//!
//! Deviations from Python, all documented on the affected item:
//! - multipart `form` parameters (and their `form_handler_type` handlers) are out
//!   of scope; a `"location": "form"` property is a validation error;
//! - gzip/deflate decompression is not reimplemented — the transport negotiates
//!   and decodes it, so the parser only splits JSON from text;
//! - the SSL/auth callback framework is not ported: authentication material
//!   travels in the configured headers, query, or path parameters.
//!
//! ```
//! use openjiuwen_foundation::tool::card::ToolCard;
//! use openjiuwen_foundation::tool::restful::{RestfulApiConfig, RestfulApiTool, HttpMethod};
//! use serde_json::json;
//!
//! let card = ToolCard::new(
//!     "get_user",
//!     "fetches a user",
//!     json!({"type": "object", "properties": {
//!         "id": {"type": "integer", "location": "path"},
//!     }}),
//! );
//! let config = RestfulApiConfig::new("https://api.example.com/users/{id}")
//!     .with_method(HttpMethod::Get);
//! let tool = RestfulApiTool::new(card, config).unwrap();
//! assert_eq!(tool.config().url, "https://api.example.com/users/{id}");
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse, Method};
use regex::Regex;
use reqwest::header::HeaderMap;
use schemars::JsonSchema;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use url::Url;

use super::card::ToolCard;
use super::context::ToolContext;
use super::schema::{apply_defaults, ensure_object_schema, validate_inputs};
use super::traits::Tool;

/// Default request timeout, in seconds.
///
/// Python parity: `RestfulApiCard.timeout` defaults to `60.0`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Default cap on a response body, in bytes (10 MiB).
///
/// Python parity: `RestfulApiCard.max_response_byte_size` defaults to
/// `10 * 1024 * 1024`.
pub const DEFAULT_MAX_RESPONSE_BYTES: u64 = 10 * 1024 * 1024;

/// URL path-placeholder pattern, e.g. `{id}` in `/users/{id}`.
///
/// The pattern is a compile-time constant, so it cannot fail to compile; the
/// `expect` below documents that invariant instead of hiding it.
#[allow(clippy::expect_used)]
static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\{(\w+)\}").expect("the path-placeholder pattern is a valid regex")
});

/// HTTP methods accepted by a REST tool.
///
/// Python parity: `RestfulApiCard.SUPPORTED_METHODS`. Serializes as the
/// upper-case wire form (`"GET"`) and deserializes case-insensitively, so
/// `"get"`, `"Get"` and `"GET"` all produce [`HttpMethod::Get`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, JsonSchema)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    /// `GET`.
    Get,
    /// `POST`; the default, matching `RestfulApiCard.method`.
    #[default]
    Post,
    /// `PUT`.
    Put,
    /// `PATCH`.
    Patch,
    /// `DELETE`.
    Delete,
    /// `HEAD`.
    Head,
    /// `OPTIONS`.
    Options,
}

impl HttpMethod {
    /// Returns the method name as used on the wire, e.g. `"GET"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }

    /// Returns whether the mapped body bucket travels as query parameters.
    ///
    /// Python parity: `RestfulApi._async_request` sends the body bucket through
    /// `params` for `GET`, `HEAD`, `OPTIONS` and `DELETE`, and as a JSON body for
    /// `POST`, `PUT` and `PATCH`. This returns `true` for the former group.
    #[must_use]
    pub fn is_bodyless(self) -> bool {
        matches!(self, Self::Get | Self::Head | Self::Options | Self::Delete)
    }
}

impl From<HttpMethod> for Method {
    fn from(method: HttpMethod) -> Self {
        match method {
            HttpMethod::Get => Self::GET,
            HttpMethod::Post => Self::POST,
            HttpMethod::Put => Self::PUT,
            HttpMethod::Patch => Self::PATCH,
            HttpMethod::Delete => Self::DELETE,
            HttpMethod::Head => Self::HEAD,
            HttpMethod::Options => Self::OPTIONS,
        }
    }
}

impl FromStr for HttpMethod {
    type Err = Error;

    /// Parses a method name case-insensitively.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the name is not one of the seven
    /// supported methods.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "GET" => Ok(Self::Get),
            "POST" => Ok(Self::Post),
            "PUT" => Ok(Self::Put),
            "PATCH" => Ok(Self::Patch),
            "DELETE" => Ok(Self::Delete),
            "HEAD" => Ok(Self::Head),
            "OPTIONS" => Ok(Self::Options),
            other => Err(Error::validation(format!(
                "unsupported http method '{other}': only GET, POST, PUT, PATCH, DELETE, HEAD and OPTIONS are supported"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for HttpMethod {
    /// Deserializes a method name case-insensitively.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(de::Error::custom)
    }
}

/// Where an input parameter travels in the HTTP request.
///
/// Python parity: `APIParamLocation` in
/// `openjiuwen/core/foundation/tool/service_api/api_param_mapper.py`. Python's
/// `FORM` variant is not ported: multipart form handling is out of scope, and a
/// `"location": "form"` property is rejected as a validation error.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ParamLocation {
    /// URL query parameter (`?key=value`).
    Query,
    /// URL path parameter (`/users/{id}`).
    Path,
    /// JSON request body; the default when the schema declares no location.
    #[default]
    Body,
    /// HTTP request header.
    Header,
}

impl ParamLocation {
    /// Every location, in a stable order.
    pub const ALL: [Self; 4] = [Self::Path, Self::Query, Self::Header, Self::Body];

    /// Locations whose configured defaults are merged into the mapped values.
    ///
    /// Python parity: `APIParamMapper.map` merges defaults for `PATH`, `QUERY`
    /// and `HEADER` only.
    pub const DEFAULTABLE: [Self; 3] = [Self::Path, Self::Query, Self::Header];

    /// Returns the lower-case wire name, e.g. `"query"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Path => "path",
            Self::Body => "body",
            Self::Header => "header",
        }
    }
}

impl FromStr for ParamLocation {
    type Err = Error;

    /// Parses a location name case-insensitively.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for anything but `query`, `path`, `body` and
    /// `header` — including Python's `form`, which is out of scope.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "query" => Ok(Self::Query),
            "path" => Ok(Self::Path),
            "body" => Ok(Self::Body),
            "header" => Ok(Self::Header),
            other => Err(Error::validation(format!(
                "unsupported parameter location '{other}': only query, path, body and header are supported"
            ))),
        }
    }
}

/// Input parameters grouped by where they travel.
pub type ParamMap = BTreeMap<ParamLocation, BTreeMap<String, Value>>;

/// Splits tool inputs into query, path, body and header buckets.
///
/// Python parity: `APIParamMapper`. The mapper is built from the card's input
/// schema (each property may carry a `"location"` keyword) and the config's
/// default query, header, and path parameters.
///
/// ```
/// use openjiuwen_foundation::tool::restful::{ApiParamMapper, ParamLocation};
/// use serde_json::json;
///
/// let mapper = ApiParamMapper::new(json!({
///     "type": "object",
///     "properties": {
///         "id": {"type": "integer", "location": "path"},
///         "name": {"type": "string", "location": "query"},
///     },
/// }));
/// let params = mapper.map(&json!({"id": 7, "name": "bob"})).unwrap();
/// assert_eq!(params[&ParamLocation::Path]["id"], json!(7));
/// assert_eq!(params[&ParamLocation::Query]["name"], json!("bob"));
/// assert_eq!(
///     mapper.render_url("https://api.example.com/users/{id}", &params).unwrap(),
///     "https://api.example.com/users/7",
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ApiParamMapper {
    schema: Value,
    defaults: BTreeMap<ParamLocation, BTreeMap<String, Value>>,
}

impl ApiParamMapper {
    /// Creates a mapper for `schema` with no configured defaults.
    #[must_use]
    pub fn new(schema: Value) -> Self {
        Self::with_defaults(schema, BTreeMap::new(), BTreeMap::new(), BTreeMap::new())
    }

    /// Creates a mapper with the config's default queries, headers, and paths.
    ///
    /// Defaults are merged into the mapped values of their own location, with
    /// input values winning unless they are `null` or the empty string (Python
    /// parity).
    #[must_use]
    pub fn with_defaults(
        schema: Value,
        queries: BTreeMap<String, Value>,
        headers: BTreeMap<String, Value>,
        paths: BTreeMap<String, Value>,
    ) -> Self {
        let mut defaults = BTreeMap::new();
        defaults.insert(ParamLocation::Query, queries);
        defaults.insert(ParamLocation::Header, headers);
        defaults.insert(ParamLocation::Path, paths);
        Self { schema, defaults }
    }

    /// Returns the input schema this mapper was built from.
    #[must_use]
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Returns the configured defaults for `location`, if the location takes any.
    #[must_use]
    pub fn defaults_for(&self, location: ParamLocation) -> Option<&BTreeMap<String, Value>> {
        self.defaults.get(&location)
    }

    /// Splits `inputs` into the four [`ParamLocation`] buckets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a schema property declares a location
    /// other than `query`, `path`, `body` or `header`; the message names the
    /// property. Python's multipart `form` location falls into this category.
    ///
    /// Only properties declared in `schema.properties` are mapped (Python
    /// iterates the schema, so undeclared input keys are dropped). When the
    /// schema declares no `properties` at all, every input key is sent as a body
    /// parameter — a documented deviation, because Python would silently drop
    /// them.
    pub fn map(&self, inputs: &Value) -> Result<ParamMap> {
        let mut result = empty_param_map();
        let input_object = inputs.as_object();

        match self.schema.get("properties").and_then(Value::as_object) {
            Some(properties) => {
                for (name, property_schema) in properties {
                    let Some(value) = input_object.and_then(|object| object.get(name)) else {
                        continue;
                    };
                    let location = location_for(name, property_schema)?;
                    result
                        .entry(location)
                        .or_default()
                        .insert(name.clone(), value.clone());
                }
            }
            None => {
                if let Some(object) = input_object {
                    let body = result.entry(ParamLocation::Body).or_default();
                    body.extend(
                        object
                            .iter()
                            .map(|(key, value)| (key.clone(), value.clone())),
                    );
                }
            }
        }

        self.merge_defaults(&mut result);
        Ok(result)
    }

    /// Substitutes every `{name}` placeholder in `url` from the path bucket.
    ///
    /// String values are used verbatim; numbers, booleans, arrays and objects are
    /// rendered as compact JSON (Python converts through `str()`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a placeholder has no path value.
    pub fn render_url(&self, url: &str, params: &ParamMap) -> Result<String> {
        let empty = BTreeMap::new();
        let path_params = params.get(&ParamLocation::Path).unwrap_or(&empty);

        let mut missing: Vec<&str> = Vec::new();
        for captures in PLACEHOLDER.captures_iter(url) {
            let Some(name) = captures.get(1).map(|matched| matched.as_str()) else {
                continue;
            };
            if !path_params.contains_key(name) && !missing.contains(&name) {
                missing.push(name);
            }
        }
        if !missing.is_empty() {
            return Err(Error::validation(format!(
                "missing values for URL path parameters {} in '{url}'",
                quoted_set(&missing)
            )));
        }

        let rendered = PLACEHOLDER.replace_all(url, |captures: &regex::Captures<'_>| {
            captures
                .get(1)
                .and_then(|matched| path_params.get(matched.as_str()))
                .map_or_else(String::new, value_to_text)
        });
        Ok(rendered.into_owned())
    }

    /// Merges the configured defaults into `result`.
    ///
    /// Python parity: defaults are loaded first and then overridden by input
    /// values that are neither `null` nor `""`.
    fn merge_defaults(&self, result: &mut ParamMap) {
        for location in ParamLocation::DEFAULTABLE {
            let Some(defaults) = self.defaults.get(&location) else {
                continue;
            };
            let mut merged = defaults.clone();
            let mapped = std::mem::take(result.entry(location).or_default());
            for (key, value) in mapped {
                if !is_blank(&value) {
                    merged.insert(key, value);
                }
            }
            result.insert(location, merged);
        }
    }
}

/// The transport configuration of a REST tool.
///
/// Python parity: the transport fields of `RestfulApiCard`. The card keeps the
/// model-facing description; this struct keeps the url, method, defaults,
/// timeout, and response cap, so it can be persisted and edited independently.
///
/// `timeout` serializes as a number of seconds (integers and floats are both
/// accepted on the way in), matching Python's float `timeout` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RestfulApiConfig {
    /// Absolute `http`/`https` URL; `{name}` placeholders are filled from the
    /// path bucket.
    pub url: String,
    /// HTTP method of the call; `POST` by default.
    pub method: HttpMethod,
    /// Default request headers, merged with the `header`-located inputs.
    pub headers: BTreeMap<String, Value>,
    /// Default query parameters, merged with the `query`-located inputs.
    pub queries: BTreeMap<String, Value>,
    /// Default path parameters, merged with the `path`-located inputs.
    pub paths: BTreeMap<String, Value>,
    /// Total request timeout, in seconds on the wire.
    #[serde(with = "duration_secs")]
    #[schemars(with = "f64")]
    pub timeout: Duration,
    /// Reject responses whose body exceeds this many bytes.
    pub max_response_bytes: u64,
}

impl Default for RestfulApiConfig {
    /// Python defaults: `POST`, no header/query/path defaults, a 60 second
    /// timeout, and a 10 MiB response cap.
    fn default() -> Self {
        Self {
            url: String::new(),
            method: HttpMethod::Post,
            headers: BTreeMap::new(),
            queries: BTreeMap::new(),
            paths: BTreeMap::new(),
            timeout: DEFAULT_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }
}

impl RestfulApiConfig {
    /// Creates a configuration for `url` with every other field defaulted.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Self::default()
        }
    }

    /// Returns this configuration with the given HTTP method.
    #[must_use]
    pub fn with_method(mut self, method: HttpMethod) -> Self {
        self.method = method;
        self
    }

    /// Returns this configuration with one additional default header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: Value) -> Self {
        self.headers.insert(name.into(), value);
        self
    }

    /// Returns this configuration with one additional default query parameter.
    #[must_use]
    pub fn with_query(mut self, name: impl Into<String>, value: Value) -> Self {
        self.queries.insert(name.into(), value);
        self
    }

    /// Returns this configuration with one additional default path parameter.
    #[must_use]
    pub fn with_path(mut self, name: impl Into<String>, value: Value) -> Self {
        self.paths.insert(name.into(), value);
        self
    }

    /// Returns this configuration with the given request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Returns this configuration with the given response-size cap in bytes.
    #[must_use]
    pub fn with_max_response_bytes(mut self, max_response_bytes: u64) -> Self {
        self.max_response_bytes = max_response_bytes;
        self
    }
}

/// Seconds-based `Duration` serde helpers for [`RestfulApiConfig::timeout`].
mod duration_secs {
    use std::fmt;
    use std::time::Duration;

    use serde::de::{self, Unexpected, Visitor};
    use serde::{Deserializer, Serializer};

    /// Upper bound accepted on the way in, so [`Duration::from_secs_f64`] cannot
    /// overflow (about 31.7 billion years).
    const MAX_SECONDS: f64 = 1e18;

    /// Serializes a duration as a whole number of seconds when it is exact, and
    /// as fractional seconds otherwise.
    pub(super) fn serialize<S>(
        duration: &Duration,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if duration.subsec_nanos() == 0 {
            serializer.serialize_u64(duration.as_secs())
        } else {
            serializer.serialize_f64(duration.as_secs_f64())
        }
    }

    /// Deserializes a duration from integer or float seconds.
    pub(super) fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SecondsVisitor;

        impl Visitor<'_> for SecondsVisitor {
            type Value = Duration;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a non-negative number of seconds")
            }

            fn visit_u64<E: de::Error>(self, seconds: u64) -> std::result::Result<Duration, E> {
                Ok(Duration::from_secs(seconds))
            }

            fn visit_i64<E: de::Error>(self, seconds: i64) -> std::result::Result<Duration, E> {
                u64::try_from(seconds)
                    .map(Duration::from_secs)
                    .map_err(|_| E::invalid_value(Unexpected::Signed(seconds), &self))
            }

            fn visit_f64<E: de::Error>(self, seconds: f64) -> std::result::Result<Duration, E> {
                if !seconds.is_finite() || !(0.0..=MAX_SECONDS).contains(&seconds) {
                    return Err(E::invalid_value(Unexpected::Float(seconds), &self));
                }
                Ok(Duration::from_secs_f64(seconds))
            }
        }

        deserializer.deserialize_any(SecondsVisitor)
    }
}

/// A tool that calls an HTTP endpoint and returns Python's response envelope.
///
/// Python parity: `RestfulApi` in
/// `openjiuwen/core/foundation/tool/service_api/restful_api.py`. Invocations
/// return the envelope object `{"code", "message", "data", "url", "headers"}`
/// and never stream: [`Tool::stream`] reports [`Error::Unsupported`], matching
/// Python's `TOOL_STREAM_NOT_SUPPORTED`.
#[derive(Debug, Clone)]
pub struct RestfulApiTool {
    card: ToolCard,
    config: RestfulApiConfig,
    mapper: ApiParamMapper,
    http: HttpClient,
}

impl RestfulApiTool {
    /// Builds a REST tool from a card and a transport configuration.
    ///
    /// The card's `input_params` schema declares the accepted arguments and their
    /// locations; the config carries the url, method, defaults, timeout, and
    /// response cap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the url is not an absolute `http`/
    /// `https` url, or when a `{name}` placeholder in the url is not declared as
    /// `"location": "path"` in the card's schema. Path properties that the url
    /// does not use are only logged at [`tracing::warn`] level, as in Python.
    pub fn new(card: ToolCard, config: RestfulApiConfig) -> Result<Self> {
        validate_absolute_url(&config.url)?;

        let placeholders = url_placeholders(&config.url);
        let path_properties = path_properties(&card.input_params);
        let missing: Vec<&str> = placeholders
            .iter()
            .filter(|name| !path_properties.contains(*name))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(Error::validation(placeholder_message(
                &card.input_params,
                &missing,
            )));
        }

        let unused: Vec<&str> = path_properties
            .iter()
            .filter(|name| !placeholders.contains(*name))
            .map(String::as_str)
            .collect();
        if !unused.is_empty() {
            tracing::warn!(
                target: "openjiuwen_foundation::tool::restful",
                url = %config.url,
                parameters = ?unused,
                "schema declares path parameters that the url does not use"
            );
        }

        let mapper = ApiParamMapper::with_defaults(
            card.input_params.clone(),
            config.queries.clone(),
            config.headers.clone(),
            config.paths.clone(),
        );
        // The client-level `max_response_bytes` is deliberately left unset: the
        // cap is enforced per call so an oversized body surfaces as
        // `Error::Tool` rather than as a transport error.
        let http = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;

        Ok(Self {
            card,
            config,
            mapper,
            http,
        })
    }

    /// Returns the transport configuration of this tool.
    #[must_use]
    pub fn config(&self) -> &RestfulApiConfig {
        &self.config
    }

    /// Builds the HTTP request for already-mapped `params`.
    fn build_request(
        &self,
        url: &str,
        params: &ParamMap,
        ctx: &ToolContext,
    ) -> Result<HttpRequest> {
        let method = self.config.method;
        let mut request = HttpRequest::new(Method::from(method), url)
            .with_headers(header_pairs(params))
            .queries(query_pairs(params));

        if method.is_bodyless() {
            request = request.queries(
                params
                    .get(&ParamLocation::Body)
                    .map_or_else(Vec::new, query_pairs_for_body),
            );
        } else {
            let body = params.get(&ParamLocation::Body).map_or_else(
                || json!({}),
                |body| Value::Object(body.clone().into_iter().collect()),
            );
            request = request.json(&body)?;
        }

        Ok(request
            .timeout(self.config.timeout)
            .cancel(ctx.cancellation.clone()))
    }

    /// Turns a successful response into the Python envelope.
    fn envelope(response: HttpResponse) -> Result<Value> {
        let message = if response.is_success() {
            "success".to_owned()
        } else {
            canonical_reason(response.status).unwrap_or_default()
        };
        let content_type = response
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let data = if is_json_content_type(content_type) {
            if response.body.is_empty() {
                json!({})
            } else {
                serde_json::from_slice::<Value>(&response.body).map_err(|error| {
                    Error::tool(format!(
                        "restful api response from {} is not valid json: {error}",
                        response.url
                    ))
                })?
            }
        } else {
            Value::String(response.text())
        };
        let response_headers = header_object(&response.headers);

        Ok(json!({
            "code": response.status,
            "message": message,
            "data": data,
            "url": response.url,
            "headers": response_headers,
        }))
    }
}

#[async_trait]
impl Tool for RestfulApiTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;

        // Python `SchemaUtils.format_with_schema`: fill schema defaults, then
        // validate the resulting object against the same schema.
        let schema = ensure_object_schema(self.card.input_params.clone());
        let mut inputs = inputs;
        apply_defaults(&schema, &mut inputs);
        validate_inputs(&schema, &inputs)?;

        let params = self.mapper.map(&inputs)?;
        let url = self.mapper.render_url(&self.config.url, &params)?;
        let request = self.build_request(&url, &params, ctx)?;

        let response = match self.http.execute(request).await {
            Ok(response) => response,
            Err(error) if matches!(error, Error::Timeout(_) | Error::Cancelled) => {
                return Err(error)
            }
            Err(error) => {
                return Err(Error::tool(format!(
                    "restful api request to {} failed: {error}",
                    self.config.url
                )))
            }
        };

        if !response.is_success() {
            return Err(Error::tool(format!(
                "restful api {} {} returned HTTP {}{}",
                self.config.method.as_str(),
                response.url,
                response.status,
                body_excerpt(&response.text())
            )));
        }

        let length = response.body.len() as u64;
        if length > self.config.max_response_bytes {
            return Err(Error::tool(format!(
                "restful api response from {} is {length} bytes, exceeding the {} byte limit",
                response.url, self.config.max_response_bytes
            )));
        }

        Self::envelope(response)
    }

    fn stream(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: ToolContext,
    ) -> BoxStream<'static, Result<Value>> {
        Box::pin(futures::stream::once(async {
            Err(Error::unsupported(
                "restful api tools do not stream".to_owned(),
            ))
        }))
    }
}

/// Creates a map with an empty bucket for every [`ParamLocation`].
fn empty_param_map() -> ParamMap {
    ParamLocation::ALL
        .into_iter()
        .map(|location| (location, BTreeMap::new()))
        .collect()
}

/// Returns the location declared by `property_schema`, defaulting to `Body`.
fn location_for(name: &str, property_schema: &Value) -> Result<ParamLocation> {
    match property_schema.get("location") {
        Some(Value::String(raw)) => raw.parse().map_err(|_: Error| {
            Error::validation(format!(
                "unsupported parameter location '{raw}' for parameter '{name}': only query, path, body and header are supported"
            ))
        }),
        _ => Ok(ParamLocation::Body),
    }
}

/// Returns whether `value` is a value that must not override a configured default.
fn is_blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        _ => false,
    }
}

/// Renders a JSON value as request text (Python's `str()`/`urlencode` behavior).
fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Renders `"{'a', 'b'}"`, the Python set literal used in the validation messages.
fn quoted_set(names: &[&str]) -> String {
    let rendered: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
    format!("{{{}}}", rendered.join(", "))
}

/// Collects the `{name}` placeholders of `url`.
fn url_placeholders(url: &str) -> BTreeSet<String> {
    PLACEHOLDER
        .captures_iter(url)
        .filter_map(|captures| captures.get(1))
        .map(|matched| matched.as_str().to_owned())
        .collect()
}

/// Collects the schema properties declared as `"location": "path"`.
fn path_properties(schema: &Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return names;
    };
    for (name, property_schema) in properties {
        if property_schema
            .get("location")
            .and_then(Value::as_str)
            .is_some_and(|location| location.eq_ignore_ascii_case("path"))
        {
            names.insert(name.clone());
        }
    }
    names
}

/// Builds the Python-shaped message for undeclared url placeholders.
fn placeholder_message(schema: &Value, missing: &[&str]) -> String {
    let first = missing.first().copied().unwrap_or_default();
    let declared = schema
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|properties| !properties.is_empty());

    if declared {
        format!(
            "URL contains path parameters {} that are not defined in input_params schema with 'location': 'path'. \
             Please add them to your schema. Example: '{first}': {{'type': 'string', 'description': 'Parameter description', 'location': 'path'}}",
            quoted_set(missing)
        )
    } else {
        format!(
            "URL contains path parameters {} but input_params schema is not defined. \
             You must define input_params with 'location': 'path' for each path parameter. \
             Example: {{'type': 'object', 'properties': {{'{first}': {{'type': 'string', 'location': 'path'}}}}}}",
            quoted_set(missing)
        )
    }
}

/// Checks that `url` is an absolute `http`/`https` url.
fn validate_absolute_url(url: &str) -> Result<()> {
    let parsed = Url::parse(url)
        .map_err(|error| Error::validation(format!("invalid restful api url '{url}': {error}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(Error::validation(format!(
            "invalid restful api url '{url}': an absolute http or https url is required"
        )));
    }
    Ok(())
}

/// Converts the header bucket into request header pairs.
fn header_pairs(params: &ParamMap) -> Vec<(String, String)> {
    params
        .get(&ParamLocation::Header)
        .map_or_else(Vec::new, |headers| {
            headers
                .iter()
                .map(|(name, value)| (name.clone(), value_to_text(value)))
                .collect()
        })
}

/// Converts the query bucket into request query pairs.
///
/// Arrays expand to one parameter per item (Python `urlencode` behavior).
fn query_pairs(params: &ParamMap) -> Vec<(String, String)> {
    params
        .get(&ParamLocation::Query)
        .map_or_else(Vec::new, query_pairs_for_body)
}

/// Converts one bucket into request query pairs, expanding arrays.
fn query_pairs_for_body(bucket: &BTreeMap<String, Value>) -> Vec<(String, String)> {
    let mut pairs = Vec::with_capacity(bucket.len());
    for (name, value) in bucket {
        match value {
            Value::Array(items) => {
                pairs.extend(items.iter().map(|item| (name.clone(), value_to_text(item))));
            }
            other => pairs.push((name.clone(), value_to_text(other))),
        }
    }
    pairs
}

/// Returns whether `content_type` denotes a JSON body.
///
/// Python parity: `ParserRegistry.parse` consults the lower-cased
/// `content-type`; `JsonResponseParser.can_parse` accepts `application/json`,
/// `text/json`, `text/x-json`, `application/javascript`, and any RFC 6839
/// `+json` suffix. A `charset` parameter is ignored.
fn is_json_content_type(content_type: &str) -> bool {
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    matches!(
        essence.as_str(),
        "application/json" | "text/json" | "text/x-json" | "application/javascript"
    ) || essence.ends_with("+json")
}

/// Returns the canonical reason phrase of `status`, when it is a known status.
fn canonical_reason(status: u16) -> Option<String> {
    reqwest::StatusCode::from_u16(status)
        .ok()
        .and_then(|status| status.canonical_reason())
        .map(str::to_owned)
}

/// Renders the response headers as a JSON object, joining duplicates with `", "`.
fn header_object(headers: &HeaderMap) -> Value {
    let mut joined: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in headers {
        let rendered = String::from_utf8_lossy(value.as_bytes()).into_owned();
        joined
            .entry(name.as_str().to_ascii_lowercase())
            .and_modify(|existing| {
                existing.push_str(", ");
                existing.push_str(&rendered);
            })
            .or_insert(rendered);
    }
    Value::Object(
        joined
            .into_iter()
            .map(|(name, value)| (name, Value::String(value)))
            .collect(),
    )
}

/// Renders `": <excerpt>"` for an error message, or nothing for an empty body.
fn body_excerpt(body: &str) -> String {
    const MAX_CHARS: usize = 200;

    let trimmed = body.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut excerpt: String = trimmed.chars().take(MAX_CHARS).collect();
    if trimmed.chars().count() > MAX_CHARS {
        excerpt.push('…');
    }
    format!(": {excerpt}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use wiremock::matchers::{body_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn bucket(params: &ParamMap, location: ParamLocation) -> Value {
        serde_json::to_value(params.get(&location).cloned().unwrap_or_default()).unwrap()
    }

    fn card(schema: Value) -> ToolCard {
        ToolCard::new("call_api", "calls an api", schema)
    }

    fn tool(url: String, method: HttpMethod, schema: Value) -> RestfulApiTool {
        RestfulApiTool::new(card(schema), RestfulApiConfig::new(url).with_method(method)).unwrap()
    }

    // port of tests/unit_tests/core/foundation/tool/test_restfulapi.py

    #[tokio::test]
    async fn post_routes_parameters_and_returns_the_json_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/users/42"))
            .and(query_param("q", "hello"))
            .and(header("token", "secret"))
            .and(body_json(json!({"name": "bob"})))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(r#"{"ok": true}"#, "application/json"),
            )
            .mount(&server)
            .await;

        let schema = json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "location": "path"},
                "q": {"type": "string", "location": "query"},
                "token": {"type": "string", "location": "header"},
                "name": {"type": "string", "location": "body"},
            },
        });
        let tool = tool(
            format!("{}/users/{{id}}", server.uri()),
            HttpMethod::Post,
            schema,
        );

        let envelope = tool
            .invoke(
                json!({"id": 42, "q": "hello", "token": "secret", "name": "bob"}),
                &ToolContext::new(),
            )
            .await
            .unwrap();

        assert_eq!(envelope["code"], json!(200));
        assert_eq!(envelope["message"], json!("success"));
        assert_eq!(envelope["data"], json!({"ok": true}));
        assert_eq!(
            envelope["url"],
            json!(format!("{}/users/42?q=hello", server.uri()))
        );
        assert_eq!(
            envelope["headers"]["content-type"],
            json!("application/json")
        );

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.method.as_str(), "POST");
        assert_eq!(request.url.path(), "/users/42");
        assert_eq!(
            request
                .headers
                .get("token")
                .map(|value| value.to_str().unwrap()),
            Some("secret")
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({"name": "bob"})
        );
    }

    #[tokio::test]
    async fn get_sends_body_parameters_as_query_pairs_and_returns_text() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "rust"))
            .and(query_param("page", "2"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/plain")
                    .set_body_string("plain text"),
            )
            .mount(&server)
            .await;

        let schema = json!({
            "type": "object",
            "properties": {
                "q": {"type": "string", "location": "query"},
                "page": {"type": "integer", "location": "body"},
            },
        });
        let tool = tool(format!("{}/search", server.uri()), HttpMethod::Get, schema);

        let envelope = tool
            .invoke(json!({"q": "rust", "page": 2}), &ToolContext::new())
            .await
            .unwrap();

        assert_eq!(envelope["code"], json!(200));
        assert_eq!(envelope["message"], json!("success"));
        assert_eq!(envelope["data"], json!("plain text"));

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].body.is_empty(),
            "a GET must not carry a request body"
        );
        assert!(requests[0].url.query().unwrap().contains("page=2"));
    }

    #[tokio::test]
    async fn json_suffix_content_types_are_parsed_as_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/items"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"{"items": [1, 2]}"#,
                "application/hal+json; charset=utf-8",
            ))
            .mount(&server)
            .await;

        let tool = tool(
            format!("{}/items", server.uri()),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        );
        let envelope = tool.invoke(json!({}), &ToolContext::new()).await.unwrap();

        assert_eq!(envelope["data"], json!({"items": [1, 2]}));
    }

    #[tokio::test]
    async fn non_success_statuses_are_tool_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/missing"))
            .respond_with(ResponseTemplate::new(404).set_body_string("no such endpoint"))
            .mount(&server)
            .await;

        let tool = tool(
            format!("{}/missing", server.uri()),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        );
        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Tool);
        assert!(error.to_string().contains("404"), "{error}");
        assert!(error.to_string().contains("no such endpoint"), "{error}");
    }

    #[tokio::test]
    async fn oversized_responses_are_tool_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/large"))
            .respond_with(ResponseTemplate::new(200).set_body_string("0123456789abcdef"))
            .mount(&server)
            .await;

        let config =
            RestfulApiConfig::new(format!("{}/large", server.uri())).with_max_response_bytes(8);
        let tool =
            RestfulApiTool::new(card(json!({"type": "object", "properties": {}})), config).unwrap();

        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Tool);
        assert!(error.to_string().contains("8 byte limit"), "{error}");
    }

    #[test]
    fn new_rejects_url_placeholders_without_path_properties() {
        let error = RestfulApiTool::new(
            card(json!({"type": "object", "properties": {"id": {"type": "string"}}})),
            RestfulApiConfig::new("http://localhost:9/users/{id}"),
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("'id'"), "{error}");
        assert!(error.to_string().contains("location"), "{error}");

        let no_schema = RestfulApiTool::new(
            card(json!({})),
            RestfulApiConfig::new("http://localhost:9/users/{id}"),
        )
        .unwrap_err();
        assert_eq!(no_schema.kind(), ErrorKind::Validation);
        assert!(
            no_schema.to_string().contains("is not defined"),
            "{no_schema}"
        );
    }

    #[tokio::test]
    async fn missing_path_values_are_validation_errors_at_invoke_time() {
        let tool = tool(
            "http://localhost:9/users/{id}".to_owned(),
            HttpMethod::Get,
            json!({"type": "object", "properties": {"id": {"type": "string", "location": "path"}}}),
        );

        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("'id'"), "{error}");
    }

    #[tokio::test]
    async fn new_rejects_relative_or_non_http_urls() {
        let relative = RestfulApiTool::new(
            card(json!({"type": "object", "properties": {}})),
            RestfulApiConfig::new("/api/v1/users"),
        )
        .unwrap_err();
        assert_eq!(relative.kind(), ErrorKind::Validation);

        let file = RestfulApiTool::new(
            card(json!({"type": "object", "properties": {}})),
            RestfulApiConfig::new("file:///tmp/api"),
        )
        .unwrap_err();
        assert_eq!(file.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn stream_is_unsupported() {
        let tool: Arc<dyn Tool> = Arc::new(tool(
            "http://localhost:9/users".to_owned(),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        ));

        let items: Vec<Result<Value>> = tool
            .clone()
            .stream(json!({"id": 1}), ToolContext::new())
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        let error = items[0].as_ref().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[tokio::test]
    async fn cancelled_contexts_short_circuit_the_call() {
        let tool = tool(
            "http://localhost:9/users".to_owned(),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        );
        let ctx = ToolContext::new();
        ctx.cancellation.cancel();

        let error = tool.invoke(json!({}), &ctx).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
    }

    #[test]
    fn config_defaults_are_merged_and_blank_inputs_keep_them() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "location": "query"},
                "lang": {"type": "string", "location": "query"},
            },
        });
        let mut queries = BTreeMap::new();
        queries.insert("lang".to_owned(), json!("en"));
        queries.insert("name".to_owned(), json!("default name"));
        let mut headers = BTreeMap::new();
        headers.insert("X-API-Key".to_owned(), json!("key"));
        let mut paths = BTreeMap::new();
        paths.insert("version".to_owned(), json!("v1"));

        let mapper = ApiParamMapper::with_defaults(schema, queries, headers, paths);

        let overridden = mapper.map(&json!({"name": "given", "lang": "zh"})).unwrap();
        assert_eq!(bucket(&overridden, ParamLocation::Query)["name"], "given");
        assert_eq!(bucket(&overridden, ParamLocation::Query)["lang"], "zh");

        let blank = mapper.map(&json!({"name": "", "lang": null})).unwrap();
        assert_eq!(
            bucket(&blank, ParamLocation::Query)["name"],
            json!("default name")
        );
        assert_eq!(bucket(&blank, ParamLocation::Query)["lang"], json!("en"));

        assert_eq!(bucket(&blank, ParamLocation::Header)["X-API-Key"], "key");
        assert_eq!(bucket(&blank, ParamLocation::Path)["version"], "v1");
        assert_eq!(bucket(&blank, ParamLocation::Body), json!({}));
        assert_eq!(
            mapper.defaults_for(ParamLocation::Query).map(BTreeMap::len),
            Some(2)
        );
        assert_eq!(
            mapper.schema(),
            &json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "location": "query"},
                    "lang": {"type": "string", "location": "query"},
                },
            })
        );
    }

    // port of tests/unit_tests/core/foundation/tool/test_api_param_mapper.py

    #[test]
    fn mapper_routes_every_declared_location() {
        let schema = json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "location": "path"},
                "name": {"type": "string", "location": "query"},
                "data": {"type": "object", "location": "body"},
                "auth_token": {"type": "string", "location": "header"},
                "unlocated": {"type": "string"},
            },
        });
        let mapper = ApiParamMapper::new(schema);

        let params = mapper
            .map(&json!({
                "id": 123,
                "name": "John",
                "data": {"key": "value"},
                "auth_token": "abc123",
                "unlocated": "dropped?",
                "undeclared": "dropped",
            }))
            .unwrap();

        assert_eq!(bucket(&params, ParamLocation::Path), json!({"id": 123}));
        assert_eq!(
            bucket(&params, ParamLocation::Query),
            json!({"name": "John"})
        );
        assert_eq!(
            bucket(&params, ParamLocation::Body),
            json!({"data": {"key": "value"}, "unlocated": "dropped?"})
        );
        assert_eq!(
            bucket(&params, ParamLocation::Header),
            json!({"auth_token": "abc123"})
        );
        assert!(
            !bucket(&params, ParamLocation::Body)
                .as_object()
                .unwrap()
                .contains_key("undeclared"),
            "undeclared input keys are dropped, as in Python"
        );
    }

    #[test]
    fn mapper_forwards_everything_to_the_body_without_properties() {
        let mapper = ApiParamMapper::new(json!({"type": "object"}));
        let params = mapper.map(&json!({"a": 1, "b": "two"})).unwrap();

        assert_eq!(
            bucket(&params, ParamLocation::Body),
            json!({"a": 1, "b": "two"})
        );
        assert_eq!(bucket(&params, ParamLocation::Query), json!({}));
    }

    #[test]
    fn mapper_accepts_upper_case_locations_and_rejects_form() {
        let mapper = ApiParamMapper::new(json!({
            "type": "object",
            "properties": {"q": {"type": "string", "location": "QUERY"}},
        }));
        let params = mapper.map(&json!({"q": "x"})).unwrap();
        assert_eq!(bucket(&params, ParamLocation::Query), json!({"q": "x"}));

        let form = ApiParamMapper::new(json!({
            "type": "object",
            "properties": {"file": {"type": "string", "location": "form"}},
        }));
        let error = form.map(&json!({"file": "a"})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("form"), "{error}");
        assert!(error.to_string().contains("'file'"), "{error}");
    }

    #[test]
    fn mapper_renders_urls_and_reports_missing_path_values() {
        let mapper = ApiParamMapper::new(json!({}));
        let mut params = empty_param_map();
        let path_bucket = params.entry(ParamLocation::Path).or_default();
        path_bucket.insert("id".to_owned(), json!(42));
        path_bucket.insert("flag".to_owned(), json!(true));
        path_bucket.insert("name".to_owned(), json!("a b"));

        assert_eq!(
            mapper
                .render_url("https://api.example.com/users/{id}/{flag}/{name}", &params)
                .unwrap(),
            "https://api.example.com/users/42/true/a b"
        );
        assert_eq!(
            mapper
                .render_url("https://api.example.com/users", &params)
                .unwrap(),
            "https://api.example.com/users"
        );

        let error = mapper
            .render_url("https://api.example.com/users/{id}/{other}", &params)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("'other'"), "{error}");
    }

    #[test]
    fn http_method_serde_and_schema_use_the_wire_form() {
        assert_eq!(
            serde_json::from_str::<HttpMethod>("\"get\"").unwrap(),
            HttpMethod::Get
        );
        assert_eq!(
            serde_json::from_str::<HttpMethod>("\"GET\"").unwrap(),
            HttpMethod::Get
        );
        assert_eq!(
            serde_json::from_str::<HttpMethod>("\"Options\"").unwrap(),
            HttpMethod::Options
        );
        assert_eq!(
            serde_json::to_string(&HttpMethod::Delete).unwrap(),
            "\"DELETE\""
        );
        assert_eq!(HttpMethod::default(), HttpMethod::Post);
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert!(HttpMethod::Get.is_bodyless());
        assert!(!HttpMethod::Patch.is_bodyless());
        assert!(serde_json::from_str::<HttpMethod>("\"TRACE\"").is_err());

        let schema = serde_json::to_value(
            schemars::SchemaGenerator::default().into_root_schema_for::<HttpMethod>(),
        )
        .unwrap();
        // An enum of unit variants is rendered as `oneOf` of `const` values.
        let variants: Vec<Value> = schema
            .get("oneOf")
            .and_then(Value::as_array)
            .map(|variants| {
                variants
                    .iter()
                    .filter_map(|variant| variant.get("const").cloned())
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(variants.len(), 7, "{schema}");
        assert!(variants.contains(&json!("GET")), "{schema}");
        assert!(variants.contains(&json!("OPTIONS")), "{schema}");
    }

    #[test]
    fn param_location_serde_is_lower_case() {
        assert_eq!(
            serde_json::to_string(&ParamLocation::Header).unwrap(),
            "\"header\""
        );
        assert_eq!(
            serde_json::from_str::<ParamLocation>("\"query\"").unwrap(),
            ParamLocation::Query
        );
        assert_eq!(ParamLocation::default(), ParamLocation::Body);
        assert_eq!(ParamLocation::as_str(ParamLocation::Path), "path");
        assert!(ParamLocation::from_str("form").is_err());
    }

    #[test]
    fn config_round_trips_with_seconds_timeout() {
        let config = RestfulApiConfig::new("https://api.example.com/users")
            .with_method(HttpMethod::Get)
            .with_timeout(Duration::from_secs(30))
            .with_max_response_bytes(1024)
            .with_query("lang", json!("en"))
            .with_header("X-Api-Key", json!("key"))
            .with_path("version", json!("v1"));

        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["method"], json!("GET"));
        assert_eq!(value["timeout"], json!(30));
        assert_eq!(value["max_response_bytes"], json!(1024));
        assert_eq!(value["queries"]["lang"], json!("en"));

        let restored: RestfulApiConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);

        let fractional: RestfulApiConfig =
            serde_json::from_value(json!({"url": "https://api.example.com", "timeout": 1.5}))
                .unwrap();
        assert_eq!(fractional.timeout, Duration::from_millis(1500));

        let partial: RestfulApiConfig =
            serde_json::from_value(json!({"url": "https://api.example.com"})).unwrap();
        assert_eq!(partial.method, HttpMethod::Post);
        assert_eq!(partial.timeout, DEFAULT_TIMEOUT);
        assert_eq!(partial.max_response_bytes, DEFAULT_MAX_RESPONSE_BYTES);

        let negative: std::result::Result<RestfulApiConfig, _> =
            serde_json::from_value(json!({"url": "https://api.example.com", "timeout": -1}));
        assert!(negative.is_err(), "a negative timeout must be rejected");
    }

    #[tokio::test]
    async fn query_arrays_expand_to_one_parameter_per_item() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/batch"))
            .and(query_param("ids", "1"))
            .and(query_param("ids", "2"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .expect(1)
            .mount(&server)
            .await;

        let schema = json!({
            "type": "object",
            "properties": {"ids": {"type": "array", "location": "query"}},
        });
        let tool = tool(format!("{}/batch", server.uri()), HttpMethod::Get, schema);

        let envelope = tool
            .invoke(json!({"ids": [1, 2]}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(envelope["data"], json!("ok"));

        let requests = server.received_requests().await.unwrap();
        let query = requests[0].url.query().unwrap();
        assert!(
            query.contains("ids=1") && query.contains("ids=2"),
            "{query}"
        );
    }
}
