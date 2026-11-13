//! Built-in `http_request` workflow component.
//!
//! Ports `openjiuwen/core/workflow/components/tool/http/http_request_component.py`
//! (`HttpComponentConfig`, `HTTPRequestExecutable`, `HTTPRequestComponent`). The component
//! turns its configuration into an [`HttpRequest`], executes it through the shared
//! [`openjiuwen_common::http::HttpClient`], and projects the response into Python's
//! `statusCode` / `headers` / `body` / `url` / `ok` wire object.
//!
//! # Behaviour
//!
//! - **Inputs** are merged with the configuration the way Python's `process_inputs`
//!   does: `inputs["method"]`, `inputs["headers"]`, and `inputs["query_parameters"]`
//!   override/extend the configured values, and a non-null `inputs["body"]` becomes a
//!   JSON body when no `body` config exists.
//! - **Templating**: `{{key}}` placeholders are replaced by the top-level input value
//!   `key` (Python `_resolve_placeholders`). As a Rust extension, embedded `${path}`
//!   references are resolved against the component inputs with
//!   [`crate::reference::resolve_path`] *before* `{{key}}` substitution; an unresolvable
//!   reference stays literal. Non-string values render as compact JSON text.
//!   Applied to `url`, `method`, every header value, string query-parameter values,
//!   `body.json_data` (when it is a string), and `body.text_data`.
//!   Authentication secrets (`token`, `api_key`, `username`, `password`) are passed
//!   through untouched, exactly like Python.
//! - **Retries** live in [`HttpClientConfig`]: `retry_config.enabled` maps to
//!   `max_retries` / `retry_statuses` / `retry_initial_backoff`, plus a backoff cap.
//! - **HTTP failures are data**: a 4xx/5xx response is returned as an ordinary output
//!   with `ok = false`. Only transport, timeout, cancellation, size, and client-build
//!   failures become [`component_tool_execution_error`] (`[102000]`).
//!
//! # Deviations from Python
//!
//! - **Abilities.** Python declares `invoke`, `stream`, `collect`, and `transform`. The
//!   engine's ability priority is `Transform > Collect > Stream > Invoke`, so a plain
//!   Start → HTTP → End node would run `transform` and buffer frames; this component
//!   therefore advertises only [`ComponentAbility::Invoke`] while still implementing
//!   `stream`/`collect`/`transform` (each yields the single `invoke` result) for callers
//!   that select them explicitly.
//! - **String rendering.** Python uses `str(value)` for non-string placeholders; Rust
//!   uses compact JSON text (`{"a":1}` instead of `{'a': 1}`).
//! - **Embedded `${path}` references** are a Rust extension (Python only knows
//!   `{{key}}`), as is templating dictionary-shaped header values and the configured
//!   `method` (Python only resolves the string header form and never templated the
//!   method).
//! - **Retry backoff.** Retries run inside `HttpClient` with jittered backoff.
//!   `fixed` caps the backoff at `retry_delay` (constant delay) and `linear` is
//!   approximated by `fixed`; `exponential` caps at `retry_delay * 2^max_retries`.
//!   Transport errors are retried too while enabled — Python does the same.
//! - **Multipart bodies** are not supported: the shared transport has no multipart
//!   request body, so `content_type = multipart_form` returns [`Error::Unsupported`].
//! - **Binary responses** are base64-encoded (`BASE64_STANDARD`) because JSON has no
//!   bytes type; Python returns `bytes`. `binary_data` requests are base64-decoded.
//! - **JSON string bodies** are parsed with `serde_json` instead of Python's
//!   `ast.literal_eval`, so single-quoted Python dict literals are rejected; a parse
//!   failure sends the resolved text as a text body (Python substitutes `{}`).
//! - **Bodies** are sent for every method (Python sends them only for POST/PUT/PATCH),
//!   and a body `content-type` is only added when the header is still absent (Python
//!   overwrites `Content-Type`).
//! - **Digest / AWS authentication** are unimplemented ([`Error::Unsupported`]); Python
//!   never implemented them either. An API-key `in_location = body` is ignored, as in
//!   Python.
//! - `throw_on_http_error` is parsed but never applied (Python never raises there
//!   either), `rate_limit_config` is parsed and preserved but not enforced, and
//!   `advanced_options.use_stream` is accepted but ignored (responses are buffered).
//! - `advanced_options.max_body_length` is accepted but not enforced; the enforced cap is
//!   `max_response_byte_size` (both Python fields are independent, and only the latter is
//!   used by Python's request loop).
//! - Python's `response_handling.max_redirects` is not modelled; redirect handling
//!   follows `advanced_options.follow_redirect`.
//! - `timeout` is clamped into `1.0..=300.0` seconds at use time (Python validates the
//!   same range); a non-finite timeout falls back to 60 seconds instead of panicking.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse, Method};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::error::{component_tool_execution_error, component_tool_input_param_error};
use crate::reference;

/// The `Content-Type` used for JSON request bodies.
const APPLICATION_JSON: &str = "application/json";
/// The `Content-Type` used for text request bodies.
const TEXT_PLAIN: &str = "text/plain";
/// The `Content-Type` used for form request bodies.
const FORM_URLENCODED: &str = "application/x-www-form-urlencoded";
/// The `Content-Type` used for binary request bodies.
const APPLICATION_OCTET_STREAM: &str = "application/octet-stream";
/// The timeout applied when the configured value is not a finite number.
const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
/// The Python range for `request_params.timeout`, in seconds.
const MIN_TIMEOUT_SECONDS: f64 = 1.0;
/// The Python range for `request_params.timeout`, in seconds.
const MAX_TIMEOUT_SECONDS: f64 = 300.0;
/// The default `request_params.max_response_byte_size` (10 MiB).
const DEFAULT_MAX_RESPONSE_BYTE_SIZE: u64 = 10 * 1024 * 1024;
/// The default `advanced_options.max_body_length` (1 MiB).
const DEFAULT_MAX_BODY_LENGTH: u64 = 1024 * 1024;

/// The HTTP body encoding (`HttpRequestBodyConfig.content_type`).
///
/// The wire values are exactly Python's (`"json"`, `"form"`, `"multipart_form"`,
/// `"binary"`, `"text"`, `"auto"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpContentType {
    /// JSON body from `json_data`; the Python default.
    #[default]
    Json,
    /// URL-encoded form body from `form_data`.
    Form,
    /// Multipart form body from `multipart_form` (unsupported by the shared transport).
    MultipartForm,
    /// Raw bytes from base64-encoded `binary_data`.
    Binary,
    /// Plain-text body from `text_data`.
    Text,
    /// Pick a body kind from whichever field is present.
    Auto,
}

/// The authentication scheme (`HttpAuthConfig.type`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpAuthType {
    /// No authentication; the Python default.
    #[default]
    None,
    /// HTTP basic authentication with `username`/`password`.
    Basic,
    /// `Authorization: Bearer <token>`.
    Bearer,
    /// An API key placed in a header or query parameter.
    ApiKey,
    /// HTTP digest authentication (unimplemented, like Python).
    Digest,
    /// AWS signature authentication (unimplemented, like Python).
    Aws,
}

/// Where an API key is placed (`HttpAuthConfig.in_location`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpAuthIn {
    /// Send the key as a request header; the Python default.
    #[default]
    Header,
    /// Append the key as a query parameter.
    Query,
    /// Put the key in the body (ignored, as in Python).
    Body,
}

/// How the response body is parsed (`HttpResponseHandlingConfig.response_format`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpResponseFormat {
    /// Detect the format from the `content-type` response header; the Python default.
    #[default]
    Autodetect,
    /// Parse the body as JSON, falling back to text.
    Json,
    /// Decode the body as lossy UTF-8 text.
    Text,
    /// Base64-encode the raw bytes.
    Binary,
    /// Buffered bytes; behaves like [`Self::Binary`] in Rust.
    Buffer,
}

/// Which responses are returned (`HttpResponseHandlingConfig.response_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpResponseMode {
    /// Always return the full response object; the Python default.
    #[default]
    #[serde(rename = "full")]
    Full,
    /// Return `{}` unless the status counts as a success.
    #[serde(rename = "on-success")]
    OnSuccess,
    /// Return `{}` when the status counts as a success.
    #[serde(rename = "on-error")]
    OnError,
}

/// The retry backoff strategy (`HttpRetryConfig.backoff_type`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpBackoffType {
    /// Constant delay; the Python options are `fixed`, `exponential`, `linear`.
    Fixed,
    /// Exponentially growing delay; the Python default.
    #[default]
    Exponential,
    /// Linearly growing delay (approximated by [`Self::Fixed`] in Rust).
    Linear,
}

/// The rate-limit window unit (`HttpRateLimitConfig.unit`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpRateLimitUnit {
    /// Requests per second; the Python default.
    #[default]
    Second,
    /// Requests per minute.
    Minute,
    /// Requests per hour.
    Hour,
}

/// Request headers, accepting either a JSON object or a template string.
///
/// Python declares `headers: Union[Dict[str, str], str]`; the string form holds a JSON
/// object (Python evaluates it with `ast.literal_eval`), which is parsed after templating.
/// An unparseable template string yields no headers, matching Python's `{}` fallback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HttpHeadersConfig {
    /// A plain name → value map.
    Map(BTreeMap<String, String>),
    /// A template string holding a JSON object; placeholders are resolved first.
    Template(String),
}

impl Default for HttpHeadersConfig {
    fn default() -> Self {
        Self::Map(BTreeMap::new())
    }
}

impl HttpHeadersConfig {
    /// Resolves the headers against `inputs`.
    fn resolved(&self, inputs: &Value) -> BTreeMap<String, String> {
        match self {
            Self::Map(headers) => headers
                .iter()
                .map(|(name, value)| (name.clone(), resolve_template(value, inputs)))
                .collect(),
            Self::Template(template) => {
                let resolved = resolve_template(template, inputs);
                match serde_json::from_str::<Value>(&resolved) {
                    Ok(Value::Object(map)) => map
                        .into_iter()
                        .map(|(name, value)| (name, stringify(&value)))
                        .collect(),
                    _ => BTreeMap::new(),
                }
            }
        }
    }
}

/// The request body configuration (`HttpRequestBodyConfig`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpRequestBodyConfig {
    /// How the body is encoded; Python's default is [`HttpContentType::Json`].
    pub content_type: HttpContentType,
    /// JSON body: either a JSON value or a string holding JSON that is templated first.
    pub json_data: Option<Value>,
    /// URL-encoded form fields (`application/x-www-form-urlencoded`).
    pub form_data: Option<BTreeMap<String, Value>>,
    /// Multipart form fields; the shared transport cannot send them.
    pub multipart_form: Option<BTreeMap<String, Value>>,
    /// Base64-encoded request bytes (`application/octet-stream`).
    pub binary_data: Option<String>,
    /// Plain-text body (`text/plain`).
    pub text_data: Option<String>,
}

/// The authentication configuration (`HttpAuthConfig`).
///
/// The wire key of [`Self::auth_type`] is `type`, mirroring Python; secrets are passed
/// through without templating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpAuthConfig {
    /// The authentication scheme; Python's default is [`HttpAuthType::None`].
    #[serde(rename = "type")]
    pub auth_type: HttpAuthType,
    /// Bearer token.
    pub token: Option<String>,
    /// API key value.
    pub api_key: Option<String>,
    /// Name of the API-key header or query parameter (Python default `Authorization`).
    pub name: String,
    /// Where an API key is placed (Python default [`HttpAuthIn::Header`]).
    pub in_location: HttpAuthIn,
    /// Basic-authentication user name.
    pub username: Option<String>,
    /// Basic-authentication password.
    pub password: Option<String>,
}

impl Default for HttpAuthConfig {
    fn default() -> Self {
        Self {
            auth_type: HttpAuthType::None,
            token: None,
            api_key: None,
            name: "Authorization".to_owned(),
            in_location: HttpAuthIn::Header,
            username: None,
            password: None,
        }
    }
}

/// How responses are interpreted and returned (`HttpResponseHandlingConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpResponseHandlingConfig {
    /// How the body is parsed; Python's default is [`HttpResponseFormat::Autodetect`].
    pub response_format: HttpResponseFormat,
    /// Statuses that count as a success (Python default `[200, 201, 202, 204]`).
    pub response_code_success_codes: Vec<u16>,
    /// Statuses that count as a failure (Python default: empty).
    pub response_code_failure_codes: Vec<u16>,
    /// Which responses are returned; Python's default is [`HttpResponseMode::Full`].
    pub response_mode: HttpResponseMode,
    /// Object property extracted from the parsed body, when present.
    pub response_data_property: Option<String>,
    /// Parsed for parity; Python never raises on an HTTP error either.
    pub throw_on_http_error: bool,
}

impl Default for HttpResponseHandlingConfig {
    fn default() -> Self {
        Self {
            response_format: HttpResponseFormat::Autodetect,
            response_code_success_codes: vec![200, 201, 202, 204],
            response_code_failure_codes: Vec::new(),
            response_mode: HttpResponseMode::Full,
            response_data_property: None,
            throw_on_http_error: true,
        }
    }
}

/// Transport tuning (`HttpAdvancedOptionsConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpAdvancedOptionsConfig {
    /// Follow redirects; the Python default is `true`.
    pub follow_redirect: bool,
    /// Disable TLS certificate verification.
    pub ignore_ssl_issues: bool,
    /// Explicit proxy URL.
    pub proxy: Option<String>,
    /// Accepted but not enforced; `max_response_byte_size` is the enforced cap.
    pub max_body_length: u64,
    /// Accepted but ignored; responses are always buffered.
    pub use_stream: bool,
}

impl Default for HttpAdvancedOptionsConfig {
    fn default() -> Self {
        Self {
            follow_redirect: true,
            ignore_ssl_issues: false,
            proxy: None,
            max_body_length: DEFAULT_MAX_BODY_LENGTH,
            use_stream: false,
        }
    }
}

/// Retry configuration (`HttpRetryConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpRetryConfig {
    /// Whether retries are attempted at all.
    pub enabled: bool,
    /// Additional attempts after the first one (Python default `3`).
    pub max_retries: u32,
    /// Statuses that trigger a retry (Python default `[429, 500, 502, 503, 504]`).
    pub retry_on_status_codes: Vec<u16>,
    /// Base retry delay in milliseconds (Python default `1000`).
    pub retry_delay: u64,
    /// Backoff strategy; Python's default is [`HttpBackoffType::Exponential`].
    pub backoff_type: HttpBackoffType,
}

impl Default for HttpRetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_retries: 3,
            retry_on_status_codes: vec![429, 500, 502, 503, 504],
            retry_delay: 1000,
            backoff_type: HttpBackoffType::Exponential,
        }
    }
}

/// Rate-limit configuration (`HttpRateLimitConfig`).
///
/// Parsed and preserved for parity but never enforced: no process-global limiter is
/// ported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpRateLimitConfig {
    /// Whether rate limiting is requested.
    pub enabled: bool,
    /// Requests allowed per `unit` (Python default `1`).
    pub requests_per_unit: u32,
    /// The rate-limit window; Python's default is [`HttpRateLimitUnit::Second`].
    pub unit: HttpRateLimitUnit,
}

impl Default for HttpRateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_unit: 1,
            unit: HttpRateLimitUnit::Second,
        }
    }
}

/// The request parameters (`HttpRequestParamConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpRequestParamConfig {
    /// Target URL; an empty value is rejected at invoke time.
    pub url: String,
    /// HTTP method (Python default `GET`), templated and upper-cased.
    pub method: String,
    /// Request headers, as a map or a template string.
    pub headers: HttpHeadersConfig,
    /// Query parameters; values are stringified (strings are templated first).
    pub query_parameters: BTreeMap<String, Value>,
    /// Request body configuration; `None` allows a JSON `inputs["body"]` instead.
    pub body: Option<HttpRequestBodyConfig>,
    /// Authentication configuration.
    pub authentication: Option<HttpAuthConfig>,
    /// Response interpretation.
    pub response_handling: HttpResponseHandlingConfig,
    /// Transport tuning.
    pub advanced_options: HttpAdvancedOptionsConfig,
    /// Retry configuration.
    pub retry_config: HttpRetryConfig,
    /// Parsed and preserved, never enforced.
    pub rate_limit_config: HttpRateLimitConfig,
    /// Request timeout in seconds, clamped into `1.0..=300.0` at use time.
    pub timeout: f64,
    /// Maximum accepted response size in bytes (Python default 10 MiB).
    pub max_response_byte_size: u64,
}

impl Default for HttpRequestParamConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            method: "GET".to_owned(),
            headers: HttpHeadersConfig::default(),
            query_parameters: BTreeMap::new(),
            body: None,
            authentication: None,
            response_handling: HttpResponseHandlingConfig::default(),
            advanced_options: HttpAdvancedOptionsConfig::default(),
            retry_config: HttpRetryConfig::default(),
            rate_limit_config: HttpRateLimitConfig::default(),
            timeout: DEFAULT_TIMEOUT_SECONDS,
            max_response_byte_size: DEFAULT_MAX_RESPONSE_BYTE_SIZE,
        }
    }
}

/// The component configuration (`HttpComponentConfig`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HttpComponentConfig {
    /// The request to perform.
    pub request_params: HttpRequestParamConfig,
    /// Arbitrary component metadata, ignored by the request pipeline.
    pub metadata: Option<Value>,
}

/// The built-in `http_request` component.
///
/// `invoke` resolves the request from the component inputs and executes it; failures are
/// reported as `[102000]`/`[102001]` errors and non-2xx responses are ordinary outputs.
#[derive(Debug, Clone)]
pub struct HttpRequestComponent {
    config: HttpComponentConfig,
}

impl HttpRequestComponent {
    /// Creates a component from `config`.
    #[must_use]
    pub fn new(config: HttpComponentConfig) -> Self {
        Self { config }
    }

    /// The component configuration (Python's `config` property).
    #[must_use]
    pub fn config(&self) -> &HttpComponentConfig {
        &self.config
    }
}

#[async_trait::async_trait]
impl Component for HttpRequestComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("http_request", "http_request").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        // Only `invoke` is advertised; see the module docs for the rationale.
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "http_request"
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let params = &self.config.request_params;
        let request = build_request(params, &inputs, ctx.cancellation().clone())?;
        let client = HttpClient::new(client_config(params))
            .map_err(|error| request_failed(&error.message()))?;
        let response = client
            .execute_cancellable(request, ctx.cancellation().clone())
            .await
            .map_err(|error| request_failed(&error.message()))?;
        Ok(handle_response(params, &response))
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        futures::stream::once(async move { self.invoke(inputs, ctx.as_ref()).await }).boxed()
    }

    async fn collect(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        self.invoke(inputs, ctx).await
    }

    fn transform(
        self: Arc<Self>,
        inputs: Value,
        ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        futures::stream::once(async move { self.invoke(inputs, ctx.as_ref()).await }).boxed()
    }
}

/// Renders a transport failure as Python's `[102000]` component-tool error.
fn request_failed(message: &str) -> Error {
    component_tool_execution_error(&format!("HTTP request failed: {message}"))
}

/// The body kinds the shared transport can send.
#[derive(Debug)]
enum RequestBody {
    /// A JSON value, serialized by the transport.
    Json(Value),
    /// A pre-rendered text body.
    Text(String),
    /// Raw request bytes.
    Bytes(Bytes),
}

/// Which body field an `auto` content type selected.
enum BodyField {
    /// `json_data`.
    Json,
    /// `text_data`.
    Text,
    /// `form_data`.
    Form,
}

/// Resolves `${path}` references embedded in `template`; unresolvable ones stay literal.
fn resolve_embedded_refs(template: &str, inputs: &Value) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("${") {
        let (before, candidate) = rest.split_at(start);
        rendered.push_str(before);
        let Some(end) = candidate.find('}') else {
            rendered.push_str(candidate);
            return rendered;
        };
        let path = &candidate[2..end];
        match reference::resolve_path(inputs, path) {
            Some(value) if !path.is_empty() => rendered.push_str(&stringify(&value)),
            _ => rendered.push_str(&candidate[..=end]),
        }
        rest = &candidate[end + 1..];
    }
    rendered.push_str(rest);
    rendered
}

/// Replaces `{{key}}` placeholders with the top-level input values (Python parity).
fn resolve_placeholders(template: &str, inputs: &Value) -> String {
    let Value::Object(map) = inputs else {
        return template.to_owned();
    };
    let mut rendered = template.to_owned();
    for (key, value) in map {
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), &stringify(value));
    }
    rendered
}

/// Resolves `${path}` references first, then `{{key}}` placeholders.
fn resolve_template(template: &str, inputs: &Value) -> String {
    resolve_placeholders(&resolve_embedded_refs(template, inputs), inputs)
}

/// Renders a value as text: strings verbatim, everything else compact JSON.
fn stringify(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        // Serializing a `Value` cannot fail; the fallback keeps the path panic-free.
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// The timeout applied to the request and the client, clamped into Python's range.
fn request_timeout(params: &HttpRequestParamConfig) -> Duration {
    if params.timeout.is_finite() {
        Duration::from_secs_f64(
            params
                .timeout
                .clamp(MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS),
        )
    } else {
        Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS)
    }
}

/// Maps the component configuration onto the shared HTTP client configuration.
///
/// Retries are driven entirely by [`HttpClientConfig`]; see the module docs for the
/// backoff mapping.
fn client_config(params: &HttpRequestParamConfig) -> HttpClientConfig {
    let retry = &params.retry_config;
    let base = HttpClientConfig {
        timeout: request_timeout(params),
        max_response_bytes: Some(params.max_response_byte_size),
        verify_ssl: !params.advanced_options.ignore_ssl_issues,
        follow_redirects: params.advanced_options.follow_redirect,
        proxy: params.advanced_options.proxy.clone(),
        ..HttpClientConfig::default()
    };
    if !retry.enabled {
        return base;
    }
    HttpClientConfig {
        max_retries: retry.max_retries,
        retry_statuses: retry.retry_on_status_codes.clone(),
        retry_initial_backoff: Duration::from_millis(retry.retry_delay),
        retry_max_backoff: match retry.backoff_type {
            // A cap equal to the base makes every retry wait exactly `retry_delay`.
            HttpBackoffType::Fixed | HttpBackoffType::Linear => {
                Duration::from_millis(retry.retry_delay)
            }
            HttpBackoffType::Exponential => {
                let factor = 1_u64.checked_shl(retry.max_retries).unwrap_or(u64::MAX);
                let cap = retry.retry_delay.saturating_mul(factor);
                Duration::from_millis(cap.max(retry.retry_delay))
            }
        },
        ..base
    }
}
/// Builds the request, merging the component inputs with the configuration.
fn build_request(
    params: &HttpRequestParamConfig,
    inputs: &Value,
    cancellation: CancellationToken,
) -> Result<HttpRequest> {
    let url = resolve_template(&params.url, inputs);
    if url.trim().is_empty() {
        return Err(component_tool_input_param_error("url is required"));
    }

    let method_text = request_method(params, inputs);
    let method = Method::from_bytes(method_text.as_bytes()).map_err(|_| {
        component_tool_input_param_error(&format!("invalid HTTP method '{method_text}'"))
    })?;

    let mut headers = params.headers.resolved(inputs);
    merge_input_headers(&mut headers, inputs)?;
    let mut query = resolved_query_parameters(params, inputs)?;
    apply_authentication(params.authentication.as_ref(), &mut headers, &mut query)?;

    let (body, content_type) = prepare_body(params, inputs)?;
    if let Some(content_type) = content_type {
        set_header_if_absent(&mut headers, "content-type", content_type);
    }

    let request = HttpRequest::new(method, url)
        .with_headers(headers)
        .queries(query)
        .timeout(request_timeout(params))
        .cancel(cancellation);

    match body {
        Some(RequestBody::Json(value)) => request.json(&value).map_err(|error| {
            component_tool_input_param_error(&format!("request body is invalid, {error}"))
        }),
        Some(RequestBody::Text(text)) => Ok(request.text(text)),
        Some(RequestBody::Bytes(bytes)) => Ok(request.bytes(bytes)),
        None => Ok(request),
    }
}

/// The request method: `inputs["method"]` wins, is templated, and is upper-cased.
fn request_method(params: &HttpRequestParamConfig, inputs: &Value) -> String {
    let raw = match inputs.get("method") {
        Some(Value::String(text)) => text.clone(),
        Some(value) if !value.is_null() => stringify(value),
        _ => params.method.clone(),
    };
    resolve_template(&raw, inputs).to_uppercase()
}

/// Appends the `inputs["headers"]` overlay (object or template string).
fn merge_input_headers(headers: &mut BTreeMap<String, String>, inputs: &Value) -> Result<()> {
    match inputs.get("headers") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(map)) => {
            for (name, value) in map {
                let _ = headers.insert(name.clone(), resolve_template(&stringify(value), inputs));
            }
            Ok(())
        }
        Some(Value::String(template)) => {
            let overlay = HttpHeadersConfig::Template(template.clone()).resolved(inputs);
            headers.extend(overlay);
            Ok(())
        }
        Some(other) => Err(component_tool_input_param_error(&format!(
            "headers must be an object or a template string, got {other}"
        ))),
    }
}

/// Resolves the configured query parameters and the `inputs["query_parameters"]` overlay.
fn resolved_query_parameters(
    params: &HttpRequestParamConfig,
    inputs: &Value,
) -> Result<Vec<(String, String)>> {
    let mut query: BTreeMap<String, String> = params
        .query_parameters
        .iter()
        .map(|(key, value)| (key.clone(), query_value(value, inputs)))
        .collect();
    match inputs.get("query_parameters") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (key, value) in map {
                let _ = query.insert(key.clone(), query_value(value, inputs));
            }
        }
        Some(other) => {
            return Err(component_tool_input_param_error(&format!(
                "query_parameters must be an object, got {other}"
            )));
        }
    }
    Ok(query.into_iter().collect())
}

/// Renders one query-parameter value: strings are templated, everything else is JSON.
fn query_value(value: &Value, inputs: &Value) -> String {
    match value {
        Value::String(text) => resolve_template(text, inputs),
        other => stringify(other),
    }
}

/// Adds a header unless a case-insensitive equivalent is already present.
fn set_header_if_absent(headers: &mut BTreeMap<String, String>, name: &str, value: &str) {
    if !headers
        .keys()
        .any(|existing| existing.eq_ignore_ascii_case(name))
    {
        let _ = headers.insert(name.to_owned(), value.to_owned());
    }
}

/// Applies the authentication configuration to the headers and query parameters.
fn apply_authentication(
    authentication: Option<&HttpAuthConfig>,
    headers: &mut BTreeMap<String, String>,
    query: &mut Vec<(String, String)>,
) -> Result<()> {
    let Some(auth) = authentication else {
        return Ok(());
    };
    match auth.auth_type {
        HttpAuthType::None => {}
        HttpAuthType::Basic => {
            // Python only applies basic auth when both values are truthy.
            if let (Some(username), Some(password)) = (
                auth.username.as_deref().filter(|value| !value.is_empty()),
                auth.password.as_deref().filter(|value| !value.is_empty()),
            ) {
                let encoded = BASE64_STANDARD.encode(format!("{username}:{password}"));
                let _ = headers.insert("Authorization".to_owned(), format!("Basic {encoded}"));
            }
        }
        HttpAuthType::Bearer => {
            if let Some(token) = auth.token.as_deref().filter(|value| !value.is_empty()) {
                let _ = headers.insert("Authorization".to_owned(), format!("Bearer {token}"));
            }
        }
        HttpAuthType::ApiKey => {
            if let Some(api_key) = auth.api_key.as_deref().filter(|value| !value.is_empty()) {
                match auth.in_location {
                    HttpAuthIn::Header => {
                        let _ = headers.insert(auth.name.clone(), api_key.to_owned());
                    }
                    HttpAuthIn::Query => query.push((auth.name.clone(), api_key.to_owned())),
                    // Python ignores the body location as well.
                    HttpAuthIn::Body => {}
                }
            }
        }
        HttpAuthType::Digest => {
            return Err(Error::unsupported(
                "digest HTTP authentication is not supported by the shared HTTP transport",
            ));
        }
        HttpAuthType::Aws => {
            return Err(Error::unsupported(
                "aws HTTP authentication is not supported by the shared HTTP transport",
            ));
        }
    }
    Ok(())
}

/// Selects the body fields an `auto` content type may use.
fn auto_body_field(body: &HttpRequestBodyConfig) -> Option<BodyField> {
    if body.json_data.is_some() {
        Some(BodyField::Json)
    } else if body.text_data.is_some() {
        Some(BodyField::Text)
    } else if body.form_data.is_some() {
        Some(BodyField::Form)
    } else {
        None
    }
}

/// Builds the request body and the `Content-Type` it implies.
fn prepare_body(
    params: &HttpRequestParamConfig,
    inputs: &Value,
) -> Result<(Option<RequestBody>, Option<&'static str>)> {
    let Some(body) = params.body.as_ref() else {
        // Python sends no body here; sending `inputs["body"]` as JSON is a Rust extension.
        return Ok(match inputs.get("body") {
            Some(value) if !value.is_null() => (
                Some(RequestBody::Json(value.clone())),
                Some(APPLICATION_JSON),
            ),
            _ => (None, None),
        });
    };

    let field = match body.content_type {
        HttpContentType::Auto => auto_body_field(body),
        HttpContentType::Json => Some(BodyField::Json),
        HttpContentType::Text => Some(BodyField::Text),
        HttpContentType::Form => Some(BodyField::Form),
        HttpContentType::Binary => return decode_binary_body(body, inputs),
        HttpContentType::MultipartForm => {
            return Err(Error::unsupported(
                "multipart/form-data request bodies are not supported by the shared HTTP transport",
            ));
        }
    };

    match field {
        Some(BodyField::Json) => Ok(json_body(body.json_data.as_ref(), inputs)),
        Some(BodyField::Text) => Ok(text_body(body.text_data.as_deref(), inputs)),
        Some(BodyField::Form) => Ok(form_body(body.form_data.as_ref(), inputs)),
        None => Ok((None, None)),
    }
}

/// A JSON body: a string field is templated and parsed, object values are sent as-is.
fn json_body(data: Option<&Value>, inputs: &Value) -> (Option<RequestBody>, Option<&'static str>) {
    let Some(data) = data else {
        return (None, None);
    };
    match data {
        Value::String(template) => {
            let resolved = resolve_template(template, inputs);
            let body = serde_json::from_str::<Value>(&resolved)
                .map_or_else(|_| RequestBody::Text(resolved), RequestBody::Json);
            (Some(body), Some(APPLICATION_JSON))
        }
        other => (
            Some(RequestBody::Json(other.clone())),
            Some(APPLICATION_JSON),
        ),
    }
}

/// A text body.
fn text_body(data: Option<&str>, inputs: &Value) -> (Option<RequestBody>, Option<&'static str>) {
    data.map_or((None, None), |text| {
        (
            Some(RequestBody::Text(resolve_template(text, inputs))),
            Some(TEXT_PLAIN),
        )
    })
}

/// A URL-encoded form body.
fn form_body(
    fields: Option<&BTreeMap<String, Value>>,
    inputs: &Value,
) -> (Option<RequestBody>, Option<&'static str>) {
    fields.map_or((None, None), |fields| {
        let encoded = fields
            .iter()
            .map(|(key, value)| {
                let rendered = resolve_template(&stringify(value), inputs);
                format!("{}={}", form_encode(key), form_encode(&rendered))
            })
            .collect::<Vec<String>>()
            .join("&");
        (Some(RequestBody::Text(encoded)), Some(FORM_URLENCODED))
    })
}

/// A base64-decoded binary body.
fn decode_binary_body(
    body: &HttpRequestBodyConfig,
    inputs: &Value,
) -> Result<(Option<RequestBody>, Option<&'static str>)> {
    let Some(encoded) = body.binary_data.as_deref() else {
        return Ok((None, None));
    };
    let resolved = resolve_template(encoded, inputs);
    let decoded = BASE64_STANDARD
        .decode(resolved.as_bytes())
        .map_err(|error| {
            component_tool_input_param_error(&format!("binary_data is not valid base64, {error}"))
        })?;
    Ok((
        Some(RequestBody::Bytes(Bytes::from(decoded))),
        Some(APPLICATION_OCTET_STREAM),
    ))
}

/// Percent-encodes one `application/x-www-form-urlencoded` component.
fn form_encode(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => {
                encoded.push(char::from(byte));
            }
            b' ' => encoded.push('+'),
            other => {
                encoded.push('%');
                encoded.push(char::from(HEX[usize::from(other >> 4)]));
                encoded.push(char::from(HEX[usize::from(other & 0x0f)]));
            }
        }
    }
    encoded
}

/// The response format after `autodetect` has been resolved.
enum ResolvedFormat {
    /// Parse the body as JSON.
    Json,
    /// Decode the body as text.
    Text,
    /// Base64-encode the raw body bytes.
    Binary,
}

/// Resolves `autodetect` against the response `content-type`.
fn resolved_format(format: HttpResponseFormat, content_type: &str) -> ResolvedFormat {
    match format {
        HttpResponseFormat::Json => ResolvedFormat::Json,
        HttpResponseFormat::Text => ResolvedFormat::Text,
        HttpResponseFormat::Binary | HttpResponseFormat::Buffer => ResolvedFormat::Binary,
        HttpResponseFormat::Autodetect => {
            let lowered = content_type.to_lowercase();
            if lowered.contains("application/json") {
                ResolvedFormat::Json
            } else if lowered.contains("text/") {
                ResolvedFormat::Text
            } else {
                ResolvedFormat::Binary
            }
        }
    }
}

/// Projects an [`HttpResponse`] into Python's wire object.
fn handle_response(params: &HttpRequestParamConfig, response: &HttpResponse) -> Value {
    let handling = &params.response_handling;
    let status = response.status;
    let headers: BTreeMap<String, String> = response
        .headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_ascii_lowercase(),
                value.to_str().unwrap_or_default().to_owned(),
            )
        })
        .collect();

    let is_success = handling.response_code_success_codes.contains(&status)
        || (handling.response_code_failure_codes.is_empty() && (200..300).contains(&status));
    let is_failure = handling.response_code_failure_codes.contains(&status) || status >= 400;
    let ok = is_success && !is_failure;

    let content_type = headers.get("content-type").cloned().unwrap_or_default();
    let text = response.text();
    let mut body = match resolved_format(handling.response_format, &content_type) {
        ResolvedFormat::Json => serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)),
        ResolvedFormat::Text => Value::String(text),
        ResolvedFormat::Binary => Value::String(BASE64_STANDARD.encode(response.bytes())),
    };

    if let Some(property) = handling.response_data_property.as_deref() {
        if let Value::Object(map) = &body {
            if let Some(extracted) = map.get(property) {
                body = extracted.clone();
            }
        }
    }

    match handling.response_mode {
        HttpResponseMode::OnSuccess if !is_success => json!({}),
        HttpResponseMode::OnError if is_success => json!({}),
        _ => json!({
            "statusCode": status,
            "headers": headers,
            "body": body,
            "url": response.url,
            "ok": ok,
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("http-tests"));
        ComponentContext::new(session, "http", "wf")
    }

    // port of tests/unit_tests/core/component/test_http_component.py::test_component_creation
    #[test]
    fn config_defaults_match_python() {
        let config: HttpComponentConfig =
            serde_json::from_value(json!({"request_params": {"url": "https://example.com"}}))
                .unwrap();
        let params = &config.request_params;
        assert_eq!(params.url, "https://example.com");
        assert_eq!(params.method, "GET");
        assert_eq!(params.headers, HttpHeadersConfig::default());
        assert!(params.query_parameters.is_empty());
        assert!(params.body.is_none());
        assert!(params.authentication.is_none());
        assert_eq!(
            params.response_handling.response_code_success_codes,
            vec![200, 201, 202, 204]
        );
        assert!(params
            .response_handling
            .response_code_failure_codes
            .is_empty());
        assert_eq!(
            params.response_handling.response_mode,
            HttpResponseMode::Full
        );
        assert!(params.response_handling.throw_on_http_error);
        assert_eq!(
            params.response_handling.response_format,
            HttpResponseFormat::Autodetect
        );
        assert!(params.advanced_options.follow_redirect);
        assert!(!params.advanced_options.ignore_ssl_issues);
        assert_eq!(params.retry_config.max_retries, 3);
        assert_eq!(params.retry_config.retry_delay, 1000);
        assert_eq!(
            params.retry_config.backoff_type,
            HttpBackoffType::Exponential
        );
        assert!(!params.retry_config.enabled);
        assert_eq!(params.rate_limit_config.requests_per_unit, 1);
        assert_eq!(params.rate_limit_config.unit, HttpRateLimitUnit::Second);
        assert!((params.timeout - 60.0).abs() < f64::EPSILON);
        assert_eq!(params.max_response_byte_size, 10 * 1024 * 1024);
        let expected = HttpRequestParamConfig {
            url: "https://example.com".to_owned(),
            ..HttpRequestParamConfig::default()
        };
        assert_eq!(*params, expected);
    }

    #[test]
    fn config_tolerates_unknown_fields() {
        let config: HttpComponentConfig = serde_json::from_value(json!({
            "request_params": {
                "url": "https://example.com",
                "method": "post",
                "unknown_field": {"nested": true},
                "response_handling": {"response_mode": "on-success", "extra": 1},
                "authentication": {"type": "api_key", "api_key": "k", "in_location": "query"}
            },
            "unknown_top_level": 3
        }))
        .unwrap();
        assert_eq!(config.request_params.method, "post");
        assert_eq!(
            config.request_params.response_handling.response_mode,
            HttpResponseMode::OnSuccess
        );
        let auth = config.request_params.authentication.unwrap();
        assert_eq!(auth.auth_type, HttpAuthType::ApiKey);
        assert_eq!(auth.in_location, HttpAuthIn::Query);
        assert_eq!(auth.name, "Authorization");
        assert_eq!(config.metadata, None);
    }

    #[test]
    fn templating_resolves_braces_and_embedded_references() {
        let inputs = json!({"url": "https://example.com", "n": 3, "flag": true});
        assert_eq!(
            resolve_template("{{url}}/items/{{n}}?f={{flag}}", &inputs),
            "https://example.com/items/3?f=true"
        );
        assert_eq!(
            resolve_template("{{url}}/${n}/${missing}", &inputs),
            "https://example.com/3/${missing}"
        );
        // Text that looks like a placeholder but has no matching key stays literal.
        assert_eq!(resolve_template("{{nope}}", &inputs), "{{nope}}");
        // A scalar input object leaves the template untouched.
        assert_eq!(resolve_template("{{a}}", &json!("x")), "{{a}}");
    }

    #[test]
    fn header_string_form_is_templated_and_parsed() {
        let headers = HttpHeadersConfig::Template(
            r#"{"Authorization": "Bearer {{token}}", "X-N": {{n}}}"#.to_owned(),
        );
        let resolved = headers.resolved(&json!({"token": "abc", "n": 2}));
        assert_eq!(resolved.get("Authorization").unwrap(), "Bearer abc");
        assert_eq!(resolved.get("X-N").unwrap(), "2");
        // An unparseable template yields no headers (Python's `{}` fallback).
        assert!(HttpHeadersConfig::Template("not json".to_owned())
            .resolved(&json!({}))
            .is_empty());
    }

    #[test]
    fn dictionary_headers_are_templated_too() {
        let headers = HttpHeadersConfig::Map(BTreeMap::from([(
            "X-Token".to_owned(),
            "{{token}}".to_owned(),
        )]));
        assert_eq!(
            headers
                .resolved(&json!({"token": "abc"}))
                .get("X-Token")
                .unwrap(),
            "abc"
        );
    }

    #[test]
    fn retry_configuration_maps_onto_the_client_config() {
        let mut params = HttpRequestParamConfig {
            retry_config: HttpRetryConfig {
                enabled: true,
                max_retries: 4,
                retry_delay: 500,
                backoff_type: HttpBackoffType::Fixed,
                ..HttpRetryConfig::default()
            },
            ..HttpRequestParamConfig::default()
        };
        let config = client_config(&params);
        assert_eq!(config.max_retries, 4);
        assert_eq!(config.retry_initial_backoff, Duration::from_millis(500));
        assert_eq!(config.retry_max_backoff, Duration::from_millis(500));
        assert_eq!(config.retry_statuses, vec![429, 500, 502, 503, 504]);

        params.retry_config.backoff_type = HttpBackoffType::Exponential;
        params.retry_config.max_retries = 3;
        params.retry_config.retry_delay = 100;
        assert_eq!(
            client_config(&params).retry_max_backoff,
            Duration::from_millis(800)
        );

        params.retry_config.enabled = false;
        assert_eq!(client_config(&params).max_retries, 0);
    }

    #[test]
    fn timeout_is_clamped_into_pythons_range() {
        let params = HttpRequestParamConfig {
            timeout: 0.1,
            ..HttpRequestParamConfig::default()
        };
        assert_eq!(request_timeout(&params), Duration::from_secs(1));
        let params = HttpRequestParamConfig {
            timeout: 900.0,
            ..HttpRequestParamConfig::default()
        };
        assert_eq!(request_timeout(&params), Duration::from_secs(300));
        let params = HttpRequestParamConfig {
            timeout: f64::INFINITY,
            ..HttpRequestParamConfig::default()
        };
        assert_eq!(request_timeout(&params), Duration::from_secs(60));
    }

    #[test]
    fn form_encoding_uses_the_urlencoded_alphabet() {
        assert_eq!(form_encode("a b&c=d/é"), "a+b%26c%3Dd%2F%C3%A9");
    }

    #[test]
    fn abilities_and_card_advertise_invoke_only() {
        let component = HttpRequestComponent::new(HttpComponentConfig::default());
        assert_eq!(component.component_type(), "http_request");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        let card = component.card();
        assert_eq!(card.id, "http_request");
        assert_eq!(card.name, "http_request");
        assert_eq!(card.component_type, "http_request");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
        assert_eq!(component.config(), &HttpComponentConfig::default());
        let wire = serde_json::to_value(component.config()).unwrap();
        assert_eq!(wire["request_params"]["url"], json!(""));
        assert_eq!(wire["request_params"]["method"], json!("GET"));
        assert_eq!(wire["request_params"]["headers"], json!({}));
        assert_eq!(
            wire["request_params"]["response_handling"]["response_mode"],
            json!("full")
        );
        assert_eq!(wire["metadata"], Value::Null);
    }

    #[tokio::test]
    async fn stream_yields_exactly_one_frame() {
        // An empty URL fails before any network access, so the frame is the single error.
        let component = Arc::new(HttpRequestComponent::new(HttpComponentConfig::default()));
        let frames: Vec<_> = component
            .stream(json!({}), Arc::new(context()))
            .collect()
            .await;
        assert_eq!(frames.len(), 1);
        let error = frames.into_iter().next().unwrap().unwrap_err();
        assert!(error.message().contains("[102001]"));
    }

    #[tokio::test]
    async fn empty_url_is_an_input_parameter_error() {
        let component = HttpRequestComponent::new(HttpComponentConfig::default());
        let error = component.invoke(json!({}), &context()).await.unwrap_err();
        assert!(error.message().contains("url is required"));
    }

    #[test]
    fn missing_body_config_uses_the_input_body_as_json() {
        let params = HttpRequestParamConfig::default();
        let (body, content_type) = prepare_body(&params, &json!({"body": {"a": 1}})).unwrap();
        assert_eq!(content_type, Some(APPLICATION_JSON));
        assert!(matches!(body, Some(RequestBody::Json(Value::Object(_)))));
        // A null input body sends nothing at all.
        assert!(prepare_body(&params, &json!({})).unwrap().0.is_none());
    }

    #[test]
    fn multipart_bodies_are_unsupported() {
        let params = HttpRequestParamConfig {
            body: Some(HttpRequestBodyConfig {
                content_type: HttpContentType::MultipartForm,
                multipart_form: Some(BTreeMap::from([("f".to_owned(), json!("v"))])),
                ..HttpRequestBodyConfig::default()
            }),
            ..HttpRequestParamConfig::default()
        };
        let error = prepare_body(&params, &json!({})).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
    }

    #[test]
    fn digest_and_aws_authentication_are_unsupported() {
        for (auth_type, needle) in [(HttpAuthType::Digest, "digest"), (HttpAuthType::Aws, "aws")] {
            let auth = HttpAuthConfig {
                auth_type,
                ..HttpAuthConfig::default()
            };
            let mut headers = BTreeMap::new();
            let mut query = Vec::new();
            let error = apply_authentication(Some(&auth), &mut headers, &mut query).unwrap_err();
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Unsupported
            );
            assert!(error.message().contains(needle));
        }
    }

    #[test]
    fn authentication_places_credentials_like_python() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Basic,
            username: Some("user".to_owned()),
            password: Some("pass".to_owned()),
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        let mut query = Vec::new();
        apply_authentication(Some(&auth), &mut headers, &mut query).unwrap();
        assert_eq!(headers.get("Authorization").unwrap(), "Basic dXNlcjpwYXNz");

        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::ApiKey,
            api_key: Some("k".to_owned()),
            name: "X-Key".to_owned(),
            in_location: HttpAuthIn::Query,
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        let mut query = Vec::new();
        apply_authentication(Some(&auth), &mut headers, &mut query).unwrap();
        assert_eq!(query, vec![("X-Key".to_owned(), "k".to_owned())]);
        assert!(headers.is_empty());

        // `in_location = body` is ignored, as in Python.
        let auth = HttpAuthConfig {
            in_location: HttpAuthIn::Body,
            ..auth
        };
        let mut query = Vec::new();
        apply_authentication(Some(&auth), &mut BTreeMap::new(), &mut query).unwrap();
        assert!(query.is_empty());
    }
}
