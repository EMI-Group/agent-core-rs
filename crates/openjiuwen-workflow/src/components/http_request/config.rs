//! Configuration tree of the `http_request` component.
//!
//! Ports Python's `HttpComponentConfig` and its nested request, body, auth, response,
//! retry, rate-limit and advanced-option models together with their wire enums.
//! The component itself and the templating/execution logic live in the parent module.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{resolve_template, stringify, DEFAULT_TIMEOUT_SECONDS};

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
    /// Multipart form body from `multipart_form`, encoded in-crate into bytes.
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
    /// HTTP digest authentication, answered from the server's challenge.
    Digest,
    /// AWS Signature Version 4, signed in pre-flight.
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
    /// Raw bytes as a Node-style buffer object
    /// (`{"type": "Buffer", "data": [..u8..], "length": n}`).
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
    pub(super) fn resolved(&self, inputs: &Value) -> BTreeMap<String, String> {
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
    /// Multipart form fields, encoded in-crate into a `multipart/form-data` body.
    ///
    /// A value that is an object with a `filename` plus a base64 `content_base64` (or the
    /// `data`/`content` aliases) becomes a file part; every other value is a text part.
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
    /// Return a coded [`super::COMPONENT_HTTP_ERROR`] error for a failed status instead
    /// of the `statusCode`/`ok = false` data object (Python default `true`).
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
    /// Largest accepted request body in bytes (Python default 1 MiB); `0` is unlimited.
    ///
    /// Enforced for every body kind (JSON, text, form, binary, multipart) with a typed
    /// input-parameter error naming the limit and the actual size.
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
/// Enforced by a per-component sliding-window limiter (`super::rate_limit`): one limiter
/// per component instance, never process-global, and the required delay is awaited before
/// each send while honouring the run's cancellation token. `requests_per_unit = 0`
/// disables the limiter even when `enabled` is set.
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
    /// Sliding-window rate limiting of the sends this component performs.
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

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
}
