//! Built-in `http_request` workflow component.
//!
//! Ports `openjiuwen/core/workflow/components/tool/http/http_request_component.py`
//! (`HttpComponentConfig`, `HTTPRequestExecutable`, `HTTPRequestComponent`). The component
//! turns its configuration into an [`HttpRequest`], executes it through the shared
//! [`openjiuwen_common::http::HttpClient`], and projects the response into Python's
//! `statusCode` / `headers` / `body` / `url` / `ok` wire object.
//!
//! The implementation is split by responsibility: [`config`] holds the configuration tree,
//! [`template`] the `{{key}}`/`${path}` templating, [`body`] and [`multipart`] the request
//! bodies, [`auth`] with [`digest`] and [`sigv4`] the authentication schemes, [`response`]
//! the response projection, and [`rate_limit`] the sliding-window limiter.
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
//! - **HTTP failures** are returned as data (`ok = false`) when
//!   `response_handling.throw_on_http_error` is `false`, and as a coded
//!   [`COMPONENT_HTTP_ERROR`] (`[102002]`) error otherwise. `ok` and the status code are
//!   part of that error's message. Transport, timeout, cancellation, size, and
//!   client-build failures are always [`component_tool_execution_error`] (`[102000]`).
//! - **Bodies**: a body whose encoded size exceeds `advanced_options.max_body_length`
//!   (default 1 MiB, `0` = unlimited) is rejected with a typed input-parameter error
//!   naming both sizes. The limit covers JSON, text, form, binary and multipart bodies.
//! - **Multipart**: `content_type = multipart_form` encodes the form (boundary, parts,
//!   CRLF framing) into bytes and sends it as a binary body, so retries, cancellation and
//!   the response cap keep applying. A `{"filename": …, "content_base64": …}` value is a
//!   file part (`data`/`content` are accepted aliases); other objects are JSON text.
//! - **Digest authentication** is challenge-driven: the request is sent unauthenticated,
//!   and a `401` carrying `WWW-Authenticate: Digest …` is answered by replaying the
//!   identical request once with an `Authorization: Digest …` header (MD5, MD5-sess,
//!   SHA-256, SHA-256-sess; `qop=auth` when offered, otherwise the legacy form). A missing
//!   `username`/`password` pair is a typed input-parameter error.
//! - **AWS SigV4 authentication** signs in pre-flight, after the body bytes are known:
//!   credentials come from `authentication.username`/`password` or
//!   `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, the region from `AWS_REGION`/
//!   `AWS_DEFAULT_REGION` (default `us-east-1`), and the service from `AWS_SERVICE`
//!   (default `execute-api`). `X-Amz-Date`, `X-Amz-Content-Sha256`, the session token of
//!   `AWS_SESSION_TOKEN`, and every sent header are signed.
//! - **Rate limiting**: `rate_limit_config.enabled` installs one sliding-window limiter per
//!   component instance (no process-global state); the delay it requires is awaited
//!   before each send, honouring the run's cancellation token.
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
//! - **Binary responses** are base64-encoded (`BASE64_STANDARD`) because JSON has no
//!   bytes type; Python returns `bytes`. `response_format = buffer` projects the bytes as
//!   `{"type": "Buffer", "data": [..u8..], "length": n}` instead (lossless, Node-style).
//!   `binary_data` requests are base64-decoded.
//! - **JSON string bodies** are parsed with `serde_json` instead of Python's
//!   `ast.literal_eval`, so single-quoted Python dict literals are rejected; a parse
//!   failure sends the resolved text as a text body (Python substitutes `{}`).
//! - **Bodies** are sent for every method (Python sends them only for POST/PUT/PATCH),
//!   and a body `content-type` is only added when the header is still absent (Python
//!   overwrites `Content-Type`).
//! - **Digest / AWS authentication** are implemented here while Python only declares the
//!   configuration (`# Note: Digest and AWS authentication would require additional
//!   implementation`). A digest challenge the component cannot answer (another scheme,
//!   an unsupported algorithm, or a replay that is still `401`) is returned as ordinary
//!   response data. An API-key `in_location = body` is ignored, as in Python.
//! - `advanced_options.use_stream` is accepted but ignored (responses are buffered), as
//!   is `advanced_options.disable_compression` (it is not part of the Rust config).
//! - Python's `response_handling.max_redirects` is not modelled; redirect handling
//!   follows `advanced_options.follow_redirect`.
//! - `timeout` is clamped into `1.0..=300.0` seconds at use time (Python validates the
//!   same range); a non-finite timeout falls back to 60 seconds instead of panicking.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse, Method};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::error::{component_tool_execution_error, component_tool_input_param_error};

mod auth;
mod body;
mod config;
mod digest;
mod multipart;
mod rate_limit;
mod response;
mod sigv4;
mod template;

pub use config::{
    HttpAdvancedOptionsConfig, HttpAuthConfig, HttpAuthIn, HttpAuthType, HttpBackoffType,
    HttpComponentConfig, HttpContentType, HttpHeadersConfig, HttpRateLimitConfig,
    HttpRateLimitUnit, HttpRequestBodyConfig, HttpRequestParamConfig, HttpResponseFormat,
    HttpResponseHandlingConfig, HttpResponseMode, HttpRetryConfig,
};

pub(crate) use template::{resolve_template, stringify};

use body::RequestBody;
use rate_limit::RateLimiter;
use response::handle_response;

/// The component-local status code of a response that `throw_on_http_error` turns into an
/// error (`102002`).
///
/// The numeric slot is shared with Python's `COMPONENT_TOOL_INIT_FAILED`: the Python
/// component never raises on an HTTP status, so no status code was ever allocated for it.
/// Keeping the constant here (instead of in [`crate::error`]) documents that the meaning is
/// local to this component.
pub const COMPONENT_HTTP_ERROR: u32 = 102_002;

/// The timeout applied when the configured value is not a finite number.
const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
/// The Python range for `request_params.timeout`, in seconds.
const MIN_TIMEOUT_SECONDS: f64 = 1.0;
/// The Python range for `request_params.timeout`, in seconds.
const MAX_TIMEOUT_SECONDS: f64 = 300.0;

/// The built-in `http_request` component.
///
/// `invoke` resolves the request from the component inputs and executes it; failures are
/// reported as `[102000]`/`[102001]`/`[102002]` errors and non-2xx responses are ordinary
/// outputs unless `response_handling.throw_on_http_error` says otherwise.
#[derive(Debug, Clone)]
pub struct HttpRequestComponent {
    config: HttpComponentConfig,
    /// The sliding-window limiter, shared by every clone of this component.
    limiter: Option<Arc<RateLimiter>>,
}

impl HttpRequestComponent {
    /// Creates a component from `config`.
    #[must_use]
    pub fn new(config: HttpComponentConfig) -> Self {
        let limiter = RateLimiter::new(&config.request_params.rate_limit_config);
        Self { config, limiter }
    }

    /// The component configuration (Python's `config` property).
    #[must_use]
    pub fn config(&self) -> &HttpComponentConfig {
        &self.config
    }

    /// Sends one request, waiting for the rate limiter and honouring cancellation.
    async fn send(
        &self,
        client: &HttpClient,
        request: HttpRequest,
        ctx: &ComponentContext,
    ) -> Result<HttpResponse> {
        if let Some(limiter) = &self.limiter {
            limiter.acquire(ctx.cancellation()).await?;
        }
        client
            .execute_cancellable(request, ctx.cancellation().clone())
            .await
            .map_err(|error| request_failed(&error.message()))
    }

    /// Replays `request` once with a digest `Authorization` header when `response` is a
    /// `401` challenge this component can answer.
    async fn replay_digest(
        &self,
        params: &HttpRequestParamConfig,
        prepared: &PreparedRequest,
        response: HttpResponse,
        client: &HttpClient,
        ctx: &ComponentContext,
    ) -> Result<HttpResponse> {
        if response.status != 401 {
            return Ok(response);
        }
        let Some(auth) = params
            .authentication
            .as_ref()
            .filter(|auth| auth.auth_type == HttpAuthType::Digest)
        else {
            return Ok(response);
        };
        let Some(header) = response
            .headers
            .get("www-authenticate")
            .and_then(|value| value.to_str().ok())
        else {
            return Ok(response);
        };
        let Some(challenge) = digest::DigestChallenge::parse(header) else {
            // Another scheme or an unsupported algorithm: the 401 stays ordinary data.
            return Ok(response);
        };
        let (username, password) = auth::digest_credentials(auth)?;
        let authorization = digest::authorization(
            &challenge,
            username,
            password,
            prepared.request.method().as_str(),
            &prepared.digest_uri,
        );
        let replay = prepared
            .request
            .clone()
            .header("Authorization", authorization);
        self.send(client, replay, ctx).await
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
        let prepared = build_request(params, &inputs, ctx.cancellation().clone())?;
        let client = HttpClient::new(client_config(params))
            .map_err(|error| request_failed(&error.message()))?;
        let response = self.send(&client, prepared.request.clone(), ctx).await?;
        let response = self
            .replay_digest(params, &prepared, response, &client, ctx)
            .await?;
        handle_response(params, &response)
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

/// A built request plus the request-target a digest challenge must echo.
#[derive(Debug)]
struct PreparedRequest {
    /// The request to send.
    request: HttpRequest,
    /// The `uri` parameter of a digest `Authorization` header.
    digest_uri: String,
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
) -> Result<PreparedRequest> {
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

    let (body, content_type) = body::prepare_body(params, inputs)?;
    body::enforce_max_body_length(params.advanced_options.max_body_length, body.as_ref())?;
    if let Some(content_type) = content_type {
        set_header_if_absent(&mut headers, "content-type", &content_type);
    }

    auth::apply_authentication(params.authentication.as_ref(), &mut headers, &mut query)?;
    if auth::is_aws(params.authentication.as_ref()) {
        let payload = body
            .as_ref()
            .map_or_else(|| Ok(bytes::Bytes::new()), RequestBody::payload)?;
        auth::sign_aws(
            params.authentication.as_ref(),
            &method_text,
            &url,
            &query,
            &mut headers,
            &payload,
        )?;
    }

    let digest_uri = digest_uri(&url, &query);
    let request = HttpRequest::new(method, url)
        .with_headers(headers)
        .queries(query)
        .timeout(request_timeout(params))
        .cancel(cancellation);

    let request = match body {
        Some(RequestBody::Json(value)) => request.json(&value).map_err(|error| {
            component_tool_input_param_error(&format!("request body is invalid, {error}"))
        })?,
        Some(RequestBody::Text(text)) => request.text(text),
        Some(RequestBody::Bytes(bytes)) => request.bytes(bytes),
        None => request,
    };
    Ok(PreparedRequest {
        request,
        digest_uri,
    })
}

/// The request-target a digest `Authorization` header echoes: path plus query parameters.
fn digest_uri(url: &str, query: &[(String, String)]) -> String {
    let without_fragment = url.split('#').next().unwrap_or(url);
    let after_scheme = without_fragment
        .split_once("://")
        .map_or(without_fragment, |(_, rest)| rest);
    let mut uri = after_scheme
        .find('/')
        .map_or_else(|| "/".to_owned(), |index| after_scheme[index..].to_owned());
    for (index, (key, value)) in query.iter().enumerate() {
        uri.push(if index == 0 && !uri.contains('?') {
            '?'
        } else {
            '&'
        });
        uri.push_str(&sigv4::rfc3986_encode(key));
        uri.push('=');
        uri.push_str(&sigv4::rfc3986_encode(value));
    }
    uri
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use openjiuwen_common::http::HttpBody;

    use super::*;
    use crate::components::http_request::{
        HttpAuthConfig, HttpAuthType, HttpContentType, HttpRequestBodyConfig,
    };

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("http-tests"));
        ComponentContext::new(session, "http", "wf")
    }

    /// Request params with `body` and `authentication` filled in.
    fn params(
        body: Option<HttpRequestBodyConfig>,
        authentication: Option<HttpAuthConfig>,
    ) -> HttpRequestParamConfig {
        HttpRequestParamConfig {
            url: "https://example.com/x".to_owned(),
            body,
            authentication,
            ..HttpRequestParamConfig::default()
        }
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
        assert!(
            component.limiter.is_none(),
            "rate limiting is off by default"
        );
    }

    #[test]
    fn a_rate_limited_component_holds_one_shared_limiter() {
        let config: HttpComponentConfig = serde_json::from_value(json!({
            "request_params": {
                "rate_limit_config": {
                    "enabled": true,
                    "requests_per_unit": 2,
                    "unit": "minute"
                }
            }
        }))
        .unwrap();
        let component = HttpRequestComponent::new(config);
        let limiter = component.limiter.clone().expect("the limiter is installed");
        assert!(Arc::ptr_eq(
            &limiter,
            component.limiter.as_ref().expect("still installed")
        ));
        // A clone shares the same window.
        let clone = component.clone();
        let cloned_limiter = clone.limiter.clone().expect("clones share it");
        assert!(Arc::ptr_eq(&limiter, &cloned_limiter));
    }

    #[tokio::test]
    async fn stream_yields_exactly_one_frame() {
        // An empty URL fails before any network access, so the frame is the single error.
        let component = Arc::new(HttpRequestComponent::new(HttpComponentConfig::default()));
        let ctx = Arc::new(context());
        let frames: Vec<_> = component.stream(json!({}), ctx).collect().await;
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
    fn a_multipart_body_becomes_a_binary_request_with_its_boundary() {
        let params = params(
            Some(HttpRequestBodyConfig {
                content_type: HttpContentType::MultipartForm,
                multipart_form: Some(BTreeMap::from([("f".to_owned(), json!("v"))])),
                ..HttpRequestBodyConfig::default()
            }),
            None,
        );
        let prepared = build_request(&params, &json!({}), CancellationToken::new()).unwrap();
        let content_type = prepared
            .request
            .headers()
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
            .map(|(_, value)| value.clone())
            .unwrap();
        assert!(
            content_type.starts_with("multipart/form-data; boundary="),
            "unexpected content type: {content_type}"
        );
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        assert!(matches!(
            prepared.request.body(),
            Some(HttpBody::Bytes(bytes)) if bytes.starts_with(format!("--{boundary}\r\n").as_bytes())
        ));
    }

    #[test]
    fn a_body_over_the_configured_limit_is_rejected() {
        let params = HttpRequestParamConfig {
            url: "https://example.com/x".to_owned(),
            body: Some(HttpRequestBodyConfig {
                content_type: HttpContentType::Text,
                text_data: Some("0123456789".to_owned()),
                ..HttpRequestBodyConfig::default()
            }),
            advanced_options: HttpAdvancedOptionsConfig {
                max_body_length: 4,
                ..HttpAdvancedOptionsConfig::default()
            },
            ..HttpRequestParamConfig::default()
        };
        let error = build_request(&params, &json!({}), CancellationToken::new()).unwrap_err();
        assert!(error.message().contains("[102001]"), "{}", error.message());
        assert!(error.message().contains("10 bytes"));
        assert!(error.message().contains("max_body_length of 4"));
    }

    #[test]
    fn digest_credentials_are_validated_but_not_applied() {
        let with_credentials = params(
            None,
            Some(HttpAuthConfig {
                auth_type: HttpAuthType::Digest,
                username: Some("u".to_owned()),
                password: Some("p".to_owned()),
                ..HttpAuthConfig::default()
            }),
        );
        let prepared =
            build_request(&with_credentials, &json!({}), CancellationToken::new()).unwrap();
        assert!(!prepared
            .request
            .headers()
            .keys()
            .any(|name| name.eq_ignore_ascii_case("authorization")));

        let without_credentials = params(
            None,
            Some(HttpAuthConfig {
                auth_type: HttpAuthType::Digest,
                ..HttpAuthConfig::default()
            }),
        );
        let error =
            build_request(&without_credentials, &json!({}), CancellationToken::new()).unwrap_err();
        assert!(error.message().contains("[102001]"), "{}", error.message());
    }

    #[test]
    fn aws_authentication_signs_the_request_in_pre_flight() {
        let params = params(
            Some(HttpRequestBodyConfig {
                content_type: HttpContentType::Json,
                json_data: Some(json!({"a": 1})),
                ..HttpRequestBodyConfig::default()
            }),
            Some(HttpAuthConfig {
                auth_type: HttpAuthType::Aws,
                username: Some("AKIDEXAMPLE".to_owned()),
                password: Some("secret".to_owned()),
                ..HttpAuthConfig::default()
            }),
        );
        let prepared = build_request(&params, &json!({}), CancellationToken::new()).unwrap();
        let authorization = prepared
            .request
            .headers()
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.clone())
            .expect("the request is signed");
        assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
        assert!(
            authorization
                .contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date"),
            "unexpected authorization: {authorization}"
        );
        let content_hash = prepared
            .request
            .headers()
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-amz-content-sha256"))
            .map(|(_, value)| value.clone())
            .unwrap();
        assert_eq!(content_hash, sigv4::sha256_hex(br#"{"a":1}"#));
    }

    #[test]
    fn a_missing_aws_credential_pair_is_an_input_parameter_error() {
        let params = params(
            None,
            Some(HttpAuthConfig {
                auth_type: HttpAuthType::Aws,
                // `auth::aws_credentials` falls back to the environment; an empty
                // configuration cannot name a pair, and the process is not expected to
                // have AWS credentials exported in the test environment.
                username: None,
                password: None,
                ..HttpAuthConfig::default()
            }),
        );
        if std::env::var(auth::ACCESS_KEY_ENV).is_ok()
            && std::env::var(auth::SECRET_KEY_ENV).is_ok()
        {
            // A developer machine with real credentials still signs; nothing to assert.
            assert!(build_request(&params, &json!({}), CancellationToken::new()).is_ok());
            return;
        }
        let error = build_request(&params, &json!({}), CancellationToken::new()).unwrap_err();
        assert!(
            error.message().contains(auth::ACCESS_KEY_ENV),
            "{}",
            error.message()
        );
    }

    #[test]
    fn the_digest_uri_carries_the_query_parameters() {
        assert_eq!(digest_uri("https://example.com/a/b", &[]), "/a/b");
        assert_eq!(digest_uri("https://example.com", &[]), "/");
        assert_eq!(
            digest_uri(
                "https://example.com/a?x=1",
                &[("y".to_owned(), "a b".to_owned())]
            ),
            "/a?x=1&y=a%20b"
        );
        assert_eq!(
            digest_uri(
                "https://example.com/a",
                &[
                    ("y".to_owned(), "1".to_owned()),
                    ("z".to_owned(), "2".to_owned())
                ]
            ),
            "/a?y=1&z=2"
        );
    }
}
