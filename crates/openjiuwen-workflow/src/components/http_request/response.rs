//! Response projection for the `http_request` component.
//!
//! A response always becomes Python's `statusCode` / `headers` / `body` / `url` / `ok`
//! wire object — unless `response_handling.throw_on_http_error` is set and the status
//! counts as a failure, in which case the component returns a coded error instead.
//!
//! # Body shapes
//!
//! | `response_format` | `body` wire value |
//! | --- | --- |
//! | `json` (and `autodetect` for `application/json`) | the parsed JSON value |
//! | `text` (and `autodetect` for `text/*`) | a lossy UTF-8 string |
//! | `binary` (and `autodetect` otherwise) | a base64 string (`BASE64_STANDARD`, Python parity) |
//! | `buffer` | `{"type": "Buffer", "data": [..u8..], "length": n}` |
//!
//! `buffer` is the lossless, JSON-encodable byte shape (Node's `Buffer#toJSON`); `binary`
//! keeps the base64 string Python returns. `response_data_property` extracts one property
//! from an object body, including the buffer object's own keys.

use std::collections::BTreeMap;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;

use super::config::{
    HttpRequestParamConfig, HttpResponseFormat, HttpResponseHandlingConfig, HttpResponseMode,
};
use super::COMPONENT_HTTP_ERROR;

/// How many body characters an HTTP-error message carries.
const ERROR_BODY_PREVIEW: usize = 512;

/// The buffered body of a response, before it is projected onto the JSON wire.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResponseBody {
    /// A parsed JSON document.
    Json(Value),
    /// Lossy UTF-8 text.
    Text(String),
    /// A base64-encoded binary body (Python parity).
    Base64(String),
    /// The raw bytes, projected as a Node-style buffer object.
    Buffer(Bytes),
}

impl ResponseBody {
    /// The JSON wire value of this body.
    pub(crate) fn into_wire(self) -> Value {
        match self {
            Self::Json(value) => value,
            Self::Text(text) => Value::String(text),
            Self::Base64(encoded) => Value::String(encoded),
            Self::Buffer(bytes) => json!({
                "type": "Buffer",
                "data": bytes.to_vec(),
                "length": bytes.len(),
            }),
        }
    }
}

/// A response in the shape the projection logic consumes.
pub(crate) struct ResponseView<'a> {
    /// The numeric status code.
    pub(crate) status: u16,
    /// The headers, names lower-cased.
    pub(crate) headers: BTreeMap<String, String>,
    /// The final URL after redirects.
    pub(crate) url: &'a str,
    /// The raw body.
    pub(crate) body: &'a Bytes,
}

impl<'a> ResponseView<'a> {
    /// Views one [`HttpResponse`].
    pub(crate) fn new(response: &'a HttpResponse) -> Self {
        Self {
            status: response.status,
            headers: response
                .headers
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_ascii_lowercase(),
                        value.to_str().unwrap_or_default().to_owned(),
                    )
                })
                .collect(),
            url: &response.url,
            body: &response.body,
        }
    }

    /// The body decoded as lossy UTF-8 text.
    fn text(&self) -> String {
        String::from_utf8_lossy(self.body).into_owned()
    }
}

/// The response format after `autodetect` has been resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedFormat {
    /// Parse the body as JSON.
    Json,
    /// Decode the body as text.
    Text,
    /// Base64-encode the raw body bytes.
    Binary,
    /// Project the raw body bytes as a buffer object.
    Buffer,
}

/// Resolves `autodetect` against the response `content-type`.
fn resolved_format(format: HttpResponseFormat, content_type: &str) -> ResolvedFormat {
    match format {
        HttpResponseFormat::Json => ResolvedFormat::Json,
        HttpResponseFormat::Text => ResolvedFormat::Text,
        HttpResponseFormat::Binary => ResolvedFormat::Binary,
        HttpResponseFormat::Buffer => ResolvedFormat::Buffer,
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

/// Whether `status` counts as a success under `handling`.
fn is_success(handling: &HttpResponseHandlingConfig, status: u16) -> bool {
    handling.response_code_success_codes.contains(&status)
        || (handling.response_code_failure_codes.is_empty() && (200..300).contains(&status))
}

/// Whether `status` counts as a failure under `handling`.
fn is_failure(handling: &HttpResponseHandlingConfig, status: u16) -> bool {
    handling.response_code_failure_codes.contains(&status) || status >= 400
}

/// Projects a response into Python's wire object, or into a coded HTTP error.
pub(crate) fn handle_response(
    params: &HttpRequestParamConfig,
    response: &HttpResponse,
) -> Result<Value> {
    project(params, &ResponseView::new(response))
}

/// Projects one [`ResponseView`].
fn project(params: &HttpRequestParamConfig, response: &ResponseView<'_>) -> Result<Value> {
    let handling = &params.response_handling;
    let status = response.status;
    let success = is_success(handling, status);
    let failure = is_failure(handling, status);
    let ok = success && !failure;

    if failure && handling.throw_on_http_error {
        return Err(http_error(response, ok));
    }

    let content_type = response
        .headers
        .get("content-type")
        .cloned()
        .unwrap_or_default();
    let body = match resolved_format(handling.response_format, &content_type) {
        ResolvedFormat::Json => {
            let text = response.text();
            ResponseBody::Json(serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)))
                .into_wire()
        }
        ResolvedFormat::Text => ResponseBody::Text(response.text()).into_wire(),
        ResolvedFormat::Binary => {
            ResponseBody::Base64(BASE64_STANDARD.encode(response.body)).into_wire()
        }
        ResolvedFormat::Buffer => ResponseBody::Buffer(response.body.clone()).into_wire(),
    };

    let body = match handling.response_data_property.as_deref() {
        Some(property) => match &body {
            Value::Object(map) => map.get(property).cloned().unwrap_or(body),
            _ => body,
        },
        None => body,
    };

    Ok(match handling.response_mode {
        HttpResponseMode::OnSuccess if !success => json!({}),
        HttpResponseMode::OnError if success => json!({}),
        _ => json!({
            "statusCode": status,
            "headers": response.headers,
            "body": body,
            "url": response.url,
            "ok": ok,
        }),
    })
}

/// Renders a failed response as the component-local `[102002]` error.
fn http_error(response: &ResponseView<'_>, ok: bool) -> Error {
    let text = response.text();
    let truncated = text.chars().count() > ERROR_BODY_PREVIEW;
    let mut preview: String = text.chars().take(ERROR_BODY_PREVIEW).collect();
    if truncated {
        preview.push('…');
    }
    Error::workflow(format!(
        "[{COMPONENT_HTTP_ERROR}] the HTTP request failed, statusCode={}, ok={ok}, url={}, \
         body={preview}",
        response.status, response.url,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::components::http_request::{HttpRequestParamConfig, HttpResponseMode};

    /// A response view over `body` with the given status and content type.
    fn view<'a>(status: u16, content_type: &str, body: &'a Bytes) -> ResponseView<'a> {
        ResponseView {
            status,
            headers: BTreeMap::from([("content-type".to_owned(), content_type.to_owned())]),
            url: "https://example.com/x",
            body,
        }
    }

    /// Request params with `handling` as the response handling.
    fn params(handling: HttpResponseHandlingConfig) -> HttpRequestParamConfig {
        HttpRequestParamConfig {
            response_handling: handling,
            ..HttpRequestParamConfig::default()
        }
    }

    #[test]
    fn a_binary_body_is_base64_and_a_buffer_body_is_bytes() {
        let body = Bytes::from_static(&[0, 1, 2, 250]);
        let handling = HttpResponseHandlingConfig {
            response_format: HttpResponseFormat::Binary,
            ..HttpResponseHandlingConfig::default()
        };
        let output = project(&params(handling), &view(200, "application/pdf", &body)).unwrap();
        assert_eq!(output["body"], json!("AAEC+g=="));

        let handling = HttpResponseHandlingConfig {
            response_format: HttpResponseFormat::Buffer,
            ..HttpResponseHandlingConfig::default()
        };
        let output = project(&params(handling), &view(200, "application/pdf", &body)).unwrap();
        assert_eq!(output["body"]["type"], json!("Buffer"));
        assert_eq!(output["body"]["data"], json!([0, 1, 2, 250]));
        assert_eq!(output["body"]["length"], json!(4));
        assert_eq!(output["ok"], json!(true));
    }

    #[test]
    fn autodetect_picks_json_text_or_binary() {
        let json_body = Bytes::from_static(br#"{"a":1}"#);
        let handling = HttpResponseHandlingConfig::default();
        assert_eq!(
            project(
                &params(handling.clone()),
                &view(200, "application/json", &json_body)
            )
            .unwrap()["body"],
            json!({"a": 1})
        );
        let text_body = Bytes::from_static(b"hello");
        assert_eq!(
            project(
                &params(handling.clone()),
                &view(200, "text/plain; charset=utf-8", &text_body)
            )
            .unwrap()["body"],
            json!("hello")
        );
        assert_eq!(
            project(&params(handling), &view(200, "image/png", &text_body)).unwrap()["body"],
            json!("aGVsbG8=")
        );
    }

    #[test]
    fn a_failure_throws_by_default_and_is_data_when_disabled() {
        let body = Bytes::from_static(br#"{"error":"boom"}"#);
        let error = project(
            &params(HttpResponseHandlingConfig::default()),
            &view(500, "application/json", &body),
        )
        .unwrap_err();
        assert!(error.message().contains("[102002]"), "{}", error.message());
        assert!(error.message().contains("statusCode=500"));
        assert!(error.message().contains("ok=false"));
        assert!(error.message().contains("boom"));

        let handling = HttpResponseHandlingConfig {
            throw_on_http_error: false,
            ..HttpResponseHandlingConfig::default()
        };
        let output = project(&params(handling), &view(500, "application/json", &body)).unwrap();
        assert_eq!(output["statusCode"], json!(500));
        assert_eq!(output["ok"], json!(false));
        assert_eq!(output["body"], json!({"error": "boom"}));
    }

    #[test]
    fn a_configured_failure_status_throws_too() {
        let body = Bytes::from_static(b"ok");
        let handling = HttpResponseHandlingConfig {
            response_code_failure_codes: vec![200],
            ..HttpResponseHandlingConfig::default()
        };
        let error = project(&params(handling), &view(200, "text/plain", &body)).unwrap_err();
        assert!(error.message().contains("statusCode=200"));
        assert!(error.message().contains("ok=false"));
    }

    #[test]
    fn response_data_property_extracts_object_fields() {
        let body = Bytes::from_static(br#"{"data":{"id":7},"other":1}"#);
        let handling = HttpResponseHandlingConfig {
            response_data_property: Some("data".to_owned()),
            ..HttpResponseHandlingConfig::default()
        };
        let output = project(&params(handling), &view(200, "application/json", &body)).unwrap();
        assert_eq!(output["body"], json!({"id": 7}));

        // A missing property leaves the body untouched.
        let handling = HttpResponseHandlingConfig {
            response_data_property: Some("missing".to_owned()),
            ..HttpResponseHandlingConfig::default()
        };
        let output = project(&params(handling), &view(200, "application/json", &body)).unwrap();
        assert_eq!(output["body"], json!({"data": {"id": 7}, "other": 1}));

        // A non-object body cannot be projected.
        let handling = HttpResponseHandlingConfig {
            response_data_property: Some("data".to_owned()),
            ..HttpResponseHandlingConfig::default()
        };
        let raw = Bytes::from_static(b"plain");
        let output = project(&params(handling), &view(200, "text/plain", &raw)).unwrap();
        assert_eq!(output["body"], json!("plain"));
    }

    #[test]
    fn response_mode_projects_on_success_and_on_error() {
        let body = Bytes::from_static(br#"{"v":1}"#);
        let handling = HttpResponseHandlingConfig {
            response_mode: HttpResponseMode::OnSuccess,
            throw_on_http_error: false,
            ..HttpResponseHandlingConfig::default()
        };
        assert_eq!(
            project(
                &params(handling.clone()),
                &view(500, "application/json", &body)
            )
            .unwrap(),
            json!({})
        );
        let handling = HttpResponseHandlingConfig {
            response_mode: HttpResponseMode::OnError,
            throw_on_http_error: false,
            ..HttpResponseHandlingConfig::default()
        };
        assert_eq!(
            project(
                &params(handling.clone()),
                &view(200, "application/json", &body)
            )
            .unwrap(),
            json!({})
        );
        let output = project(&params(handling), &view(403, "application/json", &body)).unwrap();
        assert_eq!(output["statusCode"], json!(403));
        assert_eq!(output["ok"], json!(false));
    }

    #[test]
    fn a_buffer_body_exposes_its_length_as_a_data_property() {
        let body = Bytes::from_static(&[9, 8, 7]);
        let handling = HttpResponseHandlingConfig {
            response_format: HttpResponseFormat::Buffer,
            response_data_property: Some("length".to_owned()),
            ..HttpResponseHandlingConfig::default()
        };
        let output = project(&params(handling), &view(200, "image/png", &body)).unwrap();
        assert_eq!(output["body"], json!(3));
    }

    #[test]
    fn headers_are_lower_cased_in_the_output() {
        let body = Bytes::from_static(b"x");
        let mut response = view(200, "text/plain", &body);
        let _ = response
            .headers
            .insert("x-trace".to_owned(), "abc".to_owned());
        let output = project(&params(HttpResponseHandlingConfig::default()), &response).unwrap();
        assert_eq!(output["headers"]["x-trace"], json!("abc"));
        assert_eq!(output["headers"]["content-type"], json!("text/plain"));
        assert_eq!(output["url"], json!("https://example.com/x"));
    }
}
