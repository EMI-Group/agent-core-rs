//! Request-body preparation for the `http_request` component.
//!
//! Every body kind is rendered here — JSON, text, URL-encoded form, base64 binary, and
//! `multipart/form-data` — together with the `Content-Type` it implies and the
//! `advanced_options.max_body_length` check.

use std::collections::BTreeMap;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use serde_json::Value;

use openjiuwen_common::error::Result;

use super::config::{HttpContentType, HttpRequestBodyConfig, HttpRequestParamConfig};
use super::multipart::{self, MULTIPART_FORM_DATA};
use super::template::{resolve_template, stringify};
use crate::error::component_tool_input_param_error;

/// The `Content-Type` used for JSON request bodies.
const APPLICATION_JSON: &str = "application/json";
/// The `Content-Type` used for text request bodies.
const TEXT_PLAIN: &str = "text/plain";
/// The `Content-Type` used for form request bodies.
const FORM_URLENCODED: &str = "application/x-www-form-urlencoded";
/// The `Content-Type` used for binary request bodies.
const APPLICATION_OCTET_STREAM: &str = "application/octet-stream";

/// The body kinds the shared transport can send.
#[derive(Debug, Clone)]
pub(crate) enum RequestBody {
    /// A JSON value, serialized by the transport.
    Json(Value),
    /// A pre-rendered text body.
    Text(String),
    /// Raw request bytes.
    Bytes(Bytes),
}

impl RequestBody {
    /// The exact bytes the transport will send for this body.
    pub(crate) fn payload(&self) -> Result<Bytes> {
        match self {
            Self::Json(value) => serde_json::to_vec(value).map(Bytes::from).map_err(|error| {
                component_tool_input_param_error(&format!("request body is invalid, {error}"))
            }),
            Self::Text(text) => Ok(Bytes::copy_from_slice(text.as_bytes())),
            Self::Bytes(bytes) => Ok(bytes.clone()),
        }
    }

    /// The encoded length of this body in bytes.
    pub(crate) fn payload_len(&self) -> Result<usize> {
        Ok(self.payload()?.len())
    }
}

/// Which body field an `auto` content type selected.
enum BodyField {
    /// `json_data`.
    Json,
    /// `text_data`.
    Text,
    /// `form_data`.
    Form,
    /// `multipart_form`.
    Multipart,
}

/// Builds the request body and the `Content-Type` it implies.
pub(crate) fn prepare_body(
    params: &HttpRequestParamConfig,
    inputs: &Value,
) -> Result<(Option<RequestBody>, Option<String>)> {
    let Some(body) = params.body.as_ref() else {
        // Python sends no body here; sending `inputs["body"]` as JSON is a Rust extension.
        return Ok(match inputs.get("body") {
            Some(value) if !value.is_null() => (
                Some(RequestBody::Json(value.clone())),
                Some(APPLICATION_JSON.to_owned()),
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
        HttpContentType::MultipartForm => Some(BodyField::Multipart),
    };

    match field {
        Some(BodyField::Json) => Ok(json_body(body.json_data.as_ref(), inputs)),
        Some(BodyField::Text) => Ok(text_body(body.text_data.as_deref(), inputs)),
        Some(BodyField::Form) => Ok(form_body(body.form_data.as_ref(), inputs)),
        Some(BodyField::Multipart) => Ok(multipart_body(body.multipart_form.as_ref(), inputs)),
        None => Ok((None, None)),
    }
}

/// Rejects a body larger than `advanced_options.max_body_length` (`0` means unlimited).
pub(crate) fn enforce_max_body_length(
    max_body_length: u64,
    body: Option<&RequestBody>,
) -> Result<()> {
    let Some(body) = body else {
        return Ok(());
    };
    if max_body_length == 0 {
        return Ok(());
    }
    let actual = body.payload_len()?;
    let actual = u64::try_from(actual).unwrap_or(u64::MAX);
    if actual > max_body_length {
        return Err(component_tool_input_param_error(&format!(
            "request body is {actual} bytes, exceeding advanced_options.max_body_length of \
             {max_body_length} bytes"
        )));
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
    } else if body.multipart_form.is_some() {
        Some(BodyField::Multipart)
    } else {
        None
    }
}

/// A JSON body: a string field is templated and parsed, object values are sent as-is.
fn json_body(data: Option<&Value>, inputs: &Value) -> (Option<RequestBody>, Option<String>) {
    let Some(data) = data else {
        return (None, None);
    };
    let content_type = Some(APPLICATION_JSON.to_owned());
    match data {
        Value::String(template) => {
            let resolved = resolve_template(template, inputs);
            let body = serde_json::from_str::<Value>(&resolved)
                .map_or_else(|_| RequestBody::Text(resolved), RequestBody::Json);
            (Some(body), content_type)
        }
        other => (Some(RequestBody::Json(other.clone())), content_type),
    }
}

/// A text body.
fn text_body(data: Option<&str>, inputs: &Value) -> (Option<RequestBody>, Option<String>) {
    data.map_or((None, None), |text| {
        (
            Some(RequestBody::Text(resolve_template(text, inputs))),
            Some(TEXT_PLAIN.to_owned()),
        )
    })
}

/// A URL-encoded form body.
fn form_body(
    fields: Option<&BTreeMap<String, Value>>,
    inputs: &Value,
) -> (Option<RequestBody>, Option<String>) {
    fields.map_or((None, None), |fields| {
        let encoded = fields
            .iter()
            .map(|(key, value)| {
                let rendered = resolve_template(&stringify(value), inputs);
                format!("{}={}", form_encode(key), form_encode(&rendered))
            })
            .collect::<Vec<String>>()
            .join("&");
        (
            Some(RequestBody::Text(encoded)),
            Some(FORM_URLENCODED.to_owned()),
        )
    })
}

/// A `multipart/form-data` body, encoded into bytes by [`super::multipart`].
fn multipart_body(
    fields: Option<&BTreeMap<String, Value>>,
    inputs: &Value,
) -> (Option<RequestBody>, Option<String>) {
    fields.map_or((None, None), |fields| {
        let encoded = multipart::encode(fields, inputs);
        let content_type = encoded.content_type();
        debug_assert!(content_type.starts_with(MULTIPART_FORM_DATA));
        (Some(RequestBody::Bytes(encoded.body)), Some(content_type))
    })
}

/// A base64-decoded binary body.
fn decode_binary_body(
    body: &HttpRequestBodyConfig,
    inputs: &Value,
) -> Result<(Option<RequestBody>, Option<String>)> {
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
        Some(APPLICATION_OCTET_STREAM.to_owned()),
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Request params carrying `body`.
    fn params(body: HttpRequestBodyConfig) -> HttpRequestParamConfig {
        HttpRequestParamConfig {
            body: Some(body),
            ..HttpRequestParamConfig::default()
        }
    }

    #[test]
    fn form_encoding_uses_the_urlencoded_alphabet() {
        assert_eq!(form_encode("a b&c=d/é"), "a+b%26c%3Dd%2F%C3%A9");
    }

    #[test]
    fn missing_body_config_uses_the_input_body_as_json() {
        let default_params = HttpRequestParamConfig::default();
        let (body, content_type) =
            prepare_body(&default_params, &json!({"body": {"a": 1}})).unwrap();
        assert_eq!(content_type.as_deref(), Some(APPLICATION_JSON));
        assert!(matches!(body, Some(RequestBody::Json(Value::Object(_)))));
        // A null input body sends nothing at all.
        assert!(prepare_body(&default_params, &json!({}))
            .unwrap()
            .0
            .is_none());
    }

    #[test]
    fn every_body_kind_reports_its_content_type() {
        let (_, content_type) = prepare_body(
            &params(HttpRequestBodyConfig {
                content_type: HttpContentType::Json,
                json_data: Some(json!({"a": 1})),
                ..HttpRequestBodyConfig::default()
            }),
            &json!({}),
        )
        .unwrap();
        assert_eq!(content_type.as_deref(), Some(APPLICATION_JSON));

        let (_, content_type) = prepare_body(
            &params(HttpRequestBodyConfig {
                content_type: HttpContentType::Text,
                text_data: Some("hi".to_owned()),
                ..HttpRequestBodyConfig::default()
            }),
            &json!({}),
        )
        .unwrap();
        assert_eq!(content_type.as_deref(), Some(TEXT_PLAIN));

        let (_, content_type) = prepare_body(
            &params(HttpRequestBodyConfig {
                content_type: HttpContentType::Form,
                form_data: Some(BTreeMap::from([("a".to_owned(), json!("b c"))])),
                ..HttpRequestBodyConfig::default()
            }),
            &json!({}),
        )
        .unwrap();
        assert_eq!(content_type.as_deref(), Some(FORM_URLENCODED));

        let (_, content_type) = prepare_body(
            &params(HttpRequestBodyConfig {
                content_type: HttpContentType::Binary,
                binary_data: Some("eA==".to_owned()),
                ..HttpRequestBodyConfig::default()
            }),
            &json!({}),
        )
        .unwrap();
        assert_eq!(content_type.as_deref(), Some(APPLICATION_OCTET_STREAM));

        let (body, content_type) = prepare_body(
            &params(HttpRequestBodyConfig {
                content_type: HttpContentType::MultipartForm,
                multipart_form: Some(BTreeMap::from([("f".to_owned(), json!("v"))])),
                ..HttpRequestBodyConfig::default()
            }),
            &json!({}),
        )
        .unwrap();
        assert_eq!(body.unwrap().payload().unwrap(), {
            let content_type = content_type.expect("multipart implies a content type");
            let boundary = content_type
                .strip_prefix("multipart/form-data; boundary=")
                .expect("the content type carries the boundary");
            Bytes::from(format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"f\"\r\n\r\nv\r\n\
                 --{boundary}--\r\n"
            ))
        });
    }

    #[test]
    fn auto_selects_the_first_present_field() {
        let body = HttpRequestBodyConfig {
            content_type: HttpContentType::Auto,
            text_data: Some("t".to_owned()),
            form_data: Some(BTreeMap::from([("a".to_owned(), json!(1))])),
            multipart_form: Some(BTreeMap::from([("b".to_owned(), json!(2))])),
            ..HttpRequestBodyConfig::default()
        };
        let (prepared, content_type) = prepare_body(&params(body), &json!({})).unwrap();
        assert_eq!(content_type.as_deref(), Some(TEXT_PLAIN));
        assert!(matches!(prepared, Some(RequestBody::Text(_))));

        let body = HttpRequestBodyConfig {
            content_type: HttpContentType::Auto,
            multipart_form: Some(BTreeMap::from([("b".to_owned(), json!(2))])),
            ..HttpRequestBodyConfig::default()
        };
        let (prepared, content_type) = prepare_body(&params(body), &json!({})).unwrap();
        assert!(
            content_type.unwrap().starts_with(MULTIPART_FORM_DATA),
            "auto falls back to multipart when only `multipart_form` is set"
        );
        assert!(matches!(prepared, Some(RequestBody::Bytes(_))));
    }

    #[test]
    fn an_invalid_binary_body_is_an_input_parameter_error() {
        let error = prepare_body(
            &params(HttpRequestBodyConfig {
                content_type: HttpContentType::Binary,
                binary_data: Some("not base64!".to_owned()),
                ..HttpRequestBodyConfig::default()
            }),
            &json!({}),
        )
        .unwrap_err();
        assert!(error.message().contains("[102001]"));
        assert!(error.message().contains("binary_data"));
    }

    #[test]
    fn the_body_length_limit_applies_to_every_kind() {
        let limit = 4;
        let too_long = RequestBody::Text("hello".to_owned());
        let error = enforce_max_body_length(limit, Some(&too_long)).unwrap_err();
        assert!(error.message().contains("[102001]"));
        assert!(error.message().contains("5 bytes"), "{}", error.message());
        assert!(error.message().contains("max_body_length of 4"));

        // Exactly at the limit passes, and `0` disables the check.
        assert!(enforce_max_body_length(4, Some(&RequestBody::Text("hi!!".to_owned()))).is_ok());
        assert!(enforce_max_body_length(0, Some(&too_long)).is_ok());
        // No body never exceeds the limit.
        assert!(enforce_max_body_length(1, None).is_ok());
        // JSON bodies are measured on their serialized form.
        assert_eq!(RequestBody::Json(json!({"a": 1})).payload_len().unwrap(), 7);
    }
}
