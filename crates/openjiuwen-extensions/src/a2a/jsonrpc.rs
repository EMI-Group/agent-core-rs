//! `JSON-RPC` 2.0 envelopes, A2A error codes, and method names.
//!
//! Port of the transport half of `openjiuwen.extensions.a2a` (v0.1.20): the
//! Python implementation delegates the envelopes to `a2a-sdk`, which speaks
//! `JSON-RPC` 2.0 over HTTP with `text/event-stream` for streaming calls. This
//! module owns those envelopes directly.
//!
//! Two versions of the method namespace exist on the wire: A2A v1 (the
//! canonical `PascalCase` names, e.g. `SendMessage`) and A2A v0.3 (the lowercase
//! slash names, e.g. `message/send`). Outgoing calls always use the canonical
//! name; [`method::canonical`] accepts both so a v0.3 peer can still talk to the
//! server.
use openjiuwen_common::error::Error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
/// The `JSON-RPC` version string every envelope carries.
const DEFAULT_VERSION: &str = "2.0";
/// The `jsonrpc` field default: `"2.0"`.
fn default_version() -> String {
    DEFAULT_VERSION.to_owned()
}
/// The standard and A2A-specific `JSON-RPC` error codes.
pub mod error_code {
    /// The request body was not valid JSON.
    pub const PARSE: i32 = -32700;
    /// The request was valid JSON but not a valid request object.
    pub const INVALID_REQUEST: i32 = -32600;
    /// The requested method does not exist.
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// The request parameters were invalid.
    pub const INVALID_PARAMS: i32 = -32602;
    /// The server hit an internal error.
    pub const INTERNAL: i32 = -32603;
    /// The referenced task does not exist.
    pub const TASK_NOT_FOUND: i32 = -32001;
    /// The referenced task is not in a cancelable state.
    pub const TASK_NOT_CANCELABLE: i32 = -32002;
    /// The agent does not support push notifications.
    pub const PUSH_NOT_SUPPORTED: i32 = -32003;
    /// The agent does not support the requested operation.
    pub const UNSUPPORTED_OPERATION: i32 = -32004;
    /// The request media type is not supported.
    pub const CONTENT_TYPE_NOT_SUPPORTED: i32 = -32005;
    /// The agent produced a response that violates the protocol.
    pub const INVALID_AGENT_RESPONSE: i32 = -32006;
    /// The extended agent card was requested but is not configured.
    pub const EXTENDED_CARD_NOT_CONFIGURED: i32 = -32007;
    /// The peer requires an extension that the client did not declare.
    pub const EXTENSION_SUPPORT_REQUIRED: i32 = -32008;
    /// The requested protocol version is not supported.
    pub const VERSION_NOT_SUPPORTED: i32 = -32009;
}
/// The A2A method names, canonical form and v0.3 aliases.
pub mod method {
    /// Send one message and wait for the final result.
    pub const SEND_MESSAGE: &str = "SendMessage";
    /// Send one message and stream the results back.
    pub const SEND_STREAMING_MESSAGE: &str = "SendStreamingMessage";
    /// Fetch a task by id.
    pub const GET_TASK: &str = "GetTask";
    /// Cancel a task by id.
    pub const CANCEL_TASK: &str = "CancelTask";
    /// Re-subscribe to the event stream of an existing task.
    pub const SUBSCRIBE_TO_TASK: &str = "SubscribeToTask";
    /// The v0.3 lowercase aliases, mapped to their canonical v1 names.
    pub const V0_3_ALIASES: [(&str, &str); 5] = [
        ("message/send", SEND_MESSAGE),
        ("message/stream", SEND_STREAMING_MESSAGE),
        ("tasks/get", GET_TASK),
        ("tasks/cancel", CANCEL_TASK),
        ("tasks/resubscribe", SUBSCRIBE_TO_TASK),
    ];
    /// The canonical `PascalCase` method name for a canonical name or a v0.3
    /// alias.
    ///
    /// Matching is exact (case-sensitive); surrounding whitespace and a
    /// trailing slash are tolerated, so `"message/send/"` still resolves. A
    /// name that is neither canonical nor a known alias yields `None`.
    #[must_use]
    pub fn canonical(method: &str) -> Option<&'static str> {
        let trimmed = method.trim().trim_end_matches('/');
        let canonical_names = [
            SEND_MESSAGE,
            SEND_STREAMING_MESSAGE,
            GET_TASK,
            CANCEL_TASK,
            SUBSCRIBE_TO_TASK,
        ];
        for name in canonical_names {
            if name == trimmed {
                return Some(name);
            }
        }
        for (alias, name) in V0_3_ALIASES {
            if alias == trimmed {
                return Some(name);
            }
        }
        None
    }
    /// Whether a method streams its results (`SendStreamingMessage`,
    /// `SubscribeToTask`, and their v0.3 aliases).
    #[must_use]
    pub fn is_streaming(method: &str) -> bool {
        matches!(
            canonical(method),
            Some(SEND_STREAMING_MESSAGE | SUBSCRIBE_TO_TASK)
        )
    }
}
/// A `JSON-RPC` 2.0 request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// The protocol version; defaults to `"2.0"` when the peer omits it.
    #[serde(default = "default_version")]
    pub jsonrpc: String,
    /// The request id, echoed back on the response; `null` for notifications.
    #[serde(default)]
    pub id: Value,
    /// The method name, canonical or a v0.3 alias.
    #[serde(default)]
    pub method: String,
    /// The method parameters, omitted when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}
impl JsonRpcRequest {
    /// A request carrying the `"2.0"` version.
    #[must_use]
    pub fn new(id: impl Into<Value>, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: default_version(),
            id: id.into(),
            method: method.into(),
            params,
        }
    }
}
impl Default for JsonRpcRequest {
    /// An empty request with no id, no method, and no params, at version
    /// `"2.0"`.
    fn default() -> Self {
        Self {
            jsonrpc: default_version(),
            id: Value::Null,
            method: String::new(),
            params: None,
        }
    }
}
/// A `JSON-RPC` 2.0 response: exactly one of `result` or `error` is present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// The protocol version, `"2.0"`.
    #[serde(default = "default_version")]
    pub jsonrpc: String,
    /// The id of the request this response answers.
    #[serde(default)]
    pub id: Value,
    /// The success payload; omitted for failures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// The failure payload; omitted for successes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}
impl JsonRpcResponse {
    /// A success response.
    #[must_use]
    pub fn success(id: impl Into<Value>, result: Value) -> Self {
        Self {
            jsonrpc: default_version(),
            id: id.into(),
            result: Some(result),
            error: None,
        }
    }
    /// A failure response.
    #[must_use]
    pub fn failure(id: impl Into<Value>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: default_version(),
            id: id.into(),
            result: None,
            error: Some(error),
        }
    }
    /// Whether this response carries a result rather than an error.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.error.is_none()
    }
}
/// A `JSON-RPC` 2.0 error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// One of the [`error_code`] constants (or a vendor-specific code).
    pub code: i32,
    /// A human-readable message.
    #[serde(default)]
    pub message: String,
    /// Optional structured detail; omitted when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}
impl JsonRpcError {
    /// An error without structured detail.
    #[must_use]
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
    /// The same error with structured detail attached.
    #[must_use]
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
    /// Converts this wire error into the crate-wide [`Error`].
    ///
    /// The code decides the [`openjiuwen_common::error::ErrorKind`]: a missing
    /// method or an unsupported operation is `Unsupported`, a missing task is
    /// `NotFound`, and everything else — including
    /// [`error_code::TASK_NOT_CANCELABLE`] and [`error_code::INVALID_PARAMS`] —
    /// is `Validation`. The message is `json-rpc error <code>: <message>`, with
    /// the `data` payload appended as compact JSON when present.
    pub fn into_error(self) -> Error {
        let mut message = format!("json-rpc error {}: {}", self.code, self.message);
        if let Some(data) = self.data.as_ref() {
            message.push_str(": ");
            message.push_str(&data.to_string());
        }
        match self.code {
            error_code::METHOD_NOT_FOUND | error_code::UNSUPPORTED_OPERATION => {
                Error::unsupported(message)
            }
            error_code::TASK_NOT_FOUND => Error::not_found(message),
            _ => Error::validation(message),
        }
    }
}
/// Parses a `JSON-RPC` request body.
///
/// # Errors
///
/// Returns [`error_code::PARSE`] when the body is not valid UTF-8 or not valid
/// `JSON-RPC` JSON, and [`error_code::INVALID_REQUEST`] when the request has no
/// method.
pub fn parse_request(body: &[u8]) -> std::result::Result<JsonRpcRequest, JsonRpcError> {
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        JsonRpcError::new(
            error_code::PARSE,
            format!("invalid JSON-RPC request: {error}"),
        )
    })?;
    if !value.is_object() {
        return Err(JsonRpcError::new(
            error_code::INVALID_REQUEST,
            "JSON-RPC request must be a JSON object",
        ));
    }
    let request: JsonRpcRequest = serde_json::from_value(value).map_err(|error| {
        JsonRpcError::new(
            error_code::PARSE,
            format!("invalid JSON-RPC request: {error}"),
        )
    })?;
    if request.method.trim().is_empty() {
        return Err(JsonRpcError::new(
            error_code::INVALID_REQUEST,
            "JSON-RPC request is missing a method",
        ));
    }
    Ok(request)
}
/// Serializes a `JSON-RPC` response.
///
/// Never panics and never fails: if the response cannot be serialized (which a
/// plain `serde_json::Value` payload cannot cause) an
/// [`error_code::INTERNAL`] envelope is returned instead.
#[must_use]
pub fn serialize(response: &JsonRpcResponse) -> String {
    serde_json::to_string(response).unwrap_or_else(|_| {
        r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"failed to serialize JSON-RPC response"}}"#
            .to_owned()
    })
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    #[test]
    fn error_codes_match_the_a2a_spec() {
        // pins the table from `a2a.types.a2a_pb2` / the JSON-RPC 2.0 spec.
        assert_eq!(error_code::PARSE, -32700);
        assert_eq!(error_code::INVALID_REQUEST, -32600);
        assert_eq!(error_code::METHOD_NOT_FOUND, -32601);
        assert_eq!(error_code::INVALID_PARAMS, -32602);
        assert_eq!(error_code::INTERNAL, -32603);
        assert_eq!(error_code::TASK_NOT_FOUND, -32001);
        assert_eq!(error_code::TASK_NOT_CANCELABLE, -32002);
        assert_eq!(error_code::PUSH_NOT_SUPPORTED, -32003);
        assert_eq!(error_code::UNSUPPORTED_OPERATION, -32004);
        assert_eq!(error_code::CONTENT_TYPE_NOT_SUPPORTED, -32005);
        assert_eq!(error_code::INVALID_AGENT_RESPONSE, -32006);
        assert_eq!(error_code::EXTENDED_CARD_NOT_CONFIGURED, -32007);
        assert_eq!(error_code::EXTENSION_SUPPORT_REQUIRED, -32008);
        assert_eq!(error_code::VERSION_NOT_SUPPORTED, -32009);
    }
    #[test]
    fn canonical_methods_resolve_to_themselves() {
        for name in [
            method::SEND_MESSAGE,
            method::SEND_STREAMING_MESSAGE,
            method::GET_TASK,
            method::CANCEL_TASK,
            method::SUBSCRIBE_TO_TASK,
        ] {
            assert_eq!(method::canonical(name), Some(name));
        }
        assert_eq!(method::SEND_MESSAGE, "SendMessage");
        assert_eq!(method::SEND_STREAMING_MESSAGE, "SendStreamingMessage");
        assert_eq!(method::GET_TASK, "GetTask");
        assert_eq!(method::CANCEL_TASK, "CancelTask");
        assert_eq!(method::SUBSCRIBE_TO_TASK, "SubscribeToTask");
    }
    #[test]
    fn v0_3_aliases_resolve_to_the_canonical_name() {
        // ports the v0.3-compatible dispatch in `test_a2a_server.py`.
        assert_eq!(method::V0_3_ALIASES.len(), 5);
        for (alias, expected) in [
            ("message/send", method::SEND_MESSAGE),
            ("message/stream", method::SEND_STREAMING_MESSAGE),
            ("tasks/get", method::GET_TASK),
            ("tasks/cancel", method::CANCEL_TASK),
            ("tasks/resubscribe", method::SUBSCRIBE_TO_TASK),
        ] {
            assert_eq!(method::canonical(alias), Some(expected), "alias {alias}");
        }
    }
    #[test]
    fn canonical_tolerates_whitespace_and_trailing_slash() {
        assert_eq!(
            method::canonical("  message/send  "),
            Some(method::SEND_MESSAGE)
        );
        assert_eq!(
            method::canonical("message/stream/"),
            Some(method::SEND_STREAMING_MESSAGE)
        );
        assert_eq!(method::canonical("CancelTask/"), Some(method::CANCEL_TASK));
    }
    #[test]
    fn unknown_methods_resolve_to_none() {
        for raw in ["", "Unknown", "message", "message/unknown", "sendMessage"] {
            assert_eq!(method::canonical(raw), None, "`{raw}` must not resolve");
        }
    }
    #[test]
    fn streaming_methods_are_flagged() {
        for raw in [
            "SendStreamingMessage",
            "message/stream",
            "SubscribeToTask",
            "tasks/resubscribe",
        ] {
            assert!(method::is_streaming(raw), "`{raw}` streams");
        }
        for raw in [
            "SendMessage",
            "message/send",
            "GetTask",
            "tasks/get",
            "CancelTask",
            "Unknown",
        ] {
            assert!(!method::is_streaming(raw), "`{raw}` does not stream");
        }
    }
    #[test]
    fn json_rpc_error_maps_to_error_kinds() {
        let unsupported = [
            error_code::METHOD_NOT_FOUND,
            error_code::UNSUPPORTED_OPERATION,
        ];
        for code in unsupported {
            assert_eq!(
                JsonRpcError::new(code, "nope").into_error().kind(),
                ErrorKind::Unsupported
            );
        }
        assert_eq!(
            JsonRpcError::new(error_code::TASK_NOT_FOUND, "gone")
                .into_error()
                .kind(),
            ErrorKind::NotFound
        );
        for code in [
            error_code::TASK_NOT_CANCELABLE,
            error_code::INVALID_PARAMS,
            error_code::PARSE,
            error_code::INVALID_REQUEST,
            error_code::CONTENT_TYPE_NOT_SUPPORTED,
            error_code::INVALID_AGENT_RESPONSE,
            error_code::INTERNAL,
            error_code::PUSH_NOT_SUPPORTED,
            error_code::EXTENDED_CARD_NOT_CONFIGURED,
            error_code::EXTENSION_SUPPORT_REQUIRED,
            error_code::VERSION_NOT_SUPPORTED,
            1234,
        ] {
            assert_eq!(
                JsonRpcError::new(code, "bad").into_error().kind(),
                ErrorKind::Validation,
                "code {code}"
            );
        }
    }
    #[test]
    fn json_rpc_error_message_includes_code_message_and_data() {
        let plain = JsonRpcError::new(error_code::TASK_NOT_FOUND, "no such task").into_error();
        assert_eq!(
            plain.to_string(),
            "not_found: json-rpc error -32001: no such task"
        );
        let detailed = JsonRpcError::new(error_code::INVALID_PARAMS, "bad params")
            .with_data(json!({ "field": "message" }))
            .into_error();
        assert_eq!(
            detailed.to_string(),
            "validation: json-rpc error -32602: bad params: {\"field\":\"message\"}"
        );
    }
    #[test]
    fn responses_carry_exactly_one_of_result_or_error() {
        let success = JsonRpcResponse::success(json!("req-1"), json!({ "ok": true }));
        assert!(success.is_success());
        assert_eq!(
            serde_json::to_value(&success).unwrap(),
            json!({ "jsonrpc": "2.0", "id": "req-1", "result": { "ok": true } })
        );
        let failure = JsonRpcResponse::failure(
            json!(7),
            JsonRpcError::new(error_code::TASK_NOT_FOUND, "no such task"),
        );
        assert!(!failure.is_success());
        assert_eq!(
            serde_json::to_value(&failure).unwrap(),
            json!({
                "jsonrpc": "2.0",
                "id": 7,
                "error": { "code": -32001, "message": "no such task" },
            })
        );
        let decoded: JsonRpcResponse =
            serde_json::from_value(json!({ "jsonrpc": "2.0", "id": 1, "result": null })).unwrap();
        assert!(decoded.is_success());
        // `null` and an absent key both decode to `None`; a success response
        // never carries an `error`.
        assert!(decoded.result.is_none());
        assert!(decoded.error.is_none());
        let error_only: JsonRpcResponse =
            serde_json::from_value(json!({ "id": 2, "error": { "code": 1, "message": "x" } }))
                .unwrap();
        assert_eq!(error_only.jsonrpc, "2.0");
        assert!(error_only.result.is_none());
        assert!(!error_only.is_success());
    }
    #[test]
    fn request_constructor_and_defaults() {
        let request = JsonRpcRequest::new(1_u64, method::SEND_MESSAGE, Some(json!({ "a": 1 })));
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.id, json!(1));
        assert_eq!(request.method, "SendMessage");
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({ "jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": { "a": 1 } })
        );
        let bare = JsonRpcRequest::new(Value::Null, method::GET_TASK, None);
        assert_eq!(
            serde_json::to_value(&bare).unwrap(),
            json!({ "jsonrpc": "2.0", "id": null, "method": "GetTask" })
        );
        let default = JsonRpcRequest::default();
        assert_eq!(default.jsonrpc, "2.0");
        assert_eq!(default.id, Value::Null);
        assert_eq!(default.method, "");
        assert!(default.params.is_none());
    }
    #[test]
    fn parse_request_accepts_a_well_formed_body() {
        let request =
            parse_request(br#"{"jsonrpc":"2.0","id":3,"method":"message/send"}"#).unwrap();
        assert_eq!(request.id, json!(3));
        assert_eq!(request.method, "message/send");
        assert!(request.params.is_none());
        let with_params =
            parse_request(br#"{"id":"x","method":"SendMessage","params":{"message":{}}}"#).unwrap();
        assert_eq!(with_params.jsonrpc, "2.0");
        assert_eq!(with_params.params, Some(json!({ "message": {} })));
    }
    #[test]
    fn parse_request_reports_parse_errors() {
        for body in [&b"not json"[..], &b"{"[..], &b"\xff\xfe"[..]] {
            let error = parse_request(body).unwrap_err();
            assert_eq!(error.code, error_code::PARSE);
            assert!(error.message.starts_with("invalid JSON-RPC request: "));
        }
        // Valid JSON that is not a request object is an invalid request, not a
        // parse error.
        for body in [
            &b"[]"[..],
            &b"42"[..],
            &b"null"[..],
            &br#""SendMessage""#[..],
        ] {
            let error = parse_request(body).unwrap_err();
            assert_eq!(error.code, error_code::INVALID_REQUEST);
            assert_eq!(error.message, "JSON-RPC request must be a JSON object");
        }
        // Wrong field types are a decode failure of the envelope.
        let error = parse_request(br#"{"method":42}"#).unwrap_err();
        assert_eq!(error.code, error_code::PARSE);
    }
    #[test]
    fn parse_request_rejects_an_empty_method() {
        for body in [
            r#"{"jsonrpc":"2.0","id":1,"method":""}"#,
            r#"{"jsonrpc":"2.0","id":1,"method":"   "}"#,
            r#"{"jsonrpc":"2.0","id":1}"#,
        ] {
            let error = parse_request(body.as_bytes()).unwrap_err();
            assert_eq!(error.code, error_code::INVALID_REQUEST);
            assert_eq!(error.message, "JSON-RPC request is missing a method");
        }
    }
    #[test]
    fn serialize_produces_parseable_json() {
        let response = JsonRpcResponse::success(json!(1), json!({ "taskId": "t" }));
        let text = serialize(&response);
        let parsed: JsonRpcResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, response);
        let failure = JsonRpcResponse::failure(
            Value::Null,
            JsonRpcError::new(error_code::INTERNAL, "boom").with_data(json!([1, 2])),
        );
        let parsed: JsonRpcResponse = serde_json::from_str(&serialize(&failure)).unwrap();
        assert_eq!(parsed, failure);
    }
}
