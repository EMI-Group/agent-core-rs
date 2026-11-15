#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    dead_code,
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound,
    clippy::assert_is_empty
)]
//! `http_request` component behavior through the public workflow API.
//!
//! The component is exercised end to end against a loopback `wiremock` server, so the
//! real request encoding, response projection, and retry loop run — no transport stub is
//! involved. An HTTP failure is data (`statusCode`, `ok = false`) when
//! `response_handling.throw_on_http_error` is `false` and a coded `[102002]` error
//! otherwise; transport failures become the coded `[102000]` workflow error.
//!
//! The suite also covers the body encodings (JSON, form, multipart), the authentication
//! schemes (basic, API key, digest challenge/replay, AWS `SigV4`), `max_body_length`,
//! `rate_limit_config` (including cancellation), and the difference between the `binary`
//! and `buffer` response formats.
//!
//! Ports of `tests/unit_tests/core/component/test_http_component.py` and
//! `…/test_http_component_comprehensive.py`, rewritten against `openjiuwen_workflow`'s
//! `Component`/`Workflow` API. Everything is offline and deterministic: the mock server
//! binds loopback, URLs come from its `uri()`, and no assertion depends on timing.
mod common;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::wiremock::matchers::{
    body_json, body_string_contains, header_regex, method, path, query_param,
};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use openjiuwen_workflow::components::{HttpComponentConfig, HttpRequestComponent};
use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, ComponentRegistry, EndComponent, StartComponent,
    Workflow, WorkflowExecutionState,
};

use common::{card, session};

/// Builds a component from a `request_params` object; every other field keeps its default.
fn component(request_params: &Value) -> HttpRequestComponent {
    let config = serde_json::from_value::<HttpComponentConfig>(json!({
        "request_params": request_params
    }))
    .expect("the request params describe a valid http component config");
    HttpRequestComponent::new(config)
}

/// The session facade a directly-invoked component receives.
fn context() -> ComponentContext {
    ComponentContext::new(session("http-tests"), "http", "wf")
}

/// A component for `path` on `server`, with `extra` merged into its request params.
fn component_at(server: &MockServer, path: &str, extra: &Value) -> HttpRequestComponent {
    let mut params = extra
        .as_object()
        .cloned()
        .expect("the extra request params are a JSON object");
    let _ = params.insert("url".to_owned(), json!(format!("{}{path}", server.uri())));
    component(&Value::Object(params))
}

// port of tests/unit_tests/core/component/test_http_component.py::test_http_comp_002
// (the response wire object) and …_comprehensive.py::test_basic_get_request
#[tokio::test]
async fn get_returns_status_body_headers_and_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/greet"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(r#"{"message":"hi"}"#, "application/json"),
        )
        .mount(&server)
        .await;

    let output = component(&json!({"url": format!("{}/greet", server.uri())}))
        .invoke(json!({}), &context())
        .await
        .unwrap();

    assert_eq!(output["statusCode"], json!(200));
    assert_eq!(output["body"], json!({"message": "hi"}));
    assert_eq!(output["ok"], json!(true));
    assert_eq!(
        output["headers"]["content-type"],
        json!("application/json"),
        "response header names are lower-cased"
    );
    let response_url = output["url"].as_str().expect("url is a string");
    assert!(
        response_url.ends_with("/greet"),
        "unexpected response url: {response_url}"
    );

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method.as_str(), "GET");
    assert_eq!(requests[0].url.path(), "/greet");
}

// port of tests/unit_tests/core/component/test_http_component.py::test_start_http_request_end_in_workflow
// (url built from an input) and ::test_http_comp_002 (templated query parameters)
#[tokio::test]
async fn placeholders_and_references_resolve_from_the_inputs() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/items/42"))
        .and(query_param("q", "hello"))
        .and(query_param("page", "2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"found": true})))
        .mount(&server)
        .await;

    let http = component(&json!({
        "url": "{{base}}/{{resource}}/items/${id}",
        "query_parameters": {"q": "{{term}}", "page": 2, "unmatched": "{{nope}}"}
    }));
    let output = http
        .invoke(
            json!({"base": server.uri(), "resource": "v1", "id": "42", "term": "hello"}),
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(output["statusCode"], json!(200));
    assert_eq!(output["body"], json!({"found": true}));

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    let recorded = &requests[0];
    assert_eq!(recorded.url.path(), "/v1/items/42");
    let pairs: Vec<(String, String)> = recorded
        .url
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    assert!(pairs.contains(&("q".to_owned(), "hello".to_owned())));
    assert!(pairs.contains(&("page".to_owned(), "2".to_owned())));
    assert_eq!(
        pairs
            .iter()
            .find(|(key, _)| key == "unmatched")
            .map(|(_, value)| value.as_str()),
        Some("{{nope}}"),
        "a placeholder with no matching input stays literal"
    );
}

// port of tests/unit_tests/core/component/test_http_component_comprehensive.py::test_post_request_with_body
#[tokio::test]
async fn a_post_sends_the_configured_json_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/users"))
        .and(body_json(json!({"name": "ada", "n": 1})))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({"created": true})))
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/users",
        &json!({
            "method": "post",
            "body": {"content_type": "json", "json_data": {"name": "ada", "n": 1}}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();

    assert_eq!(output["statusCode"], json!(201));
    assert_eq!(output["body"], json!({"created": true}));
    assert_eq!(output["ok"], json!(true));
}

// port of tests/unit_tests/core/component/test_http_component.py::test_retry_count_and_timeout_session_handling
// (Python returns the 500 response body instead of raising)
#[tokio::test]
async fn a_server_error_is_data_when_throwing_is_disabled() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"error": "boom"})))
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/broken",
        &json!({"response_handling": {"throw_on_http_error": false}}),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();

    assert_eq!(output["statusCode"], json!(500));
    assert_eq!(output["ok"], json!(false));
    assert_eq!(output["body"], json!({"error": "boom"}));
}

// `throw_on_http_error` defaults to `true`, so a failed status becomes the component-local
// `[102002]` error naming the status and `ok`.
#[tokio::test]
async fn a_server_error_throws_by_default() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({"error": "boom"})))
        .mount(&server)
        .await;

    let error = component_at(&server, "/broken", &json!({}))
        .invoke(json!({}), &context())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    let text = error.to_string();
    assert!(text.contains("[102002]"), "unexpected error text: {text}");
    assert!(
        text.contains("statusCode=503"),
        "unexpected error text: {text}"
    );
    assert!(text.contains("ok=false"), "unexpected error text: {text}");
}

// port of tests/unit_tests/core/component/test_http_component_comprehensive.py::test_response_handling
#[tokio::test]
async fn response_data_property_and_response_mode_project_the_output() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/data"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"data": {"id": 7}, "other": 1})),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/server-error"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"error": "boom"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/ok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"value": 1})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/forbidden"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({"error": "denied"})))
        .mount(&server)
        .await;

    let extracted = component_at(
        &server,
        "/data",
        &json!({"response_handling": {"response_data_property": "data"}}),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(
        extracted["body"],
        json!({"id": 7}),
        "response_data_property extracts the nested object"
    );

    let success_only_error = component_at(
        &server,
        "/server-error",
        &json!({
            "response_handling": {"response_mode": "on-success", "throw_on_http_error": false}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(success_only_error, json!({}));

    let error_only_success = component_at(
        &server,
        "/ok",
        &json!({"response_handling": {"response_mode": "on-error"}}),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(error_only_success, json!({}));

    let error_only_failure = component_at(
        &server,
        "/forbidden",
        &json!({
            "response_handling": {"response_mode": "on-error", "throw_on_http_error": false}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(error_only_failure["statusCode"], json!(403));
    assert_eq!(error_only_failure["ok"], json!(false));
    assert_eq!(error_only_failure["body"], json!({"error": "denied"}));
}

// port of tests/unit_tests/core/component/test_http_component.py::test_retry_count_and_timeout_session_handling
#[tokio::test]
async fn a_status_based_retry_recovers() {
    let server = MockServer::start().await;
    // Mounted first, so it wins on the equal (default) priority until it is exhausted.
    Mock::given(method("GET"))
        .and(path("/flaky"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/flaky"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"recovered": true})))
        .mount(&server)
        .await;

    let output = component(&json!({
        "url": format!("{}/flaky", server.uri()),
        "retry_config": {
            "enabled": true,
            "max_retries": 2,
            "retry_delay": 10,
            "backoff_type": "fixed"
        }
    }))
    .invoke(json!({}), &context())
    .await
    .unwrap();

    assert_eq!(output["statusCode"], json!(200));
    assert_eq!(output["body"], json!({"recovered": true}));
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(
        requests.len(),
        2,
        "one failed attempt plus one retry reaches the server"
    );
}

// port of tests/unit_tests/core/component/test_http_component.py::test_http_comp_008
// (the Python skip-system variant asserts StatusCode.COMPONENT_TOOL_EXECUTION_ERROR)
#[tokio::test]
async fn a_transport_failure_is_a_coded_workflow_error() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
    let address = listener.local_addr().expect("read the bound address");
    drop(listener);

    let error = component(&json!({"url": format!("http://{address}/x")}))
        .invoke(json!({}), &context())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    let text = error.to_string();
    assert!(text.contains("[102000]"), "unexpected error text: {text}");
    assert!(
        text.contains("HTTP request failed"),
        "unexpected error text: {text}"
    );
}

// port of tests/unit_tests/core/component/test_http_component.py::test_start_http_request_end_in_workflow
#[tokio::test]
async fn a_start_http_end_workflow_resolves_the_response_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/item"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"value": 7})))
        .mount(&server)
        .await;

    let mut workflow = Workflow::new(card("http-workflow"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"base": "${inputs.base}"})),
        )
        .add_workflow_comp(
            "http",
            component(&json!({"url": "{{base}}/item"})),
            ComponentOptions::new().with_inputs_schema(json!({"base": "${start.base}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"value": "${http.body.value}"})),
        )
        .add_connection("start", "http")
        .add_connection("http", "end");

    let output = workflow
        .invoke(json!({"base": server.uri()}), session("http-workflow"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"value": 7}}));
}

// port of tests/unit_tests/core/component/test_http_component.py::test_component_creation
#[test]
fn the_registry_builds_http_request_components() {
    let registry = ComponentRegistry::builtin();

    let created = registry
        .create(
            "http_request",
            json!({"request_params": {"url": "http://example.invalid/x"}}),
        )
        .unwrap();
    assert_eq!(created.component_type(), "http_request");

    // Unknown fields are tolerated and the url is validated at invoke time, not here.
    let without_url = registry
        .create(
            "http_request",
            json!({"request_params": {"bogus_field_only": 1}}),
        )
        .unwrap();
    assert_eq!(without_url.component_type(), "http_request");

    let error = registry
        .create("http_request", json!("not-an-object"))
        .err()
        .expect("a non-object config is rejected");
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(
        error.message().contains("[102001]"),
        "unexpected error text: {}",
        error.message()
    );
}

// The multipart encoder builds the whole form itself and sends it as bytes, so the server
// sees a well-formed `multipart/form-data` body with a generated boundary.
#[tokio::test]
async fn a_multipart_body_reaches_the_server_with_a_boundary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/upload"))
        .and(header_regex(
            "content-type",
            r"^multipart/form-data; boundary=----openjiuwen-[0-9a-f]+$",
        ))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({"stored": true})))
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/upload",
        &json!({
            "method": "post",
            "body": {
                "content_type": "multipart_form",
                "multipart_form": {
                    "note": "hello {{who}}",
                    "upload": {"filename": "a.bin", "content_base64": "AAEC+g=="},
                    "meta": {"n": 1}
                }
            }
        }),
    )
    .invoke(json!({"who": "ada"}), &context())
    .await
    .unwrap();

    assert_eq!(output["statusCode"], json!(201));
    assert_eq!(output["body"], json!({"stored": true}));

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    let boundary = requests[0]
        .headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("multipart/form-data; boundary="))
        .expect("the request carries a multipart content type")
        .to_owned();
    let body = String::from_utf8_lossy(&requests[0].body).into_owned();
    assert!(
        body.starts_with(&format!("--{boundary}\r\n")),
        "unexpected body: {body}"
    );
    assert!(
        body.ends_with(&format!("--{boundary}--\r\n")),
        "the body is closed by the boundary: {body}"
    );
    assert!(body.contains("Content-Disposition: form-data; name=\"note\"\r\n\r\nhello ada\r\n"));
    assert!(body.contains("Content-Disposition: form-data; name=\"meta\"\r\n\r\n{\"n\":1}\r\n"));
    assert!(body.contains(
        "Content-Disposition: form-data; name=\"upload\"; filename=\"a.bin\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    ));
    assert!(
        requests[0]
            .body
            .windows(4)
            .any(|window| window == [0_u8, 1, 2, 250]),
        "the decoded file bytes are sent verbatim"
    );
}

// Digest authentication is challenge-driven: an unauthenticated request, then exactly one
// replay carrying the computed `Authorization` header.
#[tokio::test]
async fn digest_authentication_answers_the_challenge_and_replays_once() {
    let server = MockServer::start().await;
    // The `cnonce` and therefore the `response` are freshly generated per run, so the
    // replay is matched on the shape of the header rather than a fixed digest.
    let expected_authorization = concat!(
        "^Digest username=\"mufasa\", realm=\"testrealm@host\\.com\", ",
        r#"nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", uri="/protected", "#,
        r#"qop=auth, nc=00000001, cnonce="[0-9a-f]{32}", "#,
        r#"response="[0-9a-f]{32}", algorithm=MD5, "#,
        r#"opaque="5ccc069c403ebaf9f0171e9517f40e41"$"#,
    );
    Mock::given(method("GET"))
        .and(path("/protected"))
        .respond_with(ResponseTemplate::new(401).insert_header(
            "www-authenticate",
            r#"Digest realm="testrealm@host.com", qop="auth,auth-int", "#.to_owned()
                + r#"nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", "#
                + r#"opaque="5ccc069c403ebaf9f0171e9517f40e41", algorithm=MD5"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/protected"))
        .and(header_regex("authorization", expected_authorization))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"secret": 42})))
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/protected",
        &json!({
            "authentication": {"type": "digest", "username": "mufasa", "password": "pw"}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(output["statusCode"], json!(200));
    assert_eq!(output["body"], json!({"secret": 42}));

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 2, "the challenge is answered exactly once");
    assert!(
        requests[0].headers.get("authorization").is_none(),
        "the first attempt is unauthenticated"
    );
    let authorization = requests[1]
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .expect("the replay carries the digest header");
    assert!(authorization.starts_with("Digest username=\"mufasa\""));
}

// A `401` this component cannot answer (no digest challenge) stays ordinary data.
#[tokio::test]
async fn a_401_without_a_digest_challenge_is_data() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/protected"))
        .respond_with(
            ResponseTemplate::new(401)
                .insert_header("www-authenticate", "Basic realm=\"x\"")
                .set_body_json(json!({"error": "unauthorized"})),
        )
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/protected",
        &json!({
            "authentication": {"type": "digest", "username": "u", "password": "p"},
            "response_handling": {"throw_on_http_error": false}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(output["statusCode"], json!(401));
    assert_eq!(output["ok"], json!(false));
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
}

/// The lower-case hexadecimal SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

// AWS SigV4 is computed in pre-flight, so every header the signature covers is on the wire.
#[tokio::test]
async fn aws_sigv4_signs_the_request_before_sending() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/signed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"signed": true})))
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/signed",
        &json!({
            "method": "post",
            "body": {"content_type": "json", "json_data": {"a": 1}},
            "authentication": {
                "type": "aws",
                "username": "AKIDEXAMPLE",
                "password": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
            }
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(output["statusCode"], json!(200));

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    let authorization = requests[0]
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .expect("the request is signed");
    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
        "unexpected authorization: {authorization}"
    );
    assert!(
        authorization.contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date"),
        "unexpected authorization: {authorization}"
    );
    assert!(
        authorization.contains(", Signature="),
        "unexpected authorization: {authorization}"
    );
    let payload_hash = requests[0]
        .headers
        .get("x-amz-content-sha256")
        .and_then(|value| value.to_str().ok())
        .expect("the payload hash header is sent");
    assert_eq!(payload_hash, sha256_hex(&requests[0].body));
    assert!(
        requests[0].headers.get("x-amz-date").is_some(),
        "the timestamp header is sent"
    );
}

// `advanced_options.max_body_length` rejects an oversized body before anything is sent.
#[tokio::test]
async fn a_body_over_the_configured_limit_is_rejected_before_sending() {
    let server = MockServer::start().await;

    let error = component_at(
        &server,
        "/upload",
        &json!({
            "method": "post",
            "body": {"content_type": "text", "text_data": "0123456789"},
            "advanced_options": {"max_body_length": 4}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    let text = error.to_string();
    assert!(text.contains("[102001]"), "unexpected error text: {text}");
    assert!(text.contains("10 bytes"), "unexpected error text: {text}");
    assert!(
        text.contains("max_body_length of 4"),
        "unexpected error text: {text}"
    );
    assert_eq!(
        server
            .received_requests()
            .await
            .expect("recorded requests")
            .len(),
        0,
        "the request never reaches the server"
    );
}

// The limiter is per component instance and honours the run's cancellation token, so a
// cancelled run stops waiting instead of sleeping through the window.
#[tokio::test]
async fn the_rate_limiter_delays_sends_and_honours_cancellation() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/limited"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let http = component(&json!({
        "url": format!("{}/limited", server.uri()),
        "rate_limit_config": {"enabled": true, "requests_per_unit": 1, "unit": "second"}
    }));
    let cancellation = CancellationToken::new();
    let ctx = context().with_cancellation(cancellation.clone());

    // The first send is immediate and consumes the window.
    let output = http.invoke(json!({}), &ctx).await.unwrap();
    assert_eq!(output["statusCode"], json!(200));

    // The second send would wait a second; a cancelled run stops immediately instead.
    cancellation.cancel();
    let error = http.invoke(json!({}), &ctx).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(
        server
            .received_requests()
            .await
            .expect("recorded requests")
            .len(),
        1,
        "the cancelled send never reaches the server"
    );
}

// A generous window keeps ordinary traffic unthrottled.
#[tokio::test]
async fn a_generous_rate_limit_still_serves_every_request() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/burst"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let http = component(&json!({
        "url": format!("{}/burst", server.uri()),
        "rate_limit_config": {"enabled": true, "requests_per_unit": 100, "unit": "minute"}
    }));
    for _ in 0..3 {
        let output = http.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["statusCode"], json!(200));
    }
    assert_eq!(
        server
            .received_requests()
            .await
            .expect("recorded requests")
            .len(),
        3
    );
}

// `buffer` is lossless (byte array + length); `binary` keeps Python's base64 string.
#[tokio::test]
async fn buffer_and_binary_response_formats_differ() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/raw"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(vec![0_u8, 1, 2, 250]),
        )
        .mount(&server)
        .await;

    let buffered = component_at(
        &server,
        "/raw",
        &json!({"response_handling": {"response_format": "buffer"}}),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(buffered["body"]["type"], json!("Buffer"));
    assert_eq!(buffered["body"]["data"], json!([0, 1, 2, 250]));
    assert_eq!(buffered["body"]["length"], json!(4));

    let binary = component_at(
        &server,
        "/raw",
        &json!({"response_handling": {"response_format": "binary"}}),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(binary["body"], json!("AAEC+g=="));

    // `response_data_property` works on the buffer object's own keys.
    let length_only = component_at(
        &server,
        "/raw",
        &json!({"response_handling": {"response_format": "buffer", "response_data_property": "length"}}),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(length_only["body"], json!(4));
}

// The form body stays URL-encoded (the multipart encoder is opt-in through content_type).
#[tokio::test]
async fn a_form_body_is_url_encoded_and_the_multipart_body_is_not() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/form"))
        .and(header_regex(
            "content-type",
            "^application/x-www-form-urlencoded$",
        ))
        .and(body_string_contains("a=b+c&d=%C3%A9"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let output = component_at(
        &server,
        "/form",
        &json!({
            "method": "post",
            "body": {"content_type": "form", "form_data": {"a": "b c", "d": "é"}}
        }),
    )
    .invoke(json!({}), &context())
    .await
    .unwrap();
    assert_eq!(output["statusCode"], json!(200));
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/x-www-form-urlencoded")
    );
    assert_eq!(requests[0].body, b"a=b+c&d=%C3%A9".to_vec());
}
