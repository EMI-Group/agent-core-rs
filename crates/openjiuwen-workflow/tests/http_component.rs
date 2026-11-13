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
//! involved. Only HTTP failures are data (`statusCode`, `ok = false`); transport failures
//! become the coded `[102000]` workflow error.
//!
//! Ports of `tests/unit_tests/core/component/test_http_component.py` and
//! `…/test_http_component_comprehensive.py`, rewritten against `openjiuwen_workflow`'s
//! `Component`/`Workflow` API. Everything is offline and deterministic: the mock server
//! binds loopback, URLs come from its `uri()`, and no assertion depends on timing.
mod common;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::wiremock::matchers::{body_json, method, path, query_param};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use serde_json::{json, Value};

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
async fn a_server_error_is_data_not_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"error": "boom"})))
        .mount(&server)
        .await;

    let output = component_at(&server, "/broken", &json!({}))
        .invoke(json!({}), &context())
        .await
        .unwrap();

    assert_eq!(output["statusCode"], json!(500));
    assert_eq!(output["ok"], json!(false));
    assert_eq!(output["body"], json!({"error": "boom"}));
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
        &json!({"response_handling": {"response_mode": "on-success"}}),
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
        &json!({"response_handling": {"response_mode": "on-error"}}),
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
