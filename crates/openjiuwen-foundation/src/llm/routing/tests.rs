//! Offline tests of the routing layer: schema, compiler, and the routing model.
//!
//! Every HTTP call goes to a `wiremock` server on loopback, so no test needs the
//! network, a real credential, or an environment variable. The per-route
//! `ModelClientConfig`s are built explicitly (provider `openai`, the mock server as
//! `api_base`, a dummy key, `max_retries = 0`) so a fallback can be observed as an
//! exact request count per endpoint.
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_common::security::SecretString;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::compiler::{
    compile_model_selection, compile_route, merge_request_defaults, resolve_model_group_routes,
    ROUTING_PLAN_KEY,
};
use super::model::{is_retryable_failure, RoutingModel};
use super::schema::{
    ModelSelection, ModelSelectionWire, ResolvedModel, ResolvedModelGroup, ResolvedRoute,
    ResolvedRoutingConfig, RoutingStrategy, SelectionKind,
};
use crate::llm::config::ModelRequestConfig;
use crate::llm::model::{Model, StreamChunk};
use crate::prompt::BaseMessage;

// --- fixtures ---------------------------------------------------------------

/// A resolved OpenAI-compatible model pointed at a mock server.
fn resolved_model(model_id: &str, model_name: &str, api_base: &str) -> ResolvedModel {
    ResolvedModel {
        model_id: model_id.to_owned(),
        model_name: model_name.to_owned(),
        provider: "openai".to_owned(),
        api_key: Some(SecretString::new("test-key")),
        api_base: Some(format!("{api_base}/v1")),
        client_options: json!({"max_retries": 0}),
        ..ResolvedModel::default()
    }
}

/// One route wrapping a resolved model.
fn route(route_id: &str, model: ResolvedModel) -> ResolvedRoute {
    ResolvedRoute {
        route_id: route_id.to_owned(),
        model,
        ..ResolvedRoute::default()
    }
}

/// A group with the given routes and no strategy.
fn group(routes: Vec<ResolvedRoute>) -> ResolvedModelGroup {
    ResolvedModelGroup {
        group_id: "mg-1".to_owned(),
        routes,
        ..ResolvedModelGroup::default()
    }
}

/// A canned non-streaming completion.
fn completion(model: &str, content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    })
}

/// Starts a server that answers every chat-completions request with `template`.
async fn server_with(template: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(template)
        .mount(&server)
        .await;
    server
}

/// Starts a server that answers one complete chat completion with `content`.
async fn ok_server(content: &str) -> MockServer {
    server_with(ResponseTemplate::new(200).set_body_json(completion("gpt-4o", content))).await
}

/// How many requests a server received.
async fn request_count(server: &MockServer) -> usize {
    server.received_requests().await.unwrap_or_default().len()
}

/// The JSON body of the first request a server received.
async fn first_body(server: &MockServer) -> serde_json::Value {
    let requests = server.received_requests().await.unwrap_or_default();
    let body = requests
        .first()
        .expect("the server received a request")
        .body
        .clone();
    serde_json::from_slice(&body).expect("the request body is JSON")
}

/// An SSE response body carrying two content deltas and the terminator.
fn sse_template(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body.to_owned())
}

/// A valid SSE completion.
const SSE_COMPLETION: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
    "data: [DONE]\n\n"
);

// --- schema -----------------------------------------------------------------

#[test]
fn resolved_route_round_trips_with_unknown_fields_and_defaults() {
    let payload = json!({
        "route_id": "r-1",
        "priority": 2,
        "tag": "east",
        "vendor_extension": {"nested": true},
        "model": {
            "model_id": "m-1",
            "model_name": "gpt-4o",
            "provider": "openai",
            "api_key": "sk-test",
            "unknown_model_field": 1,
            "request_defaults": {"temperature": 0.1}
        }
    });
    let route: ResolvedRoute = serde_json::from_value(payload).expect("the route parses");
    assert_eq!(route.route_id, "r-1");
    assert!(route.enabled, "enabled defaults to true");
    assert_eq!(route.priority, Some(2));
    assert_eq!(route.tag.as_deref(), Some("east"));
    assert_eq!(
        route
            .model
            .api_key
            .as_ref()
            .map(SecretString::expose_secret),
        Some("sk-test")
    );
    assert_eq!(route.model.request_defaults, json!({"temperature": 0.1}));

    let encoded = serde_json::to_value(&route).expect("the route serializes");
    let decoded: ResolvedRoute = serde_json::from_value(encoded).expect("the route reparses");
    assert_eq!(decoded, route, "the route round-trips");
}

#[test]
fn blank_credentials_and_bases_deserialize_as_unset() {
    let model: ResolvedModel = serde_json::from_value(json!({
        "model_id": "m-1",
        "api_key": "",
        "api_base": "   "
    }))
    .expect("the model parses");
    assert!(model.api_key.is_none());
    assert!(model.api_base.is_none());
    assert_eq!(model.client_options, serde_json::Value::Null);
}

#[test]
fn model_group_accepts_the_python_nested_routing_and_id_alias() {
    let payload = json!({
        "model_group_id": "mg-python",
        "routes": [{
            "route_id": "r-1",
            "model": {"model_id": "m-1", "model_name": "gpt-4o", "provider": "openai"}
        }],
        "request_config": {"temperature": 0.3},
        "routing": {
            "strategy": "ordered_failover",
            "num_retries": 2,
            "strategy_kwargs": {"tag": "east"},
            "timeout_seconds": 12.5
        }
    });
    let group: ResolvedModelGroup = serde_json::from_value(payload).expect("the group parses");
    assert_eq!(group.group_id, "mg-python");
    assert_eq!(group.strategy.as_deref(), Some("ordered_failover"));
    assert_eq!(group.num_retries, Some(2));
    assert_eq!(group.timeout_seconds, Some(12.5));
    assert_eq!(group.request_config, json!({"temperature": 0.3}));
    assert_eq!(group.routing().strategy, RoutingStrategy::OrderedFailover);
    assert_eq!(
        group.routing().num_retries,
        2,
        "an explicit retry budget wins"
    );
    assert_eq!(group.routing().strategy_kwargs, json!({"tag": "east"}));
}

#[test]
fn flat_group_fields_win_over_the_nested_python_routing_object() {
    let group: ResolvedModelGroup = serde_json::from_value(json!({
        "group_id": "mg-1",
        "routes": [],
        "strategy": "tag-filtered",
        "num_retries": 0,
        "routing": {"strategy": "ordered-failover", "num_retries": 5}
    }))
    .expect("the group parses");
    assert_eq!(group.strategy.as_deref(), Some("tag-filtered"));
    assert_eq!(group.num_retries, Some(0));
}

#[test]
fn routing_strategy_names_parse_leniently() {
    assert_eq!(
        RoutingStrategy::parse("  Ordered_Failover "),
        RoutingStrategy::OrderedFailover
    );
    assert_eq!(
        RoutingStrategy::parse("TAG-FILTERED"),
        RoutingStrategy::TagFiltered
    );
    let custom = RoutingStrategy::parse("simple-shuffle");
    assert_eq!(custom, RoutingStrategy::Custom("simple-shuffle".to_owned()));
    assert!(!custom.is_known());
    assert_eq!(custom.as_str(), "simple-shuffle");
    assert_eq!(RoutingStrategy::default(), RoutingStrategy::OrderedFailover);

    let encoded = serde_json::to_value(RoutingStrategy::TagFiltered).expect("a strategy");
    assert_eq!(encoded, json!("tag-filtered"));
    let decoded: RoutingStrategy = serde_json::from_value(json!("simple_shuffle")).expect("a name");
    assert_eq!(
        decoded,
        RoutingStrategy::Custom("simple-shuffle".to_owned())
    );
}

#[test]
fn model_selection_wire_discriminates_like_the_python_compiler() {
    let model = ModelSelectionWire::from_value(json!({
        "model_id": "m-1",
        "model_name": "gpt-4o",
        "provider": "openai"
    }))
    .expect("a model payload");
    assert_eq!(model.kind(), SelectionKind::Model);
    assert!(model.as_model().is_some());
    assert!(model.as_group().is_none());

    let group = ModelSelectionWire::from_value(json!({"group_id": "mg-1", "routes": []}))
        .expect("a group payload");
    assert_eq!(group.kind(), SelectionKind::ModelGroup);
    assert!(group.as_group().is_some());
    assert!(group.as_model().is_none());

    let unresolved = ModelSelectionWire::from_value(json!({"type": "model_group", "id": "mg-1"}))
        .expect_err("a persisted selection has to be resolved first");
    assert_eq!(unresolved.kind(), ErrorKind::Validation);
    assert!(unresolved.message().contains("is not resolved"));

    let not_an_object =
        ModelSelectionWire::from_value(json!(["model"])).expect_err("a payload is an object");
    assert_eq!(not_an_object.kind(), ErrorKind::Validation);
    assert!(not_an_object.message().contains("must be a JSON object"));

    let decoded: ModelSelectionWire = serde_json::from_str(
        r#"{"group_id":"mg-1","routes":[{"route_id":"r-1",
             "model":{"model_id":"m-1","model_name":"gpt-4o","provider":"openai"}}]}"#,
    )
    .expect("the wire type deserializes from JSON");
    assert_eq!(decoded.kind(), SelectionKind::ModelGroup);
}

#[test]
fn model_selection_reference_round_trips() {
    let selection = ModelSelection {
        kind: SelectionKind::ModelGroup,
        id: "mg-1".to_owned(),
    };
    let encoded = serde_json::to_value(&selection).expect("the reference serializes");
    assert_eq!(encoded, json!({"type": "model_group", "id": "mg-1"}));
    let decoded: ModelSelection = serde_json::from_value(encoded).expect("the reference reparses");
    assert_eq!(decoded, selection);
}

#[test]
fn resolved_routing_config_defaults_match_the_python_compiler() {
    let config = ResolvedRoutingConfig::default();
    assert_eq!(config.strategy, RoutingStrategy::OrderedFailover);
    assert_eq!(config.num_retries, 0);
    assert_eq!(config.strategy_kwargs, json!({}));
    assert_eq!(config.timeout_seconds, None);
}

// --- request-default merging ------------------------------------------------

#[test]
fn request_defaults_merge_route_then_group_then_overrides() {
    let route_defaults = json!({
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 128,
        "vendor": {"a": 1, "b": {"x": 1}}
    });
    let group_config = json!({
        "temperature": 0.5,
        "stop": ["END"],
        "vendor": {"b": {"y": 2}, "c": 3}
    });
    let overrides = json!({"max_tokens": 512, "vendor": {"b": {"x": 9}}});

    let merged = merge_request_defaults(&[&route_defaults, &group_config, &overrides]);
    assert_eq!(
        merged["temperature"],
        json!(0.5),
        "the group overrides the route"
    );
    assert_eq!(
        merged["top_p"],
        json!(0.9),
        "an untouched route value survives"
    );
    assert_eq!(merged["max_tokens"], json!(512), "the route override wins");
    assert_eq!(merged["stop"], json!(["END"]));
    assert_eq!(
        merged["vendor"],
        json!({"a": 1, "b": {"x": 9, "y": 2}, "c": 3}),
        "nested objects merge recursively"
    );
}

#[test]
fn a_null_never_unsets_an_earlier_value() {
    let merged = merge_request_defaults(&[
        &json!({"temperature": 0.7, "vendor": {"a": 1}}),
        &json!({"temperature": null, "vendor": {"a": null}}),
        &json!({"vendor": null}),
    ]);
    assert_eq!(merged, json!({"temperature": 0.7, "vendor": {"a": 1}}));
}

#[test]
fn client_and_compiler_owned_keys_are_stripped_from_request_defaults() {
    let source = json!({
        "model": "gpt-4",
        "model_name": "gpt-4",
        "api_key": "sk-leak",
        "api_base": "https://api.example",
        "provider": "openai",
        "client_provider": "openai",
        "client_id": "client-1",
        "stream": true,
        "verify_ssl": false,
        "timeout": 30,
        "stream_first_chunk_timeout": 5,
        "stream_idle_timeout": 5,
        "max_retries": 2,
        "headers": {"x-trace": "1"},
        "deployment": "prod",
        "messages": [],
        "endpoint_profile": "deepseek",
        "temperature": 0.2
    });
    assert_eq!(
        merge_request_defaults(&[&source]),
        json!({"temperature": 0.2}),
        "only request-owned keys survive"
    );
}

#[test]
fn compile_route_builds_the_client_and_the_merged_request() {
    let mut model = resolved_model("m-1", "gpt-4o", "https://api.example");
    model.endpoint_profile = Some("deepseek".to_owned());
    model.request_defaults = json!({
        "temperature": 0.1,
        "max_tokens": 64,
        "api_key": "sk-leak",
        "reasoning": {"mode": "enabled"},
        "extra_body": {"x": 1}
    });
    model.client_options = json!({
        "max_retries": 2,
        "timeout": 30,
        "verify_ssl": false,
        "custom_knob": true
    });
    let route = ResolvedRoute {
        request_overrides: json!({"temperature": 0.9}),
        timeout: Some(12.5),
        ..route("r-1", model)
    };

    let (client, request) =
        compile_route(&route, &json!({"top_p": 0.4})).expect("the route compiles");
    assert_eq!(client.provider, "openai");
    assert_eq!(client.api_base.as_deref(), Some("https://api.example/v1"));
    assert!(!client.verify_ssl);
    assert_eq!(client.max_retries, 2);
    assert_eq!(
        client.timeout,
        Some(Duration::from_secs_f64(12.5)),
        "the route timeout overrides the client option"
    );
    assert_eq!(
        client.extra,
        json!({"endpoint_profile": "deepseek", "custom_knob": true})
    );
    assert_eq!(request.model_name, "gpt-4o");
    assert_eq!(request.temperature, Some(0.9));
    assert_eq!(request.top_p, Some(0.4));
    assert_eq!(request.max_tokens, Some(64));
    assert_eq!(
        request.extra,
        json!({"reasoning": {"mode": "enabled"}, "extra_body": {"x": 1}}),
        "vendor knobs survive in extra"
    );
}

// --- compilation ------------------------------------------------------------

#[test]
fn single_model_compiles_to_its_own_configs() {
    let wire =
        ModelSelectionWire::from(resolved_model("m-1", "gpt-4o-mini", "https://api.example"));
    let compiled = compile_model_selection(&wire).expect("the model compiles");
    assert_eq!(compiled.selected_type, SelectionKind::Model);
    assert_eq!(compiled.selected_id, "m-1");
    assert_eq!(compiled.model_client_config.provider, "openai");
    assert_eq!(
        compiled
            .model_client_config
            .api_key
            .as_ref()
            .map(SecretString::expose_secret),
        Some("test-key")
    );
    assert_eq!(compiled.model_request_config.model_name, "gpt-4o-mini");
    assert_eq!(
        compiled.model_request_config.extra,
        serde_json::Value::Null,
        "no vendor knob means an empty extra"
    );
}

#[test]
fn blank_model_identity_is_rejected() {
    let mut model = resolved_model("m-1", "   ", "https://api.example");
    let error = compile_model_selection(&ModelSelectionWire::from(model.clone()))
        .expect_err("a blank model name is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("empty model_name"));

    model.model_name = "gpt-4o".to_owned();
    model.provider = "  ".to_owned();
    let error = compile_model_selection(&ModelSelectionWire::from(model))
        .expect_err("a blank provider is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("empty provider"));
}

#[test]
fn a_group_without_a_strategy_collapses_to_its_first_enabled_route() {
    let mut disabled = route("r-0", resolved_model("m-0", "gpt-3.5", "https://a.example"));
    disabled.enabled = false;
    let mut group = group(vec![
        disabled,
        route(
            "r-1",
            resolved_model("m-1", "gpt-4o-mini", "https://b.example"),
        ),
        route("r-2", resolved_model("m-2", "gpt-4o", "https://c.example")),
    ]);
    group.request_config = json!({"temperature": 0.25});

    let compiled = compile_model_selection(&ModelSelectionWire::from(group))
        .expect("the group compiles to its first enabled route");
    assert_eq!(compiled.selected_type, SelectionKind::ModelGroup);
    assert_eq!(compiled.selected_id, "mg-1");
    assert_eq!(compiled.model_client_config.provider, "openai");
    assert_eq!(
        compiled.model_client_config.api_base.as_deref(),
        Some("https://b.example/v1")
    );
    assert_eq!(compiled.model_request_config.model_name, "gpt-4o-mini");
    assert_eq!(compiled.model_request_config.temperature, Some(0.25));
}

#[test]
fn an_empty_strategy_means_no_routing() {
    let mut group = group(vec![route(
        "r-1",
        resolved_model("m-1", "gpt-4o-mini", "https://a.example"),
    )]);
    group.strategy = Some("   ".to_owned());
    let compiled = compile_model_selection(&ModelSelectionWire::from(group))
        .expect("a blank strategy is no strategy");
    assert_eq!(compiled.model_client_config.provider, "openai");
    assert_eq!(compiled.model_client_config.extra, json!({}));
}

#[test]
fn a_routed_group_records_the_plan_and_defaults_num_retries() {
    let mut disabled = route("r-0", resolved_model("m-0", "gpt-3.5", "https://a.example"));
    disabled.enabled = false;
    let mut group = group(vec![
        disabled,
        route(
            "r-1",
            resolved_model("m-1", "gpt-4o-mini", "https://b.example"),
        ),
        route("r-2", resolved_model("m-2", "gpt-4o", "https://c.example")),
        route("r-3", resolved_model("m-3", "gpt-4.1", "https://d.example")),
    ]);
    group.strategy = Some("ordered-failover".to_owned());
    group.strategy_kwargs = json!({"tag": "east"});

    let compiled = compile_model_selection(&ModelSelectionWire::from(group.clone()))
        .expect("the group routes");
    assert_eq!(
        compiled.model_client_config.provider, "intelli_router",
        "router mode keeps the routing identity"
    );
    let plan = &compiled.model_client_config.extra[ROUTING_PLAN_KEY];
    assert_eq!(plan["strategy"], json!("ordered-failover"));
    assert_eq!(
        plan["num_retries"],
        json!(2),
        "three enabled routes mean two additional attempts"
    );
    assert_eq!(plan["route_ids"], json!(["r-1", "r-2", "r-3"]));
    assert_eq!(plan["strategy_kwargs"], json!({"tag": "east"}));
    assert_eq!(compiled.model_request_config.model_name, "mg-1");

    group.num_retries = Some(0);
    let compiled = compile_model_selection(&ModelSelectionWire::from(group))
        .expect("an explicit retry budget compiles");
    assert_eq!(
        compiled.model_client_config.extra[ROUTING_PLAN_KEY]["num_retries"],
        json!(0)
    );
}

#[test]
fn a_group_without_enabled_routes_is_rejected() {
    let mut disabled = route("r-1", resolved_model("m-1", "gpt-4o", "https://a.example"));
    disabled.enabled = false;
    let error = compile_model_selection(&ModelSelectionWire::from(group(vec![disabled])))
        .expect_err("a group needs one enabled route");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("has no enabled routes"));
}

#[test]
fn an_unknown_strategy_is_a_validation_error() {
    let mut group = group(vec![route(
        "r-1",
        resolved_model("m-1", "gpt-4o", "https://a.example"),
    )]);
    group.strategy = Some("simple-shuffle".to_owned());
    let error = compile_model_selection(&ModelSelectionWire::from(group))
        .expect_err("the strategy is not implemented");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message().to_string();
    assert!(message.contains("unknown routing strategy"), "{message}");
    assert!(message.contains("ordered-failover"), "{message}");
    assert!(message.contains("tag-filtered"), "{message}");
}

#[test]
fn a_malformed_tag_argument_is_a_validation_error() {
    let mut group = group(vec![route(
        "r-1",
        resolved_model("m-1", "gpt-4o", "https://a.example"),
    )]);
    group.strategy = Some("tag-filtered".to_owned());
    group.strategy_kwargs = json!({"tag": 5});
    let error = compile_model_selection(&ModelSelectionWire::from(group))
        .expect_err("a numeric tag is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error
        .message()
        .contains("must be a string or an array of strings"));
}

#[test]
fn router_mode_keeps_the_minimum_context_window() {
    let mut first = route("r-1", resolved_model("m-1", "gpt-4o", "https://a.example"));
    first.model.request_defaults = json!({"context_window": 128_000});
    let mut second = route("r-2", resolved_model("m-2", "gpt-4o", "https://b.example"));
    second.request_overrides = json!({"context_window": 32000});
    let mut group = group(vec![first, second]);
    group.strategy = Some("ordered-failover".to_owned());

    let compiled =
        compile_model_selection(&ModelSelectionWire::from(group)).expect("the group compiles");
    assert_eq!(
        compiled.model_request_config.extra,
        json!({"context_window": 32000})
    );
}

// --- route ordering ---------------------------------------------------------

#[test]
fn priorities_order_routes_and_their_absence_keeps_list_order() {
    let first = ResolvedRoute {
        priority: Some(1),
        ..route("a", resolved_model("m-a", "gpt-4o", "https://a.example"))
    };
    let second = route("b", resolved_model("m-b", "gpt-4o", "https://b.example"));
    let third = ResolvedRoute {
        priority: Some(5),
        ..route("c", resolved_model("m-c", "gpt-4o", "https://c.example"))
    };
    let fourth = ResolvedRoute {
        priority: Some(1),
        ..route("d", resolved_model("m-d", "gpt-4o", "https://d.example"))
    };
    let mut disabled = ResolvedRoute {
        priority: Some(9),
        ..route("e", resolved_model("m-e", "gpt-4o", "https://e.example"))
    };
    disabled.enabled = false;
    let ordered_group = group(vec![first, second, third, fourth, disabled]);

    let resolved = resolve_model_group_routes(&ordered_group);
    let ordered: Vec<&str> = resolved
        .iter()
        .map(|route| route.route_id.as_str())
        .collect();
    assert_eq!(
        ordered,
        ["c", "a", "d", "b"],
        "descending priority, stable on ties, disabled routes dropped"
    );

    let plain = group(vec![
        route("a", resolved_model("m-a", "gpt-4o", "https://a.example")),
        route("b", resolved_model("m-b", "gpt-4o", "https://b.example")),
        route("c", resolved_model("m-c", "gpt-4o", "https://c.example")),
    ]);
    let untouched_routes = resolve_model_group_routes(&plain);
    let untouched: Vec<&str> = untouched_routes
        .iter()
        .map(|route| route.route_id.as_str())
        .collect();
    assert_eq!(untouched, ["a", "b", "c"], "no priorities means list order");
}

// --- the routing model ------------------------------------------------------

#[test]
fn retryable_failures_are_transient_only() {
    assert!(is_retryable_failure(&Error::transport("connection reset")));
    assert!(is_retryable_failure(&Error::timeout("deadline")));
    assert!(is_retryable_failure(&Error::llm("server error")));
    assert!(is_retryable_failure(&Error::external(
        std::io::Error::other("io")
    )));
    assert!(!is_retryable_failure(&Error::validation("bad request")));
    assert!(!is_retryable_failure(&Error::config("no credential")));
    assert!(!is_retryable_failure(&Error::not_found("no model")));
    assert!(!is_retryable_failure(&Error::unsupported("no transport")));
}

#[tokio::test]
async fn routing_model_reports_the_routing_identity() {
    let group = group(vec![
        route(
            "r-1",
            resolved_model("m-1", "gpt-4o-mini", "http://127.0.0.1:9"),
        ),
        route("r-2", resolved_model("m-2", "gpt-4o", "http://127.0.0.1:9")),
    ]);
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds without a request");
    assert_eq!(routing.provider(), "intelli_router");
    assert_eq!(routing.model_name(), "mg-1");
    assert_eq!(routing.model_group_id(), "mg-1");
    assert_eq!(routing.route_count(), 2);
    assert_eq!(routing.route_ids(), vec!["r-1", "r-2"]);
    assert_eq!(routing.routing_config().num_retries, 1);
    assert_eq!(routing.routes()[1].model.model_id, "m-2");

    let renamed = routing.with_provider("my-router");
    assert_eq!(renamed.provider(), "my-router");
}

#[test]
fn with_models_requires_one_model_per_enabled_route() {
    let group = group(vec![route(
        "r-1",
        resolved_model("m-1", "gpt-4o", "http://127.0.0.1:9"),
    )]);
    let error = RoutingModel::with_models(&group, Vec::new())
        .expect_err("no model was supplied for the enabled route");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("0 model(s) were supplied"));
}

#[tokio::test]
async fn invoke_fails_over_to_the_next_route() {
    let first = server_with(ResponseTemplate::new(500).set_body_string("upstream exploded")).await;
    let second = ok_server("from the second route").await;
    let group = group(vec![
        route("r-1", resolved_model("m-1", "gpt-4o-mini", &first.uri())),
        route("r-2", resolved_model("m-2", "gpt-4o", &second.uri())),
    ]);
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let response = routing
        .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new(""))
        .await
        .expect("the second route answers");
    assert_eq!(
        response.message().expect("one choice").text(),
        "from the second route"
    );
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(request_count(&first).await, 1);
    assert_eq!(request_count(&second).await, 1);
    assert_eq!(
        first_body(&first).await["model"],
        json!("gpt-4o-mini"),
        "each attempt uses its own route model name"
    );
    assert_eq!(first_body(&second).await["model"], json!("gpt-4o"));
}

#[tokio::test]
async fn auth_failures_are_not_retried_on_another_route() {
    let first = server_with(ResponseTemplate::new(401).set_body_string("bad credential")).await;
    let second = ok_server("must not be used").await;
    let group = group(vec![
        route("r-1", resolved_model("m-1", "gpt-4o-mini", &first.uri())),
        route("r-2", resolved_model("m-2", "gpt-4o", &second.uri())),
    ]);
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let error = routing
        .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new(""))
        .await
        .expect_err("an authentication failure is terminal");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(request_count(&first).await, 1);
    assert_eq!(
        request_count(&second).await,
        0,
        "a 401 must not be retried on another route"
    );
}

#[tokio::test]
async fn zero_retries_means_a_single_attempt() {
    let first = server_with(ResponseTemplate::new(500).set_body_string("upstream exploded")).await;
    let second = ok_server("never reached").await;
    let mut group = group(vec![
        route("r-1", resolved_model("m-1", "gpt-4o-mini", &first.uri())),
        route("r-2", resolved_model("m-2", "gpt-4o", &second.uri())),
    ]);
    group.strategy = Some("ordered-failover".to_owned());
    group.num_retries = Some(0);
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let error = routing
        .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new(""))
        .await
        .expect_err("the single attempt failed");
    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.message().to_string();
    assert!(message.contains("all routes failed"), "{message}");
    assert!(message.contains("r-1"), "{message}");
    assert!(!message.contains("r-2"), "no second attempt: {message}");
    assert_eq!(request_count(&second).await, 0);
}

#[tokio::test]
async fn tag_filtered_strategy_pins_the_matching_route() {
    let east = ok_server("east answer").await;
    let west = ok_server("west answer").await;
    let untagged = ok_server("untagged answer").await;
    let tagged = |id: &str, model_id: &str, base: &str, tag: Option<&str>| {
        let mut model = resolved_model(model_id, "gpt-4o", base);
        model.fallback_tag = tag.map(str::to_owned);
        route(id, model)
    };
    let mut group = group(vec![
        tagged("r-east", "m-east", &east.uri(), Some("east")),
        tagged("r-west", "m-west", &west.uri(), Some("west")),
        tagged("r-untagged", "m-untagged", &untagged.uri(), None),
    ]);
    group.strategy = Some("tag-filtered".to_owned());
    group.strategy_kwargs = json!({"tag": ["west"]});
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let response = routing
        .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new(""))
        .await
        .expect("the tagged route answers");
    assert_eq!(
        response.message().expect("one choice").text(),
        "west answer"
    );
    assert_eq!(request_count(&east).await, 0);
    assert_eq!(request_count(&west).await, 1);
    assert_eq!(request_count(&untagged).await, 0);
}

#[tokio::test]
async fn tag_filter_without_a_matching_route_is_a_validation_error() {
    let east = ok_server("east answer").await;
    let mut model = resolved_model("m-east", "gpt-4o", &east.uri());
    model.fallback_tag = Some("east".to_owned());
    let mut group = group(vec![route("r-east", model)]);
    group.strategy = Some("tag-filtered".to_owned());
    group.strategy_kwargs = json!({"tag": "nowhere"});
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let error = routing
        .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new(""))
        .await
        .expect_err("no candidate carries the tag");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message().to_string();
    assert!(message.contains("no enabled route tagged"), "{message}");
    assert!(
        message.contains("east"),
        "the available tags are listed: {message}"
    );
    assert_eq!(request_count(&east).await, 0);
}

#[tokio::test]
async fn stream_fails_over_before_the_first_chunk() {
    let first = server_with(ResponseTemplate::new(500).set_body_string("upstream exploded")).await;
    let second = server_with(sse_template(SSE_COMPLETION)).await;
    let group = group(vec![
        route("r-1", resolved_model("m-1", "gpt-4o-mini", &first.uri())),
        route("r-2", resolved_model("m-2", "gpt-4o", &second.uri())),
    ]);
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let mut stream = routing.stream(
        vec![BaseMessage::user("hello")],
        ModelRequestConfig::new(""),
    );
    let mut content = String::new();
    let mut done = false;
    while let Some(item) = stream.next().await {
        match item.expect("every chunk decodes") {
            StreamChunk::Delta {
                content: Some(piece),
                ..
            } => content.push_str(&piece),
            StreamChunk::Done => done = true,
            _ => {}
        }
    }
    assert_eq!(content, "hello world");
    assert!(done, "the stream terminates with Done");
    assert_eq!(request_count(&first).await, 1);
    assert_eq!(request_count(&second).await, 1);
}

#[tokio::test]
async fn stream_errors_after_the_first_chunk_are_propagated() {
    let mid_stream_failure = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"upstream exploded\",\"type\":\"server_error\"}}\n\n"
    );
    let first = server_with(sse_template(mid_stream_failure)).await;
    let second = ok_server("must not be used").await;
    let group = group(vec![
        route("r-1", resolved_model("m-1", "gpt-4o-mini", &first.uri())),
        route("r-2", resolved_model("m-2", "gpt-4o", &second.uri())),
    ]);
    let routing = RoutingModel::from_group(&group)
        .await
        .expect("the routing model builds");

    let mut stream = routing.stream(
        vec![BaseMessage::user("hello")],
        ModelRequestConfig::new(""),
    );
    let mut content = String::new();
    let mut failure = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamChunk::Delta {
                content: Some(piece),
                ..
            }) => content.push_str(&piece),
            Ok(_) => {}
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    assert_eq!(content, "partial", "the first chunk reached the caller");
    let failure = failure.expect("the mid-stream failure surfaces");
    assert_eq!(failure.kind(), ErrorKind::Llm);
    assert_eq!(request_count(&first).await, 1);
    assert_eq!(
        request_count(&second).await,
        0,
        "the router must not splice a second answer onto partial output"
    );
}
